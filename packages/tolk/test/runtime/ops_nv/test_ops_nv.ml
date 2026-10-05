(* Tests of Tolk.Ops_nv: the batches of NV devices, their command words and host
   programs, are tinygrad's, as the generator changes tinygrad where tolk
   departs from it. *)

open Windtrap
open Tolk

let plain f = Helpers.context [ B (Helpers.no_color, true) ] f

let is_batch c =
  match Ops.arg (Ops.without_after c) with
  | Call { aux = Some _; _ } -> true
  | _ -> false

(* Recorded cases *)

let host_target =
  {
    Helpers.Target.device = "CPU";
    renderer = "";
    arch = "x86_64,x86-64";
    interface = "";
    indices = "";
  }

(* The GPU the generator describes: an Ada GPU, or a Blackwell one. *)
let props ~blackwell =
  {
    Ops_nv.compute_class = (if blackwell then 0xcdc0 else 0xc9c0);
    sass_version = 0x89;
    shared_window = 0x729400000000;
    local_window = 0x729300000000;
    compute = { entries = 0x10000; token = 0x11 };
    copy = { entries = 0x10000; token = 0x22 };
  }

(* The host, and NV, which reaches the host's memory. *)
let recorded_devices ~blackwell = function
  | "CPU" -> { Hcq2.target = host_target; queues = None }
  | _ ->
      {
        Hcq2.target =
          { host_target with device = "NV"; renderer = "CUDA"; arch = "sm_89" };
        queues =
          Some
            (Ops_nv.queues ~host:"CPU"
               ~reaches:(fun d -> d = "CPU")
               (props ~blackwell));
      }

(* A kernel's profile key is its program's BLAKE2 digest, where tinygrad's is a
   SHA-256: recorded graphs are compared without them. *)
let without_profile_keys u =
  let unkeyed c =
    match Ops.arg c with
    | Call ({ aux = Some info; _ } as ci) ->
        let kernels =
          List.map
            (fun (k : Ops.hcq_kernel) -> { k with profile_key = None })
            info.kernels
        in
        Some
          ( c,
            Ops.replace
              ~arg:(Call { ci with aux = Some { info with kernels } })
              c )
    | _ -> None
  in
  Ops.substitute u (List.filter_map unkeyed (Ops.toposort u))

let host_sources linear =
  String.concat ""
    (List.map
       (fun b ->
         match Ops.arg (Ops.nth (Ops.nth (Ops.without_after b) 0) 2) with
         | String src -> src
         | _ -> fail "a compiled host program holds its source")
       (List.filter is_batch (Ops.src linear)))

(* Each case, whether its GPU is a Blackwell one, and whether it profiles. *)
let recorded_cases =
  [
    ("chain", false, false);
    ("chain_blackwell", true, false);
    ("chain_profile", false, true);
    ("variable", false, false);
    ("copy_in", false, false);
    ("copy_out_profile", false, true);
    ("copy_large", false, false);
    ("host_split", false, false);
    ("simple_add", false, false);
    ("simple_add_blackwell", true, false);
    ("simple_add_chain", false, false);
    ("simple_add_chain_blackwell", true, false);
    ("simple_add_profile", false, true);
    ("simple_add_grid", false, false);
    ("crafted", false, false);
    ("crafted_blackwell", true, false);
  ]

let recorded =
  group "recorded cases"
    (List.map
       (fun (case, blackwell, profile) ->
         let compiled () =
           plain (fun () ->
               Hcq2.compile_linear ~profile
                 ~devices:(recorded_devices ~blackwell)
                 (Golden.sink (case ^ "_prepared.golden")))
         in
         group case
           [
             test (case ^ "_compiled.golden") (fun () ->
                 let golden = Golden.sink (case ^ "_compiled.golden") in
                 let same u = Graph.to_string (without_profile_keys u) in
                 equal text (same golden)
                   (same
                      (Uops.placeholders_like golden
                         (Uops.binaries_as_sources (compiled ())))));
             Golden.text (case ^ "_host.golden") (fun () ->
                 host_sources (compiled ()));
           ])
       recorded_cases)

(* Command words *)

(* The effect of the host program that submits one queue of NV's commands
   [cmds], followed by [pad] (default [0]) zero words. *)
let submission ?(pad = 0) queue cmds =
  let captured = ref None in
  let devices = function
    | "CPU" -> { Hcq2.target = host_target; queues = None }
    | _ ->
        let qs =
          Ops_nv.queues ~host:"CPU"
            ~reaches:(fun _ -> false)
            (props ~blackwell:false)
        in
        let commands q =
          let c = qs.commands q in
          {
            c with
            submit =
              (fun () ->
                ignore
                  (Hcq2.Queue.q q
                     [
                       Ops.v Binary ~arg:(Bytes (String.make (4 * pad) '\000'));
                     ]);
                captured := Some (c.submit ());
                raise Exit);
          }
        in
        {
          Hcq2.target = { host_target with device = "NV" };
          queues = Some { qs with commands };
        }
  in
  let lin = Ops.v Linear ~arg:(Queue { devices = [ "NV" ]; queue }) ~src:cmds in
  let info : Ops.hcq_info =
    {
      device = [ "NV" ];
      kernels = [];
      estimates = { ops = Int 0; lds = Int 0; mem = Int 0 };
      nargs = 0;
      table = -1;
      inputs = [];
      slots = [];
      written_bufs = [];
      writes = [];
      copies = [];
    }
  in
  let batch =
    Ops.call ~aux:info
      (Ops.sink
         ~kernel:(Ops.kernel_info ~name:"hcq_submit" ())
         [ Ops.custom_function "submit_nv" [ lin ] ])
      []
  in
  (try ignore (Hcq2.lower_call ~devices batch) with Exit -> ());
  match !captured with
  | Some g -> g
  | None -> fail "the queue was not submitted"

(* The command buffer of one queue of NV's commands [cmds], as the queue encodes
   it: its bytes, with the words its patches write, each address of the signal
   word [signal] and of the sink word [sink]. *)
let signal = 0x1000
let sink = 0x2000

let command_buffer queue cmds =
  let g = submission queue cmds in
  let address u =
    let rec storage u =
      match Ops.src u with s :: _ when Ops.op u <> Param -> storage s | _ -> u
    in
    match Ops.tag (storage (Ops.nth u 0)) with
    | Some (String "nv_sink") -> sink
    | _ -> signal
  in
  let value w =
    let addrs =
      List.filter_map
        (fun u ->
          if Ops.op u = Getaddr then Some (u, Ops.int ~dtype:Uint64 (address u))
          else None)
        (Ops.toposort w)
    in
    match Interpreter.eval (Ops.substitute w addrs) with
    | `Int z ->
        Bigint.to_int (Bigint.extract z 0 (8 * Dtype.itemsize (Ops.dtype w)))
    | _ -> fail "a command word is no integer"
  in
  (* The stores into the command buffer, not into the regions' buffers. *)
  let stores =
    let rec storage u =
      match Ops.op u with
      | Index | Bitcast | Shrink | After -> storage (Ops.nth u 0)
      | _ -> u
    in
    List.filter
      (fun st ->
        Ops.op st = Store
        &&
        match Ops.tag (storage (Ops.nth st 0)) with
        | Some (String t) -> String.starts_with ~prefix:"cmdbuf" t
        | _ -> false)
      (Ops.toposort g)
  in
  let bytes = ref Bytes.empty in
  List.iter
    (fun st ->
      match Ops.src st with
      | [ _; v ] -> (
          let rec binary u =
            match (Ops.op u, Ops.arg u) with
            | Binary, Bytes b -> Some b
            | Bitcast, _ -> binary (Ops.nth u 0)
            | _ -> None
          in
          match binary v with
          | Some b -> bytes := Bytes.of_string b
          | None -> ())
      | _ -> ())
    stores;
  List.iter
    (fun st ->
      match Ops.src st with
      | [ dst; words ] when Ops.op dst = Index && Ops.op words = Stack ->
          let view = Ops.nth dst 0 and offs = Ops.src (Ops.nth dst 1) in
          let size = Dtype.itemsize (Ops.dtype view) in
          let phase =
            let under = Ops.nth view 0 in
            if Ops.op under <> Shrink then 0
            else match Ops.marg under with Shrink [ (Int p, _) ] -> p | _ -> 0
          in
          List.iter2
            (fun o w ->
              let at = phase + (value o * size) and x = value w in
              for k = 0 to size - 1 do
                Bytes.set !bytes (at + k) (Char.chr ((x lsr (8 * k)) land 0xff))
              done)
            offs (Ops.src words)
      | _ -> ())
    stores;
  List.init
    (Bytes.length !bytes / 4)
    (fun i -> Int32.to_int (Bytes.get_int32_le !bytes (4 * i)) land 0xffff_ffff)

(* What a channel does with command words: waits, and writes of 32 or 64 bits,
   in order. *)
type action =
  | Wait of int * int
  | Write32 of int * int
  | Write64 of int * int
  | Stamp of int

let pp_action ppf = function
  | Wait (a, v) -> Format.fprintf ppf "wait 0x%x >= 0x%x" a v
  | Write32 (a, v) -> Format.fprintf ppf "write32 0x%x 0x%x" a v
  | Write64 (a, v) -> Format.fprintf ppf "write64 0x%x 0x%x" a v
  | Stamp a -> Format.fprintf ppf "stamp 0x%x" a

let action = Testable.make ~pp:pp_action ~equal:( = )

(* The methods and fields the tests decode, from NVIDIA's class headers clc56f.h
   (host), clc6c0.h (compute) and clc6b5.h (copy engine). Fields are (lowest
   bit, bits). *)
module G = struct
  let nvc56f_sem_addr_lo = 0x5c
  let nvc56f_sem_execute = 0x6c
  let nvc56f_sem_execute_operation = (0, 3)
  let nvc56f_sem_execute_operation_acq_circ_geq = 3
  let nvc56f_sem_execute_release_timestamp = (25, 1)
  let nvc6c0_send_pcas_a = 0x2b4
  let nvc6b5_set_semaphore_a = 0x240
  let nvc6b5_launch_dma = 0x300
  let nvc6b5_launch_dma_semaphore_type = (3, 2)
  let nvc6b5_launch_dma_semaphore_type_release_one_word_semaphore = 1
  let nvc6b5_launch_dma_semaphore_type_release_four_word_semaphore = 2
end

(* The methods of the host (subchannel 0) and the copy engine (subchannel 4)
   that semaphores use. *)
let decode words =
  let state = Hashtbl.create 8 in
  let get m = Option.value ~default:0 (Hashtbl.find_opt state m) in
  let field (lo, bits) v = (v lsr lo) land ((1 lsl bits) - 1) in
  let actions = ref [] in
  let meth subc m v =
    Hashtbl.replace state (subc, m) v;
    if subc = 0 && m = G.nvc56f_sem_execute then begin
      let a =
        get (0, G.nvc56f_sem_addr_lo)
        lor (get (0, G.nvc56f_sem_addr_lo + 4) lsl 32)
      and p =
        get (0, G.nvc56f_sem_addr_lo + 8)
        lor (get (0, G.nvc56f_sem_addr_lo + 12) lsl 32)
      in
      let op = field G.nvc56f_sem_execute_operation v in
      if op = G.nvc56f_sem_execute_operation_acq_circ_geq then
        actions := Wait (a, p) :: !actions
      else if field G.nvc56f_sem_execute_release_timestamp v = 1 then
        actions := Stamp a :: !actions
      else actions := Write64 (a, p) :: !actions
    end
    else if subc = 4 && m = G.nvc6b5_launch_dma then begin
      let a =
        (get (4, G.nvc6b5_set_semaphore_a) lsl 32)
        lor get (4, G.nvc6b5_set_semaphore_a + 4)
      and p = get (4, G.nvc6b5_set_semaphore_a + 8) in
      let typ = field G.nvc6b5_launch_dma_semaphore_type v in
      if typ = G.nvc6b5_launch_dma_semaphore_type_release_one_word_semaphore
      then actions := Write32 (a, p) :: !actions
      else if
        typ = G.nvc6b5_launch_dma_semaphore_type_release_four_word_semaphore
      then actions := Stamp a :: !actions
    end
  in
  let rec go = function
    | [] -> ()
    | h :: rest ->
        let n = (h lsr 16) land 0x1fff
        and subc = (h lsr 13) land 7
        and m = (h land 0x1fff) lsl 2 in
        let vals = List.filteri (fun i _ -> i < n) rest in
        List.iteri (fun i v -> meth subc (m + (4 * i)) v) vals;
        go (List.filteri (fun i _ -> i >= n) rest)
  in
  go words;
  List.rev !actions

let ins code src = Ops.v Ins ~src ~arg:(Code { code; dtype = Void })
let u64 v = Ops.int ~dtype:Uint64 v

let signalled queue v =
  decode (command_buffer queue [ ins "store" [ Hcq2.signal_word "NV"; u64 v ] ])

let values =
  [ (1 lsl 32) - 1; 1 lsl 32; (1 lsl 32) + 1; (1 lsl 33) - 1; 1 lsl 33 ]

(* The methods of subchannel [subc] (default [1], the compute engine's) in the
   command words [words], in order. *)
let methods ?(subc = 1) words =
  let rec go = function
    | [] -> []
    | h :: rest ->
        let n = (h lsr 16) land 0x1fff and s = (h lsr 13) land 7 in
        let rest = List.filteri (fun i _ -> i >= n) rest in
        if s = subc then ((h land 0x1fff) lsl 2) :: go rest else go rest
  in
  go words

let words =
  let hex v = Printf.sprintf "0x%x" v in
  group "command words"
    [
      cases
        ~name:(fun (n, _) -> string_of_int n)
        "a copy goes in lines of at most 2 GiB"
        [
          (1, 1);
          ((1 lsl 31) - 1, 1);
          (1 lsl 31, 1);
          ((1 lsl 31) + 1, 2);
          (1 lsl 32, 2);
        ]
        (fun (n, lines) ->
          let buf () = Ops.new_buffer (Single "NV") n Uint8 in
          equal int lines
            (List.length
               (List.filter
                  (( = ) G.nvc6b5_launch_dma)
                  (methods ~subc:4
                     (command_buffer "COPY:0"
                        [ Ops.store_call (buf ()) (buf ()) ])))));
      cases ~name:hex "the compute channel releases a value whole" values
        (fun v ->
          equal (list action) [ Write64 (signal, v) ] (signalled "COMPUTE:0" v));
      cases ~name:hex
        "the copy engine releases the low word, then the high word where the \
         value lives"
        values (fun v ->
          let lo = v land 0xffff_ffff and hi = v lsr 32 in
          equal (list action)
            [
              Write32 (signal, lo);
              Write32 ((if lo = 0 then signal + 4 else sink), hi);
            ]
            (signalled "COPY:0" v));
      cases ~name:hex "a queue waits for a value whole" values (fun v ->
          equal (list action)
            [ Wait (signal, v) ]
            (decode
               (command_buffer "COPY:0"
                  [ ins "wait" [ Hcq2.signal_word "NV"; u64 v ] ])));
    ]

(* Chains *)

(* A launch of the fixture's [simple_add] of 32 integers, [out = a + b]. *)
let simple_add out a b =
  let call =
    List.find
      (fun u -> Ops.op u = Call)
      (Ops.toposort (Golden.sink "simple_add_chain_prepared.golden"))
  in
  match Ops.src call with
  | [ prg; _; _; _; n ] -> Ops.replace ~src:[ prg; out; a; b; n ] call
  | _ -> fail "simple_add takes three buffers and a size"

(* The launches the channel schedules among the commands [cmds] of the compute
   queue: those that chain onto no launch before them. *)
let scheduled cmds =
  let buf () = Ops.new_buffer (Single "NV") 32 Int32 in
  let launch = function
    | `Launch -> simple_add (buf ()) (buf ()) (buf ())
    | `Wait -> ins "wait" [ Hcq2.signal_word "NV"; u64 1 ]
    | `Signal -> ins "store" [ Hcq2.signal_word "NV"; u64 1 ]
    | `Barrier -> ins "barrier" []
  in
  List.length
    (List.filter
       (( = ) G.nvc6c0_send_pcas_a)
       (methods (command_buffer "COMPUTE:0" (List.map launch cmds))))

let chains =
  group "chains"
    [
      test "a launch after a launch chains onto its descriptor" (fun () ->
          equal int 1 (scheduled [ `Launch; `Launch; `Launch ]));
      test "a wait between two launches ends their chain" (fun () ->
          equal int 2 (scheduled [ `Launch; `Wait; `Launch ]));
      test "a barrier between two launches ends their chain" (fun () ->
          equal int 2 (scheduled [ `Launch; `Barrier; `Launch ]));
      test "a signal the descriptor releases keeps the chain" (fun () ->
          equal int 1 (scheduled [ `Launch; `Signal; `Signal; `Launch ]));
      test "a signal past the descriptor's two releases ends the chain"
        (fun () ->
          equal int 2
            (scheduled [ `Launch; `Signal; `Signal; `Signal; `Launch ]));
      test "a signal before any launch is the channel's" (fun () ->
          equal int 1 (scheduled [ `Signal; `Launch ]));
    ]

(* Refusals *)

(* A launch of [simple_add], or of the crafted cubin's kernel, whose program is
   [f] of the fixture's. *)
let launch_of ?(crafted = false) f =
  let buf () = Ops.new_buffer (Single "NV") 32 Int32 in
  let call =
    if crafted then
      List.find
        (fun u -> Ops.op u = Call)
        (Ops.toposort (Golden.sink "crafted_prepared.golden"))
    else simple_add (buf ()) (buf ()) (buf ())
  in
  match Ops.src call with
  | prg :: args -> Ops.replace ~src:(f prg :: args) call
  | [] -> fail "a call has a body"

let sized ?global ?local prg =
  match Ops.arg prg with
  | Program p ->
      let or_ o d = Option.value o ~default:d in
      Ops.replace
        ~arg:
          (Program
             {
               p with
               global_size = or_ global p.global_size;
               local_size = or_ local p.local_size;
             })
        prg
  | _ -> fail "a launch runs a program"

let refused ~why queue cmds =
  raises_match (Exn.invalid_arg ~substring:why) (fun () ->
      command_buffer queue cmds)

let accepted queue cmds = ignore (command_buffer queue cmds)

let refusals =
  group "refusals"
    [
      test "a block of 1024 threads is accepted" (fun () ->
          accepted "COMPUTE:0"
            [ launch_of (sized ~local:[ Int 1024; Int 1; Int 1 ]) ]);
      test
        "a block of as many threads as the registers allow is accepted, and \
         one more is refused" (fun () ->
          (* 128 registers a thread leave 512 threads a block. *)
          accepted "COMPUTE:0"
            [ launch_of ~crafted:true (sized ~local:[ Int 512; Int 1; Int 1 ]) ];
          refused ~why:"Too many resources" "COMPUTE:0"
            [ launch_of ~crafted:true (sized ~local:[ Int 513; Int 1; Int 1 ]) ]);
      test "a block of more than 1024 threads is refused" (fun () ->
          refused ~why:"Too many resources" "COMPUTE:0"
            [ launch_of (sized ~local:[ Int 2048; Int 1; Int 1 ]) ]);
      test "a grid of more than 65535 blocks on its second axis is refused"
        (fun () ->
          refused ~why:"Invalid global/local dims" "COMPUTE:0"
            [ launch_of (sized ~global:[ Int 1; Int 65536; Int 1 ]) ]);
      test "a block of more than 64 threads on its third axis is refused"
        (fun () ->
          refused ~why:"Invalid global/local dims" "COMPUTE:0"
            [ launch_of (sized ~local:[ Int 1; Int 1; Int 65 ]) ]);
      test "a program that is no cubin is refused" (fun () ->
          refused ~why:"ELF" "COMPUTE:0"
            [
              launch_of (fun prg ->
                  Ops.replace
                    ~src:
                      (List.filteri (fun i _ -> i < 3) (Ops.src prg)
                      @ [ Ops.v Binary ~arg:(Bytes "\x7fELG not a cubin") ])
                    prg);
            ]);
      test "a program whose cubin lacks its kernel is refused" (fun () ->
          let crafted = launch_of ~crafted:true Fun.id in
          let cubin = Ops.nth (Ops.nth crafted 0) 3 in
          refused ~why:"no kernel simple_add, only k" "COMPUTE:0"
            [
              launch_of (fun prg ->
                  Ops.replace
                    ~src:
                      (List.filteri (fun i _ -> i < 3) (Ops.src prg) @ [ cubin ])
                    prg);
            ]);
      test
        "a command buffer of 2^21 words is over capacity, one word fewer is not"
        (fun () ->
          (* An entry's length field holds 21 bits of words. *)
          let words n = submission ~pad:n "COMPUTE:0" [] in
          ignore (words ((1 lsl 21) - 1));
          match words (1 lsl 21) with
          | _ -> fail "a command buffer over capacity was submitted"
          | exception Hcq2.Over_capacity _ -> ());
      test "a copy queue runs no program" (fun () ->
          refused ~why:"runs no program" "COPY:0" [ launch_of Fun.id ]);
      test "a compute queue copies nothing" (fun () ->
          let buf () = Ops.new_buffer (Single "NV") 32 Int32 in
          refused ~why:"copies nothing" "COMPUTE:0"
            [ Ops.store_call (buf ()) (buf ()) ]);
    ]

(* Storage *)

let pp_storage ppf = function
  | Ops_nv.Program { name; _ } -> Format.fprintf ppf "Program %s" name
  | Ring q -> Format.fprintf ppf "Ring %s" q
  | Gp_put q -> Format.fprintf ppf "Gp_put %s" q
  | Put q -> Format.fprintf ppf "Put %s" q
  | Doorbell q -> Format.fprintf ppf "Doorbell %s" q
  | Local n -> Format.fprintf ppf "Local %d" n

let storage = Testable.make ~pp:pp_storage ~equal:( = )

(* The storage of the placeholders of the submission of [cmds] on [queue] that
   NV's commands name. *)
let named queue cmds =
  List.sort_uniq compare
    (List.filter_map
       (fun u -> if Ops.op u = Param then Ops_nv.storage u else None)
       (Ops.toposort (submission queue cmds)))

let storages =
  let buf () = Ops.new_buffer (Single "NV") 32 Int32 in
  group "storage"
    [
      test
        "a compute queue names its program's cubin, its channel's words, and \
         the local memory its launches need" (fun () ->
          let call = simple_add (buf ()) (buf ()) (buf ()) in
          let binary =
            match Ops.arg (Ops.nth (Ops.nth call 0) 3) with
            | Bytes b -> b
            | _ -> fail "a program holds its binary"
          in
          let local, words =
            List.partition
              (function Ops_nv.Local _ -> true | _ -> false)
              (named "COMPUTE:0" [ call ])
          in
          equal (list storage)
            [
              Program { binary; name = "simple_add" };
              Ring "COMPUTE:0";
              Gp_put "COMPUTE:0";
              Put "COMPUTE:0";
              Doorbell "COMPUTE:0";
            ]
            words;
          equal int 1 (List.length local));
      test
        "the launches of two programs in one batch read one local memory word, \
         whatever the engine binds to it" (fun () ->
          let call = simple_add (buf ()) (buf ()) (buf ()) in
          let prg = Ops.nth call 0 in
          let binary = Ops.nth prg 3 in
          (* Another program: the same image, a word longer. *)
          let other =
            match Ops.arg binary with
            | Bytes b ->
                Ops.replace call
                  ~src:
                    (Ops.replace prg
                       ~src:
                         (List.mapi
                            (fun i s ->
                              if i = 3 then
                                Ops.replace binary
                                  ~arg:(Bytes (b ^ "\000\000\000\000"))
                              else s)
                            (Ops.src prg))
                    :: List.tl (Ops.src call))
            | _ -> fail "a program holds its binary"
          in
          let compiled =
            plain (fun () ->
                Hcq2.compile_linear
                  ~devices:(recorded_devices ~blackwell:false)
                  (Ops.v Linear ~src:[ call; other ]))
          in
          let tagged name =
            List.filter
              (fun u ->
                match Ops.tag u with
                | Some (Tuple (String n :: _)) -> n = name
                | _ -> false)
              (Ops.toposort ~enter_calls:true compiled)
          in
          equal ~msg:"programs" int 2 (List.length (tagged "program"));
          let words = tagged "nv_local" in
          is_true ~msg:"a local memory word" (words <> []);
          List.iter (fun u -> equal int 1 (Ops.max_numel u)) words);
      test "a copy queue names its channel's words" (fun () ->
          equal (list storage)
            [ Ring "COPY:0"; Gp_put "COPY:0"; Put "COPY:0"; Doorbell "COPY:0" ]
            (named "COPY:0" [ Ops.store_call (buf ()) (buf ()) ]));
      test "a placeholder NV's commands do not name has no storage" (fun () ->
          let cmdbuf =
            Ops.placeholder ~device:(Single "NV")
              ~tag:(String "cmdbuf_compute_0") [ 4 ] Uint8
          in
          equal (option storage) None (Ops_nv.storage cmdbuf));
    ]

(* The carry law *)

(* Works on the channels that share a device's signal word, in the order of
   their values: work [i] signals [v + i] with the [writes] of its channel, once
   the word reads at least [v + i - 1], each channel running its works in order.
   Every order in which the channels' writes can land is explored: in each, the
   word never reads above the greatest value whose writes to it have all landed,
   so a waiter never passes early, and, once no work is mid-write, never below a
   value whose writes have landed. A value written as two words dips below its
   predecessor between them; no waiter passes early then. *)
type channel = { name : string; writes : int -> action list }

let compute = { name = "compute"; writes = signalled "COMPUTE:0" }
let copy = { name = "copy"; writes = signalled "COPY:0" }

(* nx.nv.device's own copies: the low word, then the high word when the low one
   is 0. *)
let nx_copy =
  {
    name = "nx copy";
    writes =
      (fun v ->
        let lo = v land 0xffff_ffff in
        Write32 (signal, lo)
        :: (if lo = 0 then [ Write32 (signal + 4, v lsr 32) ] else []));
  }

(* tinygrad's copy queue, which releases the low word alone. *)
let tinygrad_copy =
  {
    name = "tinygrad copy";
    writes = (fun v -> [ Write32 (signal, v land 0xffff_ffff) ]);
  }

(* A copy queue that rewrites the high word on every release. *)
let high_always =
  {
    name = "high always";
    writes =
      (fun v ->
        [ Write32 (signal, v land 0xffff_ffff); Write32 (signal + 4, v lsr 32) ]);
  }

(* The first state the law fails in, if any. *)
let carry_violation ~start channels =
  let works =
    List.mapi
      (fun i c -> (c, start + 1 + i, Array.of_list (c.writes (start + 1 + i))))
      channels
  in
  let n = List.length works in
  let targets_word a = a = signal || a = signal + 4 in
  let word (lo, hi) = lo lor (hi lsl 32) in
  let write (lo, hi) = function
    | Write32 (a, x) when a = signal -> (x, hi)
    | Write32 (a, x) when a = signal + 4 -> (lo, x)
    | Write64 (a, x) when a = signal -> (x land 0xffff_ffff, x lsr 32)
    | _ -> (lo, hi)
  in
  let complete pcs i =
    let _, _, ws = List.nth works i in
    let rec landed k =
      k >= Array.length ws
      || (pcs.(i) > k
         ||
         match ws.(k) with
         | Write32 (a, _) | Write64 (a, _) -> not (targets_word a)
         | _ -> true)
         && landed (k + 1)
    in
    (landed 0 && pcs.(i) > 0) || Array.length ws = 0
  in
  let rec explore mem pcs =
    let w = word mem in
    let done_ = List.filteri (fun i _ -> complete pcs i) works in
    let top = List.fold_left (fun m (_, v, _) -> max m v) start done_ in
    (* A work whose first write to the word landed and its last did not is
       mid-write: the word may dip then, below its old value. *)
    let mid_write =
      List.exists
        (fun i -> pcs.(i) > 0 && not (complete pcs i))
        (List.init n Fun.id)
    in
    if w > top then Some (Printf.sprintf "0x%x read above 0x%x" w top)
    else if (not mid_write) && List.exists (fun (_, v, _) -> w < v) done_ then
      Some (Printf.sprintf "0x%x read below a landed value" w)
    else
      List.find_map
        (fun i ->
          let c, v, ws = List.nth works i in
          let earlier_on_channel =
            List.filteri (fun j (c', _, _) -> j < i && c'.name = c.name) works
          in
          let channel_free =
            List.for_all
              (fun (_, v', ws') ->
                let j = v' - start - 1 in
                pcs.(j) >= Array.length ws')
              earlier_on_channel
          in
          if
            pcs.(i) >= Array.length ws
            || (not channel_free)
            || (pcs.(i) = 0 && w < v - 1)
          then None
          else
            let pcs' = Array.copy pcs in
            pcs'.(i) <- pcs.(i) + 1;
            explore (write mem ws.(pcs.(i))) pcs')
        (List.init n Fun.id)
  in
  explore (start land 0xffff_ffff, start lsr 32) (Array.make n 0)

let rec assignments k chans =
  if k = 0 then [ [] ]
  else
    List.concat_map
      (fun rest -> List.map (fun c -> c :: rest) chans)
      (assignments (k - 1) chans)

let carry =
  let starts =
    [ (1 lsl 32) - 3; (1 lsl 32) - 2; (1 lsl 32) - 1; (1 lsl 33) - 2 ]
  in
  let name (start, chans) =
    Printf.sprintf "from 0x%x: %s" start
      (String.concat ", " (List.map (fun c -> c.name) chans))
  in
  let runs chans =
    List.concat_map
      (fun start -> List.map (fun a -> (start, a)) (assignments 3 chans))
      starts
  in
  group "the 32-bit carry law"
    [
      cases ~name "no waiter passes early and no landed value is taken back"
        (runs [ compute; copy; nx_copy ])
        (fun (start, chans) ->
          equal (option string) None (carry_violation ~start chans));
      test "tinygrad's copy queue takes a value back across 2^32" (fun () ->
          is_true
            (Option.is_some
               (carry_violation
                  ~start:((1 lsl 32) - 1)
                  [ tinygrad_copy; compute; compute ])));
      test "a high word rewritten late takes a value back across 2^32"
        (fun () ->
          is_true
            (Option.is_some
               (carry_violation
                  ~start:((1 lsl 32) - 3)
                  [ compute; high_always; compute ])));
    ]

(* Loops *)

(* A range of three trips around two launches of the fixture's [simple_add],
   each trip on its own windows of 32 integers: [t = a + b], then [out = t +
   b]. *)
let ranged () =
  let r = Ops.range (Int 3) [ 7 ] in
  let window u =
    let start = Ops.mul r (Ops.int 32) in
    Ops.shrink u [ Some (Sym start, Sym (Ops.add start (Ops.int 32))) ]
  in
  let buf () = Ops.new_buffer (Single "NV") 96 Int32 in
  let a = buf () and b = buf () and t = buf () and out = buf () in
  let launch dst x y = simple_add (window dst) (window x) (window y) in
  Ops.end_ (Ops.v Linear ~src:[ launch t a b; launch out t b ]) [ r ]

(* The compute methods of the command buffers [submitted] holds, in order: the
   buffers' bytes, whose patched words are 0. *)
let compute_methods submitted =
  let blobs =
    List.filter_map
      (fun u ->
        match Ops.src u with
        | [ dst; v ] when Ops.op u = Store -> (
            let rec binary u =
              match (Ops.op u, Ops.arg u) with
              | Binary, Bytes b -> Some b
              | Bitcast, _ -> binary (Ops.nth u 0)
              | _ -> None
            in
            match (binary v, Ops.tag (Ops.without_after dst)) with
            | Some b, Some (String name)
              when String.starts_with ~prefix:"cmdbuf" name ->
                Some b
            | _ -> None)
        | _ -> None)
      (Ops.toposort submitted)
  in
  List.concat_map
    (fun b ->
      methods
        (List.init
           (String.length b / 4)
           (fun i ->
             Int32.to_int (String.get_int32_le b (4 * i)) land 0xffff_ffff)))
    blobs

let loops =
  group "loops"
    [
      test
        "each trip's two launches chain on descriptors of the trip's own, and \
         the channel schedules each trip's chain" (fun () ->
          let submitted = ref [] in
          let devices d =
            let dev = recorded_devices ~blackwell:false d in
            match dev.queues with
            | None -> dev
            | Some qs ->
                let commands q =
                  let c = qs.commands q in
                  {
                    c with
                    submit =
                      (fun () ->
                        let e = c.submit () in
                        submitted := e :: !submitted;
                        e);
                  }
                in
                { dev with queues = Some { qs with commands } }
          in
          ignore
            (plain (fun () ->
                 Hcq2.compile_linear ~devices (Ops.v Linear ~src:[ ranged () ])));
          let submitted = Ops.sink !submitted in
          equal int 3
            (List.length
               (List.filter
                  (( = ) G.nvc6c0_send_pcas_a)
                  (compute_methods submitted)));
          (* Two descriptors of 256 bytes a trip. *)
          equal (list int)
            [ 3 * 2 * 256 ]
            (List.sort_uniq Int.compare
               (List.filter_map
                  (fun u ->
                    if Ops.tag u = Some (String "qmd_compute_0") then
                      Some (Ops.max_numel u)
                    else None)
                  (Ops.toposort submitted))));
    ]

let () =
  exit
    (run "Tolk.Ops_nv"
       [ recorded; words; chains; refusals; storages; carry; loops ])
