open Windtrap
open Tolk

let plain f = Setting.context [ B (Setting.no_color, true) ] f

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

let ring = 16 lsl 20

(* The GPUs of the generator's cases, as tinygrad describes them. *)
let rec gpu = function
  | ("gfx1100" | "gfx1100_sdma5" | "gfx1100_sdma52" | "gfx1100_no_sdma") as name
    ->
      {
        Ops_amd.target = (11, 0, 0);
        gc = (11, 0, 0);
        sdma =
          (match name with
          | "gfx1100_sdma5" -> (5, 0, 0)
          | "gfx1100_sdma52" -> (5, 2, 0)
          | _ -> (6, 0, 0));
        xccs = 1;
        shader_engines = 6;
        compute_units = 48;
        scratch_slots_per_cu = 32;
        aql = false;
        compute_ring = ring;
        copy_rings =
          (match name with
          | "gfx1100" -> [ ring; ring ]
          | "gfx1100_sdma5" | "gfx1100_sdma52" -> [ ring ]
          | _ -> []);
        profiling = None;
      }
  | "gfx1201" ->
      {
        target = (12, 0, 1);
        gc = (12, 0, 1);
        sdma = (7, 0, 0);
        xccs = 1;
        shader_engines = 4;
        compute_units = 32;
        scratch_slots_per_cu = 32;
        aql = false;
        compute_ring = ring;
        copy_rings = [ ring ];
        profiling = None;
      }
  | ("gfx942" | "gfx942_cpx") as name ->
      let cpx = name = "gfx942_cpx" in
      {
        target = (9, 4, 2);
        gc = (9, 4, 3);
        sdma = (4, 4, 2);
        xccs = (if cpx then 1 else 8);
        shader_engines = 4;
        compute_units = 38;
        scratch_slots_per_cu = 32;
        aql = not cpx;
        compute_ring = ring;
        copy_rings = [ ring ];
        profiling = None;
      }
  | name -> (
      (* A GPU that counts, traces, or both, as [gpu_counters], [gpu_traces] and
         [gpu_counters_traces] name it, with tinygrad's 32 runs and 256 MiB of
         each shader engine's traces. *)
      match String.split_on_char '_' name with
      | g :: (_ :: _ as what) ->
          let g = gpu g in
          let counts = List.mem "counters" what
          and traces = List.mem "traces" what in
          {
            g with
            profiling =
              Some
                {
                  Ops_amd.slots = 32;
                  counting = (if counts then Some (counting g) else None);
                  tracing =
                    (if traces then
                       Some
                         {
                           Ops_amd.window = (256 lsl 20) / 32;
                           engines = g.shader_engines * g.xccs;
                         }
                     else None);
                };
          }
      | _ -> fail ("no GPU " ^ name))

(* The counting of tinygrad's default counters on [g], laid out as tinygrad's
   pmc_start lays them out, with 4 compute units to a shader array and the
   work-group processor 2 of the shader engine 1 inactive, as the generator's
   cases describe it: each block's counters take its registers in turn, and an
   SQ counter is counted in each engine, and from GFX11 in each array and
   work-group processor. *)
and counting (g : Ops_amd.gpu) =
  let major, _, _ = g.target in
  let defaults =
    match major with
    | 9 ->
        [
          ("SQ", 3);
          ("SQ", 26);
          ("SQ", 60);
          ("SQ", 131);
          ("SQ", 126);
          ("GRBM", 2);
          ("TCC", 17);
          ("TCC", 19);
        ]
    | 11 ->
        [
          ("SQ", 3);
          ("SQ", 62);
          ("SQ", 58);
          ("SQ", 261);
          ("SQ", 256);
          ("GRBM", 2);
          ("GL2C", 42);
          ("GL2C", 43);
        ]
    | _ ->
        [
          ("SQ", 3);
          ("SQ", 50);
          ("SQ", 46);
          ("SQ", 293);
          ("SQ", 288);
          ("GRBM", 2);
          ("GL2C", 41);
          ("GL2C", 42);
        ]
  in
  let registers = Hashtbl.create 4 and offset = ref 0 in
  let counter (block, event) =
    let register = Option.value ~default:0 (Hashtbl.find_opt registers block) in
    Hashtbl.replace registers block (register + 1);
    let instances, engines, arrays, wgps =
      match block with
      | "GRBM" -> (1, 1, 1, 1)
      | "GL2C" -> (32, 1, 1, 1)
      | "TCC" -> (16, 1, 1, 1)
      | _ when major = 9 -> (1, g.shader_engines, 1, 1)
      | _ -> (1, g.shader_engines, 2, 2)
    in
    let at = !offset in
    offset := at + (g.xccs * instances * engines * arrays * wgps * 8);
    {
      Ops_amd.block;
      event;
      register;
      instances;
      engines;
      arrays;
      wgps;
      offset = at;
    }
  in
  let counters = List.map counter defaults in
  {
    Ops_amd.counters;
    size = !offset;
    wgp_active = (fun ~engine ~array:_ ~wgp -> (engine, wgp) <> (1, 2));
  }

(* The host, and AMD, whose queues address the host's memory. *)
let gpu_devices (g : Ops_amd.gpu) = function
  | "CPU" -> { Hcq2.target = host_target; queues = None }
  | _ ->
      let m, n, s = g.target in
      {
        Hcq2.target =
          {
            host_target with
            device = "AMD";
            arch = Printf.sprintf "gfx%d%x%x" m n s;
          };
        queues =
          Some (Ops_amd.queues ~host:"CPU" ~reaches:(String.equal "CPU") g);
      }

let recorded_devices name = gpu_devices (gpu name)

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
  Ops.substitute ~calls:Skip ~pass:Fixed_point u
    (List.filter_map unkeyed (Ops.toposort ~calls:Enter u))

let host_sources linear =
  String.concat ""
    (List.map
       (fun b ->
         match Ops.arg (Ops.nth (Ops.nth (Ops.without_after b) 0) 2) with
         | String src -> src
         | _ -> fail "a compiled host program holds its source")
       (List.filter is_batch (Ops.src linear)))

(* Each case with its GPU and profile. *)
let cases =
  [
    ("chain", "gfx1100", Hcq2.Unstamped);
    ("chain_gfx1201", "gfx1201", Hcq2.Unstamped);
    ("chain_gfx942", "gfx942", Hcq2.Unstamped);
    ("chain_gfx942_cpx", "gfx942_cpx", Hcq2.Unstamped);
    ("profile", "gfx1100", Hcq2.Stamped);
    ("profile_gfx942", "gfx942", Hcq2.Stamped);
    ("copies", "gfx1100", Hcq2.Unstamped);
    ("copies_gfx942", "gfx942", Hcq2.Unstamped);
    ("copies_no_sdma", "gfx1100_no_sdma", Hcq2.Unstamped);
    ("copies_profile", "gfx1100", Hcq2.Stamped);
    ("large_copy", "gfx1100_sdma5", Hcq2.Unstamped);
    ("large_copy_sdma52", "gfx1100_sdma52", Hcq2.Unstamped);
    ("large_copy_gfx942", "gfx942", Hcq2.Unstamped);
    ("lds", "gfx1100", Hcq2.Unstamped);
    ("variable", "gfx1100", Hcq2.Unstamped);
    ("variable_gfx942", "gfx942", Hcq2.Unstamped);
    ("scratch", "gfx1100", Hcq2.Unstamped);
    ("scratch_gfx942", "gfx942", Hcq2.Unstamped);
    ("counters", "gfx1100_counters", Hcq2.Unstamped);
    ("counters_gfx1201", "gfx1201_counters", Hcq2.Unstamped);
    ("counters_gfx942", "gfx942_counters", Hcq2.Unstamped);
    ("traces", "gfx1100_traces", Hcq2.Unstamped);
    ("traces_gfx1201", "gfx1201_traces", Hcq2.Unstamped);
    ("traces_gfx942", "gfx942_traces", Hcq2.Unstamped);
    ("counters_traces", "gfx1100_counters_traces", Hcq2.Unstamped);
  ]

let recorded =
  group "recorded cases"
    (List.map
       (fun (case, name, profile) ->
         (* Compiled in each test, so that a mutant armed in a test's process
            compiles the case again. *)
         let compiled () =
           plain (fun () ->
               Hcq2.compile_linear ~profile ~devices:(recorded_devices name)
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
       cases)

(* Signal words *)

let signal_word = 0x10000
and slot = 0x20000

(* The words of [command] on [queue] of a device [gpu], on the device's signal
   word with its batch's value [v], or on a queue's signal with [3]; the signal
   word is at [signal_word] and the queue's signal at [slot]. *)
let command_words ~gpu ~queue ~command ~word v =
  let device = "AMD" in
  let q = Hcq2.Queue.v ~devices:[ device ] queue in
  let cmds =
    (Ops_amd.queues ~host:"CPU" ~reaches:(fun _ -> false) gpu).commands q
  in
  let target, value =
    if word = "signal word" then (Hcq2.signal_word device, Hcq2.value device)
    else
      ( Ops.placeholder ~slot:0 ~device:(Multi [ device ]) ~volatile:true
          ~tag:(String "slots") [ 2 ] Dtype.Uint64,
        Ops.int ~dtype:Dtype.Uint64 3 )
  in
  (if command = "wait" then cmds.wait else cmds.signal) target value;
  let blob = Bytes.create (Hcq2.Queue.size q) in
  for i = 0 to (Hcq2.Queue.size q / 4) - 1 do
    Bytes.set_int32_le blob (4 * i)
      (Int32.of_int (Hcq2.Queue.get_dword q (4 * i)))
  done;
  let address g =
    let at = if word = "signal word" then signal_word else slot in
    (g, Ops.int ~dtype:Dtype.Uint64 at)
  in
  List.iter
    (fun (off, w) ->
      let getaddrs =
        List.filter (fun g -> Ops.op g = Getaddr) (Ops.toposort ~calls:Enter w)
      in
      let w =
        Ops.substitute ~calls:Skip ~pass:Fixed_point w
          (List.map address getaddrs)
      in
      let vars =
        [
          ( Option.get (Interpreter.name (Hcq2.value device)),
            `Int (Bigint.of_int v) );
        ]
      in
      match Interpreter.eval ~vars w with
      | `Int z ->
          let n = Dtype.itemsize (Ops.dtype w) in
          let off =
            match Ops.arg off with
            | Const (`Int z) -> Bigint.to_int z
            | _ -> fail "a word's offset is a constant"
          in
          for k = 0 to n - 1 do
            Bytes.set blob (off + k)
              (Char.chr (Bigint.to_int (Bigint.extract z (8 * k) 8)))
          done
      | _ -> fail "a word is an integer")
    (Hcq2.Queue.words q);
  List.init
    (Bytes.length blob / 4)
    (fun i -> Int32.to_int (Bytes.get_int32_le blob (4 * i)) land 0xffff_ffff)

let hex words = String.concat " " (List.map (Printf.sprintf "%08x") words)

let recorded_words =
  Golden.cases ~key:[ "gpu"; "queue"; "command"; "word"; "value" ]
    "signal_words.golden" (fun cell ->
      let words =
        command_words
          ~gpu:(gpu (cell "gpu"))
          ~queue:(cell "queue") ~command:(cell "command") ~word:(cell "word")
          (int_of_string (cell "value"))
      in
      equal text (cell "words") (hex words))

(* The carry law. A device's work signals its values in order, each work after
   waiting for the value before its own. Decoded into memory writes, the rest of
   the signal of a value, the signal of the next, and the wait and the signal of
   the one after interleave in every order their queues allow. A 64-bit value
   written as two halves, its low half first, reads below the previous value
   between them, which no wait can take for a later value. The law is that the
   word never reads above the latest value whose work is complete, so no wait
   passes early, and never goes below a value once its writes landed, so a high
   half that lands late takes nothing back; and that each wait passes only once
   the value it waits for is written. *)

type write = Low of int | High of int | Whole of int
type test = { eq : bool; value : int; mask : int }

let lo32 v = v land 0xffff_ffff

(* The writes to the signal word and the waits on it of a queue's words. *)
let rec decode ~sdma words =
  let at lo hi = lo lor (hi lsl 32) in
  let write a data =
    if a = signal_word then [ `Write (Low data) ]
    else if a = signal_word + 4 then [ `Write (High data) ]
    else []
  in
  match words with
  | [] -> []
  | h :: rest when sdma -> (
      match (h land 0xff, rest) with
      | 0, rest -> decode ~sdma rest
      | 5, alo :: ahi :: data :: rest ->
          write (at alo ahi) data @ decode ~sdma rest
      | 6, _ :: rest -> decode ~sdma rest
      | 8, _ :: _ :: value :: mask :: _ :: rest ->
          `Wait { eq = (h lsr 28) land 7 = 3; value; mask } :: decode ~sdma rest
      | op, _ -> fail (Printf.sprintf "an unknown SDMA packet %d" op))
  | h :: rest ->
      let n = ((h lsr 16) land 0x3fff) + 1 in
      let body = List.filteri (fun i _ -> i < n) rest
      and rest = List.filteri (fun i _ -> i >= n) rest in
      let packet =
        match ((h lsr 8) land 0xff, body) with
        | 0x49, [ _; memsel; alo; ahi; dlo; dhi; _ ] when memsel lsr 29 = 2 ->
            if at alo ahi = signal_word then [ `Write (Whole (at dlo dhi)) ]
            else []
        | 0x49, [ _; _; alo; ahi; dlo; _; _ ] -> write (at alo ahi) dlo
        | 0x3c, [ info; _; _; value; mask; _ ] ->
            [ `Wait { eq = info land 7 = 3; value; mask } ]
        | _ -> []
      in
      packet @ decode ~sdma rest

let signal_writes ~gpu queue v =
  List.filter_map
    (function `Write w -> Some w | `Wait _ -> None)
    (decode ~sdma:(queue = "COPY:0")
       (command_words ~gpu ~queue ~command:"signal" ~word:"signal word" v))

let wait_test ~gpu queue v =
  match
    decode ~sdma:(queue = "COPY:0")
      (command_words ~gpu ~queue ~command:"wait" ~word:"signal word" v)
  with
  | [ `Wait t ] -> t
  | _ -> fail "a wait is one test"

let apply word = function
  | Low x -> word land lnot 0xffff_ffff lor x
  | High x -> lo32 word lor (x lsl 32)
  | Whole x -> x

let passes t word =
  let low = lo32 word land t.mask and value = t.value land t.mask in
  if t.eq then low = value else low >= value

(* Every interleaving of the writes of [writers] and of [waiter]'s wait, then
   writes, from [word]: [f] sees each state with the writes applied so far. *)
let rec interleave word writers waiter f =
  f word writers waiter;
  List.iteri
    (fun i ws ->
      match ws with
      | w :: rest ->
          let writers =
            List.mapi (fun j ws -> if i = j then rest else ws) writers
          in
          interleave (apply word w) writers waiter f
      | [] -> ())
    writers;
  match waiter with
  | `Waiting (t, writes) when passes t word ->
      interleave word writers (`Writing writes) f
  | `Writing (w :: rest) -> interleave (apply word w) writers (`Writing rest) f
  | `Waiting _ | `Writing [] -> ()

let carry_law =
  let queues = [ "COMPUTE:0"; "COPY:0" ] in
  group "carry law"
    (List.concat_map
       (fun name ->
         let gpu = gpu name in
         List.concat_map
           (fun v ->
             List.concat_map
               (fun (prior, first) ->
                 List.map
                   (fun next ->
                     test
                       (Printf.sprintf
                          "%s: %d from %s after %s, then %d from %s" name v
                          first prior (v + 1) next)
                       (fun () ->
                         (* The value before: its first write landed, the rest
                            may land late. *)
                         let late, word =
                           match signal_writes ~gpu prior (v - 1) with
                           | w :: late -> (late, apply (v - 2) w)
                           | [] -> fail "a signal writes"
                         in
                         let a = signal_writes ~gpu first v
                         and b = signal_writes ~gpu next (v + 1)
                         and waits = wait_test ~gpu next v
                         and later = wait_test ~gpu next (v + 1) in
                         (* The values whose work completed, by how many of
                            their writes landed; the value before's first write
                            landed. *)
                         let values landed_late landed_a landed_b =
                           List.filter_map
                             (fun (value, landed, all) ->
                               if landed > 0 then Some (value, landed = all)
                               else None)
                             [
                               (v - 1, 1 + landed_late, 1 + List.length late);
                               (v, landed_a, List.length a);
                               (v + 1, landed_b, List.length b);
                             ]
                         in
                         interleave word [ late; a ]
                           (`Waiting (waits, b))
                           (fun word writers waiter ->
                             let landed l left =
                               List.length l - List.length left
                             in
                             let landed_b =
                               match waiter with
                               | `Waiting _ -> 0
                               | `Writing ws -> landed b ws
                             in
                             let values =
                               values
                                 (landed late (List.nth writers 0))
                                 (landed a (List.nth writers 1))
                                 landed_b
                             in
                             let completed =
                               List.fold_left (fun m (x, _) -> max m x) 0 values
                             in
                             if word > completed then
                               fail "the word reads above the completed value";
                             let mid_write =
                               List.exists (fun (_, whole) -> not whole) values
                             in
                             if (not mid_write) && word < completed then
                               fail "the word went back below a value it held";
                             (match waiter with
                             | `Waiting _ when passes waits word ->
                                 if not (List.mem_assoc v values) then
                                   fail
                                     "the wait passes before its value is \
                                      written"
                             | _ -> ());
                             if
                               passes later word
                               && not (List.mem_assoc (v + 1) values)
                             then
                               fail
                                 "the next wait passes before its value is \
                                  written")))
                   queues)
               (List.concat_map
                  (fun p -> List.map (fun f -> (p, f)) queues)
                  queues))
           [
             (1 lsl 32) - 1; 1 lsl 32; (1 lsl 32) + 1; (1 lsl 33) - 1; 1 lsl 33;
           ])
       [ "gfx1100"; "gfx942_cpx" ])

(* Room in the rings *)

(* The submission of a queue of [gpu] with [commands] encoded. *)
let submits gpu queue commands =
  let q = Hcq2.Queue.v ~devices:[ "AMD" ] queue in
  let cmds =
    (Ops_amd.queues ~host:"CPU" ~reaches:(fun _ -> false) gpu).commands q
  in
  commands cmds;
  ignore (cmds.submit ())

let buffer n =
  Ops.placeholder ~slot:0 ~device:(Multi [ "AMD" ]) [ n ] Dtype.Uint8

let room =
  let small = { (gpu "gfx1100") with copy_rings = [ 224 ] } in
  let copies n (cmds : Hcq2.commands) =
    for _ = 1 to n do
      cmds.copy (buffer 16) (buffer 16) 16
    done
  in
  let aql = { (gpu "gfx942") with compute_ring = 256 } in
  let signals n (cmds : Hcq2.commands) =
    for _ = 1 to n do
      cmds.wait (Hcq2.signal_word "AMD") (Hcq2.submitted "AMD");
      cmds.signal (Hcq2.signal_word "AMD") (Hcq2.value "AMD")
    done
  in
  group "room"
    [
      test "an SDMA command buffer of up to a quarter of its ring submits"
        (fun () -> submits small "COPY:0" (copies 2));
      test
        "a larger SDMA command buffer is over its queue's capacity, since \
         zeroing the tail doubles it" (fun () ->
          raises
            (Hcq2.Over_capacity
               "an SDMA command buffer of 84 bytes exceeds a quarter of its \
                ring of 224 bytes") (fun () ->
              submits small "COPY:0" (copies 3)));
      test "AQL packets of up to half their ring submit" (fun () ->
          submits aql "COMPUTE:0" (signals 1));
      test "more AQL packets are over their queue's capacity" (fun () ->
          raises
            (Hcq2.Over_capacity
               "AQL packets of 192 bytes exceed half their ring of 256 bytes")
            (fun () -> submits aql "COMPUTE:0" (signals 2)));
    ]

(* What the engine links *)

let pp_storage ppf = function
  | Ops_amd.Ring q -> Format.fprintf ppf "Ring %s" q
  | Write_ptr q -> Format.fprintf ppf "Write_ptr %s" q
  | Put q -> Format.fprintf ppf "Put %s" q
  | Doorbell q -> Format.fprintf ppf "Doorbell %s" q
  | Program { name; _ } -> Format.fprintf ppf "Program %s" name
  | Scratch n -> Format.fprintf ppf "Scratch %d" n
  | Log -> Format.pp_print_string ppf "Log"
  | Samples -> Format.pp_print_string ppf "Samples"
  | Traces -> Format.pp_print_string ppf "Traces"
  | Trace_ends -> Format.pp_print_string ppf "Trace_ends"

let storage = Testable.make ~pp:pp_storage ~equal:( = )

let tagged tag =
  Ops.placeholder ~slot:0 ~device:(Multi [ "AMD" ]) ~tag [ 8 ] Dtype.Uint64

(* The placeholders the engine allocates itself: command buffers, kernel
   arguments and indirect buffers, the batch's slots and address table, the
   device's signal word. *)
let allocated t =
  List.exists
    (fun prefix -> String.starts_with ~prefix t)
    [
      "cmdbuf_";
      "kernargs_";
      "ib_";
      "aql_";
      "slots";
      "inputs";
      "timeline";
      "staging";
    ]

let linking =
  group "what the engine links"
    [
      test "the words of each queue are the storage of their placeholder"
        (fun () ->
          List.iter
            (fun (tag, expected) ->
              equal ~msg:tag (option storage) expected
                (Ops_amd.storage (tagged (Ops.Tag.String tag))))
            [
              ("ring_compute_0", Some (Ops_amd.Ring "COMPUTE:0"));
              ("write_ptr_copy_1", Some (Write_ptr "COPY:1"));
              ("put_value_copy_0", Some (Put "COPY:0"));
              ("doorbell_compute_0", Some (Doorbell "COMPUTE:0"));
              ("prof_log", Some Log);
              ("pmc_buf", Some Samples);
              ("sqtt_buf", Some Traces);
              ("sqtt_wptrs", Some Trace_ends);
              ("cmdbuf_compute_0", None);
              ("slots", None);
              ("timeline", None);
            ]);
      test
        "a program's placeholder names its code object and kernel, and scratch \
         memory its bytes per lane" (fun () ->
          equal (option storage)
            (Some (Ops_amd.Program { binary = "\x7fELF"; name = "k" }))
            (Ops_amd.storage
               (tagged
                  (Ops.Tag.Tuple
                     [ String "program"; Bytes "\x7fELF"; String "k" ])));
          equal (option storage) (Some (Ops_amd.Scratch 260))
            (Ops_amd.storage
               (Ops.rtag ~tag:(Ops.Tag.String "scratch")
                  (Ops.placeholder ~slot:0 ~device:(Multi [ "AMD" ]) [ 260 ]
                     Dtype.Uint8))));
      test
        "every placeholder of a recorded batch is AMD's storage, or the \
         engine's to allocate" (fun () ->
          List.iter
            (fun (case, name, profile) ->
              let compiled =
                plain (fun () ->
                    Hcq2.compile_linear ~profile
                      ~devices:(recorded_devices name)
                      (Golden.sink (case ^ "_prepared.golden")))
              in
              List.iter
                (fun u ->
                  match (Ops.op u, Ops.tag u) with
                  | Param, Some tag -> (
                      match (Ops_amd.storage u, tag) with
                      | Some _, _ -> ()
                      | None, String t when allocated t -> ()
                      | None, t ->
                          fail (Format.asprintf "%s: %a" case Ops.Tag.pp t))
                  | _ -> ())
                (Ops.toposort ~calls:Enter compiled))
            cases);
    ]

let refusals =
  group "refusals"
    [
      test "a target none of gfx942, gfx950, gfx11 and gfx12 is refused"
        (fun () ->
          raises (Invalid_argument "gfx1030 is not a supported AMD GPU")
            (fun () ->
              ignore
                (Ops_amd.queues ~host:"CPU"
                   ~reaches:(fun _ -> false)
                   { (gpu "gfx1100") with target = (10, 3, 0) })));
      test "PM4 packets on several dies are refused" (fun () ->
          raises
            (Invalid_argument "an AMD GPU of several dies takes AQL packets")
            (fun () ->
              ignore
                (Ops_amd.queues ~host:"CPU"
                   ~reaches:(fun _ -> false)
                   { (gpu "gfx942") with aql = false })));
      test "a copy queue the GPU lacks is refused" (fun () ->
          raises (Invalid_argument "the AMD GPU has no copy queue COPY:2")
            (fun () -> submits (gpu "gfx1100") "COPY:2" ignore));
      test "a compute queue does not copy, and a copy queue runs no program"
        (fun () ->
          raises (Invalid_argument "an AMD compute queue does not copy")
            (fun () ->
              submits (gpu "gfx1100") "COMPUTE:0" (fun cmds ->
                  cmds.copy (buffer 4) (buffer 4) 4));
          raises (Invalid_argument "an AMD copy queue runs no program")
            (fun () ->
              submits (gpu "gfx1100") "COPY:0" (fun cmds ->
                  cmds.exec (buffer 4) (buffer 4))));
    ]

(* Loops *)

(* The code object of the recorded case [case]'s first kernel. *)
let code_object case =
  let binaries =
    List.filter_map
      (fun u ->
        match (Ops.op u, Ops.arg u) with Binary, Bytes s -> Some s | _ -> None)
      (Ops.toposort ~calls:Enter (Golden.sink (case ^ "_prepared.golden")))
  in
  List.find (fun b -> String.starts_with ~prefix:"\x7fELF" b) binaries

(* A range of three trips around a kernel adding one on [name]'s AMD device,
   each trip on its own window of four floats, as a schedule. *)
let ranged ?(trips = 3) ?(copy = false) name case =
  let g = gpu name in
  let m, n, s = g.target in
  let target =
    { host_target with device = "AMD"; arch = Printf.sprintf "gfx%d%x%x" m n s }
  in
  let renderer =
    Renderer.with_compiler
      (Renderer.Compiler.v (fun _ -> code_object case))
      (Cstyle.hip target)
  in
  let param slot =
    Ops.param ~shape:[ Int 4 ] ~device:(Single "AMD") slot Float32
  in
  let i = Ops.range (Int 4) [ 0 ] in
  let x = Ops.load (Ops.index (param 1) [ i ]) [] in
  let st =
    Ops.store
      (Ops.index (param 0) [ i ])
      (Ops.add x (Ops.float ~dtype:Float32 1.))
  in
  let kernel =
    Ops.sink ~kernel:(Ops.kernel_info ~name:"k" ()) [ Ops.end_ st [ i ] ]
  in
  let r = Ops.range (Int trips) [ 7 ] in
  let window u =
    let start = Ops.mul r (Ops.int 4) in
    Ops.shrink u [ Some (Sym start, Sym (Ops.add start (Ops.int 4))) ]
  in
  let buf () = Ops.new_buffer (Single "AMD") (4 * trips) Float32 in
  let call =
    if copy then Ops.store_call (window (buf ())) (window (buf ()))
    else
      Ops.call
        (Codegen.to_program kernel renderer)
        [ window (buf ()); window (buf ()) ]
  in
  Ops.v Linear ~src:[ Ops.end_ call [ r ] ]

let batched ?(profile = Hcq2.Unstamped) name case =
  let linear =
    plain (fun () ->
        Hcq2.compile_linear ~profile ~devices:(recorded_devices name)
          (ranged name case))
  in
  let contains sub =
    let src = host_sources linear in
    let n = String.length sub in
    let rec go i =
      i + n <= String.length src && (String.sub src i n = sub || go (i + 1))
    in
    go 0
  in
  (List.length (List.filter is_batch (Ops.src linear)), contains)

let loops =
  group "loops"
    [
      test
        "a range is one batch whose PM4 commands and arguments repeat per trip"
        (fun () ->
          let batches, contains = batched "gfx1100" "chain" in
          equal ~msg:"one batch" int 1 batches;
          is_true ~msg:"a loop of three trips" (contains "< 3; Lidx");
          is_false ~msg:"addresses in integers" (contains "float"));
      test
        "a profiled range is one batch whose AQL packets repeat per trip, each \
         running its trip's commands" (fun () ->
          let batches, contains =
            batched ~profile:Stamped "gfx942" "chain_gfx942"
          in
          equal ~msg:"one batch" int 1 batches;
          is_true ~msg:"a loop of three trips" (contains "< 3; Lidx");
          (* A trip's timestamps are 64 bytes of PM4 commands. *)
          is_true ~msg:"each trip's indirect buffer runs its trip's commands"
            (contains "))<<6ul))");
          is_false ~msg:"addresses in integers" (contains "float"));
    ]

(* Batches split at their queues' capacity *)

(* The batches of a range of [trips] trips of a kernel, or of a copy, on a
   device [g] like the one [name] names, each with its trips and the bytes its
   queue's ring takes, from the placeholder [tag]: those of the range's chunks
   first. *)
let pieces ?copy ~trips g name case tag =
  let linear =
    plain (fun () ->
        Hcq2.compile_linear ~profile:Unstamped ~devices:(gpu_devices g)
          (ranged ?copy ~trips name case))
  in
  List.map
    (fun b ->
      let trips =
        match Ops.arg (Ops.without_after b) with
        | Call { aux = Some info; _ } -> List.length info.kernels
        | _ -> fail "a batch has its kernels"
      in
      let bytes =
        List.find_map
          (fun u ->
            match Ops.tag u with
            | Some (String t) when t = tag -> Some (Ops.max_numel u)
            | _ -> None)
          (Ops.toposort ~calls:Enter b)
      in
      (trips, Option.get bytes))
    (List.filter is_batch
       (List.concat_map
          (fun e ->
            if Ops.op e = End then Ops.src (Ops.nth e 0) @ [ Ops.nth e 0 ]
            else [ e ])
          (Ops.src linear)))

let splits =
  group "splits"
    [
      test
        "a range of 10,000 trips on AQL runs as batches of up to half the ring"
        (fun () ->
          let ring = 128 * 1024 in
          let g = { (gpu "gfx942") with compute_ring = ring } in
          let ps =
            pieces ~trips:10_000 g "gfx942" "chain_gfx942" "aql_compute_0"
          in
          (* A chunk of 1,024 trips of 64 bytes is over half the ring, so its
             batch runs as two of 512, and the 784 trips left fit. *)
          equal ~msg:"trips and AQL bytes of each batch"
            (list (pair int int))
            [ (512, 32_896); (512, 32_896); (784, 50_304) ]
            ps;
          List.iter (fun (_, bytes) -> at_most int ~than:(ring / 2) bytes) ps);
      test
        "a range of 1,000 copies runs as batches of up to a quarter of the \
         copy ring" (fun () ->
          let ring = 64 * 1024 in
          let g = { (gpu "gfx1100") with copy_rings = [ ring ] } in
          let ps =
            pieces ~copy:true ~trips:1_000 g "gfx1100" "chain" "cmdbuf_copy_0"
          in
          equal ~msg:"trips and command bytes of each batch"
            (list (pair int int))
            [ (500, 14_064); (500, 14_064) ]
            ps;
          List.iter (fun (_, bytes) -> at_most int ~than:(ring / 4) bytes) ps);
    ]

(* Packets *)

module P = Nx_amd_packet

type width = Word32 | Word64

(* A packet over values of [widths], whose other parameters it reads from
   [ints], drawn alike for both encodings. *)
type command = {
  name : string;
  widths : width list;
  build : 'v. int array -> 'v array -> 'v P.word list;
}

(* A caller's value as a word's term, as it is. *)
let t value = P.Value value
let gcs = [| (9, 4, 3); (11, 0, 0); (11, 5, 0); (12, 0, 0) |]
let sdmas = [| (4, 4, 2); (5, 0, 0); (5, 2, 0); (6, 0, 0); (7, 0, 0) |]
let pick a i = a.(i mod Array.length a)
let flag i = i mod 2 = 1
let cmp i : P.comparison = if flag i then Equal else Greater_equal

let commands =
  [
    {
      name = "Pm4.set_reg of an SH register";
      widths = [ Word32; Word64 ];
      build =
        (fun i v ->
          P.Pm4.set_reg
            (0x2c00 + (i.(0) mod 0x400))
            [
              W32 (t v.(0));
              W64 (Shift (Add (Value v.(1), Int64.of_int i.(2)), i.(3) mod 32));
              Dword i.(1);
            ]);
    };
    {
      name = "Pm4.set_reg of a UCONFIG register";
      widths = [ Word32 ];
      build =
        (fun i v ->
          P.Pm4.set_reg (0xc000 + (i.(0) mod 0x1000)) [ W32 (t v.(0)) ]);
    };
    {
      name = "Pm4.wait on memory";
      widths = [ Word64; Word32 ];
      build =
        (fun i v ->
          P.Pm4.wait
            ~gc:(pick gcs i.(4))
            (Memory v.(0))
            (cmp i.(0))
            v.(1) ~mask:i.(1) ~interval:i.(2));
    };
    {
      name = "Pm4.wait on a register";
      widths = [ Word32 ];
      build =
        (fun i v ->
          P.Pm4.wait
            ~gc:(pick gcs i.(4))
            (Register i.(3))
            (cmp i.(0))
            v.(0) ~mask:i.(1) ~interval:i.(2));
    };
    {
      name = "Pm4.acquire_mem";
      widths = [];
      build =
        (fun i _ ->
          P.Pm4.acquire_mem
            ~gc:(pick gcs i.(0))
            (if flag i.(1) then All_caches else Data_caches));
    };
    {
      name = "Pm4.release_mem";
      widths = [ Word64; Word64 ];
      build =
        (fun i v ->
          let data : _ P.Pm4.data =
            if flag i.(1) then Low_32 v.(1) else Data_64 v.(1)
          in
          P.Pm4.release_mem ~gc:(pick gcs i.(0)) v.(0) data);
    };
    {
      name = "Pm4.pred_exec";
      widths = [];
      build =
        (fun i _ ->
          P.Pm4.pred_exec ~xcc_mask:(i.(0) land 0xff) ~dwords:(i.(1) land 0x3fff));
    };
    {
      name = "Pm4.event_write";
      widths = [];
      build =
        (fun i _ ->
          P.Pm4.event_write
            (pick
               [|
                 P.Pm4.Cs_partial_flush;
                 Thread_trace_marker;
                 Thread_trace_finish;
               |]
               i.(0)));
    };
    {
      name = "Pm4.copy_data";
      widths = [ Word64 ];
      build =
        (fun i v ->
          P.Pm4.copy_data
            (if flag i.(0) then Confirmed else Posted)
            (if flag i.(2) then Clock else Counter i.(1))
            v.(0));
    };
    {
      name = "Pm4.write_data to a register";
      widths = [ Word32 ];
      build = (fun i v -> P.Pm4.write_data (Register i.(0)) v.(0));
    };
    {
      name = "Pm4.write_data to memory";
      widths = [ Word64; Word32 ];
      build = (fun _ v -> P.Pm4.write_data (Memory v.(0)) v.(1));
    };
    {
      name = "Pm4.indirect_buffer";
      widths = [ Word64 ];
      build =
        (fun i v -> P.Pm4.indirect_buffer v.(0) ~dwords:(i.(0) land 0xfffff));
    };
    {
      name = "Pm4.dispatch_direct";
      widths = [ Word32; Word32; Word32 ];
      build =
        (fun i v ->
          P.Pm4.dispatch_direct
            ~gc:(pick gcs i.(0))
            (if flag i.(1) then Wave32 else Wave64)
            (v.(0), v.(1), v.(2)));
    };
    {
      name = "Aql.dispatch";
      widths = [ Word32; Word32; Word32; Word64; Word64 ];
      build =
        (fun i v ->
          P.Aql.dispatch
            ~threads:(i.(0) land 0x3ff, i.(1) land 0x3ff, i.(2) land 0x3ff)
            ~grid:(v.(0), v.(1), v.(2))
            ~private_segment:i.(3) ~group_segment:i.(4) ~descriptor:v.(3)
            ~args:v.(4));
    };
    {
      name = "Aql.indirect_buffer";
      widths = [ Word64 ];
      build =
        (fun i v -> P.Aql.indirect_buffer v.(0) ~dwords:(i.(0) land 0xfffff));
    };
    {
      name = "Sdma.copy";
      widths = [ Word64; Word64 ];
      build =
        (fun i v ->
          P.Sdma.copy ~sdma:(pick sdmas i.(0)) ~dst:v.(0) ~src:v.(1) i.(1));
    };
    {
      name = "Pm4.set_program";
      widths = [ Word64 ];
      build = (fun i v -> P.Pm4.set_program ~gc:(pick gcs i.(0)) v.(0));
    };
    {
      name = "Pm4.set_scratch";
      widths = [ Word64 ];
      build = (fun i v -> P.Pm4.set_scratch ~gc:(pick gcs i.(0)) v.(0));
    };
    {
      name = "Sdma.poll";
      widths = [ Word64; Word32 ];
      build = (fun i v -> P.Sdma.poll v.(0) (cmp i.(0)) v.(1) ~mask:i.(1));
    };
    {
      name = "Sdma.fence";
      widths = [ Word64; Word32 ];
      build = (fun i v -> P.Sdma.fence ~sdma:(pick sdmas i.(0)) v.(0) v.(1));
    };
    { name = "Sdma.trap"; widths = []; build = (fun _ _ -> P.Sdma.trap) };
    {
      name = "Sdma.timestamp";
      widths = [ Word64 ];
      build = (fun _ v -> P.Sdma.timestamp v.(0));
    };
  ]

(* A 64-bit value is any non-negative integer, so that a term's sum reaches past
   bit 62. *)
let largest = function Word32 -> 0xffff_ffff | Word64 -> max_int
let dtype_of = function Word32 -> Dtype.Uint32 | Word64 -> Dtype.Uint64

(* A value of a word, its bounds and their neighbours among them. *)
let value w =
  let n = largest w in
  Gen.frequency
    [
      (6, Gen.int_range 0 n);
      (1, Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 0xffff_ffff; n - 1; n ]);
    ]

let values ws =
  List.fold_right
    (fun w acc ->
      let open Gen in
      let+ v = value w and+ vs = acc in
      v :: vs)
    ws (Gen.constant [])

(* The words a queue holds after the packet [c] over variables, each bound to
   its value of [vs]: the packet as tolk encodes it, evaluated. *)
let symbolic c ints vs =
  let vars =
    List.mapi
      (fun i w ->
        Ops.variable ~dtype:(dtype_of w) (Printf.sprintf "v%d" i)
          (`Int Bigint.zero)
          (`Int (Bigint.of_int (largest w))))
      c.widths
  in
  let nodes = Ops_amd.lower (c.build ints (Array.of_list vars)) in
  let bound =
    Ops.src
      (Ops.substitute ~calls:Skip ~pass:Fixed_point (Ops.sink nodes)
         (List.map2
            (fun var (w, x) -> (var, Ops.int ~dtype:(dtype_of w) x))
            vars (List.combine c.widths vs)))
  in
  (* A word that computes on its bound value, such as an offset's sum, is
     evaluated to the constant of its type. *)
  let bound =
    List.map
      (fun n ->
        match Ops.value n with
        | _ -> n
        | exception Invalid_argument _ ->
            Ops.const ~dtype:(Ops.dtype n) (`Int (Ops.to_z n)))
      bound
  in
  let q = Hcq2.Queue.v ~devices:[ "AMD" ] "COMPUTE:0" in
  ignore (Hcq2.Queue.q q bound);
  List.init (Hcq2.Queue.size q / 4) (fun i -> Hcq2.Queue.get_dword q (4 * i))

let packets =
  group "packets"
    (List.map
       (fun c ->
         prop
           (c.name ^ ": tolk's encoding, evaluated, is the integer encoding")
           Gen.(
             pair
               (array ~size:(constant 5) (int_range 0 (1 lsl 30)))
               (values c.widths))
           (fun (ints, vs) ->
             equal (list int)
               (P.dwords (c.build ints (Array.of_list vs)))
               (symbolic c ints vs)))
       commands)

let waves_keyed =
  test "WAVES_PER_SH is keyed with what shapes compilation" (fun () ->
      equal (option string) (Some "0")
        (List.assoc_opt "WAVES_PER_SH" (Setting.shaping ())))

let () =
  exit
    (run "Tolk.Ops_amd"
       [
         recorded;
         recorded_words;
         carry_law;
         room;
         splits;
         linking;
         refusals;
         loops;
         packets;
         waves_keyed;
       ])
