open Windtrap
open Tolk

let rejects f = raises_match (Exn.invalid_arg ?substring:None) f
let uop = Uops.uop
let plain f = Helpers.context [ B (Helpers.no_color, true) ] f
let op = Testable.make ~pp:Op.pp ~equal:Op.equal

(* Nodes *)

let placement u =
  match Ops.device u with
  | Some (Single d) -> [ d ]
  | Some (Multi ds) -> ds
  | None -> []

let info_of call =
  match Ops.arg (Ops.without_after call) with
  | Call { aux = Some info; _ } -> info
  | _ -> failf "%a is no batch" Ops.pp call

let is_batch c =
  match Ops.arg (Ops.without_after c) with
  | Call { aux = Some _; _ } -> true
  | _ -> false

let batches linear = List.filter is_batch (Ops.src linear)

let the_batch linear =
  match batches linear with
  | [ b ] -> b
  | bs -> failf "one batch, not %d" (List.length bs)

let tag_of u = match Ops.tag u with Some (String s) -> s | _ -> ""
let nodes op u = List.filter (fun n -> Ops.op n = op) (Ops.toposort u)

let instruction c =
  match (Ops.op c, Ops.arg c) with
  | Ins, Code { code; _ } -> code
  | Call, _ -> "call"
  | _ -> "?"

let zero_estimates : Ops.estimates = { ops = Int 0; lds = Int 0; mem = Int 0 }

let batch_info devices : Ops.hcq_info =
  {
    device = devices;
    kernels = [];
    estimates = zero_estimates;
    nargs = 0;
    table = -1;
    inputs = [];
    slots = [];
    written_bufs = [];
    writes = [];
    copies = [];
  }

(* Kernels and calls *)

let recorded_target =
  {
    Helpers.Target.device = "CPU";
    renderer = "";
    arch = "x86_64,x86-64";
    interface = "";
    indices = "";
  }

(* A renderer whose binary is its source's bytes: batching compiles nothing. *)
let uncompiled =
  Renderer.with_compiler
    (Renderer.Compiler.v Fun.id)
    (Cstyle.clang recorded_target)

let param ?(dtype = Dtype.Float32) device slot n =
  Ops.param ~shape:[ Int n ] ~device slot dtype

(* The kernel that stores [f] of each element of slot 1 into slot 0. *)
let map_kernel ?(name = "k") device n f =
  let out = param device 0 n and inp = param device 1 n in
  let i = Ops.range (Int n) [ 0 ] in
  let st = Ops.store (Ops.index out [ i ]) (f (Ops.index inp [ i ])) in
  Ops.sink ~kernel:(Ops.kernel_info ~name ()) [ Ops.end_ st [ i ] ]

let plus c x = Ops.add x (Ops.float ~dtype:Float32 c)
let storage ?(n = 4) device = Ops.new_buffer (Single device) n Float32
let linear calls = Ops.v Linear ~src:calls

(* A call of the kernel adding [c] on [out]'s device, and that call compiled, as
   a batch holds it. *)
let kernel_adds ?(c = 1.) out inp =
  Ops.call (map_kernel (Option.get (Ops.device out)) 4 (plus c)) [ out; inp ]

let adds ?c out inp =
  let call = kernel_adds ?c out inp in
  Ops.replace
    ~src:
      (Codegen.to_program (Ops.nth call 0) uncompiled :: List.tl (Ops.src call))
    call

(* Devices by kind *)

let kind name =
  match String.index_opt name ':' with
  | Some i -> String.sub name 0 i
  | None -> name

(* The host "CPU", and any "KIND:i" as a device of that kind with queues. *)
let kinds ?(copy_queue = true) ?(submission = Hcq2.Buffered) () =
  let events = Null_queue.events () in
  fun name ->
    if name = "CPU" then { Hcq2.target = recorded_target; queues = None }
    else
      let queues =
        {
          Hcq2.commands = Null_queue.commands events;
          copy_queue;
          submission;
          host = "CPU";
          reaches = (fun _ -> true);
        }
      in
      {
        Hcq2.target = { recorded_target with device = kind name };
        queues = Some queues;
      }

let sched ?(profile = false) ?copy_queue ?submission calls =
  Hcq2.sched_batches
    ~devices:(kinds ?copy_queue ?submission ())
    ~profile (linear calls)

(* Recorded cases *)

(* The devices of the goldens: the host, and CPU:1 to CPU:3 with the NULL
   device's queues, numbering their events afresh. *)
let recorded_devices ?(copy_queue = true) () =
  let events = Null_queue.events () in
  function
  | "CPU" -> { Hcq2.target = recorded_target; queues = None }
  | _ ->
      let queues =
        {
          Hcq2.commands = Null_queue.commands events;
          copy_queue;
          submission = Buffered;
          host = "CPU";
          reaches = (fun _ -> true);
        }
      in
      { Hcq2.target = recorded_target; queues = Some queues }

let recorded_graph file actual =
  test file (fun () ->
      let golden = Golden.sink file in
      let same u = Graph.to_string (Uops.without_profile_keys u) in
      equal text (same golden)
        (same (Uops.placeholders_like golden (actual ()))))

let host_sources linear =
  String.concat ""
    (List.map
       (fun b ->
         match Ops.arg (Ops.nth (Ops.nth (Ops.without_after b) 0) 2) with
         | String src -> src
         | _ -> fail "a compiled host program holds its source")
       (batches linear))

(* Each case with whether it profiles and has copy queues. *)
let cases =
  [
    ("chain", false, true);
    ("chain_profile", true, true);
    ("peer_copy", false, true);
    ("peer_copy_kernel", false, false);
    ("peer_copy_profile", true, true);
    ("sharded", false, true);
    ("host_split", false, true);
    ("sharded_sum", false, true);
    ("copies", false, true);
    ("variable", false, true);
  ]

(* Compiling these takes long. *)
let heavy = [ "sharded"; "host_split"; "sharded_sum" ]

let recorded =
  group "recorded cases"
    (List.map
       (fun (case, profile, copy_queue) ->
         let compiled =
           lazy
             (plain (fun () ->
                  Hcq2.compile_linear ~profile
                    ~devices:(recorded_devices ~copy_queue ())
                    (Golden.sink (case ^ ".golden"))))
         in
         [
           recorded_graph (case ^ "_batched.golden") (fun () ->
               plain (fun () ->
                   Hcq2.sched_batches
                     ~devices:(recorded_devices ~copy_queue ())
                     ~profile
                     (Golden.sink (case ^ "_prepared.golden"))));
           recorded_graph (case ^ "_compiled.golden") (fun () ->
               Uops.binaries_as_sources (Lazy.force compiled));
           Golden.text (case ^ "_host.golden") (fun () ->
               host_sources (Lazy.force compiled));
         ]
         |> group ~tags:(if List.mem case heavy then [ "slow" ] else []) case)
       cases)

let profile_keys =
  group "profile keys"
    [
      test "a kernel's profile key is its program's key" (fun () ->
          let b = List.init 3 (fun _ -> storage "CPU:1") in
          let batch =
            the_batch
              (sched
                 [
                   adds (List.nth b 1) (List.nth b 0);
                   adds (List.nth b 2) (List.nth b 1);
                 ])
          in
          let programs =
            List.map (fun c -> Ops.key (Ops.nth c 0)) (Batches.calls batch)
          in
          equal
            (list (option string))
            (List.map Option.some programs)
            (List.map
               (fun (k : Ops.hcq_kernel) -> k.profile_key)
               (info_of batch).kernels));
      test "a copy has no profile key" (fun () ->
          let batch =
            the_batch
              (sched [ Ops.store_call (storage "CPU:2") (storage "CPU:1") ])
          in
          equal
            (list (option string))
            [ None ]
            (List.map
               (fun (k : Ops.hcq_kernel) -> k.profile_key)
               (info_of batch).kernels));
    ]

(* Words and storage *)

let bytes8 = Ops.new_buffer (Single "CPU:1") 64 Uint8

let views =
  group "unwrap_view and unwrap_lane"
    [
      test "storage is its own view, from byte 0" (fun () ->
          let base, off = Hcq2.unwrap_view bytes8 in
          equal uop bytes8 base;
          equal int 0 off);
      test "a shrink is a view from its start, in bytes of its type" (fun () ->
          let floats = Ops.new_buffer (Single "CPU:1") 16 Float32 in
          equal int 16
            (snd
               (Hcq2.unwrap_view (Ops.shrink floats [ Some (Int 4, Int 12) ]))));
      test "a shrink of a bitcast counts in the bitcast's type" (fun () ->
          let v =
            Ops.shrink (Ops.bitcast bytes8 Uint64) [ Some (Int 2, Int 6) ]
          in
          let base, off = Hcq2.unwrap_view v in
          equal uop bytes8 base;
          equal int 16 off);
      test "nested shrinks add up" (fun () ->
          let v =
            Ops.shrink
              (Ops.shrink bytes8 [ Some (Int 8, Int 40) ])
              [ Some (Int 4, Int 12) ]
          in
          equal int 12 (snd (Hcq2.unwrap_view v)));
      test "sees through an after" (fun () ->
          let v =
            Ops.shrink
              (Ops.after bytes8 [ Ops.store_call bytes8 bytes8 ])
              [ Some (Int 8, Int 16) ]
          in
          let base, off = Hcq2.unwrap_view v in
          equal uop bytes8 base;
          equal int 8 off);
      test "unwrap_lane is unwrap_view with no lane off a shard selection"
        (fun () ->
          let v = Ops.shrink bytes8 [ Some (Int 8, Int 16) ] in
          let base, lane, off = Hcq2.unwrap_lane v in
          equal uop bytes8 base;
          equal (option int) None lane;
          equal int 8 off);
      test
        "unwrap_lane reads a selected shard's lane and offset, either side of \
         the selection" (fun () ->
          let b =
            Ops.param ~shape:[ Int 64 ]
              ~device:(Multi [ "CPU:1"; "CPU:2" ])
              0 Float32
          in
          let seen v =
            let base, lane, off = Hcq2.unwrap_lane v in
            (base == b, lane, off)
          in
          let t = triple bool (option int) int in
          equal t (true, Some 0, 32)
            (seen (Ops.shrink (Ops.mselect b 0) [ Some (Int 8, Int 16) ]));
          equal t (true, Some 0, 32)
            (seen (Ops.mselect (Ops.shrink b [ Some (Int 8, Int 16) ]) 0));
          equal t (true, Some 1, 32)
            (seen
               (Ops.shrink
                  (Ops.mselect (Ops.shrink b [ Some (Int 4, Int 32) ]) 1)
                  [ Some (Int 4, Int 12) ])));
    ]

let names =
  group "to_name"
    [
      test "joins with _, lowercases and replaces each :" (fun () ->
          equal string "cmdbuf_compute_0"
            (Hcq2.to_name [ "cmdbuf"; "COMPUTE:0" ]);
          equal string "submitted_cpu_1" (Hcq2.to_name [ "submitted"; "CPU:1" ]));
      test "is empty for no part" (fun () -> equal string "" (Hcq2.to_name []));
    ]

(* Timeline values are parameters *)

let param_arg u =
  match Ops.arg u with Param p -> p | _ -> failf "%a is no parameter" Ops.pp u

let timeline_values =
  group "timeline values"
    [
      test "a signal word is one volatile uint64 of its device, tagged timeline"
        (fun () ->
          let w = Hcq2.signal_word "CPU:1" in
          let p = param_arg w in
          equal (option int) (Some 1) p.size;
          is_true ~msg:"volatile" p.volatile;
          equal string "timeline" (tag_of w);
          equal (list string) [ "CPU:1" ] (placement w);
          is_true ~msg:"uint64" (Dtype.equal Uint64 p.dtype));
      test "submitted and value are uint64 variables named after their device"
        (fun () ->
          let check what v =
            let p = param_arg v in
            is_true ~msg:"a variable" (Ops.is_variable v);
            equal string (Hcq2.to_name [ what; "CPU:1" ]) (Ops.expr v);
            is_true ~msg:"uint64" (Dtype.equal Uint64 p.dtype)
          in
          check "submitted" (Hcq2.submitted "CPU:1");
          check "value" (Hcq2.value "CPU:1"));
      test
        "a queue starts with a barrier, then waits for its devices' submitted \
         work" (fun () ->
          let src = storage "CPU:1" and dst = storage "CPU:2" in
          let batch =
            the_batch
              (sched [ Ops.store_call dst src; adds (storage "CPU:2") dst ])
          in
          List.iter
            (fun ((d, q), cmds) ->
              let first = List.filteri (fun i _ -> i < 2) cmds in
              equal (list string)
                ~msg:(d ^ " " ^ q)
                [ "barrier"; "wait" ]
                (List.map instruction first);
              let wait = List.nth first 1 in
              equal uop ~msg:"the signal word" (Hcq2.signal_word d)
                (Ops.nth wait 0);
              equal uop ~msg:"the submitted value" (Hcq2.submitted d)
                (Ops.nth wait 1))
            (Batches.queues batch));
      test
        "a queue touching a peer's memory waits for the peer's submitted work \
         too" (fun () ->
          let src = storage "CPU:1" and dst = storage "CPU:2" in
          let batch = the_batch (sched [ Ops.store_call dst src ]) in
          let copy = List.assoc ("CPU:1", "COPY:0") (Batches.queues batch) in
          let waits = List.filter (fun c -> instruction c = "wait") copy in
          equal (list uop)
            [ Hcq2.submitted "CPU:1"; Hcq2.submitted "CPU:2" ]
            (List.map
               (fun w -> Ops.nth w 1)
               (List.filteri (fun i _ -> i < 2) waits)));
      test "each device's work ends storing its value into its signal word"
        (fun () ->
          let batch =
            the_batch (sched [ adds (storage "CPU:1") (storage "CPU:1") ])
          in
          let cmds = List.assoc ("CPU:1", "COMPUTE:0") (Batches.queues batch) in
          let last = List.nth cmds (List.length cmds - 1) in
          equal string "store" (instruction last);
          equal uop (Hcq2.signal_word "CPU:1") (Ops.nth last 0);
          equal uop (Hcq2.value "CPU:1") (Ops.nth last 1));
      test "a host program neither reads a signal word nor loops" (fun () ->
          let b = List.init 3 (fun _ -> storage "CPU:1") in
          let compiled =
            Hcq2.compile_linear ~devices:(recorded_devices ())
              (linear
                 [
                   adds (List.nth b 1) (List.nth b 0);
                   adds (List.nth b 2) (List.nth b 1);
                 ])
          in
          let host = Ops.nth (Ops.without_after (the_batch compiled)) 0 in
          let loads =
            List.filter
              (fun l ->
                tag_of (fst (Hcq2.unwrap_view (Ops.nth (Ops.nth l 0) 0)))
                = "timeline")
              (nodes Load host)
          in
          equal int ~msg:"loads of a signal word" 0 (List.length loads);
          equal int ~msg:"loops" 0
            (List.length (nodes Backedge host) + List.length (nodes Range host)));
      test "the fence is of every queue's signal" (fun () ->
          let src = storage "CPU:1" and dst = storage "CPU:2" in
          let batch =
            the_batch
              (sched [ Ops.store_call dst src; adds (storage "CPU:2") dst ])
          in
          let fences =
            List.sort_uniq Ops.compare
              (List.filter
                 (fun n ->
                   Ops.op n = Custom_function && Ops.arg n = String "hcq_fence")
                 (Ops.toposort batch))
          in
          match fences with
          | [ fence ] ->
              equal int ~msg:"queue signals, one per queue"
                (List.length (Batches.queues batch))
                (List.length (Ops.src fence));
              List.iter
                (fun s ->
                  equal string "slots" (tag_of (fst (Hcq2.unwrap_view s))))
                (Ops.src fence)
          | fs -> failf "one fence, not %d" (List.length fs));
      test "the fence lowers to stores of zero into the slots, and nothing else"
        (fun () ->
          let src = storage "CPU:1" and dst = storage "CPU:2" in
          let batch =
            the_batch
              (sched [ Ops.store_call dst src; adds (storage "CPU:2") dst ])
          in
          let host =
            Ops.nth
              (Ops.without_after (Hcq2.lower_call ~devices:(kinds ()) batch))
              0
          in
          let into_slots st =
            match Ops.src st with
            | dst :: _ ->
                tag_of (fst (Hcq2.unwrap_view (Ops.nth dst 0))) = "slots"
                || String.starts_with ~prefix:"slots"
                     (Option.value ~default:""
                        (param_arg (fst (Hcq2.unwrap_view (Ops.nth dst 0))))
                          .name)
            | [] -> false
          in
          let stores = List.filter into_slots (nodes Store host) in
          equal int ~msg:"stores into the slots" 2 (List.length stores);
          List.iter
            (fun st ->
              equal uop
                (Ops.const ~dtype:Uint64 (`Int Bigint.zero))
                (Ops.nth st 1))
            stores);
    ]

(* Layout of arguments *)

let words_testable = list (pair int uop)
let u8 n = Ops.int ~dtype:Uint8 n
let u16 n = Ops.int ~dtype:Uint16 n
let u32 n = Ops.int ~dtype:Uint32 n
let u64 n = Ops.int ~dtype:Uint64 n
let binary n = Ops.v Binary ~arg:(Bytes (String.make n '\000'))

let layouts =
  group "layout_args and pack_args"
    [
      test "lays each argument out aligned to its size, in order" (fun () ->
          let args = [ u8 1; u32 2; u16 3; u64 4; u8 5 ] in
          equal words_testable
            (List.combine [ 0; 4; 8; 16; 24 ] args)
            (Hcq2.layout_args args));
      test "starts from its offset" (fun () ->
          let args = [ u16 1; u64 2 ] in
          equal words_testable
            (List.combine [ 6; 8 ] args)
            (Hcq2.layout_args ~offset:6 args));
      test "packs words at their offsets, zeros between and after" (fun () ->
          equal (list uop)
            [ binary 2; u16 7; binary 4; u64 9; binary 4 ]
            (Hcq2.pack_args [ (8, u64 9); (2, u16 7) ] 20));
      test "leaves an empty zero run where the words fill the size" (fun () ->
          equal (list uop)
            [ u32 1; u32 2; binary 0 ]
            (Hcq2.pack_args (Hcq2.layout_args [ u32 1; u32 2 ]) 8));
    ]

(* Dependencies *)

let deps_tests =
  let access t bufs writes x = Hcq2.Deps.access t bufs ~writes x in
  let b = Ops.param ~shape:[ Int 16 ] ~device:(Single "CPU:1") 0 Uint8 in
  let v o n = Ops.shrink b [ Some (Int o, Int (o + n)) ] in
  group "Deps"
    [
      test "views of one storage depend on the bytes they share" (fun () ->
          let floats = Ops.bitcast b Uint16 in
          List.iter
            (fun writes ->
              let t = Hcq2.Deps.make () in
              ignore (access t [ b ] writes 0);
              equal (list int) [ 0 ]
                (access t [ Ops.shrink floats [ Some (Int 2, Int 6) ] ] [ 0 ] 1);
              equal (list int) [ 0 ] (access t [ v 0 4 ] [ 0 ] 2);
              equal (list int) [ 0 ] (access t [ v 12 4 ] [ 0 ] 3);
              equal (list int) [ 1 ] (access t [ v 4 8 ] [] 4))
            [ []; [ 0 ] ]);
      test "shard selections depend on their lane's bytes alone" (fun () ->
          let m =
            Ops.param ~shape:[ Int 64 ]
              ~device:(Multi [ "CPU:1"; "CPU:2" ])
              0 Float32
          in
          let s u o n = Ops.shrink u [ Some (Int o, Int (o + n)) ] in
          List.iter
            (fun view ->
              let t = Hcq2.Deps.make () in
              ignore (access t [ view ] [ 0 ] 0);
              equal (list int) ~msg:"another lane" []
                (access t [ Ops.mselect m 1 ] [] 1);
              equal (list int) ~msg:"other bytes" []
                (access t [ s (Ops.mselect m 0) 16 8 ] [] 2);
              equal (list int) ~msg:"overlapping bytes" [ 0 ]
                (access t [ s (Ops.mselect m 0) 12 8 ] [] 3))
            [
              s (Ops.mselect m 0) 8 8;
              Ops.mselect (s m 8 8) 0;
              s (Ops.mselect (s m 4 28) 0) 4 8;
            ]);
      test
        "a write that does not trim keeps the accesses to the bytes it writes"
        (fun () ->
          let t = Hcq2.Deps.make () in
          ignore (access t [ v 0 4 ] [ 0 ] 0);
          ignore (Hcq2.Deps.access ~trim:false t [ v 0 4 ] ~writes:[ 0 ] 1);
          equal (list int) [ 0; 1 ] (access t [ v 0 4 ] [] 2));
      test "forgotten accesses are no longer followed" (fun () ->
          let t = Hcq2.Deps.make () in
          ignore (access t [ v 0 4 ] [ 0 ] 0);
          ignore (access t [ v 4 4 ] [ 0 ] 1);
          Hcq2.Deps.forget t (fun x -> x = 0);
          equal (list int) [] (access t [ v 0 4 ] [] 2);
          equal (list int) [ 1 ] (access t [ v 4 4 ] [] 3));
      test "a write of other bytes keeps the dependencies" (fun () ->
          List.iter
            (fun writes ->
              let t = Hcq2.Deps.make () in
              ignore (access t [ v 0 4 ] writes 0);
              equal (list int) [] (access t [ v 4 4 ] [ 0 ] 1);
              equal (list int) [ 0 ] (access t [ v 0 4 ] [ 0 ] 2))
            [ []; [ 0 ] ]);
      test "a partial write keeps the dependencies of the bytes it leaves"
        (fun () ->
          List.iter
            (fun writes ->
              let t = Hcq2.Deps.make () in
              ignore (access t [ b ] writes 0);
              equal (list int) [ 0 ] (access t [ v 4 8 ] [ 0 ] 1);
              equal (list int) [ 0 ] (access t [ v 0 4 ] [ 0 ] 2);
              equal (list int) [ 0 ] (access t [ v 12 4 ] [ 0 ] 3);
              equal (list int) [ 1 ] (access t [ v 4 8 ] [] 4))
            [ []; [ 0 ] ]);
      test "a write waits for every read since the last write" (fun () ->
          let t = Hcq2.Deps.make () in
          equal (list int) [] (access t [ b ] [] 0);
          equal (list int) [] (access t [ b ] [] 1);
          equal (list int) [ 0; 1 ] (access t [ b ] [ 0 ] 2);
          equal (list int) [ 2 ] (access t [ b ] [] 3));
      test "an access never waits for itself through an alias" (fun () ->
          List.iter
            (fun writes ->
              let t = Hcq2.Deps.make () in
              ignore (access t [ b ] [ 0 ] 0);
              equal (list int) [ 0 ] (access t [ b; v 4 8 ] writes 1);
              equal (list int) [ 1 ] (access t [ v 4 8 ] [ 0 ] 2))
            [ [ 0 ]; [ 1 ]; [ 0; 1 ] ]);
    ]

(* The law: Deps agrees with a model that remembers, for each byte of each lane,
   its last write and its reads since. *)

type access = { lane : int; start : int; stop : int; writes : bool }

let pp_access ppf a =
  Format.fprintf ppf "%s lane %d [%d, %d)"
    (if a.writes then "write" else "read")
    a.lane a.start a.stop

let access_gen =
  Gen.with_pp pp_access
    Gen.(
      let+ lane = int_range 0 1
      and+ start = int_range 0 23
      and+ len = int_range 1 8
      and+ writes = bool in
      { lane; start; stop = min 24 (start + len); writes })

let byte_model =
  prop "Deps agrees with a byte-by-byte model"
    Gen.(list ~size:(int_range 1 12) (list ~size:(int_range 1 3) access_gen))
    (fun steps ->
      let m =
        Ops.param ~shape:[ Int 24 ] ~device:(Multi [ "CPU:1"; "CPU:2" ]) 0 Uint8
      in
      let t = Hcq2.Deps.make () in
      let written = Array.make_matrix 2 24 None
      and read = Array.make_matrix 2 24 [] in
      List.iteri
        (fun x accesses ->
          let expected = ref [] in
          List.iter
            (fun a ->
              for i = a.start to a.stop - 1 do
                Option.iter
                  (fun w -> expected := w :: !expected)
                  written.(a.lane).(i);
                if a.writes then expected := read.(a.lane).(i) @ !expected
              done)
            accesses;
          let bufs =
            List.map
              (fun a ->
                Ops.shrink (Ops.mselect m a.lane)
                  [ Some (Int a.start, Int a.stop) ])
              accesses
          in
          let writes =
            List.concat
              (List.mapi (fun i a -> if a.writes then [ i ] else []) accesses)
          in
          let actual = Hcq2.Deps.access t bufs ~writes x in
          let others =
            List.filter (fun y -> y <> x) (List.sort_uniq compare !expected)
          in
          equal (list int)
            ~msg:(Printf.sprintf "access %d" x)
            others
            (List.sort_uniq compare actual);
          equal int ~msg:"each once"
            (List.length (List.sort_uniq compare actual))
            (List.length actual);
          List.iter
            (fun a ->
              for i = a.start to a.stop - 1 do
                if a.writes then begin
                  written.(a.lane).(i) <- Some x;
                  read.(a.lane).(i) <- []
                end
                else read.(a.lane).(i) <- x :: read.(a.lane).(i)
              done)
            accesses)
        steps)

(* Batches, run on a model of their devices *)

let devices_of call = List.concat_map placement (Realize.get_call_arg_uops call)

(* What every batch promises, whatever order its queues run in: no queue is left
   waiting, each device's signal word takes its value once, after every call on
   its memory, and none of them runs before the device's earlier work is
   done. *)
let well_formed batch =
  let calls = Batches.calls batch in
  let touches d i = List.mem d (devices_of (List.nth calls i)) in
  let devices = (info_of batch).device in
  List.iter
    (fun order ->
      let o = Batches.run ~order batch in
      List.iter
        (fun ((d, q), n) ->
          equal int ~msg:(Printf.sprintf "commands left on %s %s" d q) 0 n)
        o.left;
      List.iter
        (fun d ->
          let signals =
            List.filter (fun e -> e = Batches.Signaled d) o.events
          in
          equal int ~msg:(d ^ " signals once") 1 (List.length signals);
          equal int ~msg:(d ^ "'s signal word") (Batches.submitted + 1)
            (List.assoc d o.signal_words);
          let rec before = function
            | Batches.Signaled d' :: _ when d' = d -> ()
            | Call i :: rest ->
                ignore i;
                before rest
            | _ :: rest -> before rest
            | [] -> ()
          in
          before o.events;
          let rec after_signal seen = function
            | Batches.Signaled d' :: rest when d' = d -> after_signal true rest
            | Call i :: rest ->
                if seen && touches d i then
                  failf "call %d on %s runs after its signal" i d;
                after_signal seen rest
            | _ :: rest -> after_signal seen rest
            | [] -> ()
          in
          after_signal false o.events)
        devices)
    (Batches.rotations batch);
  List.iter
    (fun d ->
      let o = Batches.run ~finished:[ (d, Batches.submitted - 1) ] batch in
      List.iter
        (function
          | Batches.Call i when touches d i ->
              failf "call %d on %s runs before the work before it" i d
          | _ -> ())
        o.events)
    devices

let orders batch =
  List.sort_uniq compare
    (List.map
       (fun order ->
         List.filter_map
           (function Batches.Call i -> Some i | _ -> None)
           (Batches.run ~order batch).events)
       (Batches.rotations batch))

let checked calls =
  let bs = batches (sched calls) in
  List.iter well_formed bs;
  bs

let one calls =
  match checked calls with
  | [ b ] -> b
  | bs -> failf "one batch, not %d" (List.length bs)

let chain device n = List.init (n + 1) (fun _ -> storage device)

let chained_with adds bufs =
  List.init
    (List.length bufs - 1)
    (fun k -> adds (List.nth bufs (k + 1)) (List.nth bufs k))

let chained bufs = chained_with (fun out inp -> adds out inp) bufs
let kernel_chained bufs = chained_with (fun out inp -> kernel_adds out inp) bufs

let scheduling =
  group "sched_batches"
    [
      test "kernels on one queue run in order" (fun () ->
          equal
            (list (list int))
            [ [ 0; 1; 2 ] ]
            (orders (one (chained (chain "CPU:1" 3)))));
      test "a kernel runs after the copy that feeds it" (fun () ->
          let src = storage "CPU:1" and dst = storage "CPU:2" in
          equal
            (list (list int))
            [ [ 0; 1 ] ]
            (orders
               (one [ Ops.store_call dst src; adds (storage "CPU:2") dst ])));
      test "kernels of different devices do not wait for each other" (fun () ->
          let b =
            one
              [
                adds (storage "CPU:1") (storage "CPU:1");
                adds (storage "CPU:2") (storage "CPU:2");
              ]
          in
          equal (list (list int)) [ [ 0; 1 ]; [ 1; 0 ] ] (orders b);
          let o =
            Batches.run ~finished:[ ("CPU:1", Batches.submitted - 1) ] b
          in
          equal int ~msg:"calls that run" 1
            (List.length
               (List.filter
                  (function Batches.Call _ -> true | _ -> false)
                  o.events)));
      test "a call on a device without queues splits the batch" (fun () ->
          let a = storage "CPU:1" and h = storage "CPU" in
          let out =
            sched
              [
                adds (storage "CPU:1") a;
                adds (storage "CPU") h;
                adds (storage "CPU:1") a;
              ]
          in
          equal (list bool) [ true; false; true ]
            (List.map is_batch (Ops.src out));
          List.iter well_formed (batches out));
      test
        "consecutive calls of two kinds make a batch of each, in the order \
         they appear" (fun () ->
          let out =
            sched
              [
                adds (storage "NV:1") (storage "NV:1");
                adds (storage "CPU:1") (storage "CPU:1");
              ]
          in
          equal
            (list (list string))
            [ [ "NV:1" ]; [ "CPU:1" ] ]
            (List.map (fun b -> (info_of b).device) (batches out)));
      test
        "a program runs on its compute queue, a copy on its source's copy queue"
        (fun () ->
          let src = storage "CPU:1" and dst = storage "CPU:2" in
          let b = one [ Ops.store_call dst src; adds (storage "CPU:2") dst ] in
          equal
            (list (pair string string))
            [ ("CPU:1", "COPY:0"); ("CPU:2", "COMPUTE:0") ]
            (List.map fst (Batches.queues b)));
      test "copies between devices of three kinds are well formed" (fun () ->
          let a = storage "CPU:1"
          and b = storage "CPU:2"
          and c = storage "CPU:3"
          and d = storage "CPU:1" in
          ignore
            (checked
               [
                 Ops.store_call b a;
                 adds (storage "CPU:2") b;
                 Ops.store_call c b;
                 Ops.store_call d c;
                 adds (storage "CPU:1") d;
               ]));
      test
        "Metal's copies stay outside batches, since the host copies its memory"
        (fun () ->
          let out =
            sched
              [
                adds (storage "METAL:1") (storage "METAL:1");
                Ops.store_call (storage "METAL:2") (storage "METAL:1");
              ]
          in
          equal (list bool) [ true; false ] (List.map is_batch (Ops.src out)));
      test
        "on NV a compute queue that waits for another queue waits for its \
         previous call too" (fun () ->
          let waits_before_last kind =
            let a = storage (kind ^ ":1")
            and b = storage (kind ^ ":1")
            and c = storage (kind ^ ":1") in
            let batch =
              one
                [ adds c b; Ops.store_call a b; adds (storage (kind ^ ":1")) a ]
            in
            let compute =
              List.assoc (kind ^ ":1", "COMPUTE:0") (Batches.queues batch)
            in
            let rec count acc = function
              | c :: rest when instruction c = "call" ->
                  if List.exists (fun x -> instruction x = "call") rest then
                    count 0 rest
                  else acc
              | c :: rest ->
                  count (if instruction c = "wait" then acc + 1 else acc) rest
              | [] -> acc
            in
            count 0 compute
          in
          equal int ~msg:"NV" 2 (waits_before_last "NV");
          equal int ~msg:"another kind" 1 (waits_before_last "CPU"));
      test
        "AMD copies between peers take one queue, and one per peer with ALL2ALL"
        (fun () ->
          let copy_queues ~all2all =
            let src = storage "AMD:0" in
            let calls =
              List.map
                (fun d -> Ops.store_call (storage d) src)
                [ "AMD:1"; "AMD:2" ]
            in
            let out =
              Helpers.context
                [ B (Helpers.all2all, all2all) ]
                (fun () -> sched calls)
            in
            List.sort_uniq compare
              (List.map (fun ((_, q), _) -> q) (Batches.queues (the_batch out)))
            |> List.filter (fun q -> String.starts_with ~prefix:"COPY" q)
          in
          equal (list string) ~msg:"default" [ "COPY:0" ]
            (copy_queues ~all2all:0);
          equal (list string) ~msg:"ALL2ALL" [ "COPY:0"; "COPY:1" ]
            (copy_queues ~all2all:1));
      test "the fence orders every submission, and the submissions in turn"
        (fun () ->
          let src = storage "CPU:1" and dst = storage "CPU:2" in
          let batch =
            one [ Ops.store_call dst src; adds (storage "CPU:2") dst ]
          in
          let submissions = Ops.src (Ops.nth batch 0) in
          List.iteri
            (fun i s ->
              equal op ~msg:"ordered" After (Ops.op s);
              let deps = List.tl (Ops.src s) in
              is_true ~msg:"after the fence"
                (List.exists
                   (fun d ->
                     Ops.op d = Custom_function
                     && Ops.arg d = String "hcq_fence")
                   deps);
              if i > 0 then
                is_true ~msg:"after the one before"
                  (List.exists
                     (fun d -> d == List.nth submissions (i - 1))
                     deps))
            submissions);
      test "names each submission after its kind and queue" (fun () ->
          let src = storage "CPU:1" and dst = storage "CPU:2" in
          let batch =
            one [ Ops.store_call dst src; adds (storage "CPU:2") dst ]
          in
          equal (list string)
            [ "submit_cpu_copy"; "submit_cpu_compute" ]
            (List.map
               (fun s ->
                 match Ops.arg (Ops.without_after s) with
                 | String n -> n
                 | _ -> "")
               (Ops.src (Ops.nth batch 0))));
      test
        "a batch holds its kernels' names, costs, written storage and total \
         cost" (fun () ->
          let a = storage "CPU:1"
          and b = storage "CPU:1"
          and c = storage "CPU:2" in
          let calls = [ adds b a; Ops.store_call c b ] in
          let info = info_of (plain (fun () -> one calls)) in
          equal (list string)
            [ "k"; "copy       16 B,   CPU:2 <- CPU:1  " ]
            (List.map (fun (k : Ops.hcq_kernel) -> k.name) info.kernels);
          equal (list uop) [ b; c ] info.written_bufs;
          equal int ~msg:"lds" 48
            (match info.estimates.lds with Int n -> n | Sym _ -> -1));
    ]

(* Stamp slots follow Submission.record *)

let stamps =
  group "stamp slots"
    [
      test
        "a device's slots are one per queue, then two per call when profiling"
        (fun () ->
          let calls = chained (chain "CPU:1" 2) in
          let slot_sizes profile =
            List.map
              (fun p -> (param_arg p).size)
              (List.filter
                 (fun n -> tag_of n = "slots")
                 (Ops.toposort (sched ~profile calls)))
          in
          equal
            (list (option int))
            ~msg:"profiling"
            [ Some (2 * (1 + (2 * 2))) ]
            (slot_sizes true);
          equal
            (list (option int))
            ~msg:"not profiling" [ Some 2 ] (slot_sizes false));
      test
        "a kernel's stamps are the second word of its two slots after the \
         queues'" (fun () ->
          let info =
            info_of
              (the_batch (sched ~profile:true (chained (chain "CPU:1" 2))))
          in
          equal
            (list (list int))
            [ [ 3; 5 ]; [ 7; 9 ] ]
            (List.map (fun (k : Ops.hcq_kernel) -> k.stamps) info.kernels));
      test
        "profiling passes each device's slots to the batch, and timestamps \
         each call" (fun () ->
          let batch =
            the_batch (sched ~profile:true (chained (chain "CPU:1" 2)))
          in
          equal (list string) [ "slots" ]
            (List.map tag_of (List.tl (Ops.src batch)));
          let cmds = List.assoc ("CPU:1", "COMPUTE:0") (Batches.queues batch) in
          equal int ~msg:"timestamps" 4
            (List.length
               (List.filter (fun c -> instruction c = "timestamp") cmds)));
      test "no stamps without profiling" (fun () ->
          let info = info_of (the_batch (sched (chained (chain "CPU:1" 2)))) in
          equal
            (list (list int))
            [ []; [] ]
            (List.map (fun (k : Ops.hcq_kernel) -> k.stamps) info.kernels));
    ]

(* Lowering *)

let lowering =
  group "lower_call"
    [
      test "raises Invalid_argument for a call that is no batch" (fun () ->
          rejects (fun () ->
              Hcq2.lower_call ~devices:(kinds ())
                (adds (storage "CPU:1") (storage "CPU:1"))));
      test "raises Invalid_argument for a batch lowered already" (fun () ->
          let lowered =
            Hcq2.lower_call ~devices:(kinds ())
              (the_batch (sched (chained (chain "CPU:1" 1))))
          in
          rejects (fun () ->
              Hcq2.lower_call ~devices:(kinds ()) (Ops.without_after lowered)));
      test "its information counts its arguments and places each device's slots"
        (fun () ->
          let lowered =
            Hcq2.lower_call ~devices:(kinds ())
              (the_batch (sched ~profile:true (chained (chain "CPU:1" 2))))
          in
          let call = Ops.without_after lowered in
          let info = info_of call in
          let args = List.tl (Ops.src call) in
          equal int ~msg:"nargs" (List.length args) info.nargs;
          equal
            (list (pair string string))
            [ ("CPU:1", "slots") ]
            (List.map (fun (d, i) -> (d, tag_of (List.nth args i))) info.slots));
      test "merges the placeholders of one tag and device into one argument"
        (fun () ->
          let src = storage "CPU:1" and dst = storage "CPU:1" in
          let lowered =
            Hcq2.lower_call ~devices:(kinds ())
              (the_batch
                 (sched [ Ops.store_call dst src; adds (storage "CPU:1") dst ]))
          in
          let keys =
            List.map
              (fun a -> (tag_of a, placement a))
              (List.tl (Ops.src (Ops.without_after lowered)))
          in
          equal int ~msg:"distinct"
            (List.length (List.sort_uniq compare keys))
            (List.length keys));
      test "its link patches read no variable, no load and no register"
        (fun () ->
          let lowered =
            Hcq2.lower_call ~devices:(kinds ())
              (the_batch (sched (chained (chain "CPU:1" 2))))
          in
          let patches = List.tl (Ops.src lowered) in
          is_true ~msg:"some patches" (patches <> []);
          List.iter
            (fun p ->
              equal op ~msg:"a store" Store (Ops.op p);
              List.iter
                (fun n ->
                  is_false ~msg:"a variable" (Ops.is_variable n);
                  is_false ~msg:"a load" (Ops.op n = Load))
                (Ops.toposort p))
            patches);
      test "an input's address is loaded from the address table on each run"
        (fun () ->
          let a =
            Ops.param ~shape:[ Int 4 ] ~device:(Single "CPU:1") 0 Float32
          in
          let b =
            Ops.param ~shape:[ Int 4 ] ~device:(Single "CPU:1") 1 Float32
          in
          let lowered =
            Hcq2.lower_call ~devices:(kinds ()) (the_batch (sched [ adds b a ]))
          in
          let call = Ops.without_after lowered in
          let info = info_of call in
          is_true ~msg:"a table" (info.table >= 0);
          equal string "inputs"
            (tag_of (List.nth (List.tl (Ops.src call)) info.table));
          equal
            (list (triple uop int string))
            [ (b, 0, "CPU:1"); (a, 0, "CPU:1") ]
            (List.sort (fun (x, _, _) (y, _, _) -> Ops.compare y x) info.inputs));
    ]

(* Compiling *)

let compiling =
  group "compile_linear"
    [
      test "returns a linear holding a lowered batch as it is" (fun () ->
          let compiled =
            Hcq2.compile_linear ~devices:(recorded_devices ())
              (linear (chained (chain "CPU:1" 1)))
          in
          is_true
            (Hcq2.compile_linear ~devices:(recorded_devices ()) compiled
            == compiled));
      test "makes each kernel ask for a beam of the width BEAM sets" (fun () ->
          let widths = ref [] in
          let search w s =
            widths := w :: !widths;
            s
          in
          (* A kernel no other call compiles, even in a rerun: programs are
             kept. *)
          let fresh = float_of_int (Ops.unique_num ()) in
          let calls =
            [
              Ops.call
                (map_kernel (Single "CPU:1") 4 (plus fresh))
                [ storage "CPU:1"; storage "CPU:1" ];
            ]
          in
          ignore
            (Helpers.context
               [ B (Helpers.beam, 1) ]
               (fun () ->
                 Hcq2.compile_linear ~search ~devices:(recorded_devices ())
                   (linear calls)));
          equal (list int) [ 1 ] !widths);
      test "profiles when DEBUG is 2 or more" (fun () ->
          let calls = chained (chain "CPU:1" 1) in
          let slots debug =
            let c =
              Helpers.context
                [ B (Helpers.debug, debug) ]
                (fun () ->
                  Hcq2.compile_linear ~devices:(recorded_devices ())
                    (linear calls))
            in
            List.exists
              (fun (k : Ops.hcq_kernel) -> k.stamps <> [])
              (info_of (the_batch c)).kernels
          in
          is_false ~msg:"DEBUG=0" (slots 0);
          is_true ~msg:"DEBUG=2" (slots 2));
      test "runs a call on sharded buffers once per device, each on its shard"
        (fun () ->
          let devices = Ops.Multi [ "CPU:1"; "CPU:2" ] in
          let sharded () = Ops.new_buffer devices 4 Float32 in
          let call =
            Ops.call (map_kernel devices 4 (plus 1.)) [ sharded (); sharded () ]
          in
          let info =
            info_of
              (the_batch
                 (Hcq2.compile_linear ~devices:(recorded_devices ())
                    (linear [ call ])))
          in
          equal
            (list (list string))
            [ [ "CPU:1" ]; [ "CPU:2" ] ]
            (List.map (fun (k : Ops.hcq_kernel) -> k.devices) info.kernels));
      test
        "copies through the halves of a staging buffer of the host where the \
         queues cannot reach" (fun () ->
          let events = Null_queue.events () in
          let devices = function
            | "CPU" -> { Hcq2.target = recorded_target; queues = None }
            | _ ->
                let queues =
                  {
                    Hcq2.commands = Null_queue.commands events;
                    copy_queue = true;
                    submission = Buffered;
                    host = "CPU";
                    reaches = (fun d -> d <> "CPU:2");
                  }
                in
                { Hcq2.target = recorded_target; queues = Some queues }
          in
          let big = 48 * 1024 * 1024 in
          let src = Ops.new_buffer (Single "CPU:1") big Float32
          and dst = Ops.new_buffer (Single "CPU:2") big Float32 in
          let compiled =
            Hcq2.compile_linear ~devices (linear [ Ops.store_call dst src ])
          in
          let staging =
            List.filter (fun n -> tag_of n = "staging") (Ops.toposort compiled)
          in
          equal
            (list (pair (option int) (list string)))
            [ (Some (128 * 1024 * 1024), [ "CPU" ]) ]
            (List.sort_uniq compare
               (List.map (fun s -> ((param_arg s).size, placement s)) staging));
          equal int ~msg:"copies" 6
            (List.length (info_of (the_batch compiled)).kernels));
      test "a copy on a device without copy queues is a kernel" (fun () ->
          let events = Null_queue.events () in
          ignore events;
          let compiled =
            Hcq2.compile_linear
              ~devices:(recorded_devices ~copy_queue:false ())
              (linear [ Ops.store_call (storage "CPU:2") (storage "CPU:1") ])
          in
          let info = info_of (the_batch compiled) in
          is_false ~msg:"a copy"
            (List.exists
               (fun (k : Ops.hcq_kernel) ->
                 String.starts_with ~prefix:"copy" k.name)
               info.kernels));
    ]

(* Linking and running on the NULL device *)

let nx name = if name = "CPU" then Nx_device.host else Null_device.device name
let floats = array float_exact

let host_view b =
  if Nx_device.equal (Nx_device.Buffer.device b) Nx_device.host then b
  else Result.get_ok (Nx_device.Buffer.borrow Nx_device.host b)

let floats_of b =
  let a = Nx_device.Buffer.bigarray Bigarray.float32 (host_view b) in
  Array.init (Bigarray.Array1.dim a) (fun i -> a.{i})

let new_floats name xs =
  let b = Nx_device.Buffer.create (nx name) Float32 (Array.length xs) in
  let a = Nx_device.Buffer.bigarray Bigarray.float32 (host_view b) in
  Array.iteri (fun i x -> a.{i} <- x) xs;
  b

(* The storage of a schedule, in order. *)
let storage_of calls =
  List.sort_uniq Ops.compare
    (List.concat_map
       (fun c -> List.filter (fun n -> Ops.op n = Buffer) (Ops.toposort c))
       calls)

(* Buffers for each storage node of [calls], one per device of its placement,
   the [i]th holding 10i, 10i + 1, ... *)
let bound_storage calls =
  List.mapi
    (fun i u ->
      let n = Option.get (param_arg u).size in
      ( u,
        List.map
          (fun d ->
            new_floats d (Array.init n (fun k -> float_of_int ((100 * i) + k))))
          (placement u) ))
    (storage_of calls)

let run_calls ?(devices = Null_device.devices ()) ?profile ?(vars = [])
    ?(slots = [||]) ~bound calls =
  let compiled =
    Hcq2.compile_linear ?profile
      ~devices:(fun n -> (devices n).compiler)
      (linear calls)
  in
  let s = Tolk_engine.link ~devices ~bound compiled in
  Tolk_engine.run ~vars s slots;
  Null_device.synchronize ();
  s

(* The NULL devices as devices without queues, whose calls run one by one. *)
let one_by_one name =
  let d = Null_device.devices () name in
  { d with compiler = { d.compiler with queues = None } }

(* Each storage node's values after running [calls] batched on the NULL device's
   queues, and one by one, from the same values. *)
let agrees ?(heavy = false) ?(latency = 0.) name calls =
  (if heavy then slow else test) name (fun () ->
      let results devices =
        let bound = bound_storage calls in
        ignore
          (Null_device.with_latency latency (fun () ->
               run_calls ~devices ~bound calls));
        List.concat_map (fun (_, bs) -> List.map floats_of bs) bound
      in
      equal (list floats) (results one_by_one)
        (results (Null_device.devices ())))

let params name n =
  List.init n (fun slot ->
      Ops.param ~shape:[ Int 4 ] ~device:(Single name) slot Float32)

let running =
  group "linking and running"
    [
      test "a chain of kernels computes what the interpreter says" (fun () ->
          let b = chain "CPU:1" 3 in
          let calls = kernel_chained b in
          let bound =
            List.map
              (fun u -> (u, [ new_floats "CPU:1" [| 1.; 2.; 3.; 4. |] ]))
              b
          in
          ignore (run_calls ~bound calls);
          let slot u = match Ops.arg u with Param p -> p.slot | _ -> -1 in
          let interpreted =
            Kernel_graphs.linear_writes
              ~buffers:
                [
                  ( slot (List.hd b),
                    Array.map (fun x -> `Float x) [| 1.; 2.; 3.; 4. |] );
                ]
              (linear calls)
          in
          List.iter
            (fun out ->
              let expected =
                List.filter_map
                  (fun (s, _, v) ->
                    match v with
                    | `Float x when s = slot out -> Some x
                    | _ -> None)
                  interpreted
              in
              equal floats (Array.of_list expected)
                (floats_of (List.hd (List.assq out bound))))
            (List.tl b));
      agrees
        "a copy between devices and the kernel it feeds agree with running \
         them one by one"
        (let src = storage "CPU:1" and dst = storage "CPU:2" in
         [ Ops.store_call dst src; kernel_adds (storage "CPU:2") dst ]);
      agrees ~heavy:true
        "copies across three devices agree with running them one by one"
        (let a = storage "CPU:1"
         and b = storage "CPU:2"
         and c = storage "CPU:3" in
         [
           Ops.store_call b a;
           kernel_adds ~c:2. c b;
           Ops.store_call (storage "CPU:1") c;
         ]);
      agrees "kernels on two devices at once agree with running them one by one"
        [
          kernel_adds (storage "CPU:1") (storage "CPU:1");
          kernel_adds ~c:5. (storage "CPU:2") (storage "CPU:2");
        ];
      (* The queues run late, so that a host kernel that did not wait for them
         would read what they have not written yet. *)
      agrees ~latency:0.02
        "a batch split by a host kernel agrees with running them one by one"
        (let a = storage "CPU:1"
         and h = storage "CPU"
         and b = storage "CPU:1" in
         [
           kernel_adds a (storage "CPU:1");
           Ops.store_call h a;
           kernel_adds ~c:3. (storage "CPU") h;
           Ops.store_call b h;
           kernel_adds (storage "CPU:1") b;
         ]);
      test "a copy without copy queues runs as a kernel, and copies" (fun () ->
          let src = storage "CPU:1" and dst = storage "CPU:2" in
          let bound =
            [
              (src, [ new_floats "CPU:1" [| 5.; 6.; 7.; 8. |] ]);
              (dst, [ new_floats "CPU:2" [| 0.; 0.; 0.; 0. |] ]);
            ]
          in
          ignore
            (run_calls
               ~devices:(Null_device.devices ~copy_queue:false ())
               ~bound
               [ Ops.store_call dst src ]);
          equal floats [| 5.; 6.; 7.; 8. |]
            (floats_of (List.hd (List.assq dst bound))));
      test "a sharded kernel computes each shard on its device" (fun () ->
          let devices = Ops.Multi [ "CPU:1"; "CPU:2" ] in
          let a = Ops.new_buffer devices 4 Float32
          and b = Ops.new_buffer devices 4 Float32 in
          let shards x =
            List.map
              (fun d -> new_floats d (Array.make 4 x))
              [ "CPU:1"; "CPU:2" ]
          in
          let bound =
            [
              ( a,
                [
                  new_floats "CPU:1" (Array.make 4 0.);
                  new_floats "CPU:2" (Array.make 4 1.);
                ] );
              (b, shards 0.);
            ]
          in
          ignore
            (run_calls ~bound
               [ Ops.call (map_kernel devices 4 (plus 1.)) [ b; a ] ]);
          equal (list floats)
            [ Array.make 4 1.; Array.make 4 2. ]
            (List.map floats_of (List.assq b bound)));
      test
        "a kernel on buffers stacked from two devices computes each on its \
         device" (fun () ->
          let devices = Ops.Multi [ "CPU:1"; "CPU:2" ] in
          let one d = Ops.new_buffer (Single d) 4 Float32 in
          let a1 = one "CPU:1"
          and a2 = one "CPU:2"
          and b1 = one "CPU:1"
          and b2 = one "CPU:2" in
          let bound =
            [
              (a1, [ new_floats "CPU:1" (Array.make 4 3.) ]);
              (a2, [ new_floats "CPU:2" (Array.make 4 5.) ]);
              (b1, [ new_floats "CPU:1" (Array.make 4 0.) ]);
              (b2, [ new_floats "CPU:2" (Array.make 4 0.) ]);
            ]
          in
          ignore
            (run_calls ~bound
               [
                 Ops.call
                   (map_kernel devices 4 (plus 1.))
                   [ Ops.mstack b1 [ b2 ]; Ops.mstack a1 [ a2 ] ];
               ]);
          equal (list floats)
            [ Array.make 4 4.; Array.make 4 6. ]
            (List.map
               (fun u -> floats_of (List.hd (List.assq u bound)))
               [ b1; b2 ]));
      test "a schedule's variables reach its kernels" (fun () ->
          let v =
            Ops.variable ~dtype:Int32 "v" (`Int Bigint.one)
              (`Int (Bigint.of_int 4))
          in
          let out = param (Single "CPU:1") 0 4 in
          let i = Ops.range (Sym v) [ 0 ] in
          let kernel =
            Ops.sink
              ~kernel:(Ops.kernel_info ~name:"up_to_v" ())
              [
                Ops.end_
                  (Ops.store (Ops.index out [ i ])
                     (Ops.float ~dtype:Float32 9.))
                  [ i ];
              ]
          in
          let o = storage "CPU:1" in
          let bound = [ (o, [ new_floats "CPU:1" (Array.make 4 0.) ]) ] in
          ignore (run_calls ~vars:[ ("v", 3) ] ~bound [ Ops.call kernel [ o ] ]);
          equal floats [| 9.; 9.; 9.; 0. |]
            (floats_of (List.hd (List.assq o bound))));
      test "a link serves any buffers bound to its inputs" (fun () ->
          let a, b =
            match params "CPU:1" 2 with [ a; b ] -> (a, b) | _ -> assert false
          in
          let devices = Null_device.devices () in
          let compiled =
            Hcq2.compile_linear
              ~devices:(fun n -> (devices n).compiler)
              (linear [ kernel_adds ~c:10. b a ])
          in
          let s = Tolk_engine.link ~devices compiled in
          List.iter
            (fun x ->
              let src = new_floats "CPU:1" (Array.make 4 x)
              and dst = new_floats "CPU:1" (Array.make 4 0.) in
              Tolk_engine.run s [| [ src ]; [ dst ] |];
              Null_device.synchronize ();
              equal floats ~msg:(string_of_float x)
                (Array.make 4 (x +. 10.))
                (floats_of dst))
            [ 1.; 2.; 3. ]);
      test
        "a run waits for its batch's previous run before it rewrites the \
         batch's memory" (fun () ->
          let a, b =
            match params "CPU:1" 2 with [ a; b ] -> (a, b) | _ -> assert false
          in
          let devices = Null_device.devices () in
          let compiled =
            Hcq2.compile_linear
              ~devices:(fun n -> (devices n).compiler)
              (linear [ kernel_adds ~c:10. b a ])
          in
          let s = Tolk_engine.link ~devices compiled in
          let runs =
            List.map
              (fun x ->
                ( x,
                  new_floats "CPU:1" (Array.make 4 x),
                  new_floats "CPU:1" (Array.make 4 0.) ))
              [ 1.; 2.; 3. ]
          in
          Null_device.with_latency 0.02 (fun () ->
              List.iter
                (fun (_, src, dst) -> Tolk_engine.run s [| [ src ]; [ dst ] |])
                runs);
          Null_device.synchronize ();
          List.iter
            (fun (x, _, dst) ->
              equal floats ~msg:(string_of_float x)
                (Array.make 4 (x +. 10.))
                (floats_of dst))
            runs);
      test "a run allocates no device memory" (fun () ->
          let b = chain "CPU:1" 2 in
          let bound =
            List.map (fun u -> (u, [ new_floats "CPU:1" (Array.make 4 0.) ])) b
          in
          let s = run_calls ~bound (kernel_chained b) in
          let allocated () =
            List.map
              (fun d -> Nx_device.Stats.allocated (Nx_device.stats (nx d)))
              [ "CPU"; "CPU:1" ]
          in
          let before = allocated () in
          Tolk_engine.run s [||];
          Null_device.synchronize ();
          equal (list int) before (allocated ()));
      test "a profile records a span of each kernel on its device, in order"
        (fun () ->
          let b = chain "CPU:1" 2 in
          let bound =
            List.map (fun u -> (u, [ new_floats "CPU:1" (Array.make 4 0.) ])) b
          in
          let p = Nx_device.Profile.start () in
          let events =
            Fun.protect
              ~finally:(fun () ->
                if Nx_device.Profile.enabled () then
                  ignore (Nx_device.Profile.stop p))
              (fun () ->
                ignore (run_calls ~profile:true ~bound (kernel_chained b));
                Nx_device.Profile.stop p)
          in
          let spans =
            List.filter_map
              (function
                | Nx_device.Profile.Span { device; name; start; stop; _ }
                  when Nx_device.equal device (nx "CPU:1") ->
                    Some (name, start, stop)
                | _ -> None)
              events
          in
          equal (list string) [ "k"; "k" ] (List.map (fun (n, _, _) -> n) spans);
          List.iter
            (fun (_, start, stop) -> at_least int ~than:start stop)
            spans;
          match spans with
          | [ (_, _, first_stop); (_, second_start, _) ] ->
              at_least int ~than:first_stop second_start
          | _ -> ());
    ]

(* Host functions and C structures *)

(* Runs the batch of [effects] on CPU:1, which signals its value itself, with
   the placeholders tagged as in [outputs] held in their buffers. *)
let host_batch ?(outputs = []) effects =
  let d = "CPU:1" in
  let signal =
    Ops.store (Ops.index (Hcq2.signal_word d) [ Ops.int 0 ]) (Hcq2.value d)
  in
  let batch =
    Ops.call ~aux:(batch_info [ d ])
      (Ops.sink
         ~kernel:(Ops.kernel_info ~name:"test" ())
         (effects @ [ signal ]))
      []
  in
  let null = Null_device.devices () in
  let devices name =
    let dev = null name in
    let placeholder u =
      match List.assoc_opt (tag_of u) outputs with
      | Some b -> Some b
      | None -> dev.placeholder u
    in
    { dev with placeholder }
  in
  let lowered =
    Hcq2.lower_call ~devices:(fun n -> (devices n).compiler) batch
  in
  let compiled =
    Realize.lower_and_compile
      ~targets:(fun n -> (devices n).compiler.target)
      (linear [ lowered ])
  in
  Tolk_engine.run (Tolk_engine.link ~devices compiled) [||];
  Null_device.synchronize ()

let struct_t : Hcq2.c_struct =
  {
    struct_name = "fields";
    struct_size = 16;
    fields = [ ("u8", 0, 1); ("u16", 2, 2); ("u32", 4, 4); ("u64", 8, 8) ];
  }

let words_of b =
  let a = Nx_device.Buffer.bigarray Bigarray.int8_unsigned (host_view b) in
  let byte i = Bigint.of_int a.{i} in
  fun off size ->
    List.fold_left
      (fun acc k -> Bigint.add (Bigint.shift_left acc 8) (byte (off + k)))
      Bigint.zero
      (List.init size (fun k -> size - 1 - k))

(* The same word in hexadecimal, two digits a byte. *)
let hex_of b =
  let a = Nx_device.Buffer.bigarray Bigarray.int8_unsigned (host_view b) in
  fun off size ->
    String.concat ""
      (List.init size (fun k -> Printf.sprintf "%02x" a.{off + size - 1 - k}))

let output () = Nx_device.Buffer.create (nx "CPU:1") UInt8 16

let host_functions =
  group "ccall, cstruct and cfield"
    [
      test "a host program calls a C function and keeps its result" (fun () ->
          let out =
            Ops.placeholder ~device:(Single "CPU:1") ~volatile:true
              ~tag:(String "result") [ 1 ] Int32
          in
          let ffs =
            Hcq2.ccall ~host:"CPU:1" ~lib:"libc" ~ret:Int32 "ffs"
              [ Ops.int ~dtype:Int32 0x10 ]
          in
          let b = output () in
          host_batch
            ~outputs:[ ("result", b) ]
            [ Ops.store (Ops.index out [ Ops.int 0 ]) ffs ];
          equal string "5" (Bigint.to_string (words_of b 0 4)));
      test
        "a C structure holds each field it is given, as its size's unsigned \
         integer" (fun () ->
          let s =
            Hcq2.cstruct ~host:"CPU:1" struct_t
              [
                ("u8", u8 0x12);
                ("u16", u16 0x3456);
                ("u32", u32 0x789ABCDE);
                ( "u64",
                  Ops.const ~dtype:Uint64
                    (`Int (Bigint.of_string "0xFEDCBA9876543210")) );
              ]
          in
          let out =
            Ops.placeholder ~device:(Single "CPU:1") ~tag:(String "copied")
              [ 16 ] Uint8
          in
          let copy =
            Hcq2.ccall ~host:"CPU:1" ~lib:"libc" "memcpy"
              [
                Ops.getaddr ~device:"CPU:1" out;
                Ops.getaddr ~device:"CPU:1" s;
                u64 16;
              ]
          in
          let b = output () in
          host_batch ~outputs:[ ("copied", b) ] [ copy ];
          equal (list string)
            [ "12"; "3456"; "789abcde"; "fedcba9876543210" ]
            (List.map (fun (_, off, size) -> hex_of b off size) struct_t.fields));
      test "cfield reads a field of a structure" (fun () ->
          let s = Hcq2.cstruct ~host:"CPU:1" struct_t [ ("u32", u32 42) ] in
          let out =
            Ops.placeholder ~device:(Single "CPU:1") ~volatile:true
              ~tag:(String "field") [ 1 ] Uint32
          in
          let b = output () in
          host_batch
            ~outputs:[ ("field", b) ]
            [
              Ops.store
                (Ops.index out [ Ops.int 0 ])
                (Ops.load (Hcq2.cfield s struct_t "u32") []);
            ];
          equal string "42" (Bigint.to_string (words_of b 0 4)));
      test
        "cstruct and cfield raise Invalid_argument for a field the layout lacks"
        (fun () ->
          rejects (fun () ->
              Hcq2.cstruct ~host:"CPU:1" struct_t [ ("nope", u8 1) ]);
          rejects (fun () ->
              Hcq2.cfield
                (Hcq2.cstruct ~host:"CPU:1" struct_t [])
                struct_t "nope"));
    ]

(* Command queues *)

(* The devices of kinds (), whose compute queues' exec first runs [probe] on the
   queue. *)
let probing probe =
  let base = kinds () in
  fun name ->
    let d = base name in
    match d.queues with
    | None -> d
    | Some qs ->
        let commands q =
          let c = qs.commands q in
          {
            c with
            exec =
              (fun call prg ->
                probe q;
                c.exec call prg);
          }
        in
        { d with queues = Some { qs with commands } }

let lower_with devices calls =
  Hcq2.lower_call ~devices
    (the_batch (Hcq2.sched_batches ~devices ~profile:false (linear calls)))

let queues =
  group "Queue"
    [
      test "dword is a word's low 32 bits, as a uint32" (fun () ->
          equal uop
            (Ops.const ~dtype:Uint32 (`Int (Bigint.of_int 0x23456789)))
            (Hcq2.Queue.dword 0x1_2345_6789));
      test
        "q appends constants at their width, bytes as they are, and room for \
         other words" (fun () ->
          let seen = ref None in
          let probe q =
            let before = Hcq2.Queue.size q in
            let after =
              Hcq2.Queue.q q
                [
                  Hcq2.Queue.dword 0x1_2345_6789;
                  u16 0xABCD;
                  Ops.v Binary ~arg:(Bytes "xy");
                  Hcq2.submitted "CPU:1";
                ]
            in
            seen :=
              Some
                ( before,
                  after,
                  Hcq2.Queue.get_dword q before,
                  Hcq2.Queue.get_dword q (before + 4),
                  Hcq2.Queue.get_dword q (before + 8),
                  Hcq2.Queue.name q,
                  Hcq2.Queue.devices q )
          in
          ignore
            (lower_with (probing probe)
               [ adds (storage "CPU:1") (storage "CPU:1") ]);
          match !seen with
          | Some (before, after, w0, w1, w2, name, devices) ->
              equal int ~msg:"size" (before + 4 + 2 + 2 + 8) after;
              equal int ~msg:"the dword" 0x23456789 w0;
              equal int ~msg:"the uint16 and the bytes" 0x7978ABCD w1;
              equal int ~msg:"room for the variable" 0 w2;
              equal string ~msg:"name" "COMPUTE:0" name;
              equal (list string) ~msg:"devices" [ "CPU:1" ] devices
          | None -> fail "the queue was never encoded");
      test "q grows a queue past any size" (fun () ->
          let seen = ref [] in
          let probe q =
            let before = Hcq2.Queue.size q in
            let big = String.make (1 lsl 17) '\x01' in
            let after =
              Hcq2.Queue.q q
                [ Ops.v Binary ~arg:(Bytes big); Hcq2.Queue.dword 9 ]
            in
            seen :=
              [
                after - before;
                Hcq2.Queue.get_dword q (before + (1 lsl 17) - 4);
                Hcq2.Queue.get_dword q (before + (1 lsl 17));
              ]
          in
          ignore
            (lower_with (probing probe)
               [ adds (storage "CPU:1") (storage "CPU:1") ]);
          equal (list int) [ (1 lsl 17) + 4; 0x01010101; 9 ] !seen);
      test "set_dword rewrites a word a constant wrote" (fun () ->
          let seen = ref [] in
          let probe q =
            let at = Hcq2.Queue.size q in
            ignore (Hcq2.Queue.q q [ Hcq2.Queue.dword 5 ]);
            Hcq2.Queue.set_dword q at 7;
            seen := [ Hcq2.Queue.get_dword q at ]
          in
          ignore
            (lower_with (probing probe)
               [ adds (storage "CPU:1") (storage "CPU:1") ]);
          equal (list int) [ 7 ] !seen);
      test
        "loop moves each word by its trip, a uint64 as two dwords and a uint16 \
         as itself" (fun () ->
          let q = Hcq2.Queue.v ~devices:[ "CPU:1" ] "COMPUTE:0" in
          let r = Ops.range (Int 3) [ Ops.unique_num () ] in
          Hcq2.Queue.loop q r (fun () ->
              ignore
                (Hcq2.Queue.q q [ Ops.cast r Uint64; Ops.cast r Uint16; u16 0 ]));
          let words = Hcq2.Queue.words q in
          equal
            (list (Testable.make ~pp:Dtype.pp ~equal:Dtype.equal))
            ~msg:"types" [ Uint32; Uint32; Uint16 ]
            (List.map (fun (_, w) -> Ops.dtype w) words);
          is_true ~msg:"each offset moves with the trip"
            (List.for_all (fun (o, _) -> Ops.Nodes.mem r (Ops.ranges o)) words);
          equal int ~msg:"bytes" 36 (Hcq2.Queue.size q));
      test "reset empties a queue" (fun () ->
          let seen = ref [] in
          let probe q =
            Hcq2.Queue.reset q;
            seen := [ Hcq2.Queue.size q ]
          in
          ignore
            (lower_with (probing probe)
               [ adds (storage "CPU:1") (storage "CPU:1") ]);
          equal (list int) [ 0 ] !seen);
    ]

(* Words and patches, run *)

let u64_of b off =
  let a = Nx_device.Buffer.bigarray Bigarray.int8_unsigned (host_view b) in
  List.fold_left
    (fun acc k -> (acc lsl 8) lor a.{off + k})
    0
    (List.init 8 (fun k -> 7 - k))

let word_tests =
  group "patch and bufferize_cmdbuf"
    [
      test "patch writes its blob, then its rows, known at link or when run"
        (fun () ->
          let d = "CPU:1" in
          let buf =
            Ops.placeholder ~device:(Single d) ~tag:(String "patched") [ 32 ]
              Uint8
          in
          let rows =
            [
              (Ops.int 0, u32 1);
              (Ops.int 4, u32 2);
              (Ops.int 8, Ops.cast (Hcq2.value d) Uint32);
            ]
          in
          let patched = Hcq2.patch ~blob:(String.make 32 '\xaa') buf rows in
          let b = Nx_device.Buffer.create (nx d) UInt8 32 in
          let value = Nx_device.submitted (nx d) + 1 in
          host_batch ~outputs:[ ("patched", b) ] (List.tl (Ops.src patched));
          let word = words_of b in
          equal (list string)
            [ "1"; "2"; string_of_int (value land 0xFFFFFFFF) ]
            (List.map (fun off -> Bigint.to_string (word off 4)) [ 0; 4; 8 ]);
          equal string ~msg:"the blob elsewhere"
            (String.concat "" (List.init 20 (fun _ -> "aa")))
            (hex_of b 12 20));
      test
        "a row known at link is written at link, the others by the host program"
        (fun () ->
          let d = "CPU:1" in
          let buf =
            Ops.placeholder ~device:(Single d) ~tag:(String "patched") [ 32 ]
              Uint8
          in
          let patched =
            Hcq2.patch buf
              [
                (Ops.int 0, u32 1); (Ops.int 8, Ops.cast (Hcq2.value d) Uint32);
              ]
          in
          let out =
            Ops.placeholder ~device:(Single d) ~tag:(String "out") [ 1 ] Uint32
          in
          let read_back =
            Ops.store
              (Ops.index out [ Ops.int 0 ])
              (Ops.load
                 (Ops.index (Ops.bitcast patched Uint32) [ Ops.int 0 ])
                 [])
          in
          let signal =
            Ops.store
              (Ops.index (Hcq2.signal_word d) [ Ops.int 0 ])
              (Hcq2.value d)
          in
          let batch =
            Ops.call ~aux:(batch_info [ d ])
              (Ops.sink
                 ~kernel:(Ops.kernel_info ~name:"test" ())
                 [ read_back; signal ])
              []
          in
          let lowered = Hcq2.lower_call ~devices:(kinds ()) batch in
          let stores =
            List.filter
              (fun st -> Ops.op st = Store)
              (List.tl (Ops.src lowered))
          in
          let link_words =
            List.concat_map (fun st -> Ops.toposort (Ops.nth st 1)) stores
          in
          is_true ~msg:"the constant row at link"
            (List.exists (fun n -> n == u32 1) link_words);
          let host = Ops.nth (Ops.without_after lowered) 0 in
          is_true ~msg:"the value row in the host program"
            (List.exists
               (fun n ->
                 Ops.op n = Param
                 && Ops.is_variable n
                 && Ops.expr n = Ops.expr (Hcq2.value d))
               (Ops.toposort host)));
      test "each region starts at its own alignment in the buffer of its name"
        (fun () ->
          let d = "CPU:1" in
          let region align c =
            Ops.v Linear
              ~src:[ Ops.v Binary ~arg:(Bytes (String.make 8 c)) ]
              ~arg:(Region { name = "r"; align })
          in
          let null = Null_device.devices () in
          let devices name =
            let dev = null name in
            match dev.compiler.queues with
            | None -> dev
            | Some qs ->
                let commands q =
                  let c = qs.commands q in
                  let exec call prg =
                    c.exec call prg;
                    ignore
                      (Hcq2.Queue.q q
                         (List.map
                            (fun (a, c) -> Ops.getaddr ~device:d (region a c))
                            [ (128, 'a'); (256, 'b'); (128, 'c') ]))
                  in
                  { c with exec }
                in
                {
                  dev with
                  compiler =
                    { dev.compiler with queues = Some { qs with commands } };
                }
          in
          let compiled =
            Hcq2.compile_linear
              ~devices:(fun n -> (devices n).compiler)
              (linear [ kernel_adds (storage d) (storage d) ])
          in
          let starts =
            List.filter_map
              (fun u ->
                match (Ops.op u, Ops.src u) with
                | Shrink, b :: _
                  when Ops.tag (Ops.without_after b)
                       = Some (String "r_compute_0") -> (
                    match Ops.marg u with
                    | Shrink [ (Int start, _) ] -> Some start
                    | _ -> None)
                | _ -> None)
              (Ops.toposort ~enter_calls:true compiled)
          in
          equal (list int) [ 0; 256; 384 ] (List.sort_uniq Int.compare starts));
      test
        "regions that address the two before them, forty deep, are laid out at \
         once" (fun () ->
          let d = "CPU:1" in
          let bytes = Ops.v Binary ~arg:(Bytes (String.make 8 'a')) in
          let region src =
            Ops.v Linear ~src ~arg:(Region { name = "r"; align = 8 })
          in
          let rec chain k a b =
            if k = 0 then b
            else
              chain (k - 1) b
                (region [ Ops.getaddr ~device:d a; Ops.getaddr ~device:d b ])
          in
          let first = region [ bytes ] in
          let q = Hcq2.Queue.v ~devices:[ d ] "q" in
          ignore
            (Hcq2.Queue.q q
               [ Ops.getaddr ~device:d (chain 40 first (region [ first ])) ]);
          let buf = Hcq2.bufferize_cmdbuf q "test" in
          is_true ~msg:"the buffer of the regions"
            (List.exists
               (fun u -> Ops.tag u = Some (String "r_q"))
               (Ops.toposort buf)));
      test
        "a region addressed through another region is laid out once, in the \
         buffer of its name" (fun () ->
          let d = "CPU:1" in
          let region name src =
            Ops.v Linear ~src ~arg:(Region { name; align = 128 })
          in
          let bytes c = Ops.v Binary ~arg:(Bytes (String.make 8 c)) in
          let through = region "inner" [ bytes 'a' ]
          and direct = region "inner" [ bytes 'b' ] in
          let outer = region "outer" [ Ops.getaddr ~device:d through ] in
          let null = Null_device.devices () in
          let devices name =
            let dev = null name in
            match dev.compiler.queues with
            | None -> dev
            | Some qs ->
                let commands q =
                  let c = qs.commands q in
                  let exec call prg =
                    c.exec call prg;
                    ignore
                      (Hcq2.Queue.q q
                         [
                           Ops.getaddr ~device:d outer;
                           Ops.getaddr ~device:d direct;
                         ])
                  in
                  { c with exec }
                in
                {
                  dev with
                  compiler =
                    { dev.compiler with queues = Some { qs with commands } };
                }
          in
          let compiled =
            Hcq2.compile_linear
              ~devices:(fun n -> (devices n).compiler)
              (linear [ kernel_adds (storage d) (storage d) ])
          in
          let inner =
            List.filter
              (fun u -> Ops.tag u = Some (String "inner_compute_0"))
              (Ops.toposort ~enter_calls:true compiled)
          in
          equal int 1 (List.length inner));
      test "a word written at several offsets is written by one loop" (fun () ->
          let d = "CPU:1" in
          let scratch = Ops.new_buffer (Single d) 1 Uint64 in
          let null = Null_device.devices () in
          let devices name =
            let dev = null name in
            match dev.compiler.queues with
            | None -> dev
            | Some qs ->
                let commands q =
                  let c = qs.commands q in
                  let exec call prg =
                    c.exec call prg;
                    for _ = 1 to 10 do
                      c.signal scratch (Hcq2.submitted d)
                    done
                  in
                  { c with exec }
                in
                {
                  dev with
                  compiler =
                    { dev.compiler with queues = Some { qs with commands } };
                }
          in
          let calls = [ kernel_adds (storage d) (storage d) ] in
          let bound =
            (scratch, [ Nx_device.Buffer.create (nx d) UInt64 1 ])
            :: bound_storage calls
          in
          let compiled =
            Hcq2.compile_linear
              ~devices:(fun n -> (devices n).compiler)
              (linear calls)
          in
          let host = Ops.nth (Ops.without_after (the_batch compiled)) 0 in
          equal int ~msg:"loops" 1 (List.length (nodes Range host));
          let s = Tolk_engine.link ~devices ~bound compiled in
          Tolk_engine.run s [||];
          Null_device.synchronize ();
          let before = Nx_device.submitted (nx d) in
          Tolk_engine.run s [||];
          Null_device.synchronize ();
          equal int before (u64_of (List.hd (List.assq scratch bound)) 0));
      test "a command buffer is a placeholder whose tag starts with cmdbuf"
        (fun () ->
          let tagged t =
            Ops.placeholder ~device:(Single "CPU:1") ~tag:(String t) [ 8 ] Uint8
          in
          equal (list bool)
            [ true; true; false; false; false ]
            (List.map Hcq2.is_cmdbuf
               [
                 tagged (Hcq2.to_name [ "cmdbuf"; "COMPUTE:0" ]);
                 tagged "cmdbuf";
                 tagged (Hcq2.to_name [ "aql"; "COMPUTE:0" ]);
                 tagged "kernargs_cmdbuf";
                 Hcq2.signal_word "CPU:1";
               ]));
    ]

(* Commands a queue repeats: an end of a linear of commands over a range. *)
let looped_batch d words =
  let r = Ops.range (Int 3) [ 11 ] in
  let ins code src = Ops.v Ins ~src ~arg:(Code { code; dtype = Void }) in
  let word = Ops.shrink words [ Some (Sym r, Sym (Ops.add r (Ops.int 1))) ] in
  let body =
    Ops.v Linear
      ~src:[ ins "store" [ word; Ops.cast (Ops.add r (Ops.int 40)) Uint64 ] ]
  in
  let cmds =
    [
      ins "barrier" [];
      ins "wait" [ Hcq2.signal_word d; Hcq2.submitted d ];
      Ops.end_ body [ r ];
      ins "store" [ Hcq2.signal_word d; Hcq2.value d ];
    ]
  in
  let queue =
    Ops.v Linear ~arg:(Queue { devices = [ d ]; queue = "COMPUTE:0" }) ~src:cmds
  in
  let submission = Ops.custom_function "submit_cpu_compute" [ queue ] in
  Ops.call ~aux:(batch_info [ d ])
    (Ops.sink ~kernel:(Ops.kernel_info ~name:"hcq_submit" ()) [ submission ])
    []

let loops =
  group "loops of commands"
    [
      test "a loop of commands runs its body once for each value of its range"
        (fun () ->
          let d = "CPU:1" in
          let words = Ops.new_buffer (Single d) 3 Uint64 in
          let b = Nx_device.Buffer.create (nx d) UInt64 3 in
          let devices = Null_device.devices () in
          let lowered =
            Hcq2.lower_call
              ~devices:(fun n -> (devices n).compiler)
              (looped_batch d words)
          in
          let compiled =
            Realize.lower_and_compile
              ~targets:(fun n -> (devices n).compiler.target)
              (linear [ lowered ])
          in
          Tolk_engine.run
            (Tolk_engine.link ~devices ~bound:[ (words, [ b ]) ] compiled)
            [||];
          Null_device.synchronize ();
          equal (list int) [ 40; 41; 42 ]
            (List.map (fun k -> u64_of b (8 * k)) [ 0; 1; 2 ]));
    ]

(* A range around calls is a loop in its batch *)

let r = Ops.range (Int 3) [ 7 ]

let window u =
  let start = Ops.mul r (Ops.int 4) in
  Ops.shrink u [ Some (Sym start, Sym (Ops.add start (Ops.int 4))) ]

let ranged d =
  let src = Ops.new_buffer (Single d) 12 Float32
  and dst = Ops.new_buffer (Single d) 12 Float32 in
  (src, dst, Ops.end_ (kernel_adds (window dst) (window src)) [ r ])

(* A copy of each window of [src] into [tmp] on the copy queue, then [tmp] plus
   one into the window of [dst] on the compute queue: a trip's copy waits for
   the kernel of the trip before, which reads [tmp]. *)
let staged d =
  let src = storage ~n:12 d and dst = storage ~n:12 d and tmp = storage d in
  Ops.end_
    (linear [ Ops.store_call tmp (window src); kernel_adds (window dst) tmp ])
    [ r ]

let windows_bound src dst =
  [
    (src, [ new_floats "CPU:1" (Array.init 12 float_of_int) ]);
    (dst, [ new_floats "CPU:1" (Array.make 12 0.) ]);
  ]

(* Whether [s] holds [sub]. *)
let contains s sub =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
  in
  go 0

(* Buffers of [n] floats of [d], the [i]th holding [f i]. *)
let big_floats d n f =
  let b = Nx_device.Buffer.create (nx d) Float32 n in
  let a = Nx_device.Buffer.bigarray Bigarray.float32 (host_view b) in
  for i = 0 to n - 1 do
    Bigarray.Array1.unsafe_set a i (f i)
  done;
  b

(* A call of the kernel adding one on the windows [out] and [inp] of four
   floats, its parameters those of the windows, as a schedule makes them. *)
let windowed_adds out inp =
  let o = Ops.param_like out 0 and i = Ops.param_like inp 1 in
  let k = Ops.range (Int 4) [ 0 ] in
  let st = Ops.store (Ops.index o [ k ]) (plus 1. (Ops.index i [ k ])) in
  Ops.call
    (Ops.sink ~kernel:(Ops.kernel_info ~name:"k" ()) [ Ops.end_ st [ k ] ])
    [ out; inp ]

(* The source of the kernel of [call]. *)
let kernel_source call =
  match
    Ops.arg (Ops.nth (Codegen.to_program (Ops.nth call 0) uncompiled) 2)
  with
  | String src -> src
  | _ -> fail "a program holds its source"

(* The windows of four floats of [u], [stride] floats apart, one a trip of
   [r]. *)
let strided r stride u =
  let start = Ops.mul r (Ops.int stride) in
  Ops.shrink u [ Some (Sym start, Sym (Ops.add start (Ops.int 4))) ]

(* Two trips of a kernel adding one to windows of four floats [2^24 + 1] floats
   apart: the second trip's window is [2^26 + 4] bytes in, which a float does
   not hold, and 4 bytes past a 16-byte boundary. *)
let reads_its_window_past_a_float () =
  let stride = (1 lsl 24) + 1 in
  let n = stride + 4 in
  let r = Ops.range (Int 2) [ Ops.unique_num () ] in
  let window = strided r stride in
  let src = storage ~n "CPU:1" and dst = storage ~n "CPU:1" in
  let bound =
    [
      (src, [ big_floats "CPU:1" n (fun i -> Float.of_int (i mod 1000)) ]);
      (dst, [ big_floats "CPU:1" n (fun _ -> 0.) ]);
    ]
  in
  ignore
    (run_calls ~bound
       [ Ops.end_ (windowed_adds (window dst) (window src)) [ r ] ]);
  let a =
    Nx_device.Buffer.bigarray Bigarray.float32
      (host_view (List.hd (List.assq dst bound)))
  in
  equal (list float_exact) ~msg:"the second trip's window"
    (List.init 4 (fun i -> Float.of_int (((stride + i) mod 1000) + 1)))
    (List.init 4 (fun i -> a.{stride + i}))

(* Twelve trips of a kernel on its own window, on NULL queues that hold 256
   bytes of commands a submission: the range runs as batches of runs of its
   trips, one after the other. *)
let splits_a_range_its_queue_cannot_hold () =
  let n = 12 in
  let r = Ops.range (Int n) [ Ops.unique_num () ] in
  let window u =
    let start = Ops.mul r (Ops.int 4) in
    Ops.shrink u [ Some (Sym start, Sym (Ops.add start (Ops.int 4))) ]
  in
  let src = storage ~n:(4 * n) "CPU:1" and dst = storage ~n:(4 * n) "CPU:1" in
  let devices = Null_device.devices ~ring:256 () in
  let compiled =
    Hcq2.compile_linear
      ~devices:(fun d -> (devices d).compiler)
      (linear [ Ops.end_ (kernel_adds (window dst) (window src)) [ r ] ])
  in
  let trips b =
    match Ops.arg (Ops.without_after b) with
    | Call { aux = Some info; _ } -> List.length info.kernels
    | _ -> 0
  in
  let pieces = List.map trips (List.filter is_batch (Ops.src compiled)) in
  is_true ~msg:"several batches" (List.length pieces > 1);
  equal int ~msg:"every trip once" n (List.fold_left ( + ) 0 pieces);
  let bound =
    [
      (src, [ new_floats "CPU:1" (Array.init (4 * n) float_of_int) ]);
      (dst, [ new_floats "CPU:1" (Array.make (4 * n) 0.) ]);
    ]
  in
  Tolk_engine.run (Tolk_engine.link ~devices ~bound compiled) [||];
  Null_device.synchronize ();
  equal floats
    (Array.init (4 * n) (fun i -> float_of_int (i + 1)))
    (floats_of (List.hd (List.assq dst bound)))

(* [n] trips of a kernel on its own window, more calls than a batch holds: the
   range runs as one batch of a chunk of its trips, which the engine runs once
   per chunk, and a batch of the trips left. *)
let chunks_a_long_range () =
  let n = (2 * Hcq2.chunk_calls) + 5 in
  let r = Ops.range (Int n) [ Ops.unique_num () ] in
  let window u =
    let start = Ops.mul r (Ops.int 4) in
    Ops.shrink u [ Some (Sym start, Sym (Ops.add start (Ops.int 4))) ]
  in
  let src = storage ~n:(4 * n) "CPU:1" and dst = storage ~n:(4 * n) "CPU:1" in
  let devices = Null_device.devices () in
  let compiled =
    Hcq2.compile_linear
      ~devices:(fun d -> (devices d).compiler)
      (linear [ Ops.end_ (kernel_adds (window dst) (window src)) [ r ] ])
  in
  let kernels b =
    match Ops.arg (Ops.without_after b) with
    | Call { aux = Some info; _ } -> List.length info.kernels
    | _ -> 0
  in
  (match Ops.src compiled with
  | [ chunks; left ] ->
      equal op ~msg:"a range of the engine" End (Ops.op chunks);
      equal int ~msg:"a chunk's trips" Hcq2.chunk_calls
        (kernels (Ops.nth chunks 0));
      equal int ~msg:"the trips left" 5 (kernels left)
  | es -> failf "a range and a batch, not %d entries" (List.length es));
  let bound =
    [
      (src, [ new_floats "CPU:1" (Array.init (4 * n) float_of_int) ]);
      (dst, [ new_floats "CPU:1" (Array.make (4 * n) 0.) ]);
    ]
  in
  let d = Null_device.device "CPU:1" in
  let before = Nx_device.submitted d in
  Tolk_engine.run (Tolk_engine.link ~devices ~bound compiled) [||];
  Null_device.synchronize ();
  equal int ~msg:"a submission per chunk, and one for the trips left"
    (before + 3) (Nx_device.submitted d);
  equal floats
    (Array.init (4 * n) (fun i -> float_of_int (i + 1)))
    (floats_of (List.hd (List.assq dst bound)))

let ranges =
  group "ranges"
    [
      test
        "a range of more calls than a chunk runs as a batch of a chunk, once \
         per chunk"
        chunks_a_long_range;
      test "a range its queue cannot hold in one submission runs as several"
        splits_a_range_its_queue_cannot_hold;
      test "a call its queue cannot hold in one submission is refused"
        (fun () ->
          let devices = Null_device.devices ~ring:64 () in
          raises_match Exn.invalid_arg (fun () ->
              Hcq2.compile_linear
                ~devices:(fun d -> (devices d).compiler)
                (linear [ kernel_adds (storage "CPU:1") (storage "CPU:1") ])));
      test "a ranged batch's addresses are integers, profiled or not" (fun () ->
          let _, _, e = ranged "CPU:1" in
          let e =
            Ops.replace
              ~src:
                (adds (Ops.nth (Ops.nth e 0) 1) (Ops.nth (Ops.nth e 0) 2)
                :: List.tl (Ops.src e))
              e
          in
          List.iter
            (fun profile ->
              let src =
                host_sources
                  (Hcq2.compile_linear ~profile ~devices:(recorded_devices ())
                     (linear [ e ]))
              in
              is_false ~msg:"float" (contains src "float"))
            [ false; true ]);
      test "a trip reads its window past what a float offset holds"
        reads_its_window_past_a_float;
      test
        "a kernel reads windows an odd number of floats apart a float at a \
         time, and windows four floats apart four at a time" (fun () ->
          let r = Ops.range (Int 2) [ Ops.unique_num () ] in
          let src = storage ~n:64 "CPU:1" and dst = storage ~n:64 "CPU:1" in
          let source stride =
            kernel_source
              (windowed_adds (strided r stride dst) (strided r stride src))
          in
          is_false ~msg:"5 floats apart" (contains (source 5) "float4");
          is_true ~msg:"4 floats apart" (contains (source 4) "float4"));
      test "a range of enqueued calls is a loop in the queue of their batch"
        (fun () ->
          let _, _, e = ranged "CPU:1" in
          let compiled =
            sched
              [
                Ops.replace
                  ~src:
                    (adds (Ops.nth (Ops.nth e 0) 1) (Ops.nth (Ops.nth e 0) 2)
                    :: List.tl (Ops.src e))
                  e;
              ]
          in
          let batch =
            match Ops.src compiled with
            | [ u ] -> u
            | us -> failf "one entry, not %d" (List.length us)
          in
          is_true ~msg:"a batch" (is_batch batch);
          let loops =
            List.filter
              (fun n -> Ops.op n = End && List.memq r (List.tl (Ops.src n)))
              (Ops.toposort batch)
          in
          equal int ~msg:"one loop over the range" 1 (List.length loops);
          is_true ~msg:"around the queue's commands"
            (Ops.op (Ops.nth (List.hd loops) 0) = Linear));
      test "each trip of a batched range runs its calls on its own window"
        (fun () ->
          let src, dst, e = ranged "CPU:1" in
          let bound = windows_bound src dst in
          ignore (run_calls ~bound [ e ]);
          equal floats
            (Array.init 12 (fun i -> float_of_int (i + 1)))
            (floats_of (List.hd (List.assq dst bound))));
      test "a run of a batched range is one submission, whatever its trips"
        (fun () ->
          let src, dst, e = ranged "CPU:1" in
          let d = Null_device.device "CPU:1" in
          let s = run_calls ~bound:(windows_bound src dst) [ e ] in
          let before = Nx_device.submitted d in
          Tolk_engine.run s [||];
          Tolk_engine.run s [||];
          Null_device.synchronize ();
          equal int (before + 2) (Nx_device.submitted d));
      test "a profiled batched range records a span of each trip's kernel"
        (fun () ->
          let src, dst, e = ranged "CPU:1" in
          let p = Nx_device.Profile.start () in
          let events =
            Fun.protect
              ~finally:(fun () ->
                if Nx_device.Profile.enabled () then
                  ignore (Nx_device.Profile.stop p))
              (fun () ->
                ignore
                  (run_calls ~profile:true ~bound:(windows_bound src dst) [ e ]);
                Nx_device.Profile.stop p)
          in
          let spans =
            List.filter
              (function
                | Nx_device.Profile.Span sp ->
                    Nx_device.equal sp.device (Null_device.device "CPU:1")
                | _ -> false)
              events
          in
          equal int 3 (List.length spans));
      agrees "a batched range agrees with running its trips one by one"
        (let _, _, e = ranged "CPU:2" in
         [ e ]);
      agrees
        "a trip's copy waits for the kernel of the trip before, on another \
         queue"
        [ staged "CPU:1" ];
      test
        "on NV a compute queue's call past a loop's edge waits for its \
         previous call" (fun () ->
          let waits_before_calls kind =
            let d = kind ^ ":1" in
            let a = storage ~n:12 d and b = storage ~n:12 d in
            let batch =
              the_batch
                (sched
                   [
                     adds a a;
                     Ops.end_ (adds (window b) (window a)) [ r ];
                     adds b b;
                   ])
            in
            let rec flat cmds =
              List.concat_map
                (fun c ->
                  if Ops.op c = End then flat (Ops.src (Ops.nth c 0)) else [ c ])
                cmds
            in
            let rec count acc = function
              | [] -> []
              | c :: rest when instruction c = "call" -> acc :: count 0 rest
              | c :: rest ->
                  count (if instruction c = "wait" then acc + 1 else acc) rest
            in
            count 0 (flat (List.assoc (d, "COMPUTE:0") (Batches.queues batch)))
          in
          (* The first call waits for the device's earlier work, a trip's call
             for the call before the loop and the trip before's, and the call
             after the loop for the last trip's. *)
          equal (list int) ~msg:"NV" [ 1; 2; 1 ] (waits_before_calls "NV");
          equal (list int) ~msg:"another kind" [ 1; 0; 0 ]
            (waits_before_calls "CPU"));
      agrees ~latency:0.01 "a staged range agrees under queue latency"
        [ staged "CPU:2" ];
      test "a range of kernels on devices with queues stages, compiled or not"
        (fun () ->
          let src = storage ~n:12 "CPU:1" and dst = storage ~n:12 "CPU:1" in
          let stages = Hcq2.stages ~devices:(kinds ()) in
          is_true ~msg:"not compiled"
            (stages (Ops.end_ (kernel_adds (window dst) (window src)) [ r ]));
          is_true ~msg:"compiled"
            (stages (Ops.end_ (adds (window dst) (window src)) [ r ]));
          is_true ~msg:"with a copy" (stages (staged "CPU:1")));
      test
        "a range with a host program, a host copy or two kinds of device does \
         not stage" (fun () ->
          let stages = Hcq2.stages ~devices:(kinds ()) in
          let on d = kernel_adds (window (storage ~n:12 d)) (storage d) in
          is_false ~msg:"a host program"
            (stages (Ops.end_ (linear [ on "CPU:1"; on "CPU" ]) [ r ]));
          is_false ~msg:"all on the host" (stages (Ops.end_ (on "CPU") [ r ]));
          is_false ~msg:"a Metal copy"
            (stages
               (Ops.end_
                  (linear
                     [
                       on "METAL:0";
                       Ops.store_call (storage "METAL:0")
                         (window (storage ~n:12 "METAL:0"));
                     ])
                  [ r ]));
          is_false ~msg:"two kinds"
            (stages (Ops.end_ (linear [ on "AMD:0"; on "NV:0" ]) [ r ]));
          is_false ~msg:"no range" (stages (on "CPU:1")));
      test "a range of calls on the host and on queues is refused" (fun () ->
          let e =
            Ops.end_
              (linear
                 [
                   adds (window (storage ~n:12 "CPU:1")) (storage "CPU:1");
                   adds (storage "CPU") (storage "CPU");
                 ])
              [ r ]
          in
          raises_match Exn.invalid_arg (fun () -> sched [ e ]));
      test "a range of calls on devices of two kinds is refused" (fun () ->
          let e =
            Ops.end_
              (linear
                 [
                   adds (window (storage ~n:12 "AMD:0")) (storage "AMD:0");
                   adds (storage "NV:0") (storage "NV:0");
                 ])
              [ r ]
          in
          raises_match Exn.invalid_arg (fun () -> sched [ e ]));
    ]

(* Streamed queues *)

(* A batch's calls on the streamed device "CPU:1" and the host "CPU": the
   kernels on its compute queue, the copies on its copy queue. *)
let streamed calls = batches (sched ~submission:Streamed calls)

(* The queues of [batch] run to the end when the host hands each its commands as
   it submits them and a queue holds two. *)
let runs_streamed batch =
  well_formed batch;
  List.iter
    (fun ((d, q), n) ->
      equal int ~msg:(Printf.sprintf "commands left on %s %s" d q) 0 n)
    (Batches.run ~capacity:2 batch).left

let streamed_queues =
  group "streamed queues"
    [
      test
        "a queue is submitted after the queue it waits for, which it follows \
         in the batch" (fun () ->
          let x = storage "CPU:1" and d = storage "CPU:1" in
          let calls =
            [
              adds x (storage "CPU:1");
              Ops.store_call d (storage "CPU");
              adds (storage "CPU:1") d;
              adds (storage "CPU:1") x;
            ]
          in
          match streamed calls with
          | [ b ] ->
              runs_streamed b;
              equal
                (list (pair string string))
                [ ("CPU:1", "COPY:0"); ("CPU:1", "COMPUTE:0") ]
                (List.map fst (Batches.queues b))
          | bs -> failf "one batch, not %d" (List.length bs));
      test "a copy of a kernel's output runs after it" (fun () ->
          let x = storage "CPU:1" in
          let calls =
            [ adds x (storage "CPU:1"); Ops.store_call (storage "CPU") x ]
          in
          List.iter runs_streamed (streamed calls));
      test "queues that wait for each other run as several batches" (fun () ->
          let x = storage "CPU:1" and d = storage "CPU:1" in
          let calls =
            [
              adds x (storage "CPU:1");
              Ops.store_call (storage "CPU") x;
              Ops.store_call d (storage "CPU");
              adds (storage "CPU:1") d;
            ]
          in
          let bs = streamed calls in
          greater ~msg:"batches" int ~than:1 (List.length bs);
          List.iter runs_streamed bs);
      test "a range whose queues wait for each other runs to the end" (fun () ->
          let r = Ops.range (Int 3) [ 9 ] in
          let x = storage "CPU:1" and d = storage "CPU:1" in
          let e =
            Ops.end_
              (linear
                 [
                   adds x (storage "CPU:1");
                   Ops.store_call (storage "CPU") x;
                   Ops.store_call d (storage "CPU");
                   adds (storage "CPU:1") d;
                 ])
              [ r ]
          in
          List.iter runs_streamed (streamed [ e ]));
    ]

let () =
  exit
    (run "Hcq2"
       [
         recorded;
         profile_keys;
         views;
         names;
         timeline_values;
         layouts;
         deps_tests;
         group "Deps law" [ byte_model ];
         scheduling;
         stamps;
         lowering;
         compiling;
         running;
         host_functions;
         queues;
         word_tests;
         loops;
         ranges;
         streamed_queues;
       ])
