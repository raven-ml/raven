(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop

let parameter ?(device = "NV") slot =
  U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int 32) ~device:(U.Single device) ()
let slice base start size = U.shrink ~src:base ~offset:(U.const_int start) ~size:(U.const_int size)
let instruction name u = match U.arg u with U.Arg.Typed (n, _) -> n = name | _ -> false
let constant_waits nodes = List.filter_map (fun u ->
    if instruction "wait" u then U.const_int_value (U.src u).(1) else None) nodes
let copy device queue dst src = Hcq2.{call = U.store_call ~dst ~src; device; queue}
let queue plan device name = List.find_map (fun (d, q, nodes) ->
    if d = device && q = name then Some nodes else None) plan.Hcq2.queues |> Option.get

let byte_dependencies () =
  let random = Random.State.make [| 1972 |] and tracker = Deps_tracker.create () in
  let written = Array.make_matrix 4 24 None and read = Array.make_matrix 4 24 [] in
  for token = 0 to 499 do
    let count = 1 + Random.State.int random 4 in
    let regions = List.init count (fun _ ->
        let start = Random.State.int random 25 in
        let stop = start + Random.State.int random (25 - start) in
        let lane = Random.State.int random 4 in
        Deps_tracker.{base = Graph 1; lane = Some lane; start; stop}) in
    let writes = List.init count Fun.id |> List.filter (fun _ -> Random.State.bool random) in
    let expected = ref [] in
    List.iteri (fun i (r : Deps_tracker.region) ->
        let lane = Option.get r.lane in
        for at = r.start to r.stop - 1 do
          Option.iter (fun t -> expected := t :: !expected) written.(lane).(at);
          if List.mem i writes then expected := read.(lane).(at) @ !expected
        done) regions;
    let actual = Deps_tracker.access tracker regions ~writes token in
    equal (list int) (List.sort_uniq Int.compare !expected) (List.sort_uniq Int.compare actual);
    List.iteri (fun i (r : Deps_tracker.region) ->
        let lane = Option.get r.lane in
        for at = r.start to r.stop - 1 do
          if List.mem i writes then begin written.(lane).(at) <- Some token; read.(lane).(at) <- [] end
          else read.(lane).(at) <- token :: read.(lane).(at)
        done) regions
  done

let region_identity () =
  let device = Tolk_cpu.create "CPU:regions" in
  let root = Device.create_buffer ~size:32 ~dtype:Dtype.int32 device in
  let view = Device.Buffer.view root ~size:8 ~dtype:Dtype.int32 ~offset:16 in
  let a = Deps_tracker.uop (slice (U.from_buffer root) 4 8)
  and b = Deps_tracker.uop (U.from_buffer view) in
  equal bool true (a = b);
  equal int 16 a.start;
  equal int 48 a.stop;
  let multi = U.param ~slot:100 ~dtype:Dtype.int32 ~shape:(U.const_int 32)
      ~device:(U.Multi ["NV"; "NV:1"]) () in
  let a = Deps_tracker.uop (slice (U.mselect ~src:multi ~index:0) 4 8)
  and b = Deps_tracker.uop (slice (U.mselect ~src:multi ~index:1) 4 8) in
  equal bool true (a.base = b.base);
  equal bool false (a.lane = b.lane)

let overlap_waits () =
  let a = parameter 0 and b = parameter 1 and c = parameter 2 in
  let calls = [copy "NV" "COPY:0" (slice a 0 8) (slice b 0 8);
               copy "NV" "COMPUTE:0" (slice c 0 8) (slice a 0 8)] in
  let plan = Hcq2.plan calls in
  equal (list int) [1; 1] (constant_waits (queue plan "NV" "COMPUTE:0"));
  let disjoint = Hcq2.plan [List.hd calls;
      copy "NV" "COMPUTE:0" (slice c 8 8) (slice a 8 8)] in
  (* The epilogue still joins the copy queue before advancing the timeline. *)
  equal (list int) [1] (constant_waits (queue disjoint "NV" "COMPUTE:0"));
  equal int 2 (List.length plan.signals);
  equal int 1 (List.length plan.timelines)

let parameter_views () =
  let small = U.param ~slot:0 ~dtype:Dtype.int32 ~shape:(U.const_int 1)
      ~device:(U.Single "NV") () in
  let large = U.param ~slot:0 ~dtype:Dtype.int32 ~shape:(U.const_int 32)
      ~device:(U.Single "NV") () in
  let plan = Hcq2.plan [copy "NV" "COPY:0" small (slice (parameter 1) 0 1);
      copy "NV" "COMPUTE:0" (parameter 2) large] in
  equal (list int) [1; 1] (constant_waits (queue plan "NV" "COMPUTE:0"))

let nv_chain () =
  let a = parameter 0 and b = parameter 1 and c = parameter 2 in
  let plan = Hcq2.plan [copy "NV" "COMPUTE:0" (slice c 0 8) (slice b 0 8);
      copy "NV" "COPY:0" (slice a 0 8) (slice b 0 8);
      copy "NV" "COMPUTE:0" (slice c 8 8) (slice a 0 8)] in
  equal (list int) [1; 2; 2] (constant_waits (queue plan "NV" "COMPUTE:0"))

let peers_and_timestamps () =
  let src = parameter 0 and dst = parameter ~device:"NV:1" 1 in
  let plan = Hcq2.plan ~profile:true [copy "NV" "COPY:0" dst src] in
  equal int 2 (List.length plan.timelines);
  equal (list int) [1] (constant_waits (queue plan "NV:1" "COMPUTE:0"));
  let stamps = List.filter (instruction "timestamp") (queue plan "NV" "COPY:0") in
  equal int 2 (List.length stamps);
  let offsets = List.map (fun n -> (Deps_tracker.uop (U.src n).(0)).start) stamps in
  equal (list int) [32; 48] offsets

let compiled_host_submission () =
  let host = Tolk_cpu.create "CPU" in
  let name = "CPU:queue-test" in
  let allocator = Device.Allocator.Pack (Storage.Host_allocator.make ~synchronize:(fun () -> ())) in
  let timeline = Device.Buffer.create ~device:name ~size:2 ~dtype:Dtype.uint64 allocator in
  Device.Buffer.ensure_allocated timeline;
  Device.Buffer.copyin timeline (Bytes.make 16 '\000');
  let observed = Device.Buffer.create ~device:name ~size:1 ~dtype:Dtype.uint64 allocator in
  let encode u = match U.op u, U.arg u, U.children u with
    | Ops.Custom_function, U.Arg.String "submit_cpu_copy", [linear; dependency] ->
        let trace = U.placeholder ~shape:[1] ~dtype:Dtype.uint64 ~slot:0
            ~device:(U.Single name) () |> U.with_tag "trace" in
        let observe = Hcq2.ccall ~after:[dependency] ~name:"memcpy" ~dtype:Dtype.uint64
            [U.getaddr ~device:name ~src:trace ();
             U.getaddr ~device:name ~src:(slice (Hcq2.timeline name) 1 1) ();
             U.const (Const.int Dtype.uint64 8)] in
        let previous = ref [observe] in
        let nodes = List.map (fun op ->
            let node = match U.as_call op, U.arg op with
              | Some {args = [dst; src]; _}, _ ->
                  Hcq2.ccall ~after:!previous ~name:"memcpy" ~dtype:Dtype.uint64
                    [U.getaddr ~device:name ~src:dst (); U.getaddr ~device:name ~src ();
                     U.const (Const.int Dtype.uint64 (U.max_numel src * Dtype.itemsize (U.dtype src)))]
              | _, U.Arg.Typed ("store", _) ->
                  U.store ~dst:(U.index ~ptr:(U.after ~src:(U.src op).(0) ~deps:!previous) ~idxs:[U.const_int 0] ())
                    ~value:(U.src op).(1) ()
              | _, U.Arg.Typed (("wait" | "barrier"), _) -> U.noop ~dtype:Dtype.void ()
              | _ -> fail "unexpected host queue instruction" in
            if U.op node <> Ops.Noop then previous := [node]; node) (U.children linear) in
        Some (U.group nodes)
    | _ -> None in
  let queue = Device.{host = "CPU"; copy = (fun _ -> true); encode; lower = (fun _ -> None);
    compile = Codegen.to_program ~optimize:false host (Device.renderer host)} in
  let renderer_set = Device.Renderer_set.make ~device:name
      ["CLANG", (fun target -> Renderer.with_target target (Device.renderer host))] in
  let device = Device.make ~name ~allocator ~renderer_set ~runtime:(Device.runtime host)
      ~synchronize:(fun () -> ()) ~queue
      ~bufferize:(fun p -> match U.node_tag p with
        | Some "timeline" -> Some timeline | Some "trace" -> Some observed | _ -> None) () in
  let ptr slot = U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int 1) ~device:(U.Single name) () in
  let linear = U.linear [U.store_call ~dst:(ptr 1) ~src:(ptr 0);
                         U.store_call ~dst:(ptr 2) ~src:(ptr 1)] in
  let to_program = Codegen.to_program host (Device.renderer host) in
  let compiled = Realize.compile_linear ~device ~to_program linear in
  let binding = Realize.Buffers.create () in
  let linked = Realize.link_linear binding compiled in
  let buffer value =
    let b = Device.create_buffer ~size:1 ~dtype:Dtype.int32 device in
    Device.Buffer.ensure_allocated b;
    let bytes = Bytes.create 4 in
    Bytes.set_int32_le bytes 0 value;
    Device.Buffer.copyin b bytes; b in
  List.iter (fun value ->
      let src = buffer value and mid = buffer 0l and dst = buffer 0l in
      Realize.run_linear ~device ~to_program binding ~jit:true ~wait:true
        ~input_uops:(Array.map U.from_buffer [|src; mid; dst|]) linked;
      equal int32 value (Bytes.get_int32_le (Device.Buffer.as_bytes dst) 0)) [42l; 71l];
  equal int64 2L (Bytes.get_int64_le (Device.Buffer.as_bytes timeline) 0);
  equal int64 2L (Bytes.get_int64_le (Device.Buffer.as_bytes observed) 0);
  let src = U.from_buffer (buffer 19l) and dst = U.from_buffer (buffer 0l) in
  let compiled = Realize.compile_linear ~device ~to_program
      (U.linear [U.store_call ~dst ~src]) in
  let linked = Realize.link_linear binding compiled in
  Realize.run_linear ~device ~to_program binding ~jit:true linked;
  let replacement = buffer 91l and output = buffer 0l in
  Realize.Buffers.seed binding src replacement;
  Realize.Buffers.seed binding dst output;
  Realize.run_linear ~device ~to_program binding ~jit:true linked;
  equal int32 91l (Bytes.get_int32_le (Device.Buffer.as_bytes output) 0)

let () = run "Engine_hcq2" [
  test "byte intervals match a per-byte dependency model" byte_dependencies;
  test "owned aliases and device lanes preserve allocation identity" region_identity;
  test "only overlapping accesses wait across queues" overlap_waits;
  test "parameter views retain their shared runtime slot" parameter_views;
  test "NV cross-queue waits close the previous compute chain" nv_chain;
  test "peer epilogues and profiling slots participate in timelines" peers_and_timestamps;
  test "compiled host submission patches addresses and replays through timelines" compiled_host_submission;
]
