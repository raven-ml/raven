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

let () = run "Engine_hcq2" [
  test "byte intervals match a per-byte dependency model" byte_dependencies;
  test "owned aliases and device lanes preserve allocation identity" region_identity;
  test "only overlapping accesses wait across queues" overlap_waits;
  test "NV cross-queue waits close the previous compute chain" nv_chain;
  test "peer epilogues and profiling slots participate in timelines" peers_and_timestamps;
]
