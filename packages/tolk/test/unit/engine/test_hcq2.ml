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

let alias_ordering () =
  let a = parameter 0 and b = parameter 1 and c = parameter 2
  and d = parameter 3 in
  let independent = Hcq2.plan [copy "NV" "COPY:0" a b;
      copy "NV" "COPY:1" c d] in
  equal int 3 (List.length independent.independent_accesses);
  let same_queue = Hcq2.plan [copy "NV" "COPY:0" a b;
      copy "NV" "COPY:0" c d] in
  equal int 0 (List.length same_queue.independent_accesses);
  let transitive = Hcq2.plan [copy "NV" "COPY:0" a b;
      copy "NV" "COPY:1" c a; copy "NV" "COMPUTE:0" d c] in
  equal int 0 (List.length transitive.independent_accesses)

let peers_and_timestamps () =
  let src = parameter 0 and dst = parameter ~device:"NV:1" 1 in
  let plan = Hcq2.plan ~profile:true [copy "NV" "COPY:0" dst src] in
  equal int 2 (List.length plan.timelines);
  equal (list int) [1] (constant_waits (queue plan "NV:1" "COMPUTE:0"));
  let stamps = List.filter (instruction "timestamp") (queue plan "NV" "COPY:0") in
  equal int 2 (List.length stamps);
  let offsets = List.map (fun n -> (Deps_tracker.uop (U.src n).(0)).start) stamps in
  equal (list int) [32; 48] offsets

let all_to_all_copy_queues () =
  let host = Tolk_cpu.create "CPU" and selected = ref [] in
  let devices = List.init 3 (fun i -> "AMD:all-to-all-" ^ string_of_int i) in
  List.iter (fun name ->
      let encode u = match U.op u, U.arg u, U.children u with
        | Ops.Custom_function, U.Arg.String kind, [_; dependency]
            when String.starts_with ~prefix:"submit_amd_" kind ->
            if String.starts_with ~prefix:"submit_amd_copy_" kind then
              selected := kind :: !selected;
            Some (U.group [dependency])
        | _ -> None in
      let queue = Device.{timestamp_divider = 1.; profile_offset = (fun () -> 0.); completion = (fun () -> Fun.const ());
        prepare = (fun () -> ()); host = "CPU"; max_kernel_bindings = None; config = (fun () -> ""); copy = (fun _ -> Some "COPY:0");
        encode; lower = (fun _ -> None);
        compile = Codegen.to_program ~optimize:false host (Device.renderer host)} in
      let allocator = Device.Allocator.Pack (Storage.Host_allocator.make ~synchronize:(fun () -> ())) in
      ignore (Device.make ~name ~allocator
        ~renderer_set:(Device.Renderer_set.make ~device:name
          ["CLANG", (fun target -> Renderer.with_target target (Device.renderer host))])
        ~runtime:(Device.runtime host) ~synchronize:(fun timeout -> ignore timeout; ()) ~queue ())) devices;
  let src = parameter ~device:(List.hd devices) 0 in
  let linear = U.linear (List.mapi (fun i device ->
      U.store_call ~dst:(parameter ~device (i + 1)) ~src) devices) in
  let old_count = Sys.getenv_opt "HCQ_NUM_SDMA" in
  Fun.protect ~finally:(fun () -> Unix.putenv "HCQ_NUM_SDMA" (Option.value old_count ~default:"")) (fun () ->
      let check all2all count expected =
        selected := [];
        Unix.putenv "HCQ_NUM_SDMA" count;
        Helpers.Context_var.with_context [Helpers.Context_var.B (Helpers.all2all, all2all)] (fun () ->
            ignore (Hcq2.compile ~to_program:(fun device -> Codegen.to_program device (Device.renderer device)) linear));
        equal (list string) expected (List.sort_uniq String.compare !selected) in
      check 0 "" ["submit_amd_copy_0"];
      check 1 "" ["submit_amd_copy_0"; "submit_amd_copy_1"; "submit_amd_copy_2"];
      check 1 "2" ["submit_amd_copy_0"; "submit_amd_copy_1"];
      check 1 "0" ["submit_amd_copy_0"])

let peer_group_batches () =
  let host = Tolk_cpu.create "CPU" and prepared = ref 0 in
  let names = ["CPU:batch-a"; "CPU:batch-b"] in
  let make name =
    let encode u = match U.op u, U.arg u, U.children u with
      | Ops.Custom_function, U.Arg.String "submit_cpu_copy_0", [_; dependency] ->
          Some (U.group [dependency])
      | _ -> None in
    let copy call = match U.as_call call with
      | Some {args = dst :: _; _} when U.device_of dst = Some (U.Single name) -> Some "COPY:0"
      | _ -> None in
    let queue = Device.{timestamp_divider = 1.; profile_offset = (fun () -> 0.); completion = (fun () -> Fun.const ());
      prepare = (fun () -> incr prepared); host = "CPU"; max_kernel_bindings = None; config = (fun () -> ""); copy; encode; lower = (fun _ -> None);
      compile = Codegen.to_program ~optimize:false host (Device.renderer host)} in
    let allocator = Device.Allocator.Pack (Storage.Host_allocator.make ~synchronize:(fun () -> ())) in
    Device.make ~name ~peer_group:name ~allocator
      ~renderer_set:(Device.Renderer_set.make ~device:name
        ["CLANG", (fun target -> Renderer.with_target target (Device.renderer host))])
      ~runtime:(Device.runtime host) ~synchronize:(fun timeout -> ignore timeout; ()) ~queue () in
  let devices = List.map make names in
  let a slot = parameter ~device:(List.nth names 0) slot
  and b slot = parameter ~device:(List.nth names 1) slot in
  let store dst src = U.store_call ~dst ~src in
  let to_program device = Codegen.to_program device (Device.renderer device) in
  let compile calls = Hcq2.compile ~to_program (U.linear calls) in
  let info call = match U.arg (U.without_after call) with
    | U.Arg.Call_info {aux = Some info; _} -> info
    | _ -> fail "expected a queue batch" in
  let sizes linear = List.map (fun call -> List.length (info call).fallback) (U.children linear) in
  let linear = compile [store (a 1) (a 0); store (b 3) (b 2); store (a 5) (a 4)] in
  equal (list int) [2; 1] (sizes linear);
  let first = info (List.hd (U.children linear)) in
  equal int 3 (List.length first.independent_accesses);
  equal (list int) [1; 1; 1]
    (sizes (compile [store (a 1) (a 0); store (b 3) (a 1); store (a 5) (b 3)]));
  equal (list int) [1; 1; 1]
    (sizes (compile [store (a 1) (a 0); store (b 3) (a 5); store (a 5) (a 4)]));
  let separated = compile [store (a 1) (a 0); U.noop ~dtype:Dtype.void ();
      store (b 3) (b 2); store (a 5) (a 4)] in
  equal int 4 (List.length (U.children separated));
  let imported = U.import (U.export linear) in
  equal string (U.semantic_key linear) (U.semantic_key imported);
  let device = List.hd devices and binding = Realize.Buffers.create () in
  let linked = Realize.link_linear binding imported in
  let buffers = Array.init 6 (fun _ ->
      let buffer = Device.create_buffer ~size:32 ~dtype:Dtype.int32 host in
      Device.Buffer.ensure_allocated buffer; buffer) in
  (* These slots occur in different batches. Checking only the first batch's
     kernel arguments would miss this runtime alias introduced by reordering. *)
  buffers.(5) <- buffers.(3);
  raises (Invalid_argument "queue replay: bindings introduce an untracked writable alias")
    (fun () -> Realize.run_linear ~device ~to_program binding ~jit:true
      ~input_uops:(Array.map U.from_buffer buffers) linked);
  equal int 0 !prepared

let staged_peer_dependencies () =
  let staging_mode = ref `Accept in
  let host = Tolk_cpu.create "CPU" in
  let make name =
    let raw = Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
    let mapping = Option.get raw.mapping in
    let allocator = Device.Allocator.Pack {raw with mapping = Some {mapping with
        map = (fun source ->
          if String.starts_with ~prefix:"NV:staging" (Device.Buffer.device source) then
            raise (Storage.Mapping_unavailable "peer mapping unavailable");
          (match !staging_mode with
           | `Accept -> ()
           | `Reject -> raise (Storage.Mapping_unavailable "host mapping unavailable")
           | `Fault -> failwith "host mapping fault");
          mapping.map source)}} in
    let queue = Device.{timestamp_divider = 1.; profile_offset = (fun () -> 0.); completion = (fun () -> Fun.const ());
      prepare = (fun () -> ()); host = "CPU"; max_kernel_bindings = None; config = (fun () -> ""); copy = (fun _ -> Some "COPY:0");
      encode = (fun _ -> None); lower = (fun _ -> None);
      compile = (fun _ -> fail "staging plan should not compile")} in
    Device.make ~name ~allocator
      ~renderer_set:(Device.Renderer_set.make ~device:name
        ["CLANG", (fun target -> Renderer.with_target target (Device.renderer host))])
      ~runtime:(Device.runtime host) ~synchronize:(fun timeout -> ignore timeout; ()) ~queue () in
  let source = make "NV:staging-source" and target = make "NV:staging-target" in
  let chunk = 64 lsl 20 and size = (128 lsl 20) + 32 in
  let src_buffer = Device.create_buffer ~size ~dtype:Dtype.uint8 source
  and dst_buffer = Device.create_buffer ~size ~dtype:Dtype.uint8 target in
  let src = U.from_buffer src_buffer and dst = U.from_buffer dst_buffer in
  let linear = U.linear [U.store_call ~dst ~src] in
  let resolve node =
    if U.equal node src then src_buffer
    else if U.equal node dst then dst_buffer
    else fail "unexpected staging input" in
  let staged = Option.get (Hcq2.stage_copies ~resolve linear) in
  let calls = U.children staged in
  equal int 6 (List.length calls);
  let slots = List.filteri (fun i _ -> i mod 2 = 0) calls
      |> List.map (fun c -> List.hd (Option.get (U.as_call c)).args) in
  equal (list int) [0; chunk; 0] (List.map (fun s -> (Deps_tracker.uop s).start) slots);
  equal (list int) [chunk; chunk; 32] (List.map U.max_numel slots);
  let plan = Hcq2.plan (List.mapi (fun i call -> Hcq2.{call;
      device = Device.name (if i mod 2 = 0 then source else target); queue = "COPY:0"}) calls) in
  equal (list int) [2] (constant_waits (queue plan (Device.name source) "COPY:0"));
  equal (list int) [1; 3; 5] (constant_waits (queue plan (Device.name target) "COPY:0"));
  let other = Option.get (Hcq2.stage_copies ~resolve linear) in
  let other_slot = List.hd (Option.get (U.as_call (List.hd (U.children other)))).args in
  is_false ~msg:"independently prepared batches must not share staging storage"
    ((Deps_tracker.uop (List.hd slots)).base = (Deps_tracker.uop other_slot).base);
  staging_mode := `Reject;
  is_true (Option.is_none (Hcq2.stage_copies ~resolve linear));
  staging_mode := `Fault;
  raises (Failure "host mapping fault") (fun () -> Hcq2.stage_copies ~resolve linear)

let compiled_host_submission () =
  let host = Tolk_cpu.create "CPU" in
  let name = "CPU:queue-test" in
  let import_mode = ref `Accept in
  let implicit_waits = ref 0 in
  let raw = Storage.Host_allocator.make ~synchronize:(fun () -> incr implicit_waits) in
  let mapping = Option.get raw.mapping in
  let map source =
    if Device.Buffer.device source = "CPU:unmappable" then begin
      match !import_mode with
      | `Reject -> raise (Storage.Mapping_unavailable "test import is unsupported")
      | `Fault -> failwith "test import hardware fault"
      | `Accept -> ()
    end;
    mapping.map source in
  let allocator = Device.Allocator.Pack {raw with mapping = Some {mapping with map}} in
  let timeline = Device.Buffer.create ~device:name ~size:2 ~dtype:Dtype.uint64 allocator in
  Device.Buffer.ensure_allocated timeline;
  Device.Buffer.copyin timeline (Bytes.make 16 '\000');
  let observed = Device.Buffer.create ~device:name ~size:1 ~dtype:Dtype.uint64 allocator in
  let timestamp_step = ref 10 in
  let encode u = match U.op u, U.arg u, U.children u with
    | Ops.Custom_function, U.Arg.String ("submit_cpu_copy_0" | "submit_cpu_compute_0"), [linear; dependency] ->
        let trace = U.placeholder ~shape:[1] ~dtype:Dtype.uint64 ~slot:0
            ~device:(U.Single name) () |> U.with_tag "trace" in
        let arena = U.placeholder ~shape:[8] ~dtype:Dtype.uint8 ~slot:7
            ~device:(U.Single name) () |> U.with_tag "argument_arena" in
        let patched = Hcq2.patch ~after:[dependency] arena
            [0, U.alu_binary ~op:Ops.Add
              ~lhs:(U.load ~src:(U.index ~ptr:(Hcq2.timeline name)
                ~idxs:[U.const_int 1] ()) ())
              ~rhs:(U.const (Const.int Dtype.uint64 1))] in
        let observe = Hcq2.ccall ~after:[dependency] ~name:"memcpy" ~dtype:Dtype.uint64
            [U.getaddr ~device:name ~src:trace ();
             U.getaddr ~device:name ~src:patched ();
             U.const (Const.int Dtype.uint64 8)] in
        let previous = ref [observe] and stamp = ref 0 in
        let nodes = List.map (fun op ->
            let node = match U.as_call op, U.arg op with
              | Some {args = [dst; src]; _}, _ ->
                  Hcq2.ccall ~after:!previous ~name:"memcpy" ~dtype:Dtype.uint64
                    [U.getaddr ~device:name ~src:dst (); U.getaddr ~device:name ~src ();
                     U.const (Const.int Dtype.uint64 (U.max_numel src * Dtype.itemsize (U.dtype src)))]
              | _, U.Arg.Typed ("store", _) ->
                  U.store ~dst:(U.index ~ptr:(U.after ~src:(U.src op).(0) ~deps:!previous) ~idxs:[U.const_int 0] ())
                    ~value:(U.src op).(1) ()
              | _, U.Arg.Typed ("timestamp", _) ->
                  stamp := !stamp + !timestamp_step;
                  U.store ~dst:(U.index ~ptr:(U.after ~src:(U.src op).(0) ~deps:!previous)
                    ~idxs:[U.const_int 1] ()) ~value:(U.const (Const.int Dtype.uint64 !stamp)) ()
              | _, U.Arg.Typed (("wait" | "barrier"), _) -> U.noop ~dtype:Dtype.void ()
              | _ -> fail "unexpected host queue instruction" in
            if U.op node <> Ops.Noop then previous := [node]; node) (U.children linear) in
        Some (U.group nodes)
    | _ -> None in
  let copy_queue = ref "COPY:0" and generated = ref [] in
  let completions = ref [] in
  let completion () =
    let value = Bytes.get_int64_le (Device.Buffer.as_bytes timeline) 8 in
    fun timeout -> ignore timeout; completions := value :: !completions in
  let compilations = ref 0 in
  let compile sink =
    incr compilations;
    let program = Codegen.to_program ~optimize:false host (Device.renderer host) sink in
    Spec.type_verify Spec.program_spec (U.src program).(0);
    program in
  let ring = ref "A" in
  let queue = Device.{timestamp_divider = 1000.; profile_offset = (fun () -> 0.); completion; prepare = (fun () -> ()); host = "CPU"; max_kernel_bindings = None; config = (fun () -> "RING=" ^ !ring); copy = (fun _ -> Some !copy_queue); encode; lower = (fun _ -> None);
    compile} in
  let renderer_set = Device.Renderer_set.make ~device:name
      ["CLANG", (fun target -> Renderer.with_target target (Device.renderer host))] in
  let links = ref 0 in
  let device = Device.make ~name ~allocator ~renderer_set ~runtime:(Device.runtime host)
      ~synchronize:(fun timeout -> ignore timeout; ()) ~queue
      ~bufferize:(fun p -> match U.node_tag p with
        | Some "timeline" -> Some timeline
        | Some "trace" -> Some observed
        | Some "slots" -> incr links; None
        | _ -> None) () in
  let ptr slot = U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int 1) ~device:(U.Single name) () in
  let linear = U.linear [U.store_call ~dst:(ptr 1) ~src:(ptr 0);
                         U.store_call ~dst:(ptr 2) ~src:(ptr 1)] in
  let to_program device sink =
    let program = Codegen.to_program device (Device.renderer device) sink in
    generated := program :: !generated;
    program in
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
  let inputs = Array.map U.from_buffer [|buffer 12l; buffer 0l; buffer 0l|] in
  let enqueue () = Realize.run_linear ~device ~to_program binding ~jit:true
      ~input_uops:inputs linked in
  enqueue ();
  let before = !implicit_waits in
  enqueue ();
  equal ~msg:"bound queue replay must not synchronize storage owners" int before !implicit_waits;
  let replay linear inputs = Realize.run_linear ~device ~to_program binding
      ~jit:true ~wait:true ~input_uops:(Array.map U.from_buffer inputs) linear in
  let src = buffer 12l and dst = buffer 0l in
  replay linked [|src; src; dst|];
  equal int32 12l (Bytes.get_int32_le (Device.Buffer.as_bytes dst) 0);
  (* The host queue executes both kinds as copies; PROGRAM selects COMPUTE
     so the real planner must distinguish FIFO, waits and independent calls. *)
  let compute dst src =
    let p slot = U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int 1) () in
    let at ptr = U.index ~ptr ~idxs:[U.const_int 0] () in
    let store = U.store ~dst:(at (p 0)) ~value:(U.load ~src:(at (p 1)) ()) () in
    let kernel_info = U.{name = "host_queue_copy"; applied_opts = []; opts_to_apply = Some [];
      estimates = None; beam = 0} in
    let body = Codegen.to_program ~optimize:false host (Device.renderer host)
        (U.sink ~kernel_info [store]) in
    U.call ~body ~args:[dst; src]
      ~info:{grad_fxn = None; name = None; precompile = false;
        precompile_backward = false; dtype = Dtype.void; aux = None} in
  let compile ?profile calls = U.linear calls |> Realize.compile_linear ~device ?profile ~to_program
      |> Realize.link_linear binding in
  let independent = compile [U.store_call ~dst:(ptr 2) ~src:(ptr 0);
      compute (ptr 3) (ptr 1)] in
  let rejected inputs =
    let before = Bytes.get_int64_le (Device.Buffer.as_bytes timeline) 0 in
    raises (Invalid_argument "queue replay: bindings introduce an untracked writable alias")
      (fun () -> replay independent inputs);
    equal int64 before (Bytes.get_int64_le (Device.Buffer.as_bytes timeline) 0) in
  rejected [|src; src; dst; dst|];
  let root = Device.create_buffer ~size:4 ~dtype:Dtype.int32 device in
  Device.Buffer.ensure_allocated root;
  let bytes = Bytes.make 16 '\000' in
  Bytes.set_int32_le bytes 0 73l;
  Bytes.set_int32_le bytes 4 89l;
  Device.Buffer.copyin root bytes;
  let view offset = Device.Buffer.view root ~size:1 ~dtype:Dtype.int32 ~offset in
  replay independent [|view 0; view 4; view 8; view 12|];
  equal int32 73l (Bytes.get_int32_le (Device.Buffer.as_bytes root) 8);
  equal int32 89l (Bytes.get_int32_le (Device.Buffer.as_bytes root) 12);
  rejected [|view 0; view 4; view 8; view 8|];
  let external_view offset =
    let spec = {Device.Buffer_spec.default with
      external_ptr = Some (Nativeint.add (Device.Buffer.addr root) (Nativeint.of_int offset))} in
    Device.create_buffer ~size:1 ~dtype:Dtype.int32 ~spec device in
  rejected [|src; src; external_view 8; external_view 10|];
  replay independent [|external_view 0; external_view 4; external_view 8; external_view 12|];
  equal int32 89l (Bytes.get_int32_le (Device.Buffer.as_bytes root) 12);
  let dst1 = buffer 0l and dst2 = buffer 0l in
  replay independent [|src; src; dst1; dst2|];
  equal int32 12l (Bytes.get_int32_le (Device.Buffer.as_bytes dst1) 0);
  equal int32 12l (Bytes.get_int32_le (Device.Buffer.as_bytes dst2) 0);
  let ordered = compile [U.store_call ~dst:(ptr 1) ~src:(ptr 0);
      compute (ptr 2) (ptr 1)] in
  replay ordered [|src; dst1; src|];
  equal int32 12l (Bytes.get_int32_le (Device.Buffer.as_bytes src) 0);
  let timed = compile ~profile:true [U.store_call ~dst:(ptr 1) ~src:(ptr 0);
      compute (ptr 2) (ptr 1)] in
  let before = !(Helpers.Global_counters.time_sum_s) in
  replay timed [|src; dst1; dst2|];
  equal (float 1e-15) 2e-8 (!(Helpers.Global_counters.time_sum_s) -. before);
  equal int32 12l (Bytes.get_int32_le (Device.Buffer.as_bytes dst2) 0);
  timestamp_step := 1_000_000_000;
  let kernels = !(Helpers.Global_counters.kernel_count) in
  Realize.time_call ~device ~to_program
    (compute (U.from_buffer dst1) (U.from_buffer src)) (fun sample ->
      equal ~msg:"timing uses the device timestamp interval" (float 1e-12) 1. (sample ());
      equal ~msg:"retained timing storage supports another sample" (float 1e-12) 1. (sample ()));
  equal int kernels !(Helpers.Global_counters.kernel_count);
  equal int32 12l (Bytes.get_int32_le (Device.Buffer.as_bytes dst1) 0);
  timestamp_step := 10;
  let old_profile = Sys.getenv_opt "PROFILE" in
  Fun.protect ~finally:(fun () -> Unix.putenv "PROFILE" (Option.value old_profile ~default:"0")) (fun () ->
      Unix.putenv "PROFILE" "1";
      let profiled = compile [U.store_call ~dst:(ptr 1) ~src:(ptr 0);
          compute (ptr 2) (ptr 1)] in
      (* Profiling adds timestamps by default, but does not force a wait. *)
      let run () = Realize.run_linear ~device ~to_program binding ~jit:true
          ~input_uops:(Array.map U.from_buffer [|src; dst1; dst2|]) profiled in
      run (); run ();
      let events = Device.profile device in
      equal (list string) ["copy"; "host_queue_copy"]
        (List.map (fun e -> e.Profile.name) events |> List.sort String.compare);
      List.iter (fun e -> equal (float 1e-15) 0.01 e.Profile.duration_us) events;
      equal int 0 (List.length (Device.profile device)));
  ignore (Sys.opaque_identity root);
  let owner = Tolk_cpu.create "CPU:unmappable" in
  let foreign = Device.create_buffer ~size:1 ~dtype:Dtype.int32 owner in
  Device.Buffer.ensure_allocated foreign;
  let bytes = Bytes.create 4 in
  Bytes.set_int32_le bytes 0 347l;
  Device.Buffer.copyin foreign bytes;
  let input = U.param ~slot:0 ~dtype:Dtype.int32 ~shape:(U.const_int 1)
      ~device:(U.Single "CPU:unmappable") () in
  let template = Realize.compile_linear ~device ~to_program
      (U.linear [U.store_call ~dst:(ptr 1) ~src:input;
                 compute (ptr 2) (ptr 1)]) in
  let imported = U.import (U.export template) in
  equal string (U.semantic_key template) (U.semantic_key imported);
  (match U.arg (U.without_after (List.hd (U.children imported))) with
   | U.Arg.Call_info {aux = Some info; _} ->
       equal (list (pair string string)) [("CPU:unmappable", name)] info.host_deps
   | _ -> fail "import lost host dependencies");
  let transfer = Realize.link_linear binding imported in
  let middle = buffer 0l and output = buffer 0l in
  let before = Bytes.get_int64_le (Device.Buffer.as_bytes timeline) 0 in
  import_mode := `Reject;
  replay transfer [|foreign; middle; output|];
  equal int32 347l (Bytes.get_int32_le (Device.Buffer.as_bytes output) 0);
  equal int64 (Int64.succ before) (Bytes.get_int64_le (Device.Buffer.as_bytes timeline) 0);
  let compiled_before = !compilations in
  Bytes.set_int32_le bytes 0 643l;
  Device.Buffer.copyin foreign bytes;
  replay transfer [|foreign; middle; output|];
  equal int32 643l (Bytes.get_int32_le (Device.Buffer.as_bytes output) 0);
  equal int compiled_before !compilations;
  let rebound_weak =
    let rebound = Device.create_buffer ~size:1 ~dtype:Dtype.int32 owner in
    Device.Buffer.ensure_allocated rebound;
    Bytes.set_int32_le bytes 0 811l;
    Device.Buffer.copyin rebound bytes;
    let weak = Stdlib.Weak.create 1 in
    Stdlib.Weak.set weak 0 (Some rebound);
    Realize.run_linear ~device ~to_program binding ~jit:true
      ~input_uops:(Array.map U.from_buffer [|rebound; middle; output|]) transfer;
    weak in
  equal int32 811l (Bytes.get_int32_le (Device.Buffer.as_bytes output) 0);
  equal int compiled_before !compilations;
  Gc.full_major (); Gc.full_major ();
  is_false ~msg:"staged replay cache must not retain runtime inputs" (Stdlib.Weak.check rebound_weak 0);
  let before = Bytes.get_int64_le (Device.Buffer.as_bytes timeline) 0 in
  import_mode := `Fault;
  raises (Failure "test import hardware fault") (fun () -> replay transfer [|foreign; middle; output|]);
  equal int64 before (Bytes.get_int64_le (Device.Buffer.as_bytes timeline) 0);
  import_mode := `Accept;
  Bytes.set_int32_le bytes 0 347l;
  Device.Buffer.copyin foreign bytes;
  replay transfer [|foreign; middle; output|];
  equal int32 347l (Bytes.get_int32_le (Device.Buffer.as_bytes output) 0);
  equal int64 (Int64.succ before) (Bytes.get_int64_le (Device.Buffer.as_bytes timeline) 0);
  let completed_before = !completions in
  Device.synchronize owner;
  equal (list int64) (Int64.succ before :: completed_before) !completions;
  let waited = ref false in
  let timeline_before = Bytes.get_int64_le (Device.Buffer.as_bytes timeline) 0 in
  let completion () timeout =
    ignore timeout;
    equal ~msg:"foreign host writer must retire before the new submission" int64
      timeline_before (Bytes.get_int64_le (Device.Buffer.as_bytes timeline) 0);
    waited := true in
  let outsider = Device.make ~name:"OUTSIDE:writer" ~allocator ~renderer_set
      ~runtime:(Device.runtime host) ~synchronize:(fun timeout -> ignore timeout; fail "must wait only captured work")
      ~queue:{queue with completion} () in
  Device.depend_on owner outsider;
  Realize.run_linear ~device ~to_program binding ~jit:true
    ~input_uops:(Array.map U.from_buffer [|foreign; middle; output|]) transfer;
  is_true ~msg:"queue execution waits for uncovered host dependencies" !waited;
  Device.synchronize owner;
  let src = U.from_buffer (buffer 19l) and dst = U.from_buffer (buffer 0l) in
  let compiled = Realize.compile_linear ~device ~to_program
      (U.linear [U.store_call ~dst ~src]) in
  let linked = Realize.link_linear binding compiled in
  Realize.run_linear ~device ~to_program binding ~jit:true linked;
  let replacement = buffer 91l and output = buffer 0l in
  Realize.Buffers.seed binding src replacement;
  Realize.Buffers.seed binding dst output;
  Realize.run_linear ~device ~to_program binding ~jit:true linked;
  equal int32 91l (Bytes.get_int32_le (Device.Buffer.as_bytes output) 0);
  let run_eager src mid dst =
    let src = U.from_buffer src and mid = U.from_buffer mid and dst_node = U.from_buffer dst in
    let linear = U.linear [U.store_call ~dst:mid ~src;
        U.store_call ~dst:dst_node ~src:mid] in
    Realize.run_linear ~device ~to_program (Realize.Buffers.create ()) ~wait:true linear;
    Device.Buffer.ensure_allocated dst;
    Bytes.get_int32_le (Device.Buffer.as_bytes dst) 0 in
  let run_separate value = run_eager (buffer value) (buffer 0l) (buffer 0l) in
  let before = !compilations and linked_before = !links in
  equal int32 123l (run_separate 123l);
  equal int32 456l (run_separate 456l);
  equal int (before + 1) !compilations;
  equal int (linked_before + 1) !links;
  let weak = Stdlib.Weak.create 1 in
  let run_aliases value =
    let root = Device.create_buffer ~size:3 ~dtype:Dtype.int32 device in
    let bytes = Bytes.make 12 '\000' in
    Bytes.set_int32_le bytes 0 value;
    Device.Buffer.ensure_allocated root;
    Device.Buffer.copyin root bytes;
    Stdlib.Weak.set weak 0 (Some root);
    let view offset = Device.Buffer.view root ~size:1 ~dtype:Dtype.int32 ~offset in
    run_eager (view 0) (view 4) (view 8) in
  equal int32 789l (run_aliases 789l);
  equal int32 987l (run_aliases 987l);
  equal int (before + 2) !compilations;
  Gc.full_major ();
  Gc.full_major ();
  is_false ~msg:"cached submission does not retain input storage" (Stdlib.Weak.check weak 0);
  equal int32 111l (run_separate 111l);
  equal int (before + 2) !compilations;
  equal int (linked_before + 2) !links;
  Helpers.Context_var.with_context [Helpers.Context_var.B (Helpers.hcq_cache_thresh, 0)] (fun () ->
      equal int32 654l (run_separate 654l);
      equal int32 321l (run_separate 321l));
  equal int (before + 3) !compilations;
  equal int (linked_before + 4) !links;
  (* The encoder reads the queue's state: another config compiles anew. *)
  ring := "B";
  equal int32 222l (run_separate 222l);
  equal ~msg:"another queue config compiles its own template" int (before + 4)
    !compilations;
  copy_queue := "COMPUTE:0";
  let byte slot = U.param ~slot ~dtype:Dtype.uint8 ~shape:(U.const_int 37)
      ~device:(U.Single name) () in
  let linear = compile ~profile:true [U.store_call ~dst:(byte 1) ~src:(byte 0)] in
  let program = List.hd !generated in
  (match U.arg (U.without_after (List.hd (U.children linear))) with
   | U.Arg.Call_info {aux = Some info; _} ->
       equal int 1 (List.length info.fallback);
       is_true ~msg:"compute copy retains bulk-store fallback for later staging"
         (U.op (Option.get (U.as_call (List.hd info.fallback))).body = Ops.Store)
   | _ -> fail "compute copy was not queued");
  let source_bytes = Bytes.init 47 (fun i -> Char.chr ((i * 113 + 17) land 255)) in
  let destination_bytes = Bytes.make 45 '\x55' in
  let raw bytes =
    let b = Device.create_buffer ~size:(Bytes.length bytes) ~dtype:Dtype.uint8 device in
    Device.Buffer.ensure_allocated b; Device.Buffer.copyin b bytes; b in
  let source = raw source_bytes and destination = raw destination_bytes in
  let src = Device.Buffer.view source ~size:37 ~dtype:Dtype.uint8 ~offset:4
  and dst = Device.Buffer.view destination ~size:37 ~dtype:Dtype.uint8 ~offset:3 in
  let kernel = Device.runtime host (U.to_elf program) in
  Fun.protect ~finally:kernel.free (fun () ->
      ignore (kernel.call [|dst; src|] ~global:[|1; 1; 1|] ~local:None
        ~vals:[||] ~wait:true ~timeout:None : float option));
  Bytes.blit source_bytes 4 destination_bytes 3 37;
  equal Windtrap.bytes destination_bytes (Device.Buffer.as_bytes destination);
  Fun.protect ~finally:(fun () -> Unix.putenv "PROFILE" (Option.value old_profile ~default:"0")) (fun () ->
      Unix.putenv "PROFILE" "1";
      replay linear [|src; dst|];
      let events = Device.profile device in
      equal (list string) ["COMPUTE:0"] (List.map (fun e -> e.Profile.queue) events));
  (* A compute-only device must still insert staging on runtime import failure. *)
  import_mode := `Reject;
  let staged_compute = compile [U.store_call ~dst:(ptr 1) ~src:input] in
  replay staged_compute [|foreign; middle|];
  equal int32 347l (Bytes.get_int32_le (Device.Buffer.as_bytes middle) 0);
  let empty slot = U.param ~slot ~dtype:Dtype.uint8 ~shape:(U.const_int 0)
      ~device:(U.Single name) () in
  let empty_copy = compile [U.store_call ~dst:(empty 1) ~src:(empty 0)] in
  let src = Device.create_buffer ~size:0 ~dtype:Dtype.uint8 device
  and dst = Device.create_buffer ~size:0 ~dtype:Dtype.uint8 device in
  let before = !Realize.queue_submissions in
  replay empty_copy [|src; dst|];
  equal int before !Realize.queue_submissions

let () = run "Engine_hcq2" [
  test "AMD all-to-all honors default and explicit SDMA queue counts" all_to_all_copy_queues;
  test "staging alternates bounded slots with read-before-reuse dependencies" staged_peer_dependencies;
  test "interleaved groups preserve dependencies and check reordered aliases" peer_group_batches;
  test "byte intervals match a per-byte dependency model" byte_dependencies;
  test "owned aliases and device lanes preserve allocation identity" region_identity;
  test "only overlapping accesses wait across queues" overlap_waits;
  test "parameter views retain their shared runtime slot" parameter_views;
  test "NV cross-queue waits close the previous compute chain" nv_chain;
  test "alias constraints follow FIFO and transitive cross-queue waits" alias_ordering;
  test "peer epilogues and profiling slots participate in timelines" peers_and_timestamps;
  test "compiled host submission patches addresses and replays through timelines" compiled_host_submission;
]
