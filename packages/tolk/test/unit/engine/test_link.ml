(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop
module B = Device.Buffer

let info = U.{ grad_fxn = None; name = None; precompile = false;
  precompile_backward = false; dtype = Dtype.void; aux = None }

let call args = U.call ~body:(U.custom_function ~name:"inspect" ~srcs:[]) ~args ~info
let resolve = Realize.resolve (Realize.exec_context ())
let rec bare u = if U.op u = Ops.After then bare (U.src u).(0) else u
let args linear = match U.as_call (bare (U.src linear).(0)) with
  | Some {args; _} -> args | None -> fail "missing call"
let placeholder device tag dtype size =
  U.placeholder ~shape:[size] ~dtype ~slot:0 ~device:(U.Single (Device.name device)) ()
  |> U.with_tag tag
let initialized p bytes = U.set ~target:p ~value:(U.binary bytes) ()
let link ?allow_cache l = Realize.link_linear ?allow_cache l

let allocation_specs () =
  let device = Tolk_cpu.create "CPU:link-allocation-specs" in
  List.iter (fun (tag, volatile, host, uncached) ->
      let p = U.placeholder ~shape:[16] ~dtype:Dtype.uint8 ~slot:0
          ~device:(U.Single (Device.name device)) ~volatile () |> U.with_tag tag in
      let linked = link (U.linear [call [p]]) in
      let spec = B.spec (resolve (List.hd (args linked))) in
      equal ~msg:(tag ^ " host") bool host spec.host;
      equal ~msg:(tag ^ " uncached") bool uncached spec.uncached;
      equal ~msg:(tag ^ " CPU access") bool true spec.cpu_access)
    ["cmdbuf_compute", false, false, true;
     "cmdbuf_copy_0", false, false, true;
     "slots", true, true, true;
     "retired_compute", true, true, true;
     "qmd", false, false, false;
     "kernargs", false, false, false]

let initialization () =
  let device = Tolk_cpu.create "CPU:link-patches" in
  let p = placeholder device "commands" Dtype.uint8 24 in
  let initial = initialized p (String.make 24 '\255') in
  let words = U.bitcast ~src:initial ~dtype:Dtype.uint32 in
  let indices = U.stack [U.const_int 1; U.const_int 4] in
  let values = U.stack [U.const (Const.int Dtype.uint32 17); U.const (Const.int Dtype.uint32 42)] in
  let patch = U.store ~dst:(U.index ~ptr:words ~idxs:[indices] ()) ~value:values () in
  let range = U.range ~size:(U.const_int 2) ~axis:0 ~kind:Axis_type.Loop () in
  let offset = U.alu_binary ~op:Ops.Add ~lhs:range ~rhs:(U.const_int 2) in
  let value = U.cast ~src:(U.alu_binary ~op:Ops.Add ~lhs:range ~rhs:(U.const_int 100)) ~dtype:Dtype.uint32 in
  let store = U.store ~dst:(U.index ~ptr:words ~idxs:[offset] ()) ~value () in
  let ready = U.after ~src:initial ~deps:[patch; U.end_ ~value:store ~ranges:[range]] in
  let linear = U.linear [call [ready]] in
  let linked = link linear in
  let output = resolve (List.hd (args linked)) in
  let contents = B.as_bytes output in
  equal (list int32) [-1l; 17l; 100l; 101l; 42l; -1l]
    (List.init 6 (fun i -> Bytes.get_int32_le contents (i * 4)));
  B.copyin output (Bytes.make 24 '\000');
  let cached = link linear in
  equal bool true (U.equal cached linked);
  equal bytes (Bytes.make 24 '\000') (B.as_bytes output);
  let independent = link ~allow_cache:false linear in
  let fresh = resolve (List.hd (args independent)) in
  equal bool false (B.id output = B.id fresh);
  equal int32 17l (Bytes.get_int32_le (B.as_bytes fresh) 4)

let cast_patches () =
  let device = Tolk_cpu.create "CPU:link-casts" in
  let p = placeholder device "words" Dtype.uint8 12 in
  let uint64 n = U.const (Const.int64 Dtype.uint64 n) in
  let address = U.alu_binary ~op:Ops.Add ~lhs:(uint64 0x1234567800000000L)
      ~rhs:(uint64 0xabcdef01L) in
  let lower = U.cast ~src:address ~dtype:Dtype.uint32 in
  let upper = U.cast ~dtype:Dtype.uint32
      ~src:(U.alu_binary ~op:Ops.Shr ~lhs:address ~rhs:(uint64 32L)) in
  let byte = U.bitcast ~dtype:Dtype.uint8
      ~src:(U.cast ~src:(U.alu_unary ~op:Ops.Neg ~src:(U.const_int 17)) ~dtype:Dtype.int8) in
  let ready = Hcq2.patch ~blob:(String.make 12 '\000') p [0, lower; 4, upper; 8, byte] in
  let linked = link (U.linear [call [ready]]) in
  let contents = B.as_bytes (resolve (List.hd (args linked))) in
  equal int64 0x12345678abcdef01L (Bytes.get_int64_le contents 0);
  equal int 239 (Bytes.get_uint8 contents 8)

let addresses () =
  let device = Tolk_cpu.create "CPU:link-addresses" in
  let source = B.on_device ~device:(Device.name device) ~size:4 ~dtype:Dtype.uint64 () in
  let view = U.shrink ~src:(U.from_buffer source) ~offset:(U.const_int 2) ~size:(U.const_int 1) in
  let table = placeholder device "addresses" Dtype.uint64 1 in
  let addr = U.getaddr ~device:(Device.name device) ~src:view () in
  let patch = U.store ~dst:(U.index ~ptr:table ~idxs:[U.const_int 0] ()) ~value:addr () in
  let linked = link (U.linear [call [U.after ~src:table ~deps:[patch]]]) in
  let result = resolve (List.hd (args linked)) in
  equal int64 (Int64.add (Int64.of_nativeint (B.addr source)) 16L)
    (Bytes.get_int64_le (B.as_bytes result) 0);
  equal bool true (List.exists (fun n -> match U.as_buffer n with
      | Some {buffer = {buffer = Some [buf]; _}; _} -> B.base_id buf = B.id source
      | _ -> false) (U.toposort ~enter_calls:false linked))

let input_links () =
  let device = Tolk_cpu.create "CPU:link-inputs" in
  let p = placeholder device "lt_input" Dtype.int32 1 in
  let linear = U.linear [call [p]] in
  let input () = U.from_buffer (Device.create_buffer ~size:1 ~dtype:Dtype.int32 device) in
  let a = input () and c = input () in
  let first = Realize.link_linear ~ctx:(Realize.exec_context ~input_uops:[|a|] ()) linear in
  let second = Realize.link_linear ~ctx:(Realize.exec_context ~input_uops:[|c|] ()) linear in
  equal bool true (U.equal a (List.hd (args first)));
  equal bool true (U.equal c (List.hd (args second)))

let preserve_runtime () =
  let device = Tolk_cpu.create "CPU:link-runtime" in
  let p = U.param ~slot:0 ~shape:(U.const_int 1) ~dtype:Dtype.uint64 () in
  let inside = placeholder device "kernel-local" Dtype.uint64 1 in
  let body = U.sink [U.store ~dst:(U.index ~ptr:inside ~idxs:[U.const_int 0] ())
      ~value:(U.const (Const.int Dtype.uint64 0)) ()] in
  let original = U.call ~body ~args:[p] ~info in
  let linked = link (U.linear [original]) in
  equal bool true (U.equal original (U.src linked).(0))

let host_call_replay () =
  let device = Tolk_cpu.create "CPU:linked-host-call" in
  let slot i dtype = U.param ~slot:i ~shape:(U.const_int 1) ~dtype () in
  let out = slot 0 Dtype.int32 and fn = slot 1 Dtype.uint64 in
  let index p = U.index ~ptr:p ~idxs:[U.const_int 0] () in
  let n = U.variable ~param:true ~name:"n" ~min_val:(-100) ~max_val:100
      ~dtype:Dtype.int32 () in
  let body = U.custom_function ~name:"abs" ~srcs:[U.load ~src:(index fn) ()] in
  let invocation = U.call ~body ~args:[n] ~info:{info with dtype = Dtype.int32} in
  let kernel_info = U.{name = "linked_host_call"; applied_opts = [];
    opts_to_apply = None; estimates = None; beam = 0} in
  let sink = U.sink ~kernel_info [U.store ~dst:(index out) ~value:invocation ()] in
  let pointer = placeholder device "abs-pointer" Dtype.uint64 1 in
  let address = U.const (Const.int64 Dtype.uint64
      (Int64.of_nativeint (Tolk_cpu.link_symbol "abs"))) in
  let patch = U.store ~dst:(index pointer) ~value:address () in
  let args = [out; U.after ~src:pointer ~deps:[patch]] in
  let linear = U.linear [U.call ~body:sink ~args ~info] in
  let to_program device = Codegen.to_program ~optimize:false device (Device.renderer device) in
  let compiled = Realize.compile_linear ~device ~to_program linear in
  let linked = link compiled in
  let execute value =
    let result = Device.create_buffer ~size:1 ~dtype:Dtype.int32 device in
    Realize.run_linear ~device ~to_program ~jit:true ~wait:true
      ~input_uops:[|U.from_buffer result|] ~var_vals:["n", Int64.of_int value] linked;
    equal int32 (Int32.of_int (abs value)) (Bytes.get_int32_le (B.as_bytes result) 0) in
  execute (-31);
  Gc.full_major ();
  execute (-9)

let replacement_ownership () =
  let primary = Tolk_cpu.create "CPU:link-primary-owner" in
  let name = "CPU:link-secondary-owner" in
  let secondary = Tolk_cpu.create name in
  let linear = U.linear [call [placeholder primary "primary" Dtype.uint8 4;
      placeholder secondary "secondary" Dtype.uint8 4]] in
  let original = link linear in
  let previous = resolve (List.nth (args original) 1) in
  B.ensure_allocated previous;
  B.copyin previous (Bytes.of_string "kept");
  let replacement = Tolk_cpu.create name in
  let refreshed = link linear in
  let current = resolve (List.nth (args refreshed) 1) in
  is_false ~msg:"replacing a secondary owner invalidates cached links while old owners remain live"
    (B.id previous = B.id current);
  equal bytes (Bytes.of_string "kept") (B.as_bytes previous);
  is_true (U.equal refreshed (link linear));
  ignore (Sys.opaque_identity (primary, secondary, replacement, original))

let obsolete_link_collection () =
  let primary = Tolk_cpu.create "CPU:link-live-primary" in
  let name = "CPU:link-retired-secondary" in
  let weak_owner = Stdlib.Weak.create 1 and weak_buffer = Stdlib.Weak.create 1 in
  let populate () =
    let secondary = Tolk_cpu.create name in
    Stdlib.Weak.set weak_owner 0 (Some secondary);
    let linear = U.linear [call [placeholder primary "primary" Dtype.uint8 4;
        placeholder secondary "secondary" Dtype.uint8 4]] in
    let linked = link linear in
    Stdlib.Weak.set weak_buffer 0 (Some (resolve (List.nth (args linked) 1)));
    linear in
  let linear = populate () in
  let replacement = Tolk_cpu.create name in
  for _ = 1 to 5 do Gc.full_major () done;
  is_false ~msg:"cache keys do not retain an obsolete secondary device"
    (Stdlib.Weak.check weak_owner 0);
  is_false ~msg:"a live input graph and primary device do not retain obsolete linked storage"
    (Stdlib.Weak.check weak_buffer 0);
  ignore (Sys.opaque_identity (primary, replacement, linear))

let concurrent_link_publication () =
  let entered = Atomic.make 0 in
  let allocator = Device.Allocator.Pack
      (Storage.Host_allocator.make ~synchronize:(fun () -> ())) in
  let name = "CPU:parallel-link-publication" in
  let host = Tolk_cpu.create "CPU:parallel-link-host" in
  let device = Device.make ~name ~allocator
      ~renderer_set:(Device.Renderer_set.make ~device:"CPU"
        ["CLANG", (fun target -> Renderer.with_target target (Device.renderer host))])
      ~runtime:(Device.runtime host) ~synchronize:(fun timeout -> ignore timeout)
      ~bufferize:(fun node ->
        ignore (Atomic.fetch_and_add entered 1);
        while Atomic.get entered <> 2 do Domain.cpu_relax () done;
        Some (B.create ~device:name ~size:(U.max_numel node) ~dtype:(U.dtype node) allocator)) () in
  let linear = U.linear [call [placeholder device "shared" Dtype.uint8 4]] in
  let first = Domain.spawn (fun () -> link linear)
  and second = Domain.spawn (fun () -> link linear) in
  let first_result = Domain.join first and second_result = Domain.join second in
  is_true ~msg:"simultaneous misses publish one retained linked graph"
    (U.equal first_result second_result);
  is_true (U.equal first_result (link linear));
  ignore (Sys.opaque_identity device)

let () = exit (run "Engine_link" [
  test "secondary owner replacement invalidates cached links without invalidating retained links" replacement_ownership;
  test "obsolete owner storage retires even while the original graph remains live" obsolete_link_collection;
  test "concurrent first links publish one retained graph" concurrent_link_publication;
  test "allocates command streams uncached and volatile slots on the host" allocation_specs;
  test "initializes blobs, sparse words and ranged patches once" initialization;
  test "folds nested casts and bitcasts in link patches" cast_patches;
  test "retains mapped addresses and byte view offsets" addresses;
  test "does not cache link-time inputs" input_links;
  test "preserves runtime parameters and call bodies" preserve_runtime;
  test "executes linked host calls with rebound buffers and scalars" host_call_replay;
])
