(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop

let i32_param ~slot =
  U.param ~slot ~dtype:Dtype.int32
    ~shape:(U.stack [ U.const_int 16 ]) ~addrspace:Dtype.Global ()

let int32_to_bytes values =
  let bytes = Bytes.create (List.length values * 4) in
  let set off value =
    let open Int32 in
    Bytes.set bytes off (Char.chr (to_int (logand value 0xFFl)));
    Bytes.set bytes (off + 1)
      (Char.chr (to_int (logand (shift_right_logical value 8) 0xFFl)));
    Bytes.set bytes (off + 2)
      (Char.chr (to_int (logand (shift_right_logical value 16) 0xFFl)));
    Bytes.set bytes (off + 3)
      (Char.chr (to_int (logand (shift_right_logical value 24) 0xFFl)))
  in
  List.iteri (fun i value -> set (i * 4) (Int32.of_int value)) values;
  bytes

let int32_list_of_bytes bytes =
  let len = Bytes.length bytes / 4 in
  let get off =
    let open Int32 in
    logor
      (of_int (Char.code (Bytes.get bytes off)))
      (logor
         (shift_left (of_int (Char.code (Bytes.get bytes (off + 1)))) 8)
         (logor
            (shift_left (of_int (Char.code (Bytes.get bytes (off + 2)))) 16)
            (shift_left (of_int (Char.code (Bytes.get bytes (off + 3)))) 24)))
  in
  List.init len (fun i -> Int32.to_int (get (i * 4)))

let cpu name = Tolk_cpu.create ("CPU:" ^ name)

let create_i32_buffer device values =
  let buf =
    Device.create_buffer ~size:(List.length values) ~dtype:Dtype.int32 device
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (int32_to_bytes values);
  buf

let read_i32_buffer buf = Device.Buffer.as_bytes buf |> int32_list_of_bytes

let i32_view buf ~offset ~size =
  let view = Device.Buffer.view buf ~size ~dtype:Dtype.int32 ~offset in
  Device.Buffer.ensure_allocated view;
  view

let increment_program () =
  let dt = Dtype.int32 in
  let p0 = i32_param ~slot:0 in
  let p1 = i32_param ~slot:1 in
  let c0 = U.const (Const.int Dtype.int32 0) in
  let idx_src = U.index ~ptr:p1 ~idxs:[c0] () in
  let idx_dst = U.index ~ptr:p0 ~idxs:[c0] () in
  let l0 = U.load ~src:idx_src () in
  let c1 = U.const (Const.int dt 1) in
  let sum = U.alu_binary ~op:Ops.Add ~lhs:l0 ~rhs:c1 in
  let store = U.store ~dst:idx_dst ~value:sum () in
  [ p0; p1; c0; idx_src; idx_dst; l0; c1; sum; store ]

let direct_binding_waits_for_foreign_storage () =
  let host = cpu "direct-binding" in
  let syncs = ref 0 in
  let allocator = Device.Allocator.Pack
      (Storage.Host_allocator.make ~synchronize:(fun () -> incr syncs)) in
  let source = Device.Buffer.create ~device:"FOREIGN:binding" ~size:16
      ~dtype:Dtype.int32 allocator in
  Device.Buffer.ensure_allocated source;
  Device.Buffer.copyin source (int32_to_bytes (List.init 16 (fun i -> i + 41)));
  let importer_syncs = ref 0 in
  let imported_allocator = Device.Allocator.Pack
      (Storage.Host_allocator.make ~synchronize:(fun () -> incr importer_syncs)) in
  let importer = Device.make ~name:"CPU:binding-importer" ~allocator:imported_allocator
      ~renderer_set:(Device.Renderer_set.make ~device:"CPU:binding-importer"
        ["CLANG", (fun _ -> Device.renderer host)])
      ~runtime:(Device.runtime host) ~synchronize:(fun timeout -> ignore timeout; incr importer_syncs) () in
  ignore (Device.Buffer.get ~device:(Device.name importer) Storage.Host_allocator.kind source);
  let output = create_i32_buffer host (List.init 16 (fun _ -> 0)) in
  let spec = Device.compile_program host ~name:"foreign_increment" (increment_program ()) in
  let runtime = Device.runtime host (Program_spec.to_elf spec) in
  Fun.protect ~finally:runtime.free (fun () ->
      let run () = ignore (runtime.call [|output; source|] ~global:[|1; 1; 1|]
          ~local:None ~vals:[||] ~wait:false ~timeout:None) in
      run ();
      let before = !syncs and imported_before = !importer_syncs in
      run ();
      equal int (imported_before + 1) !importer_syncs;
      equal ~msg:"direct calls still synchronize a cached foreign binding" int (before + 1) !syncs;
      equal int 42 (List.hd (read_i32_buffer output)))

let core_id_program () =
  let dt = Dtype.int32 in
  let p0 = i32_param ~slot:0 in
  let core_id =
    U.variable ~param:true ~name:"core_id" ~min_val:2 ~max_val:7 ~dtype:dt ()
  in
  let idx = U.index ~ptr:p0 ~idxs:[core_id] () in
  let store = U.store ~dst:idx ~value:core_id () in
  [ p0; core_id; idx; store ]

let program_call spec bufs =
  let info = Program_spec.program_info spec in
  let kernel_info = U.{name = Program_spec.name spec; applied_opts = [];
    opts_to_apply = None; estimates = None; beam = 0} in
  let body = U.program ~sink:(U.sink ~kernel_info [U.linear (Program_spec.program spec)])
      ~linear:(U.linear (Program_spec.program spec)) ~source:(U.source (Program_spec.src spec))
      ~binary:(U.binary (Bytes.to_string (Option.get (Program_spec.lib spec)))) ~info () in
  let selected = List.combine info.globals bufs in
  let args = List.init (1 + List.fold_left max (-1) info.globals) (fun slot ->
      match List.assoc_opt slot selected with
      | Some buf -> U.from_buffer buf | None -> U.noop ()) in
  U.call ~body ~args ~info:U.{grad_fxn = None; name = None; precompile = false;
    precompile_backward = false; dtype = Dtype.void; aux = None}

let to_program device = Codegen.to_program ~optimize:false device (Device.renderer device)

let runtime_survives_owner_replacement () =
  let name = "cached-executable-lifetime" in
  let program, output, input =
    let device = cpu name in
    let spec = Device.compile_program device ~name:"retained_executable"
        (increment_program ()) in
    let output = create_i32_buffer device [0] and input = create_i32_buffer device [41] in
    Device.runtime device (Program_spec.to_elf spec), output, input in
  ignore (cpu name);
  for _ = 1 to 3 do Gc.full_major () done;
  let call () = ignore (program.Device.call [|output; input|]
      ~global:[|1; 1; 1|] ~local:None ~vals:[||] ~wait:false ~timeout:None) in
  Fun.protect ~finally:program.free (fun () ->
      call ();
      equal (list int) [42] (read_i32_buffer output));
  program.free ();
  raises (Invalid_argument "CPU program has been unloaded") call

let timing_cache_eviction () =
  let host = cpu "eviction-host" in
  let renderer_set = Device.Renderer_set.make ~device:"CPU"
      ["CLANG", Fun.const (Device.renderer host)] in
  let allocator = Device.Buffer.allocator
      (Device.create_buffer ~size:1 ~dtype:Dtype.uint8 host) in
  let fills = ref 0 and compilations = ref 0 in
  let eviction_buffer = ref None in
  let runtime object_ =
    let program = Device.runtime host object_ in
    let call buffers ~global ~local ~vals ~wait ~timeout =
      let elapsed = program.Device.call buffers ~global ~local ~vals ~wait ~timeout in
      if Array.length buffers = 1 && Device.Buffer.nbytes buffers.(0) = 1024 * 1024 * 4 then begin
        incr fills;
        eviction_buffer := Some buffers.(0);
        is_true (Device.Buffer.spec buffers.(0)).nolru;
        equal bool false wait;
        is_true (Dtype.equal (Device.Buffer.dtype buffers.(0)) Dtype.float32);
        let bytes = Device.Buffer.as_bytes buffers.(0) in
        for i = 0 to 1024 * 1024 - 1 do
          equal int32 0x3f800000l (Bytes.get_int32_le bytes (4 * i))
        done
      end;
      elapsed in
    {program with Device.call} in
  let device = Device.make ~name:"CPU:cache-eviction" ~allocator ~renderer_set
      ~runtime ~synchronize:(fun timeout -> Device.synchronize ?timeout host) () in
  let spec = Device.compile_program device ~name:"eviction_candidate" (increment_program ()) in
  let dst = create_i32_buffer device [0] and src = create_i32_buffer device [41] in
  let compile device sink =
    incr compilations;
    equal int 0 (Helpers.Context_var.get Helpers.beam);
    let ranges = U.toposort sink |> List.filter (fun u -> U.op u = Ops.Range) in
    equal (list int) [1024; 1024]
      (List.map (fun u -> Option.get (U.const_int_value (U.src u).(0))) ranges);
    (* Direct codegen consumes the kernel's resolved beam policy; it must not
       read a surrounding context again after compile_linear selected zero. *)
    Helpers.Context_var.with_context [B (Helpers.beam, 3)] (fun () ->
      Codegen.to_program device (Device.renderer device) sink) in
  let kernels = (Helpers.Global_counters.snapshot ()).kernel_count in
  Realize.with_capture
    (fun linear vars ->
      ignore (linear, vars); fail "cache eviction entered JIT capture")
    (fun () ->
      Helpers.Context_var.with_context [B (Helpers.beam, 3)] (fun () ->
        Realize.time_call ~device ~to_program:compile ~clear_l2:true
          (program_call spec [dst; src]) (fun sample ->
            for _ = 1 to 2 do
              is_true (sample () > 0.);
              equal int 3 (Helpers.Context_var.get Helpers.beam)
            done)));
  equal int 1 !compilations;
  equal int 2 !fills;
  is_false (Device.Buffer.is_allocated (Option.get !eviction_buffer));
  equal int kernels (Helpers.Global_counters.snapshot ()).kernel_count;
  equal (list int) [42] (read_i32_buffer dst);
  let failing_device = Device.make ~name:"CPU:cache-eviction-failure" ~allocator ~renderer_set
      ~runtime:(Device.runtime host)
      ~synchronize:(fun timeout -> Device.synchronize ?timeout host) () in
  let failed_dst = create_i32_buffer failing_device [0]
  and failed_src = create_i32_buffer failing_device [41] in
  Helpers.Context_var.with_context [B (Helpers.beam, 5)] (fun () ->
    raises Exit (fun () ->
      Realize.time_call ~device:failing_device
        ~to_program:(fun device sink ->
          equal string "CPU:cache-eviction-failure" (Device.name device);
          equal (option string) (Some "clear_l2")
            (Option.map (fun (info : U.kernel_info) -> info.name) (U.as_kernel_info sink));
          equal int 0 (Helpers.Context_var.get Helpers.beam);
          raise Exit)
        ~clear_l2:true (program_call spec [failed_dst; failed_src])
        (fun sample -> ignore (sample ())));
    equal int 5 (Helpers.Context_var.get Helpers.beam))

let run_spec device spec bufs =
  Realize.run_linear ~device ~to_program ~wait:true (U.linear [program_call spec bufs])

(* Like [increment_program] but subtracting, so no other test builds this
   graph: nodes exported by the forked child below are genuinely foreign to
   the parent process until imported. *)
let decrement_program () =
  let dt = Dtype.int32 in
  let p0 = i32_param ~slot:0 in
  let p1 = i32_param ~slot:1 in
  let c0 = U.const (Const.int Dtype.int32 0) in
  let idx_src = U.index ~ptr:p1 ~idxs:[ c0 ] () in
  let idx_dst = U.index ~ptr:p0 ~idxs:[ c0 ] () in
  let l0 = U.load ~src:idx_src () in
  let c1 = U.const (Const.int dt 1) in
  let diff = U.alu_binary ~op:Ops.Sub ~lhs:l0 ~rhs:c1 in
  let store = U.store ~dst:idx_dst ~value:diff () in
  [ p0; p1; c0; idx_src; idx_dst; l0; c1; diff; store ]

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

(* Re-run this executable with [TOLK_EXPORT_BLOB] set to export a program
   from a separate process, then import and execute it here. *)
let export_blob_var = "TOLK_EXPORT_BLOB"

let export_child path =
  let oc = open_out_bin path in
  output_string oc (U.export (U.sink (decrement_program ())));
  close_out oc

let imported_program_runs () =
  let blob_file = Filename.temp_file "tolk-import-exec" ".blob" in
  Fun.protect
    ~finally:(fun () -> Sys.remove blob_file)
    (fun () ->
      let env =
        Array.append (Unix.environment ())
          [| export_blob_var ^ "=" ^ blob_file |]
      in
      let pid =
        Unix.create_process_env Sys.executable_name
          [| Sys.executable_name |]
          env Unix.stdin Unix.stdout Unix.stderr
      in
      let _, status = Unix.waitpid [] pid in
      (match status with
       | Unix.WEXITED 0 -> ()
       | _ -> fail "exporting child failed");
      let imported = U.import (read_file blob_file) in
      let device = cpu "imported" in
      let spec =
        Device.compile_program device ~name:"sub_one" (U.children imported)
      in
      let dst = create_i32_buffer device [ 0 ] in
      let src = create_i32_buffer device [ 42 ] in
      run_spec device spec [ dst; src ];
      equal (list int) [ 41 ] (read_i32_buffer dst))

(* Coverage gaps accepted as not unit-testable at this level:
   - the bfloat16 compiler probe (Compiler_cpu.supports_bf16) is gated on the
     host clang version;
   - the AArch64 CALL26/JUMP26 trampoline only fires for branch targets beyond
     +/-128 MiB, unreachable with kernel-sized images. *)
let test_compilation_preserves_alignment () =
  let param slot =
    U.param ~slot ~dtype:Dtype.float32 ~shape:(U.const_int 4)
      ~addrspace:Dtype.Global () in
  let src = param 0 and dst = param 1 in
  let zero = U.const (Const.int Dtype.int32 0) in
  let window p = U.shrink ~src:p ~offset:zero ~size:(U.const_int 4) in
  let input = window src and output = window dst in
  let value = U.load ~src:input () in
  let program = [ src; dst; zero; input; output; value; U.store ~dst:output ~value () ] in
  let compile aligned name =
    let device = Tolk_cpu.create ~aligned ("CPU:" ^ name) in
    let renderer = Device.renderer device in
    let expected = Renderer.render renderer ~name:"alignment_cache" program in
    let spec = Device.compile_program device ~name:"alignment_cache" program in
    equal string expected (Program_spec.src spec) in
  compile true "aligned-cache";
  compile false "unaligned-cache"

let test_same_name_compilation_preserves_alignment () =
  let name = "CPU:replaced-alignment" in
  let param slot =
    U.param ~slot ~dtype:Dtype.float32 ~shape:(U.const_int 4)
      ~addrspace:Dtype.Global () in
  let src = param 0 and dst = param 1 in
  let zero = U.const (Const.int Dtype.int32 0) in
  let window p = U.shrink ~src:p ~offset:zero ~size:(U.const_int 4) in
  let input = window src and output = window dst in
  let value = U.load ~src:input () in
  let store = U.store ~dst:output ~value () in
  let instructions = [src; dst; zero; input; output; value; store] in
  let kernel_info = U.{name = "same_name_alignment"; applied_opts = [];
    opts_to_apply = None; estimates = None; beam = 0} in
  let sink = U.sink ~kernel_info [store] in
  let compile device =
    let to_program owner body =
      let spec = Device.compile_program owner ~name:kernel_info.name instructions in
      U.program ~sink:body ~linear:(U.linear instructions)
        ~source:(U.source (Program_spec.src spec))
        ~binary:(U.binary (Bytes.to_string (Option.get (Program_spec.lib spec))))
        ~info:(Program_spec.program_info spec) () in
    let args = List.init 2 (fun _ ->
        U.from_buffer (Device.create_buffer ~size:4 ~dtype:Dtype.float32 device)) in
    let call = U.call ~body:sink ~args
        ~info:U.{grad_fxn = None; name = None; precompile = false;
          precompile_backward = false; dtype = Dtype.void; aux = None} in
    let linear = Realize.compile_linear ~device ~to_program (U.linear [call]) in
    match U.children linear with
    | [call] -> (Option.get (U.as_call call)).body
    | _ -> fail "expected one compiled CPU call" in
  let source program =
    match U.children program with
    | [_; _; source; _] -> Option.get (U.Arg.as_string (U.arg source))
    | _ -> fail "expected PROGRAM source" in
  let aligned = Tolk_cpu.create ~aligned:true name in
  let first = compile aligned in
  let unaligned = Tolk_cpu.create ~aligned:false name in
  let second = compile unaligned in
  List.iter (fun (device, program) ->
      equal string
        (Renderer.render (Device.renderer device) ~name:kernel_info.name instructions)
        (source program);
      equal string (Target.to_string (Renderer.target (Device.renderer device)))
        (Target.to_string (Option.get (U.as_program_info program)).target))
    [aligned, first; unaligned, second];
  is_true ~msg:"alignment changes generated source" (source first <> source second);
  is_true ~msg:"retained old owner keeps its own cached program" (compile aligned == first);
  is_true ~msg:"replacement owner reuses its own cached program" (compile unaligned == second)

let test_split_axis_identity () =
  let device = cpu "split-axis-identity" in
  let range sub size =
    U.range ~size:(U.const_int size) ~axis:7 ~sub ~kind:Axis_type.Upcast () in
  let row = range [ 0 ] 2 and col = range [ 1 ] 3 in
  let mul x n = U.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:(U.const_int n) in
  let add a b = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
  let dst = U.param ~slot:0 ~dtype:Dtype.int32 ~shape:(U.const_int 6)
      ~addrspace:Dtype.Global () in
  let index = U.index ~ptr:dst ~idxs:[ add (mul row 3) col ] () in
  let value = U.cast ~src:(add (mul row 10) col) ~dtype:Dtype.int32 in
  let store = U.store ~dst:index ~value () in
  let sink = U.sink [ U.end_ ~value:store ~ranges:[ row; col ] ] in
  let program = Codegen_lower.lower (Device.renderer device) sink |> Linearizer.linearize in
  let spec = Device.compile_program device ~name:"split_axis_identity" program in
  let output = create_i32_buffer device [ -1; -1; -1; -1; -1; -1 ] in
  run_spec device spec [ output ];
  equal (list int) [ 0; 1; 2; 10; 11; 12 ] (read_i32_buffer output)

let test_conditional_loop () =
  let device = cpu "conditional-loop" in
  let dst = U.param ~slot:0 ~dtype:Dtype.int32 ~shape:(U.const (Const.int Dtype.int32 1))
      ~addrspace:Dtype.Global () in
  let loop = U.loop ~axis:0 in
  let ptr = U.after ~src:dst ~deps:[ loop ] in
  let idx = U.index ~ptr ~idxs:[ U.const (Const.int Dtype.int32 0) ] () in
  let value = U.load ~src:idx () in
  let next = U.alu_binary ~op:Ops.Add ~lhs:value
      ~rhs:(U.const (Const.int Dtype.int32 1)) in
  let store = U.store ~dst:idx ~value:next () in
  let cond = U.alu_binary ~op:Ops.Cmplt ~lhs:next
      ~rhs:(U.const (Const.int Dtype.int32 5)) in
  let edge = U.backedge ~body:store ~loop ~cond in
  let kernel_info = U.{name = "conditional_loop"; applied_opts = []; opts_to_apply = None;
    estimates = None; beam = 0} in
  let sink = U.sink ~kernel_info [ edge ]
      |> Codegen.full_rewrite_to_sink (Device.renderer device) in
  let program = Linearizer.linearize sink in
  Spec.verify_list Spec.program_spec program;
  let spec = Device.compile_program device ~name:"conditional_loop" program in
  let output = create_i32_buffer device [ 0 ] in
  run_spec device spec [ output ];
  equal (list int) [ 5 ] (read_i32_buffer output)

let test_host_calls () =
  let device = cpu "host-calls" in
  let renderer = Device.renderer device in
  let compiler = Option.get (Renderer.compiler renderer) in
  let i32 n = U.const (Const.int Dtype.int32 n) in
  let dst = U.param ~slot:0 ~dtype:Dtype.int32 ~shape:(i32 4)
      ~addrspace:Dtype.Global () in
  let pointer name = U.custom_inline ~fmt:("(unsigned long)&" ^ name)
      ~args:[] ~dtype:Dtype.uint64 in
  let invoke name dtype args =
    let body = U.custom_function ~name ~srcs:[ pointer name ] in
    let info : U.call_info =
      { grad_fxn = None; name = None; precompile = false;
        precompile_backward = false; aux = None; dtype }
    in
    U.call ~body ~args ~info
  in
  let index offset = U.index ~ptr:dst ~idxs:[ offset ] () in
  let compile_run name helper sink output =
    let sink = Linearizer.pm_add_control_flow sink in
    let program = Linearizer.linearize sink in
    Spec.verify_list Spec.program_spec program;
    let source = helper ^ "\n" ^ Renderer.render renderer ~name program in
    let lib = Compiler.compile compiler source in
    let spec = Program_spec.of_program ~name ~src:source
        ~device:(Device.name device) ~lib program in
    run_spec device spec [ output ]
  in
  let void = invoke "call_out" Dtype.void [ i32 3; index (i32 0) ] in
  let output = create_i32_buffer device [ 0; 0; 0; 0 ] in
  compile_run "host_call_out"
    "static void call_out(int n, int *out) { out[0] = n * 2; }"
    (U.sink [ void ]) output;
  equal (list int) [ 6; 0; 0; 0 ] (read_i32_buffer output);
  let range = U.range ~size:(i32 3) ~axis:0 ~kind:Axis_type.Loop
      ~dtype:Dtype.int32 () in
  let call = invoke "call_ret" Dtype.int32 [ range; index (i32 0) ] in
  let value = U.alu_binary ~op:Ops.Add ~lhs:call ~rhs:call in
  let store = U.store ~dst:(index (U.alu_binary ~op:Ops.Add ~lhs:range ~rhs:(i32 1)))
      ~value () in
  let output = create_i32_buffer device [ 0; 0; 0; 0 ] in
  compile_run "host_call_ret"
    "static int call_ret(int n, int *count) { count[0]++; return n + 1; }"
    (U.sink [ U.end_ ~value:store ~ranges:[ range ] ]) output;
  equal (list int) [ 3; 2; 4; 6 ] (read_i32_buffer output)

let test_emulated_long_to_float64 () =
  let device = cpu "emulated-long-cast" in
  let cases =
    [ Dtype.int64, 0L, 0.0;
      Dtype.int64, -1L, -1.0;
      Dtype.int64, 2147483648L, 2147483648.0;
      Dtype.int64, 4294967297L, 4294967297.0;
      Dtype.int64, 4886718345L, 4886718345.0;
      Dtype.int64, -4294967297L, -4294967297.0;
      Dtype.int64, 9007199254740991L, 9007199254740991.0;
      Dtype.int64, -9007199254740991L, -9007199254740991.0;
      Dtype.int64, Int64.min_int, -9223372036854775808.0;
      Dtype.uint64, 4294967297L, 4294967297.0;
      Dtype.uint64, 4886718345L, 4886718345.0;
      Dtype.uint64, Int64.min_int, 9223372036854775808.0;
      Dtype.uint64, -1L, 18446744073709551616.0 ] in
  let count = List.length cases in
  let dst = U.param ~slot:0 ~dtype:Dtype.float64 ~shape:(U.const_int count)
      ~addrspace:Dtype.Global () in
  let stores = List.mapi (fun i (dtype, value, _) ->
      let cast = U.cast ~src:(U.const (Const.int64 dtype value))
          ~dtype:Dtype.float64 in
      (* Explicit decomposition exercises the emulation on CPUs with native
         64-bit integers as well. Lowering must preserve its numeric result. *)
      let value = U.graph_rewrite ~bottom_up:true
          (Upat.Pattern_matcher.rewrite (Decomp_dtype.pm_long_decomp ())) cast in
      let index = U.index ~ptr:dst ~idxs:[ U.const_int i ] () in
      U.store ~dst:index ~value ()) cases in
  let program = Codegen_lower.lower (Device.renderer device) (U.sink stores)
      |> Linearizer.linearize in
  let spec = Device.compile_program device ~name:"emulated_long_cast" program in
  let output = Device.create_buffer ~size:count ~dtype:Dtype.float64 device in
  Device.Buffer.ensure_allocated output;
  run_spec device spec [ output ];
  let bytes = Device.Buffer.as_bytes output in
  List.iteri (fun i (dtype, value, expected) ->
      let actual = Int64.float_of_bits (Bytes.get_int64_le bytes (8 * i)) in
      equal ~msg:(Printf.sprintf "%s %Ld" (Dtype.to_string dtype) value)
        float_exact expected actual) cases

let test_emulated_long_buffer_arithmetic () =
  let device = cpu "emulated-long-buffer" in
  let renderer = Renderer.make ~name:"emulation" ~device:"TEST"
      ~has_local:false ~has_shared:false ~shared_max:0
      ~supports_dtype:(fun dtype -> dtype <> Dtype.int64 && dtype <> Dtype.uint64)
      ~render:(fun ?name:_ _ -> "") () in
  let values = [| 0L; 4294967295L; Int64.max_int; Int64.min_int;
                  -4294967296L; 4294967299L; 3L |] in
  let increment = 4294967299L and fallback = 8589934595L in
  let sentinel = 0xdeadbeef11223344L in
  let count = Array.length values in
  List.iter (fun (dtype, guarded, literal, op) ->
      let constant value =
        let weak = U.const (Const.int64 Dtype.weakint value) in
        match literal with
        | `Weak -> weak
        | `Cast -> U.cast ~src:weak ~dtype
        | `Typed -> U.const (Const.int64 dtype value) in
      let param slot = U.param ~slot ~dtype ~shape:(U.const_int count)
          ~addrspace:Dtype.Global () in
      let dst = param 0 and src = param 1 in
      let range = U.range ~size:(U.const_int count) ~axis:0 ~kind:Axis_type.Weak () in
      let index ptr i = U.index ~ptr ~idxs:[ i ] () in
      let load =
        if guarded then
          let offset = U.O.(range + int_ 1) in
          let cond = U.O.(range < int_ (Stdlib.pred count)) in
          let loaded = U.load ~src:(index src (U.valid ~src:offset ~cond)) () in
          U.alu_ternary ~op:Ops.Where ~a:cond ~b:loaded ~c:(constant fallback)
        else U.load ~src:(index src range) () in
      let value =
        let result = U.alu_binary ~op ~lhs:load ~rhs:(constant increment) in
        if Ops.Group.is_comparison op then
          U.alu_ternary ~op:Ops.Where ~a:result ~b:(constant fallback) ~c:load
        else result in
      let offset = if guarded then
          U.valid ~src:range ~cond:(U.O.ne range (U.const_int 0)) else range in
      let store = U.store ~dst:(index dst offset) ~value () in
      let sink = U.sink [ U.end_ ~value:store ~ranges:[ range ] ] in
      let decomposed = Decomp_dtype.do_dtype_decomps renderer sink in
      Spec.type_verify Spec.full_spec decomposed;
      is_false ~msg:"buffer emulation leaves no 64-bit integer operations"
        (List.exists (fun n -> U.dtype n = Dtype.int64 || U.dtype n = Dtype.uint64)
           (U.toposort decomposed));
      let program = Codegen_lower.lower (Device.renderer device) decomposed
          |> Linearizer.linearize in
      let spec = Device.compile_program device ~name:"emulated_long_buffer" program in
      let create values =
        let buffer = Device.create_buffer ~size:count ~dtype device in
        Device.Buffer.ensure_allocated buffer;
        let bytes = Bytes.create (8 * count) in
        Array.iteri (fun i value -> Bytes.set_int64_le bytes (8 * i) value) values;
        Device.Buffer.copyin buffer bytes;
        buffer in
      let input = create values and output = create (Array.make count sentinel) in
      run_spec device spec [ output; input ];
      let result = Device.Buffer.as_bytes output in
      let expected = List.init count (fun i ->
          if guarded && i = 0 then sentinel
          else
            let value = if not guarded then values.(i)
              else if i = count - 1 then fallback else values.(i + 1) in
            let cmp = if dtype = Dtype.uint64 then Int64.unsigned_compare value increment
              else Int64.compare value increment in
            match op with
            | Ops.Add -> Int64.add value increment
            | Ops.Sub -> Int64.sub value increment
            | Ops.Mul -> Int64.mul value increment
            | Ops.And -> Int64.logand value increment
            | Ops.Or -> Int64.logor value increment
            | Ops.Xor -> Int64.logxor value increment
            | Ops.Max -> if cmp < 0 then increment else value
            | Ops.Cmplt -> if cmp < 0 then fallback else value
            | Ops.Cmpeq -> if cmp = 0 then fallback else value
            | Ops.Cmpne -> if cmp <> 0 then fallback else value
            | _ -> assert false) in
      let literal_name = match literal with
        | `Weak -> "weak" | `Cast -> "cast" | `Typed -> "typed" in
      equal ~msg:(Printf.sprintf "%s %s guarded=%b literal=%s"
                    (Ops.name op) (Dtype.to_string dtype) guarded literal_name)
        (list int64) expected
        (List.init count (fun i -> Bytes.get_int64_le result (8 * i))))
    (List.concat_map (fun op ->
         List.concat_map (fun literal ->
             [ Dtype.int64, false, literal, op; Dtype.uint64, false, literal, op;
               Dtype.int64, true, literal, op; Dtype.uint64, true, literal, op ])
           [ `Typed; `Weak; `Cast ])
       [ Ops.Add; Ops.Sub; Ops.Mul; Ops.And; Ops.Or; Ops.Xor; Ops.Max;
         Ops.Cmplt; Ops.Cmpeq; Ops.Cmpne ])

let test_emulated_long_division () =
  let device = cpu "emulated-long-division" in
  let renderer = Renderer.make ~name:"emulation" ~device:"TEST"
      ~has_local:false ~has_shared:false ~shared_max:0
      ~supports_dtype:(fun dtype -> dtype <> Dtype.int64 && dtype <> Dtype.uint64)
      ~render:(fun ?name:_ _ -> "") () in
  let cases = [| Int64.min_int, 1L; Int64.min_int, 7L; -1L, 7L;
                 -4294967299L, 3L; Int64.max_int, 3L; Int64.max_int, -7L;
                 4294967299L, 4294967297L; 0L, -3L |] in
  let count = Array.length cases in
  List.iter (fun dtype ->
      let param slot size = U.param ~slot ~dtype ~shape:(U.const_int size)
          ~addrspace:Dtype.Global () in
      let dst = param 0 (2 * count) and lhs = param 1 count and rhs = param 2 count in
      let range = U.range ~size:(U.const_int count) ~axis:0 ~kind:Axis_type.Weak () in
      let index ptr offset = U.index ~ptr ~idxs:[ offset ] () in
      let a = U.load ~src:(index lhs range) () and b = U.load ~src:(index rhs range) () in
      let stores = List.mapi (fun i op ->
          let offset = U.const_int (i * count) in
          U.store ~dst:(index dst U.O.(range + offset))
            ~value:(U.alu_binary ~op ~lhs:a ~rhs:b) ()) [ Ops.Cdiv; Ops.Cmod ] in
      let sink = U.sink [ U.end_ ~value:(U.group stores) ~ranges:[ range ] ] in
      let decomposed = Decomp_dtype.do_dtype_decomps renderer sink in
      Spec.type_verify Spec.full_spec decomposed;
      is_false ~msg:"division and remainder leave no 64-bit operations"
        (List.exists (fun n -> U.dtype n = Dtype.int64 || U.dtype n = Dtype.uint64)
           (U.toposort decomposed));
      let program = Codegen_lower.lower (Device.renderer device) decomposed
          |> Linearizer.linearize in
      let spec = Device.compile_program device ~name:"emulated_long_division" program in
      let create values =
        let size = Array.length values in
        let buffer = Device.create_buffer ~size ~dtype device in
        Device.Buffer.ensure_allocated buffer;
        let bytes = Bytes.create (8 * size) in
        Array.iteri (fun i value -> Bytes.set_int64_le bytes (8 * i) value) values;
        Device.Buffer.copyin buffer bytes;
        buffer in
      let input = create (Array.map fst cases) and divisors = create (Array.map snd cases) in
      let output = create (Array.make (2 * count) 0L) in
      run_spec device spec [ output; input; divisors ];
      let bytes = Device.Buffer.as_bytes output in
      let div, rem = if dtype = Dtype.uint64 then Int64.unsigned_div, Int64.unsigned_rem
        else Int64.div, Int64.rem in
      Array.iteri (fun i (a, b) ->
          let msg = Printf.sprintf "%s %Ld / %Ld" (Dtype.to_string dtype) a b in
          equal ~msg int64 (div a b) (Bytes.get_int64_le bytes (8 * i));
          equal ~msg int64 (rem a b) (Bytes.get_int64_le bytes (8 * (i + count)))) cases)
    [ Dtype.int64; Dtype.uint64 ]

let test_emulated_fp8_loads () =
  let device = cpu "emulated-fp8-loads" in
  let renderer = Renderer.make ~name:"emulation" ~device:"TEST"
      ~has_local:false ~has_shared:false ~shared_max:0
      ~supports_dtype:(fun dtype -> not (Dtype.is_fp8 dtype))
      ~render:(fun ?name:_ _ -> "") () in
  List.iter (fun dtype ->
      let param slot dtype = U.param ~slot ~dtype ~shape:(U.const_int 256)
          ~addrspace:Dtype.Global () in
      let dst = param 0 Dtype.float32 and src = param 1 dtype in
      let range = U.range ~size:(U.const_int 256) ~axis:0 ~kind:Axis_type.Weak () in
      let index ptr = U.index ~ptr ~idxs:[ range ] () in
      let value = U.cast ~src:(U.load ~src:(index src) ()) ~dtype:Dtype.float32 in
      let sink = U.sink [ U.end_ ~value:(U.store ~dst:(index dst) ~value ())
                              ~ranges:[ range ] ] in
      let decomposed = Decomp_dtype.do_dtype_decomps renderer sink in
      Spec.type_verify Spec.full_spec decomposed;
      is_false ~msg:"FP8 arithmetic uses float32 even when float16 is supported"
        (List.exists (fun n -> Dtype.is_fp8 (U.dtype n) || U.dtype n = Dtype.float16)
           (U.toposort decomposed));
      let program = Codegen_lower.lower (Device.renderer device) decomposed
          |> Linearizer.linearize in
      let spec = Device.compile_program device ~name:"emulated_fp8_loads" program in
      let input = Device.create_buffer ~size:256 ~dtype device in
      let output = Device.create_buffer ~size:256 ~dtype:Dtype.float32 device in
      Device.Buffer.ensure_allocated input;
      Device.Buffer.ensure_allocated output;
      Device.Buffer.copyin input (Bytes.init 256 Char.chr);
      run_spec device spec [ output; input ];
      let bytes = Device.Buffer.as_bytes output in
      let exponent_bits, mantissa_bits = Dtype.finfo dtype in
      let fnuz = dtype = Dtype.fp8e4m3fnuz || dtype = Dtype.fp8e5m2fnuz in
      for bits = 0 to 255 do
        let exponent = (bits lsr mantissa_bits) land ((1 lsl exponent_bits) - 1) in
        (* The emulation contract flushes subnormals; the storage decoder
           preserves them. FNUZ's signed-zero bit pattern is NaN. *)
        let expected =
          if fnuz && bits = 128 then Float.nan
          else if exponent = 0 then (if bits land 128 = 0 then 0.0 else -0.0)
          else
            Nx_dtype.Scalar.decode (Option.get (Dtype.to_scalar dtype)) bits in
        let actual = Int32.float_of_bits (Bytes.get_int32_le bytes (4 * bits)) in
        let msg = Printf.sprintf "%s 0x%02x" (Dtype.to_string dtype) bits in
        if Float.is_nan expected then is_true ~msg (Float.is_nan actual)
        else equal ~msg int32 (Int32.bits_of_float expected) (Int32.bits_of_float actual)
      done)
    [ Dtype.fp8e4m3; Dtype.fp8e5m2; Dtype.fp8e4m3fnuz; Dtype.fp8e5m2fnuz ]

let test_emulated_compact_float_storage () =
  let device = cpu "emulated-compact-float-storage" in
  let fp8_cases = List.map (fun dtype ->
      let code = Nx_dtype.Scalar.encode (Option.get (Dtype.to_scalar dtype)) in
      dtype, List.map code [ 1.0; -1.0; 2.0; 0.5; 4.0 ], code 1.5, code 0.25)
      [ Dtype.fp8e4m3; Dtype.fp8e5m2; Dtype.fp8e4m3fnuz; Dtype.fp8e5m2fnuz ] in
  let cases =
    (Dtype.float16, [ 0x3c00; 0xbc00; 0x4000; 0x3800; 0x4400 ], 0x3e00, 0x3400)
    :: (Dtype.bfloat16, [ 0x3f80; 0xbf80; 0x4000; 0x3f00; 0x4080 ], 0x3fc0, 0x3e80)
    :: fp8_cases in
  List.iter (fun (dtype, bits, fallback, sentinel) ->
      let renderer = Renderer.make ~name:"emulation" ~device:"TEST"
          ~has_local:false ~has_shared:false ~shared_max:0
          ~supports_dtype:(fun dt -> dt <> dtype) ~render:(fun ?name:_ _ -> "") () in
      let input_count = List.length bits in
      let count = input_count + 1 in
      let word_size = Dtype.itemsize dtype in
      let uint = if word_size = 1 then Dtype.uint8 else Dtype.uint16 in
      let param slot dt size = U.param ~slot ~dtype:dt ~shape:(U.const_int size)
          ~addrspace:Dtype.Global () in
      let dst = param 0 Dtype.float32 count and src = param 1 dtype input_count in
      let roundtrip = param 2 dtype count and raw = param 3 uint count in
      let range = U.range ~size:(U.const_int count) ~axis:0 ~kind:Axis_type.Weak () in
      let index ptr offset = U.index ~ptr ~idxs:[ offset ] () in
      let cond = U.O.(range < int_ input_count) in
      let offset = U.valid ~src:range ~cond in
      let loaded = U.load ~src:(index src offset) () in
      let value = U.alu_ternary ~op:Ops.Where ~a:cond ~b:loaded
          ~c:(U.const (Const.float dtype 1.5)) in
      let guarded = U.valid ~src:range ~cond:(U.O.ne range (U.const_int 0)) in
      let stores =
        [ U.store ~dst:(index dst range) ~value:(U.cast ~src:value ~dtype:Dtype.float32) ();
          U.store ~dst:(index roundtrip guarded) ~value ();
          U.store ~dst:(index raw range)
            ~value:(U.bitcast ~src:loaded ~dtype:uint) () ] in
      let sink = U.sink [ U.end_ ~value:(U.group stores) ~ranges:[ range ] ] in
      let decomposed = Decomp_dtype.do_dtype_decomps renderer sink in
      Spec.type_verify Spec.full_spec decomposed;
      is_false ~msg:"compact-float accesses leave no unsupported dtype"
        (List.exists (fun n -> U.dtype n = dtype) (U.toposort decomposed));
      let program = Codegen_lower.lower (Device.renderer device) decomposed
          |> Linearizer.linearize in
      let spec = Device.compile_program device ~name:"emulated_compact_storage" program in
      let allocate dtype size =
        let buffer = Device.create_buffer ~size ~dtype device in
        Device.Buffer.ensure_allocated buffer;
        buffer in
      let encode bits =
        let bytes = Bytes.create (List.length bits * word_size) in
        List.iteri (fun i v ->
            if word_size = 1 then Bytes.set_uint8 bytes i v
            else Bytes.set_uint16_le bytes (2 * i) v) bits;
        bytes in
      let input = allocate dtype input_count and output = allocate Dtype.float32 count in
      let copied = allocate dtype count and bitcast = allocate uint count in
      Device.Buffer.copyin input (encode bits);
      Device.Buffer.copyin copied (encode (List.init count (fun _ -> sentinel)));
      run_spec device spec [ output; input; copied; bitcast ];
      let result = Device.Buffer.as_bytes output in
      equal ~msg:(Dtype.to_string dtype ^ " masked read with fallback") (list float_exact)
        [ 1.0; -1.0; 2.0; 0.5; 4.0; 1.5 ]
        (List.init count (fun i -> Int32.float_of_bits (Bytes.get_int32_le result (4 * i))));
      equal ~msg:(Dtype.to_string dtype ^ " masked write") string
        (Bytes.to_string (encode (sentinel :: List.tl bits @ [ fallback ])))
        (Bytes.to_string (Device.Buffer.as_bytes copied));
      equal ~msg:(Dtype.to_string dtype ^ " raw bitcast") string
        (Bytes.to_string (encode (bits @ [ 0 ]))) (Bytes.to_string (Device.Buffer.as_bytes bitcast))) cases

(* A bfloat16 load gated on data, whose fallback is a constant. On x86-64
   without AVX512-BF16 the compiler rounds the merged value back to bfloat16
   with a call to __truncsfbf2, which the loader must link, and which must
   give back every bit: NaNs with their payloads, a subnormal, both zeros. *)
let test_bfloat16_gated_load_keeps_bits () =
  let device = cpu "bfloat16-gated-load" in
  let bits = [ 0x7f81; 0xffc3; 0x0001; 0x8000; 0x7f80; 0x3f80 ] in
  let n = List.length bits and count = List.length bits + 2 in
  let param slot dtype size =
    U.param ~slot ~dtype ~shape:(U.const_int size) ~addrspace:Dtype.Global () in
  let dst = param 0 Dtype.bfloat16 count and src = param 1 Dtype.bfloat16 n in
  let flags = param 2 Dtype.int32 count in
  let range = U.range ~size:(U.const_int count) ~axis:0 ~kind:Axis_type.Weak () in
  let index ptr offset = U.index ~ptr ~idxs:[ offset ] () in
  let cond = U.O.ne (U.load ~src:(index flags range) ()) (U.const (Const.int Dtype.int32 0)) in
  let loaded = U.load ~src:(index src (U.valid ~src:range ~cond)) () in
  let value = U.alu_ternary ~op:Ops.Where ~a:cond ~b:loaded
      ~c:(U.const (Const.float Dtype.bfloat16 0.0)) in
  let store = U.store ~dst:(index dst range) ~value () in
  let sink = U.sink [ U.end_ ~value:store ~ranges:[ range ] ] in
  let program = Codegen_lower.lower (Device.renderer device) sink
      |> Linearizer.linearize in
  let spec = Device.compile_program device ~name:"bfloat16_gated_load" program in
  let allocate dtype size =
    let buffer = Device.create_buffer ~size ~dtype device in
    Device.Buffer.ensure_allocated buffer;
    buffer in
  let encode bits =
    let bytes = Bytes.create (2 * List.length bits) in
    List.iteri (fun i v -> Bytes.set_uint16_le bytes (2 * i) v) bits;
    bytes in
  let input = allocate Dtype.bfloat16 n and output = allocate Dtype.bfloat16 count in
  let gates = create_i32_buffer device (List.init count (fun i -> if i < n then 1 else 0)) in
  Device.Buffer.copyin input (encode bits);
  run_spec device spec [ output; input; gates ];
  let result = Device.Buffer.as_bytes output in
  equal (list int) (bits @ [ 0; 0 ])
    (List.init count (fun i -> Bytes.get_uint16_le result (2 * i)))

(* A kernel that calls __truncsfbf2 links it on every host, and it rounds
   float32 to bfloat16 to nearest, ties to even, keeping a NaN a NaN. *)
let test_truncsfbf2_rounds () =
  let device = cpu "truncsfbf2" in
  let renderer = Device.renderer device in
  let compiler = Option.get (Renderer.compiler renderer) in
  let cases =
    [ 0x3f800000, 0x3f80; 0x3f808000, 0x3f80; 0x3f818000, 0x3f82;
      0x3f808001, 0x3f81; 0xbf80ffff, 0xbf81; 0x7f7fffff, 0x7f80;
      0x7f800000, 0x7f80; 0xff800000, 0xff80; 0x7fc00001, 0x7fc1;
      0x7f800001, 0x7f81; 0x00000001, 0x0000; 0x80000000, 0x8000 ] in
  let count = List.length cases in
  let param slot dtype =
    U.param ~slot ~dtype ~shape:(U.const_int count) ~addrspace:Dtype.Global () in
  let dst = param 0 Dtype.bfloat16 and src = param 1 Dtype.float32 in
  let range = U.range ~size:(U.const (Const.int Dtype.int32 count)) ~axis:0
      ~kind:Axis_type.Loop ~dtype:Dtype.int32 () in
  let index ptr = U.index ~ptr ~idxs:[ range ] () in
  let body = U.custom_function ~name:"round_bf16"
      ~srcs:[ U.custom_inline ~fmt:"(unsigned long)&round_bf16" ~args:[]
                ~dtype:Dtype.uint64 ] in
  let info : U.call_info =
    { grad_fxn = None; name = None; precompile = false;
      precompile_backward = false; aux = None; dtype = Dtype.void } in
  let call = U.call ~body ~args:[ U.load ~src:(index src) (); index dst ] ~info in
  let sink = Linearizer.pm_add_control_flow
      (U.sink [ U.end_ ~value:call ~ranges:[ range ] ]) in
  let program = Linearizer.linearize sink in
  let helper = "__bf16 __truncsfbf2(float);\n\
                static void round_bf16(float x, __bf16 *out) { *out = __truncsfbf2(x); }" in
  let name = "truncsfbf2_rounds" in
  let source = helper ^ "\n" ^ Renderer.render renderer ~name program in
  let lib = Compiler.compile compiler source in
  let spec = Program_spec.of_program ~name ~src:source ~device:(Device.name device)
      ~lib program in
  let input = Device.create_buffer ~size:count ~dtype:Dtype.float32 device in
  let output = Device.create_buffer ~size:count ~dtype:Dtype.bfloat16 device in
  Device.Buffer.ensure_allocated input;
  Device.Buffer.ensure_allocated output;
  let bytes = Bytes.create (4 * count) in
  List.iteri (fun i (f, _) -> Bytes.set_int32_le bytes (4 * i) (Int32.of_int f)) cases;
  Device.Buffer.copyin input bytes;
  run_spec device spec [ output; input ];
  let result = Device.Buffer.as_bytes output in
  equal (list int) (List.map snd cases)
    (List.init count (fun i -> Bytes.get_uint16_le result (2 * i)))

let test_emulated_fp8_raw_bitcasts () =
  let device = cpu "fp8-raw-bitcasts" in
  let expected = Bytes.init 256 Char.chr in
  List.iter (fun dtype ->
      List.iter (fun (input_dtype, output_dtype) ->
          let param slot dtype = U.param ~slot ~dtype ~shape:(U.const_int 256)
              ~addrspace:Dtype.Global () in
          let output = param 0 output_dtype and input = param 1 input_dtype in
          let range = U.range ~size:(U.const_int 256) ~axis:0 ~kind:Axis_type.Weak () in
          let index ptr = U.index ~ptr ~idxs:[range] () in
          let value = U.bitcast ~src:(U.load ~src:(index input) ()) ~dtype:output_dtype in
          let sink = U.sink [U.end_ ~value:(U.store ~dst:(index output) ~value ())
              ~ranges:[range]] in
          let program = Codegen_lower.lower (Device.renderer device) sink
              |> Linearizer.linearize in
          let spec = Device.compile_program device ~name:"fp8_raw_bitcasts" program in
          let allocate dtype =
            let buffer = Device.create_buffer ~size:256 ~dtype device in
            Device.Buffer.ensure_allocated buffer;
            buffer in
          let input = allocate input_dtype and output = allocate output_dtype in
          Device.Buffer.copyin input expected;
          run_spec device spec [output; input];
          let actual = Device.Buffer.as_bytes output in
          for i = 0 to 255 do
            equal ~msg:(Printf.sprintf "%s -> %s encoding 0x%02x"
                (Dtype.to_string input_dtype) (Dtype.to_string output_dtype) i)
              int i (Bytes.get_uint8 actual i)
          done)
        [dtype, Dtype.uint8; Dtype.uint8, dtype])
    [Dtype.fp8e4m3; Dtype.fp8e5m2; Dtype.fp8e4m3fnuz; Dtype.fp8e5m2fnuz]

let test_software_sin_large_arguments () =
  let device = cpu "software-sin-large" in
  let values = [| 0.0; 1.0; Float.pi; 39800.0; 1.0e6; 1.0e10; 2.0 ** 31.0;
                  2.0 ** 32.0; 1.0e20; 1.0e30; 3.4028234663852886e38 |] in
  let values = Array.append values (Array.map Float.neg values)
      |> Array.map (fun x -> Int32.float_of_bits (Int32.bits_of_float x)) in
  let count = Array.length values in
  let param slot = U.param ~slot ~dtype:Dtype.float32 ~shape:(U.const_int count)
      ~addrspace:Dtype.Global () in
  let dst = param 0 and src = param 1 in
  let range = U.range ~size:(U.const_int count) ~axis:0 ~kind:Axis_type.Weak () in
  let index ptr = U.index ~ptr ~idxs:[ range ] () in
  let value = U.alu_unary ~op:Ops.Sin ~src:(U.load ~src:(index src) ()) in
  let sink = U.sink [ U.end_ ~value:(U.store ~dst:(index dst) ~value ())
                          ~ranges:[ range ] ] in
  let lowered = Helpers.Context_var.with_context
      [ Helpers.Context_var.B (Helpers.transcendental, 2) ]
      (fun () -> Codegen_lower.lower (Device.renderer device) sink) in
  is_false ~msg:"software sine has no native sine operation"
    (List.exists (fun n -> U.op n = Ops.Sin) (U.toposort lowered));
  let spec = Device.compile_program device ~name:"software_sin_large"
      (Linearizer.linearize lowered) in
  let input = Device.create_buffer ~size:count ~dtype:Dtype.float32 device in
  let output = Device.create_buffer ~size:count ~dtype:Dtype.float32 device in
  Device.Buffer.ensure_allocated input;
  Device.Buffer.ensure_allocated output;
  let bytes = Bytes.create (count * 4) in
  Array.iteri (fun i x -> Bytes.set_int32_le bytes (i * 4) (Int32.bits_of_float x)) values;
  Device.Buffer.copyin input bytes;
  run_spec device spec [ output; input ];
  let result = Device.Buffer.as_bytes output in
  Array.iteri (fun i x ->
      let expected = Float.sin x in
      let actual = Int32.float_of_bits (Bytes.get_int32_le result (i * 4)) in
      is_true ~msg:(Printf.sprintf "sin(%g): expected %.9g, got %.9g" x expected actual)
        (Float.abs (expected -. actual) < 2.0e-6)) values

let test_padded_reduction op transform values expected () =
  let device = cpu "padded-reduction" in
  let dtype = Dtype.float32 in
  let count = List.length values in
  let param slot size = U.param ~slot ~dtype ~shape:(U.const_int size) () in
  let output = param 0 1 and input = param 1 count in
  let range = U.range ~size:(U.const_int count) ~axis:0 ~kind:Axis_type.Reduce () in
  let loaded = U.load ~src:(U.index ~ptr:input ~idxs:[ range ] ()) () in
  let value = transform loaded in
  let reduced = U.reduce ~op ~src:value ~ranges:[ range ] in
  let dst = U.index ~ptr:output ~idxs:[ U.const_int 0 ] () in
  let kernel_info : U.kernel_info =
    { name = "padded_reduction";
      applied_opts = []; opts_to_apply = None; estimates = None; beam = 0 } in
  let sink = U.sink ~kernel_info [ U.store ~dst ~value:reduced () ] in
  let scheduler = Postrange.create sink (Device.renderer device) in
  ignore (Postrange.apply_opt scheduler (U.Opt.Padto { axis = 0; amount = 4 }));
  let lowered = Codegen.full_rewrite_to_sink ~optimize:false (Device.renderer device)
      (Postrange.ast scheduler) in
  let spec = Device.compile_program device ~name:"padded_reduction"
      (Linearizer.linearize lowered) in
  let buffer values =
    let buf = Device.create_buffer ~size:(List.length values) ~dtype device in
    let bytes = Bytes.create (List.length values * 4) in
    List.iteri (fun i v -> Bytes.set_int32_le bytes (i * 4) (Int32.bits_of_float v)) values;
    Device.Buffer.ensure_allocated buf;
    Device.Buffer.copyin buf bytes;
    buf in
  let output_buf = buffer [ nan ] and input_buf = buffer values in
  run_spec device spec [ output_buf; input_buf ];
  let actual = Int32.float_of_bits (Bytes.get_int32_le (Device.Buffer.as_bytes output_buf) 0) in
  is_true ~msg:(Printf.sprintf "expected %g, got %g" expected actual)
    (Float.abs (actual -. expected) < 1.e-6)

let test_zero_initialized_custom_storage () =
  let device = cpu "custom-bss" in
  let source = {|
    static volatile int zeroes[4];
    void read_zeroes(const unsigned long long *bufs, const long long *vals) {
      int *out = (int *)bufs[0];
      for (int i = 0; i < 4; i++) out[i] = zeroes[i] + (int)vals[0];
    }
  |} in
  let renderer = Device.renderer device in
  let lib = Compiler.compile (Option.get (Renderer.compiler renderer)) source in
  let elf = Elf.load lib in
  let bss = Option.get (Elf.find_section elf ".bss") in
  equal int 16 bss.size;
  is_true ~msg:"zero storage is separate from instructions" (bss.addr >= 16);
  let obj : Tiny_elf.t = {
    lib; name = "read_zeroes"; target = Renderer.target renderer; profile_key = None;
    signature = [
      {name = None; slot = 0; dtype = Dtype.int32; shape = [4]; addrspace = Dtype.Global};
      {name = None; slot = 1; dtype = Dtype.int32; shape = []; addrspace = Dtype.Alu}];
  } in
  let output = create_i32_buffer device [-1; -1; -1; -1] in
  let program = Device.runtime device obj in
  Fun.protect ~finally:program.free (fun () ->
      List.iter (fun value ->
          ignore (program.call [|output|] ~global:[|1; 1; 1|] ~local:None
              ~vals:[|Int64.of_int value|] ~wait:false ~timeout:None);
          equal (list int) (List.init 4 (fun _ -> value)) (read_i32_buffer output))
        [0; 7; -3])

let test_sparse_program_arguments () =
  let device = cpu "sparse-arguments" in
  let ptr slot = U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int 1) () in
  let output = ptr 3 and input = ptr 11 in
  let value = U.variable ~name:"increment" ~min_val:(-100) ~max_val:100
      ~dtype:Dtype.int32 () in
  let index ptr = U.index ~ptr ~idxs:[ U.const_int 0 ] () in
  let sum = U.alu_binary ~op:Ops.Add ~lhs:(U.load ~src:(index input) ()) ~rhs:value in
  let kernel_info : U.kernel_info =
    { name = "sparse_arguments";
      applied_opts = []; opts_to_apply = None; estimates = None; beam = 0 } in
  let sink = U.sink ~kernel_info [ U.store ~dst:(index output) ~value:sum () ] in
  let program = Codegen.to_program ~optimize:false device (Device.renderer device) sink in
  let bind values =
    let buffer = create_i32_buffer device values in
    U.from_buffer buffer, buffer in
  let output_node, output_buffer = bind [ 0 ] in
  let input_node, input_buffer = bind [ 41 ] in
  let unused = ptr 999 in
  let args = List.init 12 (function 3 -> output_node | 11 -> input_node | _ -> unused) in
  let info : U.call_info =
    { grad_fxn = None; name = None; precompile = false;
      precompile_backward = false; aux = None; dtype = Dtype.void } in
  let call = U.call ~body:program ~args ~info in
  Realize.run_linear ~device ~to_program:(fun device -> Codegen.to_program device (Device.renderer device))
    ~var_vals:[ "increment", 1L ] (U.linear [ call ]);
  equal (list int) [ 42 ] (read_i32_buffer output_buffer);
  equal (list int) [ 41 ] (read_i32_buffer input_buffer);
  let obj = U.to_elf program in
  let prg = Device.runtime device obj in
  Fun.protect ~finally:prg.free (fun () ->
      List.iter (fun (nbufs, nvals) ->
          raises_match (function Invalid_argument _ -> true | _ -> false)
            (fun () -> ignore (prg.call (Array.make nbufs input_buffer)
                ~global:[| 1; 1; 1 |] ~local:None ~vals:(Array.make nvals 0L)
                ~wait:false ~timeout:None)))
        [ 1, 1; 3, 1; 2, 0; 2, 2 ]);
  let bad = { obj with signature = List.map (fun (a : Tiny_elf.argument) ->
      { a with slot = 0 }) obj.signature } in
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Device.runtime device bad))

let test_linear_formal_order ?(permute_slots = false) ~reverse_buffers ~reverse_scalars () =
  let device = cpu (Printf.sprintf "formal-order-%b-%b" reverse_buffers reverse_scalars) in
  let param slot = U.param ~slot ~dtype:Dtype.int64 ~shape:(U.const_int 1) () in
  let output = param (if reverse_buffers then 7 else 2) in
  let input = param (if reverse_buffers then 2 else 7) in
  let small = U.variable ~param:true ~name:"z_small" ~min_val:0 ~max_val:16 ~dtype:Dtype.int32 () in
  let wide_value = 0x1_0000_0002 in
  let wide = U.variable ~param:true ~name:"a_wide" ~min_val:0 ~max_val:(wide_value + 10)
      ~dtype:Dtype.int64 () in
  let zero = U.const (Const.int Dtype.int32 0) in
  let index ptr = U.index ~ptr ~idxs:[ zero ] () in
  let src = index input and dst = index output in
  let loaded = U.load ~src () in
  let small64 = U.cast ~src:small ~dtype:Dtype.int64 in
  let scalar_sum = U.alu_binary ~op:Ops.Add ~lhs:small64 ~rhs:wide in
  let sum = U.alu_binary ~op:Ops.Add ~lhs:loaded ~rhs:scalar_sum in
  let scalars = if reverse_scalars then [ small; wide ] else [ wide; small ] in
  let linear = [ output; input ] @ scalars @
    [ zero; src; dst; loaded; small64; scalar_sum; sum; U.store ~dst ~value:sum () ] in
  let spec = Device.compile_program device ~name:"formal_order" linear in
  let buffer n =
    let b = Device.create_buffer ~size:1 ~dtype:Dtype.int64 device in
    Device.Buffer.ensure_allocated b;
    let bytes = Bytes.create 8 in
    Bytes.set_int64_le bytes 0 n;
    Device.Buffer.copyin b bytes;
    b in
  let out = buffer 0L and inp = buffer 5L in
  let slots = [ (if reverse_buffers then 7 else 2), out;
                (if reverse_buffers then 2 else 7), inp ] in
  let buffers = List.map (fun slot -> List.assoc slot slots) (Program_spec.globals spec) in
  if permute_slots then begin
    let obj = Program_spec.to_elf spec in
    let signature = List.map (fun (arg : Tiny_elf.argument) ->
        let slot = if arg.addrspace = Dtype.Alu then 5 - arg.slot else 1 - arg.slot in
        { arg with slot }) obj.signature in
    let prg = Device.runtime device { obj with signature } in
    Fun.protect ~finally:prg.free (fun () ->
        let vals = if reverse_scalars then [| Int64.of_int wide_value; 3L |]
          else [| 3L; Int64.of_int wide_value |] in
        ignore (prg.call [| inp; out |]
          ~global:[| 1; 1; 1 |] ~local:None ~vals ~wait:true ~timeout:None))
  end else begin
    Realize.run_linear ~device ~to_program ~wait:true
      ~var_vals:["z_small", 3L; "a_wide", Int64.of_int wide_value]
      (U.linear [program_call spec buffers])
  end;
  equal int64 (Int64.of_int (wide_value + 8)) (Bytes.get_int64_le (Device.Buffer.as_bytes out) 0);
  equal int64 5L (Bytes.get_int64_le (Device.Buffer.as_bytes inp) 0)

let test_store_gate_only_arguments () =
  let device = cpu "store-gate-arguments" in
  let ptr slot = U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int 1) () in
  let output = ptr 0 and input = ptr 1 in
  let threshold = U.variable ~name:"threshold" ~min_val:0 ~max_val:10
      ~dtype:Dtype.int32 () in
  let index ptr = U.index ~ptr ~idxs:[U.const_int 0] () in
  let gate = U.alu_binary ~op:Ops.Cmplt
      ~lhs:(U.load ~src:(index input) ()) ~rhs:threshold in
  let destination = U.index ~ptr:output
      ~idxs:[U.valid ~src:(U.const_int 0) ~cond:gate] () in
  let store = U.store ~dst:destination
      ~value:(U.const (Const.int Dtype.int32 42)) () in
  let kernel_info : U.kernel_info = {name = "store_gate_arguments";
      applied_opts = []; opts_to_apply = None; estimates = None; beam = 0} in
  let program = to_program device (U.sink ~kernel_info [store]) in
  let obj = U.to_elf program in
  equal int ~msg:"gate-only buffer and scalar remain in the signature" 3
    (List.length obj.signature);
  let output_buffer = create_i32_buffer device [-1] in
  let input_buffer = create_i32_buffer device [5] in
  let call = U.call ~body:program
      ~args:[U.from_buffer output_buffer; U.from_buffer input_buffer]
      ~info:U.{grad_fxn = None; name = None; precompile = false;
        precompile_backward = false; dtype = Dtype.void; aux = None} in
  Fun.protect ~finally:(fun () ->
      Device.Buffer.deallocate output_buffer;
      Device.Buffer.deallocate input_buffer) (fun () ->
      List.iter (fun (threshold, expected) ->
          Device.Buffer.copyin output_buffer (int32_to_bytes [-1]);
          Realize.run_linear ~device ~to_program ~wait:true
            ~var_vals:["threshold", threshold] (U.linear [call]);
          equal (list int) [expected] (read_i32_buffer output_buffer))
        [3L, -1; 8L, 42; 3L, -1])

let test_full_width_scalar_bindings () =
  let device = cpu "full-width-scalars" in
  let output = U.param ~slot:0 ~dtype:Dtype.int64 ~shape:(U.const_int 1) () in
  let value = U.param ~slot:1 ~name:"value" ~dtype:Dtype.int64
      ~addrspace:Dtype.Alu ~vmin_vmax:(Dtype.min Dtype.int64, Dtype.max Dtype.int64) () in
  let store = U.store ~dst:(U.index ~ptr:output ~idxs:[U.const_int 0] ()) ~value () in
  let info : U.kernel_info = {name = "full_width_scalars"; applied_opts = [];
      opts_to_apply = None; estimates = None; beam = 0} in
  let to_program device = Codegen.to_program ~optimize:false device (Device.renderer device) in
  let program = to_program device (U.sink ~kernel_info:info [store]) in
  let buffer = Device.create_buffer ~size:1 ~dtype:Dtype.int64 device in
  let call_info : U.call_info = {grad_fxn = None; name = None; precompile = false;
      precompile_backward = false; aux = None; dtype = Dtype.void} in
  let call = U.call ~body:program ~args:[U.from_buffer buffer] ~info:call_info in
  let linear = U.linear [call] in
  Fun.protect ~finally:(fun () -> Device.Buffer.deallocate buffer) (fun () ->
      List.iter (fun value ->
          Realize.run_linear ~device ~to_program ~var_vals:["value", value]
            ~wait:true ~jit:true linear;
          equal int64 value (Bytes.get_int64_le (Device.Buffer.as_bytes buffer) 0))
        [Int64.min_int; Int64.max_int; Int64.min_int])

let test_symbolic_stage_extents () =
  let device = cpu "symbolic-stage-extents" in
  let n = U.variable ~name:"stage_n" ~min_val:1 ~max_val:8 ~param:true () in
  let shape = U.stack [U.const_int 8; U.const_int 3] in
  let source = U.param ~slot:0 ~dtype:Dtype.int32 ~shape
      ~device:(U.Single (Device.name device)) () in
  let value = U.O.(source + U.cconst (Const.int Dtype.int32 1) Dtype.int32) in
  let active = U.shrink ~src:value
      ~offset:(U.stack [U.const_int 0; U.const_int 0])
      ~size:(U.stack [n; U.const_int 3]) in
  let intermediate = U.contiguous ~src:active () in
  let output = U.contiguous
      ~src:(U.reduce_axis ~src:intermediate ~op:Ops.Add ~axes:[1]) () in
  let graph = Rangeify.get_kernel_graph (U.sink [output]) in
  let nodes = U.toposort graph in
  let sizes = List.filter_map (fun node ->
      if U.op node = Ops.Alloc then Some (U.max_numel node) else None) nodes in
  equal (list int) [8; 24] (List.sort compare sizes);
  let kernels = List.filter_map (fun node ->
      Option.map (fun (call : U.call_view) -> call.body) (U.as_call node)) nodes in
  equal int 2 (List.length kernels);
  let programs = List.mapi (fun i body ->
      let kernel_info : U.kernel_info = {
        name = Printf.sprintf "symbolic_stage_%d" i;
        applied_opts = []; opts_to_apply = None; estimates = None; beam = 0} in
      to_program device (U.sink ~kernel_info (U.children body))) kernels in
  let input = create_i32_buffer device (List.init 24 Fun.id) in
  let stage = create_i32_buffer device (List.init 24 (fun _ -> -777)) in
  let result = create_i32_buffer device (List.init 8 (fun _ -> -777)) in
  let call program buffers =
    U.call ~body:program ~args:(List.map U.from_buffer buffers)
      ~info:U.{grad_fxn = None; name = None; precompile = false;
               precompile_backward = false; aux = None; dtype = Dtype.void} in
  let linear = match programs with
    | [write_stage; reduce_stage] ->
        U.linear [call write_stage [stage; input]; call reduce_stage [result; stage]]
    | _ -> fail "expected materialization and reduction kernels" in
  for active_rows = 1 to 8 do
    Device.Buffer.copyin stage (int32_to_bytes (List.init 24 (fun _ -> -777)));
    Device.Buffer.copyin result (int32_to_bytes (List.init 8 (fun _ -> -777)));
    Realize.run_linear ~device ~to_program ~wait:true ~jit:true
      ~var_vals:["stage_n", Int64.of_int active_rows] linear;
    equal (list int)
      (List.init 24 (fun i -> if i < 3 * active_rows then i + 1 else -777))
      (read_i32_buffer stage);
    equal (list int)
      (List.init 8 (fun i -> if i < active_rows then 9 * i + 6 else -777))
      (read_i32_buffer result)
  done

(* Borrowed buffers belong to the host device, which the tests open by name. *)
let () = Device.register "CPU" Tolk_cpu.create

let borrowed_storage () =
  let device = cpu "borrow" in
  let owner = create_i32_buffer device [ 1; 2; 3; 4 ] in
  let used = Device.Buffer.mem_used () in
  let borrowed =
    Device.Buffer.borrow ~size:4 ~dtype:Dtype.int32
      ~source:owner (Device.Buffer.addr owner)
  in
  let view = i32_view borrowed ~offset:4 ~size:2 in
  is_true (Device.Buffer.ownership owner = Device.Buffer.Owned);
  is_true (Device.Buffer.ownership borrowed = Device.Buffer.Borrowed);
  is_true (Device.Buffer.ownership view = Device.Buffer.Borrowed);
  equal int used (Device.Buffer.mem_used ());
  equal (list int) [ 2; 3 ] (read_i32_buffer view);
  let importer = cpu "borrow-importer" in
  equal nativeint
    (Nativeint.add (Device.Buffer.addr owner) 4n)
    (Device.Buffer.addr ~device:(Device.name importer) view);
  Device.Buffer.deallocate view;
  Device.Buffer.deallocate borrowed;
  equal int used (Device.Buffer.mem_used ());
  equal (list int) [ 1; 2; 3; 4 ] (read_i32_buffer owner)

(* The source of a borrowed buffer is collected only after the buffer. *)
let borrowed_source_lifetime () =
  let device = cpu "borrow-lifetime" in
  let collected = ref false in
  let[@inline never] make () =
    let owner = create_i32_buffer device [ 5; 6 ] in
    Gc.finalise (fun _ -> collected := true) owner;
    Device.Buffer.borrow ~size:2 ~dtype:Dtype.int32
      ~source:owner (Device.Buffer.addr owner)
  in
  let borrowed = ref (Some (make ())) in
  Gc.full_major ();
  is_false ~msg:"the source outlives no buffer over it" !collected;
  equal (list int) [ 5; 6 ] (read_i32_buffer (Option.get !borrowed));
  borrowed := None;
  Gc.full_major ();
  Gc.full_major ();
  is_true ~msg:"the source goes with the last buffer over it" !collected

let main () =
  run "Cpu_runtime"
    [
      group "Execution"
        [
          test "symbolic stages reserve maxima and execute active extents"
            test_symbolic_stage_extents;
          test "store gates retain their buffer and scalar arguments"
            test_store_gate_only_arguments;
          test "retained execution preserves both signed int64 endpoints"
            test_full_width_scalar_bindings;
          test "direct binding waits for foreign storage" direct_binding_waits_for_foreign_storage;
          test "padded exponential sum uses zero for extra lanes"
            (test_padded_reduction Ops.Add
               (fun x -> U.alu_unary ~op:Ops.Exp2 ~src:x) [ 0.; 1.; 2. ] 7.);
          test "padded negative maximum uses negative infinity"
            (test_padded_reduction Ops.Max Fun.id [ -2.; -3.; -4. ] (-2.));
          test "padded product uses one for extra lanes"
            (test_padded_reduction Ops.Mul Fun.id [ -2.; -3.; -4. ] (-24.));
          test "linear buffer formals retain their declaration order"
            (test_linear_formal_order ~reverse_buffers:true ~reverse_scalars:false);
          test "mixed-width scalar formals retain their declaration order"
            (test_linear_formal_order ~reverse_buffers:false ~reverse_scalars:true);
          test "binary signature slots select buffers and mixed-width scalars"
            (test_linear_formal_order ~permute_slots:true ~reverse_buffers:false ~reverse_scalars:true);
          test "custom kernels read zero-initialized ELF storage"
            test_zero_initialized_custom_storage;
          test "compiled signatures bind sparse arguments and reject wrong arities"
            test_sparse_program_arguments;
          test "software sine handles large arguments and word boundaries"
            test_software_sin_large_arguments;
          test "emulated compact-float storage preserves masks and bitcasts"
            test_emulated_compact_float_storage;
          test "emulated FP8 loads preserve all normal values" test_emulated_fp8_loads;
          test "emulated FP8 raw bitcasts preserve all byte encodings" test_emulated_fp8_raw_bitcasts;
          test "a bfloat16 gated load keeps every bit" test_bfloat16_gated_load_keeps_bits;
          test "__truncsfbf2 links and rounds to nearest even" test_truncsfbf2_rounds;
          test "emulated long division preserves quotient and remainder"
            test_emulated_long_division;
          test "emulated long buffer arithmetic preserves both words" test_emulated_long_buffer_arithmetic;
          test "emulated long casts preserve float64 precision" test_emulated_long_to_float64;
          test "split ranges with one root axis retain distinct lanes" test_split_axis_identity;
          test "compilation preserves the selected buffer alignment" test_compilation_preserves_alignment;
          test "same-named device replacement preserves compiled alignment"
            test_same_name_compilation_preserves_alignment;
          test "host calls preserve effects and loop arguments" test_host_calls;
          test "execute a conditional loop" test_conditional_loop;
          test "compile and run one kernel" (fun () ->
            let device = cpu "run-one" in
            let spec =
              Device.compile_program device ~name:"add_one"
                (increment_program ())
            in
            let dst = create_i32_buffer device [ 0 ] in
            let src = create_i32_buffer device [ 41 ] in
            run_spec device spec [ dst; src ];
            equal (list int) [ 42 ] (read_i32_buffer dst));
          test "exec is ordered" (fun () ->
            let device = cpu "ordered" in
            let spec =
              Device.compile_program device ~name:"ordered_add_one"
                (increment_program ())
            in
            let a = create_i32_buffer device [ 0 ] in
            let b = create_i32_buffer device [ 0 ] in
            run_spec device spec [ b; a ];
            run_spec device spec [ a; b ];
            equal (list int) [ 2 ] (read_i32_buffer a);
            equal (list int) [ 1 ] (read_i32_buffer b));
          test "core_id uses its explicit scalar value" (fun () ->
            let device = cpu "core-id" in
            let spec = Device.compile_program device ~name:"write_core_id"
                (core_id_program ()) in
            let dst = create_i32_buffer device [ 0; 0; 0; 0; 0; 0; 0; 0 ] in
            Realize.run_linear ~device ~to_program ~var_vals:["core_id", 5L]
              (U.linear [program_call spec [dst]]);
            equal (list int) [ 0; 0; 0; 0; 0; 5; 0; 0 ] (read_i32_buffer dst));
          test "untimed calls finish before returning" (fun () ->
            let device = cpu "synchronous" in
            let p = U.param ~slot:0 ~dtype:Dtype.int32 ~volatile:true
                ~shape:(U.const_int 1) () in
            let count = U.const (Const.int Dtype.int32 1_000_000) in
            let range = U.range ~axis:0 ~size:count
                ~kind:Axis_type.Weak ~dtype:Dtype.int32 () in
            let zero = U.const (Const.int Dtype.int32 0) in
            let dst = U.index ~ptr:p ~idxs:[ zero ] () in
            let store = U.store ~dst ~value:range () in
            let end_ = U.end_ ~value:store ~ranges:[ range ] in
            let spec = Device.compile_program device ~name:"write_until_done"
                [ p; zero; count; dst; range; store; end_ ] in
            let buf = create_i32_buffer device [ -1 ] in
            (* A separate device observes the memory without synchronizing the
               executing device, so copyout cannot hide an asynchronous call. *)
            let observer = cpu "synchronous-observer" in
            let options = { Device.Buffer_spec.default with
                external_ptr = Some (Device.Buffer.addr buf) } in
            let observed = Device.create_buffer ~size:1 ~dtype:Dtype.int32
                ~spec:options observer in
            Device.Buffer.ensure_allocated observed;
            let runtime = Device.runtime device (Program_spec.to_elf spec) in
            Fun.protect ~finally:runtime.free (fun () ->
                is_none (runtime.call [|buf|] ~global:[|1; 1; 1|] ~local:None
                  ~vals:[||] ~wait:false ~timeout:None));
            equal (list int) [ 999_999 ] (read_i32_buffer observed);
            ignore (Sys.opaque_identity buf));
          test "live executables survive device replacement and major collection"
            runtime_survives_owner_replacement;
          test "cache eviction materializes ones without beam search, stats, or capture"
            timing_cache_eviction;
          test "wait returns positive elapsed time" (fun () ->
            (* BEAM search selects kernels by this timing; [None] would collapse
               every candidate to infinity. Assert a real, positive measurement
               reaches the caller through the same path search.ml uses. *)
            let device = cpu "wait-timing" in
            let spec =
              Device.compile_program device ~name:"timed_add_one"
                (increment_program ())
            in
            let dst = create_i32_buffer device [ 0 ] in
            let src = create_i32_buffer device [ 41 ] in
            let elapsed = Realize.time_call ~device ~to_program
                (program_call spec [dst; src]) (fun sample -> sample ()) in
            is_true ~msg:"elapsed time is positive" (elapsed > 0.);
            Device.synchronize device;
            equal (list int) [ 42 ] (read_i32_buffer dst));
          test "external_ptr wraps caller memory zero-copy" (fun () ->
            let device = cpu "external-ptr" in
            (* Caller-owned backing storage: a normal buffer's memory, whose
               address stands in for memory tolk does not own (e.g. an nx
               buffer wrapped zero-copy by rune's jit). *)
            let backing = create_i32_buffer device [ 41 ] in
            let ptr = Device.Buffer.addr backing in
            let spec =
              { Device.Buffer_spec.default with external_ptr = Some ptr }
            in
            let external_ =
              Device.create_buffer ~size:1 ~dtype:Dtype.int32 ~spec device
            in
            Device.Buffer.ensure_allocated external_;
            (* alloc hands [ptr] back verbatim: no copy, no fresh allocation. *)
            is_true ~msg:"external buffer aliases caller memory"
              (Nativeint.equal ptr (Device.Buffer.addr external_));
            (* An in-place kernel over the external buffer mutates the caller's
               memory directly, visible through the backing buffer. *)
            let prog =
              Device.compile_program device ~name:"cpu_external_add_one"
                (increment_program ())
            in
            let runtime = Device.runtime device (Program_spec.to_elf prog) in
            Fun.protect ~finally:runtime.free (fun () ->
                ignore (runtime.call [|external_; external_|]
                  ~global:[|1; 1; 1|] ~local:None ~vals:[||]
                  ~wait:true ~timeout:None));
            equal (list int) [ 42 ] (read_i32_buffer backing);
            (* Freeing the external buffer must neither free nor cache the caller
               memory (LRU skip): the backing buffer stays valid afterwards. *)
            Device.Buffer.deallocate external_;
            equal (list int) [ 42 ] (read_i32_buffer backing));
          test "borrowed storage is neither owned nor counted" borrowed_storage;
          test "a borrowed buffer keeps its source reachable"
            borrowed_source_lifetime;
          test "buffer views copy at byte offsets" (fun () ->
            let device = cpu "views-copy" in
            let base = create_i32_buffer device [ 1; 2; 3; 4 ] in
            let view = i32_view base ~offset:4 ~size:2 in
            equal (list int) [ 2; 3 ] (read_i32_buffer view);
            Device.Buffer.copyin view (int32_to_bytes [ 20; 30 ]);
            equal (list int) [ 1; 20; 30; 4 ] (read_i32_buffer base));
          test "as_buffer aliases a view's bytes" (fun () ->
            let device = cpu "as-buffer" in
            let base = create_i32_buffer device [ 1; 2; 3; 4 ] in
            let view = i32_view base ~offset:4 ~size:2 in
            match Device.Buffer.as_buffer view with
            | None -> fail "CPU memory is host memory"
            | Some mem ->
                equal int 8 (Bigarray.Array1.dim mem);
                equal int 2 (Bigarray.Array1.get mem 0);
                Bigarray.Array1.set mem 4 30;
                equal (list int) [ 1; 2; 30; 4 ] (read_i32_buffer base));
          test "nested buffer views compose byte offsets" (fun () ->
            let device = cpu "nested-views" in
            let base = create_i32_buffer device [ 1; 2; 3; 4 ] in
            let mid = i32_view base ~offset:4 ~size:3 in
            let leaf = i32_view mid ~offset:4 ~size:1 in
            Device.Buffer.copyin leaf (int32_to_bytes [ 33 ]);
            equal (list int) [ 1; 2; 33; 4 ] (read_i32_buffer base));
          test "kernel dispatch binds buffer view offsets" (fun () ->
            let device = cpu "views-dispatch" in
            let spec =
              Device.compile_program device ~name:"cpu_view_add_one"
                (increment_program ())
            in
            let dst_base = create_i32_buffer device [ 0; 0; 0; 0 ] in
            let src_base = create_i32_buffer device [ 10; 41; 99; 100 ] in
            let dst = i32_view dst_base ~offset:4 ~size:1 in
            let src = i32_view src_base ~offset:4 ~size:1 in
            run_spec device spec [ dst; src ];
            equal (list int) [ 0; 42; 0; 0 ] (read_i32_buffer dst_base));
          test "oversized views are rejected" (fun () ->
            let device = cpu "view-bounds" in
            let base = create_i32_buffer device [ 1; 2; 3; 4 ] in
            raises (Invalid_argument "buffer view exceeds base buffer")
              (fun () ->
                ignore
                  (Device.Buffer.view base ~size:2 ~dtype:Dtype.int32
                     ~offset:12)));
          test "compile and run a cross-process imported kernel"
            imported_program_runs;
        ];
    ]

let () =
  match Sys.getenv_opt export_blob_var with
  | Some path -> export_child path
  | None -> exit (main ())
