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
      | Some buf -> U.from_buffer buf | None -> U.noop ~dtype:Dtype.void ()) in
  U.call ~body ~args ~info:U.{grad_fxn = None; name = None; precompile = false;
    precompile_backward = false; dtype = Dtype.void; aux = None}

let to_program device = Codegen.to_program ~optimize:false device (Device.renderer device)

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
  let sink = U.sink [ edge ] |> Codegen.full_rewrite_to_sink (Device.renderer device) in
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
  let reduced = U.reduce ~op ~src:value ~ranges:[ range ] ~dtype in
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
    ~var_vals:[ "increment", 1 ] (U.linear [ call ]);
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
      ~var_vals:["z_small", 3; "a_wide", wide_value]
      (U.linear [program_call spec buffers])
  end;
  equal int64 (Int64.of_int (wide_value + 8)) (Bytes.get_int64_le (Device.Buffer.as_bytes out) 0);
  equal int64 5L (Bytes.get_int64_le (Device.Buffer.as_bytes inp) 0)

let main () =
  run "Cpu_runtime"
    [
      group "Execution"
        [
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
          test "compiled signatures bind sparse arguments and reject wrong arities"
            test_sparse_program_arguments;
          test "software sine handles large arguments and word boundaries"
            test_software_sin_large_arguments;
          test "emulated compact-float storage preserves masks and bitcasts"
            test_emulated_compact_float_storage;
          test "emulated FP8 loads preserve all normal values" test_emulated_fp8_loads;
          test "emulated long division preserves quotient and remainder"
            test_emulated_long_division;
          test "emulated long buffer arithmetic preserves both words" test_emulated_long_buffer_arithmetic;
          test "emulated long casts preserve float64 precision" test_emulated_long_to_float64;
          test "split ranges with one root axis retain distinct lanes" test_split_axis_identity;
          test "compilation preserves the selected buffer alignment" test_compilation_preserves_alignment;
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
            Realize.run_linear ~device ~to_program ~var_vals:["core_id", 5]
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
  | None -> main ()
