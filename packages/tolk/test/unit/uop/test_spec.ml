(* Per-rule tests for Spec. *)

open Windtrap
open Tolk_uop

(* Helpers *)

let accepts = Spec.accepts

let rejected spec u =
  try Spec.type_verify spec u; false
  with Spec.Verification_failed failed -> ignore failed; true

let rejected_list spec us =
  try Spec.verify_list spec us; false
  with Spec.Verification_failed failed -> ignore failed; true

let with_env name value f =
  let old = Sys.getenv_opt name in
  Unix.putenv name value;
  match f () with
  | result ->
      (match old with
       | Some old_value -> Unix.putenv name old_value
       | None -> Unix.putenv name "");
      result
  | exception exn ->
      (match old with
       | Some old_value -> Unix.putenv name old_value
       | None -> Unix.putenv name "");
      raise exn

let global_i32_param ?(slot = 0) ?(size = 16) () =
  Uop.param ~slot ~dtype:Dtype.int32
    ~shape:(Uop.stack [ Uop.const_int size ]) ~addrspace:Dtype.Global ()

let global_i32_param_with_shape ?(slot = 0) shape =
  Uop.param ~slot ~dtype:Dtype.int32
    ~shape:(Uop.stack (List.map Uop.const_int shape)) ~addrspace:Dtype.Global ()

let i32 n = Uop.const (Const.int Dtype.int32 n)


let stack srcs =
  Uop.replace (Uop.const_int 0) ~op:Ops.Stack
    ~src:(Array.of_list srcs) ~arg:Uop.Arg.Empty ()

let load_with_gate ~idx ~alt ~gate =
  Uop.load ~src:idx ~alt ~gate ()

let store_with_gate ~idx ~value ~gate =
  Uop.store ~dst:idx ~value ~gate ()

let copy_current ~src ~device =
  Uop.copy ~src ~device ()

let call_info name : Uop.call_info =
  {
    Uop.grad_fxn = None;
    name = Some (Uop.Label name);
    precompile = false;
    precompile_backward = false;
    dtype = Dtype.void;
    aux = None;
  }

(* Shared spec *)

let sink_void () =
  is_true ~msg:"sink accepted" (accepts Spec.shared_spec (Uop.sink []))

let const_matching_dtype () =
  is_true ~msg:"const accepted" (accepts Spec.shared_spec (i32 42))

let param_with_param_arg () =
  let p = Uop.variable ~param:true ~name:"n" ~min_val:0 ~max_val:8 () in
  is_true ~msg:"Param with Param_arg accepted" (accepts Spec.shared_spec p)

let param_rejects_empty_arg () =
  raises (Invalid_argument "Uop: storage requires ParamArg") (fun () ->
      ignore (Uop.replace (Uop.variable ~name:"n" ~min_val:0 ~max_val:8 ())
        ~arg:Uop.Arg.Empty ()))

let buffer_with_param_arg () =
  let b =
    Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~addrspace:Dtype.Local ()
  in
  is_true ~msg:"Buffer with Param_arg accepted" (accepts Spec.shared_spec b)

let shared_rejects_global_buffer () =
  let b = Uop.buffer ~slot:0 ~dtype:Dtype.int32 () in
  is_true ~msg:"Global Buffer rejected by shared_spec"
    (rejected Spec.shared_spec b)

let buffer_rejects_empty_arg () =
  raises (Invalid_argument "Uop: storage requires ParamArg") (fun () ->
      ignore (Uop.replace
        (Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~addrspace:Dtype.Local ())
        ~arg:Uop.Arg.Empty ()))

let buffer_rejects_alu_addrspace () =
  let b =
    Uop.replace
      (Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~addrspace:Dtype.Local ())
      ~arg:
        (Uop.Arg.Param_arg
           { slot = 0; dtype = Dtype.int32; size = None; image = None; vmin_vmax = None;
             multiple_of = None; name = None; addrspace = Dtype.Alu;
             device = None; volatile = false; bind_on_realize = false; buffer = None; allocation = None })
      ()
  in
  is_true ~msg:"Buffer with ALU addrspace rejected"
    (rejected Spec.shared_spec b)

let empty_stack_void () =
  let s =
    Uop.replace (Uop.const_int 0) ~op:Ops.Stack
      ~src:[||] ~arg:Uop.Arg.Empty ()
  in
  is_true ~msg:"empty void stack accepted" (accepts Spec.shared_spec s)

let stack_sources_match () =
  let s = stack [ i32 1; i32 2 ] in
  is_true ~msg:"stack accepted" (accepts Spec.shared_spec s)

let stack_derives_child_dtype () =
  let s = stack [ i32 1; i32 2 ] in
  let f n = Uop.const (Const.float Dtype.float32 n) in
  let s = Uop.replace s ~src:[| f 1.; f 2. |] () in
  is_true ~msg:"source replacement derives the new stack dtype"
    (Dtype.equal (Uop.dtype s) Dtype.float32);
  Spec.type_verify Spec.shared_spec s

(* A weak child is fine — lowering commits it later — but two committed dtypes
   must agree. *)
let stack_rejects_mixed_dtype () =
  let f32 = Uop.const (Const.float Dtype.float32 2.0) in
  let s = stack [ i32 1; f32 ] in
  is_true ~msg:"stack children must share the stack dtype"
    (rejected Spec.shared_spec s);
  let weak = stack [ i32 1; Uop.const_int 2 ] in
  is_true ~msg:"a weak child is accepted" (accepts Spec.shared_spec weak)

let stack_rejects_mixed_child_counts () =
  let pair = stack [ i32 1; i32 2 ] in
  let s = stack [ pair; i32 3 ] in
  is_true ~msg:"mixed child vector counts rejected"
    (rejected Spec.shared_spec s)

let where_bool_cond () =
  let t = i32 1 and e = i32 2 in
  let c = Uop.const_bool true in
  let w = Uop.alu_ternary ~op:Ops.Where ~a:c ~b:t ~c:e in
  is_true ~msg:"valid where accepted" (accepts Spec.shared_spec w)

(* A WHERE over a non-bool condition has no dtype to derive, so it is
   refused where it is built rather than where it is checked. *)
let where_rejects_non_bool_cond () =
  let raised =
    try
      ignore (Uop.alu_ternary ~op:Ops.Where ~a:(i32 0) ~b:(i32 1) ~c:(i32 2));
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"non-bool cond rejected at construction" raised

let cmplt_is_bool () =
  let c = Uop.alu_binary ~op:Ops.Cmplt ~lhs:(i32 3) ~rhs:(i32 4) in
  is_true ~msg:"cmplt accepted" (accepts Spec.shared_spec c)

let cdiv_rejects_float () =
  let d =
    Uop.alu_binary ~op:Ops.Cdiv
      ~lhs:(Uop.const_float 3.0) ~rhs:(Uop.const_float 4.0)
  in
  is_true ~msg:"float cdiv rejected" (rejected Spec.shared_spec d)

let alu_operand_scalars_match () =
  let sum = Uop.alu_binary ~op:Ops.Add ~lhs:(i32 3) ~rhs:(i32 4) in
  is_true ~msg:"add accepted" (accepts Spec.shared_spec sum)

let shift_count_dtypes () =
  let shl ~rhs = Uop.alu_binary ~op:Ops.Shl ~lhs:(i32 8) ~rhs in
  is_true ~msg:"uint32 shift count accepted"
    (accepts Spec.shared_spec (shl ~rhs:(Uop.const (Const.int Dtype.uint32 2))));
  is_true ~msg:"same-dtype shift count accepted"
    (accepts Spec.shared_spec (Uop.alu_binary ~op:Ops.Shr ~lhs:(i32 8)
       ~rhs:(i32 2)));
  is_true ~msg:"weak shift count accepted"
    (accepts Spec.shared_spec
       (shl ~rhs:(Uop.const (Const.int Dtype.weakint 2))));
  is_true ~msg:"narrower shift count rejected"
    (rejected Spec.shared_spec
       (shl ~rhs:(Uop.const (Const.int Dtype.uint8 2))));
  let widened = shl ~rhs:(Uop.const (Const.int Dtype.uint32 2)) in
  is_true ~msg:"shift result must match the shifted operand"
    (Dtype.equal (Uop.dtype widened) Dtype.int32)

(* Bitwise and shift operands must be integral. *)
let bitwise_rejects_float_operands () =
  let f32 x = Uop.const (Const.float Dtype.float32 x) in
  let check op =
    is_true
      ~msg:(Ops.name op ^ " with a float operand rejected")
      (rejected Spec.shared_spec
         (Uop.alu_binary ~op ~lhs:(f32 1.0) ~rhs:(f32 2.0)))
  in
  List.iter check [ Ops.And; Ops.Or; Ops.Xor ];
  (* A shift takes its result dtype from the value being shifted, so a float
     operand leaves nothing to derive and is refused at construction. The spec
     keeps a rule for it too, which no constructible node can reach. *)
  let check_shift op =
    let raised =
      try
        ignore (Uop.alu_binary ~op ~lhs:(f32 1.0) ~rhs:(f32 2.0));
        false
      with Invalid_argument _ -> true
    in
    is_true
      ~msg:(Ops.name op ^ " with a float operand rejected at construction")
      raised
  in
  List.iter check_shift [ Ops.Shl; Ops.Shr ]

let index_accepts_integer_offsets () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  is_true ~msg:"int32 index accepted" (accepts Spec.shared_spec idx)

let index_rejects_gate_source () =
  let p = global_i32_param () in
  let idx0 = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let idx =
    Uop.replace idx0 ~src:[| p; i32 0; Uop.const_bool true |] ()
  in
  is_true ~msg:"gated Index rejected" (rejected Spec.shared_spec idx)

let special_accepts_raw_name () =
  let special =
    Uop.special ~name:"thread0" ~size:(Uop.const_int 8) ~dtype:Dtype.weakint ()
  in
  is_true ~msg:"weakint Special raw name is accepted"
    (accepts Spec.tensor_spec special)

let special_dtype_by_stage () =
  let idx_special =
    Uop.special ~name:"lidx0" ~size:(Uop.const_int 8) ~dtype:Dtype.weakint ()
  in
  is_true ~msg:"weakint Special accepted by tensor spec"
    (accepts Spec.tensor_spec idx_special);
  let i32_special =
    Uop.special ~name:"lidx0" ~size:(i32 8) ~dtype:Dtype.int32 ()
  in
  is_true ~msg:"int32 Special accepted by program spec"
    (accepts Spec.program_spec i32_special);
  let mismatch = Uop.replace idx_special ~src:[| i32 8 |] () in
  is_true ~msg:"Special result and size dtype must match"
    (rejected Spec.tensor_spec mismatch);
  is_true ~msg:"weakint Special rejected by program spec"
    (rejected Spec.program_spec idx_special)

let group_rejects_value_source () =
  let noop0 = Uop.noop () in
  let noop1 = Uop.noop () in
  let bad =
    Uop.replace (Uop.group [ noop0; noop1 ]) ~src:[| i32 1 |] ()
  in
  is_true ~msg:"Group source must be group/store/noop/unroll/ins"
    (rejected Spec.shared_spec bad)

let after_rejects_value_first_source () =
  let bad =
    Uop.replace (i32 1) ~op:Ops.After ~src:[| i32 1; Uop.noop () |]
      ()
  in
  is_true ~msg:"After first source must be orderable"
    (rejected Spec.shared_spec bad)

let end_rejects_non_range_tail () =
  let bad =
    Uop.replace (i32 1) ~op:Ops.End ~src:[| i32 1; i32 2 |] ()
  in
  is_true ~msg:"End tail sources must be ranges"
    (rejected Spec.shared_spec bad)

let range_rejects_bad_layouts () =
  let r =
    Uop.range ~size:(i32 4) ~axis:0 ~kind:Axis_type.Global
      ~dtype:Dtype.int32 ()
  in
  let missing_arg = Uop.replace r ~arg:Uop.Arg.Empty () in
  is_true ~msg:"Range requires Range_info"
    (rejected Spec.shared_spec missing_arg);
  is_true ~msg:"Range dtype must match size dtype"
    (Dtype.equal (Uop.dtype r) Dtype.int32)

let barrier_boundaries () =
  let barrier = Uop.barrier ~srcs:[ i32 1 ] () in
  is_true ~msg:"Barrier accepted in shared_spec"
    (accepts Spec.shared_spec barrier);
  is_true ~msg:"Barrier must be void"
    (Dtype.equal (Uop.dtype barrier) Dtype.void)

let group_after_bad_layouts () =
  let grouped =
    Uop.group [ Uop.noop (); Uop.noop () ]
  in
  is_true ~msg:"Group must be void"
    (Dtype.equal (Uop.dtype grouped) Dtype.void);
  let bad_group_src =
    Uop.replace grouped ~src:[| Uop.barrier () |] ()
  in
  is_true ~msg:"Group source must be group/store/noop/ins/end"
    (rejected Spec.shared_spec bad_group_src);
  raises (Invalid_argument "Uop: missing source for AFTER") (fun () ->
      ignore (Uop.replace (i32 1) ~op:Ops.After ~src:[||] ()))

(* Tensor spec *)

let tensor_accepts_global_buffer () =
  let shape = Uop.const_int 16 in
  let b =
    Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~shape
      ~device:(Uop.Single "CPU") ()
  in
  is_true ~msg:"Global Buffer accepted by tensor_spec"
    (accepts Spec.tensor_spec b)

let copy_matching_dtype () =
  let x = i32 1 in
  let c = copy_current ~src:x ~device:(Uop.Single "CPU") in
  is_true ~msg:"current copy accepted" (accepts Spec.tensor_spec c)

let copy_rejects_device_source_layout () =
  let src = i32 1 in
  let c =
    Uop.replace (Uop.copy ~src ~device:(Uop.Single "CPU") ())
      ~src:[| src; i32 0 |] ~arg:Uop.Arg.Empty ()
  in
  is_true ~msg:"copy with extra source rejected" (rejected Spec.tensor_spec c)

let copy_rejects_lowered_range_sources () =
  let src = i32 1 in
  let r =
    Uop.range ~size:(i32 4) ~axis:0 ~kind:Axis_type.Global
      ~dtype:Dtype.int32 ()
  in
  let c =
    Uop.replace (Uop.copy ~src ~device:(Uop.Single "CPU") ())
      ~src:[| src; r |] ()
  in
  is_true ~msg:"Copy must be eliminated before range indexing"
    (rejected Spec.tensor_spec c)

let copy_rejects_bad_device_or_dtype () =
  raises (Invalid_argument "Uop.copy: storage requires a concrete dtype")
    (fun () -> ignore (Uop.copy ~src:(Uop.const_int 1) ~device:(Uop.Single "CPU") ()));
  raises (Invalid_argument "Uop.copy: disk destinations require an explicit store")
    (fun () -> ignore (Uop.copy ~src:(i32 1) ~device:(Uop.Single "DISK:weights") ()));
  let src = i32 1 in
  let copy = Uop.copy ~src ~device:(Uop.Single "CPU") () in
  is_true ~msg:"Copy result dtype must match source"
    (Dtype.equal (Uop.dtype copy) Dtype.int32);
  List.iter (fun device ->
      let disk = Uop.replace copy ~arg:(Uop.Arg.Device device) () in
      is_true ~msg:"Copy spec rejects disk destinations"
        (rejected Spec.tensor_spec disk))
    [Uop.Single "DISK:weights"; Uop.Multi ["CPU"; "DISK:weights"]];
  let bad_index = Uop.copy ~src ~device:(Uop.Index 0) () in
  is_true ~msg:"Copy rejects positional device selector"
    (rejected Spec.tensor_spec bad_index);
  let bad_empty = Uop.copy ~src ~device:(Uop.Multi []) () in
  is_true ~msg:"Copy rejects empty multi-device target"
    (rejected Spec.tensor_spec bad_empty)

let call_reject_bad_layouts () =
  let info = call_info "f" in
  let arg = Uop.const_int 2 in
  let call = Uop.call ~body:(Uop.sink []) ~args:[ arg ] ~info in
  is_true ~msg:"opaque calls are valid in the shared spec"
    (accepts Spec.shared_spec call);
  let missing_info = Uop.replace call ~arg:Uop.Arg.Empty () in
  is_true ~msg:"Call requires Call_info"
    (rejected Spec.tensor_spec missing_info);
  let raw_value_body =
    Uop.replace call ~src:[| i32 1; arg |] ()
  in
  is_true ~msg:"Call rejects value-producing bodies"
    (rejected Spec.tensor_spec raw_value_body);
  raises (Invalid_argument "Uop.call: value-producing bodies require call_with_outputs")
    (fun () -> ignore (Uop.call ~body:(i32 1) ~args:[arg] ~info))

let call_source_contracts () =
  let info = call_info "f" in
  let call = Uop.call ~body:(Uop.custom_function ~name:"extern" ~srcs:[])
      ~args:[Uop.const_int 2] ~info in
  let outputs = Uop.call_with_outputs
      ~values:[i32 1; Uop.const (Const.float Dtype.float32 2.0)]
      ~args:[Uop.const_int 2] ~info () in
  is_true ~msg:"Call accepts opaque Custom_function bodies" (accepts Spec.tensor_spec call);
  List.iter (fun output -> Spec.type_verify Spec.tensor_spec output) outputs;
  let owners = List.map Uop.storage_base outputs in
  is_true ~msg:"each output has a separate allocation" (List.hd owners != List.nth owners 1)

let call_outputs_reject_invalid_positions () =
  let info = call_info "f" in
  List.iter (fun output_pos ->
      raises (Invalid_argument "Uop.call_with_outputs: invalid output positions")
        (fun () -> ignore (Uop.call_with_outputs ~output_pos ~values:[i32 1; i32 2]
          ~args:[i32 3] ~info ())))
    [[]; [1; 1]; [2; 1]; [-1; 2]; [1; 3]]

let reduce_arg_required () =
  let src =
    Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~shape:(Uop.stack [ Uop.const_int 4 ]) ()
  in
  let r = Uop.reduce_axis ~src ~op:Ops.Add ~axes:[ 0 ] in
  is_true ~msg:"tensor reduce accepted" (accepts Spec.tensor_spec r);
  let lowered =
    Uop.reduce ~src:(i32 1)
      ~ranges:[ Uop.range ~size:(Uop.const_int 4) ~axis:0 ~kind:Axis_type.Reduce () ]
      ~op:Ops.Add
  in
  is_true ~msg:"lowered reduce accepted by tensor spec"
    (accepts Spec.tensor_spec lowered)

let tensor_reduce_accepts_lowered_integer_tail () =
  let r =
    Uop.reduce ~src:(i32 1) ~ranges:[] ~op:Ops.Add
  in
  is_true ~msg:"tensor spec accepts axes-empty reduce"
    (accepts Spec.tensor_spec r);
  let lowered =
    Uop.reduce ~src:(i32 1)
      ~ranges:[ Uop.range ~size:(Uop.const_int 4) ~axis:0 ~kind:Axis_type.Reduce () ]
      ~op:Ops.Add
  in
  is_true ~msg:"tensor spec accepts lowered integer tail"
    (accepts Spec.tensor_spec lowered);
  let bad_tail =
    Uop.replace lowered ~src:[| i32 1; Uop.const_float 1.0 |] ()
  in
  is_true ~msg:"tensor reduce tail must be integer-valued"
    (rejected Spec.tensor_spec bad_tail)

let reduce_rejects_old_op_arg () =
  let src =
    Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~shape:(Uop.stack [ Uop.const_int 4 ]) ()
  in
  let r =
    Uop.replace
      (Uop.reduce_axis ~src ~op:Ops.Add ~axes:[ 0 ])
      ~arg:(Uop.Arg.Op Ops.Add) ()
  in
  is_true ~msg:"old Op arg reduce rejected" (rejected Spec.tensor_spec r)

let allreduce_layouts () =
  let src = i32 1 in
  let red =
    Uop.allreduce ~src ~device:(Uop.Multi [ "CPU"; "GPU" ]) ~op:Ops.Add
  in
  is_true ~msg:"Allreduce accepted" (accepts Spec.tensor_spec red);
  let missing_arg = Uop.replace red ~arg:Uop.Arg.Empty () in
  is_true ~msg:"Allreduce requires op/device arg"
    (rejected Spec.tensor_spec missing_arg);
  let bad_op =
    Uop.replace red ~arg:(Uop.Arg.Op_device (Ops.Cmplt, Uop.Single "CPU")) ()
  in
  is_true ~msg:"Allreduce rejects non-reduce op"
    (rejected Spec.tensor_spec bad_op)

let allreduce_rejects_bad_device_or_dtype () =
  let src = i32 1 in
  let red =
    Uop.allreduce ~src ~device:(Uop.Multi [ "CPU"; "GPU" ]) ~op:Ops.Add
  in
  is_true ~msg:"Allreduce result dtype must match source"
    (Dtype.equal (Uop.dtype red) Dtype.int32);
  let single = Uop.allreduce ~src ~device:(Uop.Single "CPU") ~op:Ops.Add in
  is_true ~msg:"Allreduce accepts a single destination device"
    (accepts Spec.tensor_spec single);
  let bad_index = Uop.allreduce ~src ~device:(Uop.Index 0) ~op:Ops.Add in
  is_true ~msg:"Allreduce rejects positional device selector"
    (rejected Spec.tensor_spec bad_index);
  let bad_empty = Uop.allreduce ~src ~device:(Uop.Multi []) ~op:Ops.Add in
  is_true ~msg:"Allreduce rejects empty multi-device group"
    (rejected Spec.tensor_spec bad_empty)

let multi_device_selection_layouts () =
  let shape = Uop.stack [ Uop.const_int 4 ] in
  let sharded =
    Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~shape ~axis:0
      ~device:(Uop.Multi [ "CPU"; "GPU" ]) ()
  in
  let multi = Uop.unshard ~src:sharded ~axes:[0] () in
  let selected = Uop.mselect ~src:multi ~index:1 in
  is_true ~msg:"Mselect accepts tuple-device source"
    (accepts Spec.tensor_spec selected);
  let out_of_range = Uop.mselect ~src:multi ~index:2 in
  is_true ~msg:"Mselect rejects out-of-range tuple-device index"
    (rejected Spec.tensor_spec out_of_range);
  let negative = Uop.mselect ~src:multi ~index:(-1) in
  is_true ~msg:"Mselect rejects negative tuple-device index"
    (rejected Spec.tensor_spec negative);
  is_true ~msg:"Mselect result dtype must match source"
    (Dtype.equal (Uop.dtype selected) Dtype.int32);
  let single =
    Uop.buffer ~slot:1 ~dtype:Dtype.int32 ~device:(Uop.Single "CPU") ()
  in
  let bad_single = Uop.mselect ~src:single ~index:0 in
  is_true ~msg:"Mselect rejects non-tuple device source"
    (rejected Spec.tensor_spec bad_single)

let multi_device_stack_layouts () =
  let cpu =
    Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~device:(Uop.Single "CPU") ()
  in
  let gpu =
    Uop.buffer ~slot:1 ~dtype:Dtype.int32 ~device:(Uop.Single "GPU") ()
  in
  let stacked = Uop.mstack [ cpu; gpu ] in
  is_true ~msg:"Mstack accepts single-device sources"
    (accepts Spec.tensor_spec stacked);
  let x = i32 1 in
  let repeated = Uop.mstack [ x; x ] in
  is_true ~msg:"Mstack accepts all-same device-less sources"
    (accepts Spec.tensor_spec repeated);
  let mixed_none = Uop.mstack [ i32 1; i32 2 ] in
  is_true ~msg:"Mstack rejects distinct device-less sources"
    (rejected Spec.tensor_spec mixed_none);
  let mixed_single_none = Uop.mstack [ cpu; i32 1 ] in
  is_true ~msg:"Mstack rejects mixed concrete and absent devices"
    (rejected Spec.tensor_spec mixed_single_none);
  let opts : Uop.stage_opts =
    { device = Some (Uop.Index 0); addrspace = Dtype.Global;
      removable = false }
  in
  let indexed = Uop.stage ~src:(i32 1) ~ranges:[] ~opts in
  let bad_indexed = Uop.mstack [ indexed ] in
  is_true ~msg:"Mstack rejects positional device selectors"
    (rejected Spec.tensor_spec bad_indexed);
  let multi_src =
    Uop.buffer ~slot:2 ~dtype:Dtype.int32 ~shape:(Uop.stack [ Uop.const_int 4 ])
      ~axis:0 ~device:(Uop.Multi [ "CPU"; "GPU" ]) ()
  in
  let bad_multi = Uop.mstack [ Uop.unshard ~src:multi_src ~axes:[0] () ] in
  is_true ~msg:"Mstack rejects already-multi sources"
    (rejected Spec.tensor_spec bad_multi);
  is_true ~msg:"Mstack result dtype must match sources"
    (Dtype.equal (Uop.dtype stacked) Dtype.int32)

let multi_device_multi_layouts () =
  let shape = Uop.stack [ Uop.const_int 4; Uop.const_int 4 ] in
  let sharded =
    Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~shape ~axis:1
      ~device:(Uop.Multi [ "CPU"; "GPU" ]) ()
  in
  let ok = Uop.unshard ~src:sharded ~axes:[1] () in
  is_true ~msg:"Multi accepts in-range sharding axis"
    (accepts Spec.tensor_spec ok);
  let negative = Uop.replace ok ~arg:(Uop.Arg.Ints [-1]) () in
  is_true ~msg:"Multi rejects negative sharding axis"
    (rejected Spec.tensor_spec negative);
  let out_of_range = Uop.unshard ~src:sharded ~axes:[2] () in
  is_true ~msg:"Multi rejects out-of-range sharding axis"
    (rejected Spec.tensor_spec out_of_range);
  is_true ~msg:"Multi result dtype must match source"
    (Dtype.equal (Uop.dtype ok) Dtype.int32);
  let unplaced = Uop.buffer ~slot:1 ~dtype:Dtype.int32 ~shape () in
  let no_device = Uop.replace ok ~src:[|unplaced|] () in
  is_true ~msg:"Multi rejects sources without multi-device placement"
    (rejected Spec.tensor_spec no_device);
  let empty_group =
    let base = Uop.buf_uop sharded in
    let param = Option.get (Uop.Arg.as_param_arg (Uop.arg base)) in
    Uop.replace base
      ~arg:(Uop.Arg.Param_arg { param with device = Some (Uop.Multi []) }) ()
  in
  let empty_multi = Uop.replace ok ~src:[|empty_group|] () in
  is_true ~msg:"Multi rejects empty multi-device placement"
    (rejected Spec.tensor_spec empty_multi)

let stage_rejects_bad_layouts () =
  let opts : Uop.stage_opts =
    { device = None; addrspace = Dtype.Global; removable = false }
  in
  let stage = Uop.stage ~src:(i32 1) ~ranges:[ Uop.const_int 4 ] ~opts in
  is_true ~msg:"Stage with integer ranges accepted"
    (accepts Spec.tensor_spec stage);
  let bad_range =
    Uop.stage ~src:(i32 1) ~ranges:[ Uop.const_float 4.0 ] ~opts
  in
  is_true ~msg:"Stage range sources must be integer-valued"
    (rejected Spec.tensor_spec bad_range);
  let bad_arg = Uop.replace stage ~arg:Uop.Arg.Empty () in
  is_true ~msg:"Indexed Stage requires placement options"
    (rejected Spec.tensor_spec bad_arg);
  let input = Uop.buffer ~slot:888 ~dtype:Dtype.float32 ~shape:(Uop.const_int 4)
      ~device:(Uop.Single "CPU") () in
  let bare = Uop.contiguous ~src:input ~force:true () in
  is_true ~msg:"bare materialization is a tensor Stage" (Uop.op bare = Ops.Stage);
  is_true ~msg:"bare Stage retains source placement" (Uop.device_of bare = Some (Uop.Single "CPU"));
  Spec.type_verify Spec.tensor_spec bare

let bind_accepts_variable_const () =
  let var =
    Uop.variable ~name:"n" ~min_val:0 ~max_val:4 ~dtype:Dtype.weakint ()
  in
  let b = Uop.bind ~var ~value:(Uop.const_int 3) in
  is_true ~msg:"scalar variable bound to weakint const accepted"
    (accepts Spec.tensor_spec b)

let bind_rejects_nonconstant_value () =
  let var = Uop.variable ~name:"shape" ~min_val:0 ~max_val:8 () in
  let value = stack [ Uop.const_int 1; Uop.const_int 2 ] in
  raises (Invalid_argument "Uop.bind: expected a constant value")
    (fun () -> ignore (Uop.bind ~var ~value))

let bind_rejects_parameter () =
  let var = Uop.variable ~param:true ~name:"formal" ~min_val:0 ~max_val:8 () in
  raises (Invalid_argument "Uop.bind: expected a variable")
    (fun () -> ignore (Uop.bind ~var ~value:(i32 3)))

let bind_accepts_64_bit_variable () =
  let var = Uop.variable ~name:"n" ~min_val:0 ~max_val:8 ~dtype:Dtype.int64 () in
  let b = Uop.bind ~var ~value:(Uop.const (Const.int Dtype.int64 3)) in
  is_true ~msg:"64-bit binding accepted" (accepts Spec.tensor_spec b);
  is_true ~msg:"bound literal is bare" (Uop.op (Option.get (Uop.as_bind b)).value = Ops.Const)

let bind_variable_supplies_dtype () =
  let var = Uop.variable ~name:"n" ~min_val:0 ~max_val:4 ~dtype:Dtype.int32 () in
  let b = Uop.bind ~var ~value:(Uop.const_int 3) in
  is_true ~msg:"variable supplies the literal width" (accepts Spec.tensor_spec b)

let movement_validates_shape_contracts () =
  let shape2 = Uop.stack [ Uop.const_int 2; Uop.const_int 4 ] in
  let tensor = Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~shape:shape2 () in
  let reshape_ok = Uop.reshape ~src:tensor ~shape:(Uop.const_int 8) in
  is_true ~msg:"valid Reshape accepted" (accepts Spec.tensor_spec reshape_ok);
  let reshape_bad = Uop.reshape ~src:tensor ~shape:(Uop.const_int 7) in
  is_true ~msg:"Reshape changing element count rejected"
    (rejected Spec.tensor_spec reshape_bad);
  let expand_src =
    Uop.buffer ~slot:1 ~dtype:Dtype.int32
      ~shape:(Uop.stack [ Uop.const_int 1; Uop.const_int 4 ]) ()
  in
  let expand_ok =
    Uop.broadcast_to ~src:expand_src ~shape:(Uop.stack [ Uop.const_int 2; Uop.const_int 4 ])
  in
  is_true ~msg:"valid Expand accepted" (accepts Spec.tensor_spec expand_ok);
  (* EXPAND only prepends leading axes, so its shape is always well-formed and
     the spec cannot reject it on shape grounds; an incompatible non-one
     dimension is rejected eagerly when broadcast_to builds the movement. *)
  is_true ~msg:"broadcast_to rejects incompatible non-one dimension"
    (try
       ignore
         (Uop.broadcast_to ~src:expand_src ~shape:(Uop.stack [ Uop.const_int 2; Uop.const_int 5 ]));
       false
     with Invalid_argument _ -> true);
  let pad_ok =
    Uop.pad ~src:tensor
      ~offset:(Uop.stack [ Uop.const_int 1; Uop.const_int 0 ])
      ~size:(Uop.stack [ Uop.const_int 4; Uop.const_int 4 ])
  in
  is_true ~msg:"valid Pad accepted" (accepts Spec.tensor_spec pad_ok);
  let pad_bad_shape =
    Uop.pad ~src:tensor ~offset:(Uop.const_int 0)
      ~size:(Uop.stack [ Uop.const_int 2; Uop.const_int 4 ])
  in
  is_true ~msg:"Pad offset/size shape mismatch rejected"
    (rejected Spec.tensor_spec pad_bad_shape);
  let pad_bad_bounds =
    Uop.pad ~src:tensor
      ~offset:(Uop.stack [ Uop.const_int 1; Uop.const_int 0 ])
      ~size:(Uop.stack [ Uop.const_int 2; Uop.const_int 4 ])
  in
  is_true ~msg:"Pad output size smaller than offset plus input rejected"
    (rejected Spec.tensor_spec pad_bad_bounds);
  let shrink_ok =
    Uop.shrink ~src:tensor
      ~offset:(Uop.stack [ Uop.const_int 1; Uop.const_int 0 ])
      ~size:(Uop.stack [ Uop.const_int 1; Uop.const_int 4 ])
  in
  is_true ~msg:"valid Shrink accepted" (accepts Spec.tensor_spec shrink_ok);
  let shrink_bad =
    Uop.shrink ~src:tensor
      ~offset:(Uop.stack [ Uop.const_int 1; Uop.const_int 0 ])
      ~size:(Uop.stack [ Uop.const_int 3; Uop.const_int 4 ])
  in
  is_true ~msg:"Shrink extending past input rejected"
    (rejected Spec.tensor_spec shrink_bad);
  let permute_bad = Uop.permute ~src:tensor ~order:[ 0; 0 ] in
  is_true ~msg:"invalid Permute rejected"
    (rejected Spec.tensor_spec permute_bad);
  let flip_bad = Uop.flip ~src:tensor ~dims:[ true ] in
  is_true ~msg:"invalid Flip rank rejected"
    (rejected Spec.tensor_spec flip_bad)

let full_spec_accepts_intermediate_forms () =
  let loose_after =
    Uop.replace (i32 1) ~op:Ops.After
      ~src:[| i32 1; Uop.const_float 0.0 |] ()
  in
  let loose_end =
    Uop.replace (i32 1) ~op:Ops.End ~src:[| i32 1; i32 2 |] ()
  in
  let loose_load =
    Uop.replace (i32 1) ~op:Ops.Load ~src:[| i32 1 |] ()
  in
  let loose_store =
    Uop.replace (Uop.noop ()) ~op:Ops.Store
      ~src:[| i32 1; Uop.const_float 2.0 |] ()
  in
  let value_index =
    let src =
      Uop.stack
        [
          Uop.variable ~name:"a" ~min_val:0 ~max_val:4 ();
          Uop.variable ~name:"b" ~min_val:0 ~max_val:4 ();
        ]
    in
    Uop.index ~ptr:src ~idxs:[ i32 1 ] ()
  in
  is_true ~msg:"full_spec accepts loose After intermediate"
    (accepts Spec.full_spec loose_after);
  is_true ~msg:"full_spec rejects malformed End payloads"
    (rejected Spec.full_spec loose_end);
  let special = Uop.special ~name:"gidx0" ~size:(i32 4) ~dtype:Dtype.int32 () in
  is_true ~msg:"full_spec accepts ranges replaced by hardware indices"
    (accepts Spec.full_spec
       (Uop.end_ ~value:(Uop.noop ()) ~ranges:[ special ]));
  is_true ~msg:"full_spec accepts transitional Load intermediate"
    (accepts Spec.full_spec loose_load);
  is_true ~msg:"full_spec accepts transitional Store intermediate"
    (accepts Spec.full_spec loose_store);
  is_true ~msg:"full_spec accepts value INDEX lane selection"
    (accepts Spec.full_spec value_index)

let tensor_rejects_if_endif () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let if_ = Uop.if_ ~cond:(Uop.const_bool true) ~idx_for_dedup:idx in
  let endif = Uop.endif ~if_ in
  is_true ~msg:"If rejected by tensor_spec" (rejected Spec.tensor_spec if_);
  is_true ~msg:"Endif rejected by tensor_spec"
    (rejected Spec.tensor_spec endif)

(* Program spec *)

let program_rejects_invalid_const () =
  is_true ~msg:"Invalid const rejected"
    (rejected Spec.program_spec (Uop.invalid ()))

let program_rejects_weakint () =
  Spec.type_verify Spec.program_spec (Uop.const_int 1);
  is_true ~msg:"weak computations are rejected in program_spec"
    (rejected Spec.program_spec Uop.O.(Uop.const_int 1 + Uop.const_int 2))

let program_buffer_rules () =
  let local =
    Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~addrspace:Dtype.Local ()
  in
  let reg =
    Uop.buffer ~slot:1 ~dtype:Dtype.int32 ~addrspace:Dtype.Reg ()
  in
  let global =
    Uop.buffer ~slot:2 ~dtype:Dtype.int32 ~shape:(Uop.const_int 16)
      ~device:(Uop.Single "CPU") ()
  in
  is_true ~msg:"Local Buffer accepted in program_spec"
    (accepts Spec.program_spec local);
  is_true ~msg:"Reg Buffer accepted in program_spec"
    (accepts Spec.program_spec reg);
  is_true ~msg:"Global Buffer rejected in program_spec"
    (rejected Spec.program_spec global)

let program_range_forms () =
  let int_range =
    Uop.range ~size:(i32 4) ~axis:0 ~kind:Axis_type.Global
      ~dtype:Dtype.int32 ()
  in
  let weak_range =
    Uop.range ~size:(Uop.const_int 4) ~axis:0 ~kind:Axis_type.Global ()
  in
  is_true ~msg:"int32 Range accepted in program_spec"
    (accepts Spec.program_spec int_range);
  is_true ~msg:"weakint Range rejected in program_spec"
    (rejected Spec.program_spec weak_range)

let program_rejects_tensor_only_ops () =
  let copy = Uop.copy ~src:(i32 1) ~device:(Uop.Single "CPU") () in
  let allreduce =
    Uop.allreduce ~src:(i32 1) ~device:(Uop.Single "CPU") ~op:Ops.Add
  in
  let reduce =
    Uop.reduce_axis
      ~src:(Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~shape:(Uop.stack [ Uop.const_int 4 ]) ())
      ~op:Ops.Add ~axes:[ 0 ]
  in
  let mstack = Uop.mstack [ i32 1; i32 1 ] in
  let mselect = Uop.mselect ~src:mstack ~index:0 in
  List.iter
    (fun u ->
      is_true ~msg:"tensor-only op rejected by program_spec"
        (not (accepts Spec.program_spec u)))
    [ copy; allreduce; reduce; mstack; mselect ]

let program_accepts_plain_load () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let ld = Uop.load ~src:idx () in
  Spec.verify_list Spec.program_spec [ idx; ld ];
  is_true ~msg:"plain load accepted" true

let program_accepts_cast_index_load () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let cast = Uop.cast ~src:idx ~dtype:(Uop.dtype idx) in
  let ld = Uop.load ~src:cast () in
  Spec.verify_list Spec.program_spec [ idx; cast; ld ];
  is_true ~msg:"Cast(Index) load accepted" true

let program_bitcast_index_same_dtype_is_plain_index () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let bitcast = Uop.bitcast ~src:idx ~dtype:(Uop.dtype idx) in
  is_true ~msg:"Bitcast(Index) to same dtype returns index" (bitcast == idx);
  let ld = Uop.load ~src:bitcast () in
  Spec.verify_list Spec.program_spec [ idx; ld ];
  is_true ~msg:"same-dtype Bitcast(Index) load accepted as Index" true

let program_rejects_real_bitcast_index_load () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let bitcast = Uop.bitcast ~src:idx ~dtype:Dtype.float32 in
  let ld = Uop.load ~src:bitcast () in
  is_true ~msg:"real Bitcast(Index) load rejected"
    (rejected_list Spec.program_spec [ idx; bitcast; ld ])

let program_accepts_load_gate_on_load () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let ld = load_with_gate ~idx ~alt:(i32 0) ~gate:(Uop.const_bool true) in
  Spec.verify_list Spec.program_spec [ idx; ld ];
  is_true ~msg:"load-gated alt load accepted" true

let program_rejects_load_gate_on_index () =
  let p = global_i32_param () in
  let idx0 = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let idx =
    Uop.replace idx0 ~src:[| p; i32 0; Uop.const_bool true |] ()
  in
  let ld =
    Uop.replace (Uop.load ~src:idx ())
      ~src:[| idx; i32 0 |] ()
  in
  is_true ~msg:"INDEX-gated alt load rejected"
    (rejected_list Spec.program_spec [ idx; ld ])

let program_accepts_plain_store () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let st = Uop.store ~dst:idx ~value:(i32 1) () in
  Spec.verify_list Spec.program_spec [ idx; st ];
  is_true ~msg:"plain store accepted" true;
  let value = Uop.alu_binary ~op:Ops.Add ~lhs:(i32 1) ~rhs:(i32 2) in
  is_true ~msg:"a computed value is not a storage destination"
    (rejected Spec.shared_spec (Uop.store ~dst:value ~value:(i32 3) ()))

let program_oob_disabled_accepts_out_of_bounds_load () =
  let p = global_i32_param ~size:16 () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 16)] () in
  let ld = Uop.load ~src:idx () in
  with_env "CHECK_OOB" "0" (fun () ->
      Spec.verify_list Spec.program_spec [ p; idx; ld ]);
  is_true ~msg:"CHECK_OOB=0 does not validate bounds" true

let program_oob_rejects_unproved_indices () =
  let p = global_i32_param ~size:16 () in
  let indices =
    [ Uop.bitcast ~src:(i32 16) ~dtype:Dtype.uint32;
      Uop.stack [ i32 0; i32 16 ] ]
  in
  with_env "CHECK_OOB" "1" (fun () ->
      List.iter (fun idx ->
          is_false (Validate.validate_index_source (Uop.index ~ptr:p ~idxs:[ idx ] ()))) indices;
      let shapeless = Uop.custom_inline ~fmt:"0" ~args:[] ~dtype:Dtype.int32 in
      is_false (Validate.validate_index_source
          (Uop.index ~ptr:shapeless ~idxs:[ i32 0 ] ()));
      let narrowed = Uop.cast ~src:(i32 128) ~dtype:Dtype.int8 in
      is_false (Validate.validate_index_source ~gate:(Uop.const_bool true)
          (Uop.index ~ptr:p ~idxs:[ narrowed ] ())))

let program_oob_enabled_rejects_out_of_bounds_load () =
  let p = global_i32_param ~size:16 () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 16)] () in
  let ld = Uop.load ~src:idx () in
  is_true ~msg:"CHECK_OOB=1 rejects proven out-of-bounds load"
    (with_env "CHECK_OOB" "1" (fun () ->
         rejected_list Spec.program_spec [ p; idx; ld ]))

let program_oob_enabled_accepts_minmax_in_bounds_load () =
  let p = global_i32_param ~size:16 () in
  let n =
    Uop.variable ~param:true ~name:"n" ~min_val:0 ~max_val:15
      ~dtype:Dtype.int32 ()
  in
  let idx = Uop.index ~ptr:p ~idxs:[n] () in
  let ld = Uop.load ~src:idx () in
  with_env "CHECK_OOB" "1" (fun () ->
      Spec.verify_list Spec.program_spec [ p; n; idx; ld ]);
  is_true ~msg:"CHECK_OOB=1 accepts min/max-proven in-bounds load" true

let program_oob_uses_explicit_buffer_shape () =
  let p = global_i32_param_with_shape [ 8 ] in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 9)] () in
  let ld = Uop.load ~src:idx () in
  is_true ~msg:"CHECK_OOB=1 validates against buffer max_numel"
    (with_env "CHECK_OOB" "1" (fun () ->
         rejected_list Spec.program_spec [ p; idx; ld ]))

let program_oob_image_pointer_bypasses_bounds () =
  let p =
    Uop.param ~slot:0 ~dtype:Dtype.float32
      ~image:(4, 4) ()
  in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 999)] () in
  let ld = Uop.load ~src:idx () in
  with_env "CHECK_OOB" "1" (fun () ->
      Spec.verify_list Spec.program_spec [ p; idx; ld ]);
  is_true ~msg:"CHECK_OOB=1 accepts image pointer accesses" true

let program_oob_false_gate_accepts_out_of_bounds_load () =
  let p = global_i32_param ~size:16 () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 16)] () in
  let ld = load_with_gate ~idx ~alt:(i32 0) ~gate:(Uop.const_bool false) in
  with_env "CHECK_OOB" "1" (fun () ->
      Spec.verify_list Spec.program_spec [ p; idx; ld ]);
  is_true ~msg:"CHECK_OOB=1 accepts statically false-gated load" true

let program_oob_symbolic_false_gate_accepts_out_of_bounds_load () =
  let p = global_i32_param ~size:16 () in
  let n =
    Uop.variable ~param:true ~name:"n" ~min_val:0 ~max_val:16
      ~dtype:Dtype.int32 ()
  in
  let gate = Uop.alu_binary ~op:Ops.Cmplt ~lhs:n ~rhs:(i32 0) in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 16)] () in
  let ld = load_with_gate ~idx ~alt:(i32 0) ~gate in
  with_env "CHECK_OOB" "1" (fun () ->
      Spec.verify_list Spec.program_spec [ p; n; gate; idx; ld ]);
  is_true ~msg:"CHECK_OOB=1 accepts max-zero symbolic gate" true

let program_oob_symbolic_store_remains_rejected () =
  let p = global_i32_param ~size:16 () in
  let n =
    Uop.variable ~param:true ~name:"n" ~min_val:(-1) ~max_val:16
      ~dtype:Dtype.int32 ()
  in
  let idx = Uop.index ~ptr:p ~idxs:[n] () in
  let st = Uop.store ~dst:idx ~value:(i32 1) () in
  is_true ~msg:"CHECK_OOB=1 rejects unmasked symbolic store"
    (with_env "CHECK_OOB" "1" (fun () ->
         rejected_list Spec.program_spec [ p; n; idx; st ]))

let program_oob_masked_symbolic_bounds_are_accepted () =
  let p = global_i32_param ~size:16 () in
  let n =
    Uop.variable ~param:true ~name:"n" ~min_val:(-1) ~max_val:16
      ~dtype:Dtype.int32 ()
  in
  let ge_zero = Uop.alu_binary ~op:Ops.Cmplt ~lhs:(i32 (-1)) ~rhs:n in
  let lt_size = Uop.alu_binary ~op:Ops.Cmplt ~lhs:n ~rhs:(i32 16) in
  let gate = Uop.alu_binary ~op:Ops.And ~lhs:ge_zero ~rhs:lt_size in
  let idx = Uop.index ~ptr:p ~idxs:[n] () in
  let ld = load_with_gate ~idx ~alt:(i32 0) ~gate in
  with_env "CHECK_OOB" "1" (fun () ->
      Spec.verify_list Spec.program_spec
        [ p; n; ge_zero; lt_size; gate; idx; ld ]);
  is_true ~msg:"CHECK_OOB=1 accepts mask-proven symbolic bounds" true

let program_oob_masked_symbolic_lower_bound_only_rejected () =
  let p = global_i32_param ~size:16 () in
  let n =
    Uop.variable ~param:true ~name:"n" ~min_val:(-1) ~max_val:16
      ~dtype:Dtype.int32 ()
  in
  let gate = Uop.alu_binary ~op:Ops.Cmplt ~lhs:(i32 (-1)) ~rhs:n in
  let idx = Uop.index ~ptr:p ~idxs:[n] () in
  let ld = load_with_gate ~idx ~alt:(i32 0) ~gate in
  is_true ~msg:"CHECK_OOB=1 rejects incomplete masked symbolic bounds"
    (with_env "CHECK_OOB" "1" (fun () ->
         rejected_list Spec.program_spec [ p; n; gate; idx; ld ]))

let masked_access_accepted ~size ~index ~gate =
  let p = global_i32_param ~size () in
  let idx = Uop.index ~ptr:p ~idxs:[ index ] () in
  let ld = load_with_gate ~idx ~alt:(i32 0) ~gate in
  with_env "CHECK_OOB" "1" (fun () -> not (rejected Spec.program_spec ld))

let program_oob_loaded_component_bounds () =
  let rows = global_i32_param ~slot:1 ~size:1 () in
  let row = Uop.load ~src:(Uop.index ~ptr:rows ~idxs:[ i32 0 ] ()) () in
  let col = Uop.variable ~param:true ~name:"col" ~min_val:0 ~max_val:2
      ~dtype:Dtype.int32 () in
  let index = Uop.O.(row * i32 3 + col) in
  let lower = Uop.O.(i32 (-1) < row) and upper = Uop.O.(row < i32 4) in
  let gate = Uop.alu_binary ~op:Ops.And ~lhs:lower ~rhs:upper in
  is_true ~msg:"guard bounds the loaded row before stride multiplication"
    (masked_access_accepted ~size:12 ~index ~gate);
  is_false ~msg:"missing lower bound must not prove the loaded row safe"
    (masked_access_accepted ~size:12 ~index ~gate:upper);
  is_false ~msg:"last row still exceeds a shorter buffer"
    (masked_access_accepted ~size:11 ~index ~gate)

let program_oob_offset_component_bounds () =
  let x = Uop.variable ~param:true ~name:"scan" ~min_val:0 ~max_val:1019
      ~dtype:Dtype.int32 () in
  let index = Uop.O.(x + i32 (-511)) in
  is_true ~msg:"scan mask proves a nonnegative offset"
    (masked_access_accepted ~size:512 ~index ~gate:Uop.O.(i32 510 < x));
  is_false ~msg:"one smaller guard permits index minus one"
    (masked_access_accepted ~size:512 ~index ~gate:Uop.O.(i32 509 < x))

let program_oob_shift_component_bounds () =
  let r = Uop.variable ~param:true ~name:"selection" ~min_val:0 ~max_val:7
      ~dtype:Dtype.int32 () in
  let half = Uop.alu_binary ~op:Ops.Shr ~lhs:Uop.O.(r + i32 1) ~rhs:(i32 1) in
  let index = Uop.O.(half + i32 (-3)) in
  is_true ~msg:"selection guard bounds the shifted component"
    (masked_access_accepted ~size:4 ~index ~gate:Uop.O.(i32 4 < r));
  is_false ~msg:"weaker selection guard permits a negative index"
    (masked_access_accepted ~size:4 ~index ~gate:Uop.O.(i32 2 < r))

let program_oob_committed_guard_constants () =
  let x = Uop.variable ~param:true ~name:"guard_constant" ~min_val:0 ~max_val:31
      ~dtype:Dtype.int32 () in
  let upper = Uop.const (Const.integer Dtype.int32 (Z.of_string "-2147483649")) in
  is_false ~msg:"an overflowing committed guard constant is not its weak payload"
    (masked_access_accepted ~size:16 ~index:x ~gate:Uop.O.(x < upper));
  let lower = Uop.const (Const.integer Dtype.int32 (Z.of_string "4294967295")) in
  is_false ~msg:"component proof must preserve overflowing guard casts"
    (masked_access_accepted ~size:1 ~index:Uop.O.(x + i32 (-31))
       ~gate:Uop.O.(lower < x))

let program_oob_proof_variables_are_fresh () =
  let x = Uop.variable ~param:true ~name:"x" ~min_val:0 ~max_val:6
      ~dtype:Dtype.int32 () in
  let user = Uop.param ~slot:(-1) ~name:"fake0" ~dtype:Dtype.int32
      ~shape:(Uop.stack []) ~vmin_vmax:(Bound.int 0, Bound.int 3)
      ~multiple_of:1 ~addrspace:Dtype.Alu () in
  is_false ~msg:"a proof variable must not alias an unrelated input parameter"
    (masked_access_accepted ~size:1 ~index:Uop.O.(x + user * i32 (-1))
       ~gate:Uop.O.(x < i32 4))

let program_oob_component_bounds_preserve_casts () =
  let x = Uop.variable ~param:true ~name:"narrow" ~min_val:0 ~max_val:255
      ~dtype:Dtype.int32 () in
  let index = Uop.cast ~src:x ~dtype:Dtype.int8 in
  is_false ~msg:"a valid source interval does not remove narrowing wraparound"
    (masked_access_accepted ~size:256 ~index ~gate:Uop.O.(i32 127 < x));
  let y = Uop.variable ~param:true ~name:"wrap" ~min_val:0 ~max_val:3
      ~dtype:Dtype.int32 () in
  let index = Uop.cast ~src:Uop.O.(y + i32 127) ~dtype:Dtype.int8 in
  is_false ~msg:"component bounds preserve narrowing after arithmetic"
    (masked_access_accepted ~size:256 ~index ~gate:Uop.O.(i32 0 < y))

let program_rejects_nested_casted_index_source () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let cast1 =
    Uop.replace idx ~op:Ops.Cast ~src:[| idx |] ~arg:(Uop.Arg.Dtype Dtype.int32) ()
  in
  let cast2 =
    Uop.replace cast1 ~op:Ops.Cast ~src:[| cast1 |] ~arg:(Uop.Arg.Dtype Dtype.int32) ()
  in
  let ld = Uop.load ~src:cast2 () in
  is_true ~msg:"shared spec accepts only one cast around INDEX/SHRINK"
    (rejected_list Spec.program_spec [ p; idx; cast1; cast2; ld ])

let program_accepts_store_gate_on_store () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let st = store_with_gate ~idx ~value:(i32 1) ~gate:(Uop.const_bool true) in
  Spec.verify_list Spec.program_spec [ idx; st ];
  is_true ~msg:"store-gated store accepted" true

let program_rejects_store_gate_on_index () =
  let p = global_i32_param () in
  let idx0 = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let idx =
    Uop.replace idx0 ~src:[| p; i32 0; Uop.const_bool true |] ()
  in
  let st = Uop.store ~dst:idx ~value:(i32 1) () in
  is_true ~msg:"INDEX-gated store rejected"
    (rejected_list Spec.program_spec [ idx; st ])

let program_accepts_if_with_shrink_index () =
  let p = global_i32_param () in
  let sh = Uop.shrink ~src:p ~offset:(i32 0) ~size:(i32 1) in
  let if_ = Uop.if_ ~cond:(Uop.const_bool true) ~idx_for_dedup:sh in
  is_true ~msg:"If accepts Shrink as index source"
    (accepts Spec.program_spec if_)

let program_control_flow_boundaries () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let cast_idx = Uop.cast ~src:idx ~dtype:Dtype.int64 in
  let if_index = Uop.if_ ~cond:(Uop.const_bool true) ~idx_for_dedup:idx in
  let if_cast =
    Uop.if_ ~cond:(Uop.const_bool true) ~idx_for_dedup:cast_idx
  in
  let endif = Uop.endif ~if_:if_index in
  is_true ~msg:"If(Index) is program-stage"
    (accepts Spec.program_spec if_index);
  is_true ~msg:"If(Cast) is program-stage"
    (accepts Spec.program_spec if_cast);
  is_true ~msg:"Endif closes If in program_spec"
    (accepts Spec.program_spec endif);
  is_true ~msg:"If is rejected by tensor_spec"
    (rejected Spec.tensor_spec if_index);
  is_true ~msg:"Endif is rejected by tensor_spec"
    (rejected Spec.tensor_spec endif);
  is_true ~msg:"If remains accepted by full_spec through program_spec"
    (accepts Spec.full_spec if_index);
  is_true ~msg:"Endif remains accepted by full_spec through program_spec"
    (accepts Spec.full_spec endif)

let program_rejects_bad_if_layouts () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let valid = Uop.if_ ~cond:(Uop.const_bool true) ~idx_for_dedup:idx in
  let non_bool_cond = Uop.if_ ~cond:(i32 1) ~idx_for_dedup:idx in
  let value_dedup =
    Uop.if_ ~cond:(Uop.const_bool true) ~idx_for_dedup:(Uop.const_int 0)
  in
  let missing_dedup =
    Uop.replace valid ~src:[| Uop.const_bool true |] ()
  in
  let extra_src =
    Uop.replace valid ~src:[| Uop.const_bool true; idx; idx |] ()
  in
  List.iter
    (fun u ->
      is_true ~msg:"malformed If layout rejected by program_spec"
        (rejected Spec.program_spec u))
    [ non_bool_cond; value_dedup; missing_dedup; extra_src ]

let program_rejects_loose_after_layout () =
  let bad =
    Uop.replace (i32 1) ~op:Ops.After
      ~src:[| i32 1; Uop.noop () |] ()
  in
  is_true ~msg:"program_spec rejects loose value-first After"
    (rejected Spec.program_spec bad)

let program_rejects_if_dedup_source_matrix () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let bad_bitcast = Uop.bitcast ~src:idx ~dtype:Dtype.float32 in
  let bad_after = Uop.after ~src:idx ~deps:[ Uop.noop () ] in
  let bad_buffer =
    Uop.buffer ~slot:0 ~dtype:Dtype.int32 ~addrspace:Dtype.Local ()
  in
  let bad_load = Uop.load ~src:idx () in
  List.iter
    (fun dedup ->
      let if_ = Uop.if_ ~cond:(Uop.const_bool true) ~idx_for_dedup:dedup in
      is_true ~msg:"If dedup source must be Cast, Index, or Shrink"
        (rejected Spec.program_spec if_))
    [ bad_bitcast; bad_after; bad_buffer; bad_load ]

let program_rejects_bad_endif_layouts () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let if_ = Uop.if_ ~cond:(Uop.const_bool true) ~idx_for_dedup:idx in
  let endif = Uop.endif ~if_ in
  let non_if_src =
    Uop.replace endif ~src:[| Uop.const_bool true |] ()
  in
  let extra_src = Uop.replace endif ~src:[| if_; if_ |] () in
  List.iter
    (fun u ->
      is_true ~msg:"malformed Endif layout rejected by program_spec"
        (rejected Spec.program_spec u))
    [ non_if_src; extra_src ]

let program_end_range_boundaries () =
  let int_range =
    Uop.range ~size:(i32 4) ~axis:0 ~kind:Axis_type.Global
      ~dtype:Dtype.int32 ()
  in
  let weak_range =
    Uop.range ~size:(Uop.const_int 4) ~axis:0 ~kind:Axis_type.Global ()
  in
  let body = Uop.noop () in
  let closed = Uop.end_ ~value:body ~ranges:[ int_range ] in
  let weak_closed = Uop.end_ ~value:body ~ranges:[ weak_range ] in
  let bad_tail =
    Uop.replace closed ~src:[| body; i32 2 |] ()
  in
  is_true ~msg:"End with int32 range accepted in program_spec"
    (accepts Spec.program_spec closed);
  is_true ~msg:"End with weakint dtype rejected in program_spec"
    (rejected Spec.program_spec weak_closed);
  is_true ~msg:"End with non-range tail rejected in program_spec"
    (rejected Spec.program_spec bad_tail)

let verify_list_validates_flat_program () =
  let p = global_i32_param () in
  let idx = Uop.index ~ptr:p ~idxs:[(i32 0)] () in
  let ld = Uop.load ~src:idx () in
  Spec.verify_list Spec.program_spec [ i32 0; idx; ld ];
  is_true ~msg:"verify_list rejects supplied invalid node"
    (rejected_list Spec.program_spec [ Uop.O.(Uop.const_int 1 + Uop.const_int 2) ])

(* Full spec *)

let full_spec_has_no_catch_all () =
  let unknown = Uop.replace (Uop.noop ())
      ~op:Ops.Wait () in
  is_true ~msg:"full_spec rejects unrecognised node"
    (rejected Spec.full_spec unknown)

let typed_host_call_contract () =
  let range = Uop.range ~size:(i32 4) ~axis:0 ~kind:Axis_type.Loop
      ~dtype:Dtype.int32 () in
  let body = Uop.custom_function ~name:"external"
      ~srcs:[ Uop.const (Const.int Dtype.uint64 0) ] in
  let info = { (call_info "external") with dtype = Dtype.int32 } in
  let buffer = global_i32_param () in
  let opaque = Uop.call ~body ~args:[ buffer ]
      ~info:{ info with dtype = Dtype.void } in
  let program_info = Uop.program_info_from_sink (Uop.sink [ opaque ]) in
  equal (list int) [ 0 ] program_info.ins;
  equal (list int) [ 0 ] program_info.outs;
  let call = Uop.call ~body ~args:[ range ] ~info in
  List.iter (fun spec -> Spec.type_verify spec call)
    [ Spec.tensor_spec; Spec.program_spec; Spec.full_spec ];
  equal int 1 (List.length (Uop.ranges call));
  is_true ~msg:"host calls preserve argument ranges"
    (Uop.equal range (List.hd (Uop.ranges call)));
  is_true ~msg:"host returns have scalar shape" (Uop.shape_opt call = Some []);
  let changed = Uop.replace call
      ~arg:(Uop.Arg.Call_info { info with dtype = Dtype.float32 }) () in
  is_true ~msg:"metadata replacement derives the new return type"
    (Dtype.equal (Uop.dtype changed) Dtype.float32);
  is_true ~msg:"serialized calls retain the declared return type"
    (Uop.equal changed (Uop.import (Uop.export changed)))

let end_requires_an_effect () =
  let range = Uop.range ~size:(i32 4) ~axis:0 ~kind:Axis_type.Loop
      ~dtype:Dtype.int32 () in
  let body = Uop.noop () in
  let closed = Uop.end_ ~value:body ~ranges:[ range ] in
  List.iter (fun spec ->
      is_true ~msg:"END accepts a void effect" (accepts spec closed);
      List.iter (fun node ->
          is_true ~msg:"END rejects discarded values and unbounded ranges"
            (rejected spec node))
        [ Uop.end_ ~value:(i32 1) ~ranges:[ range ];
          Uop.end_ ~value:body ~ranges:[ Uop.loop ~axis:1 ];
          Uop.replace closed ~arg:(Uop.Arg.Int 1) () ])
    [ Spec.shared_spec; Spec.program_spec; Spec.full_spec ]

let conditional_loop_contract () =
  let outer = Uop.range ~size:(i32 4) ~axis:1 ~kind:Axis_type.Loop
      ~dtype:Dtype.int32 () in
  let loop = Uop.loop ~axis:2 in
  let cond = Uop.alu_binary ~op:Ops.Cmplt ~lhs:outer ~rhs:(i32 3) in
  let edge = Uop.backedge ~body:(Uop.noop ~src:loop ())
      ~loop ~cond in
  List.iter (fun spec -> Spec.type_verify spec edge)
    [ Spec.tensor_spec; Spec.program_spec; Spec.full_spec ];
  equal int 1 (List.length (Uop.ranges edge));
  is_true ~msg:"condition retains the enclosing range"
    (Uop.equal outer (List.hd (Uop.ranges edge)));
  is_true ~msg:"BACKEDGE has no value shape" (Uop.shape_opt edge = None);
  is_true ~msg:"barrier propagates the loop closure"
    (not (List.exists (Uop.equal loop)
       (Uop.ranges (Uop.after ~src:loop
          ~deps:[ Uop.barrier ~srcs:[ edge ] () ]))));
  List.iter (fun bad -> is_true (rejected Spec.full_spec bad))
    [ Uop.backedge ~body:(i32 0) ~loop:outer ~cond;
      Uop.backedge ~body:(i32 0) ~loop ~cond:(i32 1);
      Uop.backedge ~body:(i32 0) ~loop
        ~cond:(Uop.stack [ Uop.const_bool true; Uop.const_bool false ]) ];
  is_true ~msg:"conditional END is rejected"
    (rejected Spec.full_spec
       (Uop.replace (Uop.end_ ~value:(i32 0) ~ranges:[ loop ])
          ~src:[| i32 0; loop; cond |] ()))

let () =
  run "tolk.uop.spec"
    [
      group "shared_spec"
        [
          test "Typed host call contract" typed_host_call_contract;
          test "Conditional loop contract" conditional_loop_contract;
          test "END requires an effect" end_requires_an_effect;
          test "Sink void accepted" sink_void;
          test "Const matching dtype" const_matching_dtype;
          test "Param with Param_arg" param_with_param_arg;
          test "Param without Param_arg rejected" param_rejects_empty_arg;
          test "Buffer with Param_arg" buffer_with_param_arg;
          test "Global Buffer rejected by shared_spec"
            shared_rejects_global_buffer;
          test "Buffer without Param_arg rejected" buffer_rejects_empty_arg;
          test "Buffer ALU addrspace rejected" buffer_rejects_alu_addrspace;
          test "Empty Stack void accepted" empty_stack_void;
          test "Stack source contract" stack_sources_match;
          test "Stack derives child dtype"
            stack_derives_child_dtype;
          test "Stack rejects mixed child dtype" stack_rejects_mixed_dtype;
          test "Stack mixed child counts rejected"
            stack_rejects_mixed_child_counts;
          test "Where with bool cond" where_bool_cond;
          test "Where non-bool cond rejected" where_rejects_non_bool_cond;
          test "Cmplt returns bool" cmplt_is_bool;
          test "Cdiv with float rejected" cdiv_rejects_float;
          test "ALU operand scalars match" alu_operand_scalars_match;
          test "shift count dtypes" shift_count_dtypes;
          test "bitwise rejects float operands"
            bitwise_rejects_float_operands;
          test "Index accepts integer offsets" index_accepts_integer_offsets;
          test "Index rejects gate source" index_rejects_gate_source;
          test "Special raw name accepted" special_accepts_raw_name;
          test "Special dtype by stage" special_dtype_by_stage;
          test "Group value source rejected" group_rejects_value_source;
          test "After value first source rejected"
            after_rejects_value_first_source;
          test "End non-range tail rejected" end_rejects_non_range_tail;
          test "Range bad layouts rejected" range_rejects_bad_layouts;
          test "Barrier boundaries" barrier_boundaries;
          test "Group/After bad layouts rejected" group_after_bad_layouts;
        ];
      group "tensor_spec"
        [
          test "Global Buffer accepted by tensor_spec"
            tensor_accepts_global_buffer;
          test "Copy matching dtype" copy_matching_dtype;
          test "Copy device source rejected" copy_rejects_device_source_layout;
          test "Copy must be eliminated before range indexing"
            copy_rejects_lowered_range_sources;
          test "Copy bad device or dtype rejected"
            copy_rejects_bad_device_or_dtype;
          test "Call bad layouts rejected"
            call_reject_bad_layouts;
          test "Call source contracts"
            call_source_contracts;
          test "call outputs reject invalid positions" call_outputs_reject_invalid_positions;
          test "Reduce op arg required" reduce_arg_required;
          test "Tensor reduce accepts lowered integer tail"
            tensor_reduce_accepts_lowered_integer_tail;
          test "Reduce rejects old Op arg" reduce_rejects_old_op_arg;
          test "Allreduce layouts" allreduce_layouts;
          test "Allreduce bad device or dtype rejected"
            allreduce_rejects_bad_device_or_dtype;
          test "Mselect layouts" multi_device_selection_layouts;
          test "Mstack layouts" multi_device_stack_layouts;
          test "Multi layouts" multi_device_multi_layouts;
          test "Stage bad layouts rejected" stage_rejects_bad_layouts;
          test "bind accepts a variable and constant" bind_accepts_variable_const;
          test "bind rejects a nonconstant value" bind_rejects_nonconstant_value;
          test "bind rejects a call parameter" bind_rejects_parameter;
          test "bind accepts a 64-bit variable" bind_accepts_64_bit_variable;
          test "variable supplies the bound literal dtype" bind_variable_supplies_dtype;
          test "Movement validates shape contracts"
            movement_validates_shape_contracts;
          test "If/Endif rejected" tensor_rejects_if_endif;
        ];
      group "program_spec"
        [
          test "Invalid const rejected" program_rejects_invalid_const;
          test "Weakint rejected" program_rejects_weakint;
          test "Buffer addrspace rules" program_buffer_rules;
          test "Range forms" program_range_forms;
          test "Tensor-only ops rejected" program_rejects_tensor_only_ops;
          test "Plain load accepted" program_accepts_plain_load;
          test "Cast(Index) load accepted" program_accepts_cast_index_load;
          test "same-dtype Bitcast(Index) load accepted"
            program_bitcast_index_same_dtype_is_plain_index;
          test "real Bitcast(Index) load rejected"
            program_rejects_real_bitcast_index_load;
          test "Load gate on Load accepted" program_accepts_load_gate_on_load;
          test "Load gate on Index rejected" program_rejects_load_gate_on_index;
          test "Plain store accepted" program_accepts_plain_store;
          test "CHECK_OOB disabled accepts out-of-bounds load"
            program_oob_disabled_accepts_out_of_bounds_load;
          test "CHECK_OOB rejects unproved indices"
            program_oob_rejects_unproved_indices;
          test "CHECK_OOB rejects proven out-of-bounds load"
            program_oob_enabled_rejects_out_of_bounds_load;
          test "CHECK_OOB accepts minmax-proven in-bounds load"
            program_oob_enabled_accepts_minmax_in_bounds_load;
          test "CHECK_OOB uses explicit buffer max_numel"
            program_oob_uses_explicit_buffer_shape;
          test "CHECK_OOB accepts image pointer accesses"
            program_oob_image_pointer_bypasses_bounds;
          test "CHECK_OOB accepts false-gated load"
            program_oob_false_gate_accepts_out_of_bounds_load;
          test "CHECK_OOB accepts symbolic false-gated load"
            program_oob_symbolic_false_gate_accepts_out_of_bounds_load;
          test "CHECK_OOB rejects unmasked symbolic store"
            program_oob_symbolic_store_remains_rejected;
          test "CHECK_OOB accepts mask-proven symbolic bounds"
            program_oob_masked_symbolic_bounds_are_accepted;
          test "CHECK_OOB uses loaded component bounds"
            program_oob_loaded_component_bounds;
          test "CHECK_OOB uses offset component bounds"
            program_oob_offset_component_bounds;
          test "CHECK_OOB uses shifted component bounds"
            program_oob_shift_component_bounds;
          test "CHECK_OOB preserves committed guard constants"
            program_oob_committed_guard_constants;
          test "CHECK_OOB proof variables cannot alias user parameters"
            program_oob_proof_variables_are_fresh;
          test "CHECK_OOB component bounds preserve casts and wrapping"
            program_oob_component_bounds_preserve_casts;
          test "CHECK_OOB rejects incomplete masked symbolic bounds"
            program_oob_masked_symbolic_lower_bound_only_rejected;
          test "Nested casted index source rejected"
            program_rejects_nested_casted_index_source;
          test "Store gate on Store accepted" program_accepts_store_gate_on_store;
          test "Store gate on Index rejected" program_rejects_store_gate_on_index;
          test "If accepts Shrink index" program_accepts_if_with_shrink_index;
          test "Control-flow boundaries" program_control_flow_boundaries;
          test "Bad If layouts rejected" program_rejects_bad_if_layouts;
          test "Program rejects loose After layout"
            program_rejects_loose_after_layout;
          test "Bad If dedup source matrix rejected"
            program_rejects_if_dedup_source_matrix;
          test "Bad Endif layouts rejected" program_rejects_bad_endif_layouts;
          test "End/Range boundaries" program_end_range_boundaries;
          test "verify_list validates flat programs"
            verify_list_validates_flat_program;
        ];
      group "full_spec"
        [
          test "full_spec has no catch-all" full_spec_has_no_catch_all;
          test "full_spec accepts intermediate forms"
            full_spec_accepts_intermediate_forms;
        ];
    ]
