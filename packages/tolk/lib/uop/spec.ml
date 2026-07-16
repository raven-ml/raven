(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Types *)

type rule =
  | Strict of Upat.t * (Uop.t -> Upat.bindings -> bool)
  | Tentative of Upat.t * (Uop.t -> Upat.bindings -> bool option)

type t = rule list

let ( =?> ) pat pred : rule = Strict (pat, pred)
let ( =??> ) pat pred : rule = Tentative (pat, pred)
let make rules : t = rules
let ( ++ ) = List.append

(* Dtype helpers *)

let is_void u = Dtype.equal (Uop.dtype u) Dtype.void
let is_bool u = Dtype.equal (Uop.dtype u) Dtype.bool
let is_index u = Dtype.equal (Uop.dtype u) Dtype.index
let is_weak u = Dtype.is_weak (Uop.dtype u)
let is_int u = Dtype.is_int (Uop.dtype u)
let same_dtype a b = Dtype.equal (Uop.dtype a) (Uop.dtype b)
let matches_or_weak u s = same_dtype u s || is_weak s
let int32_or_index u = Dtype.equal (Uop.dtype u) Dtype.int32 || is_index u
let arg_empty u = match Uop.arg u with Uop.Arg.Empty -> true | _ -> false

let option_for_all p = function
  | None -> true
  | Some x -> p x

let valid_device_payload = function
  | Uop.Single _ -> true
  | Uop.Multi devs -> devs <> []
  | Uop.Index _ -> false

let valid_multi_device_payload = function
  | Uop.Multi devs -> devs <> []
  | Uop.Single _ | Uop.Index _ -> false

let valid_sharding_axis shape axis device =
  match axis with
  | None -> true
  | Some axis ->
      axis >= 0
      &&
      match device with
      | Some (Uop.Multi devs) when devs <> [] ->
          axis < List.length (Uop.as_shape shape)
      | _ -> false

let valid_shape_child u =
  is_int u
  || (Uop.op u = Ops.Stack && is_void u && Array.length (Uop.src u) = 0)
  || (Uop.op u = Ops.Noop && is_void u && Array.length (Uop.src u) = 0)

let valid_param u =
  match Uop.arg u with Uop.Arg.Param_arg _ -> true | _ -> false

let valid_buffer u =
  match Uop.arg u with
  | Uop.Arg.Param_arg { addrspace = Dtype.Local | Dtype.Reg; _ } -> true
  | _ -> false

let valid_global_buffer u =
  match Uop.as_buffer u with
  | None -> false
  | Some { Uop.buffer; shape } ->
      buffer.slot >= 0
      && buffer.addrspace = Dtype.Global
      && option_for_all valid_device_payload buffer.device
      && valid_shape_child shape
      && valid_sharding_axis shape buffer.axis buffer.device
      && is_index shape

let is_const_invalid u =
  match Uop.op u, Uop.arg u with
  | Ops.Const, Uop.Arg.Value c -> Const.view c = Const.Invalid
  | _ -> false

let tail_srcs p u =
  let srcs = Uop.src u in
  let n = Array.length srcs in
  let rec loop i = i = n || (p srcs.(i) && loop (i + 1)) in
  loop 1

let shape_equal a b =
  try
    let sa = Uop.shape a and sb = Uop.shape b in
    List.length sa = List.length sb && List.for_all2 Uop.equal sa sb
  with Invalid_argument _ -> false

let stack_ok u =
  let srcs = Uop.src u in
  Array.length srcs = 0
  || (Array.for_all (shape_equal srcs.(0)) srcs
      && Array.for_all (same_dtype u) srcs)

let movement_shape_ok u =
  try ignore (Uop.shape u); true with Invalid_argument _ -> false

let end_ok u = tail_srcs (fun s -> Uop.op s = Ops.Range) u

let valid_full_slice u =
  match Uop.as_slice u, Uop.src u with
  | Some { offset; _ }, srcs ->
      Array.length srcs >= 2
      && (match Uop.op srcs.(0) with
          | op when Ops.Group.is_movement op -> true
          | Ops.Buffer | Ops.Param | Ops.Stage | Ops.After -> true
          | _ -> false)
      && Uop.op offset = Ops.Const
      && is_index offset
  | _ -> false

let call_info_arg u =
  match Uop.arg u with
  | Uop.Arg.Call_info _ -> true
  | _ -> false

let valid_gettuple g t =
  match Uop.arg g with
  | Uop.Arg.Int i ->
      let tuple =
        match Uop.op t, Uop.src t with
        | Ops.Tuple, _ -> Some t
        | Ops.Function, srcs
          when Array.length srcs > 0 && Uop.op srcs.(0) = Ops.Tuple ->
            Some srcs.(0)
        | _ -> None
      in
      (match tuple with
       | None -> false
       | Some tuple ->
           let srcs = Uop.src tuple in
           i >= 0 && i < Array.length srcs && same_dtype g srcs.(i))
  | _ -> false

let opaque_call_body = function
  | Ops.Sink | Ops.Program | Ops.Linear | Ops.Copy | Ops.Slice
  | Ops.Custom_function -> true
  | _ -> false

let call_ok u body =
  call_info_arg u
  && opaque_call_body (Uop.op body)
  && Uop.ranges body = []

let function_ok u body =
  call_info_arg u && Uop.op body = Ops.Tuple && Uop.ranges body = []

let stage_ok u = Option.is_some (Uop.as_stage u) && tail_srcs is_int u

let valid_reduce_op op = Ops.Group.mem op Ops.Group.reduce

let reduce_tail_ok s = is_index s || Dtype.equal (Uop.dtype s) Dtype.int32

let reduce_arg_ok u =
  match Uop.Arg.as_reduce_arg (Uop.arg u) with
  | Some { op; _ } -> valid_reduce_op op && tail_srcs reduce_tail_ok u
  | None -> false

let copy_arg_device u =
  match Uop.arg u with
  | Uop.Arg.Device device -> valid_device_payload device
  | _ -> false

let copy_ok u x = same_dtype u x && copy_arg_device u

let allreduce_ok u x =
  match Uop.as_allreduce u with
  | Some { src; op; device } ->
      Uop.equal src x
      && same_dtype u x
      && valid_reduce_op op
      && valid_multi_device_payload device
  | None -> false

let local_reg_buffer u =
  match Uop.as_buffer u with
  | Some { buffer; _ } ->
      valid_buffer u
      &&
      (match buffer.addrspace with Dtype.Local | Dtype.Reg -> true | _ -> false)
  | None -> false

let alu_param u =
  match Uop.as_param u with
  | Some { param = { addrspace = Dtype.Alu; _ }; _ } ->
      is_int u
  | _ -> false

let bind_value u = Uop.op u = Ops.Const && int32_or_index u

let bind_ok u var value =
  arg_empty u
  && alu_param var
  && bind_value value
  && same_dtype u var
  && same_dtype var value

let mselect_ok u =
  match Uop.Arg.as_int (Uop.arg u), Uop.src u with
  | Some i, [| src |] ->
      i >= 0
      && same_dtype u src
      &&
      (match Uop.device_of src with
       | Some (Uop.Multi devs) -> i < List.length devs
       | _ -> false)
  | _ -> false

let mstack_ok u =
  let srcs = Uop.src u in
  Array.length srcs > 0
  && Array.for_all (same_dtype u) srcs
  &&
  let all_single =
    Array.for_all
      (fun s ->
        match Uop.device_of s with
        | Some (Uop.Single _) -> true
        | _ -> false)
      srcs
  in
  let all_same_none =
    match Uop.device_of srcs.(0) with
    | None ->
        let first = srcs.(0) in
        Array.for_all (Uop.equal first) srcs
    | Some _ -> false
  in
  all_single || all_same_none

let multi_ok u =
  match Uop.Arg.as_int (Uop.arg u), Uop.src u with
  | Some axis, [| src |] ->
      axis >= 0
      && same_dtype u src
      &&
      (match Uop.device_of u with
       | Some (Uop.Multi devs) when devs <> [] ->
           (try axis < List.length (Uop.shape src) with
            | Invalid_argument _ -> false)
       | _ -> false)
  | _ -> false

(* Shared spec — rules valid at every stage. *)

let shared_spec : t =
  let open Upat in
  make [
    op ~dtype:Dtype.void Ops.Sink =?> (fun _ _ -> true);

    op Ops.Noop =?> (fun _ _ -> true);

    op ~src:[] Ops.Const
    =?> (fun u _ -> match Uop.arg u with
      | Uop.Arg.Value c -> Dtype.equal (Const.dtype c) (Uop.dtype u)
      | _ -> false);

    op Ops.Param =?> (fun u _ -> valid_param u);

    op Ops.Buffer =?> (fun u _ -> valid_buffer u);

    op ~dtype:Dtype.void ~src:[] Ops.Stack =?> (fun _ _ -> true);

    op ~allow_any_len:true ~src:[ any ] Ops.Stack
    =?> (fun u _ -> stack_ok u);

    op ~src:[ var "c"; var "t"; var "e" ] Ops.Where
    =?> (fun u bs ->
      is_bool (bs $ "c")
      && matches_or_weak u (bs $ "t")
      && matches_or_weak u (bs $ "e"));

    ops ~dtype:Dtype.bool ~src:[ var "x"; var "y" ]
      Ops.Group.comparison
    =?> (fun _ bs ->
      let x = bs $ "x" and y = bs $ "y" in
      same_dtype x y || is_weak x || is_weak y);

    ops ~src:[ var "x"; var "y" ] [ Ops.Shl; Ops.Shr ]
    =?> (fun u bs ->
      let x = bs $ "x" and y = bs $ "y" in
      let rhs_ok =
        same_dtype x y || Dtype.equal (Uop.dtype y) Dtype.uint32 || is_weak y
      in
      same_dtype u x && rhs_ok);

    ops [ Ops.Cdiv; Ops.Cmod; Ops.Floordiv; Ops.Floormod ]
    =??> (fun u _ -> if is_int u then None else Some false);

    ops Ops.Group.alu
    =?> (fun u _ -> Array.for_all (matches_or_weak u) (Uop.src u));

    ops ~src:[ any ] [ Ops.Cast; Ops.Bitcast ]
    =?> (fun u _ -> arg_empty u);

    op ~allow_any_len:true ~src:[ var "size" ] Ops.Range
    =?> (fun u bs ->
      same_dtype u (bs $ "size")
      && Option.is_some (Uop.as_range u));

    op ~allow_any_len:true ~src:[ any ] Ops.Index
    =?> (fun u _ -> tail_srcs is_int u);

    op ~allow_any_len:true ~src:[ any ] Ops.End
    =?> (fun u _ -> end_ok u);

    op_src ~dtype:(exact_dtype Dtype.void)
      ~src:(repeat (ops [ Ops.Group; Ops.Store; Ops.Noop; Ops.Ins; Ops.End ]))
      Ops.Group
    =?> (fun _ _ -> true);

    op_src
      ~src:(prefix [
        ops (Ops.Group.movement @
             [ Ops.Param; Ops.Buffer; Ops.Contiguous; Ops.Index; Ops.After;
               Ops.Multi; Ops.Bitcast; Ops.Ins ])
      ])
      Ops.After
    =?> (fun _ _ -> true);

    op Ops.Custom =?> (fun _ _ -> true);
    op Ops.Customi =?> (fun _ _ -> true);

    op Ops.Pyliteral =?> (fun _ _ -> true);

    op ~allow_any_len:true ~dtype:Dtype.void Ops.Barrier
    =?> (fun _ _ -> true);

    op ~dtype:Dtype.void ~src:[ var "x" ] Ops.Wait
    =?> (fun _ bs -> is_bool (bs $ "x"));

    op Ops.Ins =?> (fun _ _ -> true);

    op ~src:[ var "idx" ] Ops.Load
    =?> (fun _ bs -> Validate.validate_index_source (bs $ "idx"));

    op ~src:[ var "idx"; var "alt"; var "gate" ] Ops.Load
    =?> (fun u bs ->
      is_bool (bs $ "gate")
      && same_dtype u (bs $ "alt")
      && Validate.validate_index_source ~gate:(bs $ "gate") (bs $ "idx"));

    op ~dtype:Dtype.void ~src:[ var "idx"; any ] Ops.Store
    =??> (fun _ bs ->
      if Validate.is_index_source (bs $ "idx")
      then Some (Validate.validate_index_source (bs $ "idx"))
      else None);

    op ~dtype:Dtype.void ~src:[ var "idx"; any; var "gate" ] Ops.Store
    =?> (fun _ bs ->
      is_bool (bs $ "gate")
      && Validate.validate_index_source ~gate:(bs $ "gate") (bs $ "idx"));

    op ~dtype:Dtype.void ~src:[ any; any ] Ops.Store =?> (fun _ _ -> true);

    op ~src:[ any; any; any ] Ops.Wmma
    =?> (fun u _ -> Option.is_some (Uop.as_wmma u));
  ]

(* Tensor spec. *)

let tensor_spec : t =
  let open Upat in
  let tensor_only = make [
    ops ~src:[ any ] [ Ops.Sin; Ops.Log2; Ops.Exp2; Ops.Sqrt; Ops.Reciprocal ]
    =?> (fun u _ -> Dtype.is_float (Uop.dtype u));

    op Ops.Buffer =??> (fun u _ ->
      if valid_global_buffer u then Some true else None);

    op ~src:[ var "var"; var "value" ] Ops.Bind
    =?> (fun u bs -> bind_ok u (bs $ "var") (bs $ "value"));

    op ~dtype:Dtype.void Ops.Custom_function
    =?> (fun u _ -> Option.is_some (Uop.Arg.as_string (Uop.arg u)));

    op ~allow_any_len:true ~dtype:Dtype.void
      ~src:[ ops [ Ops.Sink; Ops.Linear; Ops.Program; Ops.Copy;
                   Ops.Custom_function ] ]
      Ops.Call
    =?> (fun u _ ->
      let srcs = Uop.src u in
      Array.length srcs > 0 && call_ok u srcs.(0));

    op ~allow_any_len:true ~dtype:Dtype.void ~src:[ op Ops.Tuple ]
      Ops.Function
    =?> (fun u _ ->
      let srcs = Uop.src u in
      Array.length srcs > 0 && function_ok u srcs.(0));

    op ~dtype:Dtype.void Ops.Tuple =?> (fun _ _ -> true);

    op ~src:[ var "t" ] Ops.Gettuple
    =?> (fun u bs -> valid_gettuple u (bs $ "t"));

    op ~src:[ var "x" ] Ops.Special
    =?> (fun u bs ->
      let x = bs $ "x" in
      Option.is_some (Uop.as_special u) && same_dtype u x && is_index x);

    ops ~dtype:Dtype.index [ Ops.Add; Ops.Mul; Ops.Cdiv; Ops.Floordiv ]
    =?> (fun _ _ -> true);

    ops ~src:[ any; any ] [ Ops.Reshape; Ops.Expand ]
    =?> (fun u _ -> movement_shape_ok u);

    ops ~src:[ any; any; any ] [ Ops.Pad; Ops.Shrink ]
    =?> (fun u _ ->
      let srcs = Uop.src u in
      shape_equal srcs.(1) srcs.(2) && movement_shape_ok u);

    op ~src:[ any ] Ops.Permute
    =?> (fun u _ ->
      Option.is_some (Uop.Arg.as_ints (Uop.arg u))
      && movement_shape_ok u);

    op ~src:[ any ] Ops.Flip
    =?> (fun u _ ->
      Option.is_some (Uop.Arg.as_bools (Uop.arg u))
      && movement_shape_ok u);

    op ~allow_any_len:true ~src:[ any ] Ops.Reduce
    =?> (fun u _ -> reduce_arg_ok u);

    op ~allow_any_len:true ~src:[ var "x" ] Ops.Copy
    =?> (fun u bs -> copy_ok u (bs $ "x"));

    op ~src:[ var "x" ] Ops.Allreduce
    =?> (fun u bs -> allreduce_ok u (bs $ "x"));

    op Ops.Multi =?> (fun u _ -> multi_ok u);

    op Ops.Mselect =?> (fun u _ -> mselect_ok u);

    op Ops.Mstack =?> (fun u _ -> mstack_ok u);

    ops ~allow_any_len:true ~src:[ var "x" ]
      [ Ops.Detach; Ops.Contiguous; Ops.Contiguous_backward ]
    =?> (fun u bs -> same_dtype u (bs $ "x"));

    op ~allow_any_len:true ~src:[ var "x" ] Ops.Stage
    =?> (fun u _ -> stage_ok u);

    op ~dtype:Dtype.void Ops.Linear =?> (fun _ _ -> true);

    op ~dtype:Dtype.void ~src:[] Ops.Source =?> (fun _ _ -> true);

    op ~dtype:Dtype.uint8 ~src:[] Ops.Binary
    =?> (fun u _ -> Option.is_some (Uop.Arg.as_string (Uop.arg u)));

    op ~dtype:Dtype.void ~src:[ op Ops.Sink ] Ops.Program
    =?> (fun _ _ -> true);
    op ~dtype:Dtype.void ~src:[ op Ops.Sink; op Ops.Linear ] Ops.Program
    =?> (fun _ _ -> true);
    op ~dtype:Dtype.void
      ~src:[ op Ops.Sink; op Ops.Linear; op Ops.Source ] Ops.Program
    =?> (fun _ _ -> true);
    op ~dtype:Dtype.void
      ~src:[ op Ops.Sink; op Ops.Linear; op Ops.Source; op Ops.Binary ]
      Ops.Program
    =?> (fun _ _ -> true);
  ] in
  tensor_only ++ shared_spec

(* Program spec. *)

let program_spec : t =
  let open Upat in
  let program_only = make [
    ops Ops.Group.all =??> (fun u _ ->
      if is_index u || is_weak u then Some false else None);

    op ~src:[ ops [ Ops.Param; Ops.Buffer; Ops.After ]; any; op Ops.Const ]
      Ops.Shrink
    =?> (fun _ _ -> true);

    ops Ops.Group.movement =?> (fun _ _ -> false);

    op Ops.Buffer =?> (fun u _ -> local_reg_buffer u);

    op Ops.Const =??> (fun u _ ->
      if is_const_invalid u then Some false else None);

    op ~src:[ any; any ] Ops.Load =?> (fun _ _ -> false);

    op ~allow_any_len:true ~src:[ any ] Ops.End
    =?> (fun u _ -> end_ok u);

    op ~dtype:Dtype.void
      ~src:[ var "cond"; ops [ Ops.Cast; Ops.Index; Ops.Shrink ] ] Ops.If
    =?> (fun _ bs -> is_bool (bs $ "cond"));

    op ~dtype:Dtype.void ~src:[ op Ops.If ] Ops.Endif
    =?> (fun _ _ -> true);

    op ~src:[ var "x" ] Ops.Special
    =?> (fun u bs ->
      let x = bs $ "x" in
      Option.is_some (Uop.as_special u) && same_dtype u x
      && Dtype.equal (Uop.dtype x) Dtype.int32);
  ] in
  program_only ++ shared_spec

(* Full spec — explicit intermediate forms plus tensor and program specs. *)

let full_only_spec : t =
  let open Upat in
  make [
    op ~dtype:Dtype.void Ops.Rewrite_error
    =?> (fun u _ -> Option.is_some (Uop.Arg.as_string (Uop.arg u)));

    op Ops.Slice =?> (fun u _ -> valid_full_slice u);

    op ~allow_any_len:true ~dtype:Dtype.void ~src:[ op Ops.Slice ] Ops.Call
    =?> (fun _ _ -> true);

    op ~allow_any_len:true ~src:[ any; any ] Ops.End
    =?> (fun _ _ -> true);

    op ~allow_any_len:true ~src:[ any ] Ops.After
    =?> (fun _ _ -> true);

    ops [ Ops.Load; Ops.Store ] =?> (fun _ _ -> true);

    op ~src:[ any; any ] Ops.Bind
    =?> (fun u _ -> arg_empty u && int32_or_index u);
  ]

let full_spec : t =
  full_only_spec ++ tensor_spec ++ program_spec

(* Verification *)

(* First-match semantics: a [Strict] rule whose pattern matches decides
   the node (accept on [true], reject on [false]). A [Tentative] rule
   that returns [None] instead defers to the next matching rule; a
   [Tentative] rule returning [Some b] behaves like [Strict]. *)
let accepts (spec : t) u =
  let rec try_rules = function
    | [] -> false
    | Strict (pat, pred) :: rest -> (
        match Upat.match_ pat u with
        | [] -> try_rules rest
        (* First matching binding decides, like tinygrad's PatternMatcher
           taking the first non-None callback result. *)
        | bs :: _ -> pred u bs)
    | Tentative (pat, pred) :: rest -> (
        match Upat.match_ pat u with
        | [] -> try_rules rest
        | matches ->
            (* First binding with a decision ([Some]) wins; all-[None]
               defers to the next rule. *)
            let rec first = function
              | [] -> try_rules rest
              | bs :: more -> (
                  match pred u bs with Some b -> b | None -> first more)
            in
            first matches)
  in
  try_rules spec

exception Verification_failed of Uop.t

let () =
  Printexc.register_printer (function
    | Verification_failed u ->
        Some (Printf.sprintf "Spec.Verification_failed: op=%s dtype=%s"
          (Ops.name (Uop.op u))
          (Dtype.to_string (Uop.dtype u)))
    | _ -> None)

let verify_list spec program =
  List.iter (fun u ->
    if not (accepts spec u) then raise (Verification_failed u))
    program

let type_verify spec root =
  verify_list spec (Uop.toposort root)
