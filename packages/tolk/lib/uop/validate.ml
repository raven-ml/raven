(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let is_const_invalid u =
  match Uop.op u, Uop.arg u with
  | Ops.Const, Uop.Arg.Value c -> Const.view c = Const.Invalid
  | _ -> false

let is_false_const u =
  match Uop.op u, Uop.arg u with
  | Ops.Const, Uop.Arg.Value c ->
      (match Const.view c with Const.Bool false -> true | _ -> false)
  | _ -> false

let prod_ints xs = List.fold_left ( * ) 1 xs

let explicit_max_shape u =
  match Uop.src u with
  | [| shape; _ |] | [| shape |] when Uop.op shape <> Ops.Noop ->
      Some (List.map Uop.vmax (Uop.as_shape shape))
  | _ -> None

let max_numel u =
  match Uop.dtype u with
  | Dtype.Ptr p when Dtype.Ptr.is_image p -> None
  | Dtype.Ptr p ->
      (match explicit_max_shape u with
       | Some shape -> Some (prod_ints shape)
       | None -> Some (Dtype.Ptr.size p))
  | Dtype.Val _ -> None

let check_oob_enabled () =
  match Sys.getenv_opt "CHECK_OOB" with
  | None -> false
  | Some "" | Some "0" | Some "false" | Some "False" | Some "FALSE" ->
      false
  | Some _ -> true

let has_oob_bypass u =
  let rec walk u =
    match Uop.op u, Uop.src u with
    | Ops.Bitcast, _ | Ops.Stack, _ -> true
    | Ops.Cast, [| src |] when Dtype.is_ptr (Uop.dtype src) -> true
    | Ops.Param, _ -> false
    | _ -> Array.exists walk (Uop.src u)
  in
  walk u

let sat_pred n = if n = min_int then min_int else n - 1
let sat_succ n = if n = max_int then max_int else n + 1

let interval_empty (lo, hi) = lo > hi

let tighten_lower lo candidate = max lo candidate
let tighten_upper hi candidate = min hi candidate

let refine_cmp_bound idx (lo, hi) lhs rhs =
  match Uop.const_int_value lhs, Uop.const_int_value rhs with
  | _, Some n when Uop.equal lhs idx -> lo, tighten_upper hi (sat_pred n)
  | Some n, _ when Uop.equal rhs idx -> tighten_lower lo (sat_succ n), hi
  | _ -> lo, hi

let refine_eq_bound idx (lo, hi) lhs rhs =
  match Uop.const_int_value lhs, Uop.const_int_value rhs with
  | _, Some n when Uop.equal lhs idx ->
      tighten_lower lo n, tighten_upper hi n
  | Some n, _ when Uop.equal rhs idx ->
      tighten_lower lo n, tighten_upper hi n
  | _ -> lo, hi

let bool_and u =
  match Uop.dtype u with
  | Dtype.Val v -> Dtype.Val.is_bool v && Uop.op u = Ops.And
  | Dtype.Ptr _ -> false

let rec refine_index_bounds_under_gate idx bounds gate =
  match Uop.op gate, Uop.src gate with
  | Ops.And, [| lhs; rhs |] when bool_and gate ->
      let bounds = refine_index_bounds_under_gate idx bounds lhs in
      if interval_empty bounds then bounds
      else refine_index_bounds_under_gate idx bounds rhs
  | Ops.Cmplt, [| lhs; rhs |] -> refine_cmp_bound idx bounds lhs rhs
  | Ops.Cmpeq, [| lhs; rhs |] -> refine_eq_bound idx bounds lhs rhs
  | _ -> bounds

let validate_index_with_gate_bounds size idx gate =
  let lo, hi =
    refine_index_bounds_under_gate idx (Uop.vmin idx, Uop.vmax idx) gate
  in
  interval_empty (lo, hi) || (0 <= lo && hi < size)

let validate_index ?gate uidx =
  let srcs = Uop.src uidx in
  if Array.length srcs < 2 then true
  else
    let buf = srcs.(0) in
    let idxs = Array.sub srcs 1 (Array.length srcs - 1) |> Array.to_list in
    List.exists is_const_invalid idxs
    || (not (check_oob_enabled ()))
    || List.exists has_oob_bypass idxs
    || (match Uop.dtype buf with
       | Dtype.Ptr p -> Dtype.Ptr.is_image p
       | Dtype.Val _ -> false)
    ||
    match gate with
    | Some g when is_false_const g || Uop.vmax g = 0 -> true
    | Some g when has_oob_bypass g -> true
    | _ -> (
        let check_axis size idx =
          (0 <= Uop.vmin idx && Uop.vmax idx < size)
          ||
          match gate with
          | Some gate -> validate_index_with_gate_bounds size idx gate
          | None -> false
        in
        match idxs with
        | [ idx ] -> (
            match max_numel buf with
            | None -> true
            | Some size -> check_axis size idx)
        | _ -> (
            try
              let shape = Uop.shape buf in
              List.length shape <> List.length idxs
              ||
              List.for_all2
                (fun dim idx -> check_axis (Uop.vmin dim) idx)
                shape idxs
            with Invalid_argument _ -> true))

let index_source u =
  match Uop.op u, Uop.src u with
  | Ops.Cast, [| x |] ->
      (match Uop.op x with
       | Ops.Index | Ops.Shrink -> Some x
       | _ -> None)
  | (Ops.Index | Ops.Shrink), _ -> Some u
  | _ -> None

let is_index_source u = Option.is_some (index_source u)

let validate_index_source ?gate u =
  match index_source u with
  | None -> false
  | Some uidx -> validate_index ?gate uidx
