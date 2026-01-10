(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* error.ml *)

exception Shape_mismatch of string

let cannot ~op ~what ~from ~to_ ?reason ?hint () =
  let msg =
    let base = Printf.sprintf "%s: cannot %s %s to %s" op what from to_ in
    let with_reason =
      match reason with
      | None -> base
      | Some r -> Printf.sprintf "%s (%s)" base r
    in
    match hint with
    | None -> with_reason
    | Some h -> Printf.sprintf "%s\nhint: %s" with_reason h
  in
  invalid_arg msg

let invalid ~op ~what ?reason ?hint () =
  let msg =
    let base = Printf.sprintf "%s: invalid %s" op what in
    let with_reason =
      match reason with
      | None -> base
      | Some r -> Printf.sprintf "%s (%s)" base r
    in
    match hint with
    | None -> with_reason
    | Some h -> Printf.sprintf "%s\nhint: %s" with_reason h
  in
  invalid_arg msg

let failed ~op ~what ?reason ?hint () =
  let msg =
    let base = Printf.sprintf "%s: %s" op what in
    let with_reason =
      match reason with
      | None -> base
      | Some r -> Printf.sprintf "%s (%s)" base r
    in
    match hint with
    | None -> with_reason
    | Some h -> Printf.sprintf "%s\nhint: %s" with_reason h
  in
  invalid_arg msg

let shape_to_string shape =
  let elements = Array.map string_of_int shape |> Array.to_list in
  Printf.sprintf "[%s]" (String.concat "," elements)

let shape_mismatch ~op ~expected ~actual ?hint () =
  let expected_str = shape_to_string expected in
  let actual_str = shape_to_string actual in
  let expected_size = Array.fold_left ( * ) 1 expected in
  let actual_size = Array.fold_left ( * ) 1 actual in

  (* Only show element count if dimensions differ but we're doing a reshape-like
     operation *)
  let reason =
    if Array.length expected = Array.length actual then
      (* Same rank, just different dimensions *)
      let mismatches =
        Array.mapi
          (fun i (e, a) ->
            if e <> a then Some (Printf.sprintf "dim %d: %d≠%d" i e a) else None)
          (Array.combine expected actual)
        |> Array.to_list
        |> List.filter_map (fun x -> x)
      in
      String.concat ", " mismatches
    else if expected_size <> actual_size then
      (* Different rank and different total size - show element count *)
      Printf.sprintf "%d→%d elements" expected_size actual_size
    else
      (* Different rank but same size - just show the shapes *)
      Printf.sprintf "incompatible ranks %d and %d" (Array.length expected)
        (Array.length actual)
  in

  cannot ~op ~what:"reshape" ~from:expected_str ~to_:actual_str ~reason ?hint ()

let broadcast_incompatible ~op ~shape1 ~shape2 ?hint () =
  let shape1_str = shape_to_string shape1 in
  let shape2_str = shape_to_string shape2 in

  (* Find specific dimension mismatches *)
  let ndim1 = Array.length shape1 in
  let ndim2 = Array.length shape2 in
  let max_ndim = max ndim1 ndim2 in

  let mismatches = ref [] in
  for i = 0 to max_ndim - 1 do
    let dim1 = if i < ndim1 then shape1.(ndim1 - 1 - i) else 1 in
    let dim2 = if i < ndim2 then shape2.(ndim2 - 1 - i) else 1 in
    if dim1 <> dim2 && dim1 <> 1 && dim2 <> 1 then
      mismatches :=
        Printf.sprintf "dim %d: %d≠%d" (max_ndim - 1 - i) dim1 dim2
        :: !mismatches
  done;

  let reason = String.concat ", " (List.rev !mismatches) in
  let default_hint =
    "broadcasting requires dimensions to be either equal or 1"
  in
  let hint = Option.value hint ~default:default_hint in

  cannot ~op ~what:"broadcast" ~from:shape1_str ~to_:shape2_str ~reason ~hint ()

let dtype_mismatch ~op ~expected ~actual ?hint () =
  let default_hint = Printf.sprintf "cast one array to %s" expected in
  let hint = Option.value hint ~default:default_hint in
  cannot ~op ~what:op ~from:expected ~to_:("with " ^ actual)
    ~reason:"dtype mismatch" ~hint ()

let axis_out_of_bounds ~op ~axis ~ndim ?hint () =
  invalid ~op
    ~what:(Printf.sprintf "axis %d" axis)
    ~reason:(Printf.sprintf "out of bounds for %dD array" ndim)
    ?hint ()

let invalid_shape ~op ~shape ~reason ?hint () =
  invalid ~op
    ~what:(Printf.sprintf "shape %s" (shape_to_string shape))
    ~reason ?hint ()

let empty_input ~op ~what = invalid ~op ~what ~reason:"cannot be empty" ()

let check_bounds ~op ~name ~value ?min ?max () =
  let check_min =
    match min with
    | Some m when value < m -> Some (Printf.sprintf "%s=%d < %d" name value m)
    | _ -> None
  in
  let check_max =
    match max with
    | Some m when value > m -> Some (Printf.sprintf "%s=%d > %d" name value m)
    | _ -> None
  in
  match (check_min, check_max) with
  | Some msg, _ | _, Some msg -> invalid ~op ~what:name ~reason:msg ()
  | None, None -> ()

let check_axes ~op ~axes ~ndim =
  Array.iter
    (fun axis ->
      if axis < -ndim || axis >= ndim then axis_out_of_bounds ~op ~axis ~ndim ())
    axes

let multi_issue ~op issues =
  let formatted_issues =
    List.map
      (fun (issue, detail) -> Printf.sprintf "  - %s: %s" issue detail)
      issues
  in
  let msg =
    Printf.sprintf "%s: invalid configuration\n%s" op
      (String.concat "\n" formatted_issues)
  in
  invalid_arg msg
