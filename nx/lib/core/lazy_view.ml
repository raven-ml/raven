(** View tracking for lazy tensor operations. *)

type t = { views : View.t list }

(* Compute real strides for a list of views, even if they can't be composed into
   a single view *)
let views_to_real_strides views =
  match views with
  | [] -> None
  | [ single ] -> Some (View.strides single)
  | _ -> (
      (* For multiple views that can't be composed, compute effective strides *)
      (* by tracing how indices map to physical memory locations *)

      (* First, try to simplify/compose as much as possible *)
      let rec simplify_views views =
        match views with
        | [] | [ _ ] -> views
        | v1 :: v2 :: rest -> (
            match View.merge v1 v2 with
            | Some merged -> simplify_views (merged :: rest)
            | None -> v1 :: simplify_views (v2 :: rest))
      in

      let simplified = simplify_views views in

      match simplified with
      | [ single ] -> Some (View.strides single)
      | _ -> (
          (* For symbolic shapes, we might still be able to return strides *)
          (* of the last view even if we can't fully compose *)
          match List.rev simplified with
          | [] -> None
          | last :: _ ->
              (* Check if the last view has valid strides *)
              let last_strides = View.strides last in
              if Array.length last_strides > 0 then Some last_strides else None)
      )

let create shape =
  let view = View.create shape in
  { views = [ view ] }

let shape t =
  match t.views with
  | [] -> Error.failed ~op:"view_tracker.shape" ~what:"empty views list" ()
  | _ ->
      let last_view = List.hd (List.rev t.views) in
      View.shape last_view

let ndim t = Symbolic_shape.rank (shape t)

let numel t =
  let s = shape t in
  let n = Symbolic_shape.rank s in
  if n = 0 then Symbolic_shape.static 1
  else
    Array.fold_left
      (fun acc dim ->
        match (Symbolic_shape.eval_dim acc, Symbolic_shape.eval_dim dim) with
        | Some a, Some b -> Symbolic_shape.static (a * b)
        | _ ->
            (* If either is symbolic, we need to create an expression *)
            Symbolic_shape.mul acc dim)
      s.(0)
      (Array.sub s 1 (n - 1))

let offset t =
  match t.views with
  | [] -> Symbolic_shape.static 0
  | views -> (
      let last_view = List.hd (List.rev views) in
      match View.offset last_view with
      | n -> Symbolic_shape.static n (* Assuming offset is still int *))

let rec is_contiguous t =
  (* First simplify views to merge what can be merged *)
  let simplified = simplify t in
  match simplified.views with
  | [ view ] -> View.is_c_contiguous view
  | _ -> false (* Multiple views are not considered contiguous *)

and simplify t =
  (* First try to merge adjacent views *)
  let rec merge_adjacent views =
    match views with
    | [] | [ _ ] -> views
    | v1 :: v2 :: rest -> (
        match View.merge v1 v2 with
        | Some merged -> merge_adjacent (merged :: rest)
        | None -> v1 :: merge_adjacent (v2 :: rest))
  in

  (* Then simplify each individual view *)
  let merged_views = merge_adjacent t.views in
  let views = List.map View.simplify merged_views in
  { views }

let add_view view t = { views = t.views @ [ view ] }

let get_last_view t =
  match List.rev t.views with
  | [] -> Error.failed ~op:"view_tracker" ~what:"empty views list" ()
  | view :: _ -> view

let reshape new_shape t =
  let current_view = get_last_view t in
  let reshaped = View.reshape current_view new_shape in
  let result = add_view reshaped t in
  result

let permute axes t =
  let current_view = get_last_view t in
  let permuted = View.permute current_view axes in
  let result = add_view permuted t in
  result

let expand new_shape t =
  let current_view = get_last_view t in
  let expanded = View.expand current_view new_shape in
  add_view expanded t

let shrink bounds t =
  let current_view = get_last_view t in
  let shrunk = View.shrink current_view bounds in
  add_view shrunk t

let pad padding t =
  let current_view = get_last_view t in
  let padded = View.pad current_view padding in
  add_view padded t

let flip axes_to_flip t =
  let current_view = get_last_view t in
  let flipped = View.flip current_view axes_to_flip in
  add_view flipped t

let strides t =
  (* First try to simplify *)
  let simplified = simplify t in
  match views_to_real_strides simplified.views with
  | Some s -> Some s
  | None -> (
      (* If we can't compose into a single view, try returning strides of the last view *)
      (* This works for cases like reshape+transpose where each view has valid strides *)
      match List.rev simplified.views with
      | [] -> None
      | last :: _ -> Some (View.strides last))

let can_get_strides t = Option.is_some (strides t)

let is_materializable t =
  (* A view is materializable if it can be represented in memory *)
  match t.views with
  | [] -> false
  | _views ->
      (* Check if the final view composition can be materialized *)
      let final_shape = shape t in
      Symbolic_shape.is_static final_shape
      &&
      (* Could add more checks here *)
      true

let compose t =
  (* Try to compose all views into a single view *)
  match t.views with
  | [] -> None
  | [ single ] -> Some single
  | first :: rest ->
      let result =
        List.fold_left
          (fun acc v ->
            match acc with
            | None -> None
            | Some acc_view ->
                let merged = View.merge acc_view v in
                merged)
          (Some first) rest
      in
      result
