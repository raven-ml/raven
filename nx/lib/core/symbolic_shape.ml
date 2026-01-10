(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Symbolic shapes for shape-polymorphic tensors. *)

type var = {
  id : int;
  name : string;
  min : int;
  max : int;
  mutable value : int option;
}

type expr =
  | Const of int
  | Var of var
  | Add of expr * expr
  | Mul of expr * expr
  | Neg of expr

type dim = expr
type t = dim array

let next_id =
  let counter = ref 0 in
  fun () ->
    let id = !counter in
    incr counter;
    id

let static n =
  if n < 0 then
    Error.invalid ~op:"static"
      ~what:(Printf.sprintf "dimension %d" n)
      ~reason:"negative dimension" ();
  Const n

let var name ~min ~max =
  if min < 0 then
    Error.invalid ~op:"dynamic"
      ~what:(Printf.sprintf "min=%d" min)
      ~reason:"must be non-negative" ();
  if min > max then
    Error.invalid ~op:"dynamic"
      ~what:(Printf.sprintf "bounds [%d, %d]" min max)
      ~reason:(Printf.sprintf "min > max")
      ();
  { id = next_id (); name; min; max; value = None }

let dim_of_var var = Var var
let dynamic name ~min ~max = dim_of_var (var name ~min ~max)
let add d1 d2 = Add (d1, d2)
let mul d1 d2 = Mul (d1, d2)
let neg d = Neg d
let of_ints arr = Array.map (fun n -> static n) arr
let of_list lst = of_ints (Array.of_list lst)

let bind_var var value =
  if value < var.min || value > var.max then
    Error.invalid ~op:"bind"
      ~what:(Printf.sprintf "value %d for variable %s" value var.name)
      ~reason:(Printf.sprintf "outside bounds [%d, %d]" var.min var.max)
      ();
  var.value <- Some value

let bind target value shape =
  bind_var target value;
  let rec find_and_bind expr =
    match expr with
    | Var v when v.id = target.id -> bind_var v value
    | Add (e1, e2) | Mul (e1, e2) ->
        find_and_bind e1;
        find_and_bind e2
    | Neg e -> find_and_bind e
    | _ -> ()
  in
  Array.iter find_and_bind shape

let rec eval_expr = function
  | Const n -> Some n
  | Var var -> var.value
  | Add (e1, e2) -> (
      match (eval_expr e1, eval_expr e2) with
      | Some v1, Some v2 -> Some (v1 + v2)
      | _ -> None)
  | Mul (e1, e2) -> (
      match (eval_expr e1, eval_expr e2) with
      | Some v1, Some v2 -> Some (v1 * v2)
      | _ -> None)
  | Neg e -> ( match eval_expr e with Some v -> Some (-v) | None -> None)

let eval_dim = eval_expr

let eval shape =
  let rec loop acc i =
    if i < 0 then Some (Array.of_list acc)
    else
      match eval_dim shape.(i) with
      | None -> None
      | Some n -> loop (n :: acc) (i - 1)
  in
  loop [] (Array.length shape - 1)

let partial_eval shape = Array.map eval_expr shape

let rec expr_is_bound = function
  | Const _ -> true
  | Var v -> Option.is_some v.value
  | Add (e1, e2) | Mul (e1, e2) -> expr_is_bound e1 && expr_is_bound e2
  | Neg e -> expr_is_bound e

let is_fully_bound shape = Array.for_all expr_is_bound shape

let rec expr_vars = function
  | Const _ -> []
  | Var v -> [ v ]
  | Add (e1, e2) | Mul (e1, e2) -> expr_vars e1 @ expr_vars e2
  | Neg e -> expr_vars e

let vars shape =
  let rec dedup acc = function
    | [] -> List.rev acc
    | v :: rest ->
        if List.exists (fun existing -> existing.id = v.id) acc then
          dedup acc rest
        else dedup (v :: acc) rest
  in
  shape |> Array.to_list |> List.concat_map expr_vars |> dedup []

let var_id v = v.id
let var_name v = v.name
let var_bounds v = (v.min, v.max)

let rec expr_is_static = function
  | Const _ -> true
  | Var _ -> false
  | Add (e1, e2) | Mul (e1, e2) -> expr_is_static e1 && expr_is_static e2
  | Neg e -> expr_is_static e

let is_static shape = Array.for_all expr_is_static shape
let rank shape = Array.length shape

let to_string shape =
  let rec expr_to_string = function
    | Const n -> string_of_int n
    | Var var -> (
        match var.value with
        | None ->
            if var.name = "" then Printf.sprintf "v%d" var.id
            else Printf.sprintf "%s#%d" var.name var.id
        | Some n ->
            if var.name = "" then Printf.sprintf "v%d=%d" var.id n
            else Printf.sprintf "%s#%d=%d" var.name var.id n)
    | Add (e1, e2) ->
        Printf.sprintf "(%s + %s)" (expr_to_string e1) (expr_to_string e2)
    | Mul (e1, e2) ->
        Printf.sprintf "(%s * %s)" (expr_to_string e1) (expr_to_string e2)
    | Neg e -> Printf.sprintf "(-%s)" (expr_to_string e)
  in
  "["
  ^ String.concat "; " (Array.to_list (Array.map expr_to_string shape))
  ^ "]"

let rec expr_equal e1 e2 =
  match (e1, e2) with
  | Const n1, Const n2 -> n1 = n2
  | Var v1, Var v2 -> v1.id = v2.id
  | Add (a1, b1), Add (a2, b2) | Mul (a1, b1), Mul (a2, b2) ->
      expr_equal a1 a2 && expr_equal b1 b2
  | Neg e1', Neg e2' -> expr_equal e1' e2'
  | _ -> false

let equal s1 s2 =
  Array.length s1 = Array.length s2 && Array.for_all2 expr_equal s1 s2

let numel shape =
  let n = Array.length shape in
  if n = 0 then Some 1
  else
    let rec compute_product i acc =
      if i >= n then acc
      else
        match (acc, eval_dim shape.(i)) with
        | None, _ -> None
        | _, None -> None
        | Some acc_val, Some dim_val ->
            compute_product (i + 1) (Some (acc_val * dim_val))
    in
    compute_product 0 (Some 1)

(** Special dimension value representing "infer from context" (like -1 in NumPy
    reshape) *)
let infer = Const (-1)

let is_infer dim = match eval_dim dim with Some -1 -> true | _ -> false

let resolve_reshape ~from_shape ~to_shape =
  (* Resolve a reshape operation with potential -1 (infer) dimensions *)
  match numel from_shape with
  | None -> None (* Can't resolve if source shape is not fully known *)
  | Some total_elements -> (
      (* Count infer dimensions and compute known product *)
      let infer_indices = ref [] in
      let known_product = ref 1 in
      let resolved = Array.copy to_shape in

      Array.iteri
        (fun i dim ->
          if is_infer dim then infer_indices := i :: !infer_indices
          else
            match eval_dim dim with
            | Some n when n > 0 -> known_product := !known_product * n
            | Some n ->
                Error.invalid ~op:"resolve_reshape"
                  ~what:(Printf.sprintf "dimension %d" i)
                  ~reason:(Printf.sprintf "invalid size %d" n)
                  ()
            | None -> () (* Keep symbolic dimension as is *))
        to_shape;

      match !infer_indices with
      | [] -> Some resolved (* No inference needed *)
      | [ idx ] ->
          (* Exactly one dimension to infer *)
          if !known_product > 0 && total_elements mod !known_product = 0 then (
            let inferred_size = total_elements / !known_product in
            resolved.(idx) <- static inferred_size;
            Some resolved)
          else None (* Can't evenly divide *)
      | _ ->
          Error.invalid ~op:"resolve_reshape" ~what:"shape"
            ~reason:"can only infer one dimension" ())

let substitute bindings shape =
  (* Substitute variable bindings into a shape *)
  Array.map
    (fun dim ->
      let rec subst = function
        | Const n -> Const n
        | Var v as var -> (
            let rec lookup = function
              | [] -> None
              | (var', value) :: rest ->
                  if var'.id = v.id then Some value else lookup rest
            in
            match lookup bindings with Some value -> Const value | None -> var)
        | Add (e1, e2) -> Add (subst e1, subst e2)
        | Mul (e1, e2) -> Mul (subst e1, subst e2)
        | Neg e -> Neg (subst e)
      in
      subst dim)
    shape
