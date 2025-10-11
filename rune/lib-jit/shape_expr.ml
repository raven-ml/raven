(* Minimal symbolic shape expressions for Rune JIT.

   This module mirrors the capabilities we need from Nx's Symbolic_shape while
   keeping lib-jit independent from Nx. Shapes are arrays of expressions. Each
   expression can be a constant, a symbolic variable, or basic arithmetic
   combinations (addition, multiplication, negation).

   Variables carry a unique id, an optional user-facing name, and bounds. *)

module Var = struct
  type t = { id : int; name : string; min : int; max : int }

  let create ~id ~name ~min ~max =
    if min < 0 then
      invalid_arg "Shape_expr.Var.create: min must be non-negative";
    if min > max then invalid_arg "Shape_expr.Var.create: min must be <= max";
    { id; name; min; max }

  let id v = v.id
  let name v = v.name
  let min v = v.min
  let max v = v.max
end

type expr =
  | Const of int
  | Var of Var.t
  | Add of expr * expr
  | Mul of expr * expr
  | Neg of expr

type shape = expr array

let const n = Const n
let var v = Var v
let add a b = Add (a, b)
let mul a b = Mul (a, b)
let neg e = Neg e
let of_int_array arr = Array.map const arr

let rec to_string_expr = function
  | Const n -> string_of_int n
  | Var v ->
      if v.name = "" then Printf.sprintf "v%d" v.id
      else Printf.sprintf "%s#%d" v.name v.id
  | Add (a, b) ->
      Printf.sprintf "(%s + %s)" (to_string_expr a) (to_string_expr b)
  | Mul (a, b) ->
      Printf.sprintf "(%s * %s)" (to_string_expr a) (to_string_expr b)
  | Neg e -> Printf.sprintf "(-%s)" (to_string_expr e)

let to_string shape =
  "["
  ^ String.concat "; "
      (Array.to_list (Array.map (fun e -> to_string_expr e) shape))
  ^ "]"

let rec eval_expr bindings = function
  | Const n -> Some n
  | Var v -> List.assoc_opt v.id bindings
  | Add (a, b) -> (
      match (eval_expr bindings a, eval_expr bindings b) with
      | Some x, Some y -> Some (x + y)
      | _ -> None)
  | Mul (a, b) -> (
      match (eval_expr bindings a, eval_expr bindings b) with
      | Some x, Some y -> Some (x * y)
      | _ -> None)
  | Neg e -> Option.map (fun x -> -x) (eval_expr bindings e)

let eval bindings shape = Array.map (fun e -> eval_expr bindings e) shape

let to_int_array_exn bindings shape =
  let evaluated = eval bindings shape in
  Array.mapi
    (fun i -> function
      | Some n -> n
      | None ->
          invalid_arg (Printf.sprintf "Shape_expr: dimension %d unresolved" i))
    evaluated

let map f shape = Array.map f shape

let map2 f s1 s2 =
  if Array.length s1 <> Array.length s2 then
    invalid_arg "Shape_expr.map2: shape rank mismatch";
  Array.init (Array.length s1) (fun i -> f s1.(i) s2.(i))

let fold f init shape = Array.fold_left f init shape

let rec upper_bound_expr = function
  | Const n -> n
  | Var v -> v.max
  | Add (a, b) -> upper_bound_expr a + upper_bound_expr b
  | Mul (a, b) -> upper_bound_expr a * upper_bound_expr b
  | Neg _ -> 0

let upper_bounds shape = Array.map upper_bound_expr shape
