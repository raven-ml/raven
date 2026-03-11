(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ir.Program

type var = { name : string; lo : int; hi : int; dtype : Dtype.t }
type core_id = { var_index : int; lo : int; hi : int }

let thread_count core_id = core_id.hi - core_id.lo + 1

type launch_kind = Serial | Thread_groups | Threads

module Scalar_expr = struct
  type t =
    | Const of int
    | Var of int
    | Neg of t
    | Add of t * t
    | Sub of t * t
    | Mul of t * t
    | Idiv of t * t
    | Mod of t * t
    | Shl of t * t
    | Shr of t * t
    | Max of t * t

  let rec eval_with args = function
    | Const n -> n
    | Var i -> args.(i)
    | Neg expr -> -eval_with args expr
    | Add (lhs, rhs) -> eval_with args lhs + eval_with args rhs
    | Sub (lhs, rhs) -> eval_with args lhs - eval_with args rhs
    | Mul (lhs, rhs) -> eval_with args lhs * eval_with args rhs
    | Idiv (lhs, rhs) -> eval_with args lhs / eval_with args rhs
    | Mod (lhs, rhs) -> eval_with args lhs mod eval_with args rhs
    | Shl (lhs, rhs) -> eval_with args lhs lsl eval_with args rhs
    | Shr (lhs, rhs) -> eval_with args lhs asr eval_with args rhs
    | Max (lhs, rhs) -> max (eval_with args lhs) (eval_with args rhs)

  let eval args expr = eval_with (Array.of_list args) expr
end

module Estimates = struct
  type estimate = Int of int | Symbolic of string
  type t = { ops : estimate; lds : estimate; mem : estimate }

  let zero = { ops = Int 0; lds = Int 0; mem = Int 0 }

  let add_estimate a b =
    match (a, b) with
    | Int a, Int b -> Int (a + b)
    | Symbolic s, Int 0 | Int 0, Symbolic s -> Symbolic s
    | Symbolic a, Symbolic b when String.equal a b -> Symbolic a
    | Symbolic a, Int b | Int b, Symbolic a ->
        Symbolic (Printf.sprintf "(%s)+(%d)" a b)
    | Symbolic a, Symbolic b -> Symbolic (Printf.sprintf "(%s)+(%s)" a b)

  let ( + ) a b =
    {
      ops = add_estimate a.ops b.ops;
      lds = add_estimate a.lds b.lds;
      mem = add_estimate a.mem b.mem;
    }

  let of_kernel (estimates : Ir.Kernel.estimates) =
    let of_estimate = function
      | Ir.Kernel.Int n -> Int n
      | Ir.Kernel.Symbolic s -> Symbolic s
    in
    {
      ops = of_estimate estimates.ops;
      lds = of_estimate estimates.lds;
      mem = of_estimate estimates.mem;
    }
end

type launch = {
  kind : launch_kind;
  global : Scalar_expr.t array;
  local : Scalar_expr.t array option;
}

type t = {
  name : string;
  program : Ir.Program.t;
  vars : var list;
  outs : int list;
  ins : int list;
  launch : launch;
  estimates : Estimates.t;
  core_id : core_id option;
}

let unsupported_launch_expr ~ref_ instr =
  invalid_arg
    (Format.asprintf "unsupported launch expression at ref %d: %a" ref_ pp_instr
       instr)

let mark_axis seen ~kind axis =
  if axis < 0 || axis >= Array.length seen then
    invalid_arg (Printf.sprintf "launch axis %d out of bounds" axis);
  if seen.(axis) then
    invalid_arg (Printf.sprintf "%s axis %d appears more than once" kind axis);
  seen.(axis) <- true

let set_axis dims axis value =
  if axis < 0 || axis >= Array.length dims then
    invalid_arg (Printf.sprintf "launch axis %d out of bounds" axis);
  dims.(axis) <- value

let trace_to_param (program : Ir.Program.t) (ref_ : int) : int option =
  let index_ptr =
    match program.(ref_) with
    | Index { ptr; _ } -> Some ptr
    | Cast { src; _ } | Bitcast { src; _ } -> (
        match program.(src) with Index { ptr; _ } -> Some ptr | _ -> None)
    | _ -> None
  in
  Option.bind index_ptr (fun ptr ->
      match program.(ptr) with Param { idx; _ } -> Some idx | _ -> None)

let scalar_expr_of_program (program : Ir.Program.t) var_index_of_ref =
  let rec expr_of_ref ref_ =
    match program.(ref_) with
    | Const { value = Int n; _ } -> Scalar_expr.Const n
    | Define_var _ -> (
        match Hashtbl.find_opt var_index_of_ref ref_ with
        | Some index -> Scalar_expr.Var index
        | None ->
            invalid_arg
              (Printf.sprintf "unknown scalar variable at ref %d" ref_))
    | Cast { src; _ } | Bitcast { src; _ } -> expr_of_ref src
    | Neg { src; _ } -> Scalar_expr.Neg (expr_of_ref src)
    | Add { lhs; rhs; _ } -> Scalar_expr.Add (expr_of_ref lhs, expr_of_ref rhs)
    | Sub { lhs; rhs; _ } -> Scalar_expr.Sub (expr_of_ref lhs, expr_of_ref rhs)
    | Mul { lhs; rhs; _ } -> Scalar_expr.Mul (expr_of_ref lhs, expr_of_ref rhs)
    | Idiv { lhs; rhs; _ } -> Scalar_expr.Idiv (expr_of_ref lhs, expr_of_ref rhs)
    | Mod { lhs; rhs; _ } -> Scalar_expr.Mod (expr_of_ref lhs, expr_of_ref rhs)
    | Shl { lhs; rhs; _ } -> Scalar_expr.Shl (expr_of_ref lhs, expr_of_ref rhs)
    | Shr { lhs; rhs; _ } -> Scalar_expr.Shr (expr_of_ref lhs, expr_of_ref rhs)
    | Max { lhs; rhs; _ } -> Scalar_expr.Max (expr_of_ref lhs, expr_of_ref rhs)
    | instr -> unsupported_launch_expr ~ref_ instr
  in
  expr_of_ref

let default_dims () =
  [| Scalar_expr.Const 1; Scalar_expr.Const 1; Scalar_expr.Const 1 |]

let collect_vars (program : Ir.Program.t) =
  let raw = ref [] in
  Array.iteri
    (fun ref_ (instr : Ir.Program.instr) ->
      match instr with
      | Define_var { name; lo; hi; dtype } ->
          raw := (ref_, { name; lo; hi; dtype }) :: !raw
      | _ -> ())
    program;
  let sorted =
    List.sort
      (fun (_, (a : var)) (_, (b : var)) ->
        compare (a.name, a.lo, a.hi) (b.name, b.lo, b.hi))
      !raw
  in
  let var_index_of_ref = Hashtbl.create (List.length sorted) in
  List.iteri
    (fun index (ref_, _) -> Hashtbl.add var_index_of_ref ref_ index)
    sorted;
  (List.map snd sorted, var_index_of_ref)

let collect_buffers (program : Ir.Program.t) =
  let outs = ref [] in
  let ins = ref [] in
  Array.iter
    (fun (instr : Ir.Program.instr) ->
      match instr with
      | Store { dst; _ } ->
          trace_to_param program dst
          |> Option.iter (fun idx -> outs := idx :: !outs)
      | Load { src; _ } ->
          trace_to_param program src
          |> Option.iter (fun idx -> ins := idx :: !ins)
      | _ -> ())
    program;
  (List.sort_uniq Int.compare !outs, List.sort_uniq Int.compare !ins)

let collect_launch (program : Ir.Program.t) var_index_of_ref scalar_expr =
  let global = default_dims () in
  let local = default_dims () in
  let seen_global = Array.make 3 false in
  let seen_local = Array.make 3 false in
  let has_thread_groups = ref false in
  let has_threads = ref false in
  let core_id = ref None in
  Array.iteri
    (fun ref_ (instr : Ir.Program.instr) ->
      match instr with
      | Special { dim; size; _ } ->
          let expr = scalar_expr size in
          let axis = Ir.special_axis dim in
          begin match dim with
          | Group_id _ ->
              if !has_threads then
                invalid_arg
                  "launch metadata cannot mix flat-thread and thread-group \
                   specials";
              has_thread_groups := true;
              mark_axis seen_global ~kind:"group_id" axis;
              set_axis global axis expr
          | Local_id _ ->
              if !has_threads then
                invalid_arg
                  "launch metadata cannot mix flat-thread and thread-group \
                   specials";
              has_thread_groups := true;
              mark_axis seen_local ~kind:"local_id" axis;
              set_axis local axis expr
          | Global_idx _ ->
              if !has_thread_groups then
                invalid_arg
                  "launch metadata cannot mix flat-thread and thread-group \
                   specials";
              has_threads := true;
              mark_axis seen_global ~kind:"global_idx" axis;
              set_axis global axis expr
          end
      | Define_var { name = "core_id"; lo; hi; _ } -> (
          match !core_id with
          | Some _ -> invalid_arg "core_id must be defined at most once"
          | None when lo <> 0 -> invalid_arg "core_id must have lower bound 0"
          | None ->
              let var_index =
                match Hashtbl.find_opt var_index_of_ref ref_ with
                | Some index -> index
                | None -> invalid_arg "core_id missing from variable table"
              in
              core_id := Some { var_index; lo; hi })
      | _ -> ())
    program;
  let launch =
    if !has_threads then { kind = Threads; global; local = None }
    else if !has_thread_groups then
      { kind = Thread_groups; global; local = Some local }
    else { kind = Serial; global; local = Some local }
  in
  (launch, !core_id)

let of_program ?(estimates = Estimates.zero) ~name (program : Ir.Program.t) : t
    =
  let vars, var_index_of_ref = collect_vars program in
  let scalar_expr = scalar_expr_of_program program var_index_of_ref in
  let outs, ins = collect_buffers program in
  let launch, core_id = collect_launch program var_index_of_ref scalar_expr in
  { name; program; vars; outs; ins; launch; estimates; core_id }

let with_estimates estimates t = { t with estimates }
let name t = t.name
let program t = t.program
let vars t = t.vars
let outs t = t.outs
let ins t = t.ins
let core_id t = t.core_id
let launch_kind t = t.launch.kind
let estimates t = t.estimates

let launch_dims t args =
  let eval_dims dims = Array.map (Scalar_expr.eval args) dims in
  let global = eval_dims t.launch.global in
  let local = Option.map eval_dims t.launch.local in
  (global, local)
