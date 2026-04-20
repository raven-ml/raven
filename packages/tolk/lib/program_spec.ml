(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir

type var = { name : string; lo : int; hi : int; dtype : Dtype.t }
type core_id = { var_index : int; lo : int; hi : int }

let thread_count core_id = core_id.hi - core_id.lo + 1

type launch_kind = Serial | Thread_groups | Threads

module K = Kernel

module Estimates = struct
  type estimate = Int of int | Symbolic of Kernel.t
  type t = { ops : estimate; lds : estimate; mem : estimate }

  let zero = { ops = Int 0; lds = Int 0; mem = Int 0 }

  let add_estimate a b =
    match (a, b) with
    | Int a, Int b -> Int (a + b)
    | Symbolic s, Int 0 | Int 0, Symbolic s -> Symbolic s
    | Symbolic a, Symbolic b when a == b -> Symbolic a
    | Symbolic a, Int b ->
        Symbolic (Kernel.binary ~op:`Add ~lhs:a ~rhs:(Kernel.const_int b))
    | Int a, Symbolic b ->
        Symbolic (Kernel.binary ~op:`Add ~lhs:(Kernel.const_int a) ~rhs:b)
    | Symbolic a, Symbolic b ->
        Symbolic (Kernel.binary ~op:`Add ~lhs:a ~rhs:b)

  let ( + ) a b =
    {
      ops = add_estimate a.ops b.ops;
      lds = add_estimate a.lds b.lds;
      mem = add_estimate a.mem b.mem;
    }

  let of_kernel (estimates : Kernel.estimates) =
    let of_estimate = function
      | Kernel.Int n -> Int n
      | Kernel.Symbolic s -> Symbolic s
    in
    {
      ops = of_estimate estimates.ops;
      lds = of_estimate estimates.lds;
      mem = of_estimate estimates.mem;
    }

  let of_program (program : Program.t) =
    let ( + ) = Stdlib.( + ) in
    let ( * ) = Stdlib.( * ) in
    let module P = Program in
    let flops = ref 0 in
    let lds = ref 0 in
    let mem : (int * bool, int) Hashtbl.t = Hashtbl.create 16 in
    let mults = ref 1 in
    let mult_stack = Stack.create () in
    let const_of_id id =
      match P.view program id with
      | Const { value; _ } ->
          (match Const.view value with Int n -> Int64.to_int n | _ -> 1)
      | _ -> 1
    in
    let scalar_itemsize (dtype : Dtype.t) =
      Dtype.itemsize (Dtype.scalarize dtype)
    in
    let rec find_param id =
      match P.view program id with
      | Param { idx; dtype; _ } -> Some (idx, dtype)
      | Index { ptr; _ } -> find_param ptr
      | After { src; _ } -> find_param src
      | _ -> None
    in
    let track_mem key (ptr : Dtype.Ptr.t) itemsize =
      let prev = Option.value ~default:0 (Hashtbl.find_opt mem key) in
      let accessed = prev + itemsize * !mults in
      Hashtbl.replace mem key
        (if Dtype.Ptr.size ptr > 0 then
           min accessed (Dtype.Ptr.size ptr * Dtype.Val.itemsize (Dtype.Ptr.base ptr))
         else accessed)
    in
    let is_reg_access id =
      match P.view program id with
      | Index { dtype = ptr; _ } -> Dtype.Ptr.addrspace ptr = Reg
      | _ -> false
    in
    let store_itemsize value =
      match P.dtype program value with
      | Some dt -> scalar_itemsize (Dtype.Val dt)
      | None -> 1
    in
    (* Exclude load/store indexing and if-conditions from FLOP counting. *)
    let dont_count : (P.id, unit) Hashtbl.t = Hashtbl.create 64 in
    let rec collect_deps_range_gated id =
      if not (Hashtbl.mem dont_count id) then begin
        Hashtbl.replace dont_count id ();
        match P.view program id with
        | Range _ -> ()
        | _ -> List.iter collect_deps_range_gated (P.children program id)
      end
    in
    let rec collect_deps_all id =
      if not (Hashtbl.mem dont_count id) then begin
        Hashtbl.replace dont_count id ();
        List.iter collect_deps_all (P.children program id)
      end
    in
    P.iteri
      (fun _id v ->
        match v with
        | Load { src; _ } | Store { dst = src; _ } ->
            (match P.view program src with
             | Index { idxs; gate; _ } ->
                 List.iter collect_deps_range_gated idxs;
                 Option.iter collect_deps_range_gated gate
             | _ -> ())
        | If { cond; _ } -> collect_deps_all cond
        | _ -> ())
      program;
    P.iteri
      (fun id v ->
        (match v with
         | Load { src; dtype; _ } ->
             (match find_param src with
              | Some (idx, ptr) ->
                  track_mem (idx, false) ptr (scalar_itemsize (Dtype.Val dtype))
              | None -> ())
         | Store { dst; value; _ } ->
             (match find_param dst with
              | Some (idx, ptr) ->
                  track_mem (idx, true) ptr (store_itemsize value)
              | None -> ())
         | _ -> ());
        match v with
        | Range { size; _ } ->
            Stack.push !mults mult_stack;
            mults := !mults * const_of_id size
        | End_range _ -> mults := Stack.pop mult_stack
        | Special { size; _ } -> mults := !mults * const_of_id size
        | Define_var { name; hi; _ } when name = "core_id" ->
            mults := !mults * (hi + 1)
        | Load { src; dtype; _ } ->
            if not (is_reg_access src) then
              lds := !lds + scalar_itemsize (Dtype.Val dtype) * !mults
        | Store { dst; value; _ } ->
            if not (is_reg_access dst) then
              lds := !lds + store_itemsize value * !mults
        | Unary { dtype; _ } | Binary { dtype; _ }
          when not (Hashtbl.mem dont_count id) ->
            flops := !flops + !mults * Dtype.Val.count dtype
        | Ternary { op = `Mulacc; dtype; _ }
          when not (Hashtbl.mem dont_count id) ->
            flops := !flops + 2 * !mults * Dtype.Val.count dtype
        | Ternary { dtype; _ } when not (Hashtbl.mem dont_count id) ->
            flops := !flops + !mults * Dtype.Val.count dtype
        | Wmma { dims = m, n, k; threads; _ }
          when not (Hashtbl.mem dont_count id) ->
            flops := !flops + 2 * (m * n * k / threads) * !mults
        | _ -> ())
      program;
    let total_mem = Hashtbl.fold (fun _ bytes acc -> acc + bytes) mem 0 in
    { ops = Int !flops; lds = Int !lds; mem = Int total_mem }
end

type launch = {
  kind : launch_kind;
  global : K.t array;
  local : K.t array option;
}

type t = {
  name : string;
  src : string;
  device : string;
  program : Program.t;
  lib : bytes option;
  applied_opts : Kernel.Opt.t list;
  vars : var list;
  globals : int list;
  outs : int list;
  ins : int list;
  launch : launch;
  estimates : Estimates.t;
  core_id : core_id option;
}

let unsupported_launch_expr ~ref_ view =
  invalid_arg
    (Format.asprintf "unsupported launch expression at ref %d: %a" ref_
       Program.pp_view view)

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

let trace_to_param (program : Program.t) (ref_ : int) : int option =
  let index_ptr =
    match Program.view program ref_ with
    | Index { ptr; _ } -> Some ptr
    | Cast { src; _ } | Bitcast { src; _ } -> (
        match Program.view program src with
        | Index { ptr; _ } -> Some ptr
        | _ -> None)
    | _ -> None
  in
  Option.bind index_ptr (fun ptr ->
      match Program.view program ptr with
      | Param { idx; _ } -> Some idx
      | _ -> None)

(* Convert a Program IR reference to a K.t expression for launch dimensions. *)
let kernel_expr_of_program (program : Program.t) var_nodes =
  let rec expr_of_ref ref_ =
    match Program.view program ref_ with
    | Const { value; _ } -> (
        match Const.view value with
        | Int n -> K.const_int (Int64.to_int n)
        | _ ->
            invalid_arg
              (Printf.sprintf "non-integer constant at ref %d" ref_))
    | Define_var _ -> (
        match Hashtbl.find_opt var_nodes ref_ with
        | Some node -> node
        | None ->
            invalid_arg
              (Printf.sprintf "unknown scalar variable at ref %d" ref_))
    | Cast { src; _ } | Bitcast { src; _ } -> expr_of_ref src
    | Unary { op = `Neg; src; _ } -> K.unary ~op:`Neg ~src:(expr_of_ref src)
    | Binary { op; lhs; rhs; _ } ->
        K.binary ~op ~lhs:(expr_of_ref lhs) ~rhs:(expr_of_ref rhs)
    | v -> unsupported_launch_expr ~ref_ v
  in
  expr_of_ref

let default_dims () = [| K.const_int 1; K.const_int 1; K.const_int 1 |]

let collect_vars (program : Program.t) =
  let raw = ref [] in
  let var_nodes = Hashtbl.create 8 in
  Program.iteri
    (fun ref_ (v : Program.view) ->
      match v with
      | Define_var { name; lo; hi; dtype } ->
          raw := (ref_, { name; lo; hi; dtype = Dtype.Val dtype }) :: !raw;
          Hashtbl.replace var_nodes ref_
            (K.define_var ~name ~lo ~hi ~dtype ())
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
  (List.map snd sorted, var_index_of_ref, var_nodes)

let collect_buffers (program : Program.t) =
  let outs = ref [] in
  let ins = ref [] in
  Program.iteri
    (fun _id (v : Program.view) ->
      match v with
      | Store { dst; _ } ->
          trace_to_param program dst
          |> Option.iter (fun idx -> outs := idx :: !outs)
      | Load { src; _ } ->
          trace_to_param program src
          |> Option.iter (fun idx -> ins := idx :: !ins)
      | _ -> ())
    program;
  (List.sort_uniq Int.compare !outs, List.sort_uniq Int.compare !ins)

(* Extracts launch grid/block dimensions from Special nodes in the program.
   Enforces mutual exclusion between the flat-thread paradigm (global_idx only)
   and the thread-group paradigm (group_id + local_id), raising if both are
   mixed. Also captures the optional core_id variable for CPU dispatch. *)
let collect_launch (program : Program.t) var_index_of_ref scalar_expr =
  let global = default_dims () in
  let local = default_dims () in
  let seen_global = Array.make 3 false in
  let seen_local = Array.make 3 false in
  let has_thread_groups = ref false in
  let has_threads = ref false in
  let core_id = ref None in
  Program.iteri
    (fun ref_ (v : Program.view) ->
      match v with
      | Special { dim; size; _ } ->
          let expr = scalar_expr size in
          let axis = Special_dim.axis dim in
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
              global.(0) <- K.const_int (hi + 1);
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

let of_program ~name ~src ~device ?lib ?(applied_opts = [])
    ?(estimates = Estimates.zero) (program : Program.t) : t =
  let vars, var_index_of_ref, var_nodes = collect_vars program in
  let kernel_expr = kernel_expr_of_program program var_nodes in
  let outs, ins = collect_buffers program in
  let globals = List.sort_uniq Int.compare (outs @ ins) in
  let launch, core_id = collect_launch program var_index_of_ref kernel_expr in
  { name; src; device; program; lib; applied_opts; vars; globals; outs; ins;
    launch; estimates; core_id }

let with_lib lib t = { t with lib = Some lib }
let with_estimates estimates t = { t with estimates }

let with_global_dims dims t =
  { t with launch = { t.launch with global = Array.map K.const_int dims } }

let name t = t.name
let src t = t.src
let device t = t.device
let program t = t.program
let lib t = t.lib
let applied_opts t = t.applied_opts
let vars t = t.vars
let globals t = t.globals
let outs t = t.outs
let ins t = t.ins
let core_id t = t.core_id
let launch_kind t = t.launch.kind
let estimates t = t.estimates

let global_size t = t.launch.global
let local_size t = t.launch.local

let launch_dims t var_vals =
  let eval_dims dims = Array.map (fun d -> K.sym_infer d var_vals) dims in
  let global = eval_dims t.launch.global in
  let local = Option.map eval_dims t.launch.local in
  (global, local)
