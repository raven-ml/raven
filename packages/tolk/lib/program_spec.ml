(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop

(* Types *)

module U = Uop

type var = { name : string; lo : Bound.t; hi : Bound.t; dtype : Dtype.t }
type var_def = { node : U.t; var : var }
type launch_kind = Serial | Thread_groups | Threads

(* The lowered program representation. In the uop IR, a linearized
   program is a {!Uop.t list} returned by the linearizer. *)
type program = U.t list

let slot_of_param (p : U.param_arg) =
  if p.slot >= 0 then Some p.slot else None

let slot_of_define (u : U.t) =
  match U.op u, U.arg u with
  | (Ops.Param | Ops.Buffer), U.Arg.Param_arg param ->
      if param.addrspace = Dtype.Global then slot_of_param param else None
  | _ -> None

(* Trace a pointer expression back to its originating buffer node. *)
let rec trace_to_buffer_uop (u : U.t) : U.t option =
  match slot_of_define u with
  | Some _ -> Some u
  | None ->
      match U.op u with
  | Ops.Index ->
      (match U.as_index u with
       | Some v -> trace_to_buffer_uop v.ptr
       | None -> None)
  | Ops.Cast | Ops.Bitcast | Ops.After | Ops.Shrink ->
      (match U.src u with
       | [| s |] -> trace_to_buffer_uop s
       | srcs when Array.length srcs > 0 -> trace_to_buffer_uop srcs.(0)
       | _ -> None)
  | _ -> None

let trace_to_buffer_slot (u : U.t) : int option =
  Option.bind (trace_to_buffer_uop u) slot_of_define

module Estimates = struct
  type estimate = Int of int | Symbolic of U.t
  type t = { ops : estimate; lds : estimate; mem : estimate }

  let zero = { ops = Int 0; lds = Int 0; mem = Int 0 }

  let of_integer n =
    if Z.fits_int n then Int (Z.to_int n)
    else Symbolic (U.const (Const.integer Dtype.weakint n))

  let of_node node =
    match U.const_int_value node with
    | Some n -> Int n
    | None -> Symbolic node

  let as_node = function
    | Int n -> U.const_int n
    | Symbolic node -> U.cast ~src:node ~dtype:Dtype.weakint

  let add_estimate a b =
    match (a, b) with
    | Int 0, x | x, Int 0 -> x
    | Int a, Int b -> of_integer (Z.add (Z.of_int a) (Z.of_int b))
    | _ -> Symbolic (U.alu_binary ~op:Ops.Add ~lhs:(as_node a) ~rhs:(as_node b))

  let ( + ) a b =
    {
      ops = add_estimate a.ops b.ops;
      lds = add_estimate a.lds b.lds;
      mem = add_estimate a.mem b.mem;
    }

  let of_uop (estimates : U.estimates) =
    let of_estimate = function
      | U.Int n -> Int n
      | U.Sym s -> Symbolic s
    in
    {
      ops = of_estimate estimates.ops;
      lds = of_estimate estimates.lds;
      mem = of_estimate estimates.mem;
    }

  let to_uop t =
    let to_estimate = function Int n -> U.Int n | Symbolic s -> U.Sym s in
    {
      U.ops = to_estimate t.ops;
      lds = to_estimate t.lds;
      mem = to_estimate t.mem;
    }

  let mul_estimate a b =
    match (a, b) with
    | Int 0, _ | _, Int 0 -> Int 0
    | Int 1, x | x, Int 1 -> x
    | Int a, Int b -> of_integer (Z.mul (Z.of_int a) (Z.of_int b))
    | _ -> Symbolic (U.alu_binary ~op:Ops.Mul ~lhs:(as_node a) ~rhs:(as_node b))

  let min_estimate a b =
    match (a, b) with
    | Int a, Int b -> Int (min a b)
    | _ -> of_node (U.smin [ as_node a; as_node b ])

  (* A trip count read from memory (a loop bounded by a loaded id) is known
     only when the kernel runs, so it counts at its upper bound. No tinygrad
     counterpart: the reference has no such loop. *)
  let estimate_of_size u =
    match U.const_int_value u with
    | Some n -> Int n
    | None
      when U.op u = Ops.Load || List.exists (fun n -> U.op n = Ops.Load) (U.backward_slice u) ->
        of_integer (Bound.integer (U.vmax u))
    | None -> Symbolic u

  let rec add_reachable set u =
    if U.Tbl.mem set u then ()
    else begin
      U.Tbl.add set u ();
      Array.iter (add_reachable set) (U.src u)
    end

  let add_indexing_roots set u =
    let add_index idx =
      match U.as_index idx with
      | Some iv ->
          List.iter (add_reachable set) iv.idxs
      | None -> add_reachable set idx
    in
    match U.op u with
    | Ops.Load ->
        (match U.as_load u with
         | Some lv ->
             add_index lv.src;
             Option.iter (add_reachable set) lv.alt;
             Option.iter (add_reachable set) lv.gate
         | None -> ())
    | Ops.Store ->
        (match U.as_store u with
         | Some sv ->
             add_index sv.dst;
             Option.iter (add_reachable set) sv.gate
         | None -> ())
    | Ops.If ->
        (match U.as_if u with
         | Some iv -> add_reachable set iv.cond
         | None -> ())
    | _ -> ()

  let is_reg_access ptr = U.addrspace ptr = Some Dtype.Reg

  (* Bytes touched by a single access to [u]: one lane per shape element,
     each the width of the scalar element type. *)
  let access_bytes u = mul_estimate (Int (U.max_numel u)) (Int (Dtype.itemsize (U.dtype u)))

  let of_program (program : program) =
    let ignored = U.Tbl.create 64 in
    List.iter (add_indexing_roots ignored) program;
    let ops = ref (Int 0) in
    let lds = ref (Int 0) in
    let mem = Hashtbl.create 16 in
    let caps = Hashtbl.create 16 in
    let mults = ref (Int 1) in
    let mult_stack = Stack.create () in
    let add_ops n = ops := add_estimate !ops (mul_estimate !mults n) in
    let add_lds n = lds := add_estimate !lds (mul_estimate !mults n) in
    let add_mem buf op bytes =
      match slot_of_define buf with
      | None -> ()
      | Some slot ->
          let key = (slot, op) in
          let prev =
            match Hashtbl.find_opt mem key with Some v -> v | None -> Int 0
          in
          Hashtbl.replace mem key
            (add_estimate prev (mul_estimate !mults bytes));
          (* Re-reads of a buffer count each byte at most once, so accumulated
             traffic is capped at the buffer's own footprint. *)
          if not (Hashtbl.mem caps slot) then
            Hashtbl.replace caps slot (access_bytes buf)
    in
    List.iter (fun u ->
      begin match U.op u with
      | Ops.Load ->
          (match U.as_load u with
           | Some lv ->
               if not (is_reg_access lv.src) then add_lds (access_bytes u);
               Option.iter
                 (fun buf -> add_mem buf Ops.Load (access_bytes lv.src))
                 (trace_to_buffer_uop lv.src)
           | None -> ())
      | Ops.Store ->
          (match U.as_store u with
           | Some sv ->
               if not (is_reg_access sv.dst) then
                 add_lds (mul_estimate (Int (U.max_numel u))
                   (Int (Dtype.itemsize (U.dtype sv.value))));
               Option.iter
                 (fun buf -> add_mem buf Ops.Store (access_bytes sv.dst))
                 (trace_to_buffer_uop sv.dst)
           | None -> ())
      | _ -> ()
      end;
      match U.op u with
      | Ops.Range ->
          Stack.push !mults mult_stack;
          (* A void range is an unbounded loop closed by a BACKEDGE: its
             trip count is unknown, so the body contributes at multiplicity 1. *)
          if not (Dtype.equal (U.dtype u) Dtype.void) then
            (match U.as_range u with
             | Some rv -> mults := mul_estimate !mults (estimate_of_size rv.size)
             | None -> ())
      | Ops.Backedge ->
          if not (Stack.is_empty mult_stack) then
            mults := Stack.pop mult_stack
      | Ops.End ->
          (match U.as_end u with
           | Some ev ->
               List.iter
                 (fun _ ->
                   if not (Stack.is_empty mult_stack) then
                     mults := Stack.pop mult_stack)
                 ev.ranges
           | None -> ())
      | Ops.Special ->
          (match U.as_special u with
           | Some sv -> mults := mul_estimate !mults (estimate_of_size sv.size)
           | None -> ())
      | Ops.Mulacc when not (U.Tbl.mem ignored u) ->
          add_ops (mul_estimate (Int 2) (Int (U.max_numel u)))
      | op when Ops.Group.is_alu op && not (U.Tbl.mem ignored u) ->
          add_ops (Int (U.max_numel u))
      | Ops.Wmma when not (U.Tbl.mem ignored u) ->
          (match U.as_wmma u with
           | Some { info; _ } ->
               let m, n, k = info.dims in
               add_ops (of_integer Z.(div (of_int 2 * of_int m * of_int n * of_int k)
                 (of_int info.threads)))
           | None -> add_ops (Int 2))
      | _ -> ())
      program;
    let mem =
      Hashtbl.fold
        (fun (slot, _op) accessed acc ->
          let capped =
            match Hashtbl.find_opt caps slot with
            | Some cap -> min_estimate accessed cap
            | None -> accessed
          in
          add_estimate acc capped)
        mem (Int 0)
    in
    let ops =
      match !ops with
      | Int _ as ops -> ops
      | Symbolic _ as ops -> of_node (Symbolic.simplify (as_node ops))
    in
    { ops; lds = !lds; mem }
end

type launch = {
  kind : launch_kind;
  global : U.t array;
  local : U.t array option;
}

type t = {
  target : Target.t;
  name : string;
  src : string;
  device : string;
  program : program;
  lib : bytes option;
  applied_opts : U.Opt.t list;
  vars : var list;
  var_defs : var_def list;
  globals : int list;
  outs : int list;
  ins : int list;
  launch : launch;
  estimates : Estimates.t;
}

let default_dims () = [| U.const_int 1; U.const_int 1; U.const_int 1 |]

(* The renderer consumes formals in linear order. Independently sorting their
   names would bind a different value to each emitted argument. *)
let collect_vars (program : program) =
  let seen = U.Ref_tbl.create 8 in
  List.filter_map (fun u ->
      match U.op u, U.arg u with
      | (Ops.Param | Ops.Buffer), U.Arg.Param_arg
          { name; vmin_vmax; addrspace = Dtype.Alu; slot; _ }
        when not (U.Ref_tbl.mem seen u) ->
          let name, lo, hi = match name, vmin_vmax with
            | Some name, Some (lo, hi) -> name, lo, hi
            | _ -> invalid_arg (Printf.sprintf
                "Program_spec: scalar parameter slot %d requires a name and bounds" slot) in
          U.Ref_tbl.add seen u ();
          Some { node = u; var = { name; lo; hi; dtype = U.dtype u } }
      | _ -> None) program

let collect_globals (program : program) =
  let seen = Hashtbl.create 8 in
  List.filter_map (fun u ->
      match slot_of_define u with
      | Some slot when not (Hashtbl.mem seen slot) ->
          Hashtbl.add seen slot ();
          Some slot
      | _ -> None) program

let collect_buffers (program : program) =
  let outs = ref [] in
  let ins = ref [] in
  List.iter (fun u ->
    match U.op u with
    | Ops.Store ->
        (match U.as_store u with
         | Some sv ->
             Option.iter
               (fun i -> outs := i :: !outs)
               (trace_to_buffer_slot sv.dst)
         | None -> ())
    | Ops.Load ->
        (match U.as_load u with
         | Some lv ->
             Option.iter
               (fun i -> ins := i :: !ins)
               (trace_to_buffer_slot lv.src)
         | None -> ())
    | _ -> ())
    program;
  (List.sort_uniq Int.compare !outs, List.sort_uniq Int.compare !ins)

let set_dim seen label dims axis size =
  if axis < 0 || axis >= Array.length dims then
    invalid_arg (Printf.sprintf "%s axis %d is outside 0..2" label axis);
  if Array.get seen axis then
    invalid_arg (Printf.sprintf "%s axis %d appears more than once" label axis);
  Array.set seen axis true;
  Array.set dims axis size

let collect_launch (program : program) : launch =
  let global = default_dims () in
  let local = default_dims () in
  let group_seen = Array.make 3 false in
  let local_seen = Array.make 3 false in
  let flat_seen = Array.make 3 false in
  let has_group = ref false in
  let has_flat = ref false in
  List.iter (fun u ->
    match U.as_special u with
    | None -> ()
    | Some sv ->
        (match Gpu_dim.of_special_name sv.name with
         | None -> ()
         | Some (Gpu_dim.Group_id axis) ->
             has_group := true;
             set_dim group_seen "group_id" global axis sv.size
         | Some (Gpu_dim.Local_id axis) ->
             has_group := true;
             set_dim local_seen "local_id" local axis sv.size
         | Some (Gpu_dim.Global_idx axis) ->
             has_flat := true;
             set_dim flat_seen "global_idx" global axis sv.size))
    program;
  if !has_group && !has_flat then
    invalid_arg
      "launch metadata cannot mix flat-thread and thread-group specials";
  if !has_flat then { kind = Threads; global; local = None }
  else if !has_group then { kind = Thread_groups; global; local = Some local }
  else { kind = Serial; global; local = Some local }

let of_program ~name ~src ~device ?(target = Target.of_string "") ?lib ?(applied_opts = [])
    ?estimates (program : program) : t =
  let var_defs = collect_vars program in
  let vars = List.map (fun def -> def.var) var_defs in
  let outs, ins = collect_buffers program in
  let globals = collect_globals program in
  let launch = collect_launch program in
  let estimates = match estimates with
    | Some estimates -> estimates
    | None -> Estimates.of_program program
  in
  { target; name; src; device; program; lib; applied_opts; vars; var_defs; globals;
    outs; ins; launch; estimates }

let with_lib lib t = { t with lib = Some lib }
let with_estimates estimates t = { t with estimates }

let with_global_dims dims t =
  { t with launch = { t.launch with global = Array.map U.const_int dims } }

let launch_dim u =
  match U.arg u with
  | U.Arg.Value c ->
      (match Const.view c with
       | Const.Int n -> U.Launch_int (Z.to_int n)
       | Const.Float f -> U.Launch_float f
       | _ -> U.Launch_sym u)
  | _ -> U.Launch_sym u

let program_info t : U.program_info =
  {
    target = t.target;
    global_size = List.map launch_dim (Array.to_list t.launch.global);
    local_size = List.map launch_dim
      (Array.to_list (Option.value t.launch.local ~default:(default_dims ())));
    vars = List.map (fun def -> def.node) t.var_defs;
    globals = t.globals;
    outs = t.outs;
    ins = t.ins;
  }

let to_elf t =
  let lib = match t.lib with
    | Some lib -> lib
    | None -> invalid_arg "Program_spec.to_elf: missing compiled binary" in
  Tiny_elf.{ lib; name = U.sanitize_function_name t.name; target = t.target;
    signature = U.program_signature (program_info t) t.program;
    profile_key = None }

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
let launch_kind t = t.launch.kind
let estimates t = t.estimates

let global_size t = t.launch.global
let local_size t = t.launch.local

let launch_dims t var_vals =
  let eval d =
    try U.sym_infer d var_vals with
    | Invalid_argument message ->
        invalid_arg (Printf.sprintf "program %S: %s" t.name message) in
  let eval_dims dims = Array.map eval dims in
  let global = eval_dims t.launch.global in
  let local = Option.map eval_dims t.launch.local in
  (global, local)
