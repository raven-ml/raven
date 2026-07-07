(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop

(* Types *)

module U = Uop

type var = { name : string; lo : int; hi : int; dtype : Dtype.t }
type var_def = { node : U.t; var : var }
type core_id = { var_index : int; lo : int; hi : int }

let thread_count core_id = core_id.hi - core_id.lo + 1

type launch_kind = Serial | Thread_groups | Threads

(* The lowered program representation. In the uop IR, a linearized
   program is a {!Uop.t list} returned by the linearizer. *)
type program = U.t list

let slot_of_param (p : U.param_arg) =
  if p.slot >= 0 then Some p.slot else None

let slot_of_define (u : U.t) =
  match U.op u, U.arg u with
  | Ops.Param, U.Arg.Param_arg param ->
      if param.addrspace = Dtype.Alu then None else slot_of_param param
  | Ops.Buffer, U.Arg.Param_arg param -> slot_of_param param
  | _ -> None

(* Trace a pointer expression to its originating buffer slot. *)
let rec trace_to_buffer_slot (u : U.t) : int option =
  match slot_of_define u with
  | Some _ as slot -> slot
  | None ->
      match U.op u with
  | Ops.Index ->
      (match U.as_index u with
       | Some v -> trace_to_buffer_slot v.ptr
       | None -> None)
  | Ops.Cast | Ops.Bitcast | Ops.After | Ops.Shrink ->
      (match U.src u with
       | [| s |] -> trace_to_buffer_slot s
       | srcs when Array.length srcs > 0 -> trace_to_buffer_slot srcs.(0)
       | _ -> None)
  | _ -> None

module Estimates = struct
  type estimate = Int of int | Symbolic of U.t
  type t = { ops : estimate; lds : estimate; mem : estimate }

  let zero = { ops = Int 0; lds = Int 0; mem = Int 0 }

  let add_estimate a b =
    match (a, b) with
    | Int a, Int b -> Int (a + b)
    | Symbolic s, Int 0 | Int 0, Symbolic s -> Symbolic s
    | Symbolic a, Symbolic b when U.equal a b -> Symbolic a
    | Symbolic a, Int b ->
        Symbolic (U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:(U.const_int b))
    | Int a, Symbolic b ->
        Symbolic (U.alu_binary ~op:Ops.Add ~lhs:(U.const_int a) ~rhs:b)
    | Symbolic a, Symbolic b ->
        Symbolic (U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b)

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

  let mul_estimate a b =
    match (a, b) with
    | Int 0, _ | _, Int 0 -> Int 0
    | Int 1, x | x, Int 1 -> x
    | Int a, Int b -> Int (a * b)
    | Symbolic a, Symbolic b ->
        Symbolic (U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:b)
    | Symbolic a, Int b ->
        Symbolic (U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:(U.const_int b))
    | Int a, Symbolic b ->
        Symbolic (U.alu_binary ~op:Ops.Mul ~lhs:(U.const_int a) ~rhs:b)

  let estimate_of_size u =
    match U.const_int_value u with
    | Some n -> Int n
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

  let is_reg_access ptr =
    match U.dtype ptr with
    | Dtype.Ptr p -> Dtype.Ptr.addrspace p = Reg
    | Dtype.Val _ -> false

  let value_itemsize u = Dtype.itemsize (U.dtype u)

  let access_itemsize u =
    match U.dtype u with
    | Dtype.Ptr p -> Dtype.Val.itemsize (Dtype.Ptr.base p)
    | Dtype.Val v -> Dtype.Val.itemsize v

  let of_program (program : program) =
    let ignored = U.Tbl.create 64 in
    List.iter (add_indexing_roots ignored) program;
    let ops = ref (Int 0) in
    let lds = ref (Int 0) in
    let mem = Hashtbl.create 16 in
    let mults = ref (Int 1) in
    let mult_stack = Stack.create () in
    let add_ops n = ops := add_estimate !ops (mul_estimate !mults (Int n)) in
    let add_lds n = lds := add_estimate !lds (mul_estimate !mults (Int n)) in
    let add_mem param_idx op bytes =
      let key = (param_idx, op) in
      let prev = match Hashtbl.find_opt mem key with
        | Some v -> v
        | None -> Int 0
      in
      Hashtbl.replace mem key (add_estimate prev (mul_estimate !mults (Int bytes)))
    in
    List.iter (fun u ->
      begin match U.op u with
      | Ops.Load ->
          (match U.as_load u with
           | Some lv ->
               if not (is_reg_access lv.src) then add_lds (value_itemsize u);
               Option.iter
                 (fun idx -> add_mem idx Ops.Load (access_itemsize lv.src))
                 (trace_to_buffer_slot lv.src)
           | None -> ())
      | Ops.Store ->
          (match U.as_store u with
           | Some sv ->
               if not (is_reg_access sv.dst) then add_lds (value_itemsize sv.value);
               Option.iter
                 (fun idx -> add_mem idx Ops.Store (access_itemsize sv.dst))
                 (trace_to_buffer_slot sv.dst)
           | None -> ())
      | _ -> ()
      end;
      match U.op u with
      | Ops.Range ->
          (match U.as_range u with
           | Some rv ->
               Stack.push !mults mult_stack;
               mults := mul_estimate !mults (estimate_of_size rv.size)
           | None -> ())
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
      | Ops.Param ->
          (match U.arg u with
           | U.Arg.Param_arg
               { name = Some "core_id"; vmin_vmax = Some (_lo, hi); _ } ->
               mults := mul_estimate !mults (Int (Stdlib.( + ) hi 1))
           | _ -> ())
      | Ops.Mulacc when not (U.Tbl.mem ignored u) ->
          add_ops (2 * Dtype.count (U.dtype u))
      | op when Ops.Group.is_alu op && not (U.Tbl.mem ignored u) ->
          add_ops (Dtype.count (U.dtype u))
      | Ops.Wmma when not (U.Tbl.mem ignored u) ->
          add_ops 2
      | _ -> ())
      program;
    let mem =
      Hashtbl.fold (fun _ v acc -> add_estimate acc v) mem (Int 0)
    in
    { ops = !ops; lds = !lds; mem }
end

type launch = {
  kind : launch_kind;
  global : U.t array;
  local : U.t array option;
}

type t = {
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
  core_id : core_id option;
  aux : string list;
}

let default_dims () = [| U.const_int 1; U.const_int 1; U.const_int 1 |]

(* Collect bounded scalar parameter nodes visible from the program. *)
let collect_vars (program : program) =
  let raw = ref [] in
  List.iter (fun u ->
    match U.op u, U.arg u with
    | ( Ops.Param,
        U.Arg.Param_arg { name = Some name; vmin_vmax = Some (lo, hi); _ } )
      ->
        (match U.dtype u with
         | Dtype.Val _ as dtype ->
             raw := { node = u; var = { name; lo; hi; dtype } } :: !raw
         | Dtype.Ptr _ -> ())
    | _ -> ())
    program;
  let sorted =
    List.sort
      (fun (a : var_def) (b : var_def) ->
        compare
          (a.var.name, a.var.lo, a.var.hi)
          (b.var.name, b.var.lo, b.var.hi))
      !raw
  in
  (* Deduplicate by (name, lo, hi). *)
  let rec dedup = function
    | [] -> []
    | [x] -> [x]
    | { var = { name = an; lo = alo; hi = ahi; _ }; _ }
      :: ({ var = { name = bn; lo = blo; hi = bhi; _ }; _ } :: _ as rest)
      when an = bn && alo = blo && ahi = bhi ->
        dedup rest
    | a :: rest -> a :: dedup rest
  in
  dedup sorted

let collect_globals (program : program) =
  let globals = ref [] in
  List.iter (fun u ->
    Option.iter (fun slot -> globals := slot :: !globals) (slot_of_define u))
    program;
  List.sort_uniq Int.compare !globals

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

let collect_core_id vars =
  let found = ref None in
  List.iteri (fun i (v : var) ->
    if v.name = "core_id" then begin
      if v.lo <> 0 then invalid_arg "core_id must have lower bound 0";
      match !found with
      | Some _ -> invalid_arg "core_id appears more than once"
      | None -> found := Some { var_index = i; lo = v.lo; hi = v.hi }
    end)
    vars;
  !found

let collect_launch (program : program) (vars : var list) : launch * core_id option =
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
  let core_id = collect_core_id vars in
  begin match core_id, !has_group, !has_flat with
  | Some cid, false, false -> global.(0) <- U.const_int (thread_count cid)
  | _ -> ()
  end;
  let launch =
    if !has_flat then { kind = Threads; global; local = None }
    else if !has_group then { kind = Thread_groups; global; local = Some local }
    else { kind = Serial; global; local = Some local }
  in
  (launch, core_id)

let of_program ~name ~src ~device ?lib ?(applied_opts = [])
    ?estimates ?(aux = []) (program : program) : t =
  let var_defs = collect_vars program in
  let vars = List.map (fun def -> def.var) var_defs in
  let outs, ins = collect_buffers program in
  let globals =
    List.sort_uniq Int.compare (collect_globals program @ outs @ ins)
  in
  let launch, core_id = collect_launch program vars in
  let estimates = match estimates with
    | Some estimates -> estimates
    | None -> Estimates.of_program program
  in
  { name; src; device; program; lib; applied_opts; vars; var_defs; globals;
    outs; ins; launch; estimates; core_id; aux }

let with_lib lib t = { t with lib = Some lib }
let with_estimates estimates t = { t with estimates }

let with_global_dims dims t =
  { t with launch = { t.launch with global = Array.map U.const_int dims } }

let launch_dim u =
  match U.arg u with
  | U.Arg.Value c ->
      (match Const.view c with
       | Const.Int n -> U.Launch_int (Int64.to_int n)
       | Const.Float f -> U.Launch_float f
       | _ -> U.Launch_sym u)
  | _ -> U.Launch_sym u

let fixed_dims dims =
  let rec loop acc i =
    if i < 0 then Some acc
    else
      match U.const_int_value dims.(i) with
      | Some n -> loop (n :: acc) (i - 1)
      | None -> None
  in
  loop [] (Array.length dims - 1)

let program_info t : U.program_info =
  {
    name = t.name;
    global_size = List.map launch_dim (Array.to_list t.launch.global);
    local_size = Option.bind t.launch.local fixed_dims;
    vars = List.map (fun def -> def.node) t.var_defs;
    globals = t.globals;
    outs = t.outs;
    ins = t.ins;
    aux = t.aux;
  }

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

let floor_div a b =
  let q = a / b and r = a mod b in
  if r <> 0 && ((a < 0) <> (b < 0)) then q - 1 else q

let floor_mod a b = a - (floor_div a b * b)

let launch_dims t var_vals =
  let rec eval d =
    match U.arg d with
    | U.Arg.Value c ->
        (match Const.view c with
         | Const.Int n -> Int64.to_int n
         | Const.Bool b -> if b then 1 else 0
         | _ -> invalid_arg "launch dimension is not an integer expression")
    | _ ->
        (match U.op d, U.arg d with
         | ( Ops.Param,
             U.Arg.Param_arg { name = Some name; vmin_vmax = Some _; _ } )
           ->
             (match List.assoc_opt name var_vals with
              | Some v -> v
              | None ->
                  invalid_arg
                    (Printf.sprintf "missing launch variable %S" name))
         | _ ->
             match U.op d, U.src d with
             | (Ops.Cast | Ops.Bitcast), [| x |] -> eval x
             | Ops.Add, [| a; b |] -> eval a + eval b
             | Ops.Sub, [| a; b |] -> eval a - eval b
             | Ops.Mul, [| a; b |] -> eval a * eval b
             | Ops.Cdiv, [| a; b |] -> eval a / eval b
             | Ops.Cmod, [| a; b |] ->
                 let a = eval a and b = eval b in
                 a - (a / b) * b
             | Ops.Floordiv, [| a; b |] -> floor_div (eval a) (eval b)
             | Ops.Floormod, [| a; b |] -> floor_mod (eval a) (eval b)
             | Ops.Max, [| a; b |] -> max (eval a) (eval b)
             | Ops.Shl, [| a; b |] -> eval a lsl eval b
             | Ops.Shr, [| a; b |] -> eval a lsr eval b
             | _ ->
                 invalid_arg
                   (Printf.sprintf "unsupported launch dimension op %s"
                      (Ops.name (U.op d))))
  in
  let eval_dims dims = Array.map eval dims in
  let global = eval_dims t.launch.global in
  let local = Option.map eval_dims t.launch.local in
  (global, local)
