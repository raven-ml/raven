(* Builders, the passes as the pipeline runs them, and the law that a rewrite
   keeps a value, shared by the suite's groups. *)

open Windtrap
open Tolk_next
module Pm = Ops.Pattern_matcher

let uop = Uops.uop
let i n = `Int (Z.of_int n)

(* Passes *)

let rewrite m u = Ops.graph_rewrite ~ctx:() u m
let flatten u = rewrite Simplify.pm_flatten_range u
let unparent u = rewrite Simplify.pm_reduce_unparented u
let collapse u = rewrite Simplify.pm_reduce_collapse u
let reduce_simplify u = rewrite Simplify.pm_reduce_simplify u
let load_collapse u = rewrite Simplify.pm_load_collapse u

(* The range passes read a context, which starts empty; the pipeline flattens
   the ranges of what they build. *)
let split u =
  Ops.graph_rewrite ~ctx:(Ops.Tbl.create 8) u
    (Pm.append Simplify.pm_split_ranges (Pm.with_ctx Simplify.pm_flatten_range))

let simplify u =
  Ops.graph_rewrite ~ctx:(Ops.Tbl.create 8) u
    (Pm.append
       (Pm.with_ctx Simplify.pm_flatten_range)
       Simplify.pm_simplify_ranges)

(* The symbolic pass the pipeline runs before simplifying ranges, which removes
   the guards that hold everywhere. *)
let initial_symbolic u =
  rewrite (Pm.append Symbolic.sym Simplify.pm_flatten_range) u

(* Builders *)

let range ?(axis_type = Ops.Axis_type.Weak) n axis =
  Ops.range ~axis_type (Int n) [ axis ]

let reduce_range n axis = range ~axis_type:Reduce n axis
let var ?dtype name lo hi = Ops.variable ?dtype name (i lo) (i hi)
let buf ?(slot = 0) ?(size = 1024) dt = Ops.param ~shape:[ Int size ] slot dt
let kernel us = Ops.sink ~kernel:(Ops.kernel_info ()) us
let f32 x = Ops.float ~dtype:Float32 x
let i32 n = Ops.int ~dtype:Int32 n
let sum u rs = Ops.reduce u Add rs
let store_at idx x = Ops.store (Ops.index (buf Float32) [ idx ]) x

let gated_load valid idx =
  Ops.load (Ops.index (buf Float32) [ Ops.valid idx valid ]) []

(* Inspection *)

let ranges u = List.filter (fun n -> Ops.op n = Range) (Ops.toposort u)
let size r = Z.to_int (Ops.to_z (Ops.nth r 0))
let sizes u = List.sort compare (List.map size (ranges u))

let count op u =
  List.length (List.filter (fun n -> Ops.op n = op) (Ops.toposort u))

(* Values

   The law that a rewrite keeps the value of what it rewrites: at bindings of
   the leaves it reads (variables, the ranges it runs inside, scalar parameters
   and the elements of storage), the rewritten graph evaluates to what the graph
   does. Integer leaves are drawn from their bounds, the extremes first, floats
   as integers within theirs, so that sums are exact whatever order they add
   in. *)

let draw rng k lo hi =
  match k with 0 -> lo | 1 -> hi | _ -> lo + Random.State.int rng (hi - lo + 1)

let draw_value rng k dt lo hi : Dtype.value =
  match (dt : Dtype.t) with
  | Bool -> `Bool (if k < 2 then k = 1 else Random.State.bool rng)
  | dt when Dtype.is_float dt -> `Float (float_of_int (draw rng k lo hi))
  | _ -> i (draw rng k lo hi)

let bound_int (v : Dtype.value) =
  match v with
  | `Int z -> Z.to_int z
  | `Float f -> int_of_float f
  | `Bool b -> Bool.to_int b

type binding = {
  vars : (string * Dtype.value) list;
  params : (int * Dtype.value) list;
  buffers : (int * Dtype.value array) list;
}

let eval b u =
  Interpreter.eval ~vars:b.vars ~params:b.params ~buffers:b.buffers u

(* The leaves [u] reads, and the ranges it runs inside, each after the leaves
   its value depends on. *)
let leaves u =
  let free = Ops.ranges u in
  List.filter
    (fun n ->
      match Ops.op n with
      | Range -> Ops.Nodes.mem n free
      | Param -> true
      | _ -> false)
    (Ops.toposort u)

(* [binding rng k us] is the [k]th binding of the leaves of [us], or [None] if a
   range it binds is empty. Storage elements lie in [-3, 12], so that an index
   read from memory falls on both sides of a small table's bounds. *)
let binding rng k us =
  let leaves = List.concat_map leaves us in
  let bind b u =
    Option.bind b (fun b ->
        match (Ops.op u, Ops.arg u) with
        | Param, Param { vmin_vmax = Some (lo, hi); dtype; name = Some name; _ }
          ->
            if List.mem_assoc name b.vars then Some b
            else
              let v = draw_value rng k dtype (bound_int lo) (bound_int hi) in
              Some { b with vars = (name, v) :: b.vars }
        | Param, Param { slot; size = None; dtype; _ } ->
            if List.mem_assoc slot b.params then Some b
            else
              Some
                {
                  b with
                  params = (slot, draw_value rng k dtype 0 9) :: b.params;
                }
        | Param, Param { slot; size = Some n; dtype; _ } ->
            if List.mem_assoc slot b.buffers then Some b
            else
              let elements =
                Array.init n (fun j -> draw_value rng (k + j) dtype (-3) 12)
              in
              Some { b with buffers = (slot, elements) :: b.buffers }
        | Range, _ -> (
            let name = Option.get (Interpreter.name u) in
            if List.mem_assoc name b.vars then Some b
            else
              match eval b (Ops.nth u 0) with
              | `Int n when Z.(n > zero) ->
                  let v = i (draw rng k 0 (Z.to_int n - 1)) in
                  Some { b with vars = (name, v) :: b.vars }
              | _ -> None)
        | _ -> Some b)
  in
  List.fold_left bind (Some { vars = []; params = []; buffers = [] }) leaves

let pp_binding ppf b =
  let pp_one ppf (name, v) = Format.fprintf ppf "%s=%a" name Dtype.pp_const v in
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
    pp_one ppf (List.rev b.vars)

(* [keeps_value ~name before after] checks that [after] evaluates to what
   [before] does at [count] bindings of their leaves. A count times a value that
   is no sum of zeros may be [-0.] where the sum is [0.]: zeros compare by
   value. *)
let keeps_value ?(count = 12) ~name before after =
  let rng = Random.State.make [| Hashtbl.hash name |] in
  for k = 0 to count - 1 do
    match binding rng k [ before; after ] with
    | Some b ->
        let msg = Format.asprintf "%s at %a" name pp_binding b in
        let value u =
          match eval b u with `Float f -> `Float (f +. 0.) | v -> v
        in
        equal ~msg Dtypes.const (value before) (value after)
    | None -> ()
  done

(* The law that a rewrite of a kernel keeps what the kernel writes, at bindings
   of its variables and storage. *)

let write = triple int int Dtypes.value

let keeps_writes ?(count = 4) ~name before after =
  let rng = Random.State.make [| Hashtbl.hash name |] in
  let writes b u =
    List.map
      (fun (s, i, v) ->
        (s, i, match v with `Float f -> `Float (f +. 0.) | v -> v))
      (Interpreter.writes ~vars:b.vars ~params:b.params ~buffers:b.buffers u)
  in
  for k = 0 to count - 1 do
    match binding rng k [ before; after ] with
    | Some b ->
        let msg = Format.asprintf "%s at %a" name pp_binding b in
        equal ~msg (list write) (writes b before) (writes b after)
    | None -> ()
  done
