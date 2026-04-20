(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Hash-consed tensor graph IR, modelled after Kernel.t. *)

type device = Single of string | Multi of string list
type metadata = { name : string; caller : string; backward : bool }

type view =
  | Sink of { srcs : t list; kernel_info : Kernel.kernel_info option }
  | Group of { srcs : t list }
  | After of { src : t; deps : t list; dtype : Dtype.t }
  | Unique of { id : int }
  | Lunique of { id : int }
  | Device of { device : device }
  | Buffer of { unique : t; device : t; size : int; dtype : Dtype.t }
  | Buffer_view of { src : t; size : int; offset : int; dtype : Dtype.t }
  | Const of { value : Const.t; dtype : Dtype.t; srcs : t list }
  | Vconst of { values : Const.t list; dtype : Dtype.t; srcs : t list }
  | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
  | Bind of { var : t; value : t option; dtype : Dtype.t }
  | Param of { slot : int; dtype : Dtype.t; shape : t option; device : t option }
  | Call of { callee : callee; args : t list; info : call_info; dtype : Dtype.t }
  | Detach of { src : t; dtype : Dtype.t }
  | Contiguous of { src : t; ranges : t list; opts : Kernel.Opt.t list; dtype : Dtype.t }
  | Contiguous_backward of { src : t; dtype : Dtype.t }
  | Copy of { src : t; device : t; dtype : Dtype.t }
  | Allreduce of { src : t; device : t; op : Op.reduce; dtype : Dtype.t }
  | Multi of { src : t; axis : int; dtype : Dtype.t }
  | Mstack of { srcs : t list; dtype : Dtype.t }
  | Mselect of { src : t; index : int; dtype : Dtype.t }
  | Reduce_axis of { src : t; op : Op.reduce; axes : int list; dtype : Dtype.t }
  | Reduce of { src : t; ranges : t list; op : Op.reduce; dtype : Dtype.t }
  | Reshape of { src : t; shape : t; dtype : Dtype.t }
  | Expand of { src : t; shape : t; dtype : Dtype.t }
  | Pad of { src : t; before : t; after : t; dtype : Dtype.t }
  | Shrink of { src : t; before : t; after : t; dtype : Dtype.t }
  | Permute of { src : t; order : int list; dtype : Dtype.t }
  | Flip of { src : t; dims : bool list; dtype : Dtype.t }
  | Range of { size : t; dtype : Dtype.t; axis : int; sub : int list; kind : Axis_kind.t }
  | End of { value : t; ranges : t list }
  | Index of { ptr : t; idxs : t list; gate : t option; dtype : Dtype.t }
  | Store of { dst : t; value : t }
  | Vectorize of { srcs : t list; dtype : Dtype.t }
  | Cast of { src : t; dtype : Dtype.t }
  | Bitcast of { src : t; dtype : Dtype.t }
  | Unary of { op : Op.unary; src : t; dtype : Dtype.t }
  | Binary of { op : Op.binary; lhs : t; rhs : t; dtype : Dtype.t }
  | Ternary of { op : Op.ternary; a : t; b : t; c : t; dtype : Dtype.t }
  | Noop of { src : t option; dtype : Dtype.t }
  | Bufferize of { src : t; ranges : t list; dtype : Dtype.t; opts : Kernel.bufferize_opts }
  | Invalid_index of { dtype : Dtype.t }
  | Define_local of { size : int; dtype : Dtype.Ptr.t }
  | Barrier
  | Linear of { srcs : t list }
  | Shaped_wmma of {
      a : t; b : t; acc : t;
      dims : int * int * int;
      device : string;
      threads : int;
      dtype : Dtype.t;
    }

and t = view Hashcons.hash_consed

and callee = Ref of t | Ast of Kernel.t

and call_info = {
  grad_fxn : grad_fxn option;
  metadata : metadata list;
  name : string option;
  precompile : bool;
}

and grad_fxn = grad_output:t -> call:t -> t option list

(* Hash-consing *)

let phys_list_eq a b =
  List.length a = List.length b && List.for_all2 (==) a b

let phys_opt_eq a b =
  match a, b with None, None -> true | Some x, Some y -> x == y | _ -> false

let rec shallow_hash_view = function
  | Sink { srcs; kernel_info } ->
      Hashtbl.hash (0, List.map (fun n -> n.Hashcons.tag) srcs, kernel_info)
  | Group { srcs } ->
      Hashtbl.hash (1, List.map (fun n -> n.Hashcons.tag) srcs)
  | After { src; deps; dtype } ->
      Hashtbl.hash (2, src.Hashcons.tag,
        List.map (fun n -> n.Hashcons.tag) deps, dtype)
  | Unique { id } -> Hashtbl.hash (3, id)
  | Lunique { id } -> Hashtbl.hash (4, id)
  | Device { device } -> Hashtbl.hash (5, device)
  | Buffer { unique; device; size; dtype } ->
      Hashtbl.hash (6, unique.Hashcons.tag, device.Hashcons.tag, size, dtype)
  | Buffer_view { src; size; offset; dtype } ->
      Hashtbl.hash (7, src.Hashcons.tag, size, offset, dtype)
  | Const { value; dtype; srcs } ->
      Hashtbl.hash (8, value, dtype, List.map (fun n -> n.Hashcons.tag) srcs)
  | Vconst { values; dtype; srcs } ->
      Hashtbl.hash (9, values, dtype, List.map (fun n -> n.Hashcons.tag) srcs)
  | Define_var { name; lo; hi; dtype } ->
      Hashtbl.hash (10, name, lo, hi, dtype)
  | Bind { var; value; dtype } ->
      Hashtbl.hash (11, var.Hashcons.tag,
        (match value with None -> -1 | Some v -> v.Hashcons.tag), dtype)
  | Param { slot; dtype; shape; device } ->
      Hashtbl.hash (12, slot, dtype,
        (match shape with None -> -1 | Some s -> s.Hashcons.tag),
        (match device with None -> -1 | Some d -> d.Hashcons.tag))
  | Call { callee; args; dtype; _ } ->
      Hashtbl.hash (13,
        (match callee with Ref r -> r.Hashcons.tag | Ast _ -> -1),
        List.map (fun n -> n.Hashcons.tag) args, dtype)
  | Detach { src; dtype } -> Hashtbl.hash (14, src.Hashcons.tag, dtype)
  | Contiguous { src; ranges; opts; dtype } ->
      Hashtbl.hash (15, src.Hashcons.tag,
        List.map (fun n -> n.Hashcons.tag) ranges, opts, dtype)
  | Contiguous_backward { src; dtype } ->
      Hashtbl.hash (16, src.Hashcons.tag, dtype)
  | Copy { src; device; dtype } ->
      Hashtbl.hash (17, src.Hashcons.tag, device.Hashcons.tag, dtype)
  | Allreduce { src; device; op; dtype } ->
      Hashtbl.hash (18, src.Hashcons.tag, device.Hashcons.tag, op, dtype)
  | Multi { src; axis; dtype } ->
      Hashtbl.hash (19, src.Hashcons.tag, axis, dtype)
  | Mstack { srcs; dtype } ->
      Hashtbl.hash (20, List.map (fun n -> n.Hashcons.tag) srcs, dtype)
  | Mselect { src; index; dtype } ->
      Hashtbl.hash (21, src.Hashcons.tag, index, dtype)
  | Reduce_axis { src; op; axes; dtype } ->
      Hashtbl.hash (22, src.Hashcons.tag, op, axes, dtype)
  | Reduce { src; ranges; op; dtype } ->
      Hashtbl.hash (23, src.Hashcons.tag,
        List.map (fun n -> n.Hashcons.tag) ranges, op, dtype)
  | Reshape { src; shape; dtype } ->
      Hashtbl.hash (24, src.Hashcons.tag, shape.Hashcons.tag, dtype)
  | Expand { src; shape; dtype } ->
      Hashtbl.hash (25, src.Hashcons.tag, shape.Hashcons.tag, dtype)
  | Pad { src; before; after; dtype } ->
      Hashtbl.hash (26, src.Hashcons.tag, before.Hashcons.tag,
        after.Hashcons.tag, dtype)
  | Shrink { src; before; after; dtype } ->
      Hashtbl.hash (27, src.Hashcons.tag, before.Hashcons.tag,
        after.Hashcons.tag, dtype)
  | Permute { src; order; dtype } ->
      Hashtbl.hash (28, src.Hashcons.tag, order, dtype)
  | Flip { src; dims; dtype } ->
      Hashtbl.hash (29, src.Hashcons.tag, dims, dtype)
  | Range { size; dtype; axis; sub; kind } ->
      Hashtbl.hash (30, size.Hashcons.tag, dtype, axis, sub, kind)
  | End { value; ranges } ->
      Hashtbl.hash (31, value.Hashcons.tag,
        List.map (fun n -> n.Hashcons.tag) ranges)
  | Index { ptr; idxs; gate; dtype } ->
      Hashtbl.hash (32, ptr.Hashcons.tag,
        List.map (fun n -> n.Hashcons.tag) idxs,
        (match gate with None -> -1 | Some g -> g.Hashcons.tag), dtype)
  | Store { dst; value } ->
      Hashtbl.hash (33, dst.Hashcons.tag, value.Hashcons.tag)
  | Vectorize { srcs; dtype } ->
      Hashtbl.hash (34, List.map (fun n -> n.Hashcons.tag) srcs, dtype)
  | Cast { src; dtype } -> Hashtbl.hash (35, src.Hashcons.tag, dtype)
  | Bitcast { src; dtype } -> Hashtbl.hash (36, src.Hashcons.tag, dtype)
  | Unary { op; src; dtype } ->
      Hashtbl.hash (37, op, src.Hashcons.tag, dtype)
  | Binary { op; lhs; rhs; dtype } ->
      Hashtbl.hash (38, op, lhs.Hashcons.tag, rhs.Hashcons.tag, dtype)
  | Ternary { op; a; b; c; dtype } ->
      Hashtbl.hash (39, op, a.Hashcons.tag, b.Hashcons.tag,
        c.Hashcons.tag, dtype)
  | Noop { src; dtype } ->
      Hashtbl.hash (40, (match src with None -> -1 | Some s -> s.Hashcons.tag),
        dtype)
  | Bufferize { src; ranges; dtype; opts } ->
      Hashtbl.hash (41, src.Hashcons.tag,
        List.map (fun n -> n.Hashcons.tag) ranges, dtype, opts)
  | Invalid_index { dtype } -> Hashtbl.hash (42, dtype)
  | Define_local { size; dtype } -> Hashtbl.hash (43, size, dtype)
  | Barrier -> Hashtbl.hash 44
  | Linear { srcs } ->
      Hashtbl.hash (45, List.map (fun n -> n.Hashcons.tag) srcs)
  | Shaped_wmma { a; b; acc; dims; device; threads; dtype } ->
      Hashtbl.hash (46, a.Hashcons.tag, b.Hashcons.tag, acc.Hashcons.tag,
        dims, device, threads, dtype)

and shallow_equal_view v1 v2 =
  match v1, v2 with
  | Sink s1, Sink s2 ->
      phys_list_eq s1.srcs s2.srcs && s1.kernel_info = s2.kernel_info
  | Group g1, Group g2 -> phys_list_eq g1.srcs g2.srcs
  | After a1, After a2 ->
      a1.src == a2.src && phys_list_eq a1.deps a2.deps && a1.dtype = a2.dtype
  | Unique u1, Unique u2 -> u1.id = u2.id
  | Lunique l1, Lunique l2 -> l1.id = l2.id
  | Device d1, Device d2 -> d1.device = d2.device
  | Buffer b1, Buffer b2 ->
      b1.unique == b2.unique && b1.device == b2.device
      && b1.size = b2.size && b1.dtype = b2.dtype
  | Buffer_view b1, Buffer_view b2 ->
      b1.src == b2.src && b1.size = b2.size
      && b1.offset = b2.offset && b1.dtype = b2.dtype
  | Const c1, Const c2 ->
      c1.value = c2.value && c1.dtype = c2.dtype && phys_list_eq c1.srcs c2.srcs
  | Vconst v1, Vconst v2 ->
      v1.values = v2.values && v1.dtype = v2.dtype && phys_list_eq v1.srcs v2.srcs
  | Define_var d1, Define_var d2 ->
      d1.name = d2.name && d1.lo = d2.lo && d1.hi = d2.hi && d1.dtype = d2.dtype
  | Bind b1, Bind b2 ->
      b1.var == b2.var && phys_opt_eq b1.value b2.value && b1.dtype = b2.dtype
  | Param p1, Param p2 ->
      p1.slot = p2.slot && p1.dtype = p2.dtype
      && phys_opt_eq p1.shape p2.shape && phys_opt_eq p1.device p2.device
  | Call c1, Call c2 ->
      (match c1.callee, c2.callee with
       | Ref r1, Ref r2 -> r1 == r2
       | Ast a1, Ast a2 -> a1 == a2
       | _ -> false)
      && phys_list_eq c1.args c2.args && c1.dtype = c2.dtype
  | Detach d1, Detach d2 -> d1.src == d2.src && d1.dtype = d2.dtype
  | Contiguous c1, Contiguous c2 ->
      c1.src == c2.src && phys_list_eq c1.ranges c2.ranges
      && c1.opts = c2.opts && c1.dtype = c2.dtype
  | Contiguous_backward c1, Contiguous_backward c2 ->
      c1.src == c2.src && c1.dtype = c2.dtype
  | Copy c1, Copy c2 ->
      c1.src == c2.src && c1.device == c2.device && c1.dtype = c2.dtype
  | Allreduce a1, Allreduce a2 ->
      a1.src == a2.src && a1.device == a2.device
      && a1.op = a2.op && a1.dtype = a2.dtype
  | Multi m1, Multi m2 ->
      m1.src == m2.src && m1.axis = m2.axis && m1.dtype = m2.dtype
  | Mstack m1, Mstack m2 ->
      phys_list_eq m1.srcs m2.srcs && m1.dtype = m2.dtype
  | Mselect m1, Mselect m2 ->
      m1.src == m2.src && m1.index = m2.index && m1.dtype = m2.dtype
  | Reduce_axis r1, Reduce_axis r2 ->
      r1.src == r2.src && r1.op = r2.op && r1.axes = r2.axes && r1.dtype = r2.dtype
  | Reduce r1, Reduce r2 ->
      r1.src == r2.src && phys_list_eq r1.ranges r2.ranges
      && r1.op = r2.op && r1.dtype = r2.dtype
  | Reshape r1, Reshape r2 ->
      r1.src == r2.src && r1.shape == r2.shape && r1.dtype = r2.dtype
  | Expand e1, Expand e2 ->
      e1.src == e2.src && e1.shape == e2.shape && e1.dtype = e2.dtype
  | Pad p1, Pad p2 ->
      p1.src == p2.src && p1.before == p2.before
      && p1.after == p2.after && p1.dtype = p2.dtype
  | Shrink s1, Shrink s2 ->
      s1.src == s2.src && s1.before == s2.before
      && s1.after == s2.after && s1.dtype = s2.dtype
  | Permute p1, Permute p2 ->
      p1.src == p2.src && p1.order = p2.order && p1.dtype = p2.dtype
  | Flip f1, Flip f2 ->
      f1.src == f2.src && f1.dims = f2.dims && f1.dtype = f2.dtype
  | Range r1, Range r2 ->
      r1.size == r2.size && r1.dtype = r2.dtype && r1.axis = r2.axis
      && r1.sub = r2.sub && r1.kind = r2.kind
  | End e1, End e2 ->
      e1.value == e2.value && phys_list_eq e1.ranges e2.ranges
  | Index i1, Index i2 ->
      i1.ptr == i2.ptr && phys_list_eq i1.idxs i2.idxs
      && phys_opt_eq i1.gate i2.gate && i1.dtype = i2.dtype
  | Store s1, Store s2 -> s1.dst == s2.dst && s1.value == s2.value
  | Vectorize v1, Vectorize v2 ->
      phys_list_eq v1.srcs v2.srcs && v1.dtype = v2.dtype
  | Cast c1, Cast c2 -> c1.src == c2.src && c1.dtype = c2.dtype
  | Bitcast b1, Bitcast b2 -> b1.src == b2.src && b1.dtype = b2.dtype
  | Unary u1, Unary u2 ->
      u1.op = u2.op && u1.src == u2.src && u1.dtype = u2.dtype
  | Binary b1, Binary b2 ->
      b1.op = b2.op && b1.lhs == b2.lhs && b1.rhs == b2.rhs && b1.dtype = b2.dtype
  | Ternary t1, Ternary t2 ->
      t1.op = t2.op && t1.a == t2.a && t1.b == t2.b
      && t1.c == t2.c && t1.dtype = t2.dtype
  | Noop n1, Noop n2 -> phys_opt_eq n1.src n2.src && n1.dtype = n2.dtype
  | Bufferize b1, Bufferize b2 ->
      b1.src == b2.src && phys_list_eq b1.ranges b2.ranges
      && b1.dtype = b2.dtype && b1.opts = b2.opts
  | Invalid_index i1, Invalid_index i2 -> i1.dtype = i2.dtype
  | Define_local d1, Define_local d2 -> d1.size = d2.size && d1.dtype = d2.dtype
  | Barrier, Barrier -> true
  | Linear l1, Linear l2 -> phys_list_eq l1.srcs l2.srcs
  | Shaped_wmma w1, Shaped_wmma w2 ->
      w1.a == w2.a && w1.b == w2.b && w1.acc == w2.acc
      && w1.dims = w2.dims && w1.device = w2.device
      && w1.threads = w2.threads && Dtype.equal w1.dtype w2.dtype
  | _ -> false

module View_hc = Hashcons.Make (struct
  type nonrec t = view
  let equal = shallow_equal_view
  let hash = shallow_hash_view
end)

let hc_table = View_hc.create 4096
let mk v = View_hc.hashcons hc_table v

(* Accessors *)

let view (n : t) = n.Hashcons.node
let tag (n : t) = n.Hashcons.tag

let node_dtype = function
  | After { dtype; _ } | Buffer { dtype; _ } | Buffer_view { dtype; _ }
  | Const { dtype; _ } | Vconst { dtype; _ } | Define_var { dtype; _ }
  | Bind { dtype; _ } | Param { dtype; _ } | Call { dtype; _ }
  | Detach { dtype; _ } | Contiguous { dtype; _ }
  | Contiguous_backward { dtype; _ } | Copy { dtype; _ }
  | Allreduce { dtype; _ } | Multi { dtype; _ } | Mstack { dtype; _ }
  | Mselect { dtype; _ } | Reduce_axis { dtype; _ } | Reduce { dtype; _ }
  | Reshape { dtype; _ } | Expand { dtype; _ } | Pad { dtype; _ }
  | Shrink { dtype; _ } | Permute { dtype; _ } | Flip { dtype; _ }
  | Range { dtype; _ } | Index { dtype; _ } | Vectorize { dtype; _ }
  | Cast { dtype; _ } | Bitcast { dtype; _ } | Unary { dtype; _ }
  | Binary { dtype; _ } | Ternary { dtype; _ } | Noop { dtype; _ }
  | Bufferize { dtype; _ } | Invalid_index { dtype; _ }
  | Shaped_wmma { dtype; _ } ->
      Some dtype
  | Define_local { dtype; _ } -> Some (Dtype.Val (Dtype.Ptr.base dtype))
  | Sink _ | Group _ | Unique _ | Lunique _ | Device _ | End _ | Store _
  | Barrier | Linear _ -> None

let dtype n = node_dtype (view n)
let node_dtype_of n = node_dtype (view n)

let children_of = function
  | Sink { srcs; _ } | Group { srcs } | Linear { srcs } -> srcs
  | After { src; deps; _ } -> src :: deps
  | Unique _ | Lunique _ | Device _ | Define_var _ | Invalid_index _
  | Barrier | Define_local _ -> []
  | Buffer { unique; device; _ } -> [ unique; device ]
  | Buffer_view { src; _ } -> [ src ]
  | Const { srcs; _ } | Vconst { srcs; _ } -> srcs
  | Bind { var; value; _ } ->
      var :: (match value with Some v -> [v] | None -> [])
  | Param { shape; device; _ } ->
      List.filter_map Fun.id [shape; device]
  | Call { callee; args; _ } ->
      (match callee with Ref r -> r :: args | Ast _ -> args)
  | Detach { src; _ } | Contiguous_backward { src; _ }
  | Multi { src; _ } | Mselect { src; _ }
  | Cast { src; _ } | Bitcast { src; _ }
  | Unary { src; _ } -> [src]
  | Contiguous { src; ranges; _ } | Reduce { src; ranges; _ }
  | Bufferize { src; ranges; _ } -> src :: ranges
  | Copy { src; device; _ } | Allreduce { src; device; _ } -> [src; device]
  | Mstack { srcs; _ } | Vectorize { srcs; _ } -> srcs
  | Reduce_axis { src; _ } -> [src]
  | Reshape { src; shape; _ } | Expand { src; shape; _ } -> [src; shape]
  | Pad { src; before; after; _ } | Shrink { src; before; after; _ } ->
      [src; before; after]
  | Permute { src; _ } | Flip { src; _ } -> [src]
  | Range { size; _ } -> [size]
  | End { value; ranges } -> value :: ranges
  | Index { ptr; idxs; gate; _ } ->
      ptr :: idxs @ (match gate with Some g -> [g] | None -> [])
  | Store { dst; value } -> [dst; value]
  | Binary { lhs; rhs; _ } -> [lhs; rhs]
  | Ternary { a; b; c; _ } -> [a; b; c]
  | Shaped_wmma { a; b; acc; _ } -> [a; b; acc]
  | Noop { src; _ } -> (match src with Some s -> [s] | None -> [])

let children n = children_of (view n)

(* [map_children] applies [f] to every child in the same order as
   [children_of].  All [f] calls use explicit [let]-bindings so that
   evaluation order is left-to-right regardless of the compiler's
   record-field evaluation order.  This matters when [f] carries
   mutable state (e.g. the index counter in [replace]). *)
let map_children (f : t -> t) = function
  | Sink { srcs; kernel_info } -> Sink { srcs = List.map f srcs; kernel_info }
  | Group { srcs } -> Group { srcs = List.map f srcs }
  | After { src; deps; dtype } ->
      let src = f src in let deps = List.map f deps in
      After { src; deps; dtype }
  | (Unique _ | Lunique _ | Device _ | Define_var _ | Invalid_index _
    | Barrier | Define_local _) as v -> v
  | Buffer { unique; device; size; dtype } ->
      let unique = f unique in let device = f device in
      Buffer { unique; device; size; dtype }
  | Buffer_view { src; size; offset; dtype } ->
      Buffer_view { src = f src; size; offset; dtype }
  | Const { value; dtype; srcs } -> Const { value; dtype; srcs = List.map f srcs }
  | Vconst { values; dtype; srcs } -> Vconst { values; dtype; srcs = List.map f srcs }
  | Bind { var; value; dtype } ->
      let var = f var in let value = Option.map f value in
      Bind { var; value; dtype }
  | Param { slot; dtype; shape; device } ->
      let shape = Option.map f shape in let device = Option.map f device in
      Param { slot; dtype; shape; device }
  | Call { callee; args; info; dtype } ->
      let callee = match callee with Ref r -> Ref (f r) | Ast _ -> callee in
      let args = List.map f args in
      Call { callee; args; info; dtype }
  | Detach { src; dtype } -> Detach { src = f src; dtype }
  | Contiguous { src; ranges; opts; dtype } ->
      let src = f src in let ranges = List.map f ranges in
      Contiguous { src; ranges; opts; dtype }
  | Contiguous_backward { src; dtype } ->
      Contiguous_backward { src = f src; dtype }
  | Copy { src; device; dtype } ->
      let src = f src in let device = f device in
      Copy { src; device; dtype }
  | Allreduce { src; device; op; dtype } ->
      let src = f src in let device = f device in
      Allreduce { src; device; op; dtype }
  | Multi { src; axis; dtype } -> Multi { src = f src; axis; dtype }
  | Mstack { srcs; dtype } -> Mstack { srcs = List.map f srcs; dtype }
  | Mselect { src; index; dtype } -> Mselect { src = f src; index; dtype }
  | Reduce_axis { src; op; axes; dtype } ->
      Reduce_axis { src = f src; op; axes; dtype }
  | Reduce { src; ranges; op; dtype } ->
      let src = f src in let ranges = List.map f ranges in
      Reduce { src; ranges; op; dtype }
  | Reshape { src; shape; dtype } ->
      let src = f src in let shape = f shape in
      Reshape { src; shape; dtype }
  | Expand { src; shape; dtype } ->
      let src = f src in let shape = f shape in
      Expand { src; shape; dtype }
  | Pad { src; before; after; dtype } ->
      let src = f src in let before = f before in let after = f after in
      Pad { src; before; after; dtype }
  | Shrink { src; before; after; dtype } ->
      let src = f src in let before = f before in let after = f after in
      Shrink { src; before; after; dtype }
  | Permute { src; order; dtype } -> Permute { src = f src; order; dtype }
  | Flip { src; dims; dtype } -> Flip { src = f src; dims; dtype }
  | Range { size; dtype; axis; sub; kind } ->
      Range { size = f size; dtype; axis; sub; kind }
  | End { value; ranges } ->
      let value = f value in let ranges = List.map f ranges in
      End { value; ranges }
  | Index { ptr; idxs; gate; dtype } ->
      let ptr = f ptr in let idxs = List.map f idxs in
      let gate = Option.map f gate in
      Index { ptr; idxs; gate; dtype }
  | Store { dst; value } ->
      let dst = f dst in let value = f value in
      Store { dst; value }
  | Vectorize { srcs; dtype } -> Vectorize { srcs = List.map f srcs; dtype }
  | Cast { src; dtype } -> Cast { src = f src; dtype }
  | Bitcast { src; dtype } -> Bitcast { src = f src; dtype }
  | Unary { op; src; dtype } -> Unary { op; src = f src; dtype }
  | Binary { op; lhs; rhs; dtype } ->
      let lhs = f lhs in let rhs = f rhs in
      Binary { op; lhs; rhs; dtype }
  | Ternary { op; a; b; c; dtype } ->
      let a = f a in let b = f b in let c = f c in
      Ternary { op; a; b; c; dtype }
  | Noop { src; dtype } -> Noop { src = Option.map f src; dtype }
  | Bufferize { src; ranges; dtype; opts } ->
      let src = f src in let ranges = List.map f ranges in
      Bufferize { src; ranges; dtype; opts }
  | Linear { srcs } -> Linear { srcs = List.map f srcs }
  | Shaped_wmma w ->
      let a = f w.a in let b = f w.b in let acc = f w.acc in
      Shaped_wmma { w with a; b; acc }

(* Helpers used by both validation and analysis *)

let extract_int_shape n =
  match view n with
  | Const { value; _ } ->
      (match Const.view value with
       | Int i -> Some [ Int64.to_int i ]
       | _ -> None)
  | Vconst { values = []; _ } -> Some []
  | Vectorize { srcs; _ } ->
      let ints = List.filter_map (fun s ->
        match view s with
        | Const { value; _ } ->
            (match Const.view value with
             | Int i -> Some (Int64.to_int i)
             | _ -> None)
        | _ -> None) srcs
      in
      if List.length ints = List.length srcs then Some ints else None
  | _ -> None

(* Validation *)

let check cond msg = if not cond then failwith msg

let is_device_node n = match view n with Device _ -> true | _ -> false
let is_unique_node n = match view n with Unique _ | Lunique _ -> true | _ -> false
let is_define_var_node n = match view n with Define_var _ -> true | _ -> false

let is_index_dtype dt = Dtype.equal (Dtype.scalarize dt) Dtype.index

let is_index_vector_node n =
  match view n with
  | Const { dtype; _ } -> is_index_dtype dtype
  | Vectorize { dtype; _ } -> is_index_dtype dtype
  | Vconst { dtype; _ } -> is_index_dtype dtype
  | _ -> false

let is_comparison = function `Cmplt | `Cmpeq | `Cmpne -> true | _ -> false
let is_shift = function `Shl | `Shr -> true | _ -> false

(* Constructors *)

let sink ?kernel_info srcs = mk (Sink { srcs; kernel_info })
let group srcs = match srcs with [x] -> x | _ -> mk (Group { srcs })
let after ~src ~deps =
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  mk (After { src; deps; dtype })
let unique ~id = mk (Unique { id })
let lunique ~id = mk (Lunique { id })
let device d = mk (Device { device = d })

let buffer ~unique ~device ~size ~dtype =
  check (size >= 0) "Buffer size must be non-negative";
  check (is_unique_node unique) "Buffer unique must be Unique/Lunique";
  check (is_device_node device) "Buffer device must be Device";
  mk (Buffer { unique; device; size; dtype })

let buffer_view ~src ~size ~offset ~dtype =
  check (size >= 0) "Buffer_view size must be non-negative";
  check (offset >= 0) "Buffer_view offset must be non-negative";
  check (match view src with Buffer _ | Index _ -> true | _ -> false)
    "Buffer_view src must be Buffer or Index";
  mk (Buffer_view { src; size; offset; dtype })

let const ?(srcs = []) value dtype =
  (match Const.view value with
   | Bool _ -> check (Dtype.is_bool dtype) "Bool const must have bool dtype"
   | Int _ -> check (Dtype.is_int dtype) "Int const must have int/index dtype"
   | Float _ -> check (Dtype.is_float dtype) "Float const must have float dtype");
  mk (Const { value; dtype; srcs })

let vconst ~values ~dtype ?(srcs = []) () =
  check (List.length values = Dtype.count dtype)
    "Vconst values must match vector width";
  let scalar_dt = Dtype.scalarize dtype in
  List.iter (fun v ->
    match Const.view v with
    | Int _ -> check (Dtype.is_int scalar_dt) "Vconst: expected int elements"
    | Float _ -> check (Dtype.is_float scalar_dt) "Vconst: expected float elements"
    | Bool _ -> check (Dtype.is_bool scalar_dt) "Vconst: expected bool elements")
    values;
  mk (Vconst { values; dtype; srcs })

let define_var ~name ~lo ~hi ?(dtype = Dtype.index) () =
  check (Dtype.is_int dtype) "Define_var dtype must be int/index";
  check (lo <= hi) "Define_var lo > hi";
  mk (Define_var { name; lo; hi; dtype })

let bind ~var ?value ~dtype () =
  check (is_define_var_node var) "Bind var must be Define_var";
  (match value with
   | Some v ->
     let vdt = Option.value ~default:Dtype.void (node_dtype_of v) in
     check (Dtype.equal vdt dtype) "Bind value dtype must match"
   | None -> ());
  mk (Bind { var; value; dtype })

let param ~slot ~dtype ?shape ?device () =
  (match shape with
   | Some s -> check (is_index_vector_node s) "Param shape must be index vector"
   | None -> ());
  (match device with
   | Some d -> check (is_device_node d) "Param device must be Device"
   | None -> ());
  mk (Param { slot; dtype; shape; device })

let call ~callee ~args ~info ~dtype =
  (match callee with
   | Ref r ->
     let rdt = Option.value ~default:Dtype.void (node_dtype_of r) in
     check (Dtype.equal rdt dtype) "Call dtype must match Ref dtype"
   | Ast _ -> ());
  mk (Call { callee; args; info; dtype })

let detach ~src =
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  mk (Detach { src; dtype })

let contiguous ~src ?(ranges = []) ?(opts = []) () =
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  List.iter (fun r ->
    let rdt = Option.value ~default:Dtype.void (node_dtype_of r) in
    check (is_index_dtype rdt && Dtype.count rdt = 1)
      "Contiguous range must be index scalar")
    ranges;
  mk (Contiguous { src; ranges; opts; dtype })

let contiguous_backward ~src =
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  mk (Contiguous_backward { src; dtype })

let copy ~src ~device () =
  check (is_device_node device) "Copy device must be Device";
  let dt = Option.value ~default:Dtype.void (dtype src) in
  mk (Copy { src; device; dtype = dt })

let allreduce ~src ~device ~op ~dtype =
  check (is_device_node device) "Allreduce device must be Device";
  mk (Allreduce { src; device; op; dtype })

let multi ~src ~axis =
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  mk (Multi { src; axis; dtype })

let mstack ~srcs =
  check (srcs <> []) "Mstack must have srcs";
  let dtype = match srcs with
    | s :: _ -> Option.value ~default:Dtype.void (dtype s)
    | [] -> Dtype.void in
  List.iter (fun s ->
    let sdt = Option.value ~default:Dtype.void (node_dtype_of s) in
    check (Dtype.equal sdt dtype) "Mstack src dtypes must match")
    srcs;
  mk (Mstack { srcs; dtype })

let mselect ~src ~index =
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  mk (Mselect { src; index; dtype })

let reduce_axis ~src ~op ~axes =
  check (axes <> []) "Reduce_axis must have at least one axis";
  check (List.length (List.sort_uniq Int.compare axes) = List.length axes)
    "Reduce_axis axes must be unique";
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  mk (Reduce_axis { src; op; axes; dtype })

let reduce ~src ~ranges ~op ~dtype = mk (Reduce { src; ranges; op; dtype })

let reshape ~src ~shape =
  (match extract_int_shape shape with
   | Some dims ->
     check (List.for_all (fun d -> d >= 0) dims)
       "Reshape dims must not be negative"
   | None -> ());
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  mk (Reshape { src; shape; dtype })

let expand ~src ~shape =
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  mk (Expand { src; shape; dtype })

let pad_shrink_width n =
  match view n with
  | Vectorize { srcs; _ } -> Some (List.length srcs)
  | Const _ -> Some 1
  | Vconst { values; _ } -> Some (List.length values)
  | _ -> None

let pad ~src ~before ~after =
  (match pad_shrink_width before, pad_shrink_width after with
   | Some bw, Some aw ->
     check (bw = aw) "Pad before/after width mismatch"
   | _ -> ());
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  mk (Pad { src; before; after; dtype })

let shrink ~src ~before ~after =
  (match pad_shrink_width before, pad_shrink_width after with
   | Some bw, Some aw ->
     check (bw = aw) "Shrink before/after width mismatch"
   | _ -> ());
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  mk (Shrink { src; before; after; dtype })

let permute ~src ~order =
  check (List.sort Int.compare order = List.init (List.length order) Fun.id)
    "Permute order must be valid permutation";
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  mk (Permute { src; order; dtype })

let flip ~src ~dims =
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  mk (Flip { src; dims; dtype })
let range ~size ~axis ?(sub = []) ~kind ?(dtype = Dtype.index) () =
  mk (Range { size; dtype; axis; sub; kind })
let end_ ~value ~ranges = mk (End { value; ranges })

let index ~ptr ~idxs ?gate ~dtype () = mk (Index { ptr; idxs; gate; dtype })

let store ~dst ~value = mk (Store { dst; value })

let vectorize ~srcs =
  if srcs = [] then invalid_arg "Vectorize: srcs must not be empty";
  let dtype = match srcs with
    | s :: _ -> (match dtype s with Some d -> Dtype.vec (List.length srcs) (Dtype.scalarize d) | None -> Dtype.void)
    | [] -> Dtype.void in
  mk (Vectorize { srcs; dtype })

let cast ~src ~dtype =
  let src_dt = Option.value ~default:Dtype.void (node_dtype_of src) in
  check (Dtype.count src_dt = Dtype.count dtype)
    "Cast must preserve vector width";
  mk (Cast { src; dtype })

let bitcast ~src ~dtype = mk (Bitcast { src; dtype })

let unary ~op ~src =
  let dtype = Option.value ~default:Dtype.void (dtype src) in
  mk (Unary { op; src; dtype })

let binary ~op ~lhs ~rhs =
  let lhs_dt = Option.value ~default:Dtype.void (dtype lhs) in
  let rhs_dt = Option.value ~default:Dtype.void (dtype rhs) in
  if is_comparison op then begin
    check (Dtype.equal (Dtype.scalarize lhs_dt) (Dtype.scalarize rhs_dt))
      "Comparison operands don't match";
    let c = Dtype.count lhs_dt in
    let dtype = if c > 1 then Dtype.vec c Dtype.bool else Dtype.bool in
    mk (Binary { op; lhs; rhs; dtype })
  end else if is_shift op then begin
    check (Dtype.is_int lhs_dt) "Shift lhs must be int/index";
    check (Dtype.equal (Dtype.scalarize rhs_dt) (Dtype.scalarize lhs_dt)
           || Dtype.equal rhs_dt (Dtype.Val Dtype.Val.uint32))
      "Shift rhs dtype must match lhs or be uint";
    mk (Binary { op; lhs; rhs; dtype = lhs_dt })
  end else begin
    (match op with
     | `Idiv | `Mod ->
       check (Dtype.is_int lhs_dt) "Idiv/Mod must be int/index"
     | _ -> ());
    mk (Binary { op; lhs; rhs; dtype = lhs_dt })
  end

let ternary ~op ~a ~b ~c =
  (match op with
   | `Where ->
     let adt = Option.value ~default:Dtype.void (dtype a) in
     check (Dtype.is_bool adt && Dtype.count adt = 1)
       "Where condition must be bool scalar";
     let bdt = Option.value ~default:Dtype.void (dtype b) in
     let cdt = Option.value ~default:Dtype.void (dtype c) in
     check (Dtype.equal bdt cdt) "Where arms must match"
   | `Mulacc ->
     let adt = Option.value ~default:Dtype.void (dtype a) in
     let bdt = Option.value ~default:Dtype.void (dtype b) in
     let cdt = Option.value ~default:Dtype.void (dtype c) in
     check (Dtype.equal adt bdt && Dtype.equal adt cdt)
       "Mulacc operands must all match");
  let dtype = Option.value ~default:Dtype.void (dtype b) in
  mk (Ternary { op; a; b; c; dtype })

let noop ?src ~dtype () = mk (Noop { src; dtype })
let bufferize ~src ~ranges ~dtype ~opts = mk (Bufferize { src; ranges; dtype; opts })
let invalid_index ~dtype = mk (Invalid_index { dtype })
let define_local ~size ~dtype = mk (Define_local { size; dtype })
let barrier = mk Barrier
let linear srcs = mk (Linear { srcs })
let shaped_wmma ~a ~b ~acc ~dims ~device ~threads ~dtype =
  mk (Shaped_wmma { a; b; acc; dims; device; threads; dtype })

let assign ~target ~value ?(extras = []) () =
  let st = store ~dst:target ~value in
  after ~src:target ~deps:(st :: extras)

(* Replace *)

let replace n ?children:new_ch ?dtype:new_dt () =
  let v = view n in
  let v = match new_ch with
    | None -> v
    | Some ch ->
        let i = ref 0 in
        map_children (fun _ -> let c = List.nth ch !i in incr i; c) v
  in
  let v = match new_dt with
    | None -> v
    | Some dt ->
        (match v with
         | After r -> After { r with dtype = dt }
         | Buffer r -> Buffer { r with dtype = dt }
         | Buffer_view r -> Buffer_view { r with dtype = dt }
         | Const r -> Const { r with dtype = dt }
         | Vconst r -> Vconst { r with dtype = dt }
         | Define_var r -> Define_var { r with dtype = dt }
         | Bind r -> Bind { r with dtype = dt }
         | Param r -> Param { r with dtype = dt }
         | Call r -> Call { r with dtype = dt }
         | Detach r -> Detach { r with dtype = dt }
         | Contiguous r -> Contiguous { r with dtype = dt }
         | Contiguous_backward r -> Contiguous_backward { r with dtype = dt }
         | Copy r -> Copy { r with dtype = dt }
         | Allreduce r -> Allreduce { r with dtype = dt }
         | Multi r -> Multi { r with dtype = dt }
         | Mstack r -> Mstack { r with dtype = dt }
         | Mselect r -> Mselect { r with dtype = dt }
         | Reduce_axis r -> Reduce_axis { r with dtype = dt }
         | Reduce r -> Reduce { r with dtype = dt }
         | Reshape r -> Reshape { r with dtype = dt }
         | Expand r -> Expand { r with dtype = dt }
         | Pad r -> Pad { r with dtype = dt }
         | Shrink r -> Shrink { r with dtype = dt }
         | Permute r -> Permute { r with dtype = dt }
         | Flip r -> Flip { r with dtype = dt }
         | Range r -> Range { r with dtype = dt }
         | Index r -> Index { r with dtype = dt }
         | Vectorize r -> Vectorize { r with dtype = dt }
         | Cast r -> Cast { r with dtype = dt }
         | Bitcast r -> Bitcast { r with dtype = dt }
         | Unary r -> Unary { r with dtype = dt }
         | Binary r -> Binary { r with dtype = dt }
         | Ternary r -> Ternary { r with dtype = dt }
         | Noop r -> Noop { r with dtype = dt }
         | Bufferize r -> Bufferize { r with dtype = dt }
         | Invalid_index _ -> Invalid_index { dtype = dt }
         | Shaped_wmma r -> Shaped_wmma { r with dtype = dt }
         | v -> v)
  in
  let result = mk v in
  if result == n then n else result

(* Traversal *)

let toposort ?(gate = fun _ -> true) ?(enter_calls = true) root =
  let visited : (int, unit) Hashtbl.t = Hashtbl.create 256 in
  let result = ref [] in
  let stack : (t * bool) Stack.t = Stack.create () in
  Stack.push (root, false) stack;
  while not (Stack.is_empty stack) do
    let node, processed = Stack.pop stack in
    if Hashtbl.mem visited node.Hashcons.tag then ()
    else if not processed then begin
      if gate node then begin
        Stack.push (node, true) stack;
        let srcs = match view node with
          | Call { callee = Ref c; args; _ } when not enter_calls ->
              args @ [c]  (* skip callee body but include the ref *)
          | Call { args; _ } when not enter_calls -> args
          | _ -> children node
        in
        List.iter (fun s ->
          if not (Hashtbl.mem visited s.Hashcons.tag) then
            Stack.push (s, false) stack)
          (List.rev srcs)
      end
    end else begin
      Hashtbl.replace visited node.Hashcons.tag ();
      result := node :: !result
    end
  done;
  List.rev !result

let backward_slice root =
  let nodes = toposort root in
  List.filter (fun n -> n != root) nodes

let variables root =
  List.filter (fun n ->
    match view n with Define_var _ -> true | _ -> false)
    (toposort root)

let ranges root =
  List.filter (fun n ->
    match view n with Range _ -> true | _ -> false)
    (toposort root)

(* Rewriting *)

module Ref_tbl = Hashtbl.Make (struct
  type nonrec t = t
  let equal a b = a == b
  let hash (n : t) = n.Hashcons.tag
end)

let first_match rules n =
  List.find_map (fun rule -> rule n) rules

(* 3-stage stack-based graph rewrite. When a rewrite produces a new
   node, that node is fully processed (children visited, rewrite
   applied). Waitlists handle nodes whose dependencies aren't yet
   resolved. *)
let graph_rewrite ?(name = "") ?(enter_calls = true)
    ?(on_rebuild : old_n:t -> new_n:t -> unit = fun ~old_n:_ ~new_n:_ -> ())
    rewrite root =
  let replace : t Ref_tbl.t = Ref_tbl.create 256 in
  let on_stack : unit Ref_tbl.t = Ref_tbl.create 256 in
  let waitlist : (t * int * t) list Ref_tbl.t = Ref_tbl.create 16 in
  let stack : (t * int * t) Stack.t = Stack.create () in
  let lookup c =
    match Ref_tbl.find_opt replace c with Some r -> r | None -> c
  in
  let set_replace n v =
    Ref_tbl.replace replace n v;
    match Ref_tbl.find_opt waitlist n with
    | Some waiting ->
        Ref_tbl.remove waitlist n;
        List.iter (fun entry -> Stack.push entry stack) waiting
    | None -> ()
  in
  Stack.push (root, 0, root) stack;
  Ref_tbl.replace on_stack root ();
  let counter = ref 0 in
  while not (Stack.is_empty stack) do
    let n, stage, new_n = Stack.pop stack in
    if Ref_tbl.mem replace n then ()
    else begin
      incr counter;
      if !counter > 250000 then
        failwith (Printf.sprintf "graph_rewrite(%s): %d nodes" name !counter);
      if stage = 0 then begin
        (* Stage 0: push self at stage 1, then push children *)
        Stack.push (n, 1, new_n) stack;
        let srcs =
          if not enter_calls then
            match view new_n with
            | Call { args; _ } -> args
            | _ -> children new_n
          else children new_n
        in
        List.iter (fun x ->
          if not (Ref_tbl.mem on_stack x) then begin
            Stack.push (x, 0, x) stack;
            Ref_tbl.replace on_stack x ()
          end) (List.rev srcs)
      end
      else if stage = 1 then begin
        (* Stage 1: check all children are ready *)
        let all_ready = ref true in
        let new_src =
          List.map (fun x ->
            match Ref_tbl.find_opt replace x with
            | Some r -> r
            | None -> all_ready := false; x)
            (children new_n)
        in
        if not !all_ready then begin
          let missing =
            List.find (fun x -> not (Ref_tbl.mem replace x))
              (children new_n)
          in
          let prev = match Ref_tbl.find_opt waitlist missing with
            | Some l -> l | None -> [] in
          Ref_tbl.replace waitlist missing ((n, 1, new_n) :: prev)
        end
        else begin
          let old_src = children new_n in
          let changed =
            List.length old_src = List.length new_src
            && not (List.for_all2 (==) old_src new_src)
          in
          if not changed then begin
            match rewrite new_n with
            | None -> set_replace n new_n
            | Some rewritten when rewritten == new_n -> set_replace n new_n
            | Some rewritten ->
                Stack.push (n, 2, rewritten) stack;
                Stack.push (rewritten, 0, rewritten) stack
          end
          else begin
            let rebuilt = mk (map_children lookup (view new_n)) in
            on_rebuild ~old_n:new_n ~new_n:rebuilt;
            Stack.push (n, 2, rebuilt) stack;
            Stack.push (rebuilt, 0, rebuilt) stack
          end
        end
      end
      else begin
        (* Stage 2: link n → result of new_n *)
        match Ref_tbl.find_opt replace new_n with
        | Some result -> set_replace n result
        | None ->
            let prev = match Ref_tbl.find_opt waitlist new_n with
              | Some l -> l | None -> [] in
            Ref_tbl.replace waitlist new_n ((n, 2, new_n) :: prev)
      end
    end
  done;
  lookup root

let substitute mappings root =
  let tbl : t Ref_tbl.t = Ref_tbl.create (List.length mappings) in
  List.iter (fun (old_n, new_n) -> Ref_tbl.replace tbl old_n new_n) mappings;
  graph_rewrite (fun n ->
    Ref_tbl.find_opt tbl n)
    root

(* Analysis *)

let rec base n =
  match view n with
  | Reshape { src; _ } | Expand { src; _ } | Pad { src; _ }
  | Shrink { src; _ } | Permute { src; _ } | Flip { src; _ }
  | Multi { src; _ } | Detach { src; _ } -> base src
  | _ -> n

let extract_marg v =
  match v with
  | Reshape { shape; _ } | Expand { shape; _ } -> extract_int_shape shape
  | _ -> None

let extract_marg_pairs v =
  match v with
  | Pad { before; after; _ } | Shrink { before; after; _ } ->
      (match extract_int_shape before, extract_int_shape after with
       | Some bs, Some als when List.length bs = List.length als ->
           Some (List.combine bs als)
       | _ -> None)
  | _ -> None

let compute_shapes root =
  let tbl : (int, int list option) Hashtbl.t = Hashtbl.create 256 in
  let nodes = toposort root in
  let sh n = match Hashtbl.find_opt tbl n.Hashcons.tag with
    | Some s -> s | None -> None in
  List.iter (fun n ->
    let shape = match view n with
      | Sink _ | Group _ | Unique _ | Lunique _ | Device _ | Range _
      | Store _ | End _ | Barrier | Define_local _ | Linear _ -> None
      | Const _ | Vconst _ | Define_var _ | Bind _ | Invalid_index _ ->
          Some []
      | Buffer { size; _ } | Buffer_view { size; _ } -> Some [ size ]
      | Param { shape; _ } ->
          Option.bind shape extract_int_shape
      | Reshape { shape; _ } | Expand { shape; _ } ->
          extract_int_shape shape
      | Pad { src; before; after; _ } ->
          (match sh src, extract_int_shape before, extract_int_shape after with
           | Some s, Some b, Some a ->
               Some (List.map2 (fun si (bi, ai) -> si + bi + ai)
                 s (List.combine b a))
           | _ -> None)
      | Shrink { src; before; after; _ } ->
          (match sh src, extract_int_shape before, extract_int_shape after with
           | Some s, Some b, Some a ->
               Some (List.map2 (fun si (bi, ai) -> si - bi - ai)
                 s (List.combine b a))
           | _ -> None)
      | Permute { src; order; _ } ->
          Option.map (fun s ->
            List.map (fun i -> List.nth s i) order) (sh src)
      | Flip { src; _ } -> sh src
      | Vectorize { srcs; _ } ->
          (match srcs with
           | s :: _ ->
               Option.map (fun dims -> List.length srcs :: dims) (sh s)
           | [] -> Some [0])
      | Reduce_axis { src; axes; _ } ->
          Option.map (fun s ->
            List.mapi (fun i d -> if List.mem i axes then 1 else d) s) (sh src)
      | Multi { src; _ } | Mselect { src; _ }
      | Detach { src; _ } | Contiguous { src; _ }
      | Contiguous_backward { src; _ } | Copy { src; _ }
      | Cast { src; _ } | Bitcast { src; _ }
      | Unary { src; _ } | Noop { src = Some src; _ } -> sh src
      | Mstack { srcs; _ } ->
          (match srcs with s :: _ -> sh s | [] -> None)
      | Binary { lhs; _ } -> sh lhs
      | Ternary { b; _ } -> sh b
      | Call { callee = Ast _; args; _ } ->
          (match args with a :: _ -> sh a | [] -> None)
      | _ -> None
    in
    Hashtbl.replace tbl n.Hashcons.tag shape)
    nodes;
  fun n -> match Hashtbl.find_opt tbl n.Hashcons.tag with
    | Some s -> s | None -> None

let compute_devices root =
  let tbl : (int, device option) Hashtbl.t = Hashtbl.create 256 in
  let nodes = toposort root in
  let dev n = match Hashtbl.find_opt tbl n.Hashcons.tag with
    | Some d -> d | None -> None in
  List.iter (fun n ->
    let d = match view n with
      | Device { device = d } -> Some d
      | Buffer { device = d; _ } -> dev d
      | Copy { device = d; _ } -> dev d
      | After { src; _ } | Detach { src; _ }
      | Contiguous { src; _ } | Contiguous_backward { src; _ }
      | Cast { src; _ } | Bitcast { src; _ }
      | Unary { src; _ } | Reshape { src; _ }
      | Expand { src; _ } | Pad { src; _ }
      | Shrink { src; _ } | Permute { src; _ }
      | Flip { src; _ } | Reduce_axis { src; _ }
      | Multi { src; _ } | Mselect { src; _ }
      | Noop { src = Some src; _ } -> dev src
      | Binary { lhs; _ } -> dev lhs
      | Ternary { b; _ } -> dev b
      | Mstack { srcs; _ } ->
          (match srcs with s :: _ -> dev s | [] -> None)
      | Param { device = d; _ } ->
          Option.bind d dev
      | Call { callee = Ast _; args; _ } ->
          (match args with a :: _ -> dev a | [] -> None)
      | Allreduce { device = d; _ } -> dev d
      | _ -> None
    in
    Hashtbl.replace tbl n.Hashcons.tag d)
    nodes;
  fun n -> match Hashtbl.find_opt tbl n.Hashcons.tag with
    | Some d -> d | None -> None

let consumer_map root =
  let tbl : (int, t list) Hashtbl.t = Hashtbl.create 256 in
  let nodes = toposort root in
  List.iter (fun n ->
    List.iter (fun c ->
      let prev = match Hashtbl.find_opt tbl c.Hashcons.tag with
        | Some l -> l | None -> [] in
      Hashtbl.replace tbl c.Hashcons.tag (n :: prev))
      (children n))
    nodes;
  fun n -> match Hashtbl.find_opt tbl n.Hashcons.tag with
    | Some l -> l | None -> []

(* Formatting *)

let pp_view fmt v =
  let pp_node fmt (n : t) = Format.fprintf fmt "%%%d" n.Hashcons.tag in
  let pp_nodes fmt ns =
    Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ", ")
      pp_node fmt ns
  in
  match v with
  | Sink { srcs; _ } -> Format.fprintf fmt "sink [%a]" pp_nodes srcs
  | Group { srcs } -> Format.fprintf fmt "group [%a]" pp_nodes srcs
  | After { src; deps; _ } ->
      Format.fprintf fmt "after %a, deps=[%a]" pp_node src pp_nodes deps
  | Unique { id } -> Format.fprintf fmt "unique %d" id
  | Lunique { id } -> Format.fprintf fmt "lunique %d" id
  | Device { device = Single d } -> Format.fprintf fmt "device %s" d
  | Device { device = Multi ds } ->
      Format.fprintf fmt "device [%s]" (String.concat ", " ds)
  | Buffer { unique; device; size; dtype } ->
      Format.fprintf fmt "buffer unique=%a device=%a size=%d : %a"
        pp_node unique pp_node device size Dtype.pp dtype
  | Buffer_view { src; size; offset; dtype } ->
      Format.fprintf fmt "buffer_view %a size=%d offset=%d : %a"
        pp_node src size offset Dtype.pp dtype
  | Const { dtype; _ } -> Format.fprintf fmt "const : %a" Dtype.pp dtype
  | Vconst { dtype; _ } -> Format.fprintf fmt "vconst : %a" Dtype.pp dtype
  | Define_var { name; lo; hi; dtype } ->
      Format.fprintf fmt "define_var %s [%d, %d] : %a" name lo hi Dtype.pp dtype
  | Bind { var; value; _ } ->
      Format.fprintf fmt "bind %a = %a" pp_node var
        (Format.pp_print_option pp_node) value
  | Param { slot; dtype; _ } ->
      Format.fprintf fmt "param %d : %a" slot Dtype.pp dtype
  | Call { args; dtype; _ } ->
      Format.fprintf fmt "call [%a] : %a" pp_nodes args Dtype.pp dtype
  | Linear { srcs } -> Format.fprintf fmt "linear [%a]" pp_nodes srcs
  | Shaped_wmma { a; b; acc; dims = (m, n, k); device; threads; dtype } ->
      Format.fprintf fmt "shaped_wmma %a, %a, %a dims=(%d,%d,%d) dev=%s thr=%d : %a"
        pp_node a pp_node b pp_node acc m n k device threads Dtype.pp dtype
  | Store { dst; value } ->
      Format.fprintf fmt "store %a <- %a" pp_node dst pp_node value
  | End { value; ranges } ->
      Format.fprintf fmt "end %a ranges=[%a]" pp_node value pp_nodes ranges
  | Barrier -> Format.fprintf fmt "barrier"
  | _ -> Format.fprintf fmt "<%s>" (match v with
      | Detach _ -> "detach" | Contiguous _ -> "contiguous"
      | Contiguous_backward _ -> "contiguous_backward"
      | Copy _ -> "copy" | Allreduce _ -> "allreduce"
      | Multi _ -> "multi" | Mstack _ -> "mstack" | Mselect _ -> "mselect"
      | Reduce_axis _ -> "reduce_axis" | Reduce _ -> "reduce"
      | Reshape _ -> "reshape" | Expand _ -> "expand"
      | Pad _ -> "pad" | Shrink _ -> "shrink"
      | Permute _ -> "permute" | Flip _ -> "flip"
      | Range _ -> "range" | Index _ -> "index"
      | Vectorize _ -> "vectorize" | Cast _ -> "cast" | Bitcast _ -> "bitcast"
      | Unary _ -> "unary" | Binary _ -> "binary" | Ternary _ -> "ternary"
      | Noop _ -> "noop" | Bufferize _ -> "bufferize"
      | Invalid_index _ -> "invalid_index" | Define_local _ -> "define_local"
      | _ -> "unknown")

let pp fmt root =
  let nodes = toposort root in
  List.iter (fun n ->
    Format.fprintf fmt "%3d: %a@\n" n.Hashcons.tag pp_view (view n))
    nodes
