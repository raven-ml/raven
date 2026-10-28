(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operations as data, and values' facts: the functions over {!Value.prim} and
    {!Value.t} that the engine and [Nx.Prim] share.

    Every function is total over the constructors. {!val-route} states each
    operation's rule once: the shapes, dtypes, axes and attributes its
    operands must have, where it reads them, and each result's layout and
    placement, from {!Route}; {!results_at} makes the results' forms from what
    it decided. The eager engine routes, places and allocates destinations
    through them, and interpretations make traced results through {!results},
    so the two agree in dtype, layout and placement. A sharded value's shape
    is the whole's, which it holds; each shard is its device's window of it.
    This module lies above {!Value} and {!Route}. *)

open Value

(** {1:facts Values' facts}

    [placement] allocates nothing, and so do [dtype], [rank] and [dim] of a
    value on one device. *)

val form : ('v, 's, 'd) t -> ('v, 's, 'd) form
val dtype : ('v, 's, 'd) t -> ('v, 's) dtype

val placement : ('v, 's, 'd) t -> 'd Devices.placement
(** [placement x] is where [x]'s bytes are, or will be for a traced value: its
    form's. *)

val of_arrays :
  'd Devices.placement ->
  shape:int array ->
  ('v, 's) Nx_array.t iarray ->
  ('v, 's, 'd) t
(** [of_arrays p ~shape arrays] is the live value of [shape] over [arrays], one
    per device of [p], each its window of [shape]: [Array] for one, [Shards]
    otherwise. Nothing checks the windows. *)

val death : ('v, 's, 'd) t -> death
(** [death x] is whether [x] lives: for a handle, its donor's death. It
    allocates nothing. *)

val alive : by:string -> ('v, 's, 'd) t -> unit
(** [alive ~by x] raises [Invalid_argument] naming [by], [x] as {!pp_value}
    prints it and the function [x] was donated to, if [x] died, as
    [Nx.add: float32 [2; 2] was donated to Nx.mul]. *)

val arrays : ('v, 's, 'd) t -> ('v, 's) Nx_array.t iarray
(** [arrays x] is [x]'s arrays, one per device of its placement, whether [x]
    lives or not. Raises [Invalid_argument] for a traced value. *)

val any : ('v, 's) Nx_array.t iarray -> Nx_array.any array
(** [any xs] is [xs] untyped, in order. *)

val rank : ('v, 's, 'd) t -> int

val dim : ('v, 's, 'd) t -> int -> int
(** Raises [Invalid_argument] if the axis is not below {!rank}. *)

val same_shape : ('v, 's, 'd) t -> ('w, 'r, 'd) t -> bool
(** [same_shape x y] is [shape x = shape y]. It allocates nothing. *)

val shape : ('v, 's, 'd) t -> int array
(** [shape x] is a fresh array. *)

val has_shape : ('v, 's, 'd) t -> int array -> bool
(** [has_shape x s] is [shape x = s]. It allocates nothing. *)

val axis : by:string -> int -> int -> (Format.formatter -> unit) -> int
(** [axis ~by r a what] is [a] as an axis of a value of rank [r], counted from
    the end where [a] is negative. Raises [Invalid_argument] naming [by], [a]
    and the value as [what] prints it, if [a] is no axis of it. *)

val merge : int array -> int array -> (int array, int * int * int) result
(** [merge s s'] is the shape [s] and [s'] broadcast to, aligned at their last
    axes, each extent equal or [1]; [Error (a, e, e')] at the first axis [a] of
    that shape where [s] has [e] and [s'] has [e'], neither [1] nor the other.
*)

val pp_shape : Format.formatter -> int array -> unit
(** [pp_shape ppf s] prints [s] as nx's messages do: [[2; 3]]. *)

val pp_value : Format.formatter -> ('v, 's, 'd) t -> unit
(** [pp_value ppf x] prints [x]'s dtype and shape as nx's messages do:
    [float32 [2; 3]]. *)

val check_shape : by:string -> int iarray -> unit
(** [check_shape ~by s] raises [Invalid_argument] naming [by] and [s] unless
    [s] is a layout's shape ({!Nx_array.Layout.fits}): where an extent is
    negative, [s] has more than {!Nx_array.Layout.max_rank} axes, or its
    extents other than [0] multiply past {!Nx_array.Layout.max_numel}. *)

val check_value : by:string -> ('v, 's) dtype -> 'v -> unit
(** [check_value ~by dt v] raises [Invalid_argument] naming [by], [v] and
    [dt]'s range unless [dt] holds [v] ({!Nx_array.Dtype.holds}). *)

val broadcast_shape : by:string -> int array -> int array -> int array
(** [broadcast_shape ~by s s'] is [merge s s']'s shape. Raises
    [Invalid_argument] naming [by] and both shapes where they do not broadcast.
*)

val expect : ('v, 's) dtype -> 'd any -> ('v, 's, 'd) t
(** [expect dt (Any x)] is [x] at [dt]'s type. Raises [Invalid_argument] naming
    both dtypes if [x]'s dtype is another. *)

(** {1:operations Operations} *)

type operands =
  | Operands : 'd any list -> operands
      (** An operation's operands, all of one brand. *)

val name : 'r prim -> string
(** [name op] is [op]'s constructor, as ["Map"]. *)

val pp : Format.formatter -> 'r prim -> unit
(** [pp] formats an operation: its name, its programs as expressions over its
    operands [x0], [x1], …, and each operand's dtype, shape and placement. A
    program node read more than once prints once, as [nK = …] before the
    outputs, and as [nK] where it is read. *)

val kind : Nx_kernel.Prog.node -> string
(** [kind n] names [n] in messages by its innermost constructor, as ["Exp"]
    for [Op1 (Unary Exp, _, _)], ["Where"] or ["In"]. A [Coord] is named
    ["Iota"] and a [Const] ["Fill"]. *)

val operands : 'r prim -> operands
(** [operands op] is [op]'s operands in order: a map's or a reduction's loads;
    a scan's loads then its carry; a gather's [idx] then [x]; a scatter's
    [idx], [updates] then [into]; a contraction's [a], [b], then each it has of
    [init], [a_scale], [b_scale] and [segments]' offsets; an assembly's pieces;
    a triangular solve's [a] then [b]; [Check]'s [ok] then its data; the one
    operand of the others. Every position below is one in this order. *)

val own_index :
  'r prim -> (int list * Nx_array.Layout.t * Nx_array.Dtype.any) option
(** [own_index op] is the positions in {!val-operands} of the operands [op]
    reads only at its first result's own index, each of which a kernel may
    take as that result's destination ({!Nx_kernel.S}), with that result's
    layout and dtype: a map's plain loads, a scatter's [into], a
    contraction's [init], an assembly's first piece where it covers the whole
    result. [None] for an operation with none. *)

val iteri : ('v 's 'd. int -> ('v, 's, 'd) t -> unit) -> 'r prim -> unit
(** [iteri f op] is [f i x] for each operand [x] of [op], at its position [i] in
    {!val-operands}. *)

val exists : ('v 's 'd. ('v, 's, 'd) t -> bool) -> 'r prim -> bool
(** [exists f op] is whether [f x] for some operand [x] of [op]. Neither it
    nor {!iteri} allocates for an operation other than a factorisation, a
    contraction or a check. *)

val map : ('v 's 'd. ('v, 's, 'd) t -> ('v, 's, 'd) t) -> 'r prim -> 'r prim
(** [map m op] is [op] with each operand [x] replaced by [m x]. *)

val mapi :
  ('v 's 'd. int -> ('v, 's, 'd) t -> ('v, 's, 'd) t) -> 'r prim -> 'r prim
(** [mapi m op] is [op] with each operand [x] at position [i] in {!val-operands}
    replaced by [m i x]: [op] itself where [m] returns each operand
    itself. *)

val rule : 'r prim -> Route.rule
(** [rule op] is how [op]'s results depend on its operands: its route and each
    device's windows of its operands ({!Route.window}) come from it alone. A
    loop of no load has the rule of one load of its shape: each device of its
    placement computes its window of the loop. A contraction's scale is read
    at its operand's windows where a cut keeps its blocks whole, and whole
    elsewhere. [Rows]' axis of [a] and its offsets are read whole, and [b]'s
    axis of its groups is reduced, a factor after the pairs that no axis of
    [a] shares, whose fences the offsets hold ({!Partial}). A gather's axis
    of [x] is reduced, each device gathering from its window of it, where
    [x] has fewer than {!Nx_array.Layout.max_rank} axes, a [complex128] one
    fewer, since its partials stack along a new axis; it is read whole
    otherwise. Its positions are read at the results' windows. A segmented
    contraction is read whole along its contracting pairs. An assembly's
    axis becomes the result's where every piece's region covers all of it
    in order, and is read whole elsewhere. *)

val local : 'r prim -> Nx_array.Move.range array -> 'r prim
(** [local op w] is [op] computing the window [w] of its first result, over
    operands the caller gives at their windows ({!Route.window}): a loop over
    its window of its shape, its programs' coordinates offset by that window's
    start, a fold of [w]'s shape, a movement by {!local_move}, and an
    assembly into [w]'s shape, each region covering all of [w]'s range in
    order along an axis [w] cuts.

    Raises [Invalid_argument] for a window of an assembly that cuts an axis a
    piece's region does not cover all of in order: its rule reads that axis
    whole. *)

val local_move :
  Nx_array.Move.t -> Nx_array.Move.range array -> Nx_array.Move.t
(** [local_move mv w] is the movement that gives the window [w] of [mv]'s
    result from the operand's window [w] reads: a reshape or a broadcast to
    [w]'s extents; a slice that takes all of the operand's window along each
    axis whose range starts at [0] by step [1], and keeps any other range; a
    permutation or a sliding window unchanged. [w] must be a window a route
    gives a device ({!Route.route}), which holds whole rows of both extents
    along an axis [mv] regroups ({!Route.Regroups}).

    [Exec] moves each device's array by it, since {!local}'s type does not
    say that it gives a movement back for a movement. *)

val within : 'r prim -> Nx_array.Move.range array -> 'r prim
(** [within op l] is the loop [op], a map, a reduction or a scan, over the
    window [l] of its loop's shape, over loads the caller gives at their
    windows: a loop of [l]'s extents, its programs' coordinates offset by [l]'s
    start. A reduction folds the terms [l] holds alone. {!local} of a loop is
    [within] the loop's window that its result's window reads.

    Raises [Invalid_argument] for an operation that is not a loop. *)

val layouts : by:string -> 'r prim -> Nx_array.Layout.t array
(** [layouts ~by op] is the layouts of [op]'s results that eager execution
    gives them, after checking its operands against its rule: one per result
    of a [Linalg], one that every result of any other operation shares, none
    for a [Check]. A [Place]'s is its result's at its own placement.

    Raises [Invalid_argument] naming [by] where [op]'s operands break its
    rule, a map of no loads with no placement or one with loads that names
    one included, and where a map of no loads names a placement that cuts an
    axis its shape lacks ({!Devices.window}). *)

(** The type for what an operation's rule decides. *)
type routed = {
  route : unit Route.t option;
      (** Where it reads each operand, in {!val-operands}' order, where its
          results lie and how its devices compute them: [None] for [Place]
          and [Check], which are not routed. *)
  layouts : Nx_array.Layout.t array;  (** Its results' ({!val-layouts}). *)
}

val route : by:string -> 'r prim -> routed
(** [route ~by op] is what [op]'s rule decides, checked once. Operands that
    lie at one placement that cuts no axis are read there and their results
    lie there; otherwise {!Route.route} of {!rule} decides, given a
    gather's or a scatter's positions, operand 0, as its [positions]. An
    operation of
    no operands lies where its map names, else on the host, each device of
    that placement computing its window of the results.

    Raises as {!val-layouts} does, then as {!Route.route} does. *)

val results_at :
  unit Devices.placement ->
  Nx_array.Layout.t array ->
  ('v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) t) ->
  'r prim ->
  'r
(** [results_at p layouts m op] is [op]'s result, its value at position [k]
    made by [m k f], [f] the form eager execution gives it at [p], laid out as
    [layouts], [op]'s {!val-layouts}; [()] for [Check]. A [Place]'s result lies
    at its own placement. It checks nothing. *)

val results :
  by:string ->
  ('v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) t) ->
  'r prim ->
  'r
(** [results ~by m op] is {!results_at} of what {!val-route} decides for [op]:
    its results' layouts, at the placement of its results.

    Raises [Invalid_argument] naming [by], before [m] is called, where [op]'s
    operands break its rule. *)

val reduction :
  ('d, 'a) reduction -> Nx_kernel.reduction * int * Nx_array.Dtype.any
(** [reduction r] is what [r] computes, the program output it reduces and the
    dtype of its first result, as [Nx.Prim.reduction] states. A kernel takes
    the first two ({!Nx_kernel.S.reduce}). *)

val reductions :
  ('d, 'r) reductions -> (Nx_kernel.reduction * int * Nx_array.Dtype.any) array
(** [reductions rs] is {!val-reduction} of each of [rs], in order. *)

val reduction_name : Nx_kernel.reduction -> string
(** [reduction_name r] names [r] in messages, as ["Logsumexp"] or
    ["Arg Max"]. *)

val reduced : int array -> int iarray -> int array
(** [reduced s axes] is [s] without [axes]. *)

val node1 :
  Nx_kernel.Prog.op1 ->
  ('a, 'b) dtype ->
  ('v, 's) dtype ->
  Nx_kernel.Prog.t option

val node2 : Nx_kernel.Prog.op2 -> ('a, 'b) dtype -> Nx_kernel.Prog.t option

val node3 :
  Nx_kernel.Prog.op3 ->
  ('c, 'e) dtype ->
  ('a, 'b) dtype ->
  Nx_kernel.Prog.t option
(** [nodeN k x …] is [Some] the program {!Nx_kernel.Prog.of_node} makes of
    the one node [k] reading its operands in order, over an operand of [x]
    into [y] for [node1 k x y], two of [x] for [node2 k x], and one of [c]
    then two of [x] for [node3 k c x]. It is [None] where [k] does not take
    them ({!Nx_kernel.Prog.accepts1} and its siblings). A program is kept
    once made: a call that finds it allocates nothing. Domains may call it
    concurrently; two that find its slot empty at once each make an equal
    program. *)

val identity : ('v, 's) dtype -> Nx_kernel.Prog.t
(** [identity dt] is the program of one operand of [dt] that gives it back
    ({!Nx_kernel.Prog.is_identity}). It is kept once made, as {!node1}'s
    programs are. *)

val op1 :
  by:string ->
  Nx_kernel.Prog.op1 ->
  ('w, 'r) dtype ->
  ('v, 's, 'd) t ->
  (('w, 'r, 'd) t * unit) prim

val op2 :
  by:string ->
  Nx_kernel.Prog.op2 ->
  ('w, 'r) dtype ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  (('w, 'r, 'd) t * unit) prim

val op3 :
  by:string ->
  Nx_kernel.Prog.op3 ->
  ('a, 'b, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  (('v, 's, 'd) t * unit) prim
(** [opN ~by k dt x …] is the map of the one node [k] over [x …], of [x]'s
    shape, with result dtype [dt] ([op3]'s: its second operand's).

    Raises [Invalid_argument] naming [by] where [k] does not take the operands'
    dtypes ({!Nx_kernel.Prog.accepts2} and its siblings). *)

val literal2 :
  by:string ->
  swapped:bool ->
  Nx_kernel.Prog.op2 ->
  ('v, 's) dtype ->
  'v ->
  Nx_kernel.Prog.t
(** [literal2 ~by ~swapped k dt c] is the program of [k] over one operand of
    [dt] and the literal [c] of [dt], held in the program: [k x c], or [k c x]
    where [swapped]. A call that repeats the literal's bits of the last call
    of its [k], [dt] and side gives that call's program; any other makes
    one.

    Raises [Invalid_argument] naming [by] and [c] where [dt] does not hold [c]
    ({!check_value}), and naming [by] where [k] does not take [dt]. *)

val contract_shape :
  by:string ->
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  segments:'d index Nx_kernel.segments option ->
  acc:Nx_array.Dtype.any ->
  ('a, 'b, 'd) t ->
  ('c, 'e, 'd) t ->
  ('v, 's, 'd) t option ->
  'd scale option ->
  'd scale option ->
  int array
(** [contract_shape ~by ~batch ~contracting ~segments ~acc a b init sa sb] is
    the shape of the contraction's result, [sa] and [sb] [a]'s and [b]'s
    scales, as {!Nx_kernel.Shape.contract} gives it: the contraction's rule,
    which {!results} applies to [Contract].

    Raises [Invalid_argument] naming [by] where the pairs, the scales or the
    segments do not fit [a] and [b], [init] has another shape, or [acc] is a
    boolean or a float narrower than float32. *)

val scale_shape : 'd scale -> int array
(** [scale_shape s] is the shape of [s]'s array. *)

(** {1:loads Loads} *)

val load_any : 'd load -> 'd any
(** [load_any l] is the operand [l] reads. *)

(** {1:families Transforms and factorisations} *)

val routine_name : ('d, 'r) linalg -> string
(** [routine_name l] names [l]'s routine in messages, as ["Cholesky"]. *)

(** {1:reading Reading results} *)

(** The outputs of a map whose dtypes are known at run time, and how to read
    its result as values. *)
type 'd outputs =
  | Outputs : ('d, 'r) outs * ('r -> 'd any list) -> 'd outputs

val outputs : Nx_array.Dtype.any list -> 'd outputs
(** [outputs dts] is the outputs of the dtypes [dts], in order. *)

val iter_outs :
  ('v 's. ('v, 's, 'd) t -> unit) -> ('d, 'r) outs -> 'r -> unit
(** [iter_outs f outs r] is [f y] for each value [y] of a map's result [r], in
    position order. *)

val iter_reduction :
  ('v 's. ('v, 's, 'd) t -> unit) -> ('d, 'r) reduction -> 'r -> unit

val iter_reductions :
  ('v 's. ('v, 's, 'd) t -> unit) -> ('d, 'r) reductions -> 'r -> unit
(** [iter_reduction] and [iter_reductions] are {!iter_outs} for a loop's
    results. *)

val iter_results : ('v 's 'd. ('v, 's, 'd) t -> unit) -> 'r prim -> 'r -> unit
(** [iter_results f op r] is [f y] for each value [y] of [op]'s result [r], in
    position order. *)

val result_arrays : 'r prim -> 'r -> Nx_array.any array array
(** [result_arrays op r] is, per result of [op] in position order, the arrays
    of [r]'s value there: one per device of its placement. Raises
    [Invalid_argument] if a result is traced. *)
