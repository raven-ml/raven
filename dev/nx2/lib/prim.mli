(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operations as data, and values' facts: the functions over {!Value.prim} and
    {!Value.t} that the engine and [Nx.Prim] share.

    Every function is total over the constructors. {!route} states each
    operation's rule once: the shapes, dtypes, axes and attributes its
    operands must have, where it reads them, and each result's layout and
    placement, from {!Route}; {!results_at} makes the results' forms from what
    it decided. The eager engine routes, places and allocates destinations
    through them, and interpretations make traced results through {!results},
    so the two agree in dtype, layout and placement.
    A sharded value's shape is the whole's, which it holds; each shard is its
    device's window of it. *)

open Value

(** {1:facts Values' facts} *)

val form : ('v, 's, 'd) t -> ('v, 's, 'd) form
val dtype : ('v, 's, 'd) t -> ('v, 's) dtype

val at : ('v, 's, 'd) t -> 'd Devices.placement option
(** [at x] is where [x]'s bytes are, or will be for a traced value: [None] for a
    value of every set. *)

val placement : ('v, 's, 'd) t -> 'd Devices.placement
(** [placement x] is [x]'s placement, as {!at} gives it. Raises
    [Invalid_argument] for a value of every set. *)

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

val alive : by:string -> int -> ('v, 's, 'd) t -> unit
(** [alive ~by i x] raises [Invalid_argument] naming [by], operand [i + 1] and
    the function [x] was donated to, if [x] died. *)

val arrays : ('v, 's, 'd) t -> ('v, 's) Nx_array.t iarray
(** [arrays x] is [x]'s arrays, one per device of its placement, whether [x]
    lives or not. Raises [Invalid_argument] for a constant or a traced value. *)

val any : ('v, 's) Nx_array.t iarray -> Nx_array.any array
(** [any xs] is [xs] untyped, in order. *)

val rank : ('v, 's, 'd) t -> int

val dim : ('v, 's, 'd) t -> int -> int
(** Raises [Invalid_argument] if the axis is not below {!rank}. *)

val same_shape : ('v, 's, 'd) t -> ('w, 'r, 'd) t -> bool
(** [same_shape x y] is [shape x = shape y]. It allocates nothing. *)

val shape : ('v, 's, 'd) t -> int array
(** [shape x] is a fresh array. [dtype], [placement], [rank] and [dim] of a
    value on one device allocate nothing. *)

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

val check_shape : by:string -> int array -> unit
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
(** [kind n] names [n]'s kind in messages, as ["Exp"] or ["Less"]. *)

val operands : 'r prim -> operands
(** [operands op] is [op]'s operands in order: a map's loads, a contraction's
    [a], [b] then [init], [Check]'s [ok] then its data, the one operand of the
    others. *)

val own_index :
  'r prim -> (int list * Nx_array.Layout.t * Nx_array.Dtype.any) option
(** [own_index op] is the positions in {!operands} of the operands [op] reads
    only at its first result's own index, each of which a kernel may take as
    that result's destination ({!Nx_kernel.S}), with that result's layout and
    dtype: a map's plain loads, a scatter's [into], a contraction's [init], an
    assembly's first piece where it covers the whole result. [None] for an
    operation with none. *)

val iteri : ('v 's 'd. int -> ('v, 's, 'd) t -> unit) -> 'r prim -> unit
(** [iteri f op] is [f i x] for each operand [x] of [op], at its position [i] in
    {!operands}. *)

val exists : ('v 's 'd. ('v, 's, 'd) t -> bool) -> 'r prim -> bool
(** [exists f op] is whether [f x] for some operand [x] of [op]. Neither
    allocates for an operation other than [Check]. *)

val map : ('v 's 'd. ('v, 's, 'd) t -> ('v, 's, 'd) t) -> 'r prim -> 'r prim
(** [map m op] is [op] with each operand [x] replaced by [m x]. *)

val mapi :
  ('v 's 'd. int -> ('v, 's, 'd) t -> ('v, 's, 'd) t) -> 'r prim -> 'r prim
(** [mapi m op] is [op] with each operand [x] at position [i] in {!operands}
    replaced by [m i x]: [op] itself where [m] returns each operand
    itself. *)

val rule : 'r prim -> Route.rule
(** [rule op] is how [op]'s results depend on its operands: its route and each
    device's windows of its operands ({!Route.window}) come from it alone. A
    loop of no load has the rule of one load of its shape: each device of its
    placement computes its window of the loop. A contraction's scale is read
    at its operand's windows where a cut keeps its blocks whole, and whole
    elsewhere; its segmented axes are read whole. A segmented contraction
    is read whole along its contracting pairs, as is a scaled one unless
    each scale's blocks merge with its operand's batch, free and
    contracting axes ({!block_run}) and the run of a block of its contracted
    extent divides {!contract_block} or that extent is at most one block. *)

val local : 'r prim -> Nx_array.Move.range array -> 'r prim
(** [local op w] is [op] computing the window [w] of its first result, over
    operands the caller gives at their windows ({!Route.window}): a loop over
    its window of its shape, its programs' coordinates offset by that window's
    start, and a fold of [w]'s shape.

    Raises [Invalid_argument] for a window of an assembly that is not the whole:
    its rule computes it whole. *)

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
    rule. *)

(** The type for what an operation's rule decides. *)
type routed = {
  route : unit Route.t option;
      (** Where it reads each operand, in {!operands}' order, where its
          results lie and how its devices compute them: [None] where every
          operand is of every set, and for [Place] and [Check], which are not
          routed. *)
  layouts : Nx_array.Layout.t array;  (** Its results' ({!layouts}). *)
}

val route : by:string -> 'r prim -> routed
(** [route ~by op] is what [op]'s rule decides, checked once. Operands that
    lie at one placement that cuts no axis, the others of every set, are read
    there and their results lie there; otherwise {!Route.route} of {!rule}
    decides. A deferred value's placement and its eager twin's come from this
    one route over the same operands.

    Raises as {!layouts} does, then as {!Route.route} does. *)

val results_at :
  unit Devices.placement option ->
  Nx_array.Layout.t array ->
  ('v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) t) ->
  'r prim ->
  'r
(** [results_at p layouts m op] is [op]'s result, its value at position [k]
    made by [m k f], [f] the form eager execution gives it at [p], laid out as
    [layouts], [op]'s {!layouts}; [()] for [Check]. A [Place]'s result lies
    at its own placement. It checks nothing. *)

val results :
  by:string ->
  ('v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) t) ->
  'r prim ->
  'r
(** [results ~by m op] is {!results_at} of what {!route} decides for [op]:
    its results' layouts, at the placement of its results.

    Raises [Invalid_argument] naming [by], before [m] is called, where [op]'s
    operands break its rule. *)

val reduction : ('d, 'a) reduction -> Nx_kernel.reduction * int
(** [reduction r] is [r] as {!Nx_kernel.S.scan} takes it: what it computes and
    the program output it reduces. *)

val reductions : ('d, 'r) reductions -> (Nx_kernel.reduction * int) array
(** [reductions rs] is [rs] as {!Nx_kernel.S.reduce} takes them. *)

val reduction_name : Nx_kernel.reduction -> string
(** [reduction_name r] names [r] in messages, as ["Logsumexp"] or
    ["Arg Max"]. *)

val reduced : int array -> int iarray -> int array
(** [reduced s axes] is [s] without [axes]. *)

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

val accepted2 : by:string -> Nx_kernel.Prog.op2 -> ('v, 's, 'd) t -> unit
(** [accepted2 ~by k x] raises as {!op2} does where [k] does not take operands
    of [x]'s dtype. *)

val accepted3 :
  by:string -> Nx_kernel.Prog.op3 -> ('a, 'b, 'd) t -> ('v, 's, 'd) t -> unit
(** [accepted3 ~by k c x] raises as {!op3} does where [k] does not take [c] and
    operands of [x]'s dtype. *)

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

val contract_block : int
(** [contract_block] is [4096], the products of a contraction's block
    ({!Nx_kernel.S.contract}). *)

val block_run : int array -> int array -> int array -> int option
(** [block_run s ss axes] is [Some r] where the axes [axes] of an operand of
    shape [s], merged into one in order, leave each block of its scale of
    shape [ss] [r] consecutive elements of that axis: [axes] hold one
    element per block, then one axis of any blocks, then axes of one block.
    It is [Some 1] where one of [axes] has no element, and [None] where
    blocks do not merge so. *)

val contract_groups :
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  int ->
  int ->
  int array array * int array array
(** [contract_groups ~batch ~contracting ra rb] groups the axes of a
    contraction's operands of ranks [ra] and [rb]: [a]'s as
    [[|batch; rows; terms|]] and [b]'s as [[|batch; terms; columns|]], the
    batch and term axes in their pairs' order and the others increasing.
    Partials lay each operand and its scale out by these groups, and
    {!rule} lets them take a scale by the same groups. *)

val contract_order :
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  ('a, 'b, 'd) t ->
  ('c, 'e, 'd) t ->
  Nx_kernel.order
(** [contract_order ~batch ~contracting a b] is the order of summation
    ({!Nx_kernel.section-orders}) of a contraction of [a] with [b]: the one the
    kernels of [a]'s set, or of [b]'s where [a] is of every set, choose
    ({!Nx_kernel.S.contract_order}) for its rows, the product of [a]'s extents
    along the axes no pair names, its columns, [b]'s likewise, and its
    depth, [a]'s extents along the contracting pairs multiplied, each
    [max_int] past it. Where that set has no kernels, as one of devices of
    several kinds, or both are of every set, it is the one nx.cpu chooses. *)

(** {1:loads Loads} *)

val load_any : 'd load -> 'd any
(** [load_any l] is the operand [l] reads. *)

val load_dtype : 'd load -> Nx_array.Dtype.any
val is_padded : 'd load -> bool

(** {1:families Transforms and factorisations} *)

val routine_name : ('d, 'r) linalg -> string
(** [routine_name l] names [l]'s routine in messages, as ["Cholesky"]. *)

val is_constant : ('v, 's, 'd) t -> bool

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
    [Invalid_argument] if a result is a constant. *)
