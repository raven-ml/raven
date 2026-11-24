(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Maps, reductions and scans of programs: plans of the loop kernels, which run
    a tape the program is lowered to (nx_metal.mli's Loops).

    Each plan takes [holds], where [holds k] is the threads Metal lets a group
    of [k]'s pipeline have, at least {!Kernels.threads}[ k]: a group runs on one
    core, so a core holds as many threads of [k] at once, and a launch's
    estimated time counts its groups' waves by it. *)

val map :
  holds:(Kernels.t -> int) ->
  Nx_kernel.Prog.t ->
  Operands.t ->
  (Plan.t option, Nx_array.answer) result
(** [map ~holds p xs] plans [p] from the loads of the call of facts [xs] into
    its destinations ({!Operands}). It is [Error Declined] where a kind computes
    on a float64 or complex128 value or a launch's least work exceeds
    {!Plan.allowed}, and for a call of no element where the call of one element
    is; it is [Ok None] for any other call of no element.

    nx.metal has checked the call: [p] and [xs] fit. *)

val reduce :
  holds:(Kernels.t -> int) ->
  Nx_kernel.Prog.t ->
  shape:int iarray ->
  axes:int iarray ->
  (Nx_kernel.reduction * int) iarray ->
  Operands.t ->
  (Plan.t option, Nx_array.answer) result
(** [reduce ~holds p ~shape ~axes rs xs] plans the reductions [rs] of [p]'s
    outputs over [axes] of a loop of [shape], of at least one term, into [xs]'
    destinations, {!destinations} each in order. It is [Ok None] where [shape]
    has no element and declines as {!map} does. *)

val scan :
  holds:(Kernels.t -> int) ->
  Nx_kernel.Prog.t ->
  axis:int ->
  Nx_kernel.reduction * int ->
  Operands.t ->
  (Plan.t option, Nx_array.answer) result
(** [scan ~holds p ~axis (r, k) xs] plans the scan by [r] of [p]'s output [k]
    along [axis] into [xs]' destinations, from its carry where it has one. It is
    [Ok None] for a call of no element and declines as {!map} does. *)

val bound_direct : bool -> unit
(** [bound_direct b] is nx_metal.mli's [bound_direct]. *)

val destinations : Nx_kernel.reduction -> int
(** [destinations r] is the destinations a reduction by [r] writes: two for an
    [Arg] (the extreme and its position) and for [Moments] (the mean and the
    variance), one otherwise. *)

(** {1:temps Temporaries of nodes}

    Another family's call computes nodes of a program by maps into temporaries
    of its own builder's workspace, which its later launches read. A temporary
    of a node of dtype [d] holds an element of [d] for each index of the loop's
    axes along which the node varies, C-contiguous from a byte on 256, and is
    read at stride [0] along the other axes. *)

type temps
(** The type for planned maps of nodes into temporaries. *)

val temps :
  holds:(Kernels.t -> int) ->
  Nx_kernel.Prog.t ->
  shape:int array ->
  score:bool ->
  Operands.operand array ->
  int list ->
  (temps, Nx_array.answer) result
(** [temps ~holds p ~shape ~score loads roots] plans the maps of [p]'s nodes
    [roots], in order, into temporaries over a loop of [shape], [p]'s input [0]
    being the {e score} and its inputs from [1] the loads of facts [loads],
    whose buffers are the call's operands they name ({!Operands.operand}'s
    [buffer]). With [score], the score is a C-contiguous [float32] temporary of
    [shape], which the caller writes before the maps run ({!score_at}); without,
    each read of [p]'s input [0] is a zero. It is [Error Declined] where {!map}
    declines a call.

    Raises [Invalid_argument] if an extent of [shape] is not positive, or if
    [roots] is empty or names no node of [p]. *)

val score_at : Plan.builder -> temps -> int
(** [score_at b t] is the byte offset into [b]'s workspace of a piece that holds
    [t]'s score and stays taken. The caller adds the launch that writes the
    score there to [b] before {!emit}.

    Raises [Invalid_argument] if [t] was planned without [score]. *)

val emit : ?score:int -> Plan.builder -> temps -> (int * int array) array
(** [emit ~score b t] adds [t]'s launches to [b], after its other launches,
    which read the score at byte [score] of [b]'s workspace ({!score_at}). It
    is, for each of [t]'s roots in order, the byte offset into [b]'s workspace
    of its temporary and its element strides over the loop's axes. The roots'
    pieces stay taken, so the launches the caller adds after read them; the
    other pieces [t]'s launches take are given back ({!Plan.release}).

    Raises [Invalid_argument] if [score] is given and [t] was planned without
    [score], or the other way round. *)

(**/**)

(** A launch's mode: a map of the tape or of a direct form; a cast by the named
    kernel; a reduction of the tape, a direct one, a direct [Logsumexp]'s or
    [Moments]' second pass, or a row's; the merge of a reduction's states; a
    scan; a scan's carries; a network. *)
type mode =
  | Map
  | Direct
  | Cast of Kernels.t
  | Reduce
  | Reduce_direct
  | Reduce_centered
  | Reduce_row of Kernels.t
  | Merge
  | Scan
  | Carries
  | Network

val kernel_of : mode -> int array -> Kernels.t
(** [kernel_of mode h] is the kernel of [mode] that a launch of header [h] (the
    words from {!Kernels.loop_mode}) runs, of the v, width, tiles, padded loads
    and lanes' width [h] names. Raises [Invalid_argument] where no kernel has
    them: a map's tiles, a direct map or a cast other than at 4 elements a
    thread of 32-bit indices with no padded load, or a cast's tiles; 32-bit
    lanes ({!Kernels.loop_bits32}) but in a direct map of quads whose form
    ({!Kernels.loop_direct}) takes each element and gives its result as it is;
    16-bit lanes ({!Kernels.loop_bits16}) but in a direct map of quads whose
    form decodes each element and encodes its result by one narrow float's
    codec; 8-bit lanes ({!Kernels.loop_bits8}) but in a direct map of quads
    whose form takes each element as it is or as a truth, gives its result as it
    is or wrapped, and computes no long kind; lanes of two widths together; a
    map, a reduction, a scan or a network of a v other than 1 and 4, or of
    64-bit indices at 4. *)
