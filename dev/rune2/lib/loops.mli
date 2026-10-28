(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Loops' programs taken apart and put together, for the rules of the
    transformations over a scan ({!Program.Scan}).

    A scan's step takes its parameters in three runs: the carry, a row of each
    of the rows' arrays, and the captures; it returns the carry, the outputs,
    then the additions. A rule derives another step from it, whose parameters
    and outputs are laid out in the same three runs. *)

val row : Program.value -> Program.form
(** [row x] is the form of [x]'s rows along axis 0: [x]'s dtype, a contiguous
    layout of its shape without axis 0, its placement without its leading axis.
*)

val filled : zero:bool -> Program.form -> Program.value
(** [filled ~zero f] is a value of [f]'s dtype, shape and placement, every
    element [+0] where [zero] and otherwise the additive identity that keeps
    every sign: [-0.], or [0] where the dtype has no negative zero. *)

val zeros_like : Program.value -> Program.value
(** [zeros_like x] is [Nx.zeros_like x]. *)

val add : Program.value -> Program.value -> Program.value
(** [add a b] is [Nx.add a b], [b] read at [a]'s form. *)

val holds_iterate : Program.t -> bool
(** [holds_iterate p] is [true] iff [p] or a program its constructs carry has an
    iterate. *)

val flip : Program.value -> Program.value
(** [flip x] is [x] with its axis 0 reversed, a view. *)

val reorder : Program.t -> int array -> Program.t
(** [reorder p order] is [p] whose argument [k] is [p]'s argument [order.(k)],
    its totals' running values in last as in [p]. *)

val pick : bool array -> 'a array -> 'a array
(** [pick mask a] is the elements of [a] where [mask] holds, in order. *)

val split : int list -> 'a array -> 'a array list
(** [split lengths a] is [a] cut into consecutive runs of [lengths].

    Raises [Invalid_argument] unless the lengths sum to [a]'s length. *)

val merge : bool array -> 'a array -> 'a array -> 'a array
(** [merge mask vs ws] is [vs] at the positions where [mask] holds and [ws] at
    the others, each in order. *)
