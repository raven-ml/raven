(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Loops' programs taken apart and put together, for the rules of the
    transformations over a scan ({!Program.Scan}).

    A scan's step takes as parameters the carry, a row of each array the scan
    reads by rows, the captures, then each total's running value in. It returns
    the carry, the outputs, then the running values out. A rule derives another
    step from it, whose parameters and outputs are laid out in the same order.
*)

val row : Program.value -> Program.form
(** [row x] is the form of [x]'s rows along axis 0. It has [x]'s dtype, a
    contiguous layout of [x]'s shape without axis 0, and [x]'s placement without
    its leading axis. *)

val filled : zero:bool -> Program.form -> Program.value
(** [filled ~zero f] is a value of [f]'s dtype, shape and placement, every
    element [+0] where [zero]. Otherwise every element is the additive identity
    that keeps every sign. That is [-0.], or [0] for a dtype without a negative
    zero. *)

val zeros_like : Program.value -> Program.value
(** [zeros_like x] is [Nx.zeros_like x]. *)

val add : Program.value -> Program.value -> Program.value
(** [add a b] is [Nx.add a b], [b] read at [a]'s form. *)

val reorder : Program.t -> int array -> Program.t
(** [reorder p order] is [p] whose argument [k] is [p]'s argument [order.(k)],
    its totals' running values in last as in [p]. *)

val pick : bool array -> 'a array -> 'a array
(** [pick mask a] is the elements of [a] where [mask] holds, in order. [mask]
    must be at least as long as [a]. *)

val split : int list -> 'a array -> 'a array list
(** [split lengths a] is [a] cut into consecutive runs of [lengths].

    Raises [Invalid_argument] unless the lengths sum to [a]'s length. *)

val merge : bool array -> 'a array -> 'a array -> 'a array
(** [merge mask vs ws] is [vs] at the positions where [mask] holds and [ws] at
    the others, each in order. [vs] must hold an element for each position where
    [mask] holds, and [ws] one for each other. *)
