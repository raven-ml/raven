(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scans as requests, and their eager fold.

    A scan travels through the installations between its performer and a
    compiled call as a {!request} over its tensors in walk order
    ({!Nx.Ptree.flatten}): the carry, the rows and the outputs are structures
    whose types a construct cannot carry. Each installation passes it on
    transformed, with the leaves it adds after the scan's. Only a compiled call
    stages a scan; a scan that none stages raises {!Not_staged} back to its
    performer, which folds it with {!fold} where it is written, inside every
    installation around it. *)

type leaves = Nx.packed list
(** The type for a structure's tensors, in walk order. *)

type request = {
  req_carry : leaves;  (** The initial carry. *)
  req_xs : leaves;
      (** The stacked rows: at least one tensor, every one of the same positive
          leading length, the number of steps. *)
  req_step : leaves -> leaves -> leaves * leaves;
      (** [req_step carry row] is the next carry and the step's outputs. Every
          step's outputs have the first step's dtypes and shapes. *)
  req_reverse : bool;
      (** Whether the steps run from the last row to the first, as the transpose
          of a scan does. *)
}
(** The type for scan requests. *)

type result = {
  r_carry : leaves;  (** The final carry. *)
  r_ys : leaves;
      (** The outputs, each stacked along a new leading axis, row [i] from the
          step that read row [i]. *)
}
(** The type for the results of scans. *)

exception Not_staged
(** [Not_staged] is the answer to a {!request} that no compiled call stages. An
    installation lets it pass to the scan's performer, which folds. *)

val fold : request -> result
(** [fold r] runs [r]'s steps one after the other in the caller's
    interpretation, so every installation around the caller sees each step's
    operations. An exception of [r.req_step] propagates unchanged. *)
