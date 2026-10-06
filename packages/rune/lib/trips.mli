(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Loops as requests, and their eager fold.

    A loop travels through the installations between its performer and a
    compiled call as a {!request} over its tensors in walk order
    ({!Nx.Ptree.flatten}): the carry, the rows and the outputs are structures
    whose types a construct cannot carry. A loop runs its step over stacked
    rows, as a scan does, or until a stop holds, as an iterate does. Each
    installation passes it on transformed, with the leaves it adds after the
    loop's. Only a compiled call stages a loop; a loop that none stages raises
    {!Not_staged} back to its performer, which folds it with {!fold} where it is
    written, inside every installation around it. *)

type leaves = Nx.packed list
(** The type for a structure's tensors, in walk order. *)

(** The type for the trips a loop takes. *)
type trips =
  | Rows of {
      xs : leaves;
          (** The stacked rows: at least one tensor, every one of the same
              positive leading length, the number of steps. *)
      reverse : bool;
          (** Whether the steps run from the last row to the first, as the
              transpose of a scan does. *)
    }  (** One step per row, each passed its row. *)
  | Until of {
      until : leaves -> Nx.bool_t;
          (** [until carry] holds, in every element, once the loop is done. It
              is tested before each step. *)
      max : int;  (** The most steps the loop takes, at least [0]. *)
      failure : int array -> string;
          (** [failure i] is the message of the error the loop raises when
              [until] still fails after [max] steps, [i] the index of its first
              false element. *)
    }  (** Steps until a stop holds, each passed no row. *)

type trip = {
  index : (int32, Nx.int32_elt) Nx.t;
      (** The trip's index, an [int32] scalar: the row a loop over rows reads,
          and the count of the steps before it for a loop until a stop. A
          compiled call's loop passes the trip index of its program. *)
  key : unit -> Nx.Rng.t;
      (** [key ()] is the loop's key, which the step takes from the scope around
          the loop at its first draw: {!Nx.Rng.next_key}, or {!Nx.Rng.peek} for
          a run of the step that a loop which takes no trip makes, which takes
          no key. *)
}
(** The type for trips of a loop. *)

type request = {
  req_carry : leaves;  (** The initial carry. *)
  req_trips : trips;  (** The trips the loop takes. *)
  req_step : trip -> leaves -> leaves -> leaves * leaves;
      (** [req_step trip carry row] is the next carry and the step's outputs.
          Every step's outputs have the first step's dtypes and shapes. *)
}
(** The type for loop requests. *)

type result = {
  r_carry : leaves;  (** The final carry. *)
  r_ys : leaves;
      (** The outputs of the steps taken, each stacked along a new leading axis:
          over rows, row [i] from the step that read row [i]; until a stop, row
          [k] from step [k]. A loop that took no step has none. A compiled
          call's loop until a stop has [max] rows, those past its last step
          holding no value of the loop. *)
}
(** The type for the results of loops. *)

exception Not_staged
(** [Not_staged] is the answer to a {!request} that no compiled call stages. An
    installation lets it pass to the loop's performer, which folds. *)

val fold : request -> result
(** [fold r] runs [r]'s steps one after the other in the caller's
    interpretation, so every installation around the caller sees each step's
    operations. A stop is read with {!Nx.item}, and once it fails after [max]
    steps, {!Nx.check} raises its failure; inside a compiled call, which cannot
    read it, it raises {!Lower.Jit_error}. A traced step's carry is copied
    ({!Nx.copy}), so that a compiled call stores it before the next step reads
    it. An exception of [r.req_step] propagates unchanged. *)

val no_rows : leaves -> leaves
(** [no_rows ys] is the outputs of a loop that took no step, whose steps output
    values like [ys]: each tensor stacked along a new leading axis of length
    [0], on the host. *)

(** {1:transformed Transformed loops}

    A transformation passes a loop on with the leaves it adds after the loop's:
    the tangents of the carry tensors that have one, or the lanes of those that
    are lanes. A carry tensor that has none at [init] may have one after a step,
    which the transformed loop cannot carry: the attempt is abandoned and
    restarted with that tensor added. *)

val split : int -> 'a list -> 'a list * 'a list
(** [split n l] is the first [n] elements of [l] and the others. *)

val fixpoint : bool list -> (grow:(bool list -> unit) -> bool list -> 'r) -> 'r
(** [fixpoint active attempt] is [attempt ~grow active], where [active] marks
    what the transformed loop carries, such as which carry tensors it carries
    values of. A step of the attempt calls [grow next] with what the carry it
    returns has: when [next] marks something the attempt's [active] lacks,
    [grow] abandons the attempt and [fixpoint] restarts it with their union.
    Otherwise [grow] returns.

    An attempt changes no state that outlives it before its loop returns, so a
    restart forgets the abandoned one. [grow] belongs to its attempt: a step of
    a nested fixpoint's loop calls its own. Only an answer that runs a step
    before it returns, a compiled call or a map's fold of a loop whose lanes
    stop apart, restarts an attempt. *)
