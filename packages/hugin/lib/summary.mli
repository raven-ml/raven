(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Summaries: what fitting needs of an occurrence's data, computed where it
    lives and read to the host at once. *)

(** A channel of an occurrence, as summarising reads it. *)
type input =
  | In : {
      index : int;  (** Its binding's index in the mark. *)
      keeps_row : bool;  (** A missing value keeps its row (colours). *)
      fitted : bool;  (** Its hull or kept codes are summarised. *)
      lift : 'd Lift.t;
    }
      -> input

type t = {
  hulls : (int * (float * float)) list;
      (** Per binding index, when some value is kept. *)
  codes : (int * int list) list;
      (** Per binding index of indexed codes ({!Channel.cat} without labels),
          increasing. *)
  notes : string list;  (** Problems with the data, for warnings. *)
}

val summarise :
  ?stack:Stacked.t -> int array -> input list -> Nx.host Nx.bool_t option -> t
(** [summarise ~stack shape inputs filter] is the summary of the [inputs] of a
    mark of shape [shape], keeping only the rows where [filter] holds, if given.
    A row missing in an input that does not keep its row is dropped from every
    other. With [stack], the hull of the stacked input is that of the intervals
    its kept rows cover, stacked over every row that is not dropped, so that
    each panel's stacks are those it draws. *)
