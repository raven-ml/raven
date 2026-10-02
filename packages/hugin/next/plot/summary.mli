(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Summaries: what fitting needs of an occurrence's data, computed where it
    lives and read to the host at once. *)

module Scale := Hugin_next_kit.Scale

(** A channel of an occurrence, as summarising reads it. *)
type input =
  | In : {
      index : int;  (** Its binding's index in the mark. *)
      role : string;
      colour : bool;  (** A missing value keeps its row. *)
      lift : 'd Channel.lift;
      spec : 'd Scale.t;  (** Finds the missing values. *)
      fitted : bool;  (** Its hull or kept codes are summarised. *)
    }
      -> input

type summary = {
  hulls : (int * (float * float)) list;
      (** Per binding index, when some value is kept. *)
  codes : (int * int list) list;  (** Per binding index, increasing. *)
  notes : string list;  (** Problems with the data, for warnings. *)
}

val summarise : int array -> input list -> Nx.bool_t option -> summary
(** [summarise shape inputs filter] is the summary of the [inputs] of a mark of
    shape [shape], keeping only the rows where [filter] holds, if given. *)

val explicit_domain : 'd Scale.t -> 'd Scale.domain option
(** [explicit_domain s] is the domain of [s] if [s] sets it. *)

val along : int array -> int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [along shape a t] is the vector [t] of the length of axis [a], shaped to
    broadcast along that axis of [shape]. *)
