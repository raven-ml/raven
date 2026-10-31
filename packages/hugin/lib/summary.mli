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
  spans : (int * (Hugin_kit.Time.t * Hugin_kit.Time.t)) list;
      (** Likewise, of instants. *)
  codes : (int * int list) list;
      (** Per binding index of indexed codes ({!Channel.cat} without labels),
          increasing. *)
  notes : string list;  (** Problems with the data, for warnings. *)
}

(** A mark's stacked length, stacked over every row that is not dropped, so
    that each panel's stacks are those it draws. *)
type stacked = {
  index : int;  (** The stacked input's binding index. *)
  starts : Nx.host Nx.float64_t;  (** [nan] where dropped. *)
  ends : Nx.host Nx.float64_t;
  placed : Nx.host Nx.bool_t * Nx.host Nx.bool_t;
      (** Where the starts and the ends are finite and not missing for the
          scale. *)
}

val stack :
  int array ->
  input list ->
  Stacked.t ->
  missing:(Nx.host Nx.float64_t -> Nx.host Nx.bool_t) ->
  stacked option
(** [stack shape inputs st ~missing] is the stacked length [st] of a mark of
    shape [shape], [missing] telling the values its scale cannot place. *)

val summarise :
  ?stacked:stacked -> int array -> input list -> Nx.host Nx.bool_t option -> t
(** [summarise ~stacked shape inputs filter] is the summary of the [inputs] of
    a mark of shape [shape], keeping only the rows where [filter] holds, if
    given. A row missing in an input that does not keep its row is dropped from
    every other. With [stacked], the hull of the stacked input is that of the
    placed ends of its kept rows and of their placed starts. *)
