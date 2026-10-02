(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Resolved figures: expanded, arranged, with every scale merged and fitted. *)

module Scale := Hugin_kit.Scale

(** {1:scales Scales} *)

type 'd member = {
  m_occ : Arrange.occ;
  m_pid : Common.id;
  m_index : int;
  m_role : string;
  m_use : Role.use;
  m_d : 'd Channel.data;
  m_imply : 'd Scale.t option;
  m_guide : bool option;
}
(** A channel that reads a scale, in one cell. *)

val by_order : 'd member list -> 'd member list
(** [by_order ms] is [ms] in the order of the figure. *)

(** A scale of a scope, with the channels that read it. *)
type fitted =
  | F : {
      name : string;
      key : Arrange.key;
      kind : 'd Channel.kind;
      members : 'd member list;
      legend : bool;
          (** A role other than a position or facet reads it without
              [map_range]. *)
      guide : bool option;
      spec : 'd Scale.t;  (** Merged. *)
      scale : 'd Scale.t;  (** Fitted, then zoomed. *)
    }
      -> fitted

val category_names : string Scale.t -> string list
(** [category_names s] is the names of the categories of the domain of [s]. *)

(** {1:facets Facets} *)

(** Where the rows of a mark go along one facet of its cell. *)
type facet =
  | Every  (** It binds no data to the facet: every category. *)
  | One of int option  (** A constant: the index of its category, if any. *)
  | Each of Nx.int64_t
      (** Each row's category index, [-1] for none, where the data lives. *)

type part = { px : facet; py : facet }
(** Where the rows of a mark go among the panels of its cell. *)

type panel = {
  pnid : Common.id;  (** The cell's, or its facet panel's. *)
  pfx : int option;  (** Its category in the cell's fx scale. *)
  pfy : int option;
  reads : (Common.id * int option array) list;
      (** Per mark, the index in the scales of the scale each binding reads. *)
}

type cell = {
  content : Arrange.content;
  fx : int option;  (** The index in the scales of its fx scale. *)
  fy : int option;
  panels : panel list;  (** By fy category, then fx category. *)
  parts : (Common.id * part) list;  (** Per mark. *)
}

val mask : part -> panel -> [ `All | `None | `Mask of Nx.bool_t ]
(** [mask part p] is the rows of a mark of [part] in [p]: all, none, or those of
    a mask that broadcasts to the mark's shape. *)

val shown :
  Arrange.content ->
  (Common.id * int option array) list ->
  Role.shown ->
  int option
(** [shown c reads on] is the index in the scales of the scale that the guide
    [on] shows where the marks of [c] read the scales [reads], as a panel's. *)

(** {1:resolved Resolved figures} *)

type t = {
  figure : Figure.t;
  view : View.t;
  shaped : Arrange.shaped;
  cells : (Common.id * cell) list;
  scales : fitted list;  (** In the order of their first readers. *)
  nodes : (Common.id * (Common.id * Arrange.shares) list) list;
      (** Each node with the cells it lies in and the scopes it reads there. *)
  warnings : Common.warning list;
}

val resolve : ?prev:t -> ?view:View.t -> Figure.t -> t
val scale : ?at:Common.id -> t -> 'd Scale.t -> 'd Scale.t
val warnings : t -> Common.warning list

val panel_of : Arrange.key -> Common.id option
(** [panel_of k] is the facet panel of the scope [k], if any. *)

val find_path : Common.id -> (Common.id * 'a) list -> 'a option
(** [find_path id l] is the value of [id] in [l], if any. *)

val dedupe : Common.warning list -> Common.warning list
(** [dedupe ws] is [ws] without repeated warnings, first ones kept. *)

val equal_warning : Common.warning -> Common.warning -> bool
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
