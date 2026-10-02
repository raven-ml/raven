(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Resolved figures: expanded, arranged, with every scale merged and fitted. *)

module Scale := Hugin_next_kit.Scale

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

val placed : 'd member -> string option
(** [placed m] is the position or facet scale whose axis shows the role of [m]:
    ["x"] for [x] and [x2]. *)

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

type facet_panel = {
  pnid : Common.id;
  pfy : string option;
  pfx : string option;
}

(** {1:resolved Resolved figures} *)

type entry
(** A summary that a later resolve may reuse. *)

type t = {
  figure : Figure.t;
  view : View.t;
  shaped : Arrange.shaped;
  facets : (Common.id * facet_panel list) list;
  scales : fitted list;  (** In the order of their first readers. *)
  nodes : (Common.id * (Common.id * Arrange.shares) list) list;
      (** Each node with the cells it lies in and the scopes it reads there. *)
  warnings : Common.warning list;
  cache : entry list;
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
