(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Leaves and columns: a Parquet schema as the columns talon holds, and the
    talon types they read as.

    A leaf is a primitive field of the schema: a path, a repetition, a physical
    type and the annotation talon reads it with. The annotation is
    normalized: a legacy [converted_type] becomes the logical type it means, and
    an annotation that does not apply to the physical type, that talon does not
    know, or that is [UNKNOWN] (the null type), is dropped. This module holds
    the mapping of {!Talon_parquet}'s {{!Talon_parquet.types}types} and the
    alternatives of [with_type]. *)

type t = {
  rev_path : string list;
      (** [path_in_schema], last name first, sharing its tail with its parent's
          field. *)
  index : int;  (** Its position among the file's leaves and column chunks. *)
  nested : bool;
      (** [false] for a top-level field that is not repeated: a flat column,
          whose levels are at most one bit. *)
  optional : bool;  (** [true] for a field declared optional. *)
  physical : Meta.physical;
  length : int;  (** The length of a [fixed_len_byte_array], or [0]. *)
  annotation : Meta.logical option;
      (** One of [String], [Enum], [Json], [Bson], [Uuid], [Float16], [Date],
          [Time], [Timestamp], [Integer], [Geometry], [Geography] or [Decimal],
          that applies to [physical]. *)
  rep : int;  (** Its maximum repetition level. *)
  def : int;  (** Its maximum definition level. *)
  slot_def : int;
      (** The definition level of the slots its values fill: the deepest
          repeated field above it or itself, or [0]. *)
}
(** The type for leaves: a primitive field and the levels it is striped with. *)

type slots = { rep : int; def : int; nonnull : int }
(** The type for a node's slots among a leaf's level entries. An entry of
    repetition level [r] and definition level [d] opens a slot iff [r <= rep]
    and [d >= def], and the slot holds a value iff [d >= nonnull]. A node with
    [nonnull <= def] is never null. *)

(** The type for the talon column a schema field reads as. *)
type node =
  | Prim of t
  | Struct of { slots : slots; fields : (string * node) list }
      (** A record: every field fills its slots. *)
  | List of {
      slots : slots;
      elem_rep : int;
      elem_def : int;
      element : node;
    }
      (** A list. An entry of levels [r] and [d] adds an element iff
          [r <= elem_rep] and [d >= elem_def]. *)

type column = { name : string; node : node; text : string }
(** The type for a top-level column: its name, its node, and its Parquet type in
    Parquet's schema language, without its name. *)

type schema = { leaves : t array; columns : column array }
(** The type for a file's schema: its leaves, in column chunk order, and its
    top-level columns. *)

val of_schema : Meta.element array -> schema
(** [of_schema elements] is the schema of [elements]. A group annotated [LIST]
    reads as a list whose element follows the five rules of Parquet's
    {{:https://github.com/apache/parquet-format/blob/master/LogicalTypes.md}
    LogicalTypes}, a group annotated [MAP] or [MAP_KEY_VALUE] as the list of its
    repeated group's records, any other group as a record, and a repeated field
    outside those groups as a required list of required elements.

    Raises {!Meta.Error} if the schema is malformed (its root is not a group,
    its children do not account for the other elements, a field has no
    repetition, a group has no fields, a [LIST] or [MAP] group does not hold one
    repeated field, or a map's entries are not a group), if two columns or two
    fields of a group have the same name or a name is not valid UTF-8, or if a
    field is one talon refuses, named in the message: a repeated [LIST] or [MAP]
    outside a list, an interval or a variant. *)

val name : t -> string
(** [name l] is [l]'s dotted path: ["a.list.element"]. It takes time linear in
    the path. *)

(** {1:types Types} *)

val default : t -> Talon.Type.any option
(** [default l] is the type [l] reads as when no format says otherwise, or
    [None] for a decimal, which a format must declare. *)

val reads_as : t -> Talon.Type.any -> bool
(** [reads_as l t] is [true] iff [l] reads as [t]: [t] is [default l], or
    [string], [binary] or a categorical when [l] holds byte strings, or a
    wall-clock datetime of any unit when [l] is an [int96], or, for a
    decimal, [float64], and [int64] up to 18 digits. *)

val default_node : node -> Talon.Type.any option
(** [default_node n] is the type [n] reads as when no format says otherwise, or
    [None] when a leaf under it is a decimal. *)

val node_reads_as : node -> Talon.Type.any -> bool
(** [node_reads_as n t] is [true] iff [n] reads as [t]: a leaf as {!reads_as}
    says, a list as a list of what its element reads as, and a record as a
    record of the same field names in order, each reading as its field does. *)

(** {1:fmt Formatting} *)

val pp_reads : Format.formatter -> t -> unit
(** [pp_reads ppf l] formats the types [l] reads as, for [with_type]'s errors:
    [float64], [string, binary or a categorical],
    [wall-clock datetime of any unit], [float64 or int64]. *)

val pp_node_reads : Format.formatter -> node -> unit
(** [pp_node_reads ppf n] formats the types [n] reads as, as {!pp_reads} does
    for a leaf: [list[float64]], [record[a int32, b float64 or int64]]. *)
