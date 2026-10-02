(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Leaves: the columns of a flat Parquet schema, and the talon types they read
    as.

    A leaf is a primitive field of the schema's root: a name, a repetition, a
    physical type and the annotation talon reads it with. The annotation is
    normalized: a legacy [converted_type] becomes the logical type it means, and
    an annotation that does not apply to the physical type, that talon does not
    know, or that is [UNKNOWN] (the null type), is dropped. This module holds
    the mapping of {!Talon_next_parquet}'s {{!Talon_next_parquet.types}types}
    and the alternatives of [with_type]. *)

type t = {
  name : string;
  optional : bool;  (** [false] for a required field. *)
  physical : Meta.physical;
  length : int;  (** The length of a [fixed_len_byte_array], or [0]. *)
  annotation : Meta.logical option;
      (** One of [String], [Enum], [Json], [Bson], [Uuid], [Float16], [Date],
          [Time], [Timestamp], [Integer], [Geometry], [Geography] or [Decimal],
          that applies to [physical]. *)
}
(** The type for leaves. *)

val of_schema : Meta.element array -> t array
(** [of_schema elements] is the leaves of the schema [elements], in order.

    Raises {!Meta.Error} if the schema is malformed (its root is not a group,
    its children do not account for the other elements, or a field has no
    repetition), if two leaves have the same name or a name is not valid UTF-8,
    or if a field is one talon refuses, named in the message: a group, a
    repeated field or an interval. *)

(** {1:types Types} *)

val default : t -> Talon_next.Type.any option
(** [default l] is the type [l] reads as when no format says otherwise, or
    [None] for a decimal, which a format must declare. *)

val reads_as : t -> Talon_next.Type.any -> bool
(** [reads_as l t] is [true] iff [l] reads as [t]: [t] is [default l], or
    [string], [binary] or a categorical when [l] holds byte strings, or a
    datetime of any unit without a zone when [l] is an [int96], or, for a
    decimal, [float64], and [int64] up to 18 digits. *)

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp ppf l] formats [l]'s repetition, physical type and annotation in
    Parquet's schema language, without its name:
    [optional int32 (INTEGER(8,true))],
    [required fixed_len_byte_array(16) (UUID)]. *)

val pp_reads : Format.formatter -> t -> unit
(** [pp_reads ppf l] formats the types [l] reads as, for [with_type]'s errors:
    [float64], [string, binary or a categorical],
    [datetime of any unit, without a zone], [float64 or int64]. *)
