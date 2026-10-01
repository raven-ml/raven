(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kinds: the OCaml types that cells read as.

    [Talon_next.Kind] documents kinds and the binding rule. This interface
    exposes the representation of kinds and of record cells, which [Talon_next]
    hides. Kinds and record cells are defined together because they refer to
    each other: the record kind reads record cells, and a record cell's fields
    carry their kinds. *)

(** {1:repr Representation} *)

(** The index of extension types. It has no values: an extension column's cells
    read only through the extension's declaration. *)
type ext = |

(** The type for kinds whose cells read as ['a]. *)
type _ t =
  | Bool : bool t
  | Int : int t
  | Float : float t
  | String : string t
  | Binary : Binary.t t
  | Decimal : Decimal.t t
  | Date : Time.date t
  | Instant : Time.instant t
  | Span : Time.span t
  | List : 'a t -> 'a array t
  | Record : record t
  | Tensor : ('a, 'b) Nx.dtype -> ('a, 'b) Nx.t t
  | Ext : ext t  (** The kind of every extension type. It reads nothing. *)

and record = { fields : (string * field) iarray }
(** The type for record cells: fields in order. The names of a record's fields
    are distinct, and a field whose type is or contains an extension type is a
    {!Storage} field. *)

(** The type for the fields of a record cell. *)
and field =
  | Value : 'a t * 'a option -> field
      (** [Value (k, v)] is a field of kind [k], null when [v] is [None]. *)
  | Storage : 'a t * 'a option -> field
      (** [Storage (k, v)] is a field whose type is or contains an extension,
          holding its value with every extension replaced by its storage, read
          as [k]. No public kind reads it. *)

(** {1:kinds Kinds} *)

val bool : bool t
val int : int t
val float : float t
val string : string t
val binary : Binary.t t
val decimal : Decimal.t t
val date : Time.date t
val instant : Time.instant t
val span : Time.span t
val list : 'a t -> 'a array t
val tensor : ('a, 'b) Nx.dtype -> ('a, 'b) Nx.t t
val provably_equal : 'a t -> 'b t -> ('a, 'b) Stdlib.Type.eq option

val equal_witness : 'a t -> 'b t -> ('a, 'b) Stdlib.Type.eq option
(** [equal_witness k0 k1] is [Some Equal] iff [k0] and [k1] are the same kind,
    extension kinds included: the equality of the types of cells, where
    {!provably_equal} is the binding rule. *)

val has_ext : 'a t -> bool
(** [has_ext k] is [true] iff [k] is or contains {!Ext}. *)

val pp : Format.formatter -> 'a t -> unit
