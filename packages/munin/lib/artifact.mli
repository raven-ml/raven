(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Versioned local artifacts.

    Artifacts are immutable named payloads with versions, aliases, and lineage.
*)

(** {1:types Types} *)

type kind =
  [ `dataset  (** Dataset. *)
  | `model  (** Model weights. *)
  | `checkpoint  (** Checkpoint. *)
  | `file  (** Single file. *)
  | `dir  (** Directory tree. *)
  | `other  (** Unclassified artifact. *) ]
(** The type for logical artifact kinds. *)

type payload =
  [ `file  (** Single file payload. *) | `dir  (** Directory tree payload. *) ]
(** The type for materialized payload kinds. *)

type t
(** The type for artifact handles. *)

(** {1:loading Loading} *)

val load : root:string -> name:string -> version:string -> t option
(** [load ~root ~name ~version] is the artifact named [name] at [version], if
    present.

    If [version] does not match an explicit version, it is resolved as an alias.
    Returns [None] if neither matches. *)

val list :
  root:string ->
  ?name:string ->
  ?kind:kind ->
  ?alias:string ->
  ?producer_run:string ->
  ?consumer_run:string ->
  unit ->
  t list
(** [list ~root ()] is the artifacts stored under [root], filtered when the
    optional selectors are provided. Results are sorted by name, then by version
    number. *)

(** {1:identity Identity} *)

val name : t -> string
(** [name t] is the logical artifact name. *)

val kind : t -> kind
(** [kind t] is the artifact's logical kind. *)

val payload : t -> payload
(** [payload t] is the materialized payload kind. *)

val version : t -> string
(** [version t] is the explicit version such as ["v1"]. *)

(** {1:content Content} *)

val digest : t -> string
(** [digest t] is the content-addressed SHA-256 digest of the payload. *)

val path : t -> string
(** [path t] is the absolute path to the materialized payload in the blob store.
*)

val size_bytes : t -> int
(** [size_bytes t] is the total byte size of the materialized payload. *)

val metadata : t -> (string * Value.t) list
(** [metadata t] is the artifact metadata. *)

val aliases : t -> string list
(** [aliases t] is the alias list attached to the version. *)

val has_alias : t -> string -> bool
(** [has_alias t alias] is [true] iff [alias] points at [t]. *)

(** {1:lineage Lineage} *)

val producer_run_id : t -> string option
(** [producer_run_id t] is the producing run identifier, if known. *)

val consumer_run_ids : t -> string list
(** [consumer_run_ids t] is the list of consuming run identifiers. *)

val created_at : t -> float
(** [created_at t] is the artifact creation timestamp ([Unix.gettimeofday] at
    creation time). *)

(**/**)

val create :
  root:string ->
  name:string ->
  kind:kind ->
  payload:payload ->
  digest:string ->
  path:string ->
  metadata:(string * Jsont.json) list ->
  aliases:string list ->
  producer_run_id:string option ->
  t

val add_consumer :
  root:string -> name:string -> version:string -> string -> unit

(**/**)
