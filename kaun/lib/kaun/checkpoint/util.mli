val mkdir_p : string -> unit
(** Recursively create a directory if it does not already exist. *)

val remove_tree : string -> unit
(** Recursively delete a directory and all of its contents. *)

val encode_path : string -> string
(** Encode an arbitrary path so it can be used as a filesystem-friendly name. *)

val decode_path : string -> string
(** Inverse of {!encode_path}. Unknown encodings are returned unchanged. *)

val slugify : string -> string
(** Convert an arbitrary label into a lower-case, dash-separated slug. *)

val now_unix : unit -> float
(** Current Unix timestamp, in seconds. *)
