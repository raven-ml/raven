(** Processes over the disk cache, and damage to its entries.

    A suite plays a part in a child: it starts again with the environment
    variable [TOLK_TEST_ROLE] naming the part, in the cache directory [CACHEDB]
    that the test chooses, and {!play} runs the part instead of the suite. *)

val play : (string * (unit -> unit)) list -> unit
(** [play parts] runs the part [TOLK_TEST_ROLE] names among [parts] and exits,
    if it names one. It returns if [TOLK_TEST_ROLE] is unset.

    Raises [Failure] if [TOLK_TEST_ROLE] names no part of [parts]. *)

val fresh : unit -> string
(** [fresh ()] is a cache directory no child has used, under the working
    directory. *)

type child
(** The type for children started and not yet finished. *)

val start : ?env:(string * string) list -> cachedb:string -> string -> child
(** [start ~env ~cachedb part] starts playing [part] with [CACHEDB] set to
    [cachedb] and each variable of [env] set. *)

val finish : child -> (string, string) result
(** [finish c] waits for [c]: [Ok] of what it wrote on its standard output if it
    exited with [0], [Error] of what it wrote on its standard error otherwise.
*)

val child :
  ?env:(string * string) list ->
  cachedb:string ->
  string ->
  (string, string) result
(** [child ~env ~cachedb part] is [finish (start ~env ~cachedb part)]. *)

val outcome : (string, string) result Windtrap.Testable.t
(** [outcome] tells outcomes apart: two [Ok] by their outputs, and two [Error]
    when one's text holds the other's. *)

val entries : string -> string list
(** [entries cachedb] is the files of the entries of [cachedb]. *)

val damage : string -> (string -> string) -> unit
(** [damage cachedb f] replaces the contents of each entry of [cachedb] by [f]
    of them. *)

val truncated : string -> string
(** [truncated e] is the first half of the entry [e]. *)

val not_a_graph : string -> string
(** [not_a_graph e] is the entry [e] whole, its value's last line blanked, so
    that it does not read as a graph. *)

val of_another_build : string -> string
(** [of_another_build e] is the entry [e] whole, its key's first 32 characters,
    the digest of the library's sources, replaced. *)
