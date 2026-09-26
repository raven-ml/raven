(** Bounded synchronous compilation batches. *)

val map : ('a -> 'b) -> 'a list -> 'b array
(** [map f tasks] runs [f] under the caller's context and returns results in
    input order. The first parallel batch fixes the shared admission limit
    from [PARALLEL]. Later positive settings reuse that limit; [0] runs inline.
    Nested batches execute inline. A failure stops pending tasks; all started tasks finish
    before the exception and its backtrace propagate. Native calls are not
    interrupted. *)
