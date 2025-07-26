(** OxCaml backend for Nx - high-performance tensor operations using unboxed types *)

(** {1 Backend Interface} *)

(** This backend implements the Nx backend interface using OxCaml's unboxed
    types for improved performance. It leverages float#, int32#, int64# and
    other unboxed representations to reduce allocation overhead and improve
    cache locality. *)

include Nx_core.Backend_intf.S

(** {2 Additional Functions} *)

(** Create a new execution context.
    @param n_threads Number of threads for parallel execution (default: 8) *)
val create_context : ?n_threads:int -> unit -> context