(* This module provides access to Nx_metal when available, or stubs when not *)

include Nx_core.Backend_intf.S

val is_available : bool
val create_context : unit -> context
