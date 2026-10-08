(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A device's ring, by hand *)

type ring

external ring : int -> ring = "device_metal_test_ring"
external commit : ring -> last:bool -> int = "device_metal_test_commit"

external complete : ring -> int -> failed:bool -> unit
  = "device_metal_test_complete"

external defer : ring -> int = "device_metal_test_defer"
external ran : ring -> int array = "device_metal_test_ran"
external word : ring -> int = "device_metal_test_word"
external times : ring -> int -> int * int = "device_metal_test_times"
external failure : ring -> string option = "device_metal_test_failure"
external sleep : ring -> string option = "device_metal_test_sleep"
external stop : ring -> bool = "device_metal_test_stop"
