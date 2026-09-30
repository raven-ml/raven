(* Metal exists on macOS alone: no device is a Metal device. *)

let claims _ = false
let queues _ _ _ = None
