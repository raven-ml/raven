(** What the suites' schedules are told of each device's kernels. *)

val host : (Tolk.Ops.device * Tolk.Heuristic.cores) list
(** [host] is what the host's renderer tells of its kernels
    ({!Tolk.Heuristic.val-cores}): no tensor core, so a schedule given it
    keeps every kernel a schedule of no cores makes, and points run in turn,
    as its loops run them. The suites give it to schedules of any device: a
    device it does not list takes no core and runs its points in turn. *)

val at_once : (Tolk.Ops.device * Tolk.Heuristic.cores) list
(** [at_once] is {!host} on a device taken to run its kernels' points at
    once, as a GPU's threads. *)
