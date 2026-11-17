(** The tensor cores the suites' schedules are given. *)

val host : (Tolk.Ops.device * Tolk.Tc.t list) list
(** [host] is the host's tensor cores, as its renderer's kernels take them
    ({!Tolk.Heuristic.val-cores}): none, so a schedule given them keeps every
    kernel a schedule of no cores makes. The suites give them to schedules
    of any device: a device they do not list takes none. *)
