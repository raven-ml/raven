(* Probes of nx_pool.h's threads (thread_probe_stubs.c). *)

(* [burst ()] runs a job on every core, then jobs of two chunks on two threads
   until [burst_stop ()]. *)
external burst : unit -> unit = "probe_burst"
external burst_stop : unit -> unit = "probe_burst_stop"

(* Bodies on a worker *)

(* Whether a worker ran a chunk, and the signals of a table with whether that
   worker blocks them. *)
external worker_mask : unit -> bool * (string * bool) list = "probe_worker_mask"

(* The signals a body may raise itself, in the order of the child's bits. *)
let faults =
  [ "SIGSEGV"; "SIGBUS"; "SIGFPE"; "SIGILL"; "SIGTRAP"; "SIGABRT"; "SIGSYS" ]

(* Children made by fork, right after a job of the parent's, and the values they
   answer. *)
type scenario =
  (* The process's threads before any job, after a job of one thread, after the
     first job of two, after jobs on every core. *)
  | Threads
  (* Whether a worker ran a chunk; the units of a job on every core not run
     once. *)
  | Job
  (* Whether a worker ran a chunk that used 7 MiB of stack. *)
  | Stack
  (* Whether a worker ran a chunk; a bit per signal of [faults] that it raised
     and whose handler ran before [raise] returned. *)
  | Faults

(* [in_child s] is (how the child ended, "exit 0" once it answered; its four
   values). *)
external in_child : scenario -> string * int array = "probe_in_child"
external fork : unit -> unit = "probe_fork"

(* The threads of the process, other than the calling one, that are running now,
   or -1 where the system does not say. *)
external running_threads : unit -> int = "probe_running_threads"
