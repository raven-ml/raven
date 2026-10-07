(* Probes of nx_pool.h (pool_probe_stubs.c). *)

(* Recorded jobs *)

type call = { lo : int64; hi : int64; worker : int; thread : int }

(* [calls] in the order they began; [thread] numbers the threads in order of
   their first call, the caller's 0. [count] is every call, [overlaps] the calls
   that began while another of their worker ran. *)
type job = { calls : call list; count : int; overlaps : int }

external record_raw :
  int -> int64 -> int64 -> (int64 * int64 * int * int) array * int * int
  = "probe_record"

let record ~threads ~total ~chunks =
  let calls, count, overlaps = record_raw threads total chunks in
  let call (lo, hi, worker, thread) = { lo; hi; worker; thread } in
  { calls = List.map call (Array.to_list calls); count; overlaps }

(* [visibility ~jobs ~threads] is (values bodies read stale, values the caller
   read stale) over [jobs] jobs. *)
external visibility : jobs:int -> threads:int -> int * int = "probe_visibility"

(* [nested ~threads ~outer ~inner] is (outer calls, inner calls, inner calls off
   their body's thread or not worker 0, inner units not run once). *)
external nested : threads:int -> outer:int -> inner:int -> int * int * int * int
  = "probe_nested"

(* [balance chunks] is whether chunks 1 to [chunks - 1] of a job on two threads
   ran while chunk 0 lasted. *)
external balance : int -> bool = "probe_balance"

(* Jobs observed from another domain *)

external reset : unit -> unit = "probe_reset"

(* [hold ~only_worker] runs a job of two chunks on two threads whose chunks, or
   with [only_worker] the worker's once both began, wait for [hold_release]. It
   is whether they were released before patience ran out. *)
external hold : only_worker:bool -> bool = "probe_hold"
external hold_arrived : unit -> int = "probe_hold_arrived"
external hold_release : unit -> unit = "probe_hold_release"

external counted : threads:int -> total:int64 -> chunks:int64 -> unit
  = "probe_counted"

external counted_calls : unit -> int = "probe_counted_calls"

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

(* The host *)

external cores : unit -> int = "probe_cores"
external performance_cores : unit -> int = "probe_performance_cores"
external system : unit -> string = "probe_system"
external sysctl : string -> int = "probe_sysctl"
external pinned_cores : unit -> int * int = "probe_pinned_cores"
external running_threads : unit -> int = "probe_running_threads"
