(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Thumper suites of the benchmark's workloads.

    A suite times talon's answer to each question of each workload whose data
    exists under the directory [$TALON_BENCH_DATA], as the case
    [WORKLOAD/SIZE/QUESTION/talon], such as [groupby/1e7/q03/talon]: the ids of
    the baselines' cases, with talon as the engine. A case loads its workload's
    tables before it is measured, and each measured call runs the question to
    its answer, materialized as one table. *)

val run :
  string ->
  sizes:string list ->
  (string * (data:string -> string -> Workload.t)) list ->
  unit
(** [run suite ~sizes families] is thumper's command line ({!Thumper.run}) over
    the suite [suite] of the workloads [workload ~data size] of each
    [(name, workload)] of [families] and each size of [sizes] whose data exists.
    A workload whose data is missing is skipped, saying so on stderr.

    Exits with an error if [$TALON_BENCH_DATA] is unset. *)
