/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's worker threads.

   The pool is the set of threads that compute on the host for one process.
   nx's CPU kernels and the host's launches in rig share its workers. A
   caller hands the pool a job with rig_pool_run, which runs it on the
   calling thread and on the pool's workers.

   A job is a range of units [0, total) cut into contiguous chunks, run on
   at most a given number of threads, which call the job's body on the
   chunks they claim. The calling thread is worker 0, and the pool's
   workers take the other indices. Each thread has a strip of
   consecutive chunks, which it runs first. A strip is the same in every job
   of the same total, chunks and threads, so a job run again over the same
   memory finds each strip in the caches of the core that ran it. A thread
   that finishes its strip early runs the chunks left in the others'.
   rig_pool_run returns once every chunk has run. Only one job runs on
   several threads at a time (see Jobs).

   Workers block every signal except those a body raises itself (SIGSEGV,
   SIGBUS, SIGFPE, SIGILL, SIGTRAP, SIGABRT, SIGSYS). A signal sent to the
   process therefore reaches one of the program's own threads, and a fault
   in a body is delivered on the thread that runs it. A profiler that
   samples threads by SIGPROF never samples a worker.

   The life of the pool goes like this:

     1. The first job on several threads makes the workers, one fewer
        than rig_pool_cores ().
     2. Each job calls on the workers it needs. A worker that waits for a
        job spins for up to 100 us before it sleeps (see Jobs).
     3. The workers live until the process exits. fork waits for a running
        job on several threads to end, and the child makes a pool of
        its own at its first job on several threads.

   No function fails. A pool that cannot make every worker runs its jobs on
   those it made, down to the calling thread alone. Every function may be
   called from any thread, concurrently with others.

   References. The counts of cores come from sched_getaffinity(2), the
   cpu.max files of the Linux kernel's cgroup v2
   (Documentation/admin-guide/cgroup-v2.rst), the cpu.cfs_quota_us and
   cpu.cfs_period_us files of its cgroup v1 (Documentation/scheduler/
   sched-bwc.rst), the cpu_capacity files of its CPUs
   (Documentation/ABI/testing/sysfs-devices-system-cpu) and the CPU list of
   the cpu_core PMU it registers on Intel hybrid CPUs
   (/sys/devices/cpu_core/cpus), from sysctl(3) on macOS, and from
   GetActiveProcessorCount on Windows. */

#ifndef RIG_POOL_H
#define RIG_POOL_H

#include <stdint.h>

/* Cores

   A caller sizes a job by two counts of the host's cores: the cores the
   process may occupy, which bound the threads of every job, and the
   performance cores, those the host classes as fast where its cores run at
   several speeds. A job of compute-bound chunks of equal size is sized by
   the performance cores, since a chunk that a slower core claims takes
   longer and delays the end of the job.

   Both counts are taken at the first call of either function and stay the
   same for the life of the process, so a later change of affinity or quota
   is not seen. */

/* rig_pool_cores () is the number of cores the process may occupy, the most
   threads a job runs on. 1 <= rig_pool_cores ().

   On Linux it is min (a, ceil q). a is the number of CPUs in the affinity
   mask of the thread that makes the first call, or of the online CPUs if
   the mask cannot be read. q is the smallest quota / period of the
   process's cgroups and their ancestors, from cgroup v2's cpu.max and
   cgroup v1's cpu.cfs_quota_us and cpu.cfs_period_us, and without a quota
   the count is a. On macOS it is the physical cores, on Windows the active
   processors, and elsewhere the online CPUs. */
int rig_pool_cores(void);

/* rig_pool_performance_cores () is the number of the performance cores the
   process may occupy. 1 <= rig_pool_performance_cores () <= rig_pool_cores ().

   On macOS it is the performance cores (hw.perflevel0.physicalcpu) where the
   host reports them. On Linux it is at most rig_pool_cores (), and counts
   CPUs of the affinity mask: where the kernel reports each one's
   cpu_capacity, those whose capacity is more than half the largest of
   theirs; otherwise, where it lists an Intel hybrid CPU's performance cores
   in /sys/devices/cpu_core/cpus, those in the list, or every one if none
   is. Elsewhere, and on a host that reports none of these, it is
   rig_pool_cores (). */
int rig_pool_performance_cores(void);

/* Jobs

   A job of total >= 1 units has c chunks and runs on at most t threads,
   the arguments of rig_pool_run bounded:

     c = min (max (chunks, 1), total)
     t = min (max (threads, 1), rig_pool_cores (), c)

   Chunk i, for 0 <= i < c, is exactly

     [lo, hi) = [floor (i * total / c), floor ((i + 1) * total / c))

   for every total and c. The chunks partition [0, total), none is empty,
   and floor (total / c) <= hi - lo <= ceil (total / c).

   On t > 1 threads, worker w's strip is the chunks

     [floor (w * c / t), floor ((w + 1) * c / t))

   none empty, where t counts only the workers the pool made. A thread
   claims its own strip's chunks in index order, then those left in each
   other strip in turn, from worker w + 1's on, in index order within each.
   A thread that never comes to the job leaves its whole strip to the
   others.

   A thread claims a run of consecutive chunks of a strip at a time and
   runs it in one call. A run is about an eighth of the chunks left in the
   strip, and at most max (1, floor (s / 8)) chunks, where s is the number
   of chunks in the strip. A job of many short chunks then costs few
   claims, and a slow thread still leaves most of its strip to faster ones.
   A job of at most 15 t chunks makes one call a chunk.

   Only one job runs on several threads at a time. A job of t = 1, or a job
   begun from a body of a job on several threads, runs immediately on the
   calling thread alone, in one call as worker 0. Any other job first waits
   for another thread's job to end.

   A thread that waits, for a job or for the end of its own, spins for up
   to 100 us before it sleeps, and keeps its core while it spins. Jobs that
   follow each other closely therefore start without a system call, and
   other programs get those cores only when the system preempts a spinning
   thread. A thread that the system takes off its core inside a job, while
   every core is busy, can end the job up to 100 us late, once the job's
   other threads stop spinning. */

/* The type for the function a job calls on each of its ranges. [lo, hi)
   are the range's units, [worker] the index of the thread that runs it, and
   [ctx] the job's context.

   0 <= worker < t <= max (threads, 1), and the calling thread's worker is
   0. A thread keeps its index for the whole job and claims chunks only
   after its last call has returned, so two calls with the same worker never
   overlap and scratch indexed by worker is never shared. An index may run
   no chunk, so per-worker partials start at their identity.

   Calls run in parallel, or one after another on one thread, in any
   interleaving, so a body must not wait for another call. A body must not
   call the OCaml runtime, since a worker is not an OCaml thread, and must
   not fork. A body may begin a job of its own (see Jobs). A body on a
   worker has 8 MiB of stack, backed by memory as it is touched. On the
   calling thread it has the caller's stack. */
typedef void (*rig_pool_body)(int64_t lo, int64_t hi, int worker, void *ctx);

/* rig_pool_run (threads, total, chunks, body, ctx) runs the job [0, total)
   cut into [chunks] chunks on at most [threads] threads (see Jobs). It
   calls body (lo, hi, worker, ctx) on disjoint ranges [lo, hi) that cover
   [0, total), each made of whole consecutive chunks, and returns once every
   call has returned. A job of total <= 0 calls nothing, and a job that runs
   on one thread makes one call, over [0, total).

   Writes the caller made before the call are visible to every body, and the
   bodies' writes are visible to the caller once it returns.

   A caller should release the OCaml runtime around a job of t > 1. Holding
   it keeps every domain of the program from collecting until the job ends,
   including the time the job waits for another thread's job. */
void rig_pool_run(int threads, int64_t total, int64_t chunks, rig_pool_body body,
                 void *ctx);

#endif
