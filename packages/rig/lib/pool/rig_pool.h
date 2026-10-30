/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's worker threads.

   The pool is the set of threads that run work on the host for one
   process. It holds two sets of workers: compute workers, which nx's CPU
   kernels and the host's launches in rig share, and blocking workers, for
   jobs whose bodies wait on the system, such as rig.disk's long reads. A
   caller hands the pool a job with rig_pool_run, or rig_pool_run_blocking
   for a blocking job, which runs it on the calling thread and on the
   workers of its kind.

   A job is a range of units [0, total) cut into contiguous chunks, run on
   at most a given number of threads, which call the job's body on the
   chunks they claim. The calling thread is worker 0, and the pool's
   workers take the other indices. Threads claim chunks as they go, so a
   thread that finishes early runs chunks another would have run. A job
   returns once every chunk has run. Only one job of each kind runs on
   several threads at a time (see Jobs and Blocking jobs).

   Workers block every signal except those a body raises itself (SIGSEGV,
   SIGBUS, SIGFPE, SIGILL, SIGTRAP, SIGABRT, SIGSYS). A signal sent to the
   process therefore reaches one of the program's own threads, and a fault
   in a body is delivered on the thread that runs it. A profiler that
   samples threads by SIGPROF never samples a worker.

   The life of the pool goes like this:

     1. The first job of a kind on several threads makes that kind's
        workers.
     2. Each job calls on the workers it needs. A worker that waits for a
        job spins for a moment before it sleeps (see Jobs).
     3. The workers live until the process exits. fork waits for a running
        compute job on several threads to end, but not for a blocking job
        (see Blocking jobs), and the child makes workers of its own at its
        first job of each kind on several threads.

   No function fails. A pool that cannot make every worker runs its jobs on
   those it made, down to the calling thread alone. Every function may be
   called from any thread, concurrently with others.

   References. The counts of cores come from sched_getaffinity(2), the
   cpu.max files of the Linux kernel's cgroup v2
   (Documentation/admin-guide/cgroup-v2.rst), the cpu_capacity and
   acpi_cppc/highest_perf files of its CPUs
   (Documentation/ABI/testing/sysfs-devices-system-cpu, and
   Documentation/admin-guide/acpi/cppc_sysfs.rst), from sysctl(3) on macOS,
   and from GetActiveProcessorCount on Windows. */

#ifndef RIG_POOL_H
#define RIG_POOL_H

#include <stddef.h>
#include <stdint.h>

/* A cache line on Apple silicon, and the pair of lines x86 fetches
   together: memory that threads write apart is split on multiples of it. */
#define RIG_LINE_BYTES 128

/* Cores

   A caller sizes a job by two counts of the host's cores: the cores the
   process may occupy, which bound the threads of every job, and the
   performance cores, those that run compute-bound work at more than half
   the speed of the fastest. A job of compute-bound chunks of equal size is
   sized by the performance cores, since a chunk that a slower core claims
   takes longer and delays the end of the job.

   Both counts are taken at the first call of either function and stay the
   same for the life of the process, so a later change of affinity or quota
   is not seen. */

/* rig_pool_cores () is the number of cores the process may occupy, the most
   threads a job runs on. 1 <= rig_pool_cores ().

   On Linux it is min (a, ceil q). a is the number of CPUs in the affinity
   mask of the thread that makes the first call, or of the online CPUs if
   the mask cannot be read. q is the smallest quota / period of the
   process's cgroup and its ancestors, from cgroup v2's cpu.max, and
   without a quota the count is a. cgroup v1's quotas are not read. On
   macOS it is the physical cores, on Windows the active processors, and
   elsewhere the online CPUs. */
int rig_pool_cores(void);

/* rig_pool_performance_cores () is the number of the cores the process may
   occupy that run compute-bound work at more than half the speed of the
   fastest. 1 <= rig_pool_performance_cores () <= rig_pool_cores ().

   On macOS it is the performance cores (hw.perflevel0.physicalcpu) where the
   host reports them. On Linux it is at most rig_pool_cores (), and counts
   the CPUs of the affinity mask whose rating is more than half the largest
   of theirs: the cpu_capacity the kernel reports for each of them, or else
   the highest performance ACPI CPPC reports for each (acpi_cppc/
   highest_perf). Elsewhere, and on a host that reports neither, it is
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

   A thread claims a run of consecutive chunks at a time and runs it in one
   call. A run is a fraction of the chunks left, and one chunk where they
   are few, so a job of many short chunks costs few claims. A job of at most
   15 t chunks makes one call a chunk, so a costly chunk holds up only the
   thread that claimed it. A thread that finishes early runs chunks another
   would have run, so a thread slowed or taken off its core delays the end
   of the job by about the call it is in. Which thread runs which chunks,
   and in which order, is unspecified.

   Only one compute job runs on several threads at a time. A job of t = 1,
   or a job of either kind begun from a body of a job on several threads,
   runs immediately on the calling thread alone, in one call as worker 0.
   Any other compute job first waits for another thread's compute job to
   end.

   A thread that waits, for a job or for the end of its own, spins for a
   moment before it sleeps, and keeps its core while it spins. Jobs that
   follow each other closely therefore start without a system call, an
   idle pool takes no core, and other programs get the spinning cores only
   when the system preempts a spinning thread. A thread that the system
   takes off its core inside a job, while every core is busy, can end the
   job late by up to a spin's length, once the job's other threads stop
   spinning. */

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

/* Blocking jobs

   A blocking job is one whose bodies spend their time waiting on the
   system, such as reads of a file that storage serves. Its length is the
   system's, not the cores': a compute job that waited for it, or that had
   to do without the workers it holds, would wait on the system too. So
   blocking jobs run on workers of their own, apart from the compute
   workers and as many, and a blocking job and a compute job never wait for
   each other. Their threads share the cores: a blocking job whose bodies
   copy memory, such as reads of a file the system holds in memory, slows a
   compute job that runs at the same time, as a program's other threads do.

   A blocking job follows the rules of Jobs, with one difference. A blocking
   job begun while another thread's blocking job runs on several threads
   runs immediately on the calling thread alone, in one call as worker 0,
   since waiting for a job whose end the system decides could take any
   time. For the same reason fork does not wait for a running blocking
   job. */

/* rig_pool_run_blocking (threads, total, chunks, body, ctx) is
   rig_pool_run (threads, total, chunks, body, ctx) for a blocking job (see
   Blocking jobs). */
void rig_pool_run_blocking(int threads, int64_t total, int64_t chunks,
                           rig_pool_body body, void *ctx);

/* For the pool's suite: the Linux counts behind rig_pool_cores () and
   rig_pool_performance_cores (), read from any tree of files. Each function
   reads the files of a Linux host under the directory root, and a root of
   "" reads the host's own. The pool calls them on the host's files, and the
   suite on trees it writes. */

/* rig_pool_cgroup_cpus (root) is ceil q of rig_pool_cores (), read under
   root from cgroup v2. The process's cgroup is the path of the line
   "0::PATH" in root/proc/self/cgroup, found below the cgroup2 mount of
   root/proc/self/mountinfo that holds it, and the cpu.max files read are
   those of that cgroup and its ancestors up to the mount. It is -1 without
   a quota on those paths, or without cgroup v2. */
long rig_pool_cgroup_cpus(const char *root);

/* rig_pool_performance_cpus (root, cpus, n) is the number of the [n] CPUs
   [cpus] that are performance cores, read under root. Where each of them
   has a root/sys/devices/system/cpu/cpuN/cpu_capacity, they are those of
   capacity more than half the largest of theirs. Otherwise, where each has
   a root/sys/devices/system/cpu/cpuN/acpi_cppc/highest_perf, they are
   those of highest performance more than half the largest of theirs.
   Otherwise it is -1. */
long rig_pool_performance_cpus(const char *root, const int *cpus, long n);

/* For the pool's suite: hook points and reads of the compute pool's state,
   with which the suite holds an interleaving of the protocol of rig_pool.c.

   A hook point is a named place in that protocol. The thread that reaches
   it calls the hook set with rig_pool_set_hook, with its point, on any job
   of either kind, holding whatever the pool holds there. Without a hook,
   the default, a point costs a load and a branch.

   rig_pool_set_hook and rig_pool_set_calloc store a pointer that the
   pool's threads read without ordering. Set each before the first job on
   several threads, which makes the workers, and never change it after: a
   thread in a job may run a point with the pointer it read before. */

/* The bit of rig_pool_inside () set while the job is closed. */
#define RIG_POOL_CLOSED ((uint64_t)1 << 63)

/* The hook points of rig_pool.c. A worker is a thread of the pool, never
   the caller. */
typedef enum {
  RIG_POOL_AT_ENTER_OPEN,    /* a worker read the job open, before it adds
                                itself to the threads inside */
  RIG_POOL_AT_ENTER_REFUSED, /* that add found the job closed, before its
                                undo */
  RIG_POOL_AT_LEFT,          /* a worker's leave, or that undo, landed */
  RIG_POOL_AT_ENTERED,       /* a worker entered a job, before it claims */
  RIG_POOL_AT_CHUNK,         /* a thread claimed a run of chunks, before it
                                calls the body on it */
  RIG_POOL_AT_PARK,          /* a worker decided to park, before it takes
                                the parking mutex */
  RIG_POOL_AT_PARKING,       /* a worker set its parked bit, holding the
                                parking mutex, before it reads the job's
                                generation and waits */
  RIG_POOL_AT_OPENED,        /* the caller opened a job, before it stores
                                the job's generation */
  RIG_POOL_AT_PUBLISHED,     /* the caller stored the generation, before it
                                wakes the parked workers */
  RIG_POOL_AT_CLAIMED,       /* the caller's claims found no chunk left,
                                before it closes the job */
  RIG_POOL_AT_CLOSED,        /* the caller closed the job, before it waits
                                for the workers inside */
  RIG_POOL_AT_WAITING,       /* the caller set waiting, holding the parking
                                mutex, before it reads the workers inside
                                and sleeps */
} rig_pool_point;

typedef void (*rig_pool_hook)(rig_pool_point at);

/* rig_pool_set_hook (hook) makes every hook point call hook (at); NULL, the
   default, calls nothing. */
void rig_pool_set_hook(rig_pool_hook hook);

/* rig_pool_set_calloc (f) makes the pool take the memory of each pool it
   makes from f (n, size) in place of calloc (n, size), one call at a time,
   on the thread whose job makes it. f may answer NULL, as calloc does on a
   host out of memory: that pool is then not made, and the job runs on the
   calling thread alone. NULL, the default, restores calloc. */
void rig_pool_set_calloc(void *(*f)(size_t n, size_t size));

/* rig_pool_threads () is the number of threads of the compute pool, its
   workers and the caller, making the pool if it was not; 1 if it cannot be
   made. */
int rig_pool_threads(void);

/* rig_pool_inside () is the number of the compute pool's workers inside its
   job, plus RIG_POOL_CLOSED while the job is closed; 0 without a pool. */
uint64_t rig_pool_inside(void);

/* rig_pool_parked () has bit w set, 1 <= w < 64, while the compute pool's
   worker w has its parked bit, read under the parking mutex: a worker
   whose bit it reads waits for a job, or was woken and has not run since;
   0 without a pool. */
uint64_t rig_pool_parked(void);

#endif
