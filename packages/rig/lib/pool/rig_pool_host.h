/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The Linux counts behind rig_pool_cores () and
   rig_pool_performance_cores (), read from any tree of files.

   Each function reads the files of a Linux host under the directory root,
   and a root of "" reads the host's own. The pool calls them on the host's
   files, and the pool's suite on trees it writes. This header is not
   installed. */

#ifndef RIG_POOL_HOST_H
#define RIG_POOL_HOST_H

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

#endif
