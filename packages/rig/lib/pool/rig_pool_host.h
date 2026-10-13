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
   root. It reads cgroup v2 and the cgroup v1 hierarchy that holds the cpu
   controller. In each, the process's cgroup is the path of the hierarchy's
   line in root/proc/self/cgroup, found below the mount of
   root/proc/self/mountinfo that holds it, and the quota files read are
   those of that cgroup and its ancestors up to the mount. It is -1 without
   a quota on those paths, or without either hierarchy. */
long rig_pool_cgroup_cpus(const char *root);

/* rig_pool_performance_cpus (root, cpus, n) is the number of the [n] CPUs
   [cpus] that are performance cores, read under root. Where each of them
   has a root/sys/devices/system/cpu/cpuN/cpu_capacity, they are those of
   capacity more than half the largest of theirs. Otherwise, where the CPU
   list root/sys/devices/cpu_core/cpus reads, they are those in the list,
   or all of [cpus] if none is. Otherwise it is -1. */
long rig_pool_performance_cpus(const char *root, const int *cpus, long n);

#endif
