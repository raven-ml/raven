/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The cgroup v2 bound of rig_pool_cores (), read from any tree of files. Not
   installed: the pool calls it on the host's files, and the pool's suite on
   trees it writes. */

#ifndef RIG_POOL_CGROUP_H
#define RIG_POOL_CGROUP_H

/* rig_pool_cgroup_cpus (root) is ceil q of rig_pool_cores (), read under the
   directory [root]: the cgroup of root/proc/self/cgroup's "0::" line, found
   below the cgroup2 mount of root/proc/self/mountinfo that holds it, and
   the cpu.max files of that cgroup and its ancestors up to the mount, each
   under root. It is -1 without a quota on that path, or without a cgroup
   v2. A root of "" reads the host's files. */
long rig_pool_cgroup_cpus(const char *root);

#endif
