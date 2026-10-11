/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* This executable's select: the next one that waits on descriptors without
   a timeout first sends the armed signal, then waits at most [bound]. A
   signal sent there comes after the OCaml runtime's last look at its pending
   signals, just before the system call blocks. */

#define _GNU_SOURCE
#include <caml/mlvalues.h>
#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <unistd.h>

static pid_t child;
static int armed;
static int blocked;

static void fail(const char *what) {
  perror(what);
  exit(2);
}

/* Ends [child] and returns once it is waitable, leaving it unreaped: its
   SIGCHLD has then been handled. */
static void end_child(void) {
  siginfo_t info;
  int rc;
  if (kill(child, SIGKILL) == -1) fail("kill");
  do rc = waitid(P_PID, child, &info, WEXITED | WNOWAIT);
  while (rc == -1 && errno == EINTR);
  if (rc == -1) fail("waitid");
}

/* [pid] is the child to end, or 0 for a SIGTERM to this process. */
value gap_arm(value pid) {
  child = Int_val(pid);
  armed = 1;
  return Val_unit;
}

value gap_blocked(value unit) {
  (void)unit;
  return Val_bool(blocked);
}

int select(int n, fd_set *r, fd_set *w, fd_set *e, struct timeval *timeout) {
  typedef int (*select_fn)(int, fd_set *, fd_set *, fd_set *,
                           struct timeval *);
  select_fn real = (select_fn)dlsym(RTLD_NEXT, "select");
  if (real == NULL) fail("dlsym select");
  if (!armed || r == NULL || timeout != NULL) return real(n, r, w, e, timeout);
  armed = 0;
  if (child == 0) raise(SIGTERM);
  else end_child();
  struct timeval bound = {1, 0};
  int rc = real(n, r, w, e, &bound);
  blocked = rc == 0;
  return rc;
}
