/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* This executable's wait, the system call rig's waits make: select's form
   for sets past FD_SETSIZE on macOS, poll elsewhere. The next one that
   waits on descriptors without a timeout first sends the armed signal, then
   waits at most a second. A signal sent there comes after the OCaml
   runtime's last look at its pending signals, just before the system call
   blocks. */

#define _GNU_SOURCE
#include <caml/mlvalues.h>
#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <poll.h>
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

/* Sends the armed signal, once. */
static int fire(void) {
  if (!armed) return 0;
  armed = 0;
  if (child == 0) raise(SIGTERM);
  else end_child();
  return 1;
}

#if defined(__APPLE__)
#define WAIT_NAME "select$DARWIN_EXTSN"
int gap_select(int n, fd_set *r, fd_set *w, fd_set *e, struct timeval *t)
    __asm("_" WAIT_NAME);

int gap_select(int n, fd_set *r, fd_set *w, fd_set *e, struct timeval *t) {
  typedef int (*select_fn)(int, fd_set *, fd_set *, fd_set *,
                           struct timeval *);
  select_fn real = (select_fn)dlsym(RTLD_NEXT, WAIT_NAME);
  if (real == NULL) fail("dlsym select");
  if (r == NULL || t != NULL || !fire()) return real(n, r, w, e, t);
  struct timeval bound = {1, 0};
  int rc = real(n, r, w, e, &bound);
  blocked = rc == 0;
  return rc;
}
#else
int poll(struct pollfd *fds, nfds_t n, int timeout) {
  typedef int (*poll_fn)(struct pollfd *, nfds_t, int);
  poll_fn real = (poll_fn)dlsym(RTLD_NEXT, "poll");
  if (real == NULL) fail("dlsym poll");
  if (timeout >= 0 || !fire()) return real(fds, n, timeout);
  int rc = real(fds, n, 1000);
  blocked = rc == 0;
  return rc;
}
#endif
