/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Signals as bytes on a pipe. An OCaml handler runs at the runtime's next
   safe point, and a signal that comes after the runtime's last look before a
   blocking call waits for that call to return. This handler writes the
   signal's number on the pipe at once, so a blocked select wakes. */

#define _GNU_SOURCE

#include <errno.h>
#include <signal.h>
#include <unistd.h>

#include <caml/mlvalues.h>
#include <caml/unixsupport.h>

static int wake = -1;

static void note(int s) {
  int saved = errno;
  unsigned char b = (unsigned char)s;
  (void)!write(wake, &b, 1);
  errno = saved;
}

/* [rig_proc_catch w s] has [note] handle signal number [s], writing on [w],
   and is whether [s] was ignored. The flags are the runtime's: the handler
   runs on the alternate stack, since OCaml code may run on a small fiber
   stack, and a blocking call it interrupts fails with EINTR. */
value rig_proc_catch(value w, value s) {
  struct sigaction act, old;
  wake = Int_val(w);
  act.sa_handler = note;
  act.sa_flags = SA_ONSTACK;
  sigemptyset(&act.sa_mask);
  if (sigaction(Int_val(s), &act, &old) != 0)
    caml_uerror("sigaction", Nothing);
  return Val_bool(old.sa_handler == SIG_IGN);
}
