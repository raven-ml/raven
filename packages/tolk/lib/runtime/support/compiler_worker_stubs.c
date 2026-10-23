/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A worker's program in memory, on Linux, and the channels of serving
   workers. */

/* memfd_create and file seals are GNU extensions of the C library. */
#ifdef __linux__
#define _GNU_SOURCE
#endif

/* A process start's attribute list is Windows Vista's. */
#if defined(_WIN32) && (!defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0600)
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#ifdef __linux__
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MFD_EXEC
#define MFD_EXEC 0x0010U
#endif

static void refused(const char *call, int fd) {
  value message = caml_alloc_sprintf("%s: %s", call, strerror(errno));
  if (fd >= 0) close(fd);
  caml_failwith_value(message);
}

/* The program */

/* A file of no file system, so that it runs where the home directory is
   mounted noexec. */

/* A sealed memory file named [v_name] holding [v_program], open
   close-on-exec: the descriptor of the program the path /proc/self/fd/N runs.
   Raises Failure naming the call the kernel refused. */
value caml_tolk_worker_memfd(value v_name, value v_program) {
  CAMLparam2(v_name, v_program);
  unsigned flags = MFD_CLOEXEC | MFD_ALLOW_SEALING;
  /* Kernels from 6.3 on make memory files executable only on request, and
     older ones reject the request. */
  int fd = memfd_create(String_val(v_name), flags | MFD_EXEC);
  if (fd < 0 && errno == EINVAL) fd = memfd_create(String_val(v_name), flags);
  if (fd < 0) refused("memfd_create", -1);
  size_t size = caml_string_length(v_program), done = 0;
  while (done < size) {
    ssize_t n = write(fd, String_val(v_program) + done, size - done);
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) refused("write", fd);
    done += n;
  }
  if (fcntl(fd, F_ADD_SEALS,
            F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE) < 0)
    refused("fcntl", fd);
  CAMLreturn(Val_int(fd));
}
#else
value caml_tolk_worker_memfd(value v_name, value v_program) {
  (void)v_name;
  (void)v_program;
  caml_failwith("memfd_create: Linux only");
}
#endif

/* Channels */

#ifndef _WIN32
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/* A peer that has closed its end raises SIGPIPE in a process that writes to
   it, which ends an OCaml program, and a library installs no handler. Sends
   ask for EPIPE instead: by the flag MSG_NOSIGNAL where the system has it,
   and on macOS also by the socket's option SO_NOSIGPIPE, set when the channel
   is made, since a system older than the SDK ignores the flag. */
#if defined(MSG_NOSIGNAL)
#define NO_SIGPIPE MSG_NOSIGNAL
#elif defined(__APPLE__)
#define NO_SIGPIPE 0
#else
#error "no way to send without SIGPIPE"
#endif

/* [v_fd] if it is above standard input, output and error, else a copy of it
   above them, close-on-exec, and [v_fd] closed: a descriptor among them
   would take the place of the stream it replaces, in this process and in a
   child given it. */
value caml_tolk_worker_above_stdio(value v_fd) {
  int fd = Int_val(v_fd);
  if (fd > 2) return v_fd;
  int above = fcntl(fd, F_DUPFD_CLOEXEC, 3);
  if (above < 0)
    caml_failwith_value(caml_alloc_sprintf("fcntl: %s", strerror(errno)));
  close(fd);
  return Val_int(above);
}

/* Asks the socket [v_fd] for EPIPE rather than SIGPIPE on macOS; elsewhere
   the flag of each send asks. */
value caml_tolk_worker_no_sigpipe(value v_fd) {
#ifdef __APPLE__
  int one = 1;
  if (setsockopt(Int_val(v_fd), SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one))
    caml_failwith_value(
        caml_alloc_sprintf("setsockopt: %s", strerror(errno)));
#else
  (void)v_fd;
#endif
  return Val_unit;
}

/* Sends the bytes of [v_data], all of them, on the socket [v_fd]: false if
   its peer has closed it. Runs without the runtime lock, on a copy. Raises
   Failure naming any other error. */
value caml_tolk_worker_send(value v_fd, value v_data) {
  CAMLparam2(v_fd, v_data);
  int fd = Int_val(v_fd), error = 0;
  size_t size = caml_string_length(v_data), done = 0;
  char *copy = malloc(size > 0 ? size : 1);
  if (copy == NULL) caml_raise_out_of_memory();
  memcpy(copy, String_val(v_data), size);
  caml_release_runtime_system();
  while (error == 0 && done < size) {
    ssize_t n = send(fd, copy + done, size - done, NO_SIGPIPE);
    if (n < 0 && errno == EINTR) continue;
    if (n < 0)
      error = errno;
    else
      done += n;
  }
  caml_acquire_runtime_system();
  free(copy);
  if (error == EPIPE || error == ECONNRESET) CAMLreturn(Val_false);
  if (error != 0)
    caml_failwith_value(caml_alloc_sprintf("send: %s", strerror(error)));
  CAMLreturn(Val_true);
}

value caml_tolk_worker_spawn(value v_program, value v_cmdline, value v_in,
                             value v_out, value v_err) {
  (void)v_program;
  (void)v_cmdline;
  (void)v_in;
  (void)v_out;
  (void)v_err;
  caml_failwith("spawn: Windows only");
}
#else
#include <caml/osdeps.h>
#include <caml/unixsupport.h>
#include <stdlib.h>
#include <windows.h>

/* Windows has no SIGPIPE, and handles are no descriptors of a C library:
   requests go to pipes, which these leave alone. */
value caml_tolk_worker_above_stdio(value v_fd) { return v_fd; }

value caml_tolk_worker_no_sigpipe(value v_fd) {
  (void)v_fd;
  return Val_unit;
}

value caml_tolk_worker_send(value v_fd, value v_data) {
  (void)v_fd;
  (void)v_data;
  caml_failwith("send: Windows writes to its workers' pipes");
}

/* Starts [v_program] with the command line [v_cmdline] and the handles of
   [v_in], [v_out] and [v_err] as its standard input, output and error, and
   gives its process handle, as Unix.create_process does. The process inherits
   those three alone: inheritable copies of them, which its attribute list
   names and which live only for the call. Unix.create_process lets a child
   inherit every inheritable handle of this process, among them the copies
   another start is passing at that moment, so that a worker could hold
   another's pipes open past the other's death. A start of Unix.create_process
   during this call can still inherit this call's copies. Raises Unix_error if
   the process does not start. */
value caml_tolk_worker_spawn(value v_program, value v_cmdline, value v_in,
                             value v_out, value v_err) {
  CAMLparam5(v_program, v_cmdline, v_in, v_out, v_err);
  HANDLE self = GetCurrentProcess();
  HANDLE given[3] = {Handle_val(v_in), Handle_val(v_out), Handle_val(v_err)};
  HANDLE inherited[3] = {NULL, NULL, NULL};
  LPPROC_THREAD_ATTRIBUTE_LIST attributes = NULL;
  SIZE_T size = 0;
  STARTUPINFOEXW startup;
  PROCESS_INFORMATION process;
  DWORD error = ERROR_SUCCESS;
  wchar_t *program = caml_stat_strdup_to_utf16(String_val(v_program));
  wchar_t *cmdline = caml_stat_strdup_to_utf16(String_val(v_cmdline));
  for (int i = 0; i < 3 && error == ERROR_SUCCESS; i++)
    if (!DuplicateHandle(self, given[i], self, &inherited[i], 0, TRUE,
                         DUPLICATE_SAME_ACCESS))
      error = GetLastError();
  if (error == ERROR_SUCCESS) {
    /* The first call gives the list's size and fails by design. */
    InitializeProcThreadAttributeList(NULL, 1, 0, &size);
    if ((attributes = malloc(size)) == NULL)
      error = ERROR_NOT_ENOUGH_MEMORY;
    else if (!InitializeProcThreadAttributeList(attributes, 1, 0, &size)) {
      error = GetLastError();
      free(attributes);
      attributes = NULL;
    } else if (!UpdateProcThreadAttribute(
                   attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited,
                   sizeof inherited, NULL, NULL))
      error = GetLastError();
  }
  if (error == ERROR_SUCCESS) {
    ZeroMemory(&startup, sizeof startup);
    startup.StartupInfo.cb = sizeof startup;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = inherited[0];
    startup.StartupInfo.hStdOutput = inherited[1];
    startup.StartupInfo.hStdError = inherited[2];
    startup.lpAttributeList = attributes;
    if (!CreateProcessW(program, cmdline, NULL, NULL, TRUE,
                        EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW, NULL,
                        NULL, &startup.StartupInfo, &process))
      error = GetLastError();
    else
      CloseHandle(process.hThread);
  }
  caml_stat_free(program);
  caml_stat_free(cmdline);
  if (attributes != NULL) {
    DeleteProcThreadAttributeList(attributes);
    free(attributes);
  }
  for (int i = 0; i < 3; i++)
    if (inherited[i] != NULL) CloseHandle(inherited[i]);
  if (error != ERROR_SUCCESS) {
    caml_win32_maperr(error);
    caml_uerror("CreateProcess", v_program);
  }
  CAMLreturn(Val_long((intnat)process.hProcess));
}
#endif
