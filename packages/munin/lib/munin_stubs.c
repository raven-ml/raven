/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What OCaml's Unix library lacks for a store: whole-file locks that belong
   to an open file description, a sync that reaches the storage on macOS,
   and the kernel's boot and machine identities.

   flock(2): "Locks created by flock() are associated with an open file
   description", so another descriptor's close does not release them, which
   POSIX record locks (lockf, fcntl) do. On Windows the lock is LockFileEx
   on one byte at offset 2^62, past any data: "Locking a region that goes
   beyond the current end-of-file position is not an error". BLAKE2b is in
   munin_blake2_stubs.c, which needs the runtime's internals that the macOS
   system headers here clash with. */

/* -std=c11 hides what C11 does not name, and the -pthread in OCaml's flags
   brings back only POSIX.1-1995 on glibc (as _REENTRANT) and nothing on
   musl. flock is BSD and clock_gettime POSIX: _DEFAULT_SOURCE asks both
   libraries for them. It precedes every header, since the first one
   included fixes what all of them declare. */
#define _DEFAULT_SOURCE

#if defined(_WIN32)
/* caml/unixsupport.h includes winsock2.h, which must precede windows.h. */
#include <winsock2.h>
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <sys/sysctl.h>
#endif

#include <string.h>
#include <time.h>

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>
#include <caml/unixsupport.h>

/* caml/unixsupport.h declares the unix library's functions without
   CAMLextern: a stub DLL loaded at run time on Windows would call them
   directly, which flexdll links within 2 GiB only. Declared again with
   CAMLextern, each call goes through a pointer. */
#ifdef _WIN32
CAMLnoret CAMLextern void caml_uerror(const char *cmdname, value arg);
CAMLextern void caml_win32_maperr(DWORD errcode);
#endif

CAMLprim value caml_munin_getpid(value unit) {
  (void)unit;
#if defined(_WIN32)
  return Val_long(GetCurrentProcessId());
#else
  return Val_long(getpid());
#endif
}

/* [monotonic ()] is seconds on a clock that never steps, from an arbitrary
   origin: CLOCK_MONOTONIC, or Windows's performance counter. */
CAMLprim value caml_munin_monotonic(value unit) {
  (void)unit;
#if defined(_WIN32)
  LARGE_INTEGER now, freq;
  QueryPerformanceCounter(&now);
  QueryPerformanceFrequency(&freq);
  return caml_copy_double((double)now.QuadPart / (double)freq.QuadPart);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return caml_copy_double((double)ts.tv_sec + (double)ts.tv_nsec / 1e9);
#endif
}

/* Locks */

/* [kind]: 0 exclusive, 1 shared, 2 unlock. [wait]: block until granted.
   Returns true if granted, false if another holder refused it without
   waiting. Raises Unix_error otherwise. */
CAMLprim value caml_munin_lock(value fd, value kind, value wait) {
  CAMLparam3(fd, kind, wait);
  int k = Int_val(kind);
  int w = Bool_val(wait);
#if defined(_WIN32)
  HANDLE h = Handle_val(fd);
  OVERLAPPED ov;
  memset(&ov, 0, sizeof ov);
  ov.Offset = 0;
  ov.OffsetHigh = 0x40000000; /* 2^62 */
  BOOL ok;
  if (k == 2) {
    ok = UnlockFileEx(h, 0, 1, 0, &ov);
    if (!ok) {
      win32_maperr(GetLastError());
      caml_uerror("UnlockFileEx", Nothing);
    }
    CAMLreturn(Val_true);
  }
  DWORD flags = (k == 0 ? LOCKFILE_EXCLUSIVE_LOCK : 0) |
                (w ? 0 : LOCKFILE_FAIL_IMMEDIATELY);
  caml_enter_blocking_section();
  ok = LockFileEx(h, flags, 0, 1, 0, &ov);
  caml_leave_blocking_section();
  if (ok) CAMLreturn(Val_true);
  DWORD err = GetLastError();
  if (err == ERROR_LOCK_VIOLATION || err == ERROR_IO_PENDING)
    CAMLreturn(Val_false);
  win32_maperr(err);
  caml_uerror("LockFileEx", Nothing);
#else
  int op = k == 0 ? LOCK_EX : k == 1 ? LOCK_SH : LOCK_UN;
  if (k != 2 && !w) op |= LOCK_NB;
  int r;
  caml_enter_blocking_section();
  do r = flock(Int_val(fd), op); while (r == -1 && errno == EINTR);
  caml_leave_blocking_section();
  if (r == 0) CAMLreturn(Val_true);
  if (errno == EWOULDBLOCK) CAMLreturn(Val_false);
  caml_uerror("flock", Nothing);
#endif
}

/* Syncing */

/* fsync(2) on macOS "does not guarantee that the data is written to the
   physical media"; F_FULLFSYNC asks the drive to flush its cache. */
CAMLprim value caml_munin_full_fsync(value fd) {
  CAMLparam1(fd);
#if defined(_WIN32)
  BOOL ok;
  caml_enter_blocking_section();
  ok = FlushFileBuffers(Handle_val(fd));
  caml_leave_blocking_section();
  if (!ok) {
    win32_maperr(GetLastError());
    caml_uerror("FlushFileBuffers", Nothing);
  }
#else
  int r;
  caml_enter_blocking_section();
#if defined(__APPLE__)
  r = fcntl(Int_val(fd), F_FULLFSYNC);
  /* A file system that does not support it, such as a network one. */
  if (r == -1) r = fsync(Int_val(fd));
#else
  r = fsync(Int_val(fd));
#endif
  caml_leave_blocking_section();
  if (r == -1) caml_uerror("fsync", Nothing);
#endif
  CAMLreturn(Val_unit);
}

/* Systems */

/* [uname ()] is [(sysname ^ " " ^ release, machine)]. */
CAMLprim value caml_munin_uname(value unit) {
  CAMLparam1(unit);
  CAMLlocal3(res, os, arch);
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetNativeSystemInfo(&si);
  const char *a = "unknown";
  switch (si.wProcessorArchitecture) {
  case PROCESSOR_ARCHITECTURE_AMD64: a = "x86_64"; break;
  case PROCESSOR_ARCHITECTURE_ARM64: a = "arm64"; break;
  case PROCESSOR_ARCHITECTURE_INTEL: a = "x86"; break;
  }
  os = caml_copy_string("Windows");
  arch = caml_copy_string(a);
#else
  struct utsname u;
  if (uname(&u) == -1) caml_uerror("uname", Nothing);
  char buf[sizeof u.sysname + sizeof u.release + 2];
  strcpy(buf, u.sysname);
  strcat(buf, " ");
  strcat(buf, u.release);
  os = caml_copy_string(buf);
  arch = caml_copy_string(u.machine);
#endif
  res = caml_alloc_tuple(2);
  Store_field(res, 0, os);
  Store_field(res, 1, arch);
  CAMLreturn(res);
}

/* [sysctl_string name] is the string sysctl [name] holds, or None. macOS
   alone; elsewhere None. */
CAMLprim value caml_munin_sysctl_string(value name) {
  CAMLparam1(name);
  CAMLlocal1(res);
#if defined(__APPLE__)
  char buf[256];
  size_t len = sizeof buf;
  if (sysctlbyname(String_val(name), buf, &len, NULL, 0) == 0 && len > 0) {
    buf[sizeof buf - 1] = '\0';
    res = caml_alloc_some(caml_copy_string(buf));
    CAMLreturn(res);
  }
#else
  (void)name;
#endif
  CAMLreturn(Val_none);
}

/* [boot_time ()] is the time the kernel booted, in seconds since the Unix
   epoch, as macOS's kern.boottime says; None elsewhere. */
CAMLprim value caml_munin_boot_time(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(res);
#if defined(__APPLE__)
  struct timeval tv;
  size_t len = sizeof tv;
  if (sysctlbyname("kern.boottime", &tv, &len, NULL, 0) == 0) {
    res = caml_alloc_some(
        caml_copy_double((double)tv.tv_sec + (double)tv.tv_usec / 1e6));
    CAMLreturn(res);
  }
#else
  (void)unit;
#endif
  CAMLreturn(Val_none);
}

/* [platform_uuid ()] is IOKit's IOPlatformUUID of the machine, as IOKit
   returns it, on macOS; None elsewhere. */
CAMLprim value caml_munin_platform_uuid(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(res);
#if defined(__APPLE__)
  char buf[128];
  int got = 0;
  io_service_t svc = IOServiceGetMatchingService(
      0, IOServiceMatching("IOPlatformExpertDevice"));
  if (svc) {
    CFTypeRef uuid = IORegistryEntryCreateCFProperty(
        svc, CFSTR("IOPlatformUUID"), kCFAllocatorDefault, 0);
    if (uuid) {
      if (CFGetTypeID(uuid) == CFStringGetTypeID() &&
          CFStringGetCString((CFStringRef)uuid, buf, sizeof buf,
                             kCFStringEncodingUTF8))
        got = 1;
      CFRelease(uuid);
    }
    IOObjectRelease(svc);
  }
  if (got) {
    res = caml_alloc_some(caml_copy_string(buf));
    CAMLreturn(res);
  }
#else
  (void)unit;
#endif
  CAMLreturn(Val_none);
}
