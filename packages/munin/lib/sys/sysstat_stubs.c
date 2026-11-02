/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The counters munin.sys reads, per system. Linux reads /proc in OCaml and
   uses the clock alone. A stub that cannot read raises Sys_error. Windows
   links kernel32 alone. */

/* -std=c11 hides what C11 does not name, and the -pthread in OCaml's flags
   brings back only POSIX.1-1995 on glibc (as _REENTRANT) and nothing on
   musl. getloadavg is BSD and clock_gettime POSIX: _DEFAULT_SOURCE asks
   both libraries for them. It precedes every header, since the first one
   included fixes what all of them declare. */
#define _DEFAULT_SOURCE

/* PSAPI_VERSION 2 maps GetProcessMemoryInfo to K32GetProcessMemoryInfo, in
   Windows 7's kernel32, so the stubs link no psapi. Before any header:
   <stdint.h> fixes _WIN32_WINNT. */
#if defined(_WIN32)
#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0601
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#undef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
/* psapi.h and winioctl.h need windows.h's types. */
#include <windows.h>
#include <psapi.h>
#include <wchar.h>
#include <winioctl.h>
#else
#include <time.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/storage/IOBlockStorageDriver.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

CAMLnoret static void fail(const char* what) {
  caml_raise_sys_error(caml_copy_string(what));
}

/* The system: 0 for Linux, 1 for macOS, 2 for Windows, 3 for another. */
CAMLprim value caml_munin_sys_system(value unit) {
  (void)unit;
#if defined(__linux__)
  return Val_int(0);
#elif defined(__APPLE__)
  return Val_int(1);
#elif defined(_WIN32)
  return Val_int(2);
#else
  return Val_int(3);
#endif
}

/* Seconds on a clock that never steps back. */
double caml_munin_sys_now(value unit) {
  (void)unit;
#if defined(_WIN32)
  LARGE_INTEGER count, freq;
  QueryPerformanceCounter(&count);
  QueryPerformanceFrequency(&freq);
  return (double)count.QuadPart / (double)freq.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

CAMLprim value caml_munin_sys_now_byte(value unit) {
  return caml_copy_double(caml_munin_sys_now(unit));
}

#if defined(__APPLE__) || defined(_WIN32)
static value pair(int64_t a, int64_t b) {
  value r = caml_alloc_small(2, 0);
  Field(r, 0) = Val_long(a);
  Field(r, 1) = Val_long(b);
  return r;
}
#endif

#if defined(_WIN32)
static int64_t filetime(FILETIME t) {
  ULARGE_INTEGER u;
  u.LowPart = t.dwLowDateTime;
  u.HighPart = t.dwHighDateTime;
  return (int64_t)u.QuadPart;
}
#endif

/* CPU: (busy, idle) ticks summed over the CPUs. */

CAMLprim value caml_munin_sys_cpu(value unit) {
  (void)unit;
#if defined(__APPLE__)
  /* Each processor's ticks are 32 bits wide: summing them as unsigned values
     keeps a processor's counter from turning negative after 2^31 ticks. */
  natural_t ncpu = 0;
  processor_info_array_t info;
  mach_msg_type_number_t ninfo;
  host_t host = mach_host_self();
  kern_return_t kr =
      host_processor_info(host, PROCESSOR_CPU_LOAD_INFO, &ncpu, &info, &ninfo);
  mach_port_deallocate(mach_task_self(), host);
  if (kr != KERN_SUCCESS) fail("host_processor_info failed");
  int64_t busy = 0, idle = 0;
  for (natural_t i = 0; i < ncpu; i++) {
    integer_t* t = info + CPU_STATE_MAX * i;
    busy += (uint32_t)t[CPU_STATE_USER] + (uint32_t)t[CPU_STATE_NICE] +
            (uint32_t)t[CPU_STATE_SYSTEM];
    idle += (uint32_t)t[CPU_STATE_IDLE];
  }
  vm_deallocate(mach_task_self(), (vm_address_t)info,
                (vm_size_t)(ninfo * sizeof(integer_t)));
  return pair(busy, idle);
#elif defined(_WIN32)
  /* In 100 ns. Kernel time includes idle time. */
  FILETIME idle_ft, kernel_ft, user_ft;
  if (!GetSystemTimes(&idle_ft, &kernel_ft, &user_ft)) {
    fail("GetSystemTimes failed");
  }
  int64_t idle = filetime(idle_ft);
  return pair(filetime(user_ft) + filetime(kernel_ft) - idle, idle);
#else
  fail("no CPU counters on this system");
#endif
}

/* Memory: (total, available, swap used) in bytes. */

CAMLprim value caml_munin_sys_memory(value unit) {
  (void)unit;
  int64_t total = 0, avail = 0, swap = 0;
#if defined(__APPLE__)
  int64_t memsize;
  size_t size = sizeof memsize;
  if (sysctlbyname("hw.memsize", &memsize, &size, NULL, 0) != 0) {
    fail("sysctl hw.memsize failed");
  }
  host_t host = mach_host_self();
  vm_statistics64_data_t vm;
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  kern_return_t kr =
      host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count);
  mach_port_deallocate(mach_task_self(), host);
  if (kr != KERN_SUCCESS) fail("host_statistics64 failed");
  struct xsw_usage sw;
  size = sizeof sw;
  if (sysctlbyname("vm.swapusage", &sw, &size, NULL, 0) != 0) {
    fail("sysctl vm.swapusage failed");
  }
  total = memsize;
  /* Available is free and inactive pages, as psutil's virtual_memory
     reads them (avail = inactive + free). Purgeable pages sit in the
     active and inactive queues, so adding them would count some twice.
     The counts are of the kernel's pages, larger than this process's
     (vm_page_size) in a process translated from x86_64. */
  avail = ((int64_t)vm.free_count + (int64_t)vm.inactive_count) *
          (int64_t)vm_kernel_page_size;
  swap = (int64_t)sw.xsu_used;
#elif defined(_WIN32)
  /* The commit charge beyond physical memory stands in for swap. */
  MEMORYSTATUSEX s;
  s.dwLength = sizeof s;
  if (!GlobalMemoryStatusEx(&s)) fail("GlobalMemoryStatusEx failed");
  total = (int64_t)s.ullTotalPhys;
  avail = (int64_t)s.ullAvailPhys;
  int64_t commit_total = (int64_t)s.ullTotalPageFile;
  int64_t commit_used = commit_total - (int64_t)s.ullAvailPageFile;
  int64_t swap_total = commit_total > total ? commit_total - total : 0;
  swap = commit_used - (total - avail);
  if (swap < 0) swap = 0;
  if (swap > swap_total) swap = swap_total;
#else
  fail("no memory counters on this system");
#endif
  value r = caml_alloc_small(3, 0);
  Field(r, 0) = Val_long(total);
  Field(r, 1) = Val_long(avail);
  Field(r, 2) = Val_long(swap);
  return r;
}

/* The one-minute load average and the number of logical CPUs online. */

CAMLprim value caml_munin_sys_load(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(r);
#if defined(_WIN32)
  fail("Windows keeps no load average");
#else
  double load;
  if (getloadavg(&load, 1) != 1) fail("getloadavg failed");
  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpu < 1) fail("sysconf _SC_NPROCESSORS_ONLN failed");
  r = caml_alloc_tuple(2);
  Store_field(r, 0, caml_copy_double(load));
  Store_field(r, 1, Val_long(ncpu));
#endif
  CAMLreturn(r);
}

/* This process's resident memory in bytes. */

CAMLprim value caml_munin_sys_rss(value unit) {
  (void)unit;
#if defined(__APPLE__)
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                &count) != KERN_SUCCESS) {
    fail("task_info failed");
  }
  return Val_long((int64_t)info.resident_size);
#elif defined(_WIN32)
  /* The working set is what is resident. */
  PROCESS_MEMORY_COUNTERS pmc;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
    fail("GetProcessMemoryInfo failed");
  }
  return Val_long((int64_t)pmc.WorkingSetSize);
#else
  fail("no process memory counters on this system");
#endif
}

/* Disks: an array of (name, bytes read, bytes written), one per physical
   disk that gives its counters. */

#if defined(_WIN32)
/* [disk_perf h] is the drive [h]'s counters, in a buffer the caller frees, or
   NULL. A driver may return a structure larger than DISK_PERFORMANCE: the
   buffer grows while it answers ERROR_INSUFFICIENT_BUFFER or
   ERROR_MORE_DATA. */
static DISK_PERFORMANCE* disk_perf(HANDLE h) {
  DWORD size = sizeof(DISK_PERFORMANCE);
  for (;;) {
    DISK_PERFORMANCE* p = malloc(size);
    if (p == NULL) return NULL;
    DWORD got;
    if (DeviceIoControl(h, IOCTL_DISK_PERFORMANCE, NULL, 0, p, size, &got,
                        NULL)) {
      return p;
    }
    DWORD err = GetLastError();
    free(p);
    if (err != ERROR_INSUFFICIENT_BUFFER && err != ERROR_MORE_DATA) return NULL;
    if (size > MAXDWORD / 2) return NULL;
    size *= 2;
  }
}

/* [dos_names ()] is the list of MS-DOS device names, each ended by a NUL,
   then an empty string, in a buffer the caller frees, or NULL. */
static wchar_t* dos_names(void) {
  DWORD cap = 1 << 14;
  for (;;) {
    wchar_t* names = malloc(cap * sizeof *names);
    if (names == NULL) return NULL;
    if (QueryDosDeviceW(NULL, names, cap) != 0) return names;
    DWORD err = GetLastError();
    free(names);
    if (err != ERROR_INSUFFICIENT_BUFFER || cap > MAXDWORD / 2) return NULL;
    cap *= 2;
  }
}
#endif

struct disk {
  char name[64];
  int64_t read;
  int64_t written;
};

/* A growing table of disks. */
struct disks {
  struct disk* d;
  int n;
  int cap;
};

#if defined(__APPLE__) || defined(_WIN32)
/* [next t] is a fresh entry at the end of [t], or NULL if memory ran out. */
static struct disk* next(struct disks* t) {
  if (t->n == t->cap) {
    int cap = t->cap == 0 ? 8 : 2 * t->cap;
    struct disk* d = realloc(t->d, cap * sizeof *d);
    if (d == NULL) return NULL;
    t->d = d;
    t->cap = cap;
  }
  return &t->d[t->n];
}
#endif

static value disk_array(struct disks* t) {
  CAMLparam0();
  CAMLlocal3(r, row, name);
  r = caml_alloc_tuple(t->n);
  for (int i = 0; i < t->n; i++) {
    name = caml_copy_string(t->d[i].name);
    row = caml_alloc_tuple(3);
    Store_field(row, 0, name);
    Store_field(row, 1, Val_long(t->d[i].read));
    Store_field(row, 2, Val_long(t->d[i].written));
    Store_field(r, i, row);
  }
  CAMLreturn(r);
}

#ifdef __APPLE__
static int64_t stat_of(CFDictionaryRef stats, CFStringRef key) {
  CFNumberRef n = CFDictionaryGetValue(stats, key);
  int64_t v = 0;
  if (n != NULL) CFNumberGetValue(n, kCFNumberSInt64Type, &v);
  return v;
}
#endif

CAMLprim value caml_munin_sys_disks(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(r);
  struct disks t = {NULL, 0, 0};
  const char* err = NULL;
#if defined(__APPLE__)
  /* Each IOBlockStorageDriver drives one disk and keeps its statistics. Its
     registry entry ID names it while it stays attached. */
  io_iterator_t it;
  if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                   IOServiceMatching("IOBlockStorageDriver"),
                                   &it) != KERN_SUCCESS) {
    fail("IOServiceGetMatchingServices failed");
  }
  io_registry_entry_t drive;
  while ((drive = IOIteratorNext(it)) != 0) {
    CFMutableDictionaryRef props = NULL;
    uint64_t id = 0;
    if (err == NULL &&
        IORegistryEntryGetRegistryEntryID(drive, &id) == KERN_SUCCESS &&
        IORegistryEntryCreateCFProperties(drive, &props, kCFAllocatorDefault,
                                          0) == KERN_SUCCESS &&
        props != NULL) {
      CFDictionaryRef stats = CFDictionaryGetValue(
          props, CFSTR(kIOBlockStorageDriverStatisticsKey));
      struct disk* d = stats == NULL ? NULL : next(&t);
      if (stats != NULL && d == NULL) err = "cannot allocate the disk table";
      if (d != NULL) {
        snprintf(d->name, sizeof d->name, "%llx", (unsigned long long)id);
        d->read =
            stat_of(stats, CFSTR(kIOBlockStorageDriverStatisticsBytesReadKey));
        d->written = stat_of(
            stats, CFSTR(kIOBlockStorageDriverStatisticsBytesWrittenKey));
        t.n++;
      }
    }
    if (props != NULL) CFRelease(props);
    IOObjectRelease(drive);
  }
  IOObjectRelease(it);
#elif defined(_WIN32)
  /* The physical drives are the MS-DOS devices PhysicalDriveN.
     IOCTL_DISK_PERFORMANCE asks for no access right, so a drive opens with
     none; a drive that refuses it, such as a virtual one, is left out. The
     call enables the drive's counters and no IOCTL_DISK_PERFORMANCE_OFF
     follows: Microsoft does not say whether it balances one caller's
     enabling or stops the counters for every reader, such as Task Manager. */
  static const wchar_t prefix[] = L"PhysicalDrive";
  wchar_t* names = dos_names();
  if (names == NULL) fail("QueryDosDeviceW failed");
  for (wchar_t* nm = names; *nm != L'\0' && err == NULL; nm += wcslen(nm) + 1) {
    if (wcsncmp(nm, prefix, wcslen(prefix)) != 0) continue;
    wchar_t path[MAX_PATH];
    if (_snwprintf(path, MAX_PATH, L"\\\\.\\%ls", nm) < 0) continue;
    path[MAX_PATH - 1] = L'\0';
    HANDLE h = CreateFileW(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                           OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) continue;
    DISK_PERFORMANCE* perf = disk_perf(h);
    CloseHandle(h);
    if (perf == NULL) continue;
    struct disk* d = next(&t);
    if (d == NULL) err = "cannot allocate the disk table";
    if (d != NULL && WideCharToMultiByte(CP_UTF8, 0, nm, -1, d->name,
                                         sizeof d->name, NULL, NULL) != 0) {
      d->read = (int64_t)perf->BytesRead.QuadPart;
      d->written = (int64_t)perf->BytesWritten.QuadPart;
      t.n++;
    }
    free(perf);
  }
  free(names);
#else
  fail("no disk counters on this system");
#endif
  if (err != NULL) {
    free(t.d);
    fail(err);
  }
  r = disk_array(&t);
  free(t.d);
  CAMLreturn(r);
}
