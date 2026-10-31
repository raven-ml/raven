/*
 * sysstat_stubs.c - OCaml FFI bindings for system metrics
 *
 * Platform-specific implementations.
 * macOS uses Mach APIs, IOKit, and libproc.
 * Linux reads most statistics from /proc in OCaml.
 * Windows links kernel32 alone. The network counters and per-core times live
 * in iphlpapi.dll and ntdll.dll, which every Windows since Vista has: the
 * stubs load their functions at run time (system_proc), as Microsoft's
 * documentation of NtQuerySystemInformation requires.
 * A stub that cannot read what it returns raises Failure, as the load average
 * stub does on Windows, which keeps no load average.
 */

#define _GNU_SOURCE

/* _WIN32_WINNT 0x0600 (Vista) declares GetTickCount64; PSAPI_VERSION 2 maps
   GetProcessMemoryInfo to K32GetProcessMemoryInfo, in Windows 7's kernel32,
   so the stubs link no psapi. Both before any header: <stdint.h> fixes
   _WIN32_WINNT. */
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
#include <sys/param.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#if defined(_WIN32)
/* winsock2.h before windows.h, which would include the older winsock.h;
   ws2ipdef.h before iphlpapi.h, which declares the interface table only
   after it; iphlpapi.h, psapi.h, tlhelp32.h, winioctl.h and winternl.h need
   windows.h's types. */
#include <winsock2.h>
#include <ws2ipdef.h>
#include <windows.h>
#include <iphlpapi.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <winioctl.h>
#include <winternl.h>
#include <wchar.h>
#else
#include <sys/statvfs.h>
#endif

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/storage/IOBlockStorageDriver.h>
#include <libproc.h>
#include <mach/host_info.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/mach_time.h>
#include <mach/task.h>
#include <mach/task_info.h>
#include <mach/vm_statistics.h>
#include <net/if.h>
#include <net/if_types.h>
#include <net/route.h>
#include <pwd.h>
#include <sys/mount.h>
#include <sys/proc_info.h>
#include <sys/sysctl.h>
#endif

#ifdef __linux__
#include <mntent.h>
#endif

/* Helpers */

#if defined(_WIN32)
/* [system_proc dll name] loads [dll] from the system directory alone, so that
   no DLL of the same name elsewhere on the search path is loaded in its place,
   and is its function [name], or NULL. The caller frees [*module] once done
   with the function. The cast through void (*)(void) is how GCC lets a
   FARPROC become the function's own type. */
typedef void (*system_fn)(void);

static system_fn system_proc(const wchar_t* dll, const char* name,
                             HMODULE* module) {
  *module = LoadLibraryExW(dll, NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (*module == NULL) return NULL;
  system_fn f = (system_fn)GetProcAddress(*module, name);
  if (f == NULL) {
    FreeLibrary(*module);
    *module = NULL;
  }
  return f;
}

static int64_t large(LARGE_INTEGER x) { return (int64_t)x.QuadPart; }

static int64_t filetime(FILETIME t) {
  ULARGE_INTEGER u;
  u.LowPart = t.dwLowDateTime;
  u.HighPart = t.dwHighDateTime;
  return (int64_t)u.QuadPart;
}

/* [utf8 w out n] writes the NUL-ended UTF-16 string [w] to [out] as UTF-8, at
   most [n] bytes with the NUL, and is [out]; an unconvertible string is
   empty. */
static char* utf8(const wchar_t* w, char* out, int n) {
  if (WideCharToMultiByte(CP_UTF8, 0, w, -1, out, n, NULL, NULL) == 0) {
    out[0] = '\0';
  }
  return out;
}
#endif

#if defined(__APPLE__) || defined(_WIN32)
/* Allocate a 4-element int64 array for CPU time counters */
static value alloc_cpu_row(int64_t user, int64_t nice, int64_t sys,
                           int64_t idle) {
  CAMLparam0();
  CAMLlocal1(arr);
  arr = caml_alloc(4, 0);
  Store_field(arr, 0, caml_copy_int64(user));
  Store_field(arr, 1, caml_copy_int64(nice));
  Store_field(arr, 2, caml_copy_int64(sys));
  Store_field(arr, 3, caml_copy_int64(idle));
  CAMLreturn(arr);
}
#endif

/* CPU load */

CAMLprim value caml_sysstat_get_cpu_load(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);

#ifdef __APPLE__
  CAMLlocal2(row, totalrow);

  natural_t ncpu = 0;
  processor_info_array_t cpuInfo;
  mach_msg_type_number_t numCpuInfo;

  host_t host = mach_host_self();
  kern_return_t kr = host_processor_info(host, PROCESSOR_CPU_LOAD_INFO, &ncpu,
                                         &cpuInfo, &numCpuInfo);
  mach_port_deallocate(mach_task_self(), host);

  if (kr != KERN_SUCCESS || ncpu == 0) {
    result = caml_alloc(0, 0);
    CAMLreturn(result);
  }

  int64_t sum_user = 0, sum_nice = 0, sum_sys = 0, sum_idle = 0;
  result = caml_alloc((mlsize_t)(ncpu + 1), 0);

  for (natural_t i = 0; i < ncpu; i++) {
    integer_t* base = cpuInfo + (CPU_STATE_MAX * i);
    int64_t user = (int64_t)base[CPU_STATE_USER];
    int64_t nice = (int64_t)base[CPU_STATE_NICE];
    int64_t sys = (int64_t)base[CPU_STATE_SYSTEM];
    int64_t idle = (int64_t)base[CPU_STATE_IDLE];

    sum_user += user;
    sum_nice += nice;
    sum_sys += sys;
    sum_idle += idle;

    row = alloc_cpu_row(user, nice, sys, idle);
    Store_field(result, (mlsize_t)(i + 1), row);
  }

  totalrow = alloc_cpu_row(sum_user, sum_nice, sum_sys, sum_idle);
  Store_field(result, 0, totalrow);

  vm_deallocate(mach_task_self(), (vm_address_t)cpuInfo,
                (vm_size_t)(numCpuInfo * sizeof(integer_t)));
#elif defined(_WIN32)
  /* The system-wide row, then one per logical processor, as on macOS, in
     100 ns ticks. Kernel time includes idle time. The per-processor times
     come from NtQuerySystemInformation, one entry per processor; where it
     gives fewer than the processors of every group, as it may for the
     calling thread's group alone, the result has the system-wide row only. */
  CAMLlocal1(row);
  FILETIME idle_ft, kernel_ft, user_ft;
  if (!GetSystemTimes(&idle_ft, &kernel_ft, &user_ft)) {
    result = caml_alloc(0, 0);
    CAMLreturn(result);
  }
  int64_t idle = filetime(idle_ft);
  int64_t kernel = filetime(kernel_ft);
  int64_t user = filetime(user_ft);

  typedef NTSTATUS(NTAPI * query_fn)(SYSTEM_INFORMATION_CLASS, PVOID, ULONG,
                                     PULONG);
  enum { SYSTEM_PROCESSOR_PERFORMANCE = 8 };
  DWORD ncpu = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION* cores =
      ncpu == 0 ? NULL : calloc(ncpu, sizeof *cores);
  ULONG size = (ULONG)(ncpu * sizeof *cores);
  ULONG got = 0;
  HMODULE ntdll = NULL;
  query_fn query =
      cores == NULL
          ? NULL
          : (query_fn)system_proc(L"ntdll.dll", "NtQuerySystemInformation",
                                  &ntdll);
  /* An NTSTATUS below zero is an error. */
  int per_core = query != NULL &&
                 query((SYSTEM_INFORMATION_CLASS)SYSTEM_PROCESSOR_PERFORMANCE,
                       cores, size, &got) >= 0 &&
                 got == size;
  if (ntdll != NULL) FreeLibrary(ntdll);

  result = caml_alloc(per_core ? ncpu + 1 : 1, 0);
  row = alloc_cpu_row(user, 0, kernel - idle, idle);
  Store_field(result, 0, row);
  for (DWORD i = 0; per_core && i < ncpu; i++) {
    int64_t c_idle = large(cores[i].IdleTime);
    row = alloc_cpu_row(large(cores[i].UserTime), 0,
                        large(cores[i].KernelTime) - c_idle, c_idle);
    Store_field(result, i + 1, row);
  }
  free(cores);
#else
  /* Linux: CPU stats read from /proc/stat in OCaml */
  result = caml_alloc(0, 0);
#endif

  CAMLreturn(result);
}

/* Memory */

CAMLprim value caml_sysstat_get_memory(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);

#ifdef __APPLE__
  host_t host = mach_host_self();

  /* Get total memory */
  host_basic_info_data_t hbi;
  mach_msg_type_number_t hbi_count = HOST_BASIC_INFO_COUNT;
  kern_return_t kr =
      host_info(host, HOST_BASIC_INFO, (host_info_t)&hbi, &hbi_count);
  if (kr != KERN_SUCCESS) {
    mach_port_deallocate(mach_task_self(), host);
    caml_failwith("host_info failed");
  }

  /* Get VM statistics */
  vm_statistics64_data_t vm_stats;
  mach_msg_type_number_t vm_count = HOST_VM_INFO64_COUNT;
  kr = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stats,
                         &vm_count);
  mach_port_deallocate(mach_task_self(), host);
  if (kr != KERN_SUCCESS) {
    caml_failwith("host_statistics64 failed");
  }

  /* Get swap usage */
  int mib[2] = {CTL_VM, VM_SWAPUSAGE};
  struct xsw_usage swap_info;
  size_t swap_size = sizeof(swap_info);
  int swap_ok = (sysctl(mib, 2, &swap_info, &swap_size, NULL, 0) == 0);

  /* Return raw page counts - OCaml computes derived values */
  result = caml_alloc_tuple(12);
  Store_field(result, 0, caml_copy_int64((int64_t)hbi.max_mem));
  Store_field(result, 1, caml_copy_int64((int64_t)vm_page_size));
  Store_field(result, 2, caml_copy_int64((int64_t)vm_stats.active_count));
  Store_field(result, 3, caml_copy_int64((int64_t)vm_stats.inactive_count));
  Store_field(result, 4, caml_copy_int64((int64_t)vm_stats.speculative_count));
  Store_field(result, 5, caml_copy_int64((int64_t)vm_stats.wire_count));
  Store_field(result, 6,
              caml_copy_int64((int64_t)vm_stats.compressor_page_count));
  Store_field(result, 7, caml_copy_int64((int64_t)vm_stats.purgeable_count));
  Store_field(result, 8,
              caml_copy_int64((int64_t)vm_stats.external_page_count));
  Store_field(result, 9, caml_copy_int64((int64_t)vm_stats.free_count));
  Store_field(result, 10,
              caml_copy_int64(swap_ok ? (int64_t)swap_info.xsu_total : 0));
  Store_field(result, 11,
              caml_copy_int64(swap_ok ? (int64_t)swap_info.xsu_used : 0));
#elif defined(_WIN32)
  /* Physical memory is free plus used, in bytes (page size 1). The page file
     beyond physical memory stands in for swap. */
  MEMORYSTATUSEX status;
  status.dwLength = sizeof status;
  if (!GlobalMemoryStatusEx(&status)) {
    caml_failwith("GlobalMemoryStatusEx failed");
  }
  int64_t total = (int64_t)status.ullTotalPhys;
  int64_t avail = (int64_t)status.ullAvailPhys;
  int64_t commit_total = (int64_t)status.ullTotalPageFile;
  int64_t commit_used = commit_total - (int64_t)status.ullAvailPageFile;
  int64_t swap_total = commit_total > total ? commit_total - total : 0;
  int64_t swap_used = commit_used - (total - avail);
  if (swap_used < 0) swap_used = 0;
  if (swap_used > swap_total) swap_used = swap_total;
  result = caml_alloc_tuple(12);
  for (int i = 0; i < 12; i++) {
    Store_field(result, i, caml_copy_int64(0));
  }
  Store_field(result, 0, caml_copy_int64(total));
  Store_field(result, 1, caml_copy_int64(1));
  Store_field(result, 5, caml_copy_int64(total - avail));
  Store_field(result, 9, caml_copy_int64(avail));
  Store_field(result, 10, caml_copy_int64(swap_total));
  Store_field(result, 11, caml_copy_int64(swap_used));
#else
  /* Linux reads /proc/meminfo in OCaml. */
  caml_failwith("no memory statistics on this system");
#endif

  CAMLreturn(result);
}

/* Network I/O */

CAMLprim value caml_sysstat_get_network_io(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);

#ifdef __APPLE__
  int mib[6] = {CTL_NET, PF_ROUTE, 0, 0, NET_RT_IFLIST2, 0};
  size_t len = 0;
  char* buf = NULL;

  for (int retry = 0; retry < 4; retry++) {
    len = 0;
    if (sysctl(mib, 6, NULL, &len, NULL, 0) < 0 || len == 0) goto fail;

    len += 16 * retry * retry * sizeof(struct if_msghdr2);
    buf = malloc(len);
    if (!buf) goto fail;

    if (sysctl(mib, 6, buf, &len, NULL, 0) == 0) break;
    free(buf);
    buf = NULL;
    if (retry == 3) goto fail;
  }

  uint64_t bytes_rx = 0, packets_rx = 0, bytes_tx = 0, packets_tx = 0;
  for (char* next = buf; next < buf + len;) {
    struct if_msghdr* ifm = (struct if_msghdr*)next;
    next += ifm->ifm_msglen;
    if (ifm->ifm_type != RTM_IFINFO2) continue;

    struct if_msghdr2* ifm2 = (struct if_msghdr2*)ifm;
    if (ifm2->ifm_data.ifi_type != IFT_LOOP) {
      bytes_rx += ifm2->ifm_data.ifi_ibytes;
      packets_rx += ifm2->ifm_data.ifi_ipackets;
      bytes_tx += ifm2->ifm_data.ifi_obytes;
      packets_tx += ifm2->ifm_data.ifi_opackets;
    }
  }
  free(buf);

  result = caml_alloc_tuple(4);
  Store_field(result, 0, caml_copy_int64((int64_t)bytes_rx));
  Store_field(result, 1, caml_copy_int64((int64_t)packets_rx));
  Store_field(result, 2, caml_copy_int64((int64_t)bytes_tx));
  Store_field(result, 3, caml_copy_int64((int64_t)packets_tx));
  CAMLreturn(result);

fail:
  caml_failwith("reading the interface list failed");
#elif defined(_WIN32)
  /* Every interface but loopbacks and filter modules, which a filter driver
     stacks on an adapter and which count the adapter's traffic again. */
  typedef NETIO_STATUS(WINAPI * table_fn)(PMIB_IF_TABLE2*);
  typedef VOID(WINAPI * free_fn)(PVOID);
  HMODULE dll;
  table_fn get_table =
      (table_fn)system_proc(L"iphlpapi.dll", "GetIfTable2", &dll);
  if (get_table == NULL) caml_failwith("GetIfTable2 not found");
  free_fn free_table = (free_fn)(system_fn)GetProcAddress(dll, "FreeMibTable");
  PMIB_IF_TABLE2 table = NULL;
  if (free_table == NULL || get_table(&table) != NO_ERROR) {
    FreeLibrary(dll);
    caml_failwith("GetIfTable2 failed");
  }

  uint64_t bytes_rx = 0, packets_rx = 0, bytes_tx = 0, packets_tx = 0;
  for (ULONG i = 0; i < table->NumEntries; i++) {
    const MIB_IF_ROW2* row = &table->Table[i];
    if (row->Type == IF_TYPE_SOFTWARE_LOOPBACK) continue;
    if (row->InterfaceAndOperStatusFlags.FilterInterface) continue;
    bytes_rx += row->InOctets;
    packets_rx += row->InUcastPkts + row->InNUcastPkts;
    bytes_tx += row->OutOctets;
    packets_tx += row->OutUcastPkts + row->OutNUcastPkts;
  }
  free_table(table);
  FreeLibrary(dll);

  result = caml_alloc_tuple(4);
  Store_field(result, 0, caml_copy_int64((int64_t)bytes_rx));
  Store_field(result, 1, caml_copy_int64((int64_t)packets_rx));
  Store_field(result, 2, caml_copy_int64((int64_t)bytes_tx));
  Store_field(result, 3, caml_copy_int64((int64_t)packets_tx));
#else
  /* Linux reads /proc/net/dev in OCaml. */
  caml_failwith("no network counters on this system");
#endif

  CAMLreturn(result);
}

/* Disk I/O */

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

/* The ticks per millisecond of the clock of [query], a DISK_PERFORMANCE's
   QueryTime, or 0 if it is neither clock below. Microsoft calls QueryTime
   "the system time stamp" and gives ReadTime and WriteTime no unit; they are
   read in QueryTime's clock: the system time, in 100 ns since 1601, if
   QueryTime lies within a minute of it, else the performance counter if it
   lies within a minute of that. */
static double disk_ticks_per_ms(int64_t query) {
  enum { MINUTE_100NS = 600000000 };
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  if (llabs(query - filetime(ft)) < MINUTE_100NS) return 10000.;
  LARGE_INTEGER count, freq;
  if (QueryPerformanceCounter(&count) && QueryPerformanceFrequency(&freq) &&
      llabs(query - count.QuadPart) < 60 * freq.QuadPart) {
    return (double)freq.QuadPart / 1000.;
  }
  return 0.;
}
#endif

CAMLprim value caml_sysstat_get_disk_io(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);

#ifdef __APPLE__
  io_iterator_t drive_list;

  kern_return_t kr = IOServiceGetMatchingServices(
      kIOMainPortDefault, IOServiceMatching("IOBlockStorageDriver"),
      &drive_list);

  if (kr != KERN_SUCCESS) {
    caml_failwith("IOServiceGetMatchingServices failed");
  }

  uint64_t read_sum = 0, write_sum = 0, time_sum = 0;
  uint64_t num_disks = 0;

  io_registry_entry_t drive;
  while ((drive = IOIteratorNext(drive_list)) != 0) {
    CFMutableDictionaryRef properties = NULL;

    if (IORegistryEntryCreateCFProperties(
            drive, &properties, kCFAllocatorDefault, 0) != KERN_SUCCESS) {
      IOObjectRelease(drive);
      continue;
    }

    if (!properties) {
      IOObjectRelease(drive);
      continue;
    }

    CFDictionaryRef statistics = CFDictionaryGetValue(
        properties, CFSTR(kIOBlockStorageDriverStatisticsKey));

    if (statistics) {
      num_disks++;
      CFNumberRef number;
      uint64_t value;

      number = CFDictionaryGetValue(
          statistics, CFSTR(kIOBlockStorageDriverStatisticsBytesReadKey));
      if (number) {
        CFNumberGetValue(number, kCFNumberSInt64Type, &value);
        read_sum += value;
      }

      number = CFDictionaryGetValue(
          statistics, CFSTR(kIOBlockStorageDriverStatisticsBytesWrittenKey));
      if (number) {
        CFNumberGetValue(number, kCFNumberSInt64Type, &value);
        write_sum += value;
      }

      number = CFDictionaryGetValue(
          statistics, CFSTR(kIOBlockStorageDriverStatisticsTotalReadTimeKey));
      if (number) {
        CFNumberGetValue(number, kCFNumberSInt64Type, &value);
        time_sum += value;
      }

      number = CFDictionaryGetValue(
          statistics, CFSTR(kIOBlockStorageDriverStatisticsTotalWriteTimeKey));
      if (number) {
        CFNumberGetValue(number, kCFNumberSInt64Type, &value);
        time_sum += value;
      }
    }

    CFRelease(properties);
    IOObjectRelease(drive);
  }

  IOObjectRelease(drive_list);

  result = caml_alloc_tuple(4);
  Store_field(result, 0, caml_copy_int64((int64_t)read_sum));
  Store_field(result, 1, caml_copy_int64((int64_t)write_sum));
  Store_field(result, 2,
              caml_copy_int64((int64_t)(time_sum / 1000000))); /* ns to ms */
  Store_field(result, 3, caml_copy_int64((int64_t)num_disks));
#elif defined(_WIN32)
  /* The physical drives are the MS-DOS device names PhysicalDriveN, which
     QueryDosDeviceW lists, each ended by a NUL, then an empty string; the
     list grows until the buffer holds it. IOCTL_DISK_PERFORMANCE asks for no
     access right, so a drive opens with none. A drive that refuses it, such
     as a virtual one, or whose clock is unknown (disk_ticks_per_ms), is left
     out of the counts. Like macOS's, the time is the sum of each request's,
     so concurrent requests count more than once (Disk_io.compute caps
     utilization at 100). IOCTL_DISK_PERFORMANCE enables the counters, and
     no IOCTL_DISK_PERFORMANCE_OFF follows: Microsoft does not say whether it
     balances one caller's enabling or stops the counters for every reader,
     such as Task Manager. */
  static const wchar_t prefix[] = L"PhysicalDrive";
  DWORD cap = 1 << 14;
  wchar_t* names = NULL;
  for (;;) {
    names = malloc(cap * sizeof *names);
    if (names == NULL) caml_failwith("QueryDosDeviceW failed");
    if (QueryDosDeviceW(NULL, names, cap) != 0) break;
    DWORD err = GetLastError();
    free(names);
    if (err != ERROR_INSUFFICIENT_BUFFER || cap > MAXDWORD / 2) {
      caml_failwith("QueryDosDeviceW failed");
    }
    cap *= 2;
  }

  uint64_t read_sum = 0, write_sum = 0;
  double time_ms = 0.;
  uint64_t num_disks = 0, drives = 0;
  for (wchar_t* n = names; *n != L'\0'; n += wcslen(n) + 1) {
    if (wcsncmp(n, prefix, wcslen(prefix)) != 0) continue;
    drives++;
    wchar_t path[MAX_PATH];
    if (_snwprintf(path, MAX_PATH, L"\\\\.\\%ls", n) < 0) continue;
    path[MAX_PATH - 1] = L'\0';
    HANDLE h = CreateFileW(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                           OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) continue;
    DISK_PERFORMANCE* perf = disk_perf(h);
    CloseHandle(h);
    if (perf == NULL) continue;
    double ticks = disk_ticks_per_ms(large(perf->QueryTime));
    if (ticks > 0.) {
      num_disks++;
      read_sum += (uint64_t)large(perf->BytesRead);
      write_sum += (uint64_t)large(perf->BytesWritten);
      time_ms +=
          (double)(large(perf->ReadTime) + large(perf->WriteTime)) / ticks;
    }
    free(perf);
  }
  free(names);
  if (drives > 0 && num_disks == 0) {
    caml_failwith("no physical drive gives its counters");
  }

  result = caml_alloc_tuple(4);
  Store_field(result, 0, caml_copy_int64((int64_t)read_sum));
  Store_field(result, 1, caml_copy_int64((int64_t)write_sum));
  Store_field(result, 2, caml_copy_int64((int64_t)time_ms));
  Store_field(result, 3, caml_copy_int64((int64_t)num_disks));
#else
  /* Linux reads /proc/diskstats in OCaml. */
  caml_failwith("no disk I/O counters on this system");
#endif

  CAMLreturn(result);
}

/* Process list */

CAMLprim value caml_sysstat_list_pids(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);

#ifdef __APPLE__
  int bufsize = proc_listpids(PROC_ALL_PIDS, 0, NULL, 0);
  if (bufsize <= 0) caml_failwith("proc_listpids failed");

  pid_t* pids = malloc(bufsize);
  if (!pids) caml_failwith("proc_listpids failed");

  int bytes_written = proc_listpids(PROC_ALL_PIDS, 0, pids, bufsize);
  if (bytes_written <= 0) {
    free(pids);
    caml_failwith("proc_listpids failed");
  }

  int num_pids = bytes_written / (int)sizeof(pid_t);

  result = caml_alloc(num_pids, 0);
  for (int i = 0; i < num_pids; i++) {
    Store_field(result, i, Val_int(pids[i]));
  }

  free(pids);
#else
  /* Linux enumerates /proc in OCaml. */
  caml_failwith("no process table on this system");
#endif

  CAMLreturn(result);
}

/* Process info */

#ifdef __APPLE__
/* Map macOS process state to a character code matching Linux convention */
static char macos_state_char(int p_stat) {
  switch (p_stat) {
    case SIDL:
      return 'I'; /* Idle (being created) */
    case SRUN:
      return 'R'; /* Running */
    case SSLEEP:
      return 'S'; /* Sleeping */
    case SSTOP:
      return 'T'; /* Stopped */
    case SZOMB:
      return 'Z'; /* Zombie */
    default:
      return '?';
  }
}

/* Get command line arguments for a process using sysctl KERN_PROCARGS2 */
static int get_proc_cmdline(pid_t pid, char* buf, size_t bufsize) {
  if (bufsize == 0) return 0;
  buf[0] = '\0';

  int mib[3] = {CTL_KERN, KERN_PROCARGS2, pid};
  size_t argmax = 0;

  /* Get maximum argument size */
  size_t argmax_size = sizeof(argmax);
  int argmax_mib[2] = {CTL_KERN, KERN_ARGMAX};
  if (sysctl(argmax_mib, 2, &argmax, &argmax_size, NULL, 0) != 0) {
    argmax = 65536; /* fallback */
  }

  char* procargs = malloc(argmax);
  if (!procargs) return 0;

  size_t size = argmax;
  if (sysctl(mib, 3, procargs, &size, NULL, 0) != 0) {
    free(procargs);
    return 0;
  }

  /* Skip argc (first 4 bytes) */
  if (size < sizeof(int)) {
    free(procargs);
    return 0;
  }

  int argc;
  memcpy(&argc, procargs, sizeof(int));
  char* p = procargs + sizeof(int);
  char* end = procargs + size;

  /* Skip executable path */
  while (p < end && *p != '\0') p++;
  while (p < end && *p == '\0') p++;

  /* Copy arguments up to bufsize, replacing nulls with spaces */
  size_t written = 0;
  int arg_count = 0;
  while (p < end && arg_count < argc && written < bufsize - 1) {
    if (*p == '\0') {
      if (written > 0 && written < bufsize - 1) {
        buf[written++] = ' ';
      }
      arg_count++;
      p++;
    } else {
      buf[written++] = *p++;
    }
  }

  /* Trim trailing space if present */
  if (written > 0 && buf[written - 1] == ' ') {
    written--;
  }
  buf[written] = '\0';

  free(procargs);
  return (int)written;
}
#endif

CAMLprim value caml_sysstat_get_proc_info(value v_pid) {
  CAMLparam1(v_pid);

#ifdef __APPLE__
  CAMLlocal5(result, name_str, cmdline_str, user_str, some);

  pid_t pid = Int_val(v_pid);

  /* Get task info for CPU times, memory, threads */
  struct proc_taskinfo pti;
  int ret = proc_pidinfo(pid, PROC_PIDTASKINFO, 0, &pti, PROC_PIDTASKINFO_SIZE);

  if (ret != PROC_PIDTASKINFO_SIZE) {
    CAMLreturn(Val_int(0)); /* None */
  }

  /* Get BSD info for ppid, state, nice, uid */
  struct proc_bsdinfo pbi;
  ret = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &pbi, PROC_PIDTBSDINFO_SIZE);

  int ppid = 0;
  char state_char = '?';
  int priority = 0;
  int nice = 0;
  uid_t uid = 0;

  if (ret == PROC_PIDTBSDINFO_SIZE) {
    ppid = (int)pbi.pbi_ppid;
    state_char = macos_state_char(pbi.pbi_status);
    nice = (int)pbi.pbi_nice;
    uid = pbi.pbi_uid;
  }

  /* Get priority from kinfo_proc via sysctl */
  {
    struct kinfo_proc kp;
    size_t kp_size = sizeof(kp);
    int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
    if (sysctl(mib, 4, &kp, &kp_size, NULL, 0) == 0 && kp_size > 0) {
      priority = (int)kp.kp_proc.p_priority;
    }
  }

  /* Get process name from path */
  char name[MAXPATHLEN];
  ret = proc_pidpath(pid, name, sizeof(name));
  if (ret <= 0) {
    proc_name(pid, name, sizeof(name));
  } else {
    char* slash = strrchr(name, '/');
    if (slash) {
      memmove(name, slash + 1, strlen(slash + 1) + 1);
    }
  }

  /* Get command line */
  char cmdline[4096];
  get_proc_cmdline(pid, cmdline, sizeof(cmdline));

  /* Get username from UID */
  char username[256] = "";
  struct passwd* pw = getpwuid(uid);
  if (pw && pw->pw_name) {
    strncpy(username, pw->pw_name, sizeof(username) - 1);
    username[sizeof(username) - 1] = '\0';
  }

  /* Build result tuple with 14 fields */
  result = caml_alloc_tuple(14);
  name_str = caml_copy_string(name);
  Store_field(result, 0, name_str);            /* name */
  Store_field(result, 1, Val_int(ppid));       /* ppid */
  Store_field(result, 2, Val_int(state_char)); /* state (as char code) */
  Store_field(result, 3, Val_int(priority));   /* priority */
  Store_field(result, 4, Val_int(nice));       /* nice */
  cmdline_str = caml_copy_string(cmdline);
  Store_field(result, 5, cmdline_str); /* cmdline */
  user_str = caml_copy_string(username);
  Store_field(result, 6, user_str); /* user */
  Store_field(result, 7,
              caml_copy_int64((int64_t)pti.pti_total_user)); /* user_time */
  Store_field(result, 8,
              caml_copy_int64((int64_t)pti.pti_total_system)); /* system_time */
  Store_field(
      result, 9,
      caml_copy_int64((int64_t)pti.pti_resident_size)); /* resident_size */
  Store_field(
      result, 10,
      caml_copy_int64((int64_t)pti.pti_virtual_size));  /* virtual_size */
  Store_field(result, 11, Val_int(pti.pti_threadnum));  /* num_threads */
  Store_field(result, 12, Val_int(pti.pti_numrunning)); /* num_running */
  Store_field(result, 13,
              caml_copy_int64((int64_t)pti.pti_faults)); /* faults */

  some = caml_alloc(1, 0);
  Store_field(some, 0, result);
  CAMLreturn(some);
#else
  /* Linux: Process info read from /proc/[pid] in OCaml */
  (void)v_pid;
  CAMLreturn(Val_int(0)); /* None */
#endif
}

/* Process table on Windows */

#if defined(_WIN32)
struct win_proc {
  DWORD pid, ppid, threads;
  LONG priority;
  char name[MAX_PATH * 3 + 1];
  int64_t user, kernel, resident, faults;
};

/* [win_proc_times p] reads the CPU times and memory of [p->pid], leaving them
   zero for a process this one may not open, such as another user's. Both
   take PROCESS_QUERY_LIMITED_INFORMATION alone; GetProcessMemoryInfo took
   PROCESS_VM_READ besides only on Windows XP and Server 2003. */
static void win_proc_times(struct win_proc* p) {
  HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, p->pid);
  if (h == NULL) return;
  FILETIME created, exited, kernel, user;
  if (GetProcessTimes(h, &created, &exited, &kernel, &user)) {
    p->user = filetime(user);
    p->kernel = filetime(kernel);
  }
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(h, &pmc, sizeof pmc)) {
    p->resident = (int64_t)pmc.WorkingSetSize;
    p->faults = (int64_t)pmc.PageFaultCount;
  }
  CloseHandle(h);
}
#endif

/* The process table as an array of (pid, info), info laid out as
   caml_sysstat_get_proc_info's. Windows lists it in one snapshot, from which
   the processes are read into C memory before the snapshot closes and any
   OCaml value is made. The idle process, pid 0, is left out: its CPU time is
   the processors' idle time. */
CAMLprim value caml_sysstat_proc_table(value unit) {
  CAMLparam1(unit);
  CAMLlocal4(result, info, pair, str);

#if defined(_WIN32)
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snap == INVALID_HANDLE_VALUE) {
    caml_failwith("CreateToolhelp32Snapshot failed");
  }
  size_t cap = 256, count = 0;
  struct win_proc* procs = malloc(cap * sizeof *procs);
  PROCESSENTRY32W e;
  e.dwSize = sizeof e;
  BOOL more = procs != NULL && Process32FirstW(snap, &e);
  int failed = !more;
  for (; more; more = Process32NextW(snap, &e)) {
    if (e.th32ProcessID == 0) continue;
    if (count == cap) {
      struct win_proc* grown = realloc(procs, 2 * cap * sizeof *procs);
      if (grown == NULL) {
        failed = 1;
        break;
      }
      procs = grown;
      cap *= 2;
    }
    struct win_proc* p = &procs[count++];
    memset(p, 0, sizeof *p);
    p->pid = e.th32ProcessID;
    p->ppid = e.th32ParentProcessID;
    p->threads = e.cntThreads;
    p->priority = e.pcPriClassBase;
    utf8(e.szExeFile, p->name, sizeof p->name);
    win_proc_times(p);
  }
  CloseHandle(snap);
  if (failed) {
    free(procs);
    caml_failwith("reading the process snapshot failed");
  }

  result = caml_alloc(count, 0);
  for (size_t i = 0; i < count; i++) {
    const struct win_proc* p = &procs[i];
    info = caml_alloc_tuple(14);
    for (int f = 0; f < 14; f++) Store_field(info, f, Val_int(0));
    str = caml_copy_string(p->name);
    Store_field(info, 0, str);
    Store_field(info, 1, Val_long(p->ppid));
    Store_field(info, 2, Val_int('?'));
    Store_field(info, 3, Val_long(p->priority));
    str = caml_copy_string("");
    Store_field(info, 5, str);
    Store_field(info, 6, str);
    str = caml_copy_int64(p->user);
    Store_field(info, 7, str);
    str = caml_copy_int64(p->kernel);
    Store_field(info, 8, str);
    str = caml_copy_int64(p->resident);
    Store_field(info, 9, str);
    str = caml_copy_int64(0);
    Store_field(info, 10, str);
    Store_field(info, 11, Val_long(p->threads));
    str = caml_copy_int64(p->faults);
    Store_field(info, 13, str);
    pair = caml_alloc_tuple(2);
    Store_field(pair, 0, Val_long(p->pid));
    Store_field(pair, 1, info);
    Store_field(result, i, pair);
  }
  free(procs);
#else
  /* Linux reads /proc in OCaml, macOS lists pids and reads each. */
  caml_failwith("no process snapshot on this system");
#endif

  CAMLreturn(result);
}

/* Mach timebase */

CAMLprim value caml_sysstat_get_timebase(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);

#ifdef __APPLE__
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);

  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_int(info.numer));
  Store_field(result, 1, Val_int(info.denom));
#elif defined(_WIN32)
  /* Process times are in 100 ns ticks. */
  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_int(100));
  Store_field(result, 1, Val_int(1));
#else
  /* Linux: Not needed, return 1:1 ratio */
  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_int(1));
  Store_field(result, 1, Val_int(1));
#endif

  CAMLreturn(result);
}

/* Cross-platform */

/* Get system load averages (1, 5, 15 minute) */
CAMLprim value caml_sysstat_get_loadavg(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);

#if defined(_WIN32)
  caml_failwith("Windows keeps no load average");
#else
  double loadavg[3];
  if (getloadavg(loadavg, 3) != 3) {
    caml_failwith("getloadavg failed");
  }

  result = caml_alloc_tuple(3);
  Store_field(result, 0, caml_copy_double(loadavg[0]));
  Store_field(result, 1, caml_copy_double(loadavg[1]));
  Store_field(result, 2, caml_copy_double(loadavg[2]));
#endif

  CAMLreturn(result);
}

/* Get system uptime in seconds */
CAMLprim value caml_sysstat_get_uptime(value unit) {
  CAMLparam1(unit);

#ifdef __APPLE__
  struct timeval boottime;
  size_t size = sizeof(boottime);
  int mib[2] = {CTL_KERN, KERN_BOOTTIME};

  if (sysctl(mib, 2, &boottime, &size, NULL, 0) != 0) {
    caml_failwith("sysctl KERN_BOOTTIME failed");
  }

  struct timeval now;
  gettimeofday(&now, NULL);

  int64_t uptime = (int64_t)(now.tv_sec - boottime.tv_sec);
  CAMLreturn(caml_copy_int64(uptime));
#elif defined(_WIN32)
  /* Milliseconds since boot. */
  CAMLreturn(caml_copy_int64((int64_t)(GetTickCount64() / 1000)));
#else
  /* Linux */
  FILE* f = fopen("/proc/uptime", "r");
  if (!f) caml_failwith("cannot open /proc/uptime");

  double uptime_secs = 0.0;
  int n = fscanf(f, "%lf", &uptime_secs);
  fclose(f);
  if (n != 1) caml_failwith("cannot read /proc/uptime");

  CAMLreturn(caml_copy_int64((int64_t)uptime_secs));
#endif
}

/* Get system page size */
CAMLprim value caml_sysstat_getpagesize(value unit) {
  CAMLparam1(unit);
#if defined(_WIN32)
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  CAMLreturn(Val_long(info.dwPageSize));
#else
  CAMLreturn(Val_long(sysconf(_SC_PAGESIZE)));
#endif
}

/* Get disk filesystem statistics using statvfs */
CAMLprim value caml_sysstat_statvfs(value v_path) {
  CAMLparam1(v_path);
  CAMLlocal1(tup);

  const char* path = String_val(v_path);
#if defined(_WIN32)
  ULARGE_INTEGER avail_bytes, total_bytes, free_bytes;
  if (!GetDiskFreeSpaceExA(path, &avail_bytes, &total_bytes, &free_bytes)) {
    caml_failwith("statvfs failed");
  }
  uint64_t total = total_bytes.QuadPart;
  uint64_t free = free_bytes.QuadPart;
  uint64_t avail = avail_bytes.QuadPart;
#else
  struct statvfs st;
  if (statvfs(path, &st) != 0) {
    caml_failwith("statvfs failed");
  }

  uint64_t fr =
      (st.f_frsize != 0) ? (uint64_t)st.f_frsize : (uint64_t)st.f_bsize;
  uint64_t total = fr * (uint64_t)st.f_blocks;
  uint64_t free = fr * (uint64_t)st.f_bfree;
  uint64_t avail = fr * (uint64_t)st.f_bavail;
#endif

  tup = caml_alloc_tuple(3);
  Store_field(tup, 0, caml_copy_int64((int64_t)total));
  Store_field(tup, 1, caml_copy_int64((int64_t)free));
  Store_field(tup, 2, caml_copy_int64((int64_t)avail));
  CAMLreturn(tup);
}

/* Get process (self) memory statistics */
CAMLprim value caml_sysstat_proc_self_mem(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(tup);

#ifdef __APPLE__
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                               (task_info_t)&info, &count);
  if (kr != KERN_SUCCESS) caml_failwith("task_info failed");
  tup = caml_alloc_tuple(2);
  Store_field(tup, 0, caml_copy_int64((int64_t)info.resident_size));
  Store_field(tup, 1, caml_copy_int64((int64_t)info.virtual_size));
#elif defined(_WIN32)
  /* The working set is resident. The virtual size is the user address space
     reserved or committed, which GlobalMemoryStatusEx gives for the calling
     process. */
  PROCESS_MEMORY_COUNTERS pmc;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
    caml_failwith("GetProcessMemoryInfo failed");
  }
  MEMORYSTATUSEX status;
  status.dwLength = sizeof status;
  if (!GlobalMemoryStatusEx(&status)) {
    caml_failwith("GlobalMemoryStatusEx failed");
  }
  DWORDLONG vsize = status.ullTotalVirtual - status.ullAvailVirtual;
  tup = caml_alloc_tuple(2);
  Store_field(tup, 0, caml_copy_int64((int64_t)pmc.WorkingSetSize));
  Store_field(tup, 1, caml_copy_int64((int64_t)vsize));
#else
  /* Linux reads /proc/self/stat in OCaml. */
  caml_failwith("no process memory statistics on this system");
#endif

  CAMLreturn(tup);
}

/* Get clock ticks per second (for Linux jiffies conversion) */
CAMLprim value caml_sysstat_clk_tck(value unit) {
  CAMLparam1(unit);
#if defined(_WIN32)
  /* No jiffies on Windows; the value only scales Linux /proc fields. */
  CAMLreturn(Val_long(1));
#else
  CAMLreturn(Val_long(sysconf(_SC_CLK_TCK)));
#endif
}

/* Mount enumeration */

/* Get list of mounted filesystems.
   Returns array of (mount_point, device, fstype) tuples. */
CAMLprim value caml_sysstat_getmounts(value unit) {
  CAMLparam1(unit);
  CAMLlocal3(result, tup, str);

#ifdef __APPLE__
  struct statfs* mntbuf;
  int count = getmntinfo(&mntbuf, MNT_NOWAIT);
  if (count <= 0) caml_failwith("getmntinfo failed");

  result = caml_alloc(count, 0);
  for (int i = 0; i < count; i++) {
    tup = caml_alloc_tuple(3);
    str = caml_copy_string(mntbuf[i].f_mntonname);
    Store_field(tup, 0, str);
    str = caml_copy_string(mntbuf[i].f_mntfromname);
    Store_field(tup, 1, str);
    str = caml_copy_string(mntbuf[i].f_fstypename);
    Store_field(tup, 2, str);
    Store_field(result, i, tup);
  }
#elif defined(__linux__)
  FILE* f = setmntent("/proc/mounts", "r");
  if (!f) caml_failwith("cannot open /proc/mounts");

  /* Read all entries into a temporary buffer to avoid TOCTOU race */
  struct mount {
    char* dir;
    char* dev;
    char* type;
  };
  size_t capacity = 64;
  size_t count = 0;
  struct mount* mounts = malloc(capacity * sizeof *mounts);
  int failed = mounts == NULL;

  struct mntent* mnt;
  while (!failed && (mnt = getmntent(f)) != NULL) {
    if (count == capacity) {
      struct mount* grown = realloc(mounts, 2 * capacity * sizeof *mounts);
      if (!grown) {
        failed = 1;
        break;
      }
      mounts = grown;
      capacity *= 2;
    }
    struct mount m = {strdup(mnt->mnt_dir), strdup(mnt->mnt_fsname),
                      strdup(mnt->mnt_type)};
    mounts[count++] = m;
    if (!m.dir || !m.dev || !m.type) failed = 1;
  }
  endmntent(f);

  /* Now allocate OCaml structures from the buffered data */
  if (!failed) {
    result = caml_alloc(count, 0);
    for (size_t i = 0; i < count; i++) {
      tup = caml_alloc_tuple(3);
      str = caml_copy_string(mounts[i].dir);
      Store_field(tup, 0, str);
      str = caml_copy_string(mounts[i].dev);
      Store_field(tup, 1, str);
      str = caml_copy_string(mounts[i].type);
      Store_field(tup, 2, str);
      Store_field(result, i, tup);
    }
  }

  for (size_t i = 0; i < count; i++) {
    free(mounts[i].dir);
    free(mounts[i].dev);
    free(mounts[i].type);
  }
  free(mounts);
  if (failed) caml_failwith("cannot read /proc/mounts");
#elif defined(_WIN32)
  /* The drive roots, such as C:\, each ended by a NUL, then an empty string:
     26 letters of 4 bytes each fit. Only fixed drives are listed. */
  enum { MAX_DRIVES = 26 };
  char drives[MAX_DRIVES * 4 + 1];
  DWORD len = GetLogicalDriveStringsA(sizeof drives, drives);
  if (len == 0 || len > sizeof drives) {
    caml_failwith("GetLogicalDriveStringsA failed");
  }

  char* fixed[MAX_DRIVES];
  int count = 0;
  for (char* d = drives; *d != '\0' && count < MAX_DRIVES; d += strlen(d) + 1) {
    if (GetDriveTypeA(d) == DRIVE_FIXED) fixed[count++] = d;
  }

  result = caml_alloc(count, 0);
  for (int i = 0; i < count; i++) {
    char fstype[MAX_PATH + 1];
    if (!GetVolumeInformationA(fixed[i], NULL, 0, NULL, NULL, NULL, fstype,
                               sizeof fstype)) {
      fstype[0] = '\0';
    }
    tup = caml_alloc_tuple(3);
    str = caml_copy_string(fixed[i]);
    Store_field(tup, 0, str);
    str = caml_copy_string(fixed[i]);
    Store_field(tup, 1, str);
    str = caml_copy_string(fstype);
    Store_field(tup, 2, str);
    Store_field(result, i, tup);
  }
#else
  caml_failwith("no mount table on this system");
#endif

  CAMLreturn(result);
}

/* GPUs */

#ifdef __APPLE__
/* The integer at [key] of the dictionary [d], or -1. */
static int64_t cf_int(CFDictionaryRef d, CFStringRef key) {
  CFTypeRef n = CFDictionaryGetValue(d, key);
  int64_t v;
  if (n == NULL || CFGetTypeID(n) != CFNumberGetTypeID()) return -1;
  if (!CFNumberGetValue((CFNumberRef)n, kCFNumberSInt64Type, &v)) return -1;
  return v;
}
#endif

/* The GPUs as an array of (name, load in percent, memory held in bytes), -1
   for what an accelerator does not report. The name is the accelerator's
   IOKit class and registry entry ID, which tells two of one class apart.
   Apple documents no key of an accelerator's PerformanceStatistics: these
   are the keys ioreg -c IOAccelerator shows, the memory on Apple silicon.
   The accelerators are read into C memory before any OCaml value is made.
   Linux reads sysfs in OCaml. */
CAMLprim value caml_sysstat_get_gpus(value unit) {
  CAMLparam1(unit);
  CAMLlocal3(result, tup, str);

#ifdef __APPLE__
  struct gpu {
    char name[sizeof(io_name_t) + 20];
    int64_t busy, used;
  };
  io_iterator_t it;
  if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                   IOServiceMatching("IOAccelerator"),
                                   &it) != KERN_SUCCESS) {
    caml_failwith("IOServiceGetMatchingServices failed");
  }
  size_t cap = 4, count = 0;
  struct gpu* gpus = malloc(cap * sizeof *gpus);
  int failed = gpus == NULL;
  io_registry_entry_t entry;
  while (!failed && (entry = IOIteratorNext(it)) != 0) {
    if (count == cap) {
      struct gpu* grown = realloc(gpus, 2 * cap * sizeof *gpus);
      if (grown == NULL) {
        IOObjectRelease(entry);
        failed = 1;
        break;
      }
      gpus = grown;
      cap *= 2;
    }
    struct gpu* g = &gpus[count++];
    g->busy = -1;
    g->used = -1;
    io_name_t cls;
    uint64_t entry_id = 0;
    if (IOObjectGetClass(entry, cls) != KERN_SUCCESS) {
      strcpy(cls, "IOAccelerator");
    }
    IORegistryEntryGetRegistryEntryID(entry, &entry_id);
    snprintf(g->name, sizeof g->name, "%s:%llx", cls,
             (unsigned long long)entry_id);
    CFTypeRef stats = IORegistryEntryCreateCFProperty(
        entry, CFSTR("PerformanceStatistics"), kCFAllocatorDefault, 0);
    if (stats != NULL && CFGetTypeID(stats) == CFDictionaryGetTypeID()) {
      g->busy = cf_int((CFDictionaryRef)stats, CFSTR("Device Utilization %"));
      g->used = cf_int((CFDictionaryRef)stats, CFSTR("In use system memory"));
    }
    if (stats != NULL) CFRelease(stats);
    IOObjectRelease(entry);
  }
  IOObjectRelease(it);
  if (failed) {
    free(gpus);
    caml_failwith("out of memory listing the accelerators");
  }

  result = caml_alloc(count, 0);
  for (size_t i = 0; i < count; i++) {
    tup = caml_alloc_tuple(3);
    str = caml_copy_string(gpus[i].name);
    Store_field(tup, 0, str);
    str = caml_copy_int64(gpus[i].busy);
    Store_field(tup, 1, str);
    str = caml_copy_int64(gpus[i].used);
    Store_field(tup, 2, str);
    Store_field(result, i, tup);
  }
  free(gpus);
#else
  caml_failwith("no GPU statistics on this system");
#endif

  CAMLreturn(result);
}
