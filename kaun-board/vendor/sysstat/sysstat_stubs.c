/*
 * sysstat_stubs.c - OCaml FFI bindings for system metrics
 *
 * Platform-specific implementations.
 * macOS uses Mach APIs, IOKit, and libproc.
 * Linux uses /proc filesystem (parsed in OCaml), stubs return defaults.
 */

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/statvfs.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

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

/* ---------- Helpers ---------- */

#ifdef __APPLE__
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

/* ---------- CPU Load ---------- */

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
#else
  /* Linux: CPU stats read from /proc/stat in OCaml */
  result = caml_alloc(0, 0);
#endif

  CAMLreturn(result);
}

/* ---------- Memory Statistics ---------- */

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
#else
  /* Linux: Memory stats read from /proc/meminfo in OCaml */
  result = caml_alloc_tuple(12);
  for (int i = 0; i < 12; i++) {
    Store_field(result, i, caml_copy_int64(0));
  }
#endif

  CAMLreturn(result);
}

/* ---------- Network I/O ---------- */

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
  result = caml_alloc_tuple(4);
  for (int i = 0; i < 4; i++) Store_field(result, i, caml_copy_int64(0));
#else
  /* Linux: Network stats read from /proc/net/dev in OCaml */
  result = caml_alloc_tuple(4);
  for (int i = 0; i < 4; i++) Store_field(result, i, caml_copy_int64(0));
#endif

  CAMLreturn(result);
}

/* ---------- Disk I/O ---------- */

CAMLprim value caml_sysstat_get_disk_io(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);

#ifdef __APPLE__
  io_iterator_t drive_list;

  kern_return_t kr = IOServiceGetMatchingServices(
      kIOMainPortDefault, IOServiceMatching("IOBlockStorageDriver"),
      &drive_list);

  if (kr != KERN_SUCCESS) {
    result = caml_alloc_tuple(4);
    for (int i = 0; i < 4; i++) Store_field(result, i, caml_copy_int64(0));
    CAMLreturn(result);
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
#else
  /* Linux: Disk stats read from /proc/diskstats in OCaml */
  result = caml_alloc_tuple(4);
  for (int i = 0; i < 4; i++) {
    Store_field(result, i, caml_copy_int64(0));
  }
#endif

  CAMLreturn(result);
}

/* ---------- Process List ---------- */

CAMLprim value caml_sysstat_list_pids(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);

#ifdef __APPLE__
  int bufsize = proc_listpids(PROC_ALL_PIDS, 0, NULL, 0);
  if (bufsize <= 0) {
    result = caml_alloc(0, 0);
    CAMLreturn(result);
  }

  pid_t* pids = malloc(bufsize);
  if (!pids) {
    result = caml_alloc(0, 0);
    CAMLreturn(result);
  }

  int bytes_written = proc_listpids(PROC_ALL_PIDS, 0, pids, bufsize);
  if (bytes_written <= 0) {
    free(pids);
    result = caml_alloc(0, 0);
    CAMLreturn(result);
  }

  int num_pids = bytes_written / (int)sizeof(pid_t);

  result = caml_alloc(num_pids, 0);
  for (int i = 0; i < num_pids; i++) {
    Store_field(result, i, Val_int(pids[i]));
  }

  free(pids);
#else
  /* Linux: PIDs enumerated from /proc in OCaml */
  result = caml_alloc(0, 0);
#endif

  CAMLreturn(result);
}

/* ---------- Process Info ---------- */

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

/* ---------- Mach Timebase ---------- */

CAMLprim value caml_sysstat_get_timebase(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);

#ifdef __APPLE__
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);

  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_int(info.numer));
  Store_field(result, 1, Val_int(info.denom));
#else
  /* Linux: Not needed, return 1:1 ratio */
  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_int(1));
  Store_field(result, 1, Val_int(1));
#endif

  CAMLreturn(result);
}

/* ---------- Cross-Platform Functions ---------- */

/* Get system load averages (1, 5, 15 minute) */
CAMLprim value caml_sysstat_get_loadavg(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);

  double loadavg[3];
  if (getloadavg(loadavg, 3) != 3) {
    loadavg[0] = loadavg[1] = loadavg[2] = 0.0;
  }

  result = caml_alloc_tuple(3);
  Store_field(result, 0, caml_copy_double(loadavg[0]));
  Store_field(result, 1, caml_copy_double(loadavg[1]));
  Store_field(result, 2, caml_copy_double(loadavg[2]));

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
    CAMLreturn(caml_copy_int64(0));
  }

  struct timeval now;
  gettimeofday(&now, NULL);

  int64_t uptime = (int64_t)(now.tv_sec - boottime.tv_sec);
  CAMLreturn(caml_copy_int64(uptime));
#else
  /* Linux: read from /proc/uptime */
  FILE* f = fopen("/proc/uptime", "r");
  if (!f) {
    CAMLreturn(caml_copy_int64(0));
  }

  double uptime_secs = 0.0;
  if (fscanf(f, "%lf", &uptime_secs) != 1) {
    uptime_secs = 0.0;
  }
  fclose(f);

  CAMLreturn(caml_copy_int64((int64_t)uptime_secs));
#endif
}

/* Get system page size */
CAMLprim value caml_sysstat_getpagesize(value unit) {
  CAMLparam1(unit);
  CAMLreturn(Val_long(sysconf(_SC_PAGESIZE)));
}

/* Get disk filesystem statistics using statvfs */
CAMLprim value caml_sysstat_statvfs(value v_path) {
  CAMLparam1(v_path);
  CAMLlocal1(tup);

  const char* path = String_val(v_path);
  struct statvfs st;
  if (statvfs(path, &st) != 0) {
    caml_failwith("statvfs failed");
  }

  uint64_t fr =
      (st.f_frsize != 0) ? (uint64_t)st.f_frsize : (uint64_t)st.f_bsize;
  uint64_t total = fr * (uint64_t)st.f_blocks;
  uint64_t free = fr * (uint64_t)st.f_bfree;
  uint64_t avail = fr * (uint64_t)st.f_bavail;

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
  if (kr != KERN_SUCCESS) {
    tup = caml_alloc_tuple(2);
    Store_field(tup, 0, caml_copy_int64(-1));
    Store_field(tup, 1, caml_copy_int64(-1));
    CAMLreturn(tup);
  }
  tup = caml_alloc_tuple(2);
  Store_field(tup, 0, caml_copy_int64((int64_t)info.resident_size));
  Store_field(tup, 1, caml_copy_int64((int64_t)info.virtual_size));
#else
  /* Linux: Handled in OCaml via /proc/self/stat */
  tup = caml_alloc_tuple(2);
  Store_field(tup, 0, caml_copy_int64(-1));
  Store_field(tup, 1, caml_copy_int64(-1));
#endif

  CAMLreturn(tup);
}

/* Get clock ticks per second (for Linux jiffies conversion) */
CAMLprim value caml_sysstat_clk_tck(value unit) {
  CAMLparam1(unit);
  CAMLreturn(Val_long(sysconf(_SC_CLK_TCK)));
}

/* ---------- Mount Enumeration ---------- */

/* Get list of mounted filesystems.
   Returns array of (mount_point, device, fstype) tuples. */
CAMLprim value caml_sysstat_getmounts(value unit) {
  CAMLparam1(unit);
  CAMLlocal3(result, tup, str);

#ifdef __APPLE__
  struct statfs* mntbuf;
  int count = getmntinfo(&mntbuf, MNT_NOWAIT);
  if (count <= 0) {
    result = caml_alloc(0, 0);
    CAMLreturn(result);
  }

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
  if (!f) {
    result = caml_alloc(0, 0);
    CAMLreturn(result);
  }

  /* Read all entries into a temporary buffer to avoid TOCTOU race */
  int capacity = 64;
  int count = 0;
  char** dirs = malloc(capacity * sizeof(char*));
  char** devs = malloc(capacity * sizeof(char*));
  char** types = malloc(capacity * sizeof(char*));

  if (!dirs || !devs || !types) {
    free(dirs);
    free(devs);
    free(types);
    endmntent(f);
    result = caml_alloc(0, 0);
    CAMLreturn(result);
  }

  struct mntent* mnt;
  while ((mnt = getmntent(f)) != NULL) {
    if (count >= capacity) {
      capacity *= 2;
      char** new_dirs = realloc(dirs, capacity * sizeof(char*));
      char** new_devs = realloc(devs, capacity * sizeof(char*));
      char** new_types = realloc(types, capacity * sizeof(char*));
      if (!new_dirs || !new_devs || !new_types) {
        /* Clean up on allocation failure */
        for (int j = 0; j < count; j++) {
          free(dirs[j]);
          free(devs[j]);
          free(types[j]);
        }
        free(new_dirs ? new_dirs : dirs);
        free(new_devs ? new_devs : devs);
        free(new_types ? new_types : types);
        endmntent(f);
        result = caml_alloc(0, 0);
        CAMLreturn(result);
      }
      dirs = new_dirs;
      devs = new_devs;
      types = new_types;
    }
    dirs[count] = strdup(mnt->mnt_dir);
    devs[count] = strdup(mnt->mnt_fsname);
    types[count] = strdup(mnt->mnt_type);
    if (!dirs[count] || !devs[count] || !types[count]) {
      /* Clean up on strdup failure */
      for (int j = 0; j <= count; j++) {
        free(dirs[j]);
        free(devs[j]);
        free(types[j]);
      }
      free(dirs);
      free(devs);
      free(types);
      endmntent(f);
      result = caml_alloc(0, 0);
      CAMLreturn(result);
    }
    count++;
  }
  endmntent(f);

  /* Now allocate OCaml structures from the buffered data */
  result = caml_alloc(count, 0);
  for (int i = 0; i < count; i++) {
    tup = caml_alloc_tuple(3);
    str = caml_copy_string(dirs[i]);
    Store_field(tup, 0, str);
    str = caml_copy_string(devs[i]);
    Store_field(tup, 1, str);
    str = caml_copy_string(types[i]);
    Store_field(tup, 2, str);
    Store_field(result, i, tup);
  }

  /* Free temporary buffers */
  for (int i = 0; i < count; i++) {
    free(dirs[i]);
    free(devs[i]);
    free(types[i]);
  }
  free(dirs);
  free(devs);
  free(types);
#else
  result = caml_alloc(0, 0);
#endif

  CAMLreturn(result);
}
