/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Files for the disk: opening, identity, positional reads and writes between
   a file and host memory, read-ahead advice and ordering a file's writes
   before later changes.

   A descriptor crosses to OCaml as an int: a file descriptor on POSIX, a
   HANDLE on Windows. A system call's failure raises [Unix.Unix_error],
   which the OCaml side reports naming the file.

   Opening and closing release the runtime running no signal handler
   first: a handler that raised there would end the call holding what it
   took, the path's copy before an open or the descriptor a close marked
   closed, which nothing gives back then. */

#define _GNU_SOURCE

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/osdeps.h>
#include <caml/signals.h>
#include <caml/threads.h>
#include <caml/unixsupport.h>
#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/mount.h>
#endif

#include "rig_pool.h"
#endif

/* A path that names no regular file. */
#define NOT_REGULAR (-1)

/* How a file is opened: for reading, for reading and writing, or created for
   reading and writing where its path names nothing. */
#define MODE_READ 0
#define MODE_WRITE 1
#define MODE_CREATE 2

/* The most bytes one read or write system call moves: below the limits of
   Linux (0x7ffff000) and of the BSDs (INT_MAX). */
#define CALL_BYTES ((intnat)1 << 30)

/* The most bytes of an identity: a device, a handle's type and its bytes. */
#define IDENTITY_BYTES 160

/* Windows */

#ifdef _WIN32

/* The volume's serial and the 128-bit file ID: NTFS's IDs carry a sequence
   number a reused record advances, and ReFS's are never reused. A file
   system without them, such as FAT, gives no exact identity. */
static void identify(HANDLE h, unsigned char *id, size_t *len) {
  FILE_ID_INFO info;
  *len = 0;
  if (GetFileInformationByHandleEx(h, FileIdInfo, &info, sizeof info)) {
    memcpy(id, &info, sizeof info);
    *len = sizeof info;
  }
}

static int open_file(const char *path, int mode, int64_t size, intnat *handle,
                     int64_t *file_size, unsigned char *id, size_t *len) {
  int create = mode == MODE_CREATE;
  wchar_t *wpath = caml_stat_strdup_to_utf16(path);
  caml_enter_blocking_section_no_pending();
  HANDLE h = CreateFileW(
      wpath, mode == MODE_READ ? GENERIC_READ : GENERIC_READ | GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
      create ? CREATE_NEW : OPEN_EXISTING,
      /* A link at the path is the name that exists: creating there fails
         rather than create the link's target. */
      FILE_ATTRIBUTE_NORMAL | (create ? FILE_FLAG_OPEN_REPARSE_POINT : 0),
      NULL);
  int code = 0, unsized = 0;
  BY_HANDLE_FILE_INFORMATION info;
  LARGE_INTEGER n;
  if (h == INVALID_HANDLE_VALUE) {
    DWORD e = GetLastError();
    /* CreateFileW opens no directory, and fails on one as access denied. */
    DWORD attributes = e == ERROR_ACCESS_DENIED ? GetFileAttributesW(wpath)
                                                : INVALID_FILE_ATTRIBUTES;
    if (attributes != INVALID_FILE_ATTRIBUTES &&
        (attributes & FILE_ATTRIBUTE_DIRECTORY))
      code = NOT_REGULAR;
    else
      code = (int)e;
  } else if (GetFileType(h) != FILE_TYPE_DISK ||
             !GetFileInformationByHandle(h, &info) ||
             (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
    code = NOT_REGULAR;
  } else if (create) {
    n.QuadPart = size;
    if (!SetFilePointerEx(h, n, NULL, FILE_BEGIN) || !SetEndOfFile(h))
      code = (int)GetLastError();
    unsized = code != 0;
    *file_size = size;
  } else if (!GetFileSizeEx(h, &n)) {
    code = (int)GetLastError();
  } else {
    *file_size = n.QuadPart;
  }
  if (code == 0) identify(h, id, len);
  if (code != 0 && h != INVALID_HANDLE_VALUE) CloseHandle(h);
  /* The file this call created and could not size goes. */
  if (unsized) DeleteFileW(wpath);
  caml_leave_blocking_section();
  caml_stat_free(wpath);
  *handle = (intnat)h;
  return code;
}

static void close_file(intnat h) { CloseHandle((HANDLE)h); }

static int sync_file(intnat h) {
  return FlushFileBuffers((HANDLE)h) ? 0 : (int)GetLastError();
}

/* The writes through the mapping at [addr] of [n] bytes, in its file. */
static int sync_mapping(void *addr, uintnat n) {
  return FlushViewOfFile(addr, n) ? 0 : (int)GetLastError();
}

/* The errno of the system's error [code], GetLastError's. */
static int unix_code(int code) {
  caml_win32_maperr((DWORD)code);
  return errno;
}

static intnat transfer_calls(int write, intnat h, int64_t pos, char *buf,
                             intnat n) {
  intnat done = 0;
  while (done < n) {
    DWORD len = (DWORD)(n - done < CALL_BYTES ? n - done : CALL_BYTES);
    OVERLAPPED ov;
    memset(&ov, 0, sizeof ov);
    ov.Offset = (DWORD)(uint64_t)(pos + done);
    ov.OffsetHigh = (DWORD)((uint64_t)(pos + done) >> 32);
    DWORD moved = 0;
    BOOL ok = write ? WriteFile((HANDLE)h, buf + done, len, &moved, &ov)
                    : ReadFile((HANDLE)h, buf + done, len, &moved, &ov);
    if (!ok) {
      DWORD e = GetLastError();
      if (!write && e == ERROR_HANDLE_EOF) break;
      return -(intnat)e;
    }
    if (moved == 0) {
      if (write) return -(intnat)ERROR_WRITE_FAULT;
      break;
    }
    done += moved;
  }
  return done;
}

static intnat transfer(int write, intnat h, int64_t pos, char *buf,
                       intnat n) {
  return transfer_calls(write, h, pos, buf, n);
}

static void watch_forks(void) {}

static int fork_count(void) { return 0; }

extern value caml_win32_alloc_handle(HANDLE);

static value handle_descr(intnat h) {
  return caml_win32_alloc_handle((HANDLE)h);
}

static void advise_file(intnat h, char *pages, int64_t pos, int64_t n) {
  (void)h;
  if (pages == NULL || n <= 0) return;
  WIN32_MEMORY_RANGE_ENTRY range = {pages + pos, (SIZE_T)n};
  PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
}

#else

/* POSIX */

static void append(unsigned char *id, size_t *len, const void *p, size_t n) {
  memcpy(id + *len, p, n);
  *len += n;
}

#if defined(__linux__)

/* The device and the file's handle (name_to_handle_at(2)), which holds the
   inode's number and generation on ext4, xfs and btrfs: a reused inode has
   a new generation, while its number and its coarse-clock times may repeat.
   A file system that gives no handle gives no exact identity. */
_Static_assert(sizeof(dev_t) + sizeof(int) + MAX_HANDLE_SZ <= IDENTITY_BYTES,
               "an identity holds a device, a handle's type and its bytes");

static void identify(int fd, const struct stat *st, unsigned char *id,
                     size_t *len) {
  struct {
    struct file_handle fh;
    unsigned char bytes[MAX_HANDLE_SZ];
  } handle;
  int mount;
  *len = 0;
  handle.fh.handle_bytes = MAX_HANDLE_SZ;
  if (name_to_handle_at(fd, "", &handle.fh, &mount, AT_EMPTY_PATH) != 0)
    return;
  append(id, len, &st->st_dev, sizeof st->st_dev);
  append(id, len, &handle.fh.handle_type, sizeof handle.fh.handle_type);
  append(id, len, handle.fh.f_handle, handle.fh.handle_bytes);
}

#elif defined(__APPLE__)

/* The device, the inode and its birth time, on APFS, which takes inode
   numbers from a counter it never winds back, and HFS+, where the birth time
   tells apart a number reused after its counter wrapped. Other file systems,
   such as FAT, exFAT or a network's, may give a new file an old number and no
   birth time: they give no exact identity. */
static void identify(int fd, const struct stat *st, unsigned char *id,
                     size_t *len) {
  struct statfs fs;
  *len = 0;
  if (fstatfs(fd, &fs) != 0 || (strcmp(fs.f_fstypename, "apfs") != 0 &&
                                strcmp(fs.f_fstypename, "hfs") != 0))
    return;
  append(id, len, &st->st_dev, sizeof st->st_dev);
  append(id, len, &st->st_ino, sizeof st->st_ino);
  append(id, len, &st->st_birthtimespec, sizeof st->st_birthtimespec);
}

#else

/* No exact identity known for this system. */
static void identify(int fd, const struct stat *st, unsigned char *id,
                     size_t *len) {
  (void)fd;
  (void)st;
  (void)id;
  *len = 0;
}

#endif

static int open_file(const char *path, int mode, int64_t size, intnat *handle,
                     int64_t *file_size, unsigned char *id, size_t *len) {
  int create = mode == MODE_CREATE;
  char *p = caml_stat_strdup(path);
  caml_enter_blocking_section_no_pending();
  /* Non-blocking, so that a FIFO is refused rather than waited on. It changes
     nothing for a regular file. */
  int flags = O_CLOEXEC | O_NONBLOCK |
              (create              ? O_RDWR | O_CREAT | O_EXCL
               : mode == MODE_READ ? O_RDONLY
                                   : O_RDWR);
  int fd;
  do
    fd = open(p, flags, 0666);
  while (fd < 0 && errno == EINTR);
  int code = 0, unsized = 0;
  struct stat st;
  if (fd < 0)
    code = errno;
  else if (fstat(fd, &st) != 0)
    code = errno;
  else if (!S_ISREG(st.st_mode))
    code = NOT_REGULAR;
  else if (create && ftruncate(fd, (off_t)size) != 0) {
    code = errno;
    unsized = 1;
  } else {
    *file_size = create ? size : (int64_t)st.st_size;
    identify(fd, &st, id, len);
  }
  if (code != 0 && fd >= 0) close(fd);
  /* The file this call created and could not size goes. */
  if (unsized) unlink(p);
  caml_leave_blocking_section();
  caml_stat_free(p);
  *handle = fd;
  return code;
}

static void close_file(intnat h) { close((int)h); }

static int fsync_retrying(int fd) {
  int r;
  do
    r = fsync(fd);
  while (r != 0 && errno == EINTR);
  return r == 0 ? 0 : errno;
}

#if defined(__APPLE__)

/* macOS's fsync reaches the drive, which may still reorder the writes
   (fsync(2)); a barrier orders them before later ones. A file system without
   barriers may take a full flush to the medium, which orders too, and one
   with neither, such as some network file systems, only fsync. Only a file
   system's refusal of a call falls through to the next: any other failure,
   such as a write the drive could not make, is the sync's, since a later
   call's success would hide it. */
static int refused(int code) {
  return code == ENOTSUP || code == ENOTTY || code == EINVAL;
}

static int fcntl_retrying(int fd, int cmd) {
  int r;
  do
    r = fcntl(fd, cmd);
  while (r == -1 && errno == EINTR);
  return r == -1 ? errno : 0;
}

static int sync_file(intnat h) {
  int fd = (int)h;
  int code = fcntl_retrying(fd, F_BARRIERFSYNC);
  if (code == 0 || !refused(code)) return code;
  code = fcntl_retrying(fd, F_FULLFSYNC);
  if (code == 0 || !refused(code)) return code;
  return fsync_retrying(fd);
}

#else

static int sync_file(intnat h) { return fsync_retrying((int)h); }

#endif

static int sync_mapping(void *addr, uintnat n) {
  return msync(addr, n, MS_SYNC) == 0 ? 0 : errno;
}

static int unix_code(int code) { return code; }

static intnat transfer_calls(int write, intnat h, int64_t pos, char *buf,
                             intnat n) {
  int fd = (int)h;
  intnat done = 0;
  while (done < n) {
    size_t len = (size_t)(n - done < CALL_BYTES ? n - done : CALL_BYTES);
    ssize_t moved = write ? pwrite(fd, buf + done, len, (off_t)(pos + done))
                          : pread(fd, buf + done, len, (off_t)(pos + done));
    if (moved < 0) {
      if (errno == EINTR) continue;
      return -(intnat)errno;
    }
    if (moved == 0) {
      if (write) return -(intnat)EIO;
      break;
    }
    done += moved;
  }
  return done;
}

/* Long reads

   A read of more than [SEGMENT] bytes is cut into segments that the calling
   thread and the disk's readers claim in turn, each read by system calls. A
   read copies the file's cached pages on the thread that makes it, so one
   thread moves them at the speed of one core's copy, and storage serves
   several reads at once. A 32 MiB piece of a file took, on fudge (Core
   Ultra 5 235, Linux 6.12, ext4 on NVMe), 3.0 ms in one pread from the page
   cache and 1.1 ms on six threads that claim its segments, and 10.2 and 6.1
   ms from the disk. io_uring with 16 reads in flight took as long as one
   pread, 3.0 and 10.5 ms: it copies cached pages on the submitting thread
   too.

   The readers are threads of the disk's own, one fewer than the
   performance cores the process may occupy (rig_pool_performance_cores),
   made at the first long read and asleep between reads; a read wakes as
   many as it has segments past its first. They are not the host's pool,
   whose one job at a time would make a read and a kernel on another domain
   wait for each other, however long the storage takes. One read at a time
   holds them; a read that finds them held reads alone. They block every
   signal but those a fault raises, as the pool's workers do. A forked child
   has none: it leaves the parent's state, whose lock and condition
   variables the parent's readers may hold or wait on, and makes its own at
   its first long read.

   A read that finds the end of the file reads up to it: the bytes read are
   those before the first segment that came short. Segments past that end,
   or after an error, are not read. */

#define SEGMENT ((intnat)2 << 20)

/* A read's segments: [n] bytes of [h] from [pos] into [buf]. [next] is the
   next segment to claim, [end] where a segment found the end of the file,
   else [n], [error] the first system error, else 0, [inside] the readers
   reading it, under the readers' lock, and [number] the read's, which
   tells a reader a read it has not worked on. */
struct reading {
  uint64_t number;
  intnat h;
  int64_t pos;
  char *buf;
  intnat n;
  int64_t segments;
  _Atomic int64_t next;
  _Atomic intnat end;
  _Atomic int error;
  int inside;
};

/* The readers' state, on the heap so that a forked child can leave it.
   [mu] guards [current], the read the readers work on or NULL, and each
   reading's [inside]. */
struct readers {
  pthread_mutex_t mu;
  pthread_cond_t work, out;
  struct reading *current;
  uint64_t reads; /* the long reads begun */
  int made;
};

/* Made by the holder of [readers_held], which alone reads it; NULL before
   the first long read, and in a forked child. */
static struct readers *readers;
static atomic_int readers_held;

/* Reads the segments of [r] this thread claims, until none is left, an
   error came, or the claimed segment starts past the end of the file. */
static void read_claimed(struct reading *r) {
  while (atomic_load(&r->error) == 0) {
    int64_t s = atomic_fetch_add(&r->next, 1);
    intnat at = (intnat)s * SEGMENT;
    if (s >= r->segments || at >= atomic_load(&r->end)) return;
    intnat len = r->n - at < SEGMENT ? r->n - at : SEGMENT;
    intnat got = transfer_calls(0, r->h, r->pos + at, r->buf + at, len);
    if (got < 0) {
      int none = 0;
      atomic_compare_exchange_strong(&r->error, &none, (int)-got);
    } else if (got < len) {
      intnat end = atomic_load(&r->end);
      while (at + got < end &&
             !atomic_compare_exchange_weak(&r->end, &end, at + got)) {
      }
    }
  }
}

static void *reader(void *arg) {
  struct readers *rs = arg;
  uint64_t last = 0;
  pthread_mutex_lock(&rs->mu);
  for (;;) {
    while (rs->current == NULL || rs->current->number == last)
      pthread_cond_wait(&rs->work, &rs->mu);
    struct reading *r = rs->current;
    last = r->number;
    r->inside++;
    pthread_mutex_unlock(&rs->mu);
    read_claimed(r);
    pthread_mutex_lock(&rs->mu);
    if (--r->inside == 0) pthread_cond_signal(&rs->out);
  }
  return NULL;
}

/* The readers, made with every signal but a fault's blocked, or NULL if
   their state cannot be. */
static struct readers *make_readers(void) {
  struct readers *rs = calloc(1, sizeof *rs);
  if (rs == NULL || pthread_mutex_init(&rs->mu, NULL) != 0 ||
      pthread_cond_init(&rs->work, NULL) != 0 ||
      pthread_cond_init(&rs->out, NULL) != 0) {
    free(rs);
    return NULL;
  }
  sigset_t all, old;
  sigfillset(&all);
  int faults[] = {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGTRAP, SIGABRT, SIGSYS};
  for (size_t i = 0; i < sizeof faults / sizeof *faults; i++)
    sigdelset(&all, faults[i]);
  pthread_sigmask(SIG_SETMASK, &all, &old);
  for (int i = 1; i < rig_pool_performance_cores(); i++) {
    pthread_t t;
    if (pthread_create(&t, NULL, reader, rs) != 0) break;
    pthread_detach(t);
    rs->made++;
  }
  pthread_sigmask(SIG_SETMASK, &old, NULL);
  return rs;
}

static intnat read_long(intnat h, int64_t pos, char *buf, intnat n) {
  if (atomic_exchange(&readers_held, 1) != 0)
    return transfer_calls(0, h, pos, buf, n);
  if (readers == NULL) readers = make_readers();
  struct readers *rs = readers;
  if (rs == NULL) {
    atomic_store(&readers_held, 0);
    return transfer_calls(0, h, pos, buf, n);
  }
  struct reading r = {0, h, pos, buf, n, (n + SEGMENT - 1) / SEGMENT,
                      0, n, 0, 0};
  pthread_mutex_lock(&rs->mu);
  r.number = ++rs->reads;
  rs->current = &r;
  int64_t wake = r.segments - 1 < rs->made ? r.segments - 1 : rs->made;
  for (int64_t i = 0; i < wake; i++) pthread_cond_signal(&rs->work);
  pthread_mutex_unlock(&rs->mu);
  read_claimed(&r);
  pthread_mutex_lock(&rs->mu);
  rs->current = NULL;
  while (r.inside > 0) pthread_cond_wait(&rs->out, &rs->mu);
  pthread_mutex_unlock(&rs->mu);
  atomic_store(&readers_held, 0);
  return r.error != 0 ? -(intnat)r.error : atomic_load(&r.end);
}

static intnat transfer(int write, intnat h, int64_t pos, char *buf,
                       intnat n) {
  if (!write && n > SEGMENT) return read_long(h, pos, buf, n);
  return transfer_calls(write, h, pos, buf, n);
}

/* Forks

   A forked child makes anew what a thread of its parent may have held at the
   fork: the descriptor table's lock and the files' sync locks, which the
   OCaml side makes anew once [forks] has moved, and the readers of long
   reads. */

static atomic_int forks;

static void forked_child(void) {
  atomic_fetch_add(&forks, 1);
  readers = NULL;
  atomic_store(&readers_held, 0);
}

static pthread_once_t watch_once = PTHREAD_ONCE_INIT;
static void watch(void) { pthread_atfork(NULL, NULL, forked_child); }

static void watch_forks(void) { pthread_once(&watch_once, watch); }

static int fork_count(void) { return atomic_load(&forks); }

static value handle_descr(intnat h) { return Val_long(h); }

static void advise_file(intnat h, char *pages, int64_t pos, int64_t n) {
#if defined(__APPLE__)
  (void)pages;
  while (n > 0) {
    int count = n > (1 << 30) ? (1 << 30) : (int)n;
    struct radvisory ra = {.ra_offset = (off_t)pos, .ra_count = count};
    if (fcntl((int)h, F_RDADVISE, &ra) != 0) break;
    pos += count;
    n -= count;
  }
#else
  (void)pages;
  if (n > 0) posix_fadvise((int)h, (off_t)pos, (off_t)n, POSIX_FADV_WILLNEED);
#endif
}

#endif

/* Raises [Unix.Unix_error] for the system's error [code] (errno, or
   GetLastError on Windows) of [call] on [arg]. */
static void fail(int code, const char *call, value arg) {
  caml_unix_error(unix_code(code), call, arg);
}

/* [open_file path mode size] is [(handle, size, id)], [handle] -1 if [path]
   names no regular file: [id] names the opened file exactly, two files never
   having the same, or is empty where the system gives no exact identity.
   [MODE_CREATE] creates the file at [size] bytes. Releases the runtime. */
value caml_rig_disk_open(value v_path, value v_mode, value v_size) {
  CAMLparam3(v_path, v_mode, v_size);
  CAMLlocal2(r, s);
  intnat h = -1;
  int64_t size = 0;
  unsigned char id[IDENTITY_BYTES];
  size_t len = 0;
  int code = open_file(String_val(v_path), Int_val(v_mode), Long_val(v_size),
                       &h, &size, id, &len);
  if (code == NOT_REGULAR) h = -1;
  else if (code != 0) fail(code, "open", v_path);
  s = caml_alloc_initialized_string(len, (const char *)id);
  r = caml_alloc_tuple(3);
  Store_field(r, 0, Val_long(h));
  Store_field(r, 1, Val_long(size));
  Store_field(r, 2, s);
  CAMLreturn(r);
}

/* [close h] closes the descriptor [h]. Releases the runtime. */
value caml_rig_disk_close(value v_handle) {
  intnat h = Long_val(v_handle);
  caml_enter_blocking_section_no_pending();
  close_file(h);
  caml_leave_blocking_section();
  return Val_unit;
}

/* [sync h] returns once the bytes written to the file [h] are ordered before
   every later change to the file system, across a crash. Releases the
   runtime. */
value caml_rig_disk_sync(value v_handle) {
  intnat h = Long_val(v_handle);
  caml_release_runtime_system();
  int code = sync_file(h);
  caml_acquire_runtime_system();
  if (code != 0) fail(code, "fsync", Nothing);
  return Val_unit;
}

/* Transfers

   [transfer write h pos buf n] moves [n] bytes between the file [h] from byte
   [pos] and host memory at [buf]: from the file when [write] is 0, into it
   otherwise. It is the number of bytes moved, fewer than [n] when a read
   reaches the end of the file, or the system's error negated. */

/* [read h pos dst n] reads [n] bytes of the file [h] from byte [pos] into host
   memory at [dst]: the bytes read. Releases the runtime. */
intnat caml_rig_disk_read(intnat h, intnat pos, intnat dst, intnat n) {
  caml_release_runtime_system();
  intnat r = transfer(0, h, pos, (char *)dst, n);
  caml_acquire_runtime_system();
  if (r < 0) fail((int)-r, "pread", Nothing);
  return r;
}

value caml_rig_disk_read_byte(value h, value pos, value dst, value n) {
  return Val_long(caml_rig_disk_read(Long_val(h), Long_val(pos),
                                        Long_val(dst), Long_val(n)));
}

/* [write h pos src n] writes [n] bytes of host memory at [src] into the file
   [h] from byte [pos]: [n]. Releases the runtime. */
intnat caml_rig_disk_write(intnat h, intnat pos, intnat src, intnat n) {
  caml_release_runtime_system();
  intnat r = transfer(1, h, pos, (char *)src, n);
  caml_acquire_runtime_system();
  if (r < 0) fail((int)-r, "pwrite", Nothing);
  return r;
}

value caml_rig_disk_write_byte(value h, value pos, value src, value n) {
  return Val_long(caml_rig_disk_write(Long_val(h), Long_val(pos),
                                         Long_val(src), Long_val(n)));
}

/* [watch_forks ()] has a forked child run [forked_child]. Called as the
   library starts, before any of its locks can be held. Keeps the runtime. */
value caml_rig_disk_watch_forks(value unit) {
  (void)unit;
  watch_forks();
  return Val_unit;
}

/* [forks ()] is the number of forks between the process the library started
   in and this one. Keeps the runtime. */
value caml_rig_disk_forks(value unit) {
  (void)unit;
  return Val_int(fork_count());
}

/* [descr h] is the descriptor [h] as a [Unix.file_descr]. Keeps the
   runtime. */
value caml_rig_disk_descr(value v_handle) {
  return handle_descr(Long_val(v_handle));
}

/* [advise h pages pos n] asks the system to read the [n] bytes of the file
   [h] from byte [pos] ahead of their use, without waiting for them. Linux and
   macOS take advice on the descriptor, so the pages read belong to the cache,
   which macOS does not count in the process's footprint; Windows takes it on
   the file's mapping [pages]. A hint: failures are ignored. Releases the
   runtime, with [pages] rooted: Linux allocates the range's pages and submits
   their reads in the caller. */

value caml_rig_disk_advise(value v_handle, value v_pages, value v_pos,
                           value v_n) {
  CAMLparam4(v_handle, v_pages, v_pos, v_n);
  char *pages =
      Is_some(v_pages) ? (char *)Caml_ba_data_val(Some_val(v_pages)) : NULL;
  intnat h = Long_val(v_handle);
  int64_t pos = Long_val(v_pos), n = Long_val(v_n);
  caml_release_runtime_system();
  advise_file(h, pages, pos, n);
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}

/* [msync pages] returns once the writes through the shared mapping [pages]
   are in its file, as the file's sync then makes durable. Releases the
   runtime, with [pages] rooted so the mapping outlives the call. */
value caml_rig_disk_msync(value v_pages) {
  CAMLparam1(v_pages);
  void *addr = Caml_ba_data_val(v_pages);
  uintnat n = caml_ba_byte_size(Caml_ba_array_val(v_pages));
  caml_release_runtime_system();
  int code = sync_mapping(addr, n);
  caml_acquire_runtime_system();
  if (code != 0) fail(code, "msync", Nothing);
  CAMLreturn(Val_unit);
}
