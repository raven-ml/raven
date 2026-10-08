/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Files for the disk: opening, identity, positional reads and writes between
   a file and host memory, mappings of a file's pages, read-ahead advice and
   ordering a file's writes before later changes.

   A descriptor crosses to OCaml as an int: a file descriptor on POSIX, a
   HANDLE on Windows. A failure crosses as a code: 0 for none, the system's
   error (errno, or GetLastError on Windows) positive, or one of NOT_REGULAR
   and TOO_MANY; a transfer returns the error negated. The OCaml side names
   the file. */

#define _GNU_SOURCE

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/camlatomic.h>
#include <caml/custom.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/osdeps.h>
#include <caml/threads.h>
#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__linux__) && defined(__has_include)
#if __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#include <sys/syscall.h>
#if defined(IO_URING_OP_SUPPORTED) && defined(__NR_io_uring_setup)
#define DEVICE_DISK_IO_URING 1
#endif
#endif
#endif

/* A path that names no regular file. */
#define NOT_REGULAR (-1)

/* Too many open files, in the process or the system. */
#define TOO_MANY (-2)

/* How a file is opened: for reading, for reading and writing, or created for
   reading and writing where its path names nothing. */
#define MODE_READ 0
#define MODE_WRITE 1
#define MODE_CREATE 2

/* The most bytes one read or write system call moves: below the limits of
   Linux (0x7ffff000) and of the BSDs (INT_MAX). */
#define CALL_BYTES ((intnat)1 << 30)

/* Opening, identity, closing, syncing */

#ifdef _WIN32

static int open_file(const char *path, int mode, int64_t size, intnat *handle,
                     int64_t *file_size) {
  int create = mode == MODE_CREATE;
  wchar_t *wpath = caml_stat_strdup_to_utf16(path);
  caml_release_runtime_system();
  HANDLE h = CreateFileW(
      wpath, mode == MODE_READ ? GENERIC_READ : GENERIC_READ | GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
      create ? CREATE_NEW : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  int code = 0;
  BY_HANDLE_FILE_INFORMATION info;
  LARGE_INTEGER n;
  if (h == INVALID_HANDLE_VALUE) {
    DWORD e = GetLastError();
    code = e == ERROR_TOO_MANY_OPEN_FILES ? TOO_MANY : (int)e;
  } else if (GetFileType(h) != FILE_TYPE_DISK ||
             !GetFileInformationByHandle(h, &info) ||
             (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
    code = NOT_REGULAR;
  } else if (create) {
    n.QuadPart = size;
    if (!SetFilePointerEx(h, n, NULL, FILE_BEGIN) || !SetEndOfFile(h))
      code = (int)GetLastError();
    *file_size = size;
  } else if (!GetFileSizeEx(h, &n)) {
    code = (int)GetLastError();
  } else {
    *file_size = n.QuadPart;
  }
  if (code != 0 && h != INVALID_HANDLE_VALUE) CloseHandle(h);
  caml_acquire_runtime_system();
  caml_stat_free(wpath);
  *handle = (intnat)h;
  return code;
}

static void close_file(intnat h) { CloseHandle((HANDLE)h); }

static int identify(intnat h, int64_t identity[3]) {
  BY_HANDLE_FILE_INFORMATION info;
  FILE_BASIC_INFO basic;
  if (!GetFileInformationByHandle((HANDLE)h, &info) ||
      !GetFileInformationByHandleEx((HANDLE)h, FileBasicInfo, &basic,
                                    sizeof basic))
    return (int)GetLastError();
  identity[0] = (int64_t)info.dwVolumeSerialNumber;
  identity[1] = ((int64_t)info.nFileIndexHigh << 32) | info.nFileIndexLow;
  identity[2] = basic.ChangeTime.QuadPart * 100;
  return 0;
}

static int sync_file(intnat h) {
  return FlushFileBuffers((HANDLE)h) ? 0 : (int)GetLastError();
}

#else

static int open_file(const char *path, int mode, int64_t size, intnat *handle,
                     int64_t *file_size) {
  int create = mode == MODE_CREATE;
  char *p = caml_stat_strdup(path);
  caml_release_runtime_system();
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
  int code = 0;
  struct stat st;
  if (fd < 0)
    code = errno == EMFILE || errno == ENFILE ? TOO_MANY : errno;
  else if (fstat(fd, &st) != 0)
    code = errno;
  else if (!S_ISREG(st.st_mode))
    code = NOT_REGULAR;
  else if (create && ftruncate(fd, (off_t)size) != 0)
    code = errno;
  else
    *file_size = create ? size : (int64_t)st.st_size;
  if (code != 0 && fd >= 0) close(fd);
  caml_acquire_runtime_system();
  caml_stat_free(p);
  *handle = fd;
  return code;
}

static void close_file(intnat h) { close((int)h); }

static int identify(intnat h, int64_t identity[3]) {
  struct stat st;
  if (fstat((int)h, &st) != 0) return errno;
  identity[0] = (int64_t)st.st_dev;
  identity[1] = (int64_t)st.st_ino;
#if defined(__APPLE__)
  identity[2] =
      (int64_t)st.st_ctimespec.tv_sec * 1000000000 + st.st_ctimespec.tv_nsec;
#else
  identity[2] = (int64_t)st.st_ctim.tv_sec * 1000000000 + st.st_ctim.tv_nsec;
#endif
  return 0;
}

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
   with neither, such as some network file systems, only fsync. */
static int sync_file(intnat h) {
  int fd = (int)h;
  if (fcntl(fd, F_BARRIERFSYNC) == 0 || fcntl(fd, F_FULLFSYNC) == 0) return 0;
  return fsync_retrying(fd);
}

#else

static int sync_file(intnat h) { return fsync_retrying((int)h); }

#endif

#endif

/* [open_file path mode size] is [(code, handle, size)]. [MODE_CREATE] creates
   the file at [size] bytes. Releases the runtime. */
value caml_device_disk_open(value v_path, value v_mode, value v_size) {
  CAMLparam3(v_path, v_mode, v_size);
  CAMLlocal1(r);
  intnat h = -1;
  int64_t size = 0;
  int code = open_file(String_val(v_path), Int_val(v_mode), Long_val(v_size),
                       &h, &size);
  r = caml_alloc_tuple(3);
  Store_field(r, 0, Val_int(code));
  Store_field(r, 1, Val_long(h));
  Store_field(r, 2, Val_long(size));
  CAMLreturn(r);
}

/* [close h] closes the descriptor [h]. Releases the runtime. */
value caml_device_disk_close(value v_handle) {
  intnat h = Long_val(v_handle);
  caml_release_runtime_system();
  close_file(h);
  caml_acquire_runtime_system();
  return Val_unit;
}

/* [identity h] is [[| code; device; inode; change |]]: the file [h] is the one
   on [device] numbered [inode], whose data or metadata last changed at
   [change], in nanoseconds. Keeps the runtime: fstat does not block on a
   descriptor already open. */
value caml_device_disk_identity(value v_handle) {
  CAMLparam1(v_handle);
  CAMLlocal1(r);
  int64_t identity[3] = {0, 0, 0};
  int code = identify(Long_val(v_handle), identity);
  r = caml_alloc_tuple(4);
  Store_field(r, 0, Val_int(code));
  for (int i = 0; i < 3; i++) Store_field(r, i + 1, Val_long(identity[i]));
  CAMLreturn(r);
}

/* [sync h] is 0 once the bytes written to the file [h] are ordered before
   every later change to the file system, across a crash, or a code. Releases
   the runtime. */
value caml_device_disk_sync(value v_handle) {
  intnat h = Long_val(v_handle);
  caml_release_runtime_system();
  int code = sync_file(h);
  caml_acquire_runtime_system();
  return Val_int(code);
}

/* [error code] is the system's message for the positive [code]. Keeps the
   runtime. */
value caml_device_disk_error(value v_code) {
#ifdef _WIN32
  char msg[512];
  DWORD n = FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
      (DWORD)Int_val(v_code), 0, msg, sizeof msg, NULL);
  while (n > 0 &&
         (msg[n - 1] == '\r' || msg[n - 1] == '\n' || msg[n - 1] == '.'))
    n--;
  if (n == 0) return caml_alloc_sprintf("error %d", Int_val(v_code));
  msg[n] = '\0';
  return caml_copy_string(msg);
#else
  return caml_copy_string(strerror(Int_val(v_code)));
#endif
}

/* Transfers

   [transfer write h pos buf n] moves [n] bytes between the file [h] from byte
   [pos] and host memory at [buf]: from the file when [write] is 0, into it
   otherwise. It is the number of bytes moved, fewer than [n] when a read
   reaches the end of the file, or a negated code. */

#ifdef _WIN32

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

#else

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

#endif

#ifdef DEVICE_DISK_IO_URING

/* Linux's io_uring: a transfer is cut into requests of [SEGMENT] bytes,
   [DEPTH] of them in flight. The ring is made by the first transfer that
   takes it, if the kernel allows it and its probe reports the read and write
   operations. One transfer at a time holds it; a transfer that finds it held
   moves its bytes with system calls. */

#define DEPTH 16
#define SEGMENT ((intnat)2 << 20)

/* 0 before the first transfer that takes the ring, 1 ready, -1 unavailable.
   Written only by the ring's holder. */
static int ring_state;
static int ring_held;

static struct {
  int fd;
  unsigned *sq_head, *sq_tail, *sq_mask, *sq_array;
  unsigned *cq_head, *cq_tail, *cq_mask;
  struct io_uring_sqe *sqes;
  struct io_uring_cqe *cqes;
} ring;

static int supports(struct io_uring_probe *probe, int op) {
  return op <= probe->last_op &&
         (probe->ops[op].flags & IO_URING_OP_SUPPORTED) != 0;
}

static int ring_setup(void) {
  struct io_uring_params p;
  memset(&p, 0, sizeof p);
  int fd = (int)syscall(__NR_io_uring_setup, DEPTH, &p);
  if (fd < 0) return 0;
  size_t probe_size =
      sizeof(struct io_uring_probe) + 256 * sizeof(struct io_uring_probe_op);
  struct io_uring_probe *probe = calloc(1, probe_size);
  int ok = probe != NULL &&
           syscall(__NR_io_uring_register, fd, IORING_REGISTER_PROBE, probe,
                   256) == 0 &&
           supports(probe, IORING_OP_READ) && supports(probe, IORING_OP_WRITE);
  free(probe);
  size_t sq_size = p.sq_off.array + p.sq_entries * sizeof(unsigned);
  size_t cq_size = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
  int single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
  if (single && cq_size > sq_size) sq_size = cq_size;
  void *sq = MAP_FAILED, *cq = MAP_FAILED, *sqes = MAP_FAILED;
  if (ok) {
    sq = mmap(NULL, sq_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
              fd, IORING_OFF_SQ_RING);
    cq = single ? sq
                : mmap(NULL, cq_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
    sqes = mmap(NULL, p.sq_entries * sizeof(struct io_uring_sqe),
                PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd,
                IORING_OFF_SQES);
  }
  if (sq == MAP_FAILED || cq == MAP_FAILED || sqes == MAP_FAILED) {
    if (sq != MAP_FAILED) munmap(sq, sq_size);
    if (cq != MAP_FAILED && !single) munmap(cq, cq_size);
    if (sqes != MAP_FAILED)
      munmap(sqes, p.sq_entries * sizeof(struct io_uring_sqe));
    close(fd);
    return 0;
  }
  ring.fd = fd;
  ring.sq_head = (unsigned *)((char *)sq + p.sq_off.head);
  ring.sq_tail = (unsigned *)((char *)sq + p.sq_off.tail);
  ring.sq_mask = (unsigned *)((char *)sq + p.sq_off.ring_mask);
  ring.sq_array = (unsigned *)((char *)sq + p.sq_off.array);
  ring.cq_head = (unsigned *)((char *)cq + p.cq_off.head);
  ring.cq_tail = (unsigned *)((char *)cq + p.cq_off.tail);
  ring.cq_mask = (unsigned *)((char *)cq + p.cq_off.ring_mask);
  ring.cqes = (struct io_uring_cqe *)((char *)cq + p.cq_off.cqes);
  ring.sqes = (struct io_uring_sqe *)sqes;
  return 1;
}

/* A request's bytes: [len] from [off] of the transfer, [queued] once it is in
   the submission queue or in flight. */
struct request {
  intnat off, len;
  int queued;
};

static intnat transfer_ring(int write, intnat h, int64_t pos, char *buf,
                            intnat n) {
  struct request req[DEPTH];
  memset(req, 0, sizeof req);
  intnat next = 0; /* the first byte no request covers */
  intnat end = n;  /* where a read found the end of the file */
  int error = 0, inflight = 0;
  unsigned mask = *ring.sq_mask;
  for (;;) {
    unsigned tail = *ring.sq_tail;
    for (int s = 0; s < DEPTH && error == 0; s++) {
      if (req[s].queued) continue;
      if (req[s].len == 0) {
        if (next >= end) continue;
        req[s].off = next;
        req[s].len = end - next < SEGMENT ? end - next : SEGMENT;
        next += req[s].len;
      }
      struct io_uring_sqe *sqe = &ring.sqes[tail & mask];
      memset(sqe, 0, sizeof *sqe);
      sqe->opcode = write ? IORING_OP_WRITE : IORING_OP_READ;
      sqe->fd = (int)h;
      sqe->off = (uint64_t)(pos + req[s].off);
      sqe->addr = (uint64_t)(uintptr_t)(buf + req[s].off);
      sqe->len = (uint32_t)req[s].len;
      sqe->user_data = (uint64_t)s;
      ring.sq_array[tail & mask] = tail & mask;
      tail++;
      req[s].queued = 1;
      inflight++;
    }
    __atomic_store_n(ring.sq_tail, tail, __ATOMIC_RELEASE);
    if (inflight == 0) break;
    unsigned pending = tail - __atomic_load_n(ring.sq_head, __ATOMIC_ACQUIRE);
    if (syscall(__NR_io_uring_enter, ring.fd, pending, 1,
                IORING_ENTER_GETEVENTS, NULL, 0) < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EBUSY) continue;
      /* The ring is unusable: nothing it holds can be waited for. */
      ring_state = -1;
      return -(intnat)errno;
    }
    unsigned head = *ring.cq_head;
    unsigned cq_tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);
    for (; head != cq_tail; head++) {
      struct io_uring_cqe *cqe = &ring.cqes[head & *ring.cq_mask];
      struct request *r = &req[cqe->user_data];
      int res = cqe->res;
      r->queued = 0;
      inflight--;
      if (res == -EINTR || res == -EAGAIN) continue;
      if (res < 0 || (res == 0 && write)) {
        if (error == 0) error = res < 0 ? -res : EIO;
        r->len = 0;
      } else if (res == 0) {
        if (r->off < end) end = r->off;
        r->len = 0;
      } else {
        r->off += res;
        r->len -= res;
      }
      if (error != 0 || r->off >= end) r->len = 0;
    }
    __atomic_store_n(ring.cq_head, head, __ATOMIC_RELEASE);
  }
  return error != 0 ? -(intnat)error : end;
}

static intnat transfer(int write, intnat h, int64_t pos, char *buf,
                       intnat n) {
  if (__atomic_exchange_n(&ring_held, 1, __ATOMIC_ACQUIRE) == 0) {
    if (ring_state == 0) ring_state = ring_setup() ? 1 : -1;
    intnat r = ring_state == 1 ? transfer_ring(write, h, pos, buf, n)
                               : transfer_calls(write, h, pos, buf, n);
    __atomic_store_n(&ring_held, 0, __ATOMIC_RELEASE);
    return r;
  }
  return transfer_calls(write, h, pos, buf, n);
}

#else

static intnat transfer(int write, intnat h, int64_t pos, char *buf,
                       intnat n) {
  return transfer_calls(write, h, pos, buf, n);
}

#endif

/* [read h pos dst n] reads [n] bytes of the file [h] from byte [pos] into host
   memory at [dst]: the bytes read, or a negated code. Releases the runtime. */
intnat caml_device_disk_read(intnat h, intnat pos, intnat dst, intnat n) {
  caml_release_runtime_system();
  intnat r = transfer(0, h, pos, (char *)dst, n);
  caml_acquire_runtime_system();
  return r;
}

value caml_device_disk_read_byte(value h, value pos, value dst, value n) {
  return Val_long(caml_device_disk_read(Long_val(h), Long_val(pos),
                                        Long_val(dst), Long_val(n)));
}

/* [write h pos src n] writes [n] bytes of host memory at [src] into the file
   [h] from byte [pos]: [n], or a negated code. Releases the runtime. */
intnat caml_device_disk_write(intnat h, intnat pos, intnat src, intnat n) {
  caml_release_runtime_system();
  intnat r = transfer(1, h, pos, (char *)src, n);
  caml_acquire_runtime_system();
  return r;
}

value caml_device_disk_write_byte(value h, value pos, value src, value n) {
  return Val_long(caml_device_disk_write(Long_val(h), Long_val(pos),
                                         Long_val(src), Long_val(n)));
}

/* Mappings

   [map h n shared] maps the [n > 0] bytes of the file [h] into host memory:
   shared, where the mapping is the file, or copy-on-write, where a write
   makes the process's own copy of a page. The result is a bigarray of chars
   over the mapping, unmapped once it and every sub-array of it are
   unreachable. Its custom operations are its own, so it neither compares,
   hashes nor marshals. */

static void unmap(void *addr, uintnat n) {
#ifdef _WIN32
  (void)n;
  UnmapViewOfFile(addr);
#else
  munmap(addr, n);
#endif
}

static void mapping_finalize(value v) {
  struct caml_ba_array *b = Caml_ba_array_val(v);
  if (b->proxy == NULL) {
    unmap(b->data, caml_ba_byte_size(b));
  } else if (atomic_fetch_sub(&b->proxy->refcount, 1) == 1) {
    unmap(b->proxy->data, b->proxy->size);
    free(b->proxy);
  }
}

static struct custom_operations mapping_ops = {
    "device_disk_pages",        mapping_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

static void *map_file(intnat h, intnat n, int shared, int *code) {
#ifdef _WIN32
  HANDLE m = CreateFileMappingW((HANDLE)h, NULL,
                                shared ? PAGE_READWRITE : PAGE_WRITECOPY, 0, 0,
                                NULL);
  if (m == NULL) {
    *code = (int)GetLastError();
    return NULL;
  }
  void *addr =
      MapViewOfFile(m, shared ? FILE_MAP_WRITE : FILE_MAP_COPY, 0, 0, (SIZE_T)n);
  if (addr == NULL) *code = (int)GetLastError();
  CloseHandle(m);
  return addr;
#else
  void *addr = mmap(NULL, (size_t)n, PROT_READ | PROT_WRITE,
                    shared ? MAP_SHARED : MAP_PRIVATE, (int)h, 0);
  if (addr != MAP_FAILED) return addr;
  *code = errno;
  return NULL;
#endif
}

/* [map h n shared] is [(code, mapping)], [mapping] empty unless [code] is 0.
   Releases the runtime. */
value caml_device_disk_map(value v_handle, value v_n, value v_shared) {
  CAMLparam3(v_handle, v_n, v_shared);
  CAMLlocal2(r, ba);
  intnat h = Long_val(v_handle), n = Long_val(v_n);
  int shared = Bool_val(v_shared), code = 0;
  caml_release_runtime_system();
  void *addr = map_file(h, n, shared, &code);
  caml_acquire_runtime_system();
  if (code == 0) {
    ba = caml_alloc_custom(&mapping_ops, SIZEOF_BA_ARRAY + sizeof(intnat), 0,
                           1);
    struct caml_ba_array *b = Caml_ba_array_val(ba);
    b->data = addr;
    b->num_dims = 1;
    b->flags = CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_MAPPED_FILE;
    b->proxy = NULL;
    b->dim[0] = n;
  } else {
    ba = caml_ba_alloc_dims(CAML_BA_CHAR | CAML_BA_C_LAYOUT, 1, NULL, 0);
  }
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_int(code));
  Store_field(r, 1, ba);
  CAMLreturn(r);
}

/* [advise h pos n] asks the system to read the [n] bytes of the file [h] from
   byte [pos] into its cache, without waiting for them. Advice on the
   descriptor, not the mapping: the pages read belong to the cache, which
   macOS does not count in the process's footprint. A hint: failures are
   ignored. Keeps the runtime: the call only queues the reads. */
value caml_device_disk_advise(value v_handle, value v_pos, value v_n) {
#if defined(__APPLE__)
  int fd = (int)Long_val(v_handle);
  int64_t pos = Long_val(v_pos), n = Long_val(v_n);
  while (n > 0) {
    int count = n > (1 << 30) ? (1 << 30) : (int)n;
    struct radvisory ra = {.ra_offset = (off_t)pos, .ra_count = count};
    if (fcntl(fd, F_RDADVISE, &ra) != 0) break;
    pos += count;
    n -= count;
  }
#elif defined(_WIN32)
  (void)v_handle;
  (void)v_pos;
  (void)v_n;
#else
  if (Long_val(v_n) > 0)
    posix_fadvise((int)Long_val(v_handle), (off_t)Long_val(v_pos),
                  (off_t)Long_val(v_n), POSIX_FADV_WILLNEED);
#endif
  return Val_unit;
}

/* [msync pages] is 0 once the writes through the shared mapping [pages] are
   in its file, as the file's sync then makes durable, or a code. Releases the
   runtime, with [pages] rooted so the mapping outlives the call. */
value caml_device_disk_msync(value v_pages) {
  CAMLparam1(v_pages);
  void *addr = Caml_ba_data_val(v_pages);
  uintnat n = caml_ba_byte_size(Caml_ba_array_val(v_pages));
  caml_release_runtime_system();
#ifdef _WIN32
  int code = FlushViewOfFile(addr, n) ? 0 : (int)GetLastError();
#else
  int code = msync(addr, n, MS_SYNC) == 0 ? 0 : errno;
#endif
  caml_acquire_runtime_system();
  CAMLreturn(Val_int(code));
}
