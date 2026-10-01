/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Files for the disk device: opening, closing, positional reads and writes
   between a file and host memory, and copy-on-write mappings of a file.
   Errors are returned as codes, the system's (errno, or GetLastError on
   Windows) negated, and the OCaml side names the file. Every call that may
   block releases the runtime. */

#define _GNU_SOURCE
/* The bigarray of a mapping is built as the runtime builds those of
   Unix.map_file, whose operations are internal. */
#define CAML_INTERNALS

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/camlatomic.h>
#include <caml/custom.h>
#include <caml/memory.h>
#include <caml/misc.h>
#include <caml/mlvalues.h>
#include <caml/osdeps.h>
#include <caml/threads.h>
#include <errno.h>
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
#define NX_DEVICE_IO_URING 1
#endif
#endif
#endif

/* A file that is not a regular one, as [open_file] reports it. */
#define NOT_REGULAR (-1)

/* Too many open files, of the process or the system. */
#define TOO_MANY (-2)

/* How [open_file] opens a file: as the OCaml side's [read], [write] and
   [create]. */
#define MODE_READ 0
#define MODE_WRITE 1
#define MODE_CREATE 2

/* The most bytes one read or write system call moves. */
#define CALL_BYTES ((intnat)1 << 30)

/* Opening */

#ifdef _WIN32

static int open_file(const char *path, int mode, int64_t size,
                     intptr_t *handle, int64_t *file_size) {
  int create = mode == MODE_CREATE;
  wchar_t *wpath = caml_stat_strdup_to_utf16(path);
  caml_release_runtime_system();
  HANDLE h = CreateFileW(
      wpath, mode == MODE_READ ? GENERIC_READ : GENERIC_READ | GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
      create ? CREATE_ALWAYS : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
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
  *handle = (intptr_t)h;
  return code;
}

static void close_file(intptr_t h) { CloseHandle((HANDLE)h); }

static int identify(intptr_t h, int64_t identity[3]) {
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

#else

static int open_file(const char *path, int mode, int64_t size,
                     intptr_t *handle, int64_t *file_size) {
  int create = mode == MODE_CREATE;
  char *p = caml_stat_strdup(path);
  caml_release_runtime_system();
  /* Non-blocking, so that a FIFO is refused rather than waited on. It changes
     nothing for a regular file. */
  int flags = O_CLOEXEC | O_NONBLOCK |
              (create              ? O_RDWR | O_CREAT | O_TRUNC
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

static void close_file(intptr_t h) { close((int)h); }

static int identify(intptr_t h, int64_t identity[3]) {
  struct stat st;
  if (fstat((int)h, &st) != 0) return errno;
  identity[0] = (int64_t)st.st_dev;
  identity[1] = (int64_t)st.st_ino;
#if defined(__APPLE__)
  identity[2] = (int64_t)st.st_ctimespec.tv_sec * 1000000000 +
                st.st_ctimespec.tv_nsec;
#else
  identity[2] = (int64_t)st.st_ctim.tv_sec * 1000000000 + st.st_ctim.tv_nsec;
#endif
  return 0;
}

#endif

/* [open_file path mode size] is [(code, handle, size)]: [code] is 0, a
   system error, [NOT_REGULAR] or [TOO_MANY]. [mode] is [MODE_READ], for
   reading; [MODE_WRITE], for reading and writing; or [MODE_CREATE], which
   creates or truncates the file, for reading and writing, at [size] bytes. */
value caml_nx_device_file_open(value v_path, value v_mode, value v_size) {
  CAMLparam3(v_path, v_mode, v_size);
  CAMLlocal1(r);
  intptr_t h = -1;
  int64_t size = 0;
  int code = open_file(String_val(v_path), Int_val(v_mode), Long_val(v_size),
                       &h, &size);
  r = caml_alloc_tuple(3);
  Store_field(r, 0, Val_int(code));
  Store_field(r, 1, caml_copy_nativeint(h));
  Store_field(r, 2, Val_long(size));
  CAMLreturn(r);
}

value caml_nx_device_file_close(value v_handle) {
  intptr_t h = Nativeint_val(v_handle);
  caml_release_runtime_system();
  close_file(h);
  caml_acquire_runtime_system();
  return Val_unit;
}

/* [file_identity h] is [[| code; device; inode; change |]]: [code] is 0 or
   a system error, and the file [h] is the one on [device] numbered [inode],
   whose data or metadata last changed at [change], in nanoseconds. */
value caml_nx_device_file_identity(value v_handle) {
  CAMLparam1(v_handle);
  CAMLlocal1(r);
  int64_t identity[3] = {0, 0, 0};
  int code = identify(Nativeint_val(v_handle), identity);
  r = caml_alloc_tuple(4);
  Store_field(r, 0, Val_int(code));
  for (int i = 0; i < 3; i++) Store_field(r, i + 1, Val_long(identity[i]));
  CAMLreturn(r);
}

value caml_nx_device_error_message(value v_code) {
#ifdef _WIN32
  char msg[512];
  DWORD n = FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
      (DWORD)Int_val(v_code), 0, msg, sizeof msg, NULL);
  while (n > 0 && (msg[n - 1] == '\r' || msg[n - 1] == '\n' ||
                   msg[n - 1] == '.'))
    n--;
  if (n == 0) return caml_alloc_sprintf("error %d", Int_val(v_code));
  msg[n] = '\0';
  return caml_copy_string(msg);
#else
  return caml_copy_string(strerror(Int_val(v_code)));
#endif
}

/* Transfers

   [transfer write h pos buf n] moves [n] bytes between the file [h] from
   byte [pos] and the host memory [buf]: from the file with [write] false,
   into it otherwise. It is the number of bytes moved, fewer than [n] when a
   read reaches the end of the file, or a negated error code. */

#ifdef _WIN32

static intnat transfer_calls(int write, intptr_t h, int64_t pos, char *buf,
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

static intnat transfer_calls(int write, intptr_t h, int64_t pos, char *buf,
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

#ifdef NX_DEVICE_IO_URING

/* Linux's io_uring: a transfer is cut into requests of [SEGMENT] bytes,
   [DEPTH] of them in flight. The ring is made at the first transfer, if the
   kernel allows it and its probe reports the read and write operations, and
   used by one transfer at a time: the disk device is taken around every
   transfer. */

#define DEPTH 16
#define SEGMENT ((intnat)2 << 20)

static struct {
  int state; /* 0 before the first transfer, 1 ready, -1 unavailable */
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

/* A request's bytes: [len] from [off] of the transfer, [queued] once it is
   in the submission queue or in flight. */
struct request {
  intnat off, len;
  int queued;
};

static intnat transfer_ring(int write, intptr_t h, int64_t pos, char *buf,
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
      ring.state = -1;
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

#endif

static intnat transfer(int write, intptr_t h, int64_t pos, char *buf,
                       intnat n) {
#ifdef NX_DEVICE_IO_URING
  if (ring.state == 0) ring.state = ring_setup() ? 1 : -1;
  if (ring.state == 1) return transfer_ring(write, h, pos, buf, n);
#endif
  return transfer_calls(write, h, pos, buf, n);
}

intnat caml_nx_device_file_read(intnat h, intnat pos, intnat dst, intnat n) {
  caml_release_runtime_system();
  intnat r = transfer(0, h, pos, (char *)dst, n);
  caml_acquire_runtime_system();
  return r;
}

value caml_nx_device_file_read_byte(value h, value pos, value dst, value n) {
  return Val_long(caml_nx_device_file_read(
      Nativeint_val(h), Long_val(pos), Nativeint_val(dst), Long_val(n)));
}

intnat caml_nx_device_file_write(intnat h, intnat pos, intnat src, intnat n) {
  caml_release_runtime_system();
  intnat r = transfer(1, h, pos, (char *)src, n);
  caml_acquire_runtime_system();
  return r;
}

value caml_nx_device_file_write_byte(value h, value pos, value src, value n) {
  return Val_long(caml_nx_device_file_write(
      Nativeint_val(h), Long_val(pos), Nativeint_val(src), Long_val(n)));
}

/* Mappings

   [map_file h n] maps the [n] bytes of the file [h] copy-on-write: the
   process reads the file's pages, and a write to one makes a private copy of
   it, which never reaches the file. The result is a bigarray of chars over
   the mapping; the mapping is released once it and every sub-array of it are
   unreachable, as for Unix.map_file. */

static void unmap(void *addr, uintnat n) {
  if (n == 0) return;
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
  } else if (caml_atomic_counter_decr(&b->proxy->refcount) == 0) {
    unmap(b->proxy->data, b->proxy->size);
    free(b->proxy);
  }
}

static struct custom_operations mapping_ops = {
    "_bigarray",       mapping_finalize,           caml_ba_compare,
    caml_ba_hash,      caml_ba_serialize,          caml_ba_deserialize,
    custom_compare_ext_default, custom_fixed_length_default};

/* [file_map h n] is [(code, mapping)]: [code] is 0 or a system error, and
   [mapping] a bigarray over the file's [n > 0] bytes when [code] is 0. */
value caml_nx_device_file_map(value v_handle, value v_n) {
  CAMLparam2(v_handle, v_n);
  CAMLlocal2(r, ba);
  intptr_t h = Nativeint_val(v_handle);
  intnat n = Long_val(v_n);
  void *addr = NULL;
  int code = 0;
  caml_release_runtime_system();
#ifdef _WIN32
  HANDLE m = CreateFileMappingW((HANDLE)h, NULL, PAGE_WRITECOPY, 0, 0, NULL);
  if (m == NULL) {
    code = (int)GetLastError();
  } else {
    addr = MapViewOfFile(m, FILE_MAP_COPY, 0, 0, (SIZE_T)n);
    if (addr == NULL) code = (int)GetLastError();
    CloseHandle(m);
  }
#else
  addr = mmap(NULL, (size_t)n, PROT_READ | PROT_WRITE, MAP_PRIVATE, (int)h, 0);
  if (addr == MAP_FAILED) {
    code = errno;
    addr = NULL;
  }
#endif
  caml_acquire_runtime_system();
  r = caml_alloc_tuple(2);
  if (code == 0) {
    ba = caml_alloc_custom(&mapping_ops,
                           SIZEOF_BA_ARRAY + sizeof(intnat), 0, 1);
    struct caml_ba_array *b = Caml_ba_array_val(ba);
    b->data = addr;
    b->num_dims = 1;
    b->flags = CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_MAPPED_FILE;
    b->proxy = NULL;
    b->dim[0] = n;
  } else {
    ba = caml_ba_alloc_dims(CAML_BA_CHAR | CAML_BA_C_LAYOUT, 1, NULL, 0);
  }
  Store_field(r, 0, Val_int(code));
  Store_field(r, 1, ba);
  CAMLreturn(r);
}

/* Asks the system to read the [n] bytes of the file [h] from byte [pos] into
   its cache ahead of their use, without waiting for them: advice on the file,
   so that the pages it reads belong to the cache and not to the process. */
value caml_nx_device_file_advise(value v_handle, value v_pos, value v_n) {
  intptr_t h = Nativeint_val(v_handle);
  int64_t pos = Long_val(v_pos), n = Long_val(v_n);
#if defined(__APPLE__)
  while (n > 0) {
    int count = n > (1 << 30) ? (1 << 30) : (int)n;
    struct radvisory ra = {.ra_offset = (off_t)pos, .ra_count = count};
    if (fcntl((int)h, F_RDADVISE, &ra) != 0) break;
    pos += count;
    n -= count;
  }
#elif defined(_WIN32)
  (void)h;
  (void)pos;
  (void)n;
#else
  if (n > 0) posix_fadvise((int)h, (off_t)pos, (off_t)n, POSIX_FADV_WILLNEED);
#endif
  return Val_unit;
}
