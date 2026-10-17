/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The system calls of the uverbs path: the verbs ioctl, which runs a call
   the OCaml side lays out, the verbs file's mappings, the wait on its event
   files, and the kernel's id of a device's driver.

   A descriptor is an int, as the verbs structures hold it. A failed call
   raises Unix.Unix_error; the OCaml side says what it was doing. Linux only:
   elsewhere every call raises ENOSYS. The ioctl and the wait release the
   runtime; the others are short. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <caml/unixsupport.h>

#define PTR(v) ((void *)Long_val(v))

value caml_rig_mlx5_uverbs_address(value v_buffer) {
  return Val_long((intnat)Caml_ba_data_val(v_buffer));
}

#ifdef __linux__
#include <linux/netlink.h>
#include <poll.h>
#include <rdma/ib_user_verbs.h>
#include <rdma/rdma_netlink.h>
#include <rdma/rdma_user_ioctl_cmds.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

/* Raises Unix.Unix_error for the -errno [e] of [call]. */
static void check(int e, const char *call) {
  if (e < 0) caml_unix_error(-e, call, Nothing);
}

/* The descriptor [v_fd], which the kernel gave in an answer, as a
   Unix.file_descr. */
value caml_rig_mlx5_uverbs_descr(value v_fd) { return v_fd; }

value caml_rig_mlx5_uverbs_page_size(value unit) {
  (void)unit;
  return Val_long(sysconf(_SC_PAGESIZE));
}

/* Maps the [v_n] bytes at offset [v_off] of [v_fd], shared, readable and
   writable: their address. */
value caml_rig_mlx5_uverbs_map(value v_fd, value v_off, value v_n) {
  void *p = mmap(NULL, (size_t)Long_val(v_n), PROT_READ | PROT_WRITE,
                 MAP_SHARED, Int_val(v_fd), (off_t)Long_val(v_off));
  if (p == MAP_FAILED) caml_unix_error(errno, "mmap", Nothing);
  return Val_long((intnat)p);
}

value caml_rig_mlx5_uverbs_unmap(value v_at, value v_n) {
  munmap(PTR(v_at), (size_t)Long_val(v_n));
  return Val_unit;
}

/* Waits until [v_a] or [v_b] is readable or hung up, at most [v_ms]
   milliseconds: bit 0 set if [v_a] is readable, bit 1 if [v_b] is, bit 2 if
   either hung up or failed. Releases the runtime. */
value caml_rig_mlx5_uverbs_poll(value v_a, value v_b, value v_ms) {
  struct pollfd fds[2] = {{.fd = Int_val(v_a), .events = POLLIN},
                          {.fd = Int_val(v_b), .events = POLLIN}};
  int ms = Int_val(v_ms);
  caml_release_runtime_system();
  int r = poll(fds, 2, ms);
  int e = errno;
  caml_acquire_runtime_system();
  if (r < 0 && e != EINTR) caml_unix_error(e, "poll", Nothing);
  if (r < 0) return Val_long(0);
  short down = POLLHUP | POLLERR | POLLNVAL;
  return Val_long((fds[0].revents & POLLIN ? 1 : 0) |
                  (fds[1].revents & POLLIN ? 2 : 0) |
                  ((fds[0].revents | fds[1].revents) & down ? 4 : 0));
}

/* Reads the records of [v_size] bytes the event file [v_fd] holds into
   [v_buf] from byte [v_at], until the file holds no more or [v_buf] has no
   room for another: the bytes filled after. The file does not block, so the
   read holds the runtime, and no signal handler, finaliser or Gc.Memprof
   callback runs between the kernel giving the records and the caller
   storing their end. A failed read takes nothing. */
value caml_rig_mlx5_uverbs_take(value v_fd, value v_buf, value v_at,
                                value v_size) {
  size_t at = Long_val(v_at), size = Long_val(v_size);
  size_t room = (caml_string_length(v_buf) - at) / size * size;
  int fd = Int_val(v_fd);
  ssize_t r;
  while (room > 0 && (r = read(fd, Bytes_val(v_buf) + at, room)) > 0) {
    at += (size_t)r;
    room -= (size_t)r;
  }
  return Val_long(at);
}

/* Runs the verbs ioctl on [v_fd] with the call laid out in [v_call],
   retrying an interrupted one. Releases the runtime, with [v_call] rooted:
   the kernel may take long, such as when it pins memory. */
value caml_rig_mlx5_uverbs_ioctl(value v_fd, value v_call) {
  CAMLparam1(v_call);
  void *call = Caml_ba_data_val(v_call);
  int fd = Int_val(v_fd), r;
  caml_release_runtime_system();
  do r = ioctl(fd, RDMA_VERBS_IOCTL, call);
  while (r < 0 && errno == EINTR);
  int e = errno;
  caml_acquire_runtime_system();
  if (r < 0) caml_unix_error(e, "ioctl", Nothing);
  CAMLreturn(Val_unit);
}

/* RDMA netlink */

/* The attributes of the message [h], by type. */
static void nl_attrs(struct nlmsghdr *h, const struct nlattr **tb) {
  memset(tb, 0, RDMA_NLDEV_ATTR_MAX * sizeof *tb);
  int rest = (int)NLMSG_PAYLOAD(h, 0);
  const struct nlattr *a = NLMSG_DATA(h);
  while (rest >= NLA_HDRLEN && a->nla_len >= NLA_HDRLEN && a->nla_len <= rest) {
    int type = a->nla_type & NLA_TYPE_MASK;
    if (type < RDMA_NLDEV_ATTR_MAX) tb[type] = a;
    rest -= NLA_ALIGN(a->nla_len);
    a = (const struct nlattr *)((const char *)a + NLA_ALIGN(a->nla_len));
  }
}

/* The 32-bit attribute [a], or -1 if it is not one. The kernel puts a
   driver's id in 32 bits, whatever rdma_netlink.h's comment says. */
static long nl_u32(const struct nlattr *a) {
  uint32_t v;
  if (a == NULL || a->nla_len < NLA_HDRLEN + sizeof v) return -1;
  memcpy(&v, (const char *)a + NLA_HDRLEN, sizeof v);
  return v;
}

/* Sends [req] on [s] and calls [each] with the attributes of every message
   of the answer: 0 once it ends, or -errno. */
static int nl_ask(int s, struct nlmsghdr *req,
                  void (*each)(const struct nlattr **, void *), void *data) {
  if (send(s, req, req->nlmsg_len, 0) < 0) return -errno;
  char buf[1 << 14];
  const struct nlattr *tb[RDMA_NLDEV_ATTR_MAX];
  for (;;) {
    ssize_t n = recv(s, buf, sizeof buf, 0);
    if (n < 0) return -errno;
    if (n == 0) return 0;
    /* Unsigned, as NLMSG_OK compares it with a length, and a multiple of
       NLMSG_ALIGNTO, so NLMSG_NEXT never takes more than is left: a message
       that fits ends, aligned, within it. */
    unsigned int left = (unsigned int)n & ~(NLMSG_ALIGNTO - 1);
    for (struct nlmsghdr *h = (void *)buf; NLMSG_OK(h, left);
         h = NLMSG_NEXT(h, left)) {
      if (h->nlmsg_type == NLMSG_DONE) return 0;
      if (h->nlmsg_type == NLMSG_ERROR)
        return ((struct nlmsgerr *)NLMSG_DATA(h))->error;
      nl_attrs(h, tb);
      each(tb, data);
      if (!(h->nlmsg_flags & NLM_F_MULTI)) return 0;
    }
  }
}

/* A device's name and, once found, its index. */
typedef struct {
  const char *name;
  long index;
} device_t;

static void index_of(const struct nlattr **tb, void *data) {
  device_t *d = data;
  const struct nlattr *name = tb[RDMA_NLDEV_ATTR_DEV_NAME];
  if (name != NULL && strcmp((const char *)name + NLA_HDRLEN, d->name) == 0)
    d->index = nl_u32(tb[RDMA_NLDEV_ATTR_DEV_INDEX]);
}

static void driver_of(const struct nlattr **tb, void *data) {
  *(long *)data = nl_u32(tb[RDMA_NLDEV_ATTR_UVERBS_DRIVER_ID]);
}

/* The kernel's id of the driver of the RDMA device [v_name], which its
   verbs file's entry in the RDMA netlink lists; ENODEV if it lists none.
   The verbs ioctl names the driver, and the kernel refuses a call that
   names another. */
value caml_rig_mlx5_uverbs_driver(value v_name) {
  char name[IB_DEVICE_NAME_MAX];
  if (!caml_string_is_c_safe(v_name) ||
      caml_string_length(v_name) >= sizeof name)
    caml_unix_error(ENODEV, "netlink", v_name);
  strcpy(name, String_val(v_name));
  int s = socket(AF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, NETLINK_RDMA);
  if (s < 0) caml_unix_error(errno, "socket", Nothing);
  struct nlmsghdr list = {
      .nlmsg_len = NLMSG_LENGTH(0),
      .nlmsg_type = RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, RDMA_NLDEV_CMD_GET),
      .nlmsg_flags = NLM_F_REQUEST | NLM_F_DUMP,
      .nlmsg_seq = 1};
  device_t d = {.name = name, .index = -1};
  int e = nl_ask(s, &list, index_of, &d);
  long id = -1;
  if (e == 0 && d.index >= 0) {
    struct {
      struct nlmsghdr h;
      struct nlattr index_attr;
      uint32_t index;
      struct nlattr type_attr;
      char type[8];
    } verbs = {.h = {.nlmsg_len = sizeof verbs,
                     .nlmsg_type = RDMA_NL_GET_TYPE(RDMA_NL_NLDEV,
                                                    RDMA_NLDEV_CMD_GET_CHARDEV),
                     .nlmsg_flags = NLM_F_REQUEST,
                     .nlmsg_seq = 2},
               .index_attr = {.nla_len = NLA_HDRLEN + sizeof(uint32_t),
                              .nla_type = RDMA_NLDEV_ATTR_DEV_INDEX},
               .index = d.index,
               .type_attr = {.nla_len = NLA_HDRLEN + sizeof "uverbs",
                             .nla_type = RDMA_NLDEV_ATTR_CHARDEV_TYPE},
               .type = "uverbs"};
    e = nl_ask(s, &verbs.h, driver_of, &id);
  }
  close(s);
  check(e < 0 ? e : id < 0 ? -ENODEV : 0, "netlink");
  return Val_long(id);
}

#else

/* Off Linux no verbs file exists, so a path never opens to reach these. */
#define NONE(name, call, ...)                       \
  value caml_rig_mlx5_uverbs_##name(__VA_ARGS__) { \
    caml_unix_error(ENOSYS, call, Nothing);         \
  }

#define UNUSED __attribute__((unused))
NONE(ioctl, "ioctl", value a UNUSED, value b UNUSED)
NONE(driver, "netlink", value a UNUSED)
NONE(descr, "descr", value a UNUSED)
NONE(page_size, "sysconf", value a UNUSED)
NONE(map, "mmap", value a UNUSED, value b UNUSED, value c UNUSED)
NONE(unmap, "munmap", value a UNUSED, value b UNUSED)
NONE(poll, "poll", value a UNUSED, value b UNUSED, value c UNUSED)

value caml_rig_mlx5_uverbs_take(value v_fd UNUSED, value v_buf UNUSED,
                                value v_at, value v_size UNUSED) {
  return v_at;
}

#endif
