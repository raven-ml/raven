/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The system calls of the uverbs path: the verbs file, its ioctl, its
   mappings, its event files, and the kernel's id of a device's driver.

   Each call of the verbs ioctl is packed here, in the structures of Linux's
   rdma headers, so the compiler lays them out. A command of the kernel's
   older interface, such as registering memory, is the device's method
   INVOKE_WRITE, whose attributes carry the command's request, its answer
   and the driver's data.

   A descriptor is an int, as the verbs structures hold it. A failed call
   raises Unix.Unix_error; the OCaml side says what it was doing. Linux
   only: elsewhere every call raises ENOSYS. The ioctl and the wait release
   the runtime; the others are short. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>

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
#include <fcntl.h>
#include <linux/netlink.h>
#include <poll.h>
#include <rdma/ib_user_ioctl_cmds.h>
#include <rdma/ib_user_ioctl_verbs.h>
#include <rdma/ib_user_verbs.h>
#include <rdma/rdma_netlink.h>
#include <rdma/rdma_user_ioctl_cmds.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/* Raises Unix.Unix_error for the -errno [e] of [call]. */
static void check(int e, const char *call) {
  if (e < 0) caml_unix_error(-e, call, Nothing);
}

/* The kernel's values that no uapi header names, as libibverbs' verbs.h
   (rdma-core v56.0) names them: queue pair states, the attribute mask of a
   state change, the first MTU, a port's state and link layers, and events. */
enum { QPS_INIT = 1, QPS_RTR = 2, QPS_RTS = 3 };
enum {
  QP_STATE = 1 << 0,
  QP_ACCESS_FLAGS = 1 << 3,
  QP_PKEY_INDEX = 1 << 4,
  QP_PORT = 1 << 5,
  QP_AV = 1 << 7,
  QP_PATH_MTU = 1 << 8,
  QP_TIMEOUT = 1 << 9,
  QP_RETRY_CNT = 1 << 10,
  QP_RNR_RETRY = 1 << 11,
  QP_RQ_PSN = 1 << 12,
  QP_MAX_QP_RD_ATOMIC = 1 << 13,
  QP_MIN_RNR_TIMER = 1 << 15,
  QP_SQ_PSN = 1 << 16,
  QP_MAX_DEST_RD_ATOMIC = 1 << 17,
  QP_DEST_QPN = 1 << 20
};
enum { MTU_256 = 1, PORT_ACTIVE = 4 };
enum { LINK_LAYER_INFINIBAND = 1, LINK_LAYER_ETHERNET = 2 };

/* The asynchronous events the path reports, in the order of Path.async's
   constructors. */
static const uint32_t events[] = {0 /* IBV_EVENT_CQ_ERR */,
                                  1 /* IBV_EVENT_QP_FATAL */,
                                  2 /* IBV_EVENT_QP_REQ_ERR */,
                                  3 /* IBV_EVENT_QP_ACCESS_ERR */,
                                  8 /* IBV_EVENT_DEVICE_FATAL */,
                                  9 /* IBV_EVENT_PORT_ACTIVE */,
                                  10 /* IBV_EVENT_PORT_ERR */};

/* The port the path works on. */
#define PORT 1

/* What a queue pair answers a peer's request that finds no receive, after
   which a peer retries: 12, 0.64 ms, InfiniBand's [min_rnr_timer] encoding.
   These queue pairs post no receives, and peers send them none. */
#define RNR_TIMER 12
#define RNR_RETRIES 7

/* The hops a RoCE v2 packet crosses before a router drops it. */
#define HOP_LIMIT 64

/* The access flags of Path's access codes: bits 0 and 1 the access (local,
   remote write, remote), bit 2 set for writes in any order. */
static const uint32_t accesses[] = {
    IB_UVERBS_ACCESS_LOCAL_WRITE,
    IB_UVERBS_ACCESS_LOCAL_WRITE | IB_UVERBS_ACCESS_REMOTE_WRITE,
    IB_UVERBS_ACCESS_LOCAL_WRITE | IB_UVERBS_ACCESS_REMOTE_WRITE |
        IB_UVERBS_ACCESS_REMOTE_READ};

static uint32_t access_flags(value v_access) {
  int code = Int_val(v_access);
  return accesses[code & 3] |
         (code & 4 ? IB_UVERBS_ACCESS_RELAXED_ORDERING : 0);
}

value caml_rig_mlx5_uverbs_open(value v_path) {
  int fd = open(String_val(v_path), O_RDWR | O_CLOEXEC);
  if (fd < 0) caml_unix_error(errno, "open", v_path);
  return Val_long(fd);
}

value caml_rig_mlx5_uverbs_close(value v_fd) {
  close(Int_val(v_fd));
  return Val_unit;
}

/* Makes reads of [v_fd] answer -EAGAIN when it holds nothing. */
static int nonblock(int fd) {
  int flags = fcntl(fd, F_GETFL);
  if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) return -errno;
  return 0;
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

/* The verbs ioctl */

/* The attributes the longest call carries. */
#define MOST_ATTRS 9

typedef union {
  struct ib_uverbs_ioctl_hdr hdr;
  char bytes[sizeof(struct ib_uverbs_ioctl_hdr) +
             MOST_ATTRS * sizeof(struct ib_uverbs_attr)];
} call_t;

/* Starts the call of the method [meth] of the object [obj], on a device
   whose driver is [driver]. */
static void start(call_t *c, int driver, int obj, int meth) {
  memset(c, 0, sizeof *c);
  c->hdr.object_id = obj;
  c->hdr.method_id = meth;
  c->hdr.driver_id = driver;
}

/* Adds the attribute [id] of [len] bytes and data [data]: inline bytes, a
   handle, or the address of bytes. Every attribute is mandatory, so a
   kernel that does not know one refuses the call. */
static struct ib_uverbs_attr *attr(call_t *c, int id, int len, uint64_t data) {
  struct ib_uverbs_attr *a = &c->hdr.attrs[c->hdr.num_attrs++];
  a->attr_id = id;
  a->len = len;
  a->flags = UVERBS_ATTR_F_MANDATORY;
  a->data = data;
  return a;
}

/* Bytes the kernel reads or writes, inline if at most 8 when read. */
static void bytes_in(call_t *c, int id, const void *p, int len) {
  if (len <= 8) {
    struct ib_uverbs_attr *a = attr(c, id, len, 0);
    memcpy(&a->data, p, len);
  } else
    attr(c, id, len, (uintptr_t)p);
}

static void bytes_out(call_t *c, int id, void *p, int len) {
  attr(c, id, len, (uintptr_t)p);
}

/* Runs the call on [fd], retrying an interrupted one: 0, or -errno.
   Releases the runtime: the kernel may take long, such as when it pins
   memory. */
static int run(int fd, call_t *c) {
  c->hdr.length = sizeof c->hdr + c->hdr.num_attrs * sizeof(c->hdr.attrs[0]);
  int r;
  caml_release_runtime_system();
  do r = ioctl(fd, RDMA_VERBS_IOCTL, &c->hdr);
  while (r < 0 && errno == EINTR);
  int e = errno;
  caml_acquire_runtime_system();
  return r < 0 ? -e : 0;
}

/* Invokes the command [cmd] with the request [in], its answer in [out],
   the driver's data [uhw] and the driver's answer in [uhw_out]. An empty
   [out], [uhw] or [uhw_out] is left out. */
static int invoke(int fd, int driver, int cmd, const void *in, int in_len,
                  void *out, int out_len, const void *uhw, int uhw_len,
                  void *uhw_out, int uhw_out_len) {
  call_t c;
  start(&c, driver, UVERBS_OBJECT_DEVICE, UVERBS_METHOD_INVOKE_WRITE);
  attr(&c, UVERBS_ATTR_WRITE_CMD, 8, cmd);
  bytes_in(&c, UVERBS_ATTR_CORE_IN, in, in_len);
  if (out_len > 0) bytes_out(&c, UVERBS_ATTR_CORE_OUT, out, out_len);
  if (uhw_len > 0) bytes_in(&c, UVERBS_ATTR_UHW_IN, uhw, uhw_len);
  if (uhw_out_len > 0) bytes_out(&c, UVERBS_ATTR_UHW_OUT, uhw_out, uhw_out_len);
  return run(fd, &c);
}

/* A command with no driver data, with and without an answer. */
#define WRITE(fd, driver, cmd, in, out)                                    \
  invoke(fd, driver, cmd, &(in), sizeof(in), &(out), sizeof(out), NULL, 0, \
         NULL, 0)
#define WRITE_IN(fd, driver, cmd, in) \
  invoke(fd, driver, cmd, &(in), sizeof(in), NULL, 0, NULL, 0, NULL, 0)

/* The driver's data and its answer: [v_d]'s bytes, and room for [v_n]. */
typedef struct {
  char *in;
  int in_len;
  char *out;
  int out_len;
} uhw_t;

static uhw_t uhw_of(value v_d, value v_n) {
  uhw_t u = {.in_len = caml_string_length(v_d), .out_len = Int_val(v_n)};
  u.in = caml_stat_alloc(u.in_len + 1);
  memcpy(u.in, String_val(v_d), u.in_len);
  u.out = caml_stat_alloc(u.out_len + 1);
  memset(u.out, 0, u.out_len);
  return u;
}

/* The driver's answer, freeing [u]. */
static value uhw_answer(uhw_t *u) {
  value s = caml_alloc_initialized_string(u->out_len, u->out);
  caml_stat_free(u->in);
  caml_stat_free(u->out);
  return s;
}

/* Makes the process's context of the device, giving the driver [v_d] with
   room for [v_n] bytes of its answer: [(async_fd, answer)], the descriptor
   of asynchronous events, made non-blocking. */
value caml_rig_mlx5_uverbs_get_context(value v_fd, value v_driver, value v_d,
                                       value v_n) {
  CAMLparam2(v_d, v_n);
  CAMLlocal2(answer, r);
  struct ib_uverbs_get_context in = {0};
  struct ib_uverbs_get_context_resp out = {0};
  uhw_t u = uhw_of(v_d, v_n);
  int e = invoke(Int_val(v_fd), Int_val(v_driver),
                 IB_USER_VERBS_CMD_GET_CONTEXT, &in, sizeof in, &out,
                 sizeof out, u.in, u.in_len, u.out, u.out_len);
  answer = uhw_answer(&u);
  if (e == 0 && (e = nonblock(out.async_fd)) < 0) close(out.async_fd);
  check(e, "ioctl");
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_int(out.async_fd));
  Store_field(r, 1, answer);
  CAMLreturn(r);
}

/* Makes the completion channel: its descriptor, made non-blocking. */
value caml_rig_mlx5_uverbs_comp_channel(value v_fd, value v_driver) {
  struct ib_uverbs_create_comp_channel in = {0};
  struct ib_uverbs_create_comp_channel_resp out = {0};
  int e = WRITE(Int_val(v_fd), Int_val(v_driver),
                IB_USER_VERBS_CMD_CREATE_COMP_CHANNEL, in, out);
  if (e == 0 && (e = nonblock(out.fd)) < 0) close(out.fd);
  check(e, "ioctl");
  return Val_long((long)out.fd);
}

/* Makes the protection domain: its handle. */
value caml_rig_mlx5_uverbs_alloc_pd(value v_fd, value v_driver) {
  struct ib_uverbs_alloc_pd in = {0};
  struct ib_uverbs_alloc_pd_resp out = {0};
  check(WRITE(Int_val(v_fd), Int_val(v_driver), IB_USER_VERBS_CMD_ALLOC_PD,
              in, out),
        "ioctl");
  return Val_long((long)out.pd_handle);
}

/* The device's limits: [(reads, served)], the reads in flight a queue
   pair starts and serves concurrently. */
value caml_rig_mlx5_uverbs_query_device(value v_fd, value v_driver) {
  CAMLparam0();
  CAMLlocal1(r);
  struct ib_uverbs_query_device in = {0};
  struct ib_uverbs_query_device_resp out = {0};
  check(WRITE(Int_val(v_fd), Int_val(v_driver),
              IB_USER_VERBS_CMD_QUERY_DEVICE, in, out),
        "ioctl");
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_long(out.max_qp_init_rd_atom));
  Store_field(r, 1, Val_long(out.max_qp_rd_atom));
  CAMLreturn(r);
}

/* Port 1: [(active, link, lid, mtu)], the link a Path.link (InfiniBand,
   Ethernet, or another by its value) and the active MTU in bytes. */
value caml_rig_mlx5_uverbs_query_port(value v_fd, value v_driver) {
  CAMLparam0();
  CAMLlocal2(link, r);
  struct ib_uverbs_query_port in = {.port_num = PORT};
  struct ib_uverbs_query_port_resp out = {0};
  check(WRITE(Int_val(v_fd), Int_val(v_driver), IB_USER_VERBS_CMD_QUERY_PORT,
              in, out),
        "ioctl");
  if (out.link_layer == LINK_LAYER_INFINIBAND)
    link = Val_int(0);
  else if (out.link_layer == LINK_LAYER_ETHERNET)
    link = Val_int(1);
  else {
    link = caml_alloc_small(1, 0);
    Field(link, 0) = Val_int(out.link_layer);
  }
  r = caml_alloc_tuple(4);
  Store_field(r, 0, Val_bool(out.state == PORT_ACTIVE));
  Store_field(r, 1, link);
  Store_field(r, 2, Val_int(out.lid));
  Store_field(
      r, 3,
      Val_long(out.active_mtu < MTU_256 ? 0
                                        : 256L << (out.active_mtu - MTU_256)));
  CAMLreturn(r);
}

/* A registered region: [(handle, lkey, rkey)]. */
static value registered(uint32_t handle, uint32_t lkey, uint32_t rkey) {
  CAMLparam0();
  CAMLlocal1(r);
  r = caml_alloc_tuple(3);
  Store_field(r, 0, Val_long(handle));
  Store_field(r, 1, Val_long(lkey));
  Store_field(r, 2, Val_long(rkey));
  CAMLreturn(r);
}

/* Registers the [v_n] bytes at [v_at] for the access code [v_access], in
   the protection domain [v_pd]. The NIC knows them by their address. */
value caml_rig_mlx5_uverbs_reg_mr(value v_fd, value v_driver, value v_pd,
                                  value v_at, value v_n, value v_access) {
  struct ib_uverbs_reg_mr in = {.start = Long_val(v_at),
                                .length = Long_val(v_n),
                                .hca_va = Long_val(v_at),
                                .pd_handle = Int_val(v_pd),
                                .access_flags = access_flags(v_access)};
  struct ib_uverbs_reg_mr_resp out = {0};
  check(WRITE(Int_val(v_fd), Int_val(v_driver), IB_USER_VERBS_CMD_REG_MR, in,
              out),
        "ioctl");
  return registered(out.mr_handle, out.lkey, out.rkey);
}

value caml_rig_mlx5_uverbs_reg_mr_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_mlx5_uverbs_reg_mr(argv[0], argv[1], argv[2], argv[3],
                                     argv[4], argv[5]);
}

/* Registers the [v_n] bytes at offset [v_off] of the dma-buf [v_dmabuf],
   which the NIC knows by their offset. */
value caml_rig_mlx5_uverbs_reg_dmabuf(value v_fd, value v_driver, value v_pd,
                                      value v_dmabuf, value v_off, value v_n,
                                      value v_access) {
  call_t c;
  uint32_t lkey = 0, rkey = 0;
  uint32_t dmabuf = Int_val(v_dmabuf), access = access_flags(v_access);
  start(&c, Int_val(v_driver), UVERBS_OBJECT_MR, UVERBS_METHOD_REG_DMABUF_MR);
  struct ib_uverbs_attr *handle =
      attr(&c, UVERBS_ATTR_REG_DMABUF_MR_HANDLE, 0, 0);
  attr(&c, UVERBS_ATTR_REG_DMABUF_MR_PD_HANDLE, 0, Int_val(v_pd));
  attr(&c, UVERBS_ATTR_REG_DMABUF_MR_OFFSET, 8, Long_val(v_off));
  attr(&c, UVERBS_ATTR_REG_DMABUF_MR_LENGTH, 8, Long_val(v_n));
  attr(&c, UVERBS_ATTR_REG_DMABUF_MR_IOVA, 8, Long_val(v_off));
  bytes_in(&c, UVERBS_ATTR_REG_DMABUF_MR_FD, &dmabuf, sizeof dmabuf);
  bytes_in(&c, UVERBS_ATTR_REG_DMABUF_MR_ACCESS_FLAGS, &access, sizeof access);
  bytes_out(&c, UVERBS_ATTR_REG_DMABUF_MR_RESP_LKEY, &lkey, sizeof lkey);
  bytes_out(&c, UVERBS_ATTR_REG_DMABUF_MR_RESP_RKEY, &rkey, sizeof rkey);
  check(run(Int_val(v_fd), &c), "ioctl");
  return registered(handle->data, lkey, rkey);
}

value caml_rig_mlx5_uverbs_reg_dmabuf_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_mlx5_uverbs_reg_dmabuf(argv[0], argv[1], argv[2], argv[3],
                                         argv[4], argv[5], argv[6]);
}

/* Makes a completion queue of [v_entries] entries, completing into the
   channel [v_channel] with events that carry [v_tag], giving the driver
   [v_d]: [(handle, answer)]. */
value caml_rig_mlx5_uverbs_create_cq(value v_fd, value v_driver,
                                     value v_channel, value v_entries,
                                     value v_tag, value v_d, value v_n) {
  CAMLparam2(v_d, v_n);
  CAMLlocal2(answer, r);
  struct ib_uverbs_create_cq in = {.user_handle = Long_val(v_tag),
                                   .cqe = Int_val(v_entries) - 1,
                                   .comp_channel = Int_val(v_channel)};
  struct ib_uverbs_create_cq_resp out = {0};
  uhw_t u = uhw_of(v_d, v_n);
  int e =
      invoke(Int_val(v_fd), Int_val(v_driver), IB_USER_VERBS_CMD_CREATE_CQ, &in,
             sizeof in, &out, sizeof out, u.in, u.in_len, u.out, u.out_len);
  answer = uhw_answer(&u);
  check(e, "ioctl");
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_long(out.cq_handle));
  Store_field(r, 1, answer);
  CAMLreturn(r);
}

value caml_rig_mlx5_uverbs_create_cq_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_mlx5_uverbs_create_cq(argv[0], argv[1], argv[2], argv[3],
                                        argv[4], argv[5], argv[6]);
}

/* Makes a reliable-connected queue pair of [v_entries] send entries and no
   receive ring, in the protection domain [v_pd], completing into the queue
   [v_cq] with events that carry [v_tag], giving the driver [v_d]:
   [(handle, number, answer)]. */
value caml_rig_mlx5_uverbs_create_qp(value v_fd, value v_driver, value v_pd,
                                     value v_cq, value v_entries, value v_tag,
                                     value v_d, value v_n) {
  CAMLparam2(v_d, v_n);
  CAMLlocal2(answer, r);
  struct ib_uverbs_create_qp in = {.user_handle = Long_val(v_tag),
                                   .pd_handle = Int_val(v_pd),
                                   .send_cq_handle = Int_val(v_cq),
                                   .recv_cq_handle = Int_val(v_cq),
                                   .max_send_wr = Int_val(v_entries),
                                   .max_send_sge = 1,
                                   .qp_type = IB_UVERBS_QPT_RC};
  struct ib_uverbs_create_qp_resp out = {0};
  uhw_t u = uhw_of(v_d, v_n);
  int e =
      invoke(Int_val(v_fd), Int_val(v_driver), IB_USER_VERBS_CMD_CREATE_QP, &in,
             sizeof in, &out, sizeof out, u.in, u.in_len, u.out, u.out_len);
  answer = uhw_answer(&u);
  check(e, "ioctl");
  r = caml_alloc_tuple(3);
  Store_field(r, 0, Val_long(out.qp_handle));
  Store_field(r, 1, Val_long(out.qpn));
  Store_field(r, 2, answer);
  CAMLreturn(r);
}

value caml_rig_mlx5_uverbs_create_qp_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_mlx5_uverbs_create_qp(argv[0], argv[1], argv[2], argv[3],
                                        argv[4], argv[5], argv[6], argv[7]);
}

/* Changes the state of the queue pair [v_h] as the Path.change [v_t] says:
   [Init] an immediate, [Rtr] and [Rts] blocks of their fields in order. */
value caml_rig_mlx5_uverbs_modify_qp(value v_fd, value v_driver, value v_h,
                                     value v_t) {
  struct ib_uverbs_modify_qp in = {.qp_handle = Int_val(v_h)};
  if (Is_long(v_t)) {
    in.qp_state = QPS_INIT;
    in.pkey_index = 0;
    in.port_num = PORT;
    in.qp_access_flags =
        IB_UVERBS_ACCESS_REMOTE_WRITE | IB_UVERBS_ACCESS_REMOTE_READ;
    in.attr_mask = QP_STATE | QP_PKEY_INDEX | QP_PORT | QP_ACCESS_FLAGS;
  } else if (Tag_val(v_t) == 0) {
    /* Rtr { qp; psn; served; mtu; dest }, dest Lid of int or Gid of
       string * int, the global identifier and the index it sends from. */
    value dest = Field(v_t, 4);
    in.qp_state = QPS_RTR;
    in.dest_qp_num = Int_val(Field(v_t, 0));
    in.rq_psn = Int_val(Field(v_t, 1));
    in.max_dest_rd_atomic = Int_val(Field(v_t, 2));
    in.path_mtu = MTU_256;
    for (long b = Long_val(Field(v_t, 3)); b > 256; b /= 2) in.path_mtu++;
    in.min_rnr_timer = RNR_TIMER;
    in.dest.port_num = PORT;
    if (Tag_val(dest) == 0)
      in.dest.dlid = Int_val(Field(dest, 0));
    else {
      in.dest.is_global = 1;
      memcpy(in.dest.dgid, String_val(Field(dest, 0)), sizeof in.dest.dgid);
      in.dest.sgid_index = Int_val(Field(dest, 1));
      in.dest.hop_limit = HOP_LIMIT;
    }
    in.attr_mask = QP_STATE | QP_AV | QP_PATH_MTU | QP_DEST_QPN | QP_RQ_PSN |
                   QP_MAX_DEST_RD_ATOMIC | QP_MIN_RNR_TIMER;
  } else {
    /* Rts { psn; timeout; retries; reads } */
    in.qp_state = QPS_RTS;
    in.sq_psn = Int_val(Field(v_t, 0));
    in.timeout = Int_val(Field(v_t, 1));
    in.retry_cnt = Int_val(Field(v_t, 2));
    in.rnr_retry = RNR_RETRIES;
    in.max_rd_atomic = Int_val(Field(v_t, 3));
    in.attr_mask = QP_STATE | QP_TIMEOUT | QP_RETRY_CNT | QP_RNR_RETRY |
                   QP_SQ_PSN | QP_MAX_QP_RD_ATOMIC;
  }
  check(WRITE_IN(Int_val(v_fd), Int_val(v_driver), IB_USER_VERBS_CMD_MODIFY_QP,
                 in),
        "ioctl");
  return Val_unit;
}

/* Destroys the object of handle [v_h] and kind code [v_kind]: a region, a
   completion queue or a queue pair. */
value caml_rig_mlx5_uverbs_destroy(value v_fd, value v_driver, value v_kind,
                                   value v_h) {
  int fd = Int_val(v_fd), driver = Int_val(v_driver), e;
  uint32_t h = Int_val(v_h);
  switch (Int_val(v_kind)) {
    case 0: {
      struct ib_uverbs_dereg_mr in = {.mr_handle = h};
      e = WRITE_IN(fd, driver, IB_USER_VERBS_CMD_DEREG_MR, in);
      break;
    }
    case 1: {
      struct ib_uverbs_destroy_cq in = {.cq_handle = h};
      struct ib_uverbs_destroy_cq_resp out;
      e = WRITE(fd, driver, IB_USER_VERBS_CMD_DESTROY_CQ, in, out);
      break;
    }
    default: {
      struct ib_uverbs_destroy_qp in = {.qp_handle = h};
      struct ib_uverbs_destroy_qp_resp out;
      e = WRITE(fd, driver, IB_USER_VERBS_CMD_DESTROY_QP, in, out);
      break;
    }
  }
  check(e, "ioctl");
  return Val_unit;
}

/* Events */

/* Prepends [(a, b)] to [list]. */
static value cons_pair(value a, value b, value list) {
  CAMLparam3(a, b, list);
  CAMLlocal2(pair, cell);
  pair = caml_alloc_tuple(2);
  Store_field(pair, 0, a);
  Store_field(pair, 1, b);
  cell = caml_alloc_small(2, Tag_cons);
  Field(cell, 0) = pair;
  Field(cell, 1) = list;
  CAMLreturn(cell);
}

/* The asynchronous events [v_fd] holds, newest first: each Path.async and
   the element it is about. Events of other kinds are left out. */
value caml_rig_mlx5_uverbs_async_events(value v_fd) {
  CAMLparam0();
  CAMLlocal1(list);
  struct ib_uverbs_async_event_desc buf[64];
  list = Val_emptylist;
  ssize_t n;
  while ((n = read(Int_val(v_fd), buf, sizeof buf)) > 0) {
    for (ssize_t i = 0; i < n / (ssize_t)sizeof buf[0]; i++)
      for (size_t k = 0; k < sizeof events / sizeof events[0]; k++)
        if (buf[i].event_type == events[k])
          list = cons_pair(Val_int(k), Val_long(buf[i].element), list);
  }
  CAMLreturn(list);
}

/* The completion queues whose events [v_fd] holds, by their tags, newest
   first. */
value caml_rig_mlx5_uverbs_comp_events(value v_fd) {
  CAMLparam0();
  CAMLlocal2(list, cell);
  struct ib_uverbs_comp_event_desc buf[64];
  list = Val_emptylist;
  ssize_t n;
  while ((n = read(Int_val(v_fd), buf, sizeof buf)) > 0) {
    for (ssize_t i = 0; i < n / (ssize_t)sizeof buf[0]; i++) {
      cell = caml_alloc_small(2, Tag_cons);
      Field(cell, 0) = Val_long(buf[i].cq_handle);
      Field(cell, 1) = list;
      list = cell;
    }
  }
  CAMLreturn(list);
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

value caml_rig_mlx5_uverbs_page_size(value unit) {
  (void)unit;
  return Val_long(sysconf(_SC_PAGESIZE));
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

#else

#define NONE(name, ...)                                                   \
  value caml_rig_mlx5_uverbs_##name(__VA_ARGS__) {                        \
    caml_unix_error(ENOSYS, #name, Nothing);                             \
  }

#define UNUSED __attribute__((unused))
NONE(open, value a UNUSED)
NONE(close, value a UNUSED)
NONE(driver, value a UNUSED)
NONE(get_context, value a UNUSED, value b UNUSED, value c UNUSED,
     value d UNUSED)
NONE(comp_channel, value a UNUSED, value b UNUSED)
NONE(alloc_pd, value a UNUSED, value b UNUSED)
NONE(query_device, value a UNUSED, value b UNUSED)
NONE(query_port, value a UNUSED, value b UNUSED)
NONE(reg_mr_byte, value *a UNUSED, int b UNUSED)
NONE(reg_mr, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED,
     value e UNUSED, value f UNUSED)
NONE(reg_dmabuf_byte, value *a UNUSED, int b UNUSED)
NONE(reg_dmabuf, value a UNUSED, value b UNUSED, value c UNUSED,
     value d UNUSED, value e UNUSED, value f UNUSED, value g UNUSED)
NONE(create_cq_byte, value *a UNUSED, int b UNUSED)
NONE(create_cq, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED,
     value e UNUSED, value f UNUSED, value g UNUSED)
NONE(create_qp_byte, value *a UNUSED, int b UNUSED)
NONE(create_qp, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED,
     value e UNUSED, value f UNUSED, value g UNUSED, value h UNUSED)
NONE(modify_qp, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED)
NONE(destroy, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED)
NONE(async_events, value a UNUSED)
NONE(comp_events, value a UNUSED)
NONE(map, value a UNUSED, value b UNUSED, value c UNUSED)
NONE(unmap, value a UNUSED, value b UNUSED)
NONE(page_size, value a UNUSED)
NONE(poll, value a UNUSED, value b UNUSED, value c UNUSED)

#endif
