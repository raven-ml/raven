(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The verbs ioctls the uverbs path makes, with no NIC: each request's header
   and attributes are the bytes a C program compiled against Linux's uapi
   headers (rdma_user_ioctl_cmds.h, ib_user_ioctl_cmds.h, ib_user_verbs.h) packs
   for the same call, filling attributes as rdma-core's cmd_ioctl.h does, every
   one mandatory. The expected bytes are that program's, with the pointers
   zeroed; each pointer must name its attribute's bytes. *)

open Windtrap
module R = Request

external address : R.params -> int = "caml_rig_mlx5_uverbs_address"

let hex p =
  String.concat ""
    (List.init (Bigarray.Array1.dim p) (fun i ->
         Printf.sprintf "%02x" (Char.code (Bigarray.Array1.get p i))))

let attr_size = Defs.Attr.sizeof
let header = Defs.Ioctl_hdr.sizeof

(* [r]'s bytes, each pointer, which must name the bytes [named] gives its
   attribute, zeroed. *)
let packed r named =
  let b =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout
      (Bigarray.Array1.dim (R.bytes r))
  in
  Bigarray.Array1.blit (R.bytes r) b;
  List.iter
    (fun (i, p) ->
      let at = header + (i * attr_size) + fst Defs.Attr.data in
      equal
        ~msg:(Printf.sprintf "attribute %d names its bytes" i)
        int (address p)
        (R.field b (at, 8));
      R.set b (at, 8) 0)
    named;
  hex b

let mlx5 = Defs.rdma_driver_mlx5

let requests =
  group "requests"
    [
      test "the request number is _IOWR(0x1b, 1, header)" (fun () ->
          equal int 0xc0181b01 R.number);
      test "a command with an inline request and an answer" (fun () ->
          let out = R.params Defs.Alloc_pd_resp.sizeof in
          let r =
            R.write ~driver:mlx5 ~cmd:Defs.ib_user_verbs_cmd_alloc_pd
              (R.params Defs.Alloc_pd.sizeof)
              ~out ~uhw:"" ~uhw_out:(R.params 0)
          in
          equal string
            "480000000000030000000000000000000100000000000000020008000100000003000000000000000000080001000000000000000000000001000400010000000000000000000000"
            (packed r [ (2, out) ]));
      test "a command with a request and driver data by address" (fun () ->
          let req = R.params Defs.Create_cq.sizeof in
          let out = R.params Defs.Create_cq_resp.sizeof in
          let uhw_out = R.params 8 in
          let r =
            R.write ~driver:mlx5 ~cmd:Defs.ib_user_verbs_cmd_create_cq req ~out
              ~uhw:(String.make 32 'x') ~uhw_out
          in
          (* The driver data's copy is the request's own: its address is not
             zero. *)
          let b = R.bytes r in
          let uhw_at = header + (3 * attr_size) + fst Defs.Attr.data in
          not_equal ~msg:"the driver data is by address" int 0
            (R.field b (uhw_at, 8));
          R.set b (uhw_at, 8) 0;
          equal string
            "6800000000000500000000000000000001000000000000000200080001000000120000000000000000002000010000000000000000000000010008000100000000000000000000000010200001000000000000000000000001100800010000000000000000000000"
            (packed r [ (1, req); (2, out); (4, uhw_out) ]));
      test "driver data of at most 8 bytes is inline" (fun () ->
          let out = R.params Defs.Get_context_resp.sizeof in
          let uhw_out = R.params 72 in
          let r =
            R.write ~driver:mlx5 ~cmd:Defs.ib_user_verbs_cmd_get_context
              (R.params Defs.Get_context.sizeof)
              ~out ~uhw:"\001\002\003" ~uhw_out
          in
          equal string
            "6800000000000500000000000000000001000000000000000200080001000000000000000000000000000800010000000000000000000000010008000100000000000000000000000010030001000000010203000000000001104800010000000000000000000000"
            (packed r [ (2, out); (4, uhw_out) ]));
      test "a method with objects, values and answers" (fun () ->
          let lkey = R.params 4 and rkey = R.params 4 in
          let r =
            R.call ~driver:mlx5 ~obj:Defs.uverbs_object_mr
              ~meth:Defs.uverbs_method_reg_dmabuf_mr
              [
                Made Defs.uverbs_attr_reg_dmabuf_mr_handle;
                Handle (Defs.uverbs_attr_reg_dmabuf_mr_pd_handle, 3);
                Value (Defs.uverbs_attr_reg_dmabuf_mr_offset, 0x20_0000);
                Value (Defs.uverbs_attr_reg_dmabuf_mr_length, 0x400_0000);
                Value (Defs.uverbs_attr_reg_dmabuf_mr_iova, 0x20_0000);
                Word (Defs.uverbs_attr_reg_dmabuf_mr_fd, 17);
                Word
                  ( Defs.uverbs_attr_reg_dmabuf_mr_access_flags,
                    Defs.ib_uverbs_access_local_write
                    lor Defs.ib_uverbs_access_remote_write
                    lor Defs.ib_uverbs_access_relaxed_ordering );
                Out (Defs.uverbs_attr_reg_dmabuf_mr_resp_lkey, lkey);
                Out (Defs.uverbs_attr_reg_dmabuf_mr_resp_rkey, rkey);
              ]
          in
          equal string
            "a80007000400090000000000000000000100000000000000000000000100000000000000000000000100000001000000030000000000000002000800010000000000200000000000030008000100000000000004000000000400080001000000000020000000000005000400010000001100000000000000060004000100000003001000000000000700040001000000000000000000000008000400010000000000000000000000"
            (packed r [ (7, lkey); (8, rkey) ]));
      test "the kernel's handle of a made object is read from its attribute"
        (fun () ->
          let r =
            R.call ~driver:mlx5 ~obj:Defs.uverbs_object_mr
              ~meth:Defs.uverbs_method_reg_dmabuf_mr
              [ Handle (1, 3); Made 0 ]
          in
          R.set (R.bytes r) (header + attr_size + fst Defs.Attr.data, 8) 0x2a;
          equal int 0x2a (R.made r 0));
      test "a header past a page is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"past a page") (fun () ->
              R.call ~driver:mlx5 ~obj:0 ~meth:0
                (List.init 256 (fun i -> R.Value (i, 0)))));
    ]

let () = exit (run "rig_mlx5_uverbs.request" [ requests ])
