(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The stubs of the uverbs path, which pack each verbs call in the structures of
   Linux's rdma headers, and the values they answer. A call takes the verbs
   file's descriptor and the kernel's id of the device's driver, and raises
   [Unix.Unix_error] if the kernel refuses it. The stubs build the variants
   here. *)

(* A port's link layer. *)
type link = Infiniband | Ethernet | Link of int

(* The asynchronous events the path reports, in the stub's order. *)
type async =
  | Cq_err
  | Qp_fatal
  | Qp_req_err
  | Qp_access_err
  | Device_fatal
  | Port_active
  | Port_err

(* A queue pair's state change as the stub reads it, its fields in order. *)
type dest = Lid of int | Gid of string * int

type change =
  | Init
  | Rtr of { qp : int; psn : int; served : int; mtu : int; dest : dest }
  | Rts of { psn : int; timeout : int; retries : int; reads : int }

external open_raw : string -> int = "caml_rig_mlx5_uverbs_open"

external address :
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t -> int
  = "caml_rig_mlx5_uverbs_address"

external driver_id : string -> int = "caml_rig_mlx5_uverbs_driver"
external close_fd : int -> unit = "caml_rig_mlx5_uverbs_close"

external get_context : int -> int -> string -> int -> int * string
  = "caml_rig_mlx5_uverbs_get_context"

external comp_channel : int -> int -> int = "caml_rig_mlx5_uverbs_comp_channel"
external alloc_pd : int -> int -> int = "caml_rig_mlx5_uverbs_alloc_pd"

external query_device : int -> int -> int * int
  = "caml_rig_mlx5_uverbs_query_device"

external query_port : int -> int -> bool * link * int * int
  = "caml_rig_mlx5_uverbs_query_port"

external reg_mr :
  int -> int -> int -> int -> int -> int -> int * int * int
  = "caml_rig_mlx5_uverbs_reg_mr_byte" "caml_rig_mlx5_uverbs_reg_mr"

external reg_dmabuf :
  int -> int -> int -> int -> int -> int -> int -> int * int * int
  = "caml_rig_mlx5_uverbs_reg_dmabuf_byte" "caml_rig_mlx5_uverbs_reg_dmabuf"

external create_cq :
  int -> int -> int -> int -> int -> string -> int -> int * string
  = "caml_rig_mlx5_uverbs_create_cq_byte" "caml_rig_mlx5_uverbs_create_cq"

external create_qp :
  int ->
  int ->
  int ->
  int ->
  int ->
  int ->
  string ->
  int ->
  int * int * string
  = "caml_rig_mlx5_uverbs_create_qp_byte" "caml_rig_mlx5_uverbs_create_qp"

external modify_qp : int -> int -> int -> change -> unit
  = "caml_rig_mlx5_uverbs_modify_qp"

external destroy_raw : int -> int -> int -> int -> unit
  = "caml_rig_mlx5_uverbs_destroy"

external async_events : int -> (async * int) list
  = "caml_rig_mlx5_uverbs_async_events"

external comp_events : int -> int list = "caml_rig_mlx5_uverbs_comp_events"
external map_raw : int -> int -> int -> int = "caml_rig_mlx5_uverbs_map"
external unmap : int -> int -> unit = "caml_rig_mlx5_uverbs_unmap"
external page_size : unit -> int = "caml_rig_mlx5_uverbs_page_size"
external poll : int -> int -> int -> int = "caml_rig_mlx5_uverbs_poll"
