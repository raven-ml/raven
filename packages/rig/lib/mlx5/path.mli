(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What a path gives a NIC.

    The types live here, apart from {!Rig_mlx5} and {!Uverbs}, because both
    use them: {!Uverbs} makes the path, and {!Rig_mlx5} drives the NIC through
    it. [rig_mlx5.mli] says what each type and field is. *)

type memory =
  | Host of Rig_mlx5_abi.buffer
  | Dmabuf of { fd : int; offset : int; bytes : int }

type access = Local | Remote_write | Remote
type order = In_order | Any_order
type address = Lid of int | Gid of string
type endpoint = { qp : int; psn : int; address : address; mtu : int }

type kernel_event =
  | Completed of int
  | Cq_error of int
  | Qp_failed of int * string
  | Nic_failed
  | Port_changed of string

type transition =
  | Init
  | Ready_to_receive of {
      peer : endpoint;
      mtu : int;
      gid : int option;
      served : int;
    }
  | Ready_to_send of { psn : int; timeout : int; retries : int; reads : int }

type context = {
  answer : string;
  port : [ `Infiniband of int | `Ethernet of (int * string) list ];
  mtu : int;
  reads : int;
  served : int;
  cqes : int;
}

type registered = { handle : int; local : int; remote : int }

type t = {
  name : string;
  bus : string;
  page : int;
  context : string -> int -> (context, string) result;
  map : int -> int -> (int, string) result;
  register : access -> order -> memory -> (registered, string) result;
  cq :
    entries:int ->
    tag:int ->
    string ->
    int ->
    (int * (unit -> int) * string, string) result;
  qp :
    cq:int ->
    entries:int ->
    tag:int ->
    string ->
    int ->
    (int * int * string, string) result;
  modify : int -> transition -> (unit, string) result;
  destroy : [ `Region | `Cq | `Qp ] -> int -> (unit, string) result;
  wait : int -> kernel_event list;
  reported : unit -> unit;
  close : unit -> unit;
}
