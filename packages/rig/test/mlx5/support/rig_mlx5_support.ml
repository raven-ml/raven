(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let read path =
  match In_channel.with_open_bin path In_channel.input_all with
  | s -> Some (String.trim s)
  | exception Sys_error _ -> None

(* Whether [name]'s port 1 carries RDMA, from /sys/class/infiniband: the type
   file of an unused identifier cannot be read, and a port whose link layer
   cannot be read is left to opening. *)
let carries_rdma name =
  let port = "/sys/class/infiniband/" ^ name ^ "/ports/1" in
  match read (port ^ "/link_layer") with
  | Some "Ethernet" ->
      let types = port ^ "/gid_attrs/types" in
      let entries = try Sys.readdir types with Sys_error _ -> [||] in
      Array.exists (fun e -> read (types ^ "/" ^ e) = Some "RoCE v2") entries
  | _ -> true

(* [name]'s verbs file, /dev/infiniband/uverbsN for the uverbsN of
   /sys/class/infiniband_verbs whose ibdev is [name], if sysfs lists one and
   it is a character device: a container shows the host's sysfs, and only
   the device files it was given. *)
let verbs_file name =
  let dir = "/sys/class/infiniband_verbs" in
  let entries = try Array.to_list (Sys.readdir dir) with Sys_error _ -> [] in
  let file e =
    if read (Filename.concat dir e ^ "/ibdev") <> Some name then None
    else
      let f = "/dev/infiniband/" ^ e in
      match (Unix.stat f).st_kind with
      | Unix.S_CHR -> Some f
      | _ | (exception Unix.Unix_error _) -> None
  in
  List.find_map file entries

let nic_name () =
  match Rig_mlx5.names () with
  | Error e -> failf "listing the NICs: %s" e
  | Ok [] -> skip ~reason:"no NIC the mlx5 driver holds" ()
  | Ok names -> (
      match List.find_opt carries_rdma names with
      | None ->
          skip
            ~reason:
              (Printf.sprintf
                 "no RDMA on %s: port 1 is Ethernet with no RoCE v2 identifier"
                 (String.concat ", " names))
            ()
      | Some name when verbs_file name = None ->
          skip
            ~reason:
              (Printf.sprintf
                 "the process does not reach %s: no verbs file in \
                  /dev/infiniband"
                 name)
            ()
      | Some name -> name)
