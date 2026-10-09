(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The machine's RDMA devices as their files show them, with no NIC: a tree
   under a temporary directory of this process's own, with two ConnectX
   functions, a Soft-RoCE device (no PCI function) and an Intel NIC, as Linux's
   /sys/class/infiniband lists them. *)

open Windtrap
module U = Rig_mlx5_uverbs

let rec mkdirs d =
  if not (Sys.file_exists d) then begin
    mkdirs (Filename.dirname d);
    Sys.mkdir d 0o755
  end

let rec remove p =
  if Sys.file_exists p then
    if Sys.is_directory p then begin
      Array.iter (fun e -> remove (Filename.concat p e)) (Sys.readdir p);
      Sys.rmdir p
    end
    else Sys.remove p

(* A directory of this run's own, removed at its exit. *)
let temp () =
  let d = Filename.temp_dir "rig-mlx5" "" in
  at_exit (fun () -> remove d);
  d

let write path s =
  mkdirs (Filename.dirname path);
  Out_channel.with_open_bin path (fun oc -> output_string oc s)

let device root name ?driver ?bus () =
  let dir = Filename.concat root ("sys/class/infiniband/" ^ name) in
  mkdirs dir;
  match driver with
  | None -> ()
  | Some d ->
      let bus = Option.value ~default:"0000:17:00.0" bus in
      write
        (Filename.concat dir "device/uevent")
        (Printf.sprintf "DRIVER=%s\nPCI_CLASS=20700\nPCI_SLOT_NAME=%s\n" d bus)

let verbs root n name =
  write
    (Filename.concat root
       (Printf.sprintf "sys/class/infiniband_verbs/uverbs%d/ibdev" n))
    (name ^ "\n")

let machine =
  fixture (fun () ->
      let root = temp () in
      device root "mlx5_1" ~driver:"mlx5_core" ~bus:"0000:2a:00.0" ();
      device root "mlx5_0" ~driver:"mlx5_core" ();
      device root "rxe0" ();
      device root "irdma0" ~driver:"irdma" ();
      verbs root 0 "mlx5_0";
      verbs root 1 "mlx5_1";
      verbs root 2 "rxe0";
      root)

let listing =
  group "listing"
    [
      test "the devices the mlx5 driver holds, in name order" (fun () ->
          equal (list string) [ "mlx5_0"; "mlx5_1" ]
            (U.names ~root:(machine ()) ()));
      test "a machine with no RDMA devices lists none" (fun () ->
          equal (list string) [] (U.names ~root:(temp ()) ()));
    ]

let opening =
  let refused name sub =
    test name (fun () ->
        match U.open_ ~root:(machine ()) sub.(0) with
        | Ok _ -> fail "opened"
        | Error e -> contains ~sub:sub.(1) e)
  in
  group "opening"
    [
      refused "a device that does not exist"
        [| "mlx5_7"; "mlx5_7: no RDMA device" |];
      refused "a device of another driver"
        [| "irdma0"; "irdma0: the mlx5_core driver does not hold it" |];
      refused "a device of no PCI function"
        [| "rxe0"; "the mlx5_core driver does not hold it" |];
      test "a verbs file that cannot be opened is named" (fun () ->
          let root = machine () in
          match U.open_ ~root "mlx5_0" with
          | Ok _ -> fail "opened"
          | Error e ->
              starts_with
                ~affix:(Filename.concat root "dev/infiniband/uverbs0: ")
                e);
    ]

let () = exit (run "rig_mlx5_uverbs" [ listing; opening ])
