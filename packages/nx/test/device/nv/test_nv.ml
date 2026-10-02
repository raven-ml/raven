(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The NVIDIA runtime on a machine without an NVIDIA GPU: opening refuses with a
   message, fixes no interface, and the queries refuse devices of other
   vendors. *)

open Windtrap

(* The machine's NVIDIA display controllers, as Linux lists its PCI
   functions. *)
let linux_gpus () =
  let root = "/sys/bus/pci/devices" in
  let read f file =
    In_channel.with_open_text
      (Filename.concat (Filename.concat root f) file)
      In_channel.input_all
    |> String.trim
  in
  if not (Sys.file_exists root) then 0
  else
    Array.fold_left
      (fun n f ->
        let class_ = read f "class" in
        if
          read f "vendor" = "0x10de"
          && List.exists
               (fun prefix -> String.starts_with ~prefix class_)
               [ "0x03" ]
        then n + 1
        else n)
      0 (Sys.readdir root)

(* Both interfaces number the machine's GPUs, so the kernel driver refuses an
   index past them with their count, whatever GPUs it holds. The test opens no
   GPU, so it runs on a machine with GPUs too. *)
let test_past_the_gpus () =
  if not (Sys.file_exists "/sys/bus/pci") then skip ~reason:"no PCI devices" ();
  let n = Nx_nv_device.count () in
  match Nx_nv_device.get ~interface:Kernel n with
  | Ok _ -> fail "a GPU past the count opened"
  | Error msg ->
      let name = if n = 0 then "NV" else Printf.sprintf "NV:%d" n in
      equal string
        (Printf.sprintf "%s: no GPU %d; there are %d NVIDIA GPUs" name n n)
        msg

let no_gpu () =
  if Nx_nv_device.count () > 0 then
    skip ~reason:"this machine has an NVIDIA GPU" ()

(* Another machine without GPUs: this process serves it on the loopback, then
   stops serving it. *)
let test_other_machine () =
  let key = "a key of the test, long enough" in
  let s =
    Nx_remote_device.listen ~key (Unix.ADDR_INET (Unix.inet_addr_loopback, 0))
  in
  let port =
    match Nx_device_support.Remote_server.address s with
    | Unix.ADDR_INET (_, p) -> p
    | Unix.ADDR_UNIX _ -> assert false
  in
  match Nx_remote_device.connect ~port ~key "127.0.0.1" with
  | Error why -> fail why
  | Ok host ->
      if Nx_nv_device.count ~host () > 0 then
        skip ~reason:"the machine has a GPU" ();
      let named = Printf.sprintf "NV-PCI@127.0.0.1:%d: " port in
      (match Nx_nv_device.get ~host 0 with
      | Ok _ -> fail "a GPU opened"
      | Error msg ->
          is_true ~msg:"the GPU's name starts it"
            (String.starts_with ~prefix:named msg));
      (match Nx_nv_device.get ~host ~interface:Kernel 0 with
      | Ok _ -> fail "a GPU opened"
      | Error msg -> contains ~msg:"over PCI only" ~sub:"over PCI" msg);
      Nx_device_support.Remote_server.stop s;
      let lost = function Nx_device.Lost (d, _) -> d == host | _ -> false in
      raises_match ~msg:"count, the machine gone" lost (fun () ->
          Nx_nv_device.count ~host ());
      raises_match ~msg:"get, the machine gone" lost (fun () ->
          Nx_nv_device.get ~host 1)

let () =
  exit
    (run "nx.nv.device"
       [
         test "without a GPU, an open fails with a message and fixes nothing"
           (fun () ->
             no_gpu ();
             List.iter
               (fun interface ->
                 match Nx_nv_device.get ~interface 0 with
                 | Ok _ -> fail "a GPU opened"
                 | Error msg ->
                     let name =
                       match interface with
                       | Nx_nv_device.Kernel -> "NV: "
                       | Pci -> "NV-PCI: "
                     in
                     is_true ~msg:"the failure names the GPU and its interface"
                       (String.starts_with ~prefix:name msg))
               [ Nx_nv_device.Kernel; Pci; Kernel ];
             is_error (Nx_nv_device.get 0));
         test "the GPUs are the machine's display controllers" (fun () ->
             equal int (linux_gpus ()) (Nx_nv_device.count ()));
         test "an index past the GPUs is refused with their count"
           test_past_the_gpus;
         test "another machine's GPUs are opened over PCI" test_other_machine;
         test "a negative index is refused" (fun () ->
             raises_match (Exn.invalid_arg ~substring:"-1 < 0") (fun () ->
                 Nx_nv_device.get (-1)));
         test "a device of another vendor has no NV channels" (fun () ->
             is_true (Option.is_none (Nx_nv_device.of_device Nx_device.host)));
       ])
