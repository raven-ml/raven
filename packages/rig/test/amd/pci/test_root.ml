(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Driver-less opens of this machine's AMD GPU 0, as root, with the firmware the
   directories of the variable RIG_AMD_PCI_FIRMWARE hold, separated by [:]. A
   session on a host whose GPU is detached runs this suite; it skips
   elsewhere. *)

open Windtrap

let strf = Printf.sprintf
let killed = "--killed"

let firmware () =
  match Sys.getenv_opt "RIG_AMD_PCI_FIRMWARE" with
  | Some dirs when Unix.geteuid () = 0 -> String.split_on_char ':' dirs
  | Some _ -> skip ~reason:"taking a GPU's function needs root" ()
  | None ->
      skip
        ~reason:
          "needs a detached AMD GPU and its firmware (RIG_AMD_PCI_FIRMWARE)"
        ()

let open_ firmware =
  match Rig_gpu_support.opened (Rig_amd_pci.gpu ~firmware 0) with
  | Ok g -> g
  | Error why -> failf "GPU 0 did not open: %s" why

let size = 0x10000

let alloc g kind =
  match Rig_amd.alloc g kind size with
  | Some r -> r
  | None -> fail "a fresh device has no room for 64 KiB"

(* Two domains close one device at once, which rig stops once; the GPU opens
   again. *)
let test_stop_twice () =
  let firmware = firmware () in
  let d =
    match Rig.open_ (Rig_amd_pci.gpu ~firmware 0) with
    | Ok d -> d
    | Error why -> failf "GPU 0 did not open: %s" why
  in
  let ds = List.init 2 (fun _ -> Domain.spawn (fun () -> Rig.close d)) in
  List.iter Domain.join ds;
  equal (option string) (Some "closed") (Rig.lost d);
  Rig_amd.stop (open_ firmware) ~fault:None

(* A device's memory is freed after the GPU it was on opened again: the frees
   write nothing of the new device, whose memory keeps its bytes, Mapped
   memory where the device offers it, and which goes on allocating and stops
   clean. *)
let pattern = '\x5a'

let test_free_after_reopen () =
  let firmware = firmware () in
  let g = open_ firmware in
  let kinds =
    Rig_edge.[ Device; Pinned ]
    @ if (Rig_amd.facts g).mapped then [ Rig_edge.Mapped ] else []
  in
  let rs = List.map (alloc g) kinds in
  Rig_amd.stop g ~fault:None;
  let d =
    match Rig.open_ (Rig_amd_pci.gpu ~firmware 0) with
    | Ok d -> d
    | Error why -> failf "GPU 0 did not open again: %s" why
  in
  let memory = if Rig.offers d Mapped then Rig.Buffer.Mapped else Device in
  let b = Rig.Buffer.create ~memory d size in
  Rig.Buffer.fill b pattern;
  List.iter (Rig_amd.free g) rs;
  let back = Bytes.create size in
  Rig.Buffer.blit_to_bytes b 0 back 0 size;
  equal ~msg:"the new device's bytes" string (String.make size pattern)
    (Bytes.to_string back);
  ignore (Rig.Buffer.free b : bool);
  ignore (Rig.Buffer.free (Rig.Buffer.create ~memory:Pinned d size) : bool);
  Rig.close d;
  Rig_amd.stop (open_ firmware) ~fault:None

(* A child opens the GPU, takes system memory for it and dies by SIGKILL with
   the GPU running: the next open resets the GPU before it boots, and the memory
   the child left goes. *)
let die_holding firmware =
  let g = open_ firmware in
  ignore (alloc g Rig_edge.Pinned);
  Unix.kill (Unix.getpid ()) Sys.sigkill

let test_killed_holder () =
  let firmware = firmware () in
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe
      [| exe; killed; String.concat ":" firmware |]
      Unix.stdin Unix.stdout Unix.stderr
  in
  (match snd (Unix.waitpid [] pid) with
  | WSIGNALED s when s = Sys.sigkill -> ()
  | WEXITED c -> failf "the child exited %d" c
  | WSIGNALED s | WSTOPPED s -> failf "the child ended by signal %d" s);
  let left () =
    Sys.readdir "/dev/hugepages"
    |> Array.exists (String.starts_with ~prefix:(strf "rig-pci-%d-" pid))
  in
  equal ~msg:"the child's memory outlived it" bool true (left ());
  let g = open_ firmware in
  equal ~msg:"the open's reset let the child's memory go" bool false (left ());
  Rig_amd.stop g ~fault:None

(* Each [Machine.at "/"] is a machine of its own, whose devices never peer
   with another's. GPU 1 opens through a second machine once its device on the
   first is closed, where GPUs 0 and 1 reached each other: GPU 0's device does
   not reach it. *)
let test_peer_machines () =
  let firmware = firmware () in
  let m = Rig_pci.Machine.at "/" in
  if require_ok (Rig_amd_pci.count ~machine:m ()) < 2 then
    skip ~reason:"the machine has one AMD GPU" ();
  let open_on machine i =
    match Rig.open_ (Rig_amd_pci.gpu ~machine ~firmware i) with
    | Ok d -> d
    | Error why -> failf "GPU %d did not open: %s" i why
  in
  let a = open_on m 0 in
  Fun.protect ~finally:(fun () -> Rig.close a) @@ fun () ->
  let b = open_on m 1 in
  let peers = Rig.reaches a b in
  Rig.close b;
  if not peers then
    skip ~reason:"GPUs 0 and 1 reach no memory of each other" ();
  let b' = open_on (Rig_pci.Machine.at "/") 1 in
  Fun.protect ~finally:(fun () -> Rig.close b') @@ fun () ->
  equal ~msg:"GPU 0 reaches GPU 1 of another machine" bool false
    (Rig.reaches a b');
  equal ~msg:"GPU 1 reaches GPU 0 of another machine" bool false
    (Rig.reaches b' a)

let () =
  match Sys.argv with
  | [| _; arg; dirs |] when arg = killed ->
      die_holding (String.split_on_char ':' dirs)
  | _ ->
      exit
        (run "rig_amd_pci_root"
           [
             group ~timeout:120. "a detached GPU, as root"
               [
                 test "a device two domains close at once opens again"
                   test_stop_twice;
                 test
                   "memory of a stopped device is freed after its GPU opened \
                    again, and the new device stays whole"
                   test_free_after_reopen;
                 test
                   "a GPU whose holder was killed is reset by the next open \
                    (SIGKILL in a child)"
                   test_killed_holder;
                 test
                   "devices of two machines at one root never reach each \
                    other"
                   test_peer_machines;
               ];
           ])
