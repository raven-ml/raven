(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPUs numbered on machines that fixture trees show, and the opens that answer
   before a GPU is touched. *)

open Windtrap

let strf = Printf.sprintf

module Tree = Rig_pci_support.Tree

(* A machine whose NVIDIA GPUs are a display controller and a 3D controller,
   beside an NVIDIA audio function and another vendor's GPU, listed out of bus
   order. *)
let machine () =
  let fn bus vendor class_ = { (Tree.gpu bus) with vendor; class_ } in
  Tree.make
    [
      fn "0000:83:00.0" 0x10de 0x030200;
      fn "0000:03:00.1" 0x10de 0x040300;
      fn "0000:03:00.0" 0x10de 0x030000;
      fn "0000:01:00.0" 0x1002 0x030000;
    ]

let numbering =
  group "numbering"
    [
      test "GPUs are NVIDIA's display and 3D controllers in bus order"
        (fun () ->
          let machine = Rig_pci.Machine.at (machine ()) in
          equal int 2 (Rig_nv_pci.count ~machine ()));
      test "a machine with no PCI functions has no GPU" (fun () ->
          let machine = Rig_pci.Machine.at (Tree.make []) in
          equal int 0 (Rig_nv_pci.count ~machine ()));
      cases ~name:fst "GPUs are named by number"
        [
          ("0", (0, "NV-PCI")); ("1", (1, "NV-PCI:1")); ("12", (12, "NV-PCI:12"));
        ]
        (fun (_, (i, name)) -> equal string name (Rig_nv_pci.device_name i));
    ]

let opening =
  group "opening"
    [
      test "an open past the last GPU names the GPU and the count" (fun () ->
          let machine = Rig_pci.Machine.at (machine ()) in
          match Rig_nv_pci.open_ ~machine ~firmware:[] 2 with
          | Ok _ -> fail "GPU 2 opened"
          | Error why ->
              equal string "NV-PCI:2: no such GPU; the machine has 2" why);
      cases ~name:fst "a negative GPU number raises"
        [
          ("device_name", fun () -> ignore (Rig_nv_pci.device_name (-1)));
          ("open_", fun () -> ignore (Rig_nv_pci.open_ ~firmware:[] (-1)));
          ("detach", fun () -> ignore (Rig_nv_pci.detach (-1)));
          ("attach", fun () -> ignore (Rig_nv_pci.attach (-1)));
          ("reset", fun () -> ignore (Rig_nv_pci.reset (-1)));
        ]
        (fun (_, f) -> raises_match Exn.invalid_arg f);
    ]

(* Stopping *)

let gpus () =
  Rig_pci.Gpus.make ~memory_bar:1
    ~nodes:(fun ~root:_ _ -> [])
    ~unreleased:(fun ~root:_ _ -> None)
    ~teardown_ms:0 ~reset:Rig_pci.Function.reset
    (fun (id : Rig_pci.Machine.id) ->
      Rig_nv.is_gpu ~vendor:id.vendor ~class_:id.class_)

(* [stopped gpu] opens [gpu], runs [before], and gives the GPU up, recording the
   releases that preceded each unload. *)
let stopped ?(before = ignore) (gpu : Rig_nv_pci_support.gpu) =
  let g = gpus () in
  let unloads = ref [] in
  let hold, fn =
    match
      Rig_pci.Gpus.open_ g gpu.machine 0 ~at_exit:ignore (fun h fn ->
          Ok (h, fn))
    with
    | Ok v -> v
    | Error why -> failf "GPU 0 did not open: %s" why
  in
  before ();
  let s =
    Rig_nv_pci.give_up hold fn ~unload:(fun () ->
        unloads := !(gpu.released) :: !unloads)
  in
  (g, s, !unloads)

let reopen g (gpu : Rig_nv_pci_support.gpu) =
  Rig_pci.Gpus.open_ g gpu.machine 0 ~at_exit:ignore (fun _ _ -> Ok ())

let state =
  Testable.contramap
    (function `Stopped -> "`Stopped" | `Unknown -> "`Unknown")
    string

let stopping =
  group "stopping"
    [
      test
        "a stop unloads the GPU before giving it back, and the GPU opens again"
        (fun () ->
          let gpu = Rig_nv_pci_support.gpu () in
          let g, s, unloads = stopped gpu in
          equal state `Stopped s;
          equal (list int) ~msg:"releases before each unload" [ 0 ] unloads;
          equal int ~msg:"releases" 1 !(gpu.released);
          require_ok (reopen g gpu));
      test
        "a stop on a failed machine gives the GPU back unknown, unloading \
         nothing, and loses it" (fun () ->
          let gpu = Rig_nv_pci_support.gpu () in
          let before () = Rig_pci_support.break gpu.far in
          let g, s, unloads = stopped ~before gpu in
          equal state `Unknown s;
          equal (list int) ~msg:"unloads" [] unloads;
          equal int ~msg:"releases" 1 !(gpu.released);
          contains ~sub:"was lost" (require_error (reopen g gpu)));
      test
        "a stop of a GPU that left the bus gives it back unknown, unloading \
         nothing, and loses it" (fun () ->
          let gone = ref false in
          let gpu =
            Rig_nv_pci_support.gpu
              ~vendor:(fun () -> if !gone then 0xffff else 0x10de)
              ()
          in
          let before () = gone := true in
          let g, s, unloads = stopped ~before gpu in
          equal state `Unknown s;
          equal (list int) ~msg:"unloads" [] unloads;
          equal int ~msg:"releases" 1 !(gpu.released);
          contains ~sub:"was lost" (require_error (reopen g gpu)));
    ]

(* Booted GPUs

   A GPU whose GSP runs, its protected region of memory (WPR2) up, is reset by
   an open when no process holds it, however the last one let go of it, and is
   refused while one does. A child of the suite takes the GPU of a fixture tree,
   whose BAR 0 reads an AD102 with WPR2 up. The tree's reset file records a
   reset and clears nothing, so the open answers that the firmware outlived
   it. *)

let gpu_bus = "0000:03:00.0"
let taking = "--take"

let fixture () =
  if not Rig_pci_support.on_linux then
    skip ~reason:"flock on a function's file needs Linux" ();
  let root =
    Tree.make [ { (Tree.gpu gpu_bus) with vendor = 0x10de; class_ = 0x030000 } ]
  in
  let device = strf "sys/bus/pci/devices/%s/" gpu_bus in
  let set32 r x =
    let fd =
      Unix.openfile (Filename.concat root (device ^ "resource0")) [ O_WRONLY ] 0
    in
    let b = Bytes.create 4 in
    Bytes.set_int32_le b 0 (Int32.of_int x);
    ignore (Unix.lseek fd r SEEK_SET);
    ignore (Unix.write fd b 0 4);
    Unix.close fd
  in
  (* NV_PMC_BOOT_42: architecture 0x19 (Ada), implementation 2. *)
  set32 0xa00 ((0x19 lsl 24) lor (2 lsl 20));
  (* NV_PFB_PRI_MMU_WPR2_ADDR_HI *)
  set32 0x1fa828 0x7ff;
  Tree.add root (device ^ "reset") "";
  root

let resets root =
  let file =
    Filename.concat root (strf "sys/bus/pci/devices/%s/reset" gpu_bus)
  in
  In_channel.with_open_bin file In_channel.input_all

(* With [how] ["stop"] the child gives the GPU up as a device's stop does and
   exits; with ["kill"] it dies by SIGKILL holding it, having allocated nothing;
   with ["hold"] it says "holding" and holds it until its standard input
   closes. *)
let take how root =
  let m = Rig_pci.Machine.at root in
  match how with
  | "stop" -> (
      match
        Rig_pci.Gpus.open_ (gpus ()) m 0 ~at_exit:ignore (fun h fn ->
            Ok (h, fn))
      with
      | Ok (h, fn) ->
          ignore (Rig_nv_pci.give_up h fn ~unload:ignore);
          exit 0
      | Error why ->
          prerr_endline why;
          exit 2)
  | _ -> (
      match Rig_pci.Function.take m gpu_bus with
      | Error why ->
          prerr_endline why;
          exit 2
      | Ok _ when how = "kill" -> Unix.kill (Unix.getpid ()) Sys.sigkill
      | Ok _ ->
          print_endline "holding";
          ignore (In_channel.input_all stdin);
          exit 0)

(* [child how root] starts [take how root]: its pid, the pipe on its standard
   input and its standard output. *)
let child how root =
  let exe = Sys.executable_name in
  let input, feed = Unix.pipe ~cloexec:true () in
  let said, output = Unix.pipe ~cloexec:true () in
  let pid =
    Unix.create_process exe
      [| exe; taking; how; root |]
      input output Unix.stderr
  in
  Unix.close input;
  Unix.close output;
  (pid, feed, said)

let ended pid =
  let status = ref None in
  let exited () =
    match Unix.waitpid [ WNOHANG ] pid with
    | 0, _ -> false
    | _, s ->
        status := Some s;
        true
  in
  if not (Rig_pci_support.poll exited) then begin
    Unix.kill pid Sys.sigkill;
    failf "the child %d did not exit" pid
  end;
  Option.get !status

let pp_status ppf = function
  | Unix.WEXITED c -> Format.fprintf ppf "exited %d" c
  | WSIGNALED s -> Format.fprintf ppf "killed by signal %d" s
  | WSTOPPED s -> Format.fprintf ppf "stopped by signal %d" s

let status = Testable.make ~pp:pp_status ~equal:( = )

(* The open after the child let go of the GPU [how] resets it. *)
let test_left how expected () =
  let root = fixture () in
  let pid, feed, said = child how root in
  Unix.close feed;
  Unix.close said;
  equal ~msg:"the child" status expected (ended pid);
  let machine = Rig_pci.Machine.at root in
  let why = require_error (Rig_nv_pci.open_ ~machine ~firmware:[] 0) in
  equal ~msg:"reset" string "1" (resets root);
  contains ~msg:"the firmware outlived the reset" ~sub:"after its reset" why

let test_held () =
  let root = fixture () in
  let pid, feed, said = child "hold" root in
  let line = In_channel.input_line (Unix.in_channel_of_descr said) in
  Fun.protect
    ~finally:(fun () ->
      Unix.close feed;
      ignore (ended pid))
    (fun () ->
      equal ~msg:"the child holds" (option string) (Some "holding") line;
      let machine = Rig_pci.Machine.at root in
      let why = require_error (Rig_nv_pci.open_ ~machine ~firmware:[] 0) in
      contains ~msg:"the take's reason" ~sub:"taken already" why;
      equal ~msg:"no reset" string "" (resets root))

let booted =
  group ~timeout:Rig_pci_support.patience "booted GPUs"
    [
      test "a GPU stopped by a process that exited is reset by the next open"
        (test_left "stop" (WEXITED 0));
      test
        "a GPU whose holder died before allocating memory is reset by the next \
         open (SIGKILL in a child)"
        (test_left "kill" (WSIGNALED Sys.sigkill));
      test "a GPU another process holds is refused, never reset (a child holds)"
        test_held;
    ]

let () =
  match Sys.argv with
  | [| _; arg; how; root |] when arg = taking -> take how root
  | _ -> exit (run "rig_nv_pci" [ numbering; opening; stopping; booted ])
