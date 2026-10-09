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
          equal (list string)
            [ "0000:03:00.0"; "0000:83:00.0" ]
            (Rig_nv_pci.buses ~machine ());
          equal int 2 (Rig_nv_pci.count ~machine ()));
      test "a machine with no PCI functions has no GPU" (fun () ->
          let machine = Rig_pci.Machine.at (Tree.make []) in
          equal (list string) [] (Rig_nv_pci.buses ~machine ());
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

(* The pinned firmware, from the directories the variable RIG_NV_PCI_FIRMWARE
   lists, separated by [:]: an open reaches the GPU's boot only with them. *)
let firmware () =
  match Sys.getenv_opt "RIG_NV_PCI_FIRMWARE" with
  | Some dirs -> String.split_on_char ':' dirs
  | None -> skip ~reason:"needs NVIDIA's firmware (RIG_NV_PCI_FIRMWARE)" ()

(* Boot reports *)

let rom name =
  In_channel.with_open_bin
    (Filename.concat "fixtures" name)
    In_channel.input_all

(* NV_PMC_BOOT_42: the architecture in bits 29:24, the implementation in bits
   23:20; GA100, AD100 and GB200 are architectures 0x17, 0x19 and 0x1b
   (nv_ref.h). *)
let boot42 ~arch ~impl = (arch lsl 24) lor (impl lsl 20)

let found =
  Testable.contramap
    (fun (i : Rig_nv_pci.image) -> (i.file, i.found))
    (pair string (option string))

(* The files the preamble lists for each family. *)
let gsp = "nvidia/ga102/gsp/gsp-570.144.bin"

let files = function
  | `Ampere ->
      [
        gsp;
        "nvidia/ga102/gsp/bootloader-570.144.bin";
        "nvidia/ga102/gsp/booter_load-570.144.bin";
      ]
  | `Ada ->
      [
        gsp;
        "nvidia/ad102/gsp/bootloader-570.144.bin";
        "nvidia/ad102/gsp/booter_load-570.144.bin";
      ]
  | `Blackwell ->
      [
        gsp;
        "nvidia/gb202/gsp/bootloader-570.144.bin";
        "nvidia/gb202/gsp/fmc-570.144.bin";
      ]

let report ?(firmware = []) ?(vbios = rom "vbios.rom") ~arch ~impl () =
  Rig_nv_pci.report ~firmware ~chip:(boot42 ~arch ~impl) ~vbios

let reports =
  group "boot reports"
    [
      cases "a listed chip is named, with its family's files, none found"
        ~name:(fun (name, _, _, _) -> name)
        [
          ("GA102", 0x17, 2, `Ampere);
          ("GA107", 0x17, 7, `Ampere);
          ("AD102", 0x19, 2, `Ada);
          ("AD104", 0x19, 4, `Ada);
          ("GB202", 0x1b, 2, `Blackwell);
          ("GB207", 0x1b, 7, `Blackwell);
        ]
        (fun (name, arch, impl, family) ->
          let r = require_ok (report ~arch ~impl ()) in
          equal string name r.chip;
          equal (list found)
            (List.map
               (fun file -> { Rig_nv_pci.file; found = None })
               (files family))
            r.images);
      cases "an unlisted chip is an error naming it"
        ~name:(fun (name, _, _, _) -> name)
        [
          ("GA100", 0x17, 0, "GA100");
          ("GA105", 0x17, 5, "GA105");
          ("GB204", 0x1b, 4, "GB204");
          ("GH100", 0x18, 0, "architecture 0x18");
          ("TU102", 0x16, 2, "architecture 0x16");
        ]
        (fun (_, arch, impl, named) ->
          contains ~sub:named (require_error (report ~arch ~impl ())));
      test "only the architecture and implementation name the chip" (fun () ->
          let noise = 0xc00f_ff00 in
          let r =
            require_ok
              (Rig_nv_pci.report ~firmware:[] ~vbios:(rom "vbios.rom")
                 ~chip:(boot42 ~arch:0x19 ~impl:3 lor noise))
          in
          equal string "AD103" r.chip);
      cases "a VBIOS without what FWSEC needs is an error naming it"
        ~name:(fun (name, _) -> name)
        [
          ("vbios_no_bit.rom", "no BIT table");
          ("vbios_debug.rom", "no production FWSEC");
          ("vbios_no_mapper.rom", "no DMEM mapper");
        ]
        (fun (name, why) ->
          contains ~sub:why
            (require_error (report ~vbios:(rom name) ~arch:0x19 ~impl:2 ())));
      (* fixtures/vbios.py puts FWSEC's descriptor at 0x200 of the extension
         image at 1536, 812 bytes, then 512 bytes of code and 1024 of data. *)
      (let needed = 1536 + 0x200 + 812 + 0x200 + 0x400 in
       let full = rom "vbios.rom" in
       prop "a VBIOS cut before FWSEC's end is an error, never read past"
         (Gen.int_range 0 (String.length full))
         (fun n ->
           cover "cut before" (n < needed);
           cover "cut after" (n >= needed);
           let r = report ~vbios:(String.sub full 0 n) ~arch:0x17 ~impl:2 () in
           equal bool (n >= needed) (Result.is_ok r)));
      test "Blackwell reads no VBIOS" (fun () ->
          ignore (require_ok (report ~vbios:"" ~arch:0x1b ~impl:3 ())));
      test "a file with another digest is not found" (fun () ->
          let dir = Filename.temp_dir "rig_nv_pci" "" in
          let rec remove p =
            if Sys.is_directory p then begin
              Array.iter (fun f -> remove (Filename.concat p f)) (Sys.readdir p);
              Sys.rmdir p
            end
            else Sys.remove p
          in
          Fun.protect ~finally:(fun () -> remove dir) @@ fun () ->
          List.iter
            (fun file ->
              let path = Filename.concat dir file in
              let rec mkdir d =
                if not (Sys.file_exists d) then begin
                  mkdir (Filename.dirname d);
                  Sys.mkdir d 0o755
                end
              in
              mkdir (Filename.dirname path);
              Out_channel.with_open_bin path (fun oc ->
                  output_string oc "not NVIDIA's firmware"))
            (files `Ada);
          let r = require_ok (report ~firmware:[ dir ] ~arch:0x19 ~impl:2 ()) in
          equal (list found)
            (List.map
               (fun file -> { Rig_nv_pci.file; found = None })
               (files `Ada))
            r.images);
      test
        "each file is found in the first directory that holds it with its \
         digest (RIG_NV_PCI_FIRMWARE)" (fun () ->
          let dirs = firmware () in
          let r =
            require_ok
              (report ~firmware:("/nonexistent" :: dirs) ~arch:0x19 ~impl:2 ())
          in
          let holds dir file = Sys.file_exists (Filename.concat dir file) in
          List.iter
            (fun (i : Rig_nv_pci.image) ->
              let first = List.find_opt (fun d -> holds d i.file) dirs in
              equal (option string)
                (Option.map (fun d -> Filename.concat d i.file) first)
                i.found)
            r.images);
    ]

(* Stopping *)

let gpus () =
  Rig_pci.Gpus.make ~name:"NV-PCI" ~memory_bar:1
    ~nodes:(fun ~root:_ _ -> [])
    ~unreleased:(fun ~root:_ _ -> None)
    ~teardown_ms:0 ~reset:Rig_pci.Function.reset
    (fun (id : Rig_pci.Machine.id) ->
      Rig_nv.is_gpu ~vendor:id.vendor ~class_:id.class_)

let hold g m =
  Rig_pci.Gpus.open_ g m 0 (fun h _ ->
      Rig_pci.Gpus.set_stop h (fun () -> `Clean);
      Ok h)

let state =
  Testable.contramap
    (function `Stopped -> "`Stopped" | `Unknown -> "`Unknown")
    string

(* Booted GPUs

   A GPU whose GSP runs, its protected region of memory (WPR2) up, is reset by
   an open when no process holds it, however the last one let go of it, and is
   refused while one does. A child of the suite takes the GPU of a fixture tree,
   whose BAR 0 reads an AD102 with WPR2 up. The tree's reset file records a
   reset and clears nothing, so the open answers that the firmware outlived
   it. *)

let gpu_bus = "0000:03:00.0"
let taking = "--take"

let fixture ?(booted = true) () =
  if not Rig_pci_support.on_linux then
    skip ~reason:"flock on a function's file needs Linux" ();
  let root =
    Tree.make [ { (Tree.gpu gpu_bus) with vendor = 0x10de; class_ = 0x030000 } ]
  in
  let device = strf "sys/bus/pci/devices/%s/" gpu_bus in
  let write r s =
    let fd =
      Unix.openfile (Filename.concat root (device ^ "resource0")) [ O_WRONLY ] 0
    in
    ignore (Unix.lseek fd r SEEK_SET);
    ignore (Unix.write_substring fd s 0 (String.length s));
    Unix.close fd
  in
  let set32 r x =
    let b = Bytes.create 4 in
    Bytes.set_int32_le b 0 (Int32.of_int x);
    write r (Bytes.to_string b)
  in
  (* NV_PMC_BOOT_42: architecture 0x19 (Ada), implementation 2. *)
  set32 0xa00 ((0x19 lsl 24) lor (2 lsl 20));
  (* NV_PFB_PRI_MMU_WPR2_ADDR_HI *)
  if booted then set32 0x1fa828 0x7ff
  else begin
    (* The GPU's own boot done: its progress unlocked, then completed. Its VBIOS
       in the PROM window, from 0x300000. *)
    set32 0x118128 1;
    set32 0x118234 0xff;
    write 0x300000 (rom "vbios.rom")
  end;
  Tree.add root (device ^ "reset") "";
  root

let resets root =
  let file =
    Filename.concat root (strf "sys/bus/pci/devices/%s/reset" gpu_bus)
  in
  In_channel.with_open_bin file In_channel.input_all

(* [stop g m] opens GPU 0 of [m] through [g] and stops it. *)
let stop g m = Result.map Rig_pci.Gpus.stop (hold g m)

(* With [how] ["stop"] the child stops the GPU and exits; with ["kill"] it dies
   by SIGKILL holding it, having allocated nothing; with ["hold"] it says
   "holding" and holds it until its standard input closes. *)
let take how root =
  let m = Rig_pci.Machine.at root in
  match how with
  | "stop" -> (
      match stop (gpus ()) m with
      | Ok _ -> exit 0
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

(* The stop gives the GPU back without losing it: this process opens it again,
   and the open resets it. *)
let test_stopped_here () =
  let root = fixture () in
  let g = gpus () and machine = Rig_pci.Machine.at root in
  equal ~msg:"the stop" state `Stopped (require_ok (stop g machine));
  ignore (Rig_pci.Gpus.stop (require_ok (hold g machine)));
  let why = require_error (Rig_nv_pci.open_ ~machine ~firmware:[] 0) in
  equal ~msg:"reset" string "1" (resets root);
  contains ~msg:"the firmware outlived the reset" ~sub:"after its reset" why

(* A GPU whose WPR2 is still up after the open's reset is refused, and given
   back unlost: the next open resets it again. *)
let test_outlived () =
  let root = fixture () in
  let machine = Rig_pci.Machine.at root in
  let opened () =
    let why = require_error (Rig_nv_pci.open_ ~machine ~firmware:[] 0) in
    let reset = resets root in
    Tree.add root (strf "sys/bus/pci/devices/%s/reset" gpu_bus) "";
    (reset, why)
  in
  let reset, why = opened () in
  equal ~msg:"reset" string "1" reset;
  contains ~msg:"refused" ~sub:"still runs the GSP's firmware after its reset"
    why;
  let reset, why = opened () in
  equal ~msg:"reset again" string "1" reset;
  contains ~msg:"refused again" ~sub:"after its reset" why

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

(* A GPU whose GSP is down boots without a reset. Its open fails at the memory
   size its firmware should have written, before the GSP's memory is taken: it
   gives the GPU back as found, so the next open does not reset it either. *)
let test_failed_early () =
  let root = fixture ~booted:false () in
  let firmware = firmware () in
  let machine = Rig_pci.Machine.at root in
  for _ = 1 to 2 do
    let why = require_error (Rig_nv_pci.open_ ~machine ~firmware 0) in
    contains ~msg:"refused" ~sub:"wrote no memory size" why
  done;
  equal ~msg:"no reset" string "" (resets root)

let booted =
  group ~timeout:Rig_pci_support.patience "booted GPUs"
    [
      test "a GPU stopped by a process that exited is reset by the next open"
        (test_left "stop" (WEXITED 0));
      test
        "a GPU whose holder died before allocating memory is reset by the next \
         open (SIGKILL in a child)"
        (test_left "kill" (WSIGNALED Sys.sigkill));
      test "a GPU this process stopped opens again, its open resetting it"
        test_stopped_here;
      test
        "a GPU whose WPR2 is up after the open's reset is refused, and reset \
         again by the next open"
        test_outlived;
      test "a GPU another process holds is refused, never reset (a child holds)"
        test_held;
      test
        "an open that fails before the GSP's memory is taken gives the GPU \
         back unreset (RIG_NV_PCI_FIRMWARE)"
        test_failed_early;
    ]

let () =
  match Sys.argv with
  | [| _; arg; how; root |] when arg = taking -> take how root
  | _ -> exit (run "rig_nv_pci" [ numbering; opening; reports; booted ])
