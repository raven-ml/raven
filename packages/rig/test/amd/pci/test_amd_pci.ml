(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let strf = Printf.sprintf

let fixture name =
  In_channel.with_open_bin ("../fixtures/" ^ name) In_channel.input_all

(* The R9700's blocks as the amdgpu driver lists them (../fixtures/r9700.txt):
   hardware ID, instance, version and segment bases. *)
let listing =
  fixture "r9700.txt" |> String.split_on_char '\n'
  |> List.filter (( <> ) "")
  |> List.map (fun line ->
      match String.split_on_char ' ' line with
      | hw :: inst :: version :: _harvest :: bases ->
          let version =
            match List.map int_of_string (String.split_on_char '.' version) with
            | [ a; b; c ] -> (a, b, c)
            | _ -> failwith ("a version in r9700.txt: " ^ line)
          in
          ( int_of_string hw,
            int_of_string inst,
            version,
            Array.of_list (List.map int_of_string bases) )
      | _ -> failwith ("a line of r9700.txt: " ^ line))

let version = triple int int int

let contains ~sub s =
  let n = String.length sub in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = sub || at (i + 1))
  in
  at 0

(* Boot reports *)

(* The blocks a boot programs: GC, SDMA0, MP0, MP1, MMHUB, OSSSYS, NBIF and HDP
   (soc15_hw_ip.h's IDs), with the names amdgpu gives them. *)
let programmed =
  [
    (11, "GC");
    (42, "SDMA0");
    (255, "MP0");
    (1, "MP1");
    (34, "MMHUB");
    (40, "OSSSYS");
    (108, "NBIF");
    (41, "HDP");
  ]

let r9700_files =
  [
    "amdgpu/psp_14_0_3_sos.bin";
    "amdgpu/smu_14_0_3.bin";
    "amdgpu/sdma_7_0_1.bin";
    "amdgpu/gc_12_0_1_pfp.bin";
    "amdgpu/gc_12_0_1_me.bin";
    "amdgpu/gc_12_0_1_mec.bin";
    "amdgpu/gc_12_0_1_imu.bin";
    "amdgpu/gc_12_0_1_rlc.bin";
  ]

let block =
  Testable.contramap
    (fun (b : Rig_amd_pci.block) -> (b.name, b.version, b.instances))
    (triple string version (list int))

let image =
  Testable.contramap
    (fun (i : Rig_amd_pci.image) -> (i.file, i.found))
    (pair string (option string))

(* The programmed blocks, in that order, as amdgpu lists them: the version of
   instance 0 and every instance. *)
let listed_blocks =
  List.map
    (fun (hw, name) ->
      let insts = List.filter (fun (h, _, _, _) -> h = hw) listing in
      let version = List.hd (List.map (fun (_, _, v, _) -> v) insts) in
      {
        Rig_amd_pci.name;
        version;
        instances = List.map (fun (_, i, _, _) -> i) insts;
      })
    programmed

let rec remove path =
  if Sys.is_directory path then begin
    Array.iter (fun f -> remove (Filename.concat path f)) (Sys.readdir path);
    Sys.rmdir path
  end
  else Sys.remove path

let temp_dir () =
  let d = Filename.temp_dir "rig_amd_pci" "" in
  at_exit (fun () -> if Sys.file_exists d then remove d);
  d

let reports =
  group ~timeout:10. "boot reports"
    [
      cases "the R9700's report names its blocks and images" ~id:Fun.id
        [ "r9700.bin"; "r9700_wide.bin" ] (fun name ->
          let r = require_ok (Rig_amd_pci.report ~firmware:[] (fixture name)) in
          equal (list block) listed_blocks r.blocks;
          equal (list image)
            (List.map
               (fun file -> { Rig_amd_pci.file; found = None })
               r9700_files)
            r.images);
      test "a file with another digest is not found" (fun () ->
          let dir = temp_dir () in
          let file = List.hd r9700_files in
          let path = Filename.concat dir file in
          Sys.mkdir (Filename.dirname path) 0o755;
          Out_channel.with_open_bin path (fun oc ->
              output_string oc "not AMD's firmware");
          let r =
            require_ok
              (Rig_amd_pci.report ~firmware:[ dir ] (fixture "r9700.bin"))
          in
          equal (list image)
            (List.map
               (fun file -> { Rig_amd_pci.file; found = None })
               r9700_files)
            r.images);
      test "what is no discovery table is refused" (fun () ->
          ignore (require_error (Rig_amd_pci.report ~firmware:[] "short"));
          let t = Bytes.of_string (fixture "r9700.bin") in
          Bytes.set t 0 '\000';
          ignore
            (require_error
               (Rig_amd_pci.report ~firmware:[] (Bytes.to_string t))));
    ]

(* Tables of other GPUs: the R9700's, each block of die 0 given another version,
   or removed under a hardware ID no block has, its checksums made good again.
   discovery.h lays the binary out with its checksum at byte 8, over the bytes
   from 10 to its size at 10, and the IP table's offset, checksum and size at
   12, 14 and 16. The IP table lists die 0's offset at 16, where its block count
   is at 2 and its blocks start at 4, each a hardware ID at 0, its version at 4
   to 6, and its 32-bit bases after 8. *)

let unnamed = 0x7fff

let patched blocks =
  let t = Bytes.of_string (fixture "r9700.bin") in
  let u16 at = Bytes.get_uint16_le t at in
  let sum first n =
    let s = ref 0 in
    for i = first to first + n - 1 do
      s := !s + Bytes.get_uint8 t i
    done;
    !s land 0xffff
  in
  let ips = u16 12 in
  let die = u16 (ips + 16) in
  let at = ref (die + 4) in
  for _ = 1 to u16 (die + 2) do
    let e = !at in
    (match List.assoc_opt (u16 e) blocks with
    | Some (Some (a, b, c)) ->
        Bytes.set_uint8 t (e + 4) a;
        Bytes.set_uint8 t (e + 5) b;
        Bytes.set_uint8 t (e + 6) c
    | Some None -> Bytes.set_uint16_le t e unnamed
    | None -> ());
    at := e + 8 + (4 * Bytes.get_uint8 t (e + 3))
  done;
  Bytes.set_uint16_le t 14 (sum ips (u16 16));
  Bytes.set_uint16_le t 8 (sum 10 (u16 10 - 10));
  Bytes.to_string t

(* The block versions of other GPUs: GC, NBIO, MMHUB, OSSSYS, HDP, SDMA, MP0 and
   MP1. *)
let blocks gc nbio mmhub osssys hdp sdma mp =
  [
    (11, gc);
    (108, nbio);
    (34, mmhub);
    (40, osssys);
    (41, hdp);
    (42, sdma);
    (255, mp);
    (1, mp);
  ]

let navi31 =
  blocks (11, 0, 0) (4, 3, 0) (3, 0, 0) (6, 0, 0) (6, 0, 0) (6, 0, 0) (13, 0, 0)

let mi300 =
  blocks (9, 4, 3) (7, 9, 0) (1, 8, 0) (4, 4, 2) (4, 4, 2) (4, 4, 2) (13, 0, 6)

let mi350 =
  (11, (9, 5, 0))
  :: (42, (4, 4, 5))
  :: List.remove_assoc 42 (List.remove_assoc 11 mi300)

let report_of blocks =
  Rig_amd_pci.report ~firmware:[]
    (patched (List.map (fun (b, v) -> (b, Some v)) blocks))

let refused =
  group ~timeout:10. "refusals"
    [
      cases
        "a block of a version this library does not boot is refused, naming it"
        ~id:(fun (b, v, _) ->
          strf "block %d at %s" b
            (let a, b, c = v in
             strf "%d.%d.%d" a b c))
        [
          (34, (1, 7, 0), "MMHUB 1.7.0 is a version this library does not boot");
          (34, (2, 0, 0), "MMHUB 2.0.0 is a version this library does not boot");
          (11, (10, 3, 0), "GC 10.3.0 is a version this library does not boot");
          (1, (13, 0, 4), "MP1 13.0.4 is a version this library does not boot");
          ( 255,
            (13, 0, 5),
            "MP0 13.0.5 is a version this library does not boot: no image \
             amdgpu/psp_13_0_5_sos.bin" );
        ]
        (fun (b, v, msg) ->
          equal (result pass string) (Error msg)
            (Result.map ignore (report_of [ (b, v) ])));
      test "a GPU without a block a boot programs is refused" (fun () ->
          equal (result pass string) (Error "the GPU has no OSSSYS block")
            (Result.map ignore
               (Rig_amd_pci.report ~firmware:[] (patched [ (40, None) ]))));
      cases "a GPU of each generation this library boots is reported" ~id:fst
        [ ("Navi 31", navi31); ("MI300", mi300); ("MI350", mi350) ]
        (fun (_, blocks) ->
          let r = require_ok (report_of blocks) in
          equal (list version)
            (List.map (fun (hw, _) -> List.assoc hw blocks) programmed)
            (List.map (fun (b : Rig_amd_pci.block) -> b.version) r.blocks));
    ]

(* Numbering *)

module Tree = Rig_pci_support.Tree

(* A machine whose AMD GPUs are a display controller, a processing accelerator
   and a display controller of another subclass, beside an AMD audio function
   and another vendor's GPU, listed out of bus order. *)
let machine () =
  let fn bus vendor class_ = { (Tree.gpu bus) with vendor; class_ } in
  Tree.make
    [
      fn "0000:83:00.0" 0x1002 0x120000;
      fn "0000:03:00.1" 0x1002 0x040300;
      fn "0000:03:00.0" 0x1002 0x030000;
      fn "0000:01:00.0" 0x10de 0x030000;
      fn "0000:02:00.0" 0x1002 0x038000;
    ]

(* What [call ()] answered, ["refused"] for the refusal of [fn] inside a
   lock. *)
let answer fn call =
  let prefix = strf "Rig_amd_pci.%s: called from" fn in
  match call () with
  | Ok () -> "ok"
  | Error _ -> "error"
  | exception Invalid_argument why when String.starts_with ~prefix why ->
      "refused"

let numbering =
  group "numbering"
    [
      test "GPUs are AMD's display controllers and accelerators in bus order"
        (fun () ->
          let root = machine () in
          equal
            (result (list string) string)
            (Ok [ "0000:02:00.0"; "0000:03:00.0"; "0000:83:00.0" ])
            (Rig_amd_pci.buses ~machine:(Rig_pci.Machine.at root) ());
          equal (result int string) (Ok 3)
            (Rig_amd_pci.count ~machine:(Rig_pci.Machine.at root) ()));
      test "a machine with no PCI functions has no GPU" (fun () ->
          equal (result int string) (Ok 0)
            (Rig_amd_pci.count ~machine:(Rig_pci.Machine.at (Tree.make [])) ()));
      test "a machine whose functions cannot be listed says why" (fun () ->
          let machine = Rig_pci_support.unlisted () in
          equal (result int string) (Error Rig_pci_support.unlisted_why)
            (Rig_amd_pci.count ~machine ());
          equal
            (result (list string) string)
            (Error Rig_pci_support.unlisted_why)
            (Rig_amd_pci.buses ~machine ());
          let why =
            require_error
              (Rig_gpu_support.opened (Rig_amd_pci.gpu ~machine 0))
          in
          equal string ("AMD-PCI: " ^ Rig_pci_support.unlisted_why) why);
      cases "GPUs are named by number" ~id:fst
        [
          ("0", (0, "AMD-PCI"));
          ("1", (1, "AMD-PCI:1"));
          ("12", (12, "AMD-PCI:12"));
        ]
        (fun (_, (i, name)) ->
          equal string name (Rig_amd_pci.gpu i).name);
      test "an open past the last GPU names the GPU and the count" (fun () ->
          let machine = Rig_pci.Machine.at (machine ()) in
          match Rig_gpu_support.opened (Rig_amd_pci.gpu ~machine 3) with
          | Ok _ -> fail "GPU 3 opened"
          | Error why ->
              equal string "AMD-PCI:3: no such function; AMD-PCI drives 3 here" why);
      test "a reset past the last GPU names the GPU and the count" (fun () ->
          let machine = Rig_pci.Machine.at (machine ()) in
          equal (result pass string)
            (Error "AMD-PCI:3: no such function; AMD-PCI drives 3 here")
            (Rig_amd_pci.reset ~machine 3));
      cases "a negative GPU number raises" ~id:fst
        [
          ("gpu", fun () -> ignore (Rig_amd_pci.gpu (-1)));
          ("detach", fun () -> ignore (Rig_amd_pci.detach (-1)));
          ("attach", fun () -> ignore (Rig_amd_pci.attach (-1)));
          ("reset", fun () -> ignore (Rig_amd_pci.reset (-1)));
        ]
        (fun (_, f) -> raises_match Exn.invalid_arg f);
      cases
        "a call inside a lock of its thread's raises naming it, changing \
         nothing"
        ~id:fst
        (let on () = Rig_pci.Machine.at (machine ()) in
         let past () = require_ok (Rig_amd_pci.count ()) in
         [
           ( "gpu",
             fun () ->
               Rig_gpu_support.opened (Rig_amd_pci.gpu ~machine:(on ()) 3)
               |> Result.map ignore );
           ( "detach",
             fun () -> Rig_amd_pci.detach (past ()) |> Result.map ignore );
           ("attach", fun () -> Rig_amd_pci.attach (past ()));
           ("reset", fun () -> Rig_amd_pci.reset ~machine:(on ()) 3);
           ( "report",
             fun () -> Rig_amd_pci.report ~firmware:[] "" |> Result.map ignore
           );
         ])
        (fun (fn, call) ->
          let inside = Rig_support.inside_lock (fun () -> answer fn call) in
          equal ~msg:"inside the lock" string "refused" inside;
          equal ~msg:"once it is given back" string "error" (answer fn call));
    ]

(* Letting go: a fixture tree's unbound AMD GPU at 0000:05:00.0, and what of
   amdgpu's device the kernel still lists for it. amdgpu's release removes its
   node from KFD's topology at its start and its [ip_discovery] directory at its
   end, so either one listed means amdgpu has not let go. *)

let r9700 = "0000:05:00.0"

let unbound () =
  Tree.make [ { (Tree.gpu r9700) with vendor = 0x1002; class_ = 0x030000 } ]

(* The open asks under the function's lock, which needs Linux. *)
let refusal root =
  if not Rig_pci_support.on_linux then
    skip ~reason:"flock on a function's file needs Linux" ();
  match
    Rig_gpu_support.opened
      (Rig_amd_pci.gpu ~machine:(Rig_pci.Machine.at root) 0)
  with
  | Ok _ -> fail "a GPU opened"
  | Error why -> why

(* An open on a machine, refused or not, keeps nothing of the machine once
   the process drops it: the GC takes it. The open of its unbound GPU boots
   the GPU as far as the fixture lets it. *)
let[@inline never] open_dropped w =
  let machine = Rig_pci.Machine.at (unbound ()) in
  Weak.set w 0 (Some machine);
  ignore (Rig_gpu_support.opened (Rig_amd_pci.gpu ~machine 0))

let test_dropped () =
  if not Rig_pci_support.on_linux then
    skip ~reason:"flock on a function's file needs Linux" ();
  let w = Weak.create 1 in
  open_dropped w;
  Gc.full_major ();
  equal ~msg:"the machine still alive" bool false (Weak.check w 0)

let letting_go =
  group "letting go"
    [
      test "an unbound GPU whose ip_discovery stays is refused an open"
        (fun () ->
          let root = unbound () in
          Tree.add root
            (strf "sys/bus/pci/devices/%s/ip_discovery/die/0/GC/0/major" r9700)
            "12\n";
          Windtrap.contains ~sub:"ip_discovery" (refusal root));
      test "an unbound GPU KFD's topology lists is refused an open" (fun () ->
          let root = unbound () in
          Tree.add root "sys/class/kfd/kfd/topology/nodes/1/properties"
            "domain 0\nlocation_id 1280\n";
          Windtrap.contains ~sub:"topology" (refusal root));
      test "an unbound GPU amdgpu let go of is not refused for it" (fun () ->
          let root = unbound () in
          Tree.add root "sys/class/kfd/kfd/topology/nodes/1/properties"
            "domain 0\nlocation_id 17152\n";
          let why = refusal root in
          if contains ~sub:"not let go" why then
            failf "refused as held by amdgpu: %s" why);
      test "a machine the process dropped is collected after an open on it"
        test_dropped;
    ]

let () = exit (run "rig_amd_pci" [ reports; refused; numbering; letting_go ])
