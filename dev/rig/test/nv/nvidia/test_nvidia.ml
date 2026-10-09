(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module N = Rig_nv_nvidia
module S = Rig_nv_nvidia_support

(* Machines in a tree *)

let rec mkdir_p d =
  if not (Sys.file_exists d) then begin
    mkdir_p (Filename.dirname d);
    Sys.mkdir d 0o755
  end

let rec remove path =
  if Sys.file_exists path then
    if Sys.is_directory path then begin
      Array.iter (fun f -> remove (Filename.concat path f)) (Sys.readdir path);
      Sys.rmdir path
    end
    else Sys.remove path

(* A machine whose PCI functions are [functions], each a bus address and its
   [vendor] and [class] files' contents, [None] for a file it lacks, as Linux's
   sysfs shows them, under a directory beside the suite in _build, which a
   killed run may leave. Bus addresses hold colons, which some file systems
   refuse, so the tree is made here. *)
let tree name functions =
  let root = Filename.concat "trees" name in
  remove root;
  let put bus file contents =
    let dir = Filename.concat root ("sys/bus/pci/devices/" ^ bus) in
    mkdir_p dir;
    Option.iter
      (fun s ->
        Out_channel.with_open_bin (Filename.concat dir file) (fun oc ->
            output_string oc (s ^ "\n")))
      contents
  in
  List.iter
    (fun (bus, vendor, class_) ->
      put bus "vendor" vendor;
      put bus "class" class_)
    functions;
  root

let numbering () =
  let root =
    tree "numbering"
      [
        ("0000:01:00.0", Some "0x10de", Some "0x030000");
        ("0000:01:00.1", Some "0x10de", Some "0x040300");
        ("0000:02:00.0", Some "0x1002", Some "0x030000");
        ("0000:03:00.0", Some "0x10de", None);
        ("0000:0a:00.0", Some "0x10de", Some "0x030200");
        (* Their order by number is the reverse of their order as strings. *)
        ("10000:00:01.0", Some "0x10de", Some "0x030000");
        ("ffff:00:00.0", Some "0x10de", Some "0x030000");
      ]
  in
  equal (list string)
    [ "0000:01:00.0"; "0000:0a:00.0"; "ffff:00:00.0"; "10000:00:01.0" ]
    (N.buses ~root ());
  equal int 4 (N.count ~root ())

(* The kernel driver lists the GPUs it holds under /proc, by bus address: on a
   machine whose driver holds every NVIDIA GPU, they are the GPUs the path
   numbers. *)
let proc_gpus = "/proc/driver/nvidia/gpus"

let kernel_list () =
  if not (Sys.file_exists proc_gpus) then
    skip ~reason:"NVIDIA's kernel driver is not loaded" ();
  let held = List.sort compare (Array.to_list (Sys.readdir proc_gpus)) in
  equal (list string) held (List.sort compare (N.buses ()))

(* A device's objects are the GPU's for the process: a stopped device's GPU
   opens again. *)
let once () =
  if N.count () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  S.hold_gpu ();
  let g = require_ok (N.open_ 0) in
  let e = require_error ~msg:"a second open" (N.open_ 0) in
  contains ~sub:"has a device open" e;
  Rig_nv.stop g ~fault:None;
  let g' = require_ok ~msg:"an open after stop" (N.open_ 0) in
  Rig_nv.stop g' ~fault:None

(* Every root shows this machine: a root that links to [/] names the same
   GPUs, so a GPU open through one is open through the other. The link lives
   beside the suite in _build; it is unlinked, never walked. *)
let mirror = Filename.concat "trees" "mirror"

let other_root () =
  if N.count () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  S.hold_gpu ();
  mkdir_p "trees";
  (match Unix.lstat mirror with
  | _ -> Unix.unlink mirror
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> ());
  Unix.symlink "/" mirror;
  equal (list string) (N.buses ()) (N.buses ~root:mirror ());
  let g = require_ok (N.open_ 0) in
  let e =
    require_error ~msg:"an open through the other root"
      (N.open_ ~root:mirror 0)
  in
  contains ~sub:"has a device open" e;
  Rig_nv.stop g ~fault:None;
  let g' =
    require_ok ~msg:"an open through it after stop" (N.open_ ~root:mirror 0)
  in
  let e = require_error ~msg:"an open through /" (N.open_ 0) in
  contains ~sub:"has a device open" e;
  Rig_nv.stop g' ~fault:None

(* Opening and stopping a GPU again and again, its word freed after each stop,
   leaves the process's mappings as one open and stop left them. *)
let mappings () =
  In_channel.with_open_text "/proc/self/maps" In_channel.input_lines
  |> List.length

let reopen () =
  if N.count () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  S.hold_gpu ();
  let cycle () =
    let g = require_ok (N.open_ 0) in
    Rig_nv.stop g ~fault:None;
    Rig_nv.free g (Rig_nv.facts g).word
  in
  cycle ();
  let before = mappings () in
  for _ = 1 to 16 do
    cycle ()
  done;
  equal int ~msg:"the process's mappings" before (mappings ())

(* Opening and closing a GPU through rig again and again leaves the process's
   mappings as one open and close left them: rig gives each stopped device's
   word back. *)
let reclose () =
  if N.count () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  S.hold_gpu ();
  let cycle () =
    let d =
      require_ok ~pp:Format.pp_print_string
        (Rig.open_ (module Rig_nv) ~name:"NV" (fun () -> N.open_ 0))
    in
    Rig.close d;
    Gc.full_major ();
    ignore (Rig.Buffer.create Rig.host 8);
    Gc.full_major ();
    ignore (Rig.Buffer.create Rig.host 8)
  in
  cycle ();
  let before = mappings () in
  for _ = 1 to 16 do
    cycle ()
  done;
  equal int ~msg:"the process's mappings" before (mappings ())

let () =
  S.hold_gpu ();
  exit
  @@ run "rig_nv_nvidia"
       [
         group ~timeout:10. "numbering"
           [
             test "the GPUs are NVIDIA's display functions in bus order"
               numbering;
             test "a machine without PCI functions has no GPU" (fun () ->
                 equal (list string) [] (N.buses ~root:"no-such-directory" ()));
             test "this machine's GPUs are those NVIDIA's kernel driver lists"
               kernel_list;
             test "names GPU 0 NV and GPU i NV:i" (fun () ->
                 equal (list string) [ "NV"; "NV:1"; "NV:7" ]
                   (List.map N.device_name [ 0; 1; 7 ]));
           ];
         group ~timeout:60. "opening"
           [
             test "count answers on a machine whose /sys reads" (fun () ->
                 at_least int ~than:0 (N.count ()));
             test "a negative GPU raises" (fun () ->
                 raises_match Exn.invalid_arg (fun () -> N.open_ (-1));
                 raises_match Exn.invalid_arg (fun () -> N.device_name (-1)));
             test "a GPU past the count is an error naming the count" (fun () ->
                 let n = N.count () in
                 let e = require_error (N.open_ n) in
                 contains ~sub:(Printf.sprintf "has %d NVIDIA GPUs" n) e);
             test "a GPU has one device until it is stopped" once;
             test "a GPU has one device whatever the root" other_root;
             test "opening and stopping a GPU leaves no mapping behind" reopen;
             test "opening and closing a GPU through rig leaves no mapping \
                   behind"
               reclose;
           ];
       ]
