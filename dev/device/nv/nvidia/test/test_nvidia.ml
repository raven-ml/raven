(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module N = Device_nv_nvidia
module S = Device_nv_nvidia_support

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
    (N.gpus_at root)

(* A device's objects are the GPU's for the process: a stopped device's GPU
   opens again. *)
let once () =
  S.hold_gpu ();
  let g = require_ok (N.open_ 0) in
  let e = require_error ~msg:"a second open" (N.open_ 0) in
  contains ~sub:"has a device open" e;
  Device_nv.stop g;
  let g' = require_ok ~msg:"an open after stop" (N.open_ 0) in
  Device_nv.stop g'

let () =
  exit
  @@ run "device_nv_nvidia"
       [
         group ~timeout:10. "numbering"
           [
             test "the GPUs are NVIDIA's display functions in bus order"
               numbering;
             test "a machine without PCI functions has no GPU" (fun () ->
                 equal (list string) [] (N.gpus_at "no-such-directory"));
             test "this machine's GPUs are those its /sys lists" (fun () ->
                 equal int (List.length (N.gpus_at "/")) (N.count ()));
             test "names GPU 0 NV and GPU i NV:i" (fun () ->
                 equal (list string) [ "NV"; "NV:1"; "NV:7" ]
                   (List.map N.device_name [ 0; 1; 7 ]));
           ];
         group ~timeout:60. "opening"
           [
             test "count never raises" (fun () ->
                 at_least int ~than:0 (N.count ()));
             test "a negative GPU raises" (fun () ->
                 raises_match Exn.invalid_arg (fun () -> N.open_ (-1));
                 raises_match Exn.invalid_arg (fun () -> N.device_name (-1)));
             test "a GPU past the count is an error naming the count" (fun () ->
                 let n = N.count () in
                 let e = require_error (N.open_ n) in
                 contains ~sub:(Printf.sprintf "has %d NVIDIA GPUs" n) e);
             test "a GPU has one device until it is stopped" once;
           ];
       ]
