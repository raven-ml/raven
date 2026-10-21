(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module N = Rig_nv_nvidia
module S = Rig_nv_nvidia_support

(* [listed ~root ()] and [counted ~root ()] are the path's GPUs under [root]
   (defaults to ["/"]), which the test requires it lists. *)
let listed ?(root = "/") () =
  require_ok ~pp:Format.pp_print_string (N.buses_under ~root)

let counted ?(root = "/") () =
  require_ok ~pp:Format.pp_print_string (N.count_under ~root)

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
   refuse, so the tree is made here, and Windows file names refuse them. *)
let tree name functions =
  if Sys.win32 then skip ~reason:"Windows file names hold no PCI address" ();
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
    (listed ~root ());
  equal int 4 (counted ~root ())

(* A machine whose functions' directory cannot be read lists no GPU and says
   why, as an open does. A function whose [vendor] file cannot be read is left
   out, as rig.pci leaves it out. The test makes the file unreadable, which a
   process that reads every file cannot test. *)
let unreadable file =
  let root =
    tree "unreadable" [ ("0000:01:00.0", Some "0x10de", Some "0x030000") ]
  in
  let path = Filename.concat root file in
  let mode = (Unix.stat path).st_perm in
  Unix.chmod path 0;
  Fun.protect ~finally:(fun () -> Unix.chmod path mode) @@ fun () ->
  (match open_in_bin path with
  | ic ->
      close_in ic;
      skip ~reason:"the process reads unreadable files" ()
  | exception Sys_error _ -> ());
  if Filename.basename file = "vendor" then
    equal (result (list string) string) (Ok []) (N.buses_under ~root)
  else begin
    let why = require_error (N.buses_under ~root) in
    contains ~msg:"names the file" ~sub:path why;
    equal (result int string) (Error why) (N.count_under ~root);
    let e = require_error (Rig_gpu_support.opened (N.gpu_under ~root 0)) in
    contains ~sub:"reading the machine's GPUs" e;
    contains ~sub:why e
  end

(* The kernel driver lists the GPUs it holds under /proc, by bus address: on a
   machine whose driver holds every NVIDIA GPU, they are the GPUs the path
   numbers. *)
let proc_gpus = "/proc/driver/nvidia/gpus"

let kernel_list () =
  if not (Sys.file_exists proc_gpus) then
    skip ~reason:"NVIDIA's kernel driver is not loaded" ();
  let held = List.sort compare (Array.to_list (Sys.readdir proc_gpus)) in
  equal (list string) held (List.sort compare (listed ()))

(* A device's objects are the GPU's for the process: a stopped device's GPU
   opens again. *)
let once () =
  if counted () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  let g = require_ok (Rig_gpu_support.opened (N.gpu 0)) in
  let e =
    require_error ~msg:"a second open" (Rig_gpu_support.opened (N.gpu 0))
  in
  contains ~sub:"has a device open" e;
  Rig_nv.stop g ~fault:None;
  let g' =
    require_ok ~msg:"an open after stop" (Rig_gpu_support.opened (N.gpu 0))
  in
  Rig_nv.stop g' ~fault:None

(* Whether GPU [bus] shows a display, as nvidia-smi reports it, or [None]
   where nvidia-smi does not answer. *)
let display_active bus =
  let cmd =
    Filename.quote_command "nvidia-smi"
      [ "--query-gpu=display_active"; "--format=csv,noheader"; "-i"; bus ]
  in
  let ic = Unix.open_process_in cmd in
  let line = In_channel.input_line ic in
  match (Unix.close_process_in ic, line) with
  | WEXITED 0, Some "Enabled" -> Some true
  | WEXITED 0, Some "Disabled" -> Some false
  | _ -> None

(* The RM's watchdog does not run on a GPU without a display: rig then loses
   the device once its work has made no progress for 30 s. *)
let hang_bound () =
  if counted () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  let g = require_ok (Rig_gpu_support.opened (N.gpu 0)) in
  let hang_ms = (Rig_nv.facts g).hang_ms in
  Rig_nv.stop g ~fault:None;
  mem (option int) ~msg:"hang_ms" hang_ms [ None; Some 30_000 ];
  match display_active (List.hd (listed ())) with
  | None -> skip ~reason:"nvidia-smi does not answer" ()
  | Some true -> ()
  | Some false ->
      equal (option int) ~msg:"without a display" (Some 30_000) hang_ms

(* Every root shows this machine: a root that links to [/] names the same
   GPUs, so a GPU open through one is open through the other. The link lives
   beside the suite in _build; it is unlinked, never walked. *)
let mirror = Filename.concat "trees" "mirror"

let other_root () =
  if counted () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  mkdir_p "trees";
  (match Unix.lstat mirror with
  | _ -> Unix.unlink mirror
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> ());
  Unix.symlink "/" mirror;
  equal (list string) (listed ()) (listed ~root:mirror ());
  let g = require_ok (Rig_gpu_support.opened (N.gpu 0)) in
  let e =
    require_error ~msg:"an open through the other root"
      (Rig_gpu_support.opened (N.gpu_under ~root:mirror 0))
  in
  contains ~sub:"has a device open" e;
  Rig_nv.stop g ~fault:None;
  let g' =
    require_ok ~msg:"an open through it after stop"
      (Rig_gpu_support.opened (N.gpu_under ~root:mirror 0))
  in
  let e =
    require_error ~msg:"an open through /" (Rig_gpu_support.opened (N.gpu 0))
  in
  contains ~sub:"has a device open" e;
  Rig_nv.stop g' ~fault:None

(* The page a stopped device keeps of the GPU's mappings: its timeline word's,
   which lives for the process. *)
let page = Rig_gpu_support.Host.page

(* The bytes of the GPU's mappings that are not reservations ([---p]). *)
let mapped_bytes () =
  List.fold_left
    (fun n line ->
      Scanf.sscanf line "%x-%x %s" (fun a b perms ->
          if perms = "---p" then n else n + (b - a)))
    0 (S.mappings ())

(* [cycles cycle] runs [cycle] once, which makes what the process keeps for
   good, then [runs] times more, and checks that each run past the first keeps
   at most a page of the GPU's mappings. *)
let runs = 17

let cycles cycle =
  cycle ();
  let first = mapped_bytes () in
  for _ = 1 to runs do
    cycle ()
  done;
  at_most int ~msg:"the bytes of the GPU's mappings past the first run"
    ~than:(runs * page)
    (mapped_bytes () - first)

(* Opening and stopping a GPU again and again keeps at most one device's
   mappings per open. *)
let reopen () =
  if counted () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  cycles (fun () ->
      let g = require_ok (Rig_gpu_support.opened (N.gpu 0)) in
      Rig_nv.stop g ~fault:None)

(* Opening and closing a GPU through rig again and again keeps at most one
   device's mappings per open, once its memory returned. *)
let reclose () =
  if counted () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  cycles (fun () ->
      let d = require_ok ~pp:Format.pp_print_string (Rig.open_ (N.gpu 0)) in
      Rig.close d;
      Gc.full_major ();
      Rig.free_cache d)

(* Host memory maps for the GPU past the 39 GiB at once at which the RM's
   own mappings of it stopped on a GPU of 32 GiB in a machine of 62: one
   GiB of a shared memory file, mapped once in the host, maps 40 times. *)
let gib = 1 lsl 30
let maps = 40

let many_maps () =
  if counted () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  if not (Sys.file_exists "/dev/shm") then skip ~reason:"no /dev/shm" ();
  let file = Filename.temp_file ~temp_dir:"/dev/shm" "rig-nv" ".map" in
  let fd = Unix.openfile file [ O_RDWR ] 0o600 in
  let m =
    Fun.protect ~finally:(fun () -> Unix.close fd; Sys.remove file) @@ fun () ->
    Unix.map_file fd Bigarray.char Bigarray.c_layout true [| gib |]
  in
  let m = Bigarray.array1_of_genarray m in
  Bigarray.Array1.fill m '\001';
  let a = Rig_gpu_support.Host.data m in
  let g = require_ok (Rig_gpu_support.opened (N.gpu 0)) in
  let rec map i acc =
    if i = maps then acc
    else
      match Rig_nv.map_host g a gib with
      | Some r -> map (i + 1) (r :: acc)
      | None -> List.iter (Rig_nv.free g) acc; failf "map %d refused" i
  in
  let rs = map 0 [] in
  List.iter (Rig_nv.free g) rs;
  Rig_nv.stop g ~fault:None;
  ignore (Sys.opaque_identity m)

(* An open inside a lock of its thread's, as from a callback inside one,
   raises naming the opener, having opened nothing: GPU 0 then opens. *)
let inside_lock () =
  let refused =
    Rig_support.inside_lock (fun () ->
        match Rig_gpu_support.opened (N.gpu 0) with
        | Ok g ->
            Rig_nv.stop g ~fault:None;
            "opened"
        | Error why -> why
        | exception Invalid_argument why
          when String.starts_with ~prefix:"Rig_nv_nvidia.gpu: called from" why
          ->
            "refused")
  in
  equal ~msg:"the open inside the lock" string "refused" refused;
  if counted () > 0 then
    let opened = Rig_gpu_support.opened (N.gpu 0) in
    Rig_nv.stop (require_ok ~pp:Format.pp_print_string opened) ~fault:None

let () =
  exit
  @@ run "rig_nv_nvidia"
       [
         group ~timeout:10. "numbering"
           [
             test "the GPUs are NVIDIA's display functions in bus order"
               numbering;
             test "a machine without PCI functions has no GPU" (fun () ->
                 equal (list string) [] (listed ~root:"no-such-directory" ()));
             cases
               ~name:(fun f ->
                 if Filename.basename f = "vendor" then
                   "an unreadable vendor file leaves its function out"
                 else "an unreadable functions' directory is an Error naming it")
               "unreadable files"
               [
                 "sys/bus/pci/devices";
                 "sys/bus/pci/devices/0000:01:00.0/vendor";
               ]
               unreadable;
             test "this machine's GPUs are those NVIDIA's kernel driver lists"
               kernel_list;
             test "names GPU 0 NV and GPU i NV:i" (fun () ->
                 equal (list string) [ "NV"; "NV:1"; "NV:7" ]
                   (List.map (fun i -> (N.gpu i).name) [ 0; 1; 7 ]));
           ];
         group ~timeout:60. "opening"
           [
             test "count answers on a machine whose /sys reads" (fun () ->
                 at_least int ~than:0 (counted ()));
             test "a negative GPU raises" (fun () ->
                 raises_match Exn.invalid_arg (fun () ->
                     Rig_gpu_support.opened (N.gpu (-1)));
                 raises_match Exn.invalid_arg (fun () -> (N.gpu (-1)).name));
             test "a GPU past the count is an error naming the count" (fun () ->
                 let n = counted () in
                 let e = require_error (Rig_gpu_support.opened (N.gpu n)) in
                 contains ~sub:(Printf.sprintf "has %d NVIDIA GPUs" n) e);
             test "an open inside a lock of its thread's raises" inside_lock;
             test "a GPU has one device until it is stopped" once;
             test "a GPU has one device whatever the root" other_root;
             test "a GPU without a display bounds hangs at 30 s" hang_bound;
             test "opening and stopping a GPU keeps a device's mappings each"
               reopen;
             test
               "opening and closing a GPU through rig keeps a device's \
                mappings each"
               reclose;
           ];
         group ~timeout:60. "memory"
           [ test "40 GiB of host memory maps at once" many_maps ];
       ]
