(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Rig_pci
open Rig_pci_support

let strf = Printf.sprintf

external c_combines : Window.t -> bool = "rig_pci_test_combines"

let map ?combine ?off ?length f i =
  require_ok (Function.map ?combine ?off ?length f i)

let alloc_dma ?contiguous ?va f n =
  require_some (require_ok (Function.alloc_dma ?contiguous ?va f n))

let pin f a n = require_ok (Function.pin f a n)

let addressing =
  Testable.make
    ~pp:(fun ppf a ->
      Format.pp_print_string ppf
        (match a with Machine.Physical -> "Physical" | Iommu -> "Iommu"))
    ~equal:( = )

(* A bus that is no bus address reaches no file, here or on another machine:
   these name sysfs's directory, its parent, or a path through it. *)
let not_buses =
  [
    "";
    ".";
    "..";
    "0000:01:00.0/..";
    "../../../etc";
    "0000:01:00.0\000";
    "0000:01:00";
  ]

(* This machine *)

let sysfs bus file =
  Filename.concat (Filename.concat "/sys/bus/pci/devices" bus) file

(* What a take must not change: the function's driver and whether it is
   enabled. *)
let state bus =
  let driver =
    match Unix.readlink (sysfs bus "driver") with
    | l -> Some (Filename.basename l)
    | exception Unix.Unix_error _ -> None
  in
  let enabled =
    match
      In_channel.with_open_text (sysfs bus "enable") In_channel.input_all
    with
    | s -> Some (String.trim s)
    | exception Sys_error _ -> None
  in
  (driver, enabled)

let test_refused_here () =
  if on_linux then skip ~reason:"this machine has /sys/bus/pci" ();
  match Function.take Machine.this "0000:00:00.0" with
  | Ok _ -> fail "a function taken on a machine without PCI functions"
  | Error why -> contains ~msg:"names the bus" ~sub:"0000:00:00.0" why

(* Listing and taking change nothing on the machine. Only GPUs are taken:
   another class of function may be another user's device. *)
let test_changes_nothing () =
  if not on_linux then skip ~reason:"this machine has no /sys/bus/pci" ();
  hold_gpu ();
  let buses = List.map (fun (d : Machine.id) -> d.bus) (this_gpus ()) in
  let before = List.map state buses in
  List.iter
    (fun bus ->
      match Function.take Machine.this bus with
      | Ok f -> Function.release f
      | Error _ -> ())
    buses;
  let state_w = pair (option string) (option string) in
  equal
    (list (pair string state_w))
    (List.combine buses before)
    (List.combine buses (List.map state buses))

let takeable () =
  this_gpus ()
  |> List.find_map (fun (d : Machine.id) ->
      match Function.take Machine.this d.bus with
      | Ok f ->
          Function.release f;
          Some d
      | Error _ -> None)

(* One process holds a function at a time, this one included. *)
let test_held_here () =
  if not on_linux then skip ~reason:"this machine has no /sys/bus/pci" ();
  hold_gpu ();
  let d =
    match takeable () with
    | Some d -> d
    | None -> skip ~reason:"no function this process may take" ()
  in
  let f = Result.get_ok (Function.take Machine.this d.bus) in
  is_error ~msg:"held" (Function.take Machine.this d.bus);
  Function.release f;
  let g = Result.get_ok (Function.take Machine.this d.bus) in
  Function.release g

let vfio_function () =
  if not (Sys.file_exists "/dev/vfio") then skip ~reason:"no /dev/vfio" ();
  this_gpus ()
  |> List.find_map (fun (d : Machine.id) ->
      match Function.take Machine.this d.bus with
      | Ok f when Function.addressing f = Iommu -> Some (d, f)
      | Ok f ->
          Function.release f;
          None
      | Error _ -> None)
  |> function
  | Some x -> x
  | None -> skip ~reason:"no function this process may take behind an IOMMU" ()

(* Behind an IOMMU, a function needs no root. *)
let test_vfio () =
  hold_gpu ();
  let d, f = vfio_function () in
  Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
  equal ~msg:"its vendor" int d.vendor (Function.config16 f 0);
  equal ~msg:"its device" int d.device (Function.config16 f 2);
  let page = Machine.page Machine.this in
  let allocs = List.map (fun n -> alloc_dma f n) [ 1; 3 * page; 2 * mib ] in
  List.iter
    (fun (w, runs) ->
      equal ~msg:"one run of its bytes" (list int)
        [ Window.length w ]
        (List.map snd runs))
    allocs;
  let runs = List.sort compare (List.concat_map snd allocs) in
  List.iter2
    (fun (a, n) (b, _) ->
      at_most ~msg:"device addresses apart" int ~than:b (a + n))
    (List.filteri (fun i _ -> i < List.length runs - 1) runs)
    (List.tl runs);
  let w, _ = List.hd allocs in
  let pinned = pin f (Window.address w) (Window.length w) in
  equal ~msg:"a pin is one run" int 1 (List.length pinned);
  Function.unpin f (Window.address w) (Window.length w);
  List.iter (fun (w, _) -> Function.free_dma f w) allocs

(* A machine's files *)

(* [take_on fns bus] takes [bus] on a machine whose functions are [fns]. *)
let take_on ?lockdown ?groups ?noiommu fns bus =
  Function.take (Machine.at (Tree.make ?lockdown ?groups ?noiommu fns)) bus

let audio bus =
  { (Tree.gpu ~driver:"snd_hda_intel" bus) with class_ = 0x040300; bars = [] }

(* Each refusal names the function and its cause, and what cures it where a
   detach does. *)
let refusals =
  [
    ( "a bus the machine lacks",
      [ Tree.gpu "0000:03:00.0" ],
      "0000:04:00.0",
      [],
      [ "0000:04:00.0 is no PCI function" ] );
    ( "a driver other than vfio-pci, without an IOMMU",
      [ Tree.gpu ~driver:"amdgpu" "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 is bound to the driver amdgpu"; "detach the GPU" ] );
    ( "a driver other than vfio-pci, behind an IOMMU",
      [ Tree.gpu ~driver:"amdgpu" ~group:"12" "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "bound to the driver amdgpu"; "vfio-pci" ] );
    ( "no driver behind a translating IOMMU",
      [ Tree.gpu ~group:"12" "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "the IOMMU translates the addresses 0000:03:00.0 reaches"; "iommu=pt" ]
    );
    ( "a device shared with another function",
      [ Tree.gpu "0000:03:00.0"; audio "0000:03:00.1" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 shares its device with 0000:03:00.1"; "detach the GPU" ]
    );
    ( "a device shared with another function, bound to vfio-pci without an IOMMU",
      [ Tree.gpu ~driver:"vfio-pci" "0000:03:00.0"; audio "0000:03:00.1" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 shares its device with 0000:03:00.1" ] );
    ( "a function on vfio-pci in no IOMMU group",
      [ Tree.gpu ~driver:"vfio-pci" "0000:03:00.0" ],
      "0000:03:00.0",
      [ "flock" ],
      [ "0000:03:00.0 is bound to vfio-pci but in no IOMMU group" ] );
    ( "a disabled function",
      [ Tree.gpu ~enabled:false "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 is disabled"; "detach the GPU" ] );
    ( "a configuration file the process may not write",
      [ Tree.gpu "0000:03:00.0" ],
      "0000:03:00.0",
      [ "read-only" ],
      [ "taking 0000:03:00.0 needs write access"; "run as root" ] );
    ( "a locked-down kernel",
      [ Tree.gpu "0000:03:00.0" ],
      "0000:03:00.0",
      [ "lockdown" ],
      [ "the kernel is locked down"; "0000:03:00.0" ] );
    ( "BARs that cannot be read",
      [ Tree.gpu "0000:03:00.0" ],
      "0000:03:00.0",
      [ "unreadable:sys/bus/pci/devices/0000:03:00.0/resource" ],
      [ "reading "; "0000:03:00.0/resource: " ] );
    ( "an IOMMU group whose type cannot be read",
      [ Tree.gpu ~group:"12" "0000:03:00.0" ],
      "0000:03:00.0",
      [ "identity"; "unreadable:sys/kernel/iommu_groups/12/type" ],
      [ "reading "; "iommu_groups/12/type: " ] );
  ]

let test_refusal (_, fns, bus, opts, subs) =
  if List.mem "flock" opts && not on_linux then
    skip ~reason:"flock on a function's file needs Linux" ();
  let lockdown =
    if List.mem "lockdown" opts then Some "none [integrity] confidentiality"
    else None
  in
  let groups =
    if List.mem "identity" opts then [ ("12", "identity") ] else []
  in
  let root = Tree.make ?lockdown ~groups fns in
  let chmod file mode =
    if Unix.geteuid () = 0 then
      skip ~reason:"root opens a file whatever its mode" ();
    Unix.chmod (Filename.concat root file) mode
  in
  List.iter
    (fun opt ->
      match String.split_on_char ':' opt with
      | [ "read-only" ] ->
          chmod ("sys/bus/pci/devices/" ^ bus ^ "/config") 0o444
      | "unreadable" :: path -> chmod (String.concat ":" path) 0o000
      | _ -> ())
    opts;
  let why = require_error (Function.take (Machine.at root) bus) in
  List.iter (fun sub -> contains ~sub why) subs

(* An identity IOMMU passes physical addresses through, as does VFIO's no-IOMMU
   mode: a function under either is taken physically. On Linux the take locks
   its configuration file; elsewhere flock is refused. *)
let test_physical () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  List.iter
    (fun (msg, groups, noiommu, group) ->
      let fn = Tree.gpu ?group "0000:03:00.0" in
      let m = Machine.at (Tree.make ~groups ~noiommu [ fn ]) in
      let f = require_ok ~msg (Function.take m fn.bus) in
      equal ~msg:"bus" string fn.bus (Function.bus f);
      equal ~msg:"released" bool false (Function.released f);
      equal ~msg addressing Physical (Function.addressing f);
      equal ~msg:"vendor" int 0x1002 (Function.config16 f 0);
      equal ~msg:"device" int 0x744c (Function.config16 f 2);
      let bars = List.init 7 (Function.bar f) in
      equal ~msg:"BARs"
        (list (option (pair hex int)))
        [
          Some (0x7c_0000_0000, 256 * mib);
          None;
          Some (0xfc00_0000, 2 * mib);
          None;
          Some (0xe000, 256);
          Some (0xfcc0_0000, mib);
          None;
        ]
        bars;
      equal ~msg:"past the 64 bytes Linux shows" hex 0xffff_ffff
        (Function.config32 f 64);
      equal ~msg:"no interrupt to wait for" bool false
        (Function.interrupt f max_int);
      contains ~msg:"a second take" ~sub:"0000:03:00.0 is taken already"
        (require_error (Function.take m fn.bus));
      Function.release f;
      Function.release
        (require_ok ~msg:"a take once released" (Function.take m fn.bus)))
    [
      ("no IOMMU", [], [], None);
      ("an identity IOMMU", [ ("12", "identity") ], [], Some "12");
      ("VFIO's no-IOMMU mode", [], [ "12" ], Some "12");
    ]

(* A BAR's file is as long as the BAR, and maps whole. A file shorter than its
   BAR would end the process with SIGBUS at the first access past its end, so
   the map refuses it, naming the file. *)
let test_map_files () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let root = Tree.make [ fn ] in
  Tree.add root "sys/bus/pci/devices/0000:03:00.0/resource5"
    (String.make 4096 '\000');
  let f = require_ok (Function.take (Machine.at root) fn.bus) in
  Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
  List.iter
    (fun (i, combine, n) ->
      let msg = strf "BAR %d, combine:%b" i combine in
      let w = require_ok ~msg (Function.map ~combine f i) in
      equal ~msg int n (Window.length w);
      equal ~msg:(msg ^ ", its last word") hex 0 (Window.get32 w (n - 4));
      Function.unmap f w)
    [ (0, false, 256 * mib); (0, true, 256 * mib); (2, false, 2 * mib) ];
  let why = require_error ~msg:"BAR 5" (Function.map f 5) in
  List.iter
    (fun sub -> contains ~sub why)
    [ "0000:03:00.0/resource5"; "4096 bytes"; "BAR 5"; "1048576" ]

(* Bound to vfio-pci, a function is taken through VFIO, which takes its IOMMU
   group whole: neither what shares its device nor whether it is enabled refuses
   it. The machine has none of VFIO's files, so the take is refused naming the
   first one it opens. In VFIO's no-IOMMU mode the take locks the function's
   file first. *)
let through_vfio =
  [
    ("behind a translating IOMMU, beside its audio", [], [], true, true);
    ("behind an identity IOMMU", [ ("12", "identity") ], [], false, true);
    ("behind an IOMMU, disabled", [], [], false, false);
    ("in VFIO's no-IOMMU mode", [], [ "12" ], false, true);
  ]

let test_through_vfio (_, groups, noiommu, beside, enabled) =
  if noiommu <> [] && not on_linux then
    skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu ~driver:"vfio-pci" ~group:"12" ~enabled "0000:03:00.0" in
  let fns = if beside then [ fn; audio "0000:03:00.1" ] else [ fn ] in
  contains ~sub:"dev/vfio/vfio does not exist"
    (require_error (take_on ~groups ~noiommu fns fn.bus))

(* The command register and two of its bits (PCI Express Base Specification,
   7.5.1.1.3): the function answers at its memory BARs, and it masters the bus,
   reaching system memory by DMA. *)
let command = 0x04
let memory_space = 0x2
let bus_master = 0x4

(* The command register's bit that keeps the function from signalling legacy
   interrupts, its INTx line (PCI Express Base Specification, 7.5.1.1.3). *)
let intx_disable = 0x400

(* [take_mastering m bus] takes [bus] on [m] and turns its memory space and bus
   mastering on, keeping the command register's other bits, as a driver does. *)
let take_mastering m bus =
  let f = require_ok (Function.take m bus) in
  Function.set_config16 f command
    (Function.config16 f command lor memory_space lor bus_master);
  f

let test_release_stops_dma () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let m = Machine.at (Tree.make [ fn ]) in
  Function.release (take_mastering m fn.bus);
  let f = require_ok (Function.take m fn.bus) in
  equal hex (memory_space lor intx_disable) (Function.config16 f command);
  Function.release f

(* The command register as the fixture's configuration file holds it. *)
let config_file root bus =
  Filename.concat root (strf "sys/bus/pci/devices/%s/config" bus)

let command_in root bus =
  let s =
    In_channel.with_open_bin (config_file root bus) In_channel.input_all
  in
  String.get_uint16_le s command

let set_command root bus v =
  let fd = Unix.openfile (config_file root bus) [ O_WRONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  let b = Bytes.create 2 in
  Bytes.set_uint16_le b 0 v;
  ignore (Unix.lseek fd command SEEK_SET);
  ignore (Unix.write fd b 0 2)

(* Whatever the command register holds, set_bus_master changes its bus master
   bit alone. *)
let test_set_bus_master =
  prop
    "set_bus_master sets or clears the command register's bus master bit alone"
    ~count:50
    (Gen.pair (Gen.with_pp pp_hex (Gen.int_range 0 0xffff)) Gen.bool)
    (fun (found, on) ->
      if not on_linux then
        skip ~reason:"flock on a function's file needs Linux" ();
      let fn = Tree.gpu "0000:03:00.0" in
      let root = Tree.make [ fn ] in
      set_command root fn.bus found;
      let f = require_ok (Function.take (Machine.at root) fn.bus) in
      Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
      let before = command_in root fn.bus in
      cover "the bit set before" (before land bus_master <> 0);
      cover "the bit clear before" (before land bus_master = 0);
      Function.set_bus_master f on;
      equal hex
        (if on then before lor bus_master else before land lnot bus_master)
        (command_in root fn.bus))

(* A function taken physically, with no interrupt route through VFIO, signals no
   legacy interrupt while taken: nothing handles it, and bus mastering does not
   gate it. Release gives its INTx back as it found it. *)
let test_intx (_, before) =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let root = Tree.make [ fn ] in
  let m = Machine.at root in
  set_command root fn.bus (memory_space lor before);
  let f = require_ok (Function.take m fn.bus) in
  equal ~msg:"taken" hex
    (memory_space lor intx_disable)
    (Function.config16 f command);
  Function.release f;
  equal ~msg:"released" hex (memory_space lor before) (command_in root fn.bus)

let intx_states = [ ("INTx on", 0); ("INTx off", intx_disable) ]

(* The test's executable, run with [exiting], is [exit_mastering]'s process. *)
let exiting = "--exit-mastering"

(* Takes [bus] of the machine at [root] with its bus mastering on, forks a child
   that exits, and exits, with 0 iff the child's exit left the bus mastering
   on. *)
let exit_mastering root bus =
  let f = take_mastering (Machine.at root) bus in
  (match Unix.fork () with
  | 0 -> exit 0
  | child -> ignore (Unix.waitpid [] child));
  exit (if Function.config16 f command land bus_master <> 0 then 0 else 1)

let test_exit (_, before) =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let root = Tree.make [ fn ] in
  set_command root fn.bus (memory_space lor before);
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe
      [| exe; exiting; root; fn.bus |]
      Unix.stdin Unix.stdout Unix.stderr
  in
  let status = ref None in
  let exited () =
    match Unix.waitpid [ WNOHANG ] pid with
    | 0, _ -> false
    | _, s ->
        status := Some s;
        true
  in
  if not (poll exited) then begin
    Unix.kill pid Sys.sigkill;
    failf "the process holding %s did not exit" fn.bus
  end;
  let code = match !status with Some (WEXITED c) -> c | _ -> -1 in
  equal ~msg:"its child's exit left it mastering the bus" int 0 code;
  equal ~msg:"its own exit stopped it" hex 0
    (command_in root fn.bus land bus_master);
  equal ~msg:"its own exit gave its INTx back" hex before
    (command_in root fn.bus land intx_disable)

(* Linux offers a prefetchable BAR combining through [resourceN_wc]: BAR 0 of
   the fixture's GPU is prefetchable, BAR 5 is not. *)
let test_combining () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let f = require_ok (Function.take (Machine.at (Tree.make [ fn ])) fn.bus) in
  Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
  let combines ?combine i =
    let w = map ?combine ~length:4096 f i in
    equal ~msg:"its bytes" int 0 (Window.get32 w 0);
    (w, c_combines w)
  in
  let w, c = combines ~combine:true 0 in
  equal ~msg:"a prefetchable BAR, asked" bool true c;
  raises_match ~msg:"the other way while it lives" Exn.invalid_arg (fun () ->
      Function.map ~combine:false f 0);
  let w', c = combines 0 in
  equal ~msg:"not asked, as the live window" bool true c;
  Function.unmap f w';
  Function.unmap f w;
  let w, c = combines 0 in
  equal ~msg:"a prefetchable BAR, not asked" bool false c;
  Function.unmap f w;
  let _, c = combines ~combine:true 5 in
  equal ~msg:"a BAR that is not prefetchable, asked" bool false c

let tree_files =
  group ~timeout:patience "a machine's files"
    [
      cases "a take is refused, naming the function and the cause"
        ~name:(fun (n, _, _, _, _) -> n)
        refusals test_refusal;
      test
        "a function alone and enabled, under no translating IOMMU, is taken \
         physically by one take at a time, its BARs as its registers and \
         resource file say, all ones past 64 bytes, without interrupts"
        test_physical;
      test
        "a BAR of a function taken physically maps whole from its file, and a \
         file shorter than its BAR is refused"
        test_map_files;
      cases
        "a function bound to vfio-pci is taken through VFIO, whatever shares \
         its device or whether it is enabled"
        ~name:(fun (n, _, _, _, _) -> n)
        through_vfio test_through_vfio;
      test
        "a prefetchable BAR of a function taken physically combines where \
         asked, one way at a time"
        test_combining;
      test_set_bus_master;
      test "a function taken physically stops mastering the bus when released"
        test_release_stops_dma;
      cases
        "a function taken physically has its INTx off while taken, as found \
         after"
        ~name:fst intx_states test_intx;
      cases
        "a function taken physically stops mastering the bus and has its INTx \
         as found when its process exits, and a child that process forked \
         exits without stopping it"
        ~name:fst intx_states test_exit;
    ]

(* Locked system memory

   Functions of a machine in a fixture tree, taken physically, reach this
   process's memory as GPUs of this machine do, at the physical addresses the
   tree's page map gives. Locking memory needs the locked-memory limit: a test
   the machine refuses it skips with the reason. *)

(* [with_fixtures n f] is [f root fns], [fns] the [n] functions of a fixture
   machine at [root], taken. *)
let with_fixtures n f =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let buses = List.init n (fun i -> strf "0000:%02x:00.0" (3 + i)) in
  let root = Tree.make (List.map Tree.gpu buses) in
  let m = Machine.at root in
  let fns = List.map (fun bus -> require_ok (Function.take m bus)) buses in
  Fun.protect
    ~finally:(fun () -> List.iter Function.release fns)
    (fun () -> f root fns)

let with_fixture f = with_fixtures 1 (fun root fns -> f root (List.hd fns))
let huge = 2 * mib

(* Frames the fixture's page map gives the 2 MiB blocks of addresses that hold
   the [n] bytes at [a], one block of frames each, from a frame no other block
   of a test uses; and the physical addresses of the pages of the [n] bytes. *)
let frames root a n =
  let page = Machine.page Machine.this in
  let lo = a / huge * huge and hi = round_up (a + n) huge in
  let first block = 0x10_0000 + (block / huge land 0xfff * (huge / page)) in
  let rec give block =
    if block < hi then begin
      Tree.pagemap root ~page block
        (List.init (huge / page) (fun i -> first block + i));
      give (block + huge)
    end
  in
  give lo;
  List.init
    ((n + page - 1) / page)
    (fun i ->
      let p = a + (i * page) in
      (first (p / huge * huge) * page) + (p mod huge))

(* The runs of pages [pas]: those that follow each other merged. *)
let runs_of pas =
  let page = Machine.page Machine.this in
  List.fold_left
    (fun acc pa ->
      match acc with
      | (a, n) :: rest when a + n = pa -> (a, n + page) :: rest
      | _ -> (pa, page) :: acc)
    [] pas
  |> List.rev

let granted = function Ok x -> x | Error why -> skip ~reason:why ()

let given r =
  match granted r with
  | Some x -> x
  | None -> skip ~reason:"the machine has no free memory" ()

(* Memory reached physically lies in huge pages: its runs are those of the
   blocks of frames that hold it. *)
let test_dma () =
  with_fixture @@ fun root f ->
  granted (Machine.reserve (Function.machine f) ~base:free_base (8 * mib));
  let va = free_base + mib in
  let pas = frames root va (3 * mib) in
  let w, runs = given (Function.alloc_dma ~va f (3 * mib)) in
  equal ~msg:"a run per block of frames"
    (list (pair hex int))
    (runs_of pas) runs;
  equal ~msg:"zeroed" string
    (String.make (3 * mib) '\000')
    (Window.read w 0 (3 * mib));
  Function.free_dma f w

(* The process's pages go back to the system when it dies: a function taken
   physically, which would keep writing them, is refused them. *)
let test_pin_physical () =
  with_fixture @@ fun root f ->
  let page = Machine.page Machine.this in
  let a = round_up (memory (2 * page)) page in
  ignore (frames root a page);
  contains ~sub:"without an IOMMU" (require_error (Function.pin f a page))

let test_contiguous () =
  with_fixture @@ fun root f ->
  granted (Machine.reserve (Function.machine f) ~base:free_base (8 * mib));
  let va = free_base + (2 * mib) in
  ignore (frames root va (2 * mib));
  let w, runs = given (Function.alloc_dma ~contiguous:true ~va f (300 * kib)) in
  equal ~msg:"at the address asked" hex va (Window.address w);
  equal ~msg:"one run" int 1 (List.length runs);
  Function.free_dma f w

(* Memory that outlives the process

   A function taken physically keeps writing memory after its process dies: its
   memory lies in huge pages of files under the machine's [dev/hugepages], which
   keep their pages until no function reaches them. *)

let memory_files root =
  Sys.readdir (Filename.concat root "dev/hugepages")
  |> Array.to_list
  |> List.filter (fun f ->
      String.starts_with ~prefix:"rig-pci-" f
      && not (String.ends_with ~suffix:".reach" f))

let memory_file root =
  match memory_files root with
  | [ f ] -> Filename.concat root ("dev/hugepages/" ^ f)
  | fs -> failf "%d memory files" (List.length fs)

(* Every entry of the machine's [dev/hugepages]: files and their lists. *)
let hugepages root =
  Sys.readdir (Filename.concat root "dev/hugepages")
  |> Array.to_list |> List.sort compare

let reachers file =
  In_channel.with_open_bin (file ^ ".reach") In_channel.input_all

let reserved f =
  granted (Machine.reserve (Function.machine f) ~base:free_base (8 * mib))

(* A memory file is locked by its process from the moment it has a name, and
   lists the function before it holds a page: a process that finds it unlocked
   may take it for one a dead process left. *)
let test_memory_file () =
  with_fixture @@ fun root f ->
  reserved f;
  let va = free_base + (4 * mib) in
  ignore (frames root va mib);
  let w, _ = given (Function.alloc_dma ~va f mib) in
  equal ~msg:"a file of its own" int 1 (List.length (memory_files root));
  let file = memory_file root in
  equal ~msg:"locked by its process" bool true (Tree.flocked file);
  equal ~msg:"listing the function" string
    (Function.bus f ^ "\n")
    (reachers file);
  Function.free_dma f w;
  equal ~msg:"kept while the function is held" int 1
    (List.length (memory_files root));
  Function.release f;
  equal ~msg:"gone once released" (list string) [] (memory_files root)

(* Taken physically, a function pins memory alloc_dma gave on its machine, for
   itself or another function: its huge page stays through its free until each
   pin is unpinned. *)
let test_pin_dma () =
  with_fixtures 2 @@ fun root fns ->
  let f = List.nth fns 0 and g = List.nth fns 1 in
  reserved f;
  let va = free_base + (2 * mib) in
  ignore (frames root va (64 * kib));
  let w, runs = given (Function.alloc_dma ~va f (64 * kib)) in
  let n = Window.length w in
  equal ~msg:"pinned for itself, at its runs"
    (list (pair hex int))
    runs (pin f va n);
  equal ~msg:"pinned for another function, at its runs"
    (list (pair hex int))
    runs (pin g va n);
  let file = memory_file root in
  Function.free_dma f w;
  equal ~msg:"kept through its free" int huge (Tree.stored file);
  Function.unpin f va n;
  equal ~msg:"kept while one pin holds it" int huge (Tree.stored file);
  Function.unpin g va n;
  equal ~msg:"given back once unpinned" int 0 (Tree.stored file)

(* Memory another machine's function allocated is not this machine's. *)
let test_pin_other_machine () =
  with_fixture @@ fun root f ->
  with_fixture @@ fun _ f' ->
  reserved f;
  let va = free_base + (2 * mib) in
  ignore (frames root va (64 * kib));
  let w, _ = given (Function.alloc_dma ~va f (64 * kib)) in
  contains ~sub:"alloc_dma" (require_error (Function.pin f' va (64 * kib)));
  Function.free_dma f w

(* A function is named by its machine and bus: the release of one leaves the
   memory of a function at the same bus on another machine. *)
let test_memory_machines () =
  with_fixture @@ fun root f ->
  with_fixture @@ fun root' f' ->
  reserved f;
  reserved f';
  let va = free_base + (4 * mib) and va' = free_base + (6 * mib) in
  ignore (frames root va mib);
  ignore (frames root' va' mib);
  let w, _ = given (Function.alloc_dma ~va f mib) in
  let w', _ = given (Function.alloc_dma ~va:va' f' mib) in
  equal ~msg:"one bus on both machines" string (Function.bus f)
    (Function.bus f');
  let kept = memory_files root in
  Function.release f';
  equal ~msg:"the other machine's memory stays" (list string) kept
    (memory_files root);
  Function.free_dma f' w';
  Function.free_dma f w

let fixture_gpus ?(reset = fun _ -> Ok ()) () =
  Gpus.make ~name:"fixture" ~memory_bar:0
    ~nodes:(fun ~root:_ _ -> [])
    ~unreleased:(fun ~root:_ _ -> None)
    ~teardown_ms:0 ~reset
    (fun (id : Machine.id) -> id.class_ lsr 16 = 0x03)

(* The test's executable, run with [holding], is [hold_memory]'s process. *)
let holding = "--hold-memory"

(* The code [hold_memory] exits with when the machine refuses it the function's
   memory. *)
let refused_code = 3

(* Takes the fixture's function at [bus] and allocates its memory at [va], whose
   frames the tree's page map gives, which it never frees. With [how]
   ["renew-die"] it opens the fixture's GPU 0 instead and renews it after the
   allocation, then dies by SIGKILL. With [how] ["die"] it then dies by SIGKILL,
   which runs no exit function; with ["wait"] it waits for its standard input to
   close and exits. ["released-die"] and ["released-wait"] release the function
   first. ["released-quit"] releases it, waits as ["wait"] does, and leaves by
   [_exit], which runs no exit function either. *)
let hold_memory how root bus va =
  let ok = function Ok x -> x | Error _ -> exit refused_code in
  if how = "renew-die" then begin
    (* Opens the GPU, allocates its memory, renews it, and dies. *)
    let g = fixture_gpus () in
    ignore
      (Gpus.open_ g (Machine.at root) 0 (fun h f ->
           ok (Machine.reserve (Function.machine f) ~base:free_base (8 * mib));
           if Option.is_none (ok (Function.alloc_dma ~va f mib)) then
             exit refused_code;
           ok (Gpus.renew h);
           Unix.kill (Unix.getpid ()) Sys.sigkill;
           Ok ()))
  end;
  let f = ok (Function.take (Machine.at root) bus) in
  ok (Machine.reserve (Function.machine f) ~base:free_base (8 * mib));
  if Option.is_none (ok (Function.alloc_dma ~va f mib)) then exit refused_code;
  if String.starts_with ~prefix:"released-" how then Function.release f;
  match how with
  | "die" | "released-die" -> Unix.kill (Unix.getpid ()) Sys.sigkill
  | "released-quit" ->
      print_endline "holding";
      ignore (In_channel.input_all stdin);
      Unix._exit 0
  | _ ->
      print_endline "holding";
      ignore (In_channel.input_all stdin);
      exit 0

(* [holder how root bus va] starts [hold_memory]: its pid, the pipe on its
   standard input, and its first line of output, which waits until it holds. *)
let holder how root bus va =
  let exe = Sys.executable_name in
  let input, feed = Unix.pipe ~cloexec:true () in
  let said, output = Unix.pipe ~cloexec:true () in
  let pid =
    Unix.create_process exe
      [| exe; holding; how; root; bus; string_of_int va |]
      input output Unix.stderr
  in
  Unix.close input;
  Unix.close output;
  (pid, feed, said)

let wait_exit pid =
  let status = ref None in
  let exited () =
    match Unix.waitpid [ WNOHANG ] pid with
    | 0, _ -> false
    | _, s ->
        status := Some s;
        true
  in
  if not (poll exited) then begin
    Unix.kill pid Sys.sigkill;
    failf "the process %d did not exit" pid
  end;
  match !status with
  | Some (WEXITED c) when c = refused_code ->
      skip ~reason:"the machine refused the function's memory" ()
  | Some s -> s
  | None -> assert false

(* The root of a fixture tree whose one GPU a child that allocated its memory
   left by dying of SIGKILL. *)
let left_by_death ?(how = "die") () =
  let fn = Tree.gpu "0000:03:00.0" in
  let root = Tree.make [ fn ] in
  let va = free_base + (6 * mib) in
  ignore (frames root va mib);
  let pid, feed, said = holder how root fn.bus va in
  Unix.close feed;
  Unix.close said;
  (match wait_exit pid with
  | WSIGNALED s when s = Sys.sigkill -> ()
  | _ -> fail "the process holding the function did not die by SIGKILL");
  root

(* A vendor whose resets are counted in [n]. *)
let counted_gpus n =
  fixture_gpus
    ~reset:(fun _ ->
      Atomic.incr n;
      Ok ())
    ()

(* [kept h] gives [h] a clean stop and keeps it. *)
let kept h =
  Gpus.set_stop h (fun () -> `Clean);
  Ok h

let open_held g m = Gpus.open_ g m 0 (fun h _ -> kept h)
let stop h = ignore (Gpus.stop h : [ `Stopped | `Unknown ])

(* What a process that died left stays through another take and release, which
   does not reset the GPU, and goes once its GPU is reset. *)
let test_death () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let root = Tree.make [ fn ] in
  let va = free_base + (6 * mib) in
  ignore (frames root va mib);
  let pid, feed, said = holder "die" root fn.bus va in
  Unix.close feed;
  Unix.close said;
  (match wait_exit pid with
  | WSIGNALED s when s = Sys.sigkill -> ()
  | _ -> fail "the process holding the function did not die by SIGKILL");
  let left = memory_files root in
  equal ~msg:"left by the dead process" int 1 (List.length left);
  let m = Machine.at root in
  Function.release (require_ok (Function.take m fn.bus));
  equal ~msg:"kept through a take and a release" (list string) left
    (memory_files root);
  require_ok (Gpus.reset (fixture_gpus ()) m 0);
  equal ~msg:"gone once the GPU is reset" (list string) [] (memory_files root)

(* A reset of a GPU another process holds is refused, its memory untouched; the
   holder's exit gives the memory back. *)
let test_reset_held () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let root = Tree.make [ fn ] in
  let va = free_base + (6 * mib) in
  ignore (frames root va mib);
  let pid, feed, said = holder "wait" root fn.bus va in
  let line = In_channel.input_line (Unix.in_channel_of_descr said) in
  if line <> Some "holding" then begin
    Unix.close feed;
    ignore (wait_exit pid);
    fail "the holder did not hold the function"
  end;
  let held = memory_files root in
  equal ~msg:"the holder's memory" int 1 (List.length held);
  ignore (require_error (Gpus.reset (fixture_gpus ()) (Machine.at root) 0));
  equal ~msg:"kept while held" (list string) held (memory_files root);
  Unix.close feed;
  (match wait_exit pid with
  | WEXITED 0 -> ()
  | _ -> fail "the holder did not exit");
  equal ~msg:"given back at the holder's exit" (list string) []
    (memory_files root)

(* A released function's memory stays, and a later take of the machine allocates
   beside it in its huge page. *)
let test_released_block () =
  with_fixture @@ fun root f ->
  reserved f;
  let va = free_base + (2 * mib) and page = Machine.page Machine.this in
  ignore (frames root va (2 * page));
  let w, _ = given (Function.alloc_dma ~va f page) in
  Function.release f;
  equal ~msg:"kept while it holds memory" int 1
    (List.length (memory_files root));
  let f' = require_ok (Function.take (Function.machine f) (Function.bus f)) in
  let w', _ = given (Function.alloc_dma ~va:(va + page) f' page) in
  Function.free_dma f w;
  Function.free_dma f' w';
  Function.release f';
  equal ~msg:"gone once it lists no function and holds nothing" (list string) []
    (memory_files root)

(* A reset deletes the file of a process that died after it released its
   function, and keeps that of a process that lives. *)
let test_released_holder how () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let root = Tree.make [ fn ] in
  let va = free_base + (6 * mib) in
  ignore (frames root va mib);
  let pid, feed, said = holder how root fn.bus va in
  let reset () =
    require_ok (Gpus.reset (fixture_gpus ()) (Machine.at root) 0)
  in
  match how with
  | "released-die" ->
      Unix.close feed;
      Unix.close said;
      (match wait_exit pid with
      | WSIGNALED s when s = Sys.sigkill -> ()
      | _ -> fail "the holder did not die by SIGKILL");
      equal ~msg:"left by the dead process" int 1
        (List.length (memory_files root));
      reset ();
      equal ~msg:"gone at the reset" (list string) [] (memory_files root)
  | _ ->
      if In_channel.input_line (Unix.in_channel_of_descr said) <> Some "holding"
      then begin
        Unix.close feed;
        ignore (wait_exit pid);
        fail "the holder did not hold the function"
      end;
      let held = memory_files root in
      equal ~msg:"the holder's memory" int 1 (List.length held);
      reset ();
      equal ~msg:"kept while its process lives" (list string) held
        (memory_files root);
      Unix.close feed;
      (match wait_exit pid with
      | WEXITED 0 -> ()
      | _ -> fail "the holder did not exit");
      equal ~msg:"gone at its process's exit" (list string) []
        (memory_files root)

(* The memory of a function its process released, its bus mastering off, stays
   while the process lives, through takes of other functions of the machine. A
   death that runs no exit function leaves it, and the next take of any function
   of the machine deletes it. *)
let test_released_death () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" and other = "0000:04:00.0" in
  let root = Tree.make [ fn; Tree.gpu other ] in
  let m = Machine.at root and va = free_base + (6 * mib) in
  ignore (frames root va mib);
  let pid, feed, said = holder "released-quit" root fn.bus va in
  if In_channel.input_line (Unix.in_channel_of_descr said) <> Some "holding"
  then begin
    Unix.close feed;
    ignore (wait_exit pid);
    fail "the holder did not hold the function"
  end;
  let file = memory_file root in
  equal ~msg:"listing no function" string "" (reachers file);
  Function.release (require_ok (Function.take m other));
  equal ~msg:"kept through a take while its process lives" bool true
    (Sys.file_exists file);
  Unix.close feed;
  (match wait_exit pid with
  | WEXITED 0 -> ()
  | _ -> fail "the holder did not exit");
  equal ~msg:"left by its death" bool true (Sys.file_exists file);
  Function.release (require_ok (Function.take m other));
  equal ~msg:"gone at the next take" (list string) [] (hugepages root)

(* At a take, the files processes that died left go if they list no function:
   one with no list, which died being made, and one with an empty list. A list
   whose file is gone goes too. One that lists a function stays until that
   function's GPU is reset. *)
let test_left_files () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let gpu0 = "0000:03:00.0" and gpu1 = "0000:04:00.0" in
  let root = Tree.make [ Tree.gpu gpu0; Tree.gpu gpu1 ] in
  let add name s = Tree.add root ("dev/hugepages/" ^ name) s in
  add "rig-pci-1-1" "";
  add "rig-pci-2-2" "";
  add "rig-pci-2-2.reach" "";
  add "rig-pci-3-3" "";
  add "rig-pci-3-3.reach" (gpu1 ^ "\n");
  add "rig-pci-4-4.reach" (gpu1 ^ "\n");
  let m = Machine.at root in
  Function.release (require_ok (Function.take m gpu0));
  equal ~msg:"the file listing a function stays" (list string)
    [ "rig-pci-3-3"; "rig-pci-3-3.reach" ]
    (hugepages root);
  require_ok (Gpus.reset (fixture_gpus ()) m 1);
  equal ~msg:"gone at its GPU's reset" (list string) [] (hugepages root)

(* A driver that finds firmware it cannot continue from renews the GPU it holds:
   the vendor's reset runs, the memory of processes that died goes, and the
   process's own stays listed, so that its death leaves the GPU to be reset by
   the next open. *)
let test_renew () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let root = left_by_death () in
  let m = Machine.at root and resets = Atomic.make 0 in
  let renewing g =
    Gpus.open_ g m 0 (fun h _ -> Result.bind (Gpus.renew h) (fun () -> kept h))
  in
  let g = counted_gpus resets in
  Atomic.set resets 0;
  let h = require_ok (renewing g) in
  equal ~msg:"reset at open, then renewed" int 2 (Atomic.get resets);
  equal ~msg:"the dead process's memory given back" (list string) []
    (memory_files root);
  stop h;
  let root = left_by_death ~how:"renew-die" () in
  equal ~msg:"the renewing process's memory left" int 1
    (List.length (memory_files root));
  let resets = Atomic.make 0 in
  stop (require_ok (open_held (counted_gpus resets) (Machine.at root)));
  equal ~msg:"reset by the next open" int 1 (Atomic.get resets);
  equal ~msg:"and given back" (list string) [] (memory_files root)

(* Only a hugetlbfs of 2 MiB pages mounted at the machine's [dev/hugepages]
   gives huge pages that are one block of frames: memory on another mount is
   refused, naming the file system it needs. The mounts are the machine's
   [proc/self/mounts], where a mount covers the earlier ones it holds. *)
let other_mounts =
  [
    ("a tmpfs over dev", [ "tmpfs /dev tmpfs rw,nosuid 0 0" ]);
    ( "a hugetlbfs of 1 GiB pages",
      [ "none /dev/hugepages hugetlbfs rw,relatime,pagesize=1024M 0 0" ] );
    ( "a hugetlbfs elsewhere",
      [ "/dev/sda1 / ext4 rw 0 0"; "none /mnt/huge hugetlbfs rw,pagesize=2M 0 0" ]
    );
    ( "a hugetlbfs a later mount covers",
      [
        "none /dev/hugepages hugetlbfs rw,pagesize=2M 0 0";
        "tmpfs /dev/hugepages tmpfs rw 0 0";
      ] );
  ]

let test_other_mount (_, mounts) =
  with_fixture @@ fun root f ->
  reserved f;
  Tree.add root "proc/self/mounts" (String.concat "\n" mounts ^ "\n");
  let va = free_base + (2 * mib) in
  ignore (frames root va (300 * kib));
  contains ~sub:"hugetlbfs"
    (require_error (Function.alloc_dma ~va f (300 * kib)))

(* A page map read without the privilege gives every frame as 0: the refusal
   names it. *)
let test_unprivileged () =
  with_fixture @@ fun root f ->
  reserved f;
  let va = free_base + (2 * mib) and page = Machine.page Machine.this in
  Tree.pagemap root ~page va (List.init (2 * mib / page) (fun _ -> 0));
  contains ~sub:"run as root"
    (require_error (Function.alloc_dma ~va f (300 * kib)))

(* Memory whose addresses share a 2 MiB block shares its huge page, which goes
   back once neither holds it. *)
let test_shared_page () =
  with_fixture @@ fun root f ->
  reserved f;
  let va = free_base + (2 * mib) in
  let pas = frames root va (128 * kib) in
  let a, ra = given (Function.alloc_dma ~va f (64 * kib)) in
  let b, rb = given (Function.alloc_dma ~va:(va + (64 * kib)) f (64 * kib)) in
  let pa = List.hd pas in
  equal ~msg:"the first at its frames"
    (list (pair hex int))
    [ (pa, 64 * kib) ]
    ra;
  equal ~msg:"the second after it, in the same huge page"
    (list (pair hex int))
    [ (pa + (64 * kib), 64 * kib) ]
    rb;
  let file = memory_file root in
  equal ~msg:"one huge page" int huge (Tree.stored file);
  Function.free_dma f a;
  equal ~msg:"kept while the second holds it" int huge (Tree.stored file);
  Function.free_dma f b;
  equal ~msg:"given back once neither does" int 0 (Tree.stored file)

(* Memory handed out again in a huge page that stayed is zeroed. *)
let test_reused_zeroed () =
  with_fixture @@ fun root f ->
  reserved f;
  let va = free_base + (2 * mib) and page = Machine.page Machine.this in
  ignore (frames root va (2 * page));
  let a, _ = given (Function.alloc_dma ~va f page) in
  let b, _ = given (Function.alloc_dma ~va:(va + page) f page) in
  Window.write a 0 (String.make page 'x');
  Function.free_dma f a;
  let a, _ = given (Function.alloc_dma ~va f page) in
  equal ~msg:"zeroed" string (String.make page '\000') (Window.read a 0 page);
  Function.free_dma f a;
  Function.free_dma f b

(* Addresses whose 2 MiB block another machine's memory holds are refused. *)
let test_block_taken () =
  with_fixture @@ fun root f ->
  with_fixture @@ fun root' f' ->
  reserved f;
  reserved f';
  let va = free_base + (2 * mib) and page = Machine.page Machine.this in
  ignore (frames root va (2 * page));
  ignore (frames root' va (2 * page));
  let w, _ = given (Function.alloc_dma ~va f page) in
  ignore (require_error (Function.alloc_dma ~va:(va + page) f' page));
  Function.free_dma f w

(* The huge page around memory at a reserved address must be reserved whole. *)
let test_block_reserved () =
  with_fixture @@ fun root f ->
  let page = Machine.page Machine.this in
  let base = free_base + (16 * mib) + page in
  granted (Machine.reserve (Function.machine f) ~base (4 * mib));
  ignore (frames root base page);
  raises_match (Exn.invalid_arg ~substring:"2 MiB") (fun () ->
      Function.alloc_dma ~va:base f page)

let system_memory =
  group ~timeout:patience "system memory"
    [
      test "DMA memory, zeroed, at the frames of the huge pages that hold it"
        test_dma;
      test "a function taken physically is refused the process's pages"
        test_pin_physical;
      test
        "a function taken physically pins memory alloc_dma gave, its own or \
         another function's, until unpinned"
        test_pin_dma;
      test
        "a function taken physically is refused memory another machine's \
         function allocated"
        test_pin_other_machine;
      test "contiguous memory is one run at the reserved address asked"
        test_contiguous;
      test
        "a physical take's memory lies in a file of its own, gone once released"
        test_memory_file;
      test "a release leaves the memory of another machine's function"
        test_memory_machines;
      test
        "memory a killed process left stays through a release and goes at its \
         GPU's reset (SIGKILL in a child)"
        test_death;
      test
        "a reset of a GPU another process holds is refused, its memory kept (a \
         child holds)"
        test_reset_held;
      test
        "a renew resets the GPU held and gives back only what processes that \
         died left (SIGKILL in a child)"
        test_renew;
      test
        "a released function's memory stays, and a later take shares its page"
        test_released_block;
      test
        "a reset deletes the file of a process that died after releasing its \
         function (SIGKILL in a child)"
        (test_released_holder "released-die");
      test
        "a reset keeps the file of a living process that released its function \
         (a child holds)"
        (test_released_holder "released-wait");
      test
        "a released function's memory stays while its process lives and goes \
         at the first take after its death (a child holds, then _exits)"
        test_released_death;
      test
        "a take deletes the files processes that died left listing no \
         function, and keeps one that lists a function until its reset"
        test_left_files;
      cases
        "memory on a mount that is no hugetlbfs of 2 MiB pages is refused"
        ~name:fst other_mounts test_other_mount;
      test "memory whose frames read without the privilege is refused"
        test_unprivileged;
      test "memory in one 2 MiB block shares a huge page, gone once both are"
        test_shared_page;
      test "memory handed out again in a huge page is zeroed" test_reused_zeroed;
      test "a 2 MiB block another machine's memory holds is refused"
        test_block_taken;
      test "the 2 MiB block around memory must be reserved" test_block_reserved;
    ]

(* Taking and using a function of a machine's files *)

let test_take_no_bus () =
  let m = Machine.at (Tree.make []) in
  List.iter
    (fun bus ->
      equal ~msg:(String.escaped bus) (result pass string)
        (Error (strf "%S is no PCI bus address, expected DDDD:BB:DD.F" bus))
        (Function.take m bus))
    not_buses

(* After release, all but free_dma, unpin and wait raise. *)
let test_released () =
  with_fixture @@ fun root f ->
  reserved f;
  let va = free_base + (2 * mib) and page = Machine.page Machine.this in
  ignore (frames root va page);
  let d, _ = given (Function.alloc_dma ~va f page) in
  ignore (pin f va page : (int * int) list);
  Function.release f;
  List.iter
    (fun (name, use) ->
      raises_match ~msg:name (Exn.invalid_arg ~substring:"released") use)
    [
      ("failed", fun () -> ignore (Function.failed f : string option));
      ("config16", fun () -> ignore (Function.config16 f 0 : int));
      ("set_config16", fun () -> Function.set_config16 f 0 0);
      ("bar", fun () -> ignore (Function.bar f 0 : (int * int) option));
      ("map", fun () -> ignore (Function.map f 0 : _ result));
      ("interrupt", fun () -> ignore (Function.interrupt f 0 : bool));
      ("reset", fun () -> ignore (Function.reset f : _ result));
      ("alloc_dma", fun () -> ignore (Function.alloc_dma f page : _ result));
      ("pin", fun () -> ignore (Function.pin f va page : _ result));
      ("set_bus_master", fun () -> Function.set_bus_master f false);
    ];
  equal ~msg:"a wait" (result unit string) (Ok ())
    (Function.wait f ~us:0 "a wait" (fun () -> true));
  Function.free_dma f d;
  Function.unpin f va page;
  Function.release f

(* BAR 2 of the fixture's GPU is 2 MiB. *)
let bar2 = 2 * mib

let test_defaults () =
  with_fixture @@ fun _ f ->
  let lengths =
    List.map Window.length [ map f 2; map f 2 ~off:256; map f 2 ~length:16 ]
  in
  equal (list int) [ bar2; bar2 - 256; 16 ] lengths

(* Bytes [0, 0) at the BAR's size lie in the BAR, as a window's [sub] takes
   them. *)
let test_empty () =
  with_fixture @@ fun _ f ->
  equal ~msg:"at the start" int 0 (Window.length (map f 2 ~length:0));
  equal ~msg:"at the end" int 0 (Window.length (map f 2 ~off:bar2))

let test_other_machine () =
  with_fixture @@ fun _ f ->
  with_fixture @@ fun _ g ->
  let w = map g 2 in
  raises_match (Exn.invalid_arg ~substring:"") (fun () -> Function.unmap f w);
  Function.unmap g w

(* A function whose vendor ID reads all ones left the bus. *)
let test_left_bus () =
  with_fixture @@ fun _ f ->
  equal ~msg:"live" (option string) None (Function.failed f);
  Function.set_config16 f 0 0xffff;
  equal (option string)
    (Some (Function.bus f ^ " left the bus: its vendor ID reads 0xffff"))
    (Function.failed f)

(* Resets *)

(* The 16 bits at [off] of the fixture's configuration file, written as the
   function would change them. *)
let set_config_file root bus off v =
  let fd = Unix.openfile (config_file root bus) [ O_WRONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  let b = Bytes.create 2 in
  Bytes.set_uint16_le b 0 v;
  ignore (Unix.lseek fd off SEEK_SET);
  ignore (Unix.write fd b 0 2)

let reset_path bus = strf "sys/bus/pci/devices/%s/reset" bus
let read_text file = In_channel.with_open_bin file In_channel.input_all

(* The function answers again once its reset file was written: the reset waits
   for its vendor ID. *)
let test_reset_waits () =
  with_fixture @@ fun root f ->
  let bus = Function.bus f in
  Tree.add root (reset_path bus) "";
  let file = Filename.concat root (reset_path bus) in
  Function.set_config16 f 0 0xffff;
  let answering =
    Domain.spawn (fun () ->
        poll (fun () -> String.trim (read_text file) = "1")
        && begin
          set_config_file root bus 0 0x1002;
          true
        end)
  in
  let r = Function.reset f in
  equal ~msg:"the reset file written" bool true (Domain.join answering);
  equal ~msg:"the reset" (result unit string) (Ok ()) r;
  equal ~msg:"its vendor" hex 0x1002 (Function.config16 f 0)

(* A function whose vendor ID reads all ones does not answer. *)
let test_reset_silent () =
  with_fixture @@ fun root f ->
  Tree.add root (reset_path (Function.bus f)) "";
  Function.set_config16 f 0 0xffff;
  let why = require_error (Function.reset f) in
  contains ~msg:"names the function" ~sub:(Function.bus f) why;
  contains ~msg:"says it does not answer" ~sub:"does not answer" why

(* Linux has no reset for a function without a reset file. *)
let test_reset_no_file () =
  with_fixture @@ fun _ f ->
  contains ~sub:(Function.bus f ^ "/reset") (require_error (Function.reset f))

let resets =
  group ~timeout:patience "resets"
    [
      test "a reset waits for the function to answer" test_reset_waits;
      test "a function that does not answer after its reset fails"
        test_reset_silent;
      test "a function without a reset file is refused, naming it"
        test_reset_no_file;
    ]

(* Pins and DMA memory from two domains

   One GPU of a fixture tree, taken by each program and released at its end. Its
   memory lies at addresses of a range of its own, a 2 MiB block each, whose
   frames the tree's page map gives. *)

let dma_base = free_base + (64 * mib)
let dma_blocks = 32

(* A program takes a function at most once a step: the tree has a GPU for each
   step. *)
let dma_steps = 16

let dma_tree =
  lazy
    (let buses = List.init dma_steps (fun i -> strf "0000:%02x:00.0" (3 + i)) in
     let root = Tree.make (List.map Tree.gpu buses) in
     let m = Machine.at root in
     (match Machine.reserve m ~base:dma_base (dma_blocks * huge) with
     | Ok () -> ()
     | Error why -> skip ~reason:why ());
     ignore (frames root dma_base (dma_blocks * huge));
     (m, buses))

type dma_sys = {
  f : Function.t;
  blocks : int list Atomic.t;  (** The blocks it took. *)
  kept : Window.t list Atomic.t;  (** Memory alloc_dma gave, unnamed. *)
  made : Window.t list Atomic.t;  (** Memory alloc_dma gave, named. *)
}

type dma_ref = { mutable dmas : int }
type win_ref = { mutable pins : int; mutable freed : bool }

let rec push l x =
  let xs = Atomic.get l in
  if not (Atomic.compare_and_set l xs (x :: xs)) then push l x

let rec pop l =
  match Atomic.get l with
  | [] -> None
  | x :: rest as xs ->
      if Atomic.compare_and_set l xs rest then Some x else pop l

(* The blocks no program holds. A block goes back at the end of the program that
   took it, so that memory freed during a program is not given again in it. *)
let free_blocks = Atomic.make (List.init dma_blocks Fun.id)

(* A new page of memory, in a block of its own. *)
let alloc_block s =
  match pop free_blocks with
  | None -> failf "more than %d blocks" dma_blocks
  | Some k ->
      push s.blocks k;
      let va = dma_base + (k * huge) in
      fst (require_some (require_ok (Function.alloc_dma ~va s.f 4096)))

let rec unpin_all f w =
  match Function.unpin f (Window.address w) (Window.length w) with
  | () -> unpin_all f w
  | exception Invalid_argument _ -> ()

let free_live f w =
  match Function.free_dma f w with
  | () -> ()
  | exception Invalid_argument _ -> ()

(* At a program's end its function goes, then its memory, pins and blocks. *)
let release_dma s =
  Function.release s.f;
  List.iter
    (fun w ->
      unpin_all s.f w;
      free_live s.f w)
    (Atomic.get s.made);
  List.iter (free_live s.f) (Atomic.get s.kept);
  List.iter (push free_blocks) (Atomic.get s.blocks)

let dma_t = abstract "f" ~release:release_dma
let win_t = abstract "w"

(* The first GPU of the tree no take holds. *)
let take_dma () =
  let m, buses = Lazy.force dma_tree in
  let f =
    match
      List.find_map (fun b -> Result.to_option (Function.take m b)) buses
    with
    | Some f -> f
    | None -> failf "every one of the %d GPUs is taken" dma_steps
  in
  { f; blocks = Atomic.make []; kept = Atomic.make []; made = Atomic.make [] }

let pin_ref w =
  if w.freed then false
  else begin
    w.pins <- w.pins + 1;
    true
  end

let pin_sys (s, w) =
  Result.is_ok (Function.pin s.f (Window.address w) (Window.length w))

let unpin_ref w =
  if w.pins = 0 then invalid_arg "not pinned";
  cover "a page pinned twice, unpinned once" (w.pins > 1);
  w.pins <- w.pins - 1

let dma_commands =
  [
    Windtrap.command "take"
      (Gen.unit @-> makes dma_t)
      (fun () -> { dmas = 0 })
      take_dma;
    Windtrap.command "alloc_dma, named"
      (dma_t ^-> makes win_t)
      (fun _ -> { pins = 0; freed = false })
      (fun s ->
        let w = alloc_block s in
        push s.made w;
        (s, w));
    Windtrap.command "free_dma, named"
      ~pre:(fun w -> w.pins = 0 && not w.freed)
      (win_t ^-> returns unit)
      (fun w -> w.freed <- true)
      (fun (s, w) -> Function.free_dma s.f w);
    Windtrap.command "pin" (win_t ^-> returns bool) pin_ref pin_sys;
    Windtrap.command "unpin"
      (win_t ^-> returns unit)
      unpin_ref
      (fun (s, w) -> Function.unpin s.f (Window.address w) (Window.length w));
    Windtrap.command "alloc_dma"
      (dma_t ^-> returns unit)
      (fun r -> r.dmas <- r.dmas + 1)
      (fun s -> push s.kept (alloc_block s));
    Windtrap.command "free_dma"
      (dma_t ^-> returns unit)
      (fun r ->
        if r.dmas = 0 then raise Not_found;
        r.dmas <- r.dmas - 1)
      (fun s ->
        match pop s.kept with
        | None -> raise Not_found
        | Some w -> Function.free_dma s.f w);
  ]

(* 25 programs of 50 runs each, a hand-off between domains each, which waits for
   a time slice when the processors are busy: its limit is three patiences. *)
let dma_law = "pins and DMA memory are counted the same from two domains"

let dma_domains =
  group ~timeout:patience "two domains"
    [
      (if on_linux then
         stateful dma_law ~timeout:(3. *. patience) ~domains:2 ~count:25
           ~steps:dma_steps dma_commands
       else
         test dma_law (fun () ->
             skip ~reason:"flock on a function's file needs Linux" ()));
    ]

let uses =
  group ~timeout:patience "uses"
    [
      test "a string that is no bus address is refused" test_take_no_bus;
      test "a released function refuses all but free_dma, unpin and wait"
        test_released;
      test "a BAR window is the rest of the BAR from its offset by default"
        test_defaults;
      test "a window of no bytes inside a BAR is mapped" test_empty;
      test "another machine's window is refused" test_other_machine;
      test "a function whose vendor ID reads 0xffff left the bus" test_left_bus;
    ]

(* Misuse at the bounds raises Invalid_argument. *)
let misuse name use =
  test name (fun () ->
      with_fixture @@ fun _ f ->
      reserved f;
      raises_match (Exn.invalid_arg ~substring:"") (fun () ->
          use f (Machine.page Machine.this)))

let misuse_refused =
  group ~timeout:patience "misuse"
    [
      misuse "a BAR index below zero" (fun f _ -> Function.bar f (-1));
      misuse "the least BAR index" (fun f _ -> Function.bar f min_int);
      misuse "a map of a BAR index below zero" (fun f _ -> map f (-1));
      misuse "a pin off a page" (fun f page ->
          Function.pin f (free_base + 1) page);
      misuse "DMA memory at an address off a page" (fun f page ->
          Function.alloc_dma ~va:(free_base + 1) f page);
      misuse "contiguous DMA memory above 2 MiB" (fun f _ ->
          Function.alloc_dma ~contiguous:true f ((2 * mib) + 1));
      misuse "a huge page at an address off 2 MiB" (fun f page ->
          Function.alloc_dma ~contiguous:true ~va:(free_base + page) f (2 * page));
      misuse "DMA memory of no bytes" (fun f _ -> alloc_dma f 0);
      misuse "DMA memory of more bytes than an int holds" (fun f _ ->
          alloc_dma f max_int);
      misuse "a pin of no bytes" (fun f _ -> Function.pin f free_base 0);
      misuse "configuration space below its first byte" (fun f _ ->
          Function.config8 f (-1));
      misuse "configuration space past its 4096 bytes" (fun f _ ->
          Function.set_config32 f 4094 0);
      misuse "an interrupt wait below zero" (fun f _ ->
          Function.interrupt f (-1));
      misuse "DMA memory at an address no reservation holds" (fun f page ->
          Function.alloc_dma ~va:(free_base + (8 * mib)) f page);
      misuse "DMA memory that ends past its reservation" (fun f page ->
          Function.alloc_dma ~va:(free_base + (8 * mib) - page) f (2 * page));
      misuse "a wait of fewer than 0 us" (fun f _ ->
          Function.wait f ~us:(-1) "the fence" (fun () -> true));
    ]

(* Waits *)

let counter () =
  let n = ref 0 in
  ( n,
    fun k () ->
      incr n;
      !n >= k )

let waited = result unit string

(* The GPU of a fixture tree, taken once for the waits that leave it as it was,
   and released at the end of the run. *)
let waiter =
  fixture ~teardown:Function.release (fun () ->
      if not on_linux then
        skip ~reason:"flock on a function's file needs Linux" ();
      let fn = Tree.gpu "0000:03:00.0" in
      require_ok (Function.take (Machine.at (Tree.make [ fn ])) fn.bus))

let test_at_once () =
  let n, cond = counter () in
  equal ~msg:"result" waited (Ok ())
    (Function.wait (waiter ()) ~us:10_000_000 "the fence" (cond 1));
  equal ~msg:"calls" int 1 !n

let until_true =
  prop "a wait calls its condition until it holds, and no more"
    (Gen.int_range 1 200) (fun k ->
      let n, cond = counter () in
      equal ~msg:"result" waited (Ok ())
        (Function.wait (waiter ()) ~us:10_000_000 "the fence" (cond k));
      equal ~msg:"calls" int k !n)

let test_times_out () =
  let n, cond = counter () in
  let why =
    require_error
      (Function.wait (waiter ()) ~us:30_000 "the fence" (cond max_int))
  in
  starts_with ~msg:"names what it waited for" ~affix:"the fence" why;
  contains ~msg:"and its bound" ~sub:"30" why;
  at_least ~msg:"calls" int ~than:1 !n

(* Ten waits of each bound, below and past the first millisecond's spin, so that
   one cut short shows however the machine is loaded. *)
let test_full_time () =
  List.iter
    (fun us ->
      for _ = 1 to 10 do
        let t0 = now_ns () in
        ignore
          (Function.wait (waiter ()) ~us "a pause" (fun () -> false)
            : (unit, string) result);
        at_least
          ~msg:(strf "ns waited for %d us" us)
          int ~than:(us * 1000)
          (now_ns () - t0)
      done)
    [ 50; 2000 ]

(* A wait of 100 ms spins for its first millisecond only: the rest sleeps. *)
let test_naps () =
  let f = waiter () in
  let t0 = Sys.time () in
  ignore
    (Function.wait f ~us:100_000 "a pause" (fun () -> false)
      : (unit, string) result);
  less ~msg:"CPU ms" int ~than:50 (int_of_float ((Sys.time () -. t0) *. 1000.))

let test_zero () =
  let n, cond = counter () in
  equal ~msg:"result" waited (Ok ())
    (Function.wait (waiter ()) ~us:0 "the fence" (cond 1));
  equal ~msg:"calls" int 1 !n

(* A function that left the bus goes unnoticed until the bound, where the wait
   answers why. *)
let test_wait_left_bus () =
  with_fixture @@ fun _ f ->
  Function.set_config16 f 0 0xffff;
  let why =
    require_error (Function.wait f ~us:10_000 "the fence" (fun () -> false))
  in
  starts_with ~msg:"names what it waited for" ~affix:"the fence" why;
  contains ~msg:"names the cause" ~sub:"left the bus" why

let test_wait_released () =
  with_fixture @@ fun _ f ->
  Function.set_config16 f 0 0xffff;
  Function.release f;
  equal ~msg:"held" waited (Ok ())
    (Function.wait f ~us:10_000 "the fence" (fun () -> true));
  let why =
    require_error (Function.wait f ~us:10_000 "the fence" (fun () -> false))
  in
  starts_with ~msg:"names what it waited for" ~affix:"the fence" why;
  not_contains ~msg:"reads no configuration space" ~sub:"left the bus" why

let waits =
  group ~timeout:patience "waits"
    [
      test "a wait whose condition holds at once is Ok after one call"
        test_at_once;
      until_true;
      test
        "a wait whose condition never holds is an Error naming what it waited \
         for, asked at least once"
        test_times_out;
      test "a wait whose condition never holds lasts its whole time"
        test_full_time;
      test "a long wait holds no core" test_naps;
      test "a wait of 0 us asks its condition once (unstated)" test_zero;
      test
        "a wait on a function that left the bus answers, at its bound, that it \
         left"
        test_wait_left_bus;
      test "a released function waits as its machine does" test_wait_released;
    ]

let this_machine =
  group ~timeout:patience "this machine"
    [
      test "this machine without /sys/bus/pci refuses a take, naming the bus"
        test_refused_here;
      test "listing functions and taking GPUs change nothing on this machine"
        test_changes_nothing;
      test "a GPU taken here refuses a second take until released"
        test_held_here;
      test "a GPU behind an IOMMU reaches its memory at one run apart" test_vfio;
    ]

let () =
  match Sys.argv with
  | [| _; arg; root; bus |] when arg = exiting -> exit_mastering root bus
  | [| _; arg; how; root; bus; va |] when arg = holding ->
      hold_memory how root bus (int_of_string va)
  | _ ->
      hold_gpu ();
      exit
      @@ run "rig_pci.function"
           [
             uses;
             resets;
             waits;
             dma_domains;
             misuse_refused;
             tree_files;
             system_memory;
             this_machine;
           ]
