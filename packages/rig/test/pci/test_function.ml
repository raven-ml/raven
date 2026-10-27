(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Rig_pci
open Rig_pci_support

let strf = Printf.sprintf

external c_combines : Window.t -> bool = "rig_pci_test_combines"
external c_prefetchable : Window.t -> bool = "rig_pci_test_prefetchable"

let map ?combine ?off ?length f i =
  require_ok (Function.map ?combine ?off ?length f i)

let alloc_dma ?contiguous ?va f n =
  require_some (require_ok (Function.alloc_dma ?contiguous ?va f n))

let pin f a n = require_some (require_ok (Function.pin f a n))

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

let entry_limit = "/sys/module/vfio_iommu_type1/parameters/dma_entry_limit"

(* Each page alloc_dma gives or pin pins behind an IOMMU is one of the
   container's [dma_entry_limit] mappings and counts against the process's
   locked-memory limit: whichever runs out first, the next alloc_dma and pin
   are [Ok None]. *)
let test_vfio_limit () =
  let _, f = vfio_function () in
  Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
  let limit =
    In_channel.with_open_bin entry_limit In_channel.input_all
    |> String.trim |> int_of_string
  in
  let page = Machine.page Machine.this in
  let own = round_up (memory (2 * page)) page in
  let rec fill ws k =
    match Function.alloc_dma f page with
    | Ok (Some (w, _)) when k < limit -> fill (w :: ws) (k + 1)
    | r -> (r, ws)
  in
  let last, ws = fill [] 0 in
  Fun.protect ~finally:(fun () -> List.iter (Function.free_dma f) ws)
  @@ fun () ->
  (match require_ok ~msg:"alloc_dma: a limit is no refusal" last with
  | None -> ()
  | Some (w, _) ->
      Function.free_dma f w;
      failf "%d mappings, beyond the limit of %d" (limit + 1) limit);
  match
    require_ok ~msg:"pin: a limit is no refusal" (Function.pin f own page)
  with
  | None -> ()
  | Some _ ->
      Function.unpin f own page;
      failf "a pin beyond the limit"

(* The functions behind one machine's IOMMU share its container: pages pinned
   for two of them map once, at one device address. *)
let test_vfio_shared () =
  if not (Sys.file_exists "/dev/vfio") then skip ~reason:"no /dev/vfio" ();
  let fns =
    this_gpus ()
    |> List.filter_map (fun (d : Machine.id) ->
        match Function.take Machine.this d.bus with
        | Ok f when Function.addressing f = Iommu -> Some f
        | Ok f ->
            Function.release f;
            None
        | Error _ -> None)
  in
  Fun.protect ~finally:(fun () -> List.iter Function.release fns) @@ fun () ->
  match fns with
  | f :: g :: _ ->
      let page = Machine.page Machine.this in
      let a = round_up (memory (2 * page)) page in
      let runs = pin f a page in
      equal ~msg:"pinned for another function, at the same device address"
        (list (pair hex int))
        runs (pin g a page);
      Function.unpin f a page;
      Function.unpin g a page
  | _ ->
      skip
        ~reason:"fewer than two GPUs this process may take behind an IOMMU" ()

(* Behind an IOMMU, the process places a function's device addresses below
   its bound, which starts above the 4 GiB they start from. *)
let test_vfio_bits () =
  let _, f = vfio_function () in
  Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
  Function.set_dma_bits f 34;
  let w, runs = alloc_dma f (2 * mib) in
  List.iter
    (fun (a, n) -> less ~msg:"a run's end" int ~than:(1 + (1 lsl 34)) (a + n))
    runs;
  Function.free_dma f w;
  Function.set_dma_bits f 32;
  is_none ~msg:"below 4 GiB"
    (require_ok (Function.alloc_dma f (2 * mib)))

(* The GPUs this machine has behind an IOMMU, by bus. *)
let vfio_buses () =
  if not (Sys.file_exists "/dev/vfio") then skip ~reason:"no /dev/vfio" ();
  this_gpus ()
  |> List.filter_map (fun (d : Machine.id) ->
      match Function.take Machine.this d.bus with
      | Ok f ->
          let iommu = Function.addressing f = Iommu in
          Function.release f;
          if iommu then Some d.bus else None
      | Error _ -> None)

(* A child of fork that takes another GPU behind the IOMMU maps its memory in
   a container of its own: the parent's next allocation, at the device
   address the child's would take in a shared one, still maps. *)
let test_vfio_fork () =
  match vfio_buses () with
  | b :: b' :: _ ->
      let f = require_ok (Function.take Machine.this b) in
      Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
      let w, _ = alloc_dma f mib in
      let ready_r, ready_w = Unix.pipe () and done_r, done_w = Unix.pipe () in
      let child =
        match Unix.fork () with
        | 0 ->
            let ok =
              match Function.take Machine.this b' with
              | Ok g -> (
                  match Function.alloc_dma g mib with
                  | Ok (Some _) -> true
                  | Ok None | Error _ -> false)
              | Error _ -> false
            in
            ignore (Unix.write_substring ready_w (if ok then "y" else "n") 0 1);
            ignore (Unix.read done_r (Bytes.create 1) 0 1);
            Unix._exit 0
        | pid -> pid
      in
      let answer = Bytes.create 1 in
      ignore (Unix.read ready_r answer 0 1);
      let parent = Function.alloc_dma f mib in
      ignore (Unix.write_substring done_w "x" 0 1);
      ignore (Unix.waitpid [] child);
      List.iter Unix.close [ ready_r; ready_w; done_r; done_w ];
      equal ~msg:"the child's allocation" string "y" (Bytes.to_string answer);
      (match require_ok ~msg:"the parent's allocation" parent with
      | Some (w', _) -> Function.free_dma f w'
      | None -> fail "the parent's allocation found no memory");
      Function.free_dma f w
  | _ ->
      skip
        ~reason:"fewer than two GPUs this process may take behind an IOMMU" ()

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
      [ "0000:03:00.0 is bound to the driver amdgpu"; "detach it first" ] );
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
      [ "0000:03:00.0 shares its device with 0000:03:00.1"; "detach it first" ]
    );
    ( "a device shared with another function, bound to vfio-pci without an IOMMU",
      [ Tree.gpu ~driver:"vfio-pci" "0000:03:00.0"; audio "0000:03:00.1" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 shares its device with 0000:03:00.1" ] );
    ( "a function on vfio-pci in no IOMMU group",
      [ Tree.gpu ~driver:"vfio-pci" "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 is bound to vfio-pci but in no IOMMU group" ] );
    ( "a disabled function",
      [ Tree.gpu ~enabled:false "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 is disabled"; "detach it first" ] );
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

(* A take reads a function the machine has under its lock, which needs flock. *)
let test_refusal (_, fns, bus, opts, subs) =
  if List.exists (fun (fn : Tree.fn) -> fn.bus = bus) fns && not flocks then
    skip ~reason:"locking a function's file needs flock" ();
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

(* Linux resets a function with the first method its [reset_method] lists,
   and refuses [bus] and [cxl_bus] while another function shares the bus: a
   function sharing its device is taken physically unless they reset it. A
   kernel older than [reset_method] lists no order, so its function with a
   [reset] counts as reset through its bus. *)
let shared_resets =
  [
    ("a reset of its own first", `Methods "flr bus", true);
    ("power management alone", `Methods "pm", true);
    ("no reset", `None, true);
    ("the bus's", `Methods "bus", false);
    ("a CXL bus's", `Methods "cxl_bus", false);
    ("a reset without reset_method", `Old_kernel, false);
  ]

let test_shared_reset (_, reset, taken) =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let bus = "0000:03:00.0" in
  let methods =
    match reset with `Methods m -> Some m | `None | `Old_kernel -> None
  in
  let root = Tree.make [ Tree.gpu ~reset:methods bus; audio "0000:03:00.1" ] in
  if reset = `Old_kernel then
    Tree.add root ("sys/bus/pci/devices/" ^ bus ^ "/reset") "";
  let take = Function.take (Machine.at root) bus in
  if taken then Function.release (require_ok take)
  else
    contains ~sub:"0000:03:00.0 shares its device with 0000:03:00.1"
      (require_error take)

(* An identity IOMMU passes physical addresses through, as does VFIO's no-IOMMU
   mode: a function under either is taken physically. On Linux the take locks
   its configuration file; elsewhere flock is refused. *)
let test_physical () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
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

(* A take reads how the function is taken under the lock that takes and changes
   hold, so that no change comes between: a function the lock's holder binds to
   vfio-pci meanwhile, as an attach would, is refused as taken. Another function
   bound to vfio-pci in group 12 gives the tree the driver and the group. *)
let test_take_locked () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let root =
    Tree.make [ fn; Tree.gpu ~driver:"vfio-pci" ~group:"12" "0000:43:00.0" ]
  in
  let m = Machine.at root in
  let f = require_ok (Function.take m fn.bus) in
  Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
  let d = "sys/bus/pci/devices/0000:03:00.0/" in
  Tree.link root (d ^ "driver")
    (Filename.concat root "sys/bus/pci/drivers/vfio-pci");
  Tree.link root (d ^ "iommu_group")
    (Filename.concat root "sys/kernel/iommu_groups/12");
  contains ~sub:"0000:03:00.0 is taken already"
    (require_error (Function.take m fn.bus))

(* A BAR's file is as long as the BAR, and maps whole. A file shorter than its
   BAR would end the process with SIGBUS at the first access past its end, so
   the map refuses it, naming the file. *)
let test_map_files () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
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
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
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
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
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
      if not flocks then
        skip ~reason:"locking a function's file needs flock" ();
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

(* Behind an IOMMU a write reaches the function: its bus master bit reads back
   as set_bus_master sets it. *)
let test_vfio_bus_master () =
  let _, f = vfio_function () in
  Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
  Function.set_bus_master f true;
  equal ~msg:"set" hex bus_master (Function.config16 f command land bus_master);
  Function.set_bus_master f false;
  equal ~msg:"cleared" hex 0 (Function.config16 f command land bus_master)

(* A function taken physically, with no interrupt route through VFIO, signals no
   legacy interrupt while taken: nothing handles it, and bus mastering does not
   gate it. Release gives its INTx back as it found it. *)
let test_intx (_, before) =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
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
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
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
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
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

(* A window on a BAR whose register marks it prefetchable takes wide copies
   whatever [combine], as rig_pci.h sees it: BARs 0 and 2 of the fixture's GPU
   are prefetchable, BAR 5 is not. *)
let test_prefetchable () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let f = require_ok (Function.take (Machine.at (Tree.make [ fn ])) fn.bus) in
  Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
  List.iter
    (fun (msg, combine, i, want) ->
      let w = map ~combine ~length:4096 f i in
      equal ~msg bool want (c_prefetchable w);
      Function.unmap f w)
    [
      ("a prefetchable BAR, combining", true, 0, true);
      ("a prefetchable BAR, not combining", false, 0, true);
      ("another prefetchable BAR", false, 2, true);
      ("a BAR that is not prefetchable", false, 5, false);
    ]

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
      cases
        "a function sharing its device is taken physically unless Linux \
         resets it through its bus"
        ~name:(fun (n, _, _) -> n)
        shared_resets test_shared_reset;
      test
        "a function the lock's holder binds to vfio-pci is refused as taken"
        test_take_locked;
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
      test
        "a window on a prefetchable BAR takes wide copies whatever combine, \
         and one on another BAR does not"
        test_prefetchable;
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
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
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
   keep their pages until no function reaches them. Each function that reaches a
   file is named by an empty file beside it, [<file>.reach-<bus>]. *)

let memory_files root =
  Sys.readdir (Filename.concat root "dev/hugepages")
  |> Array.to_list
  |> List.filter (fun f ->
      String.starts_with ~prefix:"rig-pci-" f && not (String.contains f '.'))

let memory_file root =
  match memory_files root with
  | [ f ] -> Filename.concat root ("dev/hugepages/" ^ f)
  | fs -> failf "%d memory files" (List.length fs)

(* Every entry of the machine's [dev/hugepages]: files and their markers. *)
let hugepages root =
  Sys.readdir (Filename.concat root "dev/hugepages")
  |> Array.to_list |> List.sort compare

(* The buses the markers of [file] name. *)
let reachers file =
  let marker = Filename.basename file ^ ".reach-" in
  Sys.readdir (Filename.dirname file)
  |> Array.to_list
  |> List.filter_map (fun f ->
      if String.starts_with ~prefix:marker f then
        Some (String.sub f (String.length marker)
                (String.length f - String.length marker))
      else None)
  |> List.sort compare

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
  equal ~msg:"locked by its process" bool true (Tree.held file);
  equal ~msg:"listing the function" (list string) [ Function.bus f ]
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

(* A function reaches system memory below its bound alone: alloc_dma and pin
   are [Ok None] for memory past [2^n], and memory below it comes as before. *)
let test_dma_bits () =
  with_fixtures 2 @@ fun root fns ->
  let f = List.nth fns 0 and g = List.nth fns 1 in
  reserved f;
  let page = Machine.page Machine.this in
  let va = free_base + (2 * mib) in
  let top = List.fold_left max 0 (frames root va (64 * kib)) + page in
  let rec bits n = if 1 lsl n >= top then n else bits (n + 1) in
  let fits = bits 1 in
  Function.set_dma_bits f (fits - 1);
  is_none ~msg:"alloc_dma past the bound"
    (require_ok (Function.alloc_dma ~va f (64 * kib)));
  Function.set_dma_bits f fits;
  let w, runs = given (Function.alloc_dma ~va f (64 * kib)) in
  let n = Window.length w in
  Function.set_dma_bits g (fits - 1);
  is_none ~msg:"pin past the bound" (require_ok (Function.pin g va n));
  Function.set_dma_bits g fits;
  equal ~msg:"pin within the bound"
    (list (pair hex int))
    runs (pin g va n);
  Function.unpin g va n;
  Function.free_dma f w;
  raises_match ~msg:"no bits" Exn.invalid_arg (fun () ->
      Function.set_dma_bits f 0);
  raises_match ~msg:"past 64 bits" Exn.invalid_arg (fun () ->
      Function.set_dma_bits f 65)

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

let fixture_gpus ?(reset = fun _ -> Ok ()) ?(ready = fun _ -> Ok ()) () =
  Driver.make ~name:"fixture" ~memory_bar:(Some 0) ~siblings:Remove
    ~nodes:(fun ~root:_ _ -> [])
    ~unreleased:(fun ~root:_ _ -> None)
    ~teardown_ms:0 ~reset ~ready
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
      (Driver.open_ g (Machine.at root) 0 (fun h f ->
           ok (Machine.reserve (Function.machine f) ~base:free_base (8 * mib));
           if Option.is_none (ok (Function.alloc_dma ~va f mib)) then
             exit refused_code;
           ok (Driver.renew h);
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
  Driver.set_stop h (fun () -> `Clean);
  Ok h

let open_held g m = Driver.open_ g m 0 (fun h _ -> kept h)
let stop h = ignore (Driver.stop h : [ `Stopped | `Unknown ])

(* What a process that died left stays through another take and release, which
   does not reset the GPU, and goes once its GPU is reset. *)
let death root (fn : Tree.fn) =
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
  require_ok (Driver.reset (fixture_gpus ()) m 0);
  equal ~msg:"gone once the GPU is reset" (list string) [] (memory_files root)

let test_death () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let fn = Tree.gpu "0000:03:00.0" in
  death (Tree.make [ fn ]) fn

(* A reset that answers, of a function its driver then finds not ready, gives
   back what a dead process left all the same, and loses the function: the
   next open resets it again. *)
let test_not_ready () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let root = left_by_death () in
  equal ~msg:"left by the dead process" int 1 (List.length (memory_files root));
  let resets = Atomic.make 0 in
  let g =
    fixture_gpus
      ~reset:(fun _ ->
        Atomic.incr resets;
        Ok ())
      ~ready:(fun _ -> Error "not ready")
      ()
  in
  let m = Machine.at root in
  contains ~msg:"the driver's reason" ~sub:"not ready"
    (require_error (Driver.reset g m 0));
  equal ~msg:"gone though not ready" (list string) [] (memory_files root);
  ignore (open_held g m);
  equal ~msg:"the next open resets it" int 2 (Atomic.get resets)

(* A directory of its own on this machine's hugetlbfs of 2 MiB pages at
   /dev/hugepages, if it has a free page and this process may write it. A
   hugetlbfs refuses write(2), which an ordinary file system's file accepts, so
   only it shows that what the memory files record survives that refusal. *)
let hugetlbfs_dir () =
  let mounted line =
    match String.split_on_char ' ' line with
    | _ :: "/dev/hugepages" :: "hugetlbfs" :: options :: _ ->
        List.mem "pagesize=2M" (String.split_on_char ',' options)
    | _ -> false
  in
  let free line =
    match String.split_on_char ':' line with
    | [ "HugePages_Free"; n ] -> int_of_string (String.trim n) > 0
    | _ -> false
  in
  let lines file = In_channel.with_open_text file In_channel.input_lines in
  if not (on_linux && List.exists mounted (lines "/proc/self/mounts")) then None
  else if not (List.exists free (lines "/proc/meminfo")) then None
  else
    let dir = strf "/dev/hugepages/rig-pci-test-%d" (Unix.getpid ()) in
    match Unix.mkdir dir 0o700 with
    | () -> Some dir
    | exception Unix.Unix_error _ -> None

(* As [test_death], with the machine's memory files on a real hugetlbfs. *)
let test_death_hugetlbfs () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  match hugetlbfs_dir () with
  | None ->
      skip
        ~reason:
          "no hugetlbfs of 2 MiB pages at /dev/hugepages with a free page this \
           process may write"
        ()
  | Some dir ->
      let fn = Tree.gpu "0000:03:00.0" in
      let root = Tree.make [ fn ] in
      let hugepages = Filename.concat root "dev/hugepages" in
      Unix.rmdir hugepages;
      Unix.symlink dir hugepages;
      let clear () =
        Array.iter
          (fun f -> Sys.remove (Filename.concat dir f))
          (Sys.readdir dir);
        Unix.rmdir dir
      in
      Fun.protect ~finally:clear (fun () -> death root fn)

(* A reset of a GPU another process holds is refused, its memory untouched; the
   holder's exit gives the memory back. *)
let test_reset_held () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
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
  ignore (require_error (Driver.reset (fixture_gpus ()) (Machine.at root) 0));
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
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let root = Tree.make [ fn ] in
  let va = free_base + (6 * mib) in
  ignore (frames root va mib);
  let pid, feed, said = holder how root fn.bus va in
  let reset () =
    require_ok (Driver.reset (fixture_gpus ()) (Machine.at root) 0)
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
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
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
  equal ~msg:"listing no function" (list string) [] (reachers file);
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

(* At a take, the files processes that died left go if they list no function,
   as one that died being made lists none. A marker whose file is gone goes
   too. One that lists a function stays until that function's GPU is reset. *)
let test_left_files () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let gpu0 = "0000:03:00.0" and gpu1 = "0000:04:00.0" in
  let root = Tree.make [ Tree.gpu gpu0; Tree.gpu gpu1 ] in
  let add name = Tree.add root ("dev/hugepages/" ^ name) "" in
  add "rig-pci-1-1";
  add "rig-pci-3-3";
  add ("rig-pci-3-3.reach-" ^ gpu1);
  add ("rig-pci-4-4.reach-" ^ gpu1);
  let m = Machine.at root in
  Function.release (require_ok (Function.take m gpu0));
  equal ~msg:"the file listing a function stays" (list string)
    [ "rig-pci-3-3"; "rig-pci-3-3.reach-" ^ gpu1 ]
    (hugepages root);
  require_ok (Driver.reset (fixture_gpus ()) m 1);
  equal ~msg:"gone at its GPU's reset" (list string) [] (hugepages root)

(* Memory a dead process left reaching a function is read as left while
   another process holds its file open, as a sweep judging it does, since only
   its process's lock tells it alive; memory whose process lives is not, and
   stays. *)
let test_left_while_swept () =
  if not ofd_locks then
    skip ~reason:"telling a dead process's memory needs file locks" ();
  let gpu0 = "0000:03:00.0" in
  let root = Tree.make [ Tree.gpu gpu0 ] in
  let add name = Tree.add root ("dev/hugepages/" ^ name) "" in
  add "rig-pci-3-3";
  add ("rig-pci-3-3.reach-" ^ gpu0);
  let dir = Filename.concat root "dev/hugepages" in
  let file = Filename.concat dir "rig-pci-3-3" in
  let m = Machine.at root in
  let inherited () =
    let f = require_ok (Function.take m gpu0) in
    Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
    Function.inherited f
  in
  let judge = Unix.openfile file [ O_RDONLY; O_CLOEXEC ] 0 in
  equal ~msg:"a dead process's memory, open elsewhere" bool true
    (inherited ());
  Unix.close judge;
  let owner = Tree.hold file in
  equal ~msg:"a live process's memory" bool false (inherited ());
  equal ~msg:"kept for its live process" (list string)
    [ "rig-pci-3-3"; "rig-pci-3-3.reach-" ^ gpu0 ]
    (hugepages root);
  Unix.close owner

(* Two functions' resets after a dead process's file listed both: each
   reset's sweep removes its function's marker, and the file goes once
   neither is listed, in whichever order the resets run. One process's
   sweeps take turns; two processes' at once are argued in sysmem.ml. *)
type dead_file = { root : string; m : Machine.t; gpus : Driver.t }
type dead_model = { mutable listed : string list }

let dead_buses = [ "0000:03:00.0"; "0000:04:00.0" ]

let make_dead_file () =
  if not ofd_locks then
    skip ~reason:"telling a dead process's memory needs file locks" ();
  let root = Tree.make (List.map Tree.gpu dead_buses) in
  let add name = Tree.add root ("dev/hugepages/" ^ name) "" in
  add "rig-pci-3-3";
  List.iter (fun bus -> add ("rig-pci-3-3.reach-" ^ bus)) dead_buses;
  { root; m = Machine.at root; gpus = fixture_gpus () }

(* The machine's files, as the reference lists them. *)
let dead_listing r =
  match r.listed with
  | [] -> []
  | buses ->
      List.sort compare
        ("rig-pci-3-3" :: List.map (fun b -> "rig-pci-3-3.reach-" ^ b) buses)

let dead_file =
  abstract
    ~pp:(fun ppf r ->
      Format.fprintf ppf "listing %s" (String.concat ", " r.listed))
    "f"

let gpu_index = Gen.with_pp Format.pp_print_int (Gen.int_range 0 1)

let dead_file_commands =
  [
    Windtrap.command "open"
      (Gen.unit @-> makes dead_file)
      (fun () -> { listed = dead_buses })
      make_dead_file;
    Windtrap.command "reset"
      (dead_file ^-> gpu_index @-> returns unit)
      (fun r i ->
        let bus = List.nth dead_buses i in
        r.listed <- List.filter (( <> ) bus) r.listed)
      (fun t i -> require_ok (Driver.reset t.gpus t.m i));
    Windtrap.command "files"
      (dead_file ^-> returns (list string))
      dead_listing
      (fun t -> hugepages t.root);
  ]

(* A driver that finds firmware it cannot continue from renews the GPU it holds:
   the vendor's reset runs, the memory of processes that died goes, and the
   process's own stays listed, so that its death leaves the GPU to be reset by
   the next open. *)
let test_renew () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let root = left_by_death () in
  let m = Machine.at root and resets = Atomic.make 0 in
  let renewing g =
    Driver.open_ g m 0 (fun h _ -> Result.bind (Driver.renew h) (fun () -> kept h))
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
        "alloc_dma and pin give a function only memory below its DMA bound"
        test_dma_bits;
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
        "memory a killed process left on a hugetlbfs stays through a release \
         and goes at its GPU's reset (SIGKILL in a child)"
        test_death_hugetlbfs;
      test
        "a reset of a function its driver finds not ready gives back what a \
         killed process left and loses it (SIGKILL in a child)"
        test_not_ready;
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
      test
        "a take reads the memory a dead process left as left while its file \
         is open elsewhere, and a live process's as its own"
        test_left_while_swept;
      stateful ~count:10 ~steps:4
        "a dead process's file goes once the resets of every function it \
         lists ran"
        dead_file_commands;
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
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let fn = Tree.gpu ~reset:None "0000:03:00.0" in
  let f = require_ok (Function.take (Machine.at (Tree.make [ fn ])) fn.bus) in
  Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
  contains ~sub:(fn.bus ^ "/reset") (require_error (Function.reset f))

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
  match Function.pin s.f (Window.address w) (Window.length w) with
  | Ok (Some _) -> true
  | Ok None | Error _ -> false

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

(* Releases and finishes of the releases from two domains at once. A program
   takes a function at most once a step, the one released last first, with
   windows on three BARs and its bus mastering on, and opens files that take
   the numbers its release closes. The release that begins gives the function
   back before it returns, whatever finish runs meanwhile, and a finish gives
   back what a release an exception cut left. At its end the function is
   released, every file it opened holds what it wrote, and the function is
   taken again. *)
let release_steps = 6

let release_tree =
  lazy
    (let buses =
       List.init release_steps (fun i -> strf "0000:%02x:00.0" (3 + i))
     in
     let root = Tree.make (List.map Tree.gpu buses) in
     (root, Machine.at root, buses))

type release_sys = {
  fn : Function.t;
  files : (string * Unix.file_descr) list Atomic.t;  (** The files opened. *)
}

type release_ref = { mutable gone : bool }

(* What each file opened holds, and the count of files opened, which names
   the next. *)
let release_canary = "canary"
let release_files = Atomic.make 0

(* The bus released last, which the next take tries first. *)
let last_released = Atomic.make ""

let take_released () =
  let _, m, buses = Lazy.force release_tree in
  let first = Atomic.get last_released in
  let buses = first :: List.filter (( <> ) first) buses in
  match List.find_map (fun b -> Result.to_option (Function.take m b)) buses with
  | None -> failf "every one of the %d GPUs is taken" release_steps
  | Some f ->
      Function.set_bus_master f true;
      List.iter (fun i -> ignore (map ~length:4096 f i : Window.t)) [ 0; 2; 5 ];
      { fn = f; files = Atomic.make [] }

(* Whether a take holds [s]'s function: the process has its configuration
   file open. A probe that locks the file would be seen by another domain's
   as a take. *)
let taken s =
  let root, _, _ = Lazy.force release_tree in
  Tree.opened (config_file root (Function.bus s.fn)) > 0

(* Releases [s]'s function, and is whether it is given back as the release
   returns. *)
let release_sys s =
  Function.release s.fn;
  Atomic.set last_released (Function.bus s.fn);
  not (taken s)

(* Releases [s]'s function with an exception at the [n]-th allocation of the
   release, then finishes the releases, and is whether the finish gave back
   what the release left. *)
let cut_sys n s =
  ignore (Rig_support.interrupted n (fun () -> Function.release s.fn) : bool);
  let left = Function.released s.fn && taken s in
  Function.finish_releases ();
  Atomic.set last_released (Function.bus s.fn);
  left && not (taken s)

(* The first release gives the function back; one after returns at once. *)
let release_ref r outcome =
  match outcome with
  | Error e -> failf "the release raised %s" (Printexc.to_string e)
  | Ok given ->
      if not r.gone then begin
        r.gone <- true;
        equal ~msg:"the function given back as the release returned" bool
          true given
      end

let cut_ref _ r outcome =
  match outcome with
  | Error e -> failf "the cut release raised %s" (Printexc.to_string e)
  | Ok finished ->
      let first = not r.gone in
      r.gone <- true;
      if first then cover "a finish gave back what a cut release left" finished

let open_file s =
  let root, _, _ = Lazy.force release_tree in
  let file =
    Filename.concat root
      (strf "release-file-%d" (Atomic.fetch_and_add release_files 1))
  in
  let fd = Unix.openfile file [ O_RDWR; O_CREAT; O_TRUNC; O_CLOEXEC ] 0o600 in
  ignore
    (Unix.write_substring fd release_canary 0 (String.length release_canary)
      : int);
  push s.files (file, fd)

let end_released s =
  Function.release s.fn;
  Function.finish_releases ();
  List.iter
    (fun (file, fd) ->
      equal ~msg:"a file opened during the releases" string release_canary
        (In_channel.with_open_bin file In_channel.input_all);
      Unix.close fd;
      Sys.remove file)
    (Atomic.get s.files);
  match Function.take (Function.machine s.fn) (Function.bus s.fn) with
  | Ok f -> Function.release f
  | Error why -> failf "the take after the releases: %s" why

let released_t = abstract "f" ~release:end_released

let release_commands =
  [
    Windtrap.command "take"
      (Gen.unit @-> makes released_t)
      (fun () -> { gone = false })
      take_released;
    Windtrap.command "release"
      (released_t ^-> judges bool)
      release_ref release_sys;
    Windtrap.command "release, cut, then finish"
      (Gen.int_range 1 24 @-> released_t ^-> judges bool)
      cut_ref cut_sys;
    Windtrap.command "released"
      (released_t ^-> returns bool)
      (fun r -> r.gone)
      (fun s -> Function.released s.fn);
    Windtrap.command "finish the releases"
      (Gen.unit @-> returns unit)
      (fun () -> ())
      Function.finish_releases;
    Windtrap.command "open a file"
      (released_t ^-> returns unit)
      (fun _ -> ())
      open_file;
  ]

let release_law =
  "releases and finishes from two domains give a function back once"

let dma_domains =
  group ~timeout:patience "two domains"
    [
      (if on_linux then
         stateful dma_law ~timeout:(3. *. patience) ~domains:2 ~count:25
           ~steps:dma_steps dma_commands
       else
         test dma_law (fun () ->
             skip ~reason:"system memory for a function needs Linux" ()));
      (if flocks then
         stateful release_law ~timeout:(3. *. patience) ~domains:2 ~count:25
           ~steps:release_steps release_commands
       else
         test release_law (fun () ->
             skip ~reason:"locking a function's file needs flock" ()));
    ]

(* A release that finds a descriptor of its take closed behind its back, as
   only a bug of the library's closes one, raises [Failure], and closes none
   of the process's files twice: the rest of the take's descriptors close as
   its release finishes. *)
let test_release_closed () =
  if not on_linux then skip ~reason:"finding a descriptor needs Linux" ();
  with_fixture @@ fun root f ->
  let config = config_file root (Function.bus f) in
  equal ~msg:"the take's lock found" bool true (Tree.close_lock config);
  raises_match
    (Exn.failure ~substring:"Bad file descriptor")
    (fun () -> Function.release f);
  Function.finish_releases ();
  equal ~msg:"descriptors left on the configuration space" int 0
    (Tree.opened config)

(* A release called from a callback inside a release, at any of its
   allocations, returns at once. A finish of the releases there raises
   [Invalid_argument] inside a lock and returns between locks. The release
   around it gives the function back. *)
let test_release_inside () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let bus = "0000:03:00.0" in
  let m = Machine.at (Tree.make [ Tree.gpu bus ]) in
  let refused = ref false and returned = ref false in
  let rec walk n =
    let f = take_mastering m bus in
    List.iter (fun i -> ignore (map ~length:4096 f i : Window.t)) [ 0; 2; 5 ];
    let finished = ref None in
    let inside () =
      let held = Rig_support.holds () <> 0 in
      Function.release f;
      finished :=
        Some
          ( held,
            match Function.finish_releases () with
            | () -> "returned"
            | exception Invalid_argument _ -> "refused" )
    in
    let reached =
      Rig_support.interrupted ~landing:(Call inside) n (fun () ->
          Function.release f)
    in
    let at what = strf "%s, called at %d" what n in
    if n = 1 then equal ~msg:"the walk lands" bool true reached;
    (match !finished with
    | Some (held, answer) ->
        if held then refused := true else returned := true;
        equal ~msg:(at "the finish inside") string
          (if held then "refused" else "returned")
          answer
    | None -> ());
    (match Function.take m bus with
    | Ok f -> Function.release f
    | Error why -> failf "%s: %s" (at "the take after") why);
    if reached then walk (n + 1)
  in
  walk 1;
  equal ~msg:"landings inside a lock and between locks" (pair bool bool)
    (true, true) (!refused, !returned)

(* A finish of the releases from a callback inside a lock raises
   [Invalid_argument], and leaves a release an exception cut to the next
   finish. *)
let test_finish_inside_lock () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let bus = "0000:03:00.0" in
  let root = Tree.make [ Tree.gpu bus ] in
  let f = take_mastering (Machine.at root) bus in
  ignore (Rig_support.interrupted 1 (fun () -> Function.release f) : bool);
  raises_match Exn.invalid_arg (fun () ->
      Rig_support.inside_lock Function.finish_releases);
  equal ~msg:"taken after the refusal" bool true
    (Tree.flocked (config_file root (Function.bus f)));
  Function.finish_releases ();
  equal ~msg:"taken after the next finish" bool false
    (Tree.flocked (config_file root (Function.bus f)))

let uses =
  group ~timeout:patience "uses"
    [
      test "a string that is no bus address is refused" test_take_no_bus;
      test "a released function refuses all but free_dma, unpin and wait"
        test_released;
      test
        "a release that finds a descriptor closed behind it raises Failure, \
         closing the rest once"
        test_release_closed;
      test
        "a release called inside a release returns at once, and a finish of \
         the releases there raises"
        test_release_inside;
      test "a finish of the releases from inside a lock raises"
        test_finish_inside_lock;
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
      if not flocks then
        skip ~reason:"locking a function's file needs flock" ();
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
          (Function.wait (waiter ()) ~us "the fence" (fun () -> false)
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
    (Function.wait f ~us:100_000 "the fence" (fun () -> false)
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

(* Interrupts *)

(* A function of a fixture of one GPU, taken physically, which the test
   releases. *)
let fixture () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let bus = "0000:03:00.0" in
  let root = Tree.make [ Tree.gpu bus ] in
  (root, require_ok (Function.take (Machine.at root) bus))

(* [f]'s release gives it back as it returns: a release leaves the give-back
   to the last pin, DMA allocation or wait that uses [f], so a use an
   exception left open would leave [f] taken for good. *)
let releases root f =
  Function.release f;
  equal ~msg:"taken after the release" bool false
    (Tree.flocked (config_file root (Function.bus f)))

(* An exception at any allocation of a wait that reads the configuration space
   at its bound ends the wait's use of the function: the release gives it
   back. *)
let test_interrupted_wait () =
  let root, f = fixture () in
  Function.set_config16 f 0 0xffff;
  let wait () =
    ignore (Function.wait f ~us:0 "the fence" (fun () -> false) : _ result)
  in
  let rec from n = if Rig_support.interrupted n wait then from (n + 1) in
  from 1;
  releases root f

(* An exception at any allocation of a DMA allocation and its free leaves the
   function's lock free and ends the allocation's use of the function: the
   next allocation runs, and the release gives the function back. *)
let test_interrupted_dma () =
  let root, f = fixture () in
  reserved f;
  let page = Machine.page Machine.this in
  ignore (frames root free_base (8 * mib));
  let next = ref free_base in
  let dma () =
    let va = !next in
    next := va + page;
    let w, _ = given (Function.alloc_dma ~va f page) in
    Function.free_dma f w
  in
  let rec from n =
    let raised = Rig_support.interrupted n dma in
    dma ();
    if raised then from (n + 1)
  in
  from 1;
  releases root f

(* A free_dma or an unpin gives memory back while no give-back of the
   function runs, since each is a use of it. The call runs with a landing at
   its [n]-th allocation, for each [n]; there a release on another domain
   either returns, leaving the give-back to the call, or gives the function
   back, held at the give-back's first allocation, its bus mastering still
   on. While the give-back is held, the call gives back no memory. *)
let blocks_given = ref 0

let test_under_give_back (_, call) =
  let refused = ref false in
  let rec walk n =
    let at what = strf "%s, landing at %d" what n in
    let root, f = fixture () in
    Function.set_bus_master f true;
    (* A block of its own for each landing: memory kept under a give-back
       stays the process's until it exits. *)
    incr blocks_given;
    let va = free_base + (1 lsl 38) + (!blocks_given * huge) in
    granted (Machine.reserve (Function.machine f) ~base:va huge);
    ignore (frames root va (64 * kib));
    let w, _ = given (Function.alloc_dma ~va f (64 * kib)) in
    let free = call f w va in
    let file = memory_file root in
    let go = Atomic.make false and inside = Atomic.make false in
    let start = Atomic.make false and returned = Atomic.make false in
    let wait_go () =
      Atomic.set inside true;
      while not (Atomic.get go) do
        Domain.cpu_relax ()
      done
    in
    let releasing =
      Domain.spawn (fun () ->
          while not (Atomic.get start) do
            Domain.cpu_relax ()
          done;
          ignore
            (Rig_support.interrupted
               ~from:(fun _ -> Function.released f)
               ~landing:(Call wait_go) 1
               (fun () -> Function.release f)
              : bool);
          Atomic.set returned true)
    in
    let landing () =
      Atomic.set start true;
      ignore
        (poll (fun () -> Atomic.get inside || Atomic.get returned) : bool)
    in
    let reached =
      Fun.protect ~finally:(fun () ->
          Atomic.set start true;
          Atomic.set go true)
      @@ fun () ->
      let reached = Rig_support.interrupted ~landing:(Call landing) n free in
      if Atomic.get inside && not (Atomic.get returned) then begin
        refused := true;
        equal ~msg:(at "the memory kept during the give-back") int huge
          (Tree.stored file)
      end;
      reached
    in
    Domain.join releasing;
    equal ~msg:(at "taken once both returned") bool false
      (Tree.flocked (config_file root (Function.bus f)));
    if n = 1 then equal ~msg:"the walk lands" bool true reached;
    if reached then walk (n + 1)
  in
  walk 1;
  equal ~msg:"a landing met a give-back" bool true !refused

let under_give_back =
  [
    ("free_dma", fun f w _ () -> Function.free_dma f w);
    ( "unpin",
      fun f w va ->
        let n = Window.length w in
        ignore (pin f va n : (int * int) list);
        Function.free_dma f w;
        fun () -> Function.unpin f va n );
  ]

(* Whether [a] allocates the [Invalid_argument] of a refusal: its innermost
   frame is [Stdlib.invalid_arg]. *)
let refusal (a : Gc.Memprof.allocation) =
  match Printexc.backtrace_slots a.callstack with
  | Some slots when Array.length slots > 0 ->
      Printexc.Slot.name slots.(0) = Some "Stdlib.invalid_arg"
  | Some _ | None -> false

(* A use that an exception ends while another domain holds the function's
   lock, with SIGINT pending, ends: the release gives the function back. The
   other domain refuses to unpin a range never pinned, and parks as it
   allocates the refusal, under the lock. The use, a pin of no bytes,
   allocates its own refusal with SIGINT then pending, and raises it with no
   poll point before its end, which meets the held lock and the signal at
   once. *)
let test_interrupted_held () =
  let root, f = fixture () in
  let page = Machine.page Machine.this in
  let go = Atomic.make false in
  let holder =
    Domain.spawn (fun () ->
        while not (Atomic.get go) do
          Domain.cpu_relax ()
        done;
        let unpin () =
          try Function.unpin f page page with Invalid_argument _ -> ()
        in
        Rig_support.(interrupted ~from:refusal ~landing:(Call park) 1 unpin))
  in
  let hold () =
    Atomic.set go true;
    while not (Rig_support.parked ()) do
      Domain.cpu_relax ()
    done;
    Rig_support.interrupt ()
  in
  let pin () =
    try ignore (Function.pin f page 0) with Invalid_argument _ -> ()
  in
  Sys.catch_break true;
  let reached =
    Fun.protect ~finally:(fun () -> Sys.catch_break false) @@ fun () ->
    Rig_support.(interrupted ~from:refusal ~landing:(Call hold) 1 pin)
  in
  Rig_support.unpark ();
  equal ~msg:"the unpin's refusal" bool true (Domain.join holder);
  equal ~msg:"the pin's refusal" bool true reached;
  releases root f

(* The calling thread's signals blocked, as rig blocks them across a call
   below its core ([rig_mask] in [rig_edge.h]). *)
external mask : unit -> unit = "caml_rig_edge_mask" [@@noalloc]
external unmask : unit -> unit = "caml_rig_edge_unmask"

(* A finish of the releases with the thread's signals blocked takes a
   SIGINT while it waits for another domain's use of a released function:
   it raises [Sys.Break] while the use runs, and leaves the function taken
   for the next finish, which gives it back once the use has ended. *)
let test_finish_interrupted () =
  if Sys.win32 then skip ~reason:"Windows has no SIGINT from a thread" ();
  let root, f = fixture () in
  let taken () = Tree.flocked (config_file root (Function.bus f)) in
  let user =
    Domain.spawn (fun () ->
        if Function.use f then begin
          Rig_support.park ();
          Function.unuse f
        end)
  in
  while not (Rig_support.parked ()) do
    Domain.cpu_relax ()
  done;
  Function.release f;
  equal ~msg:"taken while the use runs" bool true (taken ());
  Sys.catch_break true;
  Rig_support.interrupt_after 0.05;
  mask ();
  let finished =
    Fun.protect ~finally:(fun () ->
        (* A SIGINT the finish did not take lands here. *)
        (try
           unmask ();
           Rig_support.interrupt_join ()
         with Sys.Break -> ());
        Sys.catch_break false)
    @@ fun () ->
    match Function.finish_releases () with
    | () -> "returned"
    | exception Sys.Break ->
        if Rig_support.parked () then "raised while the use ran"
        else "raised once the use ended"
  in
  equal ~msg:"the finish" string "raised while the use ran" finished;
  equal ~msg:"taken after the finish raised" bool true (taken ());
  Rig_support.unpark ();
  Domain.join user;
  Function.finish_releases ();
  equal ~msg:"taken after the next finish" bool false (taken ())

(* The test's executable, run with [interrupting], is [interrupted_release]'s
   process. *)
let interrupting = "--interrupted-release"

(* The files [interrupted_release] opens once its release raised, more than a
   take holds descriptors, so that they take the numbers of those the release
   closed; and what each holds. *)
let canaries = 8
let canary = String.make 16 '\xff'
let canary_file root i = Filename.concat root (strf "canary-%d" i)

(* Takes the fixture's function at [bus] with its bus mastering on, maps three
   of its BARs, allocates and frees memory at [va] and allocates more 2 MiB
   above, whose frames the tree's page map gives, and releases it with an
   exception at the [n]-th allocation of the release. It then opens the
   canaries, frees the memory left, releases the function again, finishes the
   releases, takes it again and exits: with 0 if the [n]-th allocation was
   reached, 4 if it was not, 5 if the finish closed a canary, 6 if the free
   gave back memory while the function still mastered the bus, 7 if the
   function could be taken again meanwhile, and 8 if it could not be taken
   again after the finish. *)
let interrupted_release root bus va n =
  let ok = function Ok x -> x | Error _ -> exit refused_code in
  let f = take_mastering (Machine.at root) bus in
  ok (Machine.reserve (Function.machine f) ~base:free_base (8 * mib));
  List.iter
    (fun i -> ignore (ok (Function.map ~length:4096 f i) : Window.t))
    [ 0; 2; 5 ];
  let alloc va =
    match ok (Function.alloc_dma ~va f mib) with
    | Some (w, _) -> w
    | None -> exit refused_code
  in
  Function.free_dma f (alloc va);
  let kept = alloc (va + (2 * mib)) in
  let reached = Rig_support.interrupted n (fun () -> Function.release f) in
  let open_canary i =
    let fd =
      Unix.openfile (canary_file root i) [ O_RDWR; O_CREAT; O_CLOEXEC ] 0o600
    in
    ignore (Unix.write_substring fd canary 0 (String.length canary) : int);
    fd
  in
  let fds = List.init canaries open_canary in
  let mastering = command_in root bus land bus_master <> 0 in
  if mastering && Result.is_ok (Function.take (Machine.at root) bus) then
    exit 7;
  Function.free_dma f kept;
  let stored f = Tree.stored (Filename.concat root ("dev/hugepages/" ^ f)) in
  let still = command_in root bus land bus_master <> 0 in
  if still && List.for_all (fun f -> stored f = 0) (memory_files root) then
    exit 6;
  Function.release f;
  Function.finish_releases ();
  let is_open fd =
    match Unix.fstat fd with _ -> true | exception Unix.Unix_error _ -> false
  in
  if not (List.for_all is_open fds) then exit 5;
  (match Function.take (Machine.at root) bus with
  | Ok f -> Function.release f
  | Error _ -> exit 8);
  exit (if reached then 0 else 4)

(* An exception at any allocation of a release, with windows mapped and
   system memory taken, leaves what the release had not given back to the
   finish of the releases, which gives it back and nothing twice: the files
   the process opened since stay open and unwritten, through its exit too,
   and the function is taken again. While the function may still master the
   bus, no take gets it and memory freed stays. Once the process exits, a
   reset of the function leaves no memory file or marker. *)
let test_interrupted_release () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let rec from n =
    let fn = Tree.gpu "0000:03:00.0" in
    let root = Tree.make [ fn ] in
    let va = free_base + (2 * mib) in
    ignore (frames root va (3 * mib));
    let exe = Sys.executable_name in
    let pid =
      Unix.create_process exe
        [| exe; interrupting; root; fn.bus; string_of_int va; string_of_int n |]
        Unix.stdin Unix.stdout Unix.stderr
    in
    let at what = strf "%s, interrupted at %d" what n in
    let reached =
      match wait_exit pid with
      | WEXITED 0 -> true
      | WEXITED 4 -> false
      | WEXITED 5 -> failf "%s" (at "the finish closed a file")
      | WEXITED 6 ->
          failf "%s" (at "a free gave back memory the function reached")
      | WEXITED 7 ->
          failf "%s" (at "a take while the function mastered the bus")
      | WEXITED 8 -> failf "%s" (at "the finish left the function held")
      | _ -> failf "%s" (at "the process did not exit")
    in
    for i = 0 to canaries - 1 do
      equal ~msg:(at "a file opened after the release") string canary
        (In_channel.with_open_bin (canary_file root i) In_channel.input_all)
    done;
    require_ok ~msg:(at "the reset after the exit")
      (Driver.reset (fixture_gpus ()) (Machine.at root) 0);
    equal ~msg:(at "the memory files and markers left") (list string) []
      (hugepages root);
    if reached then from (n + 1)
  in
  from 1

(* Two releases an exception cut, then a finish of the releases with an
   exception once at its [n]-th allocation, for [n] from 1 until the finish
   makes fewer: the finish goes on to the other function and the exception
   passes, so at most the function it cut is left. The next finish gives that
   one back. *)
let test_finish_raises () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ();
  let buses = [ "0000:03:00.0"; "0000:04:00.0" ] in
  let root = Tree.make (List.map Tree.gpu buses) in
  let m = Machine.at root in
  let taken () =
    List.length
      (List.filter (fun b -> Tree.flocked (config_file root b)) buses)
  in
  let rec walk n =
    let at what = strf "%s, from allocation %d" what n in
    List.iter
      (fun b ->
        let f = take_mastering m b in
        ignore (map ~length:4096 f 0 : Window.t);
        ignore (Rig_support.interrupted 1 (fun () -> Function.release f)))
      buses;
    equal ~msg:(at "the functions the cut releases left") int 2 (taken ());
    let raised = ref false in
    let once () =
      if not !raised then begin
        raised := true;
        raise Exit
      end
    in
    let reached =
      Rig_support.interrupted ~landing:(Call once) n Function.finish_releases
    in
    if n = 1 then equal ~msg:"the walk lands" bool true reached;
    less ~msg:(at "the functions the raising finish left") int ~than:2
      (taken ());
    Function.finish_releases ();
    equal ~msg:(at "the functions the next finish left") int 0 (taken ());
    if reached then walk (n + 1)
  in
  walk 1

let interrupts =
  group ~timeout:patience "interrupts"
    [
      test
        "a finish of the releases that an exception cuts finishes the others \
         and passes it"
        test_finish_raises;
      test
        "an exception at any allocation of a release leaves the rest to the \
         finish of the releases, giving nothing back twice"
        test_interrupted_release;
      test "an exception at any allocation of a wait ends its use"
        test_interrupted_wait;
      test
        "an exception at any allocation of a DMA allocation leaves the lock \
         free and ends its use"
        test_interrupted_dma;
      test
        "a use an exception ends while another domain holds the lock, with \
         SIGINT pending, ends"
        test_interrupted_held;
      test
        "a finish of the releases takes a signal while it waits for a use, \
         with signals blocked"
        test_finish_interrupted;
      cases "a free_dma or an unpin gives no memory back under a give-back"
        ~name:fst under_give_back test_under_give_back;
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
      test "a GPU behind an IOMMU out of mappings or locked memory gets none"
        test_vfio_limit;
      test "GPUs behind one IOMMU map the pages pinned for both once"
        test_vfio_shared;
      test "a child of fork maps its GPU's memory apart from its parent's"
        test_vfio_fork;
      test "a GPU behind an IOMMU gets device addresses below its DMA bound"
        test_vfio_bits;
      test "a GPU behind an IOMMU masters the bus as set_bus_master sets it"
        test_vfio_bus_master;
    ]

let () =
  match Sys.argv with
  | [| _; arg; root; bus |] when arg = exiting -> exit_mastering root bus
  | [| _; arg; how; root; bus; va |] when arg = holding ->
      hold_memory how root bus (int_of_string va)
  | [| _; arg; root; bus; va; n |] when arg = interrupting ->
      interrupted_release root bus (int_of_string va) (int_of_string n)
  | _ ->
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
             interrupts;
             this_machine;
           ]
