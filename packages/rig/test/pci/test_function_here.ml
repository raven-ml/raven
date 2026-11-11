(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Functions of this machine's GPUs, taken here, and memory in this machine's
   huge pages. test_function holds the functions of machines it makes of
   files. *)

open Windtrap
open Rig_pci
open Rig_pci_support

let alloc_dma f n = require_some (require_ok (Function.alloc_dma f n))

let pin f a n = require_some (require_ok (Function.pin f a n))

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

(* The command register and its bus master bit (PCI Express Base
   Specification, 7.5.1.1.3): the function masters the bus, reaching system
   memory by DMA. *)
let command = 0x04
let bus_master = 0x4

(* Behind an IOMMU a write reaches the function: its bus master bit reads back
   as set_bus_master sets it. *)
let test_vfio_bus_master () =
  let _, f = vfio_function () in
  Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
  Function.set_bus_master f true;
  equal ~msg:"set" hex bus_master (Function.config16 f command land bus_master);
  Function.set_bus_master f false;
  equal ~msg:"cleared" hex 0 (Function.config16 f command land bus_master)

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
    let dir =
      Printf.sprintf "/dev/hugepages/rig-pci-test-%d" (Unix.getpid ())
    in
    match Unix.mkdir dir 0o700 with
    | () -> Some dir
    | exception Unix.Unix_error _ -> None

(* As test_function's death of a process, with the machine's memory files in
   one of this machine's huge pages. *)
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
      test
        "memory a killed process left on a hugetlbfs stays through a release \
         and goes at its GPU's reset (SIGKILL in a child)"
        test_death_hugetlbfs;
    ]

let () =
  match Sys.argv with
  | [| _; arg; how; root; bus; va |] when arg = holding ->
      hold_memory how root bus (int_of_string va)
  | _ -> exit @@ run "rig_pci.function here" [ this_machine ]
