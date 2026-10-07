(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external page_size : unit -> int = "caml_device_pci_page_size"
external reserve_at : int -> int -> unit = "caml_device_pci_reserve"

external map_at : int -> int -> bool -> bool -> int
  = "caml_device_pci_sysmem_map"

external release_at : int -> int -> unit = "caml_device_pci_sysmem_release"
external unmap_at : int -> int -> unit = "caml_device_pci_sysmem_unmap"
external lock_at : int -> int -> unit = "caml_device_pci_sysmem_lock"
external unlock_at : int -> int -> unit = "caml_device_pci_sysmem_unlock"
external pagemap : int -> int -> string = "caml_device_pci_pagemap"

let page = page_size ()
let huge = 2 lsl 20
let lock = Mutex.create ()
let round_page n = (n + page - 1) / page * page

(* The ranges [reserve] reserved, and the memory mapped where the system chose,
   which no reservation takes back. *)
let reserved : (int * int, unit) Hashtbl.t = Hashtbl.create 4
let placed : (int, unit) Hashtbl.t = Hashtbl.create 16

(* Pins, counted per page across the process: munlock is not counted. *)
let pins : (int, int) Hashtbl.t = Hashtbl.create 64

let reserve ~base n =
  Mutex.protect lock @@ fun () ->
  if not (Hashtbl.mem reserved (base, n)) then begin
    reserve_at base n;
    Hashtbl.add reserved (base, n) ()
  end

let on_page fn a =
  if a mod page <> 0 then
    invalid_arg (Printf.sprintf "Sysmem.%s: 0x%x is not on a page" fn a)

let pages_of a n = List.init ((n + page - 1) / page) (fun i -> a + (i * page))

(* Physical addresses *)

(* Locked pages stay at the physical address the process read for them only
   while the kernel does not compact them. *)
let setting = "/proc/sys/vm/compact_unevictable_allowed"
let checked = Atomic.make false

let check_setting () =
  if not (Atomic.get checked) then begin
    let value =
      try In_channel.with_open_text setting In_channel.input_all |> String.trim
      with Sys_error _ -> "0"
    in
    if value <> "0" then
      failwith
        (Printf.sprintf
           "the kernel may move locked pages; run: sudo sysctl -w \
            vm.compact_unevictable_allowed=0 (%s)"
           setting);
    Atomic.set checked true
  end

(* Bits 0-54 of a page-map entry are the page frame, which reads as 0 without
   the privilege. *)
let frame_mask = 0x7F_FFFF_FFFF_FFFFL

let physical a n =
  check_setting ();
  let count = (n + page - 1) / page in
  let map = pagemap a count in
  List.init count (fun i ->
      let frame =
        Int64.to_int (Int64.logand (String.get_int64_le map (8 * i)) frame_mask)
      in
      if frame = 0 then
        failwith "reading physical addresses needs CAP_SYS_ADMIN (run as root)";
      frame * page)

(* Pins *)

let add_pins a n =
  List.iter
    (fun p ->
      Hashtbl.replace pins p
        (1 + Option.value ~default:0 (Hashtbl.find_opt pins p)))
    (pages_of a n)

(* Drops one pin of each page of [a, a + n), unlocking the pages whose last pin
   goes if [unlock]. [lock] is held. *)
let drop_pins ~unlock a n =
  let pages = pages_of a n in
  if List.exists (fun p -> not (Hashtbl.mem pins p)) pages then
    invalid_arg (Printf.sprintf "Sysmem.unpin: 0x%x is not pinned" a);
  List.iter
    (fun p ->
      match Hashtbl.find pins p with
      | 1 ->
          Hashtbl.remove pins p;
          if unlock then unlock_at p page
      | k -> Hashtbl.replace pins p (k - 1))
    pages

let pin a n =
  on_page "pin" a;
  Mutex.protect lock (fun () ->
      lock_at a n;
      add_pins a n);
  match physical a n with
  | addresses -> addresses
  | exception e ->
      Mutex.protect lock (fun () -> drop_pins ~unlock:true a n);
      raise e

let unpin a n = Mutex.protect lock (fun () -> drop_pins ~unlock:true a n)

(* Memory *)

(* Maps [n] bytes at [va], or where the system chooses. *)
let map_bytes ?va n ~huge ~locked =
  let a = map_at (Option.value va ~default:0) n huge locked in
  if va = None then Mutex.protect lock (fun () -> Hashtbl.replace placed a ());
  a

(* Returns [n] bytes at [a] to their reservation, or to the system. *)
let unmap a n =
  let was_placed =
    Mutex.protect lock (fun () ->
        let p = Hashtbl.mem placed a in
        Hashtbl.remove placed a;
        p)
  in
  if was_placed then unmap_at a n else release_at a n

let positive fn n =
  if n <= 0 then invalid_arg (Printf.sprintf "Sysmem.%s: %d bytes" fn n)

let map ?va n =
  positive "map" n;
  Option.iter (on_page "map") va;
  let n = round_page n in
  Window.v (map_bytes ?va n ~huge:false ~locked:false) n

let alloc ?(contiguous = false) ?va n =
  positive "alloc" n;
  Option.iter (on_page "alloc") va;
  if contiguous && n > huge then
    invalid_arg "Sysmem.alloc: contiguous memory is at most 2 MiB";
  let huge_page = contiguous && n > page in
  (match va with
  | Some va when huge_page && va mod huge <> 0 ->
      invalid_arg (Printf.sprintf "Sysmem.alloc: 0x%x is not on 2 MiB" va)
  | _ -> ());
  let n = if huge_page then huge else round_page n in
  let a =
    try map_bytes ?va n ~huge:huge_page ~locked:true
    with Failure why when huge_page ->
      failwith
        (why
       ^ "; contiguous memory needs a free huge page: sudo sysctl -w \
          vm.nr_hugepages=16")
  in
  let pages =
    try physical a n
    with e ->
      unmap a n;
      raise e
  in
  let first = List.hd pages in
  let scattered = List.filteri (fun i p -> p <> first + (i * page)) pages in
  if contiguous && scattered <> [] then begin
    unmap a n;
    failwith "the system gave contiguous memory in scattered pages"
  end;
  Mutex.protect lock (fun () -> add_pins a n);
  (Window.v a n, if contiguous then [ first ] else pages)

let free w =
  let a = Window.address w and n = Window.length w in
  Mutex.protect lock (fun () ->
      List.iter
        (fun p ->
          match Hashtbl.find_opt pins p with
          | Some 1 | None -> Hashtbl.remove pins p
          | Some k -> Hashtbl.replace pins p (k - 1))
        (pages_of a n));
  unmap a n
