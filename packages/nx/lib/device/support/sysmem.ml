(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external page_size : unit -> int = "caml_nx_support_page_size"
external reserve_at : nativeint -> int -> unit = "caml_nx_sysmem_reserve"
external map_at : nativeint -> int -> bool -> nativeint = "caml_nx_sysmem_alloc"
external release_at : nativeint -> int -> unit = "caml_nx_sysmem_release"
external unmap_at : nativeint -> int -> unit = "caml_nx_sysmem_unmap"
external lock_at : nativeint -> int -> unit = "caml_nx_sysmem_lock"
external unlock_at : nativeint -> int -> unit = "caml_nx_sysmem_unlock"
external pagemap : nativeint -> int -> string = "caml_nx_sysmem_pagemap"

let page = page_size ()
let lock = Mutex.create ()
let reserved = Hashtbl.create 4

let reserve ~base n =
  Mutex.protect lock (fun () ->
      if not (Hashtbl.mem reserved (base, n)) then begin
        reserve_at (Nativeint.of_int base) n;
        Hashtbl.add reserved (base, n) ()
      end)

let unreserve ~base n =
  Mutex.protect lock (fun () ->
      if Hashtbl.mem reserved (base, n) then begin
        unmap_at (Nativeint.of_int base) n;
        Hashtbl.remove reserved (base, n)
      end)

(* Locked pages must stay at the physical address the process read for them,
   which the kernel guarantees only when it does not compact them. *)
let setting = "/proc/sys/vm/compact_unevictable_allowed"
let checked = ref false

let check_setting () =
  if not !checked then begin
    let read () =
      try In_channel.with_open_text setting In_channel.input_all |> String.trim
      with Sys_error _ -> "0"
    in
    (if read () <> "0" then
       try Out_channel.with_open_text setting (fun oc -> output_string oc "0")
       with Sys_error _ -> ());
    if read () <> "0" then
      failwith
        (Printf.sprintf
           "the kernel may move locked pages; run: sudo sysctl -w \
            vm.compact_unevictable_allowed=0 (%s)"
           setting);
    checked := true
  end

(* Physical addresses of the [n] bytes at [va], from the page map: bits 0-54 of
   an entry are the page frame, which reads as 0 without the privilege. *)
let physical va n =
  check_setting ();
  let pages = (n + page - 1) / page in
  let map = pagemap va pages in
  List.init pages (fun i ->
      let e = String.get_int64_le map (8 * i) in
      let frame = Int64.to_int (Int64.logand e 0x7F_FFFF_FFFF_FFFFL) in
      if frame = 0 then
        failwith "reading physical addresses needs CAP_SYS_ADMIN (run as root)";
      frame * page)

(* Pins, counted per page across the process: munlock is not counted. The pages
   {!alloc} maps locked hold one pin until {!free}, so that no {!unpin} unlocks
   them. *)
let pins : (nativeint, int) Hashtbl.t = Hashtbl.create 64

let pages_of a n =
  List.init
    ((n + page - 1) / page)
    (fun i -> Nativeint.add a (Nativeint.of_int (i * page)))

let add_pins a n =
  List.iter
    (fun p ->
      Hashtbl.replace pins p
        (1 + Option.value ~default:0 (Hashtbl.find_opt pins p)))
    (pages_of a n)

let huge = 2 lsl 20

let extent ?(contiguous = false) n =
  if contiguous && n > page then huge else (n + page - 1) / page * page

(* Memory mapped where the system chose, which no reservation holds. *)
let placed : (nativeint, unit) Hashtbl.t = Hashtbl.create 16

(* Returns [n] bytes at [a] to their reservation, or to the system. *)
let unmap a n =
  if Mutex.protect lock (fun () -> Hashtbl.mem placed a) then begin
    Mutex.protect lock (fun () -> Hashtbl.remove placed a);
    unmap_at a n
  end
  else release_at a n

let alloc ?(contiguous = false) ?va n =
  Option.iter
    (fun va ->
      if va mod page <> 0 then
        invalid_arg (Printf.sprintf "Sysmem.alloc: 0x%x is not on a page" va))
    va;
  if contiguous && n > huge then
    invalid_arg "Sysmem.alloc: contiguous memory is at most 2 MiB";
  let huge_page = contiguous && n > page in
  Option.iter
    (fun va ->
      if huge_page && va mod huge <> 0 then
        invalid_arg (Printf.sprintf "Sysmem.alloc: 0x%x is not on 2 MiB" va))
    va;
  let n = extent ~contiguous n in
  let a =
    try map_at (Nativeint.of_int (Option.value ~default:0 va)) n huge_page
    with Failure why when huge_page ->
      failwith
        (why
       ^ "; contiguous memory needs a free huge page: sudo sysctl -w \
          vm.nr_hugepages=16")
  in
  if va = None then Mutex.protect lock (fun () -> Hashtbl.replace placed a ());
  let m = Mmio.v a n in
  match physical a n with
  | exception e ->
      unmap a n;
      raise e
  | pages ->
      let first = List.hd pages in
      if
        contiguous
        && List.filteri (fun i p -> p <> first + (i * page)) pages <> []
      then begin
        unmap a n;
        failwith "the system gave contiguous memory in scattered pages"
      end;
      Mutex.protect lock (fun () -> add_pins a n);
      (m, if contiguous then [ first ] else pages)

let free m =
  let a = Mmio.address m and n = Mmio.length m in
  Mutex.protect lock (fun () ->
      List.iter
        (fun p ->
          match Hashtbl.find_opt pins p with
          | Some 1 | None -> Hashtbl.remove pins p
          | Some k -> Hashtbl.replace pins p (k - 1))
        (pages_of a n));
  unmap a n

let unpin_locked a n =
  List.iter
    (fun p ->
      match Hashtbl.find_opt pins p with
      | Some 1 ->
          Hashtbl.remove pins p;
          unlock_at p page
      | Some k -> Hashtbl.replace pins p (k - 1)
      | None -> ())
    (pages_of a n)

let unpin a n = Mutex.protect lock (fun () -> unpin_locked a n)

let pin a n =
  if Nativeint.rem a (Nativeint.of_int page) <> 0n then
    invalid_arg (Printf.sprintf "Sysmem.pin: 0x%nx is not on a page" a);
  Mutex.protect lock (fun () ->
      lock_at a n;
      add_pins a n);
  match physical a n with
  | addrs -> addrs
  | exception e ->
      unpin a n;
      raise e
