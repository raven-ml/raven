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

(* [f ()], whose system call failing is reported as [what]. *)
let step what f =
  try f ()
  with Unix.Unix_error (e, _, _) ->
    failwith (Printf.sprintf "%s: %s" what (Unix.error_message e))

let page = page_size ()
let round_page n = (n + page - 1) / page * page

(* The huge page of x86-64 and arm64 with 4 KiB pages, which contiguous memory
   larger than a page is. *)
let huge = 2 lsl 20

(* The ranges [reserve] reserved, and pins, counted per page across the process:
   munlock is not counted. [lock] guards both. *)
let lock = Mutex.create ()
let reserved : (int * int, unit) Hashtbl.t = Hashtbl.create 4
let pins : (int, int) Hashtbl.t = Hashtbl.create 64
let range a n = Printf.sprintf "[0x%x, 0x%x)" a (a + n)

let reserve ~base n =
  Mutex.protect lock @@ fun () ->
  if not (Hashtbl.mem reserved (base, n)) then begin
    (match reserve_at base n with
    | () -> ()
    | exception Unix.Unix_error (EEXIST, _, _) ->
        failwith (Printf.sprintf "addresses %s are in use" (range base n))
    | exception Unix.Unix_error (e, _, _) ->
        failwith
          (Printf.sprintf "reserving addresses %s: %s" (range base n)
             (Unix.error_message e)));
    Hashtbl.add reserved (base, n) ()
  end

(* Whether the [n] bytes at [a] lie in a range [reserve] reserved. [lock] is
   held. *)
let reserved_at a n =
  Hashtbl.fold
    (fun (base, len) () r -> r || (a >= base && a + n <= base + len))
    reserved false

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
           "the kernel may move locked pages (%s is %s); forbid it: sudo \
            sysctl -w vm.compact_unevictable_allowed=0"
           setting value);
    Atomic.set checked true
  end

(* The page-map entries of the [pages] pages from [a], 8 bytes each. The kernel
   walks the page tables for what is read: a channel, which reads 64 KiB ahead,
   would make it walk 32 MiB of them to pin one page. [Unix.read] reads at most
   64 KiB a call and releases the runtime for each. *)
let pagemap_file = "/proc/self/pagemap"

let pagemap a pages =
  let n = 8 * pages and b = Bytes.create (8 * pages) in
  let read fd =
    ignore (Unix.lseek fd (a / page * 8) SEEK_SET);
    let rec go got =
      if got < n then
        match Unix.read fd b got (n - got) with
        | 0 -> raise (Unix.Unix_error (EIO, "read", pagemap_file))
        | k -> go (got + k)
    in
    go 0
  in
  match Unix.openfile pagemap_file [ O_RDONLY; O_CLOEXEC ] 0 with
  | exception Unix.Unix_error (e, _, _) ->
      failwith
        (Printf.sprintf "opening %s: %s" pagemap_file (Unix.error_message e))
  | fd -> (
      Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
      match read fd with
      | () -> Bytes.unsafe_to_string b
      | exception Unix.Unix_error (e, _, _) ->
          failwith
            (Printf.sprintf "reading %s: %s" pagemap_file (Unix.error_message e))
      )

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
        failwith "reading physical addresses needs CAP_SYS_ADMIN; run as root";
      frame * page)

(* Pins *)

let add_pins a n =
  List.iter
    (fun p ->
      Hashtbl.replace pins p
        (1 + Option.value ~default:0 (Hashtbl.find_opt pins p)))
    (pages_of a n)

(* Drops one pin of each page of [a, a + n), unlocking the pages whose last pin
   goes. [lock] is held. *)
let drop_pins a n =
  List.iter
    (fun p ->
      match Hashtbl.find pins p with
      | 1 ->
          Hashtbl.remove pins p;
          step "unlocking memory" (fun () -> unlock_at p page)
      | k -> Hashtbl.replace pins p (k - 1))
    (pages_of a n)

let pin a n =
  Mutex.protect lock (fun () ->
      (match lock_at a n with
      | () -> ()
      | exception Unix.Unix_error (((ENOMEM | EPERM) as e), _, _) ->
          failwith
            (Printf.sprintf
               "locking %d bytes for a GPU: %s; raise the locked-memory limit \
                (ulimit -l)"
               n (Unix.error_message e))
      | exception Unix.Unix_error (e, _, _) ->
          failwith
            (Printf.sprintf "locking %d bytes for a GPU: %s" n
               (Unix.error_message e)));
      add_pins a n);
  match physical a n with
  | addresses -> addresses
  | exception e ->
      Mutex.protect lock (fun () -> drop_pins a n);
      raise e

let unpin a n = Mutex.protect lock (fun () -> drop_pins a n)

(* Memory *)

(* Maps [n] bytes at [va], which must lie in a reservation: mapping over
   anything else would replace the process's own memory. Without [va], where the
   system chooses. A huge page the system lacks is ENOMEM, and locked memory
   past the locked-memory limit EAGAIN (mmap(2)). *)
let map_bytes ?va n ~huge ~locked =
  Option.iter
    (fun va ->
      if not (Mutex.protect lock (fun () -> reserved_at va n)) then
        invalid_arg
          (Printf.sprintf "Function.alloc_dma: 0x%x is in no reserved range" va))
    va;
  match map_at (Option.value va ~default:0) n huge locked with
  | a -> a
  | exception Unix.Unix_error (e, _, _) ->
      let why =
        Printf.sprintf "allocating %d bytes of system memory: %s" n
          (Unix.error_message e)
      in
      failwith
        (match e with
        | ENOMEM when huge ->
            why
            ^ "; reserve huge pages for contiguous memory: sudo sysctl -w \
               vm.nr_hugepages=16"
        | EAGAIN when locked ->
            why ^ "; raise the locked-memory limit (ulimit -l)"
        | _ -> why)

(* Returns [n] bytes at [a] to their reservation, or to the system. *)
let unmap a n =
  step "freeing system memory" (fun () ->
      if Mutex.protect lock (fun () -> reserved_at a n) then release_at a n
      else unmap_at a n)

let map ?va n =
  let n = round_page n in
  Window.v (map_bytes ?va n ~huge:false ~locked:false) n

let alloc ?(contiguous = false) ?va n =
  let huge_page = contiguous && n > page in
  let n = if huge_page then huge else round_page n in
  let a = map_bytes ?va n ~huge:huge_page ~locked:true in
  let pages =
    try physical a n
    with e ->
      unmap a n;
      raise e
  in
  Mutex.protect lock (fun () -> add_pins a n);
  (Window.v a n, pages)

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
