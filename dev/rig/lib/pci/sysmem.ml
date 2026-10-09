(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

external page_size : unit -> int = "caml_rig_pci_page_size"
external reserve_at : int -> int -> unit = "caml_rig_pci_reserve"
external map_at : int -> int -> int = "caml_rig_pci_sysmem_map"
external release_at : int -> int -> unit = "caml_rig_pci_sysmem_release"
external unmap_at : int -> int -> unit = "caml_rig_pci_sysmem_unmap"

external map_file : Unix.file_descr -> int -> int -> int -> int
  = "caml_rig_pci_sysmem_map_file"

external punch : Unix.file_descr -> int -> int -> unit
  = "caml_rig_pci_sysmem_punch"

external allocate : Unix.file_descr -> int -> int -> unit
  = "caml_rig_pci_sysmem_allocate"

external region : int -> int = "caml_rig_pci_sysmem_region"

external create : string -> string -> Unix.file_descr
  = "caml_rig_pci_sysmem_create"

external own_lock : Unix.file_descr -> unit = "caml_rig_pci_flock"
external zero : int -> int -> unit = "caml_rig_pci_sysmem_zero"

let page = page_size ()
let round_page n = (n + page - 1) / page * page

(* The huge page memory reached physically lies in: one entry of the
   second-level page table of x86-64 and of arm64 with 4 KiB pages maps 512
   pages of 4 KiB. It comes from the machine's hugetlbfs, whose pages must be of
   this size: arm64 with 16 or 64 KiB pages has larger default huge pages, which
   a mount's pagesize=2M option changes. *)
let huge = 2 lsl 20

(* The ranges [reserve] reserved, which [lock] guards. *)
let lock = Mutex.create ()
let reserved : unit Tables.Range.t = Tables.Range.create 4
let range a n = strf "[0x%x, 0x%x)" a (a + n)

let reserve ~base n =
  Mutex.protect lock @@ fun () ->
  if not (Tables.Range.mem reserved (base, n)) then begin
    (match reserve_at base n with
    | () -> ()
    | exception Unix.Unix_error (EEXIST, _, _) ->
        Fail.fail "addresses %s are in use" (range base n)
    | exception Unix.Unix_error (ENOSYS, _, _) ->
        Fail.fail "reserving addresses %s needs Linux" (range base n)
    | exception Unix.Unix_error (e, _, _) ->
        Fail.fail "reserving addresses %s: %s" (range base n)
          (Unix.error_message e));
    Tables.Range.add reserved (base, n) ()
  end

(* Whether the [n] bytes at [a] lie in a range [reserve] reserved. [lock] is
   held. *)
let reserved_at a n =
  Tables.Range.fold
    (fun (base, len) () r -> r || (a >= base && a + n <= base + len))
    reserved false

(* Physical addresses *)

(* The page-map entry of the page at [a], 8 bytes. The kernel walks the page
   tables for what is read: a channel, which reads 64 KiB ahead, would make it
   walk 32 MiB of them to pin one page. *)
let pagemap root a =
  let pagemap_file = Filename.concat root "proc/self/pagemap" in
  let b = Bytes.create 8 in
  let read fd =
    ignore (Unix.lseek fd (a / page * 8) SEEK_SET);
    let rec go got =
      if got < 8 then
        match Unix.read fd b got (8 - got) with
        | 0 -> Fail.fail "reading %s: got %d of 8 bytes" pagemap_file got
        | k -> go (got + k)
    in
    go 0
  in
  match Unix.openfile pagemap_file [ O_RDONLY; O_CLOEXEC ] 0 with
  | exception Unix.Unix_error (e, _, _) ->
      Fail.fail "opening %s: %s" pagemap_file (Unix.error_message e)
  | fd -> (
      Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
      match read fd with
      | () -> Bytes.get_int64_le b 0
      | exception Unix.Unix_error (e, _, _) ->
          Fail.fail "reading %s: %s" pagemap_file (Unix.error_message e))

(* Bit 63 of a page-map entry says the page is present; bits 0-54 are its frame,
   which reads as 0 without the privilege (proc(5)). *)
let present = Int64.shift_left 1L 63
let frame_mask = 0x7F_FFFF_FFFF_FFFFL

(* The physical address of the page at [a]. *)
let physical root a =
  let entry = pagemap root a in
  if Int64.logand entry present = 0L then
    Fail.fail "the page at 0x%x is not in memory" a;
  let frame = Int64.to_int (Int64.logand entry frame_mask) in
  if frame = 0 then
    Fail.fail "reading physical addresses needs CAP_SYS_ADMIN; run as root";
  frame * page

(* Memory *)

(* Maps [n] bytes at [va], which lie in a reservation, as Function checked:
   mapping over anything else would replace the process's own memory. Without
   [va], where the system chooses. *)
let map_bytes ?va n =
  try map_at (Option.value va ~default:0) n
  with Unix.Unix_error (e, _, _) ->
    Fail.fail "allocating %d bytes of system memory: %s" n
      (Unix.error_message e)

(* Returns [n] bytes at [a] to their reservation, or to the system. *)
let unmap_range a n =
  Fail.bug "freeing system memory" (fun () ->
      if Mutex.protect lock (fun () -> reserved_at a n) then release_at a n
      else unmap_at a n)

let map ?va n =
  let n = round_page n in
  Window.v (map_bytes ?va n) n

let unmap w = unmap_range (Window.address w) (Window.length w)

(* Memory that outlives the process

   A GPU that masters the bus without an IOMMU writes physical pages, and keeps
   writing after its process dies, whatever killed it: no exit function runs for
   SIGKILL or a crash. Its memory therefore lies in huge pages of a file under
   the machine's root, [dev/hugepages], named [rig-pci-PID-TIME] for the process
   and the time it made it, so that a file a dead process left is never reused.
   The file keeps its pages when the process dies, and the kernel neither swaps
   nor compacts huge pages.

   Each 2 MiB block of the process's addresses that holds memory maps one huge
   page of the file, which every allocation in the block shares, from any
   function of the machine: memory at an address the caller chose lies in the
   block of that address, and memory without one at addresses a [Tlsf] chooses
   in a range of the process's own. A block's page goes back once no allocation
   holds it and no function maps it as a peer's.

   addresses | block | block | block | 2 MiB each, on 2 MiB | | file [ page ][
   page ][ hole ] one huge page a block

   The file lists the functions that reach it: its reachers, which the process
   keeps in memory and mirrors in a sidecar [<file>.reach], one bus a line,
   written before the function first reaches it, for when the process dies. A
   function that no longer reaches memory leaves the list: at its release the
   process's file, from memory; at its GPU's reset every file under the root
   that a process that died left. A file whose list is empty and that holds no
   block goes, and a dead process's file whose list is empty goes at the next
   take of a function of the machine, blocks and all. The process holds a shared
   flock on its file from before the file has a name until it dies, so a take or
   a reset tells a dead process's file by taking the lock. *)

type file = {
  pid : int; (* the process that made it *)
  root : string;
  fd : Unix.file_descr;
  path : string;
  mutable size : int;
  mutable spare : int list; (* offsets of pages given back *)
  mutable reachers : string list;
}

type block = {
  at : int;
  file : file;
  off : int;
  frame : int; (* the physical address of its first byte *)
  mutable users : int; (* bytes allocations and peers hold in it *)
}

type store = {
  root : string;
  bus : string;
  placed : unit Tables.Address.t; (* the addresses of its live windows *)
}

(* [state] guards every value below and every store, file and block. *)
let state = Mutex.create ()
let files : file list ref = ref [] (* the process's live file of each root *)
let blocks : block Tables.Address.t = Tables.Address.create 16

(* The range memory without an address takes its addresses from. *)
let own_length = 1 lsl 30
let own : Tlsf.t option ref = ref None
let store ~root ~bus = { root; bus; placed = Tables.Address.create 16 }
let root s = s.root
let prefix = "rig-pci-"
let hugepages = "dev/hugepages"
let reach_suffix = ".reach"
let tmp_suffix = reach_suffix ^ ".tmp"

let hugetlbfs =
  "it must be a hugetlbfs of 2 MiB pages: mount -t hugetlbfs -o pagesize=2M \
   none /dev/hugepages"

(* The machine has no free huge page, or the process no free addresses of its
   own: freeing memory makes room. *)
exception Exhausted

let with_remedy why = function None -> why | Some r -> why ^ "; " ^ r

(* Failures to give a file back leak it, which is safe: they are reported and
   the rest goes on. *)
let unlink path =
  try Unix.unlink path with
  | Unix.Unix_error (ENOENT, _, _) -> ()
  | Unix.Unix_error (e, _, _) ->
      prerr_endline
        (strf "rig.pci: keeping %s, which could not be deleted: %s" path
           (Unix.error_message e))

(* A list is written whole to a new file renamed over the old one, so that a
   reader never meets a part of it. *)
let persist path buses =
  let tmp = path ^ tmp_suffix in
  try
    Out_channel.with_open_text tmp (fun oc ->
        List.iter (fun b -> output_string oc (b ^ "\n")) buses);
    Unix.rename tmp (path ^ reach_suffix)
  with Sys_error why | Unix.Unix_error (_, why, _) ->
    Fail.fail "the reachers of %s could not be written: %s" path why

(* A list that loses a bus and is not written keeps a bus that no longer reaches
   the file, which only leaks it. *)
let write_reachers path buses =
  try persist path buses
  with Fail.Failed why -> prerr_endline ("rig.pci: " ^ why)

(* The file goes first: a crash in between leaves a list without its file, which
   the next take of a function of the machine takes. *)
let delete path =
  unlink path;
  unlink (path ^ tmp_suffix);
  unlink (path ^ reach_suffix)

let has_blocks f =
  Tables.Address.fold (fun _ b r -> r || b.file == f) blocks false

(* A child of [fork] leaves its parent's files, and takes no lock a thread of
   its parent may have held. *)
let exit () =
  let pid = Unix.getpid () in
  if List.exists (fun f -> f.pid = pid) !files then
    Mutex.protect state @@ fun () ->
    List.iter (fun f -> if f.reachers = [] then delete f.path) !files

(* A file that lists no function and holds no block goes. *)
let collect f =
  if f.reachers = [] && not (has_blocks f) then begin
    files := List.filter (( != ) f) !files;
    delete f.path;
    Unix.close f.fd
  end

(* The process's file under [root], made with its list naming [bus] before it
   holds a page. It has a name only once locked, and its list only once named: a
   process that meets a file unlocked with no list takes it for one that died
   being made. *)
let file_of ~root bus =
  match List.find_opt (fun (f : file) -> f.root = root) !files with
  | Some f -> f
  | None ->
      let name =
        strf "%s%d-%.0f" prefix (Unix.getpid ()) (Unix.gettimeofday () *. 1e6)
      in
      let dir = Filename.concat root hugepages in
      let path = Filename.concat dir name in
      let fd =
        try create dir path
        with Unix.Unix_error (e, _, _) ->
          Fail.fail "%s"
            (with_remedy
               (strf "creating %s for a GPU's memory: %s" path
                  (Unix.error_message e))
               (match e with
               | ENOENT | EOPNOTSUPP -> Some hugetlbfs
               | _ -> None))
      in
      (try persist path [ bus ]
       with e ->
         delete path;
         Unix.close fd;
         raise e);
      let f =
        {
          pid = Unix.getpid ();
          root;
          fd;
          path;
          size = 0;
          spare = [];
          reachers = [ bus ];
        }
      in
      files := f :: !files;
      f

(* [bus] joins [f]'s list, on disk first. *)
let join f bus =
  if not (List.mem bus f.reachers) then begin
    let buses = f.reachers @ [ bus ] in
    persist f.path buses;
    f.reachers <- buses
  end

let release_range fd off n =
  Fail.bug "punching out a GPU's memory" (fun () -> punch fd off n)

(* A new block at [at] of [root]'s file, its huge page mapped there. *)
let new_block ~root ~bus at =
  let f = file_of ~root bus in
  join f bus;
  let off, spare =
    match f.spare with o :: rest -> (o, rest) | [] -> (f.size, [])
  in
  (match allocate f.fd off huge with
  | () -> ()
  | exception Unix.Unix_error (ENOSPC, _, _) -> raise Exhausted
  | exception Unix.Unix_error (e, _, _) ->
      Fail.fail "%s"
        (with_remedy
           (strf "giving %s a huge page: %s" f.path (Unix.error_message e))
           (match e with EINVAL -> Some hugetlbfs | _ -> None)));
  f.spare <- spare;
  f.size <- Int.max f.size (off + huge);
  let give_back () =
    release_range f.fd off huge;
    f.spare <- off :: f.spare
  in
  (match map_file f.fd off at huge with
  | _ -> ()
  | exception Unix.Unix_error (ENOMEM, _, _) ->
      give_back ();
      raise Exhausted
  | exception Unix.Unix_error (e, _, _) ->
      give_back ();
      Fail.fail "%s"
        (with_remedy
           (strf "mapping a huge page of %s: %s" f.path (Unix.error_message e))
           (match e with EINVAL -> Some hugetlbfs | _ -> None)));
  (* A huge page is one block of frames, so its first page's frame gives them
     all, and its last confirms it: a file system that is no hugetlbfs gives
     pages anywhere. *)
  let first, last =
    try (physical root at, physical root (at + huge - page))
    with e ->
      unmap_range at huge;
      give_back ();
      raise e
  in
  if last - first <> huge - page then begin
    unmap_range at huge;
    give_back ();
    Fail.fail "%s gave a huge page that is not one block of memory; %s"
      (Filename.dirname f.path) hugetlbfs
  end;
  let b = { at; file = f; off; frame = first; users = 0 } in
  Tables.Address.replace blocks at b;
  b

(* The blocks of the [n] bytes at [a], each with the part of them it holds. *)
let parts a n =
  let rec go acc x =
    if x >= a + n then List.rev acc
    else
      let at = x / huge * huge in
      let next = Int.min (a + n) (at + huge) in
      go ((at, x, next - x) :: acc) next
  in
  go [] a

(* [b] goes once nothing holds it. *)
let drop b len =
  b.users <- b.users - len;
  if b.users = 0 then begin
    Tables.Address.remove blocks b.at;
    unmap_range b.at huge;
    let f = b.file in
    release_range f.fd b.off huge;
    f.spare <- b.off :: f.spare;
    collect f
  end

(* The addresses for [n] bytes without an address asked. *)
let own_addresses ~align n =
  let t =
    match !own with
    | Some t -> t
    | None ->
        let base =
          try region own_length
          with Unix.Unix_error (e, _, _) ->
            Fail.fail "reserving addresses for system memory: %s"
              (Unix.error_message e)
        in
        Mutex.protect lock (fun () ->
            Tables.Range.add reserved (base, own_length) ());
        let t = Tlsf.create ~base own_length in
        own := Some t;
        t
  in
  match Tlsf.alloc ~align t n with Some a -> a | None -> raise Exhausted

let own_free a =
  match !own with
  | Some t when a >= Tlsf.base t && a < Tlsf.base t + Tlsf.length t ->
      (* [free] gives back only addresses [placed] holds. *)
      if not (Tlsf.free t a) then
        invalid_arg (strf "Function.free_dma: no addresses at 0x%x" a)
  | _ -> ()

(* The physical runs of the [n] bytes at [a], whose blocks exist: one a block,
   merged where frames follow. *)
let runs a n =
  List.fold_left
    (fun acc (at, x, len) ->
      let pa = (Tables.Address.find blocks at).frame + (x - at) in
      match acc with
      | (p, k) :: rest when p + k = pa -> (p, k + len) :: rest
      | _ -> (pa, len) :: acc)
    [] (parts a n)
  |> List.rev

(* [s] holds the blocks of the [n] bytes at [a], or none of them. *)
let hold s a n =
  let held = ref [] in
  try
    List.iter
      (fun (at, x, len) ->
        let b =
          match Tables.Address.find_opt blocks at with
          | Some b when b.file.root <> s.root ->
              Fail.fail
                "the 2 MiB of addresses at 0x%x hold another machine's memory"
                at
          | Some b ->
              join b.file s.bus;
              zero x len;
              b
          | None -> new_block ~root:s.root ~bus:s.bus at
        in
        b.users <- b.users + len;
        held := (b, len) :: !held)
      (parts a n)
  with e ->
    List.iter (fun (b, len) -> drop b len) !held;
    raise e

let alloc ?(contiguous = false) ?va s n =
  let n = round_page n in
  Mutex.protect state @@ fun () ->
  match
    match va with
    | Some a ->
        hold s a n;
        a
    | None -> (
        let a =
          own_addresses ~align:(if contiguous && n > page then huge else page) n
        in
        try
          hold s a n;
          a
        with e ->
          own_free a;
          raise e)
  with
  | exception Exhausted -> None
  | a ->
      Tables.Address.replace s.placed a ();
      Some (Window.v a n, runs a n)

let free s w =
  let a = Window.address w and n = Window.length w in
  Mutex.protect state @@ fun () ->
  if Tables.Address.mem s.placed a then begin
    Tables.Address.remove s.placed a;
    List.iter
      (fun (at, _, len) ->
        Option.iter (fun b -> drop b len) (Tables.Address.find_opt blocks at))
      (parts a n);
    own_free a
  end

(* The function at [bus] under [root] leaves the lists of the process's
   files. *)
let leave ~root bus =
  List.iter
    (fun (f : file) ->
      if f.root = root && List.mem bus f.reachers then begin
        f.reachers <- List.filter (( <> ) bus) f.reachers;
        write_reachers f.path f.reachers;
        collect f
      end)
    !files

let close s = Mutex.protect state @@ fun () -> leave ~root:s.root s.bus

(* Every block of the [n] bytes at [a] must be [root]'s: memory of the process's
   own, or of another machine, would not outlive the process. *)
let reach ~root ~bus a n =
  Mutex.protect state @@ fun () ->
  let held (at, _, _) =
    match Tables.Address.find_opt blocks at with
    | Some b -> b.file.root = root
    | None -> false
  in
  let ps = parts a n in
  if not (List.for_all held ps) then None
  else begin
    let bs =
      List.map (fun (at, _, len) -> (Tables.Address.find blocks at, len)) ps
    in
    (* Each file lists [bus] before a block counts it: a failed record counts
       nothing, and a list longer than needed only leaks. *)
    List.iter (fun (b, _) -> join b.file bus) bs;
    List.iter (fun (b, len) -> b.users <- b.users + len) bs;
    Some (runs a n)
  end

let unreach ~a ~n =
  Mutex.protect state @@ fun () ->
  List.iter
    (fun (at, _, len) ->
      Option.iter (fun b -> drop b len) (Tables.Address.find_opt blocks at))
    (parts a n)

(* [f ()] if the process that made the file at [path] died: the lock it held
   while it lived is free. *)
let if_dead path f =
  match Unix.openfile path [ O_RDWR; O_CLOEXEC ] 0 with
  | exception Unix.Unix_error _ -> ()
  | fd -> (
      Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
      match own_lock fd with () -> f () | exception Unix.Unix_error _ -> ())

(* A dead process's file without a list died being made, and holds no page. *)
let reachers_on_disk path =
  let list = path ^ reach_suffix in
  match Unix.openfile list [ O_RDONLY; O_CLOEXEC ] 0 with
  | exception Unix.Unix_error (ENOENT, _, _) -> None
  | exception Unix.Unix_error (e, _, _) ->
      Fail.fail "reading %s: %s" list (Unix.error_message e)
  | fd ->
      let ic = Unix.in_channel_of_descr fd in
      Fun.protect ~finally:(fun () -> close_in_noerr ic) @@ fun () ->
      Some (List.filter (( <> ) "") (In_channel.input_lines ic))

(* The memory files under [root] other processes made, and the lists that
   outlived their files. [state] is held. *)
let others ~root =
  let mine = List.map (fun (f : file) -> f.path) !files in
  let dir = Filename.concat root hugepages in
  match Sys.readdir dir with
  | exception Sys_error _ -> ([], [])
  | names ->
      Array.fold_right
        (fun name (files, orphans) ->
          let path = Filename.concat dir name in
          if
            String.ends_with ~suffix:reach_suffix name
            && not (Sys.file_exists (Filename.chop_suffix path reach_suffix))
          then (files, path :: orphans)
          else if
            String.starts_with ~prefix name
            && (not (String.ends_with ~suffix:reach_suffix name))
            && (not (String.ends_with ~suffix:tmp_suffix name))
            && not (List.mem path mine)
          then (path :: files, orphans)
          else (files, orphans))
        names ([], [])

(* The bus whose GPU was [reset], if any, leaves the lists of the files under
   [root] that processes that died left, and the files no function reaches go.
   [state] is held. *)
let sweep ~root ~reset =
  let files, orphans = others ~root in
  List.iter unlink orphans;
  (* A list that cannot be read keeps its file, which only leaks it. *)
  List.iter
    (fun path ->
      if_dead path (fun () ->
          match reachers_on_disk path with
          | exception Fail.Failed why -> prerr_endline ("rig.pci: " ^ why)
          | None | Some [] -> delete path
          | Some buses -> (
              match List.filter (fun b -> Some b <> reset) buses with
              | [] -> delete path
              | rest when rest <> buses -> write_reachers path rest
              | _ -> ())))
    files

let collect_dead ~root =
  Mutex.protect state @@ fun () -> sweep ~root ~reset:None

let forget ~root ~bus =
  Mutex.protect state @@ fun () -> sweep ~root ~reset:(Some bus)

(* A file that cannot be opened for another reason than its end might hold the
   GPU's memory: it is no answer. *)
let left ~root ~bus =
  Mutex.protect state @@ fun () ->
  let listed = ref false in
  List.iter
    (fun path ->
      if not !listed then
        match Unix.openfile path [ O_RDWR; O_CLOEXEC ] 0 with
        | exception Unix.Unix_error (ENOENT, _, _) -> ()
        | exception Unix.Unix_error (e, _, _) ->
            Fail.fail "reading %s: %s" path (Unix.error_message e)
        | fd -> (
            Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
            match own_lock fd with
            | exception Unix.Unix_error _ -> ()
            | () ->
                listed :=
                  List.mem bus
                    (Option.value (reachers_on_disk path) ~default:[])))
    (fst (others ~root));
  !listed
