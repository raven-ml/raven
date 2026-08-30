(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module File_io = Hcq.File_io

module Ffi = struct
  type constants = {
    o_wronly : int;
    o_creat : int;
    o_sync : int;
    map_locked : int;
    map_populate : int;
    map_hugetlb : int;
    map_fixed_noreplace : int;
    page_size : int;
    linux : bool;
  }

  external constants : unit -> constants = "caml_tolk_amd_system_constants"

  external open_mode : string -> int -> int -> int
    = "caml_tolk_amd_system_open_mode"

  external flock_try : int -> bool = "caml_tolk_amd_system_flock_try"
  [@@noalloc]

  external mlock : nativeint -> int -> bool = "caml_tolk_amd_system_mlock"
  [@@noalloc]

  external madvise_dontfork : nativeint -> int -> unit
    = "caml_tolk_amd_system_madvise_dontfork"

  external pread : int -> bytes -> int -> int -> int64 -> int
    = "caml_tolk_amd_system_pread"

  external pwrite : int -> bytes -> int -> int -> int64 -> int
    = "caml_tolk_amd_system_pwrite"

  external write : int -> bytes -> int -> int = "caml_tolk_amd_system_write"
end

let {
      Ffi.o_wronly;
      o_creat;
      o_sync;
      map_locked;
      map_populate;
      map_hugetlb;
      map_fixed_noreplace;
      page_size;
      linux = available;
    } =
  Ffi.constants ()

(* Small helpers *)

let round_up x align = (x + align - 1) / align * align
let ceildiv x d = (x + d - 1) / d

let bit_length n =
  let rec go acc n = if n = 0 then acc else go (acc + 1) (n lsr 1) in
  go 0 n

let has_substring hay needle =
  let nlen = String.length needle in
  let rec at i =
    i + nlen <= String.length hay
    && (String.equal (String.sub hay i nlen) needle || at (i + 1))
  in
  at 0

(* Sysfs files are short: whole-file reads and single-write stores. *)

let read_file path =
  match In_channel.with_open_bin path In_channel.input_all with
  | s -> s
  | exception Sys_error msg -> failwith msg

let first_line s =
  match String.index_opt s '\n' with
  | Some i -> String.sub s 0 i
  | None -> s

let hex_value ~ctx s =
  let s = String.trim s in
  let prefixed =
    if String.length s >= 2 && s.[0] = '0' && (s.[1] = 'x' || s.[1] = 'X') then
      s
    else "0x" ^ s
  in
  match int_of_string_opt prefixed with
  | Some v -> v
  | None ->
      failwith
        (Printf.sprintf "%s: expected a hex value, got %s" ctx
           (if s = "" then "an empty file" else s))

let read_hex path = hex_value ~ctx:path (read_file path)

let write_str path s =
  let fd = File_io.openfile path ~flags:o_wronly in
  Fun.protect
    ~finally:(fun () -> File_io.close fd)
    (fun () ->
      let b = Bytes.of_string s in
      let n =
        try Ffi.write fd b (Bytes.length b)
        with Failure msg -> failwith (path ^ ": " ^ msg)
      in
      if n <> Bytes.length b then failwith (path ^ ": short write"))

let pread_exact fd buf ~off ~what =
  let len = Bytes.length buf in
  let rec go pos off =
    if pos < len then begin
      let n = Ffi.pread fd buf pos (len - pos) off in
      if n = 0 then failwith (what ^ ": unexpected end of file");
      go (pos + n) (Int64.add off (Int64.of_int n))
    end
  in
  go 0 off

let pwrite_exact fd buf ~off ~what =
  let len = Bytes.length buf in
  let rec go pos off =
    if pos < len then begin
      let n = Ffi.pwrite fd buf pos (len - pos) off in
      if n = 0 then failwith (what ^ ": short write");
      go (pos + n) (Int64.add off (Int64.of_int n))
    end
  in
  go 0 off

(* System facilities *)

let memory_barrier = Hcq.Mmio.fence

let write_sysfs ?expected path ~value ~msg =
  let expected = match expected with Some e -> e | None -> value in
  if not (String.equal (first_line (read_file path)) expected) then begin
    let cmd = Printf.sprintf "sudo sh -c 'echo %s > %s'" value path in
    ignore (Sys.command cmd : int);
    if not (String.equal (first_line (read_file path)) expected) then
      failwith (Printf.sprintf "%s. Please run %s manually." msg cmd)
  end

(* Locked pages must not migrate, or the physical addresses handed to
   devices would go stale. *)
let pagemap =
  lazy
    (write_sysfs "/proc/sys/vm/compact_unevictable_allowed" ~value:"0"
       ~msg:"Failed to disable migration of locked pages";
     File_io.openfile "/proc/self/pagemap" ~flags:File_io.o_rdonly)

let reserved = Hashtbl.create 4

let reserve_va ~va_start ~va_size =
  if not (Hashtbl.mem reserved (va_start, va_size)) then begin
    ignore
      (File_io.mmap ~addr:va_start ~size:va_size ~prot:File_io.prot_none
         ~flags:
           (File_io.map_private lor File_io.map_anonymous
          lor File_io.map_noreserve lor map_fixed_noreplace)
         ~fd:(-1) ~offset:0L
        : nativeint);
    Hashtbl.add reserved (va_start, va_size) ()
  end

let lock_memory ~addr ~size =
  if not (Ffi.mlock addr size) then
    failwith
      (Printf.sprintf "Failed to lock memory at 0x%nx with size 0x%x" addr size)

let system_paddrs ?pagemap:pm ~vaddr size =
  let fd = match pm with Some fd -> fd | None -> Lazy.force pagemap in
  let n = size / page_size in
  let buf = Bytes.create (n * 8) in
  pread_exact fd buf
    ~off:(Int64.of_int (Nativeint.to_int vaddr / page_size * 8))
    ~what:"pagemap";
  (* A page-map entry is the physical frame number in its low 55 bits. *)
  List.init n (fun i ->
      Int64.to_int
        (Int64.logand (Bytes.get_int64_le buf (8 * i)) 0x7F_FFFF_FFFF_FFFFL)
      * page_size)

let pci_scan_bus ?(sysfs = "/sys") ?base_class ~vendor devices =
  let dir = sysfs ^ "/bus/pci/devices" in
  let entries =
    match Sys.readdir dir with
    | a -> Array.to_list a
    | exception Sys_error _ -> failwith "no pcie"
  in
  let matches pcibus =
    let file f = String.concat "/" [ dir; pcibus; f ] in
    (match base_class with
    | None -> true
    | Some bc -> read_hex (file "class") lsr 16 = bc)
    && read_hex (file "vendor") = vendor
    &&
    let device = read_hex (file "device") in
    List.exists (fun (mask, ids) -> List.mem (device land mask) ids) devices
  in
  List.sort String.compare (List.filter matches entries)

let flock_acquire name =
  let lock_name = Filename.concat (Filename.get_temp_dir_name ()) name in
  let fd =
    if Sys.file_exists lock_name then
      File_io.openfile lock_name ~flags:File_io.o_rdwr
    else Ffi.open_mode lock_name (File_io.o_rdwr lor o_creat) 0o666
  in
  if not (Ffi.flock_try fd) then begin
    File_io.close fd;
    failwith
      (Printf.sprintf
         "Failed to acquire lock file %s. `sudo lsof %s` may help identify \
          the process holding the lock."
         name lock_name)
  end;
  fd

(* PCI devices *)

module Pci_device = struct
  type t = {
    pcibus : string;
    dev_path : string;
    lock_fd : int;
    cfg_fd : int;
    bar_fds : (int, int) Hashtbl.t;
    bar_infos : (int, int * int) Hashtbl.t;
  }

  let pcibus t = t.pcibus
  let lock_fd t = t.lock_fd

  let create ?(sysfs = "/sys") ~devpref pcibus =
    let lock_fd =
      flock_acquire
        (String.lowercase_ascii devpref ^ "_" ^ String.lowercase_ascii pcibus
       ^ ".lock")
    in
    let dev_path = sysfs ^ "/bus/pci/devices/" ^ pcibus in
    (match File_io.openfile (dev_path ^ "/enable") ~flags:File_io.o_rdwr with
    | fd -> File_io.close fd
    | exception Failure msg ->
        if
          has_substring msg "Permission denied"
          || has_substring msg "Operation not permitted"
        then
          failwith
            (Printf.sprintf
               "Cannot access PCI device %s: run as root or grant the process \
                CAP_SYS_ADMIN with setcap"
               pcibus)
        else failwith msg);
    if Sys.file_exists (dev_path ^ "/driver") then
      write_str (dev_path ^ "/driver/unbind") pcibus;
    if Sys.file_exists (dev_path ^ "/driver") then
      failwith ("Driver is bound to " ^ pcibus);
    for fn = 1 to 7 do
      let sib =
        Printf.sprintf "%s/bus/pci/devices/%s%d" sysfs
          (String.sub pcibus 0 (String.length pcibus - 1))
          fn
      in
      if Sys.file_exists sib then write_str (sib ^ "/remove") "1"
    done;
    write_str (dev_path ^ "/enable") "1";
    let cfg_fd =
      File_io.openfile (dev_path ^ "/config")
        ~flags:(File_io.o_rdwr lor o_sync)
    in
    {
      pcibus;
      dev_path;
      lock_fd;
      cfg_fd;
      bar_fds = Hashtbl.create 4;
      bar_infos = Hashtbl.create 4;
    }

  let alloc_sysmem ?(vaddr = 0n) ?(contiguous = false) size =
    if contiguous && size > 2 lsl 20 then
      invalid_arg "Contiguous allocation is only supported for sizes up to 2MB";
    let size = if contiguous then round_up size page_size else size in
    let flags =
      (if contiguous && size > page_size then map_hugetlb else 0)
      lor if vaddr <> 0n then File_io.map_fixed else 0
    in
    let va =
      try
        File_io.mmap ~addr:vaddr ~size
          ~prot:(File_io.prot_read lor File_io.prot_write)
          ~flags:
            (File_io.map_shared lor File_io.map_anonymous lor map_populate
           lor map_locked lor flags)
          ~fd:(-1) ~offset:0L
      with Failure msg ->
        let hint =
          if contiguous && size > page_size then
            " (raise the locked-memory limit with `ulimit -l` and reserve \
             huge pages via /proc/sys/vm/nr_hugepages)"
          else " (raise the locked-memory limit with `ulimit -l`)"
        in
        failwith (msg ^ hint)
    in
    let paddrs = system_paddrs ~vaddr:va size in
    let paddrs_4k =
      List.concat_map
        (fun p -> List.init (page_size / 0x1000) (fun i -> p + (i * 0x1000)))
        paddrs
    in
    ( Hcq.Mmio.make ~addr:va ~size,
      List.filteri (fun i _ -> i < ceildiv size 0x1000) paddrs_4k )

  let reset t =
    ignore
      (Sys.command (Printf.sprintf "sudo sh -c 'echo 1 > %s/reset'" t.dev_path)
        : int)

  let read_config t ~offset ~size =
    let buf = Bytes.create size in
    pread_exact t.cfg_fd buf ~off:(Int64.of_int offset) ~what:"config";
    let v = ref 0 in
    for i = size - 1 downto 0 do
      v := (!v lsl 8) lor Char.code (Bytes.get buf i)
    done;
    !v

  let write_config t ~offset ~value ~size =
    let buf = Bytes.create size in
    for i = 0 to size - 1 do
      Bytes.set buf i (Char.chr ((value lsr (8 * i)) land 0xFF))
    done;
    pwrite_exact t.cfg_fd buf ~off:(Int64.of_int offset) ~what:"config"

  let write_config_flush t ~offset ~value ~size =
    write_config t ~offset ~value ~size;
    ignore (read_config t ~offset ~size : int)

  let bar_fd t bar =
    match Hashtbl.find_opt t.bar_fds bar with
    | Some fd -> fd
    | None ->
        let fd =
          File_io.openfile
            (Printf.sprintf "%s/resource%d" t.dev_path bar)
            ~flags:(File_io.o_rdwr lor o_sync)
        in
        Hashtbl.add t.bar_fds bar fd;
        fd

  let bar_info t bar =
    match Hashtbl.find_opt t.bar_infos bar with
    | Some info -> info
    | None ->
        let path = t.dev_path ^ "/resource" in
        let line =
          match
            List.nth (String.split_on_char '\n' (read_file path)) bar
          with
          | line -> line
          | exception (Failure _ | Invalid_argument _) ->
              failwith (Printf.sprintf "%s: no line for BAR %d" path bar)
        in
        let fields =
          List.filter (fun s -> s <> "") (String.split_on_char ' ' line)
        in
        (match fields with
        | s :: e :: _ ->
            let s = hex_value ~ctx:path s and e = hex_value ~ctx:path e in
            let info = (s, e - s + 1) in
            Hashtbl.add t.bar_infos bar info;
            info
        | _ -> failwith (Printf.sprintf "%s: no line for BAR %d" path bar))

  let map_bar t ?(off = 0) ?(addr = 0n) ?size bar =
    let fd = bar_fd t bar in
    let sz =
      match size with Some s -> s | None -> snd (bar_info t bar) - off
    in
    let loc =
      File_io.mmap ~addr ~size:sz
        ~prot:(File_io.prot_read lor File_io.prot_write)
        ~flags:
          (File_io.map_shared
          lor if addr <> 0n then File_io.map_fixed else 0)
        ~fd ~offset:(Int64.of_int off)
    in
    Ffi.madvise_dontfork loc sz;
    Hcq.Mmio.make ~addr:loc ~size:sz

  let resize_bar t bar =
    let rpath = Printf.sprintf "%s/resource%d_resize" t.dev_path bar in
    match write_str rpath (string_of_int (bit_length (read_hex rpath) - 1)) with
    | () -> ()
    | exception Failure msg ->
        failwith
          (Printf.sprintf
             "Cannot resize BAR %d: %s. Ensure the resizable BAR option is \
              enabled."
             bar msg)
end
