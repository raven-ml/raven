(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let strf = Printf.sprintf

(* Sizes and addresses *)

let kib = 1024
let mib = 1 lsl 20
let gib = 1 lsl 30
let round_up n a = (n + a - 1) / a * a
let is_pow2 n = n > 0 && n land (n - 1) = 0

let pow2_floor n =
  let rec go p = if p > n / 2 then p else go (2 * p) in
  go 1

let pp_hex ppf x = Format.fprintf ppf "0x%x" x

let hex =
  Testable.with_compare Int.compare (Testable.make ~pp:pp_hex ~equal:Int.equal)

let free_base = 0x100_0000_0000

let largest_gap lo hi live =
  let sorted = List.sort compare live in
  let at, gap =
    List.fold_left
      (fun (at, gap) (a, n) -> (a + n, max gap (a - at)))
      (lo, 0) sorted
  in
  max gap (hi - at)

let fits ~gap n a = n <= gap / 2 && a <= (gap / 2) - n
let on_linux = Sys.file_exists "/sys/bus/pci/devices"
let flocks = not Sys.win32
let ofd_locks = not Sys.win32

external now_ns : unit -> int = "rig_pci_test_now_ns"

(* The factor RIG_TEST_TIME_SCALE names, which a profile whose build slows
   every test sets, as Rig_support.time reads it. *)
let time_scale =
  match Sys.getenv_opt "RIG_TEST_TIME_SCALE" with
  | None -> 1.
  | Some s -> (
      match float_of_string_opt s with
      | Some k when k >= 1. -> k
      | _ -> invalid_arg ("RIG_TEST_TIME_SCALE=" ^ s ^ ": not a factor >= 1"))

let patience = 10. *. time_scale
let patience_ns = int_of_float (patience *. 1e9)

let poll f =
  let t0 = now_ns () in
  let rec go () =
    f ()
    || now_ns () - t0 < patience_ns
       && begin
         Unix.sleepf 0.0005;
         go ()
       end
  in
  go ()

(* This machine's GPUs *)

let this_gpus () =
  match Rig_pci.Machine.functions Rig_pci.Machine.this with
  | Error why -> failwith why
  | Ok ids ->
      List.filter (fun (id : Rig_pci.Machine.id) -> id.class_ lsr 16 = 0x03) ids

(* Process memory and far machines *)

external memory : int -> int = "rig_pci_test_memory"
external far : int -> int -> int = "rig_pci_test_far"
external break : int -> unit = "rig_pci_test_far_break"
external break_at : int -> int -> unit = "rig_pci_test_far_break_at"
external hold : int -> unit = "rig_pci_test_far_hold"
external waiting : int -> bool = "rig_pci_test_far_waiting"
external let_go : int -> unit = "rig_pci_test_far_let_go"
external log : int -> (bool * int * int) list = "rig_pci_test_far_log"

let unlisted_why = "far: the link broke"

let unlisted () =
  let page = 4096 in
  let tr = far 0 page in
  break tr;
  Rig_pci.Machine.make ~name:"unlisted"
    {
      transport = Rig_pci.Window.unsafe_transport tr;
      page;
      functions = (fun () -> Error unlisted_why);
      take = (fun _ -> Windtrap.fail "unlisted: a take");
      reserve = (fun ~base:_ _ -> Windtrap.fail "unlisted: a reservation");
    }

(* Page tables in a fake format *)

module Tables = struct
  open Rig_pci

  type memory = {
    entries : (int, int64) Hashtbl.t;
    mutable zeroed : (int * int) list;
    mutable unflushed : int;
    mutable touches : int;
    mutable confirms : bool;
  }

  let memory () =
    {
      entries = Hashtbl.create 64;
      zeroed = [];
      unflushed = 0;
      touches = 0;
      confirms = true;
    }

  let shifts = [| 39; 30; 21; 12 |]
  let leaf = 3
  let large l = l >= 1
  let address_mask = 0xF_FFFF_FFFF_F000
  let bit b i = if b then 1 lsl i else 0

  let target_code = function
    | Page_table.Gpu -> 0
    | System -> 4
    | Peer i -> 8 lor ((i land 15) lsl 52)

  let target_of e =
    match e land 12 with
    | 0 -> Page_table.Gpu
    | 4 -> System
    | _ -> Peer ((e lsr 52) land 15)

  let entry_at m table i =
    Option.value ~default:0L (Hashtbl.find_opt m.entries (table + (8 * i)))

  let format m =
    let touch () = m.touches <- m.touches + 1 in
    let set table i e =
      touch ();
      Hashtbl.replace m.entries (table + (8 * i)) (Int64.of_int e);
      m.unflushed <- m.unflushed + 1
    in
    {
      Page_table.levels = [ 12; 21; 30; 39 ];
      bits = 48;
      pa_bits = 52;
      first = 0;
      set_table = (fun ~level:_ ~table i ~child -> set table i (child lor 1));
      set_page =
        (fun ~level:_ ~table i ~pa tg ~uncached ~snooped ~fragment ->
          set table i
            (pa lor 3 lor target_code tg lor bit uncached 4 lor bit snooped 5
            lor ((fragment land 63) lsl 6)));
      clear = (fun ~level:_ ~table i -> set table i 0);
      large = (fun ~level -> large level);
      zero =
        (fun pa n ->
          touch ();
          m.zeroed <- (pa, n) :: m.zeroed;
          Hashtbl.filter_map_inplace
            (fun k e -> if k >= pa && k < pa + n then None else Some e)
            m.entries);
      flush =
        (fun () ->
          touch ();
          m.unflushed <- 0;
          m.confirms);
    }

  type entry = {
    va : int;
    level : int;
    pa : int;
    target : Page_table.target;
    uncached : bool;
    snooped : bool;
    fragment : int;
  }

  let pp_target ppf t =
    Format.pp_print_string ppf
      (match t with
      | Page_table.Gpu -> "gpu"
      | System -> "system"
      | Peer i -> "peer " ^ string_of_int i)

  let target = Testable.make ~pp:pp_target ~equal:( = )

  let pp_entry ppf e =
    Format.fprintf ppf "{va 0x%x; level %d; pa 0x%x; %a%s%s; fragment %d}" e.va
      e.level e.pa pp_target e.target
      (if e.uncached then " uncached" else "")
      (if e.snooped then " snooped" else "")
      e.fragment

  let entry = Testable.make ~pp:pp_entry ~equal:( = )

  let walk m t =
    let pages = ref [] and tables = ref [] in
    let rec go table level va =
      tables := table :: !tables;
      for i = 0 to 511 do
        let e = Int64.to_int (entry_at m table i) in
        let va = va + (i lsl shifts.(level)) in
        if e land 1 = 0 then ()
        else if level = leaf || e land 2 <> 0 then
          pages :=
            {
              va;
              level;
              pa = e land address_mask;
              target = target_of e;
              uncached = e land 16 <> 0;
              snooped = e land 32 <> 0;
              fragment = (e lsr 6) land 63;
            }
            :: !pages
        else go (e land address_mask) (level + 1) va
      done
    in
    go (Page_table.root t) 0 (Page_table.base t);
    (List.rev !pages, List.rev !tables)
end

(* Page tables in a buffer *)

module Buffer_tables = struct
  open Rig_pci

  let memory = gib
  let boot = mib
  let tables_end = boot + (2 * mib)
  let pages = [ (2 * mib, 2 * mib); (4 * kib, 4 * kib) ]

  type entries =
    (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t

  let entries () : entries =
    let b =
      Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout (tables_end / 8)
    in
    Bigarray.Array1.fill b 0L;
    b

  let set (b : entries) table i e =
    Bigarray.Array1.set b ((table lsr 3) + i) (Int64.of_int e)

  (* Memory past the tables holds no entries and is not kept. *)
  let zero (b : entries) pa n =
    if pa < tables_end then
      for i = pa lsr 3 to ((pa + n) lsr 3) - 1 do
        Bigarray.Array1.set b i 0L
      done

  let large ~level = level >= 2
  let levels = [ 12; 21; 30; 39 ]
  let bits = 48

  let create space =
    let b = entries () in
    let format =
      {
        Page_table.levels;
        bits;
        pa_bits = bits;
        first = 0;
        set_table =
          (fun ~level:_ ~table i ~child -> set b table i (child lor 3));
        set_page =
          (fun ~level:_ ~table i ~pa _ ~uncached:_ ~snooped:_ ~fragment:_ ->
            set b table i (pa lor 1));
        clear = (fun ~level:_ ~table i -> set b table i 0);
        large;
        zero = zero b;
        flush = (fun () -> true);
      }
    in
    let t = Page_table.create format space ~memory ~boot ~tables:Pool ~pages in
    Page_table.booted t;
    t
end

(* Machines in a fixture tree *)

external device_number : string -> string = "rig_pci_test_device_number"
external stored : string -> int = "rig_pci_test_stored"
external flocked : string -> bool = "rig_pci_test_flocked"
external flocked_fd : Unix.file_descr -> bool = "rig_pci_test_flocked_fd"
external ofd_hold : string -> Unix.file_descr = "rig_pci_test_ofd_hold"
external ofd_held : string -> bool = "rig_pci_test_ofd_held"

module Tree = struct
  type bar = Mem32 of int * int | Mem64 of int * int | Io of int * int

  type fn = {
    bus : string;
    vendor : int;
    device : int;
    class_ : int;
    driver : string option;
    group : string option;
    enabled : bool;
    reset : string option;
    host : string;
    bridges : string list;
    bars : bar list;
  }

  let gpu ?driver ?group ?(enabled = true) ?(reset = Some "bus")
      ?(host = "pci0000:00") ?(bridges = [ "0000:00:01.0" ]) bus =
    {
      bus;
      host;
      bridges;
      vendor = 0x1002;
      device = 0x744c;
      class_ = 0x030000;
      driver;
      group;
      enabled;
      reset;
      bars =
        [
          Mem64 (0x7c_0000_0000, 256 * mib);
          Mem64 (0xfc00_0000, 2 * mib);
          Io (0xe000, 256);
          Mem32 (0xfcc0_0000, mib);
        ];
    }

  let ( / ) = Filename.concat

  let rec mkdir_p d =
    if not (Sys.file_exists d) then begin
      mkdir_p (Filename.dirname d);
      Sys.mkdir d 0o755
    end

  let rec remove path =
    match Unix.lstat path with
    | { st_kind = S_DIR; _ } ->
        Array.iter (fun f -> remove (path / f)) (Sys.readdir path);
        Unix.rmdir path
    | _ -> Unix.unlink path
    | exception Unix.Unix_error (ENOENT, _, _) -> ()

  let write file s =
    mkdir_p (Filename.dirname file);
    Out_channel.with_open_bin file (fun oc -> output_string oc s)

  (* The process's trees are under a new temporary directory of its own, so
     that no two runs, of one suite or of two, share a tree. The process that
     made it removes it at its exit; a child of [fork] leaves it. It is under
     /dev/shm, a memory file system, where the system has one: a suite makes
     thousands of trees, which took ten times as long on a CI runner whose
     temporary directory is on disk. *)
  let trees =
    lazy
      (let temp_dir =
         if Sys.file_exists "/dev/shm" then "/dev/shm"
         else Filename.get_temp_dir_name ()
       in
       let d = Filename.temp_dir ~temp_dir "rig-pci-trees" "" in
       let maker = Unix.getpid () in
       at_exit (fun () -> if Unix.getpid () = maker then remove d);
       d)

  let count = Atomic.make 0

  (* BAR registers: memory BARs keep their low four bits for flags, a 64-bit one
     has type bits 0b10 and its upper half in the next register, an I/O BAR has
     bit 0 set. *)
  let registers bars =
    List.concat_map
      (function
        | Mem32 (a, _) -> [ a ]
        | Mem64 (a, _) -> [ a land 0xffff_fff0 lor 0b100; a lsr 32 ]
        | Io (a, _) -> [ a lor 1 ])
      bars

  (* The resource file's lines: a BAR's first and last address, or zeroes for an
     index without one, such as a 64-bit BAR's upper half. *)
  let resource bars =
    let line (a, n) = strf "0x%016x 0x%016x 0x%016x" a (a + n - 1) 0 in
    let zero = strf "0x%016x 0x%016x 0x%016x" 0 0 0 in
    let lines =
      List.concat_map
        (function
          | Mem32 (a, n) | Io (a, n) -> [ line (a, n) ]
          | Mem64 (a, n) -> [ line (a, n); zero ])
        bars
    in
    String.concat "\n"
      (lines @ List.init (13 - List.length lines) (fun _ -> zero))
    ^ "\n"

  (* Configuration space as a reader without CAP_SYS_ADMIN sees it, its first
     64 bytes, or as root does, all 4096. *)
  let config_bytes privileged = if privileged then 4096 else 64

  let config ~privileged fn =
    let b = Bytes.make (config_bytes privileged) '\000' in
    Bytes.set_uint16_le b 0 fn.vendor;
    Bytes.set_uint16_le b 2 fn.device;
    Bytes.set_uint8 b 0x09 (fn.class_ land 0xff);
    Bytes.set_uint8 b 0x0a ((fn.class_ lsr 8) land 0xff);
    Bytes.set_uint8 b 0x0b (fn.class_ lsr 16);
    List.iteri
      (fun i r -> Bytes.set_int32_le b (0x10 + (4 * i)) (Int32.of_int r))
      (registers fn.bars);
    Bytes.to_string b

  (* A bridge above functions: an AMD root port, PCI-to-PCI bridges' class. *)
  let bridge ~privileged d =
    if not (Sys.file_exists (d / "class")) then begin
      write (d / "vendor") "0x1022\n";
      write (d / "device") "0x1483\n";
      write (d / "class") "0x060400\n";
      write (d / "config") (String.make (config_bytes privileged) '\000')
    end

  (* The link at [link] to [target], relative, so that a clone of the tree
     links within itself. *)
  let relative link target =
    let parts p = List.filter (( <> ) "") (String.split_on_char '/' p) in
    let rec drop a b =
      match (a, b) with
      | x :: a', y :: b' when x = y -> drop a' b'
      | _ -> (a, b)
    in
    let from, rest = drop (parts (Filename.dirname link)) (parts target) in
    Unix.symlink
      (String.concat "/" (List.map (fun _ -> "..") from @ rest))
      link

  let build ~lockdown ~groups ~noiommu ~privileged root fns =
    let sys = root / "sys" in
    let devices = sys / "bus/pci/devices" in
    write (sys / "kernel/security/lockdown") (lockdown ^ "\n");
    write (sys / "bus/pci/rescan") "";
    write (sys / "bus/pci/drivers_probe") "";
    List.iter
      (fun fn ->
        (* Linux keeps a function under its host bridge and bridges in
           /sys/devices, and links it from /sys/bus/pci/devices. *)
        let above =
          List.fold_left
            (fun dir b ->
              let d = dir / b in
              bridge ~privileged d;
              d)
            (sys / "devices" / fn.host)
            fn.bridges
        in
        let d = above / fn.bus in
        mkdir_p d;
        mkdir_p devices;
        relative (devices / fn.bus) d;
        write (d / "vendor") (strf "0x%04x\n" fn.vendor);
        write (d / "device") (strf "0x%04x\n" fn.device);
        write (d / "class") (strf "0x%06x\n" fn.class_);
        write (d / "enable") (if fn.enabled then "1\n" else "0\n");
        write (d / "resource") (resource fn.bars);
        write (d / "config") (config ~privileged fn);
        (* A BAR's file is as long as the BAR, as Linux makes it: sparse, so
           it holds no bytes on disk. *)
        let zeroes file n =
          write file "";
          Unix.truncate file n
        in
        List.iteri
          (fun i r ->
            let file = d / strf "resource%d" i in
            match r with
            | `Mem n -> zeroes file n
            | `Prefetchable n ->
                zeroes file n;
                zeroes (file ^ "_wc") n
            | `None -> ())
          (List.concat_map
             (function
               | Mem32 (_, n) -> [ `Mem n ]
               | Mem64 (_, n) -> [ `Prefetchable n; `None ]
               | Io _ -> [ `None ])
             fn.bars);
        write (d / "remove") "";
        write (d / "driver_override") "(null)\n";
        Option.iter
          (fun methods ->
            write (d / "reset") "";
            write (d / "reset_method") (methods ^ "\n"))
          fn.reset;
        Option.iter
          (fun drv ->
            write (sys / "bus/pci/drivers" / drv / "bind") "";
            write (sys / "bus/pci/drivers" / drv / "unbind") "";
            relative (d / "driver") (sys / "bus/pci/drivers" / drv))
          fn.driver;
        Option.iter
          (fun g ->
            relative (d / "iommu_group") (sys / "kernel/iommu_groups" / g);
            let gd = sys / "kernel/iommu_groups" / g in
            write (gd / "devices" / fn.bus) "";
            write (gd / "type")
              (Option.value (List.assoc_opt g groups) ~default:"DMA" ^ "\n"))
          fn.group)
      fns;
    List.iter (fun g -> write (root / "dev/vfio" / ("noiommu-" ^ g)) "") noiommu;
    mkdir_p (root / "proc/self/fd");
    mkdir_p (root / "dev/hugepages");
    write (root / "proc/self/mounts")
      "none /dev/hugepages hugetlbfs rw,relatime,pagesize=2M 0 0\n"

  (* Trees are built once per description and cloned where the file system
     clones a directory in one call, as APFS does: a disk without /dev/shm
     made each tree ten times as slowly. *)
  external clone : string -> string -> bool = "rig_pci_test_clone"

  let templates = Hashtbl.create 16
  let templates_lock = Mutex.create ()

  (* Whether the trees' file system clones a directory. *)
  let clones =
    lazy
      (let probe = Lazy.force trees / "probe" in
       mkdir_p probe;
       clone probe (probe ^ "-clone"))

  let make ?(lockdown = "[none] integrity confidentiality") ?(groups = [])
      ?(noiommu = []) ?(privileged = false) fns =
    if Sys.win32 then skip ~reason:"Windows names no file with a colon" ();
    let root =
      Lazy.force trees / string_of_int (Atomic.fetch_and_add count 1)
    in
    let build = build ~lockdown ~groups ~noiommu ~privileged in
    if not (Lazy.force clones) then build root fns
    else begin
      let described = (lockdown, groups, noiommu, privileged, fns) in
      let key =
        Digest.to_hex (Digest.string (Marshal.to_string described []))
      in
      let template =
        Mutex.protect templates_lock @@ fun () ->
        match Hashtbl.find_opt templates key with
        | Some t -> t
        | None ->
            let t = Lazy.force trees / ("template-" ^ key) in
            build t fns;
            Hashtbl.add templates key t;
            t
      in
      if not (clone template root) then build root fns
    end;
    root

  let device_number = device_number
  let stored = stored
  let flocked = flocked
  let flocked_fd = flocked_fd
  let hold = ofd_hold
  let held = ofd_held
  let add root file s = write (root / file) s

  (* An entry of a page map: bit 63 says the page is present, bits 0-54 hold its
     frame (proc(5), /proc/pid/pagemap). *)
  let pagemap root ~page a frames =
    let file = root / "proc/self/pagemap" in
    mkdir_p (Filename.dirname file);
    let fd = Unix.openfile file [ O_WRONLY; O_CREAT ] 0o644 in
    Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
    let b = Bytes.create (8 * List.length frames) in
    List.iteri
      (fun i frame ->
        Bytes.set_int64_le b (8 * i)
          (Int64.logor (Int64.shift_left 1L 63) (Int64.of_int frame)))
      frames;
    ignore (Unix.lseek fd (Int.div a page * 8) SEEK_SET);
    ignore (Unix.write fd b 0 (Bytes.length b))

  let link root file target =
    mkdir_p (Filename.dirname (root / file));
    Unix.symlink target (root / file)
end
