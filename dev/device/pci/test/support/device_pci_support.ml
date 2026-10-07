(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* Sizes and addresses *)

let kib = 1024
let mib = 1 lsl 20
let gib = 1 lsl 30
let round_up n a = (n + a - 1) / a * a
let pp_hex ppf x = Format.fprintf ppf "0x%x" x

let hex =
  Testable.with_compare Int.compare (Testable.make ~pp:pp_hex ~equal:Int.equal)

let on_linux = Sys.file_exists "/sys/bus/pci/devices"

let contains ~sub s =
  let n = String.length sub in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = sub || at (i + 1))
  in
  at 0

let failed ?(substring = "") = function
  | Device_pci.Failed why -> contains ~sub:substring why
  | _ -> false

external now_ns : unit -> int = "device_pci_test_now_ns"

(* This machine's GPUs *)

external flock : Unix.file_descr -> bool = "device_pci_test_flock"

let gpu_lock = "DEVICE_PCI_TEST_GPU_LOCK"

let host_gpus () =
  List.filter
    (fun (id : Device_pci.Machine.id) -> id.class_ = 0x03)
    (Device_pci.Machine.functions Device_pci.Machine.this)

let with_gpu_lock f =
  match Sys.getenv_opt gpu_lock with
  | None | Some "" -> skip ~reason:(gpu_lock ^ " names no lock file") ()
  | Some file ->
      let fd = Unix.openfile file [ O_RDONLY; O_CREAT; O_CLOEXEC ] 0o644 in
      Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
      if not (flock fd) then skip ~reason:(file ^ " is held by another") ();
      f ()

(* Process memory and far machines *)

external memory : int -> int = "device_pci_test_memory"
external far : int -> int -> int = "device_pci_test_far"
external break : int -> unit = "device_pci_test_far_break"
external hold : int -> unit = "device_pci_test_far_hold"
external waiting : int -> bool = "device_pci_test_far_waiting"
external let_go : int -> unit = "device_pci_test_far_let_go"
external log : int -> (bool * int * int) list = "device_pci_test_far_log"

(* Page tables in a fake format *)

module Tables = struct
  open Device_pci

  type memory = {
    entries : (int, int64) Hashtbl.t;
    mutable zeroed : (int * int) list;
    mutable unflushed : int;
    mutable touches : int;
  }

  let memory () =
    { entries = Hashtbl.create 64; zeroed = []; unflushed = 0; touches = 0 }

  let shifts = [| 39; 30; 21; 12 |]
  let leaf = 3
  let large l = l >= 1
  let address_mask = 0xF_FFFF_FFFF_F000
  let bit b i = if b then 1 lsl i else 0
  let target_code = function Page_table.Gpu -> 0 | System -> 4 | Peer -> 8

  let target_of e =
    match e land 12 with 0 -> Page_table.Gpu | 4 -> System | _ -> Peer

  let entry_at m table i =
    Option.value ~default:0L (Hashtbl.find_opt m.entries (table + (8 * i)))

  let format m =
    let touch () = m.touches <- m.touches + 1 in
    {
      Page_table.levels = [ 12; 21; 30; 39 ];
      bits = 48;
      first = 0;
      get =
        (fun ~level:_ ~table i ->
          touch ();
          entry_at m table i);
      set =
        (fun ~level:_ ~table i e ->
          touch ();
          Hashtbl.replace m.entries (table + (8 * i)) e;
          m.unflushed <- m.unflushed + 1);
      encode =
        (fun ~level:_ ~table tg ~uncached ~snooped ~fragment ~valid pa ->
          Int64.of_int
            (pa lor bit valid 0 lor bit (not table) 1 lor target_code tg
           lor bit uncached 4 lor bit snooped 5
            lor ((fragment land 63) lsl 6)));
      valid = (fun e -> Int64.logand e 1L <> 0L);
      leaf = (fun ~level e -> level = leaf || Int64.logand e 2L <> 0L);
      address = (fun e -> Int64.to_int e land address_mask);
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
          m.unflushed <- 0);
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
      | Peer -> "peer")

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
