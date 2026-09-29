(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An NVIDIA GPU the process drives over PCI: its registers, identity, memory
   and page tables. *)

module D = Nv_defs
module Mmio = Nx_device_support.Mmio
module Pci = Nx_device_support.Pci
module Page_table = Nx_device_support.Page_table
module Sysmem = Nx_device_support.Sysmem

(* Registers *)

type reg = int * int * (string * (int * int)) list

type t = {
  pci : Pci.t;
  mmio : Mmio.t; (* BAR 0: the registers *)
  vram : Mmio.t; (* BAR 1: the memory *)
  mutable regs : (string * reg) list list;
      (* the included headers, last first *)
  architecture : int;
  implementation : int;
  mmu_ver : int;
  fmc_boot : bool; (* booted by the FSP from the FMC (Blackwell) *)
  vram_size : int;
  large_bar : bool;
  mutable mm : Page_table.t option;
}

let include_regs d name arch =
  match List.find_opt (fun (n, a, _) -> n = name && a = arch) D.registers with
  | Some (_, _, regs) -> d.regs <- regs :: d.regs
  | None -> invalid_arg (Printf.sprintf "no registers %s %s" name arch)

let reg d name =
  match List.find_map (List.assoc_opt name) d.regs with
  | Some r -> r
  | None -> invalid_arg ("Nvdev: no register " ^ name)

let rreg d addr = Mmio.get32 d.mmio addr
let wreg d addr v = Mmio.set32 d.mmio addr v

let addr d ?(base = 0) ?(i = 0) name =
  let off, stride, _ = reg d name in
  base + off + (i * stride)

let encode d name fields =
  let _, _, fs = reg d name in
  List.fold_left
    (fun w (f, v) ->
      match List.assoc_opt f fs with
      | Some (lo, n) -> w lor ((v land ((1 lsl n) - 1)) lsl lo)
      | None -> invalid_arg (Printf.sprintf "Nvdev: %s has no field %s" name f))
    0 fields

let field d name f w =
  let _, _, fs = reg d name in
  match List.assoc_opt f fs with
  | Some (lo, n) -> (w lsr lo) land ((1 lsl n) - 1)
  | None -> invalid_arg (Printf.sprintf "Nvdev: %s has no field %s" name f)

let read d ?base ?i name = rreg d (addr d ?base ?i name)
let read_field d ?base ?i name f = field d name f (read d ?base ?i name)

let write d ?base ?i ?(value = 0) name fields =
  wreg d (addr d ?base ?i name) (value lor encode d name fields)

let update d ?base ?i name fields =
  let _, _, fs = reg d name in
  let mask =
    List.fold_left
      (fun m (f, _) ->
        let lo, n =
          match List.assoc_opt f fs with
          | Some b -> b
          | None ->
              invalid_arg (Printf.sprintf "Nvdev: %s has no field %s" name f)
        in
        m lor (((1 lsl n) - 1) lsl lo))
      0 fields
  in
  write d ?base ?i ~value:(read d ?base ?i name land lnot mask) name fields

(* Polls [f] until it holds, for at most [timeout_ms] (defaults to 30 seconds),
   and raises [Failure] naming [what] otherwise. *)
let wait_until ?(timeout_ms = 30_000) what f =
  let t0 = Unix.gettimeofday () in
  while not (f ()) do
    if (Unix.gettimeofday () -. t0) *. 1000. > float timeout_ms then
      failwith (Printf.sprintf "timed out waiting for %s" what);
    Domain.cpu_relax ()
  done

(* Identity *)

let chip_name d =
  let family =
    match d.architecture with
    | 0x17 -> "GA1"
    | 0x19 -> "AD1"
    | 0x1b -> "GB2"
    | a ->
        failwith
          (Printf.sprintf "NVIDIA GPUs of architecture 0x%x are not supported" a)
  in
  Printf.sprintf "%s%02d" family d.implementation

(* The directory of the GPU's firmware under linux-firmware's nvidia/. *)
let firmware_dir d =
  match String.sub (chip_name d) 0 3 with
  | "GA1" -> "ga102"
  | "AD1" -> "ad102"
  | _ -> "gb202"

let pci_command = 0x04
let pci_command_master = 0x4

let set_bus_master pci on =
  let v = Pci.read_config pci pci_command 2 in
  Pci.write_config pci pci_command 2
    (if on then v lor pci_command_master else v land lnot pci_command_master)

(* Opens the GPU at [pci]: a GPU whose WPR2 is up, left by its kernel driver or
   a process that did not finish, is reset first. *)
let create pci =
  let d =
    {
      pci;
      mmio = Pci.map_bar pci 0;
      vram = Pci.map_bar pci 1;
      regs = [];
      architecture = 0;
      implementation = 0;
      mmu_ver = 0;
      fmc_boot = false;
      vram_size = 0;
      large_bar = false;
      mm = None;
    }
  in
  include_regs d "nv_ref" "";
  include_regs d "dev_fb" "tu102";
  include_regs d "dev_gc6_island" "ga102";
  if read d "NV_PFB_PRI_MMU_WPR2_ADDR_HI" <> 0 then begin
    set_bus_master pci false;
    Pci.reset pci
  end;
  set_bus_master pci true;
  let boot42 = read d "NV_PMC_BOOT_42" in
  let architecture = field d "NV_PMC_BOOT_42" "architecture" boot42 in
  let d =
    {
      d with
      architecture;
      implementation = field d "NV_PMC_BOOT_42" "implementation" boot42;
      mmu_ver = (if architecture >= 0x1a then 3 else 2);
      fmc_boot = architecture >= 0x1a;
      vram_size = read d "NV_PGC6_AON_SECURE_SCRATCH_GROUP_42" lsl 20;
    }
  in
  ignore (chip_name d);
  { d with large_bar = Mmio.length d.vram >= d.vram_size }

let chip_id d = read d "NV_PMC_BOOT_0"

(* Page tables *)

(* NVIDIA's page directory levels: a PDE0 entry is two in one, a PTE for a 2 MiB
   page in its low 64 bits or a PDE of the small-page table in its high 64 bits.
   [get] and [set] give a PDE0 entry as the half it uses. *)
let mmu_fields ver kind = List.assoc kind (List.assoc ver D.mmu)

let levels ver =
  if ver = 3 then [ 12; 21; 29; 38; 47; 56 ] else [ 12; 21; 29; 38; 47 ]

(* The entries of the MMU version [ver], in the GPU memory [vram], [flush]
   making the GPU see them. *)
let entry ~ver ~vram ~flush =
  let n = List.length (levels ver) in
  let dual = n - 2 in
  let pte = mmu_fields ver "pte"
  and pde = mmu_fields ver "pde"
  and dpde = mmu_fields ver "dual_pde" in
  let enc fields kvs =
    List.fold_left
      (fun w (k, v) ->
        let lo, bits = List.assoc k fields in
        let lo = if lo >= 64 then lo - 64 else lo in
        Int64.logor w
          (Int64.shift_left
             (Int64.logand (Int64.of_int v)
                (Int64.pred (Int64.shift_left 1L bits)))
             lo))
      0L kvs
  in
  let dec fields k e =
    let lo, bits = List.assoc k fields in
    let lo = if lo >= 64 then lo - 64 else lo in
    Int64.to_int
      (Int64.logand
         (Int64.shift_right_logical e lo)
         (Int64.pred (Int64.shift_left 1L bits)))
  in
  let at table i level = table + if level = dual then 16 * i else 8 * i in
  let no_ats = if ver = 2 then enc dpde [ ("no_ats", 1) ] else 0L in
  let get ~level ~table i =
    let lo = Mmio.get64 vram (at table i level) in
    if level <> dual || Int64.logand lo 1L = 1L then lo
    else Mmio.get64 vram (at table i level + 8)
  in
  let valid e = Int64.logand e 1L = 1L || dec pde "aperture" e <> 0 in
  let set ~level ~table i e =
    let a = at table i level in
    if level <> dual then Mmio.set64 vram a e
    else if Int64.logand e 1L = 0L && valid e then begin
      Mmio.set64 vram a no_ats;
      Mmio.set64 vram (a + 8) e
    end
    else begin
      Mmio.set64 vram a e;
      Mmio.set64 vram (a + 8) 0L
    end
  in
  let cache = if ver = 3 then "pcf" else "vol" in
  let encode ~level ~table space ~uncached ~snooped:_ ~fragment:_ ~valid pa =
    if not table then
      enc pte
        [
          ("valid", Bool.to_int valid);
          ("address_sys", pa lsr 12);
          ("aperture", if space = Page_table.Sys then 2 else 0);
          ("kind", 6);
          (cache, Bool.to_int uncached);
        ]
    else if level = dual then
      let small = if ver = 3 then "address_small" else "address_small_sys" in
      enc dpde
        ([ ("aperture_small", Bool.to_int valid); (small, pa lsr 12) ]
        @ if ver = 3 then [ ("pcf_small", 0b10) ] else [])
    else
      let address = if ver = 3 then "address" else "address_sys" in
      enc pde
        ([
           ("is_pte", 0); ("aperture", Bool.to_int valid); (address, pa lsr 12);
         ]
        @ if ver = 3 then [ ("pcf", 0b10) ] else [ ("no_ats", 1) ])
  in
  let address e =
    (* a PDE's address field is where a PDE0's small half has it *)
    let k = if ver = 3 then "address" else "address_sys" in
    dec pde k e lsl 12
  in
  {
    Page_table.levels = levels ver;
    bits = (if ver = 3 then 56 else 48);
    first = 0;
    get;
    set;
    encode;
    valid;
    leaf = (fun ~level e -> level = n - 1 || Int64.logand e 1L = 1L);
    address;
    large = (fun ~level -> level >= n - 3);
    zero = (fun pa len -> Mmio.fill vram pa len '\000');
    flush;
  }

(* The virtual addresses of every NVIDIA GPU the process drives. *)
let space = Page_table.Space.create ~base:0x10_0000_0000 (1 lsl 44)

(* Manages the memory below [top]: the page tables, 2 MiB of boot memory, and
   the rest, in blocks of 512 MiB, 2 MiB and 4 KiB. *)
let init_mm d ~top =
  include_regs d "dev_vm" "tu102";
  let mm =
    let flush () =
      wreg d
        (addr d "NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE")
        ((1 lsl 0) lor (1 lsl 1) lor (1 lsl 6) lor (1 lsl 31))
    in
    Page_table.create ~base:0
      (entry ~ver:d.mmu_ver ~vram:d.vram ~flush)
      space ~memory:top ~boot:(2 lsl 20) ~tables:(not d.large_bar)
      ~pages:
        [ (512 lsl 20, 512 lsl 20); (2 lsl 20, 2 lsl 20); (0x1000, 0x1000) ]
  in
  Page_table.booted mm;
  d.mm <- Some mm

let mm d = Option.get d.mm

(* Boot memory *)

let round_up n a = (n + a - 1) / a * a

(* [boot_mem d n] is [n] bytes the GPU's falcons reach: the process's view of
   them, their address in the GPU's memory if there, and the bus address of each
   page. They are system memory if [sysmem] (defaults to whether the BAR is too
   small to reach the GPU's memory), and the GPU's memory otherwise. The memory
   is kept for the life of the process. *)
let boot_mem d ?sysmem ?data n =
  let sz = round_up n 0x1000 in
  let view, paddr, pages =
    if Option.value sysmem ~default:(not d.large_bar) then
      let va =
        match Page_table.Space.alloc space sz with
        | Some va -> va
        | None -> failwith "no addresses for the GPU's boot memory"
      in
      let view, pages = Sysmem.alloc ~va sz in
      (view, None, pages)
    else
      match Page_table.palloc (mm d) sz with
      | None -> failwith "no GPU memory for booting"
      | Some pa ->
          let bar = fst (Pci.bar d.pci 1) in
          ( Mmio.sub d.vram pa sz,
            Some pa,
            List.init (sz / 0x1000) (fun i -> bar + pa + (i * 0x1000)) )
  in
  Option.iter (fun s -> Mmio.write view 0 s) data;
  (view, paddr, pages)
