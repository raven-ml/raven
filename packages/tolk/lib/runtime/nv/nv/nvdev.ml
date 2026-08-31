(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Helpers = Tolk.Helpers
module Memory = Tolk.Memory
module Tlsf = Tolk.Tlsf
module Mmio = Tolk_hcq.Hcq.Mmio
module System = Tolk_hcq.System

let nv_debug = Helpers.getenv "NV_DEBUG" 0
let debug = Helpers.getenv "DEBUG" 0

(* Nv_reg: nvdev.py NVReg *)

module Nv_reg = struct
  type t = {
    name : string;
    addr : (int * Nv_reg_defs.off) option;
        (* block base and offset; [None] for bitfield-only descriptors *)
    fields : (string * (int * int)) list;
    rreg : int -> int;
    wreg : int -> int -> unit;
  }

  let make ~name ~entry ~rreg ~wreg =
    match (entry : Nv_reg_defs.entry) with
    | Const _ ->
        invalid_arg (Printf.sprintf "%s is a constant, not a register" name)
    | Reg { base; off; fields } ->
        { name; addr = Some (base, off); fields; rreg; wreg }
    | Group { fields } -> { name; addr = None; fields; rreg; wreg }

  let name t = t.name
  let fields t = t.fields

  let addr t =
    match t.addr with
    | Some (base, Fixed off) -> base + off
    | Some (_, Indexed _) ->
        invalid_arg
          (Printf.sprintf "register %s is indexed, apply with_idx first" t.name)
    | None ->
        invalid_arg
          (Printf.sprintf "%s is a bitfield descriptor with no address" t.name)

  let with_base t base =
    match t.addr with
    | Some (b, off) -> { t with addr = Some (base + b, off) }
    | None ->
        invalid_arg
          (Printf.sprintf "%s is a bitfield descriptor with no address" t.name)

  let with_idx t idx =
    match t.addr with
    | Some (b, Indexed { base; stride }) ->
        { t with addr = Some (b, Fixed (base + (stride * idx))) }
    | Some (_, Fixed _) ->
        invalid_arg (Printf.sprintf "register %s is not indexed" t.name)
    | None ->
        invalid_arg
          (Printf.sprintf "%s is a bitfield descriptor with no address" t.name)

  let field t nm =
    match List.assoc_opt nm t.fields with
    | Some r -> r
    | None ->
        invalid_arg (Printf.sprintf "register %s has no field %s" t.name nm)

  (* The wide page-table descriptor fields reach past the native int
     and cannot be shifted into one; their consumers work from the raw
     ranges in [fields] instead. *)
  let checked_range t nm (lo, hi) =
    if hi > 62 then
      invalid_arg
        (Printf.sprintf
           "field %s of %s reaches bit %d, past the native int; read its \
            range from fields"
           nm t.name hi)
    else (lo, hi)

  let encode t values =
    List.fold_left
      (fun acc (nm, v) ->
        let lo, _ = checked_range t nm (field t nm) in
        acc lor (v lsl lo))
      0 values

  let decode t v =
    List.map
      (fun (nm, r) ->
        let lo, hi = checked_range t nm r in
        (nm, (v lsr lo) land ((1 lsl (hi - lo + 1)) - 1)))
      t.fields

  let mask t names =
    List.fold_left
      (fun acc nm ->
        let lo, hi = checked_range t nm (field t nm) in
        acc lor (((1 lsl (hi - lo + 1)) - 1) lsl lo))
      0 names

  let read t = t.rreg (addr t)
  let read_bitfields t = decode t (read t)
  let write t ?(value = 0) fields = t.wreg (addr t) (value lor encode t fields)

  let update t fields =
    let m = mask t (List.map fst fields) in
    write t ~value:(read t land lnot m) fields
end

(* Nv_page_table: nvdev.py NVPageTableEntry *)

module Nv_page_table = struct
  type t = { view : Mmio.t; paddr : int; lv : int }

  let paddr pt = pt.paddr
  let lv pt = pt.lv

  (* The entry-format generation fixes the tree: five levels over a
     49-bit space for version 2, six levels over a 57-bit space for
     version 3. nvdev.py:143 *)
  let geometry mmu_ver =
    if mmu_ver = 3 then (56, [ 12; 21; 29; 38; 47; 56 ])
    else (48, [ 12; 21; 29; 38; 47 ])

  (* The descriptor field ranges run past bit 63 at the paired level,
     so encoding and extraction work on a (low, high) word pair; a
     value shifted near the top of the low word carries into the high
     one. *)
  let field d nm =
    match List.assoc_opt nm (Nv_reg.fields d) with
    | Some r -> r
    | None ->
        invalid_arg (Printf.sprintf "%s has no field %s" (Nv_reg.name d) nm)

  let encode d values =
    List.fold_left
      (fun (w0, w1) (nm, v) ->
        let lo, _ = field d nm in
        let v = Int64.of_int v in
        if lo >= 64 then (w0, Int64.logor w1 (Int64.shift_left v (lo - 64)))
        else
          ( Int64.logor w0 (Int64.shift_left v lo),
            if lo = 0 then w1
            else Int64.logor w1 (Int64.shift_right_logical v (64 - lo)) ))
      (0L, 0L) values

  let extract d nm (w0, w1) =
    let lo, hi = field d nm in
    let bits =
      if lo >= 64 then Int64.shift_right_logical w1 (lo - 64)
      else if lo = 0 then w0
      else
        Int64.logor
          (Int64.shift_right_logical w0 lo)
          (Int64.shift_left w1 (64 - lo))
    in
    Int64.to_int bits land ((1 lsl (hi - lo + 1)) - 1)

  let ops ~vram ~mmu_ver ~pte ~pde ~dual_pde () =
    let _, shifts = geometry mmu_ver in
    let level_cnt = List.length shifts in
    let pte_covers = Array.of_list (List.rev_map (fun s -> 1 lsl s) shifts) in
    (* nvdev.py:36 _is_dual_pde: the next-to-leaf level packs two
       64-bit words per logical entry. *)
    let is_dual pt = pt.lv = level_cnt - 2 in
    let words pt idx =
      if is_dual pt then
        (Mmio.read64 pt.view (16 * idx), Mmio.read64 pt.view ((16 * idx) + 8))
      else (Mmio.read64 pt.view (8 * idx), 0L)
    in
    (* nvdev.py:58 is_page *)
    let is_page pt idx =
      pt.lv >= level_cnt - 1 || Int64.logand (fst (words pt idx)) 1L = 1L
    in
    (* nvdev.py:54 read_fields: page entries decode as PTEs, table
       entries by their level's directory descriptor. *)
    let descriptor pt idx =
      if is_page pt idx then pte else if is_dual pt then dual_pde else pde
    in
    {
      Memory.make =
        (fun ~paddr ~lv ->
          { view = Mmio.view vram ~off:paddr ~size:0x1000 (); paddr; lv });
      set_entry =
        (fun pt ~idx ~paddr ?(table = false) ?(uncached = false)
             ?(aspace = Memory.Phys) ?snooped:_ ?frag:_ ~valid () ->
          (* nvdev.py:38 set_entry *)
          let w0, w1 =
            if not table then
              encode pte
                ([
                   ("valid", if valid then 1 else 0);
                   ("address_sys", paddr lsr 12);
                   ("aperture", if aspace = Memory.Sys then 2 else 0);
                   ("kind", 6);
                 ]
                @
                if mmu_ver = 3 then [ ("pcf", if uncached then 1 else 0) ]
                else [ ("vol", if uncached then 1 else 0) ])
            else
              let small = if is_dual pt then "_small" else "" in
              let sys = if mmu_ver = 3 then "" else "_sys" in
              encode
                (if is_dual pt then dual_pde else pde)
                ([
                   ("is_pte", 0);
                   ("aperture" ^ small, if valid then 1 else 0);
                   ("address" ^ small ^ sys, paddr lsr 12);
                 ]
                @
                if mmu_ver = 3 then [ ("pcf" ^ small, 0b10) ]
                else [ ("no_ats", 1) ])
          in
          if is_dual pt then begin
            Mmio.write64 pt.view (16 * idx) w0;
            Mmio.write64 pt.view ((16 * idx) + 8) w1
          end
          else Mmio.write64 pt.view (8 * idx) w0);
      entry = (fun pt idx -> fst (words pt idx));
      valid =
        (fun pt idx ->
          (* nvdev.py:61 valid *)
          let w = words pt idx in
          if is_page pt idx then extract pte "valid" w <> 0
          else
            extract
              (if is_dual pt then dual_pde else pde)
              (if is_dual pt then "aperture_small" else "aperture")
              w
            <> 0);
      address =
        (fun pt idx ->
          (* nvdev.py:65 address *)
          let small = if is_dual pt then "_small" else "" in
          let sys =
            if mmu_ver = 2 || pt.lv = level_cnt - 1 then "_sys" else ""
          in
          extract (descriptor pt idx)
            ("address" ^ small ^ sys)
            (words pt idx)
          lsl 12);
      is_page;
      supports_huge_page =
        (fun pt ~paddr ->
          pt.lv >= level_cnt - 3 && paddr mod pte_covers.(pt.lv) = 0);
      paddr = (fun pt -> pt.paddr);
      lv = (fun pt -> pt.lv);
    }
end

(* Devices: nvdev.py NVDev, without the layers that build on this core
   (firmware, falcons, GSP client). *)

type resolved = Register of Nv_reg.t | Constant of int

type t = {
  pci_dev : System.Pci_device.t option;
  devfmt : string;
  mmio : Mmio.t;
  vram : Mmio.t;
  rreg : int -> int;
  wreg : int -> int -> unit;
  regs : (string, resolved) Hashtbl.t;
  chip_id : int;
  chip_name : string;
  fw_name : string;
  mmu_ver : int;
  fmc_boot : bool;
  vram_size : int;
  large_bar : bool;
  is_booting : bool ref;
  is_err_state : bool ref;
  now_ms : unit -> int;
  bar1_base : int;
  alloc_sysmem : contiguous:bool -> int -> Mmio.t * int list;
  mm : Nv_page_table.t Memory.t;
}

(* nvdev.py:70: one virtual address space shared by every device. *)
let va_base = 0x1000000000
let va_size = 1 lsl 44
let va_allocator = lazy (Tlsf.create ~size:va_size ~base:va_base ())

external monotonic_ms : unit -> int = "caml_tolk_hcq_monotonic_ms" [@@noalloc]

(* nvdev.py:161 include *)
let include_into regs ~rreg ~wreg family arch =
  let arch = if arch = "" then "regs" else arch in
  let arches =
    match List.assoc_opt family Nv_reg_defs.families with
    | Some a -> a
    | None -> invalid_arg (Printf.sprintf "no register family %s" family)
  in
  let entries =
    match List.assoc_opt arch arches with
    | Some e -> e
    | None ->
        invalid_arg
          (Printf.sprintf "register family %s has no %s table" family arch)
  in
  List.iter
    (fun (name, entry) ->
      let r =
        match (entry : Nv_reg_defs.entry) with
        | Const v -> Constant v
        | Reg _ | Group _ -> Register (Nv_reg.make ~name ~entry ~rreg ~wreg)
      in
      Hashtbl.replace regs name r)
    entries

let find_resolved regs name =
  match Hashtbl.find_opt regs name with
  | Some r -> r
  | None -> invalid_arg (Printf.sprintf "device has no register %s" name)

let reg_in regs name =
  match find_resolved regs name with
  | Register r -> r
  | Constant _ ->
      invalid_arg (Printf.sprintf "%s is a constant, not a register" name)

(* Linux pci_regs.h: the command register and its bus-master bit. *)
let pci_command = 0x04
let pci_command_master = 0x4

(* The post-reset settle delay only needs elapsed wall time, so waiting
   on the device clock keeps it scriptable. *)
let sleep_ms now_ms ms =
  let start = now_ms () in
  while now_ms () - start < ms do
    ()
  done

let setup ~pci_dev ~devfmt ~mmio ~map_vram ~rreg ~wreg:raw_wreg ~read_config
    ~write_config_flush ~reset ~now_ms ~alloc_sysmem ~bar1_base () =
  (* nvdev.py:92 wreg *)
  let wreg addr v =
    raw_wreg addr v;
    if nv_debug >= 4 then Printf.printf "wreg: 0x%x = 0x%x\n%!" addr v
  in
  let regs = Hashtbl.create 512 in
  let include_regs family arch = include_into regs ~rreg ~wreg family arch in
  let reg name = reg_in regs name in
  (* nvdev.py:97 _early_ip_init *)
  include_regs "nv_ref" "";
  include_regs "dev_fb" "tu102";
  include_regs "dev_gc6_island" "ga102";
  (* A nonzero write-protected region means secure firmware from a
     previous boot is still resident; only a full reset returns the
     device to a bootable state. *)
  if Nv_reg.read (reg "NV_PFB_PRI_MMU_WPR2_ADDR_HI") <> 0 then begin
    write_config_flush ~offset:pci_command
      ~value:
        (read_config ~offset:pci_command ~size:2 land lnot pci_command_master)
      ~size:2;
    if debug >= 2 then
      Printf.printf "nv %s: WPR2 is up. Issuing a full reset.\n%!" devfmt;
    reset ();
    (* wait until device can respond again *)
    sleep_ms now_ms 100
  end;
  write_config_flush ~offset:pci_command
    ~value:(read_config ~offset:pci_command ~size:2 lor pci_command_master)
    ~size:2;
  let chip_id = Nv_reg.read (reg "NV_PMC_BOOT_0") in
  let chip_details = Nv_reg.read_bitfields (reg "NV_PMC_BOOT_42") in
  let architecture = List.assoc "architecture" chip_details in
  let arch_prefix, fw_name =
    match architecture with
    | 0x17 -> ("GA1", "ga102")
    | 0x19 -> ("AD1", "ad102")
    | 0x1b -> ("GB2", "gb202")
    | a ->
        failwith
          (Printf.sprintf
             "nv %s: unsupported chip, architecture 0x%x (NV_PMC_BOOT_0 0x%x)"
             devfmt a chip_id)
  in
  let chip_name =
    arch_prefix ^ Printf.sprintf "%02d" (List.assoc "implementation" chip_details)
  in
  let mmu_ver, fmc_boot =
    if architecture >= 0x1a then (3, true) else (2, false)
  in
  (* nvdev.py:123 _early_mmu_init *)
  include_regs "dev_vm" "tu102";
  include_regs "dev_mmu" (if mmu_ver = 3 then "gh100" else "tu102");
  (* nvdev.py:128: the entry descriptors of the chip's format. *)
  let ver = Printf.sprintf "NV_MMU_VER%d" mmu_ver in
  let pte_t = reg (ver ^ "_PTE")
  and pde_t = reg (ver ^ "_PDE")
  and dual_pde_t = reg (ver ^ "_DUAL_PDE") in
  let vram_size =
    Nv_reg.read (reg "NV_PGC6_AON_SECURE_SCRATCH_GROUP_42") lsl 20
  in
  let vram = map_vram () in
  let large_bar = Mmio.size vram >= vram_size in
  let va_bits, va_shifts = Nv_page_table.geometry mmu_ver in
  let is_booting = ref true in
  (* nvdev.py:146: the tail of vram is reserved for falcon structs. *)
  let mm =
    Memory.create
      ~pt_ops:
        (Nv_page_table.ops ~vram ~mmu_ver ~pte:pte_t ~pde:pde_t
           ~dual_pde:dual_pde_t ())
      ~vram_size:(vram_size - (64 lsl 20))
      ~boot_size:(2 lsl 20) ~va_bits ~va_shifts ~va_base:0
      ~palloc_ranges:
        (List.map (fun x -> (x, x)) [ 512 lsl 20; 2 lsl 20; 4 lsl 10 ])
      ~va_allocator:(Lazy.force va_allocator)
      ~is_booting:(fun () -> !is_booting)
      ~zero_vram:(fun ~paddr ~size ->
        Mmio.blit_bytes vram ~off:paddr (Bytes.make size '\000'))
      ~reserve_ptable:(not large_bar) ~dbg_name:devfmt
      ~on_range_mapped:(fun () ->
        (* nvdev.py:72 on_range_mapped: invalidate the TLBs after
           every mapping. *)
        Nv_reg.write
          (reg "NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE")
          ~value:((1 lsl 0) lor (1 lsl 1) lor (1 lsl 6) lor (1 lsl 31))
          [])
      ()
  in
  let t =
    {
      pci_dev;
      devfmt;
      mmio;
      vram;
      rreg;
      wreg;
      regs;
      chip_id;
      chip_name;
      fw_name;
      mmu_ver;
      fmc_boot;
      vram_size;
      large_bar;
      is_booting;
      is_err_state = ref false;
      now_ms;
      bar1_base;
      alloc_sysmem;
      mm;
    }
  in
  (* No booting state, the gsp client is reinited every run. *)
  t.is_booting := false;
  t

let make ?pci_dev ?read_config ?write_config_flush ?reset
    ?(now_ms = monotonic_ms) ?alloc_sysmem ?(bar1_base = 0) ~rreg ~wreg ~mmio
    ~vram ~devfmt () =
  let read_config =
    match (read_config, pci_dev) with
    | Some f, _ -> f
    | None, Some p ->
        fun ~offset ~size -> System.Pci_device.read_config p ~offset ~size
    | None, None -> fun ~offset:_ ~size:_ -> 0
  in
  let write_config_flush =
    match (write_config_flush, pci_dev) with
    | Some f, _ -> f
    | None, Some p ->
        fun ~offset ~value ~size ->
          System.Pci_device.write_config_flush p ~offset ~value ~size
    | None, None -> fun ~offset:_ ~value:_ ~size:_ -> ()
  in
  let reset =
    match (reset, pci_dev) with
    | Some f, _ -> f
    | None, Some p -> fun () -> System.Pci_device.reset p
    | None, None -> fun () -> ()
  in
  let alloc_sysmem =
    match alloc_sysmem with
    | Some f -> f
    | None ->
        fun ~contiguous size ->
          System.Pci_device.alloc_sysmem ~contiguous size
  in
  setup ~pci_dev ~devfmt ~mmio
    ~map_vram:(fun () -> vram)
    ~rreg ~wreg ~read_config ~write_config_flush ~reset ~now_ms ~alloc_sysmem
    ~bar1_base ()

let create pci_dev =
  let devfmt = System.Pci_device.pcibus pci_dev in
  let mmio = System.Pci_device.map_bar pci_dev 0 in
  setup ~pci_dev:(Some pci_dev) ~devfmt ~mmio
    ~map_vram:(fun () -> System.Pci_device.map_bar pci_dev 1)
    ~rreg:(fun addr -> Int32.to_int (Mmio.read32 mmio addr) land 0xffffffff)
    ~wreg:(fun addr v -> Mmio.write32 mmio addr (Int32.of_int v))
    ~read_config:(fun ~offset ~size ->
      System.Pci_device.read_config pci_dev ~offset ~size)
    ~write_config_flush:(fun ~offset ~value ~size ->
      System.Pci_device.write_config_flush pci_dev ~offset ~value ~size)
    ~reset:(fun () -> System.Pci_device.reset pci_dev)
    ~now_ms:monotonic_ms
    ~alloc_sysmem:(fun ~contiguous size ->
      System.Pci_device.alloc_sysmem ~contiguous size)
    ~bar1_base:(fst (System.Pci_device.bar_info pci_dev 1))
    ()

let pci_dev t = t.pci_dev
let devfmt t = t.devfmt
let mmio t = t.mmio
let vram t = t.vram
let vram_size t = t.vram_size
let large_bar t = t.large_bar
let chip_id t = t.chip_id
let chip_name t = t.chip_name
let fw_name t = t.fw_name
let mmu_ver t = t.mmu_ver
let fmc_boot t = t.fmc_boot
let is_booting t = !(t.is_booting)
let set_is_booting t v = t.is_booting := v
let mm t = t.mm
let is_err_state t = !(t.is_err_state)
let set_err_state t v = t.is_err_state := v
let now_ms t = t.now_ms ()
let rreg t addr = t.rreg addr
let wreg t addr v = t.wreg addr v
let reg t name = reg_in t.regs name

let const t name =
  match find_resolved t.regs name with
  | Constant v -> v
  | Register _ ->
      invalid_arg (Printf.sprintf "%s is a register, not a constant" name)

let include_regs t ~family ~arch =
  include_into t.regs ~rreg:t.rreg ~wreg:t.wreg family arch

(* nvdev.py:149 _alloc_boot_mem *)
let alloc_boot_mem t ?data ?(contiguous = false) ?sysmem size =
  let sz = (size + 0xfff) land lnot 0xfff in
  let sysmem = match sysmem with Some b -> b | None -> not t.large_bar in
  let view, paddr, sysaddr =
    if sysmem then
      let view, sysaddr = t.alloc_sysmem ~contiguous size in
      (view, None, sysaddr)
    else
      let paddr = Memory.palloc t.mm sz () in
      let view = Mmio.view t.vram ~off:paddr ~size:sz () in
      let sysaddr =
        List.init (sz / 0x1000) (fun i -> t.bar1_base + paddr + (i * 0x1000))
      in
      (view, Some paddr, sysaddr)
  in
  (match data with Some d -> Mmio.blit_bytes view ~off:0 d | None -> ());
  (view, paddr, sysaddr)
