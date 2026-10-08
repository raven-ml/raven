(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs
module Window = Rig_pci.Window

let strf = Printf.sprintf

type report = Page_fault | Fault of string

(* Decoding *)

(* The bits [lo] to [hi] of [v]. *)
let bits v lo hi = (v lsr lo) land ((1 lsl (hi - lo + 1)) - 1)

(* The sources whose interrupts report an error, as the kernel driver treats
   them; the others wake a sleep and report nothing. *)
let errors =
  [
    "CP_BAD_OPCODE_ERROR";
    "CP_ECC_ERROR";
    "CP_FUE_ERROR";
    "CP_GDS_ALLOC_ERROR";
    "CP_GPF";
    "CP_PM4_PKT_RSVD_BIT_ERROR";
    "CP_PRIV_INSTR_FAULT";
    "CP_PRIV_REG_FAULT";
    "CP_WAIT_MEM_SEM_FAULT";
    "GRBM_RD_TIMEOUT_ERROR";
    "RLC_GC_FED_INTERRUPT";
    "SDMA_ATOMIC_TIMEOUT";
    "SDMA_DOORBELL_INVALID";
    "SDMA_ECC";
    "SDMA_FROZEN";
    "SDMA_PAGE_FAULT";
    "SDMA_PAGE_NULL";
    "SDMA_PAGE_TIMEOUT";
    "SDMA_POLL_TIMEOUT";
    "SDMA_QUEUE_HANG";
    "SDMA_SEM_INCOMPLETE_TIMEOUT";
    "SDMA_SEM_WAIT_FAIL_TIMEOUT";
    "SDMA_SRAM_ECC";
    "SDMA_VM_HOLE";
    "SDMA_XNACK";
    "UTCL2_DATA_POISONING";
  ]

(* A shader's interrupt of encoding 2 reports an error: its type indexes
   these. *)
let shader_errors = [| "EDC_FUE"; "ILLEGAL_INST"; "MEMVIOL"; "EDC_FED" |]
let shader_error_encoding = 2
let major (m, _, _) = m

(* The block whose sources an interrupt of [client] names, and the client's
   name. SOC21 GPUs (GC 11 on) report GC's and SDMA's through two clients;
   SOC15's through GC's, its shader engines' and each SDMA engine's. *)
let block ~gc ~sdma client =
  if major gc >= 11 then
    let block =
      if
        client = D.soc21_ih_clientid_grbm_cp || client = D.soc21_ih_clientid_gfx
      then Some (strf "GFX_%d_" (major gc))
      else None
    in
    (block, D.soc21_client_name client)
  else
    let gfx =
      D.
        [
          soc15_ih_clientid_grbm_cp;
          soc15_ih_clientid_se0sh;
          soc15_ih_clientid_se1sh;
          soc15_ih_clientid_se2sh;
          soc15_ih_clientid_se3sh;
        ]
    in
    let sdmas =
      D.
        [
          soc15_ih_clientid_sdma0;
          soc15_ih_clientid_sdma1;
          soc15_ih_clientid_sdma2;
          soc15_ih_clientid_sdma3;
          soc15_ih_clientid_sdma4;
          soc15_ih_clientid_sdma5;
          soc15_ih_clientid_sdma6;
          soc15_ih_clientid_sdma7;
        ]
    in
    let block =
      if List.mem client gfx then Some "GFX_9_"
      else if List.mem client sdmas then Some (strf "SDMA0_%d_" (major sdma))
      else None
    in
    (block, D.soc15_client_name client)

let source block id =
  List.find_map
    (fun (b, i, name) ->
      if i = id && String.starts_with ~prefix:block b then Some name else None)
    D.ih_sources

let decode ~gc ~sdma e =
  if Array.length e <> 8 then
    invalid_arg
      (strf "Ih.decode: an entry of %d words, expected 8" (Array.length e));
  let client = bits e.(0) 0 7 and id = bits e.(0) 8 15 in
  let block, client_name = block ~gc ~sdma client in
  let name =
    Option.value ~default:"" (Option.bind block (fun b -> source b id))
  in
  let line () =
    strf
      "interrupt client=%s src=%s(%d) ring=%d vmid=%d(%d) pasid=%d node=%d \
       ctx=[0x%x, 0x%x, 0x%x, 0x%x]"
      (if client_name = "" then string_of_int client else client_name)
      name id
      (bits e.(0) 16 23)
      (bits e.(0) 24 27)
      (bits e.(0) 31 31)
      (bits e.(3) 0 15)
      (bits e.(3) 16 23)
      e.(4) e.(5) e.(6) e.(7)
  in
  let soc21 = major gc >= 11 in
  if name = "UTCL2_FAULT" || ((not soc21) && client = D.soc15_ih_clientid_utcl2)
  then Some Page_fault
  else if name = "SQ_INTERRUPT_ID" then
    let encoding = if soc21 then bits e.(5) 6 7 else bits e.(4) 26 27 in
    if encoding <> shader_error_encoding then None
    else
      (* The kernel's SQ_INTERRUPT_WORD_WAVE: its error type in bits 21-24 of
         the first context word on SOC21, in bits 4-7 of the second on SOC15. *)
      let kind = if soc21 then bits e.(4) 21 24 else bits e.(5) 4 7 in
      let kind =
        if kind < Array.length shader_errors then shader_errors.(kind)
        else string_of_int kind
      in
      Some (Fault (strf "%s: shader error %s" (line ()) kind))
  else if List.mem name errors then Some (Fault (line ()))
  else None

(* Rings *)

let bytes = 256 lsl 10
let entry_bytes = 32

(* A ring's size field is the log2 of its 32-bit words. *)
let rb_size =
  let rec log2 n = if n <= 1 then 0 else 1 + log2 (n lsr 1) in
  log2 (bytes / 4)

(* The write pointer's overflow bit, in the register and in its copy. *)
let overflow_bit = 1

type ring = { at : int; suffix : string; first : bool }

type t = {
  r : Regs.t;
  gmc : Gmc.t;
  view : Window.t; (* the first ring *)
  wptr_at : int; (* the physical address of its write pointer's copy *)
  wptr : Window.t;
  rings : ring list;
  mutable rptr : int; (* bytes into the first ring *)
}

let make r gmc vram ~rings:(ring0, ring1) ~wptr =
  {
    r;
    gmc;
    view = Window.sub vram ring0 bytes;
    wptr_at = wptr;
    wptr = Window.sub vram wptr 4;
    rings =
      [
        { at = ring0; suffix = ""; first = true };
        { at = ring1; suffix = "_RING1"; first = false };
      ];
    rptr = 0;
  }

(* The interrupt storm and flood controls IH 4.4.2 lacks. *)
let storm_control = (4, 4, 2)

let start t =
  let r = t.r in
  List.iter
    (fun ring ->
      let reg s = s ^ ring.suffix in
      Regs.write64 r "regIH_RB_BASE" ~lo:ring.suffix ~hi:("_HI" ^ ring.suffix)
        (Gmc.mc t.gmc ring.at lsr 8);
      Regs.write r (reg "regIH_RB_CNTL")
        ([
           ("mc_space", 4);
           ("wptr_overflow_clear", 1);
           ("rb_size", rb_size);
           ("mc_snoop", 1);
           ("mc_ro", 0);
           ("mc_vmid", 0);
         ]
        @
        if ring.first then [ ("wptr_overflow_enable", 1); ("rptr_rearm", 1) ]
        else [ ("rb_full_drain_enable", 1) ]);
      if ring.first then
        Regs.write64 r "regIH_RB_WPTR_ADDR" ~lo:"_LO" ~hi:"_HI"
          (Gmc.mc t.gmc t.wptr_at);
      Regs.write ~value:0 r (reg "regIH_RB_WPTR") [];
      Regs.write ~value:0 r (reg "regIH_RB_RPTR") [];
      Regs.write r (reg "regIH_DOORBELL_RPTR") [ ("enable", 0) ])
    t.rings;
  if Regs.version (Regs.layout_of r) D.osssys_hwid <> storm_control then begin
    Regs.update r "regIH_STORM_CLIENT_LIST_CNTL"
      [ ("client18_is_storm_client", 1) ];
    Regs.update r "regIH_INT_FLOOD_CNTL" [ ("flood_cntl_enable", 1) ];
    Regs.update r "regIH_MSI_STORM_CTRL" [ ("delay", 3) ]
  end;
  List.iter
    (fun ring ->
      Regs.update r
        ("regIH_RB_CNTL" ^ ring.suffix)
        (("rb_enable", 1) :: (if ring.first then [ ("enable_intr", 1) ] else [])))
    t.rings;
  t.rptr <- 0

let offset w = w land (bytes - 1) land lnot (entry_bytes - 1)

(* The ring's write pointer, as the kernel's [ih_v6_0_get_wptr] reads it: its
   copy, or after an overflow the register, the overflow then cleared and
   reading resumed at the oldest entry not overwritten. A failed function's
   pointer reads all ones, so its failure is checked before the pointer is
   trusted. *)
let wptr t =
  let w = Window.get32 t.wptr 0 in
  (match Rig_pci.Function.failed (Regs.fn t.r) with
  | Some why -> raise (Regs.Stuck (strf "reading the interrupt ring: %s" why))
  | None -> ());
  if w land overflow_bit = 0 then offset w
  else
    let w = Regs.read t.r "regIH_RB_WPTR" in
    if w land overflow_bit = 0 then offset w
    else begin
      Regs.update t.r "regIH_RB_WPTR" [ ("rb_overflow", 0) ];
      t.rptr <- offset (w + entry_bytes);
      Regs.update t.r "regIH_RB_CNTL" [ ("wptr_overflow_clear", 1) ];
      Regs.update t.r "regIH_RB_CNTL" [ ("wptr_overflow_clear", 0) ];
      offset w
    end

let pending t = offset (Window.get32 t.wptr 0) <> t.rptr

let read t =
  let l = Regs.layout_of t.r in
  let gc = Regs.version l D.gc_hwid and sdma = Regs.version l D.sdma0_hwid in
  let w = wptr t in
  let rec go acc =
    if t.rptr = w then List.rev acc
    else
      let e = Array.init 8 (fun i -> Window.get32 t.view (t.rptr + (4 * i))) in
      t.rptr <- (t.rptr + entry_bytes) land (bytes - 1);
      go (match decode ~gc ~sdma e with Some rep -> rep :: acc | None -> acc)
  in
  let reports = go [] in
  Regs.write ~value:t.rptr t.r "regIH_RB_RPTR" [];
  reports
