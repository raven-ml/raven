(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs
module Window = Rig_pci.Window

let strf = Printf.sprintf

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
  (* The write pointer's copy starts at zero, as the ring is empty: the memory
     holds whatever the last user left. *)
  Window.set32 t.wptr 0 0;
  Window.flush t.wptr;
  Gmc.flush_hdp t.gmc;
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
        if ring.first then
          [
            ("wptr_overflow_enable", 1);
            ("rptr_rearm", 1);
            ("wptr_writeback_enable", 1);
          ]
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
  (* The rings fill without signalling interrupts: this library polls the first
     ring's write-pointer copy, and a function with no handler must not raise a
     line interrupt, which bus mastering does not hold back. *)
  List.iter
    (fun ring ->
      Regs.update r ("regIH_RB_CNTL" ^ ring.suffix) [ ("rb_enable", 1) ])
    t.rings;
  t.rptr <- 0

let ring_controls = [ "regIH_RB_CNTL"; "regIH_RB_CNTL_RING1" ]

let running r =
  List.exists (fun c -> Regs.field r c "rb_enable" = 1) ring_controls

let halt r =
  List.iter (fun c -> Regs.update r c [ ("rb_enable", 0) ]) ring_controls

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
      go
        (match Rig_amd_abi.Ih.decode ~gc ~sdma e with
        | Some rep -> rep :: acc
        | None -> acc)
  in
  let reports = go [] in
  Regs.write ~value:t.rptr t.r "regIH_RB_RPTR" [];
  reports
