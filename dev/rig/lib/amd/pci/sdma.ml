(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs

let strf = Printf.sprintf
let rec log2 n = if n <= 1 then 0 else 1 + log2 (n lsr 1)

type t = {
  r : Regs.t;
  v : Discovery.version; (* SDMA0's *)
  name : string; (* its micro-engine's: F32 before SDMA 7, MCU from it *)
}

let make r =
  let v = Regs.version (Regs.layout_of r) D.sdma0_hwid in
  { r; v; name = (if v < (7, 0, 0) then "F32" else "MCU") }

(* The one queue this library programs, queue 0 of engine 0: SDMA 4 names it
   regSDMA_GFX, later engines regSDMA0_QUEUE0. *)
let queue_regs s =
  match s.v with 4, _, _ -> "regSDMA_GFX" | _ -> "regSDMA0_QUEUE0"

(* SDMA 4's engines are sixteen instances of one pipe; later ones one instance
   of pipes named by number. *)
let pipes s = if s.v < (5, 0, 0) then 16 else 1
let pipe s p = if s.v < (5, 0, 0) then ("", p) else (string_of_int p, 0)

let start s =
  let r = s.r in
  for p = 0 to pipes s - 1 do
    let pipe, inst = pipe s p in
    let reg n = strf "regSDMA%s_%s" pipe n in
    if s.v >= (6, 0, 0) then begin
      Regs.update ~inst r (reg "WATCHDOG_CNTL") [ ("queue_hang_count", 100) ];
      Regs.update ~inst r (reg "UTCL1_CNTL")
        [ ("resp_mode", 3); ("redo_delay", 9) ];
      Regs.update ~inst r (reg "UTCL1_PAGE")
        ([ ("rd_l2_policy", 2); ("wr_l2_policy", 3) ]
        @ if s.name = "F32" then [ ("llc_noalloc", 1) ] else []);
      Regs.update ~inst r
        (reg (s.name ^ "_CNTL"))
        [ ("halt", 0); ((if s.name = "F32" then "th1_reset" else "reset"), 0) ]
    end;
    Regs.update ~inst r (reg "CNTL")
      (("trap_enable", 1)
      :: (if s.v <= (5, 2, 0) then [ ("utc_l1_enable", 1) ] else []))
  done;
  (* The engines' doorbells: on NBIO 7.9, four ports of each live die, each a
     range of its own; elsewhere one port for all. *)
  if Soc.nbio79 r then
    List.iter
      (fun aid ->
        List.iteri
          (fun dev (port, awid, offset, awaddr) ->
            let entry = dev + 1 + (4 * aid) in
            Regs.write r
              (strf "regDOORBELL0_CTRL_ENTRY_%d" entry)
              [
                (strf "bif_doorbell%d_range_size_entry" entry, 20);
                ( strf "bif_doorbell%d_range_offset_entry" entry,
                  (D.amdgpu_navi10_doorbell_sdma_engine0 + ((entry - 1) * 0xa))
                  * 2 );
              ];
            Soc.route ~aid ~offset ~size:4 r ~port ~awid ~awaddr)
          [
            (1, 0xe, 0xe, 0x1);
            (2, 0x8, 0x8, 0x2);
            (5, 0x9, 0x9, 0x8);
            (6, 0xa, 0xa, 0x9);
          ])
      (Discovery.aids (Regs.discovery (Regs.layout_of r)))
  else
    Soc.route r ~port:2 ~awid:0xe ~awaddr:0x3
      ~offset:(D.amdgpu_navi10_doorbell_sdma_engine0 * 2)
      ~size:4

let halt s =
  if s.v >= (6, 0, 0) then
    Regs.update s.r (strf "regSDMA0_%s_CNTL" s.name) [ ("halt", 1) ]

(* The engines take 10 ms to leave their soft reset, with no state to poll. *)
let reset_ms = 10

(* The queue is disabled whichever process programmed it: a session that died
   left it enabled for the next boot. *)
let stop s =
  let r = s.r in
  let reg = queue_regs s in
  Regs.update r (reg ^ "_RB_CNTL") [ ("rb_enable", 0) ];
  Regs.update r (reg ^ "_IB_CNTL") [ ("ib_enable", 0) ];
  Regs.update r (reg ^ "_DOORBELL") [ ("enable", 0) ];
  Regs.update r (reg ^ "_DOORBELL_OFFSET") [ ("offset", 0) ];
  (* The engine halted and held in reset, its queue's preemption cleared, before
     its soft reset, each write of which is read back, as the kernel's
     sdma_v7_0_soft_reset; the next start lets it run. *)
  if s.v >= (6, 0, 0) then begin
    Regs.update r
      (strf "regSDMA0_%s_CNTL" s.name)
      [ ("halt", 1); ((if s.name = "F32" then "th1_reset" else "reset"), 1) ];
    if Regs.has (Regs.layout_of r) "regSDMA0_QUEUE0_PREEMPT" then
      Regs.write ~value:0 r "regSDMA0_QUEUE0_PREEMPT" [];
    Regs.write r "regGRBM_SOFT_RESET" [ ("soft_reset_sdma0", 1) ];
    ignore (Regs.read r "regGRBM_SOFT_RESET");
    Regs.pause r reset_ms;
    Regs.write ~value:0 r "regGRBM_SOFT_RESET" [];
    ignore (Regs.read r "regGRBM_SOFT_RESET")
  end

(* Its doorbell is the first SDMA engine's. *)
let queue s ~ring ~bytes ~read ~write =
  let r = s.r in
  let reg = queue_regs s in
  let doorbell = D.amdgpu_navi10_doorbell_sdma_engine0 in
  let w64 n ~lo ~hi v = Regs.write64 r (reg ^ n) ~lo ~hi v in
  Regs.write ~value:1 r (reg ^ "_MINOR_PTR_UPDATE") [];
  w64 "_RB_RPTR" ~lo:"" ~hi:"_HI" 0;
  w64 "_RB_WPTR" ~lo:"" ~hi:"_HI" 0;
  w64 "_RB_BASE" ~lo:"" ~hi:"_HI" (ring lsr 8);
  w64 "_RB_RPTR_ADDR" ~lo:"_LO" ~hi:"_HI" read;
  w64 "_RB_WPTR_POLL_ADDR" ~lo:"_LO" ~hi:"_HI" write;
  Regs.update r (reg ^ "_DOORBELL_OFFSET") [ ("offset", doorbell * 2) ];
  Regs.update r (reg ^ "_DOORBELL") [ ("enable", 1) ];
  Regs.write ~value:0 r (reg ^ "_MINOR_PTR_UPDATE") [];
  Regs.write r (reg ^ "_RB_CNTL")
    ((match s.v with
       | 4, _, _ -> []
       | _ -> [ (String.lowercase_ascii s.name ^ "_wptr_poll_enable", 1) ])
    @ [
        ("rb_vmid", 0);
        ("rptr_writeback_enable", 1);
        ("rptr_writeback_timer", 4);
        ("rb_enable", 1);
        ("rb_priv", 1);
        ("rb_size", log2 (bytes / 4));
      ]);
  Regs.update r (reg ^ "_IB_CNTL") [ ("ib_enable", 1) ];
  doorbell
