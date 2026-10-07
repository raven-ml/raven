(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Packet

let major (g : Gpu.t) =
  let m, _, _ = g.gc in
  m

(* Recording

   The program follows Mesa's ac_sqtt.c (ac_sqtt_emit_start, ac_sqtt_emit_stop,
   ac_sqtt_emit_wait, ac_sqtt_get_ctrl) for a compute queue, register by
   register, with the departures the comments name: the values traces were
   captured with on gfx1100, gfx1201 and gfx942. *)

(* A trace buffer's size is in pages of 4096 bytes, its address from bit 12
   (SQTT_BUFFER_ALIGN_SHIFT). *)
let page = 4096
let page_shift = 12

(* GFX9 and GFX11 hold the address's bits from 44 in a register of their own
   (SQ_THREAD_TRACE_BASE2, BUF0_SIZE.BASE_HI). *)
let high_shift = 44

(* The poll interval of the waits for the engines, as ac_sqtt_emit_wait's. *)
let interval = 4

(* The engines that trace instructions, where Mesa takes a mask. *)
let itraced e = e < 2

let register fn g name =
  match Register.find g ("reg" ^ name) with
  | Some r -> r
  | None ->
      let a, b, c = g.gc in
      invalid_arg (Printf.sprintf "%s: GC %d.%d.%d has no reg%s" fn a b c name)

(* The words of a register program: [set] writes fields, [write] words. *)
let set fn g name fields =
  let r = register fn g name in
  Pm4.set_reg (Register.address g r) [ Dword (Register.encode r fields) ]

let write fn g name ws =
  Pm4.set_reg (Register.address g (register fn g name)) ws

(* Selects the engine [se] and shader array [sa] for the register writes after
   it, or every one of them. *)
let grbm fn g ?se ?sa () =
  let field key = function
    | None -> (key ^ "_broadcast_writes", 1)
    | Some v -> (key ^ "_index", v)
  in
  set fn g "GRBM_GFX_INDEX"
    [
      field "instance" None;
      field "se" se;
      field (if major g = 9 then "sh" else "sa") sa;
    ]

(* Words for the dies of [xcc_mask], on a GPU of several. *)
let on_dies (g : Gpu.t) xcc_mask p =
  if g.xccs > 1 then Pm4.pred_exec ~xcc_mask p else p

(* The engines of every die, each with its die's mask and its number there. *)
let engines (g : Gpu.t) f =
  List.concat
    (List.init (g.shader_engines * g.xccs) (fun e ->
         on_dies g (1 lsl (e / g.shader_engines)) (f e (e mod g.shader_engines))))

(* A packet of known values, as constant words of any packet. *)
let known (p : int Packet.t) =
  let s = Packet.encode Int64.of_int p in
  List.init
    (String.length s / 4)
    (fun i ->
      Dword (Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff))

(* The SPI's configuration around a trace, as Mesa's radv_sqtt.c sets it. *)
let spi_config fn g ~tracing =
  let t = Bool.to_int tracing in
  set fn g "SPI_CONFIG_CNTL"
    [
      ("ps_pkr_priority_cntl", 3);
      ("exp_priority_order", 3);
      ("gpr_write_priority", 0x2c688);
      ("enable_sqg_bop_events", t);
      ("enable_sqg_top_events", t);
    ]

(* ac_sqtt_get_ctrl's, but for HIWATER, 1 where Mesa writes 5, and GFX12's
   LOWATER_OFFSET, which Mesa sets to 4. *)
let trace_config fn g ~tracing =
  set fn g "SQ_THREAD_TRACE_CTRL"
    ([
       ("draw_event_en", 1);
       ("spi_stall_en", 1);
       ("sq_stall_en", 1);
       ("reg_at_hwm", 2);
       ("hiwater", 1);
       ("util_timer", 1);
       ("mode", Bool.to_int tracing);
     ]
    @ if major g >= 12 then [] else [ ("rt_freq", Defs.sq_tt_rt_freq_4096_clk) ]
    )

(* GFX9's tokens, by SQ_THREAD_TRACE_TOKEN_* type (vega10_enum.h): MISC,
   TIMESTAMP, REG, WAVE_START, WAVE_END, INST_USERDATA, REG_CSPRIV and REG_CS;
   and INST, INST_PC and ISSUE on the engines that trace instructions. Mesa
   traces every token (0xbfff). *)
let gfx9_tokens = [ 0; 1; 2; 3; 6; 12; 5; 15 ]
let gfx9_instruction_tokens = [ 10; 11; 13 ]
let bits = List.fold_left (fun m b -> m lor (1 lsl b)) 0

(* The registers a GFX11 trace includes: Mesa's, but CONFIG. *)
let included =
  Defs.sq_tt_token_mask_sqdec_bit lor Defs.sq_tt_token_mask_shdec_bit
  lor Defs.sq_tt_token_mask_gfxudec_bit lor Defs.sq_tt_token_mask_comp_bit
  lor Defs.sq_tt_token_mask_context_bit

(* The tokens an engine that traces no instructions excludes: Mesa's five.
   GFX12's enumeration names few of the field's bits, so GFX11's are taken for
   them; GFX12 also sets bit 11, PERF on GFX11, which Mesa does not (an
   unverified value, plan decision 22). Mesa excludes PERF on every GFX11
   engine; here no traced engine excludes anything. *)
let instructions_excluded g =
  let gfx11 =
    bits
      Defs.
        [
          sq_tt_token_exclude_vmemexec_shift;
          sq_tt_token_exclude_aluexec_shift;
          sq_tt_token_exclude_valuinst_shift;
          sq_tt_token_exclude_immediate_shift;
          sq_tt_token_exclude_inst_shift;
        ]
  in
  if major g >= 12 then gfx11 lor (1 lsl Defs.sq_tt_token_exclude_perf_shift)
  else gfx11

(* Mesa writes BASE2 before BASE ("order seems important") and the mask per
   engine with its active compute unit, sets PERF_MASK, HIWATER, STATUS and
   every stage's MODE bit, and enables the trace with
   COMPUTE_THREAD_TRACE_ENABLE. *)
let start_gfx9 fn g ~size buffer =
  let base e shift = W32 (Shift (Value (buffer e), shift)) in
  grbm fn g ()
  @ set fn g "SQ_THREAD_TRACE_MASK"
      [
        ("simd_en", 0xf);
        ("cu_sel", 0);
        ("sq_stall_en", 1);
        ("spi_stall_en", 1);
        ("reg_stall_en", 1);
        ("vm_id_mask", 0);
      ]
  @ engines g (fun e se ->
      let tokens =
        bits gfx9_tokens
        lor if itraced e then bits gfx9_instruction_tokens else 0
      in
      grbm fn g ~se ~sa:0 ()
      @ set fn g "SQ_THREAD_TRACE_TOKEN_MASK"
          [ ("reg_mask", 0xf); ("token_mask", tokens) ]
      @ set fn g "SQ_THREAD_TRACE_TOKEN_MASK2" [ ("inst_mask", 0xffff_ffff) ]
      @ write fn g "SQ_THREAD_TRACE_BASE" [ base e page_shift ]
      @ write fn g "SQ_THREAD_TRACE_BASE2" [ base e high_shift ]
      @ set fn g "SQ_THREAD_TRACE_SIZE" [ ("size", size / page) ]
      @ set fn g "SQ_THREAD_TRACE_CTRL" [ ("reset_buffer", 1) ]
      @ set fn g "SQ_THREAD_TRACE_MODE"
          [ ("mask_cs", 1); ("autoflush_en", 1); ("mode", 1) ])

(* Mesa traces every stage and the first active work-group processor, and on
   GFX12 zeroes SQ_THREAD_TRACE_WPTR and excludes CP_ME_MC_RADDR. *)
let start_gfx11 fn g ~size buffer =
  let base e shift = Shift (Value (buffer e), shift) in
  let gfx12 = major g >= 12 in
  let buffer_words e =
    if gfx12 then
      set fn g "SQ_THREAD_TRACE_BUF0_SIZE" [ ("size", size / page) ]
      @ write fn g "SQ_THREAD_TRACE_BUF0_BASE_LO" [ W32 (base e page_shift) ]
      @ write fn g "SQ_THREAD_TRACE_BUF0_BASE_HI" [ W32 (base e high_shift) ]
    else
      let r = register fn g "SQ_THREAD_TRACE_BUF0_SIZE" in
      let size = Int64.of_int (Register.encode r [ ("size", size / page) ]) in
      write fn g "SQ_THREAD_TRACE_BUF0_SIZE"
        [ W32 (Or (base e high_shift, size)) ]
      @ write fn g "SQ_THREAD_TRACE_BUF0_BASE" [ W32 (base e page_shift) ]
  in
  spi_config fn g ~tracing:true
  @ engines g (fun e se ->
      grbm fn g ~se ~sa:0 () @ buffer_words e
      @ set fn g "SQ_THREAD_TRACE_MASK"
          [
            ("wtype_include", Defs.sq_tt_wtype_include_cs_bit);
            ("simd_sel", 0);
            ("wgp_sel", 0);
            ("sa_sel", 0);
          ]
      @ set fn g "SQ_THREAD_TRACE_TOKEN_MASK"
          ([
             ("reg_include", included);
             ("token_exclude", if itraced e then 0 else instructions_excluded g);
             ("bop_events_token_include", 1);
           ]
          @
          if gfx12 then [ ("exclude_barrier_wait", 1) ]
          else [ ("ttrace_exec", 1) ])
      @ trace_config fn g ~tracing:true)

let start (g : Gpu.t) ~size buffer =
  let fn = "Thread_trace.start" in
  if size <= 0 || size mod page <> 0 then
    invalid_arg
      (Printf.sprintf "%s: size %d, expected a positive multiple of 4096" fn
         size);
  let program, enable =
    if major g = 9 then (start_gfx9 fn g ~size buffer, [])
    else
      ( start_gfx11 fn g ~size buffer,
        write fn g "COMPUTE_THREAD_TRACE_ENABLE" [ Dword 1 ] )
  in
  Pm4.acquire_mem g System @ program @ grbm fn g () @ enable
  @ Pm4.acquire_mem g System

(* GFX11 on waits for FINISH_PENDING to clear, where Mesa waits for FINISH_DONE
   to be set; GFX9 stops by MODE alone. *)
let stop (g : Gpu.t) ends =
  let fn = "Thread_trace.stop" in
  let gfx9 = major g = 9 in
  let status = register fn g "SQ_THREAD_TRACE_STATUS" in
  let wptr = Register.address g (register fn g "SQ_THREAD_TRACE_WPTR") in
  let idle field =
    let mask = Register.encode status [ (field, -1) ] in
    known
      (Pm4.wait g
         (Register (Register.address g status))
         Equal 0 ~mask ~interval ())
  in
  let stopping =
    if gfx9 then
      set fn g "SQ_THREAD_TRACE_MODE"
        [ ("mask_cs", 1); ("autoflush_en", 1); ("mode", 0) ]
    else
      write fn g "COMPUTE_THREAD_TRACE_ENABLE" [ Dword 0 ]
      @ Pm4.event_write Thread_trace_finish
  in
  Pm4.acquire_mem g System @ grbm fn g () @ stopping
  @ engines g (fun e se ->
      grbm fn g ~se ~sa:0 ()
      @ (if gfx9 then []
         else idle "finish_pending" @ trace_config fn g ~tracing:false)
      @ idle "busy"
      @ Pm4.event_write Cs_partial_flush
      @ Pm4.copy_data Confirmed (Counter wptr) (ends e))
  @ grbm fn g ()
  @ (if gfx9 then [] else spi_config fn g ~tracing:false)
  @ Pm4.acquire_mem g System

(* An engine's write pointer counts 32-byte units in 29 bits from the trace's
   start, or from address 0 on GFX 11.0 (ac_sqtt_copy_info_regs). *)
let units w = w land 0x1fff_ffff * 32

let length (g : Gpu.t) ~buffer w =
  match g.target with 11, 0, _ -> units w - units (buffer / 32) | _ -> units w

(* Decoding *)

(* A GFX9 trace is a stream of tokens of 16-bit words, little-endian. A token's
   first word names its type in its low 4 bits, which gives its words
   (Defs.sq_thread_trace_tokens), and the SQ_THREAD_TRACE_WORD_* registers lay
   out their fields. A token advances the trace's time by its delta, in units of
   4 shader cycles; a TIMESTAMP token holds an absolute time, which sets the
   time from its second on. Neither rule is in the headers: they are how
   tinygrad's decoder reads GFX9 traces (tinygrad/renderer/amd/sqtt.py),
   unverified (plan decision 22). *)

let cycles_per_delta = 4
let token_type = (0, 3)

(* A TIMESTAMP's bits that must be clear for its time to count. *)
let timestamp_reserved = (4, 15)
let bits_of w (lo, hi) = (w lsr lo) land ((1 lsl (hi - lo + 1)) - 1)

let gfx9_iter f data =
  let n = String.length data / 2 in
  let words = Array.make 16 1 in
  List.iter (fun (t, w) -> words.(t) <- w) Defs.sq_thread_trace_tokens;
  let half i = String.get_uint16_le data (2 * i) in
  let word32 i = half i lor (half (i + 1) lsl 16) in
  let time = ref 0 and offset = ref None and i = ref 0 in
  let advance w field = time := !time + (bits_of w field * cycles_per_delta) in
  let wave w cu slot simd = (bits_of w cu, bits_of w simd, bits_of w slot) in
  while !i < n && !i + words.(bits_of (half !i) token_type) <= n do
    let w = half !i in
    let t = bits_of w token_type in
    if t = Defs.sq_thread_trace_token_timestamp then begin
      let lo = word32 !i in
      if bits_of lo timestamp_reserved = 0 then
        let abs =
          bits_of lo Defs.sq_thread_trace_word_timestamp_1_of_2__time_lo
          lor bits_of
                (word32 (!i + 2))
                Defs.sq_thread_trace_word_timestamp_2_of_2__time_hi
              lsl 16
        in
        match !offset with
        | None -> offset := Some (abs - !time)
        | Some o -> time := ((abs - o) land lnot 3) - 4
    end
    else if t = Defs.sq_thread_trace_token_misc then
      advance w Defs.sq_thread_trace_word_misc__time_delta
    else begin
      advance w Defs.sq_thread_trace_word_cmn__time_delta;
      if t = Defs.sq_thread_trace_token_wave_start then
        let cu, simd, slot =
          Defs.(
            wave w sq_thread_trace_word_wave_start__cu_id
              sq_thread_trace_word_wave_start__wave_id
              sq_thread_trace_word_wave_start__simd_id)
        in
        f (Rdna_trace.Wave_start { time = !time; cu; simd; slot })
      else if t = Defs.sq_thread_trace_token_wave_end then
        let cu, simd, slot =
          Defs.(
            wave w sq_thread_trace_word_wave__cu_id
              sq_thread_trace_word_wave__wave_id
              sq_thread_trace_word_wave__simd_id)
        in
        f (Rdna_trace.Wave_end { time = !time; cu; simd; slot })
    end;
    i := !i + words.(t)
  done

let iter (g : Gpu.t) f data =
  if major g = 9 then gfx9_iter f data else Rdna_trace.iter f data

type wave = { cu : int; simd : int; slot : int; start : int; stop : int }

let waves g data =
  let started = Hashtbl.create 64 and waves = ref [] in
  let on : Rdna_trace.event -> unit = function
    | Wave_start { time; cu; simd; slot } ->
        Hashtbl.replace started (cu, simd, slot) time
    | Wave_end { time; cu; simd; slot } -> (
        match Hashtbl.find_opt started (cu, simd, slot) with
        | Some start ->
            Hashtbl.remove started (cu, simd, slot);
            waves := { cu; simd; slot; start; stop = time } :: !waves
        | None -> ())
    | Marker _ -> ()
  in
  iter g on data;
  List.rev !waves

let markers g data =
  let markers = ref [] in
  let on : Rdna_trace.event -> unit = function
    | Marker { time; realtime } -> markers := (time, realtime) :: !markers
    | Wave_start _ | Wave_end _ -> ()
  in
  iter g on data;
  List.rev !markers

(* The realtime of shader time [t], on the line through the markers around it,
   or through the first two or the last two outside them. *)
let realtime_of markers =
  let markers = Array.of_list markers in
  let n = Array.length markers in
  fun t ->
    let rec segment i =
      if i + 2 >= n || t < fst markers.(i + 1) then i else segment (i + 1)
    in
    let i = segment 0 in
    let s0, r0 = markers.(i) and s1, r1 = markers.(i + 1) in
    r0 + ((t - s0) * (r1 - r0) / (s1 - s0))

let clock g data =
  (* Markers of one shader time give no rate. *)
  let rec distinct = function
    | (s, _) :: ((s', _) :: _ as rest) when s = s' -> distinct rest
    | m :: rest -> m :: distinct rest
    | [] -> []
  in
  match distinct (markers g data) with
  | _ :: _ :: _ as markers -> Some (realtime_of markers)
  | _ -> None
