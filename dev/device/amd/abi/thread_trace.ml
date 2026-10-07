(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Packet

let major (g : Gpu.t) =
  let m, _, _ = g.gc in
  m

(* Recording

   The program is Mesa's for a compute queue (src/amd/common/ac_sqtt.c and
   src/amd/vulkan/tools/radv_sqtt.c at mesa e11d8ed6: radv_begin_sqtt,
   ac_sqtt_emit_start, radv_end_sqtt, ac_sqtt_emit_stop, ac_sqtt_emit_wait,
   ac_sqtt_get_ctrl), register by register. It departs where the interface
   decides: the caches are coherent before and after (radv waits for idle), only
   engines 0 and 1 trace instructions (Mesa takes a mask), the traced work-group
   processor is the first (Mesa reads which are active), each die's engines are
   predicated, and the engine's write pointer alone is stored. *)

(* A trace buffer's size is in pages of 4096 bytes, its address from bit 12
   (SQTT_BUFFER_ALIGN_SHIFT). *)
let page = 4096
let page_shift = 12

(* The address's bits from 44, in BASE2 (GFX9), BUF0_SIZE (GFX11) or
   BUF0_BASE_HI (GFX12). *)
let high_shift = 44

(* The poll interval of the waits for the engines (ac_sqtt_emit_wait). *)
let interval = 4

(* The engines that trace instructions. *)
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

(* Selects engine [se] and its shader array 0 for the register writes after it,
   or every engine and array; every instance either way. *)
let grbm fn g ?se () =
  let array = if major g = 9 then "sh" else "sa" in
  let fields =
    match se with
    | Some se -> [ ("se_index", se); (array ^ "_index", 0) ]
    | None -> [ ("se_broadcast_writes", 1); (array ^ "_broadcast_writes", 1) ]
  in
  set fn g "GRBM_GFX_INDEX" (("instance_broadcast_writes", 1) :: fields)

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

let bits l = List.fold_left (fun m b -> m lor (1 lsl b)) 0 l

(* The SQ's thread trace events, on or off (ac_emit_cp_spi_config_cntl):
   SPI_SQG_EVENT_CTL on GFX12, SPI_CONFIG_CNTL with the SPI's priorities
   before. *)
let gpr_write_priority = 0x2c688
let exp_priority_order = 3
let ps_pkr_priority_cntl = 3

let sqg_events fn g ~on =
  let t = Bool.to_int on in
  let events = [ ("enable_sqg_top_events", t); ("enable_sqg_bop_events", t) ] in
  if major g >= 12 then set fn g "SPI_SQG_EVENT_CTL" events
  else
    set fn g "SPI_CONFIG_CNTL"
      ([
         ("gpr_write_priority", gpr_write_priority);
         ("exp_priority_order", exp_priority_order);
       ]
      @ events
      @
      if major g >= 10 then [ ("ps_pkr_priority_cntl", ps_pkr_priority_cntl) ]
      else [])

(* SQ_THREAD_TRACE_CTRL (ac_sqtt_get_ctrl). *)
let hiwater = 5
let reg_at_hwm = 2
let lowater_offset = 4

let ctrl fn g ~on =
  set fn g "SQ_THREAD_TRACE_CTRL"
    ([
       ("mode", Bool.to_int on);
       ("hiwater", hiwater);
       ("util_timer", 1);
       ("draw_event_en", 1);
       ("spi_stall_en", 1);
       ("sq_stall_en", 1);
       ("reg_at_hwm", reg_at_hwm);
     ]
    @
    if major g >= 12 then [ ("lowater_offset", lowater_offset) ]
    else [ ("rt_freq", Defs.sq_tt_rt_freq_4096_clk) ])

(* The waves GFX11 on traces: every stage it has (ac_sqtt_get_shader_mask), of
   which a compute queue runs compute waves alone. *)
let stages =
  Defs.(
    sq_tt_wtype_include_ps_bit lor sq_tt_wtype_include_gs_bit
    lor sq_tt_wtype_include_hs_bit lor sq_tt_wtype_include_cs_bit)

(* The registers GFX11 on traces, and those GFX12 excludes: CP_ME_MC_RADDR. *)
let included =
  Defs.(
    sq_tt_token_mask_sqdec_bit lor sq_tt_token_mask_shdec_bit
    lor sq_tt_token_mask_gfxudec_bit lor sq_tt_token_mask_comp_bit
    lor sq_tt_token_mask_context_bit lor sq_tt_token_mask_config_bit)

let cp_me_mc_raddr = 2

(* The tokens an engine excludes: those of instruction timing where it traces
   none, and on GFX11 the performance counters', which Mesa calls deprecated.
   GFX12's field takes GFX11's bits. *)
let excluded g e =
  let timing =
    if itraced e then 0
    else
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
  if major g >= 12 then timing
  else timing lor (1 lsl Defs.sq_tt_token_exclude_perf_shift)

let start_gfx11 fn g ~size buffer =
  let base e shift = Shift (Value (buffer e), shift) in
  let gfx12 = major g >= 12 in
  (* BUF0_SIZE before the base: "order seems important". *)
  let buffer_words e =
    if gfx12 then
      set fn g "SQ_THREAD_TRACE_BUF0_SIZE" [ ("size", size / page) ]
      @ write fn g "SQ_THREAD_TRACE_BUF0_BASE_LO" [ W32 (base e page_shift) ]
      @ write fn g "SQ_THREAD_TRACE_BUF0_BASE_HI" [ W32 (base e high_shift) ]
      @ write fn g "SQ_THREAD_TRACE_WPTR" [ Dword 0 ]
    else
      let r = register fn g "SQ_THREAD_TRACE_BUF0_SIZE" in
      let size = Int64.of_int (Register.encode r [ ("size", size / page) ]) in
      write fn g "SQ_THREAD_TRACE_BUF0_SIZE"
        [ W32 (Or (base e high_shift, size)) ]
      @ write fn g "SQ_THREAD_TRACE_BUF0_BASE" [ W32 (base e page_shift) ]
  in
  engines g (fun e se ->
      grbm fn g ~se () @ buffer_words e
      @ set fn g "SQ_THREAD_TRACE_MASK"
          [
            ("wtype_include", stages);
            ("sa_sel", 0);
            ("wgp_sel", 0);
            ("simd_sel", 0);
          ]
      @ set fn g "SQ_THREAD_TRACE_TOKEN_MASK"
          ([
             ("reg_include", included);
             ("token_exclude", excluded g e);
             ("bop_events_token_include", 1);
           ]
          @
          if gfx12 then
            [ ("exclude_barrier_wait", 1); ("reg_exclude", cp_me_mc_raddr) ]
          else [])
      (* Last: it enables the trace. *)
      @ ctrl fn g ~on:true)

(* GFX9's tokens, by SQ_THREAD_TRACE_TOKEN_* type (vega10_enum.h): every one but
   PERF (14), as Mesa's 0xbfff; on an engine that traces no instructions,
   neither INST, INST_PC nor ISSUE (10, 11, 13). *)
let gfx9_tokens = 0xbfff
let gfx9_instruction_tokens = bits [ 10; 11; 13 ]

(* GFX9's SQ_THREAD_TRACE_HIWATER, and its MODE: every stage, flushed to memory
   periodically, counted in TCC's counters. *)
let gfx9_hiwater = 4

let gfx9_mode ~on =
  [
    ("mask_ps", 1);
    ("mask_vs", 1);
    ("mask_gs", 1);
    ("mask_es", 1);
    ("mask_hs", 1);
    ("mask_ls", 1);
    ("mask_cs", 1);
    ("autoflush_en", 1);
    ("mode", Bool.to_int on);
    ("tc_perf_en", 1);
  ]

let start_gfx9 fn g ~size buffer =
  let base e shift = W32 (Shift (Value (buffer e), shift)) in
  engines g (fun e se ->
      let tokens =
        if itraced e then gfx9_tokens
        else gfx9_tokens land lnot gfx9_instruction_tokens
      in
      (* BASE2, BASE, SIZE and CTRL in this order: "order seems important". *)
      grbm fn g ~se ()
      @ write fn g "SQ_THREAD_TRACE_BASE2" [ base e high_shift ]
      @ write fn g "SQ_THREAD_TRACE_BASE" [ base e page_shift ]
      @ set fn g "SQ_THREAD_TRACE_SIZE" [ ("size", size / page) ]
      @ set fn g "SQ_THREAD_TRACE_CTRL" [ ("reset_buffer", 1) ]
      @ set fn g "SQ_THREAD_TRACE_MASK"
          [
            ("cu_sel", 0);
            ("sh_sel", 0);
            ("simd_en", 0xf);
            ("vm_id_mask", 0);
            ("reg_stall_en", 1);
            ("spi_stall_en", 1);
            ("sq_stall_en", 1);
          ]
      @ set fn g "SQ_THREAD_TRACE_TOKEN_MASK"
          [
            ("token_mask", tokens); ("reg_mask", 0xff); ("reg_drop_on_stall", 0);
          ]
      @ set fn g "SQ_THREAD_TRACE_PERF_MASK"
          [ ("sh0_mask", 0xffff); ("sh1_mask", 0xffff) ]
      @ write fn g "SQ_THREAD_TRACE_TOKEN_MASK2" [ Dword 0xffff_ffff ]
      @ set fn g "SQ_THREAD_TRACE_HIWATER" [ ("hiwater", gfx9_hiwater) ]
      @ set fn g "SQ_THREAD_TRACE_STATUS" [ ("utc_error", 0) ]
      @ set fn g "SQ_THREAD_TRACE_MODE" (gfx9_mode ~on:true))

let start (g : Gpu.t) ~size buffer =
  let fn = "Thread_trace.start" in
  if size <= 0 || size mod page <> 0 then
    invalid_arg
      (Printf.sprintf "%s: size %d, expected a positive multiple of 4096" fn
         size);
  let program =
    if major g = 9 then start_gfx9 fn g ~size buffer
    else start_gfx11 fn g ~size buffer
  in
  Pm4.acquire_mem g System @ sqg_events fn g ~on:true @ program @ grbm fn g ()
  @ set fn g "COMPUTE_THREAD_TRACE_ENABLE" [ ("thread_trace_enable", 1) ]
  @ Pm4.acquire_mem g System

let stop (g : Gpu.t) ends =
  let fn = "Thread_trace.stop" in
  let status = register fn g "SQ_THREAD_TRACE_STATUS" in
  let wptr = Register.address g (register fn g "SQ_THREAD_TRACE_WPTR") in
  (* Until the status's [field] compares to [v] as [cmp] says. *)
  let await field cmp v =
    let mask = Register.encode status [ (field, -1) ] in
    known
      (Pm4.wait g
         (Register (Register.address g status))
         cmp v ~mask ~interval ())
  in
  let finished =
    if major g = 9 then
      set fn g "SQ_THREAD_TRACE_MODE" [ ("mode", 0) ] @ await "busy" Equal 0
    else
      (* FINISH_DONE set: Mesa waits for it to differ from 0. *)
      await "finish_done" Greater_equal 1
      @ ctrl fn g ~on:false @ await "busy" Equal 0
  in
  Pm4.acquire_mem g System
  @ set fn g "COMPUTE_THREAD_TRACE_ENABLE" [ ("thread_trace_enable", 0) ]
  @ Pm4.event_write Thread_trace_finish
  @ engines g (fun e se ->
      grbm fn g ~se () @ finished
      @ Pm4.copy_data Confirmed (Counter wptr) (ends e))
  @ grbm fn g () @ sqg_events fn g ~on:false @ Pm4.acquire_mem g System

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
   unverified on hardware. *)

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
