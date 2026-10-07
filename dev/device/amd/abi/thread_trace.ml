(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Packet

let major (g : Gpu.t) =
  let m, _, _ = g.gc in
  m

(* Recording *)

(* A trace buffer's size is in pages of 4096 bytes, its address from bit 12. *)
let page = 4096
let page_shift = 12

(* GFX9 and GFX11 hold the address's bits from 44 in a register of their own. *)
let high_shift = 44

(* The poll interval of the waits for the engines. *)
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

(* The SPI's and the trace's configuration, as Mesa's ac_sqtt.c sets them. *)
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

(* GFX9's tokens: misc, time, registers, wave starts and ends, user data and
   compute registers; and the instructions, on the engines that trace them. *)
let gfx9_tokens = [ 0; 1; 2; 3; 6; 12; 5; 15 ]
let gfx9_instruction_tokens = [ 10; 11; 13 ]
let bits = List.fold_left (fun m b -> m lor (1 lsl b)) 0

(* The registers a GFX11 trace includes. *)
let included =
  Defs.sq_tt_token_mask_sqdec_bit lor Defs.sq_tt_token_mask_shdec_bit
  lor Defs.sq_tt_token_mask_gfxudec_bit lor Defs.sq_tt_token_mask_comp_bit
  lor Defs.sq_tt_token_mask_context_bit

(* The tokens an engine that traces no instructions excludes. GFX12 lays the
   field out as GFX11, whose enumeration names its bits, and excludes
   performance counter tokens too. *)
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

(* An engine's write pointer counts 32-byte units from the trace's start, or
   from address 0 on GFX 11.0. *)
let units w = w land 0x1fff_ffff * 32

let length (g : Gpu.t) ~buffer w =
  match g.target with 11, 0, _ -> units w - units (buffer / 32) | _ -> units w

(* Decoding *)

(* A trace is a stream of packets of 1 to 18 nibbles, least significant nibble
   first. A packet's first byte names its type, whose size gives where the next
   packet starts; most packets advance the trace's time, in shader cycles, by a
   delta field. An RDNA trace starts with a layout header that names the format:
   layout 3 (RDNA3), or layout 4 (RDNA4). A CDNA trace has none. *)

type field = int * int (* lowest, highest bit *)

(* The compute unit, SIMD and slot of a wave. A compute unit of RDNA is its
   work-group processor and the shader array above them. *)
type wave_fields = {
  cu : field;
  array : (field * int) option;
  simd : field;
  slot : field;
}

type kind =
  | Plain
  | Short (* a delta of 4 more than its field *)
  | Mark of { rt : int; pl : int }
    (* a realtime marker when [rt] is set and [pl] clear, a delta otherwise *)
  | Timestamp (* CDNA's absolute time *)
  | Layout
  | Start of wave_fields
  | End of wave_fields

(* A packet type: its number, as AMD's decoder numbers them; the bits of the
   first byte that name it and their value; its nibbles; its delta field; and
   its kind. *)
type packet = int * int * int * int * field option * kind

let rdna3 : packet list =
  [
    (1, 0x7, 0x3, 3, Some (3, 5), Plain);
    (2, 0xf, 0xf, 2, Some (4, 5), Plain);
    (3, 0xf, 0xe, 2, Some (4, 5), Plain);
    (4, 0xf, 0xd, 3, Some (4, 6), Plain);
    (5, 0x1f, 0x4, 6, Some (5, 7), Plain);
    (6, 0x1f, 0x14, 6, Some (5, 7), Plain);
    (7, 0x7f, 0x21, 18, Some (8, 10), Plain);
    ( 8,
      0x1f,
      0x15,
      5,
      Some (5, 7),
      End
        {
          cu = (11, 13);
          array = Some ((8, 8), 3);
          simd = (9, 10);
          slot = (15, 19);
        } );
    ( 9,
      0x1f,
      0xc,
      8,
      Some (5, 6),
      Start
        {
          cu = (10, 12);
          array = Some ((7, 7), 3);
          simd = (8, 9);
          slot = (13, 17);
        } );
    (10, 0x1f, 0x1c, 12, Some (5, 6), Plain);
    (11, 0x1f, 0x5, 5, Some (5, 7), Plain);
    (12, 0x1f, 0x6, 13, Some (5, 7), Plain);
    (13, 0x1f, 0x16, 7, Some (5, 7), Plain);
    (14, 0x7f, 0x31, 12, Some (7, 8), Plain);
    (15, 0xf, 0x8, 2, Some (4, 7), Short);
    (16, 0xf, 0x0, 1, None, Plain);
    (17, 0x7f, 0x51, 6, Some (7, 15), Plain);
    (18, 0xff, 0x61, 6, Some (8, 10), Plain);
    (19, 0xff, 0xe1, 8, Some (8, 10), Plain);
    (20, 0xf, 0x9, 16, Some (4, 6), Plain);
    (21, 0x7f, 0x71, 16, Some (7, 9), Plain);
    (22, 0x7f, 0x1, 12, Some (12, 47), Mark { rt = 9; pl = 8 });
    (23, 0x7f, 0x11, 16, None, Layout);
    (24, 0x7, 0x2, 5, Some (4, 6), Plain);
  ]

(* RDNA4 widens the work-group processor and moves fields of eight types. *)
let rdna4 : packet list =
  [
    (1, 0x7, 0x3, 3, Some (3, 5), Plain);
    (2, 0xf, 0xf, 2, Some (4, 5), Plain);
    (3, 0xf, 0xe, 2, Some (4, 5), Plain);
    (4, 0xf, 0xd, 3, Some (4, 6), Plain);
    (5, 0x1f, 0x4, 6, Some (5, 7), Plain);
    (6, 0x1f, 0x14, 6, Some (5, 7), Plain);
    (7, 0x7f, 0x21, 18, Some (8, 10), Plain);
    ( 8,
      0x1f,
      0x15,
      5,
      Some (5, 7),
      End
        {
          cu = (11, 14);
          array = Some ((8, 8), 4);
          simd = (9, 10);
          slot = (15, 19);
        } );
    ( 9,
      0x1f,
      0xc,
      8,
      Some (5, 6),
      Start
        {
          cu = (10, 13);
          array = Some ((7, 7), 4);
          simd = (8, 9);
          slot = (15, 19);
        } );
    (10, 0x1f, 0x1c, 10, Some (5, 6), Plain);
    (11, 0x1f, 0x5, 6, Some (5, 7), Plain);
    (12, 0x1f, 0x6, 14, Some (7, 9), Plain);
    (13, 0x1f, 0x16, 8, Some (7, 9), Plain);
    (14, 0x7f, 0x31, 12, Some (7, 8), Plain);
    (15, 0xf, 0x8, 2, Some (4, 7), Short);
    (16, 0xf, 0x0, 1, None, Plain);
    (17, 0x7f, 0x51, 6, Some (7, 15), Plain);
    (18, 0xff, 0x61, 6, Some (8, 10), Plain);
    (19, 0xff, 0xe1, 8, Some (8, 10), Plain);
    (20, 0xf, 0x9, 16, Some (4, 6), Plain);
    (21, 0x7f, 0x71, 16, Some (7, 9), Plain);
    (22, 0x7f, 0x1, 16, Some (12, 63), Mark { rt = 7; pl = 8 });
    (23, 0x7f, 0x11, 16, None, Layout);
    (24, 0x7, 0x2, 5, Some (3, 5), Plain);
  ]

(* CDNA's deltas count 4 cycles, from bit 4 of every packet. *)
let cdna : packet list =
  [
    (0, 0xf, 0x0, 4, Some (4, 11), Plain);
    (1, 0xf, 0x1, 16, Some (4, 4), Timestamp);
    (2, 0xf, 0x2, 16, Some (4, 4), Plain);
    ( 3,
      0xf,
      0x3,
      8,
      Some (4, 4),
      Start { cu = (6, 9); array = None; simd = (14, 15); slot = (10, 13) } );
    (4, 0xf, 0x4, 4, Some (4, 4), Plain);
    (5, 0xf, 0x5, 12, Some (4, 4), Plain);
    ( 6,
      0xf,
      0x6,
      4,
      Some (4, 4),
      End { cu = (6, 9); array = None; simd = (14, 15); slot = (10, 13) } );
    (7, 0xf, 0x7, 4, Some (4, 4), Plain);
    (8, 0xf, 0x8, 4, Some (4, 4), Plain);
    (9, 0xf, 0x9, 4, Some (4, 4), Plain);
    (10, 0xf, 0xa, 4, Some (4, 4), Plain);
    (11, 0xf, 0xb, 16, Some (4, 4), Plain);
    (12, 0xf, 0xc, 12, Some (4, 4), Plain);
    (13, 0xf, 0xd, 8, Some (4, 4), Plain);
    (14, 0xf, 0xe, 16, Some (4, 4), Plain);
    (15, 0xf, 0xf, 12, Some (4, 4), Plain);
    (16, 0x7f, 0x11, 16, None, Layout);
  ]

(* A format: the packet type of each first byte, and the cycles a delta
   counts. *)
type format = { types : packet array; scale : int }

let popcount n =
  let rec go n c = if n = 0 then c else go (n land (n - 1)) (c + 1) in
  go n 0

(* A byte names the type of the most bits that match it; among types of as many
   bits, the first listed, type 16 after the others. Type 16 names the bytes no
   other type does. *)
let format ~scale packets =
  let rank (id, mask, _, _, _, _) = (-popcount mask, id = 16) in
  let ordered =
    List.stable_sort (fun a b -> compare (rank a) (rank b)) packets
  in
  let default = List.find (fun (id, _, _, _, _, _) -> id = 16) packets in
  let matches b (_, mask, value, _, _, _) = b land mask = value in
  let of_byte b = Option.value ~default (List.find_opt (matches b) ordered) in
  { types = Array.init 256 of_byte; scale }

let field reg (lo, hi) =
  Int64.(
    to_int
      (logand
         (shift_right_logical reg lo)
         (sub (shift_left 1L (hi - lo + 1)) 1L)))

type event =
  | Marker of { time : int; realtime : int }
  | Wave_start of { time : int; cu : int; simd : int; slot : int }
  | Wave_end of { time : int; cu : int; simd : int; slot : int }

let wave_of reg w =
  let cu =
    match w.array with
    | None -> field reg w.cu
    | Some (array, shift) -> field reg w.cu lor (field reg array lsl shift)
  in
  (cu, field reg w.simd, field reg w.slot)

(* Calls [f] on the markers and the starts and ends of waves of [data], in
   order. The reader holds the next 16 nibbles in [reg], the current packet's
   from bit 0, and shifts in as many as the packet before it took. *)
let iter (g : Gpu.t) f data =
  let n = String.length data in
  let byte i = Char.code (String.unsafe_get data i) in
  let cdna = lazy (format ~scale:4 cdna) in
  let fmt =
    ref (if major g = 9 then Lazy.force cdna else format ~scale:1 rdna3)
  in
  let reg = ref 0L and pos = ref 0 and nib_off = ref 0 and nibbles = ref 16 in
  let time = ref 0 and ts_offset = ref None in
  while !pos + ((!nibbles + !nib_off + 1) lsr 1) <= n do
    let need = !nibbles - !nib_off in
    if !nib_off = 1 then begin
      (reg :=
         Int64.(
           logor
             (shift_right_logical !reg 4)
             (shift_left (of_int (byte !pos lsr 4)) 60)));
      incr pos
    end;
    let bytes = need lsr 1 in
    if bytes > 0 then begin
      let k = Int.min bytes 8 in
      let chunk = ref 0L in
      for i = k - 1 downto 0 do
        chunk := Int64.(logor (shift_left !chunk 8) (of_int (byte (!pos + i))))
      done;
      let kept = if k = 8 then 0L else Int64.shift_right_logical !reg (8 * k) in
      (reg := Int64.(logor kept (shift_left !chunk (64 - (8 * k)))));
      pos := !pos + bytes
    end;
    nib_off := need land 1;
    (if !nib_off = 1 then
       reg :=
         Int64.(
           logor
             (shift_right_logical !reg 4)
             (shift_left (of_int (byte !pos land 0xf)) 60)));
    let _, _, _, size, delta, kind = !fmt.types.(Int64.to_int !reg land 0xff) in
    nibbles := size;
    let delta = Option.fold ~none:0 ~some:(field !reg) delta in
    let marker =
      match kind with
      | Mark { rt; pl } -> field !reg (rt, rt) = 1 && field !reg (pl, pl) = 0
      | Plain | Short | Timestamp | Layout | Start _ | End _ -> false
    in
    (match kind with
    | Mark _ when marker -> ()
    | Short -> time := !time + delta + 4
    | Timestamp -> (
        if field !reg (4, 15) = 0 then
          let abs = field !reg (16, 63) in
          match !ts_offset with
          | None -> ts_offset := Some (abs - !time)
          | Some o -> time := ((abs - o) land lnot 3) - 4)
    | Plain | Mark _ | Layout | Start _ | End _ ->
        time := !time + (delta * !fmt.scale));
    match kind with
    | Layout -> (
        match field !reg (7, 12) with
        | 3 -> ()
        | 4 -> fmt := format ~scale:1 rdna4
        | _ ->
            (* Not a layout header: the trace is CDNA's, and this packet one of
               its own. *)
            fmt := Lazy.force cdna;
            let _, _, _, size, _, kind =
              !fmt.types.(Int64.to_int !reg land 0xff)
            in
            nibbles := size;
            if kind = Timestamp && field !reg (4, 15) = 0 then
              ts_offset := Some (field !reg (16, 63) - !time))
    | Mark _ when marker -> f (Marker { time = !time; realtime = delta })
    | Start w ->
        let cu, simd, slot = wave_of !reg w in
        f (Wave_start { time = !time; cu; simd; slot })
    | End w ->
        let cu, simd, slot = wave_of !reg w in
        f (Wave_end { time = !time; cu; simd; slot })
    | Plain | Short | Mark _ | Timestamp -> ()
  done

type wave = { cu : int; simd : int; slot : int; start : int; stop : int }

let waves g data =
  let started = Hashtbl.create 64 and waves = ref [] in
  let on = function
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
  let on = function
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
