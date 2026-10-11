(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* The packet tables follow tinygrad/renderer/amd/sqtt.py and AMD's MIT-licensed
   rocprof-trace-decoder at 1753e7a4, source/gfx{10,11,12}/gfx*token.cpp.

   A trace is a stream of packets of 1 to 18 nibbles, least significant nibble
   first. A packet's first byte names its type, whose size gives where the next
   packet starts; most packets advance the trace's time, in shader cycles, by a
   delta field. A trace starts with a layout header that names its format:
   layout 3 (RDNA3), or layout 4 (RDNA4). *)

type event =
  | Marker of { time : int; realtime : int }
  | Wave_start of { time : int; cu : int; simd : int; slot : int }
  | Wave_end of { time : int; cu : int; simd : int; slot : int }

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
  | Nop
  | Plain
  | Short (* a delta of 4 more than its field *)
  | Mark of { rt : int; pl : int }
    (* a realtime marker when [rt] is set and [pl] clear, a delta otherwise *)
  | Layout
  | Start of wave_fields
  | Start_ext of wave_fields (* RDNA4 start followed by continuation bytes *)
  | End of wave_fields

(* A packet type: the bits of the first byte that name it and their value; its
   nibbles; its delta field; and its kind. *)
type packet = int * int * int * field option * kind

let rdna3 : packet list =
  let start =
    { cu = (10, 12); array = Some ((7, 7), 3); simd = (8, 9); slot = (13, 17) }
  in
  [
    (0x7, 0x3, 3, Some (3, 5), Plain);
    (0xf, 0xf, 2, Some (4, 5), Plain);
    (0xf, 0xe, 2, Some (4, 5), Plain);
    (0xf, 0xd, 3, Some (4, 6), Plain);
    (0x1f, 0x4, 6, Some (5, 7), Plain);
    (0x1f, 0x14, 6, Some (5, 7), Plain);
    (0x7f, 0x21, 16, Some (8, 10), Plain);
    ( 0x1f,
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
    (0x1f, 0xc, 8, Some (5, 6), Start start);
    (0x1f, 0x1c, 12, Some (5, 6), Start start);
    (0x1f, 0x5, 5, Some (5, 7), Plain);
    (0x1f, 0x6, 13, Some (5, 7), Plain);
    (0x1f, 0x16, 7, Some (5, 7), Plain);
    (0x7f, 0x31, 12, Some (7, 8), Plain);
    (0xf, 0x8, 2, Some (4, 7), Short);
    (0xf, 0x0, 1, None, Nop);
    (0x7f, 0x51, 6, Some (7, 15), Plain);
    (0xff, 0x61, 6, Some (8, 10), Plain);
    (0xff, 0xe1, 8, Some (8, 10), Plain);
    (0xf, 0x9, 16, Some (4, 6), Plain);
    (0x7f, 0x71, 16, Some (7, 9), Plain);
    (0x7f, 0x1, 12, Some (12, 47), Mark { rt = 9; pl = 8 });
    (0x7f, 0x11, 16, None, Layout);
    (0x7, 0x2, 5, Some (4, 6), Plain);
  ]

(* RDNA4 widens the work-group processor and overrides specific prefixes. The
   inherited 5-bit 0x6 and 0x16 rows retain their lengths and delta fields for
   bytes that no more specific prefix matches. *)
let rdna4 : packet list =
  let start =
    { cu = (10, 13); array = Some ((7, 7), 4); simd = (8, 9); slot = (15, 19) }
  in
  [
    (0x7, 0x3, 3, Some (3, 5), Plain);
    (0xf, 0xf, 2, Some (4, 5), Plain);
    (0xf, 0xe, 2, Some (4, 5), Plain);
    (0xf, 0xd, 3, Some (4, 6), Plain);
    (0x1f, 0x4, 6, Some (5, 7), Plain);
    (0x1f, 0x14, 6, Some (5, 7), Plain);
    (0x7f, 0x21, 18, Some (8, 10), Plain);
    ( 0x1f,
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
    (0x1f, 0xc, 8, Some (5, 6), Start start);
    (0x1f, 0x1c, 10, Some (5, 6), Start_ext start);
    (0x1f, 0x5, 6, Some (5, 7), Plain);
    (0x1f, 0x6, 13, Some (5, 7), Plain);
    (0x7f, 0x6, 14, Some (7, 9), Plain);
    (0x7f, 0x46, 8, Some (7, 9), Plain);
    (0x7f, 0x66, 6, Some (7, 9), Plain);
    (0x1f, 0x16, 7, Some (5, 7), Plain);
    (0x3f, 0x16, 12, Some (6, 8), Plain);
    (0x7f, 0x31, 12, Some (7, 8), Plain);
    (0xf, 0x8, 2, Some (4, 7), Short);
    (0xf, 0x0, 1, None, Nop);
    (0x7f, 0x51, 6, Some (7, 15), Plain);
    (0xff, 0x61, 6, Some (8, 10), Plain);
    (0xff, 0xe1, 8, Some (8, 10), Plain);
    (0xf, 0x9, 16, Some (4, 6), Plain);
    (0x7f, 0x71, 16, Some (7, 9), Plain);
    (0x7f, 0x1, 16, Some (12, 63), Mark { rt = 7; pl = 8 });
    (0x7f, 0x11, 16, None, Layout);
    (0x7, 0x2, 5, Some (3, 5), Plain);
  ]

(* A format: the packet type of each first byte. *)
type format = packet array

let popcount n =
  let rec go n c = if n = 0 then c else go (n land (n - 1)) (c + 1) in
  go n 0

(* A byte names the type of the most bits that match it; among types of as many
   bits, the first listed, NOP after the others. NOP names the bytes no other
   type does. *)
let format packets : format =
  let rank (mask, _, _, _, kind) = (-popcount mask, kind = Nop) in
  let ordered =
    List.stable_sort (fun a b -> compare (rank a) (rank b)) packets
  in
  let default = List.find (fun (_, _, _, _, kind) -> kind = Nop) packets in
  let matches b (mask, value, _, _, _) = b land mask = value in
  let of_byte b = Option.value ~default (List.find_opt (matches b) ordered) in
  Array.init 256 of_byte

let field reg (lo, hi) =
  Int64.(
    to_int
      (logand
         (shift_right_logical reg lo)
         (sub (shift_left 1L (hi - lo + 1)) 1L)))

let wave_of reg w =
  let cu =
    match w.array with
    | None -> field reg w.cu
    | Some (array, shift) -> field reg w.cu lor (field reg array lsl shift)
  in
  (cu, field reg w.simd, field reg w.slot)

(* The layouts a header names. *)
let rdna4_layout = 4

(* The nibble [i] of [data], least significant first, [0] past its end. *)
let nibble data i =
  if i lsr 1 >= String.length data then 0
  else
    let b = Char.code (String.unsafe_get data (i lsr 1)) in
    if i land 1 = 0 then b land 0xf else b lsr 4

(* The reader holds the 16 nibbles from the current packet's start in [reg], the
   packet's first at bit 0, and reads a packet only if the trace holds all its
   nibbles. *)
let iter f data =
  let total = 2 * String.length data in
  let fmt = ref (format rdna3) in
  let reg = ref 0L and start = ref 0 and time = ref 0 in
  let extended = ref false in
  let shift_in i =
    reg :=
      Int64.(
        logor
          (shift_right_logical !reg 4)
          (shift_left (of_int (nibble data i)) 60))
  in
  (* Takes the packet in [reg] into the trace's time and events. *)
  let decode (_, _, _, delta, kind) =
    let delta = Option.fold ~none:0 ~some:(field !reg) delta in
    let marker =
      match kind with
      | Mark { rt; pl } -> field !reg (rt, rt) = 1 && field !reg (pl, pl) = 0
      | Nop | Plain | Short | Layout | Start _ | Start_ext _ | End _ -> false
    in
    (match kind with
    | Mark _ when marker -> ()
    | Short -> time := !time + delta + 4
    | Nop | Plain | Mark _ | Layout | Start _ | Start_ext _ | End _ ->
        time := !time + delta);
    match kind with
    | Layout -> if field !reg (7, 12) = rdna4_layout then fmt := format rdna4
    | Mark _ when marker -> f (Marker { time = !time; realtime = delta })
    | Start w | Start_ext w ->
        let cu, simd, slot = wave_of !reg w in
        f (Wave_start { time = !time; cu; simd; slot })
    | End w ->
        let cu, simd, slot = wave_of !reg w in
        f (Wave_end { time = !time; cu; simd; slot })
    | Nop | Plain | Short | Mark _ -> ()
  in
  for i = 0 to 15 do
    shift_in i
  done;
  while !start < total do
    let byte = Int64.to_int !reg land 0xff in
    let ((_, _, size, _, kind) as packet) = !fmt.(byte) in
    (* Extended starts leave tagged bytes before the next packet. NOPs keep that
       context; another non-NOP packet ends it. *)
    let continuation = !extended && byte land 1 <> 0 in
    let size = if continuation then 2 else size in
    if !start + size > total then start := total
    else begin
      if not continuation then begin
        decode packet;
        if kind <> Nop then
          extended := match kind with Start_ext _ -> true | _ -> false
      end;
      for i = !start + 16 to !start + 15 + size do
        shift_in i
      done;
      start := !start + size
    end
  done
