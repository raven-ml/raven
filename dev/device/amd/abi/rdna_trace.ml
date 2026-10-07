(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* No AMD document or open decoder describes RDNA's thread trace packets. The
   tables below are tinygrad's reverse engineering of them
   (tinygrad/renderer/amd/sqtt.py), checked by tinygrad against traces of
   gfx1100 and gfx1200 GPUs.

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
  | Plain
  | Short (* a delta of 4 more than its field *)
  | Mark of { rt : int; pl : int }
    (* a realtime marker when [rt] is set and [pl] clear, a delta otherwise *)
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

(* A format: the packet type of each first byte. *)
type format = packet array

let popcount n =
  let rec go n c = if n = 0 then c else go (n land (n - 1)) (c + 1) in
  go n 0

(* A byte names the type of the most bits that match it; among types of as many
   bits, the first listed, type 16 after the others. Type 16 names the bytes no
   other type does. *)
let format packets : format =
  let rank (id, mask, _, _, _, _) = (-popcount mask, id = 16) in
  let ordered =
    List.stable_sort (fun a b -> compare (rank a) (rank b)) packets
  in
  let default = List.find (fun (id, _, _, _, _, _) -> id = 16) packets in
  let matches b (_, mask, value, _, _, _) = b land mask = value in
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

(* The reader holds the next 16 nibbles in [reg], the current packet's from bit
   0, and shifts in as many as the packet before it took. *)
let iter f data =
  let n = String.length data in
  let byte i = Char.code (String.unsafe_get data i) in
  let fmt = ref (format rdna3) in
  let reg = ref 0L and pos = ref 0 and nib_off = ref 0 and nibbles = ref 16 in
  let time = ref 0 in
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
    let _, _, _, size, delta, kind = !fmt.(Int64.to_int !reg land 0xff) in
    nibbles := size;
    let delta = Option.fold ~none:0 ~some:(field !reg) delta in
    let marker =
      match kind with
      | Mark { rt; pl } -> field !reg (rt, rt) = 1 && field !reg (pl, pl) = 0
      | Plain | Short | Layout | Start _ | End _ -> false
    in
    (match kind with
    | Mark _ when marker -> ()
    | Short -> time := !time + delta + 4
    | Plain | Mark _ | Layout | Start _ | End _ -> time := !time + delta);
    match kind with
    | Layout -> if field !reg (7, 12) = rdna4_layout then fmt := format rdna4
    | Mark _ when marker -> f (Marker { time = !time; realtime = delta })
    | Start w ->
        let cu, simd, slot = wave_of !reg w in
        f (Wave_start { time = !time; cu; simd; slot })
    | End w ->
        let cu, simd, slot = wave_of !reg w in
        f (Wave_end { time = !time; cu; simd; slot })
    | Plain | Short | Mark _ -> ()
  done
