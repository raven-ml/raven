(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* A thread trace is a stream of packets of 1 to 18 nibbles, least significant
   nibble first. A packet's first byte names its type, whose size gives where
   the next packet starts; most packets advance the trace's time, in shader
   cycles, by a delta field. The first packet is a layout header that names the
   format: layout 3 (RDNA3), layout 4 (RDNA4), or another value for CDNA, whose
   first packet is then read again as a CDNA packet. *)

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

type packet = {
  id : int; (* the packet's type, as AMD's decoder numbers them *)
  mask : int; (* the bits of the first byte that name the type *)
  value : int;
  nibbles : int;
  delta : field option;
  kind : kind;
}

let packet ?delta ?(kind = Plain) id mask value nibbles =
  { id; mask; value; nibbles; delta; kind }

let rdna3_wave_start =
  { cu = (10, 12); array = Some ((7, 7), 3); simd = (8, 9); slot = (13, 17) }

let rdna3_wave_end =
  { cu = (11, 13); array = Some ((8, 8), 3); simd = (9, 10); slot = (15, 19) }

let rdna3 =
  [
    packet 1 0x7 0x3 3 ~delta:(3, 5);
    packet 2 0xf 0xf 2 ~delta:(4, 5);
    packet 3 0xf 0xe 2 ~delta:(4, 5);
    packet 4 0xf 0xd 3 ~delta:(4, 6);
    packet 5 0x1f 0x4 6 ~delta:(5, 7);
    packet 6 0x1f 0x14 6 ~delta:(5, 7);
    packet 7 0x7f 0x21 18 ~delta:(8, 10);
    packet 8 0x1f 0x15 5 ~delta:(5, 7) ~kind:(End rdna3_wave_end);
    packet 9 0x1f 0xc 8 ~delta:(5, 6) ~kind:(Start rdna3_wave_start);
    packet 10 0x1f 0x1c 12 ~delta:(5, 6);
    packet 11 0x1f 0x5 5 ~delta:(5, 7);
    packet 12 0x1f 0x6 13 ~delta:(5, 7);
    packet 13 0x1f 0x16 7 ~delta:(5, 7);
    packet 14 0x7f 0x31 12 ~delta:(7, 8);
    packet 15 0xf 0x8 2 ~delta:(4, 7) ~kind:Short;
    packet 16 0xf 0x0 1;
    packet 17 0x7f 0x51 6 ~delta:(7, 15);
    packet 18 0xff 0x61 6 ~delta:(8, 10);
    packet 19 0xff 0xe1 8 ~delta:(8, 10);
    packet 20 0xf 0x9 16 ~delta:(4, 6);
    packet 21 0x7f 0x71 16 ~delta:(7, 9);
    packet 22 0x7f 0x1 12 ~delta:(12, 47) ~kind:(Mark { rt = 9; pl = 8 });
    packet 23 0x7f 0x11 16 ~kind:Layout;
    packet 24 0x7 0x2 5 ~delta:(4, 6);
  ]

(* RDNA4 widens the work-group processor and moves fields of seven types. *)
let rdna4 =
  let wave_start =
    { cu = (10, 13); array = Some ((7, 7), 4); simd = (8, 9); slot = (15, 19) }
  and wave_end =
    { cu = (11, 14); array = Some ((8, 8), 4); simd = (9, 10); slot = (15, 19) }
  in
  let changed =
    [
      packet 8 0x1f 0x15 5 ~delta:(5, 7) ~kind:(End wave_end);
      packet 9 0x1f 0xc 8 ~delta:(5, 6) ~kind:(Start wave_start);
      packet 10 0x1f 0x1c 10 ~delta:(5, 6);
      packet 11 0x1f 0x5 6 ~delta:(5, 7);
      packet 12 0x1f 0x6 14 ~delta:(7, 9);
      packet 13 0x1f 0x16 8 ~delta:(7, 9);
      packet 22 0x7f 0x1 16 ~delta:(12, 63) ~kind:(Mark { rt = 7; pl = 8 });
      packet 24 0x7 0x2 5 ~delta:(3, 5);
    ]
  in
  List.map
    (fun p ->
      Option.value ~default:p (List.find_opt (fun c -> c.id = p.id) changed))
    rdna3

(* CDNA's deltas count 4 cycles, from bit 4 of every packet. *)
let cdna =
  let wave = { cu = (6, 9); array = None; simd = (14, 15); slot = (10, 13) } in
  let p ?kind id nibbles = packet ?kind id 0xf id nibbles ~delta:(4, 4) in
  [
    packet 0 0xf 0x0 4 ~delta:(4, 11);
    p 1 16 ~kind:Timestamp;
    p 2 16;
    p 3 8 ~kind:(Start wave);
    p 4 4;
    p 5 12;
    p 6 4 ~kind:(End wave);
    p 7 4;
    p 8 4;
    p 9 4;
    p 10 4;
    p 11 16;
    p 12 12;
    p 13 8;
    p 14 16;
    p 15 12;
    packet 16 0x7f 0x11 16 ~kind:Layout;
  ]

type format = { types : packet array; (* by first byte *) delta_scale : int }

let popcount n =
  let rec go n c = if n = 0 then c else go (n land (n - 1)) (c + 1) in
  go n 0

(* A byte names the type of the most bits that match it; among types of as many
   bits, the first listed, type 16 after the others. *)
let format ?(delta_scale = 1) packets =
  let ordered =
    List.stable_sort
      (fun a b ->
        compare (-popcount a.mask, a.id = 16) (-popcount b.mask, b.id = 16))
      packets
  in
  let default = List.find (fun p -> p.id = 16) packets in
  {
    types =
      Array.init 256 (fun b ->
          Option.value ~default
            (List.find_opt (fun p -> b land p.mask = p.value) ordered));
    delta_scale;
  }

let rdna3 = format rdna3
let rdna4 = format rdna4
let cdna = format ~delta_scale:4 cdna

(* Packets *)

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
   order. *)
let iter f data =
  let n = String.length data in
  let byte i = Char.code (String.unsafe_get data i) in
  let fmt = ref rdna3 in
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
    let p = !fmt.types.(Int64.to_int !reg land 0xff) in
    nibbles := p.nibbles;
    let delta = Option.fold ~none:0 ~some:(field !reg) p.delta in
    let marker =
      match p.kind with
      | Mark { rt; pl } -> field !reg (rt, rt) = 1 && field !reg (pl, pl) = 0
      | Plain | Short | Timestamp | Layout | Start _ | End _ -> false
    in
    (match p.kind with
    | Mark _ when marker -> ()
    | Short -> time := !time + delta + 4
    | Timestamp -> (
        if field !reg (4, 15) = 0 then
          let abs = field !reg (16, 63) in
          match !ts_offset with
          | None -> ts_offset := Some (abs - !time)
          | Some o -> time := ((abs - o) land lnot 3) - 4)
    | Plain | Mark _ | Layout | Start _ | End _ ->
        time := !time + (delta * !fmt.delta_scale));
    match p.kind with
    | Layout -> (
        match field !reg (7, 12) with
        | 3 -> ()
        | 4 -> fmt := rdna4
        | _ ->
            (* Not a layout header: the trace is CDNA's, and this packet one of
               its own. *)
            fmt := cdna;
            let p = cdna.types.(Int64.to_int !reg land 0xff) in
            nibbles := p.nibbles;
            if p.kind = Timestamp && field !reg (4, 15) = 0 then
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

(* Waves *)

type wave = { cu : int; simd : int; slot : int; start : int; stop : int }

let waves data =
  let started = Hashtbl.create 64 and waves = ref [] in
  iter
    (function
      | Wave_start { time; cu; simd; slot } ->
          Hashtbl.replace started (cu, simd, slot) time
      | Wave_end { time; cu; simd; slot } -> (
          match Hashtbl.find_opt started (cu, simd, slot) with
          | Some start ->
              Hashtbl.remove started (cu, simd, slot);
              waves := { cu; simd; slot; start; stop = time } :: !waves
          | None -> ())
      | Marker _ -> ())
    data;
  List.rev !waves

let markers data =
  let markers = ref [] in
  iter
    (function
      | Marker { time; realtime } -> markers := (time, realtime) :: !markers
      | Wave_start _ | Wave_end _ -> ())
    data;
  List.rev !markers

(* Realtime *)

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

let clock data =
  (* Markers of one shader time give no rate. *)
  let rec distinct = function
    | (s, _) :: ((s', _) :: _ as rest) when s = s' -> distinct rest
    | m :: rest -> m :: distinct rest
    | [] -> []
  in
  match distinct (markers data) with
  | _ :: _ :: _ as markers -> Some (realtime_of markers)
  | _ -> None
