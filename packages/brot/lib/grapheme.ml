(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let unicode_version = Grapheme_data.unicode_version

(* Classes *)

(* Grapheme_Cluster_Break values, as Grapheme_data numbers them. *)
let cr = 1
let lf = 2
let control = 3
let extend = 4
let zwj = 5
let regional_indicator = 6
let prepend = 7
let spacing_mark = 8
let l = 9
let v = 10
let t = 11
let lv = 12
let lvt = 13

(* Indic_Conjunct_Break values. *)
let consonant = 1
let incb_extend = 2
let linker = 3

(* The class of a code point: the last range starting at or before it. A byte
   that does not start a character reads as a control. *)
let malformed_class = control

let class_of code =
  let starts = Grapheme_data.starts in
  let lo = ref 0 and hi = ref (Array.length starts - 1) in
  while !lo < !hi do
    let mid = (!lo + !hi + 1) lsr 1 in
    if Array.unsafe_get starts mid <= code then lo := mid else hi := mid - 1
  done;
  Char.code (String.unsafe_get Grapheme_data.classes !lo)

let[@inline] gcb c = c land 0xF
let[@inline] incb c = (c lsr 4) land 3
let[@inline] pictographic c = c land 0x40 <> 0

(* The class and the byte length of the character at [i]. *)
let[@inline] next s i =
  let d = String.get_utf_8_uchar s i in
  if Uchar.utf_decode_is_valid d then
    (class_of (Uchar.to_int (Uchar.utf_decode_uchar d)) lsl 3)
    lor Uchar.utf_decode_length d
  else (malformed_class lsl 3) lor 1

(* Rules *)

(* Whether no boundary falls between a character of class [prev] and one of
   class [cur], given the state the cluster so far left. GB3 to GB13 of UAX #29
   in their order; GB999 breaks. *)
let joins prev cur ~linked ~zwj_after_pictograph ~odd_ri =
  let p = gcb prev and c = gcb cur in
  if p = cr && c = lf then true
  else if p = cr || p = lf || p = control then false
  else if c = cr || c = lf || c = control then false
  else if p = l && (c = l || c = v || c = lv || c = lvt) then true
  else if (p = lv || p = v) && (c = v || c = t) then true
  else if (p = lvt || p = t) && c = t then true
  else if c = extend || c = zwj || c = spacing_mark || p = prepend then true
  else if linked && incb cur = consonant then true
  else if p = zwj && zwj_after_pictograph && pictographic cur then true
  else p = regional_indicator && c = regional_indicator && odd_ri

(* States carried along a cluster, for the rules that look further back than
   one character. [conjunct] is 0 outside an Indic conjunct, 1 after its
   consonant and the extends and linkers after it, 2 once a linker is among
   them (GB9c). [emoji] is 0, 1 after an extended pictograph and its extends,
   2 once a ZWJ follows (GB11). [ri] counts the regional indicators ending the
   cluster (GB12, GB13). *)
let next_conjunct state cur =
  let k = incb cur in
  if k = consonant then 1
  else if k = linker then if state > 0 then 2 else 0
  else if k = incb_extend then state
  else 0

let next_emoji state cur =
  if pictographic cur then 1
  else if state = 1 && gcb cur = extend then 1
  else if state = 1 && gcb cur = zwj then 2
  else 0

let stop s i =
  let len = String.length s in
  let d = next s i in
  let prev = ref (d lsr 3) in
  let conjunct = ref (next_conjunct 0 !prev) in
  let emoji = ref (next_emoji 0 !prev) in
  let ri = ref (if gcb !prev = regional_indicator then 1 else 0) in
  let j = ref (i + (d land 7)) in
  let stop = ref (-1) in
  while !stop < 0 do
    if !j >= len then stop := len
    else begin
      let d = next s !j in
      let cur = d lsr 3 in
      if
        joins !prev cur ~linked:(!conjunct = 2)
          ~zwj_after_pictograph:(!emoji = 2) ~odd_ri:(!ri land 1 = 1)
      then begin
        conjunct := next_conjunct !conjunct cur;
        emoji := next_emoji !emoji cur;
        ri := if gcb cur = regional_indicator then !ri + 1 else 0;
        prev := cur;
        j := !j + (d land 7)
      end
      else stop := !j
    end
  done;
  !stop
