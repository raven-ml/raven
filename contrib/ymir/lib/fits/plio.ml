(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* IRAF's PLIO line lists (FITS 4.0 §10.4.3; IRAF sys/plio/PLIO.hlp, the
   header in lib/plio.h). A list is 16-bit words: a header, then
   instructions

     +------+--------+------------+
     | sign | opcode |    data    |
     |  15  | 14..12 |   11..0    |
     +------+--------+------------+

   run against two values, the high value (1 at the start) and the output
   position. ZN outputs [data] zeros, HN [data] high values, PN [data - 1]
   zeros and one high value; SH sets the high value to the next word times
   4096 plus [data]; IH and DH add or subtract [data]; IS and DS do so and
   output one high value. The sign bit is unused, so a word with it set is
   no instruction. Pixels past the list's end are zeros, as IRAF's pl_l2p
   fills them: its pl_r2l leaves out a line's trailing zeros. FITS 4.0 says
   writers keep values in 0 .. 2^24; IRAF's masks reach 27 bits, and the
   caller checks values against ZBITPIX. *)

open Err

(* The instructions, by opcode as PLIO.hlp's table numbers them. *)
type instr = Zn | Sh | Ih | Dh | Hn | Pn | Is | Ds

let instrs = [| Zn; Sh; Ih; Dh; Hn; Pn; Is; Ds |]

(* The header: three words with the length third (OLL_LEN), or, when the
   third word is a version below 1, LL_HDRLEN words with the length's low
   and high 15 bits in words 4 and 5. *)
let old_header = 3
let header_words = 7
let data_bits = 12
let data_mask = (1 lsl data_bits) - 1
let len_bits = 15

let decode (src : Checksum.bigbytes) off len n ~until fill =
  let words = len / 2 in
  (* word [k] of the list, from 0, as the signed short IRAF reads *)
  let word k =
    let b = off + (2 * k) in
    Rice.signed 2
      ((Bigarray.Array1.unsafe_get src b lsl 8)
      lor Bigarray.Array1.unsafe_get src (b + 1))
  in
  if n > 0 then begin
    if words < old_header then
      fail "the line list holds %d words, short of a header" words;
    let stop, first =
      if word 2 > 0 then (word 2, old_header)
      else begin
        if words < header_words then
          fail "the line list holds %d words, short of its header" words;
        let h = word 1 in
        if h < header_words then
          fail "the line list's header is %d words, under %d" h header_words;
        ((word 4 lsl len_bits) + word 3, h)
      end
    in
    if stop > words then
      fail "the line list is %d words, %d are stored" stop words;
    let pv = ref 1 and x = ref 0 and ip = ref first in
    (* [out v m] outputs [m] values [v]; runs of zeros are not reported. *)
    let out v m =
      if v <> 0 && m > 0 then fill !x m v;
      x := !x + m
    in
    let last = Int.min n until in
    while !x < last && !ip < stop do
      let w = word !ip in
      if w < 0 then
        fail "word %d of the line list has its sign bit set" (!ip + 1);
      let data = w land data_mask in
      let m = Int.min data (n - !x) in
      (match Array.unsafe_get instrs (w lsr data_bits) with
      | Zn -> out 0 m
      | Hn -> out !pv m
      | Pn ->
          (* the high value lands only if the run ends within the tile *)
          if m = data && data > 0 then begin
            out 0 (data - 1);
            out !pv 1
          end
          else out 0 m
      | Sh ->
          if !ip + 1 >= stop then
            fail "the SH at word %d of the line list lacks its second word"
              (!ip + 1);
          pv := (word (!ip + 1) lsl data_bits) + data;
          incr ip
      | Ih -> pv := !pv + data
      | Dh -> pv := !pv - data
      | Is ->
          pv := !pv + data;
          out !pv 1
      | Ds ->
          pv := !pv - data;
          out !pv 1);
      incr ip
    done
  end
