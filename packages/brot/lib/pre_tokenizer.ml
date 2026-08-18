(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Types *)

type behavior =
  [ `Isolated
  | `Removed
  | `Merged_with_previous
  | `Merged_with_next
  | `Contiguous ]

type prepend_scheme = [ `First | `Never | `Always ]

type t =
  | Byte_level of {
      add_prefix_space : bool;
      use_regex : bool;
      trim_offsets : bool;
    }
  | Bert
  | Whitespace
  | Whitespace_split
  | Punctuation of { behavior : behavior }
  | Split of { pattern : string; behavior : behavior; invert : bool }
  | Char_delimiter of string
  | Digits of { individual : bool }
  | Metaspace of {
      replacement : string;
      prepend_scheme : prepend_scheme;
      split : bool;
    }
  | Sequence of t list
  | Fixed_length of { length : int }
  | Unicode_scripts

type rewrite =
  | Verbatim
  | Prefix_space
  | Space_marker of { marker : string; prepend : prepend_scheme }

type plan =
  | Walk of { rewrite : rewrite; splittable : bool }
  | Segmented of { outer : t; rewrite : rewrite; inner : t; splittable : bool }
  | Pieces

(* Errors *)

let strf = Printf.sprintf
let err_unknown_behavior s = strf "unknown punctuation behavior '%s'" s
let err_unknown_scheme s = strf "unknown prepend_scheme '%s'" s
let err_unsupported_type s = strf "unsupported pre-tokenizer type '%s'" s
let err_missing_type = "missing 'type' field"
let err_expected_object = "expected JSON object"
let err_missing_behavior = "missing 'behavior' field"
let err_split_missing = "requires 'pattern' and 'behavior'"
let err_char_delim_missing = "requires 'delimiter' as one character"
let err_split_pattern = "expected 'pattern' as {\"String\": ...}"
let err_split_regex = "regular expression 'pattern' is not supported"
let err_metaspace_missing = "requires a non-empty 'replacement'"
let err_metaspace_scheme = "expected a string for 'prepend_scheme'"
let err_sequence_missing = "requires 'pretokenizers' list"
let err_fixed_length = "requires positive length"
let err_no_walk = "pre-tokenizer is not a span walker"
let err_range = "range is not within the text"
let err_replacement = "metaspace replacement must be one character"
let err_delimiter = "delimiter must be one character"

(* Byte-level encoding *)

(* Pre-computed byte ↔ unicode mappings for byte-level encode/decode *)
let byte_to_unicode, unicode_to_byte =
  let is_direct = Array.make 256 false in
  for i = 33 to 126 do
    is_direct.(i) <- true
  done;
  for i = 161 to 172 do
    is_direct.(i) <- true
  done;
  for i = 174 to 255 do
    is_direct.(i) <- true
  done;
  let byte_to_unicode = Array.make 256 0 in
  let next_code = ref 0 in
  let max_code = ref 0 in
  for b = 0 to 255 do
    let code =
      if is_direct.(b) then b
      else
        let code = 256 + !next_code in
        incr next_code;
        code
    in
    byte_to_unicode.(b) <- code;
    if code > !max_code then max_code := code
  done;
  let unicode_to_byte = Array.make (!max_code + 1) (-1) in
  for b = 0 to 255 do
    let code = byte_to_unicode.(b) in
    if code < Array.length unicode_to_byte then unicode_to_byte.(code) <- b
  done;
  (byte_to_unicode, unicode_to_byte)

(* Writes the byte-level encoding of [s.\[start..stop-1\]] at the start of
   [buf], which must hold [2 * (stop - start)] bytes, and returns its length. *)
let byte_level_blit buf s ~start ~stop =
  let j = ref 0 in
  for i = start to stop - 1 do
    let u =
      Array.unsafe_get byte_to_unicode (Char.code (String.unsafe_get s i))
    in
    if u < 128 then begin
      Bytes.unsafe_set buf !j (Char.unsafe_chr u);
      incr j
    end
    else begin
      Bytes.unsafe_set buf !j (Char.unsafe_chr (0xC0 lor (u lsr 6)));
      Bytes.unsafe_set buf (!j + 1) (Char.unsafe_chr (0x80 lor (u land 0x3F)));
      j := !j + 2
    end
  done;
  !j

let byte_level_encode text =
  let stop = String.length text in
  let buf = Bytes.create (stop * 2) in
  Bytes.sub_string buf 0 (byte_level_blit buf text ~start:0 ~stop)

(* One character outside the alphabet costs the whole token its mapping, as it
   does in HuggingFace. Every character of the alphabet is one or two UTF-8
   bytes, so a longer one is outside it, and so is a byte sequence that is not
   UTF-8 at all. *)
let byte_level_decode text =
  let len = String.length text in
  let bytes = Bytes.create len in
  let i = ref 0 and j = ref 0 and mapped = ref true in
  while !mapped && !i < len do
    let b0 = Char.code (String.unsafe_get text !i) in
    let code, width =
      if b0 < 0x80 then (b0, 1)
      else if b0 < 0xC2 || b0 >= 0xE0 || !i + 1 >= len then (-1, 1)
      else
        let b1 = Char.code (String.unsafe_get text (!i + 1)) in
        if b1 land 0xC0 <> 0x80 then (-1, 1)
        else (((b0 land 0x1F) lsl 6) lor (b1 land 0x3F), 2)
    in
    let byte =
      if code < 0 || code >= Array.length unicode_to_byte then -1
      else Array.unsafe_get unicode_to_byte code
    in
    if byte < 0 then mapped := false
    else begin
      Bytes.unsafe_set bytes !j (Char.unsafe_chr byte);
      incr j;
      i := !i + width
    end
  done;
  if !mapped then Bytes.sub_string bytes 0 !j else text

(* Byte-level walker

   The GPT-2 pattern ['s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
   ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+], matched by taking the first alternative
   that applies at each position. The dispatch is on the leading byte: the four
   character categories plus the space and the apostrophe, which open
   alternatives of their own, and the two shapes of non-ASCII byte. *)

let c_other = Char_class.other
let c_whitespace = Char_class.whitespace
let c_letter = Char_class.letter
let c_numeric = Char_class.numeric
let c_space = 4
let c_apostrophe = 5
let c_lead = 6
let c_continuation = 7

let lead_class =
  let t = Bytes.make 256 (Char.unsafe_chr c_lead) in
  for b = 0 to 127 do
    Bytes.set t b (Char.unsafe_chr (Char_class.category b))
  done;
  Bytes.set t 32 (Char.unsafe_chr c_space);
  Bytes.set t 39 (Char.unsafe_chr c_apostrophe);
  for b = 0x80 to 0xBF do
    Bytes.set t b (Char.unsafe_chr c_continuation)
  done;
  t

external get64u : string -> int -> int64 = "%caml_string_get64u"

let debruijn = 0x03f79d71b4ca8b09L

let debruijn_index =
  let t = Bytes.make 64 '\000' in
  for i = 0 to 63 do
    let k =
      Int64.to_int
        (Int64.shift_right_logical
           (Int64.mul (Int64.shift_left 1L i) debruijn)
           58)
    in
    Bytes.set t k (Char.unsafe_chr i)
  done;
  t

let[@inline] ctz64 x =
  let low = Int64.logand x (Int64.neg x) in
  Char.code
    (Bytes.unsafe_get debruijn_index
       (Int64.to_int (Int64.shift_right_logical (Int64.mul low debruijn) 58)))

let hi_bits = 0x8080808080808080L
let lo_bits = 0x7F7F7F7F7F7F7F7FL

(* Bit 7 of each byte that is not an ASCII letter. *)
let[@inline] non_letters w =
  let ascii = Int64.logand w lo_bits in
  let lowered = Int64.logor ascii 0x2020202020202020L in
  let ge_a = Int64.sub (Int64.logor lowered hi_bits) 0x6161616161616161L in
  let le_z = Int64.sub 0xFAFAFAFAFAFAFAFAL lowered in
  let letters = Int64.logand (Int64.logand ge_a le_z) hi_bits in
  Int64.logand (Int64.logor (Int64.lognot letters) w) hi_bits

let letters_swar s i stop =
  let j = ref i in
  let scanning = ref true in
  while !scanning && !j + 8 <= stop do
    let m = non_letters (get64u s !j) in
    if Int64.equal m 0L then j := !j + 8
    else begin
      j := !j + (ctz64 m lsr 3);
      scanning := false
    end
  done;
  !j

(* End of the run of characters of category [category] starting at [i]. The
   apostrophe and the space are folded back into their category. *)
let category_run s i stop category =
  let j = ref (if category = c_letter then letters_swar s i stop else i) in
  let scanning = ref true in
  while !scanning && !j < stop do
    let b = Char.code (String.unsafe_get s !j) in
    let c = Char.code (Bytes.unsafe_get lead_class b) in
    if c < c_lead then begin
      let c =
        if c = c_apostrophe then c_other
        else if c = c_space then c_whitespace
        else c
      in
      if c = category then incr j else scanning := false
    end
    else if c = c_lead then begin
      let d = Char_class.at s !j ~stop in
      if Char_class.at_category d = category then j := !j + Char_class.at_len d
      else scanning := false
    end
    else if category = c_other then incr j
    else scanning := false
  done;
  !j

(* End of the [\s+(?!\S)] or [\s+] span starting at the whitespace at [i]: the
   whole run, or all but its last character when a non-whitespace character
   follows it. *)
let whitespace_span s i stop =
  let j = ref i in
  let last = ref i in
  let scanning = ref true in
  while !scanning && !j < stop do
    let b = Char.code (String.unsafe_get s !j) in
    let c = Char.code (Bytes.unsafe_get lead_class b) in
    if c = c_whitespace || c = c_space then begin
      last := !j;
      incr j
    end
    else if c = c_lead then begin
      let d = Char_class.at s !j ~stop in
      if Char_class.at_category d = c_whitespace then begin
        last := !j;
        j := !j + Char_class.at_len d
      end
      else scanning := false
    end
    else scanning := false
  done;
  if !j = stop then !j else if !last > i then !last else !j

let contraction s i stop =
  if stop - i < 2 then stop
  else
    let c1 = String.unsafe_get s (i + 1) in
    if c1 = 's' || c1 = 't' || c1 = 'm' || c1 = 'd' then i + 2
    else if
      stop - i >= 3
      &&
      let c2 = String.unsafe_get s (i + 2) in
      (c1 = 'r' && c2 = 'e') || (c1 = 'v' && c2 = 'e') || (c1 = 'l' && c2 = 'l')
    then i + 3
    else category_run s (i + 1) stop c_other

let fill_byte_level s ~pos ~stop spans =
  let capacity = Spans.capacity spans in
  let n = ref (Spans.count spans) in
  let p = ref pos in
  while !p < stop && !n < capacity do
    let i = !p in
    let b = Char.code (String.unsafe_get s i) in
    let c = Char.code (Bytes.unsafe_get lead_class b) in
    let e =
      if c = c_letter then category_run s (i + 1) stop c_letter
      else if c = c_space then
        if i + 1 >= stop then stop
        else
          let d = Char_class.at s (i + 1) ~stop in
          let category = Char_class.at_category d in
          if category = c_whitespace then whitespace_span s i stop
          else category_run s (i + 1 + Char_class.at_len d) stop category
      else if c = c_apostrophe then contraction s i stop
      else if c = c_numeric then category_run s (i + 1) stop c_numeric
      else if c = c_other then category_run s (i + 1) stop c_other
      else if c = c_whitespace then whitespace_span s i stop
      else if c = c_lead then
        let d = Char_class.at s i ~stop in
        let category = Char_class.at_category d in
        if category = c_whitespace then whitespace_span s i stop
        else category_run s (i + Char_class.at_len d) stop category
      else category_run s (i + 1) stop c_other
    in
    Spans.write spans !n i e;
    incr n;
    p := e
  done;
  Spans.set_count spans !n;
  !p

(* The other walkers *)

let fill_whole ~pos ~stop spans =
  let n = Spans.count spans in
  if pos < stop && n < Spans.capacity spans then begin
    Spans.write spans n pos stop;
    Spans.set_count spans (n + 1);
    stop
  end
  else pos

let skip_whitespace s i stop =
  let j = ref i in
  let scanning = ref true in
  while !scanning && !j < stop do
    let d = Char_class.at s !j ~stop in
    if Char_class.at_category d = Char_class.whitespace then
      j := !j + Char_class.at_len d
    else scanning := false
  done;
  !j

let fill_whitespace_split s ~pos ~stop spans =
  let capacity = Spans.capacity spans in
  let n = ref (Spans.count spans) in
  let p = ref pos in
  let go = ref true in
  while !go do
    let i = skip_whitespace s !p stop in
    p := i;
    if i >= stop || !n >= capacity then go := false
    else begin
      let j = ref i in
      let scanning = ref true in
      while !scanning && !j < stop do
        let d = Char_class.at s !j ~stop in
        if Char_class.at_category d = Char_class.whitespace then
          scanning := false
        else j := !j + Char_class.at_len d
      done;
      Spans.write spans !n i !j;
      incr n;
      p := !j
    end
  done;
  Spans.set_count spans !n;
  !p

(* [\w+|[^\w\s]+] *)
let fill_whitespace s ~pos ~stop spans =
  let capacity = Spans.capacity spans in
  let n = ref (Spans.count spans) in
  let p = ref pos in
  let go = ref true in
  while !go do
    let i = skip_whitespace s !p stop in
    p := i;
    if i >= stop || !n >= capacity then go := false
    else begin
      let d = Char_class.at s i ~stop in
      let word = Char_class.at_is_word d in
      let j = ref (i + Char_class.at_len d) in
      let scanning = ref true in
      while !scanning && !j < stop do
        let d = Char_class.at s !j ~stop in
        if
          Char_class.at_is_word d = word
          && (word || Char_class.at_category d <> Char_class.whitespace)
        then j := !j + Char_class.at_len d
        else scanning := false
      done;
      Spans.write spans !n i !j;
      incr n;
      p := !j
    end
  done;
  Spans.set_count spans !n;
  !p

let fill_bert s ~pos ~stop spans =
  let capacity = Spans.capacity spans in
  let n = ref (Spans.count spans) in
  let p = ref pos in
  let go = ref true in
  while !go do
    let i = skip_whitespace s !p stop in
    p := i;
    if i >= stop || !n >= capacity then go := false
    else begin
      let d = Char_class.at s i ~stop in
      let l = Char_class.at_len d in
      if Char_class.at_is_punctuation d then begin
        Spans.write spans !n i (i + l);
        incr n;
        p := i + l
      end
      else begin
        let j = ref (i + l) in
        let scanning = ref true in
        while !scanning && !j < stop do
          let d = Char_class.at s !j ~stop in
          if
            Char_class.at_is_punctuation d
            || Char_class.at_category d = Char_class.whitespace
          then scanning := false
          else j := !j + Char_class.at_len d
        done;
        Spans.write spans !n i !j;
        incr n;
        p := !j
      end
    end
  done;
  Spans.set_count spans !n;
  !p

(* Splitting

   The pieces of [Punctuation] and [Split] are runs of segments, which alternate
   between the delimiters — the punctuation characters, or the occurrences of
   the pattern — and the text they separate. Each behavior is a rule over
   neighbouring segments. [`Isolated] gives every segment a piece of its own,
   and [`Removed] does the same but drops the delimiters. [`Contiguous] joins
   the neighbours that share a role. [`Merged_with_previous] and
   [`Merged_with_next] are one rule read from either side — a segment takes in
   the one that follows it when their roles differ — applied to the text and to
   the delimiters respectively.

   A segment is read as [(stop lsl 1) lor is_delimiter], which allocates
   nothing, and its role alone is read without walking to its end, which is all
   the grouping loop needs. The two walkers below are this one loop over the two
   kinds of delimiter: a change to either belongs in both. *)

let[@inline] segment_stop g = g lsr 1
let[@inline] segment_is_delimiter g = g land 1 = 1

let plain_run s i stop =
  let j = ref i in
  let scanning = ref true in
  while !scanning && !j < stop do
    let d = Char_class.at s !j ~stop in
    if Char_class.at_is_punctuation d then scanning := false
    else j := !j + Char_class.at_len d
  done;
  !j

let[@inline] punctuation_delimits s i stop =
  Char_class.at_is_punctuation (Char_class.at s i ~stop)

let punctuation_segment s i stop =
  let d = Char_class.at s i ~stop in
  if Char_class.at_is_punctuation d then ((i + Char_class.at_len d) lsl 1) lor 1
  else plain_run s i stop lsl 1

let fill_punctuation ~behavior s ~pos ~stop spans =
  let keep = behavior <> `Removed in
  let group = behavior = `Contiguous in
  let merge_previous = behavior = `Merged_with_previous in
  let merge_next = behavior = `Merged_with_next in
  let capacity = Spans.capacity spans in
  let n = ref (Spans.count spans) in
  let p = ref pos in
  while !p < stop && !n < capacity do
    let i = !p in
    let g = punctuation_segment s i stop in
    let delimiter = segment_is_delimiter g in
    let merge = if delimiter then merge_next else merge_previous in
    let e = ref (segment_stop g) in
    if group then
      while !e < stop && punctuation_delimits s !e stop = delimiter do
        e := segment_stop (punctuation_segment s !e stop)
      done
    else if merge && !e < stop then begin
      let g = punctuation_segment s !e stop in
      if segment_is_delimiter g <> delimiter then e := segment_stop g
    end;
    if keep || not delimiter then begin
      Spans.write spans !n i !e;
      incr n
    end;
    p := !e
  done;
  Spans.set_count spans !n;
  !p

let fill_digits ~individual s ~pos ~stop spans =
  let capacity = Spans.capacity spans in
  let n = ref (Spans.count spans) in
  let p = ref pos in
  while !p < stop && !n < capacity do
    let i = !p in
    let d = Char_class.at s i ~stop in
    let digit = Char_class.at_category d = Char_class.numeric in
    let e =
      if digit && individual then i + Char_class.at_len d
      else begin
        let j = ref (i + Char_class.at_len d) in
        let scanning = ref true in
        while !scanning && !j < stop do
          let d = Char_class.at s !j ~stop in
          let numeric = Char_class.at_category d = Char_class.numeric in
          if numeric = digit then j := !j + Char_class.at_len d
          else scanning := false
        done;
        !j
      end
    in
    Spans.write spans !n i e;
    incr n;
    p := e
  done;
  Spans.set_count spans !n;
  !p

let[@inline] matches pattern plen s i stop =
  i + plen <= stop
  &&
  let k = ref 0 in
  while
    !k < plen && String.unsafe_get s (i + !k) = String.unsafe_get pattern !k
  do
    incr k
  done;
  !k = plen

(* [invert] makes the text between the occurrences the delimiters, and the
   occurrences the text. *)

let[@inline] split_delimits pattern plen invert s i stop =
  matches pattern plen s i stop <> invert

let split_segment pattern plen invert s i stop =
  if matches pattern plen s i stop then
    ((i + plen) lsl 1) lor if invert then 0 else 1
  else begin
    let j = ref (i + 1) in
    while !j < stop && not (matches pattern plen s !j stop) do
      incr j
    done;
    (!j lsl 1) lor if invert then 1 else 0
  end

let fill_split ~pattern ~behavior ~invert s ~pos ~stop spans =
  let plen = String.length pattern in
  let keep = behavior <> `Removed in
  let group = behavior = `Contiguous in
  let merge_previous = behavior = `Merged_with_previous in
  let merge_next = behavior = `Merged_with_next in
  let capacity = Spans.capacity spans in
  let n = ref (Spans.count spans) in
  let p = ref pos in
  while !p < stop && !n < capacity do
    let i = !p in
    let g = split_segment pattern plen invert s i stop in
    let delimiter = segment_is_delimiter g in
    let merge = if delimiter then merge_next else merge_previous in
    let e = ref (segment_stop g) in
    if group then
      while
        !e < stop && split_delimits pattern plen invert s !e stop = delimiter
      do
        e := segment_stop (split_segment pattern plen invert s !e stop)
      done
    else if merge && !e < stop then begin
      let g = split_segment pattern plen invert s !e stop in
      if segment_is_delimiter g <> delimiter then e := segment_stop g
    end;
    if keep || not delimiter then begin
      Spans.write spans !n i !e;
      incr n
    end;
    p := !e
  done;
  Spans.set_count spans !n;
  !p

let fill_characters s ~pos ~stop spans =
  let capacity = Spans.capacity spans in
  let n = ref (Spans.count spans) in
  let p = ref pos in
  while !p < stop && !n < capacity do
    let e = !p + Char_class.at_len (Char_class.at s !p ~stop) in
    Spans.write spans !n !p e;
    incr n;
    p := e
  done;
  Spans.set_count spans !n;
  !p

(* Metaspace splits before every occurrence of its marker. *)
let fill_marker marker s ~pos ~stop spans =
  let mlen = String.length marker in
  let capacity = Spans.capacity spans in
  let n = ref (Spans.count spans) in
  let p = ref pos in
  while !p < stop && !n < capacity do
    let i = !p in
    let j = ref (if matches marker mlen s i stop then i + mlen else i) in
    let scanning = ref true in
    while !scanning && !j < stop do
      if matches marker mlen s !j stop then scanning := false else incr j
    done;
    Spans.write spans !n i !j;
    incr n;
    p := !j
  done;
  Spans.set_count spans !n;
  !p

(* [`Any] joins whichever run surrounds it. HuggingFace gives it to U+0020 and
   to the code points its script table does not know; every other whitespace
   character keeps its own script, [`Zyyy] (Common) for all of them but the
   Ogham space mark. *)
type script = [ `Any | Uucp.Script.t ]

let fixed_script code : script =
  if not (Uchar.is_valid code) then `Any
  else if code = 0x30FC then (`Hani :> script)
  else
    match Uucp.Script.script (Uchar.unsafe_of_int code) with
    | `Hira | `Kana -> (`Hani :> script)
    | `Zzzz -> `Any
    | s -> (s :> script)

let script_at s i d : script =
  let b = Char.code (String.unsafe_get s i) in
  match Char_class.at_len d with
  | 1 ->
      if b >= 0x80 then `Any
      else if b = 32 then `Any
      else if Char_class.category b = Char_class.letter then `Latn
      else `Zyyy
  | 2 ->
      fixed_script
        (((b land 0x1F) lsl 6)
        lor (Char.code (String.unsafe_get s (i + 1)) land 0x3F))
  | 3 ->
      fixed_script
        (((b land 0x0F) lsl 12)
        lor ((Char.code (String.unsafe_get s (i + 1)) land 0x3F) lsl 6)
        lor (Char.code (String.unsafe_get s (i + 2)) land 0x3F))
  | _ ->
      fixed_script
        (((b land 0x07) lsl 18)
        lor ((Char.code (String.unsafe_get s (i + 1)) land 0x3F) lsl 12)
        lor ((Char.code (String.unsafe_get s (i + 2)) land 0x3F) lsl 6)
        lor (Char.code (String.unsafe_get s (i + 3)) land 0x3F))

(* A piece runs from a script change to the next one, absorbing the [`Any]
   characters it meets. A leading [`Any] run belongs to no piece and is dropped:
   a piece can only open on a script change. *)
let fill_unicode_scripts s ~pos ~stop spans =
  let capacity = Spans.capacity spans in
  let n = ref (Spans.count spans) in
  let p = ref pos in
  let go = ref true in
  while !go do
    let i = ref !p in
    let opening = ref true in
    while !opening && !i < stop do
      let d = Char_class.at s !i ~stop in
      if script_at s !i d = `Any then i := !i + Char_class.at_len d
      else opening := false
    done;
    p := !i;
    if !i >= stop || !n >= capacity then go := false
    else begin
      let start = !i in
      let d = Char_class.at s start ~stop in
      let script = script_at s start d in
      let j = ref (start + Char_class.at_len d) in
      let scanning = ref true in
      while !scanning && !j < stop do
        let d = Char_class.at s !j ~stop in
        let s' = script_at s !j d in
        if s' <> `Any && s' <> script then scanning := false
        else j := !j + Char_class.at_len d
      done;
      Spans.write spans !n start !j;
      incr n;
      p := !j
    end
  done;
  Spans.set_count spans !n;
  !p

(* Plan and fill *)

let rec last = function [] -> None | [ t ] -> Some t | _ :: ts -> last ts

(* The pieces of a byte-level pre-tokenizer are byte-level encoded, so a
   sequence member that ends in one hands encoded text to the next. *)
let rec ends_byte_level t =
  match t with
  | Byte_level _ -> true
  | Sequence ts -> (
      match last ts with Some t -> ends_byte_level t | None -> false)
  | _ -> false

let encodes_bytes = ends_byte_level

(* Whether the pieces of [t] are bytes of the text it was given, which is what
   lets the member of a sequence that follows it place its own pieces inside
   them. A byte-level pre-tokenizer encodes its pieces and a metaspace marks
   them, and neither hands on text that the next member's offsets could
   index. *)
let rec pieces_are_slices t =
  match t with
  | Byte_level _ | Metaspace _ -> false
  | Sequence ts -> List.for_all pieces_are_slices ts
  | Bert | Whitespace | Whitespace_split | Punctuation _ | Split _
  | Char_delimiter _ | Digits _ | Fixed_length _ | Unicode_scripts ->
      true

(* Shared, so that [plan] allocates only for the pre-tokenizers whose rewrite
   carries a marker. *)
let walk_verbatim = Walk { rewrite = Verbatim; splittable = false }
let walk_split_free = Walk { rewrite = Verbatim; splittable = true }
let walk_prefix = Walk { rewrite = Prefix_space; splittable = false }
let walk_prefix_split_free = Walk { rewrite = Prefix_space; splittable = true }

(* A sequence nested in a sequence is its members in place. *)
let rec flatten ts =
  List.concat_map (function Sequence ts -> flatten ts | t -> [ t ]) ts

let of_members = function [ t ] -> t | ts -> Sequence ts

let rec plan t =
  match t with
  | Byte_level { add_prefix_space; use_regex; _ } -> (
      match (add_prefix_space, use_regex) with
      | false, false -> walk_verbatim
      | false, true -> walk_split_free
      | true, false -> walk_prefix
      | true, true -> walk_prefix_split_free)
  | Bert | Whitespace | Whitespace_split -> walk_split_free
  | Punctuation _ | Digits _ | Char_delimiter _ | Unicode_scripts | Split _ ->
      walk_verbatim
  | Metaspace { replacement; prepend_scheme; _ } ->
      let marker = replacement and prepend = prepend_scheme in
      Walk { rewrite = Space_marker { marker; prepend }; splittable = false }
  | Fixed_length _ -> Pieces
  | Sequence ts -> plan_sequence (flatten ts)

(* The members before the one that rewrites cut the text into segments, and it
   and the members after it walk each segment once rewritten. Whether a cut at a
   space is safe is the first member's business, whichever role it plays: the
   spans of every later member lie inside its own. *)
and plan_sequence ts =
  let rec inner_byte_level = function
    | [] | [ _ ] -> false
    | t :: ts -> ends_byte_level t || inner_byte_level ts
  in
  let walks_verbatim t =
    match plan t with Walk { rewrite = Verbatim; _ } -> true | _ -> false
  in
  let splittable =
    match ts with
    | [] -> false
    | first :: _ -> (
        match plan first with
        | Walk { splittable; _ } -> splittable
        | Segmented _ | Pieces -> false)
  in
  let rec split outer = function
    | [] ->
        if outer = [] then Pieces else Walk { rewrite = Verbatim; splittable }
    | t :: rest -> (
        match plan t with
        | Walk { rewrite = Verbatim; _ } -> split (t :: outer) rest
        | Walk { rewrite; _ } ->
            if not (List.for_all walks_verbatim rest) then Pieces
            else if outer = [] then Walk { rewrite; splittable }
            else
              Segmented
                {
                  outer = of_members (List.rev outer);
                  rewrite;
                  inner = of_members (t :: rest);
                  splittable;
                }
        | Segmented _ | Pieces -> Pieces)
  in
  if inner_byte_level ts then Pieces else split [] ts

(* Whether [fill] applies: a single walk, which for a sequence takes the plan to
   tell. *)
let walkable t =
  match t with
  | Fixed_length _ -> false
  | Sequence ts -> (
      match plan_sequence (flatten ts) with
      | Walk _ -> true
      | Segmented _ | Pieces -> false)
  | _ -> true

let rec fill_walk t s ~pos ~stop spans =
  match t with
  | Byte_level { use_regex; _ } ->
      if use_regex then fill_byte_level s ~pos ~stop spans
      else fill_whole ~pos ~stop spans
  | Bert -> fill_bert s ~pos ~stop spans
  | Whitespace -> fill_whitespace s ~pos ~stop spans
  | Whitespace_split -> fill_whitespace_split s ~pos ~stop spans
  | Punctuation { behavior } -> fill_punctuation ~behavior s ~pos ~stop spans
  (* An empty pattern matches at every position, so its occurrences are empty
     and the pieces are the characters — unless [invert] makes those occurrences
     the text, and [`Removed] then leaves nothing, the range being consumed
     without a span. *)
  | Split { pattern = ""; behavior = `Removed; invert = true } -> stop
  | Split { pattern = ""; _ } -> fill_characters s ~pos ~stop spans
  | Split { pattern; behavior; invert } ->
      fill_split ~pattern ~behavior ~invert s ~pos ~stop spans
  | Char_delimiter delimiter ->
      fill_split ~pattern:delimiter ~behavior:`Removed ~invert:false s ~pos
        ~stop spans
  | Digits { individual } -> fill_digits ~individual s ~pos ~stop spans
  (* Without splitting the marked text is one piece, and one span is what places
     the tokens of the model inside it. *)
  | Metaspace { split = false; _ } -> fill_whole ~pos ~stop spans
  | Metaspace { replacement; _ } -> fill_marker replacement s ~pos ~stop spans
  | Unicode_scripts -> fill_unicode_scripts s ~pos ~stop spans
  | Fixed_length _ -> invalid_arg err_no_walk
  | Sequence ts -> fill_sequence ts s ~pos ~stop spans

(* Each member is filled over the spans of the previous one, through one scratch
   buffer per member rather than one per span, sized to the range since no span
   is empty: a segment of a few bytes costs a few words. A member that runs out
   of room leaves its outer span unfinished: the spans it did emit are dropped
   and the fill resumes at that span's start. *)
and fill_sequence ts s ~pos ~stop out =
  match ts with
  | [] -> invalid_arg err_no_walk
  | [ t ] -> fill_walk t s ~pos ~stop out
  | _ ->
      let capacity = min (Spans.capacity out) (stop - pos + 1) in
      let scratch =
        Array.init (List.length ts - 1) (fun _ -> Spans.create ~capacity)
      in
      fill_chain ts s ~pos ~stop out scratch 0

and fill_chain ts s ~pos ~stop out scratch level =
  match ts with
  | [] -> invalid_arg err_no_walk
  | [ t ] -> fill_walk t s ~pos ~stop out
  | t :: rest ->
      let capacity = Spans.capacity out in
      let buffer = scratch.(level) in
      let p = ref pos in
      let go = ref true in
      while !go && !p < stop && Spans.count out < capacity do
        Spans.clear buffer;
        let resume = fill_walk t s ~pos:!p ~stop buffer in
        let n = Spans.count buffer in
        if n = 0 then begin
          if resume <= !p then go := false;
          p := resume
        end
        else begin
          let k = ref 0 in
          while !go && !k < n do
            let start = Spans.start buffer !k in
            let finish = Spans.stop buffer !k in
            let mark = Spans.count out in
            if
              fill_chain rest s ~pos:start ~stop:finish out scratch (level + 1)
              < finish
            then begin
              Spans.set_count out mark;
              p := start;
              go := false
            end
            else incr k
          done;
          if !go then p := resume
        end
      done;
      !p

let fill t s ~pos ~stop spans =
  if not (walkable t) then invalid_arg err_no_walk;
  if pos < 0 || stop < pos || stop > String.length s then invalid_arg err_range;
  fill_walk t s ~pos ~stop spans

(* Pre-tokenize *)

let span_chunk = 1024

(* A rewrite is the normalizer it behaves as, which is the one [Brot] composes
   for the same plan, so that a piece is placed back in the text the caller gave
   through the one alignment the library has. [None] is the rewrite firing on
   nothing, which leaves the text as it is. Prepending the marker before or
   after replacing the spaces gives the same text, the marker holding no
   space. *)
let rewriter rewrite =
  match rewrite with
  | Verbatim -> fun ~first:_ _ -> None
  | Prefix_space ->
      let space = Normalizer.prepend " " in
      fun ~first:_ text ->
        if String.length text > 0 && String.unsafe_get text 0 <> ' ' then
          Some space
        else None
  | Space_marker { marker; prepend } -> (
      let mark = Normalizer.replace ~pattern:" " ~replacement:marker in
      let mark_and_prepend =
        Normalizer.sequence [ mark; Normalizer.prepend marker ]
      in
      let unmarked text =
        String.unsafe_get text 0 <> ' '
        && not (String.starts_with ~prefix:marker text)
      in
      match prepend with
      | `Never -> fun ~first:_ text -> if text = "" then None else Some mark
      | `Always ->
          fun ~first:_ text ->
            if text = "" then None
            else if unmarked text then Some mark_and_prepend
            else Some mark
      | `First ->
          fun ~first text ->
            if text = "" then None
            else if unmarked text && Lazy.force first then Some mark_and_prepend
            else Some mark)

let pre_tokenize_fixed_length ~length text =
  if length <= 0 || String.length text = 0 then []
  else
    let pieces = ref [] in
    let stop = String.length text in
    let i = ref 0 in
    while !i < stop do
      let start = !i in
      let count = ref 0 in
      while !i < stop && !count < length do
        i := !i + Char_class.at_len (Char_class.at text !i ~stop);
        incr count
      done;
      pieces := (String.sub text start (!i - start), (start, !i)) :: !pieces
    done;
    List.rev !pieces

(* Walks [text] a chunk of spans at a time. [span] places a byte range of [text]
   in the text the caller gave, which a rewrite may have made a longer one, so
   that offsets are always the caller's. *)
let walk_pieces t text ~encode ~span =
  let stop = String.length text in
  let capacity = ref (max 32 (min span_chunk ((stop / 8) + 1))) in
  let spans = ref (Spans.create ~capacity:!capacity) in
  let scratch = ref Bytes.empty in
  let pieces = ref [] in
  let p = ref 0 in
  while !p < stop do
    Spans.clear !spans;
    let resume = fill t text ~pos:!p ~stop !spans in
    let n = Spans.count !spans in
    if n = 0 && resume = !p then begin
      capacity := !capacity * 2;
      spans := Spans.create ~capacity:!capacity
    end
    else begin
      for k = 0 to n - 1 do
        let start = Spans.start !spans k and stop = Spans.stop !spans k in
        let piece =
          if not encode then String.sub text start (stop - start)
          else begin
            let need = (stop - start) * 2 in
            if Bytes.length !scratch < need then scratch := Bytes.create need;
            let buf = !scratch in
            Bytes.sub_string buf 0 (byte_level_blit buf text ~start ~stop)
          end
        in
        pieces := (piece, span ~start ~stop) :: !pieces
      done;
      p := resume
    end
  done;
  List.rev !pieces

let is_one_character s =
  let stop = String.length s in
  stop > 0 && Char_class.at_len (Char_class.at s 0 ~stop) = stop

let opening = Lazy.from_val true
let continuing = Lazy.from_val false

(* The pieces of [inner] over [text] once rewritten, placed in the text the
   caller gave, where [text] starts at [base]; [first] says whether it opens the
   document. *)
let walk_rewritten inner rewrite text ~base ~first ~encode =
  let span ~start ~stop = (base + start, base + stop) in
  match rewriter rewrite ~first text with
  | None -> walk_pieces inner text ~encode ~span
  | Some n ->
      let rewritten, align = Normalizer.apply_aligned n text in
      let span ~start ~stop =
        let start, stop = Normalizer.original_span align ~start ~stop in
        (base + start, base + stop)
      in
      walk_pieces inner rewritten ~encode ~span

(* An empty text has no piece, whichever the pre-tokenizer — a sequence of none
   being the identity, and so the exception. *)
let rec pieces t ~first text =
  if text = "" then match t with Sequence [] -> [ ("", (0, 0)) ] | _ -> []
  else pre_tokenize_planned t ~first text

and verbatim_span ~start ~stop = (start, stop)

and pre_tokenize_planned t ~first text =
  let encode = ends_byte_level t in
  match plan t with
  | Pieces -> pre_tokenize_pieces t ~first text
  | Walk { rewrite; _ } -> walk_rewritten t rewrite text ~base:0 ~first ~encode
  | Segmented { outer; rewrite; inner; _ } ->
      List.concat_map
        (fun (segment, (base, _)) ->
          walk_rewritten inner rewrite segment ~base
            ~first:(if base = 0 then first else continuing)
            ~encode)
        (walk_pieces outer text ~encode:false ~span:verbatim_span)

and pre_tokenize_pieces t ~first text =
  match t with
  | Fixed_length { length } -> pre_tokenize_fixed_length ~length text
  | Sequence ts -> pre_tokenize_sequence ts ~first text
  (* [plan] is [Pieces] for those two alone. *)
  | _ -> assert false

(* Each member cuts the pieces of the one before it and places its own inside
   them, which reads as the caller's text only while a piece still stands byte
   for byte where it was cut from. A member that encodes or marks its pieces
   ends that, and nothing cut from what it hands on can be placed more finely
   than the whole of it. A piece opens the document iff the text does and it
   starts where the text does. *)
and pre_tokenize_sequence ts ~first text =
  let cut =
    List.fold_left
      (fun previous t ->
        let slices = pieces_are_slices t in
        List.concat_map
          (fun (s, ((o_start, _) as span), stands) ->
            let cut =
              pieces t ~first:(if o_start = 0 then first else continuing) s
            in
            if stands then
              List.map
                (fun (p, (p_start, p_end)) ->
                  (p, (o_start + p_start, o_start + p_end), slices))
                cut
            else List.map (fun (p, _) -> (p, span, false)) cut)
          previous)
      [ (text, (0, String.length text), true) ]
      ts
  in
  List.map (fun (p, span, _) -> (p, span)) cut

let pre_tokenize t text = pieces t ~first:opening text

(* Constructors *)

let whitespace () = Whitespace
let whitespace_split () = Whitespace_split
let bert () = Bert

let byte_level ?(add_prefix_space = true) ?(use_regex = true)
    ?(trim_offsets = true) () =
  Byte_level { add_prefix_space; use_regex; trim_offsets }

let punctuation ?(behavior = `Isolated) () = Punctuation { behavior }

let split ~pattern ?(behavior = `Removed) ?(invert = false) () =
  Split { pattern; behavior; invert }

let char_delimiter delimiter =
  if not (is_one_character delimiter) then invalid_arg err_delimiter;
  Char_delimiter delimiter

let digits ?(individual_digits = false) () =
  Digits { individual = individual_digits }

let metaspace ?(replacement = "\xe2\x96\x81") ?(prepend_scheme = `Always)
    ?(split = true) () =
  if not (is_one_character replacement) then invalid_arg err_replacement;
  Metaspace { replacement; prepend_scheme; split }

let unicode_scripts () = Unicode_scripts
let fixed_length n = Fixed_length { length = n }
let sequence ts = Sequence ts

(* Serialization *)

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let behavior_to_string = function
  | `Isolated -> "Isolated"
  | `Removed -> "Removed"
  | `Merged_with_previous -> "MergedWithPrevious"
  | `Merged_with_next -> "MergedWithNext"
  | `Contiguous -> "Contiguous"

let behavior_of_string = function
  | "Isolated" -> Ok `Isolated
  | "Removed" -> Ok `Removed
  | "MergedWithPrevious" -> Ok `Merged_with_previous
  | "MergedWithNext" -> Ok `Merged_with_next
  | "Contiguous" -> Ok `Contiguous
  | other -> Error (err_unknown_behavior other)

let scheme_to_string = function
  | `First -> "first"
  | `Never -> "never"
  | `Always -> "always"

let scheme_of_string = function
  | "first" -> Ok `First
  | "never" -> Ok `Never
  | "always" -> Ok `Always
  | other -> Error (err_unknown_scheme other)

(* Formatting *)

let rec pp ppf = function
  | Byte_level { add_prefix_space; use_regex; trim_offsets } ->
      Format.fprintf ppf
        "@[<1>ByteLevel(add_prefix_space=%b,@ use_regex=%b,@ trim_offsets=%b)@]"
        add_prefix_space use_regex trim_offsets
  | Bert -> Format.pp_print_string ppf "Bert"
  | Whitespace -> Format.pp_print_string ppf "Whitespace"
  | Whitespace_split -> Format.pp_print_string ppf "WhitespaceSplit"
  | Punctuation { behavior } ->
      Format.fprintf ppf "@[<1>Punctuation(%s)@]" (behavior_to_string behavior)
  | Split { pattern; behavior; invert } ->
      Format.fprintf ppf "@[<1>Split(%S,@ %s,@ invert=%b)@]" pattern
        (behavior_to_string behavior)
        invert
  | Char_delimiter delimiter -> Format.fprintf ppf "CharDelimiter(%S)" delimiter
  | Digits { individual } ->
      Format.fprintf ppf "Digits(individual=%b)" individual
  | Metaspace { replacement; prepend_scheme; split } ->
      Format.fprintf ppf "@[<1>Metaspace(%S,@ %s,@ split=%b)@]" replacement
        (scheme_to_string prepend_scheme)
        split
  | Sequence ts ->
      Format.fprintf ppf "@[<1>Sequence[%a]@]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ")
           pp)
        ts
  | Fixed_length { length } -> Format.fprintf ppf "FixedLength(%d)" length
  | Unicode_scripts -> Format.pp_print_string ppf "UnicodeScripts"

let rec to_json = function
  | Byte_level { add_prefix_space; use_regex; trim_offsets } ->
      json_obj
        [
          ("type", Jsont.Json.string "ByteLevel");
          ("add_prefix_space", Jsont.Json.bool add_prefix_space);
          ("use_regex", Jsont.Json.bool use_regex);
          ("trim_offsets", Jsont.Json.bool trim_offsets);
        ]
  | Bert -> json_obj [ ("type", Jsont.Json.string "BertPreTokenizer") ]
  | Whitespace -> json_obj [ ("type", Jsont.Json.string "Whitespace") ]
  | Whitespace_split ->
      json_obj [ ("type", Jsont.Json.string "WhitespaceSplit") ]
  | Punctuation { behavior } ->
      json_obj
        [
          ("type", Jsont.Json.string "Punctuation");
          ("behavior", Jsont.Json.string (behavior_to_string behavior));
        ]
  | Split { pattern; behavior; invert } ->
      json_obj
        [
          ("type", Jsont.Json.string "Split");
          ("pattern", json_obj [ ("String", Jsont.Json.string pattern) ]);
          ("behavior", Jsont.Json.string (behavior_to_string behavior));
          ("invert", Jsont.Json.bool invert);
        ]
  | Char_delimiter delimiter ->
      json_obj
        [
          ("type", Jsont.Json.string "CharDelimiterSplit");
          ("delimiter", Jsont.Json.string delimiter);
        ]
  | Digits { individual } ->
      json_obj
        [
          ("type", Jsont.Json.string "Digits");
          ("individual_digits", Jsont.Json.bool individual);
        ]
  | Metaspace { replacement; prepend_scheme; split } ->
      json_obj
        [
          ("type", Jsont.Json.string "Metaspace");
          ("replacement", Jsont.Json.string replacement);
          ("prepend_scheme", Jsont.Json.string (scheme_to_string prepend_scheme));
          ("split", Jsont.Json.bool split);
        ]
  | Sequence ts ->
      json_obj
        [
          ("type", Jsont.Json.string "Sequence");
          ("pretokenizers", Jsont.Json.list (List.map to_json ts));
        ]
  | Fixed_length { length } ->
      json_obj
        [
          ("type", Jsont.Json.string "FixedLength");
          ("length", Jsont.Json.int length);
        ]
  | Unicode_scripts -> json_obj [ ("type", Jsont.Json.string "UnicodeScripts") ]

let find_field name fields = Option.map snd (Jsont.Json.find_mem name fields)

let bool_field name default fields =
  match find_field name fields with
  | Some (Jsont.Bool (b, _)) -> b
  | Some (Jsont.Number (f, _)) -> int_of_float f <> 0
  | Some (Jsont.String (s, _)) -> (
      match String.lowercase_ascii s with
      | "true" | "1" -> true
      | "false" | "0" -> false
      | _ -> default)
  | _ -> default

let int_field name default fields =
  match find_field name fields with
  | Some (Jsont.Number (f, _)) -> int_of_float f
  | Some (Jsont.String (s, _)) -> (
      match int_of_string_opt s with Some v -> v | None -> default)
  | _ -> default

(* HuggingFace tags a split pattern with the way it is to be matched. *)
let split_pattern_of_json = function
  | Jsont.Object (fields, _) -> (
      match (find_field "String" fields, find_field "Regex" fields) with
      | Some (Jsont.String (pattern, _)), _ -> Ok pattern
      | _, Some _ -> Error err_split_regex
      | _ -> Error err_split_pattern)
  | _ -> Error err_split_pattern

let scheme_field fields =
  match find_field "prepend_scheme" fields with
  | None -> Ok `Always
  | Some (Jsont.String (scheme, _)) -> scheme_of_string scheme
  | Some _ -> Error err_metaspace_scheme

let rec of_json = function
  | Jsont.Object (fields, _) -> (
      match find_field "type" fields with
      | Some (Jsont.String ("ByteLevel", _)) ->
          let add_prefix_space = bool_field "add_prefix_space" true fields in
          let use_regex = bool_field "use_regex" true fields in
          let trim_offsets = bool_field "trim_offsets" true fields in
          Ok (Byte_level { add_prefix_space; use_regex; trim_offsets })
      | Some (Jsont.String ("BertPreTokenizer", _)) -> Ok Bert
      | Some (Jsont.String ("Whitespace", _)) -> Ok Whitespace
      | Some (Jsont.String ("WhitespaceSplit", _)) -> Ok Whitespace_split
      | Some (Jsont.String ("Punctuation", _)) -> (
          match find_field "behavior" fields with
          | None -> Ok (Punctuation { behavior = `Isolated })
          | Some (Jsont.String (s, _)) ->
              Result.map
                (fun b -> Punctuation { behavior = b })
                (behavior_of_string s)
          | Some _ -> Error err_missing_behavior)
      | Some (Jsont.String ("Split", _)) -> (
          match (find_field "pattern" fields, find_field "behavior" fields) with
          | Some pattern, Some (Jsont.String (behavior_str, _)) ->
              Result.bind (split_pattern_of_json pattern) (fun pattern ->
                  Result.map
                    (fun behavior ->
                      let invert = bool_field "invert" false fields in
                      Split { pattern; behavior; invert })
                    (behavior_of_string behavior_str))
          | _ -> Error err_split_missing)
      | Some (Jsont.String ("CharDelimiterSplit", _)) -> (
          match find_field "delimiter" fields with
          | Some (Jsont.String (delimiter, _)) when is_one_character delimiter
            ->
              Ok (Char_delimiter delimiter)
          | _ -> Error err_char_delim_missing)
      | Some (Jsont.String ("Digits", _)) ->
          let individual = bool_field "individual_digits" false fields in
          Ok (Digits { individual })
      | Some (Jsont.String ("Metaspace", _)) -> (
          match find_field "replacement" fields with
          | Some (Jsont.String (repl, _)) when is_one_character repl ->
              Result.map
                (fun prepend_scheme ->
                  let split = bool_field "split" true fields in
                  Metaspace { replacement = repl; prepend_scheme; split })
                (scheme_field fields)
          | _ -> Error err_metaspace_missing)
      | Some (Jsont.String ("Sequence", _)) -> (
          match find_field "pretokenizers" fields with
          | Some (Jsont.Array (elements, _)) ->
              let rec build acc = function
                | [] -> Ok (Sequence (List.rev acc))
                | item :: rest -> (
                    match of_json item with
                    | Ok t -> build (t :: acc) rest
                    | Error _ as e -> e)
              in
              build [] elements
          | _ -> Error err_sequence_missing)
      | Some (Jsont.String ("FixedLength", _)) ->
          let length = int_field "length" 0 fields in
          if length <= 0 then Error err_fixed_length
          else Ok (Fixed_length { length })
      | Some (Jsont.String ("UnicodeScripts", _)) -> Ok Unicode_scripts
      | Some (Jsont.String (other, _)) -> Error (err_unsupported_type other)
      | _ -> Error err_missing_type)
  | _ -> Error err_expected_object
