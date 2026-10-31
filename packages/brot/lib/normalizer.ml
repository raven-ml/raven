(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Errors *)

let err_expected_object = "expected JSON object"
let err_missing_type = "missing type field"
let err_replace_invalid_pattern = "invalid pattern"
let err_replace_missing_pattern = "missing pattern"
let err_replace_missing_content = "missing content"
let err_prepend_missing = "missing prepend field"
let err_sequence_missing = "missing normalizers"
let err_charsmap_missing = "missing precompiled_charsmap"
let err_charsmap_base64 = "precompiled_charsmap is not base64"
let strf = Printf.sprintf

(* Type *)

(* A SentencePiece [precompiled_charsmap]: a little-endian [u32], the size in
   bytes of a darts-clone double-array trie that follows it as little-endian
   [u32] units, then the replacements, each ended by a NUL. The trie maps the
   bytes of a rule's source to the offset of its replacement. [ascii.(b)] is the
   replacement offset of the source made of the one byte [b], or [-1]. *)
type charsmap = {
  blob : string;
  units : int;
  replacements : string;
  ascii : int array;
}

type t =
  | Bert of {
      clean_text : bool;
      handle_chinese_chars : bool;
      strip_accents : bool option;
      lowercase : bool;
    }
  | Strip of { left : bool; right : bool }
  | Strip_accents
  | NFC
  | NFD
  | NFKC
  | NFKD
  | Lowercase
  | Replace of { pattern : pattern; replacement : string }
  | Prepend of string
  | Byte_level
  | Nmt
  | Precompiled of charsmap
  | Sequence of t list

and pattern =
  | Literal of string
  | Regex of { source : string; compiled : Regex.t }

(* UTF-8 helpers *)

(* Returns (codepoint lsl 3) lor byte_length — zero allocation. *)
let[@inline] utf8_next s i =
  let d = String.get_utf_8_uchar s i in
  (Uchar.to_int (Uchar.utf_decode_uchar d) lsl 3) lor Uchar.utf_decode_length d

let[@inline] is_continuation s i =
  Char.code (String.unsafe_get s i) land 0xC0 = 0x80

let[@inline] is_ascii s i = Char.code (String.unsafe_get s i) < 0x80

(* Character classification *)

let[@inline] is_whitespace code =
  code = 0x09 || code = 0x0A || code = 0x0D || code = 0x20
  || Uucp.White.is_white_space (Uchar.of_int code)

(* Unassigned codepoints are not controls: they survive cleaning and reach the
   model, which maps them to its unknown token. *)
let[@inline] is_control code =
  if code = 0x09 || code = 0x0A || code = 0x0D then false
  else
    match Uucp.Gc.general_category (Uchar.of_int code) with
    | `Cc | `Cf | `Co -> true
    | _ -> false

(* Byte scanning, eight bytes at a time *)

external word : string -> int -> int64 = "%caml_string_get64u"

let ones = 0x0101010101010101L
let highs = 0x8080808080808080L

(* [true] iff a byte of [w] is zero. *)
let[@inline] has_zero w =
  Int64.logand (Int64.logand (Int64.sub w ones) (Int64.lognot w)) highs <> 0L

let lows = 0x7F7F7F7F7F7F7F7FL

(* The number of zero bytes of [w]: the high bit of every zero byte and no other
   is set, then those bits are summed into the top byte. *)
let[@inline] zero_bytes w =
  let marks =
    Int64.logand
      (Int64.lognot (Int64.logor (Int64.add (Int64.logand w lows) lows) w))
      highs
  in
  Int64.to_int
    (Int64.shift_right_logical
       (Int64.mul (Int64.shift_right_logical marks 7) ones)
       56)

let rec has_non_ascii_from s i =
  let len = String.length s in
  if i + 8 <= len then
    Int64.logand (word s i) highs <> 0L || has_non_ascii_from s (i + 8)
  else i < len && ((not (is_ascii s i)) || has_non_ascii_from s (i + 1))

(* Text that is all ASCII holds no mark, no ideograph and nothing to decompose,
   so the stages that only touch those can be skipped whole. *)
let has_non_ascii s = has_non_ascii_from s 0

let needs_lowering s =
  let len = String.length s in
  let rec loop i =
    i < len
    &&
    let byte = Char.code (String.unsafe_get s i) in
    (byte >= 0x41 && byte <= 0x5A) || byte >= 128 || loop (i + 1)
  in
  loop 0

(* The number of bytes [c] in [s]. *)
let count_byte s c =
  let len = String.length s in
  let pattern = Int64.mul (Int64.of_int (Char.code c)) ones in
  let count = ref 0 and i = ref 0 in
  while !i + 8 <= len do
    count := !count + zero_bytes (Int64.logxor (word s !i) pattern);
    i := !i + 8
  done;
  while !i < len do
    if String.unsafe_get s !i = c then incr count;
    incr i
  done;
  !count

(* The first byte [c] of [s] at or after [pos], or [-1]. *)
let index_byte s c pos =
  let len = String.length s in
  let pattern = Int64.mul (Int64.of_int (Char.code c)) ones in
  let i = ref pos and found = ref (-1) in
  while !found < 0 && !i + 8 <= len do
    if has_zero (Int64.logxor (word s !i) pattern) then begin
      while String.unsafe_get s !i <> c do
        incr i
      done;
      found := !i
    end
    else i := !i + 8
  done;
  if !found < 0 then begin
    while !i < len && String.unsafe_get s !i <> c do
      incr i
    done;
    if !i < len then found := !i
  end;
  !found

let[@inline] is_chinese_char code =
  (code >= 0x4E00 && code <= 0x9FFF)
  || (code >= 0x3400 && code <= 0x4DBF)
  || (code >= 0x20000 && code <= 0x2A6DF)
  || (code >= 0x2A700 && code <= 0x2B73F)
  || (code >= 0x2B740 && code <= 0x2B81F)
  || (code >= 0x2B920 && code <= 0x2CEAF)
  || (code >= 0xF900 && code <= 0xFAFF)
  || (code >= 0x2F800 && code <= 0x2FA1F)

(* Alignment *)

(* Every byte of the normalized text stands for a range of the original, and all
   the bytes of a character stand for the same one; a span of normalized bytes
   reports the range of its first byte joined with the range of its last.
   Insertions stand for the range of the character they were put next to,
   deletions for nothing at all, so the ranges are ascending but need not tile
   the original.

   A range is one [int], its start in the high half and its stop in the low one,
   which holds texts up to 2 GiB, and a map keeps one per byte in a byte string,
   which the collector does not scan, so a lookup is one load. Text that was
   only prefixed or sliced keeps a view on the alignment of what it came from
   rather than a copy of it: byte [i] of the text stands for what byte [anchor]
   of the base stands for while [i < head], and byte [i + shift] after that.
   Views flatten, so the base of a view is never a view. *)
type alignment =
  | Identity of string
  | Map of { spans : Bytes.t; count : int; original : int }
  | View of {
      head : int;
      anchor : int;
      shift : int;
      count : int;
      base : alignment;
    }

let[@inline] pack high low = (high lsl 32) lor low
let[@inline] high p = p lsr 32
let[@inline] low p = p land 0xFFFF_FFFF

external span_get64 : Bytes.t -> int -> int64 = "%caml_bytes_get64u"
external span_set64 : Bytes.t -> int -> int64 -> unit = "%caml_bytes_set64u"

let[@inline] span_get spans i = Int64.to_int (span_get64 spans (i lsl 3))
let[@inline] span_set spans i r = span_set64 spans (i lsl 3) (Int64.of_int r)
let spans_create n = Bytes.create (n lsl 3)

let spans_blit src pos dst at n =
  Bytes.blit src (pos lsl 3) dst (at lsl 3) (n lsl 3)

let identity s = Identity s

let normalized_length = function
  | Identity s -> String.length s
  | Map m -> m.count
  | View v -> v.count

let rec original_length = function
  | Identity s -> String.length s
  | Map m -> m.original
  | View v -> original_length v.base

(* A character of text that was not normalized is a byte that is not a
   continuation byte and the continuation bytes after it. *)
let identity_range s i =
  let len = String.length s in
  let start = ref i in
  while !start > 0 && is_continuation s !start do
    decr start
  done;
  let stop = ref (i + 1) in
  while !stop < len && is_continuation s !stop do
    incr stop
  done;
  pack !start !stop

(* The range of the character byte [i] falls in. Byte [-1] is what a character
   inserted before anything stands for. *)
let rec range_through a i =
  match a with
  | Identity s -> identity_range s i
  | Map m -> span_get m.spans i
  | View v ->
      range_through v.base (if i < v.head then v.anchor else i + v.shift)

let[@inline] range a i =
  if i < 0 then 0
  else
    match a with
    | Identity s -> identity_range s i
    | Map m -> span_get m.spans i
    | View _ -> range_through a i

(* The range of ASCII byte [i] of a stage's input, which is the original text
   when the alignment is the identity: the byte itself, and any continuation
   bytes that follow it. *)
let[@inline] ascii_range a i =
  match a with
  | Identity s ->
      if i + 1 < String.length s && is_continuation s (i + 1) then
        identity_range s i
      else pack i (i + 1)
  | Map m -> span_get m.spans i
  | View _ -> range a i

(* Entries [at] onwards of [spans] receive the ranges of bytes [pos] to [pos + n
   - 1] of [a]: a blit from a map, and from the identity a walk that only stops
   on the bytes of a multi-byte character. *)
let rec fill spans at a pos n =
  match a with
  | Map m -> spans_blit m.spans pos spans at n
  | Identity s ->
      let len = String.length s in
      let r = ref 0 in
      for k = 0 to n - 1 do
        let j = pos + k in
        let b = Char.code (String.unsafe_get s j) in
        if k > 0 && b land 0xC0 = 0x80 then ()
        else if b < 0x80 && (j + 1 = len || not (is_continuation s (j + 1)))
        then r := pack j (j + 1)
        else r := identity_range s j;
        span_set spans (at + k) !r
      done
  | View v ->
      let h = min n (max 0 (v.head - pos)) in
      if h > 0 then begin
        let r = range v.base v.anchor in
        for k = at to at + h - 1 do
          span_set spans k r
        done
      end;
      if n > h then fill spans (at + h) v.base (pos + h + v.shift) (n - h)

let prefixed a prefix =
  match a with
  | View v ->
      View
        {
          head = prefix + v.head;
          anchor = (if v.head > 0 then v.anchor else v.shift);
          shift = v.shift - prefix;
          count = prefix + v.count;
          base = v.base;
        }
  | base ->
      View
        {
          head = prefix;
          anchor = 0;
          shift = -prefix;
          count = prefix + normalized_length base;
          base;
        }

let sliced a offset count =
  match a with
  | View v ->
      View
        {
          head = max 0 (v.head - offset);
          anchor = v.anchor;
          shift = offset + v.shift;
          count;
          base = v.base;
        }
  | base -> View { head = 0; anchor = 0; shift = offset; count; base }

let original_span a ~start ~stop =
  let len = normalized_length a in
  if start < 0 || stop < start || stop > len then
    invalid_arg
      (strf "%d,%d is not a span of the %d normalized bytes" start stop len);
  (* Normalizing the text away leaves nothing to map through, so the only span
     there is stands for the whole of what was normalized. *)
  if len = 0 then (0, original_length a)
  else if start < stop then (high (range a start), low (range a (stop - 1)))
  else
    let at = if start < len then high (range a start) else original_length a in
    (at, at)

(* Output *)

(* A stage's text and, while tracking, the range of the original every output
   byte stands for, looked up through the alignment of the stage's input as the
   byte is written. [apply] runs untracked, so the only cost it pays is the test
   of [track]. *)
type out = {
  mutable bytes : Bytes.t;
  mutable spans : Bytes.t;
  mutable len : int;
  base : alignment;
  track : bool;
}

let out ~track base n =
  let cap = max n 16 in
  {
    bytes = Bytes.create cap;
    spans = (if track then spans_create cap else Bytes.empty);
    len = 0;
    base;
    track;
  }

let grow o n =
  let cap = max (o.len + n) (2 * Bytes.length o.bytes) in
  let bytes = Bytes.create cap in
  Bytes.blit o.bytes 0 bytes 0 o.len;
  o.bytes <- bytes;
  if o.track then begin
    let spans = spans_create cap in
    spans_blit o.spans 0 spans 0 o.len;
    o.spans <- spans
  end

let[@inline] ensure o n = if o.len + n > Bytes.length o.bytes then grow o n

(* The next [n] output bytes stand for input byte [src]. *)
let[@inline] mark o src n =
  if o.track then begin
    let r = range o.base src in
    for k = o.len to o.len + n - 1 do
      span_set o.spans k r
    done
  end

let[@inline] add_char o src c =
  ensure o 1;
  Bytes.unsafe_set o.bytes o.len c;
  mark o src 1;
  o.len <- o.len + 1

let add_code o src u =
  ensure o 4;
  let b = o.bytes and p = o.len in
  let n =
    if u < 0x80 then begin
      Bytes.unsafe_set b p (Char.unsafe_chr u);
      1
    end
    else if u < 0x800 then begin
      Bytes.unsafe_set b p (Char.unsafe_chr (0xC0 lor (u lsr 6)));
      Bytes.unsafe_set b (p + 1) (Char.unsafe_chr (0x80 lor (u land 0x3F)));
      2
    end
    else if u < 0x10000 then begin
      Bytes.unsafe_set b p (Char.unsafe_chr (0xE0 lor (u lsr 12)));
      Bytes.unsafe_set b (p + 1)
        (Char.unsafe_chr (0x80 lor ((u lsr 6) land 0x3F)));
      Bytes.unsafe_set b (p + 2) (Char.unsafe_chr (0x80 lor (u land 0x3F)));
      3
    end
    else begin
      Bytes.unsafe_set b p (Char.unsafe_chr (0xF0 lor (u lsr 18)));
      Bytes.unsafe_set b (p + 1)
        (Char.unsafe_chr (0x80 lor ((u lsr 12) land 0x3F)));
      Bytes.unsafe_set b (p + 2)
        (Char.unsafe_chr (0x80 lor ((u lsr 6) land 0x3F)));
      Bytes.unsafe_set b (p + 3) (Char.unsafe_chr (0x80 lor (u land 0x3F)));
      4
    end
  in
  mark o src n;
  o.len <- p + n

let rec add_codes o src = function
  | [] -> ()
  | u :: us ->
      add_code o src (Uchar.to_int u);
      add_codes o src us

let add_sub o src s pos n =
  ensure o n;
  Bytes.blit_string s pos o.bytes o.len n;
  mark o src n;
  o.len <- o.len + n

let add_string o src str = add_sub o src str 0 (String.length str)

(* Bytes kept as they are, each standing for itself. *)
let copy o s pos n =
  ensure o n;
  Bytes.blit_string s pos o.bytes o.len n;
  if o.track then fill o.spans o.len o.base pos n;
  o.len <- o.len + n

(* The ASCII bytes of [s] from [i] on, each standing for itself and becoming
   [table.(byte)], or nothing when that is negative; stops at the first byte
   that is not ASCII, or, with [lookahead], at the last one before it. Returns
   where it stopped. The stage's state is held in locals through the loop, which
   is what makes this the fast lane. *)
let ascii_run o s i ~lookahead table =
  let len = String.length s in
  ensure o (len - i);
  let bytes = o.bytes
  and spans = o.spans
  and base = o.base
  and track = o.track in
  let p = ref o.len and j = ref i in
  while
    !j < len && is_ascii s !j
    && ((not lookahead) || !j + 1 = len || is_ascii s (!j + 1))
  do
    let c = Array.unsafe_get table (Char.code (String.unsafe_get s !j)) in
    if c >= 0 then begin
      Bytes.unsafe_set bytes !p (Char.unsafe_chr c);
      if track then span_set spans !p (ascii_range base !j);
      incr p
    end;
    incr j
  done;
  o.len <- !p;
  !j

let ascii_identity = Array.init 128 (fun b -> b)

let text o =
  if o.len = Bytes.length o.bytes then Bytes.unsafe_to_string o.bytes
  else Bytes.sub_string o.bytes 0 o.len

let compose a o =
  Map { spans = o.spans; count = o.len; original = original_length a }

(* Unicode normalization *)

(* Hangul composition is arithmetic. It is done here rather than left to
   [Uunf.composite], whose trailing jamo bound in uunf 17.0.0 is one too high
   (U+11C3, a plain starter, composes as a T). *)
let hangul_l = 0x1100
let hangul_v = 0x1161
let hangul_t = 0x11A7
let hangul_s = 0xAC00
let hangul_v_count = 21
let hangul_t_count = 28

(* The primary composite of [starter] and [scalar], or [-1]. *)
let composite starter scalar =
  if starter >= hangul_l && starter < hangul_l + 19 then
    if scalar >= hangul_v && scalar < hangul_v + hangul_v_count then
      hangul_s
      + (((starter - hangul_l) * hangul_v_count) + (scalar - hangul_v))
        * hangul_t_count
    else -1
  else if starter >= hangul_s && starter <= 0xD7A3 then
    if
      (starter - hangul_s) mod hangul_t_count = 0
      && scalar > hangul_t
      && scalar < hangul_t + hangul_t_count
    then starter + scalar - hangul_t
    else -1
  else
    match
      Uunf.composite (Uchar.unsafe_of_int starter) (Uchar.unsafe_of_int scalar)
    with
    | Some c -> Uchar.to_int c
    | None -> -1

(* A value computed on first use and kept. A domain that finds the cell empty
   while another is filling it computes the same value again, harmlessly. *)
let once f =
  let cell = ref None in
  fun () ->
    match !cell with
    | Some v -> v
    | None ->
        let v = f () in
        cell := Some v;
        v

(* Three bitmaps, one bit per code point below [bits_limit], computed once from
   the decompositions. [seconds]: the code points that can be the second
   character of a primary composite, that is the second of a two-character
   canonical decomposition, and the vowel and trailing jamo. [stable_nfc] and
   [stable_nfkc]: the starters that are no second and normalize to themselves,
   so that one followed by another comes out as it is, without decomposing.
   Nothing decomposes into two characters above the Musical Symbols block
   (U+1D1C0); the limit covers the CJK ideographs above, whose compatibility
   forms normalize to something else. *)
type bits = { seconds : Bytes.t; stable_nfc : Bytes.t; stable_nfkc : Bytes.t }

let bits_limit = 0x30000

let[@inline] bit bits u =
  u < bits_limit
  && Char.code (Bytes.unsafe_get bits (u lsr 3)) land (1 lsl (u land 7)) <> 0

let set_bit bits u =
  let i = u lsr 3 in
  Bytes.unsafe_set bits i
    (Char.unsafe_chr
       (Char.code (Bytes.unsafe_get bits i) lor (1 lsl (u land 7))))

let bits =
  once (fun () ->
      let seconds = Bytes.make (bits_limit lsr 3) '\000' in
      let stable_nfc = Bytes.make (bits_limit lsr 3) '\000' in
      let stable_nfkc = Bytes.make (bits_limit lsr 3) '\000' in
      let hangul u = u >= hangul_s && u <= 0xD7A3 in
      for u = 0 to bits_limit - 1 do
        if Uchar.is_valid u && not (hangul u) then begin
          let d = Uunf.decomp (Uchar.unsafe_of_int u) in
          if Array.length d = 2 && not (Uunf.d_compatibility d.(0)) then
            set_bit seconds d.(1)
        end
      done;
      for u = hangul_v to hangul_v + hangul_v_count - 1 do
        set_bit seconds u
      done;
      for u = hangul_t + 1 to hangul_t + hangul_t_count - 1 do
        set_bit seconds u
      done;
      let rec self ~compat u =
        hangul u
        ||
        let d = Uunf.decomp (Uchar.unsafe_of_int u) in
        Array.length d = 0
        || ((not compat) && Uunf.d_compatibility d.(0))
        || Array.length d = 2
           &&
           let a = Uchar.to_int (Uunf.d_uchar d.(0)) in
           self ~compat a && self ~compat d.(1) && composite a d.(1) = u
      in
      for u = 0 to bits_limit - 1 do
        if
          Uchar.is_valid u
          && Uunf.ccc (Uchar.unsafe_of_int u) = 0
          && not (bit seconds u)
        then begin
          if self ~compat:false u then set_bit stable_nfc u;
          if self ~compat:true u then set_bit stable_nfkc u
        end
      done;
      { seconds; stable_nfc; stable_nfkc })

(* Normalization tracks alignments by decomposing every character on its own:
   the first character of a decomposition stands for its source, the rest stand
   for nothing. Canonical ordering then permutes a run of marks, carrying that
   with it, so a character can end up standing for a source other than the one
   it came from; composition adds up what it folds together. This is the
   accounting HuggingFace reports offsets with.

   The text is normalized as it is read. Characters are decomposed; the marks
   join a run, which is canonically ordered and handed to the composer once a
   starter arrives, since nothing after a starter moves before it, and the
   starter follows; what comes out of the composer goes to [sink] with the
   source byte it stands for: a character standing for one or more source
   characters takes the first byte of the first of them, one standing for none
   takes the last byte of the last source character accounted for. *)
type nf = {
  compat : bool;
  compose : bool;
  seconds : Bytes.t;
  stable : Bytes.t;
  sink : int -> int -> unit;
  (* a stable starter waiting for the next character: it comes out as it is if
     that is stable too, and goes through the decomposer otherwise *)
  mutable plain : int;
  (* the run of marks: scalars, combining classes, source characters each stands
     for *)
  mutable scalars : int array;
  mutable classes : int array;
  mutable stands : int array;
  mutable count : int;
  (* the composer: the starter what follows may compose into, [-1] for none, and
     the marks blocked from it, which come out after it *)
  mutable starter : int;
  mutable starter_stands : int;
  mutable last_class : int;
  mutable held : int array;
  mutable held_stands : int array;
  mutable held_count : int;
  (* first and last byte of the source characters not accounted for yet, and the
     last byte of the last one that was *)
  mutable sources : int array;
  mutable head : int;
  mutable tail : int;
  mutable last : int;
}

let nf ~compat ~compose sink =
  {
    compat;
    compose;
    seconds = (if compose then (bits ()).seconds else Bytes.empty);
    stable =
      (if not compose then Bytes.empty
       else if compat then (bits ()).stable_nfkc
       else (bits ()).stable_nfc);
    sink;
    plain = -1;
    scalars = Array.make 16 0;
    classes = Array.make 16 0;
    stands = Array.make 16 0;
    count = 0;
    starter = -1;
    starter_stands = 0;
    last_class = -1;
    held = Array.make 16 0;
    held_stands = Array.make 16 0;
    held_count = 0;
    sources = Array.make 16 0;
    head = 0;
    tail = 0;
    last = -1;
  }

let grown a =
  let b = Array.make (2 * Array.length a) 0 in
  Array.blit a 0 b 0 (Array.length a);
  b

(* The source byte a character standing for [stands] source characters stands
   for, those characters accounted for. *)
let account nf stands =
  if stands = 0 then nf.last
  else begin
    let first = high (Array.unsafe_get nf.sources nf.head) in
    nf.head <- nf.head + stands;
    nf.last <- low (Array.unsafe_get nf.sources (nf.head - 1));
    if nf.head = nf.tail then begin
      nf.head <- 0;
      nf.tail <- 0
    end;
    first
  end

let emit nf scalar stands = nf.sink scalar (account nf stands)

(* Composition folds a starter and the marks that reach it into one character
   standing for all of them; marks blocked from the starter are held back and
   come out after it. *)
let hold nf scalar stands cls =
  if nf.held_count = Array.length nf.held then begin
    nf.held <- grown nf.held;
    nf.held_stands <- grown nf.held_stands
  end;
  nf.held.(nf.held_count) <- scalar;
  nf.held_stands.(nf.held_count) <- stands;
  nf.held_count <- nf.held_count + 1;
  nf.last_class <- cls

let release nf =
  emit nf nf.starter nf.starter_stands;
  for i = 0 to nf.held_count - 1 do
    emit nf nf.held.(i) nf.held_stands.(i)
  done;
  nf.held_count <- 0;
  nf.starter <- -1;
  nf.last_class <- -1

let feed nf scalar stands cls =
  if not nf.compose then emit nf scalar stands
  else if nf.starter < 0 then
    if cls <> 0 then emit nf scalar stands
    else begin
      nf.starter <- scalar;
      nf.starter_stands <- stands
    end
  else
    let c =
      if nf.last_class >= cls || not (bit nf.seconds scalar) then -1
      else composite nf.starter scalar
    in
    if c >= 0 then begin
      nf.starter <- c;
      nf.starter_stands <- nf.starter_stands + stands
    end
    else if cls <> 0 then hold nf scalar stands cls
    else begin
      release nf;
      nf.starter <- scalar;
      nf.starter_stands <- stands
    end

(* Insertion sort: stable, as canonical ordering must be, and a run of marks is
   short. *)
let order nf =
  for i = 1 to nf.count - 1 do
    let scalar = nf.scalars.(i)
    and cls = nf.classes.(i)
    and stands = nf.stands.(i) in
    let j = ref (i - 1) in
    while !j >= 0 && nf.classes.(!j) > cls do
      nf.scalars.(!j + 1) <- nf.scalars.(!j);
      nf.classes.(!j + 1) <- nf.classes.(!j);
      nf.stands.(!j + 1) <- nf.stands.(!j);
      decr j
    done;
    nf.scalars.(!j + 1) <- scalar;
    nf.classes.(!j + 1) <- cls;
    nf.stands.(!j + 1) <- stands
  done

let finish_run nf =
  order nf;
  for i = 0 to nf.count - 1 do
    feed nf nf.scalars.(i) nf.stands.(i) nf.classes.(i)
  done;
  nf.count <- 0

(* A starter never moves, so it ends the run of marks before it and goes
   straight to the composer; a mark joins the run. *)
let push nf scalar stands =
  let cls = Uunf.ccc (Uchar.unsafe_of_int scalar) in
  if cls = 0 then begin
    if nf.count > 0 then finish_run nf;
    feed nf scalar stands 0
  end
  else begin
    if nf.count = Array.length nf.scalars then begin
      nf.scalars <- grown nf.scalars;
      nf.classes <- grown nf.classes;
      nf.stands <- grown nf.stands
    end;
    nf.scalars.(nf.count) <- scalar;
    nf.classes.(nf.count) <- cls;
    nf.stands.(nf.count) <- stands;
    nf.count <- nf.count + 1
  end

let flush_run nf =
  if nf.count > 0 then finish_run nf;
  if nf.starter >= 0 then release nf

let rec push_decomposed nf scalar stands =
  let d = Uunf.decomp (Uchar.unsafe_of_int scalar) in
  if Array.length d = 0 || ((not nf.compat) && Uunf.d_compatibility d.(0)) then
    push nf scalar stands
  else begin
    push_decomposed nf (Uchar.to_int (Uunf.d_uchar d.(0))) stands;
    for i = 1 to Array.length d - 1 do
      push_decomposed nf d.(i) 0
    done
  end

(* A source character, bytes [first] to [last] of the input. *)
let push_char nf scalar ~first ~last =
  if nf.tail = Array.length nf.sources then
    begin if nf.head > 0 then begin
      Array.blit nf.sources nf.head nf.sources 0 (nf.tail - nf.head);
      nf.tail <- nf.tail - nf.head;
      nf.head <- 0
    end
    else nf.sources <- grown nf.sources
    end;
  nf.sources.(nf.tail) <- pack first last;
  nf.tail <- nf.tail + 1;
  if nf.compose && bit nf.stable scalar then begin
    if nf.plain >= 0 then emit nf nf.plain 1
    else if nf.count > 0 || nf.starter >= 0 then flush_run nf;
    nf.plain <- scalar
  end
  else begin
    if nf.plain >= 0 then begin
      let plain = nf.plain in
      nf.plain <- -1;
      push_decomposed nf plain 1
    end;
    if scalar < 0x80 then push nf scalar 1 else push_decomposed nf scalar 1
  end

let pending nf = nf.plain >= 0 || nf.count > 0 || nf.starter >= 0

(* Everything read comes out: what follows is a starter that nothing composes
   into, or the end. *)
let flush nf =
  if nf.plain >= 0 then begin
    emit nf nf.plain 1;
    nf.plain <- -1
  end;
  flush_run nf

let normalize_text form s o =
  let compat = match form with `NFKC | `NFKD -> true | `NFC | `NFD -> false in
  let compose =
    match form with `NFC | `NFKC -> true | `NFD | `NFKD -> false
  in
  let nf = nf ~compat ~compose (fun scalar src -> add_code o src scalar) in
  let len = String.length s in
  let i = ref 0 in
  while !i < len do
    if is_ascii s !i then begin
      (* An ASCII byte is a starter nothing composes into, so what came before
         it is complete; and it neither decomposes nor composes with an ASCII
         neighbour, so with ASCII after it it comes out as it is. *)
      if pending nf then flush nf;
      let j = ascii_run o s !i ~lookahead:compose ascii_identity in
      if j > !i then begin
        nf.last <- j - 1;
        i := j
      end
      else begin
        push_char nf (Char.code (String.unsafe_get s !i)) ~first:!i ~last:!i;
        incr i
      end
    end
    else begin
      let d = String.get_utf_8_uchar s !i in
      let n = Uchar.utf_decode_length d in
      push_char nf
        (Uchar.to_int (Uchar.utf_decode_uchar d))
        ~first:!i
        ~last:(!i + n - 1);
      i := !i + n
    end
  done;
  flush nf;
  text o

(* Text transforms *)

let ascii_lower =
  Array.init 128 (fun b -> if b >= 0x41 && b <= 0x5A then b + 32 else b)

(* The full Unicode lowercase mapping, which is not case folding: ["ß"] and
   ["ﬁ"] lowercase to themselves but fold to ["ss"] and ["fi"]. *)
let lowercase_text s o =
  let len = String.length s in
  let i = ref 0 in
  while !i < len do
    if is_ascii s !i then i := ascii_run o s !i ~lookahead:false ascii_lower
    else begin
      let d = String.get_utf_8_uchar s !i in
      let n = Uchar.utf_decode_length d in
      (if Uchar.utf_decode_is_valid d then
         match Uucp.Case.Map.to_lower (Uchar.utf_decode_uchar d) with
         | `Self -> add_sub o !i s !i n
         | `Uchars us -> add_codes o !i us);
      i := !i + n
    end
  done;
  text o

(* Every mark goes: spacing ([Mc]) and enclosing ([Me]) as well as the accents
   proper ([Mn]). *)
let drop_marks s o =
  let len = String.length s in
  let i = ref 0 in
  while !i < len do
    if is_ascii s !i then i := ascii_run o s !i ~lookahead:false ascii_identity
    else begin
      let d = String.get_utf_8_uchar s !i in
      let n = Uchar.utf_decode_length d in
      (if Uchar.utf_decode_is_valid d then
         match Uucp.Gc.general_category (Uchar.utf_decode_uchar d) with
         | `Mn | `Mc | `Me -> ()
         | _ -> add_sub o !i s !i n);
      i := !i + n
    end
  done;
  text o

(* BERT *)

(* What an ASCII byte becomes: the byte to write, or [-1] to drop it. *)
let bert_ascii ~clean_text ~lowercase =
  Array.init 128 (fun b ->
      let b =
        if not clean_text then b
        else if b = 9 || b = 10 || b = 13 || b = 32 then 32
        else if b >= 33 && b < 127 then b
        else -1
      in
      if lowercase && b >= 0x41 && b <= 0x5A then b + 32 else b)

let bert_tables =
  Array.init 4 (fun k ->
      bert_ascii ~clean_text:(k land 1 = 1) ~lowercase:(k land 2 = 2))

(* Cleaning, spacing of ideographs, decomposition with the nonspacing marks
   dropped, and lowercasing, in that order for every character, so that the text
   and its alignment are those of the four passes. ASCII is decided by a table
   lookup; other characters go through the Unicode tables, and through the
   decomposer when accents are stripped, in which case an ASCII byte that is
   kept, being a starter, first flushes the run of marks before it. A dropped
   character leaves the run open, as it is not there for the marks after it to
   see. *)
let bert_text ~clean_text ~handle_chinese_chars ~strip_accents ~lowercase s o =
  let table =
    bert_tables.((if clean_text then 1 else 0) lor if lowercase then 2 else 0)
  in
  let sink scalar src =
    if scalar < 0x80 then
      let c =
        if lowercase && scalar >= 0x41 && scalar <= 0x5A then scalar + 32
        else scalar
      in
      add_char o src (Char.unsafe_chr c)
    else
      match Uucp.Gc.general_category (Uchar.unsafe_of_int scalar) with
      | `Mn -> ()
      | _ -> (
          if not lowercase then add_code o src scalar
          else
            match Uucp.Case.Map.to_lower (Uchar.unsafe_of_int scalar) with
            | `Self -> add_code o src scalar
            | `Uchars us -> add_codes o src us)
  in
  let nf = nf ~compat:false ~compose:false sink in
  let space src =
    if pending nf then flush nf;
    add_char o src ' '
  in
  let len = String.length s in
  let i = ref 0 in
  while !i < len do
    if is_ascii s !i then
      begin if Array.unsafe_get table (Char.code (String.unsafe_get s !i)) < 0
      then incr i
      else begin
        if pending nf then flush nf;
        i := ascii_run o s !i ~lookahead:false table
      end
      end
    else begin
      let d = String.get_utf_8_uchar s !i in
      let n = Uchar.utf_decode_length d in
      let code = Uchar.to_int (Uchar.utf_decode_uchar d) in
      if clean_text && (code = 0xFFFD || is_control code) then ()
      else if clean_text && is_whitespace code then space !i
      else begin
        let ideograph = handle_chinese_chars && is_chinese_char code in
        if ideograph then space !i;
        if strip_accents then push_char nf code ~first:!i ~last:!i
        else if lowercase then
          begin if Uchar.utf_decode_is_valid d then
            match Uucp.Case.Map.to_lower (Uchar.unsafe_of_int code) with
            | `Self -> add_sub o !i s !i n
            | `Uchars us -> add_codes o !i us
          end
        else add_sub o !i s !i n;
        if ideograph then space !i
      end;
      i := !i + n
    end
  done;
  flush nf;
  text o

(* Operations *)

(* [true] iff the ASCII byte [b] is white space. *)
let[@inline] is_ascii_space b = (b >= 0x09 && b <= 0x0D) || b = 0x20

let strip_bounds s ~left ~right =
  let len = String.length s in
  let start =
    if left then
      let rec loop i =
        if i >= len then len
        else
          let b = Char.code (String.unsafe_get s i) in
          if b < 0x80 then if is_ascii_space b then loop (i + 1) else i
          else
            let p = utf8_next s i in
            let code = p lsr 3 and clen = p land 7 in
            if is_whitespace code then loop (i + clen) else i
      in
      loop 0
    else 0
  in
  let stop =
    if right then
      let rec loop i last =
        if i >= len then last
        else
          let b = Char.code (String.unsafe_get s i) in
          if b < 0x80 then
            loop (i + 1) (if is_ascii_space b then last else i + 1)
          else
            let p = utf8_next s i in
            let code = p lsr 3 and clen = p land 7 in
            let next = i + clen in
            if is_whitespace code then loop next last else loop next next
      in
      loop start start
    else len
  in
  (start, stop)

(* The first match of [pattern] at or after [pos], or [-1]. Matches of the empty
   pattern are the character boundaries. *)
let rec literal_at s pattern start k =
  k = String.length pattern
  || String.unsafe_get s (start + k) = String.unsafe_get pattern k
     && literal_at s pattern start (k + 1)

let rec literal_from s pattern first pos =
  let at = index_byte s first pos in
  if at < 0 || at + String.length pattern > String.length s then -1
  else if literal_at s pattern at 1 then at
  else literal_from s pattern first (at + 1)

let next_literal ~pattern s pos =
  if String.length pattern = 0 then if pos <= String.length s then pos else -1
  else literal_from s pattern (String.unsafe_get pattern 0) pos

(* Empty text has no match, not even of the empty pattern. *)
let count_literal ~pattern s =
  let len = String.length s and plen = String.length pattern in
  if len = 0 then 0
  else if plen = 1 then count_byte s (String.unsafe_get pattern 0)
  else
    let rec loop pos count =
      let at = next_literal ~pattern s pos in
      if at < 0 then count
      else if plen > 0 then loop (at + plen) (count + 1)
      else if at = len then count + 1
      else
        loop
          (at + Uchar.utf_decode_length (String.get_utf_8_uchar s at))
          (count + 1)
    in
    loop 0 0

(* The replacement stands for the last byte of what it replaced, or, when it
   replaced nothing, for the byte before, so a token made of it reports the last
   character of the match. *)
let replace_literal ~pattern ~replacement s ~matches o =
  let len = String.length s and plen = String.length pattern in
  let rec loop pos last left =
    if left = 0 then copy o s last (len - last)
    else
      let at = next_literal ~pattern s pos in
      copy o s last (at - last);
      if plen > 0 then begin
        add_string o (at + plen - 1) replacement;
        loop (at + plen) (at + plen) (left - 1)
      end
      else begin
        add_string o (at - 1) replacement;
        if at < len then begin
          let n = Uchar.utf_decode_length (String.get_utf_8_uchar s at) in
          copy o s at n;
          loop (at + n) (at + n) (left - 1)
        end
        else loop (at + 1) at (left - 1)
      end
  in
  loop 0 0 matches;
  text o

(* Matches are taken leftmost first without overlap. An empty match right after
   a match is skipped, and the search resumes one character further. *)
let replace_matches compiled ~replacement s o =
  let len = String.length s in
  let rec search pos last last_match matched =
    if pos > len then finish last matched
    else
      match Regex.find compiled s ~pos ~stop:len with
      | None -> finish last matched
      | Some (start, stop) ->
          if start < stop then begin
            copy o s last (start - last);
            add_string o (stop - 1) replacement;
            search stop stop stop true
          end
          else
            let next =
              if stop < len then
                stop + Uchar.utf_decode_length (String.get_utf_8_uchar s stop)
              else stop + 1
            in
            if stop = last_match then search next last last_match matched
            else begin
              copy o s last (start - last);
              add_string o (stop - 1) replacement;
              search next stop stop true
            end
  and finish last matched =
    if not matched then s
    else begin
      copy o s last (len - last);
      text o
    end
  in
  search 0 0 (-1) false

(* Byte-level encoding *)

let byte_to_unicode =
  let is_direct b =
    (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174
  in
  let tbl = Array.make 256 0 in
  let n = ref 0 in
  for b = 0 to 255 do
    if is_direct b then tbl.(b) <- b
    else (
      tbl.(b) <- 256 + !n;
      incr n)
  done;
  tbl

let byte_level_text s o =
  for i = 0 to String.length s - 1 do
    add_code o i byte_to_unicode.(Char.code (String.unsafe_get s i))
  done;
  text o

(* Machine translation cleanup *)

let nmt_removes code =
  (code >= 0x01 && code <= 0x08)
  || code = 0x0B
  || (code >= 0x0E && code <= 0x1F)
  || code = 0x7F || code = 0x8F || code = 0x9F

let nmt_spaces code =
  code = 0x09 || code = 0x0A || code = 0x0C || code = 0x0D || code = 0x1680
  || (code >= 0x200B && code <= 0x200F)
  || code = 0x2028 || code = 0x2029 || code = 0x2581 || code = 0xFEFF
  || code = 0xFFFD

let ascii_nmt =
  Array.init 128 (fun b ->
      if nmt_spaces b then 32 else if nmt_removes b then -1 else b)

let nmt_text s o =
  let len = String.length s in
  let i = ref 0 in
  while !i < len do
    if is_ascii s !i then i := ascii_run o s !i ~lookahead:false ascii_nmt
    else begin
      let d = String.get_utf_8_uchar s !i in
      let clen = Uchar.utf_decode_length d in
      let code = Uchar.to_int (Uchar.utf_decode_uchar d) in
      if not (Uchar.utf_decode_is_valid d) then add_sub o !i s !i clen
      else if nmt_spaces code then add_char o !i ' '
      else if not (nmt_removes code) then add_sub o !i s !i clen;
      i := !i + clen
    end
  done;
  text o

(* SentencePiece charsmaps *)

(* The format is SentencePiece's (src/normalizer.cc, EncodePrecompiledCharsMap)
   around darts-clone's double array (third_party/darts_clone/darts.h). A unit
   is a 32-bit word: bit 31 marks a leaf, whose low 31 bits are a value; any
   other unit has its label in the low byte, bit 8 set when a leaf hangs off it,
   and an offset to its children in bits 10 to 31, shifted left by 8 more when
   bit 9 is set. From unit [id], the child labelled [c] is at [id lxor offset
   lxor c] and the leaf at [id lxor offset]. *)

let[@inline] unit_at m id =
  Int32.to_int (String.get_int32_le m.blob (4 + (id lsl 2))) land 0xFFFF_FFFF

let[@inline] unit_has_leaf u = (u lsr 8) land 1 = 1
let[@inline] unit_value u = u land 0x7FFF_FFFF
let[@inline] unit_label u = u land 0x8000_00FF
let[@inline] unit_offset u = (u lsr 10) lsl ((u land 0x200) lsr 6)

(* The replacement of the shortest source that bytes [start] to [stop - 1] of
   [s] start with, as an offset into the replacements, or [-1]. A NUL byte ends
   the search, as no source holds one. Every index the walk reaches is in the
   trie: the charsmap was validated. *)
let shortest m s start stop =
  let rec walk id i =
    if i = stop then -1
    else
      let c = Char.code (String.unsafe_get s i) in
      if c = 0 then -1
      else
        let id = id lxor c in
        let u = unit_at m id in
        if unit_label u <> c then -1
        else
          let id = id lxor unit_offset u in
          if unit_has_leaf u then unit_value (unit_at m id) else walk id (i + 1)
  in
  walk (unit_offset (unit_at m 0)) start

let replacement_stop m v =
  match String.index_from_opt m.replacements v '\000' with
  | Some e -> e
  | None -> String.length m.replacements

let utf8_chars s start stop =
  let n = ref 0 in
  for i = start to stop - 1 do
    if not (is_continuation s i) then incr n
  done;
  !n

(* A byte that does not start a valid character is read as a character of one
   byte, [malformed], which becomes U+FFFD. SentencePiece reads it so, where
   [String.get_utf_8_uchar] would take the maximal invalid subpart. [decode]
   returns the code point and the length as [utf8_next] does. *)
let malformed = 0x110000
let replacement_char = "\xEF\xBF\xBD"

let[@inline] decode s i =
  let d = String.get_utf_8_uchar s i in
  if Uchar.utf_decode_is_valid d then
    (Uchar.to_int (Uchar.utf_decode_uchar d) lsl 3)
    lor Uchar.utf_decode_length d
  else (malformed lsl 3) lor 1

let utf8_valid s =
  let len = String.length s in
  let rec go i =
    i >= len
    ||
    let d = String.get_utf_8_uchar s i in
    Uchar.utf_decode_is_valid d && go (i + Uchar.utf_decode_length d)
  in
  go 0

(* Offsets follow the edit script HuggingFace's tokenizers builds for a
   charsmap, so that tokens report the same spans: one entry per output
   character, [0] for a character standing for the next input character, [1]
   for one inserted after the last, [-n] for one standing for the next input
   character and the [n] after it. Replacing [k] characters by [r] marks the
   last [r - k] of them inserted when [r > k], and when [r < k] folds the
   characters dropped into the last entry written, whichever replacement that
   came from; with nothing written yet, they stand for nothing and the
   characters after them shift onto them. *)
type script = { mutable edits : int array; mutable count : int }

let push_edit sc e =
  if sc.count = Array.length sc.edits then begin
    let a = Array.make (2 * sc.count) 0 in
    Array.blit sc.edits 0 a 0 sc.count;
    sc.edits <- a
  end;
  Array.unsafe_set sc.edits sc.count e;
  sc.count <- sc.count + 1

let edit_replace sc ~removed ~added =
  for _ = 1 to added do
    push_edit sc 0
  done;
  let diff = added - removed in
  if diff > 0 then
    for k = sc.count - diff to sc.count - 1 do
      sc.edits.(k) <- 1
    done
  else if diff < 0 && sc.count > 0 then
    sc.edits.(sc.count - 1) <- sc.edits.(sc.count - 1) + diff

(* The spans of [normalized], the text [s] became, from its edit script. A
   character stands for the range of the input character it is aligned with. In
   text not normalized before, that is the character's bytes as [decode] reads
   them, a malformed byte alone, where the identity would join a byte to any
   continuation bytes after it. *)
let replay s base normalized sc =
  let len = String.length s in
  let spans = spans_create (String.length normalized) in
  let char_range i =
    if i < 0 then 0
    else
      match base with
      | Identity _ -> pack i (i + (decode s i land 7))
      | _ -> range base i
  in
  (* [consumed] is the next input byte to read, [last] where the last input
     character read starts. *)
  let consumed = ref 0 and last = ref (-1) and p = ref 0 in
  let consume () =
    if !consumed < len then begin
      last := !consumed;
      consumed := !consumed + (decode s !consumed land 7)
    end
  in
  for k = 0 to sc.count - 1 do
    let e = sc.edits.(k) in
    let r =
      if e > 0 then char_range !last else char_range (min !consumed (len - 1))
    in
    if e <= 0 then
      for _ = 0 to -e do
        consume ()
      done;
    let n = Uchar.utf_decode_length (String.get_utf_8_uchar normalized !p) in
    for q = !p to !p + n - 1 do
      span_set spans q r
    done;
    p := !p + n
  done;
  spans

(* Text is read one extended grapheme cluster at a time, as {!Grapheme} cuts
   it: UAX #29 at Unicode 17, as HuggingFace's tokenizers has it through
   unicode-segmentation. A cluster under six bytes that starts with a source
   becomes the replacement of the shortest one, whatever follows it in the
   cluster; any other cluster is read one character at a time. This is
   HuggingFace's reading (tokenizers, normalizers/precompiled.rs), which brot
   matches; SentencePiece's own takes the longest source over the bytes, with
   no clusters. An ASCII byte followed by another, CR before LF aside, is a
   cluster of its own, and goes through [ascii]. *)
let charsmap_text m s o sc =
  let len = String.length s in
  let modified = ref false in
  let edit ~removed ~added =
    match sc with Some sc -> edit_replace sc ~removed ~added | None -> ()
  in
  let keep () = match sc with Some sc -> push_edit sc 0 | None -> () in
  let emit src v ~removed =
    let e = replacement_stop m v in
    add_sub o src m.replacements v (e - v);
    (match sc with
    | Some sc -> edit_replace sc ~removed ~added:(utf8_chars m.replacements v e)
    | None -> ());
    modified := true
  in
  let i = ref 0 in
  while !i < len do
    let b = Char.code (String.unsafe_get s !i) in
    let alone =
      b < 0x80
      && (!i + 1 = len
         ||
         let c = Char.code (String.unsafe_get s (!i + 1)) in
         c < 0x80 && not (b = 0x0D && c = 0x0A))
    in
    if alone then begin
      let v = Array.unsafe_get m.ascii b in
      if v < 0 then begin
        add_char o !i (Char.unsafe_chr b);
        keep ()
      end
      else emit !i v ~removed:1;
      incr i
    end
    else if decode s !i lsr 3 = malformed then begin
      add_string o !i replacement_char;
      edit ~removed:1 ~added:1;
      modified := true;
      incr i
    end
    else begin
      let stop = Grapheme.stop s !i in
      let v = if stop - !i < 6 then shortest m s !i stop else -1 in
      if v >= 0 then emit !i v ~removed:(utf8_chars s !i stop)
      else begin
        let j = ref !i in
        while !j < stop do
          let n = decode s !j land 7 in
          let v = shortest m s !j (!j + n) in
          if v >= 0 then emit !j v ~removed:1
          else begin
            add_sub o !j s !j n;
            keep ()
          end;
          j := !j + n
        done
      end;
      i := stop
    end
  done;
  if !modified then text o else s

let valid_trie m =
  let size = m.units in
  let u0 = unit_at m 0 in
  let leaf_ok u =
    let v = unit_value u in
    v < String.length m.replacements && not (is_continuation m.replacements v)
  in
  let rec units_ok i =
    i >= size
    ||
    let u = unit_at m i in
    (if unit_label u <= 0xFF then
       let child = i lxor unit_offset u in
       child lor 0xFF < size
       && ((not (unit_has_leaf u)) || unit_label (unit_at m child) > 0xFF)
     else leaf_ok u)
    && units_ok (i + 1)
  in
  unit_label u0 = 0
  && (not (unit_has_leaf u0))
  && unit_offset u0 <> 0
  && unit_offset u0 lor 0xFF < size
  && units_ok 1

(* SentencePiece's checks of a charsmap (Normalizer::Init and darts-clone's
   validate), and the replacements being UTF-8, which HuggingFace's tokenizers
   requires. Every index a search can reach is then in range. *)
let charsmap_of_blob blob =
  let len = String.length blob in
  let size =
    if len <= 4 then -1
    else Int32.to_int (String.get_int32_le blob 0) land 0xFFFF_FFFF
  in
  if size < 0 then Error "the charsmap is truncated"
  else if size >= len - 4 then Error "the trie overruns the charsmap"
  else if size < 1024 || size land 0x3FF <> 0 then
    Error "the trie size is not a multiple of 1024"
  else
    let replacements = String.sub blob (4 + size) (len - 4 - size) in
    let m = { blob; units = size lsr 2; replacements; ascii = [||] } in
    if not (utf8_valid replacements) then
      Error "the replacements are not valid UTF-8"
    else if not (valid_trie m) then Error "the trie is malformed"
    else
      let ascii =
        Array.init 128 (fun b -> shortest m (String.make 1 (Char.chr b)) 0 1)
      in
      Ok { m with ascii }

(* Base64, the standard alphabet, as tokenizer files store a charsmap
   (RFC 4648, section 4). Decoding reads what HuggingFace's tokenizers reads:
   the padding of the last group may be complete, partial or absent, but no
   more than it needs, and the bits left over in its last character must be
   zero. *)

let base64_alphabet =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

let base64_encode s =
  let len = String.length s in
  let out = Bytes.make ((len + 2) / 3 * 4) '=' in
  let digit k v = Bytes.set out k base64_alphabet.[v land 63] in
  for g = 0 to ((len + 2) / 3) - 1 do
    let i = 3 * g in
    let byte k = if i + k < len then Char.code s.[i + k] else 0 in
    let w = (byte 0 lsl 16) lor (byte 1 lsl 8) lor byte 2 in
    digit (4 * g) (w lsr 18);
    digit ((4 * g) + 1) (w lsr 12);
    if i + 1 < len then digit ((4 * g) + 2) (w lsr 6);
    if i + 2 < len then digit ((4 * g) + 3) w
  done;
  Bytes.unsafe_to_string out

let base64_digit = function
  | 'A' .. 'Z' as c -> Char.code c - 65
  | 'a' .. 'z' as c -> Char.code c - 71
  | '0' .. '9' as c -> Char.code c + 4
  | '+' -> 62
  | '/' -> 63
  | _ -> -1

let base64_decode s =
  let len = String.length s in
  let data = ref len in
  while !data > 0 && s.[!data - 1] = '=' do
    decr data
  done;
  let data = !data in
  let rest = data land 3 in
  if rest = 1 || len - data > (4 - rest) land 3 then None
  else
    let out = Bytes.create ((data / 4 * 3) + max 0 (rest - 1)) in
    let ok = ref true in
    for g = 0 to ((data + 3) / 4) - 1 do
      let w = ref 0 in
      for k = 0 to 3 do
        let i = (4 * g) + k in
        let d = if i >= data then 0 else base64_digit s.[i] in
        if d < 0 then ok := false;
        w := (!w lsl 6) lor (d land 63)
      done;
      for k = 0 to 2 do
        let at = (3 * g) + k in
        let b = (!w lsr (16 - (8 * k))) land 0xFF in
        if at < Bytes.length out then Bytes.set out at (Char.chr b)
        else if b <> 0 then ok := false
      done
    done;
    if !ok then Some (Bytes.unsafe_to_string out) else None

(* Constructors *)

let nfc = NFC
let nfd = NFD
let nfkc = NFKC
let nfkd = NFKD
let lowercase = Lowercase
let strip_accents = Strip_accents
let strip ?(left = true) ?(right = true) () = Strip { left; right }

let replace ~pattern ~replacement =
  Replace { pattern = Literal pattern; replacement }

let regex source =
  Result.map (fun compiled -> Regex { source; compiled }) (Regex.compile source)

let replace_regex ~pattern ~replacement =
  match regex pattern with
  | Ok pattern -> Replace { pattern; replacement }
  | Error msg ->
      invalid_arg (strf "invalid regular expression %S: %s" pattern msg)

let prepend s = Prepend s
let byte_level = Byte_level
let nmt = Nmt

let precompiled blob =
  match charsmap_of_blob blob with
  | Ok m -> Precompiled m
  | Error msg -> invalid_arg (strf "invalid precompiled charsmap: %s" msg)

let bert ?(clean_text = true) ?(handle_chinese_chars = true)
    ?(strip_accents = None) ?(lowercase = true) () =
  Bert { clean_text; handle_chinese_chars; strip_accents; lowercase }

let sequence ns = Sequence ns

(* Apply *)

(* A stage that returns its input unchanged leaves the alignment alone; one that
   rewrites it has recorded the alignment of what it wrote. *)
let step ?cap s a ~track transform =
  let cap = match cap with Some cap -> cap | None -> String.length s in
  let o = out ~track a cap in
  let s' = transform o in
  if s' == s then (s, a) else (s', if track then compose a o else a)

(* Skipping a stage that cannot change the text also skips the buffer it would
   have needed. *)
let step_if changes s a ~track transform =
  if changes s then step s a ~track transform else (s, a)

let rec normalize t s a ~track =
  match t with
  | NFC -> step_if has_non_ascii s a ~track (normalize_text `NFC s)
  | NFD -> step_if has_non_ascii s a ~track (normalize_text `NFD s)
  | NFKC -> step_if has_non_ascii s a ~track (normalize_text `NFKC s)
  | NFKD -> step_if has_non_ascii s a ~track (normalize_text `NFKD s)
  | Lowercase -> step_if needs_lowering s a ~track (lowercase_text s)
  | Strip_accents -> step_if has_non_ascii s a ~track (drop_marks s)
  | Strip { left; right } ->
      let start, stop = strip_bounds s ~left ~right in
      if start = 0 && stop = String.length s then (s, a)
      else
        let count = stop - start in
        (String.sub s start count, if track then sliced a start count else a)
  | Replace { pattern = Literal pattern; replacement } ->
      let matches = count_literal ~pattern s in
      if matches = 0 then (s, a)
      else
        let cap =
          String.length s
          + (matches * (String.length replacement - String.length pattern))
        in
        step ~cap s a ~track (replace_literal ~pattern ~replacement s ~matches)
  | Replace { pattern = Regex { compiled; _ }; replacement } ->
      if String.length s = 0 then (s, a)
      else step s a ~track (replace_matches compiled ~replacement s)
  | Prepend prefix ->
      if String.length s = 0 then (s, a)
      else (prefix ^ s, if track then prefixed a (String.length prefix) else a)
  | Byte_level -> step s a ~track (byte_level_text s)
  | Nmt -> step s a ~track (nmt_text s)
  | Precompiled m ->
      if String.length s = 0 then (s, a)
      else
        (* The spans come from the edit script once the text is known, so the
           output does not track them as it is written. *)
        let o = out ~track:false a (String.length s) in
        let sc =
          if track then Some { edits = Array.make 16 0; count = 0 } else None
        in
        let s' = charsmap_text m s o sc in
        if s' == s then (s, a)
        else begin
          match sc with
          | None -> (s', a)
          | Some sc ->
              let spans = replay s a s' sc in
              ( s',
                Map
                  {
                    spans;
                    count = String.length s';
                    original = original_length a;
                  } )
        end
  | Bert { clean_text; handle_chinese_chars; strip_accents; lowercase } ->
      let strip_accents =
        match strip_accents with Some v -> v | None -> lowercase
      in
      (* Without cleaning, the passes left only touch what is not ASCII, or not
         lower case. *)
      let nothing_to_do =
        (not clean_text)
        && if lowercase then not (needs_lowering s) else not (has_non_ascii s)
      in
      if
        nothing_to_do
        || not (clean_text || handle_chinese_chars || strip_accents || lowercase)
      then (s, a)
      else
        step s a ~track
          (bert_text ~clean_text ~handle_chinese_chars ~strip_accents ~lowercase
             s)
  | Sequence ns ->
      List.fold_left (fun (s, a) n -> normalize n s a ~track) (s, a) ns

let apply t s = fst (normalize t s (Identity s) ~track:false)
let apply_aligned t s = normalize t s (Identity s) ~track:true

(* Formatting *)

let pp_bool_opt ppf = function
  | None -> Format.pp_print_string ppf "None"
  | Some b -> Format.fprintf ppf "Some(%b)" b

let rec pp ppf = function
  | NFC -> Format.pp_print_string ppf "NFC"
  | NFD -> Format.pp_print_string ppf "NFD"
  | NFKC -> Format.pp_print_string ppf "NFKC"
  | NFKD -> Format.pp_print_string ppf "NFKD"
  | Lowercase -> Format.pp_print_string ppf "Lowercase"
  | Strip_accents -> Format.pp_print_string ppf "StripAccents"
  | Strip { left; right } ->
      Format.fprintf ppf "@[<1>Strip(left=%b,@ right=%b)@]" left right
  | Replace { pattern = Literal pattern; replacement } ->
      Format.fprintf ppf "@[<1>Replace(%S,@ %S)@]" pattern replacement
  | Replace { pattern = Regex { source; _ }; replacement } ->
      Format.fprintf ppf "@[<1>Replace(Regex(%S),@ %S)@]" source replacement
  | Prepend s -> Format.fprintf ppf "Prepend(%S)" s
  | Byte_level -> Format.pp_print_string ppf "ByteLevel"
  | Nmt -> Format.pp_print_string ppf "Nmt"
  | Precompiled m ->
      Format.fprintf ppf "Precompiled(%d bytes)" (String.length m.blob)
  | Bert { clean_text; handle_chinese_chars; strip_accents; lowercase } ->
      Format.fprintf ppf
        "@[<1>Bert(clean_text=%b,@ handle_chinese_chars=%b,@ \
         strip_accents=%a,@ lowercase=%b)@]"
        clean_text handle_chinese_chars pp_bool_opt strip_accents lowercase
  | Sequence ns ->
      Format.fprintf ppf "@[<1>Sequence[%a]@]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ")
           pp)
        ns

(*---------------------------------------------------------------------------
  Serialization
  ---------------------------------------------------------------------------*)

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let typed name = json_obj [ ("type", Jsont.Json.string name) ]
let typed_with name pairs = json_obj (("type", Jsont.Json.string name) :: pairs)

let rec to_json = function
  | Bert { clean_text; handle_chinese_chars; strip_accents; lowercase } ->
      typed_with "BertNormalizer"
        [
          ("clean_text", Jsont.Json.bool clean_text);
          ("handle_chinese_chars", Jsont.Json.bool handle_chinese_chars);
          ( "strip_accents",
            match strip_accents with
            | None -> Jsont.Json.null ()
            | Some b -> Jsont.Json.bool b );
          ("lowercase", Jsont.Json.bool lowercase);
        ]
  | Strip { left; right } ->
      typed_with "Strip"
        [
          ("strip_left", Jsont.Json.bool left);
          ("strip_right", Jsont.Json.bool right);
        ]
  | Strip_accents -> typed "StripAccents"
  | NFC -> typed "NFC"
  | NFD -> typed "NFD"
  | NFKC -> typed "NFKC"
  | NFKD -> typed "NFKD"
  | Lowercase -> typed "Lowercase"
  | Replace { pattern; replacement } ->
      let pattern =
        match pattern with
        | Literal s -> ("String", Jsont.Json.string s)
        | Regex { source; _ } -> ("Regex", Jsont.Json.string source)
      in
      typed_with "Replace"
        [
          ("pattern", json_obj [ pattern ]);
          ("content", Jsont.Json.string replacement);
        ]
  | Prepend prefix ->
      typed_with "Prepend" [ ("prepend", Jsont.Json.string prefix) ]
  | Byte_level -> typed "ByteLevel"
  | Nmt -> typed "Nmt"
  | Precompiled m ->
      typed_with "Precompiled"
        [ ("precompiled_charsmap", Jsont.Json.string (base64_encode m.blob)) ]
  | Sequence ns ->
      typed_with "Sequence"
        [ ("normalizers", Jsont.Json.list (List.map to_json ns)) ]

let rec of_json = function
  | Jsont.Object (fields, _) -> (
      let find name = Option.map snd (Jsont.Json.find_mem name fields) in
      let get_bool name default =
        match find name with Some (Jsont.Bool (b, _)) -> b | _ -> default
      in
      match find "type" with
      | Some (Jsont.String (("Bert" | "BertNormalizer"), _)) ->
          let strip_accents =
            match find "strip_accents" with
            | Some (Jsont.Bool (b, _)) -> Some b
            | _ -> None
          in
          Ok
            (Bert
               {
                 clean_text = get_bool "clean_text" true;
                 handle_chinese_chars = get_bool "handle_chinese_chars" true;
                 strip_accents;
                 lowercase = get_bool "lowercase" true;
               })
      | Some (Jsont.String ("Strip", _)) ->
          (* HuggingFace requires both members; brot falls back on the defaults
             of {!val-strip} rather than rejecting the file. *)
          Ok
            (Strip
               {
                 left = get_bool "strip_left" true;
                 right = get_bool "strip_right" true;
               })
      | Some (Jsont.String ("StripAccents", _)) -> Ok Strip_accents
      | Some (Jsont.String ("NFC", _)) -> Ok NFC
      | Some (Jsont.String ("NFD", _)) -> Ok NFD
      | Some (Jsont.String ("NFKC", _)) -> Ok NFKC
      | Some (Jsont.String ("NFKD", _)) -> Ok NFKD
      | Some (Jsont.String ("Lowercase", _)) -> Ok Lowercase
      | Some (Jsont.String ("Replace", _)) ->
          let pattern =
            match find "pattern" with
            | Some (Jsont.Object (pf, _)) -> (
                match
                  ( Jsont.Json.find_mem "String" pf,
                    Jsont.Json.find_mem "Regex" pf )
                with
                | Some (_, Jsont.String (s, _)), None -> Ok (Literal s)
                | None, Some (_, Jsont.String (r, _)) ->
                    Result.map_error
                      (fun msg ->
                        strf "invalid regular expression %S: %s" r msg)
                      (regex r)
                | _ -> Error err_replace_invalid_pattern)
            | _ -> Error err_replace_missing_pattern
          in
          let replacement =
            match find "content" with
            | Some (Jsont.String (r, _)) -> Ok r
            | _ -> Error err_replace_missing_content
          in
          Result.bind pattern (fun pattern ->
              Result.map
                (fun replacement -> Replace { pattern; replacement })
                replacement)
      | Some (Jsont.String ("Prepend", _)) -> (
          match find "prepend" with
          | Some (Jsont.String (p, _)) -> Ok (Prepend p)
          | _ -> Error err_prepend_missing)
      | Some (Jsont.String ("ByteLevel", _)) -> Ok Byte_level
      | Some (Jsont.String ("Nmt", _)) -> Ok Nmt
      | Some (Jsont.String ("Precompiled", _)) -> (
          match find "precompiled_charsmap" with
          | Some (Jsont.String (b64, _)) -> (
              match base64_decode b64 with
              | None -> Error err_charsmap_base64
              | Some blob ->
                  Result.map
                    (fun m -> Precompiled m)
                    (Result.map_error
                       (strf "invalid precompiled_charsmap: %s")
                       (charsmap_of_blob blob)))
          | _ -> Error err_charsmap_missing)
      | Some (Jsont.String ("Sequence", _)) -> (
          match find "normalizers" with
          | Some (Jsont.Array (l, _)) ->
              let rec build acc = function
                | [] -> Ok (Sequence (List.rev acc))
                | item :: rest ->
                    Result.bind (of_json item) (fun n -> build (n :: acc) rest)
              in
              build [] l
          | _ -> Error err_sequence_missing)
      | Some (Jsont.String (other, _)) ->
          Error (strf "Unknown normalizer type: %s" other)
      | _ -> Error err_missing_type)
  | _ -> Error err_expected_object
