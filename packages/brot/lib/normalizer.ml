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
let strf = Printf.sprintf

(* Type *)

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
  | Sequence of t list

and pattern =
  | Literal of string
  | Regex of { source : string; compiled : Re.re }

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
      match Re.exec_opt ~pos compiled s with
      | None -> finish last matched
      | Some group ->
          let start = Re.Group.start group 0 and stop = Re.Group.stop group 0 in
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
