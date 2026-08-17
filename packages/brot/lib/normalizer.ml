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
  | Replace of { pattern : string; replacement : string; compiled : Re.re }
  | Prepend of string
  | Byte_level of { add_prefix_space : bool; use_regex : bool }
  | Sequence of t list

(* UTF-8 helpers *)

(* Returns (codepoint lsl 3) lor byte_length — zero allocation. *)
let[@inline] utf8_next s i =
  let d = String.get_utf_8_uchar s i in
  (Uchar.to_int (Uchar.utf_decode_uchar d) lsl 3) lor Uchar.utf_decode_length d

let[@inline] is_continuation s i =
  Char.code (String.unsafe_get s i) land 0xC0 = 0x80

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

(* Text that is all ASCII holds no mark, no ideograph and nothing to decompose,
   so the stages that only touch those can be skipped whole. *)
let has_non_ascii s =
  let len = String.length s in
  let rec loop i =
    i < len && (Char.code (String.unsafe_get s i) >= 128 || loop (i + 1))
  in
  loop 0

let needs_lowering s =
  let len = String.length s in
  let rec loop i =
    i < len
    &&
    let byte = Char.code (String.unsafe_get s i) in
    (byte >= 0x41 && byte <= 0x5A) || byte >= 128 || loop (i + 1)
  in
  loop 0

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
   the original. *)
type alignment =
  | Identity of string
  | Map of { starts : int array; stops : int array; original : int }

let identity s = Identity s

let normalized_length = function
  | Identity s -> String.length s
  | Map m -> Array.length m.starts

let original_length = function
  | Identity s -> String.length s
  | Map m -> m.original

(* The bytes of the character byte [i] falls in. Byte [-1] is what a character
   inserted before anything stands for. Kept as two lookups rather than one
   returning a pair: they run once per byte of every stage, and a pair would be
   a boxed allocation each time. *)
let span_start a i =
  if i < 0 then 0
  else
    match a with
    | Identity s ->
        let start = ref i in
        while !start > 0 && is_continuation s !start do
          decr start
        done;
        !start
    | Map m -> Array.unsafe_get m.starts i

let span_stop a i =
  if i < 0 then 0
  else
    match a with
    | Identity s ->
        let len = String.length s in
        let stop = ref (i + 1) in
        while !stop < len && is_continuation s !stop do
          incr stop
        done;
        !stop
    | Map m -> Array.unsafe_get m.stops i

let original_span a ~start ~stop =
  let len = normalized_length a in
  if start < 0 || stop < start || stop > len then
    invalid_arg
      (strf "%d,%d is not a span of the %d normalized bytes" start stop len);
  (* Normalizing the text away leaves nothing to map through, so the only span
     there is stands for the whole of what was normalized. *)
  if len = 0 then (0, original_length a)
  else if start < stop then (span_start a start, span_stop a (stop - 1))
  else
    let at = if start < len then span_start a start else original_length a in
    (at, at)

(* Output *)

(* A stage's text, and, while tracking, the input byte every output byte stands
   for. [apply] runs untracked, so [mark] is the only cost it pays. *)
type out = {
  buf : Buffer.t;
  mutable from : int array;
  mutable len : int;
  track : bool;
}

let out ~track n =
  {
    buf = Buffer.create (max n 16);
    from = (if track then Array.make (max n 16) 0 else [||]);
    len = 0;
    track;
  }

let room o n =
  let need = o.len + n in
  if need > Array.length o.from then begin
    let grown = Array.make (max need (2 * Array.length o.from)) 0 in
    Array.blit o.from 0 grown 0 o.len;
    o.from <- grown
  end

let[@inline] mark o src n =
  if o.track then begin
    room o n;
    Array.fill o.from o.len n src;
    o.len <- o.len + n
  end

let[@inline] add_char o src c =
  Buffer.add_char o.buf c;
  mark o src 1

let[@inline] add_uchar o src u =
  Buffer.add_utf_8_uchar o.buf u;
  mark o src (Uchar.utf_8_byte_length u)

let[@inline] add_sub o src s pos n =
  Buffer.add_substring o.buf s pos n;
  mark o src n

let[@inline] add_string o src str =
  Buffer.add_string o.buf str;
  mark o src (String.length str)

(* Bytes kept as they are, each standing for itself. *)
let copy o s pos n =
  Buffer.add_substring o.buf s pos n;
  if o.track then begin
    room o n;
    for i = 0 to n - 1 do
      Array.unsafe_set o.from (o.len + i) (pos + i)
    done;
    o.len <- o.len + n
  end

let text o = Buffer.contents o.buf

let compose a o =
  let n = o.len in
  let starts = Array.make n 0 and stops = Array.make n 0 in
  for i = 0 to n - 1 do
    let from = Array.unsafe_get o.from i in
    Array.unsafe_set starts i (span_start a from);
    Array.unsafe_set stops i (span_stop a from)
  done;
  Map { starts; stops; original = original_length a }

(* Unicode normalization *)

(* Normalization tracks alignments by decomposing every character on its own:
   the first character of a decomposition stands for its source, the rest stand
   for nothing. Canonical ordering then permutes a run of marks, carrying that
   with it, so a character can end up standing for a source other than the one
   it came from; composition adds up what it folds together. This is the
   accounting HuggingFace reports offsets with. *)

(* Decomposed characters: scalar values, combining classes, and the number of
   input characters each stands for. [ordered] is how much of the run has been
   canonically ordered already. *)
type run = {
  mutable scalars : int array;
  mutable classes : int array;
  mutable stands_for : int array;
  mutable count : int;
  mutable ordered : int;
}

let run n =
  {
    scalars = Array.make n 0;
    classes = Array.make n 0;
    stands_for = Array.make n 0;
    count = 0;
    ordered = 0;
  }

let grow r =
  let cap = 2 * Array.length r.scalars in
  let scalars = Array.make cap 0
  and classes = Array.make cap 0
  and stands_for = Array.make cap 0 in
  Array.blit r.scalars 0 scalars 0 r.count;
  Array.blit r.classes 0 classes 0 r.count;
  Array.blit r.stands_for 0 stands_for 0 r.count;
  r.scalars <- scalars;
  r.classes <- classes;
  r.stands_for <- stands_for

(* Insertion sort: stable, as canonical ordering must be, and a run of marks is
   short. *)
let order r =
  for i = r.ordered + 1 to r.count - 1 do
    let scalar = r.scalars.(i)
    and cls = r.classes.(i)
    and stands = r.stands_for.(i) in
    let j = ref (i - 1) in
    while !j >= r.ordered && r.classes.(!j) > cls do
      r.scalars.(!j + 1) <- r.scalars.(!j);
      r.classes.(!j + 1) <- r.classes.(!j);
      r.stands_for.(!j + 1) <- r.stands_for.(!j);
      decr j
    done;
    r.scalars.(!j + 1) <- scalar;
    r.classes.(!j + 1) <- cls;
    r.stands_for.(!j + 1) <- stands
  done;
  r.ordered <- r.count

let push r u stands =
  let cls = Uunf.ccc u in
  if cls = 0 then order r;
  if r.count = Array.length r.scalars then grow r;
  r.scalars.(r.count) <- Uchar.to_int u;
  r.classes.(r.count) <- cls;
  r.stands_for.(r.count) <- stands;
  r.count <- r.count + 1

let rec decompose ~compatibility u emit =
  let d = Uunf.decomp u in
  if Array.length d = 0 || ((not compatibility) && Uunf.d_compatibility d.(0))
  then emit u
  else begin
    decompose ~compatibility (Uunf.d_uchar d.(0)) emit;
    for i = 1 to Array.length d - 1 do
      decompose ~compatibility (Uchar.of_int d.(i)) emit
    done
  end

let decomposed ~compatibility s =
  let len = String.length s in
  let r = run (max 16 len) in
  let i = ref 0 in
  while !i < len do
    let d = String.get_utf_8_uchar s !i in
    let u = Uchar.utf_decode_uchar d in
    if Uchar.to_int u < 0x80 then push r u 1
    else begin
      let first = ref true in
      decompose ~compatibility u (fun c ->
          push r c (if !first then 1 else 0);
          first := false)
    end;
    i := !i + Uchar.utf_decode_length d
  done;
  order r;
  r

(* Composition folds a starter and the marks that reach it into one character
   standing for all of them; marks blocked from the starter are held back and
   come out after it. *)
let composed r =
  let n = r.count in
  let scalars = Array.make (max n 1) 0
  and stands_for = Array.make (max n 1) 0 in
  let count = ref 0 in
  let emit scalar stands =
    scalars.(!count) <- scalar;
    stands_for.(!count) <- stands;
    incr count
  in
  let held = Array.make (max n 1) 0 and held_stands = Array.make (max n 1) 0 in
  let held_count = ref 0 in
  let starter = ref (-1) and starter_stands = ref 0 and last_class = ref (-1) in
  let hold scalar stands cls =
    held.(!held_count) <- scalar;
    held_stands.(!held_count) <- stands;
    incr held_count;
    last_class := cls
  in
  let release () =
    for i = 0 to !held_count - 1 do
      emit held.(i) held_stands.(i)
    done;
    held_count := 0
  in
  for i = 0 to n - 1 do
    let scalar = r.scalars.(i)
    and stands = r.stands_for.(i)
    and cls = r.classes.(i) in
    if !starter < 0 then
      if cls <> 0 then emit scalar stands
      else begin
        starter := scalar;
        starter_stands := stands
      end
    else
      match
        if !last_class >= cls then None
        else Uunf.composite (Uchar.of_int !starter) (Uchar.of_int scalar)
      with
      | Some c ->
          starter := Uchar.to_int c;
          starter_stands := !starter_stands + stands
      | None ->
          if cls <> 0 then hold scalar stands cls
          else begin
            emit !starter !starter_stands;
            release ();
            starter := scalar;
            starter_stands := stands;
            last_class := -1
          end
  done;
  if !starter >= 0 then emit !starter !starter_stands;
  release ();
  (scalars, stands_for, !count)

(* Walk the output against the input: a character standing for one or more input
   characters takes the range of the first of them, one standing for none takes
   the range of the byte before, which is the last input character consumed. *)
let emit_normalized s scalars stands_for count o =
  let at = ref 0 in
  for i = 0 to count - 1 do
    let stands = Array.unsafe_get stands_for i in
    add_uchar o
      (if stands = 0 then !at - 1 else !at)
      (Uchar.of_int (Array.unsafe_get scalars i));
    for _ = 1 to stands do
      at := !at + Uchar.utf_decode_length (String.get_utf_8_uchar s !at)
    done
  done;
  text o

let normalize_aligned nf s o =
  let compatibility =
    match nf with `NFKC | `NFKD -> true | `NFC | `NFD -> false
  in
  let r = decomposed ~compatibility s in
  match nf with
  | `NFD | `NFKD -> emit_normalized s r.scalars r.stands_for r.count o
  | `NFC | `NFKC ->
      let scalars, stands_for, count = composed r in
      emit_normalized s scalars stands_for count o

(* Text transforms *)

(* The full Unicode lowercase mapping, which is not case folding: ["ß"] and
   ["ﬁ"] lowercase to themselves but fold to ["ss"] and ["fi"]. *)
let lowercase_text s o =
  let len = String.length s in
  let i = ref 0 in
  while !i < len do
    let byte = Char.code (String.unsafe_get s !i) in
    if byte < 128 then begin
      let c = if byte >= 0x41 && byte <= 0x5A then byte + 32 else byte in
      add_char o !i (Char.unsafe_chr c);
      incr i
    end
    else
      let d = String.get_utf_8_uchar s !i in
      let n = Uchar.utf_decode_length d in
      (if Uchar.utf_decode_is_valid d then
         let u = Uchar.utf_decode_uchar d in
         match Uucp.Case.Map.to_lower u with
         | `Self -> add_uchar o !i u
         | `Uchars us -> List.iter (fun u -> add_uchar o !i u) us);
      i := !i + n
  done;
  text o

(* [~nonspacing_only] drops the marks of general category [Mn], which are the
   accents proper; otherwise every mark goes, spacing ([Mc]) and enclosing
   ([Me]) included. *)
let drop_marks ~nonspacing_only s o =
  let len = String.length s in
  let i = ref 0 in
  while !i < len do
    let byte = Char.code (String.unsafe_get s !i) in
    if byte < 128 then begin
      add_char o !i (Char.unsafe_chr byte);
      incr i
    end
    else
      let d = String.get_utf_8_uchar s !i in
      let n = Uchar.utf_decode_length d in
      (if Uchar.utf_decode_is_valid d then
         let u = Uchar.utf_decode_uchar d in
         match Uucp.Gc.general_category u with
         | `Mn -> ()
         | (`Mc | `Me) when not nonspacing_only -> ()
         | _ -> add_uchar o !i u);
      i := !i + n
  done;
  text o

(* Operations *)

let clean_text s o =
  let len = String.length s in
  let i = ref 0 in
  while !i < len do
    let b0 = Char.code (String.unsafe_get s !i) in
    if b0 < 128 then begin
      if b0 = 9 || b0 = 10 || b0 = 13 || b0 = 32 then add_char o !i ' '
      else if b0 >= 33 && b0 < 127 then add_char o !i (Char.unsafe_chr b0);
      incr i
    end
    else begin
      let p = utf8_next s !i in
      let code = p lsr 3 and clen = p land 7 in
      if code <> 0xFFFD && not (is_control code) then
        if is_whitespace code then add_char o !i ' ' else add_sub o !i s !i clen;
      i := !i + clen
    end
  done;
  text o

let handle_chinese_chars s o =
  let len = String.length s in
  let i = ref 0 in
  while !i < len do
    let b0 = Char.code (String.unsafe_get s !i) in
    if b0 < 128 then begin
      add_char o !i (Char.unsafe_chr b0);
      incr i
    end
    else begin
      let p = utf8_next s !i in
      let code = p lsr 3 and clen = p land 7 in
      if is_chinese_char code then (
        add_char o !i ' ';
        add_sub o !i s !i clen;
        add_char o !i ' ')
      else add_sub o !i s !i clen;
      i := !i + clen
    end
  done;
  text o

let strip_bounds s ~left ~right =
  let len = String.length s in
  let start =
    if left then
      let rec loop i =
        if i >= len then len
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
          let p = utf8_next s i in
          let code = p lsr 3 and clen = p land 7 in
          let next = i + clen in
          if is_whitespace code then loop next last else loop next next
      in
      loop start start
    else len
  in
  (start, stop)

(* The replacement stands for the last byte of what it replaced, so a token made
   of it reports the last character of the match. Empty matches are advanced
   past exactly as [Re.replace] does, the byte they sit on being copied
   through. *)
let replace_text compiled ~replacement s o =
  let len = String.length s in
  let rec search pos last on_match matched =
    if pos > len then done_ last matched
    else
      match Re.exec_opt ~pos compiled s with
      | None -> done_ last matched
      | Some group ->
          let start = Re.Group.start group 0 and stop = Re.Group.stop group 0 in
          (* An empty match right after a match is stepped over, the byte it
             sits on staying in the text. *)
          if on_match && start = pos && start = stop then
            search (pos + 1) last false matched
          else begin
            copy o s last (start - last);
            add_string o (stop - 1) replacement;
            if start < stop then search stop stop true true
            else begin
              if stop < len then copy o s stop 1;
              search (stop + 1) (stop + 1) false true
            end
          end
  and done_ last matched =
    if not matched then s
    else begin
      if last < len then copy o s last (len - last);
      text o
    end
  in
  search 0 0 false false

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

let apply_byte_level ~add_prefix_space ~use_regex:_ s o =
  let len = String.length s in
  if add_prefix_space && len > 0 && not (is_whitespace (utf8_next s 0 lsr 3))
  then add_uchar o 0 (Uchar.of_int byte_to_unicode.(Char.code ' '));
  for i = 0 to len - 1 do
    add_uchar o i
      (Uchar.of_int byte_to_unicode.(Char.code (String.unsafe_get s i)))
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
  Replace { pattern; replacement; compiled = Re.compile (Re.Pcre.re pattern) }

let prepend s = Prepend s

let byte_level ?(add_prefix_space = false) () =
  Byte_level { add_prefix_space; use_regex = false }

let bert ?(clean_text = true) ?(handle_chinese_chars = true)
    ?(strip_accents = None) ?(lowercase = true) () =
  Bert { clean_text; handle_chinese_chars; strip_accents; lowercase }

let sequence ns = Sequence ns

(* Apply *)

(* A stage that returns its input unchanged leaves the alignment alone; one that
   rewrites it composes what it recorded onto the alignment so far. *)
let step s a ~track transform =
  let o = out ~track (String.length s) in
  let s' = transform o in
  if s' == s then (s, a) else (s', if track then compose a o else a)

(* Skipping a stage that cannot change the text also skips the buffer it would
   have needed. *)
let step_if changes s a ~track transform =
  if changes s then step s a ~track transform else (s, a)

let normalize_step nf s a ~track =
  if not (has_non_ascii s) then (s, a)
  else if track then step s a ~track (normalize_aligned nf s)
  else (Uunf_string.normalize_utf_8 nf s, a)

(* The prefix and the first character of the text both stand for that first
   character. *)
let prepend_alignment a prefix len =
  let n = prefix + len in
  let starts = Array.make n 0 and stops = Array.make n 0 in
  for i = 0 to n - 1 do
    let from = if i < prefix then 0 else i - prefix in
    starts.(i) <- span_start a from;
    stops.(i) <- span_stop a from
  done;
  Map { starts; stops; original = original_length a }

let slice_alignment a start len =
  let starts = Array.make len 0 and stops = Array.make len 0 in
  for i = 0 to len - 1 do
    starts.(i) <- span_start a (start + i);
    stops.(i) <- span_stop a (start + i)
  done;
  Map { starts; stops; original = original_length a }

let rec normalize t s a ~track =
  match t with
  | NFC -> normalize_step `NFC s a ~track
  | NFD -> normalize_step `NFD s a ~track
  | NFKC -> normalize_step `NFKC s a ~track
  | NFKD -> normalize_step `NFKD s a ~track
  | Lowercase -> step_if needs_lowering s a ~track (lowercase_text s)
  | Strip_accents ->
      step_if has_non_ascii s a ~track (drop_marks ~nonspacing_only:false s)
  | Strip { left; right } ->
      let start, stop = strip_bounds s ~left ~right in
      if start = 0 && stop = String.length s then (s, a)
      else
        let len = stop - start in
        ( String.sub s start len,
          if track then slice_alignment a start len else a )
  | Replace { compiled; replacement; _ } ->
      step s a ~track (replace_text compiled ~replacement s)
  | Prepend prefix ->
      if String.length s = 0 then (s, a)
      else
        ( prefix ^ s,
          if track then
            prepend_alignment a (String.length prefix) (String.length s)
          else a )
  | Byte_level { add_prefix_space; use_regex } ->
      step s a ~track (apply_byte_level ~add_prefix_space ~use_regex s)
  | Bert
      {
        clean_text = ct;
        handle_chinese_chars = hcc;
        strip_accents = sa;
        lowercase = lc;
      } ->
      let s, a = if ct then step s a ~track (clean_text s) else (s, a) in
      let s, a =
        if hcc then step_if has_non_ascii s a ~track (handle_chinese_chars s)
        else (s, a)
      in
      let s, a =
        if match sa with Some v -> v | None -> lc then
          let s, a = normalize_step `NFD s a ~track in
          step_if has_non_ascii s a ~track (drop_marks ~nonspacing_only:true s)
        else (s, a)
      in
      if lc then step_if needs_lowering s a ~track (lowercase_text s) else (s, a)
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
  | Replace { pattern; replacement; _ } ->
      Format.fprintf ppf "@[<1>Replace(%S,@ %S)@]" pattern replacement
  | Prepend s -> Format.fprintf ppf "Prepend(%S)" s
  | Byte_level { add_prefix_space; use_regex } ->
      Format.fprintf ppf "@[<1>ByteLevel(add_prefix_space=%b,@ use_regex=%b)@]"
        add_prefix_space use_regex
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
  | Replace { pattern; replacement; _ } ->
      typed_with "Replace"
        [
          ("pattern", json_obj [ ("String", Jsont.Json.string pattern) ]);
          ("content", Jsont.Json.string replacement);
        ]
  | Prepend prefix ->
      typed_with "Prepend" [ ("prepend", Jsont.Json.string prefix) ]
  | Byte_level { add_prefix_space; use_regex } ->
      typed_with "ByteLevel"
        [
          ("add_prefix_space", Jsont.Json.bool add_prefix_space);
          ("use_regex", Jsont.Json.bool use_regex);
        ]
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
                match Jsont.Json.find_mem "String" pf with
                | Some (_, Jsont.String (p, _)) -> Ok p
                | _ -> Error err_replace_invalid_pattern)
            | _ -> Error err_replace_missing_pattern
          in
          let replacement =
            match find "content" with
            | Some (Jsont.String (r, _)) -> Ok r
            | _ -> Error err_replace_missing_content
          in
          Result.bind pattern (fun p ->
              Result.map
                (fun r -> replace ~pattern:p ~replacement:r)
                replacement)
      | Some (Jsont.String ("Prepend", _)) -> (
          match find "prepend" with
          | Some (Jsont.String (p, _)) -> Ok (Prepend p)
          | _ -> Error err_prepend_missing)
      | Some (Jsont.String ("ByteLevel", _)) ->
          Ok
            (Byte_level
               {
                 add_prefix_space = get_bool "add_prefix_space" false;
                 use_regex = get_bool "use_regex" false;
               })
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
