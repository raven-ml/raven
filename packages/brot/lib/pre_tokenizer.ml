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
  | Char_delimiter of char
  | Digits of { individual : bool }
  | Metaspace of {
      replacement : char;
      prepend_scheme : prepend_scheme;
      split : bool;
    }
  | Sequence of t list
  | Fixed_length of { length : int }
  | Unicode_scripts

(* Errors *)

let strf = Printf.sprintf
let err_unknown_behavior s = strf "unknown punctuation behavior '%s'" s
let err_unknown_scheme s = strf "unknown prepend_scheme '%s'" s
let err_unsupported_type s = strf "unsupported pre-tokenizer type '%s'" s
let err_expected_char name = strf "expected single character for '%s'" name
let err_missing_type = "missing 'type' field"
let err_expected_object = "expected JSON object"
let err_missing_behavior = "missing 'behavior' field"
let err_split_missing = "requires 'pattern' and 'behavior'"
let err_char_delim_missing = "requires 'delimiter'"
let err_metaspace_missing = "requires 'replacement' and 'prepend_scheme'"
let err_sequence_missing = "requires 'pretokenizers' list"
let err_fixed_length = "requires positive length"

(* Character classification *)

(* ASCII property table: packed flags for O(1) classification. bit 0:
   whitespace, bit 1: alphabetic, bit 2: numeric, bit 3: punctuation *)

let ascii_props =
  let t = Array.make 128 0 in
  for i = 9 to 13 do
    t.(i) <- t.(i) lor 1
  done;
  t.(32) <- t.(32) lor 1;
  for i = 65 to 90 do
    t.(i) <- t.(i) lor 2
  done;
  for i = 97 to 122 do
    t.(i) <- t.(i) lor 2
  done;
  for i = 48 to 57 do
    t.(i) <- t.(i) lor 4
  done;
  List.iter
    (fun i -> t.(i) <- t.(i) lor 8)
    [
      33;
      34;
      35;
      37;
      38;
      39;
      40;
      41;
      42;
      44;
      45;
      46;
      47;
      58;
      59;
      63;
      64;
      91;
      92;
      93;
      95;
      123;
      125;
    ];
  t

let[@inline] is_whitespace code =
  if code < 128 then Array.unsafe_get ascii_props code land 1 <> 0
  else Uucp.White.is_white_space (Uchar.of_int code)

let[@inline] is_alphabetic code =
  if code < 128 then Array.unsafe_get ascii_props code land 2 <> 0
  else Uucp.Alpha.is_alphabetic (Uchar.of_int code)

let[@inline] is_numeric code =
  if code < 128 then Array.unsafe_get ascii_props code land 4 <> 0
  else
    match Uucp.Gc.general_category (Uchar.of_int code) with
    | `Nd | `Nl | `No -> true
    | _ -> false

let[@inline] is_punctuation code =
  if code < 128 then Array.unsafe_get ascii_props code land 8 <> 0
  else
    match Uucp.Gc.general_category (Uchar.of_int code) with
    | `Pc | `Pd | `Pe | `Pf | `Pi | `Po | `Ps -> true
    | _ -> false

(* Returns (codepoint lsl 3) lor byte_length — zero allocation. *)
let[@inline] utf8_next s i =
  let c = Char.code (String.unsafe_get s i) in
  if c < 0x80 then (c lsl 3) lor 1
  else if c < 0xE0 then
    (((c land 0x1F) lsl 6)
    lor (Char.code (String.unsafe_get s (i + 1)) land 0x3F))
    lsl 3
    lor 2
  else if c < 0xF0 then
    (((c land 0x0F) lsl 12)
    lor ((Char.code (String.unsafe_get s (i + 1)) land 0x3F) lsl 6)
    lor (Char.code (String.unsafe_get s (i + 2)) land 0x3F))
    lsl 3
    lor 3
  else
    (((c land 0x07) lsl 18)
    lor ((Char.code (String.unsafe_get s (i + 1)) land 0x3F) lsl 12)
    lor ((Char.code (String.unsafe_get s (i + 2)) land 0x3F) lsl 6)
    lor (Char.code (String.unsafe_get s (i + 3)) land 0x3F))
    lsl 3
    lor 4

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

let byte_level_encode text =
  let len = String.length text in
  (* Worst case: every byte remaps to a 2-byte UTF-8 sequence *)
  let result = Bytes.create (len * 2) in
  let j = ref 0 in
  for i = 0 to len - 1 do
    let u =
      Array.unsafe_get byte_to_unicode (Char.code (String.unsafe_get text i))
    in
    if u < 128 then begin
      Bytes.unsafe_set result !j (Char.unsafe_chr u);
      incr j
    end
    else begin
      Bytes.unsafe_set result !j (Char.unsafe_chr (0xC0 lor (u lsr 6)));
      Bytes.unsafe_set result (!j + 1)
        (Char.unsafe_chr (0x80 lor (u land 0x3F)));
      j := !j + 2
    end
  done;
  Bytes.sub_string result 0 !j

let byte_level_encode_range text ~start ~len =
  let result = Bytes.create (len * 2) in
  let j = ref 0 in
  for i = start to start + len - 1 do
    let u =
      Array.unsafe_get byte_to_unicode (Char.code (String.unsafe_get text i))
    in
    if u < 128 then begin
      Bytes.unsafe_set result !j (Char.unsafe_chr u);
      incr j
    end
    else begin
      Bytes.unsafe_set result !j (Char.unsafe_chr (0xC0 lor (u lsr 6)));
      Bytes.unsafe_set result (!j + 1)
        (Char.unsafe_chr (0x80 lor (u land 0x3F)));
      j := !j + 2
    end
  done;
  Bytes.sub_string result 0 !j

let byte_level_decode text =
  let len = String.length text in
  let result = Buffer.create len in
  let i = ref 0 in
  while !i < len do
    let b0 = Char.code (String.unsafe_get text !i) in
    if b0 < 128 then begin
      (* ASCII: direct lookup, no utf8_next needed *)
      let byte = Array.unsafe_get unicode_to_byte b0 in
      Buffer.add_char result
        (if byte >= 0 then Char.chr byte else Char.unsafe_chr b0);
      incr i
    end
    else begin
      let p = utf8_next text !i in
      let code = p lsr 3 and clen = p land 7 in
      let byte =
        if code < Array.length unicode_to_byte then unicode_to_byte.(code)
        else -1
      in
      if byte >= 0 then Buffer.add_char result (Char.chr byte)
      else
        for j = !i to !i + clen - 1 do
          Buffer.add_char result (String.unsafe_get text j)
        done;
      i := !i + clen
    end
  done;
  Buffer.contents result

let[@inline] is_other code =
  (not (is_whitespace code))
  && (not (is_alphabetic code))
  && not (is_numeric code)

let split_gpt2_pattern text =
  let len = String.length text in
  if len = 0 then []
  else
    let spans = ref [] in
    let pos = ref 0 in

    (* Try: optional leading space + run of chars matching a class.
       [ascii_mask]: bitmask into ascii_props for the ASCII fast path. [invert]:
       when true, match chars where (props land mask) = 0. [classify]: predicate
       for non-ASCII codepoints (slow path only). *)
    let try_space_run ~ascii_mask ~invert ~classify () =
      let start = !pos in
      let b0 = Char.code (String.unsafe_get text !pos) in
      let has_space =
        if b0 < 128 then Array.unsafe_get ascii_props b0 land 1 <> 0
        else is_whitespace b0
      in
      let run_start = if has_space then start + 1 else start in
      if run_start < len then
        let b = Char.code (String.unsafe_get text run_start) in
        let ok, clen =
          if b < 128 then
            let v = Array.unsafe_get ascii_props b land ascii_mask in
            ((if invert then v = 0 else v <> 0), 1)
          else
            let p = utf8_next text run_start in
            let code = p lsr 3 and cl = p land 7 in
            (classify code, cl)
        in
        if ok then (
          let j = ref (run_start + clen) in
          let continue = ref true in
          while !j < len && !continue do
            let b = Char.code (String.unsafe_get text !j) in
            if b < 128 then
              let v = Array.unsafe_get ascii_props b land ascii_mask in
              if if invert then v = 0 else v <> 0 then j := !j + 1
              else continue := false
            else
              let p = utf8_next text !j in
              if classify (p lsr 3) then j := !j + (p land 7)
              else continue := false
          done;
          spans := (start, !j - start) :: !spans;
          pos := !j;
          true)
        else false
      else false
    in

    let[@inline] next_is_alnum next_pos =
      if next_pos >= len then false
      else
        let nb = Char.code (String.unsafe_get text next_pos) in
        if nb < 128 then Array.unsafe_get ascii_props nb land 6 <> 0
        else
          let nc = utf8_next text next_pos lsr 3 in
          is_alphabetic nc || is_numeric nc
    in

    let rec loop () =
      if !pos >= len then ()
      else begin
        (* 1. Contractions: 's 't 'm 'd 're 've 'll *)
        let matched_contraction =
          text.[!pos] = '\''
          &&
          let remaining = len - !pos in
          remaining >= 2
          &&
          let c1 = String.unsafe_get text (!pos + 1) in
          if c1 = 's' || c1 = 't' || c1 = 'm' || c1 = 'd' then (
            spans := (!pos, 2) :: !spans;
            pos := !pos + 2;
            true)
          else
            remaining >= 3
            &&
            let c2 = String.unsafe_get text (!pos + 2) in
            if
              (c1 = 'r' && c2 = 'e')
              || (c1 = 'v' && c2 = 'e')
              || (c1 = 'l' && c2 = 'l')
            then (
              spans := (!pos, 3) :: !spans;
              pos := !pos + 3;
              true)
            else false
        in
        if matched_contraction then ()
        else if
          try_space_run ~ascii_mask:2 ~invert:false ~classify:is_alphabetic ()
        then ()
        else if
          try_space_run ~ascii_mask:4 ~invert:false ~classify:is_numeric ()
        then ()
        else if try_space_run ~ascii_mask:7 ~invert:true ~classify:is_other ()
        then ()
        (* 5 & 6. Whitespace run *)
          else begin
          let b0 = Char.code (String.unsafe_get text !pos) in
          let is_ws, clen =
            if b0 < 128 then (Array.unsafe_get ascii_props b0 land 1 <> 0, 1)
            else
              let p = utf8_next text !pos in
              let code = p lsr 3 and cl = p land 7 in
              (is_whitespace code, cl)
          in
          if is_ws then begin
            let j = ref (!pos + clen) in
            let continue = ref true in
            while !j < len && !continue do
              let b = Char.code (String.unsafe_get text !j) in
              if b < 128 then
                if Array.unsafe_get ascii_props b land 1 <> 0 then
                  if next_is_alnum (!j + 1) && b = 0x20 then continue := false
                  else j := !j + 1
                else continue := false
              else
                let p = utf8_next text !j in
                let code = p lsr 3 and cl = p land 7 in
                if is_whitespace code then
                  if next_is_alnum (!j + cl) && code = 0x20 then
                    continue := false
                  else j := !j + cl
                else continue := false
            done;
            spans := (!pos, !j - !pos) :: !spans;
            pos := !j
          end
          else begin
            (* Fallback: single character *)
            spans := (!pos, clen) :: !spans;
            pos := !pos + clen
          end
        end;
        loop ()
      end
    in
    loop ();
    List.rev !spans

(* Pre-tokenize implementations *)

let pre_tokenize_whitespace_split text =
  let pieces = ref [] in
  let start = ref (-1) in
  let i = ref 0 in
  let len = String.length text in
  let flush () =
    if !start >= 0 then begin
      pieces := (String.sub text !start (!i - !start), (!start, !i)) :: !pieces;
      start := -1
    end
  in
  while !i < len do
    let b = Char.code (String.unsafe_get text !i) in
    if b < 128 then
      if Array.unsafe_get ascii_props b land 1 <> 0 then (
        flush ();
        i := !i + 1)
      else (
        if !start < 0 then start := !i;
        i := !i + 1)
    else
      let p = utf8_next text !i in
      let code = p lsr 3 and l = p land 7 in
      if is_whitespace code then (
        flush ();
        i := !i + l)
      else (
        if !start < 0 then start := !i;
        i := !i + l)
  done;
  flush ();
  List.rev !pieces

let pre_tokenize_whitespace text =
  let pieces = ref [] in
  let start = ref (-1) in
  let i = ref 0 in
  let len = String.length text in
  let in_word = ref false in
  let in_punct = ref false in
  let flush () =
    if !start >= 0 then begin
      pieces := (String.sub text !start (!i - !start), (!start, !i)) :: !pieces;
      start := -1
    end
  in
  while !i < len do
    let b = Char.code (String.unsafe_get text !i) in
    if b < 128 then
      let p = Array.unsafe_get ascii_props b in
      if p land 6 <> 0 || b = 95 then (
        if !in_punct then flush ();
        if !start < 0 then start := !i;
        in_word := true;
        in_punct := false;
        i := !i + 1)
      else if p land 1 <> 0 then (
        flush ();
        in_word := false;
        in_punct := false;
        i := !i + 1)
      else (
        if !in_word then flush ();
        if !start < 0 then start := !i;
        in_word := false;
        in_punct := true;
        i := !i + 1)
    else
      let p = utf8_next text !i in
      let code = p lsr 3 and l = p land 7 in
      if is_alphabetic code || is_numeric code then (
        if !in_punct then flush ();
        if !start < 0 then start := !i;
        in_word := true;
        in_punct := false;
        i := !i + l)
      else if is_whitespace code then (
        flush ();
        in_word := false;
        in_punct := false;
        i := !i + l)
      else (
        if !in_word then flush ();
        if !start < 0 then start := !i;
        in_word := false;
        in_punct := true;
        i := !i + l)
  done;
  flush ();
  List.rev !pieces

let pre_tokenize_byte_level ~add_prefix_space ~use_regex ~trim_offsets:_ text =
  let orig_len = String.length text in
  let text, prefix_added =
    if
      add_prefix_space && orig_len > 0
      && not (is_whitespace (Char.code text.[0]))
    then (" " ^ text, true)
    else (text, false)
  in
  if use_regex then
    let spans = split_gpt2_pattern text in
    List.map
      (fun (start, plen) ->
        let o_start =
          if prefix_added then if start = 0 then 0 else start - 1 else start
        in
        let o_end =
          min orig_len (if prefix_added then start + plen - 1 else start + plen)
        in
        (byte_level_encode_range text ~start ~len:plen, (max 0 o_start, o_end)))
      spans
  else [ (byte_level_encode text, (0, orig_len)) ]

let pre_tokenize_bert text =
  let pieces = ref [] in
  let start = ref (-1) in
  let i = ref 0 in
  let len = String.length text in
  let flush () =
    if !start >= 0 then begin
      pieces := (String.sub text !start (!i - !start), (!start, !i)) :: !pieces;
      start := -1
    end
  in
  while !i < len do
    let b = Char.code (String.unsafe_get text !i) in
    if b < 128 then
      let p = Array.unsafe_get ascii_props b in
      if p land 1 <> 0 then (
        flush ();
        i := !i + 1)
      else if p land 8 <> 0 then (
        flush ();
        pieces := (String.sub text !i 1, (!i, !i + 1)) :: !pieces;
        i := !i + 1)
      else (
        if !start < 0 then start := !i;
        i := !i + 1)
    else
      let p = utf8_next text !i in
      let code = p lsr 3 and l = p land 7 in
      if is_whitespace code then (
        flush ();
        i := !i + l)
      else if is_punctuation code then (
        flush ();
        pieces := (String.sub text !i l, (!i, !i + l)) :: !pieces;
        i := !i + l)
      else (
        if !start < 0 then start := !i;
        i := !i + l)
  done;
  flush ();
  List.rev !pieces

let pre_tokenize_punctuation ~behavior text =
  let pieces = ref [] in
  let start = ref (-1) in
  let i = ref 0 in
  let len = String.length text in
  let last_was_punc = ref false in
  let flush () =
    if !start >= 0 then begin
      pieces := (String.sub text !start (!i - !start), (!start, !i)) :: !pieces;
      start := -1
    end
  in
  let handle_char is_p l =
    if is_p then (
      (match behavior with
      | `Isolated ->
          flush ();
          pieces := (String.sub text !i l, (!i, !i + l)) :: !pieces
      | `Removed -> flush ()
      | `Merged_with_previous -> if !start < 0 then start := !i
      | `Merged_with_next ->
          flush ();
          start := !i
      | `Contiguous ->
          if not (!start >= 0 && !last_was_punc) then begin
            flush ();
            start := !i
          end);
      last_was_punc := true;
      i := !i + l)
    else (
      if behavior = `Contiguous && !start >= 0 && !last_was_punc then flush ();
      if !start < 0 then start := !i;
      i := !i + l;
      last_was_punc := false)
  in
  while !i < len do
    let b = Char.code (String.unsafe_get text !i) in
    if b < 128 then handle_char (Array.unsafe_get ascii_props b land 8 <> 0) 1
    else
      let p = utf8_next text !i in
      let code = p lsr 3 and l = p land 7 in
      handle_char (is_punctuation code) l
  done;
  flush ();
  List.rev !pieces

let pre_tokenize_split ~pattern ~behavior ~invert text =
  let plen = String.length pattern in
  if plen = 0 then [ (text, (0, String.length text)) ]
  else
    let pieces = ref [] in
    let current = Buffer.create 16 in
    let current_start = ref 0 in
    let i = ref 0 in
    let flush_current () =
      if Buffer.length current > 0 then (
        pieces :=
          ( Buffer.contents current,
            (!current_start, !current_start + Buffer.length current) )
          :: !pieces;
        Buffer.clear current)
    in
    while !i < String.length text do
      let is_match =
        !i + plen <= String.length text && String.sub text !i plen = pattern
      in
      let is_delim = if invert then not is_match else is_match in
      let delim_len = if is_delim then if invert then 1 else plen else 1 in
      if is_delim then (
        (match behavior with
        | `Removed -> flush_current ()
        | `Isolated ->
            flush_current ();
            let delim_str = String.sub text !i delim_len in
            pieces := (delim_str, (!i, !i + delim_len)) :: !pieces
        | `Merged_with_previous ->
            Buffer.add_string current (String.sub text !i delim_len);
            flush_current ()
        | `Merged_with_next ->
            flush_current ();
            current_start := !i;
            Buffer.add_string current (String.sub text !i delim_len)
        | `Contiguous ->
            if Buffer.length current > 0 && is_delim then
              Buffer.add_string current (String.sub text !i delim_len)
            else (
              flush_current ();
              Buffer.add_string current (String.sub text !i delim_len)));
        i := !i + delim_len)
      else (
        if Buffer.length current = 0 then current_start := !i;
        Buffer.add_string current (String.sub text !i 1);
        i := !i + 1)
    done;
    flush_current ();
    List.rev !pieces

let pre_tokenize_digits ~individual text =
  let pieces = ref [] in
  let start = ref (-1) in
  let i = ref 0 in
  let len = String.length text in
  let in_digits = ref false in
  let flush () =
    if !start >= 0 then begin
      pieces := (String.sub text !start (!i - !start), (!start, !i)) :: !pieces;
      start := -1
    end
  in
  let handle_char is_d l =
    if individual && is_d then (
      flush ();
      pieces := (String.sub text !i l, (!i, !i + l)) :: !pieces;
      i := !i + l)
    else (
      if is_d <> !in_digits then (
        flush ();
        in_digits := is_d);
      if !start < 0 then start := !i;
      i := !i + l)
  in
  while !i < len do
    let b = Char.code (String.unsafe_get text !i) in
    if b < 128 then handle_char (Array.unsafe_get ascii_props b land 4 <> 0) 1
    else
      let p = utf8_next text !i in
      let code = p lsr 3 and l = p land 7 in
      handle_char (is_numeric code) l
  done;
  flush ();
  List.rev !pieces

let pre_tokenize_metaspace ~replacement ~prepend_scheme ~split text =
  let repl = String.make 1 replacement in
  let text =
    match prepend_scheme with
    | (`Always | `First) when String.length text > 0 && text.[0] <> ' ' ->
        " " ^ text
    | _ -> text
  in
  let len = String.length text in
  let buf = Buffer.create len in
  let i = ref 0 in
  while !i < len do
    if text.[!i] = ' ' then (
      Buffer.add_string buf repl;
      incr i)
    else
      let l = utf8_next text !i land 7 in
      Buffer.add_substring buf text !i l;
      i := !i + l
  done;
  let transformed = Buffer.contents buf in
  if split then (
    let tlen = String.length transformed in
    let rlen = String.length repl in
    let splits = ref [] in
    let start = ref 0 in
    let pos = ref 0 in
    while !pos < tlen do
      if !pos + rlen <= tlen && String.sub transformed !pos rlen = repl then (
        if !pos > !start then
          splits :=
            (String.sub transformed !start (!pos - !start), (!start, !pos))
            :: !splits;
        start := !pos;
        pos := !pos + rlen)
      else incr pos
    done;
    if !pos > !start then
      splits :=
        (String.sub transformed !start (!pos - !start), (!start, !pos))
        :: !splits;
    List.rev !splits)
  else [ (transformed, (0, len)) ]

let pre_tokenize_fixed_length ~length text =
  if length <= 0 || String.length text = 0 then []
  else
    let pieces = ref [] in
    let len = String.length text in
    let i = ref 0 in
    while !i < len do
      let start = !i in
      let count = ref 0 in
      while !i < len && !count < length do
        let l = utf8_next text !i land 7 in
        i := !i + l;
        incr count
      done;
      pieces := (String.sub text start (!i - start), (start, !i)) :: !pieces
    done;
    List.rev !pieces

type script = [ `Any | Uucp.Script.t ]

let fixed_script code : script =
  if code = 0x30FC then (`Hani :> script)
  else if is_whitespace code then `Any
  else
    match Uucp.Script.script (Uchar.of_int code) with
    | `Hira | `Kana -> (`Hani :> script)
    | s -> (s :> script)

let pre_tokenize_unicode_scripts text =
  let pieces = ref [] in
  let start = ref (-1) in
  let len = String.length text in
  let i = ref 0 in
  let last_script = ref None in
  let flush () =
    if !start >= 0 then begin
      pieces := (String.sub text !start (!i - !start), (!start, !i)) :: !pieces;
      start := -1
    end
  in
  let emit (script : script) l =
    if
      script <> `Any && !last_script <> Some `Any && !last_script <> Some script
    then flush ();
    if !start < 0 then start := !i;
    i := !i + l;
    if script <> `Any then last_script := Some script
  in
  while !i < len do
    let b = Char.code (String.unsafe_get text !i) in
    if b < 128 then
      let p = Array.unsafe_get ascii_props b in
      let script : script =
        if p land 1 <> 0 then `Any else if p land 2 <> 0 then `Latn else `Zyyy
      in
      emit script 1
    else
      let p = utf8_next text !i in
      let code = p lsr 3 and l = p land 7 in
      emit (fixed_script code) l
  done;
  flush ();
  List.rev !pieces

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

let char_delimiter c = Char_delimiter c

let digits ?(individual_digits = false) () =
  Digits { individual = individual_digits }

let metaspace ?(replacement = '_') ?(prepend_scheme = `Always) ?(split = true)
    () =
  Metaspace { replacement; prepend_scheme; split }

let unicode_scripts () = Unicode_scripts
let fixed_length n = Fixed_length { length = n }
let sequence ts = Sequence ts

(* Dispatch *)

let rec pre_tokenize t text =
  match t with
  | Whitespace -> pre_tokenize_whitespace text
  | Whitespace_split -> pre_tokenize_whitespace_split text
  | Bert -> pre_tokenize_bert text
  | Byte_level { add_prefix_space; use_regex; trim_offsets } ->
      pre_tokenize_byte_level ~add_prefix_space ~use_regex ~trim_offsets text
  | Punctuation { behavior } -> pre_tokenize_punctuation ~behavior text
  | Split { pattern; behavior; invert } ->
      pre_tokenize_split ~pattern ~behavior ~invert text
  | Char_delimiter c ->
      pre_tokenize_split ~pattern:(String.make 1 c) ~behavior:`Removed
        ~invert:false text
  | Digits { individual } -> pre_tokenize_digits ~individual text
  | Metaspace { replacement; prepend_scheme; split } ->
      pre_tokenize_metaspace ~replacement ~prepend_scheme ~split text
  | Unicode_scripts -> pre_tokenize_unicode_scripts text
  | Fixed_length { length } -> pre_tokenize_fixed_length ~length text
  | Sequence ts -> pre_tokenize_sequence ts text

and pre_tokenize_sequence ts text =
  let initial = [ (text, (0, String.length text)) ] in
  List.fold_left
    (fun pieces t ->
      List.concat_map
        (fun (s, (o_start, _)) ->
          let sub_pieces = pre_tokenize t s in
          List.map
            (fun (p, (p_start, p_end)) ->
              (p, (o_start + p_start, o_start + p_end)))
            sub_pieces)
        pieces)
    initial ts

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
  | `First -> "First"
  | `Never -> "Never"
  | `Always -> "Always"

let scheme_of_string = function
  | "First" -> Ok `First
  | "Never" -> Ok `Never
  | "Always" -> Ok `Always
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
  | Char_delimiter c -> Format.fprintf ppf "CharDelimiter(%C)" c
  | Digits { individual } ->
      Format.fprintf ppf "Digits(individual=%b)" individual
  | Metaspace { replacement; prepend_scheme; split } ->
      Format.fprintf ppf "@[<1>Metaspace(%C,@ %s,@ split=%b)@]" replacement
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
          ("pattern", Jsont.Json.string pattern);
          ("behavior", Jsont.Json.string (behavior_to_string behavior));
          ("invert", Jsont.Json.bool invert);
        ]
  | Char_delimiter delimiter ->
      json_obj
        [
          ("type", Jsont.Json.string "CharDelimiterSplit");
          ("delimiter", Jsont.Json.string (String.make 1 delimiter));
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
          ("replacement", Jsont.Json.string (String.make 1 replacement));
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

let char_of_field name = function
  | Jsont.String (s, _) when String.length s = 1 -> Ok s.[0]
  | _ -> Error (err_expected_char name)

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
          | Some (Jsont.String (s, _)) ->
              Result.map
                (fun b -> Punctuation { behavior = b })
                (behavior_of_string s)
          | _ -> Error err_missing_behavior)
      | Some (Jsont.String ("Split", _)) -> (
          match (find_field "pattern" fields, find_field "behavior" fields) with
          | ( Some (Jsont.String (pattern, _)),
              Some (Jsont.String (behavior_str, _)) ) ->
              Result.map
                (fun behavior ->
                  let invert = bool_field "invert" false fields in
                  Split { pattern; behavior; invert })
                (behavior_of_string behavior_str)
          | _ -> Error err_split_missing)
      | Some (Jsont.String ("CharDelimiterSplit", _)) -> (
          match find_field "delimiter" fields with
          | Some v ->
              Result.map
                (fun c -> Char_delimiter c)
                (char_of_field "delimiter" v)
          | None -> Error err_char_delim_missing)
      | Some (Jsont.String ("Digits", _)) ->
          let individual = bool_field "individual_digits" false fields in
          Ok (Digits { individual })
      | Some (Jsont.String ("Metaspace", _)) -> (
          match
            (find_field "replacement" fields, find_field "prepend_scheme" fields)
          with
          | Some (Jsont.String (repl, _)), Some (Jsont.String (scheme, _))
            when String.length repl = 1 ->
              Result.map
                (fun prepend_scheme ->
                  let split = bool_field "split" true fields in
                  Metaspace { replacement = repl.[0]; prepend_scheme; split })
                (scheme_of_string scheme)
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
