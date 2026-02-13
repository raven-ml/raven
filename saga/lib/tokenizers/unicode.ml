(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type normalization =
  | NFC (* Canonical Decomposition, followed by Canonical Composition *)
  | NFD (* Canonical Decomposition *)
  | NFKC (* Compatibility Decomposition, followed by Canonical Composition *)
  | NFKD (* Compatibility Decomposition *)

type char_category =
  | Letter
  | Number
  | Punctuation
  | Symbol
  | Whitespace
  | Control
  | Other

let categorize_char u =
  match Unicode_data.general_category (Uchar.to_int u) with
  | `Lu | `Ll | `Lt | `Lm | `Lo -> Letter
  | `Nd | `Nl | `No -> Number
  | `Pc | `Pd | `Ps | `Pe | `Pi | `Pf | `Po -> Punctuation
  | `Sm | `Sc | `Sk | `So -> Symbol
  | `Zs | `Zl | `Zp -> Whitespace
  | `Cc | `Cf | `Cs | `Co | `Cn -> Control
  | _ -> Other

let is_whitespace u =
  match categorize_char u with Whitespace -> true | _ -> false

let is_punctuation u =
  match categorize_char u with Punctuation -> true | _ -> false

let is_word_char u =
  match categorize_char u with Letter | Number -> true | _ -> false

let is_cjk u =
  let cp = Uchar.to_int u in
  (* Common CJK ranges *)
  (cp >= 0x4E00 && cp <= 0x9FFF)
  (* CJK Unified Ideographs *)
  || (cp >= 0x3400 && cp <= 0x4DBF)
  (* CJK Extension A *)
  || (cp >= 0x3040 && cp <= 0x309F)
  (* Hiragana *)
  || (cp >= 0x30A0 && cp <= 0x30FF)
  ||
  (* Katakana *)
  (cp >= 0xAC00 && cp <= 0xD7AF)
(* Hangul Syllables *)

let normalize form text =
  try
    let b = Buffer.create (String.length text) in
    let add u = Buffer.add_utf_8_uchar b u in

    (* Apply normalization based on form *)
    let normalize_char =
      match form with
      | NFC ->
          fun u ->
            (* For now, just pass through - proper implementation needs full
               Unicode tables *)
            add u
      | NFD ->
          fun u ->
            (* Decompose but don't recompose *)
            add u
      | NFKC ->
          fun u ->
            (* Compatibility decomposition + composition *)
            add u
      | NFKD ->
          fun u ->
            (* Compatibility decomposition *)
            add u
    in

    let len = String.length text in
    let rec loop i =
      if i >= len then ()
      else
        let d = String.get_utf_8_uchar text i in
        let n = Uchar.utf_decode_length d in
        if Uchar.utf_decode_is_valid d then
          normalize_char (Uchar.utf_decode_uchar d);
        loop (i + n)
    in
    loop 0;
    Buffer.contents b
  with e ->
    Nx_core.Error.invalid ~op:"normalize" ~what:"unicode in text"
      ~reason:(Printexc.to_string e) ()

let case_fold text =
  try
    let b = Buffer.create (String.length text) in
    let len = String.length text in
    let rec loop i =
      if i >= len then ()
      else
        let d = String.get_utf_8_uchar text i in
        let n = Uchar.utf_decode_length d in
        (if Uchar.utf_decode_is_valid d then
           let u = Uchar.utf_decode_uchar d in
           let folded = Unicode_data.case_fold (Uchar.to_int u) in
           List.iter (fun cp -> Buffer.add_utf_8_uchar b (Uchar.of_int cp)) folded);
        loop (i + n)
    in
    loop 0;
    Buffer.contents b
  with e ->
    Nx_core.Error.invalid ~op:"case_fold" ~what:"unicode in text"
      ~reason:(Printexc.to_string e) ()

let strip_accents text =
  try
    let b = Buffer.create (String.length text) in
    let len = String.length text in
    let rec loop i =
      if i >= len then ()
      else
        let d = String.get_utf_8_uchar text i in
        let n = Uchar.utf_decode_length d in
        (if Uchar.utf_decode_is_valid d then
           let u = Uchar.utf_decode_uchar d in
           if
             not
               (match Unicode_data.general_category (Uchar.to_int u) with
               | `Mn | `Mc | `Me -> true
               | _ -> false)
           then Buffer.add_utf_8_uchar b u);
        loop (i + n)
    in
    loop 0;
    Buffer.contents b
  with e ->
    Nx_core.Error.invalid ~op:"strip_accents" ~what:"unicode in text"
      ~reason:(Printexc.to_string e) ()

let clean_text ?(remove_control = true) ?(normalize_whitespace = true) text =
  try
    let b = Buffer.create (String.length text) in
    let last_was_space = ref false in
    let len = String.length text in
    let rec loop i =
      if i >= len then ()
      else
        let d = String.get_utf_8_uchar text i in
        let n = Uchar.utf_decode_length d in
        (if Uchar.utf_decode_is_valid d then
           let u = Uchar.utf_decode_uchar d in
           let cat = categorize_char u in
           match cat with
           | Control when remove_control -> ()
           | Whitespace when normalize_whitespace ->
               if not !last_was_space then (
                 Buffer.add_utf_8_uchar b (Uchar.of_int 0x20);
                 last_was_space := true)
           | _ ->
               Buffer.add_utf_8_uchar b u;
               last_was_space := false);
        loop (i + n)
    in
    loop 0;

    let result = Buffer.contents b in
    if normalize_whitespace then String.trim result else result
  with e ->
    Nx_core.Error.invalid ~op:"clean_text" ~what:"unicode in text"
      ~reason:(Printexc.to_string e) ()

let split_words text =
  try
    let words = ref [] in
    let current_word = Buffer.create 16 in

    let flush_word () =
      let word = Buffer.contents current_word in
      if word <> "" then words := word :: !words;
      Buffer.clear current_word
    in

    let len = String.length text in
    let rec loop i =
      if i >= len then (
        flush_word ();
        List.rev !words)
      else
        let d = String.get_utf_8_uchar text i in
        let n = Uchar.utf_decode_length d in
        (if Uchar.utf_decode_is_valid d then
           let u = Uchar.utf_decode_uchar d in
           if is_cjk u then (
             flush_word ();
             Buffer.add_utf_8_uchar current_word u;
             flush_word ())
           else if is_word_char u then Buffer.add_utf_8_uchar current_word u
           else flush_word ());
        loop (i + n)
    in
    loop 0
  with e ->
    Nx_core.Error.invalid ~op:"split_words" ~what:"unicode in text"
      ~reason:(Printexc.to_string e) ()

let is_valid_utf8 text = String.is_valid_utf_8 text

let remove_emoji text =
  try
    let b = Buffer.create (String.length text) in
    let len = String.length text in
    let rec loop i =
      if i >= len then ()
      else
        let d = String.get_utf_8_uchar text i in
        let n = Uchar.utf_decode_length d in
        (if Uchar.utf_decode_is_valid d then
           let u = Uchar.utf_decode_uchar d in
           let cat = categorize_char u in
           if cat <> Symbol then Buffer.add_utf_8_uchar b u);
        loop (i + n)
    in
    loop 0;
    Buffer.contents b
  with e ->
    Nx_core.Error.invalid ~op:"remove_emoji" ~what:"unicode in text"
      ~reason:(Printexc.to_string e) ()
