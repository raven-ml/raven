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
  match Uucp.Gc.general_category u with
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
    let add u = Uutf.Buffer.add_utf_8 b u in

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

    (* Process each character *)
    let decoder = Uutf.decoder (`String text) in
    let rec loop () =
      match Uutf.decode decoder with
      | `Uchar u ->
          normalize_char u;
          loop ()
      | `End -> ()
      | `Malformed _ ->
          (* Skip malformed sequences *)
          loop ()
      | `Await -> assert false
    in
    loop ();
    Buffer.contents b
  with e ->
    Nx_core.Error.invalid ~op:"normalize" ~what:"unicode in text"
      ~reason:(Printexc.to_string e) ()

let case_fold text =
  try
    let b = Buffer.create (String.length text) in
    let decoder = Uutf.decoder (`String text) in
    let rec loop () =
      match Uutf.decode decoder with
      | `Uchar u ->
          (* Use Uucp for proper case folding *)
          let folded =
            match Uucp.Case.Fold.fold u with `Self -> [ u ] | `Uchars us -> us
          in
          List.iter (Uutf.Buffer.add_utf_8 b) folded;
          loop ()
      | `End -> ()
      | `Malformed _ -> loop ()
      | `Await -> assert false
    in
    loop ();
    Buffer.contents b
  with e ->
    Nx_core.Error.invalid ~op:"case_fold" ~what:"unicode in text"
      ~reason:(Printexc.to_string e) ()

let strip_accents text =
  try
    let b = Buffer.create (String.length text) in
    let decoder = Uutf.decoder (`String text) in
    let rec loop () =
      match Uutf.decode decoder with
      | `Uchar u ->
          (* Check if it's a combining diacritical mark *)
          if
            not
              (Uucp.Gc.general_category u = `Mn
              || Uucp.Gc.general_category u = `Mc
              || Uucp.Gc.general_category u = `Me)
          then Uutf.Buffer.add_utf_8 b u;
          loop ()
      | `End -> ()
      | `Malformed _ -> loop ()
      | `Await -> assert false
    in
    loop ();
    Buffer.contents b
  with e ->
    Nx_core.Error.invalid ~op:"strip_accents" ~what:"unicode in text"
      ~reason:(Printexc.to_string e) ()

let clean_text ?(remove_control = true) ?(normalize_whitespace = true) text =
  try
    let b = Buffer.create (String.length text) in
    let decoder = Uutf.decoder (`String text) in
    let last_was_space = ref false in

    let rec loop () =
      match Uutf.decode decoder with
      | `Uchar u ->
          let cat = categorize_char u in
          (match cat with
          | Control when remove_control -> () (* Skip control characters *)
          | Whitespace when normalize_whitespace ->
              if not !last_was_space then (
                Uutf.Buffer.add_utf_8 b (Uchar.of_int 0x20);
                (* Regular space *)
                last_was_space := true)
          | _ ->
              Uutf.Buffer.add_utf_8 b u;
              last_was_space := false);
          loop ()
      | `End -> ()
      | `Malformed _ -> loop ()
      | `Await -> assert false
    in
    loop ();

    (* Trim result *)
    let result = Buffer.contents b in
    if normalize_whitespace then String.trim result else result
  with e ->
    Nx_core.Error.invalid ~op:"clean_text" ~what:"unicode in text"
      ~reason:(Printexc.to_string e) ()

let split_words text =
  try
    let words = ref [] in
    let current_word = Buffer.create 16 in
    let decoder = Uutf.decoder (`String text) in

    let flush_word () =
      let word = Buffer.contents current_word in
      if word <> "" then words := word :: !words;
      Buffer.clear current_word
    in

    let rec loop () =
      match Uutf.decode decoder with
      | `Uchar u ->
          if is_cjk u then (
            (* For CJK, flush current word and treat each character as a word *)
            flush_word ();
            Uutf.Buffer.add_utf_8 current_word u;
            flush_word ())
          else if is_word_char u then Uutf.Buffer.add_utf_8 current_word u
          else flush_word ();
          loop ()
      | `End ->
          flush_word ();
          List.rev !words
      | `Malformed _ -> loop ()
      | `Await -> assert false
    in
    loop ()
  with e ->
    Nx_core.Error.invalid ~op:"split_words" ~what:"unicode in text"
      ~reason:(Printexc.to_string e) ()

let grapheme_count text =
  try
    let decoder = Uutf.decoder (`String text) in
    let count = ref 0 in
    let in_grapheme = ref false in

    let rec loop () =
      match Uutf.decode decoder with
      | `Uchar u ->
          (* Simple approximation - proper implementation needs grapheme
             segmentation *)
          let is_combining =
            match Uucp.Gc.general_category u with
            | `Mn | `Mc | `Me -> true
            | _ -> false
          in
          if not is_combining then (
            incr count;
            in_grapheme := true);
          loop ()
      | `End -> !count
      | `Malformed _ -> loop ()
      | `Await -> assert false
    in
    loop ()
  with e ->
    Nx_core.Error.invalid ~op:"grapheme_count" ~what:"unicode in text"
      ~reason:(Printexc.to_string e) ()

let is_valid_utf8 text =
  let decoder = Uutf.decoder ~encoding:`UTF_8 (`String text) in
  let rec loop () =
    match Uutf.decode decoder with
    | `Uchar _ -> loop ()
    | `End -> true
    | `Malformed _ -> false
    | `Await -> assert false
  in
  loop ()

let remove_emoji text =
  try
    let b = Buffer.create (String.length text) in
    let decoder = Uutf.decoder (`String text) in
    let rec loop () =
      match Uutf.decode decoder with
      | `Uchar u ->
          let cat = categorize_char u in
          if cat <> Symbol then Uutf.Buffer.add_utf_8 b u;
          loop ()
      | `End -> ()
      | `Malformed _ -> loop ()
      | `Await -> assert false
    in
    loop ();
    Buffer.contents b
  with e ->
    Nx_core.Error.invalid ~op:"remove_emoji" ~what:"unicode in text"
      ~reason:(Printexc.to_string e) ()
