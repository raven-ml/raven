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

(* Unicode text transforms *)

let normalize_utf8 nf text =
  let len = String.length text in
  if len = 0 then text
  else
    let rec all_ascii i =
      i >= len
      || (Char.code (String.unsafe_get text i) < 0x80 && all_ascii (i + 1))
    in
    if all_ascii 0 then text else Uunf_string.normalize_utf_8 nf text

let case_fold text =
  let len = String.length text in
  let rec needs_fold i =
    if i >= len then false
    else
      let byte = Char.code (String.unsafe_get text i) in
      if byte >= 0x41 && byte <= 0x5A then true
      else if byte >= 128 then true
      else needs_fold (i + 1)
  in
  if not (needs_fold 0) then text
  else
    let b = Buffer.create len in
    let i = ref 0 in
    while !i < len do
      let byte = Char.code (String.unsafe_get text !i) in
      if byte < 128 then (
        let c = if byte >= 0x41 && byte <= 0x5A then byte + 32 else byte in
        Buffer.add_char b (Char.unsafe_chr c);
        incr i)
      else
        let d = String.get_utf_8_uchar text !i in
        let n = Uchar.utf_decode_length d in
        (if Uchar.utf_decode_is_valid d then
           let u = Uchar.utf_decode_uchar d in
           match Uucp.Case.Fold.fold u with
           | `Self -> Buffer.add_utf_8_uchar b u
           | `Uchars us -> List.iter (fun u -> Buffer.add_utf_8_uchar b u) us);
        i := !i + n
    done;
    Buffer.contents b

let strip_accents_text text =
  let len = String.length text in
  let rec has_non_ascii i =
    if i >= len then false
    else if Char.code (String.unsafe_get text i) >= 128 then true
    else has_non_ascii (i + 1)
  in
  if not (has_non_ascii 0) then text
  else
    let b = Buffer.create len in
    let i = ref 0 in
    while !i < len do
      let byte = Char.code (String.unsafe_get text !i) in
      if byte < 128 then (
        Buffer.add_char b (Char.unsafe_chr byte);
        incr i)
      else
        let d = String.get_utf_8_uchar text !i in
        let n = Uchar.utf_decode_length d in
        (if Uchar.utf_decode_is_valid d then
           let u = Uchar.utf_decode_uchar d in
           match Uucp.Gc.general_category u with
           | `Mn | `Mc | `Me -> ()
           | _ -> Buffer.add_utf_8_uchar b u);
        i := !i + n
    done;
    Buffer.contents b

(* UTF-8 helpers *)

(* Returns (codepoint lsl 3) lor byte_length — zero allocation. *)
let[@inline] utf8_next s i =
  let d = String.get_utf_8_uchar s i in
  (Uchar.to_int (Uchar.utf_decode_uchar d) lsl 3) lor Uchar.utf_decode_length d

(* Character classification *)

let[@inline] is_whitespace code =
  code = 0x09 || code = 0x0A || code = 0x0D || code = 0x20
  || Uucp.White.is_white_space (Uchar.of_int code)

let[@inline] is_control code =
  if code = 0x09 || code = 0x0A || code = 0x0D then false
  else
    match Uucp.Gc.general_category (Uchar.of_int code) with
    | `Cc | `Cf | `Cn | `Co -> true
    | _ -> false

let[@inline] is_chinese_char code =
  (code >= 0x4E00 && code <= 0x9FFF)
  || (code >= 0x3400 && code <= 0x4DBF)
  || (code >= 0x20000 && code <= 0x2A6DF)
  || (code >= 0x2A700 && code <= 0x2B73F)
  || (code >= 0x2B740 && code <= 0x2B81F)
  || (code >= 0x2B920 && code <= 0x2CEAF)
  || (code >= 0xF900 && code <= 0xFAFF)
  || (code >= 0x2F800 && code <= 0x2FA1F)

(* Operations *)

let clean_text s =
  let len = String.length s in
  let buf = Buffer.create len in
  let i = ref 0 in
  while !i < len do
    let b0 = Char.code (String.unsafe_get s !i) in
    if b0 < 128 then begin
      if b0 = 9 || b0 = 10 || b0 = 13 || b0 = 32 then Buffer.add_char buf ' '
      else if b0 >= 33 && b0 < 127 then Buffer.add_char buf (Char.unsafe_chr b0);
      incr i
    end
    else begin
      let p = utf8_next s !i in
      let code = p lsr 3 and clen = p land 7 in
      if code <> 0xFFFD && not (is_control code) then
        if is_whitespace code then Buffer.add_char buf ' '
        else Buffer.add_substring buf s !i clen;
      i := !i + clen
    end
  done;
  Buffer.contents buf

let handle_chinese_chars s =
  let len = String.length s in
  let rec has_non_ascii i =
    i < len
    && (Char.code (String.unsafe_get s i) >= 128 || has_non_ascii (i + 1))
  in
  if not (has_non_ascii 0) then s
  else
    let buf = Buffer.create (len + (len / 4)) in
    let i = ref 0 in
    while !i < len do
      let b0 = Char.code (String.unsafe_get s !i) in
      if b0 < 128 then begin
        Buffer.add_char buf (Char.unsafe_chr b0);
        incr i
      end
      else begin
        let p = utf8_next s !i in
        let code = p lsr 3 and clen = p land 7 in
        if is_chinese_char code then (
          Buffer.add_char buf ' ';
          Buffer.add_substring buf s !i clen;
          Buffer.add_char buf ' ')
        else Buffer.add_substring buf s !i clen;
        i := !i + clen
      end
    done;
    Buffer.contents buf

let do_strip_accents s = strip_accents_text (normalize_utf8 `NFD s)
let do_lowercase s = case_fold s

let strip_whitespace s ~left ~right =
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
  if start = 0 && stop = len then s else String.sub s start (stop - start)

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

let apply_byte_level s ~add_prefix_space ~use_regex:_ =
  let s =
    if add_prefix_space && String.length s > 0 then
      let code = utf8_next s 0 lsr 3 in
      if is_whitespace code then s else " " ^ s
    else s
  in
  let len = String.length s in
  let buf = Buffer.create (len * 2) in
  for i = 0 to len - 1 do
    let b = Char.code (String.unsafe_get s i) in
    Buffer.add_utf_8_uchar buf (Uchar.of_int byte_to_unicode.(b))
  done;
  Buffer.contents buf

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

let rec apply t s =
  match t with
  | NFC -> normalize_utf8 `NFC s
  | NFD -> normalize_utf8 `NFD s
  | NFKC -> normalize_utf8 `NFKC s
  | NFKD -> normalize_utf8 `NFKD s
  | Lowercase -> do_lowercase s
  | Strip_accents -> do_strip_accents s
  | Strip { left; right } -> strip_whitespace s ~left ~right
  | Replace { compiled; replacement; _ } ->
      Re.replace_string compiled ~by:replacement s
  | Prepend prefix -> if String.length s = 0 then s else prefix ^ s
  | Byte_level { add_prefix_space; use_regex } ->
      apply_byte_level s ~add_prefix_space ~use_regex
  | Bert
      {
        clean_text = ct;
        handle_chinese_chars = hcc;
        strip_accents = sa;
        lowercase = lc;
      } ->
      let s = if ct then clean_text s else s in
      let s = if hcc then handle_chinese_chars s else s in
      let do_strip = match sa with Some v -> v | None -> lc in
      let s = if do_strip then do_strip_accents s else s in
      if lc then do_lowercase s else s
  | Sequence ns -> List.fold_left (fun s n -> apply n s) s ns

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
      typed_with "Bert"
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
          Ok
            (Strip
               {
                 left = get_bool "strip_left" false;
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
