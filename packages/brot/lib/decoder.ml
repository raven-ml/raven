(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | BPE of { suffix : string }
  | Byte_level
  | Byte_fallback
  | Word_piece of { prefix : string; cleanup : bool }
  | Metaspace of {
      replacement : string;
      prepend_scheme : Pre_tokenizer.prepend_scheme;
    }
  | CTC of { pad_token : string; word_delimiter_token : string; cleanup : bool }
  | Sequence of t list
  | Replace of { pattern : string; replacement : string }
  | Strip of { content : string; start : int; stop : int }
  | Fuse

(* Errors *)

let strf = Printf.sprintf
let err_replace_missing_pattern = "missing pattern in Replace decoder"

let err_replace_regex_pattern =
  "Replace decoder: a regular expression pattern is not supported, only a \
   literal one"

let err_seq_missing_decoders =
  "invalid Sequence decoder: missing decoders array"

let err_unknown_type typ = strf "unknown decoder type: %s" typ
let err_expected_object = "invalid decoder JSON: expected object"
let err_unknown_scheme s = strf "unknown prepend_scheme '%s'" s

(* Decoding *)

(* Literal string replacement without regex overhead. Returns [s] unchanged when
   [pattern] does not occur—no allocation on the fast path. *)
let replace_all ~pattern ~by s =
  let plen = String.length pattern in
  let slen = String.length s in
  if plen = 0 || plen > slen then s
  else
    let match_at i =
      let rec check j =
        j >= plen
        || String.unsafe_get s (i + j) = String.unsafe_get pattern j
           && check (j + 1)
      in
      check 0
    in
    let rec find_first i =
      if i > slen - plen then -1
      else if match_at i then i
      else find_first (i + 1)
    in
    let pos = find_first 0 in
    if pos < 0 then s
    else
      let buf = Buffer.create slen in
      Buffer.add_substring buf s 0 pos;
      Buffer.add_string buf by;
      let i = ref (pos + plen) in
      while !i <= slen - plen do
        if match_at !i then (
          Buffer.add_string buf by;
          i := !i + plen)
        else (
          Buffer.add_char buf (String.unsafe_get s !i);
          incr i)
      done;
      if !i < slen then Buffer.add_substring buf s !i (slen - !i);
      Buffer.contents buf

(* The suffix marks the end of a word, so it stands for the space that follows
   it—except on the last token, where no space follows. *)
let bpe_piece ~suffix ~last token =
  replace_all ~pattern:suffix ~by:(if last then "" else " ") token

let decode_bpe ~suffix tokens =
  match tokens with
  | [] -> []
  | _ ->
      let last = List.length tokens - 1 in
      List.mapi (fun i token -> bpe_piece ~suffix ~last:(i = last) token) tokens

let replacement_char = "\xef\xbf\xbd"

(* Bytes as text, the way HuggingFace reads them back: each maximal ill-formed
   subpart becomes one replacement character, which is what a decode that fails
   skips over. A lone [\xE9] and a four-byte sequence cut short cost one each,
   while [\xC3\x28] costs one and keeps the ['(']. *)
let utf8_lossy s =
  if String.is_valid_utf_8 s then s
  else begin
    let len = String.length s in
    let buf = Buffer.create (len + 8) in
    let i = ref 0 in
    while !i < len do
      let decode = String.get_utf_8_uchar s !i in
      let step = Uchar.utf_decode_length decode in
      if Uchar.utf_decode_is_valid decode then
        Buffer.add_substring buf s !i step
      else Buffer.add_string buf replacement_char;
      i := !i + step
    done;
    Buffer.contents buf
  end

(* Each token maps to bytes on its own, but the text they spell is read whole,
   so a character split across two tokens decodes and only what the bytes of no
   run of tokens spell is replaced. *)
let decode_byte_level tokens =
  let buf = Buffer.create 128 in
  List.iter
    (fun token -> Buffer.add_string buf (Pre_tokenizer.byte_level_decode token))
    tokens;
  utf8_lossy (Buffer.contents buf)

let hex_digit c =
  match c with
  | '0' .. '9' -> Char.code c - Char.code '0'
  | 'a' .. 'f' -> Char.code c - Char.code 'a' + 10
  | 'A' .. 'F' -> Char.code c - Char.code 'A' + 10
  | _ -> -1

let byte_token_value token =
  if
    String.length token = 6
    && String.starts_with ~prefix:"<0x" token
    && String.ends_with ~suffix:">" token
  then
    let hi = hex_digit (String.unsafe_get token 3) in
    let lo = hex_digit (String.unsafe_get token 4) in
    if hi >= 0 && lo >= 0 then Some (Char.unsafe_chr ((hi * 16) + lo)) else None
  else None

(* A run of byte tokens spells one UTF-8 sequence. A run that does not is
   unrecoverable, so each of its bytes becomes a replacement character.
   [fallback_run bytes acc] pushes the tokens a run of [bytes] decodes to onto
   [acc], last first. *)
let fallback_run bytes acc =
  if bytes = "" then acc
  else if String.is_valid_utf_8 bytes then bytes :: acc
  else begin
    let acc = ref acc in
    for _ = 1 to String.length bytes do
      acc := replacement_char :: !acc
    done;
    !acc
  end

let decode_byte_fallback tokens =
  let run_bytes = Buffer.create 16 in
  let flush acc =
    if Buffer.length run_bytes = 0 then acc
    else begin
      let bytes = Buffer.contents run_bytes in
      Buffer.clear run_bytes;
      fallback_run bytes acc
    end
  in
  let rec loop acc = function
    | [] -> List.rev (flush acc)
    | token :: rest -> (
        match byte_token_value token with
        | Some b ->
            Buffer.add_char run_bytes b;
            loop acc rest
        | None -> loop (token :: flush acc) rest)
  in
  loop [] tokens

let detokenization_cleanups =
  [
    (" .", ".");
    (" ?", "?");
    (" !", "!");
    (" ,", ",");
    (" ' ", "'");
    (" n't", "n't");
    (" 'm", "'m");
    (" do not", " don't");
    (" 's", "'s");
    (" 've", "'ve");
    (" 're", "'re");
  ]

(* Undo the spacing around punctuation and English contractions; the last rule
   rewrites rather than unspaces. It sees one piece at a time, so only a space
   already inside that piece is taken back: a full stop that was a token of its
   own keeps the space that follows it. *)
let cleanup_piece piece =
  List.fold_left
    (fun s (pattern, by) -> replace_all ~pattern ~by s)
    piece detokenization_cleanups

let wordpiece_piece ~prefix ~cleanup ~first token =
  let piece =
    if first then token
    else if String.starts_with ~prefix token then
      let plen = String.length prefix in
      String.sub token plen (String.length token - plen)
    else " " ^ token
  in
  if cleanup then cleanup_piece piece else piece

let decode_wordpiece ~prefix ~cleanup tokens =
  List.mapi
    (fun i token -> wordpiece_piece ~prefix ~cleanup ~first:(i = 0) token)
    tokens

(* The replacement stands for a space, except on the first token when it was
   prepended: there it stands for nothing and every occurrence goes. *)
let metaspace_piece ~replacement ~prepended ~first token =
  replace_all ~pattern:replacement
    ~by:(if prepended && first then "" else " ")
    token

let decode_metaspace ~replacement ~prepend_scheme tokens =
  let prepended = prepend_scheme <> `Never in
  List.mapi
    (fun i token ->
      metaspace_piece ~replacement ~prepended ~first:(i = 0) token)
    tokens

let ctc_piece ~pad_token ~word_delimiter_token ~cleanup token =
  let token = replace_all ~pattern:pad_token ~by:"" token in
  let token =
    if cleanup then
      replace_all ~pattern:word_delimiter_token ~by:" " (cleanup_piece token)
    else token
  in
  if String.equal token "" then None else Some token

let decode_ctc ~pad_token ~word_delimiter_token ~cleanup tokens =
  let rec dedup acc = function
    | [] -> List.rev acc
    | [ x ] -> List.rev (x :: acc)
    | x :: (y :: _ as rest) ->
        if String.equal x y then dedup acc rest else dedup (x :: acc) rest
  in
  dedup [] tokens
  |> List.filter_map (ctc_piece ~pad_token ~word_delimiter_token ~cleanup)

let strip_token ~content ~start ~stop token =
  let clen = String.length content in
  let len = String.length token in
  if clen = 0 then token
  else
    let match_at i =
      let rec check j =
        j >= clen
        || String.unsafe_get token (i + j) = String.unsafe_get content j
           && check (j + 1)
      in
      i + clen <= len && check 0
    in
    let first = ref 0 in
    let stripped = ref 0 in
    while !stripped < start && match_at !first do
      first := !first + clen;
      incr stripped
    done;
    let last = ref len in
    stripped := 0;
    while
      !stripped < stop && !last - clen >= !first && match_at (!last - clen)
    do
      last := !last - clen;
      incr stripped
    done;
    if !first >= !last then "" else String.sub token !first (!last - !first)

let rec decode_chain decoder tokens =
  match decoder with
  | BPE { suffix } -> decode_bpe ~suffix tokens
  | Byte_level -> [ decode_byte_level tokens ]
  | Byte_fallback -> decode_byte_fallback tokens
  | Word_piece { prefix; cleanup } -> decode_wordpiece ~prefix ~cleanup tokens
  | Metaspace { replacement; prepend_scheme } ->
      decode_metaspace ~replacement ~prepend_scheme tokens
  | CTC { pad_token; word_delimiter_token; cleanup } ->
      decode_ctc ~pad_token ~word_delimiter_token ~cleanup tokens
  | Replace { pattern; replacement } ->
      List.map (replace_all ~pattern ~by:replacement) tokens
  | Strip { content; start; stop } ->
      List.map (strip_token ~content ~start ~stop) tokens
  | Fuse -> [ String.concat "" tokens ]
  | Sequence decoders ->
      List.fold_left (fun toks dec -> decode_chain dec toks) tokens decoders

let decode decoder tokens = String.concat "" (decode_chain decoder tokens)

(* Streaming *)

(* A stream feeds tokens through the decoders one at a time and hands on text
   as soon as no token to come can change it. The decoders before the first
   collapsing one rewrite whole tokens, each as a stage that keeps what it
   cannot rewrite yet: [bpe] keeps the last token, which may turn out to be the
   last of all, and [byte_fallback] an open run of byte tokens, whose text
   depends on where the run ends. The collapse turns the tokens into text:
   [byte_level] keeps the bytes of a character not yet complete, while [fuse],
   or the end of a list with no collapse, hands the tokens on as they are.

   What follows a collapse sees one token arriving piece by piece, so each of
   its stages rewrites text and keeps the bytes that a later piece could still
   change: [replace] the start of a match that may complete, [strip] what it
   may yet cut from either end. A decoder that reads a token whole there, a
   [byte_level] or [byte_fallback] after a collapse, keeps all of it until the
   end.

   Every stage hands on what its batch rewrite gives for the tokens seen so
   far, less what a later token could change, so the text of a stream is the
   text [decode] gives for its tokens. *)

type token_stage =
  | Bpe_stage of { suffix : string; held : string option }
  | Fallback_stage of { bytes : string }
  | Wordpiece_stage of { prefix : string; cleanup : bool; first : bool }
  | Metaspace_stage of { replacement : string; prepended : bool; first : bool }
  | Ctc_stage of {
      pad_token : string;
      word_delimiter_token : string;
      cleanup : bool;
      previous : string option;
    }
  | Replace_stage of { pattern : string; replacement : string }
  | Strip_stage of { content : string; start : int; stop : int }

type collapse = Bytes_collapse of string | Fuse_collapse

(* [leading] counts the occurrences of [content] still to cut from the start,
   [0] once the text has moved past them. *)
type text_stage =
  | Replace_text of { pattern : string; by : string; held : string }
  | Strip_text of { content : string; leading : int; stop : int; held : string }
  | Whole_text of { decoder : t; held : string list }

type stream = {
  tokens : token_stage list;
  collapse : collapse;
  text : text_stage list;
}

let token_step stage token =
  match stage with
  | Bpe_stage { suffix; held } ->
      let out =
        match held with
        | None -> []
        | Some held -> [ bpe_piece ~suffix ~last:false held ]
      in
      (Bpe_stage { suffix; held = Some token }, out)
  | Fallback_stage { bytes } -> (
      match byte_token_value token with
      | Some b -> (Fallback_stage { bytes = bytes ^ String.make 1 b }, [])
      | None ->
          ( Fallback_stage { bytes = "" },
            List.rev (token :: fallback_run bytes []) ))
  | Wordpiece_stage ({ prefix; cleanup; first } as w) ->
      ( Wordpiece_stage { w with first = false },
        [ wordpiece_piece ~prefix ~cleanup ~first token ] )
  | Metaspace_stage ({ replacement; prepended; first } as m) ->
      ( Metaspace_stage { m with first = false },
        [ metaspace_piece ~replacement ~prepended ~first token ] )
  | Ctc_stage ({ pad_token; word_delimiter_token; cleanup; previous } as c) ->
      let stage = Ctc_stage { c with previous = Some token } in
      if previous = Some token then (stage, [])
      else
        ( stage,
          Option.to_list
            (ctc_piece ~pad_token ~word_delimiter_token ~cleanup token) )
  | Replace_stage { pattern; replacement } ->
      (stage, [ replace_all ~pattern ~by:replacement token ])
  | Strip_stage { content; start; stop } ->
      (stage, [ strip_token ~content ~start ~stop token ])

let token_finish = function
  | Bpe_stage { suffix; held = Some held } ->
      [ bpe_piece ~suffix ~last:true held ]
  | Fallback_stage { bytes } -> List.rev (fallback_run bytes [])
  | _ -> []

let rec feed_tokens stages tokens =
  match stages with
  | [] -> ([], tokens)
  | stage :: rest ->
      let stage, out =
        List.fold_left
          (fun (stage, out) token ->
            let stage, pieces = token_step stage token in
            (stage, List.rev_append pieces out))
          (stage, []) tokens
      in
      let rest, out = feed_tokens rest (List.rev out) in
      (stage :: rest, out)

let rec finish_tokens = function
  | [] -> []
  | stage :: rest ->
      let rest, out = feed_tokens rest (token_finish stage) in
      out @ finish_tokens rest

(* Where the bytes stop decoding the same whatever follows them: before a last
   sequence that is not a character, which a later byte may complete. *)
let settled bytes =
  let len = String.length bytes in
  let rec scan i =
    if i >= len then len
    else
      let d = String.get_utf_8_uchar bytes i in
      let next = i + Uchar.utf_decode_length d in
      if next >= len && not (Uchar.utf_decode_is_valid d) then i else scan next
  in
  scan 0

let collapse_tokens collapse tokens =
  match collapse with
  | Fuse_collapse -> (collapse, String.concat "" tokens)
  | Bytes_collapse pending ->
      let bytes =
        String.concat ""
          (pending :: List.map Pre_tokenizer.byte_level_decode tokens)
      in
      let cut = settled bytes in
      ( Bytes_collapse (String.sub bytes cut (String.length bytes - cut)),
        utf8_lossy (String.sub bytes 0 cut) )

let collapse_finish = function
  | Fuse_collapse -> ""
  | Bytes_collapse pending -> utf8_lossy pending

(* Whether [text] from [at] to its end is where [pattern] starts. *)
let starts_at text at pattern =
  let rec same i =
    at + i >= String.length text
    || i < String.length pattern
       && text.[at + i] = pattern.[i]
       && same (i + 1)
  in
  String.length text - at < String.length pattern && same 0

let matches_at text at pattern =
  let len = String.length pattern in
  let rec same i = i >= len || (text.[at + i] = pattern.[i] && same (i + 1)) in
  at + len <= String.length text && same 0

(* [replace_all] on as much of [text] as decides it, and the rest: a match
   could still start where what is left is the start of [pattern]. *)
let replace_some ~pattern ~by text ~final =
  let len = String.length text and plen = String.length pattern in
  let buf = Buffer.create len in
  let rec scan i =
    if i >= len then len
    else if matches_at text i pattern then begin
      Buffer.add_string buf by;
      scan (i + plen)
    end
    else if (not final) && starts_at text i pattern then i
    else begin
      Buffer.add_char buf text.[i];
      scan (i + 1)
    end
  in
  let cut = scan 0 in
  (Buffer.contents buf, String.sub text cut (len - cut))

(* [strip_token] on as much of [text] as decides it. The occurrences at the
   start are cut as they arrive. A later end cuts at most [stop] occurrences
   from the text, so what lies further from its end than those is settled; the
   cut is moved back to a character boundary. *)
let strip_some ~content ~leading ~stop text ~final =
  let len = String.length text and clen = String.length content in
  let rec lead at leading =
    if leading > 0 && matches_at text at content then
      lead (at + clen) (leading - 1)
    else if leading > 0 && (not final) && starts_at text at content then
      `Waiting (at, leading)
    else `Past at
  in
  match lead 0 leading with
  | `Waiting (at, leading) -> ("", leading, String.sub text at (len - at))
  | `Past at ->
      let rest = String.sub text at (len - at) in
      if final then (strip_token ~content ~start:0 ~stop rest, 0, "")
      else begin
        let n = String.length rest in
        let cut = ref (max 0 (n - (stop * clen))) in
        while !cut > 0 && !cut < n && Char.code rest.[!cut] land 0xc0 = 0x80 do
          decr cut
        done;
        (String.sub rest 0 !cut, 0, String.sub rest !cut (n - !cut))
      end

let text_step stage text ~final =
  match stage with
  | Replace_text { pattern; by; held } ->
      let out, held = replace_some ~pattern ~by (held ^ text) ~final in
      (Replace_text { pattern; by; held }, out)
  | Strip_text { content; leading; stop; held } ->
      let out, leading, held =
        strip_some ~content ~leading ~stop (held ^ text) ~final
      in
      (Strip_text { content; leading; stop; held }, out)
  | Whole_text { decoder; held } ->
      let held = text :: held in
      if final then
        ( Whole_text { decoder; held = [] },
          decode decoder [ String.concat "" (List.rev held) ] )
      else (Whole_text { decoder; held }, "")

let rec feed_text stages text ~final =
  match stages with
  | [] -> ([], text)
  | stage :: rest ->
      let stage, out = text_step stage text ~final in
      let rest, out = feed_text rest out ~final in
      (stage :: rest, out)

(* What a decoder after a collapse does to the one token it is given, which is
   both the first and the last. *)
let rec text_stages decoder =
  let replace pattern by =
    if pattern = "" then [] else [ Replace_text { pattern; by; held = "" } ]
  in
  let cleanups () =
    List.concat_map (fun (pattern, by) -> replace pattern by)
      detokenization_cleanups
  in
  match decoder with
  | BPE { suffix } -> replace suffix ""
  | Byte_level | Byte_fallback -> [ Whole_text { decoder; held = [] } ]
  | Word_piece { cleanup; _ } -> if cleanup then cleanups () else []
  | Metaspace { replacement; prepend_scheme } ->
      replace replacement (if prepend_scheme <> `Never then "" else " ")
  | CTC { pad_token; word_delimiter_token; cleanup } ->
      replace pad_token ""
      @ if cleanup then cleanups () @ replace word_delimiter_token " " else []
  | Replace { pattern; replacement } -> replace pattern replacement
  | Strip { content; start; stop } ->
      if content = "" then []
      else [ Strip_text { content; leading = start; stop; held = "" } ]
  | Fuse -> []
  | Sequence decoders -> List.concat_map text_stages decoders

let stream decoder =
  let rec build stages = function
    | [] -> { tokens = List.rev stages; collapse = Fuse_collapse; text = [] }
    | decoder :: rest -> (
        let collapse collapse =
          {
            tokens = List.rev stages;
            collapse;
            text = List.concat_map text_stages rest;
          }
        in
        let stage stage = build (stage :: stages) rest in
        match decoder with
        | Byte_level -> collapse (Bytes_collapse "")
        | Fuse -> collapse Fuse_collapse
        | Sequence decoders -> build stages (decoders @ rest)
        | BPE { suffix } -> stage (Bpe_stage { suffix; held = None })
        | Byte_fallback -> stage (Fallback_stage { bytes = "" })
        | Word_piece { prefix; cleanup } ->
            stage (Wordpiece_stage { prefix; cleanup; first = true })
        | Metaspace { replacement; prepend_scheme } ->
            stage
              (Metaspace_stage
                 {
                   replacement;
                   prepended = prepend_scheme <> `Never;
                   first = true;
                 })
        | CTC { pad_token; word_delimiter_token; cleanup } ->
            stage
              (Ctc_stage
                 { pad_token; word_delimiter_token; cleanup; previous = None })
        | Replace { pattern; replacement } ->
            stage (Replace_stage { pattern; replacement })
        | Strip { content; start; stop } ->
            stage (Strip_stage { content; start; stop }))
  in
  build [] [ decoder ]

let step stream token =
  let tokens, out = feed_tokens stream.tokens [ token ] in
  let collapse, text = collapse_tokens stream.collapse out in
  let stages, text = feed_text stream.text text ~final:false in
  (text, { tokens; collapse; text = stages })

let finish stream =
  let out = finish_tokens stream.tokens in
  let collapse, text = collapse_tokens stream.collapse out in
  let _, text =
    feed_text stream.text (text ^ collapse_finish collapse) ~final:true
  in
  text

(* Constructors *)

let bpe ?(suffix = "</w>") () = BPE { suffix }
let byte_level = Byte_level
let byte_fallback = Byte_fallback

let wordpiece ?(prefix = "##") ?(cleanup = true) () =
  Word_piece { prefix; cleanup }

let metaspace ?(replacement = "\xe2\x96\x81") ?(prepend_scheme = `Always) () =
  Metaspace { replacement; prepend_scheme }

let ctc ?(pad_token = "<pad>") ?(word_delimiter_token = "|") ?(cleanup = true)
    () =
  CTC { pad_token; word_delimiter_token; cleanup }

let sequence decoders = Sequence decoders
let replace ~pattern ~by () = Replace { pattern; replacement = by }

let strip ?(content = " ") ?(start = 0) ?(stop = 0) () =
  Strip { content; start; stop }

let fuse = Fuse

(* Formatting *)

let scheme_to_string = function
  | `First -> "first"
  | `Never -> "never"
  | `Always -> "always"

let rec pp ppf = function
  | BPE { suffix } -> Format.fprintf ppf "bpe ~suffix:%S" suffix
  | Byte_level -> Format.fprintf ppf "byte_level"
  | Byte_fallback -> Format.fprintf ppf "byte_fallback"
  | Word_piece { prefix; cleanup } ->
      Format.fprintf ppf "wordpiece ~prefix:%S ~cleanup:%b" prefix cleanup
  | Metaspace { replacement; prepend_scheme } ->
      Format.fprintf ppf "metaspace ~replacement:%S ~prepend_scheme:%s"
        replacement
        (scheme_to_string prepend_scheme)
  | CTC { pad_token; word_delimiter_token; cleanup } ->
      Format.fprintf ppf
        "ctc ~pad_token:%S ~word_delimiter_token:%S ~cleanup:%b" pad_token
        word_delimiter_token cleanup
  | Replace { pattern; replacement } ->
      Format.fprintf ppf "replace ~pattern:%S ~by:%S" pattern replacement
  | Strip { content; start; stop } ->
      Format.fprintf ppf "strip ~content:%S ~start:%d ~stop:%d" content start
        stop
  | Fuse -> Format.fprintf ppf "fuse"
  | Sequence decoders ->
      Format.fprintf ppf "@[<hv 2>sequence [%a]@]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ")
           pp)
        decoders

(* Serialization *)

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let rec to_json = function
  | BPE { suffix } ->
      json_obj
        [
          ("type", Jsont.Json.string "BPEDecoder");
          ("suffix", Jsont.Json.string suffix);
        ]
  (* The byte-level decoder reads neither member, but HuggingFace requires both
     to be present. *)
  | Byte_level ->
      json_obj
        [
          ("type", Jsont.Json.string "ByteLevel");
          ("add_prefix_space", Jsont.Json.bool true);
          ("trim_offsets", Jsont.Json.bool true);
        ]
  | Byte_fallback -> json_obj [ ("type", Jsont.Json.string "ByteFallback") ]
  | Word_piece { prefix; cleanup } ->
      json_obj
        [
          ("type", Jsont.Json.string "WordPiece");
          ("prefix", Jsont.Json.string prefix);
          ("cleanup", Jsont.Json.bool cleanup);
        ]
  | Metaspace { replacement; prepend_scheme } ->
      json_obj
        [
          ("type", Jsont.Json.string "Metaspace");
          ("replacement", Jsont.Json.string replacement);
          ("prepend_scheme", Jsont.Json.string (scheme_to_string prepend_scheme));
        ]
  | CTC { pad_token; word_delimiter_token; cleanup } ->
      json_obj
        [
          ("type", Jsont.Json.string "CTC");
          ("pad_token", Jsont.Json.string pad_token);
          ("word_delimiter_token", Jsont.Json.string word_delimiter_token);
          ("cleanup", Jsont.Json.bool cleanup);
        ]
  | Replace { pattern; replacement } ->
      json_obj
        [
          ("type", Jsont.Json.string "Replace");
          ("pattern", json_obj [ ("String", Jsont.Json.string pattern) ]);
          ("content", Jsont.Json.string replacement);
        ]
  | Strip { content; start; stop } ->
      json_obj
        [
          ("type", Jsont.Json.string "Strip");
          ("content", Jsont.Json.string content);
          ("start", Jsont.Json.int start);
          ("stop", Jsont.Json.int stop);
        ]
  | Fuse -> json_obj [ ("type", Jsont.Json.string "Fuse") ]
  | Sequence decoders ->
      json_obj
        [
          ("type", Jsont.Json.string "Sequence");
          ("decoders", Jsont.Json.list (List.map to_json decoders));
        ]

let find_field fields name = Option.map snd (Jsont.Json.find_mem name fields)

let string_field fields name ~default =
  match find_field fields name with
  | Some (Jsont.String (s, _)) -> s
  | _ -> default

let bool_field fields name ~default =
  match find_field fields name with
  | Some (Jsont.Bool (b, _)) -> b
  | _ -> default

let int_field fields name ~default =
  match find_field fields name with
  | Some (Jsont.Number (f, _)) -> int_of_float f
  | _ -> default

let scheme_of_string = function
  | "first" -> Ok `First
  | "never" -> Ok `Never
  | "always" -> Ok `Always
  | s -> Error (err_unknown_scheme s)

let rec of_json = function
  | Jsont.Object (fields, _) -> (
      let ( let* ) = Result.bind in
      match find_field fields "type" with
      | Some (Jsont.String ("BPEDecoder", _)) ->
          Ok (BPE { suffix = string_field fields "suffix" ~default:"</w>" })
      | Some (Jsont.String ("ByteLevel", _)) -> Ok Byte_level
      | Some (Jsont.String ("ByteFallback", _)) -> Ok Byte_fallback
      | Some (Jsont.String ("WordPiece", _)) ->
          Ok
            (Word_piece
               {
                 prefix = string_field fields "prefix" ~default:"##";
                 cleanup = bool_field fields "cleanup" ~default:true;
               })
      | Some (Jsont.String ("Metaspace", _)) ->
          let* prepend_scheme =
            scheme_of_string
              (string_field fields "prepend_scheme" ~default:"always")
          in
          Ok
            (Metaspace
               {
                 replacement =
                   string_field fields "replacement" ~default:"\xe2\x96\x81";
                 prepend_scheme;
               })
      | Some (Jsont.String ("CTC", _)) ->
          Ok
            (CTC
               {
                 pad_token = string_field fields "pad_token" ~default:"<pad>";
                 word_delimiter_token =
                   string_field fields "word_delimiter_token" ~default:"|";
                 cleanup = bool_field fields "cleanup" ~default:true;
               })
      | Some (Jsont.String ("Replace", _)) ->
          let* pattern =
            match find_field fields "pattern" with
            | Some (Jsont.String (s, _)) -> Ok s
            | Some (Jsont.Object (pattern_fields, _)) -> (
                match Jsont.Json.find_mem "String" pattern_fields with
                | Some (_, Jsont.String (p, _)) -> Ok p
                | _ ->
                    if
                      Option.is_some
                        (Jsont.Json.find_mem "Regex" pattern_fields)
                    then Error err_replace_regex_pattern
                    else Error err_replace_missing_pattern)
            | _ -> Error err_replace_missing_pattern
          in
          Ok
            (Replace
               {
                 pattern;
                 replacement = string_field fields "content" ~default:"";
               })
      | Some (Jsont.String ("Strip", _)) ->
          Ok
            (Strip
               {
                 content = string_field fields "content" ~default:" ";
                 start = int_field fields "start" ~default:0;
                 stop = int_field fields "stop" ~default:0;
               })
      | Some (Jsont.String ("Fuse", _)) -> Ok Fuse
      | Some (Jsont.String ("Sequence", _)) -> (
          match find_field fields "decoders" with
          | Some (Jsont.Array (decs, _)) ->
              let* decoders =
                List.fold_left
                  (fun acc j ->
                    let* acc = acc in
                    let* d = of_json j in
                    Ok (d :: acc))
                  (Ok []) decs
              in
              Ok (Sequence (List.rev decoders))
          | _ -> Error err_seq_missing_decoders)
      | Some (Jsont.String (typ, _)) -> Error (err_unknown_type typ)
      | _ -> Error "missing or invalid decoder type field")
  | _ -> Error err_expected_object
