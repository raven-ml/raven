(** Pre-tokenizers implementation *)

type t = string -> string list
type t_with_offsets = string -> (string * (int * int)) list

(** ByteLevel alphabet for encoding bytes as visible characters *)
let byte_level_alphabet () =
  let bytes = ref [] in
  (* First, add regular printable ASCII *)
  for i = 33 to 126 do
    (* '!' to '~' *)
    bytes := Char.escaped (Char.chr i) :: !bytes
  done;
  for i = 161 to 172 do
    (* ¡ to ¬ *)
    bytes := Char.escaped (Char.chr i) :: !bytes
  done;
  for i = 174 to 255 do
    (* ® to ÿ *)
    bytes := Char.escaped (Char.chr i) :: !bytes
  done;
  (* For remaining bytes, use Unicode Private Use Area starting at U+0100 *)
  let next_char = ref 256 in
  for _ = 0 to 32 do
    bytes := Printf.sprintf "\\u%04x" !next_char :: !bytes;
    incr next_char
  done;
  for _ = 127 to 160 do
    bytes := Printf.sprintf "\\u%04x" !next_char :: !bytes;
    incr next_char
  done;
  bytes := Printf.sprintf "\\u%04x" 173 :: !bytes;
  (* Soft hyphen *)
  List.rev !bytes

(** Apply ByteLevel encoding to text *)
let byte_level_encode text =
  (* GPT-2 uses a specific byte to unicode mapping *)
  let bs = ref [] in
  let n = ref 0 in

  (* Build the byte mapping *)
  for b = 0 to 255 do
    if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)
    then bs := (b, b) :: !bs
    else (
      bs := (b, 256 + !n) :: !bs;
      incr n)
  done;

  let byte_encoder = List.rev !bs in

  (* Convert text to bytes then map to unicode *)
  let result = Buffer.create (String.length text * 2) in
  String.iter
    (fun c ->
      let byte = Char.code c in
      let mapped = List.assoc byte byte_encoder in
      if mapped < 256 then Buffer.add_char result (Char.chr mapped)
      else
        (* Use Unicode encoding *)
        Buffer.add_utf_8_uchar result (Uchar.of_int mapped))
    text;
  Buffer.contents result

(** Simple regex-like splitting for GPT-2 pattern *)
let split_gpt2_pattern text =
  (* Simplified implementation - proper implementation would use Re library *)
  let tokens = ref [] in
  let current = Buffer.create 16 in
  let i = ref 0 in
  let len = String.length text in

  while !i < len do
    let c = text.[!i] in
    match c with
    | ' ' | '\t' | '\n' | '\r' ->
        (* Handle whitespace *)
        if Buffer.length current > 0 then (
          tokens := Buffer.contents current :: !tokens;
          Buffer.clear current);
        (* Check if leading space for next token *)
        if !i + 1 < len && text.[!i + 1] <> ' ' then Buffer.add_char current ' ';
        incr i
    | '!' | '?' | '.' | ',' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']'
    | '{' | '}' ->
        (* Punctuation - separate token *)
        if Buffer.length current > 0 then (
          tokens := Buffer.contents current :: !tokens;
          Buffer.clear current);
        tokens := String.make 1 c :: !tokens;
        incr i
    | _ ->
        (* Regular character *)
        Buffer.add_char current c;
        incr i
  done;

  if Buffer.length current > 0 then tokens := Buffer.contents current :: !tokens;

  List.rev !tokens

(** Whitespace split *)
let whitespace_split text =
  String.split_on_char ' ' text |> List.filter (fun s -> s <> "")

(** Whitespace tokenizer with pattern \w+|[^\w\s]+ *)
let whitespace text =
  let pieces = ref [] in
  let current = Buffer.create 16 in
  let i = ref 0 in
  let len = String.length text in

  let flush_current () =
    if Buffer.length current > 0 then (
      pieces := Buffer.contents current :: !pieces;
      Buffer.clear current)
  in

  while !i < len do
    let c = text.[!i] in
    if
      (c >= 'a' && c <= 'z')
      || (c >= 'A' && c <= 'Z')
      || (c >= '0' && c <= '9')
      || c = '_'
    then (
      Buffer.add_char current c;
      incr i)
    else if c = ' ' || c = '\t' || c = '\n' || c = '\r' then (
      flush_current ();
      incr i)
    else (
      flush_current ();
      Buffer.add_char current c;
      incr i;
      flush_current ())
  done;
  flush_current ();
  List.rev !pieces

(** ByteLevel pre-tokenizer *)
let byte_level ?(add_prefix_space = true) ?(use_regex = true) () text =
  (* Add prefix space if needed *)
  let text =
    if add_prefix_space && String.length text > 0 && text.[0] <> ' ' then
      " " ^ text
    else text
  in

  (* Apply byte-level encoding *)
  let encoded = byte_level_encode text in

  (* Split using GPT-2 pattern if requested *)
  if use_regex then split_gpt2_pattern encoded else [ encoded ]

(** BERT pre-tokenizer *)
let bert text =
  let pieces = ref [] in
  let current = Buffer.create 16 in
  let i = ref 0 in
  let len = String.length text in

  let flush_current () =
    if Buffer.length current > 0 then (
      pieces := Buffer.contents current :: !pieces;
      Buffer.clear current)
  in

  while !i < len do
    let c = text.[!i] in
    if c = ' ' || c = '\t' || c = '\n' || c = '\r' then (
      flush_current ();
      incr i)
    else if String.contains "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" c then (
      flush_current ();
      pieces := String.make 1 c :: !pieces;
      incr i)
    else (
      Buffer.add_char current c;
      incr i)
  done;
  flush_current ();
  List.rev !pieces

(** Punctuation splitter *)
let punctuation ?(behavior = `Isolated) () text =
  let is_punct c = String.contains "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" c in
  let pieces = ref [] in
  let current = Buffer.create 16 in
  let i = ref 0 in
  let len = String.length text in

  let flush_current () =
    if Buffer.length current > 0 then (
      pieces := Buffer.contents current :: !pieces;
      Buffer.clear current)
  in

  while !i < len do
    let c = text.[!i] in
    if is_punct c then
      match behavior with
      | `Isolated ->
          flush_current ();
          pieces := String.make 1 c :: !pieces
      | `Removed -> flush_current ()
      | `MergedWithPrevious -> Buffer.add_char current c
      | `MergedWithNext ->
          flush_current ();
          Buffer.add_char current c
      | `Contiguous ->
          if !i > 0 && is_punct text.[!i - 1] then Buffer.add_char current c
          else (
            flush_current ();
            Buffer.add_char current c)
    else (
      if
        behavior = `Contiguous
        && Buffer.length current > 0
        && !i > 0
        && is_punct text.[!i - 1]
      then flush_current ();
      Buffer.add_char current c);
    incr i
  done;

  flush_current ();
  List.rev !pieces

(** Split on pattern *)
let split ~pattern ~behavior:_ ?(invert = false) () text =
  if invert then [ text ]
    (* Simplified - proper implementation would invert the pattern *)
  else Str.split (Str.regexp pattern) text

(** Character delimiter split *)
let char_delimiter_split delim text =
  String.split_on_char delim text |> List.filter (fun s -> s <> "")

(** Digits splitter *)
let digits ?(individual_digits = false) () text =
  let pieces = ref [] in
  let current = Buffer.create 16 in
  let i = ref 0 in
  let len = String.length text in
  let in_digits = ref false in

  let flush_current () =
    if Buffer.length current > 0 then (
      pieces := Buffer.contents current :: !pieces;
      Buffer.clear current)
  in

  while !i < len do
    let c = text.[!i] in
    let is_digit = c >= '0' && c <= '9' in

    if individual_digits && is_digit then (
      flush_current ();
      pieces := String.make 1 c :: !pieces;
      incr i)
    else if is_digit <> !in_digits then (
      flush_current ();
      in_digits := is_digit;
      Buffer.add_char current c;
      incr i)
    else (
      Buffer.add_char current c;
      incr i)
  done;
  flush_current ();
  List.rev !pieces

(** Metaspace pre-tokenizer *)
let metaspace ?(replacement = "▁") ?(prepend_scheme = `Always) ?(split = true)
    () text =
  (* Add prefix space if needed *)
  let text =
    match prepend_scheme with
    | `Always when String.length text > 0 && text.[0] <> ' ' -> " " ^ text
    | `First when String.length text > 0 && text.[0] <> ' ' -> " " ^ text
    | _ -> text
  in

  if split then
    String.split_on_char ' ' text
    |> List.filter (fun s -> s <> "")
    |> List.map (fun s -> replacement ^ s)
  else [ String.map (fun c -> if c = ' ' then replacement.[0] else c) text ]

(** Sequence of pre-tokenizers *)
let sequence tokenizers text =
  List.fold_left
    (fun pieces tokenizer -> List.concat_map tokenizer pieces)
    [ text ] tokenizers
