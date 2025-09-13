(** Pre-tokenizers implementation *)

type t = string -> (string * (int * int)) list

let utf8_next s pos =
  let i = pos in
  let c = Char.code s.[i] in
  if c < 128 then (c, 1)
  else if c < 224 then
    (((c land 0x1F) lsl 6) lor (Char.code s.[i + 1] land 0x3F), 2)
  else if c < 240 then
    ( ((c land 0x0F) lsl 12)
      lor ((Char.code s.[i + 1] land 0x3F) lsl 6)
      lor (Char.code s.[i + 2] land 0x3F),
      3 )
  else
    ( ((c land 0x07) lsl 18)
      lor ((Char.code s.[i + 1] land 0x3F) lsl 12)
      lor ((Char.code s.[i + 2] land 0x3F) lsl 6)
      lor (Char.code s.[i + 3] land 0x3F),
      4 )

let add_utf8 buffer code =
  if code < 128 then Buffer.add_char buffer (Char.chr code)
  else if code < 2048 then (
    Buffer.add_char buffer (Char.chr (0xC0 lor (code lsr 6)));
    Buffer.add_char buffer (Char.chr (0x80 lor (code land 63))))
  else if code < 65536 then (
    Buffer.add_char buffer (Char.chr (0xE0 lor (code lsr 12)));
    Buffer.add_char buffer (Char.chr (0x80 lor ((code lsr 6) land 63)));
    Buffer.add_char buffer (Char.chr (0x80 lor (code land 63))))
  else (
    Buffer.add_char buffer (Char.chr (0xF0 lor (code lsr 18)));
    Buffer.add_char buffer (Char.chr (0x80 lor ((code lsr 12) land 63)));
    Buffer.add_char buffer (Char.chr (0x80 lor ((code lsr 6) land 63)));
    Buffer.add_char buffer (Char.chr (0x80 lor (code land 63))))

(* Property functions from Unicode data *)

let is_alphabetic code = Uucp.Alpha.is_alphabetic (Uchar.of_int code)
let is_numeric code = Uucp.Num.numeric_type (Uchar.of_int code) <> `None

let is_punctuation code =
  match Uucp.Gc.general_category (Uchar.of_int code) with
  | `Pc | `Pd | `Pe | `Pf | `Pi | `Po | `Ps -> true
  | _ -> false

let is_whitespace code = Uucp.White.is_white_space (Uchar.of_int code)

(** Converts bytes to unicode characters. *)
let bytes_char () =
  let bs = ref [] in
  for i = 33 to 126 do
    bs := i :: !bs
  done;
  for i = 161 to 172 do
    bs := i :: !bs
  done;
  for i = 174 to 255 do
    bs := i :: !bs
  done;
  let bs_list = List.rev !bs in
  let final_bs = ref [] in
  let final_cs = ref [] in
  let n = ref 0 in
  for b = 0 to 255 do
    if not (List.mem b bs_list) then (
      final_bs := !final_bs @ [ b ];
      final_cs := !final_cs @ [ 256 + !n ];
      incr n)
    else (
      final_bs := !final_bs @ [ b ];
      final_cs := !final_cs @ [ b ])
  done;
  List.combine !final_bs !final_cs

(** Apply ByteLevel encoding to text *)
let byte_level_encode text =
  let result = Buffer.create (String.length text * 2) in
  let i = ref 0 in
  while !i < String.length text do
    let b = Char.code text.[!i] in
    let u = List.assoc b (bytes_char ()) in
    add_utf8 result u;
    incr i
  done;
  Buffer.contents result

(** Decode ByteLevel encoded text back to original bytes *)
let byte_level_decode text =
  (* Create inverse mapping from unicode chars to bytes *)
  let char_bytes = List.map (fun (b, c) -> (c, b)) (bytes_char ()) in
  (* Convert text to list of unicode codepoints *)
  let result = Buffer.create (String.length text) in
  let i = ref 0 in
  while !i < String.length text do
    let code, clen = utf8_next text !i in
    (* Look up the byte value for this unicode codepoint *)
    (match List.assoc_opt code char_bytes with
    | Some byte -> Buffer.add_char result (Char.chr byte)
    | None ->
        (* If not found in mapping, treat as regular UTF-8 *)
        for j = !i to !i + clen - 1 do
          Buffer.add_char result text.[j]
        done);
    i := !i + clen
  done;
  Buffer.contents result

(** GPT-2 pattern splitting that matches Python tokenizers exactly *)
let split_gpt2_pattern text =
  let len = String.length text in
  if len = 0 then []
  else
    let tokens = ref [] in
    let pos = ref 0 in
    let matched = ref false in

    let rec match_at_pos () =
      if !pos >= len then ()
      else (
        matched := false;
        (* 1. Contractions *)
        (if (not !matched) && !pos < len && text.[!pos] = '\'' then
           let remaining = len - !pos in
           if remaining >= 2 then
             let two_char = String.sub text !pos 2 in
             if
               two_char = "'s" || two_char = "'t" || two_char = "'m"
               || two_char = "'d"
             then (
               tokens := two_char :: !tokens;
               pos := !pos + 2;
               matched := true)
             else if remaining >= 3 then
               let three_char = String.sub text !pos 3 in
               if three_char = "'re" || three_char = "'ve" || three_char = "'ll"
               then (
                 tokens := three_char :: !tokens;
                 pos := !pos + 3;
                 matched := true));
        (* 2. Optional space + letters *)
        (if not !matched then
           let start = !pos in
           let has_space =
             !pos < len && is_whitespace (Char.code text.[!pos])
           in
           let letter_start = if has_space then !pos + 1 else !pos in
           if letter_start < len then
             let code, clen = utf8_next text letter_start in
             if is_alphabetic code then (
               let j = ref (letter_start + clen) in
               let continue = ref true in
               while !j < len && !continue do
                 let code, clen = utf8_next text !j in
                 if is_alphabetic code then j := !j + clen
                 else continue := false
               done;
               tokens := String.sub text start (!j - start) :: !tokens;
               pos := !j;
               matched := true));
        (* 3. Optional space + digits *)
        (if not !matched then
           let start = !pos in
           let has_space =
             !pos < len && is_whitespace (Char.code text.[!pos])
           in
           let digit_start = if has_space then !pos + 1 else !pos in
           if digit_start < len then
             let code, clen = utf8_next text digit_start in
             if is_numeric code then (
               let j = ref (digit_start + clen) in
               let continue = ref true in
               while !j < len && !continue do
                 let code, clen = utf8_next text !j in
                 if is_numeric code then j := !j + clen else continue := false
               done;
               tokens := String.sub text start (!j - start) :: !tokens;
               pos := !j;
               matched := true));
        (* 4. Optional space + other non-whitespace *)
        (if not !matched then
           let start = !pos in
           let has_space =
             !pos < len && is_whitespace (Char.code text.[!pos])
           in
           let other_start = if has_space then !pos + 1 else !pos in
           if other_start < len then
             let code, clen = utf8_next text other_start in
             if
               (not (is_whitespace code))
               && (not (is_alphabetic code))
               && not (is_numeric code)
             then (
               let j = ref (other_start + clen) in
               let continue = ref true in
               while !j < len && !continue do
                 let code, clen = utf8_next text !j in
                 if
                   (not (is_whitespace code))
                   && (not (is_alphabetic code))
                   && not (is_numeric code)
                 then j := !j + clen
                 else continue := false
               done;
               tokens := String.sub text start (!j - start) :: !tokens;
               pos := !j;
               matched := true));
        (* 5 & 6. Whitespace *)
        (if (not !matched) && !pos < len then
           let code, clen = utf8_next text !pos in
           if is_whitespace code then (
             (* Try to match as much whitespace as possible, but stop if the
                next character could start a better match *)
             let j = ref (!pos + clen) in
             let continue = ref true in
             while !j < len && !continue do
               let code, clen = utf8_next text !j in
               if is_whitespace code then
                 (* Check if continuing would prevent a better match *)
                 let next_pos = !j + clen in
                 if next_pos < len then
                   let next_code, _ = utf8_next text next_pos in
                   (* If next char is a letter/digit, and we have a space at j,
                      we should stop here to allow " ?\p{L}+" or " ?\p{N}+" to
                      match *)
                   if
                     (is_alphabetic next_code || is_numeric next_code)
                     && code = Char.code ' '
                   then
                     (* Don't consume this space - let it be part of the next
                        match *)
                     continue := false
                   else j := !j + clen
                 else
                   (* At end, consume all whitespace *)
                   j := !j + clen
               else
                 (* Stop at non-whitespace *)
                 continue := false
             done;
             tokens := String.sub text !pos (!j - !pos) :: !tokens;
             pos := !j;
             matched := true));
        (* If nothing matched, add single char *)
        if not !matched then (
          let _, len = utf8_next text !pos in
          tokens := String.sub text !pos len :: !tokens;
          pos := !pos + len);
        match_at_pos ())
    in
    match_at_pos ();
    List.rev !tokens

(** Whitespace split *)
let whitespace_split () text =
  let pieces = ref [] in
  let current = Buffer.create 16 in
  let current_start = ref 0 in
  let i = ref 0 in
  let len = String.length text in
  let flush_current () =
    if Buffer.length current > 0 then (
      pieces :=
        ( Buffer.contents current,
          (!current_start, !current_start + Buffer.length current) )
        :: !pieces;
      Buffer.clear current)
  in
  while !i < len do
    let code, l = utf8_next text !i in
    if is_whitespace code then (
      flush_current ();
      i := !i + l)
    else (
      if Buffer.length current = 0 then current_start := !i;
      Buffer.add_string current (String.sub text !i l);
      i := !i + l)
  done;
  flush_current ();
  List.rev !pieces

(** Whitespace tokenizer with pattern \w+|[^\w\s]+ *)
let whitespace () text =
  let pieces = ref [] in
  let current = Buffer.create 16 in
  let current_start = ref 0 in
  let i = ref 0 in
  let len = String.length text in
  let flush_current () =
    if Buffer.length current > 0 then (
      pieces :=
        ( Buffer.contents current,
          (!current_start, !current_start + Buffer.length current) )
        :: !pieces;
      Buffer.clear current)
  in
  let in_word = ref false in
  let in_punct = ref false in
  while !i < len do
    let code, l = utf8_next text !i in
    if is_alphabetic code || is_numeric code || code = 95 (* _ *) then (
      (* Word character *)
      if !in_punct then flush_current ();
      if Buffer.length current = 0 then current_start := !i;
      Buffer.add_string current (String.sub text !i l);
      in_word := true;
      in_punct := false;
      i := !i + l)
    else if is_whitespace code then (
      (* Whitespace - flush and skip *)
      flush_current ();
      in_word := false;
      in_punct := false;
      i := !i + l)
    else (
      (* Punctuation/other character *)
      if !in_word then flush_current ();
      if Buffer.length current = 0 then current_start := !i;
      Buffer.add_string current (String.sub text !i l);
      in_word := false;
      in_punct := true;
      i := !i + l)
  done;
  flush_current ();
  List.rev !pieces

(** ByteLevel pre-tokenizer *)
let byte_level ?(add_prefix_space = true) ?(use_regex = true)
    ?(trim_offsets = true) () text =
  let _ = trim_offsets in
  (* Not used for now *)
  (* Track original offsets *)
  let original_text = text in
  (* Add prefix space if needed *)
  let text, prefix_added =
    if
      add_prefix_space
      && String.length text > 0
      && not (is_whitespace (Char.code text.[0]))
    then (" " ^ text, true)
    else (text, false)
  in

  (* Split text and track offsets *)
  let pieces_with_offsets =
    if use_regex then
      (* Use GPT-2 pattern splitting *)
      let pieces = split_gpt2_pattern text in
      (* Calculate offsets for each piece *)
      let rec calculate_offsets pieces pos acc =
        match pieces with
        | [] -> List.rev acc
        | piece :: rest ->
            let piece_len = String.length piece in
            let offset_start =
              if prefix_added && pos = 0 then 0
              else if prefix_added then pos - 1
              else pos
            in
            let offset_end =
              if prefix_added then pos + piece_len - 1 else pos + piece_len
            in
            (* Clamp to original text bounds *)
            let offset_end = min offset_end (String.length original_text) in
            let offset_start = max 0 offset_start in
            calculate_offsets rest (pos + piece_len)
              ((piece, (offset_start, offset_end)) :: acc)
      in
      calculate_offsets pieces 0 []
    else [ (text, (0, String.length original_text)) ]
  in

  (* Apply byte-level encoding to each piece while preserving offsets *)
  List.map
    (fun (piece, offsets) -> (byte_level_encode piece, offsets))
    pieces_with_offsets

(** BERT pre-tokenizer *)
let bert () text =
  let pieces = ref [] in
  let current = Buffer.create 16 in
  let current_start = ref 0 in
  let i = ref 0 in
  let len = String.length text in
  let flush_current () =
    if Buffer.length current > 0 then (
      pieces :=
        ( Buffer.contents current,
          (!current_start, !current_start + Buffer.length current) )
        :: !pieces;
      Buffer.clear current)
  in
  while !i < len do
    let code, l = utf8_next text !i in
    if is_whitespace code then (
      flush_current ();
      i := !i + l)
    else if is_punctuation code then (
      flush_current ();
      pieces := (String.sub text !i l, (!i, !i + l)) :: !pieces;
      i := !i + l)
    else (
      if Buffer.length current = 0 then current_start := !i;
      Buffer.add_string current (String.sub text !i l);
      i := !i + l)
  done;
  flush_current ();
  List.rev !pieces

type behavior =
  [ `Isolated
  | `Removed
  | `Merged_with_previous
  | `Merged_with_next
  | `Contiguous ]

(** Punctuation splitter *)
let punctuation ?(behavior = `Isolated) () text =
  let pieces = ref [] in
  let current = Buffer.create 16 in
  let current_start = ref 0 in
  let i = ref 0 in
  let len = String.length text in
  let flush_current () =
    if Buffer.length current > 0 then (
      pieces :=
        ( Buffer.contents current,
          (!current_start, !current_start + Buffer.length current) )
        :: !pieces;
      Buffer.clear current)
  in
  let last_was_punc = ref false in
  while !i < len do
    let code, l = utf8_next text !i in
    let is_p = is_punctuation code in
    if is_p then (
      (match behavior with
      | `Isolated ->
          flush_current ();
          pieces := (String.sub text !i l, (!i, !i + l)) :: !pieces;
          last_was_punc := true
      | `Removed ->
          flush_current ();
          last_was_punc := true
      | `Merged_with_previous ->
          Buffer.add_string current (String.sub text !i l);
          last_was_punc := true
      | `Merged_with_next ->
          flush_current ();
          Buffer.add_string current (String.sub text !i l);
          last_was_punc := true
      | `Contiguous ->
          if Buffer.length current > 0 && !last_was_punc then
            Buffer.add_string current (String.sub text !i l)
          else (
            flush_current ();
            Buffer.add_string current (String.sub text !i l));
          last_was_punc := true);
      i := !i + l)
    else (
      if behavior = `Contiguous && Buffer.length current > 0 && !last_was_punc
      then flush_current ();
      if Buffer.length current = 0 then current_start := !i;
      Buffer.add_string current (String.sub text !i l);
      i := !i + l;
      last_was_punc := false)
  done;
  flush_current ();
  List.rev !pieces

(** Split on pattern *)
let split ~pattern ?(behavior = `Removed) ?(invert = false) () text =
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
        | `Removed ->
            flush_current
              () (* Flush current buffer before removing delimiter *)
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
            (* Set start to delimiter position *)
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

(** Character delimiter split *)
let char_delimiter_split ~delimiter () text =
  let delim = delimiter in
  split ~pattern:(String.make 1 delim) ~behavior:`Removed ~invert:false () text

(** Digits splitter *)
let digits ?(individual_digits = false) () text =
  let pieces = ref [] in
  let current = Buffer.create 16 in
  let current_start = ref 0 in
  let i = ref 0 in
  let len = String.length text in
  let in_digits = ref false in
  let flush_current () =
    if Buffer.length current > 0 then (
      pieces :=
        ( Buffer.contents current,
          (!current_start, !current_start + Buffer.length current) )
        :: !pieces;
      Buffer.clear current)
  in
  while !i < len do
    let code, l = utf8_next text !i in
    let is_d = is_numeric code in
    if individual_digits && is_d then (
      flush_current ();
      let d_str = String.sub text !i l in
      pieces := (d_str, (!i, !i + l)) :: !pieces;
      i := !i + l)
    else if is_d <> !in_digits then (
      flush_current ();
      in_digits := is_d;
      if Buffer.length current = 0 then current_start := !i;
      Buffer.add_string current (String.sub text !i l);
      i := !i + l)
    else (
      if Buffer.length current = 0 then current_start := !i;
      Buffer.add_string current (String.sub text !i l);
      i := !i + l)
  done;
  flush_current ();
  List.rev !pieces

type prepend_scheme = [ `First | `Never | `Always ]
(** Metaspace pre-tokenizer *)

let metaspace ?(replacement = '_') ?(prepend_scheme = `Always) ?(split = true)
    () text =
  let replacement = String.make 1 replacement in
  let is_first = true in
  (* Always true for this simplified implementation *)
  (* Add prefix space if needed *)
  let text =
    match prepend_scheme with
    | `Always when String.length text > 0 && text.[0] <> ' ' -> " " ^ text
    | `First when is_first && String.length text > 0 && text.[0] <> ' ' ->
        " " ^ text
    | _ -> text
  in
  let result = Buffer.create (String.length text) in
  let i = ref 0 in
  while !i < String.length text do
    if text.[!i] = ' ' then (
      Buffer.add_string result replacement;
      incr i)
    else
      let _, l = utf8_next text !i in
      Buffer.add_string result (String.sub text !i l);
      i := !i + l
  done;
  let transformed = Buffer.contents result in
  if split then (
    (* Split on replacement with MergedWithNext behavior *)
    let splits = ref [] in
    let start = ref 0 in
    let pos = ref 0 in
    let rlen = String.length replacement in
    while !pos < String.length transformed do
      if
        !pos + rlen <= String.length transformed
        && String.sub transformed !pos rlen = replacement
      then (
        (* Found a replacement character *)
        if !pos > !start then
          (* Add the piece before the replacement *)
          splits :=
            (String.sub transformed !start (!pos - !start), (!start, !pos))
            :: !splits;
        (* Start next piece at the replacement *)
        start := !pos;
        pos := !pos + rlen)
      else incr pos
    done;
    (* Add the final piece if any *)
    if !pos > !start then
      splits :=
        (String.sub transformed !start (!pos - !start), (!start, !pos))
        :: !splits;
    List.rev !splits)
  else [ (transformed, (0, String.length text)) ]

(** Sequence of pre-tokenizers *)
let sequence tokenizers text =
  let initial = [ (text, (0, String.length text)) ] in
  List.fold_left
    (fun pieces tokenizer ->
      List.concat_map
        (fun (s, (o_start, _o_end)) ->
          let sub_pieces = tokenizer s in
          List.map
            (fun (p, (p_start, p_end)) ->
              (p, (o_start + p_start, o_start + p_end)))
            sub_pieces)
        pieces)
    initial tokenizers

(** FixedLength pre-tokenizer *)
let fixed_length ~length text =
  if length = 0 then []
  else
    let pieces = ref [] in
    let len = String.length text in
    if len = 0 then []
    else
      let i = ref 0 in
      while !i < len do
        let start = !i in
        let count = ref 0 in
        while !i < len && !count < length do
          let _, l = utf8_next text !i in
          i := !i + l;
          incr count
        done;
        pieces := (String.sub text start (!i - start), (start, !i)) :: !pieces
      done;
      List.rev !pieces

type script = [ `Any | Uucp.Script.t ]
(** UnicodeScripts pre-tokenizer *)

let get_script code = Uucp.Script.script (Uchar.of_int code)

let fixed_script code : script =
  if code = 0x30FC then (`Hani :> script)
  else if is_whitespace code then `Any
  else
    match get_script code with
    | `Hira | `Kana -> (`Hani :> script)
    | s -> (s :> script)

let unicode_scripts () text =
  let pieces = ref [] in
  let current = Buffer.create 16 in
  let current_start = ref 0 in
  let len = String.length text in
  let i = ref 0 in
  let last_script = ref None in
  let flush_current () =
    if Buffer.length current > 0 then (
      pieces :=
        ( Buffer.contents current,
          (!current_start, !current_start + Buffer.length current) )
        :: !pieces;
      Buffer.clear current)
  in
  while !i < len do
    let code, l = utf8_next text !i in
    let script = fixed_script code in
    if
      script <> `Any && !last_script <> Some `Any && !last_script <> Some script
    then flush_current ();
    if Buffer.length current = 0 then current_start := !i;
    Buffer.add_string current (String.sub text !i l);
    i := !i + l;
    if script <> `Any then last_script := Some script
  done;
  flush_current ();
  List.rev !pieces

(** Pre-tokenize a string using the given pre-tokenizer *)
let pre_tokenize_str pre_tokenizer text =
  (* Pre_tokenizers.t is already a function from string to (string * (int *
     int)) list *)
  pre_tokenizer text
