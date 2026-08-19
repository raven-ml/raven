(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type token = {
  content : string;
  id : int;
  special : bool;
  single_word : bool;
  lstrip : bool;
  rstrip : bool;
  normalized : bool;
}

(* The text a token matches: its content, or the normalized form of it. *)
type pattern = { text : string; token : token }

(* How the scan seeks the next byte that can start a pattern: a SWAR search for
   one or two broadcast bytes, eight bytes per load, or the bytewise [starts]
   test when the patterns start with more than two distinct bytes. *)
type seek = Seek_one of int64 | Seek_two of int64 * int64 | Seek_bytewise

(* Patterns bucketed by their first byte, longest first within a bucket.
   [starts] flags the 256 bytes that can start a pattern; [seek] is how the scan
   skips to the next flagged byte, so that text holding none costs a load and a
   compare per eight bytes rather than a bucket probe per byte. The empty
   matcher has no buckets at all. *)
type matcher = { buckets : pattern array array; starts : Bytes.t; seek : seek }

type t = {
  tokens : token list;
  by_content : (string, int) Hashtbl.t;
  by_id : (int, pattern) Hashtbl.t;
  raw : matcher;
  normalized : matcher;
}

(* Character boundaries. Malformed bytes are a character of no class, which
   makes them neither word nor white space, so a match beside one stands. *)

let[@inline] is_continuation c = Char.code c land 0xc0 = 0x80

(* [char_before text i] is the character ending at [i], as {!Char_class.at}
   packs it, and where it starts. *)
let char_before text i =
  if i <= 0 then None
  else begin
    let start = ref (i - 1) in
    while !start > 0 && is_continuation (String.unsafe_get text !start) do
      decr start
    done;
    let c = Char_class.at text !start ~stop:i in
    if !start + Char_class.at_len c = i then Some (c, !start) else None
  end

(* [char_at text i] is the character starting at [i] and where it ends. *)
let char_at text i =
  let stop = String.length text in
  if i >= stop then None
  else
    let c = Char_class.at text i ~stop in
    Some (c, i + Char_class.at_len c)

let ends_with_word text i =
  match char_before text i with
  | Some (c, _) -> Char_class.at_is_word c
  | None -> false

let starts_with_word text i =
  match char_at text i with
  | Some (c, _) -> Char_class.at_is_word c
  | None -> false

let[@inline] is_space c = Char_class.at_category c = Char_class.whitespace

(* [space_before text i] is the start of the white space run ending at [i]. *)
let space_before text i =
  let start = ref i and scanning = ref true in
  while !scanning do
    match char_before text !start with
    | Some (c, at) when is_space c -> start := at
    | _ -> scanning := false
  done;
  !start

(* [space_after text i] is the end of the white space run starting at [i]. *)
let space_after text i =
  let stop = ref i and scanning = ref true in
  while !scanning do
    match char_at text !stop with
    | Some (c, at) when is_space c -> stop := at
    | _ -> scanning := false
  done;
  !stop

(* Matching *)

external word64 : string -> int -> int64 = "%caml_string_get64u"

let ones = 0x0101010101010101L
let highs = 0x8080808080808080L
let lows = 0x7F7F7F7F7F7F7F7FL

(* The high bit of every zero byte of [w], and of no other byte. *)
let[@inline] zero_marks w =
  Int64.logand
    (Int64.lognot (Int64.logor (Int64.add (Int64.logand w lows) lows) w))
    highs

let[@inline] broadcast byte = Int64.mul (Int64.of_int byte) ones

let matcher patterns =
  match patterns with
  | [] -> { buckets = [||]; starts = Bytes.empty; seek = Seek_bytewise }
  | _ ->
      let buckets = Array.make 256 [||] in
      let pending = Array.make 256 [] in
      let starts = Bytes.make 256 '\000' in
      List.iter
        (fun pattern ->
          let first = Char.code pattern.text.[0] in
          Bytes.set starts first '\001';
          pending.(first) <- pattern :: pending.(first))
        patterns;
      let longest_first a b =
        Int.compare (String.length b.text) (String.length a.text)
      in
      let firsts = ref [] in
      for first = 255 downto 0 do
        match pending.(first) with
        | [] -> ()
        | candidates ->
            let bucket = Array.of_list candidates in
            Array.sort longest_first bucket;
            buckets.(first) <- bucket;
            firsts := first :: !firsts
      done;
      let seek =
        match !firsts with
        | [ b ] -> Seek_one (broadcast b)
        | [ b0; b1 ] -> Seek_two (broadcast b0, broadcast b1)
        | _ -> Seek_bytewise
      in
      { buckets; starts; seek }

let[@inline] starts_pattern m c =
  Bytes.unsafe_get m.starts (Char.code c) <> '\000'

(* [candidate m text ~pos ~len] is the first position at or after [pos] whose
   byte can start a pattern of [m], or [len] if there is none. The SWAR loops
   stop on the eight-byte window holding such a byte; the bytewise loop pins it
   down, and covers the trailing window, which keeps every load within [text] —
   the primitive is bounds-checked under the bytecode runtime. *)
let candidate m text ~pos ~len =
  let i = ref pos in
  (match m.seek with
  | Seek_one b ->
      let scanning = ref true in
      while !scanning && !i + 8 <= len do
        if zero_marks (Int64.logxor (word64 text !i) b) = 0L then i := !i + 8
        else scanning := false
      done
  | Seek_two (b0, b1) ->
      let scanning = ref true in
      while !scanning && !i + 8 <= len do
        let w = word64 text !i in
        if
          Int64.logor
            (zero_marks (Int64.logxor w b0))
            (zero_marks (Int64.logxor w b1))
          = 0L
        then i := !i + 8
        else scanning := false
      done
  | Seek_bytewise -> ());
  while !i < len && not (starts_pattern m (String.unsafe_get text !i)) do
    incr i
  done;
  !i

let matches_at text pos pattern =
  let len = String.length pattern in
  let rec same i =
    i >= len
    || String.unsafe_get text (pos + i) = String.unsafe_get pattern i
       && same (i + 1)
  in
  same 0

let find m text ~pos =
  if Array.length m.buckets = 0 then None
  else begin
    let len = String.length text in
    let at = ref pos in
    let hit = ref None in
    let scanning = ref true in
    while !scanning do
      let start = candidate m text ~pos:!at ~len in
      if start >= len then scanning := false
      else begin
        let bucket =
          Array.unsafe_get m.buckets (Char.code (String.unsafe_get text start))
        in
        let count = Array.length bucket in
        let index = ref 0 in
        let matched = ref None in
        while Option.is_none !matched && !index < count do
          let pattern = Array.unsafe_get bucket !index in
          if
            start + String.length pattern.text <= len
            && matches_at text start pattern.text
          then matched := Some pattern
          else incr index
        done;
        match !matched with
        | None -> at := start + 1
        | Some { text = content; token } ->
            let stop = start + String.length content in
            if
              token.single_word
              && (ends_with_word text start || starts_with_word text stop)
            then at := stop
            else begin
              let start =
                if token.lstrip then max pos (space_before text start)
                else start
              in
              let stop = if token.rstrip then space_after text stop else stop in
              hit := Some (start, stop, token.id);
              scanning := false
            end
      end
    done;
    !hit
  end

(* Tables *)

(* Two entries with the same content are one token: it keeps the identifier it
   was first given, and the flags of the last entry, as HuggingFace does. *)
let collapse tokens =
  let latest = Hashtbl.create 16 in
  let order = ref [] in
  List.iter
    (fun token ->
      match Hashtbl.find_opt latest token.content with
      | Some earlier ->
          Hashtbl.replace latest token.content { token with id = earlier.id }
      | None ->
          Hashtbl.replace latest token.content token;
          order := token.content :: !order)
    tokens;
  List.rev_map (Hashtbl.find latest) !order

let make ~normalize tokens =
  let tokens = collapse tokens in
  let by_content = Hashtbl.create 16 in
  let by_id = Hashtbl.create 16 in
  let kept = ref [] and raw = ref [] and normalized = ref [] in
  List.iter
    (fun token ->
      let pattern =
        if token.content = "" then None
        else if token.normalized then
          match normalize token.content with
          | "" -> None
          | text -> Some { text; token }
        else Some { text = token.content; token }
      in
      match pattern with
      | None -> ()
      | Some pattern ->
          Hashtbl.replace by_content token.content token.id;
          if not (Hashtbl.mem by_id token.id) then
            Hashtbl.replace by_id token.id pattern;
          kept := token :: !kept;
          if token.normalized then normalized := pattern :: !normalized
          else raw := pattern :: !raw)
    tokens;
  {
    tokens = List.rev !kept;
    by_content;
    by_id;
    raw = matcher (List.rev !raw);
    normalized = matcher (List.rev !normalized);
  }

let tokens t = t.tokens
let is_empty t = match t.tokens with [] -> true | _ -> false
let token_to_id t content = Hashtbl.find_opt t.by_content content

let id_to_token t id =
  match Hashtbl.find_opt t.by_id id with
  | Some pattern -> Some pattern.text
  | None -> None

let is_special t id =
  match Hashtbl.find_opt t.by_id id with
  | Some pattern -> pattern.token.special
  | None -> false

let find_raw t text ~pos = find t.raw text ~pos
let find_normalized t text ~pos = find t.normalized text ~pos
