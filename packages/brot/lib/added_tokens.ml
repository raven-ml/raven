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

(* Patterns bucketed by their first byte, longest first within a bucket, so that
   a position starting no pattern costs one array load. The empty matcher has no
   buckets at all. *)
type matcher = pattern array array

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

let matcher patterns =
  match patterns with
  | [] -> [||]
  | _ ->
      let buckets = Array.make 256 [||] in
      let pending = Array.make 256 [] in
      List.iter
        (fun pattern ->
          let first = Char.code pattern.text.[0] in
          pending.(first) <- pattern :: pending.(first))
        patterns;
      let longest_first a b =
        Int.compare (String.length b.text) (String.length a.text)
      in
      for first = 0 to 255 do
        match pending.(first) with
        | [] -> ()
        | candidates ->
            let bucket = Array.of_list candidates in
            Array.sort longest_first bucket;
            buckets.(first) <- bucket
      done;
      buckets

let matches_at text pos pattern =
  let len = String.length pattern in
  let rec same i =
    i >= len
    || String.unsafe_get text (pos + i) = String.unsafe_get pattern i
       && same (i + 1)
  in
  same 0

let find matcher text ~pos =
  if Array.length matcher = 0 then None
  else begin
    let len = String.length text in
    let at = ref pos in
    let hit = ref None in
    while Option.is_none !hit && !at < len do
      let bucket =
        Array.unsafe_get matcher (Char.code (String.unsafe_get text !at))
      in
      let count = Array.length bucket in
      let index = ref 0 in
      let candidate = ref None in
      while Option.is_none !candidate && !index < count do
        let pattern = Array.unsafe_get bucket !index in
        if
          !at + String.length pattern.text <= len
          && matches_at text !at pattern.text
        then candidate := Some pattern
        else incr index
      done;
      match !candidate with
      | None -> incr at
      | Some { text = matched; token } ->
          let start = !at and stop = !at + String.length matched in
          if
            token.single_word
            && (ends_with_word text start || starts_with_word text stop)
          then at := stop
          else
            let start =
              if token.lstrip then max pos (space_before text start) else start
            in
            let stop = if token.rstrip then space_after text stop else stop in
            hit := Some (start, stop, token.id)
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
