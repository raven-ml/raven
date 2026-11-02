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

(* How the scan seeks the next byte that can stand at the anchor of a pattern:
   a SWAR search for one or two broadcast bytes, eight bytes per load, or the
   bytewise [anchors] test when more than two bytes can stand there. *)
type seek = Seek_one of int64 | Seek_two of int64 * int64 | Seek_bytewise

(* The patterns as a trie, walked once from each candidate start: the longest
   pattern there is the last terminal node the walk passes, whatever the
   number of patterns sharing a prefix. [root] maps a first byte to its node;
   node [n]'s children are [kid_byte] and [kid_node] from [kids.(n)] to
   [kids.(n + 1)], and [term.(n)] is the pattern it ends, or [-1].

   Every match holds the patterns' common prefix, so the scan seeks the
   prefix's last byte, at [anchor] bytes from the start, and passes over text
   in which that byte does not occur at a load and a compare per eight bytes.
   Patterns that share a marker and a bracket, as SentencePiece's [▁<s>] and
   [▁</s>] do once normalized, then cost nothing where the marker alone
   occurs at every word. Without a common prefix the anchor is the first byte,
   which [anchors] flags. The empty matcher has no patterns at all. *)
type matcher = {
  patterns : pattern array;
  root : int array;
  term : int array;
  kids : int array;
  kid_byte : Bytes.t;
  kid_node : int array;
  anchor : int;
  anchors : Bytes.t;
  seek : seek;
}

type t = {
  tokens : token list;
  by_content : (string, int) Hashtbl.t;
  by_id : (int, pattern) Hashtbl.t;
  raw : matcher;
  normalized : matcher;
  match_special : bool;  (* Whether the tokens whose [special] is set match. *)
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

(* A trie under construction: each node numbered as it is made, with its
   children by byte. *)
type draft = {
  serial : int;
  mutable ends : int;
  mutable children : (char * draft) list;
}

type drafts = { mutable made : draft list; mutable count : int }

let draft drafts =
  let node = { serial = drafts.count; ends = -1; children = [] } in
  drafts.made <- node :: drafts.made;
  drafts.count <- drafts.count + 1;
  node

(* The node [text] from byte [i] ends at, below [node]. *)
let rec insert drafts node text i =
  if i = String.length text then node
  else
    let c = text.[i] in
    let child =
      match List.assoc_opt c node.children with
      | Some child -> child
      | None ->
          let child = draft drafts in
          node.children <- (c, child) :: node.children;
          child
    in
    insert drafts child text (i + 1)

(* The length of the prefix all of [patterns] share. *)
let common_prefix patterns =
  let first = patterns.(0).text in
  Array.fold_left
    (fun n p ->
      let k = ref 0 and n = min n (String.length p.text) in
      while !k < n && p.text.[!k] = first.[!k] do
        incr k
      done;
      !k)
    (String.length first) patterns

let empty_matcher =
  {
    patterns = [||];
    root = [||];
    term = [||];
    kids = [||];
    kid_byte = Bytes.empty;
    kid_node = [||];
    anchor = 0;
    anchors = Bytes.empty;
    seek = Seek_bytewise;
  }

let matcher patterns =
  match patterns with
  | [] -> empty_matcher
  | _ ->
      let patterns = Array.of_list patterns in
      let drafts = { made = []; count = 0 } in
      let top = { serial = -1; ends = -1; children = [] } in
      Array.iteri
        (fun index p ->
          let node = insert drafts top p.text 0 in
          if node.ends >= 0 then
            invalid_arg
              (Printf.sprintf
                 "added tokens %S and %S match the same text %S once \
                  normalized"
                 patterns.(node.ends).token.content p.token.content p.text);
          node.ends <- index)
        patterns;
      let n = drafts.count in
      let nodes = Array.of_list (List.rev drafts.made) in
      let root = Array.make 256 (-1) in
      List.iter
        (fun (c, child) -> root.(Char.code c) <- child.serial)
        top.children;
      let term = Array.map (fun node -> node.ends) nodes in
      let kids = Array.make (n + 1) 0 in
      Array.iteri
        (fun i node -> kids.(i + 1) <- kids.(i) + List.length node.children)
        nodes;
      let kid_byte = Bytes.create kids.(n) in
      let kid_node = Array.make kids.(n) 0 in
      Array.iteri
        (fun i node ->
          List.iteri
            (fun k (c, child) ->
              Bytes.set kid_byte (kids.(i) + k) c;
              kid_node.(kids.(i) + k) <- child.serial)
            node.children)
        nodes;
      let anchor = max 0 (common_prefix patterns - 1) in
      let anchors = Bytes.make 256 '\000' in
      Array.iter
        (fun p -> Bytes.set anchors (Char.code p.text.[anchor]) '\001')
        patterns;
      let flagged = ref [] in
      for b = 255 downto 0 do
        if Bytes.get anchors b <> '\000' then flagged := b :: !flagged
      done;
      let seek =
        match !flagged with
        | [ b ] -> Seek_one (broadcast b)
        | [ b0; b1 ] -> Seek_two (broadcast b0, broadcast b1)
        | _ -> Seek_bytewise
      in
      { patterns; root; term; kids; kid_byte; kid_node; anchor; anchors; seek }

let[@inline] is_anchor m c = Bytes.unsafe_get m.anchors (Char.code c) <> '\000'

(* [candidate m text ~pos ~len] is the first position at or after [pos] whose
   byte can stand at the anchor of a pattern of [m], or [len] if there is none.
   The SWAR loops stop on the eight-byte window holding such a byte; the
   bytewise loop pins it down, and covers the trailing window, which keeps
   every load within [text] — the primitive is bounds-checked under the
   bytecode runtime. *)
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
  while !i < len && not (is_anchor m (String.unsafe_get text !i)) do
    incr i
  done;
  !i

(* The child of [node] by byte [c], or [-1]. *)
let child m node c =
  let stop = Array.unsafe_get m.kids (node + 1) in
  let k = ref (Array.unsafe_get m.kids node) in
  while !k < stop && Bytes.unsafe_get m.kid_byte !k <> c do
    incr k
  done;
  if !k < stop then Array.unsafe_get m.kid_node !k else -1

(* The longest pattern of [m] at [start] in [text], as its index, or [-1]. *)
let longest m text start len =
  let first = Char.code (String.unsafe_get text start) in
  let node = ref (Array.unsafe_get m.root first) in
  let best = ref (-1) and i = ref (start + 1) in
  while !node >= 0 do
    let ends = Array.unsafe_get m.term !node in
    if ends >= 0 then best := ends;
    node := if !i < len then child m !node (String.unsafe_get text !i) else -1;
    incr i
  done;
  !best

(* A special token passed over keeps the text it covers, as a [single_word]
   token that does not stand alone does: no other token of [m] is sought inside
   it. *)
let find m ~match_special text ~pos =
  if Array.length m.patterns = 0 then None
  else begin
    let len = String.length text in
    let at = ref pos in
    let hit = ref None in
    let scanning = ref true in
    while !scanning do
      let start = candidate m text ~pos:(!at + m.anchor) ~len - m.anchor in
      if start + m.anchor >= len then scanning := false
      else
        let index = longest m text start len in
        if index < 0 then at := start + 1
        else begin
          let { text = content; token } = Array.unsafe_get m.patterns index in
          let stop = start + String.length content in
          if
            (token.special && not match_special)
            || token.single_word
               && (ends_with_word text start || starts_with_word text stop)
          then at := stop
          else begin
            let start =
              if token.lstrip then max pos (space_before text start) else start
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
    match_special = true;
  }

let without_special t = { t with match_special = false }

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

let find_raw t text ~pos = find t.raw ~match_special:t.match_special text ~pos

let find_normalized t text ~pos =
  find t.normalized ~match_special:t.match_special text ~pos
