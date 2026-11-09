(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Reader = Bytesrw.Bytes.Reader
module Slice = Bytesrw.Bytes.Slice

exception
  Error of {
    line : int;
    column : int;
    byte : int;
    text : string option;
    msg : string;
  }

type framing = Lines | Array

let batch_bytes = 1 lsl 20

(* What an [Array]'s next token is. *)
type phase = Start | First | Element | After | Closed

type t = {
  reader : Reader.t;
  mutable framing : framing;
  mutable buf : Bytes.t;
  mutable len : int;
  mutable eod : bool;
  mutable base : int; (* The input's bytes before [buf]. *)
  mutable start : int; (* The batch's first byte. *)
  mutable stop : int; (* The byte after the batch. *)
  mutable line : int; (* The line of [start], [0] before the first batch. *)
  mutable line_at : int; (* The input position of that line's first byte. *)
  (* The first pass. *)
  mutable scanned : int;
  mutable depth : int;
  mutable in_string : bool;
  mutable escaped : bool;
  mutable blank : bool; (* The current line has only whitespace so far. *)
  mutable found : int; (* The records [frame] has passed. *)
  (* The second pass. *)
  mutable i : int;
  mutable limit : int; (* The end of the current record. *)
  mutable phase : phase;
  mutable text_pos : int;
  mutable text_len : int;
  mutable name_pos : int; (* Where the last member's name starts. *)
  mutable stack : Bytes.t; (* [skip_value]'s open containers. *)
  sampled : bool; (* The input ends after any whole record. *)
}

let make ?(sampled = false) framing reader =
  {
    reader;
    framing;
    buf = Bytes.empty;
    len = 0;
    eod = false;
    base = 0;
    start = 0;
    stop = 0;
    line = 0;
    line_at = 0;
    scanned = 0;
    depth = 0;
    in_string = false;
    escaped = false;
    blank = true;
    found = 0;
    i = 0;
    limit = 0;
    phase = Start;
    text_pos = 0;
    text_len = 0;
    name_pos = 0;
    stack = Bytes.make 64 '\000';
    sampled;
  }

let refill s =
  let slice = Reader.read s.reader in
  if Slice.is_eod slice then s.eod <- true
  else begin
    let n = Slice.length slice in
    if s.len + n > Bytes.length s.buf then begin
      let buf = Bytes.create (max (s.len + n) (2 * Bytes.length s.buf)) in
      Bytes.blit s.buf 0 buf 0 s.len;
      s.buf <- buf
    end;
    Bytes.blit (Slice.bytes slice) (Slice.first slice) s.buf s.len n;
    s.len <- s.len + n
  end

(* The batch's input goes to the front of [buf], so that [buf] holds at most one
   batch and the slices read past it. *)
let compact s =
  let k = s.start in
  if k > 0 then begin
    Bytes.blit s.buf k s.buf 0 (s.len - k);
    s.base <- s.base + k;
    s.len <- s.len - k;
    s.start <- 0;
    s.stop <- s.stop - k;
    s.scanned <- s.scanned - k;
    s.i <- s.i - k
  end

let skip_bom s =
  while (not s.eod) && s.len < 3 do
    refill s
  done;
  if s.len >= 3 && Bytes.sub_string s.buf 0 3 = "\xEF\xBB\xBF" then begin
    s.start <- 3;
    s.stop <- 3;
    s.scanned <- 3;
    s.i <- 3;
    s.line_at <- 3
  end

let is_space c = c = ' ' || c = '\t' || c = '\r' || c = '\n'

(* The first pass moves [scanned] until a batch ends there, or the input does:
   past [bytes] of input or at [records] whole records, whichever comes
   first. *)
let frame_lines s ~bytes ~records =
  let found = ref false in
  while (not !found) && s.scanned < s.len do
    let c = Bytes.unsafe_get s.buf s.scanned in
    s.scanned <- s.scanned + 1;
    if c = '\n' then begin
      if not s.blank then s.found <- s.found + 1;
      s.blank <- true;
      if s.scanned - s.start >= bytes || s.found >= records then found := true
    end
    else if not (is_space c) then s.blank <- false
  done;
  !found

let frame_array s ~bytes ~records =
  let found = ref false in
  while (not !found) && s.scanned < s.len do
    let c = Bytes.unsafe_get s.buf s.scanned in
    s.scanned <- s.scanned + 1;
    if s.in_string then begin
      if s.escaped then s.escaped <- false
      else if c = '\\' then s.escaped <- true
      else if c = '"' then s.in_string <- false
    end
    else begin
      match c with
      | '"' -> s.in_string <- true
      | '[' | '{' -> s.depth <- s.depth + 1
      | ']' | '}' ->
          s.depth <- s.depth - 1;
          if s.depth <= 0 then found := true
      | ',' when s.depth = 1 ->
          s.found <- s.found + 1;
          if s.scanned - s.start >= bytes || s.found >= records then
            found := true
      | _ -> ()
    end;
    if s.depth <= 0 && s.scanned - s.start >= bytes then found := true
  done;
  !found

let frame s ~bytes ~records =
  s.found <- 0;
  let scan =
    match s.framing with Lines -> frame_lines | Array -> frame_array
  in
  let rec go () =
    if s.scanned = s.len && not s.eod then refill s;
    if s.scanned = s.len && s.eod then s.stop <- s.scanned
    else if scan s ~bytes ~records then s.stop <- s.scanned
    else go ()
  in
  go ()

(* Where the second pass is at the end of the batch: the lines before its next
   batch, and where the last of them starts. *)
let move_lines s stop =
  let last = ref (-1) in
  for k = s.start to stop - 1 do
    if Bytes.unsafe_get s.buf k = '\n' then begin
      s.line <- s.line + 1;
      last := k
    end
  done;
  if !last >= 0 then s.line_at <- s.base + !last + 1

let locate s p =
  let line = ref s.line and line_at = ref s.line_at in
  for k = s.start to min p s.len - 1 do
    if Bytes.unsafe_get s.buf k = '\n' then begin
      incr line;
      line_at := s.base + k + 1
    end
  done;
  (!line, s.base + p - !line_at + 1, s.base + p)

let fail s p ?text msg =
  let line, column, byte = locate s p in
  raise (Error { line; column; byte; text; msg })

let end_of_input s p =
  fail s p
    (match s.framing with
    | Lines -> "the record ends in the middle of a value"
    | Array -> "the input ends before the array does")

let unclosed s = s.framing = Array && s.phase <> Closed && not s.sampled

let next s =
  if s.line = 0 then begin
    skip_bom s;
    s.line <- 1
  end
  else begin
    move_lines s s.stop;
    s.start <- s.stop;
    compact s
  end;
  s.i <- s.start;
  if s.start = s.len && s.eod then begin
    if unclosed s then end_of_input s s.start;
    false
  end
  else begin
    frame s ~bytes:batch_bytes ~records:max_int;
    s.i <- s.start;
    s.limit <- s.stop;
    if s.stop > s.start then true
    else if unclosed s then end_of_input s s.start
    else false
  end

let bytes s = s.buf
let pos s = s.i
let name_pos s = s.name_pos
let text_pos s = s.text_pos
let text_len s = s.text_len

let skip_ws s =
  while s.i < s.limit && is_space (Bytes.unsafe_get s.buf s.i) do
    s.i <- s.i + 1
  done

let peek s =
  skip_ws s;
  if s.i < s.limit then Bytes.unsafe_get s.buf s.i else '\000'

let advance s = s.i <- s.i + 1


let unexpected s what =
  if s.i >= s.limit then end_of_input s s.i
  else
    fail s s.i
      (Printf.sprintf "%s, found the byte %C" what (Bytes.get s.buf s.i))

let expect s c =
  if peek s = c then s.i <- s.i + 1
  else unexpected s (Printf.sprintf "expected %C" c)

let literal s lit =
  let n = String.length lit in
  let rec same k =
    k = n
    || Bytes.unsafe_get s.buf (s.i + k) = String.unsafe_get lit k
       && same (k + 1)
  in
  if s.i + n <= s.limit && same 0 then s.i <- s.i + n
  else fail s s.i ("expected " ^ lit)

(* A line break ends the record: NDJSON's values hold no line feed or
   carriage return. *)
let record_end s =
  let k = ref s.i in
  while
    !k < s.stop
    &&
    let c = Bytes.unsafe_get s.buf !k in
    c <> '\n' && c <> '\r'
  do
    incr k
  done;
  !k

let rec record s =
  match s.framing with
  | Lines ->
      s.limit <- s.stop;
      skip_ws s;
      if s.i >= s.stop then false
      else begin
        s.limit <- record_end s;
        true
      end
  | Array -> (
      s.limit <- s.stop;
      skip_ws s;
      match s.phase with
      | _ when s.i >= s.stop -> false
      | Start ->
          if Bytes.get s.buf s.i <> '[' then
            fail s s.i "the input does not start with an array";
          advance s;
          s.phase <- First;
          record s
      | First ->
          if Bytes.get s.buf s.i = ']' then begin
            advance s;
            s.phase <- Closed;
            record s
          end
          else begin
            s.phase <- After;
            true
          end
      | Element ->
          if Bytes.get s.buf s.i = ']' then
            fail s s.i "a comma is followed by the array's end"
          else begin
            s.phase <- After;
            true
          end
      | After -> assert false
      | Closed -> fail s s.i "bytes follow the array")

let end_record s =
  match s.framing with
  | Lines ->
      skip_ws s;
      if s.i < s.limit then
        fail s s.i "the line holds more than one JSON value";
      (* A carriage return ends a line only before a line feed. *)
      let k =
        if s.limit < s.stop && Bytes.get s.buf s.limit = '\r' then s.limit + 1
        else s.limit
      in
      if k < s.stop && Bytes.get s.buf k <> '\n' then
        fail s s.limit "a carriage return not before a line feed";
      if k < s.stop then s.i <- k + 1 else s.i <- k
  | Array -> (
      match peek s with
      | ',' ->
          advance s;
          s.phase <- Element
      | ']' ->
          advance s;
          s.phase <- Closed
      | _ -> unexpected s "expected a comma or the array's end")

let container close = if close = ']' then "array" else "object"

let enter s close =
  advance s;
  if peek s = close then begin
    advance s;
    false
  end
  else true

let continue_ s close =
  match peek s with
  | ',' ->
      advance s;
      true
  | c when c = close ->
      advance s;
      false
  | _ -> unexpected s ("expected a comma or the " ^ container close ^ "'s end")

let is_digit c = '0' <= c && c <= '9'

(* The bytes of a number, by RFC 8259's grammar, from [i]. *)
let number s =
  let get i = if i < s.limit then Bytes.unsafe_get s.buf i else '\000' in
  let digits i =
    if not (is_digit (get i)) then begin
      s.i <- i;
      unexpected s "expected a digit"
    end;
    let k = ref i in
    while is_digit (get !k) do
      incr k
    done;
    !k
  in
  let i = if get s.i = '-' then s.i + 1 else s.i in
  let i =
    if get i = '0' then begin
      if is_digit (get (i + 1)) then
        fail s (i + 1) "a number has a leading zero";
      i + 1
    end
    else digits i
  in
  let fraction = get i = '.' in
  let i = if fraction then digits (i + 1) else i in
  let exponent = get i = 'e' || get i = 'E' in
  let i =
    if exponent then
      let j = i + 1 in
      digits (if get j = '+' || get j = '-' then j + 1 else j)
    else i
  in
  s.i <- i;
  not (fraction || exponent)

let hex c =
  match c with
  | '0' .. '9' -> Char.code c - 48
  | 'a' .. 'f' -> Char.code c - 87
  | 'A' .. 'F' -> Char.code c - 55
  | _ -> -1

(* [unicode s p] is the code unit of the [\uXXXX] at [p] and is [-1] if it is
   not one. *)
let unicode s p =
  if
    p + 6 > s.limit
    || Bytes.get s.buf p <> '\\'
    || Bytes.get s.buf (p + 1) <> 'u'
  then -1
  else
    let digit k = hex (Bytes.get s.buf (p + 2 + k)) in
    let a = digit 0 and b = digit 1 and c = digit 2 and d = digit 3 in
    if a < 0 || b < 0 || c < 0 || d < 0 then -1
    else (a * 4096) + (b * 256) + (c * 16) + d

let is_high u = u >= 0xD800 && u <= 0xDBFF
let is_low u = u >= 0xDC00 && u <= 0xDFFF

(* [check_escape s p] is the position after the escape at [p], which is
   checked. *)
let check_escape s p =
  if p + 1 >= s.limit then end_of_input s p;
  match Bytes.get s.buf (p + 1) with
  | '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' -> p + 2
  | 'u' ->
      let u = unicode s p in
      if u < 0 then fail s p "a \\u escape needs four hexadecimal digits";
      if is_high u then begin
        let v = unicode s (p + 6) in
        if not (is_low v) then fail s p "an unpaired surrogate escape";
        p + 12
      end
      else if is_low u then fail s p "an unpaired surrogate escape"
      else p + 6
  | c -> fail s p (Printf.sprintf "the escape \\%c is not JSON's" c)

let string s =
  let p0 = s.i + 1 in
  let rec go i escaped =
    if i >= s.limit then
      fail s s.i
        (if s.framing = Lines then "the string does not end on its line"
         else "the string does not end")
    else
      let c = Bytes.unsafe_get s.buf i in
      if c = '"' then begin
        s.text_pos <- p0;
        s.text_len <- i - p0;
        s.i <- i + 1;
        escaped
      end
      else if c = '\\' then go (check_escape s i) true
      else if c < ' ' then
        fail s i "a control character in a string is written as an escape"
      else if c < '\x80' then go (i + 1) escaped
      else begin
        let d = Bytes.get_utf_8_uchar s.buf i in
        if not (Uchar.utf_decode_is_valid d) then
          fail s i "a string is not valid UTF-8";
        go (i + Uchar.utf_decode_length d) escaped
      end
  in
  go p0 false

let unescape s b at =
  let first = s.text_pos in
  let last = first + s.text_len in
  let rec go i o =
    if i >= last then o - at
    else
      let c = Bytes.unsafe_get s.buf i in
      if c <> '\\' then begin
        Bytes.unsafe_set b o c;
        go (i + 1) (o + 1)
      end
      else
        let e = Bytes.get s.buf (i + 1) in
        match e with
        | 'u' ->
            let u = unicode s i in
            let u, n =
              if is_high u then
                let v = unicode s (i + 6) in
                (0x10000 + ((u - 0xD800) lsl 10) + (v - 0xDC00), 12)
              else (u, 6)
            in
            let k = Bytes.set_utf_8_uchar b o (Uchar.of_int u) in
            go (i + n) (o + k)
        | _ ->
            let c =
              match e with
              | 'b' -> '\b'
              | 'f' -> '\012'
              | 'n' -> '\n'
              | 'r' -> '\r'
              | 't' -> '\t'
              | c -> c
            in
            Bytes.unsafe_set b o c;
            go (i + 2) (o + 1)
  in
  go first at

let member s =
  if peek s <> '"' then unexpected s "expected a string as a name";
  s.name_pos <- s.i;
  let escaped = string s in
  expect s ':';
  escaped

(* [skip_value] reads a value of any depth with a stack of one bit per open
   container, [1] for an object: mutual tail calls, on no call stack. *)
let skip_value s =
  let depth = ref 0 in
  let push is_object =
    let d = !depth in
    if d / 8 >= Bytes.length s.stack then
      s.stack <- Bytes.extend s.stack 0 (Bytes.length s.stack);
    let byte = Char.code (Bytes.get s.stack (d / 8)) in
    let mask = 1 lsl (d mod 8) in
    Bytes.set s.stack (d / 8)
      (Char.chr (if is_object then byte lor mask else byte land lnot mask));
    depth := d + 1
  in
  let top_is_object () =
    let d = !depth - 1 in
    Char.code (Bytes.get s.stack (d / 8)) land (1 lsl (d mod 8)) <> 0
  in
  let rec value () =
    match peek s with
    | '"' ->
        ignore (string s);
        after ()
    | '{' ->
        if enter s '}' then begin
          push true;
          name ()
        end
        else after ()
    | '[' ->
        if enter s ']' then begin
          push false;
          value ()
        end
        else after ()
    | 't' ->
        literal s "true";
        after ()
    | 'f' ->
        literal s "false";
        after ()
    | 'n' ->
        literal s "null";
        after ()
    | '-' | '0' .. '9' ->
        ignore (number s);
        after ()
    | _ -> unexpected s "expected a value"
  and name () =
    ignore (member s);
    value ()
  and after () =
    if !depth > 0 then
      if top_is_object () then
        if continue_ s '}' then name ()
        else begin
          decr depth;
          after ()
        end
      else if continue_ s ']' then value ()
      else begin
        decr depth;
        after ()
      end
  in
  value ()

let sample ~records r =
  let s = make Lines r in
  skip_bom s;
  let rec first () =
    if s.i < s.len && is_space (Bytes.get s.buf s.i) then begin
      s.i <- s.i + 1;
      first ()
    end
    else if s.i = s.len && not s.eod then begin
      refill s;
      first ()
    end
    else if s.i < s.len then Bytes.get s.buf s.i
    else '\000'
  in
  if first () = '[' then s.framing <- Array;
  s.i <- s.start;
  frame s ~bytes:max_int ~records;
  let text = Bytes.sub_string s.buf 0 s.stop in
  if s.len > 0 then
    Reader.push_back r (Slice.of_bytes ~first:0 ~last:(s.len - 1) s.buf);
  (s.framing, text)
