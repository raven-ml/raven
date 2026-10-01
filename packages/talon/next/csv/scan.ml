(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Reader = Bytesrw.Bytes.Reader
module Slice = Bytesrw.Bytes.Slice

exception Error of { line : int; column : int; msg : string }

let batch_bytes = 1 lsl 20

type t = {
  reader : Reader.t;
  sep : char;
  quote : char;
  mutable buf : Bytes.t;
  mutable len : int; (* Bytes of [buf] that hold input. *)
  mutable eod : bool;
  mutable start : int; (* The batch's first byte. *)
  mutable stop : int; (* The byte after the batch. *)
  mutable line : int; (* The line of [start], [0] before the first batch. *)
  mutable lines : int; (* The line feeds of the batch. *)
  mutable scanned : int; (* Where the first pass is. *)
  mutable in_quotes : bool; (* The first pass's parity at [scanned]. *)
  mutable bounds : int array; (* Each field's [start] and [stop]. *)
  mutable records : int array; (* Each record's first field, then [nf]. *)
  mutable rows : int;
  mutable pending : exn option; (* The syntax error that ended the batch. *)
}

let make ~sep ~quote reader =
  {
    reader;
    sep;
    quote;
    buf = Bytes.create 65536;
    len = 0;
    eod = false;
    start = 0;
    stop = 0;
    line = 0;
    lines = 0;
    scanned = 0;
    in_quotes = false;
    bounds = Array.make 1024 0;
    records = Array.make 64 0;
    rows = 0;
    pending = None;
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
    s.len <- s.len - k;
    s.start <- 0;
    s.scanned <- s.scanned - k
  end

let skip_bom s =
  while (not s.eod) && s.len < 3 do
    refill s
  done;
  if s.len >= 3 && Bytes.sub_string s.buf 0 3 = "\xEF\xBB\xBF" then begin
    s.start <- 3;
    s.scanned <- 3
  end

let rec batch_end s =
  let b = s.buf and q = s.quote and last = s.start + batch_bytes - 1 in
  let i = ref s.scanned and in_quotes = ref s.in_quotes and found = ref (-1) in
  while !found < 0 && !i < s.len do
    let c = Bytes.unsafe_get b !i in
    if c = q then in_quotes := not !in_quotes
    else if c = '\n' && (not !in_quotes) && !i >= last then found := !i + 1;
    incr i
  done;
  s.scanned <- !i;
  s.in_quotes <- !in_quotes;
  if !found >= 0 then !found
  else if s.eod then s.len
  else begin
    refill s;
    batch_end s
  end

let locate s i =
  let line = ref s.line and first = ref s.start in
  for k = s.start to i - 1 do
    if Bytes.unsafe_get s.buf k = '\n' then begin
      incr line;
      first := k + 1
    end
  done;
  (!line, i - !first + 1)

exception Syntax of int * string

let push_field s nf first stop =
  if (2 * nf) + 2 > Array.length s.bounds then begin
    let a = Array.make (2 * Array.length s.bounds) 0 in
    Array.blit s.bounds 0 a 0 (2 * nf);
    s.bounds <- a
  end;
  Array.unsafe_set s.bounds (2 * nf) first;
  Array.unsafe_set s.bounds ((2 * nf) + 1) stop

let push_record s rows nf =
  if rows + 2 > Array.length s.records then begin
    let a = Array.make (2 * Array.length s.records) 0 in
    Array.blit s.records 0 a 0 (rows + 1);
    s.records <- a
  end;
  Array.unsafe_set s.records (rows + 1) nf

(* [split s stop] splits the batch's input, up to [stop], into records, skipping
   empty lines. On a syntax error the batch keeps the records before the one
   that holds it. *)
let split s stop =
  let b = s.buf and sep = s.sep and q = s.quote in
  let nf = ref 0 and rows = ref 0 and lines = ref 0 and i = ref s.start in
  s.records.(0) <- 0;
  (try
     while !i < stop do
       let c = Bytes.unsafe_get b !i in
       if c = '\n' then begin
         incr lines;
         incr i
       end
       else if c = '\r' && !i + 1 < stop && Bytes.unsafe_get b (!i + 1) = '\n'
       then begin
         incr lines;
         i := !i + 2
       end
       else begin
         let record_ends = ref false in
         while not !record_ends do
           let first = !i in
           let j = ref first in
           if first < stop && Bytes.unsafe_get b first = q then begin
             let closed = ref false in
             incr j;
             while not !closed do
               if !j >= stop then
                 raise (Syntax (first, "a quoted field does not end"));
               let c = Bytes.unsafe_get b !j in
               if c <> q then begin
                 if c = '\n' then incr lines;
                 incr j
               end
               else if !j + 1 < stop && Bytes.unsafe_get b (!j + 1) = q then
                 j := !j + 2
               else begin
                 closed := true;
                 incr j
               end
             done
           end
           else begin
             while
               !j < stop
               &&
               let c = Bytes.unsafe_get b !j in
               c <> sep && c <> '\n' && c <> '\r' && c <> q
             do
               incr j
             done;
             if !j < stop && Bytes.unsafe_get b !j = q then
               raise
                 (Syntax (!j, "a quote in a field that does not start with one"))
           end;
           push_field s !nf first !j;
           incr nf;
           let j = !j in
           if j = stop then begin
             i := stop;
             record_ends := true
           end
           else
             let c = Bytes.unsafe_get b j in
             if c = sep then i := j + 1
             else if c = '\n' then begin
               incr lines;
               i := j + 1;
               record_ends := true
             end
             else if
               c = '\r' && j + 1 < stop && Bytes.unsafe_get b (j + 1) = '\n'
             then begin
               incr lines;
               i := j + 2;
               record_ends := true
             end
             else if c = '\r' then
               raise (Syntax (j, "a carriage return that no line feed follows"))
             else
               raise
                 (Syntax
                    ( j,
                      "a quoted field's closing quote is followed by a byte \
                       other than a separator or a line break" ))
         done;
         push_record s !rows !nf;
         incr rows
       end
     done
   with Syntax (at, msg) ->
     let line, column = locate s at in
     s.pending <- Some (Error { line; column; msg }));
  s.rows <- !rows;
  s.lines <- !lines

let next s =
  Option.iter raise s.pending;
  if s.line = 0 then begin
    s.line <- 1;
    skip_bom s
  end
  else begin
    s.line <- s.line + s.lines;
    s.start <- s.stop;
    compact s
  end;
  let stop = batch_end s in
  s.stop <- stop;
  if stop = s.start then false
  else begin
    split s stop;
    if s.rows = 0 then Option.iter raise s.pending;
    true
  end

let rows s = s.rows
let fields s r = s.records.(r + 1) - s.records.(r)
let bytes s = s.buf
let start s r j = s.bounds.(2 * (s.records.(r) + j))
let stop s r j = s.bounds.((2 * (s.records.(r) + j)) + 1)

(* A field that does not start with the quote has no quote, so its first byte is
   the quote only when it is quoted. *)
let quoted s r j =
  let k = 2 * (s.records.(r) + j) in
  let first = Array.unsafe_get s.bounds k in
  Array.unsafe_get s.bounds (k + 1) > first
  && Bytes.unsafe_get s.buf first = s.quote

let pos s r j = if quoted s r j then start s r j + 1 else start s r j

let len s r j =
  let n = stop s r j - start s r j in
  if quoted s r j then n - 2 else n

let raw s r j = Bytes.sub_string s.buf (pos s r j) (len s r j)

let text s r j =
  let raw = raw s r j in
  if not (quoted s r j) then raw
  else begin
    let b = Buffer.create (String.length raw) in
    let i = ref 0 in
    while !i < String.length raw do
      Buffer.add_char b raw.[!i];
      i := if raw.[!i] = s.quote then !i + 2 else !i + 1
    done;
    Buffer.contents b
  end

let sample ~quote ~records r =
  let b = Buffer.create 65536 in
  let found = ref 0 and cut = ref (-1) and in_quotes = ref false in
  let empty = ref true in
  while !cut < 0 do
    let slice = Reader.read r in
    if Slice.is_eod slice then cut := Buffer.length b
    else begin
      let bytes = Slice.bytes slice and base = Buffer.length b in
      let i = ref (Slice.first slice) and last = Slice.last slice in
      while !cut < 0 && !i <= last do
        let c = Bytes.get bytes !i in
        if c = quote then in_quotes := not !in_quotes;
        if c = '\n' && not !in_quotes then begin
          if not !empty then begin
            incr found;
            if !found = records then cut := base + (!i - Slice.first slice) + 1
          end;
          empty := true
        end
        else if c <> '\r' then empty := false;
        incr i
      done;
      Slice.add_to_buffer b slice
    end
  done;
  if Buffer.length b > 0 then
    Reader.push_back r (Slice.of_bytes (Buffer.to_bytes b));
  Buffer.sub b 0 !cut
