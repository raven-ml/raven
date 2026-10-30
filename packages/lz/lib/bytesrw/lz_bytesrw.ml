open Bigarray
open Bytesrw

type Bytes.Stream.error += Error of Lz.error

let message = function
  | Error e -> Format.asprintf "%a" Lz.pp_error e
  | _ -> assert false (* Only [case] makes the errors of a format_error. *)

let format_error format =
  Bytes.Stream.make_format_error ~format ~case:(fun e -> Error e) ~message

(* A step over the input, an encoder's wrapped in [Ok]: the filters below serve
   decoders and encoders alike. *)
type step =
  last:bool ->
  Lz.bigbytes ->
  (Lz.bigbytes -> unit) ->
  (Lz.step, Lz.error) result

let decoding d ~last src f = Lz.Decoder.step d ~last src f
let encoding e ~last src f = Ok (Lz.Encoder.step e ~last src f)

(* Bytes after a deflate or zlib stream written on a decompressing writer, which
   cannot hand them back. *)
let after_stream at = { Lz.at; why = Malformed "bytes after the stream" }

(* [guard broken step ~last src f] is [step ~last src f], calling [broken x]
   before re-raising an exception [x] from [f] or [Out_of_memory]: the codec is
   dead then, and the filter raises [x] again on every later use instead of
   stepping a dead codec. *)
let guard broken (step : step) ~last src f =
  match step ~last src f with
  | r -> r
  | exception x ->
      let bt = Printexc.get_raw_backtrace () in
      broken x;
      Printexc.raise_with_backtrace x bt

(* Staging

   The codecs read bigarrays and bytesrw's slices are bytes: a slice's bytes are
   copied by memcpy into a staging bigarray, at most [stage_max] at a time, so
   that a long slice costs no more memory than a short one. *)

external stage : bytes -> int -> Lz.bigbytes -> int -> unit = "lz_bytesrw_stage"
[@@noalloc]

external unstage : Lz.bigbytes -> int -> bytes -> int -> unit
  = "lz_bytesrw_unstage"
[@@noalloc]

let stage_max = Bytes.Slice.io_buffer_size
let empty = Array1.create int8_unsigned c_layout 0

(* The input of a step: the slice's bytes from [next] to [stop] are not yet
   staged, and the staging bigarray's bytes from [pos] to [len] are staged and
   not yet taken. [taken] counts the bytes the codec took since it was made. *)
type input = {
  mutable stage : Lz.bigbytes;
  mutable pos : int;
  mutable len : int;
  mutable bytes : bytes;
  mutable next : int;
  mutable stop : int;
  mutable taken : int;
}

let input hint =
  let stage = Array1.create int8_unsigned c_layout (min hint stage_max) in
  {
    stage;
    pos = 0;
    len = 0;
    bytes = Bytes.empty;
    next = 0;
    stop = 0;
    taken = 0;
  }

let pending i = i.pos < i.len || i.next < i.stop

(* [feed i s] makes [s] the input once the last slice is all taken. *)
let feed i s =
  let n = Bytes.Slice.length s in
  if n > Array1.dim i.stage && Array1.dim i.stage < stage_max then
    i.stage <- Array1.create int8_unsigned c_layout (min n stage_max);
  i.bytes <- Bytes.Slice.bytes s;
  i.next <- Bytes.Slice.first s;
  i.stop <- i.next + n

(* [chunk i] is the staged bytes not yet taken, staging the next ones when all
   are taken. *)
let chunk i =
  if i.pos = i.len then begin
    let n = min (i.stop - i.next) (Array1.dim i.stage) in
    stage i.bytes i.next i.stage n;
    i.next <- i.next + n;
    i.pos <- 0;
    i.len <- n
  end;
  Array1.sub i.stage i.pos (i.len - i.pos)

let take i k =
  i.pos <- i.pos + k;
  i.taken <- i.taken + k

(* [rest i] is the slice of the input's bytes not taken. *)
let rest i =
  let first = i.next - (i.len - i.pos) in
  Bytes.Slice.make_or_eod i.bytes ~first ~length:(i.stop - first)

(* Readers

   A read returns the next piece of the last run, of at most [slice_length]
   bytes, and steps when the run is used up: on the staged input, on the next
   slice of [r], and with [~last:true] once [r] has returned its end. The run is
   copied out in [f], since it is lent only until [f] returns, so the reader
   holds at most one run's copy. At [Done] the bytes the codec did not take go
   back to [r]. *)

type reading = Open | Ended | Fin | Failed of exn

let reads fmt (step : step) ?(pos = 0) ?slice_length r =
  let slice_length =
    match slice_length with Some n -> n | None -> Bytes.Reader.slice_length r
  in
  let base = Bytes.Reader.pos r in
  let i = input (Bytes.Reader.slice_length r) in
  let out = ref Bytes.empty and out_pos = ref 0 and out_len = ref 0 in
  let state = ref Open in
  let step = guard (fun x -> state := Failed x) step in
  let lend run =
    let n = Array1.dim run in
    if n > Bytes.length !out then out := Bytes.create n;
    unstage run 0 !out n;
    out_pos := 0;
    out_len := n
  in
  let fail (e : Lz.error) =
    try Bytes.Reader.error fmt r ~pos:(base + e.at) e
    with x ->
      state := Failed x;
      raise x
  in
  let rec read () =
    if !out_pos < !out_len then begin
      let length = min slice_length (!out_len - !out_pos) in
      let s = Bytes.Slice.make !out ~first:!out_pos ~length in
      out_pos := !out_pos + length;
      s
    end
    else
      match !state with
      | Fin -> Bytes.Slice.eod
      | Failed x -> raise x
      | Open when pending i -> advance (step ~last:false (chunk i) lend)
      | Ended -> advance (step ~last:true empty lend)
      | Open ->
          let s = Bytes.Reader.read r in
          if Bytes.Slice.is_eod s then state := Ended else feed i s;
          read ()
  and advance = function
    | Ok (Lz.More k) ->
        take i k;
        read ()
    | Ok (Done k) ->
        take i k;
        Bytes.Reader.push_back r (rest i);
        state := Fin;
        read ()
    | Error e -> fail e
  in
  Bytes.Reader.make ~pos ~slice_length read

(* Writers

   A write steps until the codec has taken all of the slice, and each run is
   written on [w] in [f], in slices of at most [w]'s slice length copied out
   through [buf]. The end of data steps with [~last:true] until [Done]. *)

type writing = Taking | Finished | Broken of exn

let writes fmt (step : step) ?(pos = 0) ?slice_length ~eod w =
  let slice_length =
    match slice_length with Some n -> n | None -> Bytes.Writer.slice_length w
  in
  let i = input slice_length in
  let buf = ref Bytes.empty in
  let state = ref Taking in
  let step = guard (fun x -> state := Broken x) step in
  let lend run =
    let n = Array1.dim run and most = Bytes.Writer.slice_length w in
    let rec out first =
      if first < n then begin
        let length = min most (n - first) in
        if length > Bytes.length !buf then buf := Bytes.create length;
        unstage run first !buf length;
        Bytes.Writer.write w (Bytes.Slice.make !buf ~first:0 ~length);
        out (first + length)
      end
    in
    out 0
  in
  let fail wf (e : Lz.error) =
    try Bytes.Writer.error fmt wf ~pos:(pos + e.at) e
    with x ->
      state := Broken x;
      raise x
  in
  let rec push wf =
    if pending i then
      match step ~last:false (chunk i) lend with
      | Ok (Lz.More k) ->
          take i k;
          push wf
      | Ok (Done k) ->
          take i k;
          state := Finished;
          if pending i then fail wf (after_stream i.taken)
      | Error e -> fail wf e
  in
  let rec close wf =
    match step ~last:true empty lend with
    | Ok (Lz.More _) -> close wf
    | Ok (Done _) ->
        state := Finished;
        if eod then Bytes.Writer.write_eod w
    | Error e -> fail wf e
  in
  let write wf s =
    let at_end = Bytes.Slice.is_eod s in
    match !state with
    | Broken x -> raise x
    | Finished when at_end -> if eod then Bytes.Writer.write_eod w
    | Finished -> fail wf (after_stream i.taken)
    | Taking when at_end -> close wf
    | Taking ->
        feed i s;
        push wf
  in
  Bytes.Writer.make' ~pos ~slice_length write

(* Formats *)

module Zstd = struct
  let fmt = format_error "zstd"

  let decompress_reads ?window () ?pos ?slice_length r =
    reads fmt (decoding (Lz.Zstd.decoder ?window ())) ?pos ?slice_length r

  let decompress_writes ?window () ?pos ?slice_length ~eod w =
    writes fmt (decoding (Lz.Zstd.decoder ?window ())) ?pos ?slice_length ~eod w

  let compress_reads ?level () ?pos ?slice_length r =
    reads fmt (encoding (Lz.Zstd.encoder ?level ())) ?pos ?slice_length r

  let compress_writes ?level () ?pos ?slice_length ~eod w =
    writes fmt (encoding (Lz.Zstd.encoder ?level ())) ?pos ?slice_length ~eod w
end

module Gzip = struct
  let fmt = format_error "gzip"

  let decompress_reads () ?pos ?slice_length r =
    reads fmt (decoding (Lz.Gzip.decoder ())) ?pos ?slice_length r

  let decompress_writes () ?pos ?slice_length ~eod w =
    writes fmt (decoding (Lz.Gzip.decoder ())) ?pos ?slice_length ~eod w

  let compress_reads ?level () ?pos ?slice_length r =
    reads fmt (encoding (Lz.Gzip.encoder ?level ())) ?pos ?slice_length r

  let compress_writes ?level () ?pos ?slice_length ~eod w =
    writes fmt (encoding (Lz.Gzip.encoder ?level ())) ?pos ?slice_length ~eod w
end

module Zlib = struct
  let fmt = format_error "zlib"

  let decompress_reads () ?pos ?slice_length r =
    reads fmt (decoding (Lz.Zlib.decoder ())) ?pos ?slice_length r

  let decompress_writes () ?pos ?slice_length ~eod w =
    writes fmt (decoding (Lz.Zlib.decoder ())) ?pos ?slice_length ~eod w

  let compress_reads ?level () ?pos ?slice_length r =
    reads fmt (encoding (Lz.Zlib.encoder ?level ())) ?pos ?slice_length r

  let compress_writes ?level () ?pos ?slice_length ~eod w =
    writes fmt (encoding (Lz.Zlib.encoder ?level ())) ?pos ?slice_length ~eod w
end

module Deflate = struct
  let fmt = format_error "deflate"

  let decompress_reads () ?pos ?slice_length r =
    reads fmt (decoding (Lz.Deflate.decoder ())) ?pos ?slice_length r

  let decompress_writes () ?pos ?slice_length ~eod w =
    writes fmt (decoding (Lz.Deflate.decoder ())) ?pos ?slice_length ~eod w

  let compress_reads ?level () ?pos ?slice_length r =
    reads fmt (encoding (Lz.Deflate.encoder ?level ())) ?pos ?slice_length r

  let compress_writes ?level () ?pos ?slice_length ~eod w =
    writes fmt
      (encoding (Lz.Deflate.encoder ?level ()))
      ?pos ?slice_length ~eod w
end
