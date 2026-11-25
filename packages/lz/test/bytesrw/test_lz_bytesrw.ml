(* Lz_bytesrw's filters against Lz's whole-buffer calls and steps. A
   decompressing reader or writer yields the bytes [decode] yields and refuses
   what it refuses, at the same offset, whatever the slice lengths on either
   side; a deflate or zlib reader leaves its reader after the stream; a
   compressing filter writes the stream its encoder writes. *)

open Windtrap
open Bytesrw

let kib = 1024

(* Formats *)

type format = {
  name : string;
  text_100 : string; (* A golden of 100 bytes of text. *)
  decode : Lz.bigbytes -> Lz.bigbytes -> (Lz.progress, Lz.error) result;
  reads : Bytes.Reader.filter;
  writes : Bytes.Writer.filter;
  compress_reads : ?level:int -> unit -> Bytes.Reader.filter;
  compress_writes : ?level:int -> unit -> Bytes.Writer.filter;
  encoder : ?level:int -> unit -> Lz.Encoder.t;
  levels : int * int; (* The lowest and highest. *)
}

(* A mutated Zstandard header may declare any window up to the decoder's bound,
   which the decoder allocates; 4 MiB holds every golden's. *)
let zstd_window = 4 * 1024 * kib

let deflate =
  {
    name = "deflate";
    text_100 = "text_100.deflate6";
    decode = Lz.Deflate.decode;
    reads = Lz_bytesrw.Deflate.decompress_reads ();
    writes = Lz_bytesrw.Deflate.decompress_writes ();
    compress_reads = Lz_bytesrw.Deflate.compress_reads;
    compress_writes = Lz_bytesrw.Deflate.compress_writes;
    encoder = Lz.Deflate.encoder;
    levels = (1, 9);
  }

let zlib =
  {
    name = "zlib";
    text_100 = "text_100.zlib6";
    decode = Lz.Zlib.decode;
    reads = Lz_bytesrw.Zlib.decompress_reads ();
    writes = Lz_bytesrw.Zlib.decompress_writes ();
    compress_reads = Lz_bytesrw.Zlib.compress_reads;
    compress_writes = Lz_bytesrw.Zlib.compress_writes;
    encoder = Lz.Zlib.encoder;
    levels = (1, 9);
  }

let gzip =
  {
    name = "gzip";
    text_100 = "text_100.gz9";
    decode = Lz.Gzip.decode;
    reads = Lz_bytesrw.Gzip.decompress_reads ();
    writes = Lz_bytesrw.Gzip.decompress_writes ();
    compress_reads = Lz_bytesrw.Gzip.compress_reads;
    compress_writes = Lz_bytesrw.Gzip.compress_writes;
    encoder = Lz.Gzip.encoder;
    levels = (1, 9);
  }

let zstd =
  {
    name = "zstd";
    text_100 = "text_100.zst3";
    decode = Lz.Zstd.decode;
    reads = Lz_bytesrw.Zstd.decompress_reads ~window:zstd_window ();
    writes = Lz_bytesrw.Zstd.decompress_writes ~window:zstd_window ();
    compress_reads = Lz_bytesrw.Zstd.compress_reads;
    compress_writes = Lz_bytesrw.Zstd.compress_writes;
    encoder = Lz.Zstd.encoder;
    levels = (1, 12);
  }

let formats = [ deflate; zlib; gzip; zstd ]

(* Goldens *)

let data g = Lazy.force g.Lz_support.data
let raw g = Lazy.force g.Lz_support.raw

let golden name =
  List.find (fun g -> g.Lz_support.file = name) (Lz_support.goldens ())

let text_100 fmt = golden fmt.text_100
let pp_golden ppf g = Format.pp_print_string ppf g.Lz_support.file

(* A format's goldens up to 1 KiB, whose stream ends in the first slices of a
   reader, and its first past 64 KiB, the most a filter stages at a time. *)
let ends fmt =
  let goldens = Lz_support.goldens_of fmt.name in
  List.filter (fun g -> g.Lz_support.size <= kib) goldens
  @ [ List.find (fun g -> g.Lz_support.size > 64 * kib) goldens ]

(* Slice lengths: one byte, a few, odd, and up to past every golden. *)
let lengths = [ 1; 2; 3; 7; 64; 4 * kib; 64 * kib; 1024 * kib ]

let length =
  Gen.frequency
    [
      (3, Gen.int_range 1 8);
      (2, Gen.of_list lengths);
      (1, Gen.int_range 1 (64 * kib));
    ]

(* The error a writer raises at the first byte after a deflate or zlib stream,
   with the message the .mli states. *)
let after_stream at = { Lz.at; why = Malformed "bytes after the stream" }

(* Through a filter *)

let caught f =
  match f () with
  | v -> Ok v
  | exception Bytes.Stream.Error (Lz_bytesrw.Error e, _) -> Error e

let add b s =
  Buffer.add_subbytes b (Bytes.Slice.bytes s) (Bytes.Slice.first s)
    (Bytes.Slice.length s)

let check_length ~most s =
  if Bytes.Slice.length s > most then
    failf "a slice of %d bytes, above its slice length %d"
      (Bytes.Slice.length s) most

(* [read_through filter ~cut ~most data] reads [data], in slices of [cut] bytes,
   through [filter] with slice length [most], adding the bytes read to [into].
   It is the bytes read or the error, and the reader of [data]. *)
let read_through ?(into = Buffer.create 0) (filter : Bytes.Reader.filter) ~cut
    ~most data =
  let r = Bytes.Reader.of_string ~slice_length:cut data in
  let d = filter ~slice_length:most r in
  let rec go () =
    let s = Bytes.Reader.read d in
    if Bytes.Slice.is_eod s then Buffer.contents into
    else begin
      check_length ~most s;
      add into s;
      go ()
    end
  in
  (caught go, r)

(* [write_through filter ~cut ~most data] writes [data], in slices of [cut]
   bytes, and its end on [filter] over a writer of slice length [most], adding
   the bytes written on that writer to [into]. It is the bytes written or the
   error. *)
let write_through ?(into = Buffer.create 0) (filter : Bytes.Writer.filter) ~cut
    ~most data =
  let w =
    Bytes.Writer.make ~slice_length:most (fun s ->
        if not (Bytes.Slice.is_eod s) then begin
          check_length ~most s;
          add into s
        end)
  in
  let f = filter ~slice_length:cut ~eod:true w in
  caught (fun () ->
      Bytes.Writer.write_string f data;
      Bytes.Writer.write_eod f;
      Buffer.contents into)

(* [decode fmt ~room data] is the whole-buffer decode of [data] into [room]
   bytes: the bytes and the input consumed, or the error. *)
let decode fmt ~room data =
  let dst = Lz_support.create room in
  match fmt.decode (Lz_support.of_string data) dst with
  | Ok { Lz.consumed; produced } ->
      Ok (Lz_support.to_string ~length:produced dst, consumed)
  | Error e -> Error e

(* What a decompressing reader and writer yield on bytes that [decode] decodes
   to [out], consuming [consumed]: a writer refuses the bytes after a stream
   that ends early. *)
let as_read = function Ok (out, _) -> Ok out | Error e -> Error e

let as_written data = function
  | Ok (_, consumed) when consumed < String.length data ->
      Error (after_stream consumed)
  | Ok (out, _) -> Ok out
  | Error e -> Error e

(* The error that [f ()] raises, if any. *)
let refused f =
  match f () with
  | _ -> None
  | exception Bytes.Stream.Error (Lz_bytesrw.Error e, _) -> Some e

(* The message of the stream error [f ()] raises, the only place bytesrw shows
   its position. *)
let raised f =
  match f () with
  | _ -> failf "no error"
  | exception Bytes.Stream.Error err -> Bytes.Stream.error_message err

(* The end of bytesrw's message of an error [e] at [pos] for a reader or a
   writer: ["KIND:POS: MESSAGE"], the message printed by [Lz.pp_error]. The
   format's name before it is lz_bytesrw's, which its .mli does not state. *)
let message kind pos e = Format.asprintf "%s:%d: %a" kind pos Lz.pp_error e

(* Conventions *)

(* The position of the reader and the writer a filter is applied to, other than
   0, so that a filter whose position defaults to theirs is seen. *)
let elsewhere = 11

(* [reader_positions ~pos filter data] is the filtering reader's position before
   and after it reads all of [data], and the count of bytes read. *)
let reader_positions ?pos (filter : Bytes.Reader.filter) data =
  let d = filter ?pos (Bytes.Reader.of_string ~pos:elsewhere data) in
  let first = Bytes.Reader.pos d in
  let read = Bytes.Reader.to_string d in
  ((first, Bytes.Reader.pos d), String.length read)

(* [writer_positions ~pos filter data] is the filtering writer's position before
   and after [data] is written on it. *)
let writer_positions ?pos (filter : Bytes.Writer.filter) data =
  let f = filter ?pos ~eod:true (Bytes.Writer.ignore ~pos:elsewhere ()) in
  let first = Bytes.Writer.pos f in
  Bytes.Writer.write_string f data;
  (first, Bytes.Writer.pos f)

let positions fmt =
  test fmt.name (fun () ->
      let g = text_100 fmt in
      [ None; Some 7 ]
      |> List.iter (fun pos ->
          let start = Option.value pos ~default:0 in
          let reader name filter data =
            let got, n = reader_positions ?pos filter data in
            equal ~msg:name (pair int int) (start, start + n) got
          in
          let writer name filter data =
            equal ~msg:name (pair int int)
              (start, start + String.length data)
              (writer_positions ?pos filter data)
          in
          reader "decompress_reads" fmt.reads (data g);
          writer "decompress_writes" fmt.writes (data g);
          reader "compress_reads" (fmt.compress_reads ()) (raw g);
          writer "compress_writes" (fmt.compress_writes ()) (raw g)))

let reader_slice_length fmt =
  test fmt.name (fun () ->
      let length (filter : Bytes.Reader.filter) =
        Bytes.Reader.slice_length
          (filter (Bytes.Reader.of_string ~slice_length:7 ""))
      in
      equal ~msg:"decompress_reads" int 7 (length fmt.reads);
      equal ~msg:"compress_reads" int 7 (length (fmt.compress_reads ())))

let writer_slice_length fmt =
  test fmt.name (fun () ->
      let length (filter : Bytes.Writer.filter) =
        Bytes.Writer.slice_length
          (filter ~eod:true (Bytes.Writer.ignore ~slice_length:7 ()))
      in
      equal ~msg:"decompress_writes" int 7 (length fmt.writes);
      equal ~msg:"compress_writes" int 7 (length (fmt.compress_writes ())))

(* [eod_at filter ~eod data] writes [data] and its end on [filter ~eod] over a
   writer [w]. It is the count of bytes [w] had taken when [eod] reached it, if
   it did, and the bytes it took in all. *)
let eod_at (filter : Bytes.Writer.filter) ~eod data =
  let taken = Buffer.create 0 and at = ref None in
  let w =
    Bytes.Writer.make (fun s ->
        if Bytes.Slice.is_eod s then at := Some (Buffer.length taken)
        else add taken s)
  in
  let f = filter ~eod w in
  Bytes.Writer.write_string f data;
  Bytes.Writer.write_eod f;
  (!at, Buffer.contents taken)

let ends_with_eod fmt =
  test fmt.name (fun () ->
      let g = text_100 fmt in
      let check name filter data =
        let at, stream = eod_at filter ~eod:true data in
        equal ~msg:(name ^ " ~eod:true") (option int)
          (Some (String.length stream))
          at;
        let at, bytes = eod_at filter ~eod:false data in
        equal ~msg:(name ^ " ~eod:false")
          (pair (option int) Lz_support.decoded)
          (None, Ok stream) (at, Ok bytes)
      in
      check "decompress_writes" fmt.writes (data g);
      check "compress_writes" (fmt.compress_writes ()) (raw g))

let levels fmt =
  let lo, hi = fmt.levels in
  cases fmt.name ~id:string_of_int
    [ lo - 1; hi + 1 ]
    (fun level ->
      raises_match Exn.invalid_arg (fun () ->
          fmt.compress_reads ~level () (Bytes.Reader.empty ()));
      raises_match Exn.invalid_arg (fun () ->
          fmt.compress_writes ~level () ~eod:true (Bytes.Writer.ignore ())))

(* 1 KiB is the least window Lz.Zstd.decoder takes. *)
let small_window =
  test "a Zstandard window below 1 KiB raises Invalid_argument when applied"
    (fun () ->
      let reads window =
        Lz_bytesrw.Zstd.decompress_reads ~window () (Bytes.Reader.empty ())
      in
      let writes window =
        Lz_bytesrw.Zstd.decompress_writes ~window () ~eod:true
          (Bytes.Writer.ignore ())
      in
      raises_match Exn.invalid_arg (fun () -> reads 1023);
      raises_match Exn.invalid_arg (fun () -> writes 1023);
      ignore (reads 1024 : Bytes.Reader.t);
      ignore (writes 1024 : Bytes.Writer.t))

(* Errors *)

(* Every proper prefix of a golden of one member, up to 1 KiB, is cut short, at
   its length, and what the filter delivered before is decoded bytes. *)
let cut fmt =
  let files =
    List.filter
      (fun g -> g.Lz_support.members = 1 && g.Lz_support.size <= kib)
      (Lz_support.goldens_of fmt.name)
  in
  let lengths = Array.of_list lengths in
  cases fmt.name
    ~id:(fun g -> g.Lz_support.file)
    files
    (fun g ->
      let data = data g and raw = raw g in
      let into = Buffer.create (String.length raw) in
      for k = 0 to String.length data - 1 do
        let prefix = String.sub data 0 k in
        let cut = lengths.(k mod Array.length lengths) in
        let msg = Printf.sprintf "cut at %d, slices of %d bytes" k cut in
        let expected = Error { Lz.at = k; why = Truncated } in
        Buffer.clear into;
        equal ~msg Lz_support.decoded expected
          (fst (read_through ~into fmt.reads ~cut ~most:cut prefix));
        starts_with ~msg:(msg ^ ", read") ~prefix:(Buffer.contents into) raw;
        Buffer.clear into;
        equal ~msg Lz_support.decoded expected
          (write_through ~into fmt.writes ~cut ~most:cut prefix);
        starts_with ~msg:(msg ^ ", written") ~prefix:(Buffer.contents into) raw
      done)

(* A stream cut at half of a golden of 100 bytes of text, refused at the cut. *)
let half fmt =
  let data = data (text_100 fmt) in
  let k = String.length data / 2 in
  (String.sub data 0 k, { Lz.at = k; why = Truncated })

let reader_position fmt =
  test fmt.name (fun () ->
      let cut, e = half fmt in
      let skipped = 3 in
      let r = Bytes.Reader.of_string ~slice_length:7 ("xyz" ^ cut) in
      Bytes.Reader.skip skipped r;
      let d = fmt.reads r in
      ends_with
        ~suffix:(message "reader" (skipped + e.at) e)
        (raised (fun () -> Bytes.Reader.discard d)))

let writer_position fmt =
  test fmt.name (fun () ->
      let cut, e = half fmt in
      let start = 5 in
      let f = fmt.writes ~pos:start ~eod:true (Bytes.Writer.ignore ()) in
      ends_with
        ~suffix:(message "writer" (start + e.at) e)
        (raised (fun () ->
             Bytes.Writer.write_string f cut;
             Bytes.Writer.write_eod f)))

(* A first byte no stream of any format begins with. *)
let bad = "\x06\x00\x00\x00"

let decode_error fmt data =
  match decode fmt ~room:kib data with
  | Error e -> e
  | Ok _ -> failf "%S decodes" data

let raising_write fmt =
  test fmt.name (fun () ->
      let f = fmt.writes ~eod:true (Bytes.Writer.ignore ()) in
      equal ~msg:"a bad first byte" (option Lz_support.error)
        (Some (decode_error fmt bad))
        (refused (fun () -> Bytes.Writer.write_string f bad));
      let cut, e = half fmt in
      let f = fmt.writes ~eod:true (Bytes.Writer.ignore ()) in
      equal ~msg:"the cut stream" (option Lz_support.error) None
        (refused (fun () -> Bytes.Writer.write_string f cut));
      equal ~msg:"its eod" (option Lz_support.error) (Some e)
        (refused (fun () -> Bytes.Writer.write_eod f)))

let error_again fmt =
  test fmt.name (fun () ->
      let cut, e = half fmt in
      let d = fmt.reads (Bytes.Reader.of_string ~slice_length:7 cut) in
      let read () = Bytes.Reader.discard d in
      let suffix = message "reader" e.at e in
      ends_with ~msg:"read" ~suffix (raised read);
      ends_with ~msg:"read again" ~suffix (raised read);
      let e = decode_error fmt bad in
      let suffix = message "writer" e.at e in
      let f = fmt.writes ~eod:true (Bytes.Writer.ignore ()) in
      ends_with ~msg:"written" ~suffix
        (raised (fun () -> Bytes.Writer.write_string f bad));
      ends_with ~msg:"written again" ~suffix
        (raised (fun () -> Bytes.Writer.write_string f "more"));
      ends_with ~msg:"ended" ~suffix
        (raised (fun () -> Bytes.Writer.write_eod f)))

(* A writer whose first write raises: the exception propagates from the filter,
   and every later write of the filter raises it again, though [w] would now
   take the bytes. The bytes are written without their end, since bytesrw
   refuses a slice after it: a decompressing filter writes a golden's content,
   and a compressing one 1 MiB of random bytes, past what an encoder keeps
   before it writes. *)
exception Disk_full

let raised_again fmt =
  test fmt.name (fun () ->
      let check name (filter : Bytes.Writer.filter) data =
        let writes = ref 0 in
        let w =
          Bytes.Writer.make ~slice_length:64 (fun s ->
              if not (Bytes.Slice.is_eod s) then begin
                incr writes;
                if !writes = 1 then raise Disk_full
              end)
        in
        let f = filter ~eod:true w in
        raises ~msg:name Disk_full (fun () -> Bytes.Writer.write_string f data);
        raises ~msg:(name ^ ", written again") Disk_full (fun () ->
            Bytes.Writer.write_string f "more");
        raises ~msg:(name ^ ", ended again") Disk_full (fun () ->
            Bytes.Writer.write_eod f)
      in
      check "decompress_writes" fmt.writes (data (text_100 fmt));
      check "compress_writes" (fmt.compress_writes ())
        (Lz_support.corpus Random (1024 * kib)))

(* Formats *)

let goldens fmt =
  cases fmt.name
    ~id:(fun g -> g.Lz_support.file)
    (Lz_support.goldens_of fmt.name)
    (fun g ->
      let data = data g and raw = raw g in
      lengths
      |> List.iter (fun n ->
          let msg = Printf.sprintf "slices of %d bytes" n in
          equal ~msg Lz_support.decoded (Ok raw)
            (fst (read_through fmt.reads ~cut:n ~most:n data));
          equal ~msg Lz_support.decoded (Ok raw)
            (write_through fmt.writes ~cut:n ~most:n data)))

(* Any golden up to 64 KiB, mutated or not, at any slice lengths: the filters
   agree with [decode], except that a Zstandard filter refuses a frame whose
   window passes its bound, as its decoder does, where [decode] reads it. *)
let mutations fmt =
  let files =
    List.filter
      (fun g -> g.Lz_support.size <= 64 * kib)
      (Lz_support.goldens_of fmt.name)
  in
  let gen =
    Gen.(
      quad
        (of_list ~pp:pp_golden files)
        (option (pair Lz_support.anywhere (int_range 0 255)))
        length length)
  in
  (* Slices and slice lengths of one byte, which drawn cases reach one time in
     eight or nine, and the first and last bytes changed. For Zstandard, a
     Window_Descriptor (byte 5 of a frame without a single segment) changed to
     0x68, a window of 8 MiB (RFC 8878 3.1.1.1.2: 2^(10 + 13)). *)
  let g = text_100 fmt in
  let examples =
    [ (g, None, 1, 1); (g, Some (0, 7), 7, 7); (g, Some (-1, 7), 7, 7) ]
    @
    if fmt.name = "zstd" then
      [ (golden "text_100.zst3-nosize", Some (5, 0x68), 7, 7) ]
    else []
  in
  prop fmt.name ~examples gen (fun (g, mutation, cut, most) ->
      let data =
        match mutation with
        | None -> data g
        | Some (i, c) ->
            let b = Bytes.of_string (data g) in
            if Bytes.length b > 0 then
              Bytes.set b (Lz_support.position i (Bytes.length b)) (Char.chr c);
            Bytes.to_string b
      in
      let room = (2 * String.length (raw g)) + (64 * kib) in
      let decoded = decode fmt ~room data in
      (match decoded with Error { why = Too_long; _ } -> reject () | _ -> ());
      let declined =
        if fmt.name <> "zstd" then None
        else
          match
            Lz_support.stepped (Lz.Zstd.decoder ~window:zstd_window ()) data
          with
          | Error { why = Unsupported _; _ } as refused when refused <> decoded
            ->
              Some refused
          | _ -> None
      in
      let expected = Option.value declined ~default:decoded in
      let got, r = read_through fmt.reads ~cut ~most data in
      equal ~msg:"read" Lz_support.decoded (as_read expected) got;
      (match expected with
      | Ok (_, consumed) ->
          equal ~msg:"left in the reader" string
            (String.sub data consumed (String.length data - consumed))
            (Bytes.Reader.to_string r)
      | Error _ -> ());
      equal ~msg:"written" Lz_support.decoded (as_written data expected)
        (write_through fmt.writes ~cut ~most data);
      let refused why =
        match expected with Error e -> why e.Lz.why | Ok _ -> false
      in
      cover "an input that decodes" (Result.is_ok expected);
      cover "a refused input" (Result.is_error expected);
      if fmt.name = "zstd" then
        cover "a window above the bound" (Option.is_some declined);
      classify "a malformed input"
        (refused (function Malformed _ -> true | _ -> false));
      classify "an input cut short" (refused (fun w -> w = Truncated));
      classify "bytes after a stream"
        (match expected with
        | Ok (_, consumed) -> consumed < String.length data
        | Error _ -> false);
      cover "a slice of one byte" (cut = 1);
      cover "a slice length of one byte" (most = 1))

let junk = "after the stream"

(* A deflate or zlib reader takes a slice of [r] only once its decoder has taken
   all of the last, and its decoder takes at most the stream's [n] bytes, so [r]
   serves at most [n + cut] bytes. A reader that reads a slice ahead is served
   the slice after the one holding byte [n - 1], past the bound unless [n] is a
   multiple of [cut]. The bytes are counted as [r] serves them, since the filter
   pushes back what its decoder did not take. Two slices follow the stream;
   slices of 1 MiB hold all of the longest stream [ends] gives. *)
let read_ahead fmt =
  cases fmt.name
    ~id:(fun g -> g.Lz_support.file)
    (ends fmt)
    (fun g ->
      let data = data g in
      let n = String.length data in
      lengths
      |> List.filter (fun cut -> cut <= 64 * kib)
      |> List.iter (fun cut ->
          let after = String.make (2 * cut) 'x' in
          let served = ref 0 in
          let r =
            Bytes.Reader.tap
              (fun s -> served := !served + Bytes.Slice.length s)
              (Bytes.Reader.of_string ~slice_length:cut (data ^ after))
          in
          Bytes.Reader.discard (fmt.reads r);
          at_most
            ~msg:(Printf.sprintf "slices of %d bytes" cut)
            int ~than:(n + cut) !served))

let left_after fmt =
  cases fmt.name
    ~id:(fun g -> g.Lz_support.file)
    (ends fmt)
    (fun g ->
      let data = data g and raw = raw g in
      lengths
      |> List.iter (fun cut ->
          let msg = Printf.sprintf "slices of %d bytes" cut in
          let got, r = read_through fmt.reads ~cut ~most:cut (data ^ junk) in
          equal ~msg Lz_support.decoded (Ok raw) got;
          equal ~msg int (String.length data) (Bytes.Reader.pos r);
          equal ~msg string junk (Bytes.Reader.to_string r)))

let written_after fmt =
  cases fmt.name
    ~id:(fun g -> g.Lz_support.file)
    (ends fmt)
    (fun g ->
      let data = data g in
      lengths
      |> List.iter (fun cut ->
          equal
            ~msg:(Printf.sprintf "slices of %d bytes" cut)
            Lz_support.decoded
            (Error (after_stream (String.length data)))
            (write_through fmt.writes ~cut ~most:cut (data ^ junk))))

(* [decode]'s error is at the first byte where no member begins (lz.mli), and
   the filters refuse as [decode] does. *)
let no_member fmt =
  cases fmt.name
    ~id:(fun g -> g.Lz_support.file)
    (ends fmt)
    (fun g ->
      let data = data g and raw = raw g in
      let expected =
        as_read (decode fmt ~room:(String.length raw) (data ^ junk))
      in
      (match expected with
      | Error e -> equal ~msg:"decode's offset" int (String.length data) e.at
      | Ok _ -> failf "decode took the bytes after %s" g.file);
      lengths
      |> List.iter (fun cut ->
          let msg = Printf.sprintf "slices of %d bytes" cut in
          equal ~msg Lz_support.decoded expected
            (fst (read_through fmt.reads ~cut ~most:cut (data ^ junk)));
          equal ~msg Lz_support.decoded expected
            (write_through fmt.writes ~cut ~most:cut (data ^ junk))))

(* [encoded fmt ~level s] is the stream of [s] from one step of [fmt]'s encoder
   with ~last:true, and the empty steps after it. *)
let encoded fmt ~level s =
  let e = fmt.encoder ~level () in
  let b = Buffer.create (String.length s) in
  let f run = Buffer.add_string b (Lz_support.to_string run) in
  let rec go src =
    match Lz.Encoder.step e ~last:true src f with
    | Done _ -> ()
    | More k -> go (Bigarray.Array1.sub src k (Bigarray.Array1.dim src - k))
  in
  go (Lz_support.of_string s);
  Buffer.contents b

(* [spread n k] adds [n / most_slices] bytes to a slice length [k], so that a
   filter over [n] bytes takes at most [most_slices] slices, and drawn lengths
   stay distinct. Slices are unchanged on inputs below 4 KiB, more than half the
   drawn ones; over the longest, slices of a few bytes took up to a million a
   case and outran the sanitizers' time. *)
let most_slices = 4096
let spread n k = k + (n / most_slices)

(* The input's slice length, one byte drawn more often than [length] draws it:
   [spread] keeps it only below 4 KiB. *)
let cut_length =
  Gen.(frequency [ (1, constant ~pp:Format.pp_print_int 1); (5, length) ])

(* [recipe_length r] is the length of [Lz_support.build r], without building
   it. *)
let rec recipe_length : Lz_support.recipe -> int = function
  | Empty -> 0
  | Byte _ -> 1
  | Run (_, n) | Noise (_, n) | Corpus (_, n) -> n
  | Match { distance; length; seed = _ } -> distance + length + 8
  | Concat rs -> List.fold_left (fun n r -> n + recipe_length r) 0 rs

(* An input, a level, and the slice lengths of the input and of the filter. *)
let compressed fmt =
  let lo, hi = fmt.levels in
  let pp ppf (recipe, level, cut, most) =
    Format.fprintf ppf "%a, level %d, slices of %d, slice length %d"
      Lz_support.pp_recipe recipe level cut most
  in
  let spread_slices (recipe, level, cut, most) =
    let n = recipe_length recipe in
    (recipe, level, spread n cut, spread n most)
  in
  let gen =
    Gen.(
      with_pp pp
        (map spread_slices
           (quad Lz_support.recipe (int_range lo hi) cut_length length)))
  in
  (* Each kind of input once, the empty one and one past 64 KiB among them, in
     slices of one byte where that stays below [most_slices], and 4 KiB of text
     at the highest level. *)
  let examples =
    List.map (fun r -> spread_slices (r, lo, 1, 1)) Lz_support.every_kind
    @ [ (Corpus (Text, 4 * kib), hi, 7, 7) ]
  in
  (gen, examples)

let covers fmt (s, level, cut) =
  let lo, hi = fmt.levels in
  cover "the empty input" (s = "");
  cover "an input above 64 KiB" (String.length s > 64 * kib);
  cover "a slice of one byte" (cut = 1);
  cover "the lowest level" (level = lo);
  cover "the highest level" (level = hi)

let encoder_stream fmt =
  let gen, examples = compressed fmt in
  prop fmt.name ~examples gen (fun (recipe, level, cut, most) ->
      let s = Lz_support.build recipe in
      let stream = encoded fmt ~level s in
      equal ~msg:"compress_reads" Lz_support.decoded (Ok stream)
        (fst (read_through (fmt.compress_reads ~level ()) ~cut ~most s));
      equal ~msg:"compress_writes" Lz_support.decoded (Ok stream)
        (write_through (fmt.compress_writes ~level ()) ~cut ~most s);
      covers fmt (s, level, cut))

let read_back fmt =
  let gen, examples = compressed fmt in
  prop fmt.name ~examples gen (fun (recipe, level, cut, most) ->
      let s = Lz_support.build recipe in
      let stream =
        require_ok ~pp:Lz.pp_error
          (fst (read_through (fmt.compress_reads ~level ()) ~cut ~most s))
      in
      equal ~msg:"decompress_reads" Lz_support.decoded (Ok s)
        (fst (read_through fmt.reads ~cut ~most stream));
      equal ~msg:"decompress_writes" Lz_support.decoded (Ok s)
        (write_through fmt.writes ~cut ~most stream);
      covers fmt (s, level, cut))

(* Zstd *)

let window_refused =
  test "a frame whose window passes ~window is refused as by Lz.Zstd.decoder"
    (fun () ->
      let g = golden "text_65536.zst3" and window = kib in
      let expected =
        Result.map fst
          (Lz_support.stepped (Lz.Zstd.decoder ~window ()) (data g))
      in
      (match expected with
      | Error { why = Unsupported _; _ } -> ()
      | _ -> failf "the decoder reads %s in a window of 1 KiB" g.file);
      let reads = Lz_bytesrw.Zstd.decompress_reads ~window () in
      let writes = Lz_bytesrw.Zstd.decompress_writes ~window () in
      equal ~msg:"read" Lz_support.decoded expected
        (fst (read_through reads ~cut:64 ~most:64 (data g)));
      equal ~msg:"written" Lz_support.decoded expected
        (write_through writes ~cut:64 ~most:64 (data g)))

(* A frame of "hello" (RFC 8878 3.1.1): a Frame_Header_Descriptor of 0x80, a
   4-byte Frame_Content_Size and no checksum, the [window_descriptor] byte, and
   one raw last block of 5 bytes. A Window_Descriptor of 0x88, exponent 17 and
   mantissa 0, declares a window of 128 MiB, and 0x89 the next size up, 144 MiB.
   The decoder allocates the content size, the smaller. *)
let hello window_descriptor =
  "\x28\xb5\x2f\xfd\x80"
  ^ String.make 1 window_descriptor
  ^ "\x05\x00\x00\x00" ^ "\x29\x00\x00" ^ "hello"

let default_window =
  test "the default window is 128 MiB" (fun () ->
      let reads = Lz_bytesrw.Zstd.decompress_reads () in
      let writes = Lz_bytesrw.Zstd.decompress_writes () in
      let read data = fst (read_through reads ~cut:64 ~most:64 data) in
      let written data = write_through writes ~cut:64 ~most:64 data in
      equal ~msg:"128 MiB, read" Lz_support.decoded (Ok "hello")
        (read (hello '\x88'));
      equal ~msg:"128 MiB, written" Lz_support.decoded (Ok "hello")
        (written (hello '\x88'));
      (* Refused at the Window_Descriptor, byte 5. *)
      let refused = Error { Lz.at = 5; why = Unsupported "" } in
      equal ~msg:"144 MiB, read"
        (result string Lz_support.refusal)
        refused
        (read (hello '\x89'));
      equal ~msg:"144 MiB, written"
        (result string Lz_support.refusal)
        refused
        (written (hello '\x89'));
      let names_field name got =
        let why =
          require_match ~msg:name
            (function
              | Error { Lz.why = Unsupported m; _ } -> Some m | _ -> None)
            got
        in
        contains
          ~msg:(name ^ ", the message names its field")
          ~sub:"Window_Size" why
      in
      names_field "144 MiB, read" (read (hello '\x89'));
      names_field "144 MiB, written" (written (hello '\x89')))

let each name law fmts = group name (List.map law fmts)

let () =
  exit
    (run "Lz_bytesrw"
       [
         each "the position starts at ~pos, 0 by default, and counts the bytes"
           positions formats;
         each "a reader's slice length is r's by default" reader_slice_length
           formats;
         each "a writer's slice length hint is w's by default"
           writer_slice_length formats;
         each
           "a writer writes eod on w after the stream's end only with ~eod:true"
           ends_with_eod formats;
         each "a level out of range raises Invalid_argument when applied" levels
           formats;
         small_window;
         each
           "a stream cut short raises Truncated at its length, after a prefix \
            of its decoded bytes"
           cut formats;
         each "a reader's error is at its byte's position in r's positions"
           reader_position formats;
         each "a writer's error is at its byte's position in its own positions"
           writer_position formats;
         each
           "a writer raises at the write of the bytes that hold the error, and \
            Truncated at eod"
           raising_write formats;
         each "a later read or write raises the error again" error_again formats;
         each "an exception from w propagates, and is raised again" raised_again
           formats;
         each
           "a decompressing filter yields each golden's decoded bytes at any \
            slice lengths"
           goldens formats;
         each "a decompressing filter reads and refuses what decode does"
           mutations formats;
         each "a reader is at most one slice past the stream's end" read_ahead
           [ deflate; zlib ];
         each "a reader leaves r at the first byte after the stream" left_after
           [ deflate; zlib ];
         each "a writer refuses bytes after the stream as Malformed"
           written_after [ deflate; zlib ];
         each "bytes that begin no member are refused as decode refuses them"
           no_member [ gzip; zstd ];
         each
           "a compressing filter writes its encoder's stream whatever the \
            slices"
           encoder_stream formats;
         each "a decompressing filter reads back a compressing filter's stream"
           read_back formats;
         group "Zstd" [ window_refused; default_window ];
       ])
