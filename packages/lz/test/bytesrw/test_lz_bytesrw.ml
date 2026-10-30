(* Lz_bytesrw's filters against Lz's whole-buffer calls and steps. A
   decompressing reader or writer yields the bytes [decode] yields and refuses
   what it refuses, at the same offset, whatever the slice lengths on either
   side; a deflate or zlib reader leaves its reader after the stream; a
   compressing filter writes the stream its encoder writes, which decodes to the
   input through the filters and through [decode]. *)

open Windtrap
open Bytesrw

let kib = 1024

(* Formats *)

type format = {
  name : string;
  members : bool; (* Read to the input's end, as gzip and Zstandard. *)
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

let formats =
  [
    {
      name = "deflate";
      members = false;
      decode = Lz.Deflate.decode;
      reads = Lz_bytesrw.Deflate.decompress_reads ();
      writes = Lz_bytesrw.Deflate.decompress_writes ();
      compress_reads = Lz_bytesrw.Deflate.compress_reads;
      compress_writes = Lz_bytesrw.Deflate.compress_writes;
      encoder = Lz.Deflate.encoder;
      levels = (1, 9);
    };
    {
      name = "zlib";
      members = false;
      decode = Lz.Zlib.decode;
      reads = Lz_bytesrw.Zlib.decompress_reads ();
      writes = Lz_bytesrw.Zlib.decompress_writes ();
      compress_reads = Lz_bytesrw.Zlib.compress_reads;
      compress_writes = Lz_bytesrw.Zlib.compress_writes;
      encoder = Lz.Zlib.encoder;
      levels = (1, 9);
    };
    {
      name = "gzip";
      members = true;
      decode = Lz.Gzip.decode;
      reads = Lz_bytesrw.Gzip.decompress_reads ();
      writes = Lz_bytesrw.Gzip.decompress_writes ();
      compress_reads = Lz_bytesrw.Gzip.compress_reads;
      compress_writes = Lz_bytesrw.Gzip.compress_writes;
      encoder = Lz.Gzip.encoder;
      levels = (1, 9);
    };
    {
      name = "zstd";
      members = true;
      decode = Lz.Zstd.decode;
      reads = Lz_bytesrw.Zstd.decompress_reads ~window:zstd_window ();
      writes = Lz_bytesrw.Zstd.decompress_writes ~window:zstd_window ();
      compress_reads = Lz_bytesrw.Zstd.compress_reads;
      compress_writes = Lz_bytesrw.Zstd.compress_writes;
      encoder = Lz.Zstd.encoder;
      levels = (1, 12);
    };
  ]

let data g = Lazy.force g.Lz_support.data
let raw g = Lazy.force g.Lz_support.raw

let golden name =
  List.find (fun g -> g.Lz_support.file = name) (Lz_support.goldens ())

let pp_golden ppf g = Format.pp_print_string ppf g.Lz_support.file

(* Slice lengths: one byte, a few, odd, and up to past every golden. *)
let lengths = [ 1; 2; 3; 7; 64; 4 * kib; 64 * kib; 1024 * kib ]

let length =
  Gen.frequency
    [
      (3, Gen.int_range 1 8);
      (2, Gen.of_list lengths);
      (1, Gen.int_range 1 (64 * kib));
    ]

(* The error a filter raises after a stream's last byte, for a writer. *)
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
   through [filter] with slice length [most]. It is the bytes read or the error,
   and the reader of [data]. *)
let read_through (filter : Bytes.Reader.filter) ~cut ~most data =
  let r = Bytes.Reader.of_string ~slice_length:cut data in
  let d = filter ~slice_length:most r in
  let b = Buffer.create (String.length data) in
  let rec go () =
    let s = Bytes.Reader.read d in
    if Bytes.Slice.is_eod s then Buffer.contents b
    else begin
      check_length ~most s;
      add b s;
      go ()
    end
  in
  (caught go, r)

(* [write_through filter ~cut ~most data] writes [data], in slices of [cut]
   bytes, and its end on [filter] over a writer of slice length [most]. It is
   the bytes written or the error, and whether the end was written. *)
let write_through ?(eod = true) (filter : Bytes.Writer.filter) ~cut ~most data =
  let b = Buffer.create (String.length data) and ended = ref false in
  let w =
    Bytes.Writer.make ~slice_length:most (fun s ->
        if Bytes.Slice.is_eod s then ended := true
        else begin
          check_length ~most s;
          add b s
        end)
  in
  let f = filter ~slice_length:cut ~eod w in
  let got =
    caught (fun () ->
        Bytes.Writer.write_string f data;
        Bytes.Writer.write_eod f;
        Buffer.contents b)
  in
  (got, !ended)

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
let read_of = function Ok (out, _) -> Ok out | Error e -> Error e

let write_of data = function
  | Ok (_, consumed) when consumed < String.length data ->
      Error (after_stream consumed)
  | Ok (out, _) -> Ok out
  | Error e -> Error e

(* Decompression *)

let goldens fmt =
  cases
    ~name:(fun g -> g.Lz_support.file)
    fmt.name
    (Lz_support.goldens_of fmt.name)
    (fun g ->
      let data = data g and raw = raw g in
      lengths
      |> List.iter (fun n ->
          let msg = Printf.sprintf "slices of %d bytes" n in
          let got, r = read_through fmt.reads ~cut:n ~most:n data in
          equal ~msg Lz_support.decoded (Ok raw) got;
          equal ~msg string "" (Bytes.Reader.to_string r);
          let got, ended = write_through fmt.writes ~cut:n ~most:n data in
          equal ~msg Lz_support.decoded (Ok raw) got;
          equal ~msg bool true ended))

let junk = "after the stream"

(* A deflate or zlib reader stops at the stream's end and leaves the rest to its
   reader; a writer has nowhere to leave it. Gzip and Zstandard refuse the first
   byte that begins no member. *)
let after fmt =
  cases
    ~name:(fun g -> g.Lz_support.file)
    fmt.name
    (Lz_support.goldens_of fmt.name)
    (fun g ->
      let data = data g and raw = raw g in
      let n = String.length data in
      let expected = decode fmt ~room:(String.length raw) (data ^ junk) in
      (match (fmt.members, expected) with
      | true, Error e -> equal ~msg:"decode's offset" int n e.at
      | true, Ok _ -> failf "decode took %S after %s" junk g.file
      | false, _ ->
          equal ~msg:"decode"
            (result (pair string int) Lz_support.error)
            (Ok (raw, n))
            expected);
      lengths
      |> List.iter (fun cut ->
          let msg = Printf.sprintf "slices of %d bytes" cut in
          let got, r = read_through fmt.reads ~cut ~most:cut (data ^ junk) in
          equal ~msg Lz_support.decoded (read_of expected) got;
          if not fmt.members then begin
            equal ~msg int n (Bytes.Reader.pos r);
            equal ~msg string junk (Bytes.Reader.to_string r)
          end;
          let got, _ = write_through fmt.writes ~cut ~most:cut (data ^ junk) in
          equal ~msg Lz_support.decoded (write_of (data ^ junk) expected) got))

(* Every proper prefix of a golden of one member, up to 1 KiB, is cut short, at
   its length. *)
let cut fmt =
  let files =
    List.filter
      (fun g -> g.Lz_support.members = 1 && String.length (data g) <= kib)
      (Lz_support.goldens_of fmt.name)
  in
  let lengths = Array.of_list lengths in
  cases
    ~name:(fun g -> g.Lz_support.file)
    fmt.name files
    (fun g ->
      let data = data g in
      for k = 0 to String.length data - 1 do
        let prefix = String.sub data 0 k in
        let cut = lengths.(k mod Array.length lengths) in
        let msg = Printf.sprintf "cut at %d, slices of %d bytes" k cut in
        let expected = Error { Lz.at = k; why = Truncated } in
        equal ~msg Lz_support.decoded expected
          (fst (read_through fmt.reads ~cut ~most:cut prefix));
        equal ~msg Lz_support.decoded expected
          (fst (write_through fmt.writes ~cut ~most:cut prefix))
      done)

let unsupported = function
  | Error { Lz.why = Unsupported _; _ } -> true
  | _ -> false

(* Any golden up to 64 KiB, mutated or not, at any slice lengths: the filters
   agree with [decode], except on a frame whose window [decode] reads and their
   decoder's 4 MiB bound declines. *)
let mutations fmt =
  let files =
    List.filter
      (fun g -> String.length (data g) <= 64 * kib)
      (Lz_support.goldens_of fmt.name)
  in
  let gen =
    Gen.(
      quad
        (of_list ~pp:pp_golden files)
        (option (pair nat (int_range 0 255)))
        length length)
  in
  prop fmt.name gen (fun (g, mutation, cut, most) ->
      let data =
        match mutation with
        | None -> data g
        | Some (i, c) ->
            let b = Bytes.of_string (data g) in
            if Bytes.length b > 0 then
              Bytes.set b (i mod Bytes.length b) (Char.chr c);
            Bytes.to_string b
      in
      let room = (2 * String.length (raw g)) + (64 * kib) in
      let decoded = decode fmt ~room data in
      (match decoded with Error { why = Too_long; _ } -> reject () | _ -> ());
      let got, r = read_through fmt.reads ~cut ~most data in
      let declined =
        fmt.name = "zstd" && unsupported got && read_of decoded <> got
      in
      let expected =
        if declined then
          Lz_support.stepped (Lz.Zstd.decoder ~window:zstd_window ()) data
        else decoded
      in
      classify "a window above the bound" declined;
      equal ~msg:"read" Lz_support.decoded (read_of expected) got;
      (match expected with
      | Ok (_, consumed) when not fmt.members ->
          equal ~msg:"left in the reader" string
            (String.sub data consumed (String.length data - consumed))
            (Bytes.Reader.to_string r)
      | _ -> ());
      let got, _ = write_through fmt.writes ~cut ~most data in
      equal ~msg:"written" Lz_support.decoded (write_of data expected) got;
      let refused why =
        match expected with Error e -> why e.Lz.why | Ok _ -> false
      in
      cover "an input that decodes" (Result.is_ok expected);
      cover "a refused input" (Result.is_error expected);
      classify "a malformed input"
        (refused (function Malformed _ -> true | _ -> false));
      classify "an input cut short" (refused (fun w -> w = Truncated));
      classify "bytes after a stream"
        (match expected with
        | Ok (_, consumed) -> consumed < String.length data
        | Error _ -> false);
      cover "a slice of one byte" (cut = 1);
      cover "a slice length of one byte" (most = 1))

(* The message of the stream error [f ()] raises. *)
let raised f =
  match f () with
  | _ -> failf "no error"
  | exception Bytes.Stream.Error err -> Bytes.Stream.error_message err

(* A golden of 100 bytes of text in each format. *)
let text_100 = function
  | "deflate" -> golden "text_100.deflate6"
  | "zlib" -> golden "text_100.zlib6"
  | "gzip" -> golden "text_100.gz9"
  | _ -> golden "text_100.zst3"

(* An error's position is its byte's, in the reader's positions or the filtering
   writer's, and every later read or write raises it again. *)
let positions fmt =
  test fmt.name (fun () ->
      let skipped = 3 and start = 5 in
      let line kind pos e =
        Format.asprintf "%s %s:%d: %a" fmt.name kind pos Lz.pp_error e
      in
      (* Cut short: refused at the end of data, at the cut. *)
      let g = text_100 fmt.name in
      let k = String.length (data g) / 2 in
      let cut = String.sub (data g) 0 k in
      let e = { Lz.at = k; why = Truncated } in
      let r = Bytes.Reader.of_string ~slice_length:7 ("xyz" ^ cut) in
      Bytes.Reader.skip skipped r;
      let d = fmt.reads r in
      let read () = Bytes.Reader.discard d in
      equal string (line "reader" (skipped + k) e) (raised read);
      equal ~msg:"read again" string
        (line "reader" (skipped + k) e)
        (raised read);
      let f = fmt.writes ~pos:start ~eod:true (Bytes.Writer.ignore ()) in
      equal string
        (line "writer" (start + k) e)
        (raised (fun () ->
             Bytes.Writer.write_string f cut;
             Bytes.Writer.write_eod f));
      (* A first byte no stream begins with, refused on the first write. *)
      let bad = "\x06\x00\x00\x00" in
      let e =
        match decode fmt ~room:kib bad with
        | Error e -> e
        | Ok _ -> failf "%S decodes" bad
      in
      let f = fmt.writes ~pos:start ~eod:true (Bytes.Writer.ignore ()) in
      let write s () = Bytes.Writer.write_string f s in
      equal string (line "writer" (start + e.at) e) (raised (write bad));
      equal ~msg:"written again" string
        (line "writer" (start + e.at) e)
        (raised (write "more")))

(* A writer whose first write raises: the exception propagates from the filter,
   and every later write of the filter raises it again, though [w] would now
   take the bytes. The bytes are written without their end, since bytesrw
   refuses a slice after it: a decompressing filter writes a golden's content,
   and a compressing one 1 MiB of random bytes, past what an encoder holds
   before it writes. *)
exception Disk_full

let random_mib =
  let st = Random.State.make [| 1 |] in
  String.init (1024 * kib) (fun _ -> Char.chr (Random.State.int st 256))

let raised_again fmt =
  let g = text_100 fmt.name in
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
  test fmt.name (fun () ->
      check "decompress_writes" fmt.writes (data g);
      check "compress_writes" (fmt.compress_writes ()) random_mib)

(* A writer given ~eod:false leaves its writer open. *)
let open_writer fmt =
  test fmt.name (fun () ->
      let g =
        List.hd
          (List.filter (fun g -> raw g <> "") (Lz_support.goldens_of fmt.name))
      in
      let b = Buffer.create 0 in
      let w = Bytes.Writer.of_buffer b in
      let f = fmt.writes ~eod:false w in
      Bytes.Writer.write_string f (data g);
      Bytes.Writer.write_eod f;
      Bytes.Writer.write_string w "tail";
      Bytes.Writer.write_eod w;
      equal Lz_support.decoded (Ok (raw g ^ "tail")) (Ok (Buffer.contents b));
      let got, ended =
        write_through ~eod:false fmt.writes ~cut:64 ~most:64 (data g)
      in
      equal Lz_support.decoded (Ok (raw g)) got;
      equal ~msg:"eod written" bool false ended)

(* Zstandard's window bound *)

let window =
  let g = golden "text_65536.zst3" in
  let window = kib in
  let stepped = Lz_support.stepped (Lz.Zstd.decoder ~window ()) (data g) in
  let expected = match stepped with Ok (s, _) -> Ok s | Error e -> Error e in
  group "Zstandard's window"
    [
      test "a frame above the window is refused as by the decoder" (fun () ->
          if not (unsupported expected) then failf "%s fits in 1 KiB" g.file;
          let reads = Lz_bytesrw.Zstd.decompress_reads ~window () in
          let writes = Lz_bytesrw.Zstd.decompress_writes ~window () in
          equal Lz_support.decoded expected
            (fst (read_through reads ~cut:64 ~most:64 (data g)));
          equal Lz_support.decoded expected
            (fst (write_through writes ~cut:64 ~most:64 (data g))));
      test "the default window reads every golden" (fun () ->
          Lz_support.goldens_of "zstd"
          |> List.iter (fun g ->
              let reads = Lz_bytesrw.Zstd.decompress_reads () in
              equal ~msg:g.Lz_support.file Lz_support.decoded
                (Ok (raw g))
                (fst (read_through reads ~cut:kib ~most:(64 * kib) (data g)))));
      test "a window below 1 KiB raises when applied" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Lz_bytesrw.Zstd.decompress_reads ~window:1023 ()
                (Bytes.Reader.empty ()));
          raises_match Exn.invalid_arg (fun () ->
              Lz_bytesrw.Zstd.decompress_writes ~window:1023 () ~eod:true
                (Bytes.Writer.ignore ())));
    ]

(* Compression *)

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

let round_trip fmt =
  let lo, hi = fmt.levels in
  let gen = Gen.(quad Lz_support.recipe (int_range lo hi) length length) in
  prop fmt.name gen (fun (recipe, level, cut, most) ->
      let s = Lz_support.build recipe in
      let stream = encoded fmt ~level s in
      let got, _ = read_through (fmt.compress_reads ~level ()) ~cut ~most s in
      equal ~msg:"read" Lz_support.decoded (Ok stream) got;
      let got, ended =
        write_through (fmt.compress_writes ~level ()) ~cut ~most s
      in
      equal ~msg:"written" Lz_support.decoded (Ok stream) got;
      equal ~msg:"eod written" bool true ended;
      equal ~msg:"decode"
        (result (pair string int) Lz_support.error)
        (Ok (s, String.length stream))
        (decode fmt ~room:(String.length s) stream);
      equal ~msg:"decompress_reads" Lz_support.decoded (Ok s)
        (fst (read_through fmt.reads ~cut ~most stream));
      equal ~msg:"decompress_writes" Lz_support.decoded (Ok s)
        (fst (write_through fmt.writes ~cut ~most stream));
      cover "the empty input" (s = "");
      cover "an input above 64 KiB" (String.length s > 64 * kib);
      cover "a slice of one byte" (cut = 1))

let levels fmt =
  let lo, hi = fmt.levels in
  cases ~name:string_of_int fmt.name
    [ lo - 1; hi + 1 ]
    (fun level ->
      raises_match Exn.invalid_arg (fun () ->
          fmt.compress_reads ~level () (Bytes.Reader.empty ()));
      raises_match Exn.invalid_arg (fun () ->
          fmt.compress_writes ~level () ~eod:true (Bytes.Writer.ignore ())))

let each name law = group name (List.map law formats)

let () =
  exit
    (run "Lz_bytesrw"
       [
         each "decompressing filters read the goldens" goldens;
         each "bytes after the stream" after;
         each "a cut stream is Truncated at its length" cut;
         each "filters agree with decode" mutations;
         each "an error's position, raised again" positions;
         each "a raise from the writer, raised again" raised_again;
         each "a writer with ~eod:false leaves its writer open" open_writer;
         window;
         group ~tags:[ "compress" ] "compression"
           [
             each "a stream decodes to its input" round_trip;
             each "a level out of range raises when applied" levels;
           ];
       ])
