(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* FLAC. Cuts of the IETF test files load as FFmpeg decodes them: the RFC 9639
   MD5 of the samples a load returns, recomputed here, is the one FFmpeg's
   samples give; streams built to exercise each feature RFC 9639 defines load as
   the samples they were built from, each in the top bits of its dtype; every
   file the RFC forbids raises [Failure] naming the file, and nothing else. *)

open Windtrap
open Io_support
module Audio = Nx_io.Audio

let cut name = Filename.concat "cut" name
let decoded path = Audio.samples (Nx_io.load_audio path)

let facts a =
  ( (Audio.rate a, Audio.length a, Audio.channels a),
    (Audio.bits a, Audio.channel_mask a) )

let facts_t = pair (triple int int int) (pair int int)

(* The RFC 9639 MD5 (8.2) of samples a load returned at [bits]: each element
   shifted down to its [bits], little-endian in its bytes rounded up, the
   channels of a row in order. It fails if an element has a set bit below its
   top [bits]. *)
let rfc_md5 ~bits (Nx.P x) =
  let k = (bits + 7) / 8 in
  let out = Buffer.create (Nx.numel x * k) in
  let add w v =
    let v = v asr ((8 * w) - bits) in
    for i = 0 to k - 1 do
      Buffer.add_char out (Char.chr ((v asr (8 * i)) land 0xff))
    done
  in
  let low w v =
    if v land ((1 lsl ((8 * w) - bits)) - 1) <> 0 then
      failf "a sample %d has a set bit below its %d bits" v bits
  in
  let add w v =
    low w v;
    add w v
  in
  (match D.Any (Nx.dtype x) with
  | D.Any D.Int8 ->
      Array.iter (add 1) (Nx.to_array (Nx.unpack Nx.int8 (Nx.P x)))
  | D.Any D.Int16 ->
      Array.iter (add 2) (Nx.to_array (Nx.unpack Nx.int16 (Nx.P x)))
  | D.Any D.Int32 ->
      Array.iter
        (fun v -> add 4 (Int32.to_int v))
        (Nx.to_array (Nx.unpack Nx.int32 (Nx.P x)))
  | _ -> failf "a FLAC load returned %s" (D.name (Nx.dtype x)));
  Digest.to_hex (Digest.string (Buffer.contents out))

(* IETF cuts *)

(* A cut's facts: its file, rate, channels, bits, length, and the MD5 of
   FFmpeg's samples. *)
type ietf = {
  name : string;
  rate : int;
  channels : int;
  bits : int;
  length : int;
  md5 : string;
}

let ietf =
  match String.split_on_char '\n' (String.trim (read (cut "ietf.tsv"))) with
  | [] -> []
  | _ :: rows ->
      List.map
        (fun row ->
          match String.split_on_char '\t' row with
          | [ name; _; rate; channels; bits; length; _; _; _; md5 ] ->
              {
                name;
                rate = int_of_string rate;
                channels = int_of_string channels;
                bits = int_of_string bits;
                length = int_of_string length;
                md5;
              }
          | _ -> failwith ("ietf.tsv: " ^ row))
        rows

let dtype_of_bits bits =
  if bits <= 8 then "int8" else if bits <= 16 then "int16" else "int32"

let ietf_cuts =
  cases "cuts of the IETF test files load as FFmpeg decodes them"
    ~id:(fun c -> c.name)
    ietf
    (fun c ->
      let a = Nx_io.load_audio (cut c.name) in
      equal (triple int int int)
        (c.rate, c.channels, c.bits)
        (Audio.rate a, Audio.channels a, Audio.bits a);
      equal int c.length (Audio.length a);
      let (Nx.P x as p) = Audio.samples a in
      equal string (dtype_of_bits c.bits) (D.name (Nx.dtype x));
      equal (array int) [| c.length; c.channels |] (Nx.shape x);
      equal string c.md5 (rfc_md5 ~bits:c.bits p))

(* Built streams *)

(* A built stream's facts, as {!ietf}'s, and its channel mask if it states
   one. *)
let built =
  match String.split_on_char '\n' (String.trim (read (fixture "flac.tsv"))) with
  | [] -> []
  | _ :: rows ->
      List.map
        (fun row ->
          match String.split_on_char '\t' row with
          | [ name; rate; channels; bits; length; mask; md5 ] ->
              ( {
                  name;
                  rate = int_of_string rate;
                  channels = int_of_string channels;
                  bits = int_of_string bits;
                  length = int_of_string length;
                  md5;
                },
                if mask = "" then None else Some (int_of_string mask) )
          | _ -> failwith ("flac.tsv: " ^ row))
        rows

(* The speakers of a stream of [c] channels that states none. *)
let default_mask c =
  [| 0x4; 0x3; 0x7; 0x33; 0x607; 0x60F; 0x70F; 0x63F |].(c - 1)

let built_streams =
  cases "built streams load as the samples they were built from"
    ~id:(fun (c, _) -> c.name)
    built
    (fun (c, mask) ->
      let a = Nx_io.load_audio (fixture c.name) in
      equal (triple int int int)
        (c.rate, c.channels, c.bits)
        (Audio.rate a, Audio.channels a, Audio.bits a);
      equal int c.length (Audio.length a);
      equal int
        (Option.value mask ~default:(default_mask c.channels))
        (Audio.channel_mask a);
      let (Nx.P x as p) = Audio.samples a in
      equal string (dtype_of_bits c.bits) (D.name (Nx.dtype x));
      equal (array int) [| c.length; c.channels |] (Nx.shape x);
      equal string c.md5 (rfc_md5 ~bits:c.bits p))

(* Refusals *)

(* [with_md5 s md5] is the file [s] with its STREAMINFO's MD5 [md5]: the last 16
   of its 34 bytes, which follow the marker and the block's header. *)
let with_md5 s md5 =
  String.sub s 0 26 ^ md5 ^ String.sub s 42 (String.length s - 42)

let ietf_refused =
  cases "IETF files the RFC forbids raise naming the file" ~id:fst
    [
      ("ietf_faulty_01.flac", "longer than STREAMINFO's longest");
      ("ietf_faulty_02.flac", "outside STREAMINFO's sizes");
      ("ietf_faulty_03.flac", "bit depth differs");
      ("ietf_faulty_04.flac", "channels differ");
      ("ietf_faulty_05.flac", "runs past STREAMINFO's length");
      ("ietf_faulty_06.flac", "STREAMINFO is not the first");
      ("ietf_faulty_07.flac", "STREAMINFO is not the first");
      ("ietf_faulty_08.flac", "blocks of 0 to 0 rows");
      ("ietf_faulty_09.flac", "blocks of 1 to 1 rows");
      ("ietf_faulty_10.flac", "Vorbis comment is malformed");
      ("ietf_faulty_11.flac", "of type 127");
      ("ietf_uncommon_10.flac", "neither a WAV nor a FLAC");
      ("ietf_uncommon_11.flac", "neither a WAV nor a FLAC");
    ]
    (fun (name, naming) ->
      let path = cut name in
      fails ~naming ~path (fun () -> Audio.samples (Nx_io.load_audio path)))

let past_length =
  test "faulty 05 without its MD5 raises at the frame past its length"
    (fun () ->
      let s = read (cut "ietf_faulty_05.flac") in
      let path = file (with_md5 s (String.make 16 '\000')) in
      fails ~naming:"runs past STREAMINFO's length" ~path (fun () ->
          Audio.samples (Nx_io.load_audio path)))

let built_refused =
  cases "built streams the RFC forbids raise naming the fault" ~id:fst
    [
      ("bits_3", "3 bits, fewer than 4");
      ("block_65536", "forbidden 65536");
      ("block_past_end", "ends past the file");
      ("bound", "more than the frames'");
      ("bound_padding", "more than the frames'");
      ("channels_change", "channels differ");
      ("coded_number", "coded number is malformed");
      ("crc16", "CRC-16 differs");
      ("crc8", "CRC-8 differs");
      ("depth_change", "bit depth differs");
      ("ends_early", "the file ends before STREAMINFO's length");
      ("forbidden_rate_code", "sample rate code is forbidden");
      ("frame_size", "outside STREAMINFO's sizes");
      ("lpc_precision", "forbidden 0b1111");
      ("mask_hex", "no 32-bit hex number");
      ("mask_twice", "two channel masks");
      ("mask_wide", "no 32-bit hex number");
      ("min_above_max", "blocks of 32 to 16");
      ("negative_shift", "shift is negative");
      ("number", "coded number is not its number");
      ("order_past_block", "predictor order is past its block");
      ("padding_bits", "padding bits are not zero");
      ("partition_order", "does not divide its block");
      ("partition_short", "not longer than its predictor's order");
      ("past_length", "runs past STREAMINFO's length");
      ("rate_change", "sample rate differs");
      ("reserved_bit", "reserved bit is set");
      ("reserved_block_code", "block size code is reserved");
      ("reserved_channels", "channel code is reserved");
      ("reserved_depth_code", "bit depth code is reserved");
      ("residual_huge", "2^31 or more");
      ("residual_method", "coding method is reserved");
      ("residual_min", "2^31 or more");
      ("sample_past_depth", "past its depth");
      ("seektable_length", "not whole points of 18");
      ("seektable_order", "seek table is malformed");
      ("seektable_placeholder_first", "seek table is malformed");
      ("short_not_last", "shorter than STREAMINFO's shortest is not the last");
      ("short_not_last_unstated", "follows one shorter");
      ("side_range", "a sample is past its depth");
      ("strategy", "blocking strategy differs");
      ("streaminfo_33", "not 34 bytes");
      ("subframe_bit", "first bit is set");
      ("subframe_type", "type is reserved");
      ("trailing_unstated", "frame at row 32");
      ("two_comments", "two Vorbis comments");
      ("two_seektables", "two seek tables");
      ("two_streaminfo", "two STREAMINFO");
      ("wasted_all", "wasted bits leave it none");
    ]
    (fun (name, naming) ->
      let path = fixture ("flac_bad_" ^ name ^ ".flac") in
      fails ~naming ~path (fun () -> Audio.samples (Nx_io.load_audio path)))

(* Bytes *)

(* The byte of a FLAC file's first frame, after its metadata blocks. *)
let first_frame s =
  let rec go at =
    let last = Char.code s.[at] land 0x80 <> 0 in
    let n = (Char.code s.[at + 1] lsl 16) lor String.get_uint16_be s (at + 2) in
    if last then at + 4 + n else go (at + 4 + n)
  in
  go 4

let flip_bit s i b =
  String.mapi
    (fun j c -> if i = j then Char.chr (Char.code c lxor (1 lsl b)) else c)
    s

(* MD5 *)

let stated s = String.sub s 26 16

let flip_md5 s b =
  let m = stated s in
  with_md5 s
    (String.mapi
       (fun i c ->
         if i = b / 8 then Char.chr (Char.code c lxor (1 lsl (b mod 8))) else c)
       m)

let md5 =
  group "MD5"
    [
      test "a load checks the samples against STREAMINFO's MD5" (fun () ->
          let s = read (fixture "flac_depth_16.flac") in
          for b = 0 to 127 do
            let path = file (flip_md5 s b) in
            let a = Nx_io.load_audio path in
            fails ~naming:"MD5" ~path (fun () -> Audio.samples a)
          done);
      test "a load that counts the rows checks the MD5" (fun () ->
          let path = file (flip_md5 (read (fixture "flac_no_length.flac")) 0) in
          fails ~naming:"MD5" ~path (fun () -> Nx_io.load_audio path));
      test "an MD5 of zeros states none" (fun () ->
          let s = read (fixture "flac_depth_16.flac") in
          equal packed
            (decoded (fixture "flac_depth_16.flac"))
            (decoded (file (with_md5 s (String.make 16 '\000')))));
    ]

(* LibriSpeech *)

(* A LibriSpeech file, whose source and licence support/NOTICE states, and the
   MD5 of libsndfile's int16 reading of it, little-endian. *)
let librispeech = fixture "librispeech_7902-96594-0025.flac"
let librispeech_md5 = "572ee418fca9ad27597247a089821499"

let speech =
  test "a LibriSpeech file loads as libsndfile reads it" (fun () ->
      let a = Nx_io.load_audio librispeech in
      equal facts_t ((16000, 29120, 1), (16, 4)) (facts a);
      let x = Nx.unpack Nx.int16 (Audio.samples a) in
      let le = Bytes.create (2 * Nx.numel x) in
      Array.iteri (fun i v -> Bytes.set_int16_le le (2 * i) v) (Nx.to_array x);
      equal string librispeech_md5 (Digest.to_hex (Digest.bytes le));
      (* Each quotient by 32768 is exact, so it gives each sample back. *)
      let y = Nx.div_s (Nx.cast Nx.float32 x) 32768. in
      equal (array int) (Nx.to_array x)
        (Nx.to_array (Nx.cast Nx.int16 (Nx.mul_s y 32768.))))

(* Windows and blocks *)

(* Files to read in pieces: of fixed and variable block sizes, frames of 16 rows
   to 65535, with seek points confirmed and not. *)
let pieces =
  [
    fixture "flac_seek.flac";
    fixture "flac_seek_wrong.flac";
    fixture "flac_seek_variable.flac";
    fixture "flac_skewed.flac";
    fixture "flac_block_uncommon.flac";
    fixture "flac_pre_2007.flac";
    fixture "flac_depth_24.flac";
    fixture "flac_no_length.flac";
    cut "ietf_subset_03.flac";
    cut "ietf_subset_24.flac";
    cut "ietf_subset_48.flac";
    cut "ietf_subset_60.flac";
  ]

(* Each file's samples, loaded once. *)
(* Each file's rows, read from its header at start, and its samples, decoded
   once when a test first asks, so a decoder that fails fails that test. *)
let lengths =
  List.map
    (fun p -> (p, try Audio.length (Nx_io.load_audio p) with Failure _ -> 0))
    pieces

let whole =
  let memo = Hashtbl.create 16 in
  fun p ->
    match Hashtbl.find_opt memo p with
    | Some x -> x
    | None ->
        let x = decoded p in
        Hashtbl.add memo p x;
        x

let slice (Nx.P x) ~row ~rows = Nx.P (Nx.slice [ Nx.R (row, row + rows) ] x)

let join = function
  | [] -> failf "no block"
  | Nx.P x :: _ as bs ->
      let dt = Nx.dtype x in
      Nx.P (Nx.concatenate ~axis:0 (List.map (fun b -> Nx.unpack dt b) bs))

(* A file, and a row and a count of rows in it. *)
let piece =
  let open Gen in
  let* path, n =
    of_list ~pp:(fun ppf (p, _) -> Format.pp_print_string ppf p) lengths
  in
  let* row =
    frequency [ (1, constant 0); (1, constant n); (6, int_range 0 n) ]
  in
  let+ rows =
    frequency
      [
        (1, constant 0);
        (1, constant (n - row));
        (1, int_range (Int.min 1 (n - row)) (Int.min 3 (n - row)));
        (5, int_range 0 (n - row));
      ]
  in
  (path, n, row, rows)

let piece =
  Gen.with_pp
    (fun ppf (p, _, row, rows) ->
      Format.fprintf ppf "%s, %d rows from row %d" p rows row)
    piece

let windows =
  prop "a window is the slice of the samples it covers" piece
    (fun (path, n, row, rows) ->
      let x = whole path in
      cover "from row 0" (row = 0 && rows > 0);
      cover "no row" (rows = 0);
      cover "to the end" (rows > 0 && row + rows = n);
      cover "past a seek point" (row > 32 && path = fixture "flac_seek.flac");
      cover "in the denser frames" (row > 16384 && rows > 1024);
      equal packed (slice x ~row ~rows)
        (Audio.window (Nx_io.load_audio path) ~row ~rows))

(* Joined blocks are the samples from their first row. *)
let joined =
  prop "blocks joined are the samples from their row"
    Gen.(pair piece (int_range 1 64))
    (fun ((path, n, row, _), k) ->
      let (Nx.P y as x) = whole path in
      (* At most [k] blocks, and one past the end of the samples. *)
      let rows = if k = 64 then n + 1 else Int.max 1 ((n - row) / k) in
      cover "one row a block" (rows = 1 && row < n);
      cover "one block" (n - row > 0 && rows >= n - row);
      cover "from a row past 0" (row > 0 && row < n);
      cover "no block" (row = n);
      let bs = List.of_seq (Audio.blocks ~row (Nx_io.load_audio path) ~rows) in
      equal int ((n - row + rows - 1) / rows) (List.length bs);
      List.iteri
        (fun i (Nx.P b) ->
          let m =
            if i = List.length bs - 1 then n - row - (i * rows) else rows
          in
          equal (array int) [| m; (Nx.shape y).(1) |] (Nx.shape b))
        bs;
      if bs <> [] then equal packed (slice x ~row ~rows:(n - row)) (join bs))

(* The byte of the first seek table's type, if any. *)
let seektable_at s =
  let rec go at =
    let h = Char.code s.[at] in
    let n = (Char.code s.[at + 1] lsl 16) lor String.get_uint16_be s (at + 2) in
    if h land 0x7f = 3 then Some at
    else if h land 0x80 <> 0 then None
    else go (at + 4 + n)
  in
  go 4

(* [padded s] is [s] with its seek table a padding block. *)
let padded s =
  match seektable_at s with
  | None -> failf "no seek table"
  | Some at ->
      String.mapi
        (fun i c ->
          if i = at then Char.chr (Char.code c land 0x80 lor 1) else c)
        s

let seeking =
  group "seek points"
    [
      test "a seek table changes no window" (fun () ->
          List.iter
            (fun name ->
              let s = read name in
              let a = Nx_io.load_audio name
              and b = Nx_io.load_audio (file (padded s)) in
              let n = Audio.length a in
              for i = 0 to 63 do
                let row = i * n / 64 in
                let rows = Int.min 7 (n - row) in
                equal packed
                  (Audio.window a ~row ~rows)
                  (Audio.window b ~row ~rows)
              done)
            [
              fixture "flac_seek.flac";
              fixture "flac_seek_variable.flac";
              cut "ietf_subset_48.flac";
            ]);
      test "a read past a confirmed seek point decodes from it" (fun () ->
          (* Frame 0 of flac_seek.flac corrupted: a window past the point at row
             64 does not decode it, and one before every point but row 0's does,
             as does any window of the file without its table. *)
          let s = read (fixture "flac_seek.flac") in
          let bad = flip_byte s (first_frame s + 12) in
          let a = Nx_io.load_audio (file bad) in
          equal packed
            (slice (decoded (fixture "flac_seek.flac")) ~row:70 ~rows:10)
            (Audio.window a ~row:70 ~rows:10);
          let path = file bad in
          fails ~naming:"CRC-16" ~path (fun () ->
              Audio.window (Nx_io.load_audio path) ~row:10 ~rows:10);
          let path = file (padded bad) in
          fails ~naming:"CRC-16" ~path (fun () ->
              Audio.window (Nx_io.load_audio path) ~row:70 ~rows:10));
      test "a read that stops while it skips rows resumes the skip" (fun () ->
          (* flac_seek_dense.flac: 65536 rows of silence in constant frames,
             2048 dense rows in 8 verbatim frames of 1035 bytes, then 4096 rows
             of silence; 12232 bytes of frames for 71680 rows, so a row's mean
             is 0 bytes, taken as 1. Seek points at rows 0, 65536 and 67584. A
             window of 10 rows from row 67036 seeks to row 65536 and skips 1500
             rows: its first window of the file is (10 + 1500 + 256) * 1 + 1024
             = 2790 bytes, which holds 2 of the dense frames, so the decode
             stops 512 rows in, with 988 rows still to skip. *)
          let path = fixture "flac_seek_dense.flac" in
          let row = 65536 + 1500 in
          equal packed
            (slice (decoded path) ~row ~rows:10)
            (Audio.window (Nx_io.load_audio path) ~row ~rows:10));
      test "a point its frame does not confirm is passed over" (fun () ->
          (* flac_seek_wrong.flac's points but row 0's name no frame, the wrong
             row, the wrong rows or a byte past the file. With frame 0
             corrupted, every window decodes it. *)
          let s = read (fixture "flac_seek_wrong.flac") in
          let path = file (flip_byte s (first_frame s + 12)) in
          List.iter
            (fun row ->
              fails ~naming:"CRC-16" ~path (fun () ->
                  Audio.window (Nx_io.load_audio path) ~row ~rows:1))
            [ 33; 65; 81; 100 ]);
    ]

let pieces_md5 =
  group "MD5 in pieces"
    [
      test "blocks from row 0 check the MD5 when they give the last" (fun () ->
          let s = read (fixture "flac_seek.flac") in
          let path = file (flip_md5 s 3) in
          let a = Nx_io.load_audio path in
          let n = Audio.length a in
          let bs = Audio.blocks a ~rows:30 in
          (* Rows 0 to 89 are given; the block of rows 90 to 119 is not the
             last. *)
          List.iter ignore (List.of_seq (Seq.take 4 bs));
          fails ~naming:"MD5" ~path (fun () -> List.of_seq bs);
          fails ~naming:"MD5" ~path (fun () -> Audio.window a ~row:0 ~rows:n);
          ignore (Audio.window a ~row:0 ~rows:(n - 1));
          ignore (Audio.window a ~row:1 ~rows:(n - 1));
          ignore (List.of_seq (Audio.blocks ~row:1 a ~rows:30)));
      test "a tail of a pass from row 0 forced twice gives the same blocks"
        (fun () ->
          (* The tail's nodes carry the digest of the frames before them, which
             the file's MD5 checks at the last block. *)
          let a = Nx_io.load_audio (fixture "flac_seek.flac") in
          match Audio.blocks a ~rows:30 () with
          | Seq.Nil -> failf "no block"
          | Seq.Cons (_, tail) ->
              equal (list packed) (List.of_seq tail) (List.of_seq tail));
      test "a sequence forced twice gives the same blocks" (fun () ->
          let a = Nx_io.load_audio (cut "ietf_subset_24.flac") in
          let bs = Audio.blocks ~row:5 a ~rows:1000 in
          equal (list packed) (List.of_seq bs) (List.of_seq bs));
      test "a window before a corrupted frame returns, one over it raises"
        (fun () ->
          let s = read (fixture "flac_seek.flac") in
          let path = file (flip_byte s (String.length s - 1)) in
          let a = Nx_io.load_audio path in
          ignore (Audio.window a ~row:0 ~rows:112);
          fails ~naming:"CRC-16" ~path (fun () ->
              Audio.window a ~row:100 ~rows:20));
      test "arguments outside the rows" (fun () ->
          let a = Nx_io.load_audio (fixture "flac_seek.flac") in
          let n = Audio.length a in
          List.iter
            (fun (row, rows) ->
              invalid ~by:"Nx_io.Audio.window" (fun () ->
                  Audio.window a ~row ~rows))
            [ (-1, 1); (0, -1); (n, 1); (1, n) ];
          List.iter
            (fun (row, rows) ->
              invalid ~by:"Nx_io.Audio.blocks" (fun () ->
                  Audio.blocks ~row a ~rows))
            [ (0, 0); (-1, 1); (n + 1, 1) ]);
    ]

(* Domains *)

(* One file's [Audio.t] and passes of its blocks from row 0 shared between
   domains, against its samples: a pass's reference is the blocks it has left to
   give. The file states its MD5, so each node's digest is in play wherever it
   is forced. *)
let shared = fixture "flac_seek.flac"
let audio = abstract "a"
let pass = abstract "p"

let slices rows =
  let (Nx.P x as p) = whole shared in
  let n = (Nx.shape x).(0) in
  List.init
    ((n + rows - 1) / rows)
    (fun i -> slice p ~row:(i * rows) ~rows:(Int.min rows (n - (i * rows))))

let block_rows = Gen.with_pp Format.pp_print_int (Gen.int_range 1 50)

let shared_domains =
  stateful ~domains:2 ~count:20 "a file and its passes shared between domains"
    [
      command "load"
        (Gen.unit @-> makes audio)
        (fun () -> ())
        (fun () -> Nx_io.load_audio shared);
      command "blocks"
        (block_rows @-> audio ^-> makes pass)
        (fun rows () -> slices rows)
        (fun rows a -> Audio.blocks a ~rows);
      command "tail"
        (pass ^-> makes pass)
        (function [] -> [] | _ :: t -> t)
        (fun s -> match s () with Seq.Nil -> Seq.empty | Seq.Cons (_, t) -> t);
      command "force" (pass ^-> returns (list packed)) Fun.id List.of_seq;
      command "window"
        (block_rows @-> block_rows @-> audio ^-> returns packed)
        (fun row rows () -> slice (whole shared) ~row ~rows)
        (fun row rows a -> Audio.window a ~row ~rows);
    ]

(* Corruption *)

let small = read (fixture "flac_depth_16.flac")

let corrupted =
  group "corrupted files"
    [
      test "a load decodes into memory of its own" (fun () ->
          equal memory Own_memory
            (memory_of (decoded (fixture "flac_depth_16.flac"))));
      test "every proper prefix of a file raises" (fun () ->
          prefixes_fail decoded small);
      test "a flipped bit in a frame raises" (fun () ->
          for i = first_frame small to String.length small - 1 do
            for b = 0 to 7 do
              let path = file (flip_bit small i b) in
              fails ~path (fun () -> decoded path)
            done
          done);
      test "a flipped byte loads or raises Failure" (fun () ->
          flips_load_or_fail decoded small);
    ]

let () =
  exit
    (run "nx.io FLAC"
       [
         ietf_cuts;
         built_streams;
         ietf_refused;
         past_length;
         built_refused;
         md5;
         speech;
         windows;
         joined;
         seeking;
         pieces_md5;
         shared_domains;
         corrupted;
       ])
