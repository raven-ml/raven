(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* FLAC, as nx_io.mli's Audio section states it. Cuts of the IETF test files
   load as FFmpeg decodes them: the RFC 9639 MD5 of the samples a load returns,
   recomputed here, is the one FFmpeg's samples give; streams built to exercise
   each feature RFC 9639 defines load as the samples they were built from, each
   in the top bits of its dtype; every file the RFC forbids raises [Failure]
   naming the file, and nothing else. *)

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
    if v land ((1 lsl ((8 * w) - bits)) - 1) <> 0 then
      failf "a sample %d has a set bit below its %d bits" v bits;
    let v = v asr ((8 * w) - bits) in
    for i = 0 to k - 1 do
      Buffer.add_char out (Char.chr ((v asr (8 * i)) land 0xff))
    done
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

let dtype_of_bits bits =
  if bits <= 8 then "int8" else if bits <= 16 then "int16" else "int32"

(* A file's facts: its name, rate, channels, bits, length, the MD5 of its
   samples, and its channel mask if it states one. *)
type facts = {
  name : string;
  rate : int;
  channels : int;
  bits : int;
  length : int;
  md5 : string;
  mask : int option;
}

(* The rows of a table of facts, from its header's line on. *)
let rows path =
  match String.split_on_char '\n' (String.trim (read path)) with
  | [] -> []
  | _ :: rows -> List.map (String.split_on_char '\t') rows

(* The IETF cuts, as FFmpeg decodes them. *)
let ietf =
  List.map
    (function
      | [ name; _; rate; channels; bits; length; _; _; _; md5 ] ->
          {
            name;
            rate = int_of_string rate;
            channels = int_of_string channels;
            bits = int_of_string bits;
            length = int_of_string length;
            md5;
            mask = None;
          }
      | row -> failwith ("ietf.tsv: " ^ String.concat "\t" row))
    (rows (cut "ietf.tsv"))

(* The built streams, as their generator states them. *)
let built =
  List.map
    (function
      | [ name; rate; channels; bits; length; mask; md5 ] ->
          {
            name;
            rate = int_of_string rate;
            channels = int_of_string channels;
            bits = int_of_string bits;
            length = int_of_string length;
            md5;
            mask = (if mask = "" then None else Some (int_of_string mask));
          }
      | row -> failwith ("flac.tsv: " ^ String.concat "\t" row))
    (rows (fixture "flac.tsv"))

(* The speakers of a stream of [c] channels that states none: nx_io.mli's order
   of RFC 9639's section 9.1.3. *)
let default_mask c =
  [| 0x4; 0x3; 0x7; 0x33; 0x607; 0x60F; 0x70F; 0x63F |].(c - 1)

(* [loads_as path c] checks the file [path] against its facts [c]. *)
let loads_as path c =
  let a = Nx_io.load_audio path in
  equal (triple int int int)
    (c.rate, c.channels, c.bits)
    (Audio.rate a, Audio.channels a, Audio.bits a);
  equal int c.length (Audio.length a);
  equal int
    (Option.value c.mask ~default:(default_mask c.channels))
    (Audio.channel_mask a);
  let (Nx.P x as p) = Audio.samples a in
  equal string (dtype_of_bits c.bits) (D.name (Nx.dtype x));
  equal (array int) [| c.length; c.channels |] (Nx.shape x);
  equal string c.md5 (rfc_md5 ~bits:c.bits p)

(* Bytes *)

(* [with_md5 s md5] is the file [s] with its STREAMINFO's MD5 [md5]: the last 16
   of its 34 bytes, which follow the marker and the block's header. *)
let with_md5 s md5 =
  String.sub s 0 26 ^ md5 ^ String.sub s 42 (String.length s - 42)

(* [with_length s n] is the file [s] whose STREAMINFO states [n] rows: the low
   36 bits of its 64-bit word at byte 18 (RFC 9639, 8.2). *)
let with_length s n =
  let b = Bytes.of_string s in
  let word = Bytes.get_int64_be b 18 in
  let high = Int64.logand word (Int64.lognot 0xf_ffff_ffffL) in
  Bytes.set_int64_be b 18 (Int64.logor high (Int64.of_int n));
  Bytes.to_string b

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

let stated s = String.sub s 26 16

let flip_md5 s b =
  let m = stated s in
  with_md5 s
    (String.mapi
       (fun i c ->
         if i = b / 8 then Char.chr (Char.code c lxor (1 lsl (b mod 8))) else c)
       m)

(* A stream of 37 rows of 2 channels of 16 bits. *)
let depth_16 = fixture "flac_depth_16.flac"

(* FLAC *)

(* A LibriSpeech file, whose source and licence support/NOTICE states, and the
   MD5 of libsndfile's int16 reading of it, little-endian. *)
let librispeech = fixture "librispeech_7902-96594-0025.flac"
let librispeech_md5 = "572ee418fca9ad27597247a089821499"

let format =
  group "FLAC"
    [
      cases "cuts of the IETF test files load as FFmpeg decodes them"
        ~id:(fun c -> c.name)
        ietf
        (fun c -> loads_as (cut c.name) c);
      cases "built streams load as the samples they were built from"
        ~id:(fun c -> c.name)
        built
        (fun c -> loads_as (fixture c.name) c);
      test "a LibriSpeech file loads as libsndfile reads it" (fun () ->
          let a = Nx_io.load_audio librispeech in
          equal facts_t ((16000, 29120, 1), (16, 4)) (facts a);
          let x = Nx.unpack Nx.int16 (Audio.samples a) in
          let le = Bytes.create (2 * Nx.numel x) in
          Array.iteri
            (fun i v -> Bytes.set_int16_le le (2 * i) v)
            (Nx.to_array x);
          equal string librispeech_md5 (Digest.to_hex (Digest.bytes le)));
      prop "bytes after the frames of STREAMINFO's length are ignored"
        ~examples:[ read depth_16 ]
        Gen.string
        (fun tail ->
          equal packed (decoded depth_16)
            (decoded (file (read depth_16 ^ tail))));
      cases
        "a length whose samples take more than 2^16 times the frames' bytes is \
         refused at load, and one at that bound is not"
        ~id:fst
        [ ("one row past", 1); ("at the bound", 0) ]
        (fun (_, over) ->
          (* 2 channels of int16 take 4 bytes a row, so the bound is 2^14 rows a
             byte of the frames. *)
          let s = read depth_16 in
          let frames = String.length s - first_frame s in
          let n = (frames lsl 14) + over in
          let path = file (with_length s n) in
          within_budget @@ fun () ->
          if over = 1 then
            fails ~path ~naming:"STREAMINFO states" (fun () ->
                Nx_io.load_audio path)
          else equal int n (Audio.length (Nx_io.load_audio path)));
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
          fails ~naming ~path (fun () -> decoded path));
      test "faulty 05 without its MD5 raises at the frame past its length"
        (fun () ->
          let s = read (cut "ietf_faulty_05.flac") in
          let path = file (with_md5 s (String.make 16 '\000')) in
          fails ~naming:"runs past STREAMINFO's length" ~path (fun () ->
              decoded path));
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
          ( "short_not_last",
            "shorter than STREAMINFO's shortest is not the last" );
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
          within_budget @@ fun () ->
          fails ~naming ~path (fun () -> decoded path));
      test "a flipped bit in a frame raises" (fun () ->
          (* Each frame ends in a CRC-16 over its bytes (RFC 9639, 9.3), and a
             polynomial of two or more terms detects every one-bit error. *)
          let s = read depth_16 in
          (* One file, each flip made and undone in place. *)
          let path = file s in
          for i = first_frame s to String.length s - 1 do
            for b = 0 to 7 do
              write_at path i (String.sub (flip_bit s i b) i 1);
              fails ~path (fun () -> decoded path);
              write_at path i (String.sub s i 1)
            done
          done);
    ]

(* MD5 *)

let md5 =
  group "MD5"
    [
      test "a load checks the samples against STREAMINFO's MD5" (fun () ->
          let s = read depth_16 in
          for b = 0 to 127 do
            let path = file (flip_md5 s b) in
            let a = Nx_io.load_audio path in
            fails ~naming:"MD5" ~path (fun () -> Audio.samples a)
          done);
      test "a load that counts the rows checks the MD5" (fun () ->
          let path = file (flip_md5 (read (fixture "flac_no_length.flac")) 0) in
          fails ~naming:"MD5" ~path (fun () -> Nx_io.load_audio path));
      test "an MD5 of zeros states none" (fun () ->
          let s = read depth_16 in
          equal packed (decoded depth_16)
            (decoded (file (with_md5 s (String.make 16 '\000')))));
    ]

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

let seek = fixture "flac_seek.flac"

(* Each file's samples, decoded once in each worker when a test first asks, so a
   decoder that fails fails that test. *)
let whole =
  let memo = Hashtbl.create 16 in
  fun p ->
    match Hashtbl.find_opt memo p with
    | Some x -> x
    | None ->
        let x = decoded p in
        Hashtbl.add memo p x;
        x

let rows_of p =
  let (Nx.P x) = whole p in
  (Nx.shape x).(0)

let slice (Nx.P x) ~row ~rows = Nx.P (Nx.slice [ Nx.R (row, row + rows) ] x)

let join = function
  | [] -> failf "no block"
  | Nx.P x :: _ as bs ->
      let dt = Nx.dtype x in
      Nx.P (Nx.concatenate ~axis:0 (List.map (fun b -> Nx.unpack dt b) bs))

(* A file, its rows, and a row and a count of rows in it. An eighth of the
   pieces start past [seek]'s first seek point, row 32, and an eighth read more
   than 1024 rows past row 16384 of a file that long. *)
let piece =
  let open Gen in
  let any =
    let* path = of_list ~pp:Format.pp_print_string pieces in
    let n = rows_of path in
    let* row =
      frequency [ (2, constant 0); (1, constant n); (5, int_range 0 n) ]
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
  in
  let seeked =
    let* () = unit in
    let n = rows_of seek in
    let* row = int_range 33 n in
    let+ rows = int_range 0 (n - row) in
    (seek, n, row, rows)
  in
  let dense =
    let* () = unit in
    let long = List.filter (fun p -> rows_of p > 17409) pieces in
    let* path = of_list ~pp:Format.pp_print_string long in
    let n = rows_of path in
    let* row = int_range 16385 (n - 1025) in
    let+ rows = int_range 1025 (n - row) in
    (path, n, row, rows)
  in
  frequency [ (6, any); (1, seeked); (1, dense) ]

let piece =
  Gen.with_pp
    (fun ppf (p, _, row, rows) ->
      Format.fprintf ppf "%s, %d rows from row %d" p rows row)
    piece

(* [padded s] is [s] with its seek table a padding block. *)
let padded s =
  let rec go at =
    let h = Char.code s.[at] in
    let n = (Char.code s.[at + 1] lsl 16) lor String.get_uint16_be s (at + 2) in
    if h land 0x7f = 3 then
      String.mapi
        (fun i c ->
          if i = at then Char.chr (Char.code c land 0x80 lor 1) else c)
        s
    else if h land 0x80 <> 0 then failf "no seek table"
    else go (at + 4 + n)
  in
  go 4

let seeking =
  group "seek points"
    [
      test "a seek table changes no window" (fun () ->
          List.iter
            (fun name ->
              let a = Nx_io.load_audio name
              and b = Nx_io.load_audio (file (padded (read name))) in
              let n = Audio.length a in
              for i = 0 to 63 do
                let row = i * n / 64 in
                let rows = Int.min 7 (n - row) in
                equal packed
                  (Audio.window a ~row ~rows)
                  (Audio.window b ~row ~rows)
              done)
            [
              seek; fixture "flac_seek_variable.flac"; cut "ietf_subset_48.flac";
            ]);
      test "a read past a confirmed seek point decodes from it" (fun () ->
          (* Frame 0 of flac_seek.flac corrupted: a window past the point at row
             64 does not decode it, and one before every point but row 0's does,
             as does any window of the file without its table. *)
          let s = read seek in
          let bad = flip_byte s (first_frame s + 12) in
          let a = Nx_io.load_audio (file bad) in
          equal packed
            (slice (decoded seek) ~row:70 ~rows:10)
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

(* Audio.t *)

(* One file's [Audio.t] and passes of its blocks from row 0 shared between
   domains, against its samples: a pass's reference is its blocks. A force drops
   the first blocks of a pass, which forces them, and gives the rest, so two
   domains forcing one pass from different blocks force its nodes in any order.
   The file states its MD5, so each node's digest is in play wherever it is
   forced.

   Each program runs 50 times, and the sanitizers' step runs 1000 programs. The
   branches share a load and a pass, which two calls make, so the prefix and the
   suffix make at most four. A pass's blocks hold 4 rows or more: a pass of the
   file's 128 rows has at most 32 nodes, and its blocks still end inside the
   16-row frames. Blocks of 1 row are the single-domain laws'. *)
let audio = abstract "a"
let pass = abstract "p"

let slices rows =
  let p = whole seek in
  let n = rows_of seek in
  List.init
    ((n + rows - 1) / rows)
    (fun i -> slice p ~row:(i * rows) ~rows:(Int.min rows (n - (i * rows))))

let pass_rows = Gen.with_pp Format.pp_print_int (Gen.int_range 4 50)
let block_rows = Gen.with_pp Format.pp_print_int (Gen.int_range 1 50)
let dropped = Gen.with_pp Format.pp_print_int (Gen.int_range 0 4)

let shared =
  group "Audio.t"
    [
      stateful ~domains:2 ~count:20 ~steps:4
        "a file and its passes shared between domains"
        [
          command "load"
            (Gen.unit @-> makes audio)
            (fun () -> ())
            (fun () -> Nx_io.load_audio seek);
          command "blocks"
            (pass_rows @-> audio ^-> makes pass)
            (fun rows () -> slices rows)
            (fun rows a -> Audio.blocks a ~rows);
          command "force"
            (dropped @-> pass ^-> returns (list packed))
            (fun k bs -> List.filteri (fun i _ -> i >= k) bs)
            (fun k s -> List.of_seq (Seq.drop k s));
          command "window"
            (block_rows @-> block_rows @-> audio ^-> returns packed)
            (fun row rows () -> slice (whole seek) ~row ~rows)
            (fun row rows a -> Audio.window a ~row ~rows);
        ];
    ]

(* Audio.samples *)

let samples =
  group "Audio.samples"
    [
      test "a load decodes into memory of its own" (fun () ->
          equal memory Own_memory (memory_of (decoded depth_16)));
    ]

(* Files of both formats and their rows, whose arguments the checks of
   Audio.window and Audio.blocks refuse alike. *)
let outside = [ (seek, 128); (fixture "wav_sf_pcm_16.wav", 37) ]

let window =
  group "Audio.window"
    [
      (* "to the end" has p = 0.114 a case: at 200 cases it misses a run with
         0.886^200 = 3.1e-11. *)
      prop ~count:200 "a window is the slice of the samples it covers" piece
        (fun (path, n, row, rows) ->
          let x = whole path in
          cover "from row 0" (row = 0 && rows > 0);
          cover "no row" (rows = 0);
          cover "to the end" (rows > 0 && row + rows = n);
          cover "past a seek point" (row > 32 && path = seek);
          cover "in the denser frames" (row > 16384 && rows > 1024);
          equal packed (slice x ~row ~rows)
            (Audio.window (Nx_io.load_audio path) ~row ~rows));
      test "a window before a corrupted frame returns, one over it raises"
        (fun () ->
          let s = read seek in
          let path = file (flip_byte s (String.length s - 1)) in
          let a = Nx_io.load_audio path in
          ignore (Audio.window a ~row:0 ~rows:112);
          fails ~naming:"CRC-16" ~path (fun () ->
              Audio.window a ~row:100 ~rows:20));
      cases "a window outside the rows is refused"
        ~id:(fun (p, row, rows) ->
          Printf.sprintf "%s, %d rows from %d" p rows row)
        (List.concat_map
           (fun (p, n) -> [ (p, -1, 1); (p, 0, -1); (p, n, 1); (p, 1, n) ])
           outside)
        (fun (p, row, rows) ->
          let a = Nx_io.load_audio p in
          invalid ~by:"Nx_io.Audio.window" (fun () -> Audio.window a ~row ~rows));
    ]

(* Joined blocks are the samples from their first row. The draws reach one block
   in 9% of the cases and no block in 12%, which 100 cases miss in one run of
   10^4 and one of 3 x 10^5; the examples reach both in every run. *)
let blocks =
  group "Audio.blocks"
    [
      prop "blocks joined are the samples from their row"
        ~examples:[ ((seek, 128, 1, 0), 1); ((seek, 128, 128, 0), 1) ]
        Gen.(pair piece (int_range 1 64))
        (fun ((path, n, row, _), k) ->
          let (Nx.P y as x) = whole path in
          (* At most [k] blocks, and one past the end of the samples. *)
          let rows = if k = 64 then n + 1 else Int.max 1 ((n - row) / k) in
          cover "one row a block" (rows = 1 && row < n);
          cover "one block" (n - row > 0 && rows >= n - row);
          cover "from a row past 0" (row > 0 && row < n);
          cover "no block" (row = n);
          let bs =
            List.of_seq (Audio.blocks ~row (Nx_io.load_audio path) ~rows)
          in
          equal int ((n - row + rows - 1) / rows) (List.length bs);
          List.iteri
            (fun i (Nx.P b) ->
              let m =
                if i = List.length bs - 1 then n - row - (i * rows) else rows
              in
              equal (array int) [| m; (Nx.shape y).(1) |] (Nx.shape b))
            bs;
          if bs <> [] then equal packed (slice x ~row ~rows:(n - row)) (join bs));
      test "blocks from row 0 check the MD5 when they give the last" (fun () ->
          let path = file (flip_md5 (read seek) 3) in
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
          let a = Nx_io.load_audio seek in
          match Audio.blocks a ~rows:30 () with
          | Seq.Nil -> failf "no block"
          | Seq.Cons (_, tail) ->
              equal (list packed) (List.of_seq tail) (List.of_seq tail));
      test "a pass from a later row forced twice gives the same blocks"
        (fun () ->
          let a = Nx_io.load_audio (cut "ietf_subset_24.flac") in
          let bs = Audio.blocks ~row:5 a ~rows:1000 in
          equal (list packed) (List.of_seq bs) (List.of_seq bs));
      cases "a pass outside the rows is refused"
        ~id:(fun (p, row, rows) ->
          Printf.sprintf "%s, blocks of %d from %d" p rows row)
        (List.concat_map
           (fun (p, n) -> [ (p, 0, 0); (p, -1, 1); (p, n + 1, 1) ])
           outside)
        (fun (p, row, rows) ->
          let a = Nx_io.load_audio p in
          invalid ~by:"Nx_io.Audio.blocks" (fun () -> Audio.blocks ~row a ~rows));
    ]

(* load_audio *)

let load =
  group "load_audio"
    [
      test "a directory fails, naming it" (fun () ->
          let path = temp_dir () in
          fails ~path (fun () -> Nx_io.load_audio path));
      test "every proper prefix of a file raises" (fun () ->
          prefixes_fail decoded (read depth_16));
      test "a flipped byte loads or raises Failure" (fun () ->
          flips_load_or_fail decoded (read depth_16));
    ]

let () =
  exit
    (run "nx.io FLAC"
       [ format; md5; seeking; shared; samples; blocks; window; load ])
