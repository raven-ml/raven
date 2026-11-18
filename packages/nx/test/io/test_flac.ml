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

(* Corruption *)

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

let small = read (fixture "flac_depth_16.flac")
let decoded path = Audio.samples (Nx_io.load_audio path)

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
         corrupted;
       ])
