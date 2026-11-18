(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The IETF FLAC test files, whole: every file of the subset and uncommon sets a
   decoder must read loads with its MD5, and the RFC 9639 MD5 of the samples a
   load returns, recomputed here, is STREAMINFO's, which FFmpeg's decode of each
   file also gives; windows equal slices of the samples; every other file raises
   naming its fault. The dune file says how to run it. *)

open Windtrap
module Audio = Nx_io.Audio
module D = Nx_array.Dtype

let env k =
  match Sys.getenv_opt k with
  | Some v -> v
  | None -> failwith (k ^ " is not set: the dune file says how to run this")

let root = env "NX_IO_FLAC_TESTS"
let read path = In_channel.with_open_bin path In_channel.input_all

(* The file of number [n] in the set [set]. *)
let path set n =
  let dir = Filename.concat root set in
  let name =
    List.find
      (fun f -> String.starts_with ~prefix:(Printf.sprintf "%02d - " n) f)
      (Array.to_list (Sys.readdir dir))
  in
  Filename.concat dir name

(* STREAMINFO's MD5: bytes 26 to 41 of a file whose first block it is. *)
let stated path = Digest.to_hex (String.sub (read path) 26 16)

(* The RFC 9639 MD5 (8.2) of samples a load returned at [bits]. *)
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

let slice (Nx.P x) ~row ~rows = Nx.P (Nx.slice [ Nx.R (row, row + rows) ] x)

let packed =
  Testable.make
    ~pp:(fun ppf (Nx.P x) -> Nx.pp ppf x)
    ~equal:(fun (Nx.P a) (Nx.P b) ->
      D.equal (Nx.dtype a) (Nx.dtype b)
      && Nx.shape a = Nx.shape b
      && Nx.to_array (Nx.bitcast Nx.uint8 a)
         = Nx.to_array (Nx.bitcast Nx.uint8 b))

let readable =
  List.init 64 (fun i -> ("subset", i + 1))
  @ List.map (fun n -> ("uncommon", n)) [ 5; 6; 7; 8; 9 ]

let loads =
  cases "every file a decoder must read loads as its MD5 states"
    ~id:(fun (set, n) -> Printf.sprintf "%s/%02d" set n)
    readable
    (fun (set, n) ->
      let p = path set n in
      let a = Nx_io.load_audio p in
      let x = Audio.samples a in
      equal string (stated p) (rfc_md5 ~bits:(Audio.bits a) x);
      (* Three windows, from a seek point where the file has one. *)
      let len = Audio.length a in
      List.iter
        (fun (row, rows) ->
          equal packed (slice x ~row ~rows) (Audio.window a ~row ~rows))
        [ (0, Int.min len 100); (len / 3, len / 7); (len - (len / 5), len / 5) ])

let refused =
  cases "every other file raises naming its fault"
    ~id:(fun (s, n, _) -> Printf.sprintf "%s/%02d" s n)
    [
      ("uncommon", 1, "sample rate differs");
      ("uncommon", 2, "channels differ");
      ("uncommon", 3, "channels differ");
      ("uncommon", 4, "bit depth differs");
      ("uncommon", 10, "neither a WAV nor a FLAC");
      ("uncommon", 11, "neither a WAV nor a FLAC");
      ("faulty", 1, "longer than STREAMINFO's longest");
      ("faulty", 2, "outside STREAMINFO's sizes");
      ("faulty", 3, "bit depth differs");
      ("faulty", 4, "channels differ");
      ("faulty", 5, "runs past STREAMINFO's length");
      ("faulty", 6, "STREAMINFO is not the first");
      ("faulty", 7, "STREAMINFO is not the first");
      ("faulty", 8, "blocks of 0 to 0 rows");
      ("faulty", 9, "blocks of 1 to 1 rows");
      ("faulty", 10, "Vorbis comment is malformed");
      ("faulty", 11, "of type 127");
    ]
    (fun (set, n, sub) ->
      let p = path set n in
      raises_match
        (function
          | Failure m ->
              String.starts_with ~prefix:(p ^ ": ") m
              && Exn.failure ~sub (Failure m)
          | _ -> false)
        (fun () -> Audio.samples (Nx_io.load_audio p)))

let () = exit (run "nx.io FLAC test files" [ loads; refused ])
