(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Audio files to Whisper's log-mel features.

   librispeech/7902-96594-0025.flac is a LibriSpeech utterance (CC BY 4.0, see
   librispeech/NOTICE). upstream/audio.npz holds upstream's log_mel_spectrogram
   of it, with load_audio on ffmpeg, on the single-window path (short: the
   samples padded to 30 s) and the whole-file path (whole: 30 s of zeros
   appended), at float32 (up32) with the float64 reference of the same samples
   (ref64), at 80 and 128 bands; and of a seeded clip cut to several lengths.
   Upstream's float32 features lie up to about 6e-5 from float64 on LibriSpeech,
   so two float32 implementations may lie twice that apart: ours are held within
   1e-4 of float64 and 2e-4 of upstream's float32. upstream/int8.flac is an
   8-bit FLAC file of the values -128 to 127. *)

open Windtrap
module A = Whisper.Audio

(* Load *)

(* A WAV file's samples, one channel. *)
type wav =
  | U8 of int array
  | I16 of int array
  | I32 of int32 array
  | I64 of int64 array
  | F32 of float array
  | F64 of float array

let column dt a = Nx.create dt [| Array.length a; 1 |] a

let save path = function
  | U8 a -> Nx_io.save_wav path ~rate:A.rate (column Nx.uint8 a)
  | I16 a -> Nx_io.save_wav path ~rate:A.rate (column Nx.int16 a)
  | I32 a -> Nx_io.save_wav path ~rate:A.rate (column Nx.int32 a)
  | I64 a -> Nx_io.save_wav path ~rate:A.rate (column Nx.int64 a)
  | F32 a -> Nx_io.save_wav path ~rate:A.rate (column Nx.float32 a)
  | F64 a -> Nx_io.save_wav path ~rate:A.rate (column Nx.float64 a)

(* [x] rounded to float32. *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* What load's contract gives, each value exact at float64 before its one
   rounding to float32. *)
let amplitudes = function
  | U8 a -> Array.map (fun v -> f32 (Float.of_int (v - 128) /. 128.)) a
  | I16 a -> Array.map (fun v -> f32 (Float.of_int v /. 32768.)) a
  | I32 a -> Array.map (fun v -> f32 (Int32.to_float v /. 0x1p31)) a
  | I64 a -> Array.map (fun v -> f32 (Int64.to_float v /. 0x1p63)) a
  | F32 a -> a
  | F64 a -> Array.map f32 a

let wav_gen =
  let open Gen in
  let size =
    frequency [ (1, constant 0); (1, constant 1); (4, int_range 2 64) ]
  in
  let values extremes any =
    array ~size (frequency [ (1, of_list extremes); (3, any) ])
  in
  one_of
    [
      map (fun a -> U8 a) (values [ 0; 1; 127; 128; 255 ] (int_range 0 255));
      map
        (fun a -> I16 a)
        (values [ -32768; -1; 0; 1; 32767 ] (int_range (-32768) 32767));
      map
        (fun a -> I32 a)
        (values [ Int32.min_int; -1l; 0l; Int32.max_int ] int32);
      (* At most 53 significant bits, so the float64 reference is exact. *)
      map
        (fun a -> I64 (Array.map (fun v -> Int64.logand v (-2048L)) a))
        (values [ Int64.min_int; 0L; Int64.max_int ] int64);
      map
        (fun a -> F32 (Array.map f32 a))
        (values [ -0.; 1.; -1.; 1.5; Float.infinity ] (float_range (-2.) 2.));
      map
        (fun a -> F64 a)
        (values [ -0.; 3.; 1e-40; 0.1 ] (float_range (-2.) 2.));
    ]

let bits a = Array.map Int32.bits_of_float a

let load_scales =
  prop "load reads each dtype at its full scale" wav_gen (fun w ->
      let kind =
        match w with
        | U8 _ -> "uint8"
        | I16 _ -> "int16"
        | I32 _ -> "int32"
        | I64 _ -> "int64"
        | F32 _ -> "float32"
        | F64 _ -> "float64"
      in
      cover kind true;
      let want = amplitudes w in
      cover "no sample" (Array.length want = 0);
      let path = temp_file ~suffix:".wav" () in
      save path w;
      let got = A.load path in
      equal (array int) [| Array.length want |] (Nx.shape got);
      equal (array int32) (bits want) (bits (Nx.to_array got)))

let refuses =
  cases "load refuses what Whisper does not hear"
    ~id:(fun (name, _, _, _) -> name)
    [ ("stereo", A.rate, 2, "2 channels") ]
    (fun (_, rate, c, sub) ->
      let path = temp_file ~suffix:".wav" () in
      Nx_io.save_wav path ~rate (Nx.zeros Nx.int16 [| 160; c |]);
      raises_match
        (fun e -> Exn.failure ~sub e && Exn.failure ~sub:path e)
        (fun () -> A.load path))

(* 2^20 - 1 Hz, the largest rate FLAC's STREAMINFO states, loads; a rate past it
   raises. *)
let top_rate =
  test "load refuses rates past 1048575 Hz" (fun () ->
      let path rate =
        let p = temp_file ~suffix:".wav" () in
        Nx_io.save_wav p ~rate (Nx.ones Nx.int16 [| 100; 1 |]);
        p
      in
      let top = path 1_048_575 in
      (* ⌈100 × 16000 / 1048575⌉ samples. *)
      equal int 2 (Nx.dim 0 (A.load top));
      let past = path 1_048_576 in
      raises_match
        (fun e -> Exn.failure ~sub:"1048576 Hz" e && Exn.failure ~sub:past e)
        (fun () -> A.load past))

(* A WAV file whose header states a rate of 0, which Nx_io reads as stated. *)
let no_rate =
  test "load refuses a file without a rate" (fun () ->
      let path = temp_file ~suffix:".wav" () in
      Nx_io.save_wav path ~rate:1 (Nx.zeros Nx.int16 [| 160; 1 |]);
      let b =
        Bytes.of_string (In_channel.with_open_bin path In_channel.input_all)
      in
      (* nSamplesPerSec and nAvgBytesPerSec of the fmt chunk. *)
      Bytes.set_int32_le b 24 0l;
      Bytes.set_int32_le b 28 0l;
      Out_channel.with_open_bin path (fun oc -> Out_channel.output_bytes oc b);
      equal int 0 (Nx_io.Audio.rate (Nx_io.load_audio path));
      raises_match
        (fun e -> Exn.failure ~sub:"0 Hz" e && Exn.failure ~sub:path e)
        (fun () -> A.load path))

let int8 =
  test "load reads an 8-bit FLAC file over 128" (fun () ->
      let want =
        Array.init 256 (fun i -> f32 (Float.of_int (i - 128) /. 128.))
      in
      let got = A.load "upstream/int8.flac" in
      equal (array int32) (bits want) (bits (Nx.to_array got)))

(* Resampling *)

(* upstream/resample.npz holds upstream's load_audio, ffmpeg's int16 samples, of
   each of the files beside it, and of the first samples of a 44.1 kHz clip
   ("short.x") for each length of "short.lengths". *)
let resampled = lazy (Nx_io.load_npz "upstream/resample.npz")

let resampled_entry dt key =
  match Nx_io.Archive.find key (Lazy.force resampled) with
  | Some p -> Nx.unpack dt p
  | None -> failf "upstream/resample.npz has no %s" key

(* ffmpeg computes its samples in float32 and rounds each to the nearest 16-bit
   value. Its float32 samples were measured within 1.3e-7 of full scale of the
   exact computation (against PyAV's libswresample at 48, 44.1 and 16 to 48 kHz;
   the worst case of float32 sums of 92 taps is about 5e-6), and load rounds the
   exact computation to float32, within 2^-25. So each of ffmpeg's samples lies
   within half a step of 2^-15 and those bounds of load's sample, in steps: a
   bound measured on these files. *)
let ffmpeg_steps = 0.5 +. (32768. *. (1.3e-7 +. 0x1p-25))

(* [ffmpeg ~n ~from want got] checks load's samples [got] of [n] samples at
   [from] Hz against ffmpeg's [want]. load gives [⌈16000 n / from⌉]; ffmpeg's
   flush gives that many or fewer, and compares on those. *)
let ffmpeg ~n ~from want got =
  equal int (((16_000 * n) + from - 1) / from) (Nx.dim 0 got);
  let want = Nx.to_array (Nx.cast Nx.float64 want) in
  let k = Array.length want in
  at_most int ~than:(Nx.dim 0 got) k;
  let got = Nx.to_array (Nx.cast Nx.float64 (Nx.slice [ R (0, k) ] got)) in
  equal (array (float ffmpeg_steps)) want (Array.map (fun v -> v *. 32768.) got)

let ffmpeg_files =
  cases "load resamples as upstream's ffmpeg"
    ~id:(fun (path, _) -> path)
    [
      ("mono48.flac", 48_000);
      ("mono8.wav", 8_000);
      ("mono43650.wav", 43_650);
      ("mono47530.wav", 47_530);
      ("mono44101.wav", 44_101);
    ]
    (fun (path, from) ->
      let a = Nx_io.load_audio ("upstream/" ^ path) in
      equal int from (Nx_io.Audio.rate a);
      ffmpeg ~n:(Nx_io.Audio.length a) ~from
        (resampled_entry Nx.int16 path)
        (A.load ("upstream/" ^ path)))

(* Lengths about the filter's 92 taps at 44.1 kHz; ffmpeg gives fewer samples
   below about 90 and none below 62. *)
let ffmpeg_short =
  let x = resampled_entry Nx.int16 "short.x" in
  let lengths = Nx.to_array (resampled_entry Nx.int64 "short.lengths") in
  cases "load resamples short files as upstream's ffmpeg"
    ~id:(fun n -> Printf.sprintf "%Ld samples" n)
    (Array.to_list lengths)
    (fun n ->
      let n = Int64.to_int n in
      let path = temp_file ~suffix:".wav" () in
      Nx_io.save_wav path ~rate:44_100
        (Nx.reshape [| n; 1 |] (Nx.slice [ R (0, n) ] x));
      ffmpeg ~n ~from:44_100
        (resampled_entry Nx.int16 (Printf.sprintf "short.%d" n))
        (A.load path))

(* A file of [n] samples at [from] Hz, of [c] channels, and the rows of a
   block. *)
let stream_gen =
  let open Gen in
  let* from =
    frequency
      [
        (2, constant 16_000);
        (2, constant 44_101);
        (4, of_list [ 8_000; 11_025; 22_050; 44_100; 48_000 ]);
      ]
  in
  let* n = frequency [ (1, int_range 0 3); (4, int_range 4 400) ] in
  let* rows =
    frequency [ (2, int_range 1 3); (2, int_range 4 120); (1, int_range 1 500) ]
  in
  let* x = array ~size:(constant n) (int_range (-32768) 32767) in
  constant (from, x, rows)

let stream =
  prop ~count:60 "load gives the same samples for any blocks it reads"
    stream_gen (fun (from, x, rows) ->
      let n = Array.length x in
      cover "blocks shorter than any filter" (rows < 32);
      cover "one block" (rows >= n);
      cover "16 kHz" (from = A.rate);
      cover "phases interpolated" (from = 44_101);
      let path = temp_file ~suffix:".wav" () in
      Nx_io.save_wav path ~rate:from (Nx.create Nx.int16 [| n; 1 |] x);
      let whole = A.load_rows ~rows:(Int.max 1 n) path in
      equal (array int32)
        (bits (Nx.to_array whole))
        (bits (Nx.to_array (A.load_rows ~rows path))))

let stream_flac =
  cases "load reads a FLAC file in any blocks" ~id:string_of_int
    [ 1; 777; 4096 ] (fun rows ->
      let path = "upstream/mono48.flac" in
      equal (array int32)
        (bits (Nx.to_array (A.load path)))
        (bits (Nx.to_array (A.load_rows ~rows path))))

(* Log-mel *)

let golden = lazy (Nx_io.load_npz "upstream/audio.npz")

let entry dt key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "upstream/audio.npz has no %s" key

(* [want]'s frames, its last repeated up to [k]: the generator keeps them up to
   the first floor frame past the audio, which every later one equals. *)
let frames k want =
  let j = Nx.dim 0 want in
  Nx.concatenate ~axis:0
    [
      want;
      Nx.broadcast_to
        [| k - j; Nx.dim 1 want |]
        (Nx.slice [ R (j - 1, j) ] want);
    ]

(* [got] against upstream's float32 features [up32] and the float64 reference
   [ref64], within the gates. *)
let upstream ~ref64 ~up32 got =
  equal (array int) (Nx.shape ref64) (Nx.shape got);
  let got = Nx.to_array (Nx.cast Nx.float64 got) in
  equal ~msg:"float64" (array (float 1e-4)) (Nx.to_array ref64) got;
  equal ~msg:"upstream"
    (array (float 2e-4))
    (Nx.to_array (Nx.cast Nx.float64 up32))
    got

let fixture = "librispeech/7902-96594-0025.flac"

let librispeech =
  cases "LibriSpeech's features are upstream's"
    ~id:(fun (path, n) -> Printf.sprintf "%s, %d bands" path n)
    [ ("short", 80); ("short", 128); ("whole", 80); ("whole", 128) ]
    (fun (path, n) ->
      let x = A.load fixture in
      let got =
        if path = "short" then A.log_mel ~n_mels:n (A.pad_or_trim x)
        else A.log_mel ~padding:A.length ~n_mels:n x
      in
      let k =
        (if path = "short" then A.length else Nx.dim 0 x + A.length) / 160
      in
      let key what = Printf.sprintf "%s.mel%d.%s" path n what in
      upstream
        ~ref64:(frames k (entry Nx.float64 (key "ref64")))
        ~up32:(frames k (entry Nx.float32 (key "up32")))
        got)

(* A clip cut to lengths about a hop, a window and the reflection's 200 samples:
   [n / 160] frames, upstream's values where upstream computes them (more than
   200 samples), the float64 reference's from 160 to 200, and no frame below 160
   samples. *)
let lengths =
  let clip = entry Nx.float32 "clip" in
  let lengths = Nx.to_array (entry Nx.int64 "lengths") in
  let upstream_frames = Nx.to_array (entry Nx.int64 "frames") in
  cases "log_mel gives a frame every 160 samples"
    ~id:(fun (n, _) -> Printf.sprintf "%Ld samples" n)
    (Array.to_list (Array.combine lengths upstream_frames))
    (fun (n, up) ->
      let n = Int64.to_int n in
      let got = A.log_mel ~n_mels:80 (Nx.slice [ R (0, n) ] clip) in
      equal (array int) [| n / 160; 80 |] (Nx.shape got);
      let ref64 () = entry Nx.float64 (Printf.sprintf "len%d.ref64" n) in
      if up >= 0L then
        upstream ~ref64:(ref64 ())
          ~up32:(entry Nx.float32 (Printf.sprintf "len%d.up32" n))
          got
      else if n >= 160 then
        (* Upstream raises; the float64 reference mirrors as often as needed. *)
        equal ~msg:"float64"
          (array (float 1e-4))
          (Nx.to_array (ref64 ()))
          (Nx.to_array (Nx.cast Nx.float64 got)))

let silence =
  test "silence gives (log10 1e-10 + 4) / 4 everywhere" (fun () ->
      let check x = Array.iter (equal float_exact (-1.5)) (Nx.to_array x) in
      let a = A.log_mel ~n_mels:80 (Nx.zeros Nx.float32 [| A.length |]) in
      equal (array int) [| 3000; 80 |] (Nx.shape a);
      check a;
      let b =
        A.log_mel ~padding:1600 ~n_mels:128 (Nx.zeros Nx.float32 [| 0 |])
      in
      equal (array int) [| 10; 128 |] (Nx.shape b);
      check b)

(* 1 s of a tone at 1e-5 of full scale, then 1 s at 0.5: about 94 dB apart, past
   the floor's 80 dB. *)
let floor =
  test "the floor is the whole input's" (fun () ->
      let tone a i = a *. Float.sin (0.2 *. Float.of_int i) in
      let x =
        Nx.init Nx.float32
          [| 2 * A.rate |]
          (fun i -> tone (if i.(0) < A.rate then 1e-5 else 0.5) i.(0))
      in
      let quiet = Nx.slice [ R (0, A.rate) ] x in
      let both = A.log_mel ~n_mels:80 x in
      let floor = Nx.item [] (Nx.max both) -. 2. in
      (* The frames that read only the quiet second all sit on the floor the
         loud one sets. *)
      let first = Nx.to_array (Nx.slice [ R (0, 98) ] both) in
      Array.iter (equal (float 1e-6) floor) first;
      let alone = A.log_mel ~n_mels:80 quiet in
      less float_exact ~than:(floor -. 0.1) (Nx.item [] (Nx.min alone)))

(* A tone in noise, 2.5 s of int16 samples. *)
let tone =
  let rng = Random.State.make [| 7 |] in
  Array.init 40_000 (fun i ->
      let t = Float.of_int i /. Float.of_int A.rate in
      let v =
        (9000. *. Float.sin (2. *. Float.pi *. 440. *. t))
        +. Random.State.float rng 2000.
        -. 1000.
      in
      Float.to_int v)

let int16_path =
  test "a WAV file gives the features of its samples over 32768, bit for bit"
    (fun () ->
      let x = Nx.create Nx.int16 [| Array.length tone |] tone in
      let path = temp_file ~suffix:".wav" () in
      Nx_io.save_wav path ~rate:A.rate (Nx.reshape [| Array.length tone; 1 |] x);
      let of_file = A.log_mel ~n_mels:80 (A.load path) in
      let of_array =
        A.log_mel ~n_mels:80 (Nx.div_s (Nx.cast Nx.float32 x) 32768.)
      in
      equal (array int) [| 250; 80 |] (Nx.shape of_file);
      equal (array int32)
        (bits (Nx.to_array of_array))
        (bits (Nx.to_array of_file)))

let pad_or_trim =
  cases "pad_or_trim cuts or pads to 30 s"
    ~id:(fun n -> Printf.sprintf "%d samples" n)
    [ 0; 1; 16_000; A.length; A.length + 1; A.length + 3_000 ]
    (fun n ->
      let x =
        Nx.init Nx.float32 [| n |] (fun i -> Float.of_int (i.(0) mod 7) -. 3.)
      in
      let got = Nx.to_array (A.pad_or_trim x) in
      equal int A.length (Array.length got);
      let want =
        Array.init A.length (fun i ->
            if i < n then Float.of_int (i mod 7) -. 3. else 0.)
      in
      equal (array int32) (bits want) (bits got))

let shapes =
  test "log_mel and pad_or_trim take samples of one axis" (fun () ->
      raises_match Exn.invalid_arg (fun () ->
          A.log_mel ~n_mels:80 (Nx.zeros Nx.float32 [| 2; 160 |]));
      raises_match Exn.invalid_arg (fun () ->
          A.log_mel ~n_mels:0 (Nx.zeros Nx.float32 [| 160 |]));
      raises_match Exn.invalid_arg (fun () ->
          A.log_mel ~padding:(-1) ~n_mels:80 (Nx.zeros Nx.float32 [| 160 |]));
      raises_match Exn.invalid_arg (fun () ->
          A.pad_or_trim (Nx.zeros Nx.float32 [| 2; 160 |])))

let () =
  exit
    (run "whisper.audio"
       [
         load_scales;
         refuses;
         top_rate;
         no_rate;
         int8;
         ffmpeg_files;
         ffmpeg_short;
         stream;
         stream_flac;
         librispeech;
         lengths;
         silence;
         floor;
         int16_path;
         pad_or_trim;
         shapes;
       ])
