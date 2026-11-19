(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Whisper's log-mel frontend composed from alda, against the oracle's float64
   reference on three LibriSpeech utterances padded to 30 s. Upstream's
   whisper.log_mel_spectrogram on CPU float32 lies within 1e-4 of that reference
   on them (contrib/alda/test/gen/logmel.py, which writes golden/whisper.npz,
   checks it), so within 1e-4 of the reference alda is within 2e-4 of
   upstream. *)

open Windtrap

let golden = lazy (Nx_io.load_npz "golden/whisper.npz")

let entry key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack Nx.float32 p
  | None -> failf "golden/whisper.npz has no %s" key

let utterances = [ "121-121726-0002"; "61-70968-0000"; "367-130732-0001" ]

(* Upstream's load_audio: 16-bit samples over 32768, in float32. *)
let audio id =
  let pcm =
    Nx.unpack Nx.int16 (Nx_io.load_npy ("librispeech/" ^ id ^ ".npy"))
  in
  Nx.div_s (Nx.cast Nx.float32 pcm) 32768.

(* The frontend: torch's centred frames without the last, a periodic Hann
   window, |z|² as re² + im², the Slaney bank, log10 above 1e-10, the whole
   tensor's max - 8 as a floor, (x + 4) / 4. *)
let grid : [ `Symmetric_whole ] Alda.Frames.t =
  Alda.Frames.v ~edge:`Symmetric_whole ~ends:(200, 40) ~window:400 ~hop:160

let win = Alda.Window.hann Nx.float32 Alda.Window.Periodic 400

let power z =
  Nx.add (Nx.square (Nx.real Nx.float32 z)) (Nx.square (Nx.imag Nx.float32 z))

let bank b =
  Alda.Mel.bank Nx.float32 Alda.Mel.Slaney Alda.Unit_area ~rate:16_000. ~n:400 b

let mel b x = Nx.matmul (power (Alda.stft Nx.complex64 grid ~win x)) (bank b)

let log_mel b x =
  let l = Nx.log10 (Nx.maximum_s (mel b x) 1e-10) in
  let l = Nx.maximum l (Nx.sub_s (Nx.max l) 8.) in
  Nx.div_s (Nx.add_s l 4.) 4.

let thirty_seconds = 480_000

(* The golden keeps the frames up to the first floor frame, which every later
   frame equals. *)
let against key tol got =
  let want = entry key in
  let k = Nx.dim 0 want in
  equal (array int) [| 3000; Nx.dim 1 want |] (Nx.shape got);
  let expand =
    Nx.concatenate ~axis:0
      [
        want;
        Nx.broadcast_to
          [| 3000 - k; Nx.dim 1 want |]
          (Nx.slice [ R (k - 1, k) ] want);
      ]
  in
  equal (array (float tol)) (Nx.to_array expand) (Nx.to_array got)

let rows = List.concat_map (fun id -> [ (id, 80); (id, 128) ]) utterances

let parity =
  cases "Whisper's log-mel"
    ~id:(fun (id, b) -> Printf.sprintf "%s.%d" id b)
    rows
    (fun (id, b) ->
      let x =
        Nx.pad [| (0, thirty_seconds - Nx.dim 0 (audio id)) |] 0. (audio id)
      in
      let l = log_mel b x in
      against (Printf.sprintf "%s.%d.ref64" id b) 1e-4 l)

(* Streamed in 10 ms chunks, the frames and their power spectra are the whole
   call's bit for bit; the mel product of one frame at a time rounds by its own
   extent, within the product's bound: all its terms are nonnegative, so a
   different order of its 201 additions moves it by at most γ(201) of its value,
   u = 2^-24. *)
let stream =
  test "a streamed log-mel is the whole call's within the product's rounding"
    (fun () ->
      let x = audio "121-121726-0002" in
      let n = Nx.dim 0 x in
      let s = ref Alda.Frames.start and frames = ref [] in
      for j = 0 to (n / 160) - 1 do
        let s', f =
          Alda.Frames.feed grid !s (Nx.slice [ R (160 * j, 160 * (j + 1)) ] x)
        in
        s := s';
        frames := f :: !frames
      done;
      let s', f =
        Alda.Frames.feed grid !s (Nx.slice [ R (160 * (n / 160), n) ] x)
      in
      let f =
        Nx.concatenate ~axis:0
          (List.rev (Alda.Frames.finish grid s' :: f :: !frames))
      in
      let p_stream = power (Nx.rfft Nx.complex64 (Nx.mul f win)) in
      let p_whole = power (Alda.stft Nx.complex64 grid ~win x) in
      equal (array int64)
        (Array.map Int64.bits_of_float
           (Nx.to_array (Nx.cast Nx.float64 p_whole)))
        (Array.map Int64.bits_of_float
           (Nx.to_array (Nx.cast Nx.float64 p_stream)));
      let per_frame =
        Nx.concatenate ~axis:0
          (List.init (Nx.dim 0 p_stream) (fun i ->
               Nx.matmul (Nx.slice [ R (i, i + 1) ] p_stream) (bank 80)))
      in
      let whole = Nx.to_array (mel 80 x) in
      let u = Float.ldexp 1. (-24) in
      let gamma = 201. *. u /. (1. -. (201. *. u)) in
      Array.iteri
        (fun i v ->
          equal
            ~msg:(Printf.sprintf "entry %d" i)
            (float_rel ~rel:(2. *. gamma) ~abs:1e-30)
            whole.(i) v)
        (Nx.to_array per_frame))

let () = exit (run "alda.whisper" [ parity; stream ])
