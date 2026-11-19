(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let rate = 16_000
let length = 30 * rate
let failf fmt = Printf.ksprintf failwith fmt
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Load *)

(* nx.io puts a sample's bits at the top of its dtype, so each dtype's full
   scale is its own (Nx_io's Audio section). The division by a power of two is
   exact, so the cast is the one rounding. *)
let amplitudes (Nx.P x) =
  let f = Nx.cast Nx.float32 x in
  match Nx.dtype x with
  | Nx.Dtype.Uint8 -> Nx.div_s (Nx.sub_s f 128.) 128.
  | Nx.Dtype.Int8 -> Nx.div_s f 128.
  | Nx.Dtype.Int16 -> Nx.div_s f 32768.
  | Nx.Dtype.Int32 -> Nx.div_s f 0x1p31
  | Nx.Dtype.Int64 -> Nx.div_s f 0x1p63
  | Nx.Dtype.Float32 | Nx.Dtype.Float64 -> f
  | dt -> failf "Whisper.Audio.load: samples of dtype %s" (Nx.Dtype.name dt)

let load path =
  let a = Nx_io.load_audio path in
  let r = Nx_io.Audio.rate a in
  if r <> rate then failf "%s: %d Hz, where Whisper hears %d Hz" path r rate;
  let c = Nx_io.Audio.channels a in
  if c <> 1 then failf "%s: %d channels, where Whisper hears one" path c;
  Nx.reshape [| Nx_io.Audio.length a |] (amplitudes (Nx_io.Audio.samples a))

let check_samples fn x =
  if Nx.ndim x <> 1 then
    invalid_argf "Whisper.Audio.%s: samples of shape [%s], not [n]" fn
      (String.concat "; "
         (Array.to_list (Array.map string_of_int (Nx.shape x))))

let pad_or_trim x =
  check_samples "pad_or_trim" x;
  let n = Nx.dim 0 x in
  if n >= length then Nx.slice [ R (0, length) ] x
  else Nx.pad [| (0, length - n) |] 0. x

(* Log-mel *)

let window = 400
let hop = 160

(* Frame [m] centred on sample [hop * m]: [window / 2] samples mirrored before
   the first, and past the end only what the last frame reads, so [n] samples
   give [n / hop] frames, torch's centred frames without the last. *)
let grid =
  Omr.Frames.v ~edge:`Symmetric_whole
    ~ends:(window / 2, (window / 2) - hop)
    ~window ~hop

let floor_decades = 8.

let log_mel ?(padding = 0) ~n_mels x =
  check_samples "log_mel" x;
  if padding < 0 then invalid_argf "Whisper.Audio.log_mel: padding %d" padding;
  let at = Nx.beside x in
  let bank =
    Nx.place at
      (Omr.Mel.bank Nx.float32 Omr.Mel.Slaney Omr.Unit_area ~rate:(float rate)
         ~n:window n_mels)
  in
  let x = if padding = 0 then x else Nx.pad [| (0, padding) |] 0. x in
  (* No frame, no maximum to floor at. *)
  if Omr.Frames.count grid (Nx.dim 0 x) = 0 then
    Nx.zeros_at at Nx.float32 [| 0; n_mels |]
  else
    let win =
      Nx.place at (Omr.Window.hann Nx.float32 Omr.Window.Periodic window)
    in
    let z = Omr.stft Nx.complex64 grid ~win x in
    let power =
      Nx.add
        (Nx.square (Nx.real Nx.float32 z))
        (Nx.square (Nx.imag Nx.float32 z))
    in
    let l = Nx.log10 (Nx.maximum_s (Nx.matmul power bank) 1e-10) in
    let l = Nx.maximum l (Nx.sub_s (Nx.max l) floor_decades) in
    Nx.div_s (Nx.add_s l 4.) 4.
