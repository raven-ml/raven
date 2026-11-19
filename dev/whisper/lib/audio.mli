(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Whisper's input: a sound as the log-mel spectrogram its encoder reads.

    Whisper hears one channel sampled at 16 kHz, of full scale 1, as a
    {e log-mel spectrogram}: a frame of [n_mels] bands every 160 samples,
    computed in float32. Of [n] samples:
    + frame [m], for [m] from 0 to [n / 160 - 1], holds the samples
      [160 m - 200] to [160 m + 199], those before the first sample mirrored
      about it and those past the last mirrored about it, as often as a short
      signal needs; fewer than 160 samples give no frame;
    + each frame, times the periodic Hann window of 400 points, gives the power
      [re² + im²] of the 201 bins of its real Fourier transform;
    + the bins are summed into bands by the matrix product with Slaney's mel
      filterbank of unit-area triangles from 0 to 8000 Hz, computed at float64
      and rounded once;
    + each band's power [p] gives [l = log10 (max p 1e-10)], floored at the
      greatest [l] of all the frames less 8, then scaled to [(l + 4) / 4].

    The floor is the whole call's, so each frame depends on every sample of the
    call. The encoder reads a {e window} of 3000 frames, 30 s, which
    {!pad_or_trim} makes of a recording's samples. A longer recording is read in
    windows of one call's spectrogram, its floor taken over the whole recording;
    [~padding:length] gives its last window the frames to slice. The matrix
    product rounds as its kernel library's contraction does, which may depend on
    the number of frames in the call, so a window sliced from a longer call
    equals the same frames computed by a call of their own within one rounding
    of the product, and may differ from it in the last bit.

    This reads a recording of at most 30 s and encodes it with a model [m]:
    {[
    let x = Whisper.Audio.load "speech.flac" in
    let mel =
      Whisper.Audio.log_mel ~n_mels:m.dims.n_mels (Whisper.Audio.pad_or_trim x)
    in
    Whisper.Model.encode m (Nx.unsqueeze ~axes:[ 0 ] mel)
    ]} *)

val rate : int
(** [rate] is [16_000], the samples a second Whisper hears. *)

val length : int
(** [length] is [480_000], the samples of a window, 30 s at {!rate}. *)

val load : string -> Nx.host Nx.float32_t
(** [load path] is the samples of the WAV or FLAC file [path], read by
    {!Nx_io.load_audio}, as amplitudes of full scale 1, of shape [[| n |]]. An
    integer sample is divided by its dtype's full scale, a [uint8] sample less
    128 by 128, an [int8] by 128, an [int16] by 32768, an [int32] by [2{^31}]
    and an [int64] by [2{^63}], and rounded to float32 once, so a sample deeper
    than 16 bits keeps its low bits. A float sample is rounded to float32 as
    stored, unclipped: a peak above 1 stays above 1.

    Raises [Failure] whose message starts with [path] if the file's rate is not
    {!rate}, naming its rate, or if it holds more than one channel, naming them;
    and as {!Nx_io.load_audio} and {!Nx_io.Audio.samples} do. *)

val pad_or_trim : 'd Nx.float32_t -> 'd Nx.float32_t
(** [pad_or_trim x] is the samples [x], of shape [[| n |]], cut to their first
    {!length} or followed by zeros up to it: one window, shape [[| length |]].

    Raises [Invalid_argument] if [x] has another number of axes than one. *)

val log_mel : ?padding:int -> n_mels:int -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [log_mel ~n_mels x] is the log-mel spectrogram of the samples [x], of shape
    [[| n |]], followed by [padding] zeros, in [n_mels] bands: shape
    [[| (n + padding) / 160; n_mels |]]. [padding] defaults to [0]. It is
    computed on [x]'s set. {!Model.encode} reads 3000 frames of it with an axis
    of windows in front.

    Raises [Invalid_argument] if [x] has another number of axes than one, if
    [padding] is negative, if [n_mels < 1], or if [n_mels] is so large that a
    band weighs no bin. *)
