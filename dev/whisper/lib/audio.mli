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

(** {1:files Files}

    A file's samples load at their dtype's full scale ({!Nx_io.section-audio}).
    Its {e amplitudes} are its samples divided by that scale: a [uint8] sample
    less 128 by 128, an [int8] by 128, an [int16] by 32768, an [int32] by
    [2{^31}] and an [int64] by [2{^63}], a float sample as stored. A sample
    deeper than 16 bits keeps its low bits, and a float peak above 1 stays above
    1. A mixed or resampled sample may pass 1 too, as a full-scale square wave
    rings past it once resampled.

    The channels of a file of several are {e mixed} into one, as FFmpeg's
    libswresample mixes them at its defaults for one channel, which Whisper's
    upstream loader asks. Each channel feeds the speaker
    {!Nx_io.Audio.channel_mask} places it at. Where the mask's speakers do not
    number the channels, they are those FFmpeg guesses for 2 to 8 channels:
    front left and right, then low frequency (3 channels); front centre, back
    centre (4); front centre, back left and right (5); front centre, low
    frequency, back left and right (6); front centre, low frequency, back
    centre, side left and right (7); front centre, low frequency, back left and
    right, side left and right (8). A channel weighs 1 at front centre; √½ at
    front left or right, left or right of centre, or top front left or right; ½
    at back left, right or centre, or side left or right; 0 at the others. Where
    the weights sum past 1 each is divided by their sum, so the mix of two
    channels at front left and right is their mean. A file of one channel is
    that channel, whatever its speaker.

    A signal of [n] samples at a rate [r] other than {!rate} is {e resampled} to
    [⌈16000 n / r⌉] samples at {!rate}, as FFmpeg's libswresample resamples at
    its defaults, which Whisper's upstream loader runs. Its cut is
    [f = min (16000 × 0.97 / r) 1] and its {e length} [T] the least even integer
    at or above [⌈32 / f⌉], both computed at float64 in that order. With
    [s = max 1 (r / 16000)] and the kernel
    [k = Sinc {cutoff = f s / 2; half_width = T / 2s; window = Kaiser 9.}], the
    signal's value at a position [t] is [y(t) = Σᵢ x(i) k((t - i) / s) / w₀],
    over the [T] samples [i] with [-T/2 ≤ t - i < T/2], where [w₀] is
    [Σ k(q / s)] over the whole [q] with [-T/2 ≤ q < T/2]. Output [j] lies at
    [t = j r / 16000 = b + p], [b] whole and [0 ≤ p < 1]. Where [16000 / r]
    reduces to at most 1024 values of [p], output [j] is [y(t)]:
    {!Alda.resample} on [`Points] by the reduced ratio, divided by [w₀].
    Otherwise, with [p = (q + α) / 1024] for a whole [q], it is
    [v₁ + (v₂ - v₁) α] of [v₁ = y(b + q / 1024)] and
    [v₂ = y(b + (q + 1) / 1024)], the linear interpolation between 1024 values
    of [p] that libswresample computes. The signal is extended before its first
    sample by [T] samples mirrored about it ([`Symmetric_whole]), and after its
    last by [(min n T + 1) / 2] samples mirrored about the point half a sample
    past it ([`Symmetric_half]), then by zeros. *)

val load : string -> Nx.host Nx.float32_t
(** [load path] is the WAV or FLAC file [path], read by {!Nx_io.load_audio}, as
    Whisper hears it, of shape [[| n |]]. A file of one channel at {!rate} gives
    its amplitudes, each rounded to float32 once. Another file gives its
    amplitudes mixed and, at another rate, resampled, computed at float64 and
    rounded to float32 once. It reads the file in blocks
    ({!Nx_io.Audio.blocks}), so that memory holds the result, one block and, to
    resample, at most [2048 T] weights at float64; [T] grows with the rate the
    file states, about [r / 485].

    Raises [Failure] whose message starts with [path] if the file's rate is 0,
    or past 1048575 Hz ([2{^20} - 1], the largest rate a FLAC file states, above
    every rate in use), naming it, before any weight is built, so that a header
    cannot ask for the memory of a larger one; if it holds more than 8 channels
    at speakers it does not state, naming them, or if its channels' speakers all
    weigh 0, naming the mask; and as {!Nx_io.load_audio} and
    {!Nx_io.Audio.blocks} do. *)

(** {1:features Features} *)

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

(**/**)

val load_rows : rows:int -> string -> Nx.host Nx.float32_t
(** [load_rows ~rows path] is {!load}[ path], reading [rows] rows of the file at
    a time. *)
