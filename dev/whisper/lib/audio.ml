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
let amplitudes (type s) (dt : (float, s) Nx.dtype) (Nx.P x) :
    (float, s, Nx.host) Nx.t =
  let f = Nx.cast dt x in
  match Nx.dtype x with
  | Nx.Dtype.Uint8 -> Nx.div_s (Nx.sub_s f 128.) 128.
  | Nx.Dtype.Int8 -> Nx.div_s f 128.
  | Nx.Dtype.Int16 -> Nx.div_s f 32768.
  | Nx.Dtype.Int32 -> Nx.div_s f 0x1p31
  | Nx.Dtype.Int64 -> Nx.div_s f 0x1p63
  | Nx.Dtype.Float32 | Nx.Dtype.Float64 -> f
  | dt -> failf "Whisper.Audio.load: samples of dtype %s" (Nx.Dtype.name dt)

(* Mixing as ffmpeg's -ac 1 does it, libswresample 7.0 at its defaults
   (rematrix.c, swr_build_matrix2; options.c: centre and surround mix levels √½,
   LFE mix level 0, and a mix to 16-bit samples, which may not pass full scale).
   A channel feeds the speaker of a WAV channel mask's bit [k]: front left and
   right 0 and 1, centre 2, low frequency 3, back left and right 4 and 5, left
   and right of centre 6 and 7, back centre 8, side left and right 9 and 10, top
   centre 11, top front left, centre and right 12 to 14, top back 15 to 17. Into
   a mono output the centre goes whole (as [clev √2] where front left or right
   is mixed too), the front pairs at √½, back and side speakers at the surround
   level times √½, and the rest not at all; weights that sum past 1 are divided
   by their sum. *)
let weight k =
  match k with
  | 2 -> 1.
  | 0 | 1 | 6 | 7 | 12 | 14 -> Float.sqrt 0.5
  | 4 | 5 | 8 | 9 | 10 -> 0.5
  | _ -> 0.

let speakers = 18

(* The masks ffmpeg guesses for 1 to 8 channels whose speakers a file does not
   state (libavutil's av_channel_layout_default): mono, stereo, 2.1, 4.0, 5.0,
   5.1, 6.1 and 7.1. *)
let guesses = [| 0x4; 0x3; 0xB; 0x107; 0x37; 0x3F; 0x70F; 0x63F |]

let bits mask =
  List.filter (fun k -> mask land (1 lsl k) <> 0) (List.init speakers Fun.id)

(* [weights path a] is the weight of each of [a]'s channels in the mix. One
   channel is the mix, whatever its speaker (rematrix.c's clean_layout). *)
let weights path a =
  let c = Nx_io.Audio.channels a in
  if c = 1 then [| 1. |]
  else
    let stated = bits (Nx_io.Audio.channel_mask a) in
    let ks =
      if List.length stated = c then stated
      else if c <= Array.length guesses then bits guesses.(c - 1)
      else failf "%s: %d channels at speakers it does not state" path c
    in
    let w = Array.of_list (List.map weight ks) in
    let total = Array.fold_left ( +. ) 0. w in
    (* rematrix.c's sane_layout refuses a layout without a front speaker. *)
    if total = 0. then
      failf "%s: %d channels at speakers 0x%x, none of them mixed" path c
        (List.fold_left (fun m k -> m lor (1 lsl k)) 0 ks);
    if total > 1. then Array.map (fun v -> v /. total) w else w

(* [mix w x] is the channels of [x], [[rows; c]], weighted by [w] and added in
   order; a channel of weight 0 is skipped, as libswresample skips it. *)
let mix w x =
  let term j =
    if w.(j) = 1. then Nx.slice [ A; I j ] x
    else Nx.mul_s (Nx.slice [ A; I j ] x) w.(j)
  in
  match
    List.filter (fun j -> w.(j) <> 0.) (List.init (Array.length w) Fun.id)
  with
  | [] -> invalid_arg "Whisper.Audio.mix: no channel weighs"
  | j :: js -> List.fold_left (fun y j -> Nx.add y (term j)) (term j) js

(* Resampling as ffmpeg's -ar does it, libswresample 7.0 at its defaults
   (resample.c's resample_init, build_filter and resample_linear;
   resample_template.c; options.c): a Kaiser-windowed sinc, β 9, cut at 0.97 of
   the lower Nyquist frequency, over [taps] samples at the lower rate, every
   phase's weights divided by phase 0's sum. resample_init computes the cut and
   the taps at float64 from the raw rates, [min (16000 * 0.97 / r) 1] and [ceil
   (32 / factor)] aligned up to even; where [32 / factor] lies near an integer
   another order of operations rounds to the other side of it.

   Output [j] lies at input position [j r / 16000]. Where [16000 / r] reduces to
   [l] phases, at most 1024 ([exact_rational], [phase_shift] 10), swr weighs
   each output by its exact phase: omr's resampling. Otherwise swr keeps 1024
   phases and the one after and interpolates linearly between the two phases
   about the output's position ([linear_interp]).

   Before the signal swr reads its first [taps] samples mirrored about the
   first; after it, [(min n taps + 1) / 2] samples mirrored about the point half
   a sample past the last, then zeros (swr_convert's flush). *)

let kaiser_beta = 9.
let filter_size = 32.
let cutoff = 0.97
let phases = 1024

(* A resampling's weights. [Exact]'s [lead] zeros go before the mirror so that
   the signal's first output, output [first] of the extended signal, is one of
   omr's outputs, which [w0] then divides. [Interpolated] holds the [phases]
   rows of [taps] weights. *)
type bank =
  | Exact of { r : Omr.Resample.t; lead : int; first : int; w0 : float }
  | Interpolated of float array array

type swr = { from : int; taps : int; bank : bank }

(* [rows kernel ~s ~taps] is swr's bank of [phases] rows (build_filter): row
   [k]'s tap [q] weighs the input [q - (taps / 2 - 1) - k / phases] samples
   after the output, at the kernel, every row divided by row 0's sum. *)
let rows kernel ~s ~taps =
  let centre = (taps / 2) - 1 in
  (* Row by row, so that the kernel's intermediates stay the size of one. *)
  let row k =
    let tau =
      Nx.init Nx.float64 [| taps |] (fun i ->
          let d =
            Float.of_int (i.(0) - centre)
            -. (Float.of_int k /. Float.of_int phases)
          in
          -.d /. s)
    in
    Nx.to_array (Omr.Kernel.eval kernel tau)
  in
  let rows = Array.init phases row in
  let norm = Array.fold_left ( +. ) 0. rows.(0) in
  Array.iter (fun w -> Array.iteri (fun q v -> w.(q) <- v /. norm) w) rows;
  rows

let swr from =
  let factor =
    Float.min (Float.of_int rate *. cutoff /. Float.of_int from) 1.
  in
  let taps = Float.to_int (Float.ceil (filter_size /. factor)) in
  let taps = taps + (taps land 1) in
  let s = Float.max 1. (Float.of_int from /. Float.of_int rate) in
  let kernel =
    Omr.Kernel.Sinc
      {
        cutoff = factor *. s /. 2.;
        half_width = Float.of_int taps /. 2. /. s;
        window = Omr.Kernel.Kaiser kaiser_beta;
      }
  in
  let rec gcd a b = if b = 0 then a else gcd b (a mod b) in
  let g = gcd from rate in
  let l = rate / g and m = from / g in
  let bank =
    if l > phases then Interpolated (rows kernel ~s ~taps)
    else
      let r = Omr.Resample.v ~grid:`Points ~edge:`Zero ~kernel ~by:(l, m) in
      let w0 = Array.fold_left ( +. ) 0. (snd (Omr.Resample.weights r).(0)) in
      let lead = (m - (taps mod m)) mod m in
      Exact { r; lead; first = (lead + taps) * l / m; w0 }
  in
  { from; taps; bank }

(* A consumer of the extended signal from position [-taps]: [feed] takes its
   next samples and [finish] its end, past which it reads zeros. It writes
   outputs [0] to [total - 1] of the signal with [write j y], [y] outputs [j]
   onward. *)
type consumer = { feed : Nx.host Nx.float64_t -> unit; finish : unit -> unit }

let exact ~total r ~lead ~first ~w0 write =
  let module R = Omr.Resample in
  let state = ref R.start and next = ref 0 in
  (* Keep outputs [first] to [first + total - 1] of the extended signal. *)
  let emit y =
    let k = Nx.dim 0 y in
    let lo = Int.max 0 (first - !next) in
    let hi = Int.min k (first + total - !next) in
    if hi > lo then
      write (!next + lo - first) (Nx.div_s (Nx.slice [ R (lo, hi) ] y) w0);
    next := !next + k
  in
  let feed x =
    let s, y = R.feed r !state x in
    state := s;
    emit y
  in
  feed (Nx.zeros Nx.float64 [| lead |]);
  { feed; finish = (fun () -> emit (R.finish r !state)) }

(* Output [j] lies [k + α] phases past input [b = ⌊j r / 16000⌋], [k] whole. It
   is [v1 + (v2 - v1) α] (resample_linear's order) of [v1], the [taps] samples
   from [b - (taps / 2 - 1)] weighed by row [k], and [v2], by row [k + 1]. Row
   [phases] is row 0 a sample later, the [taps] samples from one further on:
   resample_init copies row 0 there shifted by one element, and the x86 AVX
   kernels read its [taps]-th element, row 0's last tap, from the allocation of
   [taps] rounded up to 8. Where [taps] is a multiple of 8, or on kernels
   without AVX where it is a multiple of 4, libswresample's row 1024 loses that
   tap, of weight about 4e-6, or moves it [taps] samples back. [buf] holds the
   samples from position [base] that later outputs read. *)
let interpolated ~total ~from ~taps rows write =
  let centre = (taps / 2) - 1 in
  let start j = (j * from / rate) - centre in
  let buf = ref [||] and base = ref (-taps) and next = ref 0 in
  let dot b at w =
    let v = ref 0. in
    for q = 0 to taps - 1 do
      v := !v +. (b.(at + q) *. w.(q))
    done;
    !v
  in
  (* Writes the outputs whose samples lie below position [known]. *)
  let run known =
    let j0 = !next in
    let j1 = ref j0 in
    while !j1 < total && start !j1 + taps + 1 <= known do
      incr j1
    done;
    let b = !buf in
    let y =
      Array.init (!j1 - j0) (fun i ->
          let j = j0 + i in
          let u = j * from mod rate * phases in
          let k = u / rate and at = start j - !base in
          let v1 = dot b at rows.(k) in
          let v2 =
            if k + 1 = phases then dot b (at + 1) rows.(0)
            else dot b at rows.(k + 1)
          in
          v1 +. ((v2 -. v1) *. (Float.of_int (u mod rate) /. Float.of_int rate)))
    in
    if !j1 > j0 then write j0 (Nx.create Nx.float64 [| !j1 - j0 |] y);
    next := !j1;
    let keep = Int.max !base (Int.min known (start !next)) in
    buf := Array.sub b (keep - !base) (known - keep);
    base := keep
  in
  let feed x =
    buf := Array.append !buf (Nx.to_array x);
    run (!base + Array.length !buf)
  in
  let finish () =
    if !next < total then begin
      let known = !base + Array.length !buf in
      let ends = start (total - 1) + taps + 1 in
      if ends > known then
        buf := Array.append !buf (Array.make (ends - known) 0.);
      run (Int.max known ends)
    end
  in
  { feed; finish }

(* [resampled sw ~n chunks write] streams the [n] samples given as [chunks]
   through [sw], calling [write j y] with outputs [j] onward. It holds the first
   [taps + 1] samples until they are known, to mirror them, and the last [(min n
   taps + 1) / 2], to mirror them at the end. *)
let resampled sw ~n chunks write =
  let total = ((n * rate) + sw.from - 1) / sw.from in
  let c =
    match sw.bank with
    | Exact { r; lead; first; w0 } -> exact ~total r ~lead ~first ~w0 write
    | Interpolated rows ->
        interpolated ~total ~from:sw.from ~taps:sw.taps rows write
  in
  let ends = (Int.min n sw.taps + 1) / 2 in
  (* The last [ends] samples of [x]. *)
  let last x =
    let k = Nx.dim 0 x in
    Nx.slice [ R (k - Int.min ends k, k) ] x
  in
  let head = ref (Some (Nx.zeros Nx.float64 [| 0 |])) in
  let tail = ref (Nx.zeros Nx.float64 [| 0 |]) in
  let push x =
    tail :=
      last
        (if Nx.dim 0 x >= ends then x else Nx.concatenate ~axis:0 [ !tail; x ]);
    match !head with
    | None -> c.feed x
    | Some h ->
        let h = Nx.concatenate ~axis:0 [ h; x ] in
        if Nx.dim 0 h < Int.min n (sw.taps + 1) then head := Some h
        else begin
          head := None;
          c.feed
            (Nx.slice
               [ R (0, sw.taps) ]
               (Nx.extend `Symmetric_whole (sw.taps, 0) h));
          c.feed h
        end
  in
  Seq.iter push chunks;
  (* The mirror past the end, about the point half a sample past the last. *)
  c.feed (Nx.flip !tail);
  c.finish ()

let block = 1 lsl 16

(* The largest rate FLAC's STREAMINFO states, 20 bits, above every PCM rate in
   use. A resampling's weights grow with the rate, [1024 taps], [taps] about [r
   / 485], so a rate past it would let a header ask for gigabytes. *)
let max_rate = (1 lsl 20) - 1

let load_rows ~rows path =
  let a = Nx_io.load_audio path in
  let r = Nx_io.Audio.rate a and n = Nx_io.Audio.length a in
  if r <= 0 then failf "%s: a rate of %d Hz" path r;
  if r > max_rate then
    failf "%s: a rate of %d Hz, past the %d Hz Whisper reads" path r max_rate;
  let w = weights path a in
  let blocks = Nx_io.Audio.blocks a ~rows in
  (* [⌈n rate / r⌉] samples, written as they complete. *)
  let out = ref (Nx.zeros Nx.float32 [| ((n * rate) + r - 1) / r |]) in
  let write j y =
    let y = Nx.cast Nx.float32 y in
    out := Nx.set [ R (j, j + Nx.dim 0 y) ] y (Nx.donate !out)
  in
  (* [each f] writes [f] of each block, consecutively. *)
  let each f =
    ignore
      (Seq.fold_left
         (fun j p ->
           let y = f p in
           write j y;
           j + Nx.dim 0 y)
         0 blocks)
  in
  let mixed p = mix w (amplitudes Nx.float64 p) in
  if r = rate && Array.length w = 1 then
    (* The samples as stored, each rounded to float32 once. *)
    each (fun p -> Nx.slice [ A; I 0 ] (amplitudes Nx.float32 p))
  else if r = rate then each mixed
  else if n > 0 then resampled (swr r) ~n (Seq.map mixed blocks) write;
  !out

let load path = load_rows ~rows:block path

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
