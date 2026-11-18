(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An audio file, named [name], from which [samples] reads. *)
type format = Wav of Wav.t | Flac of Flac.t
type t = { name : string; format : format }

let read name file =
  let head = Storage.string file ~pos:0 (Int.min 12 (Rig.Buffer.length file)) in
  let at i s = String.length head >= i + 4 && String.sub head i 4 = s in
  let format =
    if (at 0 "RIFF" || at 0 "RF64") && at 8 "WAVE" then Wav (Wav.read file)
    else if at 0 "fLaC" then Flac (Flac.read file)
    else if at 0 "RIFX" then
      failwith "the file is a RIFX form, which no WAV specification defines"
    else failwith "the file is neither a WAV nor a FLAC file"
  in
  { name; format }

let rate t = match t.format with Wav w -> Wav.rate w | Flac f -> Flac.rate f

let length t =
  match t.format with Wav w -> Wav.length w | Flac f -> Flac.length f

let channels t =
  match t.format with Wav w -> Wav.channels w | Flac f -> Flac.channels f

let bits t = match t.format with Wav w -> Wav.bits w | Flac f -> Flac.bits f

let channel_mask t =
  match t.format with
  | Wav w -> Wav.channel_mask w
  | Flac f -> Flac.channel_mask f

let samples t =
  Fail.file t.name @@ fun () ->
  match t.format with Wav w -> Wav.samples w | Flac f -> Flac.samples f
