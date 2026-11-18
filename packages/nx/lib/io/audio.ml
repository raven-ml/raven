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

(* [named name s] is [s], each node's [Failure] naming the file. *)
let rec named name s () =
  match Fail.file name s with
  | Seq.Nil -> Seq.Nil
  | Seq.Cons (x, s) -> Seq.Cons (x, named name s)

let blocks ?(row = 0) t ~rows =
  let n = length t in
  if rows < 1 then
    Fail.invalid_argf "Nx_io.Audio.blocks: blocks of %d rows" rows;
  if row < 0 || row > n then
    Fail.invalid_argf "Nx_io.Audio.blocks: row %d is outside the %d rows" row n;
  match t.format with
  | Flac f -> named t.name (Flac.blocks f ~row ~rows)
  | Wav w ->
      (* Each block reads its rows' bytes alone. *)
      let rec from row () =
        if row >= n then Seq.Nil
        else
          let m = Int.min rows (n - row) in
          Seq.Cons (Wav.window w ~row ~rows:m, from (row + m))
      in
      named t.name (from row)

let window t ~row ~rows =
  let n = length t in
  if row < 0 || rows < 0 || row > n - rows then
    Fail.invalid_argf
      "Nx_io.Audio.window: %d rows from row %d lie outside the %d rows" rows row
      n;
  Fail.file t.name @@ fun () ->
  match t.format with
  | Wav w -> Wav.window w ~row ~rows
  | Flac f -> Flac.window f ~row ~rows
