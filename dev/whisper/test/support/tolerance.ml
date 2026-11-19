(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let u64 = Float.ldexp 1. (-53)
let repeat n l = List.concat (List.init n (fun _ -> l))

let encoder (d : Whisper.Model.dims) =
  let w = d.n_audio_state in
  let head = w / d.n_audio_head in
  [ 3 * d.n_mels; 3 * w ]
  @ repeat d.n_audio_layer [ w; w; head; d.n_audio_ctx; w; w; w; 4 * w ]
  @ [ w ]

let decoder (d : Whisper.Model.dims) ~tokens =
  let w = d.n_text_state in
  let head = w / d.n_text_head in
  encoder d @ [ d.n_audio_state ]
  @ repeat d.n_text_layer
      [ w; w; head; tokens; w; w; w; head; d.n_audio_ctx; w; w; w; 4 * w ]
  @ [ w; w ]

let same depths ~n y =
  let s = List.fold_left (fun s k -> s +. Float.sqrt (float k)) 0. depths in
  2. *. Float.sqrt (Float.max 1. (Float.log (float n))) *. u64 *. s *. y

let c = 16.
