(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What the parity suites share: the recorded fixtures, read as JSON, the
   checkpoints they were recorded from, and agreement within a tolerance. *)

open Windtrap

let load name =
  let path = Filename.concat "fixtures" (name ^ ".json") in
  let text = In_channel.with_open_bin path In_channel.input_all in
  match Jsont_bytesrw.decode_string Jsont.json text with
  | Ok j -> j
  | Error e -> failwith (path ^ ": " ^ e)

let decode codec j =
  match Jsont.Json.decode codec j with
  | Ok v -> v
  | Error e -> failwith ("fixture: " ^ e)

let mem name j = decode (Jsont.mem name Jsont.json) j
let find name j = decode (Jsont.mem name ~absent:None (Jsont.some Jsont.json)) j

(* The members of an object, in the file's order. *)
let members j =
  let add _ name v acc = (name, v) :: acc in
  List.rev (decode (Jsont.fold_object Jsont.json add []) j)

let to_list j = decode (Jsont.list Jsont.json) j
let to_number j = decode Jsont.number j
let to_int j = decode Jsont.int j
let to_string j = decode Jsont.string j
let to_string_option j = decode (Jsont.option Jsont.string) j
let to_bool j = decode Jsont.bool j
let floats j = decode (Jsont.array Jsont.number) j
let ints j = decode (Jsont.array Jsont.int) j

(* The repository a fixture was recorded from, at the recorded commit. *)
let repo fx = (to_string (mem "repo" fx), to_string (mem "revision" fx))

let config fx =
  let repo, revision = repo fx in
  Gpt_oss.config_of_json (Kaun_hf.load_config ~revision repo)

let weights fx =
  let repo, revision = repo fx in
  Kaun_hf.load_safetensors ~revision repo

let flat t = Nx.to_array (Nx.reshape [| -1 |] (Nx.cast Nx.float32 t))

(* [close ~tol msg want got]: every value within [tol] of [want]'s largest
   magnitude. A NaN or an infinity fails. *)
let close ~tol msg want got =
  equal ~msg:(msg ^ ": values") int (Array.length want) (Array.length got);
  let scale = Array.fold_left (fun m e -> Float.max m (Float.abs e)) 0. want in
  let worst = ref 0.0 in
  Array.iteri
    (fun i e ->
      let d = Float.abs (e -. got.(i)) /. Float.max scale 1e-30 in
      if not (d <= !worst) then worst := if Float.is_nan d then infinity else d)
    want;
  less float_exact ~msg:(msg ^ ": worst relative error") ~than:tol !worst

(* Equality of every bit, the sign of zero included. *)
let identical msg want got =
  let bits = Array.map Int64.bits_of_float in
  equal ~msg (array int64) (bits want) (bits got)
