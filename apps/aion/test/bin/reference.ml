(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* reference.exe GAIA ROW INPUT TARGET OUT: the softmax of the decoder's logits
   for the target TARGET of row ROW of the Gaia file GAIA from its INPUT, by
   Survey, Codecs and Transformer directly, with aion-base's weights from
   Kaun_hf's cache: OUT, an npz file whose p is float64 [c; vocab].

   reference.exe generate GAIA INPUT TARGET SEED REVEAL DRAWS BATCH OUT: the
   draws of TARGET from INPUT of the Gaia file's rows, by Roar.sample directly:
   the rows in batches of BATCH, each batch's objects with a present token, and
   draw j of batch b from the key [fold_in (fold_in (key SEED) b) j] with the
   counts REVEAL, comma-separated. OUT's tokens is int64 [n; DRAWS; k]. *)

open Aion

let modality key = List.find (fun m -> Modality.key m = key) Modality.all

let posterior file row input target out =
  let source = Survey.load Gaia file in
  let codecs = Codecs.load [ input; target ] in
  let tokens = Codecs.encode codecs (Survey.read source [| row |] input) in
  let t = Transformer.of_archive (Hub.weights Base) in
  let logits =
    List.hd
      (Transformer.decode t
         (Transformer.encode t [ tokens ])
         [ { modality = target; positions = None } ])
  in
  let p = Nx.softmax ~axes:[ -1 ] (Nx.cast Nx.float64 logits) in
  Nx_io.save_npz out
    (Nx_io.Archive.of_list [ ("p", Nx.P (Nx.squeeze ~axes:[ 0 ] p)) ])

let generate file input target ~seed ~reveal ~draws ~batch out =
  let source = Survey.load Gaia file in
  let codecs = Codecs.load [ input ] in
  let t = Transformer.of_archive (Hub.weights Base) in
  let n = Survey.length source in
  let batches = (n + batch - 1) / batch in
  let draws_of b =
    let rows =
      Array.init (min batch (n - (b * batch))) (fun k -> (b * batch) + k)
    in
    let i = Codecs.encode codecs (Survey.read source rows input) in
    let kept =
      let p = Nx.to_array (Nx.any ~axes:[ 1 ] i.present) in
      Array.of_list
        (List.filter (fun k -> p.(k)) (List.init (Array.length p) Fun.id))
    in
    if kept = [||] then None
    else
      let take x =
        Nx.take ~axis:0
          (Nx.create Nx.int64
             [| Array.length kept |]
             (Array.map Int64.of_int kept))
          x
      in
      let i : _ Tokens.t =
        { i with tokens = take i.tokens; present = take i.present }
      in
      let key = Nx.Rng.fold_in (Nx.Rng.key seed) b in
      Some
        (Nx.stack ~axis:1
           (List.init draws (fun j ->
                Roar.sample ~key:(Nx.Rng.fold_in key j) ~reveal t [ i ] target)))
  in
  let tokens = List.filter_map draws_of (List.init batches Fun.id) in
  Nx_io.save_npz out
    (Nx_io.Archive.of_list [ ("tokens", Nx.P (Nx.concatenate ~axis:0 tokens)) ])

let () =
  match Array.to_list Sys.argv |> List.tl with
  | [ "generate"; file; input; target; seed; reveal; draws; batch; out ] ->
      generate file (modality input) (modality target)
        ~seed:(int_of_string seed)
        ~reveal:(List.map int_of_string (String.split_on_char ',' reveal))
        ~draws:(int_of_string draws) ~batch:(int_of_string batch) out
  | [ file; row; input; target; out ] ->
      posterior file (int_of_string row) (modality input) (modality target) out
  | _ ->
      prerr_endline
        "usage: reference.exe GAIA ROW INPUT TARGET OUT\n\
        \       reference.exe generate GAIA INPUT TARGET SEED REVEAL DRAWS \
         BATCH OUT";
      exit 2
