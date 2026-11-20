(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Aion

(* Several suites and rows read one fixture: each file is read once. *)
let loaded = Hashtbl.create 8

let load name =
  match Hashtbl.find_opt loaded name with
  | Some a -> a
  | None ->
      let a = Nx_io.load_safetensors ("upstream/" ^ name ^ ".safetensors") in
      Hashtbl.add loaded name a;
      a

let get a dt name =
  match Nx_io.Archive.find name a with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no fixture entry " ^ name)

let ensemble a = Int64.to_int (Nx.item [ 0 ] (get a Nx.int64 "n"))

let take rows x =
  let rows = Array.of_list (List.map Int64.of_int rows) in
  Nx.take ~axis:0
    (Nx.place (Nx.beside x) (Nx.create Nx.int64 [| Array.length rows |] rows))
    x

let spectra a : _ Spectrum.spectrum =
  {
    flux = get a Nx.float32 "input.flux";
    ivar = get a Nx.float32 "input.ivar";
    mask = Nx.not_equal_s (get a Nx.uint8 "input.mask") 0;
    wavelength = get a Nx.float32 "input.lambda";
  }

let model = lazy (Transformer.of_archive (load "transformer"))
let transformer () = Lazy.force model

let inputs ?(rows = [ 0; 1; 2; 3 ]) ?step ms =
  let input m tokens present : _ Tokens.t =
    { modality = m; tokens = take rows tokens; present = take rows present }
  in
  let fixture m =
    let g = load "transformer_golden" and key = "input." ^ Modality.key m in
    let absent = Nx.not_equal_s (get g Nx.uint8 (key ^ ".absent")) 0 in
    input m (get g Nx.int64 (key ^ ".tokens")) (Nx.logical_not absent)
  in
  let revealed s =
    let r = load "roar_golden" in
    let shown = Nx.not_equal_s (get r Nx.uint8 (s ^ ".revealed")) 0 in
    input (Scalar Xp_rp) (get r Nx.int64 (s ^ ".tokens")) shown
  in
  List.map fixture ms @ Option.to_list (Option.map revealed step)
