(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Encoding *)

(* Writes row [i] of [e] to row [pixel.{i}] of [out], for each [i] holding a
   pixel. *)
let scatter (out : (float, Bigarray.float32_elt) Flat.t) ~dims e
    (b : Nx.host Batch.t) =
  Flat.read b.pixel @@ fun pixel ->
  Flat.read e @@ fun (e : (float, Bigarray.float32_elt) Flat.t) ->
  for i = 0 to Bigarray.Array1.dim pixel - 1 do
    let p = Int64.to_int pixel.{i} in
    if p >= 0 then
      for c = 0 to dims - 1 do
        out.{(p * dims) + c} <- e.{(i * dims) + c}
      done
  done

(* Every batch, the last included, goes through [f], so that a tile whose pixels
   all lack an embedding still learns the embedding's length. *)
let encode ~shape at f px =
  let n = Pixels.length px in
  let counts = Batch.v2_counts px in
  let pack from = Batch.pack shape px counts ~from in
  let out = ref None in
  let rec go (b, next) =
    let e = f (Batch.place at b) in
    let following = if next < n then Some (pack next) else None in
    let e = Nx.place Nx.Host.on e in
    let dims = Nx.dim 1 e in
    let o =
      match !out with
      | Some (o, _) -> o
      | None ->
          let o = Flat.create Bigarray.float32 (n * dims) Float.nan in
          out := Some (o, dims);
          o
    in
    scatter o ~dims e b;
    Option.iter go following
  in
  go (pack 0);
  let o, dims = Option.get !out in
  Flat.wrap Nx.float32 [| n; dims |] o

(* Quantisation *)

(* GeoTessera's codes (tessera_infer_v2 student/quantize.py:37-50), whose
   [np.round] takes halves to the even neighbour. [Nx.round] takes them away
   from zero, so a half is rounded as twice the rounding of its half, which is
   exact in binary. *)
let round_even q =
  let tie = Nx.equal_s (Nx.abs (Nx.sub q (Nx.trunc q))) 0.5 in
  Nx.where tie (Nx.mul_s (Nx.round (Nx.div_s q 2.)) 2.) (Nx.round q)

let qmax = 127.
let absmax_floor = 1e-12

let quantize e =
  let absmax = Nx.maximum_s (Nx.max ~axes:[ -1 ] (Nx.abs e)) absmax_floor in
  let scale = Nx.div_s absmax qmax in
  let q = Nx.div e (Nx.unsqueeze ~axes:[ -1 ] scale) in
  let codes = Nx.clamp ~min:(-.qmax) ~max:qmax (round_even q) in
  (Nx.cast Nx.int8 codes, scale)

(* Files *)

let grid_shape name (h, w) e =
  if Nx.ndim e <> 2 || Nx.dim 0 e <> h * w then
    invalid_arg
      (Printf.sprintf "Tessera.Embedding.%s: %d rows, %d × %d expected" name
         (Nx.dim 0 e) h w);
  [| h; w; Nx.dim 1 e |]

let save name ~grid e =
  let shape = grid_shape "save" grid e in
  Nx_io.save_npy (name ^ ".npy") (Nx.reshape shape e)

let save_int8 name ~grid e =
  let shape = grid_shape "save_int8" grid e in
  let codes, scales = quantize e in
  Nx_io.save_npy (name ^ ".npy") (Nx.reshape shape codes);
  Nx_io.save_npy (name ^ "_scales.npy")
    (Nx.reshape [| shape.(0); shape.(1) |] scales)
