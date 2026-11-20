(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The scalar codecs against upstream's goldens (tests/test_data at
   db7fba6df53b) and against the exact transforms of their codebooks.

   upstream/edges.safetensors holds the scalar codecs' codebooks, under their
   directories' names; upstream/goldens.safetensors upstream's inputs, tokens
   and decoded values. Of the Gaia XP codecs both keep the coefficients and
   rows where a token may flip or is the first or the last (the rule is in the
   generator the dune file names). What covers what:

   - tokens: every case of the token checks that the whole goldens reach on
   the kept coefficients is in the kept rows;

   - decoded values: every edge of every codebook is decoded and checked against
   the exact inverse transform, which covers every token any golden holds; the
   goldens add agreement with upstream's own values.

   The whole XP codebooks and goldens add 2.2 MB, so the check of every row and
   coefficient runs outside the repository.

   Tolerances are nx.mli's float32 bounds, in floats from the exact value
   rounded ({!Bounds}): [log] 1, [exp] 2, [asinh] and [sinh] 3, each applied to
   the input it is given. Upstream's float32 functions are held to the same
   bounds: its goldens' [exp] lands up to 2 floats from the rounded value
   (FLUX_Z row 81, FLUX_W2 row 216), beyond the 1 float a faithful library
   gives. A check that assumed faithful libraries on both sides would allow 1
   ulp (log) and 3 ulp (asinh twice) for tokens, and 2 ulp (exp) and (2 + κ₂(1 +
   κ₁))u (sinh twice) for decoding. nx states wider bounds, and the checks here
   take nx's, so they are no tighter than what nx promises. *)

open Windtrap
open Aion

let edges = Nx_io.load_safetensors "upstream/edges.safetensors"
let goldens = Nx_io.load_safetensors "upstream/goldens.safetensors"

let entry name a =
  match Nx_io.Archive.find name a with
  | Some p -> p
  | None -> failwith ("no entry " ^ name)

(* A codec's name in the fixtures: the generator read
   codecs/NAME/model.safetensors, so NAME is the entry whose file is the
   codec's Hub.file. *)
let id m =
  let file name = "codecs/" ^ name ^ "/model.safetensors" in
  match
    List.find_opt
      (fun name -> file name = Hub.file (Scalar m))
      (Nx_io.Archive.names edges)
  with
  | Some name -> name
  | None -> failwith ("no codebook of " ^ Modality.key (Scalar m))

let codec m =
  Scalar.v (Scalar.transform_of m) (Nx.unpack Nx.float32 (entry (id m) edges))

let golden m part = entry (id m ^ "." ^ part) goldens
let ints (Nx.P x) = Nx.to_array (Nx.cast Nx.int64 x) |> Array.map Int64.to_int

(* The edge of token [k] for the [i]th value of a golden, of channel [i mod
   c]. *)
let edge t i k =
  let e = Scalar.edges t in
  Nx.item [ i mod Nx.dim 0 e; k ] e

(* Tokens: a token is upstream's. Under a transform that rounds, it may instead
   move across one distinct edge value [v] that lies where nx's transform may
   land. Each other difference is listed as (index, upstream's token,
   raven's). *)
let tokens =
  cases "tokens are upstream's but where rounding crosses an edge" ~id
    Modality.scalars (fun m ->
      let t = codec m and tr = Scalar.transform_of m in
      let x = Nx.unpack Nx.float32 (golden m "input") in
      let xs = Nx.to_array x in
      let want = ints (golden m "encoded") in
      let got = ints (Nx.P (Scalar.encode t x)) in
      let e = Scalar.edges t in
      let channel c = Nx.to_array (Nx.cast Nx.float64 (Nx.slice [ Nx.I c ] e)) in
      let channels = Array.init (Nx.dim 0 e) channel in
      let refuted = ref [] in
      Array.iteri
        (fun i w ->
          let g = got.(i) in
          let c = channels.(i mod Array.length channels) in
          if not (Scalar_bounds.crosses tr c xs.(i) w g) then
            refuted := (i, w, g) :: !refuted)
        want;
      equal (list (triple int int int)) [] (List.rev !refuted))

(* Decoding every edge: token [k] decodes to the inverse transform of its edge
   within nx's bounds, bit for bit under the identity. Each other value is
   listed as (channel, token, value). *)
let every_edge =
  cases "every token decodes within nx's bound of its edge's inverse" ~id
    Modality.scalars (fun m ->
      let t = codec m and tr = Scalar.transform_of m in
      let e = Scalar.edges t in
      let c = Nx.dim 0 e and n = Nx.dim 1 e in
      let k = Nx.init Nx.int64 [| n; c |] (fun i -> Int64.of_int i.(0)) in
      let d = Nx.to_array (Scalar.decode t k) in
      let refuted = ref [] in
      for k = 0 to n - 1 do
        for ch = 0 to c - 1 do
          let v = Nx.item [ ch; k ] e and got = d.((k * c) + ch) in
          let ok =
            if Float.is_nan v then Float.is_nan got
            else if tr = Scalar.Identity then Float.equal v got
            else Bounds.inside (Scalar_bounds.inverse tr v) got
          in
          if not ok then refuted := (ch, k, got) :: !refuted
        done
      done;
      equal (list (triple int int float_exact)) [] (List.rev !refuted))

(* Decoding the goldens: upstream's tokens decode to upstream's values, bit for
   bit under the identity; otherwise raven's value lies within nx's bound of the
   exact inverse, and so does upstream's. Each other value is listed as (index,
   upstream's, raven's). *)
let values =
  cases "upstream's tokens decode to upstream's values" ~id Modality.scalars
    (fun m ->
      let t = codec m and tr = Scalar.transform_of m in
      let (Nx.P k) = golden m "encoded" in
      let k = Nx.cast Nx.int64 k in
      let ks = Nx.to_array k in
      let want = Nx.to_array (Nx.unpack Nx.float32 (golden m "decoded")) in
      let got = Nx.to_array (Scalar.decode t k) in
      if tr = Scalar.Identity then equal (array float_exact) want got
      else begin
        let refuted = ref [] in
        Array.iteri
          (fun i w ->
            let e = edge t i (Int64.to_int ks.(i)) in
            let bound = Scalar_bounds.inverse tr e in
            if not (Bounds.inside bound got.(i) && Bounds.inside bound w) then
              refuted := (i, w, got.(i)) :: !refuted)
          want;
        equal (list (triple int float_exact float_exact)) [] (List.rev !refuted)
      end)

(* The log codebooks whose lowest edge is the clamp's logarithm: their training
   sample reached the clamp. *)
let at_clamp =
  Modality.
    [ Flux_g; Flux_r; Flux_i; Flux_w1; Flux_w2; Flux_w3; Flux_w4; Shape_r ]

let logs =
  List.filter
    (fun m -> match Scalar.transform_of m with Log _ -> true | _ -> false)
    Modality.scalars

(* A clamped input takes the bin its codebook built for the clamp: every edge
   within nx's bound of the clamp's exact logarithm is nx's float32 logarithm of
   the clamp, bit for bit, and so is the lowest edge of the codebooks whose
   sample reached the clamp, where a clamped input takes token 0. *)
let clamp =
  cases "edges at the log clamp are nx's log of it" ~id logs (fun m ->
      let t = codec m in
      let floor = match Scalar.transform_of m with Log f -> f | _ -> nan in
      let c =
        Nx.item [] (Nx.log (Nx.clamp ~min:floor (Nx.scalar Nx.float32 0.)))
      in
      let near = Scalar_bounds.forward (Log floor) 0. in
      let e = Nx.to_array (Scalar.edges t) in
      Array.iter (fun v -> if Bounds.inside near v then equal float_exact c v) e;
      if List.mem m at_clamp then begin
        equal float_exact c e.(0);
        let zero = Nx.create Nx.float32 [| 1 |] [| 0. |] in
        equal (array int) [| 0 |] (ints (Nx.P (Scalar.encode t zero)))
      end)

let () =
  exit
    (run "aion.parity" [ group "scalar" [ tokens; every_edge; values; clamp ] ])
