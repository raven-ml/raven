(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Codecs over scalar codecs planted in a Hub cache: a codebook of 8 edges per
   channel, for Mag_g one that ends in NaN edges, as seven of HSC's do, and for
   redshift a grid of 1024 edges, as AION-1's. *)

open Windtrap
open Aion

let book = Nx.create Nx.float32 [| 8 |] [| -3.; -1.; 0.; 0.5; 1.; 2.; 4.; 8. |]

let nan_book =
  Nx.create Nx.float32 [| 8 |]
    [| -3.; -1.; 0.; 0.5; 1.; 2.; Float.nan; Float.nan |]

let grid =
  Nx.create Nx.float32 [| 1024 |]
    (Array.init 1024 (fun i -> 6. *. float_of_int i /. 1023.))

(* A cache holding Flux_g's and Mag_g's codebooks, Xp_bp's 55 and redshift's
   grid. *)
let cache_dir =
  let dir = Filename.temp_dir "aion" "codecs" in
  let plant m entries =
    Hub_cache.plant dir (Hub.file (Scalar m))
      (Nx_io.Archive.of_list (List.map (fun (k, v) -> (k, Nx.P v)) entries))
  in
  plant Flux_g [ ("_quantizer._index_to_val", book) ];
  plant Mag_g [ ("_quantizer._index_to_val", nan_book) ];
  plant Z [ ("_quantizer.buckets", grid) ];
  plant Xp_bp
    (List.init 55 (fun i ->
         (Printf.sprintf "_quantizer.quantizers.%d._index_to_val" i, book)));
  dir

let codecs =
  Codecs.load ~cache_dir ~offline:true
    Modality.[ Scalar Flux_g; Scalar Xp_bp; Scalar Mag_g; Scalar Z ]

let ints x = Array.map Int64.to_int (Nx.to_array x)
let presence (i : _ Tokens.t) = Nx.to_array i.present

let value =
  Gen.(
    frequency
      [
        (6, float_range (-5.) 10.);
        (1, of_list [ Float.nan; Float.infinity; Float.neg_infinity; -0.; 0. ]);
      ])

(* An object with a non-finite value has its modality absent, unless the value's
   codebook holds NaN edges: it was trained with NaN as a value, and every value
   keeps its token. *)
let absent =
  prop "a non-finite value is absent unless its codebook has NaN edges"
    Gen.(
      let* modality = of_list Modality.[ Flux_g; Xp_bp; Mag_g ] in
      let c = if modality = Xp_bp then 55 else 1 in
      let* b = int_range 1 4 in
      let+ v = array ~size:(constant (b * c)) value in
      (modality, b, c, v))
    (fun (modality, b, c, v) ->
      let finite = Float.is_finite in
      let row i = Array.sub v (i * c) c in
      let ok i = modality = Mag_g || Array.for_all finite (row i) in
      let bad = not (Array.for_all finite v) in
      cover "a NaN" (Array.exists Float.is_nan v);
      cover "an infinity" (Array.exists (fun x -> Float.abs x = infinity) v);
      cover "a non-finite value on NaN edges" (modality = Mag_g && bad);
      cover "every object present" (List.for_all ok (List.init b Fun.id));
      cover "XP with one bad coefficient"
        (c = 55 && (not (ok 0)) && Array.exists finite (row 0));
      let shape = if c = 1 then [| b |] else [| b; c |] in
      let values = Nx.create Nx.float32 shape v in
      let input = Codecs.encode codecs (Scalar { modality; values }) in
      equal (array int) [| b; c |] (Nx.shape input.tokens);
      equal (array int) [| b; c |] (Nx.shape input.present);
      let expected = Array.init (b * c) (fun k -> ok (k / c)) in
      equal (array bool) expected (presence input);
      let direct =
        ints (Scalar.encode (Codecs.scalar codecs modality) values)
      in
      let got = ints input.tokens in
      Array.iteri (fun k p -> if p then equal int direct.(k) got.(k)) expected)

(* Compiled with the codecs an argument: a scalar codec's search and gathers,
   and its presence, compute exactly under Rune.jit on the host, so tokens,
   presence and values are the eager ones bit for bit. *)
let encode_c =
  Rune.jit
    Nx.Ptree.(
      instantiate (module Codecs)
      @-> v Codecs.walk_data
      @-> returns (v Tokens.walk))
    Codecs.encode

let decode_c =
  Rune.jit
    Nx.Ptree.(
      instantiate (module Codecs)
      @-> v Modality.walk @-> tensor
      @-> returns (v Codecs.walk_data))
    Codecs.decode

let others =
  group "codecs"
    [
      test "decode gives each token's value, [b] for one token" (fun () ->
          let k = Nx.create Nx.int64 [| 3; 1 |] [| 0L; 4L; 7L |] in
          match Codecs.decode codecs (Scalar Flux_g) k with
          | Scalar { modality = Flux_g; values } ->
              equal (array int) [| 3 |] (Nx.shape values);
              equal
                (array (float_rel ~rel:1e-6 ~abs:0.))
                [| exp (-3.); exp 1.; exp 8. |]
                (Nx.to_array (Nx.cast Nx.float64 values))
          | _ -> fail "not Flux_g's values");
      test "redshift's class 1024 decodes to NaN, 1025 and -1 are refused"
        (fun () ->
          let k = Nx.create Nx.int64 [| 3; 1 |] [| 0L; 1023L; 1024L |] in
          (match Codecs.decode codecs (Scalar Z) k with
          | Scalar { modality = Z; values } ->
              equal (array float_exact) [| 0.; 6.; Float.nan |]
                (Nx.to_array (Nx.cast Nx.float64 values))
          | _ -> fail "not redshift's values");
          List.iter
            (fun t ->
              let k = Nx.create Nx.int64 [| 2; 1 |] [| 0L; t |] in
              raises_match (Exn.invalid_arg ~sub:"outside [0, 1025)") (fun () ->
                  Codecs.decode codecs (Scalar Z) k))
            [ 1025L; -1L ]);
      test "an image's bands give its modality" (fun () ->
          let flux = Nx.zeros Nx.float32 [| 1; 2; 96; 96 |] in
          let image bands = Codecs.Image { bands; flux } in
          equal string "tok_image"
            (Modality.key (Codecs.modality (image [ Des_g; Des_r ])));
          equal string "tok_image_hsc"
            (Modality.key (Codecs.modality (image [ Hsc_g; Hsc_y ])));
          raises_match (Exn.invalid_arg ~sub:"no one survey") (fun () ->
              Codecs.modality (image [ Hsc_g; Des_g ])));
      test "a modality without its codec is refused, eagerly and compiled"
        (fun () ->
          let d =
            Codecs.Scalar
              { modality = Flux_r; values = Nx.zeros Nx.float32 [| 1 |] }
          in
          raises_match (Exn.invalid_arg ~sub:"tok_flux_r") (fun () ->
              Codecs.encode codecs d);
          raises_match (Exn.invalid_arg ~sub:"tok_flux_r") (fun () ->
              encode_c codecs d));
      test "load refuses the segmentation map" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"tok_segmap") (fun () ->
              Codecs.load ~cache_dir ~offline:true [ Modality.Segmap ]));
    ]

let values_of = function
  | Codecs.Scalar { values; _ } -> values
  | _ -> failwith "not a scalar's values"

let compiled =
  prop "compiled encode and decode are the eager ones, bit for bit"
    Gen.(
      let* m = of_list Modality.[ Flux_g; Xp_bp; Mag_g; Z ] in
      let c = if m = Xp_bp then 55 else 1 in
      (* The codebook's length, and the first token past it. *)
      let n = if m = Z then 1024 else 8 in
      let* v = array ~size:(constant (4 * c)) value in
      let+ k =
        array
          ~size:(constant (4 * c))
          (frequency [ (3, int_range 0 (n - 1)); (1, constant n) ])
      in
      (m, c, n, v, k))
    (fun (m, c, n, v, k) ->
      cover "a codebook of NaN edges" (m = Mag_g);
      cover "an absent value"
        (m <> Mag_g && Array.exists (fun x -> not (Float.is_finite x)) v);
      cover "a token past the codebook" (Array.exists (fun k -> k >= n) k);
      let shape = if c = 1 then [| 4 |] else [| 4; c |] in
      let d =
        Codecs.Scalar { modality = m; values = Nx.create Nx.float32 shape v }
      in
      let e = Codecs.encode codecs d and ec = encode_c codecs d in
      equal (array int) (ints e.tokens) (ints ec.tokens);
      equal (array bool) (presence e) (presence ec);
      let k = Nx.create Nx.int64 [| 4; c |] (Array.map Int64.of_int k) in
      let bits x = Array.map Int32.bits_of_float (Nx.to_array (values_of x)) in
      equal (array int32)
        (bits (Codecs.decode codecs (Scalar m) k))
        (bits (decode_c codecs (Scalar m) k)))

let () = exit (run "aion.codecs" [ absent; others; compiled ])
