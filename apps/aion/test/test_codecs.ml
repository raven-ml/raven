(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Codecs over scalar codecs planted in a Hub cache: a codebook of 8 edges per
   channel, and for Mag_g one that ends in NaN edges, as seven of HSC's do. *)

open Windtrap
open Aion

let rec mkdir_p dir =
  if not (Sys.file_exists dir) then begin
    mkdir_p (Filename.dirname dir);
    Sys.mkdir dir 0o755
  end

let book = Nx.create Nx.float32 [| 8 |] [| -3.; -1.; 0.; 0.5; 1.; 2.; 4.; 8. |]

let nan_book =
  Nx.create Nx.float32 [| 8 |] [| -3.; -1.; 0.; 0.5; 1.; 2.; Float.nan; Float.nan |]

(* A cache holding Flux_g's codebook and Xp_bp's 55. *)
let cache_dir =
  let dir = Filename.temp_dir "aion" "codecs" in
  let plant m entries =
    let file = Hub.file (Scalar m) in
    let path = Kaun_hf.cache_path ~cache_dir:dir ~revision:Hub.revision ~file Hub.repo in
    mkdir_p (Filename.dirname path);
    Nx_io.save_safetensors path
      (Nx_io.Archive.of_list (List.map (fun (k, v) -> (k, Nx.P v)) entries))
  in
  plant Flux_g [ ("_quantizer._index_to_val", book) ];
  plant Parallax [ ("_quantizer._index_to_val", book) ];
  plant Mag_g [ ("_quantizer._index_to_val", nan_book) ];
  plant Xp_bp
    (List.init 55 (fun i ->
         (Printf.sprintf "_quantizer.quantizers.%d._index_to_val" i, book)));
  dir

let codecs =
  Codecs.load ~cache_dir ~offline:true
    Modality.[ Scalar Flux_g; Scalar Xp_bp; Scalar Parallax; Scalar Mag_g ]

let ints x = Array.map Int64.to_int (Nx.to_array x)

(* The presence [encode] gives each token, [None] read as all present. *)
let presence (i : _ Tokens.t) =
  match i.present with
  | Some p -> Nx.to_array p
  | None -> Array.make (Array.fold_left ( * ) 1 (Nx.shape i.tokens)) true

let value =
  Gen.(
    frequency
      [
        (6, float_range (-5.) 10.);
        ( 1,
          of_list [ Float.nan; Float.infinity; Float.neg_infinity; -0.; 0. ] );
      ])

let absent =
  group "absent values"
    [
      prop "an object with a non-finite value has its modality absent"
        Gen.(
          let* b = int_range 1 4 in
          let* c = of_list [ 1; 55 ] in
          let+ v = array ~size:(constant (b * c)) value in
          (b, c, v))
        (fun (b, c, v) ->
          let finite = Float.is_finite in
          let row i = Array.sub v (i * c) c in
          let ok i = Array.for_all finite (row i) in
          cover "a NaN" (Array.exists Float.is_nan v);
          cover "an infinity" (Array.exists (fun x -> Float.abs x = Float.infinity) v);
          cover "every object present" (List.for_all ok (List.init b Fun.id));
          cover "XP with one bad coefficient" (c = 55 && not (ok 0) && Array.exists finite (row 0));
          let modality, shape =
            if c = 1 then (Modality.Flux_g, [| b |]) else (Modality.Xp_bp, [| b; c |])
          in
          let values = Nx.create Nx.float32 shape v in
          let input = Codecs.encode codecs (Scalar { modality; values }) in
          let expected = Array.init (b * c) (fun k -> ok (k / c)) in
          equal (array bool) expected (presence input);
          let direct = ints (Scalar.encode (Codecs.scalar codecs modality) values) in
          let got = ints input.tokens in
          Array.iteri (fun k p -> if p then equal int direct.(k) got.(k)) expected);
    ]

(* A codebook that ends in NaN edges was trained with NaN as a value, so a
   value there keeps upstream's token; any other codebook makes it absent. *)
let nan_edges =
  cases "a non-finite value is absent unless its codebook has NaN edges"
    ~id:(fun (n, _, _) -> n)
    [
      ("HSC's Mag_g, NaN", Modality.Mag_g, Float.nan);
      ("HSC's Mag_g, +inf", Modality.Mag_g, Float.infinity);
      ("Gaia's parallax, NaN", Modality.Parallax, Float.nan);
    ]
    (fun (_, modality, v) ->
      let values = Nx.create Nx.float32 [| 2 |] [| 0.25; v |] in
      let input = Codecs.encode codecs (Scalar { modality; values }) in
      let upstream = ints (Scalar.encode (Codecs.scalar codecs modality) values) in
      match modality with
      | Mag_g ->
          equal (array bool) [| true; true |] (presence input);
          equal (array int) upstream (ints input.tokens)
      | _ -> equal (array bool) [| true; false |] (presence input))

let others =
  group "codecs"
    [
      test "decode gives each token's value" (fun () ->
          let k = Nx.create Nx.int64 [| 3 |] [| 0L; 4L; 7L |] in
          match Codecs.decode codecs (Scalar Flux_g) k with
          | Scalar { modality = Flux_g; values } ->
              equal
                (array (float_rel ~rel:1e-6 ~abs:0.))
                [| exp (-3.); exp 1.; exp 8. |]
                (Nx.to_array (Nx.cast Nx.float64 values))
          | _ -> fail "not Flux_g's values");
      test "an image's bands give its modality" (fun () ->
          let flux = Nx.zeros Nx.float32 [| 1; 2; 96; 96 |] in
          let image bands = Codecs.Image { bands; flux } in
          equal string "tok_image" (Modality.key (Codecs.modality (image [ Des_g; Des_r ])));
          equal string "tok_image_hsc"
            (Modality.key (Codecs.modality (image [ Hsc_g; Hsc_y ])));
          raises_match (Exn.invalid_arg ~sub:"no one survey") (fun () ->
              Codecs.modality (image [ Hsc_g; Des_g ])));
      test "a modality without its codec is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"tok_flux_r") (fun () ->
              Codecs.encode codecs
                (Scalar { modality = Flux_r; values = Nx.zeros Nx.float32 [| 1 |] })));
      test "compiled codecs encode and decode as the eager ones" (fun () ->
          let c = Codecs.jit codecs in
          let values =
            Nx.create Nx.float32 [| 4 |] [| -2.; 0.25; Float.nan; 9. |]
          in
          let d = Codecs.Scalar { modality = Flux_g; values } in
          let e = Codecs.encode codecs d and ec = Codecs.encode c d in
          equal (array int) (ints e.tokens) (ints ec.tokens);
          equal (array bool) (presence e) (presence ec);
          let k = Nx.create Nx.int64 [| 3 |] [| 0L; 4L; 7L |] in
          match
            ( Codecs.decode codecs (Scalar Flux_g) k,
              Codecs.decode c (Scalar Flux_g) k )
          with
          | Scalar { values = v; _ }, Scalar { values = vc; _ } ->
              equal (array float_exact) (Nx.to_array v) (Nx.to_array vc)
          | _ -> fail "not Flux_g's values");
      test "load refuses the segmentation map" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"tok_segmap") (fun () ->
              Codecs.load ~cache_dir ~offline:true [ Modality.Segmap ]));
    ]

let () = exit (run "aion.codecs" [ absent; nan_edges; others ])
