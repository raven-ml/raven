(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Survey on synthetic Multimodal Universe files (mmu/*.h5, which gen/surveys.py
   writes with h5py): each holds a few rows in the layout of one survey's file,
   dataset names, dtypes, strings, chunks and filters as MMU's.
   mmu/expected.safetensors holds, for each modality, the float32 values its
   column must give, the generator naming each column by upstream's modality
   names. Pixel (b, y, x) of row r of a file is 1000 r + 10 (b + 1) + x mod
   16. *)

open Windtrap
open Aion

let expected = Nx_io.load_safetensors "mmu/expected.safetensors"

let entry dt name =
  match Nx_io.Archive.find name expected with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no expected entry " ^ name)

let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let file name = "mmu/" ^ name ^ ".h5"
let keys ms = List.map Modality.key ms

(* The sources, each with the name its expected entries start with. *)
let sources =
  [
    ("legacysurvey", Survey.load Legacy_survey (file "legacysurvey"));
    ("hsc", Survey.load Hsc (file "hsc"));
    ("desi", Survey.load Desi (file "desi"));
    ("sdss", Survey.load Sdss (file "sdss"));
    ("gaia", Survey.load Gaia (file "gaia"));
    ( "grouped/legacysurvey",
      Survey.load ~group:"/legacysurvey" Legacy_survey (file "grouped") );
    ("grouped/desi", Survey.load ~group:"/desi" Desi (file "grouped"));
  ]

let source n = List.assoc n sources

(* The map *)

let ls_scalars =
  Modality.
    [
      Flux_g;
      Flux_r;
      Flux_i;
      Flux_z;
      Flux_w1;
      Flux_w2;
      Flux_w3;
      Flux_w4;
      Shape_r;
      Shape_e1;
      Shape_e2;
      Ebv;
    ]

let hsc_scalars =
  Modality.
    [
      A_g;
      A_r;
      A_i;
      A_z;
      A_y;
      Mag_g;
      Mag_r;
      Mag_i;
      Mag_z;
      Mag_y;
      Shape11;
      Shape22;
      Shape12;
    ]

let scalar l = List.map (fun s -> Modality.Scalar s) l

let modalities =
  cases "each source gives the modalities its datasets hold" ~id:fst
    [
      ("legacysurvey", Modality.Image :: scalar ls_scalars);
      ("hsc", Modality.Image_hsc :: scalar hsc_scalars);
      ("desi", Modality.[ Spectrum Desi; Scalar Z ]);
      ("sdss", Modality.[ Spectrum Sdss; Scalar Z ]);
      ( "gaia",
        scalar
          Modality.
            [
              Flux_g_gaia;
              Flux_bp_gaia;
              Flux_rp_gaia;
              Parallax;
              Ra;
              Dec;
              Xp_bp;
              Xp_rp;
            ] );
      ( "grouped/legacysurvey",
        scalar
          Modality.
            [
              Flux_g;
              Flux_r;
              Flux_i;
              Flux_z;
              Flux_w1;
              Flux_w2;
              Flux_w3;
              Flux_w4;
              Ebv;
            ] );
      ("grouped/desi", Modality.[ Spectrum Desi ]);
    ]
    (fun (n, ms) ->
      equal (list string) (keys ms) (keys (Survey.modalities (source n))))

(* The value each row of [n]'s modality [m] must read, [[|rows; c|]] with [c]
   its columns, or the pixels of an image. *)
let truth n (m : Modality.t) =
  match m with
  | Image | Image_hsc -> None
  | Spectrum _ -> Some (floats (entry Nx.float32 (n ^ ".spectrum_flux")))
  | _ -> Some (floats (entry Nx.float32 (n ^ "." ^ Modality.key m)))

let width n (m : Modality.t) =
  match m with
  | Spectrum _ -> Nx.dim 1 (entry Nx.float32 (n ^ ".spectrum_flux"))
  | Scalar (Xp_bp | Xp_rp) -> 55
  | _ -> 1

let pixel r b x = float_of_int ((1000 * r) + (10 * (b + 1)) + (x mod 16))

(* [m]'s data of [rows] against the truth at those rows. *)
let check n rows (m : Modality.t) =
  match Survey.read (source n) rows m with
  | Scalar { values; _ } ->
      let t = Option.get (truth n m) and c = width n m in
      let want =
        Array.concat
          (Array.to_list (Array.map (fun r -> Array.sub t (r * c) c) rows))
      in
      equal (array float_exact) want (floats values)
  | Spectrum { spectrum; lengths; _ } ->
      let t = Option.get (truth n m) and c = width n m in
      let want =
        Array.concat
          (Array.to_list (Array.map (fun r -> Array.sub t (r * c) c) rows))
      in
      equal (array float_exact) want (floats spectrum.flux);
      let all = Nx.to_array (entry Nx.int64 (n ^ ".lengths")) in
      equal (array int64)
        (Array.map (fun r -> all.(r)) rows)
        (Nx.to_array (Option.get lengths))
  | Image { bands; flux } ->
      equal (list string)
        (List.map Image.name (Codecs.bands m))
        (List.map Image.name bands);
      let s = Nx.shape flux in
      equal int (Array.length rows) s.(0);
      let a = floats flux in
      Array.iteri
        (fun i r ->
          for b = 0 to s.(1) - 1 do
            for x = 0 to s.(3) - 1 do
              let at = (((((i * s.(1)) + b) * s.(2)) + 7) * s.(3)) + x in
              equal float_exact (pixel r b x) a.(at)
            done
          done)
        rows

(* Rows drawn with repeats and out of order, so runs cross chunk edges. *)
let draw_rows n =
  Gen.(array ~size:(int_range 0 6) (int_range 0 (Survey.length (source n) - 1)))

let map =
  group "map"
    (List.map
       (fun (n, t) ->
         prop (n ^ ": each modality reads its column at the rows asked")
           (draw_rows n) (fun rows ->
             cover "no row" (rows = [||]);
             cover "rows out of order"
               (let s = Array.copy rows in
                Array.sort compare s;
                s <> rows);
             List.iter (check n rows) (Survey.modalities t)))
       sources)

(* Values *)

let values =
  group "values"
    [
      test "a float64 column reads as its float32 rounding" (fun () ->
          let f =
            Nx_io.load_hdf5 (file "gaia")
            |> snd
            |> List.find (fun d -> Nx_io.Hdf5.path d = "/phot_g_mean_flux")
          in
          let (Nx.P raw) = Nx_io.Hdf5.values f in
          let raw = floats raw in
          let got =
            match
              Survey.read (source "gaia") [| 0; 1; 2; 3 |] (Scalar Flux_g_gaia)
            with
            | Scalar { values; _ } -> floats values
            | _ -> fail "not a scalar"
          in
          let f32 x = Int32.float_of_bits (Int32.bits_of_float x) in
          equal (array float_exact) (Array.map f32 raw) got;
          Array.iteri (fun i v -> not_equal float_exact v got.(i)) raw);
      test "SDSS rows are their pixels before MMU's -1 pads" (fun () ->
          match
            Survey.read (source "sdss") [| 0; 1; 2; 3 |] (Spectrum Sdss)
          with
          | Spectrum { lengths; _ } ->
              equal (array int64) [| 40L; 39L; 30L; 1L |]
                (Nx.to_array (Option.get lengths))
          | _ -> fail "not a spectrum");
      test "a -1 before a row's last pixel is refused" (fun () ->
          let t = Survey.load Sdss (file "sdss_gap") in
          ignore (Survey.read t [| 0 |] (Spectrum Sdss));
          raises_match
            (Exn.failure ~sub:"mmu/sdss_gap.h5: /spectrum_lambda: row 1")
            (fun () -> Survey.read t [| 0; 1 |] (Spectrum Sdss)));
      test "rows whose bands differ are refused" (fun () ->
          let t = Survey.load Legacy_survey (file "ls_bands") in
          ignore (Survey.read t [| 1 |] Image);
          raises_match (Exn.failure ~sub:"mmu/ls_bands.h5: /image_band: row 1")
            (fun () -> Survey.read t [| 0; 1 |] Image));
      test "a modality the source does not hold is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"tok_shape_r") (fun () ->
              Survey.read
                (source "grouped/legacysurvey")
                [| 0 |] (Scalar Shape_r));
          raises_match (Exn.invalid_arg ~sub:"tok_flux_g") (fun () ->
              Survey.read (source "desi") [| 0 |] (Scalar Flux_g)));
      test "a row outside the source is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"row 3 outside [0, 3)") (fun () ->
              Survey.read (source "legacysurvey") [| 0; 3 |] (Scalar Flux_g)));
      test "a source needs its survey's id dataset" (fun () ->
          raises_match (Exn.failure ~sub:"no source_id") (fun () ->
              Survey.load Gaia (file "legacysurvey")));
    ]

(* Matching *)

let deg = Float.pi /. 180.
let arcsec = deg /. 3600.

let unit (ra, dec) =
  ( cos (dec *. deg) *. cos (ra *. deg),
    cos (dec *. deg) *. sin (ra *. deg),
    sin (dec *. deg) )

let separation p q =
  let ax, ay, az = unit p and bx, by, bz = unit q in
  let cx = (ay *. bz) -. (az *. by)
  and cy = (az *. bx) -. (ax *. bz)
  and cz = (ax *. by) -. (ay *. bx) in
  Float.atan2
    (Float.sqrt ((cx *. cx) +. (cy *. cy) +. (cz *. cz)))
    ((ax *. bx) +. (ay *. by) +. (az *. bz))

(* The reciprocal rule by brute force: each row's nearest within 1″, the lowest
   row on ties, kept where the nearest's nearest is the row. *)
let reference a b =
  let nearest p qs =
    let best = ref None in
    Array.iteri
      (fun j q ->
        let s = separation p q in
        if s <= arcsec then
          match !best with
          | Some (_, s') when s' <= s -> ()
          | _ -> best := Some (j, s))
      qs;
    Option.map fst !best
  in
  let is = ref [] in
  Array.iteri
    (fun i p ->
      match nearest p b with
      | Some j when nearest b.(j) a = Some i -> is := (i, j) :: !is
      | _ -> ())
    a;
  let l = List.rev !is in
  (Array.of_list (List.map fst l), Array.of_list (List.map snd l))

let directions rows =
  let n = Array.length rows in
  let q f =
    Ymir.Quantity.v Ymir.Unit.degree
      (Nx.create Nx.float64 [| n |] (Array.map f rows))
  in
  Ymir.Direction.lonlat Ymir.Frame.icrs ~lon:(q fst) ~lat:(q snd)

(* Catalogues of up to 10 rows within a few arcseconds of one point: near the
   equator, across ra 0, or at a pole; some rows repeated, some NaN. *)
let catalogues =
  Gen.(
    let* ra0, dec0 = of_list [ (150., 2.); (0., -30.); (40., 89.9999) ] in
    let row =
      let+ dx = float_range (-3.) 3. and+ dy = float_range (-3.) 3. in
      (ra0 +. (dx /. 3600. /. Float.cos (dec0 *. deg)), dec0 +. (dy /. 3600.))
    in
    let cat =
      array ~size:(int_range 0 10)
        (frequency [ (12, row); (1, constant (Float.nan, dec0)) ])
    in
    let+ a = cat and+ b = cat and+ dup = bool in
    (* A repeated row of b ties at every separation. *)
    let b =
      if dup && Array.length b > 0 then Array.append b [| b.(0) |] else b
    in
    (a, b))

let matching =
  group "matching"
    [
      prop "pairs are the reciprocal nearest rows within 1\"" catalogues
        (fun (a, b) ->
          let i, j = Survey.matches (directions a) (directions b) in
          cover "a pair" (Array.length i > 0);
          cover "a row of b repeated"
            (Array.length b > 1 && b.(Array.length b - 1) = b.(0));
          cover "a NaN row" (Array.exists (fun (r, _) -> Float.is_nan r) a);
          cover "a row near b left unpaired"
            (Array.exists Fun.id
               (Array.mapi
                  (fun k p ->
                    Array.exists (fun q -> separation p q <= arcsec) b
                    && not (Array.mem k i))
                  a));
          let ri, rj = reference a b in
          equal (pair (array int) (array int)) (ri, rj) (i, j));
      test "the tutorial file's groups pair row by row" (fun () ->
          let i, j =
            Survey.pair (source "grouped/legacysurvey") (source "grouped/desi")
          in
          equal (pair (array int) (array int)) ([| 0; 1 |], [| 0; 1 |]) (i, j));
      test "Gaia pairs within 1\", reciprocally" (fun () ->
          let i, j = Survey.pair (source "legacysurvey") (source "gaia") in
          equal (pair (array int) (array int)) ([| 0; 1 |], [| 1; 0 |]) (i, j));
    ]

let () = exit (run "aion.survey" [ modalities; map; values; matching ])
