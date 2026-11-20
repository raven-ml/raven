(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let fail fmt = Printf.ksprintf failwith fmt

(* Surveys *)

type survey = Legacy_survey | Hsc | Desi | Sdss | Gaia

let name = function
  | Legacy_survey -> "legacysurvey"
  | Hsc -> "hsc"
  | Desi -> "desi"
  | Sdss -> "sdss"
  | Gaia -> "gaia"

let id = function Gaia -> "source_id" | _ -> "object_id"

(* The map *)

(* How a modality's data lie in a source: [Pixels] is image_array with its
   bands in image_band, [Spectra] the four spectrum datasets, [Column c] one
   dataset of a value per object, and [Coefficients (c, lo)] the 55 columns of
   [c] from [lo]. *)
type column =
  | Pixels
  | Spectra
  | Column of string
  | Coefficients of string * int

let spectra = [ "spectrum_flux"; "spectrum_ivar"; "spectrum_mask"; "spectrum_lambda" ]

let datasets = function
  | Pixels -> [ "image_array"; "image_band" ]
  | Spectra -> spectra
  | Column c | Coefficients (c, _) -> [ c ]

(* MMU's datasets (MultimodalUniverse v1 at Flatiron, 2026-11-19) under
   upstream's modality names (aion/modalities.py). *)
let map : (survey * Modality.t * column) list =
  let s m c = Modality.Scalar m, Column c in
  let rows survey l = List.map (fun (m, c) -> (survey, m, c)) l in
  rows Legacy_survey
    Modality.
      [
        (Image, Pixels);
        s Flux_g "FLUX_G";
        s Flux_r "FLUX_R";
        s Flux_i "FLUX_I";
        s Flux_z "FLUX_Z";
        s Flux_w1 "FLUX_W1";
        s Flux_w2 "FLUX_W2";
        s Flux_w3 "FLUX_W3";
        s Flux_w4 "FLUX_W4";
        s Shape_r "SHAPE_R";
        s Shape_e1 "SHAPE_E1";
        s Shape_e2 "SHAPE_E2";
        s Ebv "EBV";
      ]
  @ rows Hsc
      Modality.
        [
          (Image_hsc, Pixels);
          s A_g "a_g";
          s A_r "a_r";
          s A_i "a_i";
          s A_z "a_z";
          s A_y "a_y";
          s Mag_g "g_cmodel_mag";
          s Mag_r "r_cmodel_mag";
          s Mag_i "i_cmodel_mag";
          s Mag_z "z_cmodel_mag";
          s Mag_y "y_cmodel_mag";
          s Shape11 "i_sdssshape_shape11";
          s Shape22 "i_sdssshape_shape22";
          s Shape12 "i_sdssshape_shape12";
        ]
  @ rows Desi Modality.[ (Spectrum Desi, Spectra); s Z "Z" ]
  @ rows Sdss Modality.[ (Spectrum Sdss, Spectra); s Z "Z" ]
  @ rows Gaia
      Modality.
        [
          (Scalar Xp_bp, Coefficients ("coeff", 0));
          (Scalar Xp_rp, Coefficients ("coeff", 55));
          s Flux_g_gaia "phot_g_mean_flux";
          s Flux_bp_gaia "phot_bp_mean_flux";
          s Flux_rp_gaia "phot_rp_mean_flux";
          s Parallax "parallax";
          s Ra "ra";
          s Dec "dec";
        ]

(* MMU pads a spectrum file's rows with pixels of this wavelength. *)
let pad_wavelength = -1.

(* Sources *)

type t = {
  survey : survey;
  path : string;
  group : string;
  datasets : (string, Nx_io.Hdf5.t) Hashtbl.t;
  length : int;
}

let length t = t.length

(* The group's name without a trailing slash, and the prefix of its
   datasets' paths. *)
let group_prefix g =
  let g = if String.starts_with ~prefix:"/" g then g else "/" ^ g in
  let g =
    if String.length g > 1 && String.ends_with ~suffix:"/" g then
      String.sub g 0 (String.length g - 1)
    else g
  in
  (g, if g = "/" then "/" else g ^ "/")

let load ?(group = "/") survey path =
  let group, prefix = group_prefix group in
  let _, ds = Nx_io.load_hdf5 path in
  let datasets = Hashtbl.create 64 in
  List.iter
    (fun d ->
      let p = Nx_io.Hdf5.path d in
      if String.starts_with ~prefix p then
        let n = String.sub p (String.length prefix) (String.length p - String.length prefix) in
        if not (String.contains n '/') then Hashtbl.replace datasets n d)
    ds;
  List.iter
    (fun n ->
      if not (Hashtbl.mem datasets n) then
        fail "%s: %s holds no %s, so no %s source" path group n (name survey))
    [ id survey; "ra"; "dec" ];
  let shape = Nx_io.Hdf5.shape (Hashtbl.find datasets (id survey)) in
  if Array.length shape = 0 then fail "%s: %s%s is a scalar" path prefix (id survey);
  { survey; path; group; datasets; length = shape.(0) }

let holds t c = List.for_all (Hashtbl.mem t.datasets) (datasets c)

let rank m =
  let rec go i = function
    | [] -> max_int
    | x :: _ when x = m -> i
    | _ :: l -> go (i + 1) l
  in
  go 0 Modality.all

let modalities t =
  List.filter_map
    (fun (s, m, c) -> if s = t.survey && holds t c then Some m else None)
    map
  |> List.sort (fun a b -> compare (rank a) (rank b))

(* Reading *)

(* The dataset [n]'s rows [rows], each its own window. *)
let select t n rows =
  let d = Hashtbl.find t.datasets n in
  let shape = Nx_io.Hdf5.shape d in
  if Array.length shape = 0 || shape.(0) <> t.length then
    fail "%s: %s: %s rows, %d objects" t.path (Nx_io.Hdf5.path d)
      (if Array.length shape = 0 then "no" else string_of_int shape.(0))
      t.length;
  let ones = Array.make (Array.length rows) 1 in
  match Nx_io.Hdf5.rows d ~row:rows ~rows:ones with
  | Ok x -> (d, x)
  | Error _ -> fail "%s: %s: a partial read" t.path (Nx_io.Hdf5.path d)

let float32 t n rows =
  let _, Nx.P x = select t n rows in
  Nx.cast Nx.float32 x

let band_of t d row n =
  match
    List.find_opt
      (fun b -> Image.name b = String.uppercase_ascii n)
      Image.bands
  with
  | Some b -> b
  | None -> fail "%s: %s: row %d: %S names no band" t.path (Nx_io.Hdf5.path d) row n

let image t rows (m : Modality.t) : Nx.host Codecs.data =
  let flux = float32 t "image_array" rows in
  let d, names = select t "image_band" rows in
  let names = Nx_io.Hdf5.strings d names in
  let row i =
    let c = Array.length names / Array.length rows in
    List.map (band_of t d rows.(i)) (Array.to_list (Array.sub names (i * c) c))
  in
  let bands = if Array.length rows = 0 then Codecs.bands m else row 0 in
  Array.iteri
    (fun i r ->
      let bs = row i in
      if bs <> bands then
        fail "%s: %s: row %d lists the bands %s, row %d %s" t.path
          (Nx_io.Hdf5.path d) r
          (String.concat ", " (List.map Image.name bs))
          rows.(0)
          (String.concat ", " (List.map Image.name bands)))
    rows;
  let data = Codecs.Image { bands; flux } in
  if (try Codecs.modality data <> m with Invalid_argument _ -> true) then
    fail "%s: %s: bands %s are not %s's" t.path (Nx_io.Hdf5.path d)
      (String.concat ", " (List.map Image.name bands))
      (Modality.key m);
  data

(* Each row's pixels before its run of MMU's pads, which must end the row. *)
let lengths t d rows (w : Nx.host Nx.float32_t) =
  let m = Nx.dim 1 w in
  let a = Nx.to_array w in
  Array.mapi
    (fun i r ->
      let pad k = a.((i * m) + k) = pad_wavelength in
      let l = ref m in
      while !l > 0 && pad (!l - 1) do
        decr l
      done;
      if !l = 0 then fail "%s: %s: row %d holds no pixel" t.path (Nx_io.Hdf5.path d) r;
      for k = 0 to !l - 1 do
        if pad k then
          fail "%s: %s: row %d: a pad wavelength %g at pixel %d, before pixel %d"
            t.path (Nx_io.Hdf5.path d) r pad_wavelength k (!l - 1)
      done;
      Int64.of_int !l)
    rows

let spectrum t rows modality : Nx.host Codecs.data =
  let read n = float32 t n rows in
  let wavelength = read "spectrum_lambda" in
  let d = Hashtbl.find t.datasets "spectrum_lambda" in
  let lengths = lengths t d rows wavelength in
  let spectrum : _ Spectrum.spectrum =
    {
      flux = read "spectrum_flux";
      ivar = read "spectrum_ivar";
      mask = Nx.not_equal_s (read "spectrum_mask") 0.;
      wavelength;
    }
  in
  Spectrum
    {
      modality;
      spectrum;
      lengths = Some (Nx.create Nx.int64 [| Array.length rows |] lengths);
    }

let read t rows m =
  let column =
    match List.find_opt (fun (s, m', _) -> s = t.survey && m' = m) map with
    | Some (_, _, c) when holds t c -> c
    | _ ->
        invalid_arg
          (Printf.sprintf "Aion.Survey.read: %s's %s holds no %s" t.path t.group
             (Modality.key m))
  in
  Array.iter
    (fun r ->
      if r < 0 || r >= t.length then
        invalid_arg
          (Printf.sprintf "Aion.Survey.read: row %d outside [0, %d)" r
             t.length))
    rows;
  match (column, m) with
  | Pixels, _ -> image t rows m
  | Spectra, Spectrum s -> spectrum t rows s
  | Column c, Scalar modality -> Scalar { modality; values = float32 t c rows }
  | Coefficients (c, lo), Scalar modality ->
      let w = Modality.tokens m in
      let values = Nx.slice [ Nx.A; Nx.R (lo, lo + w) ] (float32 t c rows) in
      Scalar { modality; values }
  | _ -> assert false

(* Cross-matching *)

let radius = Ymir.Quantity.v Ymir.Unit.arcsecond (Nx.scalar Nx.float64 1.)

let directions t =
  let read n =
    let (Nx.P x) = Nx_io.Hdf5.values (Hashtbl.find t.datasets n) in
    Ymir.Quantity.v Ymir.Unit.degree (Nx.cast Nx.float64 x)
  in
  Ymir.Direction.lonlat Ymir.Frame.icrs ~lon:(read "ra") ~lat:(read "dec")

let pair a b =
  let i, j, _ =
    Ymir.Direction.reciprocal ~within:radius (directions a) (directions b)
  in
  let rows x = Array.map Int64.to_int (Nx.to_array x) in
  (rows i, rows j)
