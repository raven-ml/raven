(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type data =
  | Image of { bands : Image.band list; flux : Nx.host Nx.float32_t }
  | Spectrum of {
      modality : Modality.spectrum;
      spectrum : (Nx.float32_elt, Nx.host) Spectrum.spectrum;
      lengths : Nx.host Nx.int64_t option;
    }
  | Scalar of { modality : Modality.scalar; values : Nx.host Nx.float32_t }

let hsc : Image.band -> bool = function
  | Hsc_g | Hsc_r | Hsc_i | Hsc_z | Hsc_y -> true
  | Des_g | Des_r | Des_i | Des_z -> false

(* Each image modality's bands, in channel order. *)
let bands : Modality.t -> Image.band list = function
  | Image_hsc -> List.filter hsc Image.bands
  | _ -> List.filter (fun b -> not (hsc b)) Image.bands

let modality = function
  | Image { bands; _ } when bands <> [] && List.for_all hsc bands ->
      Modality.Image_hsc
  | Image { bands; _ } when bands <> [] && not (List.exists hsc bands) ->
      Modality.Image
  | Image _ -> Refuse.invalid "Codecs.modality" "bands of no one survey"
  | Spectrum { modality; _ } -> Modality.Spectrum modality
  | Scalar { modality; _ } -> Modality.Scalar modality

(* Codecs *)

type t = {
  image : (Nx.float32_elt, Nx.host) Image.t option;
  spectrum : (Nx.float32_elt, Nx.host) Spectrum.t option;
  scalars : (Modality.scalar * Nx.host Scalar.t) list;
}

let load ?cache_dir ?offline ms =
  (* Refuses a modality without a codec before any file is fetched. *)
  List.iter (fun m -> ignore (Hub.file m)) ms;
  let codec m = Hub.codec ?cache_dir ?offline m in
  (* One codec serves each kind, read from the first modality of that kind. *)
  let first f = Option.map codec (List.find_opt f ms) in
  let image =
    first (function Modality.Image | Image_hsc -> true | _ -> false)
    |> Option.map Image.of_archive
  in
  let spectrum =
    first (function Modality.Spectrum _ -> true | _ -> false)
    |> Option.map Spectrum.of_archive
  in
  let scalars =
    List.filter_map
      (function
        | Modality.Scalar s -> Some (s, Scalar.of_archive s (codec (Scalar s)))
        | _ -> None)
      ms
  in
  { image; spectrum; scalars }

let jit t =
  {
    image = Option.map Image.jit t.image;
    spectrum = Option.map Spectrum.jit t.spectrum;
    scalars = List.map (fun (m, c) -> (m, Scalar.jit c)) t.scalars;
  }

let missing m =
  Refuse.invalid "Codecs" "no codec loaded for %s" (Modality.key m)

let scalar t m =
  match List.assoc_opt m t.scalars with
  | Some c -> c
  | None -> missing (Scalar m)

let image t m = match t.image with Some c -> c | None -> missing m

let spectrum t m =
  match t.spectrum with Some c -> c | None -> missing (Modality.Spectrum m)

(* Tokens *)

(* An object is present unless one of its values is not finite where its
   channel's codebook holds no NaN edge, and every token of the object goes
   with it: [[|b|]] or [[|b; c|]] like [values]. A codebook that ends in NaN
   edges was trained with NaN as a value, so its values all keep their
   tokens. *)
let present codec values =
  let nan_edge = Nx.any ~axes:[ 1 ] (Nx.isnan (Scalar.edges codec)) in
  let ok = Nx.logical_or (Nx.isfinite values) nan_edge in
  if Nx.item [] (Nx.all ok) then None
  else if Nx.ndim values = 1 then Some ok
  else
    Some
      (Nx.broadcast_to (Nx.shape values) (Nx.all ~axes:[ 1 ] ~keepdims:true ok))

let encode t d : _ Tokens.t =
  let m = modality d in
  match d with
  | Image { bands; flux } ->
      { modality = m; tokens = Image.encode (image t m) bands flux; present = None }
  | Spectrum { modality; spectrum = x; lengths } ->
      let c = spectrum t modality in
      { modality = m; tokens = Spectrum.encode ?lengths c modality x; present = None }
  | Scalar { modality; values } ->
      let c = scalar t modality in
      { modality = m; tokens = Scalar.encode c values; present = present c values }

let decode t (m : Modality.t) k =
  match m with
  | Image | Image_hsc ->
      let bands = bands m in
      Image { bands; flux = Image.decode (image t m) bands k }
  | Spectrum s ->
      Spectrum { modality = s; spectrum = Spectrum.decode (spectrum t s) k; lengths = None }
  | Scalar s -> Scalar { modality = s; values = Scalar.decode (scalar t s) k }
  | Segmap | Catalog -> missing m
