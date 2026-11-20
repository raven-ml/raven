(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'd data =
  | Image of { bands : Image.band list; flux : 'd Nx.float32_t }
  | Spectrum of {
      modality : Modality.spectrum;
      spectrum : ('d Nx.float32_t, 'd) Spectrum.Spectra.t;
      lengths : 'd Nx.int64_t option;
    }
  | Scalar of { modality : Modality.scalar; values : 'd Nx.float32_t }
[@@deriving ptree]

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
  | Image _ -> invalid_arg "Aion.Codecs.modality: bands of no one survey"
  | Spectrum { modality; _ } -> Modality.Spectrum modality
  | Scalar { modality; _ } -> Modality.Scalar modality

(* Codecs *)

type ('a, 'd) t = {
  image : ('a, 'd) Image.t option;
  spectrum : ('a, 'd) Spectrum.t option;
  scalars : (Modality.scalar * 'd Scalar.t) list;
}
[@@deriving ptree]

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

let missing m =
  invalid_arg
    (Printf.sprintf "Aion.Codecs: no codec loaded for %s" (Modality.key m))

let scalar t m =
  match List.assoc_opt m t.scalars with
  | Some c -> c
  | None -> missing (Scalar m)

let image t m = match t.image with Some c -> c | None -> missing m

let spectrum t m =
  match t.spectrum with Some c -> c | None -> missing (Modality.Spectrum m)

(* Tokens *)

(* An object is present unless one of its values is not finite where its
   channel's codebook holds no NaN edge, and every token of the object goes with
   it: [[|b|]] or [[|b; c|]] like [values]. A codebook that ends in NaN edges
   was trained with NaN as a value, so its values all keep their tokens.
   Presence is given whatever the values, so that it computes without reading
   them and [encode] compiles. *)
let present codec values =
  let nan_edge = Nx.any ~axes:[ 1 ] (Nx.isnan (Scalar.edges codec)) in
  let ok = Nx.logical_or (Nx.isfinite values) nan_edge in
  if Nx.ndim values = 1 then ok
  else Nx.broadcast_to (Nx.shape values) (Nx.all ~axes:[ 1 ] ~keepdims:true ok)

let encode t d : _ Tokens.t =
  let m = modality d in
  let all tokens : _ Tokens.t =
    let present = Nx.ones_at (Nx.beside tokens) Nx.bool (Nx.shape tokens) in
    { modality = m; tokens; present }
  in
  match d with
  | Image { bands; flux } -> all (Image.encode (image t m) bands flux)
  | Spectrum { modality; spectrum = x; lengths } ->
      let c = spectrum t modality in
      all (Spectrum.encode c (Spectrum.pad ?lengths modality x))
  | Scalar { modality; values } ->
      let c = scalar t modality in
      (* A modality of one token has values [[|b|]]. *)
      let shape = [| Nx.dim 0 values; Modality.tokens m |] in
      {
        modality = m;
        tokens = Nx.reshape shape (Scalar.encode c values);
        present = Nx.reshape shape (present c values);
      }

(* A token from the codebook's length [n] to the vocabulary's last is a class
   the transformer predicts whose bin holds no value, as redshift's 1024, and
   decodes to NaN. Where the vocabulary exceeds the codebook, a token outside
   it is refused here, since the codec's refusal would name its own range. *)
let decode_scalar c m k =
  let n = Nx.dim 1 (Scalar.edges c) and vocab = Modality.vocab m in
  let v =
    if vocab <= n then Scalar.decode c k
    else begin
      let inside =
        Nx.logical_and (Nx.greater_equal_s k 0L)
          (Nx.less_s k (Int64.of_int vocab))
      in
      Nx.check Nx.Ptree.tensor inside k (fun i k ->
          Invalid_argument
            (Format.asprintf
               "Aion.Codecs.decode: token %Ld at %a outside [0, %d)"
               (Nx.item [] k) Nx.pp_shape i vocab));
      let past = Nx.greater_equal_s k (Int64.of_int n) in
      let v = Scalar.decode c (Nx.where past (Nx.zeros_like k) k) in
      Nx.where past (Nx.full_like v Float.nan) v
    end
  in
  (* A modality of one token has values [[|b|]], its tokens [[|b; 1|]]. *)
  if Modality.tokens m = 1 then Nx.reshape [| Nx.dim 0 k |] v else v

let decode t (m : Modality.t) k =
  match m with
  | Image | Image_hsc ->
      let bands = bands m in
      Image { bands; flux = Image.decode (image t m) bands k }
  | Spectrum s ->
      Spectrum
        {
          modality = s;
          spectrum = Spectrum.decode (spectrum t s) k;
          lengths = None;
        }
  | Scalar s ->
      Scalar { modality = s; values = decode_scalar (scalar t s) m k }
  | Segmap | Catalog -> missing m
