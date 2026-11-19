(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The transformer with AION-1's weights against upstream (PolymathicAI/AION
   db7fba6df53b) on the parity subset. It lives outside runtest: the weights are
   1.27 GB, the reference's files 0.4 GB, and a run takes minutes. It runs under
   the alias parity, on AION_PARITY's huggingface, holding aion-base's
   model.safetensors at Hub.revision, and reference/f64 and reference/f32 (REF64
   and REF32 below), the float64 and float32 PyTorch references written by
   gen/reference.py, with upstream's code at db7fba6df53b and torch 2.14.1 on
   the CPU: each directory's model.safetensors holds upstream's encodings and
   logits, and REF64's codecs.safetensors the tokens both were computed from.
   Their SHA-256:

   REF64/model.safetensors
   101c17e9f8e72ba2542a1754571b08f807fe78e9499d5af9db5d93839ab9f99d
   REF64/codecs.safetensors
   c0f5c5af858ef968e480f44ddd73562acff8a36aa65ba9bc0cfedf762824cea5
   REF32/model.safetensors
   fe621f1a11a71284eb924675de3159b96f7f575185781fa65cbaad200b077495

   Two precisions: the transformer as AION-1 ships it, in float32, and cast to
   float64. Each checks the encodings of eight sets of inputs, and the logits of
   four targets: redshift from photometry and from a DESI spectrum, one token,
   against upstream's own forward (the decoder's training mask changes nothing
   for one token), and a DESI spectrum from an image and an image from a DESI
   spectrum against the forward with training's mask. For each it prints the
   largest absolute difference from the float64 reference, that difference over
   the reference's largest magnitude, PyTorch float32's two numbers beside them,
   and, for logits, the argmax tokens that differ from the reference's.

   It reports and does not gate: these values have no rounding ensemble yet. It
   exits 0 once both precisions have run. *)

open Aion

(* The inputs, under AION_PARITY as the README's Tests section lays them out. *)
let dir =
  match Sys.getenv_opt "AION_PARITY" with
  | Some d when d <> "" -> d
  | _ ->
      prerr_endline
        "transformer_parity: AION_PARITY is unset: set it to the directory of \
         the README's Tests section";
      exit 1

let path = Filename.concat dir
let printf = Printf.printf

let get dt name a =
  match Nx_io.Archive.find name a with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no tensor " ^ name)

let first n t = Nx.slice [ Nx.R (0, n) ] t
let objects = 8

let input codecs src m n : _ Transformer.input =
  let key = Modality.key m in
  let p = Nx_io.Archive.find (src ^ "." ^ key ^ ".tokens") codecs in
  let tokens =
    match p with
    | Some (Nx.P t) -> Nx.cast Nx.int64 t
    | None -> failwith ("no tokens " ^ src ^ "." ^ key)
  in
  { modality = m; tokens = first n tokens; present = None }

let ls_scalars =
  Modality.
    [ Flux_g; Flux_r; Flux_i; Flux_z; Flux_w1; Flux_w2; Flux_w3; Flux_w4; Ebv ]

let of_scalars = List.map (fun s -> Modality.Scalar s)

(* The reference's sets, as gen/reference.py names them. *)
let embeddings =
  Modality.
    [
      ("ls_desi.image", "ls_desi", [ Image ]);
      ("ls_desi.spectrum", "ls_desi", [ Spectrum Desi ]);
      ( "ls_desi.all",
        "ls_desi",
        [ Image; Spectrum Desi ] @ of_scalars ls_scalars );
      ( "ls_catalog.all",
        "ls_catalog",
        (Image :: of_scalars ls_scalars)
        @ of_scalars [ Shape_r; Shape_e1; Shape_e2 ] );
      ( "hsc.all",
        "hsc",
        Image_hsc
        :: of_scalars
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
             ] );
      ("sdss.all", "sdss", [ Spectrum Sdss; Scalar Z ]);
      ("desi.all", "desi", [ Spectrum Desi; Scalar Z ]);
      ( "gaia.all",
        "gaia",
        of_scalars
          [
            Xp_bp;
            Xp_rp;
            Flux_g_gaia;
            Flux_bp_gaia;
            Flux_rp_gaia;
            Parallax;
            Ra;
            Dec;
          ] );
    ]

(* Name, inputs from ls_desi, target, objects, the reference's variant. *)
let logits =
  Modality.
    [
      ( "z_from_phot",
        Image :: of_scalars [ Flux_g; Flux_r; Flux_i; Flux_z ],
        Scalar Z,
        objects,
        "upstream" );
      ("z_from_desi", [ Spectrum Desi ], Scalar Z, objects, "upstream");
      ("desi_from_image", [ Image ], Spectrum Desi, objects, "patched");
      ("image_from_desi", [ Spectrum Desi ], Image, 2, "patched");
    ]

let max_abs x = Nx.item [] (Nx.max (Nx.abs x))

(* The largest difference of [x] from [r], and that over [r]'s largest
   magnitude. *)
let error x r =
  let d = max_abs (Nx.sub (Nx.cast Nx.float64 x) r) in
  (d, d /. max_abs r)

let differing x r =
  let a = Nx.argmax ~axis:(-1) x and b = Nx.argmax ~axis:(-1) r in
  ( Int64.to_int (Nx.item [] (Nx.sum (Nx.cast Nx.int64 (Nx.not_equal a b)))),
    Nx.numel a )

let compare_at (type b) name (t : (b, Nx.host) Transformer.t) ~codecs ~r64 ~r32
    =
  printf "\n== %s\n%!" name;
  List.iter
    (fun (set, src, ms) ->
      let e =
        Transformer.encode t (List.map (fun m -> input codecs src m objects) ms)
      in
      let r = get Nx.float64 ("embed." ^ set) r64 in
      let d, n = error e.context r in
      let pd, pn = error (get Nx.float32 ("embed." ^ set) r32) r in
      printf "embed.%-18s abs %.3g norm %.3g | torch f32 abs %.3g norm %.3g\n%!"
        set d n pd pn)
    embeddings;
  List.iter
    (fun (set, ms, target, n, variant) ->
      let x = List.map (fun m -> input codecs "ls_desi" m n) ms in
      let l =
        List.hd
          (Transformer.forward t x [ { modality = target; positions = None } ])
      in
      let key = "logits." ^ set ^ "." ^ variant in
      let r = get Nx.float64 key r64 in
      let p32 = get Nx.float32 key r32 in
      let d, nm = error l r and pd, pn = error p32 r in
      let k, total = differing (Nx.cast Nx.float64 l) r in
      let pk, _ = differing (Nx.cast Nx.float64 p32) r in
      printf
        "logits.%s.%s abs %.3g norm %.3g, argmax %d/%d differ | torch f32 abs \
         %.3g norm %.3g, argmax %d differ\n\
         %!"
        set variant d nm k total pd pn pk)
    logits

let () =
  let hub = path "huggingface" in
  let ref64 = path "reference/f64" and ref32 = path "reference/f32" in
  let model dir =
    Nx_io.load_safetensors (Filename.concat dir "model.safetensors")
  in
  let r64 = model ref64 and r32 = model ref32 in
  let codecs =
    Nx_io.load_safetensors (Filename.concat ref64 "codecs.safetensors")
  in
  let t = Transformer.load ~cache_dir:hub ~offline:true () in
  compare_at "float32" t ~codecs ~r64 ~r32;
  compare_at "float64" (Transformer.cast Nx.float64 t) ~codecs ~r64 ~r32
