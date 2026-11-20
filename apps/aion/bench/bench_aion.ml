(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* AION-1's calls on the CPU, for one object, the rows
   bench/pytorch.py times with upstream's code:
   - [image encode]: Image.encode of a Legacy Survey cutout, [1; 4; 160; 160];
   - [spectrum encode]: Spectrum.encode of a DESI spectrum of 7781 pixels;
   - [encode]: Transformer.encode of an image's 576 tokens and the g, r, i and
     z fluxes' four, the input set of AION-1's §7.1.1;
   - [decode 273]: Transformer.decode of a DESI spectrum's 273 tokens from
     that encoding;
   - [roar step]: Roar.sample of redshift from those inputs in one step, an
     encode and a one-token decode: a draw of §7.1.1.

   The weights are random draws at aion-base's shapes: the codecs' widths and
   depths, and the transformer's width 768, 12 encoder and 12 decoder blocks of
   12 heads and a feed-forward of 2048. The transformer holds the tables of the
   modalities the rows read and predict only, since the others take no part in
   these calls. The time is the architecture's, not the values'. A case builds
   its weights in its worker, which thumper forks, so this process computes
   nothing and starts no domain. *)

open Aion

let width = 768
let blocks = 12
let heads = 12
let hidden = 2048

(* Each weight a normal draw over [sqrt fan_in], from a key of its own. *)
let draws () =
  let k = ref 0 in
  fun shape ~fan ->
    incr k;
    Nx.mul_s
      (Nx.Rng.normal ~key:(Nx.Rng.key !k) Nx.float32 shape)
      (1. /. Float.sqrt (Float.of_int fan))

(* The image codec: 54 projected channels, widths 128, 256 and 512, four
   residual units a stage, squeeze-excite to half the width; named as AION-1's
   codecs/image/model.safetensors. *)
let image_archive () =
  let draw = draws () and entries = ref [] in
  let add name x = entries := (name, Nx.P x) :: !entries in
  let conv name o i k =
    add (name ^ ".weight") (draw [| o; i; k; k |] ~fan:(i * k * k));
    add (name ^ ".bias") (Nx.zeros Nx.float32 [| o |])
  in
  let unit name c =
    conv (name ^ ".net.0.conv") c c 3;
    conv (name ^ ".net.2") c c 1;
    conv (name ^ ".net.4.to_k") 1 c 1;
    conv (name ^ ".net.4.net.0") (c / 2) c 1;
    conv (name ^ ".net.4.net.2") c (c / 2) 1
  in
  let units name c =
    List.iter (fun i -> unit (Printf.sprintf "%s.%d" name i) c) [ 0; 1; 2; 3 ]
  in
  let p = 54 and d = 128 in
  let d1 = 2 * d and d2 = 4 * d in
  conv "model.conv_in.conv" d p 7;
  unit "model.encoder_layers.0" d;
  conv "model.encoder_layers.1.conv" d1 d 3;
  units "model.encoder_layers.2" d1;
  conv "model.encoder_layers.3.conv" d2 d1 3;
  units "model.encoder_layers.4" d2;
  conv "model.encoder_layers.5.conv" d2 d2 7;
  units "model.encoder_layers.6" d2;
  add "model.encoder_layers.7.1.weight" (Nx.ones Nx.float32 [| d2 |]);
  add "model.encoder_layers.7.1.bias" (Nx.zeros Nx.float32 [| d2 |]);
  conv "pre_quant_proj" 5 d2 1;
  conv "post_quant_proj" d2 5 1;
  units "model.decoder_layers.0" d2;
  conv "model.decoder_layers.1.conv" d2 d2 3;
  units "model.decoder_layers.2" d2;
  conv "model.decoder_layers.3.net.0" (4 * d1) d2 1;
  units "model.decoder_layers.4" d1;
  conv "model.decoder_layers.5.net.0" (4 * d) d1 1;
  unit "model.decoder_layers.6" d;
  conv "model.conv_out.conv" p d 3;
  add "subsample_in.weight" (draw [| p; 9 |] ~fan:9);
  add "subsample_in.bias" (Nx.zeros Nx.float32 [| p |]);
  add "subsample_out.weight" (draw [| 9; p |] ~fan:p);
  add "subsample_out.bias" (Nx.zeros Nx.float32 [| 9 |]);
  add "_quantizer.levels" (Nx.create Nx.int32 [| 5 |] [| 7l; 5l; 5l; 5l; 5l |]);
  Nx_io.Archive.of_list !entries

(* The spectrum codec: ConvNeXt stages of widths 64, 128, 256 and 512 and depths
   3, 3, 27 and 3 down, 256, 128, 64 and 2 and 3, 9, 3 and 3 up, a 10-bit LFQ
   and a latent grid of 8704 wavelengths; named as AION-1's
   codecs/spectrum/model.safetensors. *)
let spectrum_archive () =
  let draw = draws () and entries = ref [] in
  let add name x = entries := (name, Nx.P x) :: !entries in
  let ln name d =
    add (name ^ ".weight") (Nx.ones Nx.float32 [| d |]);
    add (name ^ ".bias") (Nx.zeros Nx.float32 [| d |])
  in
  let lin name o i =
    add (name ^ ".weight") (draw [| o; i |] ~fan:i);
    add (name ^ ".bias") (Nx.zeros Nx.float32 [| o |])
  in
  let conv name o i k =
    add (name ^ ".weight") (draw [| o; i; k |] ~fan:(i * k));
    add (name ^ ".bias") (Nx.zeros Nx.float32 [| o |])
  in
  let convt name i o k =
    add (name ^ ".weight") (draw [| i; o; k |] ~fan:(i * k));
    add (name ^ ".bias") (Nx.zeros Nx.float32 [| o |])
  in
  let block name d =
    conv (name ^ ".dwconv") d 1 7;
    ln (name ^ ".norm") d;
    lin (name ^ ".pwconv1") (4 * d) d;
    add (name ^ ".grn.gamma") (Nx.zeros Nx.float32 [| 1; 1; 4 * d |]);
    add (name ^ ".grn.beta") (Nx.zeros Nx.float32 [| 1; 1; 4 * d |]);
    lin (name ^ ".pwconv2") d (4 * d)
  in
  let stage name dims depths i =
    for j = 0 to depths.(i) - 1 do
      block (Printf.sprintf "%s.%d.%d" name i j) dims.(i)
    done
  in
  let enc = [| 64; 128; 256; 512 |] and enc_depths = [| 3; 3; 27; 3 |] in
  let dec = [| 256; 128; 64; 2 |] and dec_depths = [| 3; 9; 3; 3 |] in
  for i = 0 to 3 do
    let s = Printf.sprintf "encoder.downsample_layers.%d" i in
    if i = 0 then begin
      conv (s ^ ".0") enc.(0) 2 4;
      ln (s ^ ".1") enc.(0)
    end
    else begin
      ln (s ^ ".0") enc.(i - 1);
      conv (s ^ ".1") enc.(i) enc.(i - 1) 2
    end;
    stage "encoder.stages" enc enc_depths i
  done;
  ln "encoder.norm" 512;
  ln "pre_quant_norm" 512;
  conv "quant_conv" 10 512 1;
  conv "post_quant_conv" 512 10 1;
  for i = 0 to 3 do
    let s = Printf.sprintf "decoder.upsample_layers.%d" i in
    if i = 0 then begin
      convt (s ^ ".0") 512 dec.(0) 2;
      ln (s ^ ".1") dec.(0)
    end
    else begin
      ln (s ^ ".0") dec.(i - 1);
      convt (s ^ ".1") dec.(i - 1) dec.(i) (if i = 3 then 4 else 2)
    end;
    stage "decoder.stages" dec dec_depths i
  done;
  let grid =
    Nx.add_s (Nx.mul_s (Nx.arange_f Nx.float32 0. 8704. 1.) 0.8) 3500.
  in
  add "latent_grid._wavelength" (Nx.reshape [| 1; 8704 |] grid);
  add "normalization_quantizer.buckets" (Nx.linspace Nx.float32 (-1.) 5. 1024);
  Nx_io.Archive.of_list !entries

(* The transformer, with the tables of [ms]. *)
let transformer_archive ms =
  let draw = draws () and entries = ref [] in
  let add name x = entries := (name, Nx.P x) :: !entries in
  let norm name n =
    add (name ^ ".weight") (Nx.ones Nx.float32 [| n |]);
    add (name ^ ".bias") (Nx.zeros Nx.float32 [| n |])
  in
  let lin name o i = add (name ^ ".weight") (draw [| o; i |] ~fan:i) in
  let qk name =
    norm (name ^ ".q_norm") (width / heads);
    norm (name ^ ".k_norm") (width / heads)
  in
  let mlp name =
    lin (name ^ ".fc1") hidden width;
    lin (name ^ ".fc2") width hidden;
    lin (name ^ ".fc3") hidden width
  in
  for b = 0 to blocks - 1 do
    let e = Printf.sprintf "encoder.%d" b
    and d = Printf.sprintf "decoder.%d" b in
    norm (e ^ ".norm1") width;
    lin (e ^ ".attn.qkv") (3 * width) width;
    lin (e ^ ".attn.proj") width width;
    qk (e ^ ".attn");
    norm (e ^ ".norm2") width;
    mlp (e ^ ".mlp");
    norm (d ^ ".norm1") width;
    lin (d ^ ".self_attn.qkv") (3 * width) width;
    lin (d ^ ".self_attn.proj") width width;
    qk (d ^ ".self_attn");
    norm (d ^ ".query_norm") width;
    norm (d ^ ".context_norm") width;
    lin (d ^ ".cross_attn.q") width width;
    lin (d ^ ".cross_attn.kv") (2 * width) width;
    lin (d ^ ".cross_attn.proj") width width;
    qk (d ^ ".cross_attn");
    norm (d ^ ".norm2") width;
    mlp (d ^ ".mlp")
  done;
  List.iter
    (fun m ->
      let key = Modality.key m in
      let n = Modality.tokens m and v = Modality.vocab m in
      let pos = draw [| 1; n; width |] ~fan:1 in
      add
        ("encoder_embeddings." ^ key ^ ".token_emb.weight")
        (draw [| v; width |] ~fan:1);
      add ("encoder_embeddings." ^ key ^ ".pos_emb") pos;
      add ("decoder_embeddings." ^ key ^ ".pos_emb") pos;
      add
        ("decoder_embeddings." ^ key ^ ".mod_emb")
        (draw [| 1; 1; width |] ~fan:1);
      lin ("decoder_embeddings." ^ key ^ ".to_logits") v width)
    ms;
  norm "encoder_norm" width;
  norm "decoder_norm" width;
  lin "decoder_proj_context" width width;
  add "decoder_proj_context.bias" (Nx.zeros Nx.float32 [| width |]);
  add "mask_token" (draw [| 1; 1; width |] ~fan:1);
  Nx_io.Archive.of_list !entries

(* Inputs *)

let ls_bands = Image.[ Des_g; Des_r; Des_i; Des_z ]

let cutout () =
  Nx.Rng.normal ~key:(Nx.Rng.key 101) Nx.float32 [| 1; 4; 160; 160 |]

(* A DESI spectrum: 7781 pixels from 3600 Å at 0.8 Å, as DESI's coadds and
   Multimodal Universe's rows hold them. *)
let desi_spectrum () : (Nx.host Nx.float32_t, Nx.host) Spectrum.spectrum =
  let m = 7781 in
  {
    flux = Nx.Rng.normal ~key:(Nx.Rng.key 102) Nx.float32 [| 1; m |];
    ivar = Nx.ones Nx.float32 [| 1; m |];
    mask = Nx.zeros Nx.bool [| 1; m |];
    wavelength =
      Nx.reshape [| 1; m |]
        (Nx.add_s
           (Nx.mul_s (Nx.arange_f Nx.float32 0. (Float.of_int m) 1.) 0.8)
           3600.);
  }

let phot =
  Modality.[ Image; Scalar Flux_g; Scalar Flux_r; Scalar Flux_i; Scalar Flux_z ]

(* Tokens of [m] for one object, drawn over its vocabulary. *)
let tokens seed m : _ Tokens.t =
  let n = Modality.tokens m and v = Modality.vocab m in
  {
    modality = m;
    tokens =
      Nx.Rng.randint ~key:(Nx.Rng.key seed)
        (Nx.full Nx.int64 [| 1; n |] (Int64.of_int v));
    present = Nx.ones Nx.bool [| 1; n |];
  }

let inputs () = List.mapi (fun i m -> tokens (200 + i) m) phot

(* Cases *)

(* A case's weights and inputs are made in its worker, with one uncounted
   call. *)
let case name setup =
  Thumper.bench_with_setup name
    ~setup:(fun () ->
      let f = setup () in
      ignore (f ());
      Gc.full_major ();
      f)
    (fun f -> f ())

let image_encode () =
  let c = Image.of_archive (image_archive ()) and x = cutout () in
  fun () -> ignore (Image.encode c ls_bands x)

let spectrum_encode () =
  let c = Spectrum.of_archive (spectrum_archive ()) and x = desi_spectrum () in
  fun () -> ignore (Spectrum.encode c (Spectrum.pad Desi x))

let models = Modality.(Spectrum Desi :: Scalar Z :: phot)
let transformer () = Transformer.of_archive (transformer_archive models)

let encode () =
  let t = transformer () and i = inputs () in
  fun () -> ignore (Transformer.encode t i)

let decode () =
  let t = transformer () in
  let e = Transformer.encode t (inputs ()) in
  fun () ->
    ignore
      (Transformer.decode t e
         [ { modality = Spectrum Desi; positions = None } ])

let roar_step () =
  let t = transformer () and i = inputs () in
  fun () ->
    ignore (Roar.sample ~key:(Nx.Rng.key 7) ~reveal:[ 1 ] t i (Scalar Z))

let () =
  exit
    (Thumper.run "aion"
       ~config:Thumper.Config.(quick |> deadline 120.)
       [
         Thumper.group "cpu"
           [
             case "image encode" image_encode;
             case "spectrum encode" spectrum_encode;
             case "encode" encode;
             case "decode 273" decode;
             case "roar step" roar_step;
           ];
       ])
