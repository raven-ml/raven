(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The rows of ../PERFORMANCE.md that a suite holds, those pytorch.py times
   with AION-1's own code. The first three are main.exe's commands on a batch
   of b objects, the model alone: tokens on the host in, the result read
   back, computed as main.exe computes a batch.
   - [embed/M/b]: the embedding of the tutorial file's inputs, its Legacy
     Survey image, nine Legacy Survey values and DESI spectrum, 858 tokens;
   - [posterior/M/b]: Transformer.forward to redshift from the image and the
     g, r, i and z fluxes, 581 tokens, and its softmax on the host;
   - [generate/M/b]: a draw of a DESI spectrum's 273 tokens from Gaia's XP
     coefficients, 110 tokens, in 12 steps of upstream's linear counts;
   - [codec/image/b], [codec/spectrum/b]: the image codec's encode of Legacy
     Survey cutouts, [b; 4; 160; 160], and the spectrum codec's of DESI
     spectra of 7781 pixels, data placed on the device and tokens read back;
   - [codec/spectrum-decode/b]: the spectrum codec's decode of 273 tokens on
     its grid.
   M is aion-base, and aion-large on a GPU. b is AION_BATCH, by default
   main.exe's 16 on a GPU and 1 on the host, where generate runs at 1
   whatever AION_BATCH is; on a GPU aion-base's commands run at 1 as well.

   AION_DEVICE names the device as main.exe's --device does, as
   Rig_gpus.open_ reads it: [cpu] (the default) or a GPU. The rows compute
   there compiled, as main.exe does, or eagerly on the host where AION_EAGER
   is set, as with --eager. The codecs compute on the same device. Each
   suite, [cpu], [cpu-eager] or the GPU's name, keeps a baseline file of its
   own. Each case opens its device and makes its weights in its worker,
   which thumper forks, so this process starts no driver; the worker then
   prints the device memory raven held, its cache included, or its peak
   resident memory on the host. [--warm] runs every case once in this
   process first, to fill tolk's disk cache. [--profile ROW] runs a row's
   call twice, the second between lines naming it and its wall time, for
   roles.py to sum DEBUG=2's kernel times.

   The weights are random, at the released architectures, aion-base's and
   aion-large's: the time is the architecture's, not the values'. The
   transformer holds the tables of the modalities the rows read and predict
   only, since the others take no part in these calls. *)

open Aion

(* Devices *)

type 'd target = { at : 'd Nx.Placement.t; jit : bool }
type make = { make : 'd. 'd target -> unit -> unit }

let device = Option.value ~default:"cpu" (Sys.getenv_opt "AION_DEVICE")
let eager = Option.is_some (Sys.getenv_opt "AION_EAGER")

(* Whether AION_DEVICE names a GPU, read from the name, so that this process
   opens no driver. *)
let gpu = String.lowercase_ascii device <> "cpu"
let suite = if eager then device ^ "-eager" else device

(* The call [c] makes at AION_DEVICE, and a function stating the memory it
   held. *)
let call c =
  let peak () =
    In_channel.with_open_text "/proc/self/status" In_channel.input_lines
    |> List.find_map (fun l -> Scanf.sscanf_opt l "VmHWM: %d kB" Fun.id)
    |> Option.fold ~none:"" ~some:(fun kb ->
        Printf.sprintf "peak resident memory %d MiB" (kb / 1024))
  in
  if not gpu then (c.make { at = Nx.Host.on; jit = not eager }, peak)
  else if eager then failwith "AION_EAGER: only the host computes eagerly"
  else
    let g = Result.error_to_failure (Rig_gpus.open_ device) in
    let module D = (val Nx.devices [ g ]) in
    ( c.make { at = D.on; jit = true },
      fun () ->
        Printf.sprintf "device memory %d MiB" (Rig.allocated g / 1048576) )

(* [f], compiled where [t] compiles, [s] its signature: main.exe's way. *)
let compile t s f = if t.jit then Rune.jit s f else f
let host x = Nx.place Nx.Host.on x
let key = Nx.Rng.key 7

(* Models *)

let ls_bands = Image.[ Des_g; Des_r; Des_i; Des_z ]

let fluxes =
  Modality.[ Scalar Flux_g; Scalar Flux_r; Scalar Flux_i; Scalar Flux_z ]

let embed_inputs =
  Modality.(
    (Image :: fluxes)
    @ [
        Scalar Flux_w1;
        Scalar Flux_w2;
        Scalar Flux_w3;
        Scalar Flux_w4;
        Scalar Ebv;
        Spectrum Desi;
      ])

let posterior_inputs = Modality.Image :: fluxes
let generate_inputs = Modality.[ Scalar Xp_bp; Scalar Xp_rp ]
let desi = Modality.Spectrum Desi
let z = Modality.Scalar Z
let modalities = (z :: embed_inputs) @ generate_inputs

(* aion-base's and aion-large's config.json and model.safetensors, at the
   commits Hub reads. *)
let base : Transformer.arch =
  { width = 768; heads = 12; blocks = (12, 12); hidden = 2048; modalities }

let large : Transformer.arch =
  { width = 1024; heads = 16; blocks = (24, 24); hidden = 2730; modalities }

let transformer t a =
  Nx.Ptree.place (module Transformer) t.at (Transformer.init ~key a)

(* The codecs of aion-base's codecs/image and codecs/spectrum. *)
let image t =
  Nx.Ptree.place
    (module Image)
    t.at
    (Image.init ~key { projected = 54; width = 128 })

let spectrum t =
  Nx.Ptree.place
    (module Spectrum)
    t.at
    (Spectrum.init ~key
       {
         encoder = [ (64, 3); (128, 3); (256, 27); (512, 3) ];
         decoder = [ (256, 3); (128, 9); (64, 3) ];
         output = 3;
         grid = 8704;
       })

(* Inputs *)

(* Tokens of [m] for [b] objects, drawn over its vocabulary, all present. *)
let tokens b seed m : _ Tokens.t =
  let n = Modality.tokens m and v = Modality.vocab m in
  {
    modality = m;
    tokens =
      Nx.Rng.randint ~key:(Nx.Rng.key seed)
        (Nx.full Nx.int64 [| b; n |] (Int64.of_int v));
    present = Nx.ones Nx.bool [| b; n |];
  }

let inputs b ms = List.mapi (fun i m -> tokens b (200 + i) m) ms
let placed t i = Nx.Ptree.place_fixed Tokens.walk t.at i

(* Upstream's linear counts, as main.exe's: [n] tokens in [steps] near-equal
   counts, the larger first. *)
let linear n steps =
  List.init steps (fun s -> (n / steps) + if s < n mod steps then 1 else 0)

(* Rows *)

let model_s () = Nx.Ptree.instantiate (module Transformer)
let inputs_s () = Nx.Ptree.(list (v Tokens.walk))

let embed a b =
  let make (type d) (t : d target) =
    let tr = transformer t a in
    let f =
      compile t
        Nx.Ptree.(model_s () @-> inputs_s () @-> returns tensor)
        (fun tr x -> Transformer.embedding (Transformer.encode tr x))
    in
    let x = inputs b embed_inputs in
    fun () -> ignore (host (f tr (List.map (placed t) x)))
  in
  { make }

let posterior a b =
  let make (type d) (t : d target) =
    let tr = transformer t a in
    let forward =
      compile t
        Nx.Ptree.(
          model_s () @-> inputs_s ()
          @-> list (v Transformer.walk_target)
          @-> returns (list tensor))
        Transformer.forward
    in
    let x = inputs b posterior_inputs in
    fun () ->
      let l =
        forward tr (List.map (placed t) x) [ { modality = z; positions = None } ]
      in
      ignore (Nx.softmax ~axes:[ -1 ] (Nx.cast Nx.float64 (host (List.hd l))))
  in
  { make }

let generate a b =
  let make (type d) (t : d target) =
    let tr = transformer t a in
    let x = inputs b generate_inputs in
    let reveal = linear (Modality.tokens desi) 12 in
    let draw_s () = Nx.Ptree.v Roar.walk_draw in
    let start =
      compile t
        Nx.Ptree.(Nx.Rng.ptree @-> inputs_s () @-> returns (draw_s ()))
        (fun keys x -> Roar.start ~keys x desi)
    and step =
      compile t
        Nx.Ptree.(
          model_s () @-> v Walk.int @-> draw_s () @-> returns (draw_s ()))
        Roar.step
    in
    (* A key per object, as main.exe keys each object by its row. *)
    let keys = Nx.Rng.fold_in_tensor key (Nx.arange Nx.int64 0 b 1) in
    fun () ->
      let d = start (Nx.Rng.place t.at keys) (List.map (placed t) x) in
      let d = List.fold_left (fun d k -> step tr k d) d reveal in
      ignore (host (Roar.tokens d))
  in
  { make }

let image_encode b =
  let make (type d) (t : d target) =
    let c = image t in
    let encode =
      compile t
        Nx.Ptree.(
          instantiate (module Image)
          @-> list (v Image.walk_band)
          @-> tensor @-> returns tensor)
        Image.encode
    in
    let x =
      Nx.Rng.normal ~key:(Nx.Rng.key 101) Nx.float32 [| b; 4; 160; 160 |]
    in
    fun () -> ignore (host (encode c ls_bands (Nx.place t.at x)))
  in
  { make }

(* DESI spectra: 7781 pixels from 3600 Å at 0.8 Å, as DESI's coadds and the
   Multimodal Universe's rows hold them, padded as Codecs.encode pads them. *)
let spectrum_encode b =
  let make (type d) (t : d target) =
    let c = spectrum t in
    let encode =
      compile t
        Nx.Ptree.(
          instantiate (module Spectrum)
          @-> instantiate (module Spectrum.Spectra)
          @-> returns tensor)
        (fun c x -> Spectrum.encode c (Spectrum.pad Desi x))
    in
    let m = 7781 in
    let flux = Nx.Rng.normal ~key:(Nx.Rng.key 102) Nx.float32 [| b; m |] in
    let ivar = Nx.ones Nx.float32 [| b; m |] in
    let mask = Nx.zeros Nx.bool [| b; m |] in
    let wavelength =
      Nx.copy
        (Nx.broadcast_to [| b; m |]
           (Nx.add_s
              (Nx.mul_s (Nx.arange_f Nx.float32 0. (Float.of_int m) 1.) 0.8)
              3600.))
    in
    fun () ->
      let x : _ Spectrum.spectrum =
        {
          flux = Nx.place t.at flux;
          ivar = Nx.place t.at ivar;
          mask = Nx.place t.at mask;
          wavelength = Nx.place t.at wavelength;
        }
      in
      ignore (host (encode c x))
  in
  { make }

let spectrum_decode b =
  let make (type d) (t : d target) =
    let c = spectrum t in
    let decode =
      compile t
        Nx.Ptree.(
          instantiate (module Spectrum)
          @-> tensor
          @-> returns (instantiate (module Spectrum.Spectra)))
        (fun c k -> Spectrum.decode c k)
    in
    let k =
      Nx.Rng.randint ~key:(Nx.Rng.key 103)
        (Nx.full Nx.int64 [| b; Modality.tokens desi |] 1024L)
    in
    fun () -> ignore (host (decode c (Nx.place t.at k)).flux)
  in
  { make }

(* The batch of the batched rows: AION_BATCH, by default main.exe's 16 on a
   GPU and 1 on the host, where generate runs at 1 whatever it is. A GPU's
   suite and the host's eager one run in about two minutes; the host's
   compiled one in about five, its calls 4 to 6 times eager's.
   PERFORMANCE.md's host rows run at 4. *)
let batch =
  Option.fold ~none:(if gpu then 16 else 1) ~some:int_of_string
    (Sys.getenv_opt "AION_BATCH")

let rows =
  let row name m b c = (Printf.sprintf "%s/%s/%d" name m b, c b) in
  (* The commands' rows of model [m] at a batch of [b]: generate at 1 on the
     host. *)
  let commands m a b =
    [
      row "embed" m b (embed a);
      row "posterior" m b (posterior a);
      row "generate" m (if gpu then b else 1) (generate a);
    ]
  in
  commands "base" base batch
  @ (if gpu && batch <> 1 then commands "base" base 1 else [])
  @ (if gpu then commands "large" large batch else [])
  @ [
      row "codec" "image" batch image_encode;
      row "codec" "spectrum" batch spectrum_encode;
      row "codec" "spectrum-decode" batch spectrum_decode;
    ]

(* A case's first calls, thumper's warmup, load its programs from tolk's
   disk cache. *)
let case (name, c) =
  Thumper.bench_with_setup name
    ~setup:(fun () ->
      let f, memory = call c in
      Gc.full_major ();
      (f, memory))
    ~teardown:(fun (_, memory) ->
      Printf.eprintf "aion: %s/%s: %s\n%!" suite name (memory ()))
    (fun (f, _) -> f ())

let () =
  match Array.to_list Sys.argv with
  | [ _; "--warm" ] ->
      if not eager then
        List.iter
          (fun (_, c) ->
            let f, _ = call c in
            f ();
            Gc.full_major ())
          rows
  | [ _; "--profile"; name ] ->
      let f, _ = call (List.assoc name rows) in
      f ();
      print_endline ("profile: " ^ name);
      let t0 = Unix.gettimeofday () in
      f ();
      Printf.printf "profile: end, the call in %.3f s\n%!"
        (Unix.gettimeofday () -. t0)
  | _ ->
      (* A baseline file per suite: aion.thumper for the host's compiled
         rows, aion-cuda.thumper for CUDA's. *)
      exit
        (Thumper.run
           (if suite = "cpu" then "aion" else "aion-" ^ suite)
           ~config:
             Thumper.Config.(
               default
               |> samples (if gpu then 5 else 3)
               |> warmup 0. |> deadline 600.)
           [ Thumper.group suite (List.map case rows) ])
