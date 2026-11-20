(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compiled calls on this machine's first GPU, which has no eager kernels, on
   the narrow fixtures of upstream/ (the test dune file names their generators),
   each by the scheme of the suite that holds it on the host. The transformer is
   within 2 τ ŝ of the eager one, as in test_transformer, and start and step
   draw the eager tokens on fixed keys, as in test_roar. A set of codecs,
   compiled as main.exe compiles it, meets upstream's float64 reference within τ
   ŝ for the spectrum and image codecs, decisions near a boundary excepted, as
   in test_spectrum and test_image, and eager scalar codecs within nx's bounds,
   as in test_scalar. main.exe on the GPU computes as eager main.exe. It skips
   without a GPU, or without the compiler its kernels need. *)

open Windtrap
open Aion

let gpu =
  lazy
    (match Rig_gpus.first () with
    | Error _ -> None
    | Ok d -> Some (d, Nx.devices [ d ]))

(* The GPU's set, skipping where its renderer cannot be made: without its
   compiler's library each compiled call raises. *)
let on_gpu () =
  match Lazy.force gpu with
  | None -> skip ~reason:"no GPU" ()
  | Some (d, s) -> (
      match Tolk_engine.renderer d with
      | exception Failure why -> skip ~reason:why ()
      | _ -> s)

let floats x = Nx.to_array (Nx.cast Nx.float64 (Nx.place Nx.Host.on x))
let ints x = Array.map Int64.to_int (Nx.to_array (Nx.place Nx.Host.on x))

(* Every object, the step's positions listed in reverse. *)
let transformer (module D : Nx.Devices) =
  Compiled.agrees D.on (Compiled.transformer ()) ~rows:[ 0; 1; 2; 3 ]
    ~step:"roar.1" ~perm:[ 6; 5; 4; 3; 2; 1; 0 ] ~zs:[ 1; 2 ]

let draws temperature (module D : Nx.Devices) =
  let target = Modality.Scalar Xp_rp
  and x = Fixture.inputs Modality.[ Scalar Xp_bp; Scalar Flux_g; Scalar Z ]
  and keys = Nx.Rng.split_batch ~n:4 (Nx.Rng.key 23)
  and reveal = [ 7; 7; 7; 7; 7; 7; 7; 6 ] in
  let t = Fixture.transformer () in
  equal (array int)
    (ints (Roar.sample ~keys ~temperature ~reveal t x target))
    (ints (Compiled.draw D.on ~keys ~temperature ~reveal target x))

(* The codecs: aion.t's Hub cache, of the narrow transformer and codebooks for
   its scalar modalities, with the narrow image and spectrum codecs planted. *)

(* [exe args], its output dropped, failing the test unless it exits 0. *)
let exec ?(env = [||]) exe args =
  let null = Unix.openfile Filename.null [ O_WRONLY ] 0 in
  let env = Array.append env (Unix.environment ()) in
  let args = Array.of_list (exe :: args) in
  let pid = Unix.create_process_env exe args env Unix.stdin null Unix.stderr in
  Unix.close null;
  match Unix.waitpid [] pid with
  | _, WEXITED 0 -> ()
  | _ -> failf "%s failed" (String.concat " " (Array.to_list args))

let hub =
  lazy
    (let dir = Filename.temp_dir "aion" "gpu" in
     let cache = Filename.concat dir "huggingface" in
     exec "bin/cache.exe" [ "upstream/transformer.safetensors"; cache ];
     Hub_cache.plant cache (Hub.file Image_hsc) (Fixture.load "image_codec");
     Hub_cache.plant cache (Hub.file (Spectrum Desi))
       (Fixture.load "spectrum_codec");
     dir)

let codecs =
  lazy
    (Codecs.load
       ~cache_dir:(Filename.concat (Lazy.force hub) "huggingface")
       ~offline:true
       Modality.
         [ Image_hsc; Spectrum Desi; Scalar Flux_g; Scalar Xp_bp; Scalar Z ])

(* The set placed on the GPU with its encode and decode compiled once, as
   main.exe compiles them: the scalars' property calls them many times. *)
type compiled =
  | On :
      'd Nx.Placement.t
      * ('d Codecs.data -> 'd Tokens.t)
      * (Modality.t -> 'd Nx.int64_t -> 'd Codecs.data)
      -> compiled

let compiled =
  lazy
    (let module D = (val on_gpu ()) in
     let open Nx.Ptree in
     let set () = instantiate (module Codecs)
     and data () = v Codecs.walk_data in
     let g = place (module Codecs) D.on (Lazy.force codecs) in
     let encode =
       Rune.jit (set () @-> data () @-> returns (v Tokens.walk)) Codecs.encode
     and decode =
       Rune.jit
         (set () @-> v Modality.walk @-> tensor @-> returns (data ()))
         Codecs.decode
     in
     On (D.on, encode g, decode g))

let host_data x = Nx.Ptree.place_fixed Codecs.walk_data Nx.Host.on x

let signs b =
  Array.map
    (fun b -> if b then 1. else -1.)
    (Nx.to_array (Nx.place Nx.Host.on b))

(* The spectrum codec against upstream on DESI spectra. *)
let spectrum _ =
  let (On (at, encode, decode)) = Lazy.force compiled in
  let a = Fixture.load "spectrum_golden" in
  let n = Fixture.ensemble a
  and f64 name = floats (Fixture.get a Nx.float64 name) in
  let lfq name =
    floats (Nx.transpose ~axes:[ 0; 2; 1 ] (Fixture.get a Nx.float64 name))
  in
  let d =
    Codecs.Spectrum
      { modality = Desi; spectrum = Fixture.spectra a; lengths = None }
  in
  let k =
    Nx.place Nx.Host.on
      (encode (Nx.Ptree.place_fixed Codecs.walk_data at d)).tokens
  in
  let levels =
    floats (Spectrum.levels (Codecs.spectrum (Lazy.force codecs) Desi))
  in
  Spread.holds "normalisation token"
    (Spread.levels ~n ~edges:(Array.sub levels 0 1023)
       ~reference:(f64 "f64.desi.norm_pre") ~spread:(f64 "s.desi.norm_pre")
       (ints (Nx.slice [ A; I 0 ] k)));
  Spread.holds "LFQ bits"
    (Spread.signs ~n ~reference:(lfq "f64.desi.pre") ~spread:(lfq "s.desi.pre")
       (floats
          (Lfq.decode Nx.float64 ~bits:10 (Nx.slice [ A; R (1, Nx.dim 1 k) ] k))));
  match
    host_data
      (decode (Spectrum Desi)
         (Nx.place at (Fixture.get a Nx.int64 "f64.desi.tokens")))
  with
  | Spectrum { spectrum = d; _ } ->
      Spread.holds "flux"
        (Spread.values ~n ~reference:(f64 "f64.desi.flux")
           ~spread:(f64 "s.desi.flux") (floats d.flux));
      Spread.holds "mask"
        (Spread.signs ~n ~reference:(f64 "f64.desi.logit")
           ~spread:(f64 "s.desi.logit") (signs d.mask))
  | _ -> fail "not a spectrum"

(* The image codec against upstream on an HSC image of two bands, decoded in HSC
   i, the survey's third band. *)
let image _ =
  let (On (at, encode, decode)) = Lazy.force compiled in
  let a = Fixture.load "image_golden" in
  let n = Fixture.ensemble a
  and f64 name = floats (Fixture.get a Nx.float64 name) in
  let s name = floats (Fixture.get a Nx.float32 name) in
  let d =
    Codecs.Image
      {
        bands = Image.[ Hsc_i; Hsc_y ];
        flux = Fixture.get a Nx.float32 "input.hsc";
      }
  in
  let k = (encode (Nx.Ptree.place_fixed Codecs.walk_data at d)).tokens in
  Spread.holds "token digits"
    (Spread.levels ~n ~edges:Decisions.digit_edges
       ~reference:(f64 "f64.hsc.bounded") ~spread:(s "s.order.hsc.bounded")
       (Decisions.digits (ints k)));
  match
    host_data
      (decode Image_hsc (Nx.place at (Fixture.get a Nx.int64 "f64.hsc.tokens")))
  with
  | Image { flux; _ } ->
      Spread.holds "decoded"
        (Spread.values ~n ~reference:(f64 "f64.hsc.decoded")
           ~spread:(s "s.order.hsc.decoded")
           (floats (Nx.slice [ A; R (2, 3) ] flux)))
  | _ -> fail "not an image"

(* main.exe on the GPU against main.exe --eager on the host, on aion.t's Hub
   cache and survey files: the same objects, embeddings and probabilities within
   2e-5 of the largest, as aion.t holds the compiled host's, the redshift
   codebook's values exactly, its transform the identity, and the draws aion.t's
   compiled run makes, none of their uniforms that close to a boundary. *)
let bin _ =
  let device =
    match Lazy.force gpu with
    | None -> skip ~reason:"no GPU" ()
    | Some (d, _) -> Rig.name d
  in
  let dir = Lazy.force hub in
  let out dev f = Filename.concat dir (f ^ "." ^ dev ^ ".npz") in
  List.iter
    (fun (dev, at) ->
      let main f args =
        exec
          ~env:[| "RAVEN_CACHE_ROOT=" ^ dir |]
          "../bin/main.exe"
          (String.split_on_char ' ' args @ at @ [ "--out"; out dev f ])
      in
      main "e" "embed --legacysurvey mmu/legacysurvey.h5 --gaia mmu/gaia.h5";
      main "z" "posterior --gaia mmu/gaia.h5 --target tok_z --inputs tok_xp_bp";
      main "g"
        "generate --gaia mmu/gaia.h5 --rows 0:2 --target tok_z --steps 1 \
         --draws 3 --seed 7")
    [ (device, [ "--device"; device ]); ("cpu", [ "--eager" ]) ];
  let read f dt name =
    let read dev = floats (Fixture.get (Nx_io.load_npz (out dev f)) dt name) in
    (read device, read "cpu")
  in
  let close f name =
    let x, y = read f Nx.float32 name in
    let big = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. y in
    let d = Array.mapi (fun i v -> Float.abs (v -. y.(i))) x in
    less ~msg:name float_exact ~than:(2e-5 *. big)
      (Array.fold_left Float.max 0. d)
  in
  let same f dt name =
    let x, y = read f dt name in
    equal ~msg:name (array float_exact) y x
  in
  same "e" Nx.int64 "rows";
  same "z" Nx.int64 "rows";
  close "e" "embedding";
  close "z" "p";
  close "z" "p_1024";
  same "z" Nx.float32 "upper";
  same "g" Nx.int64 "tokens"

let rows =
  cases "compiled on the first GPU" ~id:fst
    [
      ("the transformer within 2 τ ŝ of eager", transformer);
      ("draws at T = 1 are eager's", draws 1.);
      ("draws at T = 0 are eager's", draws 0.);
      ("the spectrum codec within τ ŝ of upstream", spectrum);
      ("the image codec within τ ŝ of upstream", image);
      ("main.exe as eager main.exe", bin);
    ]
    (fun (_, f) -> f (on_gpu ()))

(* Scalar codecs against the eager set on the host, of each transform: presence
   is eager's, a token eager's but where a rounding transform crosses an edge,
   and a value within nx's bounds of its edge's inverse. *)
let scalars =
  prop "compiled on the first GPU, scalar codecs compute as eager ones"
    Gen.(
      let* m = of_list Modality.[ Flux_g; Xp_bp; Z ] in
      let c = if m = Xp_bp then 55 else 1 in
      let value =
        frequency
          [
            (6, float_range (-5.) 10.);
            (1, of_list [ Float.nan; Float.infinity; -0. ]);
          ]
      in
      let* v = array ~size:(constant (4 * c)) value in
      let+ k = array ~size:(constant (4 * c)) (int_range 0 1023) in
      (m, c, v, k))
    (fun (m, c, v, k) ->
      let (On (at, encode, decode)) = Lazy.force compiled in
      cover "a non-finite value"
        (Array.exists (fun x -> not (Float.is_finite x)) v);
      let host = Lazy.force codecs and tr = Scalar.transform_of m in
      let edges = Scalar.edges (Codecs.scalar host m) in
      let edges i =
        Nx.to_array (Nx.cast Nx.float64 (Nx.slice [ I (i mod c) ] edges))
      in
      let shape = if c = 1 then [| 4 |] else [| 4; c |] in
      let d =
        Codecs.Scalar { modality = m; values = Nx.create Nx.float32 shape v }
      in
      let e = Codecs.encode host d in
      let x = encode (Nx.Ptree.place_fixed Codecs.walk_data at d) in
      equal (array bool) (Nx.to_array e.present)
        (Nx.to_array (Nx.place Nx.Host.on x.present));
      let got = ints x.tokens and refuted = ref [] in
      Array.iteri
        (fun i w ->
          if not (Scalar_bounds.crosses tr (edges i) v.(i) w got.(i)) then
            refuted := (i, w, got.(i)) :: !refuted)
        (ints e.tokens);
      equal (list (triple int int int)) [] !refuted;
      let k = Nx.create Nx.int64 shape (Array.map Int64.of_int k) in
      match host_data (decode (Scalar m) (Nx.place at k)) with
      | Scalar { values; _ } ->
          let k = ints k and outside = ref [] in
          Array.iteri
            (fun i y ->
              if
                not
                  (Bounds.inside (Scalar_bounds.inverse tr (edges i).(k.(i))) y)
              then outside := (i, y) :: !outside)
            (Nx.to_array values);
          equal (list (pair int float_exact)) [] !outside
      | _ -> fail "not a scalar's values")

let () = exit (run "aion.gpu" [ group "gpu" [ rows; scalars ] ])
