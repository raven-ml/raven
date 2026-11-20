(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The compiled transformer, sampler and codecs on this machine's first GPU,
   which has no eager kernels, on the narrow fixtures of upstream/ (the test
   dune file names their generators). The transformer computes within 2 τ ŝ of
   the eager one on the host, ŝ its fixture's rounding ensemble's spread, as
   test_transformer holds the compiled one on the host: both are float32
   computations of one exact value, each within τ ŝ of it. A sampler draws the
   eager tokens on fixed keys, as test_roar's do. The codecs meet upstream's
   float64 reference within τ ŝ, decisions near a boundary excepted, as
   test_spectrum's and test_image's do. It skips without a GPU, or without the
   compiler its kernels need. *)

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

let fixture name = Nx_io.load_safetensors ("upstream/" ^ name ^ ".safetensors")

let get a dt name =
  match Nx_io.Archive.find name a with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no entry " ^ name)

let floats x = Nx.to_array (Nx.cast Nx.float64 (Nx.place Nx.Host.on x))
let ints x = Array.map Int64.to_int (Nx.to_array (Nx.place Nx.Host.on x))

let holds name (c : Spread.check) =
  if c.failures <> [] then failf "%a" (Spread.pp name) c

(* The transformer *)

let weights = lazy (fixture "transformer")
let model = lazy (Transformer.of_archive (Lazy.force weights))
let golden = lazy (fixture "transformer_golden")
let roar = lazy (fixture "roar_golden")
let ensemble () =
  Int64.to_int (Nx.item [ 0 ] (get (Lazy.force golden) Nx.int64 "n"))

let inputs ?(rows = [ 0; 1; 2; 3 ]) () =
  let g = Lazy.force golden in
  let pick x =
    Nx.take ~axis:0
      (Nx.create Nx.int64
         [| List.length rows |]
         (Array.of_list (List.map Int64.of_int rows)))
      x
  in
  List.map
    (fun m ->
      let key = Modality.key m in
      let absent =
        Nx.not_equal_s (get g Nx.uint8 ("input." ^ key ^ ".absent")) 0
      in
      Tokens.
        {
          modality = m;
          tokens = pick (get g Nx.int64 ("input." ^ key ^ ".tokens"));
          present = Some (pick (Nx.logical_not absent));
        })
    Modality.[ Scalar Xp_bp; Scalar Flux_g; Scalar Z ]

(* The fixture's inputs and tok_xp_rp revealed as at the step [roar.1]. *)
let step_inputs () =
  let r = Lazy.force roar in
  inputs ()
  @ [
      Tokens.
        {
          modality = Scalar Xp_rp;
          tokens = get r Nx.int64 "roar.1.tokens";
          present = Some (Nx.not_equal_s (get r Nx.uint8 "roar.1.revealed") 0);
        };
    ]

(* Within 2 τ ŝ of eager's [y]. *)
let close name ~spread x y =
  holds name
    (Spread.values ~n:(ensemble ()) ~reference:(floats y)
       ~spread:(Array.map (fun s -> 2. *. s) (floats spread))
       (floats x))

let transformer =
  test "a compiled transformer computes as the eager one" (fun () ->
      let module D = (val on_gpu ()) in
      let t = Lazy.force model in
      let g = Transformer.jit (Transformer.place D.on t) in
      let on (i : _ Tokens.t) : _ Tokens.t =
        {
          i with
          tokens = Nx.place D.on i.tokens;
          present = Option.map (Nx.place D.on) i.present;
        }
      in
      let x = inputs () in
      close "context"
        ~spread:(get (Lazy.force golden) Nx.float32 "s.embed")
        (Transformer.encode g (List.map on x)).context
        (Transformer.encode t x).context;
      let p = get (Lazy.force roar) Nx.int64 "roar.1.positions" in
      let target p : _ Transformer.target =
        { modality = Scalar Xp_rp; positions = Some p }
      in
      let x = step_inputs () in
      close "a step's logits"
        ~spread:(get (Lazy.force roar) Nx.float32 "s.roar.1")
        (List.hd
           (Transformer.forward g (List.map on x) [ target (Nx.place D.on p) ]))
        (List.hd (Transformer.forward t x [ target p ]));
      let x = inputs ~rows:[ 1; 2 ] () in
      let z : _ Transformer.target =
        { modality = Scalar Z; positions = None }
      in
      close "redshift's logits"
        ~spread:(get (Lazy.force golden) Nx.float32 "s.logits.tok_z")
        (List.hd (Transformer.forward g (List.map on x) [ z ]))
        (List.hd (Transformer.forward t x [ z ])))

(* A sampler's draws are the eager ones' on fixed keys: logits that differ in
   their last bits move a draw only where its uniform lands that close to a
   boundary between two tokens, about 10⁻⁶ of a draw at float32's precision. *)
let sampler =
  cases "a sampler draws the eager tokens" ~id:(Printf.sprintf "%g") [ 1.; 0. ]
    (fun temperature ->
      let module D = (val on_gpu ()) in
      let t = Lazy.force model and target = Modality.Scalar Xp_rp in
      let s = Roar.sampler (Transformer.place D.on t) target in
      let on (i : _ Tokens.t) : _ Tokens.t =
        {
          i with
          tokens = Nx.place D.on i.tokens;
          present = Option.map (Nx.place D.on) i.present;
        }
      in
      let x = inputs () and key = Nx.Rng.key 23 in
      let reveal = [ 7; 7; 7; 7; 7; 7; 7; 6 ] in
      let eager = Roar.sample ~key ~temperature ~reveal t x target in
      let drawn =
        Roar.draw s ~key:(Nx.Rng.place D.on key) ~temperature ~reveal
          (List.map on x)
      in
      equal (array int) (ints eager) (ints drawn))

(* The spectrum codec, against upstream on a DESI spectrum. *)

let lfq a name = floats (Nx.transpose ~axes:[ 0; 2; 1 ] (get a Nx.float64 name))
let signs b =
  Array.map
    (fun b -> if b then 1. else -1.)
    (Nx.to_array (Nx.place Nx.Host.on b))

let spectrum =
  test "a compiled spectrum codec meets upstream" (fun () ->
      let module D = (val on_gpu ()) in
      let c = Spectrum.of_archive (fixture "spectrum_codec") in
      let g = Nx.Ptree.place (module Spectrum) D.on c in
      let codec () = Nx.Ptree.instantiate (module Spectrum) in
      let spectra () = Nx.Ptree.instantiate (module Spectrum.Spectra) in
      let encode =
        Rune.jit
          Nx.Ptree.(codec () @-> spectra () @-> returns tensor)
          (fun t x -> Spectrum.encode t (Spectrum.pad Modality.Desi x))
      and decode =
        Rune.jit
          Nx.Ptree.(codec () @-> tensor @-> returns (spectra ()))
          (fun t k -> Spectrum.decode t k)
      in
      let a = fixture "spectrum_golden" in
      let n = Int64.to_int (Nx.item [ 0 ] (get a Nx.int64 "n")) in
      let f64 name = floats (get a Nx.float64 name) in
      let x : _ Spectrum.spectrum =
        {
          flux = Nx.place D.on (get a Nx.float32 "input.flux");
          ivar = Nx.place D.on (get a Nx.float32 "input.ivar");
          mask = Nx.place D.on (Nx.not_equal_s (get a Nx.uint8 "input.mask") 0);
          wavelength = Nx.place D.on (get a Nx.float32 "input.lambda");
        }
      in
      let k = Nx.place Nx.Host.on (encode g x) in
      let edges = Array.sub (floats (Spectrum.levels c)) 0 1023 in
      holds "normalisation token"
        (Spread.levels ~n ~edges ~reference:(f64 "f64.desi.norm_pre")
           ~spread:(f64 "s.desi.norm_pre")
           (ints (Nx.slice [ A; I 0 ] k)));
      holds "LFQ bits"
        (Spread.signs ~n ~reference:(lfq a "f64.desi.pre")
           ~spread:(lfq a "s.desi.pre")
           (floats
              (Lfq.decode Nx.float64 ~bits:10
                 (Nx.slice [ A; R (1, Nx.dim 1 k) ] k))));
      let d = decode g (Nx.place D.on (get a Nx.int64 "f64.desi.tokens")) in
      holds "flux"
        (Spread.values ~n ~reference:(f64 "f64.desi.flux")
           ~spread:(f64 "s.desi.flux") (floats d.flux));
      holds "mask"
        (Spread.signs ~n ~reference:(f64 "f64.desi.logit")
           ~spread:(f64 "s.desi.logit") (signs d.mask)))

(* The image codec, against upstream on an HSC image of two bands, its token's
   digits as test_image counts them. *)

let digits k =
  let levels = [ 7; 5; 5; 5; 5 ] in
  Array.concat
    (List.map
       (fun k ->
         let k = Int64.to_int k in
         Array.of_list
           (List.rev
              (snd
                 (List.fold_left
                    (fun (k, ds) l -> (k / l, ((k mod l) - (l / 2) + 3) :: ds))
                    (k, []) levels))))
       (Array.to_list (Nx.to_array (Nx.place Nx.Host.on k))))

let image =
  test "a compiled image codec meets upstream" (fun () ->
      let module D = (val on_gpu ()) in
      let g =
        Nx.Ptree.place
          (module Image)
          D.on
          (Image.of_archive (fixture "image_codec"))
      in
      let signature () =
        Nx.Ptree.(
          instantiate (module Image)
          @-> list (v Image.walk_band)
          @-> tensor @-> returns tensor)
      in
      let encode = Rune.jit (signature ()) Image.encode
      and decode = Rune.jit (signature ()) Image.decode in
      let a = fixture "image_golden" in
      let n = Int64.to_int (Nx.item [ 0 ] (get a Nx.int64 "n")) in
      let f64 name = floats (get a Nx.float64 name) in
      let s name = floats (get a Nx.float32 name) in
      let bs = Image.[ Hsc_i; Hsc_y ] in
      let x = Nx.place D.on (get a Nx.float32 "input.hsc") in
      let k = encode g bs x in
      holds "token digits"
        (Spread.levels ~n
           ~edges:[| -2.5; -1.5; -0.5; 0.5; 1.5; 2.5 |]
           ~reference:(f64 "f64.hsc.bounded") ~spread:(s "s.order.hsc.bounded")
           (digits k));
      let k = Nx.place D.on (get a Nx.int64 "f64.hsc.tokens") in
      holds "decoded"
        (Spread.values ~n ~reference:(f64 "f64.hsc.decoded")
           ~spread:(s "s.order.hsc.decoded")
           (floats (decode g [ Image.Hsc_i ] k))))

let () =
  exit
    (run "aion.gpu" [ group "gpu" [ transformer; sampler; spectrum; image ] ])
