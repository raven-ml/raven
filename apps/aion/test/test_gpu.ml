(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The compiled transformer, draws and codecs on this machine's first GPU,
   which has no eager kernels, on the narrow fixtures of upstream/ (the test
   dune file names their generators). The transformer computes within 2 τ ŝ of
   the eager one on the host, ŝ its fixture's rounding ensemble's spread, as
   test_transformer holds the compiled one on the host: both are float32
   computations of one exact value, each within τ ŝ of it. Compiled draws are
   the eager tokens on fixed keys, as test_roar's are. A set of codecs,
   compiled as main.exe compiles it, meets upstream's float64 reference
   within τ ŝ, decisions near a boundary excepted, as test_spectrum's and
   test_image's codecs do, and its scalar codecs compute as eager ones within
   nx's bounds. main.exe on the GPU computes as eager main.exe. It skips
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
          present = pick (Nx.logical_not absent);
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
          present = Nx.not_equal_s (get r Nx.uint8 "roar.1.revealed") 0;
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
      let g = Nx.Ptree.place (module Transformer) D.on t in
      let model () = Nx.Ptree.instantiate (module Transformer) in
      let inputs_s () = Nx.Ptree.(list (v Tokens.walk)) in
      let encode =
        Rune.jit
          Nx.Ptree.(
            model () @-> inputs_s ()
            @-> returns (instantiate (module Transformer.Encoding)))
          Transformer.encode
      and forward =
        Rune.jit
          Nx.Ptree.(
            model () @-> inputs_s ()
            @-> list (v Transformer.walk_target)
            @-> returns (list tensor))
          Transformer.forward
      in
      let on = Nx.Ptree.place_fixed Tokens.walk D.on in
      let x = inputs () in
      close "context"
        ~spread:(get (Lazy.force golden) Nx.float32 "s.embed")
        (encode g (List.map on x)).context
        (Transformer.encode t x).context;
      let p = get (Lazy.force roar) Nx.int64 "roar.1.positions" in
      let target p : _ Transformer.target =
        { modality = Scalar Xp_rp; positions = Some p }
      in
      let x = step_inputs () in
      close "a step's logits"
        ~spread:(get (Lazy.force roar) Nx.float32 "s.roar.1")
        (List.hd (forward g (List.map on x) [ target (Nx.place D.on p) ]))
        (List.hd (Transformer.forward t x [ target p ]));
      let x = inputs ~rows:[ 1; 2 ] () in
      let z : _ Transformer.target =
        { modality = Scalar Z; positions = None }
      in
      close "redshift's logits"
        ~spread:(get (Lazy.force golden) Nx.float32 "s.logits.tok_z")
        (List.hd (forward g (List.map on x) [ z ]))
        (List.hd (Transformer.forward t x [ z ])))

(* Compiled start and step draw the eager tokens on fixed keys: logits that
   differ in their last bits move a draw only where its uniform lands that close
   to a boundary between two tokens, about 10⁻⁶ of a draw at float32's
   precision. *)
let draws =
  cases "compiled start and step draw the eager tokens"
    ~id:(Printf.sprintf "%g") [ 1.; 0. ] (fun temperature ->
      let module D = (val on_gpu ()) in
      let t = Lazy.force model and target = Modality.Scalar Xp_rp in
      let g = Nx.Ptree.place (module Transformer) D.on t in
      let draw () = Nx.Ptree.v Roar.walk_draw in
      let start =
        Rune.jit
          Nx.Ptree.(Nx.Rng.ptree @-> list (v Tokens.walk) @-> returns (draw ()))
          (fun keys x -> Roar.start ~keys ~temperature x target)
      and step =
        Rune.jit
          Nx.Ptree.(
            instantiate (module Transformer)
            @-> v Walk.int @-> draw () @-> returns (draw ()))
          Roar.step
      in
      let on = Nx.Ptree.place_fixed Tokens.walk D.on in
      let x = inputs () and keys = Nx.Rng.split_batch ~n:4 (Nx.Rng.key 23) in
      let reveal = [ 7; 7; 7; 7; 7; 7; 7; 6 ] in
      let eager = Roar.sample ~keys ~temperature ~reveal t x target in
      let d = start (Nx.Rng.place D.on keys) (List.map on x) in
      let drawn =
        Roar.tokens (List.fold_left (fun d k -> step g k d) d reveal)
      in
      equal (array int) (ints eager) (ints drawn))

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

(* The codecs: aion.t's Hub cache, of the narrow transformer and codebooks for
   its scalar modalities, with the narrow image and spectrum codecs planted, as
   a set placed on the GPU and compiled as main.exe compiles it. *)

let hub =
  lazy
    (let dir = Filename.temp_dir "aion" "gpu" in
     let cache = Filename.concat dir "huggingface" in
     exec "bin/cache.exe" [ "upstream/transformer.safetensors"; cache ];
     Hub_cache.plant cache (Hub.file Image_hsc) (fixture "image_codec");
     Hub_cache.plant cache (Hub.file (Spectrum Desi)) (fixture "spectrum_codec");
     dir)

let codecs =
  lazy
    (Codecs.load
       ~cache_dir:(Filename.concat (Lazy.force hub) "huggingface")
       ~offline:true
       Modality.
         [ Image_hsc; Spectrum Desi; Scalar Flux_g; Scalar Xp_bp; Scalar Z ])

(* The set on the GPU with its encode and decode compiled, once: the scalars'
   property calls them many times. *)
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

(* The spectrum codec, against upstream on a DESI spectrum. *)

let lfq a name = floats (Nx.transpose ~axes:[ 0; 2; 1 ] (get a Nx.float64 name))

let spectrum =
  test "a compiled set's spectrum codec meets upstream" (fun () ->
      match Lazy.force compiled with
      | On (at, encode, decode) -> (
          let a = fixture "spectrum_golden" in
          let n = Int64.to_int (Nx.item [ 0 ] (get a Nx.int64 "n")) in
          let f64 name = floats (get a Nx.float64 name) in
          let x : _ Spectrum.spectrum =
            {
              flux = get a Nx.float32 "input.flux";
              ivar = get a Nx.float32 "input.ivar";
              mask = Nx.not_equal_s (get a Nx.uint8 "input.mask") 0;
              wavelength = get a Nx.float32 "input.lambda";
            }
          in
          let d =
            Codecs.Spectrum { modality = Desi; spectrum = x; lengths = None }
          in
          let k =
            Nx.place Nx.Host.on
              (encode (Nx.Ptree.place_fixed Codecs.walk_data at d)).tokens
          in
          let c = Spectrum.of_archive (fixture "spectrum_codec") in
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
          let k = Nx.place at (get a Nx.int64 "f64.desi.tokens") in
          match host_data (decode (Spectrum Desi) k) with
          | Spectrum { spectrum = d; _ } ->
              holds "flux"
                (Spread.values ~n ~reference:(f64 "f64.desi.flux")
                   ~spread:(f64 "s.desi.flux") (floats d.flux));
              holds "mask"
                (Spread.signs ~n ~reference:(f64 "f64.desi.logit")
                   ~spread:(f64 "s.desi.logit") (signs d.mask))
          | _ -> fail "not a spectrum"))

(* The image codec, against upstream on an HSC image of two bands, decoded in
   HSC i, the survey's third band. *)

let image =
  test "a compiled set's image codec meets upstream" (fun () ->
      match Lazy.force compiled with
      | On (at, encode, decode) -> (
          let a = fixture "image_golden" in
          let n = Int64.to_int (Nx.item [ 0 ] (get a Nx.int64 "n")) in
          let f64 name = floats (get a Nx.float64 name) in
          let s name = floats (get a Nx.float32 name) in
          let d =
            Codecs.Image
              {
                bands = Image.[ Hsc_i; Hsc_y ];
                flux = get a Nx.float32 "input.hsc";
              }
          in
          let k =
            (encode (Nx.Ptree.place_fixed Codecs.walk_data at d)).tokens
          in
          holds "token digits"
            (Spread.levels ~n ~edges:Decisions.digit_edges
               ~reference:(f64 "f64.hsc.bounded")
               ~spread:(s "s.order.hsc.bounded")
               (Decisions.digits (ints k)));
          let k = Nx.place at (get a Nx.int64 "f64.hsc.tokens") in
          match host_data (decode Image_hsc k) with
          | Image { flux; _ } ->
              holds "decoded"
                (Spread.values ~n ~reference:(f64 "f64.hsc.decoded")
                   ~spread:(s "s.order.hsc.decoded")
                   (floats (Nx.slice [ A; R (2, 3) ] flux)))
          | _ -> fail "not an image"))

(* Scalar codecs, against the eager set on the host, of each transform: presence
   is eager's, a token eager's but where a rounding transform crosses an edge,
   and a value within nx's bounds of its edge's inverse, as test_parity holds
   eager ones. *)

let scalars =
  prop "a compiled set's scalar codecs compute as eager ones"
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
      match Lazy.force compiled with
      | On (at, encode, decode) -> (
          cover "a non-finite value"
            (Array.exists (fun x -> not (Float.is_finite x)) v);
          let host = Lazy.force codecs and tr = Scalar.transform_of m in
          let edges = Scalar.edges (Codecs.scalar host m) in
          let edges i =
            Nx.to_array (Nx.cast Nx.float64 (Nx.slice [ I (i mod c) ] edges))
          in
          let shape = if c = 1 then [| 4 |] else [| 4; c |] in
          let d =
            Codecs.Scalar
              { modality = m; values = Nx.create Nx.float32 shape v }
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
                  let bound = Scalar_bounds.inverse tr (edges i).(k.(i)) in
                  if not (Bounds.inside bound y) then
                    outside := (i, y) :: !outside)
                (Nx.to_array values);
              equal (list (pair int float_exact)) [] !outside
          | _ -> fail "not a scalar's values"))

(* main.exe on the GPU against main.exe eager on the host, on aion.t's Hub cache
   and survey files: the same objects, embeddings and probabilities within 2e-5
   of the largest, as aion.t holds cpu-jit's, the redshift codebook's values
   exactly, its transform the identity, and the draws aion.t's cpu-jit run
   makes, none of their uniforms that close to a boundary. *)

let bin =
  test "main.exe on the GPU computes as eager main.exe" (fun () ->
      let device =
        match Lazy.force gpu with
        | None -> skip ~reason:"no GPU" ()
        | Some (d, _) ->
            ignore (on_gpu ());
            Rig.name d
      in
      let dir = Lazy.force hub in
      let out dev f = Filename.concat dir (f ^ "." ^ dev ^ ".npz") in
      List.iter
        (fun dev ->
          let main f args =
            exec
              ~env:[| "RAVEN_CACHE_ROOT=" ^ dir |]
              "../bin/main.exe"
              (String.split_on_char ' ' args
              @ [ "--device"; dev; "--out"; out dev f ])
          in
          main "e" "embed --legacysurvey mmu/legacysurvey.h5 --gaia mmu/gaia.h5";
          main "z"
            "posterior --gaia mmu/gaia.h5 --target tok_z --inputs tok_xp_bp";
          main "g"
            "generate --gaia mmu/gaia.h5 --rows 0:2 --target tok_z --steps 1 \
             --draws 3 --seed 7")
        [ device; "cpu" ];
      let read f dt name =
        let read dev = floats (get (Nx_io.load_npz (out dev f)) dt name) in
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
      same "g" Nx.int64 "tokens")

let () =
  exit
    (run "aion.gpu"
       [ group "gpu" [ transformer; draws; spectrum; image; scalars; bin ] ])
