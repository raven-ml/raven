(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* main.exe COMMAND [OPTION...]: AION-1 on raven, on the CPU or a GPU.

   Both commands read objects from the Multimodal Universe's HDF5 files
   (Survey), one source per survey: an object is a row of the first source with
   a partner in every other, by AION-1's reciprocal 1″ rule. They tokenize the
   inputs with aion-base's codecs and run the transformer of --model. Each
   writes one npz file whose entry [rows], int64 [n; sources], holds each
   object's row in each source, in the order the sources were given.

   [embed] writes [embedding], float32 [n; width]: each object's embedding, the
   mean of the context over its present tokens (AION-1, §7.2, Eq. 7).

   [posterior] writes the decoder's distribution over a scalar target's classes,
   the exact softmax of its logits: [p], float32 [n; k], or [n; 55; k] for an XP
   target, over the [k] classes that decode to a value; [upper], float32 [k] or
   [55; k], the value each class decodes to, which is the upper edge of its bin,
   so a mean over [p] overstates the value by about half a bin; and for
   redshift, whose class 1024 decodes to no value, [p_1024], float32 [n], that
   class's probability.

   [generate] writes draws of a target modality's tokens given the inputs, by
   random-order generation (Roar.sample) in --steps steps of upstream's linear
   counts: the target's n tokens in near-equal counts, the larger first. One
   step draws each token independently given the inputs (AION-1, Eq. 6); n
   steps, one token each, draw from the joint distribution along a random order;
   the counts between lie between the two. [tokens], int64 [n; draws; k], holds
   each object's --draws draws of the target's k tokens, and the draws decoded
   by aion-base's codec follow: [values], float32 [n; draws], or [n; draws; 55]
   for an XP target, for a scalar target, NaN where the token is redshift's
   class 1024, which decodes to no value; [flux], float32 [n; draws; 8704], and
   [wavelength], float32 [8704], the codec's latent grid, for a spectrum;
   [flux], float32 [n; draws; c; 96; 96], for an image in its survey's c bands.
   Draw j of an object whose row in the first source is r is Roar.sample's with
   the object's key [Nx.Rng.fold_in (Nx.Rng.fold_in (Nx.Rng.key seed) r) j], so
   the draws are a function of the files, the options and the weights, and an
   object's do not depend on the objects batched with it where its logits agree
   bit for bit.

   An input whose value is NaN or infinite is absent for its object, and an
   object with no present input is left out. The objects kept are cut into
   batches of --batch, all full but the last, so a compiled call runs at most
   two shapes.

   --device names the devices where the codecs and the transformer compute, as
   Rig_gpus.open_all reads them: cpu (the default), a GPU (cuda, nv, amd or
   metal) or a comma list. They compute there compiled by Rune.jit, or with
   --eager eagerly, on the host. The inputs move to the devices and the results
   back. Compiled results agree with eager ones to the accuracy Rune.jit
   states, so a draw near a boundary between two tokens may differ. *)

open Aion

let fail fmt = Printf.ksprintf failwith fmt

(* [Arg] over the command's arguments, those after its name. *)
let parse usage specs =
  let usage = "main.exe " ^ usage in
  let anon a = raise (Arg.Bad ("unexpected argument " ^ a)) in
  try Arg.parse_argv ~current:(ref 1) Sys.argv (Arg.align specs) anon usage with
  | Arg.Help m ->
      print_string m;
      exit 0
  | Arg.Bad m ->
      prerr_string m;
      exit 2

let required name = function "" -> fail "%s is required" name | v -> v

(* The survey flags, each [FILE] or [FILE:GROUP], in the order given. *)
let sources = ref []

let source_specs =
  List.map
    (fun s ->
      let n = Survey.name s in
      ( "--" ^ n,
        Arg.String (fun v -> sources := (s, v) :: !sources),
        Printf.sprintf "FILE[:GROUP] a source of %s's datasets" n ))
    Survey.[ Legacy_survey; Hsc; Desi; Sdss; Gaia ]

let model_doc =
  "M base (the default) or large, AION-1's released weights, downloaded once; \
   or a safetensors file of aion-base's layout"

let model name =
  try
    Transformer.of_archive
      (match name with
      | "base" -> Hub.weights Base
      | "large" -> Hub.weights Large
      | file -> Nx_io.load_safetensors file)
  with Failure m | Sys_error m -> fail "--model %s: %s" name m

(* Computing at a device *)

let host x = Nx.place Nx.Host.on x
let model_s () = Nx.Ptree.instantiate (module Transformer)
let inputs_s () = Nx.Ptree.(list (v Tokens.walk))

(* The codecs [c] at [d], their encode and decode compiled there where [d]
   compiles, the codecs an argument: a program per modality and shape. *)
type 'd codecs = {
  codecs : ('d Nx.float32_t, 'd) Codecs.t;
  encode : 'd Codecs.data -> 'd Tokens.t;
  decode : Modality.t -> 'd Nx.int64_t -> 'd Codecs.data;
}

let codecs (d : _ Device.t) c =
  let c = Nx.Ptree.place (module Codecs) d.at c in
  if not d.jit then
    { codecs = c; encode = Codecs.encode c; decode = Codecs.decode c }
  else
    let open Nx.Ptree in
    let codecs () = instantiate (module Codecs) in
    let data () = v Codecs.walk_data in
    let encode =
      Rune.jit (codecs () @-> data () @-> returns (v Tokens.walk)) Codecs.encode
    and decode =
      Rune.jit
        (codecs () @-> v Modality.walk @-> tensor @-> returns (data ()))
        Codecs.decode
    in
    { codecs = c; encode = encode c; decode = decode c }

(* The tokens of the inputs of [objs] at [d]. *)
let encode (d : _ Device.t) codecs o objs =
  List.map
    (fun x -> codecs.encode (Nx.Ptree.place_fixed Codecs.walk_data d.at x))
    (Objects.read o objs)

(* The options and inputs shared by the commands *)

type run = {
  model : (Nx.host Nx.float32_t, Nx.host) Transformer.t;
  codecs : (Nx.host Nx.float32_t, Nx.host) Codecs.t;
  objects : Objects.t;
  target : Modality.t option;
  batch : int;
  device : string;
  eager : bool;
  out : string;
}

(* [specs] are the command's own options. [check], given, makes the command take
   --target, and refuses a target and the command's options before any file is
   read. *)
let run ~usage ?(specs = []) ?check () =
  let rows = ref "" and keys = ref "" and m = ref "base" and size = ref 16 in
  let out = ref "" and target_key = ref "" and device = ref "cpu" in
  let eager = ref false in
  let target_specs =
    if Option.is_some check then
      [ ("--target", Arg.Set_string target_key, "K the target's key") ]
    else []
  in
  parse usage
    (source_specs @ target_specs @ specs
    @ [
        ( "--rows",
          Arg.Set_string rows,
          "A:B the first source's rows (default all)" );
        ( "--inputs",
          Arg.Set_string keys,
          "K,... the inputs' keys (default: every modality the sources hold \
           and the model reads, each from the first source that holds it)" );
        ("--model", Arg.Set_string m, model_doc);
        ("--batch", Arg.Set_int size, "N objects a call (default 16)");
        ("--device", Arg.Set_string device, Device.doc);
        ("--eager", Arg.Set eager, Device.eager_doc);
        ("--out", Arg.Set_string out, "FILE the npz file written");
      ]);
  let out = required "--out" !out in
  if !size < 1 then fail "--batch %d: at least 1" !size;
  let target =
    Option.map
      (fun check ->
        let m = Objects.modality "--target" (required "--target" !target_key) in
        check m;
        m)
      check
  in
  let srcs = Objects.sources (List.rev !sources) in
  let t = model !m in
  (match target with
  | Some tg when not (List.mem tg (Transformer.modalities t)) ->
      fail "--target %s: the model predicts no %s" !target_key !target_key
  | _ -> ());
  let except = Option.to_list target in
  let reads = Transformer.modalities t in
  let ins = Objects.inputs srcs ~reads ~except !keys in
  let range = Objects.range (List.hd srcs) !rows in
  let codecs = Codecs.load (except @ List.map snd ins) in
  let objects = Objects.v srcs ins range codecs in
  let device = !device and eager = !eager in
  { model = t; codecs; objects; target; batch = !size; device; eager; out }

(* embed *)

let embed () =
  let r = run ~usage:"embed SOURCE... --out FILE [OPTION...]" () in
  let (module D) = Device.set r.device in
  let d = { Device.at = D.on; jit = not r.eager } in
  let t = Nx.Ptree.place (module Transformer) d.at r.model in
  let embed =
    Device.compile d
      Nx.Ptree.(model_s () @-> inputs_s () @-> returns tensor)
      (fun t x -> Transformer.embedding (Transformer.encode t x))
  in
  let codecs = codecs d r.codecs in
  let width = Transformer.width r.model in
  let e = Out.v Nx.float32 [| Objects.length r.objects; width |] in
  let batches = Objects.batches r.objects r.batch in
  List.iter
    (fun (first, objs) ->
      Out.set e first (host (embed t (encode d codecs r.objects objs))))
    batches;
  Out.save r.out r.objects [ Out.entry "embedding" e ];
  Objects.summary "embed" r.objects ~batches:(List.length batches)

(* posterior *)

let posterior () =
  let scalar : Modality.t -> unit = function
    | Scalar _ -> ()
    | m -> fail "--target %s: not a scalar modality" (Modality.key m)
  in
  let r =
    run ~usage:"posterior SOURCE... --target K --out FILE [OPTION...]"
      ~check:scalar ()
  in
  let target = Option.get r.target in
  let s = match target with Scalar s -> s | _ -> assert false in
  let (module D) = Device.set r.device in
  let d = { Device.at = D.on; jit = not r.eager } in
  let t = Nx.Ptree.place (module Transformer) d.at r.model in
  let forward =
    Device.compile d
      Nx.Ptree.(
        model_s () @-> inputs_s ()
        @-> list (v Transformer.walk_target)
        @-> returns (list tensor))
      Transformer.forward
  in
  let codecs = codecs d r.codecs in
  let edges = Scalar.edges (Codecs.scalar codecs.codecs s) in
  let c = Nx.dim 0 edges and k = Nx.dim 1 edges in
  let vocab = Modality.vocab target in
  (* Each class's value, [[|k|]] or [[|c; k|]]. *)
  let upper =
    let classes = Nx.arange Nx.int64 0 k 1 in
    let classes =
      if c = 1 then classes
      else Nx.broadcast_to [| k; c |] (Nx.reshape [| k; 1 |] classes)
    in
    match codecs.decode target (Nx.place d.at classes) with
    | Scalar { values; _ } ->
        let values = host values in
        if c = 1 then values else Nx.transpose values
    | _ -> assert false
  in
  (* Each object's probabilities of the classes with a value, and of the
     others summed, from a softmax on the host in float64. *)
  let n = Objects.length r.objects in
  let p = Out.v Nx.float32 (if c = 1 then [| n; k |] else [| n; c; k |])
  and rest = Out.v Nx.float32 [| n |] in
  let batches = Objects.batches r.objects r.batch in
  List.iter
    (fun (first, objs) ->
      let x = encode d codecs r.objects objs in
      let logits =
        List.hd (forward t x [ { modality = target; positions = None } ])
      in
      let q = Nx.softmax ~axes:[ -1 ] (Nx.cast Nx.float64 (host logits)) in
      let pk = Nx.cast Nx.float32 (Nx.slice [ Nx.A; Nx.A; Nx.R (0, k) ] q) in
      Out.set p first (if c = 1 then Nx.squeeze ~axes:[ 1 ] pk else pk);
      if vocab > k then
        Out.set rest first
          (Nx.cast Nx.float32
             (Nx.sum ~axes:[ 1; 2 ]
                (Nx.slice [ Nx.A; Nx.A; Nx.R (k, vocab) ] q))))
    batches;
  let rest =
    if vocab = k then [] else [ Out.entry (Printf.sprintf "p_%d" k) rest ]
  in
  Out.save r.out r.objects
    (Out.entry "p" p :: ("upper", Nx.P upper) :: rest);
  Objects.summary "posterior" r.objects ~batches:(List.length batches)

(* generate *)

(* Upstream's linear counts: [n] tokens in [steps] near-equal counts, the larger
   first, as [n = 273] in 8 steps is [35] then seven [34]. *)
let linear n steps =
  List.init steps (fun s -> (n / steps) + if s < n mod steps then 1 else 0)

(* The side of a decoded image, whose pixels are 96 × 96 (Image.decode). *)
let side = 96

(* The draws [y], [[|b; k|]] at [d] and [h] on the host, decoded at [d]: each
   object's values or flux, and the number of tokens that decode to no value.
   A scalar's draws are read on the host, since a GPU set has no eager
   kernels, an image's or a spectrum's decode where they lie. A scalar
   codebook of [m] edges decodes the tokens below [m], the others cleared on
   the host; redshift's vocabulary holds one more. *)
let decoded (d : _ Device.t) codecs (target : Modality.t) h y =
  let decode y =
    Nx.Ptree.place_fixed Codecs.walk_data Nx.Host.on (codecs.decode target y)
  in
  match target with
  | Scalar s ->
      let m = Nx.dim 1 (Scalar.edges (Codecs.scalar codecs.codecs s)) in
      let x = if Modality.tokens target = 1 then Nx.reshape [| -1 |] h else h in
      let none = Nx.greater_equal_s x (Int64.of_int m) in
      let values =
        match decode (Nx.place d.at (Nx.where none (Nx.zeros_like x) x)) with
        | Scalar { values; _ } -> values
        | _ -> assert false
      in
      let missing = Nx.item [] (Nx.sum (Nx.cast Nx.int64 none)) in
      let values = Nx.where none (Nx.full_like values Float.nan) values in
      (values, Int64.to_int missing)
  | Spectrum _ -> (
      match decode y with
      | Spectrum { spectrum; _ } -> (spectrum.flux, 0)
      | _ -> assert false)
  | _ -> (
      match decode y with
      | Image { flux; _ } -> (flux, 0)
      | _ -> assert false)

let generate () =
  let steps = ref None and draws = ref 1 and seed = ref 0 in
  let temperature = ref 1. and reveal = ref [] in
  let specs =
    [
      ( "--steps",
        Arg.Int (fun s -> steps := Some s),
        "N the steps that reveal the target's tokens, in upstream's linear \
         counts (required): 1 draws each token independently given the inputs, \
         the target's number of tokens draws them one at a time" );
      ("--draws", Arg.Set_int draws, "M draws per object (default 1)");
      ( "--seed",
        Arg.Set_int seed,
        "S the seed of the draws' keys (default 0): an object's draws depend \
         on the seed and its row in the first source, so giving the sources \
         in another order changes them" );
      ( "--temperature",
        Arg.Set_float temperature,
        "T the temperature: 1 (the default) draws from the model's \
         distribution, 0 takes the most likely token" );
    ]
  in
  let check (m : Modality.t) =
    let key = Modality.key m in
    (match m with
    | Segmap | Catalog -> fail "--target %s: no codec decodes it" key
    | _ -> ());
    let n = Modality.tokens m in
    (match !steps with
    | None -> fail "--steps is required"
    | Some s when s < 1 || s > n ->
        fail "--steps %d: not in [1, %d], %s's number of tokens" s n key
    | Some s -> reveal := linear n s);
    if !draws < 1 then fail "--draws %d: at least 1" !draws;
    if !temperature < 0. || not (Float.is_finite !temperature) then
      fail "--temperature %g: not in [0, inf)" !temperature
  in
  let r =
    run ~usage:"generate SOURCE... --target K --steps N --out FILE [OPTION...]"
      ~specs ~check ()
  in
  let draws = !draws and temperature = !temperature and reveal = !reveal in
  let target = Option.get r.target in
  let (module D) = Device.set r.device in
  let d = { Device.at = D.on; jit = not r.eager } in
  let t = Nx.Ptree.place (module Transformer) d.at r.model in
  let codecs = codecs d r.codecs in
  (* Roar.sample's fold, compiled where [d] compiles: [step] keeps a program
     per count for the command's run. *)
  let draw =
    let draw_s () = Nx.Ptree.v Roar.walk_draw in
    let start =
      Device.compile d
        Nx.Ptree.(Nx.Rng.ptree @-> inputs_s () @-> returns (draw_s ()))
        (fun keys x -> Roar.start ~keys ~temperature x target)
    and step =
      Device.compile d
        Nx.Ptree.(
          model_s () @-> v Walk.int @-> draw_s () @-> returns (draw_s ()))
        Roar.step
    in
    fun ~keys x ->
      Roar.tokens
        (List.fold_left (fun r k -> step t k r) (start keys x) reveal)
  in
  let n = Objects.length r.objects in
  let k = Modality.tokens target in
  (* Each object's decoded draw, past its first two dimensions, and a
     spectrum's latent grid. *)
  let shape, grid =
    match target with
    | Scalar _ -> ((if k = 1 then [||] else [| k |]), [])
    | Spectrum s ->
        let g = Spectrum.grid (Codecs.spectrum r.codecs s) in
        ([| Nx.dim 0 g |], [ ("wavelength", Nx.P g) ])
    | _ -> ([| List.length (Codecs.bands target); side; side |], [])
  in
  let tokens = Out.v Nx.int64 [| n; draws; k |]
  and values = Out.v Nx.float32 (Array.append [| n; draws |] shape) in
  let missing = ref 0 in
  let root = Nx.Rng.key !seed in
  let batches = Objects.batches r.objects r.batch in
  List.iter
    (fun (first, objs) ->
      let x = encode d codecs r.objects objs in
      (* A key per object, of its row in the first source. *)
      let keys =
        Array.map (fun o -> Int64.of_int o.(0)) objs
        |> Nx.create Nx.int64 [| Array.length objs |]
        |> Nx.Rng.fold_in_tensor root
      in
      for j = 0 to draws - 1 do
        (* Keys are derived on the host, where every set computes. *)
        let keys = Nx.Rng.place d.at (Nx.Rng.fold_in keys j) in
        let y = draw ~keys x in
        let h = host y in
        let v, m = decoded d codecs target h y in
        Out.set tokens first ~at:[ Nx.I j ] h;
        Out.set values first ~at:[ Nx.I j ] v;
        missing := !missing + m
      done)
    batches;
  let name = match target with Scalar _ -> "values" | _ -> "flux" in
  Out.save r.out r.objects
    (Out.entry "tokens" tokens :: Out.entry name values :: grid);
  Objects.summary "generate" r.objects ~batches:(List.length batches);
  if !missing > 0 then
    Printf.printf
      "generate: %d draws of %s's class %d, which decodes to no value\n"
      !missing (Modality.key target)
      (Modality.vocab target - 1)

let commands =
  [ ("embed", embed); ("posterior", posterior); ("generate", generate) ]

(* A failure, such as a file the command cannot read, is printed with the
   command's refusals. *)
let () =
  match Array.to_list Sys.argv with
  | _ :: name :: _ when List.mem_assoc name commands -> (
      try (List.assoc name commands) ()
      with Failure m | Sys_error m | Invalid_argument m ->
        prerr_endline ("aion: " ^ m);
        exit 2)
  | _ ->
      prerr_endline
        ("usage: main.exe COMMAND [OPTION...], COMMAND one of "
        ^ String.concat ", " (List.map fst commands)
        ^ "; main.exe COMMAND --help lists its options");
      exit 2
