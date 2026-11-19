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
   Draw j of the objects of batch b, the b-th --batch objects in row order, is
   Roar.sample with the key [Nx.Rng.fold_in (Nx.Rng.fold_in (Nx.Rng.key seed) b)
   j], so the draws are a function of the files, the options and the weights.

   An input whose value is NaN or infinite is absent for its object, and an
   object with no present input is left out.

   --device picks where the transformer computes: cpu, eagerly (the default);
   cpu-jit, compiled on the host; or a GPU as Rig_gpus.open_ names it (cuda,
   nv, amd, metal), compiled there. The codecs compute on the host, compiled
   unless the device is cpu; tokens move to the device and its results back.
   A compiled device pads each batch to --batch rows, repeating its first
   kept object and dropping the copies' results, so that each call runs one
   program whatever the objects a batch keeps.
   Compiled results agree with eager ones to the accuracy Rune.jit states, so
   a draw near a boundary between two tokens may differ. *)

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

(* Sources *)

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

(* A path to an existing file is the file, whatever it holds; otherwise a last
   [:/] starts the group. *)
let load_source (s, v) =
  if Sys.file_exists v then Survey.load s v
  else
    match String.rindex_opt v ':' with
    | Some k when k + 1 < String.length v && v.[k + 1] = '/' ->
        Survey.load
          ~group:(String.sub v (k + 1) (String.length v - k - 1))
          s (String.sub v 0 k)
    | _ -> Survey.load s v

(* [A:B], the first source's rows from A to B, excluded. *)
let range first spec =
  let n = Survey.length first in
  match spec with
  | "" -> (0, n)
  | _ -> (
      match String.split_on_char ':' spec with
      | [ a; b ] -> (
          match (int_of_string_opt a, int_of_string_opt b) with
          | Some a, Some b when 0 <= a && a < b && b <= n -> (a, b)
          | _ -> fail "--rows %s: the first source has rows 0:%d" spec n)
      | _ -> fail "--rows %s: A:B expected" spec)

(* The objects: each row of [first] in [a, b) with a partner in every other
   source, as [rows.(o).(s)], object [o]'s row in source [s]. *)
let objects srcs (a, b) =
  let first = List.hd srcs in
  let partners =
    List.map
      (fun t ->
        let i, j = Survey.pair first t in
        let m = Hashtbl.create (Array.length i) in
        Array.iteri (fun k r -> Hashtbl.replace m r j.(k)) i;
        m)
      (List.tl srcs)
  in
  List.init (b - a) (fun k -> a + k)
  |> List.filter_map (fun r ->
      let js = List.map (fun m -> Hashtbl.find_opt m r) partners in
      if List.mem None js then None
      else Some (Array.of_list (r :: List.map Option.get js)))
  |> Array.of_list

(* Devices *)

type 'd device = { at : 'd Nx.Placement.t; jit : bool }
type 'a on = { on : 'd. 'd device -> 'a }

let device_doc =
  "D cpu (eager, the default), cpu-jit (compiled on the host) or a GPU by \
   name, compiled: cuda, nv, amd or metal"

let on name k =
  match name with
  | "cpu" -> k.on { at = Nx.Host.on; jit = false }
  | "cpu-jit" -> k.on { at = Nx.Host.on; jit = true }
  | name ->
      let gpu =
        match Rig_gpus.open_ name with
        | Ok g -> g
        | Error m -> fail "--device %s: %s" name m
      in
      let module D = (val Nx.devices [ gpu ]) in
      k.on { at = D.on; jit = true }

let transformer d t =
  let t = Transformer.place d.at t in
  if d.jit then Transformer.jit t else t

let codecs_at d c = if d.jit then Codecs.jit c else c

(* Compiled calls take batches padded to [--batch] rows, eager ones as they
   are. *)
let pad d size = if d.jit then Some size else None

let input_at d (i : _ Transformer.input) : _ Transformer.input =
  {
    i with
    tokens = Nx.place d.at i.tokens;
    present = Option.map (Nx.place d.at) i.present;
  }

let host x = Nx.place Nx.Host.on x

(* [Transformer.embedding], compiled where [d] compiles: a set without eager
   kernels computes only so. *)
let embedding (type d) (d : d device) : (_, d) Transformer.encoding -> _ =
  if not d.jit then Transformer.embedding
  else
    let encoding =
      Nx.Ptree.v (fun c (e : _ Transformer.encoding) ->
          let open Nx.Ptree.Walk in
          let context = field c "context" tensor e.context in
          let lengths = field c "lengths" tensor e.lengths in
          Transformer.{ context; lengths })
    in
    Rune.jit Nx.Ptree.(encoding @-> returns tensor) Transformer.embedding

(* Models and inputs *)

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

(* The modality of [key], which the option [flag] gave. *)
let modality_of flag key =
  match List.find_opt (fun m -> Modality.key m = key) Modality.all with
  | Some m -> m
  | None -> fail "%s %s: names no modality" flag key

(* Each input with the index of the first source that holds it: those [keys]
   name, or every modality the sources hold and [t] reads but [except]. *)
let inputs t srcs ~except keys =
  let held m =
    let rec go i = function
      | [] -> None
      | s :: l ->
          if List.mem m (Survey.modalities s) then Some i else go (i + 1) l
    in
    go 0 srcs
  in
  let reads = Transformer.modalities t in
  match keys with
  | "" ->
      List.filter_map
        (fun m ->
          if List.mem m except || not (List.mem m reads) then None
          else Option.map (fun i -> (i, m)) (held m))
        Modality.all
  | keys ->
      List.map
        (fun k ->
          let m = modality_of "--inputs" k in
          if List.mem m except then fail "--inputs %s: the target is no input" k;
          if not (List.mem m reads) then
            fail "--inputs %s: the model reads no %s" k k;
          match held m with
          | Some i -> (i, m)
          | None -> fail "--inputs %s: no source holds it" k)
        (String.split_on_char ',' keys)

(* Batches *)

let int64s a =
  Nx.create Nx.int64 [| Array.length a |] (Array.map Int64.of_int a)

(* Whether each of [b] objects has a present token in [input]. *)
let present b (i : _ Transformer.input) =
  match i.present with
  | None -> Array.make b true
  | Some p ->
      let p = if Nx.ndim p = 1 then p else Nx.any ~axes:[ 1 ] p in
      Nx.to_array p

(* The tokens of [objs]' inputs, and the objects with a present token. *)
(* A batch of [objs] padded to [pad] rows by repeating its first: the rows
   past the batch's are computed and dropped, so a compiled call sees one
   shape whatever the batch and the objects it keeps. *)
let padded pad a =
  match pad with
  | Some p when Array.length a > 0 && Array.length a < p ->
      Array.append a (Array.make (p - Array.length a) a.(0))
  | _ -> a

(* The tokens of [objs]' inputs, and the objects with a present token, whose
   tokens are the first rows, padded to [pad] rows. *)
let encode ?pad codecs srcs ins objs =
  let b = Array.length objs in
  let rows = padded pad objs in
  let read (s, m) =
    let rows = Array.map (fun o -> o.(s)) rows in
    Codecs.encode codecs (Survey.read (List.nth srcs s) rows m)
  in
  let tokens = List.map read ins in
  let n = Array.length rows in
  let kept =
    List.init b Fun.id
    |> List.filter (fun o -> List.exists (fun i -> (present n i).(o)) tokens)
    |> Array.of_list
  in
  let taken = padded pad kept in
  if Array.length taken = n && Array.length kept = b then (tokens, objs)
  else
    let take x = Nx.take ~axis:0 (int64s taken) x in
    ( List.map
        (fun (i : _ Transformer.input) ->
          { i with tokens = take i.tokens; present = Option.map take i.present })
        tokens,
      Array.map (fun o -> objs.(o)) kept )

(* The first [b] rows of [x], a batch's result. *)
let first b x = if Nx.dim 0 x = b then x else Nx.slice [ Nx.R (0, b) ] x

let chunks size a =
  let n = Array.length a in
  List.init
    ((n + size - 1) / size)
    (fun k -> Array.sub a (k * size) (min size (n - (k * size))))

(* The options and inputs shared by the commands, [specs] the command's own
   options. [check], given, makes the command take --target, and refuses a
   target and the command's options before any file is read. *)
let run ~usage ?(specs = []) ?check () =
  let rows = ref "" and keys = ref "" and m = ref "base" and size = ref 16 in
  let out = ref "" and target_key = ref "" and device = ref "cpu" in
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
        ("--device", Arg.Set_string device, device_doc);
        ("--out", Arg.Set_string out, "FILE the npz file written");
      ]);
  let out = required "--out" !out in
  if !size < 1 then fail "--batch %d: at least 1" !size;
  let target =
    Option.map
      (fun check ->
        let m = modality_of "--target" (required "--target" !target_key) in
        check m;
        m)
      check
  in
  let srcs = List.map load_source (List.rev !sources) in
  if srcs = [] then
    fail
      "no source: give one of --legacysurvey, --hsc, --desi, --sdss or --gaia";
  let t = model !m in
  (match target with
  | Some tg when not (List.mem tg (Transformer.modalities t)) ->
      fail "--target %s: the model predicts no %s" !target_key !target_key
  | _ -> ());
  let except = Option.to_list target in
  let ins = inputs t srcs ~except !keys in
  if ins = [] then fail "no input: the sources hold no modality the model reads";
  let a, b = range (List.hd srcs) !rows in
  let objs = objects srcs (a, b) in
  (t, srcs, ins, target, objs, (a, b), !size, out, !device)

let summary name ~range:(a, b) ~objs ~kept =
  Printf.printf
    "%s: %d objects; of rows %d:%d, %d without a partner in every source, %d \
     without a present input\n\
     %!"
    name kept a b
    (b - a - objs)
    (objs - kept)

let rows_entry rows k =
  Nx.create Nx.int64
    [| Array.length rows; k |]
    (Array.concat (Array.to_list (Array.map (Array.map Int64.of_int) rows)))

(* embed *)

let embed () =
  let t, srcs, ins, _, objs, range, size, out, device =
    run ~usage:"embed SOURCE... --out FILE [OPTION...]" ()
  in
  let parts =
    on device
      {
        on =
          (fun d ->
            let t = transformer d t and embedding = embedding d in
            let codecs = codecs_at d (Codecs.load (List.map snd ins)) in
            List.map
              (fun batch ->
                let tokens, kept =
                  encode ?pad:(pad d size) codecs srcs ins batch
                in
                let b = Array.length kept in
                if b = 0 then None
                else
                  let e = Transformer.encode t (List.map (input_at d) tokens) in
                  Some (first b (host (embedding e)), kept))
              (chunks size objs)
            |> List.filter_map Fun.id);
      }
  in
  let kept = Array.concat (List.map snd parts) in
  let width = match parts with (e, _) :: _ -> Nx.dim 1 e | [] -> 0 in
  let e =
    match parts with
    | [] -> Nx.zeros Nx.float32 [| 0; width |]
    | l -> Nx.concatenate ~axis:0 (List.map fst l)
  in
  Nx_io.save_npz out
    (Nx_io.Archive.of_list
       [
         ("rows", Nx.P (rows_entry kept (List.length srcs)));
         ("embedding", Nx.P e);
       ]);
  summary "embed" ~range ~objs:(Array.length objs) ~kept:(Array.length kept)

(* posterior *)

let posterior_on d t srcs ins (target : Modality.t option) objs range size
    out =
  let t = transformer d t in
  let target = Option.get target in
  let s = match target with Scalar s -> s | _ -> assert false in
  let codecs = codecs_at d (Codecs.load (target :: List.map snd ins)) in
  let edges = Scalar.edges (Codecs.scalar codecs s) in
  let c = Nx.dim 0 edges and k = Nx.dim 1 edges in
  let vocab = Modality.vocab target in
  (* Each class's value, [[|k|]] or [[|c; k|]]. *)
  let upper =
    let classes = Nx.arange Nx.int64 0 k 1 in
    let classes =
      if c = 1 then classes
      else Nx.broadcast_to [| k; c |] (Nx.reshape [| k; 1 |] classes)
    in
    match Codecs.decode codecs target classes with
    | Scalar { values; _ } -> if c = 1 then values else Nx.transpose values
    | _ -> assert false
  in
  (* The output, allocated once and filled batch by batch: each object's
     probabilities of the classes with a value, and of the others summed. *)
  let n = Array.length objs in
  let p = ref (Nx.zeros Nx.float32 [| n; c; k |]) in
  let rest = ref (Nx.zeros Nx.float32 [| n |]) in
  let filled = ref 0 in
  let kept =
    List.map
      (fun batch ->
        let tokens, kept = encode ?pad:(pad d size) codecs srcs ins batch in
        let b = Array.length kept in
        if b > 0 then begin
          let logits =
            List.hd
              (Transformer.forward t
                 (List.map (input_at d) tokens)
                 [ { modality = target; positions = None } ])
          in
          let logits = first b (host logits) in
          let q = Nx.softmax ~axes:[ -1 ] (Nx.cast Nx.float64 logits) in
          let at = [ Nx.R (!filled, !filled + b) ] in
          p :=
            Nx.set at
              (Nx.cast Nx.float32 (Nx.slice [ Nx.A; Nx.A; Nx.R (0, k) ] q))
              (Nx.donate !p);
          if vocab > k then
            rest :=
              Nx.set at
                (Nx.cast Nx.float32
                   (Nx.sum ~axes:[ 1; 2 ]
                      (Nx.slice [ Nx.A; Nx.A; Nx.R (k, vocab) ] q)))
                (Nx.donate !rest);
          filled := !filled + b
        end;
        kept)
      (chunks size objs)
    |> Array.concat
  in
  let m = Array.length kept in
  let p = Nx.slice [ Nx.R (0, m) ] !p in
  let p = if c = 1 then Nx.reshape [| m; k |] p else p in
  let rest =
    if vocab = k then []
    else [ (Printf.sprintf "p_%d" k, Nx.P (Nx.slice [ Nx.R (0, m) ] !rest)) ]
  in
  Nx_io.save_npz out
    (Nx_io.Archive.of_list
       ([
          ("rows", Nx.P (rows_entry kept (List.length srcs)));
          ("p", Nx.P p);
          ("upper", Nx.P upper);
        ]
       @ rest));
  summary "posterior" ~range ~objs:n ~kept:m

let posterior () =
  let scalar : Modality.t -> unit = function
    | Scalar _ -> ()
    | m -> fail "--target %s: not a scalar modality" (Modality.key m)
  in
  let t, srcs, ins, target, objs, range, size, out, device =
    run ~usage:"posterior SOURCE... --target K --out FILE [OPTION...]"
      ~check:scalar ()
  in
  on device
    { on = (fun d -> posterior_on d t srcs ins target objs range size out) }

(* generate *)

(* Upstream's linear counts: [n] tokens in [steps] near-equal counts, the larger
   first, as [n = 273] in 8 steps is [35] then seven [34]. *)
let linear n steps =
  List.init steps (fun s -> (n / steps) + if s < n mod steps then 1 else 0)

(* The entries of the draws [x], [[|b; k|]], decoded, and the number of tokens
   that decode to no value. A scalar codebook of [m] edges decodes the tokens
   below [m]; redshift's vocabulary holds one more. *)
let decoded codecs (target : Modality.t) ~rows x =
  let first x = first rows x in
  match target with
  | Scalar s ->
      let m = Nx.dim 1 (Scalar.edges (Codecs.scalar codecs s)) in
      let x = if Modality.tokens target = 1 then Nx.reshape [| -1 |] x else x in
      let none = Nx.greater_equal_s x (Int64.of_int m) in
      let values =
        match
          Codecs.decode codecs target (Nx.where none (Nx.zeros_like x) x)
        with
        | Scalar { values; _ } -> values
        | _ -> assert false
      in
      let none = first none in
      let missing = Nx.item [] (Nx.sum (Nx.cast Nx.int64 none)) in
      ( [
          ( "values",
            let values = first values in
            Nx.where none (Nx.full_like values Float.nan) values );
        ],
        Int64.to_int missing )
  | Spectrum _ -> (
      match Codecs.decode codecs target x with
      | Spectrum { spectrum; _ } ->
          ( [
              ("flux", first spectrum.flux);
              ("wavelength", Nx.slice [ Nx.I 0 ] spectrum.wavelength);
            ],
            0 )
      | _ -> assert false)
  | _ -> (
      match Codecs.decode codecs target x with
      | Image { flux; _ } -> ([ ("flux", first flux) ], 0)
      | _ -> assert false)

let generate_on d t srcs ins (target : Modality.t option) objs range size out
    ~draws ~seed ~temperature ~reveal =
  let t = transformer d t in
  let target = Option.get target in
  let codecs = codecs_at d (Codecs.load (target :: List.map snd ins)) in
  let root = Nx.Rng.key seed in
  (* A compiled device draws with a sampler, which keeps its programs for the
     command's run. *)
  let draw =
    if d.jit then
      let s = Roar.sampler t target in
      fun ~key x -> Roar.draw s ~key ~temperature ~reveal x
    else fun ~key x -> Roar.sample ~key ~temperature ~reveal t x target
  in
  (* The output, allocated once and filled batch by batch, as posterior's: the
     tokens, [[|n; draws; k|]], and each decoded entry likewise, allocated at
     the first decode, which gives its shape. A spectrum's wavelengths are one
     grid, kept once. *)
  let n = Array.length objs in
  let tokens =
    ref (Nx.zeros Nx.int64 [| n; draws; Modality.tokens target |])
  in
  let names = ref [] and buffers = Hashtbl.create 4 and grid = ref None in
  let missing = ref 0 and filled = ref 0 in
  let fill at (name, v) =
    if name = "wavelength" then
      begin if Option.is_none !grid then (
        grid := Some v;
        names := name :: !names)
      end
    else
      let buffer =
        match Hashtbl.find_opt buffers name with
        | Some b -> b
        | None ->
            let rest = Array.sub (Nx.shape v) 1 (Nx.ndim v - 1) in
            let shape = Array.append [| n; draws |] rest in
            names := name :: !names;
            Nx.zeros (Nx.dtype v) shape
      in
      Hashtbl.replace buffers name (Nx.set at v (Nx.donate buffer))
  in
  let batch c objs =
    let x, kept = encode ?pad:(pad d size) codecs srcs ins objs in
    let b = Array.length kept in
    if b > 0 then begin
      let key = Nx.Rng.fold_in root c in
      for j = 0 to draws - 1 do
        (* Keys are derived on the host, where every set computes. *)
        let key = Nx.Rng.place d.at (Nx.Rng.fold_in key j) in
        let y = host (draw ~key (List.map (input_at d) x)) in
        let at = [ Nx.R (!filled, !filled + b); Nx.I j ] in
        tokens := Nx.set at (first b y) (Nx.donate !tokens);
        let entries, m = decoded codecs target ~rows:b y in
        missing := !missing + m;
        List.iter (fill at) entries
      done;
      filled := !filled + b
    end;
    kept
  in
  let kept = Array.concat (List.mapi batch (chunks size objs)) in
  let m = Array.length kept in
  let first x = Nx.slice [ Nx.R (0, m) ] x in
  let entries =
    ("tokens", Nx.P (first !tokens))
    :: List.rev_map
         (fun name ->
           match !grid with
           | Some g when name = "wavelength" -> (name, Nx.P g)
           | _ -> (name, Nx.P (first (Hashtbl.find buffers name))))
         !names
  in
  Nx_io.save_npz out
    (Nx_io.Archive.of_list
       (("rows", Nx.P (rows_entry kept (List.length srcs))) :: entries));
  summary "generate" ~range ~objs:n ~kept:m;
  if !missing > 0 then
    Printf.printf
      "generate: %d draws of %s's class %d, which decodes to no value\n"
      !missing (Modality.key target)
      (Modality.vocab target - 1)

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
        "S the seed of every draw's key (default 0): the same seed reproduces \
         draws only with the same --rows and --batch" );
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
  let t, srcs, ins, target, objs, range, size, out, device =
    run ~usage:"generate SOURCE... --target K --steps N --out FILE [OPTION...]"
      ~specs ~check ()
  in
  on device
    {
      on =
        (fun d ->
          generate_on d t srcs ins target objs range size out ~draws:!draws
            ~seed:!seed ~temperature:!temperature ~reveal:!reveal);
    }

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
