(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The model against upstream's Whisper class.

   upstream/small.pt is a checkpoint of tiny's architecture at small dimensions,
   randomly initialised and saved as OpenAI saves its releases, every tensor
   float16; its encoder's positional table is moved off the sinusoids, so only
   the stored table gives upstream's outputs. upstream/heads.json is upstream's
   decoding of every release's alignment heads. upstream/small.npz holds the
   generator's inputs and upstream's outputs at float32 and float64.

   Each output is compared twice, as {!Tolerance} states: at float64 with
   upstream's float64 output, the check that both compute one function, and at
   float32 with upstream's float64 output, at most {!Tolerance.c} times
   upstream's own float32 distance to it. *)

open Windtrap
open Whisper_support
module M = Whisper.Model

let checkpoint = Nx_io.load_pytorch "upstream/small.pt"
let model = M.of_checkpoint Nx.float32 checkpoint

(* The checkpoint's own entries, as stored. *)
let state =
  match checkpoint with
  | Nx_io.Pytorch.Dict kvs ->
      List.assoc (Nx_io.Pytorch.String "model_state_dict") kvs
  | _ -> assert false

let stored ~shape name =
  Nx_io.Archive.tensor ~shape Nx.float16 name (Nx_io.Pytorch.tensors state)

let dims =
  test "dims are the checkpoint's" (fun () ->
      let d = model.M.dims in
      equal (list int)
        [ 80; 24; 64; 4; 2; 96; 16; 64; 4; 2 ]
        [
          d.n_mels;
          d.n_audio_ctx;
          d.n_audio_state;
          d.n_audio_head;
          d.n_audio_layer;
          d.n_vocab;
          d.n_text_ctx;
          d.n_text_state;
          d.n_text_head;
          d.n_text_layer;
        ])

(* [got] at [index] holds the float16 [want] at [want_index], exactly. *)
let same_at ~msg got index want want_index =
  equal ~msg float_exact (Nx.item want_index want) (Nx.item index got)

let layouts =
  test "weights hold the stored float16 values in kaun's layouts" (fun () ->
      let conv = stored ~shape:[| 64; 80; 3 |] "encoder.conv1.weight" in
      let w = model.encoder.conv1.w in
      equal (array int) [| 64; 3; 80 |] (Nx.shape w);
      List.iter
        (fun (o, k, i) -> same_at ~msg:"conv1" w [ o; k; i ] conv [ o; i; k ])
        [ (0, 0, 0); (5, 2, 79); (63, 1, 17) ];
      let query =
        stored ~shape:[| 64; 64 |] "decoder.blocks.1.cross_attn.query.weight"
      in
      let b = List.nth model.decoder.blocks 1 in
      let q = b.cross_attn.q.w in
      List.iter
        (fun (o, i) -> same_at ~msg:"query" q [ o; i ] query [ o; i ])
        [ (0, 0); (3, 60); (63, 9) ];
      is_none ~msg:"key bias" b.cross_attn.k.b;
      let pos = stored ~shape:[| 24; 64 |] "encoder.positional_embedding" in
      same_at ~msg:"positions" model.encoder.positions [ 23; 63 ] pos [ 23; 63 ])

let replace_state f =
  match checkpoint with
  | Nx_io.Pytorch.Dict kvs ->
      Nx_io.Pytorch.Dict
        (List.map
           (fun (k, v) ->
             if k = Nx_io.Pytorch.String "model_state_dict" then
               match v with
               | Nx_io.Pytorch.Dict s -> (k, Nx_io.Pytorch.Dict (f s))
               | _ -> assert false
             else (k, v))
           kvs)
  | _ -> assert false

let missing =
  test "a missing weight raises, naming it" (fun () ->
      let name = Nx_io.Pytorch.String "decoder.blocks.0.mlp.2.bias" in
      let c = replace_state (List.filter (fun (k, _) -> k <> name)) in
      raises_match (Exn.failure ~sub:"decoder.blocks.0.mlp.2.bias") (fun () ->
          M.of_checkpoint Nx.float32 c))

let heads =
  let models =
    Json.list (Json.mem "models" (Json.read "upstream/heads.json"))
  in
  cases "alignment_heads is upstream's decoding"
    ~id:(fun m -> Json.string (Json.mem "name" m))
    models
    (fun m ->
      let name = Json.string (Json.mem "name" m) in
      let r = require_some (M.release_of_string name) in
      let want =
        List.map
          (fun p ->
            match Json.ints p with [| l; h |] -> (l, h) | _ -> assert false)
          (Json.list (Json.mem "alignment_heads" m))
      in
      equal (list (pair int int)) want (M.alignment_heads r))

let aliases =
  test "large and turbo name large-v3 and large-v3-turbo" (fun () ->
      let heads name =
        M.alignment_heads (require_some (M.release_of_string name))
      in
      equal (list (pair int int)) (M.alignment_heads M.Large_v3) (heads "large");
      equal
        (list (pair int int))
        (M.alignment_heads M.Large_v3_turbo)
        (heads "turbo");
      is_none (M.release_of_string "large-v4"))

(* Every stored weight is one leaf of the structure, and casting the structure
   is loading at the dtype. *)
let structure =
  test "Params walks every weight once" (fun () ->
      let model64 = M.of_checkpoint Nx.float64 checkpoint in
      let params :
          ( ((float, Bigarray.float64_elt, Nx.host) Nx.t, Nx.host) M.t,
            Nx.host )
          Nx.Ptree.t =
        Nx.Ptree.instantiate (module M.Params)
      in
      let cast =
        Nx_io.Archive.of_value params
          (Nx.Ptree.cast (module M.Params) Nx.float64 model)
      in
      let loaded = Nx_io.Archive.of_value params model64 in
      equal int
        (List.length (Nx_io.Archive.names (Nx_io.Pytorch.tensors state)))
        (List.length (Nx_io.Archive.names loaded));
      equal (list string)
        (Nx_io.Archive.names loaded)
        (Nx_io.Archive.names cast);
      List.iter
        (fun name ->
          let get a =
            Nx.to_array
              (Nx.unpack Nx.float64 (Option.get (Nx_io.Archive.find name a)))
          in
          equal ~msg:name (array float_exact) (get loaded) (get cast))
        (Nx_io.Archive.names loaded))
(* Forward passes *)

let goldens = Nx_io.load_npz "upstream/small.npz"

let golden dt name =
  match Nx_io.Archive.find name goldens with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no golden " ^ name)

let f64 x = Nx.cast Nx.float64 x
let max_abs x = Nx.item [] (Nx.max (Nx.abs (f64 x)))
let max_diff a b = Nx.item [] (Nx.max (Nx.abs (Nx.sub (f64 a) (f64 b))))

(* upstream's [[| n; n_mels; frames |]], channels last. *)
let mel = Nx.transpose ~axes:[ 0; 2; 1 ] (golden Nx.float32 "mel")
let model64 = M.of_checkpoint Nx.float64 checkpoint

(* The port at float64 computes upstream's function: [got64] is [up64] within
   {!Tolerance.same}. *)
let same depths up64 got64 =
  let tol = Tolerance.same depths ~n:(Nx.numel up64) (max_abs up64) in
  equal (array (float tol)) (Nx.to_array (f64 up64)) (Nx.to_array (f64 got64))

(* The port at float32 is as accurate as upstream: its distance to upstream's
   float64 output is at most {!Tolerance.c} times upstream's own. *)
let accurate ~up32 ~up64 got =
  let bound = Tolerance.c *. max_diff up32 up64 in
  at_most float_exact ~than:bound (max_diff up64 got)

let up32 = golden Nx.float32 "encoder"
let up64 = golden Nx.float64 "encoder.f64"

let encoder =
  test "encode is upstream's function, as accurate as upstream" (fun () ->
      same (Tolerance.encoder model.dims) up64 (M.encode model64 (f64 mel));
      accurate ~up32 ~up64 (M.encode model mel))

(* The parameters are an argument of the compiled function. *)
let params : ((Nx.host Nx.float32_t, Nx.host) M.t, Nx.host) Nx.Ptree.t =
  Nx.Ptree.instantiate (module M.Params)

let encoder_jit =
  test "Rune.jit of encode is as accurate as upstream" (fun () ->
      let encode =
        Rune.jit Nx.Ptree.(params @-> tensor @-> returns tensor) M.encode
      in
      accurate ~up32 ~up64 (encode model mel))

let encoder_shape =
  cases "encode refuses a mel of another shape"
    ~id:(fun s -> String.concat "x" (List.map string_of_int (Array.to_list s)))
    [ [| 1; 47; 80 |]; [| 1; 48; 79 |]; [| 48; 80 |] ]
    (fun s ->
      raises_match Exn.invalid_arg (fun () ->
          M.encode model (Nx.zeros Nx.float32 s)))

(* Upstream's logits over the whole prefix ([logits]) and fed as it decodes:
   three tokens, then one a call ([logits.cached]). *)
let tokens = golden Nx.int64 "tokens"

let lanes = Nx.dim 0 tokens
and length = Nx.dim 1 tokens

let prefix = 3

let whole m features =
  let audio = M.audio m features in
  let index = Kaun.Cache_index.whole ~batch:lanes ~seq:length () in
  M.logits m
    (fst (M.decode m audio (M.cache m ~slots:0 Nx.Host.on) index tokens))

(* A decoding call: the logits of [decode]'s stream. *)
let decode_logits m audio caches index ids =
  let h, caches = M.decode m audio caches index ids in
  (M.logits m h, caches)

(* Upstream's way of decoding: [prefix] tokens at once, then one a call, each
   call through [decode]. *)
let stepwise ?(decode = decode_logits) m features =
  let audio = M.audio m features in
  let context = m.M.dims.n_text_ctx in
  let caches = M.cache m ~slots:(lanes * context) Nx.Host.on in
  let first = Kaun.Cache_index.rows ~context (Array.make lanes prefix) in
  let ids a b = Nx.slice [ A; R (a, b) ] tokens in
  let logits, caches = decode m audio caches first (ids 0 prefix) in
  let rec go acc index caches i =
    if i = length then List.rev acc
    else
      let index = Kaun.Cache_index.advance index in
      let l, caches = decode m audio caches index (ids i (i + 1)) in
      go (l :: acc) index caches (i + 1)
  in
  Nx.concatenate ~axis:1 (go [ logits ] first caches prefix)

let logits32 = golden Nx.float32 "logits"
let cached32 = golden Nx.float32 "logits.cached"
let logits64 = golden Nx.float32 "logits.f64"

(* At float64 both convert the logits to float32 at the end (upstream:
   TextDecoder.forward's [.float()]), so the two round values far closer than a
   float32 step and differ by at most one, [2 u y] with [u = 2{^-24}]; the
   float64 golden is that rounding too. *)
let same_rounded want got =
  let tol = 2. *. Float.ldexp 1. (-24) *. max_abs want in
  equal (array (float tol)) (Nx.to_array (f64 want)) (Nx.to_array (f64 got))

let decoder =
  test
    "decode is upstream's function, as accurate as upstream, whole and token \
     by token" (fun () ->
      let features64 = M.encode model64 (f64 mel) in
      same_rounded logits64 (whole model64 features64);
      same_rounded logits64 (stepwise model64 features64);
      let features = M.encode model mel in
      accurate ~up32:logits32 ~up64:logits64 (whole model features);
      accurate ~up32:cached32 ~up64:logits64 (stepwise model features))

let decoder_jit =
  test "Rune.jit of decode over the cache is as accurate as upstream" (fun () ->
      let open Nx.Ptree in
      let audio = list (instantiate (module M.Audio)) in
      let caches = list (instantiate (module Kaun.Attention.Cache)) in
      let index = v Kaun.Cache_index.walk in
      let decode =
        Rune.jit
          (params @-> audio @-> caches @-> index @-> tensor
          @-> returns (pair tensor caches))
          decode_logits
      in
      let features = M.encode model mel in
      accurate ~up32:cached32 ~up64:logits64 (stepwise ~decode model features))

(* A group of [g] lanes per audio: [decode] over each audio's keys and values
   once is [decode] over them repeated for every lane, bit for bit eagerly.
   Compiled, the two are two programs whose kernels may round differently, as a
   compiled one and an eager one do; the shared one is then as accurate as the
   repeated one eagerly ({!Tolerance}), against the repeated one at float64. The
   lanes' tokens differ, so that each lane's queries do. *)
let shared_decode m audio ids =
  let batch = Nx.dim 0 ids and seq = Nx.dim 1 ids in
  let index = Kaun.Cache_index.whole ~batch ~seq () in
  fst (M.decode m audio (M.cache m ~slots:0 Nx.Host.on) index ids)

let shared =
  let jitted =
    let open Nx.Ptree in
    Rune.jit
      (params
      @-> list (instantiate (module M.Audio))
      @-> tensor @-> returns tensor)
      shared_decode
  in
  cases "decode of a group of lanes shares its audio's keys and values"
    ~id:(fun (g, seq) -> Printf.sprintf "%d lanes, %d tokens" g seq)
    [ (1, 3); (2, 1); (3, 4); (5, 1) ]
    (fun (g, seq) ->
      let n = lanes * g in
      let ids =
        Nx.init Nx.int64 [| n; seq |] (fun i ->
            Int64.of_int
              ((Int64.to_int (Nx.item [ i.(0) / g; i.(1) ] tokens) + i.(0))
              mod model.dims.n_vocab))
      in
      let audio_of m features =
        let group =
          Nx.init Nx.int64 [| n |] (fun i -> Int64.of_int (i.(0) / g))
        in
        (M.audio m features, M.audio m (Nx.take ~axis:0 group features))
      in
      let audio, repeated = audio_of model (M.encode model mel) in
      let bits x = Nx.to_array (Nx.bitcast Nx.int32 x) in
      let per_lane = shared_decode model repeated ids in
      equal ~msg:"eager" (array int32) (bits per_lane)
        (bits (shared_decode model audio ids));
      let _, repeated64 = audio_of model64 (M.encode model64 (f64 mel)) in
      let up64 = shared_decode model64 repeated64 ids in
      accurate ~up32:per_lane ~up64 (jitted model audio ids))

let shared_refuses =
  test "decode refuses audio whose lanes do not divide the index's" (fun () ->
      let features = M.encode model mel in
      let audio = M.audio model features in
      let index = Kaun.Cache_index.whole ~batch:((2 * lanes) + 1) ~seq:1 () in
      raises_match Exn.invalid_arg (fun () ->
          M.decode model audio
            (M.cache model ~slots:0 Nx.Host.on)
            index
            (Nx.zeros Nx.int64 [| (2 * lanes) + 1; 1 |])))

let logits_refuses =
  test "logits refuses a stream of another width" (fun () ->
      raises_match Exn.invalid_arg (fun () ->
          M.logits model (Nx.zeros Nx.float32 [| 1; 2; 63 |]));
      raises_match Exn.invalid_arg (fun () ->
          M.logits model (Nx.zeros Nx.float32 [||])))

let decoder_refuses =
  test "decode refuses a context past n_text_ctx" (fun () ->
      let features = M.encode model mel in
      let audio = M.audio model features in
      let context = model.dims.n_text_ctx + 1 in
      let index = Kaun.Cache_index.rows ~context (Array.make lanes 1) in
      let caches = M.cache model ~slots:(lanes * context) Nx.Host.on in
      raises_match Exn.invalid_arg (fun () ->
          M.decode model audio caches index (Nx.slice [ A; R (0, 1) ] tokens)))

let () =
  exit
    (run "whisper.model"
       [
         group "checkpoint"
           [ dims; layouts; structure; missing; heads; aliases ];
         group "forward"
           [
             encoder;
             encoder_jit;
             encoder_shape;
             decoder;
             decoder_jit;
             decoder_refuses;
             shared;
             shared_refuses;
             logits_refuses;
           ];
       ])
