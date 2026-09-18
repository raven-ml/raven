(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The decode contract at the level of a model (RFC 0002). Kaun ships no models;
   this is the shape a decoder takes in user code, small enough to live in a
   test: one block body, [hidden] for whole sequences, [cached] for tokens that
   attend through key-value caches, and [logits] applied to either. The tests
   are the laws a model written this way must satisfy. *)

open Windtrap
open Kaun
module Span = Attention.Span

(* A Llama-shaped block: RMS norm, grouped-query attention with rotary
   positions, SwiGLU. *)

type 'a block = {
  attn_norm : 'a Rms_norm.t;
  attn : 'a Attention.t;
  ffn_norm : 'a Rms_norm.t;
  gate : 'a Linear.t;
  up : 'a Linear.t;
  down : 'a Linear.t;
}

type 'a model = {
  tok : 'a Embedding.t;
  blocks : 'a block list;
  norm : 'a Rms_norm.t;
}

let vocab = 17
and dim = 8
and head_dim = 2
and kv_dim = 4
and ffn = 16

let layers = 2
let rope = Rope.make ~head_dim ()

let model () =
  Nx.Rng.with_key (Nx.Rng.key 40) @@ fun () ->
  let linear ~inputs ~outputs =
    Linear.make ~bias:false ~inputs ~outputs Nx.float32
  in
  let block () =
    {
      attn_norm = Rms_norm.init ~dim;
      attn = Attention.make ~bias:false ~kv_dim ~embed_dim:dim Nx.float32;
      ffn_norm = Rms_norm.init ~dim;
      gate = linear ~inputs:dim ~outputs:ffn;
      up = linear ~inputs:dim ~outputs:ffn;
      down = linear ~inputs:ffn ~outputs:dim;
    }
  in
  {
    tok = Embedding.init ~vocab ~dim;
    blocks = List.init layers (fun _ -> block ());
    norm = Rms_norm.init ~dim;
  }

(* The block body is written once; [attend] is the attention to run and returns
   whatever state it carries. *)
let block ~attend b x =
  let a, carried = attend b.attn (Rms_norm.apply b.attn_norm x) in
  let x = Nx.add x a in
  let h = Rms_norm.apply b.ffn_norm x in
  let mlp =
    Linear.apply b.down
      (Nx.mul (Fn.silu (Linear.apply b.gate h)) (Linear.apply b.up h))
  in
  (Nx.add x mlp, carried)

let hidden m ids =
  let mask = Attention.causal_mask ~seq:(Nx.dim 1 ids) () in
  let attend a x = (Attention.apply ~head_dim ~mask ~rope a x, ()) in
  List.fold_left
    (fun x b -> fst (block ~attend b x))
    (Embedding.apply m.tok ids)
    m.blocks

let cache ~slots =
  List.init layers (fun _ ->
      Attention.Cache.make ~slots ~kv_heads:(kv_dim / head_dim) ~head_dim
        Nx.float32)

let cached m caches span ids =
  let slots = Nx.dim 0 (List.hd caches).Attention.Cache.keys in
  (* Resolved once: every block reads the same route. *)
  let route = Attention.route ~slots span in
  let x, rev =
    List.fold_left2
      (fun (x, cs) b c ->
        let attend a x = Attention.cached ~head_dim ~rope a c route x in
        let x, c = block ~attend b x in
        (x, c :: cs))
      (Embedding.apply m.tok ids, [])
      m.blocks caches
  in
  (x, List.rev rev)

let logits m h =
  Nx.matmul (Rms_norm.apply m.norm h) (Nx.transpose m.tok.Embedding.table)

let ids rows =
  let batch = Array.length rows and seq = Array.length rows.(0) in
  Nx.create Nx.int32 [| batch; seq |]
    (Array.map Int32.of_int (Array.concat (Array.to_list rows)))

let flat t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

(* The float32 tolerance of the law, relative to the logit's size. *)
let close ~msg expected actual =
  let e = flat expected and a = flat actual in
  equal ~msg:(msg ^ ": length") int (Array.length e) (Array.length a);
  Array.iteri
    (fun i x ->
      let tol = 1e-4 *. Float.max 1.0 (Float.abs x) in
      if Float.abs (x -. a.(i)) > tol then
        fail (Printf.sprintf "%s: entry %d is %g, expected %g" msg i a.(i) x))
    e

let prompt = [| 3; 14; 1; 5; 9; 2; 6; 5; 3 |]
let whole m = logits m (hidden m (ids [| prompt |]))

let test_hidden_is_cached_over_a_fresh_cache () =
  let m = model () in
  let n = Array.length prompt in
  let h, _ =
    cached m (cache ~slots:n) (Span.rows ~context:n [| n |]) (ids [| prompt |])
  in
  close ~msg:"hidden = fst cached" (whole m) (logits m h)

let test_chunking_is_invariant () =
  let m = model () in
  let n = Array.length prompt in
  let slots = Nx.create Nx.int32 [| 1; n |] (Array.init n Int32.of_int) in
  let feed chunks =
    let _, hs, _ =
      List.fold_left
        (fun (at, hs, caches) len ->
          let pos =
            Nx.create Nx.int32 [| 1; len |]
              (Array.init len (fun i -> Int32.of_int (at + i)))
          in
          let h, caches =
            cached m caches (Span.make ~pos ~slots)
              (ids [| Array.sub prompt at len |])
          in
          (at + len, h :: hs, caches))
        (0, [], cache ~slots:n)
        chunks
    in
    logits m (Nx.concatenate ~axis:1 (List.rev hs))
  in
  let expected = whole m in
  close ~msg:"chunks of 1" expected (feed (List.init n (fun _ -> 1)));
  close ~msg:"chunks of 7 then 2" expected (feed [ 7; 2 ]);
  close ~msg:"the whole prompt" expected (feed [ n ])

(* Two prompts of different lengths in one left-padded batch: each row's last
   logits are those it has alone. *)
let test_ragged_batch () =
  let m = model () in
  let short = [| 4; 8; 15 |] and long = [| 16; 2; 3; 4; 2; 1 |] in
  let last row =
    logits m
      (Nx.slice [ A; I (Array.length row - 1) ] (hidden m (ids [| row |])))
  in
  let padded = Array.append (Array.make 3 0) short in
  let h, _ =
    cached m (cache ~slots:16)
      (Span.rows ~context:8 [| 3; 6 |])
      (ids [| padded; long |])
  in
  let batched = logits m (Nx.slice [ A; I 5 ] h) in
  close ~msg:"short row" (last short) (Nx.slice [ R (0, 1) ] batched);
  close ~msg:"long row" (last long) (Nx.slice [ R (1, 2) ] batched)

(* Greedy generation through a jitted, donated step equals re-running the whole
   sequence for every token. *)

type state = {
  token : Nx.int32_t;
  scores : Nx.float32_t; (* the logits [token] was taken from *)
  span : Span.t;
  caches : Nx.float32_t Attention.Cache.List.t;
}

module State = struct
  type t = state

  let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t)
      { token; scores; span; caches } =
    {
      token = f token;
      scores = f scores;
      span = Span.map f span;
      caches = Attention.Cache.List.map f caches;
    }

  let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) a b =
    {
      token = f a.token b.token;
      scores = f a.scores b.scores;
      span = Span.map2 f a.span b.span;
      caches = Attention.Cache.List.map2 f a.caches b.caches;
    }

  let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) { token; scores; span; caches } =
    f token;
    f scores;
    Span.iter f span;
    Attention.Cache.List.iter f caches
end

let test_generation_matches_recomputation () =
  let m = model () in
  let steps = 5 in
  let start = [| 3; 14; 1 |] in
  (* Tokens and the logits they were taken from, recomputing the whole sequence
     for every token. *)
  let recomputed =
    let seq = ref (Array.to_list start) in
    List.init steps (fun _ ->
        let row = Array.of_list !seq in
        let last =
          Nx.slice [ I 0; I (Array.length row - 1) ] (hidden m (ids [| row |]))
        in
        let scores = logits m last in
        let next = Int32.to_int (Nx.item [] (Nx.argmax ~axis:0 scores)) in
        seq := !seq @ [ next ];
        (next, scores))
  in
  let step { token; span; caches; scores = _ } =
    let seq = Nx.dim 1 token in
    let h, caches = cached m caches span token in
    let scores = logits m (Nx.slice [ A; I (seq - 1) ] h) in
    {
      token = Nx.reshape [| 1; 1 |] (Nx.argmax ~axis:1 scores);
      scores;
      span = Span.advance span;
      caches;
    }
  in
  let step = Rune.jit2 ~donate:true (module State) (module State) step in
  let context = Array.length start + steps in
  let s =
    ref
      (step
         {
           token = ids [| start |];
           scores = Nx.zeros Nx.float32 [| 1; vocab |];
           span = Span.rows ~context [| Array.length start |];
           caches = cache ~slots:context;
         })
  in
  List.iteri
    (fun i (next, scores) ->
      let msg = Printf.sprintf "step %d" i in
      close ~msg:(msg ^ ", logits") scores !s.scores;
      equal ~msg:(msg ^ ", token") int next
        (Int32.to_int (Nx.item [ 0; 0 ] !s.token));
      if i < steps - 1 then s := step !s)
    recomputed

let () =
  run "kaun decoder"
    [
      group "one definition"
        [
          test "hidden is cached over a fresh cache"
            test_hidden_is_cached_over_a_fresh_cache;
          test "chunking is invariant" test_chunking_is_invariant;
          test "a ragged batch matches each row alone" test_ragged_batch;
          test "jitted generation matches recomputation"
            test_generation_matches_recomputation;
        ];
    ]
