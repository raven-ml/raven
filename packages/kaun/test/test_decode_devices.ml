(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A tensor-parallel decode step over four memory devices of one set, which
   nx.cpu computes: a Llama-shaped decoder whose projections are split by
   columns into the heads and by rows out of them, and whose caches are split on
   their kv-heads axis. It generates the tokens one device generates, and its
   pools stay split. A compiled program over several devices is evaluated and
   lends no donated memory (rune.mli), so the step writes new pools. *)

open Windtrap
open Kaun

type ('a, 'd) block = {
  attn_norm : ('a, 'd) Rms_norm.t;
  attn : ('a, 'd) Attention.t;
  ffn_norm : ('a, 'd) Rms_norm.t;
  gate : ('a, 'd) Linear.t;
  up : ('a, 'd) Linear.t;
  down : ('a, 'd) Linear.t;
}

type ('a, 'd) model = {
  tok : ('a, 'd) Embedding.t;
  blocks : ('a, 'd) block list;
  norm : ('a, 'd) Rms_norm.t;
}

let caches () = Nx.Ptree.list (Nx.Ptree.instantiate (module Attention.Cache))

module Four =
  (val Nx.devices
         (List.init 4 (fun i ->
              Result.get_ok (Rig.memory_device (Printf.sprintf "kaun-tp-%d" i)))))

(* Four query heads and four kv-heads of two features: one of each per
   device. *)
let vocab = 17
and dim = 8
and head_dim = 2
and kv_dim = 8
and ffn = 16
and layers = 2

let rope () = Rope.make ~head_dim ~context:64 ()

let model () =
  Nx.Rng.with_key (Nx.Rng.key 40) @@ fun () ->
  let linear ~inputs ~outputs =
    Linear.make ~key:(Nx.Rng.next_key ()) ~bias:false ~inputs ~outputs
      Nx.float32
  in
  let block () =
    {
      attn_norm = Rms_norm.init ~dim Nx.Host.on;
      attn =
        Attention.make ~key:(Nx.Rng.next_key ()) ~bias:false ~kv_dim
          ~embed_dim:dim Nx.float32;
      ffn_norm = Rms_norm.init ~dim Nx.Host.on;
      gate = linear ~inputs:dim ~outputs:ffn;
      up = linear ~inputs:dim ~outputs:ffn;
      down = linear ~inputs:ffn ~outputs:dim;
    }
  in
  {
    tok = Embedding.init ~key:(Nx.Rng.next_key ()) ~vocab ~dim;
    blocks = List.init layers (fun _ -> block ());
    norm = Rms_norm.init ~dim Nx.Host.on;
  }

(* The weights where tensor parallelism puts them: into the heads and the hidden
   features by columns, out of them by rows, the rest a copy on each device. *)
let parallel m =
  let at p (l : _ Linear.t) = { Linear.w = Nx.place p l.w; b = None } in
  let columns = at (Four.split ~axis:1) and rows = at (Four.split ~axis:0) in
  let norm (n : _ Rms_norm.t) = { Rms_norm.gamma = Nx.place Four.on n.gamma } in
  let block b =
    {
      attn_norm = norm b.attn_norm;
      attn =
        {
          Attention.q = columns b.attn.q;
          k = columns b.attn.k;
          v = columns b.attn.v;
          out = rows b.attn.out;
        };
      ffn_norm = norm b.ffn_norm;
      gate = columns b.gate;
      up = columns b.up;
      down = rows b.down;
    }
  in
  {
    tok = { Embedding.table = Nx.place Four.on m.tok.table };
    blocks = List.map block m.blocks;
    norm = norm m.norm;
  }

let kv_heads = Four.split ~axis:1

let cache ~slots at =
  List.init layers (fun _ ->
      Attention.Cache.make ~slots ~kv_heads:(kv_dim / head_dim) ~head_dim
        Nx.float32 at)

let block ~rope b cache index x =
  let a, cache =
    Attention.cached ~head_dim ~rope b.attn cache index
      (Rms_norm.apply b.attn_norm x)
  in
  let x = Nx.add x a in
  let h = Rms_norm.apply b.ffn_norm x in
  let mlp =
    Linear.apply b.down
      (Nx.mul (Fn.silu (Linear.apply b.gate h)) (Linear.apply b.up h))
  in
  (Nx.add x mlp, cache)

let cached ~rope m caches index ids =
  let x, rev =
    List.fold_left2
      (fun (x, cs) b c ->
        let x, c = block ~rope b c index x in
        (x, c :: cs))
      (Embedding.apply m.tok ids, [])
      m.blocks caches
  in
  (x, List.rev rev)

let logits m h =
  Nx.matmul (Rms_norm.apply m.norm h) (Nx.transpose m.tok.Embedding.table)

let int64s at shape a =
  Nx.place at (Nx.create Nx.int64 shape (Array.map Int64.of_int a))

(* The decode step: the last token's logits and its greedy id, the caches
   donated and returned. *)
let step ~rope m =
  let caches = caches () in
  let step =
    Rune.jit
      Nx.Ptree.(
        tensor @-> v Cache_index.walk @-> caches
        @-> returns (pair (pair tensor tensor) caches))
      (fun ids index kv ->
        let h, kv = cached ~rope m kv index ids in
        let scores = logits m (Nx.slice [ A; I (Nx.dim 1 ids - 1) ] h) in
        ((Nx.reshape [| 1; 1 |] (Nx.argmax ~axis:1 scores), scores), kv))
  in
  fun ids index kv -> step ids index (Nx.Ptree.donate caches kv)

(* A prompt and four generated tokens at the set of [at]: the ids and the logits
   each came from. *)
let generate ~slots ~at ~kv_at m =
  let step = step ~rope:(Rope.place at (rope ())) m in
  let start = [| 3; 14; 1 |] in
  let context = Array.length start + 4 in
  let index =
    ref
      (Cache_index.place at
         (Cache_index.rows ~context [| Array.length start |]))
  in
  let (id, scores), kv =
    step (int64s at [| 1; 3 |] start) !index (cache ~slots kv_at)
  in
  let s = ref (id, kv) and out = ref [ (id, scores) ] in
  for _ = 1 to 4 do
    index := Cache_index.advance !index;
    let (id, scores), kv = step (fst !s) !index (snd !s) in
    out := (id, scores) :: !out;
    s := (id, kv)
  done;
  (List.rev !out, snd !s)

let generations () =
  let m = model () in
  let slots = 8 in
  let one, _ = generate ~slots ~at:Nx.Host.on ~kv_at:Nx.Host.on m in
  let four, kv = generate ~slots ~at:Four.on ~kv_at:kv_heads (parallel m) in
  (List.combine one four, kv)

let test_generation_over_four_devices () =
  let tokens, kv = generations () in
  List.iteri
    (fun i ((id, scores), (id', scores')) ->
      let msg what = Printf.sprintf "token %d, %s" i what in
      equal ~msg:(msg "id") (array int64) (Nx.to_array id) (Nx.to_array id');
      equal ~msg:(msg "logits")
        (array (float 1e-5))
        (Nx.to_array scores) (Nx.to_array scores'))
    tokens;
  let placement =
    Testable.make ~pp:Nx.Placement.pp ~equal:(fun a b -> Nx.Placement.equal a b)
  in
  List.iter
    (fun (c : _ Attention.Cache.t) ->
      equal ~msg:"keys stay split on their kv-heads" placement kv_heads
        (Nx.placement c.keys);
      equal ~msg:"values stay split on their kv-heads" placement kv_heads
        (Nx.placement c.values))
    kv

let () =
  exit
    (run "kaun decode over devices"
       [
         group "tensor parallelism"
           [
             test "generation over four devices"
               test_generation_over_four_devices;
           ];
       ])
