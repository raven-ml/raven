(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A tensor-parallel decode step over CPU:1..CPU:4: a Llama-shaped decoder whose
   projections are split by columns into the heads and by rows out of them, and
   whose caches are split on their kv-heads axis. It generates the tokens one
   device generates and writes every pool in its own storage on every device. *)

open Windtrap
open Kaun

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

let caches : Nx.float32_t Attention.Cache.t list Nx.Ptree.t =
  Nx.Ptree.list (Nx.Ptree.instantiate (module Attention.Cache))

(* Test devices over the host's memory, which the host addresses as it is. *)
let cpus = Nx.Device.all [ Cpu 1; Cpu 2; Cpu 3; Cpu 4 ]

(* Where each pool's storage starts on each device: a call that wrote the pools
   in their own storage returns them at the addresses it was given. *)
let addresses kv =
  let of_leaf x =
    match Nx.Repr.v x with
    | Placed p ->
        List.map Nx_device.Buffer.address
          (Nx.Repr.Storage.buffers (Nx.Repr.Placed.storage p))
    | Host _ | Traced _ -> []
  in
  Nx.Ptree.fold caches (fun _ t acc -> acc @ of_leaf t) kv []

(* Four query heads and four kv-heads of two features: one of each per
   device. *)
let vocab = 17
and dim = 8
and head_dim = 2
and kv_dim = 8
and ffn = 16
and layers = 2

let rope = Rope.make ~head_dim ~context:64 ()

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

(* The weights where tensor parallelism puts them: into the heads and the hidden
   features by columns, out of them by rows, the rest a copy on each device. *)
let parallel m =
  let at p (l : _ Linear.t) = { l with Linear.w = Nx.place p l.w } in
  let columns = at (Nx.Placement.sharded ~axis:1 cpus)
  and rows = at (Nx.Placement.sharded ~axis:0 cpus) in
  let block b =
    {
      b with
      attn =
        {
          Attention.q = columns b.attn.q;
          k = columns b.attn.k;
          v = columns b.attn.v;
          out = rows b.attn.out;
        };
      gate = columns b.gate;
      up = columns b.up;
      down = rows b.down;
    }
  in
  {
    m with
    tok =
      { Embedding.table = Nx.place (Nx.Placement.replicated cpus) m.tok.table };
    blocks = List.map block m.blocks;
  }

let kv_heads = Nx.Placement.sharded ~axis:1 cpus

let cache ?(place = Fun.id) ~slots () =
  List.init layers (fun _ ->
      let c =
        Attention.Cache.make ~slots ~kv_heads:(kv_dim / head_dim) ~head_dim
          Nx.float32
      in
      { Attention.Cache.keys = place c.keys; values = place c.values })

let block b cache index x =
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

let cached m caches index ids =
  let x, rev =
    List.fold_left2
      (fun (x, cs) b c ->
        let x, c = block b c index x in
        (x, c :: cs))
      (Embedding.apply m.tok ids, [])
      m.blocks caches
  in
  (x, List.rev rev)

let logits m h =
  Nx.matmul (Rms_norm.apply m.norm h) (Nx.transpose m.tok.Embedding.table)

let int64s shape a = Nx.create Nx.int64 shape (Array.map Int64.of_int a)

(* The decode step: the last token's logits and its greedy id, the caches
   consumed and returned. *)
let step m =
  Rune.jit
    Nx.Ptree.(
      tensor @-> Cache_index.ptree @-> consumes caches
      @@ returns (pair (pair tensor tensor) caches))
    (fun ids index kv ->
      let h, kv = cached m kv index ids in
      let scores = logits m (Nx.slice [ A; I (Nx.dim 1 ids - 1) ] h) in
      ((Nx.reshape [| 1; 1 |] (Nx.argmax ~axis:1 scores), scores), kv))

(* A prompt and four generated tokens: the ids, the logits each came from, and
   whether each call after the prompt returned the pools in their storage. *)
let generate ~slots ?place m =
  let step = step m in
  let start = [| 3; 14; 1 |] in
  let context = Array.length start + 4 in
  let index = ref (Cache_index.rows ~context [| Array.length start |]) in
  let (id, scores), kv =
    step (int64s [| 1; 3 |] start) !index (cache ?place ~slots ())
  in
  let s = ref (id, kv) and out = ref [ (id, scores, true) ] in
  for _ = 1 to 4 do
    index := Cache_index.advance !index;
    let before = addresses (snd !s) in
    let (id, scores), kv = step (fst !s) !index (snd !s) in
    out := (id, scores, before = addresses kv) :: !out;
    s := (id, kv)
  done;
  (List.rev !out, snd !s)

let test_generation_over_four_devices () =
  let m = model () in
  let slots = 8 in
  let one, _ = generate ~slots m in
  let four, kv = generate ~slots ~place:(Nx.place kv_heads) (parallel m) in
  List.iteri
    (fun i ((id, scores, _), (id', scores', lent)) ->
      let msg what = Printf.sprintf "token %d, %s" i what in
      equal ~msg:(msg "id") (array int64) (Nx.to_array id) (Nx.to_array id');
      equal ~msg:(msg "logits")
        (array (float 1e-5))
        (Nx.to_array scores) (Nx.to_array scores');
      is_true ~msg:(msg "every pool is written in its own storage") lent)
    (List.combine one four);
  List.iter
    (fun (c : _ Attention.Cache.t) ->
      is_true ~msg:"the pools stay split on their kv-heads"
        (Nx.Placement.equal kv_heads (Nx.placement c.keys)
        && Nx.Placement.equal kv_heads (Nx.placement c.values)))
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
