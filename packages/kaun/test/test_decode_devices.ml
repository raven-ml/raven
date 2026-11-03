(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A tensor-parallel decode step over four memory devices of one set, which
   nx.cpu computes: a Llama-shaped decoder whose projections are split by
   columns into the heads and by rows out of them, and whose caches are split on
   their kv-heads axis. It generates the tokens one device generates, and its
   pools stay split. Each device's piece of a donated pool lends its memory to
   the step's pool there (rune.mli), so the step writes its pools in place. *)

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

let cache dtype ~slots at =
  List.init layers (fun _ ->
      Attention.Cache.make ~slots ~kv_heads:(kv_dim / head_dim) ~head_dim dtype
        at)

(* A projection cut along its inputs asks for its sum, the devices' terms
   ({!Nx.Placement.partial}): the residual add keeps it, and one [close]
   after it closes the sublayer. Where no device cuts a projection's
   inputs, both are the identity. *)
let close x = Nx.place (Nx.Placement.closed (Nx.placement x)) x

let block ~rope b cache index x =
  let a, cache =
    Attention.cached ~head_dim ~rope ~sum:true b.attn cache index
      (Rms_norm.apply b.attn_norm x)
  in
  let x = close (Nx.add x a) in
  let h = Rms_norm.apply b.ffn_norm x in
  let mlp =
    Linear.apply ~sum:true b.down
      (Nx.mul (Fn.silu (Linear.apply b.gate h)) (Linear.apply b.up h))
  in
  (close (Nx.add x mlp), cache)

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
    step (int64s at [| 1; 3 |] start) !index (cache Nx.float32 ~slots kv_at)
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

(* The model at bfloat16: each leaf cast. *)
let bfloat16 m =
  let bf x = Nx.cast Nx.bfloat16 x in
  let linear (l : _ Linear.t) = { Linear.w = bf l.w; b = None } in
  let norm (n : _ Rms_norm.t) = { Rms_norm.gamma = bf n.gamma } in
  let block b =
    {
      attn_norm = norm b.attn_norm;
      attn =
        {
          Attention.q = linear b.attn.q;
          k = linear b.attn.k;
          v = linear b.attn.v;
          out = linear b.attn.out;
        };
      ffn_norm = norm b.ffn_norm;
      gate = linear b.gate;
      up = linear b.up;
      down = linear b.down;
    }
  in
  {
    tok = { Embedding.table = bf m.tok.Embedding.table };
    blocks = List.map block m.blocks;
    norm = norm m.norm;
  }

(* At bfloat16 the prompt's logits over four devices, against one
   device's: the largest gap, a baseline, printed with [%h] to be exact
   on every system. A row projection rounds each device's term into
   bfloat16 and adds the four in bfloat16, which the norms, the attention
   and the head carry to the logits. *)
let test_bfloat16 () =
  let m = bfloat16 (model ()) in
  let slots = 8 in
  let first ~at ~kv_at m =
    let step = step ~rope:(Rope.place at (rope ())) m in
    let index =
      Cache_index.place at (Cache_index.rows ~context:7 [| 3 |])
    in
    let (_, scores), _ =
      step (int64s at [| 1; 3 |] [| 3; 14; 1 |]) index
        (cache Nx.bfloat16 ~slots kv_at)
    in
    Nx.to_array (Nx.cast Nx.float64 (Nx.place Nx.Host.on scores))
  in
  let one = first ~at:Nx.Host.on ~kv_at:Nx.Host.on m in
  let four = first ~at:Four.on ~kv_at:kv_heads (parallel m) in
  let gap = ref 0. in
  Array.iteri
    (fun i w -> gap := Float.max !gap (Float.abs (w -. four.(i))))
    one;
  expect (Printf.sprintf "largest gap %h" !gap) @@ __POS_OF__ {|
    largest gap 0x1.8p-5
  |}

(* One row projection at bfloat16: [x] of ones against a column of
   [1, 2^-8, 2^-8, 2^-8], one entry on each device. Asked for its sum, each
   device's term is exact in bfloat16, and {!Nx.place} closes the terms
   from the first, one bfloat16 addition each: [1 + 2^-8] is a tie that
   rounds to the even [1], and so twice more, giving [1], within
   [γ(4, u) Σ|term|], [u = 2^-8], of the exact [1 + 3 2^-8]. Closed in
   float32 and rounded once, the product is the tie [1 + 3 2^-8], which
   rounds to the even [1 + 2^-6]. *)
let test_row_projection () =
  let tiny = Float.ldexp 1. (-8) in
  let l =
    {
      Linear.w =
        Nx.place (Four.split ~axis:0)
          (Nx.create Nx.bfloat16 [| 4; 1 |] [| 1.; tiny; tiny; tiny |]);
      b = None;
    }
  in
  let x = Nx.place Four.on (Nx.ones Nx.bfloat16 [| 1; 4 |]) in
  let read y = Nx.item [ 0; 0 ] (Nx.place Nx.Host.on (close y)) in
  equal ~msg:"the sum, closed in bfloat16" float_exact 1.
    (read (Linear.apply ~sum:true l x));
  equal ~msg:"closed in float32" float_exact (1. +. (4. *. tiny))
    (read (Linear.apply l x))

(* Composites over a cut contracting axis *)

(* Values in [-1, 1) from [seed]. *)
let values seed shape =
  let st = Random.State.make [| seed |] in
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create Nx.float32 shape
    (Array.init n (fun _ -> Random.State.float st 2. -. 1.))

let u32 = Float.ldexp 1. (-24)
let gamma n = n *. u32 /. (1. -. (n *. u32))

let within ~msg ~tol want got =
  let want = Nx.to_array want and got = Nx.to_array (Nx.place Nx.Host.on got) in
  Array.iteri
    (fun i w ->
      if Float.abs (w -. got.(i)) > tol then
        failf "%s: element %d is %.9g over four devices, %.9g on one" msg i
          got.(i) w)
    want

(* Attention with [q] and [k] cut along their [d] features: the scores'
   product closes, so the result is one device's within its roundings. Each
   score is within [γ(d + 1, 2u) d scale] of the exact one on either side
   (nx.mli's {!Nx.contract}), [δ] apart; the weights then move by a factor
   within [e^(2δ)], at most [4δ] of their sum of one; the softmax rounds
   each weight within [(m + 11) u] on either side; and the value product of
   [m] terms within [γ(m + 1, 2u)] on either side, [|v| <= 1]. *)
let law_attention (d, m, seed) =
  cover "d of 4" (d = 4);
  cover "d of 8" (d = 8);
  let n = 3 in
  let q = values seed [| 2; n; d |] and k = values (seed + 1) [| 2; m; d |] in
  let v = values (seed + 2) [| 2; m; 4 |] in
  let one = Attention.scaled_dot_product_attention q k v in
  let cut = Four.split ~axis:2 in
  let four =
    Attention.scaled_dot_product_attention (Nx.place cut q) (Nx.place cut k)
      (Nx.place Four.on v)
  in
  let fd = Float.of_int d and fm = Float.of_int m in
  let delta = 2. *. gamma (2. *. (fd +. 1.)) *. fd /. sqrt fd in
  let tol =
    (4. *. delta)
    +. (2. *. (fm +. 11.) *. u32)
    +. (2. *. gamma (2. *. (fm +. 1.)))
  in
  within ~msg:"attention" ~tol one four

(* A convolution with [x] and the filters cut along their input channels:
   the product closes, so the result is one device's within twice the
   contraction's bound over its [K = c kh kw] products of entries in
   [-1, 1], [γ(K + 1, 2u) K], and the bias add's rounding on either side. *)
let law_conv (c, seed) =
  cover "4 channels" (c = 4);
  cover "8 channels" (c = 8);
  let kh = 2 and kw = 3 in
  let x = values seed [| 2; c; 5; 6 |] in
  let p =
    {
      Conv.w = values (seed + 1) [| 3; c; kh; kw |];
      b = Some (values (seed + 2) [| 3 |]);
    }
  in
  let one = Conv.apply p x in
  let cut = Four.split ~axis:1 in
  let four =
    Conv.apply
      { Conv.w = Nx.place cut p.w; b = Option.map (Nx.place Four.on) p.b }
      (Nx.place cut x)
  in
  let k = Float.of_int (c * kh * kw) in
  let tol = (2. *. gamma (2. *. (k +. 1.)) *. k) +. (2. *. u32 *. (k +. 1.)) in
  within ~msg:"conv" ~tol one four

let cut_gen =
  Gen.pair
    (Gen.map (fun b -> if b then 4 else 8) Gen.bool)
    (Gen.pair (Gen.int_range 1 9) (Gen.int_range 0 1_000_000))

let () =
  exit
    (run "kaun decode over devices"
       [
         group "tensor parallelism"
           [
             test "generation over four devices"
               test_generation_over_four_devices;
             test "a bfloat16 prompt over four devices" test_bfloat16;
             test "a bfloat16 row projection's sum rounds each term"
               test_row_projection;
           ];
         group "a cut contracting axis"
           [
             prop "attention gives one device's result within its roundings"
               cut_gen
               (fun (d, (m, seed)) -> law_attention (d, m, seed));
             prop "a convolution gives one device's result within its roundings"
               (Gen.pair
                  (Gen.map (fun b -> if b then 4 else 8) Gen.bool)
                  (Gen.int_range 0 1_000_000))
               law_conv;
           ];
       ])
