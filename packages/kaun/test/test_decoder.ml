(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The decode contract at the level of a model. Kaun ships no models; this is
   the shape a decoder takes in user code, small enough to live in a test: one
   fold over the blocks, [cached], that threads key-value caches along a cache
   index, [hidden], which is [cached] over a cache index that keeps nothing, and
   [logits] applied to either. The tests are the laws a model written this way
   must satisfy, each run eagerly and through a compiled, donated step. *)

open Windtrap
open Kaun

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

module Block = struct
  type 'a t = 'a block

  let walk c b =
    let open Nx.Ptree.Walk in
    let attn_norm = field c "attn_norm" Rms_norm.walk b.attn_norm in
    let attn = field c "attn" Attention.walk b.attn in
    let ffn_norm = field c "ffn_norm" Rms_norm.walk b.ffn_norm in
    let gate = field c "gate" Linear.walk b.gate in
    let up = field c "up" Linear.walk b.up in
    let down = field c "down" Linear.walk b.down in
    { attn_norm; attn; ffn_norm; gate; up; down }
end

module Model = struct
  type 'a t = 'a model

  let walk c m =
    let open Nx.Ptree.Walk in
    let tok = field c "tok" Embedding.walk m.tok in
    let blocks = field c "blocks" (list Block.walk) m.blocks in
    let norm = field c "norm" Rms_norm.walk m.norm in
    { tok; blocks; norm }
end

let model_ptree : Nx.float32_t model Nx.Ptree.t =
  Nx.Ptree.instantiate (module Model)

let caches : Nx.float32_t Attention.Cache.t list Nx.Ptree.t =
  Nx.Ptree.list (Nx.Ptree.instantiate (module Attention.Cache))

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

let cache ~slots =
  List.init layers (fun _ ->
      Attention.Cache.make ~slots ~kv_heads:(kv_dim / head_dim) ~head_dim
        Nx.float32)

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

let hidden ?lens m ids =
  let batch = Nx.dim 0 ids and seq = Nx.dim 1 ids in
  fst (cached m (cache ~slots:0) (Cache_index.whole ?lens ~batch ~seq ()) ids)

let logits m h =
  Nx.matmul (Rms_norm.apply m.norm h) (Nx.transpose m.tok.Embedding.table)

let int32s shape a = Nx.create Nx.int32 shape (Array.map Int32.of_int a)

let ids rows =
  let batch = Array.length rows and seq = Array.length rows.(0) in
  int32s [| batch; seq |] (Array.concat (Array.to_list rows))

let flat t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

(* The float32 tolerance of the law, relative to the logit's size. *)
let close ~msg expected actual =
  let e = flat expected and a = flat actual in
  equal ~msg:(msg ^ ": length") int (Array.length e) (Array.length a);
  Array.iteri
    (fun i x ->
      let tol = 1e-4 *. Float.max 1.0 (Float.abs x) in
      if not (Float.abs (x -. a.(i)) <= tol) then
        fail (Printf.sprintf "%s: entry %d is %g, expected %g" msg i a.(i) x))
    e

let finite ~msg t = is_true ~msg (Array.for_all Float.is_finite (flat t))

(* One call, eagerly or through a compiled step that reads the tokens and the
   index and consumes the caches. The step's state carries the stream it returns
   beside them; the stream it is given is a placeholder. *)

type query = { tokens : Nx.int32_t; index : Cache_index.t }

type result = {
  stream : Nx.float32_t;
  written : Nx.float32_t Attention.Cache.t list;
}

let query =
  Nx.Ptree.(
    iso
      (fun (tokens, index) -> { tokens; index })
      (fun { tokens; index } -> (tokens, index))
      (pair tensor Cache_index.ptree))

let result =
  Nx.Ptree.(
    iso
      (fun (stream, written) -> { stream; written })
      (fun { stream; written } -> (stream, written))
      (pair tensor caches))

let eager m caches index tokens = cached m caches index tokens

let compiled m =
  let step =
    Rune.jit_step query result (fun { tokens; index } { written; stream = _ } ->
        let stream, written = cached m written index tokens in
        { stream; written })
  in
  fun caches index tokens ->
    let placeholder = Nx.zeros Nx.float32 [| 1 |] in
    let { stream; written } =
      step { tokens; index } { stream = placeholder; written = caches }
    in
    (stream, written)

(* Every law below holds for both. *)
let both name f =
  [
    test (name ^ ", eager") (fun () ->
        let m = model () in
        f m (eager m));
    test (name ^ ", compiled and donated") (fun () ->
        let m = model () in
        f m (compiled m));
  ]

let prompt = [| 3; 14; 1; 5; 9; 2; 6; 5; 3 |]
let whole m = logits m (hidden m (ids [| prompt |]))

(* A cache index for [len] tokens of one sequence from position [at], the
   sequence held at [slots]. *)
let chunk ~slots ~at len =
  let n = Array.length slots in
  Cache_index.make
    ~pos:(int32s [| 1; len |] (Array.init len (fun i -> at + i)))
    ~table:(int32s [| 1; n |] slots)
    ()

(* [prompt] fed in [chunks] through [call], with [before] applied to the caches
   ahead of every call. *)
let feed_state ?(before = Fun.id) m call ~slots ~pool chunks =
  let _, hs, caches =
    List.fold_left
      (fun (at, hs, caches) len ->
        let h, caches =
          call (before caches) (chunk ~slots ~at len)
            (ids [| Array.sub prompt at len |])
        in
        (at + len, h :: hs, caches))
      (0, [], cache ~slots:pool)
      chunks
  in
  (logits m (Nx.concatenate ~axis:1 (List.rev hs)), caches)

let feed ?before m call ~slots ~pool chunks =
  fst (feed_state ?before m call ~slots ~pool chunks)

let test_agreement m call =
  let n = Array.length prompt in
  let h, _ =
    call (cache ~slots:n)
      (Cache_index.rows ~context:n [| n |])
      (ids [| prompt |])
  in
  close ~msg:"one call over Cache_index.rows = hidden" (whole m) (logits m h)

let test_chunks m call =
  let n = Array.length prompt in
  let slots = Array.init n Fun.id in
  let feed = feed_state m call ~slots ~pool:n in
  let expected = whole m in
  let ones, by_one = feed (List.init n (fun _ -> 1)) in
  let whole, at_once = feed [ n ] in
  close ~msg:"chunks of 1" expected ones;
  close ~msg:"chunks of 7 then 2" expected (fst (feed [ 7; 2 ]));
  close ~msg:"the whole prompt" expected whole;
  (* The scratch row, the last, is left out: its content is unspecified. *)
  let written written =
    Nx.concatenate ~axis:0
      (Nx.Ptree.fold caches
         (fun _ leaf acc ->
           Nx.cast Nx.float32 (Nx.slice [ R (0, n) ] leaf) :: acc)
         written [])
  in
  close ~msg:"the written slots" (written at_once) (written by_one)

(* The allocator's numbering is free: any injection of positions into slots
   gives the same outputs. *)
let test_slot_renaming m call =
  let n = Array.length prompt and pool = 23 in
  let order = Nx.to_array (Nx.Rng.permutation (Nx.Rng.key 7) pool) in
  let slots = Array.init n (fun j -> Int32.to_int order.(j)) in
  close ~msg:"a random permutation of the slots"
    (feed m call ~slots:(Array.init n Fun.id) ~pool [ 4; 3; 2 ])
    (feed m call ~slots ~pool [ 4; 3; 2 ])

(* Two sequences hold an equal prefix at the same slots, written once. *)
let test_shared_prefix m call =
  let prefix = [| 3; 14; 1; 5 |] and tails = [| [| 9; 2 |]; [| 6; 5 |] |] in
  let _, caches =
    call (cache ~slots:8)
      (Cache_index.make
         ~pos:(int32s [| 1; 4 |] [| 0; 1; 2; 3 |])
         ~table:(int32s [| 1; 6 |] [| 5; 2; 7; 0; -1; -1 |])
         ())
      (ids [| prefix |])
  in
  let h, _ =
    call caches
      (Cache_index.make
         ~pos:(int32s [| 2; 2 |] [| 4; 5; 4; 5 |])
         ~table:(int32s [| 2; 6 |] [| 5; 2; 7; 0; 1; 3; 5; 2; 7; 0; 4; 6 |])
         ())
      (ids tails)
  in
  Array.iteri
    (fun b tail ->
      let alone = hidden m (ids [| Array.append prefix tail |]) in
      close
        ~msg:(Printf.sprintf "sequence %d" b)
        (logits m (Nx.slice [ A; R (4, 6) ] alone))
        (logits m (Nx.slice [ R (b, b + 1) ] h)))
    tails

(* The tokens of one sequence as lanes of one token each, all naming the same
   row of the table: each sees the others through the written pool. *)
let test_lanes_of_one_sequence m call =
  let n = Array.length prompt in
  let h, _ =
    call (cache ~slots:n)
      (Cache_index.make
         ~row:(int32s [| n |] (Array.make n 0))
         ~pos:(int32s [| n; 1 |] (Array.init n Fun.id))
         ~table:(int32s [| 1; n |] (Array.init n (fun j -> n - 1 - j)))
         ())
      (Nx.reshape [| n; 1 |] (ids [| prompt |]))
  in
  close ~msg:"one lane per token" (whole m)
    (logits m (Nx.reshape [| 1; n; dim |] h))

(* Prompts of different lengths in one left-padded batch, then one more token
   each: every row's logits are those it has alone. *)
let test_ragged_batch m call =
  let short = [| 4; 8; 15 |] and long = [| 16; 2; 3; 4; 2; 1 |] in
  let next = [| 7; 11 |] in
  let alone row = logits m (hidden m (ids [| row |])) in
  let index = Cache_index.rows ~context:8 [| 3; 6 |] in
  let h, caches =
    call (cache ~slots:16) index
      (ids [| Array.append (Array.make 3 0) short; long |])
  in
  let h', _ =
    call caches
      (Cache_index.advance index)
      (ids [| [| next.(0) |]; [| next.(1) |] |])
  in
  finite ~msg:"padding produces no nan" h;
  let check b row pad =
    let expected = alone (Array.append row [| next.(b) |]) in
    let n = Array.length row in
    close
      ~msg:(Printf.sprintf "row %d, prompt" b)
      (Nx.slice [ A; R (0, n) ] expected)
      (logits m (Nx.slice [ R (b, b + 1); R (pad, pad + n) ] h));
    close
      ~msg:(Printf.sprintf "row %d, next token" b)
      (Nx.slice [ A; R (n, n + 1) ] expected)
      (logits m (Nx.slice [ R (b, b + 1) ] h'))
  in
  check 0 short 3;
  check 1 long 0

(* Whatever a slot no table names holds, the scratch row included, reaches no
   output. *)
let test_poisoning m call =
  let n = Array.length prompt and pool = 16 in
  (* Three columns past the prompt are unallocated: they read the scratch
     row. *)
  let slots =
    Array.init (n + 3) (fun j -> if j < n then ((j * 5) + 3) mod pool else -1)
  in
  let named =
    Nx.create Nx.bool
      [| pool + 1; 1; 1 |]
      (Array.init (pool + 1) (fun s -> Array.mem s slots))
  in
  let poison =
    List.map
      (Nx.Ptree.Payload.map
         (module Attention.Cache)
         (fun _ leaf -> Nx.where named leaf (Nx.scalar_like leaf Float.nan)))
  in
  let clean = feed m call ~slots ~pool [ 4; 1; 4 ] in
  let poisoned = feed ~before:poison m call ~slots ~pool [ 4; 1; 4 ] in
  finite ~msg:"no nan reaches the outputs" poisoned;
  close ~msg:"the outputs ignore what unnamed slots hold" clean poisoned

(* A lane that addresses nothing yields finite outputs and disturbs nothing. *)
let test_empty_lane m call =
  let n = Array.length prompt in
  let none k = Array.make k (-1) in
  let h, written =
    call (cache ~slots:n)
      (Cache_index.make
         ~pos:(int32s [| 2; n |] (Array.append (none n) (Array.init n Fun.id)))
         ~table:
           (int32s [| 2; n |] (Array.append (none n) (Array.init n Fun.id)))
         ())
      (ids [| Array.make n 0; prompt |])
  in
  finite ~msg:"the empty lane's outputs are finite" h;
  close ~msg:"its neighbour is undisturbed" (whole m)
    (logits m (Nx.slice [ R (1, 2) ] h));
  let h', _ =
    call written
      (Cache_index.make
         ~pos:(int32s [| 1; 1 |] [| -1 |])
         ~table:(int32s [| 1; n |] (none n))
         ())
      (ids [| [| 0 |] |])
  in
  finite ~msg:"a call of padding alone is finite" h'

(* Greedy generation through a compiled step that consumes its state equals
   re-running the whole sequence for every token, and every cache leaf is
   written in its own storage. *)

type state = {
  token : Nx.int32_t;
  scores : Nx.float32_t; (* the logits [token] was taken from *)
  index : Cache_index.t;
  kv : Nx.float32_t Attention.Cache.t list;
}

let state =
  Nx.Ptree.(
    iso
      (fun ((token, scores), (index, kv)) -> { token; scores; index; kv })
      (fun { token; scores; index; kv } -> ((token, scores), (index, kv)))
      (pair (pair tensor tensor) (pair Cache_index.ptree caches)))

(* On CPU:1, a device with storage of its own, the cache stays on the device and
   each step writes it in place. *)
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
  let step { token; index; kv; scores = _ } =
    let seq = Nx.dim 1 token in
    let h, kv = cached m kv index token in
    let scores = logits m (Nx.slice [ A; I (seq - 1) ] h) in
    {
      token = Nx.reshape [| 1; 1 |] (Nx.argmax ~axis:1 scores);
      scores;
      index = Cache_index.advance index;
      kv;
    }
  in
  let step =
    Rune.jit_step ~device:"CPU:1" Nx.Ptree.unit state (fun () s -> step s) ()
  in
  let context = Array.length start + steps in
  let s =
    ref
      (step
         {
           token = ids [| start |];
           scores = Nx.zeros Nx.float32 [| 1; vocab |];
           index = Cache_index.rows ~context [| Array.length start |];
           kv = cache ~slots:context;
         })
  in
  let leaves = 2 * layers * (context + 1) * kv_dim * 4 in
  List.iteri
    (fun i (next, scores) ->
      let msg = Printf.sprintf "step %d" i in
      close ~msg:(msg ^ ", logits") scores !s.scores;
      equal ~msg:(msg ^ ", token") int next
        (Int32.to_int (Nx.item [ 0; 0 ] !s.token));
      if i < steps - 1 then begin
        let before = (Rune.jit_stats ()).reused_bytes in
        s := step !s;
        is_true
          ~msg:(msg ^ ", every cache leaf is written in its own storage")
          ((Rune.jit_stats ()).reused_bytes - before >= leaves)
      end)
    recomputed

(* A fully padded row beside a real one: a query that sees no key has zero
   weights and zero gradients, so nothing poisons the parameters'. *)

let test_gradient_with_a_padded_row grad () =
  let m = model () in
  let n = Array.length prompt in
  let tokens = ids [| Array.make n 0; prompt |] in
  let loss lens m =
    let h = hidden ~lens m tokens in
    let real = Nx.slice [ R (1, 2) ] h in
    Nx.mean (Nx.mul real real)
  in
  let leaves g =
    Nx.Ptree.fold model_ptree
      (fun _ t acc -> flat (Nx.cast Nx.float32 t) :: acc)
      g []
  in
  let padded = leaves (grad (loss [| 0; n |]) m) in
  List.iter
    (fun g ->
      is_true ~msg:"a gradient leaf is finite" (Array.for_all Float.is_finite g))
    padded;
  let alone =
    Rune.grad model_ptree
      (fun m ->
        let h = hidden m (ids [| prompt |]) in
        Nx.mean (Nx.mul h h))
      m
  in
  List.iter2
    (equal ~msg:"the padded row adds nothing" (array (float 1e-5)))
    (leaves alone) padded

let () =
  run "kaun decoder"
    [
      group "one definition"
        (List.concat
           [
             both "one call over Cache_index.rows is hidden" test_agreement;
             both "chunks of 1, of 7 and whole agree" test_chunks;
             both "slots can be renamed" test_slot_renaming;
             both "two sequences share the slots of a prefix" test_shared_prefix;
             both "the tokens of a sequence can be lanes of one call"
               test_lanes_of_one_sequence;
             both "a ragged batch matches each row alone" test_ragged_batch;
             both "unnamed slots and the scratch row are never observed"
               test_poisoning;
             both "a lane that addresses nothing is finite" test_empty_lane;
           ]);
      group "generation"
        [
          test "jitted generation matches recomputation and reuses every leaf"
            test_generation_matches_recomputation;
        ];
      group "gradients"
        [
          test "a fully padded row, eager"
            (test_gradient_with_a_padded_row (fun loss ->
                 Rune.grad model_ptree loss));
          test "a fully padded row, compiled"
            (test_gradient_with_a_padded_row (fun loss ->
                 Rune.jit2 model_ptree model_ptree (Rune.grad model_ptree loss)));
        ];
    ]
