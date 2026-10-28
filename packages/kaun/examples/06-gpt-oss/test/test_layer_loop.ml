(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The layer loop at gpt-oss's depth, on small random weights placed on CPU:1, a
   device with storage of its own in memory the host addresses: each block
   program compiles and reads its layer's weights and the cache index where they
   lie, writes each pool in its own storage, and a call copies nothing onto
   CPU:1. *)

open Windtrap
open Kaun

let cfg =
  let layers =
    List.init 24 (fun i -> if i mod 2 = 0 then Gpt_oss.Sliding else Full)
  in
  {
    Gpt_oss.vocab_size = 64;
    dim = 32;
    layers;
    window = 4;
    n_heads = 4;
    n_kv_heads = 2;
    head_dim = 8;
    hidden_dim = 32;
    experts = 4;
    experts_per_token = 2;
    swiglu_limit = 7.0;
    norm_eps = 1e-5;
    rope = Rope.make ~head_dim:8 ~context:64 ();
    attention_scale = 1.0 /. sqrt 8.0;
    tied = false;
  }

let cpu1_device = Result.get_ok (Rig.memory_device "CPU:1")

module Cpu1 = (val Nx.devices [ cpu1_device ])

let cpu1 = Cpu1.on
let placed_cfg = Gpt_oss.place_config cpu1 cfg

let params () =
  Nx.Rng.with_key (Nx.Rng.key 7) @@ fun () ->
  let f shape = Nx.place cpu1 (Nx.mul_s (Nx.Rng.normal Nx.float32 shape) 0.1) in
  let linear ?(bias = true) i o =
    { Linear.w = f [| i; o |]; b = (if bias then Some (f [| o |]) else None) }
  in
  let norm () =
    { Rms_norm.gamma = Nx.place cpu1 (Nx.ones Nx.float32 [| 32 |]) }
  in
  let q = cfg.n_heads * cfg.head_dim and kv = cfg.n_kv_heads * cfg.head_dim in
  let e = cfg.experts and d = cfg.dim and h = cfg.hidden_dim in
  let block _ =
    {
      Gpt_oss.attn_norm = norm ();
      attn =
        {
          Attention.q = linear d q;
          k = linear d kv;
          v = linear d kv;
          out = linear q d;
        };
      sinks = f [| cfg.n_heads |];
      ffn_norm = norm ();
      router = linear d e;
      moe =
        {
          Moe.gate_up = Moe.Float (f [| e; d; 2 * h |]);
          gate_up_bias = f [| e; 2 * h |];
          down = Moe.Float (f [| e; h; d |]);
          down_bias = f [| e; d |];
        };
    }
  in
  {
    Gpt_oss.tok = { Embedding.table = f [| cfg.vocab_size; d |] };
    blocks = List.map block cfg.layers;
    norm = norm ();
    head = Some (linear ~bias:false d cfg.vocab_size);
  }

(* Each pool's storage, read without sharing it: a value over its memory
   would keep the next call from taking it. *)
let storage caches =
  let of_leaf x = [ Nx.Repr.read x Nx_array.buffer ] in
  List.concat_map
    (fun (c : _ Attention.Cache.t) -> of_leaf c.keys @ of_leaf c.values)
    caches

(* Buffers compare as the byte of memory they start at: a call that wrote a
   value in its own storage returns it over the memory it was given. *)
let at =
  Testable.make
    ~pp:(fun ppf b ->
      Format.fprintf ppf "%s at %d"
        (Rig.name (Rig.Buffer.device b))
        (Rig.Buffer.offset b))
    ~equal:(fun b b' ->
      Rig.Buffer.overlaps b b' && Rig.Buffer.offset b = Rig.Buffer.offset b')

(* [received m f] is [f ()] and the bytes that copies moved into the memory [m]
   while it ran. *)
let received m f =
  let y, events = Rig.Profile.take f in
  let count n = function
    | Rig.Profile.Copy { dst; bytes; _ } when Rig.equal dst m -> n + bytes
    | _ -> n
  in
  (y, List.fold_left count 0 events)

let placement =
  Testable.make ~pp:Nx.Placement.pp ~equal:(fun a b -> Nx.Placement.equal a b)

type claim = Placed | In_place | No_copy

(* [test_blocks claim] runs the loop and checks [claim]: [Placed] that every
   program compiles and the weights, the pools and the output lie on CPU:1,
   [In_place] that every pool is written in its own storage, [No_copy] that a
   call after the first copies nothing onto CPU:1. *)
let test_blocks claim () =
  let p = params () in
  let n0 = 5 and steps = 3 in
  let context = n0 + steps + 1 in
  let caches =
    Gpt_oss.cache
      ~placement:(fun _ ~axis:_ -> cpu1)
      cfg ~slots:context Nx.float32
  in
  List.iter
    (fun c ->
      equal ~msg:"the builder places each pool" placement cpu1
        (Nx.placement c.Attention.Cache.keys))
    caches;
  let cached = Layer_loop.cached ~placement:cpu1 placed_cfg p in
  let index = Cache_index.rows ~context [| n0 |] in
  let ids = Nx.create Nx.int64 [| 1; n0 |] (Array.init n0 Int64.of_int) in
  let x, caches = cached caches index ids in
  let x = ref x and caches = ref caches and index = ref index in
  for step = 1 to steps do
    let msg = Printf.sprintf "step %d" step in
    let token = Nx.create Nx.int64 [| 1; 1 |] [| Int64.of_int step |] in
    index := Cache_index.advance !index;
    let before = storage !caches in
    let evaluated = Rune.evaluations () in
    let (x', caches'), uploaded =
      received cpu1_device (fun () -> cached !caches !index token)
    in
    (match claim with
    | Placed ->
        (* Every block program compiles: an evaluated one would compute by
           nx's operations. *)
        equal ~msg:(msg ^ ": no program is evaluated") int evaluated
          (Rune.evaluations ())
    | In_place ->
        equal
          ~msg:(msg ^ ": every pool is written in its own storage")
          (list at) before (storage caches')
    | No_copy ->
        (* The first single-token call compiles its programs. After it, the
           index and the token, host values the layer loop places on CPU:1,
           are placed over their own memory, which CPU:1 reaches (Nx.place),
           and the weights and caches are there already. *)
        if step > 1 then
          equal ~msg:(msg ^ ": nothing is copied onto CPU:1") int 0 uploaded);
    x := x';
    caches := caches'
  done;
  if claim = Placed then begin
    let gamma = (List.hd p.blocks).attn_norm.gamma in
    equal ~msg:"the weights stay placed" placement cpu1 (Nx.placement gamma);
    equal ~msg:"the weights stay readable" float_exact 1.0 (Nx.item [ 0 ] gamma);
    equal ~msg:"the output is on CPU:1" placement cpu1 (Nx.placement !x)
  end

let () =
  exit
    (run "gpt-oss layer loop"
       [
         group "block programs"
           [
             test "each compiles and reads its weights and index where they lie"
               (test_blocks Placed);
             test "each writes its cache in the cache's own storage"
               (test_blocks In_place);
             test "a call after the first copies nothing onto CPU:1"
               (test_blocks No_copy);
           ];
       ])
