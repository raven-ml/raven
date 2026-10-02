(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The layer loop at gpt-oss's depth, on small random weights placed on CPU:1, a
   device with storage of its own: each block program reads its layer's weights
   and the cache index and writes the layer's cache in the cache's own storage. *)

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

(* A test device over the host's memory, which the host addresses as it is. *)
let cpu1_device =
  Nx_device.Driver.device ~name:"CPU:1" ~arch:"test" ~budget:max_int
    (Host_visible
       { memory = Nx_device.Driver.host_memory; mapping = Some Identity })

let cpu1 = Nx.Placement.on [ cpu1_device ]

let params () =
  Nx.Rng.with_key (Nx.Rng.key 7) @@ fun () ->
  let f shape = Nx.place cpu1 (Nx.mul_s (Nx.randn Nx.float32 shape) 0.1) in
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

(* Where each pool's storage starts: a call that wrote a layer's cache in its
   own storage returns it at the address it was given. *)
let addresses caches =
  let of_leaf x =
    match Nx.Repr.v x with
    | Placed p ->
        List.map Nx_device.Buffer.address
          (Nx.Repr.Storage.buffers (Nx.Repr.Placed.storage p))
    | Host _ | Traced _ -> invalid_arg "addresses: not a placed value"
  in
  List.concat_map
    (fun (c : _ Attention.Cache.t) -> of_leaf c.keys @ of_leaf c.values)
    caches

let test_blocks_read_weights_and_reuse_caches () =
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
      is_true ~msg:"the builder places each pool"
        (Nx.Placement.equal cpu1 (Nx.placement c.Attention.Cache.keys)))
    caches;
  let cached = Layer_loop.cached ~placement:cpu1 cfg p in
  let index = Cache_index.rows ~context [| n0 |] in
  let ids = Nx.create Nx.int64 [| 1; n0 |] (Array.init n0 Int64.of_int) in
  let x, caches = cached caches index ids in
  let x = ref x and caches = ref caches and index = ref index in
  for step = 1 to steps do
    let msg = Printf.sprintf "step %d" step in
    let token = Nx.create Nx.int64 [| 1; 1 |] [| Int64.of_int step |] in
    index := Cache_index.advance !index;
    let index_bytes =
      Nx.Ptree.fold Cache_index.ptree (fun _ t n -> n + Nx.nbytes t) !index 0
    in
    let before = addresses !caches in
    let s0 = Nx_device.stats cpu1_device in
    let x', caches' = cached !caches !index token in
    let s1 = Nx_device.stats cpu1_device in
    equal
      ~msg:(msg ^ ": every pool is written in its own storage")
      (list nativeint) before (addresses caches');
    (* The first single-token call compiles its programs, which upload their
       host constants once. *)
    if step > 1 then
      equal
        ~msg:(msg ^ ": only the index and the token are uploaded")
        int
        (index_bytes + Nx.nbytes token)
        Nx_device.Stats.(bytes_in (diff s0 s1));
    x := x';
    caches := caches'
  done;
  is_true ~msg:"the weights stay placed and readable"
    (Nx.Placement.equal cpu1 (Nx.placement (List.hd p.blocks).attn_norm.gamma)
    && Nx.item [ 0 ] (List.hd p.blocks).attn_norm.gamma = 1.0);
  is_true ~msg:"the output is on CPU:1"
    (Nx.Placement.equal cpu1 (Nx.placement !x))

let () =
  exit
    (run "gpt-oss layer loop"
       [
         group "block programs"
           [
             test "each reads its weights and index and reuses its cache"
               test_blocks_read_weights_and_reuse_caches;
           ];
       ])
