(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The layer loop at gpt-oss's depth, on small random weights placed on CPU:1, a
   device with storage of its own: each block program reads its layer's weights
   and the cache index and writes the layer's cache in the cache's own storage.
   The programs report what each call did to its leaves under RUNE_JIT_DEBUG=1,
   which the dune rule sets. *)

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
    rope = Rope.make ~head_dim:8 ();
    attention_scale = 1.0 /. sqrt 8.0;
    tied = false;
  }

let cpu1 = Nx.Placement.device (Rune.device "CPU:1")

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

(* The lines [f] writes to standard error. *)
let stderr_of f =
  let path = Filename.temp_file "test_layer_loop" ".log" in
  let fd = Unix.openfile path [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  flush stderr;
  let saved = Unix.dup Unix.stderr in
  Unix.dup2 fd Unix.stderr;
  Unix.close fd;
  let r =
    Fun.protect
      ~finally:(fun () ->
        flush stderr;
        Unix.dup2 saved Unix.stderr;
        Unix.close saved)
      f
  in
  let lines = In_channel.with_open_text path In_channel.input_lines in
  Sys.remove path;
  (r, lines)

(* What each call in [lines] reported per input leaf, in leaf order. *)
let leaf_reports lines =
  let status line =
    Scanf.sscanf_opt line "rune.jit: input leaf %_d: %[^\n]" Fun.id
  in
  let close calls = function [] -> calls | call -> List.rev call :: calls in
  let calls, last =
    List.fold_left
      (fun (calls, call) line ->
        match status line with
        | Some s -> (calls, s :: call)
        | None -> (close calls call, []))
      ([], []) lines
  in
  List.rev (close calls last)

let test_blocks_read_weights_and_reuse_caches () =
  let p = params () in
  let module Block =
    (val Gpt_oss.block_ptree ()
        : Nx.Ptree.S with type t = Nx.float32_t Gpt_oss.block)
  in
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
  let cached = Layer_loop.cached ~device:"CPU:1" cfg p in
  let index = Cache_index.rows ~context [| n0 |] in
  (* The leaves a block program reads: a block's weights, then the index. *)
  let reads = ref 0 in
  Block.iter (fun _ -> incr reads) (List.hd p.blocks);
  Cache_index.iter (fun _ -> incr reads) index;
  let reads = !reads in
  let ids = Nx.create Nx.int32 [| 1; n0 |] (Array.init n0 Int32.of_int) in
  let x, caches = cached caches index ids in
  let x = ref x and caches = ref caches and index = ref index in
  let pool = Nx.nbytes (List.hd !caches).Attention.Cache.keys in
  for step = 1 to steps do
    let msg = Printf.sprintf "step %d" step in
    let token = Nx.create Nx.int32 [| 1; 1 |] [| Int32.of_int step |] in
    index := Cache_index.advance !index;
    let index_bytes =
      let n = ref 0 in
      Cache_index.iter (fun t -> n := !n + Nx.nbytes t) !index;
      !n
    in
    let s0 = Rune.jit_stats () in
    let (x', caches'), lines =
      stderr_of (fun () -> cached !caches !index token)
    in
    let s1 = Rune.jit_stats () in
    let blocks =
      List.filter (fun r -> List.length r = reads + 3) (leaf_reports lines)
    in
    equal ~msg:(msg ^ ": one report per layer") int 24 (List.length blocks);
    List.iteri
      (fun layer report ->
        let msg = Printf.sprintf "%s, layer %d" msg layer in
        let read = List.filteri (fun i _ -> i < reads) report in
        let cache =
          List.filteri (fun i _ -> i >= reads && i < reads + 2) report
        in
        is_true
          ~msg:(msg ^ ": the weights and the index are read")
          (List.for_all (String.equal "read") read);
        equal
          ~msg:(msg ^ ": the keys and values take their storage")
          (list string)
          [ "storage reused"; "storage reused" ]
          cache)
      blocks;
    is_true
      ~msg:(msg ^ ": every pool's bytes are reused")
      (s1.reused_bytes - s0.reused_bytes >= 24 * 2 * pool);
    (* The first single-token call compiles its programs, which upload their
       host constants once. *)
    if step > 1 then
      equal
        ~msg:(msg ^ ": only the index and the token are uploaded")
        int
        (index_bytes + Nx.nbytes token)
        (s1.bytes_to_device - s0.bytes_to_device);
    x := x';
    caches := caches'
  done;
  is_true ~msg:"the weights stay placed and readable"
    (Nx.Placement.equal cpu1 (Nx.placement (List.hd p.blocks).attn_norm.gamma)
    && Nx.item [ 0 ] (List.hd p.blocks).attn_norm.gamma = 1.0);
  is_true ~msg:"the output is on CPU:1"
    (Nx.Placement.equal cpu1 (Nx.placement !x))

let () =
  run "gpt-oss layer loop"
    [
      group "block programs"
        [
          test "each reads its weights and index and reuses its cache"
            test_blocks_read_weights_and_reuse_caches;
        ];
    ]
