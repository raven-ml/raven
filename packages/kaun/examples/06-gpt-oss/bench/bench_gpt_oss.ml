(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* gpt-oss-20b at its own shapes on a CUDA, NV or AMD GPU, the example's model
   compiled one layer kind at a time ([Layer_loop]): a decode step and
   prefills of 512 and 12 tokens, over random weights at bfloat16 with the
   experts packed as MXFP4, a decode step with the attention projections and
   the head packed as Q8_0 too, quantised in its case's setup, and one layer's
   experts at the 512-token prefill. 12 tokens route 48 positions over 32
   experts, each a block of its own, whose experts' products stream their
   matrices as a decode step's do.

   Each case's setup builds the 13.8 GB of weights on the host, in the worker
   thumper forks for it, and places them on the GPU; a GPU's driver must not be
   initialized before the fork. Weights built in the parent would be shared
   with the worker copy-on-write, and AMD's driver faults the host pages it
   maps for the GPU as writable, so the worker would copy every page: twice
   the weights in memory, more than a 32 GB host holds. Kernels compile in
   [--warm], a process of its own, into tolk's disk cache; a case's setup reads
   them back. Without a CUDA, NV or AMD GPU there is nothing to measure. *)

open Kaun

let config =
  {
    Gpt_oss.vocab_size = 201088;
    dim = 2880;
    layers =
      List.init 24 (fun i -> if i mod 2 = 0 then Gpt_oss.Sliding else Full);
    window = 128;
    n_heads = 64;
    n_kv_heads = 8;
    head_dim = 64;
    hidden_dim = 2880;
    experts = 32;
    experts_per_token = 4;
    swiglu_limit = 7.0;
    norm_eps = 1e-5;
    rope =
      Rope.yarn ~theta:150000.0 ~head_dim:64 ~factor:32.0 ~beta_fast:32.0
        ~beta_slow:1.0 ~original_context:4096 ~context:131072;
    attention_scale = (((0.1 *. log 32.0) +. 1.0) ** 2.0) /. 8.0;
    tied = false;
  }

let context = 1024
let prompt = 512
let short_prompt = 12

(* Bytes drawn per array before they repeat. With all 10 GB of expert bytes
   drawn, building the weights took 75 s on nonnormal's i9-9900K, in every
   case's setup. *)
let period = 1 lsl 20

(* Bytes in [lo, lo + span) from a linear congruential generator, the first
   [period] repeated through the array: the expert blocks and their scales, in
   the range a trained checkpoint's scales take. *)
let random_bytes seed shape lo span =
  let n = Array.fold_left ( * ) 1 shape in
  let a = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n in
  let drawn = min n period in
  let s = ref !seed in
  for i = 0 to drawn - 1 do
    s := ((!s * 1103515245) + 12345) land 0x7fffffff;
    Bigarray.Array1.unsafe_set a i (lo + ((!s lsr 16) mod span))
  done;
  seed := !s;
  let base = Bigarray.Array1.sub a 0 drawn in
  let i = ref drawn in
  while !i < n do
    let len = min drawn (n - !i) in
    Bigarray.Array1.blit
      (Bigarray.Array1.sub base 0 len)
      (Bigarray.Array1.sub a !i len);
    i := !i + len
  done;
  Nx.of_bigarray (Bigarray.reshape (Bigarray.genarray_of_array1 a) shape)

(* Uniform values in [-scale, scale), 256 rows repeated to the shape's first
   axis. *)
let floats ~scale shape =
  let rows = shape.(0) in
  let rest = Array.sub shape 1 (Array.length shape - 1) in
  let base_rows = min rows 256 in
  let base =
    Nx.cast Nx.bfloat16
      (Nx.mul_s
         (Nx.sub_s
            (Nx.Rng.uniform Nx.float32 (Array.append [| base_rows |] rest))
            0.5)
         (2.0 *. scale))
  in
  if base_rows = rows then base
  else
    let reps = (rows + base_rows - 1) / base_rows in
    let tiled = Nx.concatenate ~axis:0 (List.init reps (fun _ -> base)) in
    Nx.copy (Nx.slice [ R (0, rows) ] tiled)

(* The weights on the host. *)
let weights () =
  let c = config and seed = ref 12345 in
  let linear ?(bias = true) i o =
    {
      Linear.w = floats ~scale:0.02 [| i; o |];
      b = (if bias then Some (floats ~scale:0.02 [| o |]) else None);
    }
  in
  let packed ~inputs ~outputs =
    let groups = inputs / 32 in
    let scales = random_bytes seed [| c.experts; outputs; groups |] 118 6 in
    let blocks = random_bytes seed [| c.experts; outputs; inputs / 2 |] 0 256 in
    Moe.Quant (Nx_quant.mxfp4 ~scales blocks)
  in
  let gamma () = { Rms_norm.gamma = floats ~scale:1.0 [| c.dim |] } in
  let q_dim = c.n_heads * c.head_dim and kv = c.n_kv_heads * c.head_dim in
  let block _ =
    {
      Gpt_oss.attn_norm = gamma ();
      attn =
        {
          Gpt_oss.q = Gpt_oss.Float (linear c.dim q_dim);
          k = Gpt_oss.Float (linear c.dim kv);
          v = Gpt_oss.Float (linear c.dim kv);
          out = Gpt_oss.Float (linear q_dim c.dim);
        };
      sinks = floats ~scale:1.0 [| c.n_heads |];
      ffn_norm = gamma ();
      router = linear c.dim c.experts;
      moe =
        {
          Moe.gate_up = packed ~inputs:c.dim ~outputs:(2 * c.hidden_dim);
          gate_up_bias = floats ~scale:0.02 [| c.experts; 2 * c.hidden_dim |];
          down = packed ~inputs:c.hidden_dim ~outputs:c.dim;
          down_bias = floats ~scale:0.02 [| c.experts; c.dim |];
        };
    }
  in
  {
    Gpt_oss.tok =
      { Embedding.table = floats ~scale:0.02 [| c.vocab_size; c.dim |] };
    blocks = List.map block c.layers;
    norm = gamma ();
    head = Some (Gpt_oss.Float (linear ~bias:false c.dim c.vocab_size));
  }

(* [weights] with the attention projections and the head packed as Q8_0
   ([Gpt_oss.Q8_0]), quantised on the host. *)
let q8_0 (w : _ Gpt_oss.params) =
  let pack = function
    | Gpt_oss.Float { Linear.w; b } ->
        Gpt_oss.Quant { w = Nx_quant.quantize_q8_0 (Nx.matrix_transpose w); b }
    | p -> p
  in
  let block (b : _ Gpt_oss.block) =
    let a = b.attn in
    {
      b with
      attn = { q = pack a.q; k = pack a.k; v = pack a.v; out = pack a.out };
    }
  in
  { w with blocks = List.map block w.blocks; head = Option.map pack w.head }

(* The paths tried, in order: an NVIDIA GPU through CUDA, then through its
   kernel driver (NV), then an AMD GPU through its kernel driver. Hiding the
   CUDA devices ([CUDA_VISIBLE_DEVICES=]) measures the card through NV. *)
let paths =
  [
    (fun () -> Rig.open_ (Rig_cuda.gpu 0));
    (fun () -> Rig.open_ (Rig_nv_nvidia.gpu 0));
    (fun () -> Rig.open_ (Rig_amd_amdgpu.gpu 0));
  ]

let first () =
  let rec go why = function
    | [] -> Error (String.concat "; " (List.rev why))
    | open_ :: rest -> (
        match open_ () with Ok d -> Ok d | Error e -> go (e :: why) rest)
  in
  go [] paths

(* The GPU [0] of the first path that opens one. A set of it computes under
   [Rune.jit] alone. *)
let gpu () = match first () with Ok d -> d | Error e -> failwith e

(* A step on the GPU warmed past its compilations: [`Decode] feeds back the
   token it predicts at a fixed position in the middle of the cache,
   [`Decode_q8_0] likewise with the weights {!q8_0} packs, [`Prefill n] runs
   [n] tokens at positions 0 onwards. Each call takes the caches of
   the one before and reads its token back on the host. *)
let step weights kind =
  let weights =
    match kind with
    | `Decode_q8_0 -> q8_0 weights
    | `Decode | `Prefill _ -> weights
  in
  let module G = (val Nx.devices [ gpu () ]) in
  let placement = G.on in
  let p = Nx.Ptree.place (module Gpt_oss.Params) placement weights in
  let step =
    Layer_loop.greedy ~jit:true ~placement
      (Gpt_oss.place_config placement config)
      p
  in
  let caches =
    ref
      (Gpt_oss.cache
         ~placement:(fun _ ~axis:_ -> placement)
         config ~slots:context Nx.bfloat16)
  in
  let index, ids =
    match kind with
    | `Decode | `Decode_q8_0 ->
        ( Cache_index.advance (Cache_index.rows ~context [| context / 2 |]),
          ref (Nx.zeros Nx.int64 [| 1; 1 |]) )
    | `Prefill n ->
        ( Cache_index.rows ~context [| n |],
          ref
            (Nx.init Nx.int64 [| 1; n |] (fun i ->
                 Int64.of_int (17 + i.(1)))) )
  in
  let call () =
    let token, c = step !caches index !ids in
    caches := c;
    let token = Nx.item [ 0 ] token in
    match kind with
    | `Decode | `Decode_q8_0 -> ids := Nx.create Nx.int64 [| 1; 1 |] [| token |]
    | `Prefill _ -> ()
  in
  call ();
  call

(* One layer's experts, [Moe.apply] of the first block's, compiled at a
   prefill of [prompt] tokens: random bfloat16 activations, routed by the
   router of random logits from a fixed key, as the tensor cores multiply
   the MXFP4 weights. *)
let experts (weights : _ Gpt_oss.params) =
  let module G = (val Nx.devices [ gpu () ]) in
  let moe = Nx.Ptree.place (module Moe) G.on (List.hd weights.blocks).moe in
  let logits =
    Nx.Rng.normal ~key:(Nx.Rng.key 7) Nx.float32 [| prompt; config.experts |]
  in
  let ids, w = Moe.route ~k:config.experts_per_token logits in
  let x =
    Nx.cast Nx.bfloat16
      (Nx.Rng.normal ~key:(Nx.Rng.key 8) Nx.float32 [| prompt; config.dim |])
  in
  let ids = Nx.place G.on ids and w = Nx.place G.on (Nx.cast Nx.bfloat16 w) in
  let x = Nx.place G.on x in
  let f =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
      (fun ids w x -> Moe.apply ~limit:config.swiglu_limit moe (ids, w) x)
  in
  let call () = ignore (Nx.item [ 0; 0 ] (f ids w x)) in
  call ();
  call

let kinds =
  [ `Decode; `Decode_q8_0; `Prefill prompt; `Prefill short_prompt; `Experts ]

let setup weights = function
  | (`Decode | `Decode_q8_0 | `Prefill _) as kind -> step weights kind
  | `Experts -> experts weights

let case kind =
  let name =
    match kind with
    | `Decode -> Printf.sprintf "decode step, cache %d" context
    | `Decode_q8_0 -> Printf.sprintf "decode step, Q8_0, cache %d" context
    | `Prefill n -> Printf.sprintf "prefill %d" n
    | `Experts -> Printf.sprintf "experts at prefill %d" prompt
  in
  Thumper.bench_with_setup
    ~setup:(fun () -> setup (weights ()) kind)
    name
    (fun call -> call ())

let opens () = Result.is_ok (first ())

(* Whether a GPU opens, asked of a fresh process: the driver must not be
   initialized before the fork that isolates a case. *)
let opens_in_child () =
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe [| exe; "--gpu" |] Unix.stdin Unix.stdout
      Unix.stderr
  in
  match Unix.waitpid [] pid with _, Unix.WEXITED 0 -> true | _ -> false

let () =
  match Array.to_list Sys.argv with
  | [ _; "--gpu" ] -> exit (if opens () then 0 else 1)
  | [ _; "--warm" ] ->
      if opens () then
        let weights = weights () in
        List.iter (fun kind -> (setup weights kind) ()) kinds
  | _ ->
      if not (opens_in_child ()) then begin
        prerr_endline "gpt_oss: no CUDA, NV or AMD GPU, nothing to measure";
        exit 0
      end;
      (* A trial's deadline covers its setup, which builds the 13.8 GB of
         weights and places them on the GPU, and its 28 calls: a prefill trial
         takes 52 s on nonnormal's R9700. *)
      Thumper.run "gpt_oss"
        ~config:Thumper.Config.(default |> deadline 120.)
        ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
        [ Thumper.group "GptOss" (List.map case kinds) ]
      |> exit
