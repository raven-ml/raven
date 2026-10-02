(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* gpt-oss-20b at its own shapes on a CUDA or AMD GPU, the example's model
   compiled one layer kind at a time ([Layer_loop]): a decode step and a prefill
   of 512 tokens, over random weights at bfloat16 with the experts packed as
   MXFP4.

   The 13.8 GB of weights are built once, on the host, before the cases fork:
   every case's worker shares them and places them on the GPU itself, since a
   GPU's driver must not be initialized before the fork. Kernels compile in
   [--warm], a process of its own, into tolk's disk cache; a case's setup reads
   them back. Without a CUDA or AMD GPU there is nothing to measure. *)

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

(* Bytes in [lo, lo + span) from a linear congruential generator: the expert
   blocks and their scales, in the range a trained checkpoint's scales take. *)
let random_bytes seed shape lo span =
  let n = Array.fold_left ( * ) 1 shape in
  let a = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n in
  let s = ref !seed in
  for i = 0 to n - 1 do
    s := ((!s * 1103515245) + 12345) land 0x7fffffff;
    Bigarray.Array1.unsafe_set a i (lo + ((!s lsr 16) mod span))
  done;
  seed := !s;
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
         (Nx.sub_s (Nx.rand Nx.float32 (Array.append [| base_rows |] rest)) 0.5)
         (2.0 *. scale))
  in
  if base_rows = rows then base
  else
    let reps = (rows + base_rows - 1) / base_rows in
    let tiled = Nx.concatenate ~axis:0 (List.init reps (fun _ -> base)) in
    Nx.contiguous (Nx.slice [ R (0, rows) ] tiled)

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
          Attention.q = linear c.dim q_dim;
          k = linear c.dim kv;
          v = linear c.dim kv;
          out = linear q_dim c.dim;
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
    head = Some (linear ~bias:false c.dim c.vocab_size);
  }

let params = Nx.Ptree.instantiate (module Gpt_oss.Params)

(* The GPU: CUDA's first, else AMD's first. *)
let gpu () = match Nx_cuda.get 0 with Ok d -> d | Error _ -> Nx_amd.device 0

(* A step on the GPU warmed past its compilations: [`Decode] feeds back the
   token it predicts at a fixed position in the middle of the cache, [`Prefill]
   runs the [prompt] tokens at positions 0 onwards. Each call takes the caches
   of the one before and reads its token back on the host. *)
let step weights kind =
  let placement = Nx.Placement.on (gpu ()) in
  let p = Nx.Ptree.place params placement weights in
  let step = Layer_loop.greedy ~placement config p in
  let caches =
    ref
      (Gpt_oss.cache
         ~placement:(fun _ ~axis:_ -> placement)
         config ~slots:context Nx.bfloat16)
  in
  let index, ids =
    match kind with
    | `Decode ->
        ( Cache_index.advance (Cache_index.rows ~context [| context / 2 |]),
          ref (Nx.zeros Nx.int64 [| 1; 1 |]) )
    | `Prefill ->
        ( Cache_index.rows ~context [| prompt |],
          ref
            (Nx.init Nx.int64 [| 1; prompt |] (fun i ->
                 Int64.of_int (17 + i.(1)))) )
  in
  let call () =
    let token, c = step !caches index !ids in
    caches := c;
    let token = Nx.item [ 0 ] token in
    match kind with
    | `Decode -> ids := Nx.create Nx.int64 [| 1; 1 |] [| token |]
    | `Prefill -> ()
  in
  call ();
  call

let kinds = [ `Decode; `Prefill ]

let case weights kind =
  let name =
    match kind with
    | `Decode -> Printf.sprintf "decode step, cache %d" context
    | `Prefill -> Printf.sprintf "prefill %d" prompt
  in
  Thumper.bench_with_setup
    ~setup:(fun () -> step (Lazy.force weights) kind)
    name
    (fun call -> call ())

let opens () = Result.is_ok (Nx_cuda.get 0) || Result.is_ok (Nx_amd.get 0)

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
        List.iter (fun kind -> (step weights kind) ()) kinds
  | argv ->
      if not (opens_in_child ()) then begin
        prerr_endline "gpt_oss: no CUDA or AMD GPU, nothing to measure";
        exit 0
      end;
      (* Built before the cases fork, so that they share them; [list] and the
         help need none. *)
      let weights = lazy (weights ()) in
      (match argv with
      | _ :: ("list" | "-h" | "--help" | "-V" | "--version") :: _ -> ()
      | _ -> ignore (Lazy.force weights));
      (* A trial places 13.8 GB on the GPU before its samples. *)
      Thumper.run "gpt_oss"
        ~config:Thumper.Config.(default |> deadline 60.)
        ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
        [ Thumper.group "GptOss" (List.map (case weights) kinds) ]
      |> exit
