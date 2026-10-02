(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The jitted decode step of a GPT-2 124M shaped decoder: one token in, one
   token out, through key-value caches, with the state consumed and the sampled
   token read back on the host as a generate loop does. The stack is built here
   from kaun layers with zero weights: the step's cost does not depend on their
   values, and kaun ships no model to depend on. Each weight, and the position
   the step reads, is copied to storage of its own, as a loaded model's are: a
   constant is one element seen at every index, which a compiled call folds.

   The cache lengths show how the step scales with the cache it carries, not
   only with the single position it writes.

   On CUDA, gpt-oss-20b at its own shapes, the model of examples/06-gpt-oss
   compiled one layer kind at a time ([Layer_loop]): a decode step and a prefill
   of 512 tokens, over random weights at bfloat16 with the experts packed as
   MXFP4. Building the 13.8 GB of weights takes most of a case's setup. *)

open Kaun

let vocab = 50257
and positions = 1024
and embd = 768
and inner = 3072

let layers = 12
and heads = 12

let head_dim = embd / heads
let eps = 1e-5

type block = {
  ln1 : Nx.float32_t Layer_norm.t;
  attn : Nx.float32_t Attention.t;
  ln2 : Nx.float32_t Layer_norm.t;
  fc : Nx.float32_t Linear.t;
  proj : Nx.float32_t Linear.t;
}

type model = {
  wte : Nx.float32_t Embedding.t;
  wpe : Nx.float32_t Embedding.t;
  blocks : block list;
  ln_f : Nx.float32_t Layer_norm.t;
}

let model () =
  let zeros ~fan_in ~fan_out dtype shape =
    Nx.copy (Init.zeros ~fan_in ~fan_out dtype shape)
  in
  let norm () =
    let { Layer_norm.gamma; beta } = Layer_norm.init ~dim:embd in
    { Layer_norm.gamma = Nx.copy gamma; beta = Nx.copy beta }
  in
  let linear ~inputs ~outputs =
    Linear.make ~w_init:zeros ~bias_init:zeros ~inputs ~outputs Nx.float32
  in
  let block () =
    {
      ln1 = norm ();
      attn =
        Attention.make ~w_init:zeros ~bias_init:zeros ~embed_dim:embd Nx.float32;
      ln2 = norm ();
      fc = linear ~inputs:embd ~outputs:inner;
      proj = linear ~inputs:inner ~outputs:embd;
    }
  in
  {
    wte = Embedding.make ~init:zeros ~vocab ~dim:embd Nx.float32;
    wpe = Embedding.make ~init:zeros ~vocab:positions ~dim:embd Nx.float32;
    blocks = List.init layers (fun _ -> block ());
    ln_f = norm ();
  }

let cached m caches index ids =
  let x =
    Nx.add
      (Embedding.apply m.wte ids)
      (Embedding.apply m.wpe (Cache_index.positions index))
  in
  let x, rev =
    List.fold_left2
      (fun (x, cs) b c ->
        let a, c =
          Attention.cached ~head_dim b.attn c index
            (Layer_norm.apply ~eps b.ln1 x)
        in
        let x = Nx.add x a in
        let mlp =
          Linear.apply b.proj
            (Fn.gelu_approx (Linear.apply b.fc (Layer_norm.apply ~eps b.ln2 x)))
        in
        (Nx.add x mlp, c :: cs))
      (x, []) m.blocks caches
  in
  (x, List.rev rev)

let logits m h =
  Nx.matmul
    (Layer_norm.apply ~eps m.ln_f h)
    (Nx.transpose m.wte.Embedding.table)

let cache ~slots =
  List.init layers (fun _ ->
      Attention.Cache.make ~slots ~kv_heads:heads ~head_dim Nx.float32)

(* A decode loop warmed past its two compilations. The returned thunk decodes
   one token at a fixed position in the middle of the cache, from the previous
   call's token and caches: every measured step is in range, whatever the number
   of samples. *)
let decoder params ~len =
  let caches = Nx.Ptree.list (Nx.Ptree.instantiate (module Attention.Cache)) in
  let step =
    Rune.jit
      Nx.Ptree.(
        tensor @-> Cache_index.ptree @-> consumes caches
        @@ returns (pair tensor caches))
      (fun token index caches ->
        let seq = (Nx.shape token).(1) in
        let h, caches = cached params caches index token in
        let last = Nx.slice [ A; I (seq - 1) ] h in
        (Nx.reshape [| 1; 1 |] (Nx.argmax ~axis:1 (logits params last)), caches))
  in
  let state =
    ref
      (step
         (Nx.zeros Nx.int64 [| 1; 8 |])
         (Cache_index.rows ~context:len [| 8 |])
         (cache ~slots:len))
  in
  let at = Nx.create Nx.int64 [| 1; 1 |] [| Int64.of_int (len / 2) |] in
  let middle =
    Cache_index.make ~pos:at
      ~table:(Nx.reshape [| 1; len |] (Nx.arange Nx.int64 0 len 1))
      ()
  in
  let advance () =
    let token, caches = !state in
    state := step token middle caches;
    ignore (Nx.item [ 0; 0 ] (fst !state) : int64)
  in
  advance ();
  advance

(* The decoder is built inside the measuring worker: a device handle does not
   survive the fork that isolates a case. *)
let case len =
  Thumper.bench_with_setup
    ~setup:(fun () -> decoder (model ()) ~len)
    (Printf.sprintf "decode step, cache %d" len)
    (fun advance -> advance ())

let lens = [ 256; 1024 ]

(* Routed quantised products at gpt-oss-20b's shapes: the gate and up projection
   of one layer's 32 experts, MXFP4 [[| 32; 5760; 2880 |]], applied to each
   token's 4 experts, compiled. One token is a decode step's product; a prompt's
   tokens share experts, 512 of them on a GPU and 64 on the host, where 512 take
   most of a minute a call. On the host, one token's product also runs eagerly.
   Zero codes: the product's cost does not depend on their values. Each operand
   is copied to storage of its own, as a model's are: a constant is one element
   seen at every index, which a compiled call folds. *)

let experts = 32
and per_token = 4
and outputs = 5760
and inputs = 2880

let compiled f = Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) f
let eager f = f

let routed ~run ~tokens device =
  let place t = Nx.place (Nx.Placement.on device) (Nx.copy t) in
  let w =
    Nx_quant.mxfp4
      ~scales:(place (Nx.full Nx.uint8 [| experts; outputs; inputs / 32 |] 127))
      (place (Nx.zeros Nx.uint8 [| experts; outputs; inputs / 2 |]))
  in
  (* Each token's experts, distinct, spread over all of them. *)
  let ids =
    Nx.init Nx.int64 [| tokens; per_token |] (fun i ->
        Int64.of_int (((i.(0) * 7) + (i.(1) * 8)) mod experts))
  in
  let x = Nx.full Nx.float32 [| tokens; 1; 1; inputs |] 0.5 in
  let f = run (fun ids x -> Nx_quant.apply ~ids w x) in
  let ids = place ids and x = place x in
  ignore (f ids x);
  fun () -> ignore (f ids x)

let synchronize device = Nx_device.synchronize (Nx.Device.memory device)

let product ~run ~device ~tokens suffix =
  Thumper.bench_with_setup
    ~setup:(fun () -> routed ~run ~tokens (device ()))
    (Printf.sprintf "routed product, %d tokens%s" tokens suffix)
    (fun call ->
      call ();
      synchronize (device ()))

let quant name ~device ~prompt =
  Thumper.group name
    (List.map
       (fun tokens -> product ~run:compiled ~device ~tokens "")
       [ 1; prompt ])

let host () =
  let device () = Nx.Device.host in
  Thumper.group "host"
    [
      product ~run:compiled ~device ~tokens:1 "";
      product ~run:compiled ~device ~tokens:64 "";
      product ~run:eager ~device ~tokens:1 ", eager";
    ]

let cuda_quant () =
  quant "cuda" ~device:(fun () -> Nx.Device.v (Cuda 0)) ~prompt:512

(* Metal's pipelines are made by [--warm], as the decode kernels are (below). *)
let metal_prompt = 512

let metal () =
  match Metal.device with
  | None -> []
  | Some device -> [ quant "metal" ~device ~prompt:metal_prompt ]

let warm_metal () =
  match Metal.device with
  | None -> ()
  | Some device ->
      List.iter
        (fun tokens ->
          (routed ~run:compiled ~tokens (device ())) ();
          synchronize (device ()))
        [ 1; metal_prompt ]

(* gpt-oss-20b *)

let gpt_oss =
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

let gpt_oss_params placement =
  let c = gpt_oss and seed = ref 12345 in
  let f ~scale shape = Nx.place placement (floats ~scale shape) in
  let linear ?(bias = true) i o =
    {
      Linear.w = f ~scale:0.02 [| i; o |];
      b = (if bias then Some (f ~scale:0.02 [| o |]) else None);
    }
  in
  let packed ~inputs ~outputs =
    let groups = inputs / 32 in
    let scales = random_bytes seed [| c.experts; outputs; groups |] 118 6 in
    let blocks = random_bytes seed [| c.experts; outputs; inputs / 2 |] 0 256 in
    Moe.Quant (Nx_quant.place placement (Nx_quant.mxfp4 ~scales blocks))
  in
  let gamma () = { Rms_norm.gamma = f ~scale:1.0 [| c.dim |] } in
  let q_dim = c.n_heads * c.head_dim and kv = c.n_kv_heads * c.head_dim in
  let block _ =
    let b =
      {
        Gpt_oss.attn_norm = gamma ();
        attn =
          {
            Attention.q = linear c.dim q_dim;
            k = linear c.dim kv;
            v = linear c.dim kv;
            out = linear q_dim c.dim;
          };
        sinks = f ~scale:1.0 [| c.n_heads |];
        ffn_norm = gamma ();
        router = linear c.dim c.experts;
        moe =
          {
            Moe.gate_up = packed ~inputs:c.dim ~outputs:(2 * c.hidden_dim);
            gate_up_bias = f ~scale:0.02 [| c.experts; 2 * c.hidden_dim |];
            down = packed ~inputs:c.hidden_dim ~outputs:c.dim;
            down_bias = f ~scale:0.02 [| c.experts; c.dim |];
          };
      }
    in
    (* The host copies of a block are garbage once it is placed. *)
    Gc.full_major ();
    b
  in
  {
    Gpt_oss.tok = { Embedding.table = f ~scale:0.02 [| c.vocab_size; c.dim |] };
    blocks = List.map block c.layers;
    norm = gamma ();
    head = Some (linear ~bias:false c.dim c.vocab_size);
  }

(* A gpt-oss step on CUDA warmed past its compilations: [`Decode] feeds back the
   token it predicts at a fixed position in the middle of the cache, [`Prefill]
   runs the [prompt] tokens at positions 0 onwards. Each call takes the caches
   of the one before and reads its token back on the host. *)
let gpt_oss_step kind =
  let placement = Nx.Placement.on (Nx.Device.v (Cuda 0)) in
  let step = Layer_loop.greedy ~placement gpt_oss (gpt_oss_params placement) in
  let caches =
    ref
      (Gpt_oss.cache
         ~placement:(fun _ ~axis:_ -> placement)
         gpt_oss ~slots:context Nx.bfloat16)
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

let gpt_oss_kinds = [ `Decode; `Prefill ]

let gpt_oss_case kind =
  let name =
    match kind with
    | `Decode -> Printf.sprintf "decode step, cache %d" context
    | `Prefill -> Printf.sprintf "prefill %d" prompt
  in
  Thumper.bench_with_setup
    ~setup:(fun () -> gpt_oss_step kind)
    name
    (fun call -> call ())

(* CUDA's driver must not be initialized before the fork that isolates a case,
   so a fresh process says whether a device opens. *)
let run_self flag =
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe [| exe; flag |] Unix.stdin Unix.stdout Unix.stderr
  in
  match Unix.waitpid [] pid with _, Unix.WEXITED 0 -> true | _ -> false

(* A forked worker cannot reach the GPU's compiler service either. A child
   process compiles the kernels first; the workers then load them from tolk's
   kernel cache. This process never touches the device. *)
let warm flag =
  if not (run_self flag) then
    failwith "bench_decode: compiling the decode kernels failed"

let () =
  match Array.to_list Sys.argv with
  | [ _; "--warm" ] ->
      List.iter (fun len -> (decoder (model ()) ~len) ()) lens;
      warm_metal ()
  | [ _; "--warm-gpt-oss" ] ->
      List.iter (fun kind -> (gpt_oss_step kind) ()) gpt_oss_kinds
  | [ _; "--cuda" ] ->
      exit (if Result.is_ok (Nx.Device.get (Cuda 0)) then 0 else 1)
  | argv ->
      let measures =
        match argv with
        | _ :: ("list" | "-h" | "--help" | "-V" | "--version") :: _ -> false
        | _ -> true
      in
      let cuda = run_self "--cuda" in
      if measures then begin
        warm "--warm";
        if cuda then warm "--warm-gpt-oss"
      end;
      let budgets =
        [ Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05 ]
      in
      (* A gpt-oss case builds its weights in its setup, and a prompt's routed
         product takes seconds a call on the host. *)
      Thumper.run "kaun_decode"
        ~config:Thumper.Config.(default |> deadline 1200.)
        ~budgets
        (Thumper.group "Gpt2" (List.map case lens)
        :: Thumper.group "Quant"
             ((host () :: metal ()) @ if cuda then [ cuda_quant () ] else [])
        ::
        (if cuda then
           [ Thumper.group "GptOss" (List.map gpt_oss_case gpt_oss_kinds) ]
         else []))
