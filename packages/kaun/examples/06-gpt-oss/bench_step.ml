(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Time of the compiled decode step, [Layer_loop.greedy], at the widths of
   gpt-oss-20b, on random weights placed on the device: no checkpoint is read.
   [--layers] sets the depth, so a few layers give the cost of one before the
   13.8 GB of the full 24 are built; [--small-vocab] leaves out the 201088-row
   tables.

   Usage: bench_step.exe [--jit DEVICE] [--layers N] [--tokens N] [--steps N]
   [--context N] [--dtype DT] [--small-vocab]. The first call takes [--tokens]
   tokens, every later one a single token. Peak memory is [/usr/bin/time -l]'s
   to read. *)

open Kaun

let cfg n_layers =
  {
    Gpt_oss.vocab_size = 201088;
    dim = 2880;
    layers =
      List.init n_layers (fun i ->
          if i mod 2 = 0 then Gpt_oss.Sliding else Full);
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
        ~beta_slow:1.0 ~original_context:4096;
    attention_scale = (((0.1 *. log 32.0) +. 1.0) ** 2.0) /. 8.0;
    tied = false;
  }

let seed = ref 12345

let random_bytes shape lo span =
  let n = Array.fold_left ( * ) 1 shape in
  let a = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n in
  let s = ref !seed in
  for i = 0 to n - 1 do
    s := ((!s * 1103515245) + 12345) land 0x7fffffff;
    Bigarray.Array1.unsafe_set a i (lo + ((!s lsr 16) mod span))
  done;
  seed := !s;
  Nx.of_bigarray (Bigarray.reshape (Bigarray.genarray_of_array1 a) shape)

let floats dt ~scale shape =
  let rows = shape.(0) in
  let rest = Array.sub shape 1 (Array.length shape - 1) in
  let base_rows = min rows 256 in
  let base =
    Nx.cast dt
      (Nx.mul_s
         (Nx.sub_s (Nx.rand Nx.float32 (Array.append [| base_rows |] rest)) 0.5)
         (2.0 *. scale))
  in
  if base_rows = rows then base
  else
    let reps = (rows + base_rows - 1) / base_rows in
    let tiled = Nx.concatenate ~axis:0 (List.init reps (fun _ -> base)) in
    Nx.contiguous (Nx.slice [ R (0, rows) ] tiled)

let params (type b) ?device c (dt : (float, b) Nx.dtype) ~skip_tables =
  let placement = Option.map Nx.Placement.device device in
  let place x = match placement with None -> x | Some p -> Nx.place p x in
  let f ~scale shape = place (floats dt ~scale shape) in
  let linear ?(bias = true) i o =
    {
      Linear.w = f ~scale:0.02 [| i; o |];
      b = (if bias then Some (f ~scale:0.02 [| o |]) else None);
    }
  in
  let packed ~inputs ~outputs =
    let groups = inputs / 32 in
    let scales = random_bytes [| c.Gpt_oss.experts; outputs; groups |] 118 6 in
    let w =
      Nx_quant.mxfp4 ~scales
        (random_bytes [| c.experts; outputs; inputs / 2 |] 0 256)
    in
    Moe.Quant (match placement with None -> w | Some p -> Nx_quant.place p w)
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
    Gc.full_major ();
    b
  in
  let vocab = if skip_tables then 1024 else c.vocab_size in
  {
    Gpt_oss.tok = { Embedding.table = f ~scale:0.02 [| vocab; c.dim |] };
    blocks = List.map block c.layers;
    norm = gamma ();
    head = Some (linear ~bias:false c.dim vocab);
  }

let run ?device c params dt ~tokens ~steps ~context =
  let placement =
    Option.map (fun d _ ~axis:_ -> Nx.Placement.device d) device
  in
  let step =
    Layer_loop.greedy ?devices:(Option.map (fun d -> [ d ]) device) c params
  in
  let timed (caches, index, ids) =
    let t0 = Unix.gettimeofday () in
    let token, caches = step caches index ids in
    let token = Nx.item [ 0 ] token in
    let t = Unix.gettimeofday () -. t0 in
    let ids = Nx.create Nx.int32 [| 1; 1 |] [| token |] in
    ((caches, Cache_index.advance index, ids), t)
  in
  let report name t =
    let st = Rune.jit_stats () in
    Printf.printf "%s: %.3f s  (to_device %d, from_device %d, resident %d)\n%!"
      name t st.bytes_to_device st.bytes_from_device st.resident_bytes;
    Rune.reset_jit_stats ()
  in
  let state, t =
    timed
      ( Gpt_oss.cache ?placement c ~slots:context dt,
        Cache_index.rows ~context [| tokens |],
        Nx.create Nx.int32 [| 1; tokens |]
          (Array.init tokens (fun i -> Int32.of_int (17 + i))) )
  in
  report (Printf.sprintf "first call, %d tokens" tokens) t;
  let state = ref state in
  if tokens > 1 then begin
    let s, t = timed !state in
    state := s;
    report "first single-token call" t
  end;
  let times =
    Array.init steps (fun _ ->
        let s, t = timed !state in
        state := s;
        t)
  in
  report "steps total" (Array.fold_left ( +. ) 0.0 times);
  Array.sort compare times;
  Printf.printf "step: min %.1f ms, median %.1f ms, max %.1f ms\n%!"
    (1e3 *. times.(0))
    (1e3 *. times.(steps / 2))
    (1e3 *. times.(steps - 1))

let () =
  let jit = ref "METAL" and layers = ref 1 and steps = ref 5 in
  let tokens = ref 1 and context = ref 64 and dtype = ref "bfloat16" in
  let skip_tables = ref false in
  Arg.parse
    [
      ("--jit", Arg.Set_string jit, "Device (default METAL)");
      ("--layers", Arg.Set_int layers, "Blocks (default 1; the model has 24)");
      ("--steps", Arg.Set_int steps, "Timed single-token calls (default 5)");
      ("--tokens", Arg.Set_int tokens, "Tokens of the first call (default 1)");
      ("--context", Arg.Set_int context, "Cache slots (default 64)");
      ("--dtype", Arg.Set_string dtype, "bfloat16 (default) or float32");
      ("--small-vocab", Arg.Set skip_tables, "A vocabulary of 1024");
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "bench_step.exe [--jit DEVICE] [--layers N] [--tokens N] [--steps N] \
     [--context N] [--dtype DT] [--small-vocab]";
  let c = cfg !layers in
  let c = if !skip_tables then { c with Gpt_oss.vocab_size = 1024 } else c in
  let device = if !jit = "" then None else Some (Rune.device !jit) in
  let (Gpt_oss.Dtype dt) = Gpt_oss.dtype_of_string !dtype in
  let t0 = Unix.gettimeofday () in
  let p = params ?device c dt ~skip_tables:!skip_tables in
  Printf.printf "weights built in %.1f s\n%!" (Unix.gettimeofday () -. t0);
  run ?device c p dt ~tokens:!tokens ~steps:!steps ~context:!context
