(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* gpt-oss-20b at its own shapes on a CUDA, NV or AMD GPU, the example's model
   compiled one layer kind at a time ([Layer_loop]): a decode step and
   prefills of 512 and 12 tokens, over random weights at bfloat16 with the
   experts packed as MXFP4, a decode step with the attention projections and
   the head packed as Q8_0 too, quantised in its case's setup, and one layer's
   experts, after its norm, at prefills of 512 and 536 tokens. 12 tokens route 48 positions over
   32 experts in blocks of 4, each block reading its expert's matrices once
   for its rows. 536 tokens fill 65 blocks of 64 rows, a
   count 5 divides, where a warp of 5 tiles of rows would cross two blocks.
   [first call 12] times a later process's first 12-token call over a warm
   disk cache ([first_config]), and the cold rows a first call that compiles
   every kernel.

   The weights are the checkpoint's entries, named, shaped and laid out as
   gpt-oss-20b's safetensors files hold them, read by the example's loader
   ([Gpt_oss.of_hf]) as a user's run reads the file: the steps compile the
   kernels a user's do, over the same layouts, and the Q8_0 cases quantise
   on the GPU, compiled, as the loader does.

   Each step case's setup builds the 13.8 GB of entries on the host, in the
   worker thumper forks for it, and loads them onto the GPU; a GPU's driver
   must not be initialized before the fork. Weights built in the parent would be shared
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
let blocked_prompt = 536

(* Bytes drawn per array before they repeat. With all 10 GB of expert bytes
   drawn, building the weights took 75 s on nonnormal's i9-9900K, in every
   case's setup. *)
let period = 1 lsl 20

(* Seconds of uncounted calls before a trial's samples. nonnormal's R9700
   idles at 41 MHz; under decode steps it holds 1.7 GHz for 0.4 s, ramps to
   3.2 GHz over 0.5 s, then creeps to 3.33 GHz by 3 s. With thumper's 0.1 s
   default, the decode step timed the ramp: 32-34 ms against a steady
   20.1. *)
let warmup_seconds = 3.

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

(* gpt-oss's checkpoint entries under [c] on the host, floats from
   [bf16 ~scale shape] and bytes from [u8 lo span shape], with the names,
   dtypes and shapes of gpt-oss-20b's safetensors files: each projection
   [outputs; inputs], each expert weight's MXFP4 blocks
   [experts; outputs; groups; 16] and its scales. *)
let entries_of c ~bf16 ~u8 =
  let float ?(scale = 0.02) name shape = (name, Nx.P (bf16 ~scale shape)) in
  let bytes lo span name shape = (name, Nx.P (u8 lo span shape)) in
  let linear ?(bias = true) name ~inputs ~outputs =
    float (name ^ ".weight") [| outputs; inputs |]
    :: (if bias then [ float (name ^ ".bias") [| outputs |] ] else [])
  in
  let experts name ~inputs ~outputs =
    let groups = inputs / 32 in
    [
      bytes 0 256 (name ^ "_blocks")
        [| c.Gpt_oss.experts; outputs; groups; 16 |];
      bytes 118 6 (name ^ "_scales") [| c.experts; outputs; groups |];
      float (name ^ "_bias") [| c.experts; outputs |];
    ]
  in
  let norm name = float ~scale:1.0 name [| c.Gpt_oss.dim |] in
  let q_dim = c.n_heads * c.head_dim and kv = c.n_kv_heads * c.head_dim in
  let block i _ =
    let at = Printf.sprintf "model.layers.%d.%s" i in
    List.concat
      [
        [ norm (at "input_layernorm.weight") ];
        linear (at "self_attn.q_proj") ~inputs:c.dim ~outputs:q_dim;
        linear (at "self_attn.k_proj") ~inputs:c.dim ~outputs:kv;
        linear (at "self_attn.v_proj") ~inputs:c.dim ~outputs:kv;
        linear (at "self_attn.o_proj") ~inputs:q_dim ~outputs:c.dim;
        [
          float ~scale:1.0 (at "self_attn.sinks") [| c.n_heads |];
          norm (at "post_attention_layernorm.weight");
        ];
        linear (at "mlp.router") ~inputs:c.dim ~outputs:c.experts;
        experts
          (at "mlp.experts.gate_up_proj")
          ~inputs:c.dim ~outputs:(2 * c.hidden_dim);
        experts (at "mlp.experts.down_proj") ~inputs:c.hidden_dim
          ~outputs:c.dim;
      ]
  in
  Nx_io.Archive.of_list
    (List.concat
       [
         [ float "model.embed_tokens.weight" [| c.vocab_size; c.dim |] ];
         List.concat (List.mapi block c.layers);
         [ norm "model.norm.weight" ];
         linear ~bias:false "lm_head" ~inputs:c.dim ~outputs:c.vocab_size;
       ])

let entries c =
  let seed = ref 12345 in
  entries_of c ~bf16:floats ~u8:(fun lo span shape ->
      random_bytes seed shape lo span)

(* The model of [entries] on [placement], loaded as {!Gpt_oss.of_hf} loads a
   checkpoint, its projections and head packed as [pack] says. *)
let load ?pack placement c entries =
  Gpt_oss.of_hf ?pack ~placement:(fun _ ~axis:_ -> placement) c Nx.bfloat16
    entries

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
   [`Decode_q8_0] likewise with the attention projections and the head
   packed as Q8_0 ([Gpt_oss.Q8_0]), [`Prefill n] runs [n] tokens at
   positions 0 onwards, [`Prefill_q8_0 n] packed as Q8_0. Each call takes
   the caches of the one before and reads its token back on the host. *)
let step entries kind =
  let pack =
    match kind with
    | `Decode_q8_0 | `Prefill_q8_0 _ -> Gpt_oss.Q8_0
    | `Decode | `Prefill _ -> Unpacked
  in
  let module G = (val Nx.devices [ gpu () ]) in
  let placement = G.on in
  let p = load ~pack placement config entries in
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
    | `Prefill n | `Prefill_q8_0 n ->
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
    | `Prefill _ | `Prefill_q8_0 _ -> ()
  in
  (* The host entries are garbage once loaded. Collected later, by thumper
     between its warmup and its samples, freeing them idled the GPU for
     200 ms, which dropped it back to its idle clock. *)
  Gc.full_major ();
  call ();
  call

(* gpt-oss of one layer and a small vocabulary: a block's experts have the
   full model's shapes, and loading the 13.8 GB of the rest to take them
   would leave GPU memory to be freed during a trial. *)
let one_layer = { config with vocab_size = 1024; layers = [ Gpt_oss.Sliding ] }

(* One layer's experts, [Moe.apply] of {!one_layer}'s, compiled at a prefill
   of [prompt] tokens: random bfloat16 activations, routed by the router of
   random logits from a fixed key, as the tensor cores multiply the MXFP4
   weights. The experts read the block's norm of the activations, as in the
   model: the norm's scaling is computed as the rows are gathered by expert,
   into a buffer the first product reads in order and stages through shared
   memory. Read from an input, the gather would fuse into that product, a
   kernel the model never runs. *)
let experts prompt =
  let module G = (val Nx.devices [ gpu () ]) in
  let block = List.hd (load G.on one_layer (entries one_layer)).blocks in
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
      (fun ids w x ->
        let h = Rms_norm.apply ~eps:config.norm_eps block.ffn_norm x in
        Moe.apply ~limit:config.swiglu_limit block.moe (ids, w) h)
  in
  let call () = ignore (Nx.item [ 0; 0 ] (f ids w x)) in
  (* The host entries and the rest of the model are garbage, collected here
     as {!step} collects its own. *)
  Gc.full_major ();
  call ();
  call

(* A later process's first call: a sample has a process that holds a small
   gpt-oss on the GPU make the 12-token call of the full model's programs over
   a disk cache one call filled, and nothing but its own programs' reads is
   timed. Both kinds of layer run one block program, so the two layers here
   cost as the 24 of the model would and its zero weights take the place of
   13.8 GB; the programs read the [dim] of the full model, on a smaller
   vocabulary. *)
let first_config =
  {
    config with
    vocab_size = 1024;
    layers = [ Gpt_oss.Sliding; Gpt_oss.Full ];
  }

let prepare_first () =
  let zeros shape = Nx.zeros Nx.bfloat16 shape in
  let entries =
    entries_of first_config
      ~bf16:(fun ~scale:_ s -> zeros s)
      ~u8:(fun _ _ s -> Nx.zeros Nx.uint8 s)
  in
  let module G = (val Nx.devices [ gpu () ]) in
  let placement = G.on in
  let p = load placement first_config entries in
  let step =
    Layer_loop.greedy ~jit:true ~placement
      (Gpt_oss.place_config placement first_config)
      p
  in
  let caches =
    Gpt_oss.cache
      ~placement:(fun _ ~axis:_ -> placement)
      first_config ~slots:context Nx.bfloat16
  in
  let index = Cache_index.rows ~context [| short_prompt |] in
  let ids =
    Nx.init Nx.int64 [| 1; short_prompt |] (fun i -> Int64.of_int (17 + i.(1)))
  in
  fun () -> ignore (Nx.item [ 0 ] (fst (step caches index ids)))

type ready = { pid : int; go : out_channel; said : in_channel }

(* A process that has prepared its first call, with [env] before its own
   environment. *)
let start_ready env =
  let exe = Sys.executable_name in
  let env = Array.append (Array.of_list env) (Unix.environment ()) in
  let child_in, go = Unix.pipe ~cloexec:true () in
  let said, child_out = Unix.pipe ~cloexec:true () in
  let pid =
    Unix.create_process_env exe [| exe; "--first" |] env child_in child_out
      Unix.stderr
  in
  Unix.close child_in;
  Unix.close child_out;
  {
    pid;
    go = Unix.out_channel_of_descr go;
    said = Unix.in_channel_of_descr said;
  }

let expect r line =
  match In_channel.input_line r.said with
  | Some l when l = line -> ()
  | _ -> failwith ("bench_gpt_oss: a first call's process did not say " ^ line)

let stop r =
  (try Unix.kill r.pid Sys.sigkill with Unix.Unix_error (ESRCH, _, _) -> ());
  ignore (Unix.waitpid [] r.pid);
  close_out_noerr r.go;
  close_in_noerr r.said

let finish r =
  output_string r.go "go\n";
  flush r.go;
  expect r "done"

(* A trial makes a call to size its batches, up to 3 to warm up and 3 for its
   samples, a process each, and one more to try a baseline's batch size.
   Thumper's [--quick] and [--precise] ask for 11 and 40 samples, a process
   each, each holding a model on the GPU: the case fails under them, and says
   why. *)
let ready_processes = 10

let no_process_left =
  Printf.sprintf
    "bench_gpt_oss: a first call row holds a process for each call, %d a \
     trial, enough for its 3 samples: run it without --quick or --precise"
    ready_processes

(* [first_call name ~cold env] times a first call made with [env]. Its
   processes share a disk cache that one call filled, or under [cold] each
   has an empty tolk cache and CUDA driver cache of its own, so a sample
   compiles every kernel. *)
let first_call name ~cold env =
  Thumper.bench_with_setup name
    ~setup:(fun () ->
      let dir = Filename.temp_dir "bench-gpt-oss" "" in
      let start i =
        if not cold then
          start_ready (("CACHEDB=" ^ Filename.concat dir "cache.db") :: env)
        else
          let own = Filename.concat dir (string_of_int i) in
          Unix.mkdir own 0o700;
          start_ready
            (("CACHEDB=" ^ Filename.concat own "cache.db")
            :: ("CUDA_CACHE_PATH=" ^ Filename.concat own "cuda")
            :: env)
      in
      if not cold then begin
        let fill = start 0 in
        expect fill "ready";
        finish fill;
        stop fill
      end;
      let all = List.init ready_processes start in
      List.iter (fun r -> expect r "ready") all;
      (dir, all, Queue.of_seq (List.to_seq all)))
    ~teardown:(fun (dir, all, _) ->
      List.iter stop all;
      ignore (Sys.command ("rm -rf " ^ Filename.quote dir)))
    (fun (_, _, waiting) ->
      match Queue.take_opt waiting with
      | None -> failwith no_process_left
      | Some r -> finish r)

let kinds =
  [
    `Decode;
    `Decode_q8_0;
    `Prefill prompt;
    `Prefill_q8_0 prompt;
    `Prefill short_prompt;
    `Experts prompt;
    `Experts blocked_prompt;
  ]

(* [full] is the full model's entries, forced by the steps alone. *)
let setup full = function
  | (`Decode | `Decode_q8_0 | `Prefill _ | `Prefill_q8_0 _) as kind ->
      step (Lazy.force full) kind
  | `Experts n -> experts n

let case kind =
  let name =
    match kind with
    | `Decode -> Printf.sprintf "decode step, cache %d" context
    | `Decode_q8_0 -> Printf.sprintf "decode step, Q8_0, cache %d" context
    | `Prefill n -> Printf.sprintf "prefill %d" n
    | `Prefill_q8_0 n -> Printf.sprintf "prefill %d, Q8_0" n
    | `Experts n -> Printf.sprintf "experts at prefill %d" n
  in
  Thumper.bench_with_setup
    ~setup:(fun () -> setup (lazy (entries config)) kind)
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
  | [ _; "--first" ] ->
      let call = prepare_first () in
      print_endline "ready";
      ignore (In_channel.input_line stdin);
      call ();
      print_endline "done";
      ignore (In_channel.input_line stdin)
  | [ _; "--warm" ] ->
      if opens () then
        let full = lazy (entries config) in
        List.iter
          (fun kind ->
            (setup full kind) ();
            Gc.full_major ())
          kinds
  | _ ->
      if not (opens_in_child ()) then begin
        prerr_endline "gpt_oss: no CUDA, NV or AMD GPU, nothing to measure";
        exit 0
      end;
      (* A trial's deadline covers its setup, which for a step builds the
         13.8 GB of entries and loads them onto the GPU, its warmup and its
         samples: the seven trials take 100 s together on fudge's RTX 5000
         Ada. *)
      let code =
        Thumper.run "gpt_oss"
          ~config:
            Thumper.Config.(default |> deadline 120. |> warmup warmup_seconds)
          ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
          [ Thumper.group "GptOss" (List.map case kinds) ]
      in
      (* Under PARALLEL=1 one domain renders and one compiler process compiles,
         so the serial row's wall is the compile work, which the parallel row
         does not time where it is off the longest chain of compiles. *)
      let first =
        Thumper.run "gpt_oss_first"
          ~config:
            Thumper.Config.(
              default |> samples 3 |> warmup 0. |> deadline 120.
              |> metrics [ Thumper.Metric.wall_time ])
          ~budgets:[ Thumper.Budget.no_slower_than 0.1 ]
          [
            Thumper.group "GptOss"
              [
                first_call "first call 12" ~cold:false [];
                first_call "first call 12, cold" ~cold:true [];
                first_call "first call 12, cold, serial" ~cold:true
                  [ "PARALLEL=1" ];
              ];
          ]
      in
      exit (max code first)
