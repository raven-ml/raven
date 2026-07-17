(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPT-2 124M training on a single fixed batch, matching the tinygrad reference
   protocol (see /_gpt2_train_reference in the repository):

   - fp32 weights from a local HuggingFace-layout safetensors file, no dropout -
   a fixed 4x65 token grid from tokens.json; inputs are columns 0..63 and
   targets columns 1..64, the same batch every step - mean cross-entropy over
   all 4*64 positions - plain SGD, lr 1e-4, no momentum; the LM head is tied to
   [wte] so the embedding's gradient accumulates from both of its uses - the
   whole step (forward, backward, update) compiles as one [Rune.jit2] program;
   the loss recorded at step i is computed before update i - [--devices]
   switches the step to data-parallel [Rune.pmap2]: parameters replicated on
   every device, the batch sharded on axis 0, gradients allreduced by
   construction — same numbers as the single-device step up to fp32 reduction
   order - [--dropout RATE] (default 0, the reference protocol's dropout-free
   graph) enables the GPT-2 dropout sites in [Gpt2.logits]; the per-step mask
   key is one more int32 leaf of the jitted step's inputs — keys must be
   inputs, never captures — derived as [Nx.Rng.fold_in root step] from
   [--seed], so a run is reproducible from its seed alone

   Per step it emits the loss (shortest round-trip float64 repr of the fp32
   value), wall-clock ms, and fingerprints of six designated weights in the
   reference's in-memory layout: kaun stores linear weights [inputs; outputs]
   like the safetensors file, while the reference fingerprints tinygrad's
   [outputs; inputs] Linear layout, so the three Conv1D-family weights are
   transposed (and the split q/k/v projections re-fused into c_attn) before
   fingerprinting. [--save-weights] writes the post-training weights as a
   drop-in replacement for the input file (same keys and layout, fused c_attn,
   untouched h.N.attn.bias mask buffers copied from the input). *)

open Kaun

let gpt2_124m : Gpt2.config =
  {
    vocab_size = 50257;
    n_positions = 1024;
    n_embd = 768;
    n_layer = 12;
    n_head = 12;
    n_inner = 3072;
    layer_norm_eps = 1e-5;
  }

let batch_size = 4
let seq_len = 64
let lr = 1e-4

(* Shortest round-trip decimal representation, as Python's repr. *)
let repr_float x =
  let rec go p =
    let s = Printf.sprintf "%.*g" p x in
    if p >= 17 || float_of_string s = x then s else go (p + 1)
  in
  go 1

(* Token grid *)

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let load_ids path =
  let json =
    match Jsont_bytesrw.decode_string Jsont.json (read_file path) with
    | Ok v -> v
    | Error e -> failwith (path ^ ": " ^ e)
  in
  let ids =
    match json with
    | Jsont.Object (mems, _) -> (
        match Jsont.Json.find_mem "ids" mems with
        | Some (_, ids) -> ids
        | None -> failwith (path ^ ": no \"ids\" member"))
    | _ -> failwith (path ^ ": expected an object")
  in
  let row = function
    | Jsont.Array (cells, _) ->
        Array.of_list
          (List.map
             (function
               | Jsont.Number (f, _) -> Int32.of_float f
               | _ -> failwith (path ^ ": non-numeric id"))
             cells)
    | _ -> failwith (path ^ ": expected rows of ids")
  in
  match ids with
  | Jsont.Array (rows, _) -> Array.of_list (List.map row rows)
  | _ -> failwith (path ^ ": expected an array of rows")

(* [inputs] is columns 0..seq_len-1 of the grid, [targets] columns 1..seq_len,
   flattened to [batch * seq_len] for the loss. *)
let batch_of_ids ids =
  if
    Array.length ids <> batch_size
    || Array.exists (fun r -> Array.length r <> seq_len + 1) ids
  then
    failwith
      (Printf.sprintf "token grid must be %d x %d" batch_size (seq_len + 1));
  let take off =
    Array.init (batch_size * seq_len) (fun i ->
        ids.(i / seq_len).(off + (i mod seq_len)))
  in
  ( Nx.create Nx.int32 [| batch_size; seq_len |] (take 0),
    Nx.create Nx.int32 [| batch_size * seq_len |] (take 1) )

(* Loss: mean cross-entropy over all positions — log-softmax over the vocab
   axis, NLL of the target id, mean. [?dropout] threads the rate and the
   step's mask key to [Gpt2.logits]; absent, the graph is exactly the
   reference's. *)
let loss_fn inputs targets ?dropout params =
  let logits = Gpt2.logits gpt2_124m ?dropout params inputs in
  let logits =
    Nx.reshape [| batch_size * seq_len; gpt2_124m.vocab_size |] logits
  in
  Loss.softmax_cross_entropy_sparse logits targets

(* Mixed-precision loss ([--compute-dtype bfloat16] or [float16]): the astype
   sandwich. [Gpt2.astype] casts the float32 master weights to the compute dtype
   at the top of the objective and the logits come back up to float32 for the
   loss, so the objective's reductions run at full precision. The cast VJP
   returns cotangents at the pre-cast dtype, so the gradients — and the
   parameters, the optimizer step and [Step_out] — stay float32 whatever the
   compute dtype. The float32 path uses the cast-free [loss_fn] above and its
   exact original graph.

   The weight casts recompute every step inside the jitted graph: on CUDA they
   neither block fusion nor tensor-core matching (the bfloat16 step's matmul
   kernels lower to mma.sync bf16 instructions with float32 accumulators —
   DEBUG=4 shows them), so the bf16 weights are never materialized as a second
   tree. *)
let loss_fn_half compute inputs targets ?dropout params =
  let params = Gpt2.astype compute params in
  let logits = Gpt2.logits gpt2_124m ?dropout params inputs in
  let logits =
    Nx.reshape [| batch_size * seq_len; gpt2_124m.vocab_size |] logits
  in
  Loss.softmax_cross_entropy_sparse (Nx.cast Nx.float32 logits) targets

(* The jitted step returns the updated parameters and the pre-update loss. *)
module Step_out = struct
  type t = { params : Gpt2.t; loss : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    { params = Gpt2.Params.map f t.params; loss = f t.loss }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { params = Gpt2.Params.map2 f a.params b.params; loss = f a.loss b.loss }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    Gpt2.Params.iter f t.params;
    f t.loss
end

let train_step objective params =
  let loss, grads = Rune.value_and_grad (module Gpt2.Params) objective params in
  (* Plain SGD: momentum is 0, so the zero velocity from [sgd_init] leaves the
     update exactly [w - lr * g] and needs no threading across steps. *)
  let state = Vega.sgd_init (module Gpt2.Params) params in
  let params =
    fst (Vega.sgd_step (module Gpt2.Params) ~lr state ~params ~grads)
  in
  { Step_out.params; loss }

(* The step's dropout key rides the input structures as one more optional
   int32 leaf: [Some key] when [--dropout] is positive, [None] otherwise —
   [None] contributes no leaf, so dropout-free runs trace the exact reference
   graph. *)
let map2_key f a b =
  match (a, b) with
  | Some a, Some b -> Some (f a b)
  | None, None -> None
  | _ -> invalid_arg "dropout key: presence mismatch"

(* Single-device step input: the parameters plus the optional dropout key; the
   fixed batch stays captured by the objective. *)
module Keyed_in = struct
  type t = { params : Gpt2.t; key : Nx.Rng.key option }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    { params = Gpt2.Params.map f t.params; key = Option.map f t.key }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    {
      params = Gpt2.Params.map2 f a.params b.params;
      key = map2_key f a.key b.key;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    Gpt2.Params.iter f t.params;
    Option.iter f t.key
end

(* Float16 compute needs loss scaling: float16 gradients underflow below 2^-24.
   The scale state rides the jitted step's input and output structures as tensor
   leaves, so the dynamic scale really updates across compiled calls. Overflowed
   steps keep the previous parameters (selected with [Nx.where] on the finite
   flag, so the step still traces once). *)

module Scaled_in = struct
  type t = {
    params : Gpt2.t;
    ls : Vega.Loss_scale.t;
    key : Nx.Rng.key option;
  }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    {
      params = Gpt2.Params.map f t.params;
      ls = Vega.Loss_scale.map f t.ls;
      key = Option.map f t.key;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    {
      params = Gpt2.Params.map2 f a.params b.params;
      ls = Vega.Loss_scale.map2 f a.ls b.ls;
      key = map2_key f a.key b.key;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    Gpt2.Params.iter f t.params;
    Vega.Loss_scale.iter f t.ls;
    Option.iter f t.key
end

module Scaled_out = struct
  type t = { params : Gpt2.t; loss : Nx.float32_t; ls : Vega.Loss_scale.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    {
      params = Gpt2.Params.map f t.params;
      loss = f t.loss;
      ls = Vega.Loss_scale.map f t.ls;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    {
      params = Gpt2.Params.map2 f a.params b.params;
      loss = f a.loss b.loss;
      ls = Vega.Loss_scale.map2 f a.ls b.ls;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    Gpt2.Params.iter f t.params;
    f t.loss;
    Vega.Loss_scale.iter f t.ls
end

let train_step_scaled objective { Scaled_in.params; ls; key } =
  (* The scale enters as the backward seed: [vjp] against the scale cotangent is
     exactly [grad (fun p -> Loss_scale.scale ls (objective p))] — every float16
     cotangent downstream carries the scale, which is the underflow protection —
     but the loss comes back unscaled for reporting. (It also sidesteps a tolk
     CUDA codegen bug, 2026-07: with the loss-multiply form, one backward kernel
     of this model renders a whole-vocab-axis vectorized store,
     [make_float50257], which NVRTC rejects.) *)
  let loss, grads =
    Rune.vjp (module Gpt2.Params) (objective key) params
      ls.Vega.Loss_scale.scale
  in
  let grads = Vega.Loss_scale.unscale (module Gpt2.Params) ls grads in
  let finite = Vega.Loss_scale.grads_finite (module Gpt2.Params) grads in
  let state = Vega.sgd_init (module Gpt2.Params) params in
  let params' =
    fst (Vega.sgd_step (module Gpt2.Params) ~lr state ~params ~grads)
  in
  let params =
    Gpt2.Params.map2 (fun p p' -> Nx.where finite p' p) params params'
  in
  { Scaled_out.params; loss; ls = Vega.Loss_scale.adjust ls ~finite }

(* Data-parallel input: the batch joins the parameters as leaves so [pmap2] can
   shard it (axis 0) while replicating the parameters. The dropout key leaf is
   replicated too, so every device draws the IDENTICAL mask and the shards are
   decorrelated only by their data; per-device mask decorrelation awaits pmap's
   axis-index extension (a per-device lane id to fold into the key). (As of
   2026-07, [--dropout] with [--devices] does not compile: tolk's CPU renderer
   miscompiles one kernel of the pmapped dropout backward, storing a 3-wide
   vector through a scalar float pointer.) *)
module Step_in = struct
  type t = {
    params : Gpt2.t;
    inputs : (int32, Nx.int32_elt) Nx.t;
    targets : (int32, Nx.int32_elt) Nx.t;
    key : Nx.Rng.key option;
  }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    {
      params = Gpt2.Params.map f t.params;
      inputs = f t.inputs;
      targets = f t.targets;
      key = Option.map f t.key;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    {
      params = Gpt2.Params.map2 f a.params b.params;
      inputs = f a.inputs b.inputs;
      targets = f a.targets b.targets;
      key = map2_key f a.key b.key;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    Gpt2.Params.iter f t.params;
    f t.inputs;
    f t.targets;
    Option.iter f t.key
end

(* [--devices] accepts a CPU device count ([--devices 2] means CPU:1,CPU:2) or
   an explicit comma-separated tuple ([--devices CUDA:0,CUDA:1]). *)
let parse_devices s =
  match int_of_string_opt s with
  | Some n when n > 0 -> List.init n (fun i -> Printf.sprintf "CPU:%d" (i + 1))
  | Some _ -> failwith "--devices: the device count must be positive"
  | None -> List.map String.trim (String.split_on_char ',' s)

(* Fingerprints: [first8] is bitwise (fp32 printed as float64 repr); [sum] and
   [abs_sum] accumulate in float64 with pairwise summation, as the reference. *)

let rec pairwise ~abs a lo n =
  if n <= 128 then begin
    let s = ref 0.0 in
    for i = lo to lo + n - 1 do
      let x = Bigarray.Array1.unsafe_get a i in
      s := !s +. if abs then Float.abs x else x
    done;
    !s
  end
  else
    let h = n / 2 in
    pairwise ~abs a lo h +. pairwise ~abs a (lo + h) (n - h)

let fingerprint t =
  let n = Nx.numel t in
  let a =
    Bigarray.array1_of_genarray
      (Bigarray.reshape (Nx.to_bigarray (Nx.contiguous t)) [| n |])
  in
  let first8 =
    List.init (min 8 n) (fun i -> repr_float (Bigarray.Array1.get a i))
  in
  ( repr_float (pairwise ~abs:false a 0 n),
    repr_float (pairwise ~abs:true a 0 n),
    first8 )

(* The six designated tensors, in the reference's in-memory layout (tinygrad
   Linear [out; in]): c_attn is the q/k/v weights re-fused into the HF [768;
   2304] matrix, then — like c_fc and c_proj — transposed from kaun's [inputs;
   outputs] to [outputs; inputs]. *)
let fingerprint_tensors (p : Gpt2.t) =
  let b0 = List.nth p.blocks 0 and b11 = List.nth p.blocks 11 in
  let c_attn =
    Nx.concatenate ~axis:1 [ b0.attn.q.w; b0.attn.k.w; b0.attn.v.w ]
  in
  [
    ("wte.weight", p.wte.table);
    ("h.0.attn.c_attn.weight", Nx.transpose c_attn);
    ("h.0.mlp.c_fc.weight", Nx.transpose b0.fc.w);
    ("h.11.attn.c_proj.weight", Nx.transpose b11.attn.out.w);
    ("ln_f.weight", p.ln_f.gamma);
    ("wpe.weight", p.wpe.table);
  ]

(* Saving: the input file's key set and layout — fused c_attn, Conv1D weights
   [in; out] (kaun's native layout), mask buffers copied from the input. *)

let bias name = function
  | Some b -> b
  | None -> failwith (name ^ ": expected a bias")

let checkpoint_of_params original (p : Gpt2.t) =
  let tensor = Checkpoint.of_tensor in
  let block i (b : Nx.float32_elt Gpt2.block) =
    let key leaf = Printf.sprintf "h.%d.%s" i leaf in
    let linear leaf (l : Linear.t) =
      [
        tensor (key (leaf ^ ".weight")) l.w;
        tensor (key (leaf ^ ".bias")) (bias (key leaf) l.b);
      ]
    in
    [
      tensor (key "ln_1.weight") b.ln1.gamma;
      tensor (key "ln_1.bias") b.ln1.beta;
      tensor (key "attn.c_attn.weight")
        (Nx.concatenate ~axis:1 [ b.attn.q.w; b.attn.k.w; b.attn.v.w ]);
      tensor (key "attn.c_attn.bias")
        (Nx.concatenate ~axis:0
           [
             bias (key "attn.q") b.attn.q.b;
             bias (key "attn.k") b.attn.k.b;
             bias (key "attn.v") b.attn.v.b;
           ]);
      (match Checkpoint.find (key "attn.bias") original with
      | Some (Rune.Ptree.P mask) -> tensor (key "attn.bias") mask
      | None -> failwith (key "attn.bias" ^ ": missing from the input file"));
      tensor (key "ln_2.weight") b.ln2.gamma;
      tensor (key "ln_2.bias") b.ln2.beta;
    ]
    @ linear "attn.c_proj" b.attn.out
    @ linear "mlp.c_fc" b.fc @ linear "mlp.c_proj" b.proj
  in
  Checkpoint.concat
    ([
       tensor "wte.weight" p.wte.table;
       tensor "wpe.weight" p.wpe.table;
       tensor "ln_f.weight" p.ln_f.gamma;
       tensor "ln_f.bias" p.ln_f.beta;
     ]
    @ List.concat (List.mapi block p.blocks))

(* Metrics JSON, in the reference schema. *)

let emit_metrics buf ~device ~steps ~n_params ~initial_loss records =
  let str s = "\"" ^ s ^ "\"" in
  Buffer.add_string buf
    (Printf.sprintf
       "{\n\
       \ \"device\": %s,\n\
       \ \"steps\": %d,\n\
       \ \"batch_size\": %d,\n\
       \ \"seq_len\": %d,\n\
       \ \"lr\": %s,\n\
       \ \"n_params\": %d,\n\
       \ \"initial_loss\": %s,\n\
       \ \"jit_calls\": null,\n\
       \ \"records\": [\n"
       (str device) steps batch_size seq_len (repr_float lr) n_params
       (str initial_loss));
  List.iteri
    (fun i (step, loss, ms, weights) ->
      if i > 0 then Buffer.add_string buf ",\n";
      Buffer.add_string buf
        (Printf.sprintf
           "  {\"step\": %d, \"loss\": %s, \"ms\": %s,\n   \"weights\": {" step
           (str loss) (repr_float ms));
      List.iteri
        (fun j (name, (sum, abs_sum, first8)) ->
          if j > 0 then Buffer.add_string buf ",";
          Buffer.add_string buf
            (Printf.sprintf
               "\n    %s: {\"sum\": %s, \"abs_sum\": %s, \"first8\": [%s]}"
               (str name) (str sum) (str abs_sum)
               (String.concat ", " (List.map str first8))))
        weights;
      Buffer.add_string buf "}}")
    records;
  Buffer.add_string buf "\n ]\n}\n"

let () =
  let device = ref "CPU" in
  let devices = ref "" in
  let compute_dtype = ref "float32" in
  let steps = ref 20 in
  let dropout = ref 0.0 in
  let seed = ref 0 in
  let here = Filename.dirname Sys.executable_name in
  let tokens = ref (Filename.concat here "tokens.json") in
  let model =
    ref
      (List.fold_left Filename.concat
         (try Sys.getenv "HOME" with Not_found -> ".")
         [ ".cache"; "tolk-gpt2"; "model.safetensors" ])
  in
  let metrics_out = ref "" in
  let save_weights = ref "" in
  Arg.parse
    [
      ("--device", Arg.Set_string device, "Device to jit for (CPU or CUDA)");
      ( "--devices",
        Arg.Set_string devices,
        "Data-parallel device tuple: a CPU count (2 = CPU:1,CPU:2) or a \
         comma-separated list (CUDA:0,CUDA:1)" );
      ( "--compute-dtype",
        Arg.Set_string compute_dtype,
        "Forward/backward dtype: float32 (default), bfloat16 or float16. \
         Parameters, gradients and the optimizer step stay float32 (master \
         weights); float16 adds dynamic loss scaling" );
      ("--steps", Arg.Set_int steps, "Number of training steps");
      ( "--dropout",
        Arg.Set_float dropout,
        "Dropout rate at the GPT-2 sites (default 0: the reference protocol's \
         dropout-free graph)" );
      ( "--seed",
        Arg.Set_int seed,
        "Root PRNG seed for the per-step dropout keys (default 0)" );
      ("--tokens", Arg.Set_string tokens, "Path to the tokens.json grid");
      ("--model", Arg.Set_string model, "Path to the initial safetensors");
      ("--metrics-out", Arg.Set_string metrics_out, "Metrics JSON path");
      ( "--save-weights",
        Arg.Set_string save_weights,
        "Save final weights (input-file layout) to this path" );
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "train --device DEV [--devices TUPLE] [--steps N] [--dropout R] [--seed N] \
     [--tokens F] [--model F] [--metrics-out F] [--save-weights F]";

  let inputs, targets = batch_of_ids (load_ids !tokens) in
  let original = Checkpoint.load !model in
  let params = ref (Gpt2.of_checkpoint gpt2_124m original) in
  let n_params =
    let n = ref 0 in
    Gpt2.Params.iter (fun _ -> incr n) !params;
    !n
  in

  (* Untrained-loss probe, eagerly on the C backend: independent of the jit, the
     optimizer and the backward pass. *)
  let initial_loss = repr_float (Nx.item [] (loss_fn inputs targets !params)) in
  Printf.printf "initial loss (before step 1): %s\n%!" initial_loss;

  (* The per-shard objective, from the compute dtype. *)
  let objective =
    match !compute_dtype with
    | "float32" -> loss_fn
    | "bfloat16" -> loss_fn_half Nx.bfloat16
    | "float16" -> loss_fn_half Nx.float16
    | d ->
        failwith
          ("--compute-dtype must be float32, bfloat16 or float16, got " ^ d)
  in
  (* [obj key inputs targets] is the shard objective for one step's dropout
     key; [None] (the default rate 0) is the dropout-free reference graph. *)
  let rate = !dropout in
  if rate < 0.0 || rate >= 1.0 then failwith "--dropout must be in [0, 1)";
  let obj key inputs targets params =
    objective inputs targets
      ?dropout:(Option.map (fun k -> (rate, k)) key)
      params
  in
  (* The per-step mask key, [fold_in]-derived from the root seed so the run is
     reproducible; it enters the jitted step as an input leaf, never a
     capture. *)
  let root = Nx.Rng.key !seed in
  let key_at i = if rate = 0.0 then None else Some (Nx.Rng.fold_in root i) in
  (* [step i] maps parameters to updated parameters and the pre-update loss;
     the float16 variant additionally threads its loss-scale state, hidden in
     the closure. *)
  let step : int -> Gpt2.t -> Gpt2.t * float =
    if !devices = "" then
      if !compute_dtype = "float16" then begin
        let f =
          Rune.jit2 ~device:!device
            (module Scaled_in)
            (module Scaled_out)
            (train_step_scaled (fun key -> obj key inputs targets))
        in
        let ls = ref (Vega.Loss_scale.dynamic ()) in
        fun i params ->
          let out = f { Scaled_in.params; ls = !ls; key = key_at i } in
          ls := out.Scaled_out.ls;
          (out.Scaled_out.params, Nx.item [] out.Scaled_out.loss)
      end
      else
        let f =
          Rune.jit2 ~device:!device
            (module Keyed_in)
            (module Step_out)
            (fun { Keyed_in.params; key } ->
              train_step (obj key inputs targets) params)
        in
        fun i params ->
          let out = f { Keyed_in.params; key = key_at i } in
          (out.Step_out.params, Nx.item [] out.Step_out.loss)
    else begin
      if !compute_dtype = "float16" then
        failwith "--compute-dtype float16 does not support --devices";
      let devs = parse_devices !devices in
      device := String.concat "," devs;
      (* One [in_axes] entry per leaf in traversal order: every parameter
         replicated, then the two batch leaves sharded on axis 0, then — when
         dropout is on — the key leaf replicated (identical masks on every
         device; see [Step_in]). *)
      let in_axes =
        List.init n_params (fun _ -> None)
        @ [ Some 0; Some 0 ]
        @ (if rate = 0.0 then [] else [ None ])
      in
      let f =
        Rune.pmap2 ~devices:devs ~in_axes
          (module Step_in)
          (module Step_out)
          (fun { Step_in.params; inputs; targets; key } ->
            train_step (obj key inputs targets) params)
      in
      fun i params ->
        let out = f { Step_in.params; inputs; targets; key = key_at i } in
        (out.Step_out.params, Nx.item [] out.Step_out.loss)
    end
  in
  let records = ref [] in
  for i = 1 to !steps do
    let t0 = Unix.gettimeofday () in
    let params', loss = step i !params in
    let ms = (Unix.gettimeofday () -. t0) *. 1e3 in
    params := params';
    Printf.printf "step %3d  loss %s  %9.2f ms\n%!" i (repr_float loss) ms;
    let weights =
      List.map (fun (n, t) -> (n, fingerprint t)) (fingerprint_tensors !params)
    in
    records := (i, repr_float loss, ms, weights) :: !records
  done;

  if !metrics_out <> "" then begin
    let buf = Buffer.create (1 lsl 16) in
    emit_metrics buf ~device:!device ~steps:!steps ~n_params ~initial_loss
      (List.rev !records);
    let oc = open_out !metrics_out in
    Fun.protect
      ~finally:(fun () -> close_out oc)
      (fun () -> output_string oc (Buffer.contents buf));
    Printf.printf "wrote %s\n%!" !metrics_out
  end;

  if !save_weights <> "" then begin
    Checkpoint.save !save_weights (checkpoint_of_params original !params);
    Printf.printf "wrote %s\n%!" !save_weights
  end
