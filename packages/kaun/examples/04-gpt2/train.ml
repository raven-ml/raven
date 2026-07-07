(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPT-2 124M training on a single fixed batch, matching the tinygrad
   reference protocol (see /_gpt2_train_reference in the repository):

   - fp32 weights from a local HuggingFace-layout safetensors file, no dropout
   - a fixed 4x65 token grid from tokens.json; inputs are columns 0..63 and
     targets columns 1..64, the same batch every step
   - mean cross-entropy over all 4*64 positions
   - plain SGD, lr 1e-4, no momentum; the LM head is tied to [wte] so the
     embedding's gradient accumulates from both of its uses
   - the whole step (forward, backward, update) compiles as one [Rune.jit2]
     program; the loss recorded at step i is computed before update i

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
   axis, NLL of the target id, mean. *)
let loss_fn inputs targets params =
  let logits = Gpt2.logits gpt2_124m params inputs in
  let logits =
    Nx.reshape [| batch_size * seq_len; gpt2_124m.vocab_size |] logits
  in
  Loss.softmax_cross_entropy_sparse logits targets

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

let train_step inputs targets params =
  let loss, grads =
    Rune.value_and_grad (module Gpt2.Params) (loss_fn inputs targets) params
  in
  (* Plain SGD: momentum is 0, so the zero velocity from [sgd_init] leaves the
     update exactly [w - lr * g] and needs no threading across steps. *)
  let state = Vega.sgd_init (module Gpt2.Params) params in
  let params =
    fst (Vega.sgd_step (module Gpt2.Params) ~lr state ~params ~grads)
  in
  { Step_out.params; loss }

(* Fingerprints: [first8] is bitwise (fp32 printed as float64 repr); [sum] and
   [abs_sum] accumulate in float64 with pairwise summation, as the reference. *)

let rec pairwise ~abs a lo n =
  if n <= 128 then begin
    let s = ref 0.0 in
    for i = lo to lo + n - 1 do
      let x = Bigarray.Array1.unsafe_get a i in
      s := !s +. (if abs then Float.abs x else x)
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
  (repr_float (pairwise ~abs:false a 0 n),
   repr_float (pairwise ~abs:true a 0 n),
   first8)

(* The six designated tensors, in the reference's in-memory layout (tinygrad
   Linear [out; in]): c_attn is the q/k/v weights re-fused into the HF
   [768; 2304] matrix, then — like c_fc and c_proj — transposed from kaun's
   [inputs; outputs] to [outputs; inputs]. *)
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
  let block i (b : Gpt2.block) =
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
      tensor
        (key "attn.c_attn.weight")
        (Nx.concatenate ~axis:1 [ b.attn.q.w; b.attn.k.w; b.attn.v.w ]);
      tensor
        (key "attn.c_attn.bias")
        (Nx.concatenate
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
    @ linear "mlp.c_fc" b.fc
    @ linear "mlp.c_proj" b.proj
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
       "{\n \"device\": %s,\n \"steps\": %d,\n \"batch_size\": %d,\n \
        \"seq_len\": %d,\n \"lr\": %s,\n \"n_params\": %d,\n \
        \"initial_loss\": %s,\n \"jit_calls\": null,\n \"records\": [\n"
       (str device) steps batch_size seq_len (repr_float lr) n_params
       (str initial_loss));
  List.iteri
    (fun i (step, loss, ms, weights) ->
      if i > 0 then Buffer.add_string buf ",\n";
      Buffer.add_string buf
        (Printf.sprintf "  {\"step\": %d, \"loss\": %s, \"ms\": %s,\n\
                        \   \"weights\": {" step (str loss) (repr_float ms));
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
  let steps = ref 20 in
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
      ("--steps", Arg.Set_int steps, "Number of training steps");
      ("--tokens", Arg.Set_string tokens, "Path to the tokens.json grid");
      ("--model", Arg.Set_string model, "Path to the initial safetensors");
      ("--metrics-out", Arg.Set_string metrics_out, "Metrics JSON path");
      ( "--save-weights",
        Arg.Set_string save_weights,
        "Save final weights (input-file layout) to this path" );
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "train --device DEV [--steps N] [--tokens F] [--model F] [--metrics-out \
     F] [--save-weights F]";

  let inputs, targets = batch_of_ids (load_ids !tokens) in
  let original = Checkpoint.load !model in
  let params = ref (Gpt2.of_checkpoint gpt2_124m original) in
  let n_params =
    let n = ref 0 in
    Gpt2.Params.iter (fun _ -> incr n) !params;
    !n
  in

  (* Untrained-loss probe, eagerly on the C backend: independent of the jit,
     the optimizer and the backward pass. *)
  let initial_loss = repr_float (Nx.item [] (loss_fn inputs targets !params)) in
  Printf.printf "initial loss (before step 1): %s\n%!" initial_loss;

  let step =
    Rune.jit2 ~device:!device
      (module Gpt2.Params)
      (module Step_out)
      (train_step inputs targets)
  in
  let records = ref [] in
  for i = 1 to !steps do
    let t0 = Unix.gettimeofday () in
    let out = step !params in
    let loss = Nx.item [] out.Step_out.loss in
    let ms = (Unix.gettimeofday () -. t0) *. 1e3 in
    params := out.Step_out.params;
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
