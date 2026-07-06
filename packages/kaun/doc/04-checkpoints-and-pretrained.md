# Checkpoints and Pretrained Models

A checkpoint is an immutable collection of tensors keyed by distinct, non-empty names, stored as a [safetensors](https://huggingface.co/docs/safetensors/) file. Typed parameter structures enter and leave checkpoints through names you declare; foreign checkpoints — HuggingFace Hub exports, say — are adapted checkpoint-to-checkpoint until they match your names. This guide covers both directions.

## Named Structures

`Checkpoint` consumes `Checkpoint.Named` modules: `Nx.Ptree.S` plus one function, `names`, giving each tensor leaf a stable name in traversal order. By convention leaves are named after record fields, with nested structures joined by `"."`:

```ocaml
open Kaun

module Mlp = struct
  type t = { l1 : Linear.t; l2 : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { l1; l2 } =
    { l1 = Linear.map f l1; l2 = Linear.map f l2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { l1 = Linear.map2 f p.l1 q.l1; l2 = Linear.map2 f p.l2 q.l2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { l1; l2 } =
    Linear.iter f l1;
    Linear.iter f l2

  let names { l1; l2 } =
    List.map (( ^ ) "l1.") (Linear.names l1)
    @ List.map (( ^ ) "l2.") (Linear.names l2)

  let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end
```

Each layer module ships its own `names` (`Linear.names` is `["w"; "b"]`, or `["w"]` without a bias), so a model's `names` is the same one-liner shape as its traversals. For structures only known at runtime, `Checkpoint.Ptree` names dynamic `Rune.Ptree.t` trees by their path from the root.

## Saving and Loading

`of_params` turns a structure into named entries; `save` writes them. Loading is template-based: construct the model first, then `to_params ~like` replaces its values with the file's entries of the same names — the template supplies structure, names, dtypes, and shapes, and its values are discarded:

```ocaml
let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let init () =
    {
      Mlp.l1 = Linear.init ~inputs:4 ~outputs:8;
      l2 = Linear.init ~inputs:8 ~outputs:2;
    }
  in
  let params = init () in

  let path = Filename.temp_file "kaun-doc" ".safetensors" in
  Checkpoint.save path
    (Checkpoint.of_params (module Mlp) ~prefix:"model" params);

  let ckpt = Checkpoint.load path in
  List.iter print_endline (Checkpoint.names ckpt);
  (* model.l1.b, model.l1.w, model.l2.b, model.l2.w *)

  let restored =
    Checkpoint.to_params (module Mlp) ~prefix:"model" ~like:(init ()) ckpt
  in
  Sys.remove path;

  (* The restored parameters equal the saved ones. *)
  let x = Nx.randn Nx.float32 [| 2; 4 |] in
  let d = Nx.max (Nx.abs (Nx.sub (Mlp.apply params x) (Mlp.apply restored x))) in
  Printf.printf "max difference: %g\n" (Nx.item [] d)
```

A missing entry, shape mismatch, or dtype mismatch raises (`~cast:true` casts mismatched dtypes instead). Entries the template does not name are ignored — the basis for both multi-section files and partial loading.

## One File, Several Sections

Because extraction ignores unnamed entries, one file holds model parameters, parameter-shaped optimizer state, and counters side by side, each under its own prefix. Saving and restoring full training state:

```ocaml
let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let init () =
    {
      Mlp.l1 = Linear.init ~inputs:4 ~outputs:8;
      l2 = Linear.init ~inputs:8 ~outputs:2;
    }
  in
  let params = init () in
  let ostate = Vega.adam_init (module Mlp) params in

  let path = Filename.temp_file "kaun-doc" ".safetensors" in
  Checkpoint.save path
    (Checkpoint.concat
       [
         Checkpoint.of_params (module Mlp) ~prefix:"model" params;
         Checkpoint.of_params (module Mlp) ~prefix:"optim.mu" ostate.mu;
         Checkpoint.of_params (module Mlp) ~prefix:"optim.nu" ostate.nu;
         Checkpoint.of_int "optim.step" ostate.step;
       ]);

  (* Resuming: extract each section with its own prefix. *)
  let ckpt = Checkpoint.load path in
  let like = init () in
  let params =
    Checkpoint.to_params (module Mlp) ~prefix:"model" ~like ckpt
  in
  let ostate =
    {
      Vega.mu = Checkpoint.to_params (module Mlp) ~prefix:"optim.mu" ~like ckpt;
      nu = Checkpoint.to_params (module Mlp) ~prefix:"optim.nu" ~like ckpt;
      step = Checkpoint.to_int "optim.step" ckpt;
    }
  in
  Sys.remove path;
  ignore params;
  Printf.printf "resumed at step %d\n" ostate.step
```

The optimizer moments checkpoint with the *model's* module because they have the model's shape — one more payoff of parameter-shaped state. `Batch_norm` running statistics work the same way, under their own prefix with `(module Model.Stats)`.

To load a file into a partially different model — a new head on a pretrained backbone, say — extract each sub-structure with its own module and prefix; entries for the parts you replace are simply never asked for.

## Foreign Checkpoints: kaun.hf

The `kaun.hf` library fetches files from [HuggingFace Hub](https://huggingface.co) repositories into a local cache and loads safetensors checkpoints — single-file or sharded — as `Checkpoint.t` values. Downloading shells out to `curl` (it must be on `PATH`); fetched files are cached, so only the first access touches the network.

<!-- $MDX skip -->
```ocaml
let ckpt = Kaun_hf.load_checkpoint "gpt2" in
List.iter print_endline (Kaun.Checkpoint.names ckpt)
(* h.0.attn.c_attn.bias, h.0.attn.c_attn.weight, ..., wte.weight *)
```

Hub checkpoints name and lay out tensors by the exporting framework's conventions, which will not match your records. Three checkpoint-to-checkpoint combinators adapt them, composed with `(|>)`:

- `rename f t` — replace every entry name `n` by `f n`; return names you do not care about unchanged.
- `transpose name t` — swap the last two axes of one entry. Use it on weights stored with the opposite orientation, such as `torch.nn.Linear`'s `outputs × inputs` weights when your model expects `inputs × outputs`.
- `split name ~into t` — replace one entry by equal sections along an axis. Use it on fused projections.

Leftover entries after adaptation are harmless: `to_params` ignores entries its template does not name.

## The GPT-2 Story

[`examples/04-gpt2`](https://github.com/raven-ml/raven/tree/main/packages/kaun/examples/04-gpt2) runs the whole pipeline: it defines GPT-2 as a record of kaun layers (~150 lines), loads the real weights, and generates text. The adaptation is instructive because the HF checkpoint differs from the natural kaun model in two ways:

1. **Fused attention projections.** HF stores each block's query, key, and value projections as one `c_attn` tensor of shape `[n_embd; 3 * n_embd]`; the model has three separate `Linear` layers. `split` cuts the fused weight and bias into thirds:

<!-- $MDX skip -->
```ocaml
let split_qkv ckpt i =
  let fused leaf = Printf.sprintf "h.%d.attn.c_attn.%s" i leaf in
  let ours p leaf = Printf.sprintf "blocks.%d.attn.%s.%s" i p leaf in
  ckpt
  |> Hf.split (fused "weight")
       ~into:[ ours "q" "w"; ours "k" "w"; ours "v" "w" ]
  |> Hf.split (fused "bias")
       ~into:[ ours "q" "b"; ours "k" "b"; ours "v" "b" ]
```

2. **Foreign names.** Everything else is a pure renaming — `wte.weight` to `wte.table`, `h.0.ln_1.weight` to `blocks.0.ln1.gamma`, and so on — one total function passed to `rename`. (GPT-2's `Conv1D` weights are already `inputs × outputs`, so no transposes are needed; a PyTorch `nn.Linear` export would need them.) Entries the model does not use, like attention mask buffers, are left in place and ignored by extraction.

Adapted, the checkpoint matches the model's own `names`, and typed parameters come out through the ordinary template-based extraction:

<!-- $MDX skip -->
```ocaml
let params =
  Kaun_hf.load_checkpoint "gpt2"
  |> Gpt2.of_hf ~n_layer:cfg.n_layer
  |> Checkpoint.to_params (module Gpt2.Params) ~like:(Gpt2.make cfg) ~cast:true
```

The `~like` template is a zero-initialized model built from the downloaded `config.json` (`Kaun_hf.load_config` fetches and parses it). There is no per-architecture loader in the library: the adaptation is ~40 lines of user code, and the same three combinators cover other exports.

## Next Steps

- [PyTorch Comparison](05-pytorch-comparison/) — `state_dict`, `torch.save`, and `from_pretrained` in kaun terms
- [Layers and Models](02-layers-and-models/) — where `names` comes from
