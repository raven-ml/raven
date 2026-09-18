# Checkpoints and Pretrained Models

A checkpoint is an immutable collection of tensors keyed by distinct, non-empty names, stored as a [safetensors](https://huggingface.co/docs/safetensors/) file. A file this library wrote names its entries after the paths of your parameter structure, and is read back against a value of that structure. A file produced elsewhere has its own names and layouts, and its importer is an ordinary function that reads each entry by name and builds the parameter record. This guide covers both.

## Named Structures

`Checkpoint` consumes the structure's `Nx.Ptree.Uniform` module — the same traversals the model already has, whose `fold`/`fold2`/`names` give each tensor leaf a stable path. Leaves are named after record fields, with nested structures joined by `"."`:

```ocaml
open Kaun

module Mlp = struct
  type 'a t = { l1 : 'a Linear.t; l2 : 'a Linear.t }

  let map f { l1; l2 } =
    { l1 = Linear.map f l1; l2 = Linear.map f l2 }

  let map2 f p q =
    { l1 = Linear.map2 f p.l1 q.l1; l2 = Linear.map2 f p.l2 q.l2 }

  let iter f { l1; l2 } =
    Linear.iter f l1;
    Linear.iter f l2

  let fold f acc { l1; l2 } =
    let acc = Linear.fold (fun s -> f ("l1." ^ s)) acc l1 in
    Linear.fold (fun s -> f ("l2." ^ s)) acc l2

  let fold2 f acc p q =
    let acc = Linear.fold2 (fun s -> f ("l1." ^ s)) acc p.l1 q.l1 in
    Linear.fold2 (fun s -> f ("l2." ^ s)) acc p.l2 q.l2

  let names { l1; l2 } =
    {
      l1 = Linear.map (( ^ ) "l1.") (Linear.names l1);
      l2 = Linear.map (( ^ ) "l2.") (Linear.names l2);
    }

  let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end
```

Each layer module ships its own traversals (`Linear.names p` is `{ w = "w"; b = Some "b" }`, with `b` absent when the layer has no bias), so a model's `fold`/`fold2`/`names` are one-liners of the same shape as its `map` — or one `[@@deriving ptree]`. Structures with mixed leaf dtypes hold packed leaves and go through `of_packed`/`to_packed` instead; for the stock dynamic tree `Rune.Ptree.t`, pass `(module Rune.Ptree.Tree)`, which names leaves by dict keys and list positions from the root.

## Saving and Loading

`of_params` turns a structure into named entries; `save` writes them. Reading them back takes a value of the structure, which a restart already holds: `to_params ~like` replaces its values with the file's entries of the same names. The template supplies structure, names, dtypes and shapes, and its values are discarded:

```ocaml
let () =
  Nx.Rng.with_key (Nx.Rng.key 0) @@ fun () ->
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

  (* The restored parameters equal the saved ones. *)
  let x = Nx.randn Nx.float32 [| 2; 4 |] in
  let d = Nx.max (Nx.abs (Nx.sub (Mlp.apply params x) (Mlp.apply restored x))) in
  Printf.printf "max difference: %g\n" (Nx.item [] d)
```

A missing entry, a shape mismatch or a dtype mismatch raises. The template states the dtype it expects and nothing is converted, so a restart that names the wrong dtype fails instead of narrowing its state. Entries the template does not name are ignored, which is what makes multi-section files and partial loading work.

## The Rule About Files

Loading reads the file's header and maps the file. Each entry is a view of it, and the system reads an entry's pages when they are first used. Loading a 2.5 GB checkpoint takes milliseconds and allocates nothing.

**A file must not change while a tensor read from it is alive.** Truncating or rewriting it in place changes the tensors' values or kills the process. Replace a checkpoint by writing a new file and renaming it over the old one, which is what `Checkpoint.save` does: saving over the path you loaded from is safe. `Nx.copy` gives a tensor that no longer depends on its file. The file stays mapped until the garbage collector has collected the last tensor over it, which can be later than the last use; on Windows it cannot be deleted until then. Keep checkpoints on a local disk.

## One File, Several Sections

Because extraction ignores unnamed entries, one file holds model parameters, parameter-shaped optimizer state, and counters side by side, each under its own prefix. Saving and restoring full training state:

```ocaml
let () =
  Nx.Rng.with_key (Nx.Rng.key 0) @@ fun () ->
  let init () =
    {
      Mlp.l1 = Linear.init ~inputs:4 ~outputs:8;
      l2 = Linear.init ~inputs:8 ~outputs:2;
    }
  in
  let params = init () in
  let ostate = Vega.adam_init (Kaun.ptree (module Mlp)) params in

  let path = Filename.temp_file "kaun-doc" ".safetensors" in
  Checkpoint.save path
    (Checkpoint.concat
       [
         Checkpoint.of_params (module Mlp) ~prefix:"model" params;
         Checkpoint.of_params (module Mlp) ~prefix:"optim.mu" ostate.mu;
         Checkpoint.of_params (module Mlp) ~prefix:"optim.nu" ostate.nu;
         Checkpoint.of_tensor "optim.step" ostate.step;
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
      step = Nx.Ptree.unpack Nx.int32 (Checkpoint.get "optim.step" ckpt);
    }
  in
  ignore params;
  Printf.printf "resumed at step %d\n" (Int32.to_int (Nx.item [] ostate.step))
```

The optimizer moments checkpoint with the *model's* module because they have the model's shape — one more payoff of parameter-shaped state. `Batch_norm` running statistics work the same way, under their own prefix with `(module Batch_norm.Stats)`.

To load a file into a partially different model — a new head on a pretrained backbone, say — extract each sub-structure with its own module and prefix; entries for the parts you replace are simply never asked for.

## Pretrained Checkpoints

A checkpoint produced elsewhere names and lays out its tensors by its exporter's conventions. Two accessors read one entry by name:

- `Checkpoint.to_tensor ~shape dtype name ckpt` is the entry, which must have that shape and dtype. It is returned as stored, a view of the file.
- `Checkpoint.to_float ~shape dtype name ckpt` is a `float16`, `bfloat16`, `float32` or `float64` entry at `dtype`: the entry itself when it already has `dtype`, and its cast otherwise, which allocates that leaf. Any other entry is refused.

Both raise `Invalid_argument` with the entry's name when it is missing or has another shape, so a configuration that disagrees with the file fails at import, before any computation. Bytes are never reinterpreted silently: only `to_float` converts, and only between floating-point dtypes. Packed integer weights are read with `to_tensor Nx.uint8` and can sit in the same record as float leaves, since each field's type is the dtype passed to its accessor.

An importer has the outline of the function that initializes the model, with a name in the file where that one has an initializer. Here is a two-layer network whose file stores each weight as `[outputs; inputs]`, the transpose of `Linear`'s layout:

```ocaml
let mlp_of_file dt ckpt =
  let linear ~inputs ~outputs name =
    {
      Linear.w =
        Nx.matrix_transpose
          (Checkpoint.to_float ~shape:[| outputs; inputs |] dt
             (name ^ ".weight") ckpt);
      b = Some (Checkpoint.to_float ~shape:[| outputs |] dt (name ^ ".bias") ckpt);
    }
  in
  {
    Mlp.l1 = linear ~inputs:4 ~outputs:8 "encoder.fc1";
    l2 = linear ~inputs:8 ~outputs:2 "encoder.fc2";
  }

let () =
  let ckpt =
    Checkpoint.concat
      [
        Checkpoint.of_tensor "encoder.fc1.weight" (Nx.ones Nx.float32 [| 8; 4 |]);
        Checkpoint.of_tensor "encoder.fc1.bias" (Nx.zeros Nx.float32 [| 8 |]);
        Checkpoint.of_tensor "encoder.fc2.weight" (Nx.ones Nx.float32 [| 2; 8 |]);
        Checkpoint.of_tensor "encoder.fc2.bias" (Nx.zeros Nx.float32 [| 2 |]);
      ]
  in
  let params = mlp_of_file Nx.float32 ckpt in
  let y = Mlp.apply params (Nx.ones Nx.float32 [| 1; 4 |]) in
  Printf.printf "%g\n" (Nx.item [ 0; 0 ] y)
```

Each name is written at the field it fills. A rename is a different string at the field, a transpose is `Nx.matrix_transpose`, and a fused tensor is cut with `Nx.split`; both are views. Structure comes from the configuration and the dtype is an argument: at the file's own dtype every leaf is a view of the file and nothing is copied, and at another one each leaf is cast as it is read. An importer that ties two weights binds the tensor once and uses it twice.

## Fetching From the Hub: kaun.hf

The `kaun.hf` library fetches files from [HuggingFace Hub](https://huggingface.co) repositories into a local cache and loads safetensors checkpoints, single-file or sharded, as `Checkpoint.t` values. Downloading shells out to `curl` (it must be on `PATH`). A download is written beside its cache path and renamed once complete, and cached files are served without touching the network.

<!-- $MDX skip -->
```ocaml
let ckpt = Kaun_hf.load_checkpoint "gpt2" in
List.iter print_endline (Kaun.Checkpoint.names ckpt)
(* h.0.attn.c_attn.bias, h.0.attn.c_attn.weight, ..., wte.weight *)
```

`Kaun_hf.load_config` fetches and parses the repository's `config.json`, from which an example builds its configuration record.

## The GPT-2 Story

[`examples/04-gpt2`](https://github.com/raven-ml/raven/tree/main/packages/kaun/examples/04-gpt2) runs the whole pipeline: it defines GPT-2 as a record of kaun layers, loads the real weights, and generates text. Its importer is about fifty lines and differs from the two-layer one above in one way: the file stores each block's query, key and value projections as one `c_attn` tensor of shape `[n_embd; 3 * n_embd]`, where the model has three `Linear` layers. `Nx.split` cuts the fused weight and bias into three views:

<!-- $MDX skip -->
```ocaml
let fused = linear ~inputs:d ~outputs:(3 * d) (at "attn.c_attn") in
let q, k, v =
  match
    List.map2
      (fun w b -> { Linear.w; b = Some b })
      (Nx.split ~axis:1 3 fused.w)
      (Nx.split ~axis:0 3 (Option.get fused.b))
  with
  | [ q; k; v ] -> (q, k, v)
  | _ -> assert false
in
```

This file's weights are already `inputs × outputs`, so nothing is transposed. Entries the model does not use, such as attention mask buffers, are never read. The whole load is:

<!-- $MDX skip -->
```ocaml
let cfg = Gpt2.config_of_json (Kaun_hf.load_config "gpt2") in
let params = Gpt2.of_hf cfg Nx.float32 (Kaun_hf.load_checkpoint "gpt2")
```

There is no per-architecture loader in the library. [`examples/05-llama`](https://github.com/raven-ml/raven/tree/main/packages/kaun/examples/05-llama) imports Llama 3.2 the same way, transposing every projection, and its `--dtype` defaults to the file's `bfloat16`, so the default run casts nothing.

## Next Steps

- [PyTorch Comparison](05-pytorch-comparison.md): `state_dict`, `torch.save`, and `from_pretrained` in kaun terms
- [Layers and Models](02-layers-and-models.md) — where `names` comes from
