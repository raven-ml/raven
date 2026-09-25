# Checkpoints and Pretrained Models

A checkpoint is an immutable collection of tensors keyed by distinct, non-empty names, stored as a [safetensors](https://huggingface.co/docs/safetensors/) file. A file this library wrote names its entries after the paths of your parameter structure, and is read back against a value of that structure. A file produced elsewhere has its own names and layouts, and its importer is an ordinary function that reads each entry by name and builds the parameter record. This guide covers both.

## Named Structures

`Checkpoint` takes the model's structure, the same `Nx.Ptree.instantiate (module Mlp)` value that the transformations and optimizers take. Each tensor's name is its path, the fields that lead to it joined by `"."`:

```ocaml
open Kaun

module Mlp = struct
  type 'a t = { l1 : 'a Linear.t; l2 : 'a Linear.t }

  let walk c { l1; l2 } =
    let open Nx.Ptree.Walk in
    let l1 = field c "l1" Linear.walk l1 in
    let l2 = field c "l2" Linear.walk l2 in
    { l1; l2 }

  let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end

let mlp = Nx.Ptree.instantiate (module Mlp)
```

Each layer's `walk` names its own fields (`w`, and `b` when the layer has a bias), so `mlp`'s leaves are `l1.w`, `l1.b`, `l2.w` and `l2.b`. A list element adds its index, so a model's third block is under `blocks.2`. Leaves may have different dtypes: each entry keeps its own, and loading checks it. A tensor of a fixed type in a structure, walked with `Nx.Ptree.Walk.tensor`, is an entry like the others.

## Saving and Loading

`of_value` turns a value of a structure into named entries; `save` writes them. Reading them back takes a value of the structure, which a restart already holds: `to_value ~like` replaces its values with the file's entries of the same names. The template supplies structure, names, dtypes and shapes, and its values are discarded:

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
    (Checkpoint.of_value ~prefix:"model" mlp params);

  let ckpt = Checkpoint.load path in
  List.iter print_endline (Checkpoint.names ckpt);
  (* model.l1.b, model.l1.w, model.l2.b, model.l2.w *)

  let restored = Checkpoint.to_value ~prefix:"model" mlp ~like:(init ()) ckpt in

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

Because extraction ignores unnamed entries, one file holds model parameters and optimizer state side by side, each under its own prefix. An optimizer state is a structure too, `Vega.adam_ptree mlp`, whose leaves are `mu.l1.w`, ..., `nu.l2.b` and `step`. Saving and restoring full training state:

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
  let adam = Vega.adam_ptree mlp in
  let ostate = Vega.adam_init mlp params in

  let path = Filename.temp_file "kaun-doc" ".safetensors" in
  Checkpoint.save path
    (Checkpoint.concat
       [
         Checkpoint.of_value ~prefix:"model" mlp params;
         Checkpoint.of_value ~prefix:"optim" adam ostate;
       ]);

  (* Resuming: extract each section with its own prefix. *)
  let ckpt = Checkpoint.load path in
  let like = init () in
  let params = Checkpoint.to_value ~prefix:"model" mlp ~like ckpt in
  let ostate =
    Checkpoint.to_value ~prefix:"optim" adam
      ~like:(Vega.adam_init mlp like) ckpt
  in
  ignore params;
  Printf.printf "resumed at step %d\n" (Int32.to_int (Nx.item [] ostate.step))
```

The optimizer moments are named after the model's paths because their structure nests the model's. `Batch_norm` running statistics work the same way, under their own prefix with `Nx.Ptree.instantiate (module Batch_norm.Stats)`. A training state saved as one value is a module whose `walk` visits each part with `Nx.Ptree.Walk.structure`, so its paths are `params.…` and `opt.mu.…`; see [Writing structures](../../nx/doc/06-structures.md).

To load a file into a partially different model (a new head on a pretrained backbone, say), extract each sub-structure with its own structure and prefix; entries for the parts you replace are never asked for.

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

## Placing Weights on a Device

A compiled function that captures host weights uploads them at its first call, once per compiled function, and the host copy stays alive beside the device copy. `Nx.place` copies a tensor into a device buffer and returns the copy, a value with the same elements, on a device that `Rune.device` names; the source is unchanged. A compiled function that captures such a value uses its buffer directly: nothing is uploaded, and prefill, decode and any other compiled function over the model share one copy. An importer places each leaf as it builds it, through a let-bound `place` that serves leaves of any dtype:

```ocaml
let mlp_of_file ?placement dt ckpt =
  let place x = match placement with None -> x | Some p -> Nx.place p x in
  let linear ~inputs ~outputs name =
    let float ~shape leaf =
      Checkpoint.to_float ~shape dt (name ^ leaf) ckpt
    in
    {
      Linear.w =
        place (Nx.matrix_transpose (float ~shape:[| outputs; inputs |] ".weight"));
      b = Some (place (float ~shape:[| outputs |] ".bias"));
    }
  in
  {
    Mlp.l1 = linear ~inputs:4 ~outputs:8 "encoder.fc1";
    l2 = linear ~inputs:8 ~outputs:2 "encoder.fc2";
  }
```

Each leaf is read, cast if asked, transposed and placed before the next is touched, so the host holds at most one leaf's cast at a time and the model ends up as one copy, on the device. Placing on Metal is `mlp_of_file ~placement:(Nx.Placement.device (Rune.device "METAL")) dt ckpt`. The examples' importers take `?placement` as a function of each leaf's role (a column or row projection, or experts) and of the axis that role's cut runs along in the leaf, so a placement that splits a model over several devices can cut each weight where it belongs. One device is `fun _ ~axis:_ -> p`. Their cache builders take the same function, with a `Kv_heads` role for the pools. A view of a placed value, such as a transpose or a slice, stays placed and a compiled function reads it in place; any other nx operation on a placed value outside a compiled function computes on the host and places its result, so place the final form of each leaf.

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
- [Layers and Models](02-layers-and-models.md): where the names come from
