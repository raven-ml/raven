# Saving and Pretrained Weights

A trained model is a set of tensors. Saving writes them to a file, each under a name, and loading reads them back into the program's values. A kaun model is a record with a structure, an `Nx.Ptree.t`, and the structure names every tensor of the record, so it is all that saving and loading need. Files are [SafeTensors](https://huggingface.co/docs/safetensors/), which `nx.io` reads and writes as an `Nx_io.Archive.t`: an immutable collection of tensors keyed by distinct, non-empty names. Weights published by others have their own names and layouts, and an ordinary function reads them by name into the model's record. This guide covers both.

## Names

The structure that the transformations and optimizers take, `Nx.Ptree.instantiate (module Mlp)`, names each tensor by its path, the fields that lead to it joined by `"."`:

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

Each layer's `walk` names its own fields (`w`, and `b` when the layer has a bias), so `mlp`'s tensors are `l1.w`, `l1.b`, `l2.w` and `l2.b`. A list element adds its index, so a model's third block is under `blocks.2`. `Nx.Ptree.field` puts a whole structure under a name: `Nx.Ptree.field "model" mlp` names the same tensors `model.l1.w`, ..., `model.l2.b`. Tensors may have different dtypes: each keeps its own in the file, and loading checks it. A tensor of a fixed type, walked with `Nx.Ptree.Walk.tensor`, is saved like the others.

## Saving and Loading

`Nx_io.Archive.of_value` turns a value into the archive of its tensors, and `Nx_io.save_safetensors` writes it. Reading back takes a value of the same shape, which a restart already holds: `Nx_io.Archive.to_value ~like` replaces its tensors with the archive's entries of the same names. `like` supplies the record, the list lengths, the options that are present, and each tensor's dtype and shape; its tensors are discarded:

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
  let model = Nx.Ptree.field "model" mlp in

  let path = Filename.temp_file "kaun-doc" ".safetensors" in
  Nx_io.save_safetensors path (Nx_io.Archive.of_value model params);

  let file = Nx_io.load_safetensors path in
  List.iter print_endline (Nx_io.Archive.names file);
  (* model.l1.b, model.l1.w, model.l2.b, model.l2.w *)

  let restored = Nx_io.Archive.to_value model ~like:(init ()) file in

  (* The restored parameters equal the saved ones. *)
  let x = Nx.randn Nx.float32 [| 2; 4 |] in
  let d = Nx.max (Nx.abs (Nx.sub (Mlp.apply params x) (Mlp.apply restored x))) in
  Printf.printf "max difference: %g\n" (Nx.item [] d)
```

A structure owns the names under its outermost fields, `model` here, and every name when it has none. Within them, reading back is strict, and fails with `Failure`, naming the entry, before any tensor is read:

- when a tensor of `like` has no entry;
- when an entry has another dtype or shape, as in `model.l1.w: shape [8; 4] in the archive, [4; 8] in the value`;
- when an entry is named by no tensor of `like`: a model of 12 blocks refuses a file of 24, since `model.blocks.12.w` names nothing.

Nothing is converted: `like` states the dtype it expects, so a restart that names the wrong dtype fails instead of narrowing its state.

## The Rule About Files

Loading reads the file's header. Each entry stays in the file, on the disk device, and is read when it is first used. Loading a 2.5 GB file takes milliseconds and allocates nothing.

**A file must not change while a tensor read from it is alive.** Truncating or rewriting it in place changes the tensors' values or kills the process. Replace a file by writing a new one and renaming it over the old one, which is what `Nx_io.save_safetensors` does: saving over the path you loaded from is safe. `Nx.copy` gives a tensor that no longer depends on its file. The file stays mapped until the garbage collector has collected the last tensor over it, which can be later than the last use; on Windows it cannot be deleted until then. Keep weight files on a local disk.

## One File, Several Sections

Entries outside a structure's names are ignored, so one file holds model parameters and optimizer state side by side, each under its own field, and `Nx_io.Archive.union` puts their archives together. An optimizer state is a structure too, `Vega.adam_ptree mlp`, whose tensors are `mu.l1.w`, ..., `nu.l2.b` and `step`. Saving and restoring the full training state:

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
  let ostate = Vega.adam_init mlp params in
  let model = Nx.Ptree.field "model" mlp in
  let optim = Nx.Ptree.field "optim" (Vega.adam_ptree mlp) in

  let path = Filename.temp_file "kaun-doc" ".safetensors" in
  Nx_io.save_safetensors path
    (Nx_io.Archive.union
       [
         Nx_io.Archive.of_value model params;
         Nx_io.Archive.of_value optim ostate;
       ]);

  (* Resuming: each section is read with its own structure. *)
  let file = Nx_io.load_safetensors path in
  let like = init () in
  let params = Nx_io.Archive.to_value model ~like file in
  let ostate =
    Nx_io.Archive.to_value optim ~like:(Vega.adam_init mlp like) file
  in
  ignore params;
  Printf.printf "resumed at step %d\n" (Int32.to_int (Nx.item [] ostate.step))
```

The optimizer moments are named after the model's paths because their structure nests the model's. `Batch_norm` running statistics are saved the same way, under their own field, with `Nx.Ptree.instantiate (module Batch_norm.Stats)`. A training state saved as one value is a module whose `walk` visits each part with `Nx.Ptree.Walk.structure`, so its paths are `params.…` and `opt.mu.…`; see [Writing structures](../../nx/doc/06-structures.md).

A section can be read into part of a model. A classifier saved under `model`, with fields `backbone` and `head`, gives its backbone to a new model through `Nx.Ptree.field "model" (Nx.Ptree.field "backbone" backbone)`: that structure owns the names under `model.backbone`, and the old head's entries are left alone.

## Pretrained Weights

Weights produced elsewhere are named and laid out by their exporter's conventions. Two functions read one entry by name:

- `Nx_io.Archive.tensor ~shape dtype name a` is the entry, which must have that shape and dtype. It is returned as stored, unread in its file.
- `Nx_io.Archive.float ~shape dtype name a` is a `float16`, `bfloat16`, `float32` or `float64` entry at `dtype`, itself one of those four: the entry as stored when it already has `dtype`, and its cast otherwise, which allocates that tensor. Any other entry is refused.

Both raise `Failure` naming the entry when it is missing or has another shape, so a configuration that disagrees with the file fails at import, before any computation. Bytes are never reinterpreted: only `float` converts. Block-quantised and packed weights are read with `tensor Nx.uint8`, and can sit in the same record as float tensors, since each field's type is the dtype passed for it.

An importer has the outline of the function that initializes the model, with a name in the file where that one has an initializer. Here is a two-layer network whose file stores each weight as `[outputs; inputs]`, the transpose of `Linear`'s layout:

```ocaml
let mlp_of_file dt weights =
  let linear ~inputs ~outputs name =
    {
      Linear.w =
        Nx.matrix_transpose
          (Nx_io.Archive.float ~shape:[| outputs; inputs |] dt
             (name ^ ".weight") weights);
      b =
        Some
          (Nx_io.Archive.float ~shape:[| outputs |] dt (name ^ ".bias") weights);
    }
  in
  {
    Mlp.l1 = linear ~inputs:4 ~outputs:8 "encoder.fc1";
    l2 = linear ~inputs:8 ~outputs:2 "encoder.fc2";
  }

let () =
  let weights =
    Nx_io.Archive.of_list
      [
        ("encoder.fc1.weight", Nx.P (Nx.ones Nx.float32 [| 8; 4 |]));
        ("encoder.fc1.bias", Nx.P (Nx.zeros Nx.float32 [| 8 |]));
        ("encoder.fc2.weight", Nx.P (Nx.ones Nx.float32 [| 2; 8 |]));
        ("encoder.fc2.bias", Nx.P (Nx.zeros Nx.float32 [| 2 |]));
      ]
  in
  let params = mlp_of_file Nx.float32 weights in
  let y = Mlp.apply params (Nx.ones Nx.float32 [| 1; 4 |]) in
  Printf.printf "%g\n" (Nx.item [ 0; 0 ] y)
```

Each name is written at the field it fills. A rename is a different string at the field, a transpose is `Nx.matrix_transpose`, and a fused tensor is cut with `Nx.split`; both are views. Structure comes from the configuration and the dtype is an argument: at the file's own dtype every tensor is the file's entry and nothing is copied, and at another one each is cast as it is read. An importer allocates no model to start from. An importer that ties two weights binds the tensor once and uses it twice.

## Placing Weights on a Device

A compiled function that captures host weights uploads them at its first call, once per compiled function, and the host copy stays alive beside the device copy. `Nx.place` copies a tensor into a device buffer and returns the copy, a value with the same elements, on a device (an `Nx.Device.t`, such as `Nx_metal.device 0`); the source is unchanged. A compiled function that captures such a value uses its buffer directly: nothing is uploaded, and prefill, decode and any other compiled function over the model share one copy. An importer places each leaf as it builds it, through a let-bound `place` that serves leaves of any dtype:

```ocaml
let mlp_of_file ?placement dt weights =
  let place x = match placement with None -> x | Some p -> Nx.place p x in
  let linear ~inputs ~outputs name =
    let float ~shape leaf =
      Nx_io.Archive.float ~shape dt (name ^ leaf) weights
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

Each leaf is read, cast if asked, transposed and placed before the next is touched, so the host holds at most one leaf's cast at a time and the model ends up as one copy, on the device. Placing on the GPU is `mlp_of_file ~placement:(Nx.Placement.on (Nx_cuda.device 0)) dt weights`. The examples' importers take `?placement` as a function of each leaf's role (a column or row projection, or experts) and of the axis that role's cut runs along in the leaf, so a placement that splits a model over several devices can cut each weight where it belongs. One device is `fun _ ~axis:_ -> p`. Their cache builders take the same function, with a `Kv_heads` role for the pools. A view of a placed value, such as a transpose or a slice, stays placed and a compiled function reads it in place; any other nx operation on a GPU value outside a compiled function raises, so place the final form of each leaf.

## Fetching From the Hub: kaun.hf

The `kaun.hf` library fetches files from [HuggingFace Hub](https://huggingface.co) repositories into a local cache and loads a model's SafeTensors weights, single-file or sharded, as an `Nx_io.Archive.t`. Downloading shells out to `curl` (it must be on `PATH`). A download is written beside its cache path and renamed once complete, and cached files are served without touching the network.

<!-- $MDX skip -->
```ocaml
let weights = Kaun_hf.load_safetensors "gpt2" in
List.iter print_endline (Nx_io.Archive.names weights)
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
let params = Gpt2.of_hf cfg Nx.float32 (Kaun_hf.load_safetensors "gpt2")
```

There is no per-architecture loader in the library. [`examples/05-llama`](https://github.com/raven-ml/raven/tree/main/packages/kaun/examples/05-llama) imports Llama 3.2 the same way, transposing every projection, and its `--dtype` defaults to the file's `bfloat16`, so the default run casts nothing.

## Next Steps

- [PyTorch Comparison](05-pytorch-comparison.md): `state_dict`, `torch.save`, and `from_pretrained` in kaun terms
- [Layers and Models](02-layers-and-models.md): where the names come from
