# Kaun-next vs. PyTorch — A Practical Comparison

This guide explains how kaun-next relates to PyTorch, focusing on:

* How core concepts map (modules, parameters, optimizers, training loops)
* Where the APIs feel similar vs. deliberately different
* How to translate common PyTorch patterns

The main shift is from mutable objects to immutable records: a PyTorch model is an object that owns its parameters and mutates them in place; a kaun-next model is a record of tensors that functions consume and produce. There is no module base class, no parameter registry, and no trainer.

---

## 1. Big-Picture Differences

| Aspect | PyTorch | kaun-next (OCaml) |
| --- | --- | --- |
| Language | Python, dynamic | OCaml, statically typed |
| Model definition | `nn.Module` subclass with `forward` | plain record + `apply` function |
| Parameter storage | mutable attributes, auto-registered | fields of your record |
| Parameter traversal | `model.parameters()` (reflection) | `map`/`map2`/`iter` — 3 one-liners you write |
| Forward pass | `model(x)` (stateful method) | `Model.apply params x` (pure function) |
| Autograd | dynamic tape on tensors (`loss.backward()`) | `Rune_next.value_and_grad` (effect handlers) |
| Gradients | `.grad` attributes, mutated in place | a fresh value of your record type |
| Optimizer | `torch.optim.AdamW(model.parameters())` | `Vega.adamw_step (module Model)` — pure, state threaded explicitly |
| Train/eval mode | `model.train()` / `model.eval()` | explicit `~training` arguments |
| Buffers (running stats) | hidden module state | explicit `Stats.t` you thread through the loop |
| Data loading | `DataLoader` | `Data.batches2` returning a `Seq.t` |
| Checkpointing | `state_dict()` + `torch.save` (pickle) | named entries + safetensors via `Checkpoint` |
| Pretrained models | `from_pretrained` per architecture | `kaun-next.hf` + generic `rename`/`transpose`/`split` |
| RNG | global `torch.manual_seed` | scoped `Nx.Rng.run ~seed` |
| Device | `model.to("cuda")` | CPU only |

---

## 2. Defining Models

**PyTorch**

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 8)
        self.l2 = nn.Linear(8, 1)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))

model = MLP()
```

**kaun-next**

```ocaml
open Kaun_next

module Mlp = struct
  type t = { l1 : Linear.t; l2 : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { l1; l2 } =
    { l1 = Linear.map f l1; l2 = Linear.map f l2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { l1 = Linear.map2 f p.l1 q.l1; l2 = Linear.map2 f p.l2 q.l2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { l1; l2 } =
    Linear.iter f l1;
    Linear.iter f l2

  let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end

let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let model =
    {
      Mlp.l1 = Linear.init ~inputs:4 ~outputs:8;
      l2 = Linear.init ~inputs:8 ~outputs:1;
    }
  in
  let y = Mlp.apply model (Nx.randn Nx.float32 [| 2; 4 |]) in
  ignore y
```

Where `nn.Module` registers parameters by reflection on attribute assignment, kaun-next asks you to write the traversal by hand — three mechanical one-liners per record. That is the entire cost, and it buys full typing: `model.l1.w` is a tensor field you can read directly, gradients of `Mlp.t` are `Mlp.t`, and there is no string-keyed parameter store to drift out of sync.

One deliberate difference: layer *hyper*-parameters that do not change the parameter shapes are arguments of `apply`, not stored configuration — `Attention.apply ~num_heads:12 ~causal:true`, `Layer_norm.apply ~eps:1e-5`.

---

## 3. The Training Step

**PyTorch** — gradients and parameters mutate in place:

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

def step(x, y):
    opt.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return loss.item()
```

**kaun-next** — the step is a pure function from state to state:

```ocaml
let step (params, ostate) (x, y) =
  let loss p = Loss.softmax_cross_entropy_sparse (Mlp.apply p x) y in
  let l, grads = Rune_next.value_and_grad (module Mlp) loss params in
  let params, ostate =
    Vega.adamw_step (module Mlp) ~lr:1e-3 ostate ~params ~grads
  in
  ((params, ostate), Nx.item [] l)
```

Point-by-point:

- `loss.backward()` → `Rune_next.value_and_grad (module Mlp) loss params`. No `.grad` attributes: the gradient is returned as a value.
- `opt.zero_grad()` → nothing. Gradients are fresh values, never accumulated in place.
- `opt.step()` → `Vega.adamw_step`. The optimizer holds no reference to the model; its state (`mu`, `nu`, `step`) is a record of `Mlp.t`-shaped values your loop threads explicitly.
- `torch.nn.utils.clip_grad_norm_` → `Vega.clip_by_global_norm (module Mlp) ~max_norm grads`, a pure function applied between the two.
- LR schedulers → a `Vega.Schedule.t` is `int -> float`; evaluate it at your own step counter and pass `~lr`.

Note the layering: `Vega` is an independent package composed in user code — kaun-next's library does not depend on it.

---

## 4. Train/Eval Mode and Buffers

**PyTorch** — mode and running statistics are hidden module state:

```python
model.train()
...
model.eval()
with torch.no_grad():
    out = model(x)
```

**kaun-next** — mode is an argument, statistics are a value:

<!-- $MDX skip -->
```ocaml
(* training *)
let pred, stats' = Net.forward params stats ~training:true x in
(* evaluation *)
let pred, _ = Net.forward params stats ~training:false x in
```

`Dropout.apply ~rate ~training` and `Batch_norm.apply p stats ~training` take the flag explicitly. What PyTorch calls a *buffer* (batch-norm running mean/var) is an explicit `Batch_norm.Stats.t` record: training forwards return the updated statistics, and `Rune_next.value_and_grad_aux` threads them out of the differentiated objective — see [Layers and Models](02-layers-and-models/). There is no `torch.no_grad()` context needed for evaluation: nothing is recorded unless you call a differentiation transformation (though `Rune_next.no_grad` exists to hold sub-computations constant *inside* one).

---

## 5. Data Loading

**PyTorch**

```python
loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=False)
for x, y in loader:
    ...
```

**kaun-next** — for in-memory tensors, a `Seq.t` of minibatches:

<!-- $MDX skip -->
```ocaml
let batches = Data.batches2 ~shuffle:true ~batch_size:128 (train_x, train_y) in
batches |> Seq.iter (fun (x, y) -> ...)
```

Iterating the sequence again reshuffles — one sequence value serves every epoch. There is no worker pool or async prefetching; datasets are plain tensors (the `kaun-next.datasets` library loads MNIST, Fashion-MNIST, CIFAR-10 that way), and stdlib `Seq` combinators replace `DataLoader` options.

---

## 6. Checkpointing

**PyTorch**

```python
torch.save({"model": model.state_dict(),
            "optim": opt.state_dict(),
            "step": step}, path)
ckpt = torch.load(path)
model.load_state_dict(ckpt["model"])
```

**kaun-next** — `state_dict()` becomes `Checkpoint.of_params` over a `Named` module (your traversals plus a `names` one-liner), and files are safetensors, not pickles:

<!-- $MDX skip -->
```ocaml
Checkpoint.save path
  (Checkpoint.concat
     [
       Checkpoint.of_params (module Mlp) ~prefix:"model" params;
       Checkpoint.of_params (module Mlp) ~prefix:"optim.mu" ostate.mu;
       Checkpoint.of_params (module Mlp) ~prefix:"optim.nu" ostate.nu;
       Checkpoint.of_int "optim.step" ostate.step;
     ]);

let params =
  Checkpoint.load path
  |> Checkpoint.to_params (module Mlp) ~prefix:"model" ~like:template
```

`load_state_dict`'s in-place mutation becomes template-based extraction: `~like` supplies structure, names, dtypes, and shapes, and a fresh value comes back. The optimizer state checkpoints with the model's own module because it has the model's shape. See [Checkpoints and Pretrained Models](04-checkpoints-and-pretrained/).

---

## 7. Pretrained Models

**PyTorch / transformers**

```python
model = AutoModel.from_pretrained("gpt2")
```

**kaun-next** — there is no per-architecture loader. `kaun-next.hf` downloads the checkpoint, and generic combinators adapt it onto your record's names:

<!-- $MDX skip -->
```ocaml
let params =
  Kaun_next_hf.load_checkpoint "gpt2"
  |> Gpt2.of_hf ~n_layer:cfg.n_layer (* split fused c_attn, rename *)
  |> Checkpoint.to_params (module Gpt2.Params) ~like:(Gpt2.make cfg) ~cast:true
```

`rename` maps foreign names to yours, `transpose` fixes `nn.Linear`'s `outputs × inputs` orientation, `split` cuts fused projections like GPT-2's `c_attn`. The GPT-2 adaptation is ~40 lines of user code; [`examples/04-gpt2`](https://github.com/raven-ml/raven/tree/main/packages/kaun-next/examples/04-gpt2) generates text with the result.

---

## 8. What Kaun-next Does Not Have

| PyTorch feature | Status in kaun-next |
| --- | --- |
| GPU / `model.to("cuda")` | Not available. Everything runs eagerly on CPU through Nx. |
| `torch.compile` / JIT | Not available. |
| Layer coverage | Deliberately small: no recurrent layers; `Attention` has no rotary embeddings or KV cache (write them from `scaled_dot_product_attention`); `Conv` is im2col-based and not tuned for large inputs. |
| Mixed precision / AMP | No; `Batch_norm` is single-precision only, other layers are generic over float dtypes. |
| `DataLoader` workers | No; data is in-memory tensors and a `Seq.t`. |
| Distributed (`DDP`) | Not available. |

---

## 9. Quick Cheat Sheet

| Task | PyTorch | kaun-next |
| --- | --- | --- |
| Define a model | `nn.Module` subclass | record + `apply` + 3 traversals |
| Construct | `MLP()` | `{ l1 = Linear.init ~inputs ~outputs; ... }` |
| Forward | `model(x)` | `Mlp.apply params x` |
| Backward | `loss.backward()` | `Rune_next.value_and_grad (module Mlp) loss params` |
| Zero grads | `opt.zero_grad()` | not needed |
| Optimizer step | `opt.step()` | `Vega.adamw_step (module Mlp) ~lr st ~params ~grads` |
| Clip gradients | `clip_grad_norm_` | `Vega.clip_by_global_norm` |
| LR schedule | `lr_scheduler` object | `Vega.Schedule.t`, evaluated at your counter |
| Train mode | `model.train()` | `~training:true` arguments |
| Running stats | hidden buffers | explicit `Stats.t` + `value_and_grad_aux` |
| Batches | `DataLoader` | `Data.batches2` |
| Save | `torch.save(model.state_dict())` | `Checkpoint.save` + `of_params` |
| Load | `load_state_dict` | `Checkpoint.to_params ~like` |
| Pretrained | `from_pretrained("gpt2")` | `Kaun_next_hf.load_checkpoint` + `rename`/`transpose`/`split` |
| Seed | `torch.manual_seed(42)` | `Nx.Rng.run ~seed:42` |
| Per-sample grads | `torch.func.vmap(grad(...))` | `Rune_next.vmap2` of `Rune_next.grad` |
