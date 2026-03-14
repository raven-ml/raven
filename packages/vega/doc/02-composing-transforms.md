# Composing Transforms

Vega's core abstraction is the composable gradient transformation. Every
optimizer — `adam`, `sgd`, `adamw` — is built by chaining small, focused
primitives. You can use these same primitives to build custom optimizers.

## How Aliases Work

Each alias is shorthand for `chain`. For example, `adamw` is:

<!-- $MDX skip -->
```ocaml
let adamw ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) ?(weight_decay = 0.01) lr =
  Vega.chain [
    Vega.scale_by_adam ~b1 ~b2 ~eps ();
    Vega.add_decayed_weights ~rate:(Vega.Schedule.constant weight_decay) ();
    Vega.scale_by_learning_rate lr;
  ]
```

The gradient flows through each primitive in order:
1. `scale_by_adam` — normalize by bias-corrected first and second moment estimates
2. `add_decayed_weights` — add `weight_decay * param` to the updates
3. `scale_by_learning_rate` — multiply by `-lr` for gradient descent

## Building Custom Optimizers

Since `chain` accepts any list of primitives, you can mix and match freely.

### Adding Gradient Clipping

Prepend a clipping transform to any optimizer:

<!-- $MDX skip -->
```ocaml
(* Clip gradient L2 norm before Adam *)
let tx =
  Vega.chain [
    Vega.clip_by_norm 1.0;
    Vega.adam (Vega.Schedule.constant 1e-3);
  ]

(* Or clip element-wise *)
let tx =
  Vega.chain [
    Vega.clip_by_value 0.5;
    Vega.adam (Vega.Schedule.constant 1e-3);
  ]
```

### Centralized Adam with Weight Decay

Combine gradient centralization, Adam, weight decay, and a schedule:

<!-- $MDX skip -->
```ocaml
let lr =
  Vega.Schedule.warmup_cosine_decay
    ~init_value:0.0 ~peak_value:1e-3
    ~warmup_steps:1000 ~decay_steps:9000 ()
in
let tx =
  Vega.chain [
    Vega.centralize;
    Vega.scale_by_adam ();
    Vega.add_decayed_weights ~rate:(Vega.Schedule.constant 0.01) ();
    Vega.scale_by_learning_rate lr;
  ]
```

### LAMB from Primitives

LAMB adds a trust ratio on top of Adam with weight decay:

<!-- $MDX skip -->
```ocaml
let tx =
  Vega.chain [
    Vega.scale_by_adam ();
    Vega.add_decayed_weights ~rate:(Vega.Schedule.constant 0.01) ();
    Vega.scale_by_trust_ratio ();
    Vega.scale_by_learning_rate lr;
  ]
```

## Primitives Reference

### Scaling

| Primitive | Description | State |
|-----------|-------------|-------|
| `scale s` | Multiply updates by constant `s` | 0 tensors |
| `scale_by_schedule f` | Multiply updates by `f step` | 0 tensors |
| `scale_by_learning_rate lr` | Multiply by `-lr step` (negates for descent) | 0 tensors |

### Adaptive Scaling

| Primitive | Description | State |
|-----------|-------------|-------|
| `scale_by_adam` | Bias-corrected 1st/2nd moments (Adam core) | 2-3 tensors |
| `scale_by_rms` | Inverse RMS of past gradients (RMSprop core) | 1 tensor |
| `scale_by_adagrad` | Inverse root of accumulated squared gradients | 1 tensor |
| `scale_by_lion` | Sign-based updates with dual momentum | 1 tensor |
| `scale_by_radam` | Rectified Adam (adaptive vs momentum switching) | 2 tensors |
| `scale_by_trust_ratio` | LAMB/LARS trust ratio `\|\|param\|\| / \|\|updates\|\|` | 0 tensors |
| `scale_by_adafactor` | Factored 2nd moments for memory efficiency | 2 tensors |
| `scale_by_adan` | Adan with gradient difference momentum | 4 tensors |

### Accumulation

| Primitive | Description | State |
|-----------|-------------|-------|
| `trace` | Momentum (EMA of updates), optional Nesterov | 1 tensor |

### Regularization

| Primitive | Description | State |
|-----------|-------------|-------|
| `add_decayed_weights` | Add `rate * param` (decoupled weight decay) | 0 tensors |

### Clipping

| Primitive | Description | State |
|-----------|-------------|-------|
| `clip_by_value delta` | Clamp to `[-delta, +delta]` | 0 tensors |
| `clip_by_norm max_norm` | Rescale if L2 norm exceeds `max_norm` | 0 tensors |

### Gradient Processing

| Primitive | Description | State |
|-----------|-------------|-------|
| `centralize` | Subtract mean (all axes except first for 2D+) | 0 tensors |
| `add_noise` | Gaussian noise with annealing schedule | 0 tensors |

### Robustness

| Primitive | Description | State |
|-----------|-------------|-------|
| `apply_if_finite tx` | Skip updates containing NaN/Inf | inner + 1 tensor |

## Chain Associativity

`chain` is associative — nesting chains produces the same optimizer:

<!-- $MDX skip -->
```ocaml
(* These are equivalent: *)
let tx1 = Vega.chain [a; b; c]
let tx2 = Vega.chain [Vega.chain [a; b]; c]
let tx3 = Vega.chain [a; Vega.chain [b; c]]
```

This means you can build reusable sub-chains and compose them freely.

## Serialization

Save and restore optimizer state for checkpointing:

<!-- $MDX skip -->
```ocaml
(* Save *)
let count, tensors = Vega.state_to_tensors state in
(* ... persist count and tensors to disk ... *)

(* Restore *)
let state = Vega.state_of_tensors tx ~count tensors
```

`n_tensors tx` returns the total number of state tensors, useful for
pre-allocating storage.

## Next Steps

- [Learning Rate Schedules](../03-schedules/) — decay, warmup, restarts, and composition
- [Getting Started](../01-getting-started/) — basic usage and optimizer aliases
- [Optax Comparison](../04-optax-comparison/) — mapping from Python's Optax to Vega
