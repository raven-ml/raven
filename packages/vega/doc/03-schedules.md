# Learning Rate Schedules

A learning rate schedule controls how the learning rate changes over the
course of training. In Vega, a schedule is simply a function from step
number to learning rate.

## How Schedules Work

`Schedule.t` is `int -> float`. Given a 1-based step number, it returns
the learning rate for that step:

<!-- $MDX skip -->
```ocaml
let lr = Vega.Schedule.constant 0.001 in
Printf.printf "step 1:   %f\n" (lr 1);    (* 0.001 *)
Printf.printf "step 100: %f\n" (lr 100)   (* 0.001 *)
```

Schedules plug into optimizers as the last positional argument:

<!-- $MDX skip -->
```ocaml
let tx = Vega.adam lr
```

Or directly as a primitive:

<!-- $MDX skip -->
```ocaml
let tx =
  Vega.chain [
    Vega.scale_by_adam ();
    Vega.scale_by_learning_rate lr;
  ]
```

## Basic Schedules

### constant

A fixed learning rate:

<!-- $MDX skip -->
```ocaml
Vega.Schedule.constant 0.001
```

### linear

Linear interpolation from `init_value` to `end_value` over `steps`. Clamps
to `end_value` after:

<!-- $MDX skip -->
```ocaml
Vega.Schedule.linear ~init_value:0.0 ~end_value:0.001 ~steps:1000
(* step 1: ~0.0, step 500: ~0.0005, step 1000: 0.001, step 2000: 0.001 *)
```

## Decay Schedules

### cosine_decay

Cosine annealing from `init_value` to `alpha * init_value` over `decay_steps`:

<!-- $MDX skip -->
```ocaml
Vega.Schedule.cosine_decay ~init_value:0.01 ~decay_steps:10000 ()
(* Decays from 0.01 to 0.0 following a cosine curve *)

(* With a minimum floor *)
Vega.Schedule.cosine_decay ~init_value:0.01 ~decay_steps:10000 ~alpha:0.001 ()
(* Decays from 0.01 to 0.00001 *)
```

### exponential_decay

Multiply by `decay_rate` every `decay_steps`:

<!-- $MDX skip -->
```ocaml
Vega.Schedule.exponential_decay ~init_value:0.01 ~decay_rate:0.96 ~decay_steps:1000
(* lr = 0.01 * 0.96^(step/1000) *)
```

### polynomial_decay

Polynomial decay from `init_value` to `end_value`. `power` defaults to 1.0
(linear). Clamps to `end_value` after `decay_steps`:

<!-- $MDX skip -->
```ocaml
(* Linear decay (power=1) *)
Vega.Schedule.polynomial_decay ~init_value:0.01 ~end_value:0.0 ~decay_steps:10000 ()

(* Quadratic decay (power=2) — decays faster initially *)
Vega.Schedule.polynomial_decay ~init_value:0.01 ~end_value:0.0 ~decay_steps:10000
  ~power:2.0 ()
```

## Warmup Schedules

### warmup_cosine

Cosine warmup from `init_value` to `peak_value` over `warmup_steps`. Clamps
to `peak_value` after:

<!-- $MDX skip -->
```ocaml
Vega.Schedule.warmup_cosine ~init_value:0.0 ~peak_value:0.001 ~warmup_steps:1000
```

### warmup_cosine_decay

The most common schedule for transformer training: linear warmup followed
by cosine decay:

<!-- $MDX skip -->
```ocaml
Vega.Schedule.warmup_cosine_decay
  ~init_value:0.0       (* start from 0 *)
  ~peak_value:0.001     (* warm up to 0.001 *)
  ~warmup_steps:1000    (* over 1000 steps *)
  ~decay_steps:9000     (* then decay over 9000 steps *)
  ~end_value:0.0        (* down to 0 *)
  ()
```

## Warm Restarts

### cosine_decay_restarts

SGDR: cosine decay that periodically resets to the initial value. After each
restart, the period is multiplied by `t_mul` and the peak by `m_mul`:

<!-- $MDX skip -->
```ocaml
(* Fixed-period restarts *)
Vega.Schedule.cosine_decay_restarts ~init_value:0.01 ~decay_steps:1000 ()

(* Increasing period: 1000, 2000, 4000, ... *)
Vega.Schedule.cosine_decay_restarts ~init_value:0.01 ~decay_steps:1000
  ~t_mul:2.0 ()

(* Decreasing peak: 0.01, 0.005, 0.0025, ... *)
Vega.Schedule.cosine_decay_restarts ~init_value:0.01 ~decay_steps:1000
  ~m_mul:0.5 ()
```

### one_cycle

The 1cycle policy: linear warmup from `max_value / div_factor` to `max_value`,
then cosine decay to `max_value / final_div_factor`:

<!-- $MDX skip -->
```ocaml
Vega.Schedule.one_cycle ~max_value:0.01 ~total_steps:10000 ()

(* Custom phase split: 40% warmup *)
Vega.Schedule.one_cycle ~max_value:0.01 ~total_steps:10000
  ~pct_start:0.4 ()
```

## Composition

### piecewise_constant

A step function. `values` has one more element than `boundaries`:

<!-- $MDX skip -->
```ocaml
Vega.Schedule.piecewise_constant
  ~boundaries:[1000; 5000]
  ~values:[0.01; 0.001; 0.0001]
(* steps 1–1000: 0.01, steps 1001–5000: 0.001, steps 5001+: 0.0001 *)
```

### join

Sequence multiple schedules end-to-end. Each `(n, schedule)` pair runs
`schedule` for `n` steps. Step numbers restart from 1 within each segment:

<!-- $MDX skip -->
```ocaml
Vega.Schedule.join [
  (1000, Vega.Schedule.linear ~init_value:0.0 ~end_value:0.001 ~steps:1000);
  (9000, Vega.Schedule.cosine_decay ~init_value:0.001 ~decay_steps:9000 ());
]
```

### Custom Schedules

Since `Schedule.t` is just `int -> float`, you can write arbitrary functions:

<!-- $MDX skip -->
```ocaml
(* Step decay: halve every 1000 steps *)
let step_decay : Vega.Schedule.t = fun step ->
  0.01 *. (0.5 ** float_of_int (step / 1000))
```

## Using Schedules with Optimizers

Schedules are passed to optimizer aliases as the last positional argument:

<!-- $MDX skip -->
```ocaml
let lr =
  Vega.Schedule.warmup_cosine_decay
    ~init_value:0.0 ~peak_value:1e-3
    ~warmup_steps:1000 ~decay_steps:9000 ()
in
let tx = Vega.adamw ~weight_decay:0.01 lr
```

When building from primitives, pass the schedule to `scale_by_learning_rate`:

<!-- $MDX skip -->
```ocaml
let tx =
  Vega.chain [
    Vega.scale_by_adam ();
    Vega.scale_by_learning_rate lr;
  ]
```

Other primitives accept schedules too. For instance, `add_decayed_weights`
takes a `~rate` schedule for dynamic weight decay:

<!-- $MDX skip -->
```ocaml
Vega.add_decayed_weights
  ~rate:(Vega.Schedule.cosine_decay ~init_value:0.01 ~decay_steps:10000 ())
  ()
```

## Next Steps

- [Composing Transforms](../02-composing-transforms/) — building custom optimizers from primitives
- [Getting Started](../01-getting-started/) — basic usage and optimizer aliases
- [Optax Comparison](../04-optax-comparison/) — mapping from Python's Optax to Vega
