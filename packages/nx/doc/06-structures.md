# Structures

A model, an optimizer state or a decoding cache is an ordinary OCaml record of arrays. `Nx.Ptree` lets code that knows nothing about your record walk its arrays: gradients, optimizer updates, saved files, dtype casts, placement and compiled functions all work on it once the record has a `walk`. This page shows how to write that function, how to use the structure it defines, and how to test it.

## A Record Is One Function

A structure is a module with a type `('a, 'd) t` and one function, `walk`, whose parameter `'a` marks where the arrays go. The second parameter, `'d`, is the device set of the record's arrays of a fixed type; a record whose every array is at `'a` ignores it. A dense layer holds a weight and an optional bias:

```ocaml
module Linear = struct
  type ('a, _) t = { w : 'a; b : 'a option }

  let walk c { w; b } =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let b = field c "b" (option leaf) b in
    { w; b }
end
```

`walk` visits each part of the value once with a cursor `c` and rebuilds the value from what each walk returns:

- `field c "w" walk x` walks `x` under the name `"w"`;
- `leaf` walks a position of the type's parameter;
- `option leaf` walks an optional part, and records whether it is present.

Every other operation of `Nx.Ptree` runs this function, so they all visit the parts in the order it gives. A record literal evaluates its fields in an order OCaml leaves unspecified, which is why the fields are walked in a sequence of `let`s.

`leaf` turns the parameter's type from the cursor's input to its output, so the compiler rejects a `walk` that forgets a field of type `'a`: the record it rebuilds would not have type `('b, _) t`.

## Using a Structure

`Nx.Ptree.instantiate` fixes the parameter to arrays and gives a description, a value that transformations, optimizers and saving take. A description is at one dtype and one device set, which its first use fixes:

```ocaml
let linear = Nx.Ptree.instantiate (module Linear)

let layer =
  { Linear.w = Nx.ones Nx.float32 [| 3; 2 |]; b = Some (Nx.zeros Nx.float32 [| 2 |]) }
```

`Nx.Ptree.map`, `map2` and `fold` pass every array its path. The function is polymorphic in the dtype, since the arrays of one value may have different dtypes:

```ocaml
# Nx.Ptree.fold linear
    (fun path t acc -> (Nx.Ptree.Path.to_string path, Nx.numel t) :: acc)
    layer [];;
- : (string * int) list = [("b", 2); ("w", 6)]
```

A path is a list of segments, `Field name` for a named part and `Index i` for a numbered one. `Path.to_string` joins them with `"."`, which is the name of the array in a saved file (see [Input/Output](04-io.md)).

## Nesting, Lists and Options

A record of layers walks each layer with the layer's own `walk`, so each path starts with the field name. `list` walks a list's elements at `Index 0`, `Index 1`, ... and records its length:

```ocaml
module Mlp = struct
  type ('a, 'd) t = { layers : ('a, 'd) Linear.t list; out : ('a, 'd) Linear.t }

  let walk c { layers; out } =
    let open Nx.Ptree.Walk in
    let layers = field c "layers" (list Linear.walk) layers in
    let out = field c "out" Linear.walk out in
    { layers; out }
end

let mlp = Nx.Ptree.instantiate (module Mlp)

let dense ?(bias = true) inputs outputs =
  {
    Linear.w = Nx.ones Nx.float32 [| inputs; outputs |];
    b = (if bias then Some (Nx.zeros Nx.float32 [| outputs |]) else None);
  }

let params =
  { Mlp.layers = [ dense 4 8; dense 8 8 ]; out = dense ~bias:false 8 2 }
```

Its arrays are at `layers.0.w`, `layers.0.b`, `layers.1.w`, `layers.1.b` and `out.w`.

## Masks and Plans by Path

A path's segments are a variant, so a per-array decision is a pattern match. Freezing the first layer zeroes its gradients:

```ocaml
let frozen path =
  match Nx.Ptree.Path.segments path with
  | Nx.Ptree.Path.Field "layers" :: Index 0 :: _ -> true
  | _ -> false

let masked grads =
  Nx.Ptree.map mlp (fun path g -> if frozen path then Nx.zeros_like g else g) grads
```

`Nx.Ptree.map2` zips two values of one structure. It raises when they differ, naming the first path where they do and what each value holds there:

```ocaml
# let no_bias = { params with Mlp.out = dense 8 2 } in
  Nx.Ptree.map2 mlp (fun _ a _ -> a) params no_bias;;
Exception:
Invalid_argument
 "Nx.Ptree.map2: out.b: None in the first value, Some in the second".
```

## Changing the Payload

`walk` also changes what the parameter's positions hold, which is how a model is cast and how metadata shaped like the model is built. These operations take the structure's module, since only a module names the type constructor:

```ocaml
let half = Nx.Ptree.cast (module Mlp) Nx.float16 params

let sizes = Nx.Ptree.Payload.map (module Mlp) (fun _ t -> Nx.numel t) params
(* sizes : (int, Nx.host) Mlp.t *)
```

`sizes` is a record of the model's type holding integers, so a per-array learning rate, a sharding plan or a dimension count is a value of the model's own type. A total over the arrays is a fold over them:

```ocaml
let total = Nx.Ptree.fold mlp (fun _ t acc -> Nx.numel t + acc) params 0
```

`Nx.Ptree.place` moves every array of a value to a placement on another device set, and changes the set in the value's type: `Nx.Ptree.place (module Mlp) Gpu.on params` for a set `Gpu` minted by `Nx.devices`.

## Fixed Arrays, Integers and Cases

A part that is an array of one fixed type, such as a step counter, is walked with `tensor`; its set is the type's second parameter. An integer that changes what a compiled program computes, such as an attention window, is reported with `int`, and a bool is reported as `0` or `1`:

```ocaml
module Block = struct
  type ('a, 'd) t = {
    proj : ('a, 'd) Linear.t;
    steps : 'd Nx.int32_t;
    window : int option;
  }

  let walk c { proj; steps; window } =
    let open Nx.Ptree.Walk in
    let proj = field c "proj" Linear.walk proj in
    let steps = field c "steps" tensor steps in
    let window = field c "window" (option int) window in
    { proj; steps; window }
end
```

`Nx.Ptree.map` and the transformations treat `steps` like any other array; `cast` and `Payload` keep it unchanged, and `place` places it. A compiled function keys its programs on every reported integer, so a block with another window compiles a program of its own.

A variant names its case with `case` before walking the case's parts, with a tag that no other case of the type uses:

```ocaml
module Weight = struct
  type ('a, _) t = Dense of 'a | Low_rank of { u : 'a; v : 'a }

  let walk c =
    let open Nx.Ptree.Walk in
    function
    | Dense w ->
        case c "dense";
        Dense (leaf c w)
    | Low_rank { u; v } ->
        case c "low_rank";
        let u = field c "u" leaf u in
        let v = field c "v" leaf v in
        Low_rank { u; v }
end
```

A dictionary reports its size with `int` and each key with `case` before walking that key's value with `field`.

## Pairs, Lists and Records Without Names

The combinators are descriptions at one type: `Nx.Ptree.tensor` and `unit`, and `pair`, `option` and `list`, which build one from others. A pair walks its sides at `0` and `1`:

```ocaml
# let pairs = Nx.Ptree.(pair tensor (list tensor)) in
  Nx.Ptree.fold pairs
    (fun path _ acc -> Nx.Ptree.Path.to_string path :: acc)
    (Nx.ones Nx.float32 [| 2 |], [ Nx.ones Nx.int32 [| 3 |] ])
    [];;
- : string list = ["1.0"; "0"]
```

## Records of Structures

A record of a fixed type, such as a training state saved as one file or the result of a compiled function, has no parameter. Its parts often have a structure at one type and no module of their own: an optimizer state over a model, a cache index, or a list of models. `structure` walks such a part at the cursor's path, and `Nx.Ptree.v` makes the record's description from its walk:

```ocaml
type train = {
  params : (Nx.host Nx.float32_t, Nx.host) Mlp.t;
  snapshots : (Nx.host Nx.float32_t, Nx.host) Mlp.t list;
}

let train =
  Nx.Ptree.v (fun c t ->
      let open Nx.Ptree.Walk in
      let params = field c "params" (structure mlp) t.params in
      let snapshots =
        field c "snapshots" (structure (Nx.Ptree.list mlp)) t.snapshots
      in
      { params; snapshots })
```

Its arrays are at `params.layers.0.w`, ..., `snapshots.0.layers.0.w`, ... A description is at one device set, so a walk that calls `structure` is too, and `Nx.Ptree.place` cannot move what it walks. A record that moves between sets is a module whose walk walks each part with the part's own walk, as `Nx.Rng.walk` walks a key.

## Testing a Structure with visits

The compiler checks the parameter's positions and nothing else: a `walk` that returns a fixed array without walking it, forgets to report an integer or a case, or reuses a case tag still compiles. A compiled function then freezes the array into its first program, or replays a program traced for other data. `Nx.Ptree.visits` lists what `walk` visits, each array and each report with its path, and a test compares it with what the structure should walk:

```ocaml
let visits s x = List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) (Nx.Ptree.visits s x)

let block =
  { Block.proj = dense 2 2; steps = Nx.zeros Nx.int32 [||]; window = Some 4 }
```

```ocaml
# List.iter print_endline (visits (Nx.Ptree.instantiate (module Block)) block);;
proj.w: a leaf
proj.b: Some
proj.b: a leaf
steps: a leaf
window: Some
window: int 4
- : unit = ()
```

A structure's test checks these lines for each shape its values take: a present and an absent option, each case of a variant, an empty and a non-empty list. It also checks the round trip: `Nx.Ptree.rebuild s ~like:x (fst (Nx.Ptree.flatten s x))` is `x`.

## Next Steps

- [Rune's transformations](../../rune/doc/02-transformations.md) take structures, and `Rune.jit` takes a signature built from them, such as `Nx.Ptree.(tensor @-> caches @-> returns (pair tensor caches))`. A caller gives up an argument's arrays by donating them at the call, `Nx.Ptree.donate caches c`.
- [Kaun's layers](../../kaun/doc/02-layers-and-models.md) each have a `walk`, and a model's is one line per field.
