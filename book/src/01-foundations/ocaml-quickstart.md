# OCaml Quickstart for ML Engineers

If you come from Python or Julia, OCaml’s syntax and type system can feel unfamiliar.  
This crash course highlights the language features we rely on throughout the book.

## Records and Variants

Records and variants let us model structured data with compile-time guarantees.

```ocaml
type optimizer =
  | SGD of { lr : float }
  | Adam of { lr : float; beta1 : float; beta2 : float }

let default = Adam { lr = 1e-3; beta1 = 0.9; beta2 = 0.999 }
```

Pattern matching keeps decision logic explicit:

```ocaml
let step opt gradients params =
  match opt with
  | SGD { lr } -> Rune.sub params (Rune.mul_s gradients lr)
  | Adam { lr; _ } -> Rune.sub params (Rune.mul_s gradients lr)
```

## Pipe Operators and Composition

The pipeline operator `|>` makes dataflow readable—perfect for preprocessing chains.

```ocaml
let process_text tokenizer text =
  text
  |> String.lowercase_ascii
  |> Saga.Tokenizer.encode tokenizer
  |> Saga.Encoding.get_ids
```

## Modules and Functors

OCaml’s module system lets us parameterize code over implementations.  
You’ll see this when we swap Nx backends or inject RNG sources.

```ocaml
module type BACKEND = sig
  val name : string
  val device_count : unit -> int
end

module Trainer (B : BACKEND) = struct
  let init () = Printf.printf "Running on %s\n" (B.name)
end
```

## Immutability with Efficient Updates

OCaml encourages immutable data, yet provides `Array`, `Bigarray`, and `Nx` tensors that manipulate large datasets efficiently.

```ocaml
let replace_last arr value =
  let copy = Array.copy arr in
  let idx = Array.length arr - 1 in
  copy.(idx) <- value;
  copy
```

When you want controlled mutation for performance, use scoped `ref`s or record fields; the type system keeps those effects contained.

## Working in the REPL

Launch `utop` or `ocaml` and load libraries with `#require`.

```ocaml
# #require "nx";;
# Nx.full Nx.float32 [|2; 2|] 1.0;;
- : (float, Bigarray.float32_elt) Nx.t = <abstr>
```

In the chapters ahead we rely on `mdx` to keep code examples executable.  
If you understand the constructs above, you are ready to follow the Raven APIs without friction.
