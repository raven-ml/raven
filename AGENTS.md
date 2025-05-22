# coding guide

*A pocket reference for humans and LLMs editing the codebase.*

---

## 1 · core principles

* clarity over cleverness.
* comments explain **why** or non‑obvious **how**—never restate code.
* public docs are user‑facing; implementation comments are teammate‑facing.

## 2 · implementation files (`.ml`)

* add section separators **only** when they aid navigation:

  ```ocaml
  (* ────────── tensor ops ────────── *)
  ```
* keep functions short; place helper functions right above their caller.

## 3 · interface files (`.mli`)

| visibility  | docstyle                                                             |
| ----------- | -------------------------------------------------------------------- |
| **public**  | rich ocamldoc with `{2 Parameters}` / `{2 Returns}` / `{2 Examples}` |
| **private** | one‑liner summary                                                    |

*example (public):*

```ocaml
val init : ('a, 'b) dtype -> int array -> (int array -> 'a) -> ('a, 'b) t
(** [init dtype shape f].

    Creates a new tensor of type [dtype] and shape [shape], where each element
    at index [idx] is initialized by [f idx].

    {2 Parameters}
    - [dtype]: element data type
    - [shape]: array specifying dimensions
    - [f]: function mapping multi-dimensional index to element value

    {2 Returns}
    - a fresh tensor whose element at index [idx] is [f idx]

    {2 Examples}
    {[
      let t = init float32 [|2;2|] (fun [|i; j|] -> float_of_int (i + j)) in
      (* t = [[0.;1.];[1.;2.]] *)
    ]} *)
```

## 4 · naming

* `My_module`, `My_variant`; everything else `snake_case`.

## 5 · types & safety

* avoid explicit annotations unless the compiler or clarity demands them.
* `Obj.magic` is a last resort; wrap and document if unavoidable.

## 6 · errors

* start messages with the function name and make them actionable:

  ```ocaml
  invalid_arg "reshape: incompatible dimensions"
  ```

---

*happy hacking!*
