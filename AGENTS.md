# Coding Guide

*A pocket reference for humans and LLMs editing the codebase.*

---

## 1 · Core Principles

* **Clarity over cleverness** — favor readable solutions and only break these rules with good justification.
* **Unix mindset** – do one thing well, compose with other simple pieces, and fail loudly when invariants break.
* Comments explain **why** or the non‑obvious **how**; never restate code.
* Public docs are user‑facing; implementation comments are teammate‑facing.
* Embrace change: design for today, not hypothetical futures.

---

## 2 · Comments

> *Write comments like a good Unix `man` page: the synopsis first, details only when essential.*

* **File preamble** – A single‑sentence purpose line, then optional paragraphs on invariants or context.
* **Section separators** – Use only when they aid navigation:

  ```ocaml
  (* ────────── tensor ops ────────── *)
  ```
* **Local helpers** – Keep functions short; place helpers directly above their caller.
* **Tone** – No fluff. State facts, constraints, and assumptions succinctly.

---

## 3 · Documentation

### 3.1 Unix‑y Docstrings

* **First line = summary.** The very first sentence tells what the value does—no more than 72 chars.
* **Next lines = essentials.** List parameters, returns, side‑effects, error conditions.
* **Examples over prose.** Provide a minimal example whenever clarity beats words.
* **Everything needed, nothing else.** If readers still need the source to use the API, the docstring is incomplete.

### 3.2 Implementation Files (`.ml`)

* Follow the comment guidelines above.
* Lean on type inference; add annotations only when the compiler or clarity demands them.

### 3.3 Interface Files (`.mli`)

* **One `.mli` per `.ml`** – the only exception is files that define *only* types.
* **No `.mli`‑only modules** – they cannot define exceptions and add little value.
* **Top‑level module doc** – The Unix rule applies: start with a one‑line purpose, then essentials.
* **Keep interfaces short & sweet** – fewer exposed names make the API easier to learn and change.

| Visibility  | Doc style                                                                                       |
| ----------- | ----------------------------------------------------------------------------------------------- |
| **Public**  | Rich ocamldoc; lead with a terse summary then `{2 Parameters}` / `{2 Returns}` / `{2 Examples}` |
| **Private** | One‑liner summary                                                                               |

#### Example (public)

```ocaml
val init : ('a, 'b) dtype -> int array -> (int array -> 'a) -> ('a, 'b) t
(** [init dtype shape f].

    Initialises a tensor of [dtype] and [shape] using [f].

    {2 Parameters}
    - [dtype]: element data type
    - [shape]: array specifying dimensions
    - [f]: function mapping multi‑index to element value
    
    {2 Returns}
    - A new tensor where element [idx] is [f idx]

    {2 Examples}
    {[
      let t = init float32 [|2;2|] (fun [|i; j|] -> float_of_int (i + j))
    ]} *)
```

---

## 4 · Code Style

### 4.1 Parameters & Signatures

* Signatures should be self‑descriptive.
* **Label parameters** when types alone are not clear:

  ```ocaml
  (* Bad *)
  val display_name : string -> string -> _ Pp.t
  (* Good *)
  val display_name : first_name:string -> last_name:string -> _ Pp.t
  ```

### 4.2 Naming

* Modules & Variants: `My_module`, `My_variant`.
* Everything else: `snake_case`.
* Avoid meaningless names (`x`, `a`, `b`, `f`)—choose descriptive identifiers or inline them.

### 4.3 Types & Aliases

* Prefer `module Foo : sig type t … end` over a free‑floating `type foo`.
* If a module `Foo` has a module type `Foo.S`, place it once in `foo_intf.ml` to avoid duplication.

---

## 5 · Error Handling

* Error messages start with the function name and clearly describe the violated invariant.
* “Fail fast, fail loud”.
* When ignoring a binding: `let (_ : t) = …` so important results don’t vanish silently.

---

*Happy hacking*
