# LLM Context for Rune Project (LLM.md)

This document provides context for Large Language Models (LLMs) to assist with the development of the Rune OCaml library.

## 1. Overview of Rune

Rune is an experimental array computation library for OCaml. Its primary goals are:

*   **High Performance:** Efficient execution of array operations.
*   **Automatic Differentiation (AD):** Support for `grad` to compute gradients of functions.
*   **Just-In-Time (JIT) Compilation:** Support for `jit` to compile computation graphs for optimization.

Rune is heavily inspired by libraries like JAX (API, `grad`, `jit` transformations) and Tinygrad (minimalist UOp-based backend design). It builds upon the **Nx** OCaml library as its foundation.

## 2. Core Architectural Principles

### 2.1. Nx as Foundation
*   Rune uses **Nx** (specifically `Nx_core`) for its user-facing NumPy-like API.
*   Nx features a functor-based design (`Nx_core.Make_frontend`) that takes a backend module implementing the `Nx_core.Backend_intf.S` signature. This signature defines a set of low-level primitive operations (UOps).
*   **Crucially, Rune itself (`Nx_rune` module) implements the `Nx_core.Backend_intf.S` signature.** This allows Rune to intercept all Nx operations.

### 2.2. Algebraic Effects for Transformations
*   OCaml 5's algebraic effects are the core mechanism enabling Rune's `grad` and `jit` capabilities non-intrusively.
*   When an Nx operation (e.g., `Nx.add`) is called on a Rune tensor, the `Nx_rune` backend performs a corresponding algebraic effect (e.g., `E_add`) instead of executing the computation directly.
*   Specialized effect handlers in `Autodiff.ml` (for `grad`) and `Jit.ml` (for `jit`) catch these effects to build gradient tapes or computation graphs.

### 2.3. Universal Operations (UOps)
*   Rune aims for a minimal set of primitive operations (UOps) that actual hardware backends need to implement, similar to Tinygrad's approach.
*   There's a direct mapping:
    1.  Algebraic effects defined in `Nx_rune.ml` (e.g., `Effect.E_add`).
    2.  Functions in the `Nx_core.Backend_intf.S` signature (e.g., `op_add`).
    3.  Tinygrad's UOps (e.g., `Ops.ADD`).
*   The JIT compiler translates a computation graph into these UOps for a target device backend.

### 2.4. Layered Architecture & Control Flow
The typical control flow is:
1.  **User Code:** User writes standard Nx-like code (e.g., `Tensor.add a b`).
2.  **Nx Frontend Dispatch:** The call goes to `Nx_core.Make_frontend(Nx_rune)`, which might perform broadcasting logic (inserting `op_expand`, `op_reshape` calls, which also go through `Nx_rune`).
3.  **Rune Backend (`Nx_rune`):** The `Nx_rune.op_add` function is called.
4.  **Effect Performance:** `Nx_rune.op_add` performs an effect (e.g., `Effect.perform (E_add (a, b, out_placeholder))`).
5.  **Handler Interception (if active):**
    *   **`jit` handler (`Jit.ml`):** Catches the effect, records it as a node in a computation graph, and returns a symbolic tensor. The operation is *not* executed.
    *   **`grad` handler (`Autodiff.ml`):** Catches the effect.
        *   It continues the computation (potentially re-raising the effect for JIT or falling to eager).
        *   It records the operation and its inputs/outputs on a gradient tape.
        *   During the backward pass (effect unwinding), it uses the tape and chain rule to compute gradients.
6.  **Eager Execution Fallback (if no handler):**
    *   If an effect from `Nx_rune` is unhandled (e.g., not inside a `grad` or `jit` block), a `try...with Effect.Unhandled` block in `Nx_rune` catches it.
    *   `Nx_rune` then unwraps its tensor/context types to their concrete `Nx_native` counterparts and calls the corresponding `Nx_native` operation (e.g., `Nx_native.op_add`) for immediate execution.
    *   The result from `Nx_native` is re-wrapped into an `Nx_rune.t` and returned.

## 3. Key Modules and Their Roles

*   **`DESIGN.md`**:
    *   The main design document. Essential reading for understanding high-level goals, concepts, and architecture.

*   **`nx/core/`**: The foundation library.
    *   `backend_intf.ml (Nx_core.Backend_intf.S)`: Defines the low-level UOp interface (e.g., `op_add`, `op_load`, `op_buffer`, `op_reshape`, `op_expand`, `op_sum`, plus JIT-specific ops like `op_define_global`, `op_range`, `op_special`). This is the contract for all backends.
    *   `dtype.ml (Nx_core.Dtype)`: GADT-based tensor data types (e.g., `Float32`, `Int32`).
    *   `view.ml (Nx_core.View)`: Defines tensor views (`shape`, `strides`, `offset`, `mask`, `layout`). Enables lazy movement operations without data copying.
    *   `frontend.ml (Nx_core.Frontend.Make)`: A functor that takes a module implementing `Backend_intf.S` and produces a high-level, NumPy-like API module. This API module typically handles broadcasting logic.
    *   `nx_core.ml`: Exposes modules from `nx/core/`.

*   **`nx/native/`**: A concrete CPU backend.
    *   `nx_native.ml (Nx_native)`: Implements `Nx_core.Backend_intf.S` for direct, eager execution on CPU using OCaml's `Bigarray`.
    *   `internal.ml (Nx_native.Internal)`: Defines `Nx_native.t` (tensor type holding a `Bigarray.Array1.t`) and some non-parallel internal operations.
    *   `ops_binary.ml`, `ops_reduce.ml`: Contain computation kernels (e.g., for add, sum) for `Nx_native`, parallelized using `Parallel.ml`.
    *   `parallel.ml (Nx_native.Parallel)`: Custom domain-based parallelism for `Nx_native` CPU operations.

*   **`rune/lib/`**: The Rune library itself.
    *   `nx_rune.ml (Nx_rune)`:
        *   **Core of Rune's transformation system.** Implements `Nx_core.Backend_intf.S`.
        *   Its functions (e.g., `op_add`, `op_reshape`) perform algebraic effects (`E_add`, `E_reshape`, etc.) corresponding to the UOps.
        *   Handles eager fallback to `Nx_native` if effects are unhandled.
        *   Defines `('a, 'b) Nx_rune.t`: Rune's tensor type, wrapping a concrete backend buffer (e.g., `Cpu_buffer of ('a, 'b) Nx_native.buffer`) and a `Nx_core.View.t`.
        *   Defines `Nx_rune.context`: Wraps a concrete backend context (e.g., `Cpu_context of Nx_native.context`).
    *   `autodiff.ml (Rune_next.Autodiff)`:
        *   Implements the `grad` effect handler (`make_reverse_handler`).
        *   Catches UOp effects, builds a gradient tape (using `t_with_grad` to pair forward values with gradients), and computes gradients via reverse-mode AD.
        *   The `retc` part of the handler seeds the output gradient with `1`.
    *   `jit.ml (Rune_next.Jit)`:
        *   Implements the `jit` effect handler (`make_jit_handler`).
        *   Catches UOp effects to build a `jit_graph`.
        *   Graph nodes are `_ jit_op_t` (a GADT for UOps like `Op_Add`, `Op_Buffer`).
        *   Symbolic tensors in the graph are represented by `Var.t` (integer IDs) with associated `var_metadata` (dtype, shape).
    *   `tensor.ml (Rune_next.Tensor)`:
        *   The main user-facing API for Rune (e.g., `Tensor.add`, `Tensor.zeros`).
        *   Generated by `Nx_core.Make_frontend(Nx_rune)`.
    *   `rune.ml (Rune_next.Rune)`: Re-exports `Tensor` to provide the `Rune.add`, etc. API.

## 4. Important Data Structures

*   **`('a, 'b) Nx_core.Dtype.t`**: (GADT) Represents tensor data types (e.g., `Float32`, `Int64`). Defined in `nx/core/dtype.ml`.
*   **`Nx_core.View.t`**: A record `{ shape: int array; strides: int array; offset: int; mask: (int * int) array option; layout: layout_type }`. Represents the logical view of tensor data, enabling lazy operations like reshape, expand, slice without copying data. Defined in `nx/core/view.ml`.
*   **`('a, 'b) Nx_native.t`**: The concrete tensor type for the `Nx_native` (CPU) backend. Contains `dtype`, a `Bigarray.Array1.t` buffer, and a `View.t`. Defined in `nx/native/internal.ml` and `nx/native/nx_native.ml`.
*   **`('a, 'b) Nx_rune.t`**: Rune's primary tensor type. A record `{ dtype: ('a,'b) Dtype.t; buffer: ('a,'b) Nx_rune.buffer; view: View.t }`.
    *   `Nx_rune.buffer` is a GADT like `| Cpu_buffer of ('a, 'b) Nx_native.buffer (* | Gpu_buffer of ... *)`. This indicates which concrete backend owns the data.
    *   Defined in `rune/lib/nx_rune.ml`.
*   **`Nx_rune.context`**: Rune's context type, a GADT wrapping concrete backend contexts (e.g., `Cpu_context of Nx_native.context`). Defined in `rune/lib/nx_rune.ml`.
*   **`Nx_rune.E_...` effects**: Algebraic effects (e.g., `E_add`, `E_reshape`) defined in `rune/lib/nx_rune.ml`. These mirror the operations in `Nx_core.Backend_intf.S`.
*   **`('a, 'b) Autodiff.t_with_grad`**: A record used by the `Autodiff` handler to store a tensor's forward value (`v: ('a,'b) Nx_rune.t`) and its accumulated gradient (`bv: ('a,'b) Nx_rune.t`), along with a unique `id`. Defined in `rune/lib/autodiff.ml`.
*   **`Jit.jit_graph`**: The JIT computation graph. A record containing:
    *   `ops: Jit.any_jit_op list`: A list of operations in the graph. `any_jit_op` wraps `_ Jit.jit_op_t`.
    *   `_ Jit.jit_op_t`: A GADT representing UOps in the graph (e.g., `Op_Buffer`, `Op_Add`, `Op_Reshape`). Operands are `Jit.Var.t`.
    *   `vars_metadata: (Jit.Var.t, Jit.var_metadata) Hashtbl.t`: Maps symbolic variable IDs to their dtype and shape.
    *   `input_vars: Jit.Var.t list` and `output_vars: Jit.Var.t list`.
    *   Defined in `rune/lib/jit.ml`.

## 5. Style Guide

*   **Simplicity and Clarity First:** Prioritize code that is easy to read, understand, and maintain. Leverage OCaml's strong type system to enhance clarity and correctness.
*   **Avoid `Obj.magic`:** **Refrain from using `Obj.magic` as much as possible.** Its use is a strong indicator of a potential type system flaw, an overly complex workaround, or a place where a GADT or other type-safe mechanism might be more appropriate. If `Obj.magic` seems necessary, critically re-evaluate the design.
*   **Judicious Use of Advanced OCaml Features:**
    *   **GADTs (Generalized Algebraic Data Types):** Use GADTs when they provide clear benefits for type safety, especially when dealing with heterogeneous data that shares a common structure or enforcing type-level constraints (e.g., `Nx_core.Dtype.t`, `Jit.jit_op_t`, `Nx_rune.buffer`). Do not use them if simpler variants or records suffice.
    *   **Functors:** Employ functors for building modular and reusable components by parameterizing modules over well-defined interfaces (signatures), as seen with `Nx_core.Make_frontend`.
    *   **Algebraic Effects:** Effects are fundamental to Rune's non-intrusive `grad` and `jit` mechanisms. Their use here is justified and core to the design. When defining new effects, ensure they are well-scoped and truly necessary.
*   **Immutability:** Prefer immutable data structures. Mutable state (e.g., counters, caches in handlers, `Bigarray` buffers) should be carefully managed, localized, and its scope minimized.
*   **Error Handling:** Use OCaml's exception system for genuinely exceptional circumstances. For expected failures like invalid user input (e.g., incompatible shapes), prefer specific error types or `Invalid_argument`.
*   **Modularity:** Design modules with a clear, single responsibility. Define clean and minimal interfaces in `.mli` files.
*   **Naming Conventions:** Adhere to standard OCaml naming conventions (e.g., `snake_case` for values, functions, and module value names; `CamelCase` for types and module/signature names).
*   **Comments and Documentation:**
    *   Write comments to explain non-obvious logic, complex design choices, or `TODO` items.
    *   Use docstrings (`(** ... *)`) for all public API elements (functions, types, module interfaces) to facilitate understanding and usage.

This context should help LLMs understand the Rune project's structure, design philosophy, and coding style when assisting with development tasks.