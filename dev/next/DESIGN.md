# Rune Design Document

## 1. Overview

Rune is an experimental array computation library for OCaml, designed for performance, automatic differentiation, and Just-In-Time (JIT) compilation. It draws heavy inspiration from libraries like JAX and Tinygrad, building upon the Nx library as its foundation.

**Core Goals:**

*   Provide a familiar, high-level array programming interface (leveraging Nx).
*   Enable automatic differentiation (`grad`).
*   Support JIT compilation for optimizing computation graphs (`jit`).
*   Utilize OCaml 5's algebraic effects for implementing `grad` and `jit` handlers non-intrusively.
*   Maintain a minimal backend interface, closely aligned with Tinygrad's Universal Operations (UOps), simplifying the addition of new hardware backends (CPU, GPU, etc.).
*   Support multi-device computation orchestration.

**Key Inspirations:**

*   **JAX:** High-level API, `grad`, and `jit` transformations.
*   **Tinygrad:** Minimalist backend UOp design, focus on a small, powerful core instruction set for hardware.
*   **Nx:** Underlying array library providing the user-facing API and a pluggable backend system.

## 2. Core Concepts

### 2.1. Nx - The Foundation (Formerly Ndarray)

Rune builds upon **Nx** (previously named Ndarray in earlier designs), an OCaml library providing a NumPy-like interface for multi-dimensional arrays. Key aspects of Nx relevant to Rune are:

*   **Functor-Based Design (`Nx_core`):** Nx defines a core module type `Nx_core.Backend_intf.S`. This signature specifies the required primitive operations (like `add`, `reshape`, `buffer`, `const`) and a backend-specific `context` type. A functor `Nx_core.Make_backend` takes a module implementing `Backend_intf.S` and produces a high-level, NumPy-like API module where operations typically take an explicit `context`.
*   **Pluggable Backends:** Different computational backends (CPU, GPU, etc.) can implement the `Nx_core.Backend_intf.S` interface. Nx operations are dispatched to the currently active backend associated with a tensor or context.
*   **User-Facing API:** Rune users interact primarily with the Nx API (e.g., `Nx.add`, `Nx.reshape`), making the experience similar to using NumPy or PyTorch tensors.

Crucially, **Rune itself implements the Nx backend interface** (`Nx_rune.ml`). This allows Rune to intercept Nx operations using algebraic effects, rather than directly executing them on a concrete device backend.

### 2.2. Algebraic Effects

OCaml 5's algebraic effects are the core mechanism enabling Rune's `grad` and `jit` capabilities without modifying the upstream Nx library or requiring complex user-side code transformations.

*   **Effect Definition:** Rune defines effects corresponding to the fundamental UOps required for array computation (e.g., `Add`, `Mul`, `Reshape`, `Expand`, `Reduce`, `Const`, `Buffer` as seen in `Nx_rune.ml`). These effects carry the necessary information about the operation and its Rune tensor operands/arguments.
*   **Raising Effects:** Instead of directly calling a concrete backend function (like `Nx_native.add`), the Rune Nx backend implementation (`Nx_rune.ml`) performs the corresponding effect (e.g., `Effect.perform (Add (a, b, out))`).
*   **Handling Effects:** Specialized effect handlers (`Autodiff.ml`, `Jit.ml`) can catch these effects to perform specific actions like building a gradient tape or constructing a computation graph.

### 2.3. Effect Handlers (`grad` and `jit`)

Handlers provide context for interpreting the effects raised by the Rune backend.

*   **`grad` Handler:**
    *   Catches computation effects.
    *   Builds a computation trace (gradient tape) recording the operations and their inputs/outputs.
    *   Uses this trace and the chain rule (similar to Tinygrad's `gradient.py`) to compute gradients when `backward` is called.
    *   Continues the original computation, potentially by re-raising the effect to be caught by another handler (like `jit`) or falling back to eager execution. Enables composable effects, e.g., nested `grad` for higher-order derivatives.
*   **`jit` Handler:**
    *   Catches computation effects.
    *   **Does not execute the operation immediately.**
    *   Constructs a computation graph where nodes represent UOps and edges represent data dependencies (tensors/buffers). This graph is analogous to Tinygrad's UOp graph.
    *   This graph is then available for optimization passes and compilation to efficient code for a target backend (CPU, GPU, etc.).

### 2.4. Universal Operations (UOps) & Backend Interface

Inspired by Tinygrad, Rune aims for an extremely minimal set of primitive operations that actual hardware backends need to implement.

*   **Minimalism:** The goal is to express all high-level array operations in terms of a small, fixed set of UOps (e.g., `LOAD`, `STORE`, `ALU` operations like `ADD`, `MUL`, `CMPLT`, movement ops like `RESHAPE`, `EXPAND`, `REDUCE`).
*   **Mapping:** There is a direct, one-to-one mapping between:
    1.  The algebraic effects defined in `Nx_rune.ml` (e.g., `Add`).
    2.  The functions defined in the abstract `Nx_core.Backend_intf.S` (e.g., `add`).
    3.  The primitive operations in Tinygrad's UOp set (`Ops.py`) (e.g., `Ops.ADD`).
*   **Target Backend:** The JIT compiler's target is code generation based on these UOps for a specific device backend.

### 2.5. Eager Execution Fallback

If an effect raised by `Nx_rune.ml` is *not* handled by any active handler (`grad`, `jit`, or others), it signifies eager execution mode.

*   **Mechanism:** The `try...with Effect.Unhandled` construct in `Nx_rune.ml` catches the unhandled effect.
*   **Action:** Inside the `with` block, Rune unwraps the concrete backend buffer/context associated with the input Rune tensors (e.g., extracts the `Nx_native.buffer` from `Nx_rune.Cpu_buffer`) and dispatches the operation to the appropriate concrete backend implementation (e.g., `Nx_native.add`).
*   **Default Behavior:** This makes Rune usable just like standard Nx when not wrapped in a `grad` or `jit` handler.

## 3. Architecture Details

### 3.1. Control Flow

1.  **User Code:** User writes standard Nx code (e.g., `let c = Nx.add a b`).
2.  **Nx Dispatch:** Nx calls the corresponding function in its configured backend, which is Rune (`Nx_rune.add`).
3.  **Effect Raise:** `Nx_rune.add` performs the `Add` effect: `Effect.perform (Add (a_rune, b_rune, out_rune))`.
4.  **Handler Interception (if active):**
    *   **`jit` handler:** Catches `Add`. Creates a `UOp.ADD` node in the computation graph, connecting inputs `a` and `b` to output `out`. Returns a placeholder/future representing `out`.
    *   **`grad` handler:** Catches `Add`. Records the operation (`Add`, inputs `a`, `b`, output `out`) in a gradient tape. It then *continues* the computation, potentially re-raising the `Add` effect (if nested under `jit`) or triggering eager execution.
5.  **Eager Fallback (if no handler):** The `try...with Effect.Unhandled` in `Nx_rune.add` catches the `Add` effect. It unwraps the concrete buffers/contexts from the `Nx_rune.t` arguments and calls the underlying concrete backend (e.g., `Nx_native.add`) to perform the addition immediately.
6.  **Result:** The appropriate result (a direct computation result in eager mode, a future/placeholder in `jit` mode) is returned to the user code.

```mermaid
graph TD
    subgraph User Code
        A[Nx.add a b]
    end
    subgraph Nx Core
        B(Nx Dispatch to Rune Backend)
    end
    subgraph Rune Backend (Nx_rune.ml)
        C{Effect.perform Add}
    end
    subgraph Effect Handling
        D{Handler Active?}
        E[Jit Handler: Build UOp Graph] -- Creates Node --> F[UOp Graph]
        G[Grad Handler: Record Op] -- Records --> H[Gradient Tape]
        I[Eager Fallback: try...with Unhandled] -- Unwraps & Calls --> J[Concrete Backend (e.g., Nx_native.add)]
    end
    subgraph Concrete Backend (e.g. Nx_native.ml)
        K[Execute Add Operation]
    end
    subgraph Result
        L(Return Result/Placeholder)
    end

    A --> B --> C --> D;
    D -- Yes --> E;
    D -- Yes --> G;
    D -- No --> I;
    E --> L;
    G -- Continues Computation --> C; // Or falls through to I if no other handler
    I --> J --> K --> L;
```

### 3.2. Data Structures

*   **`('a, 'b) Nx_rune.t`:** The main Rune tensor type visible to handlers and effect-raising functions. It implements the `Nx_core.Backend_intf.S` signature *conceptually*. It contains:
    *   `dtype`: The element data type (`Nx_core.Dtype.t`).
    *   `buffer`: A wrapped concrete buffer (GADT `('a, 'b) Nx_rune.buffer`), e.g., `Cpu_buffer of ('a, 'b) Nx_native.buffer`. This indicates which concrete backend owns the data.
    *   `view`: The logical view (`Nx_core.View.view`) of the tensor data, enabling lazy movement operations without copying data.
*   **`('a, 'b) Nx_native.t` (Example Concrete Tensor):** The tensor type for the native CPU backend. Contains `dtype`, the actual `Bigarray.Array1` buffer, and a `View.view`. Rune tensors with `Cpu_buffer` wrap this type's buffer.
*   **`Nx_rune.context`:** A wrapper for concrete backend contexts (e.g., `Cpu_context of Nx_native.context`). This context holds backend-specific resources, like the CPU worker pool in `Nx_native`.
*   **`Nx_core.View.view`:** Represents the logical view (shape, strides, offset, mask) of the tensor data, identical in concept to Tinygrad's `View`. Enables lazy movement operations.
*   **Computation Graph (Jit):** A directed acyclic graph (DAG) where nodes are UOps (conceptually similar to `tinygrad.ops.UOp`) and edges represent tensor data flow. Built by the `jit` handler from intercepted effects.
*   **Gradient Tape (Grad):** A data structure (likely a list or graph) storing the sequence of operations performed (represented by effects), their inputs, outputs, and potentially gradient functions, built by the `grad` handler.

### 3.3. Backend Implementations

*   **`Nx_native` (CPU Backend):**
    *   Implements the `Nx_core.Backend_intf.S` interface for CPU execution.
    *   Uses OCaml's `Bigarray.Array1` for storing tensor data.
    *   Features a custom domain pool manager (`Parallel.ml`) for parallelizing computations across CPU domains. This was chosen over standard libraries like `Domainslib` potentially to minimize intermediate allocations and gain finer control over task scheduling for numerical workloads.
    *   Operations directly manipulate `Bigarray` memory based on the tensor's `View`.
*   **Future Backends (e.g., GPU):**
    *   Would also implement `Nx_core.Backend_intf.S`.
    *   Would define their own buffer and context types (e.g., `Gpu_buffer`, `Gpu_context`).
    *   `Nx_rune.t` and `Nx_rune.context` would be extended with variants for these new backends (e.g., `| Gpu_buffer of ...`).
    *   The JIT compiler would target the UOp instruction set supported by the specific GPU backend.
