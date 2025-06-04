# Task: Implement Lazy View Operations and Symbolic Shapes in nx

**IMPORTANT: This checklist must be kept updated throughout implementation**

- [x] Step 1: Add symbolic dimension support to nx core types
- [x] Step 2: Implement Shape_tracker with symbolic shape support
- [x] Step 3: Update backend interface to use Shape_tracker instead of View
- [x] Step 4: Make view operations lazy in frontend
- [ ] Step 5: Update backends to handle lazy views
- [ ] Step 6: Integrate symbolic shapes with Rune effect system
- [ ] Step 7: Update JIT lowering for view realization and symbolic shapes
- [ ] Step 8: Implement view fusion and shape specialization
- [ ] Step 9: Add comprehensive tests
- [ ] Step 10: Update documentation

---

## Objective

Implement lazy view operations and symbolic shapes in nx to:
1. Avoid unnecessary memory allocations and data copies (lazy views)
2. Enable shape-polymorphic kernels and dynamic batching (symbolic shapes)
3. Improve convolution performance through better memory access patterns

## Context

- Current nx has a View system but materializes views eagerly
- All shapes are currently concrete integers - no symbolic support
- Rune's JIT has skeletal symbolic infrastructure (SymVar.t) but it's not integrated
- Tinygrad provides a proven model for both lazy views and symbolic shapes
- Shape_tracker will replace the current View in tensor metadata (addressing the confusion about having both)

## Key Design Decisions

1. **Shape_tracker replaces View**: Instead of having both `view` and `shape_tracker` in tensor metadata, Shape_tracker becomes the single source of truth. It can represent both simple views (one View.t) and complex view chains (multiple View.t).

2. **op_contiguous vs op_realize_view**: We keep `op_contiguous` as the backend operation. The name better matches the semantics - making data contiguous in memory.

3. **Backend-driven realization**: View realization happens in the backend when operations need to access data. The frontend remains agnostic about when realization occurs.

4. **Symbolic shapes throughout**: Replace `int array` with a unified shape type that supports both concrete and symbolic dimensions.

## Implementation Steps

### 1. Add symbolic dimension support (nx/lib/core/)

Create new module for symbolic shapes:

**symbolic.ml/mli**:
```ocaml
type dim = 
  | Concrete of int
  | Symbolic of SymVar.t

and SymVar.t = {
  name: string;
  min_bound: int;
  max_bound: int;
  mutable value: int option;
}

type shape = dim array

(* Constructors *)
val concrete : int -> dim
val symbolic : string -> min:int -> max:int -> dim

(* Operations *)
val bind : dim -> int -> unit
val (+) : dim -> dim -> dim
val (*) : dim -> dim -> dim
val (/) : dim -> dim -> dim
val (mod) : dim -> dim -> dim

(* Utilities *)
val to_concrete : shape -> int array option
val is_concrete : shape -> bool
val evaluate : dim -> int option
val substitute : shape -> (string * int) list -> shape
```

### 2. Update View.t for symbolic shapes (nx/lib/core/view.ml)

```ocaml
type t = {
  shape : Symbolic.shape;
  strides : Symbolic.dim array;  (* Computed from shape, not independently symbolic *)
  offset : Symbolic.dim;  (* Can be symbolic from slicing operations *)
  mask : (Symbolic.dim * Symbolic.dim) array option;
  layout : layout;
}
```

**Critical design point**: Strides are NOT independently symbolic - they are always computed from shapes using the standard row-major formula. However, when shapes contain symbolic dimensions, the computed strides will also be symbolic expressions.

For example:
- Shape: [batch_size, 128, 64] where batch_size is symbolic
- Strides: [128*64, 64, 1] = [8192, 64, 1] where the first stride is a symbolic expression

Update View functions:
- `strides_for_shape`: Compute strides from shape using accumulation formula
- `create`: Use strides_for_shape to derive strides from shape
- Other functions updated to handle symbolic dimensions

### 3. Implement Shape_tracker (nx/lib/core/shape_tracker.ml)

```ocaml
type t = {
  views: View.t list;  (* List of view transformations *)
  base_shape: Symbolic.shape;  (* Original tensor shape *)
}

(* Creation *)
val create : View.t -> t
val from_shape : Symbolic.shape -> t

(* View operations - all lazy *)
val reshape : t -> Symbolic.shape -> t option
val permute : t -> int array -> t
val expand : t -> Symbolic.shape -> t
val pad : t -> (Symbolic.dim * Symbolic.dim) array -> t
val shrink : t -> (Symbolic.dim * Symbolic.dim) array -> t
val flip : t -> bool array -> t

(* Analysis *)
val simplify : t -> t  (* Merge compatible views *)
val is_c_contiguous : t -> bool
val real_strides : t -> Symbolic.shape option  (* None if not materializable *)
val to_view : t -> View.t  (* Compose all views *)

(* Symbolic operations *)
val vars : t -> SymVar.t list
val bind_vars : t -> (string * int) list -> t
val is_concrete : t -> bool
```

### 4. Update backend interface (nx/lib/core/backend_intf.ml)

Replace `view` lens with `shape_tracker`:
```ocaml
val shape_tracker : ('a, 'b) t -> Shape_tracker.t
```

Keep shape-specific operations for Rune's effect system:
- `op_reshape`, `op_expand`, `op_permute`, `op_pad`, `op_shrink`, `op_flip`
- These operations now update Shape_tracker instead of materializing
- They still need to exist as backend ops to raise effects in nx_rune
- In eager mode (CPU/Metal), they just update metadata
- In symbolic mode (Rune), they capture the operation for JIT compilation

Keep `op_contiguous` for forcing materialization.

### 5. Update frontend for lazy views (nx/lib/core/frontend.ml)

All view operations become lazy by updating Shape_tracker:
```ocaml
let reshape t ~shape =
  let tracker = Backend.shape_tracker t in
  match Shape_tracker.reshape tracker shape with
  | Some tracker' -> Backend.update_shape_tracker t tracker'
  | None -> 
    (* Cannot reshape lazily, must materialize first *)
    let t' = contiguous t in
    let tracker = Backend.shape_tracker t' in
    match Shape_tracker.reshape tracker shape with
    | Some tracker' -> Backend.update_shape_tracker t' tracker'
    | None -> failwith "reshape: incompatible shapes"
```

### 6. Backend implementation updates

**Native backend (nx/lib/native/)**:
- When ops need to access data, check if Shape_tracker is contiguous
- For ops requiring contiguous memory: Call op_contiguous internally
- Only force realization when absolutely necessary (e.g., incompatible strides, device transfer)
- Use symbolic shape evaluation for runtime shape binding

**Metal backend (nx/lib/metal/)**:
- Similar logic, but some views can use buffer views
- Generate kernels that handle strided access patterns

**Key insight**: Realization happens automatically in the backend when needed, not explicitly in the frontend (addressing the NOTE about where realization happens).

### 7. Rune integration (nx_rune.ml)

- Symbolic_tensor carries Shape_tracker with potentially symbolic shapes
- No new effect needed - symbolic shapes are resolved during JIT lowering
- During lowering, symbolic shapes are resolved to concrete values via variable bindings
- JIT can specialize kernels for different shape bindings
- Similar to tinygrad's Variable.bind() mechanism but integrated with Rune's effect system

### 8. JIT updates (rune/lib-jit/)

**ir.ml**:
- Define independent view representation (since rune_jit is separate from nx):
  ```ocaml
  type view = {
    shape: int array;
    strides: int array; 
    offset: int;
    mask: (int * int) array option;
  }
  ```
- VIEW appears only in lowered IR, not high-level graph
- Symbolic shapes use existing nodes:
  - `Define_Var` and `Bind` already exist for symbolic variables
  - Shape arithmetic uses regular binary ops (Add, Mul, Div, Mod)
  - No special shape-specific nodes needed

**lowerer.ml**:
- Analyze Shape_tracker to determine if views can be fused
- Insert efficient index calculations for strided access
- Track shape specializations for kernel caching

### 9. Example Usage

```ocaml
(* Symbolic shapes remain internal - users use Rune.placeholder *)
let model input_shape =
  let x = Rune.placeholder ~shape:[None; None; Some 768] in
  (* Internally creates symbolic dims for batch and sequence length *)
  
  (* Lazy view operations - just update Shape_tracker *)
  let y = x 
    |> Nx.transpose ~axis:[1; 0; 2]  
    |> Nx.flatten ~start_axis:0 ~end_axis:1  (* Reshapes to [-1, 768] *)
  in
  
  (* Build computation graph with symbolic shapes *)
  let z = Nx.matmul y weight in
  z

(* At runtime, JIT specializes for concrete shapes *)
let result = Rune.run model input  (* Shape [32; 128; 768] triggers compilation *)
```

## Success Criteria

- [ ] Views are lazy and don't copy data unnecessarily
- [ ] Symbolic shapes enable dynamic batching
- [ ] Convolution performance improved
- [ ] Memory usage reduced for view-heavy code
- [ ] JIT generates efficient specialized kernels
- [ ] All existing tests pass

## Risks & Mitigation

1. **API changes**: Gradual migration with compatibility layer
2. **Symbolic complexity**: Start simple, add features incrementally  
3. **Performance regressions**: Benchmark throughout development
4. **Debugging difficulty**: Add shape tracing and view visualization

## Notes

- Shape_tracker is the single source of truth for tensor shape/view information
- Realization is backend-driven, happening only when data access is needed
- Symbolic shapes follow tinygrad's two-stage pattern: define symbolically, bind concretely
- This design maintains nx's clean separation between frontend API and backend implementation