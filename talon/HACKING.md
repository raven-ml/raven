# Talon Developer Guide

## Architecture

Talon is a dataframe library inspired by pandas and polars, providing tabular data manipulation with heterogeneous column types and comprehensive null handling.

### Core Components

- **[lib/talon.ml](lib/talon.ml)**: Core dataframe operations and column types
- **[lib/csv/](lib/csv/)**: CSV I/O (Talon_csv sublibrary)
- **[lib/json/](lib/json/)**: JSON I/O (Talon_json sublibrary)

### Key Design Principles

1. **Immutable operations**: All operations return new dataframes
2. **Explicit null semantics**: Nullable columns with optional masks for numerics
3. **Type-safe aggregations**: GADT-based column types with type-specific operations
4. **Nx integration**: Numeric columns backed by Nx tensors for vectorization
5. **Row applicative**: Efficient multi-column computations in single pass

## Data Model

### Column Types

Three column variants via GADT:

```ocaml
type t =
  | P : ('a, 'b) Nx.dtype * ('a, 'b) Nx.t * bool array option -> t  (* Numeric *)
  | S : string option array -> t                                     (* String *)
  | B : bool option array -> t                                       (* Boolean *)
```

**Design rationale:**
- `P`: Vectorized numeric operations via Nx tensors
- `S`/`B`: Direct option arrays for non-numeric data
- Optional mask on `P`: Explicit null tracking separate from sentinel values

### Null Representation

**Numeric columns (P):**
- **With mask**: `Some mask` where `true` = null (mask takes precedence)
- **Without mask**: Sentinel values (NaN for floats, min_int for integers)

**String/Boolean columns (S/B):**
- `None` = null
- `Some value` = present value

**Why both mask and sentinels?**
1. Compatibility: Accept data with sentinel nulls
2. Explicitness: Mask distinguishes NaN values from nulls
3. Performance: Mask-less columns avoid extra array

### Dataframe Structure

```ocaml
type t = {
  columns : (string * Col.t) list;  (* Named columns *)
  length : int;                      (* Row count *)
}
```

**Invariants:**
- All columns have same length
- Column names unique
- Length = 0 for empty dataframes

## Development Workflow

### Building and Testing

```bash
# Build talon
dune build talon/

# Run tests
dune build talon/test/test_talon.exe && _build/default/talon/test/test_talon.exe

# Run specific test suite
_build/default/talon/test/test_talon.exe test "Aggregations"
```

### Testing Patterns

**Test with nulls:**

```ocaml
let test_sum_with_nulls () =
  let df = create [
    ("x", Col.float64_opt [|Some 1.0; None; Some 3.0|]);
  ] in
  let sum = Agg.Float.sum df "x" in
  Alcotest.(check (float 1e-6)) "sum" 4.0 sum  (* Nulls skipped *)
```

**Test type safety:**

```ocaml
let test_type_mismatch () =
  let df = create [("name", Col.string_list ["Alice"; "Bob"])] in
  (* This should fail at compile time: *)
  (* Agg.Float.sum df "name"  -- Type error! *)
  ()
```

## Row Operations

### The Row Applicative

Efficient multi-column computation:

```ocaml
module Row : sig
  type 'a t  (* Applicative for row-wise computation *)

  val map : 'a t -> f:('a -> 'b) -> 'b t
  val map2 : 'a t -> 'b t -> f:('a -> 'b -> 'c) -> 'c t
  val int32 : string -> int32 t
  val float64 : string -> float t
  (* ... *)
end
```

**How it works:**

```ocaml
(* Build computation *)
let computation = Row.(
  map2
    (float64 "price")
    (int32 "quantity")
    ~f:(fun p q -> p *. Int32.to_float q)
)

(* Execute on dataframe *)
let totals = map_rows df computation
```

Compiles to efficient single-pass loop:

```ocaml
(* Internally generates: *)
for i = 0 to length - 1 do
  let price = get_float64 df "price" i in
  let quantity = get_int32 df "quantity" i in
  result.(i) <- price *. Int32.to_float quantity
done
```

### Why Applicative?

**Pros:**
- Single pass: Extract all columns once, loop once
- Type-safe: Column types checked at construction
- Composable: Combine operations with `map2`, `map3`, etc.

**vs. Naive approach:**

```ocaml
(* BAD: Multiple passes *)
let prices = to_float64 df "price" in
let quantities = to_int32 df "quantity" in
let totals = Array.map2 ( *. ) prices (Array.map Int32.to_float quantities)
(* 3 allocations, 3 passes! *)
```

## Null Handling

### Creating Nullable Columns

```ocaml
(* Explicit mask *)
let col = Col.float64_opt [|Some 1.0; None; Some 3.0|]
(* Creates: tensor [1.0; nan; 3.0] with mask [false; true; false] *)

(* Sentinel-based (legacy) *)
let col = Col.float64 [|1.0; Float.nan; 3.0|]
(* Creates: tensor with NaN as null indicator, no mask *)
```

### Null-Aware Operations

**Aggregations skip nulls by default:**

```ocaml
Agg.Float.sum df "col"       (* Skips nulls *)
Agg.Float.sum ~skipna:false  (* NaN if any null *)
```

**Null propagation in row ops:**

```ocaml
(* None + x = None *)
let result = Row.(
  map2
    (float64_opt "x")  (* Returns float option *)
    (float64_opt "y")
    ~f:(fun x y ->
      match x, y with
      | Some x, Some y -> Some (x +. y)
      | _ -> None)
)
```

### Mask vs. Sentinel Decision

**Use masks when:**
- Need to distinguish NaN values from nulls
- Data semantically has nulls (missing measurements)
- Interop with systems that track nulls explicitly

**Use sentinels when:**
- Source data uses NaN/min_int for nulls
- Don't care about NaN vs. null distinction
- Want minimal memory overhead

## Adding Features

### New Column Type

To add a new column type (e.g., DateTime):

```ocaml
(* 1. Add GADT variant *)
type t =
  | P : ...
  | S : ...
  | B : ...
  | D : datetime option array -> t  (* New! *)

(* 2. Add constructors *)
let datetime arr = D arr

(* 3. Add accessors *)
let to_datetime_array df name = ...

(* 4. Add Row accessor *)
module Row = struct
  ...
  let datetime name = DateTime name
end

(* 5. Add Agg module *)
module Agg = struct
  ...
  module DateTime = struct
    let min df name = ...
    let max df name = ...
  end
end
```

### New Aggregation

Add to appropriate Agg module:

```ocaml
module Agg = struct
  module Float = struct
    ...
    let std df name =
      let mean = mean df name in
      let variance = var df name in
      sqrt variance
  end
end
```

### New Operation

Add to main module:

```ocaml
let drop_nulls df =
  (* Keep only rows with no nulls *)
  filter_by df (fun _row_idx ->
    List.for_all (fun (_name, col) ->
      not (Col.is_null col row_idx)
    ) df.columns
  )
```

## Join Operations

### Join Types

```ocaml
type join_type = Inner | Left | Right | Outer
```

**Implementation strategy:**

```ocaml
let join ~left ~right ~on ~how =
  (* 1. Build index on right dataframe *)
  let right_index = build_index right on in

  (* 2. Probe left dataframe *)
  let rows = match how with
    | Inner -> inner_join left right_index on
    | Left -> left_join left right_index on
    | ...
  in

  (* 3. Concatenate results *)
  from_rows rows
```

**Hash join:**
- Index right side by join key
- Probe with left side
- O(n + m) expected time

## Group By

### Grouping Strategy

```ocaml
let group_by df key_fn =
  (* 1. Compute group keys *)
  let keys = map_rows df key_fn in

  (* 2. Build groups map *)
  let groups : (key, int list) Hashtbl.t in
  Array.iteri (fun i key ->
    Hashtbl.add_multi groups key i
  ) keys;

  (* 3. Return grouped dataframe *)
  {df; groups}
```

**Aggregation:**

```ocaml
let aggregate grouped ~agg =
  (* For each group, apply aggregation *)
  Hashtbl.fold (fun key indices acc ->
    let subset = select_rows grouped.df indices in
    let result = agg subset in
    (key, result) :: acc
  ) grouped.groups []
```

## Common Pitfalls

### Type Mismatches

GADTs prevent most type errors, but runtime checks needed:

```ocaml
(* Runtime check column exists and has correct type *)
let get_float64 df name =
  match List.assoc_opt name df.columns with
  | Some (P (Float64, tensor, mask)) -> (tensor, mask)
  | Some _ -> invalid_arg (name ^ ": not a float64 column")
  | None -> invalid_arg (name ^ ": column not found")
```

### Null Mask Consistency

Always check mask when present:

```ocaml
(* Wrong: ignore mask *)
let get_value tensor _mask i =
  Nx.get tensor [|i|]

(* Correct: check mask *)
let get_value tensor mask i =
  match mask with
  | Some m when m.(i) -> None
  | _ -> Some (Nx.get tensor [|i|])
```

### Column Length Mismatch

Verify lengths when creating:

```ocaml
let create columns =
  let length = match columns with
    | [] -> 0
    | (_, col) :: _ -> Col.length col
  in
  (* Check all columns have same length *)
  List.iter (fun (name, col) ->
    if Col.length col <> length then
      invalid_arg (name ^ ": column length mismatch")
  ) columns;
  {columns; length}
```

## Performance

- **Vectorize**: Use Nx operations on numeric columns
- **Batch row operations**: Use Row applicative for multi-column
- **Avoid intermediate dataframes**: Chain operations when possible
- **Index for joins**: Pre-build hash tables for repeated joins

## Code Style

- **Labeled arguments**: `~skipna`, `~on`, `~how`
- **Option for nullable**: Use `option` for nullable non-numeric types
- **Explicit types**: Type signatures on public functions
- **Errors**: `"column_name: error description"` or `"operation: error"`

## Related Documentation

- [CLAUDE.md](../CLAUDE.md): Project-wide conventions
- [README.md](README.md): User-facing documentation
- [nx/HACKING.md](../nx/HACKING.md): Nx tensor operations
- pandas documentation for API reference
