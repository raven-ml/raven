# `03-indexing-and-slicing`

Select, slice, and mask: extract exactly the data you need. This example uses
a grade book to demonstrate every way Nx lets you reach into an array. The
indexing operators `.%{}` and `.${}` are `Nx.Infix`'s and the index
constructors `A`, `I`, `R`, `Rs`, `M` and `T` are `Nx`'s, used together
inside `Nx.(Infix.( … ))`.

```bash
dune exec packages/nx/examples/03-indexing-and-slicing/main.exe
```

## What You'll Learn

- Reading single elements with `item`
- Selecting rows and columns with `I` and `A`
- Range slicing with `R` and strided slicing with `Rs`
- Infix indexing syntax: `.%{}` and `.${}`
- Boolean masks with `M` and `where`
- Picking rows by positions held in a tensor with `T`

## Key Functions

| Function / Index              | Purpose                                |
| ----------------------------- | -------------------------------------- |
| `item [i; j] t`               | Extract a single OCaml scalar          |
| `I n`                         | Select index `n` along one axis        |
| `A`                           | Select all indices along an axis       |
| `R (start, stop)`             | Half-open range `[start, stop)`        |
| `Rs (start, stop, step)`      | Range with stride                      |
| `t.%{[i; ...]}`               | Infix `get`: the row at `i`            |
| `t.${[...]}`                  | Infix slicing (synonym for `slice`)    |
| `M mask`                      | Keep the positions where `mask` holds  |
| `where cond then_ else_`      | Element-wise conditional selection     |
| `T positions`                 | Gather rows at positions in a tensor   |
| `greater_s t scalar`          | Element-wise `t > scalar` → bool mask  |

## Output Walkthrough

The example starts with a 5×4 grade book (5 students, 4 subjects):

```
Grade book (students × subjects):
float64 [5; 4]
  [[88, 72, 95, 83],
   [45, 90, 67, 78],
   [92, 85, 91, 70],
   [76, 63, 80, 95],
   [60, 78, 55, 82]]
```

### Single element

```ocaml
Nx.item [ 0; 1 ] grades           (* → 72.0 *)
```

### Row and column selection

`t.%{[i]}`, the infix `get`, reads row `i`. The infix `.${[...]}` slices: `I n`
picks one index, `A` keeps the full axis:

```ocaml
Nx.Infix.(grades.%{[ 2 ]})        (* student 2, all subjects → [92, 85, 91, 70] *)
Nx.(Infix.(grades.${[ A; I 0 ]}))   (* all students, Math → [88, 45, 92, 76, 60] *)
```

### Range and strided slicing

`R (start, stop)` is a half-open range. `Rs (start, stop, step)` adds a stride:

```ocaml
Nx.(Infix.(grades.${[ R (1, 4); R (0, 2) ]}))          (* students 1-3, Math & Science *)
Nx.(Infix.(grades.${[ Rs (0, 5, 2); Rs (0, 4, 2) ]}))  (* every other student & subject *)
```

### Boolean masks

Build a boolean mask, then index with `M` to filter rows:

```ocaml
let high_math = Nx.greater_s math 85.0 in
Nx.(Infix.(grades.${[ M high_math ]}))
```

```
Math > 85 mask: bool [5] [true, false, true, false, false]
Students with Math > 85:
float64 [2; 4] [[88, 72, 95, 83], [92, 85, 91, 70]]
```

The number of rows a mask keeps is data: the selection reads the mask on the
host to learn it.

### Conditional replacement

`where` replaces elements based on a condition, here flooring all grades
below 60:

```ocaml
Nx.where (Nx.less_s grades 60.0) (Nx.full Nx.float64 [| 5; 4 |] 60.0) grades
```

## Index Types at a Glance

| Index          | Meaning        | Example                        |
| -------------- | -------------- | ------------------------------ |
| `I n`          | Single index   | `I 2`: third element           |
| `A`            | All indices    | `A`: keep entire axis          |
| `R (a, b)`     | Range `[a, b)` | `R (1, 4)`: indices 1, 2, 3    |
| `Rs (a, b, s)` | Strided range  | `Rs (0, 10, 2)`: even indices  |
| `L [...]`      | Explicit list  | `L [0; 3; 7]`: pick specific   |
| `T p`          | Array of positions | `T ids`: its shape replaces the axis |
| `D (p, n)`     | Window from a position in an array | `D (start, 4)`: 4 positions |
| `M mask`       | Boolean mask   | `M bool_array`: where true     |
| `N`            | New axis       | `N`: insert dimension          |

## Try It

1. Extract the Art column (column 3) for all students.
2. Use `Rs (-1, -6, -1)` to reverse the student order (negative step; a
   negative start or stop counts from the end).
3. Find students whose average grade across all subjects exceeds 80 using
   `mean ~axes:[1]` and a boolean mask.

## Next Steps

Continue to [04-reshaping-and-broadcasting](../04-reshaping-and-broadcasting/)
to learn how to change array shapes and let broadcasting align dimensions
automatically.
