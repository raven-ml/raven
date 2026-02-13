# Nx vs NumPy Comparison

This document compares the Nx library (OCaml) with NumPy (Python), highlighting similarities, differences, and providing equivalent code examples.

- [Nx vs NumPy Comparison](#nx-vs-numpy-comparison)
  - [1. Overview](#1-overview)
  - [2. Array Creation](#2-array-creation)
    - [Basic Array Creation](#basic-array-creation)
    - [Advanced Array Creation](#advanced-array-creation)
  - [3. Array Operations](#3-array-operations)
    - [Basic Operations](#basic-operations)
    - [Array Manipulation](#array-manipulation)
  - [4. Element Access and Slicing](#4-element-access-and-slicing)
  - [5. Statistical Functions](#5-statistical-functions)
  - [6. Linear Algebra](#6-linear-algebra)
  - [7. Broadcasting](#7-broadcasting)
  - [8. Conditional Operations](#8-conditional-operations)
  - [9. Random Number Generation](#9-random-number-generation)
  - [10. Real-World Example: Linear Regression](#10-real-world-example-linear-regression)

## 1. Overview

Nx is a numerical computing library for OCaml. It takes heavy inspiration from NumPy and aims to be as familiar as possible to NumPy users. That said, there are some phyilosophical differences between the two.

- **Pure OCaml Implementation:** Nx is fully native OCaml without C bindings. For that reason, it typically doesn't match NumPy's raw performances. But it is not trying to: while we care about performance, we prioritize the local development experience, where performance is not critical. That said, Nx uses a backend architecture under the hood, so it can easily be extended to use C or CUDA backends. This is what libraries like Rune are doing, implementing custom backends for Nx, making them suitable for production use cases.
- **Portable Compilation:** In return for a pure OCaml implementation, you get to compile Nx to JavaScript, WebAssembly, or even unikernels. Making it suitable for a wide range of application.
- **Type Safety First:** Nx leverages OCaml's strong type system and doesn't perform automatic type casting between array types. You can still use the `astype` function for explicit type conversions.
- **Bigarray Foundation:** Built on OCaml's Bigarray, Nx uses uint8 instead of boolean arrays and doesn't support string arrays.

Apart from the above, Nx is designed to be as close to NumPy as possible. The broadcasting rules are the same, and most functions behave similarly. If you notice an undocumented difference, please open an issue; it's probably a bug.

## 2. Array Creation

### Basic Array Creation

**Nx:**
```ocaml
(* Creating a zeros array *)
let zeros = Nx.zeros Nx.float64 [|3; 3|]

(* Creating a ones array *)
let ones = Nx.ones Nx.float64 [|3; 3|]

(* Creating an array with a specific value *)
let full = Nx.full Nx.float64 [|3; 3|] 5.0

(* Creating a range *)
let range = Nx.arange Nx.int32 0 10 1

(* Creating an identity matrix *)
let identity = Nx.identity Nx.float64 3
```

**NumPy:**
```python
# Creating a zeros array
zeros = np.zeros((3, 3))

# Creating a ones array
ones = np.ones((3, 3))

# Creating an array with a specific value
full = np.full((3, 3), 5.0)

# Creating a range
range_array = np.arange(0, 10, 1)

# Creating an identity matrix
identity = np.identity(3)
```

### Advanced Array Creation

**Nx:**
```ocaml
(* Creating from existing data *)
let data = [|1.0; 2.0; 3.0; 4.0|]
let arr = Nx.create Nx.float64 [|2; 2|] data

(* Creating using a function *)
let init_arr = Nx.init Nx.float64 [|3; 3|] (fun idx ->
  float_of_int (idx.(0) + idx.(1)))
```

**NumPy:**
```python
# Creating from existing data
data = [1.0, 2.0, 3.0, 4.0]
arr = np.array(data).reshape(2, 2)

# Creating using a function
init_arr = np.fromfunction(lambda i, j: i + j, (3, 3))
```

## 3. Array Operations

### Basic Operations

**Nx:**
<!-- $MDX skip -->
```ocaml
(* Element-wise addition *)
let result = Nx.add arr1 arr2

(* Scalar multiplication *)
let scaled = Nx.mul_s arr 2.0

(* Matrix multiplication *)
let matmul_result = Nx.matmul arr1 arr2
```

**NumPy:**
```python
# Element-wise addition
result = arr1 + arr2

# In-place addition
arr1 += arr2

# Scalar multiplication
scaled = arr * 2.0

# Matrix multiplication
matmul_result = arr1 @ arr2  # or np.matmul(arr1, arr2)
```

### Array Manipulation

**Nx:**
<!-- $MDX skip -->
```ocaml
(* Reshape array *)
let reshaped = Nx.reshape [|1; 6|] arr

(* Transpose array *)
let transposed = Nx.transpose arr

(* Flatten array *)
let flattened = Nx.flatten arr

(* Concatenate arrays *)
let concat = Nx.concatenate ~axis:0 [arr1; arr2]
```

**NumPy:**
```python
# Reshape array
reshaped = arr.reshape(1, 6)

# Transpose array
transposed = arr.transpose()

# Flatten array
flattened = arr.flatten()

# Concatenate arrays
concat = np.concatenate([arr1, arr2], axis=0)
```

## 4. Element Access and Slicing

**Nx:**
<!-- $MDX skip -->
```ocaml
(* Get a single element *)
let element = Nx.item [0; 1] arr

(* Set a single element *)
let () = Nx.set_item [0; 1] 5.0 arr

(* Get a slice/subarray *)
let row = Nx.get [0] arr
```

**NumPy:**
```python
# Get a single element
element = arr[0, 1]

# Set a single element
arr[0, 1] = 5.0

# Get a slice/subarray
slice = arr[0]
```

## 5. Statistical Functions

**Nx:**
<!-- $MDX skip -->
```ocaml
(* Sum of all elements *)
let total = Nx.sum arr

(* Mean of all elements *)
let avg = Nx.mean arr

(* Min and max values *)
let min_val = Nx.min arr
let max_val = Nx.max arr

(* Sum along an axis *)
let axis_sum = Nx.sum ~axes:[|0|] arr
```

**NumPy:**
```python
# Sum of all elements
total = np.sum(arr)

# Mean of all elements
avg = np.mean(arr)

# Min and max values
min_val = np.min(arr)
max_val = np.max(arr)

# Sum along an axis
axis_sum = np.sum(arr, axis=0)
```

## 6. Linear Algebra

**Nx:**
<!-- $MDX skip -->
```ocaml
(* Matrix inverse *)
let inv_a = Nx.inv a

(* Solve linear system Ax = b *)
let x = Nx.solve a b

(* SVD decomposition *)
let u, s, vt = Nx.svd a

(* Eigenvalue decomposition *)
let eigenvalues, eigenvectors = Nx.eig a
```

**NumPy:**
```python
# Matrix inverse
inv_a = np.linalg.inv(a)

# Solve linear system Ax = b
x = np.linalg.solve(a, b)

# SVD decomposition
u, s, vt = np.linalg.svd(a)

# Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(a)
```

## 7. Broadcasting

**Nx:**
<!-- $MDX skip -->
```ocaml
(* Broadcast a smaller array to match dimensions *)
let broadcasted = Nx.broadcast_to [|3; 3|] smaller_arr

(* Broadcasting happens automatically in operations *)
let result = Nx.add matrix vector
```

**NumPy:**
```python
# Broadcast a smaller array to match dimensions
broadcasted = np.broadcast_to(smaller_arr, (3, 3))

# Broadcasting happens automatically in operations
result = matrix + vector
```

## 8. Conditional Operations

**Nx:**
<!-- $MDX skip -->
```ocaml
(* Create a boolean mask *)
let mask = Nx.greater arr (Nx.scalar Nx.float64 0.5)

(* Apply condition with where *)
let result = Nx.where mask arr1 arr2
```

**NumPy:**
```python
# Create a boolean mask
mask = arr > 0.5

# Apply condition with where
result = np.where(mask, arr1, arr2)
```

## 9. Random Number Generation

**Nx (keyed, JAX-style):**
```ocaml
(* Explicit key for reproducibility *)
let key = Nx.Rng.key 0

(* Generate uniform random numbers *)
let random = Nx.rand Nx.float64 ~key [|3; 3|]

(* Generate normal distributed random numbers *)
let normal = Nx.randn Nx.float64 ~key:(Nx.Rng.fold_in key 1) [|3; 3|]

(* Splitting for multiple draws *)
let key1, key2 =
  match Nx.Rng.split ~n:2 key with [| a; b |] -> (a, b) | _ -> assert false

let random' = Nx.rand Nx.float64 ~key:key1 [|3; 3|]
let normal' = Nx.randn Nx.float64 ~key:key2 [|3; 3|]
```

**NumPy:**
```python
# Generate uniform random numbers
random = np.random.rand(3, 3)

# Generate normal distributed random numbers
normal = np.random.randn(3, 3)
```

## 10. Real-World Example: Linear Regression

**Nx:**
```ocaml
(* Generate sample data *)
let base_key = Nx.Rng.key 0
let _data_key, noise_key =
  match Nx.Rng.split ~n:2 base_key with [| k1; k2 |] -> (k1, k2) | _ -> assert false

let x = Nx.linspace Nx.float64 0.0 10.0 100
let y =
  Nx.(
    add (mul_s x 2.0)
      (randn Nx.float64 ~key:noise_key [|100|]))

(* Reshape x for design matrix *)
let x_design = Nx.(concatenate ~axis:1 [ones Nx.float64 [|100; 1|];
                                           reshape [|100; 1|] x])

(* Compute coefficients using normal equation *)
let xtx = Nx.matmul (Nx.transpose x_design) x_design
let xty = Nx.matmul (Nx.transpose x_design) (Nx.reshape [|100; 1|] y)
let coeffs = Nx.solve xtx xty

(* Make predictions *)
let y_pred = Nx.matmul x_design coeffs
```

**NumPy:**
```python
# Generate sample data
x = np.linspace(0, 10, 100)
y = 2 * x + np.random.randn(100)

# Reshape x for design matrix
x_design = np.column_stack((np.ones(100), x))

# Compute coefficients using normal equation
xtx = x_design.T @ x_design
xty = x_design.T @ y.reshape(100, 1)
coeffs = np.linalg.solve(xtx, xty)

# Make predictions
y_pred = x_design @ coeffs
```
