# Ndarray vs NumPy Comparison

This document compares the Ndarray library (OCaml) with NumPy (Python), highlighting similarities, differences, and providing equivalent code examples.

- [Ndarray vs NumPy Comparison](#ndarray-vs-numpy-comparison)
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

Ndarray is a numerical computing library for OCaml. It takes heavy inspiration from NumPy and aims to be as familiar as possible to NumPy users. That said, there are some phyilosophical differences between the two.

- **Pure OCaml Implementation:** Ndarray is fully native OCaml without C bindings. For that reason, it typically doesn't match NumPy's raw performances. But it is not trying to: while we care about performance, we prioritize the local development experience, where performance is not critical. That said, Ndarray uses a backend architecture under the hood, so it can easily be extended to use C or CUDA backends. This is what libraries like Rune are doing, implementing custom backends for Ndarray, making them suitable for production use cases.
- **Portable Compilation:** In return for a pure OCaml implementation, you get to compile Ndarray to JavaScript, WebAssembly, or even unikernels. Making it suitable for a wide range of application.
- **Type Safety First:** Ndarray leverages OCaml's strong type system and doesn't perform automatic type casting between array types. You can still use the `astype` function for explicit type conversions.
- **Bigarray Foundation:** Built on OCaml's Bigarray, Ndarray uses uint8 instead of boolean arrays and doesn't support string arrays.

Apart from the above, Ndarray is designed to be as close to NumPy as possible. The broadcasting rules are the same, and most functions behave similarly. If you notice an undocumented difference, please open an issue; it's probably a bug.

## 2. Array Creation

### Basic Array Creation

**Ndarray:**
```ocaml
(* Creating a zeros array *)
let zeros = Ndarray.zeros Bigarray.float64 [|3; 3|]

(* Creating a ones array *)
let ones = Ndarray.ones Bigarray.float64 [|3; 3|]

(* Creating an array with a specific value *)
let full = Ndarray.full Bigarray.float64 [|3; 3|] 5.0

(* Creating a range *)
let range = Ndarray.arange Bigarray.int32 0 10 1

(* Creating an identity matrix *)
let identity = Ndarray.identity Bigarray.float64 3
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

**Ndarray:**
```ocaml
(* Creating from existing data *)
let data = [|1.0; 2.0; 3.0; 4.0|]
let arr = Ndarray.create Bigarray.float64 [|2; 2|] data

(* Creating using a function *)
let init_arr = Ndarray.init Bigarray.float64 [|3; 3|] (fun idx -> 
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

**Ndarray:**
```ocaml
(* Element-wise addition *)
let result = Ndarray.add arr1 arr2

(* In-place addition *)
let _ = Ndarray.add_inplace arr1 arr2

(* Scalar multiplication *)
let scaled = Ndarray.mul_scalar arr 2.0

(* Matrix multiplication *)
let matmul_result = Ndarray.matmul arr1 arr2
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

**Ndarray:**
```ocaml
(* Reshape array *)
let reshaped = Ndarray.reshape [|1; 6|] arr

(* Transpose array *)
let transposed = Ndarray.transpose arr

(* Flatten array *)
let flattened = Ndarray.flatten arr

(* Concatenate arrays *)
let concat = Ndarray.concatenate ~axis:0 [arr1; arr2]
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

**Ndarray:**
```ocaml
(* Get a single element *)
let element = Ndarray.get_item [|0; 1|] arr

(* Set a single element *)
let _ = Ndarray.set_item [|0; 1|] 5.0 arr

(* Get a slice/subarray *)
let slice = Ndarray.get [|0|] arr
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

**Ndarray:**
```ocaml
(* Sum of all elements *)
let total = Ndarray.sum arr

(* Mean of all elements *)
let avg = Ndarray.mean arr

(* Min and max values *)
let min_val = Ndarray.min arr
let max_val = Ndarray.max arr

(* Sum along an axis *)
let axis_sum = Ndarray.sum ~axes:[|0|] arr
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

**Ndarray:**
```ocaml
(* Matrix inverse *)
let inv_a = Ndarray.inv a

(* Solve linear system Ax = b *)
let x = Ndarray.solve a b

(* SVD decomposition *)
let u, s, vt = Ndarray.svd a

(* Eigenvalue decomposition *)
let eigenvalues, eigenvectors = Ndarray.eig a
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

**Ndarray:**
```ocaml
(* Broadcast a smaller array to match dimensions *)
let broadcasted = Ndarray.broadcast_to [|3; 3|] smaller_arr

(* Broadcasting happens automatically in operations *)
let result = Ndarray.add matrix vector
```

**NumPy:**
```python
# Broadcast a smaller array to match dimensions
broadcasted = np.broadcast_to(smaller_arr, (3, 3))

# Broadcasting happens automatically in operations
result = matrix + vector
```

## 8. Conditional Operations

**Ndarray:**
```ocaml
(* Create a boolean mask *)
let mask = Ndarray.greater arr (Ndarray.scalar Bigarray.float64 0.5)

(* Apply condition with where *)
let result = Ndarray.where mask arr1 arr2
```

**NumPy:**
```python
# Create a boolean mask
mask = arr > 0.5

# Apply condition with where
result = np.where(mask, arr1, arr2)
```

## 9. Random Number Generation

**Ndarray:**
```ocaml
(* Generate uniform random numbers *)
let random = Ndarray.rand Bigarray.float64 [|3; 3|]

(* Generate normal distributed random numbers *)
let normal = Ndarray.randn Bigarray.float64 [|3; 3|]
```

**NumPy:**
```python
# Generate uniform random numbers
random = np.random.rand(3, 3)

# Generate normal distributed random numbers
normal = np.random.randn(3, 3)
```

## 10. Real-World Example: Linear Regression

**Ndarray:**
```ocaml
(* Generate sample data *)
let x = Ndarray.linspace Bigarray.float64 0.0 10.0 100
let y = Ndarray.(add (mul_scalar x 2.0) (randn Bigarray.float64 [|100|]))

(* Reshape x for design matrix *)
let x_design = Ndarray.(concatenate ~axis:1 [ones Bigarray.float64 [|100; 1|]; 
                                           reshape [|100; 1|] x])

(* Compute coefficients using normal equation *)
let xtx = Ndarray.matmul (Ndarray.transpose x_design) x_design
let xty = Ndarray.matmul (Ndarray.transpose x_design) (Ndarray.reshape [|100; 1|] y)
let coeffs = Ndarray.solve xtx xty

(* Make predictions *)
let y_pred = Ndarray.matmul x_design coeffs
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
