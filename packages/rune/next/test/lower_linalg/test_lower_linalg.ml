(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Matrix products and factorizations, lowered: each is checked against nx's
   eager result by its class. An integer product wraps and gives eager's bits; a
   float product is within the error of some association of its sums, and exact
   where every association is. A factorization is within a bound measured for
   its row, a multiple of the size and the unit roundoff relative to the largest
   element of eager's factor, over well-conditioned matrices; where its factors
   are unique only up to signs, it is compared up to them. The edges are pinned:
   singular and indefinite matrices, NaN, one element and none. The graphs are
   evaluated by tolk's reference interpreter. *)

open Windtrap
open Nx_test
open Traces
open Tolk_next

let pp_float ppf x = Format.fprintf ppf "%h" x

let traced f =
  let s, y = trace f in
  value s y

let traced2 f =
  let s, (a, b) = trace f in
  (value s a, value s b)

let traced3 f =
  let s, (a, b, c) = trace f in
  (value s a, value s b, value s c)

let agrees f = exact (f ()) (traced f)

(* Operands *)

type float_dtype = F : string * (float, 'b) Nx.dtype * float -> float_dtype

(* The float dtypes and their unit roundoff. A factorization of [float16]
   computes at [float32] and rounds once, within half an ulp of [float16]. *)
let factor_dtypes =
  [
    F ("float32", Nx.float32, 0x1p-24);
    F ("float64", Nx.float64, 0x1p-53);
    F ("float16", Nx.float16, 0x1p-11);
  ]

let product_dtypes =
  List.filter (fun (F (_, dt, _)) -> Nx_dtype.itemsize dt >= 4) factor_dtypes
  @ [ F ("float16", Nx.float16, 0x1p-11); F ("bfloat16", Nx.bfloat16, 0x1p-8) ]

type float_law = { on_float : 'b. (float, 'b) Nx.dtype -> float -> test list }

let per_float dtypes name { on_float } =
  group name
    (List.map (fun (F (dname, dt, u)) -> group dname (on_float dt u)) dtypes)

let size = Gen.int_range 1 5
let batch = Gen.of_list [ [||]; [| 2 |] ]

(* Elements of at most one in magnitude, whose squares are normal floats in
   every dtype, or zero: a norm is a square root of a sum of squares. *)
let unit_value =
  Gen.map
    (fun x -> if Float.abs x < 0x1p-6 then 0. else x)
    (Gen.float_range (-1.) 1.)

(* Matrices of [dtype] and [shape], their elements drawn from [value], laid out
   by rows, by columns, or reversed. *)
let matrices dtype value shape =
  let open Gen in
  let r = Array.length shape in
  let* xs = array ~size:(constant (Array.fold_left ( * ) 1 shape)) value in
  let+ layout =
    of_list
      (if r < 2 then [ `Rows; `Reversed ] else [ `Rows; `Columns; `Reversed ])
  in
  match layout with
  | `Rows -> Nx.create dtype shape xs
  | `Reversed -> Nx.flip (Nx.create dtype shape xs)
  | `Columns ->
      let swapped = Array.copy shape in
      swapped.(r - 2) <- shape.(r - 1);
      swapped.(r - 1) <- shape.(r - 2);
      Nx.matrix_transpose (Nx.create dtype swapped xs)

let matrix ?(value = unit_value) dtype shape =
  Gen.with_pp Nx.pp (matrices dtype value shape)

let f64 x = Nx.cast Nx.float64 x

(* [conditioned x] is [x] with one more than its larger size added on the
   leading diagonal of its matrices: of full rank, well-conditioned, and
   diagonally dominant. *)
let conditioned x =
  let m = Nx.dim (-2) x and n = Nx.dim (-1) x in
  Nx.add x
    (Nx.mul_s (Nx.eye ~m:n (Nx.dtype x) m) (float_of_int (Int.max m n + 1)))

(* [spd x] is [x xᵀ] conditioned: symmetric and positive-definite. *)
let spd x = conditioned (Nx.matmul x (Nx.matrix_transpose x))

let square_matrix ?value dtype =
  let open Gen in
  let* b = batch in
  let* n = size in
  matrix ?value dtype (Array.append b [| n; n |])

let conditioned_matrix dtype ~rows ~columns =
  let open Gen in
  let* m = rows in
  let* n = columns in
  map conditioned (matrix dtype [| m; n |])

(* Conditioned matrices with their rows shuffled, so that each step pivots. *)
let pivoted dtype =
  let open Gen in
  let* a = conditioned_matrix dtype ~rows:size ~columns:size in
  let m = Nx.dim 0 a in
  let+ keys = array ~size:(constant m) float in
  let order = Array.init m Fun.id in
  Array.stable_sort (fun i j -> Float.compare keys.(i) keys.(j)) order;
  Nx.take ~axis:0
    ~indices:(Nx.create Nx.int32 [| m |] (Array.map Int32.of_int order))
    a

(* Measured bounds *)

let largest x =
  Array.fold_left
    (fun m v -> if Float.is_finite v then Float.max m (Float.abs v) else m)
    0.
    (Nx.to_array (f64 x))

(* [near ~bound expected actual] asserts that [actual] has [expected]'s shape
   and that each of its elements is within [bound] times the largest finite
   magnitude of [expected] of the element of [expected] there: the same infinity
   where that is one, and NaN where it is NaN. *)
let near ~bound expected actual =
  if Nx.shape expected <> Nx.shape actual then
    failf "shape %a, eager %a" pp_shape (Nx.shape actual) pp_shape
      (Nx.shape expected);
  let e = Nx.to_array (f64 expected) and a = Nx.to_array (f64 actual) in
  let limit = bound *. largest expected in
  Array.iteri
    (fun i ei ->
      let ai = a.(i) in
      let within =
        if Float.is_finite ei then Float.abs (ai -. ei) <= limit
        else Float.equal ai ei
      in
      if not within then
        failf "element %d: %h, eager %h, beyond %h" i ai ei limit)
    e

(* [orthonormal ~bound q] asserts that the columns of [q] are orthonormal. *)
let orthonormal ~bound q =
  let k = Nx.dim (-1) q in
  near ~bound
    (Nx.broadcast_to
       (Array.append (Array.sub (Nx.shape q) 0 (Nx.ndim q - 2)) [| k; k |])
       (Nx.eye Nx.float64 k))
    (Nx.matmul (Nx.matrix_transpose (f64 q)) (f64 q))

(* The signs that the diagonal of the first matrix of [r] and [r'] differ by, as
   a row: one where either is zero. *)
let signs r r' =
  let diagonal r = Nx.diagonal (f64 r) in
  let s = Nx.mul (diagonal r) (diagonal r') in
  let s =
    Nx.where (Nx.less_s s 0.) (Nx.full_like s (-1.)) (Nx.full_like s 1.)
  in
  Nx.unsqueeze ~axes:[ 0 ] s

(* Products *)

let product_shapes =
  let open Gen in
  let* b = batch in
  let* m = int_range 0 3 in
  let* k = int_range 0 4 in
  let+ n = int_range 0 3 in
  (Array.append b [| m; k |], [| k; n |])

let products_of dtype value =
  let open Gen in
  with_pp
    (fun ppf (a, b) -> Format.fprintf ppf "%a@ times %a" Nx.pp a Nx.pp b)
    (let* sa, sb = product_shapes in
     let* a = matrices dtype value sa in
     let+ b = matrices dtype value sb in
     (a, b))

let exact_float =
  Gen.of_list ~pp:pp_float
    [
      0.;
      -0.;
      1.;
      -1.;
      2.;
      -3.;
      0.5;
      Float.infinity;
      Float.neg_infinity;
      Float.nan;
    ]

(* [|a - e| <= 2 (k - 1) u sum |a_ik b_kj|]: [a] and [e] are each within half of
   it of the exact sum. *)
let rounded_sum u (a, b) =
  let expected = Nx.matmul a b and actual = traced (fun () -> Nx.matmul a b) in
  let magnitude = Nx.to_array (Nx.matmul (Nx.abs (f64 a)) (Nx.abs (f64 b))) in
  let k = Nx.dim (-1) a in
  Array.iteri
    (fun i e ->
      let x = (Nx.to_array (f64 actual)).(i) in
      let bound = 2. *. float_of_int (k - 1) *. u *. magnitude.(i) in
      if not (Float.abs (x -. e) <= bound) then
        failf "element %d: %h, eager %h, beyond %h" i x e bound)
    (Nx.to_array (f64 expected))

let products =
  group "products"
    [
      group "integer products wrap"
        (List.map
           (fun (Int_dtype { name; dtype; bits; signed; of_i64; _ }) ->
             prop name
               (products_of dtype (Gen.map of_i64 (int_value ~bits ~signed)))
               (fun (a, b) -> agrees (fun () -> Nx.matmul a b)))
           int_dtypes);
      per_float product_dtypes "products of small numbers are exact"
        {
          on_float =
            (fun dt _ ->
              [
                prop "matmul" (products_of dt exact_float) (fun (a, b) ->
                    agrees (fun () -> Nx.matmul a b));
              ]);
        };
      per_float product_dtypes "products are rounded sums"
        {
          on_float =
            (fun dt u ->
              [ prop "matmul" (products_of dt unit_value) (rounded_sum u) ]);
        };
      test "narrow products are exact before they are summed" (fun () ->
          let x = 1. +. 0x1p-8 in
          let a = Nx.create Nx.float16 [| 1; 2 |] [| x; -1. |] in
          let b = Nx.create Nx.float16 [| 2; 1 |] [| x; 1. |] in
          agrees (fun () -> Nx.matmul a b));
      test "batch axes broadcast" (fun () ->
          let a = Nx.reshape [| 2; 1; 3; 2 |] (Nx.arange Nx.float32 0 12 1) in
          let b = Nx.reshape [| 4; 2; 3 |] (Nx.arange Nx.float32 0 24 1) in
          agrees (fun () -> Nx.matmul a b));
      group "a zero product is +0."
        [
          test "of one term" (fun () ->
              let a = Nx.full Nx.float32 [| 2; 1 |] (-0.) in
              agrees (fun () -> Nx.matmul a (Nx.ones Nx.float32 [| 1; 2 |])));
          test "of terms of -0." (fun () ->
              let a = Nx.full Nx.float32 [| 2; 3 |] (-0.) in
              agrees (fun () -> Nx.matmul a (Nx.ones Nx.float32 [| 3; 2 |])));
          test "of no term" (fun () ->
              let a = Nx.zeros Nx.float32 [| 2; 0 |] in
              agrees (fun () -> Nx.matmul a (Nx.zeros Nx.float32 [| 0; 3 |])));
        ];
    ]

(* Tensor cores: the kernel of a narrow matrix product, optimised for a GPU,
   multiplies on its tensor cores. *)

let multiplies_on_tensor_cores ren dt =
  let a = Nx.zeros dt [| 64; 64 |] and b = Nx.zeros dt [| 64; 64 |] in
  let _, y = trace ~renderer:(fun _ -> ren) (fun () -> Nx.matmul a b) in
  List.iter
    (fun k ->
      let lowered = Codegen.full_rewrite_to_sink k ren in
      if
        not
          (List.exists
             (fun u -> Op.equal (Ops.op u) Op.Wmma)
             (Ops.toposort lowered))
      then fail "no tensor core multiplies the product")
    (Ops.src (Programs.kernels y))

let tensor_cores =
  let metal = Cstyle.metal (Helpers.target ~arch:"Apple9" "METAL") in
  let cuda = Cstyle.cuda (Helpers.target ~arch:"sm_89" "CUDA") in
  group "tensor cores"
    [
      group "metal"
        [
          test "float16" (fun () -> multiplies_on_tensor_cores metal Nx.float16);
          test "bfloat16" (fun () ->
              multiplies_on_tensor_cores metal Nx.bfloat16);
        ];
      group "cuda"
        [
          test "float16" (fun () -> multiplies_on_tensor_cores cuda Nx.float16);
          test "bfloat16" (fun () ->
              multiplies_on_tensor_cores cuda Nx.bfloat16);
        ];
    ]

(* Cholesky *)

(* [a] with NaN above its diagonal, which a factorization never reads. *)
let lower_only a =
  let n = Nx.dim (-1) a in
  Nx.where
    (Nx.triu ~k:1 (Nx.ones Nx.bool [| n; n |]))
    (Nx.full_like a Float.nan) a

let cholesky =
  group "cholesky"
    [
      per_float factor_dtypes "positive-definite matrices"
        {
          on_float =
            (fun dt u ->
              List.map
                (fun upper ->
                  prop
                    (if upper then "upper" else "lower")
                    (square_matrix dt)
                    (fun x ->
                      let a = lower_only (spd x) in
                      let n = float_of_int (Nx.dim (-1) a) in
                      near
                        ~bound:(4. *. n *. u)
                        (Nx.cholesky ~upper a)
                        (traced (fun () -> Nx.cholesky ~upper a))))
                [ false; true ]);
        };
      test "a pivot that is not positive is NaN, and every column after it"
        (fun () ->
          let a =
            Nx.create Nx.float32 [| 3; 3 |]
              [| 4.; 0.; 0.; 2.; -1.; 0.; 1.; 1.; 5. |]
          in
          raises_match
            (function
              | Nx.Linalg_error { kind = `Not_positive_definite; _ } -> true
              | _ -> false)
            (fun () -> Nx.cholesky a);
          let nan = Float.nan in
          exact
            (Nx.create Nx.float32 [| 3; 3 |]
               [| 2.; 0.; 0.; 1.; nan; 0.; 0.5; nan; nan |])
            (traced (fun () -> Nx.cholesky a)));
      test "a zero pivot is NaN" (fun () ->
          let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 0.; 0.; 0. |] in
          exact
            (Nx.create Nx.float32 [| 2; 2 |] [| 1.; 0.; 0.; Float.nan |])
            (traced (fun () -> Nx.cholesky a)));
      test "one element" (fun () ->
          agrees (fun () ->
              Nx.cholesky (Nx.create Nx.float32 [| 1; 1 |] [| 9. |])));
      test "no element" (fun () ->
          agrees (fun () -> Nx.cholesky (Nx.zeros Nx.float32 [| 2; 0; 0 |])));
    ]

(* Triangular solves *)

let right_hand_sides dt a =
  let open Gen in
  let shape = Array.sub (Nx.shape a) 0 (Nx.ndim a - 1) in
  let* columns = of_list [ None; Some 1; Some 3 ] in
  match columns with
  | None -> matrix dt shape
  | Some k -> matrix dt (Array.append shape [| k |])

(* [a] with NaN in the triangle a solve does not read, and on the diagonal under
   [unit_diag]. *)
let read_only ~upper ~unit_diag a =
  let n = Nx.dim (-1) a in
  let ones = Nx.ones Nx.bool [| n; n |] in
  let unread = if upper then Nx.tril ~k:(-1) ones else Nx.triu ~k:1 ones in
  let unread =
    if unit_diag then Nx.logical_or unread (Nx.eye Nx.bool n) else unread
  in
  Nx.where unread (Nx.full_like a Float.nan) a

let solves =
  let flags =
    List.concat_map
      (fun upper ->
        List.concat_map
          (fun transpose ->
            List.map
              (fun unit_diag -> (upper, transpose, unit_diag))
              [ false; true ])
          [ false; true ])
      [ false; true ]
  in
  let name (upper, transpose, unit_diag) =
    String.concat " "
      [
        (if upper then "upper" else "lower");
        (if transpose then "transposed" else "");
        (if unit_diag then "unit" else "");
      ]
  in
  group "triangular solves"
    [
      per_float factor_dtypes "dominant matrices"
        {
          on_float =
            (fun dt u ->
              List.map
                (fun ((upper, transpose, unit_diag) as f) ->
                  prop (name f)
                    Gen.(
                      let* a = square_matrix dt in
                      let+ b = right_hand_sides dt a in
                      (conditioned a, b))
                    (fun (a, b) ->
                      let a = read_only ~upper ~unit_diag a in
                      let n = float_of_int (Nx.dim (-1) a) in
                      let solve () =
                        Nx.solve_triangular ~upper ~transpose ~unit_diag a b
                      in
                      near ~bound:(4. *. n *. u) (solve ()) (traced solve)))
                flags);
        };
      test "a zero pivot makes its row and those after it non-finite" (fun () ->
          let lower =
            Nx.create Nx.float32 [| 3; 3 |]
              [| 1.; 0.; 0.; 1.; 0.; 0.; 2.; 1.; 1. |]
          in
          let b = Nx.ones Nx.float32 [| 3 |] in
          raises_match
            (function
              | Nx.Linalg_error { kind = `Singular; _ } -> true | _ -> false)
            (fun () -> Nx.solve_triangular lower b);
          exact
            (Nx.create Nx.float32 [| 3 |] [| 1.; Float.nan; Float.nan |])
            (traced (fun () -> Nx.solve_triangular lower b));
          let upper = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 1.; 0.; 0. |] in
          exact
            (Nx.create Nx.float32 [| 2 |]
               [| Float.neg_infinity; Float.infinity |])
            (traced (fun () ->
                 Nx.solve_triangular ~upper:true upper
                   (Nx.ones Nx.float32 [| 2 |]))));
      test "no element" (fun () ->
          agrees (fun () ->
              Nx.solve_triangular
                (Nx.zeros Nx.float32 [| 0; 0 |])
                (Nx.zeros Nx.float32 [| 0; 2 |])));
    ]

(* LU *)

let lu_agrees ~bound a =
  let perm, l, u = Nx.lu a in
  let perm', l', u' = traced3 (fun () -> Nx.lu a) in
  exact perm perm';
  near ~bound l l';
  near ~bound u u'

let lu =
  group "lu"
    [
      per_float factor_dtypes "matrices"
        {
          on_float =
            (fun dt u ->
              [
                prop "pivots by magnitude" (pivoted dt) (fun a ->
                    let n =
                      float_of_int (Int.max (Nx.dim (-1) a) (Nx.dim (-2) a))
                    in
                    lu_agrees ~bound:(4. *. n *. u) a);
              ]);
        };
      test "the first of equal magnitudes is the pivot" (fun () ->
          let a =
            Nx.create Nx.float32 [| 3; 2 |] [| -2.; 1.; 2.; 3.; -2.; 5. |]
          in
          lu_agrees ~bound:0. a);
      test "a zero column leaves its column unscaled" (fun () ->
          let a =
            Nx.create Nx.float32 [| 3; 3 |]
              [| 0.; 1.; 2.; 0.; 3.; 4.; 0.; 5.; 7. |]
          in
          lu_agrees ~bound:0x1p-22 a);
      test "a NaN below the diagonal is never the pivot" (fun () ->
          let a =
            Nx.create Nx.float32 [| 3; 2 |] [| 1.; 2.; Float.nan; 4.; 5.; 6. |]
          in
          lu_agrees ~bound:0. a);
      test "a NaN on the diagonal is the pivot" (fun () ->
          let a =
            Nx.create Nx.float32 [| 2; 2 |]
              [| Float.nan; 2.; Float.infinity; 4. |]
          in
          lu_agrees ~bound:0. a);
      test "one element" (fun () ->
          lu_agrees ~bound:0. (Nx.create Nx.float32 [| 1; 1 |] [| -3. |]));
      test "no element" (fun () ->
          lu_agrees ~bound:0. (Nx.zeros Nx.float32 [| 3; 0 |]);
          lu_agrees ~bound:0. (Nx.zeros Nx.float32 [| 0; 2 |]));
    ]

(* QR *)

let qr_agrees ~bound ~mode a =
  let q, r = Nx.qr ~mode a in
  let q', r' = traced2 (fun () -> Nx.qr ~mode a) in
  let k = Int.min (Nx.dim (-2) a) (Nx.dim (-1) a) in
  let s = signs r r' in
  let leading q = Nx.slice [ Nx.A; Nx.R (0, k) ] q in
  near ~bound (leading q) (Nx.mul (leading (f64 q')) s);
  near ~bound
    (Nx.slice [ Nx.R (0, k); Nx.A ] r)
    (Nx.mul (Nx.slice [ Nx.R (0, k); Nx.A ] (f64 r')) (Nx.matrix_transpose s));
  orthonormal ~bound q';
  exact (Nx.triu r') r';
  near ~bound a (Nx.matmul q' r')

let qr =
  group "qr"
    [
      per_float factor_dtypes "matrices"
        {
          on_float =
            (fun dt u ->
              List.map
                (fun mode ->
                  prop
                    (match mode with
                    | `Reduced -> "reduced"
                    | `Complete -> "complete")
                    (conditioned_matrix dt ~rows:size ~columns:size)
                    (fun a ->
                      let n =
                        float_of_int (Int.max (Nx.dim 0 a) (Nx.dim 1 a))
                      in
                      qr_agrees ~bound:(16. *. n *. u) ~mode a))
                [ `Reduced; `Complete ]);
        };
      test "batch axes" (fun () ->
          let a = Nx.reshape [| 2; 3; 2 |] (Nx.arange Nx.float32 1 13 1) in
          let q, r = Nx.qr a and q', r' = traced2 (fun () -> Nx.qr a) in
          ignore (q, r);
          orthonormal ~bound:0x1p-18 q';
          near ~bound:0x1p-18 a (Nx.matmul q' r'));
      test "a zero column takes no reflection" (fun () ->
          let a = Nx.zeros Nx.float32 [| 3; 2 |] in
          let q, r = traced2 (fun () -> Nx.qr ~mode:`Complete a) in
          exact (Nx.eye Nx.float32 3) q;
          exact a r);
      test "one element" (fun () ->
          let a = Nx.create Nx.float32 [| 1; 1 |] [| -3. |] in
          qr_agrees ~bound:0x1p-20 ~mode:`Reduced a);
      test "no column: q is the identity" (fun () ->
          let q, r =
            traced2 (fun () ->
                Nx.qr ~mode:`Complete (Nx.zeros Nx.float32 [| 3; 0 |]))
          in
          exact (Nx.eye Nx.float32 3) q;
          exact (Nx.zeros Nx.float32 [| 3; 0 |]) r);
    ]

(* SVD *)

let svd_agrees ~bound ~full_matrices a =
  let u, s, vt = Nx.svd ~full_matrices a in
  let u', s', vt' = traced3 (fun () -> Nx.svd ~full_matrices a) in
  near ~bound s s';
  if Nx.shape u <> Nx.shape u' || Nx.shape vt <> Nx.shape vt' then
    fail "the factors' shapes are not eager's";
  orthonormal ~bound u';
  orthonormal ~bound (Nx.matrix_transpose vt');
  let k = Nx.dim (-1) s' in
  let leading = List.init (Nx.ndim a - 2) (fun _ -> Nx.A) in
  let columns x = Nx.slice (leading @ [ Nx.A; Nx.R (0, k) ]) (f64 x) in
  let rows x = Nx.slice (leading @ [ Nx.R (0, k); Nx.A ]) (f64 x) in
  near ~bound a
    (Nx.matmul
       (Nx.mul (columns u') (Nx.unsqueeze ~axes:[ Nx.ndim s' - 1 ] s'))
       (rows vt'))

let svd =
  group "svd"
    [
      per_float factor_dtypes "matrices"
        {
          on_float =
            (fun dt u ->
              List.map
                (fun full_matrices ->
                  prop ~count:50
                    (if full_matrices then "full" else "thin")
                    (conditioned_matrix dt ~rows:(Gen.int_range 1 4)
                       ~columns:(Gen.int_range 1 4))
                    (fun a ->
                      let n =
                        float_of_int (Int.max (Nx.dim 0 a) (Nx.dim 1 a))
                      in
                      svd_agrees ~bound:(16. *. n *. u) ~full_matrices a))
                [ false; true ]);
        };
      test "batch axes" (fun () ->
          let a = Nx.reshape [| 2; 2; 3 |] (Nx.arange Nx.float32 1 13 1) in
          svd_agrees ~bound:0x1p-18 ~full_matrices:true a);
      test "float64 values of a 12 x 12 matrix reach its roundoff" (fun () ->
          let st = Random.State.make [| 12 |] in
          let a =
            Nx.create Nx.float64 [| 12; 12 |]
              (Array.init 144 (fun _ -> Random.State.float st 2. -. 1.))
          in
          let _, s, _ = Nx.svd a in
          let _, s', _ = traced3 (fun () -> Nx.svd a) in
          near ~bound:(32. *. 12. *. 0x1p-53) s s');
      test "a rank-deficient matrix has orthonormal vectors and +0. values"
        (fun () ->
          let r1 =
            Nx.create Nx.float32 [| 3; 3 |]
              [| 1.; 2.; 3.; 2.; 4.; 6.; 3.; 6.; 9. |]
          in
          svd_agrees ~bound:0x1p-19 ~full_matrices:true r1;
          let _, s, _ =
            traced3 (fun () -> Nx.svd (Nx.zeros Nx.float32 [| 2; 3 |]))
          in
          exact (Nx.zeros Nx.float64 [| 2 |]) s);
      test "NaN gives NaN values" (fun () ->
          let a = Nx.create Nx.float32 [| 2; 2 |] [| Float.nan; 1.; 2.; 3. |] in
          let _, s, _ = traced3 (fun () -> Nx.svd a) in
          exact (Nx.full Nx.float64 [| 2 |] Float.nan) s);
      test "one element" (fun () ->
          svd_agrees ~bound:0. ~full_matrices:true
            (Nx.create Nx.float32 [| 1; 1 |] [| -3. |]));
      test "no element: the full factors are identities" (fun () ->
          let u, s, vt =
            traced3 (fun () ->
                Nx.svd ~full_matrices:true (Nx.zeros Nx.float32 [| 0; 3 |]))
          in
          exact (Nx.zeros Nx.float32 [| 0; 0 |]) u;
          exact (Nx.zeros Nx.float64 [| 0 |]) s;
          exact (Nx.eye Nx.float32 3) vt);
      test "a target without float64 refuses it" (fun () ->
          let metal = Cstyle.metal (Helpers.target ~arch:"Apple9" "METAL") in
          let a = Nx.ones Nx.float32 [| 2; 2 |] in
          raises_match
            (function Rune_next.Lower.Jit_error _ -> true | _ -> false)
            (fun () -> trace ~renderer:(fun _ -> metal) (fun () -> Nx.svd a)));
    ]

(* Compiled for the host

   What a kernel computes where the interpreter cannot tell: the sign of a zero
   sum over a contraction of one term, which the compiler unrolls, and integer
   products that overflow, which C leaves undefined for signed ones. *)

let compiled f =
  let s, y = trace f in
  exact (f ()) (Programs.compiled s y)

let on_the_host =
  group ~tags:[ "slow" ] "compiled for the host"
    [
      test "a contraction of one term of -0. is +0." (fun () ->
          let a = Nx.full Nx.float32 [| 2; 1 |] (-0.) in
          compiled (fun () -> Nx.matmul a (Nx.ones Nx.float32 [| 1; 3 |])));
      test "integer products wrap" (fun () ->
          let a =
            Nx.create Nx.int32 [| 2; 2 |]
              [| Int32.max_int; 3l; Int32.min_int; -1l |]
          in
          compiled (fun () -> Nx.matmul a a);
          let b = Nx.create Nx.int8 [| 2; 2 |] [| 100; 100; 100; -100 |] in
          compiled (fun () -> Nx.matmul b b));
    ]

(* Graph parity: the kernels tinygrad schedules for the same program. *)

let parity =
  let x shape = Nx.zeros Nx.float32 shape in
  let case file f =
    Golden.graph (file ^ ".golden") (fun () ->
        let args = f () in
        Programs.kernels (snd (trace (fun () -> args ()))))
  in
  group "graph parity"
    [
      case "matmul" (fun () ->
          let a = x [| 4; 3 |] and b = x [| 3; 5 |] in
          fun () -> Nx.matmul a b);
      case "matmul_batched" (fun () ->
          let a = x [| 2; 1; 4; 3 |] and b = x [| 3; 3; 5 |] in
          fun () -> Nx.matmul a b);
      case "qr_q" (fun () ->
          let a = x [| 4; 3 |] in
          fun () -> fst (Nx.qr a));
      case "qr_r" (fun () ->
          let a = x [| 4; 3 |] in
          fun () -> snd (Nx.qr a));
    ]

let () =
  exit
  @@ run "lower_linalg"
       [
         products;
         tensor_cores;
         cholesky;
         solves;
         lu;
         qr;
         svd;
         on_the_host;
         parity;
       ]
