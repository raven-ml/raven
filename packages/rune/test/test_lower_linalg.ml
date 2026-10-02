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
open Tolk

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
    ~indices:(Nx.create Nx.int64 [| m |] (Array.map Int64.of_int order))
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
  let a = Nx.copy (Nx.zeros dt [| 64; 64 |])
  and b = Nx.copy (Nx.zeros dt [| 64; 64 |]) in
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

let target s = Result.get_ok (Helpers.Target.of_string s)

let tensor_cores =
  let metal = Cstyle.metal (target "METAL::Apple9") in
  let cuda = Cstyle.cuda (target "CUDA::sm_89") in
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

(* QR takes LAPACK's reflectors, as nx.cpu does: a column already zero below the
   diagonal is not reflected, and a reflected one's diagonal element has the
   opposite sign of the element it replaces. The factors are then eager's, signs
   included. *)
let qr_agrees ~bound ~mode a =
  let q, r = Nx.qr ~mode a in
  let q', r' = traced2 (fun () -> Nx.qr ~mode a) in
  near ~bound q (f64 q');
  near ~bound r (f64 r');
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
          let a =
            conditioned
              (Nx.div_s
                 (Nx.reshape [| 2; 3; 2 |] (Nx.arange Nx.float32 1 13 1))
                 12.)
          in
          qr_agrees ~bound:0x1p-18 ~mode:`Reduced a);
      test "the factors take eager's signs" (fun () ->
          let f32 r c xs = Nx.create Nx.float32 [| r; c |] xs in
          List.iter
            (fun a ->
              qr_agrees ~bound:0x1p-18 ~mode:`Reduced a;
              qr_agrees ~bound:0x1p-18 ~mode:`Complete a)
            [
              f32 2 2 [| 2.; 1.; 0.5; 3. |];
              f32 3 3 [| -2.; 0.; 0.; 0.; 3.; 0.; 0.; 0.; -0.5 |];
              f32 1 1 [| 0.5 |];
              f32 1 3 [| 0.5; -1.; 2. |];
              f32 3 1 [| -1.; 2.; 0.5 |];
              f32 3 2 [| 0.; 1.; 0.; 2.; 0.; 3. |];
            ]);
      test "a column whose squares underflow reflects as eager's does"
        (fun () ->
          let agrees dt xs =
            let a = Nx.create dt [| 2; 2 |] xs in
            qr_agrees ~bound:0x1p-18 ~mode:`Reduced a;
            qr_agrees ~bound:0x1p-18 ~mode:`Complete a
          in
          agrees Nx.float32 [| 1.; 0.; 1e-25; 1. |];
          agrees Nx.float32 [| 1e-30; 0.; 1e-25; 1. |];
          agrees Nx.float64 [| 1.; 0.; 1e-170; 1. |];
          agrees Nx.float64 [| 1e-200; 0.; 1e-170; 1. |]);
      test "a column whose squares overflow reflects as eager's does" (fun () ->
          let agrees dt xs =
            let a = Nx.create dt [| 2; 2 |] xs in
            List.iter
              (fun mode ->
                let q, r = Nx.qr ~mode a in
                let q', r' = traced2 (fun () -> Nx.qr ~mode a) in
                near ~bound:0x1p-18 q q';
                near ~bound:0x1p-18 r r')
              [ `Reduced; `Complete ]
          in
          agrees Nx.float32 [| 2e38; 1.; 1e30; 2. |];
          agrees Nx.float32 [| 3e38; 1.; 3e38; 2. |];
          agrees Nx.float64 [| 1e308; 1.; 1e308; 2. |]);
      test
        "a column that takes no reflection leaves q as eager's, whatever it \
         holds" (fun () ->
          let agrees r c xs =
            let a = Nx.create Nx.float32 [| r; c |] xs in
            let q, _ = Nx.qr ~mode:`Complete a in
            let q', _ = traced2 (fun () -> Nx.qr ~mode:`Complete a) in
            near ~bound:0x1p-18 q q'
          in
          agrees 3 3 [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; Float.nan |];
          agrees 2 2 [| 1.; 3e38; 1.; 3e38 |]);
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
          let metal = Cstyle.metal (target "METAL::Apple9") in
          let a = Nx.ones Nx.float32 [| 2; 2 |] in
          raises_match
            (function Rune_internals.Lower.Jit_error _ -> true | _ -> false)
            (fun () -> trace ~renderer:(fun _ -> metal) (fun () -> Nx.svd a)));
    ]

(* Symmetric eigendecomposition

   Eager and compiled eigenvalues are each within some [n u] of [a]'s norm of
   the exact ones, the largest of whose magnitudes eager's are: up to [14 n u]
   measured, and the bound is [32 n u] of it, as for the vectors. The vectors of
   eigenvalues at least [delta] apart are unique up to sign, each within [n u
   |a| / delta] of the exact one. Every decomposition is checked whole:
   orthonormal vectors that rebuild [a]. *)

(* Symmetric matrices [Q diag(w) Qᵀ], [Q] orthogonal and the spectrum [w] of
   each of [n] rows drawn by [spectrum n], with NaN above the diagonal, which
   eigh never reads. *)
let with_spectrum dt spectrum =
  let open Gen in
  with_pp Nx.pp
    (let* b = batch in
     let* n = size in
     let matrices = Array.fold_left ( * ) 1 b in
     let* w = list ~size:(constant matrices) (spectrum n) in
     let+ g =
       array ~size:(constant (matrices * n * n)) (float_range (-1.) 1.)
     in
     let q = fst (Nx.qr (Nx.create Nx.float64 (Array.append b [| n; n |]) g)) in
     let w = Nx.create Nx.float64 (Array.append b [| n |]) (Array.concat w) in
     let a =
       Nx.matmul
         (Nx.mul q (Nx.unsqueeze ~axes:[ -2 ] w))
         (Nx.matrix_transpose q)
     in
     lower_only (Nx.cast dt a))

let signed magnitude =
  Gen.(
    let+ m = magnitude and+ negative = bool in
    if negative then -.m else m)

let each g n = Gen.array ~size:(Gen.constant n) g

(* Magnitudes [0.3 k] with a little noise, [k] a shuffle of [1] to [n], and
   their negations: eigenvalues at least [0.25] apart. *)
let separated n =
  let open Gen in
  let* order = permutation (List.init n (fun k -> k + 1)) in
  let+ noise = each (float_range 0. 0.05) n and+ negative = each bool n in
  Array.of_list
    (List.mapi
       (fun i k ->
         let m = (0.3 *. float_of_int k) +. noise.(i) in
         if negative.(i) then -.m else m)
       order)

let symmetric a = Nx.add (Nx.tril a) (Nx.matrix_transpose (Nx.tril ~k:(-1) a))

(* [decomposes ~bound a (w, v)] asserts that [v] is orthonormal and that [v
   diag(w) vᵀ] is [a]'s lower triangle mirrored. *)
let decomposes ~bound a (w, v) =
  orthonormal ~bound v;
  let v = f64 v in
  near ~bound
    (symmetric (f64 a))
    (Nx.matmul
       (Nx.mul v (Nx.unsqueeze ~axes:[ Nx.ndim w - 1 ] w))
       (Nx.matrix_transpose v))

let eigh_agrees ?separation ~bound a =
  let w, v = Nx.eigh a in
  let w', v' = traced2 (fun () -> Nx.eigh a) in
  near ~bound w w';
  exact w' (traced (fun () -> Nx.eigvalsh a));
  decomposes ~bound a (w', v');
  match separation with
  | None -> ()
  | Some delta ->
      (* [vᵀ v'] is the identity, up to the signs of its columns. *)
      let n = Nx.dim (-1) a in
      let along = Nx.abs (Nx.matmul (Nx.matrix_transpose (f64 v)) (f64 v')) in
      near
        ~bound:(bound *. largest w /. delta)
        (Nx.broadcast_to (Nx.shape along) (Nx.eye Nx.float64 n))
        along

let eigh_bound u a = 32. *. float_of_int (Int.max 1 (Nx.dim (-1) a)) *. u

let eigh =
  group "eigh"
    [
      per_float factor_dtypes "spectra"
        {
          on_float =
            (fun dt u ->
              let law name ?separation spectrum =
                prop ~count:50 name (with_spectrum dt spectrum) (fun a ->
                    eigh_agrees ?separation ~bound:(eigh_bound u a) a)
              in
              [
                law "separated eigenvalues" ~separation:0.25 separated;
                law "repeated eigenvalues"
                  (each (Gen.of_list ~pp:pp_float [ -1.; 1.; 2. ]));
                law "ill-conditioned"
                  (each
                     (signed
                        (Gen.map
                           (fun e -> Float.ldexp 1. (-e))
                           (Gen.int_range 0 40))));
                prop ~count:50 "any symmetric matrix" (square_matrix dt)
                  (fun a ->
                    let a = lower_only a in
                    eigh_agrees ~bound:(eigh_bound u a) a);
              ]);
        };
      slow "a 32 x 32 matrix of two repeated eigenvalues reaches its roundoff"
        (fun () ->
          List.iter
            (fun (F (_, dt, u)) ->
              let n = 32 in
              let st = Random.State.make [| n |] in
              let g =
                Nx.create Nx.float64 [| n; n |]
                  (Array.init (n * n) (fun _ -> Random.State.float st 2. -. 1.))
              in
              let q = fst (Nx.qr g) in
              let w =
                Nx.create Nx.float64 [| n |]
                  (Array.init n (fun i -> if i mod 2 = 0 then -1. else 1.))
              in
              let a =
                Nx.cast dt
                  (Nx.matmul
                     (Nx.mul q (Nx.unsqueeze ~axes:[ 0 ] w))
                     (Nx.matrix_transpose q))
              in
              eigh_agrees ~bound:(eigh_bound u a) a)
            (List.filter
               (fun (F (_, dt, _)) -> Nx_dtype.itemsize dt >= 4)
               factor_dtypes));
      test "NaN gives NaN values" (fun () ->
          let a = Nx.create Nx.float32 [| 2; 2 |] [| Float.nan; 0.; 1.; 3. |] in
          exact
            (Nx.full Nx.float64 [| 2 |] Float.nan)
            (traced (fun () -> Nx.eigvalsh a)));
      test "one element" (fun () ->
          let a = Nx.create Nx.float32 [| 1; 1 |] [| -3. |] in
          let w, v = traced2 (fun () -> Nx.eigh a) in
          exact (Nx.create Nx.float64 [| 1 |] [| -3. |]) w;
          exact (Nx.ones Nx.float32 [| 1; 1 |]) v);
      test "no element" (fun () ->
          let w, v =
            traced2 (fun () -> Nx.eigh (Nx.zeros Nx.float32 [| 2; 0; 0 |]))
          in
          exact (Nx.zeros Nx.float64 [| 2; 0 |]) w;
          exact (Nx.zeros Nx.float32 [| 2; 0; 0 |]) v);
      test "the upper triangle is read under uplo U" (fun () ->
          let a = Nx.create Nx.float64 [| 2; 2 |] [| 2.; 1.; Float.nan; 2. |] in
          let w, _ = traced2 (fun () -> Nx.eigh ~uplo:`U a) in
          near ~bound:0x1p-50 (Nx.create Nx.float64 [| 2 |] [| 1.; 3. |]) w);
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
  (* Captures with storage of their own: a constant would fold. *)
  let x shape = Nx.copy (Nx.zeros Nx.float32 shape) in
  let case file f =
    Golden.graph
      ("golden/lower_linalg/" ^ file ^ ".golden")
      (fun () ->
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
    ]

(* Loops

   A factorization that repeats a step as many times as its shapes fix holds the
   step once, in a loop: its graph, the loop's body included, grows with the
   matrices only through its other parts, such as the sort of the eigenvalues,
   and so does what compiling it costs. A graph that held every step would more
   than double from [n] rows to [2 n], as the steps do. *)

let nodes f =
  let s, ys = trace f in
  List.length
    (Ops.toposort
       (Ops.sink (List.map (fun (Nx.P y) -> Rune_internals.Lower.uop s y) ys)))

let loops =
  let x n = Nx.copy (Nx.zeros Nx.float32 [| n; n |]) in
  let grows_by_less_than_half name f =
    test name (fun () ->
        List.iter
          (fun n ->
            let small = nodes (f (x n)) in
            less ~msg:(string_of_int n) int
              ~than:(small + (small / 2))
              (nodes (f (x (2 * n)))))
          [ 8; 16; 32 ])
  in
  group "loops"
    [
      grows_by_less_than_half "eigh" (fun a () ->
          let w, v = Nx.eigh a in
          [ Nx.P w; Nx.P v ]);
      grows_by_less_than_half "svd" (fun a () ->
          let u, s, vt = Nx.svd a in
          [ Nx.P u; Nx.P s; Nx.P vt ]);
      grows_by_less_than_half "qr" (fun a () ->
          let q, r = Nx.qr a in
          [ Nx.P q; Nx.P r ]);
    ]

(* Integers *)

let integers =
  group "integers"
    [
      test "a factorization of integers raises as eager does" (fun () ->
          let a = Nx.create Nx.int32 [| 2; 2 |] [| 4l; 1l; 1l; 3l |] in
          let both name f =
            let error =
              Invalid_argument
                (name ^ ": linalg requires a float or complex dtype")
            in
            raises error (fun () -> ignore (f ()));
            raises error (fun () -> ignore (trace f))
          in
          both "cholesky" (fun () ->
              Nx.Op.eval (Cholesky { upper = false; x = a }));
          both "qr" (fun () -> Nx.Op.eval (Qr { reduced = true; x = a }));
          both "lu" (fun () -> Nx.Op.eval (Lu a));
          both "svd" (fun () ->
              Nx.Op.eval (Svd { full_matrices = false; x = a }));
          both "eigh" (fun () -> Nx.Op.eval (Eigh { vectors = true; x = a }));
          both "solve_triangular" (fun () ->
              Nx.Op.eval
                (Solve_triangular
                   {
                     upper = true;
                     transpose = false;
                     unit_diag = false;
                     a;
                     b = a;
                   })));
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
         eigh;
         loops;
         integers;
         on_the_host;
         parity;
       ]
