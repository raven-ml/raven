(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Property-based tests for Nx operations.

   Each property verifies an algebraic law or invariant over randomly generated
   tensors. These complement the unit tests which cover edge cases, error
   conditions, and NaN/Inf behavior. *)

open Windtrap
open Test_nx_props_support

(* ── Arithmetic Properties ── *)

let arithmetic_props =
  [
    (* Addition *)
    prop "add commutative (f32)" f32_pair (fun (a, b) ->
        approx_equal (Nx.add a b) (Nx.add b a));
    prop "add commutative (i32)" i32_pair (fun (a, b) ->
        exact_equal (Nx.add a b) (Nx.add b a));
    prop "add identity (f32)" f32_any (fun a ->
        approx_equal (Nx.add a (Nx.zeros_like a)) a);
    prop "add identity (i32)" i32_any (fun a ->
        exact_equal (Nx.add a (Nx.zeros_like a)) a);
    prop "add inverse (f32)" f32_any (fun a ->
        let z = Nx.add a (Nx.neg a) in
        approx_equal z (Nx.zeros_like a));
    prop "sub is add neg (f32)" f32_pair (fun (a, b) ->
        approx_equal (Nx.sub a b) (Nx.add a (Nx.neg b)));
    prop "sub is add neg (i32)" i32_pair (fun (a, b) ->
        exact_equal (Nx.sub a b) (Nx.add a (Nx.neg b)));
    (* Multiplication *)
    prop "mul commutative (f32)" f32_pair (fun (a, b) ->
        approx_equal (Nx.mul a b) (Nx.mul b a));
    prop "mul commutative (i32)" i32_pair (fun (a, b) ->
        exact_equal (Nx.mul a b) (Nx.mul b a));
    prop "mul identity (f32)" f32_any (fun a ->
        approx_equal (Nx.mul a (Nx.ones_like a)) a);
    prop "mul identity (i32)" i32_any (fun a ->
        exact_equal (Nx.mul a (Nx.ones_like a)) a);
    prop "mul zero (f32)" f32_any (fun a ->
        approx_equal (Nx.mul a (Nx.zeros_like a)) (Nx.zeros_like a));
    prop "mul zero (i32)" i32_any (fun a ->
        exact_equal (Nx.mul a (Nx.zeros_like a)) (Nx.zeros_like a));
    prop "distributive (i32)" i32_triple (fun (a, b, c) ->
        exact_equal (Nx.mul a (Nx.add b c))
          (Nx.add (Nx.mul a b) (Nx.mul a c)));
    (* Division / Modulo *)
    prop "div inverse of mul (f32)" f32_pair (fun (a, b) ->
        assume (all_nonzero_f32 b);
        allclose ~atol:1e-3 ~rtol:1e-3 (Nx.div (Nx.mul a b) b) a);
    prop "div self = ones (f32)" f32_any (fun a ->
        assume (all_nonzero_f32 a);
        approx_equal (Nx.div a a) (Nx.ones_like a));
    prop "int div/mod relation (i32)" i32_pair_b_nonzero (fun (a, b) ->
        exact_equal (Nx.add (Nx.mul (Nx.div a b) b) (Nx.mod_ a b)) a);
    (* Negation *)
    prop "neg involution (f32)" f32_any (fun a ->
        approx_equal (Nx.neg (Nx.neg a)) a);
    prop "neg involution (i32)" i32_any (fun a ->
        exact_equal (Nx.neg (Nx.neg a)) a);
    (* Min / Max *)
    prop "maximum commutative (f32)" f32_pair (fun (a, b) ->
        assume (no_nan a && no_nan b);
        approx_equal (Nx.maximum a b) (Nx.maximum b a));
    prop "minimum commutative (f32)" f32_pair (fun (a, b) ->
        assume (no_nan a && no_nan b);
        approx_equal (Nx.minimum a b) (Nx.minimum b a));
    prop "maximum idempotent (f32)" f32_any (fun a ->
        assume (no_nan a);
        approx_equal (Nx.maximum a a) a);
  ]

(* ── Shape Manipulation Properties ── *)

let shape_props =
  [
    prop "reshape roundtrip (f32)" f32_any (fun t ->
        let flat = Nx.flatten t in
        approx_equal (Nx.reshape (Nx.shape t) flat) t);
    prop "flatten preserves data (f32)" f32_any (fun t ->
        Nx.to_array (Nx.flatten t) = Nx.to_array t);
    prop "transpose involution (2d f32)" f32_2d (fun t ->
        approx_equal (Nx.transpose (Nx.transpose t)) t);
    prop "transpose shape (2d f32)" f32_2d (fun t ->
        let s = Nx.shape t in
        let ts = Nx.shape (Nx.transpose t) in
        ts = [| s.(1); s.(0) |]);
    prop "flip involution (f32)" f32_any (fun t ->
        approx_equal (Nx.flip (Nx.flip t)) t);
    prop "copy preserves data (f32)" f32_any (fun t ->
        approx_equal (Nx.copy t) t);
    prop "copy independence (f32)" f32_any (fun t ->
        assume (Nx.numel t > 0);
        let c = Nx.copy t in
        let orig_first = Nx.item [ 0 ] (Nx.flatten t) in
        Nx.set_item [ 0 ] 99999.0 (Nx.flatten c);
        let after_first = Nx.item [ 0 ] (Nx.flatten t) in
        Float.equal orig_first after_first);
    prop "contiguous is contiguous (f32)" f32_any (fun t ->
        Nx.is_c_contiguous (Nx.contiguous t));
    prop "contiguous preserves data (f32)" f32_any (fun t ->
        approx_equal (Nx.contiguous t) t);
    prop "reshape preserves numel (f32)" f32_any (fun t ->
        Nx.numel (Nx.flatten t) = Nx.numel t);
  ]

(* ── Comparison Properties ── *)

let comparison_props =
  [
    prop "equal reflexive (f32)" f32_any (fun a ->
        assume (no_nan a);
        all_true (Nx.equal a a));
    prop "less irreflexive (f32)" f32_any (fun a ->
        all_true (Nx.logical_not (Nx.less a a)));
    prop "less/greater complement (f32)" f32_pair (fun (a, b) ->
        all_true
          (Nx.array_equal (Nx.less a b) (Nx.greater b a)));
    prop "less_equal from less|equal (f32)" f32_pair (fun (a, b) ->
        assume (no_nan a && no_nan b);
        all_true
          (Nx.array_equal (Nx.less_equal a b)
             (Nx.logical_or (Nx.less a b) (Nx.equal a b))));
    prop "not_equal complement of equal (f32)" f32_pair (fun (a, b) ->
        assume (no_nan a && no_nan b);
        all_true
          (Nx.array_equal (Nx.not_equal a b)
             (Nx.logical_not (Nx.equal a b))));
  ]

(* ── Logical & Bitwise Properties ── *)

let logical_bitwise_props =
  [
    prop "bitwise_not involution (i32)" i32_any (fun a ->
        exact_equal (Nx.bitwise_not (Nx.bitwise_not a)) a);
    prop "bitwise_and commutative (i32)" i32_pair (fun (a, b) ->
        exact_equal (Nx.bitwise_and a b) (Nx.bitwise_and b a));
    prop "bitwise_or commutative (i32)" i32_pair (fun (a, b) ->
        exact_equal (Nx.bitwise_or a b) (Nx.bitwise_or b a));
    prop "bitwise_xor self = zeros (i32)" i32_any (fun a ->
        exact_equal (Nx.bitwise_xor a a) (Nx.zeros_like a));
    prop "de morgan and (i32)" i32_pair (fun (a, b) ->
        exact_equal
          (Nx.bitwise_not (Nx.bitwise_and a b))
          (Nx.bitwise_or (Nx.bitwise_not a) (Nx.bitwise_not b)));
    prop "de morgan or (i32)" i32_pair (fun (a, b) ->
        exact_equal
          (Nx.bitwise_not (Nx.bitwise_or a b))
          (Nx.bitwise_and (Nx.bitwise_not a) (Nx.bitwise_not b)));
  ]

(* ── Rounding Properties ── *)

let rounding_props =
  let open Nx in
  [
    prop "floor <= input (f32)" f32_any (fun x ->
        assume (all_finite x);
        all_true (less_equal (floor x) x));
    prop "ceil >= input (f32)" f32_any (fun x ->
        assume (all_finite x);
        all_true (greater_equal (ceil x) x));
    prop "floor idempotent (f32)" f32_any (fun x ->
        assume (all_finite x);
        approx_equal (floor (floor x)) (floor x));
    prop "ceil idempotent (f32)" f32_any (fun x ->
        assume (all_finite x);
        approx_equal (ceil (ceil x)) (ceil x));
    prop "round idempotent (f32)" f32_any (fun x ->
        assume (all_finite x);
        approx_equal (round (round x)) (round x));
  ]

(* ── Sorting Properties ── *)

let sorting_props =
  [
    prop "sort is sorted (f32 1d)" f32_1d (fun x ->
        assume (no_nan x);
        let sorted, _indices = Nx.sort x in
        let n = Nx.numel sorted in
        let rec check i =
          if i >= n then true
          else Nx.item [ i - 1 ] sorted <= Nx.item [ i ] sorted && check (i + 1)
        in
        n <= 1 || check 1);
    prop "sort idempotent (f32 1d)" f32_1d (fun x ->
        assume (no_nan x);
        let s1, _ = Nx.sort x in
        let s2, _ = Nx.sort s1 in
        approx_equal s1 s2);
    prop "sort preserves shape (f32 1d)" f32_1d (fun x ->
        let sorted, _ = Nx.sort x in
        Nx.shape sorted = Nx.shape x);
    prop "argsort valid indices (f32 1d)" f32_1d (fun x ->
        let _, indices = Nx.sort x in
        let n = Nx.numel x in
        let valid = ref true in
        for i = 0 to n - 1 do
          let idx = Int32.to_int (Nx.item [ i ] indices) in
          if idx < 0 || idx >= n then valid := false
        done;
        !valid);
    prop "sort preserves elements (i32 1d)" i32_1d (fun x ->
        let sorted, _ = Nx.sort x in
        let a = Array.copy (Nx.to_array x) in
        let b = Array.copy (Nx.to_array sorted) in
        Array.sort Int32.compare a;
        Array.sort Int32.compare b;
        a = b);
  ]

(* ── Math Function Properties ── *)

let math_function_props =
  let mk_f32_constrained gen_val =
    let gen =
      let open Gen in
      let* shape = gen_shape ~max_ndim:3 ~max_dim:4 in
      gen_tensor_with_values Nx.float32 gen_val shape
    in
    mk_testable_f32 gen
  in
  let f32_small = mk_f32_constrained gen_float_small in
  let f32_positive = mk_f32_constrained gen_float_positive in
  let f32_unit = mk_f32_constrained gen_float_unit in
  let f32_trig = mk_f32_constrained gen_float_trig in
  let f32_recip =
    mk_f32_constrained (Gen.float_range 0.1 10.)
  in
  [
    prop "exp/log inverse (f32)" f32_small (fun x ->
        assume (all_finite x);
        allclose ~atol:1e-4 ~rtol:1e-4 (Nx.log (Nx.exp x)) x);
    prop "log/exp inverse (f32)" f32_positive (fun x ->
        allclose ~atol:1e-4 ~rtol:1e-4 (Nx.exp (Nx.log x)) x);
    prop "sin^2 + cos^2 = 1 (f32)" f32_trig (fun x ->
        let sum = Nx.add (Nx.square (Nx.sin x)) (Nx.square (Nx.cos x)) in
        allclose ~atol:1e-4 ~rtol:0. sum (Nx.ones_like x));
    prop "sqrt(square(x)) = abs(x) (f32)" f32_any (fun x ->
        assume (all_finite x);
        allclose ~atol:1e-4 ~rtol:1e-4 (Nx.sqrt (Nx.square x)) (Nx.abs x));
    prop "abs idempotent (f32)" f32_any (fun x ->
        approx_equal (Nx.abs (Nx.abs x)) (Nx.abs x));
    prop "sign * abs = x (f32)" f32_any (fun x ->
        assume (all_finite x && all_nonzero_f32 x);
        approx_equal (Nx.mul (Nx.sign x) (Nx.abs x)) x);
    prop "tanh range (f32)" f32_any (fun x ->
        assume (all_finite x);
        all_true (Nx.less_equal (Nx.abs (Nx.tanh x)) (Nx.ones_like x)));
    prop "recip involution (f32)" f32_recip (fun x ->
        allclose ~atol:1e-3 ~rtol:1e-3 (Nx.recip (Nx.recip x)) x);
    prop "square = mul self (f32)" f32_any (fun x ->
        approx_equal (Nx.square x) (Nx.mul x x));
    prop "asin(sin(x)) = x (f32)" f32_unit (fun x ->
        (* asin(sin(x)) = x only when x in [-pi/2, pi/2]; use values in (-1,1)
           which are well within that range when interpreted as radians *)
        allclose ~atol:1e-4 ~rtol:1e-4 (Nx.asin (Nx.sin x)) x);
  ]

(* ── Reduction Properties ── *)

let reduction_props =
  [
    prop "sum of ones = numel (f32)" f32_any (fun t ->
        let ones = Nx.ones_like t in
        let s = Nx.item [] (Nx.sum ones) in
        Float.abs (s -. Float.of_int (Nx.numel t)) < 1e-5);
    prop "prod of ones = 1 (f32)" f32_any (fun t ->
        let ones = Nx.ones_like t in
        Float.abs (Nx.item [] (Nx.prod ones) -. 1.0) < 1e-5);
    prop "mean = sum / numel (f32)" f32_any (fun t ->
        assume (Nx.numel t > 0);
        let m = Nx.item [] (Nx.mean t) in
        let s = Nx.item [] (Nx.sum t) in
        let n = Float.of_int (Nx.numel t) in
        Float.abs (m -. s /. n) < 1e-4);
    prop "max >= all elements (f32)" f32_any (fun t ->
        assume (no_nan t && Nx.numel t > 0);
        let mx = Nx.max t in
        all_true (Nx.less_equal t (Nx.broadcast_to (Nx.shape t) mx)));
    prop "min <= all elements (f32)" f32_any (fun t ->
        assume (no_nan t && Nx.numel t > 0);
        let mn = Nx.min t in
        all_true (Nx.greater_equal t (Nx.broadcast_to (Nx.shape t) mn)));
    prop "var >= 0 (f32)" f32_any (fun t ->
        assume (Nx.numel t > 0);
        Nx.item [] (Nx.var t) >= 0.0);
    prop "sum linearity (f32)" f32_pair (fun (a, b) ->
        let lhs = Nx.item [] (Nx.sum (Nx.add a b)) in
        let rhs = Nx.item [] (Nx.sum a) +. Nx.item [] (Nx.sum b) in
        Float.abs (lhs -. rhs) < 1e-2);
    prop "cumsum last = sum (f32 1d)" f32_1d (fun t ->
        assume (all_finite t && Nx.numel t > 0);
        let cs = Nx.cumsum t in
        let last = Nx.item [ Nx.numel t - 1 ] cs in
        let total = Nx.item [] (Nx.sum t) in
        Float.abs (last -. total) < 1e-3);
  ]

(* ── Linear Algebra Properties ── *)

let linalg_props =
  [
    prop "matmul identity (f64)" square_f64 (fun a ->
        let n = (Nx.shape a).(0) in
        let eye = Nx.identity Nx.float64 n in
        approx_equal ~epsilon:1e-10 (Nx.matmul a eye) a);
    prop "transpose matmul (f64)"
      (let gen =
         let open Gen in
         let* a = gen_square_f64 ~max_n:4 in
         let n = (Nx.shape a).(0) in
         let+ b =
           gen_tensor_with_values Nx.float64 (Gen.float_range (-5.) 5.)
             [| n; n |]
         in
         (a, b)
       in
       Testable.make
         ~pp:(pp_pair pp_tensor pp_tensor)
         ~equal:(fun (a1, b1) (a2, b2) ->
           approx_equal ~epsilon:1e-10 a1 a2
           && approx_equal ~epsilon:1e-10 b1 b2)
         ~gen ())
      (fun (a, b) ->
        let lhs = Nx.transpose (Nx.matmul a b) in
        let rhs = Nx.matmul (Nx.transpose b) (Nx.transpose a) in
        approx_equal ~epsilon:1e-8 lhs rhs);
    prop "trace = sum diagonal (f64)" square_f64 (fun a ->
        let tr = Nx.item [] (Nx.trace a) in
        let diag_sum = Nx.item [] (Nx.sum (Nx.diagonal a)) in
        Float.abs (tr -. diag_sum) < 1e-10);
    prop "inv roundtrip (f64 posdef)" posdef_f64 (fun a ->
        let n = (Nx.shape a).(0) in
        let eye = Nx.identity Nx.float64 n in
        let inv_a = Nx.inv a in
        allclose ~atol:1e-6 ~rtol:1e-6 (Nx.matmul inv_a a) eye);
    prop "qr reconstruction (f64)" square_f64 (fun a ->
        let q, r = Nx.qr a in
        allclose ~atol:1e-6 ~rtol:1e-6 (Nx.matmul q r) a);
    prop "svd reconstruction (f64)" square_f64 (fun a ->
        let u, s, vh = Nx.svd a in
        let n = (Nx.shape a).(0) in
        let s_diag = Nx.mul (Nx.identity Nx.float64 n) (Nx.reshape [| 1; n |] s) in
        let reconstructed = Nx.matmul (Nx.matmul u s_diag) vh in
        allclose ~atol:1e-6 ~rtol:1e-6 reconstructed a);
    prop "cholesky reconstruction (f64 posdef)" posdef_f64 (fun a ->
        let l = Nx.cholesky a in
        let reconstructed = Nx.matmul l (Nx.transpose l) in
        allclose ~atol:1e-6 ~rtol:1e-6 reconstructed a);
    prop "det of identity = 1"
      (Testable.make ~pp:Format.pp_print_int ~equal:Int.equal
         ~gen:(Gen.int_range 1 6) ())
      (fun n ->
        let eye = Nx.identity Nx.float64 n in
        Float.abs (Nx.item [] (Nx.det eye) -. 1.0) < 1e-10);
  ]

(* ── Concatenation Properties ── *)

let concat_props =
  [
    prop "concat single = identity (f32)" f32_any (fun t ->
        approx_equal (Nx.concatenate ~axis:0 [ t ]) t);
    prop "concat shape (f32)" f32_pair (fun (a, b) ->
        let sa = Nx.shape a and sb = Nx.shape b in
        assume
          (Array.length sa = Array.length sb
          && Array.length sa > 0
          && Array.sub sa 1 (Array.length sa - 1)
             = Array.sub sb 1 (Array.length sb - 1));
        let c = Nx.concatenate ~axis:0 [ a; b ] in
        (Nx.shape c).(0) = sa.(0) + sb.(0));
    prop "stack creates axis (f32)" f32_pair (fun (a, b) ->
        assume (Nx.shape a = Nx.shape b);
        let s = Nx.stack ~axis:0 [ a; b ] in
        Nx.ndim s = Nx.ndim a + 1 && (Nx.shape s).(0) = 2);
    prop "concat/split roundtrip (f32 1d)" f32_1d (fun t ->
        let n = Nx.numel t in
        assume (n >= 2 && n mod 2 = 0);
        let parts = Nx.split ~axis:0 2 t in
        approx_equal (Nx.concatenate ~axis:0 parts) t);
  ]

(* ── Indexing Properties ── *)

let indexing_props =
  [
    prop "item/set_item roundtrip (f32)" f32_with_index (fun (t, indices) ->
        let c = Nx.copy t in
        let v = 42.0 in
        Nx.set_item indices v c;
        Float.equal (Nx.item indices c) v);
    prop "get/set roundtrip (f32)" f32_any (fun t ->
        assume (Nx.ndim t >= 1);
        let c = Nx.copy t in
        let idx = [ 0 ] in
        let sub = Nx.get idx t in
        Nx.set idx c sub;
        approx_equal (Nx.get idx c) sub);
    prop "slice A is identity (f32)" f32_any (fun t ->
        let spec = List.init (Nx.ndim t) (fun _ -> Nx.A) in
        approx_equal (Nx.slice spec t) t);
    prop "slice full range = identity (f32 1d)" f32_1d (fun t ->
        let n = Nx.numel t in
        approx_equal (Nx.slice [ Nx.R (0, n) ] t) t);
    prop "take all indices = identity (f32 1d)" f32_1d (fun t ->
        let n = Nx.numel t in
        let indices = Nx.arange Nx.int32 0 n 1 in
        approx_equal (Nx.take indices t) t);
    prop "take indices valid (f32 1d)"
      f32_1d_with_take_indices
      (fun (t, indices) ->
        let taken = Nx.take indices t in
        let n_idx = Nx.numel indices in
        let ok = ref true in
        for i = 0 to n_idx - 1 do
          let idx = Int32.to_int (Nx.item [ i ] indices) in
          let expected = Nx.item [ idx ] t in
          let actual = Nx.item [ i ] taken in
          if not (Float.equal expected actual) then ok := false
        done;
        !ok);
    prop "take_along_axis with argsort = sort (f32 1d)" f32_1d (fun t ->
        assume (no_nan t);
        let sorted, _ = Nx.sort t in
        let arg_indices = Nx.argsort t in
        let gathered = Nx.take_along_axis ~axis:0 arg_indices t in
        approx_equal gathered sorted);
    prop "extract preserves count (f32)" f32_with_mask (fun (t, mask) ->
        let extracted = Nx.extract ~condition:mask t in
        let n_true =
          let flat = Nx.flatten mask in
          let count = ref 0 in
          for i = 0 to Nx.numel flat - 1 do
            if Nx.item [ i ] flat then incr count
          done;
          !count
        in
        Nx.numel extracted = n_true);
    prop "set_slice/slice roundtrip (f32)" f32_any (fun t ->
        assume (Nx.ndim t >= 1 && (Nx.shape t).(0) >= 1);
        let spec = [ Nx.R (0, 1) ] in
        let sub = Nx.slice spec t in
        let c = Nx.copy t in
        Nx.set_slice spec c sub;
        approx_equal c t);
    prop "nonzero indices are valid (i32 1d)" i32_1d (fun t ->
        let nz = Nx.nonzero t in
        let indices = nz.(0) in
        let n = Nx.numel t in
        let ok = ref true in
        for i = 0 to Nx.numel indices - 1 do
          let idx = Int32.to_int (Nx.item [ i ] indices) in
          if idx < 0 || idx >= n then ok := false
          else if Int32.equal (Nx.item [ idx ] t) 0l then ok := false
        done;
        !ok);
  ]

(* ── Broadcasting Properties ── *)

let broadcasting_props =
  [
    prop "broadcast_to idempotent (f32)" f32_with_broadcast_shape
      (fun (t, target) ->
        let b = Nx.broadcast_to target t in
        approx_equal (Nx.broadcast_to target b) b);
    prop "broadcast_to preserves values (f32)" f32_with_broadcast_shape
      (fun (t, target) ->
        let b = Nx.broadcast_to target t in
        (* Every element in broadcast result must exist in original *)
        let orig_vals = Nx.to_array (Nx.flatten (Nx.contiguous t)) in
        let bc_vals = Nx.to_array (Nx.flatten (Nx.contiguous b)) in
        Array.for_all
          (fun v -> Array.exists (fun o -> Float.equal v o) orig_vals)
          bc_vals);
    prop "broadcasted common shape (f32)" f32_broadcastable_pair (fun (a, b) ->
        let a', b' = Nx.broadcasted a b in
        Nx.shape a' = Nx.shape b');
    prop "broadcasted symmetric shape (f32)" f32_broadcastable_pair
      (fun (a, b) ->
        let a1, _ = Nx.broadcasted a b in
        let _, b2 = Nx.broadcasted b a in
        Nx.shape a1 = Nx.shape b2);
    prop "broadcast scalar to any shape (f32)" f32_any (fun t ->
        let v = 3.0 in
        let s = Nx.scalar Nx.float32 v in
        let b = Nx.broadcast_to (Nx.shape t) s in
        Nx.shape b = Nx.shape t
        && all_true (Nx.equal b (Nx.full_like t v)));
    prop "add with broadcast = add after broadcast (f32)" f32_broadcastable_pair
      (fun (a, b) ->
        let result = Nx.add a b in
        let a', b' = Nx.broadcasted a b in
        let result2 = Nx.add a' b' in
        approx_equal result result2);
    prop "expand_dims/squeeze roundtrip (f32)" f32_any (fun t ->
        let expanded = Nx.expand_dims [ 0 ] t in
        let squeezed = Nx.squeeze ~axes:[ 0 ] expanded in
        approx_equal squeezed t);
    prop "broadcast_arrays consistent with broadcasted (f32)"
      f32_broadcastable_pair (fun (a, b) ->
        let arr = Nx.broadcast_arrays [ a; b ] in
        let a', b' = Nx.broadcasted a b in
        approx_equal (List.nth arr 0) a' && approx_equal (List.nth arr 1) b');
  ]

(* ── Einsum Equivalence Properties ── *)

let einsum_props =
  let mk_f32_matmul_pair =
    let gen =
      let open Gen in
      let* m = int_range 1 6 in
      let* n = int_range 1 6 in
      let* k = int_range 1 6 in
      let+ a = gen_f32 [| m; n |] and+ b = gen_f32 [| n; k |] in
      (a, b)
    in
    Testable.make
      ~pp:(pp_pair pp_tensor pp_tensor)
      ~equal:(fun (a1, b1) (a2, b2) ->
        approx_equal a1 a2 && approx_equal b1 b2)
      ~gen ()
  in
  let mk_f32_1d_pair =
    let gen =
      let open Gen in
      let* n = int_range 1 10 in
      let+ a = gen_f32 [| n |] and+ b = gen_f32 [| n |] in
      (a, b)
    in
    Testable.make
      ~pp:(pp_pair pp_tensor pp_tensor)
      ~equal:(fun (a1, b1) (a2, b2) ->
        approx_equal a1 a2 && approx_equal b1 b2)
      ~gen ()
  in
  let mk_f32_outer_pair =
    let gen =
      let open Gen in
      let* m = int_range 1 8 in
      let* n = int_range 1 8 in
      let+ a = gen_f32 [| m |] and+ b = gen_f32 [| n |] in
      (a, b)
    in
    Testable.make
      ~pp:(pp_pair pp_tensor pp_tensor)
      ~equal:(fun (a1, b1) (a2, b2) ->
        approx_equal a1 a2 && approx_equal b1 b2)
      ~gen ()
  in
  [
    (* einsum matmul = Nx.matmul *)
    prop "einsum ij,jk->ik = matmul" mk_f32_matmul_pair (fun (a, b) ->
        let via_einsum = Nx.einsum "ij,jk->ik" [| a; b |] in
        let via_matmul = Nx.matmul a b in
        allclose ~atol:1e-4 ~rtol:1e-4 via_einsum via_matmul);
    (* einsum transpose = Nx.transpose *)
    prop "einsum ij->ji = transpose" f32_2d (fun a ->
        let via_einsum = Nx.einsum "ij->ji" [| a |] in
        let via_transpose = Nx.transpose a in
        approx_equal via_einsum via_transpose);
    (* einsum trace = Nx.trace *)
    prop "einsum ii-> = trace" square_f64 (fun a ->
        let via_einsum = Nx.item [] (Nx.einsum "ii->" [| a |]) in
        let via_trace = Nx.item [] (Nx.trace a) in
        Float.abs (via_einsum -. via_trace) < 1e-10);
    (* einsum diagonal = Nx.diagonal *)
    prop "einsum ii->i = diagonal" square_f64 (fun a ->
        let via_einsum = Nx.einsum "ii->i" [| a |] in
        let via_diagonal = Nx.diagonal a in
        approx_equal ~epsilon:1e-10 via_einsum via_diagonal);
    (* einsum dot product = sum of elementwise mul *)
    prop "einsum i,i-> = dot" mk_f32_1d_pair (fun (a, b) ->
        let via_einsum = Nx.item [] (Nx.einsum "i,i->" [| a; b |]) in
        let via_sum_mul = Nx.item [] (Nx.sum (Nx.mul a b)) in
        Float.abs (via_einsum -. via_sum_mul) < 1e-3);
    (* einsum outer product *)
    prop "einsum i,j->ij = outer" mk_f32_outer_pair (fun (a, b) ->
        let via_einsum = Nx.einsum "i,j->ij" [| a; b |] in
        let via_outer =
          Nx.mul (Nx.reshape [| Nx.numel a; 1 |] a) (Nx.reshape [| 1; Nx.numel b |] b)
        in
        allclose ~atol:1e-4 ~rtol:1e-4 via_einsum via_outer);
    (* einsum total sum = Nx.sum *)
    prop "einsum ij-> = sum" f32_2d (fun a ->
        let via_einsum = Nx.item [] (Nx.einsum "ij->" [| a |]) in
        let via_sum = Nx.item [] (Nx.sum a) in
        Float.abs (via_einsum -. via_sum) < 1e-3);
    (* einsum row sum = sum axis 1 *)
    prop "einsum ij->i = sum axis 1" f32_2d (fun a ->
        let via_einsum = Nx.einsum "ij->i" [| a |] in
        let via_sum = Nx.sum ~axes:[1] a in
        allclose ~atol:1e-4 ~rtol:1e-4 via_einsum via_sum);
    (* einsum col sum = sum axis 0 *)
    prop "einsum ij->j = sum axis 0" f32_2d (fun a ->
        let via_einsum = Nx.einsum "ij->j" [| a |] in
        let via_sum = Nx.sum ~axes:[0] a in
        allclose ~atol:1e-4 ~rtol:1e-4 via_einsum via_sum);
    (* einsum hadamard = elementwise mul *)
    prop "einsum i,i->i = mul" mk_f32_1d_pair (fun (a, b) ->
        let via_einsum = Nx.einsum "i,i->i" [| a; b |] in
        let via_mul = Nx.mul a b in
        approx_equal via_einsum via_mul);
    (* einsum Frobenius inner product *)
    prop "einsum ij,ij-> = sum(mul)"
      (let gen =
         let open Gen in
         let* shape = gen_shape_2d ~max_dim:5 in
         let+ a = gen_f32 shape and+ b = gen_f32 shape in
         (a, b)
       in
       Testable.make
         ~pp:(pp_pair pp_tensor pp_tensor)
         ~equal:(fun (a1, b1) (a2, b2) ->
           approx_equal a1 a2 && approx_equal b1 b2)
         ~gen ())
      (fun (a, b) ->
        let via_einsum = Nx.item [] (Nx.einsum "ij,ij->" [| a; b |]) in
        let via_sum_mul = Nx.item [] (Nx.sum (Nx.mul a b)) in
        Float.abs (via_einsum -. via_sum_mul) < 1e-3);
    (* einsum matvec = matmul with reshaped vector *)
    prop "einsum ij,j->i = matvec"
      (let gen =
         let open Gen in
         let* m = int_range 1 6 in
         let* n = int_range 1 6 in
         let+ a = gen_f32 [| m; n |] and+ b = gen_f32 [| n |] in
         (a, b)
       in
       Testable.make
         ~pp:(pp_pair pp_tensor pp_tensor)
         ~equal:(fun (a1, b1) (a2, b2) ->
           approx_equal a1 a2 && approx_equal b1 b2)
         ~gen ())
      (fun (a, b) ->
        let via_einsum = Nx.einsum "ij,j->i" [| a; b |] in
        let via_matmul =
          Nx.reshape [| (Nx.shape a).(0) |]
            (Nx.matmul a (Nx.reshape [| Nx.numel b; 1 |] b))
        in
        allclose ~atol:1e-4 ~rtol:1e-4 via_einsum via_matmul);
  ]

(* ── Stress Tests: Strided Views, Non-Contiguous Ops, High Rank ── *)

let stress_config =
  { Windtrap_prop.Prop.default_config with count = 500; max_gen = 1500 }

let stress_props =
  [
    (* Transpose then slice, verify data integrity *)
    prop ~config:stress_config "transpose+slice preserves data (f32)" f32_2d_plus
      (fun t ->
        let tr = Nx.transpose t in
        let spec = List.init (Nx.ndim tr) (fun _ -> Nx.A) in
        let sliced = Nx.slice spec tr in
        approx_equal (Nx.contiguous sliced) (Nx.contiguous tr));
    (* Transpose+slice then flatten vs direct flatten of transpose *)
    prop ~config:stress_config "transpose+contiguous = contiguous+transpose data (f32)"
      f32_2d_plus (fun t ->
        let a = Nx.to_array (Nx.contiguous (Nx.transpose t)) in
        let b = Nx.to_array (Nx.transpose t |> Nx.contiguous) in
        a = b);
    (* Slice a non-trivial range after transpose, check item access *)
    prop ~config:stress_config "item on transposed view (f32)" f32_2d_plus
      (fun t ->
        let s = Nx.shape t in
        let tr = Nx.transpose t in
        let ts = Nx.shape tr in
        (* item [0, ..., 0] of transpose should equal item [0, ..., 0] of
           original since both index the same element *)
        let zeros_orig = List.init (Array.length s) (fun _ -> 0) in
        let zeros_tr = List.init (Array.length ts) (fun _ -> 0) in
        Float.equal (Nx.item zeros_orig t) (Nx.item zeros_tr tr));
    (* Flip + slice: flip is a strided view, slicing it compounds strides *)
    prop ~config:stress_config "flip+slice data integrity (f32)" f32_2d_plus
      (fun t ->
        let flipped = Nx.flip t in
        let spec = [ Nx.R (0, (Nx.shape flipped).(0)) ] in
        let sliced = Nx.slice spec flipped in
        approx_equal (Nx.contiguous sliced) (Nx.contiguous flipped));
    (* Double transpose on high-rank tensor *)
    prop ~config:stress_config "double transpose high rank (f32)" f32_stress
      (fun t ->
        assume (Nx.ndim t >= 2);
        approx_equal (Nx.transpose (Nx.transpose t)) t);
    (* Contiguous on strided views: transpose then contiguous should equal
       copy of transpose *)
    prop ~config:stress_config "contiguous of strided view (f32)" f32_2d_plus
      (fun t ->
        let tr = Nx.transpose t in
        let c = Nx.contiguous tr in
        Nx.is_c_contiguous c && approx_equal c tr);
    (* Arithmetic on non-contiguous views *)
    prop ~config:stress_config "add on transposed views (f32)" f32_stress_pair
      (fun (a, b) ->
        assume (Nx.ndim a >= 2);
        let at = Nx.transpose a in
        let bt = Nx.transpose b in
        let sum_then_transpose = Nx.transpose (Nx.add a b) in
        let transpose_then_sum = Nx.add at bt in
        approx_equal sum_then_transpose transpose_then_sum);
    (* Reduction on transposed view *)
    prop ~config:stress_config "sum of transpose = sum of original (f32)"
      f32_stress (fun t ->
        assume (all_finite t);
        let s1 = Nx.item [] (Nx.sum t) in
        let s2 = Nx.item [] (Nx.sum (Nx.transpose t)) in
        Float.abs (s1 -. s2) < 1e-2);
    (* Broadcasting + arithmetic on high-rank tensors *)
    prop ~config:stress_config "mul broadcast high rank (f32)"
      f32_broadcastable_stress (fun (a, b) ->
        let result = Nx.mul a b in
        let a', b' = Nx.broadcasted a b in
        approx_equal result (Nx.mul a' b'));
    (* Slice with step on high-rank tensor *)
    prop ~config:stress_config "slice with step roundtrip (f32)" f32_stress
      (fun t ->
        assume (Nx.ndim t >= 1 && (Nx.shape t).(0) >= 2);
        let dim0 = (Nx.shape t).(0) in
        let sliced = Nx.slice [ Nx.Rs (0, dim0, 2) ] t in
        let expected_len = (dim0 + 1) / 2 in
        (Nx.shape sliced).(0) = expected_len
        && Nx.ndim sliced = Nx.ndim t);
    (* Copy of a strided view preserves data *)
    prop ~config:stress_config "copy strided view (f32)" f32_2d_plus (fun t ->
        let tr = Nx.transpose t in
        let c = Nx.copy tr in
        approx_equal c tr && Nx.is_c_contiguous c);
    (* Reshape after contiguous on strided view *)
    prop ~config:stress_config "reshape contiguous strided (f32)" f32_2d_plus
      (fun t ->
        let tr = Nx.contiguous (Nx.transpose t) in
        let flat = Nx.reshape [| Nx.numel t |] tr in
        Nx.numel flat = Nx.numel t
        && Nx.to_array flat = Nx.to_array tr);
  ]

(* ── Suite ── *)

let () =
  run "Nx Properties"
    [
      group "Arithmetic" arithmetic_props;
      group "Shape" shape_props;
      group "Comparison" comparison_props;
      group "Logical & Bitwise" logical_bitwise_props;
      group "Rounding" rounding_props;
      group "Sorting" sorting_props;
      group "Math Functions" math_function_props;
      group "Reductions" reduction_props;
      group "Linear Algebra" linalg_props;
      group "Concatenation" concat_props;
      group "Indexing" indexing_props;
      group "Broadcasting" broadcasting_props;
      group "Einsum" einsum_props;
      group "Stress Tests" stress_props;
    ]
