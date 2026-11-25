(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Minimize. The trusted side is minima known in closed form, the implicit
   derivative of a minimum written by hand, and the bounds jera.mli states. *)

open Windtrap
open Jera

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let one = Nx.Ptree.tensor
let tight = Tol.v ~rel:1e-10 ~abs:1e-12

(* D6: a raise names the jera function; its other words are not promised. *)
let raises_in fn f = raises_match (Exn.invalid_arg ~sub:fn) f

let centers =
  Gen.(
    map
      (fun l -> vec (Array.of_list l))
      (list ~size:(int_range 1 6) (float_range (-3.) 3.)))
  |> Gen.with_pp Nx.pp

(* (x − c)² + (x − c)⁴ has its minimum at c. A minimum is determined to about
   the square root of f's rounding over its curvature, so estimates agree with c
   to about 1e-8. *)
let bowl c x =
  let d = Nx.sub x c in
  Nx.add (Nx.square d) (Nx.square (Nx.square d))

(* A bowl's minimum, estimated to about 1e-8 by the gradient methods under
   [tight]. *)
let near () = Oracle.tensor ~abs:3e-8 ()

(* A convex problem [½ xᵀ a x + Σ log cosh x − bᵀ x] of [n] unknowns: [a] has
   entries in [-1, 1] plus [n + 1] on its diagonal, symmetrised, so its Hessian
   [a + diag (sech² x)] is positive-definite everywhere. An element of [b] below
   [10⁻³⁰⁰] is zero: a minimum between subnormals moves by steps the floats
   cannot resolve, and the search stalls there, as it states. *)
let convex =
  let open Gen in
  (let* n = int_range 0 6 in
   let+ a = list ~size:(constant (n * n)) (float_range (-1.) 1.)
   and+ b = list ~size:(constant n) (float_range (-5.) 5.) in
   let a = Nx.create f64 [| n; n |] (Array.of_list a) in
   let a =
     Nx.add
       (Nx.div_s (Nx.add a (Nx.transpose a)) 2.)
       (Nx.mul_s (Nx.eye f64 n) (Float.of_int (n + 1)))
   in
   let b = List.map (fun v -> if Float.abs v < 1e-300 then 0. else v) b in
   (a, Nx.create f64 [| n |] (Array.of_list b)))
  |> with_pp (fun ppf (a, b) ->
      Format.fprintf ppf "a = %a@ b = %a" Nx.pp a Nx.pp b)

let objective a b x =
  Nx.sub
    (Nx.add
       (Nx.mul_s (Nx.sum (Nx.mul x (Nx.matmul a x))) 0.5)
       (Nx.sum (Nx.log (Nx.cosh x))))
    (Nx.sum (Nx.mul b x))

let gradient a b x = Nx.sub (Nx.add (Nx.matmul a x) (Nx.tanh x)) b
let hessian a x = Nx.add a (Nx.diag (Nx.square (Nx.recip (Nx.cosh x))))

let gradient_methods =
  [
    ("bfgs", Minimize.bfgs ~linear:Linear.dense);
    ("lbfgs", Minimize.lbfgs ~memory:3 ~linear:Linear.dense);
    ("newton", Minimize.newton ~linear:Linear.dense);
    ( "newton-cg",
      Minimize.newton
        ~linear:(Linear.cg ~rel:1e-10 ~budget:50 ~precondition:Fun.id) );
  ]

(* The methods whose [linear] can solve a box's derivative, a system that is not
   symmetric: all but newton with [Linear.cg]. *)
let dense_methods =
  [
    ("bfgs", Minimize.bfgs ~linear:Linear.dense);
    ("lbfgs", Minimize.lbfgs ~memory:3 ~linear:Linear.dense);
    ("newton", Minimize.newton ~linear:Linear.dense);
  ]

let quasi_newton =
  [
    ("bfgs", Minimize.bfgs ~linear:Linear.dense);
    ("lbfgs", Minimize.lbfgs ~memory:3 ~linear:Linear.dense);
  ]

(* The quasi-Newton methods of the bound's laws, with L-BFGS of one pair, whose
   last steps contract slowest. *)
let bound_methods =
  ("lbfgs 1", Minimize.lbfgs ~memory:1 ~linear:Linear.dense) :: quasi_newton

let minimum m f x0 = Minimize.solve one m ~tol:tight ~budget:200 f x0

let rosenbrock v =
  let x = Nx.get [ 0 ] v and y = Nx.get [ 1 ] v in
  Nx.add
    (Nx.mul_s (Nx.square (Nx.sub y (Nx.square x))) 100.)
    (Nx.square (Nx.rsub_s 1. x))

let converged s = Nx.item [] (Solution.ok s)

let minima_tests =
  [
    prop "a converged answer is the zero of rune's gradient"
      ~examples:[ (Nx.zeros f64 [| 0; 0 |], vec [||]) ]
      convex
      (fun (a, b) ->
        cover "an empty problem" (Nx.dim 0 b = 0);
        cover "several unknowns" (Nx.dim 0 b > 2);
        List.iter
          (fun (_, m) ->
            let x =
              Solution.get (minimum m (objective a b) (Nx.zeros_like b))
            in
            (* The error is an estimate from the last steps' contraction, which
               can be about ten times optimistic for a quasi-Newton method whose
               rate jumps between steps. A distance of ten times [tol]'s scale,
               about 10⁻⁹ here, times a curvature of at most [n + 3] bounds the
               gradient by [10⁻⁸ (n + 2)] to spare. *)
            equal
              (Oracle.tensor ~abs:(1e-8 *. Float.of_int (Nx.dim 0 b + 2)) ())
              (Nx.zeros_like b) (gradient a b x))
          gradient_methods);
    test "a gradient stated with Rune.custom_jvp is the one searched" (fun () ->
        (* The value is detached, so rune's own gradient is 0 everywhere and the
           start would be its zero; the stated gradient [2 (x − c)] has its zero
           at [c]. *)
        let c = vec [| 1.; -0.5 |] in
        let f =
          Rune.custom_jvp one one (fun x ->
              ( Rune.detach (Nx.sum (Nx.square (Nx.sub x c))),
                fun dx -> Nx.sum (Nx.mul (Nx.mul_s (Nx.sub x c) 2.) dx) ))
        in
        equal (near ()) c
          (Solution.get
             (minimum (Minimize.bfgs ~linear:Linear.dense) f (vec [| 0.; 0. |]))));
    test "a structure's float tensors are its vector, its other tensors carried"
      (fun () ->
        let s = Nx.Ptree.(pair (pair tensor tensor) tensor) in
        let f ((p, q), _) =
          Nx.add
            (Nx.sum (Nx.square (Nx.sub_s p 2.)))
            (Nx.square (Nx.add_s q 1.))
        in
        let (p, q), k =
          Solution.get
            (Minimize.solve s
               (Minimize.bfgs ~linear:Linear.dense)
               ~tol:tight ~budget:100 f
               ((Nx.zeros f64 [| 3 |], Nx.scalar f64 0.), Nx.scalar Nx.int32 7l))
        in
        equal (Oracle.tensor ~rel:1e-9 ()) (Nx.full f64 [| 3 |] 2.) p;
        equal (Oracle.tensor ~rel:1e-9 ()) (Nx.scalar f64 (-1.)) q;
        equal int32 7l (Nx.item [] k));
    cases "a float32 problem is solved in float32" ~id:fst
      [
        ("bfgs", Minimize.bfgs ~linear:Linear.dense);
        ("lbfgs", Minimize.lbfgs ~memory:3 ~linear:Linear.dense);
        ("newton", Minimize.newton ~linear:Linear.dense);
        ("nelder_mead", Minimize.nelder_mead);
      ]
      (fun (_, m) ->
        (* float32's values fix a minimum to about √ε₃₂ ≈ 3.5e-4 (jera.mli,
           Tol.ulps), so [tol] is 1e-3. A converged answer's error estimate is
           within [tol]'s scale, at most 2e-3 here, in root mean square over two
           components, and an estimate: twice that bounds it. *)
        let c = Nx.create Nx.float32 [| 2 |] [| 1.; -0.5 |] in
        let s =
          Minimize.solve one m
            ~tol:(Tol.v ~rel:1e-3 ~abs:1e-3)
            ~budget:200
            (fun x -> Nx.sum (bowl c x))
            (Nx.zeros Nx.float32 [| 2 |])
        in
        equal (Oracle.tensor ~abs:4e-3 ()) c (Solution.get s));
    cases "zero unknowns converge at once" ~id:fst
      [
        ("bfgs", Minimize.bfgs ~linear:Linear.dense);
        ("lbfgs", Minimize.lbfgs ~memory:3 ~linear:Linear.dense);
        ("newton", Minimize.newton ~linear:Linear.dense);
        ("nelder_mead", Minimize.nelder_mead);
      ]
      (fun (_, m) ->
        let s =
          Minimize.solve one m ~tol:tight ~budget:10
            (fun x -> Nx.sum (Nx.square x))
            (vec [||])
        in
        equal Oracle.status_t Converged (Oracle.status s);
        equal (Oracle.tensor ()) (vec [||]) (Solution.best s));
    cases "Rosenbrock's minimum from (−1.2, 1) is (1, 1)"
      ~id:(fun (n, _, _, _) -> n)
      [
        (* jera.mli's example, with its tolerance and budget. *)
        ( "bfgs",
          Minimize.bfgs ~linear:Linear.dense,
          Tol.v ~rel:1e-8 ~abs:1e-10,
          100 );
        ("lbfgs", Minimize.lbfgs ~memory:3 ~linear:Linear.dense, tight, 500);
        ("lbfgs 1", Minimize.lbfgs ~memory:1 ~linear:Linear.dense, tight, 2000);
        ("newton", Minimize.newton ~linear:Linear.dense, tight, 500);
      ]
      (fun (_, m, tol, budget) ->
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (vec [| 1.; 1. |])
          (Solution.get
             (Minimize.solve one m ~tol ~budget rosenbrock (vec [| -1.2; 1. |]))));
    cases "vmap gives each lane its own search" ~id:fst
      (("nelder_mead", Minimize.nelder_mead) :: gradient_methods) (fun (_, m) ->
        let f c =
          Solution.get
            (minimum m
               (fun x ->
                 Nx.add
                   (Nx.sum (Nx.square (Nx.sub x c)))
                   (Nx.sum (Nx.square (Nx.square x))))
               (Nx.zeros_like c))
        in
        let cs = Nx.create f64 [| 3; 2 |] [| 1.; -0.5; 0.; 0.; -2.; 3. |] in
        equal (Oracle.tensor ())
          (Nx.stack (List.init 3 (fun i -> f (Nx.get [ i ] cs))))
          (Rune.vmap' f cs));
    cases "compiled bfgs and lbfgs are eager, bit for bit" ~id:fst quasi_newton
      (fun (_, m) ->
        Compiled.here ();
        let f c =
          Solution.get
            (minimum m
               (fun x ->
                 Nx.add
                   (Nx.sum (Nx.square (Nx.sub x c)))
                   (Nx.sum (Nx.square (Nx.square x))))
               (Nx.zeros_like c))
        in
        let c = vec [| 1.; -0.5; 2. |] in
        equal (Oracle.tensor ()) (f c) (Rune.jit' f c));
    cases "compiled newton converges where eager does" ~id:fst
      [
        ("newton", Minimize.newton ~linear:Linear.dense);
        ( "newton-cg",
          Minimize.newton
            ~linear:(Linear.cg ~rel:1e-10 ~budget:50 ~precondition:Fun.id) );
      ]
      (fun (_, m) ->
        Compiled.here ();
        (* Its factorisations round within rune's numerics compiled (jera.mli,
           Tol), so both answers meet [tight] apart: each within its scale,
           [1e-10 |x| + 1e-12] in root mean square over three components, so
           within [2 √3] scales of each other. *)
        let f c =
          Solution.get
            (minimum m
               (fun x ->
                 Nx.add
                   (Nx.sum (Nx.square (Nx.sub x c)))
                   (Nx.sum (Nx.square (Nx.square x))))
               (Nx.zeros_like c))
        in
        let c = vec [| 1.; -0.5; 2. |] in
        equal (Oracle.tensor ~rel:4e-10 ~abs:4e-12 ()) (f c) (Rune.jit' f c));
  ]

(* The magnitude of a zero's component: in [0.5, 2], 0, below the least normal
   float, where the floats resolve it only to the subnormals' spacing and an
   answer within a tolerance in ulps is the zero itself, or tiny and normal. *)
let magnitude =
  let normal = Gen.float_range 0.5 2. in
  Gen.one_of
    [
      normal;
      normal;
      normal;
      Gen.constant 0.;
      Gen.map (fun u -> Float.ldexp u (-1040)) (Gen.float_range 0.5 2.);
      Gen.map (fun u -> Float.ldexp u (-960)) (Gen.float_range 0.5 2.);
    ]

(* Covers for the kinds of zero [magnitude] draws, in a vector [z]. *)
let kinds z =
  let any p = Array.exists p (Oracle.floats z) in
  cover "a zero at 0" (any (fun v -> v = 0.));
  cover "a subnormal zero"
    (any (fun v -> v <> 0. && Float.abs v < Float.min_float));
  cover "a tiny normal zero"
    (any (fun v -> Float.abs v >= Float.min_float && Float.abs v < 1e-280))

(* The spacing of the subnormals, float64's absolute resolution. *)
let spacing = Float.ldexp 1. (-1074)

(* [z + d r]: within [r] of a zero [z] of normal magnitude, within a thousand
   subnormal spacings of a smaller one. *)
let start_at z d r =
  if Float.abs z < Float.min_float then z +. (d *. 1000. *. spacing)
  else z +. (d *. r)

(* [Σ dᵢ (xᵢ − cᵢ)² / 2] with curvatures [d] in [e^±3] and [c] of {!magnitude}s
   times [2^e], from {!start_at} [c] within [2^e] in each component, and a
   tolerance of [k] ulps: its minimum is [c], exactly. *)
let quadratics =
  let open Gen in
  (let* n = int_range 2 6 in
   (* Half the draws have every component normal: a zero or subnormal one is
      below the resolution of any tolerance in ulps, and only these can
      converge. Half of those are scaled by [2^e] below 10⁻¹⁵⁶, where [tol]'s
      scale squared is below the floats' range. *)
   let* all_normal = bool in
   let* e =
     if all_normal then one_of [ constant 0; int_range (-1015) (-520) ]
     else constant 0
   in
   let+ d = list ~size:(constant n) (float_range (-3.) 3.)
   and+ c =
     list ~size:(constant n)
       (if all_normal then float_range 0.5 2. else magnitude)
   and+ signs = list ~size:(constant n) bool
   and+ offsets = list ~size:(constant n) (float_range (-1.) 1.)
   and+ k = float_range 4. 64. in
   let c =
     Array.of_list
       (List.map2 (fun c s -> Float.ldexp (if s then c else -.c) e) c signs)
   in
   let x0 =
     Array.mapi
       (fun i o -> start_at c.(i) o (Float.ldexp 1. e))
       (Array.of_list offsets)
   in
   (vec (Array.of_list (List.map exp d)), (vec c, (vec x0, k))))
  |> with_pp (fun ppf (d, (c, (x0, k))) ->
      Format.fprintf ppf "d = %a@ c = %a@ x0 = %a@ k = %g" Nx.pp d Nx.pp c Nx.pp
        x0 k)

(* The ratio of the largest curvature to the least of [Σ dᵢ (xᵢ − cᵢ)² / 2] in
   the coordinates of a tolerance in ulps, [dᵢ cᵢ²], over the components that
   are normal floats: 1 where none is. *)
let scaled_condition d c =
  let logs =
    List.filter_map
      (fun (d, c) ->
        if Float.abs c < Float.min_float then None
        else Some (log d +. (2. *. log (Float.abs c))))
      (List.combine
         (Array.to_list (Oracle.floats d))
         (Array.to_list (Oracle.floats c)))
  in
  match logs with
  | [] -> 1.
  | l ->
      exp
        (List.fold_left Float.max Float.neg_infinity l
        -. List.fold_left Float.min Float.infinity l)

(* [½ (x₁ − c₁)² + ½ 10⁶ (x₂ − c₂)²] from [x0] under [Tol.rel 1e-8]: each
   quasi-Newton method converges, its answer within [tol] and its last step's
   one further. *)
let stiff_soft ~c ~x0 =
  let c = vec c in
  let d = vec [| 1.; 1e6 |] in
  let f x = Nx.mul_s (Nx.sum (Nx.mul d (Nx.square (Nx.sub x c)))) 0.5 in
  List.iter
    (fun (name, m) ->
      let s = Minimize.solve one m ~tol:(Tol.rel 1e-8) ~budget:200 f (vec x0) in
      equal ~msg:name Oracle.status_t Converged (Oracle.status s);
      let x = Solution.best s in
      at_most float_exact ~msg:name ~than:2.
        (Oracle.scaled ~rel:1e-8 ~abs:0. ~y:x (Nx.abs (Nx.sub x c))))
    bound_methods

(* [Σ dᵢ (xᵢ − cᵢ)² / 2] with curvatures [d] in [e^±7] and [|c|] in [0.5, 2],
   from a warm start: each component of curvature below the median within [10⁻⁷]
   to [10⁻³] of [cᵢ], relatively, and each other within 1. Its minimum is [c],
   exactly. *)
let warm_starts =
  let open Gen in
  (let* n = int_range 2 6 in
   let+ d = list ~size:(constant n) (float_range (-7.) 7.)
   and+ c = list ~size:(constant n) (pair (float_range 0.5 2.) bool)
   and+ near = list ~size:(constant n) (pair (float_range (-7.) (-3.)) bool)
   and+ far = list ~size:(constant n) (float_range (-1.) 1.) in
   let d = Array.of_list (List.map exp d) in
   let c = Array.of_list (List.map (fun (c, s) -> if s then c else -.c) c) in
   let median =
     let s = Array.copy d in
     Array.sort Float.compare s;
     s.(n / 2)
   in
   let near = Array.of_list near and far = Array.of_list far in
   let x0 =
     Array.mapi
       (fun i c ->
         if d.(i) < median then
           let e, s = near.(i) in
           c *. (1. +. ((if s then 1. else -1.) *. (10. ** e)))
         else c +. far.(i))
       c
   in
   (vec d, (vec c, vec x0)))
  |> with_pp (fun ppf (d, (c, x0)) ->
      Format.fprintf ppf "d = %a@ c = %a@ x0 = %a" Nx.pp d Nx.pp c Nx.pp x0)

(* A tolerance in ulps: below one, finer than the floats can always show, or
   from 1 to 10⁶, log-uniform. *)
let ulps_k =
  Gen.(
    one_of [ float_range 0.25 1.; map (fun e -> 10. ** e) (float_range 0. 6.) ])

let times = Nx.linspace f64 0. 1. 20

(* a e^(b t) at [times], the model of the fits. *)
let model p = Nx.mul (Nx.get [ 0 ] p) (Nx.exp (Nx.mul (Nx.get [ 1 ] p) times))
let lm = Minimize.levenberg_marquardt one ~linear:Linear.dense

(* An exponential and the frequency and amplitude of the sine added to its
   samples: residuals of a tenth of the signal or more, where a fit's cost
   changes by less than its rounding before its steps meet [tol]. *)
let noisy =
  Gen.(
    let+ a = float_range 0.5 3.
    and+ b = float_range (-2.) 1.
    and+ noise = float_range 3. 40.
    and+ amp = float_range 0.3 1. in
    (vec [| a; b |], (noise, amp)))
  |> Gen.with_pp (fun ppf (p, (noise, amp)) ->
      Format.fprintf ppf "p = %a@ sine of frequency %g, amplitude %g" Nx.pp p
        noise amp)

let noisy_data p (noise, amp) =
  Nx.add (model p) (Nx.mul_s (Nx.sin (Nx.mul_s times noise)) amp)

(* The ulps of [p]'s smaller parameter that the fit of [a e^(b t)] at [p] is
   fixed to: [κ |p|∞ / |p|min], [κ] the condition of the Jacobian, whose columns
   are [e^(b t)] and [a t e^(b t)], from the eigenvalues of its 2 × 2 Gram
   matrix. *)
let resolution_ulps p =
  let a = Nx.item [ 0 ] p and b = Nx.item [ 1 ] p in
  let ts = Oracle.floats times in
  let sum f = Array.fold_left (fun acc t -> acc +. f t) 0. ts in
  let e t = exp (2. *. b *. t) in
  let s11 = sum e
  and s12 = sum (fun t -> a *. t *. e t)
  and s22 = sum (fun t -> a *. a *. t *. t *. e t) in
  let mean = (s11 +. s22) /. 2. in
  let gap = sqrt ((((s11 -. s22) /. 2.) ** 2.) +. (s12 *. s12)) in
  let kappa = sqrt ((mean +. gap) /. (mean -. gap)) in
  kappa
  *. Float.max (Float.abs a) (Float.abs b)
  /. Float.min (Float.abs a) (Float.abs b)

(* [½ xᵀ a x − bᵀ x + λᵀ x + c] with [b = a x*] exact: [a] has integer entries,
   a diagonal of [n + 1] to [2n + 2] and off-diagonal entries in [{−1, 0, 1}],
   so its condition is at most [(3n + 1) / 2] by Gershgorin's discs, and [x*]
   has components of [k / 64], a quarter of them 0. In a box [lo = 0] the other
   components are positive and [λ] is 1 at the zeros, whose bound is then
   active; without one [λ = 0]. Either way the minimum is [x*], exactly, and [c]
   is 0 or [10⁸], a large [|f*|]. *)
let exact_minima =
  let open Gen in
  (let* n = int_range 1 6 in
   let* boxed = bool in
   let* large = bool in
   let+ off = list ~size:(constant (n * n)) (int_range (-1) 1)
   and+ diag = list ~size:(constant n) (int_range (n + 1) ((2 * n) + 2))
   and+ xs =
     list ~size:(constant n)
       (pair (int_range 0 3) (pair (int_range 1 255) bool))
   and+ k = float_range 0.25 1. in
   let off = Array.of_list off and diag = Array.of_list diag in
   let a =
     Array.init (n * n) (fun ij ->
         let i = ij / n and j = ij mod n in
         if i = j then Float.of_int diag.(i)
         else Float.of_int off.((min i j * n) + max i j))
   in
   let xs =
     List.mapi
       (fun i (zero, (m, neg)) ->
         (* One component nonzero, so the minimum has a scale. *)
         if zero = 0 && i > 0 then 0.
         else
           let v = Float.of_int m /. 64. in
           if neg && not boxed then -.v else v)
       xs
   in
   ( Nx.create f64 [| n; n |] a,
     (vec (Array.of_list xs), (boxed, ((if large then 1e8 else 0.), k))) ))
  |> with_pp (fun ppf (a, (x, (boxed, (c, k)))) ->
      Format.fprintf ppf "a = %a@ x* = %a@ boxed %b@ c = %g@ k = %g" Nx.pp a
        Nx.pp x boxed c k)

(* As {!exact_minima} without a box, with components of [k / 64] scaled by
   [2^-e], [e] up to 32, so that they spread over ten decades and [b = a x*]
   stays exact. *)
let spread_minima =
  let open Gen in
  (let* n = int_range 1 6 in
   let+ off = list ~size:(constant (n * n)) (int_range (-1) 1)
   and+ diag = list ~size:(constant n) (int_range (n + 1) ((2 * n) + 2))
   and+ xs =
     list ~size:(constant n)
       (pair (int_range 0 3)
          (pair (int_range 1 255) (pair (int_range 0 32) bool)))
   in
   let off = Array.of_list off and diag = Array.of_list diag in
   let a =
     Array.init (n * n) (fun ij ->
         let i = ij / n and j = ij mod n in
         if i = j then Float.of_int diag.(i)
         else Float.of_int off.((min i j * n) + max i j))
   in
   let xs =
     List.mapi
       (fun i (zero, (m, (e, neg))) ->
         if zero = 0 && i > 0 then 0.
         else
           let v = Float.ldexp (Float.of_int m /. 64.) (-e) in
           if neg then -.v else v)
       xs
   in
   (Nx.create f64 [| n; n |] a, vec (Array.of_list xs)))
  |> with_pp (fun ppf (a, x) ->
      Format.fprintf ppf "a = %a@ x* = %a" Nx.pp a Nx.pp x)

let exact_objective a xstar ~boxed c =
  let b = Nx.matmul a xstar in
  let lambda =
    if boxed then Nx.cast f64 (Nx.equal_s xstar 0.) else Nx.zeros_like xstar
  in
  fun x ->
    Nx.add_s
      (Nx.add
         (Nx.sub
            (Nx.mul_s (Nx.sum (Nx.mul x (Nx.matmul a x))) 0.5)
            (Nx.sum (Nx.mul b x)))
         (Nx.sum (Nx.mul lambda x)))
      c

(* The condition of a symmetric positive-definite [a]: the ratio of its largest
   eigenvalue to its least. *)
let condition a =
  let l = Oracle.floats (Nx.eigvalsh f64 a) in
  Array.fold_left Float.max 0. l /. Array.fold_left Float.min infinity l

(* [c + ½ (x − a)ᵀ h (x − a)] on 10 unknowns, [h] of condition 10⁴ in a random
   basis. *)
let rotated =
  let n = 10 in
  Gen.(
    let+ m = list ~size:(constant (n * n)) (float_range (-1.) 1.)
    and+ a = list ~size:(constant n) (float_range 0.5 2.5) in
    let q, _ = Nx.qr (Nx.create f64 [| n; n |] (Array.of_list m)) in
    let d =
      Nx.create f64 [| 1; n |]
        (Array.init n (fun i -> 1e4 ** (Float.of_int i /. Float.of_int (n - 1))))
    in
    let h = Nx.matmul (Nx.mul q d) (Nx.transpose q) in
    (Nx.div_s (Nx.add h (Nx.transpose h)) 2., vec (Array.of_list a)))
  |> Gen.with_pp (fun ppf (h, a) ->
      Format.fprintf ppf "h = %a@ a = %a" Nx.pp h Nx.pp a)

(* Each method's solve of [c + ½ (x − a)ᵀ h (x − a)] from 0, compiled, since an
   eager one on {!rotated}'s ten unknowns takes about a tenth of a second: one
   program per method, which [c] does not change. *)
let rotated_solves =
  List.map
    (fun (name, m) ->
      ( name,
        Rune.jit
          Nx.Ptree.(
            tensor @-> tensor @-> tensor @-> returns (Solution.ptree tensor))
          (fun h a c ->
            let f x =
              let d = Nx.sub x a in
              Nx.add (Nx.mul_s (Nx.sum (Nx.mul d (Nx.matmul h d))) 0.5) c
            in
            Minimize.solve one m ~tol:tight ~budget:2000 f (Nx.zeros_like a)) ))
    gradient_methods

let error_tests =
  [
    test
      "budget iterations end it Budget_spent, and get names Jera.Minimize.solve"
      (fun () ->
        let s =
          Minimize.solve one
            (Minimize.lbfgs ~memory:3 ~linear:Linear.dense)
            ~tol:tight ~budget:3 rosenbrock
            (vec [| -1.2; 1. |])
        in
        equal Oracle.status_t Budget_spent (Oracle.status s);
        raises_match (Exn.failure ~sub:"Jera.Minimize.solve") (fun () ->
            Solution.get s));
    cases "a start where f is not finite ends the lane Not_finite" ~id:fst
      [
        ("bfgs", Minimize.bfgs ~linear:Linear.dense);
        ("newton", Minimize.newton ~linear:Linear.dense);
        ("nelder_mead", Minimize.nelder_mead);
      ]
      (fun (_, m) ->
        equal Oracle.status_t Not_finite
          (Oracle.status
             (minimum m (fun x -> Nx.sum (Nx.sqrt x)) (vec [| -1. |]))));
    test "a start where the gradient is not finite ends the lane Not_finite"
      (fun () ->
        (* √|x| is 0 at 0, where its slope is infinite. *)
        equal Oracle.status_t Not_finite
          (Oracle.status
             (minimum
                (Minimize.bfgs ~linear:Linear.dense)
                (fun x -> Nx.sum (Nx.sqrt (Nx.abs x)))
                (vec [| 0. |]))));
    test "an objective unbounded below does not converge" (fun () ->
        let s =
          Minimize.solve one
            (Minimize.bfgs ~linear:Linear.dense)
            ~tol:tight ~budget:50
            (fun x -> Nx.sum (Nx.neg (Nx.exp x)))
            (vec [| 0. |])
        in
        not_equal Oracle.status_t Converged (Oracle.status s));
    prop "a converged gradient method's error estimate meets tol"
      Gen.(pair convex ulps_k)
      (fun ((a, b), k) ->
        (* Below an ulp a search whose last steps contract slowly, as L-BFGS's
           do, reaches the floats' resolution short of [tol] and stalls. The
           solve adds pairwise and [Oracle.ulps] in order: they differ by
           rounding. *)
        cover "below an ulp" (k < 1.);
        cover "above 10³ ulps" (k > 1e3);
        List.iter
          (fun (name, m) ->
            let s =
              Minimize.solve one m ~tol:(Tol.ulps k) ~budget:200 (objective a b)
                (Nx.zeros_like b)
            in
            cover (name ^ " converges") (converged s);
            if converged s then
              at_most float_exact ~msg:name ~than:(1. +. 1e-12)
                (Oracle.ulps ~k ~y:(Solution.best s) (Solution.error s)))
          gradient_methods);
    prop ~count:60
      "a converged quasi-Newton answer on a quadratic is within κ + 1 + 1/k \
       tolerances"
      ~examples:
        [
          (* Squares of [tol]'s scale at this minimum are below the floats'
             range, as are the gradient's products with it. *)
          ( vec [| 8.417159381161952; 0.5884126868560854 |],
            ( vec [| -5.1306710016229703e-290; -1.0261342003245941e-289 |],
              (vec [| 1.; 0.021989514425253542 |], 4.) ) );
          (* A zero at 0 and a subnormal one, each started within a thousand
             subnormal spacings, beside a normal one. *)
          ( Nx.exp (vec [| 0.5; -1.; 2. |]),
            ( vec [| 1.3; 0.; Float.ldexp (-1.1) (-1040) |],
              ( vec
                  [|
                    1.7;
                    500. *. spacing;
                    Float.ldexp (-1.1) (-1040) -. (700. *. spacing);
                  |],
                16. ) ) );
        ]
      quadratics
      (fun (d, (c, (x0, k))) ->
        (* In [tol]'s scaled coordinates the curvatures are [dᵢ σᵢ²], [σᵢ]
           proportional to [|cᵢ|] near the answer, and every curvature a step
           measures is a Rayleigh quotient, at least the least: the gradient's
           bound puts [x] within [κ] of [tol], [κ] the ratio of the largest
           curvature to the least, and the answer [x + δ] one [tol] and its
           rounding further. A zero at 0 or below the normal floats is met
           exactly or not at all. *)
        kinds c;
        cover "a minimum whose scale's square underflows"
          (Array.for_all
             (fun v -> v <> 0. && Float.abs v < 1e-156)
             (Oracle.floats c));
        let kappa = scaled_condition d c in
        let f x = Nx.mul_s (Nx.sum (Nx.mul d (Nx.square (Nx.sub x c)))) 0.5 in
        List.iter
          (fun (name, m) ->
            let s = Minimize.solve one m ~tol:(Tol.ulps k) ~budget:500 f x0 in
            cover (name ^ " converges") (converged s);
            if converged s then
              let x = Solution.best s in
              at_most float_exact ~msg:name
                ~than:(kappa +. 1. +. (1. /. k))
                (Oracle.ulps ~k ~y:x (Nx.abs (Nx.sub x c))))
          bound_methods);
    prop ~count:50 "a warm start's converged answer is within κ + 2 tolerances"
      ~examples:
        [
          (* Curvatures e^±5 apart by 2·10⁴, the soft component started 10⁻⁵
             from its minimum. *)
          ( Nx.exp (vec [| -5.; 5. |]),
            (vec [| 1.; -1.5 |], vec [| 1.00001; -1. |]) );
        ]
      warm_starts
      (fun (d, (c, x0)) ->
        (* Under [Tol.rel 1e-8] the curvatures in [tol]'s coordinates are [dᵢ
           (10⁻⁸ cᵢ)²] near the answer, and every curvature a step measures is a
           Rayleigh quotient, at least the least: the gradient's bound puts [x]
           within [κ] of [tol], [κ] the ratio of the largest of those curvatures
           to the least, and the answer one [tol] and its rounding further. *)
        let dc =
          Array.map2
            (fun d c -> d *. c *. c)
            (Oracle.floats d) (Oracle.floats c)
        in
        let kappa =
          Array.fold_left Float.max 0. dc
          /. Array.fold_left Float.min infinity dc
        in
        let spread =
          let d = Oracle.floats d in
          Array.fold_left Float.max 0. d /. Array.fold_left Float.min infinity d
        in
        cover "curvatures that differ by 10⁴ or more" (spread >= 1e4);
        let f x = Nx.mul_s (Nx.sum (Nx.mul d (Nx.square (Nx.sub x c)))) 0.5 in
        List.iter
          (fun (name, m) ->
            let s = Minimize.solve one m ~tol:(Tol.rel 1e-8) ~budget:500 f x0 in
            cover (name ^ " converges") (converged s);
            if converged s then
              let x = Solution.best s in
              at_most float_exact ~msg:name ~than:(kappa +. 2.)
                (Oracle.scaled ~rel:1e-8 ~abs:0. ~y:x (Nx.abs (Nx.sub x c))))
          bound_methods);
    test "a soft component left last converges within tol" (fun () ->
        (* The steps measure the stiff curvature 10⁶ while the stiff component
           converges; the soft one, 10⁻⁴ from its minimum, is then the error,
           along a curvature of 1. *)
        stiff_soft ~c:[| 1.; 1. |] ~x0:[| 1.0001; 2. |]);
    test "a stiff component near 0 converges within tol" (fun () ->
        stiff_soft ~c:[| 1.; 1e-3 |] ~x0:[| 1.00002; 1.001 |]);
    cases "Tol.rel 1e-10 stalls at a component of 1e-7 beside one of 1" ~id:fst
      gradient_methods (fun (_, m) ->
        (* jera.mli's instance: the gradient's rounding fixes each component to
           [ε κ |x|∞], 2e-16 here, coarser than 1e-10 of 1e-7. *)
        let a = Nx.create f64 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
        let f = exact_objective a (vec [| 1.; 1e-7 |]) ~boxed:false 0. in
        equal Oracle.status_t Stalled
          (Oracle.status
             (Minimize.solve one m ~tol:(Tol.rel 1e-10) ~budget:200 f
                (vec [| 0.5; 0.5 |]))));
    cases "an absolute term converges at a component of 1e-7 beside one of 1"
      ~id:fst gradient_methods (fun (_, m) ->
        let a = Nx.create f64 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
        let f = exact_objective a (vec [| 1.; 1e-7 |]) ~boxed:false 0. in
        equal Oracle.status_t Converged
          (Oracle.status
             (Minimize.solve one m
                ~tol:(Tol.v ~rel:1e-10 ~abs:1e-14)
                ~budget:200 f
                (vec [| 0.5; 0.5 |]))));
    test "newton under a tolerance below an ulp ends Stalled" (fun () ->
        (* At its minimum the gradient's rounding gives steps of 64 ulps of the
           small first component, which a tolerance of 0.27 ulps cannot
           follow. *)
        let a =
          Nx.create f64 [| 3; 3 |]
            [|
              0x1.bed1bf54c1e5ep+1;
              -0x1.23095b14ba6c1p-1;
              -0x1.2c8702f961b4p-4;
              -0x1.23095b14ba6c1p-1;
              0x1.058f40653edcp+2;
              -1.;
              -0x1.2c8702f961b4p-4;
              -1.;
              4.;
            |]
        and b =
          vec
            [|
              0x1.f8cf36f7fba8p-2; -0x1.c360e38ae8a33p+1; -0x1.7c3e8eb7c7248p+0;
            |]
        in
        List.iter
          (fun (name, m) ->
            if String.starts_with ~prefix:"newton" name then
              equal ~msg:name Oracle.status_t Stalled
                (Oracle.status
                   (Minimize.solve one m
                      ~tol:(Tol.ulps 0x1.116ff1e7a3881p-2)
                      ~budget:200 (objective a b) (Nx.zeros_like b))))
          gradient_methods);
    prop
      "under a tolerance below an ulp a search ends Stalled within 2 κ ε |x*|∞"
      ~examples:
        [
          (* Coupled, with zeros in its minimum: the gradient rounds to 0 at
             4e-17 from them. *)
          ( Nx.create f64 [| 5; 5 |]
              [|
                9.;
                0.;
                0.;
                -1.;
                0.;
                0.;
                6.;
                1.;
                -1.;
                -1.;
                0.;
                1.;
                12.;
                -1.;
                1.;
                -1.;
                -1.;
                -1.;
                12.;
                0.;
                0.;
                -1.;
                1.;
                0.;
                11.;
              |],
            ( vec [| 2.46875; 0.; -2.921875; 0.; 0.53125 |],
              (false, (0., 0x1.436881aeba15cp-1)) ) );
        ]
      exact_minima
      (fun (a, (xstar, (boxed, (c, k)))) ->
        let n = Nx.dim 0 xstar in
        cover "a zero component" (Array.mem 0. (Oracle.floats xstar));
        cover "an active bound" (boxed && Array.mem 0. (Oracle.floats xstar));
        cover "a large minimum value" (c > 0.);
        let kappa = Float.of_int ((3 * n) + 1) /. 2. in
        let top = Nx.item [] (Nx.max (Nx.abs xstar)) in
        let lo = if boxed then Some (Nx.zeros_like xstar) else None in
        (* The answer is fixed to about ε |x*|∞ times the condition: the
           gradient's rounding is ε times its largest terms. A converged answer
           is within its tolerance, up to the quadratics' law's factor. *)
        let check name s =
          let st = Oracle.status s and x = Solution.best s in
          not_equal ~msg:name Oracle.status_t Budget_spent st;
          if st = Stalled then
            at_most float_exact ~msg:name
              ~than:(2. *. kappa *. Float.epsilon *. top)
              (Nx.item [] (Nx.max (Nx.abs (Nx.sub x xstar))));
          if st = Converged then
            at_most float_exact ~msg:name
              ~than:(kappa +. 1. +. (1. /. k))
              (Oracle.ulps ~k ~y:x (Nx.abs (Nx.sub x xstar)))
        in
        let start = Nx.full_like xstar 0.5 in
        (* The residual [a x − a x*], whose fit is [x*] in or out of the box. *)
        let fit =
          let b = Nx.matmul a xstar in
          Minimize.solve one lm ?lo ~tol:(Tol.ulps k) ~budget:200
            (fun x -> Nx.sub (Nx.matmul a x) b)
            start
        in
        cover "levenberg_marquardt stalls" (Oracle.status fit = Stalled);
        check "levenberg_marquardt" fit;
        List.iter
          (fun (name, m) ->
            let s =
              Minimize.solve one m ?lo ~tol:(Tol.ulps k) ~budget:200
                (exact_objective a xstar ~boxed c)
                start
            in
            (* Newton meets a quadratic in one step; the test above stalls
               it. *)
            if List.mem_assoc name quasi_newton then
              cover (name ^ " stalls") (Oracle.status s = Stalled);
            check name s)
          gradient_methods);
    prop "an absolute term of 16 ε κ |x*|∞ converges every search"
      ~examples:
        [
          (* A component 2⁻³⁶ beside one of 1. *)
          ( Nx.create f64 [| 2; 2 |] [| 3.; 1.; 1.; 4. |],
            vec [| 1.; Float.ldexp 1. (-36) |] );
        ]
      spread_minima
      (fun (a, xstar) ->
        (* The converse of the resolution: an absolute term of [ε κ |x*|∞], [κ]
           the bound by Gershgorin's discs, is met however small a component is;
           16 of them leave room for the gradient's bound and the last step,
           each about one. The tolerance has no relative term: the gradient's
           bound is measured in [tol]'s scaled norm, where a gradient at its
           rounding bounds the error by the resolution times the ratio of the
           largest scale to the least. A relative term makes that ratio about
           [10⁻¹⁰ / (ε κ)] here, and the bound fails [tol] at a minimum the
           search reaches exactly. *)
        let n = Nx.dim 0 xstar in
        let kappa = Float.of_int ((3 * n) + 1) /. 2. in
        let top = Nx.item [] (Nx.max (Nx.abs xstar)) in
        let tol = Tol.abs (16. *. kappa *. Float.epsilon *. top) in
        cover "a component below 10⁻⁶ of the largest"
          (Array.exists
             (fun v -> v <> 0. && Float.abs v < 1e-6 *. top)
             (Oracle.floats xstar));
        List.iter
          (fun (name, m) ->
            equal ~msg:name Oracle.status_t Converged
              (Oracle.status
                 (Minimize.solve one m ~tol ~budget:200
                    (exact_objective a xstar ~boxed:false 0.)
                    (Nx.full_like xstar 0.5))))
          gradient_methods);
    test "the absolute term √(r ε κ) |x*|∞ converges at a minimum near 0"
      (fun () ->
        (* Its zero component starts 0.5 from the minimum; with an absolute term
           of [5 ε |x*|∞] alone beside [Tol.rel 1e-10], 5 the bound by
           Gershgorin's discs on [a]'s condition, both searches stall within
           0.0025 of [tol] of it. *)
        let a =
          Nx.create f64 [| 3; 3 |] [| 8.; 1.; 1.; 1.; 4.; 0.; 1.; 0.; 7. |]
        and xstar =
          vec [| 7.171183824539185e-08; 0.; -5.811452865600586e-07 |]
        in
        let abs =
          Float.sqrt (1e-10 *. Float.epsilon *. condition a)
          *. 5.811452865600586e-07
        in
        List.iter
          (fun (name, m) ->
            equal ~msg:name Oracle.status_t Converged
              (Oracle.status
                 (Minimize.solve one m ~tol:(Tol.v ~rel:1e-10 ~abs) ~budget:200
                    (exact_objective a xstar ~boxed:false 0.)
                    (Nx.full_like xstar 0.5))))
          gradient_methods);
    test "a move the gradient's rounding hides is at the floats' limit"
      (fun () ->
        (* L-BFGS reaches this minimum in nine steps. Each later move shifts
           only the components near 0, by ulps far finer than the gradient's
           rounding, which leaves the gradient unchanged: no curvature can be
           measured along it, and the lane must not search on to its budget. *)
        let a =
          Nx.create f64 [| 4; 4 |]
            [|
              10.;
              -1.;
              -1.;
              -1.;
              -1.;
              8.;
              1.;
              -1.;
              -1.;
              1.;
              10.;
              1.;
              -1.;
              -1.;
              1.;
              8.;
            |]
        and xstar =
          vec
            [| 0.01190185546875; 0.; 0.00128173828125; -4.831235855817795e-09 |]
        in
        let r = 1e-6 in
        let abs =
          Float.sqrt (r *. Float.epsilon *. condition a) *. 0.01190185546875
        in
        List.iter
          (fun (name, m) ->
            equal ~msg:name Oracle.status_t Converged
              (Oracle.status
                 (Minimize.solve one m ~tol:(Tol.v ~rel:r ~abs) ~budget:200
                    (exact_objective a xstar ~boxed:false 0.)
                    (Nx.full_like xstar 0.5))))
          gradient_methods);
    test "a small move the gradient shows is no floats' limit" (fun () ->
        (* Its stiff component learned, the model scales the soft one's step
           below [ε |x|∞]. That component's ulps are far finer and its gradient
           shows the move, so the search goes on to the minimum. *)
        let quad d xstar x =
          let e = Nx.sub x xstar in
          Nx.mul_s (Nx.sum (Nx.mul d (Nx.mul e e))) 0.5
        in
        let xstar = vec [| 1.; 1e-6 |] in
        List.iter
          (fun (stiff, e) ->
            List.iter
              (fun (name, m) ->
                equal ~msg:name
                  (Oracle.tensor ~abs:1e-9 ())
                  xstar
                  (Solution.get
                     (Minimize.solve one m
                        ~lo:(vec [| -10.; -10. |])
                        ~hi:(vec [| 10.; 10. |])
                        ~tol:(Tol.abs 1e-9) ~budget:200
                        (quad (vec [| stiff; 1. |]) xstar)
                        (vec [| 1.0001; 1e-6 +. e |]))))
              quasi_newton)
          [ (1e10, 1e-7); (1e10, 1e-6); (1e12, 1e-5) ]);
    test "a stretch where f is linear is no floats' limit" (fun () ->
        (* In a box the search accepts its longest step with no curvature
           condition. Beside a stiff large component that step on the linear
           stretch is below [ε |x|∞], and 5 from the minimum with a gradient of
           1 the lane has not converged. *)
        let top = 1e8 and stiff = 1e10 in
        let huber z =
          let a = Nx.abs z in
          Nx.where (Nx.less_equal_s a 1.)
            (Nx.mul_s (Nx.square z) 0.5)
            (Nx.sub_s a 0.5)
        in
        let f x =
          Nx.add
            (Nx.mul_s (Nx.square (Nx.sub_s (Nx.get [ 0 ] x) top)) (stiff /. 2.))
            (huber (Nx.sub_s (Nx.get [ 1 ] x) 1.))
        in
        List.iter
          (fun (name, m) ->
            let s =
              Minimize.solve one m
                ~lo:(vec [| -4. *. top; -10. |])
                ~hi:(vec [| 4. *. top; 10. |])
                ~tol:(Tol.abs 1e-6) ~budget:200 f
                (vec [| top +. 1.; 6. |])
            in
            not_equal ~msg:name Oracle.status_t Converged (Oracle.status s))
          quasi_newton);
    test "an objective whose slope is below tol does not converge" (fun () ->
        (* A slope below [tol] makes the step small, and the search doubles
           along the line through all its trials: nothing there is at the
           floats' limit. *)
        let f x =
          Nx.mul_s
            (Nx.sub (Nx.get [ 0 ] x) (Nx.mul_s (Nx.get [ 1 ] x) 2.))
            1e-12
        in
        List.iter
          (fun (name, m) ->
            let s =
              Minimize.solve one m ~tol:(Tol.abs 1e-8) ~budget:200 f
                (vec [| 0.5; 0.5 |])
            in
            not_equal ~msg:name Oracle.status_t Converged (Oracle.status s))
          quasi_newton);
    prop "beside a relative term r the absolute term √(r ε κ) |x*|∞ converges"
      Gen.(pair spread_minima (float_range (-12.) (-6.)))
      (fun ((a, xstar), e) ->
        (* [Tol.v ~rel:r ~abs:(√(r ε κ) |x*|∞)], [κ] the condition of the
           Hessian [a]: a gradient at its rounding, about [ε κ |x*|∞] times the
           least curvature, bounds the error in [tol]'s scaled norm by [(r / (M
           ε κ) + 1) / M] for an absolute term of [M ε κ |x*|∞], the resolution
           over that term times the ratio of the largest scale to the least,
           which [M = √(r / (ε κ))] holds near 1. *)
        let r = 10. ** e in
        let kappa = condition a in
        let top = Nx.item [] (Nx.max (Nx.abs xstar)) in
        let abs = Float.sqrt (r *. Float.epsilon *. kappa) *. top in
        cover "scales that differ by 10³ or more"
          (Array.exists (fun v -> Float.abs v *. r < abs) (Oracle.floats xstar)
          && (r *. top) +. abs >= 1e3 *. abs);
        List.iter
          (fun (name, m) ->
            equal ~msg:name Oracle.status_t Converged
              (Oracle.status
                 (Minimize.solve one m ~tol:(Tol.v ~rel:r ~abs) ~budget:200
                    (exact_objective a xstar ~boxed:false 0.)
                    (Nx.full_like xstar 0.5))))
          gradient_methods);
    prop ~count:2 "a constant added to f leaves a solve's status" rotated
      (fun (h, a) ->
        Compiled.here ();
        List.iter
          (fun (name, solve) ->
            let solve c = Oracle.status (solve h a (Nx.scalar f64 c)) in
            let base = solve 0. in
            List.iter
              (fun c -> equal ~msg:name Oracle.status_t base (solve c))
              [ 1e4; 1e8 ])
          rotated_solves);
  ]

let derivative_tests =
  [
    prop "grad is the implicit derivative H⁻¹ 1" convex (fun (a, b) ->
        List.iter
          (fun (_, m) ->
            let solve b =
              Solution.get (minimum m (objective a b) (Nx.zeros_like b))
            in
            (* One solve gives the minimum and its pullback. *)
            let x, back = Rune.vjp' solve b in
            let expected =
              if Nx.dim 0 b = 0 then b
              else Nx.solve (hessian a x) (Nx.ones_like b)
            in
            equal
              (Oracle.tensor ~rel:1e-8 ~abs:1e-12 ())
              expected
              (back (Nx.ones_like x)))
          gradient_methods);
    test "grad agrees with central differences" (fun () ->
        let solve c =
          Solution.get
            (minimum
               (Minimize.lbfgs ~memory:4 ~linear:Linear.dense)
               (fun x ->
                 Nx.add
                   (Nx.sum (Nx.square (Nx.sub x c)))
                   (Nx.sum (Nx.square (Nx.square x))))
               (Nx.zeros_like c))
        in
        let c = vec [| 1.; -0.5; 2. |] and v = vec [| 0.3; 1.; -1. |] in
        equal
          (Oracle.tensor ~rel:1e-6 ())
          (Oracle.central ~eps:1e-5 (fun c -> Nx.sum (solve c)) c v)
          (Nx.sum (Nx.mul (Rune.grad' (fun c -> Nx.sum (solve c)) c) v)));
    test "a lane that did not converge has a zero derivative" (fun () ->
        let g =
          Rune.grad'
            (fun c ->
              Nx.sum
                (Solution.best
                   (Minimize.solve one
                      (Minimize.bfgs ~linear:Linear.dense)
                      ~tol:tight ~budget:1
                      (fun x -> Nx.sum (Nx.square (Nx.sub x c)))
                      (Nx.zeros_like c))))
            (vec [| 1.; 2. |])
        in
        equal (Oracle.tensor ~abs:0. ()) (vec [| 0.; 0. |]) g);
    test "at a singular Hessian the derivative raises" (fun () ->
        (* (x + y − θ)² has a line of minima, along which its Hessian is
           singular. *)
        raises_in "Jera.Minimize.solve" (fun () ->
            Rune.grad'
              (fun theta ->
                Nx.sum
                  (Solution.get
                     (minimum
                        (Minimize.bfgs ~linear:Linear.dense)
                        (fun v -> Nx.square (Nx.sub (Nx.sum v) theta))
                        (vec [| 0.; 0. |]))))
              (Nx.scalar f64 1.)));
    test "at a rank-deficient fit the derivative raises" (fun () ->
        (* The residual x + y − θ has a line of fits, where JᵀJ is singular. *)
        raises_in "Jera.Minimize.solve" (fun () ->
            Rune.grad'
              (fun theta ->
                Nx.sum
                  (Solution.get
                     (Minimize.solve one lm ~tol:tight ~budget:100
                        (fun v -> Nx.reshape [| 1 |] (Nx.sub (Nx.sum v) theta))
                        (vec [| 0.; 0. |]))))
              (Nx.scalar f64 1.)));
  ]

let bfgs_tests =
  [
    test
      "a step on a linear stretch does not stop the search short of the minimum"
      (fun () ->
        (* The first step stays on Huber's linear stretch, where the gradient
           does not change. *)
        let a = vec [| 1.; 2.; 3. |] in
        let f x =
          let r = Nx.sub x a in
          let m = Nx.abs r in
          Nx.sum
            (Nx.where (Nx.less_equal_s m 1.)
               (Nx.mul_s (Nx.square r) 0.5)
               (Nx.sub_s m 0.5))
        in
        equal
          (Oracle.tensor ~abs:1e-6 ())
          a
          (Solution.get
             (Minimize.solve one
                (Minimize.bfgs ~linear:Linear.dense)
                ~lo:(Nx.full_like a (-20.)) ~hi:(Nx.full_like a 20.)
                ~tol:(Tol.abs 1e-8) ~budget:200 f
                (vec [| 10.; -8.; 15. |]))));
  ]

let lbfgs_tests =
  [
    test "memory below 1 raises" (fun () ->
        raises_in "Jera.Minimize.lbfgs" (fun () ->
            Minimize.lbfgs ~memory:0 ~linear:Linear.dense));
  ]

let newton_tests =
  [
    cases "a Hessian that is not positive-definite stalls the lane" ~id:fst
      [
        ("newton", Minimize.newton ~linear:Linear.dense);
        ( "newton-cg",
          Minimize.newton
            ~linear:(Linear.cg ~rel:1e-10 ~budget:50 ~precondition:Fun.id) );
      ]
      (fun (_, m) ->
        (* x⁴ − x² at 0.1 has [H = 12 x² − 2 < 0], and Newton's step points to
           the maximum at 0, uphill (jera.mli, newton). *)
        let f x = Nx.sum (Nx.sub (Nx.square (Nx.square x)) (Nx.square x)) in
        equal Oracle.status_t Stalled
          (Oracle.status
             (Minimize.solve one m ~tol:tight ~budget:50 f (vec [| 0.1 |]))));
    test "a step below ε |x|∞ beside a large component is taken" (fun () ->
        (* The step of 10⁻⁹ is below [ε |x|∞] and exact. *)
        let xstar = vec [| 1e8; 1e-9 |] in
        let f x =
          let e = Nx.sub x xstar in
          Nx.mul_s (Nx.sum (Nx.mul e e)) 0.5
        in
        equal
          (Oracle.tensor ~abs:1e-6 ())
          xstar
          (Solution.get
             (Minimize.solve one
                (Minimize.newton ~linear:Linear.dense)
                ~tol:(Tol.abs 1e-6) ~budget:200 f
                (vec [| 1e8; 0. |]))));
  ]

let fit y =
  Minimize.solve one lm ~tol:tight ~budget:100
    (fun p -> Nx.sub (model p) y)
    (vec [| 1.; 0. |])

let exponentials =
  Gen.(
    let+ a = float_range 0.5 3. and+ b = float_range (-2.) 1. in
    vec [| a; b |])
  |> Gen.with_pp Nx.pp

(* A bound on the curvature of a fit's cost [|r|² / 2] at [p]: the trace of
   [JᵀJ], [J]'s columns [e^(b t)] and [a t e^(b t)], plus [Σ |rᵢ| ‖∇²rᵢ‖],
   [∇²rᵢ] of entries [t e^(b t)] off its diagonal and [a t² e^(b t)] on it. *)
let fit_curvature p y =
  let a = Nx.item [ 0 ] p and b = Nx.item [ 1 ] p in
  let ts = Oracle.floats times and ys = Oracle.floats y in
  let sum = ref 0. in
  Array.iteri
    (fun i t ->
      let e = exp (b *. t) in
      let r = (a *. e) -. ys.(i) in
      sum :=
        !sum +. (e *. e)
        +. (a *. a *. t *. t *. e *. e)
        +. (Float.abs r *. ((2. *. t *. e) +. (Float.abs a *. t *. t *. e))))
    ts;
  !sum

let levenberg_marquardt_tests =
  [
    prop "it recovers the parameters of exact data" exponentials (fun p ->
        equal
          (Oracle.tensor ~rel:1e-9 ~abs:1e-11 ())
          p
          (Solution.get (fit (model p))));
    prop "a fit with residuals left converges where Jᵀ r vanishes" noisy
      (fun (p, sine) ->
        (* The guard of a fit whose cost changes by less than its rounding
           before its steps meet [tol]. Converged, its error estimate meets
           [tol], whose scale is at most [s = 1e-8 |p|∞ + 1e-10] per component,
           so the answer lies within about [2 √2 s] of the minimum, where the
           cost's gradient [Jᵀ r] is 0, and the gradient there is at most the
           cost's curvature times that. *)
        let y = noisy_data p sine in
        let s =
          Minimize.solve one lm
            ~tol:(Tol.v ~rel:1e-8 ~abs:1e-10)
            ~budget:100
            (fun p -> Nx.sub (model p) y)
            (vec [| 1.; 0. |])
        in
        equal Oracle.status_t Converged (Oracle.status s);
        let fit = Solution.best s in
        let scale = (1e-8 *. Nx.item [] (Nx.max (Nx.abs fit))) +. 1e-10 in
        let cost p = Nx.mul_s (Nx.sum (Nx.square (Nx.sub (model p) y))) 0.5 in
        equal
          (Oracle.tensor
             ~abs:(2. *. Float.sqrt 2. *. scale *. fit_curvature fit y)
             ())
          (vec [| 0.; 0. |])
          (Rune.grad' cost fit));
    test "with residuals of a twentieth of the signal, Jᵀ r vanishes at the fit"
      (fun () ->
        let y =
          Nx.add
            (model (vec [| 2.; -1.3 |]))
            (Nx.mul_s (Nx.sin (Nx.mul_s times 7.)) 0.05)
        in
        let p = Solution.get (fit y) in
        let cost p = Nx.mul_s (Nx.sum (Nx.square (Nx.sub (model p) y))) 0.5 in
        equal
          (Oracle.tensor ~abs:1e-9 ())
          (vec [| 0.; 0. |])
          (Rune.grad' cost p));
    prop "a fit scaled by a power of two reaches its fit scaled, bit for bit"
      exponentials (fun p ->
        (* [s r (x / s)] from [s x₀] evaluates [s] times what [r] does, exactly,
           with the same Jacobian, at [s = 2⁻⁵⁴⁰], whose squares are below the
           floats' range: every decision is the unscaled fit's. *)
        let s = Float.ldexp 1. (-540) in
        let y = model p in
        let r q = Nx.sub (model q) y in
        let scaled x = Nx.mul_s (r (Nx.div_s x s)) s in
        let fit r tol x0 = Minimize.solve one lm ~tol ~budget:100 r x0 in
        let plain = fit r tight (vec [| 1.; 0. |])
        and small =
          fit scaled (Tol.v ~rel:1e-10 ~abs:(1e-12 *. s)) (vec [| s; 0. |])
        in
        equal (Oracle.tensor ())
          (Nx.mul_s (Solution.best plain) s)
          (Solution.best small);
        equal Oracle.status_t (Oracle.status plain) (Oracle.status small));
    test "Rosenbrock's residual fits at (1, 1)" (fun () ->
        let r v =
          let x = Nx.get [ 0 ] v and y = Nx.get [ 1 ] v in
          Nx.stack [ Nx.mul_s (Nx.sub y (Nx.square x)) 10.; Nx.rsub_s 1. x ]
        in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (vec [| 1.; 1. |])
          (Solution.get
             (Minimize.solve one lm ~tol:tight ~budget:100 r
                (vec [| -1.2; 1. |]))));
    test "a residual of a structure is one vector" (fun () ->
        (* The samples split in two halves, as a pair: the same components in
           the same order as one tensor. *)
        let y = model (vec [| 2.; -1.3 |]) in
        let halves v =
          (Nx.slice [ Nx.R (0, 10) ] v, Nx.slice [ Nx.R (10, 20) ] v)
        in
        let pair =
          Minimize.solve one
            (Minimize.levenberg_marquardt
               Nx.Ptree.(pair tensor tensor)
               ~linear:Linear.dense)
            ~tol:tight ~budget:100
            (fun p -> halves (Nx.sub (model p) y))
            (vec [| 1.; 0. |])
        in
        equal (Oracle.tensor ()) (Solution.best (fit y)) (Solution.best pair));
    prop "a converged fit's error estimate meets tol"
      Gen.(pair noisy ulps_k)
      (fun ((p, sine), k) ->
        (* A fit's answer is fixed to [ε κ] times its largest parameter, [κ] its
           Jacobian's condition, which [k] ulps of the smaller meet only from [k
           ≥ κ |p|∞ / |p|min]: the fit takes [2 k] times that. Where the Gram
           matrix is near singular, [κ] grows without bound: 1e12 ulps keeps the
           drawn tolerance finite there. *)
        let k = k *. Float.min (2. *. resolution_ulps p) 1e12 in
        let y = noisy_data p sine in
        let s =
          Minimize.solve one lm ~tol:(Tol.ulps k) ~budget:200
            (fun p -> Nx.sub (model p) y)
            (vec [| 1.; 0. |])
        in
        cover "levenberg_marquardt converges" (converged s);
        if converged s then
          at_most float_exact ~than:(1. +. 1e-12)
            (Oracle.ulps ~k ~y:(Solution.best s) (Solution.error s)));
    test "grad in the data agrees with central differences" (fun () ->
        let y =
          Nx.add
            (model (vec [| 2.; -1.3 |]))
            (Nx.mul_s (Nx.sin (Nx.mul_s times 7.)) 0.05)
        in
        let v = Nx.cos (Nx.mul_s times 3.) in
        let f y = Nx.sum (Solution.get (fit y)) in
        equal
          (Oracle.tensor ~rel:1e-6 ())
          (Oracle.central ~eps:1e-6 f y v)
          (Nx.sum (Nx.mul (Rune.grad' f y) v)));
    test "a compiled step sums in eager's order" (fun () ->
        Compiled.here ();
        (* One unknown, [2 e^(b t)] from [b = 0], where the model's values and
           Jacobian are exact: the step sums [JᵀJ] and [Jᵀ r] over the 20
           samples in jera.mli's fixed order (Tol), and [Linear.dense]'s 1 × 1
           solve is then a division, so the step is eager's to the bit. Past it
           the compiled exponential may differ by its ulps. *)
        let decay b = Nx.mul_s (Nx.exp (Nx.mul b times)) 2. in
        let f y =
          Nx.get [ 1 ]
            (Minimize.iterates one lm ~steps:2
               (fun b -> Nx.sub (decay b) y)
               (vec [| 0. |]))
        in
        let y = decay (vec [| -1.3 |]) in
        equal (Oracle.tensor ()) (f y) (Rune.jit' f y));
    test "a compiled fit of a rational model converges where eager does"
      (fun () ->
        Compiled.here ();
        (* Its solves round within rune's numerics compiled (jera.mli, Tol), so
           compiled and eager may decide apart; both fit exact data at (2, 0.3),
           where Gauss–Newton's steps at least halve the distance and the answer
           is within [tight] of it (jera.mli, System): within [2 √2] of its
           scale. *)
        let rational p =
          Nx.div (Nx.get [ 0 ] p) (Nx.add_s (Nx.mul (Nx.get [ 1 ] p) times) 1.)
        in
        let f y =
          Solution.get
            (Minimize.solve one lm ~tol:tight ~budget:100
               (fun p -> Nx.sub (rational p) y)
               (vec [| 1.; 0. |]))
        in
        let y = rational (vec [| 2.; 0.3 |]) in
        let p = vec [| 2.; 0.3 |] in
        equal (Oracle.tensor ~rel:3e-10 ~abs:3e-12 ()) p (f y);
        equal (Oracle.tensor ~rel:3e-10 ~abs:3e-12 ()) p (Rune.jit' f y));
    test "a fit of no unknowns converges at once" (fun () ->
        equal Oracle.status_t Converged
          (Oracle.status
             (Minimize.solve one lm ~tol:tight ~budget:10 Fun.id (vec [||]))));
    test "a residual not finite at the start ends the lane Not_finite"
      (fun () ->
        equal Oracle.status_t Not_finite
          (Oracle.status
             (Minimize.solve one lm ~tol:tight ~budget:10 Nx.sqrt
                (vec [| -1. |]))));
    test "a float32 fit is solved in float32" (fun () ->
        (* As the float32 problems of [minima], within twice [tol]'s scale. *)
        let c = Nx.create Nx.float32 [| 2 |] [| 1.; -0.5 |] in
        equal
          (Oracle.tensor ~abs:4e-3 ())
          c
          (Solution.get
             (Minimize.solve one
                (Minimize.levenberg_marquardt one ~linear:Linear.dense)
                ~tol:(Tol.v ~rel:1e-3 ~abs:1e-3)
                ~budget:200
                (fun x -> Nx.sub x c)
                (Nx.zeros Nx.float32 [| 2 |]))));
  ]

let simplex ?(budget = 2000) ?(tol = Tol.v ~rel:1e-9 ~abs:1e-10) f x0 =
  Minimize.solve one Minimize.nelder_mead ~tol ~budget f x0

(* How far the answer may sit from the bowl's center [c] under [~rel ~abs]. The
   method promises the diameter per component [e_i] of its simplex around the
   best vertex [y]: the root mean square of [e_i / s_i], [s_i = abs + rel
   |y_i|], is at most 1, so one component of [n] reaches [√n s_i]. The restart's
   probes [y ± σ e_i], [σ] twice the largest [e_i] or [s_i], find none lower,
   which on the bowl, [|x − c|²] near [c], puts [c_i] within [σ / 2] of [y_i].
   So [|y_i − c_i| <= σ / 2 <= √n max_i s_i], and with [|y_i| <= |c_i| + σ / 2],
   [σ / 2 <= √n (abs + rel max_i |c_i|) / (1 − √n rel)]. *)
let around ~rel ~abs c =
  let k = Float.sqrt (Float.of_int (Nx.dim 0 c)) in
  k *. (abs +. (rel *. Nx.item [] (Nx.max (Nx.abs c)))) /. (1. -. (k *. rel))

let bowl_tol = (1e-8, 1e-9)

(* nelder_mead's answer for the bowl centred at [c] from 0, compiled, since an
   eager search takes up to a sixth of a second: one program per number of
   unknowns, whose coefficients depend on it. *)
let bowl_search =
  let rel, abs = bowl_tol in
  Rune.jit
    Nx.Ptree.(tensor @-> returns (Solution.ptree tensor))
    (fun c ->
      simplex ~budget:5000 ~tol:(Tol.v ~rel ~abs)
        (fun x -> Nx.sum (bowl c x))
        (Nx.zeros_like c))

(* Three lanes of bowls from 0 under [Tol.abs 0.01], each centred at 0 or with
   every coordinate at least 0.1 from it. The first trip moves the best vertex
   by a few of the first simplex's steps of [2.5 · 10⁻⁴], so the diameter meets
   [tol] and the restart test probes at [σ = 0.02]. A lane centred at 0 keeps
   its best vertex there and converges after that trip, within [3] evaluations
   of the simplex, [5] of the trip and [4] of the probes. Any other lane finds a
   probe lower, restarts, and meets [tol] again before it converges: at least
   [13]. *)
let restart_lanes =
  let coordinate =
    Gen.(
      map
        (fun (up, m) -> if up then m else -.m)
        (pair bool (float_range 0.1 2.)))
  in
  let center =
    Gen.one_of
      [
        Gen.constant [| 0.; 0. |];
        Gen.map (fun (a, b) -> [| a; b |]) (Gen.pair coordinate coordinate);
      ]
  in
  Gen.(
    map
      (fun l -> Nx.create f64 [| 3; 2 |] (Array.concat l))
      (list ~size:(constant 3) center))
  |> Gen.with_pp Nx.pp

let nelder_mead_tests =
  [
    prop "it finds a bowl's center"
      ~examples:[ vec [| 2.5 |]; vec [| -3.; 1.; 0.; 2.75; -0.5; 3. |] ]
      centers
      (fun c ->
        Compiled.here ();
        cover "one unknown" (Nx.dim 0 c = 1);
        cover "six unknowns" (Nx.dim 0 c = 6);
        let rel, abs = bowl_tol in
        equal
          (Oracle.tensor ~abs:(around ~rel ~abs c) ())
          c
          (Solution.get (bowl_search c)));
    test "it finds Rosenbrock's minimum" (fun () ->
        equal
          (* [Tol.v ~rel:1e-9 ~abs:1e-10] on the simplex's diameter; the
             valley's curvatures differ by 2500 at (1, 1), along which the
             diameter understates the distance: 1e-5 bounds it. *)
          (Oracle.tensor ~rel:1e-5 ())
          (vec [| 1.; 1. |])
          (Solution.get (simplex rosenbrock (vec [| -1.2; 1. |]))));
    test "it minimises a non-smooth function" (fun () ->
        (* |x − 1| + 2 |y + 0.5|, whose gradient is not defined at its
           minimum. *)
        let f v =
          Nx.add
            (Nx.abs (Nx.sub_s (Nx.get [ 0 ] v) 1.))
            (Nx.mul_s (Nx.abs (Nx.add_s (Nx.get [ 1 ] v) 0.5)) 2.)
        in
        equal
          (Oracle.tensor ~abs:1e-6 ())
          (vec [| 1.; -0.5 |])
          (Solution.get (simplex f (vec [| 3.; 2. |]))));
    test "McKinnon's function converges at its minimum" (fun () ->
        (* τ = 2, θ = 6, φ = 60: the minimum is (0, −1/2). *)
        let f v =
          let x = Nx.get [ 0 ] v and y = Nx.get [ 1 ] v in
          let side =
            Nx.where (Nx.less_equal_s x 0.) (Nx.scalar f64 360.)
              (Nx.scalar f64 6.)
          in
          Nx.add (Nx.mul side (Nx.square x)) (Nx.add y (Nx.square y))
        in
        equal
          (Oracle.tensor ~abs:1e-5 ())
          (vec [| 0.; -0.5 |])
          (Solution.get (simplex f (vec [| 1.; 1. |]))));
    cases "a scaled f converges where f does"
      ~id:(Printf.sprintf "scaled by %g") [ 1e170; 1e-170 ] (fun c ->
        (* c ½ (x² + (x + 1)²) with x ≤ 0, from 0, where the first simplex
           collapses onto the bound: the minimum is −1/2 at any scale. By 1e170
           the probe's difference toward −1/2 squares past the floats' range, by
           1e-170 below it. *)
        let s =
          Minimize.solve one Minimize.nelder_mead ~hi:(vec [| 0. |])
            ~tol:(Tol.abs 1e-6) ~budget:5000
            (fun x ->
              let squares = Nx.add (Nx.square x) (Nx.square (Nx.add_s x 1.)) in
              Nx.mul_s (Nx.sum squares) (0.5 *. c))
            (vec [| 0. |])
        in
        equal Oracle.status_t Converged (Oracle.status s);
        equal (Oracle.tensor ~abs:1e-5 ()) (vec [| -0.5 |]) (Solution.get s));
    test "a non-finite value past the start counts as +∞" (fun () ->
        (* (y − 3)² defined only where x = 0, NaN elsewhere: a vertex off the
           line is worse than any on it, so the best vertex stays on it, and
           only a NaN at the start ends the lane Not_finite. *)
        let f v =
          let x = Nx.get [ 0 ] v and y = Nx.get [ 1 ] v in
          Nx.where (Nx.equal_s x 0.)
            (Nx.square (Nx.sub_s y 3.))
            (Nx.scalar f64 Float.nan)
        in
        let s = simplex ~budget:50 f (vec [| 0.; 0. |]) in
        not_equal Oracle.status_t Not_finite (Oracle.status s);
        equal (Oracle.tensor ()) (Nx.scalar f64 0.)
          (Nx.get [ 0 ] (Solution.best s)));
    test "its answer carries no derivative" (fun () ->
        let g =
          Rune.grad'
            (fun c ->
              Nx.sum
                (Solution.get
                   (simplex (fun x -> Nx.sum (bowl c x)) (Nx.zeros_like c))))
            (vec [| 1.; 2. |])
        in
        equal (Oracle.tensor ~abs:0. ()) (vec [| 0.; 0. |]) g);
    test "budget trips end it Budget_spent" (fun () ->
        (* jera.mli's Cost: the first simplex takes n + 1 = 3 evaluations, a
           trip one to n + 3, a restart test 2n: 10 trips take 13 to 93. *)
        let s = simplex ~budget:10 rosenbrock (vec [| -1.2; 1. |]) in
        equal Oracle.status_t Budget_spent (Oracle.status s);
        let evaluations = Nx.item [] (Solution.evaluations s) in
        at_least int32 ~than:13l evaluations;
        at_most int32 ~than:93l evaluations);
    test "compiled is eager, bit for bit" (fun () ->
        Compiled.here ();
        let f c =
          Solution.get (simplex (fun x -> Nx.sum (bowl c x)) (Nx.zeros_like c))
        in
        let c = vec [| 1.; -0.5 |] in
        equal (Oracle.tensor ()) (f c) (Rune.jit' f c));
    prop ~count:20 "vmap is each lane's search, restarted or not"
      ~examples:[ Nx.create f64 [| 3; 2 |] [| 0.; 0.; 1.5; -0.25; -0.1; 2. |] ]
      restart_lanes
      (fun cs ->
        let centred i = Nx.item [] (Nx.max (Nx.abs (Nx.get [ i ] cs))) = 0. in
        let lanes = [ 0; 1; 2 ] in
        cover "a lane restarts beside one that does not"
          (List.exists centred lanes
          && List.exists (fun i -> not (centred i)) lanes);
        let f c =
          let s =
            simplex ~tol:(Tol.abs 0.01)
              (fun x -> Nx.sum (bowl c x))
              (Nx.zeros_like c)
          in
          let scalar x = Nx.reshape [| 1 |] (Nx.cast f64 x) in
          Nx.concatenate ~axis:0
            [
              Solution.best s;
              Solution.error s;
              scalar (Solution.evaluations s);
              scalar (Solution.is Converged s);
            ]
        in
        let mapped = Rune.vmap' f cs in
        equal (Oracle.tensor ())
          (Nx.stack (List.init 3 (fun i -> f (Nx.get [ i ] cs))))
          mapped;
        List.iter
          (fun i ->
            let evaluations = Nx.item [ i; 4 ] mapped in
            if centred i then at_most float_exact evaluations ~than:12.
            else at_least float_exact evaluations ~than:13.)
          lanes);
  ]

let unit_box b = (Nx.full_like b (-0.5), Nx.full_like b 0.5)

(* [centers] with each center whose square is below [10⁻³⁰⁰] set to 0, the
   margin [convex] keeps above the subnormals: from 0, a bowl whose values are
   subnormal moves its search by steps the floats cannot resolve, and it stalls,
   as it states. *)
let normal_centers =
  Gen.map
    (fun c -> Nx.where (Nx.less_s (Nx.square c) 1e-300) (Nx.zeros_like c) c)
    centers
  |> Gen.with_pp Nx.pp

(* A linear least-squares problem [½ |a x − b|²] of [n] unknowns in a box: [a]
   stacks the identity on two rows of entries in [-2, 2], so [aᵀ a] is
   positive-definite with its unknowns coupled and the minimum is unique. A
   coordinate's box is unbounded, bounded on one side, closed to a point, or an
   interval. A number below [10⁻³⁰⁰] is zero: in [a] or a bound it puts the
   minimum, or the decrease toward it, among subnormals, where the search stalls
   as it states, as in [convex]. *)
let least_squares_of sizes =
  let open Gen in
  let number lo hi =
    map (fun v -> if Float.abs v < 1e-300 then 0. else v) (float_range lo hi)
  in
  let side =
    let+ lo = number (-2.) 1.
    and+ width = number 0. 2.
    and+ kind = int_range 0 4 in
    match kind with
    | 0 -> (Float.neg_infinity, Float.infinity)
    | 1 -> (Float.neg_infinity, lo +. width)
    | 2 -> (lo, Float.infinity)
    | 3 -> (lo, lo)
    | _ -> (lo, lo +. width)
  in
  (let* n = sizes in
   let+ rows = list ~size:(constant (2 * n)) (number (-2.) 2.)
   and+ b = list ~size:(constant (n + 2)) (number (-5.) 5.)
   and+ sides = list ~size:(constant n) side in
   let a =
     Nx.concatenate ~axis:0
       [ Nx.eye f64 n; Nx.create f64 [| 2; n |] (Array.of_list rows) ]
   in
   ( a,
     Nx.create f64 [| n + 2 |] (Array.of_list b),
     ( vec (Array.of_list (List.map fst sides)),
       vec (Array.of_list (List.map snd sides)) ) ))
  |> with_pp (fun ppf (a, b, (lo, hi)) ->
      Format.fprintf ppf "a = %a@ b = %a@ lo = %a@ hi = %a" Nx.pp a Nx.pp b
        Nx.pp lo Nx.pp hi)

let least_squares = least_squares_of (Gen.int_range 1 5)

let squares_gradient a b x =
  Nx.matmul (Nx.transpose a) (Nx.sub (Nx.matmul a x) b)

(* The tolerances the box's law draws, each with its relative and absolute
   terms: [Tol.ulps k] is [Tol.rel (k ε)], [ε = 2⁻⁵²] in float64. A relative
   term alone scales to 0 at an answer of 0. *)
let simplex_tols =
  let k = 0x1p25 in
  [|
    ("Tol.abs", 0., 1e-6, Tol.abs 1e-6);
    ("Tol.rel", 1e-6, 0., Tol.rel 1e-6);
    ("Tol.ulps", k *. epsilon_float, 0., Tol.ulps k);
  |]

(* The first simplex's step at 0, which jera.mli states. *)
let first_step = 2.5e-4

(* nelder_mead's answer for [c ½ |a x − b|²] in [box] from 0, to the tolerance
   [simplex_tols.(k)]. The objective is [bad] where a coordinate is below its
   wall [below] or above its wall [above]. Compiled, since an eager search takes
   most of a second: one program per tolerance and number of unknowns. *)
let simplex_programs =
  Array.map
    (fun (_, _, _, tol) ->
      Rune.jit
        Nx.Ptree.(
          tensor @-> tensor @-> pair tensor tensor
          @-> pair (pair tensor tensor) (pair tensor tensor)
          @-> returns (Solution.ptree tensor))
        (fun a b (lo, hi) ((below, above), (c, bad)) ->
          let f x =
            let r = Nx.sub (Nx.matmul a x) b in
            let out = Nx.logical_or (Nx.less x below) (Nx.greater x above) in
            Nx.where (Nx.any out) bad
              (Nx.mul c (Nx.mul_s (Nx.sum (Nx.square r)) 0.5))
          in
          Minimize.solve one Minimize.nelder_mead ~lo ~hi ~tol ~budget:5000 f
            (Nx.zeros_like lo)))
    simplex_tols

(* No wall, no scale. *)
let plain n =
  ( (Nx.full f64 [| n |] Float.neg_infinity, Nx.full f64 [| n |] Float.infinity),
    (Nx.scalar f64 1., Nx.scalar f64 Float.nan) )

(* [least_squares] of one to five unknowns, one program per tolerance and number
   of unknowns, with the tolerance [k] of [simplex_tols], walls past which f is
   [bad], NaN, +∞ or −∞, and a scale [c] of f. Of each coordinate's four cases,
   three have a wall: up to 1 above the start, 0 projected on the box, up to 1
   below it, or both. So the start lies in f's domain and a wall often holds the
   answer. A scale of 1e170 squares the restart's differences past the floats'
   range, 1e-170 below it; f's decisions are those of [c = 1] otherwise. *)
let walled =
  let open Gen in
  (let* k = int_range 0 (Array.length simplex_tols - 1) in
   let* a, b, (lo, hi) = least_squares in
   let n = Nx.dim 0 lo in
   let+ sides =
     list ~size:(constant n)
       (pair (int_range 0 3) (pair (float_range 0. 1.) (float_range 0. 1.)))
   and+ c = of_list [ 1.; 1e170; 1e-170 ]
   and+ bad = of_list [ Float.nan; Float.infinity; Float.neg_infinity ] in
   let start = Nx.minimum hi (Nx.maximum lo (Nx.zeros_like lo)) in
   let gaps side far =
     vec
       (Array.of_list
          (List.map
             (fun (s, (u, d)) ->
               if s land side <> 0 then far u d else Float.infinity)
             sides))
   in
   let above = Nx.add start (gaps 1 (fun u _ -> u)) in
   let below = Nx.sub start (gaps 2 (fun _ d -> d)) in
   (k, (a, b, (lo, hi)), ((below, above), (c, bad))))
  |> with_pp (fun ppf (k, (a, b, (lo, hi)), ((below, above), (c, bad))) ->
      let name, _, _, _ = simplex_tols.(k) in
      Format.fprintf ppf
        "%s@ a = %a@ b = %a@ lo = %a@ hi = %a@ below = %a@ above = %a@ c = %g@ \
         bad = %g"
        name Nx.pp a Nx.pp b Nx.pp lo Nx.pp hi Nx.pp below Nx.pp above c bad)

(* How far from [P (x − g) = x], [g] the gradient and [P] the projection on the
   box, a [Converged] answer [x] of [simplex_programs.(k)] may be. Converging,
   the restart test finds no probe [y ± σ eᵢ], clipped into the box, lower than
   [y]. With [s] the tolerance's scale at [x], [abs + rel |x|∞], the diameter's
   root mean square ratio to [s] is at most 1, so [2 s ≤ σ ≤ 2 √n s], and where
   [s] is 0, [σ] is the first simplex's step. Component [i] of [x − P (x − g)]
   is [min (|gᵢ|, δ)], [δ] the distance to the box's side downhill along [eᵢ]:
   below [σ] when [δ < σ], and otherwise the probe [σ] downhill is not lower, so
   [|gᵢ| ≤ hᵢ σ / 2 + 2 ε / σ] on this parabola of curvature [hᵢ ≤ 9] along [eᵢ]
   (1 plus two entries of [a] in [[-2, 2]], squared), [ε] [f]'s rounding. So
   each component is within [4.5 σ + 2 ε / σ]; the restart's [10⁻⁴] of a
   predicted decrease stays within the margin to [5 σ]. [ε] is taken as [10⁻¹³
   (1 + f x + |b|²)], a few ulps of the sum of squares' terms. *)
let stationary k a b x =
  let _, rel, abs, _ = simplex_tols.(k) in
  let s = abs +. (rel *. Nx.item [] (Nx.max (Nx.abs x))) in
  let lower, upper =
    if s = 0. then (first_step, first_step)
    else (2. *. s, 2. *. Float.sqrt (Float.of_int (Nx.dim 0 x)) *. s)
  in
  let r = Nx.sub (Nx.matmul a x) b in
  let fx = 0.5 *. Nx.item [] (Nx.sum (Nx.square r)) in
  let rounding = 1e-13 *. (1. +. fx +. Nx.item [] (Nx.sum (Nx.square b))) in
  (5. *. upper) +. (2. *. rounding /. lower)

(* jera.mli's bound on a gradient method's evaluations: [budget] iterations of
   at most [3p] trials each, [p = 53] in float64. A search whose trials keep
   meeting NaN runs to it, a few hundred evaluations at [budget = 8], hence the
   fewer cases. *)
let evaluation_bound budget = Int32.of_int (1 + (budget * 3 * 53))

let solve_tests =
  [
    prop "a boxed answer is its projected gradient's fixed point"
      ~examples:
        [
          (* Newton held a coordinate on the far side of the box from the bound
             its gradient points to. *)
          ( Nx.create f64 [| 6; 6 |]
              [|
                6.4313272094841949;
                0.028430426682705345;
                0.;
                0.66626301127288756;
                0.30728003895831379;
                0.5341926153359895;
                0.028430426682705345;
                7.;
                -0.30692569518070034;
                -0.31513975132908201;
                -0.43512096496446973;
                0.10969130621850265;
                0.;
                -0.30692569518070034;
                6.4678997436474717;
                -0.065968985608132913;
                0.53068011374290791;
                -0.045649035063120569;
                0.66626301127288756;
                -0.31513975132908201;
                -0.065968985608132913;
                6.9165894552671086;
                0.15951257682114395;
                0.47505615008372593;
                0.30728003895831379;
                -0.43512096496446973;
                0.53068011374290791;
                0.15951257682114395;
                6.0895978175913701;
                -0.23389400887412992;
                0.5341926153359895;
                0.10969130621850265;
                -0.045649035063120569;
                0.47505615008372593;
                -0.23389400887412992;
                6.4250945471681007;
              |],
            vec
              [|
                2.1089768019734678;
                0.45815595109283014;
                2.9036475074050507;
                -4.1882141344280761;
                5.;
                -2.4466654029499448;
              |] );
        ]
      convex
      (fun (a, b) ->
        cover "a coordinate held at a bound"
          (Nx.dim 0 b > 0 && Nx.item [] (Nx.max (Nx.abs (Nx.solve a b))) > 0.5);
        List.iter
          (fun (_, m) ->
            let lo, hi = unit_box b in
            let x =
              Solution.get
                (Minimize.solve one m ~lo ~hi ~tol:tight ~budget:200
                   (objective a b) (Nx.zeros_like b))
            in
            let clipped =
              Nx.minimum hi (Nx.maximum lo (Nx.sub x (gradient a b x)))
            in
            equal
              (Oracle.tensor ~abs:(1e-8 *. Float.of_int (Nx.dim 0 b + 2)) ())
              clipped x)
          gradient_methods);
    prop "one side bounds alone" normal_centers (fun c ->
        (* The bowl is a sum of one bowl per coordinate, so its minimum in a box
           is its center clipped to the box. lbfgs takes L-BFGS-B's Cauchy step
           and newton Bertsekas's projected step, the two ways a gradient method
           steps in a box. *)
        cover "a coordinate held at its bound"
          (Nx.item [] (Nx.any (Nx.less_s c 0.)));
        cover "a free coordinate" (Nx.item [] (Nx.any (Nx.greater_s c 0.)));
        let zero = Nx.zeros_like c in
        let f x = Nx.sum (bowl c x) in
        List.iter
          (fun m ->
            let solve ?lo ?hi () =
              Solution.get
                (Minimize.solve one m ?lo ?hi ~tol:tight ~budget:200 f zero)
            in
            equal (near ()) (Nx.maximum c zero) (solve ~lo:zero ());
            equal (near ()) (Nx.minimum c zero) (solve ~hi:zero ()))
          [
            Minimize.lbfgs ~memory:3 ~linear:Linear.dense;
            Minimize.newton ~linear:Linear.dense;
          ]);
    cases "an absent side is an infinite one" ~id:fst
      (("nelder_mead", Minimize.nelder_mead) :: dense_methods) (fun (_, m) ->
        (* The answers and their derivatives in [c] agree to the bit. *)
        let zero = vec [| 0.; 0.; 0. |] in
        let inf = Nx.full_like zero Float.infinity in
        let answer (lo, hi) c =
          Solution.best
            (Minimize.solve one m ?lo ?hi ~tol:tight ~budget:2000
               (fun x -> Nx.sum (bowl c x))
               zero)
        in
        let grad box = Rune.grad' (fun c -> Nx.sum (answer box c)) in
        let c = vec [| 1.; -0.25; 2. |] in
        List.iter
          (fun (absent, infinite) ->
            equal (Oracle.tensor ()) (answer infinite c) (answer absent c);
            equal (Oracle.tensor ()) (grad infinite c) (grad absent c))
          [
            ((Some zero, None), (Some zero, Some inf));
            ((None, Some zero), (Some (Nx.neg inf), Some zero));
          ]);
    test "levenberg_marquardt's absent side is an infinite one" (fun () ->
        let zero = vec [| 0.; 0.; 0. |] in
        let inf = Nx.full_like zero Float.infinity in
        let answer (lo, hi) c =
          Solution.best
            (Minimize.solve one lm ?lo ?hi ~tol:tight ~budget:200
               (fun x -> Nx.sub x c)
               zero)
        in
        let grad box = Rune.grad' (fun c -> Nx.sum (answer box c)) in
        let c = vec [| 1.; -0.25; 2. |] in
        List.iter
          (fun (absent, infinite) ->
            equal (Oracle.tensor ()) (answer infinite c) (answer absent c);
            equal (Oracle.tensor ()) (grad infinite c) (grad absent c))
          [
            ((Some zero, None), (Some zero, Some inf));
            ((None, Some zero), (Some (Nx.neg inf), Some zero));
          ]);
    cases "x0 outside the box is projected on it first" ~id:fst
      (("nelder_mead", Minimize.nelder_mead) :: gradient_methods) (fun (_, m) ->
        (* f is NaN outside [−1, 1]², so a start not projected would end the
           lane Not_finite. The bowl's center (2, −0.25) clips to (1, −0.25). *)
        let c = vec [| 2.; -0.25 |] in
        let f x =
          let v = Nx.sum (bowl c x) in
          Nx.where
            (Nx.less_equal_s (Nx.max (Nx.abs x)) 1.)
            v (Nx.full_like v Float.nan)
        in
        let s =
          Minimize.solve one m
            ~lo:(Nx.full f64 [| 2 |] (-1.))
            ~hi:(Nx.ones f64 [| 2 |])
            ~tol:(Tol.v ~rel:1e-8 ~abs:1e-10)
            ~budget:500 f
            (vec [| 5.; -5. |])
        in
        equal
          (Oracle.tensor ~abs:1e-6 ())
          (vec [| 1.; -0.25 |])
          (Solution.get s));
    cases "a held coordinate's derivative is 0 in the target and 1 in its bound"
      ~id:fst dense_methods (fun (_, m) ->
        (* ½ |x − c|² in [0, 1]², c = (2, 0.5): the minimum is (1, 0.5), its
           first coordinate held at its upper bound. *)
        let solve (c, hi) =
          Solution.get
            (Minimize.solve one m ~lo:(Nx.zeros f64 [| 2 |]) ~hi ~tol:tight
               ~budget:100
               (fun x -> Nx.mul_s (Nx.sum (Nx.square (Nx.sub x c))) 0.5)
               (Nx.zeros f64 [| 2 |]))
        in
        let c = vec [| 2.; 0.5 |] and hi = vec [| 1.; 1. |] in
        equal (Oracle.tensor ~rel:1e-9 ()) (vec [| 1.; 0.5 |]) (solve (c, hi));
        let dc, dhi =
          Rune.grad
            Nx.Ptree.(pair tensor tensor)
            (fun ch -> Nx.sum (solve ch))
            (c, hi)
        in
        equal (Oracle.tensor ~abs:1e-12 ()) (vec [| 0.; 1. |]) dc;
        equal (Oracle.tensor ~abs:1e-12 ()) (vec [| 1.; 0. |]) dhi);
    cases "a coordinate held at a lone lower bound follows it" ~id:fst
      dense_methods (fun (_, m) ->
        (* ½ |x − c|² with x ≥ lo, c = (−1, 0.5) and lo = 0: the minimum is (0,
           0.5). *)
        let solve (c, lo) =
          Solution.get
            (Minimize.solve one m ~lo ~tol:tight ~budget:100
               (fun x -> Nx.mul_s (Nx.sum (Nx.square (Nx.sub x c))) 0.5)
               (Nx.zeros_like c))
        in
        let c = vec [| -1.; 0.5 |] and lo = vec [| 0.; 0. |] in
        equal (Oracle.tensor ~abs:1e-12 ()) (vec [| 0.; 0.5 |]) (solve (c, lo));
        let dc, dlo =
          Rune.grad
            Nx.Ptree.(pair tensor tensor)
            (fun cl -> Nx.sum (solve cl))
            (c, lo)
        in
        equal (Oracle.tensor ~abs:1e-12 ()) (vec [| 0.; 1. |]) dc;
        equal (Oracle.tensor ~abs:1e-12 ()) (vec [| 1.; 0. |]) dlo);
    test
      "levenberg_marquardt's coordinate held at a lone lower bound follows it"
      (fun () ->
        let solve (c, lo) =
          Solution.get
            (Minimize.solve one lm ~lo ~tol:tight ~budget:100
               (fun x -> Nx.sub x c)
               (Nx.zeros_like c))
        in
        let c = vec [| -1.; 0.5 |] and lo = vec [| 0.; 0. |] in
        equal (Oracle.tensor ~abs:1e-12 ()) (vec [| 0.; 0.5 |]) (solve (c, lo));
        let dc, dlo =
          Rune.grad
            Nx.Ptree.(pair tensor tensor)
            (fun cl -> Nx.sum (solve cl))
            (c, lo)
        in
        equal (Oracle.tensor ~abs:1e-12 ()) (vec [| 0.; 1. |]) dc;
        equal (Oracle.tensor ~abs:1e-12 ()) (vec [| 1.; 0. |]) dlo);
    test "levenberg_marquardt holds a fit's parameter exactly at its bound"
      (fun () ->
        let y = model (vec [| 2.; -1.3 |]) in
        let p =
          Solution.get
            (Minimize.solve one lm
               ~lo:(vec [| 0.; -5. |])
               ~hi:(vec [| 10.; -1.5 |])
               ~tol:tight ~budget:100
               (fun p -> Nx.sub (model p) y)
               (vec [| 1.; -2. |]))
        in
        equal (Oracle.tensor ~abs:0. ()) (Nx.scalar f64 (-1.5)) (Nx.get [ 1 ] p));
    prop
      "levenberg_marquardt's boxed answer is its projected gradient's fixed \
       point"
      least_squares (fun (a, b, (lo, hi)) ->
        let n = Nx.dim 0 lo in
        let gradient x =
          Nx.matmul (Nx.transpose a) (Nx.sub (Nx.matmul a x) b)
        in
        let s =
          Minimize.solve one lm ~lo ~hi ~tol:tight ~budget:200
            (fun x -> Nx.sub (Nx.matmul a x) b)
            (Nx.zeros_like lo)
        in
        let x = Solution.get s in
        let clipped = Nx.minimum hi (Nx.maximum lo (Nx.sub x (gradient x))) in
        let held =
          Nx.logical_and
            (Nx.not_equal clipped (Nx.sub x (gradient x)))
            (Nx.less lo hi)
        in
        cover "a coordinate held at a bound, with another free"
          (n > 1 && Nx.item [] (Nx.any held) && not (Nx.item [] (Nx.all held)));
        cover "a coordinate closed to a point"
          (Nx.item [] (Nx.any (Nx.equal lo hi)));
        equal (Oracle.tensor ~abs:(1e-8 *. Float.of_int (n + 2)) ()) clipped x);
    test "levenberg_marquardt finds a minimum on a bound that couples"
      (fun () ->
        (* ½ |a x − b|² with x₀ ≤ 0: the unconstrained minimum (1, 1) projects
           to (0, 1), of cost 0.5, but the minimum in the box is (0, 1.5), of
           cost 0.25. From either start, the minimum in the box. *)
        let a = Nx.create f64 [| 2; 2 |] [| 1.; 1.; 0.; 1. |] in
        let b = vec [| 2.; 1. |] in
        let hi = vec [| 0.; Float.infinity |] in
        List.iter
          (fun x0 ->
            let s =
              Minimize.solve one lm ~hi ~tol:tight ~budget:200
                (fun x -> Nx.sub (Nx.matmul a x) b)
                (vec x0)
            in
            equal
              (Oracle.tensor ~abs:1e-10 ())
              (vec [| 0.; 1.5 |])
              (Solution.get s))
          [ [| -1.; 0. |]; [| 0.; 1.5 |] ]);
    test "nelder_mead clips its vertices into the box" (fun () ->
        let c = vec [| 2.; -0.25 |] in
        equal
          (Oracle.tensor ~abs:1e-6 ())
          (vec [| 1.; -0.25 |])
          (Solution.get
             (Minimize.solve one Minimize.nelder_mead
                ~lo:(Nx.full f64 [| 2 |] (-1.))
                ~hi:(Nx.ones f64 [| 2 |])
                ~tol:(Tol.v ~rel:1e-9 ~abs:1e-10)
                ~budget:2000
                (fun x -> Nx.sum (bowl c x))
                (Nx.zeros f64 [| 2 |]))));
    prop "nelder_mead converges only at a minimum in the box"
      ~examples:
        [
          (* One unknown, whose minimum 10/9 lies past its bound 1 and its wall
             0.5, where f is NaN. *)
          ( 0,
            ( Nx.create f64 [| 3; 1 |] [| 1.; 0.5; -1. |],
              vec [| 2.; 1.; 0. |],
              (vec [| Float.neg_infinity |], vec [| 1. |]) ),
            ((vec [| Float.neg_infinity |], vec [| 0.5 |]), (1., Float.nan)) );
          (* Three unknowns, whose minimum (2, 0.3, −2) lies past a wall of +∞
             at 1 above the first unknown here, and past a wall of −∞ at −1
             below the third in the next. *)
          ( 1,
            ( Nx.concatenate ~axis:0 [ Nx.eye f64 3; Nx.eye f64 3 ],
              vec [| 2.; 0.3; -2.; 2.; 0.3; -2. |],
              ( Nx.full f64 [| 3 |] Float.neg_infinity,
                Nx.full f64 [| 3 |] Float.infinity ) ),
            ( ( Nx.full f64 [| 3 |] Float.neg_infinity,
                vec [| 1.; Float.infinity; Float.infinity |] ),
              (1., Float.infinity) ) );
          ( 2,
            ( Nx.concatenate ~axis:0 [ Nx.eye f64 3; Nx.eye f64 3 ],
              vec [| 2.; 0.3; -2.; 2.; 0.3; -2. |],
              ( Nx.full f64 [| 3 |] Float.neg_infinity,
                Nx.full f64 [| 3 |] Float.infinity ) ),
            ( ( vec [| Float.neg_infinity; Float.neg_infinity; -1. |],
                Nx.full f64 [| 3 |] Float.infinity ),
              (1., Float.neg_infinity) ) );
        ]
      walled
      (fun (k, (a, b, (lo, hi)), ((below, above), (c, bad))) ->
        Compiled.here ();
        (* f is non-finite past the walls, so its minimum is in the box whose
           sides are the nearer of each bound and its wall. *)
        let n = Nx.dim 0 lo in
        let domain = ((below, above), (Nx.scalar f64 c, Nx.scalar f64 bad)) in
        let s = simplex_programs.(k) a b (lo, hi) domain in
        let x = Solution.best s and converged = Oracle.status s = Converged in
        let bottom = Nx.maximum lo below and top = Nx.minimum hi above in
        let step = Nx.sub x (squares_gradient a b x) in
        let clipped = Nx.minimum top (Nx.maximum bottom step) in
        let held =
          Nx.logical_and (Nx.not_equal clipped step) (Nx.less bottom top)
        in
        let holds past wall bound =
          Nx.item []
            (Nx.any (Nx.logical_and (past step wall) (past bound wall)))
        in
        let walled_above = holds Nx.greater above hi in
        let walled_below = holds Nx.less below lo in
        let name, _, _, _ = simplex_tols.(k) in
        cover "a coordinate held at a bound, with another free"
          (n > 1 && Nx.item [] (Nx.any held) && not (Nx.item [] (Nx.all held)));
        Array.iter
          (fun (tol, _, _, _) ->
            cover ("it converges under " ^ tol) (converged && name = tol))
          simplex_tols;
        List.iter
          (fun v ->
            cover
              (Printf.sprintf "a wall of %g holds a converged answer" v)
              (converged && (walled_above || walled_below) && Float.equal v bad))
          [ Float.nan; Float.infinity; Float.neg_infinity ];
        cover "a wall below holds a converged answer" (converged && walled_below);
        List.iter
          (fun v ->
            cover
              (Printf.sprintf "it converges with f scaled by %g" v)
              (converged && c = v))
          [ 1e170; 1e-170 ];
        if converged then
          equal (Oracle.tensor ~abs:(stationary k a b x) ()) clipped x);
    cases "nelder_mead leaves a bound it starts on"
      ~id:(fun k ->
        let name, _, _, _ = simplex_tols.(k) in
        name)
      (List.init (Array.length simplex_tols) Fun.id)
      (fun k ->
        Compiled.here ();
        (* ½ (x² + (x + 1)²) with x ≤ 0, from 0: the minimum is −1/2. The first
           simplex's step from 0 is clipped back to 0, so every vertex lies on
           the bound and only a probe toward −∞ finds the descent. A relative
           tolerance's scale is 0 there. *)
        let a = Nx.create f64 [| 2; 1 |] [| 1.; 1. |]
        and b = vec [| 0.; -1. |] in
        let s =
          simplex_programs.(k) a b
            (vec [| Float.neg_infinity |], vec [| 0. |])
            (plain 1)
        in
        equal Oracle.status_t Converged (Oracle.status s);
        let x = Solution.best s in
        equal (Oracle.tensor ~abs:(stationary k a b x) ()) (vec [| -0.5 |]) x);
    cases "a box that holds no point ends the lane Stalled" ~id:fst
      (let inf = vec [| 0.; Float.infinity |]
       and nan = vec [| 0.; Float.nan |] in
       [
         ("lo above hi", (Some (vec [| 1.; 0. |]), Some (vec [| 0.; 1. |])));
         ("lo of +∞", (Some inf, None));
         ("hi of −∞", (None, Some (Nx.neg inf)));
         ("lo of NaN", (Some nan, None));
         ("hi of NaN", (None, Some nan));
       ])
      (fun (_, (lo, hi)) ->
        let c = vec [| 1.; -0.25 |] in
        List.iter
          (fun (name, m) ->
            equal ~msg:name Oracle.status_t Stalled
              (Oracle.status
                 (Minimize.solve one m ?lo ?hi ~tol:tight ~budget:20
                    (fun x -> Nx.sum (bowl c x))
                    (Nx.zeros_like c))))
          (("nelder_mead", Minimize.nelder_mead) :: gradient_methods);
        equal ~msg:"levenberg_marquardt" Oracle.status_t Stalled
          (Oracle.status
             (Minimize.solve one lm ?lo ?hi ~tol:tight ~budget:20
                (fun x -> Nx.sub x c)
                (Nx.zeros_like c))));
    cases "a compiled boxed search converges where eager does" ~id:fst
      gradient_methods (fun (_, m) ->
        Compiled.here ();
        (* Steps in a box round within rune's numerics compiled (jera.mli, Tol),
           so both answers meet [tight] apart: within [2 √3] scales of each
           other. *)
        let f c =
          let lo, hi = unit_box c in
          Solution.get
            (Minimize.solve one m ~lo ~hi ~tol:tight ~budget:200
               (fun x ->
                 Nx.add
                   (Nx.sum (Nx.square (Nx.sub x c)))
                   (Nx.sum (Nx.square (Nx.square x))))
               (Nx.zeros_like c))
        in
        let c = vec [| 1.; -0.25; 2. |] in
        equal (Oracle.tensor ~rel:4e-10 ~abs:4e-12 ()) (f c) (Rune.jit' f c));
    prop ~count:10
      "a NaN at the trial points ends a search within 1 + 3p budget evaluations"
      centers (fun c ->
        cover "the minimum past the NaN" (Nx.item [] (Nx.sum (Nx.square c)) > 1.);
        (* Finite inside the unit ball around 0, NaN outside: trial points past
           it are NaN, and the minimum at c may lie outside. *)
        let f x =
          let v = Nx.sum (bowl c x) in
          Nx.where
            (Nx.less_equal_s (Nx.sum (Nx.square x)) 1.)
            v (Nx.full_like v Float.nan)
        in
        List.iter
          (fun (lo, hi) ->
            List.iter
              (fun (_, m) ->
                let s =
                  Minimize.solve one m ?lo ?hi ~tol:tight ~budget:8 f
                    (Nx.zeros_like c)
                in
                at_most int32 ~than:(evaluation_bound 8)
                  (Nx.item [] (Solution.evaluations s)))
              dense_methods)
          [
            (None, None); (Some (Nx.full_like c (-2.)), Some (Nx.full_like c 2.));
          ]);
    prop ~count:10
      "a kink that collapses the line search ends it within 1 + 3p budget \
       evaluations"
      centers (fun c ->
        (* |x − c| has no derivative at its minimum: the line search's bracket
           shrinks onto the kink without the curvature condition holding. *)
        let f x = Nx.sum (Nx.abs (Nx.sub x c)) in
        List.iter
          (fun (lo, hi) ->
            List.iter
              (fun (_, m) ->
                let s =
                  Minimize.solve one m ?lo ?hi ~tol:tight ~budget:8 f
                    (Nx.zeros_like c)
                in
                at_most int32 ~than:(evaluation_bound 8)
                  (Nx.item [] (Solution.evaluations s)))
              dense_methods)
          [
            (None, None); (Some (Nx.full_like c (-1.)), Some (Nx.full_like c 1.));
          ]);
    test "budget below 1 raises" (fun () ->
        raises_in "Jera.Minimize.solve" (fun () ->
            Minimize.solve one
              (Minimize.bfgs ~linear:Linear.dense)
              ~tol:tight ~budget:0 rosenbrock
              (vec [| 0.; 0. |])));
    cases "f returning other than a scalar raises" ~id:fst
      [
        ("bfgs", Minimize.bfgs ~linear:Linear.dense);
        ("nelder_mead", Minimize.nelder_mead);
      ]
      (fun (_, m) ->
        raises_in "Jera.Minimize.solve" (fun () ->
            Minimize.solve one m ~tol:tight ~budget:10 Fun.id (vec [| 0.; 0. |])));
    test "float tensors of two dtypes in x0 raise" (fun () ->
        raises_in "Jera.Minimize.solve" (fun () ->
            Minimize.solve
              Nx.Ptree.(pair tensor tensor)
              Minimize.nelder_mead ~tol:tight ~budget:10
              (fun (a, b) -> Nx.add (Nx.sum a) (Nx.cast f64 (Nx.sum b)))
              (Nx.zeros f64 [| 2 |], Nx.zeros Nx.float32 [| 1 |])));
    test "a side not of the start's shapes raises, naming the path" (fun () ->
        (* Swapped shapes keep the size: without the check they bound other
           coordinates. *)
        (* D6: the function and the path jera.mli says it names. *)
        let names_path path f =
          raises_match
            (fun e ->
              Exn.invalid_arg ~sub:"Jera.Minimize.solve" e
              && Exn.invalid_arg ~sub:path e)
            f
        in
        let x = Nx.Ptree.(pair tensor tensor) in
        let start = (Nx.zeros f64 [| 2 |], Nx.zeros f64 [| 1 |]) in
        let f (a, b) = Nx.add (Nx.sum (Nx.square a)) (Nx.sum (Nx.square b)) in
        let hi = (Nx.ones f64 [| 2 |], Nx.ones f64 [| 1 |]) in
        let solve m lo =
          Minimize.solve x m ~lo ~hi ~tol:tight ~budget:20 f start
        in
        List.iter
          (fun m ->
            names_path "0:" (fun () ->
                solve m (Nx.zeros f64 [| 1 |], Nx.zeros f64 [| 2 |]));
            names_path "0:" (fun () ->
                solve m (Nx.scalar f64 0., Nx.zeros f64 [| 1 |])))
          [
            Minimize.lbfgs ~memory:3 ~linear:Linear.dense; Minimize.nelder_mead;
          ];
        names_path "0:" (fun () ->
            Minimize.solve x
              (Minimize.levenberg_marquardt x ~linear:Linear.dense)
              ~lo:(Nx.zeros f64 [| 1 |], Nx.zeros f64 [| 2 |])
              ~hi ~tol:tight ~budget:20 Fun.id start);
        names_path "the root:" (fun () ->
            let x = Nx.Ptree.(list tensor) and one = Nx.ones f64 [| 2 |] in
            Minimize.solve x Minimize.nelder_mead ~lo:[ one; one ] ~tol:tight
              ~budget:20
              (fun l -> Nx.sum (Nx.stack l))
              [ one ]));
  ]

let iterates_tests =
  [
    cases "the first iterate is the start, then one row per step" ~id:fst
      (("nelder_mead", Minimize.nelder_mead) :: gradient_methods) (fun (_, m) ->
        let xs =
          Minimize.iterates one m ~steps:5 rosenbrock (vec [| -1.2; 1. |])
        in
        equal (array int) [| 5; 2 |] (Nx.shape xs);
        equal (Oracle.tensor ()) (vec [| -1.2; 1. |]) (Nx.get [ 0 ] xs));
    test "one step is the start alone" (fun () ->
        equal (Oracle.tensor ())
          (Nx.create f64 [| 1; 2 |] [| -1.2; 1. |])
          (Minimize.iterates one
             (Minimize.bfgs ~linear:Linear.dense)
             ~steps:1 rosenbrock
             (vec [| -1.2; 1. |])));
    test "newton reaches a quadratic's minimum in one step and repeats it"
      (fun () ->
        let a = Nx.create f64 [| 2; 2 |] [| 2.; 0.5; 0.5; 1. |] in
        let b = vec [| 1.; -1. |] in
        let f x =
          Nx.sub
            (Nx.mul_s (Nx.sum (Nx.mul x (Nx.matmul a x))) 0.5)
            (Nx.sum (Nx.mul b x))
        in
        let xs =
          Minimize.iterates one
            (Minimize.newton ~linear:Linear.dense)
            ~steps:4 f
            (vec [| 0.; 0. |])
        in
        (* Newton's first step solves a quadratic; then the lane stops at a zero
           gradient. *)
        List.iter
          (fun i ->
            equal
              (Oracle.tensor ~rel:1e-12 ~abs:1e-15 ())
              (Nx.solve a b) (Nx.get [ i ] xs))
          [ 1; 2; 3 ]);
    test "L-BFGS's iterates reach Rosenbrock's minimum" (fun () ->
        let m = Minimize.lbfgs ~memory:6 ~linear:Linear.dense in
        let xs =
          Minimize.iterates one m ~steps:60 rosenbrock (vec [| -1.2; 1. |])
        in
        equal (Oracle.tensor ~rel:1e-6 ()) (vec [| 1.; 1. |]) (Nx.get [ 59 ] xs));
    test "levenberg_marquardt's iterates go from the start to the fit"
      (fun () ->
        let y = model (vec [| 2.; -1.3 |]) in
        let xs =
          Minimize.iterates one lm ~steps:40
            (fun p -> Nx.sub (model p) y)
            (vec [| 1.; 0. |])
        in
        equal (Oracle.tensor ()) (vec [| 1.; 0. |]) (Nx.get [ 0 ] xs);
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (vec [| 2.; -1.3 |])
          (Nx.get [ 39 ] xs));
    cases "the iterates of zero unknowns are rows of none" ~id:fst
      (("nelder_mead", Minimize.nelder_mead) :: gradient_methods)
      (fun (_, m) ->
        equal (Oracle.tensor ())
          (Nx.zeros f64 [| 3; 0 |])
          (Minimize.iterates one m ~steps:3
             (fun x -> Nx.sum (Nx.square x))
             (vec [||])));
    test "levenberg_marquardt's iterates of zero unknowns are rows of none"
      (fun () ->
        equal (Oracle.tensor ())
          (Nx.zeros f64 [| 3; 0 |])
          (Minimize.iterates one lm ~steps:3 Fun.id (vec [||])));
    test "the iterates are detached" (fun () ->
        equal (Oracle.tensor ~abs:0. ())
          (vec [| 0.; 0. |])
          (Rune.grad'
             (fun c ->
               Nx.sum
                 (Minimize.iterates one
                    (Minimize.bfgs ~linear:Linear.dense)
                    ~steps:3
                    (fun x -> Nx.sum (bowl c x))
                    (Nx.zeros_like c)))
             (vec [| 1.; 2. |])));
    test "a tensor that is not a float is the start's, repeated" (fun () ->
        let _, k =
          Minimize.iterates
            Nx.Ptree.(pair tensor tensor)
            (Minimize.bfgs ~linear:Linear.dense)
            ~steps:3
            (fun (x, _) -> rosenbrock x)
            (vec [| -1.2; 1. |], Nx.scalar Nx.int32 7l)
        in
        equal (Oracle.tensor ()) (Nx.create Nx.int32 [| 3 |] [| 7l; 7l; 7l |]) k);
    prop "f never rises from one estimate to the next" convex (fun (a, b) ->
        (* bfgs and lbfgs accept a step at the strong Wolfe conditions, newton
           at a sufficient decrease, and nelder_mead's best vertex gives way
           only to a lower one (jera.mli). Six steps stay short of the floats'
           limit, where a step may raise f by its rounding: [4 (n + 1)² ε] times
           the magnitude of f's terms bounds that rounding. *)
        let n = Nx.dim 0 b in
        let f = objective a b in
        let rounding x =
          let terms =
            Nx.item []
              (Nx.add
                 (Nx.sum (Nx.mul (Nx.abs x) (Nx.matmul (Nx.abs a) (Nx.abs x))))
                 (Nx.add
                    (Nx.sum (Nx.log (Nx.cosh x)))
                    (Nx.sum (Nx.abs (Nx.mul b x)))))
          in
          4.
          *. Float.of_int ((n + 1) * (n + 1))
          *. epsilon_float *. (1. +. terms)
        in
        List.iter
          (fun (name, m) ->
            let xs = Minimize.iterates one m ~steps:6 f (Nx.zeros_like b) in
            for k = 1 to 5 do
              let before = Nx.get [ k - 1 ] xs and after = Nx.get [ k ] xs in
              at_most float_exact ~msg:name
                ~than:(Nx.item [] (f before) +. rounding before)
                (Nx.item [] (f after))
            done)
          (("nelder_mead", Minimize.nelder_mead) :: gradient_methods));
    prop "levenberg_marquardt's |r|² / 2 never rises past its rounding"
      least_squares (fun (a, b, _) ->
        (* A step is accepted at a decrease of [|r|² / 2], measured or, within
           its rounding [√ε |r|² / 2], estimated (jera.mli). *)
        let cost x =
          0.5 *. Nx.item [] (Nx.sum (Nx.square (Nx.sub (Nx.matmul a x) b)))
        in
        let xs =
          Minimize.iterates one lm ~steps:6
            (fun x -> Nx.sub (Nx.matmul a x) b)
            (Nx.zeros f64 [| Nx.dim 1 a |])
        in
        for k = 1 to 5 do
          let before = cost (Nx.get [ k - 1 ] xs) in
          at_most float_exact
            ~than:(before *. (1. +. Float.sqrt epsilon_float))
            (cost (Nx.get [ k ] xs))
        done);
    test "steps below 1 raise" (fun () ->
        raises_in "Jera.Minimize.iterates" (fun () ->
            Minimize.iterates one
              (Minimize.bfgs ~linear:Linear.dense)
              ~steps:0 rosenbrock
              (vec [| 0.; 0. |])));
    test "f returning other than a scalar raises as solve does" (fun () ->
        raises_in "Jera.Minimize.iterates" (fun () ->
            Minimize.iterates one
              (Minimize.bfgs ~linear:Linear.dense)
              ~steps:3 Fun.id
              (vec [| 0.; 0. |])));
  ]

(* Brent's search evaluates no two points closer than √ε |x| plus a third of
   [tol]'s scale, so a bracket meets a tolerance of about 6 √ε |x|, 9e-8 |x| in
   float64, or coarser. *)
let brent = Tol.v ~rel:1e-6 ~abs:1e-9

(* A converged bracket holds the minimum and its estimate within twice [brent]'s
   scale, [2 (1e-6 |x| + 1e-9)], which is at most [4] times the larger term. *)
let bracketed () = Oracle.tensor ~rel:4e-6 ~abs:4e-9 ()

let bowl_bracket c =
  Minimize.bracket ~tol:brent (bowl c) ~lo:(Nx.sub_s c 4.) ~hi:(Nx.add_s c 1.)

let bracket_tests =
  [
    prop "the minimum of a bowl is its center" centers (fun c ->
        equal (bracketed ()) c (Solution.get (bowl_bracket c)));
    prop
      "an element ends within its budget of 3b + 9 evaluations, 201 in float64"
      centers (fun c ->
        Array.iter
          (fun n -> at_most int32 ~than:201l n)
          (Nx.to_array (Solution.evaluations (bowl_bracket c))));
    test "a float32 element ends within 3b + 9 = 105 evaluations" (fun () ->
        let c = Nx.create Nx.float32 [| 3 |] [| -2.5; 0.25; 1.75 |] in
        let s =
          Minimize.bracket
            ~tol:(Tol.v ~rel:1e-3 ~abs:1e-3)
            (bowl c) ~lo:(Nx.sub_s c 4.) ~hi:(Nx.add_s c 1.)
        in
        Array.iter
          (fun n -> at_most int32 ~than:105l n)
          (Nx.to_array (Solution.evaluations s)));
    test "a bracket too wide for 3b + 9 evaluations ends Budget_spent"
      (fun () ->
        (* Golden cuts of [−1e150, 1e150] reach the minimum's tolerance in about
           2 log(1e150 / 1e-10) / log(1 / 0.618) evaluations, past the 3b + 9 =
           201 of a float64 element (jera.mli, bracket). *)
        let wide f bound =
          let s =
            Minimize.bracket ~tol:brent f ~lo:(vec [| -.bound |])
              ~hi:(vec [| bound |])
          in
          equal Oracle.status_t Budget_spent (Oracle.status s);
          equal int32 201l (Nx.item [ 0 ] (Solution.evaluations s))
        in
        wide (fun x -> Nx.square (Nx.sub_s x 0.5)) 1e150;
        wide (fun x -> Nx.abs (Nx.sub_s x 0.5)) 1e100);
    prop "a tolerance finer than the minimum's resolution ends Stalled"
      Gen.(
        map
          (fun (m, up) -> vec [| (if up then m else -.m) |])
          (pair (float_range 0.5 2.) bool))
      (fun c ->
        (* 1 + (x − c)² rounds by ε near its minimum, so its values fix [c] only
           to about √ε, 7e7 ulps of float64 (jera.mli, Tol.ulps); 4 ulps cannot
           be met. *)
        let f x = Nx.add_s (Nx.square (Nx.sub x c)) 1. in
        let s =
          Minimize.bracket ~tol:(Tol.ulps 4.) f ~lo:(Nx.sub_s c 4.)
            ~hi:(Nx.add_s c 1.)
        in
        equal Oracle.status_t Stalled (Oracle.status s));
    test "a relative tolerance at a minimum at 0 ends Budget_spent" (fun () ->
        (* A relative tolerance's scale is 0 at 0, where no error but 0 is
           accepted (jera.mli, Tol): no bracket of floats is narrow enough. *)
        equal Oracle.status_t Budget_spent
          (Oracle.status
             (Minimize.bracket ~tol:(Tol.rel 1e-6) Nx.square ~lo:(vec [| -1. |])
                ~hi:(vec [| 2. |]))));
    test "cos has its minimum at π in [0, 5]" (fun () ->
        let s =
          Minimize.bracket ~tol:brent Nx.cos ~lo:(vec [| 0. |])
            ~hi:(vec [| 5. |])
        in
        equal (bracketed ()) (vec [| Float.pi |]) (Solution.get s));
    test "with several minima it finds one of them" (fun () ->
        (* cos has minima at π and 3π in [0, 4π]. *)
        let s =
          Minimize.bracket ~tol:brent Nx.cos ~lo:(vec [| 0. |])
            ~hi:(vec [| 4. *. Float.pi |])
        in
        let x = Nx.item [ 0 ] (Solution.get s) in
        satisfies ~claim:"within 4e-6 of π or 3π" float_exact
          (fun x ->
            Float.abs (x -. Float.pi) <= 4e-6 *. Float.pi
            || Float.abs (x -. (3. *. Float.pi)) <= 4e-6 *. 3. *. Float.pi)
          x);
    test "a minimum at an end is that end, with ends in either order" (fun () ->
        let s =
          Minimize.bracket ~tol:brent Fun.id
            ~lo:(vec [| 1.; 3. |])
            ~hi:(vec [| 2.; -1. |])
        in
        equal (Oracle.tensor ()) (vec [| 1.; -1. |]) (Solution.get s));
    test "the ends broadcast together" (fun () ->
        let s =
          Minimize.bracket ~tol:brent Nx.cos ~lo:(Nx.scalar f64 0.)
            ~hi:(vec [| 4.; 5.; 6. |])
        in
        equal (bracketed ()) (Nx.full f64 [| 3 |] Float.pi) (Solution.get s));
    test "ends that do not broadcast raise" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Minimize.bracket") (fun () ->
            Minimize.bracket ~tol:brent Nx.cos ~lo:(Nx.zeros f64 [| 2 |])
              ~hi:(Nx.ones f64 [| 3 |])));
    test "a NaN value ends the element Not_finite" (fun () ->
        let s =
          Minimize.bracket ~tol:brent Nx.log ~lo:(vec [| -2. |])
            ~hi:(vec [| -1. |])
        in
        equal Oracle.status_t Not_finite (Oracle.status s));
    prop "grad of the minimum in its center is 1" centers (fun c ->
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.ones_like c)
          (Rune.grad' (fun c -> Nx.sum (Solution.get (bowl_bracket c))) c));
    test "grad of a minimum at an end is the end's" (fun () ->
        let lo = vec [| 1.; 0.5 |] in
        let g =
          Rune.grad'
            (fun lo ->
              Nx.sum
                (Solution.get
                   (Minimize.bracket ~tol:brent Fun.id ~lo ~hi:(Nx.add_s lo 2.))))
            lo
        in
        equal (Oracle.tensor ()) (Nx.ones_like lo) g);
    test "grad in a captured scale is the implicit derivative" (fun () ->
        (* x² − θ x has its minimum at θ / 2. *)
        let theta = vec [| 1.; -2.; 0.5 |] in
        let solve t =
          Solution.get
            (Minimize.bracket ~tol:brent
               (fun x -> Nx.sub (Nx.square x) (Nx.mul t x))
               ~lo:(Nx.full_like t (-5.)) ~hi:(Nx.full_like t 5.))
        in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.full_like theta 0.5)
          (Rune.grad' (fun t -> Nx.sum (solve t)) theta));
    test "compiled is eager, bit for bit" (fun () ->
        Compiled.here ();
        let c = vec [| -1.; 0.25; 2. |] in
        let f c = Solution.get (bowl_bracket c) in
        equal (Oracle.tensor ()) (f c) (Rune.jit' f c));
    test "compiled grad is eager grad to its rounding" (fun () ->
        Compiled.here ();
        (* The answers agree to the bit (jera.mli, Tol); the derivative's
           quotient [−∂²f/∂x∂θ / ∂²f/∂x²] at them may round its few terms apart
           compiled, a few ulps: 1e-12. *)
        let c = vec [| -1.; 0.25; 2. |] in
        let g = Rune.grad' (fun c -> Nx.sum (Solution.get (bowl_bracket c))) in
        equal (Oracle.tensor ~rel:1e-12 ()) (g c) (Rune.jit' g c));
    test "vmap is each lane's search" (fun () ->
        let c = Nx.create f64 [| 2; 2 |] [| -1.; 0.25; 2.; 0. |] in
        let f c = Solution.get (bowl_bracket c) in
        equal (Oracle.tensor ()) (f c) (Rune.vmap' f c));
  ]

let () =
  exit
    (run "Jera.Minimize"
       [
         group "minima" minima_tests;
         group "error" error_tests;
         group "derivative" derivative_tests;
         group "bfgs" bfgs_tests;
         group "lbfgs" lbfgs_tests;
         group "newton" newton_tests;
         group "levenberg_marquardt" levenberg_marquardt_tests;
         group "nelder_mead" nelder_mead_tests;
         group "solve" solve_tests;
         group "iterates" iterates_tests;
         group "bracket" bracket_tests;
       ])
