(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Piecewise: one series value for interpolants and fits. The trusted side
   is scipy's cubic splines, the formulas of the broken line and of the cubic
   Hermite interpolant, the end conditions jera.mli states, and the laws of
   calculus checked against rune's derivative of evaluation. *)

open Windtrap
open Jera
module G = Golden_piecewise

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let exact () = Oracle.tensor ()

(* scipy's splines and ours solve the same diagonally dominant system in another
   order: a few ulps of the values, which are near 1. *)
let close () = Oracle.tensor ~rel:1e-13 ~abs:1e-14 ()
let knots = vec G.knots
let samples = vec G.samples
let samples2 = Nx.create f64 [| Array.length G.knots; 2 |] G.samples2
let points = vec G.points
let raises_in fn f = raises_match (Exn.invalid_arg ~sub:fn) f

(* The broken line through (0, 0), (1, 2) and (2, 3). *)
let broken = Piecewise.linear (vec [| 0.; 1.; 2. |]) (vec [| 0.; 2.; 3. |])

(* The values at each interior break of the piece that ends there and of the
   piece that starts there, for a series of [n] pieces. *)
let both_sides p n =
  let ends = Piecewise.eval_at p (Nx.arange Nx.int64 0 (n - 1) 1) in
  let starts = Piecewise.eval_at p (Nx.arange Nx.int64 1 n 1) in
  (ends (Nx.ones f64 [| n - 1 |]), starts (Nx.full f64 [| n - 1 |] (-1.)))

let rec derivatives k p =
  if k = 0 then p else derivatives (k - 1) (Piecewise.derivative p)

(* Samples *)

let floats =
  Format.pp_print_array ~pp_sep:Format.pp_print_space Format.pp_print_float

(* [n] knots, 2 to 9, with gaps in [0.05, 1], samples in [−3, 3] and five points
   in the domain, each a knot with probability 1/4. *)
let data =
  Gen.(
    let* n =
      frequency [ (1, constant 2); (1, constant 3); (2, int_range 4 9) ]
    in
    let* gaps = array ~size:(constant (n - 1)) (float_range 0.05 1.) in
    let* y = array ~size:(constant n) (float_range (-3.) 3.) in
    let x = Array.make n 0. in
    Array.iteri (fun i g -> x.(i + 1) <- x.(i) +. g) gaps;
    let point =
      frequency
        [
          (1, map (fun i -> x.(i)) (int_range 0 (n - 1)));
          (3, float_range 0. x.(n - 1));
        ]
    in
    let+ pts = array ~size:(constant 5) point in
    (x, y, pts))
  |> Gen.with_pp (fun ppf (x, y, pts) ->
      Format.fprintf ppf "@[knots [%a]@ samples [%a]@ points [%a]@]" floats x
        floats y floats pts)

(* The piece [i] whose interval holds [t], and the fraction [s] of the way. *)
let locate x t =
  let i = ref 0 in
  while !i < Array.length x - 2 && t > x.(!i + 1) do
    incr i
  done;
  (!i, (t -. x.(!i)) /. (x.(!i + 1) -. x.(!i)))

let largest a = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. a

let narrowest x =
  let m = ref infinity in
  Array.iteri (fun i v -> if i > 0 then m := Float.min !m (v -. x.(i - 1))) x;
  !m

(* A cubic spline's k-th derivative is about [y_max / h^k] for samples up to
   [y_max] and gaps down to [h], and its system, diagonally dominant with gaps
   in a ratio up to 20, has a condition below 60: it rounds below [1e3 n ε y_max
   / h^k]. *)
let spline_bound (x, y, _) k =
  let n = Array.length x in
  1e3 *. float_of_int n *. epsilon_float
  *. (largest y +. 1.)
  /. (narrowest x ** float_of_int k)

(* Series *)

(* Random series: 1 to 4 pieces of degree 0 to 8, degree 0 with probability 1/4,
   over breaks in [-2, 2] whose later pieces are empty with probability 1/5,
   coefficients in [-1, 1], and five points in the domain, each a break with
   probability 1/4. *)
let series =
  Gen.(
    let* pieces = int_range 1 4 in
    let* degree = frequency [ (1, constant 0); (3, int_range 1 8) ] in
    let* first = float_range 0.1 1. in
    let* later =
      array
        ~size:(constant (pieces - 1))
        (frequency [ (1, constant 0.); (4, float_range 0.1 1.) ])
    in
    let* start = float_range (-2.) 0. in
    let* c =
      array ~size:(constant (pieces * (degree + 1))) (float_range (-1.) 1.)
    in
    let breaks = Array.make (pieces + 1) start in
    Array.iteri
      (fun i w -> breaks.(i + 1) <- breaks.(i) +. w)
      (Array.append [| first |] later);
    let point =
      frequency
        [
          (1, map (fun i -> breaks.(i)) (int_range 0 pieces));
          (3, float_range start breaks.(pieces));
        ]
    in
    let+ pts = array ~size:(constant 5) point in
    (breaks, Nx.create f64 [| pieces; degree + 1 |] c, pts))
  |> Gen.with_pp (fun ppf (b, c, x) ->
      Format.fprintf ppf "@[breaks [%a]@ coefficients %a@ points [%a]@]" floats
        b Nx.pp c floats x)

let of_series (breaks, c, _) =
  Piecewise.v Nx.Ptree.tensor ~breaks:(vec breaks) c

let cover_series (breaks, c, pts) =
  let empty = ref false in
  Array.iteri
    (fun i b -> if i > 0 && b = breaks.(i - 1) then empty := true)
    breaks;
  cover "degree 0" (Nx.dim 1 c = 1);
  cover "one piece" (Array.length breaks = 2);
  cover "an empty piece" !empty;
  cover "a point on a break" (Array.exists (fun t -> Array.mem t breaks) pts)

(* 16 (degree + 2)² ε Σ |c_k|, scaled by the narrowest nonempty piece's
   derivative factor (2 / width)^order. *)
let bound (breaks, c, _) ~order =
  let degree = Nx.dim 1 c - 1 in
  let sum =
    Array.fold_left (fun acc x -> acc +. Float.abs x) 0. (Nx.to_array c)
  in
  let narrow = ref infinity in
  Array.iteri
    (fun i b ->
      if i > 0 && b > breaks.(i - 1) then
        narrow := Float.min !narrow (b -. breaks.(i - 1)))
    breaks;
  let scale = (2. /. !narrow) ** float_of_int order in
  float_of_int ((degree + 2) * (degree + 2))
  *. epsilon_float *. (sum +. 1.) *. Float.max scale 1. *. 16.

(* Values *)

let v_tests =
  let three = Nx.create f64 [| 3; 1 |] [| 1.; 2.; 3. |] in
  let v breaks c = Piecewise.v Nx.Ptree.tensor ~breaks:(vec breaks) c in
  [
    test "rebuilds an adapt answer from its breaks and coefficients" (fun () ->
        let p =
          Solution.get
            (Piecewise.adapt Nx.Ptree.tensor ~degree:8 ~tol:(Tol.rel 1e-10)
               ~budget:16 Nx.exp (scalar 0.) (scalar 2.))
        in
        let q =
          Piecewise.v Nx.Ptree.tensor ~breaks:(Piecewise.breaks p)
            (Piecewise.coefficients p)
        in
        let x = Nx.linspace f64 0. 2. 37 in
        equal (exact ()) (Piecewise.eval p x) (Piecewise.eval q x));
    test "takes empty pieces" (fun () ->
        equal (exact ())
          (vec [| 1.; 1.; 3. |])
          (Piecewise.eval
             (v [| 0.; 1.; 1.; 2. |] three)
             (vec [| 0.5; 1.; 1.5 |])));
    test "the first break of an empty first piece lies in the next piece"
      (fun () ->
        equal (exact ()) (vec [| 2. |])
          (Piecewise.eval (v [| 0.; 0.; 1.; 2. |] three) (vec [| 0. |])));
    test "leaves of other degrees and dtypes each evaluate their own series"
      (fun () ->
        (* In the float64 leaf piece 0 is T₂ and piece 1 T₁; in the float32 leaf
           they are 1 and 2; u = 0 at 0.5 and 1.5. *)
        let s = Nx.Ptree.(pair tensor tensor) in
        let p =
          Piecewise.v s
            ~breaks:(vec [| 0.; 1.; 2. |])
            ( Nx.create f64 [| 2; 3 |] [| 0.; 0.; 1.; 0.; 1.; 0. |],
              Nx.create Nx.float32 [| 2; 1 |] [| 1.; 2. |] )
        in
        equal (Oracle.structure s)
          (vec [| -1.; 0. |], Nx.create Nx.float32 [| 2 |] [| 1.; 2. |])
          (Piecewise.eval p (vec [| 0.5; 1.5 |])));
    test "breaks that decrease raise" (fun () ->
        raises_in "Jera.Piecewise.v" (fun () -> v [| 0.; 1.; 0.5; 2. |] three));
    test "a first break equal to the last raises" (fun () ->
        raises_in "Jera.Piecewise.v" (fun () -> v [| 1.; 1.; 1.; 1. |] three));
    cases "breaks not 1-D with two elements raise" ~id:fst
      [ ("one break", vec [| 0. |]); ("two axes", Nx.zeros f64 [| 2; 2 |]) ]
      (fun (_, breaks) ->
        raises_in "Jera.Piecewise.v" (fun () ->
            Piecewise.v Nx.Ptree.tensor ~breaks (Nx.zeros f64 [| 1; 2 |])));
    cases "a leaf without pieces rows and two axes raises" ~id:fst
      [
        ("3 rows for 2 pieces", Nx.zeros f64 [| 3; 2 |]);
        ("one axis", Nx.zeros f64 [| 2 |]);
      ]
      (fun (_, c) ->
        raises_in "Jera.Piecewise.v" (fun () -> v [| 0.; 1.; 2. |] c));
    test "an int32 leaf raises" (fun () ->
        raises_in "Jera.Piecewise.v" (fun () ->
            Piecewise.v Nx.Ptree.tensor
              ~breaks:(vec [| 0.; 1. |])
              (Nx.zeros Nx.int32 [| 1; 2 |])));
  ]

(* Interpolants *)

let linear_tests =
  [
    prop "is the broken line through the samples" data (fun (x, y, pts) ->
        cover "two knots" (Array.length x = 2);
        cover "a point on a knot" (Array.exists (fun t -> Array.mem t x) pts);
        let line t =
          let i, s = locate x t in
          y.(i) +. (s *. (y.(i + 1) -. y.(i)))
        in
        (* A value rounds in a few operations of the samples' size: 8 ε. *)
        equal
          (Oracle.tensor
             ~abs:((8. *. epsilon_float *. largest y) +. Float.min_float)
             ())
          (vec (Array.map line pts))
          (Piecewise.eval (Piecewise.linear (vec x) (vec y)) (vec pts)));
    prop "every interpolant passes through its samples" data
      (fun ((x, y, _) as d) ->
        cover "two knots" (Array.length x = 2);
        cover "three knots" (Array.length x = 3);
        let x = vec x and y = vec y in
        List.iter
          (fun (name, p) ->
            equal ~msg:name
              (Oracle.tensor ~abs:(spline_bound d 0) ())
              y (Piecewise.eval p x))
          [
            ("linear", Piecewise.linear x y);
            ("natural", Piecewise.cubic `Natural x y);
            ("not-a-knot", Piecewise.cubic `Not_a_knot x y);
            ("clamped", Piecewise.cubic (`Clamped (scalar 1., scalar 0.)) x y);
            ("steffen", Piecewise.steffen x y);
            ("hermite", Piecewise.hermite x ~values:y ~slopes:(Nx.zeros_like y));
          ]);
    test "knots that are not strictly increasing raise" (fun () ->
        raises_in "Jera.Piecewise.linear" (fun () ->
            Piecewise.linear (vec [| 0.; 1.; 1. |]) (vec [| 0.; 1.; 2. |])));
    cases "knots not 1-D with two knots raise" ~id:fst
      [ ("one knot", vec [| 0. |]); ("two axes", Nx.zeros f64 [| 2; 2 |]) ]
      (fun (_, x) ->
        raises_in "Jera.Piecewise.linear" (fun () ->
            Piecewise.linear x (Nx.zeros f64 [| 2 |])));
    test "samples of another length raise" (fun () ->
        raises_in "Jera.Piecewise.linear" (fun () ->
            Piecewise.linear (vec [| 0.; 1. |]) (vec [| 0.; 1.; 2. |])));
  ]

let cubic_tests =
  let ends =
    [
      ("natural", `Natural);
      ("not-a-knot", `Not_a_knot);
      ("clamped", `Clamped (scalar 0.5, scalar (-2.)));
    ]
  in
  let spline ends (x, y, _) = Piecewise.cubic ends (vec x) (vec y) in
  let ends_of x = vec [| x.(0); x.(Array.length x - 1) |] in
  let at_sum y =
    Nx.sum
      (Nx.sin (Piecewise.eval (Piecewise.cubic `Not_a_knot knots y) points))
  in
  [
    cases "is scipy's CubicSpline"
      ~id:(fun ((name, _), _) -> name)
      (List.combine ends
         [ G.cubic_natural; G.cubic_not_a_knot; G.cubic_clamped ])
      (fun ((_, e), values) ->
        equal (close ()) (vec values)
          (Piecewise.eval (Piecewise.cubic e knots samples) points));
    cases "not-a-knot through few knots is scipy's"
      ~id:(fun (n, _, _) -> Printf.sprintf "%d knots" n)
      [
        (2, G.not_a_knot2_points, G.not_a_knot2);
        (3, G.not_a_knot3_points, G.not_a_knot3);
        (4, G.not_a_knot4_points, G.not_a_knot4);
      ]
      (fun (n, pts, values) ->
        let first v = Nx.slice [ Nx.R (0, n) ] v in
        let p = Piecewise.cubic `Not_a_knot (first knots) (first samples) in
        equal (close ()) (vec values) (Piecewise.eval p (vec pts)));
    cases "samples with a value axis give scipy's splines" ~id:fst
      [
        ("natural", (`Natural, G.cubic_natural2));
        ("not-a-knot", (`Not_a_knot, G.cubic_not_a_knot2));
        ( "clamped",
          (`Clamped (vec [| 0.5; 0.5 |], vec [| -2.; -2. |]), G.cubic_clamped2)
        );
      ]
      (fun (_, (e, values)) ->
        equal (close ())
          (Nx.create f64 [| Array.length G.points; 2 |] values)
          (Piecewise.eval (Piecewise.cubic e knots samples2) points));
    prop "is twice continuously differentiable" data (fun ((x, _, _) as d) ->
        let n = Array.length x in
        cover "an interior knot" (n >= 3);
        List.iter
          (fun (name, e) ->
            let ends, starts =
              both_sides (derivatives 2 (spline e d)) (n - 1)
            in
            equal ~msg:name
              (Oracle.tensor ~abs:(spline_bound d 2) ())
              ends starts)
          ends);
    prop "`Natural has zero second derivative at both ends" data
      (fun ((x, _, _) as d) ->
        cover "two knots" (Array.length x = 2);
        equal
          (Oracle.tensor ~abs:(spline_bound d 2) ())
          (vec [| 0.; 0. |])
          (Piecewise.eval (derivatives 2 (spline `Natural d)) (ends_of x)));
    prop
      "`Clamped (s0, s1) has slopes s0 and s1 at the ends"
      (* Two knots once drew the line and ignored the slopes. *)
      ~examples:
        [ (([| 0.; 0.05 |], [| 1.; 0. |], [| 0.; 0.; 0.; 0.; 0. |]), (0., 0.)) ]
      Gen.(pair data (pair (float_range (-3.) 3.) (float_range (-3.) 3.)))
      (fun (((x, _, _) as d), (s0, s1)) ->
        cover "two knots" (Array.length x = 2);
        let p = spline (`Clamped (scalar s0, scalar s1)) d in
        equal
          (Oracle.tensor ~abs:(spline_bound d 1) ())
          (vec [| s0; s1 |])
          (Piecewise.eval (Piecewise.derivative p) (ends_of x)));
    prop
      "`Not_a_knot has a continuous third derivative at the second and \
       second-last knots"
      data (fun ((x, _, _) as d) ->
        let n = Array.length x in
        cover "three knots" (n = 3);
        cover "four knots or more" (n >= 4);
        (* Knots 1 and n − 2 are the interior breaks 0 and n − 3; with two knots
           there are none. *)
        let ends, starts =
          both_sides (derivatives 3 (spline `Not_a_knot d)) (n - 1)
        in
        let at v =
          if n < 3 then v
          else
            Nx.take
              (Nx.create Nx.int64 [| 2 |] [| 0L; Int64.of_int (n - 3) |])
              v
        in
        equal (Oracle.tensor ~abs:(spline_bound d 3) ()) (at ends) (at starts));
    test "`Not_a_knot through three knots is the parabola" (fun () ->
        let parabola x = Nx.sub_s (Nx.mul_s (Nx.square x) 2.) 1. in
        let x = vec [| -1.; 0.5; 2. |] in
        let p = Piecewise.cubic `Not_a_knot x (parabola x) in
        let t = Nx.linspace f64 (-1.) 2. 13 in
        equal (Oracle.tensor ~abs:1e-14 ()) (parabola t) (Piecewise.eval p t));
    test "grad in the samples is the central difference" (fun () ->
        (* eps 1e-6: truncation O(eps²), rounding ε/eps ≈ 2e-10. *)
        let v = Nx.cos (Nx.mul_s samples 3.) in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.reshape [||] (Oracle.central ~eps:1e-6 at_sum samples v))
          (scalar (Oracle.dot (Rune.grad' at_sum samples) v)));
    test "grad in the knots is the central difference" (fun () ->
        let f x =
          Nx.sum
            (Piecewise.eval
               (Piecewise.cubic `Natural x samples)
               (vec [| 0.2; 1.0 |]))
        in
        let v = Nx.sin knots in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (Nx.reshape [||] (Oracle.central ~eps:1e-6 f knots v))
          (scalar (Oracle.dot (Rune.grad' f knots) v)));
    test "compiled equals eager, the spline's solve included" (fun () ->
        Compiled.here ();
        equal (close ()) (at_sum samples) (Rune.jit' at_sum samples));
    test "vmap fits each lane's samples" (fun () ->
        let ys = Nx.stack [ samples; Nx.cos samples; Nx.neg samples ] in
        let one y = Piecewise.eval (Piecewise.cubic `Natural knots y) points in
        equal (close ())
          (Nx.stack (List.init 3 (fun i -> one (Nx.get [ i ] ys))))
          (Rune.vmap' one ys));
    test "float32 evaluates within float32's rounding" (fun () ->
        let c32 = Nx.cast Nx.float32 in
        let p = Piecewise.cubic `Natural (c32 knots) (c32 samples) in
        equal
          (Oracle.tensor ~rel:1e-5 ~abs:1e-6 ())
          (c32 (vec G.cubic_natural))
          (Piecewise.eval p (c32 points)));
    test "clamped slopes of another shape raise" (fun () ->
        raises_in "Jera.Piecewise.cubic" (fun () ->
            Piecewise.cubic
              (`Clamped (vec [| 1.; 1. |], scalar 0.))
              knots samples));
  ]

let steffen_tests =
  [
    prop "has no extremum between knots"
      Gen.(pair bool data)
      (fun (monotone, (x, y, _)) ->
        cover "monotone samples" monotone;
        (* Monotone samples rise by |y_i| at each knot. *)
        let y =
          if not monotone then y
          else
            let total = ref 0. in
            Array.map
              (fun v ->
                total := !total +. Float.abs v;
                !total)
              y
        in
        let n = Array.length x in
        let p = Piecewise.steffen (vec x) (vec y) in
        (* 21 points across each piece, against the same values clipped to the
           piece's samples, 1e-13 of their size apart. *)
        let index = Nx.arange Nx.int64 0 (n - 1) 1 in
        let u = Nx.linspace f64 (-1.) 1. 21 in
        let v =
          Piecewise.eval_at p
            (Nx.broadcast_to [| n - 1; 21 |] (Nx.reshape [| n - 1; 1 |] index))
            (Nx.broadcast_to [| n - 1; 21 |] (Nx.reshape [| 1; 21 |] u))
        in
        let tiny = 1e-13 *. (1. +. largest y) in
        let bound f d =
          Nx.reshape
            [| n - 1; 1 |]
            (vec (Array.init (n - 1) (fun i -> f y.(i) y.(i + 1) +. d)))
        in
        let clipped =
          Nx.minimum (bound Float.max tiny)
            (Nx.maximum (bound Float.min (-.tiny)) v)
        in
        equal (exact ()) clipped v);
    prop "its end slopes are the end intervals' secants" data
      (fun ((x, y, _) as d) ->
        cover "two knots" (Array.length x = 2);
        let n = Array.length x in
        let secant i = (y.(i + 1) -. y.(i)) /. (x.(i + 1) -. x.(i)) in
        let p = Piecewise.steffen (vec x) (vec y) in
        equal
          (Oracle.tensor ~abs:(spline_bound d 1) ())
          (vec [| secant 0; secant (n - 2) |])
          (Piecewise.eval (Piecewise.derivative p) (vec [| x.(0); x.(n - 1) |])));
  ]

let hermite_tests =
  [
    prop "is the cubic Hermite interpolant of its values and slopes"
      Gen.(pair data (array ~size:(constant 9) (float_range (-3.) 3.)))
      (fun ((x, y, pts), m) ->
        cover "a point on a knot" (Array.exists (fun t -> Array.mem t x) pts);
        let m = Array.sub m 0 (Array.length x) in
        let hermite t =
          let i, s = locate x t in
          let h = x.(i + 1) -. x.(i) in
          let s2 = s *. s in
          let s3 = s2 *. s in
          (((2. *. s3) -. (3. *. s2) +. 1.) *. y.(i))
          +. ((s3 -. (2. *. s2) +. s) *. h *. m.(i))
          +. (((3. *. s2) -. (2. *. s3)) *. y.(i + 1))
          +. ((s3 -. s2) *. h *. m.(i + 1))
        in
        let p = Piecewise.hermite (vec x) ~values:(vec y) ~slopes:(vec m) in
        (* A cubic of values up to max |y| + max |m|: 64 ε of that. *)
        let bound = 64. *. epsilon_float *. (largest y +. largest m) in
        equal
          (Oracle.tensor ~abs:bound ())
          (vec (Array.map hermite pts))
          (Piecewise.eval p (vec pts)));
    test "slopes of another shape raise" (fun () ->
        raises_in "Jera.Piecewise.hermite" (fun () ->
            Piecewise.hermite
              (vec [| 0.; 1.; 2. |])
              ~values:(vec [| 0.; 1.; 4. |])
              ~slopes:(vec [| 0.; 1. |])));
  ]

(* Fits *)

let chebyshev ?(degree = 3) ?(pieces = 2) f a b =
  Piecewise.chebyshev Nx.Ptree.tensor ~degree ~pieces f a b

let chebyshev_tests =
  [
    test "evaluates f at each piece's second-kind Chebyshev points" (fun () ->
        (* Pieces [0, 1], [1, 2] and [2, 3]: mid − half cos(πj/4), j = 0..4,
           ends included; the library takes the cosine as a sine, an ulp
           apart. *)
        let seen = ref [] in
        let f x =
          seen := Oracle.floats x :: !seen;
          x
        in
        ignore (chebyshev ~degree:4 ~pieces:3 f (scalar 0.) (scalar 3.));
        let point k =
          let p = float_of_int (k / 5) and j = float_of_int (k mod 5) in
          p +. 0.5 -. (0.5 *. Float.cos (Float.pi *. j /. 4.))
        in
        equal (list (array (float 1e-15))) [ Array.init 15 point ] !seen);
    test "a fit of exp converges to rounding" (fun () ->
        (* Degree 16 on pieces of width 1: truncation below 1e-19, so the error
           is Clenshaw's rounding, about 16 ε. *)
        let p = chebyshev ~degree:16 Nx.exp (scalar 0.) (scalar 2.) in
        let x = Nx.linspace f64 0. 2. 37 in
        equal (Oracle.tensor ~rel:4e-15 ()) (Nx.exp x) (Piecewise.eval p x));
    prop "reproduces a polynomial of its degree"
      Gen.(
        let* degree = frequency [ (1, constant 0); (3, int_range 1 12) ] in
        let+ pieces = int_range 1 4
        and+ c = array ~size:(constant (degree + 1)) (float_range (-1.) 1.)
        and+ a = float_range (-2.) 0.
        and+ w = float_range 0.5 3. in
        (degree, pieces, c, a, a +. w))
      (fun (degree, pieces, c, a, b) ->
        cover "degree 0" (degree = 0);
        let horner x = Array.fold_right (fun ci acc -> ci +. (x *. acc)) c 0. in
        let f x = Nx.map_item horner x in
        let p = chebyshev ~degree ~pieces f (scalar a) (scalar b) in
        let x = Nx.linspace f64 a b 11 in
        (* The fit and Clenshaw each round in about degree² terms of the size Σ
           |c_k| max |x|^k. *)
        let m = Float.max (Float.abs a) (Float.abs b) in
        let size, _ =
          Array.fold_left
            (fun (s, power) ck -> (s +. (Float.abs ck *. power), power *. m))
            (0., 1.) c
        in
        let terms = float_of_int ((degree + 1) * (degree + 1)) in
        let bound = 16. *. terms *. epsilon_float *. (size +. 1.) in
        equal (Oracle.tensor ~abs:bound ()) (f x) (Piecewise.eval p x));
    test "a fit of a structure fits each leaf" (fun () ->
        let s = Nx.Ptree.(pair tensor tensor) in
        let p =
          Piecewise.chebyshev s ~degree:20 ~pieces:1
            (fun x -> (Nx.sin x, Nx.cos x))
            (scalar 0.) (scalar 1.)
        in
        let x = vec [| 0.; 0.25; 1. |] in
        equal
          (Oracle.structure ~rel:1e-14 ~abs:1e-15 s)
          (Nx.sin x, Nx.cos x)
          (Piecewise.eval p x));
    test "degree 0 is the midpoint's value" (fun () ->
        let p = chebyshev ~degree:0 Nx.exp (scalar 0.) (scalar 2.) in
        equal (close ())
          (vec [| Float.exp 0.5; Float.exp 1.5 |])
          (Piecewise.eval p (vec [| 0.2; 1.9 |])));
    test "a negative degree raises" (fun () ->
        raises_in "Jera.Piecewise.chebyshev" (fun () ->
            chebyshev ~degree:(-1) Nx.exp (scalar 0.) (scalar 1.)));
    test "no piece raises" (fun () ->
        raises_in "Jera.Piecewise.chebyshev" (fun () ->
            chebyshev ~pieces:0 Nx.exp (scalar 0.) (scalar 1.)));
    test "ends that are not scalars raise" (fun () ->
        raises_in "Jera.Piecewise.chebyshev" (fun () ->
            chebyshev Nx.exp (vec [| 0. |]) (scalar 1.));
        raises_in "Jera.Piecewise.chebyshev" (fun () ->
            chebyshev Nx.exp (scalar 0.) (vec [| 1. |])));
    test "a leaf without the points' shape raises" (fun () ->
        raises_in "Jera.Piecewise.chebyshev" (fun () ->
            chebyshev (fun x -> Nx.sum x) (scalar 0.) (scalar 1.)));
  ]

let adapt ?(budget = 64) ?(degree = 8) f a b =
  Piecewise.adapt Nx.Ptree.tensor ~degree
    ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
    ~budget f (scalar a) (scalar b)

let between = Nx.linspace f64 0.0123 1.9871 97
let kink x = Nx.abs (Nx.sub_s x 0.3)

let adapt_tests =
  let at theta =
    Nx.sum
      (Piecewise.eval
         (Solution.get (adapt (fun z -> Nx.sin (Nx.mul z theta)) 0. 2.))
         (vec [| 0.4; 1.3 |]))
  in
  let theta = scalar 1.7 in
  [
    test "a smooth function is met between the nodes" (fun () ->
        (* The tail measures the error up to the series' last terms: 10 tol. *)
        let p = Solution.get (adapt Nx.exp 0. 2.) in
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (Nx.exp between) (Piecewise.eval p between));
    test "a kink is met" (fun () ->
        let p = Solution.get (adapt ~budget:200 kink 0. 2.) in
        equal
          (Oracle.tensor ~abs:1e-10 ())
          (kink between) (Piecewise.eval p between));
    test "a structure fits each leaf" (fun () ->
        let s2 = Nx.Ptree.(pair tensor tensor) in
        let p =
          Solution.get
            (Piecewise.adapt s2 ~degree:10
               ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
               ~budget:32
               (fun x -> (Nx.sin x, Nx.cos x))
               (scalar 0.) (scalar 2.))
        in
        equal
          (Oracle.structure ~rel:1e-9 ~abs:1e-11 s2)
          (Nx.sin between, Nx.cos between)
          (Piecewise.eval p between));
    test "the answer holds budget pieces, the last at b" (fun () ->
        let b =
          Oracle.floats
            (Piecewise.breaks (Solution.get (adapt ~budget:16 Nx.exp 0. 2.)))
        in
        equal ~msg:"breaks" int 17 (Array.length b);
        equal ~msg:"the last break" float_exact 2. b.(16));
    test "its error is a degree-0 series on its breaks" (fun () ->
        let s = adapt ~budget:16 Nx.exp 0. 2. in
        let e = Solution.error s in
        equal (array int) [| 16; 1 |] (Nx.shape (Piecewise.coefficients e));
        equal (exact ())
          (Piecewise.breaks (Solution.get s))
          (Piecewise.breaks e));
    test "budget pieces end it Budget_spent" (fun () ->
        equal Oracle.status_t Budget_spent
          (Oracle.status (adapt ~budget:3 kink 0. 1.)));
    test "evaluations count the search's pieces and the answer's budget"
      (fun () ->
        (* Budget 3 at degree 8: 2·3 − 1 pieces searched, then 3 again, 9 points
           each. *)
        let s = adapt ~budget:3 kink 0. 1. in
        equal int32
          (Int32.of_int (9 * (5 + 3)))
          (Nx.item [] (Solution.evaluations s)));
    test "a non-finite value ends it Not_finite" (fun () ->
        (* The midpoint 0.5 of [0, 1] is a node at degree 8. *)
        let s = adapt (fun x -> Nx.recip (Nx.sub_s x 0.5)) 0. 1. in
        equal Oracle.status_t Not_finite (Oracle.status s));
    test "a piece at level 62 ends it Stalled" (fun () ->
        (* √x looks the same on [0, 2^−L] at every L, so its tail there never
           meets a relative tol, and floats never run out near 0: only the level
           stops it, at 63 pieces, which evaluate 9 (2 · 63 − 1 + 400)
           points. *)
        let s =
          Piecewise.adapt Nx.Ptree.tensor ~degree:8 ~tol:(Tol.rel 1e-10)
            ~budget:400 Nx.sqrt (scalar 0.) (scalar 1.)
        in
        equal Oracle.status_t Stalled (Oracle.status s);
        equal int32 4725l (Nx.item [] (Solution.evaluations s)));
    test "its empty pieces extend from its last real piece" (fun () ->
        (* Beyond b the last piece of positive width continues, e^x to 1e-6 a
           little past the end. *)
        let p = Solution.get (adapt ~budget:16 Nx.exp 0. 2.) in
        let x = vec [| 2.05; 2.1 |] in
        equal
          (Oracle.tensor ~rel:1e-6 ())
          (Nx.exp x)
          (Piecewise.eval (Piecewise.extend `Polynomial p) x));
    test "grad in a captured parameter is the function's" (fun () ->
        let x = vec [| 0.4; 1.3 |] in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (scalar (Oracle.dot x (Nx.cos (Nx.mul_s x 1.7))))
          (Rune.grad' at theta));
    test "a fit that did not converge has a zero derivative" (fun () ->
        (* The pole of p / (x − 1/4) is the middle node of the piece [0, ½]. *)
        let fit p = adapt (fun x -> Nx.div p (Nx.sub_s x 0.25)) 0. 1. in
        let value p =
          Nx.sum (Piecewise.eval (Solution.best (fit p)) (vec [| 0.7 |]))
        in
        let p = scalar 1. in
        equal Oracle.status_t Not_finite (Oracle.status (fit p));
        equal (exact ()) (scalar 0.) (Rune.grad' value p));
    test "compiled equals eager within the fit's rounding" (fun () ->
        Compiled.here ();
        (* The fit is a matrix product, whose sums a compiled call may associate
           otherwise. *)
        let at theta =
          Piecewise.eval
            (Solution.get (adapt (fun z -> Nx.sin (Nx.mul z theta)) 0. 2.))
            between
        in
        equal
          (Oracle.tensor ~rel:1e-14 ~abs:1e-15 ())
          (at theta) (Rune.jit' at theta));
    test "compiled grad equals eager grad" (fun () ->
        Compiled.here ();
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Rune.grad' at theta)
          (Rune.jit' (Rune.grad' at) theta));
    test "vmap gives each lane its own partition and status" (fun () ->
        let fit theta =
          Solution.ok (adapt ~budget:4 (fun z -> Nx.abs (Nx.sub z theta)) 0. 1.)
        in
        (* A kink at an end needs no refinement; one inside does. *)
        equal (exact ())
          (Nx.create Nx.bool [| 2 |] [| true; false |])
          (Rune.vmap
             Nx.Ptree.(tensor @-> returns tensor)
             fit
             (vec [| 0.; 0.37 |])));
    test "degree 1 raises" (fun () ->
        raises_in "Jera.Piecewise.adapt" (fun () ->
            adapt ~degree:1 Nx.exp 0. 1.));
    test "budget 0 raises" (fun () ->
        raises_in "Jera.Piecewise.adapt" (fun () ->
            adapt ~budget:0 Nx.exp 0. 1.));
    test "ends that are not scalars raise" (fun () ->
        let fit a b =
          Piecewise.adapt Nx.Ptree.tensor ~degree:3 ~tol:(Tol.rel 1e-6)
            ~budget:8 Nx.exp a b
        in
        raises_in "Jera.Piecewise.adapt" (fun () ->
            fit (vec [| 0. |]) (scalar 1.));
        raises_in "Jera.Piecewise.adapt" (fun () ->
            fit (scalar 0.) (vec [| 1. |])));
    test "a leaf without the points' shape raises" (fun () ->
        raises_in "Jera.Piecewise.adapt" (fun () ->
            adapt (fun x -> Nx.sum x) 0. 1.));
  ]

(* Evaluation *)

let eval_tests =
  let step =
    Piecewise.v Nx.Ptree.tensor
      ~breaks:(vec [| 0.; 1.; 2. |])
      (Nx.create f64 [| 2; 1 |] [| 5.; 7. |])
  in
  [
    test "the last break is in the last piece" (fun () ->
        equal (exact ()) (vec [| 3. |]) (Piecewise.eval broken (vec [| 2. |])));
    test "an interior break is in the piece that ends there" (fun () ->
        equal (exact ())
          (vec [| 5.; 5.; 7. |])
          (Piecewise.eval step (vec [| 0.; 1.; 2. |])));
    test "NaN is NaN, even for a constant" (fun () ->
        equal (exact ())
          (vec [| nan; 5. |])
          (Piecewise.eval step (vec [| nan; 0.5 |])));
    test "a point outside raises" (fun () ->
        raises_in "Jera.Piecewise.eval" (fun () ->
            Piecewise.eval broken (vec [| 1.; 2.5 |])));
    cases "an infinite point raises under every extension" ~id:fst
      [
        ("bounded", broken);
        ("hold", Piecewise.extend `Hold broken);
        ("polynomial", Piecewise.extend `Polynomial broken);
      ]
      (fun (_, p) ->
        raises_in "Jera.Piecewise.eval" (fun () ->
            Piecewise.eval p (vec [| 0.5; infinity |]));
        raises_in "Jera.Piecewise.eval" (fun () ->
            Piecewise.eval p (vec [| neg_infinity |])));
    test "zero points give zero values" (fun () ->
        equal (exact ()) (vec [||]) (Piecewise.eval broken (vec [||])));
    test "points keep their shape and values add theirs" (fun () ->
        let q =
          Piecewise.linear
            (vec [| 0.; 1. |])
            (Nx.create f64 [| 2; 3 |] [| 0.; 1.; 2.; 1.; 1.; 1. |])
        in
        equal (exact ())
          (Nx.broadcast_to [| 2; 2; 3 |] (vec [| 0.; 1.; 2. |]))
          (Piecewise.eval q (Nx.zeros f64 [| 2; 2 |])));
  ]

let eval_at_tests =
  let index i = Nx.create Nx.int64 [| Array.length i |] i in
  [
    test "is eval at its piece's coordinate" (fun () ->
        let s = Piecewise.cubic `Natural knots samples in
        equal (close ())
          (Piecewise.eval s (vec [| -0.7; 2.45 |]))
          (Piecewise.eval_at s (index [| 0L; 5L |]) (vec [| 0.; 0. |])));
    test "extrapolates the piece's series outside [−1, 1]" (fun () ->
        (* Piece 0 is the line 2x on [0, 1]; u = 2 is x = 1.5. *)
        equal (exact ()) (vec [| 3. |])
          (Piecewise.eval_at broken (index [| 0L |]) (vec [| 2. |])));
    test "an index that is not a piece raises" (fun () ->
        raises_in "Jera.Piecewise.eval_at" (fun () ->
            Piecewise.eval_at broken (index [| 2L |]) (vec [| 0. |])));
    test "indices and coordinates of two shapes raise" (fun () ->
        raises_in "Jera.Piecewise.eval_at" (fun () ->
            Piecewise.eval_at broken
              (Nx.zeros Nx.int64 [| 2 |])
              (Nx.zeros f64 [| 3 |])));
  ]

(* Calculus *)

(* The degree of [op] applied to a one-piece series of degree [d]. *)
let degree_after op d =
  let p =
    Piecewise.v Nx.Ptree.tensor
      ~breaks:(vec [| 0.; 1. |])
      (Nx.ones f64 [| 1; d + 1 |])
  in
  Nx.dim 1 (Piecewise.coefficients (op p)) - 1

let derivative_tests =
  [
    prop "eval of the derivative is the derivative of eval" series
      (fun ((_, _, pts) as s) ->
        cover_series s;
        let p = of_series s in
        let x = vec pts in
        let along = snd (Rune.jvp' (Piecewise.eval p) x (Nx.ones_like x)) in
        equal
          (Oracle.tensor ~abs:(bound s ~order:1) ())
          along
          (Piecewise.eval (Piecewise.derivative p) x));
    cases "has one degree less, and 0 from 0" ~id:(Printf.sprintf "degree %d")
      [ 0; 1; 3 ] (fun d ->
        equal int (max 0 (d - 1)) (degree_after Piecewise.derivative d));
    test "of a held series holds its end slopes" (fun () ->
        let p = Piecewise.derivative (Piecewise.extend `Hold broken) in
        equal (exact ())
          (vec [| 2.; 1. |])
          (Piecewise.eval p (vec [| -3.; 5. |])));
  ]

let integral_tests =
  [
    prop "vanishes at the first break" series (fun ((breaks, _, _) as s) ->
        cover_series s;
        let p = Piecewise.integral (of_series s) in
        equal
          (Oracle.tensor ~abs:(bound s ~order:0) ())
          (vec [| 0. |])
          (Piecewise.eval p (vec [| breaks.(0) |])));
    prop "its derivative is the series" series (fun ((_, _, pts) as s) ->
        cover_series s;
        let p = of_series s in
        let x = vec pts in
        let i = Piecewise.integral p in
        let along = snd (Rune.jvp' (Piecewise.eval i) x (Nx.ones_like x)) in
        equal
          (Oracle.tensor ~abs:(bound s ~order:0) ())
          (Piecewise.eval p x) along);
    prop "is continuous at the breaks" series (fun ((breaks, _, _) as s) ->
        cover_series s;
        let i = Piecewise.integral (of_series s) in
        let ends, starts = both_sides i (Array.length breaks - 1) in
        equal (Oracle.tensor ~abs:(bound s ~order:0) ()) ends starts);
    cases "has one degree more" ~id:(Printf.sprintf "degree %d") [ 0; 1; 3 ]
      (fun d -> equal int (d + 1) (degree_after Piecewise.integral d));
    test "of a held series holds its end values" (fun () ->
        (* The broken line's integral over [0, 2] is 1 + 2.5. *)
        let p = Piecewise.integral (Piecewise.extend `Hold broken) in
        equal (exact ())
          (vec [| 0.; 3.5 |])
          (Piecewise.eval p (vec [| -3.; 5. |])));
  ]

let extend_tests =
  [
    test "hold takes the nearest end" (fun () ->
        equal (exact ())
          (vec [| 0.; 3. |])
          (Piecewise.eval (Piecewise.extend `Hold broken) (vec [| -4.; 9. |])));
    test "polynomial continues the end pieces" (fun () ->
        equal (exact ())
          (vec [| -2.; 4. |])
          (Piecewise.eval
             (Piecewise.extend `Polynomial broken)
             (vec [| -1.; 3. |])));
    test "replaces the extension" (fun () ->
        let p = Piecewise.extend `Hold (Piecewise.extend `Polynomial broken) in
        equal (exact ()) (vec [| 3. |]) (Piecewise.eval p (vec [| 3. |])));
  ]

(* Access *)

let ptree_tests =
  [
    cases "visits the extension, the breaks and the coefficients" ~id:fst
      [
        ("bounded", broken);
        ("hold", Piecewise.extend `Hold broken);
        ("polynomial", Piecewise.extend `Polynomial broken);
      ]
      (fun (name, p) ->
        let visits = Nx.Ptree.visits (Piecewise.ptree Nx.Ptree.tensor f64) p in
        equal (list string)
          [
            Printf.sprintf "extension: case %S" name;
            "breaks: a leaf";
            "coefficients: a leaf";
          ]
          (List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) visits));
    test "a series is a compiled function's argument" (fun () ->
        Compiled.here ();
        let p = Piecewise.cubic `Natural knots samples in
        let s = Piecewise.ptree Nx.Ptree.tensor f64 in
        let f =
          Rune.jit Nx.Ptree.(s @-> tensor @-> returns tensor) Piecewise.eval
        in
        equal (close ()) (Piecewise.eval p points) (f p points));
  ]

let () =
  exit
    (run "Jera.Piecewise"
       [
         group "v" v_tests;
         group "linear" linear_tests;
         group "cubic" cubic_tests;
         group "steffen" steffen_tests;
         group "hermite" hermite_tests;
         group "chebyshev" chebyshev_tests;
         group "adapt" adapt_tests;
         group "eval" eval_tests;
         group "eval_at" eval_at_tests;
         group "derivative" derivative_tests;
         group "integral" integral_tests;
         group "extend" extend_tests;
         group "ptree" ptree_tests;
       ])
