(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_kit
module Color = Hugin_gg.Color

let invalid f = raises_match (Exn.invalid_arg ?sub:None) f

(* Floats ordered as IEEE 754 orders them, [-0.] equal to [0.]. A monotone law
   compares by it: [Scale.normalize] states no sign for a zero. *)
let ieee = float Float.min_float

(* [near tol] compares within [tol], and exactly where [tol] is not finite, as
   it is for an infinite result. [float] takes no zero tolerance: a bound of
   [0.] becomes the least subnormal. *)
let near tol =
  if Float.is_finite tol then float (Float.max tol (Float.succ 0.))
  else float_exact

let fscale : float Scale.t testable =
  Testable.make ~pp:Scale.pp ~equal:Scale.equal

let tscale : Time.t Scale.t testable =
  Testable.make ~pp:Scale.pp ~equal:Scale.equal

let bscale : string Scale.t testable =
  Testable.make ~pp:Scale.pp ~equal:Scale.equal

let instant =
  Testable.with_compare Time.compare
    (Testable.make ~pp:Time.pp ~equal:Time.equal)

let notation =
  Testable.make ~pp:Number.pp_notation ~equal:(fun (n : Number.notation) n' ->
      n = n')

let dash = Testable.make ~pp:Dash.pp ~equal:Dash.equal
let symbol = Testable.make ~pp:Symbol.pp ~equal:Symbol.equal

let property =
  Testable.make ~pp:Scale.pp_property ~equal:(fun (p : Scale.property) q ->
      p = q)

let merged = result fscale property
let ends_w = pair float_exact float_exact

let ends s =
  let (Scale.Floats (a, b)) = Scale.domain s in
  (a, b)

let instants s =
  let (Scale.Instants (a, b)) = Scale.domain s in
  (a, b)

let names s =
  match Scale.domain s with
  | Scale.Categories (Labels l) -> l
  | Scale.Categories (Indices ix) ->
      Array.map (fun (i, _) -> string_of_int i) ix

let fit_floats lo hi s = Scale.fit (Some (Scale.Floats (lo, hi))) s
let fit_instants a b s = instants (Scale.fit (Some (Scale.Instants (a, b))) s)

let utc ?(ms = 0) d t =
  Time.add (Time.milliseconds 1) ms (Time.of_date_time (d, t))

let asinh =
  Scale.custom ~transform:"asinh" ~forward:Float.asinh ~inverse:Float.sinh

let ln = Scale.custom ~transform:"ln" ~forward:Float.log ~inverse:Float.exp

let logit () =
  Scale.custom ~transform:"logit"
    ~forward:(fun p -> Float.log (p /. (1. -. p)))
    ~inverse:(fun x -> 1. /. (1. +. Float.exp (-.x)))
    ()

(* Quantitative scales *)

(* A quantitative scale over a domain, and its transform [T] as the .mli states
   it. *)
type quantitative = {
  spec : float Scale.t;
  s : float Scale.t;
  t : float -> float;
}

let quantitative spec t (a, b) =
  { spec; s = Scale.with_domain (Scale.Floats (a, b)) spec; t = t (a, b) }

(* Symlog's [T] evaluated without overflow: past it, [log1p y] is [log y] to the
   float. *)
let symlog_t c x =
  let y = Float.abs x /. c in
  let v =
    if Float.is_finite y then Float.log1p y
    else Float.log (Float.abs x) -. Float.log c
  in
  Float.copy_sign v x

let pow_t e (a, b) =
  let m = Float.max (Float.abs a) (Float.abs b) in
  let m = if m = 0. then 1. else m in
  fun x -> Float.copy_sign (Float.pow (Float.abs (x /. m)) e) x

let on_linear = quantitative (Scale.linear ()) (fun _ -> Fun.id)

let on_symlog c =
  quantitative (Scale.symlog ~constant:c ()) (fun _ -> symlog_t c)

let on_pow e = quantitative (Scale.pow ~exponent:e ()) (pow_t e)
let on_asinh = quantitative (asinh ()) (fun _ -> Float.asinh)

let on_log b =
  quantitative (Scale.log ~base:b ()) (fun _ x -> Float.log x /. Float.log b)

let order (a, b) = if a <= b then (a, b) else (b, a)

(* Domains with distinct ends anywhere among the finite floats, subnormals and
   the extremes included. *)
let gen_domain =
  Gen.map order
    (Gen.such_that (fun (a, b) -> a <> b) (Gen.pair Gen.float Gen.float))

let gen_positive_float =
  Gen.such_that (fun x -> x > 0.) (Gen.map Float.abs Gen.float)

let gen_positive =
  Gen.map order
    (Gen.such_that
       (fun (a, b) -> a <> b)
       (Gen.pair gen_positive_float gen_positive_float))

(* Domains of moderate magnitude whose length is not tiny next to their ends. *)
let gen_moderate =
  Gen.map
    (fun (c, w) -> (c, c +. w))
    (Gen.pair (Gen.float_range (-1e6) 1e6) (Gen.float_range 1e-3 1e6))

let with_pp_q gen = Gen.with_pp (fun ppf q -> Scale.pp ppf q.s) gen

let gen_scale =
  with_pp_q
    (Gen.one_of
       [
         Gen.map on_linear gen_domain;
         Gen.map (on_symlog 1.) gen_domain;
         Gen.map (on_symlog 1e-3) gen_domain;
         Gen.map (on_pow 2.) gen_domain;
         Gen.map (on_pow 0.5) gen_domain;
         Gen.map on_asinh gen_domain;
         Gen.map (on_log 10.) gen_positive;
         Gen.map (on_log 2.) gen_positive;
         Gen.map (on_log (Float.exp 1.)) gen_positive;
       ])

let is_constant { s; t; _ } =
  let a, b = ends s in
  Float.equal (t a) (t b)

(* A scale and a value that is not missing for it, inside its domain or
   anywhere. *)
let gen_scale_value =
  let anywhere q =
    match Scale.transform q.s with
    | Log _ -> gen_positive_float
    | Linear | Symlog _ | Pow _ | Custom _ -> Gen.float
  in
  let inside q =
    let a, b = ends q.s in
    Gen.map (fun u -> (a *. (1. -. u)) +. (b *. u)) (Gen.float_range 0. 1.)
  in
  Gen.with_pp
    (fun ppf (q, x) -> Format.fprintf ppf "%a at %h" Scale.pp q.s x)
    Gen.(
      let* q = gen_scale in
      let+ x = one_of [ inside q; anywhere q ] in
      (q, x))

(* [formula q x] is [(T x - T a) / (T b - T a)], each difference halved first
   where it overflows. *)
let formula { s; t; _ } x =
  let a, b = ends s in
  let ta = t a and tb = t b and tx = t x in
  if Float.is_finite (tx -. ta) && Float.is_finite (tb -. ta) then
    (tx -. ta) /. (tb -. ta)
  else ((tx /. 2.) -. (ta /. 2.)) /. ((tb /. 2.) -. (ta /. 2.))

(* [formula_tolerance q n x] bounds the distance between two evaluations of [N x
   = n] on [q]. Each evaluates [T] at [a], [b] and [x] within 4 roundings (a
   libm call, the division by [m] or by the logarithm of the base, and that
   logarithm), so the two are within [8 ε] of each [T] value, and the
   subtraction and the division add 2 roundings a side. An error in [T x] or [T
   a] moves [N] by itself over [T b - T a], one in [T a] or [T b] by [|N|] times
   that. *)
let formula_tolerance { s; t; _ } n x =
  let a, b = ends s in
  let ta = t a and tb = t b and tx = t x in
  let w = tb -. ta in
  let over v =
    if Float.is_finite w then Float.abs v /. w
    else Float.abs (v /. 2.) /. ((tb /. 2.) -. (ta /. 2.))
  in
  Float.epsilon
  *. ((8. *. (over tx +. over ta +. (Float.abs n *. (over ta +. over tb))))
     +. (4. *. Float.abs n))

(* Band scales *)

type band = { n : int; padding : float; reverse : bool; b : string Scale.t }

let band_of (n, padding, reverse, indexed) =
  let domain =
    if indexed then Scale.Indices (Array.init n (fun i -> ((3 * i) - 4, "t")))
    else Scale.Labels (Array.init n (fun i -> "c" ^ string_of_int i))
  in
  { n; padding; reverse; b = Scale.band ~domain ~padding ~reverse () }

(* Band scales of up to six labelled or indexed categories, padded and reversed
   or not. *)
let gen_band =
  Gen.map band_of
    (Gen.quad (Gen.int_range 0 6) (Gen.float_range 0. 1.) Gen.bool Gen.bool)
  |> Gen.with_pp (fun ppf x -> Scale.pp ppf x.b)

(* The band formulas round at most five times a side, each within half an ulp of
   a value at most 1. *)
let band_tolerance = 8. *. Float.epsilon

(* Normalisation *)

let ends_law =
  prop "the ends of a domain that is not constant normalise to 0 and 1"
    ~examples:[ on_linear (-.Float.max_float, Float.max_float) ]
    gen_scale
    (fun q ->
      assume (not (is_constant q));
      let a, b = ends q.s in
      equal float_exact 0. (Scale.normalize q.s a);
      equal float_exact 1. (Scale.normalize q.s b))

let formula_law =
  prop "a value normalises to (T x - T a) / (T b - T a)"
    ~examples:
      [
        (on_linear (0., 10.), -5.);
        (on_linear (0., 10.), 20.);
        (on_linear (-.Float.max_float, Float.max_float), 0.);
        (on_linear (1e308, Float.max_float), -.Float.max_float);
        (on_symlog 1e-300 (-1e300, 1e300), 1e299);
        (on_symlog 1. (-1., 1.), 0.5);
        (on_pow 2. (0., 4.), 2.);
        (on_pow 2. (0., 1e200), 5e199);
      ]
    gen_scale_value
    (fun (q, x) ->
      let a, b = ends q.s in
      cover "a value beyond the domain" (x < a || x > b);
      cover "a value inside the domain" (a <= x && x <= b);
      assume (not (is_constant q));
      let n = formula q x in
      equal (near (formula_tolerance q n x)) n (Scale.normalize q.s x))

let monotone_law =
  prop "normalisation is monotone over the domain and beyond it"
    (Gen.triple gen_scale Gen.float Gen.float) (fun (q, x, y) ->
      let x, y = order (x, y) in
      let u = Scale.normalize q.s x and v = Scale.normalize q.s y in
      let both = not (Float.is_nan u || Float.is_nan v) in
      cover "both values normalise" both;
      if both then at_most ieee ~than:v u)

let reverse_law =
  prop "a reversed scale normalises x to 1 - N x"
    (Gen.pair gen_scale Gen.any_float) (fun (q, x) ->
      let r = Scale.imply (Scale.linear ~reverse:true ()) q.s in
      equal float_exact (1. -. Scale.normalize q.s x) (Scale.normalize r x))

let clamp_law =
  prop "a clamped scale clamps into [0;1] after reversing"
    ~examples:
      [ (on_linear (0., 1.), false, -1.); (on_linear (0., 1.), false, 2.) ]
    (Gen.triple gen_scale Gen.bool Gen.any_float)
    (fun (q, reverse, x) ->
      let u = Scale.normalize (Scale.imply (Scale.linear ~reverse ()) q.s) x in
      cover "below 0" (u < 0.);
      cover "beyond 1" (u > 1.);
      let c = Scale.imply (Scale.linear ~clamp:true ~reverse ()) q.s in
      (* Adding [0.] makes [-0.] [0.]: the .mli states no sign for a zero. *)
      equal float_exact
        (Float.min 1. (Float.max 0. u) +. 0.)
        (Scale.normalize c x +. 0.))

let band_law =
  prop "a band normalises category i to (i + (1 + p) / 2) / (n + p)"
    ~examples:
      [
        band_of (3, 0., false, false);
        band_of (3, 1., false, false);
        band_of (3, 0., true, false);
      ]
    (Gen.such_that (fun x -> x.n > 0) gen_band)
    (fun { n; padding = p; reverse; b } ->
      let n = Float.of_int n in
      Array.iteri
        (fun i c ->
          let centre = (Float.of_int i +. ((1. +. p) /. 2.)) /. (n +. p) in
          let expected = if reverse then 1. -. centre else centre in
          equal (float band_tolerance) expected (Scale.normalize b c))
        (names b))

let missing_law =
  prop "a log scale normalises to nan exactly the values that are not positive"
    ~examples:
      [
        ((1., 100.), 0.);
        ((1., 100.), -0.);
        ((1., 100.), -10.);
        ((1., 100.), Float.min_float);
      ]
    (Gen.pair gen_positive Gen.any_float)
    (fun (d, x) ->
      let positive = Float.is_finite x && x > 0. in
      cover "a positive value" positive;
      cover "a value that is not positive" (not positive);
      let u = Scale.normalize (Scale.log ~domain:d ()) x in
      if positive then
        satisfies ~claim:"a value that is not nan" float_exact
          (fun u -> not (Float.is_nan u))
          u
      else equal float_exact Float.nan u)

let normalisation =
  group "normalisation"
    [
      ends_law;
      formula_law;
      monotone_law;
      test "a constant domain normalises a value that is not missing to 0.5"
        (fun () ->
          let s = Scale.linear ~domain:(3., 3.) () in
          equal float_exact 0.5 (Scale.normalize s 3.);
          equal float_exact 0.5 (Scale.normalize s 1e9);
          let s = Scale.log ~domain:(1e308, 1.0000000000000002e308) () in
          equal float_exact 0.5 (Scale.normalize s 1e308));
      reverse_law;
      test "a reversed temporal scale normalises x to 1 - N x" (fun () ->
          let s =
            Scale.time ~reverse:true ~domain:(Time.v Ns 0L, Time.v Ns 3L) ()
          in
          equal float_exact
            (1. -. (1. /. 3.))
            (Scale.normalize s (Time.v Ns 1L)));
      clamp_law;
      test "a temporal scale divides exact nanosecond differences" (fun () ->
          let s = Scale.time ~domain:(Time.v Ns 0L, Time.v Ns 3L) () in
          equal float_exact (1. /. 3.) (Scale.normalize s (Time.v Ns 1L));
          let s =
            Scale.time
              ~domain:(Time.v S Int64.min_int, Time.v S Int64.max_int)
              ()
          in
          equal float_exact 0. (Scale.normalize s (Time.v S Int64.min_int));
          equal float_exact 1. (Scale.normalize s (Time.v S Int64.max_int));
          equal float_exact 0.5 (Scale.normalize s Time.epoch));
      test "a temporal scale extrapolates before its domain" (fun () ->
          let s = Scale.time ~domain:(Time.v Ns 7L, Time.v Ns 10L) () in
          equal float_exact (-1.) (Scale.normalize s (Time.v Ns 4L));
          let s =
            Scale.time ~domain:(Time.epoch, Time.v Ns 0x1_0000_0000L) ()
          in
          equal float_exact (-1.)
            (Scale.normalize s (Time.v Ns (-0x1_0000_0000L))));
      test "a temporal difference beyond 2^63 ns rounds once" (fun () ->
          (* [2^64 + 2049] ns over [2^65] ns: dropping the low bits would make
             it a tie, rounded to even. *)
          let at s ns = Time.add (Time.nanoseconds 1) ns (Time.v S s) in
          let s =
            Scale.time ~domain:(Time.epoch, at 36893488147L 419103232) ()
          in
          equal float_exact
            (0.5 +. Float.ldexp 1. (-53))
            (Scale.normalize s (at 18446744073L 709553665));
          equal float_exact 0.5 (Scale.normalize s (at 18446744073L 709553664));
          equal float_exact
            (0.5 +. Float.ldexp 1. (-53))
            (Scale.normalize s (at 18446744073L 709553666));
          equal float_exact
            (-0.5 -. Float.ldexp 1. (-53))
            (Scale.normalize s (at (-18446744074L) (1_000_000_000 - 709553665))));
      test "a temporal scale normalises a constant domain to 0.5" (fun () ->
          let t = Time.v S 5L in
          let s = Scale.time ~domain:(t, t) () in
          equal float_exact 0.5 (Scale.normalize s (Time.v S 100L)));
      band_law;
      test "nan and infinities are missing on every scale" (fun () ->
          List.iter
            (fun s ->
              equal float_exact Float.nan (Scale.normalize s Float.nan);
              equal float_exact Float.nan (Scale.normalize s Float.infinity);
              equal float_exact Float.nan (Scale.normalize s Float.neg_infinity))
            [
              Scale.linear ();
              Scale.log ();
              Scale.symlog ();
              Scale.pow ~exponent:2. ();
              asinh ();
              Scale.linear ~clamp:true ();
              Scale.linear ~domain:(1., 1.) ();
            ]);
      missing_law;
      test "a custom transform is missing where it is not finite" (fun () ->
          let s = ln ~domain:(1., Float.exp 2.) () in
          equal float_exact Float.nan (Scale.normalize s 0.);
          equal float_exact Float.nan (Scale.normalize s (-1.)));
      test "an unfitted logit normalises every value to nan" (fun () ->
          equal float_exact Float.nan (Scale.normalize (logit ()) 0.5));
      test "a name outside the categories is missing" (fun () ->
          let s =
            Scale.band ~domain:(Indices [| (3, "the"); (7, "cat") |]) ()
          in
          equal float_exact Float.nan (Scale.normalize s "the");
          equal float_exact Float.nan (Scale.normalize s "03"));
    ]

(* Categories *)

let categories =
  group "categories"
    [
      test "an indexed category is named by its integer" (fun () ->
          let s =
            Scale.band ~domain:(Indices [| (3, "the"); (7, "cat") |]) ()
          in
          equal float_exact 0.25 (Scale.normalize s "3");
          equal float_exact 0.75 (Scale.normalize s "7"));
      test "indexed texts may repeat" (fun () ->
          let s =
            Scale.band ~domain:(Indices [| (1, "the"); (2, "the") |]) ()
          in
          equal float_exact 0.25 (Scale.normalize s "1"));
      test "labels are copied in and out" (fun () ->
          let a = [| "a"; "b" |] in
          let s = Scale.band ~domain:(Labels a) () in
          a.(0) <- "z";
          (match Scale.domain s with
          | Categories (Labels l) -> l.(0) <- "y"
          | Categories (Indices _) -> ());
          equal (array string) [| "a"; "b" |] (names s));
      test "indices are copied in and out" (fun () ->
          let ix = [| (1, "x"); (2, "y") |] in
          let s = Scale.band ~domain:(Indices ix) () in
          ix.(0) <- (5, "z");
          (match Scale.domain s with
          | Categories (Indices ix) -> ix.(1) <- (9, "w")
          | Categories (Labels _) -> ());
          equal bscale
            (Scale.band ~domain:(Indices [| (1, "x"); (2, "y") |]) ())
            s);
    ]

(* Constructors *)

let constructors =
  group "constructors"
    [
      test "each default domain is the one its constructor states" (fun () ->
          equal ends_w (0., 1.) (ends (Scale.linear ()));
          equal ends_w (1., 10.) (ends (Scale.log ()));
          equal ends_w (1., 2.) (ends (Scale.log ~base:2. ()));
          equal ends_w (0., 1.) (ends (Scale.symlog ()));
          equal ends_w (0., 1.) (ends (Scale.pow ~exponent:3. ()));
          equal ends_w (0., 1.) (ends (asinh ()));
          equal (pair instant instant)
            (Time.epoch, Time.of_date (1970, 1, 2))
            (instants (Scale.time ()));
          equal (array string) [||] (names (Scale.band ())));
      test "a domain that is not finite and ordered is refused" (fun () ->
          invalid (fun () -> Scale.linear ~domain:(Float.nan, 1.) ());
          invalid (fun () -> Scale.linear ~domain:(0., Float.infinity) ());
          invalid (fun () -> Scale.linear ~domain:(2., 1.) ());
          invalid (fun () -> Scale.time ~domain:(Time.v S 2L, Time.v S 1L) ()));
      test "a log domain must be positive" (fun () ->
          invalid (fun () -> Scale.log ~domain:(0., 1.) ());
          invalid (fun () -> Scale.log ~domain:(-1., 1.) ()));
      test "a custom domain must have ends where forward is finite" (fun () ->
          invalid (fun () -> ln ~domain:(0., 1.) ()));
      test "a log base must be finite and greater than 1" (fun () ->
          invalid (fun () -> Scale.log ~base:1. ());
          invalid (fun () -> Scale.log ~base:Float.infinity ());
          invalid (fun () -> Scale.log ~base:Float.nan ()));
      test "a symlog constant must be finite and positive" (fun () ->
          invalid (fun () -> Scale.symlog ~constant:0. ());
          invalid (fun () -> Scale.symlog ~constant:(-1.) ());
          invalid (fun () -> Scale.symlog ~constant:Float.infinity ());
          invalid (fun () -> Scale.symlog ~constant:Float.nan ()));
      test "a pow exponent must be finite and positive" (fun () ->
          invalid (fun () -> Scale.pow ~exponent:0. ());
          invalid (fun () -> Scale.pow ~exponent:(-1.) ());
          invalid (fun () -> Scale.pow ~exponent:Float.infinity ());
          invalid (fun () -> Scale.pow ~exponent:Float.nan ()));
      test "areas must be finite and not negative" (fun () ->
          invalid (fun () -> Scale.linear ~areas:(-1., 4.) ());
          invalid (fun () -> Scale.linear ~areas:(4., -1.) ());
          invalid (fun () -> Scale.linear ~areas:(0., Float.nan) ());
          invalid (fun () -> Scale.linear ~areas:(0., Float.infinity) ()));
      test "an offset must be within a day of UTC" (fun () ->
          invalid (fun () -> Scale.time ~tz_offset_s:86_400 ());
          invalid (fun () -> Scale.time ~tz_offset_s:(-86_400) ()));
      test "a padding must lie in [0;1]" (fun () ->
          invalid (fun () -> Scale.band ~padding:(-0.1) ());
          invalid (fun () -> Scale.band ~padding:1.5 ());
          invalid (fun () -> Scale.band ~padding:Float.nan ()));
      test "a wrap must be at least 1" (fun () ->
          invalid (fun () -> Scale.band ~wrap:0 ()));
      test "labels must be distinct" (fun () ->
          invalid (fun () -> Scale.band ~domain:(Labels [| "a"; "a" |]) ()));
      test "integers must strictly increase" (fun () ->
          invalid (fun () ->
              Scale.band ~domain:(Indices [| (2, "x"); (1, "y") |]) ());
          invalid (fun () ->
              Scale.band ~domain:(Indices [| (1, "x"); (1, "y") |]) ()));
      test "symbols must not be empty" (fun () ->
          invalid (fun () -> Scale.band ~symbols:[||] ()));
      test "dashes must not be empty" (fun () ->
          invalid (fun () -> Scale.band ~dashes:[||] ()));
    ]

(* Properties *)

let pp_transform ppf = function
  | Scale.Linear -> Format.pp_print_string ppf "Linear"
  | Log b -> Format.fprintf ppf "Log %g" b
  | Symlog c -> Format.fprintf ppf "Symlog %g" c
  | Pow e -> Format.fprintf ppf "Pow %g" e
  | Custom n -> Format.fprintf ppf "Custom %S" n

let transform_w = Testable.make ~pp:pp_transform ~equal:( = )

let properties =
  group "properties"
    [
      test "kind is the kind of the constructor" (fun () ->
          is_some
            (Scale.equal_kind (Scale.kind (Scale.linear ())) Scale.Quantitative);
          is_some (Scale.equal_kind (Scale.kind (Scale.time ())) Scale.Temporal);
          is_some
            (Scale.equal_kind (Scale.kind (Scale.band ())) Scale.Categorical));
      (* [Type.eq] has no printer: [is_some] and [is_none] are its verbs. *)
      test "equal_kind holds iff the kinds are the same" (fun () ->
          is_some (Scale.equal_kind Scale.Quantitative Scale.Quantitative);
          is_some (Scale.equal_kind Scale.Temporal Scale.Temporal);
          is_some (Scale.equal_kind Scale.Categorical Scale.Categorical);
          is_none (Scale.equal_kind Scale.Quantitative Scale.Temporal);
          is_none (Scale.equal_kind Scale.Temporal Scale.Categorical);
          is_none (Scale.equal_kind Scale.Categorical Scale.Quantitative));
      test "name is the name set" (fun () ->
          equal (option string) (Some "y")
            (Scale.name (Scale.linear ~name:"y" ()));
          equal (option string) None (Scale.name (Scale.band ())));
      prop "bandwidth is (1 - p) / (n + p), and 0 without a category"
        ~examples:
          [ band_of (4, 0.2, false, false); band_of (0, 0.5, false, false) ]
        gen_band
        (fun { n; padding = p; b; _ } ->
          cover "no category" (n = 0);
          let expected =
            if n = 0 then 0. else (1. -. p) /. (Float.of_int n +. p)
          in
          equal (float band_tolerance) expected (Scale.bandwidth b));
      (* The transforms of these ends are at most 3, each within 4 roundings of
         half an ulp, and the difference adds one: 8 ε bounds its error. *)
      test "length is |T b - T a| on a continuous domain" (fun () ->
          let length s = Scale.length s in
          let tol = float (8. *. Float.epsilon) in
          equal tol 3. (length (Scale.linear ~domain:(2., 5.) ()));
          equal tol 3. (length (Scale.log ~domain:(1., 1000.) ()));
          equal tol 3. (length (Scale.log ~base:2. ~domain:(1., 8.) ()));
          equal tol 1.
            (length (Scale.symlog ~domain:(0., Float.exp 1. -. 1.) ()));
          equal tol 1. (length (Scale.pow ~exponent:2. ~domain:(0., 2.) ()));
          equal tol 2.
            (length
               (Scale.linear ~reverse:true ~clamp:true ~domain:(-1., 1.) ()));
          equal float_exact 0. (length (Scale.linear ~domain:(4., 4.) ())));
      test "length is in seconds on a temporal scale" (fun () ->
          equal float_exact 86400. (Scale.length (Scale.time ()));
          let a = Time.v Ns 0L and b = Time.v Ns 1_500_000_000L in
          equal float_exact 1.5 (Scale.length (Scale.time ~domain:(a, b) ())));
      test "length is n + p on a band scale, and 0 without a category"
        (fun () ->
          let cats = Scale.Labels [| "a"; "b"; "c" |] in
          equal float_exact 3. (Scale.length (Scale.band ~domain:cats ()));
          equal float_exact 3.5
            (Scale.length (Scale.band ~padding:0.5 ~domain:cats ()));
          equal float_exact 0. (Scale.length (Scale.band ()));
          equal float_exact 0. (Scale.length (Scale.band ~padding:0.5 ())));
      test "length is infinite where the difference overflows" (fun () ->
          equal float_exact Float.infinity
            (Scale.length
               (Scale.linear ~domain:(-.Float.max_float, Float.max_float) ())));
      test "length is nan where an end is missing" (fun () ->
          equal float_exact Float.nan (Scale.length (logit ())));
      test "wrap is the wrap set" (fun () ->
          equal (option int) (Some 3) (Scale.wrap (Scale.band ~wrap:3 ()));
          equal (option int) (Some 1) (Scale.wrap (Scale.band ~wrap:1 ()));
          equal (option int) None (Scale.wrap (Scale.band ())));
      test "scheme is the scheme set" (fun () ->
          let scheme = Testable.make ~pp:Scheme.pp ~equal:Scheme.equal in
          equal (option scheme) (Some Scheme.viridis)
            (Scale.scheme (Scale.linear ~scheme:Scheme.viridis ()));
          equal (option scheme) (Some Scheme.okabe_ito)
            (Scale.scheme (Scale.band ~scheme:Scheme.okabe_ito ()));
          equal (option scheme) None (Scale.scheme (Scale.time ())));
      test "areas is the pair set" (fun () ->
          equal
            (option (pair float_exact float_exact))
            (Some (1., 0.))
            (Scale.areas (Scale.linear ~areas:(1., 0.) ()));
          equal
            (option (pair float_exact float_exact))
            None
            (Scale.areas (Scale.linear ())));
      test "symbols are copied in and out" (fun () ->
          let a = [| Symbol.circle; Symbol.square |] in
          let s = Scale.band ~symbols:a () in
          a.(0) <- Symbol.star;
          Option.iter (fun a -> a.(1) <- Symbol.star) (Scale.symbols s);
          equal
            (option (array symbol))
            (Some [| Symbol.circle; Symbol.square |])
            (Scale.symbols s);
          equal (option (array symbol)) None (Scale.symbols (Scale.band ())));
      test "dashes are copied in and out" (fun () ->
          let ds = [| Dash.dashed |] in
          let s = Scale.band ~dashes:ds () in
          ds.(0) <- Dash.solid;
          Option.iter (fun ds -> ds.(0) <- Dash.dotted) (Scale.dashes s);
          equal (option (array dash)) (Some [| Dash.dashed |]) (Scale.dashes s));
      test "stepped holds only when set to true" (fun () ->
          equal bool false (Scale.stepped (Scale.log ()));
          equal bool false
            (Scale.stepped (Scale.pow ~exponent:2. ~stepped:false ()));
          equal bool true (Scale.stepped (Scale.symlog ~stepped:true ())));
      prop "stepped leaves normalize unchanged"
        (Gen.pair gen_scale Gen.any_float) (fun (q, x) ->
          let stepped = Scale.imply (Scale.linear ~stepped:true ()) q.s in
          equal float_exact (Scale.normalize q.s x) (Scale.normalize stepped x));
      test "ticks are copied in and out" (fun () ->
          let vs = [| 1.; 2. |] in
          let s = Scale.linear ~ticks:vs () in
          vs.(0) <- 5.;
          Option.iter (fun vs -> vs.(1) <- 5.) (Scale.ticks s);
          equal (option (array float_exact)) (Some [| 1.; 2. |]) (Scale.ticks s);
          equal (option (array string)) None (Scale.ticks (Scale.band ())));
      test "notation is the notation set" (fun () ->
          equal (option notation) (Some Exponent)
            (Scale.notation (Scale.pow ~exponent:2. ~notation:Exponent ()));
          equal (option notation) None (Scale.notation (Scale.log ())));
      test "unknown is the colour set" (fun () ->
          let color = Testable.make ~pp:Color.pp ~equal:Color.equal in
          equal (option color) (Some Color.red)
            (Scale.unknown (Scale.band ~unknown:Color.red ()));
          equal (option color) None (Scale.unknown (Scale.linear ())));
      cases "transform names the constructor's"
        ~id:(fun (n, _, _) -> n)
        [
          ("linear", (fun () -> Scale.linear ()), Scale.Linear);
          ("log", (fun () -> Scale.log ()), Scale.Log 10.);
          ("log in base 2", (fun () -> Scale.log ~base:2. ()), Scale.Log 2.);
          ("symlog", (fun () -> Scale.symlog ~constant:3. ()), Scale.Symlog 3.);
          ("pow", (fun () -> Scale.pow ~exponent:0.5 ()), Scale.Pow 0.5);
          ("custom", (fun () -> ln ()), Scale.Custom "ln");
        ]
        (fun (_, s, tf) -> equal transform_w tf (Scale.transform (s ())));
      test "tz_offset_s is the offset set, 0 unset" (fun () ->
          equal int 0 (Scale.tz_offset_s (Scale.time ()));
          equal int (-3600)
            (Scale.tz_offset_s (Scale.time ~tz_offset_s:(-3600) ())));
    ]

(* Normalising and inverting *)

let normalize_index =
  group "normalize_index"
    [
      prop
        "an index normalises as its category, and an index off the domain to \
         nan"
        (Gen.pair gen_band (Gen.int_range (-2) 8))
        (fun ({ b; _ }, k) ->
          let names = names b in
          let inside = 0 <= k && k < Array.length names in
          cover "an index of the domain" inside;
          cover "an index off the domain" (not inside);
          let expected =
            if inside then Scale.normalize b names.(k) else Float.nan
          in
          equal float_exact expected (Scale.normalize_index b k));
    ]

(* Families of moderate domains whose inverse transforms the .mli states. *)
let gen_invertible =
  with_pp_q
    (Gen.one_of
       [
         Gen.map (on_symlog 1.) gen_moderate;
         Gen.map (on_pow 2.) gen_moderate;
         Gen.map (on_pow 0.5) gen_moderate;
         Gen.map on_asinh gen_moderate;
       ])

let invert =
  group "invert"
    [
      prop "a linear scale inverts its normalisation up to rounding"
        ~examples:[ ((1.205505664932564e-305, 9914.), 0.) ]
        (Gen.pair gen_moderate (Gen.float_range (-2e6) 2e6))
        (fun ((a, b), x) ->
          let s = Scale.linear ~domain:(a, b) () in
          (* Normalising rounds twice and inverting, [(1 - u) a + u b], four
             times: six roundings, each within ε of [|x|] or of [|u| (|a| +
             |b|)], the magnitudes of the terms. 8 ε leaves two to spare. A
             subnormal [u], product or sum is within 2^-1075 of itself instead,
             which [b - a] carries to [x]: [2^-1075 (|a| + |b| + 3)] at most,
             within [2^-1074 (2 + |a| + |b|)]. *)
          let u = Scale.normalize s x in
          let tol =
            8. *. Float.epsilon
            *. (Float.abs x +. (Float.abs u *. (Float.abs a +. Float.abs b)))
            +. (0x1p-1074 *. (2. +. Float.abs a +. Float.abs b))
          in
          equal (option (near tol)) (Some x) (Scale.invert s u));
      prop "a log scale inverts its normalisation up to rounding"
        (Gen.pair (Gen.float_range 1e-3 1e3) (Gen.float_range 1e-2 1e2))
        (fun (a, x) ->
          let s = Scale.log ~domain:(a, a *. 1e3) () in
          (* The transforms are at most 6 in magnitude, the domain spans 3 and
             [|u|] is at most 5/3: about twenty roundings, each carried through
             at most twice these magnitudes, keep the exponent [v] within 100 ε,
             and [x = 10^v] within [ln 10 × 100 ε < 256 ε] of itself. *)
          equal
            (option (float_rel ~rel:(256. *. Float.epsilon) ~abs:0.))
            (Some x)
            (Scale.invert s (Scale.normalize s x)));
      prop "symlog, pow and custom scales normalise their inverses back"
        (Gen.pair gen_invertible (Gen.float_range (-0.5) 1.5))
        (fun (q, u) ->
          let x = require_some (Scale.invert q.s u) in
          (* Inverting rounds [(1 - u) T a + u T b] four times, on terms at most
             [2.5 (|T a| + |T b|)], and the inverse transform up to four times,
             which [T], of condition at most 2 here, carries to [T x];
             normalising rounds each [T] four times and the formula twice. About
             fifty half ulps of [|T a| + |T b| + |T x|], over [|T b - T a|]. *)
          let a, b = ends q.s in
          let ta = q.t a and tb = q.t b and tx = q.t x in
          let tol =
            32. *. Float.epsilon
            *. (Float.abs ta +. Float.abs tb +. Float.abs tx)
            /. Float.abs (tb -. ta)
          in
          equal (near tol) u (Scale.normalize q.s x));
      prop "a band scale inverts the centre of each category"
        (Gen.triple (Gen.int_range 1 30) (Gen.float_range 0. 1.) Gen.bool)
        (fun (n, padding, reverse) ->
          let names = Array.init n string_of_int in
          let s = Scale.band ~padding ~reverse ~domain:(Labels names) () in
          Array.iter
            (fun c ->
              equal (option string) (Some c)
                (Scale.invert s (Scale.normalize s c)))
            names);
      prop "an instant within 2^50 ns of the domain's start inverts exactly"
        (Gen.triple
           (Gen.int_range 0 (1 lsl 50))
           (Gen.int_range 1 (1 lsl 50))
           (Gen.int_range (-1_000_000_000) 1_000_000_000))
        (fun (d, w, s0) ->
          let a = Time.v S (Int64.of_int s0) in
          let ns k = Time.add (Time.nanoseconds 1) k a in
          let s = Scale.time ~domain:(a, ns w) () in
          equal (option instant)
            (Some (ns d))
            (Scale.invert s (Scale.normalize s (ns d))));
      test "an instant is inverted on a domain wider than int64 seconds"
        (fun () ->
          let a = Time.v S Int64.min_int and b = Time.v S Int64.max_int in
          let s = Scale.time ~domain:(a, b) () in
          equal (option instant) (Some b) (Scale.invert s 1.);
          equal (option instant) (Some a) (Scale.invert s 0.));
      test "an instant that is not representable is nothing" (fun () ->
          let a = Time.v S Int64.min_int and b = Time.v S Int64.max_int in
          let s = Scale.time ~domain:(a, b) () in
          equal (option instant) None (Scale.invert s 1.5));
      test "a value that is not finite inverts to nothing" (fun () ->
          let s = Scale.linear () in
          equal (option float_exact) None (Scale.invert s Float.nan);
          equal (option float_exact) None (Scale.invert s Float.infinity);
          equal (option float_exact) None (Scale.invert s Float.neg_infinity);
          equal (option string) None
            (Scale.invert (Scale.band ~domain:(Labels [| "a" |]) ()) Float.nan));
      test "an inverse that is not finite is nothing" (fun () ->
          let s = Scale.linear ~domain:(0., Float.max_float) () in
          equal (option float_exact) None (Scale.invert s 3.));
      test "an inverse that is missing is nothing" (fun () ->
          let s = Scale.log ~domain:(1., 10.) () in
          equal (option float_exact) None (Scale.invert s (-400.));
          let s = ln ~domain:(1., Float.exp 2.) () in
          equal (option float_exact) None (Scale.invert s (-400.)));
      test "a clamping scale clamps before inverting" (fun () ->
          let s = Scale.linear ~clamp:true ~domain:(0., 10.) () in
          equal (option float_exact) (Some 10.) (Scale.invert s 3.);
          let s = Scale.linear ~clamp:true ~reverse:true ~domain:(0., 10.) () in
          equal (option float_exact) (Some 0.) (Scale.invert s 3.));
      test "a constant domain inverts every finite value to its end" (fun () ->
          let s = Scale.linear ~domain:(2., 2.) () in
          equal (option float_exact) (Some 2.) (Scale.invert s 0.7);
          let t = Time.v S 9L in
          equal (option instant) (Some t)
            (Scale.invert (Scale.time ~domain:(t, t) ()) 3.));
      test "a band inverts the step that holds a value" (fun () ->
          let s = Scale.band ~padding:0.5 ~domain:(Labels [| "a"; "b" |]) () in
          equal (option string) (Some "a") (Scale.invert s 0.1);
          equal (option string) (Some "b") (Scale.invert s 0.9));
      test "a band inverts a boundary to the earlier category" (fun () ->
          let s = Scale.band ~padding:0.5 ~domain:(Labels [| "a"; "b" |]) () in
          equal (option string) (Some "a") (Scale.invert s 0.5);
          equal (option string) (Some "b") (Scale.invert s 0.51));
      test "a reversed band inverts a boundary to the earlier category"
        (fun () ->
          let s = Scale.band ~reverse:true ~domain:(Labels [| "a"; "b" |]) () in
          equal (option string) (Some "a") (Scale.invert s 0.5));
      test "a band inverts the padding, the outside and no category to nothing"
        (fun () ->
          let s = Scale.band ~padding:0.5 ~domain:(Labels [| "a"; "b" |]) () in
          equal (option string) None (Scale.invert s 0.05);
          equal (option string) None (Scale.invert s 0.95);
          equal (option string) None (Scale.invert s 1.5);
          equal (option string) None (Scale.invert (Scale.band ()) 0.5));
      test "an indexed band inverts to the integer's name" (fun () ->
          let s =
            Scale.band ~domain:(Indices [| (3, "the"); (7, "cat") |]) ()
          in
          equal (option string) (Some "7") (Scale.invert s 0.75));
    ]

(* Nice domains *)

(* Rows from d3-scale's linear-test.js and log-test.js, [nice]. *)
let d3_nice =
  [
    (0., 0.96, 0., 1.);
    (0., 96., 0., 100.);
    (-0.96, 0., -1., 0.);
    (-96., 0., -100., 0.);
    (1.1, 10.9, 1., 11.);
    (0.7, 11.001, 0., 12.);
    (6.7, 123.1, 0., 130.);
    (0., 0.49, 0., 0.5);
    (12., 87., 10., 90.);
  ]

let d3_log_nice =
  [
    (1.1, 10.9, 1., 100.);
    (0.7, 11.001, 0.1, 100.);
    (6.7, 123.1, 1., 1000.);
    (0.01, 0.49, 0.01, 1.);
    (1.5, 50., 1., 100.);
  ]

(* A step rounds to 1, 2, 5 or 10 times a power of ten by the ratio of its
   leading digits to them: below sqrt 2 to 1 and from sqrt 50 to 10. Rounding by
   1 then 2, or by 5 then 10, is rounding by the larger, so only the bands (1.4,
   1.5] and (7, 7.5] where those two thresholds lie can be observed. *)
let threshold_bands =
  [
    ("[-5;9] stays: sqrt 2 is above 1.4", (-5., 9., -5., 9.));
    ("[0;15] takes steps of 2: sqrt 2 is at most 1.5", (0., 15., 0., 16.));
    ("[-5;65] stays: sqrt 50 is above 7", (-5., 65., -5., 65.));
    ("[0;75] takes steps of 10: sqrt 50 is at most 7.5", (0., 75., 0., 80.));
  ]

let nice =
  group "nice"
    [
      cases "a linear domain rounds to multiples of the step"
        ~id:(fun (a, b, _, _) -> Printf.sprintf "[%g;%g]" a b)
        d3_nice
        (fun (a, b, a', b') ->
          equal ends_w (a', b') (ends (fit_floats a b (Scale.linear ()))));
      cases "the step changes at its thresholds" ~id:fst threshold_bands
        (fun (_, (a, b, a', b')) ->
          equal ends_w (a', b') (ends (fit_floats a b (Scale.linear ()))));
      (* The lexer gives the float nearest a decimal literal, ties to even:
         [1e23] lies halfway between two floats. *)
      cases "a multiple is the float nearest it"
        ~id:(fun (a, b, _, _) -> Printf.sprintf "[%g;%g]" a b)
        [
          (0., 7.3e30, 0., 8e30);
          (0., 7.3e-30, 0., 8e-30);
          (-1.1e24, 9.5e22, -1.1e24, 1e23);
        ]
        (fun (a, b, a', b') ->
          equal ends_w (a', b') (ends (fit_floats a b (Scale.linear ()))));
      (* A tenth of each domain is [√10] times a power of ten, where the step
         [5] begins: its ends round to multiples of [5], not of [2]. *)
      cases "a tenth at the √10 threshold takes the step 5"
        ~id:(fun (a, b, _, _, _) -> Printf.sprintf "[%g;%.17g]" a b)
        [
          (5., 36.622776601683796, Float.sqrt 10., 5., 40.);
          (0.3, 3.4622776601683793, Float.sqrt 10. /. 10., 0., 3.5);
        ]
        (fun (a, b, tenth, a', b') ->
          equal float_exact tenth ((b -. a) /. 10.);
          equal ends_w (a', b') (ends (fit_floats a b (Scale.linear ()))));
      test "a domain whose length overflows takes twice a tenth of its halves"
        (fun () ->
          (* [t = 2 × (5.5e307 + 4.5e307) / 10 = 2e307], so [f = 2] and the step
             is [2e307]; the ends round to [-5] and [6] of its multiples, which
             keep that step. *)
          equal ends_w (-1e308, 1.2e308)
            (ends (fit_floats (-9e307) 1.1e308 (Scale.linear ()))));
      prop "a custom scale fits as a linear one" gen_moderate (fun (a, b) ->
          let cube =
            Scale.custom ~transform:"cube"
              ~forward:(fun x -> x *. x *. x)
              ~inverse:Float.cbrt ()
          in
          equal ends_w
            (ends (fit_floats a b (Scale.linear ())))
            (ends (fit_floats a b cube)));
      cases "a log domain rounds to powers of the base"
        ~id:(fun (a, b, _, _) -> Printf.sprintf "[%g;%g]" a b)
        d3_log_nice
        (fun (a, b, a', b') ->
          equal ends_w (a', b') (ends (fit_floats a b (Scale.log ()))));
      test "a log domain in base 2 rounds to powers of 2" (fun () ->
          equal ends_w (2., 32.)
            (ends (fit_floats 3. 20. (Scale.log ~base:2. ()))));
      test "symlog rounds to signed powers, zero and the constant" (fun () ->
          let fit a b = ends (fit_floats a b (Scale.symlog ())) in
          equal ends_w (-100., 1000.) (fit (-42.) 250.);
          equal ends_w (0., 1.) (fit 0.2 0.7);
          equal ends_w (-1., 10.) (fit (-0.5) 3.));
      test "an end within the constant rounds to zero or the constant"
        (fun () ->
          let fit a b = ends (fit_floats a b (Scale.symlog ~constant:5. ())) in
          equal ends_w (5., 100.) (fit 5. 70.);
          equal ends_w (-5., 100.) (fit (-5.) 70.);
          equal ends_w (0., 5.) (fit 2. 3.));
      test "an end beyond the constant stays beyond it" (fun () ->
          let fit a b = ends (fit_floats a b (Scale.symlog ~constant:5. ())) in
          equal ends_w (5., 100.) (fit 7. 70.);
          equal ends_w (-100., -5.) (fit (-70.) (-7.)));
      (* Rows from d3-scale's time-test.js, [nice]. *)
      test "a temporal domain rounds to the boundaries of the nearest interval"
        (fun () ->
          let nice a b = fit_instants a b (Scale.time ()) in
          let w = pair instant instant in
          equal w
            (utc (2009, 1, 1) (0, 0, 0), utc (2009, 1, 2) (0, 0, 0))
            (nice (utc (2009, 1, 1) (0, 17, 0)) (utc (2009, 1, 1) (23, 42, 0)));
          equal w
            (utc (2013, 1, 1) (12, 0, 0), utc ~ms:130 (2013, 1, 1) (12, 0, 0))
            (nice
               (utc (2013, 1, 1) (12, 0, 0))
               (utc ~ms:128 (2013, 1, 1) (12, 0, 0)));
          equal w
            (utc (2000, 1, 1) (0, 0, 0), utc (2140, 1, 1) (0, 0, 0))
            (nice (utc (2001, 1, 1) (0, 0, 0)) (utc (2138, 1, 1) (0, 0, 0))));
      test "a constant domain is not rounded" (fun () ->
          equal ends_w (0.5, 0.5) (ends (fit_floats 0.5 0.5 (Scale.linear ())));
          equal ends_w (0.5, 0.5) (ends (fit_floats 0.5 0.5 (Scale.log ())));
          let t = utc (2009, 1, 1) (0, 12, 0) in
          equal (pair instant instant) (t, t) (fit_instants t t (Scale.time ())));
      test "an end rounded beyond the floats stays" (fun () ->
          equal ends_w (0., 1.75e308)
            (ends (fit_floats 0. 1.75e308 (Scale.linear ())));
          equal ends_w (1., 1.5e308)
            (ends (fit_floats 1. 1.5e308 (Scale.log ()))));
      test "an end rounded to a missing value stays" (fun () ->
          equal ends_w (5e-324, 1.) (ends (fit_floats 5e-324 1. (Scale.log ()))));
      test "an instant rounded beyond the representable stays" (fun () ->
          let b = Time.v S Int64.max_int in
          let a = Time.add (Time.seconds 1) (-1000) b in
          equal instant b (snd (fit_instants a b (Scale.time ()))));
      test "zero widens the domain before it is rounded" (fun () ->
          equal ends_w (0., 7.)
            (ends (fit_floats 3. 7. (Scale.linear ~zero:true ())));
          equal ends_w (-10., 0.)
            (ends (fit_floats (-9.2) (-3.) (Scale.linear ~zero:true ()))));
      test "zero never reaches a log scale" (fun () ->
          let s = Scale.imply (Scale.linear ~zero:true ()) (Scale.log ()) in
          equal ends_w (1., 10.) (ends (fit_floats 2. 8. s)));
      test "nice false keeps the hull" (fun () ->
          equal ends_w (0.3, 9.7)
            (ends (fit_floats 0.3 9.7 (Scale.linear ~nice:false ()))));
    ]

(* Specifications *)

let all_properties =
  Scale.
    [
      Name;
      Transform;
      Domain;
      Nice;
      Zero;
      Clamp;
      Reverse;
      Stepped;
      Ticks;
      Notation;
      Padding;
      Wrap;
      Tz_offset_s;
      Scheme;
      Areas;
      Symbols;
      Dashes;
      Unknown;
    ]

let set_by s = List.filter (fun p -> Scale.sets p s) all_properties

let red = Color.red
and blue = Color.blue

(* Two scales that set one property to different values. Rows hold the
   constructions, made in the test. *)
type conflict =
  | Conflict : Scale.property * (unit -> 'd Scale.t * 'd Scale.t) -> conflict

let conflicts =
  let band = Scale.band and time = Scale.time in
  [
    Conflict
      (Name, fun () -> (Scale.linear ~name:"x" (), Scale.linear ~name:"y" ()));
    Conflict (Transform, fun () -> (Scale.log (), Scale.linear ()));
    Conflict
      ( Domain,
        fun () ->
          (Scale.linear ~domain:(0., 1.) (), Scale.linear ~domain:(0., 2.) ())
      );
    Conflict
      (Nice, fun () -> (Scale.linear ~nice:true (), Scale.linear ~nice:false ()));
    Conflict
      (Zero, fun () -> (Scale.linear ~zero:true (), Scale.linear ~zero:false ()));
    Conflict
      ( Clamp,
        fun () -> (Scale.linear ~clamp:true (), Scale.linear ~clamp:false ()) );
    Conflict
      ( Reverse,
        fun () -> (Scale.linear ~reverse:true (), Scale.linear ~reverse:false ())
      );
    Conflict
      ( Stepped,
        fun () -> (Scale.linear ~stepped:true (), Scale.linear ~stepped:false ())
      );
    Conflict
      ( Ticks,
        fun () ->
          (Scale.log ~ticks:[| 1.; 10. |] (), Scale.log ~ticks:[| 10.; 1. |] ())
      );
    Conflict
      ( Notation,
        fun () ->
          (Scale.linear ~notation:Percent (), Scale.linear ~notation:Plain ())
      );
    Conflict (Padding, fun () -> (band ~padding:0.1 (), band ~padding:0.2 ()));
    Conflict (Wrap, fun () -> (band ~wrap:1 (), band ~wrap:2 ()));
    Conflict
      (Tz_offset_s, fun () -> (time ~tz_offset_s:0 (), time ~tz_offset_s:60 ()));
    Conflict
      ( Scheme,
        fun () ->
          ( Scale.linear ~scheme:Scheme.viridis (),
            Scale.linear ~scheme:Scheme.magma () ) );
    Conflict
      ( Areas,
        fun () ->
          (Scale.linear ~areas:(0., 4.) (), Scale.linear ~areas:(1., 4.) ()) );
    Conflict
      ( Symbols,
        fun () ->
          ( band ~symbols:[| Symbol.circle |] (),
            band ~symbols:[| Symbol.circle; Symbol.square |] () ) );
    Conflict
      ( Dashes,
        fun () ->
          ( band ~dashes:[| Dash.solid |] (),
            band ~dashes:[| Dash.solid; Dash.dashed |] () ) );
    Conflict
      ( Unknown,
        fun () -> (Scale.linear ~unknown:red (), Scale.linear ~unknown:blue ())
      );
  ]

let conflict_name (Conflict (p, _)) = Format.asprintf "%a" Scale.pp_property p

let sets =
  group "sets"
    [
      test "every constructor sets the transform alone" (fun () ->
          let transform = [ Scale.Transform ] in
          equal (list property) transform (set_by (Scale.linear ()));
          equal (list property) transform (set_by (Scale.log ()));
          equal (list property) transform (set_by (Scale.symlog ()));
          equal (list property) transform (set_by (Scale.pow ~exponent:2. ()));
          equal (list property) transform (set_by (asinh ()));
          equal (list property) transform (set_by (Scale.time ()));
          equal (list property) transform (set_by (Scale.band ())));
      cases "a constructor sets the properties it is given" ~id:conflict_name
        conflicts (fun (Conflict (p, make)) ->
          let s, s' = make () in
          let expected =
            List.filter (fun q -> q = p || q = Scale.Transform) all_properties
          in
          equal (list property) expected (set_by s);
          equal (list property) expected (set_by s'));
    ]

(* The properties of a quantitative specification, each set or unset, with
   values from pools of two or three so that two specifications agree, extend
   and conflict. [symlog] picks the transform, which is always set. *)
type spec = {
  symlog : bool;
  name : string option;
  domain : (float * float) option;
  nice : bool option;
  zero : bool option;
  clamp : bool option;
  reverse : bool option;
  stepped : bool option;
  ticks : float array option;
  notation : Number.notation option;
  scheme : Scheme.t option;
  areas : (float * float) option;
  unknown : Color.t option;
}

let unset =
  {
    symlog = false;
    name = None;
    domain = None;
    nice = None;
    zero = None;
    clamp = None;
    reverse = None;
    stepped = None;
    ticks = None;
    notation = None;
    scheme = None;
    areas = None;
    unknown = None;
  }

let scale p =
  let make = if p.symlog then Scale.symlog ?constant:None else Scale.linear in
  make ?name:p.name ?domain:p.domain ?nice:p.nice ?zero:p.zero ?clamp:p.clamp
    ?reverse:p.reverse ?stepped:p.stepped ?ticks:p.ticks ?notation:p.notation
    ?scheme:p.scheme ?areas:p.areas ?unknown:p.unknown ()

(* [union p q] sets each property [p] or [q] sets, to its value in [p] where [p]
   sets it, with the transform of [p]. *)
let union p q =
  let ( ||| ) o o' = if Option.is_some o then o else o' in
  {
    p with
    name = p.name ||| q.name;
    domain = p.domain ||| q.domain;
    nice = p.nice ||| q.nice;
    zero = p.zero ||| q.zero;
    clamp = p.clamp ||| q.clamp;
    reverse = p.reverse ||| q.reverse;
    stepped = p.stepped ||| q.stepped;
    ticks = p.ticks ||| q.ticks;
    notation = p.notation ||| q.notation;
    scheme = p.scheme ||| q.scheme;
    areas = p.areas ||| q.areas;
    unknown = p.unknown ||| q.unknown;
  }

(* [extend p] is a specification that agrees with [p]: it sets each property [p]
   sets to the same value or leaves it unset, and draws the others. *)
let extend p =
  let field o pool =
    match o with
    | Some _ -> Gen.map (fun keep -> if keep then o else None) Gen.bool
    | None -> Gen.option (Gen.of_list pool)
  in
  Gen.(
    let+ name = field p.name [ "x"; "y" ]
    and+ domain = field p.domain [ (0., 1.); (0., 2.) ]
    and+ nice = field p.nice [ true; false ]
    and+ zero = field p.zero [ true; false ]
    and+ clamp = field p.clamp [ true; false ]
    and+ reverse = field p.reverse [ true; false ]
    and+ stepped = field p.stepped [ true; false ]
    and+ ticks = field p.ticks [ [| 0.; 1. |]; [| 1.; 0. |]; [| 0.5 |] ]
    and+ notation = field p.notation [ Number.Percent; Si ]
    and+ scheme =
      field p.scheme [ Scheme.viridis; Scheme.reverse Scheme.viridis ]
    and+ areas = field p.areas [ (0., 4.); (1., 4.) ]
    and+ unknown = field p.unknown [ red; blue ] in
    {
      p with
      name;
      domain;
      nice;
      zero;
      clamp;
      reverse;
      stepped;
      ticks;
      notation;
      scheme;
      areas;
      unknown;
    })

let pp_spec ppf p = Scale.pp ppf (scale p)

let gen_spec =
  Gen.(
    let* symlog = frequency [ (5, constant false); (1, constant true) ] in
    extend { unset with symlog })
  |> Gen.with_pp pp_spec

let pp_specs ppf l =
  Format.pp_print_list ~pp_sep:Format.pp_print_space pp_spec ppf l

(* Pairs of which three in four agree, the second drawn from the first. *)
let gen_pair =
  Gen.(
    let* a = gen_spec in
    let+ b = frequency [ (3, extend a); (1, gen_spec) ] in
    (a, b))
  |> Gen.with_pp (fun ppf (a, b) -> pp_specs ppf [ a; b ])

let gen_agreeing =
  Gen.(
    let* a = gen_spec in
    let+ b = extend a in
    (a, b))
  |> Gen.with_pp (fun ppf (a, b) -> pp_specs ppf [ a; b ])

let merge =
  group "merge"
    [
      prop "a merge sets each property either sets, to its value there"
        ~examples:
          [
            ( { unset with name = Some "y"; zero = Some true },
              { unset with domain = Some (0., 2.) } );
          ]
        gen_agreeing
        (fun (a, b) ->
          equal merged
            (Ok (scale (union a b)))
            (Scale.merge (scale a) (scale b)));
      cases "a conflict names the property" ~id:conflict_name conflicts
        (fun (Conflict (p, make)) ->
          let s, s' = make () in
          equal (option property) (Some p)
            (match Scale.merge s s' with Ok _ -> None | Error p -> Some p));
      test "a conflict names the first property in their order" (fun () ->
          equal merged (Error Scale.Name)
            (Scale.merge
               (Scale.linear ~name:"x" ~domain:(0., 1.) ())
               (Scale.linear ~name:"y" ~domain:(0., 2.) ())));
      test "the base of a log scale is part of its transform" (fun () ->
          equal merged (Error Scale.Transform)
            (Scale.merge (Scale.log ~base:2. ()) (Scale.log ())));
      test "band domains agree when their categories are equal" (fun () ->
          let band d = Scale.band ~domain:d () in
          let merged = result bscale property in
          let same d =
            equal merged (Ok (band d)) (Scale.merge (band d) (band d))
          in
          let differ d d' =
            equal merged (Error Scale.Domain) (Scale.merge (band d) (band d'))
          in
          same (Labels [| "a"; "b" |]);
          differ (Labels [| "a"; "b" |]) (Labels [| "b"; "a" |]);
          differ (Labels [| "a" |]) (Labels [| "a"; "b" |]);
          same (Indices [| (1, "x") |]);
          differ (Indices [| (1, "x") |]) (Indices [| (1, "y") |]);
          differ (Indices [| (1, "x") |]) (Indices [| (2, "x") |]);
          differ (Indices [||]) (Indices [| (2, "x") |]));
      test "labelled and indexed categories conflict" (fun () ->
          equal (result bscale property) (Error Scale.Domain)
            (Scale.merge
               (Scale.band ~domain:(Labels [||]) ())
               (Scale.band ~domain:(Indices [||]) ())));
      test "custom transforms agree only with the same functions" (fun () ->
          let f = Float.asinh and g = Float.sinh in
          let a = Scale.custom ~transform:"asinh" ~forward:f ~inverse:g () in
          equal merged (Ok a)
            (Scale.merge a
               (Scale.custom ~transform:"asinh" ~forward:f ~inverse:g ()));
          equal merged (Error Scale.Transform) (Scale.merge a (asinh ())));
      prop "merge gives the same result in either order" gen_pair (fun (a, b) ->
          let ab = Scale.merge (scale a) (scale b)
          and ba = Scale.merge (scale b) (scale a) in
          cover ~share:0.25 "both orders succeed" (Result.is_ok ab);
          cover "both orders fail" (Result.is_error ab);
          equal merged ab ba);
      prop "merge is associative on its successes"
        (Gen.(
           let* a = gen_spec in
           let+ b = extend a and+ c = extend a in
           (a, b, c))
        |> Gen.with_pp (fun ppf (a, b, c) -> pp_specs ppf [ a; b; c ]))
        (fun (a, b, c) ->
          let a = scale a and b = scale b and c = scale c in
          let left = Result.bind (Scale.merge a b) (fun ab -> Scale.merge ab c)
          and right =
            Result.bind (Scale.merge b c) (fun bc -> Scale.merge a bc)
          in
          cover ~share:0.2 "both groupings succeed" (Result.is_ok left);
          cover "a grouping fails" (Result.is_error left);
          equal (result fscale pass) left right);
      prop "merge s s is Ok s" gen_spec (fun p ->
          equal merged (Ok (scale p)) (Scale.merge (scale p) (scale p)));
    ]

let imply =
  group "imply"
    [
      prop "imply takes each property its subject leaves unset, except the name"
        ~examples:
          [
            ( { unset with name = Some "x"; nice = Some false },
              { unset with symlog = true; name = Some "y" } );
            ({ unset with name = Some "x"; zero = Some true }, unset);
          ]
        (Gen.pair gen_spec gen_spec)
        (fun (i, s) ->
          cover "a named implication and an unnamed subject"
            (Option.is_some i.name && Option.is_none s.name);
          equal fscale
            (scale (union s { i with name = None }))
            (Scale.imply (scale i) (scale s)));
      test "imply leaves out a domain the subject refuses" (fun () ->
          equal fscale (Scale.log ())
            (Scale.imply (Scale.linear ~domain:(-1., 1.) ()) (Scale.log ()));
          equal fscale (ln ())
            (Scale.imply (Scale.linear ~domain:(0., 1.) ()) (ln ())));
      test "imply takes a domain the subject admits" (fun () ->
          equal fscale
            (Scale.log ~domain:(2., 5.) ())
            (Scale.imply (Scale.linear ~domain:(2., 5.) ()) (Scale.log ())));
    ]

(* Missing values *)

(* A set of one software device, computed by nx.cpu as the host is. *)
let memory =
  fixture (fun () -> Nx.devices [ Result.get_ok (Rig.software "m0") ])

(* A set whose kernels run no loop, as a set with no kernel library, such as a
   GPU's, computes nothing eagerly: an operation there, a cast included, raises.
   Its software device's memory is the host's, so a read works. *)
module No_loops = struct
  include Nx_cpu

  let name = "hugin.test"
  let map _ ~dsts:_ _ = Nx_array.Declined
end

let inert =
  fixture (fun () ->
      Nx.devices
        ~kernels:(module No_loops)
        [ Result.get_ok (Rig.software "m1") ])

let placement () = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal
let f64 a = Nx.create Nx.float64 [| Array.length a |] a

(* Scales with set domains, whose ends are not missing, so [normalize] is [nan]
   exactly at missing values. *)
let gen_missing_scale =
  Gen.of_list
    ~pp:(fun ppf (n, _) -> Format.pp_print_string ppf n)
    [
      ("linear", fun () -> Scale.linear ~domain:(0., 1.) ());
      ("log", fun () -> Scale.log ~domain:(1., 10.) ());
      ("symlog", fun () -> Scale.symlog ~domain:(-1., 1.) ());
      ("pow", fun () -> Scale.pow ~exponent:0.5 ~domain:(0., 4.) ());
      ("custom", fun () -> ln ~domain:(1., 10.) ());
    ]

let gen_values =
  Gen.array ~size:(Gen.int_range 0 6)
    (Gen.frequency
       [
         (4, Gen.any_float);
         ( 1,
           Gen.of_list ~pp:Format.pp_print_float
             [ 0.; -0.; -1.; Float.min_float; Float.nan; Float.infinity ] );
       ])

let missing =
  group "missing"
    [
      prop "a mask is true where normalize gives nan"
        (Gen.pair gen_missing_scale gen_values) (fun ((_, s), xs) ->
          let s = s () in
          let expected =
            Array.map (fun x -> Float.is_nan (Scale.normalize s x)) xs
          in
          cover "a missing value" (Array.exists Fun.id expected);
          equal (array bool) expected (Nx.to_array (Scale.missing s (f64 xs))));
      prop "a mask has the placement of x"
        (Gen.pair gen_missing_scale gen_values) (fun ((name, s), xs) ->
          let (module Memory : Nx.Devices) = memory () in
          cover "a custom transform" (name = "custom");
          let m = Scale.missing (s ()) (Nx.place Memory.on (f64 xs)) in
          equal (placement ()) Memory.on (Nx.placement m));
      prop "a mask computed where x lies is the host's"
        (Gen.pair gen_missing_scale gen_values) (fun ((name, s), xs) ->
          let (module Memory : Nx.Devices) = memory () in
          let s = s () in
          cover "a custom transform" (name = "custom");
          equal (array bool)
            (Nx.to_array (Scale.missing s (f64 xs)))
            (Nx.to_array (Scale.missing s (Nx.place Memory.on (f64 xs)))));
      (* A custom transform is an OCaml function, so the tensor is read to the
         host, and no operation runs where it lies. *)
      test "a custom scale reads x to the host" (fun () ->
          let (module Inert : Nx.Devices) = inert () in
          let xs = [| 0.5; -1.; 0.; 3.; Float.nan |] in
          let x = Nx.create Nx.float32 [| 5 |] xs in
          let m =
            Scale.missing (ln ~domain:(1., 10.) ()) (Nx.place Inert.on x)
          in
          equal (placement ()) Inert.on (Nx.placement m);
          equal (array bool)
            [| false; true; true; false; true |]
            (Nx.to_array m));
      test "a mask has the shape of x" (fun () ->
          let x = Nx.create Nx.float64 [| 2; 3 |] (Array.make 6 1.) in
          equal (array int) [| 2; 3 |]
            (Nx.shape (Scale.missing (Scale.log ()) x)));
      test "integers are read as floats" (fun () ->
          let x = Nx.create Nx.int32 [| 4 |] [| 0l; 1l; -3l; 5l |] in
          equal (array bool)
            [| true; false; true; false |]
            (Nx.to_array (Scale.missing (Scale.log ()) x)));
      test "complex and boolean tensors are refused" (fun () ->
          invalid (fun () ->
              Scale.missing (Scale.linear ())
                (Nx.create Nx.complex64 [| 1 |] [| Complex.one |]));
          invalid (fun () ->
              Scale.missing (Scale.linear ())
                (Nx.create Nx.bool [| 1 |] [| true |])));
    ]

(* Fitting *)

let fit =
  group "fit"
    [
      test "a fitted scale sets its domain and unsets nice and zero" (fun () ->
          equal fscale
            (Scale.linear ~name:"y" ~domain:(0., 10.) ())
            (fit_floats 0.3 9.7
               (Scale.linear ~name:"y" ~nice:true ~zero:true ())));
      test "a set domain is final" (fun () ->
          equal fscale
            (Scale.linear ~domain:(2., 3.) ())
            (fit_floats 0. 100. (Scale.linear ~domain:(2., 3.) ~nice:true ())));
      test "fitting nothing sets the default domain" (fun () ->
          equal fscale
            (Scale.log ~domain:(1., 10.) ())
            (Scale.fit None (Scale.log ()));
          equal tscale
            (Scale.time ~domain:(Time.epoch, Time.of_date (1970, 1, 2)) ())
            (Scale.fit None (Scale.time ~nice:true ())));
      test "categories are taken as observed" (fun () ->
          let c = Scale.Indices [| (1, "a"); (4, "b") |] in
          equal bscale (Scale.band ~domain:c ())
            (Scale.fit (Some (Scale.Categories c)) (Scale.band ())));
      test "fit keeps the ticks" (fun () ->
          let s = fit_floats 0. 3. (Scale.linear ~ticks:[| 1.; 7. |] ()) in
          equal (option (array float_exact)) (Some [| 1.; 7. |]) (Scale.ticks s));
      test "an observed domain that breaks the constraints is refused"
        (fun () ->
          invalid (fun () -> fit_floats 2. 1. (Scale.linear ()));
          invalid (fun () -> fit_floats Float.nan 1. (Scale.linear ()));
          invalid (fun () -> fit_floats 0. 1. (Scale.log ()));
          invalid (fun () ->
              Scale.fit
                (Some (Scale.Categories (Labels [| "a"; "a" |])))
                (Scale.band ())));
      prop "the fitted domain contains the hull" (Gen.pair gen_scale Gen.bool)
        (fun ({ spec; s; _ }, zero) ->
          let lo, hi = ends s in
          let a, b =
            ends (fit_floats lo hi (Scale.imply (Scale.linear ~zero ()) spec))
          in
          at_most ieee ~than:lo a;
          at_least ieee ~than:hi b);
      prop "fitting a fitted scale changes nothing" gen_scale
        (fun { spec; s; _ } ->
          let lo, hi = ends s in
          let once = fit_floats lo hi spec in
          equal fscale once (fit_floats 0. 1. once);
          equal fscale once (Scale.fit None once));
      prop "a nice domain is its own nice domain" gen_scale
        (fun { spec; s; _ } ->
          let lo, hi = ends s in
          let a, b = ends (fit_floats lo hi spec) in
          equal ends_w (a, b) (ends (fit_floats a b spec)));
      prop "a nice temporal domain is its own nice domain"
        (Gen.pair
           (Gen.int_range (-2_000_000_000) 2_000_000_000)
           (Gen.int_range 1 2_000_000_000))
        (fun (s0, w) ->
          let a = Time.v S (Int64.of_int s0) in
          let b = Time.add (Time.seconds 1) w a in
          let a', b' = fit_instants a b (Scale.time ()) in
          at_most instant ~than:a a';
          at_least instant ~than:b b';
          equal (pair instant instant) (a', b')
            (fit_instants a' b' (Scale.time ())));
    ]

let with_domain =
  group "with_domain"
    [
      test "with_domain sets the domain" (fun () ->
          equal fscale
            (Scale.linear ~domain:(3., 4.) ())
            (Scale.with_domain
               (Scale.Floats (3., 4.))
               (Scale.linear ~domain:(0., 1.) ())));
      test "a domain that breaks the constraints is refused" (fun () ->
          invalid (fun () ->
              Scale.with_domain (Scale.Floats (-1., 4.)) (Scale.log ())));
    ]

let pp_property =
  group "pp_property"
    [
      test "a property prints as the argument that sets it" (fun () ->
          equal (list string)
            [
              "name";
              "transform";
              "domain";
              "nice";
              "zero";
              "clamp";
              "reverse";
              "stepped";
              "ticks";
              "notation";
              "padding";
              "wrap";
              "tz_offset_s";
              "scheme";
              "areas";
              "symbols";
              "dashes";
              "unknown";
            ]
            (List.map (Format.asprintf "%a" Scale.pp_property) all_properties));
    ]

(* Comparing and formatting *)

let equal_tests =
  group "equal"
    [
      prop "equal is an equivalence"
        (Gen.map (fun (a, b) -> (scale a, scale b)) gen_pair)
        (Law.equivalence fscale);
      cases "a property set to different values differs" ~id:conflict_name
        conflicts (fun (Conflict (_, make)) ->
          let s, s' = make () in
          not_equal (Testable.make ~pp:Scale.pp ~equal:Scale.equal) s s');
      test "an unset property differs from its default set" (fun () ->
          not_equal fscale (Scale.linear ()) (Scale.linear ~nice:true ());
          not_equal fscale (Scale.linear ()) (Scale.linear ~domain:(0., 1.) ());
          not_equal fscale (Scale.linear ()) (Scale.linear ~ticks:[||] ()));
      (* [Float.equal] holds of [nan] and [nan], and of [0.] and [-0.]. *)
      test "floats compare by Float.equal" (fun () ->
          equal fscale
            (Scale.linear ~ticks:[| Float.nan |] ())
            (Scale.linear ~ticks:[| Float.nan |] ());
          equal fscale
            (Scale.linear ~ticks:[| 0. |] ())
            (Scale.linear ~ticks:[| -0. |] ()));
      test "temporal domains are equal when both ends are" (fun () ->
          let a = Time.v S 0L and b = Time.v S 10L and c = Time.v S 20L in
          not_equal tscale
            (Scale.time ~domain:(a, b) ())
            (Scale.time ~domain:(a, c) ());
          not_equal tscale
            (Scale.time ~domain:(a, c) ())
            (Scale.time ~domain:(b, c) ());
          equal tscale
            (Scale.time ~domain:(a, c) ())
            (Scale.time ~domain:(a, c) ()));
      test "ticks compare element by element" (fun () ->
          let a = Time.v S 1L in
          equal tscale
            (Scale.time ~ticks:[| a |] ())
            (Scale.time ~ticks:[| a |] ());
          not_equal tscale
            (Scale.time ~ticks:[| Time.epoch |] ())
            (Scale.time ~ticks:[| a |] ());
          equal bscale
            (Scale.band ~ticks:[| "a" |] ())
            (Scale.band ~ticks:[| "a" |] ());
          not_equal bscale
            (Scale.band ~ticks:[| "a" |] ())
            (Scale.band ~ticks:[| "a"; "b" |] ()));
      test "custom transforms compare their functions physically" (fun () ->
          let make forward =
            Scale.custom ~transform:"asinh" ~forward ~inverse:Float.sinh ()
          in
          equal fscale (asinh ()) (asinh ());
          not_equal fscale (make (fun x -> Float.asinh x)) (make Float.asinh));
    ]

let pp =
  group "pp"
    [
      test "pp formats the properties set, a custom transform by its name"
        (fun () ->
          expect (Format.asprintf "%a" Scale.pp (asinh ~areas:(1., 9.) ()))
          @@ __POS_OF__ {| (custom asinh (areas 1 9)) |});
    ]

let () =
  exit
    (run "Scale"
       [
         normalisation;
         categories;
         constructors;
         properties;
         normalize_index;
         invert;
         nice;
         sets;
         merge;
         imply;
         missing;
         fit;
         with_domain;
         pp_property;
         equal_tests;
         pp;
       ])
