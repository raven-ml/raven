(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_kit

let invalid f = raises_match (Exn.invalid_arg ?substring:None) f
let floats = array float_exact

(* Floats ordered as IEEE 754 orders them, [-0.] equal to [0.]. *)
let ieee = float Float.min_float

let fscale : float Scale.t testable =
  Testable.make ~pp:Scale.pp ~equal:Scale.equal

let tscale : Time.t Scale.t testable =
  Testable.make ~pp:Scale.pp ~equal:Scale.equal

let instant =
  Testable.with_compare Time.compare
    (Testable.make ~pp:Time.pp ~equal:Time.equal)

let property =
  Testable.make ~pp:Scale.pp_property ~equal:(fun (p : Scale.property) q ->
      p = q)

let ends s =
  let (Scale.Floats (a, b)) = Scale.domain s in
  (a, b)

let instants s =
  let (Scale.Instants (a, b)) = Scale.domain s in
  (a, b)

let fit_floats lo hi s = Scale.fit (Some (Scale.Floats (lo, hi))) s

let utc ?(ms = 0) d t =
  Time.add (Time.milliseconds 1) ms (Time.of_date_time (d, t))

let asinh =
  Scale.custom ~transform:"asinh" ~forward:Float.asinh ~inverse:Float.sinh

let ln = Scale.custom ~transform:"ln" ~forward:Float.log ~inverse:Float.exp

(* Generators *)

let order (a, b) = if a <= b then (a, b) else (b, a)

(* Domains with ends anywhere among the finite floats, subnormals and the
   extremes included, and with distinct ends. *)
let gen_domain =
  Gen.map order
    (Gen.such_that (fun (a, b) -> a <> b) (Gen.pair Gen.float Gen.float))

let gen_positive =
  Gen.map order
    (Gen.such_that
       (fun (a, b) -> a <> b)
       (Gen.pair (Gen.map Float.abs Gen.float) (Gen.map Float.abs Gen.float)))
  |> Gen.map (fun (a, b) -> if a = 0. then (Float.succ 0., b) else (a, b))
  |> Gen.such_that (fun (a, b) -> a < b)

(* Domains of moderate magnitude whose length is not tiny next to their ends. *)
let gen_moderate =
  Gen.map
    (fun (c, w) -> (c, c +. w))
    (Gen.pair (Gen.float_range (-1e6) 1e6) (Gen.float_range 1e-3 1e6))

let gen_count =
  Gen.frequency [ (4, Gen.int_range 1 20); (1, Gen.int_range 21 200) ]

(* A quantitative specification, unfitted, and the scale it gives over a domain
   drawn for its transform. *)
type q = { name : string; spec : float Scale.t; s : float Scale.t }

let gen_scale =
  let over gen name spec =
    Gen.map
      (fun (a, b) ->
        { name; spec; s = Scale.with_domain (Scale.Floats (a, b)) spec })
      gen
  in
  Gen.one_of
    [
      over gen_domain "linear" (Scale.linear ());
      over gen_domain "symlog" (Scale.symlog ());
      over gen_domain "symlog 1e-3" (Scale.symlog ~constant:1e-3 ());
      over gen_domain "pow 2" (Scale.pow ~exponent:2. ());
      over gen_domain "pow 0.5" (Scale.pow ~exponent:0.5 ());
      over gen_domain "asinh" (asinh ());
      over gen_positive "log" (Scale.log ());
      over gen_positive "log 2" (Scale.log ~base:2. ());
      over gen_positive "log e" (Scale.log ~base:(Float.exp 1.) ());
    ]
  |> Gen.with_pp (fun ppf q -> Scale.pp ppf q.s)

(* [transform name] is the transform the specification states, by the name
   [gen_scale] gives it. *)
let transform name (a, b) =
  let m = Float.max (Float.abs a) (Float.abs b) in
  let m = if m = 0. then 1. else m in
  let sym c x = Float.copy_sign (Float.log1p (Float.abs x /. c)) x in
  let pow e x = Float.copy_sign (Float.pow (Float.abs (x /. m)) e) x in
  match name with
  | "linear" -> Fun.id
  | "symlog" -> sym 1.
  | "symlog 1e-3" -> sym 1e-3
  | "pow 2" -> pow 2.
  | "pow 0.5" -> pow 0.5
  | "asinh" -> Float.asinh
  | "log" -> fun x -> Float.log x /. Float.log 10.
  | "log 2" -> fun x -> Float.log x /. Float.log 2.
  | _ -> fun x -> Float.log x /. Float.log (Float.exp 1.)

(* Constructors *)

let constructors =
  group "constructors"
    [
      test "default domains are those each constructor states" (fun () ->
          equal (pair float_exact float_exact) (0., 1.) (ends (Scale.linear ()));
          equal (pair float_exact float_exact) (1., 10.) (ends (Scale.log ()));
          equal
            (pair float_exact float_exact)
            (1., 2.)
            (ends (Scale.log ~base:2. ()));
          equal (pair float_exact float_exact) (0., 1.) (ends (Scale.symlog ()));
          equal
            (pair float_exact float_exact)
            (0., 1.)
            (ends (Scale.pow ~exponent:3. ()));
          equal (pair float_exact float_exact) (0., 1.) (ends (asinh ()));
          equal (pair instant instant)
            (Time.epoch, Time.of_date (1970, 1, 2))
            (instants (Scale.time ()));
          let (Scale.Categories c) = Scale.domain (Scale.band ()) in
          equal
            (option (array string))
            (Some [||])
            (match c with Labels l -> Some l | Indices _ -> None));
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
      test "transform parameters are checked" (fun () ->
          invalid (fun () -> Scale.log ~base:1. ());
          invalid (fun () -> Scale.log ~base:Float.infinity ());
          invalid (fun () -> Scale.log ~base:Float.nan ());
          invalid (fun () -> Scale.symlog ~constant:0. ());
          invalid (fun () -> Scale.symlog ~constant:Float.infinity ());
          invalid (fun () -> Scale.pow ~exponent:0. ());
          invalid (fun () -> Scale.pow ~exponent:(-1.) ());
          invalid (fun () -> Scale.pow ~exponent:Float.nan ()));
      test "areas must be finite and not negative" (fun () ->
          invalid (fun () -> Scale.linear ~areas:(-1., 4.) ());
          invalid (fun () -> Scale.linear ~areas:(4., -1.) ());
          is_some (Scale.areas (Scale.linear ~areas:(1., 0.) ()));
          invalid (fun () -> Scale.linear ~areas:(0., Float.nan) ());
          equal
            (option (pair float_exact float_exact))
            (Some (0., 4.))
            (Scale.areas (Scale.linear ~areas:(0., 4.) ())));
      test "an offset must be within a day of UTC" (fun () ->
          invalid (fun () -> Scale.time ~tz_offset_s:86_400 ());
          invalid (fun () -> Scale.time ~tz_offset_s:(-86_400) ()));
      test "band padding is in [0;1] and wrap at least 1" (fun () ->
          invalid (fun () -> Scale.band ~padding:(-0.1) ());
          invalid (fun () -> Scale.band ~padding:1.5 ());
          invalid (fun () -> Scale.band ~padding:Float.nan ());
          invalid (fun () -> Scale.band ~wrap:0 ());
          equal (option int) (Some 3) (Scale.wrap (Scale.band ~wrap:3 ()));
          equal (option int) (Some 1) (Scale.wrap (Scale.band ~wrap:1 ())));
      test "labels are distinct and integers strictly increasing" (fun () ->
          invalid (fun () -> Scale.band ~domain:(Labels [| "a"; "a" |]) ());
          invalid (fun () ->
              Scale.band ~domain:(Indices [| (2, "x"); (1, "y") |]) ());
          invalid (fun () ->
              Scale.band ~domain:(Indices [| (1, "x"); (1, "y") |]) ()));
      test "indexed texts may repeat" (fun () ->
          let s =
            Scale.band ~domain:(Indices [| (1, "the"); (2, "the") |]) ()
          in
          equal float_exact 0.25 (Scale.normalize s "1"));
      test "a domain array is copied in and out" (fun () ->
          let a = [| "a"; "b" |] in
          let s = Scale.band ~domain:(Labels a) () in
          a.(0) <- "z";
          equal float_exact 0.25 (Scale.normalize s "a");
          let (Scale.Categories c) = Scale.domain s in
          (match c with Labels l -> l.(0) <- "y" | Indices _ -> ());
          equal float_exact 0.25 (Scale.normalize s "a"));
      test "kinds and names" (fun () ->
          is_some
            (Scale.equal_kind (Scale.kind (Scale.linear ())) Scale.Quantitative);
          is_none (Scale.equal_kind Scale.Quantitative Scale.Temporal);
          is_some (Scale.equal_kind Scale.Quantitative Scale.Quantitative);
          is_some (Scale.equal_kind Scale.Temporal Scale.Temporal);
          is_some (Scale.equal_kind Scale.Categorical Scale.Categorical);
          equal (option string) (Some "y")
            (Scale.name (Scale.linear ~name:"y" ()));
          equal (option string) None (Scale.name (Scale.band ())));
      test "scheme is the scheme set" (fun () ->
          let scheme = Testable.make ~pp:Scheme.pp ~equal:Scheme.equal in
          equal (option scheme) (Some Scheme.viridis)
            (Scale.scheme (Scale.linear ~scheme:Scheme.viridis ()));
          equal (option scheme) (Some Scheme.okabe_ito)
            (Scale.scheme (Scale.band ~scheme:Scheme.okabe_ito ()));
          is_none (Scale.scheme (Scale.time ())));
      test "symbols are not empty and are copied in and out" (fun () ->
          let symbols =
            option (array (Testable.make ~pp:Symbol.pp ~equal:Symbol.equal))
          in
          invalid (fun () -> Scale.band ~symbols:[||] ());
          let a = [| Symbol.circle; Symbol.square |] in
          let s = Scale.band ~symbols:a () in
          a.(0) <- Symbol.star;
          equal symbols
            (Some [| Symbol.circle; Symbol.square |])
            (Scale.symbols s);
          (match Scale.symbols s with
          | Some a -> a.(0) <- Symbol.star
          | None -> ());
          equal symbols
            (Some [| Symbol.circle; Symbol.square |])
            (Scale.symbols s);
          is_none (Scale.symbols (Scale.band ())));
      test "unknown is the colour set" (fun () ->
          equal
            (option
               (Testable.make ~pp:Hugin_next_gg.Color.pp
                  ~equal:Hugin_next_gg.Color.equal))
            (Some Hugin_next_gg.Color.red)
            (Scale.unknown (Scale.band ~unknown:Hugin_next_gg.Color.red ()));
          is_none (Scale.unknown (Scale.linear ())));
    ]

(* Normalisation *)

let ends_law =
  prop "a domain's ends normalise to 0 and 1, or 0.5 when it is constant"
    gen_scale (fun { name; s; _ } ->
      let a, b = ends s in
      let t = transform name (a, b) in
      if Float.equal (t a) (t b) then begin
        equal float_exact 0.5 (Scale.normalize s a);
        equal float_exact 0.5 (Scale.normalize s b)
      end
      else begin
        equal float_exact 0. (Scale.normalize s a);
        equal float_exact 1. (Scale.normalize s b)
      end)

let monotone_law =
  prop "normalisation is monotone over the domain and beyond it"
    (Gen.triple gen_scale Gen.float Gen.float) (fun ({ s; _ }, x, y) ->
      let x, y = order (x, y) in
      let u = Scale.normalize s x and v = Scale.normalize s y in
      if not (Float.is_nan u || Float.is_nan v) then
        at_most float_exact ~than:v u)

let reverse_law =
  prop "a reversed scale normalises to 1 - N x" (Gen.pair gen_domain Gen.float)
    (fun (d, x) ->
      let s = Scale.linear ~domain:d () in
      let r = Scale.linear ~reverse:true ~domain:d () in
      equal float_exact (1. -. Scale.normalize s x) (Scale.normalize r x))

let clamp_law =
  prop "a clamped scale normalises into [0;1]"
    (Gen.pair gen_scale Gen.any_float) (fun ({ s; _ }, x) ->
      let clamped = Scale.imply (Scale.linear ~clamp:true ()) s in
      let u = Scale.normalize clamped x in
      if not (Float.is_nan u) then begin
        at_least float_exact ~than:0. u;
        at_most float_exact ~than:1. u
      end)

let missing_law =
  prop "a log scale normalises to nan exactly the values that are not positive"
    (Gen.pair gen_positive Gen.any_float) (fun (d, x) ->
      let u = Scale.normalize (Scale.log ~domain:d ()) x in
      equal bool (Float.is_finite x && x > 0.) (not (Float.is_nan u)))

let normalisation =
  group "normalisation"
    [
      ends_law;
      missing_law;
      monotone_law;
      reverse_law;
      clamp_law;
      test "a clamped value just below the domain is positive zero" (fun () ->
          let s = Scale.linear ~clamp:true ~domain:(0., 2.) () in
          equal float_exact 0. (Scale.normalize s (-5e-324)));
      test "scales extrapolate beyond their domain" (fun () ->
          let s = Scale.linear ~domain:(0., 10.) () in
          equal float_exact (-0.5) (Scale.normalize s (-5.));
          equal float_exact 2. (Scale.normalize s 20.));
      test "ends normalise exactly on the widest domain" (fun () ->
          let s =
            Scale.linear ~domain:(-.Float.max_float, Float.max_float) ()
          in
          equal float_exact 0. (Scale.normalize s (-.Float.max_float));
          equal float_exact 1. (Scale.normalize s Float.max_float);
          equal float_exact 0.5 (Scale.normalize s 0.));
      test "extrapolation beyond the floats' range does not overflow" (fun () ->
          let s = Scale.linear ~domain:(1e308, Float.max_float) () in
          let w = (Float.max_float /. 2.) -. 5e307 in
          equal (float 1e-12)
            (((-.Float.max_float /. 2.) -. 5e307) /. w)
            (Scale.normalize s (-.Float.max_float)));
      test "a constant domain normalises to 0.5" (fun () ->
          let s = Scale.linear ~domain:(3., 3.) () in
          equal float_exact 0.5 (Scale.normalize s 3.);
          equal float_exact 0.5 (Scale.normalize s 1e9);
          let s = Scale.log ~domain:(1e308, 1.0000000000000002e308) () in
          equal float_exact 0.5 (Scale.normalize s 1e308));
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
      test "values that are not positive are missing on a log scale" (fun () ->
          let s = Scale.log ~domain:(1., 100.) () in
          equal float_exact Float.nan (Scale.normalize s 0.);
          equal float_exact Float.nan (Scale.normalize s (-0.));
          equal float_exact Float.nan (Scale.normalize s (-10.));
          equal float_exact 0.5 (Scale.normalize s 10.));
      test "a custom transform is missing where it is not finite" (fun () ->
          let s = ln ~domain:(1., Float.exp 2.) () in
          equal float_exact Float.nan (Scale.normalize s 0.);
          equal float_exact Float.nan (Scale.normalize s (-1.));
          equal (float 1e-15) 0.5 (Scale.normalize s (Float.exp 1.)));
      test "an unfitted logit normalises everything to nan" (fun () ->
          let logit =
            Scale.custom ~transform:"logit"
              ~forward:(fun p -> Float.log (p /. (1. -. p)))
              ~inverse:(fun x -> 1. /. (1. +. Float.exp (-.x)))
              ()
          in
          equal float_exact Float.nan (Scale.normalize logit 0.5));
      test "symlog takes the logarithm of values too large to divide" (fun () ->
          let s = Scale.symlog ~constant:1e-300 ~domain:(-1e300, 1e300) () in
          equal (float 1e-12) (1199. /. 1200.) (Scale.normalize s 1e299));
      test "symlog is smooth through zero" (fun () ->
          let s = Scale.symlog ~domain:(-1., 1.) () in
          equal float_exact 0.5 (Scale.normalize s 0.);
          equal float_exact
            (0.5 +. (0.5 *. Float.log1p 0.5 /. Float.log 2.))
            (Scale.normalize s 0.5));
      test "pow normalises as the power of the values" (fun () ->
          let s = Scale.pow ~exponent:2. ~domain:(0., 4.) () in
          equal float_exact 0.25 (Scale.normalize s 2.);
          let s = Scale.pow ~exponent:2. ~domain:(0., 1e200) () in
          equal float_exact 0.25 (Scale.normalize s 5e199));
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
          equal float_exact (1. /. 3.) (Scale.normalize s (Time.v Ns 8L));
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
      test "band centres follow the padding" (fun () ->
          let s = Scale.band ~domain:(Labels [| "a"; "b"; "c" |]) () in
          equal float_exact (0.5 /. 3.) (Scale.normalize s "a");
          equal float_exact 0.5 (Scale.normalize s "b");
          equal float_exact (2.5 /. 3.) (Scale.normalize s "c");
          equal float_exact (1. /. 3.) (Scale.bandwidth s);
          let s =
            Scale.band ~padding:1. ~domain:(Labels [| "a"; "b"; "c" |]) ()
          in
          equal float_exact 0.25 (Scale.normalize s "a");
          equal float_exact 0.75 (Scale.normalize s "c");
          equal float_exact 0. (Scale.bandwidth s);
          let s =
            Scale.band ~padding:0.2 ~domain:(Labels [| "a"; "b"; "c"; "d" |]) ()
          in
          equal float_exact (0.8 /. 4.2) (Scale.bandwidth s));
      test "a reversed band runs from the last category" (fun () ->
          let s =
            Scale.band ~reverse:true ~domain:(Labels [| "a"; "b"; "c" |]) ()
          in
          equal float_exact (1. -. (0.5 /. 3.)) (Scale.normalize s "a"));
      test "categories off the domain are missing" (fun () ->
          let s =
            Scale.band ~domain:(Indices [| (3, "the"); (7, "cat") |]) ()
          in
          equal float_exact 0.25 (Scale.normalize s "3");
          equal float_exact Float.nan (Scale.normalize s "the");
          equal float_exact Float.nan (Scale.normalize s "03");
          equal float_exact 0. (Scale.bandwidth (Scale.band ())));
    ]

(* Inversion *)

let inversion =
  group "inversion"
    [
      prop "a linear scale inverts its normalisation up to rounding"
        (Gen.pair gen_moderate (Gen.float_range (-2e6) 2e6))
        (fun ((a, b), x) ->
          let s = Scale.linear ~domain:(a, b) () in
          (* A few roundings of each term of [(1 - u) a + u b]. *)
          let u = Scale.normalize s x in
          let tol =
            8. *. Float.epsilon
            *. (Float.abs x +. (Float.abs u *. (Float.abs a +. Float.abs b)))
          in
          let tol = Float.max tol Float.min_float in
          equal
            (option (float tol))
            (Some x)
            (Scale.invert s (Scale.normalize s x)));
      prop "a log scale inverts its normalisation up to rounding"
        (Gen.pair (Gen.float_range 1e-3 1e3) (Gen.float_range 1e-2 1e2))
        (fun (a, x) ->
          let s = Scale.log ~domain:(a, a *. 1e3) () in
          equal
            (option (float_rel ~rel:1e-12 ~abs:0.))
            (Some x)
            (Scale.invert s (Scale.normalize s x)));
      prop "symlog, pow and custom scales normalise their inverses back"
        (Gen.triple
           (Gen.of_list [ "symlog"; "pow 2"; "pow 0.5"; "asinh" ])
           gen_moderate
           (Gen.float_range (-0.5) 1.5))
        (fun (name, (a, b), u) ->
          let s =
            match name with
            | "symlog" -> Scale.symlog ~domain:(a, b) ()
            | "pow 2" -> Scale.pow ~exponent:2. ~domain:(a, b) ()
            | "pow 0.5" -> Scale.pow ~exponent:0.5 ~domain:(a, b) ()
            | _ -> asinh ~domain:(a, b) ()
          in
          let x = require_some (Scale.invert s u) in
          (* A few roundings of transforms as large as the ends', against their
             difference. *)
          let t = transform name (a, b) in
          let tol =
            64. *. Float.epsilon
            *. (Float.abs (t a) +. Float.abs (t b) +. Float.abs (t x))
            /. Float.abs (t b -. t a)
          in
          equal (float (Float.max tol 1e-15)) u (Scale.normalize s x));
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
          equal (option instant) (Some a) (Scale.invert s 0.);
          equal (option instant) None (Scale.invert s 1.5));
      test "a value that is not finite inverts to nothing" (fun () ->
          let s = Scale.linear () in
          equal (option float_exact) None (Scale.invert s Float.nan);
          equal (option float_exact) None (Scale.invert s Float.infinity);
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
          equal (option string) None (Scale.invert s 0.05);
          equal (option string) (Some "a") (Scale.invert s 0.1);
          equal (option string) (Some "a") (Scale.invert s 0.5);
          equal (option string) (Some "b") (Scale.invert s 0.51);
          equal (option string) (Some "b") (Scale.invert s 0.9);
          equal (option string) None (Scale.invert s 0.95);
          equal (option string) None (Scale.invert s 1.5);
          equal (option string) None (Scale.invert (Scale.band ()) 0.5));
      test "an indexed band inverts to the decimal name" (fun () ->
          let s =
            Scale.band ~domain:(Indices [| (3, "the"); (7, "cat") |]) ()
          in
          equal (option string) (Some "7") (Scale.invert s 0.75));
    ]

(* Guide values *)

let ticks ?(count = 10) s = Scale.ticks ~count s
let linear_ticks count (a, b) = ticks ~count (Scale.linear ~domain:(a, b) ())

(* Rows [(a, b, count, expected)] from d3-array's ticks-test.js. *)
let d3_linear =
  [
    (0., 1., 10, [| 0.0; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1.0 |]);
    (0., 1., 9, [| 0.0; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1.0 |]);
    (0., 1., 8, [| 0.0; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1.0 |]);
    (0., 1., 7, [| 0.0; 0.2; 0.4; 0.6; 0.8; 1.0 |]);
    (0., 1., 4, [| 0.0; 0.2; 0.4; 0.6; 0.8; 1.0 |]);
    (0., 1., 3, [| 0.0; 0.5; 1.0 |]);
    (0., 1., 2, [| 0.0; 0.5; 1.0 |]);
    (0., 1., 1, [| 0.0; 1.0 |]);
    (0., 10., 10, [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10. |]);
    (0., 10., 7, [| 0.; 2.; 4.; 6.; 8.; 10. |]);
    (0., 10., 3, [| 0.; 5.; 10. |]);
    (0., 10., 1, [| 0.; 10. |]);
    (-10., 10., 10, [| -10.; -8.; -6.; -4.; -2.; 0.; 2.; 4.; 6.; 8.; 10. |]);
    (-10., 10., 6, [| -10.; -5.; 0.; 5.; 10. |]);
    (-10., 10., 2, [| -10.; 0.; 10. |]);
    (-10., 10., 1, [| 0. |]);
    (0., 2.2, 3, [| 0.; 1.; 2. |]);
    (0.98, 1.14, 10, [| 0.98; 1.; 1.02; 1.04; 1.06; 1.08; 1.1; 1.12; 1.14 |]);
    (1., 364., 1, [| 200. |]);
    (1., 499., 1, [| 200.; 400. |]);
    ( -100.,
      100.,
      10,
      [| -100.; -80.; -60.; -40.; -20.; 0.; 20.; 40.; 60.; 80.; 100. |] );
    (-100., 100., 5, [| -100.; -50.; 0.; 50.; 100. |]);
  ]

(* Rows [(base, a, b, count, expected)] from d3-scale's log-test.js. *)
let d3_log =
  [
    (10., 0.15, 0.68, 10, [| 0.2; 0.3; 0.4; 0.5; 0.6 |]);
    ( 10.,
      0.1,
      10.,
      10,
      [|
        0.1;
        0.2;
        0.3;
        0.4;
        0.5;
        0.6;
        0.7;
        0.8;
        0.9;
        1.;
        2.;
        3.;
        4.;
        5.;
        6.;
        7.;
        8.;
        9.;
        10.;
      |] );
    (10., 1., 5., 10, [| 1.; 2.; 3.; 4.; 5. |]);
    (10., 286.9252014, 329.4978332, 1, [| 300. |]);
    (10., 286.9252014, 329.4978332, 2, [| 300. |]);
    (10., 286.9252014, 329.4978332, 3, [| 300.; 320. |]);
    (10., 286.9252014, 329.4978332, 4, [| 290.; 300.; 310.; 320. |]);
    ( 10.,
      286.9252014,
      329.4978332,
      10,
      [| 290.; 295.; 300.; 305.; 310.; 315.; 320.; 325. |] );
    ( 10.,
      41.,
      42.,
      10,
      [| 41.; 41.1; 41.2; 41.3; 41.4; 41.5; 41.6; 41.7; 41.8; 41.9; 42. |] );
    ( 10.,
      1400.,
      1600.,
      10,
      [|
        1400.;
        1420.;
        1440.;
        1460.;
        1480.;
        1500.;
        1520.;
        1540.;
        1560.;
        1580.;
        1600.;
      |] );
    (2., 1., 32., 10, [| 1.; 2.; 4.; 8.; 16.; 32. |]);
    ( 10.,
      1.,
      1e10,
      10,
      Array.init 11 (fun i -> Float.of_string ("1e" ^ string_of_int i)) );
    ( 10.,
      1e-29,
      1e-1,
      10,
      Array.init 14 (fun i ->
          Float.of_string ("1e" ^ string_of_int (-28 + (2 * i)))) );
  ]

let d3_tables =
  group "d3"
    [
      cases
        ~name:(fun (a, b, c, _) -> Printf.sprintf "linear [%g;%g] for %d" a b c)
        "linear" d3_linear
        (fun (a, b, count, e) -> equal floats e (linear_ticks count (a, b)));
      cases
        ~name:(fun (base, a, b, c, _) ->
          Printf.sprintf "log %g [%g;%g] for %d" base a b c)
        "log" d3_log
        (fun (base, a, b, count, e) ->
          equal floats e (ticks ~count (Scale.log ~base ~domain:(a, b) ())));
      test "log in base e takes the powers of e" (fun () ->
          let e = Float.exp 1. in
          equal floats
            [| 1.; e; Float.pow e 2.; Float.pow e 3. |]
            (ticks ~count:6 (Scale.log ~base:e ~domain:(1., 32.) ()));
          equal int 7
            (Array.length (ticks (Scale.log ~base:e ~domain:(0.1, 100.) ()))));
      test "log ticks cover a domain of six decades in multiples" (fun () ->
          let t = ticks ~count:20 (Scale.log ~domain:(0.01, 10000.) ()) in
          equal int 55 (Array.length t);
          equal float_exact 0.03 t.(2);
          equal float_exact 10000. t.(54));
      test "a tick at zero is positive zero" (fun () ->
          let t = linear_ticks 5 (-1., 0.) in
          equal float_exact Float.infinity (1. /. t.(Array.length t - 1)));
    ]

let time_ticks count (a, b) = Scale.ticks ~count (Scale.time ~domain:(a, b) ())

(* Rows from d3-scale's utcTime-test.js, d3's months counted from 1. Weeks start
   on Monday and two-day steps on even days since the epoch, where d3 counts
   from Sunday and from the first of the month. *)
let d3_time =
  let at d t = utc d t in
  [
    ( "sub-second",
      4,
      (at (2011, 1, 1) (12, 0, 0), at (2011, 1, 1) (12, 0, 1)),
      List.map
        (fun ms -> utc ~ms (2011, 1, 1) (12, 0, 0))
        [ 0; 200; 400; 600; 800; 1000 ] );
    ( "1-second",
      4,
      (at (2011, 1, 1) (12, 0, 0), at (2011, 1, 1) (12, 0, 4)),
      List.map (fun s -> at (2011, 1, 1) (12, 0, s)) [ 0; 1; 2; 3; 4 ] );
    ( "5-second",
      4,
      (at (2011, 1, 1) (12, 0, 0), at (2011, 1, 1) (12, 0, 20)),
      List.map (fun s -> at (2011, 1, 1) (12, 0, s)) [ 0; 5; 10; 15; 20 ] );
    ( "15-second",
      4,
      (at (2011, 1, 1) (12, 0, 0), at (2011, 1, 1) (12, 0, 50)),
      List.map (fun s -> at (2011, 1, 1) (12, 0, s)) [ 0; 15; 30; 45 ] );
    ( "30-second",
      4,
      (at (2011, 1, 1) (12, 0, 0), at (2011, 1, 1) (12, 1, 50)),
      [
        at (2011, 1, 1) (12, 0, 0);
        at (2011, 1, 1) (12, 0, 30);
        at (2011, 1, 1) (12, 1, 0);
        at (2011, 1, 1) (12, 1, 30);
      ] );
    ( "1-minute",
      4,
      (at (2011, 1, 1) (12, 0, 27), at (2011, 1, 1) (12, 4, 12)),
      List.map (fun m -> at (2011, 1, 1) (12, m, 0)) [ 1; 2; 3; 4 ] );
    ( "5-minute",
      4,
      (at (2011, 1, 1) (12, 3, 27), at (2011, 1, 1) (12, 21, 12)),
      List.map (fun m -> at (2011, 1, 1) (12, m, 0)) [ 5; 10; 15; 20 ] );
    ( "15-minute",
      4,
      (at (2011, 1, 1) (12, 8, 27), at (2011, 1, 1) (13, 4, 12)),
      [
        at (2011, 1, 1) (12, 15, 0);
        at (2011, 1, 1) (12, 30, 0);
        at (2011, 1, 1) (12, 45, 0);
        at (2011, 1, 1) (13, 0, 0);
      ] );
    ( "30-minute",
      4,
      (at (2011, 1, 1) (12, 28, 27), at (2011, 1, 1) (14, 4, 12)),
      [
        at (2011, 1, 1) (12, 30, 0);
        at (2011, 1, 1) (13, 0, 0);
        at (2011, 1, 1) (13, 30, 0);
        at (2011, 1, 1) (14, 0, 0);
      ] );
    ( "1-hour",
      4,
      (at (2011, 1, 1) (12, 28, 27), at (2011, 1, 1) (16, 34, 12)),
      List.map (fun h -> at (2011, 1, 1) (h, 0, 0)) [ 13; 14; 15; 16 ] );
    ( "3-hour",
      4,
      (at (2011, 1, 1) (14, 28, 27), at (2011, 1, 2) (1, 34, 12)),
      [
        at (2011, 1, 1) (15, 0, 0);
        at (2011, 1, 1) (18, 0, 0);
        at (2011, 1, 1) (21, 0, 0);
        at (2011, 1, 2) (0, 0, 0);
      ] );
    ( "6-hour",
      4,
      (at (2011, 1, 1) (16, 28, 27), at (2011, 1, 2) (14, 34, 12)),
      [
        at (2011, 1, 1) (18, 0, 0);
        at (2011, 1, 2) (0, 0, 0);
        at (2011, 1, 2) (6, 0, 0);
        at (2011, 1, 2) (12, 0, 0);
      ] );
    ( "12-hour",
      4,
      (at (2011, 1, 1) (16, 28, 27), at (2011, 1, 3) (21, 34, 12)),
      [
        at (2011, 1, 2) (0, 0, 0);
        at (2011, 1, 2) (12, 0, 0);
        at (2011, 1, 3) (0, 0, 0);
        at (2011, 1, 3) (12, 0, 0);
      ] );
    ( "1-day",
      4,
      (at (2011, 1, 1) (16, 28, 27), at (2011, 1, 5) (21, 34, 12)),
      List.map (fun d -> at (2011, 1, d) (0, 0, 0)) [ 2; 3; 4; 5 ] );
    ( "2-day, on even days",
      4,
      (at (2011, 1, 2) (16, 28, 27), at (2011, 1, 9) (21, 34, 12)),
      List.map (fun d -> at (2011, 1, d) (0, 0, 0)) [ 4; 6; 8 ] );
    ( "1-week, on Mondays",
      4,
      (at (2011, 1, 1) (16, 28, 27), at (2011, 1, 23) (21, 34, 12)),
      List.map (fun d -> at (2011, 1, d) (0, 0, 0)) [ 3; 10; 17 ] );
    ( "1-month",
      4,
      (at (2011, 1, 18) (0, 0, 0), at (2011, 5, 2) (0, 0, 0)),
      List.map (fun m -> at (2011, m, 1) (0, 0, 0)) [ 2; 3; 4; 5 ] );
    ( "3-month",
      4,
      (at (2010, 12, 18) (0, 0, 0), at (2011, 11, 2) (0, 0, 0)),
      List.map (fun m -> at (2011, m, 1) (0, 0, 0)) [ 1; 4; 7; 10 ] );
    ( "1-year",
      4,
      (at (2010, 12, 18) (0, 0, 0), at (2014, 3, 2) (0, 0, 0)),
      List.map (fun y -> at (y, 1, 1) (0, 0, 0)) [ 2011; 2012; 2013; 2014 ] );
    ( "multi-year",
      6,
      (at (0, 12, 18) (0, 0, 0), at (2014, 3, 2) (0, 0, 0)),
      List.map (fun y -> at (y, 1, 1) (0, 0, 0)) [ 500; 1000; 1500; 2000 ] );
  ]

let time_tables =
  group "time"
    [
      cases
        ~name:(fun (n, _, _, _) -> n)
        "d3" d3_time
        (fun (_, count, d, e) ->
          equal (array instant) (Array.of_list e) (time_ticks count d));
      test "an empty domain has one tick" (fun () ->
          let t = utc (2014, 3, 2) (0, 0, 0) in
          equal (array instant) [| t |] (time_ticks 6 (t, t)));
      test "a count of one with no boundary takes the next finer interval"
        (fun () ->
          let a = Time.v S 788918401L in
          let b = Time.add (Time.seconds 1) 99725589 a in
          equal (array instant)
            [| Time.of_date (1996, 1, 1); Time.of_date (1998, 1, 1) |]
            (time_ticks 1 (a, b)));
      test "boundaries follow the scale's offset" (fun () ->
          let tz_offset_s = 3600 in
          let a = Time.of_date ~tz_offset_s (2026, 1, 1)
          and b = Time.of_date ~tz_offset_s (2026, 1, 5) in
          equal (array instant)
            (Array.init 5 (fun d -> Time.of_date ~tz_offset_s (2026, 1, 1 + d)))
            (Scale.ticks ~count:4 (Scale.time ~tz_offset_s ~domain:(a, b) ())));
    ]

let band_ticks count n =
  Scale.ticks ~count
    (Scale.band ~domain:(Labels (Array.init n string_of_int)) ())

let guide_laws =
  let inside_and_increasing (a, b) t =
    Array.iteri
      (fun i v ->
        at_least ieee ~than:a v;
        at_most ieee ~than:b v;
        if i > 0 then less ieee ~than:v t.(i - 1))
      t
  in
  group "laws"
    [
      prop "guide values are inside the domain and strictly increasing"
        (Gen.pair gen_scale gen_count) (fun ({ s; _ }, count) ->
          inside_and_increasing (ends s) (Scale.ticks ~count s));
      prop "a linear domain has between 1 and 2 count + 1 guide values"
        (Gen.pair gen_domain gen_count) (fun (d, count) ->
          let n = Array.length (linear_ticks count d) in
          at_least int ~than:1 n;
          at_most int ~than:((2 * count) + 1) n);
      prop "decimal guide values are the floats nearest short decimals"
        (Gen.pair gen_moderate (Gen.int_range 1 50))
        (fun (d, count) ->
          Array.iter
            (fun v ->
              equal float_exact v (float_of_string (Printf.sprintf "%.15g" v)))
            (linear_ticks count d));
      prop "symlog guide values are at most max count 3 signed powers"
        (Gen.pair gen_domain gen_count) (fun (d, count) ->
          let t = Scale.ticks ~count (Scale.symlog ~domain:d ()) in
          let signed v =
            v = 0.
            || Float.equal (Float.abs v)
                 (Float.of_string
                    (Printf.sprintf "1e%d"
                       (Float.to_int (Float.round (Float.log10 (Float.abs v))))))
          in
          if Array.length t >= 2 && Array.for_all signed t then
            at_most int ~than:(Int.max count 3) (Array.length t)
          else equal floats (linear_ticks count d) t);
      prop "temporal guide values are inside the domain and increasing"
        (Gen.triple
           (Gen.int_range (-2_000_000_000) 2_000_000_000)
           (Gen.int_range 1 2_000_000_000)
           gen_count)
        (fun (s0, w, count) ->
          let a = Time.v S (Int64.of_int s0) in
          let b = Time.add (Time.seconds 1) w a in
          let t = time_ticks count (a, b) in
          at_least int ~than:1 (Array.length t);
          Array.iteri
            (fun i v ->
              at_least instant ~than:a v;
              at_most instant ~than:b v;
              if i > 0 then less instant ~than:v t.(i - 1))
            t);
      prop "band guide values are every kth category, at most count"
        (Gen.pair (Gen.int_range 0 500) gen_count)
        (fun (n, count) ->
          let t = band_ticks count n in
          if n > 0 then begin
            at_most int ~than:count (Array.length t);
            equal string "0" t.(0);
            if Array.length t > 1 then begin
              let k = int_of_string t.(1) in
              is_true ~msg:(string_of_int k)
                (List.mem k [ 1; 2; 5; 10; 20; 50; 100; 200; 500 ]);
              Array.iteri (fun i c -> equal string (string_of_int (i * k)) c) t
            end
          end
          else equal (array string) [||] t);
    ]

let guide_cases =
  group "values"
    [
      test "the 1e17 domain whose step is below the float spacing ends"
        (fun () ->
          equal floats
            [| 1e17; 1e17 +. 16.; 1e17 +. 32. |]
            (linear_ticks 10 (1e17, 1e17 +. 32.)));
      test "a domain of the least subnormal ends" (fun () ->
          equal floats [| 0.; 5e-324 |] (linear_ticks 10 (0., 5e-324)));
      test "a multiple that underflows is positive zero" (fun () ->
          equal floats [| 0.; 5e-324 |] (linear_ticks 10 (-0., 5e-324)));
      test "the widest domain has finite guide values" (fun () ->
          equal floats
            [| -1.5e308; -1e308; -5e307; 0.; 5e307; 1e308; 1.5e308 |]
            (linear_ticks 10 (-.Float.max_float, Float.max_float)));
      test "a step finer than the floats gives every float" (fun () ->
          let a = 1. and b = Float.succ (Float.succ 1.) in
          equal floats [| a; Float.succ 1.; b |] (linear_ticks 1000 (a, b)));
      test "a single tick reaches for a count of two" (fun () ->
          equal floats [| 0.15 |] (linear_ticks 1 (0.11, 0.19)));
      test "a constant domain has its one value" (fun () ->
          equal floats [| 3. |] (linear_ticks 10 (3., 3.));
          equal floats [| 1e308 |]
            (ticks (Scale.log ~domain:(1e308, 1.0000000000000002e308) ())));
      test "a log domain within a decade has guide values" (fun () ->
          equal floats
            [| 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
            (ticks (Scale.log ~domain:(2., 8.) ())));
      test "a log domain falls back to linear values in every base" (fun () ->
          equal floats
            [| 1.6; 1.8; 2.; 2.2; 2.4 |]
            (ticks ~count:4
               (Scale.log ~base:(Float.exp 1.) ~domain:(1.5, 2.5) ())));
      test "a large integer base follows the rule of other bases" (fun () ->
          equal floats [| 1.; 1000.; 1e6 |]
            (ticks ~count:3 (Scale.log ~base:1000. ~domain:(1., 1e6) ())));
      test "base 16 takes multiples of its powers" (fun () ->
          let t = ticks (Scale.log ~base:16. ~domain:(1., 256.) ()) in
          equal int 31 (Array.length t);
          equal float_exact 15. t.(14);
          equal float_exact 240. t.(29));
      test "log values reach the greatest and least powers of ten" (fun () ->
          let t = ticks (Scale.log ~domain:(1e300, 1e308) ()) in
          equal float_exact 1e308 t.(Array.length t - 1);
          let t = ticks (Scale.log ~domain:(1e-323, 1e-322) ()) in
          equal float_exact 1e-323 t.(0);
          equal float_exact 1e-322 t.(Array.length t - 1));
      test "multiples are the floats nearest their decimals" (fun () ->
          equal floats [| -1e308; 0.; 1e308 |]
            (linear_ticks 3 (-.Float.max_float, Float.max_float));
          equal floats
            (Array.init 11 (fun i ->
                 float_of_string (Printf.sprintf "%de-308" i)))
            (linear_ticks 10 (0., 1e-307));
          equal floats
            (Array.init 10 (fun i ->
                 float_of_string (Printf.sprintf "%de-300" (i + 1))))
            (linear_ticks 10 (1e-300, 1e-299));
          equal floats
            [| 1.7e15; 1.7e15 +. 0.25; 1.7e15 +. 0.5 |]
            (linear_ticks 5 (1.7e15, 1.7e15 +. 0.5)));
      test "a multiple halfway between two floats rounds to even" (fun () ->
          (* 10^23 is an odd multiple of 2^23 and the floats there are 2^24
             apart. *)
          equal floats [| 0.; 1e23; 2e23 |] (linear_ticks 2 (0., 2e23)));
      test "a multiple beyond 10^22 is the float nearest it" (fun () ->
          (* 3 × 10^23 is not 3 times the float nearest 10^23. *)
          equal floats [| 0.; 1e23; 2e23; 3e23 |] (linear_ticks 3 (0., 3e23)));
      test "the least power of ten is a symlog value" (fun () ->
          equal floats [| 0.; 1e-323; 1e-322 |]
            (ticks (Scale.symlog ~constant:1e-323 ~domain:(0., 1e-322) ())));
      test "a step chosen at each threshold is the larger" (fun () ->
          equal floats [| 0. |] (linear_ticks 1 (0., Float.sqrt 50.));
          equal floats [| 0. |] (linear_ticks 1 (0., Float.sqrt 10.));
          equal floats [| 0. |] (linear_ticks 1 (0., Float.sqrt 2.)));
      test "symlog values are signed powers and zero" (fun () ->
          equal floats
            [| -1000.; -100.; -10.; -1.; 0.; 1.; 10.; 100.; 1000. |]
            (ticks (Scale.symlog ~domain:(-1000., 1000.) ())));
      test "symlog values keep the fewest when none fits the count" (fun () ->
          equal floats [| -1.; 0.; 1. |]
            (ticks ~count:2 (Scale.symlog ~domain:(-10., 10.) ())));
      test "symlog skips exponents to fit the count" (fun () ->
          equal floats [| 1.; 100.; 10000. |]
            (ticks ~count:3 (Scale.symlog ~domain:(1., 10000.) ())));
      test "symlog values include zero at an end of the domain" (fun () ->
          equal floats [| 0.; 1.; 10.; 100. |]
            (ticks (Scale.symlog ~domain:(0., 100.) ()));
          equal floats [| -100.; -10.; -1.; 0. |]
            (ticks (Scale.symlog ~domain:(-100., 0.) ())));
      test "two symlog values are kept" (fun () ->
          equal floats [| 1.; 10. |] (ticks (Scale.symlog ~domain:(1., 10.) ())));
      test "symlog without powers falls back to linear values" (fun () ->
          equal floats [| 0.; 1.; 2.; 3. |]
            (ticks ~count:3 (Scale.symlog ~constant:5. ~domain:(0., 3.) ())));
      test "band strides" (fun () ->
          equal (array string) [| "0"; "5" |] (band_ticks 4 10);
          equal (array string) (Array.init 10 string_of_int) (band_ticks 10 10);
          equal (array string) [||] (band_ticks 3 0));
      test "indexed bands give their decimal names" (fun () ->
          equal (array string) [| "3"; "7" |]
            (Scale.ticks ~count:5
               (Scale.band ~domain:(Indices [| (3, "the"); (7, "cat") |]) ())));
      test "a count below 1 is refused" (fun () ->
          invalid (fun () -> Scale.ticks ~count:0 (Scale.linear ())));
    ]

(* Specifications and fitting *)

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

let ends_w = pair float_exact float_exact

let nice =
  group "nice"
    [
      cases
        ~name:(fun (a, b, _, _) -> Printf.sprintf "linear [%g;%g]" a b)
        "d3 linear" d3_nice
        (fun (a, b, a', b') ->
          equal ends_w (a', b') (ends (fit_floats a b (Scale.linear ()))));
      cases
        ~name:(fun (a, b, _, _) -> Printf.sprintf "log [%g;%g]" a b)
        "d3 log" d3_log_nice
        (fun (a, b, a', b') ->
          equal ends_w (a', b') (ends (fit_floats a b (Scale.log ()))));
      test "d3's time domains" (fun () ->
          let nice a b =
            instants (Scale.fit (Some (Scale.Instants (a, b))) (Scale.time ()))
          in
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
            (nice (utc (2001, 1, 1) (0, 0, 0)) (utc (2138, 1, 1) (0, 0, 0)));
          let t = utc (2009, 1, 1) (0, 12, 0) in
          equal w (t, t) (nice t t));
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
      test "a constant domain is not rounded" (fun () ->
          equal ends_w (0.5, 0.5) (ends (fit_floats 0.5 0.5 (Scale.linear ())));
          equal ends_w (0.5, 0.5) (ends (fit_floats 0.5 0.5 (Scale.log ()))));
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
          let fitted =
            Scale.fit (Some (Scale.Instants (a, b))) (Scale.time ())
          in
          equal instant b (snd (instants fitted)));
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
          let s = Scale.fit (Some (Scale.Categories c)) (Scale.band ()) in
          equal float_exact 0.75 (Scale.normalize s "4"));
      test "an observed domain that breaks the constraints is refused"
        (fun () ->
          invalid (fun () -> fit_floats 2. 1. (Scale.linear ()));
          invalid (fun () -> fit_floats Float.nan 1. (Scale.linear ()));
          invalid (fun () -> fit_floats 0. 1. (Scale.log ()));
          invalid (fun () ->
              Scale.fit
                (Some (Scale.Categories (Labels [| "a"; "a" |])))
                (Scale.band ())));
      test "with_domain sets the domain or refuses it" (fun () ->
          let s =
            Scale.with_domain
              (Scale.Floats (3., 4.))
              (Scale.linear ~domain:(0., 1.) ())
          in
          equal ends_w (3., 4.) (ends s);
          invalid (fun () ->
              Scale.with_domain (Scale.Floats (-1., 4.)) (Scale.log ())));
      prop "the fitted domain contains the hull" (Gen.pair gen_scale Gen.bool)
        (fun ({ spec; s; _ }, zero) ->
          let lo, hi = ends s in
          let a, b =
            ends (fit_floats lo hi (Scale.imply (Scale.linear ~zero ()) spec))
          in
          at_most ieee ~than:lo a;
          at_least ieee ~than:hi b);
      prop "fitting is a fixed point" gen_scale (fun { spec; s; _ } ->
          let lo, hi = ends s in
          let once = fit_floats lo hi spec in
          equal fscale once (fit_floats 0. 1. once);
          equal fscale once (Scale.fit None once));
      prop "a nice domain is its own nice domain" gen_scale
        (fun { spec; s; _ } ->
          let lo, hi = ends s in
          let a, b = ends (fit_floats lo hi spec) in
          equal ends_w (a, b) (ends (fit_floats a b spec)));
      prop "nice time domains are their own nice domains"
        (Gen.pair
           (Gen.int_range (-2_000_000_000) 2_000_000_000)
           (Gen.int_range 1 2_000_000_000))
        (fun (s0, w) ->
          let a = Time.v S (Int64.of_int s0) in
          let b = Time.add (Time.seconds 1) w a in
          let fit a b =
            instants (Scale.fit (Some (Scale.Instants (a, b))) (Scale.time ()))
          in
          let a', b' = fit a b in
          at_most instant ~than:a a';
          at_least instant ~than:b b';
          equal (pair instant instant) (a', b') (fit a' b'));
    ]

(* Merging and comparing *)

let red = Hugin_next_gg.Color.red
and blue = Hugin_next_gg.Color.blue

(* Specifications drawn from small pools of values, so that pairs agree, extend
   and conflict. *)
let gen_spec =
  let some l = Gen.option (Gen.of_list l) in
  Gen.map
    (fun ( (log, name, domain, nice),
           (zero, clamp, reverse),
           (scheme, areas, unknown) ) ->
      let make = if log then Scale.symlog ?constant:None else Scale.linear in
      make ?name ?domain ?nice ?zero ?clamp ?reverse ?scheme ?areas ?unknown ())
    (Gen.triple
       (Gen.quad
          (Gen.frequency [ (5, Gen.constant false); (1, Gen.constant true) ])
          (some [ "x"; "y" ])
          (some [ (0., 1.); (0., 2.) ])
          (some [ true; false ]))
       (Gen.triple
          (some [ true; false ])
          (some [ true; false ])
          (some [ true; false ]))
       (Gen.triple
          (some [ Scheme.viridis; Scheme.reverse Scheme.viridis ])
          (some [ (0., 4.); (1., 4.) ])
          (some [ red; blue ])))
  |> Gen.with_pp Scale.pp

let merged = result fscale property

(* Two scales that set one property to different values. *)
type conflict =
  | Conflict : Scale.property * 'd Scale.t * 'd Scale.t -> conflict

let conflicts =
  let band = Scale.band and time = Scale.time in
  [
    Conflict (Name, Scale.linear ~name:"x" (), Scale.linear ~name:"y" ());
    Conflict (Transform, Scale.log (), Scale.linear ());
    Conflict
      ( Domain,
        Scale.linear ~domain:(0., 1.) (),
        Scale.linear ~domain:(0., 2.) () );
    Conflict (Nice, Scale.linear ~nice:true (), Scale.linear ~nice:false ());
    Conflict (Zero, Scale.linear ~zero:true (), Scale.linear ~zero:false ());
    Conflict (Clamp, Scale.linear ~clamp:true (), Scale.linear ~clamp:false ());
    Conflict
      (Reverse, Scale.linear ~reverse:true (), Scale.linear ~reverse:false ());
    Conflict (Padding, band ~padding:0.1 (), band ~padding:0.2 ());
    Conflict (Wrap, band ~wrap:1 (), band ~wrap:2 ());
    Conflict (Tz_offset_s, time ~tz_offset_s:0 (), time ~tz_offset_s:60 ());
    Conflict
      ( Scheme,
        Scale.linear ~scheme:Scheme.viridis (),
        Scale.linear ~scheme:Scheme.magma () );
    Conflict
      (Areas, Scale.linear ~areas:(0., 4.) (), Scale.linear ~areas:(1., 4.) ());
    Conflict
      ( Symbols,
        band ~symbols:[| Symbol.circle |] (),
        band ~symbols:[| Symbol.circle; Symbol.square |] () );
    Conflict
      (Unknown, Scale.linear ~unknown:red (), Scale.linear ~unknown:blue ());
  ]

let conflict_name (Conflict (p, _, _)) =
  Format.asprintf "%a" Scale.pp_property p

let merging =
  group "merge"
    [
      test "a conflict names the first property in their order" (fun () ->
          equal merged (Error Scale.Domain)
            (Scale.merge
               (Scale.linear ~domain:(0., 1.) ())
               (Scale.linear ~domain:(0., 2.) ()));
          equal merged (Error Scale.Name)
            (Scale.merge
               (Scale.linear ~name:"x" ~domain:(0., 1.) ())
               (Scale.linear ~name:"y" ~domain:(0., 2.) ()));
          equal merged (Error Scale.Transform)
            (Scale.merge (Scale.log ()) (Scale.linear ()));
          equal merged (Error Scale.Transform)
            (Scale.merge (Scale.log ~base:2. ()) (Scale.log ()));
          equal (result tscale property) (Error Scale.Tz_offset_s)
            (Scale.merge
               (Scale.time ~tz_offset_s:0 ())
               (Scale.time ~tz_offset_s:60 ())));
      cases ~name:conflict_name "a conflict on each property" conflicts
        (fun (Conflict (p, s, s')) ->
          equal (option property) (Some p)
            (match Scale.merge s s' with Ok _ -> None | Error p -> Some p));
      test "band domains agree when their categories do" (fun () ->
          let band d = Scale.band ~domain:d () in
          let merged =
            result (Testable.make ~pp:Scale.pp ~equal:Scale.equal) property
          in
          is_ok
            (Scale.merge
               (band (Labels [| "a"; "b" |]))
               (band (Labels [| "a"; "b" |])));
          equal merged (Error Scale.Domain)
            (Scale.merge
               (band (Labels [| "a"; "b" |]))
               (band (Labels [| "b"; "a" |])));
          equal merged (Error Scale.Domain)
            (Scale.merge (band (Labels [| "a" |]))
               (band (Labels [| "a"; "b" |])));
          is_ok
            (Scale.merge
               (band (Indices [| (1, "x") |]))
               (band (Indices [| (1, "x") |])));
          equal merged (Error Scale.Domain)
            (Scale.merge
               (band (Indices [| (1, "x") |]))
               (band (Indices [| (1, "y") |])));
          equal merged (Error Scale.Domain)
            (Scale.merge
               (band (Indices [| (1, "x") |]))
               (band (Indices [| (2, "x") |])));
          equal merged (Error Scale.Domain)
            (Scale.merge (band (Indices [||])) (band (Indices [| (2, "x") |]))));
      test "labelled and indexed categories conflict" (fun () ->
          is_error
            (Scale.merge
               (Scale.band ~domain:(Labels [||]) ())
               (Scale.band ~domain:(Indices [||]) ())));
      test "custom transforms agree only with the same functions" (fun () ->
          let f = Float.asinh and g = Float.sinh in
          let a = Scale.custom ~transform:"asinh" ~forward:f ~inverse:g () in
          is_ok
            (Scale.merge a
               (Scale.custom ~transform:"asinh" ~forward:f ~inverse:g ()));
          equal merged (Error Scale.Transform) (Scale.merge a (asinh ())));
      test "a merge takes each property either side sets" (fun () ->
          equal merged
            (Ok (Scale.linear ~name:"y" ~zero:true ~domain:(0., 2.) ()))
            (Scale.merge
               (Scale.linear ~name:"y" ~zero:true ())
               (Scale.linear ~domain:(0., 2.) ())));
      prop "merge is commutative on its successes" (Gen.pair gen_spec gen_spec)
        (fun (a, b) ->
          match (Scale.merge a b, Scale.merge b a) with
          | Ok m, Ok m' -> equal fscale m m'
          | Error p, Error p' -> equal property p p'
          | Ok _, Error _ | Error _, Ok _ -> fail "one order succeeds");
      prop "merge is associative on its successes"
        (Gen.triple gen_spec gen_spec gen_spec) (fun (a, b, c) ->
          let ( let* ) = Result.bind in
          let left =
            let* ab = Scale.merge a b in
            Scale.merge ab c
          in
          let right =
            let* bc = Scale.merge b c in
            Scale.merge a bc
          in
          match (left, right) with
          | Ok l, Ok r -> equal fscale l r
          | Error _, Error _ -> ()
          | Ok _, Error _ | Error _, Ok _ -> fail "one grouping succeeds");
      prop "merge s s is Ok s" gen_spec (fun s ->
          equal merged (Ok s) (Scale.merge s s));
      prop "imply keeps every property its subject sets"
        (Gen.pair gen_spec gen_spec) (fun (i, s) ->
          let r = Scale.imply i s in
          equal merged (Ok r) (Scale.merge s r));
      test "imply fills only what is unset" (fun () ->
          equal fscale
            (Scale.log ~name:"y" ~nice:false ())
            (Scale.imply
               (Scale.linear ~name:"x" ~nice:false ())
               (Scale.log ~name:"y" ())));
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

let comparing =
  group "comparing"
    [
      prop "equal is an equivalence"
        (Gen.pair gen_spec gen_spec)
        (Law.equivalence fscale);
      cases ~name:conflict_name "a property set to different values differs"
        conflicts (fun (Conflict (_, s, s')) ->
          not_equal (Testable.make ~pp:Scale.pp ~equal:Scale.equal) s s');
      test "an unset property differs from its default set" (fun () ->
          not_equal fscale (Scale.linear ()) (Scale.linear ~nice:true ());
          not_equal fscale (Scale.linear ()) (Scale.linear ~domain:(0., 1.) ()));
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
      test "custom transforms compare their functions physically" (fun () ->
          let make forward =
            Scale.custom ~transform:"asinh" ~forward ~inverse:Float.sinh ()
          in
          equal fscale (asinh ()) (asinh ());
          not_equal fscale (make (fun x -> Float.asinh x)) (make Float.asinh));
      test "properties print as their arguments are named" (fun () ->
          let names =
            List.map
              (Format.asprintf "%a" Scale.pp_property)
              [
                Name;
                Transform;
                Domain;
                Nice;
                Zero;
                Clamp;
                Reverse;
                Padding;
                Wrap;
                Tz_offset_s;
                Scheme;
                Areas;
                Symbols;
                Unknown;
              ]
          in
          equal (list string)
            [
              "name";
              "transform";
              "domain";
              "nice";
              "zero";
              "clamp";
              "reverse";
              "padding";
              "wrap";
              "tz_offset_s";
              "scheme";
              "areas";
              "symbols";
              "unknown";
            ]
            names);
      test "pp prints the properties set" (fun () ->
          let pp s = Format.asprintf "%a" Scale.pp s in
          expect
            (String.concat "\n"
               [
                 pp (Scale.linear ~name:"y" ~domain:(0., 0.5) ~nice:false ());
                 pp (Scale.linear ~domain:(0.1 +. 0.2, 1e17 +. 16.) ());
                 pp (Scale.log ~base:2. ~reverse:true ());
                 pp (asinh ~areas:(1., 9.) ());
                 pp
                   (Scale.time ~tz_offset_s:3600
                      ~domain:(Time.epoch, Time.v S 60L)
                      ());
                 pp
                   (Scale.band
                      ~domain:(Indices [| (3, "the") |])
                      ~padding:0.1 ~wrap:2 ());
                 pp (Scale.linear ~scheme:(Scheme.reverse Scheme.rdbu) ());
                 pp
                   (Scale.band ~scheme:Scheme.okabe_ito
                      ~symbols:[| Symbol.circle; Symbol.triangle |]
                      ());
               ])
          @@ __POS_OF__
               {|
            (linear (name "y") (domain 0 0.5) (nice false))
            (linear (domain 0.30000000000000004 1.0000000000000002e+17))
            (log 2 (reverse true))
            (custom asinh (areas 1 9))
            (time (domain 1970-01-01T00:00:00Z 1970-01-01T00:01:00Z) (tz_offset_s 3600))
            (band (domain (indices (3 "the"))) (padding 0.1) (wrap 2))
            (linear (scheme reverse(rdbu)))
            (band (scheme okabe_ito) (symbols circle triangle))
            |});
    ]

(* Hulls *)

let hull_w = option (pair float_exact float_exact)
let f64 a = Nx.create Nx.float64 [| Array.length a |] a

let hulls =
  group "hull"
    [
      test "missing values are left out" (fun () ->
          equal hull_w
            (Some (-2., 7.))
            (Scale.hull (Scale.linear ())
               (f64
                  [|
                    3.; Float.nan; -2.; Float.infinity; 7.; Float.neg_infinity;
                  |])));
      test "a log hull leaves out values that are not positive" (fun () ->
          equal hull_w
            (Some (0.5, 8.))
            (Scale.hull (Scale.log ()) (f64 [| 0.; -1.; 0.5; 8. |]));
          equal hull_w None (Scale.hull (Scale.log ()) (f64 [| 0.; -3. |])));
      test "a custom hull leaves out values its transform cannot take"
        (fun () ->
          equal hull_w
            (Some (2., 5.))
            (Scale.hull (ln ()) (f64 [| -1.; 2.; 0.; 5. |])));
      test "valid keeps elements and broadcasts" (fun () ->
          let x =
            Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
          in
          let valid = Nx.create Nx.bool [| 3 |] [| false; true; false |] in
          equal hull_w (Some (2., 5.)) (Scale.hull ~valid (Scale.linear ()) x);
          let valid = Nx.create Nx.bool [| 2; 1 |] [| false; true |] in
          equal hull_w (Some (4., 6.)) (Scale.hull ~valid (Scale.linear ()) x);
          let valid = Nx.create Nx.bool [| 3 |] [| false; false; false |] in
          equal hull_w None (Scale.hull ~valid (Scale.linear ()) x);
          let valid =
            Nx.create Nx.bool [| 2; 3 |]
              [| true; false; false; false; false; true |]
          in
          equal hull_w (Some (1., 6.)) (Scale.hull ~valid (Scale.linear ()) x));
      test "one element is its own hull" (fun () ->
          equal hull_w
            (Some (3., 3.))
            (Scale.hull (Scale.linear ()) (f64 [| 3. |]));
          equal hull_w (Some (3., 3.)) (Scale.hull (ln ()) (f64 [| 3. |])));
      test "valid must broadcast without growing the tensor" (fun () ->
          let x = f64 [| 1.; 2.; 3. |] in
          let invalid f =
            raises_match (Exn.invalid_arg ~substring:"broadcast") f
          in
          invalid (fun () ->
              Scale.hull
                ~valid:(Nx.create Nx.bool [| 2 |] [| true; true |])
                (Scale.linear ()) x);
          invalid (fun () ->
              Scale.hull
                ~valid:(Nx.create Nx.bool [| 2; 3 |] (Array.make 6 true))
                (Scale.linear ()) x));
      test "an empty tensor has no hull" (fun () ->
          equal hull_w None (Scale.hull (Scale.linear ()) (f64 [||])));
      test "integers are read as floats" (fun () ->
          equal hull_w
            (Some (-4., 9.))
            (Scale.hull (Scale.linear ())
               (Nx.create Nx.int32 [| 3 |] [| 9l; -4l; 0l |])));
      test "complex and boolean tensors are refused" (fun () ->
          invalid (fun () ->
              Scale.hull (Scale.linear ())
                (Nx.create Nx.complex64 [| 1 |] [| Complex.one |]));
          invalid (fun () ->
              Scale.hull (Scale.linear ())
                (Nx.create Nx.bool [| 1 |] [| true |])));
    ]

(* Missing values *)

let missings =
  let scales =
    [
      ("linear", Scale.linear ~domain:(0., 1.) ());
      ("log", Scale.log ~domain:(1., 10.) ());
      ("symlog", Scale.symlog ~domain:(-1., 1.) ());
      ("pow", Scale.pow ~exponent:0.5 ~domain:(0., 4.) ());
      ("custom", ln ~domain:(1., 10.) ());
    ]
  in
  let gen_values =
    Gen.array ~size:(Gen.int_range 0 6)
      (Gen.frequency
         [
           (4, Gen.any_float);
           ( 1,
             Gen.of_list ~pp:Format.pp_print_float
               [ 0.; -0.; -1.; Float.min_float; Float.nan; Float.infinity ] );
         ])
  in
  group "missing"
    [
      (* A set domain has ends that are not missing, so [normalize] is [nan]
         exactly at missing values. *)
      prop "agrees with normalize on a set domain"
        (Gen.pair
           (Gen.of_list
              ~pp:(fun ppf (n, _) -> Format.pp_print_string ppf n)
              scales)
           gen_values)
        (fun ((_, s), xs) ->
          cover "a missing value"
            (Array.exists (fun x -> Float.is_nan (Scale.normalize s x)) xs);
          let expected =
            Array.map (fun x -> Float.is_nan (Scale.normalize s x)) xs
          in
          equal (array bool) expected (Nx.to_array (Scale.missing s (f64 xs))));
      test "keeps the shape and reads integers as floats" (fun () ->
          let x = Nx.create Nx.int32 [| 2; 2 |] [| 0l; 1l; -3l; 5l |] in
          let m = Scale.missing (Scale.log ()) x in
          equal (array int) [| 2; 2 |] (Nx.shape m);
          equal (array bool) [| true; false; true; false |] (Nx.to_array m));
      test "complex and boolean tensors are refused" (fun () ->
          invalid (fun () ->
              Scale.missing (Scale.linear ())
                (Nx.create Nx.complex64 [| 1 |] [| Complex.one |]));
          invalid (fun () ->
              Scale.missing (Scale.linear ())
                (Nx.create Nx.bool [| 1 |] [| true |])));
    ]

let () =
  exit
    (run "Scale"
       [
         constructors;
         normalisation;
         inversion;
         d3_tables;
         time_tables;
         guide_laws;
         guide_cases;
         nice;
         fit;
         merging;
         comparing;
         hulls;
         missings;
       ])
