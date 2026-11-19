(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Directions: constructors return unit vectors, readers read the ray, rotations
   are linear and exact to rounding, and every function has a derivative or a
   stated value with derivative 0. *)

open Windtrap
open Ymir
open Ymir_test.Frames
module Reference = Ymir_test.Frames_reference

let eps = epsilon_float
let pi = Float.pi
let two_pi = 2. *. Float.pi
let bits = array float_exact

(* [ulps n] compares within [n] ulp, subnormals included. *)
let ulps n =
  let n = float_of_int n in
  float_rel ~rel:(n *. eps) ~abs:(n *. Float.succ 0.)

(* [Direction.separation]'s stated relative accuracy. *)
let separation_accuracy = 1.3e-15

(* [Direction.position_angle]'s stated accuracy, in radians. *)
let position_accuracy = 2.1e-15

let row (x, y, z) = vector x y z
let icrs v = raw Frame.icrs v
let xyz d = Nx.to_array (Direction.xyz d)

let norm v =
  Float.sqrt ((v.(0) *. v.(0)) +. (v.(1) *. v.(1)) +. (v.(2) *. v.(2)))

let lonlat l b =
  Direction.lonlat Frame.icrs
    ~lon:(radians (scalar l))
    ~lat:(radians (scalar b))

let pp_row ppf (x, y, z) = Format.fprintf ppf "(%h, %h, %h)" x y z
let is_zero_row (x, y, z) = x = 0. && y = 0. && z = 0.

(* Rows of any finite components, magnitudes over the whole exponent range. *)
let rows =
  Gen.with_pp pp_row
    (Gen.such_that
       (fun r -> not (is_zero_row r))
       Gen.(triple float float float))

(* Rows of components in [-1, 1], away from zero and from the axis. *)
let plain =
  let c = Gen.float_range (-1.) 1. in
  Gen.with_pp pp_row
    (Gen.such_that
       (fun (x, y, z) ->
         let r = Float.sqrt ((x *. x) +. (y *. y) +. (z *. z)) in
         r > 0.1 && Float.sqrt ((x *. x) +. (y *. y)) > 0.25 *. r)
       (Gen.triple c c c))

let angles =
  Gen.pair (Gen.float_range 0. 6.28) (Gen.float_range (-.pi /. 2.) (pi /. 2.))

(* Angles whose latitude lies within 0.07 rad of a pole on two draws of
   five, where [angles] puts one in twenty there. *)
let angles_near_poles =
  let lat =
    Gen.frequency
      [
        (3, Gen.float_range (-.pi /. 2.) (pi /. 2.));
        (1, Gen.float_range 1.5 (pi /. 2.));
        (1, Gen.float_range (-.pi /. 2.) (-1.5));
      ]
  in
  Gen.pair (Gen.float_range 0. 6.28) lat

(* [exact k (x, y, z)] is whether no component of [k] times the row rounds. *)
let exact k (x, y, z) =
  let exact c = Float.is_finite (c *. k) && c *. k /. k = c in
  exact x && exact y && exact z

(* [exact_multiple k r] is [k] times the row [r], which [exact k r] holds
   for. *)
let exact_multiple k (x, y, z) =
  assert (exact k (x, y, z));
  (x *. k, y *. k, z *. k)

(* Rows with a power of two whose multiple of the row rounds no component:
   2^500 or 2^-500 where one is exact, as for most rows, else 2^±100, else
   2^±1. Only a row with components near both ends of the exponent range takes
   none. *)
let multiples =
  let tiers =
    [ [ 0x1p500; 0x1p-500 ]; [ 0x1p100; 0x1p-100 ]; [ 0x1p1; 0x1p-1 ] ]
  in
  let tier r =
    List.map (List.filter (fun k -> exact k r)) tiers
    |> List.find_opt (fun ks -> ks <> [])
  in
  let pp ppf (r, k) = Format.fprintf ppf "%a times %h" pp_row r k in
  Gen.with_pp pp
    Gen.(
      let* r = such_that (fun r -> Option.is_some (tier r)) rows in
      let+ k = of_list (Option.get (tier r)) in
      (r, k))

(* Constructors *)

let constructors =
  group "Constructors return unit vectors"
    [
      prop "lonlat is unit within 3 ulp at any finite angles"
        Gen.(pair (float_range (-1e6) 1e6) (float_range (-1e6) 1e6))
        (fun (l, b) -> equal (float (3. *. eps)) 1. (norm (xyz (lonlat l b))));
      prop "of_xyz is unit within 3 ulp" rows (fun r ->
          let d = Direction.of_xyz Frame.icrs (row r) in
          equal (float (3. *. eps)) 1. (norm (xyz d)));
      prop "of_xyz reads the ray: a power-of-two multiple gives the same bits"
        multiples
        (fun (r, k) ->
          cover "2^500" (k = 0x1p500);
          cover "2^-500" (k = 0x1p-500);
          let k_r = exact_multiple k r in
          equal bits
            (xyz (Direction.of_xyz Frame.icrs (row r)))
            (xyz (Direction.of_xyz Frame.icrs (row k_r))));
      prop "of_xyz (xyz d) returns each component within 3 ulp" angles
        (fun (l, b) ->
          let d = lonlat l b in
          equal
            (array (float (3. *. eps)))
            (xyz d)
            (xyz (Direction.of_xyz Frame.icrs (Direction.xyz d))));
      test "t_s2c" (fun () ->
          equal (array (ulps 4)) Reference.s2c (xyz (lonlat 3.0123 (-0.999))));
      test "lonlat broadcasts its batch axes" (fun () ->
          let lon = radians (Nx.create Nx.float64 [| 2; 1 |] [| 0.; pi |]) in
          let lat =
            radians (Nx.create Nx.float64 [| 3 |] [| -0.5; 0.; 0.5 |])
          in
          let d = Direction.lonlat Frame.icrs ~lon ~lat in
          equal (array int) [| 2; 3; 3 |] (Nx.shape (Direction.xyz d)));
      test "of_xyz keeps a NaN row NaN" (fun () ->
          let d = Direction.of_xyz Frame.icrs (vector Float.nan 1. 0.) in
          equal bits [| Float.nan; Float.nan; Float.nan |] (xyz d));
      test "lonlat raises on an infinite angle, naming it" (fun () ->
          let lat = Nx.create Nx.float64 [| 4 |] [| 0.; 0.; 0.; infinity |] in
          raises (Invalid_argument "Direction.lonlat: lat at [3] is infinite")
            (fun () ->
              Direction.lonlat Frame.icrs
                ~lon:(radians (scalar 0.))
                ~lat:(radians lat)));
      test "lonlat raises on an angle that is not one" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Direction.lonlat Frame.icrs
                ~lon:(Quantity.v Unit.metre (scalar 0.))
                ~lat:(radians (scalar 0.))));
      test "of_xyz raises on a zero vector, naming it" (fun () ->
          let v = Nx.zeros Nx.float64 [| 18; 3 |] in
          let v =
            Nx.add v
              (Nx.create Nx.float64 [| 18; 1 |]
                 (Array.init 18 (fun i -> if i = 17 then 0. else 1.)))
          in
          raises
            (Invalid_argument
               "Direction.of_xyz: the vector at [17] is zero and names no \
                direction") (fun () -> Direction.of_xyz Frame.icrs v));
      test "of_xyz raises on an infinite vector" (fun () ->
          raises (Invalid_argument "Direction.of_xyz: the vector is infinite")
            (fun () -> Direction.of_xyz Frame.icrs (vector 1. infinity 0.)));
      test "of_xyz raises on a last axis without 3 elements" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Direction.of_xyz")
            (fun () ->
              Direction.of_xyz Frame.icrs (Nx.ones Nx.float64 [| 2; 4 |])));
    ]

(* Readers *)

let lon_lat d = (one (lon d), one (lat d))

let round_trips =
  group "lon and lat invert lonlat"
    [
      prop "round trip within 4 ulp" angles_near_poles (fun (l, b) ->
          cover "near a pole" (Float.abs b > 1.5);
          let l', b' = lon_lat (lonlat l b) in
          equal (ulps 4) l l';
          equal (ulps 4) b b');
      cases
        "stated angles" ~id:(fun (n, _, _, _) -> n)
        [
          ("the north pole", 1., pi /. 2., (1., pi /. 2.));
          ("the south pole", 1., -.pi /. 2., (1., -.pi /. 2.));
          ("lon -0.", -0., 0.3, (0., 0.3));
          ("lat -0.", 2., -0., (2., -0.));
          ("lon 2π", two_pi, 0.3, (0., 0.3));
          ("lat 100° at lon 0°", 0., 100. *. pi /. 180., (pi, 80. *. pi /. 180.));
        ]
        (fun (_, l, b, (l', b')) ->
          let got_l, got_b = lon_lat (lonlat l b) in
          equal (ulps 4) l' got_l;
          equal (ulps 4) b' got_b);
      test "lon -0. reads +0." (fun () ->
          equal float_exact 0. (one (lon (lonlat (-0.) 0.3))));
      test "y = -1e-300 below the x axis reads +0., never 2π" (fun () ->
          equal float_exact 0. (one (lon (icrs (vector 1. (-1e-300) 0.)))));
      test "t_c2s" (fun () ->
          let d = icrs (vector 100. (-50.) 25.) in
          equal (ulps 4) (two_pi -. 0.4636476090008061162) (one (lon d));
          equal (ulps 4) 0.2199879773954594463 (one (lat d)));
    ]

let stated =
  group "Readers give stated values on zero, tiny, NaN and infinite rows"
    [
      test "a zero row reads lon 0 and lat 0" (fun () ->
          let d = icrs (vector 0. 0. 0.) in
          equal float_exact 0. (one (lon d));
          equal float_exact 0. (one (lat d)));
      test "the poles read lon 0 and lat ±π/2" (fun () ->
          equal bits
            [| 0.; pi /. 2. |]
            [|
              one (lon (icrs (vector 0. 0. 3.)));
              one (lat (icrs (vector 0. 0. 3.)));
            |];
          equal float_exact (-.pi /. 2.) (one (lat (icrs (vector 0. 0. (-3.))))));
      test "subnormal rows read their direction" (fun () ->
          let tiny = 1e-310 in
          equal float_exact (pi /. 2.) (one (lon (icrs (vector 0. tiny 0.))));
          equal float_exact (-.pi /. 2.)
            (one (lat (icrs (vector 0. 0. (-.tiny)))));
          equal (ulps 4) 0.6154797086703873
            (one (lat (icrs (vector tiny tiny tiny)))));
      test "a zero row is 0 from anything" (fun () ->
          let z = icrs (vector 0. 0. 0.) and b = lonlat 1. 0.5 in
          equal bits [| 0.; 0.; 0.; 0. |]
            [|
              one (separation z b);
              one (separation b z);
              one (position_angle z b);
              one (position_angle b z);
            |]);
      test "coincident and antipodal rows" (fun () ->
          let a = lonlat 1. 0.5 in
          let anti = icrs (Nx.neg (Direction.xyz a)) in
          equal bits [| 0.; pi; 0.; 0. |]
            [|
              one (separation a a);
              one (separation a anti);
              one (position_angle a a);
              one (position_angle a anti);
            |]);
      test "a NaN component gives NaN" (fun () ->
          let d = icrs (vector Float.nan 0. 1.) and b = lonlat 1. 0.5 in
          equal bits
            [| Float.nan; Float.nan; Float.nan; Float.nan |]
            [|
              one (lon d);
              one (lat d);
              one (separation d b);
              one (position_angle b d);
            |]);
      cases
        "an infinite component raises" ~id:(fun (n, _) -> n)
        [
          ("lon", fun d -> ignore (Direction.lon d));
          ("lat", fun d -> ignore (Direction.lat d));
          ("separation", fun d -> ignore (Direction.separation (lonlat 0. 0.) d));
          ( "position_angle",
            fun d -> ignore (Direction.position_angle d (lonlat 0. 0.)) );
        ]
        (fun (n, f) ->
          raises_match
            (Exn.invalid_arg ~sub:("Direction." ^ n))
            (fun () -> f (icrs (vector 1. neg_infinity 0.))));
    ]

let reads_the_ray =
  let rows = multiples in
  let covers (_, k) =
    cover "2^500" (k = 0x1p500);
    cover "2^-500" (k = 0x1p-500)
  in
  let reads (r, k) f =
    let k_r = exact_multiple k r in
    equal bits (f (icrs (row r))) (f (icrs (row k_r)))
  in
  let other = lonlat 0.7 (-0.2) in
  group "Readers read the ray"
    [
      prop "lon and lat: a power-of-two multiple reads the same bits" rows
        (fun c ->
          covers c;
          reads c lon;
          reads c lat);
      prop "separation and position_angle: the same, on either side" rows
        (fun c ->
          covers c;
          reads c (fun d -> separation d other);
          reads c (fun d -> separation other d);
          reads c (fun d -> position_angle d other);
          reads c (fun d -> position_angle other d));
      cases
        "a row of subnormal components reads as its normal multiple" ~id:(fun r -> Format.asprintf "%a" pp_row r)
        [
          (0., -0x1p-1074, 0.);
          (0x1p-1074, 0x1p-1074, -0x1p-1074);
          (0x0.fffffffffffffp-1022, 0x1p-1060, 0.);
          (0x1p-1050, 0., 0x1.8p-1049);
        ]
        (fun r ->
          let c = (r, 0x1p600) in
          reads c lon;
          reads c lat;
          reads c (fun d -> separation d other);
          reads c (fun d -> separation other d);
          reads c (fun d -> position_angle d other);
          reads c (fun d -> position_angle other d));
      prop "a rounded multiple reads within the rounding"
        Gen.(pair plain (of_list [ 1e200; 1e-200; 3.7; 0.1 ]))
        (fun ((x, y, z), k) ->
          let v = icrs (row (x, y, z))
          and kv = icrs (row (x *. k, y *. k, z *. k)) in
          equal (float (4. *. eps)) (one (lon v)) (one (lon kv));
          equal (float (4. *. eps)) (one (lat v)) (one (lat kv));
          equal
            (float (4. *. eps))
            (one (separation v other))
            (one (separation kv other)));
    ]

(* Measures *)

(* Pairs of rows: of any magnitudes, plain, and near each other or each
   other's antipode. *)
let row_pairs =
  let near =
    Gen.(
      let+ (x, y, z) = plain
      and+ d = triple (float_range (-1e-8) 1e-8) (float_range (-1e-8) 1e-8)
          (float_range (-1e-8) 1e-8)
      and+ k = of_list [ 1.; -1.; 3.7 ] in
      let dx, dy, dz = d in
      ((x, y, z), (k *. (x +. dx), k *. (y +. dy), k *. (z +. dz))))
  in
  Gen.one_of [ Gen.pair rows rows; Gen.pair plain plain; near ]

let measures =
  group "Separation and position angle"
    [
      cases
        "against a 60-digit reference" ~id:(fun (n, _, _, _, _) -> n) Reference.pairs
        (fun (_, a, b, sep, pa) ->
          let a = icrs (Nx.create Nx.float64 [| 3 |] a)
          and b = icrs (Nx.create Nx.float64 [| 3 |] b) in
          equal
            (float_rel ~rel:separation_accuracy ~abs:0.)
            sep
            (one (separation a b));
          equal (float position_accuracy) pa (one (position_angle a b)));
      cases
        "rows of unequal norms, against a 60-digit reference" ~id:(fun (n, _, _, _, _) -> n)
        Ymir_test.Unequal_norms_reference.pairs
        (fun (_, a, b, sep, pa) ->
          let a = icrs (Nx.create Nx.float64 [| 3 |] a)
          and b = icrs (Nx.create Nx.float64 [| 3 |] b) in
          equal
            (float_rel ~rel:separation_accuracy ~abs:0.)
            sep
            (one (separation a b));
          equal (float position_accuracy) pa (one (position_angle a b)));
      cases
        "a separation far below float64's rounding keeps its accuracy" ~id:(fun t -> Printf.sprintf "%g" t)
        [ 1e-160; 1e-300 ]
        (fun t ->
          (* From (1, 0, 0) to (1, t, 0) the separation is atan t, which is t
             to rounding, and the bearing is east. [|a × b|]'s square is
             below float64's least normal. *)
          let a = icrs (vector 1. 0. 0.) and b = icrs (vector 1. t 0.) in
          equal
            (float_rel ~rel:separation_accuracy ~abs:0.)
            t
            (one (separation a b));
          equal (float position_accuracy) (pi /. 2.)
            (one (position_angle a b)));
      test "t_sepp, on vectors that are not unit" (fun () ->
          equal
            (float_rel ~rel:1e-15 ~abs:0.)
            2.860391919024660768
            (one
               (separation
                  (icrs (vector 1. 0.1 0.2))
                  (icrs (vector (-3.) 1e-3 0.2)))));
      test "t_pap" (fun () ->
          equal (float 1e-15) 0.3671514267841113674
            (one
               (position_angle
                  (icrs (vector 1. 0.1 0.2))
                  (icrs (vector (-3.) 1e-3 0.2)))));
      test "t_pas" (fun () ->
          equal (float 1e-15)
            (two_pi -. 2.724544922932270424)
            (one (position_angle (lonlat 1.0 0.1) (lonlat 0.2 (-1.0)))));
      test "at the north pole, north is toward 180° and east toward 90°"
        (fun () ->
          let pole = icrs (vector 0. 0. 1.) in
          equal (float 1e-15) 0. (one (position_angle pole (lonlat pi 0.)));
          equal (float 1e-15) (pi /. 2.)
            (one (position_angle pole (lonlat (pi /. 2.) 0.))));
      prop "separation b a has the bits of separation a b" ~count:2000 row_pairs
        (fun (a, b) ->
          let a = icrs (row a) and b = icrs (row b) in
          let s = one (separation a b) in
          cover "below 1e-6 rad" (s < 1e-6);
          cover "near pi" (s > pi -. 1e-6);
          equal float_exact s (one (separation b a)));
      (let compiled =
         Rune.jit
           Nx.Ptree.(tensor @-> tensor @-> returns tensor)
           (fun a b -> in_radians (Direction.separation (icrs a) (icrs b)))
       in
       let side f ps =
         vectors
           (Array.concat
              (Array.to_list
                 (Array.map
                    (fun p ->
                      let x, y, z = f p in
                      [| x; y; z |])
                    ps)))
       in
       prop "compiled, separation b a has the bits of separation a b"
         ~count:5
         (Gen.array ~size:(Gen.constant 512) row_pairs)
         (fun ps ->
           let a = side fst ps and b = side snd ps in
           equal bits
             (Nx.to_array (compiled a b))
             (Nx.to_array (compiled b a))));
      prop "a rotation keeps separations within 1e-15 rad"
        (Gen.pair plain plain) (fun (a, b) ->
          let a = Direction.of_xyz Frame.icrs (row a)
          and b = Direction.of_xyz Frame.icrs (row b) in
          equal (float 1e-15)
            (one (separation a b))
            (one
               (separation
                  (Direction.rotate Frame.galactic a)
                  (Direction.rotate Frame.galactic b))));
      test "leading axes broadcast" (fun () ->
          let ls = [| 0.; 1.; 2.; 3. |] in
          let many =
            Direction.lonlat Frame.icrs
              ~lon:(radians (Nx.create Nx.float64 [| 4 |] ls))
              ~lat:(radians (scalar 0.3))
          in
          let b = lonlat 0.5 (-0.1) in
          equal bits
            (Array.map (fun l -> one (separation (lonlat l 0.3) b)) ls)
            (separation many b);
          equal bits
            (Array.map (fun l -> one (position_angle b (lonlat l 0.3))) ls)
            (position_angle b many));
    ]

(* Rotations *)

let frame = Gen.of_list ~pp:pp_fixed fixed

let rotations =
  group "rotate is the matrix product, rounded once per term"
    [
      prop "each component is within 1.5 · 2⁻⁵² |v| of the exact product"
        Gen.(triple frame frame angles)
        (fun (F a, F b, (l, la)) ->
          let d =
            Direction.lonlat a
              ~lon:(radians (scalar l))
              ~lat:(radians (scalar la))
          in
          let v = xyz d in
          let exact = Array.map fst (exact_rotate (F a) (F b) v) in
          equal
            (array (float (1.5 *. 0x1p-52 *. norm v)))
            exact
            (xyz (Direction.rotate b d)));
      prop "rotate keeps the norm within 3 ulp"
        Gen.(triple frame frame angles)
        (fun (F a, F b, (l, la)) ->
          let d =
            Direction.lonlat a
              ~lon:(radians (scalar l))
              ~lat:(radians (scalar la))
          in
          equal (float (3. *. eps)) 1. (norm (xyz (Direction.rotate b d))));
      prop "rotate is linear"
        Gen.(pair frame plain)
        (fun (F b, r) ->
          let v = row r in
          let twice = Nx.mul_s v 2. in
          equal bits
            (Array.map (fun x -> 2. *. x) (xyz (Direction.rotate b (icrs v))))
            (xyz (Direction.rotate b (icrs twice))));
      test "rotate to its own frame is the identity" (fun () ->
          let d = lonlat 1. 0.2 in
          equal bits (xyz d) (xyz (Direction.rotate Frame.icrs d)));
      test "eager and compiled agree bit for bit" (fun () ->
          let v =
            Direction.xyz
              (Direction.lonlat Frame.galactic
                 ~lon:(radians (Nx.linspace Nx.float64 0. 6. 64))
                 ~lat:(radians (Nx.linspace Nx.float64 (-1.5) 1.5 64)))
          in
          let f v =
            Direction.xyz
              (Direction.rotate Frame.ecliptic_j2000 (raw Frame.galactic v))
          in
          equal bits (Nx.to_array (f v)) (Nx.to_array (Rune.jit' f v)));
    ]

(* Gradients *)

let grad f v = Nx.to_array (Rune.grad' (fun v -> Nx.sum (in_radians (f v))) v)

(* [central f v] is the central difference of [f], an angle, along each
   component of the 3-vector [v]. The difference is taken modulo 2π, so a step
   across a longitude's or a bearing's cut costs nothing. *)
let central f v =
  let h = 1e-6 in
  let at i s =
    let w = Array.copy v in
    w.(i) <- w.(i) +. s;
    one (Nx.to_array (in_radians (f (Nx.create Nx.float64 [| 3 |] w))))
  in
  let wrap d = Float.rem (d +. (3. *. pi)) two_pi -. pi in
  Array.init 3 (fun i -> wrap (at i h -. at i (-.h)) /. (2. *. h))

let near_grad = array (float_rel ~rel:1e-7 ~abs:1e-9)
let other = lonlat 2.1 (-0.4)

let gradients =
  group "Derivatives are analytic, and 0 where none exists"
    [
      prop "grad matches central differences away from the singular sets" plain
        (fun r ->
          let x, y, z = r in
          let v = [| x; y; z |] in
          let t = row r in
          assume (one (separation (icrs t) other) > 0.1);
          assume (one (separation (icrs t) other) < pi -. 0.1);
          List.iter
            (fun f -> equal near_grad (central f v) (grad f t))
            [
              (fun v -> Direction.lon (icrs v));
              (fun v -> Direction.lat (icrs v));
              (fun v -> Direction.separation (icrs v) other);
              (fun v -> Direction.position_angle (icrs v) other);
              (fun v -> Direction.position_angle other (icrs v));
            ]);
      cases
        "exact 0 and no NaN on the singular sets" ~id:(fun (n, _, _) -> n)
        (let a = Direction.xyz (lonlat 1. 0.5) in
         let pole = vector 0. 0. 1. and zero = vector 0. 0. 0. in
         [
           ("lon at a pole", (fun v -> Direction.lon (icrs v)), pole);
           ("lat at a pole", (fun v -> Direction.lat (icrs v)), pole);
           ("lat at a zero row", (fun v -> Direction.lat (icrs v)), zero);
           ( "separation at a coincident row",
             (fun v -> Direction.separation (icrs v) (icrs a)),
             a );
           ( "separation at an antipodal row",
             (fun v -> Direction.separation (icrs v) (icrs a)),
             Nx.neg a );
           ( "separation at a zero row",
             (fun v -> Direction.separation (icrs v) (icrs a)),
             zero );
           ( "position_angle at b = a",
             (fun v -> Direction.position_angle (icrs a) (icrs v)),
             a );
           ( "position_angle at an antipodal b",
             (fun v -> Direction.position_angle (icrs a) (icrs v)),
             Nx.neg a );
           ( "position_angle at a zero row",
             (fun v -> Direction.position_angle (icrs v) (icrs a)),
             zero );
           ( "position_angle with respect to a at a pole",
             (fun v -> Direction.position_angle (icrs v) (icrs a)),
             pole );
         ])
        (fun (_, f, v) -> equal bits [| 0.; 0.; 0. |] (grad f v));
      test "position_angle from a pole differentiates with respect to b"
        (fun () ->
          let b = [| 0.3; 0.4; 0.5 |] in
          let f v =
            Direction.position_angle (icrs (vector 0. 0. 1.)) (icrs v)
          in
          equal near_grad (central f b)
            (grad f (Nx.create Nx.float64 [| 3 |] b)));
      test "the Guide's slope near the pole and at zero" (fun () ->
          let p0 =
            Direction.lonlat Frame.icrs
              ~lon:(radians (scalar 0.3))
              ~lat:(radians (scalar 0.5))
          in
          let pole =
            Direction.lonlat Frame.icrs
              ~lon:(degrees (scalar 0.))
              ~lat:(degrees (scalar 90.))
          in
          let from target lat =
            let p =
              Direction.lonlat Frame.icrs
                ~lon:(radians (scalar 0.3))
                ~lat:(radians lat)
            in
            Quantity.value Unit.arcsecond (Direction.separation p target)
          in
          let slope = Rune.grad' (from pole) (scalar ((pi /. 2.) -. 1e-7)) in
          let at_zero = Rune.grad' (from p0) (scalar 0.5) in
          equal
            (float_rel ~rel:1e-9 ~abs:0.)
            (-180. *. 3600. /. pi)
            (Nx.item [] slope);
          equal float_exact 0. (Nx.item [] at_zero));
    ]

(* Cross-matching *)

(* A cross-match case: catalogues [a] and [b] in one frame, scattered by [h]
   around a few centres (at the poles, across longitude 0 and anywhere), with
   rays of several norms, antipodes, NaN and zero rows, and copies; and a radius
   of a kind, for every row or scaled per row. *)
type radius = Zero | Tiny | Small | Large | Near_pi | Past_pi | Boundary

type xmatch = {
  frame : fixed;
  radius : radius;
  r : float;
  per_row : float list option;
  a : (float * float * float) list;
  b : (float * float * float) list;
}

let radius_name = function
  | Zero -> "zero"
  | Tiny -> "tiny"
  | Small -> "small"
  | Large -> "large"
  | Near_pi -> "near pi"
  | Past_pi -> "past pi"
  | Boundary -> "boundary"

let pp_xmatch ppf c =
  let rows ppf l =
    Format.pp_print_list ~pp_sep:Format.pp_print_space pp_row ppf l
  in
  Format.fprintf ppf "@[<v>%a, %s radius %h%a@,a: @[%a@]@,b: @[%a@]@]" pp_fixed
    c.frame (radius_name c.radius) c.r
    (fun ppf -> function
      | None -> ()
      | Some f ->
          let comma ppf () = Format.pp_print_string ppf ", " in
          Format.fprintf ppf " times @[<h>%a@]"
            (Format.pp_print_list ~pp_sep:comma Format.pp_print_float)
            f)
    c.per_row rows c.a rows c.b

let unit_vector lon lat = (cos lat *. cos lon, cos lat *. sin lon, sin lat)
let log_uniform lo hi = Gen.map (fun e -> 10. ** e) (Gen.float_range lo hi)

let centre =
  Gen.(
    let* kind = int_range 0 2 in
    match kind with
    | 0 ->
        let+ lon = float_range 0. 6.28 and+ lat = float_range (-1.5) 1.5 in
        unit_vector lon lat
    | 1 ->
        let+ lon = float_range 0. 6.28
        and+ d = float_range 0. 1e-6
        and+ north = bool in
        unit_vector lon (if north then (pi /. 2.) -. d else d -. (pi /. 2.))
    | _ ->
        let+ lon = float_range (-1e-6) 1e-6 and+ lat = float_range (-1.5) 1.5 in
        unit_vector lon lat)

(* The kind of radius, its value and the spread [h] of the rows that test it. A
   boundary radius is the separation of planted rows, set later. *)
let radius =
  Gen.(
    let* kind =
      of_list [ Zero; Tiny; Small; Large; Near_pi; Past_pi; Boundary ]
    in
    let spread r = map (fun u -> r *. u) (float_range 0. 2.) in
    match kind with
    | Zero ->
        let+ h = of_list [ 0.; 1e-15; 1e-12 ] in
        (kind, 0., h)
    | Tiny ->
        let* r = log_uniform (-12.) (-6.) in
        let+ h = spread r in
        (kind, r, h)
    | Small ->
        let* r = log_uniform (-6.) (-2.) in
        let+ h = spread r in
        (kind, r, h)
    | Large ->
        let+ r = float_range 0.1 3. and+ h = float_range 0. 1. in
        (kind, r, h)
    | Near_pi ->
        let* d = log_uniform (-9.) (-3.) in
        let+ h = spread d in
        (kind, pi -. d, h)
    | Past_pi ->
        let+ r = of_list [ pi; 3.5; 1e300; infinity ]
        and+ h = float_range 0. 1. in
        (kind, r, h)
    | Boundary ->
        let+ h = log_uniform (-8.) (-1.) in
        (kind, 0., h))

let row centres h =
  Gen.(
    let unit = float_range (-1.) 1. in
    let* c, u, k =
      triple (of_list centres) (triple unit unit unit)
        (of_list [ 1.; 1.; 3.7; 0x1p-600; 1e10 ])
    in
    let (cx, cy, cz), (ux, uy, uz) = (c, u) in
    let near s =
      ( k *. ((s *. cx) +. (h *. ux)),
        k *. ((s *. cy) +. (h *. uy)),
        k *. ((s *. cz) +. (h *. uz)) )
    in
    frequency
      [
        (8, constant (near 1.));
        (1, constant (near (-1.)));
        (1, of_list [ (nan, k, 0.); (nan, nan, nan) ]);
        (1, of_list [ (0., 0., 0.); (-0., 0., 0.) ]);
      ])

(* [copies rows] is up to three rows of [rows], drawn again. *)
let copies rows =
  if rows = [] then Gen.constant []
  else Gen.list ~size:(Gen.int_range 0 3) (Gen.of_list rows)

let xmatches =
  Gen.with_pp pp_xmatch
    Gen.(
      let factors = list ~size:(constant 16) (of_list [ 0.; 0.5; 1.; 1.5 ]) in
      let* frame, (radius, r, h), centres, per_row =
        quad (of_list fixed) radius
          (list ~size:(int_range 1 3) centre)
          (option factors)
      in
      let size = frequency [ (1, constant 0); (6, int_range 1 10) ] in
      let* a = list ~size (row centres h) in
      let* b = list ~size (row centres h) in
      let* a' = copies a in
      let+ b' = copies (a @ b) in
      { frame; radius; r; per_row; a = a @ a'; b = b @ b' })

let rows_of f l =
  raw f
    (vectors (Array.of_list (List.concat_map (fun (x, y, z) -> [ x; y; z ]) l)))

let names_direction (x, y, z) =
  not
    (Float.is_nan x || Float.is_nan y || Float.is_nan z || is_zero_row (x, y, z))

(* [separations c] is every row of [c.a] against every row of [c.b], by
   {!Direction.separation}, row-major. *)
let separations c =
  let na = List.length c.a and nb = List.length c.b in
  if na = 0 || nb = 0 then [||]
  else
    let (F f) = c.frame in
    let shape s d =
      Nx.Ptree.map (Direction.ptree ()) (fun _ x -> Nx.reshape s x) d
    in
    let a = shape [| na; 1; 3 |] (rows_of f c.a) in
    let b = shape [| 1; nb; 3 |] (rows_of f c.b) in
    Nx.to_array (in_radians (Direction.separation a b))

(* [radii c s] is each row of [a]'s radius. A boundary radius is the separation
   of row [i] from row [i mod nb] of [b], or of the first rows. *)
let radii c s =
  let na = List.length c.a and nb = List.length c.b in
  let at i j = if nb = 0 then 0. else s.((i * nb) + j) in
  let finite x = if Float.is_nan x then 0. else x in
  match (c.radius, c.per_row) with
  | Boundary, None -> Array.make na (if na = 0 then 0. else finite (at 0 0))
  | Boundary, Some _ -> Array.init na (fun i -> finite (at i (i mod max nb 1)))
  | _, None -> Array.make na c.r
  | _, Some f ->
      (* 0 times an infinite radius is 0 *)
      let scaled k = if k = 0. then 0. else c.r *. k in
      Array.init na (fun i -> scaled (List.nth f i))

(* [brute c s r] is every pair [(i, j)] of rows naming directions whose
   separation is at most [r.(i)], in increasing [i], then [j]. *)
let brute c s r =
  let a = Array.of_list c.a and b = Array.of_list c.b in
  let nb = Array.length b in
  List.concat
    (List.init (Array.length a) (fun i ->
         List.filter
           (fun (i, j) ->
             names_direction a.(i)
             && names_direction b.(j)
             && s.((i * nb) + j) <= r.(i))
           (List.init nb (fun j -> (i, j)))))

(* [triples (i, j, s)] is a cross-match's result as a list of [(i, j, s)], [s]
   in radians. *)
let triples (i, j, s) =
  let ints t = Array.map Int64.to_int (Nx.to_array t) in
  let i = ints i and j = ints j and s = Nx.to_array (in_radians s) in
  List.init (Array.length i) (fun k -> (i.(k), j.(k), s.(k)))

(* [pairs_of c r] is [Direction.pairs]' triples [(i, j, s)]. Radii per row are a
   call at the largest, then the documented filter. *)
let pairs_of c r =
  let (F f) = c.frame in
  let largest = Array.fold_left Float.max c.r r in
  let within = if c.per_row = None && r <> [||] then r.(0) else largest in
  let i, j, s =
    Direction.pairs
      ~within:(radians (scalar within))
      (rows_of f c.a) (rows_of f c.b)
  in
  let i, j, s =
    if c.per_row = None then (i, j, s)
    else
      let s = in_radians s in
      let rr = Nx.create Nx.float64 [| Array.length r |] r in
      let keep = Nx.positions (Nx.less_equal s (Nx.take i rr)) in
      (Nx.take keep i, Nx.take keep j, radians (Nx.take keep s))
  in
  triples (i, j, s)

let near_pole (x, y, z) =
  Float.abs z > (1. -. 1e-9) *. Float.sqrt ((x *. x) +. (y *. y) +. (z *. z))

(* Two rows on either side of longitude 0, near it. *)
let across_zero (x, y, _) (x', y', _) = x > 0. && x' > 0. && y *. y' < 0.

(* [nearest_brute n rows] is, for each of [n] rows, the partner at the least
   separation among [rows], a list of [(row, partner, s)], the lowest on ties,
   or -1. *)
let nearest_brute n rows =
  let best = Array.make n (-1, infinity) in
  List.iter
    (fun (k, partner, s) ->
      let p, bs = best.(k) in
      if s < bs || (s = bs && (p < 0 || partner < p)) then
        best.(k) <- (partner, s))
    rows;
  Array.map fst best

let swap (i, j, s) = (j, i, s)

(* [reciprocal_brute c s r] is [(expected, all, ab, ba)]: the triples of
   [brute c s r] whose rows are each other's nearest, all its triples, and each
   row's nearest in the other catalogue, or -1. *)
let reciprocal_brute c s r =
  let nb = List.length c.b in
  let all = List.map (fun (i, j) -> (i, j, s.((i * nb) + j))) (brute c s r) in
  let ab = nearest_brute (List.length c.a) all in
  let ba = nearest_brute nb (List.map swap all) in
  (List.filter (fun (i, j, _) -> ab.(i) = j && ba.(j) = i) all, all, ab, ba)

(* [tied rows] holds when a row of [rows], a list of [(row, partner, s)], has
   two partners at its least separation. *)
let tied rows =
  List.exists
    (fun (k, _, _) ->
      let mine = List.filter (fun (k', _, _) -> k' = k) rows in
      let least =
        List.fold_left (fun m (_, _, s) -> Float.min m s) infinity mine
      in
      List.length (List.filter (fun (_, _, s) -> s = least) mine) > 1)
    rows

(* [within_of c r] is the 0-d radius of a call without radii per row. *)
let within_of c r = radians (scalar (if r = [||] then c.r else r.(0)))

let reciprocal_of c within =
  let (F f) = c.frame in
  triples (Direction.reciprocal ~within (rows_of f c.a) (rows_of f c.b))

let crossmatch =
  group "pairs"
    [
      prop
        "pairs are every pair within the radius, in order, with its separation"
        ~count:300 xmatches (fun c ->
          let s = separations c in
          let r = radii c s in
          let expected = brute c s r in
          let a = Array.of_list c.a and b = Array.of_list c.b in
          let sep (i, j) = s.((i * Array.length b) + j) in
          let some p = List.exists p expected in
          let valid l = List.length (List.filter names_direction l) in
          cover "a pair at a pole" (some (fun (i, _) -> near_pole a.(i)));
          cover "a pair across longitude 0"
            (some (fun (i, j) -> across_zero a.(i) b.(j)));
          cover "radius 0 with a pair" (c.radius = Zero && expected <> []);
          cover "radius near pi, a pair outside it"
            (c.radius = Near_pi && List.length expected < valid c.a * valid c.b);
          cover "radius pi or more" (c.radius = Past_pi && expected <> []);
          cover "a pair at the radius"
            (some (fun (i, j) -> r.(i) > 0. && sep (i, j) = r.(i)));
          cover "tied pairs"
            (some (fun (i, j) ->
                 some (fun (i', j') ->
                     i' = i && j' <> j && sep (i, j') = sep (i, j))));
          cover "a NaN row"
            (List.exists (fun (x, _, _) -> Float.is_nan x) (c.a @ c.b));
          cover "a zero row" (List.exists is_zero_row (c.a @ c.b));
          cover "an empty catalogue" (c.a = [] || c.b = []);
          cover "a frame other than ICRS" (name c.frame <> "icrs");
          cover "radius per row" (Option.is_some c.per_row && expected <> []);
          let expected = List.map (fun p -> (fst p, snd p, sep p)) expected in
          equal (list (triple int int float_exact)) expected (pairs_of c r));
      test "raises on a radius that is negative, NaN, not 0-d or not an angle"
        (fun () ->
          let d = lonlat 1. 0.2 in
          let one =
            Nx.Ptree.map (Direction.ptree ())
              (fun _ x -> Nx.reshape [| 1; 3 |] x)
              d
          in
          let bad w =
            raises_match (Exn.invalid_arg ~sub:"Direction.pairs") (fun () ->
                Direction.pairs ~within:w one one)
          in
          bad (radians (scalar (-1.)));
          bad (radians (scalar nan));
          bad (radians (Nx.create Nx.float64 [| 2 |] [| 1.; 1. |]));
          raises_match Exn.invalid_arg (fun () ->
              Direction.pairs
                ~within:(Quantity.v Unit.metre (scalar 1.))
                one one));
      test "raises on rows not [n; 3] and on infinite rows" (fun () ->
          let within = radians (scalar 1.) in
          let raises a b =
            raises_match (Exn.invalid_arg ~sub:"Direction.pairs") (fun () ->
                Direction.pairs ~within a b)
          in
          let d = lonlat 1. 0.2 in
          raises d (icrs (vectors [| 1.; 0.; 0. |]));
          raises
            (icrs (vectors [| infinity; 0.; 0. |]))
            (icrs (vectors [| 1.; 0.; 0. |]));
          raises
            (icrs (vectors [| 1.; 0.; 0. |]))
            (icrs (vectors [| 0.; nan; neg_infinity |])));
    ]

let matches = list (triple int int float_exact)
let icrs_rows l = rows_of Frame.icrs l

let north = (0., 0., 1.)
let south = (0., 0., -1.)

let reciprocity =
  group "reciprocal"
    [
      prop "reciprocal is the pairs whose rows are each other's nearest"
        xmatches (fun c ->
          let c = { c with per_row = None } in
          let s = separations c in
          let r = radii c s in
          let expected, all, ab, ba = reciprocal_brute c s r in
          let a = Array.of_list c.a in
          let some f v = Array.exists Fun.id (Array.mapi f v) in
          cover "a mutual pair" (expected <> []);
          cover "a tie for a row of a" (tied all);
          cover "a tie for a row of b" (tied (List.map swap all));
          cover "a one-way nearest"
            (some (fun i j -> j >= 0 && ba.(j) <> i) ab);
          cover "a row with no candidate"
            (some (fun i j -> j < 0 && names_direction a.(i)) ab);
          cover "a NaN row"
            (List.exists (fun (x, _, _) -> Float.is_nan x) (c.a @ c.b));
          cover "a zero row" (List.exists is_zero_row (c.a @ c.b));
          cover "an empty catalogue" (c.a = [] || c.b = []);
          equal matches expected (reciprocal_of c (within_of c r)));
      prop "reciprocal b a is reciprocal a b with its sides swapped" ~count:200
        xmatches
        (fun c ->
          let c = { c with per_row = None } in
          let s = separations c in
          let r = radii c s in
          let within = within_of c r in
          let _, all, _, _ = reciprocal_brute c s r in
          let got = reciprocal_of c within in
          cover "a tie for a row of a" (tied all);
          cover "a tie for a row of b" (tied (List.map swap all));
          cover "a large radius, a mutual pair"
            (c.radius = Large && got <> []);
          cover "radius pi or more, a mutual pair"
            (c.radius = Past_pi && got <> []);
          let swapped = List.sort compare (List.map swap got) in
          let flipped = { c with a = c.b; b = c.a } in
          equal matches swapped (reciprocal_of flipped within));
      prop "reciprocal selects among pairs' triples, each row at most once"
        xmatches (fun c ->
          let c = { c with per_row = None } in
          let r = radii c (separations c) in
          let got = reciprocal_of c (within_of c r) in
          let pairs = pairs_of c r in
          cover "a pair left out" (List.length pairs > List.length got);
          equal matches got (List.filter (fun t -> List.mem t got) pairs);
          let i = List.map (fun (i, _, _) -> i) got
          and j = List.map (fun (_, j, _) -> j) got in
          equal (list int) (List.sort_uniq compare i) (List.sort compare i);
          equal (list int) (List.sort_uniq compare j) (List.sort compare j));
      test "at radius 0, coincident rows pair, the lowest row on ties"
        (fun () ->
          let p = unit_vector 1. 0.2 and q = unit_vector 1.5 (-0.3) in
          let got =
            Direction.reciprocal ~within:(radians (scalar 0.))
              (icrs_rows [ p; p; q ]) (icrs_rows [ p; q; q ])
          in
          equal matches [ (0, 0, 0.); (2, 1, 0.) ] (triples got));
      test "at radius pi, antipodal rows pair" (fun () ->
          let poles within =
            triples
              (Direction.reciprocal ~within:(radians (scalar within))
                 (icrs_rows [ north ]) (icrs_rows [ south ]))
          in
          equal matches [ (0, 0, pi) ] (poles pi);
          equal matches [] (poles (Float.pred pi));
          (* Every pair of rows is within pi, and the rows 0.1 apart are
             each other's nearest. *)
          let i, j, s =
            Direction.reciprocal ~within:(radians (scalar pi))
              (icrs_rows [ north; unit_vector 0. 0. ])
              (icrs_rows [ south; unit_vector 0.1 0. ])
          in
          equal (pair (array int64) (array int64)) ([| 1L |], [| 1L |])
            (Nx.to_array i, Nx.to_array j);
          equal (array (float 1e-15)) [| 0.1 |] (Nx.to_array (in_radians s)));
      test "raises as pairs does, naming reciprocal" (fun () ->
          let d = icrs_rows [ north ] in
          let raises within a b =
            raises_match (Exn.invalid_arg ~sub:"Direction.reciprocal")
              (fun () ->
                Direction.reciprocal ~within:(radians (scalar within)) a b)
          in
          raises (-1.) d d;
          raises nan d d;
          raises 1. (lonlat 1. 0.2) d;
          raises 1. d (icrs (vectors [| infinity; 0.; 0. |])));
    ]

(* Structure *)

let structure =
  group "A direction walks its frame, then its vectors"
    [
      test "visits" (fun () ->
          let d = Direction.rotate Frame.galactic (lonlat 1. 0.2) in
          List.iter
            (fun v -> Format.printf "%a@." Nx.Ptree.pp_visit v)
            (Nx.Ptree.visits (Direction.ptree ()) d);
          expect (output ())
          @@ __POS_OF__
               {|
            the root: case "galactic"
            the root: a leaf
            |});
      test "cast keeps the vectors float64" (fun () ->
          let d = lonlat 1. 0.2 in
          let module D = struct
            type (_, 'd) t = (Frame.icrs, 'd) Direction.t

            let walk = Direction.walk
          end in
          let d' = Nx.Ptree.cast (module D) Nx.float32 d in
          equal bits (xyz d) (xyz d'));
      test "frame is the frame" (fun () ->
          equal string "supergalactic"
            (Frame.name
               (Direction.frame
                  (Direction.rotate Frame.supergalactic (lonlat 1. 0.2)))));
    ]

let () =
  exit
    (run "Direction"
       [
         constructors;
         round_trips;
         stated;
         reads_the_ray;
         measures;
         rotations;
         gradients;
         structure;
         crossmatch;
         reciprocity;
       ])
