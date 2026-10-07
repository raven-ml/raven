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

(* [exact_multiple k (x, y, z)] is [k] times the row when no component
   rounds. *)
let exact_multiple k (x, y, z) =
  let exact c = Float.is_finite (c *. k) && c *. k /. k = c in
  assume (exact x && exact y && exact z);
  (x *. k, y *. k, z *. k)

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
        Gen.(pair rows (of_list [ 0x1p-500; 0x1p500 ]))
        (fun (r, k) ->
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
          raises_match (Exn.invalid_arg ~substring:"Direction.of_xyz")
            (fun () ->
              Direction.of_xyz Frame.icrs (Nx.ones Nx.float64 [| 2; 4 |])));
    ]

(* Readers *)

let lon_lat d = (one (lon d), one (lat d))

let round_trips =
  group "lon and lat invert lonlat"
    [
      prop "round trip within 4 ulp" angles (fun (l, b) ->
          cover "near a pole" (Float.abs b > 1.5);
          let l', b' = lon_lat (lonlat l b) in
          equal (ulps 4) l l';
          equal (ulps 4) b b');
      cases
        ~name:(fun (n, _, _, _) -> n)
        "stated angles"
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
        ~name:(fun (n, _) -> n)
        "an infinite component raises"
        [
          ("lon", fun d -> ignore (Direction.lon d));
          ("lat", fun d -> ignore (Direction.lat d));
          ("separation", fun d -> ignore (Direction.separation (lonlat 0. 0.) d));
          ( "position_angle",
            fun d -> ignore (Direction.position_angle d (lonlat 0. 0.)) );
        ]
        (fun (n, f) ->
          raises_match
            (Exn.invalid_arg ~substring:("Direction." ^ n))
            (fun () -> f (icrs (vector 1. neg_infinity 0.))));
    ]

let reads_the_ray =
  let powers = Gen.of_list [ 0x1p-500; 0x1p500 ] in
  let reads (r, k) f =
    let k_r = exact_multiple k r in
    equal bits (f (icrs (row r))) (f (icrs (row k_r)))
  in
  let other = lonlat 0.7 (-0.2) in
  group "Readers read the ray"
    [
      prop "lon and lat: a power-of-two multiple reads the same bits"
        (Gen.pair rows powers) (fun c ->
          reads c lon;
          reads c lat);
      prop "separation and position_angle: the same, on either side"
        (Gen.pair rows powers) (fun c ->
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

let measures =
  group "Separation and position angle"
    [
      cases
        ~name:(fun (n, _, _, _, _) -> n)
        "against a 60-digit reference" Reference.pairs
        (fun (_, a, b, sep, pa) ->
          let a = icrs (Nx.create Nx.float64 [| 3 |] a)
          and b = icrs (Nx.create Nx.float64 [| 3 |] b) in
          equal (float_rel ~rel:1e-15 ~abs:0.) sep (one (separation a b));
          equal (float 1e-15) pa (one (position_angle a b)));
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
      prop "separation is symmetric within 2 ulp" (Gen.pair plain plain)
        (fun (a, b) ->
          let a = icrs (row a) and b = icrs (row b) in
          equal (ulps 2) (one (separation a b)) (one (separation b a)));
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
        ~name:(fun (n, _, _) -> n)
        "exact 0 and no NaN on the singular sets"
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
            type 'a t = Frame.icrs Direction.t

            let walk c d = Nx.Ptree.Walk.structure (Direction.ptree ()) c d
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
       ])
