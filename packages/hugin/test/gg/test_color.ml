(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg

(* [Color.pp] rounds to 8 bits, so the witnesses print every digit. *)
let pp_color ppf c =
  Format.fprintf ppf "(v ~alpha:%.17g %.17g %.17g %.17g)" (Color.alpha c)
    (Color.r c) (Color.g c) (Color.b c)

let color =
  Testable.with_compare Color.compare
    (Testable.make ~pp:pp_color ~equal:Color.equal)

let color_near eps =
  Testable.make ~pp:pp_color ~equal:(fun c c' ->
      let close x y = Float.abs (x -. y) <= eps in
      close (Color.r c) (Color.r c')
      && close (Color.g c) (Color.g c')
      && close (Color.b c) (Color.b c')
      && close (Color.alpha c) (Color.alpha c'))

let triple_near eps = triple (float eps) (float eps) (float eps)
let unit = Gen.float_range 0. 1.

let gen_color =
  Gen.with_pp pp_color
    (Gen.map
       (fun ((r, g, b), alpha) -> Color.v ~alpha r g b)
       (Gen.pair (Gen.triple unit unit unit) unit))

let gen_opaque =
  Gen.with_pp pp_color
    (Gen.map (fun (r, g, b) -> Color.v r g b) (Gen.triple unit unit unit))

(* Colours whose components are multiples of 1/255, the values of hex. *)
let gen_byte_color =
  let byte = Gen.map (fun n -> Float.of_int n /. 255.) (Gen.int_range 0 255) in
  Gen.with_pp pp_color
    (Gen.map
       (fun ((r, g, b), alpha) -> Color.v ~alpha r g b)
       (Gen.pair (Gen.triple byte byte byte) byte))

(* Colours from small sets, so that equal values are drawn. *)
let gen_small_color =
  let c = Gen.of_list [ 0.; 0.5; 1. ] in
  Gen.with_pp pp_color
    (Gen.map
       (fun ((r, g, b), alpha) -> Color.v ~alpha r g b)
       (Gen.pair (Gen.triple c c c) c))

(* Constructors *)

let constants =
  [
    ("black", Color.black, (0., 0., 0., 1.));
    ("white", Color.white, (1., 1., 1., 1.));
    ("red", Color.red, (1., 0., 0., 1.));
    ("green", Color.green, (0., 1., 0., 1.));
    ("blue", Color.blue, (0., 0., 1., 1.));
    ("transparent", Color.transparent, (0., 0., 0., 0.));
  ]

let out_of_range =
  [
    ("a negative red", fun () -> Color.v (-0.1) 0. 0.);
    ("a green above 1", fun () -> Color.v 0. 1.1 0.);
    ("a NaN blue", fun () -> Color.v 0. 0. Float.nan);
    ("an alpha above 1", fun () -> Color.v ~alpha:2. 0. 0. 0.);
    ("a gray above 1", fun () -> Color.gray 1.5);
    ("with_alpha below 0", fun () -> Color.with_alpha (-1.) Color.red);
    ("with_alpha NaN", fun () -> Color.with_alpha Float.nan Color.red);
  ]

let constructors =
  group "constructors"
    [
      prop "v keeps its components"
        (Gen.pair (Gen.triple unit unit unit) unit)
        (fun ((r, g, b), alpha) ->
          let c = Color.v ~alpha r g b in
          equal (list float_exact) [ r; g; b; alpha ]
            [ Color.r c; Color.g c; Color.b c; Color.alpha c ]);
      test "v is opaque by default" (fun () ->
          equal float_exact 1. (Color.alpha (Color.v 0.2 0.3 0.4)));
      test "gray l is v l l l" (fun () ->
          equal color
            (Color.v ~alpha:0.5 0.3 0.3 0.3)
            (Color.gray ~alpha:0.5 0.3));
      cases
        ~name:(fun (n, _, _) -> n)
        "constant" constants
        (fun (_, c, (r, g, b, a)) -> equal color (Color.v ~alpha:a r g b) c);
      prop "with_alpha replaces the opacity only" (Gen.pair gen_color unit)
        (fun (c, a) ->
          equal color
            (Color.v ~alpha:a (Color.r c) (Color.g c) (Color.b c))
            (Color.with_alpha a c));
      cases ~name:fst "raises on" out_of_range (fun (_, f) ->
          raises_match Exn.invalid_arg f);
    ]

(* Oklab and Oklch *)

let deg d = d *. Float.pi /. 180.

(* Reference coordinates from CSS Color 4's conversions. *)
let primaries =
  [
    ( "red",
      Color.red,
      (0.627955, 0.224863, 0.125846),
      (0.627955, 0.257683, deg 29.2339) );
    ( "green",
      Color.green,
      (0.866440, -0.233888, 0.179498),
      (0.866440, 0.294827, deg 142.4953) );
    ( "blue",
      Color.blue,
      (0.452014, -0.032457, -0.311528),
      (0.452014, 0.313214, deg 264.0520) );
  ]

let greys = [ "#000000"; "#010101"; "#808080"; "#fefefe"; "#ffffff"; "#777" ]
let hex s = match Color.of_hex s with Ok c -> c | Error e -> failwith e

let powerless_hue s =
  let _, c, h = Color.to_oklch (hex s) in
  less float_exact ~than:4e-6 c;
  equal float_exact Float.nan h

let gamut_cases =
  [
    ("a lightness of 1 is white", (1., 0.1, 0.1), Color.white);
    ("a lightness above 1 is white", (1.5, -0.3, 0.), Color.white);
    ("a lightness of 0 is black", (0., 0.2, 0.2), Color.black);
    ("a negative lightness is black", (-1., 0., 0.), Color.black);
  ]

(* The distance in Oklab from [c] to the segment of chroma [0, chroma] at
   lightness [l] and hue [h]. *)
let distance_to_ray c (l, chroma, h) =
  let l', a, b = Color.to_oklab c in
  let t =
    Float.min chroma (Float.max 0. ((a *. Float.cos h) +. (b *. Float.sin h)))
  in
  Float.sqrt
    (((l' -. l) ** 2.)
    +. ((a -. (t *. Float.cos h)) ** 2.)
    +. ((b -. (t *. Float.sin h)) ** 2.))

let gen_oklch =
  Gen.triple
    (Gen.float_range 0.02 0.98)
    (Gen.float_range 0. 0.5) (Gen.float_range 0. 6.28)

(* Chromas up to [0.5] and, now and then, up to [1e300], where the cubes of the
   conversion to linear sRGB overflow. *)
let gen_wide_oklch =
  Gen.triple
    (Gen.float_range 0.02 0.98)
    (Gen.frequency
       [
         (6, Gen.float_range 0. 0.5);
         (1, Gen.map (fun e -> 10. ** e) (Gen.float_range 0. 300.));
       ])
    (Gen.float_range 0. 6.28)

let in_gamut c =
  List.iter
    (fun x ->
      at_least float_exact ~than:0. x;
      at_most float_exact ~than:1. x)
    [ Color.r c; Color.g c; Color.b c ]

let gamut_preserves_hue (l, c, h) =
  let mapped = Color.of_oklch l c h in
  in_gamut mapped;
  at_most ~msg:"distance to the constant lightness and hue segment" float_exact
    ~than:0.021
    (distance_to_ray mapped (l, c, h))

(* Inputs and results of CSS Color 4's gamut mapping as ColorAide 4 implements
   it, in Oklch with hues in degrees. The second one ends its search on the
   chroma range, not on the distance. *)
let css_gamut_cases =
  [
    ( (0.7, 0.4, 30.),
      (0.68310161672924008, 0.20619576749445087, 30.181628135756416) );
    ( (0.22497252961306358, 0.4054886027556328, 77.8037887761551),
      (0.22664060715005102, 0.050903995374624429, 64.863328693935273) );
    ( (0.9, 0.3, 250.),
      (0.89496974724299738, 0.057657837114716892, 239.02308312139809) );
    ( (0.3, 0.35, 140.),
      (0.30388195644888649, 0.1034032516478345, 142.49534504144381) );
  ]

let css_gamut_example ((l, c, h), (l', c', h')) =
  equal (triple_near 1e-9)
    (l', c', deg h')
    (Color.to_oklch (Color.of_oklch l c (deg h)))

let invalid_oklab =
  [
    ("of_oklab with a NaN lightness", fun () -> Color.of_oklab Float.nan 0. 0.);
    ("of_oklab with an infinite a", fun () -> Color.of_oklab 0.5 infinity 0.);
    ( "of_oklab with an alpha below 0",
      fun () -> Color.of_oklab ~alpha:(-0.1) 0.5 0. 0. );
    ("of_oklch with a negative chroma", fun () -> Color.of_oklch 0.5 (-0.1) 0.);
    ( "of_oklch with an infinite chroma",
      fun () -> Color.of_oklch 0.5 infinity 0. );
    ( "of_oklch with an infinite hue",
      fun () -> Color.of_oklch 0.5 0.1 neg_infinity );
    ("of_oklch with a NaN lightness", fun () -> Color.of_oklch Float.nan 0.1 0.);
    ( "of_oklch with an alpha above 1",
      fun () -> Color.of_oklch ~alpha:1.5 0.5 0.1 0. );
  ]

let oklab =
  group "Oklab"
    [
      cases
        ~name:(fun (n, _, _, _) -> n)
        "to_oklab and to_oklch of the primary" primaries
        (fun (_, c, lab, lch) ->
          equal (triple_near 1e-5) lab (Color.to_oklab c);
          equal (triple_near 1e-5) lch (Color.to_oklch c));
      test "to_oklab of white is lightness 1 without chroma" (fun () ->
          equal (triple_near 1e-7) (1., 0., 0.) (Color.to_oklab Color.white));
      cases ~name:Fun.id "to_oklch gives a powerless hue for the grey" greys
        powerless_hue;
      test "to_oklch gives a hue to the least chromatic 8-bit colour" (fun () ->
          let _, c, h = Color.to_oklch (hex "#808081") in
          greater float_exact ~than:4e-6 c;
          is_true (Float.is_finite h));
      prop "to_oklch is the polar form of to_oklab, hue in [0, 2pi[" gen_color
        (fun c ->
          let l, a, b = Color.to_oklab c and l', chroma, h = Color.to_oklch c in
          equal float_exact l l';
          equal float_exact (Float.hypot a b) chroma;
          if Float.is_nan h then less float_exact ~than:4e-6 chroma
          else begin
            at_least float_exact ~than:0. h;
            less float_exact ~than:(2. *. Float.pi) h;
            equal (float 1e-9) a (chroma *. Float.cos h);
            equal (float 1e-9) b (chroma *. Float.sin h)
          end);
      prop "of_oklab inverts to_oklab on the gamut" gen_opaque
        (Law.round_trip (color_near 1e-6)
           (triple float_exact float_exact float_exact) Color.to_oklab
           (fun (l, a, b) -> Color.of_oklab l a b));
      prop "to_oklab inverts of_oklab on coordinates of the gamut" gen_opaque
        (fun c ->
          let l, a, b = Color.to_oklab c in
          equal (triple_near 1e-6) (l, a, b)
            (Color.to_oklab (Color.of_oklab l a b)));
      prop "of_oklab keeps its alpha" (Gen.pair gen_oklch unit)
        (fun ((l, a, b), alpha) ->
          equal float_exact alpha (Color.alpha (Color.of_oklab ~alpha l a b)));
      prop "of_oklch is of_oklab of the Cartesian coordinates" gen_oklch
        (fun (l, c, h) ->
          equal color
            (Color.of_oklab l (c *. Float.cos h) (c *. Float.sin h))
            (Color.of_oklch l c h));
      test "of_oklch takes a NaN hue as 0" (fun () ->
          equal color
            (Color.of_oklch 0.6 0.1 0.)
            (Color.of_oklch 0.6 0.1 Float.nan));
      cases
        ~name:(fun (n, _, _) -> n)
        "gamut mapping" gamut_cases
        (fun (_, (l, a, b), c) ->
          equal color c (Color.of_oklab l a b);
          equal color (Color.with_alpha 0.25 c)
            (Color.of_oklab ~alpha:0.25 l a b));
      prop "gamut mapping lands within a JND of the lightness and hue"
        gen_wide_oklch
        ~examples:
          [
            (0.22497252961306358, 0.4054886027556328, 1.357932284670116);
            (0.5, 1e120, 1.);
            (0.5, 1e200, 1.);
          ]
        gamut_preserves_hue;
      test "of_oklab maps a huge chroma into the gamut" (fun () ->
          in_gamut (Color.of_oklab 0.5 1e300 0.);
          in_gamut (Color.of_oklab 0.5 (-1e300) 1e300));
      test "of_oklch maps a huge chroma into the gamut" (fun () ->
          in_gamut (Color.of_oklch 0.5 1e120 1.);
          in_gamut (Color.of_oklch 0.5 1e120 0.));
      cases
        ~name:(fun ((l, c, h), _) -> Printf.sprintf "oklch(%g %g %gdeg)" l c h)
        "gamut mapping agrees with CSS Color 4 on" css_gamut_cases
        css_gamut_example;
      cases ~name:fst "raises on" invalid_oklab (fun (_, f) ->
          raises_match Exn.invalid_arg f);
    ]

(* Mixing and contrast *)

let hue c =
  let _, _, h = Color.to_oklch c in
  h

let mix_tests =
  group "mix and contrast"
    [
      prop "mix 0 is the first colour, mix 1 the second"
        (Gen.pair gen_opaque gen_opaque) (fun (c, c') ->
          equal (color_near 1e-6) c (Color.mix 0. c c');
          equal (color_near 1e-6) c' (Color.mix 1. c c'));
      prop "mix t c c' is mix (1 - t) c' c"
        (Gen.triple
           (Gen.map (fun k -> Float.of_int k /. 16.) (Gen.int_range 0 16))
           gen_color gen_color)
        (fun (t, c, c') ->
          equal (color_near 1e-12) (Color.mix t c c') (Color.mix (1. -. t) c' c));
      prop "mix interpolates opacity linearly"
        (Gen.triple unit gen_color gen_color) (fun (t, c, c') ->
          equal (float 1e-12)
            (((1. -. t) *. Color.alpha c) +. (t *. Color.alpha c'))
            (Color.alpha (Color.mix t c c')));
      cases ~name:(Printf.sprintf "t = %g")
        "mix of white and blue stays within a JND of blue's hue"
        [ 0.1; 0.25; 0.5; 0.75; 1. ] (fun t ->
          let l, a, b = Color.to_oklab Color.white in
          let l', a', b' = Color.to_oklab Color.blue in
          let at x x' = ((1. -. t) *. x) +. (t *. x') in
          let mixed =
            (at l l', Float.hypot (at a a') (at b b'), hue Color.blue)
          in
          at_most float_exact ~than:0.021
            (distance_to_ray (Color.mix t Color.white Color.blue) mixed));
      test "mix with a transparent end adds opacity and no colour" (fun () ->
          equal (color_near 1e-6)
            (Color.with_alpha 0.5 Color.blue)
            (Color.mix 0.5 Color.transparent Color.blue));
      test "mix of transparent colours is transparent" (fun () ->
          equal color Color.transparent
            (Color.mix 0.3
               (Color.with_alpha 0. Color.red)
               (Color.with_alpha 0. Color.blue)));
      cases ~name:(Printf.sprintf "t = %g") "mix raises on"
        [ -0.1; 1.1; Float.nan ] (fun t ->
          raises_match Exn.invalid_arg (fun () ->
              Color.mix t Color.red Color.blue));
      cases ~name:fst "contrast is black on"
        [
          ("white", Color.white);
          ("yellow", hex "#ffff00");
          ("#767676", hex "#767676");
          ("a transparent white", Color.with_alpha 0. Color.white);
        ]
        (fun (_, c) -> equal color Color.black (Color.contrast c));
      cases ~name:fst "contrast is white on"
        [
          ("black", Color.black);
          ("blue", Color.blue);
          ("#757575", hex "#757575");
        ]
        (fun (_, c) -> equal color Color.white (Color.contrast c));
    ]

(* Hexadecimal notation *)

let parsed =
  [
    ("#fff", Color.white);
    ("#FFF", Color.white);
    ("#0000", Color.transparent);
    ("#1f77b4", Color.v (31. /. 255.) (119. /. 255.) (180. /. 255.));
    ( "#1F77B480",
      Color.v ~alpha:(128. /. 255.) (31. /. 255.) (119. /. 255.) (180. /. 255.)
    );
    ("#a0b", Color.v (170. /. 255.) 0. (187. /. 255.));
  ]

(* Texts of_hex rejects, and a part of the reason it gives. *)
let rejected =
  [
    ("", "starts with '#'");
    ("fff", "starts with '#'");
    (" #fff", "starts with '#'");
    ("#", "found 0");
    ("#ff", "found 2");
    ("#fffff", "found 5");
    ("#fffffff", "found 7");
    ("#ggg", "'g' is not a hex digit");
    ("#12345g", "'g' is not a hex digit");
    ("##fff", "'#' is not a hex digit");
  ]

let gen_canonical_hex =
  let digit = Gen.of_list [ '0'; '1'; '7'; '9'; 'a'; 'c'; 'f' ] in
  Gen.map
    (fun (rgb, a) ->
      let s = String.of_seq (List.to_seq rgb) in
      match a with
      | Some [ 'f'; 'f' ] | None -> "#" ^ s
      | Some a -> "#" ^ s ^ String.of_seq (List.to_seq a))
    (Gen.pair
       (Gen.list ~size:(Gen.constant 6) digit)
       (Gen.option (Gen.list ~size:(Gen.constant 2) digit)))

let hex_tests =
  group "hex"
    [
      cases ~name:fst "of_hex reads" parsed (fun (s, c) ->
          equal (result color string) (Ok c) (Color.of_hex s));
      cases
        ~name:(fun (s, _) -> Printf.sprintf "%S" s)
        "of_hex rejects" rejected
        (fun (s, why) ->
          contains ~sub:why (require_error ~pp:pp_color (Color.of_hex s)));
      test "to_hex writes lowercase and omits an opaque alpha" (fun () ->
          equal string "#1f77b4" (Color.to_hex (hex "#1F77B4"));
          equal string "#1f77b480" (Color.to_hex (hex "#1F77B480")));
      test "to_hex rounds halves away from zero" (fun () ->
          equal string "#00000080" (Color.to_hex (Color.v ~alpha:0.5 0. 0. 0.)));
      prop "of_hex inverts to_hex on multiples of 1/255" gen_byte_color
        (Law.round_trip color string Color.to_hex (fun s ->
             require_ok (Color.of_hex s)));
      prop "to_hex inverts of_hex on canonical text" gen_canonical_hex
        (Law.round_trip string color
           (fun s -> require_ok (Color.of_hex s))
           Color.to_hex);
      test "pp formats to_hex" (fun () ->
          equal string "#1f77b480"
            (Format.asprintf "%a" Color.pp (hex "#1f77b480")));
    ]

let comparing =
  group "comparing"
    [
      prop "equal is an equivalence"
        (Gen.pair gen_small_color gen_small_color)
        (Law.equivalence color);
      prop "compare is a total order compatible with equal"
        (Gen.triple gen_small_color gen_small_color gen_small_color)
        (Law.order color);
      test "compare is lexicographic by red, green, blue and alpha" (fun () ->
          let ordered =
            [
              Color.v ~alpha:0.5 0. 0. 0.;
              Color.black;
              Color.blue;
              Color.green;
              Color.red;
            ]
          in
          equal (list color) ordered
            (List.sort Color.compare (List.rev ordered)));
    ]

let () =
  exit
    (run "hugin.gg color"
       [ constructors; oklab; mix_tests; hex_tests; comparing ])
