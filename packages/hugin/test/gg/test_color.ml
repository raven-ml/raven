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

let components c = [ Color.r c; Color.g c; Color.b c; Color.alpha c ]

(* Witnesses within an absolute tolerance, which report the largest error they
   saw. *)
let color_within eps = Testable.contramap components (list (float eps))
let triple_within eps = triple (float eps) (float eps) (float eps)
let unit = Gen.float_range 0. 1.
let hex s = require_ok (Color.of_hex s)

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
    ("black is v 0 0 0", Color.black, (0., 0., 0., 1.));
    ("white is v 1 1 1", Color.white, (1., 1., 1., 1.));
    ("red is v 1 0 0", Color.red, (1., 0., 0., 1.));
    ("green is v 0 1 0", Color.green, (0., 1., 0., 1.));
    ("blue is v 0 0 1", Color.blue, (0., 0., 1., 1.));
    ("transparent is v ~alpha:0 0 0 0", Color.transparent, (0., 0., 0., 0.));
  ]

let constructors =
  group "constructors"
    [
      prop "v keeps its components"
        (Gen.pair (Gen.triple unit unit unit) unit)
        (fun ((r, g, b), alpha) ->
          equal (list float_exact) [ r; g; b; alpha ]
            (components (Color.v ~alpha r g b)));
      test "v is opaque by default" (fun () ->
          equal float_exact 1. (Color.alpha (Color.v 0.2 0.3 0.4)));
      cases "v raises on" ~id:fst
        [
          ("a negative red", (-0.1, 0., 0., 1.));
          ("a green above 1", (0., 1.1, 0., 1.));
          ("a NaN blue", (0., 0., Float.nan, 1.));
          ("an alpha above 1", (0., 0., 0., 2.));
          ("a NaN alpha", (0., 0., 0., Float.nan));
        ]
        (fun (_, (r, g, b, alpha)) ->
          raises_match Exn.invalid_arg (fun () -> Color.v ~alpha r g b));
      prop "gray l is v l l l"
        (Gen.pair unit (Gen.option unit))
        (fun (l, alpha) ->
          equal color (Color.v ?alpha l l l) (Color.gray ?alpha l));
      test "gray raises on a level above 1" (fun () ->
          raises_match Exn.invalid_arg (fun () -> Color.gray 1.5));
      cases "the constants are as stated"
        ~id:(fun (n, _, _) -> n)
        constants
        (fun (_, c, (r, g, b, a)) -> equal color (Color.v ~alpha:a r g b) c);
      prop "with_alpha replaces the opacity only" (Gen.pair gen_color unit)
        (fun (c, a) ->
          equal color
            (Color.v ~alpha:a (Color.r c) (Color.g c) (Color.b c))
            (Color.with_alpha a c));
      cases "with_alpha raises on" ~id:(Printf.sprintf "%g") [ -1.; Float.nan ]
        (fun a ->
          raises_match Exn.invalid_arg (fun () -> Color.with_alpha a Color.red));
    ]

(* Oklab and Oklch *)

let deg d = d *. Float.pi /. 180.

(* Coordinates from CSS Color 4's conversions, given to six decimals and hues to
   four decimals of a degree: within 5e-7, and 8.8e-7 radians. *)
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

let byte n = Float.of_int n /. 255.

(* Greys of 8 bits a channel, whose hue is powerless, and the least chromatic
   colour of 8 bits a channel, [#808081]. *)
let hue_examples =
  List.map (fun n -> Color.gray (byte n)) [ 0; 1; 0x77; 0x80; 0xfe; 0xff ]
  @ [ Color.v (byte 0x80) (byte 0x80) (byte 0x81) ]

let hue_is_nan_iff_grey c =
  let _, chroma, h = Color.to_oklch c in
  let powerless = chroma < 4e-6 in
  cover "a powerless hue" powerless;
  cover "a hue" (not powerless);
  if powerless then equal float_exact Float.nan h
  else satisfies ~claim:"a finite hue" float_exact Float.is_finite h

(* [h] rounds within an ulp of 2 pi, 8.9e-16, and the chroma is below 0.4: its
   Cartesian coordinates lie within 1e-15 of [a] and [b]. *)
let polar_form c =
  let l, a, b = Color.to_oklab c and l', chroma, h = Color.to_oklch c in
  equal float_exact l l';
  equal float_exact (Float.hypot a b) chroma;
  if not (Float.is_nan h) then
    equal
      (pair (float 1e-15) (float 1e-15))
      (a, b)
      (chroma *. Float.cos h, chroma *. Float.sin h)

let hue_in_range c =
  let _, _, h = Color.to_oklch c in
  let _, _, b = Color.to_oklab c in
  cover "a hue below the x axis" (b < 0. && not (Float.is_nan h));
  if not (Float.is_nan h) then begin
    at_least float_exact ~than:0. h;
    less float_exact ~than:(2. *. Float.pi) h
  end

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

(* Lightnesses through black and white, where of_oklab takes its other paths. *)
let gen_lab =
  Gen.triple
    (Gen.float_range (-0.5) 1.5)
    (Gen.float_range (-0.5) 0.5)
    (Gen.float_range (-0.5) 0.5)

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

(* The gamut section's bound: clipping moves the colour by less than 0.02 from
   the colour at the reduced chroma, a point of the segment. *)
let jnd_bound = 0.02

let gamut_preserves_hue (l, c, h) =
  let mapped = Color.of_oklch l c h in
  in_gamut mapped;
  at_most ~msg:"distance to the constant lightness and hue segment" float_exact
    ~than:jnd_bound
    (distance_to_ray mapped (l, c, h))

(* Inputs and results of CSS Color 4's gamut mapping as ColorAide 4 implements
   it, in Oklch with hues in degrees. The second one ends its search on the
   chroma range, not on the distance. ColorAide makes the same decisions in
   binary64 and rounds in another order, which the conversions carry to about
   1e-14 here. *)
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

let oklab =
  group "Oklab"
    [
      cases "to_oklab of a primary is CSS Color 4's"
        ~id:(fun (n, _, _, _) -> n)
        primaries
        (fun (_, c, lab, _) ->
          equal (triple_within 5e-7) lab (Color.to_oklab c));
      (* Oklab puts white at lightness 1 without chroma; the conversion rounds a
         few times within an ulp of 1. *)
      test "to_oklab of white is lightness 1 without chroma" (fun () ->
          equal (triple_within 1e-15) (1., 0., 0.) (Color.to_oklab Color.white));
      cases "to_oklch of a primary is CSS Color 4's"
        ~id:(fun (n, _, _, _) -> n)
        primaries
        (fun (_, c, _, lch) ->
          equal
            (triple (float 5e-7) (float 5e-7) (float 8.8e-7))
            lch (Color.to_oklch c));
      prop "to_oklch is the polar form of to_oklab" gen_color polar_form;
      prop "to_oklch gives a hue in [0, 2 pi[" gen_color hue_in_range;
      prop "to_oklch gives a NaN hue iff the chroma is below 4e-6"
        ~examples:hue_examples gen_color hue_is_nan_iff_grey;
      (* Encoding multiplies an error of a linear component by at most 12.92,
         the slope of the sRGB curve at zero; the round trip rounds each linear
         component within a few ulps of 1: within 1e-13. *)
      prop "of_oklab inverts to_oklab on the gamut" gen_opaque
        (Law.round_trip (color_within 1e-13)
           (triple float_exact float_exact float_exact) Color.to_oklab
           (fun (l, a, b) -> Color.of_oklab l a b));
      prop "to_oklab inverts of_oklab on coordinates of the gamut" gen_opaque
        (fun c ->
          let l, a, b = Color.to_oklab c in
          equal (triple_within 1e-13) (l, a, b)
            (Color.to_oklab (Color.of_oklab l a b)));
      prop "of_oklab keeps its alpha" (Gen.pair gen_lab unit)
        (fun ((l, a, b), alpha) ->
          cover "white" (l >= 1.);
          cover "black" (l <= 0.);
          equal float_exact alpha (Color.alpha (Color.of_oklab ~alpha l a b)));
      prop "of_oklch is of_oklab of the Cartesian coordinates"
        (Gen.pair gen_oklch unit) (fun ((l, c, h), alpha) ->
          equal color
            (Color.of_oklab ~alpha l (c *. Float.cos h) (c *. Float.sin h))
            (Color.of_oklch ~alpha l c h));
      test "of_oklch takes a NaN hue as 0" (fun () ->
          equal color
            (Color.of_oklch 0.6 0.1 0.)
            (Color.of_oklch 0.6 0.1 Float.nan));
      cases "of_oklab maps"
        ~id:(fun (n, _, _) -> n)
        gamut_cases
        (fun (_, (l, a, b), c) -> equal color c (Color.of_oklab l a b));
      prop "of_oklch lands within a JND of its lightness and hue" gen_wide_oklch
        ~examples:
          [
            (0.22497252961306358, 0.4054886027556328, 1.357932284670116);
            (0.5, 1e120, 0.);
            (0.5, 1e120, 1.);
            (0.5, 1e200, 1.);
            (0.5, 1e300, 0.);
            (0.5, 1.4142135623730951e300, 2.356194490192345);
          ]
        gamut_preserves_hue;
      cases "of_oklch agrees with CSS Color 4 on"
        ~id:(fun ((l, c, h), _) -> Printf.sprintf "oklch(%g %g %gdeg)" l c h)
        css_gamut_cases
        (fun ((l, c, h), (l', c', h')) ->
          equal (triple_within 1e-12)
            (l', c', deg h')
            (Color.to_oklch (Color.of_oklch l c (deg h))));
      cases "of_oklab raises on" ~id:fst
        [
          ("a NaN lightness", fun () -> Color.of_oklab Float.nan 0. 0.);
          ("an infinite a", fun () -> Color.of_oklab 0.5 infinity 0.);
          ("an alpha below 0", fun () -> Color.of_oklab ~alpha:(-0.1) 0.5 0. 0.);
        ]
        (fun (_, f) -> raises_match Exn.invalid_arg f);
      cases "of_oklch raises on" ~id:fst
        [
          ("a negative chroma", fun () -> Color.of_oklch 0.5 (-0.1) 0.);
          ("an infinite chroma", fun () -> Color.of_oklch 0.5 infinity 0.);
          ("an infinite hue", fun () -> Color.of_oklch 0.5 0.1 neg_infinity);
          ("a NaN lightness", fun () -> Color.of_oklch Float.nan 0.1 0.);
          ("an alpha above 1", fun () -> Color.of_oklch ~alpha:1.5 0.5 0.1 0.);
        ]
        (fun (_, f) -> raises_match Exn.invalid_arg f);
    ]

(* Mixing and contrast *)

(* The .mli's formula: opacity [(1 - t) αc + t αc'], each Oklab coordinate the
   average weighted by those terms, gamut mapped. *)
let premultiplied_mix t c c' =
  let w = (1. -. t) *. Color.alpha c and w' = t *. Color.alpha c' in
  let alpha = w +. w' in
  if alpha = 0. then Color.transparent
  else
    let l, a, b = Color.to_oklab c and l', a', b' = Color.to_oklab c' in
    let avg x x' = ((w *. x) +. (w' *. x')) /. alpha in
    Color.of_oklab ~alpha:(Float.min 1. alpha) (avg l l') (avg a a') (avg b b')

(* Both sides compute the same formula, whose roundings, within an ulp of 1, the
   conversion to encoded sRGB multiplies by at most a few hundred. *)
let mix_is_average (t, c, c') =
  cover "an end of the mix" (t = 0. || t = 1.);
  cover "a transparent end" (Color.alpha c = 0. || Color.alpha c' = 0.);
  cover "a zero opacity"
    (((1. -. t) *. Color.alpha c) +. (t *. Color.alpha c') = 0.);
  equal (color_within 1e-12) (premultiplied_mix t c c') (Color.mix t c c')

let hue c =
  let _, _, h = Color.to_oklch c in
  h

(* Rows of the WCAG 2 threshold: the neighbours [#767676] and [#757575] have
   relative luminances 0.1812 and 0.1779 around 0.179. *)
let contrasts =
  [
    ("white", "#ffffff", Color.black);
    ("yellow", "#ffff00", Color.black);
    ("#767676", "#767676", Color.black);
    ("black", "#000000", Color.white);
    ("blue", "#0000ff", Color.white);
    ("#757575", "#757575", Color.white);
  ]

let mix_tests =
  group "mix and contrast"
    [
      prop "mix is the premultiplied average in Oklab"
        ~examples:
          [
            (0., Color.red, Color.blue);
            (1., Color.red, Color.blue);
            (0.5, Color.transparent, Color.blue);
            (0.3, Color.with_alpha 0. Color.red, Color.with_alpha 0. Color.blue);
          ]
        (Gen.triple unit gen_color gen_color)
        mix_is_average;
      cases "mix of white and blue stays within a JND of blue's hue"
        ~id:(Printf.sprintf "t = %g") [ 0.1; 0.25; 0.5; 0.75; 1. ] (fun t ->
          let l, a, b = Color.to_oklab Color.white in
          let l', a', b' = Color.to_oklab Color.blue in
          let at x x' = ((1. -. t) *. x) +. (t *. x') in
          let mixed =
            (at l l', Float.hypot (at a a') (at b b'), hue Color.blue)
          in
          at_most float_exact ~than:jnd_bound
            (distance_to_ray (Color.mix t Color.white Color.blue) mixed));
      cases "mix raises on" ~id:(Printf.sprintf "t = %g")
        [ -0.1; 1.1; Float.nan ] (fun t ->
          raises_match Exn.invalid_arg (fun () ->
              Color.mix t Color.red Color.blue));
      cases "contrast is black iff the relative luminance is at least 0.179"
        ~id:(fun (n, _, _) -> n)
        contrasts
        (fun (_, s, expected) -> equal color expected (Color.contrast (hex s)));
      prop "contrast ignores the alpha" (Gen.pair gen_color unit) (fun (c, a) ->
          Law.ignores color color Color.contrast (Color.with_alpha a) c);
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

let rejected =
  [
    "";
    "fff";
    " #fff";
    "#";
    "#ff";
    "#fffff";
    "#fffffff";
    "#ggg";
    "#12345g";
    "##fff";
  ]

let reasons () =
  let why s = require_error ~pp:pp_color (Color.of_hex s) in
  expect (String.concat "\n" (List.map why rejected))
  @@ __POS_OF__
       {|
       "": a hex colour starts with '#'
       "fff": a hex colour starts with '#'
       " #fff": a hex colour starts with '#'
       "#": expected 3, 4, 6 or 8 hex digits after '#', found 0
       "#ff": expected 3, 4, 6 or 8 hex digits after '#', found 2
       "#fffff": expected 3, 4, 6 or 8 hex digits after '#', found 5
       "#fffffff": expected 3, 4, 6 or 8 hex digits after '#', found 7
       "#ggg": 'g' is not a hex digit
       "#12345g": 'g' is not a hex digit
       "##fff": '#' is not a hex digit
       |}

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

let on_byte_grid x =
  Float.of_int (Float.to_int (Float.round (x *. 255.))) /. 255. = x

let hex_round_trip c =
  let on_grid = List.for_all on_byte_grid (components c) in
  cover "components of n / 255" on_grid;
  cover "a component off the grid" (not on_grid);
  let read = Color.of_hex (Color.to_hex c) in
  let w = result color string in
  if on_grid then equal w (Ok c) read else not_equal w (Ok c) read

let hex_tests =
  group "hex"
    [
      cases "of_hex reads" ~id:fst parsed (fun (s, c) ->
          equal (result color string) (Ok c) (Color.of_hex s));
      cases "of_hex rejects" ~id:(Printf.sprintf "%S") rejected (fun s ->
          is_error ~pp:pp_color (Color.of_hex s));
      test "of_hex says why it rejects a text" reasons;
      test "to_hex writes lowercase digits" (fun () ->
          equal string "#1f77b480"
            (Color.to_hex
               (Color.v ~alpha:(128. /. 255.) (31. /. 255.) (119. /. 255.)
                  (180. /. 255.))));
      cases "to_hex omits an alpha pair of ff" ~id:(Printf.sprintf "alpha %g")
        [ 1.; 0.999 ] (fun alpha ->
          equal string "#1f77b4"
            (Color.to_hex
               (Color.v ~alpha (31. /. 255.) (119. /. 255.) (180. /. 255.))));
      test "to_hex rounds halves away from zero" (fun () ->
          equal string "#00000080" (Color.to_hex (Color.v ~alpha:0.5 0. 0. 0.)));
      prop "of_hex (to_hex c) is Ok c iff every component is n / 255"
        (Gen.one_of [ gen_byte_color; gen_color ])
        hex_round_trip;
      prop "to_hex inverts of_hex on canonical text" gen_canonical_hex
        (Law.round_trip string color
           (fun s -> require_ok (Color.of_hex s))
           Color.to_hex);
      prop "pp formats to_hex" gen_color (fun c ->
          equal string (Color.to_hex c) (Format.asprintf "%a" Color.pp c));
    ]

let comparing =
  group "comparing"
    [
      cases "equal tells colours apart by" ~id:fst
        [
          ("red", Color.v 0.5 0. 0.);
          ("green", Color.v 0. 0.5 0.);
          ("blue", Color.v 0. 0. 0.5);
          ("alpha", Color.v ~alpha:0.5 0. 0. 0.);
        ]
        (fun (_, c) -> not_equal color Color.black c);
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
