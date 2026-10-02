(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_kit

let invalid f = raises_match (Exn.invalid_arg ?substring:None) f

(* [Color.pp] rounds to 8 bits, so the witness prints every digit. *)
let pp_color ppf c =
  Format.fprintf ppf "(v ~alpha:%.17g %.17g %.17g %.17g)" (Color.alpha c)
    (Color.r c) (Color.g c) (Color.b c)

let color = Testable.make ~pp:pp_color ~equal:Color.equal
let colors = array color
let scheme = Testable.make ~pp:Scheme.pp ~equal:Scheme.equal

(* [of_hex s] is the colours of the hexadecimal digits [s], six per colour, as
   the published sources write them. *)
let of_hex s =
  Array.init
    (String.length s / 6)
    (fun i -> Result.get_ok (Color.of_hex ("#" ^ String.sub s (6 * i) 6)))

let rev cs =
  let n = Array.length cs in
  Array.init n (fun i -> cs.(n - 1 - i))

(* The lightness of Oklab (Björn Ottosson, 2020), from the linear components. *)
let lightness c =
  let linear x =
    if x <= 0.04045 then x /. 12.92 else Float.pow ((x +. 0.055) /. 1.055) 2.4
  in
  let r = linear (Color.r c) and g = linear (Color.g c) in
  let b = linear (Color.b c) in
  let l =
    Float.cbrt
      ((0.4122214708 *. r) +. (0.5363325363 *. g) +. (0.0514459929 *. b))
  in
  let m =
    Float.cbrt
      ((0.2119034982 *. r) +. (0.6806995451 *. g) +. (0.1073969566 *. b))
  in
  let s =
    Float.cbrt
      ((0.0883024619 *. r) +. (0.2817188376 *. g) +. (0.6299787005 *. b))
  in
  (0.2104542553 *. l) +. (0.7936177850 *. m) -. (0.0040720468 *. s)

let uniform = Scheme.[ viridis; magma; inferno; plasma; cividis ]

let brewer_sequential =
  Scheme.
    [
      blues;
      greens;
      greys;
      oranges;
      purples;
      reds;
      bugn;
      bupu;
      gnbu;
      orrd;
      pubu;
      pubugn;
      purd;
      rdpu;
      ylgn;
      ylgnbu;
      ylorbr;
      ylorrd;
    ]

let brewer_diverging =
  Scheme.[ brbg; piyg; prgn; puor; rdbu; rdgy; rdylbu; rdylgn; spectral ]

let qualitative =
  Scheme.
    [
      okabe_ito;
      tableau10;
      accent;
      dark2;
      paired;
      pastel1;
      pastel2;
      set1;
      set2;
      set3;
    ]

let brewer = brewer_sequential @ brewer_diverging

(* The classes of a Brewer scheme's largest designed table. *)
let largest s = if List.memq s brewer_diverging then 11 else 9

(* Generators *)

let unit =
  Gen.frequency [ (3, Gen.float_range 0. 1.); (1, Gen.of_list [ 0.; 0.5; 1. ]) ]

let gen_color =
  Gen.with_pp pp_color
    (Gen.map
       (fun ((r, g, b), alpha) -> Color.v ~alpha r g b)
       (Gen.pair
          (Gen.triple unit unit unit)
          (Gen.frequency [ (3, Gen.constant 1.); (1, unit) ])))

let gen_colors = Gen.array ~size:(Gen.int_range 1 12) gen_color
let gen_brewer = Gen.of_list ~pp:Scheme.pp brewer
let gen_count = Gen.int_range 0 40

let gen_scheme =
  let named = (Scheme.turbo :: uniform) @ brewer @ qualitative in
  let base =
    Gen.one_of
      [
        Gen.of_list ~pp:Scheme.pp named;
        Gen.map Scheme.ramp gen_colors;
        Gen.map Scheme.palette gen_colors;
      ]
  in
  Gen.with_pp Scheme.pp
    (Gen.map
       (fun (s, reversed) -> if reversed then Scheme.reverse s else s)
       (Gen.pair base Gen.bool))

(* A scheme with a value drawn to reach every case of the index: bin boundaries
   and their neighbours, the ends, values beyond them, infinities and [nan]. *)
let gen_reading =
  let pp ppf (s, u) = Format.fprintf ppf "(%a, %h)" Scheme.pp s u in
  Gen.with_pp pp
    (Gen.bind gen_scheme (fun s ->
         let n = Array.length (Scheme.table s) in
         let boundary =
           Gen.map
             (fun (i, nudge) ->
               let u = Float.of_int i /. Float.of_int n in
               match nudge with 0 -> Float.pred u | 1 -> u | _ -> Float.succ u)
             (Gen.pair (Gen.int_range 0 n) (Gen.int_range 0 2))
         in
         Gen.map
           (fun u -> (s, u))
           (Gen.frequency
              [
                (4, boundary);
                (2, Gen.float_range (-0.5) 1.5);
                (1, Gen.any_float);
              ])))

(* Making schemes *)

let copies make read () =
  let cs = [| Color.black; Color.white |] in
  let s = make cs in
  cs.(0) <- Color.red;
  equal colors [| Color.black; Color.white |] (read s)

let making =
  group "making"
    [
      test "ramp raises on no colours" (fun () ->
          invalid (fun () -> Scheme.ramp [||]));
      test "palette raises on no colours" (fun () ->
          invalid (fun () -> Scheme.palette [||]));
      prop "a ramp's table is its reading in 256 classes" gen_colors (fun cs ->
          let s = Scheme.ramp cs in
          equal colors (Scheme.colors 256 s) (Scheme.table s));
      prop "a ramp through 256 colours has them as its table"
        (Gen.array ~size:(Gen.constant 256) gen_color)
        (fun cs -> equal colors cs (Scheme.table (Scheme.ramp cs)));
      test "a ramp copies its colours" (copies Scheme.ramp (Scheme.colors 2));
      prop "a palette's table is its colours" gen_colors (fun cs ->
          equal colors cs (Scheme.table (Scheme.palette cs)));
      test "a palette copies its colours" (copies Scheme.palette Scheme.table);
      prop "reverse is an involution" gen_scheme
        (Law.involutive scheme Scheme.reverse);
      prop "reverse reverses the table" gen_scheme (fun s ->
          equal colors (rev (Scheme.table s)) (Scheme.table (Scheme.reverse s)));
    ]

(* Readings *)

(* The colour [s] paints [u] with, by the index the continuous reading states:
   the floor of [N *. u], clamped. *)
let gathered s u =
  let t = Scheme.table s in
  let n = Array.length t in
  let i = Float.floor (Float.of_int n *. u) in
  if i < 0. then t.(0)
  else if i > Float.of_int (n - 1) then t.(n - 1)
  else t.(int_of_float i)

let fresh read () =
  let a = read () in
  let first = a.(0) in
  a.(0) <- Color.red;
  equal color first (read ()).(0)

let ramp_classes () =
  let bw = [| Color.black; Color.white |] in
  let grey = Color.mix 0.5 Color.black Color.white in
  equal colors [| grey |] (Scheme.colors 1 (Scheme.ramp bw));
  equal colors
    [| Color.black; grey; Color.white |]
    (Scheme.colors 3 (Scheme.ramp bw));
  equal colors [| Color.red; Color.red |]
    (Scheme.colors 2 (Scheme.ramp [| Color.red |]))

let listed_classes () =
  let t = Scheme.table Scheme.viridis in
  equal colors [| t.(128) |] (Scheme.colors 1 Scheme.viridis);
  equal colors [| t.(0); t.(255) |] (Scheme.colors 2 Scheme.viridis);
  equal colors [| t.(0); t.(128); t.(255) |] (Scheme.colors 3 Scheme.viridis);
  equal colors
    [| t.(255); t.(127); t.(0) |]
    (Scheme.colors 3 (Scheme.reverse Scheme.viridis))

(* A palette's class [i] has colour [i mod k] of its table, reversed or not. *)
let palette_classes (s, n) =
  let t = Scheme.table s in
  let k = Array.length t in
  equal colors (Array.init n (fun i -> t.(i mod k))) (Scheme.colors n s)

let gen_palette =
  Gen.with_pp Scheme.pp
    (Gen.map
       (fun (s, reversed) -> if reversed then Scheme.reverse s else s)
       (Gen.pair
          (Gen.one_of
             [
               Gen.of_list ~pp:Scheme.pp qualitative;
               Gen.map Scheme.palette gen_colors;
             ])
          Gen.bool))

let brewer_below_three s =
  let t3 = Scheme.colors 3 s in
  let two =
    if List.memq s brewer_diverging then [| t3.(0); t3.(2) |]
    else [| t3.(1); t3.(2) |]
  in
  equal ~msg:"one class" colors [| t3.(1) |] (Scheme.colors 1 s);
  equal ~msg:"two classes" colors two (Scheme.colors 2 s)

(* Past its largest table a Brewer scheme, continuous reading included, reads as
   the ramp through that table. *)
let brewer_past_largest (s, extra) =
  let ramp = Scheme.ramp (Scheme.colors (largest s) s) in
  let n = largest s + 1 + extra in
  equal ~msg:"classes" colors (Scheme.colors n ramp) (Scheme.colors n s);
  equal ~msg:"table" colors (Scheme.table ramp) (Scheme.table s)

let readings =
  group "readings"
    [
      prop "color gathers the table by the stated index" gen_reading
        (fun (s, u) ->
          let expected =
            if Float.is_nan u then Color.transparent else gathered s u
          in
          equal color expected (Scheme.color s u));
      test "nan paints the unknown colour" (fun () ->
          equal color Color.red
            (Scheme.color ~unknown:Color.red Scheme.viridis nan));
      test "table is a fresh array"
        (fresh (fun () -> Scheme.table Scheme.viridis));
      test "colors is a fresh array"
        (fresh (fun () -> Scheme.colors 3 Scheme.blues));
      test "colors raises on a negative count" (fun () ->
          invalid (fun () -> Scheme.colors (-1) Scheme.viridis));
      prop "colors n has n colours" (Gen.pair gen_scheme gen_count)
        (fun (s, n) -> equal int n (Array.length (Scheme.colors n s)));
      prop "a palette starts again after its last colour"
        (Gen.pair gen_palette gen_count)
        palette_classes;
      test "a ramp's classes fall on its colours" ramp_classes;
      prop "a ramp through m colours gives them in m classes" gen_colors
        (fun cs ->
          equal colors cs (Scheme.colors (Array.length cs) (Scheme.ramp cs)));
      prop "a reversed ramp or Brewer scheme reverses its classes"
        (Gen.pair
           (Gen.one_of [ gen_brewer; Gen.map Scheme.ramp gen_colors ])
           gen_count)
        (fun (s, n) ->
          equal colors
            (rev (Scheme.colors n s))
            (Scheme.colors n (Scheme.reverse s)));
      cases "Brewer below three classes"
        ~name:(Format.asprintf "%a" Scheme.pp)
        brewer brewer_below_three;
      prop "Brewer past its largest table"
        (Gen.pair gen_brewer (Gen.int_range 0 20))
        brewer_past_largest;
      test "a table scheme samples its ends and middle" listed_classes;
    ]

(* The catalogue *)

(* Entries 0, 128 and 255 of the published float tables. *)
let anchors =
  [
    ( Scheme.viridis,
      [
        (0.267004, 0.004874, 0.329415);
        (0.127568, 0.566949, 0.550556);
        (0.993248, 0.906157, 0.143936);
      ] );
    ( Scheme.magma,
      [
        (0.001462, 0.000466, 0.013866);
        (0.716387, 0.214982, 0.47529);
        (0.987053, 0.991438, 0.749504);
      ] );
    ( Scheme.inferno,
      [
        (0.001462, 0.000466, 0.013866);
        (0.735683, 0.215906, 0.330245);
        (0.988362, 0.998364, 0.644924);
      ] );
    ( Scheme.plasma,
      [
        (0.050383, 0.029803, 0.527975);
        (0.798216, 0.280197, 0.469538);
        (0.940015, 0.975158, 0.131326);
      ] );
    ( Scheme.cividis,
      [
        (0.0, 0.135112, 0.304751);
        (0.488697, 0.485318, 0.471008);
        (0.995737, 0.909344, 0.217772);
      ] );
    ( Scheme.turbo,
      [
        (0.18995, 0.07176, 0.23217);
        (0.64362, 0.98999, 0.23356);
        (0.4796, 0.01583, 0.01055);
      ] );
  ]

(* A component rounded to the nearest multiple of [1/255] is within half of one
   of the published one. *)
let rounded = float ((1. /. 510.) +. 1e-12)

let published (s, entries) =
  let t = Scheme.table s in
  equal ~msg:"length" int 256 (Array.length t);
  List.iter2
    (fun i (r, g, b) ->
      let c = t.(i) and msg = Printf.sprintf "entry %d" i in
      let component x x' =
        equal ~msg rounded x x';
        equal ~msg float_exact (Float.round (x' *. 255.) /. 255.) x'
      in
      component r (Color.r c);
      component g (Color.g c);
      component b (Color.b c))
    [ 0; 128; 255 ] entries

let designed =
  Scheme.
    [
      (blues, "deebf79ecae13182bd");
      (blues, "f7fbffdeebf7c6dbef9ecae16baed64292c62171b508519c08306b");
      (greens, "edf8e9bae4b374c47631a354006d2c");
      (ylorrd, "ffffb2fed976feb24cfd8d3cfc4e2ae31a1cb10026");
      (pubugn, "f6eff7bdc9e167a9cf02818a");
      (rdbu, "ef8a62f7f7f767a9cf");
      ( rdbu,
        "67001fb2182bd6604df4a582fddbc7f7f7f7d1e5f092c5de4393c32166ac053061" );
      (spectral, "d53e4ff46d43fdae61fee08be6f598abdda466c2a53288bd");
      (puor, "5e3c99b2abd2fdb863e66101");
      (okabe_ito, "000000e69f0056b4e9009e73f0e4420072b2d55e00cc79a7");
      (tableau10, "4e79a7f28e2ce1575976b7b259a14fedc949af7aa1ff9da79c755fbab0ab");
      ( paired,
        "a6cee31f78b4b2df8a33a02cfb9a99e31a1cfdbf6fff7f00cab2d66a3d9affff99b15928"
      );
      (set1, "e41a1c377eb84daf4a984ea3ff7f00ffff33a65628f781bf999999");
    ]

(* [monotone ~slack ~msg ~rising cs] asserts that the lightness of [cs] rises,
   or falls, between consecutive colours, give or take [slack]. *)
let monotone ?(slack = 0.) ~msg ~rising cs =
  for i = 0 to Array.length cs - 2 do
    let msg = Printf.sprintf "%s, %d to %d" msg i (i + 1) in
    let l = lightness cs.(i) and l' = lightness cs.(i + 1) in
    if rising then less ~msg float_exact ~than:(l' +. slack) l
    else greater ~msg float_exact ~than:(l' -. slack) l
  done

(* Rounding a published table to 8 bits moves the lightness of neighbouring
   entries by less than a level, and can swap a pair of them: viridis's entries
   94 and 95, cividis's 24 and 25. *)
let rounding = 1. /. 255.

let sequential_darkens s =
  monotone ~msg:"table" ~rising:false (Scheme.table s);
  for n = 3 to 9 do
    monotone ~msg:(string_of_int n) ~rising:false (Scheme.colors n s)
  done

(* Diverging tables of an odd number of classes are lightest at their middle. *)
let diverging_lightest_in_middle s =
  for k = 1 to 5 do
    let n = (2 * k) + 1 in
    let cs = Scheme.colors n s and msg = string_of_int n in
    monotone ~msg ~rising:true (Array.sub cs 0 (k + 1));
    monotone ~msg ~rising:false (Array.sub cs k (k + 1))
  done

let name = Format.asprintf "%a" Scheme.pp

let catalogue =
  group "catalogue"
    [
      cases "tables are the published ones, rounded"
        ~name:(fun (s, _) -> name s)
        anchors published;
      cases "designed tables and palettes"
        ~name:(fun (s, h) ->
          Printf.sprintf "%s, %d" (name s) (String.length h / 6))
        designed
        (fun (s, h) ->
          let cs = of_hex h in
          equal colors cs (Scheme.colors (Array.length cs) s));
      cases "perceptually uniform lightness rises, up to rounding" ~name uniform
        (fun s ->
          monotone ~slack:rounding ~msg:"table" ~rising:true (Scheme.table s));
      cases "Brewer sequential schemes darken" ~name brewer_sequential
        sequential_darkens;
      cases "Brewer diverging tables are lightest at their middle" ~name
        brewer_diverging diverging_lightest_in_middle;
    ]

(* Comparing and formatting *)

let rbw = [| Color.blue; Color.white; Color.red |]

let unequal =
  [
    ("two named schemes", Scheme.viridis, Scheme.magma);
    ("a scheme and its reverse", Scheme.rdbu, Scheme.reverse Scheme.rdbu);
    ( "a ramp through viridis's table and viridis",
      Scheme.ramp (Scheme.table Scheme.viridis),
      Scheme.viridis );
    ("a ramp and a palette alike", Scheme.ramp rbw, Scheme.palette rbw);
    ("ramps through other colours", Scheme.ramp rbw, Scheme.ramp (rev rbw));
    ("palettes of other colours", Scheme.palette rbw, Scheme.palette (rev rbw));
  ]

let printing () =
  let print s = Format.asprintf "%a" Scheme.pp s in
  expect
    (String.concat "\n"
       (List.map print
          [
            Scheme.viridis;
            Scheme.reverse Scheme.rdbu;
            Scheme.ramp [| Color.black; Color.white |];
            Scheme.reverse (Scheme.palette [| Color.red; Color.blue |]);
          ]))
  @@ __POS_OF__
       {|
    viridis
    reverse(rdbu)
    ramp(#000000 #ffffff)
    reverse(palette(#ff0000 #0000ff))
    |}

let comparing =
  group "comparing and formatting"
    [
      prop "equal is an equivalence"
        (Gen.pair gen_scheme gen_scheme)
        (Law.equivalence scheme);
      test "schemes made alike are equal" (fun () ->
          equal scheme (Scheme.ramp rbw) (Scheme.ramp (Array.copy rbw));
          equal scheme (Scheme.palette rbw) (Scheme.palette (Array.copy rbw)));
      cases "unequal schemes"
        ~name:(fun (n, _, _) -> n)
        unequal
        (fun (_, s, s') -> not_equal scheme s s');
      test "pp names named schemes and shows made ones" printing;
    ]

let () = exit (run "Scheme" [ making; readings; catalogue; comparing ])
