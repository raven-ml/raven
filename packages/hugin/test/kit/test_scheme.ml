(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_kit

let invalid f = raises_match (Exn.invalid_arg ?sub:None) f

(* [Color.pp] rounds to 8 bits, so the witness prints every digit. *)
let pp_color ppf c =
  Format.fprintf ppf "(v ~alpha:%.17g %.17g %.17g %.17g)" (Color.alpha c)
    (Color.r c) (Color.g c) (Color.b c)

let color = Testable.make ~pp:pp_color ~equal:Color.equal
let colors = array color
let scheme = Testable.make ~pp:Scheme.pp ~equal:Scheme.equal
let name = Format.asprintf "%a" Scheme.pp

(* [of_hex s] is the colours of the hexadecimal digits [s], six per colour, as
   the published sources write them. *)
let of_hex s =
  Array.init
    (String.length s / 6)
    (fun i -> Result.get_ok (Color.of_hex ("#" ^ String.sub s (6 * i) 6)))

let rev cs =
  let n = Array.length cs in
  Array.init n (fun i -> cs.(n - 1 - i))

(* The lightness of Oklab (Björn Ottosson, 2020) from encoded sRGB
   components. *)
let oklab_l r g b =
  let linear x =
    if x <= 0.04045 then x /. 12.92 else Float.pow ((x +. 0.055) /. 1.055) 2.4
  in
  let r = linear r and g = linear g and b = linear b in
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

let lightness c = oklab_l (Color.r c) (Color.g c) (Color.b c)

(* [rises cs] asserts that the lightness of each colour of [cs] is below that of
   the next. *)
let rises cs =
  for i = 0 to Array.length cs - 2 do
    let msg = Printf.sprintf "lightness of %d below that of %d" i (i + 1) in
    less ~msg float_exact ~than:(lightness cs.(i + 1)) (lightness cs.(i))
  done

(* [falls cs] asserts that the lightness of each colour of [cs] is above that of
   the next. *)
let falls cs =
  for i = 0 to Array.length cs - 2 do
    let msg = Printf.sprintf "lightness of %d above that of %d" i (i + 1) in
    greater ~msg float_exact ~than:(lightness cs.(i + 1)) (lightness cs.(i))
  done

(* [rounding_range c] is the least and greatest lightness of the colours that
   round to [c]: each component within [1/510] of [c]'s, and in [0, 1]. Across a
   cube this small lightness is nearly linear, so its extremes lie at the 8
   corners. *)
let rounding_range c =
  let near x =
    [ Float.max 0. (x -. (1. /. 510.)); Float.min 1. (x +. (1. /. 510.)) ]
  in
  let corners =
    List.concat_map
      (fun r ->
        List.concat_map
          (fun g -> List.map (fun b -> oklab_l r g b) (near (Color.b c)))
          (near (Color.g c)))
      (near (Color.r c))
  in
  ( List.fold_left Float.min infinity corners,
    List.fold_left Float.max neg_infinity corners )

(* [rises_up_to_rounding cs] asserts that the colours [cs] rounds can have
   rising lightness: no entry is darker than an earlier one beyond what rounding
   explains, the greatest lightness of [j] above the least of every [i < j]. *)
let rises_up_to_rounding cs =
  let floor = ref neg_infinity and at = ref 0 in
  Array.iteri
    (fun j c ->
      let lo, hi = rounding_range c in
      if j > 0 then
        less
          ~msg:(Printf.sprintf "entry %d against entry %d" j !at)
          float_exact ~than:hi !floor;
      if lo > !floor then begin
        floor := lo;
        at := j
      end)
    cs

let falls_up_to_rounding cs = rises_up_to_rounding (rev cs)
let uniform = Scheme.[ viridis; magma; inferno; plasma; cividis ]
let listed = Scheme.turbo :: uniform

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

let brewer = brewer_sequential @ brewer_diverging

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

(* The classes of a Brewer scheme's largest designed table. *)
let largest s = if List.memq s brewer_diverging then 11 else 9
let reversed_if b s = if b then Scheme.reverse s else s

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
  let named = listed @ brewer @ qualitative in
  let base =
    Gen.one_of
      [
        Gen.of_list ~pp:Scheme.pp named;
        Gen.map Scheme.ramp gen_colors;
        Gen.map Scheme.palette gen_colors;
      ]
  in
  Gen.with_pp Scheme.pp
    (Gen.map (fun (s, b) -> reversed_if b s) (Gen.pair base Gen.bool))

(* A scheme with a value drawn to reach every case of the index: bin boundaries
   and their neighbours, the ends, values beyond them and the infinities. *)
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
                (1, Gen.of_list [ Float.neg_infinity; Float.infinity ]);
              ])))

(* Continuous reading *)

(* The colour of [u] by the stated index, the floor of [N *. u] clamped to the
   table. *)
let gathered s u =
  let t = Scheme.table s in
  let n = Array.length t in
  let i = Float.floor (Float.of_int n *. u) in
  if i < 0. then t.(0)
  else if i > Float.of_int (n - 1) then t.(n - 1)
  else t.(int_of_float i)

let continuous =
  group "continuous reading"
    [
      prop "a value takes the colour of its bin, the floor of N u clamped"
        ~examples:
          [
            (Scheme.viridis, Float.neg_infinity);
            (Scheme.viridis, Float.infinity);
            (Scheme.reverse Scheme.blues, -0.);
            (Scheme.ramp [| Color.black; Color.white |], 1.);
          ]
        gen_reading
        (fun (s, u) ->
          cover "below 0" (u < 0.);
          cover "from 1 up" (u >= 1.);
          cover "an infinity" (Float.abs u = Float.infinity);
          equal color (gathered s u) (Scheme.color s u));
      test "nan takes the colour given for unknown values" (fun () ->
          equal color Color.red
            (Scheme.color ~unknown:Color.red Scheme.viridis Float.nan));
      prop "a ramp's table is its reading in 256 classes" gen_colors (fun cs ->
          let s = Scheme.ramp cs in
          equal colors (Scheme.colors 256 s) (Scheme.table s));
      prop "a palette's table is its colours" gen_colors (fun cs ->
          equal colors cs (Scheme.table (Scheme.palette cs)));
    ]

(* Discrete reading *)

(* A palette, a qualitative scheme or the reverse of either. *)
let gen_palette =
  Gen.with_pp Scheme.pp
    (Gen.map
       (fun (s, b) -> reversed_if b s)
       (Gen.pair
          (Gen.one_of
             [
               Gen.of_list ~pp:Scheme.pp qualitative;
               Gen.map Scheme.palette gen_colors;
             ])
          Gen.bool))

(* Class [i] of [n] of the ramp through [cs], by the stated division. *)
let ramp_class cs n i =
  let q = i * (Array.length cs - 1) in
  let j = q / (n - 1) and r = q mod (n - 1) in
  if r = 0 then cs.(j)
  else Color.mix (Float.of_int r /. Float.of_int (n - 1)) cs.(j) cs.(j + 1)

(* The designed tables of ColorBrewer's colorbrewer_schemes.js
   (https://github.com/axismaps/colorbrewer): of three classes and of the
   largest, and a few between. *)
let designed =
  Scheme.
    [
      (blues, "deebf79ecae13182bd");
      (blues, "f7fbffdeebf7c6dbef9ecae16baed64292c62171b508519c08306b");
      (greens, "e5f5e0a1d99b31a354");
      (greens, "edf8e9bae4b374c47631a354006d2c");
      (greens, "f7fcf5e5f5e0c7e9c0a1d99b74c47641ab5d238b45006d2c00441b");
      (greys, "f0f0f0bdbdbd636363");
      (greys, "fffffff0f0f0d9d9d9bdbdbd969696737373525252252525000000");
      (oranges, "fee6cefdae6be6550d");
      (oranges, "fff5ebfee6cefdd0a2fdae6bfd8d3cf16913d94801a636037f2704");
      (purples, "efedf5bcbddc756bb1");
      (purples, "fcfbfdefedf5dadaebbcbddc9e9ac8807dba6a51a354278f3f007d");
      (reds, "fee0d2fc9272de2d26");
      (reds, "fff5f0fee0d2fcbba1fc9272fb6a4aef3b2ccb181da50f1567000d");
      (bugn, "e5f5f999d8c92ca25f");
      (bugn, "f7fcfde5f5f9ccece699d8c966c2a441ae76238b45006d2c00441b");
      (bupu, "e0ecf49ebcda8856a7");
      (bupu, "f7fcfde0ecf4bfd3e69ebcda8c96c68c6bb188419d810f7c4d004b");
      (gnbu, "e0f3dba8ddb543a2ca");
      (gnbu, "f7fcf0e0f3dbccebc5a8ddb57bccc44eb3d32b8cbe0868ac084081");
      (orrd, "fee8c8fdbb84e34a33");
      (orrd, "fff7ecfee8c8fdd49efdbb84fc8d59ef6548d7301fb300007f0000");
      (pubu, "ece7f2a6bddb2b8cbe");
      (pubu, "fff7fbece7f2d0d1e6a6bddb74a9cf3690c00570b0045a8d023858");
      (pubugn, "ece2f0a6bddb1c9099");
      (pubugn, "f6eff7bdc9e167a9cf02818a");
      (pubugn, "fff7fbece2f0d0d1e6a6bddb67a9cf3690c002818a016c59014636");
      (purd, "e7e1efc994c7dd1c77");
      (purd, "f7f4f9e7e1efd4b9dac994c7df65b0e7298ace125698004367001f");
      (rdpu, "fde0ddfa9fb5c51b8a");
      (rdpu, "fff7f3fde0ddfcc5c0fa9fb5f768a1dd3497ae017e7a017749006a");
      (ylgn, "f7fcb9addd8e31a354");
      (ylgn, "ffffe5f7fcb9d9f0a3addd8e78c67941ab5d238443006837004529");
      (ylgnbu, "edf8b17fcdbb2c7fb8");
      (ylgnbu, "ffffd9edf8b1c7e9b47fcdbb41b6c41d91c0225ea8253494081d58");
      (ylorbr, "fff7bcfec44fd95f0e");
      (ylorbr, "ffffe5fff7bcfee391fec44ffe9929ec7014cc4c02993404662506");
      (ylorrd, "ffeda0feb24cf03b20");
      (ylorrd, "ffffccffeda0fed976feb24cfd8d3cfc4e2ae31a1cb10026");
      (ylorrd, "ffffccffeda0fed976feb24cfd8d3cfc4e2ae31a1cbd0026800026");
      (brbg, "d8b365f5f5f55ab4ac");
      ( brbg,
        "5430058c510abf812ddfc27df6e8c3f5f5f5c7eae580cdc135978f01665e003c30" );
      (piyg, "e9a3c9f7f7f7a1d76a");
      ( piyg,
        "8e0152c51b7dde77aef1b6dafde0eff7f7f7e6f5d0b8e1867fbc414d9221276419" );
      (prgn, "af8dc3f7f7f77fbf7b");
      ( prgn,
        "40004b762a839970abc2a5cfe7d4e8f7f7f7d9f0d3a6dba05aae611b783700441b" );
      (puor, "f1a340f7f7f7998ec3");
      (puor, "e66101fdb863b2abd25e3c99");
      ( puor,
        "7f3b08b35806e08214fdb863fee0b6f7f7f7d8daebb2abd28073ac5427882d004b" );
      (rdbu, "ef8a62f7f7f767a9cf");
      ( rdbu,
        "67001fb2182bd6604df4a582fddbc7f7f7f7d1e5f092c5de4393c32166ac053061" );
      (rdgy, "ef8a62ffffff999999");
      ( rdgy,
        "67001fb2182bd6604df4a582fddbc7ffffffe0e0e0bababa8787874d4d4d1a1a1a" );
      (rdylbu, "fc8d59ffffbf91bfdb");
      ( rdylbu,
        "a50026d73027f46d43fdae61fee090ffffbfe0f3f8abd9e974add14575b4313695" );
      (rdylgn, "fc8d59ffffbf91cf60");
      ( rdylgn,
        "a50026d73027f46d43fdae61fee08bffffbfd9ef8ba6d96a66bd631a9850006837" );
      (spectral, "fc8d59ffffbf99d594");
      (spectral, "d53e4ff46d43fdae61fee08be6f598abdda466c2a53288bd");
      ( spectral,
        "9e0142d53e4ff46d43fdae61fee08bffffbfe6f598abdda466c2a53288bd5e4fa2" );
    ]

let gen_listed =
  Gen.with_pp
    (fun ppf (s, n) -> Format.fprintf ppf "(%a, %d)" Scheme.pp s n)
    (Gen.map
       (fun ((s, b), n) -> (reversed_if b s, n))
       Gen.(pair (pair (of_list listed) bool) gen_count))

let discrete =
  group "discrete reading"
    [
      prop "the reverse of a ramp or Brewer scheme reverses its classes"
        (Gen.pair
           (Gen.one_of [ gen_brewer; Gen.map Scheme.ramp gen_colors ])
           gen_count)
        (fun (s, n) ->
          equal colors
            (rev (Scheme.colors n s))
            (Scheme.colors n (Scheme.reverse s)));
      prop "a palette's class i has colour i mod k of its table"
        (Gen.pair gen_palette gen_count) (fun (s, n) ->
          let t = Scheme.table s in
          let k = Array.length t in
          cover "past the last colour" (n > k);
          equal colors (Array.init n (fun i -> t.(i mod k))) (Scheme.colors n s));
      cases "a Brewer scheme gives the table designed for its classes"
        ~id:(fun (s, h) ->
          Printf.sprintf "%s, %d classes" (name s) (String.length h / 6))
        designed
        (fun (s, h) ->
          let cs = of_hex h in
          equal colors cs (Scheme.colors (Array.length cs) s));
      cases "one class of a Brewer scheme is the middle of its 3-class table"
        ~id:name brewer (fun s ->
          equal colors [| (Scheme.colors 3 s).(1) |] (Scheme.colors 1 s));
      cases
        "two classes of a sequential scheme are its 3-class table's last two"
        ~id:name brewer_sequential (fun s ->
          let t3 = Scheme.colors 3 s in
          equal colors [| t3.(1); t3.(2) |] (Scheme.colors 2 s));
      cases
        "two classes of a diverging scheme are its 3-class table's first and \
         last"
        ~id:name brewer_diverging (fun s ->
          let t3 = Scheme.colors 3 s in
          equal colors [| t3.(0); t3.(2) |] (Scheme.colors 2 s));
      prop "a ramp's class i mixes the colours about i (m - 1) / (n - 1)"
        ~examples:[ ([| Color.red |], 2); ([| Color.black; Color.white |], 3) ]
        (Gen.pair gen_colors (Gen.int_range 2 40))
        (fun (cs, n) ->
          cover "a ramp of one colour" (Array.length cs = 1);
          equal colors
            (Array.init n (ramp_class cs n))
            (Scheme.colors n (Scheme.ramp cs)));
      prop "one class of a ramp is class 1 of three" gen_colors (fun cs ->
          equal colors
            [| ramp_class cs 3 1 |]
            (Scheme.colors 1 (Scheme.ramp cs)));
      prop "a ramp through m colours gives them in m classes" gen_colors
        (fun cs ->
          equal colors cs (Scheme.colors (Array.length cs) (Scheme.ramp cs)));
      (* 27 schemes and 6 counts, checked whole. *)
      prop "a Brewer scheme past its largest table reads as the ramp through it"
        ~count:(27 * 6)
        (Gen.pair gen_brewer (Gen.int_range 1 6))
        (fun (s, extra) ->
          let n = largest s + extra in
          equal colors
            (Scheme.colors n (Scheme.ramp (Scheme.colors (largest s) s)))
            (Scheme.colors n s));
      prop "another scheme reads its continuous reading at i / (n - 1)"
        ~examples:[ (Scheme.viridis, 1); (Scheme.reverse Scheme.turbo, 1) ]
        gen_listed
        (fun (s, n) ->
          cover "one class" (n = 1);
          let at i =
            if n = 1 then Scheme.color s 0.5
            else Scheme.color s (Float.of_int i /. Float.of_int (n - 1))
          in
          equal colors (Array.init n at) (Scheme.colors n s));
    ]

(* Making schemes *)

(* [copies make read] asserts that [make] copies the colours [read] gives back,
   so changing the array afterwards changes nothing. *)
let copies make read () =
  let cs = [| Color.black; Color.white |] in
  let s = make cs in
  cs.(0) <- Color.red;
  equal colors [| Color.black; Color.white |] (read s)

let ramp =
  group "ramp"
    [
      test "copies its colours" (copies Scheme.ramp (Scheme.colors 2));
      test "raises on no colours" (fun () ->
          invalid (fun () -> Scheme.ramp [||]));
    ]

let palette =
  group "palette"
    [
      test "copies its colours" (copies Scheme.palette Scheme.table);
      test "raises on no colours" (fun () ->
          invalid (fun () -> Scheme.palette [||]));
    ]

let reverse =
  group "reverse"
    [
      prop "reverses the table" gen_scheme (fun s ->
          equal colors (rev (Scheme.table s)) (Scheme.table (Scheme.reverse s)));
      prop "is an involution" gen_scheme (Law.involutive scheme Scheme.reverse);
    ]

(* Readings *)

(* [fresh read] asserts that what [read] returns is a new array each time. *)
let fresh read () =
  let a = read () in
  let first = a.(0) in
  a.(0) <- Color.red;
  equal color first (read ()).(0)

let color_group =
  group "color"
    [
      test "paints nan transparent by default" (fun () ->
          equal color Color.transparent (Scheme.color Scheme.viridis Float.nan));
    ]

let table =
  group "table"
    [ test "is a fresh array" (fresh (fun () -> Scheme.table Scheme.viridis)) ]

let colors_group =
  group "colors"
    [
      prop "gives n colours"
        ~examples:[ (Scheme.viridis, 0); (Scheme.palette [| Color.red |], 0) ]
        (Gen.pair gen_scheme gen_count)
        (fun (s, n) ->
          cover "no class" (n = 0);
          equal int n (Array.length (Scheme.colors n s)));
      test "is a fresh array" (fresh (fun () -> Scheme.colors 3 Scheme.blues));
      test "raises on a negative count" (fun () ->
          invalid (fun () -> Scheme.colors (-1) Scheme.viridis));
    ]

(* Sequential schemes *)

(* Entries 0, 128 and 255 of the published float tables: Matplotlib's
   _cm_listed.py for the first five, Mikhailov's turbo_colormap.py for turbo. *)
let published =
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
        (0.716387, 0.214982, 0.475290);
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
        (0.000000, 0.135112, 0.304751);
        (0.488697, 0.485318, 0.471008);
        (0.995737, 0.909344, 0.217772);
      ] );
    ( Scheme.turbo,
      [
        (0.18995, 0.07176, 0.23217);
        (0.64362, 0.98999, 0.23356);
        (0.47960, 0.01583, 0.01055);
      ] );
  ]

(* [level x] is [x] rounded to the nearest multiple of [1/255]. *)
let level x = Float.round (x *. 255.) /. 255.

let is_published (s, entries) =
  let t = Scheme.table s in
  equal ~msg:"length" int 256 (Array.length t);
  List.iter2
    (fun i (r, g, b) ->
      equal
        ~msg:(Printf.sprintf "entry %d" i)
        colors
        [| Color.v (level r) (level g) (level b) |]
        [| t.(i) |])
    [ 0; 128; 255 ] entries

(* Index of the lightest colour of [cs]. *)
let lightest cs =
  let best = ref 0 in
  Array.iteri
    (fun i c -> if lightness c > lightness cs.(!best) then best := i)
    cs;
  !best

let sequential =
  group "sequential schemes"
    [
      cases "a published table has its 256 colours, each component rounded"
        ~id:(fun (s, _) -> name s)
        published is_published;
      cases "the lightness of a uniform table rises" ~id:name uniform (fun s ->
          rises_up_to_rounding (Scheme.table s));
      test "turbo's lightness rises then falls" (fun () ->
          let t = Scheme.table Scheme.turbo in
          let k = lightest t in
          rises_up_to_rounding (Array.sub t 0 (k + 1));
          falls_up_to_rounding (Array.sub t k (256 - k)));
    ]

let brewer_sequential_group =
  group "Brewer sequential schemes"
    [
      cases "designed tables darken" ~id:name brewer_sequential (fun s ->
          for n = 3 to 9 do
            falls (Scheme.colors n s)
          done);
      cases "the table is the ramp through the 9-class table" ~id:name
        brewer_sequential (fun s ->
          equal colors
            (Scheme.table (Scheme.ramp (Scheme.colors 9 s)))
            (Scheme.table s));
      cases "the table darkens" ~id:name brewer_sequential (fun s ->
          falls (Scheme.table s));
    ]

let diverging =
  group "diverging schemes"
    [
      cases
        "designed tables of an odd number of classes are lightest at the middle"
        ~id:name brewer_diverging (fun s ->
          for k = 1 to 5 do
            let cs = Scheme.colors ((2 * k) + 1) s in
            rises (Array.sub cs 0 (k + 1));
            falls (Array.sub cs k (k + 1))
          done);
      cases "the table is the ramp through the 11-class table" ~id:name
        brewer_diverging (fun s ->
          equal colors
            (Scheme.table (Scheme.ramp (Scheme.colors 11 s)))
            (Scheme.table s));
    ]

(* Okabe and Ito's and Tableau 10's colours are those their entries list; the
   others are ColorBrewer's colorbrewer_schemes.js. *)
let palettes =
  Scheme.
    [
      (okabe_ito, "000000e69f0056b4e9009e73f0e4420072b2d55e00cc79a7");
      (tableau10, "4e79a7f28e2ce1575976b7b259a14fedc949af7aa1ff9da79c755fbab0ab");
      (accent, "7fc97fbeaed4fdc086ffff99386cb0f0027fbf5b17666666");
      (dark2, "1b9e77d95f027570b3e7298a66a61ee6ab02a6761d666666");
      ( paired,
        "a6cee31f78b4b2df8a33a02cfb9a99e31a1cfdbf6fff7f00cab2d66a3d9affff99b15928"
      );
      (pastel1, "fbb4aeb3cde3ccebc5decbe4fed9a6ffffcce5d8bdfddaecf2f2f2");
      (pastel2, "b3e2cdfdcdaccbd5e8f4cae4e6f5c9fff2aef1e2cccccccc");
      (set1, "e41a1c377eb84daf4a984ea3ff7f00ffff33a65628f781bf999999");
      (set2, "66c2a5fc8d628da0cbe78ac3a6d854ffd92fe5c494b3b3b3");
      ( set3,
        "8dd3c7ffffb3bebadafb807280b1d3fdb462b3de69fccde5d9d9d9bc80bdccebc5ffed6f"
      );
    ]

let qualitative_group =
  group "qualitative schemes"
    [
      cases "each is the palette of its colours"
        ~id:(fun (s, _) -> name s)
        palettes
        (fun (s, h) -> equal colors (of_hex h) (Scheme.table s));
    ]

(* Comparing and formatting *)

(* How a scheme is made: a named scheme by its index in [named], or a
   constructor and its colours; then whether it is reversed. *)
type made = Named of int | Ramp of Color.t array | Palette of Color.t array

let named = Array.of_list (listed @ brewer @ qualitative)

let make (m, reversed) =
  reversed_if reversed
    (match m with
    | Named i -> named.(i)
    | Ramp cs -> Scheme.ramp cs
    | Palette cs -> Scheme.palette cs)

let same_colors cs cs' =
  Array.length cs = Array.length cs' && Array.for_all2 Color.equal cs cs'

let same_making (m, reversed) (m', reversed') =
  Bool.equal reversed reversed'
  &&
  match (m, m') with
  | Named i, Named i' -> i = i'
  | Ramp cs, Ramp cs' | Palette cs, Palette cs' -> same_colors cs cs'
  | _ -> false

(* Few colours, so that two drawn makings often agree. *)
let gen_making =
  let few =
    Gen.array ~size:(Gen.int_range 1 3)
      (Gen.of_list ~pp:pp_color [ Color.black; Color.white; Color.red ])
  in
  Gen.pair
    (Gen.one_of
       [
         Gen.map (fun i -> Named i) (Gen.int_range 0 (Array.length named - 1));
         Gen.map (fun cs -> Ramp cs) few;
         Gen.map (fun cs -> Palette cs) few;
       ])
    Gen.bool

let bw = [| Color.black; Color.white |]

let equal_group =
  group "equal"
    [
      prop "is sameness of name or constructor and colours, and of reversal"
        ~examples:
          [
            ((Ramp bw, false), (Ramp (Array.copy bw), false));
            ((Named 0, false), (Named 0, true));
            ((Ramp bw, false), (Palette bw, false));
          ]
        (Gen.with_pp
           (fun ppf (a, b) ->
             Format.fprintf ppf "(%a, %a)" Scheme.pp (make a) Scheme.pp (make b))
           (Gen.pair gen_making gen_making))
        (fun (a, b) ->
          let same = same_making a b in
          cover "the same making" same;
          cover "reversal alone differs"
            (same_making (fst a, true) (fst b, true) && not same);
          if same then equal scheme (make a) (make b)
          else not_equal scheme (make a) (make b));
      test "a ramp through viridis's table is not viridis" (fun () ->
          not_equal scheme Scheme.viridis
            (Scheme.ramp (Scheme.table Scheme.viridis)));
    ]

let pp =
  group "pp"
    [
      cases "names a scheme as the interface shows" ~id:snd
        [
          (Scheme.viridis, "viridis");
          (Scheme.ramp bw, "ramp(#000000 #ffffff)");
          (Scheme.reverse Scheme.rdbu, "reverse(rdbu)");
        ]
        (fun (s, e) -> equal string e (name s));
    ]

let () =
  exit
    (run "Scheme"
       [
         continuous;
         discrete;
         ramp;
         palette;
         reverse;
         color_group;
         table;
         colors_group;
         sequential;
         brewer_sequential_group;
         diverging;
         qualitative_group;
         equal_group;
         pp;
       ])
