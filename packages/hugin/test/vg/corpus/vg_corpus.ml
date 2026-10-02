(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Pictures that every renderer's golden outputs draw, each case of a picture at
   least once. *)

open Hugin_gg
open Hugin_font
open Hugin_vg

let rect x y w h = Path.rect (Box2.v x y w h)
let paper w h = Picture.fill Color.white (rect 0. 0. w h)
let page w h ps = Renderable.v w h (Picture.group (paper w h :: ps))

(* [typeset text size] sets [text] in Inter Regular, kerned. *)
let typeset ?(font = Font.regular) text size =
  let chars = ref [] in
  String.iter (fun c -> chars := Uchar.of_char c :: !chars) text;
  let glyphs = Array.of_list (List.rev_map (Font.glyph font) !chars) in
  let n = Array.length glyphs in
  let xs = Array.make n 0. in
  for i = 1 to n - 1 do
    xs.(i) <-
      xs.(i - 1)
      +. size
         *. (Font.advance font glyphs.(i - 1)
            +. Font.kerning font glyphs.(i - 1) glyphs.(i))
  done;
  Run.v ~font ~size ~text ~glyphs ~xs ()

let ring = Path.append (rect 30. 30. 20. 20.) (rect 20. 20. 40. 40.)

let shapes =
  page 160. 100.
    [
      Picture.fill (Color.v 0.2 0.5 0.8) ring;
      Picture.fill ~rule:`Even_odd (Color.v 0.8 0.3 0.2)
        (Path.transform (Affine.translate 50. 0.) ring);
      Picture.clip
        (Path.circle (P2.v 130. 40.) 18.)
        (Picture.fill (Color.v 0.3 0.7 0.3) (rect 100. 20. 60. 20.));
      Picture.transform
        Affine.(translate 40. 80. * rotate (Float.pi /. 7.))
        (Picture.fill
           (Color.v ~alpha:0.6 0.5 0.2 0.7)
           (rect (-15.) (-8.) 30. 16.));
      Picture.opacity 0.5
        (Picture.group
           [
             Picture.fill Color.red (Path.circle (P2.v 100. 80.) 12.);
             Picture.fill Color.blue (Path.circle (P2.v 112. 80.) 12.);
           ]);
      Picture.clip ~rule:`Even_odd
        (Path.append
           (Path.circle (P2.v 145. 82.) 5.)
           (Path.circle (P2.v 145. 82.) 12.))
        (Picture.fill Color.black (rect 130. 65. 30. 35.));
    ]

let strokes =
  let zig = Path.polyline [| 10.; 25.; 40.; 55. |] [| 30.; 10.; 30.; 10. |] in
  let wave =
    Path.empty
    |> Path.move_to (P2.v 10. 60.)
    |> Path.cubic_to (P2.v 40. 20.) (P2.v 60. 100.) (P2.v 90. 60.)
    |> Path.quad_to (P2.v 110. 40.) (P2.v 130. 60.)
  in
  let dot =
    Path.empty |> Path.move_to (P2.v 150. 20.) |> Path.line_to (P2.v 150. 20.)
  in
  page 160. 100.
    [
      Picture.stroke (Stroke.v ~cap:`Butt ~join:`Miter 4.) Color.black zig;
      Picture.stroke
        (Stroke.v ~cap:`Round ~join:`Round 4.)
        (Color.v 0.8 0.2 0.2)
        (Path.transform (Affine.translate 50. 0.) zig);
      Picture.stroke
        (Stroke.v ~cap:`Square ~join:`Bevel 4.)
        (Color.v 0.2 0.2 0.8)
        (Path.transform (Affine.translate 0. 70.) zig);
      Picture.stroke
        (Stroke.v ~dash:[ 6.; 3.; 1.; 3. ] ~dash_offset:2. 2.)
        (Color.v 0.1 0.5 0.3) wave;
      Picture.stroke
        (Stroke.v ~cap:`Square ~dash:[ 0.; 5. ] 2.)
        Color.black
        (Path.polyline [| 100.; 150. |] [| 90.; 70. |]);
      Picture.stroke (Stroke.v 6.) (Color.v ~alpha:0.5 0.6 0.1 0.6) dot;
      Picture.stroke
        (Stroke.v ~join:`Miter 1.5)
        (Color.v 0.4 0.4 0.4)
        (Path.circle (P2.v 140. 50.) 10.);
      Picture.transform (Affine.scale 1. 0.5)
        (Picture.stroke (Stroke.v 3.) (Color.v 0.6 0.3 0.6)
           (Path.circle (P2.v 120. 170.) 8.));
    ]

let marks =
  let n = 24 in
  let xs = Array.init n (fun i -> 10. +. (6.25 *. float i)) in
  let ys = Array.init n (fun i -> 25. +. (12. *. sin (float i /. 2.))) in
  let fills =
    Array.init n (fun i -> Color.of_oklch 0.65 0.15 (0.26 *. float i))
  in
  let disc = Path.circle (P2.v 0. 0.) 2.5 in
  let marker =
    Picture.group
      [
        Picture.fill Color.black disc;
        Picture.stroke (Stroke.v 0.75) Color.white disc;
      ]
  in
  let tag rows =
    { Picture.id = Nx.Ptree.Path.v [ Index 0; Field "dots" ]; rows }
  in
  let image =
    Nx.init Nx.uint8 [| 3; 4; 4 |] (fun i ->
        if i.(2) = 3 then 255 - (40 * i.(0))
        else ((i.(0) * 80) + (i.(1) * 50) + (i.(2) * 30)) mod 256)
  in
  let bubble =
    Picture.stroke (Stroke.v 1.) (Color.v 0.2 0.3 0.6)
      (Path.circle (P2.v 0. 0.) 1.)
  in
  page 160. 100.
    [
      Picture.tag
        (tag (Rows (Array.init n Fun.id)))
        (Picture.stamp ~fills xs ys marker);
      Picture.stamp ~scales:[| 3.; 5.; 8. |] [| 20.; 45.; 80. |]
        [| 75.; 75.; 75. |] bubble;
      Picture.stamp
        ~strokes:[| Color.red; Color.blue |]
        [| 120.; 140. |] [| 75.; 75. |]
        (Picture.opacity 0.7 marker);
      Picture.glyphs Color.black (P2.v 10. 55.)
        (typeset "Hugin draws AV, Wa: 0.25" 11.);
      Picture.glyphs (Color.v 0.6 0.1 0.1) (P2.v 100. 95.)
        (typeset ~font:Font.bold "Bold" 9.);
      Picture.tag
        (tag (Cells { box = Box2.v 120. 5. 32. 24.; width = 4; height = 3 }))
        (Picture.image (Box2.v 120. 5. 32. 24.) image);
    ]

let pages = [ ("shapes", shapes); ("strokes", strokes); ("marks", marks) ]

(* Reading outputs *)

let of_base64 s =
  let value c =
    match c with
    | 'A' .. 'Z' -> Char.code c - 65
    | 'a' .. 'z' -> Char.code c - 71
    | '0' .. '9' -> Char.code c + 4
    | '+' -> 62
    | '/' -> 63
    | _ -> invalid_arg "of_base64"
  in
  let b = Buffer.create (String.length s * 3 / 4) in
  let acc = ref 0 and bits = ref 0 in
  String.iter
    (fun c ->
      if c <> '=' then begin
        acc := (!acc lsl 6) lor value c;
        bits := !bits + 6;
        if !bits >= 8 then begin
          bits := !bits - 8;
          Buffer.add_char b (Char.chr ((!acc lsr !bits) land 0xff))
        end
      end)
    s;
  Buffer.contents b

(* [load_png png] is the RGB pixels of the PNG file [png]. *)
let load_png png =
  let path = Filename.temp_file "hugin" ".png" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      Out_channel.with_open_bin path (fun oc -> output_string oc png);
      Nx_io.load_image path)

(* [path_of_data d] is the path of the SVG path data [d] of absolute moves,
   lines, cubics and closes. *)
let path_of_data d =
  let tokens = ref [] and b = Buffer.create 16 in
  let flush () =
    if Buffer.length b > 0 then begin
      tokens := `Num (float_of_string (Buffer.contents b)) :: !tokens;
      Buffer.clear b
    end
  in
  String.iter
    (fun c ->
      match c with
      | '0' .. '9' | '.' | 'e' -> Buffer.add_char b c
      | '-' ->
          flush ();
          Buffer.add_char b c
      | 'M' | 'L' | 'C' | 'Z' ->
          flush ();
          tokens := `Cmd c :: !tokens
      | _ -> flush ())
    d;
  flush ();
  let rec go q = function
    | `Cmd 'M' :: `Num x :: `Num y :: rest ->
        go (Path.move_to (P2.v x y) q) rest
    | `Cmd 'L' :: `Num x :: `Num y :: rest ->
        go (Path.line_to (P2.v x y) q) rest
    | `Cmd 'C'
      :: `Num a
      :: `Num b
      :: `Num c
      :: `Num d
      :: `Num x
      :: `Num y
      :: rest ->
        go (Path.cubic_to (P2.v a b) (P2.v c d) (P2.v x y) q) rest
    | `Cmd 'Z' :: rest -> go (Path.close q) rest
    | [] -> q
    | _ -> invalid_arg ("path_of_data: " ^ d)
  in
  go Path.empty (List.rev !tokens)

(* Generated pictures *)

open Windtrap

let coord = Gen.float_range (-20.) 120.
let unit = Gen.float_range 0. 1.

let gen_color =
  Gen.map
    (fun (r, g, b, alpha) -> Color.v ~alpha r g b)
    (Gen.quad unit unit unit
       (Gen.frequency [ (2, Gen.constant 1.); (1, unit) ]))

let gen_path =
  Gen.one_of
    [
      Gen.map
        (fun ((x, y), (w, h)) -> rect x y w h)
        (Gen.pair (Gen.pair coord coord)
           (Gen.pair (Gen.float_range 0. 40.) (Gen.float_range 0. 40.)));
      Gen.map
        (fun pts ->
          Path.polyline
            (Array.of_list (List.map fst pts))
            (Array.of_list (List.map snd pts)))
        (Gen.list ~size:(Gen.int_range 2 5) (Gen.pair coord coord));
      Gen.map
        (fun ((x, y), r) -> Path.circle (P2.v x y) r)
        (Gen.pair (Gen.pair coord coord) (Gen.float_range 0. 30.));
    ]

let gen_stroke =
  Gen.map
    (fun ((w, cap, join), dash) -> Stroke.v ~cap ~join ~dash w)
    (Gen.pair
       (Gen.triple (Gen.float_range 0. 4.)
          (Gen.of_list [ `Butt; `Round; `Square ])
          (Gen.of_list [ `Miter; `Round; `Bevel ]))
       (Gen.of_list [ []; [ 3.; 1. ]; [ 0.; 2.; 1. ] ]))

let tensor =
  Nx.init Nx.uint8 [| 2; 3; 4 |] (fun i ->
      (40 * i.(1)) + (90 * i.(0)) + (20 * i.(2)))

let gen_leaf =
  Gen.one_of
    [
      Gen.map
        (fun (rule, c, q) -> Picture.fill ~rule c q)
        (Gen.triple (Gen.of_list [ `Nonzero; `Even_odd ]) gen_color gen_path);
      Gen.map
        (fun (s, c, q) -> Picture.stroke s c q)
        (Gen.triple gen_stroke gen_color gen_path);
      Gen.map
        (fun (c, (x, y), (text, size)) ->
          Picture.glyphs c (P2.v x y) (typeset text size))
        (Gen.triple gen_color (Gen.pair coord coord)
           (Gen.pair
              (Gen.of_list [ "Wa"; "0.5"; "Hugin" ])
              (Gen.float_range 0. 20.)));
      Gen.map
        (fun ((x, y), (w, h)) -> Picture.image (Box2.v x y w h) tensor)
        (Gen.pair (Gen.pair coord coord)
           (Gen.pair (Gen.float_range 0. 30.) (Gen.float_range 0. 30.)));
    ]

let gen_map =
  Gen.map
    (fun ((dx, dy), (a, (sx, sy))) ->
      Affine.(translate dx dy * rotate a * scale sx sy))
    (Gen.pair (Gen.pair coord coord)
       (Gen.pair
          (Gen.of_list [ 0.; 0.5; Float.pi ])
          (Gen.pair (Gen.of_list [ 1.; 0.5; 2.; -1. ]) (Gen.of_list [ 1.; 3. ]))))

let gen_positions = Gen.list ~size:(Gen.int_range 1 4) (Gen.pair coord coord)

let rec gen_picture depth =
  if depth = 0 then gen_leaf
  else
    let sub = gen_picture (depth - 1) in
    Gen.frequency
      [
        (3, gen_leaf);
        (1, Gen.map Picture.group (Gen.list ~size:(Gen.int_range 0 3) sub));
        (1, Gen.map (fun (q, p) -> Picture.clip q p) (Gen.pair gen_path sub));
        (1, Gen.map (fun (m, p) -> Picture.transform m p) (Gen.pair gen_map sub));
        (1, Gen.map (fun (a, p) -> Picture.opacity a p) (Gen.pair unit sub));
        ( 1,
          Gen.map
            (fun ((pts, styled), p) ->
              let xs = Array.of_list (List.map fst pts)
              and ys = Array.of_list (List.map snd pts) in
              let n = Array.length xs in
              let each f = if styled then Some (Array.init n f) else None in
              Picture.stamp
                ?fills:(each (fun i -> Color.v (Float.of_int i /. 4.) 0.2 0.5))
                ?strokes:
                  (each (fun i ->
                       Color.v ~alpha:0.5 0.1 (Float.of_int i /. 4.) 0.3))
                ?scales:(each (fun i -> 1. +. Float.of_int i))
                xs ys p)
            (Gen.pair (Gen.pair gen_positions Gen.bool) sub) );
        ( 1,
          Gen.map
            (fun (field, p) ->
              let rows =
                match (p : Picture.t) with
                | Stamp { xs; _ } -> Array.init (Array.length xs) Fun.id
                | _ -> [| 4; 2 |]
              in
              Picture.tag
                {
                  id = Nx.Ptree.Path.v [ Index 1; Field field ];
                  rows = Rows rows;
                }
                p)
            (Gen.pair (Gen.of_list [ "a"; "b.c" ]) sub) );
      ]

let gen_picture = Gen.with_pp Picture.pp (gen_picture 3)

(* [extremes] is pictures whose scales reach either end of the range of floats
   in a stamp's picture, where nothing cuts geometry at the page. *)
let extremes =
  let disc = Path.circle (P2.v 0. 0.) 2. in
  let huge p = Picture.transform (Affine.scale 1e300 1e300) p in
  let within p = Picture.stamp [| 0. |] [| 0. |] p in
  [
    huge
      (within (Picture.stamp [| 1e10 |] [| 0. |] (Picture.fill Color.red disc)));
    huge (within (Picture.glyphs Color.red (P2.v 1e10 0.) (typeset "a" 10.)));
    huge (within (Picture.image (Box2.v 1e10 0. 1. 1.) tensor));
    huge (huge (Picture.stroke (Stroke.v 1.) Color.red disc));
    Picture.transform
      (Affine.scale 1e-300 1e-300)
      (Picture.stamp ~scales:[| 1e20 |] [| 1. |] [| 1. |]
         (Picture.stroke
            (Stroke.v ~dash:[ 3.; 1. ] 2.)
            Color.red
            (Path.polyline [| 0.; 50.; 80. |] [| 0.; 30.; 5. |])));
    Picture.stamp ~scales:[| 1e-300 |] [| 10. |] [| 10. |]
      (huge
         (huge
            (Picture.group
               [
                 Picture.stroke (Stroke.v 1.) Color.red disc;
                 Picture.stroke (Stroke.v ~dash:[ 3.; 1. ] 2.) Color.red disc;
               ])));
    (* Its pen, as large as the scale is small, stretched by a turned map whose
       coefficients a float holds but whose stretch it does not. *)
    Picture.stamp ~scales:[| 1e-300 |] [| 10. |] [| 10. |]
      (huge
         (Picture.transform
            Affine.(rotate (Float.pi /. 4.) * scale 2e8 2e8)
            (Picture.stroke (Stroke.v 1.) Color.red disc)));
  ]

(* [gen_extreme] is a generated picture under a transform and within a stamp,
   each scaling by a factor drawn near either end of the range of floats or
   not. *)
let gen_extreme =
  let k = Gen.of_list [ 1e-300; 1e-170; 1e-20; 1.; 1e20; 1e170; 1e300 ] in
  Gen.with_pp Picture.pp
    (Gen.map
       (fun ((a, b), (c, p)) ->
         Picture.transform (Affine.scale a a)
           (Picture.stamp
              ?scales:(Option.map (fun c -> [| c; 1. |]) c)
              [| 1.; 0. |] [| 1.; 0. |]
              (Picture.transform (Affine.scale b b) p)))
       (Gen.pair (Gen.pair k k) (Gen.pair (Gen.option k) gen_picture)))

(* [respell p] is a picture equal to [p] built anew: arrays and paths rebuilt,
   images and fonts copied, and zeros given the other sign. *)
let respell p =
  let flip v = if v = 0. then -.v else v in
  let path q =
    Path.fold
      ~move:(fun q x y -> Path.move_to (P2.v (flip x) (flip y)) q)
      ~line:(fun q x y -> Path.line_to (P2.v (flip x) (flip y)) q)
      ~cubic:(fun q x1 y1 x2 y2 x y ->
        Path.cubic_to (P2.v x1 y1) (P2.v x2 y2) (P2.v (flip x) (flip y)) q)
      ~close:Path.close Path.empty q
  in
  let font f =
    match Font.of_string (Font.bytes f) with
    | Ok f -> f
    | Error e -> Format.kasprintf failwith "%a" Font.pp_error e
  in
  let run r =
    let n = Run.length r in
    Run.v
      ~ys:(Array.init n (fun i -> flip (Run.y r i)))
      ~clusters:(Array.init n (Run.cluster r))
      ~font:(font (Run.font r))
      ~size:(Run.size r) ~text:(Run.text r)
      ~glyphs:(Array.init n (Run.glyph r))
      ~xs:(Array.init n (fun i -> flip (Run.x r i)))
      ()
  in
  let rec go (p : Picture.t) =
    match p with
    | Empty -> Picture.empty
    | Fill { rule; color; path = q } -> Picture.fill ~rule color (path q)
    | Stroke { stroke; color; path = q } -> Picture.stroke stroke color (path q)
    | Glyphs { color; at; run = r } ->
        Picture.glyphs color (P2.v (flip (P2.x at)) (flip (P2.y at))) (run r)
    | Image { box; pixels } -> Picture.image box (Nx.copy pixels)
    | Group ps -> Picture.group (List.map go ps)
    | Clip { rule; path = q; picture } ->
        Picture.clip ~rule (path q) (go picture)
    | Transform { m; picture } -> Picture.transform m (go picture)
    | Opacity { opacity; picture } -> Picture.opacity opacity (go picture)
    | Stamp { picture; xs; ys; scales; fills; strokes } ->
        Picture.stamp ?fills ?strokes ?scales (Array.map flip xs)
          (Array.map flip ys) (go picture)
    | Tag { tag; picture } -> Picture.tag tag (go picture)
  in
  go p

(* [golden file png] checks that the PNG file [png] shows the colours of [file],
   relative to the test's directory, each within a level: the machine's
   floating-point contractions move a few by one. On a difference it writes
   [png] to [file ^ ".corrected"] and, unless the run writes corrections
   ([--corrected]) for dune to diff and promote, fails. *)
let golden file png =
  let same =
    Sys.file_exists file
    &&
    let expected = Nx_io.load_image file and got = load_png png in
    Nx.shape expected = Nx.shape got
    &&
    let a = Nx.to_array expected and b = Nx.to_array got in
    let within = ref true in
    Array.iteri (fun i v -> if abs (v - b.(i)) > 1 then within := false) a;
    !within
  in
  if not same then begin
    Out_channel.with_open_bin (file ^ ".corrected") (fun oc ->
        output_string oc png);
    if not (Array.mem "--corrected" Sys.argv) then
      Windtrap.failf "%s differs from its golden by more than a level" file
  end
