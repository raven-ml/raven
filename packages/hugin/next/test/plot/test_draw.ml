(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_next
open Windtrap
module Raster = Hugin_next_vg_raster
module Svg = Hugin_next_vg_svg

(* Data *)

let f64 a = Nx.create Nx.float64 [| Array.length a |] a
let i32 a = Nx.create Nx.int32 [| Array.length a |] (Array.map Int32.of_int a)
let bools a = Nx.create Nx.bool [| Array.length a |] a
let nan = Float.nan
let path segs = Nx.Ptree.Path.v segs

(* Drawing *)

let drawn ?view ?theme ?(density = 1.) ?(size = Size.panels 100. 100.) f =
  render ?view ?theme ~density size f

(* [probe ~reduce bindings read] is a mark whose draw function records [read
   rows] for each panel it draws, and the records, latest first. Its legend
   swatches draw nothing. *)
let probe ?reduce bindings read =
  let seen = ref [] in
  let m =
    Mark.v ~name:"probe" ?reduce
      ~swatch:(fun _ -> Picture.empty)
      bindings
      (fun rows ->
        seen := read rows :: !seen;
        Picture.empty)
  in
  (m, seen)

let only seen =
  match !seen with [ x ] -> x | l -> failf "drawn %d times" (List.length l)

(* [rows_of bindings read] is [read] of the one panel of a mark of
   [bindings]. *)
let rows_of ?(size = Size.panels 100. 100.) bindings read =
  let m, seen = probe bindings read in
  ignore (drawn ~size m);
  only seen

(* Walking pictures *)

let rec fold f acc (p : Picture.t) =
  let acc = f acc p in
  match p with
  | Group ps -> List.fold_left (fold f) acc ps
  | Clip { picture; _ }
  | Transform { picture; _ }
  | Opacity { picture; _ }
  | Stamp { picture; _ }
  | Tag { picture; _ } ->
      fold f acc picture
  | Empty | Fill _ | Stroke _ | Glyphs _ | Image _ -> acc

let picture d = Renderable.picture (Drawing.renderable d)

let collect f d =
  List.rev
    (fold
       (fun acc p -> match f p with Some x -> x :: acc | None -> acc)
       [] (picture d))

(* [tags id d] is the rows and picture of each tag of [id] in [d]. *)
let tags id d =
  collect
    (function
      | Picture.Tag { tag; picture } when Nx.Ptree.Path.equal tag.id id ->
          Some (tag.rows, picture)
      | _ -> None)
    d

let images d =
  collect (function Picture.Image i -> Some (i.box, i.pixels) | _ -> None) d

let stamps d = collect (function Picture.Stamp s -> Some s.xs | _ -> None) d
let color = Testable.make ~pp:Color.pp ~equal:Color.equal
let text_t = Testable.make ~pp:Text.pp ~equal:Text.equal
let drawing = Testable.make ~pp:Drawing.pp ~equal:Drawing.equal
let is_nan = Float.is_nan
let floats = array float_exact
let close = float 1e-9

(* [within_one a b] states that two images differ by at most one colour level in
   each channel they both have. *)
let within_one ~msg a b =
  let channels t = (Nx.shape t).(2) in
  let c = Int.min (channels a) (channels b) in
  let a = Nx.slice [ A; A; R (0, c) ] a and b = Nx.slice [ A; A; R (0, c) ] b in
  equal ~msg (array int) (Nx.shape a) (Nx.shape b);
  let a = Nx.to_array a and b = Nx.to_array b in
  let worst = ref 0 in
  Array.iteri (fun i v -> worst := Int.max !worst (abs (v - b.(i)))) a;
  at_most ~msg int ~than:1 !worst

(* The colour a pixel of raster output holds for [c]. *)
let byte v = Float.to_int (Float.round (255. *. v))

let rgba c =
  [|
    byte (Color.r c); byte (Color.g c); byte (Color.b c); byte (Color.alpha c);
  |]

(* Rows *)

let dropped_rows () =
  let x = f64 [| 0.; 1.; 2.; 3. |] and y = f64 [| 0.; nan; 2.; 3. |] in
  let (xs, ys), (a, b), nx, ny, gx =
    rows_of
      [ Mark.bind Role.x (num x); Mark.bind Role.y (num y) ]
      (fun r ->
        ( Mark.points r,
          Mark.extent r `X,
          Mark.normalized r Role.x,
          Mark.normalized r Role.y,
          Mark.get r Role.x ))
  in
  List.iter
    (fun (name, v) -> is_true ~msg:name (is_nan v))
    [ ("x", xs.(1)); ("y", ys.(1)); ("lo", a.(1)); ("hi", b.(1)) ];
  List.iter
    (fun i -> is_false ~msg:(string_of_int i) (is_nan xs.(i) || is_nan ys.(i)))
    [ 0; 2; 3 ];
  (* The x domain is [0, 3], and the dropped row keeps its x. *)
  equal (option floats) (Some [| 0.; 1. /. 3.; 2. /. 3.; 1. |]) nx;
  equal (option floats) nx gx;
  equal (option floats) (Some [| 0.; nan; 2. /. 3.; 1. |]) ny

let missing_kinds =
  [
    ("nan", num (f64 [| 1.; nan; 2. |]));
    ("infinity", num (f64 [| 1.; Float.infinity; 2. |]));
    ("negative infinity", num (f64 [| 1.; Float.neg_infinity; 2. |]));
    ("a mask", num ~valid:(bools [| true; false; true |]) (f64 [| 1.; 5.; 2. |]));
    ("zero on a log scale", num ~scale:(Scale.log ()) (f64 [| 1.; 0.; 2. |]));
    ( "a negative value on a log scale",
      num ~scale:(Scale.log ()) (f64 [| 1.; -1.; 2. |]) );
  ]

let missing_drops (_, y) =
  let _, ys =
    rows_of
      [ Mark.bind Role.x (num (f64 [| 0.; 1.; 2. |])); Mark.bind Role.y y ]
      Mark.points
  in
  is_true (is_nan ys.(1));
  is_false (is_nan ys.(0) || is_nan ys.(2))

(* Extents, from the spec of [Mark.extent]. *)
let extents =
  let domain = Scale.linear ~domain:(-2., 4.) () in
  let abc = Scale.band ~domain:(Scale.Labels [| "a"; "b"; "c" |]) () in
  [
    ( "a continuous x alone runs from zero",
      [ Mark.bind Role.x (num ~scale:domain (f64 [| 1.; 4. |])) ],
      ([| 1. /. 3.; 1. /. 3. |], [| 0.5; 1. |]) );
    ( "on a log scale a length starts at the domain's lower end",
      [
        Mark.bind Role.x
          (num ~scale:(Scale.log ~domain:(1., 100.) ()) (f64 [| 10. |]));
      ],
      ([| 0. |], [| 0.5 |]) );
    ( "a band x alone covers its band",
      [ Mark.bind Role.x (strings [| "a"; "b" |]) ],
      ([| 0.; 0.5 |], [| 0.5; 1. |]) );
    ( "x and x2 cover their hull",
      [
        Mark.bind Role.x
          (num ~scale:(Scale.linear ~domain:(0., 4.) ()) (f64 [| 3. |]));
        Mark.bind Role.x2 (num (f64 [| 1. |]));
      ],
      ([| 0.25 |], [| 0.75 |]) );
    ( "band ends cover the hull of their bands",
      [
        Mark.bind Role.x (strings ~scale:abc [| "a" |]);
        Mark.bind Role.x2 (strings [| "c" |]);
      ],
      ([| 0. |], [| 1. |]) );
    ( "a constant x covers its value",
      [ Mark.bind Role.x (const 0.3) ],
      ([| 0.3 |], [| 0.3 |]) );
    ("without x a row covers the panel", [], ([| 0. |], [| 1. |]));
    ( "an end beyond the domain is clamped into it",
      [
        Mark.bind Role.x
          (num ~scale:(Scale.linear ~domain:(0., 4.) ()) (f64 [| 2. |]));
        Mark.bind Role.x2 (num (f64 [| 6. |]));
      ],
      ([| 0.5 |], [| 1. |]) );
    ( "a length beyond the domain stops at its edge",
      [
        Mark.bind Role.x
          (num ~scale:(Scale.linear ~domain:(-4., 4.) ()) (f64 [| -6. |]));
      ],
      ([| 0.5 |], [| 0. |]) );
  ]

let extent_case (_, bindings, expected) =
  let got = rows_of bindings (fun r -> Mark.extent r `X) in
  equal (pair (array close) (array close)) expected got

let series_lengths bindings =
  rows_of bindings (fun r -> List.map Mark.index (Mark.series r))

let series =
  [
    test "a categorical x does not split a series" (fun () ->
        equal
          (list (array int))
          [ [| 0; 1; 2 |] ]
          (series_lengths
             [
               Mark.bind Role.x (strings [| "a"; "b"; "c" |]);
               Mark.bind Role.y (num (f64 [| 1.; 2.; 3. |]));
             ]));
    test "a categorical stroke splits the rows" (fun () ->
        equal
          (list (array int))
          [ [| 0; 2 |]; [| 1; 3 |] ]
          (series_lengths
             [
               Mark.bind Role.y (num (f64 [| 1.; 2.; 3.; 4. |]));
               Mark.bind Role.stroke
                 (cat ~labels:[| "p"; "q" |] (i32 [| 0; 1; 0; 1 |]));
             ]));
    test "the leading axes split the rows" (fun () ->
        equal
          (list (array int))
          [ [| 0; 1; 2 |]; [| 3; 4; 5 |] ]
          (series_lengths
             [ Mark.bind Role.y (num (Nx.zeros Nx.float64 [| 2; 3 |])) ]));
    test "a missing category is a series of its own" (fun () ->
        equal
          (list (array int))
          [ [| 0; 2 |]; [| 1 |] ]
          (series_lengths
             [
               Mark.bind Role.y (num (f64 [| 1.; 2.; 3. |]));
               Mark.bind Role.stroke (cat ~labels:[| "p" |] (i32 [| 0; 5; 0 |]));
             ]));
  ]

let missing_values =
  [
    test "a missing colour is its scale's unknown colour" (fun () ->
        let c =
          rows_of
            [
              Mark.bind Role.fill
                (num
                   ~scale:(Scale.linear ~unknown:Color.red ())
                   (f64 [| 1.; nan |]));
            ]
            (fun r -> Mark.get r Role.fill)
        in
        equal color Color.red (Option.get c).(1));
    test "a missing symbol is the first of its scale" (fun () ->
        let symbols = [| Symbol.square; Symbol.circle |] in
        let s =
          rows_of
            [
              Mark.bind Role.symbol
                (cat ~scale:(Scale.band ~symbols ()) ~labels:[| "a"; "b" |]
                   (i32 [| 1; 9 |]));
            ]
            (fun r -> Mark.get r Role.symbol)
        in
        is_true (Symbol.equal Symbol.square (Option.get s).(1)));
    test "a missing text is empty" (fun () ->
        let t =
          rows_of
            [ Mark.bind Role.text (num (f64 [| 1.5; nan |])) ]
            (fun r -> Mark.get r Role.text)
        in
        equal text_t (Text.v "") (Option.get t).(1));
    test "a value on a band scale that no step holds is missing" (fun () ->
        let f =
          rows_of
            [
              Mark.bind Role.fill
                (strings
                   ~scale:(Scale.band ~padding:0.5 ~unknown:Color.red ())
                   [| "a"; "b" |]);
            ]
            (fun r -> Mark.range r Role.fill)
        in
        equal color Color.red ((Option.get f) 0.01));
  ]

(* [get] is [range] applied to [normalized], on any data. *)
let get_is_range_of_normalized (values, mapped) =
  let x = f64 (Array.of_list values) in
  let fill = num x in
  let fill = if mapped then map_range Color.contrast fill else fill in
  let got, f, us =
    rows_of
      [ Mark.bind Role.fill fill ]
      (fun r ->
        ( Mark.get r Role.fill,
          Mark.range r Role.fill,
          Mark.normalized r Role.fill ))
  in
  cover "a missing value"
    (List.exists (fun v -> not (Float.is_finite v)) values);
  cover "mapped" mapped;
  let f = require_some f and us = require_some us in
  equal (option (array color)) (Some (Array.map f us)) got

let band_get_is_range (codes, mapped) =
  let fill = cat ~labels:[| "a"; "b"; "c" |] (i32 (Array.of_list codes)) in
  let fill = if mapped then map_range Color.contrast fill else fill in
  let got, f, us =
    rows_of
      [ Mark.bind Role.fill fill ]
      (fun r ->
        ( Mark.get r Role.fill,
          Mark.range r Role.fill,
          Mark.normalized r Role.fill ))
  in
  cover "a code outside the labels"
    (List.exists (fun c -> c < 0 || c > 2) codes);
  let f = require_some f and us = require_some us in
  equal (option (array color)) (Some (Array.map f us)) got

let value_gen =
  Gen.frequency
    [
      (8, Gen.float_range (-10.) 10.);
      (1, Gen.of_list ~pp:Format.pp_print_float [ nan; Float.infinity ]);
    ]

let rows =
  group "Rows"
    [
      test
        "a dropped row is at nan in points and extent, and keeps its own \
         values in get and normalized"
        dropped_rows;
      cases ~name:fst "a missing value drops its row" missing_kinds
        missing_drops;
      cases ~name:(fun (n, _, _) -> n) "extent" extents extent_case;
      group "series" series;
      group "missing values" missing_values;
      prop "get is the range of normalized on a continuous scale"
        (Gen.pair (Gen.list ~size:(Gen.int_range 1 6) value_gen) Gen.bool)
        get_is_range_of_normalized;
      prop "get is the range of normalized on a band scale"
        (Gen.pair
           (Gen.list ~size:(Gen.int_range 1 6) (Gen.int_range (-1) 4))
           Gen.bool)
        band_get_is_range;
      test "the ticks of a reversed scale increase" (fun () ->
          let t =
            rows_of
              [
                Mark.bind Role.y
                  (num
                     ~scale:(Scale.linear ~reverse:true ())
                     (f64 [| 0.; 10. |]));
              ]
              (fun r -> Mark.ticks r Role.y)
          in
          let t = require_some t in
          greater int ~than:1 (Array.length t);
          Array.iteri
            (fun i u -> if i > 0 then greater float_exact ~than:t.(i - 1) u)
            t);
      test "a swatch has one row of the legend's shape, index and id" (fun () ->
          let swatches = ref [] in
          let m =
            Mark.v ~name:"probe"
              ~swatch:(fun r ->
                swatches := (Mark.shape r, Mark.index r, Mark.id r) :: !swatches;
                Picture.empty)
              [
                Mark.bind Role.x (num (f64 [| 0.; 1.; 2. |]));
                Mark.bind Role.fill
                  (cat ~labels:[| "a"; "b"; "c" |] (i32 [| 0; 1; 2 |]));
              ]
              (fun _ -> Picture.empty)
          in
          ignore (drawn m);
          let legend = path [ Field "legend"; Field "color"; Field "cat" ] in
          let id =
            Testable.make ~pp:Nx.Ptree.Path.pp ~equal:Nx.Ptree.Path.equal
          in
          equal
            (list (triple (array int) (array int) id))
            (List.init 3 (fun k -> ([| 3 |], [| k |], legend)))
            (List.rev !swatches));
      test "Mark.warn warns under the mark's id" (fun () ->
          let m =
            Mark.v ~name:"probe" [] (fun r ->
                Mark.warn r "no luck";
                Picture.empty)
          in
          let ws = Drawing.warnings (drawn (layer [ m ])) in
          equal (list string) [ "no luck" ]
            (List.filter_map
               (fun (id, s) ->
                 if Nx.Ptree.Path.equal id (path [ Index 0 ]) then Some s
                 else None)
               ws));
      test "text no face draws warns" (fun () ->
          let d =
            drawn
              (Hugin_next.text ~x:(const 0.5) ~y:(const 0.5)
                 ~text:(strings [| "\u{10FFFD}" |])
                 ())
          in
          equal int 1 (List.length (Drawing.warnings d)));
    ]

(* The domain: positions are clipped to it, ink is not *)

let unit = Scale.linear ~domain:(0., 1.) ()

(* [clips d] is the number of clips in [d]. *)
let clips d =
  List.length (collect (function Picture.Clip _ -> Some () | _ -> None) d)

let page_box =
  Testable.make
    ~pp:(Format.pp_print_option Box2.pp)
    ~equal:
      (Option.equal (fun a b ->
           let near x y = Float.abs (x -. y) <= 1e-9 in
           near (Box2.minx a) (Box2.minx b)
           && near (Box2.miny a) (Box2.miny b)
           && near (Box2.maxx a) (Box2.maxx b)
           && near (Box2.maxy a) (Box2.maxy b)))

let domain =
  group "Domain"
    [
      test "positions keep a position outside the domain, points drop it"
        (fun () ->
          let (us, _), (xs, _) =
            rows_of
              [
                Mark.bind Role.x (num ~scale:unit (f64 [| 0.5; 1.5; 1.; nan |]));
                Mark.bind Role.y (const 0.5);
              ]
              (fun r -> (Mark.positions r, Mark.points r))
          in
          equal floats [| 0.5; 1.5; 1.; nan |] us;
          equal (array bool)
            [| false; true; false; true |]
            (Array.map is_nan xs));
      test "an extent wholly beyond the domain covers nothing" (fun () ->
          let ext =
            rows_of
              [
                Mark.bind Role.x
                  (num ~scale:(Scale.linear ~domain:(0., 4.) ()) (f64 [| 5. |]));
                Mark.bind Role.x2 (num (f64 [| 6. |]));
              ]
              (fun r -> Mark.extent r `X)
          in
          equal (pair floats floats) ([| nan |], [| nan |]) ext);
      test "project cuts a path at the domain's edges" (fun () ->
          let got, edge, start =
            rows_of [] (fun r ->
                let p = Path.polyline [| 0.5; 2. |] [| 0.5; 0.5 |] in
                ( Path.bounds (Mark.project r p),
                  Coord.point (Mark.projection r) 1. 0.5,
                  Coord.point (Mark.projection r) 0.5 0.5 ))
          in
          equal page_box (Some (Box2.of_pts start edge)) got);
      test "a dot at the domain's corner is drawn whole, one outside not at all"
        (fun () ->
          let d =
            drawn
              (layer
                 [
                   dot
                     ~x:(num ~scale:unit (f64 [| 1.; 1.5 |]))
                     ~y:(num ~scale:unit (f64 [| 1.; 0.5 |]))
                     ();
                 ])
          in
          equal int 0 (clips d);
          match stamps d with
          | [ xs ] -> equal (array bool) [| false; true |] (Array.map is_nan xs)
          | l -> failf "%d stamps" (List.length l));
      test "a line is cut at the domain's edge, its stroke drawn whole"
        (fun () ->
          let f =
            layer
              [
                line
                  ~x:(num ~scale:unit (f64 [| 0.; 2. |]))
                  ~y:(num ~scale:unit (f64 [| 0.5; 0.5 |]))
                  ();
              ]
          in
          let size = Size.panels 100. 100. in
          let box = (List.hd (Layout.panels (layout size (resolve f)))).box in
          let strokes =
            List.concat_map
              (fun (_, p) ->
                fold
                  (fun acc p ->
                    match p with
                    | Picture.Stroke { path; _ } -> Path.bounds path :: acc
                    | _ -> acc)
                  [] p)
              (tags (path [ Index 0 ]) (drawn ~size f))
          in
          equal int 0 (clips (drawn ~size f));
          let mid = (Box2.miny box +. Box2.maxy box) /. 2. in
          equal (list page_box)
            [ Some (Box2.v (Box2.minx box) mid (Box2.w box) 0.) ]
            strokes);
      test "an image beyond its domain keeps its pixels, clipped to the panel"
        (fun () ->
          let zoom =
            dot
              ~x:(num ~scale:(Scale.linear ~domain:(1., 3.) ()) (f64 [| 2. |]))
              ~y:(num (f64 [| 2. |]))
              ()
          in
          let f = layer [ image (Nx.ones Nx.float32 [| 4; 4 |]); zoom ] in
          let size = Size.panels 100. 100. in
          let box = (List.hd (Layout.panels (layout size (resolve f)))).box in
          let d = drawn ~size f in
          equal int 1 (clips d);
          match images d with
          | [ (b, _) ] -> equal (float 1e-9) (2. *. Box2.w box) (Box2.w b)
          | l -> failf "%d images" (List.length l));
      test "a text anchored at the domain's corner is drawn whole" (fun () ->
          let d =
            drawn
              (layer
                 [
                   Hugin_next.text ~dy:5.
                     ~x:(num ~scale:unit (f64 [| 1. |]))
                     ~y:(num ~scale:unit (f64 [| 1. |]))
                     ~text:(strings [| "0.851" |]) ();
                 ])
          in
          equal int 0 (clips d);
          greater int ~than:0
            (List.length
               (collect (function Picture.Glyphs _ -> Some () | _ -> None) d)));
    ]

(* Drawings *)

let pair_figure b =
  grid [ [ dot ~x:(num (f64 [| 1.; 2. |])) ~y:(num (f64 [| 1.; 2. |])) (); b ] ]

let other () = line ~y:(num (f64 [| 3.; 1.; 2. |])) ()

let reuse =
  let size = Size.panels 60. 40. in
  let fresh ?theme ?(density = 1.) f =
    draw ~density (layout ?theme size (resolve f))
  in
  let again ?(theme = Theme.default) ?(density = 1.) ~prev f =
    draw ~prev ~density (layout ~theme size (resolve f))
  in
  [
    ( "with nothing changed",
      fun () ->
        let f = pair_figure (other ()) in
        (again ~prev:(fresh f) f, fresh f) );
    ( "after a tensor changes",
      fun () ->
        let prev = fresh (pair_figure (other ())) in
        let f = pair_figure (line ~y:(num (f64 [| 0.; 5.; 2. |])) ()) in
        (again ~prev f, fresh f) );
    ( "after the theme changes",
      fun () ->
        let f = pair_figure (other ()) in
        let theme = Theme.v ~accent:Color.red () in
        (again ~theme ~prev:(fresh f) f, fresh ~theme f) );
    ( "after the density changes",
      fun () ->
        let f = pair_figure (image (Nx.ones Nx.float32 [| 4; 4 |])) in
        (again ~density:2. ~prev:(fresh f) f, fresh ~density:2. f) );
    ( "after a cell is named",
      fun () ->
        let prev = fresh (pair_figure (other ())) in
        let f = pair_figure (name "val" (other ())) in
        (again ~prev f, fresh f) );
    ( "after a layer child is named",
      fun () ->
        let a = dot ~x:(const 0.5) ~y:(const 0.5) () in
        let prev = fresh (layer [ a; other () ]) in
        let f = layer [ a; name "fit" (other ()) ] in
        (again ~prev f, fresh f) );
  ]

(* Dots fill and draw no text, so every stroke is an axis line or tick and every
   run of glyphs a title or label: the figure's, the axes' and the legend's
   titles are four runs. *)
let inked () =
  let ink = Color.v ~alpha:0.8 0.2 0.1 0.3 in
  let part a = Color.with_alpha (0.8 *. a) ink in
  let title t = Text.v t in
  let f =
    dot
      ~x:(num ~title:(title "x") (f64 [| 0.; 1.; 2. |]))
      ~y:(num ~title:(title "y") (f64 [| 0.; 1.; 2. |]))
      ~fill:(strings ~title:(title "kind") [| "a"; "b"; "a" |])
      ()
    |> Hugin_next.title (title "T")
  in
  let d = drawn ~theme:(Theme.v ~ink ()) f in
  let strokes =
    collect (function Picture.Stroke s -> Some s.color | _ -> None) d
  in
  let titles, labels =
    List.partition (Color.equal ink)
      (collect (function Picture.Glyphs g -> Some g.color | _ -> None) d)
  in
  at_least int ~than:2 (List.length strokes);
  List.iter (equal color (part 0.6)) strokes;
  equal int 4 (List.length titles);
  at_least int ~than:2 (List.length labels);
  List.iter (equal color (part 0.75)) labels

let drawings =
  group "Drawing"
    [
      cases ~name:Float.to_string "draw raises on the density"
        [ 0.; -1.; nan; Float.infinity ] (fun density ->
          raises_match Exn.invalid_arg (fun () ->
              draw ~density (layout (Size.panels 10. 10.) (resolve (other ())))));
      test "the page is the layout's size" (fun () ->
          let l = layout (Size.panels 80. 50.) (resolve (other ())) in
          let r = Drawing.renderable (draw ~density:1. l) in
          equal
            (pair float_exact float_exact)
            (Layout.size l)
            (Renderable.w r, Renderable.h r));
      test "a drawing with images is equal to itself drawn again" (fun () ->
          let l =
            layout (Size.panels 40. 40.)
              (resolve (image (Nx.ones Nx.uint8 [| 3; 3; 3 |])))
          in
          equal drawing (draw ~density:1. l) (draw ~density:1. l));
      test "drawings of other data differ" (fun () ->
          not_equal drawing
            (drawn (other ()))
            (drawn (line ~y:(num (f64 [| 1.; 2. |])) ())));
      cases ~name:fst "draw ~prev is the drawing drawn afresh" reuse
        (fun (_, f) ->
          let reused, fresh = f () in
          equal drawing fresh reused);
      test "render draws the layout of the resolved figure" (fun () ->
          let f = other () in
          equal drawing
            (draw ~density:2. (layout (Size.figure 200. 100.) (resolve f)))
            (render (Size.figure 200. 100.) f));
      test "the warnings of resolving are the drawing's" (fun () ->
          let view = View.(set (number "unread" ~init:0.) 1. empty) in
          let d = drawn ~view (other ()) in
          equal int 1 (List.length (Drawing.warnings d)));
      test "a stamp of one instance per row is tagged instance by instance"
        (fun () ->
          let d =
            drawn
              (layer
                 [
                   dot
                     ~x:(num (f64 [| 1.; 2.; 3. |]))
                     ~y:(num (f64 [| 1.; 2.; 3. |]))
                     ();
                 ])
          in
          match tags (path [ Index 0 ]) d with
          | [ (Picture.Rows rows, Picture.Stamp _) ] ->
              equal (array int) [| 0; 1; 2 |] rows
          | _ -> fail "no tagged stamp");
      test "a facet channel puts each row in its panel" (fun () ->
          let m, seen =
            probe
              [
                Mark.bind Role.y (num (Nx.zeros Nx.float64 [| 2; 3 |]));
                Mark.bind Role.fx (dim 0);
              ]
              Mark.index
          in
          ignore (drawn m);
          equal
            (slist (array int) compare)
            [ [| 0; 1; 2 |]; [| 3; 4; 5 |] ]
            !seen);
      test "a facet a mark leaves unbound puts its rows in every panel"
        (fun () ->
          let m, seen =
            probe [ Mark.bind Role.fx (strings [| "a"; "b"; "a" |]) ] Mark.index
          in
          let rows = dot ~x:(const 0.5) ~y:(const 0.5) in
          ignore (drawn (layer [ rows ~fy:(strings [| "p"; "q" |]) (); m ]));
          equal
            (slist (array int) compare)
            [ [| 0; 2 |]; [| 0; 2 |]; [| 1 |]; [| 1 |] ]
            !seen);
      test "a row whose facet value is missing is in no panel" (fun () ->
          let valid = Nx.create Nx.bool [| 3 |] [| true; false; true |] in
          let m, seen =
            probe
              [
                Mark.bind Role.y (num (Nx.zeros Nx.float64 [| 3 |]));
                Mark.bind Role.fx (dim ~valid 0);
              ]
              Mark.index
          in
          ignore (drawn m);
          equal (slist (array int) compare) [ [| 0 |]; [| 2 |] ] !seen);
      test "a facet constant draws in its panel only" (fun () ->
          let m, seen = probe [ Mark.bind Role.fx (const "b") ] Mark.index in
          ignore
            (drawn
               (layer
                  [
                    dot ~x:(const 0.5) ~y:(const 0.5)
                      ~fx:(strings [| "a"; "b" |])
                      ();
                    m;
                  ]));
          equal (list (array int)) [ [| 0 |] ] !seen);
      test "titles take the ink, labels and axes a part of its opacity" inked;
      test "a dot has the area of a circle 0.6 em across" (fun () ->
          (* A stamp scales a symbol of unit area by the root of its area: a
             circle 6 points across at the default size of 10 points. *)
          let d = drawn (dot ~x:(const 0.5) ~y:(const 0.5) ()) in
          let scales =
            collect (function Picture.Stamp s -> Some s.scales | _ -> None) d
          in
          equal
            (list (option floats))
            [ Some [| Float.sqrt (Float.pi *. 9.) |] ]
            scales);
      test "the paper is painted first, and a transparent one not at all"
        (fun () ->
          let first theme =
            match picture (drawn ~theme (other ())) with
            | Group (p :: _) -> p
            | p -> p
          in
          (match first (Theme.v ~paper:Color.blue ()) with
          | Fill { color = c; _ } -> equal color Color.blue c
          | _ -> fail "no paper");
          match first (Theme.v ~paper:Color.transparent ()) with
          | Fill { color = c; _ } when Color.equal c Color.transparent ->
              fail "a transparent paper"
          | _ -> ());
    ]

(* Output *)

let read_file p = In_channel.with_open_bin p In_channel.input_all

let output =
  group "Output"
    [
      test "save refuses another extension and writes nothing" (fun () ->
          let dir = temp_dir () in
          let file = Filename.concat dir "f.jpg" in
          raises_match Exn.invalid_arg (fun () -> save file (other ()));
          is_false (Sys.file_exists file));
      cases ~name:fst "save writes the format its extension names"
        [ ("f.png", "\x89PNG"); ("f.SVG", "<svg"); ("f.pdf", "%PDF") ]
        (fun (name, magic) ->
          let file = Filename.concat (temp_dir ()) name in
          save file (other ());
          let data = read_file file in
          contains ~sub:magic
            (String.sub data 0 (Int.min 200 (String.length data))));
      test "a PNG records its density and colour space" (fun () ->
          let file = Filename.concat (temp_dir ()) "f.png" in
          save ~density:3. file (other ());
          let data = read_file file in
          contains ~sub:"pHYs" data;
          contains ~sub:"sRGB" data);
      test "save gives each warning before writing" (fun () ->
          let file = Filename.concat (temp_dir ()) "f.svg" in
          let view = View.(set (number "unread" ~init:0.) 1. empty) in
          let seen = ref [] in
          save ~view
            ~warn:(fun w ->
              seen := w :: !seen;
              is_false (Sys.file_exists file))
            file (other ());
          equal int 1 (List.length !seen));
      test "pp prints a summary where tags are not shown" (fun () ->
          let warned =
            dot
              ~x:(num (f64 [| 1.; 2. |]))
              ~y:(num (f64 [| 1.; 2. |]))
              ~fill:(cat ~labels:[| "a" |] (i32 [| 0; 3 |]))
              ()
          in
          equal string "hugin figure" (Format.asprintf "%a" pp (other ()));
          equal string "hugin figure (1 warning)"
            (Format.asprintf "%a" pp warned));
      test "pp opens a display tag holding the SVG document save writes"
        (fun () ->
          let b = Buffer.create 1024 in
          let ppf = Format.formatter_of_buffer b in
          let tags = ref [] in
          Format.pp_set_tags ppf true;
          Format.pp_set_print_tags ppf false;
          Format.pp_set_mark_tags ppf true;
          Format.pp_set_formatter_stag_functions ppf
            {
              (Format.pp_get_formatter_stag_functions ppf ()) with
              mark_open_stag =
                (function
                | Format.String_tag s ->
                    tags := s :: !tags;
                    ""
                | _ -> "");
              mark_close_stag = (fun _ -> "");
            };
          Format.fprintf ppf "%a@?" pp (other ());
          let file = Filename.concat (temp_dir ()) "f.svg" in
          save file (other ());
          let svg = In_channel.with_open_bin file In_channel.input_all in
          match !tags with
          | [ s ] -> equal text ("quill.display\nimage/svg+xml\n\n" ^ svg) s
          | l -> failf "%d tags" (List.length l));
    ]

(* Reducers *)

(* [m4_reference w us ys] is what M4 keeps of one series: in each of the [w]
   columns, and in the bins beyond each side, the first and last rows and the
   first rows of the lowest and highest values. *)
let m4_reference column ys =
  let n = Array.length ys in
  let bins = Hashtbl.create 64 in
  for i = 0 to n - 1 do
    let c = column i in
    let first, last, lo, hi =
      match Hashtbl.find_opt bins c with
      | None -> (i, i, i, i)
      | Some (f, _, lo, hi) ->
          ( f,
            i,
            (if ys.(i) < ys.(lo) then i else lo),
            if ys.(i) > ys.(hi) then i else hi )
    in
    Hashtbl.replace bins c (first, last, lo, hi)
  done;
  Hashtbl.fold (fun _ (a, b, c, d) acc -> a :: b :: c :: d :: acc) bins []
  |> List.sort_uniq Int.compare |> Array.of_list

(* A series on x in [0, 1000] in a panel 50 device pixels wide: column [i / 20]
   holds row [i]. *)
let m4_series ys =
  let x = index ~scale:(Scale.linear ~domain:(0., 1000.) ()) (-1) in
  let m, seen =
    probe ~reduce:Mark.m4
      [ Mark.bind Role.x x; Mark.bind Role.y (num ys) ]
      Mark.index
  in
  ignore (drawn ~size:(Size.panels 50. 50.) m);
  only seen

let ys n = Nx.Rng.normal (Nx.Rng.key 3) Nx.float64 [| n |]

(* [dots_with ~alpha reduce n] is a mark of [n] dots filled at [alpha], reduced
   by [reduce]. *)
let dots_with ?(alpha = 0.3) reduce n =
  let e = Nx.Rng.uniform (Nx.Rng.key 4) Nx.float64 [| n; 2 |] in
  let disc = Picture.fill Color.black (Path.circle (P2.v 0. 0.) 1.5) in
  Mark.v ~name:"dots" ?reduce
    [
      Mark.bind Role.x (num Nx.(slice [ A; I 0 ] e));
      Mark.bind Role.y (num Nx.(slice [ A; I 1 ] e));
    ]
    (fun r ->
      let xs, ys = Mark.points r in
      Picture.stamp
        ~fills:(Array.make (Mark.length r) (Color.v ~alpha 0.1 0.3 0.8))
        xs ys disc)

(* A panel of 20 by 20 points has 400 device pixels at density 1 and 1,600 at
   density 2, fewer than the dots, so raster applies at both. *)
let raster_as_drawn =
  prop "raster draws what the picture paints, within a level" ~count:30
    (Gen.triple
       (Gen.of_list ~pp:Format.pp_print_float [ 0.02; 0.3; 1. ])
       (Gen.of_list ~pp:Format.pp_print_float [ 1.; 2. ])
       (Gen.int_range 1_601 4_000))
    (fun (alpha, density, n) ->
      cover "the lightest opacity" (alpha = 0.02);
      cover "an opaque opacity" (alpha = 1.);
      let size = Size.panels 20. 20. in
      let page reduce =
        let d = drawn ~density ~size (dots_with ~alpha reduce n) in
        (d, Raster.render ~density (Drawing.renderable d))
      in
      let reduced, got = page (Some Mark.raster) in
      let _, expected = page None in
      equal int 1 (List.length (images reduced));
      within_one ~msg:"pixels" expected got)

(* Heatmaps *)

let viridis_of r v =
  let s = Resolved.scale r (Scale.linear ~name:"color" ()) in
  Scheme.color Scheme.viridis (Scale.normalize s v)

(* [cells_image h w at] is the image whose pixel [(i, j)] is [rgba (at i j)]. *)
let cells_image h w at =
  let a = Array.make (h * w * 4) 0 in
  for i = 0 to h - 1 do
    for j = 0 to w - 1 do
      Array.blit (rgba (at i j)) 0 a (((i * w) + j) * 4) 4
    done
  done;
  Nx.create Nx.uint8 [| h; w; 4 |] a

let heatmap z = layer [ rect ~x:(dim 1) ~y:(dim 0) ~fill:(num z) () ]

(* [cells d] is the box and pixels of each cells image of the mark [0] of
   [d]. *)
let cells d =
  List.filter_map
    (function
      | Picture.Cells _, Picture.Image i -> Some (i.box, i.pixels) | _ -> None)
    (tags (path [ Index 0 ]) d)

(* [same_raster page a b] states that the pictures [a] and [b] draw the same
   pixels on [page]. *)
let same_raster (w, h) a b =
  let r p = Raster.render ~density:1. (Renderable.v w h p) in
  equal (array int) (Nx.to_array (r a)) (Nx.to_array (r b))

let reducers =
  group "Reducers"
    [
      test "m4 keeps every row at four rows per column" (fun () ->
          equal int 200 (Array.length (m4_series (ys 200))));
      test "m4 keeps each column's first, last, lowest and highest rows"
        (fun () ->
          let n = 1001 in
          let y = ys n in
          equal (array int)
            (m4_reference (fun i -> i / 20) (Nx.to_array y))
            (m4_series y));
      test "m4 keeps every row of a series with a missing value" (fun () ->
          let y = Nx.concatenate ~axis:0 [ ys 1000; f64 [| nan |] ] in
          equal int 1001 (Array.length (m4_series y)));
      test "m4 keeps every row of a series whose x is not monotone" (fun () ->
          let x =
            Nx.concatenate ~axis:0
              [ Nx.linspace Nx.float64 0. 1. 1000; f64 [| 0. |] ]
          in
          let m, seen =
            probe ~reduce:Mark.m4
              [ Mark.bind Role.x (num x); Mark.bind Role.y (num (ys 1001)) ]
              Mark.length
          in
          ignore (drawn ~size:(Size.panels 50. 50.) m);
          equal int 1001 (only seen));
      raster_as_drawn;
      cases
        ~name:(fun (n, _, _) -> Printf.sprintf "%d dots in %s" n "a panel")
        "raster draws dots as one image past its thresholds"
        [
          (20_000, Size.panels 400. 400., false);
          (20_001, Size.panels 400. 400., true);
          (100, Size.panels 10. 10., false);
          (101, Size.panels 10. 10., true);
        ]
        (fun (n, size, image) ->
          let d = drawn ~size (dots_with (Some Mark.raster) n) in
          equal bool image (images d <> []);
          equal bool (not image) (stamps d <> []));
      test
        "raster paints the panel's picture at the density, over the device \
         pixels it reaches" (fun () ->
          let density = 2. and size = Size.panels 60.3 40.7 in
          let pictures = ref [] in
          let f =
            layer
              [
                Mark.v ~name:"dots" ~reduce:Mark.raster
                  [
                    Mark.bind Role.x
                      (num
                         (Nx.Rng.uniform (Nx.Rng.key 5) Nx.float64 [| 25_000 |]));
                    Mark.bind Role.y
                      (num
                         (Nx.Rng.uniform (Nx.Rng.key 6) Nx.float64 [| 25_000 |]));
                  ]
                  (fun r ->
                    let xs, ys = Mark.points r in
                    let p =
                      Picture.stamp
                        ~fills:
                          (Array.make (Mark.length r)
                             (Color.v ~alpha:0.3 0.1 0.3 0.8))
                        xs ys
                        (Picture.fill Color.black
                           (Path.circle (P2.v 0. 0.) 1.5))
                    in
                    pictures := p :: !pictures;
                    p);
              ]
          in
          let d = drawn ~density ~size f in
          match (images d, !pictures) with
          | [ (b, px) ], [ p ] ->
              let bounds = Option.get (Picture.bounds p) in
              let snap g v = g (v *. density) /. density in
              let x0 = snap Float.floor (Box2.minx bounds)
              and y0 = snap Float.floor (Box2.miny bounds) in
              let window =
                Box2.v x0 y0
                  (snap Float.ceil (Box2.maxx bounds) -. x0)
                  (snap Float.ceil (Box2.maxy bounds) -. y0)
              in
              equal (Testable.make ~pp:Box2.pp ~equal:Box2.equal) window b;
              let painted =
                Raster.render ~density
                  (Renderable.v (Box2.w window) (Box2.h window)
                     (Picture.transform (Affine.translate (-.x0) (-.y0)) p))
              in
              equal (array int) (Nx.to_array painted) (Nx.to_array px)
          | is, ps ->
              failf "%d images, %d pictures" (List.length is) (List.length ps));
      test "cells paints each cell with its row's fill" (fun () ->
          let z =
            Nx.create Nx.float64 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |]
          in
          let f = heatmap z in
          let r = resolve f in
          match cells (drawn f) with
          | [ (_, px) ] ->
              let at i j = viridis_of r (Float.of_int ((i * 3) + j)) in
              equal (array int)
                (Nx.to_array (cells_image 2 3 at))
                (Nx.to_array px)
          | l -> failf "%d images" (List.length l));
      test "a cell no row covers paints nothing" (fun () ->
          let f =
            layer
              [
                rect
                  ~x:(strings [| "a"; "b"; "a" |])
                  ~y:(strings [| "p"; "p"; "q" |])
                  ();
              ]
          in
          match cells (drawn f) with
          | [ (_, px) ] -> equal int 0 (Nx.item [ 1; 1; 3 ] px)
          | l -> failf "%d images" (List.length l));
      test "rows that share a cell draw as rectangles" (fun () ->
          let f =
            layer
              [
                rect ~x:(strings [| "a"; "a" |]) ~y:(strings [| "p"; "p" |]) ();
              ]
          in
          equal int 0 (List.length (cells (drawn f))));
      test "a heatmap on a padded band scale draws rectangles" (fun () ->
          let z =
            Nx.create Nx.float64 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |]
          in
          let f =
            layer
              [
                rect
                  ~x:(dim ~scale:(Scale.band ~padding:0.1 ()) 1)
                  ~y:(dim 0) ~fill:(num z) ();
              ]
          in
          let d = drawn f and r = resolve f in
          let fills =
            List.concat_map
              (fun (_, p) ->
                List.rev
                  (fold
                     (fun acc -> function
                       | Picture.Fill { color; _ } -> color :: acc | _ -> acc)
                     [] p))
              (tags (path [ Index 0 ]) d)
          in
          equal int 0 (List.length (cells d));
          equal (list color)
            (List.init 6 (fun k -> viridis_of r (Float.of_int k)))
            fills);
      test "a gathered heatmap draws what the whole image draws" (fun () ->
          let h = 300 and w = 400 in
          let z =
            Nx.init Nx.float64 [| h; w |] (fun i ->
                Float.of_int ((i.(0) * 7) + (i.(1) * 13 mod 101)))
          in
          let f = heatmap z in
          let d = drawn ~size:(Size.panels 40. 30.) f in
          let r = resolve f in
          match cells d with
          | [ (window, px) ] ->
              less int ~than:(h * w) (Nx.numel px / 4);
              let l = layout (Size.panels 40. 30.) r in
              let box = (List.hd (Layout.panels l)).box in
              let whole =
                cells_image h w (fun i j -> viridis_of r (Nx.item [ i; j ] z))
              in
              same_raster (Layout.size l) (Picture.image box whole)
                (Picture.image window px)
          | l -> failf "%d images" (List.length l));
      test "a gathered image draws what the whole image draws" (fun () ->
          let px =
            Nx.init Nx.uint8 [| 256; 256; 3 |] (fun i ->
                ((i.(0) * 3) + (i.(1) * 5) + (i.(2) * 70)) mod 256)
          in
          let f = image px in
          let size = Size.panels 20. 20. in
          let d = drawn ~size f in
          let l = layout size (resolve f) in
          let box = (List.hd (Layout.panels l)).box in
          match images d with
          | [ (window, gathered) ] ->
              less int ~than:(256 * 256) (Nx.numel gathered / 4);
              same_raster (Layout.size l) (Picture.image box px)
                (Picture.image window gathered)
          | l -> failf "%d images" (List.length l));
      test "image clamps floats and leaves a pixel with a NaN transparent"
        (fun () ->
          let px =
            Nx.create Nx.float32 [| 1; 2; 3 |] [| nan; 0.; 0.; 2.; 0.5; -1. |]
          in
          match images (drawn (image px)) with
          | [ (_, got) ] ->
              equal (array int)
                [| 0; 0; 0; 0; 255; 128; 0; 255 |]
                (Nx.to_array got)
          | l -> failf "%d images" (List.length l));
    ]

(* Goldens *)

let golden name = Filename.concat "golden" name

(* [same_text name expected actual] states that two documents are equal, naming
   the first byte where they differ. *)
let same_text name expected actual =
  if not (String.equal expected actual) then
    let n = Int.min (String.length expected) (String.length actual) in
    let rec first i =
      if i < n && expected.[i] = actual.[i] then first (i + 1) else i
    in
    failf "%s differs from its golden at byte %d" name (first 0)

let goldens =
  let open Hugin_next_test_figures in
  cases
    ~name:(fun (n, _, _, _) -> n)
    "the benchmark figures draw their goldens" Figures.goldens
    (fun ((name, _, _, _) as g) ->
      let r = Drawing.renderable (Figures.drawing g) in
      same_text name (read_file (golden (name ^ ".svg"))) (Svg.render r);
      within_one ~msg:name
        (Nx_io.load_image (golden (name ^ ".png")))
        (Raster.render ~density:Figures.density r))

let () = exit (run "Draw" [ rows; domain; drawings; output; reducers; goldens ])
