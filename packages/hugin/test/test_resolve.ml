(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin
open Windtrap

(* Data *)

let f64 a = Nx.create Nx.float64 [| Array.length a |] a
let i32 a = Nx.create Nx.int32 [| Array.length a |] (Array.map Int32.of_int a)
let mask a = Nx.create Nx.bool [| Array.length a |] a
let path segs = Nx.Ptree.Path.v segs
let field s = Nx.Ptree.Path.Field s
let index i = Nx.Ptree.Path.Index i

(* Scales keep their hull when they do not round it. *)
let exact = Scale.linear ~nice:false ()
let exact_log = Scale.log ~nice:false ()

(* Reading resolved figures *)

let ends s =
  let (Scale.Floats (a, b)) = Scale.domain s in
  (a, b)

let floats = pair float_exact float_exact
let quant ?at r name = Resolved.scale ?at r (Scale.linear ~name ())
let categ ?at r name = Resolved.scale ?at r (Scale.band ~name ())
let hull ?at r name = ends (quant ?at r name)

let labels s =
  match Scale.domain s with
  | Scale.Categories (Scale.Labels l) -> Array.to_list l
  | Scale.Categories (Scale.Indices _) -> fail "indexed categories"

let indices s =
  match Scale.domain s with
  | Scale.Categories (Scale.Indices ix) -> Array.to_list ix
  | Scale.Categories (Scale.Labels _) -> fail "labelled categories"

let resolved = Testable.make ~pp:Resolved.pp ~equal:Resolved.equal
let printed r = Format.asprintf "%a" Resolved.pp r

let warning =
  pair (Testable.make ~pp:Nx.Ptree.Path.pp ~equal:Nx.Ptree.Path.equal) string

let warnings r = Resolved.warnings r

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec at i = i + m <= n && (String.sub s i m = sub || at (i + 1)) in
  at 0

(* [fails_naming subs f] states that [f] raises [Invalid_argument] with a
   message holding each of [subs]. *)
let fails_naming subs f =
  raises_match
    (function
      | Invalid_argument m -> List.for_all (contains m) subs | _ -> false)
    f

let invalid f = raises_match (Exn.invalid_arg ?substring:None) f
let draw_nothing (_ : Mark.rows) = Picture.empty

let dot1 ?fill ?fx x y =
  dot ?fill ?fx ~x:(num ~scale:exact (f64 x)) ~y:(num (f64 y)) ()

(* Generators *)

(* Floats with the values that make a datum missing. *)
let gen_value =
  Gen.frequency
    [
      (6, Gen.float_range (-10.) 10.);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_float
          [ Float.nan; Float.infinity; Float.neg_infinity; 0.; -0. ] );
    ]

let gen_values n = Gen.array ~size:(Gen.constant n) gen_value

(* Errors *)

let errors =
  let x = f64 [| 1.; 2. |] in
  let lr = Scale.linear ~name:"lr" () in
  let cases_ =
    [
      ( "share of a scale nothing reads",
        (fun () ->
          resolve
            (share
               [ ("size", `Shared) ]
               (layer [ dot ~x:(num x) ~y:(num x) () ]))),
        [ "root"; "size" ] );
      ( "independent x across a layer",
        (fun () ->
          resolve
            (share
               [ ("x", `Independent) ]
               (layer [ dot1 [| 1. |] [| 1. |]; dot1 [| 2. |] [| 2. |] ]))),
        [ "root"; "x" ] );
      ( "independent fx across a layer",
        (fun () ->
          resolve
            (share
               [ ("fx", `Independent) ]
               (layer
                  [
                    dot ~fx:(strings [| "a" |]) ~x:(const 0.5) ~y:(const 0.5) ();
                  ]))),
        [ "root"; "fx" ] );
      ( "two x scales in one panel",
        (fun () ->
          resolve
            (layer
               [
                 dot ~x:(num ~scale:lr x) ~y:(num x) ();
                 dot ~x:(num x) ~y:(num x) ();
               ])),
        [ "0 and 1 read two x scales in the panel root" ] );
      ( "two guides implied for one scale",
        (fun () ->
          let m g =
            Mark.v ~name:"m"
              [ Mark.bind ~guide:g Role.fill (num x) ]
              draw_nothing
          in
          resolve (layer [ m true; m false ])),
        [ "0 and 1 imply different guides for the scale \"color\"" ] );
      ( "two names on a child",
        (fun () ->
          resolve (layer [ name "a" (name "b" (dot1 [| 1. |] [| 1. |])) ])),
        [ "root" ] );
      ( "two names under a wrapper",
        (fun () ->
          resolve
            (layer
               [
                 name "a"
                   (title (Text.v "t") (name "b" (dot1 [| 1. |] [| 1. |])));
               ])),
        [ "root" ] );
      ( "two names on the root",
        (fun () -> resolve (name "a" (name "b" (layer [])))),
        [ "root" ] );
      ( "two siblings of one name",
        (fun () -> resolve (layer [ name "a" (layer []); name "a" (layer []) ])),
        [ "root"; "a" ] );
      ( "two coordinate systems over a panel",
        (fun () ->
          resolve
            (coord (Coord.cartesian ())
               (layer
                  [
                    coord
                      (Coord.cartesian ~aspect:1. ())
                      (dot1 [| 1. |] [| 1. |]);
                  ]))),
        [ "the panel root lies under two coordinate systems, at root and 0" ] );
      ( "two implied coordinate systems",
        (fun () ->
          let m a =
            Mark.v ~name:"m"
              ~coord:(Coord.cartesian ~aspect:a ())
              [] draw_nothing
          in
          resolve (layer [ m 1.; m 2. ])),
        [ "0 and 1 imply two coordinate systems in the panel root" ] );
      ( "rows of different lengths",
        (fun () -> resolve (grid [ [ layer []; layer [] ]; [ layer [] ] ])),
        [ "root" ] );
      ( "a span past the last row",
        (fun () -> resolve (grid [ [ span ~rows:2 (layer []) ] ])),
        [ "the span 0 reaches past the last row" ] );
      ( "a span over another cell",
        (fun () ->
          resolve
            (grid
               [
                 [ layer []; layer []; span ~rows:2 (layer []) ];
                 [ layer []; span ~cols:2 (layer []) ];
               ])),
        [ "the span 4 covers another cell" ] );
      ( "widths of the wrong length",
        (fun () -> resolve (grid ~widths:[ 1. ] [ [ layer []; layer [] ] ])),
        [ "root"; "widths" ] );
      ( "heights of the wrong length",
        (fun () -> resolve (grid ~heights:[ 1.; 1. ] [ [ layer [] ] ])),
        [ "root"; "heights" ] );
      ( "a span outside a grid",
        (fun () -> resolve (layer [ span (layer []) ])),
        [ "0 spans cells outside a grid" ] );
      ( "grids that do not broadcast",
        (fun () ->
          resolve
            (layer
               [
                 grid [ [ layer [] ]; [ layer [] ] ];
                 grid [ [ layer [] ]; [ layer [] ]; [ layer [] ] ];
               ])),
        [ "root" ] );
      ( "a mark layered with a grid of no cells",
        (fun () -> resolve (layer [ dot1 [| 1. |] [| 1. |]; grid [] ])),
        [ "root" ] );
      ( "a grid with spans and another grid",
        (fun () ->
          resolve
            (layer
               [
                 grid [ [ span ~cols:2 (layer []) ] ];
                 grid [ [ layer []; layer [] ] ];
               ])),
        [ "root" ] );
      ( "two titles among layered children",
        (fun () ->
          resolve
            (layer
               [ title (Text.v "a") (layer []); title (Text.v "b") (layer []) ])),
        [ "root" ] );
      ( "two explicit values of one property",
        (fun () ->
          resolve
            (layer
               [
                 dot
                   ~x:(num ~scale:(Scale.linear ~zero:true ()) x)
                   ~y:(num x) ();
                 dot
                   ~x:(num ~scale:(Scale.linear ~zero:false ()) x)
                   ~y:(num x) ();
               ])),
        [ "0 and 1 give the scale \"x\" two explicit values"; "zero" ] );
      ( "two implied values of one property",
        (fun () ->
          let m c =
            Mark.v ~name:"m"
              [ Mark.bind ~imply:(Scale.linear ~clamp:c ()) Role.fill (num x) ]
              draw_nothing
          in
          resolve (layer [ m true; m false ])),
        [ "0 and 1 give the scale \"color\" two implied values"; "clamp" ] );
      ( "two kinds on a user's scale",
        (fun () ->
          let s = Scale.band ~name:"lr" () in
          resolve
            (layer
               [
                 dot ~x:(num ~scale:lr x) ~y:(num x) ();
                 dot ~x:(const 0.5) ~y:(num x)
                   ~fill:(cat ~scale:s (i32 [| 0 |]))
                   ();
               ])),
        [
          "the scale \"lr\" is read as quantitative by 0 and as categorical by \
           1";
        ] );
      ( "two kinds on x",
        (fun () ->
          resolve
            (layer [ dot1 [| 1. |] [| 1. |]; rect ~x:(dim 0) ~y:(num x) () ])),
        [
          "the scale \"x\" is read as quantitative by 0 and as categorical by 1";
        ] );
      ( "labelled and indexed categories on one scale",
        (fun () ->
          resolve
            (layer
               [
                 dot ~x:(num x) ~y:(num x)
                   ~fill:(cat ~labels:[| "a"; "b" |] (i32 [| 0; 1 |]))
                   ();
                 dot ~x:(num x) ~y:(num x) ~fill:(dim 0) ();
               ])),
        [
          "0 and 1 read labelled and indexed categories on the scale \"color\"";
        ] );
      ( "a labelled domain read by indexed categories",
        (fun () ->
          resolve
            (dot ~x:(num x) ~y:(num x)
               ~fill:
                 (dim ~scale:(Scale.band ~domain:(Scale.Labels [| "a" |]) ()) 0)
               ())),
        [ "root"; "color" ] );
      ( "two texts for one index",
        (fun () ->
          resolve
            (layer
               [
                 dot ~x:(num x) ~y:(num x)
                   ~fill:(dim ~labels:[| "a"; "b" |] 0)
                   ();
                 dot ~x:(num x) ~y:(num x)
                   ~fill:(dim ~labels:[| "a"; "c" |] 0)
                   ();
               ])),
        [ "0 and 1 give the category 1 of the scale \"color\" two texts" ] );
      ( "an axis of no position scale",
        (fun () -> resolve (layer [ dot1 [| 1. |] [| 1. |]; axis "color" ])),
        [ "the axis 1 names \"color\"" ] );
      ( "two different axes for one scale",
        (fun () ->
          resolve
            (layer [ dot1 [| 1. |] [| 1. |]; axis "x"; axis ~grid:true "x" ])),
        [ "root"; "x" ] );
      ( "an x axis on the left",
        (fun () ->
          resolve
            (layer [ dot1 [| 1. |] [| 1. |]; axis ~side:`Left "x" |> name "a" ])),
        [ "the axis a of \"x\""; "left" ] );
      ( "a y axis of a user's name at the top",
        (fun () ->
          resolve
            (layer
               [
                 dot ~x:(num x) ~y:(num ~scale:lr x) ();
                 axis ~side:`Top "lr" |> name "a";
               ])),
        [ "the axis a of \"lr\""; "top" ] );
      ( "a legend of no scale with a legend",
        (fun () -> resolve (layer [ dot1 [| 1. |] [| 1. |]; legend "x" ])),
        [ "the legend 1 names \"x\"" ] );
      ( "two different legends for one scale",
        (fun () ->
          resolve
            (layer
               [
                 dot ~fill:(num x) ~x:(num x) ~y:(num x) ();
                 legend "color";
                 legend ~show:false "color";
               ])),
        [ "1 and 2 are two different legends for \"color\"" ] );
      ( "two inside legends of one scale in different corners",
        (fun () ->
          resolve
            (layer
               [
                 dot ~fill:(num x) ~x:(num x) ~y:(num x) ();
                 legend ~side:(`Inside `Top_left) "color";
                 legend ~side:(`Inside `Top_right) "color";
               ])),
        [ "1 and 2 are two different legends for \"color\"" ] );
      ( "a wrapping fx with an fy",
        (fun () ->
          resolve
            (dot ~x:(num x) ~y:(num x)
               ~fx:(strings ~scale:(Scale.band ~wrap:2 ()) [| "a"; "b" |])
               ~fy:(strings [| "c"; "d" |])
               ())),
        [ "root" ] );
      ( "a facet scale independent per panel",
        (fun () ->
          resolve
            (share
               [ ("fx", `Independent) ]
               (dot ~x:(num x) ~y:(num x) ~fx:(strings [| "a"; "b" |]) ()))),
        [ "root"; "fx" ] );
      ( "a user-named facet scale independent per panel",
        (fun () ->
          let g = Scale.band ~name:"g" () in
          resolve
            (share
               [ ("g", `Independent) ]
               (dot ~x:(num x) ~y:(num x)
                  ~fx:(strings ~scale:g [| "a"; "b" |])
                  ()))),
        [ "root"; "g" ] );
      ( "two keys of one name and two sorts",
        (fun () ->
          let n = View.number "k" ~init:1. and c = View.choice "k" ~init:"a" in
          resolve
            (layer [ bind n (fun _ -> layer []); bind c (fun _ -> layer []) ])),
        [ "k" ] );
    ]
  in
  cases
    ~name:(fun (n, _, _) -> n)
    "resolve raises, naming the nodes at fault" cases_
    (fun (_, f, subs) -> fails_naming subs f)

(* A traced tensor outside any trace: reading it raises. *)
type (_, _) Nx.Repr.node += Outside

let traced () =
  Nx.Repr.Traced.v ~context:Nx.Placement.host Nx.Placement.host Nx.float64
    [| 2 |] Outside

let reads =
  group "reads"
    [
      test "a tensor that cannot be read raises naming its mark" (fun () ->
          let t = dot ~x:(num (traced ())) ~y:(const 0.5) () |> name "t" in
          fails_naming [ "resolve: t: "; "traced" ] (fun () ->
              resolve (layer [ dot1 [| 1. |] [| 1. |]; t ])));
    ]

let resolved_scale =
  let r = resolve (layer [ dot1 [| 1. |] [| 2. |] |> name "a" ]) in
  group "Resolved.scale"
    [
      test "refuses an unnamed scale" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"unnamed") (fun () ->
              Resolved.scale r (Scale.linear ())));
      test "refuses an id no node has" (fun () ->
          fails_naming [ "b" ] (fun () -> quant ~at:(path [ field "b" ]) r "x"));
      test "refuses a scale its scope lacks" (fun () ->
          fails_naming [ "lr" ] (fun () -> quant r "lr"));
      test "refuses a scale of another kind" (fun () ->
          fails_naming [ "x" ] (fun () -> categ r "x"));
      test "refuses a time scale, which no channel reads" (fun () ->
          fails_naming [ "x" ] (fun () ->
              Resolved.scale r (Scale.time ~name:"x" ())));
      test "finds a scale through a named node" (fun () ->
          equal floats (1., 1.) (hull ~at:(path [ field "a" ]) r "x"));
    ]

(* Composition laws *)

(* [pp_equal f g] states that [f] and [g] resolve to the same printed form. *)
let pp_equal f g = equal text (printed (resolve f)) (printed (resolve g))

let gen_mark =
  Gen.map
    (fun (xs, k) ->
      let x = f64 xs in
      match k with
      | 0 -> dot ~x:(num x) ~y:(num x) ()
      | 1 -> line ~y:(num x) ~stroke:(num x) ()
      | _ -> rect ~y:(num x) ())
    (Gen.pair (gen_values 3) (Gen.int_range 0 2))

let composition =
  group "composition"
    [
      prop "layer lifts a title" (Gen.pair gen_mark gen_mark) (fun (a, b) ->
          let w = title (Text.v "w") in
          pp_equal (layer [ w a; b ]) (w (layer [ a; b ]));
          pp_equal (layer [ a; w b ]) (w (layer [ a; b ])));
      prop "layer lifts titles one at a time" (Gen.pair gen_mark gen_mark)
        (fun (a, b) ->
          let t s = title (Text.v s) in
          pp_equal
            (layer [ t "w" (t "v" a); t "w" b ])
            (t "w" (t "v" (layer [ a; b ])));
          pp_equal
            (layer [ t "w" a; t "w" (t "v" b) ])
            (t "w" (t "v" (layer [ a; b ])));
          invalid (fun () -> resolve (layer [ t "w" (t "v" a); t "v" b ])));
      prop "layer lifts a coordinate system" (Gen.pair gen_mark gen_mark)
        (fun (a, b) ->
          let w = coord (Coord.cartesian ~aspect:2. ()) in
          pp_equal (layer [ w a; b ]) (w (layer [ a; b ])));
      prop "layered grids broadcast as their shapes do"
        Gen.(
          pair
            (pair (int_range 1 3) (int_range 1 3))
            (pair (int_range 1 3) (int_range 1 3)))
        (fun ((r, c), (r', c')) ->
          let g r c =
            grid (List.init r (fun _ -> List.init c (fun _ -> layer [])))
          in
          let dim d d' =
            if d = d' || d' = 1 then Some d else if d = 1 then Some d' else None
          in
          match (dim r r', dim c c') with
          | Some rows, Some cols ->
              cover "a grid repeats" (r <> r' || c <> c');
              contains
                (printed (resolve (layer [ g r c; g r' c' ])))
                (Format.asprintf "%d × %d" rows cols)
              |> equal bool true
          | _ ->
              cover "shapes that do not broadcast" true;
              invalid (fun () -> resolve (layer [ g r c; g r' c' ])));
      test "grids of no cells layer into a grid of none" (fun () ->
          equal bool true
            (contains (printed (resolve (layer [ grid []; grid [] ]))) "0 × 0"));
      test "a layered mark joins each cell of a grid" (fun () ->
          let r =
            resolve
              (layer
                 [
                   grid [ [ dot1 [| 0. |] [| 0. |]; dot1 [| 5. |] [| 0. |] ] ];
                   dot1 [| 2. |] [| 0. |];
                 ])
          in
          equal floats (0., 2.) (hull ~at:(path [ index 0; index 0 ]) r "x");
          equal floats (2., 5.) (hull ~at:(path [ index 0; index 1 ]) r "x"));
      test "share restated is share" (fun () ->
          let a = dot1 [| 1. |] [| 1. |] and b = dot1 [| 2. |] [| 2. |] in
          let p = [ ("x", `Shared) ] in
          let f = grid [ [ a; b ] ] in
          pp_equal (share p (share p f)) (share p f));
      test "share of x on a grid of one cell is the grid" (fun () ->
          let f = grid [ [ dot1 [| 1. |] [| 1. |] ] ] in
          pp_equal (share [ ("x", `Shared) ] f) f);
      test "a nested layer draws what the flat layer draws" (fun () ->
          let a = dot1 [| 1. |] [| 1. |] and b = dot1 [| 2. |] [| 2. |] in
          let r = resolve (layer [ layer [ a; b ]; a ])
          and r' = resolve (layer [ a; b; a ]) in
          equal floats (hull r "x") (hull r' "x");
          equal bool false (Resolved.equal r r'));
    ]

(* Ids *)

let ids =
  group "ids"
    [
      test "ids are stable under insertion before a name" (fun () ->
          let a = dot1 [| 1. |] [| 1. |] ~fill:(num (f64 [| 1. |])) in
          let b = dot1 [| 2. |] [| 2. |] ~fill:(num (f64 [| 7.; 9. |])) in
          let f before =
            share
              [ ("color", `Independent) ]
              (layer (before @ [ a; name "n" b ]))
          in
          let at = path [ field "n" ] in
          let s = quant ~at (resolve (f [])) "color" in
          let s' = quant ~at (resolve (f [ layer [] ])) "color" in
          equal floats (7., 9.) (ends s);
          equal bool true (Scale.equal s s'));
      test
        "a facet panel does not collide with a sibling named like its category"
        (fun () ->
          let f =
            share
              [ ("color", `Independent) ]
              (layer
                 [
                   dot ~x:(const 0.5) ~y:(const 0.5)
                     ~fill:(num (f64 [| 1.; 2. |]))
                     ~fx:(strings [| "a"; "b" |])
                     ();
                   name "a"
                     (dot ~x:(const 0.5) ~y:(const 0.5)
                        ~fill:(num (f64 [| 5. |]))
                        ());
                 ])
          in
          let r = resolve f in
          equal floats (5., 5.) (hull ~at:(path [ field "a" ]) r "color");
          equal bool true (contains (printed r) "panel.a"));
      test "layered grids broadcast into cells of their own" (fun () ->
          (* A 1 × 3 grid over a 2 × 1 grid: cell [k] holds column [k mod 3] of
             the first and row [k / 3] of the second. *)
          let row =
            grid [ List.init 3 (fun c -> dot1 [| Float.of_int c |] [| 0. |]) ]
          and col =
            grid
              (List.init 2 (fun r ->
                   [ dot1 [| Float.of_int (10 + r) |] [| 0. |] ]))
          in
          let r = resolve (layer [ row; col ]) in
          for k = 0 to 5 do
            equal ~msg:(string_of_int k) floats
              (Float.of_int (k mod 3), Float.of_int (10 + (k / 3)))
              (hull ~at:(path [ field "cell"; index k ]) r "x")
          done);
      test "an independent position on a broadcast grid stays per cell"
        (fun () ->
          let row =
            grid [ List.init 3 (fun c -> dot1 [| Float.of_int c |] [| 0. |]) ]
            |> share [ ("x", `Independent) ]
          and col =
            grid
              (List.init 2 (fun r ->
                   [ dot1 [| Float.of_int (10 + r) |] [| 0. |] ]))
          in
          let r = resolve (layer [ row; col ]) in
          equal floats (1., 11.)
            (hull ~at:(path [ field "cell"; index 4 ]) r "x"));
      test "facet panels are named by their categories" (fun () ->
          let f =
            share
              [ ("y", `Independent) ]
              (dot ~x:(const 0.5)
                 ~y:(num ~scale:exact (f64 [| 1.; 5.; 3. |]))
                 ~fx:(strings [| "a"; "b"; "a" |])
                 ())
          in
          let r = resolve f in
          let panel c = path [ field "panel"; field c ] in
          equal floats (1., 3.) (hull ~at:(panel "a") r "y");
          equal floats (5., 5.) (hull ~at:(panel "b") r "y"));
    ]

(* Scopes *)

let scopes =
  let cell i = path [ index i ] in
  let filled x = dot1 x [| 0.; 1. |] ~fill:(num ~scale:exact (f64 x)) in
  let two = grid [ [ filled [| 0.; 1. |]; filled [| 5.; 6. |] ] ] in
  group "scopes"
    [
      test "grid cells keep their positions" (fun () ->
          let r = resolve two in
          equal floats (0., 1.) (hull ~at:(cell 0) r "x");
          equal floats (5., 6.) (hull ~at:(cell 1) r "x"));
      test "grid cells share their colour" (fun () ->
          equal floats (0., 6.) (hull ~at:(cell 0) (resolve two) "color"));
      test "share x gives a grid one x" (fun () ->
          let r = resolve (share [ ("x", `Shared) ] two) in
          equal floats (0., 6.) (hull ~at:(cell 0) r "x");
          equal floats (0., 6.) (hull ~at:(cell 1) r "x"));
      test "share color independent gives each cell its colour" (fun () ->
          let r = resolve (share [ ("color", `Independent) ] two) in
          equal floats (0., 1.) (hull ~at:(cell 0) r "color");
          equal floats (5., 6.) (hull ~at:(cell 1) r "color"));
      test "facets share their scales" (fun () ->
          let r =
            resolve
              (dot ~x:(const 0.5)
                 ~y:(num ~scale:exact (f64 [| 1.; 5.; 3. |]))
                 ~fx:(strings [| "a"; "b"; "a" |])
                 ())
          in
          equal floats (1., 5.) (hull r "y"));
      test "a layer's children share every scale" (fun () ->
          let r =
            resolve (layer [ dot1 [| 0. |] [| 0. |]; dot1 [| 4. |] [| 0. |] ])
          in
          equal floats (0., 4.) (hull r "x"));
      test "a node a layer repeats over cells names no position" (fun () ->
          let line = name "r" (filled [| 2.; 3. |]) in
          let r = resolve (layer [ two; line ]) in
          let at = path [ field "r" ] in
          fails_naming [ "no scope of the scale \"x\" holds r" ] (fun () ->
              quant ~at r "x");
          equal floats (0., 6.) (hull ~at r "color"));
      test "a grid names no position" (fun () ->
          let r = resolve two in
          fails_naming [ "no scope of the scale \"x\" holds root" ] (fun () ->
              quant r "x");
          equal floats (0., 6.) (hull r "color"));
      test "a grid names the position its cells share" (fun () ->
          equal floats (0., 6.)
            (hull (resolve (share [ ("x", `Shared) ] two)) "x"));
      test "a grid names no colour its cells keep apart" (fun () ->
          let r = resolve (share [ ("color", `Independent) ] two) in
          fails_naming [ "no scope of the scale \"color\" holds root" ]
            (fun () -> quant r "color"));
      test "a grid names no colour its cells keep apart beside another scope"
        (fun () ->
          let apart = share [ ("color", `Independent) ] two |> name "g" in
          let r = resolve (grid [ [ apart; filled [| 7.; 8. |] ] ]) in
          equal floats (7., 8.) (hull ~at:(path [ index 1 ]) r "color");
          fails_naming [ "no scope of the scale \"color\" holds g" ] (fun () ->
              quant ~at:(path [ field "g" ]) r "color"));
      test "a layer names no colour its children keep apart" (fun () ->
          let apart =
            share
              [ ("color", `Independent) ]
              (layer [ filled [| 1.; 2. |]; filled [| 5.; 6. |] ])
            |> name "l"
          in
          let r = resolve (grid [ [ apart; filled [| 7.; 8. |] ] ]) in
          fails_naming [ "no scope of the scale \"color\" holds l" ] (fun () ->
              quant ~at:(path [ field "l" ]) r "color"));
      test "an independent x on a grid of grids is one x per outer cell"
        (fun () ->
          let inner =
            grid [ [ dot1 [| 0. |] [| 0. |]; dot1 [| 5. |] [| 0. |] ] ]
          in
          let at = path [ index 0; index 0 ] in
          equal floats (0., 0.) (hull ~at (resolve (grid [ [ inner ] ])) "x");
          equal floats (0., 5.)
            (hull ~at
               (resolve (share [ ("x", `Independent) ] (grid [ [ inner ] ])))
               "x"));
      test "a mark whose facet constant names no panel still fits x" (fun () ->
          let r =
            resolve
              (layer
                 [
                   dot1 [| 0.; 1. |] [| 0.; 0. |] ~fx:(strings [| "a"; "a" |]);
                   dot1 [| 100. |] [| 0. |] ~fx:(const "z");
                 ])
          in
          equal floats (0., 100.) (hull r "x"));
      test "a cell of a layered grid names its broadcast cell" (fun () ->
          let r = resolve (layer [ two; name "r" (filled [| 2.; 3. |]) ]) in
          equal floats (0., 3.) (hull ~at:(path [ index 0; index 0 ]) r "x");
          equal floats (2., 6.)
            (hull ~at:(path [ field "cell"; index 1 ]) r "x"));
      test "a mark independent per panel holds its own scale" (fun () ->
          let own = share [ ("color", `Independent) ] (filled [| 1.; 2. |]) in
          let r = resolve (layer [ own; filled [| 5.; 6. |] ]) in
          equal floats (1., 2.) (hull ~at:(path [ index 0 ]) r "color");
          equal floats (5., 6.) (hull ~at:(path [ index 1 ]) r "color");
          equal floats (5., 6.) (hull r "color"));
      test "a zoom at a mark independent per panel zooms its own scale"
        (fun () ->
          let own = share [ ("color", `Independent) ] (filled [| 1.; 2. |]) in
          let f = layer [ own; filled [| 5.; 6. |] ] in
          let color = Scale.linear ~name:"color" () in
          let zoomed at =
            resolve
              ~view:(View.set (View.zoom ?at color) (Some (8., 9.)) View.empty)
              f
          in
          let r = zoomed (Some (path [ index 0 ])) in
          equal floats (8., 9.) (hull ~at:(path [ index 0 ]) r "color");
          equal floats (5., 6.) (hull r "color");
          let r = zoomed None in
          equal floats (1., 2.) (hull ~at:(path [ index 0 ]) r "color");
          equal floats (8., 9.) (hull r "color"));
      test "a facetted mark independent per panel names no one scale" (fun () ->
          let f =
            share
              [ ("color", `Independent) ]
              (dot ~x:(const 0.5) ~y:(const 0.5)
                 ~fill:(num ~scale:exact (f64 [| 1.; 2. |]))
                 ~fx:(strings [| "a"; "b" |])
                 ())
            |> name "m"
          in
          let r = resolve (layer [ f ]) in
          fails_naming [ "no scope of the scale \"color\" holds m" ] (fun () ->
              quant ~at:(path [ field "m" ]) r "color");
          equal floats (2., 2.)
            (hull ~at:(path [ field "panel"; field "b" ]) r "color"));
      prop "facets share one scale" (gen_values 6) (fun ys ->
          let y = Nx.reshape [| 2; 3 |] (f64 ys) in
          let plain = dot ~x:(const 0.5) ~y:(num ~scale:exact y) () in
          let facetted =
            dot ~x:(const 0.5) ~y:(num ~scale:exact y) ~fx:(dim 0) ()
          in
          equal bool true
            (Scale.equal
               (quant (resolve plain) "y")
               (quant (resolve facetted) "y")));
    ]

(* Merging *)

let merging =
  let x = f64 [| 2.; 3. |] in
  group "merging"
    [
      test "a length implies zero" (fun () ->
          equal floats (0., 3.)
            (hull (resolve (rect ~x:(dim 0) ~y:(num ~scale:exact x) ())) "y"));
      test "an explicit property beats an implied one" (fun () ->
          let y = num ~scale:(Scale.linear ~zero:false ~nice:false ()) x in
          equal floats (2., 3.) (hull (resolve (rect ~x:(dim 0) ~y ())) "y"));
      test "a stem implies zero on its length" (fun () ->
          equal floats (0., 3.)
            (hull (resolve (rule ~x:(dim 0) ~y:(num ~scale:exact x) ())) "y"));
      test "an area alone implies zero" (fun () ->
          equal floats (0., 3.)
            (hull (resolve (area ~y:(num ~scale:exact x) ())) "y"));
      test "an area with a baseline fits the hull of its curves" (fun () ->
          let lo = f64 [| 1.; 1.5 |] in
          equal floats (1., 3.)
            (hull (resolve (area ~y:(num ~scale:exact x) ~y2:(num lo) ())) "y"));
      test "an error bar fits the hull of its ends" (fun () ->
          let lo = f64 [| 1.; 1.5 |] in
          equal floats (1., 3.)
            (hull
               (resolve
                  (rule ~x:(dim 0) ~y:(num ~scale:exact lo) ~y2:(num x) ()))
               "y"));
      test "the size role implies zero" (fun () ->
          equal floats (0., 3.)
            (hull
               (resolve
                  (dot ~x:(const 0.) ~y:(const 0.) ~size:(num ~scale:exact x) ()))
               "size"));
      test "a band scale read by y is reversed" (fun () ->
          let r =
            resolve
              (dot ~x:(const 0.5)
                 ~y:(cat ~labels:[| "a"; "b" |] (i32 [| 0; 1 |]))
                 ())
          in
          let y = categ r "y" in
          greater float_exact ~than:0.5 (Scale.normalize y "a"));
      test "a band scale read by x is not reversed" (fun () ->
          let r =
            resolve
              (dot
                 ~x:(cat ~labels:[| "a"; "b" |] (i32 [| 0; 1 |]))
                 ~y:(const 0.5) ())
          in
          less float_exact ~than:0.5 (Scale.normalize (categ r "x") "a"));
      test "a contour's x and y domains are its grid's hull" (fun () ->
          let r =
            resolve
              (contour
                 ~x:(num (f64 [| -1.5; 0.; 1.5 |]))
                 ~y:(num (Nx.create Nx.float64 [| 2; 1 |] [| -0.3; 0.7 |]))
                 ~fill:(num (Nx.zeros Nx.float64 [| 2; 3 |]))
                 ())
          in
          equal floats (-1.5, 1.5) (hull r "x");
          equal floats (-0.3, 0.7) (hull r "y"));
      test "an image fixes its pixel domains" (fun () ->
          let r = resolve (image (Nx.zeros Nx.float32 [| 2; 3 |])) in
          equal floats (0., 3.) (hull r "x");
          equal floats (0., 2.) (hull r "y");
          equal float_exact 1. (Scale.normalize (quant r "y") 0.));
      test "explicit specifications merge" (fun () ->
          let r =
            resolve
              (layer
                 [
                   dot
                     ~x:(num ~scale:(Scale.linear ~nice:false ()) x)
                     ~y:(num x) ();
                   dot
                     ~x:(num ~scale:(Scale.linear ~clamp:true ()) x)
                     ~y:(num x) ();
                 ])
          in
          let expected =
            Scale.fit
              (Some (Scale.Floats (2., 3.)))
              (Scale.linear ~nice:false ~clamp:true ())
          in
          equal bool true (Scale.equal expected (quant r "x")));
      test "implied specifications give neither name nor transform" (fun () ->
          let m imply =
            Mark.v ~name:"m" [ Mark.bind ~imply Role.fill (num x) ] draw_nothing
          in
          let r =
            resolve
              (layer
                 [
                   m (Scale.log ~name:"a" ~nice:false ());
                   m (Scale.linear ~name:"b" ~clamp:true ());
                 ])
          in
          let expected =
            Scale.fit
              (Some (Scale.Floats (2., 3.)))
              (Scale.linear ~nice:false ~clamp:true ())
          in
          equal bool true (Scale.equal expected (quant r "color")));
      test "a mark implies properties of a band scale" (fun () ->
          let m =
            Mark.v ~name:"m"
              [
                Mark.bind
                  ~imply:(Scale.band ~padding:0.5 ())
                  Role.x
                  (strings [| "a"; "b" |]);
              ]
              draw_nothing
          in
          equal (float 1e-12) (0.5 /. 2.5)
            (Scale.bandwidth (categ (resolve m) "x")));
      test "a bar implies padding on its band position" (fun () ->
          let r = resolve (rect ~x:(strings [| "a"; "b" |]) ~y:(num x) ()) in
          equal (float 1e-12) (0.8 /. 2.2) (Scale.bandwidth (categ r "x")));
      test "a heatmap implies no padding" (fun () ->
          let z = Nx.zeros Nx.float64 [| 2; 3 |] in
          let r = resolve (rect ~x:(dim 1) ~y:(dim 0) ~fill:(num z) ()) in
          equal (float 1e-12) (1. /. 3.) (Scale.bandwidth (categ r "x"));
          equal (float 1e-12) 0.5 (Scale.bandwidth (categ r "y")));
      test "an explicit padding beats a bar's" (fun () ->
          let bars =
            strings ~scale:(Scale.band ~padding:0. ()) [| "a"; "b" |]
          in
          let r = resolve (rect ~x:bars ~y:(num x) ()) in
          equal (float 1e-12) 0.5 (Scale.bandwidth (categ r "x")));
      test "colour scales of two kinds are two scales" (fun () ->
          let r =
            resolve
              (layer
                 [
                   dot ~x:(num x) ~y:(num x) ~fill:(num ~scale:exact x) ();
                   dot ~x:(num x) ~y:(num x)
                     ~fill:(cat ~labels:[| "a" |] (i32 [| 0; 0 |]))
                     ();
                 ])
          in
          equal floats (2., 3.) (hull r "color");
          equal (list string) [ "a" ] (labels (categ r "color")));
    ]

(* Categories *)

let categories =
  let at0 = f64 [| 0. |] in
  let filled fill = dot ~x:(num at0) ~y:(num at0) ~fill () in
  let index_w = list (pair int string) in
  group "categories"
    [
      test "labels are kept whole" (fun () ->
          let r =
            resolve (filled (cat ~labels:[| "a"; "b"; "c" |] (i32 [| 0 |])))
          in
          equal (list string) [ "a"; "b"; "c" ] (labels (categ r "color")));
      test "labels unite in written order, before broadcasting" (fun () ->
          let one l =
            dot ~x:(num at0) ~y:(num at0) ~fill:(strings [| l |]) ()
          in
          let r = resolve (layer [ grid [ [ one "a"; one "b" ] ]; one "c" ]) in
          equal (list string) [ "a"; "b"; "c" ] (labels (categ r "color")));
      test "strings contribute in order of first appearance" (fun () ->
          let r =
            resolve
              (dot ~x:(const 0.5) ~y:(const 0.5)
                 ~fill:(strings [| "q"; "p"; "q" |])
                 ())
          in
          equal (list string) [ "q"; "p" ] (labels (categ r "color")));
      test "indexed categories unite in increasing order" (fun () ->
          let r =
            resolve
              (layer
                 [
                   dot
                     ~x:(num (f64 [| 0.; 0. |]))
                     ~y:(num at0)
                     ~fill:(cat (i32 [| 5; 1 |]))
                     ();
                   dot
                     ~x:(num (f64 [| 0.; 0.; 0. |]))
                     ~y:(num at0) ~fill:(dim 0) ();
                 ])
          in
          equal index_w
            [ (0, "0"); (1, "1"); (2, "2"); (5, "5") ]
            (indices (categ r "color")));
      test "codes of dropped rows are no categories" (fun () ->
          let r =
            resolve
              (dot
                 ~x:(num (f64 [| 0.; Float.nan; 0. |]))
                 ~y:(num at0)
                 ~fill:
                   (cat
                      ~valid:(mask [| true; true; false |])
                      (i32 [| 3; 4; 7 |]))
                 ())
          in
          equal index_w [ (3, "3") ] (indices (categ r "color")));
      test "a dim without labels beside a labelled one takes its texts"
        (fun () ->
          let r =
            resolve
              (layer
                 [
                   dot
                     ~x:(num (f64 [| 0.; 0. |]))
                     ~y:(num at0)
                     ~fill:(dim ~labels:[| "a"; "b" |] 0)
                     ();
                   dot
                     ~x:(num (f64 [| 0.; 0.; 0. |]))
                     ~y:(num at0) ~fill:(dim 0) ();
                 ])
          in
          equal index_w
            [ (0, "a"); (1, "b"); (2, "2") ]
            (indices (categ r "color")));
      test "a dim contributes every index of its axis" (fun () ->
          let r =
            resolve
              (dot
                 ~x:(num (f64 [| Float.nan; Float.nan |]))
                 ~y:(num at0) ~fill:(dim 0) ())
          in
          equal index_w [ (0, "0"); (1, "1") ] (indices (categ r "color")));
      test "floats fit their finite values" (fun () ->
          let r =
            resolve
              (dot
                 ~x:
                   (Hugin.floats ~scale:exact
                      [|
                        2.; Float.nan; Float.infinity; -1.; Float.neg_infinity;
                      |])
                 ~y:(num at0) ())
          in
          equal floats (-1., 2.) (hull r "x"));
      test "strings of dropped rows are categories" (fun () ->
          let r =
            resolve
              (dot
                 ~x:(num (f64 [| 1.; Float.nan |]))
                 ~y:(num at0)
                 ~fill:(strings [| "kept"; "dropped" |])
                 ())
          in
          equal (list string) [ "kept"; "dropped" ] (labels (categ r "color")));
      test "an explicit domain drops the categories outside it" (fun () ->
          let s = Scale.band ~domain:(Scale.Labels [| "b" |]) () in
          let r =
            resolve
              (dot
                 ~x:(num ~scale:exact (f64 [| 1.; 2. |]))
                 ~y:(num at0)
                 ~symbol:(strings ~scale:s [| "a"; "b" |])
                 ())
          in
          equal floats (2., 2.) (hull r "x"));
      test "an explicit domain drops the labelled codes outside it" (fun () ->
          let s = Scale.band ~domain:(Scale.Labels [| "b" |]) () in
          let r =
            resolve
              (dot
                 ~x:(num ~scale:exact (f64 [| 1.; 2.; 3. |]))
                 ~y:(num at0)
                 ~symbol:
                   (cat ~scale:s ~labels:[| "a"; "b" |] (i32 [| 0; 1; 0 |]))
                 ())
          in
          equal floats (2., 2.) (hull r "x"));
      test "an explicit domain drops the indexed codes outside it" (fun () ->
          let s =
            Scale.band ~domain:(Scale.Indices [| (2, "two"); (5, "five") |]) ()
          in
          let r =
            resolve
              (dot
                 ~x:(num ~scale:exact (f64 [| 1.; 2.; 3.; 4. |]))
                 ~y:(num at0)
                 ~symbol:(cat ~scale:s (i32 [| 2; 3; 5; 9 |]))
                 ())
          in
          equal floats (1., 3.) (hull r "x"));
      test "an explicit domain drops the indices of a dim outside it" (fun () ->
          let s = Scale.band ~domain:(Scale.Indices [| (1, "one") |]) () in
          let r =
            resolve
              (dot
                 ~x:(num ~scale:exact (f64 [| 1.; 2.; 3. |]))
                 ~y:(num at0) ~symbol:(dim ~scale:s 0) ())
          in
          equal floats (2., 2.) (hull r "x"));
    ]

(* Facet panels *)

(* Each panel of a facetted mark whose y is independent per panel fits its y to
   the rows the panel draws. *)
let facet_panels =
  let ys = [| 1.; 5.; 3.; 9. |] in
  let per fx =
    share
      [ ("y", `Independent) ]
      (dot ~x:(const 0.5) ~y:(num ~scale:exact (f64 ys)) ~fx ())
  in
  let per2 ?fy fx =
    let y = num ~scale:exact (Nx.create Nx.float64 [| 2; 2 |] ys) in
    share [ ("y", `Independent) ] (dot ~x:(const 0.5) ~y ?fy ~fx ())
  in
  let panel cs = path (field "panel" :: List.map field cs) in
  let hulls f cats =
    let r = resolve f in
    List.map (fun cs -> hull ~at:(panel cs) r "y") cats
  in
  let rows = list floats in
  group "facet panels"
    [
      test "labelled codes select their panels" (fun () ->
          equal rows
            [ (1., 3.); (5., 9.) ]
            (hulls
               (per (cat ~labels:[| "p"; "q" |] (i32 [| 0; 1; 0; 1 |])))
               [ [ "p" ]; [ "q" ] ]));
      test "indexed codes select their panels" (fun () ->
          equal rows
            [ (5., 9.); (1., 3.) ]
            (hulls (per (cat (i32 [| 7; 2; 7; 2 |]))) [ [ "2" ]; [ "7" ] ]));
      test "strings select their panels" (fun () ->
          equal rows
            [ (1., 3.); (5., 9.) ]
            (hulls
               (per (strings [| "p"; "q"; "p"; "q" |]))
               [ [ "p" ]; [ "q" ] ]));
      test "a dim selects the panel of its index" (fun () ->
          equal rows
            [ (1., 5.); (3., 9.) ]
            (hulls (per2 (dim 0)) [ [ "0" ]; [ "1" ] ]);
          equal rows
            [ (1., 3.); (5., 9.) ]
            (hulls (per2 (dim (-1))) [ [ "0" ]; [ "1" ] ]));
      test "a panel's id names its fy category, then its fx one" (fun () ->
          equal rows
            [ (1., 1.); (5., 5.); (3., 3.); (9., 9.) ]
            (hulls
               (per2 ~fy:(dim 0) (dim 1))
               [ [ "0"; "0" ]; [ "0"; "1" ]; [ "1"; "0" ]; [ "1"; "1" ] ]));
    ]

(* Nulls *)

(* A host reference for Law 15: a row is in a hull iff none of its values other
   than its colour is missing. *)
let law15 =
  let gen =
    Gen.(
      let* n = int_range 0 6 in
      let+ x = gen_values n
      and+ y = gen_values n
      and+ size = gen_values n
      and+ valid = array ~size:(constant n) bool
      and+ codes = array ~size:(constant n) (int_range (-1) 2) in
      (x, y, size, valid, codes))
  in
  prop "a dropped row is in no hull" gen (fun (x, y, size, valid, codes) ->
      let n = Array.length x in
      let finite v = Float.is_finite v in
      let kept i =
        finite x.(i)
        && valid.(i)
        && finite y.(i)
        && y.(i) > 0.
        && finite size.(i)
      in
      let keeps = List.filter kept (List.init n Fun.id) in
      cover "a dropped row" (List.length keeps < n);
      cover "a kept row with a missing colour"
        (List.exists (fun i -> codes.(i) < 0 || codes.(i) > 1) keeps);
      let expect v =
        match keeps with
        | [] -> (0., 1.)
        | i :: _ ->
            List.fold_left
              (fun (a, b) i -> (Float.min a v.(i), Float.max b v.(i)))
              (v.(i), v.(i))
              keeps
      in
      let r =
        resolve
          (dot
             ~x:(num ~scale:exact ~valid:(mask valid) (f64 x))
             ~y:(num ~scale:exact_log (f64 y))
             ~size:(num (f64 size))
             ~fill:(cat ~labels:[| "a"; "b" |] (i32 codes))
             ())
      in
      equal floats (expect x) (hull r "x");
      match keeps with [] -> () | _ -> equal floats (expect y) (hull r "y"))

(* Export *)

let export =
  group "export"
    [
      prop "an exported scale normalises alike in another figure"
        Gen.(pair (gen_values 4) (gen_values 3))
        (fun (xs, xs') ->
          let lr = Scale.log ~name:"lr" () in
          let figure s x = dot ~x:(num ~scale:s (f64 x)) ~y:(const 0.5) () in
          let s = Resolved.scale (resolve (figure lr xs)) lr in
          let s' = Resolved.scale (resolve (figure s xs')) lr in
          equal bool true (Scale.equal s s');
          List.iter
            (fun v ->
              equal float_exact (Scale.normalize s v) (Scale.normalize s' v))
            [ 0.5; 1.; 7. ]);
      test "an exported band scale keeps the reverse y implies" (fun () ->
          let r = resolve (dot ~x:(const 0.5) ~y:(strings [| "a"; "b" |]) ()) in
          let s = categ r "y" in
          let r' =
            resolve (dot ~y:(const 0.5) ~x:(strings ~scale:s [| "a"; "b" |]) ())
          in
          let s' = categ r' "x" in
          greater float_exact ~than:0.5 (Scale.normalize s' "a"));
    ]

(* Views and warnings *)

let views =
  let lr = Scale.linear ~name:"lr" () in
  let fig = dot ~x:(num ~scale:lr (f64 [| 1.; 2. |])) ~y:(const 0.5) () in
  group "views"
    [
      test "a zoom sets the domain" (fun () ->
          let view = View.set (View.zoom lr) (Some (5., 6.)) View.empty in
          let r = resolve ~view fig in
          equal floats (5., 6.) (hull r "lr");
          equal (list warning) [] (warnings r));
      test "a zoom of no scale is ignored with a warning" (fun () ->
          let z = View.zoom (Scale.linear ~name:"gone" ()) in
          let r = resolve ~view:(View.set z (Some (5., 6.)) View.empty) fig in
          equal int 1 (List.length (warnings r)));
      test "a zoom of a time scale, which no channel reads, is ignored"
        (fun () ->
          let z = View.zoom (Scale.time ~name:"lr" ()) in
          let view =
            View.set z (Some Hugin_kit.Time.(epoch, epoch)) View.empty
          in
          let r = resolve ~view fig in
          equal floats (1., 2.) (hull r "lr");
          equal (list warning)
            [
              ( Nx.Ptree.Path.root,
                "the zoom of the scale \"lr\" applies to no scale" );
            ]
            (warnings r));
      test "zooms of one scale at two nodes are ignored with warnings"
        (fun () ->
          let f = layer [ fig |> name "a" ] in
          let view =
            View.empty
            |> View.set (View.zoom lr) (Some (5., 6.))
            |> View.set (View.zoom ~at:(path [ field "a" ]) lr) (Some (7., 8.))
          in
          let r = resolve ~view f in
          equal floats (1., 2.) (hull r "lr");
          equal int 2 (List.length (warnings r)));
      test "a zoom at a node in several position scopes is ignored" (fun () ->
          let f =
            layer
              [
                grid [ [ dot1 [| 0. |] [| 0. |]; dot1 [| 5. |] [| 0. |] ] ];
                name "r" (dot1 [| 2. |] [| 0. |]);
              ]
          in
          let at = path [ field "r" ] in
          let x = Scale.linear ~name:"x" () in
          let r =
            resolve
              ~view:(View.set (View.zoom ~at x) (Some (7., 8.)) View.empty)
              f
          in
          equal floats (0., 2.)
            (hull ~at:(path [ field "cell"; index 0 ]) r "x");
          equal floats (2., 5.)
            (hull ~at:(path [ field "cell"; index 1 ]) r "x");
          equal
            (list
               (Testable.make ~pp:Nx.Ptree.Path.pp ~equal:Nx.Ptree.Path.equal))
            [ at ]
            (List.map fst (warnings r)));
      test "a zoom at a broadcast cell zooms that cell" (fun () ->
          let f =
            layer
              [
                grid [ [ dot1 [| 0. |] [| 0. |]; dot1 [| 5. |] [| 0. |] ] ];
                dot1 [| 2. |] [| 0. |];
              ]
          in
          let x = Scale.linear ~name:"x" () in
          let at = path [ field "cell"; index 1 ] in
          let r =
            resolve
              ~view:(View.set (View.zoom ~at x) (Some (7., 8.)) View.empty)
              f
          in
          equal floats (0., 2.)
            (hull ~at:(path [ field "cell"; index 0 ]) r "x");
          equal floats (7., 8.) (hull ~at r "x");
          equal (list warning) [] (warnings r));
      test "a zoom the scale cannot take is ignored with a warning" (fun () ->
          let lg = Scale.log ~name:"lg" () in
          let f =
            dot ~x:(num ~scale:lg (f64 [| 1.; 10. |])) ~y:(const 0.5) ()
          in
          let r =
            resolve
              ~view:(View.set (View.zoom lg) (Some (-1., 1.)) View.empty)
              f
          in
          equal floats (1., 10.) (hull r "lg");
          equal int 1 (List.length (warnings r)));
      test "a view value no key reads is warned about" (fun () ->
          let r =
            resolve
              ~view:(View.set (View.number "k" ~init:0.) 1. View.empty)
              fig
          in
          equal (list warning)
            [
              ( Nx.Ptree.Path.root,
                "the view sets \"k\", which no key of its sort reads" );
            ]
            (warnings r));
      test "a bind reads the view once per resolve" (fun () ->
          let calls = ref 0 in
          let k = View.number "k" ~init:1. in
          let f =
            bind k (fun v ->
                incr calls;
                dot1 [| v |] [| v |])
          in
          let r = resolve ~view:(View.set k 4. View.empty) f in
          equal int 1 !calls;
          equal floats (4., 4.) (hull r "x"));
      test "warnings follow the figure, then the view's" (fun () ->
          let lg = Scale.log ~name:"lg" () in
          let f =
            layer
              [
                dot ~x:(const 0.5) ~y:(const 0.5) ~fx:(const "nowhere") ();
                dot
                  ~x:(num ~scale:exact_log (f64 [| -1.; 2. |]))
                  ~y:(const 0.5) ();
                name "z"
                  (dot ~x:(const 0.5) ~y:(const 0.5)
                     ~fill:(num ~scale:lg (f64 [| 1.; 10. |]))
                     ());
                dot ~x:(const 0.5) ~y:(const 0.5) ~fx:(const "nowhere") ();
              ]
          in
          let view =
            View.empty
            |> View.set (View.number "k" ~init:0.) 1.
            |> View.set (View.zoom ~at:(path [ field "z" ]) lg) (Some (-1., 1.))
          in
          equal (list warning)
            [
              ( path [ index 0 ],
                "the facet constant \"nowhere\" of fx names no panel" );
              (path [ index 1 ], "x: 1 finite value is missing for its scale");
              ( path [ field "z" ],
                "the zoom of the scale \"lg\" sets a domain it cannot take" );
              ( path [ index 3 ],
                "the facet constant \"nowhere\" of fx names no panel" );
              ( Nx.Ptree.Path.root,
                "the view sets \"k\", which no key of its sort reads" );
            ]
            (warnings (resolve ~view f)));
      test "a mark repeated over the cells of a grid is warned about once"
        (fun () ->
          let m =
            dot ~x:(num ~scale:exact_log (f64 [| -1.; 2. |])) ~y:(const 0.5) ()
          in
          let r = resolve (layer [ grid [ [ layer []; layer [] ] ]; m ]) in
          equal (list warning)
            [ (path [ index 1 ], "x: 1 finite value is missing for its scale") ]
            (warnings r));
      test "a uint64 code beyond int is missing, with a warning" (fun () ->
          let codes = Nx.create Nx.uint64 [| 2 |] [| -1L; 1L |] in
          let r =
            resolve (dot ~x:(const 0.5) ~y:(const 0.5) ~fill:(cat codes) ())
          in
          equal
            (list (pair int string))
            [ (1, "1") ]
            (indices (categ r "color"));
          equal (list warning)
            [ (Nx.Ptree.Path.root, "fill: 1 code is beyond the range of int") ]
            (warnings r));
      test "a masked code outside the labels is not warned about" (fun () ->
          let r =
            resolve
              (dot ~x:(const 0.5) ~y:(const 0.5)
                 ~fill:
                   (cat ~labels:[| "a" |]
                      ~valid:(mask [| true; false; true |])
                      (i32 [| 3; 5; 0 |]))
                 ())
          in
          equal (list warning)
            [ (Nx.Ptree.Path.root, "fill: 1 code is outside its 1 labels") ]
            (warnings r));
      test "data problems are warned about under the mark's id" (fun () ->
          let r =
            resolve
              (layer
                 [
                   dot
                     ~x:(num ~scale:exact_log (f64 [| 1.; 0.; -1. |]))
                     ~y:(const 0.5) ();
                   name "c"
                     (dot ~x:(const 0.5) ~y:(const 0.5)
                        ~fill:(cat ~labels:[| "a" |] (i32 [| 0; 3 |]))
                        ());
                   dot ~x:(const 0.5) ~y:(const 0.5) ~fx:(const "nowhere") ();
                 ])
          in
          equal (list warning)
            [
              (path [ index 0 ], "x: 2 finite values are missing for its scale");
              (path [ field "c" ], "fill: 1 code is outside its 1 labels");
              ( path [ index 2 ],
                "the facet constant \"nowhere\" of fx names no panel" );
            ]
            (warnings r));
    ]

(* Equality *)

let equality =
  let f x = dot1 x [| 0. |] ~fill:(num (f64 [| 1. |])) in
  let a = f [| 1. |] in
  group "Resolved.equal"
    [
      test "a figure resolved twice is equal to itself" (fun () ->
          equal resolved (resolve a) (resolve a));
      test "figures fitting equal scales differ" (fun () ->
          let r = resolve a and r' = resolve (f [| 1. |]) in
          equal bool true
            (Scale.equal (quant r "x") (quant r' "x")
            && Scale.equal (quant r "color") (quant r' "color"));
          not_equal resolved r r');
      test "views that change nothing differ" (fun () ->
          let k = View.number "k" ~init:0. in
          let g = bind k (fun _ -> a) in
          let view = View.set k 1. View.empty in
          equal (list warning) [] (warnings (resolve ~view g));
          not_equal resolved (resolve g) (resolve ~view g));
    ]

(* Reuse *)

let reuse =
  let same r f = equal resolved (resolve f) r in
  group "reuse"
    [
      prop "resolving again is equal" (Gen.pair gen_mark gen_mark)
        (fun (a, b) ->
          let f = layer [ a; grid [ [ b; a ] ] ] in
          same (resolve ~prev:(resolve f) f) f);
      prop "a reused resolution equals a fresh one after a tensor changes"
        Gen.(pair (gen_values 3) (gen_values 3))
        (fun (xs, xs') ->
          let b = dot1 [| 1. |] [| 1. |] in
          let f x = layer [ dot ~x:(num (f64 x)) ~y:(num (f64 x)) (); b ] in
          let prev = resolve (f xs) and f' = f xs' in
          same (resolve ~prev f') f');
      test "a mark beside a new log y loses its rows that are not positive"
        (fun () ->
          let a =
            dot ~x:(num (f64 [| 1.; 2. |])) ~y:(num (f64 [| -1.; 2. |])) ()
          in
          let b y = dot ~x:(const 0.5) ~y () in
          let y = f64 [| 3. |] in
          let prev = resolve (layer [ a; b (num y) ]) in
          let f = layer [ a; b (num ~scale:(Scale.log ()) y) ] in
          let r = resolve ~prev f in
          same r f;
          equal floats (1., 10.) (hull r "y"));
      test "a reused resolution equals a fresh one after the view changes"
        (fun () ->
          let k = View.number "k" ~init:1. in
          let f = bind k (fun v -> dot1 [| v |] [| 2. |]) in
          let prev = resolve f in
          let view = View.set k 3. View.empty in
          equal resolved (resolve ~view f) (resolve ~prev ~view f));
      test "a bind building new tensors resolves equal" (fun () ->
          let k = View.number "k" ~init:1. in
          let f = bind k (fun v -> dot1 [| v |] [| v |]) in
          same (resolve f) f);
      test "a reused resolution equals a fresh one after an id changes"
        (fun () ->
          let a =
            dot ~x:(num ~scale:exact_log (f64 [| 0.; 1. |])) ~y:(const 0.5) ()
          in
          let prev = resolve (layer [ a ]) in
          let f = layer [ layer []; a ] in
          same (resolve ~prev f) f);
    ]

(* Baselines *)

let baselines =
  test "Resolved.pp" (fun () ->
      let y = f64 [| 4.; Float.nan; -6. |] in
      let f =
        grid
          [
            [
              layer
                [
                  dot
                    ~x:(num (f64 [| 1.; 2.; 3. |]))
                    ~y:(num y)
                    ~fill:(cat ~labels:[| "a"; "b" |] (i32 [| 0; 1; 5 |]))
                    ();
                  line ~y:(num ~scale:(Scale.log ()) y) () |> name "fit";
                ]
              |> title (Text.v "t");
              layer
                [
                  rect ~x:(dim 0) ~y:(num y)
                    ~fx:(strings [| "p"; "q"; "p" |])
                    ();
                  rule ~y:(const 0.5) ();
                  axis ~grid:true "x";
                ];
            ];
          ]
      in
      expect (printed (resolve f))
      @@ __POS_OF__
           {|
        figure
          grid root, 1 × 2
            cell (0, 0)
              panel 0
                title (text "t") left
                dot 0.0
                line 0.fit
            cell (0, 1)
              panel 1
                rect 1.0
                rule 1.1
                axis "x" grid
                facets 1.panel.p, 1.panel.q
        scales
          "x" quantitative, read by 0.0:x, 0.fit:x
            (linear (domain 0 1))
          "y" quantitative, read by 0.0:y, 0.fit:y
            (log 10 (domain 4 4))
          "color" categorical, read by 0.0:fill
            (band (domain (labels "a" "b")))
          "x" categorical, read by 1.0:x
            (band (domain (indices (0 "0") (1 "1") (2 "2"))) (padding 0.2))
          "y" quantitative, read by 1.0:y
            (linear (domain -6 4))
          "fx" categorical, read by 1.0:fx
            (band (domain (labels "p" "q")))
        warnings
          0.0: y: 1 finite value is missing for its scale
          0.0: fill: 1 code is outside its 2 labels
          0.fit: y: 1 finite value is missing for its scale
        |})

let () =
  exit
    (run "hugin resolve"
       [
         errors;
         reads;
         resolved_scale;
         composition;
         ids;
         scopes;
         merging;
         categories;
         facet_panels;
         law15;
         export;
         views;
         equality;
         reuse;
         baselines;
       ])
