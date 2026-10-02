(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_next
open Windtrap

(* Data *)

let f64 a = Nx.create Nx.float64 [| Array.length a |] a
let ramp = f64 [| 1.; 2.; 3.; 4. |]

let plain ?fill ?y x =
  dot ?fill ~x:(num x) ~y:(num (Option.value y ~default:x)) ()

(* A missing character: no face of the default theme has the private use
   U+10FFFD. *)
let unmapped = "\u{10FFFD}"

(* Reading layouts *)

let em = Theme.size Theme.default
let lay ?theme size f = layout ?theme size (resolve f)
let boxes l = List.map (fun (p : Layout.panel) -> p.box) (Layout.panels l)
let printed l = Format.asprintf "%a" Layout.pp l
let layout_t = Testable.make ~pp:Layout.pp ~equal:Layout.equal

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

let box_of line =
  let i = String.rindex line '[' in
  Scanf.sscanf
    (String.sub line i (String.length line - i))
    "[(%f, %f) (%f, %f)]"
    (fun x0 y0 x1 y1 -> (x0, y0, x1, y1))

(* [texts l] is the box of every text [l] places, read from its printed form,
   where each text is [(text …) [(x0, y0) (x1, y1)]] on one line. *)
let texts l =
  String.split_on_char '\n' (printed l)
  |> List.filter (fun line -> contains line "(text")
  |> List.map box_of

(* [text_box l s] is the box of the first text of [l] whose printed line holds
   [s]. *)
let text_box l s =
  match
    List.find_opt
      (fun line -> contains line s)
      (String.split_on_char '\n' (printed l))
  with
  | None -> failf "no text %s" s
  | Some line -> box_of line

(* The extents of a text set as layout sets tick labels and titles. *)
let measure ~size s =
  Text.Layout.box
    (Text.Layout.v ~fonts:(Theme.fonts Theme.default) ~size (Text.v s))

let label_w s = Box2.w (measure ~size:(0.9 *. em) s)
let title_h s = Box2.h (measure ~size:em s)

(* The height of a figure title, set bold at 1.2 em. *)
let head_h s =
  Box2.h
    (Text.Layout.box
       (Text.Layout.v
          ~fonts:(Theme.fonts Theme.default)
          ~size:(1.2 *. em)
          (Text.bold (Text.v s))))

let tick = 0.35 *. em
let pad = 0.25 *. em
let close = float 1e-9
let ratio b = Box2.h b /. Box2.w b

(* [on_page l] states that every data area of [l] lies on its page. *)
let on_page l =
  let w, h = Layout.size l and tol = 1e-9 in
  List.iter
    (fun b ->
      at_least float_exact ~msg:"left" ~than:(-.tol) (Box2.minx b);
      at_least float_exact ~msg:"top" ~than:(-.tol) (Box2.miny b);
      at_most float_exact ~msg:"right" ~than:(w +. tol) (Box2.maxx b);
      at_most float_exact ~msg:"bottom" ~than:(h +. tol) (Box2.maxy b))
    (boxes l)

(* [swatches l] is the box of every legend swatch and colour bar of [l], read
   from its printed form. *)
let swatches l =
  let first line =
    let i = String.index line '[' in
    box_of (String.sub line 0 (String.index_from line i ']' + 1))
  in
  String.split_on_char '\n' (printed l)
  |> List.filter_map (fun line ->
      match String.split_on_char ' ' (String.trim line) with
      | ("bar" | "entry") :: _ -> Some (first line)
      | _ -> None)

(* [apart l] states that the data areas of [l] lie on its page, and that no
   text, swatch or colour bar of [l] overlaps another, a data area or the page's
   edge. *)
let apart l =
  on_page l;
  let w, h = Layout.size l in
  (* Printed boxes have six significant digits. *)
  let tol = 1e-5 *. Float.max 100. (Float.max w h) in
  let ts = texts l @ swatches l in
  let overlaps (a0, b0, a1, b1) (c0, d0, c1, d1) =
    Float.min a1 c1 -. Float.max a0 c0 > tol
    && Float.min b1 d1 -. Float.max b0 d0 > tol
  in
  List.iteri
    (fun i t ->
      let x0, y0, x1, y1 = t in
      if x0 < -.tol || y0 < -.tol || x1 > w +. tol || y1 > h +. tol then
        failf "a box leaves the page: %g %g %g %g" x0 y0 x1 y1;
      List.iteri
        (fun j t' ->
          if j > i && overlaps t t' then failf "boxes %d and %d overlap" i j)
        ts;
      List.iter
        (fun b ->
          let p = (Box2.minx b, Box2.miny b, Box2.maxx b, Box2.maxy b) in
          if overlaps t p then failf "box %d overlaps a data area" i)
        (boxes l))
    ts

(* A panel with an aspect of one and no axes. *)
let square =
  layer
    [
      rect ~x:(strings [| "a" |]) ~y:(strings [| "p" |]) ();
      axis ~show:false "x";
      axis ~show:false "y";
    ]
  |> coord (Coord.cartesian ~aspect:1. ())

(* Random figures: grids of dots over data of varied magnitudes, so that labels
   vary in length, with colour legends, facets, shared axes and titles or
   without. *)

type colour = No_colour | Categories | Quantities

type case = {
  rows : int;
  cols : int;
  exps : (int * int) list; (* Per cell, the magnitudes of x and y. *)
  colour : colour;
  side : side; (* The colour legend's. *)
  facets : bool;
  shared : bool;
  aspect : bool;
  top : bool; (* Whether x axes are on top. *)
  titled : bool;
  align : Text.Layout.halign;
  size : Size.t;
}

(* The case all others vary. *)
let base =
  {
    rows = 1;
    cols = 1;
    exps = [ (0, 0) ];
    colour = No_colour;
    side = `Right;
    facets = false;
    shared = false;
    aspect = false;
    top = false;
    titled = false;
    align = `Center;
    size = Size.panels 50. 50.;
  }

let pp_case ppf c =
  let side : side -> string = function
    | `Left -> "left"
    | `Right -> "right"
    | `Top -> "top"
    | `Bottom -> "bottom"
  in
  let align : Text.Layout.halign -> string = function
    | `Left -> "left"
    | `Center -> "centre"
    | `Right -> "right"
  in
  Format.fprintf ppf "%d × %d, exponents [%a]%s%s%s%s%s%s, %a" c.rows c.cols
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       (fun ppf (a, b) -> Format.fprintf ppf "(%d, %d)" a b))
    c.exps
    (match c.colour with
    | No_colour -> ""
    | Categories -> ", categories on the " ^ side c.side
    | Quantities -> ", quantities on the " ^ side c.side)
    (if c.facets then ", facets" else "")
    (if c.shared then ", shared x" else "")
    (if c.aspect then ", aspect 1" else "")
    (if c.top then ", x on top" else "")
    (if c.titled then ", titled " ^ align c.align else "")
    Size.pp c.size

let gen_size =
  Gen.one_of
    [
      Gen.map
        (fun (w, h) -> Size.figure w h)
        (Gen.pair (Gen.float_range 300. 700.) (Gen.float_range 250. 600.));
      Gen.map
        (fun (w, h) -> Size.panels w h)
        (Gen.pair (Gen.float_range 20. 150.) (Gen.float_range 20. 150.));
    ]

let gen_case =
  Gen.bind
    (Gen.pair (Gen.int_range 1 3) (Gen.int_range 1 3))
    (fun (rows, cols) ->
      let exp = Gen.int_range (-4) 8 in
      let colour = Gen.of_list [ No_colour; Categories; Quantities ] in
      let side = Gen.of_list [ `Right; `Left; `Top; `Bottom ] in
      let align = Gen.of_list [ `Center; `Left; `Right ] in
      Gen.map
        (fun ( (exps, colour, side),
               ((facets, shared, aspect), (top, titled, align)),
               size ) ->
          {
            rows;
            cols;
            exps;
            colour;
            side;
            facets;
            shared;
            aspect;
            top;
            titled;
            align;
            size;
          })
        (Gen.triple
           (Gen.triple
              (Gen.list ~size:(Gen.constant (rows * cols)) (Gen.pair exp exp))
              colour side)
           (Gen.pair
              (Gen.triple Gen.bool Gen.bool Gen.bool)
              (Gen.triple Gen.bool Gen.bool align))
           gen_size))
  |> Gen.with_pp pp_case

let figure_of c =
  let cell (a, b) =
    let x = f64 [| -.(10. ** Float.of_int a); 10. ** Float.of_int b |] in
    let y = f64 [| 0.; 10. ** Float.of_int b |] in
    let title = if c.titled then Some (Text.v "value") else None in
    let fx =
      if c.facets then Some (strings ?title [| "left"; "right" |]) else None
    in
    let mark ?fill () = dot ?fill ?fx ~x:(num x) ~y:(num ?title y) () in
    let mark =
      match c.colour with
      | No_colour -> mark ()
      | Categories -> mark ~fill:(strings [| "alpha"; "b" |]) ()
      | Quantities -> mark ~fill:(num y) ()
    in
    let guides =
      (if c.top then [ axis ~side:`Top "x" ] else [])
      @ if c.colour = No_colour then [] else [ legend ~side:c.side "color" ]
    in
    let cell = layer (mark :: guides) in
    if c.aspect then coord (Coord.cartesian ~aspect:1. ()) cell else cell
  in
  let rec rows = function
    | [] -> []
    | cells ->
        let rec take n l =
          if n = 0 then ([], l)
          else
            match l with
            | [] -> ([], [])
            | x :: l ->
                let a, b = take (n - 1) l in
                (x :: a, b)
        in
        let row, rest = take c.cols cells in
        row :: rows rest
  in
  let g = grid (rows (List.map cell c.exps)) in
  let g = if c.shared then share [ ("x", `Shared) ] g else g in
  if c.titled then title ~align:c.align (Text.v "A figure") g else g

(* [laid c] is the layout of [c], discarding a figure too small for it. *)
let laid c =
  match layout c.size (resolve (figure_of c)) with
  | l -> l
  | exception (Invalid_argument m as e) ->
      if not (contains m "needs") then raise e;
      assume false;
      assert false

(* Sizes *)

let sizes =
  group "sizes"
    [
      test "a figure size is the page size" (fun () ->
          let check f =
            equal (pair close close) (360., 240.)
              (Layout.size (lay (Size.figure 360. 240.) f))
          in
          check (plain ramp);
          check
            (grid [ [ plain ramp; plain ramp ]; [ plain ramp; plain ramp ] ]);
          check (title (Text.v "t") (plain ~fill:(num ramp) ramp)));
      test "panels sizes give each flexible track its weight's data area"
        (fun () ->
          let hidden = [ axis ~show:false "x"; axis ~show:false "y" ] in
          let cell = layer (plain ramp :: hidden) in
          let l =
            lay (Size.panels 60. 40.)
              (grid ~widths:[ 1.; 2. ] ~heights:[ 1.; 0.5 ]
                 [ [ cell; cell ]; [ cell; cell ] ])
          in
          equal
            (list (pair close close))
            [ (60., 40.); (120., 40.); (60., 20.); (120., 20.) ]
            (List.map (fun b -> (Box2.w b, Box2.h b)) (boxes l)));
      test "panels sizes grow the page to hold longer labels" (fun () ->
          let at y = lay (Size.panels 100. 50.) (plain ~y ramp) in
          let short = at ramp and long = at (f64 [| 1.; 2e6; 3.; 4. |]) in
          greater float_exact
            ~than:(fst (Layout.size short))
            (fst (Layout.size long));
          equal close 100. (Box2.w (List.hd (boxes long))));
      test
        "a figure too small for its decorations raises with the size it needs"
        (fun () ->
          let f = title (Text.v "A long title") (plain ramp) in
          fails_naming [ "needs"; "figure 40 × 30 pt" ] (fun () ->
              lay (Size.figure 40. 30.) f));
      test "flexible tracks shrink to nothing before a figure raises" (fun () ->
          let f =
            layer [ plain ramp; axis ~show:false "x"; axis ~show:false "y" ]
            |> title (Text.v "t")
          in
          let least = head_h "t" +. pad in
          let b = List.hd (boxes (lay (Size.figure 100. least) f)) in
          equal (pair close close) (100., 0.) (Box2.w b, Box2.h b);
          fails_naming [ "needs" ] (fun () ->
              lay (Size.figure 100. (least -. 0.5)) f));
    ]

(* Grids *)

let spines =
  (* Titled facet blocks are left to [titled_blocks]. *)
  let cases =
    Gen.map
      (fun c -> if c.facets then { c with titled = false } else c)
      gen_case
    |> Gen.with_pp pp_case
  in
  prop "spines align across rows and columns" ~count:60 cases (fun c ->
      cover "facets" c.facets;
      cover "aspect" c.aspect;
      (* A cell's facet panels are a grid of their own, side by side. A panel
         with an aspect is centred in its cell, and the aspects of cells
         differ. *)
      let f = if c.facets then 2 else 1 in
      let bs = Array.of_list (boxes (laid c)) in
      equal int (c.rows * c.cols * f) (Array.length bs);
      let at r q j = bs.((((r * c.cols) + q) * f) + j) in
      (* The edges a facet grid places and the middles of panels with an aspect
         are sums of lengths, some huge, so they agree up to rounding. *)
      let near = float_rel ~rel:1e-12 ~abs:1e-9 in
      let edge = if c.facets then near else float_exact in
      let hull r q = Box2.union (at r q 0) (at r q (f - 1)) in
      for r = 0 to c.rows - 1 do
        for q = 0 to c.cols - 1 do
          for j = 0 to f - 1 do
            let b = at r q j and row = at r 0 0 in
            if c.aspect then
              equal near ~msg:"middle" (P2.y (Box2.mid row)) (P2.y (Box2.mid b))
            else begin
              equal edge ~msg:"top" (Box2.miny row) (Box2.miny b);
              equal edge ~msg:"bottom" (Box2.maxy row) (Box2.maxy b)
            end
          done;
          if c.aspect then
            equal near ~msg:"centre"
              (P2.x (Box2.mid (hull 0 q)))
              (P2.x (Box2.mid (hull r q)))
          else begin
            equal edge ~msg:"left" (Box2.minx (hull 0 q)) (Box2.minx (hull r q));
            equal edge ~msg:"right"
              (Box2.maxx (hull 0 q))
              (Box2.maxx (hull r q))
          end
        done
      done)

(* [names ws] is a header per element of [ws], the [k]th one [k] followed by
   [ws.(k)] wide letters, so that headers differ in width. *)
let names ws = Array.mapi (fun k w -> string_of_int k ^ String.make w 'W') ws

(* [headed ws] is panels faceted by [names ws], over hidden axes. *)
let headed ws =
  let x = f64 (Array.init (Array.length ws) Float.of_int) in
  layer
    [
      dot ~fx:(strings (names ws)) ~x:(num x) ~y:(num x) ();
      axis ~show:false "x";
      axis ~show:false "y";
    ]

let equal_shares =
  prop "flexible tracks of equal weight above their least lengths are equal"
    (Gen.pair
       (Gen.array ~size:(Gen.int_range 2 5) (Gen.int_range 0 4))
       (Gen.float_range 400. 700.))
    (fun (ws, w) ->
      let l = lay (Size.figure w 100.) (headed ws) in
      let widths = List.map Box2.w (boxes l) in
      let share =
        List.fold_left ( +. ) 0. widths /. Float.of_int (List.length widths)
      in
      let widest =
        Array.fold_left (fun m s -> Float.max m (label_w s)) 0. (names ws)
      in
      assume (widest < share);
      List.iter (equal ~msg:"width" (float 1e-9) share) widths)

(* The first header takes more than half the page, and the others are narrow.
   Hidden axes protrude nowhere, so the panels' lengths sum to that of panels
   whose headers all fit, whatever their headers. *)
let pinned_share =
  prop "a track that needs more than its share keeps it, the others share"
    (Gen.triple (Gen.int_range 2 5)
       (Gen.float_range 0.55 0.75)
       (Gen.float_range 400. 900.))
    (fun (n, part, w) ->
      let long = truncate (part *. w /. label_w "W") in
      let ws = Array.init n (fun k -> if k = 0 then long else k mod 2) in
      let need = Array.map label_w (names ws) in
      let fitting = lay (Size.figure w 100.) (headed (Array.make n 0)) in
      let total = List.fold_left ( +. ) 0. (List.map Box2.w (boxes fitting)) in
      let rest = (total -. need.(0)) /. Float.of_int (n - 1) in
      assume (need.(0) > total /. Float.of_int n);
      assume (Array.for_all (fun m -> m < rest) (Array.sub need 1 (n - 1)));
      match boxes (lay (Size.figure w 100.) (headed ws)) with
      | [] -> fail "no panel"
      | first :: others ->
          equal (float 1e-9) ~msg:"pinned" need.(0) (Box2.w first);
          List.iter
            (fun b -> equal (float 1e-9) ~msg:"shared" rest (Box2.w b))
            others)

(* A title beside a block takes a track inside the block's cell, a gap from the
   block's panels, so blocks whose panels protrude differently put their panels
   at different depths of their cells. *)
let titled_blocks =
  xfail ~reason:"a block's title takes a track of its cell"
    (test "titled facet blocks side by side align their panels" (fun () ->
         let block x =
           layer
             [
               dot
                 ~fx:(strings ~title:(Text.v "run") [| "a"; "b" |])
                 ~x:(num (f64 x))
                 ~y:(num (f64 x))
                 ();
               axis ~side:`Top "x";
             ]
         in
         let l =
           lay (Size.figure 400. 200.)
             (grid [ [ block [| 0.; 1. |]; block [| 0.; 1e6 |] ] ])
         in
         match boxes l with
         | [ a; _; b; _ ] -> equal float_exact (Box2.miny a) (Box2.miny b)
         | _ -> fail "four panels"))

let grids =
  group "grids"
    [
      spines;
      titled_blocks;
      equal_shares;
      pinned_share;
      test "a gap is one em when nothing protrudes into it" (fun () ->
          let cell =
            layer [ plain ramp; axis ~show:false "x"; axis ~show:false "y" ]
          in
          let l = lay (Size.panels 50. 50.) (grid [ [ cell; cell ] ]) in
          match boxes l with
          | [ a; b ] ->
              equal close em (Box2.minx b -. Box2.maxx a);
              equal (pair close close) (110., 50.) (Layout.size l)
          | _ -> fail "two panels");
      test "a gap holds the protrusions that meet it plus one em" (fun () ->
          let cell = layer [ plain ramp; axis ~show:false "x" ] in
          let l = lay (Size.panels 50. 50.) (grid [ [ cell; cell ] ]) in
          match boxes l with
          | [ a; b ] ->
              (* Both panels have the same y axis: its protrusion is the left
                 one of [a], and meets the gap from [b]. *)
              equal close (Box2.minx a +. em) (Box2.minx b -. Box2.maxx a)
          | _ -> fail "two panels");
      test "a protrusion holds the tick, the widest label and the title"
        (fun () ->
          let f t =
            layer
              [
                dot
                  ~x:(num (f64 [| 1.; 2. |]))
                  ~y:(strings ?title:t [| "a"; "bb" |])
                  ();
                axis ~show:false "x";
              ]
          in
          let left t =
            Box2.minx (List.hd (boxes (lay (Size.panels 80. 80.) (f t))))
          in
          equal close (tick +. pad +. label_w "bb") (left None);
          equal close
            (tick +. pad +. label_w "bb" +. pad +. title_h "name")
            (left (Some (Text.v "name"))));
      test "a nested grid's panels align with its neighbours" (fun () ->
          let inner = grid [ [ plain ramp ]; [ plain ramp ] ] in
          let l =
            lay (Size.figure 400. 300.) (grid [ [ inner; plain ramp ] ])
          in
          match boxes l with
          | [ a; b; c ] ->
              equal close (Box2.miny c) (Box2.miny a);
              equal close (Box2.maxy c) (Box2.maxy b)
          | _ -> fail "three panels");
      test "a span covers its tracks and the gaps between them" (fun () ->
          let l =
            lay (Size.figure 400. 300.)
              (grid
                 [ [ span ~cols:2 (plain ramp) ]; [ plain ramp; plain ramp ] ])
          in
          match boxes l with
          | [ a; b; c ] ->
              equal close (Box2.minx b) (Box2.minx a);
              equal close (Box2.maxx c) (Box2.maxx a)
          | _ -> fail "three panels");
    ]

(* Aspects *)

let aspects =
  let cells =
    rect
      ~x:(strings [| "a"; "b"; "c"; "a"; "b"; "c" |])
      ~y:(strings [| "p"; "p"; "p"; "q"; "q"; "q" |])
      ()
    |> coord (Coord.cartesian ~aspect:1. ())
  in
  group "aspects"
    [
      test "an aspect of one makes band cells square" (fun () ->
          equal close (2. /. 3.)
            (ratio (List.hd (boxes (lay (Size.panels 90. 90.) cells))));
          equal close (2. /. 3.)
            (ratio (List.hd (boxes (lay (Size.figure 300. 400.) cells)))));
      test "a log scale's unit is a power of its base" (fun () ->
          let x =
            num ~scale:(Scale.log ~domain:(1., 100.) ()) (f64 [| 1.; 100. |])
          in
          let y =
            num ~scale:(Scale.linear ~domain:(0., 1.) ()) (f64 [| 0.; 1. |])
          in
          let f = dot ~x ~y () |> coord (Coord.cartesian ~aspect:3. ()) in
          equal close 1.5
            (ratio (List.hd (boxes (lay (Size.panels 50. 50.) f)))));
      test "an aspect sizes its row, so its neighbours align" (fun () ->
          let tall =
            rect ~x:(strings [| "a" |]) ~y:(strings [| "p" |]) ()
            |> coord (Coord.cartesian ~aspect:2. ())
          in
          let l = lay (Size.panels 50. 50.) (grid [ [ tall; plain ramp ] ]) in
          match boxes l with
          | [ a; b ] ->
              equal close 100. (Box2.h a);
              equal close (Box2.miny a) (Box2.miny b);
              equal close (Box2.maxy a) (Box2.maxy b)
          | _ -> fail "two panels");
      test "a nested aspect panel under a wider title stays on its page"
        (fun () ->
          let f =
            grid [ [ square ] ]
            |> title ~align:`Left (Text.v "A rather wide title")
          in
          List.iter
            (fun h -> on_page (lay (Size.panels 40. h) f))
            [ 45.; 50.; 60.; 61. ]);
      test "an aspect panel's box holds its axis titles" (fun () ->
          let m = Nx.zeros Nx.float64 [| 2; 50 |] in
          let f =
            rect
              ~x:(dim ~title:(Text.v "token") 1)
              ~y:(dim ~title:(Text.v "layer") 0)
              ~fill:(num m) ()
            |> coord (Coord.cartesian ~aspect:1. ())
          in
          let l = lay (Size.panels 100. 100.) f in
          let b = List.hd (boxes l) in
          let _, y0, _, y1 = text_box l {|(text "layer")|} in
          at_least float_exact ~than:(Box2.miny b -. 1e-3) y0;
          at_most float_exact ~than:(Box2.maxy b +. 1e-3) y1);
      test "an aspect panel's box holds its headers" (fun () ->
          let cat = "a very long facet category" in
          let f =
            rect
              ~x:(strings (Array.init 4 string_of_int))
              ~y:(strings (Array.make 4 "p"))
              ~fy:(strings (Array.make 4 cat))
              ()
            |> coord (Coord.cartesian ~aspect:1. ())
          in
          let l = lay (Size.panels 30. 30.) f in
          let b = List.hd (boxes l) in
          let _, y0, _, y1 = text_box l (Printf.sprintf "(text %S)" cat) in
          at_least float_exact ~than:(Box2.miny b -. 1e-3) y0;
          at_most float_exact ~than:(Box2.maxy b +. 1e-3) y1);
      test "an aspect row takes its height from its aspect alone" (fun () ->
          let l = lay (Size.panels 100. 400.) square in
          equal (pair close close) (100., 100.) (Layout.size l));
      test "an aspect panel is centred in a cell it cannot fill" (fun () ->
          let tall =
            layer [ plain ramp; axis ~show:false "x"; axis ~show:false "y" ]
          in
          let l = lay (Size.figure 300. 100.) (grid [ [ square ]; [ tall ] ]) in
          (match boxes l with
          | [ a; _ ] -> equal close 150. (P2.x (Box2.mid a))
          | _ -> fail "two panels");
          let b = List.hd (boxes (lay (Size.figure 300. 100.) square)) in
          equal (pair close close) (150., 50.)
            (P2.x (Box2.mid b), P2.y (Box2.mid b)));
      test "images have square pixels" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 3; 6 |] in
          let l = lay (Size.figure 300. 300.) (image px) in
          equal close 0.5 (ratio (List.hd (boxes l))));
    ]

(* Guides *)

let no_overlap =
  let examples =
    [
      (* Panels 20 points wide leave the last axis too short for its frozen
         labels, which drop alternate ones. *)
      {
        base with
        rows = 3;
        cols = 2;
        exps = [ (0, 0); (0, 0); (4, 0); (0, 0); (-3, -3); (0, 0) ];
        colour = No_colour;
        facets = true;
        shared = false;
        titled = false;
        size = Size.panels 20. 20.;
      };
      {
        base with
        rows = 1;
        cols = 3;
        exps = [ (0, 0); (0, 3); (0, 0) ];
        colour = No_colour;
        facets = true;
        shared = true;
        titled = false;
        size = Size.panels 134.474 20.;
      };
      {
        base with
        rows = 2;
        cols = 1;
        exps = [ (0, 0); (0, 6) ];
        colour = Quantities;
        facets = false;
        shared = false;
        titled = false;
        size = Size.figure 300. 250.;
      };
      {
        base with
        rows = 1;
        cols = 3;
        exps = [ (0, 0); (0, 0); (-1, -2) ];
        colour = No_colour;
        facets = false;
        shared = false;
        titled = false;
        size = Size.figure 300. 250.;
      };
    ]
  in
  prop "no text overlaps another, a data area or the page's edge" ~count:60
    ~examples gen_case (fun c -> apart (laid c))

(* Figures the property once failed on. *)
let kept_apart =
  let cases_ =
    [
      (* Legend entries taller than the panel with an aspect they stand
         beside. *)
      {
        base with
        colour = Categories;
        side = `Right;
        aspect = true;
        top = true;
        size = Size.panels 20. 20.;
      };
    ]
  in
  cases ~name:(Format.asprintf "%a" pp_case)
    "no text overlaps another, a data area or the page's edge in" cases_
    (fun c -> apart (laid c))

let guides =
  group "guides"
    [
      no_overlap;
      kept_apart;
      test "a scale on x in one cell and a colour in another has both guides"
        (fun () ->
          let rate = Scale.linear ~name:"rate" () in
          let f =
            grid
              [
                [
                  dot ~x:(num ~scale:rate ramp) ~y:(num ramp) ();
                  dot ~x:(num ramp) ~y:(num ramp) ~fill:(num ~scale:rate ramp)
                    ();
                ];
              ]
          in
          let p = printed (lay (Size.panels 80. 60.) f) in
          in_order
            ~subs:[ "axis 0.axis.rate bottom"; "legend legend.rate.num" ]
            p);
      test "a legend's id is its scale's name and kind" (fun () ->
          let f =
            layer
              [
                plain ~fill:(strings [| "a"; "b"; "a"; "b" |]) ramp;
                plain ~fill:(num ramp) ramp;
              ]
          in
          let ids =
            String.split_on_char '\n' (printed (lay (Size.panels 80. 60.) f))
            |> List.filter_map (fun line ->
                match String.split_on_char ' ' (String.trim line) with
                | "legend" :: id :: _ -> Some id
                | _ -> None)
          in
          equal (list string)
            [ "legend.color.cat"; "legend.color.num" ]
            (List.sort String.compare ids));
      test "a shared axis is labelled on the outer panel only" (fun () ->
          let l =
            lay (Size.panels 60. 40.)
              (grid [ [ plain ramp ]; [ plain ramp ] ]
              |> share [ ("x", `Shared) ])
          in
          expect (printed l)
          @@ __POS_OF__
               {|
            layout 83.1826 × 123.224
            panel 0 [(20.2646, 5.44482) (80.2646, 45.4448)] cartesian
              axis 0.axis.x bottom
              axis 0.axis.y left
                label (text "1.0") [(0, 40) (14.2646, 50.8896)]
                label (text "3.5") [(0, 6.66667) (14.2646, 17.5563)]
            panel 1 [(20.2646, 66.3345) (80.2646, 106.334)] cartesian
              axis 1.axis.x bottom
                label (text "1") [(17.3467, 112.334) (23.1826, 123.224)]
                label (text "2") [(37.3467, 112.334) (43.1826, 123.224)]
                label (text "3") [(57.3467, 112.334) (63.1826, 123.224)]
                label (text "4") [(77.3467, 112.334) (83.1826, 123.224)]
              axis 1.axis.y left
                label (text "1.0") [(0, 100.89) (14.2646, 111.779)]
                label (text "3.5") [(0, 67.5563) (14.2646, 78.446)]
            ticks
              "x" quantitative
                (ticks (0 "1") (0.333333 "2") (0.666667 "3") (1 "4")
                 (minor 0.0666667 0.133333 0.2 0.266667 0.4 0.466667 0.533333 0.6
                  0.733333 0.8 0.866667 0.933333))
              "y" quantitative
                (ticks (0 "1.0") (0.833333 "3.5")
                 (minor 0.166667 0.333333 0.5 0.666667 1))
              "y" quantitative
                (ticks (0 "1.0") (0.833333 "3.5")
                 (minor 0.166667 0.333333 0.5 0.666667 1))
            |});
      test "headers name each column and each row once" (fun () ->
          let f =
            dot ~x:(num ramp) ~y:(num ramp)
              ~fx:(strings [| "a"; "b"; "a"; "b" |])
              ~fy:(strings [| "p"; "p"; "q"; "q" |])
              ()
          in
          expect (printed (lay (Size.panels 40. 30.) f))
          @@ __POS_OF__
               {|
            layout 130.072 × 111.169
            panel panel.p.a [(20.2646, 13.3896) (60.2646, 43.3896)] cartesian
              axis panel.p.a.axis.x bottom
              axis panel.p.a.axis.y left
                label (text "1.0") [(0, 37.9448) (14.2646, 48.8345)]
                label (text "3.5") [(0, 12.9448) (14.2646, 23.8345)]
              header panel.p.a.axis.fx (text "a") [(37.7378, 0) (42.7915, 10.8896)]
            panel panel.p.b [(76.6826, 13.3896) (116.683, 43.3896)] cartesian
              axis panel.p.b.axis.x bottom
              axis panel.p.b.axis.y left
              header panel.p.b.axis.fx (text "b") [(93.9272, 0) (99.438, 10.8896)]
              header panel.p.b.axis.fy (text "p") [(119.183, 25.6343) (130.072, 31.145)] turned
            panel panel.q.a [(20.2646, 64.2793) (60.2646, 94.2793)] cartesian
              axis panel.q.a.axis.x bottom
                label (text "1") [(17.3467, 100.279) (23.1826, 111.169)]
                label (text "2") [(30.68, 100.279) (36.516, 111.169)]
                label (text "3") [(44.0133, 100.279) (49.8493, 111.169)]
                label (text "4") [(57.3467, 100.279) (63.1826, 111.169)]
              axis panel.q.a.axis.y left
                label (text "1.0") [(0, 88.8345) (14.2646, 99.7241)]
                label (text "3.5") [(0, 63.8345) (14.2646, 74.7241)]
            panel panel.q.b [(76.6826, 64.2793) (116.683, 94.2793)] cartesian
              axis panel.q.b.axis.x bottom
                label (text "1") [(73.7646, 100.279) (79.6006, 111.169)]
                label (text "2") [(87.098, 100.279) (92.9339, 111.169)]
                label (text "3") [(100.431, 100.279) (106.267, 111.169)]
                label (text "4") [(113.765, 100.279) (119.601, 111.169)]
              axis panel.q.b.axis.y left
              header panel.q.b.axis.fy (text "q") [(119.183, 76.5239) (130.072, 82.0347)] turned
            ticks
              "x" quantitative
                (ticks (0 "1") (0.333333 "2") (0.666667 "3") (1 "4")
                 (minor 0.0666667 0.133333 0.2 0.266667 0.4 0.466667 0.533333 0.6
                  0.733333 0.8 0.866667 0.933333))
              "y" quantitative
                (ticks (0 "1.0") (0.833333 "3.5")
                 (minor 0.166667 0.333333 0.5 0.666667 1))
              "fx" categorical (ticks (0.25 "a") (0.75 "b"))
              "fy" categorical (ticks (0.25 "p") (0.75 "q"))
            |});
      test "legends: entries for categories, a bar for quantities" (fun () ->
          let f =
            layer
              [
                plain
                  ~fill:
                    (strings ~title:(Text.v "kind") [| "a"; "b"; "a"; "b" |])
                  ramp;
                rect ~x:(num ramp) ~fill:(num ~title:(Text.v "load") ramp) ();
              ]
          in
          expect (printed (lay (Size.panels 80. 60.) f))
          @@ __POS_OF__
               {|
            layout 156.531 × 96.9341
            panel root [(11.8359, 20.0444) (91.8359, 80.0444)] cartesian
              axis axis.x bottom
                label (text "0") [(8.91797, 86.0444) (14.7539, 96.9341)]
                label (text "2") [(48.918, 86.0444) (54.7539, 96.9341)]
                label (text "4") [(88.918, 86.0444) (94.7539, 96.9341)]
              axis axis.y left
                label (text "1") [(0, 74.5996) (5.83594, 85.4893)]
                label (text "2") [(0, 54.5996) (5.83594, 65.4893)]
                label (text "3") [(0, 34.5996) (5.83594, 45.4893)]
                label (text "4") [(0, 14.5996) (5.83594, 25.4893)]
            legend legend.color.cat right
              title (text "kind") [(104.754, 5.44482) (124.695, 17.5444)]
              entry 0.25 [(104.754, 20.4893) (114.754, 30.4893)] (text "a") [(117.254, 20.0444) (122.308, 30.9341)]
              entry 0.75 [(104.754, 31.3789) (114.754, 41.3789)] (text "b") [(117.254, 30.9341) (122.765, 41.8237)]
            legend legend.color.num right
              title (text "load") [(134.695, 0) (154.949, 12.0996)]
              bar [(134.695, 20.0444) (144.695, 80.0444)]
              label (text "1") [(150.695, 74.5996) (156.531, 85.4893)]
              label (text "2") [(150.695, 54.5996) (156.531, 65.4893)]
              label (text "3") [(150.695, 34.5996) (156.531, 45.4893)]
              label (text "4") [(150.695, 14.5996) (156.531, 25.4893)]
            ticks
              "x" quantitative
                (ticks (0 "0") (0.5 "2") (1 "4")
                 (minor 0.125 0.25 0.375 0.625 0.75 0.875))
              "y" quantitative
                (ticks (0 "1") (0.333333 "2") (0.666667 "3") (1 "4")
                 (minor 0.0666667 0.133333 0.2 0.266667 0.4 0.466667 0.533333 0.6
                  0.733333 0.8 0.866667 0.933333))
              "color" categorical (ticks (0.25 "a") (0.75 "b"))
              "color" quantitative
                (ticks (0 "1") (0.333333 "2") (0.666667 "3") (1 "4")
                 (minor 0.0666667 0.133333 0.2 0.266667 0.4 0.466667 0.533333 0.6
                  0.733333 0.8 0.866667 0.933333))
            |});
      test "titles centre on the data areas or align with the outer edges"
        (fun () ->
          let at align =
            let l =
              lay (Size.panels 80. 60.)
                (plain ~y:(f64 [| 1.; 2e6; 3.; 4. |]) ramp
                |> title ~align (Text.v "T"))
            in
            let x0, _, x1, _ = List.hd (texts l |> List.rev) in
            (l, x0, x1)
          in
          let l, x0, x1 = at `Center in
          let b = List.hd (boxes l) in
          equal (float 1e-3) (P2.x (Box2.mid b)) ((x0 +. x1) /. 2.);
          let _, x0, _ = at `Left in
          equal (float 1e-3) 0. x0;
          let l, _, x1 = at `Right in
          equal (float 1e-3) (fst (Layout.size l)) x1);
      test "a legend on top or at the bottom holds its title" (fun () ->
          let cats =
            strings
              ~title:(Text.v "A long legend title")
              [| "a"; "b"; "a"; "b" |]
          in
          let nums = num ~title:(Text.v "A long legend title") ramp in
          let at side f = layer [ f; legend ~side "color" ] in
          List.iter
            (fun side ->
              apart
                (lay (Size.panels 30. 30.) (at side (plain ~fill:cats ramp)));
              apart
                (lay (Size.panels 30. 30.) (at side (plain ~fill:nums ramp))))
            [ `Top; `Bottom ]);
      test "legends on each side" (fun () ->
          let f side =
            layer
              [
                plain
                  ~fill:
                    (strings ~title:(Text.v "kind") [| "a"; "b"; "a"; "b" |])
                  ramp;
                rect ~x:(num ramp) ~fill:(num ~title:(Text.v "load") ramp) ();
                legend ~side "color";
              ]
          in
          let l side = printed (lay (Size.panels 80. 60.) (f side)) in
          expect (String.concat "\n" [ l `Left; l `Top; l `Bottom ])
          @@ __POS_OF__
               {|
            layout 156.531 × 96.9341
            panel root [(73.6133, 20.0444) (153.613, 80.0444)] cartesian
              axis axis.x bottom
                label (text "0") [(70.6953, 86.0444) (76.5312, 96.9341)]
                label (text "2") [(110.695, 86.0444) (116.531, 96.9341)]
                label (text "4") [(150.695, 86.0444) (156.531, 96.9341)]
              axis axis.y left
                label (text "1") [(61.7773, 74.5996) (67.6133, 85.4893)]
                label (text "2") [(61.7773, 54.5996) (67.6133, 65.4893)]
                label (text "3") [(61.7773, 34.5996) (67.6133, 45.4893)]
                label (text "4") [(61.7773, 14.5996) (67.6133, 25.4893)]
            legend legend.color.num left
              title (text "load") [(0, 0) (20.2539, 12.0996)]
              bar [(0, 20.0444) (10, 80.0444)]
              label (text "1") [(16, 74.5996) (21.8359, 85.4893)]
              label (text "2") [(16, 54.5996) (21.8359, 65.4893)]
              label (text "3") [(16, 34.5996) (21.8359, 45.4893)]
              label (text "4") [(16, 14.5996) (21.8359, 25.4893)]
            legend legend.color.cat left
              title (text "kind") [(31.8359, 5.44482) (51.7773, 17.5444)]
              entry 0.25 [(31.8359, 20.4893) (41.8359, 30.4893)] (text "a") [(44.3359, 20.0444) (49.3896, 30.9341)]
              entry 0.75 [(31.8359, 31.3789) (41.8359, 41.3789)] (text "b") [(44.3359, 30.9341) (49.8467, 41.8237)]
            ticks
              "x" quantitative
                (ticks (0 "0") (0.5 "2") (1 "4")
                 (minor 0.125 0.25 0.375 0.625 0.75 0.875))
              "y" quantitative
                (ticks (0 "1") (0.333333 "2") (0.666667 "3") (1 "4")
                 (minor 0.0666667 0.133333 0.2 0.266667 0.4 0.466667 0.533333 0.6
                  0.733333 0.8 0.866667 0.933333))
              "color" categorical (ticks (0.25 "a") (0.75 "b"))
              "color" quantitative
                (ticks (0 "1") (0.333333 "2") (0.666667 "3") (1 "4")
                 (minor 0.0666667 0.133333 0.2 0.266667 0.4 0.466667 0.533333 0.6
                  0.733333 0.8 0.866667 0.933333))
            layout 94.7539 × 169.313
            panel root [(11.8359, 92.4233) (91.8359, 152.423)] cartesian
              axis axis.x bottom
                label (text "0") [(8.91797, 158.423) (14.7539, 169.313)]
                label (text "2") [(48.918, 158.423) (54.7539, 169.313)]
                label (text "4") [(88.918, 158.423) (94.7539, 169.313)]
              axis axis.y left
                label (text "1") [(0, 146.979) (5.83594, 157.868)]
                label (text "2") [(0, 126.979) (5.83594, 137.868)]
                label (text "3") [(0, 106.979) (5.83594, 117.868)]
                label (text "4") [(0, 86.9785) (5.83594, 97.8682)]
            legend legend.color.num top
              title (text "load") [(11.8359, 0) (32.0898, 12.0996)]
              bar [(11.8359, 14.5996) (91.8359, 24.5996)]
              label (text "1") [(8.91797, 30.5996) (14.7539, 41.4893)]
              label (text "2") [(35.5846, 30.5996) (41.4206, 41.4893)]
              label (text "3") [(62.2513, 30.5996) (68.0872, 41.4893)]
              label (text "4") [(88.918, 30.5996) (94.7539, 41.4893)]
            legend legend.color.cat top
              title (text "kind") [(11.8359, 51.4893) (31.7773, 63.5889)]
              entry 0.25 [(11.8359, 66.5337) (21.8359, 76.5337)] (text "a") [(24.3359, 66.0889) (29.3896, 76.9785)]
              entry 0.75 [(34.8467, 66.5337) (44.8467, 76.5337)] (text "b") [(47.3467, 66.0889) (52.8574, 76.9785)]
            ticks
              "x" quantitative
                (ticks (0 "0") (0.5 "2") (1 "4")
                 (minor 0.125 0.25 0.375 0.625 0.75 0.875))
              "y" quantitative
                (ticks (0 "1") (0.333333 "2") (0.666667 "3") (1 "4")
                 (minor 0.0666667 0.133333 0.2 0.266667 0.4 0.466667 0.533333 0.6
                  0.733333 0.8 0.866667 0.933333))
              "color" categorical (ticks (0.25 "a") (0.75 "b"))
              "color" quantitative
                (ticks (0 "1") (0.333333 "2") (0.666667 "3") (1 "4")
                 (minor 0.0666667 0.133333 0.2 0.266667 0.4 0.466667 0.533333 0.6
                  0.733333 0.8 0.866667 0.933333))
            layout 94.7539 × 169.313
            panel root [(11.8359, 5.44482) (91.8359, 65.4448)] cartesian
              axis axis.x bottom
                label (text "0") [(8.91797, 71.4448) (14.7539, 82.3345)]
                label (text "2") [(48.918, 71.4448) (54.7539, 82.3345)]
                label (text "4") [(88.918, 71.4448) (94.7539, 82.3345)]
              axis axis.y left
                label (text "1") [(0, 60) (5.83594, 70.8896)]
                label (text "2") [(0, 40) (5.83594, 50.8896)]
                label (text "3") [(0, 20) (5.83594, 30.8896)]
                label (text "4") [(0, 0) (5.83594, 10.8896)]
            legend legend.color.cat bottom
              title (text "kind") [(11.8359, 92.3345) (31.7773, 104.434)]
              entry 0.25 [(11.8359, 107.379) (21.8359, 117.379)] (text "a") [(24.3359, 106.934) (29.3896, 117.824)]
              entry 0.75 [(34.8467, 107.379) (44.8467, 117.379)] (text "b") [(47.3467, 106.934) (52.8574, 117.824)]
            legend legend.color.num bottom
              title (text "load") [(11.8359, 127.824) (32.0898, 139.923)]
              bar [(11.8359, 142.423) (91.8359, 152.423)]
              label (text "1") [(8.91797, 158.423) (14.7539, 169.313)]
              label (text "2") [(35.5846, 158.423) (41.4206, 169.313)]
              label (text "3") [(62.2513, 158.423) (68.0872, 169.313)]
              label (text "4") [(88.918, 158.423) (94.7539, 169.313)]
            ticks
              "x" quantitative
                (ticks (0 "0") (0.5 "2") (1 "4")
                 (minor 0.125 0.25 0.375 0.625 0.75 0.875))
              "y" quantitative
                (ticks (0 "1") (0.333333 "2") (0.666667 "3") (1 "4")
                 (minor 0.0666667 0.133333 0.2 0.266667 0.4 0.466667 0.533333 0.6
                  0.733333 0.8 0.866667 0.933333))
              "color" categorical (ticks (0.25 "a") (0.75 "b"))
              "color" quantitative
                (ticks (0 "1") (0.333333 "2") (0.666667 "3") (1 "4")
                 (minor 0.0666667 0.133333 0.2 0.266667 0.4 0.466667 0.533333 0.6
                  0.733333 0.8 0.866667 0.933333))
            |});
      test "bottom entries wrap at their panels' width" (fun () ->
          let n = 40 in
          let cats = Array.init n (Printf.sprintf "category %d") in
          let f =
            layer
              [
                dot
                  ~x:(num (f64 (Array.init n Float.of_int)))
                  ~y:(num (f64 (Array.init n Float.of_int)))
                  ~fill:(strings cats) ();
                legend ~side:`Bottom "color";
              ]
          in
          let l = lay (Size.figure 360. 240.) f in
          apart l);
      test "a centred title beside a left legend stays on the page" (fun () ->
          let f =
            layer
              [
                plain ~fill:(strings [| "a"; "b"; "a"; "b" |]) ramp;
                legend ~side:`Left "color";
              ]
            |> title (Text.v "A centred title")
          in
          apart (lay (Size.panels 22. 79.) f));
      test "a laid-out text prints on one line" (fun () ->
          let t =
            "A title long enough, with a box after it, to reach past the right \
             margin of a formatter"
          in
          let l = lay (Size.panels 300. 60.) (title (Text.v t) (plain ramp)) in
          let line =
            List.find
              (fun line -> contains line "title (text")
              (String.split_on_char '\n' (printed l))
          in
          in_order ~subs:[ t; "[(" ] line);
      test "a title is a quarter em above what it titles" (fun () ->
          let f =
            layer [ plain ramp; axis ~show:false "x"; axis ~show:false "y" ]
            |> title (Text.v "T")
          in
          let l = lay (Size.panels 50. 40.) f in
          let _, _, _, y1 = text_box l {|(text ("T" bold))|} in
          equal (float 1e-3) pad (Box2.miny (List.hd (boxes l)) -. y1));
      test "a figure title is set bold at 1.2 em" (fun () ->
          let l =
            lay (Size.panels 50. 40.) (title (Text.v "Tg") (plain ramp))
          in
          let _, y0, _, y1 = text_box l {|(text ("Tg" bold))|} in
          equal (float 1e-3) (head_h "Tg") (y1 -. y0));
      test "a facet title is a quarter em above the headers" (fun () ->
          let fx = strings ~title:(Text.v "model") [| "a"; "b"; "a"; "b" |] in
          let l =
            lay (Size.panels 50. 40.) (dot ~x:(num ramp) ~y:(num ramp) ~fx ())
          in
          let _, _, _, y1 = text_box l {|(text "model")|} in
          let _, y0, _, _ = text_box l {|(text "a")|} in
          equal (float 1e-3) pad (y0 -. y1));
      test "facet titles are a quarter em beyond the headers of square panels"
        (fun () ->
          let f =
            rect
              ~x:(strings [| "a"; "b"; "a"; "b" |])
              ~y:(strings [| "p"; "p"; "q"; "q" |])
              ~fx:(strings ~title:(Text.v "head") [| "h0"; "h1"; "h0"; "h1" |])
              ~fy:(strings ~title:(Text.v "layer") [| "l0"; "l0"; "l1"; "l1" |])
              ()
            |> coord (Coord.cartesian ~aspect:1. ())
          in
          (* The panels leave room below them in a tall figure, and beside them
             in a wide one. *)
          List.iter
            (fun (w, h) ->
              let l = lay (Size.figure w h) f in
              let _, _, _, above = text_box l {|(text "head")|} in
              let _, top, _, _ = text_box l {|(text "h0")|} in
              equal ~msg:"head" (float 1e-3) pad (top -. above);
              let beside, _, _, _ = text_box l {|(text "layer")|} in
              let _, _, right, _ = text_box l {|(text "l0")|} in
              equal ~msg:"layer" (float 1e-3) pad (beside -. right))
            [ (300., 500.); (600., 200.) ]);
      test "ticks are frozen as chosen at the lengths of the second solve"
        (fun () ->
          (* Without guides x is 100 points long and takes three ticks; the
             first choice's y labels and title leave it room for two. *)
          let f =
            plain ~y:(f64 [| 0.; 2e6; 1e6; 5e5 |]) (f64 [| 0.; 1.; 0.5; 0.25 |])
          in
          let p = printed (lay (Size.figure 100. 100.) f) in
          let rec ticks i =
            if String.sub p i 6 = "\nticks" then i else ticks (i + 1)
          in
          let i = ticks 0 in
          expect (String.sub p i (String.length p - i))
          @@ __POS_OF__
               {|
            ticks
              "x" quantitative (ticks (0 "0") (1 "1") (minor 0.2 0.4 0.6 0.8))
              "y" quantitative
                (ticks (0 "0") (0.5 "1") (1 "2") (minor 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9)
                 (note "×10⁶"))
            |});
      test "a hidden axis takes no room" (fun () ->
          let l =
            lay (Size.panels 50. 40.)
              (layer [ plain ramp; axis ~show:false "x"; axis ~show:false "y" ])
          in
          equal (pair close close) (50., 40.) (Layout.size l));
    ]

(* Glyphs *)

let glyphs =
  group "glyphs"
    [
      test "a title no face draws raises naming the character and its node"
        (fun () ->
          fails_naming [ "1: "; "U+10FFFD" ] (fun () ->
              lay (Size.panels 50. 50.)
                (grid
                   [
                     [
                       plain ramp; title (Text.v ("a" ^ unmapped)) (plain ramp);
                     ];
                   ])));
      test "a channel title no face draws raises naming its axis" (fun () ->
          let f = dot ~x:(num ~title:(Text.v unmapped) ramp) ~y:(num ramp) () in
          fails_naming [ "axis.x: "; "U+10FFFD" ] (fun () ->
              lay (Size.panels 50. 50.) f));
      test "a facet title no face draws raises naming its axis" (fun () ->
          let fx = strings ~title:(Text.v unmapped) [| "a"; "b"; "a"; "b" |] in
          let f = dot ~x:(num ramp) ~y:(num ramp) ~fx () in
          fails_naming [ "axis.fx: "; "U+10FFFD" ] (fun () ->
              lay (Size.panels 50. 50.) f));
      test "a category label no face draws is a warning" (fun () ->
          let f =
            plain ~fill:(strings [| "ok"; "b" ^ unmapped; "ok"; "ok" |]) ramp
          in
          let l = lay (Size.panels 50. 50.) f in
          match Layout.warnings l with
          | [ (id, msg) ] ->
              equal string "legend.color.cat"
                (Format.asprintf "%a" Nx.Ptree.Path.pp id);
              in_order ~subs:[ "U+10FFFD" ] msg
          | ws -> failf "%d warnings" (List.length ws));
    ]

(* Reuse *)

let reuse =
  let theme = Theme.v ~size:8. () in
  let cases_ =
    [
      ("the same figure", plain ramp, plain ramp, None);
      ("other data", plain ramp, plain (f64 [| 1.; 2e6 |]), None);
      ("another theme", plain ramp, plain ramp, Some theme);
      ( "another arrangement",
        grid [ [ plain ramp; plain ramp ] ],
        grid [ [ plain ramp ]; [ plain ramp ] ],
        None );
    ]
  in
  group "reuse"
    [
      prop "laying out twice gives equal layouts" ~count:30 gen_case (fun c ->
          let r = resolve (figure_of c) in
          match layout c.size r with
          | l -> equal layout_t l (layout c.size r)
          | exception (Invalid_argument m as e) ->
              if not (contains m "needs") then raise e;
              assume false);
      cases
        ~name:(fun (n, _, _, _) -> n)
        "a previous layout reuses to the fresh one" cases_
        (fun (_, f, f', theme) ->
          let size = Size.panels 80. 60. in
          let prev = lay size f in
          let r = resolve f' in
          equal layout_t (layout ?theme size r) (layout ~prev ?theme size r));
      test "layouts in two themes differ" (fun () ->
          let r = resolve (plain ramp) in
          let size = Size.panels 80. 60. in
          equal bool false (Layout.equal (layout size r) (layout ~theme size r)));
    ]

(* Projections *)

let projections =
  let panel () =
    List.hd (Layout.panels (lay (Size.figure 200. 100.) (plain ramp)))
  in
  group "projections"
    [
      test "a cartesian projection puts the unit square on the data area"
        (fun () ->
          let (p : Layout.panel) = panel () in
          let at x y = Coord.point p.projection x y in
          equal (pair close close)
            (Box2.minx p.box, Box2.maxy p.box)
            (P2.x (at 0. 0.), P2.y (at 0. 0.));
          equal (pair close close)
            (Box2.maxx p.box, Box2.miny p.box)
            (P2.x (at 1. 1.), P2.y (at 1. 1.)));
      test "a panel of no height has no inverse" (fun () ->
          let f =
            layer [ plain ramp; axis ~show:false "x"; axis ~show:false "y" ]
            |> title (Text.v "t")
          in
          let p =
            List.hd
              (Layout.panels (lay (Size.figure 100. (head_h "t" +. pad)) f))
          in
          equal float_exact 0. (Box2.h p.box);
          equal
            (option (pair float_exact float_exact))
            None
            (Coord.invert p.projection (P2.v 1. 1.)));
      prop "inverting a projection undoes it"
        (Gen.pair (Gen.float_range (-2.) 2.) (Gen.float_range (-2.) 2.))
        (fun (x, y) ->
          let (p : Layout.panel) = panel () in
          match Coord.invert p.projection (Coord.point p.projection x y) with
          | Some (x', y') ->
              equal (pair (float 1e-9) (float 1e-9)) (x, y) (x', y')
          | None -> fail "no position");
    ]

let () =
  exit
    (run "Layout" [ sizes; grids; aspects; guides; glyphs; reuse; projections ])
