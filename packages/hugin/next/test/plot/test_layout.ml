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
let margin = 0.5 *. em
let title_gap = 0.5 *. em
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
      | ("bar" | "swatch") :: _ -> Some (first line)
      | _ -> None)

(* [apart l] states that the data areas of [l] lie on its page, and that no
   text, swatch or colour bar of [l] overlaps another, a data area or the page's
   edge: the layout's invariant. *)
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
  titles : (string * string * string) option;
      (* The titles of y and fx, and of the figure. *)
  align : Text.Layout.halign;
  size : [ `Size of Size.t | `Near of float ];
      (* A size, or a factor of the figure's least size. *)
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
    titles = None;
    align = `Center;
    size = `Size (Size.panels 50. 50.);
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
    (match c.titles with
    | None -> ""
    | Some (y, f, t) ->
        Printf.sprintf ", titled %S %S %S %s" y f t (align c.align))
    (fun ppf -> function
      | `Size s -> Size.pp ppf s
      | `Near k -> Format.fprintf ppf "%g of its least size" k)
    c.size

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

(* Titles of every length, some of several lines. *)
let words =
  [
    "a";
    "loss";
    "value";
    "a long title";
    "a much longer axis title";
    "a very very long title for a scale here";
    "two\nlines";
    "three\nline\ntitle";
  ]

let gen_titles =
  let w = Gen.of_list words in
  Gen.option (Gen.triple w w w)

let gen_case =
  Gen.bind
    (Gen.pair (Gen.int_range 1 3) (Gen.int_range 1 3))
    (fun (rows, cols) ->
      let exp = Gen.int_range (-4) 8 in
      let colour = Gen.of_list [ No_colour; Categories; Quantities ] in
      let side = Gen.of_list [ `Right; `Left; `Top; `Bottom ] in
      let align = Gen.of_list [ `Center; `Left; `Right ] in
      let size =
        Gen.one_of
          [
            Gen.map (fun s -> `Size s) gen_size;
            Gen.map (fun k -> `Near k) (Gen.float_range 0.98 1.2);
          ]
      in
      Gen.map
        (fun ( (exps, colour, side),
               ((facets, shared, aspect), (top, titles, align)),
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
            titles;
            align;
            size;
          })
        (Gen.triple
           (Gen.triple
              (Gen.list ~size:(Gen.constant (rows * cols)) (Gen.pair exp exp))
              colour side)
           (Gen.pair
              (Gen.triple Gen.bool Gen.bool Gen.bool)
              (Gen.triple Gen.bool gen_titles align))
           size))
  |> Gen.with_pp pp_case

let figure_of c =
  let cell (a, b) =
    let x = f64 [| -.(10. ** Float.of_int a); 10. ** Float.of_int b |] in
    let y = f64 [| 0.; 10. ** Float.of_int b |] in
    let title pick = Option.map (fun t -> Text.v (pick t)) c.titles in
    let fx =
      if c.facets then
        Some (strings ?title:(title (fun (_, f, _) -> f)) [| "left"; "right" |])
      else None
    in
    let ych = num ?title:(title (fun (y, _, _) -> y)) y in
    let mark ?fill () = dot ?fill ?fx ~x:(num x) ~y:ych () in
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
  match c.titles with
  | None -> g
  | Some (_, _, t) -> title ~align:c.align (Text.v t) g

(* [needs size f] is the size that [layout] names when [f] is too small for
   [size], if it is. *)
let needs size f =
  match layout size (resolve f) with
  | _ -> None
  | exception (Invalid_argument m as e) ->
      let key = "the figure needs " in
      let rec at i =
        if i + String.length key > String.length m then raise e
        else if String.sub m i (String.length key) = key then i
        else at (i + 1)
      in
      let i = at 0 in
      Scanf.sscanf
        (String.sub m i (String.length m - i))
        "the figure needs %f × %f pt"
        (fun w h -> Some (w, h))

(* [size_of c] is the size of [c]: a factor of a least size is that of the size
   a figure of one point names. *)
let size_of c =
  match c.size with
  | `Size s -> s
  | `Near k -> (
      match needs (Size.figure 1. 1.) (figure_of c) with
      | Some (w, h) -> Size.figure (k *. w) (k *. h)
      | None -> Size.figure 1. 1.)

(* [laid c] is the layout of [c], discarding a figure too small for it. *)
let laid c =
  let f = figure_of c in
  match needs (size_of c) f with
  | Some _ ->
      assume false;
      assert false
  | None -> layout (size_of c) (resolve f)

(* [lines l kind] is the printed lines of the guides of [l] of [kind], such as
   ["axis"] or ["scale title"], each with the lines of its elements. *)
let guide_lines l kind =
  let rec go acc cur = function
    | [] ->
        List.rev (match cur with None -> acc | Some g -> List.rev g :: acc)
    | line :: rest ->
        let acc =
          match cur with
          | Some g when not (String.starts_with ~prefix:"  " line) ->
              List.rev g :: acc
          | _ -> acc
        in
        let cur =
          if String.starts_with ~prefix:(kind ^ " ") line then Some [ line ]
          else if String.starts_with ~prefix:"  " line then
            Option.map (fun g -> line :: g) cur
          else None
        in
        go acc cur rest
  in
  go [] None (String.split_on_char '\n' (printed l))

(* [text_lines l s] is the printed lines of the texts of [l] holding [s]. *)
let text_lines l s =
  List.filter
    (fun line -> contains line "(text" && contains line s)
    (String.split_on_char '\n' (printed l))

(* [parts l] is [true] iff two titles of axes or headers of one node of [l]
   stand on one side at different depths. *)
let parts l =
  let node line =
    match String.split_on_char ' ' line with
    | _ :: _ :: id :: side :: _ ->
        let segs = String.split_on_char '.' id in
        let n = List.length segs in
        (String.concat "." (List.filteri (fun i _ -> i < n - 2) segs), side)
    | _ -> ("", "")
  in
  let titles =
    List.filter_map
      (function
        | head :: text :: _ ->
            let _, y0, _, _ = box_of text in
            Some (node head, y0)
        | _ -> None)
      (guide_lines l "scale title")
  in
  List.exists
    (fun (k, y) -> List.exists (fun (k', y') -> k = k' && y <> y') titles)
    titles

(* [overhangs l] is [true] iff a title of axes or headers of [l] reaches right
   of every data area. *)
let overhangs l =
  let right =
    List.fold_left (fun m b -> Float.max m (Box2.maxx b)) 0. (boxes l)
  in
  List.exists
    (function
      | _ :: text :: _ ->
          let _, _, x1, _ = box_of text in
          x1 > right
      | _ -> false)
    (guide_lines l "scale title")

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
          let at y =
            lay (Size.panels 100. 50.) (dot ~x:(num ramp) ~y:(strings y) ())
          in
          let short = at [| "a"; "b"; "a"; "b" |]
          and long = at [| "a"; "a long category"; "a"; "b" |] in
          greater float_exact
            ~than:(fst (Layout.size short))
            (fst (Layout.size long));
          equal close 100. (Box2.w (List.hd (boxes long))));
      test "a page has a half em margin" (fun () ->
          let l =
            lay (Size.panels 50. 40.)
              (layer [ plain ramp; axis ~show:false "x"; axis ~show:false "y" ])
          in
          equal (pair close close)
            (50. +. (2. *. margin), 40. +. (2. *. margin))
            (Layout.size l);
          equal (pair close close) (margin, margin)
            (Box2.minx (List.hd (boxes l)), Box2.miny (List.hd (boxes l))));
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
          let least = (2. *. margin) +. head_h "t" +. title_gap in
          let w = 100. in
          let b = List.hd (boxes (lay (Size.figure w least) f)) in
          equal (pair close close) (w -. (2. *. margin), 0.) (Box2.w b, Box2.h b);
          fails_naming [ "needs" ] (fun () ->
              lay (Size.figure w (least -. 0.5)) f));
    ]

(* Grids *)

let spines =
  prop "spines align across rows and columns" ~count:60 gen_case (fun c ->
      cover "facets" c.facets;
      cover "aspect" c.aspect;
      cover "titled facets" (c.facets && Option.is_some c.titles);
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

let grids =
  group "grids"
    [
      spines;
      equal_shares;
      pinned_share;
      test "titled facet blocks side by side align their panels" (fun () ->
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
          | _ -> fail "four panels");
      test "a gap is one em when nothing protrudes into it" (fun () ->
          let cell =
            layer [ plain ramp; axis ~show:false "x"; axis ~show:false "y" ]
          in
          let l = lay (Size.panels 50. 50.) (grid [ [ cell; cell ] ]) in
          match boxes l with
          | [ a; b ] ->
              equal close em (Box2.minx b -. Box2.maxx a);
              equal (pair close close)
                (110. +. (2. *. margin), 50. +. (2. *. margin))
                (Layout.size l)
          | _ -> fail "two panels");
      test "a gap holds the protrusions that meet it plus one em" (fun () ->
          let cell = layer [ plain ramp; axis ~show:false "x" ] in
          let l = lay (Size.panels 50. 50.) (grid [ [ cell; cell ] ]) in
          match boxes l with
          | [ a; b ] ->
              (* Both panels have the same y axis: its protrusion is the left
                 one of [a], and meets the gap from [b]. *)
              equal close
                (Box2.minx a -. margin +. em)
                (Box2.minx b -. Box2.maxx a)
          | _ -> fail "two panels");
      test "a left axis protrudes by its tick and widest label, not its title"
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
          equal close (margin +. tick +. pad +. label_w "bb") (left None);
          equal close (left None) (left (Some (Text.v "name"))));
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
            (fun h -> apart (lay (Size.panels 40. h) f))
            [ 45.; 50.; 60.; 61. ]);
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
          equal (pair close close)
            (100. +. (2. *. margin), 100. +. (2. *. margin))
            (Layout.size l));
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
        facets = true;
        size = `Size (Size.panels 20. 20.);
      };
      {
        base with
        rows = 1;
        cols = 3;
        exps = [ (0, 0); (0, 3); (0, 0) ];
        facets = true;
        shared = true;
        size = `Size (Size.panels 134.474 20.);
      };
      {
        base with
        rows = 2;
        cols = 1;
        exps = [ (0, 0); (0, 6) ];
        colour = Quantities;
        size = `Size (Size.figure 300. 250.);
      };
      {
        base with
        rows = 1;
        cols = 3;
        exps = [ (0, 0); (0, 0); (-1, -2) ];
        size = `Size (Size.figure 300. 250.);
      };
      (* Legend entries taller than the panel with an aspect they stand
         beside. *)
      {
        base with
        colour = Categories;
        aspect = true;
        top = true;
        size = `Size (Size.panels 20. 20.);
      };
      (* Titles that share a band at some lengths and part at shorter ones. *)
      {
        base with
        rows = 3;
        cols = 2;
        exps = [ (6, 0); (0, -1); (0, 0); (0, 0); (0, 6); (0, 0) ];
        colour = Quantities;
        side = `Top;
        aspect = true;
        top = true;
        titles = Some ("value", "value", "A figure");
        size = `Size (Size.figure 300. 300.609);
      };
      {
        base with
        rows = 3;
        cols = 1;
        exps = [ (0, 6); (0, 0); (0, 6) ];
        facets = true;
        aspect = true;
        titles = Some ("value", "value", "A figure");
        size = `Size (Size.figure 300. 364.662);
      };
    ]
  in
  prop "no text overlaps another, a data area or the page's edge" ~count:150
    ~examples gen_case (fun c ->
      cover "titled facets" (c.facets && Option.is_some c.titles);
      cover "legend" (c.colour <> No_colour);
      cover "near its least size"
        (match c.size with `Near _ -> true | `Size _ -> false);
      let l = laid c in
      cover "titles part" (parts l);
      cover "a title reaches past the data areas" (overhangs l);
      apart l)

(* The reach of a long y title wider than its panel. *)
let long_title =
  dot
    ~x:(num (f64 [| 0.; 1.; 3.; 4. |]))
    ~y:
      (num
         ~title:(Text.v "a very very long title for a scale here")
         (f64 [| 0.; 1.; 2.; 4. |]))
    ()

let least_sizes =
  let named =
    prop "a figure too small names a size that lays it out" ~count:40
      (Gen.pair gen_case (Gen.float_range 0.3 1.))
      (fun (c, k) ->
        let f = figure_of c in
        match needs (Size.figure 1. 1.) f with
        | None -> assume false
        | Some (w, h) -> (
            match needs (Size.figure (k *. w) (k *. h)) f with
            | None -> ()
            | Some (w', h') ->
                equal
                  (option (pair float_exact float_exact))
                  None
                  (needs (Size.figure w' h') f);
                apart (lay (Size.figure w' h') f)))
  in
  group "least sizes"
    [
      named;
      cases ~name:(Printf.sprintf "%g pt")
        "a long y title lays out or names a size that does at"
        [ 120.; 155.; 160.; 181.; 182.; 182.2; 182.29; 182.3 ] (fun w ->
          let size =
            match needs (Size.figure w 200.) long_title with
            | None -> Size.figure w 200.
            | Some (w, h) -> Size.figure w h
          in
          apart (lay size long_title));
    ]

(* [box_mid_x (x0, _, x1, _)] is the middle of a printed box along x. *)
let mid_x (x0, _, x1, _) = (x0 +. x1) /. 2.

let titles =
  let time = num ~title:(Text.v "time") ramp in
  group "titles"
    [
      test "a shared axis is labelled on the outer panel only, titled once"
        (fun () ->
          let l =
            lay (Size.panels 60. 40.)
              (grid
                 [
                   [ dot ~x:time ~y:(num ramp) () ];
                   [ dot ~x:time ~y:(num ramp) () ];
                 ]
              |> share [ ("x", `Shared) ])
          in
          let axes = guide_lines l "axis" in
          let labelled g = List.exists (fun line -> contains line "label") g in
          equal (list string) [ "axis 1.axis.x bottom" ]
            (List.filter_map
               (fun g ->
                 match g with
                 | h :: _ when contains h ".axis.x" && labelled g -> Some h
                 | _ -> None)
               axes);
          equal int 1 (List.length (text_lines l {|"time"|})));
      test "facets sharing an x axis are titled once, centred below them"
        (fun () ->
          let f =
            dot ~x:time ~y:(num ramp) ~fx:(strings [| "a"; "b"; "c"; "d" |]) ()
          in
          let l = lay (Size.panels 40. 30.) f in
          match text_lines l {|"time"|} with
          | [ line ] ->
              let hull =
                List.fold_left Box2.union (List.hd (boxes l)) (boxes l)
              in
              let _, y0, _, _ = box_of line in
              equal (float 1e-3) (P2.x (Box2.mid hull)) (mid_x (box_of line));
              greater float_exact ~than:(Box2.maxy hull) y0
          | lines -> failf "%d titles" (List.length lines));
      test "a y title stands above its axis, left-aligned with its labels"
        (fun () ->
          let f =
            dot ~x:(num ramp)
              ~y:(num ~title:(Text.v "loss") (f64 [| 0.; 10.; 100.; 1000. |]))
              ()
          in
          let l = lay (Size.panels 80. 60.) f in
          let labels =
            List.filter_map
              (fun line ->
                if contains line "label" then Some (box_of line) else None)
              (List.concat (guide_lines l "axis axis.y"))
          in
          let x0, _, _, y1 = text_box l {|(text "loss")|} in
          let left =
            List.fold_left (fun m (x, _, _, _) -> Float.min m x) infinity labels
          in
          let top =
            List.fold_left (fun m (_, y, _, _) -> Float.min m y) infinity labels
          in
          equal (float 1e-3) left x0;
          equal (float 1e-3) title_gap (top -. y1));
      test "a y title on the right is left-aligned with the labels" (fun () ->
          let f =
            layer
              [
                dot ~x:(num ramp) ~y:(num ~title:(Text.v "loss") ramp) ();
                axis ~side:`Right "y";
              ]
          in
          let l = lay (Size.panels 80. 60.) f in
          let x0, _, _, _ = text_box l {|(text "loss")|} in
          let labels = List.concat (guide_lines l "axis axis.y") in
          let lx, _, _, _ =
            box_of (List.find (fun line -> contains line "label") labels)
          in
          equal (float 1e-3) lx x0);
      test "no text is turned: headers beside a row read upright" (fun () ->
          let f =
            dot ~x:(num ramp) ~y:(num ramp)
              ~fy:(strings [| "upright"; "upright"; "row"; "row" |])
              ()
          in
          let x0, y0, x1, y1 =
            text_box (lay (Size.panels 40. 30.) f) {|(text "upright")|}
          in
          greater float_exact ~than:(y1 -. y0) (x1 -. x0));
      test "a figure title aligns with the data areas it titles" (fun () ->
          let at align =
            let l =
              lay (Size.panels 80. 60.)
                (plain ~y:(f64 [| 1.; 2e6; 3.; 4. |]) ramp
                |> title ~align (Text.v "T"))
            in
            (List.hd (boxes l), text_box l {|(text ("T" bold))|})
          in
          let b, (x0, _, _, _) = at `Left in
          equal (float 1e-3) (Box2.minx b) x0;
          let b, t = at `Center in
          equal (float 1e-3) (P2.x (Box2.mid b)) (mid_x t);
          let b, (_, _, x1, _) = at `Right in
          equal (float 1e-3) (Box2.maxx b) x1);
      test "a figure title aligns with the left edge by default" (fun () ->
          let l =
            lay (Size.panels 80. 60.)
              (plain ~y:(f64 [| 1.; 2e6; 3.; 4. |]) ramp |> title (Text.v "T"))
          in
          let x0, _, _, _ = text_box l {|(text ("T" bold))|} in
          equal (float 1e-3) (Box2.minx (List.hd (boxes l))) x0);
      test "a figure title is half an em above what it titles" (fun () ->
          let f =
            layer [ plain ramp; axis ~show:false "x"; axis ~show:false "y" ]
            |> title (Text.v "T")
          in
          let l = lay (Size.panels 50. 40.) f in
          let _, _, _, y1 = text_box l {|(text ("T" bold))|} in
          equal (float 1e-3) title_gap (Box2.miny (List.hd (boxes l)) -. y1));
      test "a figure title is set bold at 1.2 em" (fun () ->
          let l =
            lay (Size.panels 50. 40.) (title (Text.v "Tg") (plain ramp))
          in
          let _, y0, _, y1 = text_box l {|(text ("Tg" bold))|} in
          equal (float 1e-3) (head_h "Tg") (y1 -. y0));
      test "titles nest, the outer above the inner" (fun () ->
          let l =
            lay (Size.panels 50. 40.)
              (title (Text.v "outer") (title (Text.v "inner") (plain ramp)))
          in
          let _, _, _, y1 = text_box l {|(text ("outer" bold))|} in
          let _, y0, _, _ = text_box l {|(text ("inner" bold))|} in
          at_most float_exact ~than:y0 y1);
      test "a facet title is half an em above the headers" (fun () ->
          let fx = strings ~title:(Text.v "model") [| "a"; "b"; "a"; "b" |] in
          let l =
            lay (Size.panels 50. 40.) (dot ~x:(num ramp) ~y:(num ramp) ~fx ())
          in
          let _, _, _, y1 = text_box l {|(text "model")|} in
          let _, y0, _, _ = text_box l {|(text "a")|} in
          equal (float 1e-3) title_gap (y0 -. y1));
      test "a facet grid's y, fx and fy titles share a line above it" (fun () ->
          let f =
            dot ~x:(num ramp)
              ~y:(num ~title:(Text.v "y") ramp)
              ~fx:(strings ~title:(Text.v "head") [| "a"; "b"; "a"; "b" |])
              ~fy:(strings ~title:(Text.v "layer") [| "p"; "p"; "q"; "q" |])
              ()
          in
          let l = lay (Size.panels 80. 60.) f in
          let bottom s =
            let _, _, _, y1 = text_box l s in
            y1
          in
          equal (float 1e-3) (bottom {|(text "head")|}) (bottom {|(text "y")|});
          equal (float 1e-3) (bottom {|(text "head")|})
            (bottom {|(text "layer")|});
          apart l);
      cases ~name:Fun.id "titles sharing a line stand a gap apart, with"
        (List.init 12 (fun k -> "a head" ^ String.make k 'W'))
        (fun head ->
          let f =
            dot ~x:(num ramp)
              ~y:(num ~title:(Text.v "a y title") ramp)
              ~fx:(strings ~title:(Text.v head) [| "a"; "b"; "a"; "b" |])
              ()
          in
          let l = lay (Size.panels 60. 40.) f in
          let _, _, y_right, y_bottom = text_box l {|(text "a y title")|} in
          let h_left, _, _, h_bottom =
            text_box l (Printf.sprintf "(text %S)" head)
          in
          if Float.abs (y_bottom -. h_bottom) < 1e-6 then
            at_least float_exact ~than:(em -. 1e-6) (h_left -. y_right));
      test "a long y title moves above the facet title" (fun () ->
          let f =
            dot ~x:(num ramp)
              ~y:
                (num
                   ~title:(Text.v "a y title long enough to reach the middle")
                   ramp)
              ~fx:(strings ~title:(Text.v "head") [| "a"; "b"; "a"; "b" |])
              ()
          in
          let l = lay (Size.panels 60. 40.) f in
          let _, _, _, y1 = text_box l {|(text "a y title|} in
          let _, y0, _, _ = text_box l {|(text "head")|} in
          at_most float_exact ~than:y0 y1;
          apart l);
    ]

(* [legend_ids f] is the ids of the legends of [f] laid out. *)
let legend_ids f =
  String.split_on_char '\n' (printed (lay (Size.panels 80. 60.) f))
  |> List.filter_map (fun line ->
      match String.split_on_char ' ' line with
      | "legend" :: id :: _ -> Some id
      | _ -> None)

let classes = [| "cat"; "dog"; "cat"; "bird" |]

(* Figures whose categorical colours a shown axis may name, and whether each
   keeps its legend. *)
let named_by_axis =
  let bars ?fill ?(x = strings classes) () =
    rect ~x ~y:(num ramp)
      ~fill:(Option.value fill ~default:(strings classes))
      ()
  in
  [
    ("colours of the x data", bars (), false);
    ( "colours of the y data",
      rect ~x:(num ramp) ~y:(strings classes) ~fill:(strings classes) (),
      false );
    ( "facets labelling their axis on the outer panels",
      rect ~x:(strings classes) ~y:(num ramp) ~fill:(strings classes)
        ~fy:(strings [| "p"; "q"; "p"; "q" |])
        (),
      false );
    ( "colours of other data",
      bars ~fill:(strings [| "a"; "b"; "a"; "b" |]) (),
      true );
    ("a hidden axis", layer [ bars (); axis ~show:false "x" ], true);
    ("an explicit legend", layer [ bars (); legend "color" ], true);
    ( "a mark implying a legend",
      Mark.v ~name:"bars"
        [
          Mark.bind Role.x (strings classes);
          Mark.bind ~guide:true Role.fill (strings classes);
        ]
        (fun _ -> Picture.empty),
      true );
    ( "a second reader the axis does not name",
      layer
        [
          bars ();
          dot
            ~x:(strings [| "a"; "b"; "a"; "b" |])
            ~y:(num ramp) ~fill:(strings classes) ();
        ],
      true );
    ("quantities of the x data", plain ~fill:(num ramp) ramp, true);
  ]

(* [entries l] is the value and box of every legend swatch of [l], read from its
   printed form, where each is [swatch u [(x0, y0) (x1, y1)]]. *)
let entries l =
  String.split_on_char '\n' (printed l)
  |> List.filter_map (fun line ->
      match String.split_on_char ' ' (String.trim line) with
      | "swatch" :: u :: _ -> Some (float_of_string u, box_of line)
      | _ -> None)

let pp_areas ppf (a0, a1) = Format.fprintf ppf "(%g, %g)" a0 a1

(* Size legends: the areas of a scale, from a pool with an end of area 0, and
   data, some on a set domain. *)
let gen_sizes =
  Gen.triple
    (Gen.of_list ~pp:pp_areas [ (0., 100.); (0., 400.); (64., 0.); (9., 200.) ])
    (Gen.pair (Gen.int_range (-50) 50) (Gen.int_range 1 200))
    Gen.bool

let sizes_law (areas, (lo, span), set) =
  let lo = Float.of_int lo and hi = Float.of_int (lo + span) in
  let domain = if set then Some (lo, hi) else None in
  let data = f64 [| lo; (lo +. hi) /. 2.; hi |] in
  let size = num ~scale:(Scale.linear ~areas ?domain ()) data in
  let l =
    lay (Size.panels 80. 60.) (dot ~x:(num data) ~y:(num data) ~size ())
  in
  let a0, a1 = areas in
  let es = entries l in
  cover "an end of area 0 is a tick" (fst areas = 0. && List.length es > 0);
  greater int ~than:0 (List.length es);
  List.iter
    (fun (u, (x0, y0, x1, y1)) ->
      let area = a0 +. (u *. (a1 -. a0)) in
      let msg = Printf.sprintf "entry at %g" u in
      greater ~msg float_exact ~than:0. area;
      let across = 2. *. Float.sqrt (area /. Float.pi) in
      at_least ~msg float_exact ~than:(across -. 1e-3)
        (Float.min (x1 -. x0) (y1 -. y0)))
    es;
  apart l

(* The ticks [Ticks.choose] gives a guide nine labels long, with no least
   spacing, without those of area 0. *)
let size_ticks_law (areas, (lo, span), set) =
  let lo = Float.of_int lo and hi = Float.of_int (lo + span) in
  let domain = if set then Some (lo, hi) else None in
  let data = f64 [| lo; (lo +. hi) /. 2.; hi |] in
  let f =
    dot ~x:(num data) ~y:(num data)
      ~size:(num ~scale:(Scale.linear ~areas ?domain ()) data)
      ()
  in
  let r = resolve f in
  let fitted = Resolved.scale r (Scale.linear ~name:"size" ()) in
  let ticks =
    Hugin_next_kit.Ticks.choose ~length:1. ~measure:(fun _ -> 1. /. 9.) fitted
  in
  let a0, a1 = areas in
  let expected =
    List.filter_map
      (fun (t : Hugin_next_kit.Ticks.tick) ->
        if a0 +. (t.position *. (a1 -. a0)) > 0. then Some t.position else None)
      ticks.major
  in
  (* Printed values have six significant digits. *)
  equal
    (list (float 1e-5))
    expected
    (List.map fst (entries (layout (Size.panels 80. 60.) r)))

let legends =
  group "legends"
    [
      prop "a size legend leaves out area 0 and holds its circles" gen_sizes
        sizes_law;
      prop "a size legend's entries are the ticks of a guide nine labels long"
        gen_sizes size_ticks_law;
      cases
        ~name:(fun (n, _, _) -> n)
        "a categorical legend a shown axis names is left out" named_by_axis
        (fun (_, f, kept) -> equal bool kept (legend_ids f <> []));
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
            ~subs:[ "axis 0.axis.rate bottom"; "legend 1.legend.rate.num" ]
            p);
      test "a legend's id is its node's, its scale's name and kind" (fun () ->
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
                match String.split_on_char ' ' line with
                | "legend" :: id :: _ -> Some id
                | _ -> None)
          in
          equal (list string)
            [ "legend.color.cat"; "legend.color.num" ]
            (List.sort String.compare ids));
      test "a legend stands beside the smallest node holding its readers"
        (fun () ->
          let f =
            grid
              [
                [
                  plain ~fill:(strings [| "a"; "b"; "a"; "b" |]) ramp;
                  plain ramp;
                ];
              ]
          in
          let l = lay (Size.panels 60. 40.) f in
          match boxes l with
          | [ a; b ] ->
              List.iter
                (fun (x0, _, x1, _) ->
                  greater float_exact ~than:(Box2.maxx a) x0;
                  less float_exact ~than:(Box2.minx b) x1)
                (swatches l)
          | _ -> fail "two panels");
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
            layout 166.531 × 106.934
            panel root [(16.8359, 25.0444) (96.8359, 85.0444)] cartesian
            axis axis.x bottom
              rules 4
              label (text "0") [(13.918, 91.0444) (19.7539, 101.934)]
              label (text "2") [(53.918, 91.0444) (59.7539, 101.934)]
              label (text "4") [(93.918, 91.0444) (99.7539, 101.934)]
            axis axis.y left
              rules 5
              label (text "1") [(5, 79.5996) (10.8359, 90.4893)]
              label (text "2") [(5, 59.5996) (10.8359, 70.4893)]
              label (text "3") [(5, 39.5996) (10.8359, 50.4893)]
              label (text "4") [(5, 19.5996) (10.8359, 30.4893)]
            legend legend.color.cat right
              text (text "kind") [(109.754, 10.4448) (129.695, 22.5444)]
              swatch 0.25 [(109.754, 27.0444) (117.754, 35.0444)]
              swatch 0.75 [(109.754, 39.0444) (117.754, 47.0444)]
              label (text "a") [(120.254, 25.5996) (125.308, 36.4893)]
              label (text "b") [(120.254, 37.5996) (125.765, 48.4893)]
            legend legend.color.num right
              text (text "load") [(139.695, 5) (159.949, 17.0996)]
              bar [(139.695, 25.0444) (149.695, 85.0444)]
              rules 4
              label (text "1") [(155.695, 79.5996) (161.531, 90.4893)]
              label (text "2") [(155.695, 59.5996) (161.531, 70.4893)]
              label (text "3") [(155.695, 39.5996) (161.531, 50.4893)]
              label (text "4") [(155.695, 19.5996) (161.531, 30.4893)]
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
      test "legend rows are a swatch and 0.4 em apart" (fun () ->
          let f = plain ~fill:(strings [| "a"; "b"; "c"; "d" |]) ramp in
          match swatches (lay (Size.panels 80. 60.) f) with
          | (_, a0, _, a1) :: (_, b0, _, _) :: _ ->
              equal (float 1e-3) (0.8 *. em) (a1 -. a0);
              equal (float 1e-3) (1.2 *. em) (b0 -. a0)
          | _ -> fail "no swatches");
      test "a legend of stroke colours draws line swatches 1.5 em long"
        (fun () ->
          let f =
            line ~x:(num ramp) ~y:(num ramp)
              ~stroke:(strings [| "a"; "a"; "b"; "b" |])
              ()
          in
          match swatches (lay (Size.panels 80. 60.) f) with
          | (x0, y0, x1, y1) :: _ ->
              equal
                (pair (float 1e-3) (float 1e-3))
                (1.5 *. em, 0.8 *. em)
                (x1 -. x0, y1 -. y0)
          | [] -> fail "no swatches");
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
            [ `Top; `Bottom; `Left; `Right ]);
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
          let b = List.hd (boxes l) in
          List.iter
            (fun (_, _, x1, _) ->
              at_most float_exact ~than:(Box2.maxx b +. 1e-3) x1)
            (swatches l);
          apart l);
      test "a centred title beside a left legend stays on the page" (fun () ->
          let f =
            layer
              [
                plain ~fill:(strings [| "a"; "b"; "a"; "b" |]) ramp;
                legend ~side:`Left "color";
              ]
            |> title ~align:`Center (Text.v "A centred title")
          in
          apart (lay (Size.panels 22. 79.) f));
    ]

let guides =
  group "guides"
    [
      no_overlap;
      test "headers name each column and each row once" (fun () ->
          let f =
            dot ~x:(num ramp) ~y:(num ramp)
              ~fx:(strings [| "a"; "b"; "a"; "b" |])
              ~fy:(strings [| "p"; "p"; "q"; "q" |])
              ()
          in
          let l = lay (Size.panels 40. 30.) f in
          equal (list string) [ "a"; "b"; "p"; "q" ]
            (List.sort String.compare
               (List.map
                  (fun g ->
                    let line = List.nth g 1 in
                    let i = String.index line '"' in
                    String.sub line (i + 1)
                      (String.index_from line (i + 1) '"' - i - 1))
                  (guide_lines l "header"))));
      test "a laid-out text prints on one line" (fun () ->
          let t =
            "A title long enough, with a box after it, to reach past the right \
             margin of a formatter"
          in
          let l = lay (Size.panels 300. 60.) (title (Text.v t) (plain ramp)) in
          match text_lines l t with
          | [ line ] -> in_order ~subs:[ t; "[(" ] line
          | lines -> failf "%d lines" (List.length lines));
      test "ticks are frozen as chosen at the lengths of the second solve"
        (fun () ->
          (* Without guides x is 100 points long and takes three ticks; the
             first choice's y labels leave it room for two. *)
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
                (ticks (0 "0") (1 "2") (minor 0.25 0.5 0.75) (note "×10⁶"))
            |});
      test "a hidden axis takes no room" (fun () ->
          let l =
            lay (Size.panels 50. 40.)
              (layer [ plain ramp; axis ~show:false "x"; axis ~show:false "y" ])
          in
          equal (pair close close)
            (50. +. (2. *. margin), 40. +. (2. *. margin))
            (Layout.size l));
    ]

(* Explicit ticks and notations *)

(* [axis_labels l id] is the text of each label of the axis [id] of [l], read
   from its printed form, where each is [label (text "…") …]. *)
let axis_labels l id =
  match guide_lines l ("axis " ^ id) with
  | [ _ :: lines ] ->
      List.filter_map
        (fun line ->
          match String.index_opt line '"' with
          | Some i when contains line "label (text" ->
              Some
                (String.sub line (i + 1)
                   (String.index_from line (i + 1) '"' - i - 1))
          | _ -> None)
        lines
  | gs -> failf "%d axes %s" (List.length gs) id

let quantities_labels ~notation ~ticks =
  let scale = Scale.linear ~name:"x" ~ticks ~notation () in
  let unit = f64 [| 0.; 1. |] in
  let r = resolve (dot ~x:(num ~scale unit) ~y:(num unit) ()) in
  let expected =
    Hugin_next_kit.Ticks.of_values ~notation (Resolved.scale r scale) ticks
  in
  equal (list string)
    (List.map (fun (t : Hugin_next_kit.Ticks.tick) -> t.label) expected.major)
    (axis_labels (layout (Size.panels 300. 60.) r) "axis.x")

let ticks =
  group "ticks"
    [
      test "explicit ticks are labelled as of_values labels them" (fun () ->
          quantities_labels ~notation:Percent
            ~ticks:[| 1.; 0.; 0.25; 2.; Float.nan |];
          quantities_labels ~notation:Si ~ticks:[| 0.5; 0.5; 1. |]);
      test "chosen ticks are labelled in the scale's notation" (fun () ->
          let scale = Scale.linear ~notation:Percent () in
          let data = f64 [| 0.; 0.37; 1. |] in
          let l =
            lay (Size.panels 200. 60.)
              (dot ~x:(num ~scale data) ~y:(num data) ())
          in
          let labels = axis_labels l "axis.x" in
          greater int ~than:1 (List.length labels);
          List.iter
            (fun s ->
              equal ~msg:s string "%" (String.sub s (String.length s - 1) 1))
            labels);
      test "explicit ticks of a band scale are its categories among them"
        (fun () ->
          let scale = Scale.band ~ticks:[| "c"; "z"; "a"; "c" |] () in
          let f =
            rect
              ~x:(strings ~scale [| "a"; "b"; "c" |])
              ~y:(num (f64 [| 1.; 2.; 3. |]))
              ()
          in
          equal (list string) [ "a"; "c" ]
            (axis_labels (lay (Size.panels 100. 60.) f) "axis.x"));
    ]

(* Inside legends *)

let corners =
  [
    ("top left", `Top_left);
    ("top right", `Top_right);
    ("bottom left", `Bottom_left);
    ("bottom right", `Bottom_right);
  ]

let kinds = [| "cat"; "dog"; "bird" |]

(* Figures whose colour legend may stand inside: categories and a bar. *)
let inside_figures =
  let data = f64 [| 1.; 2.; 3. |] in
  [
    ( "categories",
      dot ~x:(num data) ~y:(num data)
        ~fill:(strings ~title:(Text.v "kind") kinds)
        () );
    ( "a colour bar",
      dot ~x:(num data) ~y:(num data) ~fill:(num ~title:(Text.v "z") data) () );
  ]

let named l = Gen.of_list ~pp:(fun ppf (n, _) -> Format.pp_print_string ppf n) l

let gen_inside =
  Gen.triple (named inside_figures) (named corners)
    (named
       [
         ("panels", Size.panels 200. 150.);
         ("figure", Size.figure 360. 240.);
         ("wide figure", Size.figure 600. 200.);
       ])

let unmoved_law ((_, f), (_, c), (_, size)) =
  let with_legend g = boxes (lay size (layer [ f; g ])) in
  equal
    (list (Testable.make ~pp:Box2.pp ~equal:Box2.equal))
    (with_legend (legend ~show:false "color"))
    (with_legend (legend ~side:(`Inside c) "color"))

(* [legend_box l] is the hull of the boxes of the elements of the one legend of
   [l], read from its printed form. *)
let legend_box l =
  match guide_lines l "legend" with
  | [ _ :: lines ] ->
      List.fold_left
        (fun (a0, b0, a1, b1) line ->
          if String.contains line '[' then
            let x0, y0, x1, y1 = box_of line in
            (Float.min a0 x0, Float.min b0 y0, Float.max a1 x1, Float.max b1 y1)
          else (a0, b0, a1, b1))
        (infinity, infinity, neg_infinity, neg_infinity)
        lines
  | gs -> failf "%d legends" (List.length gs)

let corner_case (_, c) =
  let l =
    lay (Size.panels 200. 150.)
      (layer [ snd (List.hd inside_figures); legend ~side:(`Inside c) "color" ])
  in
  let box = List.hd (boxes l) in
  let x0, y0, x1, y1 = legend_box l in
  (* Printed boxes have six significant digits. *)
  let near = float 1e-3 in
  let inset = 0.25 *. em in
  (match c with
  | `Top_left | `Bottom_left ->
      equal ~msg:"left" near (Box2.minx box +. inset) x0;
      at_most ~msg:"right" float_exact ~than:(Box2.maxx box -. inset) x1
  | `Top_right | `Bottom_right ->
      equal ~msg:"right" near (Box2.maxx box -. inset) x1;
      at_least ~msg:"left" float_exact ~than:(Box2.minx box +. inset) x0);
  match c with
  | `Top_left | `Top_right ->
      equal ~msg:"top" near (Box2.miny box +. inset) y0;
      at_most ~msg:"bottom" float_exact ~than:(Box2.maxy box -. inset) y1
  | `Bottom_left | `Bottom_right ->
      equal ~msg:"bottom" near (Box2.maxy box -. inset) y1;
      at_least ~msg:"top" float_exact ~than:(Box2.miny box +. inset) y0

let many = Array.init 8 (Printf.sprintf "kind %d")

(* Two marks of four categories each, with the legend [g]. *)
let crowded_with g =
  layer
    [
      dot ~x:(num ramp) ~y:(num ramp) ~fill:(strings (Array.sub many 0 4)) ();
      dot ~x:(num ramp) ~y:(num ramp) ~fill:(strings (Array.sub many 4 4)) ();
      g;
    ]

let crowded = crowded_with (legend ~side:(`Inside `Top_left) "color")

let inside_legends =
  group "inside legends"
    [
      prop "an inside legend leaves the data areas as a hidden one does"
        gen_inside unmoved_law;
      cases "an inside legend lies in its corner, a pad from both edges"
        ~name:fst corners corner_case;
      test "an inside legend keeps the id of a legend" (fun () ->
          equal (list string) [ "legend.color.cat" ]
            (legend_ids
               (layer
                  [
                    snd (List.hd inside_figures);
                    legend ~side:(`Inside `Bottom_left) "color";
                  ])));
      test "panels grow to hold an inside legend and its pads" (fun () ->
          let l = lay (Size.panels 40. 40.) crowded in
          let box = List.hd (boxes l) in
          let x0, y0, x1, y1 = legend_box l in
          let pads = 2. *. 0.25 *. em in
          at_least ~msg:"width" float_exact
            ~than:(x1 -. x0 +. pads -. 1e-3)
            (Box2.w box);
          at_least ~msg:"height" float_exact
            ~than:(y1 -. y0 +. pads -. 1e-3)
            (Box2.h box));
      test "a figure too small for an inside legend names a size that fits"
        (fun () ->
          let size = Size.figure 120. 100. in
          ignore (lay size (crowded_with (legend ~show:false "color")));
          fails_naming [ "needs"; "120" ] (fun () -> lay size crowded));
    ]

(* Theme presets *)

(* [luminance c] is the WCAG relative luminance of the opaque colour [c]. *)
let luminance c =
  let lin v =
    if v <= 0.04045 then v /. 12.92 else Float.pow ((v +. 0.055) /. 1.055) 2.4
  in
  (0.2126 *. lin (Color.r c))
  +. (0.7152 *. lin (Color.g c))
  +. (0.0722 *. lin (Color.b c))

(* [contrast a b] is the WCAG contrast ratio of [a] and [b]. *)
let contrast a b =
  let la = luminance a and lb = luminance b in
  (Float.max la lb +. 0.05) /. (Float.min la lb +. 0.05)

(* [over a c p] is [c] at the opacity [a] composited over [p], in encoded sRGB
   as renderers composite. *)
let over a c p =
  let mix f = (a *. f c) +. ((1. -. a) *. f p) in
  Color.v (mix Color.r) (mix Color.g) (mix Color.b)

let readable (name, th) =
  let ink = Theme.ink th and paper = Theme.paper th in
  at_least ~msg:(name ^ " ink") float_exact ~than:7. (contrast ink paper);
  at_least ~msg:(name ^ " axes") float_exact ~than:3.
    (contrast (over 0.6 ink paper) paper)

let homogeneous_figures =
  let fill = strings ~title:(Text.v "kind") [| "a"; "b"; "a"; "b" |] in
  [
    ("dots", plain ramp);
    ( "a legend and a title",
      title (Text.v "Load") (dot ~x:(num ramp) ~y:(num ramp) ~fill ()) );
    ( "facets and a colour bar",
      rect ~x:(num ramp) ~y:(num ramp)
        ~fill:(num ~title:(Text.v "z") ramp)
        ~fx:(strings [| "p"; "q"; "p"; "q" |])
        () );
  ]

let ticks_of l =
  let p = printed l in
  let rec at i = if String.sub p i 6 = "\nticks" then i else at (i + 1) in
  let i = at 0 in
  String.sub p i (String.length p - i)

let homogeneity ((name, f), (k, (w, h))) =
  let w = Float.of_int w and h = Float.of_int h in
  let base = lay (Size.figure w h) f in
  let theme = Theme.v ~size:(k *. Theme.size Theme.default) () in
  let scaled = lay ~theme (Size.figure (k *. w) (k *. h)) f in
  let rel = float_rel ~rel:1e-9 ~abs:0. in
  let corners b = [ Box2.minx b; Box2.miny b; Box2.maxx b; Box2.maxy b ] in
  equal ~msg:name
    (list (list rel))
    (List.map (fun b -> List.map (fun v -> k *. v) (corners b)) (boxes base))
    (List.map corners (boxes scaled));
  equal ~msg:name string (ticks_of base) (ticks_of scaled)

let themes =
  group "themes"
    [
      cases "ink reads on paper, and axes at their opacity" ~name:fst
        [ ("default", Theme.default); ("dark", Theme.dark) ]
        readable;
      test "the presets are the themes of v they state" (fun () ->
          let theme = Testable.make ~pp:Theme.pp ~equal:Theme.equal in
          equal theme
            (Theme.v ~ink:(Color.gray 0.92) ~paper:(Color.gray 0.1) ())
            Theme.dark;
          equal theme (Theme.v ~size:16. ()) Theme.talk;
          equal theme (Theme.v ~size:20. ()) Theme.poster);
      prop "a theme k times the size lays a figure k times larger out alike"
        (Gen.pair
           (Gen.of_list
              ~pp:(fun ppf (n, _) -> Format.pp_print_string ppf n)
              homogeneous_figures)
           (Gen.pair
              (Gen.of_list ~pp:Format.pp_print_float
                 [
                   Theme.size Theme.talk /. Theme.size Theme.default;
                   Theme.size Theme.poster /. Theme.size Theme.default;
                   0.5;
                   3.;
                 ])
              (Gen.pair (Gen.int_range 200 400) (Gen.int_range 150 300))))
        homogeneity;
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
          match layout (size_of c) r with
          | l -> equal layout_t l (layout (size_of c) r)
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
              (Layout.panels
                 (lay
                    (Size.figure 100.
                       ((2. *. margin) +. head_h "t" +. title_gap))
                    f))
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
    (run "Layout"
       [
         sizes;
         least_sizes;
         grids;
         aspects;
         guides;
         titles;
         legends;
         ticks;
         inside_legends;
         themes;
         glyphs;
         reuse;
         projections;
       ])
