(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin
open Windtrap
module Run = Hugin_font.Run
module Ticks = Hugin_kit.Ticks

(* Data *)

let f64 a = Nx.create Nx.float64 [| Array.length a |] a
let ramp = f64 [| 1.; 2.; 3.; 4. |]

let plain ?fill ?y x =
  dot ?fill ~x:(num x) ~y:(num (Option.value y ~default:x)) ()

(* A panel whose axes are hidden, so that nothing protrudes from it. *)
let bare = layer [ plain ramp; axis ~show:false "x"; axis ~show:false "y" ]

(* A panel with an aspect of one and no axes. *)
let square =
  layer
    [
      rect ~x:(strings [| "a" |]) ~y:(strings [| "p" |]) ();
      axis ~show:false "x";
      axis ~show:false "y";
    ]
  |> coord (Coord.cartesian ~aspect:1. ())

(* A missing character: no face of the default theme has the private use
   U+10FFFD. *)
let unmapped = "\u{10FFFD}"

(* Lengths

   The derived lengths of the default theme (Theme, "Derived lengths"). A
   laid-out length is a sum of a few dozen of them and of data lengths below
   1000 pt, each rounded once, so it agrees with its value to a few ulps of 1000
   pt: [close] allows 1e-9 pt. *)

let em = Theme.size Theme.default
let tick = 0.35 *. em
let pad = 0.25 *. em
let margin = 0.5 *. em
let title_gap = 0.5 *. em
let close = float 1e-9

(* [setting ~bold ~size s] is [s] set at [size] in the default theme's faces,
   bold if [bold], as layout sets labels (0.9 em), the titles of channels (1 em)
   and those of figures (bold, 1.2 em). *)
let setting ?(bold = false) ~size s =
  let t = Text.v s in
  Text.Layout.v
    ~fonts:(Theme.fonts Theme.default)
    ~size
    (if bold then Text.bold t else t)

let label_w s = Box2.w (Text.Layout.box (setting ~size:(0.9 *. em) s))
let head_h s = Box2.h (Text.Layout.box (setting ~bold:true ~size:(1.2 *. em) s))

(* Laying out *)

let lay ?theme size f = layout ?theme size (resolve f)
let boxes l = List.map (fun (p : Layout.panel) -> p.box) (Layout.panels l)

(* [panel_box l id] is the data area of the panel [id] of [l]. *)
let panel_box l id =
  match
    List.find_opt
      (fun (p : Layout.panel) -> Nx.Ptree.Path.to_string p.id = id)
      (Layout.panels l)
  with
  | Some p -> p.box
  | None -> failf "no panel %s" id

let box_t = Testable.make ~pp:Box2.pp ~equal:Box2.equal
let layout_t = Testable.make ~pp:Layout.pp ~equal:Layout.equal
let run_t = Testable.make ~pp:Run.pp ~equal:Run.equal
let path_t = Testable.make ~pp:Nx.Ptree.Path.pp ~equal:Nx.Ptree.Path.equal
let warning_t = pair path_t string
let ids ws = List.map (fun (id, _) -> Nx.Ptree.Path.to_string id) ws

(* [within outer b] is [true] iff [b] lies in [outer], to the rounding of
   [close]. *)
let within outer b =
  Box2.minx b >= Box2.minx outer -. 1e-9
  && Box2.miny b >= Box2.miny outer -. 1e-9
  && Box2.maxx b <= Box2.maxx outer +. 1e-9
  && Box2.maxy b <= Box2.maxy outer +. 1e-9

(* [fails_naming subs f] states that [f] raises [Invalid_argument] with a
   message holding each of [subs]. *)
let fails_naming subs f =
  raises_match
    (fun e -> List.for_all (fun sub -> Exn.invalid_arg ~sub e) subs)
    f

(* Figures too small *)

let too_small e = Exn.invalid_arg ~sub:"the figure needs " e

(* [tried size f] is the layout of [f] at [size], or the size that [layout]
   names in its message when [f] is too small for [size]. *)
let tried size f =
  match layout size (resolve f) with
  | l -> Ok l
  | exception (Invalid_argument m as e) when too_small e ->
      let key = "the figure needs " in
      let rec at i =
        if String.sub m i (String.length key) = key then i else at (i + 1)
      in
      let i = at 0 in
      Scanf.sscanf
        (String.sub m i (String.length m - i))
        "the figure needs %f × %f pt"
        (fun w h -> Error (w, h))

(* [needs size f] is the size that [layout] names when [f] is too small for
   [size], if it is. *)
let needs size f =
  Result.fold ~ok:(fun _ -> None) ~error:Option.some (tried size f)

(* Reading drawings

   A drawing shows what a layout placed. A text is drawn as glyph runs, one a
   line and style, under the tag of the axis, header or legend it belongs to; a
   figure title is under none. A legend swatch is tagged with its legend's id
   and entry, and a colour bar is an image, or steps, in its legend. Marks draw
   in their panels' tags, in stamps and transforms that no claim here reads. *)

type text = {
  id : string; (* The id of the innermost tag above the run, [""] for none. *)
  run : Run.t;
  color : Color.t;
  at : P2.t; (* The run's origin, on its baseline. *)
  ink : Box2.t;
}

let picture l = Renderable.picture (Drawing.renderable (draw ~density:1. l))
let id_of (tag : Picture.tag) = Nx.Ptree.Path.to_string tag.id
let segments id = String.split_on_char '.' id

(* [texts l] is every run with ink that [l] draws, in drawing order. *)
let texts l =
  let rec go id acc (p : Picture.t) =
    match p with
    | Glyphs { color; at; run } -> (
        match Picture.bounds p with
        | Some ink -> { id; run; color; at; ink } :: acc
        | None -> acc)
    | Group ps -> List.fold_left (go id) acc ps
    | Tag { tag; picture } -> go (id_of tag) acc picture
    | Clip { picture; _ } | Opacity { picture; _ } -> go id acc picture
    | Transform _ | Stamp _ | Empty | Fill _ | Stroke _ | Image _ -> acc
  in
  List.rev (go "" [] (picture l))

let words ts = List.map (fun t -> Run.text t.run) ts

(* Tick and legend labels are drawn at 0.75 of the ink's opacity, titles and
   headers in the ink whole (Theme, "Derived lengths"). *)
let is_label t = Color.alpha t.color < 1.

(* A header is under the tag of its panel's axis, as [panel.a.axis.fx]. *)
let is_header t = List.mem "panel" (segments t.id)

(* [titles l] is what the titles and headers of [l] read. *)
let titles l = words (List.filter (fun t -> not (is_label t)) (texts l))

(* [headers l] is what the headers of [l] read. *)
let headers l =
  words (List.filter (fun t -> is_header t && not (is_label t)) (texts l))

(* [guide_titles l] is what the titles of the axes, headers and legends of [l]
   read. *)
let guide_titles l =
  words
    (List.filter
       (fun t -> t.id <> "" && not (is_label t || is_header t))
       (texts l))

(* [labels l id] is the labels of the guide [id] of [l], in drawing order. *)
let labels l id = List.filter (fun t -> is_label t && t.id = id) (texts l)

(* [text l s] is the one text of [l] that reads [s]. *)
let text l s =
  match List.filter (fun t -> Run.text t.run = s) (texts l) with
  | [ t ] -> t
  | ts -> failf "%d texts read %S" (List.length ts) s

(* [set_box ~bold t] is the box the layout set the one-line text [t] in: its
   line's width, and its ascent above and descent below its baseline. It is the
   box of its text set at its size in the theme's faces, bold if [bold], placed
   at its origin. *)
let set_box ?bold t =
  let set = setting ?bold ~size:(Run.size t.run) (Run.text t.run) in
  Box2.transform
    (Affine.translate (P2.x t.at) (P2.y t.at))
    (Text.Layout.box set)

let is_legend (tag : Picture.tag) =
  List.mem (Nx.Ptree.Path.Field "legend") (Nx.Ptree.Path.segments tag.id)

(* [keys l] is the ink of each legend key of [l]: each swatch, and each colour
   bar whole. *)
let keys l =
  let rec inside (swatches, bar) (p : Picture.t) =
    match p with
    | Tag { picture; _ } ->
        (Option.to_list (Picture.bounds picture) @ swatches, bar)
    | Image _ | Fill _ -> (swatches, Option.to_list (Picture.bounds p) @ bar)
    | Group ps -> List.fold_left inside (swatches, bar) ps
    | Clip { picture; _ } | Opacity { picture; _ } ->
        inside (swatches, bar) picture
    | Transform _ | Stamp _ | Glyphs _ | Stroke _ | Empty -> (swatches, bar)
  in
  let rec go acc (p : Picture.t) =
    match p with
    | Tag { tag; picture } when is_legend tag -> (
        match inside ([], []) picture with
        | swatches, [] -> swatches @ acc
        | swatches, b :: bs ->
            (List.fold_left Box2.union b bs :: swatches) @ acc)
    | Tag { picture; _ } | Clip { picture; _ } | Opacity { picture; _ } ->
        go acc picture
    | Group ps -> List.fold_left go acc ps
    | Transform _ | Stamp _ | Glyphs _ | Fill _ | Stroke _ | Image _ | Empty ->
        acc
  in
  go [] (picture l)

(* [ink l id] is the box of what [l] draws under the tags of [id]. *)
let ink l id =
  let rec go acc (p : Picture.t) =
    match p with
    | Tag { tag; picture } when id_of tag = id ->
        Option.to_list (Picture.bounds picture) @ acc
    | Tag { picture; _ } | Clip { picture; _ } | Opacity { picture; _ } ->
        go acc picture
    | Group ps -> List.fold_left go acc ps
    | Transform _ | Stamp _ | Glyphs _ | Fill _ | Stroke _ | Image _ | Empty ->
        acc
  in
  match go [] (picture l) with
  | [] -> failf "nothing is drawn under %s" id
  | b :: bs -> List.fold_left Box2.union b bs

(* [guides l] is the id of each axis, header and legend that [l] draws, once
   each, sorted. *)
let guides l =
  let rec go acc (p : Picture.t) =
    match p with
    | Tag { tag; picture } ->
        let s = segments (id_of tag) in
        let guide = List.mem "axis" s || List.mem "legend" s in
        go (if guide then id_of tag :: acc else acc) picture
    | Clip { picture; _ } | Opacity { picture; _ } -> go acc picture
    | Group ps -> List.fold_left go acc ps
    | Transform _ | Stamp _ | Glyphs _ | Fill _ | Stroke _ | Image _ | Empty ->
        acc
  in
  List.sort_uniq String.compare (go [] (picture l))

let legends l =
  List.filter (fun id -> List.mem "legend" (segments id)) (guides l)

(* [apart l] states the layout's invariant: the data areas of [l] lie on its
   page, and no text's set box or legend key of [l] overlaps another, a data
   area or the page's edge, to the rounding of [close]. *)
let apart l =
  let w, h = Layout.size l in
  let on_page ~msg b =
    satisfies ~msg ~claim:"within the page" box_t (within (Box2.v 0. 0. w h)) b
  in
  List.iter (fun b -> on_page ~msg:"a data area" b) (boxes l);
  let bold t = Font.weight (Run.font t.run) = Font.weight Font.bold in
  let inks =
    List.map (fun t -> (Run.text t.run, set_box ~bold:(bold t) t)) (texts l)
    @ List.map (fun b -> ("a legend key", b)) (keys l)
  in
  let meet a b =
    match Box2.inter a b with
    | Some i -> Box2.w i > 1e-9 && Box2.h i > 1e-9
    | None -> false
  in
  List.iteri
    (fun i (s, a) ->
      on_page ~msg:s a;
      List.iteri
        (fun j (s', b) ->
          if j > i && meet a b then
            failf "%S %a and %S %a overlap" s Box2.pp a s' Box2.pp b)
        inks;
      List.iter
        (fun d ->
          if meet a d then
            failf "%S %a overlaps the data area %a" s Box2.pp a Box2.pp d)
        (boxes l))
    inks

(* [lays_out_named size f] states that [f] is too small for [size] and names a
   size at which it lays out, its texts apart. *)
let lays_out_named size f =
  match needs size f with
  | None -> fail "the figure lays out"
  | Some (w, h) ->
      equal
        (option (pair float_exact float_exact))
        None
        (needs (Size.figure w h) f);
      apart (lay (Size.figure w h) f)

(* [swatches ~guides size bindings read] lays out at [size] a probe mark of
   [bindings] layered with [guides], whose legend swatches fill their boxes, and
   draws it. It is the layout, and the box and [read rows] of each swatch, in
   entry order. *)
let swatches ?(guides = []) size bindings read =
  let seen = ref [] in
  let swatch rows =
    let p = Mark.projection rows in
    let box = Box2.of_pts (Coord.point p 0. 1.) (Coord.point p 1. 0.) in
    seen := ((Mark.index rows).(0), (box, read rows)) :: !seen;
    Picture.fill (Theme.ink (Mark.theme rows)) (Path.rect box)
  in
  let probe = Mark.v ~name:"probe" ~swatch bindings (fun _ -> Picture.empty) in
  let l = lay size (layer (probe :: guides)) in
  ignore (draw ~density:1. l);
  let entries = List.sort (fun (i, _) (j, _) -> Int.compare i j) !seen in
  (l, List.map snd entries)

(* [fill_swatches n] is the box of each swatch of a legend of [n] categories of
   fill colour. *)
let fill_swatches n =
  let cats = strings (Array.init n string_of_int) in
  let _, entries =
    swatches (Size.panels 80. 60.) [ Mark.bind Role.fill cats ] ignore
  in
  List.map fst entries

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
  size : [ `Figure of float * float | `Panels of float * float | `Near of float ];
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
    size = `Panels (50., 50.);
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
      | `Figure (w, h) -> Format.fprintf ppf "figure %g × %g" w h
      | `Panels (w, h) -> Format.fprintf ppf "panels %g × %g" w h
      | `Near k -> Format.fprintf ppf "%g of its least size" k)
    c.size

let gen_figure =
  Gen.map
    (fun s -> `Figure s)
    (Gen.pair (Gen.float_range 300. 700.) (Gen.float_range 250. 600.))

let gen_panels =
  Gen.map
    (fun s -> `Panels s)
    (Gen.pair (Gen.float_range 20. 150.) (Gen.float_range 20. 150.))

let gen_near = Gen.map (fun k -> `Near k) (Gen.float_range 0.98 1.2)

(* Titles of every length, some of several lines. *)
let title_pool =
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
  let w = Gen.of_list title_pool in
  Gen.option (Gen.triple w w w)

(* [gen_case_of size] draws cases of the sizes [size] draws. *)
let gen_case_of size =
  Gen.bind
    (Gen.pair (Gen.int_range 1 3) (Gen.int_range 1 3))
    (fun (rows, cols) ->
      let exp = Gen.int_range (-4) 8 in
      let colour = Gen.of_list [ No_colour; Categories; Quantities ] in
      let side = Gen.of_list [ `Right; `Left; `Top; `Bottom ] in
      let align = Gen.of_list [ `Center; `Left; `Right ] in
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

let gen_case =
  gen_case_of (Gen.one_of [ Gen.one_of [ gen_figure; gen_panels ]; gen_near ])

let figure_of c =
  let cell (a, b) =
    let x = f64 [| -.(10. ** Float.of_int a); 10. ** Float.of_int b |] in
    let y = f64 [| 0.; 10. ** Float.of_int b |] in
    let title pick = Option.map pick c.titles in
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
  match c.titles with None -> g | Some (_, _, t) -> title ~align:c.align t g

(* [page_of c] is the page size [c] fixes, if it fixes one: a factor of a least
   size is that of the size a figure of one point names. *)
let page_of c =
  match c.size with
  | `Figure s -> Some s
  | `Panels _ -> None
  | `Near k -> (
      match needs (Size.figure 1. 1.) (figure_of c) with
      | Some (w, h) -> Some (k *. w, k *. h)
      | None -> Some (1., 1.))

let size_of c =
  match (c.size, page_of c) with
  | `Panels (w, h), _ -> Size.panels w h
  | _, Some (w, h) -> Size.figure w h
  | _, None -> assert false

(* [laid c] is the layout of [c], discarding a figure too small for it. *)
let laid c =
  match tried (size_of c) (figure_of c) with Ok l -> l | Error _ -> reject ()

(* [titles_part ts] is [true] iff of the texts [ts], a y title and the facet
   title of its node, under the tags [n.axis.y] and [n.axis.fx], start on
   different baselines. *)
let titles_part ts =
  let top id =
    List.fold_left
      (fun m t ->
        if t.id = id && not (is_label t) then Float.min m (P2.y t.at) else m)
      infinity ts
  in
  List.exists
    (fun t ->
      (not (is_label t))
      && String.ends_with ~suffix:"axis.y" t.id
      &&
      let fx = String.sub t.id 0 (String.length t.id - 1) ^ "fx" in
      Float.is_finite (top fx) && top fx <> top t.id)
    ts

(* [wider l ts] is [true] iff of the texts [ts], a title of an axis or of
   headers of [l] is wider than the data areas together. *)
let wider l ts =
  let hull = List.fold_left Box2.union (List.hd (boxes l)) (boxes l) in
  List.exists
    (fun t ->
      (not (is_label t || is_header t))
      && List.mem "axis" (segments t.id)
      && Box2.w (set_box t) > Box2.w hull)
    ts

(* Conventions *)

let conventions =
  group "Conventions"
    [
      cases "an empty title is no title" ~id:fst
        [
          ("on a figure", (title "" (plain ramp), plain ramp));
          ( "on an axis",
            ( layer
                [
                  dot ~x:(num ~title:"time" ramp) ~y:(num ramp) ();
                  axis ~title:"" "x";
                ],
              layer [ plain ramp; axis "x" ] ) );
          ( "on a legend",
            ( layer
                [
                  plain ~fill:(num ~title:"z" ramp) ramp;
                  legend ~title:"" "color";
                ],
              plain ~fill:(num ramp) ramp ) );
          ( "on a channel",
            (dot ~x:(num ~title:"" ramp) ~y:(num ramp) (), plain ramp) );
        ]
        (fun (_, (f, untitled)) ->
          let l = lay (Size.panels 60. 40.) f
          and l' = lay (Size.panels 60. 40.) untitled in
          equal (list box_t) (boxes l') (boxes l);
          equal (list string) (words (texts l')) (words (texts l)));
      prop "a figure laid out twice gives equal layouts" ~count:30 gen_case
        (fun c ->
          let r = resolve (figure_of c) and size = size_of c in
          match layout size r with
          | l -> equal layout_t l (layout size r)
          | exception e when too_small e -> reject ());
    ]

(* Ids *)

let ids_group =
  group "Ids"
    [
      test "a generated axis' id is its node's, axis and its scale's name"
        (fun () ->
          let l =
            lay (Size.panels 60. 40.) (grid [ [ plain ramp; plain ramp ] ])
          in
          equal (list string)
            [ "0.axis.x"; "0.axis.y"; "1.axis.x"; "1.axis.y" ]
            (guides l));
      test
        "a generated legend's id is its node's, legend, its scale's name and \
         kind" (fun () ->
          let f =
            layer
              [
                plain ~fill:(strings [| "a"; "b"; "a"; "b" |]) ramp;
                plain ~fill:(num ramp) ramp;
              ]
          in
          equal (list string)
            [ "legend.color.cat"; "legend.color.num" ]
            (legends (lay (Size.panels 80. 60.) f)));
    ]

(* Marks *)

let image_group =
  group "image"
    [
      test "an image has square pixels" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 3; 6 |] in
          let b = List.hd (boxes (lay (Size.figure 300. 300.) (image px))) in
          equal close 0.5 (Box2.h b /. Box2.w b));
    ]

(* Guides *)

let classes = [| "cat"; "dog"; "cat"; "bird" |]

(* Figures whose categorical colours a shown axis may name, and the legends each
   has. *)
let named_by_axis =
  let bars ?fill ?(x = strings classes) () =
    rect ~x ~y:(num ramp)
      ~fill:(Option.value fill ~default:(strings classes))
      ()
  in
  let kept = [ "legend.color.cat" ] in
  [
    ("colours of the x data", bars (), []);
    ( "colours of the y data",
      rect ~x:(num ramp) ~y:(strings classes) ~fill:(strings classes) (),
      [] );
    ( "facets labelling their axis on the outer panels",
      rect ~x:(strings classes) ~y:(num ramp) ~fill:(strings classes)
        ~fy:(strings [| "p"; "q"; "p"; "q" |])
        (),
      [] );
    ( "colours of other data",
      bars ~fill:(strings [| "a"; "b"; "a"; "b" |]) (),
      kept );
    ("a hidden axis", layer [ bars (); axis ~show:false "x" ], kept);
    ("an explicit legend", layer [ bars (); legend "color" ], kept);
    ( "a mark implying a legend",
      Mark.v ~name:"bars"
        [
          Mark.bind Role.x (strings classes);
          Mark.bind ~guide:true Role.fill (strings classes);
        ]
        (fun _ -> Picture.empty),
      kept );
    ( "a second reader the axis does not name",
      layer
        [
          bars ();
          dot
            ~x:(strings [| "a"; "b"; "a"; "b" |])
            ~y:(num ramp) ~fill:(strings classes) ();
        ],
      kept );
    ( "quantities of the x data",
      plain ~fill:(num ramp) ramp,
      [ "legend.color.num" ] );
  ]

let pp_areas ppf (a0, a1) = Format.fprintf ppf "(%g, %g)" a0 a1

(* Size legends: the areas of a scale, from a pool with an end of area 0, data
   from [lo] to [lo + span], and whether that span is the scale's domain. *)
let gen_sizes =
  Gen.triple
    (Gen.of_list ~pp:pp_areas [ (0., 100.); (0., 400.); (64., 0.); (9., 200.) ])
    (Gen.pair (Gen.int_range (-50) 50) (Gen.int_range 1 200))
    Gen.bool

(* [size_channel (areas, (lo, span), set)] is the data and the size channel of a
   size legend case. *)
let size_channel (areas, (lo, span), set) =
  let lo = Float.of_int lo and hi = Float.of_int (lo + span) in
  let domain = if set then Some (lo, hi) else None in
  let data = f64 [| lo; (lo +. hi) /. 2.; hi |] in
  (data, num ~scale:(Scale.linear ~areas ?domain ()) data)

(* [size_legend c] is the layout of a probe of the size legend of [c], and the
   box, area and normalised value of each of its entries. *)
let size_legend c =
  let data, size = size_channel c in
  let only o = (Option.get o).(0) in
  let l, entries =
    swatches (Size.panels 80. 60.)
      [
        Mark.bind Role.x (num data);
        Mark.bind Role.y (num data);
        Mark.bind Role.size size;
      ]
      (fun rows ->
        (only (Mark.get rows Role.size), only (Mark.normalized rows Role.size)))
  in
  (l, List.map (fun (box, (area, u)) -> (box, area, u)) entries)

let no_area_0 (((a0, a1), _, _) as c) =
  cover "an end of area 0" (a0 = 0. || a1 = 0.);
  let _, entries = size_legend c in
  greater int ~than:0 (List.length entries);
  List.iteri
    (fun k (_, area, _) ->
      greater ~msg:(Printf.sprintf "entry %d" k) float_exact ~than:0. area)
    entries

(* Swatches are 0.8 em square, or as large as the legend's largest circle
   (Theme, "Derived lengths"). *)
let holds_circles c =
  let _, entries = size_legend c in
  let largest =
    List.fold_left
      (fun m (_, area, _) -> Float.max m (2. *. Float.sqrt (area /. Float.pi)))
      0. entries
  in
  let side = Float.max (0.8 *. em) largest in
  equal
    (list (pair close close))
    (List.map (fun _ -> (side, side)) entries)
    (List.map (fun (b, _, _) -> (Box2.w b, Box2.h b)) entries)

(* The ticks [Ticks.choose] gives a guide nine labels long, with no least
   spacing, without those of area 0. *)
let size_ticks_law (((a0, a1), _, _) as c) =
  let data, size = size_channel c in
  let r = resolve (dot ~x:(num data) ~y:(num data) ~size ()) in
  let fitted = Resolved.scale r (Scale.linear ~name:"size" ()) in
  let ticks = Ticks.choose ~length:1. ~measure:(fun _ -> 1. /. 9.) fitted in
  let positive (t : Ticks.tick) = a0 +. (t.position *. (a1 -. a0)) > 0. in
  let kept = List.filter positive ticks.major in
  cover "a tick of area 0 left out" (List.length kept < List.length ticks.major);
  let _, entries = size_legend c in
  equal (list float_exact)
    (List.map (fun (t : Ticks.tick) -> t.position) kept)
    (List.map (fun (_, _, u) -> u) entries)

(* [loss_y ~side ()] is dots with a y title over labels of growing width. *)
let loss_y ?side () =
  layer
    (dot ~x:(num ramp)
       ~y:(num ~title:"loss" (f64 [| 0.; 10.; 100.; 1000. |]))
       ()
    :: Option.to_list (Option.map (fun side -> axis ~side "y") side))

(* A y title on the right, beside a legend on that side. *)
let beside_legend ?(side = `Right) () =
  layer
    [
      dot ~x:(num ramp) ~y:(num ~title:"loss" ramp)
        ~fill:(strings ~title:"optimiser" [| "a"; "b"; "a"; "b" |])
        ();
      axis ~side "y";
      legend ~side "color";
    ]

(* Facets headed by the categories and titles of their rows, and of their
   columns on the left. *)
let row_headed =
  dot ~x:(num ramp) ~y:(num ramp)
    ~fy:(strings ~title:"layer" [| "p"; "p"; "q"; "q" |])
    ~fx:(strings ~title:"head" [| "a"; "b"; "a"; "b" |])
    ()

let left_headed =
  layer
    [
      dot ~x:(num ramp) ~y:(num ramp)
        ~fx:(strings ~title:"head" [| "a"; "b"; "a"; "b" |])
        ();
      axis ~side:`Left ~title:"h" "fx";
    ]

let shared_time =
  let cell = dot ~x:(num ~title:"time" ramp) ~y:(num ramp) () in
  grid [ [ cell ]; [ cell ] ] |> share [ ("x", `Shared) ]

let quantities_labels (notation, ticks) =
  let scale = Scale.linear ~name:"x" ~ticks ~notation () in
  let unit = f64 [| 0.; 1. |] in
  let r = resolve (dot ~x:(num ~scale unit) ~y:(num unit) ()) in
  let expected = Ticks.of_values ~notation (Resolved.scale r scale) ticks in
  equal (list string)
    (List.map (fun (t : Ticks.tick) -> t.label) expected.major)
    (words (labels (layout (Size.panels 300. 60.) r) "axis.x"))

(* [clear_along (lo, hi) boxes] states that each two adjacent [boxes] stand at
   least half an em apart between the ends [lo] and [hi]. *)
let clear_along (lo, hi) boxes =
  let rec go = function
    | a :: (b :: _ as rest) ->
        at_least float_exact ~than:((0.5 *. em) -. 1e-9) (lo b -. hi a);
        go rest
    | _ -> ()
  in
  go (List.sort (fun a b -> Float.compare (lo a) (lo b)) boxes)

let across_x = (Box2.minx, Box2.maxx)
let down_y = (Box2.miny, Box2.maxy)

(* A heatmap of [n] by [n] categories named from a pool that holds an empty
   name, a long one and repeats, on a band scale with [reverse] and [padding],
   with or without an aspect, laid out at panels of a size or at a figure too
   small for it and then at the size it names. *)
type named_case = {
  n : int;
  offset : int;
  reverse : bool;
  padding : float;
  aspect : bool;
  size : [ `Panels of float * float | `Figure of float * float ];
}

let pool = [| "the"; "cat"; ""; "a much longer name"; "on"; "the"; "x" |]

let gen_named =
  let size =
    Gen.one_of
      [
        Gen.map
          (fun (w, h) -> `Panels (w, h))
          (Gen.pair (Gen.float_range 5. 200.) (Gen.float_range 5. 200.));
        Gen.map
          (fun (w, h) -> `Figure (w, h))
          (Gen.pair (Gen.float_range 30. 80.) (Gen.float_range 30. 80.));
      ]
  in
  Gen.map
    (fun ((n, offset, reverse), (padding, aspect, size)) ->
      { n; offset; reverse; padding; aspect; size })
    (Gen.pair
       (Gen.triple (Gen.int_range 1 8) (Gen.int_range 0 6) Gen.bool)
       (Gen.triple (Gen.of_list [ 0.; 0.2; 1. ]) Gen.bool size))
  |> Gen.with_pp (fun ppf c ->
      Format.fprintf ppf
        "%d names from %d, reverse %b, padding %g, aspect %b, %s" c.n c.offset
        c.reverse c.padding c.aspect
        (match c.size with
        | `Panels (w, h) -> Printf.sprintf "panels %g x %g" w h
        | `Figure (w, h) -> Printf.sprintf "figure %g x %g" w h))

let named_law c =
  let names =
    Array.init c.n (fun i -> pool.((i + c.offset) mod Array.length pool))
  in
  let scale () = Scale.band ~reverse:c.reverse ~padding:c.padding () in
  let m =
    Nx.init Nx.float64 [| c.n; c.n |] (fun i -> Float.of_int (i.(0) + i.(1)))
  in
  let f =
    rect
      ~y:(dim ~scale:(scale ()) ~labels:names 0)
      ~x:(dim ~scale:(scale ()) ~labels:names 1)
      ~fill:(num m) ()
  in
  let f = if c.aspect then coord (Coord.cartesian ~aspect:1. ()) f else f in
  let named =
    match c.size with
    | `Figure (w, h) -> needs (Size.figure w h) f
    | `Panels _ -> None
  in
  cover "one category" (c.n = 1);
  cover "an aspect" c.aspect;
  cover "a figure too small" (Option.is_some named);
  let l =
    match c.size with
    | `Panels (w, h) ->
        let l = lay (Size.panels w h) f in
        if not c.aspect then begin
          let b = List.hd (boxes l) in
          (* Grown, never shrunk, to the rounding of the page. *)
          at_least float_exact ~than:(w -. 1e-9) (Box2.w b);
          at_least float_exact ~than:(h -. 1e-9) (Box2.h b)
        end;
        l
    | `Figure (w, h) ->
        let w, h = Option.value named ~default:(w, h) in
        lay (Size.figure w h) f
  in
  (* An empty name is labelled by a text with no glyph. *)
  let shown = List.filter (( <> ) "") (Array.to_list names) in
  let xs = labels l "axis.x" and ys = labels l "axis.y" in
  equal (slist string String.compare) shown (words xs);
  equal (slist string String.compare) shown (words ys);
  clear_along across_x (List.map set_box xs);
  clear_along down_y (List.map set_box ys)

let one_name =
  {
    n = 1;
    offset = 0;
    reverse = false;
    padding = 0.;
    aspect = false;
    size = `Panels (50., 50.);
  }

let guides_group =
  group "Guides"
    [
      cases "a categorical legend that a shown axis names is left out:"
        ~id:(fun (n, _, _) -> n)
        named_by_axis
        (fun (_, f, expected) ->
          equal (list string) expected (legends (lay (Size.panels 80. 60.) f)));
      prop "a size legend draws no entry of area 0" gen_sizes no_area_0;
      prop "a size legend's swatches are as large as its largest circle"
        gen_sizes holds_circles;
      prop "a size legend's entries are the ticks of a guide nine labels long"
        ~examples:[ ((0., 100.), (0, 100), true) ]
        gen_sizes size_ticks_law;
      test "a guide is titled by its axes' distinct titles, with commas"
        (fun () ->
          let cell t = layer [ plain ramp; axis ~title:t "x" ] in
          let f =
            grid [ [ cell "a" ]; [ cell "b" ]; [ cell "a" ] ]
            |> share [ ("x", `Shared) ]
          in
          equal (list string) [ "a, b" ] (titles (lay (Size.panels 60. 40.) f)));
      test "a guide is titled by its axes' titles over its channels'" (fun () ->
          let timed guides =
            layer (dot ~x:(num ~title:"time" ramp) ~y:(num ramp) () :: guides)
          in
          let f =
            grid [ [ timed [] ]; [ timed [ axis ~title:"t (s)" "x" ] ] ]
            |> share [ ("x", `Shared) ]
          in
          equal (list string) [ "t (s)" ] (titles (lay (Size.panels 60. 40.) f)));
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
          let legend = ink l "0.legend.color.cat" in
          match boxes l with
          | [ a; b ] ->
              greater float_exact ~than:(Box2.maxx a) (Box2.minx legend);
              less float_exact ~than:(Box2.minx b) (Box2.maxx legend)
          | bs -> failf "%d panels" (List.length bs));
      test "a shared axis is labelled on the outer panels only" (fun () ->
          let l = lay (Size.panels 60. 40.) shared_time in
          let labelled =
            List.filter
              (fun t -> is_label t && String.ends_with ~suffix:"axis.x" t.id)
              (texts l)
          in
          equal (list string) [ "1.axis.x" ]
            (List.sort_uniq String.compare (List.map (fun t -> t.id) labelled)));
      test "a shared axis is titled once" (fun () ->
          equal (list string) [ "time" ]
            (titles (lay (Size.panels 60. 40.) shared_time)));
      test "an x title is centred below the panels it titles" (fun () ->
          let f =
            dot ~x:(num ~title:"time" ramp) ~y:(num ramp)
              ~fx:(strings [| "a"; "b"; "c"; "d" |])
              ()
          in
          let l = lay (Size.panels 40. 30.) f in
          let hull = List.fold_left Box2.union (List.hd (boxes l)) (boxes l) in
          let b = set_box (text l "time") in
          equal close (P2.x (Box2.mid hull)) (P2.x (Box2.mid b));
          greater float_exact ~than:(Box2.maxy hull) (Box2.miny b));
      test "a y title stands half an em above its tick labels" (fun () ->
          let l = lay (Size.panels 80. 60.) (loss_y ()) in
          let top =
            List.fold_left
              (fun m t -> Float.min m (Box2.miny (set_box t)))
              infinity (labels l "axis.y")
          in
          equal close title_gap (top -. Box2.maxy (set_box (text l "loss"))));
      cases "a y title is left-aligned with its tick labels on the" ~id:fst
        [ ("left", `Left); ("right", `Right) ]
        (fun (_, side) ->
          let l = lay (Size.panels 80. 60.) (loss_y ~side ()) in
          let left =
            List.fold_left
              (fun m t -> Float.min m (Box2.minx (set_box t)))
              infinity (labels l "axis.y")
          in
          equal close left (Box2.minx (set_box (text l "loss"))));
      test "beside a legend, a y title stands half an em beyond its labels"
        (fun () ->
          let l = lay (Size.panels 80. 60.) (beside_legend ()) in
          let right =
            List.fold_left
              (fun m t -> Float.max m (Box2.maxx (set_box t)))
              neg_infinity (labels l "axis.y")
          in
          equal close title_gap (Box2.minx (set_box (text l "loss")) -. right));
      test "beside a legend, a y title is centred along its axis" (fun () ->
          let l = lay (Size.panels 80. 60.) (beside_legend ()) in
          equal close
            (P2.y (Box2.mid (List.hd (boxes l))))
            (P2.y (Box2.mid (set_box (text l "loss")))));
      cases "a y title shares no line with a legend's title on the" ~id:fst
        [ ("left", `Left); ("right", `Right) ]
        (fun (_, side) ->
          let l = lay (Size.panels 80. 60.) (beside_legend ~side ()) in
          greater float_exact
            ~msg:"the y title's top, below the legend title's bottom"
            ~than:(Box2.maxy (set_box (text l "optimiser")))
            (Box2.miny (set_box (text l "loss"))));
      test "no text is turned: headers beside a row read upright" (fun () ->
          let f =
            dot ~x:(num ramp) ~y:(num ramp)
              ~fy:(strings [| "upright"; "upright"; "row"; "row" |])
              ()
          in
          let ink = (text (lay (Size.panels 40. 30.) f) "upright").ink in
          greater float_exact ~than:(Box2.h ink) (Box2.w ink));
      test "an fx header is centred above its column" (fun () ->
          let f =
            dot ~x:(num ramp) ~y:(num ramp)
              ~fx:(strings [| "a"; "b"; "a"; "b" |])
              ()
          in
          let l = lay (Size.panels 40. 30.) f in
          let column = panel_box l "panel.b" and b = set_box (text l "b") in
          equal close (P2.x (Box2.mid column)) (P2.x (Box2.mid b));
          less float_exact ~than:(Box2.miny column) (Box2.maxy b));
      test "the title of headers above them stands half an em above them"
        (fun () ->
          let fx = strings ~title:"model" [| "a"; "b"; "a"; "b" |] in
          let l =
            lay (Size.panels 50. 40.) (dot ~x:(num ramp) ~y:(num ramp) ~fx ())
          in
          equal close title_gap
            (Box2.miny (set_box (text l "a"))
            -. Box2.maxy (set_box (text l "model"))));
      cases
        "a header on the left or right reads its scale's title before its \
         category:"
        ~id:(fun (n, _, _) -> n)
        [
          ("rows", row_headed, [ "a"; "b"; "layer p"; "layer q" ]);
          ("columns on the left", left_headed, [ "h a"; "h b" ]);
        ]
        (fun (_, f, expected) ->
          equal
            (slist string String.compare)
            expected
            (headers (lay (Size.panels 40. 30.) f)));
      cases "a title read by headers on the left or right stands nowhere else:"
        ~id:(fun (n, _, _) -> n)
        [
          ("rows", row_headed, [ "head" ]);
          ("columns on the left", left_headed, []);
        ]
        (fun (_, f, expected) ->
          equal (list string) expected
            (guide_titles (lay (Size.panels 40. 30.) f)));
      test "a header on the right with an empty title reads its category"
        (fun () ->
          let f =
            layer
              [
                dot ~x:(num ramp) ~y:(num ramp)
                  ~fy:(strings ~title:"layer" [| "p"; "p"; "q"; "q" |])
                  ();
                axis ~title:"" "fy";
              ]
          in
          equal
            (slist string String.compare)
            [ "p"; "q" ]
            (headers (lay (Size.panels 40. 30.) f)));
      test "a colour bar is as long as its panel" (fun () ->
          let l = lay (Size.panels 80. 60.) (plain ~fill:(num ramp) ramp) in
          match keys l with
          | [ bar ] -> equal close (Box2.h (List.hd (boxes l))) (Box2.h bar)
          | ks -> failf "%d keys" (List.length ks));
      test
        "a scale read by x in one cell and by fill in another has an axis \
         there and a legend" (fun () ->
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
          equal (list string)
            [
              "0.axis.rate";
              "0.axis.y";
              "1.axis.x";
              "1.axis.y";
              "1.legend.rate.num";
            ]
            (guides (lay (Size.panels 80. 60.) f)));
      test "explicit ticks of a band scale limit its axis to the categories"
        (fun () ->
          let scale = Scale.band ~ticks:[| "c"; "z"; "a"; "c" |] () in
          let f =
            rect
              ~x:(strings ~scale [| "a"; "b"; "c" |])
              ~y:(num (f64 [| 1.; 2.; 3. |]))
              ()
          in
          equal (list string) [ "a"; "c" ]
            (words (labels (lay (Size.panels 100. 60.) f) "axis.x")));
      test "facet headers name every panel whatever the ticks" (fun () ->
          let scale = Scale.band ~ticks:[| "a" |] () in
          let f =
            dot ~x:(num ramp) ~y:(num ramp)
              ~fx:(strings ~scale [| "a"; "b"; "c"; "a" |])
              ()
          in
          equal
            (slist string String.compare)
            [ "a"; "b"; "c" ]
            (headers (lay (Size.panels 40. 30.) f)));
      prop "an axis of names labels every name clear of the next, at any size"
        ~examples:[ one_name ] gen_named named_law;
      test "explicit ticks of an axis of names are each labelled, clear apart"
        (fun () ->
          let names = Array.init 12 (fun i -> Printf.sprintf "name %d" i) in
          let ticks = Array.sub names 0 6 in
          let l =
            lay (Size.panels 20. 20.)
              (dot
                 ~x:(strings ~scale:(Scale.band ~ticks ()) names)
                 ~y:(num (f64 (Array.init 12 Float.of_int)))
                 ())
          in
          let xs = labels l "axis.x" in
          equal (list string) (Array.to_list ticks) (words xs);
          clear_along across_x (List.map set_box xs));
      test "names given to a band scale are each labelled, clear apart"
        (fun () ->
          let names = Array.init 12 (fun i -> Printf.sprintf "name %d" i) in
          let l =
            lay (Size.panels 30. 30.)
              (dot ~x:(strings names)
                 ~y:(num (f64 (Array.init 12 Float.of_int)))
                 ())
          in
          let xs = labels l "axis.x" in
          equal (slist string String.compare) (Array.to_list names) (words xs);
          clear_along across_x (List.map set_box xs));
      test "indices in decimal are labelled at a stride" (fun () ->
          let n = 30 in
          let m =
            Nx.init Nx.float64 [| n; n |] (fun i ->
                Float.of_int (i.(0) + i.(1)))
          in
          let f = rect ~y:(dim 0) ~x:(dim 1) ~fill:(num m) () in
          let k = List.length (labels (lay (Size.panels 40. 40.) f) "axis.x") in
          less int ~than:n k;
          greater int ~than:1 k);
      cases "explicit ticks are labelled as of_values labels them, in" ~id:fst
        [
          ("percent", (Number.Percent, [| 1.; 0.; 0.25; 2.; Float.nan |]));
          ("SI", (Number.Si, [| 0.5; 0.5; 1. |]));
        ]
        (fun (_, c) -> quantities_labels c);
      test "chosen ticks are labelled in the scale's notation" (fun () ->
          let scale = Scale.linear ~notation:Percent () in
          let data = f64 [| 0.; 0.37; 1. |] in
          let l =
            lay (Size.panels 200. 60.)
              (dot ~x:(num ~scale data) ~y:(num data) ())
          in
          let xs = words (labels l "axis.x") in
          greater int ~than:1 (List.length xs);
          List.iter (fun s -> ends_with ~suffix:"%" s) xs);
    ]

(* Facets *)

(* [order l keep key] is the categories of the panels of [l] whose id segments
   [keep] picks, by [key] of their boxes. *)
let order l keep key =
  Layout.panels l
  |> List.filter_map (fun (p : Layout.panel) ->
      Option.map
        (fun c -> (key p.box, c))
        (keep (segments (Nx.Ptree.Path.to_string p.id))))
  |> List.sort compare |> List.map snd

(* Panels [panel.fy.fx] of fx categories c, a, b and fy categories q, p, in
   order of first appearance. *)
let faceted =
  dot ~x:(num ramp) ~y:(num ramp)
    ~fx:(strings [| "c"; "a"; "b"; "c" |])
    ~fy:(strings [| "q"; "p"; "q"; "p" |])
    ()

let facets_group =
  group "Facets"
    [
      test "fx categories are columns in domain order from the left" (fun () ->
          let l = lay (Size.panels 40. 30.) faceted in
          let in_row_q = function [ "panel"; "q"; c ] -> Some c | _ -> None in
          equal (list string) [ "c"; "a"; "b" ] (order l in_row_q Box2.minx));
      test "fy categories are rows in domain order from the top" (fun () ->
          let l = lay (Size.panels 40. 30.) faceted in
          let in_column_c = function
            | [ "panel"; r; "c" ] -> Some r
            | _ -> None
          in
          equal (list string) [ "q"; "p" ] (order l in_column_c Box2.miny));
      test "headers name each column and each row once" (fun () ->
          equal
            (slist string String.compare)
            [ "a"; "b"; "c"; "p"; "q" ]
            (headers (lay (Size.panels 40. 30.) faceted)));
      test "a wrap of 3 sets 5 columns in rows of 3 and 2" (fun () ->
          let cats = [| "a"; "b"; "c"; "d"; "e" |] in
          let x = f64 (Array.init 5 Float.of_int) in
          let fx = strings ~scale:(Scale.band ~wrap:3 ()) cats in
          let l =
            lay (Size.panels 40. 30.) (dot ~x:(num x) ~y:(num x) ~fx ())
          in
          let at c = panel_box l ("panel." ^ c) in
          let corner c = (Box2.minx (at c), Box2.miny (at c)) in
          let x0 = Box2.minx (at "a") and x1 = Box2.minx (at "b") in
          let y0 = Box2.miny (at "a") and y1 = Box2.miny (at "d") in
          greater float_exact ~than:y0 y1;
          equal
            (list (pair close close))
            [ (x0, y0); (x1, y0); (Box2.minx (at "c"), y0); (x0, y1); (x1, y1) ]
            (List.map corner (Array.to_list cats)));
    ]

(* Coordinate systems *)

let projected () =
  List.hd (Layout.panels (lay (Size.figure 200. 100.) (plain ramp)))

let coord_group =
  let cells =
    rect
      ~x:(strings [| "a"; "b"; "c"; "a"; "b"; "c" |])
      ~y:(strings [| "p"; "p"; "p"; "q"; "q"; "q" |])
      ()
    |> coord (Coord.cartesian ~aspect:1. ())
  in
  let ratio b = Box2.h b /. Box2.w b in
  group "Coord"
    [
      cases "an aspect of one makes band cells square, at" ~id:fst
        [
          ("panels 90 × 90", Size.panels 90. 90.);
          ("a figure 300 × 400", Size.figure 300. 400.);
        ]
        (fun (_, size) ->
          equal close (2. /. 3.) (ratio (List.hd (boxes (lay size cells)))));
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
          | bs -> failf "%d panels" (List.length bs));
      test "a cartesian projection puts the unit square on the data area"
        (fun () ->
          let (p : Layout.panel) = projected () in
          let at x y = Coord.point p.projection x y in
          equal (pair close close)
            (Box2.minx p.box, Box2.maxy p.box)
            (P2.x (at 0. 0.), P2.y (at 0. 0.));
          equal (pair close close)
            (Box2.maxx p.box, Box2.miny p.box)
            (P2.x (at 1. 1.), P2.y (at 1. 1.)));
      test "a panel of no height has no inverse" (fun () ->
          let f = title "t" bare in
          let least = (2. *. margin) +. head_h "t" +. title_gap in
          let (p : Layout.panel) =
            List.hd (Layout.panels (lay (Size.figure 100. least) f))
          in
          equal float_exact 0. (Box2.h p.box);
          equal
            (option (pair float_exact float_exact))
            None
            (Coord.invert p.projection (P2.v 1. 1.)));
      (* Projecting and inverting round lengths below 1000 pt a few times, which
         errs by a few 1e-13 of a unit: [1e-9] allows it. *)
      prop "inverting a projection undoes it"
        (Gen.pair (Gen.float_range (-2.) 2.) (Gen.float_range (-2.) 2.))
        (fun (x, y) ->
          let (p : Layout.panel) = projected () in
          match Coord.invert p.projection (Coord.point p.projection x y) with
          | Some (x', y') -> equal (pair close close) (x, y) (x', y')
          | None -> fail "no position");
    ]

(* Composing *)

let layer_group =
  group "layer"
    [
      test "an empty layer is one panel" (fun () ->
          equal int 1
            (List.length (Layout.panels (lay (Size.panels 50. 50.) (layer [])))));
    ]

(* Law: data areas align across the rows and columns of a grid. *)
let spines c =
  cover "facets" c.facets;
  cover "aspect" c.aspect;
  cover "titled facets" (c.facets && Option.is_some c.titles);
  (* A cell's facet panels are a grid of their own, side by side. A panel with
     an aspect is centred in its cell, and the aspects of cells differ. *)
  let f = if c.facets then 2 else 1 in
  let bs = Array.of_list (boxes (laid c)) in
  equal int (c.rows * c.cols * f) (Array.length bs);
  (* Panel sizes are least data areas, and an aspect sets the height. *)
  (match c.size with
  | `Panels (w, h) ->
      Array.iter
        (fun b ->
          at_least ~msg:"width" float_exact ~than:(w -. 1e-6) (Box2.w b);
          if not c.aspect then
            at_least ~msg:"height" float_exact ~than:(h -. 1e-6) (Box2.h b))
        bs
  | `Figure _ | `Near _ -> ());
  let at r q j = bs.((((r * c.cols) + q) * f) + j) in
  (* The edges a facet grid places and the middles of panels with an aspect are
     sums of lengths, some huge, so they agree up to rounding. *)
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
        equal edge ~msg:"right" (Box2.maxx (hull 0 q)) (Box2.maxx (hull r q))
      end
    done
  done

let grid_group =
  group "grid"
    [
      prop "data areas align across rows and columns" ~count:60 gen_case spines;
      test "a nested grid's panels align with its neighbours" (fun () ->
          let inner = grid [ [ plain ramp ]; [ plain ramp ] ] in
          let l =
            lay (Size.figure 400. 300.) (grid [ [ inner; plain ramp ] ])
          in
          match boxes l with
          | [ a; b; c ] ->
              equal close (Box2.miny c) (Box2.miny a);
              equal close (Box2.maxy c) (Box2.maxy b)
          | bs -> failf "%d panels" (List.length bs));
      test "an empty grid lays out no panel" (fun () ->
          equal int 0
            (List.length (Layout.panels (lay (Size.panels 50. 50.) (grid [])))));
    ]

let span_group =
  group "span"
    [
      test "a span across columns covers them and the gap between them"
        (fun () ->
          let l =
            lay (Size.figure 400. 300.)
              (grid
                 [ [ span ~cols:2 (plain ramp) ]; [ plain ramp; plain ramp ] ])
          in
          match boxes l with
          | [ a; b; c ] ->
              equal close (Box2.minx b) (Box2.minx a);
              equal close (Box2.maxx c) (Box2.maxx a)
          | bs -> failf "%d panels" (List.length bs));
      test "a span across rows covers them and the gap between them" (fun () ->
          let l =
            lay (Size.figure 400. 300.)
              (grid
                 [ [ span ~rows:2 (plain ramp); plain ramp ]; [ plain ramp ] ])
          in
          match boxes l with
          | [ a; b; c ] ->
              equal close (Box2.miny b) (Box2.miny a);
              equal close (Box2.maxy c) (Box2.maxy a)
          | bs -> failf "%d panels" (List.length bs));
    ]

let title_group =
  group "title"
    [
      cases "a figure title aligns with the data areas it titles:" ~id:fst
        [
          ("left", (`Left, fun b -> Box2.minx b));
          ("centre", (`Center, fun b -> P2.x (Box2.mid b)));
          ("right", (`Right, fun b -> Box2.maxx b));
        ]
        (fun (_, (align, edge)) ->
          let l =
            lay (Size.panels 80. 60.)
              (plain ~y:(f64 [| 1.; 2e6; 3.; 4. |]) ramp |> title ~align "T")
          in
          equal close
            (edge (List.hd (boxes l)))
            (edge (set_box ~bold:true (text l "T"))));
      test "a figure title stands half an em above what it titles" (fun () ->
          let l = lay (Size.panels 50. 40.) (title "T" bare) in
          equal close title_gap
            (Box2.miny (List.hd (boxes l))
            -. Box2.maxy (set_box ~bold:true (text l "T"))));
      test "a figure title is set bold at 1.2 em" (fun () ->
          let l = lay (Size.panels 50. 40.) (title "Tg" (plain ramp)) in
          let set = setting ~bold:true ~size:(1.2 *. em) "Tg" in
          let first = Text.Layout.fold (fun acc _ _ r -> r :: acc) [] set in
          equal (list run_t) first [ (text l "Tg").run ]);
      test "titles nest, the outer above the inner" (fun () ->
          let l =
            lay (Size.panels 50. 40.)
              (title "outer" (title "inner" (plain ramp)))
          in
          at_most float_exact
            ~than:(Box2.miny (set_box ~bold:true (text l "inner")))
            (Box2.maxy (set_box ~bold:true (text l "outer"))));
    ]

(* Guides as figures *)

let timed guides =
  layer (dot ~x:(num ~title:"time" ramp) ~y:(num ramp) () :: guides)

let axis_group =
  group "axis"
    [
      test "an axis title replaces the titles of the channels" (fun () ->
          let l =
            lay (Size.panels 60. 40.) (timed [ axis ~title:"t (s)" "x" ])
          in
          equal (list string) [ "t (s)" ] (titles l));
      test "an axis' title is set in its styles" (fun () ->
          let t = Text.bold (Text.v "t") in
          let l = lay (Size.panels 60. 40.) (timed [ axis' ~title:t "x" ]) in
          let set = setting ~bold:true ~size:em "t" in
          let first = Text.Layout.fold (fun acc _ _ r -> r :: acc) [] set in
          equal (list run_t) first [ (text l "t").run ]);
    ]

(* Places of a legend inside a data area. *)
let places =
  [
    ("top left", `Top_left);
    ("top", `Top);
    ("top right", `Top_right);
    ("left", `Left);
    ("centre", `Center);
    ("right", `Right);
    ("bottom left", `Bottom_left);
    ("bottom", `Bottom);
    ("bottom right", `Bottom_right);
  ]

(* [across p] and [down p] are where the place [p] lies along x and y: at the
   start, the middle or the end. *)
let across p =
  match p with
  | `Top_left | `Left | `Bottom_left -> `Start
  | `Top | `Center | `Bottom -> `Middle
  | `Top_right | `Right | `Bottom_right -> `End

let down p =
  match p with
  | `Top_left | `Top | `Top_right -> `Start
  | `Left | `Center | `Right -> `Middle
  | `Bottom_left | `Bottom | `Bottom_right -> `End

(* [nearest n e] is the row or column of [n] nearest the end [e], the earlier of
   two as near. *)
let nearest n = function `Start -> 0 | `Middle -> (n - 1) / 2 | `End -> n - 1

(* [inside_facets rows cols p] is a grid of facets, of [rows] fy categories r0,
   r1… and [cols] fx categories c0, c1…, with a legend of eight colours inside
   at the place [p]. *)
let inside_facets rows cols p =
  let n = 8 in
  let data = f64 (Array.init n Float.of_int) in
  let cats prefix k =
    strings (Array.init n (fun i -> Printf.sprintf "%s%d" prefix (k i)))
  in
  layer
    [
      dot ~x:(num data) ~y:(num data) ~fill:(cats "kind " Fun.id)
        ~fx:(cats "c" (fun i -> i mod cols))
        ~fy:(cats "r" (fun i -> i / cols mod rows))
        ();
      legend ~side:(`Inside p) "color";
    ]

(* [inside_box p] is the box of a legend of three categories inside a panel at
   the place [p], the hull of its swatches and of the boxes its texts are set
   in, and the panel's data area. *)
let inside_box p =
  let kinds = strings ~title:"kind" [| "cat"; "dog"; "bird" |] in
  let l, entries =
    swatches
      ~guides:[ legend ~side:(`Inside p) "color" ]
      (Size.panels 200. 150.)
      [ Mark.bind Role.fill kinds ]
      ignore
  in
  let texts =
    List.filter (fun t -> t.id = "legend.color.cat") (texts l)
    |> List.map (fun t -> set_box t)
  in
  match List.map fst entries @ texts with
  | [] -> failf "no legend"
  | b :: bs -> (List.fold_left Box2.union b bs, List.hd (boxes l))

(* An end a place names lies a pad from that edge of the panel; the middle is
   centred between the two. *)
let inside_pad (_, p) =
  let legend, panel = inside_box p in
  let along name e (lo, hi) =
    let msg = name in
    match e with
    | `Start -> equal ~msg close (lo panel +. pad) (lo legend)
    | `End -> equal ~msg close (hi panel -. pad) (hi legend)
    | `Middle ->
        equal ~msg close
          ((lo panel +. hi panel) /. 2.)
          ((lo legend +. hi legend) /. 2.)
  in
  along "x" (across p) (Box2.minx, Box2.maxx);
  along "y" (down p) (Box2.miny, Box2.maxy)

let inside_places =
  List.map (fun place -> (2, 2, place)) places @ [ (1, 3, ("top", `Top)) ]

let inside_place (rows, cols, (_, p)) =
  let l = lay (Size.panels 60. 60.) (inside_facets rows cols p) in
  let id =
    Printf.sprintf "panel.r%d.c%d"
      (nearest rows (down p))
      (nearest cols (across p))
  in
  satisfies
    ~claim:("within " ^ id ^ ", a pad from its edges")
    box_t
    (within (Box2.grow (-.pad) (panel_box l id)))
    (ink l "legend.color.cat")

(* Figures whose colour legend may stand inside: categories and a bar. *)
let inside_figures =
  let data = f64 [| 1.; 2.; 3. |] in
  [
    ( "categories",
      dot ~x:(num data) ~y:(num data)
        ~fill:(strings ~title:"kind" [| "cat"; "dog"; "bird" |])
        () );
    ( "a colour bar",
      dot ~x:(num data) ~y:(num data) ~fill:(num ~title:"z" data) () );
  ]

let inside_sizes =
  [
    ("panels", Size.panels 200. 150.);
    ("figure", Size.figure 360. 240.);
    ("wide figure", Size.figure 600. 200.);
  ]

let unmoved_rows =
  List.concat_map
    (fun f ->
      List.concat_map
        (fun p -> List.map (fun s -> (f, p, s)) inside_sizes)
        places)
    inside_figures

let unmoved_law ((_, f), (_, p), (_, size)) =
  let with_legend g = boxes (lay size (layer [ f; g ])) in
  equal (list box_t)
    (with_legend (legend ~show:false "color"))
    (with_legend (legend ~side:(`Inside p) "color"))

(* Two marks of four categories each, with the legend [g]. *)
let crowded_with g =
  let many = Array.init 8 (Printf.sprintf "kind %d") in
  layer
    [
      dot ~x:(num ramp) ~y:(num ramp) ~fill:(strings (Array.sub many 0 4)) ();
      dot ~x:(num ramp) ~y:(num ramp) ~fill:(strings (Array.sub many 4 4)) ();
      g;
    ]

let crowded = crowded_with (legend ~side:(`Inside `Top_left) "color")

let legend_group =
  group "legend"
    [
      test "a legend title replaces the titles of the channels" (fun () ->
          let fill = strings ~title:"kind" [| "a"; "b"; "a"; "b" |] in
          let f = layer [ plain ~fill ramp; legend ~title:"group" "color" ] in
          equal (list string) [ "group" ] (titles (lay (Size.panels 60. 40.) f)));
      test "legend swatches are 0.8 em square" (fun () ->
          let bs = fill_swatches 4 in
          equal
            (list (pair close close))
            (List.init 4 (fun _ -> (0.8 *. em, 0.8 *. em)))
            (List.map (fun b -> (Box2.w b, Box2.h b)) bs));
      test "legend rows are 0.4 em apart" (fun () ->
          let bs = fill_swatches 4 in
          let rec gaps = function
            | a :: (b :: _ as rest) -> (Box2.miny b -. Box2.maxy a) :: gaps rest
            | _ -> []
          in
          equal (list close) (List.init 3 (fun _ -> 0.4 *. em)) (gaps bs));
      test "a legend of stroke colours has line swatches 1.5 em long" (fun () ->
          let cats = strings [| "a"; "b" |] in
          let _, entries =
            swatches (Size.panels 80. 60.) [ Mark.bind Role.stroke cats ] ignore
          in
          equal
            (list (pair close close))
            [ (1.5 *. em, 0.8 *. em); (1.5 *. em, 0.8 *. em) ]
            (List.map (fun (b, ()) -> (Box2.w b, Box2.h b)) entries));
      cases "a legend holds its title on the" ~id:fst
        (List.concat_map
           (fun (name, side) ->
             let t = "A long legend title" in
             let at fill = layer [ plain ~fill ramp; legend ~side "color" ] in
             [
               ( name ^ ", of categories",
                 at (strings ~title:t [| "a"; "b"; "a"; "b" |]) );
               (name ^ ", a colour bar", at (num ~title:t ramp));
             ])
           [
             ("top", `Top);
             ("bottom", `Bottom);
             ("left", `Left);
             ("right", `Right);
           ])
        (fun (_, f) -> apart (lay (Size.panels 30. 30.) f));
      test "entries below the panels wrap at their width" (fun () ->
          let n = 40 in
          let x = f64 (Array.init n Float.of_int) in
          let f =
            layer
              [
                dot ~x:(num x) ~y:(num x)
                  ~fill:(strings (Array.init n (Printf.sprintf "category %d")))
                  ();
                legend ~side:`Bottom "color";
              ]
          in
          let l = lay (Size.figure 360. 240.) f in
          let panel = List.hd (boxes l) in
          let legend = ink l "legend.color.cat" in
          at_least float_exact ~msg:"left"
            ~than:(Box2.minx panel -. 1e-9)
            (Box2.minx legend);
          at_most float_exact ~msg:"right"
            ~than:(Box2.maxx panel +. 1e-9)
            (Box2.maxx legend));
      cases "an inside legend lies in the panel at its place, in"
        ~id:(fun (r, c, (p, _)) -> Printf.sprintf "%d × %d facets, %s" r c p)
        inside_places inside_place;
      cases
        "an inside legend lies a pad from the edges its place names, centred \
         between the others:"
        ~id:fst places inside_pad;
      cases "an inside legend leaves the data areas as a hidden one does:"
        ~id:(fun ((f, _), (p, _), (s, _)) -> Printf.sprintf "%s, %s, %s" f p s)
        unmoved_rows unmoved_law;
      test "an inside legend has the id of a legend" (fun () ->
          let f =
            layer
              [
                snd (List.hd inside_figures);
                legend ~side:(`Inside `Bottom_left) "color";
              ]
          in
          equal (list string) [ "legend.color.cat" ]
            (legends (lay (Size.panels 80. 60.) f)));
      test "panels grow to hold an inside legend and its pads" (fun () ->
          let l = lay (Size.panels 40. 40.) crowded in
          satisfies ~claim:"within the panel, a pad from its edges" box_t
            (within (Box2.grow (-.pad) (List.hd (boxes l))))
            (ink l "legend.color.cat"));
      test "a figure too small for an inside legend names a size that holds it"
        (fun () ->
          let size = Size.figure 120. 100. in
          ignore (lay size (crowded_with (legend ~show:false "color")));
          match needs size crowded with
          | None -> fail "the figure lays out"
          | Some (w, h) ->
              equal
                (option (pair float_exact float_exact))
                None
                (needs (Size.figure w h) crowded));
      test "an inside colour bar is 8 em long" (fun () ->
          let f =
            layer
              [
                plain ~fill:(num ramp) ramp;
                legend ~side:(`Inside `Top_right) "color";
              ]
          in
          match keys (lay (Size.panels 200. 200.) f) with
          | [ bar ] -> equal close (8. *. em) (Box2.h bar)
          | ks -> failf "%d keys" (List.length ks));
    ]

(* Sizes and themes *)

let size_group =
  group "Size"
    [
      test "panels give each flexible track its weight's data area" (fun () ->
          let l =
            lay (Size.panels 60. 40.)
              (grid ~widths:[ 1.; 2. ] ~heights:[ 1.; 0.5 ]
                 [ [ bare; bare ]; [ bare; bare ] ])
          in
          equal
            (list (pair close close))
            [ (60., 40.); (120., 40.); (60., 20.); (120., 20.) ]
            (List.map (fun b -> (Box2.w b, Box2.h b)) (boxes l)));
      test "panels grow the page to hold longer labels" (fun () ->
          let at y =
            Layout.size
              (lay (Size.panels 100. 50.) (dot ~x:(num ramp) ~y:(strings y) ()))
          in
          greater float_exact
            ~than:(fst (at [| "a"; "b"; "a"; "b" |]))
            (fst (at [| "a"; "a long category"; "a"; "b" |])));
      test "panels keep their data areas whatever the labels" (fun () ->
          let y = strings [| "a"; "a long category"; "a"; "b" |] in
          let l = lay (Size.panels 100. 50.) (dot ~x:(num ramp) ~y ()) in
          equal close 100. (Box2.w (List.hd (boxes l))));
      test "a figure as tall as its decorations has panels of no height"
        (fun () ->
          let least = (2. *. margin) +. head_h "t" +. title_gap in
          let b =
            List.hd (boxes (lay (Size.figure 100. least) (title "t" bare)))
          in
          equal (pair close close)
            (100. -. (2. *. margin), 0.)
            (Box2.w b, Box2.h b));
      prop "a figure laid out at a figure size is that size" ~count:60
        (gen_case_of (Gen.one_of [ gen_figure; gen_near ]))
        (fun c ->
          cover "near its least size"
            (match c.size with
            | `Near _ -> true
            | `Figure _ | `Panels _ -> false);
          let l = laid c in
          equal
            (option (pair float_exact float_exact))
            (page_of c)
            (Some (Layout.size l)));
    ]

let homogeneous_figures =
  let fill = strings ~title:"kind" [| "a"; "b"; "a"; "b" |] in
  [
    ("dots", plain ramp);
    ( "a legend and a title",
      title "Load" (dot ~x:(num ramp) ~y:(num ramp) ~fill ()) );
    ( "facets and a colour bar",
      rect ~x:(num ramp) ~y:(num ramp) ~fill:(num ~title:"z" ramp)
        ~fx:(strings [| "p"; "q"; "p"; "q" |])
        () );
  ]

(* Scaling by [k] rounds each of the few dozen lengths a box sums once more, so
   the boxes agree to a few dozen ulps: [1e-9] relative allows it. *)
let homogeneity ((_, f), (k, (w, h))) =
  let w = Float.of_int w and h = Float.of_int h in
  let base = lay (Size.figure w h) f in
  let theme = Theme.v ~size:(k *. em) () in
  let scaled = lay ~theme (Size.figure (k *. w) (k *. h)) f in
  let rel = float_rel ~rel:1e-9 ~abs:0. in
  let corners b = [ Box2.minx b; Box2.miny b; Box2.maxx b; Box2.maxy b ] in
  let times b = List.map (fun v -> k *. v) (corners b) in
  equal ~msg:"data areas"
    (list (list rel))
    (List.map times (boxes base))
    (List.map corners (boxes scaled));
  let drawn l corners =
    List.map (fun t -> (Run.text t.run, corners t.ink)) (texts l)
  in
  equal ~msg:"texts"
    (list (pair string (list rel)))
    (drawn base times) (drawn scaled corners)

let theme_group =
  group "Theme"
    [
      test "a page has a half em margin" (fun () ->
          let l = lay (Size.panels 50. 40.) bare in
          let b = List.hd (boxes l) in
          equal (pair close close)
            (50. +. (2. *. margin), 40. +. (2. *. margin))
            (Layout.size l);
          equal (pair close close) (margin, margin) (Box2.minx b, Box2.miny b));
      test "a colour bar is 1 em wide" (fun () ->
          match
            keys (lay (Size.panels 80. 60.) (plain ~fill:(num ramp) ramp))
          with
          | [ bar ] -> equal close em (Box2.w bar)
          | ks -> failf "%d keys" (List.length ks));
      prop "a theme k times the size lays a figure k times larger out alike"
        (Gen.pair
           (Gen.of_list
              ~pp:(fun ppf (n, _) -> Format.pp_print_string ppf n)
              homogeneous_figures)
           (Gen.pair
              (Gen.of_list ~pp:Format.pp_print_float
                 [
                   Theme.size Theme.talk /. em;
                   Theme.size Theme.poster /. em;
                   0.5;
                   3.;
                 ])
              (Gen.pair (Gen.int_range 200 400) (Gen.int_range 150 300))))
        homogeneity;
    ]

(* Stages *)

let layout_module =
  group "Layout"
    [
      test "warnings hold those of resolving, then those of laying out"
        (fun () ->
          let f =
            layer
              [
                dot
                  ~x:(num ~scale:(Scale.log ()) (f64 [| 1.; -1.; 4. |]))
                  ~y:(num (f64 [| 1.; 2.; 3. |]))
                  ();
                plain
                  ~fill:(strings [| "ok"; "b" ^ unmapped; "ok"; "ok" |])
                  ramp;
              ]
          in
          let r = resolve f in
          let rw = Resolved.warnings r in
          let lw = Layout.warnings (layout (Size.panels 80. 60.) r) in
          equal (list string) [ "0" ] (ids rw);
          equal (list warning_t) rw
            (List.filteri (fun i _ -> i < List.length rw) lw);
          equal (list string) [ "0"; "legend.color.cat" ] (ids lw));
      test "equal tells layouts in two themes apart" (fun () ->
          let r = resolve (plain ramp) and size = Size.panels 80. 60. in
          let theme = Theme.v ~size:8. () in
          not_equal layout_t (layout size r) (layout ~theme size r));
      test "pp formats the benchmark figures' layouts as their baseline"
        (fun () ->
          let open Hugin_test_figures in
          let one (name, f, theme, size) =
            Format.asprintf "%s@\n%a@\n" name Layout.pp
              (layout ~theme size (resolve (f ())))
          in
          expect_file ~file:"packages/hugin/test/golden/layout.expected"
            (String.concat "\n" (List.map one Figures.goldens)));
    ]

(* Law: a figure is laid out apart, the invariant [apart] states. *)
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
        size = `Panels (20., 20.);
      };
      {
        base with
        rows = 1;
        cols = 3;
        exps = [ (0, 0); (0, 3); (0, 0) ];
        facets = true;
        shared = true;
        size = `Panels (134.474, 20.);
      };
      {
        base with
        rows = 2;
        cols = 1;
        exps = [ (0, 0); (0, 6) ];
        colour = Quantities;
        size = `Figure (300., 250.);
      };
      {
        base with
        rows = 1;
        cols = 3;
        exps = [ (0, 0); (0, 0); (-1, -2) ];
        size = `Figure (300., 250.);
      };
      (* Legend entries taller than the panel with an aspect they stand
         beside. *)
      {
        base with
        colour = Categories;
        aspect = true;
        top = true;
        size = `Panels (20., 20.);
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
        size = `Figure (300., 300.609);
      };
      {
        base with
        rows = 3;
        cols = 1;
        exps = [ (0, 6); (0, 0); (0, 6) ];
        facets = true;
        aspect = true;
        titles = Some ("value", "value", "A figure");
        size = `Figure (300., 364.662);
      };
      (* A scale title wider than the panels. *)
      {
        base with
        cols = 2;
        titles = Some ("a very very long title for a scale here", "a", "a");
        size = `Panels (20., 20.);
      };
    ]
  in
  prop "no text overlaps another, a data area or the page's edge" ~count:150
    ~examples gen_case (fun c ->
      cover "titled facets" (c.facets && Option.is_some c.titles);
      cover "legend" (c.colour <> No_colour);
      cover "near its least size"
        (match c.size with `Near _ -> true | `Figure _ | `Panels _ -> false);
      let l = laid c in
      let ts = texts l in
      cover "titles part" (titles_part ts);
      cover "a title wider than the data areas" (wider l ts);
      apart l)

(* A facet grid with titles on y, fx and fy. *)
let titled_facets =
  dot ~x:(num ramp) ~y:(num ~title:"y" ramp)
    ~fx:(strings ~title:"head" [| "a"; "b"; "a"; "b" |])
    ~fy:(strings ~title:"layer" [| "p"; "p"; "q"; "q" |])
    ()

(* Figures whose guides and titles crowd one another or the page. *)
let crowded_figures =
  [
    ("a facet grid with y, fx and fy titles", titled_facets, Size.panels 80. 60.);
    ( "a long y title over a facet title",
      dot ~x:(num ramp)
        ~y:(num ~title:"a y title long enough to reach the middle" ramp)
        ~fx:(strings ~title:"head" [| "a"; "b"; "a"; "b" |])
        (),
      Size.panels 60. 40. );
    ( "a centred title beside a left legend",
      layer
        [
          plain ~fill:(strings [| "a"; "b"; "a"; "b" |]) ramp;
          legend ~side:`Left "color";
        ]
      |> title ~align:`Center "A centred title",
      Size.panels 22. 79. );
  ]
  @ List.map
      (fun h ->
        ( Printf.sprintf "a nested aspect panel under a wider title, %g pt high"
            h,
          grid [ [ square ] ] |> title ~align:`Left "A rather wide title",
          Size.panels 40. h ))
      [ 45.; 50.; 60.; 61. ]

(* The reach of a long y title wider than its panel. *)
let long_title =
  dot
    ~x:(num (f64 [| 0.; 1.; 3.; 4. |]))
    ~y:
      (num ~title:"a very very long title for a scale here"
         (f64 [| 0.; 1.; 2.; 4. |]))
    ()

(* Law: [c] laid out at [k] times the least size it names at one point, if too
   small for that, names a size that lays it out. *)
let names_law (c, k) =
  let f = figure_of c in
  match needs (Size.figure 1. 1.) f with
  | None -> fail "a figure of one point lays out"
  | Some (w, h) ->
      let size = Size.figure (k *. w) (k *. h) in
      let named = needs size f in
      cover "too small at that factor" (Option.is_some named);
      if Option.is_some named then lays_out_named size f

(* Figures too small at a factor of their least size, whose named size the law
   once found too small itself. *)
let near_least =
  [
    ( {
        base with
        cols = 2;
        exps = [ (0, -4); (7, 5) ];
        colour = Categories;
        shared = true;
        aspect = true;
        top = true;
        titles = Some ("two\nlines", "a", "a");
      },
      0.9788345544751399 );
  ]

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
      cover "headers of different widths"
        (Array.exists (fun k -> k <> ws.(0)) ws);
      let l = lay (Size.figure w 100.) (headed ws) in
      let widths = List.map Box2.w (boxes l) in
      let share =
        List.fold_left ( +. ) 0. widths /. Float.of_int (List.length widths)
      in
      let widest =
        Array.fold_left (fun m s -> Float.max m (label_w s)) 0. (names ws)
      in
      assume (widest < share);
      List.iter (fun w -> equal ~msg:"width" close share w) widths)

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
          equal close ~msg:"pinned" need.(0) (Box2.w first);
          List.iter (fun b -> equal close ~msg:"shared" rest (Box2.w b)) others)

(* [named_y t] is dots over named y categories, x hidden, y titled [t] if
   given. *)
let named_y t =
  layer
    [
      dot ~x:(num (f64 [| 1.; 2. |])) ~y:(strings ?title:t [| "a"; "bb" |]) ();
      axis ~show:false "x";
    ]

let left_edge t =
  Box2.minx (List.hd (boxes (lay (Size.panels 80. 80.) (named_y t))))

(* Thinning: explicit ticks of an axis 120 points long, labelled as
   [Ticks.of_values] labels them, of which those [thin] keeps are drawn. Two
   labels overlap iff their centres are closer than half their widths and the
   clearance. *)
let thinning_law vs =
  let scale = Scale.linear ~name:"x" ~domain:(0., 10.) ~ticks:vs () in
  let unit = f64 [| 0.; 10. |] in
  let r = resolve (dot ~x:(num ~scale unit) ~y:(num unit) ()) in
  let all =
    (Ticks.of_values (Resolved.scale r scale) vs).major
    |> List.map (fun (t : Ticks.tick) ->
        (t.label, 120. *. t.position, label_w t.label))
  in
  let drawn = words (labels (layout (Size.panels 120. 60.) r) "axis.x") in
  let reach (_, _, wa) (_, _, wb) = ((wa +. wb) /. 2.) +. (0.5 *. em) in
  let overlap ((_, a, _) as x) ((_, b, _) as y) =
    Float.abs (b -. a) < reach x y
  in
  let shown (l, _, _) = List.mem l drawn in
  let kept = List.filter shown all
  and dropped = List.filter (Fun.negate shown) all in
  cover "a dropped label" (dropped <> []);
  let rec clears = function
    | ((la, a, _) as x) :: (((lb, b, _) as y) :: _ as rest) ->
        at_least
          ~msg:(la ^ " to " ^ lb)
          float_exact ~than:(reach x y)
          (Float.abs (b -. a));
        clears rest
    | _ -> ()
  in
  clears kept;
  List.iter
    (fun d ->
      satisfies
        ~claim:"within half the widths and the clearance of a kept label"
        (triple string float_exact float_exact)
        (fun d -> List.exists (overlap d) kept)
        d)
    dropped

let gen_tick_values =
  Gen.array ~size:(Gen.int_range 1 12)
    (Gen.frequency
       [
         (3, Gen.float_range 0. 10.);
         (2, Gen.map (fun i -> Float.of_int i /. 10.) (Gen.int_range 0 100));
       ])

let reused =
  [
    ("the same figure", plain ramp, plain ramp, None);
    ("other data", plain ramp, plain (f64 [| 1.; 2e6 |]), None);
    ("another theme", plain ramp, plain ramp, Some 8.);
    ( "another arrangement",
      grid [ [ plain ramp; plain ramp ] ],
      grid [ [ plain ramp ]; [ plain ramp ] ],
      None );
  ]

let layout_group =
  group "layout"
    [
      no_overlap;
      cases "no text overlaps another, a data area or the page's edge, in"
        ~id:(fun (n, _, _) -> n)
        crowded_figures
        (fun (_, f, size) -> apart (lay size f));
      prop
        "a figure too small at a factor of its least size names a size that \
         lays it out"
        ~count:40 ~examples:near_least
        (Gen.pair gen_case (Gen.float_range 0.3 1.))
        names_law;
      cases "a figure too small names a size that lays it out:" ~id:fst
        [
          ( "a titled panel at 40 × 30 pt",
            fun () -> (Size.figure 40. 30., title "A long title" (plain ramp))
          );
          ( "a titled bare panel half a point shorter than its title",
            fun () ->
              let least = (2. *. margin) +. head_h "t" +. title_gap in
              (Size.figure 100. (least -. 0.5), title "t" bare) );
        ]
        (fun (_, row) ->
          let size, f = row () in
          lays_out_named size f);
      cases "a long y title lays out or names a size that does at"
        ~id:(Printf.sprintf "%g pt")
        [ 120.; 155.; 160.; 181.; 182.; 182.2; 182.29; 182.3 ] (fun w ->
          match tried (Size.figure w 200.) long_title with
          | Ok l -> apart l
          | Error (w, h) -> apart (lay (Size.figure w h) long_title));
      test "a gap is one em when nothing protrudes into it" (fun () ->
          match boxes (lay (Size.panels 50. 50.) (grid [ [ bare; bare ] ])) with
          | [ a; b ] -> equal close em (Box2.minx b -. Box2.maxx a)
          | bs -> failf "%d panels" (List.length bs));
      test "a gap holds the protrusions that meet it plus one em" (fun () ->
          let cell = layer [ plain ramp; axis ~show:false "x" ] in
          match boxes (lay (Size.panels 50. 50.) (grid [ [ cell; cell ] ])) with
          | [ a; b ] ->
              (* Both panels have the same y axis: its protrusion is the left
                 one of [a], and meets the gap from [b]. *)
              equal close
                (Box2.minx a -. margin +. em)
                (Box2.minx b -. Box2.maxx a)
          | bs -> failf "%d panels" (List.length bs));
      test "a left axis protrudes by its tick, a pad and its widest label"
        (fun () ->
          equal close (margin +. tick +. pad +. label_w "bb") (left_edge None));
      test "a y title adds nothing to its axis' protrusion" (fun () ->
          equal close (left_edge None) (left_edge (Some "name")));
      equal_shares;
      pinned_share;
      test "an aspect row takes its height from its aspect alone" (fun () ->
          let l = lay (Size.panels 100. 400.) square in
          equal (pair close close)
            (100. +. (2. *. margin), 100. +. (2. *. margin))
            (Layout.size l));
      test "an aspect panel is centred across a column wider than it" (fun () ->
          let l = lay (Size.figure 300. 100.) (grid [ [ square ]; [ bare ] ]) in
          match boxes l with
          | [ a; _ ] -> equal close 150. (P2.x (Box2.mid a))
          | bs -> failf "%d panels" (List.length bs));
      test "an aspect panel alone is centred on its page" (fun () ->
          let b = List.hd (boxes (lay (Size.figure 300. 100.) square)) in
          equal (pair close close) (150., 50.)
            (P2.x (Box2.mid b), P2.y (Box2.mid b)));
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
          let panel = List.hd (boxes l) and b = set_box (text l cat) in
          at_least float_exact ~than:(Box2.miny panel -. 1e-9) (Box2.miny b);
          at_most float_exact ~than:(Box2.maxy panel +. 1e-9) (Box2.maxy b));
      test "a facet grid's y and fx titles share a line" (fun () ->
          let l = lay (Size.panels 80. 60.) titled_facets in
          equal close (P2.y (text l "y").at) (P2.y (text l "head").at));
      prop "titles sharing a line stand an em apart" ~count:0
        ~examples:(List.init 12 (fun k -> "a head" ^ String.make k 'W'))
        (Gen.constant "a head")
        (fun head ->
          let f =
            dot ~x:(num ramp)
              ~y:(num ~title:"a y title" ramp)
              ~fx:(strings ~title:head [| "a"; "b"; "a"; "b" |])
              ()
          in
          let l = lay (Size.panels 60. 40.) f in
          let y = text l "a y title" and h = text l head in
          let one_line = Float.abs (P2.y y.at -. P2.y h.at) < 1e-9 in
          cover "on one line" one_line;
          if one_line then
            at_least float_exact ~than:(em -. 1e-9)
              (Box2.minx (set_box h) -. Box2.maxx (set_box y)));
      test "a long y title moves above the facet title" (fun () ->
          let f =
            dot ~x:(num ramp)
              ~y:(num ~title:"a y title long enough to reach the middle" ramp)
              ~fx:(strings ~title:"head" [| "a"; "b"; "a"; "b" |])
              ()
          in
          let l = lay (Size.panels 60. 40.) f in
          at_most float_exact
            ~than:(Box2.miny (set_box (text l "head")))
            (Box2.maxy
               (set_box (text l "a y title long enough to reach the middle"))));
      prop "thinning keeps labels apart and drops only those that meet one"
        ~examples:[ [| 0.; 0.1; 0.2; 0.3; 5.; 10.; 10. |] ]
        gen_tick_values thinning_law;
      cases "a text no face draws raises naming its character and node:"
        ~id:(fun (n, _, _, _) -> n)
        [
          ( "a figure title",
            None,
            grid [ [ plain ramp; title ("a" ^ unmapped) (plain ramp) ] ],
            "1: " );
          ( "a channel title",
            None,
            dot ~x:(num ~title:unmapped ramp) ~y:(num ramp) (),
            "axis.x: " );
          ( "a facet title",
            None,
            dot ~x:(num ramp) ~y:(num ramp)
              ~fx:(strings ~title:unmapped [| "a"; "b"; "a"; "b" |])
              (),
            "axis.fx: " );
          ( "a row facet title",
            None,
            dot ~x:(num ramp) ~y:(num ramp)
              ~fy:(strings ~title:unmapped [| "a"; "b"; "a"; "b" |])
              (),
            "axis.fy: " );
          ( "a tick label of a continuous scale",
            Some (Locale.v ~decimal:unmapped ()),
            layer [ plain (f64 [| 0.; 0.25; 0.5 |]); axis ~show:false "y" ],
            "axis.x: " );
        ]
        (fun (_, locale, f, node) ->
          let theme = Option.map (fun locale -> Theme.v ~locale ()) locale in
          fails_naming [ node; "U+10FFFD" ] (fun () ->
              lay ?theme (Size.panels 50. 50.) f));
      test "a row header's category no face draws is a warning" (fun () ->
          let fy = strings ~title:"layer" [| "a"; "b" ^ unmapped; "a"; "a" |] in
          let l =
            lay (Size.panels 50. 50.) (dot ~x:(num ramp) ~y:(num ramp) ~fy ())
          in
          match Layout.warnings l with
          | [ (_, msg) ] -> contains ~sub:"U+10FFFD" msg
          | ws -> failf "%d warnings" (List.length ws));
      test "a legend's category no face draws is a warning about the legend"
        (fun () ->
          let f =
            plain ~fill:(strings [| "ok"; "b" ^ unmapped; "ok"; "ok" |]) ramp
          in
          match Layout.warnings (lay (Size.panels 50. 50.) f) with
          | [ ((_, msg) as w) ] ->
              equal (list string) [ "legend.color.cat" ] (ids [ w ]);
              contains ~sub:"U+10FFFD" msg
          | ws -> failf "%d warnings" (List.length ws));
      cases "a previous layout gives the fresh one for"
        ~id:(fun (n, _, _, _) -> n)
        reused
        (fun (_, f, f', size) ->
          let theme = Option.map (fun size -> Theme.v ~size ()) size in
          let size = Size.panels 80. 60. in
          let prev = lay size f in
          let r = resolve f' in
          equal layout_t (layout ?theme size r) (layout ~prev ?theme size r));
    ]

let () =
  exit
    (run "Layout"
       [
         conventions;
         ids_group;
         image_group;
         guides_group;
         facets_group;
         coord_group;
         layer_group;
         grid_group;
         span_group;
         title_group;
         axis_group;
         legend_group;
         size_group;
         theme_group;
         layout_module;
         layout_group;
       ])
