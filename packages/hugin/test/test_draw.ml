(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin
open Windtrap
module Raster = Hugin_vg_raster
module Svg = Hugin_vg_svg

(* Data *)

let f64 a = Nx.create Nx.float64 [| Array.length a |] a
let i32 a = Nx.create Nx.int32 [| Array.length a |] (Array.map Int32.of_int a)
let i64 a = Nx.create Nx.int64 [| Array.length a |] a
let bools a = Nx.create Nx.bool [| Array.length a |] a
let nan = Float.nan
let path segs = Nx.Ptree.Path.v segs
let numel s = Array.fold_left ( * ) 1 s
let xs n = Nx.linspace Nx.float64 0. 1000. n
let ys n = Nx.Rng.normal ~key:(Nx.Rng.key 3) Nx.float64 [| n |]
let other () = line ~y:(num (f64 [| 3.; 1.; 2. |])) ()

(* Drawing *)

let drawn ?view ?theme ?(density = 1.) ?(size = Size.panels 100. 100.) f =
  draw ~density (layout ?theme size (resolve ?view f))

(* [panel ~size f] is the one panel of [f] laid out at [size]. *)
let panel ?(size = Size.panels 100. 100.) f =
  match Layout.panels (layout size (resolve f)) with
  | [ p ] -> p
  | ps -> failf "%d panels" (List.length ps)

let panel_boxes size f =
  List.map
    (fun (p : Layout.panel) -> p.box)
    (Layout.panels (layout size (resolve f)))

let transparent = Theme.v ~paper:Color.transparent ()

(* [page ~theme f] is [f] drawn and rendered at density 1. *)
let page ?theme f =
  Raster.render ~density:1. (Drawing.renderable (drawn ?theme f))

(* [levels_at img p] is the RGBA levels of the pixel of [img], rendered at
   density 1, that holds the point [p] of the page. *)
let levels_at img p =
  let i = Float.to_int (P2.y p) and j = Float.to_int (P2.x p) in
  Array.init 4 (fun c -> Nx.item [ i; j; c ] img)

(* [probe ~reduce bindings read] is a mark whose draw function records [read
   rows] for each panel it draws, and the records, latest first. Its legend
   swatches draw nothing. *)
let probe ?reduce ?shape bindings read =
  let seen = ref [] in
  let m =
    Mark.v ~name:"probe" ?reduce ?shape
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
let rows_of ?(size = Size.panels 100. 100.) ?shape bindings read =
  let m, seen = probe ?shape bindings read in
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

(* The id of the first child of a layer, and of its second. *)
let mark_id = path [ Index 0 ]
let second_id = path [ Index 1 ]

(* [tags id d] is the rows and picture of each tag of [id] in [d]. *)
let tags id d =
  collect
    (function
      | Picture.Tag { tag; picture } when Nx.Ptree.Path.equal tag.id id ->
          Some (tag.rows, picture)
      | _ -> None)
    d

(* [outermost id d] is the picture of each tag of [id] in [d] that no tag of
   [id] holds. *)
let outermost id d =
  let rec go acc (p : Picture.t) =
    match p with
    | Tag { tag; picture } when Nx.Ptree.Path.equal tag.id id -> picture :: acc
    | Group ps -> List.fold_left go acc ps
    | Clip { picture; _ }
    | Transform { picture; _ }
    | Opacity { picture; _ }
    | Stamp { picture; _ }
    | Tag { picture; _ } ->
        go acc picture
    | Empty | Fill _ | Stroke _ | Glyphs _ | Image _ -> acc
  in
  List.rev (go [] (picture d))

(* [under id f d] is [f] of each picture under the outermost tags of [id] in [d]
   where it is [Some], in drawing order. *)
let under id f d =
  List.concat_map
    (fun p ->
      List.rev
        (fold
           (fun acc p -> match f p with Some x -> x :: acc | None -> acc)
           [] p))
    (outermost id d)

let strokes id d =
  under id
    (function
      | Picture.Stroke s -> Some (s.stroke, s.color, s.path) | _ -> None)
    d

let stroke_colors id d = List.map (fun (_, c, _) -> c) (strokes id d)

let fills id d =
  under id
    (function Picture.Fill f -> Some (f.rule, f.color, f.path) | _ -> None)
    d

let images d =
  collect (function Picture.Image i -> Some (i.box, i.pixels) | _ -> None) d

let stamps d = collect (function Picture.Stamp s -> Some s.xs | _ -> None) d

let clips d =
  List.length (collect (function Picture.Clip _ -> Some () | _ -> None) d)

let subpaths p =
  Path.fold
    ~move:(fun n _ _ -> n + 1)
    ~line:(fun n _ _ -> n)
    ~cubic:(fun n _ _ _ _ _ _ -> n)
    ~close:Fun.id 0 p

let closes p =
  Path.fold
    ~move:(fun n _ _ -> n)
    ~line:(fun n _ _ -> n)
    ~cubic:(fun n _ _ _ _ _ _ -> n)
    ~close:succ 0 p

let points_of p =
  List.rev
    (Path.fold
       ~move:(fun acc x y -> (x, y) :: acc)
       ~line:(fun acc x y -> (x, y) :: acc)
       ~cubic:(fun acc _ _ _ _ x y -> (x, y) :: acc)
       ~close:Fun.id [] p)

let warned d = List.map fst (Drawing.warnings d)

(* [legends d] is the id of each legend drawn in [d], once. *)
let legends d =
  let is_legend id =
    List.mem (Nx.Ptree.Path.Field "legend") (Nx.Ptree.Path.segments id)
  in
  List.sort_uniq
    (fun a b ->
      String.compare (Nx.Ptree.Path.to_string a) (Nx.Ptree.Path.to_string b))
    (collect
       (function
         | Picture.Tag { tag; _ } when is_legend tag.id -> Some tag.id
         | _ -> None)
       d)

(* Witnesses *)

let color = Testable.make ~pp:Color.pp ~equal:Color.equal
let dash = Testable.make ~pp:Dash.pp ~equal:Dash.equal
let text_t = Testable.make ~pp:Text.pp ~equal:Text.equal
let drawing = Testable.make ~pp:Drawing.pp ~equal:Drawing.equal
let id = Testable.make ~pp:Nx.Ptree.Path.pp ~equal:Nx.Ptree.Path.equal
let fscale = Testable.make ~pp:Scale.pp ~equal:Scale.equal
let stroke_t = Testable.make ~pp:Stroke.pp ~equal:Stroke.equal
let floats = array float_exact

let fill_rule =
  Testable.make
    ~pp:(fun ppf r ->
      Format.pp_print_string ppf
        (match r with `Nonzero -> "nonzero" | `Even_odd -> "even-odd"))
    ~equal:( = )

(* Page coordinates here are below 1000 points and each comes from a few
   operations, whose rounding errors stay below 1e-12 points: [near] tells a
   misplaced point from them. *)
let near = float 1e-9

(* [corners b] is the extremes of [b], to compare with [near]. *)
let corners b = [ Box2.minx b; Box2.miny b; Box2.maxx b; Box2.maxy b ]

(* A colour level of raster output. Anti-aliasing and compositing round coverage
   to a level, so a raster is stated within one level. *)
let level =
  Testable.make ~pp:Format.pp_print_int ~equal:(fun a b -> abs (a - b) <= 1)

(* [within_one a b] states that two images have one shape and differ by at most
   one colour level in each channel, alpha included. *)
let within_one ~msg a b =
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

(* Missing values *)

let unit_scale = Scale.linear ~domain:(0., 1.) ()

(* Each kind of missing value as the middle of three ys, and the ys of the two
   other rows alone. *)
let missing_kinds =
  let log = Scale.log () and kept = num (f64 [| 1.; 2. |]) in
  [
    ("nan", num (f64 [| 1.; nan; 2. |]), kept);
    ("infinity", num (f64 [| 1.; Float.infinity; 2. |]), kept);
    ("negative infinity", num (f64 [| 1.; Float.neg_infinity; 2. |]), kept);
    ( "a mask",
      num ~valid:(bools [| true; false; true |]) (f64 [| 1.; 5.; 2. |]),
      kept );
    ( "zero on a log scale",
      num ~scale:log (f64 [| 1.; 0.; 2. |]),
      num ~scale:log (f64 [| 1.; 2. |]) );
    ( "a negative value on a log scale",
      num ~scale:log (f64 [| 1.; -1.; 2. |]),
      num ~scale:log (f64 [| 1.; 2. |]) );
  ]

(* A dropped row is drawn nowhere and fits no scale, so the other rows are where
   they are without it. *)
let missing_drops (_, y, kept) =
  let points x y =
    rows_of [ Mark.bind Role.x (num (f64 x)); Mark.bind Role.y y ] Mark.points
  in
  let xs, ys = points [| 0.; 1.; 2. |] y
  and xs', ys' = points [| 0.; 2. |] kept in
  let dropped a = [| a.(0); nan; a.(1) |] in
  equal (pair floats floats) (dropped xs', dropped ys') (xs, ys)

let missing_values =
  group "missing values"
    [
      cases "a missing value drops its row"
        ~id:(fun (n, _, _) -> n)
        missing_kinds missing_drops;
      test "a line is cut at the domain's edge, its stroke drawn whole"
        (fun () ->
          let f =
            layer
              [
                line
                  ~x:(num ~scale:unit_scale (f64 [| 0.; 2. |]))
                  ~y:(num ~scale:unit_scale (f64 [| 0.5; 0.5 |]))
                  ();
              ]
          in
          let at = Coord.point (panel f).projection and d = drawn f in
          equal int 0 (clips d);
          equal
            (list (list near))
            [ corners (Box2.of_pts (at 0. 0.5) (at 1. 0.5)) ]
            (List.map
               (fun (_, _, p) -> corners (Option.get (Path.bounds p)))
               (strokes mark_id d)));
      test "a dot at the domain's corner is drawn whole, one outside not at all"
        (fun () ->
          let f =
            layer
              [
                dot
                  ~x:(num ~scale:unit_scale (f64 [| 1.; 1.5 |]))
                  ~y:(num ~scale:unit_scale (f64 [| 1.; 0.5 |]))
                  ();
              ]
          in
          let corner = Coord.point (panel f).projection 1. 1. and d = drawn f in
          equal int 0 (clips d);
          equal (list floats) [ [| P2.x corner; nan |] ] (stamps d));
      test "a text anchored at the domain's corner is drawn whole" (fun () ->
          let d =
            drawn
              (layer
                 [
                   Hugin.text ~dy:5.
                     ~x:(num ~scale:unit_scale (f64 [| 1. |]))
                     ~y:(num ~scale:unit_scale (f64 [| 1. |]))
                     ~text:(strings [| "0.851" |]) ();
                 ])
          in
          equal int 0 (clips d);
          greater int ~than:0
            (List.length
               (under mark_id
                  (function Picture.Glyphs _ -> Some () | _ -> None)
                  d)));
      test "an image beyond its domain keeps its pixels, clipped to the panel"
        (fun () ->
          let zoom =
            dot
              ~x:(num ~scale:(Scale.linear ~domain:(1., 3.) ()) (f64 [| 2. |]))
              ~y:(num (f64 [| 2. |]))
              ()
          in
          let f = layer [ image (Nx.ones Nx.float32 [| 4; 4 |]); zoom ] in
          let box = (panel f).box and d = drawn f in
          equal int 1 (clips d);
          equal (list near)
            [ 2. *. Box2.w box ]
            (List.map (fun (b, _) -> Box2.w b) (images d)));
    ]

(* Roles *)

(* [written c] is what the text role writes for each value of [c]. *)
let written c =
  rows_of [ Mark.bind Role.text c ] (fun r -> Option.get (Mark.get r Role.text))

let texts l = Array.of_list (List.map Text.v l)

let roles =
  group "roles"
    [
      cases
        "the text role writes a quantity with the decimals its most demanding \
         value needs"
        ~id:(fun (n, _, _) -> n)
        [
          ( "float32 0.9234",
            num (Nx.create Nx.float32 [| 1 |] [| 0.9234 |]),
            [ "0.9234" ] );
          ("float64 1.5 and 0.25", num (f64 [| 1.5; 0.25 |]), [ "1.50"; "0.25" ]);
          ("int32 3 and 12", num (i32 [| 3; 12 |]), [ "3"; "12" ]);
        ]
        (fun (_, c, expected) ->
          equal (array text_t) (texts expected) (written c));
      test "the text role writes an instant at UTC in RFC 3339" (fun () ->
          equal (array text_t)
            (texts [ "1970-01-01T00:00:00Z"; "1970-01-01T00:00:01.25Z" ])
            (written (time Ms (i64 [| 0L; 1_250L |]))));
      test "the text role writes a year before 0 signed, with four digits"
        (fun () ->
          (* -0001-01-01 is 365 days before 0000-01-01, which is 719,528 days
             before the epoch in the proleptic Gregorian calendar. *)
          let s = Int64.mul (-719_893L) 86_400L in
          equal (array text_t)
            (texts [ "-0001-01-01T00:00:00Z" ])
            (written (time S (i64 [| s |]))));
    ]

(* time *)

(* Law: a row of instants is at its instant normalised by the fitted temporal
   scale, and a masked row is dropped. *)
let time_positions (counts, valid) =
  cover "a masked row" (Array.exists not valid);
  let (us, _), s =
    rows_of
      [ Mark.bind Role.x (time ~valid:(bools valid) Us (i64 counts)) ]
      (fun r -> (Mark.positions r, Option.get (Mark.scale r `X Scale.Temporal)))
  in
  let expected =
    Array.mapi
      (fun i n -> if valid.(i) then Scale.normalize s (Time.v Us n) else nan)
      counts
  in
  equal floats expected us

let gen_time_rows =
  let open Gen in
  let* n = int_range 1 6 in
  pair
    (array ~size:(constant n)
       (map Int64.of_int (int_range (-1_000_000_000) 1_000_000_000)))
    (array ~size:(constant n)
       (frequency [ (4, constant true); (1, constant false) ]))

let times =
  group "time"
    [
      prop "instants are at their normalised values" gen_time_rows
        time_positions;
      cases "instants at the ends of int64 keep apart"
        ~id:(fun c -> Printf.sprintf "%Ld and %Ld" c.(0) c.(1))
        [
          [| Int64.min_int; Int64.succ Int64.min_int |];
          [| Int64.pred Int64.max_int; Int64.max_int |];
          [| Int64.min_int; Int64.max_int |];
        ]
        (fun counts ->
          equal floats [| 0.; 1. |]
            (rows_of
               [
                 Mark.bind Role.x
                   (time ~scale:(Scale.time ~nice:false ()) Ns (i64 counts));
               ]
               (fun r -> fst (Mark.positions r))));
    ]

(* index *)

let value_role = Role.value ~name:"v"

let gen_axis =
  let open Gen in
  let* shape = array ~size:(int_range 1 3) (int_range 1 3) in
  let rank = Array.length shape in
  let+ k = int_range (-rank) (rank - 1) in
  (shape, k)

let index_law (shape, k) =
  cover "an axis counted from the last" (k < 0);
  let rank = Array.length shape in
  let a = if k < 0 then rank + k else k in
  let stride = numel (Array.sub shape (a + 1) (rank - a - 1)) in
  let expected =
    Array.init (numel shape) (fun i -> Float.of_int (i / stride mod shape.(a)))
  in
  equal floats expected
    (rows_of ~shape
       [ Mark.bind value_role (index k) ]
       (fun r -> Option.get (Mark.get r value_role)))

(* map_range *)

let map_ranges =
  group "map_range"
    [
      test "a missing value is not given to the function" (fun () ->
          let fill =
            map_range
              (fun _ -> Color.red)
              (num
                 ~scale:(Scale.linear ~unknown:Color.blue ())
                 (f64 [| 1.; nan |]))
          in
          equal (array color)
            [| Color.red; Color.blue |]
            (rows_of
               [ Mark.bind Role.fill fill ]
               (fun r -> Option.get (Mark.get r Role.fill))));
      test "a scale read only through map_range has no legend" (fun () ->
          let z = f64 [| 1.; 2. |] in
          let d =
            drawn
              (dot ~x:(num z) ~y:(num z)
                 ~fill:(map_range Color.contrast (num z))
                 ())
          in
          equal (list id) [] (legends d));
    ]

(* varies *)

(* A channel of distinct values, described for printing. *)
type spec =
  | Tensor of int array
  | Codes of int array
  | Strings_of of int
  | Floats_of of int
  | Index_of of int
  | Dim_of of int
  | Constant

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let pp_spec ppf = function
  | Tensor s -> Format.fprintf ppf "num %a" pp_shape s
  | Codes s -> Format.fprintf ppf "cat %a" pp_shape s
  | Strings_of n -> Format.fprintf ppf "strings %d" n
  | Floats_of n -> Format.fprintf ppf "floats %d" n
  | Index_of k -> Format.fprintf ppf "index %d" k
  | Dim_of k -> Format.fprintf ppf "dim %d" k
  | Constant -> Format.fprintf ppf "const"

let distinct s =
  Nx.reshape s (Nx.arange_f Nx.float64 0. (Float.of_int (numel s)) 1.)

let codes s = Nx.reshape s (Nx.arange Nx.int32 0 (numel s) 1)
let labels n = Array.init n (Printf.sprintf "s%d")
let values n = Array.init n Float.of_int

(* [bound spec] binds the channel [spec] describes to the text role, whose
   values tell distinct data apart. *)
let bound = function
  | Tensor s -> Mark.bind Role.text (num (distinct s))
  | Codes s -> Mark.bind Role.text (cat (codes s))
  | Strings_of n -> Mark.bind Role.text (strings (labels n))
  | Floats_of n -> Mark.bind Role.text (Hugin.floats (values n))
  | Index_of k -> Mark.bind Role.text (index k)
  | Dim_of k -> Mark.bind Role.text (dim k)
  | Constant -> Mark.bind Role.text (const (Text.v "c"))

(* [varies_of shape spec a] is [varies] of the channel [spec] describes. *)
let varies_of shape spec a =
  match spec with
  | Tensor s -> varies shape (num (distinct s)) a
  | Codes s -> varies shape (cat (codes s)) a
  | Strings_of n -> varies shape (strings (labels n)) a
  | Floats_of n -> varies shape (Hugin.floats (values n)) a
  | Index_of k -> varies shape (index k) a
  | Dim_of k -> varies shape (dim k) a
  | Constant -> varies shape (const (Text.v "c")) a

(* A mark's shape and a channel that broadcasts to it without growing it. *)
let gen_channel =
  let open Gen in
  let pp ppf (shape, spec) =
    Format.fprintf ppf "shape %a, %a" pp_shape shape pp_spec spec
  in
  with_pp pp
    (let* shape = array ~size:(int_range 1 3) (int_range 1 3) in
     let rank = Array.length shape in
     let last = shape.(rank - 1) in
     let suffix =
       let* k = int_range 0 rank in
       let+ ones = array ~size:(constant k) bool in
       Array.mapi (fun i one -> if one then 1 else shape.(rank - k + i)) ones
     in
     let+ spec =
       one_of
         [
           map (fun s -> Tensor s) suffix;
           map (fun s -> Codes s) suffix;
           map
             (fun n -> Strings_of n)
             (of_list ~pp:Format.pp_print_int [ 1; last ]);
           map
             (fun n -> Floats_of n)
             (of_list ~pp:Format.pp_print_int [ 1; last ]);
           map (fun k -> Index_of k) (int_range (-rank) (rank - 1));
           map (fun k -> Dim_of k) (int_range (-rank) (rank - 1));
           constant Constant;
         ]
     in
     (shape, spec))

(* The law: [varies] holds of an axis iff two rows a step apart along it hold
   different values. *)
let varies_law (shape, spec) =
  let texts =
    rows_of ~shape [ bound spec ] (fun r -> Option.get (Mark.get r Role.text))
  in
  let rank = Array.length shape in
  let stride a = numel (Array.sub shape (a + 1) (rank - a - 1)) in
  for a = 0 to rank - 1 do
    let s = stride a in
    let differs = ref false in
    Array.iteri
      (fun k t ->
        if k / s mod shape.(a) > 0 && not (Text.equal t texts.(k - s)) then
          differs := true)
      texts;
    cover "an axis it varies along" !differs;
    cover "an axis it does not vary along" (not !differs);
    equal
      ~msg:(Printf.sprintf "axis %d" a)
      bool !differs (varies_of shape spec a);
    equal
      ~msg:(Printf.sprintf "axis %d" (a - rank))
      bool !differs
      (varies_of shape spec (a - rank))
  done;
  equal ~msg:"an axis past the shape" bool false (varies_of shape spec rank)

let channels =
  [
    times;
    group "index"
      [
        prop "index k is the float of each row's index along k" gen_axis
          index_law;
      ];
    map_ranges;
    group "varies"
      [
        prop "varies is whether rows differ along an axis" gen_channel
          varies_law;
      ];
  ]

(* dot *)

let dot_at ?stroke () = layer [ dot ?stroke ~x:(const 0.5) ~y:(const 0.5) () ]

(* [painted f] is [f] rendered on transparent paper and the point of the page at
   the middle of its panel. *)
let painted f =
  (page ~theme:transparent f, Coord.point (panel f).projection 0.5 0.5)

let dots =
  group "dot"
    [
      test "a dot is a circle 0.6 em across" (fun () ->
          match tags mark_id (drawn (dot_at ())) with
          | [ (_, p) ] ->
              let b = Option.get (Picture.bounds p) in
              let across = 0.6 *. Theme.size Theme.default in
              equal (pair near near) (across, across) (Box2.w b, Box2.h b)
          | l -> failf "%d tags" (List.length l));
      test "a dot without a colour channel is filled with the accent" (fun () ->
          let img, mid = painted (dot_at ()) in
          equal (array level)
            (rgba (Theme.accent Theme.default))
            (levels_at img mid));
      test "a dot with a stroke alone is outlined only" (fun () ->
          let img, mid = painted (dot_at ~stroke:(const Color.red) ()) in
          (* The circle's radius is 3 points, its outline 0.8 wide. *)
          let rim = P2.v (P2.x mid +. 3.) (P2.y mid) in
          equal ~msg:"middle" int 0 (levels_at img mid).(3);
          greater ~msg:"rim" int ~than:0 (levels_at img rim).(3));
      test "a constant size is the dot's area in square points" (fun () ->
          let f =
            layer [ dot ~size:(const 50.) ~x:(const 0.5) ~y:(const 0.5) () ]
          in
          match tags mark_id (drawn f) with
          | [ (_, p) ] ->
              let b = Option.get (Picture.bounds p) in
              let across = 2. *. Float.sqrt (50. /. Float.pi) in
              equal (pair near near) (across, across) (Box2.w b, Box2.h b)
          | l -> failf "%d tags" (List.length l));
      cases
        "symbol categories take Symbol.filled or, outlined only, Symbol.stroked"
        ~id:fst
        [
          ("filled", (None, Symbol.filled));
          ("outlined", (Some (const Color.red), Symbol.stroked));
        ]
        (fun (_, (stroke, defaults)) ->
          let z = f64 [| 1.; 2.; 3. |] and ks = [| "a"; "b"; "c" |] in
          let symbols =
            Array.of_list (List.filteri (fun i _ -> i < 3) defaults)
          in
          let dots symbol =
            drawn (dot ?stroke ~symbol ~x:(num z) ~y:(num z) ())
          in
          let squares = Array.make 3 Symbol.square in
          equal drawing
            (dots (strings ~scale:(Scale.band ~symbols ()) ks))
            (dots (strings ks));
          (* Dots that ignored their symbols would draw these alike. *)
          not_equal drawing
            (dots (strings ~scale:(Scale.band ~symbols:squares ()) ks))
            (dots (strings ks)));
      cases "dots are reduced by raster past more rows than device pixels"
        ~id:(fun (n, _) -> Printf.sprintf "%d dots on 100 pixels" n)
        [ (100, false); (101, true) ]
        (fun (n, image) ->
          let u = Nx.Rng.uniform ~key:(Nx.Rng.key 8) Nx.float64 [| n; 2 |] in
          let d =
            drawn ~size:(Size.panels 10. 10.)
              (dot
                 ~x:(num Nx.(slice [ A; I 0 ] u))
                 ~y:(num Nx.(slice [ A; I 1 ] u))
                 ())
          in
          equal (pair int int)
            (if image then (1, 0) else (0, 1))
            (List.length (images d), List.length (stamps d)));
    ]

(* line *)

(* [line_rows f] is the number of rows the mark [f] draws in a layer of its
   own. *)
let line_rows f =
  List.fold_left
    (fun acc -> function
      | Picture.Rows rows, _ -> acc + Array.length rows | _ -> acc)
    0
    (tags mark_id (drawn ~size:(Size.panels 50. 50.) (layer [ f ])))

let gen_column =
  Gen.array ~size:(Gen.int_range 1 6)
    (Gen.frequency
       [
         (4, Gen.float_range (-5.) 5.);
         (1, Gen.of_list ~pp:Format.pp_print_float [ nan; Float.infinity ]);
       ])

(* [solid_law mark ys] states that [mark ys] with a constant solid dash draws
   what it draws without one. *)
let solid_law mark ys =
  let d dash = drawn (layer [ mark dash (f64 ys) ]) in
  equal drawing (d None) (d (Some (const Dash.solid)))

let pattern_cases =
  List.concat_map
    (fun d -> List.map (fun w -> (d, w)) [ 0.5; 2. ])
    [ Dash.dashed; Dash.dotted; Dash.dash_dot; Dash.v [ 3. ] ]

let pattern_id (d, w) = Format.asprintf "%a at %g" Dash.pp d w

(* [pattern_case mark (d, w)] states that each stroke of [mark d w] has [d]'s
   lengths times [w] and the width [w]. *)
let pattern_case mark (d, w) =
  let ss = strokes mark_id (drawn (layer [ mark d w ])) in
  greater int ~than:0 (List.length ss);
  let expected = List.map (fun l -> l *. w) (Dash.lengths d) in
  List.iter
    (fun (s, _, _) ->
      equal
        (pair (list float_exact) float_exact)
        (expected, w)
        (Stroke.dash s, Stroke.width s))
    ss

let lines =
  let y = num (f64 [| 1.; 2.; 3. |]) in
  group "line"
    [
      test "a line breaks at a dropped row" (fun () ->
          let d =
            drawn
              (layer [ line ~y:(num (f64 [| 1.; 2.; nan; 2.; 1.; 3. |])) () ])
          in
          equal (list int) [ 2 ]
            (List.map (fun (_, _, p) -> subpaths p) (strokes mark_id d)));
      test "a filled line fills each series closed under even-odd, unstroked"
        (fun () ->
          let d = drawn (layer [ line ~fill:(const Color.red) ~y () ]) in
          equal
            (pair (list (triple fill_rule int int)) int)
            ([ (`Even_odd, 1, 1) ], 0)
            ( List.map
                (fun (r, _, p) -> (r, subpaths p, closes p))
                (fills mark_id d),
              List.length (strokes mark_id d) ));
      test "a filled line is outlined if a stroke is given" (fun () ->
          let d =
            drawn
              (layer
                 [
                   line ~fill:(const Color.red) ~stroke:(const Color.blue) ~y ();
                 ])
          in
          equal (list color) [ Color.blue ] (stroke_colors mark_id d));
      test "a series takes its stroke from its first row that is not dropped"
        (fun () ->
          let stroke =
            num ~scale:(Scale.linear ~domain:(1., 3.) ()) (f64 [| 5.; 1.; 3. |])
          in
          let d =
            drawn (layer [ line ~y:(num (f64 [| nan; 1.; 2. |])) ~stroke () ])
          in
          equal (list color)
            [ Scheme.color (Theme.scheme Theme.default) 0. ]
            (stroke_colors mark_id d));
      test "a series whose stroke varies along it warns" (fun () ->
          let d =
            drawn (layer [ line ~y ~stroke:(num (f64 [| 1.; 2.; 3. |])) () ])
          in
          equal (list id) [ mark_id ] (warned d));
      test "each series is broken by the pattern of its category" (fun () ->
          let y =
            num (Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 3.; 3.; 2.; 1. |])
          in
          let scale = Scale.band ~dashes:[| Dash.dotted; Dash.dashed |] () in
          let d = drawn (layer [ line ~y ~dash:(dim ~scale 0) () ]) in
          let w = 0.15 *. Theme.size Theme.default in
          let times d = List.map (fun l -> l *. w) (Dash.lengths d) in
          equal
            (list (list float_exact))
            [ times Dash.dotted; times Dash.dashed ]
            (List.map (fun (s, _, _) -> Stroke.dash s) (strokes mark_id d)));
      test "a scale read by stroke and dash has one legend of both" (fun () ->
          let run = Scale.band ~name:"run" () in
          let y =
            num (Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 3.; 3.; 2.; 1. |])
          in
          let d =
            drawn
              (layer
                 [
                   line ~y ~stroke:(dim ~scale:run 0) ~dash:(dim ~scale:run 0)
                     ();
                 ])
          in
          let legend = path [ Field "legend"; Field "run"; Field "cat" ] in
          equal (list id) [ legend ] (legends d);
          let w = 0.15 *. Theme.size Theme.default in
          let times d = List.map (fun l -> l *. w) (Dash.lengths d) in
          equal
            (list (list float_exact))
            [ times Dash.solid; times Dash.dashed ]
            (List.map (fun (s, _, _) -> Stroke.dash s) (strokes legend d)));
      prop "a constant solid dash draws what no dash draws" gen_column
        (solid_law (fun dash y -> line ?dash ~y:(num y) ()));
      cases "a stroke's dash is the pattern's lengths times its width"
        ~id:pattern_id pattern_cases
        (pattern_case (fun d w ->
             line ~dash:(const d) ~width:(const w)
               ~y:(num (f64 [| 1.; 2.; 3. |]))
               ()));
      cases "a solid line drawn with linear or a step is reduced by m4" ~id:fst
        [
          ("linear", fun y -> line ~y ());
          ("step_mid", fun y -> line ~curve:Curve.step_mid ~y ());
          ("a constant solid dash", fun y -> line ~dash:(const Dash.solid) ~y ());
        ]
        (fun (_, f) -> less int ~than:2000 (line_rows (f (num (ys 2000)))));
      cases "a line with another curve, a fill or a dash draws every row"
        ~id:fst
        [
          ("monotone_x", fun y -> line ~curve:Curve.monotone_x ~y ());
          ("natural", fun y -> line ~curve:Curve.natural ~y ());
          ("catmull_rom", fun y -> line ~curve:Curve.catmull_rom ~y ());
          ("basis", fun y -> line ~curve:Curve.basis ~y ());
          ("filled", fun y -> line ~fill:(const Color.red) ~y ());
          ("dashed", fun y -> line ~dash:(const Dash.dashed) ~y ());
        ]
        (fun (_, f) -> equal int 2000 (line_rows (f (num (ys 2000)))));
    ]

(* area *)

let areas =
  group "area"
    [
      test "each series is filled alone" (fun () ->
          let y =
            Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 1.; 2.; 3.; 2. |]
          in
          let d = drawn (layer [ area ~y:(num y) ~fill:(dim 0) () ]) in
          equal (list int) [ 1; 1 ]
            (List.map (fun (_, _, p) -> subpaths p) (fills mark_id d)));
      test "an area alone fills down to zero" (fun () ->
          let ys = f64 [| 1.; 3.; 2. |] in
          within_one ~msg:"pixels"
            (page (area ~y:(num ys) ~y2:(num (f64 [| 0.; 0.; 0. |])) ()))
            (page (area ~y:(num ys) ())));
      test "swapping y and y2 fills the same pixels" (fun () ->
          let a = num (f64 [| 0.; 3.; 1.; 2. |])
          and b = num (f64 [| 2.; 1.; 3.; 0. |]) in
          within_one ~msg:"pixels"
            (page (area ~y:a ~y2:b ()))
            (page (area ~y:b ~y2:a ())));
      test "a stacked area fills between cumulative sums" (fun () ->
          let m a = Nx.create Nx.float64 [| 2; 3 |] a in
          let v = m [| 1.; 2.; 1.; 2.; 1.; 3. |] in
          let lo = m [| 0.; 0.; 0.; 1.; 2.; 1. |]
          and hi = m [| 1.; 2.; 1.; 3.; 3.; 4. |] in
          let scale = Scale.linear ~domain:(0., 4.) () in
          within_one ~msg:"pixels"
            (page
               (area ~y:(num ~scale hi) ~y2:(num ~scale lo) ~fill:(dim 0) ()))
            (page (area ~y:(num ~scale v) ~fill:(dim 0) ~stack:`Zero ())));
      test "a dropped row splits the region" (fun () ->
          let d =
            drawn
              (layer [ area ~y:(num (f64 [| 1.; 2.; nan; 2.; 1.; 3. |])) () ])
          in
          equal (list int) [ 2 ]
            (List.map (fun (_, _, p) -> subpaths p) (fills mark_id d)));
      test "a series whose fill varies along it warns" (fun () ->
          let y = num (f64 [| 1.; 2.; 3. |]) in
          let d =
            drawn (layer [ area ~y ~fill:(num (f64 [| 1.; 2.; 3. |])) () ])
          in
          equal (list id) [ mark_id ] (warned d));
    ]

(* rect *)

let rects =
  group "rect"
    [
      test "stacked bars draw the bars of their cumulative sums" (fun () ->
          let m a = Nx.create Nx.float64 [| 2; 3 |] a in
          let v = m [| 1.; -2.; 3.; 2.; 1.; -1. |] in
          let lo = m [| 0.; 0.; 0.; 1.; 0.; 0. |]
          and hi = m [| 1.; -2.; 3.; 3.; 1.; -1. |] in
          within_one ~msg:"pixels"
            (page (rect ~x:(dim 1) ~y:(num hi) ~y2:(num lo) ~fill:(dim 0) ()))
            (page (rect ~x:(dim 1) ~y:(num v) ~fill:(dim 0) ~stack:`Zero ())));
      test "a stack over instants stacks the quantities of each instant"
        (fun () ->
          let x = time S (i64 [| 0L; 0L; 60L |])
          and x2 = time S (i64 [| 30L; 30L; 90L |]) in
          within_one ~msg:"pixels"
            (page
               (rect ~x ~x2
                  ~y:(num (f64 [| 1.; 3.; 3. |]))
                  ~y2:(num (f64 [| 0.; 1.; 0. |]))
                  ()))
            (page (rect ~x ~x2 ~y:(num (f64 [| 1.; 2.; 3. |])) ~stack:`Zero ())));
      test "a rect with a stroke alone is outlined only" (fun () ->
          let d =
            drawn
              (layer
                 [
                   rect ~x:(strings [| "a" |])
                     ~y:(num (f64 [| 1. |]))
                     ~stroke:(const Color.red) ();
                 ])
          in
          equal
            (pair int (list color))
            (0, [ Color.red ])
            (List.length (fills mark_id d), stroke_colors mark_id d));
    ]

(* frame *)

let framed_dots = dot ~x:(num (f64 [| 1.; 4. |])) ~y:(num (f64 [| 2.; 3. |])) ()

let frames =
  group "frame"
    [
      cases "a frame draws what a rect stroked in the ink draws" ~id:fst
        [ ("default", Theme.default); ("dark", Theme.dark) ]
        (fun (_, theme) ->
          equal drawing
            (drawn ~theme
               (layer
                  [ framed_dots; rect ~stroke:(const (Theme.ink theme)) () ]))
            (drawn ~theme (layer [ framed_dots; frame () ])));
      test "a frame reaches half its outline beyond the data area" (fun () ->
          let size = Size.panels 100. 80. in
          let f = layer [ framed_dots; frame () ] in
          let box = (panel ~size f).box in
          let half = 0.08 *. Theme.size Theme.default /. 2. in
          match tags second_id (drawn ~size f) with
          | [ (_, p) ] ->
              equal (list near)
                [
                  Box2.minx box -. half;
                  Box2.miny box -. half;
                  Box2.maxx box +. half;
                  Box2.maxy box +. half;
                ]
                (corners (Option.get (Picture.bounds p)))
          | l -> failf "%d frames" (List.length l));
      test "a frame takes its stroke and opacity" (fun () ->
          let d =
            drawn
              (layer
                 [
                   framed_dots;
                   frame ~stroke:(const Color.red) ~opacity:(const 0.5) ();
                 ])
          in
          equal (list color)
            [ Color.with_alpha 0.5 Color.red ]
            (stroke_colors second_id d));
    ]

(* rule *)

let span = Scale.linear ~domain:(0., 4.) ()
let on_span a = num ~scale:span (f64 a)

(* [segment_boxes r] is the box of each stroke of the rule [r], and the box of
   each pair of normalised positions of [expected] in its panel. *)
let segment_boxes r expected =
  let f = layer [ r ] in
  let at = Coord.point (panel f).projection in
  ( List.map
      (fun ((u0, v0), (u1, v1)) -> corners (Box2.of_pts (at u0 v0) (at u1 v1)))
      expected,
    List.map
      (fun (_, _, p) -> corners (Option.get (Path.bounds p)))
      (strokes mark_id (drawn f)) )

let rules =
  group "rule"
    [
      cases "a rule draws the segment its positions give"
        ~id:(fun (n, _, _) -> n)
        [
          ( "x alone crosses the panel",
            rule ~x:(on_span [| 1. |]) (),
            [ ((0.25, 0.), (0.25, 1.)) ] );
          ( "x and y make a stem from zero to y",
            rule ~x:(on_span [| 1. |]) ~y:(on_span [| 3. |]) (),
            [ ((0.25, 0.), (0.25, 0.75)) ] );
          ( "y alone crosses the panel",
            rule ~y:(on_span [| 3. |]) (),
            [ ((0., 0.75), (1., 0.75)) ] );
          ( "a position on a band scale is its band's centre",
            rule ~x:(strings [| "a"; "b" |]) (),
            [ ((0.25, 0.), (0.25, 1.)); ((0.75, 0.), (0.75, 1.)) ] );
        ]
        (fun (_, r, expected) ->
          let expected, got = segment_boxes r expected in
          equal (list (list near)) expected got);
      test "x, x2, y and y2 draw the segment from (x, y) to (x2, y2)" (fun () ->
          let f =
            layer
              [
                rule ~x:(on_span [| 1. |]) ~x2:(on_span [| 2. |])
                  ~y:(on_span [| 3. |]) ~y2:(on_span [| 1. |]) ();
              ]
          in
          let at u v =
            let p = Coord.point (panel f).projection u v in
            (P2.x p, P2.y p)
          in
          equal
            (list (pair near near))
            [ at 0.25 0.75; at 0.5 0.25 ]
            (List.concat_map
               (fun (_, _, p) -> points_of p)
               (strokes mark_id (drawn f))));
      test "a rule is stroked in the ink at the line width with butt caps"
        (fun () ->
          let th = Theme.default in
          let d = drawn (layer [ rule ~y:(Hugin.floats [| 0.5 |]) () ]) in
          equal
            (list (pair stroke_t color))
            [ (Stroke.v ~cap:`Butt (0.15 *. Theme.size th), Theme.ink th) ]
            (List.map (fun (s, c, _) -> (s, c)) (strokes mark_id d)));
      prop "a constant solid dash draws what no dash draws" gen_column
        (solid_law (fun dash y -> rule ?dash ~y:(num y) ()));
      cases "a stroke's dash is the pattern's lengths times its width"
        ~id:pattern_id pattern_cases
        (pattern_case (fun d w ->
             rule ~dash:(const d) ~width:(const w)
               ~y:(num (f64 [| 1.; 2.; 3. |]))
               ()));
    ]

(* abline *)

(* Position scales of each transform, a line [y = m x + 1] over data from [(1,
   3)] to [(100, 201)], whether it is a segment, and whether it spans the x
   domain or leaves through the top of the y domain. *)
let transforms =
  let lin name = Scale.linear ~name () and log name = Scale.log ~name () in
  [
    ("linear", (lin "x", lin "y", 2., `Segment, `Across));
    ("log x", (log "x", lin "y", 2., `Curve, `Across));
    ("log y", (lin "x", log "y", 2., `Curve, `Across));
    ("log x and y", (log "x", log "y", 2., `Curve, `Across));
    ("linear, cut at the top", (lin "x", lin "y", 20., `Segment, `Top));
    ( "clamped y",
      (lin "x", Scale.linear ~name:"y" ~clamp:true (), 20., `Segment, `Top) );
    ( "clamped log y",
      (lin "x", Scale.log ~name:"y" ~clamp:true (), 20., `Curve, `Top) );
  ]

(* The points are inverted through the projection and the scales, logarithms
   included, whose rounding errors are a few ulps relative: 1e-9 tells a point
   off the line from them. *)
let equation_case (_, (sx, sy, m, shape, exit)) =
  let f =
    layer
      [
        dot
          ~x:(num ~scale:sx (f64 [| 1.; 100. |]))
          ~y:(num ~scale:sy (f64 [| 3.; 201. |]))
          ();
        abline ~slope:(const m) ~intercept:(const 1.) ();
      ]
  in
  let r = resolve f in
  let l = layout (Size.panels 100. 100.) r in
  let proj = (List.hd (Layout.panels l)).projection in
  let fx = Resolved.scale r sx and fy = Resolved.scale r sy in
  let pts =
    List.concat_map
      (fun (_, _, p) -> points_of p)
      (strokes second_id (draw ~density:1. l))
  in
  let n = List.length pts in
  (match shape with
  | `Segment -> equal ~msg:"a segment" int 2 n
  | `Curve -> greater ~msg:"a curve" int ~than:20 n);
  let uvs =
    List.map
      (fun (px, py) ->
        let u, v = Option.get (Coord.invert proj (P2.v px py)) in
        let x = Option.get (Scale.invert fx u)
        and y = Option.get (Scale.invert fy v) in
        equal
          ~msg:(Printf.sprintf "(%g, %g)" px py)
          (float_rel ~rel:1e-9 ~abs:0.)
          ((m *. x) +. 1.)
          y;
        (u, v))
      pts
  in
  let most f = List.fold_left (fun a uv -> Float.max a (f uv)) 0. uvs in
  equal ~msg:"from the domain's start" near 0.
    (List.fold_left (fun a (u, _) -> Float.min a u) 1. uvs);
  match exit with
  | `Across -> equal ~msg:"to its end" near 1. (most fst)
  | `Top -> equal ~msg:"to its top" near 1. (most snd)

let gen_coefficient =
  Gen.frequency
    [
      (4, Gen.float_range (-10.) 10.);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_float
          [ 0.; nan; Float.infinity; 1e300; -1e300 ] );
    ]

let unchanged_law (m, c) =
  let dots = dot ~x:(num (f64 [| 1.; 4. |])) ~y:(num (f64 [| -2.; 7. |])) () in
  let scales f =
    let r = resolve f in
    ( Resolved.scale r (Scale.linear ~name:"x" ()),
      Resolved.scale r (Scale.linear ~name:"y" ()) )
  in
  equal (pair fscale fscale)
    (scales (layer [ dots ]))
    (scales
       (layer
          [
            dots; abline ~slope:(const m) ~intercept:(num (f64 [| c; 0. |])) ();
          ]))

let ablines =
  group "abline"
    [
      cases "the points of an abline satisfy its equation" ~id:fst transforms
        equation_case;
      prop "an abline leaves every fitted scale as it was"
        (Gen.pair gen_coefficient gen_coefficient)
        unchanged_law;
      cases "an abline of slope 0 draws the rule at its intercept"
        ~id:string_of_float [ 1.5; 2.; 3.25 ] (fun c ->
          let dots =
            dot ~x:(num (f64 [| 1.; 4. |])) ~y:(num (f64 [| 1.; 4. |])) ()
          in
          equal drawing
            (drawn (layer [ dots; rule ~y:(Hugin.floats [| c |]) () ]))
            (drawn
               (layer
                  [ dots; abline ~slope:(const 0.) ~intercept:(const c) () ])));
      test "an abline over a band x draws nothing, with a warning" (fun () ->
          let d =
            drawn
              (layer
                 [
                   dot
                     ~x:(strings [| "a"; "b" |])
                     ~y:(num (f64 [| 1.; 4. |]))
                     ();
                   abline ~slope:(const 1.) ~intercept:(const 0.) ();
                 ])
          in
          equal
            (pair int (list id))
            (0, [ second_id ])
            (List.length (strokes second_id d), warned d));
    ]

(* text *)

let label ?dx ?dy () =
  layer
    [
      Hugin.text ?dx ?dy ~x:(const 0.5) ~y:(const 0.5)
        ~text:(strings [| "Hi" |]) ();
    ]

let text_box f =
  match tags mark_id (drawn f) with
  | [ (_, p) ] -> Option.get (Picture.bounds p)
  | l -> failf "%d tags" (List.length l)

let texts_marks =
  group "text"
    [
      test "dx and dy move a text right and up, in points" (fun () ->
          let b = text_box (label ())
          and b' = text_box (label ~dx:3. ~dy:2. ()) in
          equal (pair near near)
            (Box2.minx b +. 3., Box2.miny b -. 2.)
            (Box2.minx b', Box2.miny b'));
      test "a text is painted in the ink by default" (fun () ->
          equal (list color)
            [ Theme.ink Theme.default ]
            (under mark_id
               (function Picture.Glyphs g -> Some g.color | _ -> None)
               (drawn (label ()))));
      test "a text is set as Mark.text centres it on its point" (fun () ->
          let at f =
            under mark_id
              (function
                | Picture.Glyphs g -> Some (P2.x g.at, P2.y g.at) | _ -> None)
              (drawn f)
          in
          let centred =
            Mark.v ~name:"centred" [] (fun r ->
                let p = Coord.point (Mark.projection r) 0.5 0.5 in
                Mark.text r Color.black p (Text.v "Hi"))
          in
          equal (list (pair near near)) (at (layer [ centred ])) (at (label ())));
      test "a character no face has warns" (fun () ->
          let d =
            drawn
              (layer
                 [
                   Hugin.text ~x:(const 0.5) ~y:(const 0.5)
                     ~text:(strings [| "\u{10FFFD}" |])
                     ();
                 ])
          in
          equal (list id) [ mark_id ] (warned d));
    ]

(* image *)

let image_pixels px =
  match images (drawn (image px)) with
  | [ (_, got) ] -> Nx.to_array got
  | l -> failf "%d images" (List.length l)

let images_marks =
  group "image"
    [
      test
        "a float component is clamped and paints the level nearest 255 times \
         it, halves rounding up" (fun () ->
          (* 255 times the float32 nearest 2.5 / 255 is 2.5 in float32, and 255
             times 0.5039215683937073 is 128.5 in float32 and 128.49999994 in
             float64. *)
          let vs =
            [|
              2.;
              Float.infinity;
              -0.5;
              -0.;
              0.5;
              0.009803921915590763;
              0.5039215683937073;
            |]
          in
          let px = Nx.create Nx.float32 [| 1; Array.length vs; 1 |] vs in
          let grey l = [ l; l; l; 255 ] in
          equal (array int)
            (Array.of_list
               (List.concat_map grey [ 255; 255; 0; 0; 128; 3; 129 ]))
            (image_pixels px));
      test "a uint8 component paints its own level" (fun () ->
          let px = Nx.create Nx.uint8 [| 1; 1; 4 |] [| 0; 7; 128; 255 |] in
          equal (array int) [| 0; 7; 128; 255 |] (image_pixels px));
      test "an image covers 0 to w on x and 0 to h on y, unrounded" (fun () ->
          let r = resolve (image (Nx.zeros Nx.uint8 [| 7; 13 |])) in
          let domain name =
            let (Scale.Floats (a, b)) =
              Scale.domain (Resolved.scale r (Scale.linear ~name ()))
            in
            (a, b)
          in
          equal
            (pair (pair float_exact float_exact) (pair float_exact float_exact))
            ((0., 13.), (0., 7.))
            (domain "x", domain "y"));
      test "an image's row 0 is at the top" (fun () ->
          let r = resolve (image (Nx.zeros Nx.uint8 [| 7; 13 |])) in
          let y = Resolved.scale r (Scale.linear ~name:"y" ()) in
          equal
            (pair float_exact float_exact)
            (1., 0.)
            (Scale.normalize y 0., Scale.normalize y 7.));
      test "an image has no axis" (fun () ->
          let d = drawn (image (Nx.zeros Nx.uint8 [| 7; 13 |])) in
          equal (list id) []
            (collect
               (function
                 | Picture.Tag { tag; _ }
                   when List.mem (Nx.Ptree.Path.Field "axis")
                          (Nx.Ptree.Path.segments tag.id) ->
                     Some tag.id
                 | _ -> None)
               d));
      test "a pixel with a NaN component paints nothing" (fun () ->
          let px =
            Nx.create Nx.float32 [| 1; 2; 3 |] [| nan; 0.; 0.; 0.; 0.; 1. |]
          in
          equal (array int) [| 0; 0; 0; 0; 0; 0; 255; 255 |] (image_pixels px));
    ]

(* contour *)

let grid34 =
  Nx.init Nx.float64 [| 3; 4 |] (fun i -> Float.of_int (i.(0) * i.(1)))

let contours =
  group "contour"
    [
      test
        "each band between consecutive levels is filled with the range at its \
         middle" (fun () ->
          let z = grid34 in
          let scale = Scale.linear ~domain:(0., 6.) ~ticks:[| 2.; 4. |] () in
          let f = layer [ contour ~fill:(num ~scale z) () ] in
          let n =
            Scale.normalize
              (Resolved.scale (resolve f) (Scale.linear ~name:"color" ()))
          in
          let levels = [| 0.; n 2.; n 4.; 1. |] in
          let scheme = Theme.scheme Theme.default in
          equal (list color)
            (List.init 3 (fun k ->
                 Scheme.color scheme ((levels.(k) +. levels.(k + 1)) /. 2.)))
            (List.map (fun (_, c, _) -> c) (fills mark_id (drawn f))));
      test "contour implies stepped on its fill scale, an explicit one beats it"
        (fun () ->
          let fill s = Scale.linear ~name:"z" ?stepped:s () in
          let fitted s =
            Resolved.scale
              (resolve (contour ~fill:(num ~scale:(fill s) grid34) ()))
              (fill None)
          in
          satisfies ~claim:"stepped" fscale Scale.stepped (fitted None);
          satisfies ~claim:"not stepped" fscale
            (fun s -> not (Scale.stepped s))
            (fitted (Some false)));
      test "a missing sample cuts a hole" (fun () ->
          let field hole =
            Nx.init Nx.float64 [| 5; 5 |] (fun i ->
                if hole && i.(0) = 2 && i.(1) = 2 then nan
                else Float.of_int (i.(0) + i.(1)))
          in
          let alpha hole =
            let img, mid =
              painted (layer [ contour ~fill:(num (field hole)) () ])
            in
            (levels_at img mid).(3)
          in
          equal (pair level level) (255, 0) (alpha false, alpha true));
      test
        "a field whose x is not strictly monotone draws nothing, with a warning"
        (fun () ->
          let z = Nx.ones Nx.float64 [| 2; 3 |] in
          let d =
            drawn
              (layer
                 [ contour ~x:(num (f64 [| 0.; 2.; 1. |])) ~fill:(num z) () ])
          in
          equal
            (pair int (list id))
            (0, [ mark_id ])
            (List.length (fills mark_id d), warned d));
    ]

(* Scales *)

let stepped = Scale.linear ~stepped:true ~domain:(0., 10.) ()

(* [mid_step levels u] is the middle of the interval of [levels] holding [u],
   searched in order: the last interval holds its upper level, and a value
   beyond the levels takes the interval at its nearer end. *)
let mid_step levels u =
  let n = Array.length levels in
  let k = ref 0 in
  for i = 1 to n - 2 do
    if levels.(i) <= u then k := i
  done;
  (levels.(!k) +. levels.(!k + 1)) /. 2.

let levels ticks =
  Array.of_list (List.sort_uniq Float.compare (0. :: 1. :: Array.to_list ticks))

(* A stepped scale exported from a figure of other data, whose ticks a figure of
   another size chooses afresh. *)
let exported =
  lazy
    (let z = Scale.linear ~name:"z" ~stepped:true () in
     let f =
       dot ~x:(const 0.5) ~y:(const 0.5)
         ~fill:(num ~scale:z (f64 [| 0.3; 7.1 |]))
         ()
     in
     Resolved.scale (resolve f) z)

let steppings =
  Gen.of_list
    ~pp:(fun ppf (n, _, _) -> Format.pp_print_string ppf n)
    [
      ("a stepped scale", Lazy.from_val stepped, Size.panels 100. 100.);
      ("an exported stepped scale", exported, Size.panels 30. 230.);
    ]

let reads_steps ((_, scale, size), values) =
  let read rows =
    let get r = Option.get (r rows Role.fill) in
    (get Mark.normalized, get Mark.get, get Mark.ticks)
  in
  let norm, colours, ticks =
    rows_of ~size
      [ Mark.bind Role.fill (num ~scale:(Lazy.force scale) (f64 values)) ]
      read
  in
  cover "a value on a level" (Array.exists (fun u -> Array.mem u ticks) norm);
  cover "a value beyond the domain"
    (Array.exists (fun u -> u < 0. || u > 1.) norm);
  let scheme = Theme.scheme Theme.default in
  Array.iteri
    (fun i u ->
      let msg = Printf.sprintf "row %d at %g" i u in
      equal ~msg color
        (Scheme.color scheme (mid_step (levels ticks) u))
        colours.(i))
    norm

let scales =
  group "scales"
    [
      prop
        "a reader of a stepped scale takes the range at the middle of its step \
         between the ticks of the figure drawing it"
        (Gen.pair steppings
           (Gen.array ~size:(Gen.int_range 1 8)
              (Gen.frequency
                 [
                   (4, Gen.float_range (-2.) 12.);
                   (1, Gen.of_list ~pp:Format.pp_print_float [ 0.; 2.; 5.; 10. ]);
                 ])))
        reads_steps;
      test "a stepped scale steps at its explicit ticks" (fun () ->
          let scale =
            Scale.linear ~domain:(0., 1.) ~stepped:true ~ticks:[| 0.5; 0.25 |]
              ()
          in
          let f =
            rows_of
              [ Mark.bind Role.fill (num ~scale (f64 [| 0.; 1. |])) ]
              (fun r -> Option.get (Mark.range r Role.fill))
          in
          let scheme = Theme.scheme Theme.default in
          let at = [ (0.1, 0.125); (0.3, 0.375); (0.9, 0.75) ] in
          equal (list color)
            (List.map (fun (_, mid) -> Scheme.color scheme mid) at)
            (List.map (fun (u, _) -> f u) at));
    ]

(* Guides *)

let guides =
  group "guides"
    [
      test "a stepped colour bar paints each step from its level to the top"
        (fun () ->
          let m, seen =
            probe
              [ Mark.bind Role.fill (num ~scale:stepped (f64 [| 5. |])) ]
              (fun rows -> Option.get (Mark.ticks rows Role.fill))
          in
          let dots =
            dot ~x:(const 0.5) ~y:(const 0.5)
              ~fill:(num ~scale:stepped (f64 [| 0.; 10. |]))
              ()
          in
          let bar = path [ Field "legend"; Field "color"; Field "num" ] in
          let fills =
            List.map
              (fun (_, c, p) -> (c, Option.get (Path.bounds p)))
              (fills bar (drawn (layer [ dots; m ])))
          in
          let levels = levels (only seen) in
          let n = Array.length levels - 1 in
          let scheme = Theme.scheme Theme.default in
          equal (list color)
            (List.init n (fun k ->
                 Scheme.color scheme ((levels.(k) +. levels.(k + 1)) /. 2.)))
            (List.map fst fills);
          match List.map snd fills with
          | bar :: _ as boxes ->
              equal
                (list (pair near near))
                (List.init n (fun k ->
                     (Box2.miny bar, (1. -. levels.(k)) *. Box2.h bar)))
                (List.map (fun b -> (Box2.miny b, Box2.h b)) boxes)
          | [] -> fail "no steps");
    ]

(* Facets *)

(* [panel_rows ~around bindings] is the rows a probe of [bindings] draws in each
   panel of [around probe]. *)
let panel_rows ?(around = Fun.id) bindings () =
  let m, seen = probe bindings Mark.index in
  ignore (drawn (around m));
  !seen

let facets =
  let y shape = Mark.bind Role.y (num (Nx.zeros Nx.float64 shape)) in
  let beside ?fx ?fy m =
    layer [ dot ~x:(const 0.5) ~y:(const 0.5) ?fx ?fy (); m ]
  in
  group "facets"
    [
      cases "a mark's rows are drawn in their facet panels"
        ~id:(fun (n, _, _) -> n)
        [
          ( "a facet channel puts each row in its panel",
            panel_rows [ y [| 2; 3 |]; Mark.bind Role.fx (dim 0) ],
            [ [| 0; 1; 2 |]; [| 3; 4; 5 |] ] );
          ( "a facet on a middle axis puts each block of rows in its panel",
            panel_rows [ y [| 2; 3; 2 |]; Mark.bind Role.fx (dim 1) ],
            [ [| 0; 1; 6; 7 |]; [| 2; 3; 8; 9 |]; [| 4; 5; 10; 11 |] ] );
          ( "a facet a mark leaves unbound puts its rows in every panel",
            panel_rows
              ~around:(beside ~fy:(strings [| "p"; "q" |]))
              [ Mark.bind Role.fx (strings [| "a"; "b"; "a" |]) ],
            [ [| 0; 2 |]; [| 0; 2 |]; [| 1 |]; [| 1 |] ] );
          ( "a row whose facet value is missing is in no panel",
            panel_rows
              [
                y [| 3 |];
                Mark.bind Role.fx (dim ~valid:(bools [| true; false; true |]) 0);
              ],
            [ [| 0 |]; [| 2 |] ] );
          ( "a facet constant draws in its panel only",
            panel_rows
              ~around:(beside ~fx:(strings [| "a"; "b" |]))
              [ Mark.bind Role.fx (const "b") ],
            [ [| 0 |] ] );
        ]
        (fun (_, f, expected) ->
          equal (slist (array int) compare) expected (f ()));
    ]

(* axis *)

let all_strokes d =
  collect
    (function
      | Picture.Stroke s -> Some (s.stroke, s.color, s.path) | _ -> None)
    d

let axes =
  group "axis"
    [
      test "a grid draws a line 0.06 em wide at each tick, in the ink at 0.2"
        (fun () ->
          let m, seen =
            probe
              [ Mark.bind Role.y (num (f64 [| 0.; 10. |])) ]
              (fun r -> Option.get (Mark.ticks r Role.y))
          in
          let plain = all_strokes (drawn (layer [ m ])) in
          let gridded =
            all_strokes (drawn (layer [ m; axis ~grid:true "y" ]))
          in
          let same (s, c, p) (s', c', p') =
            Stroke.equal s s' && Color.equal c c' && Path.equal p p'
          in
          let lines =
            List.filter (fun s -> not (List.exists (same s) plain)) gridded
          in
          let th = Theme.default in
          let ink = Theme.ink th in
          let pen =
            ( 0.06 *. Theme.size th,
              Color.with_alpha (0.2 *. Color.alpha ink) ink )
          in
          greater int ~than:0 (List.length lines);
          equal
            (list (pair near color))
            (List.map (fun _ -> pen) lines)
            (List.map (fun (s, c, _) -> (Stroke.width s, c)) lines);
          equal int
            (Array.length (List.hd !seen))
            (List.fold_left (fun n (_, _, p) -> n + subpaths p) 0 lines));
    ]

(* legend *)

(* [first_tag id d] is the place of the first tag of [id] in the drawing order
   of [d]. *)
let first_tag id d =
  let order =
    collect (function Picture.Tag { tag; _ } -> Some tag.id | _ -> None) d
  in
  let rec find i = function
    | [] -> failf "no tag %s" (Nx.Ptree.Path.to_string id)
    | x :: rest -> if Nx.Ptree.Path.equal x id then i else find (i + 1) rest
  in
  find 0 order

let inside_legend side =
  let f =
    layer
      [
        dot
          ~x:(num (f64 [| 1.; 2. |]))
          ~y:(num (f64 [| 1.; 2. |]))
          ~fill:(strings [| "a"; "b" |])
          ();
        legend ~side "color";
      ]
  in
  let d = drawn ~size:(Size.panels 150. 100.) f in
  let legend = path [ Field "legend"; Field "color"; Field "cat" ] in
  (first_tag legend d, first_tag mark_id d)

(* [swatch_boxes ?size side ?ncol f n] is the box on the page of each of the [n]
   entries of the legend of the categorical colour scale that a probe mark of
   [n] categories layered under [legend ~side ~ncol "color"] draws, in entry
   order, and the panels' boxes. *)
let swatch_boxes ?(size = Size.panels 150. 120.) ?ncol side n =
  let boxes = Array.make n None in
  let m =
    Mark.v ~name:"probe"
      ~swatch:(fun rows ->
        let p = Mark.projection rows in
        let a = Coord.point p 0. 1. and b = Coord.point p 1. 0. in
        boxes.((Mark.index rows).(0)) <- Some (Box2.of_pts a b);
        Picture.empty)
      [ Mark.bind Role.fill (strings (Array.init n string_of_int)) ]
      (fun _ -> Picture.empty)
  in
  let f = layer [ m; legend ~side ?ncol "color" ] in
  ignore (drawn ~size f);
  (Array.map Option.get boxes, panel_boxes size f)

let gen_columns =
  let open Gen in
  let side =
    of_list
      ~pp:(fun ppf _ -> Format.pp_print_string ppf "side")
      [ `Right; `Left; `Bottom; `Top; `Inside `Center; `Inside `Bottom_left ]
  in
  triple (int_range 1 9) (int_range 1 4) side

(* [rank vs v] is the place of [v] among the distinct values of [vs], in
   increasing order. *)
let rank vs v =
  List.length
    (List.filter
       (fun w -> w < v)
       (List.sort_uniq Float.compare (Array.to_list vs)))

(* Law: entry [e] lies in row [e / ncol] and column [e mod ncol], rows going
   down and columns right, on every side and inside. *)
let columns_law (n, k, side) =
  cover "a short last row" (n mod k <> 0 && n > k);
  cover "several rows" (n > k);
  let boxes, _ = swatch_boxes ~ncol:k side n in
  let per = Int.min n k in
  let xs = Array.map Box2.minx boxes and ys = Array.map Box2.miny boxes in
  equal
    (array (pair int int))
    (Array.init n (fun e -> (e / per, e mod per)))
    (Array.map (fun b -> (rank ys (Box2.miny b), rank xs (Box2.minx b))) boxes)

let legends_figures =
  let place p = Box2.minx (fst (swatch_boxes (`Inside p) 3)).(0) in
  let down p = Box2.miny (fst (swatch_boxes (`Inside p) 3)).(0) in
  group "legend"
    [
      prop "ncol sets rows of ncol entries filled from the left" gen_columns
        columns_law;
      test "an inside legend lies midway between the edges' places" (fun () ->
          equal (list near)
            [
              (place `Left +. place `Right) /. 2.;
              (place `Top_left +. place `Top_right) /. 2.;
              (down `Top +. down `Bottom) /. 2.;
              (down `Top_left +. down `Bottom_left) /. 2.;
            ]
            [ place `Center; place `Top; down `Center; down `Left ]);
      test "an inside legend at a left place is 0.25 em from the left edge"
        (fun () ->
          let boxes, panels = swatch_boxes (`Inside `Left) 3 in
          let pad = 0.25 *. Theme.size Theme.default in
          equal near (Box2.minx (List.hd panels) +. pad) (Box2.minx boxes.(0)));
      test "an inside legend is drawn under the marks" (fun () ->
          let l, m = inside_legend (`Inside `Top_right) in
          less int ~than:m l);
      test "a legend on a side is drawn over the marks" (fun () ->
          let l, m = inside_legend `Right in
          greater int ~than:m l);
    ]

(* Theme *)

(* Dots fill and draw no text, so every stroke is an axis line or tick and every
   run of glyphs a title or label: the figure's, the axes' and the legend's
   titles are four runs. *)
let inked () =
  let ink = Color.v ~alpha:0.8 0.2 0.1 0.3 in
  let part a = Color.with_alpha (0.8 *. a) ink in
  let f =
    dot
      ~x:(num ~title:"x" (f64 [| 0.; 1.; 2. |]))
      ~y:(num ~title:"y" (f64 [| 0.; 1.; 2. |]))
      ~fill:(strings ~title:"kind" [| "a"; "b"; "a" |])
      ()
    |> Hugin.title "T"
  in
  let d = drawn ~theme:(Theme.v ~ink ()) f in
  let strokes = List.map (fun (_, c, _) -> c) (all_strokes d) in
  let titles, labels =
    List.partition (Color.equal ink)
      (collect (function Picture.Glyphs g -> Some g.color | _ -> None) d)
  in
  at_least int ~than:2 (List.length strokes);
  List.iter (equal color (part 0.6)) strokes;
  equal int 4 (List.length titles);
  at_least int ~than:2 (List.length labels);
  List.iter (equal color (part 0.75)) labels

let themes =
  group "Theme"
    [
      test "titles take the ink, labels and axes a part of its opacity" inked;
      test "dash categories take their scale's patterns, cycling" (fun () ->
          let ds = [| Dash.dashed; Dash.dotted |] in
          let scale = Scale.band ~dashes:ds () in
          let got =
            rows_of ~shape:[| 5; 2 |]
              [ Mark.bind Role.dash (dim ~scale 0) ]
              (fun r -> Option.get (Mark.get r Role.dash))
          in
          equal (array dash) (Array.init 10 (fun k -> ds.(k / 2 mod 2))) got);
      test "dash categories take Dash.all by default, cycling" (fun () ->
          let all = Array.of_list Dash.all in
          let got =
            rows_of ~shape:[| 6 |]
              [ Mark.bind Role.dash (dim 0) ]
              (fun r -> Option.get (Mark.get r Role.dash))
          in
          equal (array dash) (Array.init 6 (fun k -> all.(k mod 4))) got);
      test "the paper is painted first" (fun () ->
          match
            picture (drawn ~theme:(Theme.v ~paper:Color.blue ()) (other ()))
          with
          | Group (Fill { color = c; _ } :: _) | Fill { color = c; _ } ->
              equal color Color.blue c
          | _ -> fail "no paper first");
      test "a transparent paper paints nothing" (fun () ->
          let img = page ~theme:transparent (other ()) in
          equal (array int) [| 0; 0; 0; 0 |] (levels_at img (P2.v 0.5 0.5)));
    ]

(* Mark.bind *)

(* [reference offset columns lengths] is each row's interval, stacked by the
   definition of [Stack.intervals]: within a column, in row order, the lengths
   that are not negative upward from [0.] and the negative ones downward, then
   moved by [offset]. A missing length covers [(nan, nan)]. *)
let reference offset columns lengths =
  let n = Array.length lengths in
  let starts = Array.make n nan and ends = Array.make n nan in
  let up = Hashtbl.create 8 and down = Hashtbl.create 8 in
  let get t c = Option.value ~default:0. (Hashtbl.find_opt t c) in
  Array.iteri
    (fun r l ->
      if Float.is_finite l then begin
        let c = columns.(r) in
        let t = if l >= 0. then up else down in
        starts.(r) <- get t c;
        ends.(r) <- get t c +. l;
        Hashtbl.replace t c ends.(r)
      end)
    lengths;
  let move r =
    let c = columns.(r) in
    let lo = get down c and hi = get up c in
    match offset with
    | `Zero -> ()
    | `Center ->
        let d = -.(lo +. hi) /. 2. in
        starts.(r) <- starts.(r) +. d;
        ends.(r) <- ends.(r) +. d
    | `Expand ->
        if hi -. lo > 0. then begin
          starts.(r) <- (starts.(r) -. lo) /. (hi -. lo);
          ends.(r) <- (ends.(r) -. lo) /. (hi -. lo)
        end
  in
  for r = 0 to n - 1 do
    if Float.is_finite lengths.(r) then move r
  done;
  (starts, ends)

let pp_offset ppf (o : Stack.offset) =
  Format.pp_print_string ppf
    (match o with
    | `Zero -> "`Zero"
    | `Expand -> "`Expand"
    | `Center -> "`Center")

(* Long-form rows: each row's x category of three, its panel of two, and its
   length, some negative, zero or missing; none at all. *)
let gen_stacks =
  let open Gen in
  let length =
    frequency
      [
        (6, float_range (-3.) 3.);
        (1, of_list ~pp:Format.pp_print_float [ 0.; -0.; nan; Float.infinity ]);
      ]
  in
  let* n = int_range 0 9 in
  (* A code of [-1] is missing: an x drops its row, a fill keeps it. *)
  let code k = frequency [ (6, int_range 0 k); (1, constant (-1)) ] in
  let opacity = frequency [ (6, float_range 0. 1.); (1, constant nan) ] in
  let+ xs = array ~size:(constant n) (code 2)
  and+ panels = array ~size:(constant n) (int_range 0 1)
  and+ lengths = array ~size:(constant n) length
  and+ opacities = array ~size:(constant n) opacity
  and+ fills = array ~size:(constant n) (code 1)
  and+ offset = of_list ~pp:pp_offset [ `Zero; `Expand; `Center ] in
  (xs, panels, lengths, (opacities, fills), offset)

(* Law: a stacked y gives each row the interval its column's stack gives it,
   columns being told apart by x and by panel, a row that another binding drops
   stacking nothing and a missing colour dropping nothing, and its scale is
   fitted to the hull of the intervals.

   Intervals are sums of at most nine lengths below 3, so below 54 apart.
   Normalising one and inverting it costs a few rounding errors of 54, about
   7e-15 each: 1e-12 holds them. *)
let stack_law (xs, panels, lengths, (opacities, fills), offset) =
  let n = Array.length xs in
  let rows = Array.init n Fun.id in
  let columns = Array.init n (fun r -> (3 * panels.(r)) + xs.(r)) in
  let dropped r = xs.(r) < 0 || Float.is_nan opacities.(r) in
  let stacked = Array.mapi (fun r l -> if dropped r then nan else l) lengths in
  let starts, ends = reference offset columns stacked in
  let finite r = Float.is_finite lengths.(r) in
  cover "a row dropped by its opacity"
    (Array.exists (fun r -> finite r && Float.is_nan opacities.(r)) rows);
  cover "a row dropped by its x"
    (Array.exists (fun r -> finite r && xs.(r) < 0) rows);
  cover "a row kept with a missing colour"
    (Array.exists
       (fun r -> finite r && (not (dropped r)) && fills.(r) < 0)
       rows);
  cover "a negative length" (Array.exists (fun l -> l < 0.) lengths);
  cover "a dropped row" (Array.exists Float.is_nan starts);
  cover "a column of two rows"
    (Array.exists
       (fun r ->
         Array.exists (fun r' -> r' <> r && columns.(r') = columns.(r)) rows)
       rows);
  let m, seen =
    probe
      [
        Mark.bind Role.x (cat ~labels:[| "a"; "b"; "c" |] (i32 xs));
        Mark.bind ~stack:offset Role.y
          (num ~scale:(Scale.linear ~nice:false ()) (f64 lengths));
        Mark.bind Role.fx (cat ~labels:[| "p"; "q" |] (i32 panels));
        Mark.bind Role.opacity (num (f64 opacities));
        Mark.bind Role.fill (cat ~labels:[| "u"; "v" |] (i32 fills));
      ]
      (fun r ->
        let s = Option.get (Mark.scale r `Y Scale.Quantitative) in
        (Mark.index r, Mark.extent r `Y, s))
  in
  ignore (drawn m);
  let value s u =
    if Float.is_nan u then nan else Option.get (Scale.invert s u)
  in
  let close = float 1e-12 in
  let lo = ref infinity and hi = ref neg_infinity in
  List.iter
    (fun (index, (a, b), s) ->
      Array.iteri
        (fun k r ->
          let msg = Printf.sprintf "row %d" r in
          if Float.is_nan starts.(r) then
            equal ~msg floats [| nan; nan |] [| a.(k); b.(k) |]
          else
            equal ~msg (pair close close)
              (starts.(r), ends.(r))
              (value s a.(k), value s b.(k));
          if not (Float.is_nan a.(k)) then begin
            lo := Float.min !lo (Float.min a.(k) b.(k));
            hi := Float.max !hi (Float.max a.(k) b.(k))
          end)
        index)
    !seen;
  (* A domain of one value normalises it to [0.5]. *)
  let fitted = !lo < !hi in
  cover "a domain fitted to the intervals" fitted;
  if fitted then equal (pair close close) (0., 1.) (!lo, !hi)

let binds =
  group "Mark.bind"
    [
      prop "stacked intervals follow Stack.intervals by column and panel"
        ~examples:[ ([||], [||], [||], ([||], [||]), `Zero) ]
        gen_stacks stack_law;
      test "a pile on a log scale starts at the domain's lower end" (fun () ->
          let v = Nx.create Nx.float64 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
          let (a, b), s =
            rows_of
              [
                Mark.bind Role.x (dim 1);
                Mark.bind ~stack:`Zero Role.y
                  (num ~scale:(Scale.log ~nice:false ()) v);
              ]
              (fun r ->
                ( Mark.extent r `Y,
                  Option.get (Mark.scale r `Y Scale.Quantitative) ))
          in
          (* Piles of 1 then 3, and of 2 then 4: ends 1, 2, 4 and 6, and starts
             1 and 2 that the scale places. *)
          let n = Scale.normalize s in
          let (Scale.Floats (lo, hi)) = Scale.domain s in
          equal
            (triple (pair float_exact float_exact) floats floats)
            ((1., 6.), [| 0.; 0.; n 1.; n 2. |], [| n 1.; n 2.; n 4.; n 6. |])
            ((lo, hi), a, b));
      test "a stack over instants stacks the quantities of each instant"
        (fun () ->
          let t = Nx.create Nx.int64 [| 3 |] [| 0L; 0L; 60L |] in
          let e =
            rows_of
              [
                Mark.bind Role.x (time S t);
                Mark.bind ~stack:`Zero Role.y
                  (num
                     ~scale:(Scale.linear ~domain:(0., 4.) ())
                     (f64 [| 1.; 2.; 3. |]));
              ]
              (fun r -> Mark.extent r `Y)
          in
          equal (pair floats floats)
            ([| 0.; 0.25; 0. |], [| 0.25; 0.75; 0.75 |])
            e);
    ]

(* Mark rows *)

(* The rows of a probe of x [0; 1; 2; 3] and y [0; nan; 2; 3], read by
   [read]. *)
let with_dropped read =
  rows_of
    [
      Mark.bind Role.x (num (f64 [| 0.; 1.; 2.; 3. |]));
      Mark.bind Role.y (num (f64 [| 0.; nan; 2.; 3. |]));
    ]
    read

let mark_rows =
  group "Mark rows"
    [
      test "a dropped row keeps its own values in get and normalized" (fun () ->
          (* The x domain is [0, 3], fitted to the rows kept. *)
          let nx, gx, ny =
            with_dropped (fun r ->
                ( Mark.normalized r Role.x,
                  Mark.get r Role.x,
                  Mark.normalized r Role.y ))
          in
          equal
            (triple (option floats) (option floats) (option floats))
            ( Some [| 0.; 1. /. 3.; 2. /. 3.; 1. |],
              Some [| 0.; 1. /. 3.; 2. /. 3.; 1. |],
              Some [| 0.; nan; 2. /. 3.; 1. |] )
            (nx, gx, ny));
    ]

(* Mark.get, Mark.range *)

(* [get] is [range] applied to [normalized], on any data. *)
let get_is_range_of_normalized fill =
  let got, f, us =
    rows_of
      [ Mark.bind Role.fill fill ]
      (fun r ->
        ( Mark.get r Role.fill,
          Mark.range r Role.fill,
          Mark.normalized r Role.fill ))
  in
  let f = require_some f and us = require_some us in
  equal (option (array color)) (Some (Array.map f us)) got

let continuous_fill (values, mapped) =
  cover "a missing value"
    (List.exists (fun v -> not (Float.is_finite v)) values);
  cover "mapped" mapped;
  let fill = num (f64 (Array.of_list values)) in
  get_is_range_of_normalized
    (if mapped then map_range Color.contrast fill else fill)

let band_fill (codes, mapped) =
  cover "a code outside the labels"
    (List.exists (fun c -> c < 0 || c > 2) codes);
  cover "mapped" mapped;
  let fill = cat ~labels:[| "a"; "b"; "c" |] (i32 (Array.of_list codes)) in
  get_is_range_of_normalized
    (if mapped then map_range Color.contrast fill else fill)

let value_gen =
  Gen.frequency
    [
      (8, Gen.float_range (-10.) 10.);
      (1, Gen.of_list ~pp:Format.pp_print_float [ nan; Float.infinity ]);
    ]

let gets =
  group "Mark.get"
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
          equal
            (Testable.make ~pp:Symbol.pp ~equal:Symbol.equal)
            Symbol.square
            (Option.get s).(1));
      test "a missing text is empty" (fun () ->
          let t =
            rows_of
              [ Mark.bind Role.text (num (f64 [| 1.5; nan |])) ]
              (fun r -> Mark.get r Role.text)
          in
          equal text_t (Text.v "") (Option.get t).(1));
      test "a parameter's constant is given per row, under its role only"
        (fun () ->
          let k = Role.param ~name:"k" ~equal:Int.equal in
          let k' = Role.param ~name:"k" ~equal:Int.equal in
          let got =
            rows_of
              [
                Mark.bind Role.x (num (f64 [| 0.; 1. |])); Mark.bind k (const 7);
              ]
              (fun r -> (Mark.get r k, Mark.get r k'))
          in
          equal
            (pair (option (array int)) (option (array int)))
            (Some [| 7; 7 |], None)
            got);
    ]

let ranges =
  group "Mark.range"
    [
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
      prop "get is the range of normalized on a continuous scale"
        (Gen.pair (Gen.list ~size:(Gen.int_range 1 6) value_gen) Gen.bool)
        continuous_fill;
      prop "get is the range of normalized on a band scale"
        (Gen.pair
           (Gen.list ~size:(Gen.int_range 1 6) (Gen.int_range (-1) 4))
           Gen.bool)
        band_fill;
    ]

(* Mark.scale *)

let mark_scales =
  group "Mark.scale"
    [
      test "Mark.scale is the panel's fitted position scale of its kind"
        (fun () ->
          let x = Scale.log ~name:"x" () in
          let m, seen =
            probe
              [ Mark.bind Role.fill (num (f64 [| 1. |])) ]
              (fun rows ->
                ( Mark.scale rows `X Scale.Quantitative,
                  Mark.scale rows `X Scale.Categorical,
                  Mark.scale rows `Y Scale.Quantitative ))
          in
          let dots =
            dot
              ~x:(num ~scale:x (f64 [| 1.; 50. |]))
              ~y:(num (f64 [| 1.; 2. |]))
              ()
          in
          let f = layer [ dots; m ] in
          ignore (drawn f);
          let qx, cx, qy = only seen in
          let r = resolve f in
          equal
            (triple (option fscale) (option int) (option fscale))
            ( Some (Resolved.scale r x),
              None,
              Some (Resolved.scale r (Scale.linear ~name:"y" ())) )
            (qx, Option.map (fun _ -> 0) cx, qy));
    ]

(* Mark.ticks *)

let gen_ticks =
  Gen.array ~size:(Gen.int_range 0 8)
    (Gen.frequency
       [
         (4, Gen.float_range (-2.) 12.);
         ( 2,
           Gen.of_list ~pp:Format.pp_print_float
             [ 0.; -0.; 10.; 5.; 5.; nan; Float.infinity; Float.neg_infinity ]
         );
         (1, Gen.any_float);
       ])

(* [-0.] and [0.] are one tick. *)
let unsigned_zero = Array.map (fun u -> u +. 0.)

let ticks_law vs =
  let scale = Scale.linear ~domain:(0., 10.) ~ticks:vs () in
  let inside v = Float.is_finite v && 0. <= v && v <= 10. in
  cover "a value outside the domain" (Array.exists (fun v -> not (inside v)) vs);
  cover "a repeated value"
    (Array.exists
       (fun v -> List.length (List.filter (( = ) v) (Array.to_list vs)) > 1)
       vs);
  let expected =
    Array.to_list vs |> List.filter inside
    |> List.sort_uniq Float.compare
    |> List.map (fun v -> v /. 10.)
    |> Array.of_list
  in
  let ticks =
    rows_of
      [ Mark.bind Role.x (num ~scale (f64 [| 1. |])) ]
      (fun r -> Option.get (Mark.ticks r Role.x))
  in
  equal floats (unsigned_zero expected) (unsigned_zero ticks)

let mark_ticks =
  group "Mark.ticks"
    [
      test "the ticks of a reversed scale increase" (fun () ->
          let t =
            rows_of
              [
                Mark.bind Role.y
                  (num
                     ~scale:(Scale.linear ~reverse:true ())
                     (f64 [| 0.; 10. |]));
              ]
              (fun r -> Option.get (Mark.ticks r Role.y))
          in
          greater int ~than:1 (Array.length t);
          equal floats
            (Array.of_list (List.sort_uniq Float.compare (Array.to_list t)))
            t);
      prop "explicit ticks are their increasing distinct values in the domain"
        ~examples:[ [| 0.; -0.; 5.; 5. |] ]
        gen_ticks ticks_law;
    ]

(* Mark.positions, Mark.extent *)

let gen_positions =
  let value =
    Gen.frequency
      [
        (4, Gen.float_range (-2.) 6.);
        ( 1,
          Gen.of_list ~pp:Format.pp_print_float [ 0.; 4.; nan; Float.infinity ]
        );
      ]
  in
  Gen.(
    let* n = int_range 1 6 in
    pair (array ~size:(constant n) value) (array ~size:(constant n) value))

(* Law: positions are normalised and kept outside the domain, points are their
   projection inside it and [nan] elsewhere, and the extent of an x alone runs
   from zero to it; a dropped row is at [nan] in all three. On the domain [0, 4]
   normalising is a division by 4, so [near] compares. *)
let positions_law (xs, ys) =
  let rows = List.init (Array.length xs) Fun.id in
  let dropped i = not (Float.is_finite xs.(i) && Float.is_finite ys.(i)) in
  let inside i =
    let u = xs.(i) /. 4. and v = ys.(i) /. 4. in
    0. <= u && u <= 1. && 0. <= v && v <= 1.
  in
  cover "a dropped row" (List.exists dropped rows);
  cover "a row in the domain"
    (List.exists (fun i -> (not (dropped i)) && inside i) rows);
  cover "a row outside the domain"
    (List.exists (fun i -> (not (dropped i)) && not (inside i)) rows);
  let unit4 = Scale.linear ~domain:(0., 4.) () in
  let (us, vs), (px, py), (a, b), proj =
    rows_of
      [
        Mark.bind Role.x (num ~scale:unit4 (f64 xs));
        Mark.bind Role.y (num ~scale:unit4 (f64 ys));
      ]
      (fun r ->
        (Mark.positions r, Mark.points r, Mark.extent r `X, Mark.projection r))
  in
  List.iter
    (fun i ->
      let msg = Printf.sprintf "row %d" i in
      if dropped i then
        equal ~msg floats (Array.make 6 nan)
          [| us.(i); vs.(i); px.(i); py.(i); a.(i); b.(i) |]
      else begin
        let u = xs.(i) /. 4. and v = ys.(i) /. 4. in
        equal ~msg (pair near near) (u, v) (us.(i), vs.(i));
        equal ~msg (pair near near) (0., u) (a.(i), b.(i));
        if inside i then
          let p = Coord.point proj u v in
          equal ~msg (pair near near) (P2.x p, P2.y p) (px.(i), py.(i))
        else equal ~msg floats [| nan; nan |] [| px.(i); py.(i) |]
      end)
    rows

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
    ( "an end beyond the domain is kept",
      [
        Mark.bind Role.x
          (num ~scale:(Scale.linear ~domain:(0., 4.) ()) (f64 [| 2. |]));
        Mark.bind Role.x2 (num (f64 [| 6. |]));
      ],
      ([| 0.5 |], [| 1.5 |]) );
    ( "a length beyond the domain is kept",
      [
        Mark.bind Role.x
          (num ~scale:(Scale.linear ~domain:(-4., 4.) ()) (f64 [| -6. |]));
      ],
      ([| 0.5 |], [| -0.25 |]) );
    ( "an extent wholly beyond the domain is kept",
      [
        Mark.bind Role.x
          (num ~scale:(Scale.linear ~domain:(0., 4.) ()) (f64 [| 5. |]));
        Mark.bind Role.x2 (num (f64 [| 6. |]));
      ],
      ([| 1.25 |], [| 1.5 |]) );
  ]

let positions =
  group "Mark.positions"
    [
      prop "positions, points and extents agree"
        ~examples:[ ([| 0.; 1.; 2.; 3. |], [| 0.; nan; 2.; 3. |]) ]
        gen_positions positions_law;
      test "an unbound position is at 0.5" (fun () ->
          equal floats [| 0.5; 0.5 |]
            (snd
               (rows_of
                  [ Mark.bind Role.x (num (f64 [| 0.; 1. |])) ]
                  Mark.positions)));
      test "a position on a band scale is the centre of its band" (fun () ->
          equal floats [| 0.25; 0.75 |]
            (fst
               (rows_of
                  [ Mark.bind Role.x (strings [| "a"; "b" |]) ]
                  Mark.positions)));
    ]

let mark_extents =
  group "Mark.extent"
    [
      cases "an extent covers what its bindings give"
        ~id:(fun (n, _, _) -> n)
        extents
        (fun (_, bindings, expected) ->
          equal
            (pair (array near) (array near))
            expected
            (rows_of bindings (fun r -> Mark.extent r `X)));
    ]

(* Mark.project *)

let page_box =
  Testable.make
    ~pp:(Format.pp_print_option Box2.pp)
    ~equal:
      (Option.equal (fun a b ->
           List.for_all2
             (fun x y -> Float.abs (x -. y) <= 1e-9)
             (corners a) (corners b)))

let projects =
  group "Mark.project"
    [
      test "a box wholly beyond the domain is dropped" (fun () ->
          let got =
            rows_of [] (fun r ->
                Path.bounds
                  (Mark.project r (Path.rect (Box2.v 1.25 0.25 0.25 0.5))))
          in
          equal page_box None got);
      test "a box is cut at the domain's edge" (fun () ->
          let got, corner, edge =
            rows_of [] (fun r ->
                let at = Coord.point (Mark.projection r) in
                ( Path.bounds
                    (Mark.project r (Path.rect (Box2.v 0.5 0.25 1. 0.5))),
                  at 0.5 0.25,
                  at 1. 0.75 ))
          in
          equal page_box (Some (Box2.of_pts corner edge)) got);
      test "a path is cut at the domain's edges" (fun () ->
          let got, edge, start =
            rows_of [] (fun r ->
                let p = Path.polyline [| 0.5; 2. |] [| 0.5; 0.5 |] in
                ( Path.bounds (Mark.project r p),
                  Coord.point (Mark.projection r) 1. 0.5,
                  Coord.point (Mark.projection r) 0.5 0.5 ))
          in
          equal page_box (Some (Box2.of_pts start edge)) got);
    ]

(* Mark.series *)

(* Law: [series] splits the rows of a line by their index along the leading axes
   and their category in every channel on a band scale other than the positions,
   a missing category being one of its own, in order of first rows. Here [m]
   series of [n] rows each, coloured by codes of two labels, some outside them,
   beside a categorical x. *)
let gen_series =
  let open Gen in
  let* m = int_range 1 3 and+ n = int_range 1 4 in
  let+ codes = array ~size:(constant (m * n)) (int_range (-1) 2) in
  (m, n, codes)

let series_law (m, n, codes) =
  let category k = if codes.(k) = 0 || codes.(k) = 1 then codes.(k) else -1 in
  let key k = (k / n, category k) in
  let expected =
    List.fold_left
      (fun acc k ->
        match List.assoc_opt (key k) acc with
        | Some ks -> (key k, k :: ks) :: List.remove_assoc (key k) acc
        | None -> (key k, [ k ]) :: acc)
      []
      (List.init (m * n) Fun.id)
    |> List.map (fun (_, ks) -> Array.of_list (List.rev ks))
    |> List.sort (fun a b -> compare a.(0) b.(0))
  in
  cover "a missing category" (Array.exists (fun c -> c < 0 || c > 1) codes);
  cover "a series split by its stroke" (List.length expected > m);
  let got =
    rows_of
      [
        Mark.bind Role.x (strings (Array.init n string_of_int));
        Mark.bind Role.y (num (Nx.zeros Nx.float64 [| m; n |]));
        Mark.bind Role.stroke
          (cat ~labels:[| "p"; "q" |]
             (Nx.create Nx.int32 [| m; n |] (Array.map Int32.of_int codes)));
      ]
      (fun r -> List.map Mark.index (Mark.series r))
  in
  equal (list (array int)) expected got

let mark_series =
  group "Mark.series"
    [
      prop "series split rows by leading axes and categories" gen_series
        series_law;
    ]

let warns =
  group "Mark.warn"
    [
      test "a warning is given under the mark's id" (fun () ->
          let m =
            Mark.v ~name:"probe" [] (fun r ->
                Mark.warn r "no luck";
                Picture.empty)
          in
          equal
            (list (pair id string))
            [ (mark_id, "no luck") ]
            (Drawing.warnings (drawn (layer [ m ]))));
    ]

(* Reducers *)

(* [same_raster page a b] states that the pictures [a] and [b] draw the same
   pixels on [page]. *)
let same_raster (w, h) a b =
  let r p = Raster.render ~density:1. (Renderable.v w h p) in
  equal (array int) (Nx.to_array (r a)) (Nx.to_array (r b))

(* [wrapped_image wrap] is the pixels drawn of a transparent 200 by 300 image
   filling a panel of 20 by 20 points under [wrap]. *)
let wrapped_image wrap =
  let px = Nx.zeros Nx.uint8 [| 200; 300; 4 |] in
  let f =
    Mark.v ~name:"picture" [] (fun r ->
        let at = Coord.point (Mark.projection r) in
        let box = Box2.of_pts (at 0. 1.) (at 1. 0.) in
        wrap box (Picture.image box px))
  in
  match images (drawn ~size:(Size.panels 20. 20.) f) with
  | [ (_, got) ] -> got
  | l -> failf "%d images" (List.length l)

let gathering =
  group "Mark reducers"
    [
      test "a large image is gathered, drawing what the whole image draws"
        (fun () ->
          let px =
            Nx.init Nx.uint8 [| 256; 256; 3 |] (fun i ->
                ((i.(0) * 3) + (i.(1) * 5) + (i.(2) * 70)) mod 256)
          in
          let f = image px in
          let size = Size.panels 20. 20. in
          let l = layout size (resolve f) in
          let box = (List.hd (Layout.panels l)).box in
          match images (drawn ~size f) with
          | [ (window, gathered) ] ->
              less int ~than:(256 * 256) (Nx.numel gathered / 4);
              same_raster (Layout.size l) (Picture.image box px)
                (Picture.image window gathered)
          | l -> failf "%d images" (List.length l));
      cases "the large image of any mark is gathered, drawing what it draws"
        ~id:fst
        [ ("grey", 1); ("RGB", 3); ("RGBA", 4) ]
        (fun (_, c) ->
          let px =
            Nx.init Nx.uint8 [| 200; 300; c |] (fun i ->
                ((i.(0) * 3) + (i.(1) * 5) + (i.(2) * 70)) mod 256)
          in
          (* A box off the device pixels, so the gather has samples outside. *)
          let placed r =
            let at = Coord.point (Mark.projection r) in
            Box2.of_pts (at 0.13 0.87) (at 0.71 0.21)
          in
          let box = ref None in
          let f =
            Mark.v ~name:"picture" [] (fun r ->
                box := Some (placed r);
                Picture.image (placed r) px)
          in
          let size = Size.panels 20. 20. in
          let d = drawn ~size f in
          let l = layout size (resolve f) in
          match images d with
          | [ (window, gathered) ] ->
              let s = Nx.shape gathered in
              less int ~than:(200 * 300) (s.(0) * s.(1));
              same_raster (Layout.size l)
                (Picture.image (Option.get !box) px)
                (Picture.image window gathered)
          | l -> failf "%d images" (List.length l));
      cases "an image under a clip, an opacity or a tag is gathered" ~id:fst
        [
          ("clip", fun box p -> Picture.clip (Path.rect box) p);
          ("opacity", fun _ p -> Picture.opacity 0.5 p);
          ( "tag",
            fun _ p ->
              Picture.tag
                { Picture.id = Nx.Ptree.Path.root; rows = Picture.Rows [||] }
                p );
        ]
        (fun (_, wrap) ->
          let s = Nx.shape (wrapped_image wrap) in
          less int ~than:(200 * 300) (s.(0) * s.(1)));
      cases "an image under a transform or a stamp is left whole" ~id:fst
        [
          ( "transform",
            fun _ p -> Picture.transform (Affine.translate 0.5 0.5) p );
          ("stamp", fun _ p -> Picture.stamp [| 0.5 |] [| 0.5 |] p);
        ]
        (fun (_, wrap) ->
          equal (array int) [| 200; 300; 4 |] (Nx.shape (wrapped_image wrap)));
    ]

(* Mark.m4 *)

(* [m4_reference column ys rows] is what M4 keeps of the rows [rows] of one
   series in one panel, given in order: of each run of consecutive rows in one
   column, [column i], its first and last rows and the first rows of its lowest
   and highest values; of each run of rows dropped, where [ys] is [nan], its
   first row. *)
let m4_reference column ys rows =
  let gap = -1 in
  let kept = ref [] and bin = ref None in
  let flush () =
    match !bin with
    | None -> ()
    | Some (key, first, _, _, _) when key = gap -> kept := first :: !kept
    | Some (_, first, final, lo, hi) ->
        kept := first :: final :: lo :: hi :: !kept
  in
  List.iter
    (fun i ->
      let key = if Float.is_nan ys.(i) then gap else column i in
      match !bin with
      | Some (k, first, _, lo, hi) when k = key ->
          let lo = if ys.(i) < ys.(lo) then i else lo
          and hi = if ys.(i) > ys.(hi) then i else hi in
          bin := Some (k, first, i, lo, hi)
      | _ ->
          flush ();
          bin := Some (key, i, i, i, i))
    rows;
  flush ();
  Array.of_list (List.sort_uniq Int.compare !kept)

(* [columns ~density box domain] maps a quantity on a linear x scale of [domain]
   to its column in a panel of [box]: the device pixel it lies in, all those
   beyond the panel's device pixels on one side being one, cut at the domain's
   ends. The domain's ends lie in it, and a quantity on another edge lies after
   it. *)
let columns ~density box (a, b) v =
  let x = Box2.minx box +. ((v -. a) /. (b -. a) *. Box2.w box) in
  let first = Float.floor (Box2.minx box *. density)
  and last = Float.ceil (Box2.maxx box *. density) in
  let pixel = Float.floor (x *. density) in
  let pixel = Float.min last (Float.max (first -. 1.) pixel) in
  let side = if v < a then 0 else if v <= b then 1 else 2 in
  (3 * Float.to_int pixel) + side

(* [m4_kept ~size ~facets x ys] is the box of each panel of a mark of [x], [ys]
   and [facets] reduced by M4, and the rows its draw function reads in each, in
   the panels' order. The x scale's domain is [(0, 1000)]. *)
let m4_kept ?(size = Size.panels 50. 50.) ?(facets = []) x ys =
  let x = num ~scale:(Scale.linear ~domain:(0., 1000.) ()) x in
  let m, seen =
    probe ~reduce:Mark.m4
      ([ Mark.bind Role.x x; Mark.bind Role.y (num ys) ] @ facets)
      Mark.index
  in
  ignore (drawn ~size m);
  (panel_boxes size m, List.rev !seen)

(* [with_gaps y] is [y] with dropped runs of 1 to 5 rows, one of them first and
   one last. *)
let with_gaps y =
  let a = Nx.to_array y in
  let n = Array.length a in
  List.iter
    (fun (at, len) ->
      for i = at to Int.min (n - 1) (at + len - 1) do
        a.(i) <- nan
      done)
    [ (0, 2); (97, 1); (250, 5); (251 + (n / 3), 3); (n - 1, 1) ];
  f64 a

(* [m4_series (_, data)] states that M4 keeps of the one series [data ()] what
   the reference keeps, on the x domain [0, 1000] at density 1. *)
let m4_series (_, data) =
  let x, y = data () in
  let n = (Nx.shape x).(0) in
  match m4_kept x y with
  | [ box ], [ kept ] ->
      let x = Nx.to_array x in
      equal (array int)
        (m4_reference
           (fun i -> columns ~density:1. box (0., 1000.) x.(i))
           (Nx.to_array y) (List.init n Fun.id))
        kept
  | boxes, kept ->
      failf "%d panels, %d draws" (List.length boxes) (List.length kept)

let m4_inputs =
  let n = 1001 in
  let turning () =
    Nx.init Nx.float64 [| n |] (fun i ->
        Float.of_int (Int.min i.(0) (n - 1 - i.(0))) *. 2.)
  in
  [
    ("x rising", fun () -> (xs n, ys n));
    ( "x beyond the domain's ends",
      fun () -> (Nx.linspace Nx.float64 (-300.) 1300. 2001, ys 2001) );
    ("dropped runs", fun () -> (xs n, with_gaps (ys n)));
    ("x turning back", fun () -> (turning (), ys n));
  ]

(* Two facet panels of three series each, on the middle axis, so that a panel's
   series are not consecutive. *)
let m4_facets () =
  let n = 1001 and per = 3 in
  let x = Nx.broadcast_to [| per; 2; n |] (xs n) in
  let y = Nx.reshape [| per; 2; n |] (ys (2 * per * n)) in
  let boxes, kept = m4_kept x y ~facets:[ Mark.bind Role.fx (dim 1) ] in
  let x = Nx.to_array (xs n) and y = Nx.to_array y in
  let expected =
    List.mapi
      (fun p box ->
        Array.concat
          (List.init per (fun j ->
               let s = (j * 2) + p in
               m4_reference
                 (fun i -> columns ~density:1. box (0., 1000.) x.(i mod n))
                 y
                 (List.init n (fun i -> (s * n) + i)))))
      boxes
  in
  equal int 2 (List.length boxes);
  equal (list (array int)) expected kept

(* One series whose rows alternate between two facet panels by blocks of 100,
   with a dropped run in each panel. *)
let m4_split_facets () =
  let n = 2000 and block = 100 in
  let panel i = i / block mod 2 in
  let x = xs n and y = Nx.to_array (ys n) in
  List.iter (fun i -> y.(i) <- nan) [ 150; 151; 260 ];
  let side =
    Nx.create Nx.int32 [| n |] (Array.init n (fun i -> Int32.of_int (panel i)))
  in
  let boxes, kept =
    m4_kept x (f64 y) ~facets:[ Mark.bind Role.fx (cat side) ]
  in
  let x = Nx.to_array x in
  let expected =
    List.mapi
      (fun k box ->
        m4_reference
          (fun i -> columns ~density:1. box (0., 1000.) x.(i))
          y
          (List.filter (fun i -> panel i = k) (List.init n Fun.id)))
      boxes
  in
  equal int 2 (List.length boxes);
  equal (list (array int)) expected kept

(* Inked pixels *)

(* [stray a b] is the pixels that one of the raster images [a] and [b] inks with
   no pixel the other inks among the 3 by 3 they centre. A pixel is inked if its
   alpha is not [0]. *)
let stray a b =
  let s = Nx.shape a in
  let h = s.(0) and w = s.(1) in
  let a = Nx.to_array a and b = Nx.to_array b in
  let inked t i j = t.((((i * w) + j) * 4) + 3) > 0 in
  let near t i j =
    let found = ref false in
    for i' = Int.max 0 (i - 1) to Int.min (h - 1) (i + 1) do
      for j' = Int.max 0 (j - 1) to Int.min (w - 1) (j + 1) do
        if inked t i' j' then found := true
      done
    done;
    !found
  in
  let acc = ref [] in
  for i = h - 1 downto 0 do
    for j = w - 1 downto 0 do
      if (inked a i j && not (near b i j)) || (inked b i j && not (near a i j))
      then acc := (i, j) :: !acc
    done
  done;
  !acc

(* A trace: [n] rows over a panel [width] points wide drawn at [density] with a
   pen [pen] points wide along [curve]; [along] says whether x rises, turns back
   halfway or goes back and forth 7 times, [zoom] that the x domain holds the
   middle 80% of the data, and [facets] that the trace is two facet panels of
   two series each. *)
type trace = {
  n : int;
  width : float;
  density : float;
  pen : float;
  curve : Curve.t;
  signal : [ `Noise | `Walk | `Sine ];
  gaps : bool;
  along : [ `Rising | `Turning | `Waving ];
  zoom : bool;
  facets : bool;
}

let pp_trace ppf t =
  Format.fprintf ppf
    "{ n = %d; width = %g; density = %g; pen = %g; curve = %a; signal = %s; \
     gaps = %b; along = %s; zoom = %b; facets = %b }"
    t.n t.width t.density t.pen Curve.pp t.curve
    (match t.signal with
    | `Noise -> "noise"
    | `Walk -> "walk"
    | `Sine -> "sine")
    t.gaps
    (match t.along with
    | `Rising -> "rising"
    | `Turning -> "turning"
    | `Waving -> "waving")
    t.zoom t.facets

let gen_trace =
  let open Gen in
  with_pp pp_trace
    (map
       (fun ( (n, width, density),
              (pen, curve, signal),
              (gaps, along, zoom, facets) ) ->
         { n; width; density; pen; curve; signal; gaps; along; zoom; facets })
       (triple
          (triple (int_range 600 3_000)
             (of_list [ 50.; 50.37; 33.3 ])
             (of_list [ 1.; 1.3; 2. ]))
          (triple
             (of_list [ 0.5; 2.; 7. ])
             (of_list Curve.[ linear; step_after; step_before; step_mid ])
             (of_list [ `Noise; `Walk; `Sine ]))
          (quad bool (of_list [ `Rising; `Turning; `Waving ]) bool bool)))

(* Two traces that meet every cover: a thin pen over dropped rows waving, the
   largest trace, and a wide pen zoomed over facets. *)
let trace_examples =
  [
    {
      n = 6_000;
      width = 33.3;
      density = 1.3;
      pen = 0.5;
      curve = Curve.step_mid;
      signal = `Walk;
      gaps = true;
      along = `Waving;
      zoom = false;
      facets = false;
    };
    {
      n = 3_000;
      width = 50.37;
      density = 2.;
      pen = 7.;
      curve = Curve.linear;
      signal = `Noise;
      gaps = false;
      along = `Turning;
      zoom = true;
      facets = true;
    };
  ]

(* [line_at ~pen ~curve ~whole x y] is a line of [x] and [y] stroked at [pen]
   along [curve]. When [whole], its stroke is bound to one category per row, a
   channel that varies along the series, so that the line is not reduced, yet
   draws one series in one colour. *)
let line_at ?fx ~pen ~curve ~whole x y =
  let shape = if whole then Nx.shape y else [| 1 |] in
  let stroke = cat (Nx.zeros Nx.int32 shape) in
  line ?fx ~x ~y:(num y) ~stroke ~width:(const pen) ~curve ()

let inks_alike t =
  cover "a line thinner than a column" (t.pen *. t.density < 1.);
  cover "a line wider than four device pixels" (t.pen *. t.density > 4.);
  cover "dropped rows" t.gaps;
  cover "x going back and forth" (t.along = `Waving);
  cover "x beyond the domain" t.zoom;
  cover "facet panels" t.facets;
  let y =
    match t.signal with
    | `Noise -> ys t.n
    | `Walk -> Nx.cumsum (ys t.n)
    | `Sine -> Nx.sin (Nx.linspace Nx.float64 0. 6. t.n)
  in
  let y = if t.gaps then with_gaps y else y in
  let x =
    let at f = Nx.init Nx.float64 [| t.n |] (fun i -> f i.(0)) in
    match t.along with
    | `Rising -> xs t.n
    | `Turning -> at (fun i -> Float.of_int (Int.min i (t.n - i)))
    | `Waving ->
        let period = t.n / 7 in
        at (fun i ->
            let k = i mod period in
            Float.of_int (Int.min k (period - k)))
  in
  let hi = Nx.item [] (Nx.max x) in
  let x, y, fx =
    if not t.facets then (x, y, None)
    else
      let shape = [| 2; 2; t.n |] in
      let ys = [ y; Nx.neg y; Nx.mul_s y 0.5; Nx.add_s y 1. ] in
      ( Nx.broadcast_to shape x,
        Nx.reshape shape (Nx.stack ~axis:0 ys),
        Some (dim 0) )
  in
  let x =
    if t.zoom then num ~scale:(Scale.linear ~domain:(0.1 *. hi, 0.9 *. hi) ()) x
    else num x
  in
  let page ~whole =
    let f =
      layer
        [
          line_at ?fx ~pen:t.pen ~curve:t.curve ~whole x y;
          axis ~show:false "x";
          axis ~show:false "y";
        ]
    in
    let d =
      drawn ~theme:transparent ~density:t.density
        ~size:(Size.panels t.width 50.) f
    in
    (* The whole line's stroke varies along it by shape, so m4 leaves it whole:
       its reference draws every row. *)
    if whole then
      equal ~msg:"rows of the whole line" int (Nx.numel y)
        (List.fold_left
           (fun acc -> function
             | Picture.Rows rows, _ -> acc + Array.length rows | _ -> acc)
           0 (tags mark_id d));
    let r = Drawing.renderable d in
    (* Two device pixels of margin on every side, so that no ink is cut by the
       page's edge. *)
    let m = 2. /. t.density in
    Raster.render ~density:t.density
      (Renderable.v
         (Renderable.w r +. (2. *. m))
         (Renderable.h r +. (2. *. m))
         (Picture.transform (Affine.translate m m) (Renderable.picture r)))
  in
  equal (list (pair int int)) [] (stray (page ~whole:true) (page ~whole:false))

let m4s =
  group "Mark.m4"
    [
      test "m4 keeps every row at four rows per column" (fun () ->
          let _, kept = m4_kept (xs 200) (ys 200) in
          equal (list int) [ 200 ] (List.map Array.length kept));
      cases
        "m4 keeps the first, last, lowest and highest rows of each run in a \
         column, and the first of each dropped run"
        ~id:fst m4_inputs m4_series;
      test "m4 keeps every row when x is in no order" (fun () ->
          let n = 1001 in
          let x = Nx.Rng.uniform ~key:(Nx.Rng.key 7) Nx.float64 [| n |] in
          (* A few columns, so that some runs have three rows or more. *)
          let size = Size.panels 5. 50. in
          let _, kept = m4_kept ~size (Nx.mul_s x 1000.) (ys n) in
          equal (list int) [ n ] (List.map Array.length kept));
      test "m4 reduces the series of each facet panel" m4_facets;
      test "m4 joins a series' rows across the rows of other facet panels"
        m4_split_facets;
      test "m4 reduces a line over instants as over its normalised values"
        (fun () ->
          let n = 1001 in
          let counts = Nx.cast Nx.int64 (Nx.arange Nx.int32 0 n 1) in
          let ys = ys n in
          let kept x =
            let m, seen =
              probe ~reduce:Mark.m4
                [ Mark.bind Role.x x; Mark.bind Role.y (num ys) ]
                Mark.index
            in
            ignore (drawn ~size:(Size.panels 50. 50.) m);
            !seen
          in
          let span = (Time.v Ms 0L, Time.v Ms (Int64.of_int (n - 1))) in
          let on_time =
            kept (time ~scale:(Scale.time ~domain:span ()) Ms counts)
          and on_line =
            kept
              (num
                 ~scale:(Scale.linear ~domain:(0., Float.of_int (n - 1)) ())
                 counts)
          in
          less int ~than:n (Array.length (List.hd on_time));
          equal (list (array int)) on_line on_time);
      prop "m4 inks what the whole line inks, within a pixel" ~count:40
        ~examples:trace_examples gen_trace inks_alike;
    ]

(* Mark.cells *)

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

let heatmap ?opacity z =
  layer [ rect ~x:(dim 1) ~y:(dim 0) ~fill:(num z) ?opacity () ]

let z23 () = Nx.create Nx.float64 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |]

(* [cells d] is the box and pixels of each cells image of the mark [0] of
   [d]. *)
let cells d =
  List.filter_map
    (function
      | Picture.Cells _, Picture.Image i -> Some (i.box, i.pixels) | _ -> None)
    (tags mark_id d)

let one_image = function
  | [ (_, px) ] -> px
  | l -> failf "%d images" (List.length l)

let cells_group =
  group "Mark.cells"
    [
      test "cells paints each cell with its row's fill" (fun () ->
          let f = heatmap (z23 ()) in
          let r = resolve f in
          let at i j = viridis_of r (Float.of_int ((i * 3) + j)) in
          equal (array int)
            (Nx.to_array (cells_image 2 3 at))
            (Nx.to_array (one_image (cells (drawn f)))));
      test "cells paints each cell at its row's opacity" (fun () ->
          let z = z23 () in
          let opacity = num ~scale:(Scale.linear ~domain:(0., 5.) ()) z in
          let f = heatmap ~opacity z in
          let r = resolve f in
          let at i j =
            let k = Float.of_int ((i * 3) + j) in
            let c = viridis_of r k in
            Color.with_alpha (Color.alpha c *. (k /. 5.)) c
          in
          equal (array int)
            (Nx.to_array (cells_image 2 3 at))
            (Nx.to_array (one_image (cells (drawn f)))));
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
          equal int 0 (Nx.item [ 1; 1; 3 ] (one_image (cells (drawn f)))));
      test "rows that share a cell draw as rectangles" (fun () ->
          let f =
            layer
              [
                rect ~x:(strings [| "a"; "a" |]) ~y:(strings [| "p"; "p" |]) ();
              ]
          in
          let d = drawn f in
          equal (pair int int) (0, 2)
            (List.length (cells d), List.length (fills mark_id d)));
      test "a heatmap on a padded band scale draws rectangles" (fun () ->
          let f =
            layer
              [
                rect
                  ~x:(dim ~scale:(Scale.band ~padding:0.1 ()) 1)
                  ~y:(dim 0)
                  ~fill:(num (z23 ()))
                  ();
              ]
          in
          let d = drawn f and r = resolve f in
          equal
            (pair int (list color))
            (0, List.init 6 (fun k -> viridis_of r (Float.of_int k)))
            ( List.length (cells d),
              List.map (fun (_, c, _) -> c) (fills mark_id d) ));
      test "a gathered heatmap draws what the whole image draws" (fun () ->
          let h = 300 and w = 400 in
          let z =
            Nx.init Nx.float64 [| h; w |] (fun i ->
                Float.of_int ((i.(0) * 7) + (i.(1) * 13 mod 101)))
          in
          let f = heatmap z in
          let size = Size.panels 40. 30. in
          let r = resolve f in
          let l = layout size r in
          match cells (drawn ~size f) with
          | [ (window, px) ] ->
              less int ~than:(h * w) (Nx.numel px / 4);
              let box = (List.hd (Layout.panels l)).box in
              let whole =
                cells_image h w (fun i j -> viridis_of r (Nx.item [ i; j ] z))
              in
              same_raster (Layout.size l) (Picture.image box whole)
                (Picture.image window px)
          | l -> failf "%d images" (List.length l));
    ]

(* Mark.raster *)

(* [dots_with ~alpha reduce n] is a mark of [n] dots filled at [alpha], reduced
   by [reduce]. *)
let dots_with ?(alpha = 0.3) reduce n =
  let e = Nx.Rng.uniform ~key:(Nx.Rng.key 4) Nx.float64 [| n; 2 |] in
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
   density 2, fewer than the dots, so raster applies at both. The 18 cases are
   checked whole. *)
let raster_as_drawn (alpha, density, n) =
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
  within_one ~msg:"pixels" expected got

let rasters =
  group "Mark.raster"
    [
      prop "raster draws what the picture paints, within a level" ~count:30
        (Gen.triple
           (Gen.of_list ~pp:Format.pp_print_float [ 0.02; 0.3; 1. ])
           (Gen.of_list ~pp:Format.pp_print_float [ 1.; 2. ])
           (Gen.of_list ~pp:Format.pp_print_int [ 1_601; 2_400; 4_000 ]))
        raster_as_drawn;
      cases "raster draws dots as one image past its thresholds"
        ~id:(fun (n, w, _) -> Printf.sprintf "%d dots in %g by %g points" n w w)
        [
          (20_000, 400., false);
          (20_001, 400., true);
          (100, 10., false);
          (101, 10., true);
        ]
        (fun (n, w, image) ->
          let d =
            drawn ~size:(Size.panels w w) (dots_with (Some Mark.raster) n)
          in
          equal (pair int int)
            (if image then (1, 0) else (0, 1))
            (List.length (images d), List.length (stamps d)));
      test
        "raster paints the panel's picture at the density, over the device \
         pixels it reaches" (fun () ->
          let density = 2. and size = Size.panels 60.3 40.7 in
          let pictures = ref [] in
          let uniform k =
            num (Nx.Rng.uniform ~key:(Nx.Rng.key k) Nx.float64 [| 25_000 |])
          in
          let f =
            layer
              [
                Mark.v ~name:"dots" ~reduce:Mark.raster
                  [ Mark.bind Role.x (uniform 5); Mark.bind Role.y (uniform 6) ]
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
    ]

(* Mark.broadcast, Mark.v *)

let broadcast_law (shape, spec) =
  let b = bound spec in
  equal (array int)
    (rows_of ~shape [ b ] Mark.shape)
    (Mark.broadcast ~shape [ b ])

(* [swatch_fills fill] is the fill of each swatch of the size legend of a mark
   whose fill is [fill], if bound. *)
let swatch_fills fill =
  let sizes = path [ Field "legend"; Field "size"; Field "num" ] in
  let seen = ref [] in
  let m =
    Mark.v ~name:"probe"
      ~swatch:(fun rows ->
        if Nx.Ptree.Path.equal (Mark.id rows) sizes then
          seen := Mark.get rows Role.fill :: !seen;
        Picture.empty)
      (Mark.bind Role.size (num (f64 [| 1.; 2.; 3.; 4. |]))
      :: Option.to_list (Option.map (Mark.bind Role.fill) fill))
      (fun _ -> Picture.empty)
  in
  ignore (drawn m);
  List.rev !seen

let neutral =
  let ink = Theme.ink Theme.default in
  Color.with_alpha (0.5 *. Color.alpha ink) ink

let marks_v =
  group "Mark.v"
    [
      cases "a mark's shape broadcasts the shape given" ~id:fst
        [
          ("constants", ([| 2; 3 |], [], [| 2; 3 |]));
          ( "with a channel",
            ( [| 2; 1 |],
              [ Mark.bind Role.x (num (f64 [| 0.; 1.; 2. |])) ],
              [| 2; 3 |] ) );
          ("none given", ([||], [ Mark.bind Role.x (const 0.5) ], [||]));
        ]
        (fun (_, (shape, bindings, expected)) ->
          equal
            (pair (array int) int)
            (expected, numel expected)
            (rows_of ~shape bindings (fun r -> (Mark.shape r, Mark.length r))));
      test "a mark without rows is drawn in no panel" (fun () ->
          let m, seen =
            probe [ Mark.bind Role.x (num (f64 [||])) ] Mark.index
          in
          ignore (drawn m);
          equal (list (array int)) [] !seen);
      test "a stamp of one instance per row is tagged instance by instance"
        (fun () ->
          let z = f64 [| 1.; 2.; 3. |] in
          match
            tags mark_id (drawn (layer [ dot ~x:(num z) ~y:(num z) () ]))
          with
          | [ (Picture.Rows rows, Picture.Stamp _) ] ->
              equal (array int) [| 0; 1; 2 |] rows
          | l -> failf "%d tags, not one tagged stamp" (List.length l));
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
          equal
            (list (triple (array int) (array int) id))
            (List.init 3 (fun k -> ([| 3 |], [| k |], legend)))
            (List.rev !swatches));
      cases
        "a swatch's colour bound to other data is the ink at half its opacity, \
         a constant keeps its own"
        ~id:(fun (n, _, _) -> n)
        [
          ( "data the legend does not show",
            Some (strings [| "a"; "b"; "a"; "b" |]),
            Some [| neutral |] );
          ("a constant", Some (const Color.red), Some [| Color.red |]);
          ("no binding", None, None);
        ]
        (fun (_, fill, expected) ->
          let fills = swatch_fills fill in
          greater int ~than:0 (List.length fills);
          equal
            (list (option (array color)))
            (List.map (fun _ -> expected) fills)
            fills);
    ]

(* Stages *)

let pair_figure b =
  grid [ [ dot ~x:(num (f64 [| 1.; 2. |])) ~y:(num (f64 [| 1.; 2. |])) (); b ] ]

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

let unread = View.(set (number "unread" ~init:0.) 1. empty)

let drawings =
  group "Drawing"
    [
      test "the page is the layout's size" (fun () ->
          let l = layout (Size.panels 80. 50.) (resolve (other ())) in
          let r = Drawing.renderable (draw ~density:1. l) in
          equal
            (pair float_exact float_exact)
            (Layout.size l)
            (Renderable.w r, Renderable.h r));
      test "a drawing with images drawn again is equal to the first" (fun () ->
          let l =
            layout (Size.panels 40. 40.)
              (resolve (image (Nx.ones Nx.uint8 [| 3; 3; 3 |])))
          in
          equal drawing (draw ~density:1. l) (draw ~density:1. l));
      test "drawings of other data differ" (fun () ->
          not_equal drawing
            (drawn (other ()))
            (drawn (line ~y:(num (f64 [| 1.; 2. |])) ())));
      test "the warnings of resolving are the drawing's" (fun () ->
          let r = resolve ~view:unread (other ()) in
          let d = draw ~density:1. (layout (Size.panels 100. 100.) r) in
          greater int ~than:0 (List.length (Resolved.warnings r));
          equal
            (list (pair id string))
            (Resolved.warnings r) (Drawing.warnings d));
    ]

let draws =
  group "draw"
    [
      cases "draw raises on a density not finite and positive"
        ~id:Float.to_string [ 0.; -1.; nan; Float.infinity ] (fun density ->
          raises_match Exn.invalid_arg (fun () ->
              draw ~density (layout (Size.panels 10. 10.) (resolve (other ())))));
      cases "draw ~prev is the drawing drawn afresh" ~id:fst reuse
        (fun (_, f) ->
          let reused, fresh = f () in
          equal drawing fresh reused);
    ]

(* Output *)

let read_file p = In_channel.with_open_bin p In_channel.input_all

(* [chunk name data] is the bytes after the PNG chunk type [name] in [data]. *)
let chunk name data =
  let rec find i =
    if i + 4 > String.length data then failf "no %s chunk" name
    else if String.sub data i 4 = name then i + 4
    else find (i + 1)
  in
  let at = find 0 in
  String.sub data at (String.length data - at)

let u32 s i = Int32.to_int (String.get_int32_be s i)

let saves =
  group "save"
    [
      test "another extension raises before the file is created" (fun () ->
          let file = Filename.concat (temp_dir ()) "f.jpg" in
          raises_match Exn.invalid_arg (fun () -> save file (other ()));
          satisfies ~claim:"absent" string
            (fun f -> not (Sys.file_exists f))
            file);
      cases "save writes the format its extension names, ignoring case" ~id:fst
        [ ("f.png", "\x89PNG"); ("f.SVG", "<svg"); ("f.pdf", "%PDF") ]
        (fun (name, magic) ->
          let file = Filename.concat (temp_dir ()) name in
          save file (other ());
          let data = read_file file in
          contains ~sub:magic
            (String.sub data 0 (Int.min 200 (String.length data))));
      test "a PNG is 720 by 480 pixels by default" (fun () ->
          let file = Filename.concat (temp_dir ()) "f.png" in
          save file (other ());
          let s = Nx.shape (Nx_io.load_image file) in
          equal (pair int int) (720, 480) (s.(1), s.(0)));
      test "a PNG records its density in pixels per metre" (fun () ->
          (* 3 pixels a point are 3 × 72 / 0.0254 = 8503.9 a metre, unit 1. *)
          let file = Filename.concat (temp_dir ()) "f.png" in
          save ~density:3. file (other ());
          let p = chunk "pHYs" (read_file file) in
          equal (triple int int int) (8504, 8504, 1)
            (u32 p 0, u32 p 4, Char.code p.[8]));
      test "a file that cannot be written raises Sys_error" (fun () ->
          let file =
            Filename.concat (Filename.concat (temp_dir ()) "absent") "f.svg"
          in
          raises_match Exn.sys_error (fun () -> save file (other ())));
      test "a PNG records the sRGB colour space" (fun () ->
          let file = Filename.concat (temp_dir ()) "f.png" in
          save file (other ());
          contains ~sub:"sRGB" (read_file file));
      test "save gives the drawing's warnings, in order, before writing"
        (fun () ->
          let file = Filename.concat (temp_dir ()) "f.svg" in
          let f =
            layer
              [
                dot
                  ~x:(num (f64 [| 1. |]))
                  ~y:(num (f64 [| 1. |]))
                  ~fill:(cat ~labels:[| "a" |] (i32 [| 3 |]))
                  ();
                other ();
              ]
          in
          let seen = ref [] in
          save ~view:unread
            ~warn:(fun w ->
              satisfies ~claim:"not written yet" string
                (fun f -> not (Sys.file_exists f))
                file;
              seen := w :: !seen)
            file f;
          let expected =
            Drawing.warnings
              (draw ~density:2.
                 (layout (Size.figure 360. 240.) (resolve ~view:unread f)))
          in
          greater int ~than:1 (List.length expected);
          equal (list (pair id string)) expected (List.rev !seen));
    ]

let pps =
  group "pp"
    [
      test "pp prints a one-line summary that counts the warnings" (fun () ->
          let warned =
            dot
              ~x:(num (f64 [| 1.; 2. |]))
              ~y:(num (f64 [| 1.; 2. |]))
              ~fill:(cat ~labels:[| "a" |] (i32 [| 0; 3 |]))
              ()
          in
          expect (Format.asprintf "%a@.%a" pp (other ()) pp warned)
          @@ __POS_OF__
               {|
                 hugin figure
                 hugin figure (1 warning)
               |});
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
          equal (list text)
            [ "quill.display\nimage/svg+xml\n\n" ^ read_file file ]
            !tags);
    ]

(* Goldens *)

(* Each benchmark figure's PNG, written by gen/gen.exe, is a baseline within one
   colour level: arm64 fuses multiply-adds that x86-64 rounds twice, and libm's
   last bits differ, so rasters differ by a level between systems. Its SVG is an
   [expect_file] baseline, the same bytes everywhere, except for the dense
   figure's, which embeds a raster. *)
let goldens =
  let open Hugin_test_figures in
  let png (name, _, _, _) r =
    within_one ~msg:name
      (Nx_io.load_image (Filename.concat "golden" (name ^ ".png")))
      (Raster.render ~density:Figures.density r)
  in
  group "goldens"
    [
      cases "a benchmark figure draws its baselines"
        ~id:(fun (n, _, _, _) -> n)
        Figures.goldens
        (fun ((name, _, _, _) as g) ->
          let r = Drawing.renderable (Figures.drawing g) in
          expect_file
            ~file:(Printf.sprintf "packages/hugin/test/golden/%s.svg" name)
            (Svg.render r);
          png g r);
      test "the dense figure draws its PNG baseline" (fun () ->
          png Figures.dense (Drawing.renderable (Figures.drawing Figures.dense)));
    ]

let () =
  exit
    (run "Draw"
       ([ missing_values; roles ] @ channels
       @ [
           dots;
           lines;
           areas;
           rects;
           frames;
           rules;
           ablines;
           texts_marks;
           images_marks;
           contours;
           scales;
           guides;
           facets;
           axes;
           legends_figures;
           themes;
           binds;
           mark_rows;
           gets;
           ranges;
           mark_scales;
           mark_ticks;
           positions;
           mark_extents;
           projects;
           mark_series;
           warns;
           gathering;
           m4s;
           cells_group;
           rasters;
           group "Mark.broadcast"
             [
               prop "the shape of a mark drawn is Mark.broadcast's" gen_channel
                 broadcast_law;
             ];
           marks_v;
           drawings;
           draws;
           saves;
           pps;
           goldens;
         ]))
