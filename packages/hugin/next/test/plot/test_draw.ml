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
let dash = Testable.make ~pp:Dash.pp ~equal:Dash.equal
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
      cases ~name:fst "a mark's shape broadcasts the shape given"
        [
          ("constants", ([| 2; 3 |], [], [| 2; 3 |]));
          ( "with a channel",
            ( [| 2; 1 |],
              [ Mark.bind Role.x (num (f64 [| 0.; 1.; 2. |])) ],
              [| 2; 3 |] ) );
          ("none given", ([||], [ Mark.bind Role.x (const 0.5) ], [||]));
        ]
        (fun (_, (shape, bindings, expected)) ->
          let seen = ref [] in
          let m =
            Mark.v ~name:"probe" ~shape bindings (fun r ->
                seen := (Mark.shape r, Mark.length r) :: !seen;
                Picture.empty)
          in
          ignore (drawn m);
          equal
            (pair (array int) int)
            (expected, Array.fold_left ( * ) 1 expected)
            (only seen));
      test "get gives a parameter's constant per row, under its role only"
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

(* Channels on a mark's axes *)

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

let numel s = Array.fold_left ( * ) 1 s

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
  | Floats_of n -> Mark.bind Role.text (Hugin_next.floats (values n))
  | Index_of k -> Mark.bind Role.text (index k)
  | Dim_of k -> Mark.bind Role.text (dim k)
  | Constant -> Mark.bind Role.text (const (Text.v "c"))

(* [varies_of shape spec a] is [varies] of the channel [spec] describes. *)
let varies_of shape spec a =
  match spec with
  | Tensor s -> varies shape (num (distinct s)) a
  | Codes s -> varies shape (cat (codes s)) a
  | Strings_of n -> varies shape (strings (labels n)) a
  | Floats_of n -> varies shape (Hugin_next.floats (values n)) a
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
   different values, and [Mark.broadcast] is the shape of the mark drawn. *)
let varies_law (shape, spec) =
  let b = bound spec in
  let drawn_shape, texts =
    rows_of ~shape [ b ] (fun r ->
        (Mark.shape r, Option.get (Mark.get r Role.text)))
  in
  equal (array int) drawn_shape (Mark.broadcast ~shape [ b ]);
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
  group "Channels"
    [
      prop "varies is whether rows differ along an axis" gen_channel varies_law;
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
      test "project drops a box wholly beyond the domain" (fun () ->
          let got =
            rows_of [] (fun r ->
                Path.bounds
                  (Mark.project r (Path.rect (Box2.v 1.25 0.25 0.25 0.5))))
          in
          equal page_box None got);
      test "project cuts a box at the domain's edge" (fun () ->
          let got, corner, edge =
            rows_of [] (fun r ->
                let at = Coord.point (Mark.projection r) in
                ( Path.bounds
                    (Mark.project r (Path.rect (Box2.v 0.5 0.25 1. 0.5))),
                  at 0.5 0.25,
                  at 1. 0.75 ))
          in
          equal page_box (Some (Box2.of_pts corner edge)) got);
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
      test "a facet on a middle axis puts each block of rows in its panel"
        (fun () ->
          let m, seen =
            probe
              [
                Mark.bind Role.y (num (Nx.zeros Nx.float64 [| 2; 3; 2 |]));
                Mark.bind Role.fx (dim 1);
              ]
              Mark.index
          in
          ignore (drawn m);
          equal
            (slist (array int) compare)
            [ [| 0; 1; 6; 7 |]; [| 2; 3; 8; 9 |]; [| 4; 5; 10; 11 |] ]
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
      test "a dot is a circle 0.6 em across, a copy of one glyph" (fun () ->
          (* A circle 6 points across at the default size of 10 points, drawn at
             its size, so that the stamp scales no instance. *)
          let d = drawn (dot ~x:(const 0.5) ~y:(const 0.5) ()) in
          match
            collect
              (function
                | Picture.Stamp s -> Some (s.scales, Picture.bounds s.picture)
                | _ -> None)
              d
          with
          | [ (None, Some b) ] ->
              equal (pair close close) (6., 6.) (Box2.w b, Box2.h b)
          | [ (Some _, _) ] -> fail "the stamp scales its instances"
          | l -> failf "%d stamps" (List.length l));
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

let panel_boxes size f =
  List.map
    (fun (p : Layout.panel) -> p.box)
    (Layout.panels (layout size (resolve f)))

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

let xs n = Nx.linspace Nx.float64 0. 1000. n
let ys n = Nx.Rng.normal (Nx.Rng.key 3) Nx.float64 [| n |]

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

type trace = {
  n : int;
  width : float;  (** The panel's. *)
  density : float;
  pen : float;
  curve : Curve.t;
  signal : [ `Noise | `Walk | `Sine ];
  gaps : bool;
  along : [ `Rising | `Turning | `Waving ];
      (** [x] rises, turns back halfway, or goes back and forth 7 times. *)
  zoom : bool;  (** The x domain holds the middle 80% of the data. *)
  facets : bool;  (** Two facet panels of two series each. *)
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
          (triple (int_range 1_000 6_000)
             (of_list [ 50.; 50.37; 33.3 ])
             (of_list [ 1.; 1.3; 2. ]))
          (triple
             (of_list [ 0.5; 2.; 7. ])
             (of_list Curve.[ linear; step_after; step_before; step_mid ])
             (of_list [ `Noise; `Walk; `Sine ]))
          (quad bool (of_list [ `Rising; `Turning; `Waving ]) bool bool)))

(* [line_at ~pen ~curve ~whole x y] is a line of [x] and [y] stroked at [pen]
   along [curve]. When [whole], its stroke is bound to one category per row, a
   channel that varies along the series, so that the line is not reduced, yet
   draws one series in one colour. *)
let line_at ?fx ~pen ~curve ~whole x y =
  let shape = if whole then Nx.shape y else [| 1 |] in
  let stroke = cat (Nx.zeros Nx.int32 shape) in
  line ?fx ~x ~y:(num y) ~stroke ~width:(const pen) ~curve ()

let m4_inks_alike =
  prop ~tags:[ "slow" ] "m4 inks what the whole line inks, within a pixel"
    ~count:40 gen_trace (fun t ->
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
        if t.zoom then
          num ~scale:(Scale.linear ~domain:(0.1 *. hi, 0.9 *. hi) ()) x
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
        let theme = Theme.v ~paper:Color.transparent () in
        let r =
          Drawing.renderable
            (drawn ~theme ~density:t.density ~size:(Size.panels t.width 50.) f)
        in
        (* Two device pixels of margin on every side, so that no ink is cut by
           the page's edge. *)
        let m = 2. /. t.density in
        Raster.render ~density:t.density
          (Renderable.v
             (Renderable.w r +. (2. *. m))
             (Renderable.h r +. (2. *. m))
             (Picture.transform (Affine.translate m m) (Renderable.picture r)))
      in
      equal
        (list (pair int int))
        []
        (stray (page ~whole:true) (page ~whole:false)))

(* [line_rows f] is the number of rows the line [f] draws. *)
let line_rows f =
  List.fold_left
    (fun acc -> function
      | Picture.Rows rows, _ -> acc + Array.length rows | _ -> acc)
    0
    (tags (path [ Index 0 ]) (drawn ~size:(Size.panels 50. 50.) (layer [ f ])))

let m4_lines =
  let y = ys 2000 in
  cases
    ~name:(fun (name, _, _) -> name)
    "a line is reduced only if each piece depends on its two points alone"
    [
      ("linear", line ~y:(num y) (), true);
      ("monotone_x", line ~curve:Curve.monotone_x ~y:(num y) (), false);
      ("step_mid", line ~curve:Curve.step_mid ~y:(num y) (), true);
      ("natural", line ~curve:Curve.natural ~y:(num y) (), false);
      ("catmull_rom", line ~curve:Curve.catmull_rom ~y:(num y) (), false);
      ("basis", line ~curve:Curve.basis ~y:(num y) (), false);
      ("filled", line ~fill:(const Color.red) ~y:(num y) (), false);
      ("dashed", line ~dash:(const Dash.dashed) ~y:(num y) (), false);
      ("solid", line ~dash:(const Dash.solid) ~y:(num y) (), true);
    ]
    (fun (_, f, reduced) -> equal bool reduced (line_rows f < 2000))

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

(* [one_panel kept] is the rows read in the only panel. *)
let one_panel = function
  | [ box ], [ kept ] -> (box, kept)
  | boxes, kept ->
      failf "%d panels, %d draws" (List.length boxes) (List.length kept)

let m4_columns () =
  let n = 1001 in
  let x = xs n and y = ys n in
  let box, kept = one_panel (m4_kept x y) in
  let x = Nx.to_array x in
  equal (array int)
    (m4_reference
       (fun i -> columns ~density:1. box (0., 1000.) x.(i))
       (Nx.to_array y) (List.init n Fun.id))
    kept

(* A series over [-300, 1300] on the domain [0, 1000], as in a zoom. *)
let m4_zoom () =
  let n = 2001 in
  let x = Nx.linspace Nx.float64 (-300.) 1300. n and y = ys n in
  let box, kept = one_panel (m4_kept x y) in
  let x = Nx.to_array x in
  equal (array int)
    (m4_reference
       (fun i -> columns ~density:1. box (0., 1000.) x.(i))
       (Nx.to_array y) (List.init n Fun.id))
    kept

let m4_gaps () =
  let n = 1001 in
  let x = xs n and y = with_gaps (ys n) in
  let box, kept = one_panel (m4_kept x y) in
  let x = Nx.to_array x in
  less int ~than:(n / 2) (Array.length kept);
  equal (array int)
    (m4_reference
       (fun i -> columns ~density:1. box (0., 1000.) x.(i))
       (Nx.to_array y) (List.init n Fun.id))
    kept

let m4_turns () =
  let n = 1001 in
  let x =
    Nx.init Nx.float64 [| n |] (fun i ->
        Float.of_int (Int.min i.(0) (n - 1 - i.(0))) *. 2.)
  in
  let y = ys n in
  let box, kept = one_panel (m4_kept x y) in
  let x = Nx.to_array x in
  equal (array int)
    (m4_reference
       (fun i -> columns ~density:1. box (0., 1000.) x.(i))
       (Nx.to_array y) (List.init n Fun.id))
    kept

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

let reducers =
  group "Reducers"
    [
      test "m4 keeps every row at four rows per column" (fun () ->
          let _, kept = m4_kept (xs 200) (ys 200) in
          equal (list int) [ 200 ] (List.map Array.length kept));
      test
        "m4 keeps each device-pixel column's first, last, lowest and highest \
         rows"
        m4_columns;
      test "m4 cuts the columns at the domain's ends" m4_zoom;
      test "m4 keeps the first row of each run of dropped rows" m4_gaps;
      test "m4 keeps each run of a column when x turns back" m4_turns;
      test "m4 keeps every row when x is in no order" (fun () ->
          let n = 1001 in
          let x = Nx.Rng.uniform (Nx.Rng.key 7) Nx.float64 [| n |] in
          (* A few columns, so that some runs have three rows or more. *)
          let size = Size.panels 5. 50. in
          let _, kept = m4_kept ~size (Nx.mul_s x 1000.) (ys n) in
          equal (list int) [ n ] (List.map Array.length kept));
      test "m4 reduces the series of each facet panel" m4_facets;
      test "m4 joins a series' rows across the rows of other facet panels"
        m4_split_facets;
      m4_inks_alike;
      m4_lines;
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
      test "cells paints each cell at its row's opacity" (fun () ->
          let z =
            Nx.create Nx.float64 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |]
          in
          let opacity = num ~scale:(Scale.linear ~domain:(0., 5.) ()) z in
          let f =
            layer [ rect ~x:(dim 1) ~y:(dim 0) ~fill:(num z) ~opacity () ]
          in
          let r = resolve f in
          match cells (drawn f) with
          | [ (_, px) ] ->
              let at i j =
                let k = Float.of_int ((i * 3) + j) in
                let c = viridis_of r k in
                Color.with_alpha (Color.alpha c *. (k /. 5.)) c
              in
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
      cases ~name:fst
        "the large image of any mark is gathered, drawing what it draws"
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
      cases ~name:fst
        "an image under a clip, an opacity or a tag is gathered, under a \
         transform or a stamp left whole"
        [
          ("clip", (true, fun box p -> Picture.clip (Path.rect box) p));
          ("opacity", (true, fun _ p -> Picture.opacity 0.5 p));
          ( "tag",
            ( true,
              fun _ p ->
                Picture.tag
                  { Picture.id = Nx.Ptree.Path.root; rows = Picture.Rows [||] }
                  p ) );
          ( "transform",
            (false, fun _ p -> Picture.transform (Affine.translate 0.5 0.5) p)
          );
          ("stamp", (false, fun _ p -> Picture.stamp [| 0.5 |] [| 0.5 |] p));
        ]
        (fun (_, (gather, wrap)) ->
          let px = Nx.zeros Nx.uint8 [| 200; 300; 4 |] in
          let f =
            Mark.v ~name:"picture" [] (fun r ->
                let at = Coord.point (Mark.projection r) in
                let box = Box2.of_pts (at 0. 1.) (at 1. 0.) in
                wrap box (Picture.image box px))
          in
          match images (drawn ~size:(Size.panels 20. 20.) f) with
          | [ (_, got) ] ->
              let s = Nx.shape got in
              equal bool gather (s.(0) * s.(1) < 200 * 300)
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

(* Areas *)

(* [fill_paths d] is the path of each fill that the mark of [layer [ m ]] draws
   in [d]. *)
let fill_paths d =
  List.concat_map
    (fun (_, p) ->
      List.rev
        (fold
           (fun acc -> function Picture.Fill f -> f.path :: acc | _ -> acc)
           [] p))
    (tags (path [ Index 0 ]) d)

let subpaths p =
  Path.fold
    ~move:(fun n _ _ -> n + 1)
    ~line:(fun n _ _ -> n)
    ~cubic:(fun n _ _ _ _ _ _ -> n)
    ~close:Fun.id 0 p

let areas =
  group "Areas"
    [
      test "swapping y and y2 fills the same pixels" (fun () ->
          let a = num (f64 [| 0.; 3.; 1.; 2. |])
          and b = num (f64 [| 2.; 1.; 3.; 0. |]) in
          let page f =
            Raster.render ~density:1. (Drawing.renderable (drawn f))
          in
          within_one ~msg:"pixels"
            (page (area ~y:a ~y2:b ()))
            (page (area ~y:b ~y2:a ())));
      test "a dropped row splits the region" (fun () ->
          let d =
            drawn
              (layer [ area ~y:(num (f64 [| 1.; 2.; nan; 2.; 1.; 3. |])) () ])
          in
          equal (list int) [ 2 ] (List.map subpaths (fill_paths d)));
      test "each series is filled alone" (fun () ->
          let y =
            Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 1.; 2.; 3.; 2. |]
          in
          let d = drawn (layer [ area ~y:(num y) ~fill:(dim 0) () ]) in
          equal (list int) [ 1; 1 ] (List.map subpaths (fill_paths d)));
      test "an area alone fills down to zero" (fun () ->
          let ys = f64 [| 1.; 3.; 2. |] in
          let page f =
            Raster.render ~density:1. (Drawing.renderable (drawn f))
          in
          within_one ~msg:"pixels"
            (page (area ~y:(num ys) ~y2:(num (f64 [| 0.; 0.; 0. |])) ()))
            (page (area ~y:(num ys) ())));
    ]

(* Steps *)

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

let reads_steps values =
  let read rows =
    let get r = Option.get (r rows Role.fill) in
    (get Mark.normalized, get Mark.get, get Mark.ticks)
  in
  let norm, colours, ticks =
    rows_of [ Mark.bind Role.fill (num ~scale:stepped (f64 values)) ] read
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

let bar_fills d =
  let id = path [ Field "legend"; Field "color"; Field "num" ] in
  List.concat_map
    (fun (_, p) ->
      List.rev
        (fold
           (fun acc -> function
             | Picture.Fill f -> (f.color, Path.bounds f.path) :: acc | _ -> acc)
           [] p))
    (tags id d)

let steps =
  group "Steps"
    [
      prop "a reader of a stepped scale takes the range at its step's middle"
        (Gen.array ~size:(Gen.int_range 1 8)
           (Gen.frequency
              [
                (4, Gen.float_range (-2.) 12.);
                (1, Gen.of_list ~pp:Format.pp_print_float [ 0.; 2.; 5.; 10. ]);
              ]))
        reads_steps;
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
          let fills = bar_fills (drawn (layer [ dots; m ])) in
          let levels = levels (only seen) in
          let n = Array.length levels - 1 in
          let scheme = Theme.scheme Theme.default in
          equal (list color)
            (List.init n (fun k ->
                 Scheme.color scheme ((levels.(k) +. levels.(k + 1)) /. 2.)))
            (List.map fst fills);
          match List.map snd fills with
          | Some bar :: _ as boxes ->
              List.iteri
                (fun k b ->
                  let b = Option.get b in
                  let msg = Printf.sprintf "step %d" k in
                  equal ~msg close (Box2.miny bar) (Box2.miny b);
                  equal ~msg close ((1. -. levels.(k)) *. Box2.h bar) (Box2.h b))
                boxes
          | _ -> fail "no steps");
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
          List.iter
            (fun (u, mid) ->
              let msg = Printf.sprintf "at %g" u in
              equal ~msg color (Scheme.color scheme mid) (f u))
            [ (0.1, 0.125); (0.3, 0.375); (0.9, 0.75) ]);
      test "contour fills one band between consecutive explicit ticks"
        (fun () ->
          let z =
            Nx.init Nx.float64 [| 3; 4 |] (fun i ->
                Float.of_int (i.(0) * i.(1)))
          in
          let scale = Scale.linear ~domain:(0., 6.) ~ticks:[| 2.; 4. |] () in
          let d = drawn (layer [ contour ~fill:(num ~scale z) () ]) in
          let fills =
            List.concat_map
              (fun (_, p) ->
                fold
                  (fun acc -> function Picture.Fill _ -> acc + 1 | _ -> acc)
                  0 p
                :: [])
              (tags (path [ Index 0 ]) d)
          in
          equal (list int) [ 3 ] fills);
      test "contour implies stepped on its fill scale" (fun () ->
          let z =
            Nx.init Nx.float64 [| 3; 4 |] (fun i ->
                Float.of_int (i.(0) * i.(1)))
          in
          let fill s = Scale.linear ~name:"z" ?stepped:s () in
          let stepped_with s =
            Scale.stepped
              (Resolved.scale
                 (resolve (contour ~fill:(num ~scale:(fill s) z) ()))
                 (fill None))
          in
          equal bool true (stepped_with None);
          equal bool false (stepped_with (Some false)));
    ]

(* Dashes *)

(* [strokes_of id d] is the stroke style of each stroke drawn under the
   outermost tag [id] in [d], in drawing order. *)
let strokes_of id d =
  match tags id d with
  | [] -> []
  | (_, p) :: _ ->
      List.rev
        (fold
           (fun acc -> function
             | Picture.Stroke s -> s.stroke :: acc | _ -> acc)
           [] p)

let mark_id = path [ Index 0 ]
let lengths = list float_exact

let gen_column =
  Gen.array ~size:(Gen.int_range 1 6)
    (Gen.frequency
       [
         (4, Gen.float_range (-5.) 5.);
         (1, Gen.of_list ~pp:Format.pp_print_float [ nan; Float.infinity ]);
       ])

(* [solid_law (name, mark) ys] states that [mark ys] with a constant solid dash
   draws what it draws without one. *)
let solid_law ((_, mark), ys) =
  let d dash = drawn (layer [ mark dash (f64 ys) ]) in
  equal drawing (d None) (d (Some (const Dash.solid)))

let dashed_marks =
  Gen.of_list
    ~pp:(fun ppf (n, _) -> Format.pp_print_string ppf n)
    [
      ("line", fun dash y -> line ?dash ~y:(num y) ());
      ("rule", fun dash y -> rule ?dash ~y:(num y) ());
    ]

let pattern_cases =
  List.concat_map
    (fun d -> List.map (fun w -> (d, w)) [ 0.5; 2. ])
    [ Dash.dashed; Dash.dotted; Dash.dash_dot; Dash.v [ 3. ] ]

let pattern_case (d, w) =
  let y = num (f64 [| 1.; 2.; 3. |]) in
  List.iter
    (fun (name, f) ->
      let expected = List.map (fun l -> l *. w) (Dash.lengths d) in
      match strokes_of mark_id (drawn (layer [ f ])) with
      | [] -> failf "%s draws no stroke" name
      | ss ->
          List.iter
            (fun s ->
              equal ~msg:name lengths expected (Stroke.dash s);
              equal ~msg:name float_exact w (Stroke.width s))
            ss)
    [
      ("line", line ~dash:(const d) ~width:(const w) ~y ());
      ("rule", rule ~dash:(const d) ~width:(const w) ~y ());
    ]

let dashes =
  group "Dashes"
    [
      prop "a constant solid dash draws what no dash draws"
        (Gen.pair dashed_marks gen_column)
        solid_law;
      cases "a stroke's dash is the pattern's lengths times its width"
        ~name:(fun (d, w) -> Format.asprintf "%a at %g" Dash.pp d w)
        pattern_cases pattern_case;
      test "category i takes pattern i modulo their number" (fun () ->
          let ds = [| Dash.dashed; Dash.dotted |] in
          let scale = Scale.band ~dashes:ds () in
          let got =
            rows_of ~shape:[| 5; 2 |]
              [ Mark.bind Role.dash (dim ~scale 0) ]
              (fun r -> Option.get (Mark.get r Role.dash))
          in
          equal (array dash) (Array.init 10 (fun k -> ds.(k / 2 mod 2))) got);
      test "categories take Dash.all by default, cycling" (fun () ->
          let all = Array.of_list Dash.all in
          let got =
            rows_of ~shape:[| 6 |]
              [ Mark.bind Role.dash (dim 0) ]
              (fun r -> Option.get (Mark.get r Role.dash))
          in
          equal (array dash) (Array.init 6 (fun k -> all.(k mod 4))) got);
      test "a series takes the pattern of its first row" (fun () ->
          let y =
            num (Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 3.; 3.; 2.; 1. |])
          in
          let scale = Scale.band ~dashes:[| Dash.dotted; Dash.dashed |] () in
          let d = drawn (layer [ line ~y ~dash:(dim ~scale 0) () ]) in
          let w = 0.15 *. Theme.size Theme.default in
          let times d = List.map (fun l -> l *. w) (Dash.lengths d) in
          equal (list lengths)
            [ times Dash.dotted; times Dash.dashed ]
            (List.map Stroke.dash (strokes_of mark_id d)));
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
          let legends =
            List.sort_uniq
              (fun a b ->
                String.compare
                  (Nx.Ptree.Path.to_string a)
                  (Nx.Ptree.Path.to_string b))
              (collect
                 (function
                   | Picture.Tag { tag; _ }
                     when List.exists
                            (function
                              | Nx.Ptree.Path.Field "legend" -> true
                              | _ -> false)
                            (Nx.Ptree.Path.segments tag.id) ->
                       Some tag.id
                   | _ -> None)
                 d)
          in
          equal
            (list
               (Testable.make ~pp:Nx.Ptree.Path.pp ~equal:Nx.Ptree.Path.equal))
            [ path [ Field "legend"; Field "run"; Field "cat" ] ]
            legends;
          let w = 0.15 *. Theme.size Theme.default in
          equal (list lengths)
            (List.map
               (fun d -> List.map (fun l -> l *. w) (Dash.lengths d))
               [ Dash.solid; Dash.dashed ])
            (List.map Stroke.dash (strokes_of (List.hd legends) d)));
    ]

(* Ablines *)

let points_of path =
  List.rev
    (Path.fold
       ~move:(fun acc x y -> P2.v x y :: acc)
       ~line:(fun acc x y -> P2.v x y :: acc)
       ~cubic:(fun acc _ _ _ _ x y -> P2.v x y :: acc)
       ~close:Fun.id [] path)

let line_id = path [ Index 1 ]

(* Position scales of each transform, the data a line [y = 2x + 1] spans without
   leaving their domains. *)
let transforms =
  let lin name = Scale.linear ~name () and log name = Scale.log ~name () in
  [
    ("linear", (lin "x", lin "y"));
    ("log x", (log "x", lin "y"));
    ("log y", (lin "x", log "y"));
    ("log x and y", (log "x", log "y"));
  ]

let equation_case (name, (sx, sy)) =
  let f =
    layer
      [
        dot
          ~x:(num ~scale:sx (f64 [| 1.; 100. |]))
          ~y:(num ~scale:sy (f64 [| 3.; 201. |]))
          ();
        abline ~slope:(const 2.) ~intercept:(const 1.) ();
      ]
  in
  let r = resolve f in
  let l = layout (Size.panels 100. 100.) r in
  let proj = (List.hd (Layout.panels l)).projection in
  let fx = Resolved.scale r sx and fy = Resolved.scale r sy in
  let pts =
    List.concat_map
      (fun s -> points_of s)
      (List.map
         (fun (st : Picture.t) ->
           match st with Stroke s -> s.path | _ -> Path.empty)
         (List.concat_map
            (fun (_, p) ->
              fold
                (fun acc p ->
                  match p with Picture.Stroke _ -> p :: acc | _ -> acc)
                [] p)
            (tags line_id (draw ~density:1. l))))
  in
  let n = List.length pts in
  if name = "linear" then equal ~msg:"a segment" int 2 n
  else greater ~msg:"a curve" int ~than:50 n;
  let us =
    List.map
      (fun p ->
        let u, v = Option.get (Coord.invert proj p) in
        let x = Option.get (Scale.invert fx u)
        and y = Option.get (Scale.invert fy v) in
        equal
          ~msg:(Format.asprintf "%a" P2.pp p)
          (float_rel ~rel:1e-9 ~abs:0.)
          ((2. *. x) +. 1.)
          y;
        u)
      pts
  in
  equal ~msg:"from the domain's start" (float 1e-9) 0.
    (List.fold_left Float.min 1. us);
  equal ~msg:"to its end" (float 1e-9) 1. (List.fold_left Float.max 0. us)

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
  let fscale = Testable.make ~pp:Scale.pp ~equal:Scale.equal in
  equal (pair fscale fscale)
    (scales (layer [ dots ]))
    (scales
       (layer
          [
            dots; abline ~slope:(const m) ~intercept:(num (f64 [| c; 0. |])) ();
          ]))

let ablines =
  group "Ablines"
    [
      cases "the points of an abline satisfy its equation" ~name:fst transforms
        equation_case;
      prop "an abline leaves every fitted scale as it was"
        (Gen.pair gen_coefficient gen_coefficient)
        unchanged_law;
      cases "an abline of slope 0 draws the rule at its intercept"
        ~name:string_of_float [ 1.5; 2.; 3.25 ] (fun c ->
          let dots =
            dot ~x:(num (f64 [| 1.; 4. |])) ~y:(num (f64 [| 1.; 4. |])) ()
          in
          equal drawing
            (drawn (layer [ dots; rule ~y:(Hugin_next.floats [| c |]) () ]))
            (drawn
               (layer
                  [ dots; abline ~slope:(const 0.) ~intercept:(const c) () ])));
      test "an abline over a band x draws nothing, with a warning" (fun () ->
          let f =
            layer
              [
                dot ~x:(strings [| "a"; "b" |]) ~y:(num (f64 [| 1.; 4. |])) ();
                abline ~slope:(const 1.) ~intercept:(const 0.) ();
              ]
          in
          let d = drawn f in
          equal int 0 (List.length (strokes_of line_id d));
          equal (list string)
            [ "an abline needs quantitative x and y scales" ]
            (List.filter_map
               (fun (id, msg) ->
                 if Nx.Ptree.Path.equal id line_id then Some msg else None)
               (Drawing.warnings d)));
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
          let fscale = Testable.make ~pp:Scale.pp ~equal:Scale.equal in
          equal (option fscale) (Some (Resolved.scale r x)) qx;
          equal (option (Testable.make ~pp:Scale.pp ~equal:Scale.equal)) None cx;
          equal (option fscale)
            (Some (Resolved.scale r (Scale.linear ~name:"y" ())))
            qy);
    ]

(* Frames *)

let framed_dots = dot ~x:(num (f64 [| 1.; 4. |])) ~y:(num (f64 [| 2.; 3. |])) ()

let frames =
  group "Frames"
    [
      cases "a frame draws what a rect stroked in the ink draws" ~name:fst
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
          let box = (List.hd (Layout.panels (layout size (resolve f)))).box in
          let half = 0.08 *. Theme.size Theme.default /. 2. in
          match tags line_id (drawn ~size f) with
          | [ (_, p) ] ->
              let b = Option.get (Picture.bounds p) in
              equal (list close)
                [
                  Box2.minx box -. half;
                  Box2.miny box -. half;
                  Box2.maxx box +. half;
                  Box2.maxy box +. half;
                ]
                [ Box2.minx b; Box2.miny b; Box2.maxx b; Box2.maxy b ]
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
            (List.concat_map
               (fun (_, p) ->
                 fold
                   (fun acc -> function
                     | Picture.Stroke s -> s.color :: acc | _ -> acc)
                   [] p)
               (tags line_id d)));
    ]

(* Explicit ticks *)

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

let ticks_law vs =
  let scale = Scale.linear ~domain:(0., 10.) ~ticks:vs () in
  let inside v = Float.is_finite v && 0. <= v && v <= 10. in
  cover "a value outside the domain" (Array.exists (fun v -> not (inside v)) vs);
  cover "a repeated value"
    (Array.exists
       (fun v -> List.length (List.filter (( = ) v) (Array.to_list vs)) > 1)
       vs);
  (* Distinct values, which may normalise to one position. *)
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
  equal (array (float Float.min_float)) expected ticks

let explicit_ticks =
  group "Explicit ticks"
    [
      prop "the ticks are the increasing distinct values in the domain"
        gen_ticks ticks_law;
    ]

(* Swatches *)

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

let legend_swatches =
  group "Swatches"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "a size legend's swatches show their fill neutral or as bound"
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
          List.iter (equal (option (array color)) expected) fills);
    ]

(* The built-in marks on the public interface *)

module Copy = Hugin_next_test_marks.Marks

let builtins =
  let v = f64 [| 1.; 3.; 2. |]
  and z =
    Nx.init Nx.float64 [| 3; 4 |] (fun i -> Float.of_int (i.(0) * i.(1)))
  in
  let px =
    Nx.init Nx.float32 [| 2; 3; 3 |] (fun i ->
        Float.of_int (i.(0) + i.(1) + i.(2)) /. 6.)
  in
  [
    ( "dot",
      dot ~x:(num v) ~y:(num v) ~fill:(dim 0) (),
      Copy.dot ~x:(num v) ~y:(num v) ~fill:(dim 0) () );
    ( "line",
      line ~curve:Curve.natural ~y:(num v) (),
      Copy.line ~curve:Curve.natural ~y:(num v) () );
    ("area", area ~y:(num v) (), Copy.area ~y:(num v) ());
    ("rect", rect ~x:(dim 0) ~y:(num v) (), Copy.rect ~x:(dim 0) ~y:(num v) ());
    ("rule", rule ~y:(num v) (), Copy.rule ~y:(num v) ());
    ( "text",
      Hugin_next.text ~dx:2. ~x:(num v) ~y:(num v) ~text:(num v) (),
      Copy.text ~dx:2. ~x:(num v) ~y:(num v) ~text:(num v) () );
    ("image", image px, Copy.image px);
    ("contour", contour ~fill:(num z) (), Copy.contour ~fill:(num z) ());
  ]

let public_marks =
  cases
    ~name:(fun (n, _, _) -> n)
    "a built-in compiled against the public interface draws alike" builtins
    (fun (_, f, copy) -> equal drawing (drawn f) (drawn copy))

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

let () =
  exit
    (run "Draw"
       [
         rows;
         channels;
         domain;
         drawings;
         output;
         reducers;
         areas;
         steps;
         explicit_ticks;
         dashes;
         ablines;
         frames;
         legend_swatches;
         public_marks;
         goldens;
       ])
