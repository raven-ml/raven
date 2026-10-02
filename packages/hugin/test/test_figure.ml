(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin
open Windtrap

let f32 shape = Nx.zeros Nx.float32 shape
let i32 shape = Nx.zeros Nx.int32 shape
let mask shape = Nx.full Nx.bool shape true
let v = f32 [| 3 |]
let m = f32 [| 2; 3 |]
let codes = i32 [| 3 |]
let batch = f32 [| 4; 2; 3; 3 |]
let on_x c = dot ~x:c ~y:(const 0.5) ()
let on_fill c = dot ~x:(const 0.5) ~y:(const 0.5) ~fill:c ()
let draw_nothing (_ : Mark.rows) = Picture.empty
let mark ?shape bindings = Mark.v ~name:"m" ?shape bindings draw_nothing
let a = dot ~x:(num v) ~y:(num v) ()

(* [made f] makes the value [f] makes, whatever its type. *)
let made f () = ignore (f ())

(* Construction errors *)

(* Every value a constructor refuses, beside the nearest values it accepts. A
   lift's shape errors raise when its mark is made. *)
let refused =
  [
    ( "num of complex64",
      made (fun () -> on_x (num (Nx.zeros Nx.complex64 [| 2 |]))) );
    ("num of bool", made (fun () -> on_x (num (mask [| 2 |]))));
    ( "num with a valid that grows",
      made (fun () -> on_x (num ~valid:(mask [| 3; 3 |]) v)) );
    ( "num with a valid of another length",
      made (fun () -> on_x (num ~valid:(mask [| 4 |]) v)) );
    ("cat of floats", made (fun () -> on_fill (cat (f32 [| 2 |]))));
    ("cat of bools", made (fun () -> on_fill (cat (mask [| 2 |]))));
    ( "cat with a repeated label",
      made (fun () -> on_fill (cat ~labels:[| "a"; "b"; "a" |] (i32 [| 2 |])))
    );
    ( "cat with a valid that grows",
      made (fun () -> on_fill (cat ~valid:(mask [| 2; 2 |]) (i32 [| 2 |]))) );
    ( "dim one past the last axis",
      made (fun () -> line ~y:(num m) ~stroke:(dim 2) ()) );
    ( "dim one before the first axis",
      made (fun () -> line ~y:(num m) ~stroke:(dim (-3)) ()) );
    ( "dim labels one short",
      made (fun () -> line ~y:(num m) ~stroke:(dim ~labels:[| "a" |] 0) ()) );
    ( "dim labels one long",
      made (fun () ->
          line ~y:(num m) ~stroke:(dim ~labels:[| "a"; "b"; "c" |] 0) ()) );
    ( "a dim mask that does not broadcast",
      made (fun () -> rect ~fx:(dim ~valid:(mask [| 4 |]) 0) ~y:(num v) ()) );
    ( "index one past the last axis",
      made (fun () -> line ~x:(index 2) ~y:(num m) ()) );
    ( "index of a shape without axes",
      made (fun () -> line ~y:(num (f32 [||])) ()) );
    ( "channels that do not broadcast",
      made (fun () -> dot ~x:(num v) ~y:(num (f32 [| 4 |])) ()) );
    ("rect with x2 without x", made (fun () -> rect ~x2:(num v) ()));
    ("rect with y2 without y", made (fun () -> rect ~y2:(num v) ()));
    ("rule without positions", made (fun () -> rule ()));
    ("rule with x2 alone", made (fun () -> rule ~x2:(num v) ()));
    ("rule with x and x2 only", made (fun () -> rule ~x:(num v) ~x2:(num v) ()));
    ( "rule with x, x2 and y2",
      made (fun () -> rule ~x:(num v) ~x2:(num v) ~y2:(num v) ()) );
    ("image of rank 1", made (fun () -> image (f32 [| 4 |])));
    ("image with two channels", made (fun () -> image (f32 [| 2; 2; 2 |])));
    ("image with five channels", made (fun () -> image (f32 [| 3; 2; 2; 5 |])));
    ("image of int32", made (fun () -> image (i32 [| 2; 2 |])));
    ("image of bool", made (fun () -> image (mask [| 2; 2 |])));
    ("contour of rank 1", made (fun () -> contour ~fill:(num v) ()));
    ( "contour with a constant fill",
      made (fun () ->
          contour ~fill:(const Color.red) ~x:(num v)
            ~y:(num (f32 [| 2; 1 |]))
            ()) );
    ( "contour with x varying along the rows",
      made (fun () -> contour ~x:(num m) ~fill:(num m) ()) );
    ( "contour with y varying along the columns",
      made (fun () -> contour ~y:(num (f32 [| 3 |])) ~fill:(num m) ()) );
    ( "contour with x the index of the rows",
      made (fun () -> contour ~x:(index 0) ~fill:(num m) ()) );
    ( "contour with y the index of the columns",
      made (fun () -> contour ~y:(index 1) ~fill:(num m) ()) );
    ( "contour faceted along its rows",
      made (fun () -> contour ~fx:(dim 0) ~fill:(num m) ()) );
    ( "contour faceted along its columns",
      made (fun () ->
          contour ~fy:(strings [| "a"; "b"; "c" |]) ~fill:(num m) ()) );
    ( "Mark.v binding a role twice",
      made (fun () ->
          mark [ Mark.bind Role.x (num v); Mark.bind Role.x (num v) ]) );
    ( "Mark.v binding two value roles of one name",
      made (fun () ->
          mark
            [
              Mark.bind (Role.value ~name:"a") (num v);
              Mark.bind (Role.value ~name:"a") (num v);
            ]) );
    ( "Mark.v binding two parameters of one name",
      made (fun () ->
          mark
            [
              Mark.bind (Role.param ~name:"a" ~equal:Int.equal) (const 0);
              Mark.bind (Role.param ~name:"a" ~equal:Int.equal) (const 0);
            ]) );
    ( "Mark.v with x2 without x",
      made (fun () -> mark [ Mark.bind Role.x2 (num v) ]) );
    ( "Mark.v with y2 without y",
      made (fun () -> mark [ Mark.bind Role.y2 (num v) ]) );
    ( "Mark.v with x of quantities and x2 of categories",
      made (fun () ->
          mark
            [ Mark.bind Role.x (num v); Mark.bind Role.x2 (cat (i32 [| 3 |])) ])
    );
    ( "Mark.v with y of categories and y2 of quantities",
      made (fun () ->
          mark
            [ Mark.bind Role.y (cat (i32 [| 3 |])); Mark.bind Role.y2 (num v) ])
    );
    ( "Mark.v with a scale on the text role",
      made (fun () ->
          mark [ Mark.bind Role.text (num ~scale:(Scale.log ()) v) ]) );
    ( "Mark.v with a title on the text role",
      made (fun () ->
          mark [ Mark.bind Role.text (strings ~title:"t" [| "a" |]) ]) );
    ( "Mark.v with a title on a value role",
      made (fun () ->
          mark [ Mark.bind (Role.value ~name:"angle") (num ~title:"t" v) ]) );
    ( "Mark.v with a negative dimension",
      made (fun () -> mark ~shape:[| 2; -1 |] []) );
    ( "Mark.v with a shape the channels do not broadcast with",
      made (fun () -> mark ~shape:[| 2 |] [ Mark.bind Role.x (num v) ]) );
    ( "Mark.broadcast with a negative dimension",
      made (fun () -> Mark.broadcast ~shape:[| -1 |] []) );
    ( "Mark.broadcast of a dim past the shape",
      made (fun () -> Mark.broadcast [ Mark.bind Role.fill (dim 0) ]) );
    ( "Role.param of an empty name",
      made (fun () -> Role.param ~name:"" ~equal:Int.equal) );
    ( "Role.param of a built-in role's name",
      made (fun () -> Role.param ~name:"x" ~equal:Int.equal) );
    ("Role.value of an empty name", made (fun () -> Role.value ~name:""));
    ( "Role.value of a built-in role's name",
      made (fun () -> Role.value ~name:"fill") );
    ("name axis", made (fun () -> name "axis" a));
    ("name legend", made (fun () -> name "legend" a));
    ("name panel", made (fun () -> name "panel" a));
    ("name cell", made (fun () -> name "cell" a));
    ( "a grid width of zero",
      made (fun () -> grid ~widths:[ 1.; 0. ] [ [ a; a ] ]) );
    ("a negative grid height", made (fun () -> grid ~heights:[ -1. ] [ [ a ] ]));
    ("a nan grid width", made (fun () -> grid ~widths:[ Float.nan ] [ [ a ] ]));
    ( "an infinite grid width",
      made (fun () -> grid ~widths:[ Float.infinity ] [ [ a ] ]) );
    ("a span of no rows", made (fun () -> span ~rows:0 a));
    ("a span of no columns", made (fun () -> span ~cols:0 a));
    ( "a scale shared twice",
      made (fun () -> share [ ("x", `Shared); ("x", `Independent) ] a) );
    ( "an interval init out of order",
      made (fun () -> View.interval "i" ~init:(Some (2., 1.))) );
    ( "an interval init with nan",
      made (fun () -> View.interval "i" ~init:(Some (Float.nan, 1.))) );
    ( "an interval set to infinity",
      made (fun () ->
          View.set
            (View.interval "i" ~init:None)
            (Some (0., Float.infinity))
            View.empty) );
    ("a zoom of an unnamed scale", made (fun () -> View.zoom (Scale.linear ())));
    ( "a zoom of a band scale",
      made (fun () -> View.zoom (Scale.band ~name:"b" ())) );
    ("a figure of zero width", made (fun () -> Size.figure 0. 10.));
    ("a figure of nan height", made (fun () -> Size.figure 10. Float.nan));
    ("panels of infinite width", made (fun () -> Size.panels Float.infinity 10.));
    ("a theme of size zero", made (fun () -> Theme.v ~size:0. ()));
    ("a theme without fonts", made (fun () -> Theme.v ~fonts:[] ()));
    ("an aspect of zero", made (fun () -> Coord.cartesian ~aspect:0. ()));
    ("an aspect of nan", made (fun () -> Coord.cartesian ~aspect:Float.nan ()));
  ]

let accepted =
  [
    ( "num of integers with a valid that broadcasts",
      made (fun () -> on_x (num ~valid:(mask [| 2; 1 |]) (i32 [| 2; 3 |]))) );
    ( "num with a valid of rank 0",
      made (fun () -> on_x (num ~valid:(mask [||]) m)) );
    ("cat of uint8", made (fun () -> on_fill (cat (Nx.zeros Nx.uint8 [| 2 |]))));
    ( "cat of uint64",
      made (fun () -> on_fill (cat (Nx.zeros Nx.uint64 [| 2 |]))) );
    ( "dim labels that repeat",
      made (fun () -> line ~y:(num m) ~stroke:(dim ~labels:[| "a"; "a" |] 0) ())
    );
    ( "index of the first axis counted from the last",
      made (fun () -> line ~x:(index (-2)) ~y:(num m) ()) );
    ( "a dim mask that joins the shape",
      made (fun () -> rect ~fx:(dim ~valid:(mask [| 4 |]) 0) ()) );
    ( "a mark of constants",
      made (fun () -> dot ~x:(const 0.5) ~y:(const 0.5) ()) );
    ("rule with x alone", made (fun () -> rule ~x:(num v) ()));
    ( "rule with y and x2 and x",
      made (fun () -> rule ~y:(num v) ~x:(num v) ~x2:(num v) ()) );
    ( "rule with x, x2, y and y2",
      made (fun () -> rule ~x:(num v) ~x2:(num v) ~y:(num v) ~y2:(num v) ()) );
    ( "image of uint8 RGB",
      made (fun () -> image (Nx.zeros Nx.uint8 [| 2; 2; 3 |])) );
    ( "images over a datum axis",
      made (fun () -> image ~fx:(dim 0) (f32 [| 5; 2; 2; 4 |])) );
    ( "contour on the axes of a field",
      made (fun () ->
          contour
            ~x:(num (f32 [| 3 |]))
            ~y:(num (f32 [| 2; 1 |]))
            ~fill:(num m) ()) );
    ( "contour of fields faceted along a datum axis",
      made (fun () -> contour ~fill:(num (f32 [| 4; 2; 3 |])) ~fx:(dim 0) ()) );
    ( "Mark.v with x of quantities and x2 a constant",
      made (fun () ->
          mark [ Mark.bind Role.x (num v); Mark.bind Role.x2 (const 1.) ]) );
    ( "Mark.v with strings beside a tensor of their length",
      made (fun () ->
          mark
            [
              Mark.bind Role.x (num v);
              Mark.bind Role.fill (strings [| "a"; "b"; "c" |]);
            ]) );
    ("name axes", made (fun () -> name "axes" a));
    ( "the least positive weight",
      made (fun () -> grid ~widths:[ Float.min_float ] [ [ a ] ]) );
    ("a span of one cell", made (fun () -> span a));
    ( "an interval init of one point",
      made (fun () -> View.interval "i" ~init:(Some (1., 1.))) );
  ]

let construction =
  group "construction"
    [
      cases ~name:fst "refuses" refused (fun (_, f) ->
          raises_match (Exn.invalid_arg ?substring:None) f);
      cases ~name:fst "accepts" accepted (fun (_, f) -> f ());
    ]

(* Lifts *)

(* [copied lift fresh] states that the figure [lift] makes of an array [fresh]
   makes stays equal to the figure of a second such array once the first
   changes: a lift that kept the caller's array would change with it. *)
let copied lift fresh =
  let arr = fresh () in
  let f = lift arr in
  arr.(0) <- arr.(1);
  equal bool true (Hugin.equal f (lift (fresh ())))

let copies =
  let fill c = dot ~x:(num v) ~y:(num v) ~fill:c () in
  let labels () = [| "a"; "b"; "c" |] in
  [
    ("cat labels", fun () -> copied (fun a -> fill (cat ~labels:a codes)) labels);
    ("strings", fun () -> copied (fun a -> fill (strings a)) labels);
    ("dim labels", fun () -> copied (fun a -> fill (dim ~labels:a 0)) labels);
    ( "floats",
      fun () -> copied (fun a -> rule ~y:(floats a) ()) (fun () -> [| 0.; 1. |])
    );
  ]

let kinds =
  let quantitative : float Scale.kind option -> string = function
    | Some Quantitative -> "quantitative"
    | None -> "none"
  and categorical : string Scale.kind option -> string = function
    | Some Categorical -> "categorical"
    | None -> "none"
  in
  [
    ("num", "quantitative", quantitative (kind (num v)));
    ("floats", "quantitative", quantitative (kind (floats [| 0. |])));
    ("index", "quantitative", quantitative (kind (index 0)));
    ("a float constant", "none", quantitative (kind (const 0.5)));
    ("cat", "categorical", categorical (kind (cat (i32 [| 2 |]))));
    ("strings", "categorical", categorical (kind (strings [| "a" |])));
    ("dim", "categorical", categorical (kind (dim 0)));
    ("a string constant", "none", categorical (kind (const "a")));
  ]

let lifts =
  group "lifts"
    [
      cases ~name:fst "a lift copies its array" copies (fun (_, f) -> f ());
      cases
        ~name:(fun (n, _, _) -> n)
        "kind is the kind of the scale a lift reads" kinds
        (fun (_, expected, got) -> equal string expected got);
    ]

(* Equality *)

(* A recipe names the tensors and functions a figure is built from by their
   index in fixed pools, so that two recipes are equal iff they build figures
   from the same combinators and equal arguments. *)
type recipe =
  | R_dot of int * int * int option (* x, y, x's mask. *)
  | R_line of int * bool
  | R_bar of int
  | R_mapped of int * int
  | R_level of float (* A reference line at a level given as a float. *)
  | R_custom of float option (* The aspect of the coordinate system implied. *)
  | R_axis of string * bool * string option
  | R_legend of string * bool * string option
  | R_bind of string * int
  | R_layer of recipe list
  | R_grid of float list option * recipe list list
  | R_span of int * int * recipe
  | R_title of string * int * recipe
  | R_name of string * recipe
  | R_share of string * bool * recipe
  | R_coord of float option * recipe

let tensors = Array.init 3 (fun _ -> f32 [| 3 |])
let masks = Array.init 2 (fun _ -> mask [| 3 |])
let colours = [| Color.contrast; (fun c -> Color.with_alpha 0.5 c) |]
let levels = [| 0.; 1.; Float.nan |]
let aligns : Text.Layout.halign array = [| `Center; `Left; `Right |]
let bound = Array.init 2 (fun i _ -> dot ~x:(num tensors.(i)) ~y:(const 0.5) ())

let rec build = function
  | R_dot (i, j, k) ->
      let valid = Option.map (fun k -> masks.(k)) k in
      dot ~x:(num ?valid tensors.(i)) ~y:(num tensors.(j)) ()
  | R_line (i, titled) ->
      let x = if titled then index ~title:"step" (-1) else index (-1) in
      line ~x ~y:(num tensors.(i)) ()
  | R_bar i -> rect ~x:(dim 0) ~y:(num tensors.(i)) ()
  | R_mapped (i, k) ->
      dot ~x:(const 0.5)
        ~y:(num tensors.(i))
        ~fill:(map_range colours.(k) (num tensors.(i)))
        ()
  | R_level v -> rule ~y:(floats [| v |]) ()
  | R_custom aspect ->
      let coord =
        Option.map (fun aspect -> Coord.cartesian ~aspect ()) aspect
      in
      Mark.v ~name:"custom" ?coord
        [ Mark.bind Role.x (num tensors.(0)) ]
        draw_nothing
  | R_axis (s, grid, title) -> axis ~grid ?title s
  | R_legend (s, show, title) -> legend ~show ?title s
  | R_bind (s, i) -> bind (View.number s ~init:0.) bound.(i)
  | R_layer rs -> layer (List.map build rs)
  | R_grid (widths, rows) -> grid ?widths (List.map (List.map build) rows)
  | R_span (rows, cols, r) -> span ~rows ~cols (build r)
  | R_title (s, k, r) -> title ~align:aligns.(k) s (build r)
  | R_name (s, r) -> name s (build r)
  | R_share (s, indep, r) ->
      share [ (s, if indep then `Independent else `Shared) ] (build r)
  | R_coord (aspect, r) -> coord (Coord.cartesian ?aspect ()) (build r)

let pp_list pp ppf l =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ") pp)
    l

let pp_aspect = Format.pp_print_option Format.pp_print_float
let pp_title = Format.pp_print_option Format.pp_print_string
let other_title = function None -> Some "t" | Some _ -> None

let rec pp_recipe ppf = function
  | R_dot (i, j, k) ->
      Format.fprintf ppf "dot %d %d %a" i j
        (Format.pp_print_option Format.pp_print_int)
        k
  | R_line (i, t) -> Format.fprintf ppf "line %d %b" i t
  | R_bar i -> Format.fprintf ppf "bar %d" i
  | R_mapped (i, k) -> Format.fprintf ppf "mapped %d %d" i k
  | R_level v -> Format.fprintf ppf "level %g" v
  | R_custom a -> Format.fprintf ppf "custom %a" pp_aspect a
  | R_axis (s, g, t) -> Format.fprintf ppf "axis %S %b %a" s g pp_title t
  | R_legend (s, show, t) ->
      Format.fprintf ppf "legend %S %b %a" s show pp_title t
  | R_bind (s, i) -> Format.fprintf ppf "bind %S %d" s i
  | R_layer rs -> Format.fprintf ppf "@[layer %a@]" (pp_list pp_recipe) rs
  | R_grid (ws, rows) ->
      Format.fprintf ppf "@[grid %a %a@]"
        (Format.pp_print_option (pp_list Format.pp_print_float))
        ws
        (pp_list (pp_list pp_recipe))
        rows
  | R_span (rows, cols, r) ->
      Format.fprintf ppf "@[span %d %d (%a)@]" rows cols pp_recipe r
  | R_title (s, k, r) ->
      Format.fprintf ppf "@[title %S %d (%a)@]" s k pp_recipe r
  | R_name (s, r) -> Format.fprintf ppf "@[name %S (%a)@]" s pp_recipe r
  | R_share (s, i, r) ->
      Format.fprintf ppf "@[share %S %b (%a)@]" s i pp_recipe r
  | R_coord (a, r) ->
      Format.fprintf ppf "@[coord %a (%a)@]" pp_aspect a pp_recipe r

let rec gen_recipe depth =
  let open Gen in
  let idx = int_range 0 2 in
  let word = of_list [ "a"; "b" ] in
  let aspect = option (of_list [ 1.; 2. ]) in
  let leaf =
    one_of
      [
        map
          (fun (i, (j, k)) -> R_dot (i, j, k))
          (pair idx (pair idx (option (int_range 0 1))));
        map (fun (i, t) -> R_line (i, t)) (pair idx bool);
        map (fun i -> R_bar i) idx;
        map (fun (i, k) -> R_mapped (i, k)) (pair idx (int_range 0 1));
        map (fun v -> R_level v) (of_list (Array.to_list levels));
        map (fun a -> R_custom a) aspect;
        map
          (fun (s, (g, t)) -> R_axis (s, g, t))
          (pair (of_list [ "x"; "y" ]) (pair bool (option word)));
        map
          (fun (s, (show, t)) -> R_legend (s, show, t))
          (pair word (pair bool (option word)));
        map (fun (s, i) -> R_bind (s, i)) (pair word (int_range 0 1));
      ]
  in
  if depth = 0 then leaf
  else
    let sub = gen_recipe (depth - 1) in
    frequency
      [
        (3, leaf);
        (1, map (fun rs -> R_layer rs) (list ~size:(int_range 0 3) sub));
        ( 1,
          map
            (fun (ws, rows) -> R_grid (ws, rows))
            (pair
               (option (list ~size:(int_range 1 2) (of_list [ 1.; 2. ])))
               (list ~size:(int_range 1 2) (list ~size:(int_range 1 2) sub))) );
        ( 1,
          map
            (fun ((rows, cols), r) -> R_span (rows, cols, r))
            (pair (pair (int_range 1 2) (int_range 1 2)) sub) );
        ( 1,
          map
            (fun (s, (k, r)) -> R_title (s, k, r))
            (pair word (pair (int_range 0 2) sub)) );
        (1, map (fun (s, r) -> R_name (s, r)) (pair word sub));
        ( 1,
          map
            (fun (s, (i, r)) -> R_share (s, i, r))
            (pair (of_list [ "x"; "color" ]) (pair bool sub)) );
        (1, map (fun (a, r) -> R_coord (a, r)) (pair aspect sub));
      ]

let gen_recipe = Gen.with_pp pp_recipe (gen_recipe 3)

(* [variants r] is every recipe that differs from [r] in one argument of one
   node, the near misses an equality that ignores an argument confuses. *)
let rec variants r =
  let other_aspect = function None -> Some 1. | Some _ -> None in
  let here =
    match r with
    | R_dot (i, j, k) ->
        [ R_dot (i, j, match k with None -> Some 0 | Some _ -> None) ]
    | R_line (i, t) -> [ R_line (i, not t) ]
    | R_bar i -> [ R_bar ((i + 1) mod 3) ]
    | R_mapped (i, k) -> [ R_mapped (i, 1 - k) ]
    | R_level v -> [ R_level (if Float.equal v 0. then Float.nan else 0.) ]
    | R_custom a -> [ R_custom (other_aspect a) ]
    | R_axis (s, g, t) -> [ R_axis (s, not g, t); R_axis (s, g, other_title t) ]
    | R_legend (s, show, t) ->
        [ R_legend (s, not show, t); R_legend (s, show, other_title t) ]
    | R_bind (s, i) -> [ R_bind (s, 1 - i) ]
    | R_layer rs -> [ R_layer (R_bar 0 :: rs) ]
    | R_grid (ws, rows) ->
        [
          R_grid ((match ws with None -> Some [ 1. ] | Some _ -> None), rows);
        ]
    | R_span (rows, cols, x) -> [ R_span (3 - rows, cols, x) ]
    | R_title (s, k, x) -> [ R_title (s, (k + 1) mod 3, x) ]
    | R_name (s, x) -> [ R_name ((if s = "a" then "b" else "a"), x) ]
    | R_share (s, i, x) -> [ R_share (s, not i, x) ]
    | R_coord (a, x) -> [ R_coord (other_aspect a, x) ]
  in
  (* [each l] is [l] with one element replaced by one of its variants. *)
  let rec each = function
    | [] -> []
    | x :: rest ->
        List.map (fun x' -> x' :: rest) (variants x)
        @ List.map (fun rest' -> x :: rest') (each rest)
  in
  let rec each_row = function
    | [] -> []
    | row :: rest ->
        List.map (fun row' -> row' :: rest) (each row)
        @ List.map (fun rest' -> row :: rest') (each_row rest)
  in
  let below =
    match r with
    | R_layer rs -> List.map (fun rs -> R_layer rs) (each rs)
    | R_grid (ws, rows) ->
        List.map (fun rows -> R_grid (ws, rows)) (each_row rows)
    | R_span (rows, cols, x) ->
        List.map (fun x -> R_span (rows, cols, x)) (variants x)
    | R_title (s, k, x) -> List.map (fun x -> R_title (s, k, x)) (variants x)
    | R_name (s, x) -> List.map (fun x -> R_name (s, x)) (variants x)
    | R_share (s, i, x) -> List.map (fun x -> R_share (s, i, x)) (variants x)
    | R_coord (a, x) -> List.map (fun x -> R_coord (a, x)) (variants x)
    | R_dot _ | R_line _ | R_bar _ | R_mapped _ | R_level _ | R_custom _
    | R_axis _ | R_legend _ | R_bind _ ->
        []
  in
  here @ below

(* Pairs of equal recipes, of near misses, and of unrelated recipes. *)
let gen_pair =
  let pp ppf (r, r') =
    Format.fprintf ppf "@[<v>%a@,%a@]" pp_recipe r pp_recipe r'
  in
  Gen.with_pp pp
    Gen.(
      let* r = gen_recipe in
      let vs = variants r in
      let* k = int_range 0 (List.length vs - 1) in
      let+ r' =
        frequency
          [ (1, constant r); (2, constant (List.nth vs k)); (1, gen_recipe) ]
      in
      (r, r'))

(* Pairs the recipes cannot state, from [equal]'s and the lifts' contracts. *)
let param_parity = Role.param ~name:"k" ~equal:(fun a b -> a mod 2 = b mod 2)
let param_of k = mark [ Mark.bind param_parity (const k) ]

let param_call () =
  mark [ Mark.bind (Role.param ~name:"k" ~equal:Int.equal) (const 0) ]

let key = View.number "k" ~init:1.
let to_a _ = a

let pairs =
  [
    ( "a nested layer and the flat one",
      false,
      fun () ->
        let b = rule ~x:(num v) () in
        (layer [ layer [ a; b ]; a ], layer [ a; b; a ]) );
    ( "map_range of two fresh closures",
      false,
      fun () ->
        let f () = on_fill (map_range (Color.with_alpha 0.5) (num v)) in
        (f (), f ()) );
    ( "a tensor and its copy",
      false,
      fun () -> (on_x (num v), on_x (num (Nx.copy v))) );
    ( "tensors of two dtypes",
      false,
      fun () -> (on_x (num v), on_x (num (Nx.zeros Nx.float64 [| 3 |]))) );
    ( "marks of two shapes",
      false,
      fun () -> (mark ~shape:[| 2 |] [], mark ~shape:[| 3 |] []) );
    ( "marks of one shape",
      true,
      fun () -> (mark ~shape:[| 2 |] [], mark ~shape:[| 2 |] []) );
    ( "parameters by their role's equality",
      true,
      fun () -> (param_of 1, param_of 3) );
    ( "parameters unequal by their role's equality",
      false,
      fun () -> (param_of 1, param_of 2) );
    ( "parameters made by two calls",
      false,
      fun () -> (param_call (), param_call ()) );
    ( "binds of one key and function",
      true,
      fun () -> (bind key to_a, bind (View.number "k" ~init:1.) to_a) );
    ( "binds of keys of two initial values",
      false,
      fun () -> (bind key to_a, bind (View.number "k" ~init:2.) to_a) );
    ( "binds of keys of two names",
      false,
      fun () -> (bind key to_a, bind (View.number "j" ~init:1.) to_a) );
    ( "a title and its left alignment",
      true,
      fun () -> (title "t" a, title ~align:`Left "t" a) );
    ( "a line and its linear curve",
      true,
      fun () -> (line ~y:(num v) (), line ~curve:Curve.linear ~y:(num v) ()) );
  ]

(* Ruling: every built-in mark rebuilt from the same tensors and values is equal
   to the first. *)
let builtins =
  [
    ("dot", fun () -> dot ~x:(num v) ~y:(num v) ~size:(num v) ());
    ( "line with a curve",
      fun () -> line ~curve:Curve.natural ~stroke:(dim 0) ~y:(num m) () );
    ( "area with a curve",
      fun () -> area ~curve:Curve.natural ~y2:(num v) ~y:(num v) () );
    ("rect", fun () -> rect ~x:(dim 0) ~y:(num v) ());
    ("frame", fun () -> frame ~stroke:(const Color.red) ());
    ("rule at a level", fun () -> rule ~y:(floats [| 0. |]) ());
    ("abline", fun () -> abline ~slope:(const 1.) ~intercept:(num v) ());
    ( "text with offsets",
      fun () ->
        Hugin.text ~dx:2. ~dy:(-1.) ~x:(num v) ~y:(num v) ~text:(num v) () );
    ("a grey image", fun () -> image m);
    ("a batch of RGB images", fun () -> image batch);
    ("contour", fun () -> contour ~fill:(num m) ());
  ]

let equality =
  group "equal"
    [
      prop "figures are equal iff their recipes are" gen_pair (fun (r, r') ->
          (* [compare] equates the nan of a level with itself, as [Float.equal]
             does. *)
          let same = compare r r' = 0 in
          cover "equal recipes" same;
          cover "different recipes" (not same);
          cover "recipes that differ in one argument"
            (List.exists (fun v -> compare v r' = 0) (variants r));
          equal bool same (Hugin.equal (build r) (build r')));
      cases
        ~name:(Format.asprintf "%a" pp_recipe)
        "a figure differs from each of its near misses"
        [
          R_dot (0, 1, None);
          R_dot (0, 1, Some 0);
          R_line (0, false);
          R_mapped (0, 0);
          R_level 0.;
          R_level Float.nan;
          R_custom None;
          R_custom (Some 2.);
          R_axis ("x", false, None);
          R_axis ("x", false, Some "a");
          R_legend ("a", true, None);
          R_legend ("a", true, Some "a");
          R_bind ("a", 0);
          R_layer [ R_bar 0 ];
          R_grid (None, [ [ R_bar 0; R_bar 1 ] ]);
          R_grid (Some [ 2. ], [ [ R_bar 0 ] ]);
          R_span (1, 1, R_bar 0);
          R_title ("a", 0, R_bar 0);
          R_name ("a", R_bar 0);
          R_share ("x", false, R_bar 0);
          R_coord (None, R_bar 0);
        ]
        (fun r ->
          List.iter
            (fun r' ->
              equal
                ~msg:(Format.asprintf "%a" pp_recipe r')
                bool false
                (Hugin.equal (build r) (build r')))
            (variants r));
      cases
        ~name:(fun (n, _, _) -> n)
        "equal on pairs the recipes cannot state" pairs
        (fun (_, same, f) ->
          let f, f' = f () in
          equal bool same (Hugin.equal f f'));
      cases ~name:fst "a built-in mark rebuilt is equal" builtins (fun (_, f) ->
          equal bool true (Hugin.equal (f ()) (f ())));
      prop "a title is its rich text" Gen.string (fun s ->
          equal bool true (Hugin.equal (title s a) (title' (Text.v s) a));
          equal bool true
            (Hugin.equal (axis ~title:s "x") (axis' ~title:(Text.v s) "x")));
    ]

(* Views *)

(* A model of views: a user key is identified by its name and a zoom key by its
   node, and a view binds each to one value of one sort. *)
type value =
  | Number of float
  | Choice of string
  | Interval of (float * float) option
  | Zoom of (float * float) option

type slot = User of string | Zoomed of int

let names = [ "a"; "b" ]
let nodes = [ 0; 1 ]
let at k = if k = 0 then None else Some (Nx.Ptree.Path.v [ Index k ])
let number s = View.number s ~init:1.
let choice s = View.choice s ~init:"i"
let interval s = View.interval s ~init:(Some (0., 1.))

(* Zoom keys of the scale "a", which a user key may also name. *)
let zoom k = View.zoom ?at:(at k) (Scale.linear ~name:"a" ())

let set (slot, value) view =
  match (slot, value) with
  | User s, Number x -> View.set (number s) x view
  | User s, Choice c -> View.set (choice s) c view
  | User s, Interval i -> View.set (interval s) i view
  | Zoomed k, Zoom z -> View.set (zoom k) z view
  | _ -> invalid_arg "set: a value of another sort"

let apply ops = List.fold_left (fun view op -> set op view) View.empty ops

(* [model ops] is the binding of each slot, sorted by slot. *)
let model ops =
  List.fold_left
    (fun m (slot, value) -> (slot, value) :: List.remove_assoc slot m)
    [] ops
  |> List.sort compare

let equal_value a b =
  let interval =
    Option.equal (fun (a, b) (c, d) -> Float.equal a c && Float.equal b d)
  in
  match (a, b) with
  | Number x, Number y -> Float.equal x y
  | Choice x, Choice y -> String.equal x y
  | Interval x, Interval y | Zoom x, Zoom y -> interval x y
  | _ -> false

let equal_model = List.equal (fun (s, a) (s', b) -> s = s' && equal_value a b)

let pp_value ppf = function
  | Number x -> Format.fprintf ppf "number %g" x
  | Choice c -> Format.fprintf ppf "choice %S" c
  | Interval None | Zoom None -> Format.fprintf ppf "none"
  | Interval (Some (a, b)) | Zoom (Some (a, b)) ->
      Format.fprintf ppf "(%g, %g)" a b

let pp_op ppf (slot, value) =
  match slot with
  | User s -> Format.fprintf ppf "%S := %a" s pp_value value
  | Zoomed k -> Format.fprintf ppf "zoom %d := %a" k pp_value value

let gen_op =
  let open Gen in
  let float = of_list ~pp:Format.pp_print_float [ 0.; -0.; 1.; Float.nan ] in
  let span =
    option
      (of_list
         ~pp:(fun ppf (a, b) -> Format.fprintf ppf "(%g, %g)" a b)
         [ (0., 1.); (-0., 0.); (2., 2.) ])
  in
  let user = of_list names in
  one_of
    [
      map (fun (s, x) -> (User s, Number x)) (pair user float);
      map (fun (s, c) -> (User s, Choice c)) (pair user (of_list [ "i"; "j" ]));
      map (fun (s, i) -> (User s, Interval i)) (pair user span);
      map (fun (k, z) -> (Zoomed k, Zoom z)) (pair (of_list nodes) span);
    ]

let gen_ops =
  Gen.with_pp (pp_list pp_op) (Gen.list ~size:(Gen.int_range 0 6) gen_op)

(* Two programs: equal, a program and its model replayed in another order, a
   program with one call dropped, or unrelated. *)
let gen_programs =
  let pp ppf (ops, ops') =
    Format.fprintf ppf "@[<v>%a@,%a@]" (pp_list pp_op) ops (pp_list pp_op) ops'
  in
  Gen.with_pp pp
    Gen.(
      let* ops = gen_ops in
      let+ ops' =
        frequency
          [
            (1, constant ops);
            (2, permutation (model ops));
            (2, subsequence ops);
            (1, gen_ops);
          ]
      in
      (ops, ops'))

let reads_law ops =
  let view = apply ops and m = model ops in
  let bound slot = List.assoc_opt slot m in
  let sort = function
    | Number _ -> 0
    | Choice _ -> 1
    | Interval _ -> 2
    | Zoom _ -> 3
  in
  cover "a name set with two sorts in turn"
    (List.exists
       (fun (s, v) ->
         List.exists (fun (s', v') -> s = s' && sort v <> sort v') ops)
       ops);
  List.iter
    (fun s ->
      let msg = s in
      let n = match bound (User s) with Some (Number x) -> x | _ -> 1. in
      let c = match bound (User s) with Some (Choice c) -> c | _ -> "i" in
      let i =
        match bound (User s) with Some (Interval i) -> i | _ -> Some (0., 1.)
      in
      equal ~msg float_exact n (View.get (number s) view);
      equal ~msg string c (View.get (choice s) view);
      equal ~msg
        (option (pair float_exact float_exact))
        i
        (View.get (interval s) view))
    names;
  List.iter
    (fun k ->
      let z = match bound (Zoomed k) with Some (Zoom z) -> z | _ -> None in
      equal ~msg:(string_of_int k)
        (option (pair float_exact float_exact))
        z
        (View.get (zoom k) view))
    nodes

let views =
  group "View"
    [
      prop "a key reads the last value set on its name and sort, or its initial"
        gen_ops reads_law;
      prop "views are equal iff they bind the same keys to equal values"
        gen_programs (fun (ops, ops') ->
          let same = equal_model (model ops) (model ops') in
          cover "equal views" same;
          cover "equal views set in another order"
            (same && compare ops ops' <> 0);
          cover "different views" (not same);
          equal bool same (View.equal (apply ops) (apply ops')));
    ]

(* Sizes, themes and coordinate systems *)

let color = Testable.make ~pp:Color.pp ~equal:Color.equal
let first s = (Scheme.colors 1 s).(0)

let presentation =
  group "presentation"
    [
      test "mm and dpi convert to points" (fun () ->
          equal (float 1e-12) 72. (Size.mm 25.4);
          equal (float 1e-12) 2. (Size.dpi 144.));
      test "sizes are equal iff they fix the same length to equal values"
        (fun () ->
          equal bool false (Size.equal (Size.figure 1. 2.) (Size.panels 1. 2.));
          equal bool false (Size.equal (Size.panels 1. 2.) (Size.panels 1. 3.));
          equal bool true (Size.equal (Size.panels 1. 2.) (Size.panels 1. 2.)));
      test "the default theme states its defaults" (fun () ->
          let th = Theme.default in
          equal color (Color.gray 0.1) (Theme.ink th);
          equal color Color.white (Theme.paper th);
          equal color (first Scheme.tableau10) (Theme.accent th);
          equal float_exact 10. (Theme.size th);
          equal bool true
            (List.equal Font.equal [ Font.regular; Font.bold ] (Theme.fonts th));
          equal bool true (Scheme.equal Scheme.tableau10 (Theme.palette th));
          equal bool true (Scheme.equal Scheme.viridis (Theme.scheme th));
          equal bool true (Locale.equal Locale.default (Theme.locale th));
          equal bool true (Theme.equal th (Theme.v ())));
      test "the accent defaults to the palette's first colour" (fun () ->
          equal color (first Scheme.dark2)
            (Theme.accent (Theme.v ~palette:Scheme.dark2 ())));
      test "themes differing in accent alone differ" (fun () ->
          equal bool false
            (Theme.equal Theme.default (Theme.v ~accent:Color.red ())));
      test "coordinate systems compare their aspects" (fun () ->
          equal bool true
            (Coord.equal (Coord.cartesian ()) (Coord.cartesian ()));
          equal bool false
            (Coord.equal (Coord.cartesian ()) (Coord.cartesian ~aspect:1. ())));
    ]

let () =
  exit
    (run "hugin figures" [ construction; lifts; equality; views; presentation ])
