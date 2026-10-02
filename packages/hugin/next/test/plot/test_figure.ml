(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_next
open Windtrap

let invalid f = raises_match (Exn.invalid_arg ?substring:None) f

(* [rejects base rows] is one test per row, each stating that its thunk raises
   [Invalid_argument]. *)
let rejects base rows = cases ~name:fst base rows (fun (_, f) -> invalid f)

(* [accepts base rows] is one test per row, each stating that its thunk
   returns. *)
let accepts base rows =
  cases ~name:fst base rows (fun (_, f) -> ignore (f () : t))

let f32 shape = Nx.zeros Nx.float32 shape
let i32 shape = Nx.zeros Nx.int32 shape
let mask shape = Nx.full Nx.bool shape true
let v = f32 [| 3 |]
let m = f32 [| 2; 3 |]
let on_x c = dot ~x:c ~y:(const 0.5) ()
let on_fill c = dot ~x:(const 0.5) ~y:(const 0.5) ~fill:c ()

(* Lifts *)

let lifts =
  group "lifts"
    [
      rejects "refuse"
        [
          ( "num of complex64",
            fun () -> on_x @@ num (Nx.zeros Nx.complex64 [| 2 |]) );
          ("num of bool", fun () -> on_x @@ num (mask [| 2 |]));
          ( "num with a valid that grows",
            fun () -> on_x @@ num ~valid:(mask [| 3; 3 |]) v );
          ( "num with a valid of another length",
            fun () -> on_x @@ num ~valid:(mask [| 4 |]) v );
          ("cat of floats", fun () -> on_fill @@ cat (f32 [| 2 |]));
          ("cat of bools", fun () -> on_fill @@ cat (mask [| 2 |]));
          ( "cat with a repeated label",
            fun () -> on_fill @@ cat ~labels:[| "a"; "b"; "a" |] (i32 [| 2 |])
          );
          ( "cat with a valid that grows",
            fun () -> on_fill @@ cat ~valid:(mask [| 2; 2 |]) (i32 [| 2 |]) );
        ];
      test "num accepts integers and a valid that broadcasts" (fun () ->
          ignore
            (num ~valid:(mask [| 2; 1 |]) (i32 [| 2; 3 |])
              : (float, float) channel);
          ignore (num ~valid:(mask [| 3 |]) m : (float, float) channel);
          ignore (num ~valid:(mask [||]) m : (float, float) channel));
      test "cat accepts every integer dtype" (fun () ->
          ignore (cat (Nx.zeros Nx.uint8 [| 2 |]) : (string, string) channel);
          ignore (cat (Nx.zeros Nx.int64 [| 2 |]) : (string, string) channel);
          ignore (cat (Nx.zeros Nx.uint64 [| 2 |]) : (string, string) channel));
      test "dim labels may repeat" (fun () ->
          ignore (line ~y:(num m) ~stroke:(dim ~labels:[| "a"; "a" |] 0) () : t));
      (* A lift that kept the caller's array would change with it. *)
      test "labels are copied" (fun () ->
          let c = i32 [| 3 |] in
          let fill labels =
            dot ~x:(num v) ~y:(num v) ~fill:(cat ~labels c) ()
          in
          let labels = [| "a"; "b" |] in
          let f = fill labels in
          labels.(0) <- "z";
          equal bool true (Hugin_next.equal f (fill [| "a"; "b" |])));
      test "strings are copied" (fun () ->
          let fill a = dot ~x:(num v) ~y:(num v) ~fill:(strings a) () in
          let a = [| "a"; "b"; "c" |] in
          let f = fill a in
          a.(0) <- "z";
          equal bool true (Hugin_next.equal f (fill [| "a"; "b"; "c" |])));
      test "dim labels are copied" (fun () ->
          let fill labels =
            dot ~x:(num v) ~y:(num v) ~fill:(dim ~labels 0) ()
          in
          let labels = [| "a"; "b"; "c" |] in
          let f = fill labels in
          labels.(0) <- "z";
          equal bool true (Hugin_next.equal f (fill [| "a"; "b"; "c" |])));
    ]

(* Marks *)

let marks =
  group "marks"
    [
      rejects "refuse"
        [
          ( "channels that do not broadcast",
            fun () -> dot ~x:(num v) ~y:(num (f32 [| 4 |])) () );
          ( "dim one past the last axis",
            fun () -> line ~y:(num m) ~stroke:(dim 2) () );
          ( "dim one before the first axis",
            fun () -> line ~y:(num m) ~stroke:(dim (-3)) () );
          ( "index one past the last axis",
            fun () -> line ~x:(index 2) ~y:(num m) () );
          ( "index of a shape without axes",
            fun () -> line ~y:(num (f32 [||])) () );
          ( "dim labels one short",
            fun () -> line ~y:(num m) ~stroke:(dim ~labels:[| "a" |] 0) () );
          ( "dim labels one long",
            fun () ->
              line ~y:(num m) ~stroke:(dim ~labels:[| "a"; "b"; "c" |] 0) () );
          ( "a dim mask that does not broadcast",
            fun () -> rect ~fx:(dim ~valid:(mask [| 4 |]) 0) ~y:(num v) () );
          ("rect with x2 without x", fun () -> rect ~x2:(num v) ());
          ("rect with y2 without y", fun () -> rect ~y2:(num v) ());
          ( "Mark.v with x of quantities and x2 of categories",
            fun () ->
              Mark.v ~name:"r"
                [
                  Mark.bind Role.x (num v);
                  Mark.bind Role.x2 (cat (i32 [| 3 |]));
                ]
                (fun _ -> Picture.empty) );
          ("rule without positions", fun () -> rule ());
          ("rule with x2 alone", fun () -> rule ~x2:(num v) ());
          ("rule with x and x2 only", fun () -> rule ~x:(num v) ~x2:(num v) ());
          ( "rule with x, x2 and y2",
            fun () -> rule ~x:(num v) ~x2:(num v) ~y2:(num v) () );
          ("image of rank 1", fun () -> image (f32 [| 4 |]));
          ("image with two channels", fun () -> image (f32 [| 2; 2; 2 |]));
          ("image with five channels", fun () -> image (f32 [| 3; 2; 2; 5 |]));
          ("image of int32", fun () -> image (i32 [| 2; 2 |]));
          ("image of bool", fun () -> image (mask [| 2; 2 |]));
          ("contour of rank 1", fun () -> contour ~fill:(num v) ());
          ( "contour with a constant fill",
            fun () ->
              contour ~fill:(const Color.red) ~x:(num v)
                ~y:(num (f32 [| 2; 1 |]))
                () );
          ( "contour with x varying along the rows",
            fun () -> contour ~x:(num m) ~fill:(num m) () );
          ( "contour with y varying along the columns",
            fun () -> contour ~y:(num (f32 [| 3 |])) ~fill:(num m) () );
          ( "contour with x the index of the rows",
            fun () -> contour ~x:(index 0) ~fill:(num m) () );
          ( "contour with y the index of the columns",
            fun () -> contour ~y:(index 1) ~fill:(num m) () );
        ];
      accepts "accept"
        [
          ("a mark of constants", fun () -> dot ~x:(const 0.5) ~y:(const 0.5) ());
          ( "index of the first axis counted from the last",
            fun () -> line ~x:(index (-2)) ~y:(num m) () );
          ( "a dim mask that joins the shape",
            fun () -> rect ~fx:(dim ~valid:(mask [| 4 |]) 0) () );
          ("rule with x alone", fun () -> rule ~x:(num v) ());
          ("rule with y alone", fun () -> rule ~y:(num v) ());
          ("rule with x and y", fun () -> rule ~x:(num v) ~y:(num v) ());
          ( "rule with x, y and y2",
            fun () -> rule ~x:(num v) ~y:(num v) ~y2:(num v) () );
          ( "rule with y, x and x2",
            fun () -> rule ~y:(num v) ~x:(num v) ~x2:(num v) () );
          ( "rule with x, x2, y and y2",
            fun () -> rule ~x:(num v) ~x2:(num v) ~y:(num v) ~y2:(num v) () );
          ("image of rank 2", fun () -> image (f32 [| 2; 2 |]));
          ( "image of uint8 rgb",
            fun () -> image (Nx.zeros Nx.uint8 [| 2; 2; 3 |]) );
          ( "images over a datum axis",
            fun () -> image ~fx:(dim 0) (f32 [| 5; 2; 2; 4 |]) );
          ("contour with default positions", fun () -> contour ~fill:(num m) ());
          ( "contour on the axes of a field",
            fun () ->
              contour
                ~x:(num (f32 [| 3 |]))
                ~y:(num (f32 [| 2; 1 |]))
                ~fill:(num m) () );
          ( "contour of a batch of fields",
            fun () -> contour ~fill:(num (f32 [| 4; 2; 3 |])) ~fx:(dim 0) () );
        ];
    ]

let draw_nothing (_ : Mark.rows) = Picture.empty

let mark_v =
  group "Mark.v"
    [
      rejects "refuse"
        [
          ( "a role bound twice",
            fun () ->
              Mark.v ~name:"m"
                [ Mark.bind Role.x (num v); Mark.bind Role.x (num v) ]
                draw_nothing );
          ( "two value roles of one name",
            fun () ->
              Mark.v ~name:"m"
                [
                  Mark.bind (Role.value ~name:"a") (num v);
                  Mark.bind (Role.value ~name:"a") (num v);
                ]
                draw_nothing );
          ( "x2 without x",
            fun () ->
              Mark.v ~name:"m" [ Mark.bind Role.x2 (num v) ] draw_nothing );
          ( "y2 without y",
            fun () ->
              Mark.v ~name:"m" [ Mark.bind Role.y2 (num v) ] draw_nothing );
          ( "y of categories and y2 of quantities",
            fun () ->
              Mark.v ~name:"m"
                [
                  Mark.bind Role.y (cat (i32 [| 3 |]));
                  Mark.bind Role.y2 (num v);
                ]
                draw_nothing );
          ( "a scale on the text role",
            fun () ->
              Mark.v ~name:"m"
                [ Mark.bind Role.text (num ~scale:(Scale.log ()) v) ]
                draw_nothing );
          ( "a title on the text role",
            fun () ->
              Mark.v ~name:"m"
                [ Mark.bind Role.text (strings ~title:(Text.v "t") [| "a" |]) ]
                draw_nothing );
          ( "a title on a value role",
            fun () ->
              Mark.v ~name:"m"
                [
                  Mark.bind (Role.value ~name:"angle")
                    (num ~title:(Text.v "t") v);
                ]
                draw_nothing );
        ];
      accepts "accept"
        [
          ( "x of quantities and x2 a constant",
            fun () ->
              Mark.v ~name:"m"
                [ Mark.bind Role.x (num v); Mark.bind Role.x2 (const 1.) ]
                draw_nothing );
          ( "strings beside a tensor of their length",
            fun () ->
              Mark.v ~name:"m"
                [
                  Mark.bind Role.x (num v);
                  Mark.bind Role.fill (strings [| "a"; "b"; "c" |]);
                ]
                draw_nothing );
        ];
      rejects "Role.value refuses"
        [
          ( "an empty name",
            fun () ->
              ignore (Role.value ~name:"");
              layer [] );
          ( "the name of a built-in role",
            fun () ->
              ignore (Role.value ~name:"fill");
              layer [] );
        ];
    ]

(* Composing *)

let a = dot ~x:(num v) ~y:(num v) ()

let composing =
  group "composing"
    [
      rejects "refuse"
        [
          ("name axis", fun () -> name "axis" a);
          ("name legend", fun () -> name "legend" a);
          ("name panel", fun () -> name "panel" a);
          ("name cell", fun () -> name "cell" a);
          ("a zero width", fun () -> grid ~widths:[ 1.; 0. ] [ [ a; a ] ]);
          ("a negative height", fun () -> grid ~heights:[ -1. ] [ [ a ] ]);
          ("a nan width", fun () -> grid ~widths:[ Float.nan ] [ [ a ] ]);
          ( "an infinite width",
            fun () -> grid ~widths:[ Float.infinity ] [ [ a ] ] );
          ("a span of no rows", fun () -> span ~rows:0 a);
          ("a span of no columns", fun () -> span ~cols:0 a);
          ( "a scale shared twice",
            fun () -> share [ ("x", `Shared); ("x", `Independent) ] a );
        ];
      accepts "accept"
        [
          ("name axes", fun () -> name "axes" a);
          ( "the least positive weight",
            fun () -> grid ~widths:[ Float.min_float ] [ [ a ] ] );
          ( "weights of any length",
            fun () -> grid ~widths:[ 1.; 2.; 3. ] [ [ a ] ] );
          ("a span of one cell", fun () -> span a);
        ];
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
  | R_custom of float option (* The aspect of the coordinate system implied. *)
  | R_axis of string * bool
  | R_legend of string * bool
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
let aligns : Text.Layout.halign array = [| `Center; `Left; `Right |]
let custom_draw (_ : Mark.rows) = Picture.empty
let bound = Array.init 2 (fun i _ -> dot ~x:(num tensors.(i)) ~y:(const 0.5) ())

let rec build = function
  | R_dot (i, j, k) ->
      let valid = Option.map (fun k -> masks.(k)) k in
      dot ~x:(num ?valid tensors.(i)) ~y:(num tensors.(j)) ()
  | R_line (i, titled) ->
      let x =
        if titled then index ~title:(Text.v "step") (-1) else index (-1)
      in
      line ~x ~y:(num tensors.(i)) ()
  | R_bar i -> rect ~x:(dim 0) ~y:(num tensors.(i)) ()
  | R_mapped (i, k) ->
      dot ~x:(const 0.5)
        ~y:(num tensors.(i))
        ~fill:(map_range colours.(k) (num tensors.(i)))
        ()
  | R_custom aspect ->
      let coord =
        Option.map (fun aspect -> Coord.cartesian ~aspect ()) aspect
      in
      Mark.v ~name:"custom" ?coord
        [ Mark.bind Role.x (num tensors.(0)) ]
        custom_draw
  | R_axis (s, grid) -> axis ~grid s
  | R_legend (s, show) -> legend ~show s
  | R_bind (s, i) -> bind (View.number s ~init:0.) bound.(i)
  | R_layer rs -> layer (List.map build rs)
  | R_grid (widths, rows) -> grid ?widths (List.map (List.map build) rows)
  | R_span (rows, cols, r) -> span ~rows ~cols (build r)
  | R_title (s, k, r) -> title ~align:aligns.(k) (Text.v s) (build r)
  | R_name (s, r) -> name s (build r)
  | R_share (s, indep, r) ->
      share [ (s, if indep then `Independent else `Shared) ] (build r)
  | R_coord (aspect, r) -> coord (Coord.cartesian ?aspect ()) (build r)

let pp_list pp ppf l =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ") pp)
    l

let pp_aspect = Format.pp_print_option Format.pp_print_float

let rec pp_recipe ppf = function
  | R_dot (i, j, k) ->
      Format.fprintf ppf "dot %d %d %a" i j
        (Format.pp_print_option Format.pp_print_int)
        k
  | R_line (i, t) -> Format.fprintf ppf "line %d %b" i t
  | R_bar i -> Format.fprintf ppf "bar %d" i
  | R_mapped (i, k) -> Format.fprintf ppf "mapped %d %d" i k
  | R_custom a -> Format.fprintf ppf "custom %a" pp_aspect a
  | R_axis (s, g) -> Format.fprintf ppf "axis %S %b" s g
  | R_legend (s, show) -> Format.fprintf ppf "legend %S %b" s show
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
        map (fun a -> R_custom a) aspect;
        map (fun (s, g) -> R_axis (s, g)) (pair (of_list [ "x"; "y" ]) bool);
        map (fun (s, show) -> R_legend (s, show)) (pair word bool);
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
    | R_custom a -> [ R_custom (other_aspect a) ]
    | R_axis (s, g) -> [ R_axis (s, not g) ]
    | R_legend (s, show) -> [ R_legend (s, not show) ]
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
    | R_dot _ | R_line _ | R_bar _ | R_mapped _ | R_custom _ | R_axis _
    | R_legend _ | R_bind _ ->
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

let equality =
  group "equal"
    [
      prop "figures are equal iff their recipes are" gen_pair (fun (r, r') ->
          let same = r = r' in
          cover "equal recipes" same;
          cover "different recipes" (not same);
          cover "recipes that differ in one argument"
            (List.exists (( = ) r') (variants r));
          equal bool same (Hugin_next.equal (build r) (build r')));
      cases
        ~name:(Format.asprintf "%a" pp_recipe)
        "a figure differs from each of its near misses"
        [
          R_dot (0, 1, None);
          R_dot (0, 1, Some 0);
          R_line (0, false);
          R_mapped (0, 0);
          R_custom None;
          R_custom (Some 2.);
          R_axis ("x", false);
          R_legend ("a", true);
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
                (Hugin_next.equal (build r) (build r')))
            (variants r));
      test "a nested layer is not the flat layer" (fun () ->
          let b = rule ~x:(num v) () in
          equal bool false
            (Hugin_next.equal (layer [ layer [ a; b ]; a ]) (layer [ a; b; a ])));
      test "a map_range of a fresh closure is another figure" (fun () ->
          let f alpha =
            dot ~x:(num v) ~y:(num v)
              ~fill:(map_range (Color.with_alpha alpha) (num v))
              ()
          in
          equal bool false (Hugin_next.equal (f 0.5) (f 0.5)));
      test "a copied tensor is another figure" (fun () ->
          equal bool false
            (Hugin_next.equal
               (dot ~x:(num v) ~y:(num v) ())
               (dot ~x:(num (Nx.copy v)) ~y:(num v) ())));
      test "a tensor of another dtype is another figure" (fun () ->
          let w = Nx.zeros Nx.float64 [| 3 |] in
          equal bool false
            (Hugin_next.equal
               (dot ~x:(num v) ~y:(num v) ())
               (dot ~x:(num w) ~y:(num v) ())));
      test "a title's default alignment is centre" (fun () ->
          equal bool true
            (Hugin_next.equal
               (title (Text.v "t") a)
               (title ~align:`Center (Text.v "t") a)));
      test "a binding of the same key and function is equal" (fun () ->
          let k = View.number "k" ~init:1. in
          let fn _ = a in
          equal bool true
            (Hugin_next.equal (bind k fn) (bind (View.number "k" ~init:1.) fn));
          equal bool false
            (Hugin_next.equal (bind k fn) (bind (View.number "k" ~init:2.) fn));
          equal bool false
            (Hugin_next.equal (bind k fn) (bind (View.number "j" ~init:1.) fn)));
      test "line's default curve is linear" (fun () ->
          equal bool true
            (Hugin_next.equal
               (line ~y:(num v) ())
               (line ~curve:Curve.linear ~y:(num v) ())));
    ]

(* Views *)

let view_t = Testable.make ~pp:View.pp ~equal:View.equal

let views =
  let n = View.number "n" ~init:1. in
  let c = View.choice "n" ~init:"a" in
  let i = View.interval "i" ~init:None in
  group "View"
    [
      rejects "refuse"
        [
          ( "an interval init out of order",
            fun () ->
              ignore (View.interval "i" ~init:(Some (2., 1.)));
              layer [] );
          ( "an interval init with nan",
            fun () ->
              ignore (View.interval "i" ~init:(Some (Float.nan, 1.)));
              layer [] );
          ( "an interval set to infinity",
            fun () ->
              ignore (View.set i (Some (0., Float.infinity)) View.empty);
              layer [] );
          ( "a zoom of an unnamed scale",
            fun () ->
              ignore (View.zoom (Scale.linear ()));
              layer [] );
          ( "a zoom of a band scale",
            fun () ->
              ignore (View.zoom (Scale.band ~name:"b" ()));
              layer [] );
        ];
      test "an unbound key has its initial value" (fun () ->
          equal float_exact 1. (View.get n View.empty));
      prop "get reads what set binds"
        Gen.(pair float float)
        (fun (x, y) ->
          let view = View.set n x View.empty in
          equal float_exact x (View.get n view);
          equal float_exact y (View.get n (View.set n y view)));
      test "set replaces a key of the same name whatever its sort" (fun () ->
          let view = View.set n 2. View.empty |> View.set c "b" in
          equal float_exact 1. (View.get n view);
          equal string "b" (View.get c view));
      test "views binding the same keys in another order are equal" (fun () ->
          let j = View.number "j" ~init:0. in
          equal view_t
            (View.set n 2. (View.set j 3. View.empty))
            (View.set j 3. (View.set n 2. View.empty)));
      test "views compare floats by Float.equal, nan equal to nan" (fun () ->
          equal view_t
            (View.set n Float.nan View.empty)
            (View.set n Float.nan View.empty);
          not_equal view_t (View.set n 0. View.empty) (View.set n 1. View.empty));
      test "a zoom key is not a user key of its scale's name" (fun () ->
          let z = View.zoom (Scale.linear ~name:"n" ()) in
          let view = View.set z (Some (0., 1.)) View.empty in
          equal float_exact 1. (View.get n view);
          equal
            (option (pair float_exact float_exact))
            (Some (0., 1.))
            (View.get z view));
      test "zooms at two nodes are two keys" (fun () ->
          let s = Scale.linear ~name:"s" () in
          let z0 = View.zoom s
          and z1 = View.zoom ~at:(Nx.Ptree.Path.v [ Index 1 ]) s in
          let view = View.set z0 (Some (0., 1.)) View.empty in
          equal (option (pair float_exact float_exact)) None (View.get z1 view));
    ]

(* Sizes, themes and coordinate systems *)

let presentation =
  group "presentation"
    [
      rejects "refuse"
        [
          ( "a figure of zero width",
            fun () ->
              ignore (Size.figure 0. 10.);
              layer [] );
          ( "a figure of nan height",
            fun () ->
              ignore (Size.figure 10. Float.nan);
              layer [] );
          ( "panels of infinite width",
            fun () ->
              ignore (Size.panels Float.infinity 10.);
              layer [] );
          ( "a theme of size zero",
            fun () ->
              ignore (Theme.v ~size:0. ());
              layer [] );
          ( "a theme without fonts",
            fun () ->
              ignore (Theme.v ~fonts:[] ());
              layer [] );
          ( "an aspect of zero",
            fun () ->
              ignore (Coord.cartesian ~aspect:0. ());
              layer [] );
          ( "an aspect of nan",
            fun () ->
              ignore (Coord.cartesian ~aspect:Float.nan ());
              layer [] );
        ];
      test "mm and dpi convert to points" (fun () ->
          equal (float 1e-12) 72. (Size.mm 25.4);
          equal (float 1e-12) 2. (Size.dpi 144.));
      test "sizes fixing different lengths differ" (fun () ->
          equal bool false (Size.equal (Size.figure 1. 2.) (Size.panels 1. 2.));
          equal bool true (Size.equal (Size.panels 1. 2.) (Size.panels 1. 2.)));
      test "the accent defaults to the palette's first colour" (fun () ->
          let first s = (Scheme.colors 1 s).(0) in
          let c = Testable.make ~pp:Color.pp ~equal:Color.equal in
          equal c (first Scheme.tableau10) (Theme.accent Theme.default);
          equal c (first Scheme.dark2)
            (Theme.accent (Theme.v ~palette:Scheme.dark2 ())));
      test "the default theme states its defaults" (fun () ->
          let th = Theme.default in
          let c = Testable.make ~pp:Color.pp ~equal:Color.equal in
          equal c (Color.gray 0.1) (Theme.ink th);
          equal c Color.white (Theme.paper th);
          equal float_exact 10. (Theme.size th);
          equal bool true
            (List.equal Font.equal [ Font.regular; Font.bold ] (Theme.fonts th));
          equal bool true (Scheme.equal Scheme.viridis (Theme.scheme th)));
      test "themes differing in accent alone differ" (fun () ->
          equal bool false
            (Theme.equal Theme.default (Theme.v ~accent:Color.red ()));
          equal bool true (Theme.equal Theme.default (Theme.v ())));
      test "coordinate systems compare their aspects" (fun () ->
          equal bool true
            (Coord.equal (Coord.cartesian ()) (Coord.cartesian ()));
          equal bool false
            (Coord.equal (Coord.cartesian ()) (Coord.cartesian ~aspect:1. ())));
    ]

let () =
  exit
    (run "hugin.next figures"
       [ lifts; marks; mark_v; composing; equality; views; presentation ])
