(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin
open Windtrap

let f32 shape = Nx.zeros Nx.float32 shape
let i32 shape = Nx.zeros Nx.int32 shape
let i64 shape = Nx.zeros Nx.int64 shape
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

(* Figures have no printer: a failure names them by its test's row or case. *)
let figure = Testable.of_equal Hugin.equal
let path segs = Nx.Ptree.Path.v segs

(* Construction errors. [refused ~sub f] states that [f ()] raises
   [Invalid_argument], with [sub] in its message when given, and [made f] that
   it returns: the rows of a table give each refused value beside the nearest
   values that are made. *)

let refused ?sub f () = raises_match (Exn.invalid_arg ?sub) f
let made f () = ignore (f ())
let checks name rows = cases name ~id:fst rows (fun (_, check) -> check ())

(* pp_warning *)

let pp_warning_ =
  group "pp_warning"
    [
      test "formats the id, then the message" (fun () ->
          let id = path [ Field "a"; Index 1 ]
          and msg = "x: 1 value is missing" in
          in_order
            ~subs:[ Nx.Ptree.Path.to_string id; msg ]
            (Format.asprintf "%a" pp_warning (id, msg)));
    ]

(* equal *)

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

(* A recipe of each node, each paired with every near miss: the examples hold
   every argument the recipes vary. *)
let near_misses =
  List.concat_map
    (fun r -> List.map (fun r' -> (r, r')) (variants r))
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

let recipes_law (r, r') =
  (* [compare] equates the nan of a level with itself, as [Float.equal] does. *)
  let same = compare r r' = 0 in
  cover "equal recipes" same;
  cover "different recipes" (not same);
  cover "recipes that differ in one argument"
    (List.exists (fun v -> compare v r' = 0) (variants r));
  equal bool same (Hugin.equal (build r) (build r'))

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
    (* [equal] compares functions physically. A closure over constants may be
       one value that the compiler builds once, as flambda builds
       [Color.with_alpha 0.5]; a closure over a value known only at run time is
       a new one per call. *)
    ( "map_range of two closures made at run time",
      false,
      fun () ->
        let f () =
          let alpha = Sys.opaque_identity 0.5 in
          on_fill (map_range (Color.with_alpha alpha) (num v))
        in
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

let equal_ =
  group "equal"
    [
      prop "is true iff the figures' recipes are" ~examples:near_misses gen_pair
        recipes_law;
      cases "is true iff the arguments are, on leaves the recipes cannot build"
        ~id:(fun (n, _, _) -> n)
        pairs
        (fun (_, same, f) ->
          let f, f' = f () in
          equal bool same (Hugin.equal f f'));
      cases "holds of a built-in mark and itself rebuilt" ~id:fst builtins
        (fun (_, f) -> equal figure (f ()) (f ()));
    ]

(* Channels *)

(* [copied lift fresh] states that the figure [lift] makes of an array [fresh]
   makes stays equal to the figure of a second such array once the first
   changes: a lift that kept the caller's array would change with it. *)
let copied lift fresh =
  let arr = fresh () in
  let f = lift arr in
  arr.(0) <- arr.(1);
  equal figure (lift (fresh ())) f

let labels () = [| "a"; "b"; "c" |]
let fill c = dot ~x:(num v) ~y:(num v) ~fill:c ()

let num_ =
  group "num"
    [
      checks "raises iff its dtype is complex or boolean or its valid grows it"
        [
          ( "complex64",
            refused (fun () -> on_x (num (Nx.zeros Nx.complex64 [| 2 |]))) );
          ("bool", refused (fun () -> on_x (num (mask [| 2 |]))));
          ( "a valid that grows it",
            refused (fun () -> on_x (num ~valid:(mask [| 3; 3 |]) v)) );
          ( "a valid of another length",
            refused (fun () -> on_x (num ~valid:(mask [| 4 |]) v)) );
          ( "integers with a valid that broadcasts",
            made (fun () ->
                on_x (num ~valid:(mask [| 2; 1 |]) (i32 [| 2; 3 |]))) );
          ("a valid of rank 0", made (fun () -> on_x (num ~valid:(mask [||]) m)));
        ];
    ]

let time_ =
  group "time"
    [
      checks "raises iff its counts are not integers or its valid grows them"
        [
          ( "float64 counts",
            refused (fun () -> time Time.S (Nx.zeros Nx.float64 [| 1 |])) );
          ( "a valid that grows them",
            refused (fun () ->
                on_x (time ~valid:(mask [| 2; 3 |]) Time.S (i64 [| 3 |]))) );
          ("int64 counts", made (fun () -> on_x (time Time.S (i64 [| 3 |]))));
          ( "uint8 counts with a valid that broadcasts",
            made (fun () ->
                on_x
                  (time ~valid:(mask [| 1 |]) Time.Ms
                     (Nx.zeros Nx.uint8 [| 3 |]))) );
        ];
    ]

let cat_ =
  group "cat"
    [
      checks
        "raises iff its codes are not integers, a label repeats or its valid \
         grows them"
        [
          ("floats", refused (fun () -> on_fill (cat (f32 [| 2 |]))));
          ("bools", refused (fun () -> on_fill (cat (mask [| 2 |]))));
          ( "a repeated label",
            refused (fun () ->
                on_fill (cat ~labels:[| "a"; "b"; "a" |] (i32 [| 2 |]))) );
          ( "a valid that grows them",
            refused (fun () ->
                on_fill (cat ~valid:(mask [| 2; 2 |]) (i32 [| 2 |]))) );
          ("uint8", made (fun () -> on_fill (cat (Nx.zeros Nx.uint8 [| 2 |]))));
          ( "uint64 with distinct labels",
            made (fun () ->
                on_fill
                  (cat ~labels:[| "a"; "b" |] (Nx.zeros Nx.uint64 [| 2 |]))) );
        ];
      test "copies its labels" (fun () ->
          copied (fun a -> fill (cat ~labels:a codes)) labels);
    ]

let strings_ =
  group "strings"
    [
      test "copies its array" (fun () ->
          copied (fun a -> fill (strings a)) labels);
    ]

let floats_ =
  group "floats"
    [
      test "copies its array" (fun () ->
          copied (fun a -> rule ~y:(floats a) ()) (fun () -> [| 0.; 1. |]));
    ]

let dim_ =
  group "dim"
    [
      checks
        "makes its mark raise iff the shape lacks its axis or its labels \
         differ in length"
        [
          ( "one past the last axis",
            refused (fun () -> line ~y:(num m) ~stroke:(dim 2) ()) );
          ( "one before the first axis",
            refused (fun () -> line ~y:(num m) ~stroke:(dim (-3)) ()) );
          ( "labels one short",
            refused (fun () ->
                line ~y:(num m) ~stroke:(dim ~labels:[| "a" |] 0) ()) );
          ( "labels one long",
            refused (fun () ->
                line ~y:(num m) ~stroke:(dim ~labels:[| "a"; "b"; "c" |] 0) ())
          );
          ( "a mask that does not broadcast",
            refused (fun () ->
                rect ~fx:(dim ~valid:(mask [| 4 |]) 0) ~y:(num v) ()) );
          ( "the first axis counted from the last",
            made (fun () -> line ~y:(num m) ~stroke:(dim (-2)) ()) );
          ( "labels that repeat",
            made (fun () ->
                line ~y:(num m) ~stroke:(dim ~labels:[| "a"; "a" |] 0) ()) );
          ( "a mask that joins the shape",
            made (fun () -> rect ~fx:(dim ~valid:(mask [| 4 |]) 0) ()) );
        ];
      test "copies its labels" (fun () ->
          copied (fun a -> fill (dim ~labels:a 0)) labels);
      test "joins its valid to the mark's shape" (fun () ->
          equal (array int) [| 4 |]
            (Mark.broadcast [ Mark.bind Role.fx (dim ~valid:(mask [| 4 |]) 0) ]));
    ]

let index_ =
  group "index"
    [
      checks "makes its mark raise iff the shape lacks its axis"
        [
          ( "one past the last axis",
            refused (fun () -> line ~x:(index 2) ~y:(num m) ()) );
          ( "a shape without axes",
            refused (fun () -> line ~y:(num (f32 [||])) ()) );
          ( "the first axis counted from the last",
            made (fun () -> line ~x:(index (-2)) ~y:(num m) ()) );
        ];
    ]

let kind_ =
  let quantitative : float Scale.kind option -> string = function
    | Some Quantitative -> "quantitative"
    | None -> "none"
  and temporal : Time.t Scale.kind option -> string = function
    | Some Temporal -> "temporal"
    | None -> "none"
  and categorical : string Scale.kind option -> string = function
    | Some Categorical -> "categorical"
    | None -> "none"
  in
  group "kind"
    [
      cases "is the kind of the scale a lift reads, or none for a constant"
        ~id:(fun (n, _, _) -> n)
        [
          ("num", "quantitative", fun () -> quantitative (kind (num v)));
          ( "floats",
            "quantitative",
            fun () -> quantitative (kind (floats [| 0. |])) );
          ("index", "quantitative", fun () -> quantitative (kind (index 0)));
          ("a float constant", "none", fun () -> quantitative (kind (const 0.5)));
          ( "time",
            "temporal",
            fun () -> temporal (kind (time Time.S (i64 [| 1 |]))) );
          ("cat", "categorical", fun () -> categorical (kind (cat codes)));
          ( "strings",
            "categorical",
            fun () -> categorical (kind (strings [| "a" |])) );
          ("dim", "categorical", fun () -> categorical (kind (dim 0)));
          ("a string constant", "none", fun () -> categorical (kind (const "a")));
        ]
        (fun (_, expected, got) -> equal string expected (got ()));
    ]

(* Marks *)

let marks =
  group "marks"
    [
      checks "raise iff their channels do not broadcast together"
        [
          ( "x of 3 beside y of 4",
            refused (fun () -> dot ~x:(num v) ~y:(num (f32 [| 4 |])) ()) );
          ( "x of 3 beside y of 1",
            made (fun () -> dot ~x:(num v) ~y:(num (f32 [| 1 |])) ()) );
          ( "constants alone",
            made (fun () -> dot ~x:(const 0.5) ~y:(const 0.5) ()) );
        ];
    ]

let area_ =
  group "area"
    [
      checks "raises iff it stacks with y2 or stacks a y of no quantities"
        [
          ( "a stack with y2",
            refused ~sub:"stack" (fun () ->
                area ~y:(num v) ~y2:(num v) ~stack:`Zero ()) );
          ( "a stack of strings",
            refused ~sub:"stack" (fun () ->
                area ~y:(strings [| "a" |]) ~stack:`Zero ()) );
          ( "a stack of instants",
            refused ~sub:"stack" (fun () ->
                area ~y:(time Time.S (i64 [| 1 |])) ~stack:`Zero ()) );
          ( "a stack of a constant",
            refused ~sub:"stack" (fun () ->
                area ~x:(num v) ~y:(const 0.5) ~stack:`Zero ()) );
          ( "a stack of quantities",
            made (fun () -> area ~y:(num v) ~stack:`Zero ()) );
          ("y and y2", made (fun () -> area ~y:(num v) ~y2:(num v) ()));
        ];
    ]

let rect_ =
  group "rect"
    [
      checks
        "raises iff x2 comes without x, y2 without y, or a stack has not \
         exactly one length"
        [
          ("x2 without x", refused (fun () -> rect ~x2:(num v) ()));
          ("y2 without y", refused (fun () -> rect ~y2:(num v) ()));
          ( "a stack of two categories",
            refused ~sub:"stack" (fun () ->
                rect ~x:(dim 0) ~y:(dim 1) ~fill:(num m) ~stack:`Zero ()) );
          ( "two categories",
            made (fun () -> rect ~x:(dim 0) ~y:(dim 1) ~fill:(num m) ()) );
          ( "a stack of a category and an instant",
            refused ~sub:"stack" (fun () ->
                rect ~x:(dim 0) ~y:(time Time.S (i64 [| 1 |])) ~stack:`Zero ())
          );
          ( "a stack of two lengths",
            refused ~sub:"stack" (fun () ->
                rect ~x:(num v) ~y:(num v) ~stack:`Zero ()) );
          ( "a stack of a category and a length",
            made (fun () -> rect ~x:(dim 0) ~y:(num v) ~stack:`Zero ()) );
          ("x and x2", made (fun () -> rect ~x:(num v) ~x2:(num v) ()));
        ];
    ]

let rule_ =
  group "rule"
    [
      checks
        "raises iff x2 comes without x, y2 without y, or its channels match no \
         case"
        [
          ("no position", refused (fun () -> rule ()));
          ("x2 alone", refused (fun () -> rule ~x2:(num v) ()));
          ("x and x2 only", refused (fun () -> rule ~x:(num v) ~x2:(num v) ()));
          ( "x, x2 and y2",
            refused (fun () -> rule ~x:(num v) ~x2:(num v) ~y2:(num v) ()) );
          ("x alone", made (fun () -> rule ~x:(num v) ()));
          ( "y, x and x2",
            made (fun () -> rule ~y:(num v) ~x:(num v) ~x2:(num v) ()) );
          ( "x, x2, y and y2",
            made (fun () ->
                rule ~x:(num v) ~x2:(num v) ~y:(num v) ~y2:(num v) ()) );
        ];
    ]

let image_ =
  group "image"
    [
      checks "raises iff its rank, channels or dtype is not an image's"
        [
          ("rank 1", refused (fun () -> image (f32 [| 4 |])));
          ("two channels", refused (fun () -> image (f32 [| 2; 2; 2 |])));
          ("five channels", refused (fun () -> image (f32 [| 3; 2; 2; 5 |])));
          ("int32", refused (fun () -> image (i32 [| 2; 2 |])));
          ("bool", refused (fun () -> image (mask [| 2; 2 |])));
          ("uint8 RGB", made (fun () -> image (Nx.zeros Nx.uint8 [| 2; 2; 3 |])));
          ( "RGBA over a datum axis in panels",
            made (fun () -> image ~fx:(dim 0) (f32 [| 5; 2; 2; 4 |])) );
        ];
    ]

let contour_ =
  group "contour"
    [
      checks
        "raises iff its shape, fill, x, y or facets place no field in one panel"
        [
          ("rank 1", refused (fun () -> contour ~fill:(num v) ()));
          ( "a constant fill",
            refused (fun () ->
                contour ~fill:(const Color.red) ~x:(num v)
                  ~y:(num (f32 [| 2; 1 |]))
                  ()) );
          ( "x varying along the rows",
            refused (fun () -> contour ~x:(num m) ~fill:(num m) ()) );
          ( "y varying along the columns",
            refused (fun () -> contour ~y:(num (f32 [| 3 |])) ~fill:(num m) ())
          );
          ( "x the index of the rows",
            refused (fun () -> contour ~x:(index 0) ~fill:(num m) ()) );
          ( "y the index of the columns",
            refused (fun () -> contour ~y:(index 1) ~fill:(num m) ()) );
          ( "faceted along its rows",
            refused (fun () -> contour ~fx:(dim 0) ~fill:(num m) ()) );
          ( "faceted along its columns",
            refused (fun () ->
                contour ~fy:(strings [| "a"; "b"; "c" |]) ~fill:(num m) ()) );
          ( "on the axes of a field",
            made (fun () ->
                contour
                  ~x:(num (f32 [| 3 |]))
                  ~y:(num (f32 [| 2; 1 |]))
                  ~fill:(num m) ()) );
          ( "fields faceted along a datum axis",
            made (fun () ->
                contour ~fill:(num (f32 [| 4; 2; 3 |])) ~fx:(dim 0) ()) );
        ];
    ]

(* Composing *)

let grid_ =
  group "grid"
    [
      checks "raises iff a weight is not finite and positive"
        [
          ( "a width of zero",
            refused (fun () -> grid ~widths:[ 1.; 0. ] [ [ a; a ] ]) );
          ( "a negative height",
            refused (fun () -> grid ~heights:[ -1. ] [ [ a ] ]) );
          ( "a nan width",
            refused (fun () -> grid ~widths:[ Float.nan ] [ [ a ] ]) );
          ( "an infinite width",
            refused (fun () -> grid ~widths:[ Float.infinity ] [ [ a ] ]) );
          ( "the least positive width",
            made (fun () -> grid ~widths:[ Float.min_float ] [ [ a ] ]) );
        ];
    ]

let span_ =
  group "span"
    [
      checks "raises iff it covers less than one row or column"
        [
          ("no rows", refused (fun () -> span ~rows:0 a));
          ("no columns", refused (fun () -> span ~cols:0 a));
          ("one cell", made (fun () -> span a));
        ];
    ]

let share_ =
  group "share"
    [
      checks "raises iff it names a scale twice"
        [
          ( "x twice",
            refused (fun () -> share [ ("x", `Shared); ("x", `Independent) ] a)
          );
          ( "x and y",
            made (fun () -> share [ ("x", `Shared); ("y", `Independent) ] a) );
        ];
    ]

let title_ =
  group "title"
    [
      prop "title s is title' of the text s" Gen.string (fun s ->
          equal figure (title' (Text.v s) a) (title s a));
    ]

let name_ =
  group "name"
    [
      checks "raises iff the name is a generated node's segment"
        [
          ("axis", refused (fun () -> name "axis" a));
          ("legend", refused (fun () -> name "legend" a));
          ("panel", refused (fun () -> name "panel" a));
          ("cell", refused (fun () -> name "cell" a));
          ("axes", made (fun () -> name "axes" a));
        ];
    ]

(* Guides as figures *)

let axis_ =
  group "axis"
    [
      prop "axis ~title:s is axis' of the text s" Gen.string (fun s ->
          equal figure (axis' ~title:(Text.v s) "x") (axis ~title:s "x"));
    ]

let legend_ =
  group "legend"
    [
      checks "raises iff ncol is less than 1"
        [
          ("ncol 0", refused (fun () -> legend ~ncol:0 "color"));
          ("ncol 1", made (fun () -> legend ~ncol:1 "color"));
        ];
    ]

(* Coord *)

let coord_t = Testable.make ~pp:Coord.pp ~equal:Coord.equal

let coord_ =
  group "Coord"
    [
      checks "cartesian raises iff the aspect is not finite and positive"
        [
          ("zero", refused (fun () -> Coord.cartesian ~aspect:0. ()));
          ("negative", refused (fun () -> Coord.cartesian ~aspect:(-1.) ()));
          ("nan", refused (fun () -> Coord.cartesian ~aspect:Float.nan ()));
          ( "infinite",
            refused (fun () -> Coord.cartesian ~aspect:Float.infinity ()) );
          ( "the least positive float",
            made (fun () -> Coord.cartesian ~aspect:Float.min_float ()) );
        ];
      test "equal is true iff the aspects are" (fun () ->
          equal coord_t (Coord.cartesian ()) (Coord.cartesian ());
          equal coord_t
            (Coord.cartesian ~aspect:2. ())
            (Coord.cartesian ~aspect:2. ());
          not_equal coord_t (Coord.cartesian ()) (Coord.cartesian ~aspect:1. ()));
    ]

(* View *)

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
let at k = if k = 0 then None else Some (path [ Index k ])
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

let view = Testable.make ~pp:View.pp ~equal:View.equal

let equal_views_law (ops, ops') =
  let same = equal_model (model ops) (model ops') in
  cover "equal views" same;
  cover "equal views set in another order" (same && compare ops ops' <> 0);
  cover "different views" (not same);
  (if same then equal else not_equal) view (apply ops) (apply ops')

let view_ =
  let unset = View.interval "i" ~init:None in
  group "View"
    [
      checks "interval raises iff its init is not finite and ordered"
        [
          ( "an init out of order",
            refused (fun () -> View.interval "i" ~init:(Some (2., 1.))) );
          ( "an init with nan",
            refused (fun () -> View.interval "i" ~init:(Some (Float.nan, 1.)))
          );
          ( "an init reaching infinity",
            refused (fun () ->
                View.interval "i" ~init:(Some (0., Float.infinity))) );
          ( "an init of one point",
            made (fun () -> View.interval "i" ~init:(Some (1., 1.))) );
        ];
      checks "zoom raises iff its scale is unnamed or categorical"
        [
          ("an unnamed scale", refused (fun () -> View.zoom (Scale.linear ())));
          ( "a band scale",
            refused (fun () -> View.zoom (Scale.band ~name:"b" ())) );
          ( "a named time scale",
            made (fun () -> View.zoom (Scale.time ~name:"t" ())) );
        ];
      checks "set raises iff an interval is not finite and ordered"
        [
          ( "an interval reaching infinity",
            refused (fun () ->
                View.set unset (Some (0., Float.infinity)) View.empty) );
          ( "an interval with nan",
            refused (fun () -> View.set unset (Some (0., Float.nan)) View.empty)
          );
          ( "an interval out of order",
            refused (fun () -> View.set unset (Some (1., 0.)) View.empty) );
          ( "an interval of one point",
            made (fun () -> View.set unset (Some (1., 1.)) View.empty) );
          ("no interval", made (fun () -> View.set unset None View.empty));
        ];
      prop
        "get reads the last value set on its key's name and sort, or its \
         initial"
        gen_ops reads_law;
      prop "equal is true iff the views bind the same keys to equal values"
        gen_programs equal_views_law;
    ]

(* Size *)

let size_t = Testable.make ~pp:Size.pp ~equal:Size.equal

let size_ =
  group "Size"
    [
      checks "figure and panels raise iff a length is not finite and positive"
        [
          ("a figure of zero width", refused (fun () -> Size.figure 0. 10.));
          ( "a figure of negative width",
            refused (fun () -> Size.figure (-1.) 10.) );
          ( "a figure of nan height",
            refused (fun () -> Size.figure 10. Float.nan) );
          ( "panels of infinite width",
            refused (fun () -> Size.panels Float.infinity 10.) );
          ("panels of nan height", refused (fun () -> Size.panels 10. Float.nan));
          ( "a figure of the least positive lengths",
            made (fun () -> Size.figure Float.min_float Float.min_float) );
          ( "panels of the largest lengths",
            made (fun () -> Size.panels Float.max_float Float.max_float) );
        ];
      (* [mm l] is [l *. 72. /. 25.4]: two roundings, each within half an ulp,
         put 25.4 mm within [epsilon_float] of 72 relatively. *)
      test "mm 25.4 is 72 points" (fun () ->
          equal (float_rel ~rel:epsilon_float ~abs:0.) 72. (Size.mm 25.4));
      (* [dpi d] is [d /. 72.], a correctly rounded division, exact for 144. *)
      test "dpi 144 is 2 device pixels per point" (fun () ->
          equal float_exact 2. (Size.dpi 144.));
      test "equal is true iff the sizes fix the same length to equal values"
        (fun () ->
          equal size_t (Size.panels 1. 2.) (Size.panels 1. 2.);
          not_equal size_t (Size.figure 1. 2.) (Size.panels 1. 2.);
          not_equal size_t (Size.panels 1. 2.) (Size.panels 1. 3.));
    ]

(* Theme *)

let theme = Testable.make ~pp:Theme.pp ~equal:Theme.equal
let color = Testable.make ~pp:Color.pp ~equal:Color.equal
let scheme = Testable.make ~pp:Scheme.pp ~equal:Scheme.equal
let font = Testable.make ~pp:Font.pp ~equal:Font.equal
let locale = Testable.make ~pp:Locale.pp ~equal:Locale.equal
let first s = (Scheme.colors 1 s).(0)

(* The colours of viridis that [Theme.dark]'s scheme ramps through. *)
let dark_stops = List.init 13 (fun i -> 0.4 +. (0.05 *. Float.of_int i))

(* The palette Theme.v states for white paper. *)
let light_palette =
  Scheme.palette
    (Array.map
       (fun s -> Result.get_ok (Color.of_hex s))
       [|
         "#4e79a7";
         "#de7c09";
         "#e15759";
         "#60a09b";
         "#418837";
         "#b29100";
         "#9f6b92";
         "#d87984";
         "#936c56";
         "#9d938e";
       |])

(* sRGB's transfer function, from an encoded component to a linear one and
   back. *)
let linear v =
  if v <= 0.04045 then v /. 12.92 else Float.pow ((v +. 0.055) /. 1.055) 2.4

let encode v =
  if v <= 0.0031308 then 12.92 *. v
  else (1.055 *. Float.pow v (1. /. 2.4)) -. 0.055

(* WCAG 2.2's relative luminance of an opaque sRGB colour, and the contrast
   ratio of two colours. *)
let luminance c =
  (0.2126 *. linear (Color.r c))
  +. (0.7152 *. linear (Color.g c))
  +. (0.0722 *. linear (Color.b c))

let contrast a b =
  let la = luminance a and lb = luminance b in
  (Float.max la lb +. 0.05) /. (Float.min la lb +. 0.05)

(* Protanopia and deuteranopia as Machado, Oliveira and Fernandes simulate them
   at severity 1: a matrix on linear sRGB, the result clipped to the gamut. *)
let protanopia =
  [|
    [| 0.152286; 1.052583; -0.204868 |];
    [| 0.114503; 0.786281; 0.099216 |];
    [| -0.003882; -0.048116; 1.051998 |];
  |]

let deuteranopia =
  [|
    [| 0.367322; 0.860646; -0.227968 |];
    [| 0.280085; 0.672501; 0.047413 |];
    [| -0.011820; 0.042940; 0.968881 |];
  |]

let simulate m c =
  let l = [| linear (Color.r c); linear (Color.g c); linear (Color.b c) |] in
  let row i =
    let v =
      (m.(i).(0) *. l.(0)) +. (m.(i).(1) *. l.(1)) +. (m.(i).(2) *. l.(2))
    in
    encode (Float.min 1. (Float.max 0. v))
  in
  Color.v (row 0) (row 1) (row 2)

(* The Euclidean distance of two colours in Oklab. *)
let distance c c' =
  let l, a, b = Color.to_oklab c and l', a', b' = Color.to_oklab c' in
  Float.sqrt
    (((l -. l') *. (l -. l'))
    +. ((a -. a') *. (a -. a'))
    +. ((b -. b') *. (b -. b')))

(* The colours of the palettes of the default and dark themes, each with the
   name of its theme, as neighbours, classes i and i + 1 and the last and the
   first, and as any two. *)
let themes = [ ("default", Theme.default); ("dark", Theme.dark) ]

let neighbours =
  List.concat_map
    (fun (name, th) ->
      let cs = Scheme.table (Theme.palette th) in
      let n = Array.length cs in
      List.init n (fun i -> (name, cs.(i), cs.((i + 1) mod n))))
    themes

let any_two =
  List.concat_map
    (fun (name, th) ->
      let cs = Array.to_list (Scheme.table (Theme.palette th)) in
      List.concat
        (List.mapi
           (fun i c ->
             List.filteri (fun j _ -> j > i) cs
             |> List.map (fun c' -> (name, c, c')))
           cs))
    themes

let pair_id (name, c, c') =
  Printf.sprintf "%s: %s and %s" name (Color.to_hex c) (Color.to_hex c')

(* [blend a get c c'] is the component [get] of [c] at opacity [a] over the
   opaque [c'], composited on encoded components as viewers do. *)
let blend a get c c' = (a *. get c) +. ((1. -. a) *. get c')

let theme_ =
  group "Theme"
    [
      checks
        "v raises iff the size is not finite and positive or fonts is empty"
        [
          ("a size of zero", refused (fun () -> Theme.v ~size:0. ()));
          ("a negative size", refused (fun () -> Theme.v ~size:(-1.) ()));
          ( "an infinite size",
            refused (fun () -> Theme.v ~size:Float.infinity ()) );
          ("no fonts", refused (fun () -> Theme.v ~fonts:[] ()));
          ("one font", made (fun () -> Theme.v ~fonts:[ Font.bold ] ()));
          ( "the least positive size",
            made (fun () -> Theme.v ~size:Float.min_float ()) );
        ];
      test "default is v with the defaults v states" (fun () ->
          let th = Theme.default in
          equal color (Color.gray 0.1) (Theme.ink th);
          equal color Color.white (Theme.paper th);
          equal color (first Scheme.tableau10) (Theme.accent th);
          equal float_exact 10. (Theme.size th);
          equal (list font) [ Font.regular; Font.bold ] (Theme.fonts th);
          equal scheme light_palette (Theme.palette th);
          equal scheme Scheme.viridis (Theme.scheme th);
          equal locale Locale.default (Theme.locale th);
          equal theme (Theme.v ()) th);
      test "the accent defaults to the palette's first colour" (fun () ->
          equal color (first Scheme.dark2)
            (Theme.accent (Theme.v ~palette:Scheme.dark2 ())));
      cases "the presets are the themes of v they state"
        ~id:(fun (n, _, _) -> n)
        [
          ( "dark",
            (fun () ->
              let stops = List.map (Scheme.color Scheme.viridis) dark_stops in
              Theme.v ~ink:(Color.gray 0.92) ~paper:(Color.gray 0.1)
                ~palette:Scheme.tableau10
                ~scheme:(Scheme.ramp (Array.of_list stops))
                ()),
            Theme.dark );
          ("talk", (fun () -> Theme.v ~size:16. ()), Theme.talk);
          ("poster", (fun () -> Theme.v ~size:20. ()), Theme.poster);
        ]
        (fun (_, stated, preset) -> equal theme (stated ()) preset);
      cases "dark's scheme colours contrast with its paper at 3:1 or more"
        ~id:(Printf.sprintf "viridis at %.2f") dark_stops (fun u ->
          let paper = Theme.paper Theme.dark in
          at_least float_exact ~than:3.
            (contrast (Scheme.color Scheme.viridis u) paper));
      (* A figure on a page shows dark's colours on the page's background: on
         dark's paper they keep WCAG's contrast for text, 4.5:1, and for marks,
         3:1, and being lighter than it, they keep more on any darker
         background. *)
      cases "dark's ink and labels contrast with its paper at 4.5:1 or more"
        ~id:(fun (n, _) -> n)
        [ ("titles", 1.); ("tick and legend labels", 0.75) ]
        (fun (_, a) ->
          let paper = Theme.paper Theme.dark and ink = Theme.ink Theme.dark in
          let over =
            Color.v
              (blend a Color.r ink paper)
              (blend a Color.g ink paper)
              (blend a Color.b ink paper)
          in
          at_least float_exact ~than:4.5 (contrast over paper);
          greater float_exact ~than:(luminance paper) (luminance over));
      cases "dark's palette and accent contrast with its paper at 3:1 or more"
        ~id:fst
        (("accent", Theme.accent Theme.dark)
        :: List.map
             (fun c -> ("palette " ^ Color.to_hex c, c))
             (Array.to_list (Scheme.table (Theme.palette Theme.dark))))
        (fun (_, c) ->
          let paper = Theme.paper Theme.dark in
          at_least float_exact ~than:3. (contrast c paper);
          greater float_exact ~than:(luminance paper) (luminance c));
      cases
        "default's palette and accent contrast with its paper at 3:1 or more"
        ~id:fst
        (("accent", Theme.accent Theme.default)
        :: List.map
             (fun c -> ("palette " ^ Color.to_hex c, c))
             (Array.to_list (Scheme.table (Theme.palette Theme.default))))
        (fun (_, c) ->
          at_least float_exact ~than:3. (contrast c (Theme.paper Theme.default)));
      cases "palette neighbours are 0.11 apart in Oklab or more" ~id:pair_id
        neighbours (fun (_, c, c') ->
          at_least float_exact ~than:0.11 (distance c c'));
      cases
        "palette neighbours are 0.07 apart in Oklab or more under protanopia \
         and deuteranopia"
        ~id:pair_id neighbours (fun (_, c, c') ->
          at_least float_exact ~than:0.07
            (distance (simulate protanopia c) (simulate protanopia c'));
          at_least float_exact ~than:0.07
            (distance (simulate deuteranopia c) (simulate deuteranopia c')));
      cases "any two palette colours are 0.07 apart in Oklab or more"
        ~id:pair_id any_two (fun (_, c, c') ->
          at_least float_exact ~than:0.07 (distance c c'));
      prop "dark's scheme contrasts with its paper at 3:1 or more"
        (Gen.float_range 0. 1.) (fun u ->
          let paper = Theme.paper Theme.dark in
          let c = Scheme.color (Theme.scheme Theme.dark) u in
          at_least float_exact ~than:3. (contrast c paper);
          greater float_exact ~than:(luminance paper) (luminance c));
      test "equal tells apart themes that differ in accent alone" (fun () ->
          not_equal theme Theme.default (Theme.v ~accent:Color.red ()));
    ]

(* Extending *)

let role_ =
  group "Role"
    [
      checks "value and param raise iff the name is empty or a built-in role's"
        [
          ( "a parameter of an empty name",
            refused (fun () -> Role.param ~name:"" ~equal:Int.equal) );
          ( "a parameter named x",
            refused (fun () -> Role.param ~name:"x" ~equal:Int.equal) );
          ("a value of an empty name", refused (fun () -> Role.value ~name:""));
          ("a value named fill", refused (fun () -> Role.value ~name:"fill"));
          ("a value named angle", made (fun () -> Role.value ~name:"angle"));
          ( "a parameter named xs",
            made (fun () -> Role.param ~name:"xs" ~equal:Int.equal) );
        ];
    ]

let mark_ =
  group "Mark"
    [
      checks
        "bind raises iff it stacks a role other than x and y or a channel of \
         no quantities"
        [
          ( "a stacked opacity",
            refused ~sub:"stack" (fun () ->
                Mark.bind ~stack:`Zero Role.opacity (num v)) );
          ( "a stacked dim on y",
            refused ~sub:"stack" (fun () ->
                Mark.bind ~stack:`Zero Role.y (dim 0)) );
          ( "stacked quantities on y",
            made (fun () -> Mark.bind ~stack:`Zero Role.y (num v)) );
        ];
      checks
        "broadcast raises iff the shape has a negative dimension or a dim does \
         not fit"
        [
          ( "a negative dimension",
            refused (fun () -> Mark.broadcast ~shape:[| -1 |] []) );
          ( "a dim past the shape",
            refused (fun () -> Mark.broadcast [ Mark.bind Role.fill (dim 0) ])
          );
          ( "a dim of a shape given",
            made (fun () ->
                Mark.broadcast ~shape:[| 2 |] [ Mark.bind Role.fill (dim 0) ])
          );
        ];
      checks "v raises iff its bindings or its shape break a rule it states"
        [
          ( "a role bound twice",
            refused (fun () ->
                mark [ Mark.bind Role.x (num v); Mark.bind Role.x (num v) ]) );
          ( "two value roles of one name",
            refused (fun () ->
                mark
                  [
                    Mark.bind (Role.value ~name:"a") (num v);
                    Mark.bind (Role.value ~name:"a") (num v);
                  ]) );
          ( "two parameters of one name",
            refused (fun () ->
                mark
                  [
                    Mark.bind (Role.param ~name:"a" ~equal:Int.equal) (const 0);
                    Mark.bind (Role.param ~name:"a" ~equal:Int.equal) (const 0);
                  ]) );
          ( "x2 without x",
            refused (fun () -> mark [ Mark.bind Role.x2 (num v) ]) );
          ( "y2 without y",
            refused (fun () -> mark [ Mark.bind Role.y2 (num v) ]) );
          ( "x of quantities and x2 of categories",
            refused (fun () ->
                mark [ Mark.bind Role.x (num v); Mark.bind Role.x2 (cat codes) ])
          );
          ( "y of categories and y2 of quantities",
            refused (fun () ->
                mark [ Mark.bind Role.y (cat codes); Mark.bind Role.y2 (num v) ])
          );
          ( "a stacked y and y2",
            refused ~sub:"stack" (fun () ->
                mark
                  [
                    Mark.bind ~stack:`Zero Role.y (num v);
                    Mark.bind Role.y2 (num v);
                  ]) );
          ( "x and y stacked",
            refused ~sub:"stack" (fun () ->
                mark
                  [
                    Mark.bind ~stack:`Zero Role.x (num v);
                    Mark.bind ~stack:`Zero Role.y (num v);
                  ]) );
          ( "a scale on the text role",
            refused (fun () ->
                mark [ Mark.bind Role.text (num ~scale:(Scale.log ()) v) ]) );
          ( "a title on the text role",
            refused (fun () ->
                mark [ Mark.bind Role.text (strings ~title:"t" [| "a" |]) ]) );
          ( "a title on a value role",
            refused (fun () ->
                mark [ Mark.bind (Role.value ~name:"angle") (num ~title:"t" v) ])
          );
          ( "a negative dimension",
            refused (fun () -> mark ~shape:[| 2; -1 |] []) );
          ( "a shape the channels do not broadcast with",
            refused (fun () -> mark ~shape:[| 2 |] [ Mark.bind Role.x (num v) ])
          );
          ( "x of quantities and x2 a constant",
            made (fun () ->
                mark [ Mark.bind Role.x (num v); Mark.bind Role.x2 (const 1.) ])
          );
          ( "a stacked y and an x",
            made (fun () ->
                mark
                  [
                    Mark.bind Role.x (num v);
                    Mark.bind ~stack:`Zero Role.y (num v);
                  ]) );
          ( "strings beside a tensor of their length",
            made (fun () ->
                mark
                  [
                    Mark.bind Role.x (num v);
                    Mark.bind Role.fill (strings [| "a"; "b"; "c" |]);
                  ]) );
        ];
    ]

let () =
  exit
    (run "hugin figures"
       [
         pp_warning_;
         equal_;
         num_;
         time_;
         cat_;
         strings_;
         floats_;
         dim_;
         index_;
         kind_;
         marks;
         area_;
         rect_;
         rule_;
         image_;
         contour_;
         coord_;
         view_;
         grid_;
         span_;
         share_;
         title_;
         name_;
         axis_;
         legend_;
         size_;
         theme_;
         role_;
         mark_;
       ])
