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
let root = Nx.Ptree.Path.root

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
let resolved = Testable.make ~pp:Resolved.pp ~equal:Resolved.equal
let printed r = Format.asprintf "%a" Resolved.pp r
let id = Testable.make ~pp:Nx.Ptree.Path.pp ~equal:Nx.Ptree.Path.equal
let warning = pair id string

(* The domain of a categorical scale, its labels or its indices and their
   texts. *)
let categories s =
  match Scale.domain s with
  | Scale.Categories (Scale.Labels l) ->
      List.map (fun l -> (-1, l)) (Array.to_list l)
  | Scale.Categories (Scale.Indices ix) -> Array.to_list ix

let occurs s sub =
  let n = String.length s and m = String.length sub in
  let rec at i = i + m <= n && (String.sub s i m = sub || at (i + 1)) in
  at 0

(* [fails_naming subs f] states that [f] raises [Invalid_argument] with a
   message holding each of [subs]. *)
let fails_naming subs f =
  raises_match
    (function Invalid_argument m -> List.for_all (occurs m) subs | _ -> false)
    f

let invalid f = raises_match (Exn.invalid_arg ?substring:None) f
let draw_nothing (_ : Mark.rows) = Picture.empty

let dot1 ?fill ?fx x y =
  dot ?fill ?fx ~x:(num ~scale:exact (f64 x)) ~y:(num (f64 y)) ()

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
            (layer [ name "a" (title "t" (name "b" (dot1 [| 1. |] [| 1. |]))) ])),
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
          resolve (layer [ title "a" (layer []); title "b" (layer []) ])),
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
      ( "two axes for one scale with different titles",
        (fun () ->
          resolve
            (layer
               [
                 dot1 [| 1. |] [| 1. |];
                 axis ~title:"a" "x";
                 axis ~title:"b" "x";
               ])),
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
      ( "two legends for one scale with different titles",
        (fun () ->
          resolve
            (layer
               [
                 dot ~fill:(num x) ~x:(num x) ~y:(num x) ();
                 legend ~title:"a" "color";
                 legend ~title:"b" "color";
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
  test "a tensor that cannot be read raises naming its mark" (fun () ->
      let t = dot ~x:(num (traced ())) ~y:(const 0.5) () |> name "t" in
      fails_naming [ "resolve: t: "; "traced" ] (fun () ->
          resolve (layer [ dot1 [| 1. |] [| 1. |]; t ])))

(* Resolved.scale *)

let resolved_scale =
  let filled x = dot1 x [| 0.; 1. |] ~fill:(num ~scale:exact (f64 x)) in
  let two = grid [ [ filled [| 0.; 1. |]; filled [| 5.; 6. |] ] ] in
  let apart = share [ ("color", `Independent) ] in
  let r = resolve (layer [ dot1 [| 1. |] [| 2. |] |> name "a" ]) in
  let raises =
    [
      ( "an unnamed scale",
        [ "unnamed" ],
        fun () -> ignore (Resolved.scale r (Scale.linear ())) );
      ( "an id no node has",
        [ "b" ],
        fun () -> ignore (quant ~at:(path [ field "b" ]) r "x") );
      ("a name its scope lacks", [ "lr" ], fun () -> ignore (quant r "lr"));
      ("a scale of another kind", [ "x" ], fun () -> ignore (categ r "x"));
      ( "a time scale, which no channel reads",
        [ "x" ],
        fun () -> ignore (Resolved.scale r (Scale.time ~name:"x" ())) );
      ( "a position at a layer repeated over cells",
        [ "no scope of the scale \"x\" holds r" ],
        fun () ->
          ignore
            (quant
               ~at:(path [ field "r" ])
               (resolve (layer [ two; name "r" (filled [| 2. |]) ]))
               "x") );
      ( "a colour at a layer whose children keep it apart",
        [ "no scope of the scale \"color\" holds l" ],
        fun () ->
          let l =
            apart (layer [ filled [| 1. |]; filled [| 5. |] ]) |> name "l"
          in
          ignore
            (quant
               ~at:(path [ field "l" ])
               (resolve (grid [ [ l; filled [| 7. |] ] ]))
               "color") );
      ( "a colour at a facetted mark that keeps it per panel",
        [ "no scope of the scale \"color\" holds m" ],
        fun () ->
          let m =
            apart
              (dot ~x:(const 0.5) ~y:(const 0.5)
                 ~fill:(num (f64 [| 1.; 2. |]))
                 ~fx:(strings [| "a"; "b" |])
                 ())
            |> name "m"
          in
          ignore
            (quant ~at:(path [ field "m" ]) (resolve (layer [ m ])) "color") );
    ]
  in
  group "Resolved.scale"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "raises on" raises
        (fun (_, subs, f) -> fails_naming subs f);
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
      prop "layer lifts its children's wrappers, one at a time"
        (Gen.pair gen_mark gen_mark) (fun (a, b) ->
          let t s = title s and c = coord (Coord.cartesian ~aspect:2. ()) in
          pp_equal (layer [ t "w" a; b ]) (t "w" (layer [ a; b ]));
          pp_equal (layer [ a; c b ]) (c (layer [ a; b ]));
          pp_equal
            (layer [ t "w" (t "v" a); t "w" b ])
            (t "w" (t "v" (layer [ a; b ])));
          invalid (fun () -> resolve (layer [ t "w" (t "v" a); t "v" b ])));
      prop "layered grids broadcast as their shapes do"
        Gen.(
          pair
            (pair (int_range 0 3) (int_range 1 3))
            (pair (int_range 0 3) (int_range 1 3)))
        (fun ((r, c), (r', c')) ->
          let g r c =
            grid (List.init r (fun _ -> List.init c (fun _ -> layer [])))
          in
          (* A grid of no rows has no cells, whatever its columns. *)
          let c = if r = 0 then 0 else c and c' = if r' = 0 then 0 else c' in
          let dim d d' =
            if d = d' || d' = 1 then Some d else if d = 1 then Some d' else None
          in
          match (dim r r', dim c c') with
          | Some rows, Some cols
            when rows * cols = 0 && (r * c > 0 || r' * c' > 0) ->
              cover "a grid drawn nowhere" true;
              invalid (fun () -> resolve (layer [ g r c; g r' c' ]))
          | Some rows, Some cols ->
              cover "a grid repeats" (r <> r' || c <> c');
              cover "no cells" (rows = 0);
              contains
                ~sub:(Format.asprintf "%d × %d" rows cols)
                (printed (resolve (layer [ g r c; g r' c' ])))
          | _ ->
              cover "shapes that do not broadcast" true;
              invalid (fun () -> resolve (layer [ g r c; g r' c' ])));
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
      test "share is idempotent, and changes nothing on one cell" (fun () ->
          let a = dot1 [| 1. |] [| 1. |] and b = dot1 [| 2. |] [| 2. |] in
          let p = share [ ("x", `Shared) ] in
          pp_equal (p (p (grid [ [ a; b ] ]))) (p (grid [ [ a; b ] ]));
          pp_equal (p (grid [ [ a ] ])) (grid [ [ a ] ]));
      test "a nested layer draws what the flat layer draws" (fun () ->
          let a = dot1 [| 1. |] [| 1. |] and b = dot1 [| 2. |] [| 2. |] in
          let r = resolve (layer [ layer [ a; b ]; a ])
          and r' = resolve (layer [ a; b; a ]) in
          equal floats (hull r "x") (hull r' "x");
          not_equal resolved r r');
    ]

(* Scopes *)

(* Law: one scale per name, kind and scope, over generated grids. A leaf [k]
   puts the value [k] on x and on colour, so the domain a leaf reads is the hull
   of the leaves sharing its scope. *)
type tree = Leaf | Grid of sharing option * sharing option * tree list list

let pp_sharing ppf = function
  | None -> ()
  | Some `Shared -> Format.pp_print_string ppf " shared"
  | Some `Independent -> Format.pp_print_string ppf " independent"

let rec pp_tree ppf = function
  | Leaf -> Format.pp_print_string ppf "leaf"
  | Grid (x, c, rows) ->
      let pp_row = Format.pp_print_list ~pp_sep:Format.pp_print_space pp_tree in
      Format.fprintf ppf "@[<hv 1>(grid x%a color%a@ %a)@]" pp_sharing x
        pp_sharing c
        (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf r ->
             Format.fprintf ppf "@[[%a]@]" pp_row r))
        rows

let rec gen_grid depth =
  let open Gen in
  let sharing = option (of_list [ `Shared; `Independent ]) in
  let cell =
    if depth = 0 then constant Leaf
    else frequency [ (1, constant Leaf); (2, gen_grid (depth - 1)) ]
  in
  let* rows = int_range 1 2 in
  let* cols = int_range 1 2 in
  let+ x = sharing
  and+ c = sharing
  and+ cells = list ~size:(constant rows) (list ~size:(constant cols) cell) in
  Grid (x, c, cells)

(* A tree whose root is a grid, so that every leaf is a named cell. *)
let gen_tree = Gen.with_pp pp_tree (Gen.bind (Gen.int_range 0 2) gen_grid)

(* A tree with the id of each node: a leaf [k] is named ["mk"], and a grid is
   the cell of its parent it lies in. *)
type node = { id : Nx.Ptree.Path.t; body : body }

and body =
  | Leaf_k of int
  | Grid_of of sharing option * sharing option * node list list

let annotate t =
  let next = ref 0 in
  let rec go at = function
    | Leaf ->
        let k = !next in
        incr next;
        { id = at (Some ("m" ^ string_of_int k)); body = Leaf_k k }
    | Grid (xs, cs, rows) ->
        let id = at None in
        let cols = match rows with r :: _ -> List.length r | [] -> 0 in
        let child i j name =
          let segs = Nx.Ptree.Path.segments id in
          match name with
          | Some s -> path (segs @ [ field s ])
          | None -> path (segs @ [ index ((i * cols) + j) ])
        in
        let rows =
          List.mapi (fun i -> List.mapi (fun j -> go (child i j))) rows
        in
        { id; body = Grid_of (xs, cs, rows) }
  in
  go (fun _ -> root) t

let rec figure n =
  match n.body with
  | Leaf_k k ->
      let v = f64 [| Float.of_int k |] in
      dot ~x:(num ~scale:exact v) ~y:(const 0.5) ~fill:(num ~scale:exact v) ()
      |> name ("m" ^ string_of_int k)
  | Grid_of (xs, cs, rows) ->
      let pairs =
        List.filter_map
          (fun (n, s) -> Option.map (fun s -> (n, s)) s)
          [ ("x", xs); ("color", cs) ]
      in
      let g = grid (List.map (List.map figure) rows) in
      if pairs = [] then g else share pairs g

(* [scopes n ~x ~c ~cell] is each leaf of [n] with its x and colour scopes,
   named by the node that has one scale: the nearest share of the name above the
   leaf, its grid if [`Shared] and its cell if [`Independent]; else the leaf's
   innermost cell for x and the figure for colour. [x], [c] and [cell] are those
   of the shares and cells above [n]. With [~nested:false] only the shares of
   [n] itself count, as they do for a channel in a cell of [n]. *)
let rec scopes ?(nested = true) ?(apply = true) n ~x ~c ~cell =
  match n.body with
  | Leaf_k k ->
      [ (k, Option.value x ~default:cell, Option.value c ~default:root) ]
  | Grid_of (xs, cs, rows) ->
      List.concat_map
        (fun ch ->
          let scope = function
            | Some `Shared when apply -> Some n.id
            | Some `Independent when apply -> Some ch.id
            | _ -> None
          in
          let x = match scope xs with Some _ as s -> s | None -> x in
          let c = match scope cs with Some _ as s -> s | None -> c in
          scopes ~nested ~apply:nested ch ~x ~c ~cell:ch.id)
        (List.concat rows)

let rec grids n =
  match n.body with
  | Leaf_k _ -> []
  | Grid_of (_, _, rows) -> n :: List.concat_map grids (List.concat rows)

let rec leaf_ids n =
  match n.body with
  | Leaf_k k -> [ (k, n.id) ]
  | Grid_of (_, _, rows) -> List.concat_map leaf_ids (List.concat rows)

(* [context n t] is the shares and cell above the node [n] of [t]. *)
let context target t =
  let rec go n ~x ~c ~cell =
    if Nx.Ptree.Path.equal n.id target.id then Some (x, c, cell)
    else
      match n.body with
      | Leaf_k _ -> None
      | Grid_of (xs, cs, rows) ->
          List.find_map
            (fun ch ->
              let scope = function
                | Some `Shared -> Some n.id
                | Some `Independent -> Some ch.id
                | None -> None
              in
              let x = match scope xs with Some _ as s -> s | None -> x in
              let c = match scope cs with Some _ as s -> s | None -> c in
              go ch ~x ~c ~cell:ch.id)
            (List.concat rows)
  in
  Option.get (go t ~x:None ~c:None ~cell:root)

(* [independent_over_shared g] is [true] iff [g] keeps x per cell and a cell of
   [g] is a grid that shares one x: [g] then holds no x scale, though its leaves
   read one (see the xfail below). *)
let independent_over_shared g =
  match g.body with
  | Grid_of (Some `Independent, _, rows) ->
      List.exists
        (fun ch ->
          match ch.body with Grid_of (Some `Shared, _, _) -> true | _ -> false)
        (List.concat rows)
  | _ -> false

let scopes_law t =
  let t = annotate t in
  let r = resolve (figure t) in
  let leaves = scopes t ~x:None ~c:None ~cell:root in
  let ids = leaf_ids t in
  let names = [ ("x", fun (_, x, _) -> x); ("color", fun (_, _, c) -> c) ] in
  (* [hull_of pick s] is the hull of the leaves whose scope [pick] is [s]. *)
  let hull_of pick s =
    match List.filter (fun l -> Nx.Ptree.Path.equal (pick l) s) leaves with
    | [] -> None
    | ls ->
        let ks = List.map (fun (k, _, _) -> Float.of_int k) ls in
        Some
          ( List.fold_left Float.min infinity ks,
            List.fold_left Float.max neg_infinity ks )
  in
  let distinct l =
    List.sort_uniq
      (fun a b ->
        String.compare (Nx.Ptree.Path.to_string a) (Nx.Ptree.Path.to_string b))
      l
  in
  cover "leaves sharing an x"
    (List.length (distinct (List.map (fun (_, x, _) -> x) leaves))
    < List.length leaves);
  cover "colours kept apart"
    (List.length (distinct (List.map (fun (_, _, c) -> c) leaves)) > 1);
  List.iter
    (fun ((k, _, _) as l) ->
      let at = List.assoc k ids in
      List.iter
        (fun (n, pick) ->
          let msg = Format.asprintf "%s at %a" n Nx.Ptree.Path.pp at in
          equal ~msg (option floats)
            (hull_of pick (pick l))
            (Some (hull ~at r n)))
        names)
    leaves;
  List.iter
    (fun g ->
      let x, c, cell = context g t in
      let view = scopes ~nested:false g ~x ~c ~cell in
      List.iter
        (fun (n, pick) ->
          let msg = Format.asprintf "%s at %a" n Nx.Ptree.Path.pp g.id in
          if n = "x" && independent_over_shared g then ()
          else
            match distinct (List.map pick view) with
            | [ s ] -> (
                cover "a grid holding a scope" true;
                match hull_of pick s with
                | Some h -> equal ~msg floats h (hull ~at:g.id r n)
                | None -> invalid (fun () -> hull ~at:g.id r n))
            | _ ->
                cover "a grid holding no scope" true;
                fails_naming [ "no scope" ] (fun () -> hull ~at:g.id r n))
        names)
    (grids t)

let scopes =
  let filled x = dot1 x [| 0.; 1. |] ~fill:(num ~scale:exact (f64 x)) in
  let two = grid [ [ filled [| 0.; 1. |]; filled [| 5.; 6. |] ] ] in
  let own = share [ ("color", `Independent) ] (filled [| 1.; 2. |]) in
  let beside_own = layer [ own; filled [| 5.; 6. |] ] in
  let zoomed at =
    let color = Scale.linear ~name:"color" () in
    View.set (View.zoom ?at color) (Some (8., 9.)) View.empty
  in
  (* A figure, a node, a name and the hull its scale there fits. *)
  let hulls =
    [
      ( "a layer's children share every scale",
        layer [ dot1 [| 0. |] [| 0. |]; dot1 [| 4. |] [| 0. |] ],
        None,
        root,
        "x",
        (0., 4.) );
      ( "a node a layer repeats over cells names the colour",
        layer [ two; name "r" (filled [| 2.; 3. |]) ],
        None,
        path [ field "r" ],
        "color",
        (0., 6.) );
      ( "a cell of a layered grid reads its broadcast cell",
        layer [ two; name "r" (filled [| 2.; 3. |]) ],
        None,
        path [ field "cell"; index 1 ],
        "x",
        (2., 6.) );
      ( "a mark whose facet constant names no panel still fits x",
        layer
          [
            dot1 [| 0.; 1. |] [| 0.; 0. |] ~fx:(strings [| "a"; "a" |]);
            dot1 [| 100. |] [| 0. |] ~fx:(const "z");
          ],
        None,
        root,
        "x",
        (0., 100.) );
      ( "a mark independent per panel holds its own scale",
        beside_own,
        None,
        path [ index 0 ],
        "color",
        (1., 2.) );
      ( "beside a mark that keeps its own, the figure holds the rest",
        beside_own,
        None,
        root,
        "color",
        (5., 6.) );
      ( "a zoom at a mark independent per panel zooms its own scale",
        beside_own,
        Some (zoomed (Some (path [ index 0 ]))),
        path [ index 0 ],
        "color",
        (8., 9.) );
      ( "a zoom at the root leaves the scale a mark keeps",
        beside_own,
        Some (zoomed None),
        path [ index 0 ],
        "color",
        (1., 2.) );
      ( "a facet panel holds the scale its mark has there",
        share
          [ ("color", `Independent) ]
          (dot ~x:(const 0.5) ~y:(const 0.5)
             ~fill:(num ~scale:exact (f64 [| 1.; 2. |]))
             ~fx:(strings [| "a"; "b" |])
             ()),
        None,
        path [ field "panel"; field "b" ],
        "color",
        (2., 2.) );
    ]
  in
  group "scopes"
    [
      prop "one scale per name, kind and scope" gen_tree scopes_law;
      xfail
        ~reason:
          "a grid keeping x per cell keys its cell apart from the x the cell \
           shares"
        (test "a grid keeping x per cell holds the x its cell shares" (fun () ->
             let leaf = dot1 [| 1. |] [| 0. |] in
             let f =
               share
                 [ ("x", `Independent) ]
                 (grid [ [ share [ ("x", `Shared) ] (grid [ [ leaf ] ]) ] ])
             in
             equal floats (1., 1.) (hull (resolve f) "x")));
      cases
        ~name:(fun (n, _, _, _, _, _) -> n)
        "fits" hulls
        (fun (_, f, view, at, n, expected) ->
          equal floats expected (hull ~at (resolve ?view f) n));
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
          contains ~sub:"panel.a" (printed r));
      test "layered grids broadcast into cells of their own" (fun () ->
          (* A 1 × 3 grid over a 2 × 1 grid: cell [k] holds column [k mod 3] of
             the first and row [k / 3] of the second. *)
          let row ?(f = Fun.id) () =
            grid [ List.init 3 (fun c -> dot1 [| Float.of_int c |] [| 0. |]) ]
            |> f
          and col =
            grid
              (List.init 2 (fun r ->
                   [ dot1 [| Float.of_int (10 + r) |] [| 0. |] ]))
          in
          let r = resolve (layer [ row (); col ]) in
          for k = 0 to 5 do
            equal ~msg:(string_of_int k) floats
              (Float.of_int (k mod 3), Float.of_int (10 + (k / 3)))
              (hull ~at:(path [ field "cell"; index k ]) r "x")
          done;
          let independent = share [ ("x", `Independent) ] in
          let r = resolve (layer [ row ~f:independent (); col ]) in
          equal floats (1., 11.)
            (hull ~at:(path [ field "cell"; index 4 ]) r "x"));
    ]

(* Merging *)

let draw_with ~imply r c =
  Mark.v ~name:"m" [ Mark.bind ~imply r c ] draw_nothing

let merging =
  let x = f64 [| 2.; 3. |] and lo = f64 [| 1.; 1.5 |] in
  let ex = num ~scale:exact x in
  (* A figure, the name of a quantitative scale, and the hull it fits. *)
  let hulls =
    [
      ("a bar's length implies zero", rect ~x:(dim 0) ~y:ex (), "y", (0., 3.));
      ( "an explicit property beats an implied one",
        rect ~x:(dim 0)
          ~y:(num ~scale:(Scale.linear ~zero:false ~nice:false ()) x)
          (),
        "y",
        (2., 3.) );
      ("a stem implies zero", rule ~x:(dim 0) ~y:ex (), "y", (0., 3.));
      ("an area alone implies zero", area ~y:ex (), "y", (0., 3.));
      ( "an area with a baseline fits both curves",
        area ~y:ex ~y2:(num lo) (),
        "y",
        (1., 3.) );
      ( "an error bar fits both ends",
        rule ~x:(dim 0) ~y:(num ~scale:exact lo) ~y2:(num x) (),
        "y",
        (1., 3.) );
      ( "the size role implies zero",
        dot ~x:(const 0.) ~y:(const 0.) ~size:ex (),
        "size",
        (0., 3.) );
      ( "a contour's x is its grid's hull",
        contour
          ~x:(num (f64 [| -1.5; 0.; 1.5 |]))
          ~y:(num (Nx.create Nx.float64 [| 2; 1 |] [| -0.3; 0.7 |]))
          ~fill:(num (Nx.zeros Nx.float64 [| 2; 3 |]))
          (),
        "x",
        (-1.5, 1.5) );
      ( "a contour's y is its grid's hull",
        contour
          ~x:(num (f64 [| -1.5; 0.; 1.5 |]))
          ~y:(num (Nx.create Nx.float64 [| 2; 1 |] [| -0.3; 0.7 |]))
          ~fill:(num (Nx.zeros Nx.float64 [| 2; 3 |]))
          (),
        "y",
        (-0.3, 0.7) );
      ( "an image's x is its columns",
        image (Nx.zeros Nx.float32 [| 2; 3 |]),
        "x",
        (0., 3.) );
      ( "an image's y is its rows",
        image (Nx.zeros Nx.float32 [| 2; 3 |]),
        "y",
        (0., 2.) );
      ( "colour scales of two kinds are two scales",
        layer
          [
            dot ~x:(num x) ~y:(num x) ~fill:(num ~scale:exact x) ();
            dot ~x:(num x) ~y:(num x)
              ~fill:(cat ~labels:[| "a" |] (i32 [| 0; 0 |]))
              ();
          ],
        "color",
        (2., 3.) );
    ]
  in
  (* A figure, the name of a band scale, and its bandwidth. *)
  let bands =
    [
      ( "a mark implies padding on a band scale",
        draw_with
          ~imply:(Scale.band ~padding:0.5 ())
          Role.x
          (strings [| "a"; "b" |]),
        "x",
        0.5 /. 2.5 );
      ( "a bar implies padding 0.2",
        rect ~x:(strings [| "a"; "b" |]) ~y:(num x) (),
        "x",
        0.8 /. 2.2 );
      ( "a heatmap's x has no padding",
        rect ~x:(dim 1) ~y:(dim 0)
          ~fill:(num (Nx.zeros Nx.float64 [| 2; 3 |]))
          (),
        "x",
        1. /. 3. );
      ( "a heatmap's y has no padding",
        rect ~x:(dim 1) ~y:(dim 0)
          ~fill:(num (Nx.zeros Nx.float64 [| 2; 3 |]))
          (),
        "y",
        0.5 );
      ( "an explicit padding beats a bar's",
        rect
          ~x:(strings ~scale:(Scale.band ~padding:0. ()) [| "a"; "b" |])
          ~y:(num x) (),
        "x",
        0.5 );
    ]
  in
  let fitted = Testable.make ~pp:Scale.pp ~equal:Scale.equal in
  let nice_and_clamped =
    Scale.fit
      (Some (Scale.Floats (2., 3.)))
      (Scale.linear ~nice:false ~clamp:true ())
  in
  group "merging"
    [
      cases
        ~name:(fun (n, _, _, _) -> n)
        "fits" hulls
        (fun (_, f, n, h) -> equal floats h (hull (resolve f) n));
      cases
        ~name:(fun (n, _, _, _) -> n)
        "bands" bands
        (fun (_, f, n, w) ->
          equal (float 1e-12) w (Scale.bandwidth (categ (resolve f) n)));
      test "a band scale is reversed iff y reads it" (fun () ->
          let ab = cat ~labels:[| "a"; "b" |] (i32 [| 0; 1 |]) in
          let at_a f n = Scale.normalize (categ (resolve f) n) "a" in
          greater float_exact ~than:0.5 (at_a (dot ~x:(const 0.5) ~y:ab ()) "y");
          less float_exact ~than:0.5 (at_a (dot ~x:ab ~y:(const 0.5) ()) "x"));
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
          equal fitted nice_and_clamped (quant r "x"));
      test "implied specifications give neither name nor transform" (fun () ->
          let m imply = draw_with ~imply Role.fill (num x) in
          let r =
            resolve
              (layer
                 [
                   m (Scale.log ~name:"a" ~nice:false ());
                   m (Scale.linear ~name:"b" ~clamp:true ());
                 ])
          in
          equal fitted nice_and_clamped (quant r "color"));
    ]

(* Categories *)

let categories_ =
  let at0 = f64 [| 0. |] in
  let filled ?(n = 1) fill =
    dot ~x:(num (f64 (Array.make n 0.))) ~y:(num at0) ~fill ()
  in
  let one l = filled (strings [| l |]) in
  let symbols ?(x = [| 1.; 2.; 3.; 4. |]) symbol =
    dot ~x:(num ~scale:exact (f64 x)) ~y:(num at0) ~symbol ()
  in
  let labelled l = List.map (fun l -> (-1, l)) l in
  (* A figure and the categories of its colour scale: indices and their texts,
     or labels at [-1]. *)
  let domains =
    [
      ( "labels are kept whole",
        filled (cat ~labels:[| "a"; "b"; "c" |] (i32 [| 0 |])),
        labelled [ "a"; "b"; "c" ] );
      ( "labels unite in written order, before broadcasting",
        layer [ grid [ [ one "a"; one "b" ] ]; one "c" ],
        labelled [ "a"; "b"; "c" ] );
      ( "strings contribute in order of first appearance",
        filled ~n:3 (strings [| "q"; "p"; "q" |]),
        labelled [ "q"; "p" ] );
      ( "strings of dropped rows are categories",
        dot
          ~x:(num (f64 [| 1.; Float.nan |]))
          ~y:(num at0)
          ~fill:(strings [| "kept"; "dropped" |])
          (),
        labelled [ "kept"; "dropped" ] );
      ( "indexed categories unite in increasing order",
        layer [ filled ~n:2 (cat (i32 [| 5; 1 |])); filled ~n:3 (dim 0) ],
        [ (0, "0"); (1, "1"); (2, "2"); (5, "5") ] );
      ( "codes of dropped rows are no categories",
        dot
          ~x:(num (f64 [| 0.; Float.nan; 0. |]))
          ~y:(num at0)
          ~fill:(cat ~valid:(mask [| true; true; false |]) (i32 [| 3; 4; 7 |]))
          (),
        [ (3, "3") ] );
      ( "a dim without labels beside a labelled one takes its texts",
        layer
          [ filled ~n:2 (dim ~labels:[| "a"; "b" |] 0); filled ~n:3 (dim 0) ],
        [ (0, "a"); (1, "b"); (2, "2") ] );
      ( "a dim contributes every index of its axis",
        dot
          ~x:(num (f64 [| Float.nan; Float.nan |]))
          ~y:(num at0) ~fill:(dim 0) (),
        [ (0, "0"); (1, "1") ] );
    ]
  in
  (* A figure whose symbol scale has an explicit domain, and the x hull of the
     rows it keeps. *)
  let kept =
    let labels = Scale.band ~domain:(Scale.Labels [| "b" |]) () in
    [
      ( "strings",
        symbols ~x:[| 1.; 2. |] (strings ~scale:labels [| "a"; "b" |]),
        (2., 2.) );
      ( "labelled codes",
        symbols ~x:[| 1.; 2.; 3. |]
          (cat ~scale:labels ~labels:[| "a"; "b" |] (i32 [| 0; 1; 0 |])),
        (2., 2.) );
      ( "indexed codes",
        symbols
          (cat
             ~scale:
               (Scale.band
                  ~domain:(Scale.Indices [| (2, "two"); (5, "five") |])
                  ())
             (i32 [| 2; 3; 5; 9 |])),
        (1., 3.) );
      ( "the indices of a dim",
        symbols ~x:[| 1.; 2.; 3. |]
          (dim
             ~scale:(Scale.band ~domain:(Scale.Indices [| (1, "one") |]) ())
             0),
        (2., 2.) );
    ]
  in
  group "categories"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "fit" domains
        (fun (_, f, expected) ->
          equal
            (list (pair int string))
            expected
            (categories (categ (resolve f) "color")));
      test "floats fit their finite values" (fun () ->
          let x =
            Hugin.floats ~scale:exact
              [| 2.; Float.nan; Float.infinity; -1.; Float.neg_infinity |]
          in
          equal floats (-1., 2.) (hull (resolve (dot ~x ~y:(num at0) ())) "x"));
      cases
        ~name:(fun (n, _, _) -> n)
        "an explicit domain drops the rows outside it: " kept
        (fun (_, f, expected) -> equal floats expected (hull (resolve f) "x"));
    ]

(* Facet panels *)

(* Each panel of a facetted mark whose y is independent per panel fits its y to
   the rows the panel draws. *)
let facet_panels =
  let ys = [| 1.; 5.; 3.; 9. |] in
  let per ?(shape = [| 4 |]) ?fy fx =
    let y = num ~scale:exact (Nx.create Nx.float64 shape ys) in
    share [ ("y", `Independent) ] (dot ~x:(const 0.5) ~y ?fy ~fx ())
  in
  let panels =
    [
      ( "labelled codes",
        per (cat ~labels:[| "p"; "q" |] (i32 [| 0; 1; 0; 1 |])),
        [ ("p", (1., 3.)); ("q", (5., 9.)) ] );
      ( "indexed codes",
        per (cat (i32 [| 7; 2; 7; 2 |])),
        [ ("2", (5., 9.)); ("7", (1., 3.)) ] );
      ( "strings",
        per (strings [| "p"; "q"; "p"; "q" |]),
        [ ("p", (1., 3.)); ("q", (5., 9.)) ] );
      ( "a dim of the first axis",
        per ~shape:[| 2; 2 |] (dim 0),
        [ ("0", (1., 5.)); ("1", (3., 9.)) ] );
      ( "a dim of the last axis",
        per ~shape:[| 2; 2 |] (dim (-1)),
        [ ("0", (1., 3.)); ("1", (5., 9.)) ] );
      ( "fy and fx, named in that order",
        per ~shape:[| 2; 2 |] ~fy:(dim 0) (dim 1),
        [
          ("0.0", (1., 1.));
          ("0.1", (5., 5.));
          ("1.0", (3., 3.));
          ("1.1", (9., 9.));
        ] );
    ]
  in
  cases
    ~name:(fun (n, _, _) -> n)
    "facet channels select their panels: " panels
    (fun (_, f, expected) ->
      let r = resolve f in
      List.iter
        (fun (c, h) ->
          let cats = List.map field (String.split_on_char '.' c) in
          equal ~msg:c floats h (hull ~at:(path (field "panel" :: cats)) r "y"))
        expected)

(* Law 15: a row is in a hull iff none of its values other than its colour is
   missing. *)
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
          greater float_exact ~than:0.5 (Scale.normalize (categ r' "x") "a"));
    ]

(* Views and warnings *)

let lr = Scale.linear ~name:"lr" ()
let lg = Scale.log ~name:"lg" ()
let on_lr = dot ~x:(num ~scale:lr (f64 [| 1.; 2. |])) ~y:(const 0.5) ()
let zooms ?at s z view = View.set (View.zoom ?at s) (Some z) view
let x_scale = Scale.linear ~name:"x" ()
let cells = grid [ [ dot1 [| 0. |] [| 0. |]; dot1 [| 5. |] [| 0. |] ] ]

let bad_log =
  dot ~x:(num ~scale:exact_log (f64 [| -1.; 2. |])) ~y:(const 0.5) ()

let nowhere = dot ~x:(const 0.5) ~y:(const 0.5) ~fx:(const "nowhere") ()
let unread = View.set (View.number "k" ~init:0.) 1. View.empty

(* A figure, a view, the hulls of scales at nodes, and the warnings. *)
(* A figure, a view, the hulls of scales at nodes, and the ids of the warnings,
   whose messages only the view's warning cases state. *)
let zoom_cases =
  [
    ( "a zoom sets the domain",
      on_lr,
      zooms lr (5., 6.) View.empty,
      [ (root, "lr", (5., 6.)) ],
      [] );
    ( "a zoom of no scale is ignored",
      on_lr,
      zooms (Scale.linear ~name:"gone" ()) (5., 6.) View.empty,
      [ (root, "lr", (1., 2.)) ],
      [ root ] );
    ( "a zoom of a time scale, which no channel reads, is ignored",
      on_lr,
      View.set
        (View.zoom (Scale.time ~name:"lr" ()))
        (Some Hugin_kit.Time.(epoch, epoch))
        View.empty,
      [ (root, "lr", (1., 2.)) ],
      [ root ] );
    ( "zooms of one scale at two nodes are ignored",
      layer [ on_lr |> name "a" ],
      View.empty
      |> zooms lr (5., 6.)
      |> zooms ~at:(path [ field "a" ]) lr (7., 8.),
      [ (root, "lr", (1., 2.)) ],
      [ root; path [ field "a" ] ] );
    ( "a zoom at a node in several position scopes is ignored",
      layer [ cells; name "r" (dot1 [| 2. |] [| 0. |]) ],
      zooms ~at:(path [ field "r" ]) x_scale (7., 8.) View.empty,
      [
        (path [ field "cell"; index 0 ], "x", (0., 2.));
        (path [ field "cell"; index 1 ], "x", (2., 5.));
      ],
      [ path [ field "r" ] ] );
    ( "a zoom at a broadcast cell zooms that cell",
      layer [ cells; dot1 [| 2. |] [| 0. |] ],
      zooms ~at:(path [ field "cell"; index 1 ]) x_scale (7., 8.) View.empty,
      [
        (path [ field "cell"; index 0 ], "x", (0., 2.));
        (path [ field "cell"; index 1 ], "x", (7., 8.));
      ],
      [] );
    ( "a zoom the scale cannot take is ignored",
      dot ~x:(num ~scale:lg (f64 [| 1.; 10. |])) ~y:(const 0.5) (),
      zooms lg (-1., 1.) View.empty,
      [ (root, "lg", (1., 10.)) ],
      [ root ] );
    ( "a view value no key reads is warned about",
      on_lr,
      unread,
      [ (root, "lr", (1., 2.)) ],
      [ root ] );
  ]

(* A figure and its warnings, in the order of the figure, then the view's. *)
let warning_cases =
  [
    ( "warnings follow the figure, then the view's",
      layer
        [
          nowhere;
          bad_log;
          name "z"
            (dot ~x:(const 0.5) ~y:(const 0.5)
               ~fill:(num ~scale:lg (f64 [| 1.; 10. |]))
               ());
          nowhere;
        ],
      unread |> zooms ~at:(path [ field "z" ]) lg (-1., 1.),
      [
        (path [ index 0 ], "the facet constant \"nowhere\" of fx names no panel");
        (path [ index 1 ], "x: 1 finite value is missing for its scale");
        ( path [ field "z" ],
          "the zoom of the scale \"lg\" sets a domain it cannot take" );
        (path [ index 3 ], "the facet constant \"nowhere\" of fx names no panel");
        (root, "the view sets \"k\", which no key of its sort reads");
      ] );
    ( "a mark repeated over the cells of a grid is warned about once",
      layer [ grid [ [ layer []; layer [] ] ]; bad_log ],
      View.empty,
      [ (path [ index 1 ], "x: 1 finite value is missing for its scale") ] );
    ( "a uint64 code beyond int is missing",
      dot ~x:(const 0.5) ~y:(const 0.5)
        ~fill:(cat (Nx.create Nx.uint64 [| 2 |] [| -1L; 1L |]))
        (),
      View.empty,
      [ (root, "fill: 1 code is beyond the range of int") ] );
    ( "a masked code outside the labels is not warned about",
      dot ~x:(const 0.5) ~y:(const 0.5)
        ~fill:
          (cat ~labels:[| "a" |]
             ~valid:(mask [| true; false; true |])
             (i32 [| 3; 5; 0 |]))
        (),
      View.empty,
      [ (root, "fill: 1 code is outside its 1 labels") ] );
    ( "data problems are warned about under the mark's id",
      layer
        [
          dot
            ~x:(num ~scale:exact_log (f64 [| 1.; 0.; -1. |]))
            ~y:(const 0.5) ();
          name "c"
            (dot ~x:(const 0.5) ~y:(const 0.5)
               ~fill:(cat ~labels:[| "a" |] (i32 [| 0; 3 |]))
               ());
        ],
      View.empty,
      [
        (path [ index 0 ], "x: 2 finite values are missing for its scale");
        (path [ field "c" ], "fill: 1 code is outside its 1 labels");
      ] );
  ]

let views =
  group "views"
    [
      cases
        ~name:(fun (n, _, _, _, _) -> n)
        "zooms" zoom_cases
        (fun (_, f, view, hulls, ws) ->
          let r = resolve ~view f in
          List.iter
            (fun (at, n, h) -> equal ~msg:n floats h (hull ~at r n))
            hulls;
          equal (list id) ws (List.map fst (Resolved.warnings r)));
      cases
        ~name:(fun (n, _, _, _) -> n)
        "warnings" warning_cases
        (fun (_, f, view, ws) ->
          equal (list warning) ws (Resolved.warnings (resolve ~view f)));
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
    ]

(* Equality and reuse *)

let reuse =
  let a = dot1 [| 1. |] [| 0. |] ~fill:(num (f64 [| 1. |])) in
  let k = View.number "k" ~init:1. in
  let on_log x = dot ~x:(num ~scale:exact_log (f64 x)) ~y:(const 0.5) () in
  let beside y =
    layer
      [
        dot ~x:(num (f64 [| 1.; 2. |])) ~y:(num (f64 [| -1.; 2. |])) ();
        dot ~x:(const 0.5) ~y ();
      ]
  in
  (* A figure resolved first, and a figure and view resolved with it as
     [prev]. *)
  let again =
    [
      ( "after a scale beside a mark becomes log",
        fun () ->
          ( beside (num (f64 [| 3. |])),
            beside (num ~scale:(Scale.log ()) (f64 [| 3. |])),
            View.empty ) );
      ( "after the view changes",
        fun () ->
          let f = bind k (fun v -> dot1 [| v |] [| 2. |]) in
          (f, f, View.set k 3. View.empty) );
      ( "after an id changes",
        fun () ->
          ( layer [ on_log [| 0.; 1. |] ],
            layer [ layer []; on_log [| 0.; 1. |] ],
            View.empty ) );
    ]
  in
  group "equality and reuse"
    [
      test "a figure resolved twice is equal to itself" (fun () ->
          equal resolved (resolve a) (resolve a));
      test "figures fitting equal scales differ" (fun () ->
          let a' = dot1 [| 1. |] [| 0. |] ~fill:(num (f64 [| 1. |])) in
          let r = resolve a and r' = resolve a' in
          equal bool true
            (Scale.equal (quant r "x") (quant r' "x")
            && Scale.equal (quant r "color") (quant r' "color"));
          not_equal resolved r r');
      test "views that change nothing differ" (fun () ->
          let g = bind k (fun _ -> a) in
          let view = View.set k 1. View.empty in
          equal (list warning) [] (Resolved.warnings (resolve ~view g));
          not_equal resolved (resolve g) (resolve ~view g));
      test "a bind building new tensors resolves equal" (fun () ->
          let f = bind k (fun v -> dot1 [| v |] [| v |]) in
          equal resolved (resolve f) (resolve f));
      prop "resolving again is equal" (Gen.pair gen_mark gen_mark)
        (fun (a, b) ->
          let f = layer [ a; grid [ [ b; a ] ] ] in
          equal resolved (resolve f) (resolve ~prev:(resolve f) f));
      prop "a reused resolution equals a fresh one after a tensor changes"
        Gen.(pair (gen_values 3) (gen_values 3))
        (fun (xs, xs') ->
          let f x = layer [ dot ~x:(num (f64 x)) ~y:(num (f64 x)) (); a ] in
          let f' = f xs' in
          equal resolved (resolve f') (resolve ~prev:(resolve (f xs)) f'));
      cases ~name:fst "a reused resolution equals a fresh one" again
        (fun (_, f) ->
          let first, f, view = f () in
          equal resolved (resolve ~view f)
            (resolve ~prev:(resolve first) ~view f));
    ]

(* Baselines *)

let baselines =
  test "the benchmark figures resolve as their baseline" (fun () ->
      let open Hugin_test_figures in
      let one (name, f, _, _) =
        Format.asprintf "%s@\n%a@\n" name Resolved.pp (resolve (f ()))
      in
      expect_file
        (String.concat "\n" (List.map one Figures.goldens))
        "packages/hugin/test/golden/resolved.expected")

let () =
  exit
    (run "hugin resolve"
       [
         errors;
         reads;
         resolved_scale;
         composition;
         scopes;
         ids;
         merging;
         categories_;
         facet_panels;
         law15;
         export;
         views;
         reuse;
         baselines;
       ])
