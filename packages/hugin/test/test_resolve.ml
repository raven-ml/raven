(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin
open Windtrap

(* Data *)

let f64 a = Nx.create Nx.float64 [| Array.length a |] a
let i32 a = Nx.create Nx.int32 [| Array.length a |] (Array.map Int32.of_int a)
let i64 a = Nx.create Nx.int64 [| Array.length a |] a
let mask a = Nx.create Nx.bool [| Array.length a |] a
let path segs = Nx.Ptree.Path.v segs
let field s = Nx.Ptree.Path.Field s
let index i = Nx.Ptree.Path.Index i
let root = Nx.Ptree.Path.root
let cell k = path [ field "cell"; index k ]

(* Scales keep their hull when they do not round it. *)
let exact = Scale.linear ~nice:false ()
let exact_log = Scale.log ~nice:false ()
let exact_time = Scale.time ~nice:false ()
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

(* Reading resolved figures *)

let ends s =
  let (Scale.Floats (a, b)) = Scale.domain s in
  (a, b)

let instants s = match Scale.domain s with Scale.Instants (a, b) -> (a, b)
let floats = pair float_exact float_exact
let instant = Testable.make ~pp:Time.pp ~equal:Time.equal
let quant ?at r name = Resolved.scale ?at r (Scale.linear ~name ())
let categ ?at r name = Resolved.scale ?at r (Scale.band ~name ())
let hull ?at r name = ends (quant ?at r name)
let time_hull r = instants (Resolved.scale r (Scale.time ~name:"x" ()))
let fitted = Testable.make ~pp:Scale.pp ~equal:Scale.equal
let resolved = Testable.make ~pp:Resolved.pp ~equal:Resolved.equal
let id = Testable.make ~pp:Nx.Ptree.Path.pp ~equal:Nx.Ptree.Path.equal

(* The domain of a categorical scale, its labels or its indices and their
   texts. *)
let categories s =
  match Scale.domain s with
  | Scale.Categories (Scale.Labels l) ->
      List.map (fun l -> (-1, l)) (Array.to_list l)
  | Scale.Categories (Scale.Indices ix) -> Array.to_list ix

let invalid f = raises_match (Exn.invalid_arg ?sub:None) f

(* [message f] is the message of the [Invalid_argument] that [f ()] raises. *)
let message f =
  match f () with
  | exception Invalid_argument m -> m
  | _ -> fail "returned without raising Invalid_argument"

(* [messages ws] is the messages of the warnings [ws], one a line. *)
let messages ws = String.concat "\n" (List.map snd ws)

(* Drawing resolved figures *)

(* [drawn f] is the page [f] draws at one size and density, or the message of
   the [Invalid_argument] that laying it out or drawing it raises. *)
let drawn f =
  let r = resolve f in
  match draw ~density:1. (layout (Size.figure 300. 200.) r) with
  | d -> Ok (Drawing.renderable d)
  | exception Invalid_argument m -> Error m

(* [painted p] is [p] without its tags, which paint nothing. *)
let rec painted (p : Picture.t) =
  match p with
  | Tag { picture; _ } -> painted picture
  | Group ps -> Picture.group (List.map painted ps)
  | Clip { rule; path; picture } -> Picture.clip ~rule path (painted picture)
  | Transform { m; picture } -> Picture.transform m (painted picture)
  | Opacity { opacity; picture } -> Picture.opacity opacity (painted picture)
  | Empty | Fill _ | Stroke _ | Glyphs _ | Image _ | Stamp _ -> p

let page =
  result (Testable.make ~pp:Renderable.pp ~equal:Renderable.equal) string

let paint = result (Testable.make ~pp:Picture.pp ~equal:Picture.equal) string

let gen_mark =
  Gen.map
    (fun (xs, k) ->
      let x = f64 xs in
      match k with
      | 0 -> dot ~x:(num x) ~y:(num x) ()
      | 1 -> line ~y:(num x) ~stroke:(num x) ()
      | _ -> rect ~y:(num x) ())
    (Gen.pair (gen_values 3) (Gen.int_range 0 2))

(* Missing values *)

(* A row is in a hull iff none of its values other than its colour is missing.
   Without a kept row, a scale has its default domain: [[0;1]] for
   [Scale.linear], [[1;10]] for [Scale.log]. *)
let dropped_law (x, y, size, valid, codes) =
  let n = Array.length x in
  let finite v = Float.is_finite v in
  let kept i =
    finite x.(i) && valid.(i) && finite y.(i) && y.(i) > 0. && finite size.(i)
  in
  let keeps = List.filter kept (List.init n Fun.id) in
  cover "a dropped row" (List.length keeps < n);
  cover "no row kept" (keeps = []);
  cover "a kept row with a missing colour"
    (List.exists (fun i -> codes.(i) < 0 || codes.(i) > 1) keeps);
  let expect v default =
    match keeps with
    | [] -> default
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
  equal ~msg:"x" floats (expect x (0., 1.)) (hull r "x");
  equal ~msg:"y" floats (expect y (1., 10.)) (hull r "y")

let gen_rows =
  Gen.(
    let* n = int_range 0 6 in
    let+ x = gen_values n
    and+ y = gen_values n
    and+ size = gen_values n
    and+ valid = array ~size:(constant n) bool
    and+ codes = array ~size:(constant n) (int_range (-1) 2) in
    (x, y, size, valid, codes))

let missing =
  group "missing values"
    [ prop "a dropped row is in no hull" gen_rows dropped_law ]

(* Ids *)

let ids =
  group "ids"
    [
      test "are stable under insertion before a name" (fun () ->
          let a = dot1 [| 1. |] [| 1. |] ~fill:(num (f64 [| 1. |])) in
          let b = dot1 [| 2. |] [| 2. |] ~fill:(num (f64 [| 7.; 9. |])) in
          let f before =
            share
              [ ("color", `Independent) ]
              (layer (before @ [ a; name "n" b ]))
          in
          let at = path [ field "n" ] in
          let s = quant ~at (resolve (f [])) "color" in
          equal floats (7., 9.) (ends s);
          equal fitted s (quant ~at (resolve (f [ layer [] ])) "color"));
      test "of a facet panel differ from a sibling named like its category"
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
          equal floats (5., 5.)
            (hull ~at:(path [ field "a" ]) (resolve f) "color"));
      test "of a 1 × 3 grid layered with a 2 × 1 grid name six cells" (fun () ->
          (* Cell [k] holds column [k mod 3] of the first and row [k / 3] of the
             second. *)
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
              (hull ~at:(cell k) r "x")
          done;
          invalid (fun () -> hull ~at:(cell 6) r "x"));
      test "of broadcast cells name the scale a grid keeps per cell" (fun () ->
          let row =
            share
              [ ("x", `Independent) ]
              (grid
                 [ List.init 3 (fun c -> dot1 [| Float.of_int c |] [| 0. |]) ])
          and col =
            grid
              (List.init 2 (fun r ->
                   [ dot1 [| Float.of_int (10 + r) |] [| 0. |] ]))
          in
          equal floats (1., 11.)
            (hull ~at:(cell 4) (resolve (layer [ row; col ])) "x"));
    ]

(* time *)

(* Counts of two resolutions, some of them masked, on one scale. *)
let gen_times =
  let open Gen in
  let count = map Int64.of_int (int_range (-1_000_000) 1_000_000) in
  let* n = int_range 1 6 and+ m = int_range 1 6 in
  let+ ms = array ~size:(constant n) count
  and+ valid = array ~size:(constant n) bool
  and+ s = array ~size:(constant m) count in
  (ms, valid, s)

(* A temporal scale that does not round is fitted to the hull of the instants of
   its channels' kept rows, whatever their resolutions. *)
let time_law (ms, valid, s) =
  let kept =
    List.filteri (fun i _ -> valid.(i)) (Array.to_list ms)
    |> List.map (Time.v Ms)
  in
  let all = kept @ List.map (Time.v S) (Array.to_list s) in
  cover "a masked row" (Array.exists not valid);
  let pick keep =
    List.fold_left (fun a t -> if keep (Time.compare t a) then t else a)
  in
  let first = pick (fun c -> c < 0) and last = pick (fun c -> c > 0) in
  let expected = (first (List.hd all) all, last (List.hd all) all) in
  let r =
    resolve
      (layer
         [
           dot
             ~x:(time ~scale:exact_time ~valid:(mask valid) Ms (i64 ms))
             ~y:(const 0.5) ();
           dot ~x:(time S (i64 s)) ~y:(const 0.5) ();
         ])
  in
  equal (pair instant instant) expected (time_hull r)

let time_ =
  group "time"
    [
      prop "fits a scale to its instants" gen_times time_law;
      test "fits a nice domain to calendar boundaries" (fun () ->
          (* A tenth of 24 days and 5 hours is nearest by ratio to 2 days, whose
             boundaries fall on even days since the epoch. *)
          let a = Time.of_date_time ((2026, 3, 3), (10, 0, 0))
          and b = Time.of_date_time ((2026, 3, 27), (15, 0, 0)) in
          let s t = Option.get (Time.to_int64 S t) in
          let r =
            resolve (dot ~x:(time S (i64 [| s a; s b |])) ~y:(const 0.5) ())
          in
          equal (pair instant instant)
            (Time.of_date (2026, 3, 2), Time.of_date (2026, 3, 28))
            (time_hull r));
      test "reaches the ends of int64" (fun () ->
          let ends = [| Int64.max_int; Int64.min_int; 0L |] in
          let r =
            resolve
              (dot ~x:(time ~scale:exact_time Ns (i64 ends)) ~y:(const 0.5) ())
          in
          equal (pair instant instant)
            (Time.v Ns Int64.min_int, Time.v Ns Int64.max_int)
            (time_hull r));
      test "makes an unsigned count beyond int64 missing, with a warning"
        (fun () ->
          let c = Nx.create Nx.uint64 [| 2 |] [| 5L; Int64.min_int |] in
          let r =
            resolve
              (layer
                 [
                   dot ~x:(time ~scale:exact_time S c) ~y:(const 0.5) ()
                   |> name "d";
                 ])
          in
          equal (pair instant instant) (Time.v S 5L, Time.v S 5L) (time_hull r);
          equal (list id)
            [ path [ field "d" ] ]
            (List.map fst (Resolved.warnings r));
          expect (messages (Resolved.warnings r))
          @@ __POS_OF__ {| x: 1 count is beyond the range of int64 |});
    ]

(* floats *)

let floats_ =
  group "floats"
    [
      test "fits its finite values" (fun () ->
          let x =
            Hugin.floats ~scale:exact
              [| 2.; Float.nan; Float.infinity; -1.; Float.neg_infinity |]
          in
          equal floats (-1., 2.)
            (hull (resolve (dot ~x ~y:(num (f64 [| 0. |])) ())) "x"));
    ]

(* Marks *)

let field_x = f64 [| -1.5; 0.; 1.5 |]
let field_y = Nx.create Nx.float64 [| 2; 1 |] [| -0.3; 0.7 |]

let sampled =
  contour ~x:(num field_x) ~y:(num field_y)
    ~fill:(num (Nx.zeros Nx.float64 [| 2; 3 |]))
    ()

let pixels = image (Nx.zeros Nx.float32 [| 2; 3 |])

let image_ =
  group "image"
    [
      test "covers [0;w] on x and [0;h] on y" (fun () ->
          let r = resolve pixels in
          equal floats (0., 3.) (hull r "x");
          equal floats (0., 2.) (hull r "y"));
    ]

let contour_ =
  group "contour"
    [
      test "fits x and y to its grid's hull" (fun () ->
          let r = resolve sampled in
          equal floats (-1.5, 1.5) (hull r "x");
          equal floats (-0.3, 0.7) (hull r "y"));
    ]

(* Names *)

let names =
  group "names"
    [
      test "colour channels of two kinds read two default scales" (fun () ->
          let x = f64 [| 2.; 3. |] in
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
          equal
            (list (pair int string))
            [ (-1, "a") ]
            (categories (categ r "color")));
    ]

(* Scopes *)

(* A grid of leaves and grids, with shares of x and colour. A leaf [k] puts the
   value [k] on x and on colour, so the domain a leaf reads is the hull of the
   leaves sharing its scope. *)
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
          let held =
            match distinct (List.map pick view) with
            | [ s ] -> Some s
            | _ -> None
          in
          cover "a grid holding a scope" (Option.is_some held);
          cover "a grid holding no scope" (Option.is_none held);
          match held with
          | Some s -> (
              match hull_of pick s with
              | Some h -> equal ~msg floats h (hull ~at:g.id r n)
              | None -> invalid (fun () -> hull ~at:g.id r n))
          | None -> invalid (fun () -> hull ~at:g.id r n))
        names)
    (grids t)

let scopes_ =
  let filled x = dot1 x [| 0.; 1. |] ~fill:(num ~scale:exact (f64 x)) in
  let two = grid [ [ filled [| 0.; 1. |]; filled [| 5.; 6. |] ] ] in
  let own = share [ ("color", `Independent) ] (filled [| 1.; 2. |]) in
  let beside_own = layer [ own; filled [| 5.; 6. |] ] in
  let zoomed at =
    let color = Scale.linear ~name:"color" () in
    View.set (View.zoom ?at color) (Some (8., 9.)) View.empty
  in
  (* A figure, a view, a node, a name and the hull its scale there fits. *)
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
        cell 1,
        "x",
        (2., 6.) );
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
      (* The example is a grid keeping x per cell around a grid sharing x: it
         held no x, though its one leaf read one. *)
      prop "one scale per name, kind and scope"
        ~examples:
          [
            Grid
              ( Some `Independent,
                None,
                [ [ Grid (Some `Shared, None, [ [ Leaf ] ]) ] ] );
          ]
        gen_tree scopes_law;
      cases "a node holds the scale of the scope it lies in"
        ~id:(fun (n, _, _, _, _, _) -> n)
        hulls
        (fun (_, f, view, at, n, expected) ->
          equal floats expected (hull ~at (resolve ?view f) n));
      prop "facets share one scale" (gen_values 6) (fun ys ->
          let y = Nx.reshape [| 2; 3 |] (f64 ys) in
          let plain = dot ~x:(const 0.5) ~y:(num ~scale:exact y) () in
          let facetted =
            dot ~x:(const 0.5) ~y:(num ~scale:exact y) ~fx:(dim 0) ()
          in
          equal fitted
            (quant (resolve plain) "y")
            (quant (resolve facetted) "y"));
    ]

(* Merging *)

let draw_with ~imply r c =
  Mark.v ~name:"m" [ Mark.bind ~imply r c ] draw_nothing

(* The width of the bands of a band scale of [n] categories and padding [p], as
   [Scale.bandwidth] states it. *)
let bandwidth n p = (1. -. p) /. (Float.of_int n +. p)

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
    ]
  in
  (* A figure, the name of a band scale, its number of categories and the
     padding it merges. *)
  let bands =
    [
      ( "a mark implies padding on a band scale",
        draw_with
          ~imply:(Scale.band ~padding:0.5 ())
          Role.x
          (strings [| "a"; "b" |]),
        "x",
        2,
        0.5 );
      ( "a bar implies padding 0.2",
        rect ~x:(strings [| "a"; "b" |]) ~y:(num x) (),
        "x",
        2,
        0.2 );
      ( "a heatmap's x has no padding",
        rect ~x:(dim 1) ~y:(dim 0)
          ~fill:(num (Nx.zeros Nx.float64 [| 2; 3 |]))
          (),
        "x",
        3,
        0. );
      ( "a heatmap's y has no padding",
        rect ~x:(dim 1) ~y:(dim 0)
          ~fill:(num (Nx.zeros Nx.float64 [| 2; 3 |]))
          (),
        "y",
        2,
        0. );
      ( "an explicit padding beats a bar's",
        rect
          ~x:(strings ~scale:(Scale.band ~padding:0. ()) [| "a"; "b" |])
          ~y:(num x) (),
        "x",
        2,
        0. );
    ]
  in
  let nice_and_clamped =
    Scale.fit
      (Some (Scale.Floats (2., 3.)))
      (Scale.linear ~nice:false ~clamp:true ())
  in
  group "merging"
    [
      cases "fits a scale to the hull its merged specification gives"
        ~id:(fun (n, _, _, _) -> n)
        hulls
        (fun (_, f, n, h) -> equal floats h (hull (resolve f) n));
      (* The formula takes three roundings, each within half an ulp, and so may
         the library's: the two lie within [4 ε] of each other. *)
      cases "gives a band scale the padding its specifications merge"
        ~id:(fun (n, _, _, _, _) -> n)
        bands
        (fun (_, f, name, n, p) ->
          equal
            (float_rel ~rel:(4. *. epsilon_float) ~abs:0.)
            (bandwidth n p)
            (Scale.bandwidth (categ (resolve f) name)));
      test "reverses a band scale iff y reads it" (fun () ->
          let ab = cat ~labels:[| "a"; "b" |] (i32 [| 0; 1 |]) in
          let at_a f n = Scale.normalize (categ (resolve f) n) "a" in
          greater float_exact ~than:0.5 (at_a (dot ~x:(const 0.5) ~y:ab ()) "y");
          less float_exact ~than:0.5 (at_a (dot ~x:ab ~y:(const 0.5) ()) "x"));
      test "merges explicit specifications" (fun () ->
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
      test "merges implied specifications without their name or transform"
        (fun () ->
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

(* Fitting *)

let fitting =
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
  let fixed = Scale.linear ~name:"lr" ~nice:false ~domain:(0., 10.) () in
  let on_fixed =
    dot ~x:(num ~scale:fixed (f64 [| 20.; 30. |])) ~y:(const 0.5) ()
  in
  group "fitting"
    [
      test "an explicit domain replaces the fitted one" (fun () ->
          equal floats (0., 10.) (hull (resolve on_fixed) "lr"));
      test "a zoom replaces the fitted and explicit domains" (fun () ->
          let view = View.set (View.zoom fixed) (Some (3., 4.)) View.empty in
          equal floats (3., 4.) (hull (resolve ~view on_fixed) "lr"));
      (* Scale's nice rule: a tenth of 0.74 is 7.4 × 10^-2, at least √50 times
         10^-2, so the step is 10 × 10^-2. *)
      test "rounds a linear hull outward to multiples of its step" (fun () ->
          let r =
            resolve (dot ~x:(num (f64 [| 0.13; 0.87 |])) ~y:(const 0.5) ())
          in
          equal floats (0.1, 0.9) (hull r "x"));
      cases "a categorical domain holds its channels' categories in order"
        ~id:(fun (n, _, _) -> n)
        domains
        (fun (_, f, expected) ->
          equal
            (list (pair int string))
            expected
            (categories (categ (resolve f) "color")));
      cases "an explicit categorical domain drops the rows outside it"
        ~id:(fun (n, _, _) -> n)
        kept
        (fun (_, f, expected) -> equal floats expected (hull (resolve f) "x"));
    ]

(* Facets *)

(* Each panel of a facetted mark whose y is independent per panel fits its y to
   the rows the panel draws. *)
let facets =
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
  group "facets"
    [
      test "a facet constant naming no panel still fits the other scales"
        (fun () ->
          let f =
            layer
              [
                dot1 [| 0.; 1. |] [| 0.; 0. |] ~fx:(strings [| "a"; "a" |]);
                dot1 [| 100. |] [| 0. |] ~fx:(const "z");
              ]
          in
          equal floats (0., 100.) (hull (resolve f) "x"));
      cases "facet channels put each row in the panel of its categories"
        ~id:(fun (n, _, _) -> n)
        panels
        (fun (_, f, expected) ->
          let r = resolve f in
          List.iter
            (fun (c, h) ->
              let cats = List.map field (String.split_on_char '.' c) in
              equal ~msg:c floats h
                (hull ~at:(path (field "panel" :: cats)) r "y"))
            expected);
    ]

(* View.zoom *)

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

(* A figure, a view, the hulls of scales at nodes, and the ids of the
   warnings. *)
let zoom_cases =
  [
    ( "a zoom sets the domain",
      on_lr,
      zooms lr (5., 6.) View.empty,
      [ (root, "lr", (5., 6.)) ],
      [] );
    ( "a zoom at a broadcast cell zooms that cell",
      layer [ cells; dot1 [| 2. |] [| 0. |] ],
      zooms ~at:(cell 1) x_scale (7., 8.) View.empty,
      [ (cell 0, "x", (0., 2.)); (cell 1, "x", (7., 8.)) ],
      [] );
    ( "a zoom of no scale is ignored",
      on_lr,
      zooms (Scale.linear ~name:"gone" ()) (5., 6.) View.empty,
      [ (root, "lr", (1., 2.)) ],
      [ root ] );
    ( "a zoom of a time scale named like a quantitative one is ignored",
      on_lr,
      View.set
        (View.zoom (Scale.time ~name:"lr" ()))
        (Some Time.(epoch, epoch))
        View.empty,
      [ (root, "lr", (1., 2.)) ],
      [ root ] );
    ( "a zoom at a node in several position scopes is ignored",
      layer [ cells; name "r" (dot1 [| 2. |] [| 0. |]) ],
      zooms ~at:(path [ field "r" ]) x_scale (7., 8.) View.empty,
      [ (cell 0, "x", (0., 2.)); (cell 1, "x", (2., 5.)) ],
      [ path [ field "r" ] ] );
    ( "a zoom whose node is gone is ignored",
      layer [ on_lr ],
      zooms ~at:(path [ field "gone" ]) lr (5., 6.) View.empty,
      [ (root, "lr", (1., 2.)) ],
      [ path [ field "gone" ] ] );
    ( "a zoom the scale cannot take is ignored",
      dot ~x:(num ~scale:lg (f64 [| 1.; 10. |])) ~y:(const 0.5) (),
      zooms lg (-1., 1.) View.empty,
      [ (root, "lg", (1., 10.)) ],
      [ root ] );
    ( "zooms of one scale at two nodes are ignored",
      layer [ on_lr |> name "a" ],
      View.empty
      |> zooms lr (5., 6.)
      |> zooms ~at:(path [ field "a" ]) lr (7., 8.),
      [ (root, "lr", (1., 2.)) ],
      [ root; path [ field "a" ] ] );
  ]

let zoom =
  group "View.zoom"
    [
      cases
        "applies in the scope holding its node, or is ignored with a warning"
        ~id:(fun (n, _, _, _, _) -> n)
        zoom_cases
        (fun (_, f, view, hulls, ws) ->
          let r = resolve ~view f in
          List.iter
            (fun (at, n, h) -> equal ~msg:n floats h (hull ~at r n))
            hulls;
          equal (list id) ws (List.map fst (Resolved.warnings r)));
      test "applies in the scope its node lies in after the structure changes"
        (fun () ->
          let a = dot1 [| 0. |] [| 0. |] |> name "a"
          and at = path [ field "a" ] in
          let view = zooms ~at x_scale (7., 8.) View.empty in
          let alone = resolve ~view (layer [ a ])
          and in_a_cell =
            resolve ~view (grid [ [ dot1 [| 5. |] [| 0. |]; a ] ])
          in
          equal floats (7., 8.) (hull alone "x");
          equal floats (7., 8.) (hull ~at in_a_cell "x");
          equal floats (5., 5.) (hull ~at:(path [ index 0 ]) in_a_cell "x"));
      test "sets a time domain" (fun () ->
          let z = (Time.v S 100L, Time.v S 200L) in
          let view =
            View.set (View.zoom (Scale.time ~name:"x" ())) (Some z) View.empty
          in
          let r =
            resolve ~view (dot ~x:(time S (i64 [| 0L; 1L |])) ~y:(const 0.5) ())
          in
          equal (pair instant instant) z (time_hull r));
    ]

(* layer *)

(* [numbered r c base] is an [r] × [c] grid whose cell [(i, j)] puts the x [base
   + 10 i + j]. *)
let numbered r c base =
  grid
    (List.init r (fun i ->
         List.init c (fun j ->
             dot1 [| base +. Float.of_int ((10 * i) + j) |] [| 0. |])))

(* Two grids layered: the cells of their broadcast shape each hold the cell of
   each grid it repeats, or [resolve] raises. A grid of no rows has no cells,
   whatever its columns. *)
let broadcast_law ((r, c), (r', c')) =
  let c = if r = 0 then 0 else c and c' = if r' = 0 then 0 else c' in
  let dim d d' =
    if d = d' || d' = 1 then Some d else if d = 1 then Some d' else None
  in
  let shape =
    match (dim r r', dim c c') with
    | Some rows, Some cols -> Some (rows, cols)
    | _ -> None
  in
  let cells = match shape with Some (rows, cols) -> rows * cols | None -> 0 in
  let nowhere = cells = 0 && (r * c > 0 || r' * c' > 0) in
  cover "shapes that do not broadcast" (Option.is_none shape);
  cover "a grid drawn nowhere" (Option.is_some shape && nowhere);
  cover "no cells" (Option.is_some shape && cells = 0 && not nowhere);
  cover "a grid repeats" (cells > 0 && (r <> r' || c <> c'));
  let f = layer [ numbered r c 0.; numbered r' c' 100. ] in
  match shape with
  | None -> invalid (fun () -> resolve f)
  | Some _ when nowhere -> invalid (fun () -> resolve f)
  | Some (_, cols) ->
      let res = resolve f in
      (* The x of the cell of an [r] × [c] grid repeated into cell (i, j). *)
      let x r c base i j =
        base
        +. Float.of_int ((if r = 1 then 0 else 10 * i) + if c = 1 then 0 else j)
      in
      for k = 0 to cells - 1 do
        let i = k / cols and j = k mod cols in
        let a = x r c 0. i j and b = x r' c' 100. i j in
        equal ~msg:(string_of_int k) floats (a, b) (hull ~at:(cell k) res "x")
      done;
      invalid (fun () -> hull ~at:(cell cells) res "x")

(* [same_page f f'] states that [f] and [f'] draw one page, or raise one
   error. *)
let same_page f f' =
  let p = drawn f in
  cover "a figure that lays out" (Result.is_ok p);
  equal page p (drawn f')

let layer_ =
  let t s = title s and c = coord (Coord.cartesian ~aspect:2. ()) in
  let marks = Gen.pair gen_mark gen_mark in
  group "layer"
    [
      prop "lifts a child's title above the layer" marks (fun (a, b) ->
          same_page (t "w" (layer [ a; b ])) (layer [ t "w" a; b ]));
      prop "lifts a child's coord over the layer" marks (fun (a, b) ->
          same_page (c (layer [ a; b ])) (layer [ a; c b ]));
      prop "lifts wrappers one at a time, outermost first, equal ones as one"
        marks (fun (a, b) ->
          same_page
            (t "w" (t "v" (layer [ a; b ])))
            (layer [ t "w" (t "v" a); t "w" b ]));
      prop "broadcasts the arrangements of its children" ~count:144
        Gen.(
          pair
            (pair (int_range 0 3) (int_range 1 3))
            (pair (int_range 0 3) (int_range 1 3)))
        broadcast_law;
      test "joins a mark to each cell of a grid" (fun () ->
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
      test "a nested layer draws what the flat one draws" (fun () ->
          let a = dot1 [| 1. |] [| 1. |] and b = dot1 [| 2. |] [| 2. |] in
          let painted_page f =
            Result.map (fun p -> painted (Renderable.picture p)) (drawn f)
          in
          equal paint
            (painted_page (layer [ a; b; a ]))
            (painted_page (layer [ layer [ a; b ]; a ])));
    ]

(* share *)

let share_ =
  let a = dot1 [| 1. |] [| 1. |] and b = dot1 [| 2. |] [| 2. |] in
  let p = share [ ("x", `Shared) ] in
  group "share"
    [
      test "applied twice means it applied once" (fun () ->
          equal page
            (drawn (p (grid [ [ a; b ] ])))
            (drawn (p (p (grid [ [ a; b ] ])))));
      test "of x on a grid of one cell is that grid" (fun () ->
          equal page (drawn (grid [ [ a ] ])) (drawn (p (grid [ [ a ] ]))));
    ]

(* bind *)

let bind_ =
  group "bind"
    [
      test "reads the view once per resolve" (fun () ->
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

(* Resolved.scale *)

let named = lazy (resolve (layer [ dot1 [| 1. |] [| 2. |] |> name "a" ]))

let resolved_scale =
  let filled x = dot1 x [| 0.; 1. |] ~fill:(num ~scale:exact (f64 x)) in
  let two = grid [ [ filled [| 0.; 1. |]; filled [| 5.; 6. |] ] ] in
  let apart = share [ ("color", `Independent) ] in
  let r () = Lazy.force named in
  let raises =
    [
      ( "an unnamed scale",
        (fun () -> ignore (Resolved.scale (r ()) (Scale.linear ()))),
        __POS_OF__ {| Hugin.Resolved.scale: the scale is unnamed |} );
      ( "an id no node has",
        (fun () -> ignore (quant ~at:(path [ field "b" ]) (r ()) "x")),
        __POS_OF__ {| Hugin.Resolved.scale: no node has the id b |} );
      ( "a name its scope lacks",
        (fun () -> ignore (quant (r ()) "lr")),
        __POS_OF__
          {| Hugin.Resolved.scale: the scope of root has no quantitative scale "lr" |}
      );
      ( "a scale of another kind",
        (fun () -> ignore (categ (r ()) "x")),
        __POS_OF__
          {| Hugin.Resolved.scale: the scope of root has no categorical scale "x" |}
      );
      ( "a time scale the scope lacks",
        (fun () -> ignore (Resolved.scale (r ()) (Scale.time ~name:"x" ()))),
        __POS_OF__
          {| Hugin.Resolved.scale: the scope of root has no temporal scale "x" |}
      );
      ( "a position at a layer repeated over cells",
        (fun () ->
          ignore
            (quant
               ~at:(path [ field "r" ])
               (resolve (layer [ two; name "r" (filled [| 2. |]) ]))
               "x")),
        __POS_OF__ {| Hugin.Resolved.scale: no scope of the scale "x" holds r |}
      );
      ( "a colour at a layer whose children keep it apart",
        (fun () ->
          let l =
            apart (layer [ filled [| 1. |]; filled [| 5. |] ]) |> name "l"
          in
          ignore
            (quant
               ~at:(path [ field "l" ])
               (resolve (grid [ [ l; filled [| 7. |] ] ]))
               "color")),
        __POS_OF__
          {| Hugin.Resolved.scale: no scope of the scale "color" holds l |} );
      ( "a colour at a facetted mark that keeps it per panel",
        (fun () ->
          let m =
            apart
              (dot ~x:(const 0.5) ~y:(const 0.5)
                 ~fill:(num (f64 [| 1.; 2. |]))
                 ~fx:(strings [| "a"; "b" |])
                 ())
            |> name "m"
          in
          ignore
            (quant ~at:(path [ field "m" ]) (resolve (layer [ m ])) "color")),
        __POS_OF__
          {| Hugin.Resolved.scale: no scope of the scale "color" holds m |} );
    ]
  in
  group "Resolved.scale"
    [
      test "finds a scale through a named node" (fun () ->
          equal floats (1., 1.) (hull ~at:(path [ field "a" ]) (r ()) "x"));
      prop "normalises alike in another figure"
        Gen.(pair (gen_values 4) (gen_values 3))
        (fun (xs, xs') ->
          let lr = Scale.log ~name:"lr" () in
          let figure s x = dot ~x:(num ~scale:s (f64 x)) ~y:(const 0.5) () in
          let s = Resolved.scale (resolve (figure lr xs)) lr in
          let s' = Resolved.scale (resolve (figure s xs')) lr in
          equal fitted s s';
          List.iter
            (fun v ->
              equal float_exact (Scale.normalize s v) (Scale.normalize s' v))
            [ 0.5; 1.; 7. ]);
      test "keeps the reverse y implies on a band scale" (fun () ->
          let r = resolve (dot ~x:(const 0.5) ~y:(strings [| "a"; "b" |]) ()) in
          let s = categ r "y" in
          let r' =
            resolve (dot ~y:(const 0.5) ~x:(strings ~scale:s [| "a"; "b" |]) ())
          in
          greater float_exact ~than:0.5 (Scale.normalize (categ r' "x") "a"));
      cases "raises on an unnamed scale or one no scope of its node has"
        ~id:(fun (n, _, _) -> n)
        raises
        (fun (_, f, msg) -> expect (message f) msg);
    ]

(* Resolved.warnings *)

(* A figure, a view, and the ids and messages of its warnings. *)
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
        path [ index 0 ];
        path [ index 1 ];
        path [ field "z" ];
        path [ index 3 ];
        root;
      ],
      __POS_OF__
        {|
        the facet constant "nowhere" of fx names no panel
        x: 1 finite value is missing for its scale
        the zoom of the scale "lg" sets a domain it cannot take
        the facet constant "nowhere" of fx names no panel
        the view sets "k", which no key of its sort reads
        |}
    );
    ( "a mark repeated over the cells of a grid is warned about once",
      layer [ grid [ [ layer []; layer [] ] ]; bad_log ],
      View.empty,
      [ path [ index 1 ] ],
      __POS_OF__ {| x: 1 finite value is missing for its scale |} );
    ( "a uint64 code beyond int is missing",
      dot ~x:(const 0.5) ~y:(const 0.5)
        ~fill:(cat (Nx.create Nx.uint64 [| 2 |] [| -1L; 1L |]))
        (),
      View.empty,
      [ root ],
      __POS_OF__ {| fill: 1 code is beyond the range of int |} );
    ( "a masked code outside the labels is not warned about",
      dot ~x:(const 0.5) ~y:(const 0.5)
        ~fill:
          (cat ~labels:[| "a" |]
             ~valid:(mask [| true; false; true |])
             (i32 [| 3; 5; 0 |]))
        (),
      View.empty,
      [ root ],
      __POS_OF__ {| fill: 1 code is outside its 1 labels |} );
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
      [ path [ index 0 ]; path [ field "c" ] ],
      __POS_OF__
        {|
        x: 2 finite values are missing for its scale
        fill: 1 code is outside its 1 labels
        |}
    );
  ]

let resolved_warnings =
  group "Resolved.warnings"
    [
      cases "are about nodes in the order of the figure, then about the view"
        ~id:(fun (n, _, _, _, _) -> n)
        warning_cases
        (fun (_, f, view, ids, msgs) ->
          let ws = Resolved.warnings (resolve ~view f) in
          equal (list id) ids (List.map fst ws);
          expect (messages ws) msgs);
    ]

(* Resolved.equal *)

let k = View.number "k" ~init:1.
let at_one () = dot1 [| 1. |] [| 0. |] ~fill:(num (f64 [| 1. |]))

let resolved_equal =
  group "Resolved.equal"
    [
      test "holds of a figure resolved twice" (fun () ->
          let a = at_one () in
          equal resolved (resolve a) (resolve a));
      test "tells apart figures that fit equal scales" (fun () ->
          let r = resolve (at_one ()) and r' = resolve (at_one ()) in
          equal fitted (quant r "x") (quant r' "x");
          equal fitted (quant r "color") (quant r' "color");
          not_equal resolved r r');
      test "tells apart views that change nothing" (fun () ->
          let g = bind k (fun _ -> at_one ()) in
          let view = View.set k 1. View.empty in
          equal (list id) []
            (List.map fst (Resolved.warnings (resolve ~view g)));
          not_equal resolved (resolve g) (resolve ~view g));
      test "compares no figure a bind returns" (fun () ->
          let f = bind k (fun v -> dot1 [| v |] [| v |]) in
          equal resolved (resolve f) (resolve f));
    ]

(* Resolved.pp *)

let resolved_pp =
  group "Resolved.pp"
    [
      test "prints the golden figures as their baseline" (fun () ->
          let open Hugin_test_figures in
          let one (name, f, _, _) =
            Format.asprintf "%s@\n%a@\n" name Resolved.pp (resolve (f ()))
          in
          expect_file ~file:"packages/hugin/test/golden/resolved.expected"
            (String.concat "\n" (List.map one Figures.goldens)));
    ]

(* resolve *)

(* A figure each composition error or conflict of the interface makes, and the
   message that names the nodes at fault. *)
let errors =
  let x = f64 [| 1.; 2. |] in
  let lr = Scale.linear ~name:"lr" () in
  [
    ( "two x scales in one panel",
      (fun () ->
        resolve
          (layer
             [
               dot ~x:(num ~scale:lr x) ~y:(num x) ();
               dot ~x:(num x) ~y:(num x) ();
             ])),
      __POS_OF__
        {| Hugin.resolve: 0 and 1 read two x scales in the panel root |} );
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
      __POS_OF__
        {| Hugin.resolve: the scale "lr" is read as quantitative by 0 and as categorical by 1 |}
    );
    ( "two kinds on x",
      (fun () ->
        resolve
          (layer [ dot1 [| 1. |] [| 1. |]; rect ~x:(dim 0) ~y:(num x) () ])),
      __POS_OF__
        {| Hugin.resolve: the scale "x" is read as quantitative by 0 and as categorical by 1 |}
    );
    ( "two explicit values of one property",
      (fun () ->
        resolve
          (layer
             [
               dot ~x:(num ~scale:(Scale.linear ~zero:true ()) x) ~y:(num x) ();
               dot ~x:(num ~scale:(Scale.linear ~zero:false ()) x) ~y:(num x) ();
             ])),
      __POS_OF__
        {| Hugin.resolve: 0 and 1 give the scale "x" two explicit values of zero |}
    );
    ( "two implied values of one property",
      (fun () ->
        let m c =
          Mark.v ~name:"m"
            [ Mark.bind ~imply:(Scale.linear ~clamp:c ()) Role.fill (num x) ]
            draw_nothing
        in
        resolve (layer [ m true; m false ])),
      __POS_OF__
        {| Hugin.resolve: 0 and 1 give the scale "color" two implied values of clamp |}
    );
    ( "two guides implied for one scale",
      (fun () ->
        let m g =
          Mark.v ~name:"m" [ Mark.bind ~guide:g Role.fill (num x) ] draw_nothing
        in
        resolve (layer [ m true; m false ])),
      __POS_OF__
        {| Hugin.resolve: 0 and 1 imply different guides for the scale "color" |}
    );
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
      __POS_OF__
        {| Hugin.resolve: 0 and 1 read labelled and indexed categories on the scale "color" |}
    );
    ( "a labelled domain read by indexed categories",
      (fun () ->
        resolve
          (dot ~x:(num x) ~y:(num x)
             ~fill:
               (dim ~scale:(Scale.band ~domain:(Scale.Labels [| "a" |]) ()) 0)
             ())),
      __POS_OF__
        {| Hugin.resolve: the domain of the scale "color" and root identify categories differently |}
    );
    ( "two texts for one index",
      (fun () ->
        resolve
          (layer
             [
               dot ~x:(num x) ~y:(num x) ~fill:(dim ~labels:[| "a"; "b" |] 0) ();
               dot ~x:(num x) ~y:(num x) ~fill:(dim ~labels:[| "a"; "c" |] 0) ();
             ])),
      __POS_OF__
        {| Hugin.resolve: 0 and 1 give the category 1 of the scale "color" two texts, "b" and "c" |}
    );
    ( "a wrapping fx with an fy",
      (fun () ->
        resolve
          (dot ~x:(num x) ~y:(num x)
             ~fx:(strings ~scale:(Scale.band ~wrap:2 ()) [| "a"; "b" |])
             ~fy:(strings [| "c"; "d" |])
             ())),
      __POS_OF__
        {| Hugin.resolve: the fx scale of root wraps, but the cell has an fy scale |}
    );
    ( "grids that do not broadcast",
      (fun () ->
        resolve
          (layer
             [
               grid [ [ layer [] ]; [ layer [] ] ];
               grid [ [ layer [] ]; [ layer [] ]; [ layer [] ] ];
             ])),
      __POS_OF__
        {| Hugin.resolve: the arrangements of the children of root do not broadcast |}
    );
    ( "a mark layered with a grid of no cells",
      (fun () -> resolve (layer [ dot1 [| 1. |] [| 1. |]; grid [] ])),
      __POS_OF__
        {| Hugin.resolve: a child of root with cells is layered with a grid of none |}
    );
    ( "a grid with spans and another grid",
      (fun () ->
        resolve
          (layer
             [
               grid [ [ span ~cols:2 (layer []) ] ];
               grid [ [ layer []; layer [] ] ];
             ])),
      __POS_OF__
        {| Hugin.resolve: a grid with spans under root broadcasts with a grid of other cells |}
    );
    ( "two titles among layered children",
      (fun () -> resolve (layer [ title "a" (layer []); title "b" (layer []) ])),
      __POS_OF__ {| Hugin.resolve: the children of root have different titles |}
    );
    ( "two titles that would lift at once from nested wrappers",
      (fun () ->
        resolve
          (layer
             [
               title "w" (title "v" (dot1 [| 1. |] [| 1. |]));
               title "v" (dot1 [| 2. |] [| 2. |]);
             ])),
      __POS_OF__ {| Hugin.resolve: the children of root have different titles |}
    );
    ( "rows of different lengths",
      (fun () -> resolve (grid [ [ layer []; layer [] ]; [ layer [] ] ])),
      __POS_OF__
        {| Hugin.resolve: the rows of root cover different numbers of columns |}
    );
    ( "a span past the last row",
      (fun () -> resolve (grid [ [ span ~rows:2 (layer []) ] ])),
      __POS_OF__ {| Hugin.resolve: the span 0 reaches past the last row |} );
    ( "a span over another cell",
      (fun () ->
        resolve
          (grid
             [
               [ layer []; layer []; span ~rows:2 (layer []) ];
               [ layer []; span ~cols:2 (layer []) ];
             ])),
      __POS_OF__ {| Hugin.resolve: the span 4 covers another cell |} );
    ( "widths of the wrong length",
      (fun () -> resolve (grid ~widths:[ 1. ] [ [ layer []; layer [] ] ])),
      __POS_OF__ {| Hugin.resolve: root has 1 widths for 2 tracks |} );
    ( "heights of the wrong length",
      (fun () -> resolve (grid ~heights:[ 1.; 1. ] [ [ layer [] ] ])),
      __POS_OF__ {| Hugin.resolve: root has 2 heights for 1 tracks |} );
    ( "a span outside a grid",
      (fun () -> resolve (layer [ span (layer []) ])),
      __POS_OF__ {| Hugin.resolve: 0 spans cells outside a grid |} );
    ( "share of a scale nothing reads",
      (fun () ->
        resolve
          (share [ ("size", `Shared) ] (layer [ dot ~x:(num x) ~y:(num x) () ]))),
      __POS_OF__
        {| Hugin.resolve: root shares the scale "size", which nothing under it reads |}
    );
    ( "independent x across a layer",
      (fun () ->
        resolve
          (share
             [ ("x", `Independent) ]
             (layer [ dot1 [| 1. |] [| 1. |]; dot1 [| 2. |] [| 2. |] ]))),
      __POS_OF__
        {| Hugin.resolve: root makes the scale "x" independent per layer child, but a position or facet reads it |}
    );
    ( "independent fx across a layer",
      (fun () ->
        resolve
          (share
             [ ("fx", `Independent) ]
             (layer
                [ dot ~fx:(strings [| "a" |]) ~x:(const 0.5) ~y:(const 0.5) () ]))),
      __POS_OF__
        {| Hugin.resolve: root makes the scale "fx" independent per layer child, but a position or facet reads it |}
    );
    ( "a facet scale independent per panel",
      (fun () ->
        resolve
          (share
             [ ("fx", `Independent) ]
             (dot ~x:(num x) ~y:(num x) ~fx:(strings [| "a"; "b" |]) ()))),
      __POS_OF__
        {| Hugin.resolve: root makes its facet scale "fx" independent per panel |}
    );
    ( "a user-named facet scale independent per panel",
      (fun () ->
        let g = Scale.band ~name:"g" () in
        resolve
          (share
             [ ("g", `Independent) ]
             (dot ~x:(num x) ~y:(num x)
                ~fx:(strings ~scale:g [| "a"; "b" |])
                ()))),
      __POS_OF__
        {| Hugin.resolve: root makes its facet scale "g" independent per panel |}
    );
    ( "two coordinate systems over a panel",
      (fun () ->
        resolve
          (coord (Coord.cartesian ())
             (layer
                [
                  coord (Coord.cartesian ~aspect:1. ()) (dot1 [| 1. |] [| 1. |]);
                ]))),
      __POS_OF__
        {| Hugin.resolve: the panel root lies under two coordinate systems, at root and 0 |}
    );
    ( "two implied coordinate systems",
      (fun () ->
        let m a =
          Mark.v ~name:"m" ~coord:(Coord.cartesian ~aspect:a ()) [] draw_nothing
        in
        resolve (layer [ m 1.; m 2. ])),
      __POS_OF__
        {| Hugin.resolve: 0 and 1 imply two coordinate systems in the panel root |}
    );
    ( "two names on a child",
      (fun () ->
        resolve (layer [ name "a" (name "b" (dot1 [| 1. |] [| 1. |])) ])),
      __POS_OF__ {| Hugin.resolve: the child 0 of root has two names |} );
    ( "two names under a wrapper",
      (fun () ->
        resolve
          (layer [ name "a" (title "t" (name "b" (dot1 [| 1. |] [| 1. |]))) ])),
      __POS_OF__ {| Hugin.resolve: the child 0 of root has two names |} );
    ( "two names on the root",
      (fun () -> resolve (name "a" (name "b" (layer [])))),
      __POS_OF__ {| Hugin.resolve: the root has two names |} );
    ( "two siblings of one name",
      (fun () -> resolve (layer [ name "a" (layer []); name "a" (layer []) ])),
      __POS_OF__ {| Hugin.resolve: two children of root are named a |} );
    ( "two keys of one name and two sorts",
      (fun () ->
        let n = View.number "k" ~init:1. and c = View.choice "k" ~init:"a" in
        resolve
          (layer [ bind n (fun _ -> layer []); bind c (fun _ -> layer []) ])),
      __POS_OF__
        {| Hugin.resolve: the figure reads two keys "k" of different sorts |} );
    ( "an axis of no position scale",
      (fun () -> resolve (layer [ dot1 [| 1. |] [| 1. |]; axis "color" ])),
      __POS_OF__
        {| Hugin.resolve: the axis 1 names "color", no position or facet scale of its panel |}
    );
    ( "an x axis on the left",
      (fun () ->
        resolve
          (layer [ dot1 [| 1. |] [| 1. |]; axis ~side:`Left "x" |> name "a" ])),
      __POS_OF__ {| Hugin.resolve: the axis a of "x" is on the left side |} );
    ( "a y axis of a user's name at the top",
      (fun () ->
        resolve
          (layer
             [
               dot ~x:(num x) ~y:(num ~scale:lr x) ();
               axis ~side:`Top "lr" |> name "a";
             ])),
      __POS_OF__ {| Hugin.resolve: the axis a of "lr" is on the top side |} );
    ( "two different axes for one scale",
      (fun () ->
        resolve
          (layer [ dot1 [| 1. |] [| 1. |]; axis "x"; axis ~grid:true "x" ])),
      __POS_OF__
        {| Hugin.resolve: the panel root holds two different axes for "x" |} );
    ( "two axes for one scale with different titles",
      (fun () ->
        resolve
          (layer
             [
               dot1 [| 1. |] [| 1. |]; axis ~title:"a" "x"; axis ~title:"b" "x";
             ])),
      __POS_OF__
        {| Hugin.resolve: the panel root holds two different axes for "x" |} );
    ( "a legend of no scale with a legend",
      (fun () -> resolve (layer [ dot1 [| 1. |] [| 1. |]; legend "x" ])),
      __POS_OF__
        {| Hugin.resolve: the legend 1 names "x", no scale with a legend in its scope |}
    );
    ( "two different legends for one scale",
      (fun () ->
        resolve
          (layer
             [
               dot ~fill:(num x) ~x:(num x) ~y:(num x) ();
               legend "color";
               legend ~show:false "color";
             ])),
      __POS_OF__
        {| Hugin.resolve: 1 and 2 are two different legends for "color" |} );
    ( "two legends for one scale with different titles",
      (fun () ->
        resolve
          (layer
             [
               dot ~fill:(num x) ~x:(num x) ~y:(num x) ();
               legend ~title:"a" "color";
               legend ~title:"b" "color";
             ])),
      __POS_OF__
        {| Hugin.resolve: 1 and 2 are two different legends for "color" |} );
    ( "two inside legends of one scale in different corners",
      (fun () ->
        resolve
          (layer
             [
               dot ~fill:(num x) ~x:(num x) ~y:(num x) ();
               legend ~side:(`Inside `Top_left) "color";
               legend ~side:(`Inside `Top_right) "color";
             ])),
      __POS_OF__
        {| Hugin.resolve: 1 and 2 are two different legends for "color" |} );
    ( "ncol for a colour bar",
      (fun () ->
        resolve
          (layer
             [
               dot ~fill:(num x) ~x:(num x) ~y:(num x) ();
               legend ~ncol:2 "color";
             ])),
      __POS_OF__
        {| Hugin.resolve: the legend 1 of "color" is a colour bar, which has no columns |}
    );
  ]

(* A traced tensor whose interpretation has returned: any operation on it
   raises, naming the interpretation. *)
type (_, _, _) Nx.Prim.payload += Outside

let traced () =
  Nx.Prim.interpret ~name:"Test.trace" Values
    (fun _ ~by:_ _ -> invalid_arg "Test.trace: no operation reaches the rule")
    (fun i -> Nx.Prim.traced i (Nx.Prim.form (f64 [| 0.; 1. |])) Outside)

(* A donated tensor: it is dead. *)
let dead () =
  let x = f64 [| 0.; 1. |] in
  ignore (Nx.neg (Nx.donate x));
  x

let resolve_ =
  let a = at_one () in
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
  let prev = Testable.make ~pp:Resolved.pp ~equal:( == ) in
  group "resolve"
    [
      cases
        "raises on a composition error or conflict, naming the nodes at fault"
        ~id:(fun (n, _, _) -> n)
        errors
        (fun (_, f, msg) -> expect (message f) msg);
      cases "raises what reading a tensor raises, prefixed by the mark's id"
        ~id:(fun (n, _, _) -> n)
        [ ("traced", traced, "Test.trace"); ("dead", dead, "donated") ]
        (fun (_, x, cause) ->
          let t = dot ~x:(num (x ())) ~y:(const 0.5) () |> name "t" in
          let m =
            message (fun () -> resolve (layer [ dot1 [| 1. |] [| 1. |]; t ]))
          in
          in_order ~subs:[ "t: "; cause ] m);
      test "returns prev for an equal figure and view" (fun () ->
          let f = layer [ a; dot1 [| 2. |] [| 3. |] ] in
          let r = resolve f in
          equal prev r (resolve ~prev:r f));
      prop "with prev of an equal figure equals a fresh resolve"
        (Gen.pair gen_mark gen_mark) (fun (a, b) ->
          let f = layer [ a; grid [ [ b; a ] ] ] in
          equal resolved (resolve f) (resolve ~prev:(resolve f) f));
      prop "with prev resolves a figure of new tensors afresh"
        Gen.(pair (gen_values 3) (gen_values 3))
        (fun (xs, xs') ->
          let f x = layer [ dot ~x:(num (f64 x)) ~y:(num (f64 x)) (); a ] in
          let f' = f xs' in
          equal resolved (resolve f') (resolve ~prev:(resolve (f xs)) f'));
      cases "with prev resolves afresh" ~id:fst again (fun (_, f) ->
          let first, f, view = f () in
          equal resolved (resolve ~view f)
            (resolve ~prev:(resolve first) ~view f));
    ]

let () =
  exit
    (run "hugin resolve"
       [
         missing;
         ids;
         time_;
         floats_;
         image_;
         contour_;
         names;
         scopes_;
         merging;
         fitting;
         facets;
         zoom;
         layer_;
         share_;
         bind_;
         resolved_scale;
         resolved_warnings;
         resolved_equal;
         resolved_pp;
         resolve_;
       ])
