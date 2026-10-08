(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module L = Nx_array.Layout
module M = Nx_array.Move
open Nx_array_gen

let ( let* ) = Result.bind
let layout = Testable.make ~pp:L.pp ~equal:L.equal
let ints = array int
let distinct ps = List.length (List.sort_uniq compare ps) = List.length ps
let check b reason = if b then Ok () else Error reason

(* [m]'s map from an index of its result to an index of its argument of shape
   [s], as Move's interface states it. *)
let source m s idx =
  let r = Array.length s in
  match m with
  | M.Reshape s' ->
      let k = ref 0 in
      Array.iteri (fun i j -> k := (!k * s'.(i)) + j) idx;
      let src = Array.make r 0 in
      for i = r - 1 downto 0 do
        src.(i) <- !k mod s.(i);
        k := !k / s.(i)
      done;
      src
  | M.Broadcast s' ->
      let off = Array.length s' - r in
      Array.init r (fun i -> if s.(i) = 1 then 0 else idx.(off + i))
  | M.Permute p ->
      let src = Array.make r 0 in
      Array.iteri (fun i a -> src.(a) <- idx.(i)) p;
      src
  | M.Slice rs -> Array.init r (fun i -> rs.(i).start + (idx.(i) * rs.(i).step))
  | M.Window ws ->
      let src = Array.sub idx 0 r in
      Array.iteri
        (fun j (w : M.window) ->
          src.(w.axis) <- (idx.(w.axis) * w.step) + (idx.(r + j) * w.dilation))
        ws;
      src

(* Whether strides express the positions [t] gives the indices of [s]: the
   offset and strides are forced by the index 0 and the unit indices, so the one
   candidate is checked against every index. *)
let expressible s t =
  match indices s with
  | [] -> true
  | idxs ->
      let r = Array.length s in
      let zero = Array.make r 0 in
      let o = t zero in
      let st =
        Array.init r (fun i ->
            if s.(i) < 2 then 0
            else
              let e = Array.copy zero in
              e.(i) <- 1;
              t e - o)
      in
      List.for_all
        (fun idx ->
          let p = ref o in
          Array.iteri (fun i j -> p := !p + (j * st.(i))) idx;
          !p = t idx)
        idxs

(* Whether [l] is in canonical form: an extent-1 axis has stride 0, and a layout
   with no element has offset 0 and every stride 0. *)
let canonical l =
  let s = L.shape l and st = L.strides l in
  let unit_axes = Array.for_all2 (fun d t -> d <> 1 || t = 0) s st in
  if L.numel l = 0 then
    unit_axes && L.offset l = 0 && Array.for_all (( = ) 0) st
  else unit_axes

(* Construction, by the interface *)

(* The product of [s]'s extents, if it fits in an [int]; a zero extent makes it
   zero. *)
let numel s =
  if Array.exists (( = ) 0) s then Some 0
  else
    Array.fold_left
      (fun n d ->
        Option.bind n (fun n -> if d > max_int / n then None else Some (n * d)))
      (Some 1) s

let add a b =
  match (a, b) with
  | Some a, Some b ->
      let c = a + b in
      if a >= 0 = (b >= 0) && c >= 0 <> (a >= 0) then None else Some c
  | _ -> None

(* Whether [(d - 1)·|t|], the reach of an axis, fits in an [int]. *)
let fits d t = d <= 1 || (t <> min_int && abs t <= max_int / (d - 1))

(* The offset and strides [v] makes of its arguments, or why it must raise. A
   position "does not fit" if the span's end, one past the greatest position,
   does not: a layout whose greatest position is [max_int] has no span. *)
let v_reference s strides offset =
  let r = Array.length s in
  let* () = check (r <= L.max_rank) "too many axes" in
  let* () = check (Array.for_all (fun d -> d >= 0) s) "a negative extent" in
  let* () = check (numel s <> None) "too many elements" in
  let* () = check (Array.length strides = r) "strides of another length" in
  let* () = check (Array.for_all2 fits s strides) "an axis' reach overflows" in
  if numel s = Some 0 then Ok (0, Array.make r 0)
  else
    let reach f =
      Array.fold_left add (Some offset)
        (Array.map2 (fun d t -> Some ((d - 1) * f t)) s strides)
    in
    let* () =
      match (reach (min 0), reach (max 0)) with
      | Some lo, Some hi when lo >= 0 && hi < max_int -> Ok ()
      | Some lo, Some _ when lo >= 0 -> Error "a position overflows"
      | Some _, Some _ | None, _ -> Error "a negative position"
      | Some _, None -> Error "a position overflows"
    in
    Ok (offset, Array.map2 (fun d t -> if d = 1 then 0 else t) s strides)

(* Arguments of [v] about the bounds it checks. *)
let v_args =
  let open Gen in
  let stride =
    frequency
      [
        (6, int_range (-3) 3);
        ( 1,
          ints_of
            [
              max_int;
              max_int / 2;
              (max_int / 2) + 1;
              max_int / 3;
              -(max_int / 2);
              -max_int;
              min_int;
            ] );
      ]
  in
  let shapes =
    frequency
      [ (8, shape); (1, array ~size:(int_range 1 3) (int_range (-1) 3)) ]
  in
  with_pp
    (fun ppf (s, st, o) ->
      Format.fprintf ppf "~offset:%d ~strides:%a %a" o pp_ints st pp_ints s)
    (let* s = shapes in
     let* n = frequency [ (8, const (Array.length s)); (1, int_range 0 5) ] in
     let+ strides = array ~size:(const n) stride
     and+ least =
       frequency
         [
           (6, int_range (-2) 12);
           (1, ints_of [ max_int; max_int - 1; min_int; max_int / 2 ]);
         ]
     and+ lifted = bool in
     (* Raise the offset by the reach of the negative strides, where it fits, so
        that a negative stride often meets an offset that keeps positions
        non-negative. *)
     let offset =
       if lifted && n = Array.length s then
         let up =
           Array.fold_left add (Some least)
             (Array.map2
                (fun d t ->
                  if d > 1 && t < 0 && t <> min_int then
                    if fits d t then Some ((d - 1) * -t) else None
                  else Some 0)
                s strides)
         in
         Option.value up ~default:least
       else least
     in
     (s, strides, offset))

let reasons =
  [
    "a negative extent";
    "strides of another length";
    "an axis' reach overflows";
    "a negative position";
    "a position overflows";
  ]

let law_v (s, strides, offset) =
  let expected = v_reference s strides offset in
  List.iter (fun r -> cover r (expected = Error r)) reasons;
  match expected with
  | Error _ -> raises_match Exn.invalid_arg (fun () -> L.v ~offset ~strides s)
  | Ok (o, st) ->
      let l = L.v ~offset ~strides s in
      cover "a negative stride"
        (Array.exists2 (fun d t -> d > 1 && t < 0) s strides);
      cover "no element" (numel s = Some 0);
      equal ints s (L.shape l);
      equal ints st (L.strides l);
      equal int o (L.offset l);
      equal int (Array.length s) (L.rank l);
      equal ~msg:"numel" (option int) (numel s) (Some (L.numel l));
      Array.iteri
        (fun i d ->
          equal ~msg:"dim" int d (L.dim l i);
          equal ~msg:"stride" int st.(i) (L.stride l i))
        s

let law_contiguous s =
  let l = L.contiguous s in
  let n = Option.get (numel s) in
  equal ints s (L.shape l);
  equal (list int) (List.init n Fun.id) (positions l)

let refusals =
  [
    ("contiguous of too many axes", fun () -> L.contiguous (Array.make 33 1));
    ("contiguous of a negative extent", fun () -> L.contiguous [| 2; -1 |]);
    ("contiguous of too many elements", fun () -> L.contiguous [| max_int; 2 |]);
    ( "v of too many axes",
      fun () -> L.v ~strides:(Array.make 33 0) (Array.make 33 1) );
    ( "v of an axis whose reach overflows",
      fun () -> L.v ~strides:[| (max_int / 2) + 1 |] [| 3 |] );
    ( "v of too many elements",
      fun () -> L.v ~strides:[| 0; 0 |] [| max_int; 2 |] );
  ]

let test_refuses (_, f) = raises_match Exn.invalid_arg f

let test_bounds () =
  equal int 32 L.max_rank;
  equal int 32 (L.rank (L.contiguous (Array.make 32 1)));
  equal int 32 (L.rank (L.v ~strides:(Array.make 32 0) (Array.make 32 2)));
  equal int max_int (L.numel (L.contiguous [| max_int |]));
  let reach = (max_int / 2) + 1 in
  equal ints [| reach |] (L.strides (L.v ~strides:[| reach |] [| 2 |]))

let test_no_element () =
  let s = [| max_int; 2; 0 |] in
  equal ints s (L.shape (L.contiguous s));
  equal ints s (L.shape (L.v ~strides:[| 0; 0; 0 |] s))

let test_axis_refuses () =
  let l = L.contiguous [| 2; 3 |] in
  List.iter
    (fun i ->
      raises_match ~msg:(string_of_int i) Exn.invalid_arg (fun () -> L.dim l i);
      raises_match ~msg:(string_of_int i) Exn.invalid_arg (fun () ->
          L.stride l i))
    [ -1; 2; min_int; max_int ]

let test_ownership () =
  let s = [| 2; 3 |] and strides = [| 3; 1 |] in
  let l = L.v ~strides s in
  s.(0) <- 1_000_000;
  strides.(1) <- 7;
  equal ints [| 2; 3 |] (L.shape l);
  equal ints [| 3; 1 |] (L.strides l);
  let l = L.contiguous s in
  s.(0) <- 2;
  equal ints [| 1_000_000; 3 |] (L.shape l);
  let out = L.shape l in
  out.(0) <- 5;
  equal ints [| 1_000_000; 3 |] (L.shape l);
  let out = L.strides l in
  out.(0) <- 5;
  equal ints [| 3; 1 |] (L.strides l)

(* The words [f] allocates over a thousand calls, beyond the loop's own. *)
let words f =
  let count g =
    let before = Gc.minor_words () in
    for _ = 1 to 1000 do
      ignore (Sys.opaque_identity (g ()))
    done;
    Gc.minor_words () -. before
  in
  count f -. count (fun () -> 0)

let test_no_allocation () =
  let l = L.contiguous [| 2; 3; 4; 5 |] in
  let l' = L.v ~offset:1 ~strides:[| 60; 20; 5; 1 |] [| 2; 3; 4; 5 |] in
  let zero = float 0.5 in
  equal ~msg:"rank" zero 0. (words (fun () -> L.rank l));
  equal ~msg:"dim" zero 0. (words (fun () -> L.dim l 2));
  equal ~msg:"stride" zero 0. (words (fun () -> L.stride l 2));
  equal ~msg:"offset" zero 0. (words (fun () -> L.offset l));
  equal ~msg:"numel" zero 0. (words (fun () -> L.numel l));
  equal ~msg:"is_contiguous" zero 0.
    (words (fun () -> Bool.to_int (L.is_contiguous l)));
  equal ~msg:"is_distinct" zero 0.
    (words (fun () -> Bool.to_int (L.is_distinct l)));
  equal ~msg:"equal" zero 0. (words (fun () -> Bool.to_int (L.equal l l')));
  equal ~msg:"hash" zero 0. (words (fun () -> L.hash l))

(* Movements *)

let law_move (l, m) =
  let s = L.shape l in
  match M.shape m s with
  | exception Invalid_argument _ ->
      cover "refused" true;
      raises_match Exn.invalid_arg (fun () -> L.move m l)
  | s' -> (
      let target idx = position l (source m s idx) in
      match L.move m l with
      | Some l' ->
          (match m with
          | M.Reshape _ -> cover "a reshape view" true
          | M.Broadcast _ -> cover "broadcast" true
          | M.Permute _ -> cover "permute" true
          | M.Slice _ -> cover "slice" true
          | M.Window _ -> cover "window" true);
          equal ints s' (L.shape l');
          List.iter
            (fun idx ->
              equal
                ~msg:(Format.asprintf "%a" pp_ints idx)
                int (target idx) (position l' idx))
            (indices s')
      | None ->
          cover "a reshape no strides express" true;
          (match m with
          | M.Reshape _ -> ()
          | _ -> failf "%a answered None" pp_move m);
          equal ~msg:"strides express the reshape" bool false
            (expressible s' target))

let test_move_ownership () =
  let s' = [| 3; 2 |] in
  let l = Option.get (L.move (M.Reshape s') (L.contiguous [| 6 |])) in
  s'.(0) <- 1_000_000;
  equal ints [| 3; 2 |] (L.shape l);
  let p = [| 1; 0 |] in
  let l = Option.get (L.move (M.Permute p) l) in
  p.(0) <- 0;
  equal ints [| 2; 3 |] (L.shape l)

(* Canonical form *)

let law_canonical_move (l, m) =
  match L.move m l with
  | exception Invalid_argument _ -> ()
  | None -> ()
  | Some l' ->
      cover "an extent-1 axis" (Array.mem 1 (L.shape l'));
      cover "no element" (L.numel l' = 0);
      if not (canonical l') then failf "%a is not canonical" L.pp l'

(* Two layouts of one shape over strides in [-1, 1], so that many map indices
   alike. *)
let similar =
  let open Gen in
  with_pp
    (fun ppf (a, b) -> Format.fprintf ppf "%a, %a" L.pp a L.pp b)
    (let* s = shape in
     let r = const (Array.length s) in
     let layout =
       let+ strides = array ~size:r (int_range (-1) 1)
       and+ least = int_range 2 3 in
       L.v ~offset:(lift s strides least) ~strides s
     in
     pair layout layout)

let law_equal (a, b) =
  let same = positions a = positions b in
  cover "the same map" same;
  cover "different maps" (not same);
  equal bool same (L.equal a b)

let law_equivalence pair = Law.equivalence layout pair

let law_hash (a, b) =
  cover "equal" (L.equal a b);
  if L.equal a b then equal int (L.hash a) (L.hash b)

(* [v] on [l]'s own fields, with every extent-1 stride respelled. *)
let respell l =
  let s = L.shape l in
  let strides =
    Array.mapi (fun i t -> if s.(i) = 1 then 7 else t) (L.strides l)
  in
  L.v ~offset:(L.offset l) ~strides s

let law_respelled l =
  equal layout l (respell l);
  equal int (L.hash l) (L.hash (respell l))

let test_equal_shapes () =
  equal bool false (L.equal (L.contiguous [| 1 |]) (L.contiguous [||]));
  equal bool false (L.equal (L.contiguous [| 0; 2 |]) (L.contiguous [| 2; 0 |]))

(* Flags and span *)

let law_contiguous_flag l =
  let ps = positions l in
  let c_order = List.mapi (fun k _ -> L.offset l + k) ps in
  cover "contiguous" (ps = c_order && ps <> []);
  cover "not contiguous" (ps <> c_order);
  equal bool (ps = c_order) (L.is_contiguous l)

(* The distinct test: axes of extent above 1 ordered by [|stride|], ties by
   axis, each stride past the reach of the axes before it. A layout with no
   element reaches no position twice, whatever its strides. *)
let distinct_rule l =
  let axes =
    List.filter (fun i -> L.dim l i > 1) (List.init (L.rank l) Fun.id)
  in
  let key i = (abs (L.stride l i), i) in
  let axes = List.sort (fun i j -> compare (key i) (key j)) axes in
  let rec go reach = function
    | [] -> true
    | i :: rest ->
        abs (L.stride l i) > reach
        && go (reach + ((L.dim l i - 1) * abs (L.stride l i))) rest
  in
  L.numel l = 0 || go 0 axes

let law_distinct_rule l =
  cover "distinct" (L.is_distinct l);
  cover "not distinct" (not (L.is_distinct l));
  cover "no element" (L.numel l = 0);
  equal bool (distinct_rule l) (L.is_distinct l)

let law_distinct_sound l =
  cover "distinct" (L.is_distinct l);
  cover "a repeated position" (not (distinct (positions l)));
  if L.is_distinct l then equal bool true (distinct (positions l))

let law_distinct_views l =
  cover "a strided view" ((not (L.is_contiguous l)) && L.numel l > 1);
  equal bool true (L.is_distinct l)

let law_span l =
  let lo, hi = L.span l in
  at_least ~msg:"lo" int ~than:0 lo;
  match positions l with
  | [] -> equal (pair int int) (0, 0) (lo, hi)
  | ps ->
      List.iter
        (fun p -> if p < lo || p >= hi then failf "%d outside [%d, %d)" p lo hi)
        ps

let law_header l =
  let flag b f = if b then f else 0 in
  let lo, hi = L.span l in
  let r = L.rank l in
  let flags =
    flag (L.is_contiguous l) 1
    lor flag (L.is_distinct l) 2
    lor flag (L.numel l = 0) 4
  in
  equal ints
    (Array.concat
       [ [| r; flags; L.offset l; lo; hi |]; L.shape l; L.strides l ])
    (Nx_array_support.layout l)

(* Coalescing *)

let same_shape =
  let open Gen in
  with_pp
    (fun ppf ls -> Format.pp_print_list L.pp ppf ls)
    (let* l = one_of [ strided; reached ~apart:false ] in
     let s = L.shape l in
     let other = one_of [ strided_of s; constant (L.contiguous s) ] in
     let+ others = list ~size:(int_range 0 3) other in
     l :: others)

(* Whether axes [i] and [i + 1] of [l] lie as one run. *)
let one_run l i = L.stride l i = L.dim l (i + 1) * L.stride l (i + 1)

let law_coalesce ls =
  let cs = L.coalesce (Array.of_list ls) in
  let s = L.shape cs.(0) in
  let rank = List.fold_left (fun r l -> max r (L.rank l)) 1 ls in
  cover "merges axes" (Array.length s < rank);
  equal ~msg:"layouts" int (List.length ls) (Array.length cs);
  at_least ~msg:"rank" int ~than:1 (Array.length s);
  Array.iter (fun c -> equal ~msg:"one shape" ints s (L.shape c)) cs;
  if Array.length s > 1 && Array.mem 1 s then
    failf "the extent-1 axis of %a is kept" pp_ints s;
  for i = 0 to Array.length s - 2 do
    if Array.for_all (fun c -> one_run c i) cs then
      failf "axes %d and %d of %a lie as one run" i (i + 1) pp_ints s
  done;
  List.iteri (fun k l -> equal (list int) (positions l) (positions cs.(k))) ls

let test_coalesce_counts () =
  let a = L.contiguous [| 2; 3 |] in
  equal int 4 (Array.length (L.coalesce [| a; a; a; a |]));
  raises_match Exn.invalid_arg (fun () -> L.coalesce [||]);
  raises_match Exn.invalid_arg (fun () -> L.coalesce [| a; a; a; a; a |]);
  raises_match Exn.invalid_arg (fun () ->
      L.coalesce [| a; L.contiguous [| 3; 2 |] |])

let test_pp () =
  let l = L.v ~offset:17 ~strides:[| 23; 19 |] [| 7; 5 |] in
  in_order ~subs:[ "7"; "5"; "23"; "19"; "17" ] (Format.asprintf "%a" L.pp l)

let tests =
  [
    group "construction"
      [
        test "max_rank is 32, and 32 axes are accepted" test_bounds;
        prop ~count:1000
          "v keeps its arguments in canonical form, and raises iff they do not \
           fit"
          v_args law_v;
        prop "contiguous places element k at position k" shape law_contiguous;
        cases ~name:fst "refuses" refusals test_refuses;
        test "a shape with no element takes any other extent" test_no_element;
        test "dim and stride refuse an axis out of range" test_axis_refuses;
        test "no array v takes is kept, none returned is held" test_ownership;
        test "queries, flags, equal and hash allocate nothing"
          test_no_allocation;
      ];
    group "movements"
      [
        prop ~count:500
          "a movement is its index map, None is a reshape no strides express, \
           and it raises as Move.shape does"
          layout_and_move law_move;
        prop "a movement's layout is canonical" layout_and_move
          law_canonical_move;
        test "no array a movement takes is kept" test_move_ownership;
      ];
    group "equality"
      [
        prop "layouts of one shape are equal iff they map indices alike" similar
          law_equal;
        prop "equal is an equivalence" similar law_equivalence;
        prop "equal layouts hash alike" similar law_hash;
        prop
          "v of a layout's fields is the layout, whatever its extent-1 strides"
          any_layout law_respelled;
        test "layouts of other shapes differ" test_equal_shapes;
      ];
    group "flags"
      [
        prop "is_contiguous iff C order is offset + k" any_layout
          law_contiguous_flag;
        prop "is_distinct is the test its interface states" any_layout
          law_distinct_rule;
        prop "is_distinct is never true of a repeated position" any_layout
          law_distinct_sound;
        prop "is_distinct holds of views that keep positions apart"
          (Gen.with_pp L.pp (reached ~apart:true))
          law_distinct_views;
        prop "span holds every position, from 0 up" any_layout law_span;
        prop "nx_array.h reads the fields the accessors give" any_layout
          law_header;
      ];
    group "coalesce"
      [
        prop
          "merges every run, drops extent-1 axes and keeps each layout's \
           positions in order"
          same_shape law_coalesce;
        test "takes 1 to 4 layouts of one shape" test_coalesce_counts;
      ];
    test "pp shows shape, strides and offset" test_pp;
  ]

let () = exit (run "nx_array.layout" tests)
