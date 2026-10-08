(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module L = Nx_array.Layout
module M = Nx_array.Move
open Nx_array_gen

let layout = Testable.make ~pp:L.pp ~equal:L.equal
let ints = array int
let distinct ps = List.length (List.sort_uniq compare ps) = List.length ps

(* [m]'s map from an index of its result to an index of its argument of shape
   [s]. *)
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

(* Movements *)

let law_move (l, m) =
  let s = L.shape l in
  let s' = M.shape m s in
  let target idx = position l (source m s idx) in
  match L.move m l with
  | Some l' ->
      (match m with
      | M.Reshape _ -> cover "reshape is a view" true
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
      cover "reshape is refused" true;
      (match m with
      | M.Reshape _ -> ()
      | _ -> failf "%a answered None" pp_move m);
      equal ~msg:"strides express the reshape" bool false
        (expressible s' target)

let test_move_ownership () =
  let s' = [| 3; 2 |] in
  let l = Option.get (L.move (M.Reshape s') (L.contiguous [| 6 |])) in
  s'.(0) <- 1_000_000;
  equal ints [| 3; 2 |] (L.shape l);
  let p = [| 1; 0 |] in
  let l = Option.get (L.move (M.Permute p) l) in
  p.(0) <- 0;
  equal ints [| 2; 3 |] (L.shape l)

let refuses name m s =
  test name (fun () -> raises_match Exn.invalid_arg (fun () -> M.shape m s))

let refusals =
  let r start count step = { M.start; count; step } in
  let w axis size step dilation = { M.axis; size; step; dilation } in
  [
    refuses "a reshape to another number of elements" (M.Reshape [| 5 |])
      [| 2; 3 |];
    refuses "a reshape to a negative extent" (M.Reshape [| -1; -6 |]) [| 6 |];
    refuses "a reshape past the most axes" (M.Reshape (Array.make 33 1)) [| 1 |];
    refuses "a broadcast to fewer axes" (M.Broadcast [| 3 |]) [| 1; 3 |];
    refuses "a broadcast of an extent above 1" (M.Broadcast [| 4 |]) [| 3 |];
    refuses "a broadcast whose elements overflow"
      (M.Broadcast [| max_int; 2 |])
      [| 1; 2 |];
    refuses "a permutation repeating an axis" (M.Permute [| 0; 0 |]) [| 2; 2 |];
    refuses "a permutation of other axes" (M.Permute [| 0 |]) [| 2; 2 |];
    refuses "a slice of step 0" (M.Slice [| r 0 1 0 |]) [| 3 |];
    refuses "a slice of negative count" (M.Slice [| r 0 (-1) 1 |]) [| 3 |];
    refuses "a slice starting past the axis" (M.Slice [| r 3 1 1 |]) [| 3 |];
    refuses "a slice ending past the axis" (M.Slice [| r 0 3 2 |]) [| 4 |];
    refuses "a reversed slice ending before 0"
      (M.Slice [| r 1 3 (-1) |])
      [| 4 |];
    refuses "a slice whose step overflows" (M.Slice [| r 1 2 max_int |]) [| 4 |];
    refuses "a reversed slice of the least step"
      (M.Slice [| r 3 2 min_int |])
      [| 4 |];
    refuses "a slice of a missing axis" (M.Slice [||]) [| 4 |];
    refuses "a window larger than its axis" (M.Window [| w 0 4 1 1 |]) [| 3 |];
    refuses "a dilated window past its axis" (M.Window [| w 0 2 1 3 |]) [| 3 |];
    refuses "a window on an empty axis" (M.Window [| w 0 1 1 1 |]) [| 0 |];
    refuses "a window of step 0" (M.Window [| w 0 1 0 1 |]) [| 3 |];
    refuses "windows on one axis twice"
      (M.Window [| w 0 1 1 1; w 0 1 1 1 |])
      [| 3 |];
    refuses "windows out of axis order"
      (M.Window [| w 1 1 1 1; w 0 1 1 1 |])
      [| 3; 3 |];
    refuses "a window past the most axes"
      (M.Window [| w 0 1 1 1 |])
      (Array.make 32 1);
    refuses "a dilation whose reach overflows"
      (M.Window [| w 0 3 1 max_int |])
      [| 3 |];
  ]

let test_window_shape () =
  let w = { M.axis = 1; size = 3; step = 2; dilation = 2 } in
  equal ints [| 2; 4; 4; 3 |] (M.shape (M.Window [| w |]) [| 2; 11; 4 |]);
  equal ints [| 2; 1; 4; 3 |] (M.shape (M.Window [| w |]) [| 2; 5; 4 |])

(* Construction *)

let test_v_refuses () =
  let fails f = raises_match Exn.invalid_arg f in
  fails (fun () -> L.v ~strides:(Array.make 33 0) (Array.make 33 1));
  fails (fun () -> L.v ~strides:[| 1 |] [| -1 |]);
  fails (fun () -> L.v ~strides:[| 1 |] [| 2; 2 |]);
  fails (fun () -> L.v ~strides:[| (max_int / 2) + 1 |] [| 3 |]);
  fails (fun () -> L.v ~strides:[| min_int |] [| 2 |]);
  fails (fun () -> L.v ~offset:max_int ~strides:[| 1 |] [| 2 |]);
  fails (fun () -> L.v ~offset:min_int ~strides:[| -1 |] [| 2 |]);
  fails (fun () ->
      L.v ~offset:2 ~strides:[| max_int / 2; max_int / 2 |] [| 2; 2 |]);
  fails (fun () -> L.contiguous [| max_int; 2 |]);
  fails (fun () -> L.contiguous (Array.make 33 1));
  fails (fun () -> L.dim (L.contiguous [| 2 |]) 1);
  fails (fun () -> L.stride (L.contiguous [| 2 |]) (-1))

let test_v_ownership () =
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

let test_canonical_cases () =
  equal layout (L.contiguous [| 1; 3 |]) (L.v ~strides:[| 99; 1 |] [| 1; 3 |]);
  equal layout
    (L.contiguous [| 0; 3 |])
    (L.v ~offset:7 ~strides:[| 5; -2 |] [| 0; 3 |]);
  equal ints [| 0; 1 |] (L.strides (L.contiguous [| 1; 3 |]));
  equal int 0 (L.offset (L.v ~offset:7 ~strides:[| 1 |] [| 0 |]))

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
       and+ offset = int_range 2 3 in
       L.v ~offset ~strides s
     in
     pair layout layout)

let law_canonical (a, b) =
  let same = positions a = positions b in
  cover "the same map" same;
  cover "different maps" (not same);
  equal bool same (L.equal a b)

let law_idempotent l =
  equal layout l (L.v ~offset:(L.offset l) ~strides:(L.strides l) (L.shape l))

(* Flags and span *)

let law_contiguous l =
  let ps = positions l in
  let c_order = List.mapi (fun k _ -> L.offset l + k) ps in
  cover "contiguous" (ps = c_order && ps <> []);
  equal bool (ps = c_order) (L.is_contiguous l)

let law_distinct_sound l =
  cover "distinct" (L.is_distinct l);
  if L.is_distinct l then equal bool true (distinct (positions l))

let law_distinct_exact l =
  cover "a strided view" ((not (L.is_contiguous l)) && L.numel l > 1);
  equal bool (distinct (positions l)) (L.is_distinct l)

let law_span l =
  let lo, hi = L.span l in
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
     let other =
       one_of
         [
           (let+ strides =
              array ~size:(const (Array.length s)) (int_range (-5) 5)
            and+ offset = int_range 0 40 in
            L.v ~offset ~strides s);
           constant (L.contiguous s);
         ]
     in
     let+ others = list ~size:(int_range 0 3) other in
     l :: others)

let law_coalesce ls =
  let cs = L.coalesce (Array.of_list ls) in
  let rank = List.fold_left (fun r l -> max r (L.rank l)) 1 ls in
  cover "merges axes" (L.rank cs.(0) < rank);
  List.iteri
    (fun k l ->
      at_most ~msg:"rank" int ~than:rank (L.rank cs.(k));
      equal (list int) (positions l) (positions cs.(k)))
    ls

let test_coalesce_refuses () =
  let a = L.contiguous [| 2; 3 |] in
  raises_match Exn.invalid_arg (fun () ->
      L.coalesce [| a; L.contiguous [| 3; 2 |] |]);
  raises_match Exn.invalid_arg (fun () -> L.coalesce [||]);
  raises_match Exn.invalid_arg (fun () -> L.coalesce [| a; a; a; a; a |])

let test_coalesce_cases () =
  let c = L.coalesce [| L.contiguous [| 2; 1; 3 |] |] in
  equal layout (L.contiguous [| 6 |]) c.(0);
  let t =
    Option.get (L.move (M.Permute [| 1; 0 |]) (L.contiguous [| 2; 3 |]))
  in
  let c = L.coalesce [| L.contiguous [| 3; 2 |]; t |] in
  equal ints [| 3; 2 |] (L.shape c.(1));
  let c = L.coalesce [| L.contiguous [| 1; 1 |] |] in
  equal ints [| 1 |] (L.shape c.(0));
  let c = L.coalesce [| L.contiguous [| 2; 0 |] |] in
  equal ints [| 0 |] (L.shape c.(0))

let tests =
  [
    group "movements"
      [
        prop ~count:500
          "a movement is its index map, and None is a reshape no strides \
           express"
          layout_and_move law_move;
        test "no array a movement takes is kept" test_move_ownership;
        test "windows count and append axes" test_window_shape;
        group "Move.shape refuses" refusals;
      ];
    group "construction"
      [
        test "v and contiguous refuse what does not fit" test_v_refuses;
        test "no array v takes is kept, none returned is held" test_v_ownership;
        test "extent-1 axes and empty layouts are canonical"
          test_canonical_cases;
        prop "layouts of one shape are equal iff they map indices alike" similar
          law_canonical;
        prop "v is idempotent on its output" any_layout law_idempotent;
      ];
    group "flags"
      [
        prop "is_contiguous iff C order is offset + k" any_layout law_contiguous;
        prop "is_distinct is never true of a repeated position" any_layout
          law_distinct_sound;
        prop "is_distinct is exact for views that keep positions apart"
          (Gen.with_pp L.pp (reached ~apart:true))
          law_distinct_exact;
        prop "span holds every position" any_layout law_span;
        prop "nx_array.h reads the fields the accessors give" any_layout
          law_header;
      ];
    group "coalesce"
      [
        prop "coalesced layouts reach the same positions in order" same_shape
          law_coalesce;
        test "coalesce merges runs and drops extent-1 axes" test_coalesce_cases;
        test "coalesce refuses other shapes and counts" test_coalesce_refuses;
      ];
  ]

let () = exit (run "nx_array.layout" tests)
