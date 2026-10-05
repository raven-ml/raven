(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The layout a view reports is that of its shape, strides and offset, whatever
   movements made it; reshapes and coalesced operands keep each element's
   position. *)

open Windtrap
module V = Nx_array.View

(* Whether the [k]th element of [v] in C order is at [offset v + k]. *)
let in_c_order v =
  let shape = V.shape v and strides = V.strides v in
  List.for_all
    (fun k ->
      let idx = Nx_test.unravel shape k in
      let p = ref (V.offset v) in
      Array.iteri (fun a i -> p := !p + (i * strides.(a))) idx;
      !p = V.offset v + k)
    (List.init (V.numel v) Fun.id)

type step = { name : string; apply : V.t -> V.t }

let steps =
  let first v = if V.ndim v = 0 then 0 else V.dim 0 v in
  let all v x = Array.make (V.ndim v) x in
  [
    {
      name = "transposed";
      apply =
        (fun v ->
          V.permute v (Array.init (V.ndim v) (fun a -> V.ndim v - 1 - a)));
    };
    { name = "flipped"; apply = (fun v -> V.flip v (all v true)) };
    {
      name = "without its first row";
      apply =
        (fun v ->
          if first v = 0 then v
          else
            V.shrink v
              (Array.mapi
                 (fun a n -> if a = 0 then (1, n) else (0, n))
                 (V.shape v)));
    };
    {
      name = "every other row";
      apply =
        (fun v ->
          if first v = 0 then v
          else
            V.reshape
              (V.sliding_window v ~axis:0 ~window:1 ~step:2)
              (Array.mapi
                 (fun a n -> if a = 0 then ((n - 1) / 2) + 1 else n)
                 (V.shape v)));
    };
    {
      name = "its axes of one element broadcast to 3";
      apply =
        (fun v ->
          if V.ndim v = 0 then V.expand v [| 3; 2 |]
          else
            V.expand v (Array.map (fun n -> if n = 1 then 3 else n) (V.shape v)));
    };
    {
      name = "flattened, if it can be";
      apply =
        (fun v ->
          let flat = [| V.numel v |] in
          if V.can_reshape v flat then V.reshape v flat else v);
    };
    {
      name = "with a leading axis of one element";
      apply = (fun v -> V.reshape v (Array.append [| 1 |] (V.shape v)));
    };
    {
      name = "a scalar, if it has one element";
      apply = (fun v -> if V.numel v = 1 then V.reshape v [||] else v);
    };
  ]

let moved =
  let open Gen in
  let* shape = array ~size:(int_range 0 3) (int_range 0 3) in
  let+ path = list ~size:(int_range 0 4) (of_list steps) in
  (shape, path)

let pp_moved ppf (shape, path) =
  Format.fprintf ppf "%a, %a" Nx_test.pp_shape shape
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
       (fun ppf s -> Format.pp_print_string ppf s.name))
    path

let layout =
  group "layout"
    [
      prop "a view is C-contiguous exactly when its elements lie in C order"
        (Gen.with_pp pp_moved moved) (fun (shape, path) ->
          let v = List.fold_left (fun v s -> s.apply v) (V.create shape) path in
          equal bool (in_c_order v) (V.is_c_contiguous v));
      test "a scalar broadcast to more than one element is not C-contiguous"
        (fun () ->
          is_false (V.is_c_contiguous (V.expand (V.create [||]) [| 3 |])));
    ]

(* Whether every position [v] reaches is in [0, n - 1], by enumerating them. *)
let reaches_inside v n =
  let shape = V.shape v and strides = V.strides v in
  let rec go axis pos =
    if axis = Array.length shape then pos >= 0 && pos < n
    else
      let rec each i =
        i = shape.(axis)
        || (go (axis + 1) (pos + (i * strides.(axis))) && each (i + 1))
      in
      each 0
  in
  go 0 (V.offset v)

let small_views =
  Gen.with_pp Nx_test.pp_bounded_view
    Gen.(
      let* rank = int_range 0 3 in
      let+ n = int_range 0 8
      and+ shape = array ~size:(constant rank) (int_range 0 3)
      and+ strides = array ~size:(constant rank) (int_range (-3) 3)
      and+ offset = int_range (-3) 8 in
      (n, V.create ~offset ~strides shape))

let big x = x > 1 lsl 30 || x < -(1 lsl 30)

let bounds =
  group "bounds"
    [
      prop "within holds exactly when every position the view reaches is inside"
        small_views (fun (n, v) ->
          let ok = reaches_inside v n in
          cover "inside" ok;
          cover "outside" (not ok);
          equal bool ok (V.within v n));
      prop
        "within holds exactly when the view has no negative dimension, at most \
         max_int elements, and its extreme positions are inside"
        Nx_test.edge_views (fun (n, v) ->
          let ok = Nx_test.view_inside v n
          and shape = V.shape v
          and strides = V.strides v in
          cover "inside" ok;
          cover "outside" (not ok);
          cover "a zero-size view" (Array.exists (( = ) 0) shape);
          cover "a negative dimension" (Array.exists (fun d -> d < 0) shape);
          cover "an element count past max_int" (Nx_test.count_overflows shape);
          cover "an extreme offset or stride"
            (big (V.offset v) || Array.exists big strides);
          equal bool ok (V.within v n));
    ]

(* The positions of [v]'s elements, in C order. *)
let positions v =
  let shape = V.shape v and strides = V.strides v in
  List.init (V.numel v) (fun k ->
      let idx = Nx_test.unravel shape k in
      let p = ref (V.offset v) in
      Array.iteri (fun a i -> p := !p + (i * strides.(a))) idx;
      !p)

(* A moved view and a shape of as many elements: the candidates that divide what
   remains of the count, in order, then the rest, with axes of one element among
   them. *)
let reshaped =
  let open Gen in
  let* shape, path = moved in
  let+ candidates = list ~size:(int_range 0 4) (int_range 1 4) in
  let v = List.fold_left (fun v s -> s.apply v) (V.create shape) path in
  let n = V.numel v in
  let rest = ref n and dims = ref [] in
  List.iter
    (fun d ->
      if n > 0 && !rest mod d = 0 then begin
        dims := d :: !dims;
        rest := !rest / d
      end)
    candidates;
  (v, Array.of_list (List.rev (!rest :: !dims)))

let pp_reshaped ppf (v, shape) =
  Format.fprintf ppf "%a as %a" Nx_test.pp_shape (V.shape v) Nx_test.pp_shape
    shape

let reshape =
  group "reshape"
    [
      prop "a reshape it can view keeps each element's position in C order"
        (Gen.with_pp pp_reshaped reshaped) (fun (v, shape) ->
          let viewable = V.can_reshape v shape in
          cover "viewed through strides" (viewable && not (V.is_c_contiguous v));
          cover "refused" (not viewable);
          if viewable then
            equal (list int) (positions v) (positions (V.reshape v shape)));
    ]

(* Operands of one shape: each a C-contiguous view of it, or moved by
   transposing, flipping or broadcasting, or of strides of its own. *)
let operands =
  let open Gen in
  let* shape = array ~size:(int_range 0 4) (int_range 0 3) in
  let rank = Array.length shape in
  let operand =
    one_of
      [
        constant (V.create shape);
        map
          (fun flips -> V.flip (V.create shape) flips)
          (array ~size:(constant rank) bool);
        map
          (fun keep ->
            V.expand
              (V.create
                 (Array.mapi (fun a n -> if keep.(a) then n else 1) shape))
              shape)
          (array ~size:(constant rank) bool);
        map
          (fun (offset, strides) -> V.create ~offset ~strides shape)
          (pair (int_range 0 8)
             (array ~size:(constant rank) (int_range (-4) 4)));
      ]
  in
  list ~size:(int_range 1 3) operand

let pp_operands ppf vs =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
    (fun ppf v ->
      Format.fprintf ppf "%a strides %a offset %d" Nx_test.pp_shape (V.shape v)
        Nx_test.pp_shape (V.strides v) (V.offset v))
    ppf vs

let coalesce =
  group "coalesce"
    [
      prop "each result reaches its view's elements in C order"
        (Gen.with_pp pp_operands operands) (fun vs ->
          let cs = V.coalesce vs in
          let shape = V.shape (List.hd vs) in
          cover "fewer axes" (V.ndim (List.hd cs) < Array.length shape);
          cover "axes kept apart" (V.ndim (List.hd cs) > 1);
          cover "no element" (V.numel (List.hd vs) = 0);
          List.iter
            (fun c -> equal (array int) (V.shape (List.hd cs)) (V.shape c))
            cs;
          List.iter2
            (fun v c -> equal (list int) (positions v) (positions c))
            vs cs);
      prop "C-contiguous views coalesce to one axis at most"
        (Gen.with_pp Nx_test.pp_shape
           Gen.(array ~size:(int_range 0 4) (int_range 1 3)))
        (fun shape ->
          let cs = V.coalesce [ V.create shape; V.create ~offset:5 shape ] in
          at_most int ~than:1 (V.ndim (List.hd cs)));
      test "axes every view broadcasts merge" (fun () ->
          let v = V.expand (V.create [| 1; 1; 4 |]) [| 2; 3; 4 |] in
          equal (array int) [| 6; 4 |] (V.shape (List.hd (V.coalesce [ v ]))));
      test "views of different shapes are refused" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              ignore (V.coalesce [ V.create [| 2 |]; V.create [| 3 |] ])));
    ]

let () = exit (run "Nx_array.View" [ layout; bounds; reshape; coalesce ])
