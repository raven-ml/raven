(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The layout a view reports is that of its shape, strides and offset, whatever
   movements made it. *)

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

let () = exit (run "Nx_array.View" [ layout ])
