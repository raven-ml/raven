(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tensors are values: a program of movements, selections and writes against the
   reference model, with every live tensor checked after every call, so a write
   that reaches a tensor through shared storage fails at the call that made
   it. *)

open Windtrap
open Nx_test

let elt = Ref.witness int32

(* Every live tensor equals its model, and its buffer, offset and strides locate
   each element as [Nx.data] documents. *)
let t =
  abstract "t"
    ~pp:(Ref.pp (fun ppf v -> Format.fprintf ppf "%ld" v))
    ~invariant:(fun r s ->
      equal elt r (Ref.of_nx s);
      equal ~msg:"data, offset and strides locate each element" elt r
        (Ref.of_layout s))

let shape = Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 4)
let base = Gen.int_range 0 9
let value = Gen.map Int32.of_int (Gen.int_range (-9) 9)

let iota base shape =
  Array.init (Ref.numel shape) (fun i -> Int32.of_int ((base * 100) + i))

(* Arguments listed from the tensor they apply to, the simplest first. Each list
   holds at least one argument the API refuses. *)

let arg pp = Testable.make ~pp ~equal:( == )

let pp_list pp =
  Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ") pp

let pp_axes ppf = function
  | None -> Format.pp_print_string ppf "all"
  | Some l -> Format.fprintf ppf "[%a]" (pp_list Format.pp_print_int) l

let pp_pairs ppf a =
  Format.fprintf ppf "[%a]"
    (pp_list (fun ppf (x, y) -> Format.fprintf ppf "(%d, %d)" x y))
    (Array.to_list a)

let axes_of r = List.init (Ref.ndim r) Fun.id

let rec permutations = function
  | [] -> [ [] ]
  | l ->
      List.concat_map
        (fun x ->
          List.map (List.cons x) (permutations (List.filter (( <> ) x) l)))
        l

let reshapes =
  among (arg pp_shape) t (fun r ->
      let n = Ref.numel r.shape in
      let pairs =
        List.filter_map
          (fun d -> if d > 0 && n mod d = 0 then Some [| d; n / d |] else None)
          (List.init (n + 1) Fun.id)
      in
      [ [| n |]; [| -1 |] ] @ pairs
      @ [ [| 1; n; 1 |]; [| 2; -1 |]; [| 0; 2 |]; [| n + 1 |]; [| -1; -1 |] ])

let permutations_of =
  among (arg pp_axes) t (fun r ->
      let n = Ref.ndim r in
      (None :: List.map Option.some (permutations (axes_of r)))
      @ [ Some [ n ]; Some (List.init (n + 1) (fun _ -> 0)) ])

let flip_axes =
  among (arg pp_axes) t (fun r ->
      (None :: List.map (fun a -> Some [ a ]) (axes_of r))
      @ [ Some [ -1 ]; Some [ Ref.ndim r ] ])

let per_axis dim : Nx.index list =
  [
    A;
    I 0;
    I (-1);
    R (1, dim);
    R (-2, dim + 3);
    Rs (0, dim, 2);
    Rs (dim - 1, -dim - 1, -1);
    Rs (-1, 0, -2);
    L [ dim - 1; 0 ];
    M (Nx.create Nx.bool [| dim |] (Array.init dim (fun i -> i mod 2 = 0)));
    I dim;
    Rs (0, dim, 0);
  ]

(* Index lists that mix forms across axes, where a gather meets a new axis or a
   mask. *)
let combined shape : Nx.index list list =
  match Array.to_list shape with
  | d0 :: d1 :: _ ->
      let mask = List.nth (per_axis d0) 9 in
      [
        [ L [ d0 - 1; 0 ]; Rs (d1 - 1, -d1 - 1, -1) ];
        [ N; L [ 0 ] ];
        [ L [ 0 ]; N ];
        [ I (-1); N; R (0, d1) ];
        [ mask; I 0 ];
      ]
  | [ d0 ] -> [ [ N; L [ d0 - 1 ] ]; [ L [ 0 ]; N ] ]
  | [] -> []

let specs =
  among (arg pp_specs) t (fun r ->
      let n = Ref.ndim r in
      let one_axis =
        List.concat
          (List.mapi
             (fun d dim ->
               List.map
                 (fun s -> List.init d (fun _ -> Nx.A) @ [ s ])
                 (per_axis dim))
             (Array.to_list r.shape))
      in
      ([] :: [ Nx.N ] :: one_axis)
      @ [
          Array.to_list
            (Array.map (fun dim -> Nx.Rs (-1, -dim - 1, -1)) r.shape);
          List.init (n + 1) (fun _ -> Nx.A);
        ]
      @ combined r.shape)

let broadcasts =
  among (arg pp_shape) t (fun r ->
      let s = r.shape in
      [
        s;
        Array.append [| 2 |] s;
        Array.map (fun d -> if d = 1 then 3 else d) s;
        Array.map (fun d -> d + 1) s;
        [||];
      ])

let widths =
  among (arg pp_pairs) t (fun r ->
      let n = Ref.ndim r in
      let zeros = Array.make n (0, 0) in
      let at d w = Array.init n (fun k -> if k = d then w else (0, 0)) in
      zeros
      :: List.concat_map (fun d -> [ at d (1, 0); at d (0, 2) ]) (axes_of r)
      @ [ Array.make (n + 1) (0, 0); Array.make n (-1, 0) ])

(* nx.mli states no error for [shrink], so only ranges inside the axes are
   listed. *)
let ranges =
  among (arg pp_pairs) t (fun r ->
      let n = Ref.ndim r in
      let at d range =
        Array.init n (fun k -> if k = d then range else (0, r.shape.(k)))
      in
      Array.map (fun d -> (0, d)) r.shape
      :: List.concat_map
           (fun d ->
             let dim = r.shape.(d) in
             if dim = 0 then []
             else [ at d (1, dim); at d (0, dim - 1); at d (1, 1) ])
           (axes_of r))

let squeezes =
  among (arg pp_axes) t (fun r ->
      (None :: List.map (fun a -> Some [ a ]) (axes_of r))
      @ [ Some [ Ref.ndim r ] ])

let unsqueezes =
  among (arg pp_axes) t (fun r ->
      let n = Ref.ndim r in
      [
        Some [ 0 ];
        Some [ n ];
        Some [ -1 ];
        Some [ 0; n + 1 ];
        Some [ 0; 0 ];
        Some [ n + 2 ];
        None;
      ])

let windows =
  among
    (arg (fun ppf (a, w, s) ->
         Format.fprintf ppf "~axis:%d ~window:%d ~step:%d" a w s))
    t
    (fun r ->
      let last = if Ref.ndim r = 0 then 0 else r.shape.(Ref.ndim r - 1) in
      [
        (-1, 1, 1);
        (-1, 2, 1);
        (0, 2, 2);
        (-1, last, 3);
        (-1, last + 1, 1);
        (-1, 0, 1);
        (-1, 1, 0);
      ])

let concat_axes =
  among (arg Format.pp_print_int) t (fun r -> [ 0; -1; Ref.ndim r ])

let positions =
  among
    (arg (fun ppf l ->
         Format.fprintf ppf "[%a]" (pp_list Format.pp_print_int) l))
    t
    (fun r ->
      let n = Ref.ndim r in
      [
        List.init n (fun _ -> 0);
        List.init n (fun _ -> -1);
        Array.to_list (Array.map (fun d -> d - 1) r.shape);
        Array.to_list r.shape;
        List.init (n + 1) (fun _ -> 0);
      ])

let commands =
  [
    command "create"
      (base @-> shape @-> makes t)
      (fun b s -> Ref.create s (iota b s))
      (fun b s -> Nx.create Nx.int32 s (iota b s));
    command "reshape"
      (reshapes ^-> t ^-> makes t)
      Ref.reshape
      (fun shape s -> Nx.reshape shape (Nx.contiguous s));
    (* A reshape is a view, so nx refuses a layout it cannot view. *)
    command "reshape a view"
      (reshapes ^-> t ^-> judges elt)
      (fun shape r -> function
        | Ok v -> (
            match Ref.reshape shape r with
            | expected -> equal elt expected v
            | exception Invalid_argument _ ->
                fail "reshaped to a shape of another size")
        | Error (Invalid_argument _) -> ()
        | Error e -> raise e)
      (fun shape s -> Ref.of_nx (Nx.reshape shape s));
    command "transpose"
      (permutations_of ^-> t ^-> makes t)
      (fun axes r -> Ref.transpose ?axes r)
      (fun axes s -> Nx.transpose ?axes s);
    command "flip"
      (flip_axes ^-> t ^-> makes t)
      (fun axes r -> Ref.flip ?axes r)
      (fun axes s -> Nx.flip ?axes s);
    command "slice" (specs ^-> t ^-> makes t) Ref.slice Nx.slice;
    command "set"
      (t ^-> specs ^-> t ^-> makes t)
      (fun r specs v -> Ref.set specs v r)
      (fun s specs v -> Nx.set specs v s);
    command "set a scalar"
      (t ^-> specs ^-> value @-> makes t)
      (fun r specs v -> Ref.set specs (Ref.create [||] [| v |]) r)
      (fun s specs v -> Nx.set specs (Nx.scalar Nx.int32 v) s);
    command "fill" (value @-> t ^-> makes t) Ref.fill Nx.fill;
    command "broadcast_to"
      (broadcasts ^-> t ^-> makes t)
      Ref.broadcast_to Nx.broadcast_to;
    command "pad" (widths ^-> value @-> t ^-> makes t) Ref.pad Nx.pad;
    command "shrink" (ranges ^-> t ^-> makes t) Ref.shrink Nx.shrink;
    command "squeeze"
      (squeezes ^-> t ^-> makes t)
      (fun axes r -> Ref.squeeze ?axes r)
      (fun axes s -> Nx.squeeze ?axes s);
    command "unsqueeze"
      (unsqueezes ^-> t ^-> makes t)
      (fun axes r -> Ref.unsqueeze ?axes r)
      (fun axes s -> Nx.unsqueeze ?axes s);
    command "sliding_window"
      (windows ^-> t ^-> makes t)
      (fun (axis, window, step) r -> Ref.sliding_window ~axis ~window ~step r)
      (fun (axis, window, step) s -> Nx.sliding_window ~axis ~window ~step s);
    command "concatenate"
      (t ^-> concat_axes ^-> t ^-> makes t)
      (fun a axis b -> Ref.concatenate ~axis [ a; b ])
      (fun a axis b -> Nx.concatenate ~axis [ a; b ]);
    command "add" (t ^-> t ^-> makes t) (Ref.map2 Int32.add) Nx.add;
    command "contiguous" (t ^-> makes t) Fun.id Nx.contiguous;
    command "copy" (t ^-> makes t) Fun.id Nx.copy;
    command "get"
      (positions ^-> t ^-> makes t)
      (fun l r -> Ref.slice (List.map (fun i -> Nx.I i) l) r)
      Nx.get;
    command "item" (positions ^-> t ^-> returns int32) Ref.item Nx.item;
  ]

let () =
  exit (run "nx values" [ stateful ~count:300 "tensors are values" commands ])
