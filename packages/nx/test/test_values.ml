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

(* Every live tensor equals its model, and its view's offset and strides locate
   each element in its storage. *)
let t =
  abstract "t"
    ~pp:(Ref.pp (fun ppf v -> Format.fprintf ppf "%ld" v))
    ~invariant:(fun r s ->
      equal elt r (Ref.of_nx s);
      equal ~msg:"offset and strides locate each element" elt r
        (Ref.of_layout s))

let shape = Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 4)
let base = Gen.int_range 0 9
let value = Gen.map Int32.of_int (Gen.int_range (-9) 9)

let iota base shape =
  Array.init (Ref.numel shape) (fun i -> Int32.of_int ((base * 100) + i))

let int64s l =
  Nx.create Nx.int64
    [| List.length l |]
    (Array.of_list (List.map Int64.of_int l))

let at k = Nx.scalar Nx.int64 (Int64.of_int k)

(* Arguments listed from the tensor they apply to, the simplest first. Each list
   holds at least one argument the API refuses. *)

let arg pp = Testable.make ~pp ~equal:( == )

let pp_list pp =
  Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ") pp

let pp_ints ppf l = Format.fprintf ppf "[%a]" (pp_list Format.pp_print_int) l

let pp_axes ppf = function
  | None -> Format.pp_print_string ppf "all"
  | Some l -> pp_ints ppf l

let pp_axis ppf = function
  | None -> Format.pp_print_string ppf "flattened"
  | Some a -> Format.fprintf ppf "~axis:%d" a

let pp_pairs ppf a =
  Format.fprintf ppf "[%a]"
    (pp_list (fun ppf (x, y) -> Format.fprintf ppf "(%d, %d)" x y))
    (Array.to_list a)

let pp_pair ppf (a, b) = Format.fprintf ppf "%d %d" a b

let pp_spec ppf = function
  | `Count n -> Format.fprintf ppf "`Count %d" n
  | `Indices l -> Format.fprintf ppf "`Indices %a" pp_ints l

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

(* A range whose start lies outside the axis selects from the nearest end, as
   [R] does. *)
let per_axis dim =
  let always : Nx.index list =
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
      M (Nx.full Nx.bool [| dim + 1 |] true);
      M (Nx.full Nx.bool [| 1; dim |] true);
      D (at 1, Int.max 0 (dim - 1));
      D (at (-5), Int.min dim 1);
      D (at (dim + 5), Int.min dim 2);
      D (at 0, dim + 1);
    ]
  in
  always
  @ [
      Nx.Rs (-dim - 3, dim, 2);
      Nx.Rs (dim + 2, -dim - 3, -2);
      Nx.Rs (-dim - 3, dim, 1);
      Nx.Rs (dim + 2, 0, -1);
    ]

(* Index lists that mix forms across axes, where a gather meets a new axis, a
   mask or a window. *)
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
        [ D (at 9, Int.min d0 1); L [ 0 ] ];
        [ N; D (at 0, Int.min d0 2); I 0 ];
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

(* Values for a selection: its own shape, its last axis alone, or its shape with
   every other axis of size one. *)
let values_for =
  Gen.with_pp
    (fun ppf form ->
      Format.pp_print_string ppf
        (match form with
        | 0 -> "shaped like the selection"
        | 1 -> "shaped like its last axis"
        | _ -> "broadcast along every other axis"))
    (Gen.int_range 0 2)

let values form (sel : int32 Ref.t) =
  let n = Ref.ndim sel in
  let shape =
    match form with
    | 0 -> sel.shape
    | 1 -> if n = 0 then [||] else [| sel.shape.(n - 1) |]
    | _ -> Array.mapi (fun d k -> if d mod 2 = 0 then 1 else k) sel.shape
  in
  Ref.init shape (fun idx -> Int32.of_int (1000 + Ref.ravel shape idx))

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

let expansions =
  among (arg pp_shape) t (fun r ->
      let s = r.shape in
      [
        Array.map (fun _ -> -1) s;
        Array.append [| 2 |] (Array.map (fun _ -> -1) s);
        Array.map (fun d -> if d = 1 then 3 else -1) s;
        Array.map (fun d -> if d = 1 then 0 else d) s;
        Array.map (fun d -> d + 1) s;
        Array.map (fun _ -> -2) s;
        [| -2 |];
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

(* Pairs of axes, and one out of bounds. *)
let axis_pairs =
  among (arg pp_pair) t (fun r ->
      let n = Ref.ndim r in
      List.concat_map
        (fun a -> List.map (fun b -> (a, b)) (axes_of r))
        (axes_of r)
      @ [ (n, 0); (0, n) ])

let rolls =
  among
    (arg (fun ppf (a, k) -> Format.fprintf ppf "%a by %d" pp_axis a k))
    t
    (fun r ->
      List.concat_map
        (fun a -> [ (Some a, 1); (Some a, -2); (Some a, 7) ])
        (axes_of r)
      @ [ (Some (Ref.ndim r), 1) ]
      @ [ (None, 1); (None, -3) ])

(* nx.mli is silent on fewer repetitions than axes: they are refused. *)
let repetitions =
  among (arg pp_shape) t (fun r ->
      let n = Ref.ndim r in
      [
        Array.make n 1;
        Array.make n 2;
        Array.make (n + 1) 2;
        Array.init n (fun i -> if i = 0 then 0 else 1);
        Array.init n (fun i -> if i = n - 1 then -1 else 1);
        Array.make (Int.max 0 (n - 1)) 2;
      ])

let repeats =
  among
    (arg (fun ppf (a, k) -> Format.fprintf ppf "%d times %a" k pp_axis a))
    t
    (fun r ->
      List.concat_map
        (fun a -> [ (Some a, 2); (Some a, 0); (Some a, -1) ])
        (axes_of r)
      @ [ (Some (Ref.ndim r), 2) ]
      @ [ (None, 2) ])

(* nx.mli states no error for [take] on an axis out of bounds, so only axes of
   the tensor are listed. *)
let takes =
  among
    (arg (fun ppf (a, l) -> Format.fprintf ppf "%a at %a" pp_axis a pp_ints l))
    t
    (fun r ->
      List.concat_map
        (fun a ->
          let dim = r.shape.(a) in
          [ (Some a, [ dim - 1; 0; -1; dim ]); (Some a, []) ])
        (axes_of r)
      @
      let n = Ref.numel r.shape in
      [ (None, [ n - 1; 0; n; -1 ]) ])

let conditions =
  among
    (arg (fun ppf (a, l) ->
         Format.fprintf ppf "%a where %a" pp_axis a
           (pp_list Format.pp_print_bool)
           l))
    t
    (fun r ->
      List.concat_map
        (fun a ->
          let dim = r.shape.(a) in
          [
            (Some a, List.init dim (fun i -> i mod 3 <> 1));
            (Some a, List.init dim (fun _ -> false));
            (Some a, List.init (dim + 1) (fun i -> i = dim));
          ])
        (axes_of r)
      @ [ (None, List.init (Ref.numel r.shape) (fun i -> i mod 2 = 0)) ])

let stack_axes =
  among (arg Format.pp_print_int) t (fun r ->
      let n = Ref.ndim r in
      [ 0; n; -1; n + 2 ])

let array_splits =
  among
    (arg (fun ppf (a, spec, part) ->
         Format.fprintf ppf "part %d of ~axis:%d %a" part a pp_spec spec))
    t
    (fun r ->
      List.concat_map
        (fun a ->
          let dim = r.shape.(a) in
          [
            (a, `Count 1, 0);
            (a, `Count 2, 1);
            (a, `Count (dim + 1), dim);
            (a, `Indices [ 1 ], 0);
            (a, `Indices [ 1 ], 1);
            (a, `Indices [ dim + 2 ], 1);
            (a, `Indices [ 2; 1 ], 1);
            (a, `Indices [], 0);
            (a, `Count 0, 0);
          ])
        (axes_of r))

let splits =
  among
    (arg (fun ppf (a, n, part) ->
         Format.fprintf ppf "part %d of %d along %d" part n a))
    t
    (fun r ->
      List.concat_map
        (fun a -> [ (a, 1, 0); (a, 2, 1); (a, 3, 0) ] @ [ (a, 0, 0) ])
        (axes_of r))

let flattenings =
  among (arg pp_pair) t (fun r ->
      let n = Ref.ndim r in
      if n = 0 then []
      else [ (0, -1); (0, 0); (-1, -1); (0, n - 1); (1, -1); (n, -1) ])

let unflattenings =
  among
    (arg (fun ppf (d, s) -> Format.fprintf ppf "%d %a" d pp_shape s))
    t
    (fun r ->
      List.concat_map
        (fun a ->
          let dim = r.shape.(a) in
          [
            (a, [| dim |]);
            (a, [| 1; dim |]);
            (a, [| -1; 1 |]);
            (a, [| dim; -1 |]);
            (a, [| 2; -1 |]);
            (a, [| dim + 1 |]);
          ])
        (axes_of r)
      @ [ (Ref.ndim r, [| 1 |]) ])

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
  among (arg Format.pp_print_int) t (fun r ->
      [ 0; -1; Ref.ndim r ] @ [ -Ref.ndim r - 1 ])

let positions =
  among (arg pp_ints) t (fun r ->
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
    command "moveaxis"
      (axis_pairs ^-> t ^-> makes t)
      (fun (a, b) r -> Ref.moveaxis a b r)
      (fun (a, b) s -> Nx.moveaxis a b s);
    command "swapaxes"
      (axis_pairs ^-> t ^-> makes t)
      (fun (a, b) r -> Ref.swapaxes a b r)
      (fun (a, b) s -> Nx.swapaxes a b s);
    command "slice" (specs ^-> t ^-> makes t) Ref.slice Nx.slice;
    command "set"
      (t ^-> specs ^-> t ^-> makes t)
      (fun r specs v -> Ref.set specs v r)
      (fun s specs v -> Nx.set specs v s);
    command "set a scalar"
      (t ^-> specs ^-> value @-> makes t)
      (fun r specs v -> Ref.set specs (Ref.create [||] [| v |]) r)
      (fun s specs v -> Nx.set specs (Nx.scalar Nx.int32 v) s);
    command "set values"
      (t ^-> specs ^-> values_for @-> makes t)
      (fun r specs form -> Ref.set specs (values form (Ref.slice specs r)) r)
      (fun s specs form ->
        let v = values form (Ref.slice specs (Ref.of_nx s)) in
        Nx.set specs (Nx.create Nx.int32 v.shape v.data) s);
    command "fill" (value @-> t ^-> makes t) Ref.fill Nx.fill;
    command "broadcast_to"
      (broadcasts ^-> t ^-> makes t)
      Ref.broadcast_to Nx.broadcast_to;
    command "expand" (expansions ^-> t ^-> makes t) Ref.expand Nx.expand;
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
    command "unflatten"
      (unflattenings ^-> t ^-> makes t)
      (fun (d, sizes) r -> Ref.unflatten d sizes r)
      (fun (d, sizes) s -> Nx.unflatten d sizes s);
    command "roll"
      (rolls ^-> t ^-> makes t)
      (fun (axis, k) r -> Ref.roll ?axis k r)
      (fun (axis, k) s -> Nx.roll ?axis k s);
    command "tile" (repetitions ^-> t ^-> makes t) Ref.tile Nx.tile;
    command "repeat"
      (repeats ^-> t ^-> makes t)
      (fun (axis, k) r -> Ref.repeat ?axis k r)
      (fun (axis, k) s -> Nx.repeat ?axis k s);
    command "take"
      (takes ^-> t ^-> makes t)
      (fun (axis, l) r -> Ref.take ?axis ~zero:0l (Array.of_list l) r)
      (fun (axis, l) s -> Nx.take ?axis ~indices:(int64s l) s);
    command "compress"
      (conditions ^-> t ^-> makes t)
      (fun (axis, l) r -> Ref.compress ?axis (Array.of_list l) r)
      (fun (axis, l) s ->
        Nx.compress ?axis
          ~condition:(Nx.create Nx.bool [| List.length l |] (Array.of_list l))
          s);
    command "sliding_window"
      (windows ^-> t ^-> makes t)
      (fun (axis, window, step) r -> Ref.sliding_window ~axis ~window ~step r)
      (fun (axis, window, step) s -> Nx.sliding_window ~axis ~window ~step s);
    command "concatenate"
      (t ^-> concat_axes ^-> t ^-> makes t)
      (fun a axis b -> Ref.concatenate ~axis [ a; b ])
      (fun a axis b -> Nx.concatenate ~axis [ a; b ]);
    command "stack a tensor with itself"
      (stack_axes ^-> t ^-> makes t)
      (fun axis r -> Ref.stack ~axis [ r; r ])
      (fun axis s -> Nx.stack ~axis [ s; s ]);
    command "a part of array_split"
      (array_splits ^-> t ^-> makes t)
      (fun (axis, spec, part) r -> List.nth (Ref.array_split ~axis spec r) part)
      (fun (axis, spec, part) s -> List.nth (Nx.array_split ~axis spec s) part);
    command "a part of split"
      (splits ^-> t ^-> makes t)
      (fun (axis, n, part) r -> List.nth (Ref.split ~axis n r) part)
      (fun (axis, n, part) s -> List.nth (Nx.split ~axis n s) part);
    command "add" (t ^-> t ^-> makes t) (Ref.map2 Int32.add) Nx.add;
    command "contiguous" (t ^-> makes t) Fun.id Nx.contiguous;
    command "copy" (t ^-> makes t) Fun.id Nx.copy;
    command "get"
      (positions ^-> t ^-> makes t)
      (fun l r -> Ref.slice (List.map (fun i -> Nx.I i) l) r)
      Nx.get;
    command "item" (positions ^-> t ^-> returns int32) Ref.item Nx.item;
  ]
  @ [
      command "flatten"
        (flattenings ^-> t ^-> makes t)
        (fun (start_dim, end_dim) r -> Ref.flatten ~start_dim ~end_dim r)
        (fun (start_dim, end_dim) s -> Nx.flatten ~start_dim ~end_dim s);
      command "concatenate one tensor"
        (concat_axes ^-> t ^-> makes t)
        (fun axis r -> Ref.concatenate ~axis [ r ])
        (fun axis s -> Nx.concatenate ~axis [ s ]);
    ]

let () =
  exit (run "nx values" [ stateful ~count:300 "tensors are values" commands ])
