(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The movements that test_values does not draw, stated through the ones it
   does, and the views the interface promises. *)

open Windtrap
open Nx_test

let ints = Ref.witness int32
let same = tensor int32
let shape = Gen.array ~size:(Gen.int_range 1 3) (Gen.int_range 1 4)
let iota s = Array.init (Ref.numel s) (fun i -> Int32.of_int (i + 1))
let tensor_of s = Nx.create Nx.int32 s (iota s)

(* A tensor and one of its axes. *)
let with_axis =
  let open Gen in
  let* s = shape in
  let+ axis = int_range 0 (Array.length s - 1) in
  (tensor_of s, axis)

let reorderings =
  group "reorderings"
    [
      prop "swapaxes is the transpose that exchanges two axes"
        (Gen.pair with_axis Gen.nat) (fun ((t, a), b) ->
          let n = Nx.ndim t in
          let b = b mod n in
          let axes =
            List.init n (fun i -> if i = a then b else if i = b then a else i)
          in
          equal same (Nx.transpose ~axes t) (Nx.swapaxes a b t));
      prop "moveaxis is the transpose that moves one axis"
        (Gen.pair with_axis Gen.nat) (fun ((t, src), dst) ->
          let n = Nx.ndim t in
          let dst = dst mod n in
          let rest = List.filter (( <> ) src) (List.init n Fun.id) in
          let axes =
            List.filteri (fun i _ -> i < dst) rest
            @ (src :: List.filteri (fun i _ -> i >= dst) rest)
          in
          equal same (Nx.transpose ~axes t) (Nx.moveaxis src dst t));
      prop "roll shifts along an axis, wrapping around"
        (Gen.pair with_axis (Gen.int_range (-9) 9))
        (fun ((t, axis), shift) ->
          let r = Ref.of_nx t in
          let n = r.shape.(axis) in
          equal ints
            (Ref.init r.shape (fun i ->
                 let src = Array.copy i in
                 src.(axis) <- (((i.(axis) - shift) mod n) + n) mod n;
                 Ref.get r src))
            (Ref.of_nx (Nx.roll ~axis shift t)));
      prop "roll without an axis rolls the flattened tensor and keeps the shape"
        (Gen.pair shape (Gen.int_range (-9) 9))
        (fun (s, shift) ->
          let t = tensor_of s in
          equal same
            (Nx.reshape s (Nx.roll ~axis:0 shift (Nx.reshape [| -1 |] t)))
            (Nx.roll shift t));
    ]

let repetitions =
  group "repetitions"
    [
      prop "tile repeats the whole tensor along each axis"
        (Gen.pair shape
           (Gen.array ~size:(Gen.int_range 3 4) (Gen.int_range 0 3)))
        (fun (s, reps) ->
          let r = Ref.of_nx (tensor_of s) in
          let n = Int.max (Array.length s) (Array.length reps) in
          let pad a = Array.append (Array.make (n - Array.length a) 1) a in
          let s' = pad s and reps' = pad reps in
          equal ints
            (Ref.init (Array.map2 ( * ) s' reps') (fun i ->
                 Ref.get (Ref.reshape s' r)
                   (Array.mapi (fun d k -> k mod s'.(d)) i)))
            (Ref.of_nx (Nx.tile reps (tensor_of s))));
      prop "repeat repeats each element along an axis"
        (Gen.pair with_axis (Gen.int_range 0 3))
        (fun ((t, axis), k) ->
          let r = Ref.of_nx t in
          let out = Array.copy r.shape in
          out.(axis) <- out.(axis) * k;
          equal ints
            (Ref.init out (fun i ->
                 let src = Array.copy i in
                 src.(axis) <- i.(axis) / k;
                 Ref.get r src))
            (Ref.of_nx (Nx.repeat ~axis k t)));
      test "tile refuses fewer repetitions than axes (nx.mli is silent)"
        (fun () ->
          raises_invalid_arg (fun () -> Nx.tile [| 2 |] (tensor_of [| 2; 2 |])));
      test "tile and repeat refuse a negative count" (fun () ->
          raises_invalid_arg (fun () -> Nx.tile [| -1 |] (tensor_of [| 2 |]));
          raises_invalid_arg (fun () -> Nx.repeat (-1) (tensor_of [| 2 |])));
    ]

let joins =
  group "joins and splits"
    [
      prop "stack is concatenate of each tensor with a new axis"
        (Gen.pair shape (Gen.int_range 0 3))
        (fun (s, axis) ->
          let axis = axis mod (Array.length s + 1) in
          let ts = [ tensor_of s; Nx.neg (tensor_of s) ] in
          equal same
            (Nx.concatenate ~axis (List.map (Nx.unsqueeze ~axes:[ axis ]) ts))
            (Nx.stack ~axis ts));
      prop "split into parts that concatenate back" (Gen.pair with_axis Gen.nat)
        (fun ((t, axis), k) ->
          let size = Nx.dim axis t in
          let divisors =
            List.filter
              (fun d -> size mod d = 0)
              (List.init size (fun i -> i + 1))
          in
          let n = List.nth divisors (k mod List.length divisors) in
          let parts = Nx.split ~axis n t in
          equal int n (List.length parts);
          equal same t (Nx.concatenate ~axis parts));
      prop "array_split by count gives the first parts the extra elements"
        (Gen.pair with_axis (Gen.int_range 1 5))
        (fun ((t, axis), n) ->
          let size = Nx.dim axis t in
          let parts = Nx.array_split ~axis (`Count n) t in
          equal (list int)
            (List.init n (fun i -> (size / n) + if i < size mod n then 1 else 0))
            (List.map (Nx.dim axis) parts);
          equal same t (Nx.concatenate ~axis parts));
      prop "array_split at indices cuts between them"
        (Gen.pair with_axis
           (Gen.list ~size:(Gen.int_range 0 3) (Gen.int_range 0 4)))
        (fun ((t, axis), cuts) ->
          let cuts =
            List.sort compare
              (List.map (fun c -> Int.min c (Nx.dim axis t)) cuts)
          in
          let bounds = List.combine (0 :: cuts) (cuts @ [ Nx.dim axis t ]) in
          let r = Ref.of_nx t in
          let expected =
            List.map
              (fun (lo, hi) ->
                Ref.shrink
                  (Array.mapi
                     (fun d n -> if d = axis then (lo, hi) else (0, n))
                     r.shape)
                  r)
              bounds
          in
          equal (list ints) expected
            (List.map Ref.of_nx (Nx.array_split ~axis (`Indices cuts) t)));
      prop
        "broadcasted broadcasts both tensors, in swapped order under ~reverse"
        (Gen.triple shape shape Gen.bool) (fun (a, b, reverse) ->
          let b = Array.mapi (fun i d -> if i mod 2 = 0 then 1 else d) b in
          let x = tensor_of a and y = Nx.neg (tensor_of b) in
          match Ref.broadcast_shapes a b with
          | exception Invalid_argument _ ->
              raises_invalid_arg (fun () -> Nx.broadcasted ~reverse x y)
          | s ->
              let x = Nx.broadcast_to s x and y = Nx.broadcast_to s y in
              equal (pair same same)
                (if reverse then (y, x) else (x, y))
                (Nx.broadcasted ~reverse (tensor_of a) (Nx.neg (tensor_of b))));
      prop "broadcast_arrays broadcasts every tensor to the common shape"
        (Gen.pair shape shape) (fun (a, b) ->
          let b = Array.mapi (fun i d -> if i mod 2 = 0 then 1 else d) b in
          let common =
            try Some (Ref.broadcast_shapes a b)
            with Invalid_argument _ -> None
          in
          match common with
          | None ->
              raises_invalid_arg (fun () ->
                  Nx.broadcast_arrays [ tensor_of a; tensor_of b ])
          | Some s ->
              equal (list same)
                [
                  Nx.broadcast_to s (tensor_of a);
                  Nx.broadcast_to s (tensor_of b);
                ]
                (Nx.broadcast_arrays [ tensor_of a; tensor_of b ]));
      test "stack, concatenate and split refuse what they cannot join or cut"
        (fun () ->
          raises_invalid_arg (fun () -> Nx.stack []);
          raises_invalid_arg (fun () -> Nx.concatenate ~axis:0 []);
          raises_invalid_arg (fun () ->
              Nx.stack [ tensor_of [| 2 |]; tensor_of [| 3 |] ]);
          raises_invalid_arg (fun () -> Nx.split ~axis:0 2 (tensor_of [| 3 |])));
      test "split refuses zero parts" (fun () ->
          raises_invalid_arg (fun () -> Nx.split ~axis:0 0 (tensor_of [| 4 |])));
      test
        "concatenate refuses an axis out of bounds, for one tensor too (nx.mli \
         is silent)" (fun () ->
          let v = tensor_of [| 4 |] in
          raises_invalid_arg (fun () -> Nx.concatenate ~axis:(-2) [ v; v ]);
          raises_invalid_arg (fun () -> Nx.concatenate ~axis:5 [ v ]);
          raises_invalid_arg (fun () ->
              Nx.concatenate ~axis:0 [ Nx.scalar Nx.int32 1l ]));
      test
        "array_split at a negative index counts it from the end, as a range \
         does (nx.mli is silent)" (fun () ->
          let v = tensor_of [| 4 |] in
          equal (list same)
            (Nx.array_split ~axis:0 (`Indices [ 3 ]) v)
            (Nx.array_split ~axis:0 (`Indices [ -1 ]) v));
    ]

let flattening =
  group "flattening"
    [
      prop "flatten merges a run of axes, and unflatten splits it back"
        (Gen.pair with_axis Gen.nat) (fun ((t, start_dim), k) ->
          let s = Nx.shape t in
          let end_dim = start_dim + (k mod (Array.length s - start_dim)) in
          let sizes = Array.sub s start_dim (end_dim - start_dim + 1) in
          let merged =
            Array.concat
              [
                Array.sub s 0 start_dim;
                [| Ref.numel sizes |];
                Array.sub s (end_dim + 1) (Array.length s - end_dim - 1);
              ]
          in
          equal same (Nx.reshape merged t) (Nx.flatten ~start_dim ~end_dim t);
          Law.round_trip same same
            (Nx.flatten ~start_dim ~end_dim)
            (Nx.unflatten start_dim sizes)
            t;
          let inferred =
            Array.mapi
              (fun i d -> if i = k mod Array.length sizes then -1 else d)
              sizes
          in
          Law.round_trip ~msg:"with one size inferred" same same
            (Nx.flatten ~start_dim ~end_dim)
            (Nx.unflatten start_dim inferred)
            t);
      prop "flatten works on every layout, a view where the layout allows one"
        (Gen.pair shape layout) (fun (s, steps) ->
          let t = lay_out steps (tensor_of s) in
          let target = [| Nx.numel t |] in
          let f = Nx.flatten t in
          equal ints (Ref.reshape target (Ref.of_nx t)) (Ref.of_nx f);
          if viewable t target then
            is_true ~msg:"a view" (storage f == storage t));
      prop "ravel is reshape to one axis" shape (fun s ->
          equal same
            (Nx.reshape [| -1 |] (tensor_of s))
            (Nx.ravel (tensor_of s)));
      test "unflatten refuses sizes whose product differs" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.unflatten 0 [| 2; 2 |] (tensor_of [| 6 |])));
      test "flatten refuses an axis out of bounds" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.flatten ~start_dim:2 (tensor_of [| 2; 2 |])));
      cases
        "flatten, and roll and repeat without an axis, read a transposed tensor"
        ~name:fst
        [
          ( "flatten",
            fun t ->
              (Ref.flatten ~start_dim:0 ~end_dim:(-1) t, fun t -> Nx.flatten t)
          );
          ("roll", fun t -> (Ref.roll 1 t, fun t -> Nx.roll 1 t));
          ("repeat", fun t -> (Ref.repeat 2 t, fun t -> Nx.repeat 2 t));
        ]
        (fun (_, op) ->
          let t = Nx.transpose (tensor_of [| 2; 3 |]) in
          let expected, f = op (Ref.of_nx t) in
          equal ints expected (Ref.of_nx (f t)));
    ]

(* A tensor under some layout, and a shape of as many elements. *)
let reshaped =
  let open Gen in
  let* s = shape in
  let* steps = layout in
  let t = lay_out steps (tensor_of s) in
  let n = Nx.numel t in
  let+ target =
    of_list ~pp:pp_shape
      ([ [| n |]; [| 1; n |]; [| n; 1 |] ]
      @ List.filter_map
          (fun d -> if d > 0 && n mod d = 0 then Some [| d; n / d |] else None)
          (List.init (n + 1) Fun.id)
      @ List.filter_map
          (fun d ->
            if d > 0 && n mod d = 0 && n / d mod 2 = 0 then
              Some [| d; 2; n / d / 2 |]
            else None)
          (List.init (n + 1) Fun.id))
  in
  (steps, t, target)

let reshapes =
  group "reshapes"
    [
      prop "reshape views every layout it can, and copies the others"
        (Gen.with_pp
           (fun ppf (steps, t, target) ->
             Format.fprintf ppf "%a, of shape %a, to %a" pp_layout steps
               pp_shape (Nx.shape t) pp_shape target)
           reshaped)
        (fun (_, t, target) ->
          if viewable t target then begin
            cover "a view of a non-contiguous layout"
              (not (Nx.is_c_contiguous t));
            let r = Nx.reshape target t in
            equal ints (Ref.reshape target (Ref.of_nx t)) (Ref.of_nx r);
            is_true ~msg:"the result shares its source's storage"
              (storage r == storage t)
          end
          else begin
            cover "a layout no reshape can view" true;
            equal ints
              (Ref.reshape target (Ref.of_nx t))
              (Ref.of_nx (Nx.reshape target t))
          end);
    ]

(* 32 axes, the most a tensor has. *)
let high_rank =
  group "high rank"
    [
      test "a tensor of 32 axes adds, transposes and sums as one of few"
        (fun () ->
          let shape = Array.init 32 (fun i -> if i < 29 then 1 else 2) in
          let t = tensor_of shape in
          let r = Ref.of_nx t in
          equal ints (Ref.map2 Int32.add r r) (Ref.of_nx (Nx.add t t));
          equal ints (Ref.transpose r) (Ref.of_nx (Nx.transpose t));
          equal ints
            (Ref.reduce ~axes:[ 31; 30 ] Int32.add 0l r)
            (Ref.of_nx (Nx.sum ~axes:[ 31; 30 ] t)));
    ]

(* The interface promises these results share their source's storage. *)
let views =
  let shares name f =
    test (name ^ " is a view of its source") (fun () ->
        let t = tensor_of [| 4; 6 |] in
        is_true (storage (f t) == storage t))
  in
  group "views"
    [
      shares "reshape of a contiguous tensor" (Nx.reshape [| 6; 4 |]);
      shares "transpose" (fun t -> Nx.transpose t);
      shares "flip" (fun t -> Nx.flip t);
      shares "broadcast_to" (fun t -> Nx.broadcast_to [| 2; 4; 6 |] t);
      shares "slice by a range" (Nx.slice [ R (1, 3) ]);
      shares "slice by an index, the whole axis and a new axis"
        (Nx.slice [ I 1; A; N ]);
      shares "slice by a step of 1" (Nx.slice [ A; Rs (1, 5, 1) ]);
      shares "get" (Nx.get [ 2 ]);
      shares "shrink" (Nx.slice [ Nx.R (1, 3); Nx.R (0, 6) ]);
      shares "expand" (fun t -> Nx.expand [| 2; -1; -1 |] t);
      shares "swapaxes" (Nx.swapaxes 0 1);
      shares "unflatten" (Nx.unflatten 1 [| 2; 3 |]);
      shares "unsqueeze and squeeze" (fun t ->
          Nx.squeeze (Nx.unsqueeze ~axes:[ 0 ] t));
      shares "slice by a step of -1" (Nx.slice [ A; Rs (5, 0, -1) ]);
      shares "a part of split" (fun t -> List.nth (Nx.split ~axis:1 3 t) 1);
      shares "sliding_window" (Nx.sliding_window ~window:2);
      shares "broadcast_arrays" (fun t ->
          List.hd (Nx.broadcast_arrays [ t; Nx.zeros Nx.int32 [| 2; 1; 1 |] ]));
      test "a scalar expanded by a movement reshapes over its one element"
        (fun () ->
          let e = Nx.Op.eval (Move (Nx.scalar Nx.int32 7l, Expand [| 3 |])) in
          equal same
            (Nx.full Nx.int32 [| 3; 1 |] 7l)
            (Nx.Op.eval (Move (e, Reshape [| 3; 1 |]))));
      test "copy and concatenate never share" (fun () ->
          let t = tensor_of [| 2; 3 |] in
          is_false (share_memory (storage (Nx.copy t)) (storage t));
          is_false
            (share_memory (storage (Nx.concatenate ~axis:0 [ t ])) (storage t)));
      test "a range past its axis or ending before it starts is cut to the axis"
        (fun () ->
          equal (array int) [| 4 |]
            (Nx.shape (Nx.slice [ Nx.R (0, 5) ] (tensor_of [| 4 |])));
          equal (array int) [| 0 |]
            (Nx.shape (Nx.slice [ Nx.R (3, 1) ] (tensor_of [| 4 |]))));
      test "ravel copies a tensor it cannot view flat" (fun () ->
          let t = Nx.transpose (tensor_of [| 2; 3 |]) in
          equal ints
            (Ref.reshape [| 6 |] (Ref.of_nx t))
            (Ref.of_nx (Nx.ravel t)));
    ]

(* Two 4-bit elements share a byte, so a view of them rarely starts, strides or
   ends on one, and a movement writes elements whose byte it shares with
   others. *)
type movement = { name : string; move : 'b. (int, 'b) Nx.t -> (int, 'b) Nx.t }

let movements =
  let first t = if Nx.ndim t = 0 then 0 else Nx.dim 0 t in
  let along_first f t = if first t = 0 then t else f (first t) t in
  [
    { name = "as it is"; move = Fun.id };
    {
      name = "taken at its last, first, first, past-the-end and -1 positions";
      move =
        (fun t ->
          let n = Nx.numel t in
          Nx.take
            ~indices:
              (Nx.create Nx.int64 [| 5 |]
                 (Array.map Int64.of_int [| n - 1; 0; 0; n; -1 |]))
            t);
    };
    {
      name = "every other row";
      move = (fun t -> along_first (fun n -> Nx.slice [ Rs (0, n, 2) ]) t);
    };
    {
      name = "its last and first rows listed";
      move = (fun t -> along_first (fun n -> Nx.slice [ L [ n - 1; 0 ] ]) t);
    };
    {
      name = "padded by one on every axis with 5";
      move = (fun t -> Nx.pad (Array.make (Nx.ndim t) (1, 1)) 5 t);
    };
    {
      name = "concatenated to itself";
      move =
        (fun t -> along_first (fun _ t -> Nx.concatenate ~axis:0 [ t; t ]) t);
    };
    {
      name = "in patches of two along its last axis, padded before";
      move =
        (fun t ->
          if Nx.ndim t = 0 || Nx.dim (-1) t = 0 then t
          else
            Nx.extract_patches ~kernel_size:[| 2 |] ~stride:[| 1 |]
              ~dilation:[| 1 |]
              ~padding:[| (1, 0) |]
              t);
    };
    {
      name = "with its first row set to 3";
      move =
        (fun t ->
          along_first
            (fun _ t -> Nx.set [ R (0, 1) ] (Nx.full (Nx.dtype t) [||] 3) t)
            t);
    };
    {
      name = "with 7 scattered into its first row";
      move =
        (fun t ->
          along_first
            (fun _ t ->
              let s = Array.copy (Nx.shape t) in
              s.(0) <- 1;
              Nx.scatter ~axis:0 ~indices:(Nx.zeros Nx.int64 s)
                ~values:(Nx.full (Nx.dtype t) [||] 7)
                t)
            t);
    };
  ]

let packed =
  let moves_as_int8 name (dtype : (int, _) Nx.dtype) lo hi =
    let drawn =
      let open Gen in
      let* s = array ~size:(int_range 0 3) (int_range 0 4) in
      let* steps = layout in
      let* m = of_list movements in
      let+ xs = array ~size:(constant (Ref.numel s)) (int_range lo hi) in
      (s, steps, m, xs)
    in
    let pp ppf (s, steps, m, xs) =
      Format.fprintf ppf "%a %a, %s: %a" pp_shape s pp_layout steps m.name
        Format.(pp_print_list ~pp_sep:pp_print_space pp_print_int)
        (Array.to_list xs)
    in
    prop (name ^ " values move under every layout as their values at int8 do")
      (Gen.with_pp pp drawn) (fun (s, steps, m, xs) ->
        let wide = Nx.create Nx.int8 s xs in
        equal (array int)
          (Nx.to_array (m.move (lay_out steps wide)))
          (Nx.to_array (m.move (lay_out steps (Nx.cast dtype wide)))))
  in
  group "packed"
    [ moves_as_int8 "int4" Nx.int4 (-8) 7; moves_as_int8 "uint4" Nx.uint4 0 15 ]

let () =
  exit
    (run "nx movement"
       [
         high_rank;
         reshapes;
         reorderings;
         repetitions;
         joins;
         flattening;
         views;
         packed;
       ])
