(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.mli's Shapes, broadcasting and movements: each function against a
   reference that reads an OCaml array at mapped indices, over operands laid out
   in C order, reversed, with their axes reversed, at every other element and
   broadcast; the round trips; the views and copies; the extensions; every
   dtype; the raises. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move

let elements = Nx.to_array
let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

(* A raise in nx's words: the message starts with the function the user called
   and names no function of the layers below it. *)
let named ~by f =
  match f () with
  | () -> failf "%s raised nothing" by
  | exception Invalid_argument m ->
      starts_with ~prefix:(by ^ ": ") m;
      List.iter
        (fun sub -> not_contains ~sub m)
        [ "Nx_array"; "Nx_kernel"; "Layout."; "Move."; "Prog." ]

(* [by]'s raises, each row a reason and a call. *)
let raises_naming by rows =
  cases "raises naming itself" ~id:fst rows (fun (_, f) -> named ~by f)

let r reason f = (reason, fun () -> ignore (f ()))

(* The reference: a shape and its elements in C order. *)

let numel s = Array.fold_left ( * ) 1 s

let position s idx =
  let k = ref 0 in
  Array.iteri (fun a i -> k := (!k * s.(a)) + i) idx;
  !k

(* The elements of shape [s] whose element at [i] is [f i]. *)
let init s f = Array.init (numel s) (fun k -> f (index s k))
let rev_axes r = Array.init r (fun i -> r - 1 - i)

(* A drawn operand: its shape, layout, the elements it reads (distinct unless
   broadcast) and the value. *)
type operand = {
  s : int array;
  l : layout;
  e : int32 array;
  x : (int32, D.int32_elt, Nx.host) Nx.t;
}

let operand_of s l =
  let data = Array.init (numel s) (fun k -> Int32.of_int (k + 1)) in
  let x, e = laid l D.Int32 s data in
  { s; l; e; x }

let pp_operand ppf o = Format.fprintf ppf "%a %a" pp_layout o.l pp_ints o.s

let operand ?(min_rank = 0) () =
  let open Gen in
  with_pp pp_operand
    (let* rank = int_range min_rank 4 in
     (* One draw in four has one element, which a uniform draw of extents gives
        in only 8% of draws of rank 1 to 4. *)
     let* s =
       frequency
         [
           (1, constant (Array.make rank 1));
           (3, array ~size:(constant rank) (int_range 0 3));
         ]
     in
     let+ l = layouts in
     operand_of s l)

let covers o =
  cover "no element" (numel o.s = 0);
  cover "one element" (numel o.s = 1);
  cover "strided" (o.l = Reversed || o.l = Axes_reversed || o.l = Every_other)

(* Asserts that [y] is the value of shape [s] whose element at [i] is [f i]. *)
let is s f y =
  equal ~msg:"shape" (array int) s (Nx.shape y);
  equal ~msg:"elements" (array int32) (init s f) (elements y)

let at o i = o.e.(position o.s i)

(* [o]'s value through the permutation [p]: its axis [p.(i)] is axis [i]. *)
let permuted o p y =
  let s = Array.map (fun a -> o.s.(a)) p in
  is s
    (fun i ->
      let j = Array.make (Array.length p) 0 in
      Array.iteri (fun k a -> j.(a) <- i.(k)) p;
      at o j)
    y

let axis_of o = Gen.int_range 0 (Array.length o.s - 1)

(* Axis [a] of [o] as the caller writes it: from the start, or from the end
   where [neg]. *)
let spell o neg a = if neg then a - Array.length o.s else a

(* Memory *)

let held v = Option.get (Nx.Repr.array ~by:"test" v)
let shares x y = Rig.Buffer.overlaps (A.buffer (held x)) (A.buffer (held y))

(* Values, printed as their array's layout. *)
let laid_out =
  Testable.make
    ~pp:(fun ppf v -> A.Layout.pp ppf (A.layout (held v)))
    ~equal:( == )

let over ?msg x y =
  satisfies ?msg ~claim:"a view over the operand's memory" laid_out (shares x) y

let apart ?msg x y =
  satisfies ?msg ~claim:"memory of its own" laid_out
    (fun y -> not (shares x y))
    y

let x23 () = (operand_of [| 2; 3 |] C_order).x

(* Properties *)

let properties =
  group "ndim, dim, numel and nbytes"
    [
      test "read the shape and the dtype's bits" (fun () ->
          let x = Nx.zeros Nx.int4 [| 2; 3; 5 |] in
          equal ~msg:"ndim" int 3 (Nx.ndim x);
          equal ~msg:"dim 1" int 3 (Nx.dim 1 x);
          equal ~msg:"dim -1" int 5 (Nx.dim (-1) x);
          equal ~msg:"numel" int 30 (Nx.numel x);
          equal ~msg:"nbytes, (30 * 4 + 7) / 8" int 15 (Nx.nbytes x);
          equal ~msg:"numel of a 0-d value" int 1
            (Nx.numel (Nx.scalar Nx.float64 1.));
          equal ~msg:"nbytes of 9 bits" int 2
            (Nx.nbytes (Nx.zeros Nx.bit [| 9 |])));
      raises_naming "Nx.dim"
        [ r "2 of a rank-2 value" (fun () -> Nx.dim 2 (x23 ())) ];
    ]

(* reshape *)

let reshape =
  group "reshape"
    [
      prop "keeps the elements in C order" (operand ()) (fun o ->
          covers o;
          let n = numel o.s in
          is [| n |] (fun i -> o.e.(i.(0))) (Nx.reshape [| n |] o.x);
          let s' = Array.map (fun a -> o.s.(a)) (rev_axes (Array.length o.s)) in
          is s' (fun i -> o.e.(position s' i)) (Nx.reshape s' o.x));
      prop "infers its one unknown extent"
        Gen.(
          let* rank = int_range 1 4 in
          let* s = array ~size:(constant rank) (int_range 1 3) in
          let* d = int_range 0 3 in
          let+ l = layouts in
          s.(0) <- d;
          operand_of s l)
        (fun o ->
          let s' = Array.copy o.s in
          s'.(0) <- -1;
          equal (array int) o.s (Nx.shape (Nx.reshape s' o.x)));
      test "copies where no stride expresses it" (fun () ->
          let x = Nx.transpose (x23 ()) in
          let y = Nx.reshape [| 6 |] x in
          apart x y;
          equal (array int32) [| 1l; 4l; 2l; 5l; 3l; 6l |] (elements y));
      raises_naming "Nx.reshape"
        [
          r "two unknown extents" (fun () -> Nx.reshape [| -1; -1 |] (x23 ()));
          r "an extent below -1" (fun () -> Nx.reshape [| -2; -3 |] (x23 ()));
          r "an unknown that does not divide" (fun () ->
              Nx.reshape [| 4; -1 |] (x23 ()));
          r "an unknown beside extents of no element" (fun () ->
              Nx.reshape [| 0; -1 |] (Nx.zeros Nx.int32 [| 0; 3 |]));
          r "counts that differ" (fun () -> Nx.reshape [| 4; 2 |] (x23 ()));
          r "33 axes" (fun () ->
              Nx.reshape (Array.append [| 2; 3 |] (Array.make 31 1)) (x23 ()));
          r "extents past 2^53" (fun () ->
              Nx.reshape [| 0; 1 lsl 27; 1 lsl 27 |] (Nx.zeros Nx.int32 [| 0 |]));
        ];
    ]

(* broadcast_to, broadcast_shapes, broadcast_arrays *)

let broadcast_to =
  group "broadcast_to"
    [
      prop "repeats along new and unit axes"
        Gen.(
          let* o = operand () in
          let* lead = int_range 0 2 in
          let* lead = array ~size:(constant lead) (int_range 0 3) in
          let+ grown =
            array ~size:(constant (Array.length o.s)) (int_range 0 3)
          in
          (o, lead, grown))
        (fun (o, lead, grown) ->
          covers o;
          let k = Array.length lead in
          let s' =
            Array.append lead
              (Array.mapi (fun a d -> if d = 1 then grown.(a) else d) o.s)
          in
          is s'
            (fun i ->
              at o (Array.mapi (fun a d -> if d = 1 then 0 else i.(k + a)) o.s))
            (Nx.broadcast_to s' o.x));
      raises_naming "Nx.broadcast_to"
        [
          r "fewer axes" (fun () -> Nx.broadcast_to [| 3 |] (x23 ()));
          r "an extent other than 1 stretched" (fun () ->
              Nx.broadcast_to [| 2; 4 |] (x23 ()));
          r "33 axes" (fun () ->
              Nx.broadcast_to
                (Array.append (Array.make 31 1) [| 2; 3 |])
                (x23 ()));
          r "extents past 2^53" (fun () ->
              Nx.broadcast_to [| 1 lsl 27; 1 lsl 27; 2; 3 |] (x23 ()));
        ];
    ]

let broadcast_shapes =
  group "broadcast_shapes"
    [
      cases "aligns from the last axes" ~id:fst
        [
          ( "[4; 1], [2; 1; 3] and []",
            ([ [| 4; 1 |]; [| 2; 1; 3 |]; [||] ], [| 2; 4; 3 |]) );
          ("no shape", ([], [||]));
          ("[0; 1] and [3]", ([ [| 0; 1 |]; [| 3 |] ], [| 0; 3 |]));
        ]
        (fun (_, (ss, want)) -> equal (array int) want (Nx.broadcast_shapes ss));
      raises_naming "Nx.broadcast_shapes"
        [
          r "two that do not broadcast" (fun () ->
              Nx.broadcast_shapes [ [| 2; 1 |]; [| 1; 3 |]; [| 4; 3 |] ]);
          r "a negative extent" (fun () ->
              Nx.broadcast_shapes [ [| 2 |]; [| -1 |] ]);
        ];
    ]

let broadcast_arrays =
  group "broadcast_arrays"
    [
      test "stretches each to their common shape, in order" (fun () ->
          let a = (operand_of [| 2; 1 |] C_order).x
          and b = (operand_of [| 3 |] Reversed).x in
          equal
            (list (array int32))
            [ [| 1l; 1l; 1l; 2l; 2l; 2l |]; [| 1l; 2l; 3l; 1l; 2l; 3l |] ]
            (List.map elements (Nx.broadcast_arrays [ a; b ])));
      raises_naming "Nx.broadcast_arrays"
        [
          r "shapes that do not broadcast" (fun () ->
              Nx.broadcast_arrays [ x23 (); Nx.zeros Nx.int32 [| 4 |] ]);
        ];
    ]

(* An operation's operands whose broadcast passes the layout's limits: the raise
   names the operation, in nx's words (e708b7e46e: a logical map's raised the
   layout's bare message). *)
(* Operands of [2^27] elements, broadcast views of one, which allocate
   nothing. *)
let column () = Nx.broadcast_to [| 1 lsl 27; 1 |] (Nx.zeros Nx.int32 [| 1; 1 |])
let row () = Nx.broadcast_to [| 1; 1 lsl 27 |] (Nx.zeros Nx.int32 [| 1; 1 |])

let broadcasting =
  group "broadcasting"
    [
      cases "past 2^53 elements raises naming the operation" ~id:fst
        [
          ("add", ("Nx.add", fun () -> ignore (Nx.add (column ()) (row ()))));
          ( "logical_and",
            ( "Nx.logical_and",
              fun () -> ignore (Nx.logical_and (column ()) (row ())) ) );
        ]
        (fun (_, (by, f)) -> named ~by f);
    ]

(* meshgrid *)

(* Up to three 1-d operands, each laid out as drawn. *)
let grid_case =
  let open Gen in
  with_pp
    (fun ppf (ij, os) ->
      Format.fprintf ppf "%s [%s]"
        (if ij then "ij" else "xy")
        (String.concat "; " (List.map (Format.asprintf "%a" pp_operand) os)))
    (let* ij = bool in
     let* n = int_range 0 3 in
     let one =
       let* d =
         frequency [ (1, constant 0); (1, constant 1); (3, int_range 2 3) ]
       in
       let+ l = layouts in
       operand_of [| d |] l
     in
     let+ os = list ~size:(constant n) one in
     (ij, os))

(* Grid [i] holds operand [i]'s element at its axis' index: axis [i], or for
   [`Xy] axis [1 - i] of the first two. *)
let law_meshgrid (ij, os) =
  let n = List.length os in
  cover "two operands or more, xy" ((not ij) && n >= 2);
  cover "an empty operand" (List.exists (fun o -> o.s.(0) = 0) os);
  let along i = if (not ij) && n >= 2 && i < 2 then 1 - i else i in
  let s = Array.make n 0 in
  List.iteri (fun i o -> s.(along i) <- o.s.(0)) os;
  let ys =
    Nx.meshgrid ~indexing:(if ij then `Ij else `Xy) (List.map (fun o -> o.x) os)
  in
  equal ~msg:"grids" int n (List.length ys);
  List.iteri
    (fun i (o, y) ->
      equal ~msg:"shape" (array int) s (Nx.shape y);
      equal ~msg:"elements" (array int32)
        (init s (fun idx -> o.e.(idx.(along i))))
        (elements y))
    (List.combine os ys)

let meshgrid =
  group "meshgrid"
    [
      prop "holds each operand along its axis"
        ~examples:
          [
            (true, [ operand_of [| 0 |] C_order; operand_of [| 2 |] Reversed ]);
          ]
        grid_case law_meshgrid;
      cases "at a plane runs x along columns under `Xy, along rows under `Ij"
        ~id:fst
        [
          ( "`Xy",
            (`Xy, [ [| 1l; 2l; 3l; 1l; 2l; 3l |]; [| 1l; 1l; 1l; 2l; 2l; 2l |] ])
          );
          ( "`Ij",
            (`Ij, [ [| 1l; 1l; 2l; 2l; 3l; 3l |]; [| 1l; 2l; 1l; 2l; 1l; 2l |] ])
          );
        ]
        (fun (_, (indexing, want)) ->
          let x = (operand_of [| 3 |] C_order).x
          and y = (operand_of [| 2 |] C_order).x in
          equal
            (list (array int32))
            want
            (List.map elements (Nx.meshgrid ~indexing [ x; y ])));
      test "makes views" (fun () ->
          let x = (operand_of [| 3 |] C_order).x
          and y = (operand_of [| 2 |] C_order).x in
          match Nx.meshgrid ~indexing:`Xy [ x; y ] with
          | [ gx; gy ] ->
              over ~msg:"x" x gx;
              over ~msg:"y" y gy
          | ys -> failf "%d grids" (List.length ys));
      raises_naming "Nx.meshgrid"
        [
          r "a 2-d operand" (fun () ->
              Nx.meshgrid ~indexing:`Ij [ (operand_of [| 2; 2 |] C_order).x ]);
          r "a 0-d operand" (fun () ->
              Nx.meshgrid ~indexing:`Xy [ (operand_of [||] C_order).x ]);
        ];
    ]

(* squeeze, unsqueeze, flatten *)

let squeeze =
  group "squeeze"
    [
      prop "undoes unsqueeze"
        Gen.(
          let* o = operand () in
          let r = Array.length o.s in
          let* k = int_range 0 2 in
          (* Which axes of the result are new: [k] of [r + k]. *)
          let+ fresh = permutation (List.init (r + k) (fun i -> i < k)) in
          (o, List.concat (List.mapi (fun a n -> if n then [ a ] else []) fresh)))
        (fun (o, at) ->
          let y = Nx.unsqueeze ~axes:at o.x in
          equal ~msg:"rank" int (Array.length o.s + List.length at) (Nx.ndim y);
          List.iter (fun a -> equal ~msg:"a unit" int 1 (Nx.dim a y)) at;
          equal (array int32) o.e (elements y);
          let z = Nx.squeeze ~axes:at y in
          equal (array int) o.s (Nx.shape z);
          equal (array int32) o.e (elements z));
      prop "drops every unit axis without axes" (operand ()) (fun o ->
          let y = Nx.squeeze o.x in
          equal (array int)
            (Array.of_list (List.filter (( <> ) 1) (Array.to_list o.s)))
            (Nx.shape y);
          equal (array int32) o.e (elements y));
      raises_naming "Nx.squeeze"
        [
          r "an axis of extent 3" (fun () -> Nx.squeeze ~axes:[ 1 ] (x23 ()));
          r "a repeated axis" (fun () ->
              Nx.squeeze ~axes:[ 0; -2 ] (Nx.reshape [| 1; 6 |] (x23 ())));
          r "an axis past the value's" (fun () ->
              Nx.squeeze ~axes:[ 2 ] (x23 ()));
        ];
    ]

let unsqueeze =
  group "unsqueeze"
    [
      test "puts its units at positions of the result" (fun () ->
          equal (array int) [| 1; 3; 1 |]
            (Nx.shape
               (Nx.unsqueeze ~axes:[ 0; -1 ] (Nx.zeros Nx.int32 [| 3 |]))));
      raises_naming "Nx.unsqueeze"
        [
          r "a position past the result" (fun () ->
              Nx.unsqueeze ~axes:[ 3 ] (x23 ()));
          r "a repeated position" (fun () ->
              Nx.unsqueeze ~axes:[ 0; -4 ] (x23 ()));
          r "33 axes" (fun () ->
              Nx.unsqueeze ~axes:[ 0 ] (Nx.zeros Nx.int32 (Array.make 32 1)));
        ];
    ]

let flatten =
  group "flatten"
    [
      prop "merges a run of axes, a 0-d value into one element"
        ~examples:[ (operand_of [||] C_order, 0, 0, false) ]
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let* b = int_range a (Array.length o.s - 1) in
          let+ neg = bool in
          (o, a, b, neg))
        (fun (o, a, b, neg) ->
          let r = Array.length o.s in
          cover "a 0-d value" (r = 0);
          equal (array int) [| numel o.s |] (Nx.shape (Nx.flatten o.x));
          if r > 0 then begin
            let y =
              Nx.flatten ~start_dim:(spell o neg a) ~end_dim:(spell o neg b) o.x
            in
            equal (array int)
              (Array.concat
                 [
                   Array.sub o.s 0 a;
                   [| numel (Array.sub o.s a (b - a + 1)) |];
                   Array.sub o.s (b + 1) (r - b - 1);
                 ])
              (Nx.shape y);
            equal (array int32) o.e (elements y)
          end);
      raises_naming "Nx.flatten"
        [
          r "start after end" (fun () ->
              Nx.flatten ~start_dim:1 ~end_dim:0 (x23 ()));
          r "an axis past the value's" (fun () ->
              Nx.flatten ~end_dim:2 (x23 ()));
        ];
    ]

(* transpose, moveaxis, swapaxes, flip *)

let transpose =
  group "transpose"
    [
      prop "moves each axis where its permutation says, reversing without one"
        Gen.(
          let* o = operand () in
          let* p = permutation (List.init (Array.length o.s) Fun.id) in
          let+ neg = bool in
          (o, p, neg))
        (fun (o, p, neg) ->
          covers o;
          permuted o (Array.of_list p)
            (Nx.transpose ~axes:(List.map (spell o neg) p) o.x);
          permuted o (rev_axes (Array.length o.s)) (Nx.transpose o.x));
      raises_naming "Nx.transpose"
        [
          r "too few axes" (fun () -> Nx.transpose ~axes:[ 0 ] (x23 ()));
          r "a repeated axis" (fun () -> Nx.transpose ~axes:[ 0; 0 ] (x23 ()));
        ];
    ]

let moveaxis =
  group "moveaxis and swapaxes"
    [
      prop "are permutations"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let+ b = axis_of o in
          (o, a, b))
        (fun (o, a, b) ->
          covers o;
          let r = Array.length o.s in
          let rest = List.filter (( <> ) a) (List.init r Fun.id) in
          let p =
            List.filteri (fun i _ -> i < b) rest
            @ (a :: List.filteri (fun i _ -> i >= b) rest)
          in
          permuted o (Array.of_list p) (Nx.moveaxis a b o.x);
          permuted o (Array.of_list p) (Nx.moveaxis (a - r) (b - r) o.x);
          let q = Array.init r Fun.id in
          q.(a) <- b;
          q.(b) <- a;
          permuted o q (Nx.swapaxes a b o.x));
      cases "raise naming themselves" ~id:fst
        [
          ( "moveaxis, not an axis",
            ("Nx.moveaxis", fun () -> ignore (Nx.moveaxis 0 2 (x23 ()))) );
          ( "swapaxes, not an axis",
            ("Nx.swapaxes", fun () -> ignore (Nx.swapaxes (-3) 0 (x23 ()))) );
        ]
        (fun (_, (by, f)) -> named ~by f);
    ]

let flip =
  group "flip"
    [
      prop "reverses each axis it names, every axis without axes"
        Gen.(
          let* o = operand () in
          let* f = subsequence (List.init (Array.length o.s) Fun.id) in
          let+ neg = bool in
          (o, f, neg))
        (fun (o, f, neg) ->
          covers o;
          is o.s
            (fun i ->
              at o
                (Array.mapi
                   (fun a k -> if List.mem a f then o.s.(a) - 1 - k else k)
                   i))
            (Nx.flip ~axes:(List.map (spell o neg) f) o.x);
          is o.s
            (fun i -> at o (Array.mapi (fun a k -> o.s.(a) - 1 - k) i))
            (Nx.flip o.x));
      raises_naming "Nx.flip"
        [
          r "a repeated axis" (fun () -> Nx.flip ~axes:[ 1; -1 ] (x23 ()));
          r "an axis past the value's" (fun () -> Nx.flip ~axes:[ 2 ] (x23 ()));
        ];
    ]

(* sliding_window, split, split_at *)

let sliding_window =
  group "sliding_window"
    [
      prop "reads each window's elements"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let d = o.s.(a) in
          let* window = int_range 1 (d + 2) in
          let* step = int_range 1 3 in
          let+ neg = bool in
          (o, a, window, step, neg))
        (fun (o, a, window, step, neg) ->
          covers o;
          let d = o.s.(a) in
          cover "a window longer than its axis" (window > d);
          let y = Nx.sliding_window ~axis:(spell o neg a) ~window ~step o.x in
          let r = Array.length o.s in
          let s' = Array.append o.s [| window |] in
          s'.(a) <- (if window > d then 0 else ((d - window) / step) + 1);
          is s'
            (fun i ->
              let j = Array.sub i 0 r in
              j.(a) <- (i.(a) * step) + i.(r);
              at o j)
            y);
      raises_naming "Nx.sliding_window"
        [
          r "step 0" (fun () -> Nx.sliding_window ~window:1 ~step:0 (x23 ()));
          r "window 0" (fun () -> Nx.sliding_window ~window:0 (x23 ()));
          r "an axis past the value's" (fun () ->
              Nx.sliding_window ~axis:2 ~window:1 (x23 ()));
          r "33 axes" (fun () ->
              Nx.sliding_window ~window:1 (Nx.zeros Nx.int32 (Array.make 32 1)));
        ];
    ]

let split =
  group "split"
    [
      prop "cuts runs whose lengths differ by at most one, longer first"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let* n = int_range 1 4 in
          let+ neg = bool in
          (o, a, n, neg))
        (fun (o, a, n, neg) ->
          covers o;
          let parts = Nx.split ~axis:(spell o neg a) n o.x in
          let d = o.s.(a) in
          equal ~msg:"runs" int n (List.length parts);
          let start = ref 0 in
          List.iteri
            (fun k y ->
              let count = (d / n) + if k < d mod n then 1 else 0 in
              let s' = Array.copy o.s in
              s'.(a) <- count;
              let first = !start in
              is s'
                (fun i ->
                  let j = Array.copy i in
                  j.(a) <- first + i.(a);
                  at o j)
                y;
              start := !start + count)
            parts);
      raises_naming "Nx.split"
        [
          r "no runs" (fun () -> Nx.split ~axis:0 0 (x23 ()));
          r "an axis past the value's" (fun () -> Nx.split ~axis:2 2 (x23 ()));
        ];
    ]

(* Where an offset cuts an axis of [d]: counted from the end if negative, then
   clipped into [0, d]. *)
let cut_at d k = Stdlib.max 0 (Stdlib.min d (if k < 0 then k + d else k))

let law_split_at (o, a, offsets, neg) =
  covers o;
  let d = o.s.(a) in
  let by_place j k = compare (cut_at d j) (cut_at d k) in
  let ordered = List.stable_sort by_place offsets = offsets in
  cover "offsets in order" ordered;
  cover "offsets out of order" (not ordered);
  let cuts = (0 :: List.map (cut_at d) offsets) @ [ d ] in
  cover "no offset" (ordered && offsets = []);
  cover "a negative offset" (ordered && List.exists (fun k -> k < 0) offsets);
  cover "an offset outside the axis"
    (ordered && List.exists (fun k -> k < -d || k > d) offsets);
  cover "an empty run"
    (ordered
    && List.exists2 ( = )
         (List.filteri (fun i _ -> i > 0) cuts)
         (List.filteri (fun i _ -> i < List.length cuts - 1) cuts));
  let axis = spell o neg a in
  if not ordered then
    invalid ~by:"Nx.split_at" (fun () -> Nx.split_at ~axis offsets o.x)
  else begin
    let parts = Nx.split_at ~axis offsets o.x in
    equal ~msg:"runs" int (List.length offsets + 1) (List.length parts);
    List.iteri
      (fun k y ->
        let first = List.nth cuts k in
        let s' = Array.copy o.s in
        s'.(a) <- List.nth cuts (k + 1) - first;
        is s'
          (fun i ->
            let j = Array.copy i in
            j.(a) <- first + i.(a);
            at o j)
          y)
      parts
  end

let split_at =
  group "split_at"
    [
      prop "cuts before each offset, counted from the end where negative"
        ~examples:
          [
            (operand_of [| 2; 3 |] C_order, 1, [ 2; 1 ], false);
            (operand_of [| 3 |] Every_other, 0, [], true);
          ]
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let d = o.s.(a) in
          let* offsets =
            list ~size:(int_range 0 3) (int_range (-d - 2) (d + 2))
          in
          let+ neg = bool in
          (o, a, offsets, neg))
        law_split_at;
      test "cuts [w; x; y; z] before 2 and 3 into [w; x], [y] and [z]"
        (fun () ->
          let x = Nx.create Nx.int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
          equal
            (list (array int32))
            [ [| 1l; 2l |]; [| 3l |]; [| 4l |] ]
            (List.map elements (Nx.split_at ~axis:0 [ 2; 3 ] x)));
      raises_naming "Nx.split_at"
        [
          r "an offset before the one before it" (fun () ->
              Nx.split_at ~axis:1 [ 2; 1 ] (x23 ()));
          r "an offset from the end before the one before it" (fun () ->
              Nx.split_at ~axis:1 [ -1; 1 ] (x23 ()));
          r "an axis past the value's" (fun () ->
              Nx.split_at ~axis:2 [ 1 ] (x23 ()));
        ];
    ]

(* concatenate, stack, pad *)

let concatenate =
  group "concatenate"
    [
      prop "undoes split"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let+ n = int_range 1 4 in
          (o, a, n))
        (fun (o, a, n) ->
          covers o;
          is o.s (at o) (Nx.concatenate ~axis:a (Nx.split ~axis:a n o.x)));
      prop "undoes split_at"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let+ offsets = list ~size:(int_range 0 3) (int_range 0 o.s.(a)) in
          (o, a, List.sort compare offsets))
        (fun (o, a, offsets) ->
          covers o;
          is o.s (at o)
            (Nx.concatenate ~axis:a (Nx.split_at ~axis:a offsets o.x)));
      raises_naming "Nx.concatenate"
        [
          r "no value" (fun () -> Nx.concatenate ~axis:0 []);
          r "shapes that differ off the axis" (fun () ->
              let x = x23 () in
              Nx.concatenate ~axis:0 [ x; Nx.transpose x ]);
          r "an axis past the first value's" (fun () ->
              Nx.concatenate ~axis:2 [ x23 () ]);
        ];
    ]

(* [n] values of [o]'s shape and layout, the [k]th reading [o]'s elements plus
   [100 k]. *)
let shifted o n =
  List.init n (fun k ->
      let data =
        Array.init (numel o.s) (fun j -> Int32.of_int (j + 1 + (100 * k)))
      in
      fst (laid o.l D.Int32 o.s data))

let law_stack (o, a, n) =
  covers o;
  let r = Array.length o.s in
  let s' =
    Array.init (r + 1) (fun i ->
        if i < a then o.s.(i) else if i = a then n else o.s.(i - 1))
  in
  is s'
    (fun i ->
      let j = Array.init r (fun d -> if d < a then i.(d) else i.(d + 1)) in
      Int32.add (at o j) (Int32.of_int (100 * i.(a))))
    (Nx.stack ~axis:a (shifted o n))

let stack =
  group "stack"
    [
      prop "puts each value at its position along the new axis"
        ~examples:
          [
            (operand_of [| 0; 2 |] C_order, 1, 2);
            (operand_of [||] C_order, 0, 2);
            (operand_of [| 2; 3 |] Reversed, 1, 2);
            (operand_of [| 2; 3 |] Axes_reversed, 2, 2);
            (operand_of [| 2; 3 |] One_broadcast, 0, 2);
          ]
        Gen.(
          let* o = operand () in
          let* a = int_range 0 (Array.length o.s) in
          let+ n = int_range 1 3 in
          (o, a, n))
        law_stack;
      raises_naming "Nx.stack"
        [
          r "an axis past the result" (fun () -> Nx.stack ~axis:3 [ x23 () ]);
          r "shapes that differ" (fun () ->
              let x = x23 () in
              Nx.stack [ x; Nx.reshape [| 3; 2 |] x ]);
          r "no value" (fun () -> Nx.stack []);
        ];
    ]

let pad =
  group "pad"
    [
      prop "surrounds the elements with its value"
        Gen.(
          let* o = operand () in
          let+ widths =
            array
              ~size:(constant (Array.length o.s))
              (pair (int_range 0 2) (int_range 0 2))
          in
          (o, widths))
        (fun (o, widths) ->
          covers o;
          let s' =
            Array.mapi (fun i d -> fst widths.(i) + d + snd widths.(i)) o.s
          in
          is s'
            (fun i ->
              let j = Array.mapi (fun d k -> k - fst widths.(d)) i in
              if
                Array.for_all Fun.id
                  (Array.mapi (fun d k -> k >= 0 && k < o.s.(d)) j)
              then at o j
              else -7l)
            (Nx.pad widths (-7l) o.x));
      raises_naming "Nx.pad"
        [
          r "fewer pairs than axes" (fun () -> Nx.pad [| (1, 1) |] 0l (x23 ()));
          r "a negative width" (fun () ->
              Nx.pad [| (0, 0); (-1, 0) |] 0l (x23 ()));
          r "an int outside the dtype" (fun () ->
              Nx.pad [| (1, 0) |] 256 (Nx.zeros Nx.uint8 [| 2 |]));
          r "widths past 2^53" (fun () ->
              Nx.pad [| (1 lsl 27, 0); (1 lsl 27, 0) |] 0l (x23 ()));
          r "widths past an int" (fun () ->
              Nx.pad [| (max_int, max_int); (0, 0) |] 0l (x23 ()));
        ];
    ]

(* Extensions: each rule against its equations, as the type's doc states
   them. *)

let rules : Nx.extension list =
  [
    `Zero;
    `Replicate;
    `Symmetric_whole;
    `Symmetric_half;
    `Antisymmetric_whole;
    `Periodic;
  ]

let rule_name : Nx.extension -> string = function
  | `Zero -> "zero"
  | `Replicate -> "replicate"
  | `Symmetric_whole -> "symmetric_whole"
  | `Symmetric_half -> "symmetric_half"
  | `Antisymmetric_whole -> "antisymmetric_whole"
  | `Periodic -> "periodic"

(* Position [p] of the signal [f 0], …, [f (n - 1)] read under [e], by the
   rule's equations read again outside the signal; [opp a b] is [2 a - b] in the
   signal's arithmetic. *)
let rec read ~zero ~opp (e : Nx.extension) n f p =
  let again = read ~zero ~opp e n f in
  if p >= 0 && p < n then f p
  else if e = `Zero then zero
  else if n = 1 then f 0
  else if p < 0 then
    let i = -p in
    match e with
    | `Zero -> zero
    | `Replicate -> f 0
    | `Symmetric_whole -> again i
    | `Symmetric_half -> again (i - 1)
    | `Antisymmetric_whole -> opp (f 0) (again i)
    | `Periodic -> again (n - i)
  else
    let i = p - (n - 1) in
    match e with
    | `Zero -> zero
    | `Replicate -> f (n - 1)
    | `Symmetric_whole -> again (n - 1 - i)
    | `Symmetric_half -> again (n - i)
    | `Antisymmetric_whole -> opp (f (n - 1)) (again (n - 1 - i))
    | `Periodic -> again (i - 1)

let read32 = read ~zero:0l ~opp:(fun a b -> Int32.sub (Int32.mul 2l a) b)

(* [o]'s value extended along [a] by [lo] and [hi] under [e], by [read32]. *)
let extended o a e lo hi =
  let s' = Array.copy o.s in
  s'.(a) <- lo + o.s.(a) + hi;
  let at_k i k =
    let j = Array.copy i in
    j.(a) <- k;
    at o j
  in
  (s', fun i -> read32 e o.s.(a) (at_k i) (i.(a) - lo))

(* Float elements with every NaN alike, a NaN promising no bits. *)
let float_bits =
  Testable.make
    ~pp:(fun ppf v -> Format.fprintf ppf "%h" v)
    ~equal:(fun a b ->
      (Float.is_nan a && Float.is_nan b)
      || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b))

(* [2 a - b] rounded once, the antisymmetric rule's arithmetic. *)
let opp_f a b = Float.fma 2. a (-.b)

let host dt s data =
  Nx.Repr.of_array Nx.Host.v (A.of_array dt (Iarray.of_array s) data)

let extension =
  group "extension"
    [
      cases "reads a signal as the type's doc shows"
        ~id:(fun (n, _, _, _) -> n)
        [
          ( "abcd as dcb|abcd|cba, symmetric_whole",
            `Symmetric_whole,
            [| 1.; 2.; 3.; 4. |],
            [| 4.; 3.; 2.; 1.; 2.; 3.; 4.; 3.; 2.; 1. |] );
          ( "abcd as cba|abcd|dcb, symmetric_half",
            `Symmetric_half,
            [| 1.; 2.; 3.; 4. |],
            [| 3.; 2.; 1.; 1.; 2.; 3.; 4.; 4.; 3.; 2. |] );
          ( "1 2 4 as -2 0|1 2 4|6 7, antisymmetric_whole",
            `Antisymmetric_whole,
            [| 1.; 2.; 4. |],
            [| -2.; 0.; 1.; 2.; 4.; 6.; 7. |] );
        ]
        (fun (_, e, v, want) ->
          let n = Array.length v in
          let k = (Array.length want - n) / 2 in
          equal (array float_exact) want
            (elements (Nx.extend e (k, k) (host D.Float64 [| n |] v))));
    ]

(* Signals of rank 1 to 3 laid out as drawn whose drawn axis has up to five
   samples, and in a quarter of draws broadcasts of several elements. *)
let signal =
  let open Gen in
  with_pp
    (fun ppf (o, a) -> Format.fprintf ppf "%a along %d" pp_operand o a)
    (let* rank = int_range 1 3 in
     let* a = int_range 0 (rank - 1) in
     let* s =
       array ~size:(constant rank)
         (frequency [ (1, constant 1); (3, int_range 0 3) ])
     in
     let* n = int_range 0 5 in
     let* broadcast = frequency [ (3, constant false); (1, constant true) ] in
     let+ l = layouts in
     s.(a) <- n;
     if broadcast then
       (operand_of (Array.map (fun d -> Stdlib.max d 2) s) One_broadcast, a)
     else (operand_of s l, a))

(* The signal with every extent at least 1. *)
let nonempty (o, a) =
  if numel o.s > 0 then (o, a)
  else (operand_of (Array.map (Stdlib.max 1) o.s) o.l, a)

let law_extend ((o, a), neg, e, lo, hi) =
  covers o;
  let n = o.s.(a) in
  cover "a broadcast operand" (o.l = One_broadcast && numel o.s > 1);
  cover "one sample, widened" (n = 1 && e <> `Zero && lo + hi > 0);
  cover "a fold past the far end" (n >= 2 && (lo > n || hi > n));
  cover "folds twice past an end" (n = 2 && (lo > 2 * n || hi > 2 * n));
  let y () = Nx.extend ~axis:(spell o neg a) e (lo, hi) o.x in
  if n = 0 && e <> `Zero && lo + hi > 0 then invalid ~by:"Nx.extend" y
  else
    let s', f = extended o a e lo hi in
    is s' f (y ())

let law_extend_folds ((o, a), e, lo, hi) =
  let n = o.s.(a) in
  cover "one sample" (n = 1);
  cover "a fold past the far end" (n >= 2 && (lo > n || hi > n));
  let ext e (lo, hi) x = Nx.extend ~axis:a e (lo, hi) x in
  let along r = List.init a (fun _ -> Nx.A) @ [ r ] in
  let y = ext e (lo, hi) o.x in
  (* Slicing the samples back gives the signal. *)
  is o.s (at o) (Nx.slice (along (R (lo, lo + n))) y);
  (* A periodic extension indexes modulo [n]. *)
  let positions =
    Nx.create Nx.int64
      [| lo + n + hi |]
      (Array.init
         (lo + n + hi)
         (fun k -> Int64.of_int ((((k - lo) mod n) + n) mod n)))
  in
  equal ~msg:"periodic" (array int32)
    (elements (Nx.take ~axis:a positions o.x))
    (elements (ext `Periodic (lo, hi) o.x));
  (* A symmetric extension is the periodic one of the signal and its mirror, the
     whole-sample mirror without its end samples. *)
  let mirror start count =
    Nx.flip ~axes:[ a ] (Nx.slice (along (R (start, start + count))) o.x)
  in
  let periodic_of back =
    let p = Nx.concatenate ~axis:a [ o.x; back ] in
    Nx.slice (along (R (0, lo + n + hi))) (ext `Periodic (lo, hi) p)
  in
  equal ~msg:"half" (array int32)
    (elements (periodic_of (mirror 0 n)))
    (elements (ext `Symmetric_half (lo, hi) o.x));
  if n >= 2 then
    equal ~msg:"whole" (array int32)
      (elements (periodic_of (mirror 1 (n - 2))))
      (elements (ext `Symmetric_whole (lo, hi) o.x))

(* Within one fold, [x(-i) = 2 x(0) - x(i)] with [x(i)] the sample
   [`Symmetric_whole] reads, and alike at the end. *)
let law_antisymmetric (v, lo, hi) =
  let n = Array.length v in
  let x = host D.Float64 [| n |] v in
  let m =
    elements
      (Nx.take
         (Nx.extension_index `Symmetric_whole n
            (Nx.create Nx.int64
               [| lo + n + hi |]
               (Array.init (lo + n + hi) (fun k -> Int64.of_int (k - lo)))))
         x)
  in
  let expected =
    Array.init
      (lo + n + hi)
      (fun k ->
        let p = k - lo in
        if p < 0 then opp_f v.(0) m.(k)
        else if p >= n then opp_f v.(n - 1) m.(k)
        else v.(p))
  in
  equal (array float_bits) expected
    (elements (Nx.extend `Antisymmetric_whole (lo, hi) x))

let short_floats n =
  Gen.(
    array ~size:(constant n) (of_list [ 0.1; 0.7; -0.3; 1.9; 3.3; -2.6; 0.45 ]))

let law_position_bits (v, e, lo, hi) =
  let n = Array.length v in
  cover "antisymmetric, both sides past two folds"
    (e = `Antisymmetric_whole && lo > 2 * (n - 1) && hi > 2 * (n - 1));
  let x = host D.Float64 [| n |] v in
  let bits y = elements (Nx.bitcast D.Int64 y) in
  let y = bits (Nx.extend e (lo, hi) x) in
  let l = bits (Nx.extend e (lo, 0) x) in
  let r = bits (Nx.extend e (0, hi) x) in
  equal ~msg:"before" (array int64) (Array.sub l 0 lo) (Array.sub y 0 lo);
  equal ~msg:"after" (array int64) (Array.sub r n hi) (Array.sub y (lo + n) hi)

(* Rows whose antisymmetric gain is zero, [x(0) = x(n - 1)], of values every
   dtype holds exactly, extended past two folds. *)
let test_every_dtype (D.Any dt) =
  let data = [| 2.; 3.; 2.; 2.; 1.; 2. |] in
  let x =
    fst (laid Axes_reversed dt [| 2; 3 |] (Array.map (D.of_float dt) data))
  in
  let w = Testable.make ~pp:(D.pp_value dt) ~equal:( = ) in
  List.iter
    (fun e ->
      let y () = Nx.extend e (7, 7) x in
      if e = `Antisymmetric_whole && D.is D.Boolean dt then
        invalid ~by:"Nx.extend" y
      else
        let expected =
          init [| 2; 17 |] (fun i ->
              D.of_float dt
                (read ~zero:0. ~opp:opp_f e 3
                   (fun k -> data.((3 * i.(0)) + k))
                   (i.(1) - 7)))
        in
        equal ~msg:(rule_name e) (array w) expected (elements (y ())))
    rules

let test_antisymmetric_rows (_, v, lo, hi) =
  let n = Array.length v in
  let x = host D.Float64 [| n |] v in
  equal (array float_bits)
    (Array.init
       (lo + n + hi)
       (fun k ->
         read ~zero:0. ~opp:opp_f `Antisymmetric_whole n (Array.get v) (k - lo)))
    (elements (Nx.extend `Antisymmetric_whole (lo, hi) x))

(* Past the first fold, the rule reads samples it computed: each fold rounds
   once, and an error passes on to the samples reflected from it. Against the
   exact reflections, computed in double-word arithmetic, a sample [k] folds out
   is within [k u m] of its exact value, [m] the largest exact magnitude of the
   row. *)
let two_sum a b =
  let s = a +. b in
  let bb = s -. a in
  (s, a -. (s -. bb) +. (b -. bb))

let opp_exact (ah, al) (bh, bl) =
  let s, e = two_sum (2. *. ah) (-.bh) in
  two_sum s (e +. (2. *. al) -. bl)

let test_antisymmetric_folds (v, lo, hi) =
  let n = Array.length v in
  let exact =
    Array.init
      (lo + n + hi)
      (fun k ->
        fst
          (read ~zero:(0., 0.) ~opp:opp_exact `Antisymmetric_whole n
             (fun i -> (v.(i), 0.))
             (k - lo)))
  in
  let m = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. exact in
  let u = Float.ldexp 1. (-53) in
  let y =
    elements
      (Nx.extend `Antisymmetric_whole (lo, hi) (host D.Float64 [| n |] v))
  in
  Array.iteri
    (fun k x ->
      let p = k - lo in
      let past = if p < 0 then -p else if p >= n then p - (n - 1) else 0 in
      let folds = (past + n - 2) / (n - 1) in
      at_most
        ~msg:(Printf.sprintf "position %d, %d folds" p folds)
        float_exact
        ~than:(Float.of_int folds *. u *. m)
        (gap y.(k) x))
    exact

let test_antisymmetric_kinds () =
  (* [2 x(0)] overflows float16, [2 x(0) - x(1)] does not. *)
  let h = host D.Float16 [| 2 |] [| 60000.; 64992. |] in
  equal ~msg:"float16" (array float_bits)
    [| 55008.; 60000.; 64992. |]
    (elements (Nx.extend `Antisymmetric_whole (1, 0) h));
  let v = [| Int32.max_int; Int32.min_int; 5l |] in
  equal ~msg:"int32" (array int32)
    (Array.init 7 (fun k -> read32 `Antisymmetric_whole 3 (Array.get v) (k - 2)))
    (elements (Nx.extend `Antisymmetric_whole (2, 2) (host D.Int32 [| 3 |] v)));
  (* Each part by one fma, the other part not read. *)
  let z =
    [|
      { Complex.re = 0.1; im = -0.7 };
      { re = 0.7; im = max_float };
      { re = -0.3; im = 0.45 };
    |]
  in
  let parts f =
    Array.init 7 (fun k ->
        read ~zero:0. ~opp:opp_f `Antisymmetric_whole 3
          (fun i -> f z.(i))
          (k - 2))
  in
  let y =
    elements
      (Nx.extend `Antisymmetric_whole (2, 2) (host D.Complex128 [| 3 |] z))
  in
  equal ~msg:"re" (array float_bits)
    (parts (fun c -> c.Complex.re))
    (Array.map (fun c -> c.Complex.re) y);
  equal ~msg:"im" (array float_bits)
    (parts (fun c -> c.Complex.im))
    (Array.map (fun c -> c.Complex.im) y)

(* A complex value's parts take an axis more: at one axis fewer than the most,
   the rule computes. *)
let test_antisymmetric_most_axes () =
  let r = A.Layout.max_rank in
  let z =
    [|
      { Complex.re = 0.1; im = -0.7 };
      { re = 0.7; im = 2.5 };
      { re = -0.3; im = 0.45 };
    |]
  in
  let along r a = Array.init r (fun i -> if i = a then 3 else 1) in
  let part f k =
    read ~zero:0. ~opp:opp_f `Antisymmetric_whole 3 (fun i -> f z.(i)) (k - 1)
  in
  let x = host D.Complex128 (along (r - 1) (r - 2)) z in
  let y = elements (Nx.extend `Antisymmetric_whole (1, 1) x) in
  equal ~msg:"re" (array float_bits)
    (Array.init 5 (part (fun c -> c.Complex.re)))
    (Array.map (fun c -> c.Complex.re) y);
  equal ~msg:"im" (array float_bits)
    (Array.init 5 (part (fun c -> c.Complex.im)))
    (Array.map (fun c -> c.Complex.im) y)

let test_copying_bits () =
  let nan = Int64.float_of_bits 0x7ff8000000000001L in
  let x = signalling D.Float64 [| 3 |] [| nan; -0.; neg_infinity |] in
  let bits y = elements (Nx.bitcast D.Int64 y) in
  let v = bits x in
  equal ~msg:"signalling" int64 0x7ff0000000000001L v.(0);
  List.iter
    (fun e ->
      equal ~msg:(rule_name e) (array int64)
        (Array.init 11 (fun k ->
             read ~zero:0L ~opp:(fun _ _ -> 0L) e 3 (Array.get v) (k - 4)))
        (bits (Nx.extend e (4, 4) x)))
    [ `Replicate; `Symmetric_whole; `Symmetric_half; `Periodic ]

let complex_most_axes a () =
  let rk = A.Layout.max_rank in
  Nx.extend ~axis:a `Antisymmetric_whole (1, 1)
    (host D.Complex128
       (Array.init rk (fun i -> if i = a then 3 else 1))
       (Array.make 3 Complex.zero))

let extend =
  group "extend"
    [
      prop "reads each position by its rule's equations"
        ~examples:
          [
            ((operand_of [| 2 |] C_order, 0), false, `Symmetric_half, 5, 0);
            ((operand_of [| 1 |] C_order, 0), false, `Replicate, 2, 1);
          ]
        Gen.(
          let* sig_ = signal in
          let* neg = bool in
          let* e = of_list rules in
          let* lo = int_range 0 7 in
          let+ hi = int_range 0 7 in
          (sig_, neg, e, lo, hi))
        law_extend;
      prop "slices back to its signal, and folds as periodic extensions do"
        Gen.(
          let* sig_ = map nonempty signal in
          let* e = of_list rules in
          let* lo = int_range 0 7 in
          let+ hi = int_range 0 7 in
          (sig_, e, lo, hi))
        law_extend_folds;
      prop
        "reflects the whole-sample mirror's reads under the antisymmetric rule"
        Gen.(
          let* n = of_list [ 2; 3; 5 ] in
          let* v = short_floats n in
          let* lo = int_range 0 (n - 1) in
          let+ hi = int_range 0 (n - 1) in
          (v, lo, hi))
        law_antisymmetric;
      prop "gives a sample bits that depend on its position alone"
        ~examples:[ ([| 0.1; 0.7; 0.3 |], `Antisymmetric_whole, 9, 9) ]
        (Gen.with_pp
           (fun ppf (v, e, lo, hi) ->
             Format.fprintf ppf "%s (%d, %d) [%s]" (rule_name e) lo hi
               (String.concat "; "
                  (Array.to_list (Array.map string_of_float v))))
           Gen.(
             let* n = of_list [ 2; 3; 5 ] in
             let* v = short_floats n in
             let* e = of_list rules in
             let* lo = int_range 0 (4 * n) in
             let+ hi = int_range 0 (4 * n) in
             (v, e, lo, hi)))
        law_position_bits;
      cases
        "extends every dtype past two folds, under the antisymmetric rule \
         every one but booleans"
        ~id:(fun (D.Any dt) -> D.name dt)
        D.all test_every_dtype;
      cases "rounds once a fold and wraps under the antisymmetric rule"
        ~id:(fun (name, _, _, _) -> name)
        [
          ("float64 past half the range", [| 1e308; 1.5e308 |], 1, 0);
          ("float64 at the top", [| max_float; max_float; 0. |], 2, 2);
          ( "signed zeros, infinities and NaN",
            [| -0.; 0.; infinity; nan |],
            3,
            3 );
          ("float64 that rounds", [| 0.1; 0.7; 0.3 |], 2, 2);
        ]
        test_antisymmetric_rows;
      cases
        "rounds at most once a fold past the first under the antisymmetric rule"
        ~id:(fun (v, lo, hi) ->
          Printf.sprintf "%s by (%d, %d)"
            (String.concat " " (Array.to_list (Array.map string_of_float v)))
            lo hi)
        [
          ([| 0.1; 0.7; 0.3 |], 9, 9); ([| 0.1; 0.7; -0.3; 1.9; 0.45 |], 12, 13);
        ]
        test_antisymmetric_folds;
      test
        "computes float16 by one fma, wraps int32 and takes complex parts \
         alone under the antisymmetric rule"
        test_antisymmetric_kinds;
      test
        "computes complex values of one axis fewer than the most under the \
         antisymmetric rule"
        test_antisymmetric_most_axes;
      test "keeps a NaN's bits and signed zeros under the copying rules"
        test_copying_bits;
      test "widens an empty signal only under `Zero" (fun () ->
          let x = Nx.zeros Nx.int32 [| 2; 0 |] in
          equal ~msg:"zero" (array int32) (Array.make 6 0l)
            (elements (Nx.extend `Zero (1, 2) x));
          List.iter
            (fun e ->
              equal ~msg:(rule_name e) (array int) [| 2; 0 |]
                (Nx.shape (Nx.extend e (0, 0) x)))
            rules);
      raises_naming "Nx.extend"
        [
          r "a negative lo" (fun () -> Nx.extend `Zero (-1, 0) (x23 ()));
          r "a negative hi" (fun () -> Nx.extend `Periodic (0, -1) (x23 ()));
          r "an axis past the value's" (fun () ->
              Nx.extend ~axis:2 `Zero (1, 1) (x23 ()));
          r "an axis before the value's" (fun () ->
              Nx.extend ~axis:(-3) `Replicate (1, 1) (x23 ()));
          r "a 0-d value" (fun () ->
              Nx.extend `Replicate (1, 1) (Nx.scalar Nx.int32 1l));
          r "booleans under the antisymmetric rule" (fun () ->
              Nx.extend `Antisymmetric_whole (1, 1) (Nx.zeros Nx.bool [| 2 |]));
          r "an empty axis widened, replicate" (fun () ->
              Nx.extend `Replicate (0, 1) (Nx.zeros Nx.int32 [| 3; 0 |]));
          r "an empty axis widened, symmetric_whole" (fun () ->
              Nx.extend `Symmetric_whole (0, 1) (Nx.zeros Nx.int32 [| 3; 0 |]));
          r "an empty axis widened, symmetric_half" (fun () ->
              Nx.extend `Symmetric_half (0, 1) (Nx.zeros Nx.int32 [| 3; 0 |]));
          r "an empty axis widened, antisymmetric_whole" (fun () ->
              Nx.extend `Antisymmetric_whole (0, 1)
                (Nx.zeros Nx.int32 [| 3; 0 |]));
          r "an empty axis widened, periodic" (fun () ->
              Nx.extend `Periodic (0, 1) (Nx.zeros Nx.int32 [| 3; 0 |]));
          r "complex of the most axes, along 0" (complex_most_axes 0);
          r "complex of the most axes, along 1" (complex_most_axes 1);
          r "complex of the most axes, along the last"
            (complex_most_axes (A.Layout.max_rank - 1));
        ];
    ]

(* extension_index *)

(* The rules that read one sample, which {!Nx.extension_index} takes. *)
type copying =
  [ `Zero | `Replicate | `Symmetric_whole | `Symmetric_half | `Periodic ]

let copying : copying list =
  [ `Zero; `Replicate; `Symmetric_whole; `Symmetric_half; `Periodic ]

let positions lo n hi =
  Nx.create Nx.int64
    [| lo + n + hi |]
    (Array.init (lo + n + hi) (fun k -> Int64.of_int (k - lo)))

(* The sample a signal of [n] samples reads at [p] under [e], [-1] for a zero,
   by the rules' equations. [read] steps one fold a call, so [p] first moves to
   a position that reads the same: by whole periods, which the extension's doc
   states, or, past an edge where every position reads alike, to the one beside
   the edge. *)
let index_of (e : copying) n p =
  let near =
    match e with
    | (`Periodic | `Symmetric_half | `Symmetric_whole) when n > 1 ->
        let t =
          Int64.of_int
            (match e with
            | `Periodic -> n
            | `Symmetric_half -> 2 * n
            | _ -> 2 * (n - 1))
        in
        let r = Int64.rem p t in
        if r < 0L then Int64.add r t else r
    | _ -> Int64.max (-1L) (Int64.min p (Int64.of_int n))
  in
  Int64.of_int
    (read ~zero:(-1)
       ~opp:(fun _ _ -> assert false)
       (e :> Nx.extension)
       n Fun.id (Int64.to_int near))

(* Positions near either edge, past several folds of a short signal, and
   anywhere in [int64]. *)
let near_edges n =
  Gen.(
    frequency
      [
        (3, map Int64.of_int (int_range (-13) 12));
        (3, map (fun d -> Int64.of_int (n + d)) (int_range (-12) 12));
        ( 1,
          of_list
            [
              Int64.min_int;
              Int64.succ Int64.min_int;
              Int64.pred Int64.max_int;
              Int64.max_int;
            ] );
        (1, int64);
      ])

let law_index (e, n, p) =
  let past k =
    Int64.(compare k (of_int (-2 * n)) < 0 || compare k (of_int (3 * n)) > 0)
  in
  cover "an empty signal" (n = 0);
  cover "one sample" (n = 1);
  cover "a long signal" (n > 5);
  cover "folds past two periods" (n >= 2 && Array.exists past p);
  cover "an int64 extreme"
    (Array.exists (fun k -> k = Int64.min_int || k = Int64.max_int) p);
  let y () =
    Nx.extension_index e n (Nx.create Nx.int64 [| Array.length p |] p)
  in
  if n = 0 && e <> `Zero then invalid ~by:"Nx.extension_index" y
  else equal (array int64) (Array.map (index_of e n) p) (elements (y ()))

let law_take_index ((o, a), e, lo, hi) =
  let n = o.s.(a) in
  covers o;
  cover "one sample" (n = 1);
  cover "folds past two periods" (n >= 2 && (lo > 2 * n || hi > 2 * n));
  let y = Nx.extend ~axis:a (e :> Nx.extension) (lo, hi) o.x in
  let t = Nx.take ~axis:a (Nx.extension_index e n (positions lo n hi)) o.x in
  equal ~msg:"shape" (array int) (Nx.shape y) (Nx.shape t);
  equal ~msg:"elements" (array int32) (elements y) (elements t)

let pp_index_case ppf (e, n, p) =
  Format.fprintf ppf "%s %d [%s]"
    (rule_name (e :> Nx.extension))
    n
    (String.concat "; " (Array.to_list (Array.map Int64.to_string p)))

let test_index_shape () =
  let p = [| -1L; 3L; -4L; 5L; 0L; 7L |] in
  let y =
    Nx.extension_index `Symmetric_half 3 (Nx.create Nx.int64 [| 2; 3 |] p)
  in
  equal ~msg:"[2; 3]'s shape" (array int) [| 2; 3 |] (Nx.shape y);
  equal ~msg:"[2; 3]'s elements" (array int64)
    (Array.map (index_of `Symmetric_half 3) p)
    (elements y);
  List.iter
    (fun e ->
      equal
        ~msg:("[0; 4], " ^ rule_name (e :> Nx.extension))
        (array int) [| 0; 4 |]
        (Nx.shape (Nx.extension_index e 2 (Nx.zeros Nx.int64 [| 0; 4 |]))))
    copying;
  equal ~msg:"0-d" (array int64) [| 1L |]
    (elements (Nx.extension_index `Periodic 3 (Nx.scalar Nx.int64 (-2L))))

let extension_index =
  group "extension_index"
    [
      prop "folds each position by its rule's equations"
        ~examples:
          [
            (`Periodic, 0, [| 0L |]);
            (`Zero, 0, [| -1L; 0L; 1L |]);
            (`Symmetric_whole, 1, [| -3L; 0L; 4L |]);
            (`Symmetric_half, 1 lsl 40, [| -1L; Int64.max_int |]);
          ]
        (Gen.with_pp pp_index_case
           Gen.(
             let* e = of_list copying in
             let* n =
               frequency [ (6, int_range 0 5); (1, int_range 6 (1 lsl 60)) ]
             in
             let+ p = array ~size:(int_range 0 8) (near_edges n) in
             (e, n, p)))
        law_index;
      prop "reads with take as extend reads"
        ~examples:
          [
            ((operand_of [| 2; 0 |] C_order, 1), `Zero, 2, 3);
            ((operand_of [| 1 |] C_order, 0), `Periodic, 3, 2);
          ]
        Gen.(
          let* o, a = signal in
          let* e = of_list copying in
          let sig_ = if e = `Zero then (o, a) else nonempty (o, a) in
          let* lo = int_range 0 12 in
          let+ hi = int_range 0 12 in
          (sig_, e, lo, hi))
        law_take_index;
      test "keeps the positions' shape" test_index_shape;
      raises_naming "Nx.extension_index"
        [
          r "a negative length, zero" (fun () ->
              Nx.extension_index `Zero (-1) (Nx.zeros Nx.int64 [| 2 |]));
          r "a negative length, replicate" (fun () ->
              Nx.extension_index `Replicate (-1) (Nx.zeros Nx.int64 [| 2 |]));
          r "a negative length, symmetric_whole" (fun () ->
              Nx.extension_index `Symmetric_whole (-1)
                (Nx.zeros Nx.int64 [| 2 |]));
          r "a negative length, symmetric_half" (fun () ->
              Nx.extension_index `Symmetric_half (-1)
                (Nx.zeros Nx.int64 [| 2 |]));
          r "a negative length, periodic" (fun () ->
              Nx.extension_index `Periodic (-1) (Nx.zeros Nx.int64 [| 2 |]));
        ];
    ]

(* roll, tile, repeat *)

let roll =
  group "roll"
    [
      prop "shifts along an axis, wrapping, and the flat value without one"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let+ k = int_range (-7) 7 in
          (o, a, k))
        (fun (o, a, k) ->
          covers o;
          let d = o.s.(a) in
          is o.s
            (fun i ->
              let j = Array.copy i in
              j.(a) <- (((i.(a) - k) mod d) + d) mod d;
              at o j)
            (Nx.roll ~axis:a k o.x);
          let n = numel o.s in
          is o.s
            (fun i ->
              let p = position o.s i in
              o.e.((((p - k) mod n) + n) mod n))
            (Nx.roll k o.x));
      test "takes every k and an axis of max_numel" (fun () ->
          let big = Nx_array.Layout.max_numel in
          let x = Nx.zeros Nx.int32 [| 0; big |] in
          equal (array int) [| 0; big |] (Nx.shape (Nx.roll ~axis:1 1 x));
          equal (array int) [| 0; big |] (Nx.shape (Nx.roll ~axis:1 (-1) x));
          let o = operand_of [| 3 |] C_order in
          List.iter
            (fun k ->
              let k' = ((k mod 3) + 3) mod 3 in
              is o.s
                (fun i -> at o [| (i.(0) + 3 - k') mod 3 |])
                (Nx.roll ~axis:0 k o.x))
            [ min_int; min_int + 1; max_int; max_int - 1 ]);
      raises_naming "Nx.roll"
        [ r "an axis past the value's" (fun () -> Nx.roll ~axis:2 1 (x23 ())) ];
    ]

let tile =
  group "tile"
    [
      prop "repeats whole axes end to end"
        Gen.(
          let* o = operand () in
          let* extra = int_range 0 2 in
          let+ reps =
            array ~size:(constant (Array.length o.s + extra)) (int_range 0 3)
          in
          (o, reps))
        (fun (o, reps) ->
          covers o;
          let k = Array.length reps - Array.length o.s in
          let s0 = Array.append (Array.make k 1) o.s in
          is
            (Array.mapi (fun a n -> n * s0.(a)) reps)
            (fun i ->
              at o
                (Array.init (Array.length o.s) (fun a -> i.(k + a) mod o.s.(a))))
            (Nx.tile reps o.x));
      raises_naming "Nx.tile"
        [
          r "too few reps" (fun () -> Nx.tile [| 2 |] (x23 ()));
          r "a negative rep" (fun () -> Nx.tile [| 1; -1 |] (x23 ()));
          r "33 reps" (fun () -> Nx.tile (Array.make 33 1) (x23 ()));
          r "extents past 2^53" (fun () ->
              Nx.tile [| 1 lsl 27; 1 lsl 26 |] (x23 ()));
        ];
    ]

let repeat =
  group "repeat"
    [
      prop "repeats each element in place, and the flat value without an axis"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let* n = int_range 0 3 in
          let+ neg = bool in
          (o, a, n, neg))
        (fun (o, a, n, neg) ->
          covers o;
          let s' = Array.copy o.s in
          s'.(a) <- n * o.s.(a);
          is s'
            (fun i ->
              let j = Array.copy i in
              j.(a) <- i.(a) / n;
              at o j)
            (Nx.repeat ~axis:(spell o neg a) n o.x);
          is [| n * numel o.s |] (fun i -> o.e.(i.(0) / n)) (Nx.repeat n o.x));
      test "repeats [a; b] twice as [a; a; b; b]" (fun () ->
          equal (array int32) [| 1l; 1l; 2l; 2l |]
            (elements
               (Nx.repeat ~axis:0 2 (Nx.create Nx.int32 [| 2 |] [| 1l; 2l |]))));
      raises_naming "Nx.repeat"
        [
          r "a negative count" (fun () -> Nx.repeat (-1) (x23 ()));
          r "an axis past the value's" (fun () -> Nx.repeat ~axis:2 2 (x23 ()));
        ];
    ]

(* Views and copies *)

type int32_host = (int32, D.int32_elt, Nx.host) Nx.t

(* Movements a stride expresses of any operand of shape [2; 3; 4]. *)
let any_view : (string * (int32_host -> int32_host)) list =
  [
    ("broadcast_to", Nx.broadcast_to [| 5; 2; 3; 4 |]);
    ("unsqueeze", fun x -> Nx.unsqueeze ~axes:[ 0; -1 ] x);
    ("squeeze", fun x -> Nx.squeeze (Nx.unsqueeze ~axes:[ 1 ] x));
    ("transpose", fun x -> Nx.transpose x);
    ("moveaxis", Nx.moveaxis (-1) 0);
    ("swapaxes", Nx.swapaxes 0 (-1));
    ("flip", fun x -> Nx.flip x);
    ("sliding_window", fun x -> Nx.sliding_window ~axis:1 ~window:2 x);
    ("split", fun x -> List.nth (Nx.split ~axis:(-1) 2 x) 1);
    ("split_at", fun x -> List.nth (Nx.split_at ~axis:(-1) [ -2 ] x) 1);
  ]

(* Movements a stride expresses of a C-ordered operand of shape [2; 3; 4]. *)
let c_order_view : (string * (int32_host -> int32_host)) list =
  [
    ("reshape", Nx.reshape [| 6; 4 |]);
    ("flatten", fun x -> Nx.flatten x);
    ( "tile of a unit axis",
      fun x -> Nx.tile [| 3; 1 |] (Nx.reshape [| 1; 24 |] x) );
    ( "repeat of a unit axis",
      fun x -> Nx.repeat ~axis:0 3 (Nx.reshape [| 1; 24 |] x) );
  ]

let view_rows =
  List.map (fun (n, f) -> ("C order, " ^ n, C_order, f)) c_order_view
  @ List.concat_map
      (fun l ->
        List.map
          (fun (n, f) -> (Format.asprintf "%a, %s" pp_layout l n, l, f))
          any_view)
      [ C_order; Reversed; Axes_reversed; Every_other ]

let copies : (string * (int32_host -> int32_host)) list =
  [
    ("concatenate of one value", fun x -> Nx.concatenate ~axis:0 [ x ]);
    ("stack of one value", fun x -> Nx.stack [ x ]);
    ("pad by nothing", Nx.pad [| (0, 0); (0, 0) |] 0l);
    ("roll by 0", fun x -> Nx.roll ~axis:0 0 x);
    ("extend by nothing, zero", Nx.extend `Zero (0, 0));
    ("extend by nothing, periodic", Nx.extend `Periodic (0, 0));
    ("extend by (1, 2), symmetric_half", Nx.extend `Symmetric_half (1, 2));
  ]

let views =
  group "views and copies"
    [
      cases "a movement a stride expresses is over its operand's memory"
        ~id:(fun (n, _, _) -> n)
        view_rows
        (fun (_, l, f) ->
          let x = (operand_of [| 2; 3; 4 |] l).x in
          over x (f x));
      cases "a copy holds memory of its own" ~id:fst copies (fun (_, f) ->
          let x = x23 () in
          apart x (f x));
    ]

(* A movement of every dtype, against the reference over the dtype's values. *)
let dtypes =
  cases "every dtype moves"
    ~id:(fun (D.Any dt) -> D.name dt)
    D.all
    (fun (D.Any dt) ->
      let s = [| 2; 3; 2 |] in
      let data =
        Array.init 12 (fun k -> D.of_float dt (Float.of_int (k mod 7)))
      in
      let x = fst (laid Axes_reversed dt s data) in
      let y = Nx.flip ~axes:[ 1 ] (Nx.moveaxis 2 0 x) in
      let w = Testable.make ~pp:(D.pp_value dt) ~equal:( = ) in
      equal ~msg:"moveaxis, flip" (array w)
        (init [| 2; 2; 3 |] (fun i ->
             data.(position s [| 1 - i.(1); i.(2); i.(0) |])))
        (elements y);
      equal ~msg:"repeat" (array w)
        (init [| 2; 6; 2 |] (fun i ->
             data.(position s [| i.(0); i.(1) / 2; i.(2) |])))
        (elements (Nx.repeat ~axis:1 2 x));
      equal ~msg:"concatenate of split" (array w) data
        (elements (Nx.concatenate ~axis:1 (Nx.split ~axis:1 2 x)));
      equal ~msg:"pad" (array w)
        (init [| 2; 5; 2 |] (fun i ->
             if i.(1) < 2 then D.zero dt
             else data.(position s [| i.(0); i.(1) - 2; i.(2) |])))
        (elements (Nx.pad [| (0, 0); (2, 0); (0, 0) |] (D.zero dt) x)))

(* The words of a few raises: the function, the reason and the operand. *)
let messages =
  test "raises read as their reason and operand" (fun () ->
      let x = x23 () in
      let said f =
        match ignore (f ()) with
        | () -> "raised nothing"
        | exception Invalid_argument m -> m
      in
      expect
        (String.concat "\n"
           [
             said (fun () -> Nx.reshape [| 4; 2 |] x);
             said (fun () -> Nx.reshape [| max_int; 2 |] x);
             said (fun () -> Nx.reshape [| -2; 3 |] x);
             said (fun () -> Nx.reshape [| -1; -1 |] x);
             said (fun () -> Nx.reshape [| 4; -1 |] x);
             said (fun () -> Nx.broadcast_to [| 2; 4 |] x);
             said (fun () ->
                 Nx.broadcast_shapes [ [| 2; 1 |]; [| 1; 3 |]; [| 4; 3 |] ]);
             said (fun () -> Nx.broadcast_shapes [ [| 2 |]; [| -1 |] ]);
             said (fun () ->
                 Nx.broadcast_arrays [ x; Nx.zeros Nx.int32 [| 4 |] ]);
             said (fun () -> Nx.squeeze ~axes:[ 1 ] x);
             said (fun () -> Nx.transpose ~axes:[ 0 ] x);
             said (fun () -> Nx.transpose ~axes:[ 0; 0 ] x);
             said (fun () -> Nx.dim 2 x);
             said (fun () -> Nx.split ~axis:0 0 x);
             said (fun () -> Nx.split_at ~axis:1 [ 2; 1 ] x);
             said (fun () -> Nx.split_at ~axis:1 [ -1; 1 ] x);
             said (fun () -> Nx.tile [| 2 |] x);
             said (fun () -> Nx.pad [| (0, 0); (max_int, 0) |] 0l x);
           ])
      @@ __POS_OF__
           {|
           Nx.reshape: int32 [2; 3] has 6 elements, [4; 2] has 8
           Nx.reshape: int32 [2; 3] would have more elements than an int counts
           Nx.reshape: extent -2 in [-2; 3]
           Nx.reshape: [-1; -1] has two unknown extents
           Nx.reshape: int32 [2; 3] has 6 elements, which [4; -1] cannot hold
           Nx.broadcast_to: int32 [2; 3] does not broadcast to [2; 4]
           Nx.broadcast_shapes: [4; 3] does not broadcast with [2; 1], [1; 3]: axis -2 has 4, neither 1 nor 2
           Nx.broadcast_shapes: [-1] has a negative extent
           Nx.broadcast_arrays: [4] does not broadcast with [2; 3]: axis -1 has 4, neither 1 nor 3
           Nx.squeeze: axis 1 of int32 [2; 3] has extent 3
           Nx.transpose: axes [0] are not a permutation of int32 [2; 3]'s
           Nx.transpose: axis 0 of int32 [2; 3] repeats
           Nx.dim: 2 is not an axis of int32 [2; 3]
           Nx.split: 0 runs of int32 [2; 3]; give at least 1
           Nx.split_at: offset 1 cuts at 1, before offset 2 at 2, along axis 1 of int32 [2; 3]
           Nx.split_at: offset 1 cuts at 1, before offset -1 at 2, along axis 1 of int32 [2; 3]
           Nx.tile: reps [2] has fewer entries than int32 [2; 3] has axes
           Nx.pad: widths (4611686018427387903, 0) of axis 1 of int32 [2; 3] count past an int
           |})

(* Prim.eval: a movement or a bitcast through the frontend is its operation's
   meaning. *)

(* [mv] of [x] through the frontend that makes it. *)
let moved mv x =
  match (mv : M.t) with
  | Reshape s -> Nx.reshape (Iarray.to_array s) x
  | Permute p -> Nx.transpose ~axes:(Array.to_list (Iarray.to_array p)) x
  | Slice rs ->
      Nx.slice
        (Array.to_list
           (Array.map
              (fun (r : M.range) ->
                if r.step = 1 then Nx.A else Nx.Rs (-1, -r.count - 1, -1))
              (Iarray.to_array rs)))
        x
  | Broadcast s -> Nx.broadcast_to (Iarray.to_array s) x
  | Window _ -> invalid_arg "moved: a window"

(* A movement of a drawn operand: a reshape to its axes reversed or to one axis,
   a permutation, a reversing slice or a broadcast on a new leading axis. *)
let movement o =
  let open Gen in
  let r = Array.length o.s in
  let ranges rev =
    Array.map
      (fun d : M.range ->
        if rev then { start = max 0 (d - 1); count = d; step = -1 }
        else { start = 0; count = d; step = 1 })
      o.s
  in
  let+ k = int_range 0 4 and+ rev = bool and+ lead = int_range 0 3 in
  match k with
  | 0 -> M.Reshape (Iarray.of_array (Array.map (fun a -> o.s.(a)) (rev_axes r)))
  | 1 -> M.Reshape [| numel o.s |]
  | 2 -> M.Permute (Iarray.of_array (rev_axes r))
  | 3 -> M.Slice (Iarray.of_array (ranges rev))
  | _ -> M.Broadcast (Iarray.of_array (Array.append [| lead |] o.s))

let layout_w = Testable.make ~pp:A.Layout.pp ~equal:A.Layout.equal

(* What a result is to its operand [x], and its layout. *)
let made x y = ((if shares x y then "view" else "copy"), A.layout (held y))

let outcome f =
  match f () with v -> Ok v | exception Invalid_argument m -> Error m

let law_bitcast_eval o =
  covers o;
  let agree (type w r) name (dt : (w, r) D.t) =
    let seen r = Result.map (fun v -> (Nx.shape v, made o.x v)) r in
    let w = result (pair (array int) (pair string layout_w)) string in
    equal ~msg:name w
      (seen
         (outcome (fun () -> Nx.Prim.eval ~by:"Nx.bitcast" (Bitcast (dt, o.x)))))
      (seen (outcome (fun () -> Nx.bitcast dt o.x)))
  in
  agree "same width" D.Float32;
  agree "narrower" D.Uint8;
  agree "wider" D.Int64

let law_move_eval (o, mv) =
  covers o;
  let y = moved mv o.x in
  let z = Nx.Prim.eval ~by:"test" (Move (mv, o.x)) in
  cover "a view" (shares o.x z);
  cover "a copy" (not (shares o.x z));
  equal ~msg:"shape" (array int) (Nx.shape z) (Nx.shape y);
  equal ~msg:"elements" (array int32) (elements z) (elements y);
  equal ~msg:"view or copy, and layout" (pair string layout_w) (made o.x z)
    (made o.x y)

let eval =
  group "Prim.eval"
    [
      prop "a bitcast of a live array is its operation's meaning" (operand ())
        law_bitcast_eval;
      prop "a movement of a live array is its operation's meaning"
        Gen.(
          let* o = operand () in
          let+ mv = movement o in
          (o, mv))
        law_move_eval;
    ]

let () =
  exit
    (run "nx shapes"
       [
         properties;
         reshape;
         broadcast_to;
         broadcast_shapes;
         broadcast_arrays;
         broadcasting;
         meshgrid;
         squeeze;
         unsqueeze;
         flatten;
         transpose;
         moveaxis;
         flip;
         sliding_window;
         split;
         split_at;
         concatenate;
         stack;
         pad;
         extension;
         extend;
         extension_index;
         roll;
         tile;
         repeat;
         views;
         dtypes;
         messages;
         eval;
       ])
