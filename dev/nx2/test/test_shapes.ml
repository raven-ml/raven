(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shapes and movements through Nx: each movement against a reference that reads
   an OCaml array at mapped indices, over operands laid out plain, reversed,
   transposed and broadcast; the round trips; the views; every dtype; the
   refusals. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f
let elements x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

(* The reference: a shape and its elements in C order. *)

let numel s = Array.fold_left ( * ) 1 s

(* The index of [s] at C-order position [k]. *)
let index s k =
  let r = Array.length s in
  let idx = Array.make r 0 and k = ref k in
  for a = r - 1 downto 0 do
    idx.(a) <- !k mod s.(a);
    k := !k / s.(a)
  done;
  idx

let position s idx =
  let k = ref 0 in
  Array.iteri (fun a i -> k := (!k * s.(a)) + i) idx;
  !k

(* The elements of shape [s] whose element at [i] is [f i]. *)
let init s f = Array.init (numel s) (fun k -> f (index s k))

(* Operands: elements laid out C-contiguous, reversed along every axis (negative
   strides), with their axes reversed, or broadcast from one element. *)
type layout = Plain | Reversed | Transposed | Broadcast

let pp_layout ppf l =
  Format.pp_print_string ppf
    (match l with
    | Plain -> "plain"
    | Reversed -> "reversed"
    | Transposed -> "transposed"
    | Broadcast -> "broadcast")

let rev_axes r = Array.init r (fun i -> r - 1 - i)

(* [data] of shape [s] in C order, laid out as [l], on the host. *)
let lay (type v s) l (dt : (v, s) D.t) s (data : v array) : (v, s) A.t =
  let r = Array.length s in
  let view m a = Option.get (A.move m a) in
  match l with
  | Plain -> A.of_array dt s data
  | Reversed ->
      let whole =
        Array.map
          (fun d : M.range -> { start = max 0 (d - 1); count = d; step = -1 })
          s
      in
      view (Slice whole)
        (A.of_array dt s (Array.of_list (List.rev (Array.to_list data))))
  | Transposed ->
      let t = Array.map (fun a -> s.(a)) (rev_axes r) in
      let base =
        init t (fun j ->
            data.(position s (Array.map (fun a -> j.(a)) (rev_axes r))))
      in
      view (Permute (rev_axes r)) (A.of_array dt t base)
  | Broadcast ->
      if Array.length data = 0 then A.of_array dt s data
      else view (Broadcast s) (A.of_array dt [||] [| data.(0) |])

(* What [lay] reads back: a broadcast holds its first element everywhere. *)
let read l data =
  match l with
  | Broadcast when Array.length data > 0 -> Array.map (fun _ -> data.(0)) data
  | _ -> data

(* A drawn operand: its shape, layout and elements, distinct unless broadcast,
   and the value. *)
type operand = {
  s : int array;
  l : layout;
  e : int32 array;
  x : (int32, D.int32_elt, Nx.host) Nx.t;
}

let operand_of s l =
  let data = Array.init (numel s) (fun k -> Int32.of_int (k + 1)) in
  {
    s;
    l;
    e = read l data;
    x = Nx.Repr.of_array Nx.Host.v (lay l D.Int32 s data);
  }

let operand ?(min_rank = 0) () =
  let open Gen in
  with_pp
    (fun ppf o -> Format.fprintf ppf "%a %a" pp_layout o.l pp_ints o.s)
    (let* rank = int_range min_rank 4 in
     let* s = array ~size:(constant rank) (int_range 0 3) in
     let+ l =
       of_list ~pp:pp_layout [ Plain; Reversed; Transposed; Broadcast ]
     in
     operand_of s l)

let covers o =
  cover "no element" (numel o.s = 0);
  cover "one element" (numel o.s = 1);
  cover "strided" (o.l = Reversed || o.l = Transposed)

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

let laws =
  group "laws"
    [
      prop "reshape keeps the elements in row-major order" (operand ())
        (fun o ->
          covers o;
          let n = numel o.s in
          is [| n |] (fun i -> o.e.(i.(0))) (Nx.reshape [| n |] o.x);
          let r = rev_axes (Array.length o.s) in
          let s' = Array.map (fun a -> o.s.(a)) r in
          is s' (fun i -> o.e.(position s' i)) (Nx.reshape s' o.x));
      prop "reshape infers its one unknown extent" (operand ~min_rank:1 ())
        (fun o ->
          assume (numel (Array.sub o.s 1 (Array.length o.s - 1)) > 0);
          let s' = Array.copy o.s in
          s'.(0) <- -1;
          equal (array int) o.s (Nx.shape (Nx.reshape s' o.x)));
      prop "broadcast_to repeats along new and unit axes"
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
      prop "transpose moves each axis where its permutation says"
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
      prop "moveaxis and swapaxes are permutations"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let+ b = axis_of o in
          (o, a, b))
        (fun (o, a, b) ->
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
      prop "flip reverses each axis it names"
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
      prop "squeeze undoes unsqueeze"
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
      prop "squeeze drops every unit axis" (operand ()) (fun o ->
          let y = Nx.squeeze o.x in
          equal (array int)
            (Array.of_list (List.filter (( <> ) 1) (Array.to_list o.s)))
            (Nx.shape y);
          equal (array int32) o.e (elements y));
      prop "flatten merges a run of axes"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let* b = int_range a (Array.length o.s - 1) in
          let+ neg = bool in
          (o, a, b, neg))
        (fun (o, a, b, neg) ->
          let y =
            Nx.flatten ~start_dim:(spell o neg a) ~end_dim:(spell o neg b) o.x
          in
          let r = Array.length o.s in
          equal (array int)
            (Array.concat
               [
                 Array.sub o.s 0 a;
                 [| numel (Array.sub o.s a (b - a + 1)) |];
                 Array.sub o.s (b + 1) (r - b - 1);
               ])
            (Nx.shape y);
          equal (array int32) o.e (elements y);
          equal (array int) [| numel o.s |] (Nx.shape (Nx.flatten o.x)));
      prop "sliding_window reads each window's elements"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let d = o.s.(a) in
          let* window = int_range 1 (max 1 d) in
          let* step = int_range 1 3 in
          let+ neg = bool in
          (o, a, window, step, neg))
        (fun (o, a, window, step, neg) ->
          assume (window <= o.s.(a));
          covers o;
          let y = Nx.sliding_window ~axis:(spell o neg a) ~window ~step o.x in
          let r = Array.length o.s in
          let s' = Array.append o.s [| window |] in
          s'.(a) <- ((o.s.(a) - window) / step) + 1;
          is s'
            (fun i ->
              let j = Array.sub i 0 r in
              j.(a) <- (i.(a) * step) + i.(r);
              at o j)
            y);
      prop "split cuts runs whose lengths differ by at most one"
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
      prop "tile repeats whole axes end to end"
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
      prop "repeat repeats each element in place"
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
    ]

(* Joining, padding and rolling: copies that assemble pieces. *)
let joins =
  group "joins"
    [
      prop "concatenate undoes split"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let+ n = int_range 1 4 in
          (o, a, n))
        (fun (o, a, n) ->
          covers o;
          is o.s (at o) (Nx.concatenate ~axis:a (Nx.split ~axis:a n o.x)));
      prop "stack puts each value at its position along the new axis"
        Gen.(
          let* o = operand () in
          let* a = int_range 0 (Array.length o.s) in
          let+ n = int_range 1 3 in
          (o, a, n))
        (fun (o, a, n) ->
          covers o;
          let xs =
            List.init n (fun k ->
                Nx.flip ~axes:[]
                  (Nx.add o.x (Nx.scalar Nx.int32 (Int32.of_int (100 * k)))))
          in
          let r = Array.length o.s in
          let s' =
            Array.init (r + 1) (fun i ->
                if i < a then o.s.(i) else if i = a then n else o.s.(i - 1))
          in
          is s'
            (fun i ->
              let j =
                Array.init r (fun d -> if d < a then i.(d) else i.(d + 1))
              in
              Int32.add (at o j) (Int32.of_int (100 * i.(a))))
            (Nx.stack ~axis:a xs));
      prop "pad surrounds the elements with its value"
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
      prop "roll shifts along an axis, wrapping"
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
      test "join refusals name the function" (fun () ->
          let x = (operand_of [| 2; 3 |] Plain).x in
          invalid ~by:"Nx.concatenate" (fun () -> Nx.concatenate ~axis:0 []);
          invalid ~by:"Nx.concatenate" (fun () ->
              Nx.concatenate ~axis:0 [ x; Nx.transpose x ]);
          invalid ~by:"Nx.stack" (fun () -> Nx.stack ~axis:3 [ x ]);
          invalid ~by:"Nx.stack" (fun () ->
              Nx.stack [ x; Nx.reshape [| 3; 2 |] x ]);
          invalid ~by:"Nx.pad" (fun () -> Nx.pad [| (1, 1) |] 0l x);
          invalid ~by:"Nx.pad" (fun () -> Nx.pad [| (0, 0); (-1, 0) |] 0l x);
          invalid ~by:"Nx.pad" (fun () ->
              Nx.pad [| (1, 0) |] 256 (Nx.zeros Nx.uint8 [| 2 |]));
          invalid ~by:"Nx.roll" (fun () -> Nx.roll ~axis:2 1 x));
    ]

let properties =
  group "properties"
    [
      test "ndim, dim, numel and nbytes read the shape and dtype" (fun () ->
          let x = Nx.zeros Nx.int4 [| 2; 3; 5 |] in
          equal int 3 (Nx.ndim x);
          equal int 3 (Nx.dim 1 x);
          equal int 5 (Nx.dim (-1) x);
          equal int 30 (Nx.numel x);
          equal int 15 (Nx.nbytes x);
          equal int 1 (Nx.numel (Nx.scalar Nx.float64 1.));
          equal int 2 (Nx.nbytes (Nx.zeros Nx.bit [| 9 |])));
      test "broadcast_shapes aligns from the last axes" (fun () ->
          equal (array int) [| 2; 4; 3 |]
            (Nx.broadcast_shapes [ [| 4; 1 |]; [| 2; 1; 3 |]; [||] ]);
          equal (array int) [||] (Nx.broadcast_shapes []);
          equal (array int) [| 0; 3 |]
            (Nx.broadcast_shapes [ [| 0; 1 |]; [| 3 |] ]));
      test "broadcast_arrays stretches each to the common shape" (fun () ->
          let a = (operand_of [| 2; 1 |] Plain).x
          and b = (operand_of [| 3 |] Reversed).x in
          match Nx.broadcast_arrays [ a; b ] with
          | [ a'; b' ] ->
              equal (array int32) [| 1l; 1l; 1l; 2l; 2l; 2l |] (elements a');
              equal (array int32) [| 1l; 2l; 3l; 1l; 2l; 3l |] (elements b')
          | _ -> fail "two values");
    ]

(* Whether [y]'s memory is [x]'s. *)
let shares x y =
  let buffer v = A.buffer (Option.get (Nx.Repr.array v)) in
  Rig.Buffer.overlaps (buffer x) (buffer y)

let views =
  group "views"
    [
      test "movements a stride expresses share their operand's memory"
        (fun () ->
          let x = (operand_of [| 2; 3; 4 |] Plain).x in
          let each name y = equal ~msg:name bool true (shares x y) in
          each "reshape" (Nx.reshape [| 6; 4 |] x);
          each "broadcast_to" (Nx.broadcast_to [| 5; 2; 3; 4 |] x);
          each "squeeze" (Nx.squeeze (Nx.reshape [| 1; 24 |] x));
          each "unsqueeze" (Nx.unsqueeze ~axes:[ 0; 2 ] x);
          each "flatten" (Nx.flatten x);
          each "transpose" (Nx.transpose x);
          each "moveaxis" (Nx.moveaxis 0 2 x);
          each "swapaxes" (Nx.swapaxes 0 1 x);
          each "flip" (Nx.flip ~axes:[ 1 ] x);
          each "sliding_window" (Nx.sliding_window ~window:2 x);
          each "split" (List.nth (Nx.split ~axis:2 3 x) 2);
          each "tile of a unit axis"
            (Nx.tile [| 3; 1 |] (Nx.reshape [| 1; 24 |] x));
          each "repeat of a unit axis"
            (Nx.repeat ~axis:0 3 (Nx.reshape [| 1; 24 |] x)));
      cases
        ~name:(fun l -> Format.asprintf "%a" pp_layout l)
        "a permutation, flip, broadcast or window of a strided operand is a \
         view"
        [ Reversed; Transposed ]
        (fun l ->
          let x = (operand_of [| 2; 3; 4 |] l).x in
          let each name y = equal ~msg:name bool true (shares x y) in
          each "broadcast_to" (Nx.broadcast_to [| 5; 2; 3; 4 |] x);
          each "unsqueeze" (Nx.unsqueeze ~axes:[ 0; -1 ] x);
          each "squeeze" (Nx.squeeze (Nx.unsqueeze ~axes:[ 1 ] x));
          each "transpose" (Nx.transpose x);
          each "moveaxis" (Nx.moveaxis (-1) 0 x);
          each "swapaxes" (Nx.swapaxes 0 (-1) x);
          each "flip" (Nx.flip x);
          each "sliding_window" (Nx.sliding_window ~axis:1 ~window:2 x);
          each "split" (List.nth (Nx.split ~axis:(-1) 3 x) 1));
      test "a reshape no stride expresses copies" (fun () ->
          let x = Nx.transpose (operand_of [| 2; 3 |] Plain).x in
          let y = Nx.reshape [| 6 |] x in
          equal bool false (shares x y);
          equal (array int32) [| 1l; 4l; 2l; 5l; 3l; 6l |] (elements y));
    ]

(* A movement of every dtype, against the reference over the dtype's values. *)
let dtypes =
  cases
    ~name:(fun (D.Any dt) -> D.name dt)
    "every dtype" D.all
    (fun (D.Any dt) ->
      let s = [| 2; 3; 2 |] in
      let data =
        Array.init 12 (fun k -> D.of_float dt (Float.of_int (k mod 7)))
      in
      let x = Nx.Repr.of_array Nx.Host.v (lay Transposed dt s data) in
      let y = Nx.flip ~axes:[ 1 ] (Nx.moveaxis 2 0 x) in
      let w = Testable.make ~pp:(D.pp_value dt) ~equal:( = ) in
      equal (array w)
        (init [| 2; 2; 3 |] (fun i ->
             data.(position s [| 1 - i.(1); i.(2); i.(0) |])))
        (elements y);
      equal (array w)
        (init [| 2; 6; 2 |] (fun i ->
             data.(position s [| i.(0); i.(1) / 2; i.(2) |])))
        (elements (Nx.repeat ~axis:1 2 x));
      equal (array w) data
        (elements (Nx.concatenate ~axis:1 (Nx.split ~axis:1 2 x)));
      equal (array w)
        (init [| 2; 5; 2 |] (fun i ->
             if i.(1) < 2 then D.zero dt
             else data.(position s [| i.(0); i.(1) - 2; i.(2) |])))
        (elements (Nx.pad [| (0, 0); (2, 0); (0, 0) |] (D.zero dt) x)))

(* A refusal: its case's name starts with the function it calls. *)
let r name f = (name, fun () -> ignore (f ()))

let refusals =
  let x = (operand_of [| 2; 3 |] Plain).x in
  group "refusals"
    [
      test "messages name the function and the operand" (fun () ->
          raises
            (Invalid_argument
               "Nx.reshape: int32 [2; 3] has 6 elements, [4; 2] has 8")
            (fun () -> Nx.reshape [| 4; 2 |] x);
          raises
            (Invalid_argument
               "Nx.broadcast_to: int32 [2; 3] does not broadcast to [2; 4]")
            (fun () -> Nx.broadcast_to [| 2; 4 |] x);
          raises
            (Invalid_argument "Nx.squeeze: axis 1 of int32 [2; 3] has extent 3")
            (fun () -> Nx.squeeze ~axes:[ 1 ] x);
          raises
            (Invalid_argument
               "Nx.transpose: axes [0] are not a permutation of int32 [2; 3]'s")
            (fun () -> Nx.transpose ~axes:[ 0 ] x);
          raises
            (Invalid_argument "Nx.transpose: axis 0 of int32 [2; 3] repeats")
            (fun () -> Nx.transpose ~axes:[ 0; 0 ] x);
          raises (Invalid_argument "Nx.dim: 2 is not an axis of int32 [2; 3]")
            (fun () -> Nx.dim 2 x);
          raises
            (Invalid_argument
               "Nx.broadcast_shapes: [4; 3] does not broadcast with [2; 1], \
                [1; 3]: axis -2 has 4, neither 1 nor 2") (fun () ->
              Nx.broadcast_shapes [ [| 2; 1 |]; [| 1; 3 |]; [| 4; 3 |] ]);
          raises
            (Invalid_argument
               "Nx.reshape: int32 [2; 3] would have more elements than an int \
                counts") (fun () -> Nx.reshape [| max_int; 2 |] x);
          raises
            (Invalid_argument "Nx.broadcast_shapes: [-1] has a negative extent")
            (fun () -> Nx.broadcast_shapes [ [| 2 |]; [| -1 |] ]);
          raises
            (Invalid_argument
               "Nx.broadcast_arrays: [4] does not broadcast with [2; 3]: axis \
                -1 has 4, neither 1 nor 3") (fun () ->
              Nx.broadcast_arrays [ x; Nx.zeros Nx.int32 [| 4 |] ]);
          raises
            (Invalid_argument
               "Nx.split: 0 runs of int32 [2; 3]; give at least 1") (fun () ->
              Nx.split ~axis:0 0 x);
          raises
            (Invalid_argument
               "Nx.tile: reps [2] has fewer entries than int32 [2; 3] has axes")
            (fun () -> Nx.tile [| 2 |] x));
      cases ~name:fst "each refusal raises"
        [
          r "reshape, two unknown extents" (fun () -> Nx.reshape [| -1; -1 |] x);
          r "reshape, an extent below -1" (fun () -> Nx.reshape [| -2; -3 |] x);
          r "reshape, an unknown that does not divide" (fun () ->
              Nx.reshape [| 4; -1 |] x);
          r "broadcast_to, fewer axes" (fun () -> Nx.broadcast_to [| 3 |] x);
          r "squeeze, a repeated axis" (fun () ->
              Nx.squeeze ~axes:[ 0; -2 ] (Nx.reshape [| 1; 6 |] x));
          r "unsqueeze, past the result" (fun () -> Nx.unsqueeze ~axes:[ 3 ] x);
          r "unsqueeze, a repeated position" (fun () ->
              Nx.unsqueeze ~axes:[ 0; -4 ] x);
          r "flatten, start after end" (fun () ->
              Nx.flatten ~start_dim:1 ~end_dim:0 x);
          r "transpose, too few axes" (fun () -> Nx.transpose ~axes:[ 0 ] x);
          r "moveaxis, not an axis" (fun () -> Nx.moveaxis 0 2 x);
          r "swapaxes, not an axis" (fun () -> Nx.swapaxes (-3) 0 x);
          r "flip, a repeated axis" (fun () -> Nx.flip ~axes:[ 1; -1 ] x);
          r "sliding_window, too wide" (fun () -> Nx.sliding_window ~window:4 x);
          r "sliding_window, step 0" (fun () ->
              Nx.sliding_window ~window:1 ~step:0 x);
          r "tile, too few reps" (fun () -> Nx.tile [| 2 |] x);
          r "tile, a negative rep" (fun () -> Nx.tile [| 1; -1 |] x);
          r "repeat, a negative count" (fun () -> Nx.repeat (-1) x);
          r "repeat, not an axis" (fun () -> Nx.repeat ~axis:2 2 x);
        ]
        (fun (name, f) ->
          let by = "Nx." ^ String.sub name 0 (String.index name ',') in
          invalid ~by (fun () -> ignore (f ())));
      test "split refuses no runs" (fun () ->
          invalid ~by:"Nx.split" (fun () -> Nx.split ~axis:0 0 x));
    ]

let donation =
  group "donation"
    [
      test "a one-to-one movement passes the handle to the final consumer"
        (fun () ->
          let x =
            Nx.add (operand_of [| 2; 3 |] Plain).x
              (Nx.zeros Nx.int32 [| 2; 3 |])
          in
          let d = Nx.flip (Nx.transpose (Nx.donate x)) in
          equal ~msg:"the donor lives" (array int32)
            [| 1l; 2l; 3l; 4l; 5l; 6l |]
            (elements x);
          equal (array int32)
            [| 6l; 3l; 5l; 2l; 4l; 1l |]
            (elements (Nx.copy d));
          raises_match (Exn.invalid_arg ~substring:"was donated to Nx.copy")
            (fun () -> Nx.copy x));
    ]

let () =
  exit
    (run "nx shapes"
       [ laws; joins; properties; views; dtypes; refusals; donation ])
