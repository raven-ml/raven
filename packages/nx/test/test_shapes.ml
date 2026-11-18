(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shapes and movements through Nx: each movement against a reference that reads
   an OCaml array at mapped indices, over operands laid out plain, reversed,
   transposed and broadcast; the round trips; the views; the extensions;
   every dtype; the refusals. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f
let elements x =
  A.to_array (Option.get (Nx.Repr.array ~by:"test" (Nx.place Nx.Host.on x)))

(* The reference: a shape and its elements in C order. *)

let numel s = Array.fold_left ( * ) 1 s

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
  | Plain -> A.of_array dt (Iarray.of_array s) data
  | Reversed ->
      let whole =
        Array.map
          (fun d : M.range -> { start = max 0 (d - 1); count = d; step = -1 })
          s
      in
      view
        (Slice (Iarray.of_array whole))
        (A.of_array dt (Iarray.of_array s)
           (Array.of_list (List.rev (Array.to_list data))))
  | Transposed ->
      let t = Array.map (fun a -> s.(a)) (rev_axes r) in
      let base =
        init t (fun j ->
            data.(position s (Array.map (fun a -> j.(a)) (rev_axes r))))
      in
      view
        (Permute (Iarray.of_array (rev_axes r)))
        (A.of_array dt (Iarray.of_array t) base)
  | Broadcast ->
      if Array.length data = 0 then A.of_array dt (Iarray.of_array s) data
      else
        view (Broadcast (Iarray.of_array s)) (A.of_array dt [||] [| data.(0) |])

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
     (* One draw in four has one element, which a uniform draw of extents
        gives in only 8% of draws of rank 1 to 4. *)
     let* s =
       frequency
         [
           (1, constant (Array.make rank 1));
           (3, array ~size:(constant rank) (int_range 0 3));
         ]
     in
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

(* A movement of a drawn operand: a reshape to its axes reversed or to
   one axis, a permutation, a reversing slice or a broadcast on a new
   leading axis. *)
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

let laws =
  group "laws"
    [
      prop "a bitcast of a live array is its operation's meaning" (operand ())
        (fun o ->
          covers o;
          let held v = Option.get (Nx.Repr.array ~by:"test" v) in
          let shares v =
            Rig.Buffer.overlaps (A.buffer (held o.x)) (A.buffer (held v))
          in
          let run f =
            match f () with v -> Ok v | exception Invalid_argument m -> Error m
          in
          let agree (type w r) name (dt : (w, r) D.t) =
            let y = run (fun () -> Nx.bitcast dt o.x) in
            let z =
              run (fun () ->
                  Nx.Prim.eval ~by:"Nx.bitcast" (Bitcast (dt, o.x)))
            in
            match (y, z) with
            | Ok y, Ok z ->
                equal ~msg:(name ^ ": shape") (array int) (Nx.shape z)
                  (Nx.shape y);
                equal ~msg:(name ^ ": shares") bool (shares z) (shares y);
                equal ~msg:(name ^ ": layout") bool true
                  (A.Layout.equal (A.layout (held z)) (A.layout (held y)))
            | Error a, Error b ->
                equal ~msg:(name ^ ": message") string b a
            | Ok _, Error m -> fail (name ^ ": only eval raised " ^ m)
            | Error m, Ok _ -> fail (name ^ ": only the frontend raised " ^ m)
          in
          agree "same width" D.Float32;
          agree "narrower" D.Uint8;
          agree "wider" D.Int64);
      prop "a movement of a live array is its operation's meaning"
        Gen.(
          let* o = operand () in
          let+ mv = movement o in
          (o, mv))
        (fun (o, mv) ->
          covers o;
          let y = moved mv o.x in
          let z = Nx.Prim.eval ~by:"test" (Move (mv, o.x)) in
          let held v = Option.get (Nx.Repr.array ~by:"test" v) in
          let overlaps a b =
            Rig.Buffer.overlaps (A.buffer (held a)) (A.buffer (held b))
          in
          cover "a view" (overlaps o.x z);
          cover "a copy" (not (overlaps o.x z));
          equal ~msg:"shape" (array int) (Nx.shape z) (Nx.shape y);
          equal ~msg:"elements" (array int32) (elements z) (elements y);
          equal ~msg:"shares" bool (overlaps o.x z) (overlaps o.x y);
          equal ~msg:"layout" bool true
            (A.Layout.equal (A.layout (held z)) (A.layout (held y))));
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
      prop "split_at cuts before each offset"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let d = o.s.(a) in
          let* offsets =
            list ~size:(int_range 0 3) (int_range (-d - 2) (d + 2))
          in
          let+ neg = bool in
          (o, a, offsets, neg))
        (fun (o, a, offsets, neg) ->
          covers o;
          let d = o.s.(a) in
          (* Where each offset cuts: counted from the end if negative, then
             clipped into [0, d]. *)
          let place k =
            Stdlib.max 0 (Stdlib.min d (if k < 0 then k + d else k))
          in
          let by_place j k = compare (place j) (place k) in
          let ordered = List.stable_sort by_place offsets = offsets in
          cover "offsets in order" ordered;
          cover "offsets out of order" (not ordered);
          let cuts = (0 :: List.map place offsets) @ [ d ] in
          cover "no offset" (ordered && offsets = []);
          cover "a negative offset"
            (ordered && List.exists (fun k -> k < 0) offsets);
          cover "an offset outside the axis"
            (ordered && List.exists (fun k -> k < -d || k > d) offsets);
          cover "an empty run"
            (ordered
            && List.exists2 ( = ) (List.filteri (fun i _ -> i > 0) cuts)
                 (List.filteri (fun i _ -> i < List.length cuts - 1) cuts));
          let axis = spell o neg a in
          if not ordered then
            raises_match (Exn.invalid_arg ~sub:"Nx.split_at: offset ")
              (fun () -> Nx.split_at ~axis offsets o.x)
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
          end);
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
      prop "concatenate undoes split_at"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let+ offsets = list ~size:(int_range 0 3) (int_range 0 o.s.(a)) in
          (o, a, List.sort compare offsets))
        (fun (o, a, offsets) ->
          covers o;
          is o.s (at o)
            (Nx.concatenate ~axis:a (Nx.split_at ~axis:a offsets o.x)));
      prop "stack puts each value at its position along the new axis"
        Gen.(
          let* o = operand () in
          let* a = int_range 0 (Array.length o.s) in
          let+ n = int_range 1 3 in
          (o, a, n))
        (fun (o, a, n) ->
          covers o;
          (* CR: Nx.add makes every input contiguous, and flip ~axes:[] keeps
             that layout, so covers o does not describe what stack receives.
             Shift o.e in OCaml, then use lay and Repr.of_array with this oracle.
             Cover the constructed layouts; seed ~examples with empty and unit
             shapes and [2;3] reversed, transposed and broadcast inputs, each
             with two distinct values, so coverage needs no lucky draw. *)
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
      test "roll takes every k and an axis of max_numel" (fun () ->
          let big = Nx_array.Layout.max_numel in
          let x = Nx.zeros Nx.int32 [| 0; big |] in
          equal (array int) [| 0; big |] (Nx.shape (Nx.roll ~axis:1 1 x));
          equal (array int) [| 0; big |] (Nx.shape (Nx.roll ~axis:1 (-1) x));
          let o = operand_of [| 3 |] Plain in
          List.iter
            (fun k ->
              let k' = ((k mod 3) + 3) mod 3 in
              is o.s
                (fun i -> at o [| (i.(0) + 3 - k') mod 3 |])
                (Nx.roll ~axis:0 k o.x))
            [ min_int; min_int + 1; max_int; max_int - 1 ]);
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
  let buffer v = A.buffer (Option.get (Nx.Repr.array ~by:"test" v)) in
  Rig.Buffer.overlaps (buffer x) (buffer y)

(* Grids *)

(* Up to three 1-d operands, each laid out as drawn. *)
let grid_case =
  let open Gen in
  with_pp
    (fun ppf (ij, os) ->
      Format.fprintf ppf "%s [%s]" (if ij then "ij" else "xy")
        (String.concat "; "
           (List.map (fun o -> Format.asprintf "%a %a" pp_layout o.l pp_ints o.s) os)))
    (let* ij = bool in
     let* n = int_range 0 3 in
     let one =
       let* d = frequency [ (1, constant 0); (1, constant 1); (3, int_range 2 3) ] in
       let+ l = of_list [ Plain; Reversed; Broadcast ] in
       operand_of [| d |] l
     in
     let rec all k = if k = 0 then constant [] else
         let* o = one in
         let+ os = all (k - 1) in
         o :: os
     in
     let+ os = all n in
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

let grids =
  group "grids"
    [
      prop "meshgrid holds each operand along its axis" grid_case law_meshgrid;
      test "meshgrid's two indexings at a plane" (fun () ->
          let x = (operand_of [| 3 |] Plain).x and y = (operand_of [| 2 |] Plain).x in
          (match Nx.meshgrid ~indexing:`Xy [ x; y ] with
          | [ gx; gy ] ->
              equal (array int) [| 2; 3 |] (Nx.shape gx);
              equal (array int32) [| 1l; 2l; 3l; 1l; 2l; 3l |] (elements gx);
              equal (array int32) [| 1l; 1l; 1l; 2l; 2l; 2l |] (elements gy)
          | _ -> fail "two grids");
          match Nx.meshgrid ~indexing:`Ij [ x; y ] with
          | [ gx; gy ] ->
              equal (array int) [| 3; 2 |] (Nx.shape gx);
              equal (array int32) [| 1l; 1l; 2l; 2l; 3l; 3l |] (elements gx);
              equal (array int32) [| 1l; 2l; 1l; 2l; 1l; 2l |] (elements gy)
          | _ -> fail "two grids");
      test "meshgrid's grids are views" (fun () ->
          let x = (operand_of [| 3 |] Plain).x and y = (operand_of [| 2 |] Plain).x in
          match Nx.meshgrid ~indexing:`Xy [ x; y ] with
          | [ gx; gy ] ->
              equal ~msg:"x" bool true (shares x gx);
              equal ~msg:"y" bool true (shares y gy)
          | _ -> fail "two grids");
      test "meshgrid refuses an operand that is not 1-d" (fun () ->
          invalid ~by:"Nx.meshgrid" (fun () ->
              Nx.meshgrid ~indexing:`Ij [ (operand_of [| 2; 2 |] Plain).x ]);
          invalid ~by:"Nx.meshgrid" (fun () ->
              Nx.meshgrid ~indexing:`Xy [ (operand_of [||] Plain).x ]));
    ]

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
          each "split_at" (List.nth (Nx.split_at ~axis:2 [ 1; 3 ] x) 1);
          each "tile of a unit axis"
            (Nx.tile [| 3; 1 |] (Nx.reshape [| 1; 24 |] x));
          each "repeat of a unit axis"
            (Nx.repeat ~axis:0 3 (Nx.reshape [| 1; 24 |] x)));
      cases
        "a permutation, flip, broadcast or window of a strided operand is a \
         view"
        ~id:(fun l -> Format.asprintf "%a" pp_layout l)
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
          each "split" (List.nth (Nx.split ~axis:(-1) 3 x) 1);
          each "split_at" (List.nth (Nx.split_at ~axis:(-1) [ -2 ] x) 1));
      test "a reshape no stride expresses copies" (fun () ->
          let x = Nx.transpose (operand_of [| 2; 3 |] Plain).x in
          let y = Nx.reshape [| 6 |] x in
          equal bool false (shares x y);
          equal (array int32) [| 1l; 4l; 2l; 5l; 3l; 6l |] (elements y));
    ]

(* A movement of every dtype, against the reference over the dtype's values. *)
let dtypes =
  cases "every dtype"
    ~id:(fun (D.Any dt) -> D.name dt)
    D.all
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

let contains s w =
  let n = String.length w in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = w || at (i + 1))
  in
  at 0

(* A refusal in nx's words: its message starts with the function the user
   called and names no function of the layers below. *)
let named ~by f =
  match f () with
  | () -> failf "%s raised nothing" by
  | exception Invalid_argument m ->
      let n = Int.min (String.length by + 2) (String.length m) in
      let start = String.sub m 0 n in
      let below = [ "Nx_array"; "Nx_kernel"; "Layout."; "Move."; "Prog." ] in
      equal ~msg:m (pair string (list string)) (by ^ ": ", [])
        (start, List.filter (contains m) below)

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
          raises (Invalid_argument "Nx.reshape: extent -2 in [-2; 3]")
            (fun () -> Nx.reshape [| -2; 3 |] x);
          raises
            (Invalid_argument "Nx.reshape: [-1; -1] has two unknown extents")
            (fun () -> Nx.reshape [| -1; -1 |] x);
          raises
            (Invalid_argument
               "Nx.reshape: int32 [2; 3] has 6 elements, which [4; -1] cannot \
                hold") (fun () -> Nx.reshape [| 4; -1 |] x);
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
            (fun () -> Nx.tile [| 2 |] x);
          raises
            (Invalid_argument
               (Printf.sprintf
                  "Nx.pad: widths (%d, 0) of axis 1 of int32 [2; 3] count past \
                   an int"
                  max_int))
            (fun () -> Nx.pad [| (0, 0); (max_int, 0) |] 0l x);
          raises
            (Invalid_argument
               (Printf.sprintf
                  "Nx.pad: widths (%d, %d) of axis 0 of int32 [2; 3] count past \
                   an int"
                  max_int max_int))
            (fun () -> Nx.pad [| (max_int, max_int); (0, 0) |] 0l x));
      cases "each refusal raises" ~id:fst
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
          r "sliding_window, step 0" (fun () ->
              Nx.sliding_window ~window:1 ~step:0 x);
          r "tile, too few reps" (fun () -> Nx.tile [| 2 |] x);
          r "tile, a negative rep" (fun () -> Nx.tile [| 1; -1 |] x);
          r "repeat, a negative count" (fun () -> Nx.repeat (-1) x);
          r "repeat, not an axis" (fun () -> Nx.repeat ~axis:2 2 x);
          r "reshape, 33 axes" (fun () ->
              Nx.reshape (Array.append [| 2; 3 |] (Array.make 31 1)) x);
          r "reshape, extents past 2^53" (fun () ->
              Nx.reshape [| 0; 1 lsl 27; 1 lsl 27 |] (Nx.zeros Nx.int32 [| 0 |]));
          r "broadcast_to, 33 axes" (fun () ->
              Nx.broadcast_to (Array.append (Array.make 31 1) [| 2; 3 |]) x);
          r "broadcast_to, extents past 2^53" (fun () ->
              Nx.broadcast_to [| 1 lsl 27; 1 lsl 27; 2; 3 |] x);
          r "unsqueeze, 33 axes" (fun () ->
              Nx.unsqueeze ~axes:[ 0 ] (Nx.zeros Nx.int32 (Array.make 32 1)));
          r "sliding_window, 33 axes" (fun () ->
              Nx.sliding_window ~window:1 (Nx.zeros Nx.int32 (Array.make 32 1)));
          r "tile, 33 reps" (fun () -> Nx.tile (Array.make 33 1) x);
          r "tile, extents past 2^53" (fun () -> Nx.tile [| 1 lsl 27; 1 lsl 26 |] x);
          r "pad, widths past 2^53" (fun () ->
              Nx.pad [| (1 lsl 27, 0); (1 lsl 27, 0) |] 0l x);
          r "add, a broadcast past 2^53" (fun () ->
              Nx.add (Nx.zeros Nx.int32 [| 1 lsl 27; 1 |])
                (Nx.zeros Nx.int32 [| 1; 1 lsl 27 |]));
          r "logical_and, a broadcast past 2^53" (fun () ->
              Nx.logical_and (Nx.zeros Nx.int32 [| 1 lsl 27; 1 |])
                (Nx.zeros Nx.int32 [| 1; 1 lsl 27 |]));
          r "pad, an int outside the dtype" (fun () ->
              Nx.pad [| (1, 0) |] 300 (Nx.zeros Nx.uint8 [| 1 |]));
        ]
        (fun (name, f) ->
          let by = "Nx." ^ String.sub name 0 (String.index name ',') in
          named ~by f);
      test "split refuses no runs" (fun () ->
          invalid ~by:"Nx.split" (fun () -> Nx.split ~axis:0 0 x));
      test "split_at refuses decreasing offsets" (fun () ->
          raises
            (Invalid_argument
               "Nx.split_at: offset 1 cuts at 1, before offset 2 at 2, along \
                axis 1 of int32 [2; 3]") (fun () ->
              Nx.split_at ~axis:1 [ 2; 1 ] x);
          raises
            (Invalid_argument
               "Nx.split_at: offset 1 cuts at 1, before offset -1 at 2, along \
                axis 1 of int32 [2; 3]") (fun () ->
              Nx.split_at ~axis:1 [ -1; 1 ] x));
    ]

(* Extensions: each rule against its equations, numpy's pad and scipy's
   odd_ext (golden/extend.golden, which lists each case's source). *)

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
   rule's equations read again outside the signal; [opp a b] is [2 a - b]
   in the signal's arithmetic. *)
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

(* The golden *)

let golden = "golden/extend.golden"

type golden_case = {
  header : string;
  dtype : D.any;
  rule : Nx.extension;
  shape : int array;
  along : int;
  lo : int;
  hi : int;
  gx : float array;
  gy : float array;
}

let read_golden ic header =
  let values () =
    match String.split_on_char ' ' (input_line ic) with
    | _ :: vs ->
        Array.of_list
          (List.filter_map
             (fun v -> if v = "" then None else Some (float_of_string v))
             vs)
    | [] -> [||]
  in
  match String.split_on_char ' ' header with
  | [ "case"; dt; rule; shape; along; lo; hi; _ ] ->
      let rule = List.find (fun e -> rule_name e = rule) rules in
      let shape =
        Array.of_list (List.map int_of_string (String.split_on_char ',' shape))
      in
      let gx = values () in
      let gy = values () in
      {
        header;
        dtype = Option.get (D.of_name dt);
        rule;
        shape;
        along = int_of_string along;
        lo = int_of_string lo;
        hi = int_of_string hi;
        gx;
        gy;
      }
  | _ -> failf "%s: a case header %S" golden header

let goldens =
  let path = Filename.concat (Filename.dirname Sys.executable_name) golden in
  In_channel.with_open_text path (fun ic ->
      let rec go acc =
        match In_channel.input_line ic with
        | None -> List.rev acc
        | Some h -> go (read_golden ic h :: acc)
      in
      go [])

(* A golden case on its operand laid out plain, reversed and transposed, the
   axis spelled from the end for the transposed one, and by default for a
   1-d operand. *)
let law_golden g =
  let (D.Any dt) = g.dtype in
  let r = Array.length g.shape in
  let w = Testable.make ~pp:(D.pp_value dt) ~equal:( = ) in
  let s' = Array.copy g.shape in
  s'.(g.along) <- g.lo + g.shape.(g.along) + g.hi;
  List.iter
    (fun l ->
      let data = Array.map (D.of_float dt) g.gx in
      let x = Nx.Repr.of_array Nx.Host.v (lay l dt g.shape data) in
      let y =
        match l with
        | Transposed -> Nx.extend ~axis:(g.along - r) g.rule (g.lo, g.hi) x
        | _ when r = 1 -> Nx.extend g.rule (g.lo, g.hi) x
        | _ -> Nx.extend ~axis:g.along g.rule (g.lo, g.hi) x
      in
      let msg = Format.asprintf "%a" pp_layout l in
      equal ~msg (array int) s' (Nx.shape y);
      equal ~msg (array w) (Array.map (D.of_float dt) g.gy) (elements y))
    [ Plain; Reversed; Transposed ]

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

let extensions =
  group "extensions"
    [
      cases "extend gives numpy's pad and scipy's odd_ext"
        ~id:(fun g -> g.header)
        goldens law_golden;
      prop "extend reads each position by its rule's equations"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let* neg = bool in
          let* e = of_list rules in
          let* lo = int_range 0 7 in
          let+ hi = int_range 0 7 in
          (o, a, neg, e, lo, hi))
        (fun (o, a, neg, e, lo, hi) ->
          covers o;
          let n = o.s.(a) in
          cover "a broadcast operand" (o.l = Broadcast && numel o.s > 1);
          cover "one sample, widened" (n = 1 && e <> `Zero && lo + hi > 0);
          cover "a fold past the far end" (n >= 2 && (lo > n || hi > n));
          cover "folds twice past an end" (n = 2 && (lo > 2 * n || hi > 2 * n));
          let y () = Nx.extend ~axis:(spell o neg a) e (lo, hi) o.x in
          if n = 0 && e <> `Zero && lo + hi > 0 then invalid ~by:"Nx.extend" y
          else
            let s', f = extended o a e lo hi in
            is s' f (y ()));
      prop "extend's laws"
        Gen.(
          let* o = operand ~min_rank:1 () in
          let* a = axis_of o in
          let* e = of_list rules in
          let* lo = int_range 0 7 in
          let+ hi = int_range 0 7 in
          (o, a, e, lo, hi))
        (fun (o, a, e, lo, hi) ->
          let n = o.s.(a) in
          assume (n > 0);
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
          (* A symmetric extension is the periodic one of the signal and its
             mirror, the whole-sample mirror without its end samples. *)
          let mirror start count =
            Nx.flip ~axes:[ a ]
              (Nx.slice (along (R (start, start + count))) o.x)
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
              (elements (ext `Symmetric_whole (lo, hi) o.x)));
      prop ~count:300 "a sample's bits depend on its position alone"
        (Gen.with_pp
           (fun ppf (v, e, lo, hi) ->
             Format.fprintf ppf "%s (%d, %d) [%s]" (rule_name e) lo hi
               (String.concat "; "
                  (Array.to_list (Array.map string_of_float v))))
           Gen.(
             let* n = of_list [ 2; 3; 5 ] in
             let* v =
               array ~size:(constant n)
                 (of_list [ 0.1; 0.7; -0.3; 1.9; 3.3; -2.6; 0.45 ])
             in
             let* e = of_list rules in
             let* lo = int_range 0 (4 * n) in
             let+ hi = int_range 0 (4 * n) in
             (v, e, lo, hi)))
        (fun (v, e, lo, hi) ->
          let n = Array.length v in
          cover "antisymmetric, both sides past two folds"
            (e = `Antisymmetric_whole && lo > 2 * (n - 1) && hi > 2 * (n - 1));
          let x = host D.Float64 [| n |] v in
          let bits y = elements (Nx.bitcast D.Int64 y) in
          let y = bits (Nx.extend e (lo, hi) x) in
          let l = bits (Nx.extend e (lo, 0) x) in
          let r = bits (Nx.extend e (0, hi) x) in
          equal ~msg:"before" (array int64) (Array.sub l 0 lo)
            (Array.sub y 0 lo);
          equal ~msg:"after" (array int64) (Array.sub r n hi)
            (Array.sub y (lo + n) hi));
      cases "every dtype, and the antisymmetric rule on every one but booleans"
        ~id:(fun (D.Any dt) -> D.name dt)
        D.all
        (fun (D.Any dt) ->
          (* Rows whose antisymmetric extensions by two stay within [0, 4],
             which every integer and float dtype holds. *)
          let data = [| 2.; 3.; 2.; 1.; 2.; 2. |] in
          let x =
            Nx.Repr.of_array Nx.Host.v
              (lay Transposed dt [| 2; 3 |] (Array.map (D.of_float dt) data))
          in
          let w = Testable.make ~pp:(D.pp_value dt) ~equal:( = ) in
          List.iter
            (fun e ->
              let y () = Nx.extend e (2, 2) x in
              if e = `Antisymmetric_whole && D.is D.Boolean dt then
                invalid ~by:"Nx.extend" y
              else
                let expected =
                  init [| 2; 7 |] (fun i ->
                      D.of_float dt
                        (read ~zero:0. ~opp:opp_f e 3
                           (fun k -> data.((3 * i.(0)) + k))
                           (i.(1) - 2)))
                in
                equal ~msg:(rule_name e) (array w) expected (elements (y ())))
            rules);
      cases "the antisymmetric rule rounds once and wraps"
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
        (fun (_, v, lo, hi) ->
          let n = Array.length v in
          let x = host D.Float64 [| n |] v in
          equal (array float_bits)
            (Array.init
               (lo + n + hi)
               (fun k ->
                 read ~zero:0. ~opp:opp_f `Antisymmetric_whole n (Array.get v)
                   (k - lo)))
            (elements (Nx.extend `Antisymmetric_whole (lo, hi) x)));
      test "the antisymmetric rule on float16, int32 extremes and complex parts"
        (fun () ->
          (* [2 x(0)] overflows float16, [2 x(0) - x(1)] does not. *)
          let h = host D.Float16 [| 2 |] [| 60000.; 64992. |] in
          equal (array float_bits)
            [| 55008.; 60000.; 64992. |]
            (elements (Nx.extend `Antisymmetric_whole (1, 0) h));
          let v = [| Int32.max_int; Int32.min_int; 5l |] in
          let x = host D.Int32 [| 3 |] v in
          equal (array int32)
            (Array.init 7 (fun k ->
                 read32 `Antisymmetric_whole 3 (Array.get v) (k - 2)))
            (elements (Nx.extend `Antisymmetric_whole (2, 2) x));
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
              (Nx.extend `Antisymmetric_whole (2, 2)
                 (host D.Complex128 [| 3 |] z))
          in
          equal ~msg:"re" (array float_bits)
            (parts (fun c -> c.Complex.re))
            (Array.map (fun c -> c.Complex.re) y);
          equal ~msg:"im" (array float_bits)
            (parts (fun c -> c.Complex.im))
            (Array.map (fun c -> c.Complex.im) y));
      test "the antisymmetric rule on complex values of the most axes"
        (fun () ->
          (* A complex value's parts take an axis more: at the most axes the
             rule raises, one fewer computes. *)
          let r = A.Layout.max_rank in
          let z =
            [|
              { Complex.re = 0.1; im = -0.7 };
              { re = 0.7; im = 2.5 };
              { re = -0.3; im = 0.45 };
            |]
          in
          let along r a = Array.init r (fun i -> if i = a then 3 else 1) in
          List.iter
            (fun a ->
              named ~by:"Nx.extend" (fun () ->
                  ignore
                    (Nx.extend ~axis:a `Antisymmetric_whole (1, 1)
                       (host D.Complex128 (along r a) z))))
            [ 0; 1; r - 1 ];
          let part f k =
            read ~zero:0. ~opp:opp_f `Antisymmetric_whole 3
              (fun i -> f z.(i))
              (k - 1)
          in
          let x = host D.Complex128 (along (r - 1) (r - 2)) z in
          let y = elements (Nx.extend `Antisymmetric_whole (1, 1) x) in
          equal ~msg:"re" (array float_bits)
            (Array.init 5 (part (fun c -> c.Complex.re)))
            (Array.map (fun c -> c.Complex.re) y);
          equal ~msg:"im" (array float_bits)
            (Array.init 5 (part (fun c -> c.Complex.im)))
            (Array.map (fun c -> c.Complex.im) y));
      test "copying rules keep a NaN's bits and signed zeros" (fun () ->
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
            [ `Replicate; `Symmetric_whole; `Symmetric_half; `Periodic ]);
      test "an empty signal extends only under `Zero, and is copied" (fun () ->
          let x = Nx.zeros Nx.int32 [| 2; 0 |] in
          equal (array int32)
            [| 0l; 0l; 0l; 0l; 0l; 0l |]
            (elements (Nx.extend `Zero (1, 2) x));
          List.iter
            (fun e ->
              equal (array int) [| 2; 0 |] (Nx.shape (Nx.extend e (0, 0) x)))
            rules);
      test "extend copies" (fun () ->
          let x = (operand_of [| 2; 3 |] Plain).x in
          List.iter
            (fun e ->
              equal ~msg:(rule_name e) bool false
                (shares x (Nx.extend e (0, 0) x));
              equal ~msg:(rule_name e) bool false
                (shares x (Nx.extend e (1, 2) x)))
            rules);
      test "extend's refusals name it" (fun () ->
          let x = (operand_of [| 2; 3 |] Plain).x in
          let each f = named ~by:"Nx.extend" (fun () -> ignore (f ())) in
          each (fun () -> Nx.extend `Zero (-1, 0) x);
          each (fun () -> Nx.extend `Periodic (0, -1) x);
          each (fun () -> Nx.extend ~axis:2 `Zero (1, 1) x);
          each (fun () -> Nx.extend ~axis:(-3) `Replicate (1, 1) x);
          each (fun () -> Nx.extend `Replicate (1, 1) (Nx.scalar Nx.int32 1l));
          List.iter
            (fun e ->
              if e <> `Zero then
                each (fun () ->
                    Nx.extend e (0, 1) (Nx.zeros Nx.int32 [| 3; 0 |])))
            rules;
          each (fun () ->
              Nx.extend `Antisymmetric_whole (1, 1) (Nx.zeros Nx.bool [| 2 |])));
    ]

let () =
  exit
    (run "nx shapes"
       [ laws; joins; extensions; properties; grids; views; dtypes; refusals ])
