(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Double-word numbers. Each operation is held to its stated bound against
   mpmath's exact results (golden/, written by gen/generate.py), and the
   invariant, the constructor and the structure to their laws. *)

open Windtrap
open Nx_test

(* Dtypes *)

type dtype = D : string * (float, 'b) Nx.dtype * float -> dtype

let f32 = D ("f32", Nx.float32, Float.ldexp 1. (-24))
let f64 = D ("f64", Nx.float64, Float.ldexp 1. (-53))
let dtypes = [ f32; f64 ]
let name (D (n, _, _)) = n

(* Goldens: rows of a dtype's tag and hexadecimal floats. *)

let golden file =
  In_channel.with_open_text
    ("golden/" ^ file ^ ".golden")
    In_channel.input_lines
  |> List.filter (fun l -> l <> "" && l.[0] <> '#')
  |> List.map (fun l ->
      match String.split_on_char ' ' l with
      | tag :: fields -> (tag, Array.of_list fields)
      | [] -> assert false)

let rows file (D (tag, _, _)) =
  List.filter_map
    (fun (t, fields) -> if t = tag then Some fields else None)
    (golden file)

let column rows i =
  Array.of_list (List.map (fun r -> float_of_string r.(i)) rows)

let tensor (type b) (dt : (float, b) Nx.dtype) xs =
  Nx.create dt [| Array.length xs |] xs

let operand dt rows i j =
  Nx_wide.v ~lo:(tensor dt (column rows j)) (tensor dt (column rows i))

let words w = (Nx.to_array (Nx_wide.hi w), Nx.to_array (Nx_wide.lo w))

(* Errors *)

let two_sum a b =
  let s = a +. b in
  let a' = s -. b in
  let b' = s -. a' in
  (s, a -. a' +. (b -. b'))

(* [error (zh, zl) (r1, r2, r3)] is [zh + zl - (r1 + r2 + r3)], to a relative
   [2^-50] of itself: [zh - r1] is exact, as [zh] and [r1] round one number, and
   the rest is summed with each rounding error kept. *)
let error (zh, zl) (r1, r2, r3) =
  let s, c =
    List.fold_left
      (fun (s, c) t ->
        let s, e = two_sum s t in
        (s, c +. e))
      (0., 0.)
      [ zh -. r1; zl; -.r2; -.r3 ]
  in
  s +. c

(* The slack of measuring an error in float64: far below any bound's margin. *)
let slack = 1. +. Float.ldexp 1. (-40)

(* A number's words: the high word to its bits, the low word to its value, whose
   zero's sign a number does not carry. *)
let unsigned =
  Testable.contramap (fun x -> if x = 0. then 0. else x) float_exact

let number = pair float_exact unsigned

(* Bounds against exact results *)

type binary = {
  op : string;
  wide : 'b. 'b Nx_wide.t -> 'b Nx_wide.t -> 'b Nx_wide.t;
  bound : float -> float;
}

let binaries =
  let sum_bound u = 3. *. u *. u /. (1. -. (4. *. u)) in
  [
    { op = "add"; wide = Nx_wide.add; bound = sum_bound };
    { op = "sub"; wide = Nx_wide.sub; bound = sum_bound };
    { op = "mul"; wide = Nx_wide.mul; bound = (fun u -> 4. *. u *. u) };
    { op = "div"; wide = Nx_wide.div; bound = (fun u -> 10. *. u *. u) };
  ]

(* The worst row's relative error, as a multiple of [u²], and the row. *)
let worst b (D (_, dt, u) as d) =
  let rows = rows b.op d in
  let x = operand dt rows 0 1 and y = operand dt rows 2 3 in
  let zh, zl = words (b.wide x y) in
  let r1 = column rows 4 and r2 = column rows 5 and r3 = column rows 6 in
  let worst = ref (0., -1) in
  Array.iteri
    (fun k r1 ->
      let e = Float.abs (error (zh.(k), zl.(k)) (r1, r2.(k), r3.(k))) in
      let rel =
        if r1 = 0. then if e = 0. then 0. else Float.infinity
        else e /. Float.abs r1
      in
      if rel > fst !worst then worst := (rel, k))
    r1;
  (fst !worst /. (u *. u), snd !worst)

let bounds =
  cases
    ~name:(fun (b, d) -> Printf.sprintf "%s at %s" b.op (name d))
    "each operation is within its bound of the exact result"
    (List.concat_map (fun b -> List.map (fun d -> (b, d)) dtypes) binaries)
    (fun (b, (D (_, _, u) as d)) ->
      let rel, row = worst b d in
      at_most
        ~msg:(Printf.sprintf "row %d, in units of u²" row)
        float_exact
        ~than:(b.bound u *. slack /. (u *. u))
        rel)

let floors =
  cases ~name "floor is the exact floor" dtypes (fun (D (_, dt, _) as d) ->
      let rows = rows "floor" d in
      let zh, zl = words (Nx_wide.floor (operand dt rows 0 1)) in
      (* mpmath's zero has no sign; the sign of a zero floor is an edge. *)
      equal (array number)
        (Array.map2 (fun h l -> (h, l)) (column rows 2) (column rows 3))
        (Array.map2 (fun h l -> ((if h = 0. then 0. else h), l)) zh zl))

let comparisons =
  cases ~name "comparisons are exact" dtypes (fun (D (_, dt, _) as d) ->
      let rows = rows "compare" d in
      let x = operand dt rows 0 1 and y = operand dt rows 2 3 in
      let flags i = Array.map (fun v -> v = 1.) (column rows i) in
      equal ~msg:"less" (array bool) (flags 4) (Nx.to_array (Nx_wide.less x y));
      equal ~msg:"equal" (array bool) (flags 5)
        (Nx.to_array (Nx_wide.equal x y)))

(* Rows of one summand count, summed along their last axis as a batch. *)
let sum_rows (D (_, dt, u) as d) =
  let rows = rows "sum" d in
  let counts =
    List.sort_uniq Int.compare (List.map (fun r -> int_of_string r.(0)) rows)
  in
  List.iter
    (fun n ->
      let rows = List.filter (fun r -> int_of_string r.(0) = n) rows in
      let m = List.length rows in
      let word j =
        Nx.create dt [| m; n |]
          (Array.concat
             (List.map
                (fun r ->
                  Array.init n (fun i -> float_of_string r.(1 + (2 * i) + j)))
                rows))
      in
      let zh, zl =
        words (Nx_wide.sum ~axes:[ 1 ] (Nx_wide.v ~lo:(word 1) (word 0)))
      in
      let at i =
        column (List.map (fun r -> Array.sub r (1 + (2 * n)) 4) rows) i
      in
      let r1 = at 0 and r2 = at 1 and r3 = at 2 and s = at 3 in
      let levels = Float.ceil (Float.log2 (float_of_int n)) in
      let delta = 3. *. u *. u /. (1. -. (4. *. u)) in
      let gamma = Float.expm1 (levels *. Float.log1p delta) *. slack in
      Array.iteri
        (fun k r1 ->
          at_most
            ~msg:(Printf.sprintf "%d summands, row %d" n k)
            float_exact
            ~than:(gamma *. s.(k))
            (Float.abs (error (zh.(k), zl.(k)) (r1, r2.(k), r3.(k)))))
        r1)
    counts

let sums =
  cases ~name "a sum is within its bound of the exact sum" dtypes sum_rows

(* Laws *)

(* A normalised pair from two drawn floats: [hi], and [lo] a fraction of it
   [2^-k] down, which [v] normalises. *)
let pairs =
  let open Gen in
  map
    (fun (hs, (ls, ks)) ->
      let n = min (Array.length hs) (min (Array.length ls) (Array.length ks)) in
      ( Array.sub hs 0 n,
        Array.init n (fun i -> Float.ldexp (ls.(i) *. hs.(i)) (-ks.(i))) ))
    (pair
       (array ~size:(int_range 1 16) (float_range (-1e6) 1e6))
       (pair
          (array ~size:(int_range 16 16) (float_range (-1.) 1.))
          (array ~size:(int_range 16 16) (int_range 0 80))))

let drawn (type b) (dt : (float, b) Nx.dtype) (h, l) =
  Nx_wide.v ~lo:(tensor dt l) (tensor dt h)

(* [normal w] is whether each [hi] is [hi + lo] rounded, or NaN with a zero
   [lo]. *)
let normal w =
  let h = Nx_wide.hi w and l = Nx_wide.lo w in
  let nan = Nx.logical_and (Nx.isnan h) (Nx.equal l (Nx.zeros_like l)) in
  Array.for_all Fun.id
    (Nx.to_array (Nx.logical_or nan (Nx.equal h (Nx.add h l))))

let laws =
  List.concat_map
    (fun (D (tag, dt, _)) ->
      let named s = Printf.sprintf "%s at %s" s tag in
      [
        prop (named "every operation's result is normalised")
          (Gen.pair pairs pairs) (fun ((h, l), (h', l')) ->
            let n = min (Array.length h) (Array.length h') in
            let cut a = Array.sub a 0 n in
            let x = drawn dt (cut h, cut l) and y = drawn dt (cut h', cut l') in
            List.iter
              (fun (op, z) -> equal ~msg:op bool true (normal z))
              [
                ("v", x);
                ("add", Nx_wide.add x y);
                ("sub", Nx_wide.sub x y);
                ("mul", Nx_wide.mul x y);
                ("div", Nx_wide.div x y);
                ("floor", Nx_wide.floor x);
                ("sum", Nx_wide.sum x);
              ]);
        prop (named "v returns a normalised pair's words unchanged") pairs
          (fun p ->
            let w = drawn dt p in
            let w' = Nx_wide.v ~lo:(Nx_wide.lo w) (Nx_wide.hi w) in
            equal
              (pair (array float_exact) (array float_exact))
              (words w) (words w'));
        prop (named "a number less itself is zero") pairs (fun p ->
            let w = drawn dt p in
            let zh, zl = words (Nx_wide.sub w w) in
            equal (array float_exact) (Array.make (Array.length zh) 0.) zh;
            equal (array float_exact) (Array.make (Array.length zl) 0.) zl);
        prop (named "a product of floats is exact") pairs (fun (h, l) ->
            let round x = Nx.item [] (Nx.cast dt (Nx.scalar Nx.float64 x)) in
            let a = Array.map round h and b = Array.map round l in
            let zh, zl =
              words
                (Nx_wide.mul
                   (Nx_wide.v (tensor dt a))
                   (Nx_wide.v (tensor dt b)))
            in
            Array.iteri
              (fun i zh ->
                let ph = round (a.(i) *. b.(i)) in
                (* The product's rounding error: exact at float64 by [fma], and
                   by the float64 product of float32 words. *)
                let pl =
                  if tag = "f64" then Float.fma a.(i) b.(i) (-.ph)
                  else (a.(i) *. b.(i)) -. ph
                in
                equal number (ph, pl) (zh, zl.(i)))
              zh);
        prop (named "a sum's association depends only on the shape") pairs
          (fun p ->
            let w = drawn dt p in
            (* Two columns of one sum: each sums alone as the vector does. *)
            let twice =
              Nx_wide.v
                ~lo:(Nx.stack ~axis:1 [ Nx_wide.lo w; Nx.neg (Nx_wide.lo w) ])
                (Nx.stack ~axis:1 [ Nx_wide.hi w; Nx.neg (Nx_wide.hi w) ])
            in
            let zh, zl = words (Nx_wide.sum ~axes:[ 0 ] twice) in
            let sh, sl = words (Nx_wide.sum w) in
            (* A zero's sign is the plain sum's, which is [+0]. *)
            let value = pair unsigned unsigned in
            equal value (sh.(0), sl.(0)) (zh.(0), zl.(0));
            equal value (-.sh.(0), -.sl.(0)) (zh.(1), zl.(1)));
        prop (named "the structure rebuilds a value it walks unchanged") pairs
          (fun p ->
            let w = drawn dt p in
            let w' = Nx.Ptree.map (Nx_wide.ptree dt) (fun _ t -> t) w in
            equal
              (pair (array float_exact) (array float_exact))
              (words w) (words w'));
      ])
    dtypes

let structure =
  group "structure"
    [
      test "the words are at hi and lo" (fun () ->
          let w = Nx_wide.v (Nx.ones Nx.float64 [| 2 |]) in
          equal (list string) [ "hi"; "lo" ]
            (List.rev
               (Nx.Ptree.fold (Nx_wide.ptree Nx.float64)
                  (fun p _ acc -> Nx.Ptree.Path.to_string p :: acc)
                  w [])));
      test "a value rebuilt from leafwise sums is the number they make"
        (fun () ->
          (* hi words 1 + 1 and low words 2^-60 + 2^-60 make 2 + 2^-59. *)
          let w =
            Nx_wide.v
              ~lo:(Nx.scalar Nx.float64 (Float.ldexp 1. (-60)))
              (Nx.scalar Nx.float64 1.)
          in
          let s =
            Nx.Ptree.map2 (Nx_wide.ptree Nx.float64)
              (fun _ a b -> Nx.add a b)
              w w
          in
          equal
            (pair float_exact float_exact)
            (2., Float.ldexp 1. (-59))
            (Nx.item [] (Nx_wide.hi s), Nx.item [] (Nx_wide.lo s)));
      test
        "a rebuilt value walks as the words it was rebuilt from, and reads \
         normalised" (fun () ->
          let p = Nx_wide.ptree Nx.float64 in
          let w = Nx_wide.v (Nx.scalar Nx.float64 1.) in
          let s = Nx.Ptree.map p (fun _ t -> Nx.add t (Nx.ones_like t)) w in
          let walked =
            List.rev
              (Nx.Ptree.fold p
                 (fun _ t acc -> Nx.item [] (Nx.cast Nx.float64 t) :: acc)
                 s [])
          in
          equal ~msg:"walked" (list float_exact) [ 2.; 1. ] walked;
          equal ~msg:"read"
            (pair float_exact float_exact)
            (3., 0.)
            (Nx.item [] (Nx_wide.hi s), Nx.item [] (Nx_wide.lo s)));
      test "a value rebuilt from unnormalised words is normalised" (fun () ->
          let w = Nx_wide.v (Nx.scalar Nx.float64 1.) in
          let s =
            Nx.Ptree.map (Nx_wide.ptree Nx.float64)
              (fun _ t -> Nx.add t (Nx.ones_like t))
              w
          in
          (* Words 1 and 0 become 2 and 1, held as 3 and 0. *)
          equal
            (pair float_exact float_exact)
            (3., 0.)
            (Nx.item [] (Nx_wide.hi s), Nx.item [] (Nx_wide.lo s)));
    ]

(* Edges *)

let scalar w = (Nx.item [] (Nx_wide.hi w), Nx.item [] (Nx_wide.lo w))
let f x = Nx.scalar Nx.float64 x
let same = pair float_exact float_exact

let nan_pair =
  Testable.contramap (fun (h, l) -> (Float.is_nan h, l)) (pair bool float_exact)

let edges =
  group "edges"
    [
      test "an infinite number is held with a zero low word" (fun () ->
          equal same (Float.infinity, 0.)
            (scalar (Nx_wide.v ~lo:(f 1.) (f Float.infinity))));
      test "NaN is held with a zero low word" (fun () ->
          equal nan_pair (Float.nan, 0.)
            (scalar (Nx_wide.v ~lo:(f 1.) (f Float.nan))));
      test "an addition that overflows is the infinity" (fun () ->
          let m = Nx_wide.v (f Float.max_float) in
          equal same (Float.infinity, 0.) (scalar (Nx_wide.add m m)));
      test "an operation on an infinity is the float result of the high words"
        (fun () ->
          let inf = Nx_wide.v (f Float.infinity) and one = Nx_wide.v (f 1.) in
          equal same (Float.infinity, 0.) (scalar (Nx_wide.mul inf one));
          equal same (0., 0.) (scalar (Nx_wide.div one inf));
          equal nan_pair (Float.nan, 0.) (scalar (Nx_wide.sub inf inf)));
      test "NaN compares false" (fun () ->
          let n = Nx_wide.v (f Float.nan) in
          equal bool false (Nx.item [] (Nx_wide.equal n n));
          equal bool false (Nx.item [] (Nx_wide.less n n)));
      test "the floor of -0 is -0, and of a positive fraction +0" (fun () ->
          equal same (-0., 0.) (scalar (Nx_wide.floor (Nx_wide.v (f (-0.)))));
          equal same (0., 0.) (scalar (Nx_wide.floor (Nx_wide.v (f 0.5)))));
      test "x - x is +0, and -0 * x is -0" (fun () ->
          let x = Nx_wide.v ~lo:(f (Float.ldexp 1. (-60))) (f 1.) in
          equal same (0., 0.) (scalar (Nx_wide.sub x x));
          equal same (-0., 0.) (scalar (Nx_wide.mul (Nx_wide.v (f (-0.))) x)));
      test "the floor below an integer high word steps down" (fun () ->
          equal same (0., 0.)
            (scalar
               (Nx_wide.floor
                  (Nx_wide.v ~lo:(f (-.Float.ldexp 1. (-60))) (f 1.)))));
      test "a sum of no summand is zero" (fun () ->
          let w = Nx_wide.v (Nx.zeros Nx.float64 [| 2; 0 |]) in
          let zh, zl = words (Nx_wide.sum ~axes:[ 1 ] w) in
          equal (array float_exact) [| 0.; 0. |] zh;
          equal (array float_exact) [| 0.; 0. |] zl);
      test "operands broadcast" (fun () ->
          let a = Nx_wide.v (Nx.create Nx.float64 [| 2; 1 |] [| 1.; 2. |])
          and b = Nx_wide.v (Nx.create Nx.float64 [| 3 |] [| 1.; 2.; 3. |]) in
          equal (array int) [| 2; 3 |] (Nx.shape (Nx_wide.hi (Nx_wide.add a b))));
      test "v refuses a narrow float" (fun () ->
          raises_invalid_arg (fun () -> Nx_wide.v (Nx.zeros Nx.float16 [| 2 |])));
      test "the structure refuses a narrow float" (fun () ->
          raises_invalid_arg (fun () -> Nx_wide.ptree Nx.bfloat16));
      test "v refuses words that do not broadcast" (fun () ->
          raises_invalid_arg (fun () ->
              Nx_wide.v
                ~lo:(Nx.zeros Nx.float64 [| 3 |])
                (Nx.zeros Nx.float64 [| 2 |])));
      test "a sum refuses an axis out of bounds" (fun () ->
          raises_invalid_arg (fun () ->
              Nx_wide.sum ~axes:[ 2 ]
                (Nx_wide.v (Nx.zeros Nx.float64 [| 2; 2 |]))));
    ]

let () =
  exit
    (run "nx wide"
       [
         group "bounds" [ bounds; floors; comparisons; sums ];
         group "laws" laws;
         structure;
         edges;
       ])
