(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Double-word numbers. Each operation is held to its stated bound against
   mpmath's exact results (golden/, written by gen/generate.py), and the
   invariant, the constructor and the walk to their laws. *)

open Windtrap

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

(* Dtypes *)

type dtype = D : string * (float, 's) Nx.dtype * float -> dtype

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

let vector dt xs = Nx.create dt [| Array.length xs |] xs

let operand dt rows i j =
  Nx_wide.v ~lo:(vector dt (column rows j)) (vector dt (column rows i))

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

(* A float's value: zero unsigned, every NaN as one. A number's words are its
   high word to its bits and its low word to its value, whose zero's sign a
   number does not carry. *)
let unsigned =
  Testable.contramap
    (fun x -> if x = 0. then 0. else if Float.is_nan x then Float.nan else x)
    float_exact

let number = pair float_exact unsigned

(* Bounds against exact results *)

type binary = {
  op : string;
  wide :
    's.
    ('s, Nx.host) Nx_wide.t ->
    ('s, Nx.host) Nx_wide.t ->
    ('s, Nx.host) Nx_wide.t;
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
  cases "each operation is within its bound of the exact result"
    ~id:(fun (b, d) -> Printf.sprintf "%s at %s" b.op (name d))
    (List.concat_map (fun b -> List.map (fun d -> (b, d)) dtypes) binaries)
    (fun (b, (D (_, _, u) as d)) ->
      let rel, row = worst b d in
      at_most
        ~msg:(Printf.sprintf "row %d, in units of u²" row)
        float_exact
        ~than:(b.bound u *. slack /. (u *. u))
        rel)

let floors =
  cases "floor is the exact floor" ~id:name dtypes (fun (D (_, dt, _) as d) ->
      let rows = rows "floor" d in
      let zh, zl = words (Nx_wide.floor (operand dt rows 0 1)) in
      (* mpmath's zero has no sign; the sign of a zero floor is an edge. *)
      equal (array number)
        (Array.map2 (fun h l -> (h, l)) (column rows 2) (column rows 3))
        (Array.map2 (fun h l -> ((if h = 0. then 0. else h), l)) zh zl))

let comparisons =
  cases "comparisons are exact" ~id:name dtypes (fun (D (_, dt, _) as d) ->
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
  cases "a sum is within its bound of the exact sum" ~id:name dtypes sum_rows

(* Layouts: views and copies that read the same numbers from other strides. *)

type layout = {
  label : string;
  apply : 'v 's. ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t;
}

let layouts =
  let every_other axis t =
    if Nx.ndim t = 0 || Nx.dim axis t = 0 then t
    else Nx.squeeze ~axes:[ -1 ] (Nx.sliding_window ~axis ~window:1 ~step:2 t)
  in
  [
    { label = "transposed"; apply = (fun t -> Nx.transpose t) };
    { label = "flipped"; apply = (fun t -> Nx.flip t) };
    { label = "every other row"; apply = (fun t -> every_other 0 t) };
    { label = "every other column"; apply = (fun t -> every_other (-1) t) };
    {
      label = "broadcast over a new axis";
      apply = (fun t -> Nx.broadcast_to (Array.append [| 2 |] (Nx.shape t)) t);
    };
  ]

let layout =
  Gen.with_pp
    (fun ppf steps ->
      Format.pp_print_string ppf
        (String.concat ", " (List.map (fun l -> l.label) steps)))
    (Gen.list ~size:(Gen.int_range 0 2) (Gen.of_list layouts))

let lay_out steps t = List.fold_left (fun t l -> l.apply t) t steps

(* Laws *)

(* Numbers drawn for the laws: a shape of up to three axes, zero-size and
   one-element ones included, a layout, and per element a high word and a low
   word a fraction of it [2^-k] down, which [v] normalises. High words take
   signed zeros, infinities, NaN, subnormals and the format's extremes. *)
type drawn = {
  shape : int array;
  steps : layout list;
  his : float array;
  los : float array;
}

let special =
  [
    0.;
    -0.;
    Float.infinity;
    Float.neg_infinity;
    Float.nan;
    Float.min_float;
    Float.ldexp 1. (-1074);
    Float.max_float;
    -.Float.max_float;
    Float.ldexp 1. 1020;
    1.;
    -1.;
  ]

let high =
  Gen.frequency
    [
      (4, Gen.float_range (-1e6) 1e6);
      (1, Gen.of_list ~pp:Format.pp_print_float special);
    ]

let low hi (m, k) =
  let lo = Float.ldexp (m *. hi) (-k) in
  if Float.is_finite lo then lo else 0.

let numel s = Array.fold_left ( * ) 1 s

let numbers ?(count = 1) () =
  let open Gen in
  let* shape = array ~size:(int_range 0 3) (int_range 0 4) in
  let* steps = layout in
  let n = numel shape in
  let element = pair high (pair (float_range (-1.) 1.) (int_range 0 80)) in
  let+ draws = list ~size:(constant count) (array ~size:(constant n) element) in
  List.map
    (fun xs ->
      {
        shape;
        steps;
        his = Array.map fst xs;
        los = Array.map (fun (h, l) -> low h l) xs;
      })
    draws

let one = Gen.map List.hd (numbers ())

let two =
  Gen.map
    (function [ a; b ] -> (a, b) | _ -> assert false)
    (numbers ~count:2 ())

let drawn dt d =
  let t xs = lay_out d.steps (Nx.create dt d.shape xs) in
  Nx_wide.v ~lo:(t d.los) (t d.his)

(* What a law over drawn numbers needs to reach. *)
let cover_drawn d =
  cover "no element" (numel d.shape = 0);
  cover "one element" (numel d.shape = 1);
  cover "a strided or broadcast layout" (d.steps <> []);
  cover "a non-finite high word"
    (Array.exists (fun x -> not (Float.is_finite x)) d.his);
  cover "a zero high word" (Array.exists (fun x -> x = 0.) d.his)

(* [normal ~msg w] checks that each [hi] is [hi + lo] rounded, and that a NaN
   [hi] has a zero [lo]. *)
let normal ~msg w =
  let h = Nx_wide.hi w and l = Nx_wide.lo w in
  equal
    ~msg:(msg ^ ": hi is hi + lo rounded")
    (array unsigned)
    (Nx.to_array (Nx.add h l))
    (Nx.to_array h);
  let nan = Nx.isnan h in
  equal
    ~msg:(msg ^ ": a NaN's lo is zero")
    (array unsigned)
    (Nx.to_array (Nx.where nan (Nx.zeros_like l) l))
    (Nx.to_array l)

(* A number's words as values. *)
let number_words = pair (array unsigned) (array unsigned)

(* [associates_by_shape w] checks that [w] sums as each column of [w] and its
   negation stacked on a new last axis does: the same shape, laid out apart. *)
let associates_by_shape w =
  let r = Nx.ndim (Nx_wide.hi w) in
  let column f = Nx.stack ~axis:r [ f w; Nx.neg (f w) ] in
  let twice = Nx_wide.v ~lo:(column Nx_wide.lo) (column Nx_wide.hi) in
  let zh, zl = words (Nx_wide.sum ~axes:(List.init r Fun.id) twice) in
  let sh, sl = words (Nx_wide.sum w) in
  (* A zero's sign is the plain sum's, which is [+0]. *)
  let value = pair unsigned unsigned in
  equal value (sh.(0), sl.(0)) (zh.(0), zl.(0));
  equal value (-.sh.(0), -.sl.(0)) (zh.(1), zl.(1))

(* A number alone, described by its walk. *)
let alone () = Nx.Ptree.v Nx_wide.walk

let laws =
  List.concat_map
    (fun (D (tag, dt, _)) ->
      let named s = Printf.sprintf "%s at %s" s tag in
      [
        prop (named "every operation's result is normalised") two (fun (a, b) ->
            cover_drawn a;
            let x = drawn dt a and y = drawn dt b in
            List.iter
              (fun (op, z) -> normal ~msg:op z)
              [
                ("v", x);
                ("add", Nx_wide.add x y);
                ("sub", Nx_wide.sub x y);
                ("mul", Nx_wide.mul x y);
                ("div", Nx_wide.div x y);
                ("floor", Nx_wide.floor x);
                ("sum", Nx_wide.sum x);
              ]);
        prop (named "v returns a normalised pair's words unchanged") one
          (fun d ->
            cover_drawn d;
            let w = drawn dt d in
            let w' = Nx_wide.v ~lo:(Nx_wide.lo w) (Nx_wide.hi w) in
            equal
              (pair (array float_exact) (array float_exact))
              (words w) (words w'));
        prop (named "a number less itself is zero, a non-finite one NaN") one
          (fun d ->
            cover_drawn d;
            let w = drawn dt d in
            let h, _ = words w in
            let zh, zl = words (Nx_wide.sub w w) in
            equal number_words
              ( Array.map
                  (fun x -> if Float.is_finite x then 0. else Float.nan)
                  h,
                Array.make (Array.length zl) 0. )
              (zh, zl));
        prop (named "a product of floats in the domain is exact") two
          (fun (a, b) ->
            let x = Nx.to_array (Nx_wide.hi (drawn dt a))
            and y = Nx.to_array (Nx_wide.hi (drawn dt b)) in
            let x = Array.sub x 0 (min (Array.length x) (Array.length y)) in
            let y = Array.sub y 0 (Array.length x) in
            let round v = Nx.item [] (Nx.cast dt (Nx.scalar Nx.float64 v)) in
            let inside v =
              let edge = if tag = "f64" then 969 else 102 in
              v = 0.
              || Float.abs v >= Float.ldexp 1. (-edge)
                 && Float.abs v <= Float.ldexp 1. edge
            in
            let keep =
              List.filter
                (fun i ->
                  inside x.(i) && inside y.(i) && inside (x.(i) *. y.(i)))
                (List.init (Array.length x) Fun.id)
            in
            cover "a product in the domain" (keep <> []);
            let pick a = Array.of_list (List.map (fun i -> a.(i)) keep) in
            let x = pick x and y = pick y in
            let zh, zl =
              words
                (Nx_wide.mul
                   (Nx_wide.v (vector dt x))
                   (Nx_wide.v (vector dt y)))
            in
            Array.iteri
              (fun i zh ->
                let ph = round (x.(i) *. y.(i)) in
                (* The product's rounding error: exact at float64 by [fma], and
                   by the float64 product of float32 words. *)
                let pl =
                  if tag = "f64" then Float.fma x.(i) y.(i) (-.ph)
                  else (x.(i) *. y.(i)) -. ph
                in
                equal number (ph, pl) (zh, zl.(i)))
              zh);
        prop (named "a sum's association depends only on the shape") one
          (fun d ->
            cover_drawn d;
            associates_by_shape (drawn dt d));
        prop (named "the walk rebuilds a value it walks unchanged") one
          (fun d ->
            let w = drawn dt d in
            let w' = Nx.Ptree.map (alone ()) (fun _ t -> t) w in
            equal
              (pair (array float_exact) (array float_exact))
              (words w) (words w'));
      ])
    dtypes

(* Walks *)

(* A number of float64 as a structure's module, which moves between sets. *)
module Number = struct
  type ('a, 'd) t = (Nx.float64_elt, 'd) Nx_wide.t

  let walk = Nx_wide.walk
end

let scalar w = (Nx.item [] (Nx_wide.hi w), Nx.item [] (Nx_wide.lo w))
let f x = Nx.scalar Nx.float64 x
let same = pair float_exact float_exact

let walks =
  group "walk"
    [
      test "the words are at hi and lo" (fun () ->
          let w = Nx_wide.v (Nx.ones Nx.float64 [| 2 |]) in
          equal (list string) [ "hi"; "lo" ]
            (List.rev
               (Nx.Ptree.fold (alone ())
                  (fun p _ acc -> Nx.Ptree.Path.to_string p :: acc)
                  w [])));
      test "a structure walks a number at its field's path" (fun () ->
          let s =
            Nx.Ptree.v (fun c (a, b) ->
                let open Nx.Ptree.Walk in
                let a = field c "a" Nx_wide.walk a in
                let b = field c "b" tensor b in
                (a, b))
          in
          let x = (Nx_wide.v (f 1.), f 2.) in
          equal (list string) [ "a.hi"; "a.lo"; "b" ]
            (List.rev
               (Nx.Ptree.fold s
                  (fun p _ acc -> Nx.Ptree.Path.to_string p :: acc)
                  x [])));
      test "a value rebuilt from leafwise sums is the number they make"
        (fun () ->
          (* hi words 1 + 1 and low words 2^-60 + 2^-60 make 2 + 2^-59. *)
          let w = Nx_wide.v ~lo:(f (Float.ldexp 1. (-60))) (f 1.) in
          let s = Nx.Ptree.map2 (alone ()) (fun _ a b -> Nx.add a b) w w in
          equal same (2., Float.ldexp 1. (-59)) (scalar s));
      test
        "a rebuilt value walks as the words it was rebuilt from, and reads \
         normalised" (fun () ->
          let p = alone () in
          let w = Nx_wide.v (f 1.) in
          let s = Nx.Ptree.map p (fun _ t -> Nx.add t (Nx.ones_like t)) w in
          let walked =
            List.rev
              (Nx.Ptree.fold p
                 (fun _ t acc -> Nx.item [] (Nx.cast Nx.float64 t) :: acc)
                 s [])
          in
          (* Words 1 and 0 become 2 and 1, held as 3 and 0. *)
          equal ~msg:"walked" (list float_exact) [ 2.; 1. ] walked;
          equal ~msg:"read" same (3., 0.) (scalar s));
      test "axpy over a number scales and adds the number" (fun () ->
          (* 3 (1 + 2^-60) + (1 + 2^-60) is 4 + 2^-58. *)
          let w = Nx_wide.v ~lo:(f (Float.ldexp 1. (-60))) (f 1.) in
          let s = Nx.Ptree.axpy (alone ()) (f 3.) w w in
          equal same (4., Float.ldexp 1. (-58)) (scalar s));
      test "a placed number computes on its set" (fun () ->
          let module M = (val Nx.devices [ Nx_support.memory 0 ]) in
          let w = Nx_wide.v ~lo:(f (Float.ldexp 1. (-60))) (f 1.) in
          let placed = Nx.Ptree.place (module Number) M.on w in
          let s = Nx_wide.add placed placed in
          equal ~msg:"placement"
            (Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal)
            M.on
            (Nx.placement (Nx_wide.hi s));
          equal ~msg:"value" same (2., Float.ldexp 1. (-59)) (scalar s));
    ]

(* Edges *)

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
      test "a sum overflowing then meeting an infinity associates by shape"
        (fun () ->
          let m = Float.max_float in
          associates_by_shape
            (Nx_wide.v
               (Nx.create Nx.float64 [| 4; 2; 1 |]
                  [| 0.; 0.; m; 0.; m; 0.; 0.; Float.neg_infinity |])));
      test "a sum of opposite extremes associates by shape" (fun () ->
          let m = Float.max_float in
          let x =
            Nx.create Nx.float64 [| 1; 3; 4 |]
              (Array.init 12 (fun i ->
                   if i = 6 then -.m else if i = 11 then m else 0.))
          in
          associates_by_shape
            (Nx_wide.v
               (Nx.transpose
                  (Nx.broadcast_to [| 2; 4; 3; 1 |] (Nx.transpose x)))));
      test "a sum of no summand is zero" (fun () ->
          let w = Nx_wide.v (Nx.zeros Nx.float64 [| 2; 0 |]) in
          let zh, zl = words (Nx_wide.sum ~axes:[ 1 ] w) in
          equal (array float_exact) [| 0.; 0. |] zh;
          equal (array float_exact) [| 0.; 0. |] zl);
      cases "a sum of -0 terms is +0" ~id:string_of_int [ 1; 2; 3; 4; 5 ]
        (fun n ->
          let w = Nx_wide.v (Nx.full Nx.float64 [| n |] (-0.)) in
          equal same (0., 0.) (scalar (Nx_wide.sum w)));
      test "a sum along a negative axis sums that axis" (fun () ->
          let w =
            Nx_wide.v
              (Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])
          in
          equal (array float_exact) [| 6.; 15. |]
            (fst (words (Nx_wide.sum ~axes:[ -1 ] w))));
      test "operands broadcast" (fun () ->
          let a = Nx_wide.v (Nx.create Nx.float64 [| 2; 1 |] [| 1.; 2. |])
          and b = Nx_wide.v (Nx.create Nx.float64 [| 3 |] [| 1.; 2.; 3. |]) in
          equal (array int) [| 2; 3 |] (Nx.shape (Nx_wide.hi (Nx_wide.add a b))));
      test "v refuses a narrow float" (fun () ->
          invalid ~by:"Nx_wide.v" (fun () ->
              Nx_wide.v (Nx.zeros Nx.float16 [| 2 |])));
      test "v refuses words that do not broadcast" (fun () ->
          invalid ~by:"Nx_wide.v" (fun () ->
              Nx_wide.v
                ~lo:(Nx.zeros Nx.float64 [| 3 |])
                (Nx.zeros Nx.float64 [| 2 |])));
      cases "a sum refuses axes it cannot sum" ~id:fst
        [ ("an axis out of bounds", [ 2 ]); ("a repeated axis", [ 0; -2 ]) ]
        (fun (_, axes) ->
          invalid ~by:"Nx_wide.sum" (fun () ->
              Nx_wide.sum ~axes (Nx_wide.v (Nx.zeros Nx.float64 [| 2; 2 |]))));
    ]

let () =
  exit
    (run "nx wide"
       [
         group "bounds" [ bounds; floors; comparisons; sums ];
         group "laws" laws;
         walks;
         edges;
       ])
