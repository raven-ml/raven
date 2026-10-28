(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Matrices and products: each function against its elements computed in OCaml
   from its operands' by index, at every dtype it takes, over shapes with empty
   and one-element axes. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

let drawn (type v s) (dt : (v, s) D.t) s : (v, s, Nx.host) Nx.t Gen.t =
  let open Gen in
  let n = Array.fold_left ( * ) 1 s in
  let+ data = string_of ~size:(constant (D.bytes dt n)) char in
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  Nx.Repr.of_array Nx.Host.v
    (A.v dt (L.contiguous (Iarray.of_array s))
       (Rig.Buffer.of_string (if data = "" then "\000" else data)))

(* Index arithmetic *)

let numel s = Array.fold_left ( * ) 1 s

let unravel s j =
  let i = Array.make (Array.length s) 0 and j = ref j in
  for a = Array.length s - 1 downto 0 do
    i.(a) <- !j mod s.(a);
    j := !j / s.(a)
  done;
  i

let ravel s i =
  let j = ref 0 in
  Array.iteri (fun a e -> j := (!j * s.(a)) + e) i;
  !j

type batch = Batch : ('v, 's) D.t * ('v, 's, Nx.host) Nx.t -> batch

let batch ~min_rank =
  Gen.with_pp
    (fun ppf (Batch (dt, x)) -> Format.fprintf ppf "%a %a" D.pp dt pp_ints (Nx.shape x))
    (let open Gen in
     let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all in
     let* s =
       array ~size:(int_range min_rank 3)
         (frequency [ (1, constant 0); (1, constant 1); (4, int_range 2 4) ])
     in
     let+ x = drawn dt s in
     Batch (dt, x))

(* Triangles *)

let law_triangle (f : 'v 's. ?k:int -> ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t) keep
    (Batch (dt, x), k) =
  let s = Nx.shape x in
  let r = Array.length s in
  let xs = Nx.to_array x in
  let expected =
    Array.mapi
      (fun j v ->
        let i = unravel s j in
        if keep (i.(r - 1) - i.(r - 2)) k then v else D.zero dt)
      xs
  in
  equal_elements dt expected (Nx.to_array (f ~k x))

let triangles =
  group "triangles"
    [
      prop "tril keeps column - row <= k"
        (Gen.pair (batch ~min_rank:2) (Gen.int_range (-4) 4))
        (law_triangle (fun ?k x -> Nx.tril ?k x) ( <= ));
      prop "triu keeps column - row >= k"
        (Gen.pair (batch ~min_rank:2) (Gen.int_range (-4) 4))
        (law_triangle (fun ?k x -> Nx.triu ?k x) ( >= ));
      test "a vector has no triangle" (fun () ->
          invalid ~by:"Nx.tril" (fun () -> Nx.tril (Nx.zeros Nx.int8 [| 3 |]));
          invalid ~by:"Nx.triu" (fun () -> Nx.triu (Nx.zeros Nx.int8 [||])));
    ]

(* Diagonals *)

let law_diagonal (Batch (dt, x), (offset, (a1, a2))) =
  let s = Nx.shape x in
  let r = Array.length s in
  let a1 = a1 mod r and a2 = a2 mod r in
  if a1 = a2 then
    invalid ~by:"Nx.diagonal" (fun () -> Nx.diagonal ~axis1:a1 ~axis2:a2 x)
  else
    let rest = List.filter (fun a -> a <> a1 && a <> a2) (List.init r Fun.id) in
    let m = s.(a1) and n = s.(a2) in
    let count =
      Stdlib.max 0
        (if offset >= 0 then Stdlib.min m (n - offset)
         else Stdlib.min (m + offset) n)
    in
    let lead = Array.of_list (List.map (fun a -> s.(a)) rest) in
    let out = Array.append lead [| count |] in
    let xs = Nx.to_array x in
    cover "empty diagonal" (count = 0);
    let expected =
      Array.init (numel out) (fun j ->
          let o = unravel out j in
          let d = o.(Array.length o - 1) in
          let i = Array.make r 0 in
          List.iteri (fun k a -> i.(a) <- o.(k)) rest;
          i.(a1) <- (if offset >= 0 then d else d - offset);
          i.(a2) <- (if offset >= 0 then d + offset else d);
          xs.(ravel s i))
    in
    let y = Nx.diagonal ~offset ~axis1:a1 ~axis2:a2 x in
    equal (array int) out (Nx.shape y);
    equal_elements dt expected (Nx.to_array y)

let test_trace () =
  let x = Nx.reshape [| 2; 3; 3 |] (Nx.arange Nx.int32 0 18 1) in
  equal (array int32) [| 12l; 39l |] (Nx.to_array (Nx.trace x));
  equal (array int32) [| 6l; 24l |] (Nx.to_array (Nx.trace ~offset:1 x));
  equal (array int32) [| 0l; 0l |] (Nx.to_array (Nx.trace ~offset:3 x));
  equal (array float_exact) [| 0. |]
    (Nx.to_array (Nx.trace (Nx.zeros Nx.float32 [| 1; 0; 4 |]) |> Nx.flatten));
  invalid ~by:"Nx.trace" (fun () -> Nx.trace (Nx.zeros Nx.bool [| 2; 2 |]))

let test_diag () =
  let v = Nx.create Nx.int16 [| 3 |] [| 1; 2; 3 |] in
  equal (array int) [| 1; 0; 0; 0; 2; 0; 0; 0; 3 |] (Nx.to_array (Nx.diag v));
  equal (array int) [| 0; 1; 0; 0; 0; 0; 2; 0; 0; 0; 0; 3; 0; 0; 0; 0 |]
    (Nx.to_array (Nx.diag ~k:1 v));
  equal (array int) [| 0; 0; 0; 0; 1; 0; 0; 0; 0; 2; 0; 0; 0; 0; 3; 0 |]
    (Nx.to_array (Nx.diag ~k:(-1) v));
  let m = Nx.reshape [| 3; 3 |] (Nx.arange Nx.int16 0 9 1) in
  equal (array int) [| 0; 4; 8 |] (Nx.to_array (Nx.diag m));
  equal (array int) [| 3; 7 |] (Nx.to_array (Nx.diag ~k:(-1) m));
  equal (array int) [| 0; 0 |] (Nx.shape (Nx.diag (Nx.zeros Nx.int16 [| 0 |])));
  invalid ~by:"Nx.diag" (fun () -> Nx.diag (Nx.zeros Nx.int16 [| 1; 1; 1 |]))

let test_transpose () =
  let x = Nx.reshape [| 2; 2; 3 |] (Nx.arange Nx.int8 0 12 1) in
  equal (array int) [| 2; 3; 2 |] (Nx.shape (Nx.matrix_transpose x));
  equal (array int) [| 0; 3; 1; 4; 2; 5; 6; 9; 7; 10; 8; 11 |]
    (Nx.to_array (Nx.matrix_transpose x));
  let v = Nx.arange Nx.int8 0 3 1 in
  equal bool true (Nx.matrix_transpose v == v)

let diagonals =
  group "diagonals"
    [
      prop "diagonal is x at i along axis1 and i + offset along axis2"
        (Gen.pair (batch ~min_rank:2)
           (Gen.pair (Gen.int_range (-4) 4) (Gen.pair Gen.nat Gen.nat)))
        law_diagonal;
      test "trace sums each matrix's diagonal" test_trace;
      test "diag builds from a vector and reads a matrix" test_diag;
      test "matrix_transpose swaps the last two axes" test_transpose;
    ]

(* Products *)

(* [dot a b] by its definition: the last axis of [a] against [b]'s only axis or
   its second to last, the result [a]'s other axes then [b]'s. *)
let reference_dot sa xa sb xb =
  let ra = Array.length sa and rb = Array.length sb in
  let k = sa.(ra - 1) in
  let kb = if rb = 1 then 0 else rb - 2 in
  let la = Array.sub sa 0 (ra - 1) in
  let lb = Array.of_list (List.filteri (fun i _ -> i <> kb) (Array.to_list sb)) in
  let out = Array.append la lb in
  ( out,
    Array.init (numel out) (fun j ->
        let o = unravel out j in
        let ia = Array.sub o 0 (ra - 1) and ib = Array.sub o (ra - 1) (rb - 1) in
        let acc = ref 0l in
        for t = 0 to k - 1 do
          let i = Array.append ia [| t |] in
          let jb = Array.of_list (List.concat [ Array.to_list (Array.sub ib 0 kb); [ t ]; Array.to_list (Array.sub ib kb (rb - 1 - kb)) ]) in
          acc := Int32.add !acc (Int32.mul xa.(ravel sa i) xb.(ravel sb jb))
        done;
        !acc) )

let law_dot (sa, sb) =
  let xa = Array.init (numel sa) (fun i -> Int32.of_int ((i mod 7) - 3)) in
  let xb = Array.init (numel sb) (fun i -> Int32.of_int ((i mod 5) - 2)) in
  let out, expected = reference_dot sa xa sb xb in
  let y = Nx.dot (Nx.create Nx.int32 sa xa) (Nx.create Nx.int32 sb xb) in
  equal (array int) out (Nx.shape y);
  equal (array int32) expected (Nx.to_array y)

let dot_shapes =
  let open Gen in
  let ext = frequency [ (1, constant 1); (1, constant 0); (4, int_range 2 4) ] in
  let* k = ext in
  let* la = array ~size:(int_range 0 2) ext in
  let* rb = int_range 1 3 in
  let* lb = array ~size:(constant (rb - 1)) ext in
  let sa = Array.append la [| k |] in
  let sb =
    if rb = 1 then [| k |]
    else Array.concat [ Array.sub lb 0 (rb - 2); [| k |]; [| lb.(rb - 2) |] ]
  in
  constant (sa, sb)

let test_vdot () =
  let a = Nx.create Nx.complex64 [| 2 |] [| { Complex.re = 1.; im = 1. }; { re = 0.; im = 2. } |] in
  let b = Nx.create Nx.complex64 [| 1; 2 |] [| { Complex.re = 2.; im = 0. }; { re = 1.; im = 1. } |] in
  (* conj(1+i) 2 + conj(2i) (1+i) = (2-2i) + (2-2i) *)
  equal bool true
    (Nx.item [] (Nx.vdot a b) = { Complex.re = 4.; im = -4. });
  equal (array int) [||] (Nx.shape (Nx.vdot a b));
  equal int32 0l (Nx.item [] (Nx.vdot (Nx.zeros Nx.int32 [| 0 |]) (Nx.zeros Nx.int32 [| 0; 3 |])));
  invalid ~by:"Nx.vdot" (fun () -> Nx.vdot (Nx.zeros Nx.int32 [| 2 |]) (Nx.zeros Nx.int32 [| 3 |]))

(* Cross products *)

(* The distance from [r] to the next value of [dt]'s format away from zero,
   the least subnormal step below its normals. *)
let ulp (type s) (dt : (float, s) D.t) r =
  let f = D.float_format dt in
  let r = Float.abs r in
  if r < f.min_normal then f.min_normal *. f.epsilon
  else
    let _, e = Float.frexp r in
    Float.ldexp f.epsilon (e - 1)

(* [x y - u v] as the unevaluated sum [s + t] of two floats, within about
   [2^-104 max |x y| |u v|] of it: each product split exactly by a fused
   multiply-add, their difference by Knuth's two-sum. *)
let difference x y u v =
  let p = x *. y and q = u *. v in
  let pe = Float.fma x y (-.p) and qe = Float.fma u v (-.q) in
  let s = p -. q in
  let z = s -. p in
  let e = (p -. (s -. z)) +. (-.q -. z) in
  (s, e +. (pe -. qe))

(* Component [i] of [a × b] for the 3-vectors at [a.(o)] and [b.(o)]. *)
let component a b o i =
  let at v k = v.(o + ((i + k) mod 3)) in
  difference (at a 1) (at b 2) (at a 2) (at b 1)

(* Vectors of [k] 3-vectors with entries of magnitude [2^lo] to [2^hi], half
   of them pairs within [2^-nudge] of parallel, whose components cancel in
   about [nudge] bits. *)
let vectors ~lo ~hi ~nudge =
  let open Gen in
  let* k = frequency [ (1, constant 0); (4, int_range 1 4) ] in
  let entry =
    let+ m = float_range 0.5 1. and+ e = int_range lo hi and+ neg = bool in
    Float.ldexp (if neg then -.m else m) e
  in
  let* a = array ~size:(constant (3 * k)) entry in
  let* parallel = bool and+ scale = float_range 0.5 2. in
  let+ b =
    if parallel then
      let+ wobble = array ~size:(constant (3 * k)) (float_range (-1.) 1.) in
      Array.mapi
        (fun j x -> x *. scale *. (1. +. Float.ldexp wobble.(j) (-nudge)))
        a
    else array ~size:(constant (3 * k)) entry
  in
  (k, parallel, a, b)

(* [cross a b] of [dt], from [a] and [b] rounded to [dt], within [ulps] of
   the exact components wherever they fit [dt]: the count of nonzero ones
   checked. *)
let within (type s) (dt : (float, s) D.t) ulps (k, _, a, b) =
  let on xs = Nx.cast dt (Nx.create Nx.float64 [| k; 3 |] xs) in
  let ta = on a and tb = on b in
  let a = Nx.to_array (Nx.cast Nx.float64 ta)
  and b = Nx.to_array (Nx.cast Nx.float64 tb) in
  let got = Nx.to_array (Nx.cast Nx.float64 (Nx.cross ta tb)) in
  let f = D.float_format dt in
  let checked = ref 0 in
  for o = 0 to k - 1 do
    for i = 0 to 2 do
      let s, t = component a b (3 * o) i in
      let r = s +. t and v = got.((3 * o) + i) in
      if Float.abs r <= f.max_finite then begin
        if r <> 0. then incr checked;
        if Float.abs ((v -. s) -. t) > ulps *. ulp dt r then
          failf "%a, vector %d component %d: %h, exactly %h" D.pp dt o i v r
      end
    done
  done;
  !checked

let law_cross_float64 =
  prop "cross at float64 is within 2 ulps of its exact components"
    (vectors ~lo:(-20) ~hi:20 ~nudge:20)
    (fun ((k, parallel, _, _) as v) ->
      cover "nearly parallel vectors" (k > 0 && parallel);
      cover "no vector" (k = 0);
      ignore (within D.Float64 2. v))

type narrow = Narrow : (float, 's) D.t -> narrow

(* One law per format, its entries within the square root of its range so
   that most components fit, its parallel pairs within half its precision. *)
let laws_cross_narrow =
  List.map
    (fun (Narrow dt) ->
      let f = D.float_format dt in
      let _, top = Float.frexp f.max_finite in
      let _, bottom = Float.frexp f.min_normal in
      let v =
        vectors ~lo:(bottom / 2) ~hi:((top / 2) - 1)
          ~nudge:(Stdlib.max 1 (f.fraction_bits / 2))
      in
      prop
        (Format.asprintf "cross at %a is within one ulp of its exact components"
           D.pp dt)
        v
        (fun ((k, _, _, _) as v) ->
          let checked = within dt 1. v in
          cover "a nonzero component checked" (k > 0 && checked > 0)))
    [ Narrow D.Float32; Narrow D.Float16; Narrow D.Bfloat16;
      Narrow D.Float8_e4m3fn; Narrow D.Float8_e5m2; Narrow D.Float4_e2m1fn ]

let test_cross () =
  let v dt xs = Nx.create dt [| Array.length xs |] xs in
  equal_elements D.Float32 [| 0.; 0.; 1. |]
    (Nx.to_array (Nx.cross (v Nx.float32 [| 1.; 0.; 0. |]) (v Nx.float32 [| 0.; 1.; 0. |])));
  equal_elements D.Int32 [| -3l; 6l; -3l |]
    (Nx.to_array (Nx.cross (v Nx.int32 [| 1l; 2l; 3l |]) (v Nx.int32 [| 4l; 5l; 6l |])));
  (* 100 × 100 is 10000, 16 modulo 256. *)
  equal_elements D.Int8 [| 16; 0; 0 |]
    (Nx.to_array (Nx.cross (v Nx.int8 [| 0; 100; 0 |]) (v Nx.int8 [| 0; 0; 100 |])));
  let z re im = { Complex.re; im } in
  equal_elements D.Complex64 [| z 0. 0.; z 0. 0.; z 0. 1. |]
    (Nx.to_array
       (Nx.cross (v Nx.complex64 [| z 0. 1.; z 0. 0.; z 0. 0. |])
          (v Nx.complex64 [| z 0. 0.; z 1. 0.; z 0. 0. |])));
  (* Products of 1e40 past float32's range, a difference bfloat16 holds:
     [1e20 (b2 - b1)], one ulp of 1e20 times 1e20. *)
  let big = D.of_float D.Bfloat16 1e20 in
  let next = big +. ulp D.Bfloat16 big in
  equal_elements D.Bfloat16 [| big *. (next -. big); 0.; 0. |]
    (Nx.to_array
       (Nx.cross (v Nx.bfloat16 [| 0.; big; big |]) (v Nx.bfloat16 [| 0.; big; next |])));
  (* IEEE 754's difference where a product is infinite or the difference is
     zero, at float32 and float64: [1 - inf], [inf 0], and [-0 - +0]. *)
  let specials (type s) (dt : (float, s) D.t) =
    let at xs = Nx.cast dt (v Nx.float64 xs) in
    let got a b = Nx.to_array (Nx.cast Nx.float64 (Nx.cross (at a) (at b))) in
    equal_elements D.Float64 [| Float.neg_infinity; Float.nan; 0. |]
      (got [| 0.; 1.; Float.infinity |] [| 0.; 1.; 1. |]);
    equal_elements D.Float64 [| -0.; 0.; 0. |]
      (got [| 0.; -1.; 0. |] [| 0.; 1.; 0. |])
  in
  specials D.Float32;
  specials D.Float64;
  (* Products of 1e200 past float64's range: their difference's infinity. *)
  equal_elements D.Float64 [| Float.neg_infinity; 0.; 0. |]
    (Nx.to_array (Nx.cross (v Nx.float64 [| 0.; 1.; 1e200 |]) (v Nx.float64 [| 0.; 1e200; 1. |])));
  (* The axis of the broadcast shape: [e0 × e2] and [e1 × e2] in the
     columns. *)
  let rows = Nx.create Nx.float64 [| 3; 2 |] [| 1.; 0.; 0.; 1.; 0.; 0. |] in
  equal_elements D.Float64 [| 0.; 1.; -1.; 0.; 0.; 0. |]
    (Nx.to_array (Nx.cross ~axis:0 rows (Nx.create Nx.float64 [| 3; 1 |] [| 0.; 0.; 1. |])));
  equal (array int) [| 2; 0; 3 |]
    (Nx.shape (Nx.cross (Nx.zeros Nx.float32 [| 2; 0; 3 |]) (Nx.zeros Nx.float32 [| 3 |])))

let cross_refusals =
  cases "cross refusals" ~name:fst
    [
      ("2 elements", fun () -> ignore (Nx.cross (Nx.zeros Nx.float32 [| 2 |]) (Nx.zeros Nx.float32 [| 2 |])));
      ("one operand broadcast along the axis",
       fun () -> ignore (Nx.cross (Nx.zeros Nx.float32 [| 2; 3 |]) (Nx.zeros Nx.float32 [| 2; 1 |])));
      ("an axis outside the shape", fun () -> ignore (Nx.cross ~axis:1 (Nx.zeros Nx.float32 [| 3 |]) (Nx.zeros Nx.float32 [| 3 |])));
      ("booleans", fun () -> ignore (Nx.cross (Nx.zeros Nx.bool [| 3 |]) (Nx.zeros Nx.bool [| 3 |])));
    ]
    (fun (_, f) -> invalid ~by:"Nx.cross" f)

let products =
  group "products"
    ([
      prop "dot sums a's last axis against b's"
        (Gen.with_pp (fun ppf (a, b) -> Format.fprintf ppf "%a . %a" pp_ints a pp_ints b) dot_shapes)
        law_dot;
      test "vdot conjugates a and flattens both" test_vdot;
      law_cross_float64;
      test "cross at the values its definition gives" test_cross;
      cross_refusals;
      cases "dot refusals" ~name:fst
        [
          ("a 0-d operand", fun () -> ignore (Nx.dot (Nx.scalar Nx.int32 1l) (Nx.zeros Nx.int32 [| 2 |])));
          ("summed extents differ", fun () -> ignore (Nx.dot (Nx.zeros Nx.int32 [| 2 |]) (Nx.zeros Nx.int32 [| 3; 2 |])));
          ("booleans", fun () -> ignore (Nx.dot (Nx.zeros Nx.bool [| 2 |]) (Nx.zeros Nx.bool [| 2 |])));
        ]
        (fun (_, f) -> invalid ~by:"Nx.dot" f);
    ]
    @ laws_cross_narrow)

(* Named products *)

(* Each law draws a dtype that is not boolean and small integers, exact in
   every format; an operand is those integers cast to the dtype, which keeps
   an integer modulo its width. The reference computes in OCaml's [int], and
   the result of the dtype is that integer cast: a ring map for integers, and
   one rounding of an exact value for the floats, which accumulate in a wider
   format exactly and round once. *)
type numeric = Numeric : ('v, 's) D.t -> numeric

let pp_numeric ppf (Numeric dt) = D.pp ppf dt

let numerics =
  List.filter_map
    (fun (D.Any dt) ->
      if D.is D.Boolean dt then None else Some (Numeric dt))
    D.all

let numeric = Gen.of_list ~pp:pp_numeric numerics

let of_ints (type v s) (dt : (v, s) D.t) s xs : (v, s, Nx.host) Nx.t =
  Nx.cast dt (Nx.create Nx.int64 s (Array.map Int64.of_int xs))

let ints s = Gen.array ~size:(Gen.constant (numel s)) (Gen.int_range (-2) 2)

let extent =
  Gen.frequency
    [ (1, Gen.constant 0); (1, Gen.constant 1); (4, Gen.int_range 2 3) ]

(* Equal as numbers: a product of zero takes IEEE 754's sign, which an
   integer reference has none of. *)
let numbers (type v s) (dt : (v, s) D.t) =
  let same (a : v) (b : v) =
    match D.kind dt with
    | D.Float -> a = b
    | D.Complex -> a.Complex.re = b.Complex.re && a.im = b.im
    | D.Signed | D.Unsigned | D.Boolean -> a = b
  in
  Testable.make ~pp:(Format.pp_print_list (D.pp_value dt)) ~equal:(fun a b ->
      List.length a = List.length b && List.for_all2 same a b)

let equal_numbers dt a b = equal (numbers dt) (Array.to_list a) (Array.to_list b)

(* [y] against the reference [want] of shape [s], cast to [y]'s dtype. *)
let agrees (type v s) (dt : (v, s) D.t) s want (y : (v, s, Nx.host) Nx.t) =
  equal ~msg:"shape" (array int) s (Nx.shape y);
  equal_numbers dt (Nx.to_array (of_ints dt s want)) (Nx.to_array y)

(* The sum over [pairs] of [a]'s axes against [b]'s, with no batch: the
   result's axes [a]'s others then [b]'s. *)
let reference_pairs sa xa sb xb pairs =
  let ra = Array.length sa and rb = Array.length sb in
  let la = List.map fst pairs and lb = List.map snd pairs in
  let fa = List.filter (fun i -> not (List.mem i la)) (List.init ra Fun.id) in
  let fb = List.filter (fun i -> not (List.mem i lb)) (List.init rb Fun.id) in
  let out = Array.of_list (List.map (Array.get sa) fa @ List.map (Array.get sb) fb) in
  let summed = Array.of_list (List.map (fun (i, _) -> sa.(i)) pairs) in
  let want =
    Array.init (numel out) (fun j ->
        let o = unravel out j in
        let acc = ref 0 in
        for t = 0 to numel summed - 1 do
          let u = unravel summed t in
          let ia = Array.make ra 0 and ib = Array.make rb 0 in
          List.iteri (fun k i -> ia.(i) <- o.(k)) fa;
          List.iteri (fun k i -> ib.(i) <- o.(List.length fa + k)) fb;
          List.iteri
            (fun k (i, i') ->
              ia.(i) <- u.(k);
              ib.(i') <- u.(k))
            pairs;
          acc := !acc + (xa.(ravel sa ia) * xb.(ravel sb ib))
        done;
        !acc)
  in
  (out, want)

type tensordot_case = {
  td_dt : numeric;
  sa : int array;
  sb : int array;
  pairs : (int * int) list;
  default : bool;
  xa : int array;
  xb : int array;
}

(* Axes paired in a drawn order, [b]'s placed among its free axes; one case
   in three takes the default, [a]'s last two against [b]'s first two. *)
let tensordot_case =
  let open Gen in
  with_pp
    (fun ppf c ->
      Format.fprintf ppf "%a %a . %a over %s%s" pp_numeric c.td_dt pp_ints
        c.sa pp_ints c.sb
        (String.concat ", "
           (List.map (fun (i, j) -> Printf.sprintf "%d-%d" i j) c.pairs))
        (if c.default then " (default)" else ""))
    (let* td_dt = numeric in
     let* default = frequency [ (1, constant true); (2, constant false) ] in
     let* sa = array ~size:(int_range (if default then 2 else 0) 3) extent in
     let ra = Array.length sa in
     let* c = if default then constant 2 else int_range 0 (Stdlib.min ra 2) in
     let* order = permutation (List.init ra Fun.id) in
     let la =
       if default then [ ra - 2; ra - 1 ] else List.filteri (fun k _ -> k < c) order
     in
     let* free = array ~size:(int_range 0 (3 - c)) extent in
     let rb = c + Array.length free in
     let* border = permutation (List.init rb Fun.id) in
     let lb = if default then [ 0; 1 ] else List.filteri (fun k _ -> k < c) border in
     let sb = Array.make rb 0 in
     List.iter2 (fun i j -> sb.(j) <- sa.(i)) la lb;
     let rest = List.filter (fun j -> not (List.mem j lb)) (List.init rb Fun.id) in
     List.iteri (fun k j -> sb.(j) <- free.(k)) rest;
     let* xa = ints sa in
     let+ xb = ints sb in
     { td_dt; sa; sb; pairs = List.combine la lb; default; xa; xb })

let law_tensordot c =
  let (Numeric dt) = c.td_dt in
  cover "the default axes" c.default;
  cover "no axis paired" (c.pairs = []);
  cover "an empty sum" (List.exists (fun (i, _) -> c.sa.(i) = 0) c.pairs);
  let axes = (List.map fst c.pairs, List.map snd c.pairs) in
  let a = of_ints dt c.sa c.xa and b = of_ints dt c.sb c.xb in
  let y = if c.default then Nx.tensordot a b else Nx.tensordot ~axes a b in
  let out, want = reference_pairs c.sa c.xa c.sb c.xb c.pairs in
  agrees dt out want y

(* Two operands sharing a last extent, either of them possibly 0-d. *)
let inner_case =
  let open Gen in
  let* dt = numeric in
  let* k = extent in
  let* la = array ~size:(int_range 0 2) extent in
  let* lb = array ~size:(int_range 0 2) extent in
  let* za = frequency [ (1, constant true); (5, constant false) ] in
  let* zb = frequency [ (1, constant true); (5, constant false) ] in
  let sa = if za then [||] else Array.append la [| k |] in
  let sb = if zb then [||] else Array.append lb [| k |] in
  let* xa = ints sa in
  let+ xb = ints sb in
  (dt, sa, xa, sb, xb)

let law_inner (Numeric dt, sa, xa, sb, xb) =
  let ra = Array.length sa and rb = Array.length sb in
  cover "a 0-d operand" (ra = 0 || rb = 0);
  cover "an empty last axis" (ra > 0 && rb > 0 && sa.(ra - 1) = 0);
  let y = Nx.inner (of_ints dt sa xa) (of_ints dt sb xb) in
  if ra = 0 || rb = 0 then
    let s = if ra = 0 then sb else sa in
    let c = if ra = 0 then xa.(0) else xb.(0) in
    agrees dt s (Array.map (( * ) c) (if ra = 0 then xb else xa)) y
  else
    let out, want = reference_pairs sa xa sb xb [ (ra - 1, rb - 1) ] in
    agrees dt out want y

let law_outer (Numeric dt, sa, xa, sb, xb) =
  let y = Nx.outer (of_ints dt sa xa) (of_ints dt sb xb) in
  let n = numel sa and m = numel sb in
  agrees dt [| n; m |] (Array.init (n * m) (fun j -> xa.(j / m) * xb.(j mod m))) y

let pair_case =
  let open Gen in
  let* dt = numeric in
  let* sa = array ~size:(int_range 0 3) extent in
  let* sb = array ~size:(int_range 0 3) extent in
  let* xa = ints sa in
  let+ xb = ints sb in
  (dt, sa, xa, sb, xb)

(* Kronecker's element at [i sb + j], the shorter shape led by ones. *)
let law_kron (Numeric dt, sa, xa, sb, xb) =
  let r = Stdlib.max (Array.length sa) (Array.length sb) in
  let pad s = Array.append (Array.make (r - Array.length s) 1) s in
  let pa = pad sa and pb = pad sb in
  cover "ranks differ" (Array.length sa <> Array.length sb);
  cover "an empty axis" (Array.exists (( = ) 0) (Array.append pa pb));
  let out = Array.init r (fun k -> pa.(k) * pb.(k)) in
  let want =
    Array.init (numel out) (fun j ->
        let o = unravel out j in
        let i = Array.mapi (fun k e -> e / pb.(k)) o in
        let i' = Array.mapi (fun k e -> e mod pb.(k)) o in
        xa.(ravel pa i) * xb.(ravel pb i'))
  in
  agrees dt out want (Nx.kron (of_ints dt sa xa) (of_ints dt sb xb))

(* Operands with one axis [k] each at one place, the last or a nonnegative
   one both have, their other axes suffixes of one shape with some extents
   [1]: they broadcast. *)
let vecdot_case =
  let open Gen in
  let* dt = numeric in
  let* k = extent in
  let* lead = array ~size:(int_range 0 2) extent in
  let l = Array.length lead in
  let side =
    let* drop = int_range 0 l in
    let+ ones = array ~size:(constant l) bool in
    Array.init (l - drop) (fun i ->
        if ones.(i + drop) then 1 else lead.(i + drop))
  in
  let* la = side in
  let* lb = side in
  let* last = bool in
  let* p = int_range 0 (Stdlib.min (Array.length la) (Array.length lb)) in
  let ins s i =
    Array.concat [ Array.sub s 0 i; [| k |]; Array.sub s i (Array.length s - i) ]
  in
  let axis, sa, sb =
    if last then (-1, ins la (Array.length la), ins lb (Array.length lb))
    else (p, ins la p, ins lb p)
  in
  let* xa = ints sa in
  let+ xb = ints sb in
  (dt, axis, sa, xa, sb, xb)

(* [Σ_t a(..t..) b(..t..)] over each operand's own axis, the others
   broadcast. *)
let law_vecdot (Numeric dt, axis, sa, xa, sb, xb) =
  let ia = if axis < 0 then Array.length sa - 1 else axis in
  let ib = if axis < 0 then Array.length sb - 1 else axis in
  let others s i = Array.of_list (List.filteri (fun j _ -> j <> i) (Array.to_list s)) in
  let oa = others sa ia and ob = others sb ib in
  let l = Stdlib.max (Array.length oa) (Array.length ob) in
  let pad s = Array.append (Array.make (l - Array.length s) 1) s in
  let pa = pad oa and pb = pad ob in
  let lead = Array.init l (fun i -> if pa.(i) = 1 then pb.(i) else pa.(i)) in
  cover "an operand broadcast" (pa <> pb);
  cover "an axis other than the last" (axis >= 0 && ia < Array.length sa - 1);
  let k = sa.(ia) in
  let read s i pad o t =
    let lo = Array.mapi (fun j e -> if pad.(j) = 1 then 0 else e) o in
    let lo = Array.sub lo (l - (Array.length s - 1)) (Array.length s - 1) in
    let idx = Array.concat [ Array.sub lo 0 i; [| t |]; Array.sub lo i (Array.length lo - i) ] in
    ravel s idx
  in
  let want =
    Array.init (numel lead) (fun j ->
        let o = unravel lead j in
        let acc = ref 0 in
        for t = 0 to k - 1 do
          acc := !acc + (xa.(read sa ia pa o t) * xb.(read sb ib pb o t))
        done;
        !acc)
  in
  agrees dt lead want (Nx.vecdot ~axis (of_ints dt sa xa) (of_ints dt sb xb))

(* A chain of 2 to 4 operands of extents [e0 … en], an end 1-d at times. *)
let chain_case =
  let open Gen in
  let* dt = numeric in
  let* n = int_range 2 4 in
  let* e = array ~size:(constant (n + 1)) extent in
  let* row = bool in
  let* column = bool in
  let shape i =
    if i = 0 && row then [| e.(1) |]
    else if i = n - 1 && column then [| e.(n - 1) |]
    else [| e.(i); e.(i + 1) |]
  in
  let rec all = function
    | [] -> constant []
    | g :: gs ->
        let* x = g in
        let+ xs = all gs in
        x :: xs
  in
  let+ xs = all (List.init n (fun i -> ints (shape i))) in
  (dt, n, e, row, column, List.mapi (fun i x -> (shape i, x)) xs)

(* The product in [int], left to right, each end's unit axis kept, then
   dropped. *)
let law_multi_dot (Numeric dt, n, e, row, column, xs) =
  cover "a row" row;
  cover "a column" column;
  cover "an empty inner extent" (Array.exists (( = ) 0) (Array.sub e 1 (n - 1)));
  let m0 = if row then 1 else e.(0) and mn = if column then 1 else e.(n) in
  let mat i x =
    let r = if i = 0 then m0 else e.(i) in
    let c = if i = n - 1 then mn else e.(i + 1) in
    (r, c, x)
  in
  let mul (r, k, x) (_, c, y) =
    ( r, c,
      Array.init (r * c) (fun j ->
          let i = j / c and l = j mod c in
          let acc = ref 0 in
          for t = 0 to k - 1 do
            acc := !acc + (x.((i * k) + t) * y.((t * c) + l))
          done;
          !acc) )
  in
  let _, _, want =
    List.fold_left mul (mat 0 (snd (List.hd xs)))
      (List.mapi (fun i (_, x) -> mat (i + 1) x) (List.tl xs))
  in
  let out =
    Array.of_list
      ((if row then [] else [ e.(0) ]) @ if column then [] else [ e.(n) ])
  in
  agrees dt out want
    (Nx.multi_dot (List.map (fun (s, x) -> of_ints dt s x) xs))

(* Powers of matrices of [-1], [0] and [1] up to the 6th: at most 3^5 in
   magnitude, exact in every float format of 11 bits of precision or more. *)
let power_case =
  let open Gen in
  let exact =
    List.filter
      (fun (Numeric dt) ->
        match dt with
        | D.Float8_e4m3fn | D.Float8_e5m2 | D.Float4_e2m1fn -> false
        | _ -> true)
      numerics
  in
  let* dt = of_list ~pp:pp_numeric exact in
  let* m = int_range 0 3 in
  let* lead = array ~size:(int_range 0 1) extent in
  let s = Array.append lead [| m; m |] in
  let* n = int_range 0 6 in
  let+ x = array ~size:(constant (numel s)) (int_range (-1) 1) in
  (dt, s, n, x)

let law_matrix_power (Numeric dt, s, n, x) =
  let m = s.(Array.length s - 1) in
  cover "n = 0" (n = 0);
  cover "an odd n of 3 or more" (n >= 3 && n land 1 = 1);
  cover "an empty matrix" (m = 0);
  let mm = m * m in
  let product p q =
    Array.init mm (fun j ->
        let i = j / m and l = j mod m in
        let acc = ref 0 in
        for t = 0 to m - 1 do
          acc := !acc + (p.((i * m) + t) * q.((t * m) + l))
        done;
        !acc)
  in
  let identity = Array.init mm (fun j -> if j / m = j mod m then 1 else 0) in
  let want =
    Array.concat
      (List.init (numel s / Stdlib.max 1 mm) (fun b ->
           let a = Array.sub x (b * mm) mm in
           let rec pow k = if k = 0 then identity else product (pow (k - 1)) a in
           pow n))
  in
  let want = if mm = 0 then [||] else want in
  agrees dt s want (Nx.matrix_power (of_ints dt s x) n)

(* Negative powers of matrices within [1/2] of [3 I], whose inverses are
   near [I / 3], against [inv a] multiplied [n] times, within [tol] of the
   largest element. *)
type inexact = Inexact : ('v, 's) D.t * float -> inexact

let inverse_case =
  let open Gen in
  let* (Inexact (dt, tol)) =
    of_list
      ~pp:(fun ppf (Inexact (dt, _)) -> D.pp ppf dt)
      [
        Inexact (D.Float32, 1e-5);
        Inexact (D.Float64, 1e-13);
        Inexact (D.Complex64, 1e-5);
        Inexact (D.Complex128, 1e-13);
      ]
  in
  let* m = int_range 1 3 in
  let* lead = array ~size:(int_range 0 1) (int_range 1 2) in
  let s = Array.append lead [| m; m |] in
  let* n = int_range 1 6 in
  let* re = array ~size:(constant (numel s)) (int_range (-1) 1) in
  let+ im = array ~size:(constant (numel s)) (int_range (-1) 1) in
  (Inexact (dt, tol), s, n, re, im)

let law_negative_power (Inexact (dt, tol), s, n, re, im) =
  let m = s.(Array.length s - 1) in
  cover "a batch" (Array.length s > 2);
  cover "a power of 3 or more" (n >= 3);
  let part xs = Nx.create Nx.float64 s (Array.map (fun k -> Float.of_int k /. 6.) xs) in
  let diagonal = Nx.mul_s (Nx.eye Nx.float64 m) 3. in
  let a =
    Nx.cast dt
      (Nx.complex Nx.complex128 ~re:(Nx.add (part re) diagonal) ~im:(part im))
  in
  let r = Nx.inv a in
  let want = List.fold_left (fun p _ -> Nx.matmul p r) r (List.init (n - 1) Fun.id) in
  let wide x = Nx.cast Nx.complex128 x in
  let largest x = Nx.item [] (Nx.max (Nx.magnitude Nx.float64 x)) in
  let y = Nx.matrix_power a (-n) in
  equal ~msg:"shape" (array int) s (Nx.shape y);
  at_most ~msg:"error" float_exact ~than:(tol *. largest (wide want))
    (largest (Nx.sub (wide y) (wide want)))

(* A chain whose every order's work passes an [int]: three
   [2^21 × 2^21] traced operands, so nothing is computed. *)
type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

let mark i ~by op = Nx.Prim.results ~by (fun _ f -> Nx.Prim.traced i f Mark) op

let test_saturated_chain () =
  let side = 1 lsl 21 in
  let b = Nx.broadcast_to [| side; side |] (Nx.scalar Nx.float32 1.) in
  Nx.Prim.interpret ~name:"test.chain" Values mark (fun i ->
      let t = Nx.Prim.traced i (Nx.Prim.form b) Mark in
      equal (array int) [| side; side |] (Nx.shape (Nx.multi_dot [ t; t; t ])))

let test_conjugates () =
  let z re im = { Complex.re; im } in
  let a = Nx.create Nx.complex64 [| 2 |] [| z 1. 1.; z 0. 2. |] in
  let b = Nx.create Nx.complex64 [| 2 |] [| z 2. 0.; z 1. 1. |] in
  (* conj(1+i) 2 + conj(2i) (1+i) = (2-2i) + (2-2i); without conjugation,
     (2+2i) + (-2+2i). *)
  equal_elements D.Complex64 [| z 4. (-4.) |] (Nx.to_array (Nx.vecdot a b));
  equal_elements D.Complex64 [| z 0. 4. |] (Nx.to_array (Nx.inner a b));
  equal_elements D.Complex64 [| z 0. 4. |]
    (Nx.to_array (Nx.tensordot ~axes:([ 0 ], [ 0 ]) a b));
  equal_elements D.Complex64
    [| z 2. 2.; z 0. 2.; z 0. 4.; z (-2.) 2. |]
    (Nx.to_array (Nx.outer a b))

let test_products_cases () =
  let v dt s xs = Nx.create dt s xs in
  (* The textbook Kronecker product of [[1; 2]; [3; 4]] and [[0; 5]; [6; 7]]. *)
  equal (array int32)
    [| 0l; 5l; 0l; 10l; 6l; 7l; 12l; 14l; 0l; 15l; 0l; 20l; 18l; 21l; 24l; 28l |]
    (Nx.to_array
       (Nx.kron (v Nx.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |])
          (v Nx.int32 [| 2; 2 |] [| 0l; 5l; 6l; 7l |])));
  (* A rotation by a quarter turn: its 4th power the identity, its -1st its
     transpose. *)
  let r = v Nx.float64 [| 2; 2 |] [| 0.; -1.; 1.; 0. |] in
  equal_elements D.Float64 [| 1.; 0.; 0.; 1. |] (Nx.to_array (Nx.matrix_power r 4));
  equal_numbers D.Float64 [| 0.; 1.; -1.; 0. |] (Nx.to_array (Nx.matrix_power r (-1)));
  let d = v Nx.float64 [| 2; 2 |] [| 2.; 0.; 0.; 4. |] in
  equal_elements D.Float64 [| 0.25; 0.; 0.; 0.0625 |]
    (Nx.to_array (Nx.matrix_power d (-2)));
  equal (array int) [| 3; 0; 0 |]
    (Nx.shape (Nx.matrix_power (Nx.zeros Nx.int8 [| 3; 0; 0 |]) 5));
  (* 4 × 1, 1 × 4, 4 × 1: the right pair first does 4 + 4 multiply-adds,
     where the left does 16 + 16. The bits agree either way here; the shape
     is the chain's. *)
  let col = v Nx.float32 [| 4; 1 |] [| 1.; 2.; 3.; 4. |] in
  let row = v Nx.float32 [| 1; 4 |] [| 1.; 1.; 1.; 1. |] in
  equal_elements D.Float32 [| 10.; 20.; 30.; 40. |]
    (Nx.to_array (Nx.multi_dot [ col; row; col ]));
  (* (h ones) 3 is the cheaper order, 2 + 1 against 2 + 2. float16 keeps
     [h ones = 2049] in float32 and rounds [6147] once, to 6148; rounded to
     float16 first, 2049 would be 2048 and the result 6144. *)
  let h = v Nx.float16 [| 1; 2 |] [| 2048.; 1. |] in
  let ones = v Nx.float16 [| 2; 1 |] [| 1.; 1. |] in
  let three = v Nx.float16 [| 1; 1 |] [| 3. |] in
  equal_elements D.Float16 [| 6148. |]
    (Nx.to_array (Nx.multi_dot [ h; ones; three ]));
  (* The order of fewest multiply-adds, seen where the other overflows:
     2 × 1, 1 × 2, 2 × 1 is cheaper as a (b c), 4 against 8, where
     b c = 1 - 1 cancels and a b = 1e60 is infinite; 1 × 2, 2 × 1, 1 × 2 is
     cheaper as (a b) c, the mirror. *)
  let f32 s xs = v Nx.float32 s xs in
  equal_numbers D.Float32 [| 0.; 0. |]
    (Nx.to_array
       (Nx.multi_dot
          [ f32 [| 2; 1 |] [| 1e30; 1e30 |]; f32 [| 1; 2 |] [| 1e30; 1e30 |];
            f32 [| 2; 1 |] [| 1e-30; -1e-30 |] ]));
  equal_numbers D.Float32 [| 0.; 0. |]
    (Nx.to_array
       (Nx.multi_dot
          [ f32 [| 1; 2 |] [| 1e-30; -1e-30 |]; f32 [| 2; 1 |] [| 1e30; 1e30 |];
            f32 [| 1; 2 |] [| 1e30; 1e30 |] ]));
  equal (array int) [||]
    (Nx.shape (Nx.multi_dot [ v Nx.int8 [| 2 |] [| 1; 2 |]; v Nx.int8 [| 2 |] [| 3; 4 |] ]));
  equal (array int) [| 2; 3 |]
    (Nx.shape
       (Nx.tensordot ~axes:([], [])
          (Nx.zeros Nx.int8 [| 2 |]) (Nx.zeros Nx.int8 [| 3 |])))

let product_refusals =
  let z s = Nx.zeros Nx.int32 s and b s = Nx.zeros Nx.bool s in
  cases "named product refusals" ~name:(fun (by, what, _) -> by ^ ", " ^ what)
    [
      ("Nx.tensordot", "default axes of a vector", fun () -> ignore (Nx.tensordot (z [| 2 |]) (z [| 2; 2 |])));
      ("Nx.tensordot", "lists of two lengths", fun () -> ignore (Nx.tensordot ~axes:([ 0 ], []) (z [| 2 |]) (z [| 2 |])));
      ("Nx.tensordot", "a repeated axis", fun () -> ignore (Nx.tensordot ~axes:([ 0; 0 ], [ 0; 1 ]) (z [| 2; 2 |]) (z [| 2; 2 |])));
      ("Nx.tensordot", "extents differ", fun () -> ignore (Nx.tensordot ~axes:([ 0 ], [ 0 ]) (z [| 2 |]) (z [| 3 |])));
      ("Nx.tensordot", "booleans", fun () -> ignore (Nx.tensordot ~axes:([ 0 ], [ 0 ]) (b [| 2 |]) (b [| 2 |])));
      ("Nx.inner", "last extents differ", fun () -> ignore (Nx.inner (z [| 2 |]) (z [| 3 |])));
      ("Nx.inner", "booleans", fun () -> ignore (Nx.inner (b [| 2 |]) (b [| 2 |])));
      ("Nx.outer", "booleans", fun () -> ignore (Nx.outer (b [| 2 |]) (b [| 2 |])));
      ("Nx.kron", "booleans", fun () -> ignore (Nx.kron (b [| 2 |]) (b [| 2 |])));
      ("Nx.vecdot", "booleans", fun () -> ignore (Nx.vecdot (b [| 2 |]) (b [| 2 |])));
      ("Nx.kron", "33 axes of extent 2 together",
       fun () -> ignore (Nx.kron (z (Array.make 16 2)) (Nx.zeros Nx.int32 (Array.make 17 2))));
      ("Nx.vecdot", "extents differ along the axis", fun () -> ignore (Nx.vecdot (z [| 3 |]) (z [| 1 |])));
      ("Nx.vecdot", "others do not broadcast", fun () -> ignore (Nx.vecdot (z [| 2; 3 |]) (z [| 4; 3 |])));
      ("Nx.vecdot", "an axis outside", fun () -> ignore (Nx.vecdot ~axis:1 (z [| 3 |]) (z [| 3 |])));
      ("Nx.multi_dot", "one operand", fun () -> ignore (Nx.multi_dot [ z [| 2; 2 |] ]));
      ("Nx.multi_dot", "a vector in the middle", fun () -> ignore (Nx.multi_dot [ z [| 2; 2 |]; z [| 2 |]; z [| 2; 2 |] ]));
      ("Nx.multi_dot", "a 3-d end", fun () -> ignore (Nx.multi_dot [ z [| 1; 2; 2 |]; z [| 2; 2 |] ]));
      ("Nx.multi_dot", "inner extents differ", fun () -> ignore (Nx.multi_dot [ z [| 2; 3 |]; z [| 2; 2 |] ]));
      ("Nx.multi_dot", "booleans", fun () -> ignore (Nx.multi_dot [ b [| 2; 2 |]; b [| 2; 2 |] ]));
      ("Nx.matrix_power", "a vector", fun () -> ignore (Nx.matrix_power (z [| 2 |]) 2));
      ("Nx.matrix_power", "not square", fun () -> ignore (Nx.matrix_power (z [| 2; 3 |]) 2));
      ("Nx.matrix_power", "booleans", fun () -> ignore (Nx.matrix_power (b [| 2; 2 |]) 0));
      ("Nx.matrix_power", "an integer's inverse", fun () -> ignore (Nx.matrix_power (z [| 2; 2 |]) (-1)));
    ]
    (fun (by, _, f) -> invalid ~by f)

let named_products =
  group "named products"
    [
      prop "tensordot sums the paired axes" tensordot_case law_tensordot;
      prop "inner sums the last axes" inner_case law_inner;
      prop "outer multiplies every pair" pair_case law_outer;
      prop "kron is a times b at i sb + j" pair_case law_kron;
      prop "vecdot sums along each operand's axis, the others broadcast"
        vecdot_case law_vecdot;
      prop "multi_dot is the chain's product" chain_case law_multi_dot;
      prop "matrix_power is n products" power_case law_matrix_power;
      prop "a negative power is the inverse's" inverse_case law_negative_power;
      test "a chain of saturated work has an order" test_saturated_chain;
      test "vecdot conjugates a; inner, tensordot and outer do not"
        test_conjugates;
      test "named products at the values their definitions give"
        test_products_cases;
      product_refusals;
    ]

let () =
  exit (run "nx matrices" [ triangles; diagonals; products; named_products ])
