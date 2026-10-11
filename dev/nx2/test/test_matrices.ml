(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Matrices and products: each function against its elements computed in OCaml
   from its operands' by index, at every dtype it takes, over shapes with empty
   and one-element axes. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let same_float a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let same (type v s) (dt : (v, s) D.t) (a : v) (b : v) =
  match D.kind dt with
  | D.Float -> same_float a b
  | D.Complex -> same_float a.Complex.re b.Complex.re && same_float a.im b.im
  | D.Signed | D.Unsigned | D.Boolean -> a = b

let elements dt =
  Testable.make ~pp:(Format.pp_print_list (D.pp_value dt)) ~equal:(fun a b ->
      List.length a = List.length b && List.for_all2 (same dt) a b)

let equal_elements dt a b =
  equal (elements dt) (Array.to_list a) (Array.to_list b)

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
    (A.v dt (L.contiguous s)
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

let products =
  group "products"
    [
      prop "dot sums a's last axis against b's"
        (Gen.with_pp (fun ppf (a, b) -> Format.fprintf ppf "%a . %a" pp_ints a pp_ints b) dot_shapes)
        law_dot;
      test "vdot conjugates a and flattens both" test_vdot;
      cases "dot refusals" ~name:fst
        [
          ("a 0-d operand", fun () -> ignore (Nx.dot (Nx.scalar Nx.int32 1l) (Nx.zeros Nx.int32 [| 2 |])));
          ("summed extents differ", fun () -> ignore (Nx.dot (Nx.zeros Nx.int32 [| 2 |]) (Nx.zeros Nx.int32 [| 3; 2 |])));
          ("booleans", fun () -> ignore (Nx.dot (Nx.zeros Nx.bool [| 2 |]) (Nx.zeros Nx.bool [| 2 |])));
        ]
        (fun (_, f) -> invalid ~by:"Nx.dot" f);
    ]

let () = exit (run "nx matrices" [ triangles; diagonals; products ])
