(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The structures the transformations take: what they refuse at the call, how a
   mismatch between two values of one structure is named, and how a signature of
   several arguments is seen as one value whose leaves are named from the
   argument's position. *)

open Windtrap

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let exact () = Oracle.tensor ()
let starts fn m = starts_with ~affix:(fn ^ ": ") m
let pair = Nx.Ptree.(pair tensor tensor)

(* Mismatches *)

(* The structure of the values compared below: a fixed tensor first, so that
   every value has a float leaf, then a list of optional pairs. *)
let rows = Nx.Ptree.(pair tensor (list (option (pair tensor tensor))))

let pp_rows ppf (_, l) =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       (fun ppf -> function
         | None -> Format.pp_print_string ppf "None"
         | Some _ -> Format.pp_print_string ppf "Some"))
    l

let rows_gen =
  let one = vec [| 1. |] in
  Gen.with_pp pp_rows
    (Gen.map
       (fun l ->
         (one, List.map (fun b -> if b then Some (one, one) else None) l))
       (Gen.list ~size:(Gen.int_range 0 3) Gen.bool))

(* The message for the first visit at which [x] and [y] differ, built from their
   visits as the documented rule words it. *)
let expected_mismatch fn ~this x ~that y =
  let path v =
    match v with
    | Nx.Ptree.Leaf p | Report (p, _) -> (
        match Nx.Ptree.Path.to_string p with "" -> "the root" | s -> s)
  in
  let word = function
    | Nx.Ptree.Leaf _ -> "a leaf"
    | Report (_, Int n) -> Printf.sprintf "int %d" n
    | Report (_, Case t) -> Printf.sprintf "case %S" t
    | Report (_, Present true) -> "Some"
    | Report (_, Present false) -> "None"
    | Report (_, Length n) -> Printf.sprintf "length %d" n
  in
  let rec first = function
    | [], [] -> None
    | v :: _, [] ->
        Some
          (Printf.sprintf "%s: %s %s, nothing %s" (path v) (word v) this that)
    | [], w :: _ ->
        Some
          (Printf.sprintf "%s: nothing %s, %s %s" (path w) this (word w) that)
    | v :: vs, w :: ws ->
        if v = w then first (vs, ws)
        else if path v = path w then
          Some
            (Printf.sprintf "%s: %s %s, %s %s" (path v) (word v) this (word w)
               that)
        else
          Some
            (Printf.sprintf "%s: %s %s, %s at %s %s" (path v) (word v) this
               (word w) (path w) that)
  in
  Option.map
    (fun m -> fn ^ ": " ^ m)
    (first (Nx.Ptree.visits rows x, Nx.Ptree.visits rows y))

let mismatch_tests =
  [
    prop "a mismatch names the first visit at which two values differ"
      Gen.(pair rows_gen rows_gen)
      (fun (x, t) ->
        let expected =
          expected_mismatch "Rune.jvp" ~this:"in the parameters" x
            ~that:"in the tangents" t
        in
        let length (_, l) = List.length l in
        classify "equal" (expected = None);
        cover "differ in a length" (length x <> length t);
        cover "differ in a presence" (expected <> None && length x = length t);
        match expected with
        | None -> ignore (Rune.jvp rows Nx.Ptree.unit (fun _ -> ()) x t)
        | Some m ->
            raises (Invalid_argument m) (fun () ->
                Rune.jvp rows Nx.Ptree.unit (fun _ -> ()) x t));
    test "a shape mismatch names the leaf's path and both shapes" (fun () ->
        let _, pb =
          Rune.vjp Nx.Ptree.tensor pair
            (fun x -> (x, Nx.sum x))
            (vec [| 1.; 2. |])
        in
        raises
          (Invalid_argument
             "Rune.vjp: 1: shape [] in the result, [3] in the cotangents")
          (fun () -> pb (vec [| 1.; 1. |], vec [| 1.; 1.; 1. |])));
    test "the root is named the root" (fun () ->
        raises
          (Invalid_argument
             "Rune.jvp': the root: shape [2] in the parameters, [3] in the \
              tangents") (fun () ->
            Rune.jvp' Nx.sin (vec [| 1.; 2. |]) (vec [| 1.; 2.; 3. |])));
    test "a structure whose walk visits one value two ways is refused"
      (fun () ->
        (* A walk that sees the second tensor on every other call. *)
        let calls = ref 0 in
        let module Flaky = struct
          type _ t = Nx.float64_t * Nx.float64_t

          let walk c (a, b) =
            incr calls;
            let a = Nx.Ptree.Walk.tensor c a in
            let b = if !calls mod 2 = 0 then Nx.Ptree.Walk.tensor c b else b in
            (a, b)
        end in
        let flaky = Nx.Ptree.instantiate (module Flaky) in
        raises_match Exn.invalid_arg (fun () ->
            Rune.grad flaky
              (fun (a, b) -> Nx.add (Nx.sum a) (Nx.sum b))
              (vec [| 1. |], vec [| 2. |])));
  ]

(* Signatures *)

(* [f] of k arguments, and its signature. *)
let one a = Nx.sin a
let two a b = Nx.mul a b
let three a b c = (Nx.add a b, Nx.mul b c)
let four a b c d = Nx.sub (Nx.mul a b) (Nx.mul c d)

let signature_tests =
  [
    test "a function of k arguments is itself through a signature" (fun () ->
        let a = vec [| 0.5 |]
        and b = vec [| -1. |]
        and c = vec [| 2. |]
        and d = vec [| 3. |] in
        let open Nx.Ptree in
        equal ~msg:"one" (exact ()) (one a)
          (Rune.remat (tensor @-> returns tensor) one a);
        equal ~msg:"two" (exact ()) (two a b)
          (Rune.remat (tensor @-> tensor @-> returns tensor) two a b);
        let x, y =
          Rune.remat
            (tensor @-> tensor @-> tensor @-> returns (pair tensor tensor))
            three a b c
        in
        equal ~msg:"three, first" (exact ()) (fst (three a b c)) x;
        equal ~msg:"three, second" (exact ()) (snd (three a b c)) y;
        equal ~msg:"four" (exact ()) (four a b c d)
          (Rune.remat
             (tensor @-> tensor @-> tensor @-> tensor @-> returns tensor)
             four a b c d));
    test "a leaf's path starts with its argument's position" (fun () ->
        raises (Invalid_argument "Rune.vmap: 1: 3 rows along axis 0, 0: 2")
          (fun () ->
            Rune.vmap
              Nx.Ptree.(tensor @-> tensor @-> returns tensor)
              Nx.add
              (Nx.zeros f64 [| 2; 1 |])
              (Nx.zeros f64 [| 3; 1 |])));
    test "vmap refuses a consumed argument when given its signature" (fun () ->
        raises
          (Invalid_argument
             "Rune.vmap: the argument at 1 is consumed; only a compiled call \
              consumes its arguments") (fun () ->
            ignore
              (Rune.vmap
                 Nx.Ptree.(tensor @-> consumes tensor @@ returns tensor)
                 Nx.add
                : Nx.float64_t -> Nx.float64_t -> Nx.float64_t)));
    test "remat refuses a consumed argument when given its signature" (fun () ->
        raises
          (Invalid_argument
             "Rune.remat: the argument at 0 is consumed; only a compiled call \
              consumes its arguments") (fun () ->
            ignore
              (Rune.remat Nx.Ptree.(consumes tensor @@ returns tensor) Nx.sin
                : Nx.float64_t -> Nx.float64_t)));
  ]

(* Preconditions at the call *)

type ints = { count : Nx.int32_t; mask : (bool, Nx.bool_elt) Nx.t }

module Ints = struct
  type _ t = ints

  let walk c { count; mask } =
    let open Nx.Ptree.Walk in
    let count = field c "count" tensor count in
    let mask = field c "mask" tensor mask in
    { count; mask }
end

let ints = Nx.Ptree.instantiate (module Ints)

let some_ints () =
  {
    count = Nx.create Nx.int32 [| 1 |] [| 3l |];
    mask = Nx.create Nx.bool [| 1 |] [| true |];
  }

type keyed = {
  w : Nx.float64_t;
  key : Nx.Rng.t;
  mask : (bool, Nx.bool_elt) Nx.t;
}

module Keyed = struct
  type _ t = keyed

  let walk c { w; key; mask } =
    let open Nx.Ptree.Walk in
    let w = field c "w" tensor w in
    let key = field c "key" (structure Nx.Rng.ptree) key in
    let mask = field c "mask" tensor mask in
    { w; key; mask }
end

let keyed = Nx.Ptree.instantiate (module Keyed)

let not_scalar fn got =
  fn ^ ": the objective must return a real or complex scalar, got " ^ got

let no_tensor fn = fn ^ ": the parameters hold no real or complex tensor"
let ints_ () = Nx.create Nx.int32 [| 2 |] [| 1l; 2l |]

let precondition_tests =
  [
    group "a non-scalar objective is refused"
      [
        test "grad'" (fun () ->
            raises
              (Invalid_argument (not_scalar "Rune.grad'" "float64 [2]"))
              (fun () -> Rune.grad' (fun x -> Nx.mul x x) (vec [| 1.; 2. |])));
        test "grad" (fun () ->
            raises
              (Invalid_argument (not_scalar "Rune.grad" "float64 [2]"))
              (fun () ->
                Rune.grad Nx.Ptree.tensor
                  (fun x -> Nx.mul x x)
                  (vec [| 1.; 2. |])));
        test "value_and_grad_aux" (fun () ->
            raises
              (Invalid_argument
                 (not_scalar "Rune.value_and_grad_aux" "float64 [2]"))
              (fun () ->
                Rune.value_and_grad_aux Nx.Ptree.tensor Nx.Ptree.unit
                  (fun x -> (Nx.mul x x, ()))
                  (vec [| 1.; 2. |])));
        test "an integer objective" (fun () ->
            raises
              (Invalid_argument (not_scalar "Rune.grad'" "int32 []"))
              (fun () ->
                Rune.grad' (fun _ -> Nx.scalar Nx.int32 1l) (vec [| 1. |])));
      ];
    test "a scalar objective may have any shape of one element" (fun () ->
        equal ~msg:"[1]" (exact ()) (vec [| 2. |])
          (Rune.grad' (fun x -> Nx.mul x x) (vec [| 1. |]));
        equal ~msg:"[1; 1]" (exact ()) (vec [| 2. |])
          (Rune.grad'
             (fun x -> Nx.reshape [| 1; 1 |] (Nx.mul x x))
             (vec [| 1. |])));
    group "a structure with no real or complex tensor is refused"
      [
        test "grad' of an integer tensor" (fun () ->
            raises
              (Invalid_argument (no_tensor "Rune.grad'"))
              (fun () ->
                Rune.grad' (fun x -> Nx.sum (Nx.cast f64 x)) (ints_ ())));
        test "grad of integers and bools" (fun () ->
            raises
              (Invalid_argument (no_tensor "Rune.grad"))
              (fun () ->
                Rune.grad ints
                  (fun p -> Nx.sum (Nx.cast f64 p.count))
                  (some_ints ())));
        test "grad of unit" (fun () ->
            raises
              (Invalid_argument (no_tensor "Rune.grad"))
              (fun () -> Rune.grad Nx.Ptree.unit (fun () -> scalar 1.) ()));
        test "jvp of unit" (fun () ->
            raises
              (Invalid_argument (no_tensor "Rune.jvp"))
              (fun () ->
                Rune.jvp Nx.Ptree.unit Nx.Ptree.tensor
                  (fun () -> scalar 1.)
                  () ()));
        test "vjp of unit" (fun () ->
            raises
              (Invalid_argument (no_tensor "Rune.vjp"))
              (fun () ->
                Rune.vjp Nx.Ptree.unit Nx.Ptree.tensor (fun () -> scalar 1.) ()));
        test "jacrev' of an integer tensor" (fun () ->
            raises
              (Invalid_argument (no_tensor "Rune.jacrev'"))
              (fun () -> Rune.jacrev' (fun x -> Nx.cast f64 x) (ints_ ())));
      ];
    test "integer, bool and key leaves beside a float one are carried"
      (fun () ->
        let p =
          {
            w = vec [| 1.; -2. |];
            key = Nx.Rng.key 7;
            mask = Nx.create Nx.bool [| 2 |] [| true; false |];
          }
        in
        let g = Rune.grad keyed (fun p -> Nx.sum (Nx.mul p.w p.w)) p in
        equal ~msg:"w" (exact ()) (vec [| 2.; -4. |]) g.w;
        equal ~msg:"key" (array int32) [| 0l; 0l |]
          (Nx.to_array (g.key :> Nx.int32_t));
        equal ~msg:"mask" (exact ())
          (Nx.create Nx.bool [| 2 |] [| false; false |])
          g.mask);
    test "vmap of arguments with no tensor is refused" (fun () ->
        starts "Rune.vmap"
          (Oracle.message (fun () ->
               Rune.vmap
                 Nx.Ptree.(unit @-> returns tensor)
                 (fun () -> scalar 1.)
                 ())));
    test "vmap of a scalar is refused" (fun () ->
        starts "Rune.vmap'"
          (Oracle.message (fun () -> Rune.vmap' Nx.sin (scalar 1.))));
  ]

(* Arithmetic. A float32 vector, a float64 matrix held transposed, and an int32
   counter that the arithmetic carries. The values are small multiples of 1/4,
   whose products and sums are exact in any order. *)

let mixed = Nx.Ptree.(pair tensor (pair tensor tensor))

let mixed_value u m c =
  ( Nx.create Nx.float32 [| 3 |] u,
    (Nx.transpose (Nx.create f64 [| 2; 2 |] m), Nx.create Nx.int32 [| 1 |] c) )

let mixed_w =
  Testable.contramap
    (fun (u, (m, c)) ->
      (Nx.to_array (Nx.cast f64 u), Nx.to_array m, Nx.to_array c))
    (triple (array float_exact) (array float_exact) (array int32))

let x () = mixed_value [| 1.; -2.; 0.5 |] [| 3.; -1.; 2.; 0.25 |] [| 7l |]
let y () = mixed_value [| 0.25; 4.; -3. |] [| -1.; 2.; 5.; 1.5 |] [| 9l |]

let arithmetic_tests =
  [
    test
      "compiled dot and axpy equal eager over floats of two dtypes beside a \
       counter" (fun () ->
        let f a x y = (Nx.Ptree.dot mixed f64 x y, Nx.Ptree.axpy mixed a x y) in
        let g =
          Rune.jit
            Nx.Ptree.(
              tensor @-> mixed @-> mixed @-> returns (pair tensor mixed))
            f
        in
        let a = Nx.scalar Nx.float32 1.5 in
        let d, z = f a (x ()) (y ()) and d', z' = g a (x ()) (y ()) in
        equal ~msg:"dot" (exact ()) d d';
        equal ~msg:"axpy" mixed_w z z');
    test "vmap of axpy over a batch of factors is axpy at each factor"
      (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 0.5; -2.; 0. |] in
        let z =
          Rune.vmap
            Nx.Ptree.(tensor @-> returns mixed)
            (fun a -> Nx.Ptree.axpy mixed a (x ()) (y ()))
            a
        in
        let lane i (u, (m, c)) =
          (Nx.get [ i ] u, (Nx.get [ i ] m, Nx.get [ i ] c))
        in
        for i = 0 to 2 do
          equal
            ~msg:(Printf.sprintf "lane %d" i)
            mixed_w
            (Nx.Ptree.axpy mixed (Nx.get [ i ] a) (x ()) (y ()))
            (lane i z)
        done);
  ]

(* Pairing. A record whose walk reports a case and an integer beside its float
   tensors, as a record of quantities reports each unit, and carries a counter.
   The gradient is the vector the structure's [dot] pairs with tangents, so it
   keeps the reports; a pullback is a tangent map's adjoint under the [dot]s of
   the parameters and of the result. *)

type tagged = {
  tag : string;
  order : int;
  a : Nx.float64_t;
  b : Nx.float64_t;
  count : Nx.int32_t;
}

module Tagged = struct
  type _ t = tagged

  let walk c { tag; order; a; b; count } =
    let open Nx.Ptree.Walk in
    let order = field c "order" int order in
    let a =
      field c "a"
        (fun c a ->
          case c tag;
          tensor c a)
        a
    in
    let b = field c "b" tensor b in
    let count = field c "count" tensor count in
    { tag; order; a; b; count }
end

let tagged = Nx.Ptree.instantiate (module Tagged)

let pp_tagged ppf x =
  Format.fprintf ppf "@[<v>%s %d@,%a@,%a@]" x.tag x.order Nx.pp x.a Nx.pp x.b

(* Draws [n] values of one tag and order: a point and its directions. *)
let tagged_gen n =
  let open Gen in
  let floats k = array ~size:(constant k) (float_range (-2.) 2.) in
  with_pp
    (Format.pp_print_list pp_tagged)
    (let+ tag = of_list [ "m"; "1e3 m"; "K" ]
     and+ order = int_range 0 3
     and+ vs = list ~size:(constant n) (pair (floats 3) (floats 4)) in
     List.map
       (fun (a, b) ->
         {
           tag;
           order;
           a = Nx.create f64 [| 3 |] a;
           b = Nx.create f64 [| 2; 2 |] b;
           count = Nx.create Nx.int32 [| 1 |] [| 5l |];
         })
       vs)

let objective x =
  let k = float_of_int x.order in
  Nx.add
    (Nx.mul (Nx.sum (Nx.sin x.a)) (Nx.sum (Nx.mul x.b x.b)))
    (Nx.mul_s (Nx.sum (Nx.mul x.a x.a)) k)

(* A map from the record to a record of another tag, as a function of quantities
   returns its result in its own unit. *)
let image x =
  {
    x with
    tag = x.tag ^ "^2";
    a = Nx.mul (Nx.sin x.a) (Nx.sum x.b);
    b = Nx.add (Nx.mul x.b x.b) (Nx.mul_s (Nx.sum x.a) (float_of_int x.order));
  }

let dot = Nx.Ptree.dot tagged f64

let visit =
  let equal a b =
    match (a, b) with
    | Nx.Ptree.Leaf p, Nx.Ptree.Leaf q -> Nx.Ptree.Path.equal p q
    | Report (p, r), Report (q, s) -> Nx.Ptree.Path.equal p q && r = s
    | _ -> false
  in
  Testable.make ~pp:Nx.Ptree.pp_visit ~equal

(* [magnitude x y] is the sum of the magnitudes of the products [dot x y]
   adds. *)
let magnitude x y =
  let abs = Nx.Ptree.map tagged (fun _ t -> Nx.abs t) in
  Nx.item [] (dot (abs x) (abs y))

let rounding scale = 1e3 *. epsilon_float *. (1. +. scale)

let pairing_tests =
  [
    prop "a gradient paired with a tangent is the derivative along it"
      (tagged_gen 2) (fun xs ->
        let x, t = match xs with [ x; t ] -> (x, t) | _ -> assert false in
        let g = Rune.grad tagged objective x in
        let _, d = Rune.jvp tagged Nx.Ptree.tensor objective x t in
        equal
          (float (rounding (magnitude g t)))
          (Nx.item [] d)
          (Nx.item [] (dot g t)));
    prop "a gradient keeps the parameters' visits, and steps them"
      (tagged_gen 1) (fun xs ->
        let x = List.hd xs in
        let visits = Nx.Ptree.visits tagged in
        let g = Rune.grad tagged objective x in
        equal ~msg:"gradient" (list visit) (visits x) (visits g);
        let step = Nx.Ptree.axpy tagged (scalar (-0.5)) g x in
        equal ~msg:"step" (list visit) (visits x) (visits step));
    prop "a pullback is the adjoint of the tangent map under both dots"
      (tagged_gen 3) (fun xs ->
        let x, v, u =
          match xs with [ x; v; u ] -> (x, v, u) | _ -> assert false
        in
        let y, jv = Rune.jvp tagged tagged image x v in
        let _, pb = Rune.vjp tagged tagged image x in
        let u = { u with tag = y.tag } in
        let jtu = pb u in
        equal ~msg:"pullback" (list visit) (Nx.Ptree.visits tagged x)
          (Nx.Ptree.visits tagged jtu);
        equal
          (float (rounding (magnitude u jv +. magnitude jtu v)))
          (Nx.item [] (dot u jv))
          (Nx.item [] (dot jtu v)));
  ]

let () =
  exit
    (run "Rune structures"
       [
         group "mismatches" mismatch_tests;
         group "signatures" signature_tests;
         group "preconditions" precondition_tests;
         group "arithmetic" arithmetic_tests;
         group "pairing" pairing_tests;
       ])
