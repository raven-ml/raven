(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Scalar forms and operators: each is its function with the constant of the
   operand's dtype, at every dtype, broadcast to the operand's shape and placed
   where it lies; an int outside the dtype raises naming the scalar form. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])

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

(* [f ()]'s elements, or the refusal's message without the function's name. *)
type 'v outcome = Elements of 'v list | Refused of string

let outcome dt f =
  let o =
    match f () with
    | x -> Elements (Nx.to_array x |> Array.to_list)
    | exception Invalid_argument e -> (
        match String.index_opt e ':' with
        | Some i -> Refused (String.sub e (i + 1) (String.length e - i - 1))
        | None -> Refused e)
  in
  (dt, o)

let same_outcome (type v s) ((dt : (v, s) D.t), a) (_, b) =
  match (a, b) with
  | Elements a, Elements b -> equal (elements dt) a b
  | Refused a, Refused b -> equal string a b
  | Elements _, Refused e -> failf "the function refused: %s" e
  | Refused e, Elements _ -> failf "the scalar form refused: %s" e

(* Values drawn from bytes *)

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

let shape =
  Gen.array ~size:(Gen.int_range 0 3)
    (Gen.frequency
       [ (1, Gen.constant 0); (1, Gen.constant 1); (4, Gen.int_range 2 4) ])

type operand = Operand : ('v, 's) D.t * ('v, 's, Nx.host) Nx.t * 'v -> operand

let operand =
  Gen.with_pp
    (fun ppf (Operand (dt, x, c)) ->
      Format.fprintf ppf "%a %a, %a" D.pp dt pp_ints (Nx.shape x)
        (D.pp_value dt) c)
    (let open Gen in
     let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all in
     let* s = shape in
     let* x = drawn dt s in
     let+ c = drawn dt [||] in
     Operand (dt, x, Nx.item [] c))

(* Forms *)

type form = {
  name : string;
  scalar : 'v 's. ('v, 's, Nx.host) Nx.t -> 'v -> ('v, 's, Nx.host) Nx.t;
  full : 'v 's. ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t;
}

type test_form = {
  tname : string;
  tscalar : 'v 's. ('v, 's, Nx.host) Nx.t -> 'v -> Nx.host Nx.bool_t;
  tfull : 'v 's. ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t -> Nx.host Nx.bool_t;
}

let c_of x c = Nx.scalar (Nx.dtype x) c

let forms =
  [
    { name = "add_s"; scalar = (fun x c -> Nx.add_s x c); full = (fun a b -> Nx.add a b) };
    { name = "sub_s"; scalar = (fun x c -> Nx.sub_s x c); full = (fun a b -> Nx.sub a b) };
    { name = "mul_s"; scalar = (fun x c -> Nx.mul_s x c); full = (fun a b -> Nx.mul a b) };
    { name = "div_s"; scalar = (fun x c -> Nx.div_s x c); full = (fun a b -> Nx.div a b) };
    { name = "pow_s"; scalar = (fun x c -> Nx.pow_s x c); full = (fun a b -> Nx.pow a b) };
    { name = "mod_s"; scalar = (fun x c -> Nx.mod_s x c); full = (fun a b -> Nx.mod_ a b) };
    { name = "maximum_s"; scalar = (fun x c -> Nx.maximum_s x c); full = (fun a b -> Nx.maximum a b) };
    { name = "minimum_s"; scalar = (fun x c -> Nx.minimum_s x c); full = (fun a b -> Nx.minimum a b) };
    { name = "rsub_s"; scalar = (fun x c -> Nx.rsub_s c x); full = (fun a b -> Nx.sub b a) };
    { name = "rdiv_s"; scalar = (fun x c -> Nx.rdiv_s c x); full = (fun a b -> Nx.div b a) };
    { name = "rpow_s"; scalar = (fun x c -> Nx.rpow_s c x); full = (fun a b -> Nx.pow b a) };
    { name = "( +$ )"; scalar = (fun x c -> Nx.(x +$ c)); full = (fun a b -> Nx.add a b) };
    { name = "( -$ )"; scalar = (fun x c -> Nx.(x -$ c)); full = (fun a b -> Nx.sub a b) };
    { name = "( *$ )"; scalar = (fun x c -> Nx.(x *$ c)); full = (fun a b -> Nx.mul a b) };
    { name = "( /$ )"; scalar = (fun x c -> Nx.(x /$ c)); full = (fun a b -> Nx.div a b) };
    { name = "( + )"; scalar = (fun x c -> Nx.(x + c_of x c)); full = (fun a b -> Nx.add a b) };
    { name = "( - )"; scalar = (fun x c -> Nx.(x - c_of x c)); full = (fun a b -> Nx.sub a b) };
    { name = "( * )"; scalar = (fun x c -> Nx.(x * c_of x c)); full = (fun a b -> Nx.mul a b) };
    { name = "( / )"; scalar = (fun x c -> Nx.(x / c_of x c)); full = (fun a b -> Nx.div a b) };
    { name = "( ** )"; scalar = (fun x c -> Nx.(x ** c_of x c)); full = (fun a b -> Nx.pow a b) };
  ]

let tests =
  [
    { tname = "equal_s"; tscalar = (fun x c -> Nx.equal_s x c); tfull = (fun a b -> Nx.equal a b) };
    { tname = "not_equal_s"; tscalar = (fun x c -> Nx.not_equal_s x c); tfull = (fun a b -> Nx.not_equal a b) };
    { tname = "less_s"; tscalar = (fun x c -> Nx.less_s x c); tfull = (fun a b -> Nx.less a b) };
    { tname = "less_equal_s"; tscalar = (fun x c -> Nx.less_equal_s x c); tfull = (fun a b -> Nx.less_equal a b) };
    { tname = "greater_s"; tscalar = (fun x c -> Nx.greater_s x c); tfull = (fun a b -> Nx.greater a b) };
    { tname = "greater_equal_s"; tscalar = (fun x c -> Nx.greater_equal_s x c); tfull = (fun a b -> Nx.greater_equal a b) };
    { tname = "Infix ( = )"; tscalar = (fun x c -> Nx.Infix.(x = c_of x c)); tfull = (fun a b -> Nx.equal a b) };
    { tname = "Infix ( <> )"; tscalar = (fun x c -> Nx.Infix.(x <> c_of x c)); tfull = (fun a b -> Nx.not_equal a b) };
    { tname = "Infix ( < )"; tscalar = (fun x c -> Nx.Infix.(x < c_of x c)); tfull = (fun a b -> Nx.less a b) };
    { tname = "Infix ( <= )"; tscalar = (fun x c -> Nx.Infix.(x <= c_of x c)); tfull = (fun a b -> Nx.less_equal a b) };
    { tname = "Infix ( > )"; tscalar = (fun x c -> Nx.Infix.(x > c_of x c)); tfull = (fun a b -> Nx.greater a b) };
    { tname = "Infix ( >= )"; tscalar = (fun x c -> Nx.Infix.(x >= c_of x c)); tfull = (fun a b -> Nx.greater_equal a b) };
    { tname = "Infix ( =$ )"; tscalar = (fun x c -> Nx.Infix.(x =$ c)); tfull = (fun a b -> Nx.equal a b) };
    { tname = "Infix ( <>$ )"; tscalar = (fun x c -> Nx.Infix.(x <>$ c)); tfull = (fun a b -> Nx.not_equal a b) };
    { tname = "Infix ( <$ )"; tscalar = (fun x c -> Nx.Infix.(x <$ c)); tfull = (fun a b -> Nx.less a b) };
    { tname = "Infix ( <=$ )"; tscalar = (fun x c -> Nx.Infix.(x <=$ c)); tfull = (fun a b -> Nx.less_equal a b) };
    { tname = "Infix ( >$ )"; tscalar = (fun x c -> Nx.Infix.(x >$ c)); tfull = (fun a b -> Nx.greater a b) };
    { tname = "Infix ( >=$ )"; tscalar = (fun x c -> Nx.Infix.(x >=$ c)); tfull = (fun a b -> Nx.greater_equal a b) };
  ]

(* [x]'s constant [c] as a 0-d value, which broadcasts. *)
let law_form f (Operand (dt, x, c)) =
  cover "no element" (Nx.numel x = 0);
  same_outcome
    (outcome dt (fun () -> f.full x (Nx.scalar dt c)))
    (outcome dt (fun () -> f.scalar x c))

let law_test f (Operand (dt, x, c)) =
  same_outcome
    (outcome D.Bool (fun () -> f.tfull x (Nx.scalar dt c)))
    (outcome D.Bool (fun () -> f.tscalar x c))

let laws =
  group "each form is its function with a constant"
    (List.map (fun f -> prop f.name operand (law_form f)) forms
    @ List.map (fun f -> prop f.tname operand (law_test f)) tests)

(* Cases *)

let test_negation () =
  equal (array float_exact) [| -1.; 0.; -0. |]
    (Nx.to_array Nx.(-create float32 [| 3 |] [| 1.; -0.; 0. |]))

let test_logical () =
  let a = Nx.create Nx.bool [| 4 |] [| false; false; true; true |]
  and b = Nx.create Nx.bool [| 4 |] [| false; true; false; true |] in
  equal (array bool) [| false; false; false; true |] (Nx.to_array Nx.Infix.(a && b));
  equal (array bool) [| false; true; true; true |] (Nx.to_array Nx.Infix.(a || b))

let test_reads_as_prose () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  equal (array float_exact) [| 3.; 5.; 7. |] (Nx.to_array Nx.(x *$ 2. +$ 1.));
  equal (array float_exact) [| 1.; 0.; -1. |] (Nx.to_array (Nx.rsub_s 2. x));
  equal (array bool) [| false; true; false |]
    (Nx.to_array Nx.Infix.((x >$ 1.) && (x <$ 3.)))

let test_placement () =
  let x = Nx.place (S2.split ~axis:0) (Nx.zeros Nx.float32 [| 4 |]) in
  let y = Nx.add_s x 1. in
  equal bool true
    (Nx.Placement.equal (S2.split ~axis:0) (Option.get (Nx.placement y)));
  equal (array float_exact) (Array.make 4 1.) (Nx.to_array y);
  equal bool true (Nx.placement (Nx.add_s (Nx.zeros Nx.float32 [| 2 |]) 1.) = None)

let refusals =
  [
    ("Nx.add_s", fun () -> ignore (Nx.add_s (Nx.zeros Nx.uint8 [| 2 |]) 256));
    ("Nx.rsub_s", fun () -> ignore (Nx.rsub_s (-1) (Nx.zeros Nx.uint4 [| 2 |])));
    ("Nx.less_s", fun () -> ignore (Nx.less_s (Nx.zeros Nx.int4 [| 2 |]) 8));
    ("Nx.div_s", fun () -> ignore (Nx.div_s (Nx.zeros Nx.bool [| 2 |]) true));
    ("Nx.mod_s", fun () -> ignore (Nx.mod_s (Nx.zeros Nx.complex64 [| 1 |]) Complex.one));
  ]

let cases_group =
  group "cases"
    [
      test "~- negates, -0. included" test_negation;
      test "Infix's && and || are logical_and and logical_or" test_logical;
      test "a formula reads as its arithmetic" test_reads_as_prose;
      test "a scalar form computes where its operand lies" test_placement;
      cases "refusals name the scalar form" ~name:fst refusals (fun (by, f) ->
          invalid ~by f);
    ]

let () = exit (run "nx operators" [ laws; cases_group ])
