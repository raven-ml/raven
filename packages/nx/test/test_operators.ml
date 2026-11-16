(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Scalar forms and operators: each is its function with the constant of the
   operand's dtype, at every dtype, broadcast to the operand's shape and placed
   where it lies; an int outside the dtype raises naming the scalar form. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

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
    (A.v dt (L.contiguous (Iarray.of_array s))
       (Rig.Buffer.of_string (if data = "" then "\000" else data)))

let shape =
  Gen.array ~size:(Gen.int_range 0 3)
    (Gen.frequency
       [ (1, Gen.constant 0); (1, Gen.constant 1); (4, Gen.int_range 2 4) ])

(* An operand as a function reads it: whole, reversed along every axis (a
   view of negative steps), or broadcast along a new leading axis (a view
   of step zero). *)
type view = Whole | Flipped | Broadcast

let view_name = function
  | Whole -> "whole"
  | Flipped -> "flipped"
  | Broadcast -> "broadcast"

type operand =
  | Operand : ('v, 's) D.t * ('v, 's, Nx.host) Nx.t * 'v * view -> operand

let viewed v x =
  match v with
  | Whole -> x
  | Flipped -> Nx.flip x
  | Broadcast -> Nx.broadcast_to (Array.append [| 2 |] (Nx.shape x)) x

let operand =
  Gen.with_pp
    (fun ppf (Operand (dt, x, c, v)) ->
      Format.fprintf ppf "%a %a %s, %a" D.pp dt pp_ints (Nx.shape x)
        (view_name v) (D.pp_value dt) c)
    (let open Gen in
     let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all in
     let* s = shape in
     let* v = of_list [ Whole; Flipped; Broadcast ] in
     let* x = drawn dt s in
     let+ c = drawn dt [||] in
     Operand (dt, viewed v x, Nx.item [] c, v))

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
    { name = "Infix ( +$ )"; scalar = (fun x c -> Nx.Infix.(x +$ c)); full = (fun a b -> Nx.add a b) };
    { name = "Infix ( -$ )"; scalar = (fun x c -> Nx.Infix.(x -$ c)); full = (fun a b -> Nx.sub a b) };
    { name = "Infix ( *$ )"; scalar = (fun x c -> Nx.Infix.(x *$ c)); full = (fun a b -> Nx.mul a b) };
    { name = "Infix ( /$ )"; scalar = (fun x c -> Nx.Infix.(x /$ c)); full = (fun a b -> Nx.div a b) };
    { name = "Infix ( + )"; scalar = (fun x c -> Nx.Infix.(x + c_of x c)); full = (fun a b -> Nx.add a b) };
    { name = "Infix ( - )"; scalar = (fun x c -> Nx.Infix.(x - c_of x c)); full = (fun a b -> Nx.sub a b) };
    { name = "Infix ( * )"; scalar = (fun x c -> Nx.Infix.(x * c_of x c)); full = (fun a b -> Nx.mul a b) };
    { name = "Infix ( / )"; scalar = (fun x c -> Nx.Infix.(x / c_of x c)); full = (fun a b -> Nx.div a b) };
    { name = "Infix ( ** )"; scalar = (fun x c -> Nx.Infix.(x ** c_of x c)); full = (fun a b -> Nx.pow a b) };
    { name = "Infix ( **$ )"; scalar = (fun x c -> Nx.Infix.(x **$ c)); full = (fun a b -> Nx.pow a b) };
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

(* [x]'s constant [c] as a 0-d value, which broadcasts, over [x] whole or a
   view of it. *)
let law_form f (Operand (dt, x, c, v)) =
  cover "no element" (Nx.numel x = 0);
  cover "a view of negative steps" (v = Flipped && Nx.numel x > 1);
  cover "a view of step zero" (v = Broadcast && Nx.numel x > 0);
  same_outcome
    (outcome dt (fun () -> f.full x (Nx.scalar dt c)))
    (outcome dt (fun () -> f.scalar x c))

let law_test f (Operand (dt, x, c, _)) =
  same_outcome
    (outcome D.Bool (fun () -> f.tfull x (Nx.scalar dt c)))
    (outcome D.Bool (fun () -> f.tscalar x c))

let laws =
  group "each form is its function with a constant"
    (List.map (fun f -> prop f.name operand (law_form f)) forms
    @ List.map (fun f -> prop f.tname operand (law_test f)) tests)

(* Literals *)

(* A host value of [dt] over [data], each byte of a bool its low bit. *)
let of_bytes (type v s) (dt : (v, s) D.t) data : (v, s, Nx.host) Nx.t =
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  let n = String.length data * 8 / D.bits dt in
  Nx.Repr.of_array Nx.Host.v
    (A.v dt (L.contiguous [| n |]) (Rig.Buffer.of_string data))

(* Values of [dt] that a literal breaks on. The bit patterns zero, one, all
   ones and the signed range's ends: for an integer its range's ends, for int4
   and uint4 each nibble's corners, for a float NaNs and the least subnormal.
   For a float or complex dtype, from float64 also ±0, NaNs of both signs,
   quiet and signalling, with payloads, ±∞ and subnormals of each width. *)
let specials (type v s) (dt : (v, s) D.t) : (v, s, Nx.host) Nx.t =
  let w = max 1 (D.bits dt / 8) in
  let pattern f = String.init w f in
  let bits =
    String.concat ""
      [
        pattern (fun _ -> '\000');
        pattern (fun i -> if i = 0 then '\001' else '\000');
        pattern (fun _ -> '\255');
        pattern (fun i -> if i = w - 1 then '\127' else '\255');
        pattern (fun i -> if i = w - 1 then '\128' else '\000');
      ]
  in
  let bits = of_bytes dt bits in
  let from64 xs = Nx.cast dt (Nx.create Nx.float64 [| Array.length xs |] xs) in
  match D.kind dt with
  | Float | Complex ->
      Nx.concatenate ~axis:0
        [
          bits;
          from64
            [|
              0.; -0.; 1.; -2.25; 0.9; 1e-3; Float.nan;
              Int64.float_of_bits 0xFFF8000000000123L;
              Int64.float_of_bits 0x7FF4000000000001L; Float.infinity;
              Float.neg_infinity; 1e-310; 1e-40; 6e-8; 3e38; 65504.;
            |];
        ]
  | Signed | Unsigned ->
      Nx.concatenate ~axis:0 [ bits; from64 [| 2.; 3.; 7. |] ]
  | Boolean -> bits

(* Each scalar form at each dtype gives its function's bits, NaNs as one,
   for each literal fresh, after itself and after the others of its class,
   its call after another form's, as a program writes them. *)
let test_literal_orders () =
  (* The forms that take a literal: not the operators of two values. *)
  let literal name =
    (not (String.contains name ' ')) || String.contains name '$'
  in
  let scalar = List.filter (fun f -> literal f.name) forms
  and tscalar = List.filter (fun f -> literal f.tname) tests in
  List.iter
    (fun (D.Any dt) ->
      let x = specials dt in
      let lits = List.init (Nx.numel x) (fun i -> Nx.item [ i ] x) in
      let one f c =
        same_outcome
          (outcome dt (fun () -> f.full x (Nx.scalar dt c)))
          (outcome dt (fun () -> f.scalar x c))
      and tone f c =
        same_outcome
          (outcome D.Bool (fun () -> f.tfull x (Nx.scalar dt c)))
          (outcome D.Bool (fun () -> f.tscalar x c))
      in
      let each g = List.iter g scalar and teach g = List.iter g tscalar in
      List.iter
        (fun c ->
          each (fun f -> one f c);
          teach (fun f -> tone f c))
        lits;
      each (fun f -> List.iter (fun c -> one f c; one f c) lits);
      teach (fun f -> List.iter (fun c -> tone f c; tone f c) lits);
      each (fun f -> List.iter (one f) (List.rev lits));
      teach (fun f -> List.iter (tone f) (List.rev lits)))
    D.all

(* Two domains calling one scalar form of one kind, dtype and side, each with
   its own literal, a million times each: every result has its literal's bits.
   A scalar form shares nothing between calls that one domain could read
   half-written from another; at this count a torn read shows. *)
let test_two_domains () =
  let x = Nx.create Nx.float32 [| 4 |] [| 1.5; -2.25; 3.; 0.7 |] in
  let run c () =
    let want = Nx.to_array (Nx.mul x (Nx.scalar Nx.float32 c)) in
    let wrong = ref 0 in
    for _ = 1 to 1_000_000 do
      if Nx.to_array (Nx.mul_s x c) <> want then incr wrong
    done;
    !wrong
  in
  let other = Domain.spawn (run 0.001) in
  let here = run 0.9 () in
  equal (pair int int) (0, 0) (here, Domain.join other)

(* Binary operators of two values, each its function: [b] is [a]'s
   elements reversed, broadcast where [a] has a unit last axis. *)
type pair_form = {
  pname : string;
  op : 'v 's. ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t;
  fn : 'v 's. ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t;
}

let pair_forms =
  [
    { pname = "Infix ( land )"; op = (fun a b -> Nx.Infix.(a land b)); fn = Nx.bitwise_and };
    { pname = "Infix ( lor )"; op = (fun a b -> Nx.Infix.(a lor b)); fn = Nx.bitwise_or };
    { pname = "Infix ( lxor )"; op = (fun a b -> Nx.Infix.(a lxor b)); fn = Nx.bitwise_xor };
    { pname = "Infix ( *@ )"; op = (fun a b -> Nx.Infix.(a *@ b)); fn = Nx.matmul };
  ]

let law_pair f (Operand (dt, x, _, _)) =
  let y = Nx.flip x in
  same_outcome (outcome dt (fun () -> f.fn x y)) (outcome dt (fun () -> f.op x y))

let pair_laws =
  group "each binary operator is its function"
    (List.map (fun f -> prop f.pname operand (law_pair f)) pair_forms)

(* OCaml's operators under [open Nx]: each field of [under_nx] compiles only
   while [Nx]'s top level defines none of OCaml's operators. [size] counts a
   shape's elements with [( * )] beside [Nx]'s functions, as a program under
   the open does. *)

type stdlib_ops = {
  int2 : (int -> int -> int) list;
  shifts : (int -> int -> int) list;
  int1 : (int -> int) list;
  float2 : (float -> float -> float) list;
  float1 : (float -> float) list;
  compare : (int -> int -> bool) list;
  logic : (bool -> bool -> bool) list;
  append : int list -> int list -> int list;
  concat : string -> string -> string;
  pipe : int -> (int -> int) -> int;
  apply : (int -> int) -> int -> int;
  deref : int ref -> int;
  assign : int ref -> int -> unit;
  size : int array -> int;
}

let under_nx =
  let open Nx in
  {
    int2 = [ ( + ); ( - ); ( * ); ( / ); ( mod ); ( land ); ( lor ); ( lxor ) ];
    shifts = [ ( lsl ); ( lsr ); ( asr ) ];
    int1 = [ ( ~- ); ( ~+ ); lnot ];
    float2 = [ ( +. ); ( -. ); ( *. ); ( /. ); ( ** ) ];
    float1 = [ ( ~-. ); ( ~+. ) ];
    compare = [ ( = ); ( <> ); ( < ); ( <= ); ( > ); ( >= ); ( == ); ( != ) ];
    logic = [ ( && ); ( || ) ];
    append = ( @ );
    concat = ( ^ );
    pipe = ( |> );
    apply = ( @@ );
    deref = ( ! );
    assign = ( := );
    size = (fun s -> Array.fold_left ( * ) 1 (shape (zeros float32 s)));
  }

let test_stdlib () =
  let o = under_nx in
  let r = ref 0 in
  o.assign r 3;
  equal (list int) [ 9; 5; 14; 3; 1; 2; 7; 5 ]
    (List.map (fun f -> f 7 2) o.int2);
  (* [-7] is [2^63 - 7] as 63 bits, so [lsr] fills with zeros. *)
  equal (list int) [ -28; 2305843009213693950; -2 ]
    (List.map (fun f -> f (-7) 2) o.shifts);
  equal (list int) [ -7; 7; -8 ] (List.map (fun f -> f 7) o.int1);
  equal (list float_exact) [ 9.; 5.; 14.; 3.5; 49. ]
    (List.map (fun f -> f 7. 2.) o.float2);
  equal (list float_exact) [ -7.; 7. ] (List.map (fun f -> f 7.) o.float1);
  equal (list bool) [ false; true; false; false; true; true; false; true ]
    (List.map (fun f -> f 7 2) o.compare);
  equal (list bool) [ false; true ] (List.map (fun f -> f true false) o.logic);
  equal (list int) [ 7; 2 ] (o.append [ 7 ] [ 2 ]);
  equal string "72" (o.concat "7" "2");
  equal int 8 (o.pipe 7 succ);
  equal int 8 (o.apply succ 7);
  equal int 3 (o.deref r);
  equal int 6 (o.size [| 2; 3 |])

(* Cases *)

let test_negation () =
  equal (array float_exact) [| -1.; 0.; -0. |]
    (Nx.to_array Nx.Infix.(-Nx.create Nx.float32 [| 3 |] [| 1.; -0.; 0. |]))

let test_logical () =
  let a = Nx.create Nx.bool [| 4 |] [| false; false; true; true |]
  and b = Nx.create Nx.bool [| 4 |] [| false; true; false; true |] in
  equal (array bool) [| false; false; false; true |] (Nx.to_array Nx.Infix.(a && b));
  equal (array bool) [| false; true; true; true |] (Nx.to_array Nx.Infix.(a || b))

let test_reads_as_prose () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  equal (array float_exact) [| 3.; 5.; 7. |] (Nx.to_array Nx.Infix.(x *$ 2. +$ 1.));
  equal (array float_exact) [| 1.; 0.; -1. |] (Nx.to_array (Nx.rsub_s 2. x));
  equal (array bool) [| false; true; false |]
    (Nx.to_array Nx.Infix.((x >$ 1.) && (x <$ 3.)))

let test_placement () =
  let x = Nx.place (S2.split ~axis:0) (Nx.zeros Nx.float32 [| 4 |]) in
  let y = Nx.add_s x 1. in
  equal bool true
    (Nx.Placement.equal (S2.split ~axis:0) (Nx.placement y));
  equal (array float_exact) (Array.make 4 1.) (Nx.to_array y);
  equal bool true
    (Nx.Placement.equal Nx.Host.on
       (Nx.placement (Nx.add_s (Nx.zeros Nx.float32 [| 2 |]) 1.)))

let test_index_operators () =
  let x = Nx.reshape [| 2; 3 |] (Nx.arange Nx.int32 0 6 1) in
  equal (array int32) [| 5l |] (Nx.to_array Nx.Infix.(x.%{[ 1; 2 ]}));
  equal (array int32) [| 3l; 4l; 5l |] (Nx.to_array Nx.Infix.(x.%{[ 1 ]}));
  equal (array int32) [| 1l; 4l |] (Nx.to_array Nx.Infix.(x.${Nx.[ A; I 1 ]}));
  invalid ~by:"Nx.get" (fun () -> Nx.Infix.(x.%{[ 2 ]}));
  invalid ~by:"Nx.slice" (fun () -> Nx.Infix.(x.${Nx.[ A; I 3 ]}))

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
      test "a scalar form gives its function's bits whatever came before"
        test_literal_orders;
      test "two domains' scalar forms of one kind give their own literals'"
        test_two_domains;
      test ".%{} is get and .${} is slice" test_index_operators;
      test "open Nx keeps OCaml's operators" test_stdlib;
      cases "refusals name the scalar form" ~id:fst refusals (fun (by, f) ->
          invalid ~by f);
    ]

let () = exit (run "nx operators" [ laws; pair_laws; cases_group ])
