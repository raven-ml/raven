(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Scalar forms and operators, in nx.mli's order: each is its function with the
   constant of the operand's dtype, at every dtype, broadcast to the operand's
   shape and placed where it lies; an int outside the dtype raises naming the
   scalar form. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

type host = Nx.host

(* [f ()]'s elements, or the refusal's message without the function's name. *)
type 'v outcome = Elements of 'v list | Refused of string

let outcome f =
  match f () with
  | x -> Elements (Nx.to_array x |> Array.to_list)
  | exception Invalid_argument e -> (
      match String.index_opt e ':' with
      | Some i -> Refused (String.sub e (i + 1) (String.length e - i - 1))
      | None -> Refused e)

let same_outcome (type v s) (dt : (v, s) D.t) a b =
  match (a, b) with
  | Elements a, Elements b -> equal (elements dt) a b
  | Refused a, Refused b -> equal string a b
  | Elements _, Refused e -> failf "the operator refused: %s" e
  | Refused e, Elements _ -> failf "the function refused: %s" e

(* Values drawn from bytes *)

let drawn (type v s) (dt : (v, s) D.t) s : (v, s, host) Nx.t Gen.t =
  let open Gen in
  let n = Array.fold_left ( * ) 1 s in
  let+ data = string_of ~size:(constant (D.bytes dt n)) char in
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  Nx.Repr.of_array Nx.Host.v
    (A.v dt
       (L.contiguous (Iarray.of_array s))
       (Rig.Buffer.of_string (if data = "" then "\000" else data)))

let shape =
  Gen.array ~size:(Gen.int_range 0 3)
    (Gen.frequency
       [ (1, Gen.constant 0); (1, Gen.constant 1); (4, Gen.int_range 2 4) ])

(* An operand as a function reads it: whole, reversed along every axis (a view
   of negative steps), or broadcast along a new leading axis (a view of step
   zero). *)
type view = Whole | Flipped | Broadcast

let view_name = function
  | Whole -> "whole"
  | Flipped -> "flipped"
  | Broadcast -> "broadcast"

type operand =
  | Operand : ('v, 's) D.t * ('v, 's, host) Nx.t * 'v * view -> operand

let viewed v x =
  match v with
  | Whole -> x
  | Flipped -> Nx.flip x
  | Broadcast -> Nx.broadcast_to (Array.append [| 2 |] (Nx.shape x)) x

(* A constant of [dt]: one of its elements from bytes, or, for a float dtype,
   any float, which the form stores as [Dtype.of_float] does. *)
let literal (type v s) (dt : (v, s) D.t) : v Gen.t =
  let element = Gen.map (Nx.item []) (drawn dt [||]) in
  match D.kind dt with
  | D.Float ->
      Gen.frequency
        [ (2, element); (1, Gen.of_list [ 0.1; 1e-7; 1e10; 0.3; -2.7 ]) ]
  | _ -> element

let operand =
  Gen.with_pp
    (fun ppf (Operand (dt, x, c, v)) ->
      Format.fprintf ppf "%a %a %s, %a" D.pp dt pp_ints (Nx.shape x)
        (view_name v) (D.pp_value dt)
        (Nx.item [] (Nx.scalar dt c)))
    (let open Gen in
     let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all in
     let* s = shape in
     let* v = of_list [ Whole; Flipped; Broadcast ] in
     let* x = drawn dt s in
     let+ c = literal dt in
     Operand (dt, viewed v x, c, v))

(* A float literal no narrow dtype holds, at each float dtype. *)
let unheld =
  List.filter_map
    (fun (D.Any dt) ->
      match D.kind dt with
      | D.Float -> Some (Operand (dt, Nx.ones dt [| 3 |], 0.1, Whole))
      | _ -> None)
    D.all

let covers (Operand (_, x, _, v)) =
  cover "no element" (Nx.numel x = 0);
  cover "a view of negative steps" (v = Flipped && Nx.numel x > 1);
  cover "a view of step zero" (v = Broadcast && Nx.numel x > 0)

(* [scalar x c] is [full x c'] for [c'] the 0-d value of [c], which broadcasts,
   over [x] whole or a view of it. *)
let law_form (scalar : 'v 's. ('v, 's, host) Nx.t -> 'v -> ('v, 's, host) Nx.t)
    (full :
      'v 's. ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t)
    (Operand (dt, x, c, _) as o) =
  covers o;
  same_outcome dt
    (outcome (fun () -> full x (Nx.scalar dt c)))
    (outcome (fun () -> scalar x c))

let law_test (scalar : 'v 's. ('v, 's, host) Nx.t -> 'v -> host Nx.bool_t)
    (full : 'v 's. ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t -> host Nx.bool_t)
    (Operand (dt, x, c, _) as o) =
  covers o;
  same_outcome D.Bool
    (outcome (fun () -> full x (Nx.scalar dt c)))
    (outcome (fun () -> scalar x c))

let form name (scalar : 'v 's. ('v, 's, host) Nx.t -> 'v -> ('v, 's, host) Nx.t)
    (full :
      'v 's. ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t)
    =
  prop ~examples:unheld name operand (law_form scalar full)

let test_form name (scalar : 'v 's. ('v, 's, host) Nx.t -> 'v -> host Nx.bool_t)
    (full : 'v 's. ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t -> host Nx.bool_t)
    =
  prop ~examples:unheld name operand (law_test scalar full)

let c_of x c = Nx.scalar (Nx.dtype x) c

(* Operators of two values, and of one, each its function: [b] is [a]'s elements
   reversed. *)
let law_pair
    (op :
      'v 's. ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t)
    (fn :
      'v 's. ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t -> ('v, 's, host) Nx.t)
    (Operand (dt, x, _, _)) =
  let y = Nx.flip x in
  same_outcome dt (outcome (fun () -> fn x y)) (outcome (fun () -> op x y))

let law_negation (Operand (dt, x, _, _)) =
  same_outcome dt
    (outcome (fun () -> Nx.neg x))
    (outcome (fun () -> Nx.Infix.(-x)))

(* Two boolean values, the second of the first's shape. *)
let booleans =
  Gen.with_pp
    (fun ppf (a, _) -> Format.fprintf ppf "bool %a" pp_ints (Nx.shape a))
    (let open Gen in
     let* s = shape in
     let+ a = drawn D.Bool s and+ b = drawn D.Bool s in
     (a, b))

let law_logical (op : host Nx.bool_t -> host Nx.bool_t -> host Nx.bool_t)
    (fn : host Nx.bool_t -> host Nx.bool_t -> host Nx.bool_t) (a, b) =
  equal (array bool) (Nx.to_array (fn a b)) (Nx.to_array (op a b))

(* Literals *)

(* A host value of [dt] over [data], each byte of a bool its low bit. *)
let of_bytes (type v s) (dt : (v, s) D.t) data : (v, s, host) Nx.t =
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  let n = String.length data * 8 / D.bits dt in
  Nx.Repr.of_array Nx.Host.v
    (A.v dt (L.contiguous [| n |]) (Rig.Buffer.of_string data))

(* Values of [dt] that a literal breaks on. The bit patterns zero, one, all ones
   and the signed range's ends: for an integer its range's ends, for int4 and
   uint4 each nibble's corners, for a float NaNs and the least subnormal. For a
   float or complex dtype, from float64 also ±0, NaNs of both signs, quiet and
   signalling, with payloads, ±∞ and subnormals of each width. *)
let specials (type v s) (dt : (v, s) D.t) : (v, s, host) Nx.t =
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
              0.;
              -0.;
              1.;
              -2.25;
              0.9;
              1e-3;
              Float.nan;
              Int64.float_of_bits 0xFFF8000000000123L;
              Int64.float_of_bits 0x7FF4000000000001L;
              Float.infinity;
              Float.neg_infinity;
              1e-310;
              1e-40;
              6e-8;
              3e38;
              65504.;
            |];
        ]
  | Signed | Unsigned ->
      Nx.concatenate ~axis:0 [ bits; from64 [| 2.; 3.; 7. |] ]
  | Boolean -> bits

(* Each scalar form at each dtype gives its function's bits, NaNs as one,
   for each literal fresh, after itself and after the others of its class,
   its call after another form's, as a program writes them. *)
(* Each form of [x] and [c] and its function's, computed when called. *)
let forms (type v s) (x : (v, s, host) Nx.t) (c : v) =
  [
    (fun () -> (Nx.add_s x c, Nx.add x (c_of x c)));
    (fun () -> (Nx.sub_s x c, Nx.sub x (c_of x c)));
    (fun () -> (Nx.mul_s x c, Nx.mul x (c_of x c)));
    (fun () -> (Nx.div_s x c, Nx.div x (c_of x c)));
    (fun () -> (Nx.pow_s x c, Nx.pow x (c_of x c)));
    (fun () -> (Nx.mod_s x c, Nx.mod_ x (c_of x c)));
    (fun () -> (Nx.maximum_s x c, Nx.maximum x (c_of x c)));
    (fun () -> (Nx.minimum_s x c, Nx.minimum x (c_of x c)));
    (fun () -> (Nx.rsub_s c x, Nx.sub (c_of x c) x));
    (fun () -> (Nx.rdiv_s c x, Nx.div (c_of x c) x));
    (fun () -> (Nx.rpow_s c x, Nx.pow (c_of x c) x));
  ]

let tests (type v s) (x : (v, s, host) Nx.t) (c : v) =
  [
    (fun () -> (Nx.equal_s x c, Nx.equal x (c_of x c)));
    (fun () -> (Nx.not_equal_s x c, Nx.not_equal x (c_of x c)));
    (fun () -> (Nx.less_s x c, Nx.less x (c_of x c)));
    (fun () -> (Nx.less_equal_s x c, Nx.less_equal x (c_of x c)));
    (fun () -> (Nx.greater_s x c, Nx.greater x (c_of x c)));
    (fun () -> (Nx.greater_equal_s x c, Nx.greater_equal x (c_of x c)));
  ]

let check_orders (type v s) (dt : (v, s) D.t) =
  let x = specials dt in
  let lits = List.init (Nx.numel x) (fun i -> Nx.item [ i ] x) in
  let one k c =
    match (List.nth (forms x c) k) () with
    | exception Invalid_argument _ -> ()
    | got, want ->
        equal (elements dt)
          (Array.to_list (Nx.to_array want))
          (Array.to_list (Nx.to_array got))
  and tone k c =
    let got, want = (List.nth (tests x c) k) () in
    equal (array bool) (Nx.to_array want) (Nx.to_array got)
  in
  let order cs =
    List.iter (fun k -> List.iter (one k) cs) (List.init 11 Fun.id);
    List.iter (fun k -> List.iter (tone k) cs) (List.init 6 Fun.id)
  in
  order lits;
  order (List.concat_map (fun c -> [ c; c ]) lits);
  order (List.rev lits)

(* Each scalar form at each dtype gives its function's bits, NaNs as one, for
   each literal fresh, after itself and after the others of its class, its call
   after another form's, as a program writes them. *)
let test_literal_orders () = List.iter (fun (D.Any dt) -> check_orders dt) D.all

(* Domains *)

(* Scalar forms of one kind, dtype and side, each with its own literal, on two
   domains: each result is its function's of that literal. *)
let literal_of =
  Gen.with_pp
    (fun ppf c -> Format.fprintf ppf "%h" c)
    (Gen.of_list [ 0.001; 0.9; -2.5; 1e10; 0.1; 3.; -0.; 7.25 ])

let row () = Nx.create Nx.float32 [| 4 |] [| 1.5; -2.25; 3.; 0.7 |]

let domain_commands =
  let floats = array float_exact in
  [
    command "mul_s"
      (literal_of @-> returns floats)
      (fun c -> Nx.to_array (Nx.mul (row ()) (Nx.scalar Nx.float32 c)))
      (fun c -> Nx.to_array (Nx.mul_s (row ()) c));
    command "add_s"
      (literal_of @-> returns floats)
      (fun c -> Nx.to_array (Nx.add (row ()) (Nx.scalar Nx.float32 c)))
      (fun c -> Nx.to_array (Nx.add_s (row ()) c));
  ]

(* Cases *)

let test_placement () =
  let x = Nx.place (S2.split ~axis:0) (Nx.zeros Nx.float32 [| 4 |]) in
  let y = Nx.add_s x 1. in
  equal (placement ()) (S2.split ~axis:0) (Nx.placement y);
  equal (array float_exact) (Array.make 4 1.) (Nx.to_array y);
  equal (placement ()) Nx.Host.on
    (Nx.placement (Nx.add_s (Nx.zeros Nx.float32 [| 2 |]) 1.))

let test_index_operators () =
  let x = Nx.reshape [| 2; 3 |] (Nx.arange Nx.int32 0 6 1) in
  equal (array int32) [| 5l |] (Nx.to_array Nx.Infix.(x.%{[ 1; 2 ]}));
  equal (array int32) [| 3l; 4l; 5l |] (Nx.to_array Nx.Infix.(x.%{[ 1 ]}));
  equal (array int32) [| 1l; 4l |] (Nx.to_array Nx.Infix.(x.${Nx.[ A; I 1 ]}));
  invalid ~by:"Nx.get" (fun () -> Nx.Infix.(x.%{[ 2 ]}));
  invalid ~by:"Nx.slice" (fun () -> Nx.Infix.(x.${Nx.[ A; I 3 ]}))

(* OCaml's operators under [open Nx]: each field of [under_nx] compiles only
   while [Nx]'s top level defines none of OCaml's operators. [size] counts a
   shape's elements with [( * )] beside [Nx]'s functions, as a program under the
   open does. *)

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
  equal (list int)
    [ 9; 5; 14; 3; 1; 2; 7; 5 ]
    (List.map (fun f -> f 7 2) o.int2);
  (* [-7] is [2^63 - 7] as 63 bits, so [lsr] fills with zeros. *)
  equal (list int)
    [ -28; 2305843009213693950; -2 ]
    (List.map (fun f -> f (-7) 2) o.shifts);
  equal (list int) [ -7; 7; -8 ] (List.map (fun f -> f 7) o.int1);
  equal (list float_exact) [ 9.; 5.; 14.; 3.5; 49. ]
    (List.map (fun f -> f 7. 2.) o.float2);
  equal (list float_exact) [ -7.; 7. ] (List.map (fun f -> f 7.) o.float1);
  equal (list bool)
    [ false; true; false; false; true; true; false; true ]
    (List.map (fun f -> f 7 2) o.compare);
  equal (list bool) [ false; true ] (List.map (fun f -> f true false) o.logic);
  equal (list int) [ 7; 2 ] (o.append [ 7 ] [ 2 ]);
  equal string "72" (o.concat "7" "2");
  equal int 8 (o.pipe 7 succ);
  equal int 8 (o.apply succ 7);
  equal int 3 (o.deref r);
  equal int 6 (o.size [| 2; 3 |])

let refusals =
  [
    ("Nx.add_s", fun () -> ignore (Nx.add_s (Nx.zeros Nx.uint8 [| 2 |]) 256));
    ("Nx.rsub_s", fun () -> ignore (Nx.rsub_s (-1) (Nx.zeros Nx.uint4 [| 2 |])));
    ("Nx.less_s", fun () -> ignore (Nx.less_s (Nx.zeros Nx.int4 [| 2 |]) 8));
    ("Nx.div_s", fun () -> ignore (Nx.div_s (Nx.zeros Nx.bool [| 2 |]) true));
    ( "Nx.mod_s",
      fun () -> ignore (Nx.mod_s (Nx.zeros Nx.complex64 [| 1 |]) Complex.one) );
  ]

let () =
  exit
    (run "nx operators"
       [
         group "scalar forms"
           [
             form "add_s x c is add x c" Nx.add_s Nx.add;
             form "sub_s x c is sub x c" Nx.sub_s Nx.sub;
             form "mul_s x c is mul x c" Nx.mul_s Nx.mul;
             form "div_s x c is div x c" Nx.div_s Nx.div;
             form "pow_s x c is pow x c" Nx.pow_s Nx.pow;
             form "mod_s x c is mod_ x c" Nx.mod_s Nx.mod_;
             form "maximum_s x c is maximum x c" Nx.maximum_s Nx.maximum;
             form "minimum_s x c is minimum x c" Nx.minimum_s Nx.minimum;
             form "rsub_s c x is sub c x"
               (fun x c -> Nx.rsub_s c x)
               (fun a b -> Nx.sub b a);
             form "rdiv_s c x is div c x"
               (fun x c -> Nx.rdiv_s c x)
               (fun a b -> Nx.div b a);
             form "rpow_s c x is pow c x"
               (fun x c -> Nx.rpow_s c x)
               (fun a b -> Nx.pow b a);
             test_form "equal_s x c is equal x c" Nx.equal_s Nx.equal;
             test_form "not_equal_s x c is not_equal x c" Nx.not_equal_s
               Nx.not_equal;
             test_form "less_s x c is less x c" Nx.less_s Nx.less;
             test_form "less_equal_s x c is less_equal x c" Nx.less_equal_s
               Nx.less_equal;
             test_form "greater_s x c is greater x c" Nx.greater_s Nx.greater;
             test_form "greater_equal_s x c is greater_equal x c"
               Nx.greater_equal_s Nx.greater_equal;
             cases "an int outside the dtype raises naming the form" ~id:fst
               refusals (fun (by, f) -> invalid ~by f);
             test "a scalar form computes where its operand lies" test_placement;
             test "a scalar form gives its function's bits whatever came before"
               test_literal_orders;
             (* Two commands, both drawable in every program: ten programs call
                each. *)
             stateful ~domains:2 ~count:10
               "scalar forms on two domains each give their own literal's"
               domain_commands;
           ];
         group "Infix"
           [
             form "a + b is add a b" (fun x c -> Nx.Infix.(x + c_of x c)) Nx.add;
             form "a - b is sub a b" (fun x c -> Nx.Infix.(x - c_of x c)) Nx.sub;
             form "a * b is mul a b" (fun x c -> Nx.Infix.(x * c_of x c)) Nx.mul;
             form "a / b is div a b" (fun x c -> Nx.Infix.(x / c_of x c)) Nx.div;
             form "a ** b is pow a b"
               (fun x c -> Nx.Infix.(x ** c_of x c))
               Nx.pow;
             prop "-x is neg x" operand law_negation;
             form "x +$ c is add_s x c" (fun x c -> Nx.Infix.(x +$ c)) Nx.add;
             form "x -$ c is sub_s x c" (fun x c -> Nx.Infix.(x -$ c)) Nx.sub;
             form "x *$ c is mul_s x c" (fun x c -> Nx.Infix.(x *$ c)) Nx.mul;
             form "x /$ c is div_s x c" (fun x c -> Nx.Infix.(x /$ c)) Nx.div;
             form "x **$ c is pow_s x c" (fun x c -> Nx.Infix.(x **$ c)) Nx.pow;
             prop "a land b is bitwise_and a b" operand
               (law_pair (fun a b -> Nx.Infix.(a land b)) Nx.bitwise_and);
             prop "a lor b is bitwise_or a b" operand
               (law_pair (fun a b -> Nx.Infix.(a lor b)) Nx.bitwise_or);
             prop "a lxor b is bitwise_xor a b" operand
               (law_pair (fun a b -> Nx.Infix.(a lxor b)) Nx.bitwise_xor);
             prop "a *@ b is matmul a b" operand
               (law_pair (fun a b -> Nx.Infix.(a *@ b)) Nx.matmul);
             test "x.%{p} is get p x and x.${i} is slice i x"
               test_index_operators;
             test_form "a = b is equal a b"
               (fun x c -> Nx.Infix.(x = c_of x c))
               Nx.equal;
             test_form "a <> b is not_equal a b"
               (fun x c -> Nx.Infix.(x <> c_of x c))
               Nx.not_equal;
             test_form "a < b is less a b"
               (fun x c -> Nx.Infix.(x < c_of x c))
               Nx.less;
             test_form "a <= b is less_equal a b"
               (fun x c -> Nx.Infix.(x <= c_of x c))
               Nx.less_equal;
             test_form "a > b is greater a b"
               (fun x c -> Nx.Infix.(x > c_of x c))
               Nx.greater;
             test_form "a >= b is greater_equal a b"
               (fun x c -> Nx.Infix.(x >= c_of x c))
               Nx.greater_equal;
             test_form "x =$ c is equal_s x c"
               (fun x c -> Nx.Infix.(x =$ c))
               Nx.equal;
             test_form "x <>$ c is not_equal_s x c"
               (fun x c -> Nx.Infix.(x <>$ c))
               Nx.not_equal;
             test_form "x <$ c is less_s x c"
               (fun x c -> Nx.Infix.(x <$ c))
               Nx.less;
             test_form "x <=$ c is less_equal_s x c"
               (fun x c -> Nx.Infix.(x <=$ c))
               Nx.less_equal;
             test_form "x >$ c is greater_s x c"
               (fun x c -> Nx.Infix.(x >$ c))
               Nx.greater;
             test_form "x >=$ c is greater_equal_s x c"
               (fun x c -> Nx.Infix.(x >=$ c))
               Nx.greater_equal;
             prop "a && b is logical_and a b" booleans
               (law_logical (fun a b -> Nx.Infix.(a && b)) Nx.logical_and);
             prop "a || b is logical_or a b" booleans
               (law_logical (fun a b -> Nx.Infix.(a || b)) Nx.logical_or);
             test "open Nx keeps OCaml's operators" test_stdlib;
           ];
       ])
