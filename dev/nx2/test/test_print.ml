(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Printing: the form nx.mli states, each float printed so that it reads back
   to itself, and every kind of value printed without raising. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])

let printed x = Nx.to_string x

let starts ~prefix s =
  String.length s >= String.length prefix
  && String.sub s 0 (String.length prefix) = prefix

let has_prefix prefix s =
  if not (starts ~prefix s) then failf "%S does not start with %S" s prefix

(* Forms *)

let forms =
  [
    ( "a matrix",
      (fun () -> printed (Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])),
      "float32 [2; 3] [[1, 2, 3], [4, 5, 6]]" );
    ( "a 0-d value",
      (fun () -> printed (Nx.create Nx.int64 [||] [| -7L |])),
      "int64 [] -7" );
    ( "unsigned integers",
      (fun () -> printed (Nx.create Nx.uint32 [| 2 |] [| -1l; 3l |])),
      "uint32 [2] [4294967295, 3]" );
    ( "complex numbers",
      (fun () ->
        printed
          (Nx.create Nx.complex64 [| 2 |]
             [| Complex.one; { Complex.re = 0.5; im = -2. } |])),
      "complex64 [2] [1+0i, 0.5-2i]" );
    ( "floats in their shortest form",
      (fun () -> printed (Nx.create Nx.float16 [| 4 |] [| 0.1; nan; neg_infinity; -0. |])),
      "float16 [4] [0.1, nan, -inf, -0]" );
    ( "booleans", (fun () -> printed (Nx.create Nx.bool [| 2 |] [| true; false |])),
      "bool [2] [true, false]" );
    ("an empty value", (fun () -> printed (Nx.create Nx.int8 [| 0 |] [||])), "int8 [0] []");
    ( "an empty inner axis",
      (fun () -> printed (Nx.create Nx.int8 [| 2; 0 |] [||])),
      "int8 [2; 0] [[], []]" );
    ( "1000 elements in full",
      (fun () ->
        let s = printed (Nx.place Nx.Host.on (Nx.arange Nx.int16 0 1000 1)) in
        string_of_int (List.length (String.split_on_char ',' s))),
      "1000" );
    ( "past 1000, the ends of a long axis",
      (fun () -> printed (Nx.place Nx.Host.on (Nx.arange Nx.int32 0 1001 1))),
      "int32 [1001] [0, 1, 2, ..., 998, 999, 1000]" );
    ("pp_shape", (fun () -> Format.asprintf "%a" Nx.pp_shape [| 2; 3 |]), "[2; 3]");
    ("pp_shape of 0-d", (fun () -> Format.asprintf "%a" Nx.pp_shape [||]), "[]");
    ("pp_dtype", (fun () -> Format.asprintf "%a" Nx.pp_dtype Nx.bfloat16), "bfloat16");
  ]

let form_cases =
  cases "forms" ~name:(fun (n, _, _) -> n) forms (fun (_, f, expected) ->
      equal string expected (f ()))

(* Values of every kind *)

let test_on_a_set () =
  let x = Nx.place (S2.split ~axis:0) (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
  has_prefix "float32 [2] on set " (printed x);
  equal bool true (String.ends_with ~suffix:"[1, 2]" (printed x))

let test_every_set () =
  has_prefix "float32 [2; 3] of every set"
    (printed (Nx.add (Nx.zeros Nx.float32 [| 2; 3 |]) (Nx.scalar Nx.float32 1.)))

let test_dead () =
  let x = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
  ignore (Nx.add (Nx.donate x) (Nx.create Nx.float32 [| 2 |] [| 0.; 0. |]));
  equal string "float32 [2], donated to Nx.add" (printed x)

type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

let test_traced () =
  let x = Nx.create Nx.int8 [| 3 |] [| 1; 2; 3 |] in
  Nx.Prim.interpret ~name:"test.print" Values
    (fun i ~by op -> Nx.Prim.results ~by (fun _ f -> Nx.Prim.traced i f Mark) op)
    (fun i ->
      let t = Nx.Prim.traced i (Nx.Prim.form x) Mark in
      equal string "int8 [3] traced by test.print" (printed t))

(* Floats read back *)

let drawn (type s) (dt : (float, s) D.t) n : (float, s, Nx.host) Nx.t Gen.t =
  let open Gen in
  let+ data = string_of ~size:(constant (D.bytes dt n)) char in
  Nx.Repr.of_array Nx.Host.v
    (A.v dt (L.contiguous [| n |]) (Rig.Buffer.of_string (if data = "" then "\000" else data)))

let same a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* Each printed element, read by [float_of_string] and stored in [dt], is the
   element; an infinity reads as itself, since no store makes float8_e5m2's. *)
let law_reads_back (type s) (dt : (float, s) D.t) x =
  let s = printed x in
  let body = String.sub s (String.index s '[' + 1) (String.length s - String.index s '[' - 1) in
  let body = String.sub body (String.index body '[' + 1) (String.length body - String.index body '[' - 2) in
  let parsed =
    if body = "" then [||]
    else
      Array.of_list
        (List.map
           (fun e ->
             let v = float_of_string (String.trim e) in
             if Float.is_finite v then D.of_float dt v else v)
           (String.split_on_char ',' body))
  in
  let xs = Nx.to_array x in
  equal int (Array.length xs) (Array.length parsed);
  Array.iteri
    (fun i v -> if not (same v parsed.(i)) then failf "%h printed as %h" v parsed.(i))
    xs

let reads_back =
  let floats =
    List.filter_map
      (fun (D.Any dt) ->
        match D.kind dt with
        | D.Float ->
            Some
              (prop
                 (D.name dt ^ " reads back")
                 (Gen.with_pp (fun ppf _ -> D.pp ppf dt) (drawn dt 50))
                 (law_reads_back dt))
        | _ -> None)
      D.all
  in
  group "each printed float reads back to itself" floats

let () =
  exit
    (run "nx print"
       [
         form_cases;
         group "values of every kind"
           [
             test "a value on a set names the set" test_on_a_set;
             test "a value of every set prints as a formula" test_every_set;
             test "a dead value names its consumer" test_dead;
             test "a traced value names its interpretation" test_traced;
           ];
         reads_back;
       ])
