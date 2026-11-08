(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Attention through every kernel library the host runs: the refusals
   Nx_kernel.S.attention states for the dtypes of its arrays and of its
   program, for its program's number of operands and outputs, for a
   coordinate past the query's rank and for a visibility that reads the
   score, each before any write; a program that fits is computed. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module P = Nx_kernel.Prog
module Support = Nx_kernels_support

(* What a call answers: a refusal, or [Done]. *)
type want = Refused of A.answer | Fits

(* The value of each element [o] and [lse] hold before a call, and of
   the row past them. *)
let sentinel = 7.

(* [rows b dt tail tq] is an array [[1; tq] @ tail] of [dt] on [b]'s
   device, the first [tq] rows of an array [[1; tq + 1] @ tail] whose
   elements are [sentinel], and that array. *)
let rows (type r) (b : Support.backend) (dt : (float, r) D.t) tail tq =
  let n = (tq + 1) * Array.fold_left ( * ) 1 tail in
  let whole =
    Kernels_cases.on b
      (A.of_array dt
         (Iarray.of_array (Array.append [| 1; tq + 1 |] tail))
         (Array.make n sentinel))
  in
  let all count = { A.Move.start = 0; count; step = 1 } in
  let ranges = Array.append [| all 1; all tq |] (Array.map all tail) in
  (Option.get (A.move (Slice (Iarray.of_array ranges)) whole), whole)

(* [b]'s attention by [p] of a query [[1; tq; 1]] of [dt] against two keys
   and values, into [o] of [out] and [lse] of [acc], each {!rows}, over
   [loads] [[1; tq; 2]] of [lds]: its answer, and the elements of the
   arrays [o] and [lse] lie in after it. *)
let attention (type v s c r) (b : Support.backend) ?(tq = 1) ?(lds = [])
    (dt : (v, s) D.t) ~(acc : (float, c) D.t) ~(out : (float, r) D.t) p =
  let (module K : Nx_kernel.S) = b.kernels in
  let make dt s = A.create b.device dt s in
  let q = make dt [| 1; tq; 1 |] in
  let kv =
    Nx_kernel.Dense { k = make dt [| 1; 2; 1 |]; v = make dt [| 1; 2; 1 |] }
  in
  let loads =
    Array.of_list
      (List.map (fun (D.Any l) -> A.Any (make l [| 1; tq; 2 |])) lds)
  in
  let o, o_whole = rows b out [| 1 |] tq in
  let lse, lse_whole = rows b acc [||] tq in
  let answer = K.attention p ~scale:1. ~o ~lse q kv loads in
  let elements x = A.to_array (Kernels_cases.host x) in
  (answer, Array.append (elements o_whole) (elements lse_whole))

let f32 = D.Any D.Float32
let f64 = D.Any D.Float64
let i64 = D.Any D.Int64
let truth = P.Const (D.Any D.Bool, P.bits D.Bool true)
let zero = P.Const (f32, P.bits D.Float32 0.)

(* A mask that reads axis 3, past a query's rank of 3. *)
let past : P.node iarray =
  [|
    P.In 0; Coord 3; Const (i64, P.bits D.Int64 0L); Op2 (Compare Less, 2, 1);
  |]

(* [true] and the score: the program of attention without a mask or a
   score, over a score of [score] and loads of [lds]. *)
let plain score lds =
  P.v ~ins:(Iarray.of_list (score :: lds)) [| In 0; truth |] ~outs:[| 1; 0 |]

(* The program over a float32 score and loads of [lds] whose outputs are
   nodes [o0] and [o1] of [nodes]. *)
let prog ?(lds = []) nodes o0 o1 =
  P.v ~ins:(Iarray.of_list (f32 :: lds)) nodes ~outs:[| o0; o1 |]

let table (b : Support.backend) =
  let call ?tq ?lds p =
    attention b ?tq ?lds D.Float32 ~acc:D.Float32 ~out:D.Float32 p
  in
  let causal : P.node iarray =
    [| In 0; Coord 1; Coord 2; Op2 (Compare Less_equal, 2, 1) |]
  in
  [
    ("the plain program", Fits, fun () -> call (plain f32 []));
    ( "the plain program over a load",
      Fits,
      fun () -> call ~lds:[ f64 ] (plain f32 [ f64 ]) );
    ("a causal mask", Fits, fun () -> call (prog causal 3 0));
    ( "a mask after a node that takes the score",
      Fits,
      fun () ->
        call
          (prog
             [|
               In 0; Op2 (Binary Add, 0, 0); Coord 1; Coord 2;
               Op2 (Compare Less_equal, 3, 2);
             |]
             4 1) );
    ( "a query of int32",
      Refused A.Wrong_dtype,
      fun () ->
        attention b D.Int32 ~acc:D.Float32 ~out:D.Float32 (plain f32 []) );
    ( "an accumulator of float16",
      Refused A.Wrong_dtype,
      fun () ->
        attention b D.Float16 ~acc:D.Float16 ~out:D.Float16
          (plain (D.Any D.Float16) []) );
    ( "a float64 query into a float32 accumulator",
      Refused A.Wrong_dtype,
      fun () ->
        attention b D.Float64 ~acc:D.Float32 ~out:D.Float32 (plain f32 []) );
    ( "o of neither the query's dtype nor the accumulator",
      Refused A.Wrong_dtype,
      fun () ->
        attention b D.Float16 ~acc:D.Float32 ~out:D.Float64 (plain f32 []) );
    ( "the plain program over a score of float16",
      Refused A.Wrong_dtype,
      fun () ->
        attention b D.Float16 ~acc:D.Float32 ~out:D.Float32
          (plain (D.Any D.Float16) []) );
    ( "the plain program over a load of another dtype",
      Refused A.Wrong_dtype,
      fun () -> call ~lds:[ f64 ] (plain f32 [ f32 ]) );
    ( "a mask over a load of another dtype",
      Refused A.Wrong_dtype,
      fun () ->
        call ~lds:[ f64 ]
          (prog ~lds:[ f32 ]
             [| In 0; In 1; zero; Op2 (Compare Less, 2, 1) |]
             3 0) );
    ( "an output 1 of another dtype",
      Refused A.Wrong_dtype,
      fun () -> call (prog [| In 0; truth; Op1 (Cast, f64, 0) |] 1 2) );
    ( "an output 0 of float32",
      Refused A.Wrong_dtype,
      fun () -> call (prog [| In 0; zero |] 1 0) );
    ( "an operand past the loads",
      Refused A.Shape_mismatch,
      fun () -> call (plain f32 [ f32 ]) );
    ( "a load past the operands",
      Refused A.Shape_mismatch,
      fun () -> call ~lds:[ f32 ] (plain f32 []) );
    ( "three outputs",
      Refused A.Shape_mismatch,
      fun () ->
        call (P.v ~ins:[| f32 |] [| In 0; truth |] ~outs:[| 1; 0; 0 |]) );
    ( "a coordinate past the query's rank",
      Refused A.Shape_mismatch,
      fun () ->
        call (prog past 3 0) );
    ( "a coordinate past the query's rank, of no rows",
      Refused A.Shape_mismatch,
      fun () ->
        call ~tq:0 (prog past 3 0) );
    ( "a visibility that reads the score",
      Refused A.Shape_mismatch,
      fun () -> call (prog [| In 0; zero; Op2 (Compare Less, 1, 0) |] 2 0) );
    ( "a visibility that reads the score through another node",
      Refused A.Shape_mismatch,
      fun () ->
        call
          (prog
             [| In 0; Op2 (Binary Add, 0, 0); zero; Op2 (Compare Less, 2, 1) |]
             3 0) );
  ]

let check (b : Support.backend) (msg, want, call) =
  let got, held = b.around call in
  match want with
  | Fits -> equal ~msg Elements.answer A.Done got
  | Refused r ->
      equal ~msg Elements.answer r got;
      equal ~msg:(msg ^ ": o, lse and the row past them hold their values")
        (array float_exact)
        (Array.make (Array.length held) sentinel)
        held

let suite (b : Support.backend) =
  group b.name
    [
      cases ~id:(fun (msg, _, _) -> msg)
        "refuses the dtypes and programs it does not take, before any write"
        (table b) (check b);
    ]

let () =
  exit (Windtrap.run "nx_kernel.attention" (List.map suite Support.backends))
