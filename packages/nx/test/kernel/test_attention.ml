(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Attention through every kernel library the host runs: the refusals
   Nx_kernel.S.attention states for the dtypes of its arrays and of its
   program, for its program's number of operands and outputs, for a
   coordinate past the query's rank and for a visibility that reads the
   score, each before any write; a program that fits is computed; and the
   bounds of q·k and of o's weighted sum. *)

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

(* Bounds *)

(* A query's dtype and the accumulator's. *)
type dtypes = F32 | F64 | F16 | Bf16 | F32_in_f64

let pp_dtypes ppf d =
  Format.pp_print_string ppf
    (match d with
    | F32 -> "float32"
    | F64 -> "float64"
    | F16 -> "float16 in float32"
    | Bf16 -> "bfloat16 in float32"
    | F32_in_f64 -> "float32 in float64")

let accumulator = function
  | F64 | F32_in_f64 -> D.Any D.Float64
  | F32 | F16 | Bf16 -> D.Any D.Float32

(* The unit roundoff of the values' dtype where weights may round to it:
   float16's and bfloat16's. *)
let value_roundoff = function
  | F16 -> 0x1p-11
  | Bf16 -> 0x1p-8
  | F32 | F64 | F32_in_f64 -> 0.

(* A large element: a power of two the query's dtype holds, past the
   accumulator's precision where the dtype's range reaches, so that adding
   1 to it rounds. *)
let large = function
  | F64 -> 0x1p54
  | F16 -> 0x1p15
  | F32 | Bf16 | F32_in_f64 -> 0x1p25


(* A case: [tk] keys of [dk] and [dv] elements, a query [q], keys [k] and
   values [v], C-contiguous [[1; 1; dk]], [[1; tk; dk]] and [[1; tk; dv]]. *)
type case = {
  d : dtypes;
  tk : int;
  dk : int;
  dv : int;
  q : float array;
  k : float array;
  v : float array;
}

let pp_case ppf c =
  let floats =
    Format.pp_print_list ~pp_sep:Format.pp_print_space Format.pp_print_float
  in
  Format.fprintf ppf "@[%a, tk %d, dk %d, dv %d,@ q [%a],@ k [%a],@ v [%a]@]"
    pp_dtypes c.d c.tk c.dk c.dv floats (Array.to_list c.q) floats
    (Array.to_list c.k) floats (Array.to_list c.v)

let dtypes = Gen.of_list ~pp:pp_dtypes [ F32; F64; F16; Bf16; F32_in_f64 ]

(* Elements of [d]: [±large d] and [±1] often, so that sums cancel, and
   others in [[-2, 2]]. *)
let element d =
  let m = large d in
  Gen.frequency
    [
      (2, Gen.of_list [ m; -.m ]);
      (2, Gen.of_list [ 1.; -1. ]);
      (1, Gen.float_range (-2.) 2.);
    ]

(* Values [+M], [-M] and [1]. A sum that meets [M] and [1] rounds, and
   one that also meets [-M] cancels: a group summed at once, aligned to [M]
   and truncated, may lose [1]s that every order of fused multiply-adds
   keeps, which the normwise bound allows. *)
let value d =
  let m = large d in
  Gen.of_list [ m; -.m; 1. ]

(* [o]'s cases: a zero query, [tk] a power of two up to four blocks of 64
   keys. *)
let sums =
  let open Gen in
  with_pp pp_case
  @@ let* d = dtypes in
  let* tk = map (fun e -> 1 lsl e) (int_range 0 8) in
  let* dk = int_range 1 4 in
  let* dv = int_range 1 3 in
  let+ k = array ~size:(constant (tk * dk)) (element d)
  and+ v = array ~size:(constant (tk * dv)) (value d) in
  { d; tk; dk; dv; q = Array.make dk 0.; k; v }

(* [q·k]'s cases: one key. *)
let dots =
  let open Gen in
  with_pp pp_case
  @@ let* d = dtypes in
  let* dk = int_range 1 40 in
  let+ q = array ~size:(constant dk) (element d)
  and+ k = array ~size:(constant dk) (element d) in
  { d; tk = 1; dk; dv = 1; q; k; v = [| 1. |] }

(* [c]'s attention on [b] without mask or score, at scale 1, as the
   elements it stored [q], [k] and [v] as, then [o] and [lse]. *)
let attend (b : Support.backend) c =
  let go (type r a) (dt : (float, r) D.t) (acc : (float, a) D.t) =
    let (module K : Nx_kernel.S) = b.kernels in
    let make shape xs =
      Kernels_cases.on b (A.of_array dt (Iarray.of_array shape) xs)
    in
    let q = make [| 1; 1; c.dk |] c.q in
    let k = make [| 1; c.tk; c.dk |] c.k in
    let v = make [| 1; c.tk; c.dv |] c.v in
    let o = A.create b.device acc [| 1; 1; c.dv |] in
    let lse = A.create b.device acc [| 1; 1 |] in
    let answer =
      K.attention (plain (D.Any acc) []) ~scale:1. ~o ~lse q
        (Nx_kernel.Dense { k; v })
        [||]
    in
    equal ~msg:"attention's answer" Elements.answer A.Done answer;
    let floats x = A.to_array (Kernels_cases.host x) in
    ({ c with q = floats q; k = floats k; v = floats v }, floats o, floats lse)
  in
  b.around (fun () ->
      match c.d with
      | F32 -> go D.Float32 D.Float32
      | F64 -> go D.Float64 D.Float64
      | F16 -> go D.Float16 D.Float32
      | Bf16 -> go D.Bfloat16 D.Float32
      | F32_in_f64 -> go D.Float32 D.Float64)

(* [Σ xᵢ yᵢ] as an unevaluated pair [hi + lo], and [Σ |xᵢ yᵢ|] rounded.
   [hi]'s errors and the products' are exact, at most [u] times [Σ|xᵢ yᵢ|]
   each with [u] float64's, and [lo] adds the [2n] of them with [2n]
   roundings, so [hi + lo] is within [slack n] of the exact sum: at most
   [2n (n + 2) u²] times [Σ|xᵢ yᵢ|]. *)
let exact x y =
  let hi = ref 0. and lo = ref 0. and mag = ref 0. in
  for i = 0 to Array.length x - 1 do
    let p = x.(i) *. y.(i) in
    let s, t = Contract_cases.two_sum !hi p in
    hi := s;
    lo := !lo +. t +. Float.fma x.(i) y.(i) (-.p);
    mag := !mag +. Float.abs p
  done;
  (!hi, !lo, !mag)

let slack n mag = Float.of_int (2 * n * (n + 2)) *. 0x1p-106 *. mag

(* [err] at most [bound]: the subtractions that computed [err] round it by
   at most a relative [2{^-52}]. *)
let within ~msg ~bound err =
  at_most ~msg float_exact ~than:(bound *. (1. +. 0x1p-51)) err

(* A zero query gives every visible key the output [0], so each weight is
   [exp δ / tk], [|δ|] at most the [δ̄] the library states, and is applied
   within [u{_v}] of that where it rounds weights to the values' dtype
   [v]: [c w ≤ 1] and [w] is normal. Each element of [o], in the
   accumulator, is then within
   [((e{^δ̄} - 1) + e(tk) e{^δ̄} + u{_v} e{^δ̄} (1 + e(tk))) Σ|v| / tk] of
   the values' mean (nx_kernel.mli). *)
let law_sum (b : Support.backend) c =
  cover "keys in several blocks" (c.tk > 64);
  cover "values that cancel"
    (Array.exists (fun x -> x > 1.) c.v && Array.exists (fun x -> x < 0.) c.v);
  let c, o, _ = attend b c in
  let n = Float.of_int c.tk and acc = accumulator c.d in
  let e = b.attention.factor acc c.tk in
  let grow = Float.exp (b.attention.zero_weights acc c.tk) in
  let uv = if b.attention.rounds_weights then value_roundoff c.d else 0. in
  let factor =
    grow -. 1. +. (e *. grow) +. (uv *. grow *. (1. +. e))
  in
  for j = 0 to c.dv - 1 do
    let column = Array.init c.tk (fun t -> c.v.((t * c.dv) + j)) in
    let hi, lo, mag = exact column (Array.make c.tk 1.) in
    within
      ~msg:(Printf.sprintf "o at %d, %h against the mean %h" j o.(j) (hi /. n))
      ~bound:((factor *. mag /. n) +. (slack c.tk mag /. n))
      (Float.abs (o.(j) -. (hi /. n) -. (lo /. n)))
  done

(* With one key, the exact log-sum-exp is the score, [q·k] times the scale
   [1], which multiplies exactly. [lse] is within [ε] of it, and it within
   [e(dk) Σ|qᵢkᵢ|] of the exact [q·k] (nx_kernel.mli). *)
let law_dot (b : Support.backend) c =
  let c, _, lse = attend b c in
  let hi, lo, mag = exact c.q c.k in
  cover "terms that cancel" (Float.abs (hi +. lo) < 0.5 *. mag);
  let acc = accumulator c.d in
  let bound =
    (b.attention.factor acc c.dk *. mag)
    +. b.attention.epsilon acc 1 lse.(0)
    +. slack c.dk mag
  in
  within
    ~msg:(Printf.sprintf "lse %h against q·k %h" lse.(0) hi)
    ~bound
    (Float.abs (lse.(0) -. hi -. lo))

let suite (b : Support.backend) =
  group b.name
    [
      cases ~id:(fun (msg, _, _) -> msg)
        "refuses the dtypes and programs it does not take, before any write"
        (table b) (check b);
      prop "o is within its bound of the weighted sum, at weights 1/tk" sums
        (law_sum b);
      prop "q·k is within its bound of the exact sum" dots (law_dot b);
    ]

let () =
  exit (Windtrap.run "nx_kernel.attention" (List.map suite Support.backends))
