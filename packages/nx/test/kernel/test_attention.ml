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

(* [b]'s attention by [p] of a query [[1; tq; 1]] of [dt] against two keys
   and values, over [loads] [[1; tq; 2]] of [lds]: the refusal
   {!Nx_kernel.attention_refusal} names it by, the dtypes and shapes of [o]
   [[1; tq; 1]] of [out] and [lse] [[1; tq]] of [acc], and the call into
   them. *)
let attention (type v s c r) (b : Support.backend) ?(tq = 1) ?(lds = [])
    (dt : (v, s) D.t) ~(acc : (float, c) D.t) ~(out : (float, r) D.t) p =
  let (module K : Nx_kernel.S) = b.kernels in
  let make dt s = A.create b.device dt s in
  let q = make dt [| 1; tq; 1 |] in
  let kv =
    Nx_kernel.Dense { k = make dt [| 1; 2; 1 |]; v = make dt [| 1; 2; 1 |] }
  in
  let loads =
    Iarray.of_list
      (List.map (fun (D.Any l) -> A.Any (make l [| 1; tq; 2 |])) lds)
  in
  let named =
    Nx_kernel.attention_refusal p ~q:dt ~rank:3 ~o:out ~acc
      ~loads:{ Nx_kernel.arrays = List.length; dtype = List.nth }
      lds
  in
  let dsts = [| (D.Any out, [| 1; tq; 1 |]); (D.Any acc, [| 1; tq |]) |] in
  let call (g : Misfit.guard) ds =
    let kv =
      match kv with
      | Nx_kernel.Dense { k; v } -> Nx_kernel.Dense { k = g.g k; v = g.g v }
      | Paged { k; v; table } -> Paged { k = g.g k; v = g.g v; table = g.g table }
    in
    K.attention p ~scale:1. ~o:(A.expect out ds.(0))
      ~lse:(A.expect acc ds.(1)) (g.g q) kv
      (Iarray.map (fun (A.Any x) -> A.Any (g.g x)) loads)
  in
  (Option.map fst named, dsts, call)

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

(* A program that fits is computed, or declined writing nothing. One that
   does not is named by {!Nx_kernel.attention_refusal}, and is not Done
   and reads and writes nothing outside its arrays. *)
let check (b : Support.backend) (msg, want, case) =
  let named, dsts, call = case () in
  match want with
  | Fits ->
      Misfit.law ~msg ~what:"none" b ~fit:true dsts (fun g ds ->
          let got = call g ds in
          equal ~msg Elements.answer A.Done got;
          got)
  | Refused r ->
      (match named with
      | Some got -> equal ~msg Elements.answer r got
      | None -> failf "%s: the rule finds it fits" msg);
      let what = if r = A.Wrong_dtype then "a dtype" else "a program" in
      Misfit.check ~msg ~what b dsts call

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
   values [v], C-contiguous [[1; 1; dk]], [[1; tk; dk]] and [[1; tk; dv]],
   the row reading the first [len] keys, all without a length. The call
   reads the query as each of [rows] rows, so that a library whose path
   follows the rows a key head holds takes each. *)
type case = {
  d : dtypes;
  rows : int;
  tk : int;
  len : int option;
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
  Format.fprintf ppf
    "@[%a, rows %d, tk %d, length %s, dk %d, dv %d,@ q [%a],@ k [%a],@ v \
     [%a]@]"
    pp_dtypes c.d c.rows c.tk
    (match c.len with None -> "none" | Some l -> string_of_int l)
    c.dk c.dv floats (Array.to_list c.q) floats (Array.to_list c.k) floats
    (Array.to_list c.v)

(* The keys a case's row reads. *)
let keys c = Option.value c.len ~default:c.tk

let dtypes = Gen.of_list ~pp:pp_dtypes [ F32; F64; F16; Bf16; F32_in_f64 ]

(* One row, or more than 16: nx.cuda takes a key head's 16 rows or fewer
   by fused multiply-adds, and more on the matrix units for float16 and
   bfloat16. *)
let rows = Gen.of_list [ 1; 17 ]

(* The dtype of the queries, keys and values. *)
let query = function
  | F32 | F32_in_f64 -> D.Any D.Float32
  | F64 -> D.Any D.Float64
  | F16 -> D.Any D.Float16
  | Bf16 -> D.Any D.Bfloat16

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

(* [o]'s cases: a zero query, [tk] a power of two up to 16 blocks of 64
   keys, past which nx.cuda may cut a row's keys into parts, often 8 or 16
   blocks, and half of them with a length, which may end in any part. *)
let sums =
  let open Gen in
  with_pp pp_case
  @@ let* d = dtypes in
  let* rows = rows in
  let* tk =
    frequency
      [ (3, map (fun e -> 1 lsl e) (int_range 0 8)); (1, of_list [ 512; 1024 ]) ]
  in
  let* len = frequency [ (1, constant None); (1, map Option.some (int_range 1 tk)) ] in
  let* dk = int_range 1 4 in
  let* dv = int_range 1 3 in
  let+ k = array ~size:(constant (tk * dk)) (element d)
  and+ v = array ~size:(constant (tk * dv)) (value d) in
  { d; rows; tk; len; dk; dv; q = Array.make dk 0.; k; v }

(* [q·k]'s cases: one key. *)
let dots =
  let open Gen in
  with_pp pp_case
  @@ let* d = dtypes in
  let* rows = rows in
  let* dk = int_range 1 40 in
  let+ q = array ~size:(constant dk) (element d)
  and+ k = array ~size:(constant dk) (element d) in
  { d; rows; tk = 1; len = None; dk; dv = 1; q; k; v = [| 1. |] }

(* [c]'s attention on [b] without mask or score, at scale 1, as the
   elements it stored [q], [k] and [v] as, then [o] and [lse]. *)
let attend (b : Support.backend) c =
  let go (type r a) (dt : (float, r) D.t) (acc : (float, a) D.t) =
    let (module K : Nx_kernel.S) = b.kernels in
    let make shape xs =
      Kernels_cases.on b (A.of_array dt (Iarray.of_array shape) xs)
    in
    let q =
      make [| 1; c.rows; c.dk |]
        (Array.init (c.rows * c.dk) (fun i -> c.q.(i mod c.dk)))
    in
    let k = make [| 1; c.tk; c.dk |] c.k in
    let v = make [| 1; c.tk; c.dv |] c.v in
    let o = A.create b.device acc [| 1; c.rows; c.dv |] in
    let lse = A.create b.device acc [| 1; c.rows |] in
    let lengths =
      Option.map
        (fun l -> Kernels_cases.on b (A.of_array D.Int64 [||] [| Int64.of_int l |]))
        c.len
    in
    let answer =
      K.attention (plain (D.Any acc) []) ~scale:1. ~o ~lse ?lengths q
        (Nx_kernel.Dense { k; v })
        [||]
    in
    equal ~msg:"attention's answer" Elements.answer A.Done answer;
    let floats x = A.to_array (Kernels_cases.host x) in
    let row x n = Array.sub (floats x) 0 n in
    ( { c with q = row q c.dk; k = floats k; v = floats v },
      row o c.dv,
      row lse 1 )
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

(* A zero query gives every visible key the output [0], so each of a
   row's [t] weights is [exp δ / t], [|δ|] at most the [δ̄] the library
   states, and is applied within [u{_v}] of that where it rounds weights to
   the values' dtype [v]: [c w ≤ 1] and [w] is normal. Each element of [o],
   in the accumulator, is then within
   [((e{^δ̄} - 1) + e(tk) e{^δ̄} + u{_v} e{^δ̄} (1 + e(tk))) Σ|v| / t] of
   the mean of its [t] values (nx_kernel.mli). *)
let law_sum (b : Support.backend) (a : Support.attention) c =
  cover "keys in several blocks" (c.tk > 64);
  cover "values that cancel"
    (Array.exists (fun x -> x > 1.) c.v && Array.exists (fun x -> x < 0.) c.v);
  cover "a length past a part's 256 keys, short of the keys"
    (match c.len with Some l -> l > 256 && l < c.tk | None -> false);
  let c, o, _ = attend b c in
  let t = keys c and acc = accumulator c.d in
  let n = Float.of_int t in
  let e = a.factor (query c.d) acc c.tk in
  let grow = Float.exp (a.zero_weights acc t) in
  let uv = if a.rounds_weights then value_roundoff c.d else 0. in
  let factor =
    grow -. 1. +. (e *. grow) +. (uv *. grow *. (1. +. e))
  in
  for j = 0 to c.dv - 1 do
    let column = Array.init t (fun i -> c.v.((i * c.dv) + j)) in
    let hi, lo, mag = exact column (Array.make t 1.) in
    within
      ~msg:(Printf.sprintf "o at %d, %h against the mean %h" j o.(j) (hi /. n))
      ~bound:((factor *. mag /. n) +. (slack t mag /. n))
      (Float.abs (o.(j) -. (hi /. n) -. (lo /. n)))
  done

(* With one key, the exact log-sum-exp is the score, [q·k] times the scale
   [1], which multiplies exactly. [lse] is within [ε] of it, and it within
   [e(dk) Σ|qᵢkᵢ|] of the exact [q·k] (nx_kernel.mli). *)
let law_dot (b : Support.backend) (a : Support.attention) c =
  let c, _, lse = attend b c in
  let hi, lo, mag = exact c.q c.k in
  cover "terms that cancel" (Float.abs (hi +. lo) < 0.5 *. mag);
  let acc = accumulator c.d in
  let bound =
    (a.factor (query c.d) acc c.dk *. mag)
    +. a.epsilon acc 1 lse.(0)
    +. slack c.dk mag
  in
  within
    ~msg:(Printf.sprintf "lse %h against q·k %h" lse.(0) hi)
    ~bound
    (Float.abs (lse.(0) -. hi -. lo))

(* [lse]'s cases: a float32 accumulator, a call of 512 or 1024 keys, whose
   row of one head nx.cuda cuts into parts of at least 256 keys, and a
   length past the first part's; elements in [[-2, 2]]. *)
let lses =
  let open Gen in
  with_pp pp_case
  @@ let* d = of_list [ F32; F16; Bf16 ] in
  let* rows = rows in
  let* tk = of_list [ 512; 1024 ] in
  let* len = int_range 257 tk in
  let* dk = int_range 1 4 in
  let+ q = array ~size:(constant dk) (float_range (-2.) 2.)
  and+ k = array ~size:(constant (tk * dk)) (float_range (-2.) 2.) in
  { d; rows; tk; len = Some len; dk; dv = 1; q; k; v = Array.make tk 1. }

(* At a row of [t] keys, [lse] is within [ε(t)] of the exact log-sum-exp
   of its computed scores, each within [e(dk) Σ|qᵢkᵢ|] of the exact [q·k]
   at scale [1], so within [ε(t)] and the largest of those of the exact
   scores' log-sum-exp (nx_kernel.mli), which float64 computes within
   [(t + 4) 2{^-52} (1 + |L|)]. *)
let law_lse (b : Support.backend) (a : Support.attention) c =
  cover "one row, whose keys go in parts" (c.rows = 1);
  let c, _, lse = attend b c in
  let t = keys c and acc = accumulator c.d in
  let e = a.factor (query c.d) acc c.dk in
  let scores =
    Array.init t (fun j ->
        let hi, lo, mag = exact c.q (Array.sub c.k (j * c.dk) c.dk) in
        (hi +. lo, mag))
  in
  let m = Array.fold_left (fun m (s, _) -> Float.max m s) Float.neg_infinity scores in
  let sum = Array.fold_left (fun acc (s, _) -> acc +. Float.exp (s -. m)) 0. scores in
  let l = m +. Float.log sum in
  let err = Array.fold_left (fun acc (_, mag) -> Float.max acc (e *. mag)) 0. scores in
  let bound =
    a.epsilon acc t lse.(0) +. err
    +. (Float.of_int (t + 4) *. 0x1p-52 *. (1. +. Float.abs l))
  in
  within
    ~msg:(Printf.sprintf "lse %h against %h, %d keys" lse.(0) l t)
    ~bound
    (Float.abs (lse.(0) -. l))

(* The error laws hold where the backend states its attention's errors:
   one that declines every attention states none. *)
let laws (b : Support.backend) =
  match b.attention with
  | None -> []
  | Some a ->
      [
        prop "o is within its bound of the weighted sum, at weights 1/tk" sums
          (law_sum b a);
        prop "q·k is within its bound of the exact sum" dots (law_dot b a);
        prop "lse is within its bound at lengths past a part" lses (law_lse b a);
      ]

let suite (b : Support.backend) =
  group b.name
    (cases
       ~id:(fun (msg, _, _) -> msg)
       "a call of dtypes or a program it does not take is not Done and \
        writes nothing outside o and lse; its rule names its refusal"
       (table b) (check b)
    :: laws b)

(* Calls

   Calls of every kind the kernel reads: lead indices, grouped heads, rows
   past a tile of 16, keys past a block, dimensions past a chunk of 32,
   columns past 128, dense or paged keys with table entries out of range,
   lengths out of [0, tk], masks by coordinates and by loads, scores
   rewritten by a load or by coordinates, and values that are not finite;
   and programs past what a launch holds: masks of many values or many
   loads, at 32 axes, and kinds of any dtype. *)

type mask = No_mask | Causal | Docs | Causal_docs | Docs32 | Wide | Many

type rescore = No_score | Bias | Alibi | Sine | Half_bias | Wide_bias | Wide_score

type call = {
  cd : dtypes;
  nb : int;
  axes : int;  (* lead axes of one element past the first *)
  hk : int;
  g : int;
  tq : int;
  ck : int;  (* keys, a whole number of pages where paged *)
  cdk : int;
  cdv : int;
  page : int;  (* 0 for dense keys *)
  lens : int array option;
  mask : mask;
  rescore : rescore;
  seed : int;
}

let mask_name = function
  | No_mask -> "none"
  | Causal -> "causal"
  | Docs -> "documents"
  | Causal_docs -> "causal documents"
  | Docs32 -> "int32 documents"
  | Wide -> "wide causal"
  | Many -> "eight loads"

let rescore_name = function
  | No_score -> "none"
  | Bias -> "bias"
  | Alibi -> "alibi"
  | Sine -> "sine"
  | Half_bias -> "float16 bias"
  | Wide_bias -> "wide alibi"
  | Wide_score -> "wide score"

let pp_call ppf c =
  Format.fprintf ppf
    "@[%a, lead %d and %d axes, hk %d, g %d, tq %d, tk %d, dk %d, dv %d, page \
     %d,@ lengths %s, mask %s, score %s, seed %d@]"
    pp_dtypes c.cd c.nb c.axes c.hk c.g c.tq c.ck c.cdk c.cdv c.page
    (match c.lens with
    | None -> "none"
    | Some l -> String.concat " " (List.map string_of_int (Array.to_list l)))
    (mask_name c.mask) (rescore_name c.rescore) c.seed

(* Calls whose keys stay below 8 blocks, so that nx.cuda cuts no keys into
   parts, and, with [fused], whose float16 and bfloat16 key heads hold 16
   rows or fewer, which nx.cuda takes by fused multiply-adds. A quarter of
   them have 32 axes. *)
let calls ~fused =
  let open Gen in
  with_pp pp_call
  @@ let* cd = dtypes in
  let* nb = int_range 1 2 in
  let* axes = frequency [ (3, constant 0); (1, constant 28) ] in
  let* hk = int_range 1 2 in
  let* g = int_range 1 3 in
  let narrow = cd = F16 || cd = Bf16 in
  let* tq = int_range 1 (if fused && narrow then 16 / g else 40) in
  let* page = frequency [ (2, constant 0); (1, of_list [ 1; 7; 16 ]) ] in
  let* tk = int_range 1 200 in
  let ck = if page = 0 then tk else (tk + page - 1) / page * page in
  let* cdk = frequency [ (3, int_range 1 40); (1, int_range 41 140) ] in
  let* cdv = frequency [ (3, int_range 1 40); (1, int_range 120 140) ] in
  let* lens =
    frequency
      [ (1, constant None);
        (1, map Option.some (array ~size:(constant nb) (int_range (-2) (ck + 2)))) ]
  in
  let* mask = of_list [ No_mask; Causal; Docs; Causal_docs; Docs32; Wide; Many ] in
  let* rescore =
    of_list [ No_score; Bias; Alibi; Sine; Half_bias; Wide_bias; Wide_score ]
  in
  let+ seed = int_range 0 0x3fffffff in
  { cd; nb; axes; hk; g; tq; ck; cdk; cdv; page; lens; mask; rescore; seed }

(* The documents of [Many]'s eight loads: by the row for an even load, by
   the key for an odd one. *)
let many = 8

(* [c]'s program over a score in [acc] and its loads, and the loads'
   dtypes: output 0 by its mask, output 1 by its score. A document mask
   reads two loads of int64 or int32, a row's document and a key's, and
   [Many] eight int64 ones; a bias reads one load in the accumulator, or in
   float16. [Wide] is the causal mask through 48 int64 values held at once,
   past what a kernel holds in shared memory; [Many] at 32 axes reads more
   loads than a launch's parameters describe; [Sine] adds the sine of the
   score; [Wide_bias] subtracts the sum of 48 int64 distances of the row
   and the key, held at once, cast and scaled; [Wide_score] sums 96
   products of the score held at once. *)
let program c acc =
  let i64 = D.Any D.Int64 and i32 = D.Any D.Int32 in
  let nodes = ref [] and n = ref 0 in
  let add x =
    nodes := x :: !nodes;
    incr n;
    !n - 1
  in
  let lds =
    (match c.mask with
    | Docs | Causal_docs -> [ i64; i64 ]
    | Docs32 -> [ i32; i32 ]
    | Many -> List.init many (fun _ -> i64)
    | No_mask | Causal | Wide -> [])
    @
    match c.rescore with
    | Bias -> [ acc ]
    | Half_bias -> [ D.Any D.Float16 ]
    | No_score | Alibi | Sine | Wide_bias | Wide_score -> []
  in
  let rank = 4 + c.axes in
  let score = add (P.In 0) in
  let row = add (P.Coord (rank - 2)) in
  let key = add (P.Coord (rank - 1)) in
  let int64 x = add (P.Const (i64, P.bits D.Int64 (Int64.of_int x))) in
  let float x =
    add
      (P.Const
         (acc, if acc = D.Any D.Float64 then P.bits D.Float64 x else P.bits D.Float32 x))
  in
  let last () = add (P.Op2 (Binary Add, row, int64 (c.ck - c.tq))) in
  let causal () = add (P.Op2 (Compare Less_equal, key, last ())) in
  let same i j = add (P.Op2 (Compare Equal, add (P.In i), add (P.In j))) in
  let all xs = List.fold_left (fun a x -> add (P.Op2 (Binary And, a, x))) (List.hd xs) (List.tl xs) in
  let o0 =
    match c.mask with
    | No_mask -> add (P.Const (D.Any D.Bool, P.bits D.Bool true))
    | Causal -> causal ()
    | Docs | Docs32 -> same 1 2
    | Causal_docs ->
        let a = causal () in
        add (P.Op2 (Binary And, a, same 1 2))
    | Wide ->
        let keys = List.init 48 (fun i -> add (P.Op2 (Binary Add, key, int64 i))) in
        let l = last () in
        all
          (List.mapi
             (fun i k -> add (P.Op2 (Compare Less_equal, k, add (P.Op2 (Binary Add, l, int64 i)))))
             keys)
    | Many -> all (List.init (many / 2) (fun i -> same ((2 * i) + 1) ((2 * i) + 2)))
  in
  let bias = P.In (List.length lds) in
  let o1 =
    match c.rescore with
    | No_score -> score
    | Bias -> add (P.Op2 (Binary Add, score, add bias))
    | Half_bias -> add (P.Op2 (Binary Add, score, add (P.Op1 (Cast, acc, add bias))))
    | Sine -> add (P.Op2 (Binary Add, score, add (P.Op1 (Unary Sin, acc, score))))
    | Wide_bias ->
        let ds =
          List.init 48 (fun i -> add (P.Op2 (Binary Sub, add (P.Op2 (Binary Add, row, int64 i)), key)))
        in
        let fs = List.map (fun d -> add (P.Op1 (Cast, acc, d))) ds in
        let sum = List.fold_left (fun a f -> add (P.Op2 (Binary Add, a, f))) (List.hd fs) (List.tl fs) in
        add (P.Op2 (Binary Sub, score, add (P.Op2 (Binary Mul, sum, float 0x1p-8))))
    | Wide_score ->
        let small = float 0x1p-7 in
        let ps = List.init 96 (fun _ -> add (P.Op2 (Binary Mul, score, small))) in
        List.fold_left (fun a x -> add (P.Op2 (Binary Add, a, x))) (List.hd ps) (List.tl ps)
    | Alibi ->
        let d = add (P.Op2 (Binary Sub, row, key)) in
        let f = add (P.Op1 (Cast, acc, d)) in
        let slope =
          add
            (P.Const
               (acc, if acc = D.Any D.Float64 then P.bits D.Float64 0.25 else P.bits D.Float32 0.25))
        in
        add (P.Op2 (Binary Sub, score, add (P.Op2 (Binary Mul, f, slope))))
  in
  ( P.v ~ins:(Iarray.of_list (acc :: lds))
      (Iarray.of_list (List.rev !nodes))
      ~outs:(Iarray.of_list [ o0; o1 ]),
    lds )

(* [c]'s call on [b]: its destinations [o] and [lse], or none where [b]
   does not answer [Done]. *)
let run_call (b : Support.backend) c =
  let go (type r a) (dt : (float, r) D.t) (acc : (float, a) D.t) =
    let (module K : Nx_kernel.S) = b.kernels in
    let st = Random.State.make [| c.seed |] in
    let value () =
      match Random.State.int st 100 with
      | 0 -> Float.nan
      | 1 -> Float.infinity
      | _ -> Random.State.float st 4. -. 2.
    in
    let finite () = Random.State.float st 4. -. 2. in
    let lead = Array.append [| c.nb |] (Array.make c.axes 1) in
    let shape s = Iarray.of_array (Array.append lead s) in
    let arr dt shape f =
      Kernels_cases.on b
        (A.of_array dt shape
           (Array.init (Iarray.fold_left ( * ) 1 shape) (fun _ -> f ())))
    in
    let hq = c.hk * c.g in
    let q = arr dt (shape [| hq; c.tq; c.cdk |]) finite in
    let kv, tk =
      if c.page = 0 then
        ( Nx_kernel.Dense
            { k = arr dt (shape [| c.hk; c.ck; c.cdk |]) finite;
              v = arr dt (shape [| c.hk; c.ck; c.cdv |]) value },
          c.ck )
      else
        let maxp = c.ck / c.page in
        let pages = maxp + 1 in
        let table =
          arr D.Int64 (shape [| maxp |]) (fun () ->
              Int64.of_int (Random.State.int st (pages + 2) - 1))
        in
        ( Nx_kernel.Paged
            { k = arr dt [| pages; c.hk; c.page; c.cdk |] finite;
              v = arr dt [| pages; c.hk; c.page; c.cdv |] value;
              table },
          c.ck )
    in
    let p, lds = program c (D.Any acc) in
    let lshape = shape [| hq; c.tq; tk |] in
    let numel = Iarray.fold_left ( * ) 1 lshape in
    (* a row's document by its row for an even load, a key's by its key
       for an odd one *)
    let doc j e =
      let key = e mod tk and row = e / tk mod c.tq in
      if j mod 2 = 0 then row / (5 + j) else key / (6 + j)
    in
    let loads =
      Array.of_list
        (List.mapi
           (fun j (D.Any l) ->
             match l with
             | D.Int64 ->
                 let x = Array.init numel (fun e -> Int64.of_int (doc j e)) in
                 A.Any (Kernels_cases.on b (A.of_array D.Int64 lshape x))
             | D.Int32 ->
                 let x = Array.init numel (fun e -> Int32.of_int (doc j e)) in
                 A.Any (Kernels_cases.on b (A.of_array D.Int32 lshape x))
             | D.Float16 -> A.Any (arr D.Float16 lshape finite)
             | _ -> A.Any (arr acc lshape finite))
           lds)
    in
    let lengths =
      Option.map
        (fun l ->
          Kernels_cases.on b
            (A.of_array D.Int64 (Iarray.of_array lead) (Array.map Int64.of_int l)))
        c.lens
    in
    let o = A.create b.device acc (shape [| hq; c.tq; c.cdv |]) in
    let lse = A.create b.device acc (shape [| hq; c.tq |]) in
    match
      K.attention p ~scale:0.375 ~o ~lse ?lengths q kv (Iarray.of_array loads)
    with
    | A.Done -> [| A.Any o; A.Any lse |]
    | _ -> [||]
  in
  b.around (fun () ->
      match c.cd with
      | F32 -> go D.Float32 D.Float32
      | F64 -> go D.Float64 D.Float64
      | F16 -> go D.Float16 D.Float32
      | Bf16 -> go D.Bfloat16 D.Float32
      | F32_in_f64 -> go D.Float32 D.Float64)

(* [got] holds [want]'s floats, bit for bit but for any NaN, which equals
   any NaN. *)
let same_floats want got =
  equal ~msg:"answers" int (Array.length want) (Array.length got);
  Array.iteri
    (fun k (A.Any w) ->
      let (A.Any g) = got.(k) in
      let floats (type v s) (x : (v, s) A.t) : float array =
        let x = Kernels_cases.host x in
        match A.dtype x with
        | D.Float32 -> A.to_array x
        | D.Float64 -> A.to_array x
        | _ -> failwith "same_floats: not a float accumulator"
      in
      let w = floats w and g = floats g in
      Array.iteri
        (fun i x ->
          let y = g.(i) in
          if not ((Float.is_nan x && Float.is_nan y)
                  || Int64.bits_of_float x = Int64.bits_of_float y)
          then failf "destination %d, element %d of %d: %h, expected %h" k i
                 (Array.length w) y x)
        w)
    want

(* nx_cuda.mli: a call whose keys go in one part, of float32 or float64
   queries, or of float16 and bfloat16 ones whose key heads hold 16 rows
   or fewer, takes nx.cpu's arithmetic: its results are nx.cpu's bits but
   for a NaN's sign and payload. *)
let law_fused (cuda : Support.backend) c =
  cover "paged" (c.page > 0);
  cover "masked by loads" (c.mask = Docs || c.mask = Causal_docs);
  cover "rescored" (c.rescore <> No_score);
  cover "rows past 16" (c.g * c.tq > 16);
  cover "32 axes" (c.axes > 0);
  cover "a mask past a launch's planes" (c.mask = Wide);
  cover "a score through a kind of the sine or of float16"
    (c.rescore = Sine || c.rescore = Half_bias);
  cover "a score past a launch's planes"
    (c.rescore = Wide_bias || c.rescore = Wide_score);
  same_floats (run_call Support.cpu c) (run_call cuda c)

(* rig.nv loads the cubin CUDA loads, so an attention through it has
   CUDA's bits: what it can get wrong, its launches, memory and order,
   shows in them. A NaN agrees with any NaN: nx_cuda.mli leaves a NaN
   result's sign and payload unspecified. *)
let law_nv first nv c =
  Elements.agree first nv (fun b -> run_call b c)

(* Calls of float16 or bfloat16 whose key heads hold more than 16 rows,
   which nx.cuda takes on the matrix units. *)
let units =
  Gen.map
    (fun c -> { c with cd = (if c.seed mod 2 = 0 then F16 else Bf16); tq = Int.max c.tq 17 })
    (calls ~fused:false)

(* On the matrix units each element keeps nx.cpu's class, a NaN, an
   infinity or a number, and a number lies within [2{^-6}] of nx.cpu's,
   relative to [1] and to the larger magnitude: far past the bounds
   nx_cuda.mli states for these calls' few products, rounded weights and
   scores below [2{^8}], so that a pair read at the wrong row or key shows. *)
let near_cpu (cuda : Support.backend) c =
  let want = run_call Support.cpu c and got = run_call cuda c in
  equal ~msg:"answers" int (Array.length want) (Array.length got);
  Array.iteri
    (fun k (A.Any w) ->
      let (A.Any g) = got.(k) in
      let floats (type v s) (x : (v, s) A.t) : float array =
        match A.dtype x with
        | D.Float32 -> A.to_array (Kernels_cases.host x)
        | _ -> failwith "law_units: not float32"
      in
      let w = floats w and g = floats g in
      Array.iteri
        (fun i x ->
          let y = g.(i) in
          let ok =
            if Float.is_nan x || Float.is_infinite x then
              Int64.bits_of_float x = Int64.bits_of_float y
              || (Float.is_nan x && Float.is_nan y)
            else
              Float.is_finite y
              && Float.abs (x -. y)
                 <= 0x1p-6 *. Float.max 1. (Float.max (Float.abs x) (Float.abs y))
          in
          if not ok then
            failf "destination %d, element %d of %d: %h, nx.cpu's %h" k i
              (Array.length w) y x)
        w)
    want

(* nx_cuda.mli's matrix units, on float16 calls of 17 rows: [q·k]'s 8
   products and the running sum keep 25 bits each from the largest
   exponent and truncate to float32, so [1 + 3·2{^-25}] gives [1], where a
   fused multiply-add rounds it to [1 + 2{^-23}]; each weight rounds once
   to float16, so [exp (-20)], below half float16's least subnormal, applies
   as [0] to a value of 1024, which a fused multiply-add would add as
   [2.1e-6]. The row's [o] and [lse], at scale 1. *)
let units_cases =
  let case tk q k v =
    { d = F16; rows = 17; tk; len = None; dk = Array.length q; dv = 1; q; k; v }
  in
  [
    ( "q·k truncates its sum to float32",
      case 1 [| 1.; 0x1p-12 |] [| 1.; 0x3p-13 |] [| 1. |],
      ([| 1. |], [| 1. |]) );
    ( "a weight rounds to the values' dtype",
      case 2 [| 1. |] [| 0.; -20. |] [| 0.; 1024. |],
      ([| 0. |], [| 0. |]) );
  ]

let units_case (cuda : Support.backend) (msg, c, (o, lse)) =
  let _, got_o, got_lse = attend cuda c in
  equal ~msg:(msg ^ ": o") (array float_exact) o got_o;
  equal ~msg:(msg ^ ": lse") (array float_exact) lse got_lse

let law_units (cuda : Support.backend) c =
  cover "masked" (c.mask <> No_mask);
  cover "rescored" (c.rescore <> No_score);
  near_cpu cuda c

(* Calls whose program a launch does not hold whole: output 0 holding 48
   int64 values at once, eight loads at 32 axes, more than a launch's
   parameters describe, and kinds outside the accumulator, int64 and bool.
   Fused ones give nx.cpu's bits; on the matrix units, its classes and
   values near its own. *)
let past_cases =
  let c =
    { cd = F32; nb = 2; axes = 0; hk = 1; g = 2; tq = 20; ck = 150; cdk = 24;
      cdv = 20; page = 0; lens = Some [| 150; 97 |]; mask = No_mask;
      rescore = No_score; seed = 7 }
  in
  let units c = { c with cd = F16; tq = 20 } in
  (* one row of a head over 11 blocks, which nx.cuda cuts into 2 parts *)
  let parts c = { c with nb = 1; g = 1; tq = 2; ck = 700; lens = Some [| 650 |] } in
  [
    ("a mask of 48 values", { c with mask = Wide });
    ("eight loads at 32 axes", { c with axes = 28; mask = Many; rescore = Bias });
    ("int32 documents", { c with mask = Docs32 });
    ("a float16 bias", { c with rescore = Half_bias; mask = Causal });
    ("the sine of the score", { c with rescore = Sine; mask = Causal_docs });
    ("a bias of 48 values", { c with rescore = Wide_bias; mask = Causal });
    ("a score of 96 values", { c with rescore = Wide_score; page = 16; ck = 160 });
    ("a mask of 48 values and the sine of the score", { c with mask = Wide; rescore = Sine });
    ("a mask of 48 values on the matrix units", units { c with mask = Wide });
    ("eight loads at 32 axes on the matrix units", units { c with axes = 28; mask = Many });
    ("the sine of the score on the matrix units", units { c with rescore = Sine; page = 7; ck = 154 });
    ("a score of 96 values on the matrix units", units { c with rescore = Wide_score; mask = Causal });
    ( "a mask of 48 values and a score of 96 on the matrix units",
      units { c with mask = Wide; rescore = Wide_score } );
    ("a float16 bias over parts", parts { c with rescore = Half_bias; mask = Causal });
    ( "a mask of 48 values and the sine over parts",
      parts { c with mask = Wide; rescore = Sine } );
    ( "a mask of 48 values and the sine over parts on the matrix units",
      units (parts { c with mask = Wide; rescore = Sine }) );
  ]

(* Fused calls of one part give nx.cpu's bits; the others come near its
   values. *)
let past_case (cuda : Support.backend) (_, c) =
  if c.cd = F16 || c.ck >= 512 then near_cpu cuda c
  else same_floats (run_call Support.cpu c) (run_call cuda c)

(* nx_cuda.mli: a block whose ranges neither hide nor show every pair runs
   output 0 at every pair, beside skipped and shown blocks of the same
   tile. Causal calls whose diagonal ends 20 to 24 keys into a block of
   64, so that its pairs are partly shown and partly hidden, of one head's
   rows or of several heads' (a tile of 40 rows a head), over dense or
   paged keys, masks by coordinates or by loads, and lengths. Fused ones
   give nx.cpu's bits; on the matrix units, its classes and values near
   its own. *)
let cut_cases =
  let c =
    { cd = F32; nb = 1; axes = 0; hk = 1; g = 1; tq = 40; ck = 124; cdk = 24;
      cdv = 20; page = 0; lens = None; mask = Causal; rescore = No_score; seed = 11 }
  in
  [
    ("rows of one head", c);
    ("rows of three heads", { c with g = 3 });
    ("paged keys", { c with page = 16; ck = 128 });
    ("documents", { c with g = 2; mask = Causal_docs });
    ("a rescored score", { c with rescore = Alibi });
    ("lengths", { c with nb = 2; lens = Some [| 124; 101 |] });
    ("float64", { c with cd = F64; g = 2 });
    ("float16 on the matrix units", { c with cd = F16; g = 2 });
    ("bfloat16 documents on the matrix units", { c with cd = Bf16; g = 3; mask = Causal_docs });
  ]

let cut_case (cuda : Support.backend) (_, c) =
  if c.cd = F16 || c.cd = Bf16 then near_cpu cuda c
  else same_floats (run_call Support.cpu c) (run_call cuda c)

(* Programs of one mask

   nx_cuda.mli: a call's association depends on its dtypes and extents
   alone, so programs that hide the same pairs give the same bits, however
   much shared memory their values take. *)

type held = {
  hg : int;
  htq : int;
  htk : int;
  hdk : int;
  hdv : int;
  hn : int;
  hseed : int;
}

let pp_held ppf h =
  Format.fprintf ppf
    "g %d, tq %d, tk %d, dk %d, dv %d, %d values held, seed %d" h.hg h.htq
    h.htk h.hdk h.hdv h.hn h.hseed

(* float32 calls of one key head whose keys go in parts, their rows past
   one tile of 16 and their causal masks through 2 to 48 values held at
   once. *)
let helds =
  let open Gen in
  with_pp pp_held
  @@ let* hg = int_range 1 2 in
  let* htq = int_range 65 128 in
  let* htk = int_range 8192 16384 in
  let* hdk = int_range 1 16 in
  let* hdv = int_range 1 16 in
  let* hn = int_range 2 48 in
  let+ hseed = int_range 0 0x3fffffff in
  { hg; htq; htk; hdk; hdv; hn; hseed }

(* The causal mask of [h]'s rows over its keys through [n] int64 values
   held at once: [key + i <= row + tk - tq + i] for every [i < n]. *)
let held_causal h n =
  let nodes = ref [] and k = ref 0 in
  let add x =
    nodes := x :: !nodes;
    incr k;
    !k - 1
  in
  let score = add (P.In 0) in
  let row = add (P.Coord 1) and key = add (P.Coord 2) in
  let int64 x = add (P.Const (i64, P.bits D.Int64 (Int64.of_int x))) in
  let keys = List.init n (fun i -> add (P.Op2 (Binary Add, key, int64 i))) in
  let last = add (P.Op2 (Binary Add, row, int64 (h.htk - h.htq))) in
  let shown i k =
    let l = add (P.Op2 (Binary Add, last, int64 i)) in
    add (P.Op2 (Compare Less_equal, k, l))
  in
  let o0 =
    match List.mapi shown keys with
    | [] -> invalid_arg "held_causal: no value"
    | x :: xs -> List.fold_left (fun a x -> add (P.Op2 (Binary And, a, x))) x xs
  in
  P.v ~ins:(Iarray.of_list [ f32 ]) (Iarray.of_list (List.rev !nodes))
    ~outs:(Iarray.of_list [ o0; score ])

(* Fresh o and lse for [h]'s calls on [b], and [h]'s call on [b] by its
   mask through [n] values held into them, on finite q, k and v drawn
   from [h.hseed]: o and lse. *)
let held_call (b : Support.backend) h =
  let (module K : Nx_kernel.S) = b.kernels in
  let st = Random.State.make [| h.hseed |] in
  let arr s =
    let n = Array.fold_left ( * ) 1 s in
    Kernels_cases.on b
      (A.of_array D.Float32 (Iarray.of_array s)
         (Array.init n (fun _ -> Random.State.float st 4. -. 2.)))
  in
  let q = arr [| h.hg; h.htq; h.hdk |] in
  let k = arr [| 1; h.htk; h.hdk |] and v = arr [| 1; h.htk; h.hdv |] in
  let outputs () =
    ( A.create b.device D.Float32 [| h.hg; h.htq; h.hdv |],
      A.create b.device D.Float32 [| h.hg; h.htq |] )
  in
  let call (o, lse) n =
    let p = held_causal h n in
    match K.attention p ~scale:0.375 ~o ~lse q (Nx_kernel.Dense { k; v }) [||] with
    | A.Done -> [| A.Any o; A.Any lse |]
    | _ -> [||]
  in
  (outputs, call)

(* [h]'s call on [b] by its mask through one value held, then through
   [h.hn], on the same q, k and v: o and lse of each. *)
let held_calls (b : Support.backend) h =
  let outputs, call = held_call b h in
  b.around (fun () -> (call (outputs ()) 1, call (outputs ()) h.hn))

let law_held (b : Support.backend) h =
  let one, many = held_calls b h in
  same_floats one many

(* The memory of calls whose program is split

   nx_cuda.mli: such a call takes a temporary for each node [map]
   computes, an element of the node's dtype at each index of the loop's
   axes along which the node's leaves vary, and [P r (dv + 2)] accumulator elements
   for its parts, [P <= 256]; its memory returns once its work is done. *)

let held_large =
  { hg = 2; htq = 128; htk = 16384; hdk = 16; hdv = 16; hn = 48; hseed = 1 }

(* The bytes nx_cuda.mli states [h]'s call takes at most, each temporary
   from 16-byte aligned bytes: the int64 keys [key + i] and rows
   [row + tk - tq + i], their comparisons and the conjunctions of these,
   and the parts' states. *)
let held_bytes h =
  let n = h.hn and pairs = h.htq * h.htk in
  let nodes = (n * 8 * h.htk) + ((n + 1) * 8 * h.htq) + (((2 * n) - 1) * pairs) in
  nodes + (16 * ((4 * n) + 1)) + (256 * h.hg * h.htq * (h.hdv + 2) * 4)

let test_held_loop (b : Support.backend) () =
  let h = held_large in
  let outputs, call = held_call b h in
  let outs = outputs () in
  Kernels_cases.loop_held b ~calls:8 ~bound:(held_bytes h) (fun () ->
      Array.iter
        (fun (A.Any x) -> Rig.Buffer.wait (A.buffer x) Read_write)
        (call outs h.hn))

let gpu_laws (cuda : Support.backend) =
  group (cuda.name ^ " against cpu")
    [
      prop "a mask through more values held keeps the bits" helds (law_held cuda);
      test "calls of a split mask hold one call's memory"
        (test_held_loop cuda);
      prop "fused calls give nx.cpu's bits" (calls ~fused:true) (law_fused cuda);
      prop "matrix units keep nx.cpu's classes and come near its values" units
        (law_units cuda);
      cases ~id:(fun (msg, _, _) -> msg) "matrix units sum as nx_cuda.mli states"
        units_cases (units_case cuda);
      cases ~id:fst "programs past a launch compute" past_cases (past_case cuda);
      cases ~id:fst "blocks the diagonal cuts" cut_cases (cut_case cuda);
    ]

let nv_laws (first : Support.backend) (nv : Support.backend) =
  group nv.name
    [
      prop "bits are cuda's, any NaN for a NaN" (calls ~fused:false)
        (law_nv first nv);
      prop "a mask through more values held keeps the bits" helds (law_held nv);
    ]

let () =
  let nv =
    match (Support.gpus (), Support.nv ()) with
    | first :: _, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.attention"
       (List.map suite (Support.backends ())
       @ List.map gpu_laws (Support.gpus ())
       @ nv))
