(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Attention through every kernel library the host runs, as nx_kernel.mli states
   it: a program that fits is computed; the refusals Nx_kernel.S.attention
   states for the dtypes of its arrays and of its program, for its program's
   number of operands and outputs, for a coordinate past the query's rank and
   for a visibility that reads the score, each before any write, and the door's
   for an o over q's bytes; the bounds of q·k, of o's weighted sum and of lse;
   and a kept value at a score of -∞, a hidden one never. Then each GPU library
   against nx.cpu's bits. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module P = Nx_kernel.Prog
module Support = Nx_kernels_support

(* What a call answers: a refusal, or [Done]. *)
type want = Refused of A.answer | Fits | Declines

(* [b]'s attention by [p] of a query [[1; tq; 1]] of [dt] against two keys and
   values, over [loads] [[1; tq; 2]] of [lds]: the refusal
   {!Nx_kernel.attention_refusal} names it by, the dtypes and shapes of [o] [[1;
   tq; 1]] of [out] and [lse] [[1; tq]] of [acc], and the call into them. *)
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
      | Paged { k; v; table } ->
          Paged { k = g.g k; v = g.g v; table = g.g table }
    in
    K.attention p ~scale:1.
      ~o:(A.expect out ds.(0))
      ~lse:(A.expect acc ds.(1))
      (g.g q) kv
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

(* [true], the score and [true]: the program of attention without a mask, a
   score or a keep, over a score of [score] and loads of [lds]. *)
let plain score lds =
  P.v ~ins:(Iarray.of_list (score :: lds)) [| In 0; truth |] ~outs:[| 1; 0; 1 |]

(* The program over a float32 score and loads of [lds] whose outputs are nodes
   [o0], [o1] and [o2] of [nodes], [o2] by default a [true] past them. *)
let prog ?(lds = []) ?o2 nodes o0 o1 =
  let nodes, o2 =
    match o2 with
    | Some o2 -> (nodes, o2)
    | None -> (Iarray.append nodes [| truth |], Iarray.length nodes)
  in
  P.v ~ins:(Iarray.of_list (f32 :: lds)) nodes ~outs:[| o0; o1; o2 |]

(* Calls, each with what it answers on the library of [label], and the call on a
   backend. *)
let table label =
  let call b ?tq ?lds p =
    attention b ?tq ?lds D.Float32 ~acc:D.Float32 ~out:D.Float32 p
  in
  let causal : P.node iarray =
    [| In 0; Coord 1; Coord 2; Op2 (Compare Less_equal, 2, 1) |]
  in
  [
    ("the plain program", Fits, fun b -> call b (plain f32 []));
    ( "the plain program over a load",
      Fits,
      fun b -> call b ~lds:[ f64 ] (plain f32 [ f64 ]) );
    ("a causal mask", Fits, fun b -> call b (prog causal 3 0));
    ( "a mask after a node that takes the score",
      Fits,
      fun b ->
        call b
          (prog
             [|
               In 0;
               Op2 (Binary Add, 0, 0);
               Coord 1;
               Coord 2;
               Op2 (Compare Less_equal, 3, 2);
             |]
             4 1) );
    ( "a query of int32",
      Refused A.Wrong_dtype,
      fun b -> attention b D.Int32 ~acc:D.Float32 ~out:D.Float32 (plain f32 [])
    );
    ( "an accumulator of float16",
      Refused A.Wrong_dtype,
      fun b ->
        attention b D.Float16 ~acc:D.Float16 ~out:D.Float16
          (plain (D.Any D.Float16) []) );
    ( "a float64 query into a float32 accumulator",
      Refused A.Wrong_dtype,
      fun b ->
        attention b D.Float64 ~acc:D.Float32 ~out:D.Float32 (plain f32 []) );
    ( "o of neither the query's dtype nor the accumulator",
      Refused A.Wrong_dtype,
      fun b ->
        attention b D.Float16 ~acc:D.Float32 ~out:D.Float64 (plain f32 []) );
    ( "the plain program over a score of float16",
      Refused A.Wrong_dtype,
      fun b ->
        attention b D.Float16 ~acc:D.Float32 ~out:D.Float32
          (plain (D.Any D.Float16) []) );
    ( "the plain program over a load of another dtype",
      Refused A.Wrong_dtype,
      fun b -> call b ~lds:[ f64 ] (plain f32 [ f32 ]) );
    ( "a mask over a load of another dtype",
      Refused A.Wrong_dtype,
      fun b ->
        call b ~lds:[ f64 ]
          (prog ~lds:[ f32 ]
             [| In 0; In 1; zero; Op2 (Compare Less, 2, 1) |]
             3 0) );
    ( "an output 1 of another dtype",
      Refused A.Wrong_dtype,
      fun b -> call b (prog [| In 0; truth; Op1 (Cast, f64, 0) |] 1 2) );
    ( "an output 0 of float32",
      Refused A.Wrong_dtype,
      fun b -> call b (prog [| In 0; zero |] 1 0) );
    ( "an operand past the loads",
      Refused A.Shape_mismatch,
      fun b -> call b (plain f32 [ f32 ]) );
    ( "a load past the operands",
      Refused A.Shape_mismatch,
      fun b -> call b ~lds:[ f32 ] (plain f32 []) );
    ( "two outputs",
      Refused A.Shape_mismatch,
      fun b -> call b (P.v ~ins:[| f32 |] [| In 0; truth |] ~outs:[| 1; 0 |]) );
    ( "four outputs",
      Refused A.Shape_mismatch,
      fun b ->
        call b (P.v ~ins:[| f32 |] [| In 0; truth |] ~outs:[| 1; 0; 1; 1 |]) );
    (* nx_metal.mli: nx.metal declines a keep but the constant true. *)
    ( "a causal keep",
      (if label = "metal" then Declines else Fits),
      fun b -> call b (prog ~o2:3 (Iarray.append causal [| truth |]) 4 0) );
    ( "an output 2 of float32",
      Refused A.Wrong_dtype,
      fun b -> call b (prog ~o2:1 [| In 0; zero; truth |] 2 0) );
    ( "a keep that reads the score",
      Refused A.Shape_mismatch,
      fun b ->
        call b
          (prog ~o2:2 [| In 0; zero; Op2 (Compare Less, 1, 0); truth |] 3 0) );
    ( "a coordinate past the query's rank",
      Refused A.Shape_mismatch,
      fun b -> call b (prog past 3 0) );
    ( "a coordinate past the query's rank, of no rows",
      Refused A.Shape_mismatch,
      fun b -> call b ~tq:0 (prog past 3 0) );
    ( "a visibility that reads the score",
      Refused A.Shape_mismatch,
      fun b -> call b (prog [| In 0; zero; Op2 (Compare Less, 1, 0) |] 2 0) );
    ( "a visibility that reads the score through another node",
      Refused A.Shape_mismatch,
      fun b ->
        call b
          (prog
             [| In 0; Op2 (Binary Add, 0, 0); zero; Op2 (Compare Less, 2, 1) |]
             3 0) );
  ]

let fits (_, want, _) =
  match want with Refused _ -> false | Fits | Declines -> true

(* A program that fits is computed, or, where its library states it declines
   one, declined; either reads and writes nothing outside its arrays. *)
let check_fits (b : Support.backend) (msg, want, case) =
  let _, dsts, call = case b in
  let want = if want = Declines then A.Declined else A.Done in
  Misfit.law ~msg ~what:"none" b ~fit:true dsts (fun g ds ->
      let got = call g ds in
      equal ~msg Elements.answer want got;
      got)

(* A call that does not fit is named by {!Nx_kernel.attention_refusal}, is not
   Done, and reads and writes nothing outside its arrays. *)
let check_refused (b : Support.backend) (msg, want, case) =
  let named, dsts, call = case b in
  match want with
  | Fits | Declines -> invalid_arg "check_refused: a call that fits"
  | Refused r ->
      (match named with
      | Some got -> equal ~msg Elements.answer r got
      | None -> failf "%s: the rule finds it fits" msg);
      let what = if r = A.Wrong_dtype then "a dtype" else "a program" in
      Misfit.check ~msg ~what b dsts call

(* An o over q's bytes, one element on: the door refuses the call as Overlapping
   (nx_kernel.mli), before any write. *)
let test_overlapping (b : Support.backend) () =
  let (module K : Nx_kernel.S) = b.kernels in
  let x = A.create b.device D.Float32 [| 3 |] in
  let from start =
    Option.get
      (A.move
         (Nx_array.Move.Slice [| { Nx_array.Move.start; count = 2; step = 1 } |])
         x)
  in
  let shaped x = Option.get (A.move (Nx_array.Move.Reshape [| 1; 1; 2 |]) x) in
  let q = shaped (from 0) and o = shaped (from 1) in
  let k = A.create b.device D.Float32 [| 1; 2; 2 |] in
  let v = A.create b.device D.Float32 [| 1; 2; 2 |] in
  let lse = A.create b.device D.Float32 [| 1; 1 |] in
  equal Elements.answer A.Overlapping
    (K.attention (plain f32 []) ~scale:1. ~o ~lse q
       (Nx_kernel.Dense { k; v })
       [||])

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
   accumulator's precision where the dtype's range reaches, so that adding 1 to
   it rounds. *)
let large = function
  | F64 -> 0x1p54
  | F16 -> 0x1p15
  | F32 | Bf16 | F32_in_f64 -> 0x1p25

(* A case: [tk] keys of [dk] and [dv] elements, a query [q], keys [k] and values
   [v], C-contiguous [[1; 1; dk]], [[1; tk; dk]] and [[1; tk; dv]], the row
   reading the first [len] keys, all without a length. The call reads the query
   as each of [rows] rows, so that a library whose path follows the rows a key
   head holds takes each. *)
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
    "@[%a, rows %d, tk %d, length %s, dk %d, dv %d,@ q [%a],@ k [%a],@ v [%a]@]"
    pp_dtypes c.d c.rows c.tk
    (match c.len with None -> "none" | Some l -> string_of_int l)
    c.dk c.dv floats (Array.to_list c.q) floats (Array.to_list c.k) floats
    (Array.to_list c.v)

(* The keys a case's row reads. *)
let keys c = Option.value c.len ~default:c.tk
let dtypes = Gen.of_list ~pp:pp_dtypes [ F32; F64; F16; Bf16; F32_in_f64 ]

(* One row, or more than 16: nx.cuda takes a key head's 16 rows or fewer by
   fused multiply-adds, and more on the matrix units for float16 and
   bfloat16. *)
let rows = Gen.of_list [ 1; 17 ]

(* The dtype of the queries, keys and values. *)
let query = function
  | F32 | F32_in_f64 -> D.Any D.Float32
  | F64 -> D.Any D.Float64
  | F16 -> D.Any D.Float16
  | Bf16 -> D.Any D.Bfloat16

(* Elements of [d]: [±large d] and [±1] often, so that sums cancel, and others
   in [[-2, 2]]. *)
let element d =
  let m = large d in
  Gen.frequency
    [
      (2, Gen.of_list [ m; -.m ]);
      (2, Gen.of_list [ 1.; -1. ]);
      (1, Gen.float_range (-2.) 2.);
    ]

(* Values [+M], [-M] and [1]. A sum that meets [M] and [1] rounds, and one that
   also meets [-M] cancels: a group summed at once, aligned to [M] and
   truncated, may lose [1]s that every order of fused multiply-adds keeps, which
   the normwise bound allows. *)
let value d =
  let m = large d in
  Gen.of_list [ m; -.m; 1. ]

(* [o]'s cases: a zero query, [tk] a power of two up to 16 blocks of 64 keys,
   past which nx.cuda may cut a row's keys into parts, half of them 8 or 16
   blocks. A third have no length, a third any length, and a third a length past
   a part's 256 keys and short of [tk], where [tk] has one, so that about 1 case
   in 4 is one, and 100 cases cover it but with probability below 10^-13. *)
let sums dtypes =
  let open Gen in
  with_pp pp_case
  @@ let* d = dtypes in
     let* rows = rows in
     let* tk =
       frequency
         [
           (1, map (fun e -> 1 lsl e) (int_range 0 8));
           (1, of_list [ 512; 1024 ]);
         ]
     in
     let past_part =
       if tk > 257 then int_range 257 (tk - 1) else int_range 1 tk
     in
     let* len =
       frequency
         [
           (1, constant None);
           (1, map Option.some (int_range 1 tk));
           (1, map Option.some past_part);
         ]
     in
     let* dk = int_range 1 4 in
     let* dv = int_range 1 3 in
     let+ k = array ~size:(constant (tk * dk)) (element d)
     and+ v = array ~size:(constant (tk * dv)) (value d) in
     { d; rows; tk; len; dk; dv; q = Array.make dk 0.; k; v }

(* [q·k]'s cases: one key. *)
let dots dtypes =
  let open Gen in
  with_pp pp_case
  @@ let* d = dtypes in
     let* rows = rows in
     let* dk = int_range 1 40 in
     let+ q = array ~size:(constant dk) (element d)
     and+ k = array ~size:(constant dk) (element d) in
     { d; rows; tk = 1; len = None; dk; dv = 1; q; k; v = [| 1. |] }

(* [c]'s attention on [b] without mask or score, at scale 1, as the elements it
   stored [q], [k] and [v] as, then [o] and [lse]. *)
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
        (fun l ->
          Kernels_cases.on b (A.of_array D.Int64 [||] [| Int64.of_int l |]))
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

(* [Σ xᵢ yᵢ] as an unevaluated pair [hi + lo], and [Σ |xᵢ yᵢ|] rounded. [hi]'s
   errors and the products' are exact, at most [u] times [Σ|xᵢ yᵢ|] each with
   [u] float64's, and [lo] adds the [2n] of them with [2n] roundings, so [hi +
   lo] is within [slack n] of the exact sum: at most [2n (n + 2) u²] times [Σ|xᵢ
   yᵢ|]. *)
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

(* [err] at most [bound]: the subtractions that computed [err] round it by at
   most a relative [2{^-52}]. A NaN error, of a NaN result, is past every bound,
   though windtrap orders NaN below every float. *)
let within ~msg ~bound err =
  let err = if Float.is_nan err then Float.infinity else err in
  at_most ~msg float_exact ~than:(bound *. (1. +. 0x1p-51)) err

(* The lse a library's [ε] reads: the computed one where finite, else the
   reference [l], so that a NaN or infinite result leaves the bound a number. *)
let lse_at lse l = if Float.is_finite lse then lse else l

(* A zero query gives every visible key the output [0], so each of a row's [t]
   weights is [exp δ / t], [|δ|] at most the [δ̄] the library states, and is
   applied within [u{_v}] of that where it rounds weights to the values' dtype
   [v]: [c w ≤ 1] and [w] is normal. Each element of [o], in the accumulator, is
   then within [((e{^δ̄} - 1) + e(tk) e{^δ̄} + u{_v} e{^δ̄} (1 + e(tk))) Σ|v| / t]
   of the mean of its [t] values (nx_kernel.mli). *)
let law_sum (b : Support.backend) (a : Support.attention) c =
  assume (a.float64 || accumulator c.d <> f64);
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
  let factor = grow -. 1. +. (e *. grow) +. (uv *. grow *. (1. +. e)) in
  for j = 0 to c.dv - 1 do
    let column = Array.init t (fun i -> c.v.((i * c.dv) + j)) in
    let hi, lo, mag = exact column (Array.make t 1.) in
    within
      ~msg:(Printf.sprintf "o at %d, %h against the mean %h" j o.(j) (hi /. n))
      ~bound:((factor *. mag /. n) +. (slack t mag /. n))
      (Float.abs (o.(j) -. (hi /. n) -. (lo /. n)))
  done

(* With one key, the exact log-sum-exp is the score, [q·k] times the scale [1],
   which multiplies exactly. [lse] is within [ε] of it, and it within [e(dk)
   Σ|qᵢkᵢ|] of the exact [q·k] (nx_kernel.mli). *)
let law_dot (b : Support.backend) (a : Support.attention) c =
  assume (a.float64 || accumulator c.d <> f64);
  let c, _, lse = attend b c in
  let hi, lo, mag = exact c.q c.k in
  cover "terms that cancel" (Float.abs (hi +. lo) < 0.5 *. mag);
  let acc = accumulator c.d in
  let bound =
    (a.factor (query c.d) acc c.dk *. mag)
    +. a.epsilon acc 1 (lse_at lse.(0) (hi +. lo))
    +. slack c.dk mag
  in
  within
    ~msg:(Printf.sprintf "lse %h against q·k %h" lse.(0) hi)
    ~bound
    (Float.abs (lse.(0) -. hi -. lo))

(* [lse]'s cases: a float32 accumulator, a call of 512 or 1024 keys, whose row
   of one head nx.cuda cuts into parts of at least 256 keys, and a length past
   the first part's; elements in [[-2, 2]]. *)
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

(* At a row of [t] keys, [lse] is within [ε(t)] of the exact log-sum-exp of its
   computed scores, each within [e(dk) Σ|qᵢkᵢ|] of the exact [q·k] at scale [1],
   so within [ε(t)] and the largest of those of the exact scores' log-sum-exp
   (nx_kernel.mli), which float64 computes within [(t + 4) 2{^-52} (1 +
   |L|)]. *)
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
  let m =
    Array.fold_left (fun m (s, _) -> Float.max m s) Float.neg_infinity scores
  in
  let sum =
    Array.fold_left (fun acc (s, _) -> acc +. Float.exp (s -. m)) 0. scores
  in
  let l = m +. Float.log sum in
  let err =
    Array.fold_left (fun acc (_, mag) -> Float.max acc (e *. mag)) 0. scores
  in
  let bound =
    a.epsilon acc t (lse_at lse.(0) l)
    +. err
    +. (Float.of_int (t + 4) *. 0x1p-52 *. (1. +. Float.abs l))
  in
  within
    ~msg:(Printf.sprintf "lse %h against %h, %d keys" lse.(0) l t)
    ~bound
    (Float.abs (lse.(0) -. l))

(* A float dtype. *)
type floats = Floats : (float, 's) D.t -> floats

(* A query's dtype in an accumulator's, a call of 1 or 17 rows, and the block of
   512 keys that holds the values not finite: block 0, block 4, which begins the
   second of the two parts nx.cuda cuts these keys into, and block 7. 17 rows
   nx.cuda takes on its matrix units at float16 and bfloat16, and nx.cpu on its
   chain kernel. *)
let special_rows =
  List.concat_map
    (fun (q, acc) ->
      List.concat_map
        (fun rows -> List.map (fun r -> (q, acc, rows, r)) [ 0; 4; 7 ])
        [ 1; 17 ])
    [
      (Floats D.Float16, Floats D.Float32);
      (Floats D.Bfloat16, Floats D.Float32);
      (Floats D.Float32, Floats D.Float32);
      (Floats D.Float16, Floats D.Float64);
      (Floats D.Bfloat16, Floats D.Float64);
      (Floats D.Float32, Floats D.Float64);
      (Floats D.Float64, Floats D.Float64);
    ]

let special_id (Floats q, Floats acc, rows, r) =
  Printf.sprintf "%s in %s, %d rows, block %d" (D.name q) (D.name acc) rows r

(* A call of 512 keys in 8 blocks: those of block [r] hold NaN and infinite
   values under scores of [-∞], output 1 adding a load of [-∞] to them; the
   others hold [3] and [-0.5] at scores of [0]. Where [hide], a mask load hides
   block [r] but its last key, which holds [3] and [-0.5] at a score of [0] so
   that the block is read. Its o and lse. *)
let specials (b : Support.backend) (a : Support.attention) ~hide
    (Floats dt, Floats acc, rows, r) =
  let tk = 512 and special = [| nan; infinity; neg_infinity |] in
  let (module K : Nx_kernel.S) = b.kernels in
  if D.equal acc D.Float64 && not a.float64 then
    skip ~reason:"no float64 accumulator" ();
  let make dt shape xs =
    Kernels_cases.on b (A.of_array dt (Iarray.of_array shape) xs)
  in
  let bad j = j / 64 = r && not (hide && j mod 64 = 63) in
  let value i =
    let j = i / 2 and e = i mod 2 in
    if bad j then special.((j + e) mod 3) else if e = 0 then 3. else -0.5
  in
  let per_pair f = Array.init (rows * tk) (fun i -> f (i mod tk)) in
  let q = make dt [| 1; rows; 1 |] (Array.make rows 1.) in
  let k = make dt [| 1; tk; 1 |] (Array.make tk 0.) in
  let v = make dt [| 1; tk; 2 |] (Array.init (2 * tk) value) in
  let bias =
    make acc [| 1; rows; tk |]
      (per_pair (fun j -> if bad j then neg_infinity else 0.))
  in
  let seen =
    make D.Bool [| 1; rows; tk |] (per_pair (fun j -> not (hide && bad j)))
  in
  let s = D.Any acc in
  let p =
    P.v ~ins:[| s; s; D.Any D.Bool |]
      [| In 0; In 1; Op2 (Binary Add, 0, 1); In 2; truth |]
      ~outs:[| 3; 2; 4 |]
  in
  let o = A.create b.device acc [| 1; rows; 2 |] in
  let lse = A.create b.device acc [| 1; rows |] in
  equal ~msg:"answer" Elements.answer A.Done
    (K.attention p ~scale:1. ~o ~lse q
       (Nx_kernel.Dense { k; v })
       [| A.Any bias; A.Any seen |]);
  let floats x = A.to_array (Kernels_cases.host x) in
  (floats o, floats lse, a.epsilon (D.Any acc) tk)

(* nx_kernel.mli: a kept value counts where its weight underflows to 0, so a NaN
   or an infinite value at a score of [-∞] makes o NaN; lse is then the
   log-sum-exp of the 448 other keys' scores of 0. *)
let test_kept b a row =
  let _, _, rows, _ = row in
  let o, lse, eps = specials b a ~hide:false row in
  equal ~msg:"o" (array float_exact) (Array.make (2 * rows) nan) o;
  let l = Float.log 448. in
  equal ~msg:"lse" (array (float (eps l))) (Array.make rows l) lse

(* nx_cpu.mli and nx_cuda.mli: a value at a pair hidden from a row, finite or
   not, never reaches its o. The row adds the 449 exact products of the keys it
   sees, so o is [3] and [-0.5] exactly and lse [log 449]. *)
let test_hidden b a row =
  let _, _, rows, _ = row in
  let o, lse, eps = specials b a ~hide:true row in
  equal ~msg:"o" (array float_exact)
    (Array.concat (List.init rows (fun _ -> [| 3.; -0.5 |])))
    o;
  let l = Float.log 449. in
  equal ~msg:"lse" (array (float (eps l))) (Array.make rows l) lse

(* The suite *)

let name (msg, _, _) = msg

let suite (d : Support.declared) =
  let run f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  (* The laws of a library's errors, which one that declines every attention
     states none of. *)
  let stated f x =
    run
      (fun b x ->
        match b.attention with
        | None -> skip ~reason:(d.label ^ " declines every attention") ()
        | Some a -> f b a x)
      x
  in
  let rows = table d.label in
  group d.label
    [
      cases ~id:name "programs that fit are computed" (List.filter fits rows)
        (run check_fits);
      cases ~id:name
        "calls that do not fit are named by attention_refusal, not Done, and \
         write nothing outside o and lse"
        (List.filter (fun r -> not (fits r)) rows)
        (run check_refused);
      test "an o over q's bytes is refused as Overlapping"
        (run (fun b () -> test_overlapping b ()));
      prop "q·k is within its bound of the exact sum" (dots dtypes)
        (stated law_dot);
      prop "o is within its bound of the weighted sum, at weights 1/tk"
        (sums dtypes) (stated law_sum);
      prop "lse is within its bound at lengths past a part" lses
        (stated law_lse);
      cases ~id:special_id "a kept value counts at a score of -∞" special_rows
        (stated test_kept);
      cases ~id:special_id "a hidden value never reaches o" special_rows
        (stated test_hidden);
    ]

(* Calls

   Calls of every kind the kernel reads: lead indices, grouped heads, rows past
   a tile of 16, keys past a block, dimensions past a chunk of 32, columns past
   128, dense or paged keys with table entries out of range, lengths out of [0,
   tk], masks by coordinates and by loads, scores rewritten by a load or by
   coordinates, and values that are not finite; and programs past what a launch
   holds: masks of many values or many loads, at 32 axes, and kinds of any
   dtype. *)

type mask =
  | No_mask
  | Causal
  | Docs
  | Causal_docs
  | Docs32
  | Wide
  | Many
  | Window

type rescore =
  | No_score
  | Bias
  | Scaled_bias
  | Alibi
  | Sine
  | Half_bias
  | Wide_bias
  | Wide_score

(* Where q, k and v lie: each in a buffer of its own, one element into a larger
   buffer, with their last two axes stored the other way, or with their rows
   (q's queries, k's and v's keys or slots) stored last first, at a negative
   step. *)
type lay = Own | Shifted | Swapped | Flipped

(* Which visible pairs keep their weight: all, those of a row and key whose sum
   3 does not divide, or those a bool load holds, most of them. *)
type keep = Keep_all | Keep_coords | Keep_load

type call = {
  cd : dtypes;
  nb : int;
  axes : int; (* lead axes of one element past the first *)
  hk : int;
  g : int;
  tq : int;
  ck : int; (* keys, a whole number of pages where paged *)
  cdk : int;
  cdv : int;
  page : int; (* 0 for dense keys *)
  lens : int array option;
  mask : mask;
  rescore : rescore;
  lay : lay;
  bounded : bool;
      (* q's and k's elements bounded so that each score lies within ±8 *)
  poisoned : bool;
      (* loads hold NaN, or an integer's ends, at keys past the length *)
  keep : keep;
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
  | Window -> "window"

let rescore_name = function
  | No_score -> "none"
  | Bias -> "bias"
  | Scaled_bias -> "scaled bias"
  | Alibi -> "alibi"
  | Sine -> "sine"
  | Half_bias -> "float16 bias"
  | Wide_bias -> "wide alibi"
  | Wide_score -> "wide score"

let lay_name = function
  | Own -> "own"
  | Shifted -> "shifted"
  | Swapped -> "swapped"
  | Flipped -> "flipped"

let keep_name = function
  | Keep_all -> "all"
  | Keep_coords -> "by coordinates"
  | Keep_load -> "by a load"

let pp_call ppf c =
  Format.fprintf ppf
    "@[%a, lead %d and %d axes, hk %d, g %d, tq %d, tk %d, dk %d, dv %d, page \
     %d,@ lengths %s, mask %s, score %s, laid %s,%s%s keep %s, seed %d@]"
    pp_dtypes c.cd c.nb c.axes c.hk c.g c.tq c.ck c.cdk c.cdv c.page
    (match c.lens with
    | None -> "none"
    | Some l -> String.concat " " (List.map string_of_int (Array.to_list l)))
    (mask_name c.mask) (rescore_name c.rescore) (lay_name c.lay)
    (if c.bounded then " bounded," else "")
    (if c.poisoned then " poisoned," else "")
    (keep_name c.keep) c.seed

(* Calls whose keys stay below 8 blocks, so that nx.cuda cuts no keys into
   parts, and, with [fused], whose float16 and bfloat16 key heads hold 16 rows
   or fewer, which nx.cuda takes by fused multiply-adds. A quarter of them have
   32 axes. *)
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
         [
           (1, constant None);
           ( 1,
             map Option.some
               (array ~size:(constant nb) (int_range (-2) (ck + 2))) );
         ]
     in
     let* mask =
       of_list [ No_mask; Causal; Docs; Causal_docs; Docs32; Wide; Many ]
     in
     let* rescore =
       of_list [ No_score; Bias; Alibi; Sine; Half_bias; Wide_bias; Wide_score ]
     in
     let* keep =
       frequency
         [
           (2, constant Keep_all);
           (1, constant Keep_coords);
           (1, constant Keep_load);
         ]
     in
     let+ seed = int_range 0 0x3fffffff in
     {
       cd;
       nb;
       axes;
       hk;
       g;
       tq;
       ck;
       cdk;
       cdv;
       page;
       lens;
       mask;
       rescore;
       lay = Own;
       bounded = false;
       poisoned = false;
       keep;
       seed;
     }

(* The documents of [Many]'s eight loads: by the row for an even load, by the
   key for an odd one. *)
let many = 8

(* The keys [Window] shows a row: the [window] keys up to its last. *)
let window = 100

(* [c]'s program over a score in [acc] and its loads, and the loads' dtypes:
   output 0 by its mask, output 1 by its score, output 2 by its keep, whose
   load, a bool, comes last. A document mask reads two loads of int64 or int32,
   a row's document and a key's, and [Many] eight int64 ones; [Window] is the
   causal mask of a row's [window] keys up to its last; a bias reads one load in
   the accumulator, or in float16, and [Scaled_bias] adds half of it. [Wide] is
   the causal mask through 48 int64 values held at once, past what a kernel
   holds in shared memory; [Many] at 32 axes reads more loads than a launch's
   parameters describe; [Sine] adds the sine of the score; [Wide_bias] subtracts
   the sum of 48 int64 distances of the row and the key, held at once, cast and
   scaled; [Wide_score] sums 96 products of the score held at once. *)
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
      | No_mask | Causal | Wide | Window -> [])
    @ (match c.rescore with
      | Bias | Scaled_bias -> [ acc ]
      | Half_bias -> [ D.Any D.Float16 ]
      | No_score | Alibi | Sine | Wide_bias | Wide_score -> [])
    @
    match c.keep with
    | Keep_load -> [ D.Any D.Bool ]
    | Keep_all | Keep_coords -> []
  in
  let rank = 4 + c.axes in
  let score = add (P.In 0) in
  let row = add (P.Coord (rank - 2)) in
  let key = add (P.Coord (rank - 1)) in
  let int64 x = add (P.Const (i64, P.bits D.Int64 (Int64.of_int x))) in
  let float x =
    add
      (P.Const
         ( acc,
           if acc = D.Any D.Float64 then P.bits D.Float64 x
           else P.bits D.Float32 x ))
  in
  let last () = add (P.Op2 (Binary Add, row, int64 (c.ck - c.tq))) in
  let causal () = add (P.Op2 (Compare Less_equal, key, last ())) in
  let same i j = add (P.Op2 (Compare Equal, add (P.In i), add (P.In j))) in
  let all xs =
    List.fold_left
      (fun a x -> add (P.Op2 (Binary And, a, x)))
      (List.hd xs) (List.tl xs)
  in
  let o0 =
    match c.mask with
    | No_mask -> add (P.Const (D.Any D.Bool, P.bits D.Bool true))
    | Causal -> causal ()
    | Docs | Docs32 -> same 1 2
    | Causal_docs ->
        let a = causal () in
        add (P.Op2 (Binary And, a, same 1 2))
    | Wide ->
        let keys =
          List.init 48 (fun i -> add (P.Op2 (Binary Add, key, int64 i)))
        in
        let l = last () in
        all
          (List.mapi
             (fun i k ->
               add
                 (P.Op2
                    (Compare Less_equal, k, add (P.Op2 (Binary Add, l, int64 i)))))
             keys)
    | Many ->
        all (List.init (many / 2) (fun i -> same ((2 * i) + 1) ((2 * i) + 2)))
    | Window ->
        let first = add (P.Op2 (Binary Sub, last (), int64 (window - 1))) in
        let a = causal () in
        let seen = add (P.Op2 (Compare Less_equal, first, key)) in
        add (P.Op2 (Binary And, a, seen))
  in
  let bias = P.In (List.length lds - if c.keep = Keep_load then 1 else 0) in
  let o1 =
    match c.rescore with
    | No_score -> score
    | Bias -> add (P.Op2 (Binary Add, score, add bias))
    | Scaled_bias ->
        let half = add (P.Op2 (Binary Mul, add bias, float 0.5)) in
        add (P.Op2 (Binary Add, score, half))
    | Half_bias ->
        add (P.Op2 (Binary Add, score, add (P.Op1 (Cast, acc, add bias))))
    | Sine ->
        add (P.Op2 (Binary Add, score, add (P.Op1 (Unary Sin, acc, score))))
    | Wide_bias ->
        let ds =
          List.init 48 (fun i ->
              add
                (P.Op2 (Binary Sub, add (P.Op2 (Binary Add, row, int64 i)), key)))
        in
        let fs = List.map (fun d -> add (P.Op1 (Cast, acc, d))) ds in
        let sum =
          List.fold_left
            (fun a f -> add (P.Op2 (Binary Add, a, f)))
            (List.hd fs) (List.tl fs)
        in
        add
          (P.Op2 (Binary Sub, score, add (P.Op2 (Binary Mul, sum, float 0x1p-8))))
    | Wide_score ->
        let small = float 0x1p-7 in
        let ps =
          List.init 96 (fun _ -> add (P.Op2 (Binary Mul, score, small)))
        in
        List.fold_left
          (fun a x -> add (P.Op2 (Binary Add, a, x)))
          (List.hd ps) (List.tl ps)
    | Alibi ->
        let d = add (P.Op2 (Binary Sub, row, key)) in
        let f = add (P.Op1 (Cast, acc, d)) in
        let slope =
          add
            (P.Const
               ( acc,
                 if acc = D.Any D.Float64 then P.bits D.Float64 0.25
                 else P.bits D.Float32 0.25 ))
        in
        add (P.Op2 (Binary Sub, score, add (P.Op2 (Binary Mul, f, slope))))
  in
  let o2 =
    match c.keep with
    | Keep_all -> add (P.Const (D.Any D.Bool, P.bits D.Bool true))
    | Keep_coords ->
        let sum = add (P.Op2 (Binary Add, row, key)) in
        let r = add (P.Op2 (Binary Mod, sum, int64 3)) in
        add (P.Op2 (Compare Not_equal, r, int64 0))
    | Keep_load -> add (P.In (List.length lds))
  in
  ( P.v
      ~ins:(Iarray.of_list (acc :: lds))
      (Iarray.of_list (List.rev !nodes))
      ~outs:(Iarray.of_list [ o0; o1; o2 ]),
    lds )

(* [c]'s call on [b]: its answer, and its destinations [o] and [lse], or none
   where [b] does not answer [Done]. *)
let call_on (b : Support.backend) c =
  let go (type r a) (dt : (float, r) D.t) (acc : (float, a) D.t) =
    let (module K : Nx_kernel.S) = b.kernels in
    let st = Random.State.make [| c.seed |] in
    (* Bounded, an element is 0 or of magnitude in [2{^-20}, m], so that no
       product of two is subnormal. *)
    let draw m =
      let x = Random.State.float st (2. *. m) -. m in
      if c.bounded && Float.abs x < 0x1p-20 then 0. else x
    in
    let value () =
      match Random.State.int st 100 with
      | 0 -> Float.nan
      | 1 -> Float.infinity
      | _ -> draw 2.
    in
    let finite () = draw 2. in
    (* Bounded, [|q·k| ≤ dk m²], and the scale [0.375] keeps each score within
       [±8]. *)
    let qk =
      if c.bounded && c.cdk > 0 then
        Float.min 2. (Float.sqrt (8. /. (0.375 *. Float.of_int c.cdk)))
      else 2.
    in
    let element () = draw qk in
    let lead = Array.append [| c.nb |] (Array.make c.axes 1) in
    let shape s = Iarray.of_array (Array.append lead s) in
    let arr dt shape f =
      Kernels_cases.on b
        (A.of_array dt shape
           (Array.init (Iarray.fold_left ( * ) 1 shape) (fun _ -> f ())))
    in
    (* [arr] as [c.lay] lays q, k and v out, with the same elements. *)
    let laid dt shape f =
      let xs = Array.init (Iarray.fold_left ( * ) 1 shape) (fun _ -> f ()) in
      let r = Iarray.length shape in
      let view base strides ?offset s =
        A.v dt
          (A.Layout.v ?offset ~strides s)
          (A.buffer (Kernels_cases.on b base))
      in
      match c.lay with
      | Own -> Kernels_cases.on b (A.of_array dt shape xs)
      | Shifted ->
          let base =
            A.of_array dt [| Array.length xs + 1 |] (Array.append [| 0. |] xs)
          in
          view base
            (A.Layout.strides (A.Layout.contiguous shape))
            ~offset:1 shape
      | Swapped ->
          let swap a =
            Iarray.init r (fun i ->
                if i = r - 2 then Iarray.get a (r - 1)
                else if i = r - 1 then Iarray.get a (r - 2)
                else Iarray.get a i)
          in
          let stored = swap shape in
          let m = Iarray.get shape (r - 2) and n = Iarray.get shape (r - 1) in
          (* element (i, j) of a matrix of the logical array at (j, i) *)
          let ys =
            Array.init (Array.length xs) (fun e ->
                let mat = e / (m * n) and at = e mod (m * n) in
                let j = at / m and i = at mod m in
                xs.((mat * m * n) + (i * n) + j))
          in
          let base = A.of_array dt stored ys in
          view base (swap (A.Layout.strides (A.Layout.contiguous stored))) shape
      | Flipped ->
          let m = Iarray.get shape (r - 2) and n = Iarray.get shape (r - 1) in
          (* row i of a matrix of the logical array at row m - 1 - i *)
          let ys =
            Array.init (Array.length xs) (fun e ->
                let mat = e / (m * n) and at = e mod (m * n) in
                xs.((mat * m * n) + ((m - 1 - (at / n)) * n) + (at mod n)))
          in
          let strides = A.Layout.strides (A.Layout.contiguous shape) in
          let strides =
            Iarray.init r (fun i ->
                let s = Iarray.get strides i in
                if i = r - 2 then -s else s)
          in
          view (A.of_array dt shape ys) strides
            ~offset:(Int.max 0 ((m - 1) * n))
            shape
    in
    let hq = c.hk * c.g in
    let q = laid dt (shape [| hq; c.tq; c.cdk |]) element in
    let kv, tk =
      if c.page = 0 then
        ( Nx_kernel.Dense
            {
              k = laid dt (shape [| c.hk; c.ck; c.cdk |]) element;
              v = laid dt (shape [| c.hk; c.ck; c.cdv |]) value;
            },
          c.ck )
      else
        let maxp = c.ck / c.page in
        let pages = maxp + 1 in
        let table =
          arr D.Int64 (shape [| maxp |]) (fun () ->
              Int64.of_int (Random.State.int st (pages + 2) - 1))
        in
        ( Nx_kernel.Paged
            {
              k =
                laid dt
                  (Iarray.of_array [| pages; c.hk; c.page; c.cdk |])
                  element;
              v =
                laid dt (Iarray.of_array [| pages; c.hk; c.page; c.cdv |]) value;
              table;
            },
          c.ck )
    in
    let p, lds = program c (D.Any acc) in
    let lshape = shape [| hq; c.tq; tk |] in
    let numel = Iarray.fold_left ( * ) 1 lshape in
    (* a row's document by its row for an even load, a key's by its key for an
       odd one *)
    let doc j e =
      let key = e mod tk and row = e / tk mod c.tq in
      if j mod 2 = 0 then row / (5 + j) else key / (6 + j)
    in
    (* Poisoned, the element at [e] lies past its lead index's length. *)
    let past e =
      match c.lens with
      | Some l when c.poisoned -> e mod tk >= l.(e / (hq * c.tq * tk))
      | _ -> false
    in
    let floats dt =
      let x =
        Array.init numel (fun e ->
            let y = finite () in
            if past e then Float.nan else y)
      in
      A.Any (Kernels_cases.on b (A.of_array dt lshape x))
    in
    let loads =
      Array.of_list
        (List.mapi
           (fun j (D.Any l) ->
             match l with
             | D.Int64 ->
                 let x =
                   Array.init numel (fun e ->
                       if not (past e) then Int64.of_int (doc j e)
                       else if e mod 2 = 0 then Int64.max_int
                       else Int64.min_int)
                 in
                 A.Any (Kernels_cases.on b (A.of_array D.Int64 lshape x))
             | D.Int32 ->
                 let x =
                   Array.init numel (fun e ->
                       if not (past e) then Int32.of_int (doc j e)
                       else if e mod 2 = 0 then Int32.max_int
                       else Int32.min_int)
                 in
                 A.Any (Kernels_cases.on b (A.of_array D.Int32 lshape x))
             | D.Float16 -> floats D.Float16
             | D.Bool ->
                 A.Any (arr D.Bool lshape (fun () -> Random.State.int st 5 > 0))
             | _ -> floats acc)
           lds)
    in
    let lengths =
      Option.map
        (fun l ->
          Kernels_cases.on b
            (A.of_array D.Int64 (Iarray.of_array lead)
               (Array.map Int64.of_int l)))
        c.lens
    in
    let o = A.create b.device acc (shape [| hq; c.tq; c.cdv |]) in
    let lse = A.create b.device acc (shape [| hq; c.tq |]) in
    match
      K.attention p ~scale:0.375 ~o ~lse ?lengths q kv (Iarray.of_array loads)
    with
    | A.Done -> (A.Done, [| A.Any o; A.Any lse |])
    | answer -> (answer, [||])
  in
  b.around (fun () ->
      match c.cd with
      | F32 -> go D.Float32 D.Float32
      | F64 -> go D.Float64 D.Float64
      | F16 -> go D.Float16 D.Float32
      | Bf16 -> go D.Bfloat16 D.Float32
      | F32_in_f64 -> go D.Float32 D.Float64)

let run_call b c = snd (call_on b c)

(* [got] holds [want]'s floats, bit for bit but for any NaN, which equals any
   NaN. *)
let same_floats ?(msg = "") want got =
  equal ~msg:(msg ^ "answers") int (Array.length want) (Array.length got);
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
          if
            not
              ((Float.is_nan x && Float.is_nan y)
              || Int64.bits_of_float x = Int64.bits_of_float y)
          then
            failf "%sdestination %d, element %d of %d: %h, expected %h" msg k i
              (Array.length w) y x)
        w)
    want

(* nx_cuda.mli: a call whose keys go in one part, of float32 or float64 queries,
   or of float16 and bfloat16 ones whose key heads hold 16 rows or fewer, takes
   nx.cpu's arithmetic: its results are nx.cpu's bits but for a NaN's sign and
   payload. *)
(* Whether a row of [c] reads 8 keys or more: its values, drawn at full
   precision, then sum to other bits in another order but by chance. *)
let shows_order c =
  c.ck >= 8
  &&
  match c.lens with
  | None -> true
  | Some l -> Array.exists (fun x -> x >= 8) l

let law_fused (cuda : Support.backend) c =
  cover "a row whose values' sum shows its order" (shows_order c);
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
  cover "a keep" (c.keep <> Keep_all);
  cover "a keep by a load" (c.keep = Keep_load);
  same_floats (run_call Support.cpu c) (run_call cuda c)

(* rig.nv loads the cubin CUDA loads, so an attention through it has CUDA's
   bits: what it can get wrong, its launches, memory and order, shows in them. A
   NaN agrees with any NaN: nx_cuda.mli leaves a NaN result's sign and payload
   unspecified. *)
let law_nv first nv c =
  cover "a row whose values' sum shows its order" (shows_order c);
  Elements.agree first nv (fun b -> run_call b c)

(* Calls of float16 or bfloat16 whose key heads hold more than 16 rows, which
   nx.cuda takes on the matrix units. *)
let units =
  Gen.map
    (fun c ->
      {
        c with
        cd = (if c.seed mod 2 = 0 then F16 else Bf16);
        tq = Int.max c.tq 17;
      })
    (calls ~fused:false)

(* On the matrix units each element keeps nx.cpu's class, a NaN, an infinity or
   a number, and a number lies within [2{^-6}] of nx.cpu's, relative to [1] and
   to the larger magnitude: far past the bounds nx_cuda.mli states for these
   calls' few products, rounded weights and scores below [2{^8}], so that a pair
   read at the wrong row or key shows. *)
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
                 <= 0x1p-6
                    *. Float.max 1. (Float.max (Float.abs x) (Float.abs y))
          in
          if not ok then
            failf "destination %d, element %d of %d: %h, nx.cpu's %h" k i
              (Array.length w) y x)
        w)
    want

(* nx_cuda.mli's matrix units, on float16 calls of 17 rows: [q·k]'s 8 products
   and the running sum keep 25 bits each from the largest exponent and truncate
   to float32, so [1 + 3·2{^-25}] gives [1], where a fused multiply-add rounds
   it to [1 + 2{^-23}]; each weight rounds once to float16, so [exp (-20)],
   below half float16's least subnormal, applies as [0] to a value of 1024,
   which a fused multiply-add would add as [2.1e-6]; a block's products [y] sum
   from [+0] and [a] becomes [α a + y] in one fused multiply-add, so key 0's
   value 43.25 at [α = exp (-20)] against the next block's value 1 gives [1 +
   2{^-23}], where a matrix unit adding [α a] and the product would truncate it
   to [1]. The row's [o] and [lse], at scale 1. *)
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
    ( "a block's products add to α a in one fused multiply-add",
      case 65 [| 1. |]
        (Array.init 65 (fun j ->
             if j = 0 then 0. else if j = 64 then 20. else Float.neg_infinity))
        (Array.init 65 (fun j ->
             if j = 0 then 43.25 else if j = 64 then 1. else 0.)),
      ([| 1. +. 0x1p-23 |], [| 20. |]) );
  ]

let units_case (cuda : Support.backend) (msg, c, (o, lse)) =
  let _, got_o, got_lse = attend cuda c in
  equal ~msg:(msg ^ ": o") (array float_exact) o got_o;
  equal ~msg:(msg ^ ": lse") (array float_exact) lse got_lse

let law_units (cuda : Support.backend) c =
  cover "masked" (c.mask <> No_mask);
  cover "rescored" (c.rescore <> No_score);
  near_cpu cuda c

(* Calls whose program a launch does not hold whole: output 0 holding 48 int64
   values at once, eight loads at 32 axes, more than a launch's parameters
   describe, and kinds outside the accumulator, int64 and bool. Fused ones give
   nx.cpu's bits; on the matrix units, its classes and values near its own. *)
let past_cases =
  let c =
    {
      cd = F32;
      nb = 2;
      axes = 0;
      hk = 1;
      g = 2;
      tq = 20;
      ck = 150;
      cdk = 24;
      cdv = 20;
      page = 0;
      lens = Some [| 150; 97 |];
      mask = No_mask;
      rescore = No_score;
      lay = Own;
      bounded = false;
      poisoned = false;
      keep = Keep_all;
      seed = 7;
    }
  in
  let units c = { c with cd = F16; tq = 20 } in
  (* one row of a head over 11 blocks, which nx.cuda cuts into 2 parts *)
  let parts c =
    { c with nb = 1; g = 1; tq = 2; ck = 700; lens = Some [| 650 |] }
  in
  [
    ("a mask of 48 values", { c with mask = Wide });
    ( "a mask of 48 values and a keep by a load",
      { c with mask = Wide; keep = Keep_load } );
    ("eight loads at 32 axes", { c with axes = 28; mask = Many; rescore = Bias });
    ( "eight loads at 32 axes and a keep by a load",
      { c with axes = 28; mask = Many; rescore = Bias; keep = Keep_load } );
    ("int32 documents", { c with mask = Docs32 });
    ("a float16 bias", { c with rescore = Half_bias; mask = Causal });
    ("the sine of the score", { c with rescore = Sine; mask = Causal_docs });
    ("a bias of 48 values", { c with rescore = Wide_bias; mask = Causal });
    ( "a score of 96 values",
      { c with rescore = Wide_score; page = 16; ck = 160 } );
    ( "a mask of 48 values and the sine of the score",
      { c with mask = Wide; rescore = Sine } );
    ("a mask of 48 values on the matrix units", units { c with mask = Wide });
    ( "eight loads at 32 axes on the matrix units",
      units { c with axes = 28; mask = Many } );
    ( "a keep by coordinates on the matrix units",
      units { c with mask = Causal; keep = Keep_coords } );
    ( "the sine of the score on the matrix units",
      units { c with rescore = Sine; page = 7; ck = 154 } );
    ( "a score of 96 values on the matrix units",
      units { c with rescore = Wide_score; mask = Causal } );
    ( "a mask of 48 values and a score of 96 on the matrix units",
      units { c with mask = Wide; rescore = Wide_score } );
    ( "a float16 bias over parts",
      parts { c with rescore = Half_bias; mask = Causal } );
    ( "a keep by a load over parts",
      parts { c with keep = Keep_load; mask = Causal } );
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
   output 0 at every pair, beside skipped and shown blocks of the same tile.
   Causal calls whose diagonal ends 20 to 24 keys into a block of 64, so that
   its pairs are partly shown and partly hidden, of one head's rows or of
   several heads' (a tile of 40 rows a head), over dense or paged keys, masks by
   coordinates or by loads, and lengths. Fused ones give nx.cpu's bits; on the
   matrix units, its classes and values near its own. *)
let cut_cases =
  let c =
    {
      cd = F32;
      nb = 1;
      axes = 0;
      hk = 1;
      g = 1;
      tq = 40;
      ck = 124;
      cdk = 24;
      cdv = 20;
      page = 0;
      lens = None;
      mask = Causal;
      rescore = No_score;
      lay = Own;
      bounded = false;
      poisoned = false;
      keep = Keep_all;
      seed = 11;
    }
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
    ( "bfloat16 documents on the matrix units",
      { c with cd = Bf16; g = 3; mask = Causal_docs } );
  ]

(* Calls of one tile of rows over the keys, so that a causal or window mask
   hides whole blocks of a tile, shows others whole and cuts the rest: of one
   head and of several, over dense and paged keys, masks by coordinates and by
   loads, lengths short of the keys with loads poisoned past them, rows short of
   a tile, and a score through a load. *)
let hidden_cases =
  let c =
    {
      cd = F32;
      nb = 1;
      axes = 0;
      hk = 1;
      g = 1;
      tq = 300;
      ck = 300;
      cdk = 24;
      cdv = 20;
      page = 0;
      lens = None;
      mask = Causal;
      rescore = No_score;
      lay = Own;
      bounded = true;
      poisoned = true;
      keep = Keep_all;
      seed = 13;
    }
  in
  [
    ("causal rows of one head", c);
    ("causal rows of three heads", { c with g = 3; tq = 200; ck = 200 });
    ("a window", { c with mask = Window });
    ( "a window over paged keys",
      { c with mask = Window; page = 16; tq = 304; ck = 304 } );
    ("causal documents", { c with mask = Causal_docs });
    ( "a window and a scaled bias, lengths short of the keys",
      {
        c with
        nb = 2;
        mask = Window;
        rescore = Scaled_bias;
        lens = Some [| 300; 170 |];
      } );
    ( "a scaled bias at rows short of a tile, a length short of the keys",
      { c with tq = 37; rescore = Scaled_bias; lens = Some [| 250 |] } );
    ("a bfloat16 window", { c with cd = Bf16; mask = Window });
  ]

let cut_case (cuda : Support.backend) (_, c) =
  if c.cd = F16 || c.cd = Bf16 then near_cpu cuda c
  else same_floats (run_call Support.cpu c) (run_call cuda c)

(* Programs of one mask

   nx_cuda.mli: a call's association depends on its dtypes and extents alone, so
   programs that hide the same pairs give the same bits, however much shared
   memory their values take. *)

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
  Format.fprintf ppf "g %d, tq %d, tk %d, dk %d, dv %d, %d values held, seed %d"
    h.hg h.htq h.htk h.hdk h.hdv h.hn h.hseed

(* float32 calls of one key head whose keys go in parts, their rows past one
   tile of 16 and their causal masks through 2 to 48 values held at once.
   nx_cuda.mli: of [b ≥ 8] blocks of 64 keys, its query tiles being [g ≤ 2], the
   keys go in [⌈b / ⌈b / P'⌉⌉ ≥ 2] parts, [P' = min 128 ⌊b / 4⌋]; 512 to 2048
   keys make 2 to 8 parts, a last block cut short among them. A split program's
   temporaries grow with the pairs, so the draws hold no more keys than the
   parts need. *)
let helds =
  let open Gen in
  with_pp pp_held
  @@ let* hg = int_range 1 2 in
     let* htq = int_range 65 128 in
     let* htk = int_range 512 2048 in
     let* hdk = int_range 1 16 in
     let* hdv = int_range 1 16 in
     let* hn = int_range 2 48 in
     let+ hseed = int_range 0 0x3fffffff in
     { hg; htq; htk; hdk; hdv; hn; hseed }

(* The causal mask of [h]'s rows over its keys through [n] int64 values held at
   once: [key + i <= row + tk - tq + i] for every [i < n]. Every visible pair is
   kept. *)
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
  let keep = add (P.Const (D.Any D.Bool, P.bits D.Bool true)) in
  P.v ~ins:(Iarray.of_list [ f32 ])
    (Iarray.of_list (List.rev !nodes))
    ~outs:(Iarray.of_list [ o0; score; keep ])

(* Fresh o and lse for [h]'s calls on [b], and [h]'s call on [b] by its mask
   through [n] values held into them, on finite q, k and v drawn from [h.hseed]:
   o and lse. The call must answer [Done]. *)
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
    equal ~msg:(b.name ^ ": answer") Kernels_cases.answer A.Done
      (K.attention p ~scale:0.375 ~o ~lse q (Nx_kernel.Dense { k; v }) [||]);
    [| A.Any o; A.Any lse |]
  in
  (outputs, call)

(* [h]'s call on [b] by its mask through one value held, then through [h.hn], on
   the same q, k and v: o and lse of each. *)
let held_calls (b : Support.backend) h =
  let outputs, call = held_call b h in
  b.around (fun () -> (call (outputs ()) 1, call (outputs ()) h.hn))

let law_held (b : Support.backend) h =
  let one, many = held_calls b h in
  same_floats one many

(* The memory of calls whose program is split

   nx_cuda.mli: such a call takes a temporary for each node [map] computes, an
   element of the node's dtype at each index of the loop's axes along which the
   node's leaves vary, and [P r (dv + 2)] accumulator elements for its parts, [P
   <= 256]; its memory returns once its work is done. *)

let held_large =
  { hg = 2; htq = 128; htk = 16384; hdk = 16; hdv = 16; hn = 48; hseed = 1 }

(* The bytes [d] counts of what nx_cuda.mli states [h]'s call takes at most,
   each temporary a buffer of the call's own and the parts' states another, in
   the driver's units ({!Rig.allocation}): the int64 keys [key + i] and rows
   [row + tk - tq + i], their comparisons and the conjunctions of these, and the
   parts' states. *)
let held_bytes d h =
  let n = h.hn and pairs = h.htq * h.htk and a = Rig.allocation d in
  (n * a (8 * h.htk))
  + ((n + 1) * a (8 * h.htq))
  + (((2 * n) - 1) * a pairs)
  + a (256 * h.hg * h.htq * (h.hdv + 2) * 4)

let test_held_loop (b : Support.backend) () =
  let h = held_large in
  let outputs, call = held_call b h in
  let outs = outputs () in
  Kernels_cases.loop_held b ~calls:8 ~bound:(held_bytes b.device h) (fun () ->
      Array.iter
        (fun (A.Any x) -> Rig.Buffer.wait (A.buffer x) Read_write)
        (call outs h.hn))

(* [f] on [d]'s backend, opened in the test. *)
let on_opened (d : Support.declared) f x = f (Support.opened d) x

let gpu_laws (d : Support.declared) =
  let on f = on_opened d f in
  group (d.label ^ " against cpu")
    [
      prop "a mask through more values held keeps the bits" helds (on law_held);
      test "calls of a split mask hold one call's memory"
        (on (fun b () -> test_held_loop b ()));
      prop "fused calls give nx.cpu's bits" (calls ~fused:true) (on law_fused);
      prop "matrix units keep nx.cpu's classes and come near its values" units
        (on law_units);
      cases
        ~id:(fun (msg, _, _) -> msg)
        "matrix units sum as nx_cuda.mli states" units_cases (on units_case);
      cases ~id:fst "programs past a launch compute" past_cases (on past_case);
      cases ~id:fst "blocks the diagonal cuts" cut_cases (on cut_case);
    ]

(* nx.metal

   nx_metal.mli: a call has nx.cpu's bits at every dtype it takes, but where a
   value is subnormal, which it flushes to a zero of its sign; a float16 value
   is never subnormal in float32. *)

(* Calls of float16, bfloat16 or float32 in a float32 accumulator, drawn so that
   no step of either library's sums is subnormal: elements 0 or past [2{^-20}],
   scores within [±8], and outputs [1] of a row within 70 of each other, where a
   rescore by distance is drawn over at most 200 keys. Below 449 keys, of every
   path: key heads of 8 rows or fewer and of more, dimensions and columns past
   128, 4 past a multiple of 8 and none, no rows, lengths at or below 0. Loads
   are poisoned past the length (nx_kernel.mli: a key past its length is never
   read). With [~long], of 449 keys or more and few rows. *)
let metal_calls ~long =
  let open Gen in
  with_pp pp_call
  @@ let* cd = of_list [ F32; F16; Bf16 ] in
     let* nb = int_range 1 2 in
     let* axes = frequency [ (3, constant 0); (1, constant 28) ] in
     let* hk = int_range 1 2 in
     let* g = of_list [ 1; 2; 3; 4; 8 ] in
     let* tq =
       if long then int_range 1 3
       else
         frequency [ (1, constant 0); (4, int_range 1 8); (4, int_range 9 40) ]
     in
     let* page = frequency [ (2, constant 0); (1, of_list [ 1; 7; 16 ]) ] in
     let* mask =
       frequency
         [
           (4, constant No_mask);
           (6, of_list [ Causal; Docs; Causal_docs; Docs32; Wide; Many; Window ]);
         ]
     in
     let* rescore =
       of_list
         (if long then
            [ No_score; Bias; Scaled_bias; Sine; Half_bias; Wide_score ]
          else
            [
              No_score;
              Bias;
              Scaled_bias;
              Alibi;
              Sine;
              Half_bias;
              Wide_bias;
              Wide_score;
            ])
     in
     let by_distance = rescore = Alibi || rescore = Wide_bias in
     let* tk =
       if long then int_range 449 1600
       else if by_distance then int_range 1 200
       else frequency [ (3, int_range 1 200); (1, int_range 201 448) ]
     in
     let ck = if page = 0 then tk else (tk + page - 1) / page * page in
     let dims =
       frequency
         [
           (1, constant 0);
           (4, int_range 1 40);
           (2, int_range 41 128);
           (1, int_range 129 140);
           (1, map (fun k -> (8 * k) + 4) (int_range 0 15));
         ]
     in
     let* cdk = dims in
     let* cdv = dims in
     let lengths lo hi =
       map Option.some (array ~size:(constant nb) (int_range lo hi))
     in
     let* lens =
       frequency
         [ (2, constant None); (2, lengths (-2) (ck + 2)); (1, lengths (-2) 0) ]
     in
     let* lay =
       frequency
         [
           (2, constant Own);
           (1, constant Shifted);
           (1, constant Swapped);
           (1, constant Flipped);
         ]
     in
     let+ seed = int_range 0 0x3fffffff in
     {
       cd;
       nb;
       axes;
       hk;
       g;
       tq;
       ck;
       cdk;
       cdv;
       page;
       lens;
       mask;
       rescore;
       lay;
       bounded = true;
       poisoned = true;
       keep = Keep_all;
       seed;
     }

(* The keys each lead index reads: its length within [0, ck], else ck. *)
let counts c =
  match c.lens with
  | Some l -> Array.map (fun x -> max 0 (min x c.ck)) l
  | None -> [| c.ck |]

(* Calls whose keys end a block (64), pass one by a key (64 j + 1) or end in a
   partial block after whole ones, unmasked, masked and through a length, so
   that every run reaches each block end; and whole blocks of keys stored last
   first. *)
let block_ends =
  let c =
    {
      cd = F32;
      nb = 1;
      axes = 0;
      hk = 1;
      g = 4;
      tq = 8;
      ck = 64;
      cdk = 64;
      cdv = 64;
      page = 0;
      lens = None;
      mask = No_mask;
      rescore = No_score;
      lay = Own;
      bounded = true;
      poisoned = true;
      keep = Keep_all;
      seed = 17;
    }
  in
  [
    c;
    { c with ck = 129 };
    { c with ck = 192; lay = Flipped };
    {
      c with
      cd = Bf16;
      g = 1;
      tq = 40;
      ck = 200;
      mask = Causal;
      lens = Some [| 129 |];
    };
    {
      c with
      cd = F16;
      ck = 192;
      page = 16;
      mask = Docs;
      rescore = Bias;
      lens = Some [| 65 |];
    };
  ]

(* Whether a lead index's length stops short of the keys and past none. *)
let short c =
  match c.lens with
  | Some l -> Array.exists (fun x -> x > 0 && x < c.ck) l
  | None -> false

(* nx_kernel.mli: a key past its length is never read. A call whose loads are
   poisoned past its lengths has the bits of the same call unpoisoned, on
   [metal] alone. *)
let unread (metal : Support.backend) c =
  if c.poisoned && c.lens <> None then
    same_floats ~msg:"poisoned against clean: "
      (run_call metal { c with poisoned = false })
      (run_call metal c)

let law_metal_bits (metal : Support.backend) c =
  cover "a key head's rows, 8 or fewer" (c.tq > 0 && c.g * c.tq <= 8);
  cover "a key head's rows, past 8" (c.g * c.tq > 8);
  cover "pages of one slot" (c.page = 1);
  cover "pages of 7 or 16 slots" (c.page > 1);
  cover "keys past 200" (c.ck > 200);
  cover "dimensions 65 to 128" (c.cdk > 64 && c.cdk <= 128);
  cover "dimensions past 128, the last chunk partial"
    (c.cdk > 128 && c.cdk mod 128 <> 0);
  cover "columns past 128" (c.cdv > 128);
  cover "float32 past 64 dimensions or columns"
    (c.cd = F32 && (c.cdk > 64 || c.cdv > 64));
  cover "float32 within 64 dimensions and columns"
    (c.cd = F32 && c.cdk <= 64 && c.cdv <= 64);
  cover "no dimension" (c.cdk = 0);
  cover "no column" (c.cdv = 0);
  cover "no row" (c.tq = 0);
  cover "a length at or below 0"
    (match c.lens with
    | Some l -> Array.exists (fun x -> x <= 0) l
    | None -> false);
  cover "masked by loads" (c.mask = Docs || c.mask = Causal_docs);
  cover "a mask past a launch's planes" (c.mask = Wide || c.mask = Many);
  cover "neither paged nor masked, every key"
    (c.page = 0 && c.mask = No_mask && c.lens = None);
  cover "neither paged nor masked, a length short of the keys"
    (c.page = 0 && c.mask = No_mask && short c);
  cover "paged, a length short of the keys" (c.page > 0 && short c);
  cover "masked, a length short of the keys" (c.mask <> No_mask && short c);
  cover "rescored" (c.rescore <> No_score);
  cover "a score through a load and a constant" (c.rescore = Scaled_bias);
  cover "loads poisoned past a length short of the keys"
    (short c
    && (List.mem c.rescore [ Bias; Scaled_bias; Half_bias ]
       || List.mem c.mask [ Docs; Causal_docs; Docs32; Many ]));
  cover "32 axes" (c.axes > 0);
  cover "rows of 4 elements a vector"
    (c.lay = Own && c.cdk mod 4 = 0 && c.cdk > 0);
  cover "rows of vectors, the last 8 dimensions half past them"
    (c.lay = Own && c.cdk mod 8 = 4);
  cover "rows one element into their buffer" (c.lay = Shifted);
  cover "rows stored the other way" (c.lay = Swapped);
  cover "rows stored last first" (c.lay = Flipped);
  let n = counts c in
  cover "keys of one whole block" (Array.mem 64 n);
  cover "keys of 64 j + 1" (Array.exists (fun x -> x mod 64 = 1) n);
  cover "a last partial block after whole ones"
    (Array.exists (fun x -> x > 64 && x mod 64 <> 0) n);
  same_floats (run_call Support.cpu c) (run_call metal c);
  unread metal c

(* nx_metal.mli's bound_offsets: a call has the same bits whether it indexes its
   keys and values in 32 bits, as every call of a test's sizes does, or in
   64. *)
let law_metal_offsets (metal : Support.backend) c =
  cover "paged" (c.page > 0);
  cover "masked" (c.mask <> No_mask);
  cover "neither paged nor masked" (c.page = 0 && c.mask = No_mask);
  cover "rows stored the other way" (c.lay = Swapped);
  cover "rows stored last first" (c.lay = Flipped);
  let wide =
    Fun.protect
      ~finally:(fun () -> Nx_metal.bound_offsets (1 lsl 31))
      (fun () ->
        Nx_metal.bound_offsets 0;
        run_call metal c)
  in
  same_floats ~msg:"64-bit offsets against 32: " wide (run_call metal c)

let law_metal_long (metal : Support.backend) c =
  cover "keys past 1024" (c.ck > 1024);
  cover "paged" (c.page > 0);
  cover "masked" (c.mask <> No_mask);
  same_floats (run_call Support.cpu c) (run_call metal c);
  unread metal c

(* One key head's [rows] rows over [tk] keys of one dimension and two columns,
   [dt] in a float32 accumulator, at scale 1 and a zero query, so that each
   output [1] is its pair's bias: key [j] is visible where [seen j], with the
   bias [bias j] and the values [value j], and every visible pair is kept. Its o
   and lse; the call must answer [Done]. *)
let exact_call (b : Support.backend) d ~rows ~tk ~seen ~bias ~value =
  let go (type r) (dt : (float, r) D.t) =
    let (module K : Nx_kernel.S) = b.kernels in
    let make dt shape xs =
      Kernels_cases.on b (A.of_array dt (Iarray.of_array shape) xs)
    in
    let per_pair f = Array.init (rows * tk) (fun i -> f (i mod tk)) in
    let q = make dt [| 1; rows; 1 |] (Array.make rows 0.) in
    let k = make dt [| 1; tk; 1 |] (Array.make tk 1.) in
    let v =
      make dt [| 1; tk; 2 |]
        (Array.init (2 * tk) (fun i -> (value (i / 2)).(i mod 2)))
    in
    let s = D.Any D.Float32 in
    let p =
      P.v ~ins:[| s; s; D.Any D.Bool |]
        [|
          In 0;
          In 1;
          Op2 (Binary Add, 0, 1);
          In 2;
          Const (D.Any D.Bool, P.bits D.Bool true);
        |]
        ~outs:[| 3; 2; 4 |]
    in
    let o = A.create b.device D.Float32 [| 1; rows; 2 |] in
    let lse = A.create b.device D.Float32 [| 1; rows |] in
    let loads =
      [|
        A.Any (make D.Float32 [| 1; rows; tk |] (per_pair bias));
        A.Any (make D.Bool [| 1; rows; tk |] (per_pair seen));
      |]
    in
    equal ~msg:(b.name ^ ": answer") Kernels_cases.answer A.Done
      (K.attention p ~scale:1. ~o ~lse q
         (Nx_kernel.Dense { k; v })
         (Iarray.of_array loads));
    [| A.Any o; A.Any lse |]
  in
  b.around (fun () ->
      match d with
      | F16 -> go D.Float16
      | Bf16 -> go D.Bfloat16
      | F32 | F64 | F32_in_f64 -> go D.Float32)

type exact = {
  name : string;
  etk : int;
  seen : int -> bool;
  bias : int -> float;
  value : int -> float array;
}

(* Small integers, exact in every dtype, which sum exactly. *)
let small j = [| Float.of_int ((j mod 5) - 2); Float.of_int (j mod 3) |]

(* Calls of 512 keys, 8 blocks, for a key head of 1 or 17 rows, whose weights
   are 1 or 0 and whose values are small integers, so that every sum is exact
   and [o] and [lse] have nx.cpu's bits: a block of no visible pair, whose
   maximum is [-∞]; blocks of only [-∞] outputs before finite ones; and such a
   block between finite ones. *)
let exact_blocks =
  let all _ = true and zero _ = 0. in
  [
    {
      name = "a block of no visible pair";
      etk = 512;
      seen = (fun j -> j / 64 <> 3);
      bias = zero;
      value = small;
    };
    {
      name = "blocks of only -inf outputs before finite ones";
      etk = 512;
      seen = all;
      bias = (fun j -> if j < 128 then Float.neg_infinity else 0.);
      value = small;
    };
    {
      name = "a block of only -inf outputs between finite ones";
      etk = 512;
      seen = all;
      bias = (fun j -> if j / 64 = 4 then Float.neg_infinity else 0.);
      value = small;
    };
  ]

(* Calls of one block whose weights' exponents [x - m'] reach exp's near range's
   end, [-86], and pass it: a least exponent of exactly [-86] and of one ulp
   below; a score of [+∞] among finite ones, whose exponents are [-∞] and a NaN;
   a NaN score; and hidden pairs beside a visible one past the range, of a
   normal weight. *)
let near_ends =
  let all _ = true in
  let at j x i = if i = j then x else 0. in
  let below = Int32.float_of_bits (Int32.add (Int32.bits_of_float (-86.)) 1l) in
  [
    {
      name = "a least exponent of -86";
      etk = 64;
      seen = all;
      bias = at 13 (-86.);
      value = small;
    };
    {
      name = "a least exponent one ulp below -86";
      etk = 64;
      seen = all;
      bias = at 13 below;
      value = small;
    };
    {
      name = "a score of +inf among finite ones";
      etk = 64;
      seen = all;
      bias = at 13 Float.infinity;
      value = small;
    };
    {
      name = "a NaN score";
      etk = 64;
      seen = all;
      bias = at 13 Float.nan;
      value = small;
    };
    {
      name = "hidden pairs beside a visible one past the range";
      etk = 64;
      seen = (fun j -> j mod 3 <> 0);
      bias = at 14 (-87.);
      value = small;
    };
  ]

(* Calls whose block holds a NaN or an infinite value in column 0 of one key,
   the rest finite, at weights and values that round: the keys staged with it
   sum one product at a time, the others by the matrix units, and both keep
   nx.cpu's bits, column 1's included. A value at key 3, 20 or 40 of one block,
   and at key 150 of the third of four. *)
let odd_spans =
  let all _ = true in
  let bias j = -0.37 *. Float.of_int (j mod 7) in
  let odd k x j =
    if j = k then [| x; 0.7 |]
    else
      [| 0.3 *. Float.of_int ((j mod 5) - 2); 0.7 *. Float.of_int (j mod 3) |]
  in
  [
    {
      name = "a NaN value at key 3";
      etk = 64;
      seen = all;
      bias;
      value = odd 3 Float.nan;
    };
    {
      name = "a NaN value at key 20";
      etk = 64;
      seen = all;
      bias;
      value = odd 20 Float.nan;
    };
    {
      name = "an infinite value at key 40";
      etk = 64;
      seen = all;
      bias;
      value = odd 40 Float.infinity;
    };
    {
      name = "a NaN value at key 150 of 256";
      etk = 256;
      seen = all;
      bias;
      value = odd 150 Float.nan;
    };
  ]

(* Each dtype nx.metal takes, at 1 and 17 rows, and their name. *)
let each_metal f =
  List.iter
    (fun rows ->
      List.iter
        (fun d -> f (Format.asprintf "%a, %d rows: " pp_dtypes d rows) d rows)
        [ F32; F16; Bf16 ])
    [ 1; 17 ]

let exact_block (metal : Support.backend) e =
  each_metal (fun msg d rows ->
      let call b =
        exact_call b d ~rows ~tk:e.etk ~seen:e.seen ~bias:e.bias ~value:e.value
      in
      same_floats ~msg (call Support.cpu) (call metal))

let floats_of (A.Any x) : float array =
  let x = Kernels_cases.host x in
  match A.dtype x with
  | D.Float32 -> A.to_array x
  | _ -> failwith "floats_of: not float32"

(* nx_metal.mli: a block's weights sum in nx.cpu's lanes, weight [j] in lane [j
   mod 8], then lanes [+4], [+2], [+1]. One block: weight 1 at key 0, [exp
   (-17)], under half an ulp of 1, at keys 1 to 15 but 8, and 0 elsewhere. Every
   sequential order, and the sum of the values that [o]'s chain adds, rounds
   each small weight away; the lanes sum them before they meet 1. So [o] is not
   [1], and has nx.cpu's bits. *)
let test_metal_lanes (metal : Support.backend) () =
  let bias j =
    if j = 0 then 0. else if j < 16 && j <> 8 then -17. else Float.neg_infinity
  in
  each_metal (fun msg d rows ->
      let call b =
        exact_call b d ~rows ~tk:64
          ~seen:(fun _ -> true)
          ~bias
          ~value:(fun _ -> [| 1.; 1. |])
      in
      let want = call Support.cpu in
      if (floats_of want.(0)).(0) = 1. then
        failf "%snx.cpu's o is 1: the case tells no order apart" msg;
      same_floats ~msg want (call metal))

(* nx_metal.mli's subnormal numbers: a weight below [2{^-126}] applies as [+0],
   a bfloat16 subnormal value reads as a zero, a float16 one is kept. Each at 1
   and 17 rows: o and lse. *)
let test_metal_subnormal (metal : Support.backend) () =
  let check msg d ~tk ~bias ~value ~o ~lse =
    List.iter
      (fun rows ->
        let got =
          exact_call metal d ~rows ~tk ~seen:(fun _ -> true) ~bias ~value
        in
        let msg = Printf.sprintf "%s, %d rows" msg rows in
        equal ~msg:(msg ^ ": answers") int 2 (Array.length got);
        equal ~msg:(msg ^ ": o") (array float_exact)
          (Array.concat (List.init rows (fun _ -> o)))
          (floats_of got.(0));
        equal ~msg:(msg ^ ": lse") (array float_exact) (Array.make rows lse)
          (floats_of got.(1)))
      [ 1; 17 ]
  in
  check "a weight of exp (-100)" F32 ~tk:2
    ~bias:(fun j -> if j = 0 then 0. else -100.)
    ~value:(fun j -> if j = 0 then [| 0.; 0. |] else [| 0x1p100; -0x1p100 |])
    ~o:[| 0.; 0. |] ~lse:0.;
  check "a bfloat16 subnormal value" Bf16 ~tk:1
    ~bias:(fun _ -> 0.)
    ~value:(fun _ -> [| 0x1p-130; -0x1p-130 |])
    ~o:[| 0.; 0. |] ~lse:0.;
  check "a float16 subnormal value" F16 ~tk:1
    ~bias:(fun _ -> 0.)
    ~value:(fun _ -> [| 0x1p-20; -0x1p-24 |])
    ~o:[| 0x1p-20; -0x1p-24 |] ~lse:0.

(* nx_metal.mli: a pair program whose output 2 is not the constant [true]
   declines. *)
let law_metal_keep (metal : Support.backend) c =
  cover "a keep by coordinates" (c.keep = Keep_coords);
  cover "a keep by a load" (c.keep = Keep_load);
  equal ~msg:"answer" Kernels_cases.answer A.Declined (fst (call_on metal c))

let metal_keeps =
  let open Gen in
  with_pp pp_call
  @@ let* c = metal_calls ~long:false in
     let+ keep = of_list [ Keep_coords; Keep_load ] in
     { c with keep }

(* A float64 accumulator declines (nx_metal.mli). *)
let test_metal_float64 (metal : Support.backend) () =
  let c = { (List.assoc "rows of one head" cut_cases) with tq = 3 } in
  List.iter
    (fun cd ->
      equal
        ~msg:(Format.asprintf "%a" pp_dtypes cd)
        int 0
        (Array.length (run_call metal { c with cd })))
    [ F64; F32_in_f64 ]

(* The bits of a call's results, NaNs' included. *)
let bits results =
  String.concat " "
    (Array.to_list
       (Array.concat
          (Array.to_list
             (Array.map
                (fun (A.Any x) ->
                  let x = Kernels_cases.host x in
                  match A.dtype x with
                  | D.Float32 ->
                      Array.map
                        (fun f -> Printf.sprintf "%lx" (Int32.bits_of_float f))
                        (A.to_array x)
                  | _ -> failwith "bits: not float32")
                results))))

(* Determinism: calls on two domains, of a key head of 8 rows, of 40 rows and of
   one row over 1000 keys, each give the bits it gives alone. *)
let law_metal_alone (metal : unit -> Support.backend) =
  let base = List.assoc "rows of one head" cut_cases in
  let calls =
    [| { base with g = 4; tq = 2 }; base; { base with tq = 1; ck = 1000 } |]
  in
  let alone = Array.make (Array.length calls) None in
  let lock = Mutex.create () in
  let reference i =
    Mutex.protect lock (fun () ->
        match alone.(i) with
        | Some b -> b
        | None ->
            let b = bits (run_call (metal ()) calls.(i)) in
            alone.(i) <- Some b;
            b)
  in
  let which =
    Gen.with_pp Format.pp_print_int (Gen.int_range 0 (Array.length calls - 1))
  in
  stateful ~count:3 ~steps:6 ~domains:2
    "calls on two domains have the bits each has alone"
    [
      command "call"
        (which @-> returns string)
        reference
        (fun i -> bits (run_call (metal ()) calls.(i)));
    ]

let metal_case (metal : Support.backend) (_, c) =
  same_floats (run_call Support.cpu c) (run_call metal c);
  unread metal c

(* The cases nx.metal computes: nx_metal.mli has it decline a float64
   accumulator and a keep but the constant true, which "a float64 accumulator
   declines" and "a keep but the constant true declines" state. *)
let metal_computes cs =
  List.filter
    (fun (_, c) -> c.cd <> F64 && c.cd <> F32_in_f64 && c.keep = Keep_all)
    cs

(* nx_metal.mli's time budget: a call past 10 s runs as several submissions in
   order. With parts bounded to one launch, a call of several launches, its
   temporaries and stored scores read by later launches, takes several parts and
   keeps one part's bits. *)
let metal_parted (metal : Support.backend) (_, c) =
  let module Plan = Nx_metal.Plan in
  Plan.bound_parts ~launches:1 ~seconds:Plan.part_budget;
  let got, launches =
    Fun.protect
      ~finally:(fun () ->
        Plan.bound_parts ~launches:Plan.part_launches ~seconds:Plan.part_budget)
      (fun () -> Plan.traced (fun () -> run_call metal c))
  in
  at_least ~msg:"launches, a part each" int ~than:2 (List.length launches);
  same_floats (run_call Support.cpu c) got;
  unread metal c

let metal_laws (d : Support.declared) =
  let on f = on_opened d f in
  group (d.label ^ " against cpu")
    [
      prop "calls give nx.cpu's bits" ~examples:block_ends
        (metal_calls ~long:false) (on law_metal_bits);
      prop "calls past 448 keys give nx.cpu's bits" (metal_calls ~long:true)
        (on law_metal_long);
      prop "calls index keys and values in 32 or 64 bits alike"
        ~examples:block_ends (metal_calls ~long:false) (on law_metal_offsets);
      cases
        ~id:(fun e -> e.name)
        "blocks of -inf outputs keep nx.cpu's bits" exact_blocks
        (on exact_block);
      cases
        ~id:(fun e -> e.name)
        "weights at exp's near range's end keep nx.cpu's bits" near_ends
        (on exact_block);
      cases
        ~id:(fun e -> e.name)
        "a value not finite among a block's keys keeps nx.cpu's bits" odd_spans
        (on exact_block);
      test "a block's weights sum in nx.cpu's lanes"
        (on (fun b () -> test_metal_lanes b ()));
      test "subnormal weights and values flush as nx_metal.mli states"
        (on (fun b () -> test_metal_subnormal b ()));
      test "a float64 accumulator declines"
        (on (fun b () -> test_metal_float64 b ()));
      prop "a keep but the constant true declines" metal_keeps
        (on law_metal_keep);
      prop "a mask through more values held keeps the bits" helds (on law_held);
      cases ~id:fst "programs past a launch compute"
        (metal_computes past_cases)
        (on metal_case);
      cases ~id:fst "programs past a launch compute a launch a part"
        (metal_computes past_cases)
        (on metal_parted);
      cases ~id:fst "blocks the diagonal cuts" (metal_computes cut_cases)
        (on metal_case);
      cases ~id:fst "blocks a mask hides whole" hidden_cases (on metal_case);
      law_metal_alone (fun () -> Support.opened d);
    ]

let nv_laws (first : Support.declared) (nv : Support.declared) =
  group nv.label
    [
      prop "bits are cuda's, any NaN for a NaN" (calls ~fused:false) (fun c ->
          law_nv (Support.opened first) (Support.opened nv) c);
      prop "a mask through more values held keeps the bits" helds
        (on_opened nv law_held);
    ]

let () =
  let nv =
    match (Support.declared_cuda, Support.declared_nv) with
    | Some first, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  let declared = List.filter_map Fun.id in
  exit
    (Windtrap.run "nx_kernel.attention"
       (List.map suite
          (Support.declare Support.cpu
          :: declared [ Support.declared_cuda; Support.declared_metal ])
       @ List.map gpu_laws (declared [ Support.declared_cuda ])
       @ nv
       @ List.map metal_laws (declared [ Support.declared_metal ])))
