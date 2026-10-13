(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Factorisations and triangular solves through every kernel library the host
   runs: each result against its defining relation, within the bound
   nx_cpu.mli states, by residuals computed in double-double, on operands
   well and badly conditioned; LU's pivots as nx_kernel.mli's lu chooses
   them; a failing matrix NaN alone in its batch; only the triangle a routine names read; the
   same bits under every layout of the same values. nx.cpu's group checks
   that it computes what nx_cpu.mli lists, the same bits on one thread as on
   the job's, and its refusals. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move
module Support = Nx_kernels_support

let pp_dtype ppf (D.Any dt) = D.pp ppf dt
let dtypes = D.[ Any Float32; Any Float64; Any Complex64; Any Complex128 ]
let is_complex (D.Any dt) = D.is D.Complex dt

(* The dtype's unit roundoff. *)
let unit_roundoff (D.Any dt) =
  match dt with D.Float32 | D.Complex64 -> 0x1p-24 | _ -> 0x1p-53

(* γ(k) = k u / (1 − k u), for [k] operations, twice as many on complex
   numbers, as nx_cpu.mli states the bounds. *)
let gamma dt k =
  let k = if is_complex dt then 2 * k else k in
  let ku = float_of_int k *. unit_roundoff dt in
  ku /. (1. -. ku)

(* Values *)

(* An array of [dt] and [shape] holding [v], each value rounded to [dt]; a
   real dtype keeps the real parts. *)
let of_values (D.Any dt) shape (v : Complex.t array) =
  let re = Array.map (fun (c : Complex.t) -> c.re) v in
  match dt with
  | D.Float32 -> A.Any (A.of_array D.Float32 shape re)
  | D.Float64 -> A.Any (A.of_array D.Float64 shape re)
  | D.Complex64 -> A.Any (A.of_array D.Complex64 shape v)
  | D.Complex128 -> A.Any (A.of_array D.Complex128 shape v)
  | _ -> invalid_arg "of_values"

let values (A.Any x) : Complex.t array =
  let re (v : float array) = Array.map (fun re -> { Complex.re; im = 0. }) v in
  match A.dtype x with
  | D.Float32 -> re (A.to_array x)
  | D.Float64 -> re (A.to_array x)
  | D.Complex64 -> A.to_array x
  | D.Complex128 -> A.to_array x
  | _ -> invalid_arg "values"

(* The bits of each value, NaN payloads and signs of zero included. *)
let bits x =
  Array.map
    (fun (c : Complex.t) -> (Int64.bits_of_float c.re, Int64.bits_of_float c.im))
    (values x)

let same_bits = array (pair int64 int64)

type how = Contiguous | Transposed | Stepped

let pp_how ppf h =
  Format.pp_print_string ppf
    (match h with
    | Contiguous -> "contiguous"
    | Transposed -> "transposed"
    | Stepped -> "stepped")

(* [x]'s values laid out another way: C-contiguous, a copy whose last two
   axes are swapped in memory viewed back through a permutation, or every
   other element of a copy twice as wide. *)
let laid_out how (A.Any x) =
  let s = L.shape (A.layout x) in
  let r = Array.length s in
  let into y =
    A.blit ~src:x ~dst:y;
    A.Any y
  in
  match how with
  | Contiguous -> A.Any x
  | Transposed ->
      let swap i = if i = r - 2 then r - 1 else if i = r - 1 then r - 2 else i in
      let perm = Array.init r swap in
      let t = A.create Rig.host (A.dtype x) (Array.map (fun i -> s.(i)) perm) in
      into (Option.get (A.move (M.Permute perm) t))
  | Stepped ->
      let wide = Array.mapi (fun i d -> if i = r - 1 then 2 * d else d) s in
      let w = A.create Rig.host (A.dtype x) wide in
      let range i d = { M.start = 0; count = d; step = (if i = r - 1 then 2 else 1) } in
      into (Option.get (A.move (M.Slice (Array.mapi range s)) w))

(* Double-double sums *)

(* A sum held as an unevaluated pair, exact to about 2^-106 of its terms'
   magnitudes: products by TwoProd, additions by TwoSum. *)
type dd = { mutable hi : float; mutable lo : float }

let dd () = { hi = 0.; lo = 0. }

let add s x =
  let t = s.hi +. x in
  let e =
    if Float.abs s.hi >= Float.abs x then s.hi -. t +. x else x -. t +. s.hi
  in
  s.hi <- t;
  s.lo <- s.lo +. e

let add_prod s a b =
  let p = a *. b in
  add s p;
  s.lo <- s.lo +. Float.fma a b (-.p)

(* Σ x·y over complex pairs, y conjugated under [conj]: its real and
   imaginary parts. *)
type sum = { re : dd; im : dd }

let sum () = { re = dd (); im = dd () }

let add_cprod ?(conj = false) s (x : Complex.t) (y : Complex.t) =
  let yi = if conj then -.y.im else y.im in
  add_prod s.re x.re y.re;
  add_prod s.re (-.x.im) yi;
  add_prod s.im x.re yi;
  add_prod s.im x.im y.re

let total s = { Complex.re = s.re.hi +. s.re.lo; im = s.im.hi +. s.im.lo }

(* Draws *)

let uniform r = Random.State.float r 2. -. 1.

let draw r dt =
  { Complex.re = uniform r; im = (if is_complex dt then uniform r else 0.) }

(* Matrix orders on both sides of a block of 64, and small ones. *)
let orders = [ 0; 1; 2; 3; 5; 8; 17; 63; 64; 65; 129 ]

type case = {
  dt : D.any;
  batch : int array;
  n : int;
  r : int;  (** right-hand sides of a solve *)
  triangle : Nx_kernel.triangle;
  adjoint : bool;
  unit_diagonal : bool;
  seed : int;
  how : how;
  graded : bool;
      (** rows and columns scaled by powers of two from 2^-16 to 2^16: an
          operand of condition up to 2^64 whose exact factors are those of
          the ungraded one, scaled *)
}

let pp_case ppf c =
  Format.fprintf ppf "%a batch %a n %d r %d %s%s%s seed %d %a" pp_dtype c.dt
    pp_ints c.batch c.n c.r
    (match c.triangle with Lower -> "lower" | Upper -> "upper")
    (if c.adjoint then " adjoint" else "")
    (if c.unit_diagonal then " unit" else "")
    c.seed pp_how c.how;
  if c.graded then Format.pp_print_string ppf " graded"

let case =
  Gen.with_pp pp_case
    (let open Gen in
     let* dt = of_list ~pp:pp_dtype dtypes in
     let* n = of_list ~pp:Format.pp_print_int orders in
     let* batch =
       if n > 17 then array ~size:(int_range 0 1) (int_range 1 2)
       else array ~size:(int_range 0 2) (int_range 0 3)
     in
     let* r = of_list ~pp:Format.pp_print_int [ 0; 1; 3; 70 ] in
     let* triangle = of_list Nx_kernel.[ Lower; Upper ] in
     let* adjoint = bool in
     let* unit_diagonal = bool in
     let* seed = int_range 0 1_000_000 in
     let* how = of_list ~pp:pp_how [ Contiguous; Transposed; Stepped ] in
     let+ graded = bool in
     { dt; batch; n; r; triangle; adjoint; unit_diagonal; seed; how; graded })

let count c = Array.fold_left ( * ) 1 c.batch
let square c = Array.append c.batch [| c.n; c.n |]

(* Element (i, j) of matrix [b] of the [rows] × [cols] matrices [v]. *)
let at v ~rows ~cols b i j = v.((((b * rows) + i) * cols) + j)

(* [v], matrices of [rows] × [cols], with element (i, j) times
   2^(e_i + f_j) under [c.graded], e and f drawn from the seed in
   [-8, 8], f = e under [symmetric]. *)
let grade c ?(symmetric = false) ~rows ~cols v =
  if not c.graded then v
  else
    let r = Random.State.make [| c.seed + 2 |] in
    let e = Array.init rows (fun _ -> Random.State.int r 17 - 8) in
    let f = if symmetric then e else Array.init cols (fun _ -> Random.State.int r 17 - 8) in
    Array.mapi
      (fun k (x : Complex.t) ->
        let i = k / cols mod rows and j = k mod cols in
        let s = Float.ldexp 1. (e.(i) + f.(j)) in
        { Complex.re = x.re *. s; im = x.im *. s })
      v

(* A Hermitian positive definite matrix of order [n]: m mᴴ / n + δ I, its
   eigenvalues at least δ. *)
let definite r dt n =
  let m = Array.init (n * n) (fun _ -> draw r dt) in
  let delta = [| 1.; 0.1; 0.01 |].(Random.State.int r 3) in
  Array.init (n * n) (fun k ->
      let i = k / n and j = k mod n in
      let s = sum () in
      for p = 0 to n - 1 do
        add_cprod ~conj:true s m.((i * n) + p) m.((j * n) + p)
      done;
      let v = total s and f = float_of_int n in
      if i = j then { Complex.re = (v.re /. f) +. delta; im = 0. }
      else { Complex.re = v.re /. f; im = v.im /. f })

(* A Hermitian matrix of order [n], its lower triangle drawn. *)
let hermitian r dt n =
  let m = Array.init (n * n) (fun _ -> draw r dt) in
  Array.init (n * n) (fun k ->
      let i = k / n and j = k mod n in
      if i > j then m.(k)
      else if i = j then { (m.(k)) with im = 0. }
      else Complex.conj m.((j * n) + i))

(* A triangular matrix of order [n] that substitution solves stably: its
   diagonal of magnitude 1 to 2, the rest drawn in [-1/n, 1/n]. *)
let triangular r dt n =
  Array.init (n * n) (fun k ->
      let v = draw r dt and f = float_of_int n in
      if k / n <> k mod n then { Complex.re = v.re /. f; im = v.im /. f }
      else
        let m = 1. +. Random.State.float r 1. and t = Random.State.float r 6.3 in
        if is_complex dt then { Complex.re = m *. cos t; im = m *. sin t }
        else { Complex.re = (if v.re < 0. then -.m else m); im = 0. })

(* Running *)

let pp_answer = Nx_array_support.pp_answer
let answer = Testable.make ~pp:pp_answer ~equal:( = )

(* The routines the suite runs. *)
type routine =
  | Cholesky of Nx_kernel.triangle
  | Lu
  | Qr of Nx_kernel.factors
  | Eigh of bool  (** with vectors *)
  | Solve of {
      triangle : Nx_kernel.triangle;
      adjoint : bool;
      unit_diagonal : bool;
    }

(* [k]'s results of [r] on [ops], or [None] if it declined: Lu's positions
   int64, every other result of a's dtype, of the shapes Nx_kernel.Shape
   gives. *)
let run ?(k : (module Nx_kernel.S) option) (b : Support.backend) r ops =
  let module K = (val Option.value k ~default:b.kernels) in
  let module Shape = Nx_kernel.Shape in
  let (A.Any a) = ops.(0) in
  let dt = A.dtype a in
  let shape (A.Any x) = L.shape (A.layout x) in
  let make s = A.create Rig.host dt s in
  let index s = A.create Rig.host D.Int64 s in
  let ok = Result.get_ok in
  let dsts, answer =
    match r with
    | Cholesky t ->
        let dst = make (ok (Shape.cholesky (shape ops.(0)))) in
        ([| A.Any dst |], K.cholesky t ~dst a)
    | Lu ->
        let sl, sp, sq = ok (Shape.lu (shape ops.(0))) in
        let lu = make sl and pivots = index sp and perm = index sq in
        ([| A.Any lu; A.Any pivots; A.Any perm |], K.lu ~lu ~pivots ~perm a)
    | Qr f ->
        let sq, sr = ok (Shape.qr f (shape ops.(0))) in
        let q = make sq and r = make sr in
        ([| A.Any q; A.Any r |], K.qr ~q ~r a)
    | Eigh vectors ->
        let sw, sv = ok (Shape.eigh (shape ops.(0))) in
        let w = make sw in
        if vectors then
          let v = make sv in
          ([| A.Any w; A.Any v |], K.eigh ~w ~v a)
        else ([| A.Any w |], K.eigh ~w a)
    | Solve { triangle; adjoint; unit_diagonal } ->
        let rhs = A.expect dt ops.(1) in
        let dst =
          make (ok (Shape.solve_triangular (shape ops.(0)) (shape ops.(1))))
        in
        ( [| A.Any dst |],
          K.solve_triangular triangle ~adjoint ~unit_diagonal ~dst a rhs )
  in
  match answer with
  | A.Done -> Some dsts
  | A.Declined -> None
  | e -> failf "the kernels answered %a" pp_answer e

let solve c =
  Solve
    {
      triangle = c.triangle;
      adjoint = c.adjoint;
      unit_diagonal = c.unit_diagonal;
    }

(* The operands of a case: positive definite matrices, or triangular ones and
   right-hand sides. *)
let cholesky_operand c =
  let r = Random.State.make [| c.seed |] in
  let v = Array.concat (List.init (count c) (fun _ -> definite r c.dt c.n)) in
  of_values c.dt (square c) (grade c ~symmetric:true ~rows:c.n ~cols:c.n v)

let eigh_operand c =
  let r = Random.State.make [| c.seed |] in
  let v = Array.concat (List.init (count c) (fun _ -> hermitian r c.dt c.n)) in
  of_values c.dt (square c) (grade c ~symmetric:true ~rows:c.n ~cols:c.n v)

let solve_operands c =
  let r = Random.State.make [| c.seed |] in
  let a = Array.concat (List.init (count c) (fun _ -> triangular r c.dt c.n)) in
  let a = grade c ~rows:c.n ~cols:c.n a in
  let b = Array.init (count c * c.n * c.r) (fun _ -> draw r c.dt) in
  [| of_values c.dt (square c) a;
     of_values c.dt (Array.append c.batch [| c.n; c.r |]) b |]

(* Defining relations *)

(* The worst ratio, over the elements, of a residual to its bound, and where
   it is. *)
type worst = { mutable ratio : float; mutable where : string }

let worse w ~bound ~residual where =
  let q =
    if Float.is_nan residual then infinity
    else if bound = 0. then if residual = 0. then 0. else infinity
    else residual /. bound
  in
  if q > w.ratio then begin
    w.ratio <- q;
    w.where <- where ()
  end

let check_worst w = at_most ~msg:w.where float_exact ~than:1. w.ratio

(* L Lᴴ = a + Δ with |Δ| <= γ(n + 1) |L| |Lᴴ| over the lower triangle, the
   other triangle +0, the diagonal real and positive. *)
let law_cholesky (b : Support.backend) c =
  let x = cholesky_operand c in
  match run b (Cholesky Lower) [| laid_out c.how x |] with
  | None -> ()
  | Some y ->
      let a = values x and l = values y.(0) and n = c.n in
      let w = { ratio = 0.; where = "" } in
      let g = gamma c.dt (n + 1) in
      for m = 0 to count c - 1 do
        for i = 0 to n - 1 do
          for j = 0 to n - 1 do
            let lij = at l ~rows:n ~cols:n m i j in
            if j > i then
              equal ~msg:"the upper triangle" (pair int64 int64) (0L, 0L)
                (Int64.bits_of_float lij.re, Int64.bits_of_float lij.im)
            else if i = j then begin
              equal ~msg:"the diagonal is real" int64 0L
                (Int64.bits_of_float lij.im);
              greater ~msg:"the diagonal is positive" float_exact ~than:0. lij.re
            end
            else ();
            if j <= i then begin
              let s = sum () and bound = dd () in
              add s.re (at a ~rows:n ~cols:n m i j).re;
              if i <> j then add s.im (at a ~rows:n ~cols:n m i j).im;
              for p = 0 to j do
                let lip = at l ~rows:n ~cols:n m i p
                and ljp = at l ~rows:n ~cols:n m j p in
                add_cprod ~conj:true s (Complex.neg lip) ljp;
                add_prod bound (Complex.norm lip) (Complex.norm ljp)
              done;
              worse w ~bound:(g *. bound.hi)
                ~residual:(Complex.norm (total s))
                (fun () -> Printf.sprintf "matrix %d, element (%d, %d)" m i j)
            end
          done
        done
      done;
      check_worst w

(* The Upper factor is the Lower one's conjugate transpose, bit for bit: its
   diagonal the same, [+0] below it. *)
let law_upper (b : Support.backend) c =
  let x = cholesky_operand c in
  match (run b (Cholesky Lower) [| x |], run b (Cholesky Upper) [| x |]) with
  | Some l, Some u ->
      let n = c.n and l = values l.(0) in
      let lh =
        Array.init (Array.length l) (fun k ->
            let m = k / (n * n) and i = k / n mod n and j = k mod n in
            if j > i then Complex.conj (at l ~rows:n ~cols:n m j i)
            else if i = j then at l ~rows:n ~cols:n m i i
            else Complex.zero)
      in
      equal same_bits (bits (of_values c.dt (square c) lh)) (bits u.(0))
  | _ -> ()

(* (op(a) + Δ) x = b with |Δ| <= γ(2n) |op(a)| per column, op(a) the triangle
   read, transposed and conjugated under [adjoint], ones on its diagonal
   under [unit_diagonal]. *)
let law_solve (b : Support.backend) c =
  let ops = solve_operands c in
  match run b (solve c) [| laid_out c.how ops.(0); ops.(1) |] with
  | None -> ()
  | Some y ->
      let a = values ops.(0) and rhs = values ops.(1) and x = values y.(0) in
      let n = c.n and r = c.r in
      let upper = (c.triangle = Upper) <> c.adjoint in
      let op m i p =
        if i = p && c.unit_diagonal then Complex.one
        else if c.adjoint then Complex.conj (at a ~rows:n ~cols:n m p i)
        else at a ~rows:n ~cols:n m i p
      in
      let w = { ratio = 0.; where = "" } in
      let g = gamma c.dt (2 * n) in
      for m = 0 to count c - 1 do
        for i = 0 to n - 1 do
          for j = 0 to r - 1 do
            let s = sum () and bound = dd () in
            let bij = at rhs ~rows:n ~cols:r m i j in
            add s.re bij.re;
            add s.im bij.im;
            let lo, hi = if upper then (i, n - 1) else (0, i) in
            for p = lo to hi do
              let aip = op m i p and xpj = at x ~rows:n ~cols:r m p j in
              add_cprod s (Complex.neg aip) xpj;
              add_prod bound (Complex.norm aip) (Complex.norm xpj)
            done;
            worse w ~bound:(g *. bound.hi)
              ~residual:(Complex.norm (total s))
              (fun () -> Printf.sprintf "matrix %d, element (%d, %d)" m i j)
          done
        done
      done;
      check_worst w

(* The parts of [v] that are not NaN, the imaginary ones of complex [dt]. *)
let numbers dt v =
  Array.fold_left
    (fun k (c : Complex.t) ->
      k
      + Bool.to_int (not (Float.is_nan c.re))
      + Bool.to_int (is_complex dt && not (Float.is_nan c.im)))
    0 v

(* LU's operand: [n] rows and, by the seed, [n], [r] or [n + r] columns, a
   zero column now and then. *)
let lu_operand c =
  let cols = match c.seed mod 3 with 0 -> c.n | 1 -> c.r | _ -> c.n + c.r in
  let r = Random.State.make [| c.seed |] in
  let zero = if cols > 0 && c.seed mod 5 = 0 then c.seed mod cols else -1 in
  let v =
    Array.init (count c * c.n * cols) (fun k ->
        if k mod cols = zero then Complex.zero else draw r c.dt)
  in
  let v = grade c ~rows:c.n ~cols v in
  (cols, of_values c.dt (Array.append c.batch [| c.n; cols |]) v)

let int64s (A.Any x) =
  match A.dtype x with
  | D.Int64 -> Array.map Int64.to_int (A.to_array x)
  | _ -> invalid_arg "int64s"

(* L U = P a + Δ with |Δ| <= γ(2k) |L| |U|, L unit lower triangular and of
   magnitude at most 1 on reals, the interchanges [pivots] giving [perm],
   and no NaN from numbers. *)
let law_lu (b : Support.backend) c =
  let cols, x = lu_operand c in
  match run b Lu [| laid_out c.how x |] with
  | None -> ()
  | Some y ->
      let a = values x and lu = values y.(0) in
      let piv = int64s y.(1) and perm = int64s y.(2) in
      let m = c.n and n = cols in
      let k = min m n in
      let w = { ratio = 0.; where = "" } in
      let g = gamma c.dt (2 * k) in
      equal ~msg:"numbers" int (numbers c.dt a) (numbers c.dt lu);
      for t = 0 to count c - 1 do
        let p = Array.init m Fun.id in
        for j = 0 to k - 1 do
          let q = piv.((t * k) + j) in
          if q < j || q >= m then failf "pivot %d at step %d" q j;
          let s = p.(j) in
          p.(j) <- p.(q);
          p.(q) <- s
        done;
        equal ~msg:"perm follows pivots" (array int) p (Array.sub perm (t * m) m);
        let l i q =
          if i = q then Complex.one
          else if q < i then at lu ~rows:m ~cols:n t i q
          else Complex.zero
        in
        for i = 0 to m - 1 do
          for j = 0 to n - 1 do
            if j < i && j < k && not (is_complex c.dt) then
              at_most ~msg:"|L| <= 1" float_exact ~than:1.
                (Complex.norm (at lu ~rows:m ~cols:n t i j));
            let s = sum () and bound = dd () in
            let pa = at a ~rows:m ~cols:n t p.(i) j in
            add s.re pa.re;
            add s.im pa.im;
            for q = 0 to min (min i j) (k - 1) do
              let liq = l i q and uqj = at lu ~rows:m ~cols:n t q j in
              add_cprod s (Complex.neg liq) uqj;
              add_prod bound (Complex.norm liq) (Complex.norm uqj)
            done;
            worse w ~bound:(g *. bound.hi)
              ~residual:(Complex.norm (total s))
              (fun () -> Printf.sprintf "matrix %d, element (%d, %d)" t i j)
          done
        done
      done;
      check_worst w

let factors c = if c.seed mod 2 = 0 then Nx_kernel.Reduced else Nx_kernel.Complete

(* a = q r with, per column, ‖a_j − q r_j‖ <= γ(k(2m + 10)) ‖a_j‖; each element of
   qᴴq − I at most γ(k(2m + 10)); r +0 below its diagonal and real on it. *)
let law_qr (b : Support.backend) c =
  let cols, x = lu_operand c in
  let f = factors c in
  match run b (Qr f) [| laid_out c.how x |] with
  | None -> ()
  | Some y ->
      let a = values x and q = values y.(0) and r = values y.(1) in
      let m = c.n and n = cols in
      let qc = if f = Complete then m else min m n in
      let g = gamma c.dt (min m n * ((2 * m) + 10)) in
      let w = { ratio = 0.; where = "" } in
      for t = 0 to count c - 1 do
        for i = 0 to qc - 1 do
          for j = 0 to n - 1 do
            if j < i then
              equal ~msg:"r below its diagonal" (pair int64 int64) (0L, 0L)
                (let v = at r ~rows:qc ~cols:n t i j in
                 (Int64.bits_of_float v.re, Int64.bits_of_float v.im));
            if j = i then
              equal ~msg:"r's diagonal is real" int64 0L
                (Int64.bits_of_float (at r ~rows:qc ~cols:n t i j).im)
          done
        done;
        for j = 0 to n - 1 do
          let res = ref 0. and norm = ref 0. in
          for i = 0 to m - 1 do
            let s = sum () in
            let aij = at a ~rows:m ~cols:n t i j in
            add s.re aij.re;
            add s.im aij.im;
            for p = 0 to qc - 1 do
              add_cprod s
                (Complex.neg (at q ~rows:m ~cols:qc t i p))
                (at r ~rows:qc ~cols:n t p j)
            done;
            res := !res +. (Complex.norm2 (total s));
            norm := !norm +. Complex.norm2 aij
          done;
          worse w ~bound:(g *. sqrt !norm) ~residual:(sqrt !res) (fun () ->
              Printf.sprintf "matrix %d, column %d of a - q r" t j)
        done;
        for i = 0 to qc - 1 do
          for j = 0 to qc - 1 do
            let s = sum () in
            if i = j then add s.re (-1.);
            for p = 0 to m - 1 do
              add_cprod ~conj:true s
                (at q ~rows:m ~cols:qc t p j)
                (at q ~rows:m ~cols:qc t p i)
            done;
            worse w ~bound:g ~residual:(Complex.norm (total s)) (fun () ->
                Printf.sprintf "matrix %d, element (%d, %d) of qᴴq - I" t i j)
          done
        done
      done;
      check_worst w

(* a v = v diag w with ‖a − v diag w vᴴ‖_F <= γ(16n) ‖a‖_F and
   ‖vᴴv − I‖_F <= γ(16n); w ascending and real; the values alone the bits
   of the vectors' call. *)
let law_eigh (b : Support.backend) c =
  let x = laid_out c.how (eigh_operand c) in
  match (run b (Eigh true) [| x |], run b (Eigh false) [| x |]) with
  | None, _ | _, None -> ()
  | Some y, Some y' ->
      equal ~msg:"the values alone" same_bits (bits y.(0)) (bits y'.(0));
      let a = values x and w = values y.(0) and v = values y.(1) and n = c.n in
      let g = gamma c.dt (16 * n) in
      let worst = { ratio = 0.; where = "" } in
      for t = 0 to count c - 1 do
        let wt j = w.((t * n) + j) in
        for j = 0 to n - 1 do
          equal ~msg:"w is real" int64 0L (Int64.bits_of_float (wt j).im);
          if j > 0 && not ((wt (j - 1)).re <= (wt j).re) then
            failf "matrix %d: w not ascending at %d" t j
        done;
        let res = ref 0. and norm = ref 0. and orth = ref 0. in
        for i = 0 to n - 1 do
          for j = 0 to n - 1 do
            let aij = at a ~rows:n ~cols:n t i j in
            let s = sum () and o = sum () in
            add s.re aij.re;
            add s.im aij.im;
            if i = j then add o.re (-1.);
            for p = 0 to n - 1 do
              let vip = at v ~rows:n ~cols:n t i p in
              add_cprod ~conj:true s
                (Complex.mul vip { Complex.re = -.(wt p).re; im = 0. })
                (at v ~rows:n ~cols:n t j p);
              add_cprod ~conj:true o
                (at v ~rows:n ~cols:n t p i)
                (at v ~rows:n ~cols:n t p j)
            done;
            res := !res +. Complex.norm2 (total s);
            orth := !orth +. Complex.norm2 (total o);
            norm := !norm +. Complex.norm2 aij
          done
        done;
        worse worst ~bound:(g *. sqrt !norm) ~residual:(sqrt !res) (fun () ->
            Printf.sprintf "matrix %d: a - v diag w vᴴ" t);
        worse worst ~bound:g ~residual:(sqrt !orth) (fun () ->
            Printf.sprintf "matrix %d: vᴴv - I" t)
      done;
      check_worst worst

(* Of a = H diag λ H, H = I − 2 u uᵀ / uᵀu, the eigenvalues are λ's within
   γ(16n) max |λ|, ascending; λ draws repeats. *)
let law_eigh_spectrum (b : Support.backend) c =
  let n = c.n and r = Random.State.make [| c.seed |] in
  let l = Array.init n (fun _ -> Float.round (8. *. uniform r) /. 8.) in
  let u = Array.init n (fun _ -> uniform r) in
  let uu = Array.fold_left (fun s x -> s +. (x *. x)) 0. u in
  let lu = ref 0. in
  Array.iteri (fun k x -> lu := !lu +. (l.(k) *. x *. x)) u;
  let a =
    Array.init (n * n) (fun k ->
        let i = k / n and j = k mod n in
        let h = u.(i) *. u.(j) /. uu in
        let d = if i = j then l.(i) else 0. in
        { Complex.re = d -. (2. *. h *. (l.(i) +. l.(j))) +. (4. *. h *. !lu /. uu); im = 0. })
  in
  match run b (Eigh false) [| of_values c.dt [| n; n |] a |] with
  | None -> ()
  | Some y ->
      let w = values y.(0) in
      Array.sort compare l;
      let big = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. l in
      let g = gamma c.dt (16 * n) *. big in
      let worst = { ratio = 0.; where = "" } in
      Array.iteri
        (fun j x ->
          worse worst ~bound:g ~residual:(Float.abs (w.(j).re -. x)) (fun () ->
              Printf.sprintf "eigenvalue %d: %h, not %h" j w.(j).re x))
        l;
      check_worst worst

(* lu's pivot: a scan of the column from the diagonal down that
   moves to each later row of greater magnitude, |re| + |im| on complex
   numbers, a NaN never greater and nothing greater than a NaN. Each case is
   a column and the pivot the rule picks. *)
let test_pivots (b : Support.backend) () =
  let nan = Float.nan in
  let c re im = { Complex.re; im } in
  let pivot ~msg dt column want =
    let m = Array.length column in
    match run b Lu [| of_values dt [| m; 1 |] column |] with
    | None -> ()
    | Some y ->
        equal ~msg (array int) [| want |] (int64s y.(1));
        let perm = Array.init m Fun.id in
        perm.(0) <- want;
        perm.(want) <- 0;
        equal ~msg:(msg ^ ": perm") (array int) perm (int64s y.(2))
  in
  List.iter
    (fun dt ->
      pivot ~msg:"a zero diagonal" dt [| c 0. 0.; c 1. 0. |] 1;
      pivot ~msg:"a tie keeps the first" dt [| c 1. 0.; c (-1.) 0. |] 0;
      pivot ~msg:"the greatest past a smaller one" dt
        [| c 1. 0.; c 3. 0.; c (-2.) 0.; c (-4.) 0. |] 3;
      pivot ~msg:"nothing is greater than a NaN" dt
        [| c nan 0.; c 5. 0.; c (-7.) 0. |] 0;
      pivot ~msg:"a NaN is never greater" dt [| c 1. 0.; c nan 0.; c 2. 0. |] 2)
    dtypes;
  List.iter
    (fun dt ->
      pivot ~msg:"|re| + |im|, not the modulus" dt [| c 3. 0.; c 2. 2. |] 1)
    D.[ Any Complex64; Any Complex128 ]

(* Failures and reads *)

(* Matrix [m] of [v], [rows] × [cols] each. *)
let matrix v ~rows ~cols m = Array.sub v (m * rows * cols) (rows * cols)

(* A matrix that fails, between two that do not, is NaN, and leaves theirs
   the bits they have alone. [spoil] makes a matrix's values fail. *)
let check_alone (b : Support.backend) r ~dt ~rows ~cols ~spoil ops =
  let n = Array.length ops in
  let vs = Array.map values ops in
  let three k = Array.concat [ vs.(k); (if k = 0 then spoil vs.(k) else vs.(k)); vs.(k) ] in
  let shape k =
    let (A.Any x) = ops.(k) in
    Array.append [| 3 |] (L.shape (A.layout x))
  in
  let batch = Array.init n (fun k -> of_values dt (shape k) (three k)) in
  match (run b r batch, run b r ops) with
  | Some y, Some alone ->
      let yv = values y.(0) in
      let want = bits alone.(0) in
      let mat m =
        bits (of_values dt [| rows; cols |] (matrix yv ~rows ~cols m))
      in
      equal ~msg:"the first is alone" same_bits want (mat 0);
      equal ~msg:"the last is alone" same_bits want (mat 2);
      equal ~msg:"numbers in the failing one" int 0
        (numbers dt (matrix yv ~rows ~cols 1))
  | _ -> ()

let one c = { c with batch = [||] }

let law_cholesky_fails (b : Support.backend) c =
  let c = one c in
  if c.n > 0 then
    let k = c.seed mod c.n in
    let spoil v =
      let v = Array.copy v in
      v.((k * c.n) + k) <- { Complex.re = -1.; im = 0. };
      v
    in
    check_alone b (Cholesky Lower) ~dt:c.dt ~rows:c.n ~cols:c.n ~spoil
      [| cholesky_operand c |]

(* NaN or an infinity in the lower triangle. *)
let law_eigh_fails (b : Support.backend) c =
  let c = one c in
  if c.n > 0 then
    let i = c.seed mod c.n in
    let j = c.seed / c.n mod (i + 1) in
    let bad = [| Float.nan; Float.infinity; Float.neg_infinity |].(c.seed mod 3) in
    let spoil v =
      let v = Array.copy v in
      v.((i * c.n) + j) <- { Complex.re = bad; im = 0. };
      v
    in
    check_alone b (Eigh true) ~dt:c.dt ~rows:c.n ~cols:1 ~spoil
      [| eigh_operand c |]

let law_solve_fails (b : Support.backend) c =
  let c = { (one c) with unit_diagonal = false; r = max 1 c.r } in
  if c.n > 0 then
    let k = c.seed mod c.n in
    let spoil v =
      let v = Array.copy v in
      v.((k * c.n) + k) <- Complex.zero;
      v
    in
    check_alone b (solve c) ~dt:c.dt ~rows:c.n ~cols:c.r ~spoil
      (solve_operands c)

(* [v] with NaN wherever [unread i j] holds, and imaginary parts drawn on the
   diagonal where [diagonal_im]. *)
let garbage c v ~unread ~diagonal_im =
  let r = Random.State.make [| c.seed + 1 |] in
  let n = c.n in
  Array.mapi
    (fun k (x : Complex.t) ->
      let i = k / n mod n and j = k mod n in
      if unread i j then { Complex.re = Float.nan; im = Float.nan }
      else if i = j && diagonal_im && is_complex c.dt then { x with im = uniform r }
      else x)
    v

(* What a routine does not read changes no bit of its results. *)
let law_cholesky_reads (b : Support.backend) c =
  let x = cholesky_operand c in
  let v = garbage c (values x) ~unread:(fun i j -> j > i) ~diagonal_im:true in
  match (run b (Cholesky Lower) [| x |], run b (Cholesky Lower) [| of_values c.dt (square c) v |]) with
  | Some y, Some z -> equal same_bits (bits y.(0)) (bits z.(0))
  | _ -> ()

let law_eigh_reads (b : Support.backend) c =
  let x = eigh_operand c in
  let v = garbage c (values x) ~unread:(fun i j -> j > i) ~diagonal_im:true in
  let y = of_values c.dt (square c) v in
  match (run b (Eigh true) [| x |], run b (Eigh true) [| y |]) with
  | Some y, Some z ->
      equal ~msg:"w" same_bits (bits y.(0)) (bits z.(0));
      equal ~msg:"v" same_bits (bits y.(1)) (bits z.(1))
  | _ -> ()

let law_solve_reads (b : Support.backend) c =
  let ops = solve_operands c in
  let unread i j =
    (if c.triangle = Lower then j > i else j < i) || (c.unit_diagonal && i = j)
  in
  let v = garbage c (values ops.(0)) ~unread ~diagonal_im:false in
  let spoilt = [| of_values c.dt (square c) v; ops.(1) |] in
  match (run b (solve c) ops, run b (solve c) spoilt) with
  | Some y, Some z -> equal same_bits (bits y.(0)) (bits z.(0))
  | _ -> ()

(* Layouts: a matrix through another layout, or one matrix broadcast across
   the batch, gives the bits its contiguous copy gives. *)
let law_layouts (b : Support.backend) c =
  let x = cholesky_operand c in
  (match (run b (Cholesky Lower) [| x |], run b (Cholesky Lower) [| laid_out c.how x |]) with
  | Some y, Some z -> equal ~msg:"cholesky" same_bits (bits y.(0)) (bits z.(0))
  | _ -> ());
  let _, x = lu_operand c in
  (match (run b Lu [| x |], run b Lu [| laid_out c.how x |]) with
  | Some y, Some z ->
      equal ~msg:"lu" same_bits (bits y.(0)) (bits z.(0));
      equal ~msg:"lu's pivots" (array int) (int64s y.(1)) (int64s z.(1))
  | _ -> ());
  let x = eigh_operand c in
  (match (run b (Eigh true) [| x |], run b (Eigh true) [| laid_out c.how x |]) with
  | Some y, Some z ->
      equal ~msg:"eigh's w" same_bits (bits y.(0)) (bits z.(0));
      equal ~msg:"eigh's v" same_bits (bits y.(1)) (bits z.(1))
  | _ -> ());
  let ops = solve_operands { c with batch = [| 1 |] } in
  let k = 3 in
  let (A.Any a) = ops.(0) in
  let wide = Option.get (A.move (M.Broadcast [| k; c.n; c.n |]) a) in
  let rhs = Array.init (k * c.n * c.r) (fun i -> { Complex.re = float_of_int (i mod 7) -. 3.; im = 0. }) in
  let bs = of_values c.dt [| k; c.n; c.r |] rhs in
  let copy = A.Any (A.copy wide) in
  match (run b (solve c) [| A.Any wide; bs |], run b (solve c) [| copy; bs |]) with
  | Some y, Some z -> equal ~msg:"a broadcast" same_bits (bits y.(0)) (bits z.(0))
  | _ -> ()

(* nx.cpu *)

(* CR: Through public linalg, compare each matrix's complete results alone
   and in a batch, including QR's R and LU's positions omitted here. Keep
   generated cases and a varied batch large enough for parallel work.
   This removes Support.serial and its production threads switch; nested
   GEMMs choose their threads in both current paths. *)

(* nx.cpu computes the routines of [c] at its dtype, and on one thread
   gives the job's bits. *)
let threads_agree (b : Support.backend) c =
  let x = cholesky_operand c and ops = solve_operands c in
  let _, lu = lu_operand c in
  List.iter
    (fun (msg, r, ops) ->
      match (run b r ops, run ~k:Support.serial b r ops) with
      | Some y, Some z -> equal ~msg same_bits (bits y.(0)) (bits z.(0))
      | _ -> failf "nx.cpu declined %s at %a" msg pp_dtype c.dt)
    [
      ("cholesky", Cholesky Lower, [| x |]);
      ("solve", solve c, ops);
      ("lu", Lu, [| lu |]);
      ("qr", Qr (factors c), [| lu |]);
      ("eigh", Eigh true, [| eigh_operand c |]);
    ]

(* nx.cpu computes its routines at the four dtypes, and
   its matrices on one thread give the job's bits. *)
let law_cpu_threads b c =
  threads_agree b { c with batch = (if c.n > 17 then [| 2 |] else [| 9 |]) }

(* One matrix of order 256 and 160 right-hand sides: Cholesky's solve of
   128 rows, LU's of 128 columns and the solve's of 160 split into units
   of columns on the threads, which give the bits of one thread. *)
let law_cpu_units b c =
  cover "units on the threads" (Domain.recommended_domain_count () > 1);
  threads_agree b { c with batch = [||]; n = 256; r = 160 }

(* A routine's operand [a], right-hand sides [b] for a solve, and
   destinations [dsts], each near a fit. *)
type misfit = {
  routine : routine;
  a : int array;
  b : int array;
  dsts : int array array;
}

let routine_name = function
  | Cholesky _ -> "cholesky"
  | Lu -> "lu"
  | Qr Reduced -> "qr reduced"
  | Qr Complete -> "qr complete"
  | Eigh true -> "eigh"
  | Eigh false -> "eigh's values"
  | Solve _ -> "solve_triangular"

let pp_misfit ppf c =
  Format.fprintf ppf "%s of %a%s into %s" (routine_name c.routine) pp_ints c.a
    (if c.b <> [||] then Format.asprintf " and %a" pp_ints c.b else "")
    (String.concat ", "
       (Array.to_list (Array.map (Format.asprintf "%a" pp_ints) c.dsts)))

(* The destinations' shapes Nx_kernel.Shape gives. *)
let shapes c =
  let module S = Nx_kernel.Shape in
  match c.routine with
  | Cholesky _ -> Result.map (fun y -> [| y |]) (S.cholesky c.a)
  | Lu -> Result.map (fun (l, p, q) -> [| l; p; q |]) (S.lu c.a)
  | Qr f -> Result.map (fun (q, r) -> [| q; r |]) (S.qr f c.a)
  | Eigh v ->
      Result.map (fun (w, x) -> if v then [| w; x |] else [| w |]) (S.eigh c.a)
  | Solve _ -> Result.map (fun y -> [| y |]) (S.solve_triangular c.a c.b)

let arity = function
  | Cholesky _ | Solve _ | Eigh false -> 1
  | Lu -> 3
  | Qr _ | Eigh true -> 2

(* Square matrices half the time, and a vector now and then. *)
let misfits =
  Gen.with_pp pp_misfit
    (let open Gen in
     let solve =
       Solve { triangle = Lower; adjoint = false; unit_diagonal = false }
     in
     let* routine =
       of_list
         [ Cholesky Lower; Lu; Qr Reduced; Qr Complete; Eigh true; Eigh false; solve ]
     in
     let* batch = array ~size:(int_range 0 2) (int_range 0 3) in
     let* m = int_range 0 4 in
     let* n = int_range 0 4 in
     let* square = bool in
     let n = if square then m else n in
     let* vector = int_range 0 9 in
     let a = if vector = 0 then [| m |] else Array.append batch [| m; n |] in
     let* cols = int_range 0 3 in
     let* b = Elements.near (Array.append batch [| n; cols |]) in
     let b = if routine = solve then b else [||] in
     let c = { routine; a; b; dsts = [||] } in
     let+ dsts =
       match shapes c with
       | Ok ys ->
           let rec go i acc =
             if i < 0 then constant (Array.of_list acc)
             else
               let* y = Elements.near ys.(i) in
               go (i - 1) (y :: acc)
           in
           go (Array.length ys - 1) []
       | Error _ -> constant (Array.make (arity routine) a)
     in
     { c with dsts })

let law_cpu_refuses (b : Support.backend) c =
  let module K = (val b.kernels) in
  let zeros s = A.of_array D.Float32 s (Array.make (Elements.total s) 0.) in
  let dst i = A.create Rig.host D.Float32 c.dsts.(i) in
  let ints i = A.create Rig.host D.Int64 c.dsts.(i) in
  let a = zeros c.a in
  let got =
    match c.routine with
    | Cholesky t -> K.cholesky t ~dst:(dst 0) a
    | Lu -> K.lu ~lu:(dst 0) ~pivots:(ints 1) ~perm:(ints 2) a
    | Qr _ -> K.qr ~q:(dst 0) ~r:(dst 1) a
    | Eigh true -> K.eigh ~w:(dst 0) ~v:(dst 1) a
    | Eigh false -> K.eigh ~w:(dst 0) a
    | Solve { triangle; adjoint; unit_diagonal } ->
        K.solve_triangular triangle ~adjoint ~unit_diagonal ~dst:(dst 0) a
          (zeros c.b)
  in
  (* A QR's destinations fit as either factors' shapes. *)
  let fit =
    match c.routine with
    | Qr _ ->
        List.exists
          (fun f -> shapes { c with routine = Qr f } = Ok c.dsts)
          [ Nx_kernel.Reduced; Complete ]
    | _ -> shapes c = Ok c.dsts
  in
  cover "fits" fit;
  cover "misfits" (not fit);
  equal answer (if fit then A.Done else A.Shape_mismatch) got

let test_cpu_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let f32 = A.of_array D.Float32 [| 2; 2 |] [| 4.; 1.; 1.; 3. |] in
  let i32 = A.of_array D.Int32 [| 2; 2 |] [| 4l; 1l; 1l; 3l |] in
  let wide = A.of_array D.Float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let sentinel dt v s = A.of_array dt s (Array.make (Array.fold_left ( * ) 1 s) v) in
  let refuses ~msg t want got dst seven =
    equal ~msg answer want got;
    let n = Array.length (A.to_array dst) in
    equal ~msg:(msg ^ ": nothing written") (array t) (Array.make n seven)
      (A.to_array dst)
  in
  let dst = sentinel D.Int32 7l [| 2; 2 |] in
  refuses ~msg:"int32" int32 A.Wrong_dtype (K.cholesky Lower ~dst i32) dst 7l;
  let dst = sentinel D.Float32 7. [| 2; 3 |] in
  refuses ~msg:"not square" float_exact A.Shape_mismatch (K.cholesky Lower ~dst wide) dst 7.;
  let dst = sentinel D.Float32 7. [| 3; 3 |] in
  refuses ~msg:"a result of another shape" float_exact A.Shape_mismatch
    (K.cholesky Lower ~dst f32) dst 7.;
  let dst = sentinel D.Float32 7. [| 3; 2 |] in
  refuses ~msg:"right-hand sides of other rows" float_exact
    A.Shape_mismatch
    (K.solve_triangular Lower ~adjoint:false ~unit_diagonal:false ~dst f32
       (sentinel D.Float32 7. [| 3; 2 |]))
    dst 7.;
  let tall = A.of_array D.Float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let q = sentinel D.Float32 7. [| 3; 2 |] in
  refuses ~msg:"a reduced q with a complete r" float_exact
    A.Shape_mismatch
    (K.qr ~q ~r:(sentinel D.Float32 7. [| 3; 2 |]) tall)
    q 7.;
  let c64 = A.of_array D.Complex64 [| 2; 2 |] (Array.make 4 Complex.one) in
  let w () = A.create Rig.host D.Complex64 [| 2 |] in
  let declines ~msg got =
    equal ~msg:("declines the routines it does not compute: " ^ msg) answer
      A.Declined got
  in
  let sq () = A.create Rig.host D.Complex64 [| 2; 2 |] in
  declines ~msg:"svd" (K.svd ~s:(w ()) ~vectors:(~u:(sq ()), ~vh:(sq ())) c64);
  declines ~msg:"svd's values" (K.svd ~s:(w ()) c64);
  declines ~msg:"eig" (K.eig ~w:(w ()) ~v:(sq ()) c64);
  declines ~msg:"eig's values" (K.eig ~w:(w ()) c64)

(* Every routine answers Wrong_dtype at the dtypes routines do not take,
   and computes or declines float32, float64, complex64 and complex128.
   [eig] answers Wrong_dtype also for results of another precision than
   [a]'s. *)
let test_dtypes (b : Support.backend) () =
  let module K = (val b.kernels) in
  List.iter
    (fun (D.Any dt) ->
      let takes =
        match dt with
        | D.Float32 | D.Float64 | D.Complex64 | D.Complex128 -> true
        | _ -> false
      in
      let x s = A.create b.device dt s in
      let positions s = A.create b.device D.Int64 s in
      let sq = [| 2; 2 |] in
      let a = x sq in
      let wide = match dt with D.Float64 | D.Complex128 -> true | _ -> false in
      let eig (type c) (c : (Complex.t, c) D.t) =
        let z s = A.create b.device c s in
        let paired = wide = (D.Any c = D.Any D.Complex128) in
        let check ~msg got =
          if paired && takes then begin
            if got = A.Wrong_dtype then failf "%s answered Wrong_dtype" msg
          end
          else equal ~msg answer A.Wrong_dtype got
        in
        let msg = Format.asprintf "eig of %a into %a" D.pp dt D.pp c in
        check ~msg (K.eig ~w:(z [| 2 |]) ~v:(z sq) a);
        check ~msg:(msg ^ ", values") (K.eig ~w:(z [| 2 |]) a)
      in
      eig D.Complex64;
      eig D.Complex128;
      List.iter
        (fun (name, call) ->
          let msg = Format.asprintf "%s of %a" name D.pp dt in
          let got = call () in
          if not takes then equal ~msg answer A.Wrong_dtype got
          else if got = A.Wrong_dtype then failf "%s answered Wrong_dtype" msg)
        [
          ("cholesky", fun () -> K.cholesky Lower ~dst:(x sq) a);
          ( "lu",
            fun () ->
              K.lu ~lu:(x sq) ~pivots:(positions [| 2 |])
                ~perm:(positions [| 2 |]) a );
          ("qr", fun () -> K.qr ~q:(x sq) ~r:(x sq) a);
          ( "svd",
            fun () ->
              K.svd ~s:(x [| 2 |]) ~vectors:(~u:(x sq), ~vh:(x sq)) a );
          ("svd's values", fun () -> K.svd ~s:(x [| 2 |]) a);
          ("eigh", fun () -> K.eigh ~w:(x [| 2 |]) ~v:(x sq) a);
          ("eigh's values", fun () -> K.eigh ~w:(x [| 2 |]) a);
          ( "solve_triangular",
            fun () ->
              K.solve_triangular Lower ~adjoint:false ~unit_diagonal:false
                ~dst:(x sq) a (x sq) );
        ])
    D.all

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop "Cholesky's factor meets its relation within γ(n + 1)" case
        (run (law_cholesky b));
      prop "the Upper factor is the Lower one's conjugate transpose" case
        (run (law_upper b));
      prop "a triangular solve meets its relation within γ(2n)" case
        (run (law_solve b));
      prop "LU's factors meet their relation within γ(2k)" case
        (run (law_lu b));
      prop "QR's factors meet their relations within γ(k(2m + 10))" case
        (run (law_qr b));
      prop "eigh's results meet their relations within γ(16n)" case
        (run (law_eigh b));
      prop "eigh's eigenvalues are within γ(16n) ‖a‖₂ of the exact ones" case
        (run (law_eigh_spectrum b));
      prop "a matrix that is not definite is NaN alone" case
        (run (law_cholesky_fails b));
      prop "a singular triangle is NaN alone" case (run (law_solve_fails b));
      prop "eigh of a matrix with NaN or an infinity is NaN alone" case
        (run (law_eigh_fails b));
      prop "eigh reads the lower triangle and the diagonal's real parts" case
        (run (law_eigh_reads b));
      prop "Cholesky reads the lower triangle and the diagonal's real parts"
        case (run (law_cholesky_reads b));
      prop "a solve reads its triangle, and its diagonal unless unit" case
        (run (law_solve_reads b));
      prop "bits do not depend on layouts" case (run (law_layouts b));
      test "LU's pivot is the scan's" (fun () -> b.around (test_pivots b));
      test "routines take floats and complex numbers alone" (fun () ->
          b.around (test_dtypes b));
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop ~count:30 "computes every dtype, one thread giving the job's bits"
        case (run (law_cpu_threads b));
      prop ~count:8 "a solve's units of columns give one thread's bits" case
        (run (law_cpu_units b));
      prop "refuses what Nx_kernel.Shape refuses and other shapes" misfits
        (run (law_cpu_refuses b));
      test "refuses dtypes and shapes, declines the other routines" (fun () ->
          b.around (test_cpu_refusals b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.linalg"
       (List.map laws Support.backends @ List.map cpu Support.cpus))
