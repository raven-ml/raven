(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Factorisations and triangular solves through every kernel library the host
   runs: each result against its defining relation, within the bound
   nx_cpu.mli states, by residuals computed in double-double, on operands
   well and badly conditioned and below the least normal number; LU's
   pivots as nx_kernel.mli's lu chooses them; a failing matrix NaN alone in
   its batch; only the triangle a routine names read; the same bits under
   every layout of the same values. nx.cpu's group checks that it computes
   what nx_cpu.mli lists, the same bits on one thread as on the job's, and
   its refusals. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move
module Support = Nx_kernels_support

let pp_dtype ppf (D.Any dt) = D.pp ppf dt
let dtypes = D.[ Any Float32; Any Float64; Any Complex64; Any Complex128 ]
let is_complex (D.Any dt) = D.is D.Complex dt

(* The complex dtype of [dt]'s precision: eig's results'. *)
let complex_of (D.Any dt) =
  match dt with
  | D.Float32 | D.Complex64 -> D.Any D.Complex64
  | _ -> D.Any D.Complex128

(* The dtype's unit roundoff. *)
let unit_roundoff (D.Any dt) =
  match dt with D.Float32 | D.Complex64 -> 0x1p-24 | _ -> 0x1p-53

(* γ(k) = k u / (1 − k u), for [k] operations, twice as many on complex
   numbers, as nx_cpu.mli states the bounds. *)
let gamma dt k =
  let k = if is_complex dt then 2 * k else k in
  let ku = float_of_int k *. unit_roundoff dt in
  ku /. (1. -. ku)

(* ε(k) = k η, η the dtype's least positive subnormal, [k] doubled on
   complex numbers: what underflow adds to the bounds, times [2^e]. *)
let epsilon ?(e = 0) dt k =
  let k = if is_complex dt then 2 * k else k in
  let eta =
    match dt with D.Any (D.Float32 | D.Complex64) -> -149 | _ -> -1074
  in
  float_of_int k *. Float.ldexp 1. (eta + e)

(* The exponent that takes values in [-1, 1] below the dtype's least normal
   number, to [bits] bits above its least subnormal: deep enough that an
   eigenvalue's rounding outweighs γ(16n) ‖a‖. *)
let tiny ~bits (D.Any dt) =
  (match dt with D.Float32 | D.Complex64 -> 149 | _ -> 1074) - bits

(* [x] times 2^e, exact where the result is normal: a subnormal operand's
   results scaled back, so that double-double residuals do not underflow. *)
let ldexp e (x : Complex.t) =
  { Complex.re = Float.ldexp x.re e; im = Float.ldexp x.im e }

let up e v = Array.map (ldexp e) v

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

(* Matrix orders on both sides of a block of 64, 26, the first that divide
   and conquer tears, and small ones. *)
let orders = [ 0; 1; 2; 3; 5; 8; 17; 26; 63; 64; 65; 129 ]

(* How a case scales its operand. *)
type scale =
  | Plain
  | Graded
      (** rows and columns by powers of two from 2^-16 to 2^16: an operand
          of condition up to 2^64 whose exact factors are those of the
          ungraded one, scaled *)
  | Subnormal
      (** by 2^-(down c), below the least normal number: the bounds' ε
          terms. A solve scales its right-hand sides. *)

let pp_scale ppf = function
  | Plain -> ()
  | Graded -> Format.pp_print_string ppf " graded"
  | Subnormal -> Format.pp_print_string ppf " subnormal"

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
  scale : scale;
}

let pp_case ppf c =
  Format.fprintf ppf "%a batch %a n %d r %d %s%s%s seed %d %a%a" pp_dtype c.dt
    pp_ints c.batch c.n c.r
    (match c.triangle with Lower -> "lower" | Upper -> "upper")
    (if c.adjoint then " adjoint" else "")
    (if c.unit_diagonal then " unit" else "")
    c.seed pp_how c.how pp_scale c.scale

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
     let+ scale = of_list ~pp:pp_scale [ Plain; Graded; Subnormal ] in
     { dt; batch; n; r; triangle; adjoint; unit_diagonal; seed; how; scale })

let count c = Array.fold_left ( * ) 1 c.batch
let square c = Array.append c.batch [| c.n; c.n |]

(* Element (i, j) of matrix [b] of the [rows] × [cols] matrices [v]. *)
let at v ~rows ~cols b i j = v.((((b * rows) + i) * cols) + j)

(* [v], matrices of [rows] × [cols], with element (i, j) times
   2^(e_i + f_j) under [Graded], e and f drawn from the seed in [-8, 8],
   f = e under [symmetric]. *)
let grade c ?(symmetric = false) ~rows ~cols v =
  if c.scale <> Graded then v
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

(* The exponent [Subnormal] takes its operand down by, to [bits] bits, else
   0. *)
let down ?(bits = 8) c = if c.scale = Subnormal then tiny ~bits c.dt else 0

(* [v] times 2^-(down c). *)
let shrink ?bits c v = up (-down ?bits c) v

(* The bits a positive definite operand keeps: its elements, at most about 1,
   move by n 2^-19 at most, below its least eigenvalue's δ. *)
let definite_bits = 19

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
  | Svd of Nx_kernel.factors option  (** the vectors' shapes, or values alone *)
  | Eig of bool  (** with vectors *)
  | Solve of {
      triangle : Nx_kernel.triangle;
      adjoint : bool;
      unit_diagonal : bool;
    }

(* [k]'s results of [r] on [ops], or [None] if it declined: Lu's positions
   int64, eig's results complex of a's precision, every other result of a's
   dtype, of the shapes Nx_kernel.Shape gives. *)
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
        let sw, sv = ok (Shape.eigh (Vectors ()) (shape ops.(0))) in
        let w = make sw in
        if vectors then
          let v = make sv in
          ([| A.Any w; A.Any v |], K.eigh ~w ~v a)
        else ([| A.Any w |], K.eigh ~w a)
    | Svd None ->
        let s = make (fst (ok (Shape.svd No_vectors (shape ops.(0))))) in
        ([| A.Any s |], K.svd ~s a)
    | Svd (Some f) ->
        let ss, (su, sv) = ok (Shape.svd (Vectors f) (shape ops.(0))) in
        let s = make ss and u = make su and vh = make sv in
        ([| A.Any s; A.Any u; A.Any vh |], K.svd ~s ~vectors:(~u, ~vh) a)
    | Eig vectors -> (
        let sw, sv = ok (Shape.eig (Vectors ()) (shape ops.(0))) in
        let eig (type c) (cdt : (Complex.t, c) D.t) =
          let w = A.create Rig.host cdt sw in
          if vectors then
            let v = A.create Rig.host cdt sv in
            ([| A.Any w; A.Any v |], K.eig ~w ~v a)
          else ([| A.Any w |], K.eig ~w a)
        in
        match dt with
        | D.Float32 | D.Complex64 -> eig D.Complex64
        | _ -> eig D.Complex128)
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
  of_values c.dt (square c)
    (shrink ~bits:definite_bits c
       (grade c ~symmetric:true ~rows:c.n ~cols:c.n v))

let eigh_operand c =
  let r = Random.State.make [| c.seed |] in
  let v = Array.concat (List.init (count c) (fun _ -> hermitian r c.dt c.n)) in
  of_values c.dt (square c)
    (shrink c (grade c ~symmetric:true ~rows:c.n ~cols:c.n v))

(* A matrix of order [n] with |a_ij| = |a_ji|, which eig's balancing
   leaves as it is: its lower triangle drawn, and each element above the
   one below times a sign, or on complex numbers a phase, drawn. *)
let mirrored r dt n =
  let m = Array.init (n * n) (fun _ -> draw r dt) in
  Array.init (n * n) (fun k ->
      let i = k / n and j = k mod n in
      if i >= j then m.(k)
      else
        let p = draw r dt in
        let q = Complex.norm p in
        if q = 0. then m.((j * n) + i)
        else
          Complex.mul m.((j * n) + i) { Complex.re = p.re /. q; im = p.im /. q })

(* A lower triangular matrix of order [n], whose rows and columns
   balancing isolates one by one. *)
let lower r dt n =
  Array.init (n * n) (fun k -> if k / n >= k mod n then draw r dt else Complex.zero)

(* A block upper triangular matrix of order [n], its rows and columns
   permuted by [p]: [i0] rows and columns, then a mirrored block, then
   [i1], each of the outer ones upper triangular, so that balancing
   isolates the outer ones' eigenvalues by permutations and balances the
   block alone. *)
let blocks r dt n =
  let i0 = Random.State.int r 3 and i1 = Random.State.int r 3 in
  let i0 = min i0 n in
  let i1 = min i1 (n - i0) in
  let m = n - i0 - i1 in
  let mid = mirrored r dt m in
  let b =
    Array.init (n * n) (fun k ->
        let i = k / n and j = k mod n in
        let inside x = x >= i0 && x < i0 + m in
        if inside i && inside j then mid.(((i - i0) * m) + j - i0)
        else if j >= i && not (inside i && j < i0 + m) then draw r dt
        else Complex.zero)
  in
  let p = Array.init n Fun.id in
  for i = n - 1 downto 1 do
    let j = Random.State.int r (i + 1) in
    let t = p.(i) in
    p.(i) <- p.(j);
    p.(j) <- t
  done;
  Array.init (n * n) (fun k -> b.((p.(k / n) * n) + p.(k mod n)))

(* eig's operands, by the seed: mirrored, lower triangular or permuted
   block triangular, so that balancing leaves each with D = I. *)
let eig_operand c =
  let r = Random.State.make [| c.seed |] in
  let draw = [| mirrored; lower; blocks |].(c.seed mod 3) in
  let v = Array.concat (List.init (count c) (fun _ -> draw r c.dt c.n)) in
  of_values c.dt (square c)
    (shrink c (grade c ~symmetric:true ~rows:c.n ~cols:c.n v))

let solve_operands c =
  let r = Random.State.make [| c.seed |] in
  let a = Array.concat (List.init (count c) (fun _ -> triangular r c.dt c.n)) in
  let a = grade c ~rows:c.n ~cols:c.n a in
  let b = shrink c (Array.init (count c * c.n * c.r) (fun _ -> draw r c.dt)) in
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

(* L Lᴴ = a + Δ with |Δ_ij| <= γ(n + 1) (|L| |Lᴴ|)_ij + ε(n + 1) (1 + L_jj)
   over the lower triangle, the other triangle +0, the diagonal real and
   positive. A subnormal a is checked as 2^2e a = (2^e L) (2^e L)ᴴ. *)
let law_cholesky (b : Support.backend) c =
  let x = cholesky_operand c in
  match run b (Cholesky Lower) [| laid_out c.how x |] with
  | None -> ()
  | Some y ->
      let e = (down ~bits:definite_bits c + 1) / 2 in
      let a = up (2 * e) (values x) and l = up e (values y.(0)) in
      let n = c.n in
      let w = { ratio = 0.; where = "" } in
      let g = gamma c.dt (n + 1) and h = epsilon ~e:(2 * e) c.dt (n + 1) in
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
              let ljj = (at l ~rows:n ~cols:n m j j).re in
              let h = h *. (1. +. Float.ldexp ljj (-e)) in
              worse w ~bound:((g *. bound.hi) +. h)
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

(* (op(a) + Δ) x = b + δ with |Δ| <= γ(2n) |op(a)| and
   |δ_i| <= ε(2n) (1 + |op(a)_ii|) per column, op(a) the triangle read,
   transposed and conjugated under [adjoint], ones on its diagonal under
   [unit_diagonal]. A subnormal b is checked with b and x times 2^d. *)
let law_solve (b : Support.backend) c =
  let ops = solve_operands c in
  match run b (solve c) [| laid_out c.how ops.(0); ops.(1) |] with
  | None -> ()
  | Some y ->
      let d = down c in
      let a = values ops.(0) and rhs = up d (values ops.(1)) in
      let x = up d (values y.(0)) and n = c.n and r = c.r in
      let upper = (c.triangle = Upper) <> c.adjoint in
      let op m i p =
        if i = p && c.unit_diagonal then Complex.one
        else if c.adjoint then Complex.conj (at a ~rows:n ~cols:n m p i)
        else at a ~rows:n ~cols:n m i p
      in
      let w = { ratio = 0.; where = "" } in
      let g = gamma c.dt (2 * n) and h = epsilon ~e:d c.dt (2 * n) in
      for m = 0 to count c - 1 do
        for i = 0 to n - 1 do
          let h = h *. (1. +. Complex.norm (op m i i)) in
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
            worse w ~bound:((g *. bound.hi) +. h)
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
  let v = shrink c (grade c ~rows:c.n ~cols v) in
  (cols, of_values c.dt (Array.append c.batch [| c.n; cols |]) v)

(* svd's operand: lu_operand's, or its transpose by the seed, so that it has
   fewer, as many or more rows than columns; its rows and columns. *)
let svd_operand c =
  let cols, x = lu_operand c in
  if c.seed / 3 mod 2 = 0 then (c.n, cols, x)
  else
    let v = values x and m = cols and n = c.n in
    let t =
      Array.init (count c * m * n) (fun p ->
          let b = p / (m * n) and i = p / n mod m and j = p mod n in
          v.((b * m * n) + (j * m) + i))
    in
    (m, n, of_values c.dt (Array.append c.batch [| m; n |]) t)

let int64s (A.Any x) =
  match A.dtype x with
  | D.Int64 -> Array.map Int64.to_int (A.to_array x)
  | _ -> invalid_arg "int64s"

(* L U = P a + Δ with |Δ_ij| <= γ(2k) (|L| |U|)_ij + ε(2k) (1 + |U_pp|),
   p = min(i, j), L unit lower triangular and of magnitude at most 1 on
   reals, the interchanges [pivots] giving [perm], and no NaN from
   numbers. A subnormal a is checked as L (2^d U) = P (2^d a). *)
let law_lu (b : Support.backend) c =
  let cols, x = lu_operand c in
  match run b Lu [| laid_out c.how x |] with
  | None -> ()
  | Some y ->
      let d = down c and m = c.n and n = cols in
      let a = up d (values x) in
      let lu =
        Array.mapi
          (fun p v -> if p / n mod m <= p mod n then ldexp d v else v)
          (values y.(0))
      in
      let piv = int64s y.(1) and perm = int64s y.(2) in
      let k = min m n in
      let w = { ratio = 0.; where = "" } in
      let g = gamma c.dt (2 * k) and h = epsilon ~e:d c.dt (2 * k) in
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
            let p = min i j in
            let upp = Complex.norm (at lu ~rows:m ~cols:n t p p) in
            let h = h *. (1. +. Float.ldexp upp (-d)) in
            worse w ~bound:((g *. bound.hi) +. h)
              ~residual:(Complex.norm (total s))
              (fun () -> Printf.sprintf "matrix %d, element (%d, %d)" t i j)
          done
        done
      done;
      check_worst w

let factors c = if c.seed mod 2 = 0 then Nx_kernel.Reduced else Nx_kernel.Complete

(* a = q r with, per column, ‖a_j − q r_j‖ <= g ‖a_j‖ + √m h; each element
   of qᴴq − I at most g, g = γ(k(2m + 10)) and h = ε(k(2m + 10)); r +0
   below its diagonal and real on it. A subnormal a is checked as
   2^d a = q (2^d r). *)
let law_qr (b : Support.backend) c =
  let cols, x = lu_operand c in
  let f = factors c in
  match run b (Qr f) [| laid_out c.how x |] with
  | None -> ()
  | Some y ->
      let d = down c in
      let a = up d (values x) and q = values y.(0) in
      let r = up d (values y.(1)) in
      let m = c.n and n = cols in
      let qc = if f = Complete then m else min m n in
      let g = gamma c.dt (min m n * ((2 * m) + 10)) in
      let h = epsilon ~e:d c.dt (min m n * ((2 * m) + 10)) in
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
          worse w
            ~bound:((g *. sqrt !norm) +. (sqrt (float_of_int m) *. h))
            ~residual:(sqrt !res) (fun () ->
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

(* a v = v diag w with ‖a − v diag w vᴴ‖_F <= g ‖a‖_F + h and
   ‖vᴴv − I‖_F <= g, g = γ(16n) and h = ε(16n), for the [count] matrices of
   order [n] of [a]; w ascending and real. [a] and [w] are scaled by 2^e. *)
let check_eigh dt ~e ~count ~n (a : Complex.t array) (w : Complex.t array)
    (v : Complex.t array) =
  let g = gamma dt (16 * n) and h = epsilon ~e dt (16 * n) in
  let worst = { ratio = 0.; where = "" } in
  for t = 0 to count - 1 do
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
    worse worst ~bound:((g *. sqrt !norm) +. h) ~residual:(sqrt !res)
      (fun () -> Printf.sprintf "matrix %d: a - v diag w vᴴ" t);
    worse worst ~bound:g ~residual:(sqrt !orth) (fun () ->
        Printf.sprintf "matrix %d: vᴴv - I" t)
  done;
  check_worst worst

(* eigh's results meet check_eigh's relations, and the values alone lie
   within 2 (g ‖a‖_F + h) of the vectors' call's, each within g ‖a‖₂ + h
   of the exact ones. *)
let law_eigh (b : Support.backend) c =
  let x = laid_out c.how (eigh_operand c) in
  match (run b (Eigh true) [| x |], run b (Eigh false) [| x |]) with
  | Some y, Some y' ->
      let d = down c and n = c.n in
      let a = up d (values x) and w = up d (values y.(0)) in
      check_eigh c.dt ~e:d ~count:(count c) ~n a w (values y.(1));
      let w' = up d (values y'.(0)) and g = gamma c.dt (16 * n) in
      let h = epsilon ~e:d c.dt (16 * n) in
      let worst = { ratio = 0.; where = "" } in
      for t = 0 to count c - 1 do
        let norm = ref 0. in
        for k = 0 to (n * n) - 1 do
          norm := !norm +. Complex.norm2 a.((t * n * n) + k)
        done;
        for j = 0 to n - 1 do
          let k = (t * n) + j in
          worse worst
            ~bound:(2. *. ((g *. sqrt !norm) +. h))
            ~residual:(Float.abs (w.(k).re -. w'.(k).re)) (fun () ->
              Printf.sprintf "matrix %d: eigenvalue %d alone" t j)
        done
      done;
      check_worst worst
  | _ -> ()

(* Of a = H diag λ H, H the product of the reflectors I − 2 u uᵀ / uᵀu of
   u on rows [0, k) and on rows [k, n), the eigenvalues are λ's within
   γ(16n) max |λ|, ascending, with and without the vectors, and the
   vectors meet their relations; λ draws repeats, which a merge of divide
   and conquer deflates, and a's two blocks are uncoupled. *)
let law_eigh_spectrum (b : Support.backend) c =
  let n = c.n and r = Random.State.make [| c.seed |] in
  let k = c.seed mod (n + 1) in
  let block i = if i < k then 0 else 1 in
  let l = Array.init n (fun _ -> Float.round (8. *. uniform r) /. 8.) in
  let u = Array.init n (fun _ -> uniform r) in
  let uu = [| 0.; 0. |] and lu = [| 0.; 0. |] in
  Array.iteri
    (fun i x ->
      uu.(block i) <- uu.(block i) +. (x *. x);
      lu.(block i) <- lu.(block i) +. (l.(i) *. x *. x))
    u;
  let a =
    Array.init (n * n) (fun p ->
        let i = p / n and j = p mod n in
        if block i <> block j then Complex.zero
        else
          let q = block i in
          let h = u.(i) *. u.(j) /. uu.(q) in
          let d = if i = j then l.(i) else 0. in
          let re = d -. (2. *. h *. (l.(i) +. l.(j))) +. (4. *. h *. lu.(q) /. uu.(q)) in
          { Complex.re; im = 0. })
  in
  let x = of_values c.dt [| n; n |] a in
  Array.sort compare l;
  let big = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. l in
  let g = gamma c.dt (16 * n) *. big in
  let spectrum ~msg w =
    let worst = { ratio = 0.; where = "" } in
    Array.iteri
      (fun j x ->
        worse worst ~bound:g ~residual:(Float.abs (w.(j).Complex.re -. x))
          (fun () -> Printf.sprintf "%s: eigenvalue %d: %h, not %h" msg j w.(j).re x))
      l;
    check_worst worst
  in
  match (run b (Eigh false) [| x |], run b (Eigh true) [| x |]) with
  | Some y, Some y' ->
      spectrum ~msg:"alone" (values y.(0));
      spectrum ~msg:"with vectors" (values y'.(0));
      check_eigh c.dt ~e:0 ~count:1 ~n (values x) (values y'.(0))
        (values y'.(1))
  | _ -> ()

(* svd's bound: γ(16 max (m, n) + 100), the 100 for the bidiagonal
   iteration's test of convergence, which drops an e below
   max(10, min(100, u^-1/8)) u times its neighbours. *)
let svd_bound dt ~m ~n = gamma dt ((16 * max m n) + 100)

(* Its absolute term ε(16 max (m, n) + 100), times 2^e. *)
let svd_term ~e dt ~m ~n = epsilon ~e dt ((16 * max m n) + 100)

(* a = u diag s vh with ‖a − u diag s vh‖_F <= g ‖a‖_F + h,
   ‖uᴴu − I‖_F <= g and ‖vh vhᴴ − I‖_F <= g, g = svd_bound and
   h = svd_term, for the [count] matrices of [m] × [n] of [a], u of [uc]
   columns and vh of [vr] rows; s real, descending and not negative. [a]
   and [s] are scaled by 2^e. *)
let check_svd dt ~e ~count ~m ~n ~uc ~vr (a : Complex.t array)
    (s : Complex.t array) (u : Complex.t array) (vh : Complex.t array) =
  let k = min m n and g = svd_bound dt ~m ~n and h = svd_term ~e dt ~m ~n in
  let worst = { ratio = 0.; where = "" } in
  for t = 0 to count - 1 do
    let st j = s.((t * k) + j) in
    for j = 0 to k - 1 do
      equal ~msg:"s is real" int64 0L (Int64.bits_of_float (st j).im);
      if not ((st j).re >= 0.) then failf "matrix %d: s %d is %h" t j (st j).re;
      if j > 0 && not ((st (j - 1)).re >= (st j).re) then
        failf "matrix %d: s not descending at %d" t j
    done;
    let res = ref 0. and norm = ref 0. in
    for i = 0 to m - 1 do
      for j = 0 to n - 1 do
        let aij = at a ~rows:m ~cols:n t i j in
        let r = sum () in
        add r.re aij.re;
        add r.im aij.im;
        for p = 0 to k - 1 do
          add_cprod r
            (Complex.mul (at u ~rows:m ~cols:uc t i p)
               { Complex.re = -.(st p).re; im = 0. })
            (at vh ~rows:vr ~cols:n t p j)
        done;
        res := !res +. Complex.norm2 (total r);
        norm := !norm +. Complex.norm2 aij
      done
    done;
    worse worst ~bound:((g *. sqrt !norm) +. h) ~residual:(sqrt !res)
      (fun () -> Printf.sprintf "matrix %d: a - u diag s vh" t);
    (* ‖xᴴx − I‖_F for the [size] columns of [len] elements [el p i]. *)
    let gram ~msg size len el =
      let o = ref 0. in
      for i = 0 to size - 1 do
        for j = 0 to size - 1 do
          let x = sum () in
          if i = j then add x.re (-1.);
          for p = 0 to len - 1 do
            add_cprod ~conj:true x (el p j) (el p i)
          done;
          o := !o +. Complex.norm2 (total x)
        done
      done;
      worse worst ~bound:g ~residual:(sqrt !o) (fun () ->
          Printf.sprintf "matrix %d: %s" t msg)
    in
    gram ~msg:"uᴴu - I" uc m (fun p i -> at u ~rows:m ~cols:uc t p i);
    gram ~msg:"vh vhᴴ - I" vr n (fun p i -> at vh ~rows:vr ~cols:n t i p)
  done;
  check_worst worst

let svd_columns f ~m ~n =
  let k = min m n in
  if f = Nx_kernel.Complete then (m, n) else (k, k)

(* svd's results meet check_svd's relations, reduced or complete, on
   matrices of fewer, as many and more rows than columns, and the values
   alone lie within 2 (svd_bound ‖a‖_F + svd_term) of the vectors'
   call's. *)
let law_svd (b : Support.backend) c =
  let m, n, x = svd_operand c in
  let f = factors c in
  cover "wide" (m < n);
  cover "square" (m = n);
  cover "tall" (m > n);
  (* nx.cpu's paths: a QR first from 11/6 (17/9 on complex numbers) as
     many rows as columns or the other way, divide and conquer past 25
     values. *)
  let k = min m n and mm = max m n in
  let qr_first = if is_complex c.dt then k * 17 / 9 else k * 11 / 6 in
  cover "a QR first" (k > 0 && mm >= qr_first);
  cover "divide and conquer" (k > 25);
  let x' = laid_out c.how x in
  match (run b (Svd (Some f)) [| x' |], run b (Svd None) [| x' |]) with
  | Some y, Some y' ->
      let d = down c in
      let a = up d (values x) and s = up d (values y.(0)) in
      let uc, vr = svd_columns f ~m ~n in
      check_svd c.dt ~e:d ~count:(count c) ~m ~n ~uc ~vr a s (values y.(1))
        (values y.(2));
      let s' = up d (values y'.(0)) in
      let g = svd_bound c.dt ~m ~n and h = svd_term ~e:d c.dt ~m ~n in
      let worst = { ratio = 0.; where = "" } in
      for t = 0 to count c - 1 do
        let norm = ref 0. in
        for p = 0 to (m * n) - 1 do
          norm := !norm +. Complex.norm2 a.((t * m * n) + p)
        done;
        for j = 0 to k - 1 do
          let p = (t * k) + j in
          worse worst
            ~bound:(2. *. ((g *. sqrt !norm) +. h))
            ~residual:(Float.abs (s.(p).re -. s'.(p).re)) (fun () ->
              Printf.sprintf "matrix %d: value %d alone" t j)
        done
      done;
      check_worst worst
  | _ -> ()

(* Of a = H_m Σ H_n, H_k the reflector I − 2 h hᵀ / hᵀh of order k and Σ
   of [m] × [n] with σ on its diagonal, the singular values are |σ|'s,
   descending, within svd_bound max |σ|, with the vectors and
   without, and the vectors meet their relations; σ draws repeats and
   zeros, which a merge of divide and conquer deflates. *)
let law_svd_spectrum (b : Support.backend) c =
  let rows = c.n in
  let cols = match c.seed mod 3 with 0 -> c.n | 1 -> c.r | _ -> c.n + c.r in
  let m, n = if c.seed / 3 mod 2 = 0 then (rows, cols) else (cols, rows) in
  let k = min m n and r = Random.State.make [| c.seed |] in
  let sigma = Array.init k (fun _ -> Float.round (8. *. uniform r) /. 8.) in
  let reflector size =
    let h = Array.init size (fun _ -> uniform r) in
    let hh = Array.fold_left (fun s x -> s +. (x *. x)) 0. h in
    fun i j ->
      (if i = j then 1. else 0.)
      -. if hh = 0. then 0. else 2. *. h.(i) *. h.(j) /. hh
  in
  let hm = reflector m and hn = reflector n in
  let a =
    Array.init (m * n) (fun p ->
        let i = p / n and j = p mod n in
        let s = dd () in
        for q = 0 to k - 1 do
          add_prod s (hm i q *. sigma.(q)) (hn q j)
        done;
        { Complex.re = s.hi +. s.lo; im = 0. })
  in
  let x = of_values c.dt [| m; n |] a in
  let exact = Array.map Float.abs sigma in
  Array.sort (fun x y -> Float.compare y x) exact;
  let g = svd_bound c.dt ~m ~n *. Array.fold_left Float.max 0. exact in
  let spectrum ~msg s =
    let worst = { ratio = 0.; where = "" } in
    Array.iteri
      (fun j x ->
        worse worst ~bound:g ~residual:(Float.abs (s.(j).Complex.re -. x))
          (fun () -> Printf.sprintf "%s: value %d: %h, not %h" msg j s.(j).re x))
      exact;
    check_worst worst
  in
  let f = factors c in
  match (run b (Svd None) [| x |], run b (Svd (Some f)) [| x |]) with
  | Some y, Some y' ->
      spectrum ~msg:"alone" (values y.(0));
      spectrum ~msg:"with vectors" (values y'.(0));
      let uc, vr = svd_columns f ~m ~n in
      check_svd c.dt ~e:0 ~count:1 ~m ~n ~uc ~vr (values x) (values y'.(0))
        (values y'.(1)) (values y'.(2))
  | _ -> ()

(* eig's bound γ(16n), and its absolute term ε(16n) times 2^e. *)
let eig_bound dt n = gamma dt (16 * n)
let eig_term ~e dt n = epsilon ~e dt (16 * n)

(* On real a, w's values not real come in adjacent conjugate pairs, the
   one of positive imaginary part first, and its real values have +0
   imaginary parts; [v]'s columns, where given, are conjugate in a pair
   and real, with +0 imaginary parts, for a real value. *)
let check_real ~msg ~count ~n (w : Complex.t array) v =
  let b = Int64.bits_of_float in
  for t = 0 to count - 1 do
    let wt j = w.((t * n) + j) in
    let col j = Array.init n (fun i -> v.((((t * n) + i) * n) + j)) in
    let j = ref 0 in
    while !j < n do
      let x = wt !j in
      if x.im = 0. then begin
        equal ~msg:(msg ^ ": a real value's imaginary part") int64 0L (b x.im);
        if v <> [||] then
          Array.iteri
            (fun i (y : Complex.t) ->
              if b y.im <> 0L then
                failf "%s: matrix %d: v (%d, %d) of a real value is %h%+hi"
                  msg t i !j y.re y.im)
            (col !j);
        incr j
      end
      else begin
        if not (x.im > 0. && !j + 1 < n) then
          failf "%s: matrix %d: value %d, %h%+hi, starts no pair" msg t !j
            x.re x.im;
        let pair_bits (z : Complex.t) = (b z.re, b z.im) in
        equal ~msg:(msg ^ ": a pair is conjugate") (pair int64 int64)
          (pair_bits (Complex.conj x)) (pair_bits (wt (!j + 1)));
        if v <> [||] then
          equal ~msg:(msg ^ ": a pair's vectors are conjugate")
            (array (pair int64 int64))
            (Array.map (fun z -> pair_bits (Complex.conj z)) (col !j))
            (Array.map pair_bits (col (!j + 1)));
        j := !j + 2
      end
    done
  done

(* a v = v diag w with each column v_j of norm within g of 1, an element
   of largest magnitude real, positive on complex [dt], and
   ‖a v_j − w_j v_j‖₂ <= g ‖a‖_F + h, g = eig_bound and h = eig_term, for
   the [count] matrices of order [n] of [a]. [a] and [w] are scaled by
   2^e. *)
let check_eig dt ~e ~count ~n (a : Complex.t array) (w : Complex.t array)
    (v : Complex.t array) =
  let g = eig_bound dt n and h = eig_term ~e dt n in
  let worst = { ratio = 0.; where = "" } in
  for t = 0 to count - 1 do
    let norm = ref 0. in
    for p = 0 to (n * n) - 1 do
      norm := !norm +. Complex.norm2 a.((t * n * n) + p)
    done;
    for j = 0 to n - 1 do
      let wj = w.((t * n) + j) and vij i = at v ~rows:n ~cols:n t i j in
      let len = dd () and res = ref 0. in
      for i = 0 to n - 1 do
        let x = vij i in
        add_prod len x.re x.re;
        add_prod len x.im x.im;
        let s = sum () in
        for p = 0 to n - 1 do
          add_cprod s (at a ~rows:n ~cols:n t i p) (vij p)
        done;
        add_cprod s (Complex.neg wj) x;
        res := !res +. Complex.norm2 (total s)
      done;
      worse worst ~bound:g
        ~residual:(Float.abs (sqrt (len.hi +. len.lo) -. 1.))
        (fun () -> Printf.sprintf "matrix %d: ‖v_%d‖ - 1" t j);
      (* Magnitudes within 4 u of the largest tie, as the routine rounds
         them. *)
      let big =
        List.fold_left (fun m i -> Float.max m (Complex.norm (vij i))) 0.
          (List.init n Fun.id)
      in
      let real i =
        let x = vij i in
        Complex.norm x >= big *. (1. -. (4. *. unit_roundoff dt))
        && x.im = 0.
        && ((not (is_complex dt)) || x.re > 0.)
      in
      if n > 0 && not (List.exists real (List.init n Fun.id)) then
        failf "matrix %d: no element of v_%d's largest is real%s" t j
          (if is_complex dt then " and positive" else "");
      worse worst
        ~bound:((g *. sqrt !norm) +. h)
        ~residual:(sqrt !res)
        (fun () -> Printf.sprintf "matrix %d: a v_%d - w_%d v_%d" t j j j)
    done
  done;
  check_worst worst

(* eig's results meet check_eig's relations, and on real a check_real's,
   with the vectors and without, on matrices balancing leaves with D = I,
   so that the bounds read with a itself: some it permutes. *)
let law_eig (b : Support.backend) c =
  cover "mirrored" (c.seed mod 3 = 0);
  cover "lower triangular" (c.seed mod 3 = 1);
  cover "permuted block triangular" (c.seed mod 3 = 2 && c.n > 2);
  let x = laid_out c.how (eig_operand c) in
  match (run b (Eig true) [| x |], run b (Eig false) [| x |]) with
  | Some y, Some y' ->
      let d = down c and n = c.n and count = count c in
      let w = values y.(0) and v = values y.(1) and w' = values y'.(0) in
      check_eig c.dt ~e:d ~count ~n (up d (values x)) (up d w) v;
      let real = not (is_complex c.dt) in
      cover "a real matrix's complex pair"
        (real && Array.exists (fun (z : Complex.t) -> z.im <> 0.) w);
      if real then begin
        check_real ~msg:"with vectors" ~count ~n w v;
        check_real ~msg:"alone" ~count ~n w' [||]
      end
  | _ -> ()

(* Of the normal a = H B H, H the reflector I − 2 u uᵀ / uᵀu and B diagonal
   with values on a grid of eighths, or on reals block diagonal with 2 × 2
   blocks [α β; −β α] for the pairs α ± iβ, each eigenvalue is within
   g ‖a‖_F + h of one of B's, with the vectors and without, and the
   vectors meet check_eig's relations. *)
let law_eig_spectrum (b : Support.backend) c =
  let n = c.n and r = Random.State.make [| c.seed |] in
  let grid () = Float.round (8. *. uniform r) /. 8. in
  let real = not (is_complex c.dt) in
  let l = Array.make n Complex.zero in
  let bm = Array.make (n * n) Complex.zero in
  let set i j re = bm.((i * n) + j) <- { Complex.re; im = 0. } in
  let i = ref 0 in
  while !i < n do
    if real && !i + 1 < n && Random.State.bool r then begin
      let al = grid () and be = Float.abs (grid ()) +. 0.125 in
      set !i !i al;
      set !i (!i + 1) be;
      set (!i + 1) !i (-.be);
      set (!i + 1) (!i + 1) al;
      l.(!i) <- { Complex.re = al; im = be };
      l.(!i + 1) <- { Complex.re = al; im = -.be };
      i := !i + 2
    end
    else begin
      let x = { Complex.re = grid (); im = (if real then 0. else grid ()) } in
      bm.((!i * n) + !i) <- x;
      l.(!i) <- x;
      incr i
    end
  done;
  let u = Array.init n (fun _ -> uniform r) in
  let uu = Array.fold_left (fun s x -> s +. (x *. x)) 0. u in
  let hm i j = (if i = j then 1. else 0.) -. (2. *. u.(i) *. u.(j) /. uu) in
  let mul f g =
    Array.init (n * n) (fun p ->
        let i = p / n and j = p mod n and s = sum () in
        for k = 0 to n - 1 do
          add_cprod s (f i k) (g k j)
        done;
        total s)
  in
  let real_of f i j = { Complex.re = f i j; im = 0. } in
  let hb = mul (real_of hm) (fun i j -> bm.((i * n) + j)) in
  let a = mul (fun i j -> hb.((i * n) + j)) (real_of hm) in
  let x = of_values c.dt [| n; n |] a in
  let norm = sqrt (Array.fold_left (fun s z -> s +. Complex.norm2 z) 0. bm) in
  let bound = (eig_bound c.dt n *. norm) +. eig_term ~e:0 c.dt n in
  let spectrum ~msg w =
    let worst = { ratio = 0.; where = "" } in
    Array.iteri
      (fun j (z : Complex.t) ->
        let near =
          Array.fold_left
            (fun m y -> Float.min m (Complex.norm (Complex.sub z y)))
            infinity l
        in
        worse worst ~bound ~residual:near (fun () ->
            Printf.sprintf "%s: eigenvalue %d, %h%+hi" msg j z.re z.im))
      w;
    check_worst worst
  in
  match (run b (Eig false) [| x |], run b (Eig true) [| x |]) with
  | Some y, Some y' ->
      spectrum ~msg:"alone" (values y.(0));
      spectrum ~msg:"with vectors" (values y'.(0));
      check_eig c.dt ~e:0 ~count:1 ~n (values x) (values y'.(0))
        (values y'.(1))
  | _ -> ()

(* Balancing at work: a = G A G⁻¹ for A tridiagonal [1 2 1] and G =
   diag (2^30, 1, 2^-30), whose eigenvalues are A's, 2 and 2 ± √2. xGEBAL
   balances it with D = diag (2^35, 2^7, 2^-21) and no permutation (as
   LAPACK 3.12 computes it), to b = D⁻¹ a D = S⁻¹ A S, S = diag (2^5, 2^7,
   2^9). eig's results meet the bound on ‖D⁻¹ (a v_j − w_j v_j)‖₂, and its
   eigenvalues, by Bauer and Fike's theorem, are within κ(S) = 16 times
   g ‖b‖_F + h of A's. *)
let test_eig_balanced (b : Support.backend) () =
  let n = 3 and g = [| 30; 0; -30 |] and d = [| 35; 7; -21 |] in
  let am i j = if i = j then 2. else if abs (i - j) = 1 then 1. else 0. in
  let a =
    Array.init (n * n) (fun p ->
        let i = p / n and j = p mod n in
        { Complex.re = Float.ldexp (am i j) (g.(i) - g.(j)); im = 0. })
  in
  let bnorm =
    sqrt
      (Array.fold_left ( +. ) 0.
         (Array.init (n * n) (fun p ->
              let i = p / n and j = p mod n in
              Float.ldexp (am i j) (g.(i) - g.(j) - d.(i) + d.(j)) ** 2.)))
  in
  let exact = [| 2. -. sqrt 2.; 2.; 2. +. sqrt 2. |] in
  List.iter
    (fun dt ->
      let x = of_values dt [| n; n |] a in
      match run b (Eig true) [| x |] with
      | None -> ()
      | Some y ->
          let w = values y.(0) and v = values y.(1) in
          let bound = (eig_bound dt n *. bnorm) +. eig_term ~e:0 dt n in
          let worst = { ratio = 0.; where = "" } in
          for j = 0 to n - 1 do
            let res = ref 0. and len = ref 0. in
            for i = 0 to n - 1 do
              let s = sum () in
              for p = 0 to n - 1 do
                add_cprod s a.((i * n) + p) v.((p * n) + j)
              done;
              add_cprod s (Complex.neg w.(j)) v.((i * n) + j);
              res := !res +. Complex.norm2 (ldexp (-d.(i)) (total s));
              len := !len +. Complex.norm2 (ldexp (-d.(i)) v.((i * n) + j))
            done;
            worse worst ~bound:(bound *. sqrt !len) ~residual:(sqrt !res)
              (fun () ->
                Format.asprintf "%a: D⁻¹ (a v_%d - w_%d v_%d)" pp_dtype dt
                  j j j);
            let near =
              Array.fold_left
                (fun m e ->
                  let e = { Complex.re = e; im = 0. } in
                  Float.min m (Complex.norm (Complex.sub w.(j) e)))
                infinity exact
            in
            worse worst ~bound:(16. *. bound) ~residual:near (fun () ->
                Format.asprintf "%a: eigenvalue %d" pp_dtype dt j)
          done;
          check_worst worst)
    dtypes

(* Matrices whose entries span the exponent range, which eig balances
   before it scales them: [0 2^100; 2^-100 0] at float32, eigenvalues ±1,
   and [1 1e300; 1e-300 1] at float64, eigenvalues 0 and 2. Each balances
   to b, 2 × 2 with entries within a factor 2 of [0 1; 1 0] or [1 1; 1 1],
   ‖b‖_F <= 4 and its eigenvectors' condition at most 3, so that the
   balanced bound puts each eigenvalue within 3 (4 g + h) of the exact
   ones. Scaled first, as xGEEV does, the small entry flushes to zero. *)
let test_eig_extremes (b : Support.backend) () =
  let case dt (a : float array) exact =
    let x = of_values dt [| 2; 2 |] (Array.map (fun re -> { Complex.re; im = 0. }) a) in
    match run b (Eig true) [| x |] with
    | None -> ()
    | Some y ->
        let bound = 3. *. ((4. *. eig_bound dt 2) +. eig_term ~e:0 dt 2) in
        let worst = { ratio = 0.; where = "" } in
        Array.iteri
          (fun j (z : Complex.t) ->
            let near =
              List.fold_left
                (fun m e ->
                  Float.min m (Complex.norm (Complex.sub z { Complex.re = e; im = 0. })))
                infinity exact
            in
            worse worst ~bound ~residual:near (fun () ->
                Format.asprintf "%a: eigenvalue %d, %h%+hi" pp_dtype dt j z.re
                  z.im))
          (values y.(0));
        check_worst worst
  in
  case (D.Any D.Float32) [| 0.; 0x1p100; 0x1p-100; 0. |] [ -1.; 1. ];
  case (D.Any D.Complex64) [| 0.; 0x1p100; 0x1p-100; 0. |] [ -1.; 1. ];
  case (D.Any D.Float64) [| 1.; 1e300; 1e-300; 1. |] [ 0.; 2. ];
  case (D.Any D.Complex128) [| 1.; 1e300; 1e-300; 1. |] [ 0.; 2. ]

(* A graded cycle, which xTREVC3's normalisation of each back-transformed
   column keeps finite: a = D b D⁻¹ of order 30, b bidiagonal with
   b_ii = 1 + i 2^-20 and b_(i, i+1) = 1, closed by b_(29, 0) = 1e-8, and
   D = diag (2^(-30 i)). Every result is a number and each column of v
   has a norm within g of 1. *)
let test_eig_graded_cycle (b : Support.backend) () =
  let n = 30 in
  let bm i j =
    if i = j then 1. +. Float.ldexp (float_of_int i) (-20)
    else if j = i + 1 then 1.
    else if i = n - 1 && j = 0 then 1e-8
    else 0.
  in
  let a =
    Array.init (n * n) (fun p ->
        let i = p / n and j = p mod n in
        { Complex.re = Float.ldexp (bm i j) (30 * (j - i)); im = 0. })
  in
  List.iter
    (fun dt ->
      let x = of_values dt [| n; n |] a in
      match run b (Eig true) [| x |] with
      | None -> ()
      | Some y ->
          let w = values y.(0) and v = values y.(1) in
          let msg = Format.asprintf "%a" pp_dtype dt in
          equal ~msg:(msg ^ ": numbers in w") int (2 * n) (numbers (D.Any D.Complex128) w);
          equal ~msg:(msg ^ ": numbers in v") int (2 * n * n)
            (numbers (D.Any D.Complex128) v);
          let g = eig_bound dt n and worst = { ratio = 0.; where = "" } in
          for j = 0 to n - 1 do
            let len = dd () in
            for i = 0 to n - 1 do
              let z = v.((i * n) + j) in
              add_prod len z.re z.re;
              add_prod len z.im z.im
            done;
            worse worst ~bound:g
              ~residual:(Float.abs (sqrt (len.hi +. len.lo) -. 1.))
              (fun () -> Printf.sprintf "%s: ‖v_%d‖ - 1" msg j)
          done;
          check_worst worst)
    [ D.Any D.Float64; D.Any D.Complex128 ]

(* A complex matrix whose largest element is finite but of a modulus past
   the largest float, diag (0.9 max (1 + i), 1): a matrix of numbers, on
   which svd's and eig's results are no NaN, svd's values +inf and 1.
   Scaled by the modulus, the routines would scale by 0 and give NaN. *)
let test_overflowing_modulus (b : Support.backend) () =
  let big = 0.9 *. Float.max_float in
  let a =
    [| { Complex.re = big; im = big }; Complex.zero; Complex.zero; Complex.one |]
  in
  let dt = D.Any D.Complex128 in
  let x = of_values dt [| 2; 2 |] a in
  List.iter
    (fun (msg, r) ->
      match run b r [| x |] with
      | Some y ->
          Array.iteri
            (fun k y ->
              let v = values y in
              equal ~msg:(Printf.sprintf "%s: numbers in result %d" msg k) int
                (2 * Array.length v) (numbers dt v))
            y
      | None -> ())
    [ ("svd", Svd (Some Reduced)); ("eig", Eig true) ];
  match run b (Svd None) [| x |] with
  | Some y ->
      let s = values y.(0) in
      equal ~msg:"svd's values" (array float_exact) [| infinity; 1. |]
        (Array.map (fun (z : Complex.t) -> z.re) s)
  | None -> ()

(* The Jacobi matrix whose eigenvalues are [l] and whose eigenvectors'
   first components are [q], ‖q‖ = 1, by Lanczos on diag l from q,
   reorthogonalised twice against every earlier vector: its diagonal and
   subdiagonal. *)
let jacobi l q =
  let n = Array.length l in
  let dot x y = Array.fold_left ( +. ) 0. (Array.map2 ( *. ) x y) in
  let vs = Array.make n q in
  let alpha = Array.make n 0. and beta = Array.make (max 0 (n - 1)) 0. in
  for j = 0 to n - 1 do
    let w = Array.mapi (fun i x -> l.(i) *. x) vs.(j) in
    alpha.(j) <- dot vs.(j) w;
    for _ = 1 to 2 do
      for p = 0 to j do
        let c = dot vs.(p) w in
        Array.iteri (fun i x -> w.(i) <- w.(i) -. (c *. x)) vs.(p)
      done
    done;
    if j < n - 1 then begin
      beta.(j) <- sqrt (dot w w);
      vs.(j + 1) <- Array.map (fun x -> x /. beta.(j)) w
    end
  done;
  (alpha, beta)

(* Reference-LAPACK issue 1166's secular equation, 29 values and its 23rd
   root next to its pole, which LAPACK 3.12's laed4 does not solve in its 30
   steps, as the one merge of a tridiagonal matrix of order 30: halves of
   15, Jacobi matrices from their values and parts of z, the first's
   reversed so that its last row carries them, coupled by ρ / 2. The
   value 1.68e-5 sits in both halves, its weight split so that they weigh
   alike, and the merge deflates the two back into one. eigh's results
   meet check_eigh's relations. *)
let test_eigh_secular (b : Support.backend) () =
  let d =
    [| -8.9907411945275305e-17; 3.1527281176230556e-06; 7.8104209108493992e-06;
       1.4528378412105169e-05; 1.6571840586254615e-05; 1.6791959834500457e-05;
       1.6951003434697501e-05; 1.6961635049587819e-05; 1.6969218460422806e-05;
       1.7028037406879576e-05; 1.9135232081591496e-05; 2.1876238530929982e-05;
       2.1920146416725336e-05; 3.9357203097119847e-05; 1.0557423781645928e-04;
       1.3686127039388258e-04; 1.3686300570614894e-04; 1.3686313427158908e-04;
       1.3686314099616321e-04; 1.3686314127052412e-04; 1.3686314210337733e-04;
       1.3686314231692531e-04; 1.3686314257213152e-04; 5.3532577970471340e-02;
       9.9996866934433004e-01; 1.0000000849390769e+00; 1.0000000901678945e+00;
       1.0000000918404301e+00; 1.0000001095998177e+00 |]
  and z =
    [| 1.6700411548440888e-07; 2.2704845744451106e-06; -6.8596999842130412e-06;
       -1.9813464524908256e-04; 3.4676472662621390e-03; 5.8131816598176476e-02;
       -2.4389840480820292e-03; 2.2571085689296944e-03; -2.0150012240231657e-03;
       -9.7432047705138750e-03; 7.3184880793391433e-04; -2.9319792346511997e-02;
       8.3364827297377855e-07; -8.9722126354645391e-09; 7.0412738449665979e-01;
       -1.4042299148200350e-06; 3.8731670039916887e-06; -5.4330381400333468e-07;
       4.4026694557354993e-07; 2.2528424563293299e-07; 3.6064772311683760e-06;
       2.8472995783701232e-07; 8.3086036055888118e-07; 2.8464273601162125e-02;
       -7.0642255076744132e-01; -5.2009238883566685e-08; 8.6348330718758654e-08;
       -2.9664496252475970e-08; 1.4932776629316714e-09 |]
  and rho = 6.2964808955495258e-05 in
  (* The first half: d's 9th to 22nd values and the split one, 5th; the
     second: the others. Its weights z² sum to half the whole's. *)
  let split = 5 in
  let first = Array.append (Array.init 14 (fun i -> 9 + i)) [| split |] in
  let second = Array.of_list (List.filter (fun i -> i < 9 || i > 22) (List.init 29 Fun.id)) in
  let w = Array.map (fun x -> x *. x) z in
  let total = Array.fold_left ( +. ) 0. w in
  let part ix = Array.fold_left (fun s i -> if i = split then s else s +. w.(i)) 0. ix in
  let x = (total /. 2.) -. part first in
  let weights ix share =
    let v = Array.map (fun i -> if i = split then share else w.(i)) ix in
    let t = Array.fold_left ( +. ) 0. v in
    Array.map (fun x -> sqrt (x /. t)) v
  in
  let a1, b1 = jacobi (Array.map (fun i -> d.(i)) first) (weights first x) in
  let a2, b2 =
    jacobi (Array.map (fun i -> d.(i)) second) (weights second (w.(split) -. x))
  in
  let rev v = Array.init (Array.length v) (fun i -> v.(Array.length v - 1 - i)) in
  let coupling = rho *. total /. 2. in
  let diag = Array.append (rev a1) a2 and off = Array.concat [ rev b1; [| coupling |]; b2 ] in
  diag.(14) <- diag.(14) +. coupling;
  diag.(15) <- diag.(15) +. coupling;
  let n = 30 in
  let a =
    Array.init (n * n) (fun p ->
        let i = p / n and j = p mod n in
        let v = if i = j then diag.(i) else if abs (i - j) = 1 then off.(min i j) else 0. in
        { Complex.re = v; im = 0. })
  in
  let dt = D.Any D.Float64 in
  let x = of_values dt [| n; n |] a in
  match run b (Eigh true) [| x |] with
  | Some y ->
      check_eigh dt ~e:0 ~count:1 ~n (values x) (values y.(0)) (values y.(1))
  | None -> ()

(* Reference-LAPACK issue 255's upper bidiagonal of order 1000, its
   diagonal then its superdiagonal in golden/linalg, on which the merges'
   deflation at 8 u loses u's and vh's orthogonality (#1286 raised it to
   64 u). svd's results meet its bound element by element: |(uᵀu − I)_ij|,
   |(vh vhᵀ − I)_ij| <= g and |(a − u diag s vh)_ij| <= g ‖a‖_F, sums of
   1000 products whose rounding, under 1000 u, is far below g. *)
let test_svd_deflation (b : Support.backend) () =
  let n = 1000 in
  let v =
    In_channel.with_open_text "golden/linalg/bidiagonal-255.txt"
      In_channel.input_lines
    |> List.map float_of_string |> Array.of_list
  in
  let a = Array.make (n * n) 0. in
  for i = 0 to n - 1 do
    a.((i * n) + i) <- v.(i);
    if i < n - 1 then a.((i * n) + i + 1) <- v.(n + i)
  done;
  let x = A.Any (A.of_array D.Float64 [| n; n |] a) in
  match run b (Svd (Some Reduced)) [| x |] with
  | None -> ()
  | Some y ->
      let floats (A.Any x) : float array =
        match A.dtype x with D.Float64 -> A.to_array x | _ -> assert false
      in
      let s = floats y.(0) and u = floats y.(1) and vh = floats y.(2) in
      let g = svd_bound (D.Any D.Float64) ~m:n ~n in
      let norm = sqrt (Array.fold_left (fun t x -> t +. (x *. x)) 0. a) in
      let ut = Array.init (n * n) (fun p -> u.((p mod n * n) + (p / n))) in
      let w = { ratio = 0.; where = "" } in
      (* Element (i, j) of x xᵀ − I for the rows of [x]. *)
      let gram msg x =
        for i = 0 to n - 1 do
          for j = 0 to i do
            let t = ref (if i = j then -1. else 0.) in
            for p = 0 to n - 1 do
              t := !t +. (x.((i * n) + p) *. x.((j * n) + p))
            done;
            worse w ~bound:g ~residual:(Float.abs !t) (fun () ->
                Printf.sprintf "%s, element (%d, %d)" msg i j)
          done
        done
      in
      gram "uᵀu - I" ut;
      gram "vh vhᵀ - I" vh;
      let r = Array.make n 0. in
      for i = 0 to n - 1 do
        Array.blit a (i * n) r 0 n;
        for p = 0 to n - 1 do
          let us = u.((i * n) + p) *. s.(p) in
          for j = 0 to n - 1 do
            r.(j) <- r.(j) -. (us *. vh.((p * n) + j))
          done
        done;
        for j = 0 to n - 1 do
          worse w ~bound:(g *. norm) ~residual:(Float.abs r.(j)) (fun () ->
              Printf.sprintf "a - u diag s vh, element (%d, %d)" i j)
        done
      done;
      check_worst w

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
let check_alone (b : Support.backend) r ~dt ?(result = dt) ~rows ~cols ~spoil
    ops =
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
        bits (of_values result [| rows; cols |] (matrix yv ~rows ~cols m))
      in
      equal ~msg:"the first is alone" same_bits want (mat 0);
      equal ~msg:"the last is alone" same_bits want (mat 2);
      equal ~msg:"numbers in the failing one" int 0
        (numbers result (matrix yv ~rows ~cols 1))
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
    List.iter
      (fun r -> check_alone b r ~dt:c.dt ~rows:c.n ~cols:1 ~spoil [| eigh_operand c |])
      [ Eigh true; Eigh false ]

(* NaN or an infinity anywhere. *)
let law_eig_fails (b : Support.backend) c =
  let c = one c in
  if c.n > 0 then
    let p = c.seed mod (c.n * c.n) in
    let bad = [| Float.nan; Float.infinity; Float.neg_infinity |].(c.seed mod 3) in
    let spoil v =
      let v = Array.copy v in
      v.(p) <- { Complex.re = bad; im = 0. };
      v
    in
    List.iter
      (fun r ->
        check_alone b r ~dt:c.dt ~result:(complex_of c.dt) ~rows:c.n ~cols:1
          ~spoil [| eig_operand c |])
      [ Eig true; Eig false ]

(* NaN or an infinity anywhere. *)
let law_svd_fails (b : Support.backend) c =
  let c = one c in
  let m, n, x = svd_operand c in
  if m > 0 && n > 0 then
    let p = c.seed mod (m * n) in
    let bad = [| Float.nan; Float.infinity; Float.neg_infinity |].(c.seed / 7 mod 3) in
    let spoil v =
      let v = Array.copy v in
      v.(p) <- { Complex.re = bad; im = 0. };
      v
    in
    List.iter
      (fun r -> check_alone b r ~dt:c.dt ~rows:(min m n) ~cols:1 ~spoil [| x |])
      [ Svd (Some (factors c)); Svd None ]

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
  let x = eig_operand c in
  (match (run b (Eig true) [| x |], run b (Eig true) [| laid_out c.how x |]) with
  | Some y, Some z ->
      equal ~msg:"eig's w" same_bits (bits y.(0)) (bits z.(0));
      equal ~msg:"eig's v" same_bits (bits y.(1)) (bits z.(1))
  | _ -> ());
  let _, _, x = svd_operand c in
  let f = Svd (Some (factors c)) in
  (match (run b f [| x |], run b f [| laid_out c.how x |]) with
  | Some y, Some z ->
      equal ~msg:"svd's s" same_bits (bits y.(0)) (bits z.(0));
      equal ~msg:"svd's u" same_bits (bits y.(1)) (bits z.(1));
      equal ~msg:"svd's vh" same_bits (bits y.(2)) (bits z.(2))
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
      ("svd", Svd (Some (factors c)), [| (let _, _, x = svd_operand c in x) |]);
      ("eig", Eig true, [| eig_operand c |]);
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
  | Svd None -> "svd's values"
  | Svd (Some Reduced) -> "svd reduced"
  | Svd (Some Complete) -> "svd complete"
  | Eig true -> "eig"
  | Eig false -> "eig's values"
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
      Result.map (fun (w, x) -> if v then [| w; x |] else [| w |]) (S.eigh (Vectors ()) c.a)
  | Svd None -> Result.map (fun (s, ()) -> [| s |]) (S.svd No_vectors c.a)
  | Svd (Some f) ->
      Result.map (fun (s, (u, vh)) -> [| s; u; vh |]) (S.svd (Vectors f) c.a)
  | Eig v ->
      Result.map
        (fun (w, x) -> if v then [| w; x |] else [| w |])
        (S.eig (Vectors ()) c.a)
  | Solve _ -> Result.map (fun y -> [| y |]) (S.solve_triangular c.a c.b)

let arity = function
  | Cholesky _ | Solve _ | Eigh false | Svd None | Eig false -> 1
  | Lu | Svd (Some _) -> 3
  | Qr _ | Eigh true | Eig true -> 2

(* Square matrices half the time, and a vector now and then. *)
let misfits =
  Gen.with_pp pp_misfit
    (let open Gen in
     let solve =
       Solve { triangle = Lower; adjoint = false; unit_diagonal = false }
     in
     let* routine =
       of_list
         [
           Cholesky Lower;
           Lu;
           Qr Reduced;
           Qr Complete;
           Eigh true;
           Eigh false;
           Svd None;
           Svd (Some Reduced);
           Svd (Some Complete);
           Eig true;
           Eig false;
           solve;
         ]
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
  let cdst i = A.create Rig.host D.Complex64 c.dsts.(i) in
  let a = zeros c.a in
  let got =
    match c.routine with
    | Cholesky t -> K.cholesky t ~dst:(dst 0) a
    | Lu -> K.lu ~lu:(dst 0) ~pivots:(ints 1) ~perm:(ints 2) a
    | Qr _ -> K.qr ~q:(dst 0) ~r:(dst 1) a
    | Eigh true -> K.eigh ~w:(dst 0) ~v:(dst 1) a
    | Eigh false -> K.eigh ~w:(dst 0) a
    | Svd None -> K.svd ~s:(dst 0) a
    | Svd (Some _) -> K.svd ~s:(dst 0) ~vectors:(~u:(dst 1), ~vh:(dst 2)) a
    | Eig true -> K.eig ~w:(cdst 0) ~v:(cdst 1) a
    | Eig false -> K.eig ~w:(cdst 0) a
    | Solve { triangle; adjoint; unit_diagonal } ->
        K.solve_triangular triangle ~adjoint ~unit_diagonal ~dst:(dst 0) a
          (zeros c.b)
  in
  (* A QR's or an SVD's destinations fit as either factors' shapes. *)
  let either r =
    List.exists
      (fun f -> shapes { c with routine = r f } = Ok c.dsts)
      [ Nx_kernel.Reduced; Complete ]
  in
  let fit =
    match c.routine with
    | Qr _ -> either (fun f -> Qr f)
    | Svd (Some _) -> either (fun f -> Svd (Some f))
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
    q 7.

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
      prop "svd's results meet their relations within γ(16 max (m, n) + 100)"
        case (run (law_svd b));
      prop "svd's values are within γ(16 max (m, n) + 100) ‖a‖₂ of the exact ones"
        case (run (law_svd_spectrum b));
      prop "svd of a matrix with NaN or an infinity is NaN alone" case
        (run (law_svd_fails b));
      prop "eig's results meet their relations within γ(16n)" case
        (run (law_eig b));
      prop "eig's eigenvalues of a normal matrix are within γ(16n) ‖a‖_F"
        case (run (law_eig_spectrum b));
      prop "eig of a matrix with NaN or an infinity is NaN alone" case
        (run (law_eig_fails b));
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
      test "eigh solves Reference-LAPACK issue 1166's secular equation"
        (fun () -> b.around (test_eigh_secular b));
      test "eig balances a graded matrix" (fun () ->
          b.around (test_eig_balanced b));
      test "eig balances before it scales" (fun () ->
          b.around (test_eig_extremes b));
      test "eig keeps a graded cycle's vectors finite" (fun () ->
          b.around (test_eig_graded_cycle b));
      test "svd and eig scale an element whose modulus overflows" (fun () ->
          b.around (test_overflowing_modulus b));
      test "svd keeps Reference-LAPACK issue 255's bidiagonal's vectors"
        (fun () -> b.around (test_svd_deflation b));
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
      test "refuses dtypes and shapes" (fun () ->
          b.around (test_cpu_refusals b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.linalg"
       (List.map laws Support.backends @ List.map cpu Support.cpus))
