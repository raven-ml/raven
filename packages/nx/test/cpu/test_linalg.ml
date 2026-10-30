(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's factorisations and triangular solves, as nx_cpu.mli states
   them, by the promise each group holds: each result against its defining
   relation within its bound, by residuals computed in double-double, on
   operands well and badly conditioned, below the least normal number and
   near the largest, at orders past its reductions' crossovers and on
   matrices its algorithms find hard; the Upper factor's bits; the same bits
   under every layout of the same values, in a batch and on one thread as on
   the job's; a solve into its operand; its refusals; and no byte read or
   written past an operand or a result. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move
module Support = Nx_kernels_support
open Linalg_cases

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

(* Values *)

(* [x]'s values laid out another way: C-contiguous, a copy whose last two
   axes are swapped in memory viewed back through a permutation, every
   other element of a copy twice as wide, the first half of each row of a
   copy twice as wide, or a copy whose columns lie in reverse viewed back
   through a negative stride. *)
let laid_out how (A.Any x) =
  let s = L.shape (A.layout x) in
  let r = Iarray.length s in
  let into y =
    A.transfer ~src:x ~dst:y;
    A.Any y
  in
  match how with
  | Contiguous -> A.Any x
  | Transposed ->
      let swap i = if i = r - 2 then r - 1 else if i = r - 1 then r - 2 else i in
      let perm = Array.init r swap in
      let shape = Iarray.init r (fun i -> Iarray.get s perm.(i)) in
      let t = A.create Rig.host (A.dtype x) shape in
      into (Option.get (A.move (M.Permute (Iarray.of_array perm)) t))
  | Stepped ->
      let wide = Iarray.mapi (fun i d -> if i = r - 1 then 2 * d else d) s in
      let w = A.create Rig.host (A.dtype x) wide in
      let range i d = { M.start = 0; count = d; step = (if i = r - 1 then 2 else 1) } in
      let ranges = Iarray.mapi range s in
      into (Option.get (A.move (M.Slice ranges) w))
  | Padded ->
      let wide = Iarray.mapi (fun i d -> if i = r - 1 then 2 * d else d) s in
      let w = A.create Rig.host (A.dtype x) wide in
      let range _ d = { M.start = 0; count = d; step = 1 } in
      let ranges = Iarray.mapi range s in
      into (Option.get (A.move (M.Slice ranges) w))
  | Reversed ->
      let t = A.create Rig.host (A.dtype x) s in
      let range i d =
        if i = r - 1 && d > 0 then { M.start = d - 1; count = d; step = -1 }
        else { M.start = 0; count = d; step = 1 }
      in
      let ranges = Iarray.mapi range s in
      into (Option.get (A.move (M.Slice ranges) t))

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
   imaginary parts. A pair of reals adds its one product: the other three
   are zeros, which leave a sum's value as it is. *)
type sum = { re : dd; im : dd }

let sum () = { re = dd (); im = dd () }

let add_cprod ?(conj = false) s (x : Complex.t) (y : Complex.t) =
  if x.im = 0. && y.im = 0. then add_prod s.re x.re y.re
  else begin
    let yi = if conj then -.y.im else y.im in
    add_prod s.re x.re y.re;
    add_prod s.re (-.x.im) yi;
    add_prod s.im x.re yi;
    add_prod s.im x.im y.re
  end

let total s = { Complex.re = s.re.hi +. s.re.lo; im = s.im.hi +. s.im.lo }

(* Draws *)

(* Element (i, j) of matrix [b] of the [rows] × [cols] matrices [v]. *)
let at v ~rows ~cols b i j = v.((((b * rows) + i) * cols) + j)

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

(* a = q r with, per column, ‖a_j − q r_j‖ <= g ‖a_j‖ + √m h; each element
   of qᴴq − I at most g, g = γ(k(2m + 10)) and h = ε(k(2m + 10)); r +0
   below its diagonal and real on it, for the [count] matrices of [m] × [n]
   of [a], q of [qc] columns. [a] and [r] are scaled by 2^e. *)
let check_qr dt ~e ~count ~m ~n ~qc (a : Complex.t array) (q : Complex.t array)
    (r : Complex.t array) =
  let g = gamma dt (min m n * ((2 * m) + 10)) in
  let h = epsilon ~e dt (min m n * ((2 * m) + 10)) in
  let w = { ratio = 0.; where = "" } in
  for t = 0 to count - 1 do
    for i = 0 to qc - 1 do
      for j = 0 to n - 1 do
        if j < i then
          equal ~msg:"r below its diagonal" (pair int64 int64) (0L, 0L)
            (let v = at r ~rows:qc ~cols:n t i j in
             (Int64.bits_of_float v.re, Int64.bits_of_float v.im));
        (* Real: a zero imaginary part of either sign, as an operand's
           −0 passes through a reflector that leaves it. *)
        if j = i then
          equal ~msg:"r's diagonal is real"
            (Testable.make ~pp:(fun ppf x -> Format.fprintf ppf "%h" x)
               ~equal:( = ))
            0. (at r ~rows:qc ~cols:n t i j).im
      done
    done;
    (* r is zero below its diagonal: column j of q r sums its first
       j + 1 products. *)
    for j = 0 to n - 1 do
      let res = ref 0. and norm = ref 0. in
      for i = 0 to m - 1 do
        let s = sum () in
        let aij = at a ~rows:m ~cols:n t i j in
        add s.re aij.re;
        add s.im aij.im;
        for p = 0 to min j (qc - 1) do
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
    (* qᴴq − I is Hermitian: its lower triangle bounds every element. *)
    for i = 0 to qc - 1 do
      for j = 0 to i do
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

(* qr's results meet check_qr's relations, reduced or complete. A subnormal
   a is checked as 2^d a = q (2^d r). *)
let law_qr (b : Support.backend) c =
  let cols, x = lu_operand c in
  let f = factors c in
  match run b (Qr f) [| laid_out c.how x |] with
  | None -> ()
  | Some y ->
      let d = down c and m = c.n and n = cols in
      let qc = if f = Complete then m else min m n in
      check_qr c.dt ~e:d ~count:(count c) ~m ~n ~qc (up d (values x))
        (values y.(0))
        (up d (values y.(1)))

(* a v = v diag w with ‖a − v diag w vᴴ‖_F <= g ‖a‖_F + h and
   ‖vᴴv − I‖_F <= g, g = γ(16n) and h = ε(16n), for the [count] Hermitian
   matrices of order [n] of [a]; w ascending and real. [a] and [w] are
   scaled by 2^e. *)
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
    (* Both differences are Hermitian: their lower triangles' elements off
       the diagonal count twice. *)
    let res = ref 0. and norm = ref 0. and orth = ref 0. in
    let vw =
      Array.init (n * n) (fun q ->
          let i = q / n and p = q mod n in
          Complex.mul (at v ~rows:n ~cols:n t i p)
            { Complex.re = -.(wt p).re; im = 0. })
    in
    for i = 0 to n - 1 do
      for j = 0 to i do
        let aij = at a ~rows:n ~cols:n t i j in
        let s = sum () and o = sum () in
        add s.re aij.re;
        add s.im aij.im;
        if i = j then add o.re (-1.);
        for p = 0 to n - 1 do
          add_cprod ~conj:true s vw.((i * n) + p) (at v ~rows:n ~cols:n t j p);
          add_cprod ~conj:true o
            (at v ~rows:n ~cols:n t p i)
            (at v ~rows:n ~cols:n t p j)
        done;
        let twice = if i = j then 1. else 2. in
        res := !res +. (twice *. Complex.norm2 (total s));
        orth := !orth +. (twice *. Complex.norm2 (total o));
        norm := !norm +. (twice *. Complex.norm2 aij)
      done
    done;
    worse worst ~bound:((g *. sqrt !norm) +. h) ~residual:(sqrt !res)
      (fun () -> Printf.sprintf "matrix %d: a - v diag w vᴴ" t);
    worse worst ~bound:g ~residual:(sqrt !orth) (fun () ->
        Printf.sprintf "matrix %d: vᴴv - I" t)
  done;
  check_worst worst

(* check_eigh's relations, for one matrix too large for its n³ steps, on
   [k] elements of each difference drawn by [r], each at most the bound
   on its difference's Frobenius norm. *)
let check_eigh_sampled dt ~n ~k r (a : Complex.t array) (w : Complex.t array)
    (v : Complex.t array) =
  let g = gamma dt (16 * n) and h = epsilon ~e:0 dt (16 * n) in
  let norm = Array.fold_left (fun m x -> m +. Complex.norm2 x) 0. a in
  let worst = { ratio = 0.; where = "" } in
  for j = 0 to n - 1 do
    equal ~msg:"w is real" int64 0L (Int64.bits_of_float w.(j).im);
    if j > 0 && not (w.(j - 1).re <= w.(j).re) then
      failf "w not ascending at %d" j
  done;
  for _ = 1 to k do
    let i = Random.State.int r n and j = Random.State.int r n in
    let s = sum () and o = sum () in
    add s.re a.((i * n) + j).re;
    add s.im a.((i * n) + j).im;
    if i = j then add o.re (-1.);
    for p = 0 to n - 1 do
      add_cprod ~conj:true s
        (Complex.mul v.((i * n) + p) { Complex.re = -.w.(p).re; im = 0. })
        v.((j * n) + p);
      add_cprod ~conj:true o v.((p * n) + i) v.((p * n) + j)
    done;
    worse worst ~bound:((g *. sqrt norm) +. h)
      ~residual:(Complex.norm (total s)) (fun () ->
        Printf.sprintf "element (%d, %d) of a - v diag w vᴴ" i j);
    worse worst ~bound:g ~residual:(Complex.norm (total o)) (fun () ->
        Printf.sprintf "element (%d, %d) of vᴴv - I" i j)
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

(* What check_spectrum checks of the vectors: nothing, their relations,
   or [k] elements of each relation's difference. *)
type vectors_check = Unchecked | Checked | Sampled of int

(* [n] draws of [r] for a reflector of [dt]: complex on complex dtypes. *)
let reflector_draws r dt n =
  Array.init n (fun _ ->
      let re = uniform r in
      { Complex.re; im = (if is_complex dt then uniform r else 0.) })

(* a = H diag λ H of [dt], H the product of the reflectors I − 2 u uᴴ /
   uᴴu of [u] on rows [0, k) and on rows [k, n): Hermitian, its
   eigenvalues [l]'s, its two blocks uncoupled. Element (i, j) of a block
   is λ_i δ_ij − 2 h (λ_i + λ_j) + 4 h Σ λ_p |u_p|² / uᴴu with h = u_i
   conj u_j / uᴴu. *)
let spectral dt ~k ~(u : Complex.t array) l =
  let n = Array.length l in
  let block i = if i < k then 0 else 1 in
  let uu = [| 0.; 0. |] and lu = [| 0.; 0. |] in
  Array.iteri
    (fun i x ->
      let x2 = Complex.norm2 x in
      uu.(block i) <- uu.(block i) +. x2;
      lu.(block i) <- lu.(block i) +. (l.(i) *. x2))
    u;
  let a =
    Array.init (n * n) (fun p ->
        let i = p / n and j = p mod n in
        if block i <> block j then Complex.zero
        else
          let q = block i in
          let h = Complex.mul u.(i) (Complex.conj u.(j)) in
          let s = (-2. *. (l.(i) +. l.(j))) +. (4. *. lu.(q) /. uu.(q)) in
          let d = if i = j then l.(i) else 0. in
          let f = s /. uu.(q) in
          { Complex.re = d +. (f *. h.re); im = f *. h.im })
  in
  of_values dt [| n; n |] a

(* Of spectral's a, the eigenvalues are [l]'s within γ(16n) max |λ|,
   ascending, alone and, unless [Unchecked], with the vectors, which meet
   their relations. *)
let check_spectrum (b : Support.backend) ?(vectors = Checked) dt ~k ~u l =
  let n = Array.length l in
  let x = spectral dt ~k ~u l in
  let l = Array.copy l in
  Array.sort compare l;
  let big = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. l in
  let g = gamma dt (16 * n) *. big in
  let spectrum ~msg w =
    let worst = { ratio = 0.; where = "" } in
    Array.iteri
      (fun j x ->
        worse worst ~bound:g ~residual:(Float.abs (w.(j).Complex.re -. x))
          (fun () -> Printf.sprintf "%s: eigenvalue %d: %h, not %h" msg j w.(j).re x))
      l;
    check_worst worst
  in
  Option.iter (fun y -> spectrum ~msg:"alone" (values y.(0)))
    (run b (Eigh false) [| x |]);
  let relations y =
    match vectors with
    | Unchecked -> ()
    | Checked ->
        check_eigh dt ~e:0 ~count:1 ~n (values x) (values y.(0))
          (values y.(1))
    | Sampled k ->
        check_eigh_sampled dt ~n ~k (Random.State.make [| n |]) (values x)
          (values y.(0)) (values y.(1))
  in
  if vectors <> Unchecked then
    Option.iter
      (fun y ->
        spectrum ~msg:"with vectors" (values y.(0));
        relations y)
      (run b (Eigh true) [| x |])

(* λ draws repeats, which a merge of divide and conquer deflates. *)
let law_eigh_spectrum (b : Support.backend) c =
  let n = c.n and r = Random.State.make [| c.seed |] in
  let l = Array.init n (fun _ -> Float.round (8. *. uniform r) /. 8.) in
  let u = reflector_draws r c.dt n in
  check_spectrum b c.dt ~k:(c.seed mod (n + 1)) ~u l

(* [r] of [x] alone, of [x] laid out Transposed, and of a batch of two
   copies of [x] on one thread ({!Nx_cpu.one_thread}): each matrix then runs
   as a unit of a batch with at least the call's threads does, the second at
   an offset, on the arena the first used. Every result has the bits of the
   call's alone. *)
let agree_bits (b : Support.backend) r x =
  let (A.Any a) = x in
  let s = Iarray.append [| 2 |] (L.shape (A.layout a)) in
  let pair = A.Any (A.copy (Option.get (A.move (M.Broadcast s) a))) in
  let z = run b r [| laid_out Transposed x |] in
  let w = Nx_cpu.one_thread (fun () -> run b r [| pair |]) in
  match (run b r [| x |], z, w) with
  | Some y, Some z, Some w ->
      Array.iteri
        (fun j y ->
          let want = bits y in
          let n = Array.length want in
          equal ~msg:(Printf.sprintf "result %d, transposed" j) same_bits want
            (bits z.(j));
          let got = bits w.(j) in
          for i = 0 to 1 do
            equal
              ~msg:(Printf.sprintf "result %d, matrix %d of the batch" j i)
              same_bits want (Array.sub got (i * n) n)
          done)
        y
  | _ -> ()

(* Spectra a large order meets: values that repeat, that cluster 2^-30
   apart, which divide and conquer barely separates in float64, or that
   grade over 60 binades. *)
let repeating r _ = Float.round (8. *. uniform r) /. 8.

let clustered r j =
  if j mod 2 = 0 then 1. +. (float_of_int j *. 0x1p-30) else uniform r

let graded r j = Float.copy_sign (Float.ldexp 1. (-(j mod 61))) (uniform r)

(* One order past each crossover to nx.cpu's two-stage reduction, where
   the bounds hold as below it: on x86-64 from 128, for values alone of
   either kind and real vectors; on arm64 from 512 for values alone, 1024
   for complex vectors and 2048 for real ones, which x86-64 is past at
   129, so they run on arm64 alone. Complex values alone at 513 are past
   both. The relations are checked whole at 129 and on sampled elements
   past it, and the bits agree whatever the threads and layout up to
   1025. *)
let test_eigh_large (b : Support.backend) () =
  let r = Random.State.make [| 7 |] in
  let orders =
    D.
      [
        (129, Any Float64, graded r, Checked, true);
        (513, Any Complex128, repeating r, Unchecked, true);
      ]
  and arm64 =
    D.
      [
        (1025, Any Complex64, clustered r, Sampled 256, true);
        (2049, Any Float64, repeating r, Sampled 256, false);
      ]
  in
  List.iter
    (fun (n, dt, lambda, vectors, threads) ->
      let l = Array.init n lambda and u = reflector_draws r dt n in
      check_spectrum b ~vectors dt ~k:(n / 3) ~u l;
      if threads then
        agree_bits b (Eigh (vectors <> Unchecked)) (spectral dt ~k:(n / 3) ~u l))
    (if Support.x86_64 then orders else orders @ arm64)

(* svd's bound: γ(16 max (m, n) + 100), the 100 for the bidiagonal
   iteration's test of convergence, which drops an e below
   max(10, min(100, u^-1/8)) u times its neighbours. *)
let svd_bound dt ~m ~n = gamma dt ((16 * max m n) + 100)

(* Its absolute term ε(16 max (m, n) + 100), times 2^e. *)
let svd_term ~e dt ~m ~n = epsilon ~e dt ((16 * max m n) + 100)

(* a = u diag s vh with ‖a − u diag s vh‖_F <= g ‖a‖_F + h,
   ‖uᴴu − I‖_F <= g and ‖vh vhᴴ − I‖_F <= g, g = svd_bound and
   h = svd_term, for the [count] matrices of [m] × [n] of [a], u of [uc]
   columns and vh of [vr] rows; s real, descending and not negative, a zero
   one +0. [a] and [s] are scaled by 2^e. *)
let check_svd dt ~e ~count ~m ~n ~uc ~vr (a : Complex.t array)
    (s : Complex.t array) (u : Complex.t array) (vh : Complex.t array) =
  let k = min m n and g = svd_bound dt ~m ~n and h = svd_term ~e dt ~m ~n in
  let worst = { ratio = 0.; where = "" } in
  for t = 0 to count - 1 do
    let st j = s.((t * k) + j) in
    for j = 0 to k - 1 do
      equal ~msg:"s is real" int64 0L (Int64.bits_of_float (st j).im);
      if not ((st j).re >= 0.) then failf "matrix %d: s %d is %h" t j (st j).re;
      if (st j).re = 0. then
        equal ~msg:(Printf.sprintf "matrix %d: a zero s %d is +0" t j) int64 0L
          (Int64.bits_of_float (st j).re);
      if j > 0 && not ((st (j - 1)).re >= (st j).re) then
        failf "matrix %d: s not descending at %d" t j
    done;
    let res = ref 0. and norm = ref 0. in
    let us =
      Array.init (m * k) (fun q ->
          let i = q / k and p = q mod k in
          Complex.mul (at u ~rows:m ~cols:uc t i p)
            { Complex.re = -.(st p).re; im = 0. })
    in
    for i = 0 to m - 1 do
      for j = 0 to n - 1 do
        let aij = at a ~rows:m ~cols:n t i j in
        let r = sum () in
        add r.re aij.re;
        add r.im aij.im;
        for p = 0 to k - 1 do
          add_cprod r us.((i * k) + p) (at vh ~rows:vr ~cols:n t p j)
        done;
        res := !res +. Complex.norm2 (total r);
        norm := !norm +. Complex.norm2 aij
      done
    done;
    worse worst ~bound:((g *. sqrt !norm) +. h) ~residual:(sqrt !res)
      (fun () -> Printf.sprintf "matrix %d: a - u diag s vh" t);
    (* ‖xᴴx − I‖_F for the [size] columns of [len] elements [el p i]:
       Hermitian, its lower triangle's elements off the diagonal counted
       twice. *)
    let gram ~msg size len el =
      let o = ref 0. in
      for i = 0 to size - 1 do
        for j = 0 to i do
          let x = sum () in
          if i = j then add x.re (-1.);
          for p = 0 to len - 1 do
            add_cprod ~conj:true x (el p j) (el p i)
          done;
          let twice = if i = j then 1. else 2. in
          o := !o +. (twice *. Complex.norm2 (total x))
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

(* The values alone [s'] lie within 2 (svd_bound ‖a‖_F + svd_term) of the
   vectors' call's [s], for the [count] matrices of [m] × [n] of [a]. [a],
   [s] and [s'] are scaled by 2^e. *)
let check_values dt ~e ~count ~m ~n (a : Complex.t array) (s : Complex.t array)
    (s' : Complex.t array) =
  let k = min m n in
  let g = svd_bound dt ~m ~n and h = svd_term ~e dt ~m ~n in
  let worst = { ratio = 0.; where = "" } in
  for t = 0 to count - 1 do
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
      check_values c.dt ~e:d ~count:(count c) ~m ~n a s (up d (values y'.(0)))
  | _ -> ()

(* Spread magnitudes *)

(* A matrix of [rows] × [cols] whose columns, or rows by the seed, each
   lie at one of three scales: drawn in [-1, 1]; drawn, then times
   2^-(tiny ~bits dt), [bits] in [0, 8], whose parts round to subnormals,
   the least subnormal η and 0; or ±η throughout. By the seed too, the
   lines lie in that order, the largest first: the reduction then meets
   a row whose rest is subnormal beside its first element. Its parts can
   lie further apart than the largest finite number. *)
type spread = { sdt : D.any; rows : int; cols : int; key : int }

let pp_spread ppf s =
  Format.fprintf ppf "%a %d × %d seed %d" pp_dtype s.sdt s.rows s.cols s.key

let spread =
  Gen.with_pp pp_spread
    (let open Gen in
     let size =
       of_list ~pp:Format.pp_print_int [ 0; 1; 2; 3; 4; 5; 8; 17; 30 ]
     in
     let* sdt = of_list ~pp:pp_dtype dtypes in
     let* rows = size in
     let* cols = size in
     let+ key = int_range 0 1_000_000 in
     { sdt; rows; cols; key })

(* The dtype's largest finite number. *)
let largest (D.Any dt) =
  match dt with
  | D.Float32 | D.Complex64 -> Int32.float_of_bits 0x7f7fffffl
  | _ -> Float.max_float

(* The exponent that lifts a float64 matrix's subnormal parts above the
   least normal number while its parts, at most 1, square and sum below
   the largest: double-double residuals of η's would underflow. float32's
   are normal as floats. *)
let lift (D.Any dt) = match dt with D.Float32 | D.Complex64 -> 0 | _ -> 500

let spread_operand s =
  let r = Random.State.make [| s.key |] in
  let by_rows = Random.State.bool r and sorted = Random.State.bool r in
  let eta = Float.ldexp 1. (-tiny ~bits:0 s.sdt) in
  let at_scale = function
    | 0 -> Fun.id
    | 1 ->
        let e = -tiny ~bits:(Random.State.int r 9) s.sdt in
        fun x -> Float.ldexp x e
    | _ -> fun x -> Float.copy_sign eta x
  in
  let lines = if by_rows then s.rows else s.cols in
  let scales = Array.init lines (fun _ -> Random.State.int r 3) in
  if sorted then Array.sort compare scales;
  let scales = Array.map at_scale scales in
  let v =
    Array.init (s.rows * s.cols) (fun p ->
        let f = scales.(if by_rows then p / s.cols else p mod s.cols) in
        let z = draw r s.sdt in
        { Complex.re = f z.re; im = (if is_complex s.sdt then f z.im else 0.) })
  in
  let x = of_values s.sdt [| s.rows; s.cols |] v in
  let parts =
    Array.concat
      (List.map
         (fun (z : Complex.t) -> [| Float.abs z.re; Float.abs z.im |])
         (Array.to_list (values x)))
  in
  let most = Array.fold_left Float.max 0. parts in
  let least =
    Array.fold_left (fun l p -> if p > 0. then Float.min l p else l) infinity
      parts
  in
  cover "a part the least subnormal" (Array.exists (fun p -> p = eta) parts);
  cover "parts further apart than the largest finite number"
    (most /. least > largest s.sdt);
  x

(* svd's results meet check_svd's relations, and the values alone lie
   near the vectors' call's, on matrices whose parts lie at the least
   subnormal, below the least normal and near 1 at once: a reflector's
   first element can then exceed the rest's largest by more than the
   largest finite number. *)
let law_svd_spread (b : Support.backend) s =
  let m = s.rows and n = s.cols in
  let x = spread_operand s in
  cover "wide" (m < n);
  cover "square" (m = n);
  cover "tall" (m > n);
  let f = if s.key mod 2 = 0 then Nx_kernel.Reduced else Nx_kernel.Complete in
  match (run b (Svd (Some f)) [| x |], run b (Svd None) [| x |]) with
  | Some y, Some y' ->
      let e = lift s.sdt in
      let a = up e (values x) and sv = up e (values y.(0)) in
      let uc, vr = svd_columns f ~m ~n in
      check_svd s.sdt ~e ~count:1 ~m ~n ~uc ~vr a sv (values y.(1))
        (values y.(2));
      check_values s.sdt ~e ~count:1 ~m ~n a sv (up e (values y'.(0)))
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

(* check_svd's relations, for one matrix too large for its n³ steps, on
   [k] elements of each difference drawn by [r], each at most the bound
   on its difference's Frobenius norm; u and vh reduced. *)
let check_svd_sampled dt ~m ~n ~k r (a : Complex.t array) (s : Complex.t array)
    (u : Complex.t array) (vh : Complex.t array) =
  let p = min m n and g = svd_bound dt ~m ~n and h = svd_term ~e:0 dt ~m ~n in
  let norm = Array.fold_left (fun t x -> t +. Complex.norm2 x) 0. a in
  let worst = { ratio = 0.; where = "" } in
  for j = 0 to p - 1 do
    equal ~msg:"s is real" int64 0L (Int64.bits_of_float s.(j).im);
    if not (s.(j).re >= 0.) then failf "s %d is %h" j s.(j).re;
    if s.(j).re = 0. then
      equal ~msg:(Printf.sprintf "a zero s %d is +0" j) int64 0L
        (Int64.bits_of_float s.(j).re);
    if j > 0 && not (s.(j - 1).re >= s.(j).re) then
      failf "s not descending at %d" j
  done;
  (* Element (i, j) of xᴴx − I for the [p] columns of [len] elements
     [el q i]. *)
  let gram ~msg len el =
    let i = Random.State.int r p and j = Random.State.int r p in
    let o = sum () in
    if i = j then add o.re (-1.);
    for q = 0 to len - 1 do
      add_cprod ~conj:true o (el q i) (el q j)
    done;
    worse worst ~bound:g ~residual:(Complex.norm (total o)) (fun () ->
        Printf.sprintf "element (%d, %d) of %s" i j msg)
  in
  for _ = 1 to k do
    let i = Random.State.int r m and j = Random.State.int r n in
    let x = sum () in
    add x.re a.((i * n) + j).re;
    add x.im a.((i * n) + j).im;
    for q = 0 to p - 1 do
      add_cprod x
        (Complex.mul u.((i * p) + q) { Complex.re = -.s.(q).re; im = 0. })
        vh.((q * n) + j)
    done;
    worse worst ~bound:((g *. sqrt norm) +. h)
      ~residual:(Complex.norm (total x)) (fun () ->
        Printf.sprintf "element (%d, %d) of a - u diag s vh" i j);
    gram ~msg:"uᴴu - I" m (fun q i -> u.((q * p) + i));
    gram ~msg:"vh vhᴴ - I" n (fun q i -> vh.((i * n) + q))
  done;
  check_worst worst

(* a = H_m Σ H_n of [dt] and [m] × [n], H_l = I − 2 h hᴴ / hᴴh for h of
   [l] reflector_draws of [r] and Σ with [sigma] on its diagonal: its
   element (i, j) is Σ_ij − 2 x_i conj x_j σ_j − 2 σ_i y_i conj y_j +
   4 x_i c conj y_j, c = Σ_q conj x_q σ_q y_q, x and y the h's of unit
   norm. *)
let singular dt ~m ~n r sigma =
  let k = min m n in
  let unit l =
    let h = reflector_draws r dt l in
    let norm = sqrt (Array.fold_left (fun s x -> s +. Complex.norm2 x) 0. h) in
    Array.map (fun x -> Complex.div x { Complex.re = norm; im = 0. }) h
  in
  let x = unit m and y = unit n in
  let real f = { Complex.re = f; im = 0. } and ( *: ) = Complex.mul in
  let zero = Complex.zero in
  let sg i = real (if i < k then sigma.(i) else 0.) in
  let c = ref Complex.zero in
  for q = 0 to k - 1 do
    c := Complex.add !c (Complex.conj x.(q) *: sg q *: y.(q))
  done;
  let a =
    Array.init (m * n) (fun p ->
        let i = p / n and j = p mod n in
        let d = if i = j then sg i else Complex.zero in
        let xx = if j < k then x.(i) *: Complex.conj x.(j) *: sg j else zero
        and yy = if i < k then sg i *: y.(i) *: Complex.conj y.(j) else zero
        and xy = x.(i) *: !c *: Complex.conj y.(j) in
        Complex.add
          (Complex.sub d (real 2. *: Complex.add xx yy))
          (real 4. *: xy))
  in
  of_values dt [| m; n |] a

(* Of singular's a, the singular values are |σ|'s within svd_bound
   max |σ|, descending, alone and, unless [Unchecked], with the reduced
   vectors, which meet their relations. *)
let check_singular (b : Support.backend) ?(vectors = Checked) dt ~m ~n xa
    sigma =
  let k = min m n and a = values xa in
  let exact = Array.map Float.abs sigma in
  Array.sort (fun x y -> Float.compare y x) exact;
  let g = svd_bound dt ~m ~n *. Array.fold_left Float.max 0. exact in
  let spectrum ~msg s =
    let worst = { ratio = 0.; where = "" } in
    Array.iteri
      (fun j e ->
        worse worst ~bound:g ~residual:(Float.abs (s.(j).Complex.re -. e))
          (fun () -> Printf.sprintf "%s: value %d: %h, not %h" msg j s.(j).re e))
      exact;
    check_worst worst
  in
  Option.iter (fun y -> spectrum ~msg:"alone" (values y.(0)))
    (run b (Svd None) [| xa |]);
  if vectors <> Unchecked then
    Option.iter
      (fun y ->
        let s = values y.(0) and u = values y.(1) and vh = values y.(2) in
        spectrum ~msg:"with vectors" s;
        match vectors with
        | Unchecked -> ()
        | Checked -> check_svd dt ~e:0 ~count:1 ~m ~n ~uc:k ~vr:k a s u vh
        | Sampled q ->
            check_svd_sampled dt ~m ~n ~k:q (Random.State.make [| m; n |]) a
              s u vh)
      (run b (Svd (Some Reduced)) [| xa |])

(* One shape past each crossover to nx.cpu's two-stage reduction, by the
   lesser extent, where the bounds hold as below it: on x86-64 from 128
   for values alone of either kind and 256 for real vectors; on arm64 from
   128 for complex values alone, 192 for real ones, 512 for complex
   vectors and 1024 for real ones, which x86-64 is past at 257, so they
   run on arm64 alone. The relations are checked whole at 129 and on
   sampled elements past it, and the bits agree whatever the threads and
   layout up to 257. *)
let test_svd_large (b : Support.backend) () =
  let r = Random.State.make [| 11 |] in
  let shapes =
    D.
      [
        (129, 140, Any Complex128, graded r, Checked, true);
        (257, 260, Any Float64, clustered r, Sampled 256, true);
      ]
  and arm64 =
    D.
      [
        (513, 520, Any Complex128, repeating r, Sampled 256, false);
        (1025, 1030, Any Float64, graded r, Sampled 256, false);
      ]
  in
  List.iter
    (fun (m, n, dt, sigma, vectors, threads) ->
      let sigma = Array.init (min m n) sigma in
      let x = singular dt ~m ~n r sigma in
      check_singular b ~vectors dt ~m ~n x sigma;
      if threads then agree_bits b (Svd (Some Reduced)) x)
    (if Support.x86_64 then shapes else shapes @ arm64)

(* eigh, qr and svd of float64 matrices scaled as a whole by 1e-310, below
   the least normal number, and by 1e300, near the largest: every result a
   number, meeting its relations, and the spectra the known ones scaled
   within the bounds, whose ε terms cover the elements' rounding below the
   least normal number. Residuals are computed on the values scaled back
   by 2^e. *)
let test_scaled (b : Support.backend) () =
  let dt = D.Any D.Float64 and n = 30 in
  let r = Random.State.make [| 3 |] in
  let l = Array.init n (repeating r) and u = reflector_draws r dt n in
  let sigma = Array.init n (repeating r) in
  let plain = lu_operand { dt; batch = [||]; n; r = 0; triangle = Lower;
                           adjoint = false; unit_diagonal = false; seed = 3;
                           how = Contiguous; scale = Plain } in
  let times s x =
    of_values dt [| n; n |]
      (Array.map (fun (z : Complex.t) -> { z with re = z.re *. s }) (values x))
  in
  (* Each value of [w] within [bound] of one of [exact]. *)
  let near ~msg ~bound exact (w : Complex.t array) =
    let worst = { ratio = 0.; where = "" } in
    Array.iteri
      (fun j (z : Complex.t) ->
        let d =
          Array.fold_left
            (fun m y -> Float.min m (Float.abs (z.re -. y)))
            infinity exact
        in
        worse worst ~bound ~residual:d (fun () ->
            Printf.sprintf "%s: value %d, %h" msg j z.re))
      w;
    check_worst worst
  in
  let numbers_in ~msg y =
    Array.iteri
      (fun k y ->
        let v = values y in
        equal ~msg:(Printf.sprintf "%s: numbers in result %d" msg k) int
          (Array.length v) (numbers dt v))
      y
  in
  List.iter
    (fun (scale, e) ->
      let msg = Printf.sprintf "scaled by %g" scale in
      let x = times scale (spectral dt ~k:(n / 3) ~u l) in
      (match run b (Eigh true) [| x |] with
      | None -> ()
      | Some y ->
          numbers_in ~msg:(msg ^ ", eigh") y;
          let a = up e (values x) and w = up e (values y.(0)) in
          check_eigh dt ~e ~count:1 ~n a w (values y.(1));
          let big = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. l in
          let bound =
            (gamma dt (16 * n) *. Float.ldexp (big *. scale) e)
            +. epsilon ~e dt (16 * n)
          in
          near ~msg:(msg ^ ", eigh") ~bound
            (Array.map (fun v -> Float.ldexp (v *. scale) e) l)
            w);
      let x = times scale (snd plain) in
      (match run b (Qr Reduced) [| x |] with
      | None -> ()
      | Some y ->
          numbers_in ~msg:(msg ^ ", qr") y;
          check_qr dt ~e ~count:1 ~m:n ~n ~qc:n (up e (values x))
            (values y.(0))
            (up e (values y.(1))));
      let x = times scale (singular dt ~m:n ~n r sigma) in
      match (run b (Svd (Some Reduced)) [| x |], run b (Svd None) [| x |]) with
      | Some y, Some y' ->
          numbers_in ~msg:(msg ^ ", svd") y;
          let a = up e (values x) and s = up e (values y.(0)) in
          check_svd dt ~e ~count:1 ~m:n ~n ~uc:n ~vr:n a s (values y.(1))
            (values y.(2));
          let big = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. sigma in
          let bound =
            (svd_bound dt ~m:n ~n *. Float.ldexp (big *. scale) e)
            +. svd_term ~e dt ~m:n ~n
          in
          let exact = Array.map (fun v -> Float.ldexp (Float.abs v *. scale) e) sigma in
          near ~msg:(msg ^ ", svd") ~bound exact s;
          near ~msg:(msg ^ ", svd's values alone") ~bound exact (up e (values y'.(0)))
      | _ -> ())
    [ (1e-310, 1030); (1e300, -997) ]

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
  let real x = { Complex.re = x; im = 0. } in
  let cu = Array.map (fun x -> real (-2. *. x /. uu)) u in
  (* H f, f of [n] × [n] elements f i j: f + (−2u / uᵀu)(uᵀf), each sum
     in double-double; f H is (H fᵀ)ᵀ. *)
  let reflect f =
    let uf =
      Array.init n (fun j ->
          let s = sum () in
          for k = 0 to n - 1 do
            add_cprod s (real u.(k)) (f k j)
          done;
          total s)
    in
    Array.init (n * n) (fun p ->
        let i = p / n and j = p mod n and s = sum () in
        add s.re (f i j).re;
        add s.im (f i j).im;
        add_cprod s cu.(i) uf.(j);
        total s)
  in
  let hb = reflect (fun i j -> bm.((i * n) + j)) in
  let hbt = reflect (fun i j -> hb.((j * n) + i)) in
  let a = Array.init (n * n) (fun p -> hbt.((p mod n * n) + (p / n))) in
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
    let a = Array.map (fun re -> { Complex.re; im = 0. }) a in
    let x = of_values dt [| 2; 2 |] a in
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
                  let e = { Complex.re = e; im = 0. } in
                  Float.min m (Complex.norm (Complex.sub z e)))
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

(* A real matrix of subnormals whose complex pair's imaginary part lies
   below half the least subnormal η: a = η [−1 3 1; 3 1 1; 1 −1 2],
   which balancing leaves, of eigenvalues −3.22 η and (2.61 ± 0.0974 i) η.
   eig's results meet check_eig's relations and check_real's, with the
   vectors and without. *)
let test_eig_subnormal_pair (b : Support.backend) () =
  let n = 3 and m = [| -1.; 3.; 1.; 3.; 1.; 1.; 1.; -1.; 2. |] in
  List.iter
    (fun dt ->
      let e = tiny ~bits:0 dt in
      let a =
        Array.map (fun re -> { Complex.re = Float.ldexp re (-e); im = 0. }) m
      in
      let x = of_values dt [| n; n |] a in
      match (run b (Eig true) [| x |], run b (Eig false) [| x |]) with
      | Some y, Some y' ->
          let w = values y.(0) and v = values y.(1) and w' = values y'.(0) in
          let msg = Format.asprintf "%a" pp_dtype dt in
          check_eig dt ~e ~count:1 ~n (up e a) (up e w) v;
          check_real ~msg:(msg ^ " with vectors") ~count:1 ~n w v;
          check_real ~msg:(msg ^ " alone") ~count:1 ~n w' [||]
      | _ -> ())
    [ D.Any D.Float32; D.Any D.Float64 ]

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
          equal ~msg:(msg ^ ": numbers in w") int (2 * n)
            (numbers (D.Any D.Complex128) w);
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

(* Wilkinson's W⁺_n, tridiagonal with m, m − 1, …, 1, 0, 1, …, m on its
   diagonal, m = (n − 1) / 2, and ones beside it: its largest eigenvalues
   come in pairs that agree to many digits. It is symmetric, so balancing
   leaves it. eig's results meet check_eig's relations, and on reals
   check_real's, with the vectors and without, and each eigenvalue is
   within eig's bound plus eigh's of one of eigh's: a normal matrix's
   eigenvalues move by at most ‖E‖_F under a perturbation E, and ‖a‖₂ is at
   most ‖a‖_F. *)
let test_wilkinson (b : Support.backend) () =
  List.iter
    (fun (n, dt) ->
      let m = (n - 1) / 2 in
      let a =
        Array.init (n * n) (fun p ->
            let i = p / n and j = p mod n in
            let v =
              if i = j then float_of_int (abs (i - m))
              else if abs (i - j) = 1 then 1.
              else 0.
            in
            { Complex.re = v; im = 0. })
      in
      let x = of_values dt [| n; n |] a in
      let msg = Format.asprintf "%a, n %d" pp_dtype dt n in
      match (run b (Eig true) [| x |], run b (Eig false) [| x |]) with
      | Some y, Some y' ->
          let w = values y.(0) and v = values y.(1) and w' = values y'.(0) in
          check_eig dt ~e:0 ~count:1 ~n a w v;
          if not (is_complex dt) then begin
            check_real ~msg:(msg ^ " with vectors") ~count:1 ~n w v;
            check_real ~msg:(msg ^ " alone") ~count:1 ~n w' [||]
          end;
          Option.iter
            (fun z ->
              let norm = sqrt (Array.fold_left (fun s z -> s +. Complex.norm2 z) 0. a) in
              let bound = 2. *. ((eig_bound dt n *. norm) +. eig_term ~e:0 dt n) in
              let eigh = values z.(0) in
              List.iter
                (fun (how, w) ->
                  let worst = { ratio = 0.; where = "" } in
                  Array.iteri
                    (fun j (z : Complex.t) ->
                      let d =
                        Array.fold_left
                          (fun d (y : Complex.t) ->
                            Float.min d (Complex.norm (Complex.sub z y)))
                          infinity eigh
                      in
                      worse worst ~bound ~residual:d (fun () ->
                          Printf.sprintf "%s, %s: eigenvalue %d, %h%+hi, beside eigh's" msg
                            how j z.re z.im))
                    w;
                  check_worst worst)
                [ ("with vectors", w); ("alone", w') ])
            (run b (Eigh false) [| x |])
      | _ -> ())
    D.[ (21, Any Float64); (41, Any Float64); (81, Any Float64);
        (21, Any Complex128); (41, Any Complex128); (81, Any Complex128) ]

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

(* Matrices whose first row past its diagonal is tiny beside the rest,
   [d r r r] over [0 M], M = s [[1 2 3] [2 -1 1] [3 1 -2]], whose
   reduction's first right reflector takes the row's rest r. In float64
   (r = 1e-300, s = 1e-138) r is below the least normal number over u,
   so the reflector scales it up before it normalises it; in float32
   (r = 1e-30, s = 1e-14, d = 1e-12) r is near the least normal number,
   and M times r is subnormal. Neither matrix is scaled as a whole: their
   largest magnitudes are above svd's threshold. svd's results meet their
   relations. *)
let test_svd_tiny_rows (b : Support.backend) () =
  List.iter
    (fun (dt, d, r, s) ->
      let m = [| 1.; 2.; 3.; 2.; -1.; 1.; 3.; 1.; -2. |] in
      let a =
        Array.init 16 (fun p ->
            let i = p / 4 and j = p mod 4 in
            let v =
              if i = 0 then if j = 0 then d else r
              else if j = 0 then 0.
              else s *. m.(((i - 1) * 3) + j - 1)
            in
            { Complex.re = v; im = 0. })
      in
      let x = of_values dt [| 4; 4 |] a in
      match run b (Svd (Some Reduced)) [| x |] with
      | Some y ->
          check_svd dt ~e:0 ~count:1 ~m:4 ~n:4 ~uc:4 ~vr:4 (values x)
            (values y.(0)) (values y.(1)) (values y.(2))
      | None -> ())
    [
      (D.Any D.Float64, 1e-138, 1e-300, 1e-138);
      (D.Any D.Float32, 1e-12, 1e-30, 1e-14);
    ]

(* The rank-one [1 (1 + i) η] over two rows of zeros, η the least
   subnormal: the first row's rest [η] lies beside 1 − i, the reflector's
   divisor 1 + √2 − i more than the largest finite number times η. Its
   singular values are √3, 0 and 0, √2, 0 and 0 on reals, which keep
   [1 1 η]. svd's results meet their relations, its values within
   svd_bound ‖a‖₂ + svd_term of those, on every dtype. *)
let test_svd_spread_row (b : Support.backend) () =
  List.iter
    (fun dt ->
      let eta = Float.ldexp 1. (-tiny ~bits:0 dt) in
      let z re im = { Complex.re; im } in
      let a = Array.make 9 Complex.zero in
      a.(0) <- z 1. 0.;
      a.(1) <- z 1. 1.;
      a.(2) <- z eta 0.;
      let x = of_values dt [| 3; 3 |] a in
      match (run b (Svd (Some Reduced)) [| x |], run b (Svd None) [| x |]) with
      | Some y, Some y' ->
          check_svd dt ~e:0 ~count:1 ~m:3 ~n:3 ~uc:3 ~vr:3 (values x)
            (values y.(0)) (values y.(1)) (values y.(2));
          let g = svd_bound dt ~m:3 ~n:3 and h = svd_term ~e:0 dt ~m:3 ~n:3 in
          let norm = sqrt (if is_complex dt then 3. else 2.) in
          let exact = [| norm; 0.; 0. |] in
          let worst = { ratio = 0.; where = "" } in
          List.iter
            (fun (msg, s) ->
              Array.iteri
                (fun j (v : Complex.t) ->
                  worse worst
                    ~bound:((g *. norm) +. h)
                    ~residual:(Float.abs (v.re -. exact.(j)))
                    (fun () ->
                      Format.asprintf "%a: %s %d is %h" pp_dtype dt msg j v.re))
                s)
            [ ("value", values y.(0)); ("value alone", values y'.(0)) ];
          check_worst worst
      | _ -> ())
    dtypes

(* Reference-LAPACK issue 255's upper bidiagonal of order 1000, its
   diagonal then its superdiagonal in golden/linalg, on which a deflation
   of the merges at 8 u lost u's and vh's orthogonality in pairs of
   vectors. svd's results meet its bound element by element: |(uᵀu − I)_ij|,
   |(vh vhᵀ − I)_ij| <= g and |(a − u diag s vh)_ij| <= g ‖a‖_F. The
   products are the kernels' own float64 contractions, which test_contract
   holds within their bound: 1000 terms rounded fewer than 100 times, far
   below g. *)
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
      let module K = (val b.kernels) in
      let f64 y = A.expect D.Float64 y in
      let s = f64 y.(0) and u = f64 y.(1) and vh = f64 y.(2) in
      (* The n × n product of [p] and [q] contracted along [contracting]. *)
      let product ~contracting p q =
        let dst = A.create Rig.host D.Float64 [| n; n |] in
        let op x : Nx_kernel.operand = { array = A.Any x; scale = None } in
        match
          K.contract ~batch:[||] ~contracting ~segments:None
            ~acc:(D.Any D.Float64) ~dst ~init:None (op p) (op q)
        with
        | A.Done -> A.to_array dst
        | r -> failf "a contraction answered %a" pp_answer r
      in
      let sv = A.to_array s in
      let us =
        A.of_array D.Float64 [| n; n |]
          (Array.mapi (fun p x -> x *. sv.(p mod n)) (A.to_array u))
      in
      let uu = product ~contracting:[| (0, 0) |] u u in
      let vv = product ~contracting:[| (1, 1) |] vh vh in
      let usv = product ~contracting:[| (1, 0) |] us vh in
      let g = svd_bound (D.Any D.Float64) ~m:n ~n in
      let norm = sqrt (Array.fold_left (fun t x -> t +. (x *. x)) 0. a) in
      let w = { ratio = 0.; where = "" } in
      let gram msg x =
        Array.iteri
          (fun p t ->
            let i = p / n and j = p mod n in
            let t = t -. if i = j then 1. else 0. in
            worse w ~bound:g ~residual:(Float.abs t) (fun () ->
                Printf.sprintf "%s, element (%d, %d)" msg i j))
          x
      in
      gram "uᵀu - I" uu;
      gram "vh vhᵀ - I" vv;
      Array.iteri
        (fun p t ->
          worse w ~bound:(g *. norm) ~residual:(Float.abs (a.(p) -. t))
            (fun () ->
              Printf.sprintf "a - u diag s vh, element (%d, %d)" (p / n)
                (p mod n)))
        usv;
      check_worst w

(* A pivot whose reciprocal is subnormal, past 2^1022 in float64 and
   2^126 in float32: the 2 × 1 matrix [p; x]'s L meets γ(2). *)
let test_huge_pivot (b : Support.backend) () =
  let lu dt p x =
    let a = [| { Complex.re = p; im = 0. }; { Complex.re = x; im = 0. } |] in
    match run b Lu [| of_values dt [| 2; 1 |] a |] with
    | None -> ()
    | Some y ->
        let v = values y.(0) in
        let u = v.(0).re and l = v.(1).re in
        at_most ~msg:"|l u - x|" float_exact
          ~than:(gamma dt 2 *. Float.abs (l *. u))
          (Float.abs (Float.fma l u (-.x)))
  in
  lu D.(Any Float64) 0x1.8p1023 (-0x1.2000ce6d1d565p+1022);
  lu D.(Any Float32) 0x1.8p127 (-0x1.2000cep+126)

(* Failures and reads *)

(* A solve of as many right-hand sides as rows, into a destination
   identical to its triangle or to its right-hand sides, gives the bits a
   fresh destination receives. *)
let law_solve_aliased (b : Support.backend) c =
  let module K = (val b.kernels) in
  let c = { c with r = c.n } in
  let ops = solve_operands c in
  let into k =
    let (A.Any a) = ops.(0) in
    let dt = A.dtype a in
    let a = A.copy a and rhs = A.copy (A.expect dt ops.(1)) in
    let dst = if k = 0 then a else rhs in
    match K.solve_triangular c.triangle ~adjoint:c.adjoint
            ~unit_diagonal:c.unit_diagonal ~dst a rhs with
    | A.Done -> Some (bits (A.Any dst))
    | A.Declined -> None
    | e -> failf "the kernels answered %a" pp_answer e
  in
  match run b (solve c) ops with
  | Some y ->
      List.iter
        (fun (msg, k) ->
          Option.iter (equal ~msg same_bits (bits y.(0))) (into k))
        [ ("into the triangle", 0); ("into the right-hand sides", 1) ]
  | None -> ()

(* Layouts: a matrix through another layout, or one matrix broadcast across
   the batch, gives the bits its contiguous copy gives. *)
let law_layouts (b : Support.backend) c =
  cover "rows further apart than their elements" (c.how = Padded && c.n > 1);
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
  (match
     (run b (Eig true) [| x |], run b (Eig true) [| laid_out c.how x |])
   with
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
  let ops = solve_operands c in
  let laid = [| laid_out c.how ops.(0); ops.(1) |] in
  (match (run b (solve c) ops, run b (solve c) laid) with
  | Some y, Some z -> equal ~msg:"solve" same_bits (bits y.(0)) (bits z.(0))
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

(* The bits of a result, a position's as its value's. *)
let result_bits (A.Any x as r) =
  match A.dtype x with
  | D.Int64 -> Array.map float_of_int (int64s r)
  | _ -> bits r

(* [o] repeated along [c]'s batch. *)
let repeated c (A.Any x) =
  let s = Iarray.append (Iarray.of_array c.batch) (L.shape (A.layout x)) in
  A.Any (A.copy (Option.get (A.move (M.Broadcast s) x)))

(* [c] with a batch of at least as many matrices as the host has cores,
   along one axis or two by its seed: nx.cpu then computes each matrix
   on one thread. *)
let wide c =
  let t = Domain.recommended_domain_count () in
  let batch = if c.seed mod 2 = 0 then [| t |] else [| 2; (t + 1) / 2 |] in
  { c with batch }

(* The routines with work of their own on a matrix's threads, besides
   their products: solves' units, LU's steps, eigh's and svd's
   matrix-vector products and their merges' roots. *)
let with_units =
  [ "cholesky"; "solve"; "lu"; "eigh"; "svd"; "svd's values" ]

(* nx.cpu computes [routines], by name, of [c]'s matrix alone, on the
   call's threads, and of copies of it in a wide batch, one a thread:
   every result of every matrix of the batch has the bits of the
   matrix's alone. *)
let threads_agree (b : Support.backend) routines c =
  let c = wide c in
  let k = count c in
  let one = { c with batch = [||] } in
  let x = cholesky_operand one and ops = solve_operands one in
  let _, lu = lu_operand one in
  let _, _, s = svd_operand one in
  let agree msg y z =
    Array.iteri
      (fun j z ->
        let v = result_bits y.(j) and want = result_bits z in
        let n = Array.length want in
        for i = 0 to k - 1 do
          let msg = Printf.sprintf "%s's result %d, matrix %d" msg j i in
          equal ~msg same_bits want (Array.sub v (i * n) n)
        done)
      z
  in
  List.iter
    (fun (msg, r, ops) ->
      if List.mem msg routines then
        match (run b r (Array.map (repeated c) ops), run b r ops) with
        | Some y, Some z -> agree msg y z
        | _ -> failf "nx.cpu declined %s at %a" msg pp_dtype c.dt)
    [
      ("cholesky", Cholesky Lower, [| x |]);
      ("solve", solve one, ops);
      ("lu", Lu, [| lu |]);
      ("qr", Qr (factors one), [| lu |]);
      ("eigh", Eigh true, [| eigh_operand one |]);
      ("svd", Svd (Some (factors one)), [| s |]);
      ("svd's values", Svd None, [| s |]);
      ("eig", Eig true, [| eig_operand one |]);
    ]

(* A plain contiguous case of [dt], of order [n] and [r] right-hand sides. *)
let fixed dt n r =
  {
    dt;
    batch = [||];
    n;
    r;
    triangle = Lower;
    adjoint = false;
    unit_diagonal = false;
    seed = 7;
    how = Contiguous;
    scale = Plain;
  }

(* nx.cpu computes every routine at [dt], and a matrix of order 65, past a
   block of 64, alone gives the bits it has in a batch. *)
let test_cpu_threads b dt =
  threads_agree b ([ "qr"; "eig" ] @ with_units) (fixed dt 65 3)

(* Matrices of order 256 and 160 right-hand sides, of the routines whose
   work on a matrix's threads goes beyond their products, which give the
   bits of one thread. On one core nothing runs on several threads. *)
let test_cpu_units b dt =
  if Domain.recommended_domain_count () = 1 then skip ~reason:"one core" ();
  threads_agree b with_units (fixed dt 256 160)

(* One matrix of order 272, past four blocks of 64 and a partial one. *)
let test_cpu_large b dt = law_cholesky b (fixed dt 272 0)

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
  | Eigh true | Eig true -> Result.map (fun (w, x) -> [| w; x |]) (S.eigh c.a)
  | Eigh false | Eig false -> Result.map (fun (w, _) -> [| w |]) (S.eigh c.a)
  | Svd None ->
      Result.map (fun (s, _, _) -> [| s |]) (S.svd Reduced c.a)
  | Svd (Some f) -> Result.map (fun (s, u, vh) -> [| s; u; vh |]) (S.svd f c.a)
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
  let zeros s =
    A.of_array D.Float32 (Iarray.of_array s)
      (Array.make (Elements.total s) 0.)
  in
  let dst i = A.create Rig.host D.Float32 (Iarray.of_array c.dsts.(i)) in
  let ints i = A.create Rig.host D.Int64 (Iarray.of_array c.dsts.(i)) in
  let cdst i = A.create Rig.host D.Complex64 (Iarray.of_array c.dsts.(i)) in
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
  let sentinel dt v s =
    A.of_array dt s (Array.make (Iarray.fold_left ( * ) 1 s) v)
  in
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
  (* Values of another precision than [a]'s, and of [a]'s own complex
     dtype's. *)
  let s = sentinel D.Float64 7. [| 2 |] in
  refuses ~msg:"float64 singular values of float32" float_exact
    A.Wrong_dtype (K.svd ~s f32) s 7.;
  let c128 = A.of_array D.Complex128 [| 2; 2 |] (Array.make 4 Complex.one) in
  (* The vectors too are left as they were. *)
  let ones () = sentinel D.Complex128 Complex.one [| 2; 2 |] in
  let untouched ~msg x =
    equal ~msg
      (array
         (Testable.make
            ~pp:(fun ppf (z : Complex.t) -> Format.fprintf ppf "%h%+hi" z.re z.im)
            ~equal:( = )))
      (Array.make 4 Complex.one) (A.to_array x)
  in
  let s = sentinel D.Float32 7. [| 2 |] and u = ones () and vh = ones () in
  refuses ~msg:"float32 singular values of complex128" float_exact
    A.Wrong_dtype
    (K.svd ~s ~vectors:(~u, ~vh) c128)
    s 7.;
  untouched ~msg:"svd's refused u" u;
  untouched ~msg:"svd's refused vh" vh;
  let w = sentinel D.Float16 7. [| 2 |] in
  refuses ~msg:"float16 eigenvalues of float32" float_exact A.Wrong_dtype
    (K.eigh ~w f32) w 7.;
  let w = sentinel D.Float32 7. [| 2 |] and v = ones () in
  refuses ~msg:"float32 eigenvalues of complex128" float_exact A.Wrong_dtype
    (K.eigh ~w ~v c128) w 7.;
  untouched ~msg:"eigh's refused v" v

(* Guard pages *)

(* Cases of orders about a vector's tail and a block's, a few matrices at
   most, their operands through any layout. *)
let guard_cases =
  Gen.pair
    (Gen.with_pp pp_case
       (let open Gen in
        let* c = case in
        let* n =
          of_list ~pp:Format.pp_print_int
            [ 0; 1; 2; 3; 5; 8; 9; 17; 26; 33; 63; 64; 65 ]
        in
        let* batch = array ~size:(int_range 0 1) (int_range 1 2) in
        let+ how =
          of_list ~pp:pp_how
            [ Contiguous; Transposed; Stepped; Padded; Reversed ]
        in
        { c with n; batch; how }))
    Elements.sides

(* Each routine gives the same bits with its operands, through the case's
   layout, and its results against an inaccessible page. *)
let law_guards (b : Support.backend) (c, side) =
  cover "a column-reversed operand at the page after"
    (c.how = Reversed && side = Support.After);
  cover "a stepped operand at the page after"
    (c.how = Stepped && side = Support.After);
  cover "a transposed operand at the page before"
    (c.how = Transposed && side = Support.Before);
  cover "a block's order or past" (c.n >= 64);
  let lay = laid_out c.how in
  let _, lu_x = lu_operand c in
  let _, _, svd_x = svd_operand c in
  let routines =
    [
      (Cholesky c.triangle, [| cholesky_operand c |]);
      (Lu, [| lu_x |]);
      (Qr (factors c), [| lu_x |]);
      (Eigh (c.seed mod 2 = 0), [| eigh_operand c |]);
      (Svd (if c.seed mod 3 = 0 then None else Some (factors c)), [| svd_x |]);
      (Eig (c.seed mod 2 = 1), [| eig_operand c |]);
      (solve c, solve_operands c);
    ]
  in
  List.iter
    (fun (r, ops) ->
      let ops = Array.map lay ops in
      let want = Option.get (run b r ops) in
      Elements.with_guards side (fun guard place ->
          match run ~alloc:place b r (Array.map guard ops) with
          | None -> failf "%s declined on guarded memory" (routine_name r)
          | Some got ->
              Array.iteri
                (fun i w ->
                  equal ~msg:(routine_name r) same_bits (result_bits w)
                    (result_bits got.(i)))
                want))
    routines

(* The suite *)

let dtype_name = Format.asprintf "%a" pp_dtype

(* What nx_cpu.mli states beyond the contract: its bounds, the Upper factor's
   bits, a solve into its operand, layout and thread invariance, and the
   matrices its algorithms find hard. *)
let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  let test name f = test name (fun () -> b.around (f b)) in
  let cases name f =
    cases name ~id:dtype_name dtypes (fun dt -> b.around (fun () -> f b dt))
  in
  group b.name
    [
      group "each result meets its relation within its bound"
        [
          prop "Cholesky's factor within γ(n + 1)" case (run (law_cholesky b));
          cases "Cholesky's factor of order 272" test_cpu_large;
          prop "the Upper factor is the Lower one's conjugate transpose" case
            (run (law_upper b));
          prop "a triangular solve within γ(2n)" case (run (law_solve b));
          prop "LU's factors within γ(2k)" case (run (law_lu b));
          test "LU of a pivot whose reciprocal is subnormal" test_huge_pivot;
          prop "QR's factors within γ(k(2m + 10))" case (run (law_qr b));
          prop "eigh's results within γ(16n)" case (run (law_eigh b));
          prop "svd's results within γ(16 max (m, n) + 100)" case
            (run (law_svd b));
          prop
            "svd's results on parts at the least subnormal, below the least \
             normal and near 1"
            spread (run (law_svd_spread b));
          test
            "svd's results within their bound on a row whose rest is below \
             the least normal number over u"
            test_svd_tiny_rows;
          test "svd's results on a row whose rest is the least subnormal"
            test_svd_spread_row;
          test "svd's vectors on a bidiagonal whose merges deflate"
            test_svd_deflation;
          prop "eig's results within γ(16n)" case (run (law_eig b));
          test "eig's results on a graded cycle" test_eig_graded_cycle;
          test "eig's results on a subnormal matrix with a complex pair"
            test_eig_subnormal_pair;
          test "eigh's, qr's and svd's results on matrices scaled below the \
                least normal number and near the largest"
            test_scaled;
        ];
      group "each spectrum is within its bound of the exact one"
        [
          prop "eigh's eigenvalues within γ(16n) ‖a‖₂" case
            (run (law_eigh_spectrum b));
          test "eigh's eigenvalues at large orders" test_eigh_large;
          test "eigh's eigenvalues of a secular equation with a root beside \
                its pole"
            test_eigh_secular;
          prop "svd's values within γ(16 max (m, n) + 100) ‖a‖₂" case
            (run (law_svd_spectrum b));
          test "svd's values at large orders" test_svd_large;
          prop "eig's eigenvalues of a normal matrix within γ(16n) ‖a‖_F" case
            (run (law_eig_spectrum b));
          test "eig's eigenvalues of a graded matrix, balanced"
            test_eig_balanced;
          test "eig's eigenvalues of a matrix spanning the exponent range, \
                balanced before it is scaled"
            test_eig_extremes;
          test "eig's eigenvalues of Wilkinson's matrices are eigh's"
            test_wilkinson;
        ];
      group "bits are a function of the values alone"
        [
          prop "whatever the layout" case (run (law_layouts b));
          cases "whatever the threads, every routine at every dtype"
            test_cpu_threads;
          cases "whatever the threads a matrix's units run on" test_cpu_units;
          prop "a solve into its operand gives a fresh destination's bits" case
            (run (law_solve_aliased b));
        ];
      group "refusals"
        [
          prop "refuses what Nx_kernel.Shape refuses and other shapes" misfits
            (run (law_cpu_refuses b));
          test "refuses dtypes and shapes before any write" test_cpu_refusals;
        ];
      prop ~count:100
        "reads and writes no byte past its operands and results, whatever \
         pages lie beside them"
        guard_cases (run (law_guards b));
    ]

let () = exit (Windtrap.run "nx_cpu.linalg" (List.map cpu Support.cpus))
