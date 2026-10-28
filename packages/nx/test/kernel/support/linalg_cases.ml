(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What the suites of Nx_kernel.S's factorisations and solves and of
   nx.cpu's share: the cases, their values, views and draws, the operands
   each routine takes, and a routine's call. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move
module Support = Nx_kernels_support

let pp_dtype ppf (D.Any dt) = D.pp ppf dt

let dtypes = D.[ Any Float32; Any Float64; Any Complex64; Any Complex128 ]

let is_complex (D.Any dt) = D.is D.Complex dt

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
  let shape = Iarray.of_array shape in
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

(* The values' parts, a complex number's real then imaginary one, as
   floats: same_bits compares their bits, NaN payloads and signs of zero
   included. *)
let bits (A.Any x) : float array =
  let parts (v : Complex.t array) =
    Array.init (2 * Array.length v) (fun k ->
        if k land 1 = 0 then v.(k / 2).re else v.(k / 2).im)
  in
  match A.dtype x with
  | D.Float32 -> A.to_array x
  | D.Float64 -> A.to_array x
  | D.Complex64 -> parts (A.to_array x)
  | D.Complex128 -> parts (A.to_array x)
  | _ -> invalid_arg "bits"

let same_bits =
  let bits = Int64.bits_of_float in
  array
    (Testable.make
       ~pp:(fun ppf x -> Format.fprintf ppf "%h (%016Lx)" x (bits x))
       ~equal:(fun a b -> Int64.equal (bits a) (bits b)))

type how = Contiguous | Transposed | Stepped | Padded | Reversed

let pp_how ppf h =
  Format.pp_print_string ppf
    (match h with
    | Contiguous -> "contiguous"
    | Transposed -> "transposed"
    | Stepped -> "stepped"
    | Padded -> "padded"
    | Reversed -> "reversed")

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
     let* how =
       of_list ~pp:pp_how [ Contiguous; Transposed; Stepped; Padded ]
     in
     let+ scale = of_list ~pp:pp_scale [ Plain; Graded; Subnormal ] in
     { dt; batch; n; r; triangle; adjoint; unit_diagonal; seed; how; scale })

let count c = Array.fold_left ( * ) 1 c.batch

let square c = Array.append c.batch [| c.n; c.n |]

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
   eigenvalues at least δ, its upper triangle its lower one's conjugate.
   Its sums are in floats: they err by about nε, far below δ. *)
let definite r dt n =
  let m = Array.init (n * n) (fun _ -> draw r dt) in
  let delta = [| 1.; 0.1; 0.01 |].(Random.State.int r 3) in
  let a = Array.make (n * n) Complex.zero and f = float_of_int n in
  for i = 0 to n - 1 do
    for j = 0 to i do
      let re = ref 0. and im = ref 0. in
      for p = 0 to n - 1 do
        let x = m.((i * n) + p) and y = m.((j * n) + p) in
        re := !re +. (x.re *. y.re) +. (x.im *. y.im);
        im := !im +. (x.im *. y.re) -. (x.re *. y.im)
      done;
      let re = !re /. f and im = !im /. f in
      if i = j then a.((i * n) + i) <- { Complex.re = re +. delta; im = 0. }
      else begin
        a.((i * n) + j) <- { Complex.re = re; im };
        a.((j * n) + i) <- { Complex.re = re; im = -.im }
      end
    done
  done;
  a

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

let fresh (D.Any dt) s = A.Any (A.create Rig.host dt (Iarray.of_array s))

(* [b]'s results of [r] on [ops], or [None] if it declined: Lu's positions
   int64, eig's results complex of a's precision, svd's and eigh's values
   floats of a's precision, every other result of a's dtype, of the shapes
   Nx_kernel.Shape gives, made by [alloc] (defaults to fresh arrays on the
   host). *)
let run ?(alloc = fresh) (b : Support.backend) r ops =
  let module K = (val b.kernels) in
  let module Shape = Nx_kernel.Shape in
  let (A.Any a) = ops.(0) in
  let dt = A.dtype a in
  let shape (A.Any x) = Iarray.to_array (L.shape (A.layout x)) in
  let make s = A.expect dt (alloc (D.Any dt) s) in
  let index s = A.expect D.Int64 (alloc (D.Any D.Int64) s) in
  let real (type b) s (f : 'r. (float, 'r) A.t -> b) : b =
    match dt with
    | D.Float32 | D.Complex64 ->
        f (A.expect D.Float32 (alloc (D.Any D.Float32) s))
    | _ -> f (A.expect D.Float64 (alloc (D.Any D.Float64) s))
  in
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
        real sw (fun w ->
            if vectors then
              let v = make sv in
              ([| A.Any w; A.Any v |], K.eigh ~w ~v a)
            else ([| A.Any w |], K.eigh ~w a))
    | Svd None ->
        let ss, _, _ = ok (Shape.svd Reduced (shape ops.(0))) in
        real ss (fun s -> ([| A.Any s |], K.svd ~s a))
    | Svd (Some f) ->
        let ss, su, sv = ok (Shape.svd f (shape ops.(0))) in
        real ss (fun s ->
            let u = make su and vh = make sv in
            ([| A.Any s; A.Any u; A.Any vh |], K.svd ~s ~vectors:(~u, ~vh) a))
    | Eig vectors -> (
        let sw, sv = ok (Shape.eigh (shape ops.(0))) in
        let eig (type c) (cdt : (Complex.t, c) D.t) =
          let w = A.expect cdt (alloc (D.Any cdt) sw) in
          if vectors then
            let v = A.expect cdt (alloc (D.Any cdt) sv) in
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
          let phase = { Complex.re = p.re /. q; im = p.im /. q } in
          Complex.mul m.((j * n) + i) phase)

(* A lower triangular matrix of order [n], whose rows and columns
   balancing isolates one by one. *)
let lower r dt n =
  Array.init (n * n) (fun k ->
      if k / n >= k mod n then draw r dt else Complex.zero)

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

let factors c = if c.seed mod 2 = 0 then Nx_kernel.Reduced else Nx_kernel.Complete

(* Failures and reads *)

let one c = { c with batch = [||] }
