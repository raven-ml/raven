(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.metal's spectral routines, as nx_metal.mli states them, on the machine's
   Metal GPU 0, skipping without one: each result against its defining relation
   within nx.metal's bound, on the suite's cases; the bits of a matrix the same
   however its steps split into launches and whatever its batch; a matrix past
   an iteration's limit NaN alone; an operand's results those of a′, its
   subnormal elements zeros of their sign; calls in a loop holding one call's
   workspace; the calls the section does not state declined. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module Support = Nx_kernels_support
module Plan = Nx_metal.Plan
module K = Nx_metal.Kernels
open Linalg_cases
open Linalg_laws

(* nx_metal.mli's bounds: ν(k) = k 2^-126. *)
let bounds =
  {
    cholesky = (fun n -> n + 1);
    lu = (fun k -> k + 2);
    solve = (fun n -> n + 2);
    alone = (fun ~e dt k -> epsilon ~underflow:Support.Flush ~e dt k);
    positive = true;
  }

(* The machine's Metal GPU 0, opened at a test's first call; a test skips
   without it. *)
let metal () =
  match Support.metal () with
  | Some b -> b
  | None -> skip ~reason:"no Metal GPU nx.metal computes on" ()

let floats = D.[ Any Float32; Any Complex64 ]
let pp_dt = Format.asprintf "%a" pp_dtype

(* The paths nx_metal.mli states: svd by Jacobi up to 32 rows and columns, svd's
   values past them. *)
let svd_jacobi c =
  let m, n, _ = svd_operand c in
  m <= 32 && n <= 32

(* [f] on the cases [held] holds. *)
let on held f c =
  assume (held c);
  let b = metal () in
  b.around (fun () -> f b c)

let everything _ = true

(* Calls *)

(* [r]'s answer on [x], and its results on nx.metal's device, of the dtypes and
   shapes [run] gives them. *)
let call (b : Support.backend) r (A.Any x) =
  let module Kn = (val b.kernels) in
  let module Shape = Nx_kernel.Shape in
  let x = Support.on b x in
  let dt = A.dtype x in
  let shape = Iarray.to_array (L.shape (A.layout x)) in
  let make d s = A.create b.device d (Iarray.of_array s) in
  let ok = Result.get_ok in
  match r with
  | Eigh vectors ->
      let sw, sv = ok (Shape.eigh shape) in
      let w = make D.Float32 sw in
      if vectors then
        let v = make dt sv in
        (Kn.eigh ~w ~v x, [| A.Any w; A.Any v |])
      else (Kn.eigh ~w x, [| A.Any w |])
  | Svd None ->
      let ss, _, _ = ok (Shape.svd Reduced shape) in
      let s = make D.Float32 ss in
      (Kn.svd ~s x, [| A.Any s |])
  | Svd (Some f) ->
      let ss, su, sv = ok (Shape.svd f shape) in
      let s = make D.Float32 ss and u = make dt su and vh = make dt sv in
      (Kn.svd ~s ~vectors:(~u, ~vh) x, [| A.Any s; A.Any u; A.Any vh |])
  | Eig vectors ->
      let sw, sv = ok (Shape.eigh shape) in
      let w = make D.Complex64 sw in
      if vectors then
        let v = make D.Complex64 sv in
        (Kn.eig ~w ~v x, [| A.Any w; A.Any v |])
      else (Kn.eig ~w x, [| A.Any w |])
  | Cholesky _ | Lu | Qr _ | Solve _ -> invalid_arg "call: not spectral"

(* The results' bits of [r] on [x], which nx.metal computes. *)
let results b r x =
  match call b r x with
  | A.Done, ys -> Array.map bits ys
  | got, _ -> failf "the call answered %a" pp_answer got

let numbers a =
  Array.fold_left (fun k x -> if Float.is_nan x then k else k + 1) 0 a

(* NaN's bits are not stated: any NaN reads as one. *)
let one_nan a = Array.map (fun x -> if Float.is_nan x then Float.nan else x) a

let same_results ~msg want got =
  Array.iteri
    (fun k w ->
      equal
        ~msg:(Printf.sprintf "%s: result %d" msg k)
        same_bits (one_nan w)
        (one_nan got.(k)))
    want

(* [f ()] and the counts of each of [ks]' launches it planned. *)
let launches_of ks f =
  let y, traced = Plan.traced f in
  let count k =
    List.length (List.filter (fun (t : Plan.traced) -> t.kernel = k) traced)
  in
  (y, List.map count ks)

(* [f ()] and the count of [k]'s launches it planned. *)
let launches k f =
  let y, counts = launches_of [ k ] f in
  (y, List.hd counts)

(* [f ()] under the limits [~sweeps], [~steps], [~qr], [~passes] and [~secular],
   the initial ones restored after. *)
let limited ?(sweeps = 60) ?(steps = max_int) ?(qr = 30) ?(passes = 256)
    ?(secular = 30) f =
  Fun.protect
    ~finally:(fun () ->
      Nx_metal.bound_iterations ~sweeps:60 ~steps:max_int ~qr:30 ~passes:256
        ~secular:30)
    (fun () ->
      Nx_metal.bound_iterations ~sweeps ~steps ~qr ~passes ~secular;
      f ())

(* Budgets from one launch's 50 ms down, each a third of the last. *)
let budgets = List.init 12 (fun i -> Plan.budget /. (3. ** float_of_int i))

(* At each budget down to the first at which [r] on [x] declines, it gives the
   bits it gives at the default. The most launches of [k], or of every kernel
   without [k], it planned. *)
let check_splits ~msg b ?k r x =
  let want = results b r x in
  let most = ref 0 in
  Fun.protect
    ~finally:(fun () -> Plan.bound_budget Plan.budget)
    (fun () ->
      let rec go = function
        | [] -> ()
        | s :: rest -> (
            Plan.bound_budget s;
            let (got, ys), n =
              match k with
              | Some k -> launches k (fun () -> call b r x)
              | None ->
                  let y, traced = Plan.traced (fun () -> call b r x) in
                  (y, List.length traced)
            in
            match got with
            | A.Declined -> ()
            | A.Done ->
                most := max !most n;
                same_results
                  ~msg:(Printf.sprintf "%s at a budget of %g s" msg s)
                  want (Array.map bits ys);
                go rest
            | got -> failf "%s: the call answered %a" msg pp_answer got)
      in
      go budgets);
  !most

(* Bounds *)

(* Of singular's a past Jacobi's extents, of 33 to 161 rows or columns and as
   many or fewer of the other, the values alone are |σ|'s within svd_bound max
   |σ|. *)
let law_svd_values (b : Support.backend) c =
  let big = 33 + c.n in
  let small = 1 + (c.seed mod big) in
  let m, n = if c.seed / 7 mod 2 = 0 then (big, small) else (small, big) in
  cover "past 32 both" (small > 32);
  cover "wide" (m < n);
  cover "tall" (m > n);
  let r = Random.State.make [| c.seed |] in
  let draw = [| repeating; clustered; graded |].(c.seed mod 3) in
  let sigma = Array.init (min m n) (draw r) in
  check_singular b ~vectors:Unchecked c.dt ~m ~n
    (singular c.dt ~m ~n r sigma)
    sigma

let bounds_laws () =
  group "each result meets its relation within its bound"
    [
      prop "eigh within γ(16n)" case (on everything (law_eigh bounds));
      prop "eigh finds a known spectrum" case (on everything law_eigh_spectrum);
      prop "svd by Jacobi within its bound" case
        (on svd_jacobi (law_svd bounds));
      prop "svd's values past 32 find a known spectrum" case
        (on everything law_svd_values);
      prop "eig's values pair on real matrices" case
        (on everything (law_eig ~vectors:false bounds));
      prop "eig's values find a known spectrum" case
        (on everything (law_eig_spectrum ~vectors:false));
    ]

(* Splitting *)

(* Batches of more of Jacobi's groups than a launch at a small budget holds. *)
let test_jacobi_splits dt =
  let b = metal () in
  let r = Random.State.make [| 5 |] and cx = is_complex dt in
  b.around (fun () ->
      let n = 5 and count = 600 in
      let x =
        of_values dt [| count; n; n |]
          (Array.concat (List.init count (fun _ -> hermitian r dt n)))
      in
      let k = if cx then K.Sp_eigh_jacobi_c64 else K.Sp_eigh_jacobi_f32 in
      let most = check_splits ~msg:"eigh" b ~k (Eigh true) x in
      greater ~msg:"eigh's launches at the least budget" int ~than:1 most;
      let m = 6 and n = 4 in
      let x =
        of_values dt [| count; m; n |]
          (Array.init (count * m * n) (fun _ -> draw r dt))
      in
      let k = if cx then K.Sp_svd_jacobi_c64 else K.Sp_svd_jacobi_f32 in
      let most = check_splits ~msg:"svd" b ~k (Svd (Some Reduced)) x in
      greater ~msg:"svd's launches at the least budget" int ~than:1 most)

(* The cyclic permutation of order [n], each row i's one element at column i + 1
   modulo n: the standard shifts stall on it, so that the exceptional ones
   run. *)
let cyclic n =
  Array.init (n * n) (fun p ->
      if p mod n = ((p / n) + 1) mod n then Complex.one else Complex.zero)

(* eig's QR in slices of at most 1, 3 and 7 sweeps, each a launch, gives the
   bits of one slice of all of them, on dense matrices and cyclic permutations,
   at the initial limit and at 10 sweeps in all, which some of them pass; and
   slices a small budget makes give them too. *)
let test_eig_slices dt =
  let b = metal () in
  let r = Random.State.make [| 7 |] in
  let k = if is_complex dt then K.Sp_eig_qr_c64 else K.Sp_eig_qr_f32 in
  let dense n = Array.init (n * n) (fun _ -> draw r dt) in
  b.around (fun () ->
      let mats =
        List.concat_map
          (fun n -> [ (n, cyclic n); (n, dense n) ])
          [ 3; 4; 5; 8 ]
      in
      let failed = ref 0 and converged = ref 0 in
      List.iter
        (fun qr ->
          let limit = qr * 10 in
          List.iter
            (fun (n, v) ->
              let x = of_values dt [| n; n |] v in
              let want, one =
                limited ~qr (fun () ->
                    launches k (fun () -> results b (Eig false) x))
              in
              equal ~msg:"slices of all the sweeps" int 1 one;
              if Array.exists Float.is_nan want.(0) then incr failed
              else incr converged;
              List.iter
                (fun slice ->
                  let got, sliced =
                    limited ~qr ~steps:slice (fun () ->
                        launches k (fun () -> results b (Eig false) x))
                  in
                  let msg =
                    Printf.sprintf "order %d, limit %d, slices of %d" n limit
                      slice
                  in
                  equal ~msg:(msg ^ ": launches") int
                    ((limit + slice - 1) / slice)
                    sliced;
                  same_results ~msg want got)
                [ 1; 3; 7 ])
            mats)
        [ 30; 1 ];
      greater ~msg:"matrices past the limit" int ~than:0 !failed;
      greater ~msg:"matrices that converge" int ~than:0 !converged;
      let most =
        check_splits ~msg:"order 8" b ~k (Eig false)
          (of_values dt [| 8; 8 |] (dense 8))
      in
      greater ~msg:"slices at the least budget" int ~than:1 most)

(* G T G⁻¹ of order [n], T tridiagonal of ones (times 1 + i on complex [dt]) and
   G = diag 2^(s i): balancing scales it in passes that grow with [s], 1 at s =
   1 and 8 at s = 5 and n = 8 by LAPACK's xGEBAL. *)
let ramped dt n ~s =
  let one =
    if is_complex dt then { Complex.re = 1.; im = 1. } else Complex.one
  in
  Array.init (n * n) (fun p ->
      let i = p / n and j = p mod n in
      if abs (i - j) > 1 then Complex.zero
      else
        let x = Float.ldexp 1. (s * (i - j)) in
        Complex.mul one { Complex.re = x; im = 0. })

(* P u Pᵀ of the order-[n] matrix [u], P a permutation by [r]. *)
let permuted r n u =
  let p = Array.init n Fun.id in
  for i = n - 1 downto 1 do
    let j = Random.State.int r (i + 1) in
    let x = p.(i) in
    p.(i) <- p.(j);
    p.(j) <- x
  done;
  Array.init (n * n) (fun q -> u.((p.(q / n) * n) + p.(q mod n)))

(* P u Pᵀ of order [n], u upper triangular but for a dense block in rows and
   columns [3, n − 4] and P a permutation by [r]: balancing isolates rows from
   the bottom, then columns from the top, before it scales the block. *)
let isolating r dt n =
  permuted r n
    (Array.init (n * n) (fun p ->
         let i = p / n and j = p mod n in
         let block = i >= 3 && j >= 3 && i <= n - 4 && j <= n - 4 in
         if i <= j || block then draw r dt else Complex.zero))

(* P t Pᵀ of order [n], t triangular, upper where [upper], and P a permutation
   by [r]: isolation's worst case, which ends with no block left to scale. *)
let triangle r dt n ~upper =
  permuted r n
    (Array.init (n * n) (fun p ->
         let i = p / n and j = p mod n in
         if if upper then i <= j else i >= j then draw r dt else Complex.zero))

(* Balancing's isolation and scaling, each launch at most 1, 2 and 3 of their
   steps, give the bits of a launch of all of them, in as many launches as their
   steps need: on matrices that isolate rows and columns, on permuted triangles,
   which isolate every index, and on one whose scaling takes 8 passes. *)
let test_balance_splits dt =
  let b = metal () in
  let r = Random.State.make [| 9 |] and cx = is_complex dt in
  let iso = if cx then K.Sp_eig_iso_c64 else K.Sp_eig_iso_f32 in
  let rows = if cx then K.Sp_eig_rows_c64 else K.Sp_eig_rows_f32 in
  let n = 12 and passes = 10 in
  let mats =
    [
      ("isolating", isolating r dt n);
      ("isolating", isolating r dt n);
      ("upper triangle", triangle r dt n ~upper:true);
      ("lower triangle", triangle r dt n ~upper:false);
      ("slow", ramped dt n ~s:5);
    ]
  in
  b.around (fun () ->
      List.iter
        (fun (name, v) ->
          let x = of_values dt [| n; n |] v in
          let want = limited ~passes (fun () -> results b (Eig false) x) in
          List.iter
            (fun steps ->
              let got, counts =
                limited ~steps ~passes (fun () ->
                    launches_of [ iso; rows ] (fun () ->
                        results b (Eig false) x))
              in
              let isos = List.nth counts 0 and scales = List.nth counts 1 in
              let msg = Printf.sprintf "%s, %d steps a launch: %s" name steps in
              let each total = (total + steps - 1) / steps in
              equal
                ~msg:(msg "isolation's launches")
                int
                (2 * each (n + 1))
                isos;
              equal ~msg:(msg "scaling's launches") int
                (each (passes * (n + 1)))
                scales;
              same_results ~msg:(msg "bits") want got)
            [ 1; 2; 3 ])
        mats)

(* eigh past 32, its reduction's panels, divide and conquer's merges and the
   back-transformation, gives the bits of the default budget at budgets that
   split its steps into more launches: matrices of order 70, two full panels and
   part of a third, merged at two levels, with the vectors and without, more of
   them than a launch's groups at a small budget hold. *)
let test_eigh_splits dt =
  let b = metal () in
  let r = Random.State.make [| 14 |] in
  let n = 70 and count = 300 in
  let x =
    of_values dt [| count; n; n |]
      (Array.concat (List.init count (fun _ -> hermitian r dt n)))
  in
  b.around (fun () ->
      List.iter
        (fun (rt, name) ->
          let _, traced = Plan.traced (fun () -> call b rt x) in
          Array.iteri
            (fun k y ->
              equal
                ~msg:(Printf.sprintf "%s: result %d's numbers" name k)
                int (Array.length y) (numbers y))
            (results b rt x);
          let most = check_splits ~msg:name b rt x in
          greater
            ~msg:(name ^ ": launches at the least budget")
            int ~than:(List.length traced) most)
        [ (Eigh true, "eigh"); (Eigh false, "eigh's values") ])

(* Bisection, each launch at most 1, 2 and 3 of its steps, gives the bits of a
   launch of all of them, in as many launches as a value's 128 halvings of a
   test and a Sturm count each, and its last test, need: svd's values of two 40
   x 3 matrices, a Golub-Kahan matrix of order 6 each. *)
let test_bisect_splits dt =
  let b = metal () in
  let r = Random.State.make [| 10 |] in
  let m = 40 and n = 3 in
  let x =
    of_values dt [| 2; m; n |] (Array.init (2 * m * n) (fun _ -> draw r dt))
  in
  b.around (fun () ->
      let want = results b (Svd None) x in
      List.iter
        (fun steps ->
          let got, count =
            limited ~steps (fun () ->
                launches K.Sp_tri_bisect (fun () -> results b (Svd None) x))
          in
          let msg = Printf.sprintf "%d steps a launch: %s" steps in
          let total = (128 * ((2 * n) + 1)) + 1 in
          equal ~msg:(msg "launches") int ((total + steps - 1) / steps) count;
          same_results ~msg:(msg "bits") want got)
        [ 1; 2; 3 ])

(* Divide and conquer's roots, each launch at most 1, 2 and 3 of a root's steps,
   give the bits of one launch of them all, in as many launches as a root's
   start and 30 secular steps need, at each of order 70's two merge levels,
   beside the weights' and the vectors' launch. *)
let test_secular_splits dt =
  let b = metal () in
  let r = Random.State.make [| 15 |] in
  let n = 70 and count = 2 in
  let x =
    of_values dt [| count; n; n |]
      (Array.concat (List.init count (fun _ -> hermitian r dt n)))
  in
  b.around (fun () ->
      let want = results b (Eigh true) x in
      List.iter
        (fun steps ->
          let got, launched =
            limited ~steps (fun () ->
                launches K.Sp_dc_secular (fun () -> results b (Eigh true) x))
          in
          let msg = Printf.sprintf "%d steps a launch: %s" steps in
          let roots = (31 + steps - 1) / steps in
          equal ~msg:(msg "launches") int (2 * (roots + 2)) launched;
          same_results ~msg:(msg "bits") want got)
        [ 1; 2; 3 ])

(* Deflation's decisions, each launch at most 1, 2 and 3 of a merge's steps,
   give the bits of one launch of them all, in as many launches as a merge's
   values and its end need: 64 and 70 at order 70's two merge levels. *)
let test_deflate_splits dt =
  let b = metal () in
  let r = Random.State.make [| 23 |] in
  let n = 70 and count = 2 in
  let x =
    of_values dt [| count; n; n |]
      (Array.concat (List.init count (fun _ -> hermitian r dt n)))
  in
  b.around (fun () ->
      let want = results b (Eigh true) x in
      List.iter
        (fun steps ->
          let got, launched =
            limited ~steps (fun () ->
                launches K.Sp_dc_deflate (fun () -> results b (Eigh true) x))
          in
          let msg = Printf.sprintf "%d steps a launch: %s" steps in
          let rounds total = (total + steps - 1) / steps in
          equal ~msg:(msg "launches") int (rounds 65 + rounds 71) launched;
          same_results ~msg:(msg "bits") want got)
        [ 1; 2; 3 ])

(* A merge's product launches whole merges where they fit the budget, else rows
   of the merge's tiles, else tiles of a row. eigh of order 840 at a 27th of the
   budget takes each shape, its tiles of 64 rows and roots at merge level l
   numbering ⌈min(32 · 2^l, 840) / 64⌉ a side, with the bits of the call at the
   whole budget. *)
let test_product_shapes dt =
  let b = metal () in
  let r = Random.State.make [| 24 |] in
  let n = 840 in
  let x = of_values dt [| 1; n; n |] (hermitian r dt n) in
  let word (t : Plan.traced) i =
    Int64.to_int (Bytes.get_int64_le t.words (8 * i))
  in
  let shape (t : Plan.traced) =
    let tiles = (Int.min (32 lsl word t K.sp_merge_lev) n + 63) / 64 in
    match t.groups with
    | x, y, _ when x = tiles && y = tiles -> "merges"
    | x, _, 1 when x = tiles -> "rows of tiles"
    | _, 1, 1 -> "tiles of a row"
    | x, y, z -> Printf.sprintf "%d x %d x %d of %d tiles" x y z tiles
  in
  b.around (fun () ->
      limited ~sweeps:12 (fun () ->
          let want = results b (Eigh true) x in
          let got, traced =
            Fun.protect
              ~finally:(fun () -> Plan.bound_budget Plan.budget)
              (fun () ->
                Plan.bound_budget (Plan.budget /. 27.);
                Plan.traced (fun () -> results b (Eigh true) x))
          in
          let shapes =
            List.sort_uniq compare
              (List.filter_map
                 (fun (t : Plan.traced) ->
                   if t.kernel = K.Sp_dc_product then Some (shape t) else None)
                 traced)
          in
          equal (list string)
            [ "merges"; "rows of tiles"; "tiles of a row" ]
            shapes;
          same_results ~msg:"bits" want got))

let splitting () =
  group "no bit follows how a step splits into launches"
    [
      cases "eigh's steps" ~id:pp_dt floats test_eigh_splits;
      cases "divide and conquer's roots" ~id:pp_dt floats test_secular_splits;
      cases "deflation's decisions" ~id:pp_dt floats test_deflate_splits;
      cases "a merge's product" ~id:pp_dt floats test_product_shapes;
      cases "bisection's steps" ~id:pp_dt floats test_bisect_splits;
      cases "balancing's steps" ~id:pp_dt floats test_balance_splits;
      cases "Jacobi's groups" ~id:pp_dt floats test_jacobi_splits;
      cases "eig's QR slices" ~id:pp_dt floats test_eig_slices;
    ]

(* Layouts *)

(* A batch of two matrices through each layout, transposed, stepped, padded and
   reversed, gives the bits of its C-contiguous copy: eigh and svd by Jacobi,
   svd's values past 32 and eig's values. *)
let test_layouts dt =
  let b = metal () in
  let r = Random.State.make [| 13 |] in
  let dense m n =
    of_values dt [| 2; m; n |] (Array.init (2 * m * n) (fun _ -> draw r dt))
  in
  let hermitians n =
    of_values dt [| 2; n; n |]
      (Array.concat [ hermitian r dt n; hermitian r dt n ])
  in
  b.around (fun () ->
      List.iter
        (fun (name, rt, x) ->
          let want = results b rt x in
          List.iter
            (fun how ->
              let msg = Format.asprintf "%s, %a" name pp_how how in
              same_results ~msg want (results b rt (laid_out how x)))
            [ Transposed; Stepped; Padded; Reversed ])
        [
          ("eigh", Eigh true, hermitians 6);
          ("svd by Jacobi", Svd (Some Reduced), dense 7 5);
          ("svd's values", Svd None, dense 40 35);
          ("eig's values", Eig false, dense 6 6);
        ])

let layouts () =
  group "layouts"
    [
      cases "a strided operand gives its copy's bits" ~id:pp_dt floats
        test_layouts;
    ]

(* Limits *)

(* Of [good], [bad] and [good] again, the one past the limit is NaN in every
   result, the others the bits [good] has alone. *)
let check_limit b r ~dt ~n ~bad ~good =
  let alone = results b r (of_values dt [| n; n |] good) in
  let three = of_values dt [| 3; n; n |] (Array.concat [ good; bad; good ]) in
  Array.iteri
    (fun k y ->
      let per = Array.length alone.(k) in
      let part i = Array.sub y (i * per) per in
      let msg = Printf.sprintf "result %d of %s" k in
      equal ~msg:(msg "the first") same_bits alone.(k) (part 0);
      equal ~msg:(msg "the last") same_bits alone.(k) (part 2);
      equal
        ~msg:(msg "the one past the limit, numbers")
        int 0
        (numbers (part 1)))
    (results b r three)

(* One sweep of Jacobi, at order 8 and in divide and conquer's leaves of 32 at
   order 40: a diagonal matrix converges in it, a dense one does not. One step
   of a root of divide and conquer's secular equation at order 40: a diagonal
   matrix deflates every value and solves none, a dense one's roots do not
   converge in it. 10 QR sweeps in all: an upper triangular matrix deflates
   without one, the cyclic permutation of order 8 does not converge in them. *)
let test_limits dt =
  let b = metal () in
  let r = Random.State.make [| 8 |] in
  let n = 8 and leaves = 40 in
  let diagonal n =
    Array.init (n * n) (fun p ->
        if p / n = p mod n then
          { Complex.re = float_of_int (n - (p / n)); im = 0. }
        else Complex.zero)
  in
  let dense () = Array.init (n * n) (fun _ -> draw r dt) in
  b.around (fun () ->
      limited ~sweeps:1 (fun () ->
          List.iter
            (fun rt ->
              check_limit b rt ~dt ~n ~bad:(hermitian r dt n) ~good:(diagonal n))
            [ Eigh true; Eigh false ];
          check_limit b (Eigh true) ~dt ~n:leaves ~bad:(hermitian r dt leaves)
            ~good:(diagonal leaves);
          List.iter
            (fun rt ->
              check_limit b rt ~dt ~n ~bad:(dense ()) ~good:(diagonal n))
            [ Svd (Some Reduced); Svd None ]);
      limited ~secular:1 (fun () ->
          check_limit b (Eigh true) ~dt ~n:leaves ~bad:(hermitian r dt leaves)
            ~good:(diagonal leaves));
      limited ~qr:1 (fun () ->
          let upper =
            Array.mapi
              (fun p z -> if p / n <= p mod n then z else Complex.zero)
              (dense ())
          in
          check_limit b (Eig false) ~dt ~n ~bad:(cyclic n) ~good:upper))

(* Balancing stops at its pass limit. At 1, 2 and 3 passes, a matrix that needs
   8 still gets values, all numbers, of which some bits differ from the default
   limit's; one that needs 1 keeps the default's bits; and at 8 passes and more
   both keep them. *)
let test_passes dt =
  let b = metal () in
  let n = 8 in
  let slow = of_values dt [| n; n |] (ramped dt n ~s:5) in
  let quick = of_values dt [| n; n |] (ramped dt n ~s:1) in
  b.around (fun () ->
      let want_slow = results b (Eig false) slow in
      let want_quick = results b (Eig false) quick in
      let differ = ref 0 in
      List.iter
        (fun passes ->
          let msg = Printf.sprintf "at %d passes: %s" passes in
          let got_slow, got_quick =
            limited ~passes (fun () ->
                (results b (Eig false) slow, results b (Eig false) quick))
          in
          equal ~msg:(msg "the quick one") same_bits want_quick.(0)
            got_quick.(0);
          equal
            ~msg:(msg "the slow one's numbers")
            int
            (Array.length got_slow.(0))
            (numbers got_slow.(0));
          if passes >= 8 then
            equal ~msg:(msg "the slow one") same_bits want_slow.(0) got_slow.(0)
          else
            let b = Array.map Int64.bits_of_float in
            if b got_slow.(0) <> b want_slow.(0) then incr differ)
        [ 1; 2; 3; 8; 9 ];
      greater ~msg:"limits that change the slow one's bits" int ~than:0 !differ)

(* eig's values of order 340, past where its balancing ran as one launch, find a
   known spectrum, under 4 max(10, h) QR sweeps in all and 16 passes of
   balancing to keep the call's worst case short. *)
let test_past_balancing dt =
  let b = metal () in
  let c =
    {
      dt;
      batch = [||];
      n = 340;
      r = 0;
      triangle = Lower;
      adjoint = false;
      unit_diagonal = false;
      seed = 11;
      how = Contiguous;
      scale = Plain;
    }
  in
  b.around (fun () ->
      limited ~qr:4 ~passes:16 (fun () -> law_eig_spectrum ~vectors:false b c))

(* eigh's values of order 512 and svd's of a 256 x 256 matrix, past where a
   bisection ran as one launch, find a known spectrum. *)
let test_past_bisection dt =
  let b = metal () in
  let r = Random.State.make [| 12 |] in
  let m = 256 and n = 256 in
  let sigma = Array.init n (graded r) in
  let l = Array.init 512 (graded r) in
  b.around (fun () ->
      check_spectrum b ~vectors:Unchecked dt ~k:200
        ~u:(reflector_draws r dt 512) l;
      check_singular b ~vectors:Unchecked dt ~m ~n
        (singular dt ~m ~n r sigma)
        sigma)

(* eigh of order 129 with its vectors, at a budget of a 27th of a launch's,
   where a root's 30 secular steps in one launch pass it from order 78, meets
   its relations. Jacobi, which runs its leaves of 32 in one launch, is held to
   12 sweeps, in which they converge, to fit that budget. *)
let test_past_secular dt =
  let b = metal () in
  let c =
    {
      dt;
      batch = [||];
      n = 129;
      r = 0;
      triangle = Lower;
      adjoint = false;
      unit_diagonal = false;
      seed = 16;
      how = Contiguous;
      scale = Plain;
    }
  in
  Fun.protect
    ~finally:(fun () -> Plan.bound_budget Plan.budget)
    (fun () ->
      Plan.bound_budget (Plan.budget /. 27.);
      limited ~sweeps:12 (fun () -> b.around (fun () -> law_eigh bounds b c)))

(* eigh of order 512 finds a known spectrum, its vectors' relations on sampled
   elements, at a budget of a 27th of a launch's, where a group's deflation of a
   merge passes it from order 433 and one launch of a merge's product tiles from
   order 321. Jacobi is held to 12 sweeps, as above. *)
let test_past_deflation dt =
  let b = metal () in
  let r = Random.State.make [| 17 |] in
  let n = 512 in
  let l = Array.init n (graded r) in
  let u = reflector_draws r dt n in
  Fun.protect
    ~finally:(fun () -> Plan.bound_budget Plan.budget)
    (fun () ->
      Plan.bound_budget (Plan.budget /. 27.);
      limited ~sweeps:12 (fun () ->
          b.around (fun () ->
              check_spectrum b ~vectors:(Sampled 64) dt ~k:200 ~u l)))

(* eigh's values of order 840 find a known spectrum at a budget of a 27th of a
   launch's, where staging a matrix for its reduction in one group passes it
   from order 811 at float32 and 782 at complex64. *)
let test_past_stage dt =
  let b = metal () in
  let r = Random.State.make [| 18 |] in
  let n = 840 in
  let l = Array.init n (graded r) in
  let u = reflector_draws r dt n in
  Fun.protect
    ~finally:(fun () -> Plan.bound_budget Plan.budget)
    (fun () ->
      Plan.bound_budget (Plan.budget /. 27.);
      b.around (fun () -> check_spectrum b ~vectors:Unchecked dt ~k:8 ~u l))

(* eigh of order 1,900 finds a known spectrum, its vectors' relations on sampled
   elements, at a budget of a 9th of a launch's, where a group's sort of a merge
   passes it from order 1,811. Jacobi is held to 12 sweeps, as above. *)
let test_past_sort dt =
  let b = metal () in
  let r = Random.State.make [| 19 |] in
  let n = 1900 in
  let l = Array.init n (graded r) in
  let u = reflector_draws r dt n in
  Fun.protect
    ~finally:(fun () -> Plan.bound_budget Plan.budget)
    (fun () ->
      Plan.bound_budget (Plan.budget /. 9.);
      limited ~sweeps:12 (fun () ->
          b.around (fun () ->
              check_spectrum b ~vectors:(Sampled 64) dt ~k:8 ~u l)))

(* eigh of order 840 finds a known spectrum, its vectors' relations on sampled
   elements, at a budget of a 27th of a launch's, where a thread's decisions of
   a merge's deflation pass it from order 710. Jacobi is held to 12 sweeps, as
   above. *)
let test_past_decisions dt =
  let b = metal () in
  let r = Random.State.make [| 21 |] in
  let n = 840 in
  let l = Array.init n (graded r) in
  let u = reflector_draws r dt n in
  Fun.protect
    ~finally:(fun () -> Plan.bound_budget Plan.budget)
    (fun () ->
      Plan.bound_budget (Plan.budget /. 27.);
      limited ~sweeps:12 (fun () ->
          b.around (fun () ->
              check_spectrum b ~vectors:(Sampled 64) dt ~k:8 ~u l)))

(* svd's values of a 500 x 700 matrix find known singular values at a budget of
   a 100th of a launch's, where staging a matrix for its reduction in one group
   passes it from [m n] 343,553 at float32 and 313,537 at complex64. *)
let test_past_bd_stage dt =
  let b = metal () in
  let r = Random.State.make [| 20 |] in
  let m = 500 and n = 700 in
  let sigma = Array.init m (graded r) in
  Fun.protect
    ~finally:(fun () -> Plan.bound_budget Plan.budget)
    (fun () ->
      Plan.bound_budget (Plan.budget /. 100.);
      b.around (fun () ->
          check_singular b ~vectors:Unchecked dt ~m ~n
            (singular dt ~m ~n r sigma)
            sigma))

(* Subnormal elements *)

(* [v] with the elements at [at] below 2^-126 in magnitude, of both signs, in a
   real or, on complex dtypes by [r], an imaginary part. *)
let sprinkle r dt v ~at =
  let tiny () =
    let sign = if Random.State.bool r then 1. else -1. in
    Float.ldexp sign (-127 - Random.State.int r 20)
  in
  let v = Array.copy v in
  List.iter
    (fun p ->
      let (z : Complex.t) = v.(p) in
      v.(p) <-
        (if is_complex dt && Random.State.bool r then { z with im = tiny () }
         else { z with re = tiny () }))
    at;
  v

(* [v] with each part below 2^-126 in magnitude a zero of its sign. *)
let flush v =
  let f x = if Float.abs x < 0x1p-126 then Float.copy_sign 0. x else x in
  Array.map (fun (z : Complex.t) -> { Complex.re = f z.re; im = f z.im }) v

let check_flushed b r dt shape v =
  same_results ~msg:"a and a′"
    (results b r (of_values dt shape (flush v)))
    (results b r (of_values dt shape v))

let test_subnormal dt =
  let b = metal () in
  let r = Random.State.make [| 9 |] in
  let dense k = Array.init k (fun _ -> draw r dt) in
  b.around (fun () ->
      let n = 6 in
      let h =
        sprinkle r dt (hermitian r dt n) ~at:[ n; (2 * n) + 2; (3 * n) + 1 ]
      in
      List.iter
        (fun rt -> check_flushed b rt dt [| n; n |] h)
        [ Eigh true; Eigh false ];
      let m = 7 and k = 5 in
      let g = sprinkle r dt (dense (m * k)) ~at:[ 0; 9; 20 ] in
      List.iter
        (fun rt -> check_flushed b rt dt [| m; k |] g)
        [ Svd (Some Reduced); Svd None ];
      let m = 40 and k = 35 in
      let g = sprinkle r dt (dense (m * k)) ~at:[ 0; 41; 700 ] in
      check_flushed b (Svd None) dt [| m; k |] g;
      (* A row whose one element off the diagonal is subnormal, which balancing
         isolates on a′. *)
      let row = 2 in
      let e =
        Array.mapi
          (fun p z -> if p / n = row && p mod n <> row then Complex.zero else z)
          (dense (n * n))
      in
      let e = sprinkle r dt e ~at:[ (row * n) + 4; 7 ] in
      check_flushed b (Eig false) dt [| n; n |] e)

(* a = 2^-124 m, normal, of a real matrix m with a pair whose imaginary parts,
   scaled, lie below 2^-126: w holds ±2^-126, the positive first. *)
let test_flushed_pair () =
  let b = metal () in
  b.around (fun () ->
      let n = 3 and m = [| -1.; 3.; 1.; 3.; 1.; 1.; 1.; -1.; 2. |] in
      let re x = { Complex.re = Float.ldexp x (-124); im = 0. } in
      let x = of_values (D.Any D.Float32) [| n; n |] (Array.map re m) in
      let w = (results b (Eig false) x).(0) in
      let ims =
        List.filteri (fun i x -> i mod 2 = 1 && x <> 0.) (Array.to_list w)
      in
      equal ~msg:"the pair's imaginary parts" (list float_exact)
        [ 0x1p-126; -0x1p-126 ] ims)

let limits_and_reads () =
  group "limits and subnormal elements"
    [
      cases "a matrix past a limit is NaN alone" ~id:pp_dt floats test_limits;
      cases "balancing stops at its pass limit" ~id:pp_dt floats test_passes;
      cases "eig computes past balancing's old limit" ~id:pp_dt floats
        test_past_balancing;
      cases "eigh's and svd's values compute past bisection's old limit"
        ~id:pp_dt floats test_past_bisection;
      cases "eigh's vectors compute past the secular roots' old limit" ~id:pp_dt
        floats test_past_secular;
      cases
        "eigh's vectors compute past deflation's and the product's old limits"
        ~id:pp_dt floats test_past_deflation;
      cases "eigh's values compute past the stage's old limit" ~id:pp_dt floats
        test_past_stage;
      cases "eigh's vectors compute past the sort's old limit" ~id:pp_dt floats
        test_past_sort;
      cases "eigh's vectors compute past the decisions' old limit" ~id:pp_dt
        floats test_past_decisions;
      cases "svd's values compute past the stage's old limit" ~id:pp_dt floats
        test_past_bd_stage;
      cases "results are a′'s bit for bit" ~id:pp_dt floats test_subnormal;
      test "eig's flushed pair is ±2^-126" test_flushed_pair;
    ]

(* Tall columns *)

(* nx_metal.mli's chunk: a column of more than 65,536 rows sums in chunks of
   65,536 rows. *)
let chunk = 65_536

(* An m x n matrix whose column j, or row j where it is wide, is nonzero only on
   the rows, or columns, ≡ j modulo min(m, n), each entry [scale j] times a
   uniform draw: its columns are orthogonal, so its singular values are their
   norms, found in float64 from the entries rounded to float32's. *)
let orthogonal r dt ~m ~n ~scale =
  let k = min m n and cx = is_complex dt in
  let round x = Int32.float_of_bits (Int32.bits_of_float x) in
  let norms = Array.make k 0. in
  let a =
    Array.init (m * n) (fun p ->
        let i = p / n and j = p mod n in
        let line, along = if m >= n then (j, i) else (i, j) in
        if along mod k <> line then Complex.zero
        else begin
          let s = scale line in
          let re = round (s *. uniform r)
          and im = if cx then round (s *. uniform r) else 0. in
          let x = { Complex.re; im } in
          norms.(line) <- norms.(line) +. Complex.norm2 x;
          x
        end)
  in
  (of_values dt [| m; n |] a, Array.map Float.sqrt norms)

(* A column of [m] rows whose parts lie in [0.75, 1) 2^41, below the largest
   that staging leaves unscaled, so that past 2^22 rows its sum of squares
   passes 2^103: the matrix and its norm, found in float64 from the parts
   rounded to float32's. *)
let large r dt m =
  let round x = Int32.float_of_bits (Int32.bits_of_float x) in
  let part () = round (Float.ldexp (0.875 +. (0.125 *. uniform r)) 41) in
  let cx = is_complex dt in
  let a =
    Array.init m (fun _ ->
        let re = part () in
        { Complex.re; im = (if cx then part () else 0.) })
  in
  let norm = Array.fold_left (fun s x -> s +. Complex.norm2 x) 0. a in
  (of_values dt [| m; 1 |] a, [| Float.sqrt norm |])

(* svd's values of tall and wide matrices whose columns pass a chunk, by one, a
   chunk and many, against their orthogonal columns' norms, and of a dense one
   against its known values, each within nx_metal.mli's bound: a column of a
   chunk exactly, whose sums take one chunk, and one row more, whose x takes
   chunks (an lsum launch for each of the norm's two phases) and whose v does in
   each column but the last (a whc launch); a last chunk of one row; a column
   whose sum of squares is below 2^-103, and one whose sum passes 2^103, a
   finite value, which the scaled sum takes; a column whose norm is below
   2^-103, which reflects scaled by 2^103, a subnormal entry flushed among them;
   a zero column, which reflects by τ = 0; a NaN in the last chunk, which makes
   the matrix NaN. *)
let test_tall_values dt =
  let b = metal () in
  let r = Random.State.make [| 21 |] in
  let one = Fun.const 1. in
  let scaled col s j = if j = col then s else 1. in
  let check ~m ~n (x, sigma) =
    check_singular b ~vectors:Unchecked dt ~m ~n x sigma
  in
  b.around (fun () ->
      let chunked m =
        snd
          (launches_of [ K.Sp_bd_lsum; K.Sp_bd_whc ] (fun () ->
               call b (Svd None) (fst (orthogonal r dt ~m ~n:3 ~scale:one))))
      in
      equal ~msg:"chunked steps of a chunk's x" (list int) [ 0; 1 ]
        (chunked (chunk + 1));
      equal ~msg:"chunked steps past a chunk's x" (list int) [ 2; 2 ]
        (chunked (chunk + 2));
      check ~m:(chunk + 1) ~n:3 (orthogonal r dt ~m:(chunk + 1) ~n:3 ~scale:one);
      check ~m:(chunk + 2) ~n:3 (orthogonal r dt ~m:(chunk + 2) ~n:3 ~scale:one);
      check
        ~m:((3 * chunk) + 5)
        ~n:4
        (orthogonal r dt ~m:((3 * chunk) + 5) ~n:4 ~scale:one);
      check ~m:4
        ~n:((2 * chunk) + 7)
        (orthogonal r dt ~m:4 ~n:((2 * chunk) + 7) ~scale:one);
      let m = (2 * chunk) + 3 in
      let sigma = Array.init 3 (graded r) in
      check ~m ~n:3 (singular dt ~m ~n:3 r sigma, sigma);
      let m = (2 * chunk) + 9 in
      check ~m ~n:3 (orthogonal r dt ~m ~n:3 ~scale:(scaled 1 1e-19));
      check ~m ~n:3 (orthogonal r dt ~m ~n:3 ~scale:(scaled 1 1e-36));
      check ~m ~n:3 (orthogonal r dt ~m ~n:3 ~scale:(scaled 2 0.));
      let x, sigma = orthogonal r dt ~m ~n:3 ~scale:(scaled 2 1e-36) in
      let v = values x in
      let last = 2 + (3 * ((m - 3) / 3)) in
      v.((last * 3) + 2) <- { Complex.re = 1e-40; im = 0. };
      check ~m ~n:3 (of_values dt [| m; 3 |] v, sigma);
      let big = (1 lsl 22) + 1 in
      let x, sigma = large r dt big in
      check ~m:big ~n:1 (x, sigma);
      equal ~msg:"a large column's value" int 1
        (numbers (results b (Svd None) x).(0));
      let v = values (fst (orthogonal r dt ~m ~n:3 ~scale:one)) in
      v.((m * 3) - 1) <- { Complex.re = Float.nan; im = 0. };
      match results b (Svd None) (of_values dt [| m; 3 |] v) with
      | [| s |] -> equal ~msg:"a NaN's values" int 0 (numbers s)
      | _ -> fail "svd's values: one result")

(* svd's values of tall matrices give the bits they give alone at each budget
   down to a 3^11th of a launch's, at batch positions 0 and 2: a column of 10
   chunks, whose chunked steps split across launches, beside one whose second
   column's sum of squares is below 2^-103, which the scaled sum takes alone in
   the batch. *)
let test_tall_splits dt =
  let b = metal () in
  let r = Random.State.make [| 22 |] in
  let m = (10 * chunk) + 1 in
  let x, _ = orthogonal r dt ~m ~n:2 ~scale:(Fun.const 1.) in
  let y, _ =
    orthogonal r dt ~m ~n:2 ~scale:(fun j -> if j = 1 then 1e-19 else 1.)
  in
  let batch =
    of_values dt [| 3; m; 2 |] (Array.concat [ values x; values y; values x ])
  in
  b.around (fun () ->
      let alone = results b (Svd None) x and other = results b (Svd None) y in
      let all = results b (Svd None) batch in
      let at i = Array.sub all.(0) (i * 2) 2 in
      same_results ~msg:"position 0" alone [| at 0 |];
      same_results ~msg:"position 1" other [| at 1 |];
      same_results ~msg:"position 2" alone [| at 2 |];
      let most = check_splits ~msg:"the batch" b (Svd None) batch in
      let _, traced = Plan.traced (fun () -> call b (Svd None) batch) in
      greater ~msg:"launches at the least budget" int ~than:(List.length traced)
        most)

(* svd's values plan, with no buffer, for tall float32 and complex64 matrices of
   2^33 bytes, of 1, 4, 32 and 256 columns, whose workspace fills a 32 GiB
   device: no step of a column declines below memory, and each launch keeps
   within the budget. *)
let test_tall_plans () =
  let facts buffer dt shape =
    {
      Nx_metal.Operands.dtype = dt;
      layout = L.contiguous (Iarray.of_array shape);
      buffer;
      skew = 0;
      residue = 0;
      pad = None;
    }
  in
  List.iter
    (fun ((D.Any d as dt), k) ->
      let m = (1 lsl 33) / (D.bytes d 1 * k) in
      let what = Printf.sprintf "%s %d x %d" (D.name d) m k in
      match
        Nx_metal.Spectral.svd_of_facts ~holds:K.threads
          ~s:(facts 0 (D.Any D.Float32) [| k |])
          (facts 1 dt [| m; k |])
      with
      | Error a -> failf "%s: %a" what pp_answer a
      | Ok None -> failf "%s: no plan" what
      | Ok (Some p) ->
          Array.iter
            (fun (l : Plan.launch) ->
              at_most ~msg:(what ^ ", a launch") float_exact ~than:Plan.budget
                (Plan.seconds ~threads:(K.threads l.kernel) ~groups:l.groups
                   ~resident:l.resident l.cost))
            p.launches)
    (List.concat_map
       (fun k -> D.[ (Any Float32, k); (Any Complex64, k) ])
       [ 1; 4; 32; 256 ])

let tall () =
  group "tall columns"
    [
      cases "svd's values past a chunk" ~id:pp_dt floats test_tall_values;
      cases "no bit follows a tall column's launches" ~id:pp_dt floats
        test_tall_splits;
      test "svd's values plan below memory" test_tall_plans;
    ]

(* Memory *)

(* eigh past 32 takes [n² + 161 n + 2048] elements, [3 n] floats, [6 n] ints and
   32 bytes a matrix for [w] alone, and [5 n² + 129 n + 2048] elements, [3 n² +
   82 n] floats, [8 n] ints and 24 bytes with [v], alignment at most 512 bytes a
   matrix and 2 KiB a call; an [a] that is not C-contiguous [m n] elements more
   a matrix, for its copy. svd's values past 32 take [M (k + 1) + 2 k] elements,
   [7 k] floats, [6 k] ints and 88 bytes a matrix; eig's values [3 n² + n]
   elements, [2 n² + 5 n] floats and 60 bytes; alignment at most 256 bytes a
   matrix and 1 KiB a call. *)
let test_memory () =
  let b = metal () in
  let module Kn = (val b.kernels) in
  let wait x = Rig.Buffer.wait (A.buffer x) Read_write in
  let make d s = A.create b.device d (Iarray.of_array s) in
  b.around (fun () ->
      let nb = 2 and m = 70 and k = 40 in
      let a = make D.Float32 [| nb; m; k |]
      and s = make D.Float32 [| nb; k |] in
      let per = (4 * ((m * (k + 1)) + (2 * k) + (13 * k))) + 88 + 256 in
      Kernels_cases.loop_held b
        ~bound:((nb * per) + 1024)
        (fun () ->
          equal answer A.Done (Kn.svd ~s a);
          wait s);
      let at =
        Option.get
          (A.move
             (Nx_array.Move.Permute (Iarray.of_array [| 0; 2; 1 |]))
             (make D.Float32 [| nb; k; m |]))
      in
      Kernels_cases.loop_held b
        ~bound:((nb * (per + (4 * m * k))) + 1024)
        (fun () ->
          equal answer A.Done (Kn.svd ~s at);
          wait s);
      let nb = 2 and n = 40 in
      let a = make D.Float32 [| nb; n; n |]
      and w = make D.Float32 [| nb; n |]
      and v = make D.Float32 [| nb; n; n |] in
      let per = (4 * ((n * n) + (170 * n) + 2048)) + 32 + 512 in
      Kernels_cases.loop_held b
        ~bound:((nb * per) + 2048)
        (fun () ->
          equal answer A.Done (Kn.eigh ~w a);
          wait w);
      let per = (4 * ((8 * n * n) + (219 * n) + 2048)) + 24 + 512 in
      Kernels_cases.loop_held b
        ~bound:((nb * per) + 2048)
        (fun () ->
          equal answer A.Done (Kn.eigh ~w ~v a);
          wait v);
      let nb = 3 and n = 16 in
      let a = make D.Float32 [| nb; n; n |]
      and w = make D.Complex64 [| nb; n |] in
      let per = (4 * ((3 * n * n) + n + (2 * n * n) + (5 * n))) + 60 + 256 in
      Kernels_cases.loop_held b
        ~bound:((nb * per) + 1024)
        (fun () ->
          equal answer A.Done (Kn.eig ~w a);
          wait w))

(* Declines *)

(* svd's vectors past 32, eig's vectors and an operand whose batch axes do not
   merge into one strided axis decline. *)
let test_declines dt =
  let b = metal () in
  let module Kn = (val b.kernels) in
  let (D.Any dt) = dt in
  let m s = A.create b.device dt (Iarray.of_array s) in
  let re s = A.create b.device D.Float32 (Iarray.of_array s) in
  let cx s = A.create b.device D.Complex64 (Iarray.of_array s) in
  let declines msg got = equal ~msg answer A.Declined got in
  let unmerged =
    Option.get
      (A.move
         (Nx_array.Move.Permute (Iarray.of_array [| 1; 0; 2; 3 |]))
         (m [| 2; 2; 3; 3 |]))
  in
  b.around (fun () ->
      declines "svd's vectors of 33 rows"
        (Kn.svd ~s:(re [| 2 |])
           ~vectors:(~u:(m [| 33; 2 |]), ~vh:(m [| 2; 2 |]))
           (m [| 33; 2 |]));
      declines "eig's vectors"
        (Kn.eig ~w:(cx [| 2 |]) ~v:(cx [| 2; 2 |]) (m [| 2; 2 |]));
      declines "an eigh of unmerged batch axes"
        (Kn.eigh ~w:(re [| 2; 2; 3 |]) unmerged))

let rest () =
  group "memory and declines"
    [
      test "calls in a loop hold one call's workspace" test_memory;
      cases "calls past the stated paths decline" ~id:pp_dt floats test_declines;
    ]

let () =
  exit
    (Windtrap.run "nx.metal spectral routines"
       [
         bounds_laws ();
         splitting ();
         layouts ();
         limits_and_reads ();
         tall ();
         rest ();
       ])
