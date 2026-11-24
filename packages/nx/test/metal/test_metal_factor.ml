(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.metal's factorisations, as nx_metal.mli states them, on the machine's
   Metal GPU 0, skipping without one: each result against its defining relation
   within nx.metal's bound, on the suite's cases, of operands whose subnormal
   elements read as zeros of their sign; LU, Cholesky and the solves bit for bit
   the unblocked algorithm's; every call one block does not hold declined. A
   float64 or complex128 call declines, so its laws check nothing. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module Support = Nx_kernels_support
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

(* The leaf, a Cholesky block's and an LU panel's order: 64, 32 at complex64. A
   qr is one launch where it has at most 32 columns and that launch's memory
   fits 32 KiB (nx_metal.mli). *)

let leaf c = if is_complex c.dt then 32 else 64
let lu_cols c = fst (lu_operand c)

let qr_fits c =
  let n = lu_cols c and m = c.n in
  let q = if factors c = Nx_kernel.Complete then m else min m n in
  let e = if is_complex c.dt then 8 else 4 in
  n <= 32 && m <= 1024
  && (64 + n + (n * n) + (n * (n - 1) / 2) + (n * q)) * e <= 32 * 1024

let in_place c = c.how = Contiguous

(* The machine's Metal GPU 0, opened at a test's first call; a test skips
   without it. *)
let metal () =
  match Support.metal () with
  | Some b -> b
  | None -> skip ~reason:"no Metal GPU nx.metal computes on" ()

(* Declines *)

(* An operand whose batch axes do not merge into one strided axis declines: [3;
   2] matrices of 4 x 4 seen as [2; 3] through a permutation. *)
let test_declines dt =
  let b = metal () in
  let module K = (val b.kernels) in
  let (D.Any dt) = dt in
  let m s = A.create b.device dt (Iarray.of_array s) in
  let unmerged () =
    Option.get
      (A.move
         (Nx_array.Move.Permute (Iarray.of_array [| 1; 0; 2; 3 |]))
         (m [| 3; 2; 4; 4 |]))
  in
  let pos () = A.create b.device D.Int64 (Iarray.of_array [| 2; 3; 4 |]) in
  let s = [| 2; 3; 4; 4 |] in
  let declines msg got = equal ~msg answer A.Declined got in
  b.around (fun () ->
      declines "cholesky" (K.cholesky Lower ~dst:(m s) (unmerged ()));
      declines "lu"
        (K.lu ~lu:(m s) ~pivots:(pos ()) ~perm:(pos ()) (unmerged ()));
      declines "qr" (K.qr ~q:(m s) ~r:(m s) (unmerged ()));
      declines "a solve's a"
        (K.solve_triangular Lower ~adjoint:false ~unit_diagonal:false ~dst:(m s)
           (unmerged ()) (m s));
      declines "a solve's b"
        (K.solve_triangular Lower ~adjoint:false ~unit_diagonal:false ~dst:(m s)
           (m s) (unmerged ())))

(* Placements at the one launch's edge *)

(* QRs whose header and partials pass half a group's memory, so that they lie in
   device memory, while the header and the tile would fit the group: the tile
   lies there too, since a tile lies in the group's memory only with its
   partials. [rows], [cols] and whether the factors are complete; lu_operand
   takes [cols] columns at a seed of 1 mod 3, complete factors at an odd
   seed. *)
let qr_edges =
  [
    (D.Any D.Complex64, 26, 27, false);
    (D.Any D.Complex64, 26, 29, false);
    (D.Any D.Complex64, 17, 30, false);
    (D.Any D.Complex64, 63, 20, true);
    (D.Any D.Complex64, 64, 24, true);
    (D.Any D.Float32, 129, 24, true);
  ]

let test_qr_edge (dt, m, n, complete) =
  let b = metal () in
  b.around (fun () -> law_qr b (fixed ~seed:(if complete then 1 else 4) dt m n))

(* Placement *)

module Plan = Nx_metal.Plan
module K = Nx_metal.Kernels

let strf = Printf.sprintf

(* Word [i] of a traced launch. *)
let word (t : Plan.traced) i = Int64.to_int (Bytes.get_int64_le t.words (8 * i))

(* The lanes a matrix and the matrices a group of the first panel or block
   launch among [ts]. *)
let placement ts =
  List.find_map
    (fun (t : Plan.traced) ->
      match t.kernel with
      | K.La_qr_panel_f32 | La_qr_panel_c64 ->
          Some (word t K.la_qr_g, word t K.la_qr_mats)
      | La_lu_panel_f32 | La_lu_panel_c64 ->
          Some (word t K.la_panel_g, word t K.la_panel_mats)
      | La_chol_f32 | La_chol_c64 ->
          Some (word t K.la_chol_g, word t K.la_chol_mats)
      | _ -> None)
    ts

(* The first launch of a kernel among [ks] in [ts]. *)
let launch_of ks ts =
  List.find_opt (fun (t : Plan.traced) -> List.mem t.kernel ks) ts

(* Matrix [k]'s results in [y] and alone in [z], bit for bit: floats by
   [same_bits], LU's positions as int64s. *)
let same_matrix ~msg k (y : A.any array) (z : A.any array) =
  Array.iteri
    (fun i (A.Any zi as z) ->
      let msg = strf "%s, matrix %d, result %d" msg k i in
      match A.dtype zi with
      | D.Int64 ->
          let per = Array.length (int64s z) in
          equal ~msg (array int)
            (Array.sub (int64s y.(i)) (k * per) per)
            (int64s z)
      | _ -> equal ~msg same_bits (bits (alone k y.(i))) (bits z))
    z

(* [r] on [ops], a batch of one axis, and on matrices [ks] of it alone: each
   one's bits as in the batch. The placement of the batch's call is [placed]'s
   argument. *)
let batch_alone (b : Support.backend) ~msg r ops ks placed =
  let y, ts = Plan.traced (fun () -> run b r ops) in
  placed ts;
  match y with
  | None -> ()
  | Some y ->
      List.iter
        (fun k ->
          match run b r (Array.map (alone k) ops) with
          | None -> ()
          | Some z -> same_matrix ~msg k y z)
        ks

(* The calls one block holds of a case, each a name, a routine and its
   operands. *)
let held_calls ?(only = fun _ -> true) c =
  List.filter_map
    (fun (ok, name, r, ops) ->
      if only name && Lazy.force ok then Some (name, r, Lazy.force ops)
      else None)
    [
      (lazy true, "cholesky", Cholesky c.triangle, lazy [| cholesky_operand c |]);
      (lazy true, "lu", Lu, lazy [| snd (lu_operand c) |]);
      (lazy true, "qr", Qr (factors c), lazy [| snd (lu_operand c) |]);
      (lazy true, "solve", solve c, lazy (solve_operands c));
    ]

(* Batches that fill the GPU: 64 to 128 matrices of at most a leaf's order, 64
   the fewest at which a panel takes 32 lanes a matrix. *)
let wide =
  let open Gen in
  with_pp pp_case
    (let* dt = of_list ~pp:pp_dtype [ D.Any D.Float32; D.Any D.Complex64 ] in
     let* n = int_range 1 (if is_complex dt then 32 else 64) in
     let* count = int_range 64 128 in
     let* r = of_list ~pp:Format.pp_print_int [ 1; 3 ] in
     let* triangle = of_list Nx_kernel.[ Lower; Upper ] in
     let* adjoint = bool in
     let* unit_diagonal = bool in
     let* seed = int_range 0 1_000_000 in
     let+ scale = of_list ~pp:pp_scale [ Plain; Graded; Subnormal ] in
     {
       dt;
       batch = [| count |];
       n;
       r;
       triangle;
       adjoint;
       unit_diagonal;
       seed;
       how = Contiguous;
       scale;
     })

(* One case for each of law_wide's covers, so that every run reaches them: 8 x 8
   matrices take 16 lanes and 16 a group, so a batch of 65 leaves 15 past its
   last, its partner and a SIMD-group wholly past it; 40 x 40 ones take 32
   lanes. *)
let wide_examples =
  List.map
    (fun (dt, n, count, seed) ->
      {
        dt;
        batch = [| count |];
        n;
        r = 3;
        triangle = Nx_kernel.Lower;
        adjoint = false;
        unit_diagonal = false;
        seed;
        how = Contiguous;
        scale = Plain;
      })
    [
      (D.Any D.Float32, 8, 65, 1);
      (D.Any D.Complex64, 12, 77, 2);
      (D.Any D.Float32, 40, 128, 3);
    ]

(* A matrix in a batch that fills the GPU, of 16 or 32 lanes a matrix and
   several matrices a group, has the bits it has alone: the first, the middle
   and the last two, of which, at 16 lanes and an odd batch, the last shares its
   SIMD-group with no matrix of the batch. *)
let law_wide (b : Support.backend) c =
  let count = c.batch.(0) in
  let ks = List.sort_uniq compare [ 0; count / 2; count - 2; count - 1 ] in
  let gs = ref [] in
  let placed ts = Option.iter (fun p -> gs := p :: !gs) (placement ts) in
  List.iter
    (fun (msg, r, ops) -> batch_alone b ~msg r ops ks placed)
    (held_calls c);
  let at g = List.exists (fun (g', _) -> g' = g) !gs in
  cover "16 lanes a matrix" (at 16);
  cover "32 lanes a matrix" (at 32);
  cover "16 lanes past the batch's last" (at 16 && count land 1 = 1);
  (* The last unit's matrices past the batch fill a SIMD-group of their own:
     past the batch's last's partner, two more. *)
  let empty (g, mats) =
    let rem = count mod mats in
    g = 16 && rem <> 0 && rem + (rem land 1) + 2 <= mats
  in
  cover "a SIMD-group of 16 lanes wholly past the batch" (List.exists empty !gs)

(* Tall LU panels: 520 to 1100 rows of at most 8 columns, a matrix's rows over
   several groups at batch 1, and its tile in device memory at batch 16. *)
let tall =
  let open Gen in
  with_pp pp_case
    (let* dt = of_list ~pp:pp_dtype [ D.Any D.Float32; D.Any D.Complex64 ] in
     let* n = int_range 520 1100 in
     let* r = int_range 1 8 in
     let* count = of_list ~pp:Format.pp_print_int [ 1; 16 ] in
     let* s = int_range 0 300_000 in
     let+ scale = of_list ~pp:pp_scale [ Plain; Graded; Subnormal ] in
     {
       (fixed dt n r) with
       batch = (if count = 1 then [||] else [| count |]);
       (* lu_operand's columns are r where the seed is 1 mod 3. *)
       seed = (3 * s) + 1;
       scale;
     })

(* A tall panel's LU meets its bound, has the unblocked algorithm's bits, and a
   matrix of the batch has the bits it has alone. *)
let law_tall (b : Support.backend) c =
  let paths = ref [] in
  let placed ts =
    match launch_of K.[ La_lu_panel_f32; La_lu_panel_c64 ] ts with
    | Some t ->
        paths := (word t K.la_panel_p, word t K.la_panel_in_lds) :: !paths
    | None -> ()
  in
  law_lu bounds b c;
  let x = snd (lu_operand c) in
  if c.batch = [||] then begin
    let _, ts = Plan.traced (fun () -> lu_unblocked b c) in
    placed ts
  end
  else batch_alone b ~msg:"lu" Lu [| x |] [ 0; c.batch.(0) - 1 ] placed;
  cover "rows over several groups" (List.exists (fun (p, _) -> p > 1) !paths);
  cover "a tile in device memory"
    (List.exists (fun (p, l) -> p = 1 && l = 0) !paths)

(* A batch of 16 complex64 panels of 1100 rows and 8 columns, whose tiles pass a
   group's memory, so that each run reaches law_tall's tile in device memory:
   drawn, about one case in seven. *)
let tall_examples =
  [ { (fixed ~seed:16 (D.Any D.Complex64) 1100 8) with batch = [| 16 |] } ]

(* Tall LUs past the panel: 520 to 1100 rows of 65 to 80 columns, 33 to 40 at
   complex64, a batch of 16. *)
let tall_past =
  let open Gen in
  with_pp pp_case
    (let* dt = of_list ~pp:pp_dtype [ D.Any D.Float32; D.Any D.Complex64 ] in
     let* n = int_range 520 1100 in
     let* r = if is_complex dt then int_range 33 40 else int_range 65 80 in
     let* s = int_range 0 300_000 in
     let+ scale = of_list ~pp:pp_scale [ Plain; Graded; Subnormal ] in
     {
       (fixed dt n r) with
       batch = [| 16 |];
       (* lu_operand's columns are r where the seed is 1 mod 3. *)
       seed = (3 * s) + 1;
       scale;
     })

(* A tall LU past the panel has the unblocked algorithm's bits: its first matrix
   alone, whose first panel's rows lie over several groups, and each of the
   batch's first and last as alone, whose first panel's tiles lie in device
   memory. *)
let law_tall_past (b : Support.backend) c =
  let cols, x = lu_operand c in
  let panel ts =
    Option.map
      (fun t -> (word t K.la_panel_p, word t K.la_panel_in_lds))
      (launch_of K.[ La_lu_panel_f32; La_lu_panel_c64 ] ts)
  in
  let x0 = alone 0 x in
  let y, ts = Plan.traced (fun () -> run b Lu [| x0 |]) in
  cover "past the panel over several groups"
    (match panel ts with Some (p, _) -> p > 1 | None -> false);
  (match y with
  | None -> ()
  | Some y ->
      let lu, piv, perm = lu_ref ~underflow:b.underflow ~m:c.n ~n:cols x0 in
      same "lu" (of_parts c.dt [| c.n; cols |] lu) y.(0);
      equal ~msg:"lu's pivots" (array int) piv (int64s y.(1));
      equal ~msg:"lu's perm" (array int) perm (int64s y.(2)));
  let spilled = ref false in
  let placed ts =
    match panel ts with Some (1, 0) -> spilled := true | Some _ | None -> ()
  in
  batch_alone b ~msg:"lu" Lu [| x |] [ 0; 15 ] placed;
  cover "past the panel, its tile in device memory" !spilled

(* LUs whose panels span groups *)

(* The columns of an LU's leaf where its panel's rows span groups. *)
let narrow = 8

(* LUs of one matrix of 513 to 1100 rows, more than a group's panel takes, so
   that their first panels' rows span groups. [spans] draws the rest. *)
let spanning =
  let open Gen in
  with_pp pp_case
    (let* dt = of_list ~pp:pp_dtype [ D.Any D.Float32; D.Any D.Complex64 ] in
     let* n = int_range 513 1100 in
     let* r = int_range 0 10_000 in
     let* s = int_range 0 300_000 in
     let+ scale = of_list ~pp:pp_scale [ Plain; Graded; Subnormal ] in
     { (fixed dt n r) with seed = 3 * s; scale })

(* Three LUs from [c], lu_operand's columns being r where the seed is 1 mod 3
   and n + r where it is 2: 600 to 1100 rows and 18 to 71 columns, several
   narrow leaves and the last of 2 to 7; 513 to 560 rows and 57 to 96 columns,
   whose last panels' rows fall to 512, which one group holds; and 513 to 530
   rows and 1 to 24 columns past them, which a solve takes. *)
let spans c =
  let rows = 513 + (c.n mod 48) in
  [
    {
      c with
      n = 600 + (c.n mod 501);
      r = (8 * (2 + (c.r mod 7))) + 2 + (c.r mod 6);
      seed = c.seed + 1;
    };
    { c with n = rows; r = 57 + (c.r mod 40); seed = c.seed + 1 };
    { c with n = 513 + (c.n mod 18); r = 1 + (c.r mod 24); seed = c.seed + 2 };
  ]

(* The width and groups of each LU panel launch in [ts]. *)
let lu_panels ts =
  List.filter_map
    (fun (t : Plan.traced) ->
      if List.mem t.kernel K.[ La_lu_panel_f32; La_lu_panel_c64 ] then
        Some (word t K.la_panel_c0, word t K.la_panel_w, word t K.la_panel_p)
      else None)
    ts

(* [x]'s matrix with rows copied, each from a row 129, 256 or 385 rows above it,
   every other one negated: a row and its copy keep equal magnitudes until one
   of them is a step's pivot, where they tie, and the other then becomes zero.
   The copies and their sources, a list. *)
let with_copies c cols x =
  let w = if is_complex c.dt then 2 else 1 in
  let v = bits x in
  let pairs =
    List.filter_map
      (fun i ->
        let src = 3 * i and dst = (3 * i) + [| 129; 256; 385 |].(i mod 3) in
        if dst < c.n then Some (src, dst) else None)
      (List.init (c.n / 6) Fun.id)
  in
  List.iteri
    (fun i (src, dst) ->
      let sign = if i land 1 = 0 then 1. else -1. in
      for e = 0 to (w * cols) - 1 do
        v.((dst * w * cols) + e) <- sign *. v.((src * w * cols) + e)
      done)
    pairs;
  (pairs, of_parts c.dt [| c.n; cols |] v)

(* An LU whose panels span groups has the unblocked algorithm's bits and pivots,
   whatever its leaves. *)
let law_spanning (b : Support.backend) c =
  let several = ref false and fewer = ref false in
  let then_one = ref false and wide = ref false in
  List.iter
    (fun c ->
      let cols, x = lu_operand c in
      let y, ts = Plan.traced (fun () -> run b Lu [| x |]) in
      let panels = lu_panels ts in
      let narrow_at =
        List.filter (fun (_, w, p) -> p > 1 && w <= narrow) panels
      in
      let starts =
        List.sort_uniq compare (List.map (fun (c0, _, _) -> c0) narrow_at)
      in
      if List.length starts >= 2 then several := true;
      if List.exists (fun (_, w, _) -> w > 1 && w < narrow) narrow_at then
        fewer := true;
      if narrow_at <> [] && List.exists (fun (_, _, p) -> p = 1) panels then
        then_one := true;
      if cols > c.n && narrow_at <> [] then wide := true;
      match y with
      | None -> fail "declined"
      | Some y ->
          let lu, piv, perm = lu_ref ~underflow:b.underflow ~m:c.n ~n:cols x in
          same "lu" (of_parts c.dt [| c.n; cols |] lu) y.(0);
          equal ~msg:"lu's pivots" (array int) piv (int64s y.(1));
          equal ~msg:"lu's perm" (array int) perm (int64s y.(2)))
    (spans c);
  cover "several narrow leaves" !several;
  cover "a narrow leaf of fewer columns" !fewer;
  cover "narrow leaves, then a panel one group a matrix holds" !then_one;
  cover "a wide operand's columns past its rows, over narrow leaves" !wide

(* An LU whose rows tie, a row and its copy in different groups, picks the
   lesser position, as the unblocked algorithm's first largest candidate, at
   every leaf. *)
let law_ties (b : Support.backend) c =
  let c = List.hd (spans c) in
  let cols, x = lu_operand c in
  let pairs, x = with_copies c cols x in
  match run b Lu [| x |] with
  | None -> fail "declined"
  | Some y ->
      let lu, piv, perm = lu_ref ~underflow:b.underflow ~m:c.n ~n:cols x in
      same "lu" (of_parts c.dt [| c.n; cols |] lu) y.(0);
      equal ~msg:"lu's pivots" (array int) piv (int64s y.(1));
      equal ~msg:"lu's perm" (array int) perm (int64s y.(2));
      (* Step j's pivot ties where its row's twin is no earlier pivot. *)
      let perm = int64s y.(2) in
      let at = Array.make c.n 0 in
      Array.iteri (fun pos row -> at.(row) <- pos) perm;
      let twin = Hashtbl.create 64 in
      List.iter
        (fun (s, d) ->
          Hashtbl.replace twin s d;
          Hashtbl.replace twin d s)
        pairs;
      let tied j =
        match Hashtbl.find_opt twin perm.(j) with
        | Some t -> at.(t) > j
        | None -> false
      in
      let k = min c.n cols in
      let steps = List.init k Fun.id in
      cover "a tie in the first leaf"
        (List.exists (fun j -> j < narrow && tied j) steps);
      cover "a tie past the first leaf"
        (List.exists (fun j -> j >= narrow && tied j) steps)

(* An LU of 612 rows and 18 columns whose rows tie in the first leaf and past
   it: drawn, the first leaf ties in about 5 cases of 8, which 6 cases all miss
   about one run in 400. *)
let ties_examples = [ { (fixed (D.Any D.Float32) 513 0) with seed = 6 } ]

(* QRs that one launch holds: reduced of up to 1024 rows, and complete of up to
   160, which reach each placement there and whose check costs m³ (1024 rows
   took 40 s a run). *)
let deep =
  let open Gen in
  with_pp pp_case
    (let* dt = of_list ~pp:pp_dtype [ D.Any D.Float32; D.Any D.Complex64 ] in
     let* complete = bool in
     let* n = int_range 1 (if complete then 160 else 1024) in
     let* r = frequency [ (1, int_range 1 32); (1, int_range 24 32) ] in
     let* s = int_range 0 300_000 in
     let* count = of_list ~pp:Format.pp_print_int [ 1; 2 ] in
     let+ scale = of_list ~pp:pp_scale [ Plain; Graded; Subnormal ] in
     {
       (fixed dt n r) with
       batch = (if count = 1 then [||] else [| count |]);
       (* lu_operand's columns are r where the seed is 1 mod 3, the factors
          complete where it is odd. *)
       seed = ((6 * s) + if complete then 1 else 4);
       scale;
     })

(* A QR's factors meet their bound wherever its partials and tile lie. *)
let law_deep (b : Support.backend) c =
  assume (qr_fits c);
  let (), ts = Plan.traced (fun () -> law_qr b c) in
  match launch_of K.[ La_qr_panel_f32; La_qr_panel_c64 ] ts with
  | None -> ()
  | Some t ->
      let parts = word t K.la_qr_parts_in_lds
      and tile = word t K.la_qr_in_lds in
      cover "partials and tile in the group's memory" (parts = 1 && tile = 1);
      cover "the tile in device memory" (parts = 1 && tile = 0);
      cover "partials and tile in device memory" (parts = 0);
      cover "a complete QR's tile in device memory"
        (factors c = Nx_kernel.Complete && tile = 0)

(* QRs that reach each placement in device memory, so that every run does: a
   complete float32 QR of 160 x 32, its partials and tile there, and a reduced
   one of 1024 x 32, its tile alone. Drawn, a complete QR's tile lies there
   about one time in six. *)
let deep_examples =
  [
    fixed ~seed:7 (D.Any D.Float32) 160 32;
    fixed ~seed:4 (D.Any D.Float32) 1024 32;
  ]

(* Budgets *)

module F = Nx_metal.Factor

(* Runs [f] with launches of at most [t] seconds. *)
let bounded t f =
  Fun.protect
    ~finally:(fun () -> Plan.bound_budget Plan.budget)
    (fun () ->
      Plan.bound_budget t;
      f ())

(* [r]'s plan of [ops], host arrays, into host arrays of the shapes
   Nx_kernel.Shape gives, with every kernel's groups of its full size, as Metal
   grants them on the M1 Max. It raises [Invalid_argument] where one group of a
   step passes the budget. *)
let plan_of r ops =
  let module Shape = Nx_kernel.Shape in
  let ok = Result.get_ok in
  let shape (A.Any x) = Iarray.to_array (L.shape (A.layout x)) in
  let host (D.Any dt) s = A.Any (A.create Rig.host dt (Iarray.of_array s)) in
  let (A.Any a) = ops.(0) in
  let dt = D.Any (A.dtype a) and i64 = D.Any D.Int64 in
  let facts dsts =
    Nx_metal.Operands.v ~dsts:(Iarray.of_array dsts)
      (Iarray.of_array (Array.map (fun (A.Any x) -> Nx_kernel.Plain x) ops))
  in
  let holds = K.threads in
  match r with
  | Cholesky t ->
      F.cholesky ~holds t
        (facts [| host dt (ok (Shape.cholesky (shape ops.(0)))) |])
  | Lu ->
      let sl, sp, sq = ok (Shape.lu (shape ops.(0))) in
      F.lu ~holds (facts [| host dt sl; host i64 sp; host i64 sq |])
  | Qr f ->
      let sq, sr = ok (Shape.qr f (shape ops.(0))) in
      F.qr ~holds (facts [| host dt sq; host dt sr |])
  | Solve { triangle; adjoint; unit_diagonal } ->
      let s = ok (Shape.solve_triangular (shape ops.(0)) (shape ops.(1))) in
      F.solve_triangular ~holds triangle ~adjoint ~unit_diagonal
        (facts [| host dt s |])
  | Eigh _ | Svd _ | Eig _ -> invalid_arg "plan_of"

let launches r ops =
  match plan_of r ops with
  | Ok (Some p) -> p.launches
  | Ok None | Error _ -> [||]

(* The least budget, within 1%, at which [r] on [ops] plans. *)
let least_budget r ops =
  let fits t =
    bounded t (fun () ->
        match plan_of r ops with Ok _ -> true | Error _ -> false)
  in
  let rec go lo hi =
    if hi /. lo < 1.01 then hi
    else
      let mid = sqrt (lo *. hi) in
      if fits mid then go lo mid else go mid hi
  in
  go 1e-9 Plan.budget

(* Batches of 2 to 64 matrices of at most a leaf's order. *)
let several =
  Gen.map
    (fun (c, k) -> { c with batch = [| k |] })
    (Gen.pair wide (Gen.int_range 2 64))

(* At the least budget a call plans at, its steps' units split into more
   launches: the bits are the unbounded call's. *)
let law_split (b : Support.backend) c =
  let split = ref false in
  List.iter
    (fun (name, r, ops) ->
      let t = least_budget r ops in
      let n = Array.length (launches r ops) in
      if bounded t (fun () -> Array.length (launches r ops)) > n then
        split := true;
      match (run b r ops, bounded t (fun () -> run b r ops)) with
      | Some y, Some z ->
          Array.iteri
            (fun i (A.Any zi as z) ->
              let msg = strf "%s, result %d" name i in
              match A.dtype zi with
              | D.Int64 -> equal ~msg (array int) (int64s y.(i)) (int64s z)
              | _ -> equal ~msg same_bits (bits y.(i)) (bits z))
            z
      | _ -> ())
    (held_calls c);
  cover "a step over several launches" !split

(* plan.mli: no bit follows a part's bounds. Under lowered bounds, the calls of
   [c] run in several parts with the bits of a call of one: the split calls, at
   the least budget, as law_split runs them. *)
let law_split_parts b c =
  let parts (_, r, ops) =
    match bounded (least_budget r ops) (fun () -> plan_of r ops) with
    | Ok (Some (p : Plan.t)) -> Array.length p.parts
    | Ok None | Error _ -> 0
  in
  cover "a call of several parts"
    (List.exists (fun call -> parts call > 1) (held_calls c));
  law_split b c

(* The one launch's placements, planned without the GPU: a tile in the group's
   memory only with its partials, and each in device memory at an address the
   launch's refs bind. *)
let law_placed c =
  assume (qr_fits c);
  let ops = [| snd (lu_operand c) |] in
  (* The words among [ws] that [l]'s refs bind. *)
  let bound (l : Plan.launch) ws =
    List.filter
      (fun i -> Array.exists (fun (r : Plan.ref) -> r.word = i) l.refs)
      ws
  in
  Array.iter
    (fun (l : Plan.launch) ->
      if List.mem l.kernel K.[ La_qr_panel_f32; La_qr_panel_c64 ] then begin
        let w i = Int64.to_int (Bytes.get_int64_le l.words (8 * i)) in
        let parts = w K.la_qr_parts_in_lds and tile = w K.la_qr_in_lds in
        cover "partials and tile in the group's memory" (parts = 1 && tile = 1);
        cover "the tile in device memory" (parts = 1 && tile = 0);
        cover "partials and tile in device memory" (parts = 0);
        if tile = 1 then
          equal ~msg:"a tile in the group's memory has its partials there" int 1
            parts;
        let want =
          (if parts = 0 then [ K.la_qr_parts ] else [])
          @ if tile = 0 then [ K.la_qr_spill ] else []
        in
        equal ~msg:"the device memory of partials and tile is bound" (list int)
          want
          (bound l [ K.la_qr_parts; K.la_qr_spill ])
      end)
    (launches (Qr (factors c)) ops)

(* Every launch's estimate is within the budget, at the budget and at budgets
   down to the least a call plans at. Needs no GPU. *)
let law_budget ?only c =
  List.iter
    (fun (name, r, ops) ->
      let least = least_budget r ops in
      List.iter
        (fun t ->
          bounded t (fun () ->
              Array.iteri
                (fun i (l : Plan.launch) ->
                  at_most
                    ~msg:(strf "%s at %g s, launch %d" name t i)
                    float_exact ~than:t
                    (Plan.seconds ~threads:(K.threads l.kernel) ~groups:l.groups
                       ~resident:l.resident l.cost))
                (launches r ops)))
        [ Plan.budget; sqrt (least *. Plan.budget); least ])
    (held_calls ?only c)

(* Panels past their old limits

   At a budget of 30 us, which keeps a product's least launch (3.9 us) but not
   the leaf of a solve (124 us), an LU panel's groups take fewer rows and its
   steps split over launches, and a QR panel's sums' old limit falls to tens of
   thousands of rows. Tall panels past them: LUs of at most a narrow leaf's
   columns, since past them a call holds a solve's leaf, up to 8,400,000 rows;
   and a QR of more than 128 groups of 16 blocks, whose meetings' sums take a
   stage of their own, resumed over launches. *)
let cut_budget = 30e-6

let cuts =
  [
    ("lu", D.Any D.Float32, 34_000, narrow);
    ("lu", D.Any D.Float32, 33_000, 1);
    ("lu", D.Any D.Complex64, 34_000, narrow);
    ("lu", D.Any D.Float32, 8_400_000, 1);
    ("qr", D.Any D.Float32, 530_000, 4);
    ("qr", D.Any D.Complex64, 530_000, 2);
  ]

(* The distinct values of word [i] of [ks]'s launches in [ts]. *)
let words_of ks i ts =
  List.sort_uniq compare
    (List.filter_map
       (fun (t : Plan.traced) ->
         if List.mem t.kernel ks then Some (word t i) else None)
       ts)

(* Rows a group that tests force (Factor.bound_lu): one to three, so that
   hundreds of rows span more than 256 groups and each step searches a level,
   and 384 and 1152, so that a group's threads hold several rows each. *)
let forced =
  let open Gen in
  with_pp
    (fun ppf (r, c) -> Format.fprintf ppf "%d rows a group, %a" r pp_case c)
    (let* r = of_list ~pp:Format.pp_print_int [ 1; 2; 3; 384; 1152 ] in
     let* dt = of_list ~pp:pp_dtype [ D.Any D.Float32; D.Any D.Complex64 ] in
     let* n =
       if r <= 3 then int_range (if r = 3 then 800 else 600) 1100
       else int_range 2000 5000
     in
     let* cols = int_range 9 40 in
     let* s = int_range 0 300_000 in
     let+ scale = of_list ~pp:pp_scale [ Plain; Graded; Subnormal ] in
     (* lu_operand's columns are [cols] where the seed is 1 mod 3. *)
     (r, { (fixed dt n cols) with seed = (3 * s) + 1; scale }))

(* One case of each rows a group, so that every run reaches each cover. *)
let forced_examples =
  List.mapi
    (fun i (r, dt, n) ->
      (r, { (fixed dt n 17) with seed = (3 * i) + 1; scale = Plain }))
    [
      (1, D.Any D.Float32, 600);
      (2, D.Any D.Complex64, 640);
      (3, D.Any D.Float32, 800);
      (384, D.Any D.Complex64, 2000);
      (1152, D.Any D.Float32, 4000);
    ]

(* Runs [f] with LU panels over several groups of at most [r] rows. *)
let with_rows r f =
  Fun.protect
    ~finally:(fun () -> F.bound_lu ~rows:None)
    (fun () ->
      F.bound_lu ~rows:(Some r);
      f ())

(* [x]'s matrix with row 256 the negation of row 0 and both rows' first elements
   of the column's largest magnitude: step 0's pivot ties between two rows that
   one thread holds where a group takes more than 256 rows. *)
let with_twin c cols x =
  let w = if is_complex c.dt then 2 else 1 in
  let v = bits x in
  let at i = i * w * cols in
  let big = ref 0. in
  for i = 0 to c.n - 1 do
    big := Float.max !big (Float.abs v.(at i))
  done;
  v.(at 0) <- !big;
  if w = 2 then v.(at 0 + 1) <- 0.;
  for e = 0 to (w * cols) - 1 do
    v.(at 256 + e) <- -.v.(at 0 + e)
  done;
  of_parts c.dt [| c.n; cols |] v

(* An LU whose panels' groups take any number of rows, its rows tied within a
   group's thread and across groups, has the unblocked algorithm's bits and
   pivots. *)
let law_rows (b : Support.backend) (r, c) =
  let cols, x = lu_operand c in
  let _, x = with_copies c cols x in
  let x = with_twin c cols x in
  let y, ts =
    with_rows r (fun () -> Plan.traced (fun () -> run b Lu [| x |]))
  in
  let spread =
    List.filter_map
      (fun (t : Plan.traced) ->
        if
          List.mem t.kernel K.[ La_lu_panel_f32; La_lu_panel_c64 ]
          && word t K.la_panel_p > 1
        then
          Some
            ( word t K.la_panel_c0,
              word t K.la_panel_m - word t K.la_panel_c0,
              word t K.la_panel_per,
              word t K.la_panel_level )
        else None)
      ts
  in
  let threads = K.la_threads in
  cover "a level a step" (List.exists (fun (_, _, _, l) -> l >= 1) spread);
  cover "several rows a thread"
    (List.exists (fun (_, _, per, _) -> per > threads) spread);
  cover "a partial last group"
    (List.exists (fun (_, mp, per, _) -> mp mod per <> 0) spread);
  cover "a tie within a thread"
    (List.exists (fun (c0, _, per, _) -> c0 = 0 && per > threads) spread);
  match y with
  | None -> fail "declined"
  | Some y ->
      let lu, piv, perm = lu_ref ~underflow:b.underflow ~m:c.n ~n:cols x in
      same "lu" (of_parts c.dt [| c.n; cols |] lu) y.(0);
      equal ~msg:"lu's pivots" (array int) piv (int64s y.(1));
      equal ~msg:"lu's perm" (array int) perm (int64s y.(2))

(* An LU of 65,537 x 1 at one row a group, whose steps search two levels, has
   the unblocked algorithm's bits. *)
let test_two_levels () =
  let b = metal () in
  b.around (fun () ->
      let c = fixed ~seed:4 (D.Any D.Float32) 65_537 1 in
      let cols, x = lu_operand c in
      let y, ts =
        with_rows 1 (fun () -> Plan.traced (fun () -> run b Lu [| x |]))
      in
      equal ~msg:"levels" (list int) [ 0; 1; 2 ]
        (words_of K.[ La_lu_panel_f32; La_lu_panel_c64 ] K.la_panel_level ts);
      match y with
      | None -> fail "declined"
      | Some y ->
          let lu, piv, perm = lu_ref ~underflow:b.underflow ~m:c.n ~n:cols x in
          same "lu" (of_parts c.dt [| c.n; cols |] lu) y.(0);
          equal ~msg:"lu's pivots" (array int) piv (int64s y.(1));
          equal ~msg:"lu's perm" (array int) perm (int64s y.(2)))

(* The bound on a launch's units past which it divides them in 64 bits
   (Factor.bound_divide), against a batch whose panels take groups of 3 rows, so
   that a matrix's first panel takes p = ⌈n / 3⌉ groups: [Wide], 1, under which
   every launch divides in 64 bits; [Across], p (batch - 1) + 1, past p, which
   the first panel's launch crosses; [Narrow], p batch, past its last unit,
   under which every launch divides in 32 bits. *)
type side = Wide | Across | Narrow

let pp_side ppf s =
  Format.pp_print_string ppf
    (match s with Wide -> "wide" | Across -> "across" | Narrow -> "narrow")

let divide_rows = 3

let below side c =
  let p = (c.n + divide_rows - 1) / divide_rows and batch = c.batch.(0) in
  match side with
  | Wide -> 1
  | Across -> (p * (batch - 1)) + 1
  | Narrow -> p * batch

let bounded_divide =
  let open Gen in
  with_pp
    (fun ppf (side, c) -> Format.fprintf ppf "%a, %a" pp_side side pp_case c)
    (let* side = of_list ~pp:pp_side [ Wide; Across; Narrow ] in
     let* dt = of_list ~pp:pp_dtype [ D.Any D.Float32; D.Any D.Complex64 ] in
     let* n = int_range 600 1100 in
     let* cols = int_range 9 40 in
     let* batch = int_range 3 5 in
     let* s = int_range 0 300_000 in
     let+ scale = of_list ~pp:pp_scale [ Plain; Graded; Subnormal ] in
     (* lu_operand's columns are [cols] where the seed is 1 mod 3. *)
     ( side,
       { (fixed dt n cols) with seed = (3 * s) + 1; scale; batch = [| batch |] }
     ))

(* One case of each side, so that every run reaches each cover. *)
let divide_examples =
  List.mapi
    (fun i side ->
      ( side,
        {
          (fixed (D.Any D.Float32) 700 17) with
          seed = (3 * i) + 1;
          scale = Plain;
          batch = [| 3 |];
        } ))
    [ Wide; Across; Narrow ]

(* Runs [f] with launches dividing in 64 bits from [below] units. *)
let with_below below f =
  Fun.protect
    ~finally:(fun () -> F.bound_divide ~below:(1 lsl 32))
    (fun () ->
      F.bound_divide ~below;
      f ())

(* The first and last units, the divisor and whether it divides them in 64 bits
   of each traced launch whose kernel divides its units: an LU panel's over
   several groups, a step's (by its groups a matrix) or a level's (by its runs),
   the gather's and the solve's. *)
let divisions ts =
  List.filter_map
    (fun (t : Plan.traced) ->
      let span ~first ~count ~wide ~per divisor =
        let g, _, _ = t.groups in
        let f = word t first in
        Some
          ( f,
            Int.min (word t count) (f + (g * per)) - 1,
            divisor,
            word t wide = 1 )
      in
      match t.kernel with
      | (K.La_lu_panel_f32 | La_lu_panel_c64) when word t K.la_panel_p > 1 ->
          let divisor =
            if word t K.la_panel_level = 0 then word t K.la_panel_p
            else (word t K.la_panel_n_in + K.la_threads - 1) / K.la_threads
          in
          span ~first:K.la_panel_first ~count:K.la_panel_count
            ~wide:K.la_panel_wide ~per:1 divisor
      | La_gather ->
          span ~first:K.la_gather_first ~count:K.la_gather_count
            ~wide:K.la_gather_wide ~per:1 (word t K.la_gather_per)
      | La_trsm_f32 | La_trsm_c64 ->
          span ~first:K.la_trsm_first ~count:K.la_trsm_count
            ~wide:K.la_trsm_wide ~per:K.la_trsm_threads (word t K.la_trsm_r)
      | _ -> None)
    ts

(* An LU's launches divide their units in 64 bits exactly where a launch's last
   unit or its divisor reaches the bound, and the LU has the bits it has under
   the bound of 2^32, whichever division each launch takes. *)
let law_divide (b : Support.backend) (side, c) =
  let _, x = lu_operand c in
  let below = below side c in
  let lu () = with_rows divide_rows (fun () -> run b Lu [| x |]) in
  let want = lu () in
  let y, ts = with_below below (fun () -> Plan.traced lu) in
  let ds = divisions ts in
  let wide (_, _, _, w) = w in
  cover "every launch in 64 bits" (ds <> [] && List.for_all wide ds);
  cover "a launch across the bound"
    (List.exists
       (fun (first, last, _, w) -> w && first < below && last >= below)
       ds);
  cover "every launch in 32 bits"
    (ds <> [] && List.for_all (fun d -> not (wide d)) ds);
  let mistaken =
    List.filter_map
      (fun (first, last, divisor, w) ->
        if w = (last >= below || divisor >= below) then None
        else
          Some
            (strf "units %d to %d by %d in %d bits" first last divisor
               (if w then 64 else 32)))
      ds
  in
  equal
    ~msg:(strf "launches whose division misses the bound %d" below)
    (list string) [] mistaken;
  match (want, y) with
  | None, _ | _, None -> fail "declined"
  | Some want, Some y ->
      same "lu" want.(0) y.(0);
      equal ~msg:"lu's pivots" (array int) (int64s want.(1)) (int64s y.(1));
      equal ~msg:"lu's perm" (array int) (int64s want.(2)) (int64s y.(2))

(* The rows and batches at the edges of the panels' spread over groups: 512 and
   513 rows, batches of 15 and 16. *)
let edges = [ (512, 1); (513, 1); (512, 15); (513, 15); (513, 16) ]

(* An LU at an edge of the spread has the unblocked algorithm's bits, and each
   matrix of a batch the bits it has alone. *)
let test_edge (m, count) =
  let b = metal () in
  b.around (fun () ->
      let c = fixed ~seed:4 (D.Any D.Float32) m 20 in
      let c = if count = 1 then c else { c with batch = [| count |] } in
      let cols, x = lu_operand c in
      if count = 1 then (
        match run b Lu [| x |] with
        | None -> fail "declined"
        | Some y ->
            let lu, piv, perm =
              lu_ref ~underflow:b.underflow ~m:c.n ~n:cols x
            in
            same "lu" (of_parts c.dt [| c.n; cols |] lu) y.(0);
            equal ~msg:"lu's pivots" (array int) piv (int64s y.(1));
            equal ~msg:"lu's perm" (array int) perm (int64s y.(2)))
      else batch_alone b ~msg:"lu" Lu [| x |] [ 0; count - 1 ] (fun _ -> ()))

(* A tall LU has the unblocked algorithm's bits at the budget and at 30 us. *)
let test_lu_cut (b : Support.backend) c =
  let cols, x = lu_operand c in
  let lu, piv, perm = lu_ref ~underflow:b.underflow ~m:c.n ~n:cols x in
  List.iter
    (fun t ->
      let y = bounded t (fun () -> run b Lu [| x |]) in
      let msg = strf "at %g s" t in
      match y with
      | None -> failf "%s: declined" msg
      | Some y ->
          same (msg ^ ", lu") (of_parts c.dt [| c.n; cols |] lu) y.(0);
          equal ~msg:(msg ^ ", pivots") (array int) piv (int64s y.(1));
          equal ~msg:(msg ^ ", perm") (array int) perm (int64s y.(2)))
    [ Plan.budget; cut_budget ]

(* A QR past its sums' old limit meets its bound, and has the same bits at 30
   us, where each sum stage resumes its totals over several launches. *)
let test_qr_cut (b : Support.backend) c =
  let f = factors c in
  let cols, x = lu_operand c in
  let qr = K.[ La_qr_panel_f32; La_qr_panel_c64 ] in
  let y, ts = Plan.traced (fun () -> run b (Qr f) [| x |]) in
  let stages =
    K.[ la_qr_step_a; la_qr_step_b; la_qr_step_g ]
    @ K.[ la_qr_sum_b; la_qr_sum_a; la_qr_sum_g ]
  in
  equal ~msg:"stages" (list int) (List.sort compare stages)
    (words_of qr K.la_qr_stage ts);
  let z, cut =
    bounded cut_budget (fun () -> Plan.traced (fun () -> run b (Qr f) [| x |]))
  in
  let summing (t : Plan.traced) =
    List.mem t.kernel qr && word t K.la_qr_stage >= K.la_qr_sum_b
  in
  at_least ~msg:"a sum stage's first superblocks at 30 us" int ~than:2
    (List.length
       (List.sort_uniq compare
          (List.filter_map
             (fun t -> if summing t then Some (word t K.la_qr_from) else None)
             cut)));
  match (y, z) with
  | Some y, Some z ->
      let d = down c and m = c.n in
      check_qr ~underflow:b.underflow c.dt ~e:d ~count:1 ~m ~n:cols ~qc:cols
        (up d (values x))
        (values y.(0))
        (up d (values y.(1)));
      Array.iteri
        (fun i z ->
          equal
            ~msg:(strf "at 30 us, result %d" i)
            same_bits
            (bits y.(i))
            (bits z))
        z
  | _ -> fail "declined"

(* A QR of 530,000 rows, 2071 blocks, sums its meetings in sum stages alone,
   past 128 groups, and by its group's tree at a batch of 16, one group a
   matrix: a total has the tree's bits wherever its superblocks are added. *)
let test_summed () =
  let b = metal () in
  b.around (fun () ->
      let c =
        { (fixed ~seed:4 (D.Any D.Float32) 530_000 1) with batch = [| 16 |] }
      in
      let f = factors c in
      let _, x = lu_operand c in
      let qr = K.[ La_qr_panel_f32; La_qr_panel_c64 ] in
      let y, ts = Plan.traced (fun () -> run b (Qr f) [| x |]) in
      equal ~msg:"summed, a batch of 16" (list int) [ 0 ]
        (words_of qr K.la_qr_summed ts);
      match y with
      | None -> fail "a batch of 16 declined"
      | Some y ->
          List.iter
            (fun k ->
              let z, ts =
                Plan.traced (fun () -> run b (Qr f) [| alone k x |])
              in
              equal
                ~msg:(strf "summed, matrix %d alone" k)
                (list int) [ 1 ]
                (words_of qr K.la_qr_summed ts);
              match z with
              | None -> failf "matrix %d alone declined" k
              | Some z -> same_matrix ~msg:"qr, summed" k y z)
            [ 0; 15 ])

let test_cut (name, dt, m, n) =
  let b = metal () in
  (* lu_operand's columns are [n] where the seed is 1 mod 3, and QR's factors
     reduced where it is even. *)
  let c = { (fixed ~seed:4 dt m n) with batch = [| 1 |] } in
  b.around (fun () -> if name = "lu" then test_lu_cut b c else test_qr_cut b c)

(* A call one of whose groups passes the budget declines: no launch of its plan
   would keep within it. Needs no GPU. *)
let law_over_budget c =
  List.iter
    (fun (name, r, ops) ->
      let outcome () =
        match plan_of r ops with
        | Ok None -> "no element"
        | Ok (Some _) -> "planned"
        | Error A.Declined -> "declined"
        | Error _ -> "another answer"
        | exception Invalid_argument m -> "raised " ^ m
      in
      let want =
        match plan_of r ops with Ok None -> "no element" | _ -> "declined"
      in
      equal ~msg:name string want (bounded 1e-9 outcome))
    (held_calls c);
  cover "a solve past the leaf, whose first step is the contraction's copy"
    (c.n > leaf c)

(* QR past one launch *)

(* law_qr on every case, past one launch and of strided operands among them: a
   call planned is computed. *)
let law_qr_any (b : Support.backend) c =
  let x = laid_out c.how (snd (lu_operand c)) in
  let planned = launches (Qr (factors c)) [| x |] <> [||] in
  cover "planned past one launch" (planned && not (qr_fits c));
  cover "planned of an operand not read in place" (planned && not (in_place c));
  law_qr b c

(* A reduced float32 QR of 65 x 65 elements, past one launch, through a
   transposed operand, so that each run reaches law_qr_any's covers: drawn, a
   case plans past one launch about one time in seven. *)
let qr_examples =
  [ { (fixed ~seed:6 (D.Any D.Float32) 65 0) with how = Transposed } ]

(* Tall QRs past one launch *)

(* Reduced float32 QRs of 300 to 1100 rows and 33 to 64 columns, two panels past
   one block of rows each, from which [tall_qrs] draws the rest. *)
let tall_qr =
  let open Gen in
  with_pp pp_case
    (let* n = int_range 300 1100 in
     let* r = int_range 33 64 in
     let* s = int_range 0 300_000 in
     let+ scale = of_list ~pp:pp_scale [ Plain; Graded; Subnormal ] in
     {
       (fixed (D.Any D.Float32) n r) with
       batch = [| 16 |];
       (* lu_operand's columns are r where the seed is 1 mod 3, and the factors
          reduced where it is even. *)
       seed = (6 * s) + 4;
       scale;
     })

(* [c] and three QRs from it, each past one launch and with more than one block
   of rows, so that a matrix alone has panels over several groups. [c] itself
   keeps its panels' partials in the group's memory at a batch of 16 (at most 9
   blocks of rows) and their tile in device memory. A complex64 reduced QR of
   600 to 1100 rows and 33 to 40 columns puts its first panel's partials in
   device memory at a batch of 16, its 32 columns over 3 blocks or more. The
   complete ones have 257 to 300 rows, the fewest past one block, at float32
   with [c]'s columns and at complex64 with 17 to 32. *)
let tall_qrs c =
  let c64 = D.Any D.Complex64 in
  let complete c = { c with n = 257 + (c.n mod 44); seed = c.seed - 3 } in
  (* Each scale in every case: the four QRs take them in turn from [c]'s. *)
  let scales = [| Plain; Graded; Subnormal |] in
  let first =
    if c.scale = Plain then 0 else if c.scale = Graded then 1 else 2
  in
  List.mapi
    (fun i q -> { q with scale = scales.((first + i) mod 3) })
    [
      c;
      { c with dt = c64; n = 600 + (c.n mod 501); r = 33 + (c.r mod 8) };
      complete c;
      { (complete c) with dt = c64; r = 17 + (c.r mod 16) };
    ]

(* A QR panel launch's placement: its groups a matrix, and whether its partials
   and its tile lie in the group's memory. *)
let qr_placements ts =
  List.filter_map
    (fun (t : Plan.traced) ->
      if List.mem t.kernel K.[ La_qr_panel_f32; La_qr_panel_c64 ] then
        Some
          (word t K.la_qr_p, word t K.la_qr_parts_in_lds, word t K.la_qr_in_lds)
      else None)
    ts

(* A tall QR past one launch meets its bound at a batch of 2, whose panels' rows
   lie over several groups, with each matrix's bits as alone; and at a batch of
   16, whose panels are one group a matrix, its first and last matrices' bits
   are as alone. *)
let law_tall_qr (b : Support.backend) c =
  let groups = ref false and complete = ref false in
  let parts_in = ref false and parts_out = ref false in
  List.iter
    (fun c ->
      let f = factors c in
      let c2 = { c with batch = [| 2 |] } in
      let cols, x = lu_operand c2 in
      let y, ts = Plan.traced (fun () -> run b (Qr f) [| x |]) in
      if List.exists (fun (p, _, _) -> p > 1) (qr_placements ts) then begin
        groups := true;
        if f = Nx_kernel.Complete then complete := true
      end;
      (match y with
      | None -> ()
      | Some y ->
          let d = down c2 and m = c2.n in
          let qc = if f = Nx_kernel.Complete then m else min m cols in
          check_qr ~underflow:b.underflow c2.dt ~e:d ~count:2 ~m ~n:cols ~qc
            (up d (values x))
            (values y.(0))
            (up d (values y.(1)));
          List.iter
            (fun k ->
              match run b (Qr f) [| alone k x |] with
              | None -> ()
              | Some z -> same_matrix ~msg:"qr, a batch of 2" k y z)
            [ 0; 1 ]);
      let placed ts =
        List.iter
          (fun (p, parts, tile) ->
            if p = 1 && parts = 1 && tile = 0 then parts_in := true;
            if p = 1 && parts = 0 && tile = 0 then parts_out := true)
          (qr_placements ts)
      in
      batch_alone b ~msg:"qr, a batch of 16" (Qr f)
        [| snd (lu_operand c) |]
        [ 0; 15 ] placed)
    (tall_qrs c);
  List.iter
    (fun (name, scale) ->
      cover (name ^ " operand")
        (List.exists (fun q -> q.scale = scale) (tall_qrs c)))
    [ ("a plain", Plain); ("a graded", Graded); ("a subnormal", Subnormal) ];
  cover "panels over several groups" !groups;
  cover "a complete QR's panels over several groups" !complete;
  cover "one group a matrix, the tile in device memory" !parts_in;
  cover "one group a matrix, partials and tile in device memory" !parts_out

(* Orders past the leaf *)

(* Cases at float32 and complex64 of orders around and past one and two leaves,
   so that each run reaches every split, dtype and scale past them. *)
let past =
  let open Gen in
  with_pp pp_case
    (let* c = case in
     let* dt = of_list ~pp:pp_dtype [ D.Any D.Float32; D.Any D.Complex64 ] in
     let+ n =
       of_list ~pp:Format.pp_print_int [ 0; 1; 2; 17; 33; 63; 65; 97; 129; 140 ]
     in
     { c with dt; n })

(* A subnormal float32 Cholesky of order 65, past one block, whose operand
   flushes to one not definite, so that each run reaches law_cholesky_bits's
   covers of both: drawn, about one case in eight. *)
let cholesky_examples =
  [ { (fixed ~seed:1 (D.Any D.Float32) 65 0) with scale = Subnormal } ]

(* A subnormal float32 LU of 65 rows and 68 columns, past one panel and wider
   than tall, so that each run reaches law_lu_bits's subnormal and wide covers:
   drawn, one case in eight and one in seven. *)
let lu_examples =
  [ { (fixed ~seed:2 (D.Any D.Float32) 65 3) with scale = Subnormal } ]

let complex c = is_complex c.dt
let subnormal c = c.scale = Subnormal

(* Cholesky past the leaf *)

(* A Cholesky of any order and operand has the unblocked algorithm's bits: each
   element takes its updates in increasing step order, whatever blocks, solves
   and products it splits into, and a matrix that is not definite is NaN. *)
let law_cholesky_bits (b : Support.backend) c =
  let c = one c in
  let x = cholesky_operand c in
  let computed = run b (Cholesky Lower) [| laid_out c.how x |] in
  let want = cholesky_ref ~underflow:b.underflow c.dt ~n:c.n (values x) in
  let past = Option.is_some computed && c.n > leaf c in
  cover "computed past one block" past;
  cover "computed at complex64 past one block" (past && complex c);
  cover "computed of a subnormal scale past one block" (past && subnormal c);
  cover "computed not definite past one block" (past && Option.is_none want);
  cover "computed of an operand not read in place"
    (Option.is_some computed && not (in_place c));
  match (computed, want) with
  | None, _ -> ()
  | Some y, Some l -> same "cholesky" (of_values c.dt (square c) l) y.(0)
  | Some y, None ->
      equal ~msg:"a matrix not definite is NaN" int 0
        (numbers c.dt (values y.(0)))

(* LUs past the panel *)

(* An LU of any columns and operand has the unblocked algorithm's bits: each
   element takes its updates in increasing step order, whatever panels, solves
   and products the LU splits into. *)
let law_lu_bits (b : Support.backend) c =
  let c = one c in
  let cols, x = lu_operand c in
  let computed = run b Lu [| laid_out c.how x |] in
  let past = Option.is_some computed && c.n > leaf c in
  cover "computed past one panel" (past && cols > leaf c);
  cover "computed at complex64 past one panel"
    (past && cols > leaf c && complex c);
  cover "computed of a subnormal scale past one panel"
    (past && cols > leaf c && subnormal c);
  cover "computed a wide matrix's columns past its rows" (past && cols > c.n);
  cover "computed of an operand not read in place"
    (Option.is_some computed && not (in_place c));
  match computed with
  | None -> ()
  | Some y ->
      let lu, piv, perm = lu_ref ~underflow:b.underflow ~m:c.n ~n:cols x in
      same "lu" (of_parts c.dt [| c.n; cols |] lu) y.(0);
      equal ~msg:"lu's pivots" (array int) piv (int64s y.(1));
      equal ~msg:"lu's perm" (array int) perm (int64s y.(2))

(* Solves past the leaf *)

(* A solve of any order has the unblocked algorithm's bits: x_i takes its
   updates in increasing step order, whatever leaves and products the solve
   splits into, then its division. *)
let law_solve_bits (b : Support.backend) c =
  let c = one c in
  let ops = solve_operands c in
  let computed =
    run b (solve c) [| laid_out c.how ops.(0); laid_out c.how ops.(1) |]
  in
  let done_ = Option.is_some computed in
  let past = done_ && c.n > leaf c in
  cover "computed past one leaf" past;
  cover "computed at complex64 past one leaf" (past && complex c);
  cover "computed upper or adjoint past one leaf"
    (past && (c.triangle = Upper || c.adjoint));
  cover "computed past two leaves" (done_ && c.n > 2 * leaf c);
  cover "computed of a subnormal scale past one leaf" (past && subnormal c);
  cover "computed of a b not read in place within one leaf"
    (done_ && c.n <= leaf c && c.r > 0 && not (in_place c));
  match computed with
  | None -> ()
  | Some y ->
      let upper = c.triangle = Upper <> c.adjoint in
      let x =
        solve_ref ~underflow:b.underflow c.dt ~n:c.n ~r:c.r ~upper
          ~adjoint:c.adjoint ~unit:c.unit_diagonal
          (values ops.(0))
          (values ops.(1))
      in
      same "solve" (of_values c.dt [| c.n; c.r |] x) y.(0)

let laws () =
  (* [f] on the cases [held] holds. *)
  let on held f x =
    assume (held x);
    let b = metal () in
    b.around (fun () -> f b x)
  in
  group "metal"
    [
      group "each result meets its relation within its bound"
        [
          prop "Cholesky's factor within γ(n + 1)" case
            (on (fun _ -> true) (law_cholesky bounds));
          prop "the Upper factor is the Lower one's conjugate transpose" case
            (on (fun _ -> true) law_upper);
          prop "a triangular solve within γ(n + 2)" case
            (on (fun _ -> true) (law_solve bounds));
          prop "LU's factors within γ(k + 2)" case
            (on (fun _ -> true) (law_lu bounds));
          prop ~examples:qr_examples "QR's factors within γ(k(2m + 10))" case
            (on (fun _ -> true) law_qr_any);
          cases "QR whose partials pass half a group's memory"
            ~id:(fun (dt, m, n, complete) ->
              Format.asprintf "%a %dx%d%s" pp_dtype dt m n
                (if complete then " complete" else ""))
            qr_edges test_qr_edge;
        ];
      group "bits are a function of the values alone"
        [
          prop ~examples:cholesky_examples
            "a Cholesky's of any order are the unblocked algorithm's" past
            (on (fun _ -> true) law_cholesky_bits);
          prop "a solve's of any order are the unblocked algorithm's" past
            (on (fun _ -> true) law_solve_bits);
          prop ~examples:lu_examples
            "an LU's of any columns are the unblocked algorithm's" past
            (on (fun _ -> true) law_lu_bits);
        ];
      group "placement and the batch change no bit"
        [
          prop ~count:25 ~examples:wide_examples
            "a matrix of a batch that fills the GPU has its bits alone" wide
            (on (fun _ -> true) law_wide);
          (* Each case reaches every cover by construction ([tall_qrs]). *)
          prop ~count:2
            "a tall QR past one launch over several groups or in device memory"
            tall_qr
            (on (fun _ -> true) law_tall_qr);
          prop ~count:6
            "a tall LU past the panel over several groups or of a tile in \
             device memory"
            tall_past
            (on (fun _ -> true) law_tall_past);
          prop ~examples:tall_examples
            "a tall LU over several groups or of a tile in device memory" tall
            (on (fun _ -> true) law_tall);
          (* Each case reaches every cover by construction ([spans]). *)
          prop ~count:3 "an LU whose panels span groups, whatever its leaves"
            spanning
            (on (fun _ -> true) law_spanning);
          prop ~count:10 ~examples:forced_examples
            "an LU of any rows a group has its bits, ties included" forced
            (on (fun _ -> true) law_rows);
          test "an LU whose steps search two levels has its bits"
            test_two_levels;
          prop ~count:6 ~examples:divide_examples
            "an LU has its bits whichever division its launches take"
            bounded_divide
            (on (fun _ -> true) law_divide);
          cases "an LU at the edges of the spread over groups has its bits"
            ~id:(fun (m, count) -> Format.asprintf "%d rows, batch %d" m count)
            edges test_edge;
          prop ~count:6 ~examples:ties_examples
            "an LU's ties in different groups, at every leaf" spanning
            (on (fun _ -> true) law_ties);
          prop ~examples:deep_examples
            "a QR of up to 1024 rows wherever its partials and tile lie" deep
            (on (fun _ -> true) law_deep);
        ];
      group "the budget changes no bit"
        [
          (* About nine cases in ten split a step. *)
          prop ~count:30 "a step split by the budget keeps its bits" several
            (on (fun _ -> true) law_split);
          prop ~count:10 "a step split and cut at every launch keeps its bits"
            several
            (on
               (fun _ -> true)
               (fun b c ->
                 Nx_metal_support.parted ~launches:1 (fun () ->
                     law_split_parts b c)));
          prop "every launch is within the budget, without the GPU" several
            law_budget;
          prop "every tall LU's launch is within the budget, without the GPU"
            tall
            (law_budget ~only:(String.equal "lu"));
          prop "every deep QR's launch is within the budget, without the GPU"
            deep
            (law_budget ~only:(String.equal "qr"));
          prop ~examples:deep_examples
            "a QR's tile lies in the group's memory only with its partials" deep
            law_placed;
          prop "a call one of whose groups passes the budget declines" case
            law_over_budget;
          cases "a panel past its old limits computes, with the same bits"
            ~id:(fun (name, dt, m, n) ->
              Format.asprintf "%s %a %dx%d" name pp_dtype dt m n)
            cuts test_cut;
          test "a summed QR has the bits of its group's tree" test_summed;
        ];
      group "an operand whose batch axes do not merge declines"
        [
          cases "each routine"
            ~id:(Format.asprintf "%a" pp_dtype)
            [ D.Any D.Float32; D.Any D.Complex64 ]
            test_declines;
        ];
    ]

let () = exit (Windtrap.run "nx.metal factorisations" [ laws () ])
