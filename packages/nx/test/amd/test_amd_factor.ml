(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.amd's factorisations, as nx_amd.mli states them, on the machine's AMD GPU
   0, skipping without one: each result against its defining relation within
   nx.amd's bound, on the suite's cases, at orders past every blocking and a
   panel's meeting, in LU's tall panel, and at the ends of the exponent range;
   LU, Cholesky and the solves bit for bit the unblocked algorithm's; the same
   bits, the spectral routines' too, under every layout and in every batch;
   the same bits whatever lanes a matrix takes, past the most groups a
   launch has and from operands that admit only reads; the workspace each
   call holds. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module Support = Nx_kernels_support
open Linalg_cases
open Linalg_laws

(* nx_amd.mli's bounds: the unblocked algorithm's. It computes no eigh, svd or
   eig, whose bounds the suite does not read. *)
let bounds =
  {
    cholesky = (fun n -> n + 1);
    lu = (fun k -> k + 2);
    solve = (fun n -> n + 2);
    alone = (fun ~e:_ _ _ -> invalid_arg "nx.amd computes no spectral routine");
    positive = true;
  }

(* Orders past every blocking: LU's panels of 64 columns and its rows held by 9
   groups over a launch a step (1100 rows), Cholesky's blocks of 64, a solve's
   leaves of 64, QR's blocks of 128 columns, a QR whose panels meet over 10
   groups (2500 rows), the unblocked algorithm's bits past them all, LU's where
   its rows are held by 5 groups (600 rows), and QR's bits whatever groups its
   rows split over. *)
let test_large (b : Support.backend) dt =
  law_lu bounds b (fixed dt 1100 100);
  law_cholesky bounds b (fixed dt 300 0);
  law_solve bounds b (fixed dt 300 3);
  law_solve bounds b (fixed ~triangle:Upper ~adjoint:true dt 200 5);
  law_qr b (fixed ~seed:10 dt 300 140);
  law_qr b (fixed ~seed:10 dt 2500 40);
  law_unblocked b (fixed dt 200 70);
  lu_unblocked b (fixed dt 600 70);
  List.iter (qr_blocks b dt) [ (300, 40); (4500, 40); (32768, 8) ]

(* Lanes and groups *)

(* Matrix [k] of a batch of one batch axis. *)
let matrix k (A.Any x) =
  let s = Iarray.to_array (Nx_array.Layout.shape (A.layout x)) in
  let whole e = { Nx_array.Move.start = 0; count = e; step = 1 } in
  let one =
    Array.mapi
      (fun i e ->
        if i = 0 then { Nx_array.Move.start = k; count = 1; step = 1 }
        else whole e)
      s
  in
  A.Any (Option.get (A.move (Nx_array.Move.Slice (Iarray.of_array one)) x))

(* [x] and [y] alike: floats as [same_bits] compares them, LU's int64 positions
   equal. *)
let alike ~msg (A.Any x) y =
  match A.dtype x with
  | D.Int64 ->
      let y = A.expect D.Int64 y in
      equal ~msg (array int64)
        (A.to_array (Support.host x))
        (A.to_array (Support.host y))
  | _ -> equal ~msg same_bits (bits (A.Any x)) (bits y)

(* Each routine on a batch of [count] matrices of case [c]'s order: the matrices
   [ks] have the bits they have alone. A batch of at least 128 matrices of at
   most 64 rows takes a wave a matrix, at most 16 rows 16 lanes a matrix,
   several matrices a group; alone, a matrix takes a group, or 16 lanes. *)
let batch_bits (b : Support.backend) c ~count ks =
  let c = { c with batch = [| count |] } in
  let check name r ops =
    match run b r ops with
    | None -> ()
    | Some y ->
        List.iter
          (fun k ->
            match run b r (Array.map (matrix k) ops) with
            | None -> ()
            | Some z ->
                Array.iteri
                  (fun i z ->
                    alike
                      ~msg:(Printf.sprintf "%s, matrix %d, result %d" name k i)
                      (matrix k y.(i))
                      z)
                  z)
          ks
  in
  check "cholesky" (Cholesky Lower) [| cholesky_operand c |];
  check "lu" Lu [| snd (lu_operand c) |];
  check "qr" (Qr (factors c)) [| snd (lu_operand c) |];
  check "solve" (solve c) (solve_operands c)

(* Orders on both sides of 16 lanes a matrix and of a wave a matrix. *)
let test_lanes (b : Support.backend) dt =
  List.iter
    (fun n -> batch_bits b (fixed dt n 3) ~count:200 [ 0; 1; 15; 16; 199 ])
    [ 1; 2; 16; 17; 33; 64; 65 ]

(* Past the 65,535 groups a launch has, a group takes the units 65,535 groups
   after its own: 16 matrices of order 2 a group, 1,048,577 of them; a solve's 4
   columns a work-item, 64 a group. The matrices before, at and past the first
   group's second unit have the bits they have alone. *)
let test_groups (b : Support.backend) () =
  let count = (16 * 65_535) + 17 in
  let c = { (fixed (D.Any D.Float32) 2 4) with seed = 3 } in
  batch_bits b c ~count [ 0; (16 * 65_535) - 1; 16 * 65_535; count - 1 ]

(* A solve of order 65 updates its last row by a product of 70,000 matrices,
   past the 65,535 groups along y or z a launch has, one matrix of [a]
   broadcast. *)
let test_update_groups (b : Support.backend) () =
  let count = 70_000 and n = 65 in
  let c = { (fixed (D.Any D.Float32) n 1) with batch = [| 1 |] } in
  let ops = solve_operands c in
  let (A.Any a) = ops.(0) in
  let wide =
    A.Any (Option.get (A.move (Nx_array.Move.Broadcast [| count; n; n |]) a))
  in
  let rhs =
    Array.init (count * n) (fun i ->
        { Complex.re = float_of_int (i mod 13) -. 6.; im = 0. })
  in
  let bs = of_values c.dt [| count; n; 1 |] rhs in
  match run b (solve c) [| wide; bs |] with
  | None -> ()
  | Some y ->
      List.iter
        (fun k ->
          match run b (solve c) [| ops.(0); matrix k bs |] with
          | None -> ()
          | Some z ->
              equal
                ~msg:(Printf.sprintf "matrix %d" k)
                same_bits
                (bits (matrix k y.(0)))
                (bits z.(0)))
        [ 0; 65_534; 65_535; count - 1 ]

(* LU's tall panel: at most 64 columns, 32 at complex128, of more than 512 rows
   in a batch of fewer than 32 take one launch a step over 9 groups, which read
   the operand where it lies and store perm themselves. Against its relation and
   the unblocked algorithm, and with rows further apart, read from its copy. *)
let test_tall_lu (b : Support.backend) dt =
  let c = fixed dt 1100 30 in
  law_lu bounds b c;
  lu_unblocked b c;
  let _, x = lu_operand c in
  match (run b Lu [| x |], run b Lu [| laid_out Padded x |]) with
  | Some y, Some z ->
      Array.iteri
        (fun i y -> alike ~msg:(Printf.sprintf "padded, result %d" i) y z.(i))
        y
  | _ -> ()

(* Read-only operands *)

(* [x] on [b]'s device over memory that admits only [Read]: a borrow of
   host pages holding a copy of its bytes. *)
let read_only (b : Support.backend) (A.Any x) =
  let y = Nx_array_support.mapped_read_only (Support.host x) in
  match Rig.Buffer.borrow b.device (A.buffer y) with
  | Some buf -> A.Any (A.v (A.dtype y) (A.layout y) buf)
  | None -> fail "the GPU maps no host pages"

(* Operands whose memory admits only [Read] give the bits writable ones give:
   read where they lie by an LU's and a QR's one launch and LU's tall panel,
   copied by an LU and a QR past one panel, and read by Cholesky and a solve. *)
let test_read_only (b : Support.backend) dt =
  let check name r ops =
    match (run b r ops, run b r (Array.map (read_only b) ops)) with
    | Some y, Some z ->
        Array.iteri
          (fun i y ->
            alike ~msg:(Printf.sprintf "%s, result %d" name i) y z.(i))
          y
    | _ -> ()
  in
  let few = fixed dt 12 3 and tall = fixed dt 1100 30 in
  let wide = fixed dt 100 70 and square = fixed dt 100 3 in
  check "lu, one launch" Lu [| snd (lu_operand few) |];
  check "qr, one launch" (Qr Reduced) [| snd (lu_operand few) |];
  check "lu, tall" Lu [| snd (lu_operand tall) |];
  check "lu, blocked" Lu [| snd (lu_operand wide) |];
  check "qr, blocked" (Qr Reduced) [| snd (lu_operand wide) |];
  check "cholesky" (Cholesky Lower) [| cholesky_operand square |];
  check "solve" (solve square) (solve_operands square)

(* Memory *)

let kept_workspace = 64 * 1024 * 1024
let kept_tickets = 4 * 1024 * 1024

(* Runs [f] with every call's workspace a buffer of its own, which returns to
   the device once its work is done. *)
let all_fresh f =
  let restore () =
    Nx_amd.bound_fresh ~workspace:kept_workspace ~tickets:kept_tickets
  in
  Fun.protect ~finally:restore (fun () ->
      Nx_amd.bound_fresh ~workspace:0 ~tickets:0;
      f ())

(* Routine [r] on [ops], [matrices] of them, in a loop: each call holds at most
   nx_amd.mli's [bytes] of workspace a matrix and the 3 KiB its alignment adds,
   a buffer of its own, and [tickets] bytes of tickets a matrix, another, each
   counted in the driver's units ({!Rig.allocation}), into the same results. *)
let held (b : Support.backend) ?(tickets = 0) r ops ~matrices ~bytes =
  let ops = Array.map (fun (A.Any x) -> A.Any (Support.on b x)) ops in
  let ys = Option.get (run b r ops) in
  let replay () =
    let i = ref 0 in
    fun _ _ ->
      let y = ys.(!i) in
      incr i;
      y
  in
  let bound =
    Rig.allocation b.device ((matrices * bytes) + 3072)
    + Rig.allocation b.device (matrices * tickets)
  in
  all_fresh (fun () ->
      Kernels_cases.loop_held b ~bound
        (fun () ->
          ignore (run ~alloc:(replay ()) b r ops);
          Array.iter
            (fun (A.Any y) -> Rig.Buffer.wait (A.buffer y) Read_write)
            ys))

let cdiv a b = (a + b - 1) / b

(* nx_amd.mli's workspace a matrix of a [cholesky] of order [n] past one block,
   at [e] bytes an element. *)
let cholesky_bytes ~e ~n =
  let w = Int.min n (if e = 16 then 32 else 64) in
  4 + (3 * (n - w) * ((w * e) + 16))

(* A [solve_triangular]'s past one leaf, [n] x [r], its [a] copied where
   [copied]. *)
let solve_bytes ~e ~n ~r ~copied =
  (if copied then n * n * e else 0) + 4 + ((n + (2 * r)) * ((n * e) + 16))

(* An [lu]'s of [m] x [n], in a batch of [matrices]. *)
let lu_bytes ~e ~m ~n ~matrices =
  let w = Int.min n (if e = 16 then 32 else 64) in
  let p = if matrices < 32 && m > 512 then cdiv m 128 else 1 in
  let panel = (m * (((w + 1) * e) + 16)) + (p * ((3 * w * e) + e + 117)) in
  if n > w then Int.max panel ((m + (2 * n)) * ((n * e) + 16)) else panel

(* A [qr]'s workspace of [m] x [n] into [c] columns of [q], [blocked] where
   it takes the second workspace, of a [complex] dtype or a real one. *)
let qr_bytes ~e ~complex ~m ~n ~c ~blocked =
  let k = Int.min m n in
  let w = Int.min n (if e = 16 then 16 else 32) in
  let b = Int.min k (if e = 16 then 64 else 128) in
  let big = Int.max n c in
  let first = e * (k + (m * n) + ((m + 800) * (w + 2)) + (w * w)) in
  let second =
    (e * ((m * b) + (cdiv k b * b * b) + (2 * b * big)))
    + ((b + (2 * big)) * m * e)
    + (16 * (m + (2 * big)))
  in
  (* The most outputs the tiles its products may take cover. *)
  let cols, e' = if complex then (2 * big, e / 2) else (big, e) in
  let tiles =
    if e' = 4 then [ (128, 128); (64, 64); (32, 32); (8, 256) ]
    else [ (64, 64); (32, 32); (8, 128) ]
  in
  let covered (h, v) = cdiv b h * h * cdiv cols v * v in
  let outputs = List.fold_left (fun n t -> Int.max n (covered t)) 0 tiles in
  let partials =
    if m >= 2048 then Int.min (cdiv m 512) 128 * outputs * e' else 0
  in
  first + if blocked then second + partials else 0

(* A [qr]'s tickets of [m] x [n] into [c] columns of [q], [blocked] as for
   {!qr_bytes}. *)
let qr_tickets ~e ~m ~n ~c ~blocked =
  let k = Int.min m n in
  let b = Int.min k (if e = 16 then 64 else 128) in
  if blocked && m >= 2048 then 4 * cdiv b 32 * cdiv (Int.max n c) 16 else 0

let f32 = D.Any D.Float32
let f64 = D.Any D.Float64
let c64 = D.Any D.Complex64
let c128 = D.Any D.Complex128
let with_batch batch c = { c with batch }

(* Cholesky past one block, real and complex, whose update is a product. *)
let test_cholesky_held b () =
  let c = with_batch [| 1000 |] (fixed f32 65 0) in
  held b (Cholesky Lower)
    [| cholesky_operand c |]
    ~matrices:1000
    ~bytes:(cholesky_bytes ~e:4 ~n:65);
  let c = with_batch [| 4 |] (fixed c128 100 0) in
  held b (Cholesky Upper)
    [| cholesky_operand c |]
    ~matrices:4
    ~bytes:(cholesky_bytes ~e:16 ~n:100)

(* A solve past one leaf whose [a]'s batch axes do not merge: a copy of [a], its
   flags and its products. *)
let test_solve_held b () =
  let n = 100 in
  let c = with_batch [| 2; 3 |] (fixed f64 n 3) in
  let swap (A.Any x) =
    A.Any (Option.get (A.move (Nx_array.Move.Permute [| 1; 0; 2; 3 |]) x))
  in
  held b (solve c)
    (Array.map swap (solve_operands c))
    ~matrices:6
    ~bytes:(solve_bytes ~e:8 ~n ~r:3 ~copied:true)

(* LU of a matrix whose rows 9 groups hold, past one panel and in its tall panel
   alone, and of a batch of 40 that spill their panels at complex128. *)
let test_lu_held b () =
  let one = fixed f32 1100 100 in
  held b Lu
    [| snd (lu_operand one) |]
    ~matrices:1
    ~bytes:(lu_bytes ~e:4 ~m:1100 ~n:100 ~matrices:1);
  let tall = fixed f32 1100 30 in
  held b Lu
    [| snd (lu_operand tall) |]
    ~matrices:1
    ~bytes:(lu_bytes ~e:4 ~m:1100 ~n:30 ~matrices:1);
  let many = with_batch [| 40 |] (fixed c128 300 100) in
  held b Lu
    [| snd (lu_operand many) |]
    ~matrices:40
    ~bytes:(lu_bytes ~e:16 ~m:300 ~n:100 ~matrices:40)

(* QR through its blocks, its panels' rows over 10 groups, and through its one
   launch with a copy of [a], 100 x 20 transposed. *)
let test_qr_held b () =
  let tall = fixed ~seed:10 f32 2500 40 in
  held b (Qr Reduced)
    ~tickets:(qr_tickets ~e:4 ~m:2500 ~n:40 ~c:40 ~blocked:true)
    [| snd (lu_operand tall) |]
    ~matrices:1
    ~bytes:(qr_bytes ~e:4 ~complex:false ~m:2500 ~n:40 ~c:40 ~blocked:true);
  let small = with_batch [| 3 |] (fixed ~seed:10 c64 20 100) in
  let flip (A.Any x) =
    A.Any (Option.get (A.move (Nx_array.Move.Permute [| 0; 2; 1 |]) x))
  in
  held b (Qr Complete)
    [| flip (snd (lu_operand small)) |]
    ~matrices:3
    ~bytes:(qr_bytes ~e:8 ~complex:true ~m:100 ~n:20 ~c:100 ~blocked:false)

(* nx_amd.mli: a device keeps a call's submission whatever its launches. An
   LU of 300 columns takes 5 panels of two launches each, more than 8. *)
let test_kept b () =
  let c = fixed f32 300 300 in
  let ops =
    Array.map (fun (A.Any x) -> A.Any (Support.on b x)) [| snd (lu_operand c) |]
  in
  Nx_amd.drop_kept b.device;
  ignore (run b Lu ops);
  ignore (run b Lu ops);
  equal int 1 (Nx_amd.kept_calls b.device)

(* The machine's AMD GPU 0, opened at a test's first call; a test skips
   without it. *)
let amd () =
  match Support.amd () with
  | Some b -> b
  | None -> skip ~reason:"no AMD GPU nx.amd computes on" ()

let laws () =
  let on f x = f (amd ()) x in
  let cases name f =
    cases name ~id:(Format.asprintf "%a" pp_dtype) dtypes (fun dt ->
        f (amd ()) dt)
  in
  group "amd"
    [
      group "each result meets its relation within its bound"
        [
          prop "Cholesky's factor within γ(n + 1)" case
            (on (law_cholesky bounds));
          prop "the Upper factor is the Lower one's conjugate transpose" case
            (on law_upper);
          prop "a triangular solve within γ(n + 2)" case
            (on (law_solve bounds));
          prop "LU's factors within γ(k + 2)" case (on (law_lu bounds));
          test "LU of a pivot whose reciprocal is subnormal"
            (on (test_huge_pivot bounds));
          test "QR of matrices scaled to the ends of the exponent range"
            (on test_qr_scaled);
          prop "QR's factors within γ(k(2m + 10))" case (on law_qr);
          cases "past every blocking" test_large;
          cases "LU's tall panel" test_tall_lu;
        ];
      group "bits are a function of the values alone"
        [
          prop "LU's, Cholesky's and a solve's are the unblocked algorithm's"
            case (on law_unblocked);
          prop "whatever the layout" case (on law_layouts);
          prop ~count:25 "whatever the batch" case (on law_batch);
          cases "whatever lanes a matrix takes" test_lanes;
          test "past 65,535 groups of matrices" (on test_groups);
          test "past 65,535 groups of an update" (on test_update_groups);
          cases "whatever memory the operands admit" test_read_only;
        ];
      group "calls hold the memory nx_amd.mli states"
        [
          test "cholesky" (on test_cholesky_held);
          test "solve_triangular" (on test_solve_held);
          test "lu" (on test_lu_held);
          test "qr" (on test_qr_held);
          test "a call of more than 8 launches keeps its submission"
            (on test_kept);
        ];
    ]

let () = exit (Windtrap.run "nx.amd factorisations" [ laws () ])
