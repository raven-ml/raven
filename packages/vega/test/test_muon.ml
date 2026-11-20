(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Vega's Muon step. *)

open Windtrap

let p = Nx.Ptree.tensor

(* A rate exact at float64, [Vega.rate]'s being float32. *)
let rate64 v step = Nx.scalar_at (Nx.placement step) Nx.float64 v
let counter c = Nx.scalar Nx.int32 (Int32.of_int c)
let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let bits x = Array.map Int64.bits_of_float (floats x)
let numel shape = Array.fold_left ( * ) 1 shape

(* Jordan et al.'s quintic on one singular value, and what its five steps make
   of a singular value of [d / |d|]. *)
let quintic s =
  let s2 = s *. s in
  s *. (3.4445 +. (s2 *. (-4.7750 +. (2.0315 *. s2))))

let orth_value s = quintic (quintic (quintic (quintic (quintic s))))

(* The band of a singular value of at least [|d| / 300]. *)
let band_low = 0.68
let band_high = 1.21
let band_floor = 1.0 /. 300.0

(* What an element of [orth d] read back from a step may be off by. The
   iteration grows a rounding error of the dtype the leaf computes at by at
   most [3.4445] a step, under [500] over five, and a half-precision leaf
   stores each element of its step within one spacing. *)
let tolerance (type b) (dt : (float, b) Nx.dtype) =
  let computed : float =
    match dt with Float64 -> epsilon_float | _ -> 0x1p-23
  in
  (4096.0 *. computed) +. (4.0 *. (Nx.Dtype.float_format dt).epsilon)

(* [factor m n] is what a step multiplies [orth d] by. *)
let factor m n = 0.2 *. sqrt (Float.of_int (max m n))

(* [orth ?stacks ~m ~n g] is [orth] of each matrix of [g], of [m] rows and [n]
   columns, as a step shows it, at float64. From zero parameters and a fresh
   state, at a unit rate and without decay, the parameters are
   [-factor m n * orth (1.95 * g)], and [orth] follows the direction of its
   argument alone. *)
let orth ?stacks ~m ~n g =
  let params = Nx.zeros_like g in
  let params', _ =
    Vega.muon_step p ~lr:(rate64 1.0) ~weight_decay:(Vega.rate 0.0) ?stacks
      (Vega.muon_init p params) ~params ~grads:g
  in
  Nx.div_s (Nx.cast Nx.float64 params') (-.factor m n)

(* The singular values against the singular vectors *)

(* [judged s] says which of the singular values [s] count as zero, and which
   are held in the band: those of at least [|d| / 300]. *)
let judged s =
  let norm = sqrt (Array.fold_left (fun sum x -> sum +. (x *. x)) 0.0 s) in
  let zero x = x <= 1e-12 *. norm in
  let held x = x >= norm *. band_floor && not (zero x) in
  (zero, held)

(* [in_band ~tol (u, s, vh) x] checks [x], an [orth d] read back from a step,
   against the singular value decomposition [u s vᵀ] of [d] at float64. [x]
   is [u c vᵀ] for the values [c] it gives [d]'s singular values: each in the
   band where [d]'s is held, none above the band, and zero where [d]'s is
   zero. *)
let in_band ~tol (u, s, vh) x =
  let s = Nx.to_array s in
  let zero, held = judged s and k = Array.length s in
  let ut = Nx.matrix_transpose u and v = Nx.matrix_transpose vh in
  let c = Nx.matmul ut (Nx.matmul x v) in
  let c = Array.init k (fun i -> Nx.item [ i; i ] c) in
  let diagonal = Nx.create Nx.float64 [| 1; k |] c in
  equal ~msg:"d's singular vectors"
    (array (float tol))
    (Nx.to_array (Nx.matmul (Nx.mul u diagonal) vh))
    (Nx.to_array x);
  Array.iteri
    (fun i c ->
      let msg = Printf.sprintf "singular value %d of %d" (i + 1) k in
      at_most ~msg float_exact ~than:(band_high +. tol) c;
      if held s.(i) then at_least ~msg float_exact ~than:(band_low -. tol) c;
      if zero s.(i) then at_most ~msg float_exact ~than:tol (Float.abs c))
    c

(* Rows and columns, each kind of shape drawn often. *)
let shapes =
  let open Gen in
  let side = int_range 2 5 in
  frequency
    [
      (2, constant (1, 1));
      (2, pair (constant 1) side);
      (2, pair side (constant 1));
      (1, pair (constant 0) (int_range 0 3));
      (1, pair (int_range 1 3) (constant 0));
      (6, pair side side);
    ]

(* The matrix of singular values [s] on the singular vectors of the matrix
   [r], both of [m] rows and [n] columns, in C order. *)
let with_spectrum m n r s =
  let u, _, vh = Nx.svd Nx.float64 (Nx.create Nx.float64 [| m; n |] r) in
  let s = Nx.create Nx.float64 [| 1; Array.length s |] s in
  Nx.to_array (Nx.matmul (Nx.mul u s) vh)

(* A matrix of [m] rows and [n] columns at float64, in C order. It is a sum
   of [r] outer products of small integers, of rank [r] at most and exact in
   every float dtype, or drawn singular values, some below the band's floor
   and some zero, on the singular vectors of a drawn matrix. *)
let matrix m n =
  let open Gen in
  let k = min m n in
  let integers =
    let vector size = array ~size:(constant size) (int_range (-3) 3) in
    let* r =
      frequency
        [ (1, constant 0); (3, constant (min 1 k)); (3, int_range 0 k) ]
    in
    let+ terms = list ~size:(constant r) (pair (vector m) (vector n)) in
    let element e =
      let i = e / n and j = e mod n in
      List.fold_left (fun sum (a, b) -> sum + (a.(i) * b.(j))) 0 terms
    in
    Array.init (m * n) (fun e -> Float.of_int (element e))
  in
  let spectrum =
    let value =
      frequency
        [
          (3, float_range band_floor 1.0);
          (2, float_range 1e-4 band_floor);
          (1, constant 0.0);
        ]
    in
    let+ r = array ~size:(constant (m * n)) (float_range (-1.0) 1.0)
    and+ s = array ~size:(constant k) value in
    with_spectrum m n r s
  in
  if k = 0 then constant [||] else frequency [ (3, integers); (2, spectrum) ]

(* [orth d] against the singular value decomposition of [d] as the leaf's
   dtype stores it ([in_band]). A matrix of no element steps to one. *)
let follows_the_svd (type b) (dt : (float, b) Nx.dtype) =
  let tol = tolerance dt in
  let exact = match dt with Float32 | Float64 -> true | _ -> false in
  let gen =
    let open Gen in
    let* m, n = shapes in
    let+ d = matrix m n in
    (m, n, d)
  in
  prop ~count:300 "orth d has d's singular vectors and its values in their band"
    gen (fun (m, n, d) ->
      let k = min m n in
      let g = Nx.cast dt (Nx.create Nx.float64 [| m; n |] d) in
      let svd =
        if k = 0 then None else Some (Nx.svd Nx.float64 (Nx.cast Nx.float64 g))
      in
      let s =
        match svd with None -> [||] | Some (_, s, _) -> Nx.to_array s
      in
      let zero, held = judged s in
      let count f = Array.fold_left (fun c x -> if f x then c + 1 else c) 0 s in
      cover "m < n" (m < n);
      cover "m = n" (m = n);
      cover "m > n" (m > n);
      cover "one row" (m = 1 && n > 1);
      cover "one column" (n = 1 && m > 1);
      cover "one element" (m = 1 && n = 1);
      cover "an empty axis" (k = 0);
      cover "rank one" (k > 1 && count zero = k - 1);
      cover "a zero matrix" (k > 0 && count zero = k);
      cover "a zero singular value beside one in the band"
        (count zero > 0 && count held > 0);
      (* A half-precision leaf rounds a drawn value below the floor to one
         near it or above. *)
      if exact then
        cover "a singular value below the band's floor"
          (count (fun x -> not (zero x || held x)) > 0);
      let x = orth ~m ~n g in
      equal ~msg:"shape" (array int) [| m; n |] (Nx.shape x);
      Option.iter (fun svd -> in_band ~tol svd x) svd)

(* Scaling *)

(* Values on a grid of [1/64] in [-4; 4]. Each element a step computes from
   them on the way to [orth]'s argument is zero or of magnitude at least
   [2^-54], so that scaled by [2^k] for [k] in [-60; 60] it stays normal at
   float32 and every operation rounds as it did unscaled. *)
let grid size =
  let open Gen in
  array ~size:(constant size)
    (map (fun i -> Float.of_int i /. 64.0) (int_range (-256) 256))

let scale_free (type b) (dt : (float, b) Nx.dtype) =
  let gen =
    let open Gen in
    let* m, n = pair (int_range 1 4) (int_range 1 4) in
    let leaf = grid (m * n) in
    let+ x = leaf and+ g = leaf and+ v = leaf and+ k = int_range (-60) 60 in
    (m, n, x, g, v, k)
  in
  prop "gradients and a velocity scaled by a power of two step bit for bit"
    gen (fun (m, n, x, g, v, k) ->
      cover "scaled down" (k < -20);
      cover "scaled up" (k > 20);
      let leaf a = Nx.create dt [| m; n |] a in
      let step scale =
        let scaled a = leaf (Array.map (fun e -> e *. scale) a) in
        let st = { Vega.velocity = scaled v; step = counter 3 } in
        fst
          (Vega.muon_step p ~lr:(Vega.rate 0.1) st ~params:(leaf x)
             ~grads:(scaled g))
      in
      equal (array int64) (bits (step 1.0)) (bits (step (Float.ldexp 1.0 k))))

(* Layouts *)

(* How a law makes and reads the leaves of one dtype. [leaf shape a] is the
   leaf of [shape] whose elements are [a]'s in C order, [width] floats an
   element, the two parts of a complex one. [read] is its inverse at float64.
   [tol] is what an element of a step may be off by. *)
type kind =
  | Kind : {
      name : string;
      width : int;
      leaf : int array -> float array -> ('v, 's, Nx.host) Nx.t;
      read : ('v, 's, Nx.host) Nx.t -> float array;
      tol : float;
    }
      -> kind

let real (type b) (dt : (float, b) Nx.dtype) =
  let leaf shape a = Nx.create dt shape a in
  let name = Nx.Dtype.name dt and tol = tolerance dt in
  Kind { name; width = 1; leaf; read = floats; tol }

let complex dt parts =
  let leaf shape a =
    let element e = { Complex.re = a.(2 * e); im = a.((2 * e) + 1) } in
    Nx.create dt shape (Array.init (numel shape) element)
  and read z =
    let last = Nx.ndim z in
    Nx.to_array
      (Nx.stack ~axis:last [ Nx.real Nx.float64 z; Nx.imag Nx.float64 z ])
  in
  let name = Nx.Dtype.name dt and tol = tolerance parts in
  Kind { name; width = 2; leaf; read; tol }

let kinds =
  [
    real Nx.float32;
    real Nx.float64;
    complex Nx.complex64 Nx.float32;
    complex Nx.complex128 Nx.float64;
  ]

(* [each law] is [law] at each kind, grouped under the kinds' names. *)
let each law =
  List.map (fun (Kind { name; _ } as kind) -> group name [ law kind ]) kinds

(* The parameters, the gradients and the velocity of a leaf of [shape], in C
   order. *)
let drawn (Kind { width; _ }) shape =
  let size = width * numel shape in
  let values = Gen.(array ~size:(constant size) (float_range (-4.0) 4.0)) in
  Gen.triple values values values

(* One step from the parameters [x], the gradients [g] and the velocity [v] of
   a leaf of [shape]: the parameters and the velocity it returns. *)
let step (Kind k) ?stacks shape (x, g, v) =
  let st = { Vega.velocity = k.leaf shape v; step = counter 0 } in
  let y, (st : (_, _) Vega.sgd_state) =
    Vega.muon_step p ~lr:(Vega.rate 1.0) ?stacks st ~params:(k.leaf shape x)
      ~grads:(k.leaf shape g)
  in
  (k.read y, k.read st.velocity)

(* The transposed leaf's Gram product sums over its other axis, and a square
   leaf's forms the other square, so the two steps agree within rounding. The
   velocity is elementwise. *)
let transposes (type b) (dt : (float, b) Nx.dtype) =
  let kind = real dt and tol = tolerance dt in
  let gen =
    let open Gen in
    let* m, n = pair (int_range 1 4) (int_range 1 4) in
    let+ leaves = drawn kind [| m; n |] in
    (m, n, leaves)
  in
  prop "a transposed leaf gives the transposed step" gen
    (fun (m, n, (x, g, v)) ->
      cover "a square leaf" (m = n);
      cover "a wide leaf" (m < n);
      cover "a tall leaf" (m > n);
      (* Element [e] of the transposed leaf is at row [e mod m] and column
         [e / m] of the leaf. *)
      let transposed a =
        Array.init (m * n) (fun e -> a.((e mod m * n) + (e / m)))
      in
      let y, w = step kind [| m; n |] (x, g, v) in
      let yt, wt =
        step kind [| n; m |] (transposed x, transposed g, transposed v)
      in
      equal ~msg:"parameters" (array (float tol)) (transposed y) yt;
      equal ~msg:"velocity" (array float_exact) (transposed w) wt)

(* A kernel under [~stacks:(fun _ _ -> 0)] is one matrix of its first axis by
   the others merged: the leaf of two axes with its elements, bit for bit. *)
let kernels kind =
  let gen =
    let open Gen in
    let* shape = array ~size:(int_range 3 4) (int_range 1 3) in
    let+ leaves = drawn kind shape in
    (shape, leaves)
  in
  prop "a kernel under ~stacks:(fun _ _ -> 0) steps as its [out; rest] reshape"
    gen (fun (shape, leaves) ->
      cover "a kernel of four axes" (Array.length shape = 4);
      let out = shape.(0) in
      equal
        (pair (array float_exact) (array float_exact))
        (step kind [| out; numel shape / out |] leaves)
        (step kind ~stacks:(fun _ _ -> 0) shape leaves))

(* A stack under [~stacks:(fun _ _ -> 1)] is its matrices along its first
   axis. A batched product may round apart from a product of two matrices, so
   the parameters agree within rounding. *)
let stacks (Kind { width; tol; _ } as kind) =
  let gen =
    let open Gen in
    let side = int_range 1 3 in
    let* count = frequency [ (1, constant 0); (5, side) ] in
    let* m, n = pair side side in
    let+ leaves = drawn kind [| count; m; n |] in
    (count, m, n, leaves)
  in
  prop "a stack under ~stacks:(fun _ _ -> 1) steps as its matrices one by one"
    gen (fun (count, m, n, (x, g, v)) ->
      cover "a stack of no matrix" (count = 0);
      cover "a stack of several matrices" (count > 1);
      let size = width * m * n in
      let matrix i a = Array.sub a (i * size) size in
      let one i = step kind [| m; n |] (matrix i x, matrix i g, matrix i v) in
      let each = List.init count one in
      let y, w = step kind ~stacks:(fun _ _ -> 1) [| count; m; n |] (x, g, v) in
      equal ~msg:"parameters"
        (array (float tol))
        (Array.concat (List.map fst each))
        y;
      equal ~msg:"velocity" (array float_exact)
        (Array.concat (List.map snd each))
        w)

(* A leaf of three or four axes under [~stacks:(fun _ _ -> k)] holds a matrix
   for each index of its first [k] axes, of axis [k] as rows and the axes
   after it merged as columns: it steps as the stack of those matrices along
   one axis, within the rounding of a batch of another shape. *)
let any_stacks (Kind { tol; _ } as kind) =
  let gen =
    let open Gen in
    let* shape = array ~size:(int_range 3 4) (int_range 1 3) in
    let* k = int_range 0 (Array.length shape - 2) in
    let+ leaves = drawn kind shape in
    (shape, k, leaves)
  in
  prop "a leaf under ~stacks:(fun _ _ -> k) steps as the stack of its matrices"
    gen (fun (shape, k, leaves) ->
      let rank = Array.length shape in
      cover "one matrix" (k = 0);
      cover "three axes, a stack of matrices" (rank = 3 && k = 1);
      cover "four axes, rows between a stack axis and two merged ones"
        (rank = 4 && k = 1);
      cover "four axes, two stack axes" (k = 2);
      let count = numel (Array.sub shape 0 k) and rows = shape.(k) in
      let columns = numel shape / (count * rows) in
      let y, w = step kind ~stacks:(fun _ _ -> k) shape leaves in
      let ys, ws =
        step kind ~stacks:(fun _ _ -> 1) [| count; rows; columns |] leaves
      in
      equal ~msg:"parameters" (array (float tol)) ys y;
      equal ~msg:"velocity" (array float_exact) ws w)

(* Complex leaves *)

(* [real_form parts z] is [[re, -im]; [im, re]] at the parts' dtype, the real
   matrix by which the complex matrix [z] acts on its real coordinates. *)
let real_form parts z =
  let re = Nx.real parts z and im = Nx.imag parts z in
  Nx.concatenate ~axis:0
    [
      Nx.concatenate ~axis:1 [ re; Nx.neg im ];
      Nx.concatenate ~axis:1 [ im; re ];
    ]

let complex_values size =
  let part = Gen.float_range (-4.0) 4.0 in
  Gen.(array ~size:(constant size) (pair part part))

let complex_leaf dt m n zs =
  Nx.create dt [| m; n |] (Array.map (fun (re, im) -> { Complex.re; im }) zs)

(* The floats of [x] with each zero positive. *)
let unsigned x = Array.map (fun e -> e +. 0.0) (floats x)

(* A complex leaf's step is the first block column of the step of its real
   form, whose gradient is the real form of the conjugate of the leaf's and
   whose velocity that of the leaf's. The two compute the same floats, but
   for the signs of zeros. The real form's other block column repeats the
   first within rounding. *)
let as_real_form dt parts =
  let tol = tolerance parts in
  let gen =
    let open Gen in
    let side = frequency [ (1, constant 0); (5, int_range 1 3) ] in
    let* m, n = pair side side in
    let leaf = complex_values (m * n) in
    let+ leaves = triple leaf leaf leaf in
    (m, n, leaves)
  in
  prop "a complex leaf steps as the leaf of its real form" gen
    (fun (m, n, (z, g, v)) ->
      cover "an empty leaf" (m * n = 0);
      cover "a leaf of several rows and columns" (m > 1 && n > 1);
      let z = complex_leaf dt m n z
      and g = complex_leaf dt m n g
      and v = complex_leaf dt m n v in
      let lr = Vega.rate 0.5 in
      let z', (st : (_, _) Vega.sgd_state) =
        Vega.muon_step p ~lr
          { Vega.velocity = v; step = counter 0 }
          ~params:z ~grads:g
      in
      let r', (rst : (_, _) Vega.sgd_state) =
        Vega.muon_step p ~lr
          { Vega.velocity = real_form parts v; step = counter 0 }
          ~params:(real_form parts z)
          ~grads:(real_form parts (Nx.conjugate g))
      in
      let first x = Nx.slice [ Nx.A; Nx.R (0, n) ] x in
      equal ~msg:"parameters, the first block column" (array float_exact)
        (unsigned (first (real_form parts z')))
        (unsigned (first r'));
      equal ~msg:"parameters" (array (float tol))
        (floats (real_form parts z'))
        (floats r');
      equal ~msg:"velocity" (array float_exact)
        (unsigned (real_form parts st.velocity))
        (unsigned rst.velocity))

(* The momentum of a fresh state is along the conjugate of the gradient, so
   a gradient turned by [-theta] turns the step by [theta]. *)
let turns dt parts =
  let tol = tolerance parts in
  let gen =
    let open Gen in
    let* m, n = pair (int_range 1 3) (int_range 1 3) in
    let theta =
      frequency
        [
          (1, of_list [ 0.0; Float.pi /. 2.0; Float.pi ]);
          (3, float_range 0.0 (2.0 *. Float.pi));
        ]
    in
    let+ g = complex_values (m * n) and+ theta = theta in
    (m, n, g, theta)
  in
  prop "a phase on the momentum is a phase on the step" gen
    (fun (m, n, g, theta) ->
      let g = complex_leaf dt m n g in
      let step g =
        let params = Nx.zeros_like g in
        fst
          (Vega.muon_step p ~lr:(Vega.rate 1.0) ~weight_decay:(Vega.rate 0.0)
             (Vega.muon_init p params) ~params ~grads:g)
      in
      let phase = { Complex.re = cos theta; im = sin theta } in
      let coordinates z =
        Nx.to_array (Nx.stack [ Nx.real Nx.float64 z; Nx.imag Nx.float64 z ])
      in
      equal
        (array (float tol))
        (coordinates (Nx.mul_s (step g) phase))
        (coordinates (step (Nx.mul_s g (Complex.conj phase)))))

(* The formula *)

let diagonal d = Nx.create Nx.float64 [| 2; 2 |] [| d.(0); 0.0; 0.0; d.(1) |]

(* [orth] of the diagonal matrix of [d]: each entry over the Frobenius norm,
   through the five steps. *)
let orth_diagonal d =
  let norm = sqrt ((d.(0) *. d.(0)) +. (d.(1) *. d.(1))) in
  Array.map (fun x -> orth_value (x /. norm)) d

(* The defaults the interface states. [Vega.rate]'s decay is a float32. *)
let default_momentum = 0.95
let default_decay = Int32.float_of_bits (Int32.bits_of_float 0.01)

(* A product rounded on its own, as a step's is: on arm64 the compiler would
   fuse it with the sum that reads it. *)
let product a b = Sys.opaque_identity (a *. b)

(* Diagonal matrices at float64, whose step has a closed form: a velocity
   [v] and a gradient [g] make [v' = momentum * v + g] and the momentum
   [d = g + momentum * v'], or [v'] without Nesterov's form, and the
   parameters [p] step to [p - lr * (factor * orth d + wd * p)]. A row passes
   the arguments it names, and the step's defaults stand for the others. *)
let formula =
  let lr = 0.5 and x = [| 2.0; -4.0 |] in
  let rows =
    [
      (* name, momentum, nesterov, weight decay, v, g *)
      ("the defaults", None, None, None, [| 1.0; -2.0 |], [| 3.0; 0.5 |]);
      ( "momentum = 0.",
        Some 0.0,
        None,
        Some 0.25,
        [| 5.0; -2.0 |],
        [| 3.0; 4.0 |] );
      ( "without nesterov",
        Some 0.5,
        Some false,
        Some 0.25,
        [| 1.0; 0.0 |],
        [| 0.0; 1.0 |] );
      ( "with nesterov",
        Some 0.5,
        Some true,
        Some 0.25,
        [| 1.0; 0.0 |],
        [| 0.0; 1.0 |] );
    ]
  in
  cases "a diagonal matrix steps by the formula"
    ~id:(fun (name, _, _, _, _, _) -> name)
    rows
    (fun (_, momentum, nesterov, decay, v, g) ->
      let st = { Vega.velocity = diagonal v; step = counter 0 } in
      let y, (st : (_, _) Vega.sgd_state) =
        Vega.muon_step p ~lr:(rate64 lr) ?momentum
          ?weight_decay:(Option.map rate64 decay)
          ?nesterov st ~params:(diagonal x) ~grads:(diagonal g)
      in
      let momentum = Option.value momentum ~default:default_momentum in
      let wd = Option.value decay ~default:default_decay in
      let v' = Array.map2 (fun v g -> product momentum v +. g) v g in
      let d =
        if Option.value nesterov ~default:true then
          Array.map2 (fun g v' -> g +. product momentum v') g v'
        else v'
      in
      let o = orth_diagonal d in
      let expected i =
        x.(i) -. (lr *. ((factor 2 2 *. o.(i)) +. (wd *. x.(i))))
      in
      equal ~msg:"parameters"
        (array (float 1e-12))
        [| expected 0; 0.0; 0.0; expected 1 |]
        (Nx.to_array y);
      equal ~msg:"velocity" (array float_exact)
        (Nx.to_array (diagonal v'))
        (unsigned st.velocity))

(* The start *)

(* A zero momentum has no direction: the leaf steps by its decay alone, and
   by nothing where it does not decay. *)
let test_zero_momentum () =
  let x =
    Nx.create Nx.float64 [| 2; 3 |] [| 1.0; -2.0; 0.5; 4.0; 0.0; -8.0 |]
  in
  let zero = Nx.zeros_like x in
  let step ?decays () =
    fst
      (Vega.muon_step p ~lr:(rate64 0.5) ~weight_decay:(rate64 0.5) ?decays
         (Vega.muon_init p x) ~params:x ~grads:zero)
  in
  equal ~msg:"with decay" (array float_exact)
    (Array.map (fun e -> e *. 0.75) (Nx.to_array x))
    (unsigned (step ()));
  equal ~msg:"without decay" (array float_exact) (Nx.to_array x)
    (Nx.to_array (step ~decays:(fun _ _ -> false) ()))

(* [orth] is zero below the least normal float of the dtype the leaf computes
   at, and whole from it on: the diagonal matrix of [t] and [t] steps by
   nothing for the greatest subnormal [t], and for the least normal one each
   diagonal element by [factor * orth_value (1 / sqrt 2)]. *)
let least_normal (type b) (dt : (float, b) Nx.dtype) () =
  let format = Nx.Dtype.float_format dt in
  let least = format.min_normal in
  let subnormal = least *. (1.0 -. format.epsilon) in
  let step t =
    let g = Nx.create dt [| 2; 2 |] [| t; 0.0; 0.0; t |] in
    let params = Nx.zeros_like g in
    let y, _ =
      Vega.muon_step p ~lr:(Vega.rate 1.0) ~momentum:0.0
        ~weight_decay:(Vega.rate 0.0) (Vega.muon_init p params) ~params
        ~grads:g
    in
    floats y
  in
  let whole = -.factor 2 2 *. orth_value (1.0 /. sqrt 2.0) in
  equal ~msg:"the greatest subnormal float" (array float_exact)
    [| 0.0; 0.0; 0.0; 0.0 |]
    (Array.map Float.abs (step subnormal));
  equal ~msg:"the least normal float"
    (array (float (tolerance dt)))
    [| whole; 0.0; 0.0; whole |]
    (step least)

(* Sixteen elements of [2^126] have a Frobenius norm of [2^128], above
   float32's greatest value, and each square overflows. The matrix has rank
   one, so [orth] of it is [orth_value 1] over its four singular vectors'
   elements, a quarter each. *)
let test_overflowing_norm () =
  let g = Nx.full Nx.float32 [| 4; 4 |] 0x1p126 in
  equal
    (array (float (tolerance Nx.float32)))
    (Array.make 16 (orth_value 1.0 /. 4.0))
    (Nx.to_array (orth ~m:4 ~n:4 g))

(* A NaN or an infinity makes its own matrix NaN, and leaves the other of the
   stack as it steps alone. *)
let not_finite (type b) (dt : (float, b) Nx.dtype) =
  cases "a NaN or an infinity in one matrix of a stack stays in it"
    ~id:(Printf.sprintf "%h")
    [ nan; infinity; neg_infinity ]
    (fun bad ->
      let first = [| bad; 1.0; 2.0; 3.0 |]
      and second = [| 1.0; -2.0; 0.5; 4.0 |] in
      let g = Nx.create dt [| 2; 2; 2 |] (Array.append first second) in
      let x = Nx.to_array (orth ~stacks:(fun _ _ -> 1) ~m:2 ~n:2 g) in
      let alone =
        Nx.to_array (orth ~m:2 ~n:2 (Nx.create dt [| 2; 2 |] second))
      in
      equal ~msg:"its own matrix" (array float_exact) (Array.make 4 nan)
        (Array.sub x 0 4);
      equal ~msg:"the other matrix"
        (array (float (tolerance dt)))
        alone (Array.sub x 4 4))

(* The state *)

(* Muon's state is SGD's: a velocity that [sgd_step] made is the one
   [muon_step] reads, and the counter goes on. From a zero velocity and the
   gradient [g], [sgd_step] leaves the velocity [g] and the parameters
   [x - lr * g]. [muon_step] then makes the velocity [1.5 * g] and the
   momentum [1.75 * g] at a momentum of a half. The numbers are binary
   fractions. *)
let test_steps_an_sgd_state () =
  let lr = 0.25 and wd = 0.25 in
  let x = [| 2.0; -4.0 |] and g = [| 4.0; 8.0 |] in
  let grads = diagonal g in
  let params, st =
    Vega.sgd_step p ~lr:(rate64 lr) ~momentum:0.5
      (Vega.sgd_init p (diagonal x))
      ~params:(diagonal x) ~grads
  in
  let params, st =
    Vega.muon_step p ~lr:(rate64 lr) ~momentum:0.5 ~weight_decay:(rate64 wd)
      st ~params ~grads
  in
  let x = Array.map2 (fun x g -> x -. (lr *. g)) x g in
  let o = orth_diagonal (Array.map (fun g -> 1.75 *. g) g) in
  let expected i = x.(i) -. (lr *. ((factor 2 2 *. o.(i)) +. (wd *. x.(i)))) in
  equal ~msg:"parameters"
    (array (float 1e-12))
    [| expected 0; 0.0; 0.0; expected 1 |]
    (Nx.to_array params);
  equal ~msg:"velocity" (array float_exact) [| 6.0; 0.0; 0.0; 12.0 |]
    (unsigned st.velocity);
  equal ~msg:"counter" int32 2l (Nx.item [] st.step)

(* The leaves *)

(* A matrix, a kernel, a complex kernel and a counter. *)
let leaves () =
  let p = Nx.Ptree.(quad tensor tensor tensor tensor) in
  let x =
    ( Nx.ones Nx.float32 [| 2; 3 |],
      Nx.ones Nx.float64 [| 2; 2; 2 |],
      Nx.ones Nx.complex64 [| 2; 1; 3 |],
      Nx.create Nx.int32 [| 4 |] [| 3l; 1l; 4l; 1l |] )
  in
  (p, x)

(* A rule that records what it is asked, and what it was asked, sorted. *)
let counting answer =
  let asked = ref [] in
  let rule path shape =
    asked := (Nx.Ptree.Path.to_string path, Array.to_list shape) :: !asked;
    answer
  in
  (rule, fun () -> List.sort compare !asked)

(* [~stacks] and [~decays] are each asked once per float or complex leaf, with
   the parameter's own shape. A leaf that is neither passes through. *)
let test_rules_are_asked () =
  let p, x = leaves () in
  let stacks, stacked = counting 0 and decays, decayed = counting true in
  let (_, _, _, seen), _ =
    Vega.muon_step p ~lr:(Vega.rate 0.1) ~decays ~stacks (Vega.muon_init p x)
      ~params:x ~grads:x
  in
  let asked = list (pair string (list int)) in
  let expected =
    [ ("0", [ 2; 3 ]); ("1", [ 2; 2; 2 ]); ("2", [ 2; 1; 3 ]) ]
  in
  equal ~msg:"~stacks" asked expected (stacked ());
  equal ~msg:"~decays" asked expected (decayed ());
  equal ~msg:"the counter" (array int32) [| 3l; 1l; 4l; 1l |] (Nx.to_array seen)

(* Leaves cut across devices *)

(* Leaves on two software devices. A matrix's Gram product sums over its
   longer axis, the columns of a square one. Cut along that axis, the devices
   sum by partials: the leaf steps where it lies, and [orth] holds its band.
   Cut along the other axis the product has no placement, and the step raises
   as [Nx.matmul] does.

   A stack cut along its stack axis holds whole matrices on each device. A
   kernel of [3; 2; 4] is one matrix of 8 columns: a reshape keeps the cut of
   its second axis, whose windows hold whole runs of the columns, and reads
   its third whole. A complex leaf's real form joins its parts along both
   axes of its matrices, and [Nx.concatenate] reads the axis it joins whole.

   A refusal is nx's own, in the words of its routing: the function, the
   operand and where it lies, where it is read, and the [place] that puts it
   there. *)
let test_cut_leaves () =
  let device name = Result.get_ok (Rig.software name) in
  let d0 = device "vega-muon-0" and d1 = device "vega-muon-1" in
  let module S = (val Nx.devices [ d0; d1 ]) in
  let element e = sin (Float.of_int ((7 * e) + 1)) in
  let leaf shape =
    Nx.create Nx.float32 shape (Array.init (numel shape) element)
  in
  let tol = tolerance Nx.float32 in
  let cut axis g = Nx.place (S.split ~axis) g in
  (* [refused by (i, name) axis]: [by] refuses its operand [i], which lies
     cut along [axis] and is read whole on each device. *)
  let refused by (operand, name) axis =
    let devices = "[vega-muon-0; vega-muon-1]" in
    Invalid_argument
      (Printf.sprintf
         "%s: operand %d lies at split ~axis:%d %s and is read at on %s; \
          Nx.place (Placement.on s) %s"
         by operand axis devices devices name)
  in
  List.iter
    (fun (m, n, longer, operand, lies) ->
      let g = leaf [| m; n |] in
      let svd = Nx.svd Nx.float64 (Nx.cast Nx.float64 g) in
      let stepped axis = orth ~m ~n (cut axis g) in
      let msg = Printf.sprintf "[%d; %d] cut along axis %d" m n in
      subtest (msg longer) (fun () ->
          in_band ~tol svd (Nx.place Nx.Host.on (stepped longer)));
      subtest (msg (1 - longer)) (fun () ->
          raises (refused "Nx.matmul" operand lies) (fun () ->
              stepped (1 - longer))))
    [
      (3, 6, 1, (1, "y"), 1); (6, 3, 0, (1, "y"), 1); (4, 4, 1, (1, "y"), 1);
    ];
  (* A leaf cut across the devices steps as it does on the host, within
     rounding. *)
  let as_whole name host placed =
    subtest name (fun () ->
        equal
          (array (float tol))
          (Nx.to_array host)
          (Nx.to_array (Nx.place Nx.Host.on placed)))
  in
  let stack = leaf [| 2; 3; 6 |] in
  let stacked g = orth ~stacks:(fun _ _ -> 1) ~m:3 ~n:6 g in
  as_whole "a stack cut along its stack axis" (stacked stack)
    (stacked (cut 0 stack));
  as_whole "a stack cut along the longer axis of its matrices" (stacked stack)
    (stacked (cut 2 stack));
  let kernel = leaf [| 3; 2; 4 |] in
  let one g = orth ~stacks:(fun _ _ -> 0) ~m:3 ~n:8 g in
  as_whole "a kernel cut along the first axis of its columns" (one kernel)
    (one (cut 1 kernel));
  subtest "a kernel cut along the last axis of its columns" (fun () ->
      raises
        (refused "Nx.reshape" (0, "x") 2)
        (fun () -> one (cut 2 kernel)));
  let z =
    let part e = { Complex.re = element e; im = element (e + 18) } in
    Nx.create Nx.complex64 [| 3; 6 |] (Array.init 18 part)
  in
  List.iter
    (fun axis ->
      let name = Printf.sprintf "a complex leaf cut along axis %d" axis in
      subtest name (fun () ->
          raises
            (refused "Nx.concatenate" (0, "x") axis)
            (fun () -> orth ~m:6 ~n:12 (cut axis z))))
    [ 0; 1 ]

(* Refusals *)

let refusals =
  let lr = Vega.rate 0.1 in
  let matrix = Nx.ones Nx.float32 [| 2; 3 |] in
  let beside leaf = (matrix, leaf) in
  let step ?momentum ?stacks x () =
    let pairs = Nx.Ptree.(pair tensor tensor) in
    ignore
      (Vega.muon_step pairs ?momentum ?stacks ~lr (Vega.muon_init pairs x)
         ~params:x ~grads:x)
  in
  let kernel = beside (Nx.ones Nx.float32 [| 2; 3; 4; 5 |]) in
  (* A rule that answers [k] for a leaf of more than two axes. *)
  let past_matrices k _ shape = if Array.length shape = 2 then 0 else k in
  let lists = Nx.Ptree.(list tensor) in
  let two = [ matrix; matrix ] and one = [ matrix ] in
  let rows =
    [
      ( "momentum = 1.",
        "Vega.muon_step: expected 0.0 <= momentum < 1.0, got 1",
        step ~momentum:1.0 (beside matrix) );
      ( "a leaf of one axis",
        "Vega.muon_step: 1: shape [3] has fewer than two axes, so it holds no \
         matrix; step this leaf in another part",
        step (beside (Nx.ones Nx.float32 [| 3 |])) );
      ( "a complex leaf of no axis",
        "Vega.muon_step: 1: shape [] has fewer than two axes, so it holds no \
         matrix; step this leaf in another part",
        step (beside (Nx.ones Nx.complex64 [||])) );
      ( "a leaf of one axis under ~stacks",
        "Vega.muon_step: 1: shape [3] has fewer than two axes, so it holds no \
         matrix; step this leaf in another part",
        step ~stacks:(fun _ _ -> 0) (beside (Nx.ones Nx.float32 [| 3 |])) );
      ( "a leaf of four axes without ~stacks",
        "Vega.muon_step: 1: shape [2; 3; 4; 5] has more than two axes; pass \
         ~stacks to say how it stacks its matrices",
        step kernel );
      ( "~stacks below 0",
        "Vega.muon_step: 1: ~stacks gave -1 for shape [2; 3; 4; 5], expected \
         0 to 2",
        step ~stacks:(past_matrices (-1)) kernel );
      ( "~stacks above the rank less two",
        "Vega.muon_step: 1: ~stacks gave 3 for shape [2; 3; 4; 5], expected \
         0 to 2",
        step ~stacks:(past_matrices 3) kernel );
      (* What stops a stack of vectors, of two axes too, under a rule that
         answers for the stacks beside it. *)
      ( "a leaf of two axes under a constant ~stacks of 1",
        "Vega.muon_step: 0: ~stacks gave 1 for shape [2; 3], expected 0: a \
         leaf of two axes is one matrix; step a stack of vectors in another \
         part",
        step ~stacks:(fun _ _ -> 1) (beside matrix) );
      ( "gradients of another skeleton",
        "Vega.muon_step: the root: length 1 in the gradients, length 2 in the \
         parameters",
        fun () ->
          ignore
            (Vega.muon_step lists ~lr (Vega.muon_init lists two) ~params:two
               ~grads:one) );
      ( "a velocity of another skeleton",
        "Vega.muon_step: the root: length 1 in the velocity, length 2 in the \
         parameters",
        fun () ->
          ignore
            (Vega.muon_step lists ~lr (Vega.muon_init lists one) ~params:two
               ~grads:two) );
      ( "a state of no array",
        "Vega.muon_init: the parameters hold no array",
        fun () -> ignore (Vega.muon_init Nx.Ptree.unit ()) );
    ]
  in
  cases "each misuse raises, naming the step and the leaf"
    ~id:(fun (name, _, _) -> name)
    rows
    (fun (_, message, run) -> raises (Invalid_argument message) run)

let () =
  exit
    (run "vega muon"
       [
         group "orth"
           [
             group "float32"
               [ follows_the_svd Nx.float32; not_finite Nx.float32 ];
             group "float64"
               [ follows_the_svd Nx.float64; not_finite Nx.float64 ];
             group "bfloat16" [ follows_the_svd Nx.bfloat16 ];
             group "float16" [ follows_the_svd Nx.float16 ];
             test "a momentum whose norm would overflow steps in full"
               test_overflowing_norm;
           ];
         group "the start"
           [
             test "a zero momentum steps by its decay alone"
               test_zero_momentum;
             test "float32 steps from its least normal float on"
               (least_normal Nx.float32);
             test "float64 steps from its least normal float on"
               (least_normal Nx.float64);
             group "float32" [ scale_free Nx.float32 ];
             group "float64" [ scale_free Nx.float64 ];
           ];
         group "the formula"
           [
             formula;
             test "a state made by sgd_step steps under muon_step"
               test_steps_an_sgd_state;
           ];
         group "leaves"
           [
             group "float32" [ transposes Nx.float32 ];
             group "float64" [ transposes Nx.float64 ];
             group "kernels" (each kernels);
             group "stacks" (each stacks);
             group "stacks of any depth" (each any_stacks);
             test "each rule is asked of each float or complex leaf"
               test_rules_are_asked;
             test "a leaf cut across devices steps where its sums allow"
               test_cut_leaves;
           ];
         group "complex leaves"
           [
             group "complex64"
               [
                 as_real_form Nx.complex64 Nx.float32;
                 turns Nx.complex64 Nx.float32;
               ];
             group "complex128"
               [
                 as_real_form Nx.complex128 Nx.float64;
                 turns Nx.complex128 Nx.float64;
               ];
           ];
         refusals;
       ])
