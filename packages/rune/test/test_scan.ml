(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.scan: the OCaml fold of its step, bit for bit, every NaN read as one,
   eagerly, under forward mode, reverse mode and maps, and compiled where the
   step's functions are exact; traced at the call, so reads and keyless draws in
   a step raise naming the loop. *)

open Windtrap
open Rune_support
open Witness
open Expr
module T = Nx.Ptree
module D = Nx.Dtype
module Total = Rune.Total

let invalid ~sub f = raises_match (Exn.invalid_arg ~sub) f
let f32 = Nx.float32
let v xs = Nx.create f32 [| Array.length xs |] xs
let t = Nx.Ptree.tensor

(* The fold of [step] over the rows of [xs]. *)
let fold step ~init xs =
  let n = (Nx.shape xs).(0) in
  let rec go i c ys =
    if i = n then (c, List.rev ys)
    else
      let c, y = step c (Nx.get [ i ] xs) in
      go (i + 1) c (y :: ys)
  in
  go 0 init []

(* Bits, but for zeros' signs, and with [nan] NaNs the second value has: a
   loop's derivative holds a tangent or cotangent of a carry at every step, a
   zero where a step gives none, whose products are zeros of either sign, and
   NaNs with an infinite partial derivative, where the steps written out have no
   term; and it sums a row's contributions as stacked rows, where the steps
   written out add each row's slice padded with [+0.]. *)
let lenient ~nan =
  let zero x = x = 0L || x = Int64.min_int in
  Testable.make ~pp:(Testable.pp bits_t) ~equal:(fun (s, a) (s', b) ->
      s = s'
      && Array.length a = Array.length b
      && Array.for_all2
           (fun x y ->
             x = y || (nan && y = Int64.minus_one) || (zero x && zero y))
           a b)

(* Host leaves in every view: a carry of shape [s], [n] rows, the step's next
   carry [g] and output [h] of the carry [X] and the row [Y], and tangents of
   the carry and the rows. *)
type case =
  | Case :
      (float, 's) Nx.dtype * int array * int * Operand.t array * Expr.t * Expr.t
      -> case

let cases =
  let open Gen in
  (let* s = Operand.shape in
   let* n = of_list [ 0; 1; 2; 5 ] in
   let rank = Array.length s in
   let* g = gen ~rank and+ h = gen ~rank in
   let rows = Array.append [| n |] s in
   let* c = Operand.draw s and+ xs = Operand.draw rows in
   let* tc = Operand.draw s and+ txs = Operand.draw rows in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any Nx.float64 ] in
   match D.kind dt with
   | Float -> Case (dt, s, n, [| c; xs; tc; txs |], g, h)
   | _ -> assert false)
  |> with_pp (fun ppf (Case (dt, s, n, o, g, h)) ->
      Format.fprintf ppf "%a %a, %d rows, carry %a, xs %a: carry %a, output %a"
        Nx.pp_dtype dt Operand.pp_shape s n Operand.pp_view o.(0).view
        Operand.pp_view o.(1).view pp g pp h)

let covers (Case (_, s, n, o, _, _)) =
  let any p = p o.(0).Operand.view || p o.(1).Operand.view in
  cover "no row" (n = 0);
  cover "several rows" (n > 1);
  cover "a zero-size row" (Operand.numel s = 0);
  cover "a view operand" (any (( <> ) Operand.Plain));
  cover "an offset view"
    (any (function Operand.Offset _ -> true | _ -> false));
  cover "a permuted view" (any (( = ) Operand.Permuted))

(* The case's leaves, the step of [g] and [h], its fold and its scan, each of
   [(init, xs)]. The step is traced at the call, so with no row the fold still
   applies it once and its errors raise. *)
let leaves (type s) (dt : (float, s) Nx.dtype) s n (o : Operand.t array) =
  let rows = Array.append [| n |] s in
  let make k sh = Operand.make dt sh o.(k) in
  ((make 0 s, make 1 rows), (make 2 s, make 3 rows))

let loops (type s d) (dt : (float, s) Nx.dtype) s n g h =
  let step (c : (float, s, d) Nx.t) x =
    (Nx.broadcast_to s (eval c x g), Nx.broadcast_to s (eval c x h))
  in
  let folded (c, xs) =
    if n = 0 then ignore (step c (Nx.zeros_at (Nx.placement c) dt s));
    let c, ys = fold step ~init:c xs in
    let rows = Array.append [| n |] s in
    (c, if n = 0 then Nx.zeros_at (Nx.placement c) dt rows else Nx.stack ys)
  in
  (folded, fun (c, xs) -> Rune.scan' step ~init:c xs)

let pair_bits (a, b) = (bits a, bits b)
let pp2 = Testable.pair bits_t bits_t

let law (Case (dt, s, n, o, g, h) as case) =
  covers case;
  let x, _ = leaves dt s n o in
  let folded, scanned = loops dt s n g h in
  let expected = outcome (fun () -> pair_bits (folded x)) in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  let scanned =
    outcome (fun () ->
        let c, ys = scanned x in
        equal (array int) (Array.append [| n |] s) (Nx.shape ys);
        pair_bits (c, ys))
  in
  equal (outcome_t pp2) expected scanned

(* [f ()], covering that a call of a loop of [rows] rows compiled: no call it
   made was evaluated. *)
let compiled ~rows f =
  let before = Rune.evaluations () in
  let r = f () in
  cover "compiled"
    (rows > 0
    && (match r with Raised _ -> false | _ -> true)
    && Rune.evaluations () = before);
  r

(* Compiled, a scan whose step's functions are exact gives the fold's bits,
   whatever its rows' layout. *)
let law_compiled (Case (dt, s, n, o, g, h) as case) =
  covers case;
  let (c, xs), _ = leaves dt s n o in
  let folded, scanned = loops dt s n (exact g) (exact h) in
  let f = Rune.jit T.(tensor @-> tensor @-> returns (pair tensor tensor)) in
  let staged = f (fun c xs -> scanned (c, xs)) in
  equal (outcome_t pp2)
    (outcome (fun () -> pair_bits (folded (c, xs))))
    (compiled ~rows:n (fun () -> outcome (fun () -> pair_bits (staged c xs))))

(* Derivatives and a map of a scan over host leaves in every view: forward mode,
   reverse mode and two lanes of the carry, against the fold's, eagerly; forward
   mode compiled too, where the step's functions are exact and sum no terms
   along an axis. *)
let law_derived (Case (dt, s, n, o, g, h) as case) =
  covers case;
  let x, tx = leaves dt s n o in
  let p = T.(pair tensor tensor) in
  let folded, scanned = loops dt s n g h in
  let pair_bits' (y, t) = (pair_bits y, pair_bits t) in
  let jvp f = outcome (fun () -> pair_bits' (Rune.jvp p p f x tx)) in
  let expected = jvp folded in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  (* A carry whose step reads nothing keeps a zero tangent where the fold's has
     none. *)
  let l = if constant g then lenient ~nan:true else bits_t in
  let tj = outcome_t (Testable.pair pp2 (Testable.pair l l)) in
  equal ~msg:"jvp" tj expected (jvp scanned);
  let objective f x =
    let c, ys = f x in
    Nx.add (Nx.sum c) (Nx.sum ys)
  in
  let grad f = outcome (fun () -> pair_bits (Rune.grad p (objective f) x)) in
  (* The carry's cotangent is a zero where the step reads no carry. *)
  let l = lenient ~nan:(not (reads_x g || reads_x h)) in
  equal ~msg:"grad" (outcome_t (Testable.pair l l)) (grad folded) (grad scanned);
  let init, xs = x in
  let flat f c =
    let c, ys = f (c, xs) in
    Nx.concatenate ~axis:0 [ Nx.reshape [| -1 |] c; Nx.reshape [| -1 |] ys ]
  in
  let per_lane =
    outcome (fun () ->
        bits (Nx.stack [ flat folded init; flat folded (Nx.neg init) ]))
  in
  equal ~msg:"vmap" (outcome_t bits_t) per_lane
    (outcome (fun () ->
         bits (Rune.vmap' (flat scanned) (Nx.stack [ init; Nx.neg init ]))));
  let folded, scanned =
    loops dt s n (unsummed (exact g)) (unsummed (exact h))
  in
  let staged =
    Rune.jit
      T.(tensor @-> tensor @-> tensor @-> tensor @-> returns (pair p p))
      (fun c xs tc txs -> Rune.jvp p p scanned (c, xs) (tc, txs))
  in
  let (c, xs), (tc, txs) = (x, tx) in
  equal ~msg:"jit of jvp" tj (jvp folded)
    (compiled ~rows:n (fun () ->
         outcome (fun () -> pair_bits' (staged c xs tc txs))))

(* Every dtype, through the functions every dtype has. *)
type any_case =
  | Any_case :
      ('v, 's) Nx.dtype
      * int array
      * int
      * Operand.t
      * Operand.t
      * Expr.t
      * Expr.t
      -> any_case

let any_cases =
  let open Gen in
  (let* s = Operand.shape in
   let* n = of_list [ 0; 1; 3 ] in
   let rank = Array.length s in
   let* g = gen ~rank and+ h = gen ~rank in
   let* c = Operand.draw s and+ xs = Operand.draw (Array.append [| n |] s) in
   let+ (D.Any dt) = of_list D.all in
   Any_case (dt, s, n, c, xs, generic g, generic h))
  |> with_pp (fun ppf (Any_case (dt, s, n, _, _, g, h)) ->
      Format.fprintf ppf "%a %a, %d rows: carry %a, output %a" Nx.pp_dtype dt
        Operand.pp_shape s n pp g pp h)

let law_any (Any_case (dt, s, n, co, xso, g, h)) =
  let rows = Array.append [| n |] s in
  let init = Operand.make dt s co and xs = Operand.make dt rows xso in
  cover "no row" (n = 0);
  cover "a complex dtype" (D.is Complex dt);
  cover "an integer dtype"
    (match D.kind dt with Signed | Unsigned -> true | _ -> false);
  let step c x =
    (Nx.broadcast_to s (eval_any c x g), Nx.broadcast_to s (eval_any c x h))
  in
  let expected =
    outcome (fun () ->
        if n = 0 then ignore (step init (Nx.zeros dt s));
        let c, ys = fold step ~init xs in
        (codes c, List.map codes ys))
  in
  let scanned =
    outcome (fun () ->
        let c, ys = Rune.scan' step ~init xs in
        equal ~msg:"outputs' shape" (array int) rows (Nx.shape ys);
        (codes c, List.init n (fun i -> codes (Nx.get [ i ] ys))))
  in
  equal
    (outcome_t (Testable.pair codes_t (Testable.list codes_t)))
    expected scanned

(* Tracing *)

let xs3 () = Nx.create f32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]

let test_runs_once () =
  List.iter
    (fun n ->
      let calls = ref 0 in
      let step c x =
        incr calls;
        (Nx.add c x, Nx.mul c x)
      in
      ignore (Rune.scan' step ~init:(v [| 0.; 0. |]) (Nx.zeros f32 [| n; 2 |]));
      equal ~msg:(Printf.sprintf "%d rows" n) int 1 !calls)
    [ 0; 1; 7 ]

let test_read () =
  invalid ~sub:"Nx.item: the value is traced by Rune.scan" (fun () ->
      Rune.scan t t Nx.Ptree.unit
        (fun c x -> if Nx.item [ 0 ] x > 0. then (Nx.add c x, ()) else (c, ()))
        ~init:(v [| 0.; 0. |])
        (xs3 ()));
  invalid ~sub:"Nx.item: the value is traced by Rune.iterate" (fun () ->
      Rune.iterate' ~max:3
        ~until:(fun x -> Nx.less x (Nx.scalar f32 (Nx.item [ 0 ] x)))
        Nx.neg (v [| 1. |]));
  (* A captured value a differentiation tracks reads its primal in a step, the
     same at every step, and raises in remat, which stages the read. *)
  let step a c r =
    if Nx.item [] a > 1. then (Nx.add c (Nx.mul r a), c) else (c, c)
  in
  let sum a =
    fst (Rune.scan' (step a) ~init:(Nx.scalar f32 0.) (Nx.ones f32 [| 3 |]))
  in
  let two = Nx.scalar f32 2. and one = Nx.scalar f32 1. in
  equal ~msg:"jvp" float_exact 3. (Nx.item [] (snd (Rune.jvp' sum two one)));
  equal ~msg:"grad" float_exact 3. (Nx.item [] (Rune.grad' sum two));
  let count a =
    Rune.iterate' ~max:4
      ~until:(fun c -> Nx.greater c (Nx.scalar f32 (Nx.item [] a)))
      (fun c -> if Nx.item [] a > 1. then Nx.add c a else c)
      (Nx.scalar f32 0.)
  in
  equal ~msg:"iterate" float_exact 2.
    (Nx.item [] (snd (Rune.jvp' count two one)));
  invalid ~sub:"Nx.item: the value is traced by Rune.jvp'" (fun () ->
      Rune.jvp'
        (fun a ->
          Rune.remat
            Nx.Ptree.(tensor @-> returns tensor)
            (fun x -> if Nx.item [] a > 1. then Nx.mul x a else x)
            one)
        two one)

let test_keyless () =
  invalid ~sub:"no key inside Rune.scan" (fun () ->
      Rune.scan'
        (fun c x -> (Nx.add c (Nx.Rng.uniform f32 [| 2 |]), x))
        ~init:(v [| 0.; 0. |])
        (xs3 ()));
  (* Even with no row, the step is traced. *)
  invalid ~sub:"no key inside Rune.scan" (fun () ->
      Rune.scan'
        (fun c x -> (Nx.add c (Nx.Rng.uniform f32 [| 2 |]), x))
        ~init:(v [| 0.; 0. |])
        (Nx.zeros f32 [| 0; 2 |]));
  invalid ~sub:"no key inside Rune.iterate" (fun () ->
      Rune.iterate' ~max:3
        ~until:(fun x -> Nx.less (Nx.Rng.uniform f32 [||]) x)
        Nx.neg (v [| 1. |]))

(* A step draws from a key in its rows: the draws are the fold's. *)
let test_keyed () =
  let keys = Nx.Rng.split_batch ~n:4 (Nx.Rng.key 3) in
  let step c k =
    let d = Nx.Rng.normal ~key:(Nx.Rng.of_tensor k) f32 [| 2 |] in
    (Nx.add c d, d)
  in
  let init = v [| 0.; 0. |] in
  let c, ys = Rune.scan' step ~init (Nx.Rng.to_tensor keys) in
  let c', ys' = fold step ~init (Nx.Rng.to_tensor keys) in
  equal bits_t (bits c') (bits c);
  equal bits_t (bits (Nx.stack ys')) (bits ys)

let test_carry_refused () =
  invalid
    ~sub:
      "Rune.scan': the root: shape [1] in the carry the step returned, [2] in \
       the carry it received" (fun () ->
      Rune.scan'
        (fun c x -> (Nx.sum ~keepdims:true c, x))
        ~init:(v [| 0.; 0. |])
        (xs3 ()));
  invalid
    ~sub:
      "Rune.scan: the root: length 2 in the carry the step returned, length 1 \
       in the carry it received" (fun () ->
      Rune.scan (Nx.Ptree.list t) t Nx.Ptree.unit
        (fun c x -> (x :: c, ()))
        ~init:[ v [| 0.; 0. |] ]
        (xs3 ()))

let test_xs_refused () =
  invalid ~sub:"Rune.scan: xs has no leaf" (fun () ->
      Rune.scan t Nx.Ptree.unit Nx.Ptree.unit
        (fun c () -> (c, ()))
        ~init:(v [| 0. |]) ());
  invalid ~sub:"Rune.scan': xs: the root is 0-d" (fun () ->
      Rune.scan' (fun c x -> (c, x)) ~init:(v [| 0. |]) (Nx.scalar f32 1.));
  invalid ~sub:"Rune.scan: xs: 1: 3 rows, 0: 2" (fun () ->
      Rune.scan t
        Nx.Ptree.(pair tensor tensor)
        Nx.Ptree.unit
        (fun c _ -> (c, ()))
        ~init:(v [| 0. |])
        (Nx.zeros f32 [| 2 |], Nx.zeros f32 [| 3 |]))

(* A step captures an array, and a scan nested in a step captures the outer
   step's carry: both are the folds'. *)
let test_nested () =
  let w = v [| 0.5; -2. |] in
  let inner c x = (Nx.add (Nx.mul c w) x, c) in
  let outer c x =
    let c', ys =
      Rune.scan' (fun d y -> inner (Nx.add d c) y) ~init:c (Nx.stack [ x; x ])
    in
    (Nx.sub c' (Nx.sum ~axes:[ 0 ] ys), c')
  in
  let outer_fold c x =
    let c', ys =
      fold (fun d y -> inner (Nx.add d c) y) ~init:c (Nx.stack [ x; x ])
    in
    (Nx.sub c' (Nx.sum ~axes:[ 0 ] (Nx.stack ys)), c')
  in
  let init = v [| 1.; -0. |] in
  let c, ys = Rune.scan' outer ~init (xs3 ()) in
  let c', ys' = fold outer_fold ~init (xs3 ()) in
  equal bits_t (bits c') (bits c);
  equal bits_t (bits (Nx.stack ys')) (bits ys)

(* Constructs in steps *)

let test_detach () =
  let x = v [| 1.; -0. |] in
  let c, _ =
    Rune.scan' (fun c x -> (Nx.add (Rune.detach c) x, c)) ~init:x (xs3 ())
  in
  equal bits_t (bits (v [| 10.; 12. |])) (bits c)

(* With no map around, lanes and lane_index raise in a step, whatever it
   computes and with no row. *)
let test_no_map () =
  let a = Rune.axis () in
  let no_lanes = "Rune.lanes: no map named by the axis is around the call" in
  let no_index =
    "Rune.lane_index: no map named by the axis is around the call"
  in
  List.iter
    (fun n ->
      invalid ~sub:no_lanes (fun () ->
          Rune.scan'
            (fun c x -> (c, Rune.lanes a x))
            ~init:(v [| 0.; 0. |])
            (Nx.zeros f32 [| n; 2 |]));
      invalid ~sub:no_index (fun () ->
          Rune.scan'
            (fun c x -> (Nx.add c (Nx.cast f32 (Rune.lane_index a c)), x))
            ~init:(v [| 0.; 0. |])
            (Nx.zeros f32 [| n; 2 |])))
    [ 0; 2 ]

let zero () = Nx.scalar f32 (-0.)

(* Each addition a step makes counts once per step taken, a constant's too,
   chained in step order from the collect's running value. *)
let test_in_loops () =
  let t = Total.make () in
  let xs = xs3 () in
  let step c x =
    Total.add t (Nx.sum x);
    Total.add t (Nx.scalar f32 1.);
    (Nx.add c x, ())
  in
  let scanned () =
    Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit step
      ~init:(v [| 0.; 0. |])
  in
  let folded xs () = ignore (fold step ~init:(v [| 0.; 0. |]) xs) in
  let total f = snd (Total.collect t ~init:(zero ()) f) in
  equal bits_t
    (bits (total (folded xs)))
    (bits (total (fun () -> scanned () xs)));
  (* No row: the total keeps its zero's bits. *)
  equal bits_t
    (bits (zero ()))
    (bits (total (fun () -> scanned () (Nx.zeros f32 [| 0; 2 |]))));
  (* Two additions of a step to one total chain in the order they are made: the
     sum is not the sum of each one's sum. *)
  let big c _ =
    Total.add t (Nx.scalar f32 12.);
    Total.add t (Nx.scalar f32 1e8);
    (c, ())
  in
  let three = Nx.zeros f32 [| 3; 1 |] in
  equal bits_t
    (bits (Nx.scalar f32 300000064.))
    (bits
       (total (fun () ->
            Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit big
              ~init:(v [| 0. |]) three)));
  (* The chain starts from the collect's running total: after 1e8, three 12s are
     added in turn, each rounded, as the fold adds them. *)
  let after =
    total (fun () ->
        Total.add t (Nx.scalar f32 1e8);
        Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
          (fun c _ ->
            Total.add t (Nx.scalar f32 12.);
            (c, ()))
          ~init:(v [| 0. |]) three)
  in
  let twelve = Nx.scalar f32 12. in
  equal bits_t
    (bits
       Nx.(
         add (add (add (add (zero ()) (scalar f32 1e8)) twelve) twelve) twelve))
    (bits after);
  (* Per trip of an iterate, and per test of its stop. *)
  let trips =
    total (fun () ->
        Rune.iterate' ~max:10
          ~until:(fun x ->
            Total.add t (Nx.scalar f32 100.);
            Nx.less (Nx.max x) (Nx.scalar f32 1.))
          (fun x ->
            Total.add t (Nx.scalar f32 1.);
            Nx.mul x (Nx.scalar f32 0.5))
          (v [| 5. |]))
  in
  equal bits_t (bits (Nx.scalar f32 403.)) (bits trips)

(* The pool's values at float32. *)
let pool = Array.map (Operand.value f32) Operand.pool

(* The additions of a scan's steps, to two totals, collected around the scan and
   around the OCaml fold: the same totals, where the zero is 0 or -0. *)
let total_cases =
  let open Gen in
  let value = map (fun i -> pool.(i)) (int_range 0 (Array.length pool - 1)) in
  let adds = list ~size:(int_range 0 3) (pair bool value) in
  triple (of_list [ 0; 1; 3 ]) (of_list [ 0.; -0. ]) adds
  |> with_pp (fun ppf (n, z, adds) ->
      Format.fprintf ppf "%d rows, zero %h: %s" n z
        (String.concat " "
           (List.map
              (fun (u, c) -> Printf.sprintf "%s%h" (if u then "u" else "t") c)
              adds)))

let law_totals (n, z, adds) =
  cover "no row" (n = 0);
  cover "two additions to one total" (List.length adds >= 2);
  cover "an addition before the loop" (List.exists fst adds);
  let t = Total.make () and u = Total.make () in
  let step c x =
    List.iter
      (fun (to_u, k) ->
        Total.add (if to_u then u else t) (Nx.mul (Nx.sum x) (Nx.scalar f32 k)))
      adds;
    (Nx.add c x, ())
  in
  let xs =
    Nx.create f32 [| n; 2 |]
      (Array.init (2 * n) (fun i -> pool.(i mod Array.length pool)))
  in
  let init = v [| 0.; 0. |] in
  let totals f =
    let ((), a), b =
      Total.collect t ~init:(Nx.scalar f32 z) (fun () ->
          Total.collect u ~init:(Nx.scalar f32 z) f)
    in
    (bits a, bits b)
  in
  (* [u]'s additions are made once before the loop too, so its chain starts from
     a running total that is not its zero. *)
  let before () =
    List.iter
      (fun (to_u, k) -> if to_u then Total.add u (Nx.scalar f32 (k *. 1e7)))
      adds
  in
  let eager =
    totals (fun () ->
        before ();
        ignore (fold step ~init xs))
  in
  let scanned =
    totals (fun () ->
        before ();
        ignore
          (Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit step ~init xs))
  in
  equal (pair bits_t bits_t) eager scanned

(* A scope inside a step sums the step's own additions. *)
let test_scopes_inside () =
  let t = Total.make () in
  let step c x =
    let (), s =
      Total.collect t ~init:(Nx.scalar f32 0.) (fun () ->
          Total.add t (Nx.sum x))
    in
    (Nx.add c s, s)
  in
  let c, ys =
    Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.tensor step
      ~init:(Nx.scalar f32 0.) (xs3 ())
  in
  equal bits_t (bits (Nx.scalar f32 21.)) (bits c);
  equal bits_t (bits (v [| 3.; 7.; 11. |])) (bits ys)

(* Under transformations *)

module Two = (val Nx.devices [ Nx_support.memory 0; Nx_support.memory 1 ])

type where = Whole | Dev of int

let pp_where ppf = function
  | Whole -> Format.pp_print_string ppf "both devices"
  | Dev k -> Format.fprintf ppf "device %d" k

(* A leaf of shape [s]: a host operand's values placed on a device, or whole on
   both a formula, [a + i b] at the element [i] in C order, [a] and [b] the
   operand's first values. *)
let put (type s) w (dt : (float, s) Nx.dtype) shape o : (float, s, Two.d) Nx.t =
  match w with
  | Dev k ->
      Nx.place
        (Nx.Placement.device Two.v (Nx_support.memory k))
        (Operand.make dt shape o)
  | Whole ->
      let at j =
        if j < Array.length o.values then Operand.value dt o.values.(j) else 1.
      in
      let here = Nx.Placement.on Two.v in
      let n = Float.of_int (Operand.numel shape) in
      let i = Nx.reshape shape (Nx.arange_f_at here dt 0. n 1.) in
      Nx.add (Nx.full_at here dt shape (at 0)) (Nx.mul_s i (at 1))

(* A law's draw: dtype, row shape, rows, the step's three expressions, the
   operands of [c0], [c1], [xs], [w] and their tangents, and where each leaf
   lies. *)
type rule =
  | Case : {
      dt : (float, 's) Nx.dtype;
      s : int array;
      n : int;
      g : Expr.t;
      k : Expr.t;
      h : Expr.t;
      ops : Operand.t array;
      tans : Operand.t array;
      at : where array;  (** Of [c0], [c1], [xs], [w], then their tangents. *)
      free : bool;  (** Whether [c1'] reads [c0] only where [k] reads [Y]. *)
    }
      -> rule

let pp_case ppf (Case c) =
  Format.fprintf ppf "%s; tangents %a, %a, %a; "
    (if c.free then "c1' as k" else "c1' + c0 w")
    pp_where c.at.(4) pp_where c.at.(5) pp_where c.at.(6);
  Format.fprintf ppf
    "%a %a, %d rows: c0' %a, c1' %a, y %a; c0 %a, c1 %a, xs %a, w %a"
    Nx.pp_dtype c.dt Operand.pp_shape c.s c.n pp c.g pp c.k pp c.h pp_where
    c.at.(0) pp_where c.at.(1) pp_where c.at.(2) pp_where c.at.(3)

let rule_cases =
  let open Gen in
  (let* s = Operand.shape in
   let* n = of_list [ 0; 1; 2; 5 ] in
   let rank = Array.length s in
   let* g = gen ~rank and+ k = gen ~rank and+ h = gen ~rank in
   let rows = Array.append [| n |] s in
   let* c0 = Operand.draw s
   and+ c1 = Operand.draw s
   and+ xs = Operand.draw rows in
   let* w = Operand.draw s in
   let* t0 = Operand.draw s
   and+ tx = Operand.draw rows
   and+ tw = Operand.draw s in
   (* One device of the set's two per case: values on two devices meet only
      where the program places one beside the other. *)
   let* d = int_range 0 1 in
   let where = frequency [ (1, constant Whole); (1, constant (Dev d)) ] in
   let* at = array ~size:(constant 7) where in
   let* free = bool in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any Nx.float64 ] in
   match D.kind dt with
   | Float ->
       Case
         {
           dt;
           s;
           n;
           g;
           k;
           h;
           free;
           ops = [| c0; c1; xs; w |];
           tans = [| t0; tx; tw |];
           at;
         }
   | _ -> assert false)
  |> with_pp pp_case

(* The law's functions of [(c0, xs, w)]: the scan, and the fold of the same step
   from the carry [(c0, c1)], each returning the final carry and the outputs.
   [c1'] reads [c0], so that [c1] gains what [c0] has after a step. [nested]
   scans the step over its row twice inside the step, summing the outputs. *)
let functions (type s) ?(nested = false) ~free (dt : (float, s) Nx.dtype) s n g
    k h (c1 : (float, s, Two.d) Nx.t) =
  let b = Nx.broadcast_to s in
  let step w (c0, c1) x =
    let c0' = b (eval c0 x g) in
    let c0w = Nx.mul c0 w in
    let c1' =
      if free then b (eval c1 c0w k) else b (Nx.add (eval c1 c0w k) c0w)
    in
    ((c0', c1'), b (eval c1 x h))
  in
  let cs = T.(pair tensor tensor) in
  let inner ~scan w c x =
    if not nested then step w c x
    else
      let two = Nx.stack [ x; x ] in
      let c, ys =
        if scan then Rune.scan cs T.tensor T.tensor (step w) ~init:c two
        else
          (* A copy of [w] per step, whose cotangent sums the inner steps'
             before the outer loop adds it, in the order rune.mli states for
             the scan. *)
          let w = Nx.copy w in
          let c, ys = fold (step w) ~init:c two in
          (c, Nx.stack ys)
      in
      (c, Nx.sum ~axes:[ 0 ] ys)
  in
  let scanned (c0, (xs, w)) =
    let (c0, c1), ys =
      Rune.scan cs T.tensor T.tensor (inner ~scan:true w) ~init:(c0, c1) xs
    in
    (c0, (c1, ys))
  in
  let step w = inner ~scan:false w in
  let folded c1 (c0, (xs, w)) =
    (* The step is traced at the call, so with no row its errors raise. *)
    let here = Nx.Placement.without_leading_axis (Nx.placement xs) in
    ignore (step w (c0, c1) (Nx.zeros_at here dt s));
    let (c0, c1), ys = fold (step w) ~init:(c0, c1) xs in
    let rows = Array.append [| n |] s in
    let ys =
      if n = 0 then Nx.zeros_at (Nx.placement xs) dt rows
      else Nx.stack ~axis:0 ys
    in
    (c0, (c1, ys))
  in
  (scanned, folded c1, folded)

(* Outcomes, where two failures of operands on two devices are alike whichever
   operation meets them first: a backward pass orders its operations otherwise
   than the fold's. *)
let mixed m =
  let key = "place one beside the other" in
  let n = String.length key and l = String.length m in
  let rec go i = i + n <= l && (String.sub m i n = key || go (i + 1)) in
  go 0

let outcome_t' t =
  Testable.make
    ~pp:(Testable.pp (outcome_t t))
    ~equal:(fun a b ->
      match (a, b) with
      | Raised a, Raised b -> String.equal a b || (mixed a && mixed b)
      | _ -> Testable.equal (outcome_t t) a b)

let triple () = T.(pair tensor (pair tensor tensor))
let t3 = Testable.triple bits_t bits_t bits_t
let bits3 (a, (b, c)) = (bits a, bits b, bits c)

(* The leaves [c0], [c1], [xs], [w] of a case, where it puts them. *)
let operands (type s) (dt : (float, s) Nx.dtype) s n ops at :
    (float, s, Two.d) Nx.t array =
  let rows = Array.append [| n |] s in
  [|
    put at.(0) dt s ops.(0);
    put at.(1) dt s ops.(1);
    put at.(2) dt rows ops.(2);
    put at.(3) dt s ops.(3);
  |]

(* Whether [c1] gains what [c0] has after a step. *)
let gains (Case c) = (not c.free) || reads_y c.k

(* Whether a step leaves [c0] or [c1] unread: the reversed loop's carry then
   holds a zero cotangent where the fold has none. *)
let unread (Case c) =
  (not (reads_x c.k || reads_x c.h))
  || (c.free && not (reads_x c.g || reads_y c.k))

let rule_covers (Case c) =
  cover "no row" (c.n = 0);
  cover "several rows" (c.n > 1);
  cover "a placed leaf" (Array.exists (fun w -> w <> Whole) c.at);
  cover "the second device" (Array.exists (fun w -> w = Dev 1) c.at);
  cover "a leaf on both devices" (Array.exists (fun w -> w = Whole) c.at)

(* A compiled loop gives the fold's bits where its step's functions are exact
   (Rune.jit's classes): the laws compare compiled calls on the case's functions
   made exact, in one case of three. A compiled case builds and compiles two
   programs, most of a law's time without a warm cache. About two thirds of
   them compile whole, so at 100 cases a law reaches "compiled" in about 23 of
   them, and misses it with odds of (1 - 2/9)^100, below 1e-10. *)
let compiled_too =
  Gen.frequency [ (2, Gen.constant false); (1, Gen.constant true) ]

let exactly (Case c) =
  let x e = unsummed (exact e) in
  Case { c with g = x c.g; k = x c.k; h = x c.h; at = Array.make 7 (Dev 0) }

let law_jvp' ~jit (Case c as case) =
  rule_covers case;
  let o = operands c.dt c.s c.n c.ops c.at in
  let c0 = o.(0) and c1 = o.(1) and xs = o.(2) and w = o.(3) in
  let rows = Array.append [| c.n |] c.s in
  let tan k sh = put c.at.(4 + k) c.dt sh c.tans.(k) in
  let t = (tan 0 c.s, (tan 1 rows, tan 2 c.s)) in
  cover "a placed tangent" (Array.exists (( <> ) Whole) (Array.sub c.at 4 3));
  let scanned, folded, from =
    functions ~free:c.free c.dt c.s c.n c.g c.k c.h c1
  in
  let jvp f = Rune.jvp (triple ()) (triple ()) f (c0, (xs, w)) t in
  let both (y, t) = (bits3 y, bits3 t) in
  (* Where [c1] gains a tangent at the first step, the scan's starts from a
     zero: the fold's is a zero from the start. *)
  cover "a carry with no tangent" (not (gains case));
  let expected =
    let p = T.(pair tensor (pair tensor (pair tensor tensor))) in
    outcome (fun () ->
        if not (gains case) then both (jvp folded)
        else
          let y, (_, t) =
            Rune.jvp p p
              (fun (c1, x) -> (c1, from c1 x))
              (c1, (c0, (xs, w)))
              (Nx.full_at (Nx.placement c1) c.dt c.s (-0.), t)
          in
          both (snd y, t))
  in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  (* A carry whose step reads nothing keeps a zero tangent where the fold's has
     none. *)
  let l = lenient ~nan:(constant c.g) in
  let l = if constant c.g then l else bits_t in
  let tt = outcome_t' (Testable.pair t3 (Testable.triple l l l)) in
  equal ~msg:"eager" tt expected (outcome (fun () -> both (jvp scanned)));
  if jit then begin
    let a3 =
      T.(
        tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor
        @-> returns
              (pair
                 (pair tensor (pair tensor tensor))
                 (pair tensor (pair tensor tensor))))
    in
    let staged =
      Rune.jit a3 (fun c0 xs w t0 tx tw ->
          Rune.jvp (triple ()) (triple ()) scanned (c0, (xs, w)) (t0, (tx, tw)))
    in
    let t0, (tx, tw) = t in
    equal ~msg:"jit of jvp" tt expected
      (compiled ~rows:c.n (fun () ->
           outcome (fun () -> both (staged c0 xs w t0 tx tw))));
    let jitted =
      Rune.jit
        T.(
          tensor @-> tensor @-> tensor
          @-> returns (pair tensor (pair tensor tensor)))
        (fun c0 xs w -> scanned (c0, (xs, w)))
    in
    equal ~msg:"jvp of jit" tt expected
      (compiled ~rows:c.n (fun () ->
           outcome (fun () ->
               both
                 (Rune.jvp (triple ()) (triple ())
                    (fun (a, (b, c)) -> jitted a b c)
                    (c0, (xs, w))
                    t))))
  end

let law_jvp (case, compiled) =
  law_jvp' ~jit:false case;
  if compiled then law_jvp' ~jit:true (exactly case)

let law_grad' ~jit (Case c as case) =
  rule_covers case;
  let o = operands c.dt c.s c.n c.ops c.at in
  let c0 = o.(0) and c1 = o.(1) and xs = o.(2) and w = o.(3) in
  let scanned, folded, _ = functions ~free:c.free c.dt c.s c.n c.g c.k c.h c1 in
  let objective f p =
    let a, (b, y) = f p in
    Nx.add (Nx.add (Nx.sum a) (Nx.sum b)) (Nx.sum y)
  in
  let grad f =
    outcome (fun () ->
        bits3 (Rune.grad (triple ()) (objective f) (c0, (xs, w))))
  in
  let expected = grad folded in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  (* [c1]'s cotangent is a zero where the step reads no [c1]. *)
  let unread = unread case in
  let l = lenient ~nan:unread in
  let tt = outcome_t' (Testable.triple l l l) in
  equal ~msg:"eager" tt expected (grad scanned);
  if jit then begin
    let staged =
      Rune.jit
        T.(
          tensor @-> tensor @-> tensor
          @-> returns (pair tensor (pair tensor tensor)))
        (fun c0 xs w -> Rune.grad (triple ()) (objective scanned) (c0, (xs, w)))
    in
    equal ~msg:"jit of grad" tt expected
      (compiled ~rows:c.n (fun () -> outcome (fun () -> bits3 (staged c0 xs w))));
    let jitted =
      Rune.jit
        T.(
          tensor @-> tensor @-> tensor
          @-> returns (pair tensor (pair tensor tensor)))
        (fun c0 xs w -> scanned (c0, (xs, w)))
    in
    equal ~msg:"grad of jit" tt expected
      (compiled ~rows:c.n (fun () -> grad (fun (a, (b, c)) -> jitted a b c)))
  end

let law_grad (case, compiled) =
  law_grad' ~jit:false case;
  if compiled then law_grad' ~jit:true (exactly case)

(* A case whose compiled transpose tolk declines to link, so that the call
   evaluates over its arguments, a cotangent of every set among them. *)
let grad_examples =
  let plain values =
    { Operand.view = Plain; values = Array.map (fun v -> Operand.F v) values }
  in
  [
    Case
      {
        dt = Nx.float32;
        s = [| 2 |];
        n = 1;
        g = Y;
        k = Un (Neg, Y);
        h =
          Un
            ( Cumsum 0,
              Bin
                ( Add,
                  Bin
                    ( Add,
                      Y,
                      Bin (Add, Bin (Mul, X, K (F 0.)), Bin (Add, Y, K (F 0.)))
                    ),
                  Un (Cumsum 0, Un (Neg, Bin (Add, Y, X))) ) );
        ops =
          [|
            plain [| 0.5; -1. |];
            plain [| 2.; 1. |];
            plain [| 0.25; 3. |];
            plain [| 1.5; -2. |];
          |];
        tans = [| plain [| 1.; 1. |]; plain [| 1.; 1. |]; plain [| 1.; 1. |] |];
        at = Array.make 7 (Dev 0);
        free = false;
      };
  ]

(* Maps: one of [c0], [xs] and [w] has lanes, the others are captured. *)
type mapped = Carry | Rows | Capture

let map_cases =
  Gen.(
    pair rule_cases (pair (of_list [ Carry; Rows; Capture ]) (of_list [ 1; 3 ])))
  |> Gen.with_pp (fun ppf (c, (m, l)) ->
      Format.fprintf ppf "%a, %d lanes of the %s" pp_case c l
        (match m with
        | Carry -> "carry"
        | Rows -> "rows"
        | Capture -> "capture"))

let law_vmap' ~jit ((Case c as case), (m, lanes)) =
  rule_covers case;
  cover "lanes in the carry" (m = Carry);
  let o = operands c.dt c.s c.n c.ops c.at in
  let c0 = o.(0) and c1 = o.(1) and xs = o.(2) and w = o.(3) in
  let scanned, folded, _ = functions ~free:c.free c.dt c.s c.n c.g c.k c.h c1 in
  let flat f p =
    let a, (b, y) = f p in
    Nx.concatenate ~axis:0
      [ Nx.reshape [| -1 |] a; Nx.reshape [| -1 |] b; Nx.reshape [| -1 |] y ]
  in
  (* Lane [l] scales the mapped operand by [l + 1]. *)
  let pick = function Carry -> c0 | Rows -> xs | Capture -> w in
  let x = pick m in
  let batch =
    Nx.stack ~axis:0
      (List.init lanes (fun l -> Nx.mul_s x (Float.of_int (l + 1))))
  in
  let at f v =
    match m with
    | Carry -> flat f (v, (xs, w))
    | Rows -> flat f (c0, (v, w))
    | Capture -> flat f (c0, (xs, v))
  in
  let one_by_one f =
    Nx.stack ~axis:0 (List.init lanes (fun l -> at f (Nx.get [ l ] batch)))
  in
  let expected = outcome (fun () -> bits (one_by_one folded)) in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  let tt = outcome_t' bits_t in
  equal ~msg:"eager" tt expected
    (outcome (fun () -> bits (Rune.vmap' (at scanned) batch)));
  if jit then begin
    equal ~msg:"jit of vmap" tt expected
      (compiled ~rows:c.n (fun () ->
           outcome (fun () -> bits (Rune.jit' (Rune.vmap' (at scanned)) batch))));
    equal ~msg:"vmap of jit" tt expected
      (compiled ~rows:c.n (fun () ->
           outcome (fun () -> bits (Rune.vmap' (Rune.jit' (at scanned)) batch))))
  end

let law_vmap ((case, m), compiled) =
  law_vmap' ~jit:false (case, m);
  if compiled then law_vmap' ~jit:true (exactly case, m)

(* Compositions: a scan whose step scans, under jvp, grad and a map, and second
   orders through a scan. Their zeros are a loop's wherever a step reads no
   carry, so values compare leniently. *)
(* [v] within [-2, 2], a NaN as [0.5]. *)
let moderate : Operand.value -> float = function
  | Least -> -2.
  | Greatest -> 2.
  | F v | C (v, _) ->
      if Float.is_nan v then 0.5 else Float.min 2. (Float.max (-2.) v)

(* [e] with every partial derivative finite at finite operands: exp, sqrt, log,
   reciprocal, division and power replaced by smooth functions, a constant
   within [-2, 2]. *)
let rec smoothed : Expr.t -> Expr.t = function
  | (X | Y) as e -> e
  | K v -> K (F (moderate v))
  | Un (Exp, e) -> Un (Sin, smoothed e)
  | Un (Sqrt, e) -> Un (Tanh, smoothed e)
  | Un ((Log | Recip), e) -> Un (Neg, smoothed e)
  | Un (u, e) -> Un (u, smoothed e)
  | Bin ((Div | Pow), e, e') -> Bin (Mul, smoothed e, smoothed e')
  | Bin (b, e, e') -> Bin (b, smoothed e, smoothed e')

(* An operand's values within [-2, 2], a NaN as [0.5]. *)
let moderated o =
  {
    o with
    Operand.values =
      Array.map (fun v -> Operand.F (moderate v)) o.Operand.values;
  }

(* The majorant of a {!smoothed} [e]: over operands [t >= |x|], its value and
   its partial derivatives of the first two orders are non-negative and at least
   the sums of magnitudes of the terms rune's rules compute [e]'s with at [x].
   Negation and [abs] keep their operand, a difference and the selections become
   sums, and a running maximum a sum. Sin, whose rules read cos and sin, each at
   most 1, becomes [1 + t + t^2 / 2]. Tanh's tangent is [(t + t y) (1 - y)],
   terms of at most [4 |t|], its second order at most [16 |t|], and sigmoid's
   composition is within the same: both become [1 + 4 t + 8 t^2]. *)
let quadratic a b t =
  Bin
    ( Add,
      K (F 1.),
      Bin (Add, Bin (Mul, K (F a), t), Bin (Mul, K (F b), Un (Square, t))) )

let rec majorant : Expr.t -> Expr.t = function
  | (X | Y) as e -> e
  | K v -> K (F (Float.abs (moderate v)))
  | Un ((Neg | Abs), e) -> majorant e
  | Un (Sin, e) -> quadratic 1. 0.5 (majorant e)
  | Un ((Tanh | Sigmoid), e) -> quadratic 4. 8. (majorant e)
  | Un (Max a, e) -> Un (Sum a, majorant e)
  | Un (((Square | Cumsum _ | Flip _ | Sum _ | Mean _) as u), e) ->
      Un (u, majorant e)
  | Bin ((Sub | Maximum | Minimum | Where_less | Add), e, e') ->
      Bin (Add, majorant e, majorant e')
  | Bin (Mul, e, e') -> Bin (Mul, majorant e, majorant e')
  | Un ((Exp | Sqrt | Log | Recip), _) | Bin ((Div | Pow), _, _) ->
      invalid_arg "majorant: not smoothed"

let law_compose (Case c as case) =
  rule_covers case;
  let o = operands c.dt c.s c.n c.ops c.at in
  let c0 = o.(0) and c1 = o.(1) and xs = o.(2) and w = o.(3) in
  let x = (c0, (xs, w)) in
  let rows = Array.append [| c.n |] c.s in
  let t =
    ( put Whole c.dt c.s c.tans.(0),
      (put Whole c.dt rows c.tans.(1), put Whole c.dt c.s c.tans.(2)) )
  in
  let p = triple () in
  let objective f p =
    let a, (b, y) = f p in
    Nx.add (Nx.add (Nx.sum a) (Nx.sum b)) (Nx.sum y)
  in
  let check msg cmp (scanned, folded) g =
    equal ~msg
      (outcome_t' (Testable.triple cmp cmp cmp))
      (outcome (fun () -> bits3 (g folded)))
      (outcome (fun () -> bits3 (g scanned)))
  in
  let nested_s, nested_f, nested_from =
    functions ~nested:true ~free:c.free c.dt c.s c.n c.g c.k c.h c1
  in
  let nested = (nested_s, nested_f) in
  let ps, pf, _ = functions ~free:c.free c.dt c.s c.n c.g c.k c.h c1 in
  let plain = (ps, pf) in
  (* jvp of a scan whose step scans, against its fold's from a zero tangent of
     [c1] where it gains one, as [law_jvp]. *)
  let jl = if constant c.g then lenient ~nan:true else bits_t in
  let reference =
    outcome (fun () ->
        if not (gains case) then bits3 (snd (Rune.jvp p p nested_f x t))
        else
          let q = T.(pair tensor (pair tensor (pair tensor tensor))) in
          let _, (_, t) =
            Rune.jvp q q
              (fun (c1, x) -> (c1, nested_from c1 x))
              (c1, x)
              (Nx.full_at (Nx.placement c1) c.dt c.s (-0.), t)
          in
          bits3 t)
  in
  equal ~msg:"jvp of a nested scan"
    (outcome_t' (Testable.triple jl jl jl))
    reference
    (outcome (fun () -> bits3 (snd (Rune.jvp p p nested_s x t))));
  let l = lenient ~nan:(unread case) in
  check "grad of a nested scan" l nested (fun f -> Rune.grad p (objective f) x);
  let lanes f =
    let batch = Nx.stack [ c0; Nx.neg c0 ] in
    let y =
      Rune.vmap'
        (fun c0 ->
          let a, (b, y) = f (c0, (xs, w)) in
          Nx.concatenate ~axis:0
            [
              Nx.reshape [| -1 |] a;
              Nx.reshape [| -1 |] b;
              Nx.reshape [| -1 |] y;
            ])
        batch
    in
    (y, (y, y))
  in
  check "vmap of a nested scan" bits_t nested lanes;
  let sq f x =
    let a, (b, y) = Rune.grad (triple ()) (objective f) x in
    Nx.add
      (Nx.add (Nx.sum (Nx.mul a a)) (Nx.sum (Nx.mul b b)))
      (Nx.sum (Nx.mul y y))
  in
  (* Second orders over the case's functions smoothed and operands within [-2,
     2], at float64. The scan and the fold compute the same forward values and
     first-order cotangents, then a second derivation linear in them whose sums
     they associate differently: the loops' captures, and reductions nx
     associates by the shape it reads. Such a value is a sum of terms, each a
     product of factors both compute alike, and each within γ(R) of exact
     barring underflow, R the roundings on its path; so the two are within
     [2γ(R) Σ|terms|], and [(1 + γ(R))] more for the factors' own roundings. The
     same derivation over {!majorant} functions and the operands' and tangents'
     magnitudes gives at least Σ|terms| per element. R counts the roundings of a
     path: through the derivation's four passes, each of [n + 2] steps of 3
     operations per expression node and per term of a reduction, broadcast or
     sum. At float32 the majorant's growth through a carry leaves the bound past
     the values, so second orders run at float64 only. *)
  let f64 = Nx.float64 in
  let o2 = operands f64 c.s c.n (Array.map moderated c.ops) c.at in
  let x2 = (o2.(0), (o2.(2), o2.(3))) in
  let t2 =
    let m k sh = put Whole f64 sh (moderated c.tans.(k)) in
    (m 0 c.s, (m 1 rows, m 2 c.s))
  in
  let s2, f2, _ =
    functions ~free:c.free f64 c.s c.n (smoothed c.g) (smoothed c.k)
      (smoothed c.h) o2.(1)
  in
  let mag x = Nx.abs x in
  let xm = (mag o2.(0), (mag o2.(2), mag o2.(3))) in
  let tm = (mag (fst t2), (mag (fst (snd t2)), mag (snd (snd t2)))) in
  let _, fm, _ =
    let m e = majorant (smoothed e) in
    functions ~free:c.free f64 c.s c.n (m c.g) (m c.k) (m c.h) (mag o2.(1))
  in
  let roundings =
    let rec count (e : Expr.t) =
      match e with
      | X | Y | K _ -> (0, 0)
      | Un (u, e) ->
          let n, r = count e in
          let r' =
            match u with Cumsum _ | Sum _ | Max _ | Mean _ -> 1 | _ -> 0
          in
          (n + 1, r + r')
      | Bin (_, e, e') ->
          let n, r = count e and n', r' = count e' in
          (n + n' + 1, r + r')
    in
    let nodes, reductions =
      List.fold_left
        (fun (n, r) e ->
          let n', r' = count e in
          (n + n', r + r'))
        (6, 4) [ c.g; c.k; c.h ]
    in
    let terms = max 2 (max c.n (Operand.numel c.s)) in
    Float.of_int (4 * (c.n + 2) * 3 * (nodes + (reductions * terms)))
  in
  let gamma =
    let ru = roundings *. 0x1p-53 in
    if ru >= 1. then Float.infinity else ru /. (1. -. ru)
  in
  (* Within [2γ(R)] of the bound [(a', b', c')]'s value at each element. *)
  let summed (a', b', c') =
    let near (s, x) (s', y) (_, m) =
      let ok = ref (s = s') in
      Array.iteri
        (fun i x ->
          let x = Int64.float_of_bits x and y = Int64.float_of_bits y.(i) in
          let m = Int64.float_of_bits m.(i) in
          let tol = 2. *. gamma *. (1. +. gamma) *. m in
          ok := !ok && (x = y || Float.abs (x -. y) <= tol))
        x;
      !ok
    in
    Testable.make ~pp:(Testable.pp t3) ~equal:(fun (a, b, c) (a2, b2, c2) ->
        near a a2 a' && near b b2 b' && near c c2 c')
  in
  (* Where the fold's second derivative is finite: an infinity there meets a
     zero the loop holds, as at first order. *)
  let finite (_, a) =
    Array.for_all (fun x -> Float.is_finite (Int64.float_of_bits x)) a
  in
  (* The bound holds barring underflow, which reaches values the law cannot see:
     a case is left out where one it sees, the fold's outputs, first-order
     cotangents or second orders, lies below [2^-900] but for a zero. *)
  let clear (_, a) =
    Array.for_all
      (fun x ->
        let v = Float.abs (Int64.float_of_bits x) in
        v = 0. || v >= 0x1p-900)
      a
  in
  let seen =
    outcome (fun () ->
        let a, (b, c) = f2 x2
        and d, (e, f) = Rune.grad (triple ()) (objective f2) x2 in
        List.map bits [ a; b; c; d; e; f ])
  in
  let near_underflow =
    match seen with
    | Value vs -> not (List.for_all clear vs)
    | Raised _ -> false
  in
  (* The second orders reach the functions whose second derivatives take either
     sign, and negative values and [-0.] among the operands. *)
  let reaches msg =
    let rec has p (e : Expr.t) =
      p e
      ||
      match e with
      | X | Y | K _ -> false
      | Un (_, e) -> has p e
      | Bin (_, e, e') -> has p e || has p e'
    in
    let fs = [ smoothed c.g; smoothed c.k; smoothed c.h ] in
    let through label u =
      cover
        (msg ^ " through " ^ label)
        (List.exists (has (function Un (v, _) -> v = u | _ -> false)) fs)
    in
    through "sin" Sin;
    through "tanh" Tanh;
    through "sigmoid" Sigmoid;
    through "abs" Abs;
    let leaves =
      let a, (b, c') = x2 and t, (t', t'') = t2 in
      List.concat_map
        (fun x -> Array.to_list (Array.map Int64.float_of_bits (snd (bits x))))
        [ a; b; c'; o2.(1); t; t'; t'' ]
    in
    cover
      (msg ^ " of a negative operand")
      (List.exists (fun v -> v < 0.) leaves);
    cover
      (msg ^ " of a -0. operand")
      (List.exists (fun v -> v = 0. && Float.sign_bit v) leaves)
  in
  (* [second msg fold scan bound] compares [scan] with [fold] within [bound]'s
     per element, where both [fold] and [bound] are finite. *)
  let second msg fold scan bound =
    let expected = outcome (fun () -> bits3 (fold ())) in
    match (expected, outcome (fun () -> bits3 (bound ()))) with
    | Value ((a, b, c) as e), Value ((a', b', c') as m)
      when List.for_all finite [ a; b; c; a'; b'; c' ]
           && List.for_all clear [ a; b; c ]
           && not near_underflow ->
        cover ("a bounded " ^ msg) true;
        reaches msg;
        equal ~msg
          (outcome_t' (summed m))
          (Value e)
          (outcome (fun () -> bits3 (scan ())))
    | _ -> cover ("a bounded " ^ msg) false
  in
  let gg f x () = Rune.grad (triple ()) (sq f) x in
  second "grad of grad" (gg f2 x2) (gg s2 x2) (gg fm xm);
  let jg f x t () =
    let q = triple () in
    snd (Rune.jvp q q (Rune.grad q (objective f)) x t)
  in
  second "jvp of grad" (jg f2 x2 t2) (jg s2 x2 t2) (jg fm xm tm);
  let per_lane f =
    let batch = Nx.stack [ c0; Nx.mul_s c0 2. ] in
    let g =
      Rune.vmap' (fun c0 -> fst (Rune.grad p (objective f) (c0, (xs, w)))) batch
    in
    (g, (g, g))
  in
  check "vmap of grad" l plain per_lane

(* Compiled loops at the edges a program's loops meet *)

let on0 x = Nx.place (Nx.Placement.device Two.v (Nx_support.memory 0)) x

(* [f x] compiled, with no call evaluated, against [f x] eagerly. *)
let compiles ~msg f x =
  let expected = bits (f x) in
  let before = Rune.evaluations () in
  let got = bits (Rune.jit' f x) in
  equal ~msg:(msg ^ ": no call evaluated") int before (Rune.evaluations ());
  equal ~msg bits_t expected got

(* A scan of [no] rows whose step scans [ni] rows, the inner step reading the
   outer carry. Small dyadic values keep every sum exact in any order. *)
let nested xs =
  let outer c r =
    let inner d q = (Nx.add (Nx.add d (Nx.mul_s c 0.5)) q, Nx.neg d) in
    let c', ys = Rune.scan' inner ~init:c r in
    (c', Nx.sum ~axes:[ 0 ] ys)
  in
  let init = Nx.mul_s (Nx.get [ 0; 0 ] xs) 0.5 in
  let c, ys = Rune.scan' outer ~init xs in
  Nx.concatenate ~axis:0 [ c; Nx.reshape [| -1 |] ys ]

let test_nested_compiled () =
  List.iter
    (fun (no, ni) ->
      let msg = Printf.sprintf "%d rows of %d" no ni in
      let k = no * ni * 2 in
      let xs =
        on0
          (Nx.create f32 [| no; ni; 2 |]
             (Array.init k (fun i -> [| 0.5; -1.; 2.; 1. |].(i mod 4))))
      in
      compiles ~msg nested xs;
      compiles ~msg:(msg ^ ", grad")
        (Rune.grad' (fun x -> Nx.sum (nested x)))
        xs;
      equal ~msg:(msg ^ ", grad of jit") bits_t
        (bits (Rune.grad' (fun x -> Nx.sum (nested x)) xs))
        (bits (Rune.grad' (fun x -> Nx.sum (Rune.jit' nested x)) xs));
      compiles ~msg:(msg ^ ", vmap") (Rune.vmap' nested)
        (Nx.stack [ xs; Nx.neg xs ]))
    [ (1, 1); (1, 3); (3, 1); (3, 4) ]

(* Three carries that rotate, each step adding its row to the one that wraps. *)
let test_cycle () =
  let ts = T.(pair tensor (pair tensor tensor)) in
  let f xs =
    let a, (b, c) =
      fst
        (Rune.scan ts T.tensor T.unit
           (fun (a, (b, c)) r -> ((b, (c, Nx.add a r)), ()))
           ~init:(Nx.get [ 0 ] xs, (Nx.get [ 1 ] xs, Nx.get [ 2 ] xs))
           xs)
    in
    Nx.concatenate ~axis:0 [ a; b; c ]
  in
  compiles ~msg:"rotated" f
    (on0 (Nx.create f32 [| 5; 2 |] (Array.init 10 Float.of_int)))

(* Rows the step's program computes from constants: [8] int32 rows of 1 to 8 sum
   to 36; [8; 5] float32 rows of [0, 40). *)
let test_constant_rows () =
  let sum_rows (type v s) (dt : (v, s) Nx.dtype) shape x =
    let rows =
      Nx.place (Nx.placement x) (Nx.arange dt 1 (Operand.numel shape + 1) 1)
    in
    let c, _ =
      Rune.scan'
        (fun c r -> (Nx.add c (Nx.sum r), c))
        ~init:x (Nx.reshape shape rows)
    in
    c
  in
  let x = on0 (Nx.scalar Nx.int32 0l) in
  let c = Rune.jit' (sum_rows Nx.int32 [| 8 |]) x in
  equal ~msg:"int32 rows" (array int32) [| 36l |] (Nx.to_array c);
  compiles ~msg:"[8; 5] rows" (sum_rows f32 [| 8; 5 |]) (on0 (Nx.scalar f32 0.))

(* A step whose output is a constant, or has no element, over 300 rows. *)
let test_constant_outputs () =
  let xs = on0 (Nx.create f32 [| 300; 2 |] (Array.init 600 Float.of_int)) in
  let scan_with out xs =
    let c, ys =
      Rune.scan' (fun c r -> (Nx.add c r, out c)) ~init:(Nx.get [ 0 ] xs) xs
    in
    Nx.concatenate ~axis:0 [ c; Nx.reshape [| -1 |] ys ]
  in
  compiles ~msg:"a constant output"
    (scan_with (fun c -> Nx.zeros_at (Nx.placement c) f32 [||]))
    xs;
  compiles ~msg:"an empty output"
    (scan_with (fun c -> Nx.zeros_at (Nx.placement c) f32 [| 0 |]))
    xs

(* A step that takes its carry's elements at constant indices. *)
let test_constant_take () =
  let f xs =
    fst
      (Rune.scan'
         (fun c r ->
           let i =
             Nx.place (Nx.placement c)
               (Nx.create Nx.int64 [| 3 |] [| 2L; 0L; 1L |])
           in
           (Nx.add (Nx.take ~axis:0 i c) r, c))
         ~init:(Nx.get [ 0 ] xs) xs)
  in
  compiles ~msg:"take" f
    (on0 (Nx.create f32 [| 4; 3 |] (Array.init 12 Float.of_int)))

(* A carry with no element beside rows with elements: compiled, and its gradient
   mapped over no lane. *)
let test_empty_carry () =
  let f xs =
    let c, ys =
      Rune.scan'
        (fun c r -> (Nx.neg c, Nx.add r r))
        ~init:(Nx.zeros_at (Nx.placement xs) f32 [| 0 |])
        xs
    in
    Nx.concatenate ~axis:0 [ c; Nx.reshape [| -1 |] ys ]
  in
  let xs = on0 (Nx.create f32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]) in
  compiles ~msg:"an empty carry" f xs;
  let lanes = on0 (Nx.zeros f32 [| 0; 3; 2 |]) in
  let g = Rune.vmap' (Rune.grad' (fun x -> Nx.sum (f x))) in
  equal ~msg:"vmap of grad over no lane" (array int) [| 0; 3; 2 |]
    (Nx.shape (g lanes));
  equal ~msg:"compiled" (array int) [| 0; 3; 2 |] (Nx.shape (Rune.jit' g lanes))

(* Interface *)

(* Compiled, the step is traced once and the scan is one loop of the program,
   which runs no eager kernel, whatever its number of rows. *)
module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ Nx_support.memory 0 ])

let count (type d) (x : (float, Nx.float32_elt, d) Nx.t) = Nx.place Count.on x

(* The occurrences of [word] in [s]. *)
let occurrences word s =
  let n = String.length word and k = ref 0 in
  for i = 0 to String.length s - n do
    if String.sub s i n = word then incr k
  done;
  !k

let test_staged () =
  let traces = ref 0 in
  let step c r =
    incr traces;
    (Nx.add c (Nx.mul c r), Nx.mul_s c 2.)
  in
  let rows n =
    count (Nx.reshape [| n; 2 |] (Nx.linspace f32 (-0.01) 0.01 (2 * n)))
  in
  let xs = rows 100 in
  let init = count (v [| 0.5; -0. |]) in
  let f init xs = Rune.scan' step ~init xs in
  let printed xs =
    Format.asprintf "%a" Rune.pp_program
      (Rune.program
         T.(pair tensor tensor)
         T.(pair tensor tensor)
         (fun (a, b) -> f a b)
         (init, xs))
  in
  let short = printed (rows 3) and long = printed xs in
  equal ~msg:"one scan" int 1 (occurrences "scan" long);
  equal ~msg:"its step once" int (occurrences "Map" short)
    (occurrences "Map" long);
  let g = Rune.jit T.(tensor @-> tensor @-> returns (pair tensor tensor)) f in
  traces := 0;
  ignore (g init xs);
  equal ~msg:"one trace" int 1 !traces;
  traces := 0;
  C.reset ();
  let c, ys = g init xs in
  equal ~msg:"no retrace" int 0 !traces;
  equal ~msg:"no eager kernel" int 0 (C.calls ());
  let c', ys' = f init xs in
  equal ~msg:"carry" bits_t (bits c') (bits c);
  equal ~msg:"outputs" bits_t (bits ys') (bits ys)

(* A compiled call in a step stays a call, which computes on each row the bits
   it computes on that row outside the loop, over rows in every view and a
   capture. *)
type call =
  | Call : {
      dt : (float, 's) Nx.dtype;
      sine : (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t;
      s : int array;
      n : int;
      xs : Operand.t;
      c : Operand.t;
    }
      -> call

let sine32 = Rune.jit' (Nx.sin : (float, Nx.float32_elt, Nx.host) Nx.t -> _)
let sine64 = Rune.jit' (Nx.sin : (float, Nx.float64_elt, Nx.host) Nx.t -> _)

let call_cases =
  let open Gen in
  (let* s = Operand.shape in
   let* n = of_list [ 1; 2; 5 ] in
   let* xs = Operand.draw (Array.append [| n |] s) and+ c = Operand.draw s in
   let+ wide = bool in
   if wide then Call { dt = Nx.float64; sine = sine64; s; n; xs; c }
   else Call { dt = Nx.float32; sine = sine32; s; n; xs; c })
  |> with_pp (fun ppf (Call c) ->
      Format.fprintf ppf "%a %a, %d rows in %a" Nx.pp_dtype c.dt
        Operand.pp_shape c.s c.n Operand.pp_view c.xs.view)

let law_call (Call c) =
  cover "a view" (c.xs.view <> Operand.Plain);
  cover "a zero-size row" (Operand.numel c.s = 0);
  let xs = Operand.make c.dt (Array.append [| c.n |] c.s) c.xs in
  let w = Operand.make c.dt c.s c.c in
  (* Each call reads the carry or the row, so that it computes in the step. *)
  let step c' r =
    let y = c.sine (Nx.maximum c' r) in
    (y, Nx.add y (c.sine (Nx.maximum w r)))
  in
  let rows = List.init c.n (fun i -> Nx.get [ i ] xs) in
  cover "a row the call computes apart from eager"
    (List.exists (fun r -> bits (c.sine r) <> bits (Nx.sin r)) rows);
  let init = Nx.zeros c.dt c.s in
  let folded, ys = fold step ~init xs in
  let scanned, ys' = Rune.scan' step ~init xs in
  equal ~msg:"carry" bits_t (bits folded) (bits scanned);
  equal ~msg:"outputs" bits_t (bits (Nx.stack ys)) (bits ys')

(* Constants and placements *)

let host x = Nx.place Nx.Host.on x

(* Row 0 of the outputs is the constant [init], the others are placed. *)
let test_constant_row () =
  let c, ys =
    Rune.scan t t t
      (fun c x -> (Nx.add c x, c))
      ~init:(Nx.scalar f32 0.)
      (host (v [| 1.; 2.; 3. |]))
  in
  equal bits_t (bits (Nx.scalar f32 6.)) (bits c);
  equal bits_t (bits (v [| 0.; 1.; 3. |])) (bits ys)

(* A constant addition, then a placed one, to one total in a step. *)
let test_constant_addition () =
  let u = Total.make () in
  let (), total =
    Total.collect u ~init:(Nx.scalar f32 0.) (fun () ->
        ignore
          (Rune.scan t t Nx.Ptree.unit
             (fun c x ->
               Total.add u (Nx.scalar f32 1.);
               Total.add u (Nx.sum x);
               (c, ()))
             ~init:(Nx.scalar f32 0.)
             (host (v [| 1.; 2.; 3. |]))))
  in
  equal bits_t (bits (Nx.scalar f32 9.)) (bits total)

(* Leaves that swap devices of one set bind as traced: one trace. *)
let test_swap () =
  let on k x = Nx.place (Nx.Placement.device Two.v (Nx_support.memory k)) x in
  let traces = ref 0 in
  let (a, b), _ =
    Rune.scan (Nx.Ptree.pair t t) t t
      (fun (a, b) x ->
        incr traces;
        ((b, a), x))
      ~init:(on 0 (Nx.scalar f32 1.), on 1 (Nx.scalar f32 2.))
      (on 0 (v [| 0. |]))
  in
  equal ~msg:"traces" int 1 !traces;
  equal bits_t (bits (Nx.scalar f32 2.)) (bits a);
  equal bits_t (bits (Nx.scalar f32 1.)) (bits b)

let f64 = Nx.float64
let vec xs = Nx.create f64 [| Array.length xs |] xs
let floats = Testable.array Testable.float_exact
let pair = T.(pair tensor tensor)

(* [a' = 2a], [b' = a] from [b = 0], a constant: [b] gains a tangent, lanes and
   a cotangent after the first step, and after three [b = 4 a0]. *)
let doubling x =
  let (_, b), () =
    Rune.scan pair T.tensor T.unit
      (fun (a, _) _ -> ((Nx.mul_s a 2., a), ()))
      ~init:(x, Nx.zeros f64 [| 2 |])
      (Nx.zeros f64 [| 3; 1 |])
  in
  b

let test_gains () =
  let x = vec [| 1.; -2. |] and t = vec [| 0.5; 3. |] in
  let y, dy = Rune.jvp' doubling x t in
  equal ~msg:"primal" floats [| 4.; -8. |] (Nx.to_array y);
  equal ~msg:"tangent" floats [| 2.; 12. |] (Nx.to_array dy);
  equal ~msg:"gradient" floats [| 4.; 4. |]
    (Nx.to_array (Rune.grad' (fun x -> Nx.sum (doubling x)) x));
  let xs = Nx.create f64 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  equal ~msg:"lanes" floats [| 4.; 8.; 12.; 16. |]
    (Nx.to_array (Rune.vmap' doubling xs))

(* A loop's additions under a transformation are the fold's, from the collect's
   running total; inside a map, each lane's. *)
let test_totals () =
  let t = Total.make () in
  let xs = Nx.create f64 [| 3; 2 |] [| 1e16; 3.; 1.; 5.; 1.; 7. |] in
  let step c x =
    Total.add t (Nx.sum x);
    (Nx.mul c x, ())
  in
  let scanned c =
    Total.add t (Nx.scalar f64 1.);
    Nx.sum (fst (Rune.scan T.tensor T.tensor T.unit step ~init:c xs))
  in
  let folded c =
    Total.add t (Nx.scalar f64 1.);
    Nx.sum (fst (fold step ~init:c xs))
  in
  let total f = snd (Total.collect t ~init:(Nx.scalar f64 0.) f) in
  let c = vec [| 0.5; 2. |] in
  let check msg g =
    equal ~msg bits_t
      (bits (total (fun () -> ignore (g folded))))
      (bits (total (fun () -> ignore (g scanned))))
  in
  check "eager" (fun f -> f c);
  check "jvp" (fun f -> Rune.jvp' f c c);
  check "grad" (fun f -> Rune.grad' f c);
  (* A collect inside a map: each lane's total, chained from its own running
     value. *)
  let lanes f =
    Rune.vmap'
      (fun c ->
        snd
          (Total.collect t ~init:(Nx.scalar f64 0.) (fun () ->
               Total.add t (Nx.sum c);
               ignore (f c))))
      (Nx.create f64 [| 2; 2 |] [| 1e16; 1.; -1e16; 2. |])
  in
  equal ~msg:"inside a map" bits_t (bits (lanes folded)) (bits (lanes scanned))

(* A total inside a differentiated function has the derivative of the additions
   a loop makes, as of those written out; one outside it receives their primals,
   from an iterate too. *)
let test_totals_inside () =
  let t = Total.make () in
  let xs = Nx.create f64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let step x c r =
    Total.add t (Nx.sum (Nx.mul x r));
    (Nx.add c r, ())
  in
  let inside loop x =
    snd
      (Total.collect t ~init:(Nx.scalar f64 0.) (fun () ->
           Total.add t (Nx.sum (Nx.sin x));
           ignore (loop x)))
  in
  let scanned x = Rune.scan T.tensor T.tensor T.unit (step x) ~init:x xs in
  let folded x = fold (step x) ~init:x xs in
  let x = vec [| 0.5; -1. |] and v = vec [| 1.; 2. |] in
  equal ~msg:"grad" bits_t
    (bits (Rune.grad' (inside folded) x))
    (bits (Rune.grad' (inside scanned) x));
  equal ~msg:"jvp" bits_t
    (bits (snd (Rune.jvp' (inside folded) x v)))
    (bits (snd (Rune.jvp' (inside scanned) x v)));
  let remat x =
    Rune.remat
      T.(tensor @-> returns tensor)
      (fun x ->
        Total.add t (Nx.sum (Nx.mul x x));
        x)
      x
  in
  let written x =
    Total.add t (Nx.sum (Nx.mul x x));
    x
  in
  equal ~msg:"grad of a remat" bits_t
    (bits (Rune.grad' (inside written) x))
    (bits (Rune.grad' (inside remat) x));
  (* An iterate's chain reaches a collect outside the differentiation as its
     primal. *)
  let iterated x =
    Rune.iterate' ~max:5
      ~until:(fun c -> Nx.greater (Nx.max c) (Nx.scalar f64 4.))
      (fun c ->
        Total.add t (Nx.sum c);
        Nx.mul_s c 2.)
      x
  in
  let (), total =
    Total.collect t ~init:(Nx.scalar f64 0.) (fun () ->
        ignore (Rune.jvp' iterated (vec [| 1. |]) (vec [| 1. |])))
  in
  equal ~msg:"an iterate under jvp" bits_t
    (bits (Nx.scalar f64 7.))
    (bits total)

(* A scan whose step runs an iterate over a tracked value is differentiated and
   mapped as its fold. *)
let test_iterate_in_step () =
  let xs = Nx.create f64 [| 2; 1 |] [| 3.; 0.25 |] in
  let step c r =
    let c =
      Rune.iterate' ~max:10
        ~until:(fun c -> Nx.less (Nx.max c) (Nx.scalar f64 1.))
        (fun c -> Nx.mul_s c 0.5)
        (Nx.add c r)
    in
    (c, ())
  in
  let scanned x = fst (Rune.scan T.tensor T.tensor T.unit step ~init:x xs) in
  let folded x = fst (fold step ~init:x xs) in
  let x = vec [| 0.5 |] in
  equal ~msg:"jvp" bits_t
    (bits (snd (Rune.jvp' folded x x)))
    (bits (snd (Rune.jvp' scanned x x)));
  equal ~msg:"grad" bits_t
    (bits (Rune.grad' (fun x -> Nx.sum (folded x)) x))
    (bits (Rune.grad' (fun x -> Nx.sum (scanned x)) x));
  (* Under a map, an iterate whose stop the lanes share: three halvings. *)
  let step c r =
    let c, _ =
      Rune.iterate
        T.(pair tensor tensor)
        ~max:10
        ~until:(fun (_, k) -> Nx.greater_equal k (Nx.scalar f64 3.))
        (fun (c, k) -> (Nx.mul_s c 0.5, Nx.add_s k 1.))
        (Nx.add c r, Nx.scalar f64 0.)
    in
    (c, ())
  in
  let scanned x = fst (Rune.scan T.tensor T.tensor T.unit step ~init:x xs) in
  let folded x = fst (fold step ~init:x xs) in
  let lanes = Nx.create f64 [| 2; 1 |] [| 0.5; 2. |] in
  equal ~msg:"vmap" bits_t
    (bits
       (Nx.stack [ folded (Nx.get [ 0 ] lanes); folded (Nx.get [ 1 ] lanes) ]))
    (bits (Rune.vmap' scanned lanes))

(* Memory *)

(* The bytes [keep] holds after a run, beyond those an uncounted run left; what
   it kept is dropped before the next measurement. *)
let held_after kept keep =
  let held = Memory.held keep in
  ignore (Sys.opaque_identity !kept);
  kept := [];
  held

(* Reverse mode keeps per step the loads of the tangent's maps, stacked, but
   reads a row or a capture where it lies. *)
let test_residuals () =
  let n = 1024 and rows = 8 in
  let f32 = Nx.float32 in
  let x = Nx.place Memory.On.on (Nx.ones f32 [| n |]) in
  let w = Nx.place Memory.On.on (Nx.full f32 [| n |] 0.5) in
  let xs = Nx.place Memory.On.on (Nx.full f32 [| rows; n |] 0.25) in
  let kept = ref [] in
  let held step =
    held_after kept (fun () ->
        kept :=
          Rune.vjp'
            (fun c -> fst (Rune.scan T.tensor T.tensor T.unit step ~init:c xs))
            x
          :: !kept)
  in
  let bytes k = 4 * n * k in
  (* The result, and each step's carry, which the sine's tangent reads. *)
  equal ~msg:"a step's carry" int
    (bytes (1 + rows))
    (held (fun c _ -> (Nx.sin c, ())));
  equal ~msg:"a capture" int (bytes 1) (held (fun c _ -> (Nx.mul c w, ())));
  equal ~msg:"a row" int (bytes 1) (held (fun c r -> (Nx.mul c r, ())));
  (* Held to the end, so that no measurement sees them collected. *)
  ignore (Sys.opaque_identity (x, w, xs))

(* Two contributions of [-0.], through a capture and a carry, sum to [-0.]. *)
let test_signs () =
  let w = vec [| -0.; 1. |] in
  let g =
    Rune.grad'
      (fun w ->
        Nx.sum
          (fst
             (Rune.scan T.tensor T.tensor T.unit
                (fun c r -> (Nx.mul c (Nx.mul w r), ()))
                ~init:w
                (Nx.create f64 [| 1; 2 |] [| 1.; -0. |]))))
      w
  in
  (* [w r w] at row [1; -0]: [2 w r]: [-0.] and [-0.]. *)
  equal bits_t (bits (vec [| -0.; -0. |])) (bits g)

let () =
  exit
    (run "rune scan"
       [
         group "eagerly"
           [
             prop "a scan is its fold" cases law;
             prop "a scan is its fold at every dtype" any_cases law_any;
             test "the step runs once per call" test_runs_once;
             test "a read raises naming the loop" test_read;
             test "a keyless draw raises naming the loop" test_keyless;
             test "a keyed draw is the fold's" test_keyed;
             test "a carry that changes raises" test_carry_refused;
             test "rows without a leading axis raise" test_xs_refused;
             test "captures and nested scans" test_nested;
           ];
         group "constructs in steps"
           [
             test "detach in a step" test_detach;
             test "lanes and lane_index raise with no map" test_no_map;
             test "a loop counts each step's additions" test_in_loops;
             prop "a scan adds what its fold adds" total_cases law_totals;
             test "a scope inside a step" test_scopes_inside;
           ];
         group "transformations"
           [
             prop "derivatives and maps over views are the fold's" cases
               law_derived;
             prop "jvp of scan is jvp of its fold"
               (Gen.pair rule_cases compiled_too)
               law_jvp;
             prop
               ~examples:(List.map (fun c -> (c, true)) grad_examples)
               "grad of scan is grad of its fold"
               (Gen.pair rule_cases compiled_too)
               law_grad;
             prop "vmap of scan is the fold per lane"
               (Gen.pair map_cases compiled_too)
               law_vmap;
             prop "compositions are the fold's" rule_cases law_compose;
             test "a carry that gains a tangent and lanes" test_gains;
             test "additions under transformations" test_totals;
             test "a total inside a differentiation" test_totals_inside;
             test "a step that runs an iterate" test_iterate_in_step;
             test "two negative zeros sum to a negative zero" test_signs;
           ];
         group "compiled"
           [
             prop "a scan of exact functions is its fold" cases law_compiled;
             test "nested scans of one row or more" test_nested_compiled;
             test "carries in a cycle" test_cycle;
             test "rows computed from constants" test_constant_rows;
             test "constant and empty outputs" test_constant_outputs;
             test "a take at constant indices" test_constant_take;
             test "an empty carry beside rows" test_empty_carry;
             test "a step traces once and compiles one loop" test_staged;
             prop "a compiled call in a step computes its own bits" call_cases
               law_call;
           ];
         group "constants and placements"
           [
             test "a constant first row" test_constant_row;
             test "a constant addition before a placed one"
               test_constant_addition;
             test "leaves that swap devices of one set" test_swap;
           ];
         group "memory" [ test "what reverse mode keeps" test_residuals ];
       ])
