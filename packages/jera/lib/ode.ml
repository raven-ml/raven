(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Methods *)

(* An embedded error estimate: the weights [e = b − b̂] of the difference between
   the step and its embedded formula, and the controller's order, one more than
   the lower of the two orders. [dense] is the continuous extension [y (t + θh)
   = y + h Σ_i b_i(θ) k_i]: row [i] holds the coefficients of [θ, θ², ...] in
   [b_i(θ)]. *)
type embedded = { e : float array; order : int; dense : float array array }

(* An explicit Butcher tableau: row [i] of [a] has [i] elements. [fsal] when the
   last stage evaluates the field at the step's result, so the next step starts
   from it. *)
type tableau = {
  name : string;
  a : float array array;
  b : float array;
  c : float array;
  fsal : bool;
  embedded : embedded option;
}

type (-'k, 'y, 't) t = tableau

let first_same_as_last a b c =
  let s = Array.length b in
  s > 1
  && c.(s - 1) = 1.
  && b.(s - 1) = 0.
  && Array.for_all2 ( = ) a.(s - 1) (Array.sub b 0 (s - 1))

let make ?embedded name ~a ~b ~c () =
  { name; a; b; c; fsal = first_same_as_last a b c; embedded }

(* The cubic Hermite interpolant of a step's ends and fields, for a
   first-same-as-last tableau, where [y1 = y0 + h Σ b_i k_i] and the last stage
   is the field at [y1]: [b_i(θ) = (3θ² − 2θ³) b_i] plus [θ − 2θ² + θ³] for the
   first stage and [θ³ − θ²] for the last. *)
let hermite b =
  let s = Array.length b in
  Array.init s (fun i ->
      let first = if i = 0 then 1. else 0.
      and last = if i = s - 1 then 1. else 0. in
      [|
        first;
        (3. *. b.(i)) -. (2. *. first) -. last;
        (-2. *. b.(i)) +. first +. last;
      |])

let euler = make "euler" ~a:[| [||] |] ~b:[| 1. |] ~c:[| 0. |] ()

let rk4 =
  make "rk4"
    ~a:[| [||]; [| 0.5 |]; [| 0.; 0.5 |]; [| 0.; 0.; 1. |] |]
    ~b:[| 1. /. 6.; 1. /. 3.; 1. /. 3.; 1. /. 6. |]
    ~c:[| 0.; 0.5; 0.5; 1. |] ()

let ssprk3 =
  make "ssprk3"
    ~a:[| [||]; [| 1. |]; [| 0.25; 0.25 |] |]
    ~b:[| 1. /. 6.; 1. /. 6.; 2. /. 3. |]
    ~c:[| 0.; 1.; 0.5 |] ()

let bs3 =
  make "bs3"
    ~embedded:
      {
        e = [| -5. /. 72.; 1. /. 12.; 1. /. 9.; -1. /. 8. |];
        order = 3;
        dense = hermite [| 2. /. 9.; 1. /. 3.; 4. /. 9.; 0. |];
      }
    ~a:[| [||]; [| 0.5 |]; [| 0.; 0.75 |]; [| 2. /. 9.; 1. /. 3.; 4. /. 9. |] |]
    ~b:[| 2. /. 9.; 1. /. 3.; 4. /. 9.; 0. |]
    ~c:[| 0.; 0.5; 0.75; 1. |] ()

(* Tsitouras's (2011) free interpolant of order 4, as the reference
   implementations carry it. *)
let tsit5_dense =
  [|
    [| 1.0; -2.763706197274826; 2.9132554618219126; -1.0530884977290216 |];
    [| 0.; 0.13169999999999998; -0.2234; 0.1017 |];
    [| 0.; 3.9302962368947516; -5.941033872131505; 2.490627285651253 |];
    [| 0.; -12.411077166933676; 30.33818863028232; -16.548102889244902 |];
    [| 0.; 37.50931341651104; -88.1789048947664; 47.37952196281928 |];
    [| 0.; -27.896526289197286; 65.09189467479366; -34.87065786149661 |];
    [| 0.; 1.5; -4.; 2.5 |];
  |]

(* Tsitouras (2011), Table 1, as the reference implementations carry it. *)
let tsit5 =
  let b =
    [|
      0.09646076681806523;
      0.01;
      0.4798896504144996;
      1.379008574103742;
      -3.290069515436081;
      2.324710524099774;
    |]
  in
  make "tsit5"
    ~embedded:
      {
        e =
          [|
            -0.00178001105222577714;
            -0.0008164344596567469;
            0.007880878010261995;
            -0.1447110071732629;
            0.5823571654525552;
            -0.45808210592918697;
            1. /. 66.;
          |];
        order = 5;
        dense = tsit5_dense;
      }
    ~a:
      [|
        [||];
        [| 0.161 |];
        [| -0.008480655492356989; 0.335480655492357 |];
        [| 2.897153057105493; -6.359448489975075; 4.3622954328695815 |];
        [|
          5.325864828439257;
          -11.748883564062828;
          7.4955393428898365;
          -0.09249506636175525;
        |];
        [|
          5.86145544294642;
          -12.92096931784711;
          8.159367898576159;
          -0.071584973281401;
          -0.028269050394068383;
        |];
        b;
      |]
    ~b:(Array.append b [| 0. |])
    ~c:[| 0.; 0.161; 0.327; 0.9; 0.9800255409045097; 1.; 1. |]
    ()

(* Hairer's dense output of order 4 for dopri5 (Hairer, Nørsett and Wanner, I,
   §II.6): [y (t + θh) = y0 + θ (r2 + (1 − θ) (r3 + θ (r4 + (1 − θ) r5)))] with
   [r2 = y1 − y0], [r3 = h k_1 − r2], [r4 = r2 − h k_7 − r3] and [r5 = h Σ d_i
   k_i], expanded in powers of [θ]. *)
let dopri5_d =
  [|
    -12715105075. /. 11282082432.;
    0.;
    87487479700. /. 32700410799.;
    -10690763975. /. 1880347072.;
    701980252875. /. 199316789632.;
    -1453857185. /. 822651844.;
    69997945. /. 29380423.;
  |]

let dopri5_dense b =
  Array.init 7 (fun i ->
      let first = if i = 0 then 1. else 0.
      and last = if i = 6 then 1. else 0. in
      let b = b.(i) and d = dopri5_d.(i) in
      [|
        first;
        (3. *. b) -. (2. *. first) -. last +. d;
        (-2. *. b) +. first +. last -. (2. *. d);
        d;
      |])

let dopri5 =
  let b =
    [|
      35. /. 384.; 0.; 500. /. 1113.; 125. /. 192.; -2187. /. 6784.; 11. /. 84.;
    |]
  in
  make "dopri5"
    ~embedded:
      {
        e =
          [|
            71. /. 57600.;
            0.;
            -71. /. 16695.;
            71. /. 1920.;
            -17253. /. 339200.;
            22. /. 525.;
            -1. /. 40.;
          |];
        order = 5;
        dense = dopri5_dense (Array.append b [| 0. |]);
      }
    ~a:
      [|
        [||];
        [| 1. /. 5. |];
        [| 3. /. 40.; 9. /. 40. |];
        [| 44. /. 45.; -56. /. 15.; 32. /. 9. |];
        [| 19372. /. 6561.; -25360. /. 2187.; 64448. /. 6561.; -212. /. 729. |];
        [|
          9017. /. 3168.;
          -355. /. 33.;
          46732. /. 5247.;
          49. /. 176.;
          -5103. /. 18656.;
        |];
        b;
      |]
    ~b:(Array.append b [| 0. |])
    ~c:[| 0.; 1. /. 5.; 3. /. 10.; 4. /. 5.; 8. /. 9.; 1.; 1. |]
    ()

(* A sum of [n] floats is within [n * eps * Σ |b_i|] of its exact value. *)
let sums_to_one b =
  let sum = Array.fold_left ( +. ) 0. b in
  let size = Array.fold_left (fun acc x -> acc +. Float.abs x) 0. b in
  Float.abs (sum -. 1.) <= float (Array.length b) *. epsilon_float *. size

let tableau ~a ~b ~c =
  let fail fmt =
    Printf.ksprintf (fun m -> invalid_arg ("Jera.Ode.tableau: " ^ m)) fmt
  in
  let s = Array.length b in
  if s = 0 then fail "b is empty";
  if Array.length c <> s then fail "c has %d elements, b %d" (Array.length c) s;
  if Array.length a <> s then
    fail "a has %d rows, b %d elements" (Array.length a) s;
  Array.iteri
    (fun i row ->
      if Array.length row <> i then
        fail "row %d of a has %d elements; row i has i" i (Array.length row))
    a;
  let finite = Array.for_all Float.is_finite in
  if not (finite b && finite c && Array.for_all finite a) then
    fail "a coefficient is not finite";
  if not (sums_to_one b) then fail "b does not sum to 1";
  make "tableau" ~a:(Array.map Array.copy a) ~b:(Array.copy b) ~c:(Array.copy c)
    ()

(* Marches *)

type ('y, 't) field = (float, 't) Nx.t -> 'y -> 'y
type 't time = (float, 't) Nx.t

(* The leaves' dtypes and shapes, which a field's value must keep. *)
let layout y v =
  ( Nx.Ptree.visits y v,
    Nx.Ptree.fold y
      (fun _ x acc -> (Nx_dtype.to_string (Nx.dtype x), Nx.shape x) :: acc)
      v [] )

let eval fn y f t v =
  let dv = f t v in
  if layout y dv <> layout y v then
    invalid_arg
      (Printf.sprintf
         "%s: the field returned a value of another structure, dtype or shape \
          than its state"
         fn);
  dv

(* One step of [m] from [(t, v)] by [h], given the field at [(t, v)] when the
   method reuses its last stage: the new state and the stages. *)
let step fn y m f t h v k0 =
  let s = Array.length m.b in
  let ks = Array.make s v in
  let combine w =
    let acc = ref v in
    Array.iteri
      (fun j wj ->
        if wj <> 0. then acc := Nx.Ptree.axpy y (Nx.mul_s h wj) ks.(j) !acc)
      w;
    !acc
  in
  let last = ref v in
  for i = 0 to s - 1 do
    match (i, k0) with
    | 0, Some k -> ks.(0) <- k
    | _ ->
        last := combine m.a.(i);
        ks.(i) <- eval fn y f (Nx.add t (Nx.mul_s h m.c.(i))) !last
  done;
  ((if m.fsal then !last else combine m.b), ks)

let march y m ~steps f ~at y0 =
  let fn = "Jera.Ode.march" in
  March.check fn ~steps at;
  let dtype = Nx.dtype at in
  let interval c step t0 t1 carry =
    let h = Nx.div_s (Nx.sub t1 t0) (float steps) in
    March.steps c dtype steps
      (fun j carry -> step (Nx.add t0 (Nx.mul j h)) h carry)
      carry
  in
  if m.fsal then
    let c = Nx.Ptree.pair y y in
    let step t h (v, k) =
      let v, ks = step fn y m f t h v (Some k) in
      (v, ks.(Array.length ks - 1))
    in
    let k0 = eval fn y f (Nx.get [ 0 ] at) y0 in
    March.run c y ~at ~interval:(interval c step) ~state:fst (y0, k0)
  else
    let step t h v = fst (step fn y m f t h v None) in
    March.run y y ~at ~interval:(interval y step) ~state:Fun.id y0

(* Solves *)

(* The proportional–integral controller (Hairer, Nørsett and Wanner, I, §II.4):
   the next step is the last times 0.9 r^(−0.7/k) r'^(0.4/k), r the error ratio
   of the step and r' the last accepted one's, k the controller's order, bounded
   to [0.2, 10] and below 1 after a rejection. *)
let safety = 0.9
let shrink = 0.2
let grow = 10.

(* Each float leaf's [e / (abs + rel max (|v|, |w|))] as one vector of [dtype],
   then their root mean square in a fixed order. *)
let error_norm (type t) y tol (dtype : (float, t) Nx.dtype) e v w =
  let es, _ = Nx.Ptree.flatten y e in
  let vs, _ = Nx.Ptree.flatten y v and ws, _ = Nx.Ptree.flatten y w in
  let leaf (type a c) (e : (a, c) Nx.t) pv pw =
    let go (type d) (e : (float, d) Nx.t) =
      let v = Nx.unpack (Nx.dtype e) pv and w = Nx.unpack (Nx.dtype e) pw in
      let r = Tol.ratio tol ~e ~y:(Nx.maximum (Nx.abs v) (Nx.abs w)) in
      [ Nx.reshape [| -1 |] (Nx.cast dtype r) ]
    in
    match Nx.dtype e with
    | Nx.Float64 -> go e
    | Nx.Float32 -> go e
    | Nx.Float16 -> go e
    | Nx.BFloat16 -> go e
    | Nx.Float8_e4m3 -> go e
    | Nx.Float8_e5m2 -> go e
    | _ -> []
  in
  let rows =
    List.concat
      (List.map2
         (fun (Nx.P e) (pv, pw) -> leaf e pv pw)
         es (List.combine vs ws))
  in
  let flat =
    match rows with
    | [] -> Nx.zeros dtype [| 0 |]
    | rows -> Nx.concatenate ~axis:0 rows
  in
  Nx.reshape [||] (Num.rms_rows (Nx.reshape [| 1; -1 |] flat))

(* [true] when every element of every leaf of [v] is finite. *)
let finite y v =
  Nx.Ptree.fold y
    (fun _ x acc -> Nx.logical_and acc (Nx.all (Nx.isfinite x)))
    v (Nx.scalar Nx.bool true)

(* What a search carries besides its state: the history a delay reads, or the
   signs an event watches. [start t v] is the memory at the first time; [field
   m] the field the steps see; [limit m] the largest step, if any; [accept m ~t
   ~t_end ~h ~v ~ks v'] the memory after an accepted step from [(t, v)] to
   [(t_end, v')]. Each also gives conditions that end a lane, with their status,
   settled in order. *)
type outcome = (bool, Nx.bool_elt) Nx.t * Solution.status

type ('y, 't, 'm) memory = {
  tree : 'm Nx.Ptree.t;
  start : 't time -> 'y -> 'm * outcome list;
  field : 'm -> ('y, 't) field;
  limit : 'm -> 't time option;
  accept :
    'm ->
    t:'t time ->
    t_end:'t time ->
    h:'t time ->
    v:'y ->
    ks:'y array ->
    'y ->
    'm * outcome list;
}

(* The memory of a solve that needs none. *)
let plain f =
  {
    tree = Nx.Ptree.unit;
    start = (fun _ _ -> ((), []));
    field = (fun () -> f);
    limit = (fun () -> None);
    accept = (fun () ~t:_ ~t_end:_ ~h:_ ~v:_ ~ks:_ _ -> ((), []));
  }

(* The search's carry. *)
type ('y, 't, 'm) search = {
  v : 'y;  (** The state. *)
  k : 'y;  (** The field at the state. *)
  ys : 'y;  (** The states at the times of [at] reached so far. *)
  errs : 'y;  (** The accumulated local errors there. *)
  acc : 'y;  (** The accumulated local errors at the state. *)
  interval : (int32, Nx.int32_elt) Nx.t;
  sigma : (float, 't) Nx.t;  (** The state's fraction of its interval. *)
  h : (float, 't) Nx.t;  (** The next step's size, absolute. *)
  prev : (float, 't) Nx.t;  (** The last accepted error ratio. *)
  rejected : (bool, Nx.bool_elt) Nx.t;
  ends : (float, 't) Nx.t;  (** The accepted steps' ends, as fractions. *)
  counts : (int32, Nx.int32_elt) Nx.t;  (** The accepted steps per interval. *)
  accepted : (int32, Nx.int32_elt) Nx.t;
  attempts : (int32, Nx.int32_elt) Nx.t;
  evals : (int32, Nx.int32_elt) Nx.t;
  status : (int32, Nx.int32_elt) Nx.t;
  memory : 'm;
}

let search_ptree (type v t m) (y : v Nx.Ptree.t) (tree : m Nx.Ptree.t) :
    (v, t, m) search Nx.Ptree.t =
  let module M = struct
    type nonrec _ t = (v, t, m) search

    let walk c s =
      let open Nx.Ptree.Walk in
      let v = field c "v" (structure y) s.v in
      let k = field c "k" (structure y) s.k in
      let ys = field c "ys" (structure y) s.ys in
      let errs = field c "errs" (structure y) s.errs in
      let acc = field c "acc" (structure y) s.acc in
      let interval = field c "interval" tensor s.interval in
      let sigma = field c "sigma" tensor s.sigma in
      let h = field c "h" tensor s.h in
      let prev = field c "prev" tensor s.prev in
      let rejected = field c "rejected" tensor s.rejected in
      let ends = field c "ends" tensor s.ends in
      let counts = field c "counts" tensor s.counts in
      let accepted = field c "accepted" tensor s.accepted in
      let attempts = field c "attempts" tensor s.attempts in
      let evals = field c "evals" tensor s.evals in
      let status = field c "status" tensor s.status in
      let memory = field c "memory" (structure tree) s.memory in
      {
        v;
        k;
        ys;
        errs;
        acc;
        interval;
        sigma;
        h;
        prev;
        rejected;
        ends;
        counts;
        accepted;
        attempts;
        evals;
        status;
        memory;
      }
  end in
  Nx.Ptree.instantiate (module M)

let scalar_at t i =
  Nx.reshape [||] (Nx.take ~indices:(Nx.reshape [| 1 |] (Nx.cast Nx.int64 i)) t)

(* What to change for a solve that stopped short, from its lane's facts. *)
let fix tol (st : Solution.status) facts =
  let fact name = List.assoc name facts in
  let some name = Option.value ~default:0. (List.assoc_opt name facts) in
  match st with
  | Budget_spent ->
      "Raise the budget or loosen tol; if the steps stay small, the field may \
       be stiff."
  | Stalled when fact "disorder" >= 0. ->
      Printf.sprintf "The times are not strictly monotone at [%.0f]."
        (fact "disorder")
  | Stalled when some "lags not positive" > 0. -> "Every lag must be positive."
  | Stalled when some "beyond pieces" > 0. ->
      Printf.sprintf
        "The largest lag, %g, reaches back past the last pieces = %.0f steps \
         near t: raise pieces to about %.0f."
        (fact "largest lag") (fact "pieces") (fact "pieces needed")
  | Stalled ->
      "The step fell below the time's resolution near t: the solution may blow \
       up there." ^ Tol.zero_hint tol
  | Not_finite -> "The field is not finite at an accepted state near t."
  | Converged | Not_bracketed -> ""

(* The index of the first time of [at] that does not continue its direction, or
   [−1]: strictly, or with repeats when [repeats] is [`Allowed]. *)
let disorder repeats at =
  let n = Nx.dim 0 at in
  if n < 2 then Nx.scalar Nx.int32 (-1l)
  else
    let d =
      Nx.sub (Nx.shrink [| (1, n) |] at) (Nx.shrink [| (0, n - 1) |] at)
    in
    let direction = Nx.sign (Nx.sub (Nx.get [ n - 1 ] at) (Nx.get [ 0 ] at)) in
    let along = Nx.mul d direction in
    let fine =
      match repeats with
      | `Allowed -> Nx.greater_equal along (Nx.zeros_like along)
      | `Refused -> Nx.greater along (Nx.zeros_like along)
    in
    let bad = Nx.logical_not fine in
    let first = Nx.cast Nx.int32 (Nx.argmax (Nx.cast Nx.int32 bad)) in
    Nx.where (Nx.any bad) (Nx.add_s first 1l) (Nx.scalar Nx.int32 (-1l))

(* Checks the arguments every solve over [at] checks. *)
let check fn ~at ~budget =
  if Nx.ndim at <> 1 || Nx.dim 0 at = 0 then
    invalid_arg
      (Printf.sprintf "%s: at must hold at least one time, got shape %s" fn
         (Num.shape (Nx.shape at)));
  if budget < 1 then
    invalid_arg (Printf.sprintf "%s: budget = %d is below 1" fn budget)

(* The times [t0] and [t1] of a solve over one interval, as [at]. *)
let endpoints fn t0 t1 =
  if Nx.ndim t0 <> 0 || Nx.ndim t1 <> 0 then
    invalid_arg
      (Printf.sprintf "%s: t0 and t1 must be scalars, got shapes %s and %s" fn
         (Num.shape (Nx.shape t0))
         (Num.shape (Nx.shape t1)));
  Nx.stack [ t0; t1 ]

let embedded fn m =
  match m.embedded with
  | Some e -> e
  | None -> invalid_arg (fn ^ ": the method has no embedded formula")

(* The search over the times [at], two or more: its final carry, and the index
   of the first time out of order, or −1. *)
let search fn repeats y m ~tol ~budget mem ~at y0 =
  let emb = embedded fn m in
  let n = Nx.dim 0 at in
  let dtype = Nx.dtype at in
  let n_int = n - 1 in
  (* Times out of order are a status: the lane stops before its search. *)
  let disorder = disorder repeats (Rune.detach at) in
  let at0 = Rune.detach at in
  let starts = Nx.slice [ Nx.R (0, n_int) ] at0
  and stops = Nx.slice [ Nx.R (1, n) ] at0 in
  let detached v = Nx.Ptree.map y (fun _ x -> Rune.detach x) v in
  let v0 = detached y0 in
  let t0 = Nx.get [ 0 ] at0 in
  let m0, opening = mem.start t0 v0 in
  let m0 = Nx.Ptree.map mem.tree (fun _ x -> Rune.detach x) m0 in
  let cap h = match mem.limit m0 with None -> h | Some l -> Nx.minimum h l in
  let fd t v = detached (eval fn y (mem.field m0) t v) in
  let zeros_like v = Nx.Ptree.map y (fun _ x -> Nx.zeros_like x) v in
  let k0 = fd t0 v0 in
  (* The first step (Hairer, Nørsett and Wanner, I, §II.4): from the scales of
     the state and the field, and a trial Euler step, with the fallbacks taken
     relative to the span of [at]. A scale that is zero, from a component at
     zero with no [abs], makes the estimate meaningless; the fallback then
     starts small and the controller grows it. *)
  let one = Nx.scalar dtype 1. in
  let span_all = Nx.abs (Nx.sub (Nx.get [ n - 1 ] at0) t0) in
  let fallback = Nx.mul_s span_all 1e-6 in
  let d0 = error_norm y tol dtype v0 v0 v0
  and d1 = error_norm y tol dtype k0 v0 v0 in
  let tiny = Nx.logical_or (Nx.less_s d0 1e-5) (Nx.less_s d1 1e-5) in
  let h0 =
    cap
      (Nx.where tiny fallback
         (Nx.div (Nx.mul_s d0 0.01) (Nx.where tiny one d1)))
  in
  let direction = Nx.sign (Nx.sub (Nx.get [ 1 ] at0) t0) in
  let v1 = Nx.Ptree.axpy y (Nx.mul h0 direction) k0 v0 in
  let k1 = fd (Nx.add t0 (Nx.mul h0 direction)) v1 in
  let diff = Nx.Ptree.axpy y (Nx.neg one) k0 k1 in
  let d2 = Nx.div (error_norm y tol dtype diff v0 v0) h0 in
  let dmax = Nx.maximum d1 d2 in
  let small = Nx.less_equal_s dmax 1e-15 in
  let h1 =
    Nx.where small
      (Nx.maximum fallback (Nx.mul_s h0 1e-3))
      (Nx.pow_s
         (Nx.div (Nx.full_like dmax 0.01) (Nx.where small one dmax))
         (1. /. float emb.order))
  in
  let h_start = Nx.minimum (Nx.minimum (Nx.mul_s h0 100.) h1) span_all in
  let usable = Nx.logical_and (Nx.isfinite h_start) (Nx.greater_s h_start 0.) in
  let h_start = cap (Nx.where usable h_start fallback) in
  let stacked v =
    Nx.Ptree.map y
      (fun _ x ->
        Nx.broadcast_to
          (Array.append [| n |] (Nx.shape x))
          (Nx.unsqueeze ~axes:[ 0 ] x)
        |> Nx.copy)
      v
  in
  let alpha = 0.7 /. float emb.order and beta = 0.4 /. float emb.order in
  let stages = Array.length m.b in
  let attempt s =
    let s =
      match mem.limit s.memory with
      | None -> s
      | Some l -> { s with h = Nx.minimum s.h l }
    in
    let j =
      Nx.minimum s.interval (Nx.scalar Nx.int32 (Int32.of_int (n_int - 1)))
    in
    let a = scalar_at starts j and b = scalar_at stops j in
    let span = Nx.sub b a in
    (* An empty interval, of [t0 = t1] in a solve, ends with no step. *)
    let empty = Nx.equal_s span 0. in
    let t = Nx.add a (Nx.mul span s.sigma) in
    let ds = Nx.div s.h (Nx.abs span) in
    let lands = Nx.greater_equal (Nx.add s.sigma ds) one in
    let ds = Nx.where lands (Nx.rsub_s 1. s.sigma) ds in
    let sigma_end = Nx.where lands one (Nx.add s.sigma ds) in
    let hh = Nx.mul span ds in
    let v', ks =
      step fn y m
        (fun t v -> detached (mem.field s.memory t v))
        t hh s.v (Some s.k)
    in
    let e =
      let acc = ref (zeros_like s.v) in
      Array.iteri
        (fun i w ->
          if w <> 0. then acc := Nx.Ptree.axpy y (Nx.mul_s hh w) ks.(i) !acc)
        emb.e;
      !acc
    in
    let r = error_norm y tol dtype e s.v v' in
    let k' = ks.(stages - 1) in
    let running = Elementwise.searching s.status in
    let stepping = Nx.logical_and running (Nx.logical_not empty) in
    let ok =
      Nx.logical_and running (Nx.logical_or empty (Nx.less_equal_s r 1.))
    in
    let taken = Nx.logical_and ok (Nx.logical_not empty) in
    let v' =
      Nx.Ptree.map2 y
        (fun _ x z -> Nx.where (Nx.broadcast_to (Nx.shape x) empty) z x)
        v' s.v
    in
    let k' =
      Nx.Ptree.map2 y
        (fun _ x z -> Nx.where (Nx.broadcast_to (Nx.shape x) empty) z x)
        k' s.k
    in
    let e =
      Nx.Ptree.map y
        (fun _ x ->
          Nx.where (Nx.broadcast_to (Nx.shape x) empty) (Nx.zeros_like x) x)
        e
    in
    let lands = Nx.logical_or lands empty in
    let t_end = Nx.add a (Nx.mul span sigma_end) in
    (* Controller *)
    let rr = Nx.maximum r (Nx.full_like r 1e-10) in
    let fac =
      Nx.mul_s (Nx.mul (Nx.pow_s rr (-.alpha)) (Nx.pow_s s.prev beta)) safety
    in
    let fac = Nx.where (Nx.isnan fac) (Nx.full_like fac shrink) fac in
    let top = Nx.where s.rejected one (Nx.full_like one grow) in
    let fac_ok = Nx.minimum top (Nx.maximum (Nx.full_like fac shrink) fac) in
    let fac_bad = Nx.minimum one (Nx.maximum (Nx.full_like fac shrink) fac) in
    (* No step exceeds the span of [at]: past it, growth across many short
       intervals would overflow to infinity, which no rejection shrinks. *)
    let h =
      Nx.minimum span_all (Nx.where ok (Nx.mul s.h fac_ok) (Nx.mul s.h fac_bad))
    in
    let pick_y c a b =
      Nx.Ptree.map2 y
        (fun _ x z -> Nx.where (Nx.broadcast_to (Nx.shape x) c) x z)
        a b
    in
    let abs_e = Nx.Ptree.map y (fun _ x -> Nx.abs x) e in
    let acc' = Nx.Ptree.axpy y one abs_e s.acc in
    let finishing = Nx.logical_and ok lands in
    let row =
      Nx.equal (Nx.arange Nx.int32 0 n 1)
        (Nx.broadcast_to [| n |] (Nx.add_s j 1l))
    in
    let put_row c rows v =
      Nx.Ptree.map2 y
        (fun _ rows x ->
          let mask =
            Nx.reshape
              (Array.append [| n |] (Array.make (Nx.ndim x) 1))
              (Nx.logical_and row (Nx.broadcast_to [| n |] c))
          in
          Nx.where
            (Nx.broadcast_to (Nx.shape rows) mask)
            (Nx.broadcast_to (Nx.shape rows) (Nx.unsqueeze ~axes:[ 0 ] x))
            rows)
        rows v
    in
    let slot =
      Nx.equal
        (Nx.arange Nx.int32 0 budget 1)
        (Nx.broadcast_to [| budget |] s.accepted)
    in
    let ends =
      Nx.where
        (Nx.logical_and slot (Nx.broadcast_to [| budget |] taken))
        (Nx.broadcast_to [| budget |] sigma_end)
        s.ends
    in
    let at_j =
      Nx.equal (Nx.arange Nx.int32 0 n_int 1) (Nx.broadcast_to [| n_int |] j)
    in
    let counts =
      Nx.add s.counts
        (Nx.cast Nx.int32
           (Nx.logical_and at_j (Nx.broadcast_to [| n_int |] taken)))
    in
    let interval = Nx.where finishing (Nx.add_s s.interval 1l) s.interval in
    let st = s.status in
    let st =
      Elementwise.settle st
        (Nx.logical_and ok (Nx.logical_not (finite y k')))
        Not_finite
    in
    let memory, outcomes = mem.accept s.memory ~t ~t_end ~h:hh ~v:s.v ~ks v' in
    let st =
      List.fold_left
        (fun st (c, status) ->
          Elementwise.settle st (Nx.logical_and taken c) status)
        st outcomes
    in
    let st =
      Elementwise.settle st
        (Nx.logical_and finishing (Nx.equal_s interval (Int32.of_int n_int)))
        Converged
    in
    let st =
      Elementwise.settle st (Nx.logical_and stepping (Nx.equal t_end t)) Stalled
    in
    let attempts = Nx.add s.attempts (Nx.cast Nx.int32 stepping) in
    let st =
      Elementwise.settle st
        (Nx.greater_equal_s attempts (Int32.of_int budget))
        Budget_spent
    in
    {
      v = pick_y ok v' s.v;
      k = pick_y ok k' s.k;
      ys = put_row finishing s.ys v';
      errs = put_row finishing s.errs acc';
      acc = pick_y ok acc' s.acc;
      interval;
      sigma =
        Nx.where ok (Nx.where lands (Nx.zeros_like sigma_end) sigma_end) s.sigma;
      h = Nx.where stepping h s.h;
      prev = Nx.where taken (Nx.maximum r (Nx.full_like r 1e-4)) s.prev;
      rejected = Nx.logical_and stepping (Nx.logical_not ok);
      ends;
      counts;
      accepted = Nx.add s.accepted (Nx.cast Nx.int32 taken);
      attempts;
      evals =
        Nx.add s.evals
          (Nx.mul_s (Nx.cast Nx.int32 stepping) (Int32.of_int (stages - 1)));
      status = st;
      memory =
        Nx.Ptree.map2 mem.tree
          (fun _ x z -> Nx.where (Nx.broadcast_to (Nx.shape x) taken) x z)
          memory s.memory;
    }
  in
  let initial =
    {
      v = v0;
      k = k0;
      ys = stacked v0;
      errs = stacked (zeros_like v0);
      acc = zeros_like v0;
      interval = Nx.scalar Nx.int32 0l;
      sigma = Nx.zeros dtype [||];
      h = h_start;
      prev = Nx.full dtype [||] 1e-4;
      rejected = Nx.scalar Nx.bool false;
      ends = Nx.zeros dtype [| budget |];
      counts = Nx.zeros Nx.int32 [| n_int |];
      accepted = Nx.scalar Nx.int32 0l;
      attempts = Nx.scalar Nx.int32 0l;
      evals = Nx.scalar Nx.int32 2l;
      status =
        Nx.where
          (Nx.greater_equal_s disorder 0l)
          (Nx.scalar Nx.int32 (Solution.code Stalled))
          (Nx.scalar Nx.int32 Elementwise.running);
      memory = m0;
    }
  in
  let initial =
    {
      initial with
      status =
        List.fold_left
          (fun st (c, status) -> Elementwise.settle st c status)
          initial.status opening;
    }
  in
  let s =
    Rune.iterate (search_ptree y mem.tree) ~max:budget
      ~until:(fun s -> Nx.logical_not (Elementwise.searching s.status))
      ~f:attempt initial
  in
  (s, disorder)

(* The time the search's carry [s] reached over the times [at], detached. *)
let reached ~at s =
  let n = Nx.dim 0 at in
  let n_int = n - 1 in
  let at0 = Rune.detach at in
  let starts = Nx.slice [ Nx.R (0, n_int) ] at0
  and stops = Nx.slice [ Nx.R (1, n) ] at0 in
  let j =
    Nx.minimum s.interval (Nx.scalar Nx.int32 (Int32.of_int (n_int - 1)))
  in
  let a = scalar_at starts j and b = scalar_at stops j in
  Nx.where
    (Nx.equal_s s.interval (Int32.of_int n_int))
    b
    (Nx.add a (Nx.mul (Nx.sub b a) s.sigma))

(* The answer's report, from the search's carry. *)
let report ?(facts = []) fn m ~tol ~budget ~at (s, disorder) ~value ~error =
  let n = Nx.dim 0 at in
  let dtype = Nx.dtype at in
  let at0 = Rune.detach at in
  let settings =
    Format.asprintf "method %s, tol %a, budget %d" m.name Tol.pp tol budget
  in
  let count c = Nx.cast dtype c in
  Solution.v ~fn ~settings
    ~spent:{ used = s.attempts; unit = "attempted steps"; budget }
    ~fix:(fix tol) ~value ~error ~status:s.status ~evaluations:s.evals
    ~facts:
      ([
         Solution.Fact ("t", reached ~at s);
         Fact ("span", Nx.abs (Nx.sub (Nx.get [ n - 1 ] at0) (Nx.get [ 0 ] at0)));
         Fact ("step", s.h);
         Fact ("accepted", count s.accepted);
         Fact ("rejected", count (Nx.sub s.attempts s.accepted));
         Fact ("disorder", count disorder);
       ]
      @ facts)
    ()

(* The states at the times of [at]: each interval's accepted steps taken again
   with the tracked field, as fractions of the interval, so a moved end
   stretches every step. *)
let samples fn y m mem ~budget ~at y0 s =
  let n = Nx.dim 0 at in
  let n_int = n - 1 in
  let dtype = Nx.dtype at in
  let stages = Array.length m.b in
  let ok = Nx.equal_s s.status (Solution.code Converged) in
  let offsets = Nx.sub (Nx.cumsum ~axis:0 s.counts) s.counts in
  let replay (v, (k, mm)) (a, (b, (offset, count))) =
    let span = Nx.sub b a in
    let inner ((v, (k, mm)), i) =
      let idx = Nx.add offset i in
      let sigma0 =
        Nx.where (Nx.equal_s i 0l) (Nx.zeros dtype [||])
          (scalar_at s.ends
             (Nx.maximum (Nx.sub_s idx 1l) (Nx.scalar Nx.int32 0l)))
      in
      let sigma1 = scalar_at s.ends idx in
      let t = Nx.add a (Nx.mul span sigma0)
      and h = Nx.mul span (Nx.sub sigma1 sigma0) in
      let v', ks = step fn y m (mem.field mm) t h v (Some k) in
      let mm, _ = mem.accept mm ~t ~t_end:(Nx.add t h) ~h ~v ~ks v' in
      ((v', (ks.(stages - 1), mm)), Nx.add_s i 1l)
    in
    let c = Nx.Ptree.(pair y (pair y mem.tree)) in
    let carry, _ =
      Rune.iterate
        Nx.Ptree.(pair c tensor)
        ~max:budget
        ~until:(fun (_, i) -> Nx.greater_equal i count)
        ~f:inner
        ((v, (k, mm)), Nx.scalar Nx.int32 0l)
    in
    (carry, fst carry)
  in
  let c = Nx.Ptree.(pair y (pair y mem.tree)) in
  let replay =
    if n_int = 1 then replay
    else
      let r =
        Rune.remat
          Nx.Ptree.(
            c
            @-> pair tensor (pair tensor (pair tensor tensor))
            @-> returns (pair c y))
          replay
      in
      r
  in
  let m0, _ = mem.start (Nx.get [ 0 ] at) y0 in
  let k_start = eval fn y (mem.field m0) (Nx.get [ 0 ] at) y0 in
  let _, ys =
    Rune.scan c
      Nx.Ptree.(pair tensor (pair tensor (pair tensor tensor)))
      y ~f:replay
      ~init:(y0, (k_start, m0))
      ( Nx.slice [ Nx.R (0, n_int) ] at,
        (Nx.slice [ Nx.R (1, n) ] at, (offsets, s.counts)) )
  in
  let answer =
    Nx.Ptree.map2 y
      (fun _ x0 xs ->
        Nx.concatenate ~axis:0 [ Nx.unsqueeze ~axes:[ 0 ] x0; xs ])
      y0 ys
  in
  let value =
    Nx.Ptree.map2 y
      (fun _ a b -> Nx.where (Nx.broadcast_to (Nx.shape a) ok) a b)
      answer s.ys
  in
  value

let sample y m ~tol ~budget f ~at y0 =
  let fn = "Jera.Ode.sample" in
  check fn ~at ~budget;
  if Nx.dim 0 at = 1 then
    let stack1 v = Nx.Ptree.map y (fun _ x -> Nx.unsqueeze ~axes:[ 0 ] x) v in
    Solution.v ~fn ~settings:"" ~fix:(fix tol) ~value:(stack1 y0)
      ~error:(stack1 (Nx.Ptree.map y (fun _ x -> Nx.zeros_like x) y0))
      ~status:(Nx.scalar Nx.int32 (Solution.code Converged))
      ~evaluations:(Nx.scalar Nx.int32 0l) ~facts:[] ()
  else
    let ((s, _) as found) =
      search fn `Refused y m ~tol ~budget (plain f) ~at y0
    in
    report fn m ~tol ~budget ~at found
      ~value:(samples fn y m (plain f) ~budget ~at y0 s)
      ~error:s.errs

let solve y m ~tol ~budget f ~t0 ~t1 y0 =
  let fn = "Jera.Ode.solve" in
  let at = endpoints fn t0 t1 in
  check fn ~at ~budget;
  let ((s, _) as found) =
    search fn `Allowed y m ~tol ~budget (plain f) ~at y0
  in
  let last v = Nx.Ptree.map y (fun _ x -> Nx.get [ 1 ] x) v in
  report fn m ~tol ~budget ~at found
    ~value:(last (samples fn y m (plain f) ~budget ~at y0 s))
    ~error:(last s.errs)

(* Paths *)

(* The continuous extension's weights at the Chebyshev points of a step:
   [w.(j).(i)] is stage [i]'s [b_i(θ_j)], [θ_j = (u_j + 1) / 2]. *)
let node_weights dense =
  let degree = Array.length dense.(0) in
  Array.map
    (fun u ->
      let theta = (u +. 1.) /. 2. in
      Array.map
        (fun row -> Array.fold_right (fun d acc -> theta *. (d +. acc)) row 0.)
        dense)
    (Cheb.nodes degree)

(* [v + h Σ_i w.(i) k_i]. *)
let combine y h w v ks =
  let acc = ref v in
  Array.iteri
    (fun i wi ->
      if wi <> 0. then acc := Nx.Ptree.axpy y (Nx.mul_s h wi) ks.(i) !acc)
    w;
  !acc

(* Values stacked on a new leading axis of each leaf. *)
let stack y vs =
  let one v = Nx.Ptree.map y (fun _ x -> Nx.unsqueeze ~axes:[ 0 ] x) v in
  Array.fold_left
    (fun acc v ->
      Nx.Ptree.map2 y (fun _ a x -> Nx.concatenate ~axis:0 [ a; x ]) acc (one v))
    (one vs.(0))
    (Array.sub vs 1 (Array.length vs - 1))

let path y m ~tol ~budget f ~t0 ~t1 y0 =
  let fn = "Jera.Ode.path" in
  let at = endpoints fn t0 t1 in
  check fn ~at ~budget;
  let emb = embedded fn m in
  let dtype = Nx.dtype at in
  let s, disorder = search fn `Allowed y m ~tol ~budget (plain f) ~at y0 in
  let span = Nx.sub t1 t0 in
  (* A path over no time has no piece of positive width. *)
  let s =
    {
      s with
      status =
        Nx.where
          (Nx.equal_s (Rune.detach span) 0.)
          (Nx.scalar Nx.int32 (Solution.code Stalled))
          s.status;
    }
  in
  let ok = Nx.equal_s s.status (Solution.code Converged) in
  let w = node_weights emb.dense in
  let stages = Array.length m.b in
  (* Each accepted step taken again with the tracked field, and a step of zero
     length in each slot past them: its continuous extension at the Chebyshev
     points, and the accumulated magnitudes of the local estimates. *)
  let slot (v, (k, (acc, sigma0))) (i, sigma1) =
    let live = Nx.less i s.accepted in
    let sigma1 = Nx.where live sigma1 sigma0 in
    let t = Nx.add t0 (Nx.mul span sigma0)
    and h = Nx.mul span (Nx.sub sigma1 sigma0) in
    let v', ks = step fn y m f t h v (Some k) in
    let nodes = stack y (Array.map (fun wj -> combine y h wj v ks) w) in
    let e =
      combine y h emb.e (Nx.Ptree.map y (fun _ x -> Nx.zeros_like x) v) ks
    in
    let acc =
      Nx.Ptree.map2 y (fun _ a e -> Nx.add a (Rune.detach (Nx.abs e))) acc e
    in
    ((v', (ks.(stages - 1), (acc, sigma1))), (nodes, acc))
  in
  let c = Nx.Ptree.(pair y (pair y (pair y tensor))) in
  let k0 = eval fn y f t0 y0 in
  let zeros = Nx.Ptree.map y (fun _ x -> Nx.zeros_like x) y0 in
  let _, (nodes, accs) =
    Rune.scan c
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.(pair y y)
      ~f:slot
      ~init:(y0, (k0, (zeros, Nx.zeros dtype [||])))
      (Nx.arange Nx.int32 0 budget 1, s.ends)
  in
  (* The breaks as fractions of the span: past the accepted steps, the last
     one's end. *)
  let live =
    Nx.less
      (Nx.arange Nx.int32 0 budget 1)
      (Nx.broadcast_to [| budget |] s.accepted)
  in
  let last =
    Nx.where
      (Nx.greater_s s.accepted 0l)
      (scalar_at s.ends
         (Nx.maximum (Nx.sub_s s.accepted 1l) (Nx.scalar Nx.int32 0l)))
      (Nx.zeros dtype [||])
  in
  let fractions =
    Nx.concatenate ~axis:0
      [
        Nx.zeros dtype [| 1 |];
        Nx.where live s.ends (Nx.broadcast_to [| budget |] last);
      ]
  in
  let breaks = Nx.add t0 (Nx.mul span fractions) in
  (* A path backward in time holds its pieces in increasing time: reversed, each
     piece's points reversed, which is [u ↦ −u]. *)
  let backward = Nx.less_s (Rune.detach span) 0. in
  let orient axes v =
    Nx.where (Nx.broadcast_to (Nx.shape v) backward) (Nx.flip ~axes v) v
  in
  let breaks = orient [ 0 ] breaks in
  let held v = Nx.where (Nx.broadcast_to (Nx.shape v) ok) v (Rune.detach v) in
  let series coefficients =
    { Cheb.s = y; breaks = held breaks; coefficients; extension = Cheb.Bounded }
  in
  let value =
    series
      (Nx.Ptree.map y
         (fun _ x ->
           held (Num.on_float fn { f = Cheb.fit } (orient [ 0; 1 ] x)))
         nodes)
  in
  let error =
    series
      (Nx.Ptree.map y
         (fun _ x -> Nx.unsqueeze ~axes:[ 1 ] (orient [ 0 ] x))
         accs)
  in
  report fn m ~tol ~budget ~at (s, disorder) ~value ~error

(* Events *)

(* The continuous extension at fraction [theta], a scalar, of the step from [v]
   by [h] with stages [ks]: [v + h Σ_i b_i(θ) k_i]. *)
let extension y dense h v ks theta =
  let weight row =
    Array.fold_right
      (fun d acc -> Nx.mul theta (Nx.add_s acc d))
      row (Nx.zeros_like theta)
  in
  let acc = ref v in
  Array.iteri
    (fun i row ->
      if Array.exists (fun d -> d <> 0.) row then
        acc := Nx.Ptree.axpy y (Nx.mul h (weight row)) ks.(i) !acc)
    dense;
  !acc

let event y m ~tol ~budget f ~event ~t0 ~t1 y0 =
  let fn = "Jera.Ode.event" in
  let at = endpoints fn t0 t1 in
  check fn ~at ~budget;
  let emb = embedded fn m in
  let dtype = Nx.dtype at in
  let stages = Array.length m.b in
  (* The event's components, flat, in the time's dtype. *)
  let watch t v = Nx.cast dtype (Nx.reshape [| -1 |] (event t v)) in
  (* The search remembers the components' last non-zero signs at the state and
     before the last accepted step, and ends a lane at the first accepted step
     across which one changes. A zero at a step's end keeps the sign before it,
     so a crossing does not depend on whether a step lands on its zero; a zero
     at [t0] has no sign before it, and is no crossing. *)
  let signs t v = Nx.sign (Rune.detach (watch t v)) in
  let watching =
    {
      tree = Nx.Ptree.(pair tensor tensor);
      field = (fun _ -> f);
      limit = (fun _ -> None);
      start =
        (fun t v ->
          let e = signs t v in
          ((e, e), []));
      accept =
        (fun (now, _) ~t:_ ~t_end ~h:_ ~v:_ ~ks:_ v' ->
          let e = signs t_end v' in
          let e = Nx.where (Nx.equal_s e 0.) now e in
          let crossed =
            Nx.logical_and
              (Nx.not_equal now (Nx.zeros_like now))
              (Nx.not_equal e now)
          in
          ( (e, now),
            [ (Nx.any (Nx.isnan e), Not_finite); (Nx.any crossed, Converged) ]
          ));
    }
  in
  let s, disorder = search fn `Allowed y m ~tol ~budget watching ~at y0 in
  let signs, before = s.memory in
  let n = Nx.dim 0 signs in
  if n = 0 then invalid_arg (fn ^ ": the event has no component");
  let int32 x = Nx.scalar Nx.int32 x in
  let span = Nx.sub t1 t0 in
  let fraction i = scalar_at s.ends (Nx.maximum i (int32 0l)) in
  (* The accepted steps but the last, taken again with the tracked field. *)
  let replay (v, (k, i)) =
    let sigma0 =
      Nx.where (Nx.equal_s i 0l) (Nx.zeros dtype [||])
        (fraction (Nx.sub_s i 1l))
    in
    let sigma1 = fraction i in
    let t = Nx.add t0 (Nx.mul span sigma0)
    and h = Nx.mul span (Nx.sub sigma1 sigma0) in
    let v, ks = step fn y m f t h v (Some k) in
    (v, (ks.(stages - 1), Nx.add_s i 1l))
  in
  let last = Nx.sub_s s.accepted 1l in
  let v_a, (k_a, _) =
    Rune.iterate
      Nx.Ptree.(pair y (pair y tensor))
      ~max:budget
      ~until:(fun (_, (_, i)) -> Nx.greater_equal i last)
      ~f:replay
      (y0, (eval fn y f t0 y0, int32 0l))
  in
  (* The last accepted step, of zero length when there is none: the step that
     holds a crossing. *)
  let sigma0 =
    Nx.where
      (Nx.greater_equal_s s.accepted 2l)
      (fraction (Nx.sub_s s.accepted 2l))
      (Nx.zeros dtype [||])
  and sigma1 =
    Nx.where
      (Nx.greater_equal_s s.accepted 1l)
      (fraction last) (Nx.zeros dtype [||])
  in
  let t_a = Nx.add t0 (Nx.mul span sigma0)
  and h = Nx.mul span (Nx.sub sigma1 sigma0) in
  let v_b, ks = step fn y m f t_a h v_a (Some k_a) in
  let h_safe = Nx.where (Nx.equal_s h 0.) (Nx.ones_like h) h in
  let at_time t =
    extension y emb.dense h v_a ks (Nx.div (Nx.sub t t_a) h_safe)
  in
  (* The crossing, on detached values: each component that changed sign across
     the step is bracketed on the step's piece, and the earliest wins. Its time
     is the bracket's end where the component has its new sign. *)
  let crossed =
    Nx.logical_and
      (Nx.not_equal before (Nx.zeros_like before))
      (Nx.not_equal signs before)
  in
  let crossing = Nx.any crossed in
  let detached v = Nx.Ptree.map y (fun _ x -> Rune.detach x) v in
  let ta = Rune.detach t_a and hd = Rune.detach h_safe in
  let tb = Nx.add ta (Rune.detach h) in
  let vd = detached v_a and ksd = Array.map detached ks in
  let eye =
    Nx.cast dtype
      (Nx.equal
         (Nx.reshape [| n; 1 |] (Nx.arange Nx.int32 0 n 1))
         (Nx.reshape [| 1; n |] (Nx.arange Nx.int32 0 n 1)))
  in
  (* Each component, with a zero on its old side: the crossing is where it takes
     its new sign, so a step that starts on a zero brackets it. *)
  let tiny = Nx.full dtype [||] (Num.tiny dtype) in
  let component ts =
    let all =
      Rune.vmap
        Nx.Ptree.(tensor @-> returns tensor)
        (fun t ->
          watch t (extension y emb.dense hd vd ksd (Nx.div (Nx.sub t ta) hd)))
        ts
    in
    let e = Nx.sum ~axes:[ 1 ] (Nx.mul all eye) in
    Nx.where (Nx.equal_s e 0.) (Nx.mul (Nx.neg signs) tiny) e
  in
  let lo = Nx.broadcast_to [| n |] (Nx.minimum ta tb)
  and hi = Nx.broadcast_to [| n |] (Nx.maximum ta tb) in
  let ((a, b), (fa, _)), st, _ =
    Root.locate ~tol component (lo, component lo) (hi, component hi)
  in
  let found =
    Nx.logical_and crossed (Nx.equal_s st (Solution.code Converged))
  in
  let past = Nx.where (Nx.equal (Nx.sign fa) signs) a b in
  let direction = Nx.sign (Rune.detach span) in
  let key =
    Nx.where found
      (Nx.mul (Nx.sub past ta) direction)
      (Nx.full_like past Float.infinity)
  in
  let c = Nx.cast Nx.int32 (Nx.argmin key) in
  let pick v =
    Nx.reshape [||]
      (Nx.take ~indices:(Nx.reshape [| 1 |] (Nx.cast Nx.int64 c)) v)
  in
  let status =
    Nx.where
      (Nx.logical_and crossing (Nx.logical_not (Nx.any found)))
      (Nx.scalar Nx.int32 (Solution.code Stalled))
      s.status
  in
  let s = { s with status } in
  let ok = Nx.equal_s status (Solution.code Converged) in
  (* The answer: the crossing's time as a zero of the tracked component on the
     tracked piece, or [t1] without a crossing, and the state there. *)
  let onehot =
    Nx.cast dtype
      (Nx.equal (Nx.arange Nx.int32 0 n 1) (Nx.broadcast_to [| n |] c))
  in
  let time =
    Rune.root Nx.Ptree.tensor
      ~residual:(fun t ->
        Nx.where crossing
          (Nx.sum (Nx.mul onehot (watch t (at_time t))))
          (Nx.sub t t1))
      (fun () -> Nx.where crossing (pick past) t1)
  in
  let state =
    Nx.Ptree.map2 y
      (fun _ x z -> Nx.where (Nx.broadcast_to (Nx.shape x) crossing) x z)
      (at_time time) v_b
  in
  let index = Nx.where crossing c (int32 (-1l)) in
  let held v = Nx.where (Nx.broadcast_to (Nx.shape v) ok) v (Rune.detach v) in
  let value = (held time, Nx.Ptree.map y (fun _ x -> held x) state, index) in
  let error =
    ( Nx.where crossing (pick (Nx.div_s (Nx.sub b a) 2.)) (Nx.zeros dtype [||]),
      s.acc,
      int32 0l )
  in
  report fn m ~tol ~budget ~at (s, disorder) ~value ~error

(* Delays *)

(* The integer combinations [k] of [lags] lags with [1 ≤ Σ k ≤ top]: the
   breakpoints [t0 + Σ_j k_j τ_j] where the solution's derivative of order [1 +
   Σ k] may jump. *)
let combinations lags top =
  let rec go j left =
    if j = lags then [ [] ]
    else
      List.concat_map
        (fun k -> List.map (fun rest -> k :: rest) (go (j + 1) (left - k)))
        (List.init (left + 1) Fun.id)
  in
  go 0 top
  |> List.filter (fun k -> List.fold_left ( + ) 0 k >= 1)
  |> List.map (fun k -> Array.of_list (List.map float k))

let delay y m ~tol ~budget ~pieces f ~lags ~history ~at y0 =
  let fn = "Jera.Ode.delay" in
  check fn ~at ~budget;
  if pieces < 1 then
    invalid_arg (Printf.sprintf "%s: pieces = %d is below 1" fn pieces);
  if Nx.ndim lags <> 1 || Nx.dim 0 lags = 0 then
    invalid_arg
      (Printf.sprintf "%s: lags must be 1-D and non-empty, got shape %s" fn
         (Num.shape (Nx.shape lags)));
  let emb = embedded fn m in
  let dtype = Nx.dtype at in
  let n = Nx.dim 0 at in
  if n = 1 then
    let stack1 v = Nx.Ptree.map y (fun _ x -> Nx.unsqueeze ~axes:[ 0 ] x) v in
    Solution.v ~fn ~settings:"" ~fix:(fix tol) ~value:(stack1 y0)
      ~error:(stack1 (Nx.Ptree.map y (fun _ x -> Nx.zeros_like x) y0))
      ~status:(Nx.scalar Nx.int32 (Solution.code Converged))
      ~evaluations:(Nx.scalar Nx.int32 0l) ~facts:[] ()
  else
    let degree = Array.length emb.dense.(0) in
    let t_first = Nx.get [ 0 ] at and t_last = Nx.get [ n - 1 ] at in
    let smallest = Nx.min (Rune.detach lags)
    and largest = Nx.max (Rune.detach lags) in
    let int32 x = Nx.scalar Nx.int32 x in
    let n_lags = Nx.dim 0 lags in
    let past s =
      let v = history s in
      if layout y v <> layout y y0 then
        invalid_arg
          (fn
         ^ ": history returned a value of another structure, dtype or shape \
            than the state");
      v
    in
    (* The memory: the last [pieces] accepted steps' pieces in a ring, each its
       start, width and coefficients [c_q] of [y = Σ_q c_q θ^q] on [θ ∈ [0, 1]],
       leaves of shape [[pieces; degree + 1] @ value], and the count of steps
       recorded. *)
    let tree = Nx.Ptree.(pair tensor (pair tensor (pair tensor y))) in
    let slot_of k =
      Nx.mod_s (Nx.add_s k (Int32.of_int pieces)) (Int32.of_int pieces)
    in
    let lookup (starts, (widths, (count, coef))) s =
      let logical = Nx.arange Nx.int32 0 pieces 1 in
      let k =
        Nx.add
          (Nx.sub_s (Nx.broadcast_to [| pieces |] count) (Int32.of_int pieces))
          logical
      in
      let slots = Nx.cast Nx.int64 (slot_of k) in
      let ordered =
        Nx.where (Nx.greater_equal_s k 0l)
          (Nx.take ~indices:slots starts)
          (Nx.full_like starts Float.neg_infinity)
      in
      let i =
        Nx.maximum
          (Nx.sub_s (Nx.searchsorted ~side:`Right ordered s) 1L)
          (Nx.zeros Nx.int64 (Nx.shape s))
      in
      let slot = Nx.take ~indices:i slots in
      let start = Nx.take ~indices:slot starts
      and width = Nx.take ~indices:slot widths in
      let width = Nx.where (Nx.equal_s width 0.) (Nx.ones_like width) width in
      let theta = Nx.div (Nx.sub s start) width in
      let pieces =
        Nx.Ptree.map y
          (fun _ c ->
            let g = Nx.take ~axis:0 ~indices:slot c in
            let theta =
              Nx.reshape
                (Array.append [| Nx.dim 0 s |] (Array.make (Nx.ndim g - 2) 1))
                (Nx.cast (Nx.dtype c) theta)
            in
            let coefficient q = Nx.slice [ Nx.A; Nx.I q ] g in
            let acc = ref (coefficient degree) in
            for q = degree - 1 downto 0 do
              acc := Nx.add (Nx.mul !acc theta) (coefficient q)
            done;
            !acc)
          coef
      in
      let before = Nx.less_equal s t_first in
      let s = Nx.minimum s (Nx.broadcast_to (Nx.shape s) t_first) in
      let past = stack y (Array.init n_lags (fun i -> past (Nx.get [ i ] s))) in
      Nx.Ptree.map2 y
        (fun _ h p ->
          let mask =
            Nx.reshape
              (Array.append (Nx.shape before) (Array.make (Nx.ndim h - 1) 1))
              before
          in
          Nx.where (Nx.broadcast_to (Nx.shape h) mask) h p)
        past pieces
    in
    (* Whether the ring misses a time the next step from [t] reads: its stages
       read back to [t − τ_max] and, once [t] reaches the breakpoint [t0 +
       τ_min], past [t0], where only pieces hold the state. Then [needed] is the
       count of pieces that would cover the reads at the ring's mean step. *)
    let short ~t (starts, (_, (count, _))) =
      let first = Rune.detach t_first in
      let oldest = scalar_at starts (slot_of count) in
      Nx.logical_and
        (Nx.greater_equal_s count (Int32.of_int pieces))
        (Nx.logical_and
           (Nx.greater_equal t (Nx.add first smallest))
           (Nx.greater oldest (Nx.maximum (Nx.sub t largest) first)))
    in
    let needed ~t (starts, (_, (count, _))) =
      let oldest = scalar_at starts (slot_of count) in
      let reads =
        Nx.sub t (Nx.maximum (Nx.sub t largest) (Rune.detach t_first))
      in
      Nx.ceil (Nx.div (Nx.mul_s reads (float pieces)) (Nx.sub t oldest))
    in
    let memory =
      {
        tree;
        start =
          (fun _ v ->
            let coef =
              Nx.Ptree.map y
                (fun _ x ->
                  Nx.zeros (Nx.dtype x)
                    (Array.append [| pieces; degree + 1 |] (Nx.shape x)))
                v
            in
            ( ( Nx.zeros dtype [| pieces |],
                (Nx.ones dtype [| pieces |], (int32 0l, coef)) ),
              [
                ( Nx.logical_or
                    (Nx.less_equal_s smallest 0.)
                    (Nx.logical_not (Nx.all (Nx.isfinite (Rune.detach lags)))),
                  Stalled );
              ] ));
        field = (fun mm t v -> f t v (lookup mm (Nx.sub t lags)));
        limit = (fun _ -> Some smallest);
        accept =
          (fun (starts, (widths, (count, coef))) ~t ~t_end ~h ~v ~ks _ ->
            let column q = Array.map (fun row -> row.(q)) emb.dense in
            let zeros = Nx.Ptree.map y (fun _ x -> Nx.zeros_like x) v in
            let piece =
              stack y
                (Array.append [| v |]
                   (Array.init degree (fun q -> combine y h (column q) zeros ks)))
            in
            let at_slot =
              Nx.equal
                (Nx.arange Nx.int32 0 pieces 1)
                (Nx.broadcast_to [| pieces |] (slot_of count))
            in
            let put rows x =
              let mask =
                Nx.reshape
                  (Array.append [| pieces |] (Array.make (Nx.ndim rows - 1) 1))
                  at_slot
              in
              Nx.where
                (Nx.broadcast_to (Nx.shape rows) mask)
                (Nx.broadcast_to (Nx.shape rows) (Nx.unsqueeze ~axes:[ 0 ] x))
                rows
            in
            let starts = put starts t and widths = put widths h in
            let count = Nx.add_s count 1l in
            ( ( starts,
                ( widths,
                  (count, Nx.Ptree.map2 y (fun _ r x -> put r x) coef piece) )
              ),
              [ (short ~t:t_end (starts, (widths, (count, coef))), Stalled) ] ));
      }
    in
    (* Steps land on the breakpoints: the times of [at] and the breakpoints in
       their span, sorted, and the states read back at [at]'s positions. *)
    let k = combinations n_lags (emb.order - 1) in
    let merged =
      match k with
      | [] -> at
      | k ->
          let nb = List.length k in
          let kk =
            Nx.reshape [| nb; n_lags |] (Num.constant dtype (Array.concat k))
          in
          let points = Nx.add t_first (Nx.matmul kk lags) in
          let points =
            Nx.minimum
              (Nx.maximum points (Nx.broadcast_to [| nb |] t_first))
              (Nx.broadcast_to [| nb |] t_last)
          in
          Nx.concatenate ~axis:0 [ at; points ]
    in
    let order = Nx.argsort (Rune.detach merged) in
    let sorted = Nx.take ~indices:order merged in
    let rows = Nx.slice [ Nx.R (0, n) ] (Nx.argsort order) in
    let disorder =
      Nx.where
        (Nx.less (Rune.detach t_last) (Rune.detach t_first))
        (int32 1l)
        (disorder `Refused (Rune.detach at))
    in
    let memory =
      {
        memory with
        start =
          (fun t v ->
            let mm, out = memory.start t v in
            (mm, (Nx.greater_equal_s disorder 0l, Stalled) :: out));
      }
    in
    let s, _ = search fn `Allowed y m ~tol ~budget memory ~at:sorted y0 in
    let pick v =
      Nx.Ptree.map y (fun _ x -> Nx.take ~axis:0 ~indices:rows x) v
    in
    let t = reached ~at:sorted s in
    report fn m ~tol ~budget ~at:sorted (s, disorder)
      ~facts:
        [
          Fact ("lags not positive", Nx.cast dtype (Nx.less_equal_s smallest 0.));
          Fact ("beyond pieces", Nx.cast dtype (short ~t s.memory));
          Fact ("largest lag", largest);
          Fact ("pieces", Nx.full dtype [||] (float pieces));
          Fact ("pieces needed", needed ~t s.memory);
        ]
      ~value:(pick (samples fn y m memory ~budget ~at:sorted y0 s))
      ~error:(pick s.errs)
