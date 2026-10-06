(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Methods *)

(* An embedded error estimate: the weights [e = b − b̂] of the difference between
   the step and its embedded formula, and the controller's order, one more than
   the lower of the two orders. *)
type embedded = { e : float array; order : int }

(* An explicit Butcher tableau: row [i] of [a] has [i] elements. [fsal] when the
   last stage evaluates the field at the step's result, so the next step starts
   from it. *)
type tableau = {
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

let make ?embedded ~a ~b ~c () =
  { a; b; c; fsal = first_same_as_last a b c; embedded }

let euler = make ~a:[| [||] |] ~b:[| 1. |] ~c:[| 0. |] ()

let rk4 =
  make
    ~a:[| [||]; [| 0.5 |]; [| 0.; 0.5 |]; [| 0.; 0.; 1. |] |]
    ~b:[| 1. /. 6.; 1. /. 3.; 1. /. 3.; 1. /. 6. |]
    ~c:[| 0.; 0.5; 0.5; 1. |] ()

let ssprk3 =
  make
    ~a:[| [||]; [| 1. |]; [| 0.25; 0.25 |] |]
    ~b:[| 1. /. 6.; 1. /. 6.; 2. /. 3. |]
    ~c:[| 0.; 1.; 0.5 |] ()

let bs3 =
  make
    ~embedded:
      { e = [| -5. /. 72.; 1. /. 12.; 1. /. 9.; -1. /. 8. |]; order = 3 }
    ~a:[| [||]; [| 0.5 |]; [| 0.; 0.75 |]; [| 2. /. 9.; 1. /. 3.; 4. /. 9. |] |]
    ~b:[| 2. /. 9.; 1. /. 3.; 4. /. 9.; 0. |]
    ~c:[| 0.; 0.5; 0.75; 1. |] ()

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
  make
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

let dopri5 =
  let b =
    [|
      35. /. 384.; 0.; 500. /. 1113.; 125. /. 192.; -2187. /. 6784.; 11. /. 84.;
    |]
  in
  make
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
  make ~a:(Array.map Array.copy a) ~b:(Array.copy b) ~c:(Array.copy c) ()

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

(* The search's carry. *)
type ('y, 't) search = {
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
}

let search_ptree (type v t) (y : v Nx.Ptree.t) : (v, t) search Nx.Ptree.t =
  let module M = struct
    type nonrec _ t = (v, t) search

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
      }
  end in
  Nx.Ptree.instantiate (module M)

let scalar_at t i =
  Nx.reshape [||] (Nx.take ~indices:(Nx.reshape [| 1 |] (Nx.cast Nx.int64 i)) t)

let sample y m ~tol ~budget f ~at y0 =
  let fn = "Jera.Ode.sample" in
  let emb = match m.embedded with Some e -> e | None -> assert false in
  March.check fn ~steps:1 at;
  if budget < 1 then
    invalid_arg (Printf.sprintf "%s: budget = %d is below 1" fn budget);
  let n = Nx.dim 0 at in
  let dtype = Nx.dtype at in
  let n_int = n - 1 in
  let stack1 v = Nx.Ptree.map y (fun _ x -> Nx.unsqueeze ~axes:[ 0 ] x) v in
  if n_int = 0 then
    Solution.v ~fn ~settings:"" ~value:(stack1 y0)
      ~error:(stack1 (Nx.Ptree.map y (fun _ x -> Nx.zeros_like x) y0))
      ~status:(Nx.scalar Nx.int32 (Solution.code Converged))
      ~evaluations:(Nx.scalar Nx.int32 0l) ~facts:[]
  else begin
    let at0 = Rune.detach at in
    let starts = Nx.slice [ Nx.R (0, n_int) ] at0
    and stops = Nx.slice [ Nx.R (1, n) ] at0 in
    let detached v = Nx.Ptree.map y (fun _ x -> Rune.detach x) v in
    let fd t v = detached (eval fn y f t v) in
    let v0 = detached y0 in
    let t0 = Nx.get [ 0 ] at0 in
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
      Nx.where tiny fallback (Nx.div (Nx.mul_s d0 0.01) (Nx.where tiny one d1))
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
    let usable =
      Nx.logical_and (Nx.isfinite h_start) (Nx.greater_s h_start 0.)
    in
    let h_start = Nx.where usable h_start fallback in
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
      let j =
        Nx.minimum s.interval (Nx.scalar Nx.int32 (Int32.of_int (n_int - 1)))
      in
      let a = scalar_at starts j and b = scalar_at stops j in
      let span = Nx.sub b a in
      let t = Nx.add a (Nx.mul span s.sigma) in
      let ds = Nx.div s.h (Nx.abs span) in
      let lands = Nx.greater_equal (Nx.add s.sigma ds) one in
      let ds = Nx.where lands (Nx.rsub_s 1. s.sigma) ds in
      let sigma_end = Nx.where lands one (Nx.add s.sigma ds) in
      let hh = Nx.mul span ds in
      let v', ks =
        step fn y m (fun t v -> detached (f t v)) t hh s.v (Some s.k)
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
      let ok = Nx.logical_and running (Nx.less_equal_s r 1.) in
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
      let h = Nx.where ok (Nx.mul s.h fac_ok) (Nx.mul s.h fac_bad) in
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
          (Nx.logical_and slot (Nx.broadcast_to [| budget |] ok))
          (Nx.broadcast_to [| budget |] sigma_end)
          s.ends
      in
      let at_j =
        Nx.equal (Nx.arange Nx.int32 0 n_int 1) (Nx.broadcast_to [| n_int |] j)
      in
      let counts =
        Nx.add s.counts
          (Nx.cast Nx.int32
             (Nx.logical_and at_j (Nx.broadcast_to [| n_int |] ok)))
      in
      let interval = Nx.where finishing (Nx.add_s s.interval 1l) s.interval in
      let st = s.status in
      let st =
        Elementwise.settle st
          (Nx.logical_and ok (Nx.logical_not (finite y k')))
          Not_finite
      in
      let st =
        Elementwise.settle st
          (Nx.logical_and finishing (Nx.equal_s interval (Int32.of_int n_int)))
          Converged
      in
      let st = Elementwise.settle st (Nx.equal t_end t) Stalled in
      let attempts = Nx.add s.attempts (Nx.cast Nx.int32 running) in
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
          Nx.where ok
            (Nx.where lands (Nx.zeros_like sigma_end) sigma_end)
            s.sigma;
        h = Nx.where running h s.h;
        prev = Nx.where ok (Nx.maximum r (Nx.full_like r 1e-4)) s.prev;
        rejected = Nx.logical_and running (Nx.logical_not ok);
        ends;
        counts;
        accepted = Nx.add s.accepted (Nx.cast Nx.int32 ok);
        attempts;
        evals =
          Nx.add s.evals
            (Nx.mul_s (Nx.cast Nx.int32 running) (Int32.of_int (stages - 1)));
        status = st;
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
        status = Nx.scalar Nx.int32 Elementwise.running;
      }
    in
    let s =
      Rune.iterate (search_ptree y) ~max:budget
        ~until:(fun s -> Nx.logical_not (Elementwise.searching s.status))
        ~f:attempt initial
    in
    let ok = Nx.equal_s s.status (Solution.code Converged) in
    (* The answer: each interval's accepted steps taken again with the tracked
       field, as fractions of the interval, so a moved end stretches every
       step. *)
    let offsets = Nx.sub (Nx.cumsum ~axis:0 s.counts) s.counts in
    let replay (v, k) (a, (b, (offset, count))) =
      let span = Nx.sub b a in
      let inner (v, (k, i)) =
        let idx = Nx.add offset i in
        let sigma0 =
          Nx.where (Nx.equal_s i 0l) (Nx.zeros dtype [||])
            (scalar_at s.ends
               (Nx.maximum (Nx.sub_s idx 1l) (Nx.scalar Nx.int32 0l)))
        in
        let sigma1 = scalar_at s.ends idx in
        let t = Nx.add a (Nx.mul span sigma0)
        and h = Nx.mul span (Nx.sub sigma1 sigma0) in
        let v, ks = step fn y m f t h v (Some k) in
        (v, (ks.(stages - 1), Nx.add_s i 1l))
      in
      let v, (k, _) =
        Rune.iterate
          Nx.Ptree.(pair y (pair y tensor))
          ~max:budget
          ~until:(fun (_, (_, i)) -> Nx.greater_equal i count)
          ~f:inner
          (v, (k, Nx.scalar Nx.int32 0l))
      in
      ((v, k), v)
    in
    let c = Nx.Ptree.pair y y in
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
    let k_start = eval fn y f (Nx.get [ 0 ] at) y0 in
    let _, ys =
      Rune.scan c
        Nx.Ptree.(pair tensor (pair tensor (pair tensor tensor)))
        y ~f:replay ~init:(y0, k_start)
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
    (* Where the search stopped, for the report. *)
    let j =
      Nx.minimum s.interval (Nx.scalar Nx.int32 (Int32.of_int (n_int - 1)))
    in
    let a = scalar_at starts j and b = scalar_at stops j in
    let stopped =
      Nx.where
        (Nx.equal_s s.interval (Int32.of_int n_int))
        b
        (Nx.add a (Nx.mul (Nx.sub b a) s.sigma))
    in
    Solution.v ~fn
      ~settings:(Format.asprintf "tol %a, budget %d" Tol.pp tol budget)
      ~value ~error:s.errs ~status:s.status ~evaluations:s.evals
      ~facts:[ Fact ("t", stopped); Fact ("step", s.h) ]
  end

let solve y m ~tol ~budget f ~t0 ~t1 y0 =
  let at =
    Nx.concatenate ~axis:0 [ Nx.reshape [| 1 |] t0; Nx.reshape [| 1 |] t1 ]
  in
  let last v = Nx.Ptree.map y (fun _ x -> Nx.get [ 1 ] x) v in
  Solution.map ~fn:"Jera.Ode.solve" last (sample y m ~tol ~budget f ~at y0)
