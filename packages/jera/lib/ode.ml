(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Methods *)

(* An explicit Butcher tableau: row [i] of [a] has [i] elements. [fsal] when the
   last stage evaluates the field at the step's result, so the next step starts
   from it. *)
type tableau = {
  a : float array array;
  b : float array;
  c : float array;
  fsal : bool;
}

type (-'k, 'y, 't) t = tableau

let first_same_as_last a b c =
  let s = Array.length b in
  s > 1
  && c.(s - 1) = 1.
  && b.(s - 1) = 0.
  && Array.for_all2 ( = ) a.(s - 1) (Array.sub b 0 (s - 1))

let make ~a ~b ~c = { a; b; c; fsal = first_same_as_last a b c }
let euler = make ~a:[| [||] |] ~b:[| 1. |] ~c:[| 0. |]

let rk4 =
  make
    ~a:[| [||]; [| 0.5 |]; [| 0.; 0.5 |]; [| 0.; 0.; 1. |] |]
    ~b:[| 1. /. 6.; 1. /. 3.; 1. /. 3.; 1. /. 6. |]
    ~c:[| 0.; 0.5; 0.5; 1. |]

let ssprk3 =
  make
    ~a:[| [||]; [| 1. |]; [| 0.25; 0.25 |] |]
    ~b:[| 1. /. 6.; 1. /. 6.; 2. /. 3. |]
    ~c:[| 0.; 1.; 0.5 |]

let bs3 =
  make
    ~a:[| [||]; [| 0.5 |]; [| 0.; 0.75 |]; [| 2. /. 9.; 1. /. 3.; 4. /. 9. |] |]
    ~b:[| 2. /. 9.; 1. /. 3.; 4. /. 9.; 0. |]
    ~c:[| 0.; 0.5; 0.75; 1. |]

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

let dopri5 =
  let b =
    [|
      35. /. 384.; 0.; 500. /. 1113.; 125. /. 192.; -2187. /. 6784.; 11. /. 84.;
    |]
  in
  make
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
  make ~a:(Array.map Array.copy a) ~b:(Array.copy b) ~c:(Array.copy c)

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
   method reuses its last stage; the new state and the field there. *)
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
  ((if m.fsal then !last else combine m.b), ks.(s - 1))

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
    let step t h (v, k) = step fn y m f t h v (Some k) in
    let k0 = eval fn y f (Nx.get [ 0 ] at) y0 in
    March.run c y ~at ~interval:(interval c step) ~state:fst (y0, k0)
  else
    let step t h v = fst (step fn y m f t h v None) in
    March.run y y ~at ~interval:(interval y step) ~state:Fun.id y0
