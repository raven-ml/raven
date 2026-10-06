(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { kick : float array; drift : float array }

let palindromic a =
  let n = Array.length a in
  let rec go i = i >= n / 2 || (a.(i) = a.(n - 1 - i) && go (i + 1)) in
  go 0

(* A sum of [n] floats is within [n * eps * Σ |a_i|] of its exact value. *)
let sums_to_one a =
  let sum = Array.fold_left ( +. ) 0. a in
  let size = Array.fold_left (fun acc x -> acc +. Float.abs x) 0. a in
  Float.abs (sum -. 1.) <= float (Array.length a) *. epsilon_float *. size

let v ~kick ~drift =
  let fail fmt =
    Printf.ksprintf (fun m -> invalid_arg ("Jera.Split.v: " ^ m)) fmt
  in
  let finite = Array.for_all Float.is_finite in
  if not (finite kick && finite drift) then fail "a coefficient is not finite";
  if Array.length drift = 0 then fail "drift is empty";
  if Array.length kick <> Array.length drift + 1 then
    fail "kick has %d elements and drift %d; kick needs one more"
      (Array.length kick) (Array.length drift);
  if not (sums_to_one kick) then fail "kick does not sum to 1";
  if not (sums_to_one drift) then fail "drift does not sum to 1";
  if not (palindromic kick && palindromic drift) then
    fail "the sequence is not palindromic";
  { kick = Array.copy kick; drift = Array.copy drift }

(* Leapfrogs of durations [w h] in sequence: each drifts by [w_i] between two
   half kicks, and adjacent half kicks add. *)
let leapfrogs w =
  let n = Array.length w in
  let weight i = if i < 0 || i >= n then 0. else w.(i) in
  let kick = Array.init (n + 1) (fun i -> (weight (i - 1) +. weight i) /. 2.) in
  { kick; drift = Array.copy w }

(* Yoshida's weights [w_1; ...; w_m] compose the leapfrogs [w_m ... w_1 w_0 w_1
   ... w_m], with [w_0 = 1 − 2 Σ w_i]. *)
let yoshida ws =
  let m = Array.length ws in
  let w0 = 1. -. (2. *. Array.fold_left ( +. ) 0. ws) in
  let weight i = if i = 0 then w0 else ws.(i - 1) in
  leapfrogs (Array.init ((2 * m) + 1) (fun i -> weight (abs (m - i))))

let leapfrog = leapfrogs [| 1. |]

let mclachlan =
  let lambda = 0.1931833275037836 in
  { kick = [| lambda; 1. -. (2. *. lambda); lambda |]; drift = [| 0.5; 0.5 |] }

let yoshida4 =
  let cbrt2 = Float.cbrt 2. in
  yoshida [| 1. /. (2. -. cbrt2) |]

let yoshida6 =
  yoshida [| -1.17767998417887; 0.235573213359357; 0.784513610477560 |]

let yoshida8 =
  yoshida
    [|
      0.102799849391985;
      -1.96061023297549;
      1.93813913762276;
      -0.158240635368243;
      -1.44485223686048;
      0.253693336566229;
      0.914844246229740;
    |]

type ('s, 'b) flow = (float, 'b) Nx.t -> 's -> 's

(* The drifts and inner kicks of one step, without its end kicks. *)
let inner m ~kick ~drift h s =
  let s = ref (drift (Nx.mul_s h m.drift.(0)) s) in
  for i = 1 to Array.length m.drift - 1 do
    s := drift (Nx.mul_s h m.drift.(i)) (kick (Nx.mul_s h m.kick.(i)) !s)
  done;
  !s

let step m ~kick ~drift h s =
  let last = Array.length m.kick - 1 in
  let s = inner m ~kick ~drift h (kick (Nx.mul_s h m.kick.(0)) s) in
  kick (Nx.mul_s h m.kick.(last)) s

let march p m ~steps ~kick ~drift ~at s0 =
  March.check "Jera.Split.march" ~steps at;
  let last = Array.length m.kick - 1 in
  let merged = m.kick.(last) +. m.kick.(0) in
  let interval t0 t1 s =
    let h = Nx.div_s (Nx.sub t1 t0) (float steps) in
    let s = kick (Nx.mul_s h m.kick.(0)) s in
    let s =
      March.steps p (Nx.dtype at) (steps - 1)
        (fun _ s -> kick (Nx.mul_s h merged) (inner m ~kick ~drift h s))
        s
    in
    kick (Nx.mul_s h m.kick.(last)) (inner m ~kick ~drift h s)
  in
  March.run p p ~at ~interval ~state:Fun.id s0
