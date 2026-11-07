(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.attention under rune's transformations, drawn over dense and paged
   keys, lengths, grouped heads, and masks and scores that capture arrays:
   - Rune.jit against the eager kernel, each within nx.mli's bound of the
     exact result, rune.mli's constants for the compiled one, and against a
     float64 reference at every dtype, a sum of zeros +0 (Attentions);
   - Rune.vmap against each lane eagerly, bit for bit: a vmapped axis is a
     lead axis of the one operation, and nx.cpu's rows are their own;
   - Rune.jvp against central differences, and Rune.grad against them;
   - Rune.grad compiled against eager, and forward mode against reverse
     mode, through the derivative's construct and its transpose;
   - Rune.jit over keys cut across two devices against one device.
   Each in float32 and in float64, accumulating in the query's dtype, and
   for a float32 query accumulating in float64.
   Values are finite: a NaN or an infinity at a hidden key gives the
   compiled and differentiated results NaN where eager's is finite, as
   nx.mli and rune.mli state. *)

open Windtrap
module T = Nx.Ptree

let numel = Array.fold_left ( * ) 1

(* A dtype's unit roundoff, [u] in the bounds below, and its [λ + 2], the
   factor of [ln tk] in nx.cpu's [ε] (nx_cpu.mli). *)
let unit_roundoff (type s) (dt : (float, s) Nx.dtype) =
  match dt with Float64 -> 0x1p-53 | _ -> 0x1p-24

let log_factor (type s) (dt : (float, s) Nx.dtype) =
  match dt with Float64 -> 7. | _ -> 5.

type mask = Unmasked | Causal | Captured

type case = {
  hk : int;
  groups : int;
  tq : int;
  tk : int;
  dk : int;
  dv : int;
  page : int option;  (** Paged by pages of this many keys. *)
  table : int array;  (** Two lead indices' entries in [[-1, 3]]. *)
  lengths : int array option;
  mask : mask;
  alibi : bool;
  seed : float;
}

let pp_case ppf c =
  Format.fprintf ppf
    "hk %d groups %d tq %d tk %d dk %d dv %d page %s lengths %s mask %s \
     alibi %b seed %g"
    c.hk c.groups c.tq c.tk c.dk c.dv
    (match c.page with Some p -> string_of_int p | None -> "none")
    (match c.lengths with
    | Some l -> String.concat "," (Array.to_list (Array.map string_of_int l))
    | None -> "none")
    (match c.mask with
    | Unmasked -> "none"
    | Causal -> "causal"
    | Captured -> "captured")
    c.alibi c.seed

let case =
  let open Gen in
  let* hk = int_range 1 2 in
  let* groups = int_range 1 2 in
  let* tq = int_range 1 4 in
  let* page =
    frequency [ (2, constant None); (1, map Option.some (int_range 1 3)) ]
  in
  let* blocks = int_range 1 4 in
  let tk = match page with Some p -> p * blocks | None -> blocks * 3 in
  let* table = array ~size:(constant (2 * blocks)) (int_range (-1) 3) in
  let* lengths =
    frequency
      [
        (1, constant None);
        (1, map Option.some (array ~size:(constant 2) (int_range 0 (tk + 1))));
      ]
  in
  let* mask = of_list [ Unmasked; Causal; Captured ] in
  let* alibi = bool in
  let* dk = int_range 1 4 in
  let* dv = int_range 1 3 in
  let+ seed = float_range 0. 10. in
  { hk; groups; tq; tk; dk; dv; page; table; lengths; mask; alibi; seed }

let draw dt c seed s =
  Nx.create dt s
    (Array.init (numel s) (fun k ->
         Float.sin ((Float.of_int k *. 2.3) +. seed +. c.seed)))

let hq c = c.hk * c.groups

(* [c]'s call on [q] of [lead] axes, its keys and values [k] and [v] and
   its captures drawn from [c], its table's and lengths' entries from the
   first [lead], accumulating in [acc]. *)
let call (type a) (acc : (float, a) Nx.dtype) c ~lead
    (q : (float, 's, Nx.host) Nx.t) k v =
  let dt = Nx.dtype q in
  let hq = hq c in
  let n = numel lead in
  let lengths =
    Option.map
      (fun l -> Nx.create Nx.int64 lead (Array.map Int64.of_int (Array.sub l 0 n)))
      c.lengths
  in
  let mask =
    match c.mask with
    | Unmasked -> None
    | Causal -> Some Nx.causal
    | Captured ->
        let m = Nx.greater_equal_s (draw dt c 7. [| hq; c.tq; c.tk |]) (-0.5) in
        Some (fun _ -> m)
  in
  let score =
    if not c.alibi then None
    else
      let slopes = Nx.abs (draw acc c 8. [| hq; 1; 1 |]) in
      Some
        (fun (p : Nx.host Nx.pair) s ->
          Nx.(Infix.(s - (cast acc (p.q - p.k) * slopes))))
  in
  let kv =
    match c.page with
    | None -> Nx.Dense { k; v }
    | Some p ->
        let pages = c.tk / p in
        let table = Array.map Int64.of_int (Array.sub c.table 0 (n * pages)) in
        Nx.Paged { k; v; table = Nx.create Nx.int64 (Array.append lead [| pages |]) table }
  in
  Nx.attention ?lengths ?mask ?score ~acc ~kv q

(* Keys and values for [lead]: dense ones of its axes, or pools of three
   pages. *)
let keys dt c lead =
  match c.page with
  | None ->
      ( draw dt c 2. (Array.append lead [| c.hk; c.tk; c.dk |]),
        draw dt c 3. (Array.append lead [| c.hk; c.tk; c.dv |]) )
  | Some p ->
      (draw dt c 2. [| 3; c.hk; p; c.dk |], draw dt c 3. [| 3; c.hk; p; c.dv |])

(* The sum of the bounds nx.mli states for two computations of [c] in
   [acc], to first order in its [u], of a query of [dt]: one with nx.cpu's [ε],
   [(13n + 9 + (λ + 2) ln tk) u + u |lse|] and its exp of 5u, and one with
   rune.mli's compiled [Nx.logsumexp],
   [(1.37 (tk - 1) + 2E + 1 + 3 ln tk) u + u |lse|] for [E = 9] ulps, and
   its exp of 9 ulps, 19u, in [w]. Values lie in [[-1, 1]], so a score is
   at most [a = |scale| dk] plus ALiBi's slopes times [tq + tk], [b], and
   [η] at most [(2 dk + 4) u a + 2u b]. *)
let tolerances dt acc c lse =
  let u = unit_roundoff acc and uq = unit_roundoff dt in
  let scale = 1. /. Float.sqrt (Float.of_int c.dk) in
  let a = scale *. Float.of_int c.dk in
  let b = a +. Float.of_int (c.tq + c.tk) in
  let eta = (Float.of_int ((2 * c.dk) + 4) *. u *. a) +. (2. *. u *. b) in
  let ln = Float.log (Float.of_int (max 1 c.tk)) in
  let n = (c.tk + 63) / 64 in
  let kernel =
    ((Float.of_int ((13 * n) + 9) +. (log_factor acc *. ln)) *. u)
    +. (u *. Float.abs lse)
  in
  let compiled =
    (((1.37 *. Float.of_int (c.tk - 1)) +. 19. +. (3. *. ln)) *. u)
    +. (u *. Float.abs lse)
  in
  let w ~exp eps = (2. *. eta) +. eps +. (((2. *. b) +. ln +. exp) *. u) in
  let gamma = Float.of_int (c.tk + 1) *. 2. *. u in
  (* Each [o], at most 1, rounds once into [dt]: [u{_q} |o|] each. *)
  ( (2. *. eta) +. kernel +. compiled,
    w ~exp:5. kernel +. w ~exp:19. compiled +. (2. *. gamma) +. (2. *. uq) )

(* A case that reaches every cover of each law on every run: paged, with
   lengths and a captured mask, which [law_cut] makes causal. *)
let examples =
  [
    {
      hk = 1;
      groups = 2;
      tq = 2;
      tk = 6;
      dk = 3;
      dv = 2;
      page = Some 2;
      table = [| 0; 1; -1; 2; 3; 0 |];
      lengths = Some [| 3; 6 |];
      mask = Captured;
      alibi = true;
      seed = 1.5;
    };
  ]

let law_jit dt acc c =
  cover "paged" (c.page <> None);
  cover "lengths" (c.lengths <> None);
  cover "a captured mask" (c.mask = Captured);
  let lead = [| 2 |] in
  let k, v = keys dt c lead in
  let q = draw dt c 1. [| 2; hq c; c.tq; c.dk |] in
  let o, lse = call acc c ~lead q k v in
  let o' = Rune.jit' (fun q -> fst (call acc c ~lead q k v)) q in
  let lse' = Rune.jit' (fun q -> snd (call acc c ~lead q k v)) q in
  let o = Nx.to_array o and o' = Nx.to_array o' in
  let lse = Nx.to_array lse and lse' = Nx.to_array lse' in
  Array.iteri
    (fun i l ->
      if l = Float.neg_infinity then
        equal ~msg:"an empty row's lse" float_exact l lse'.(i)
      else begin
        let tl, to_ = tolerances dt acc c l in
        at_most ~msg:"lse" float_exact ~than:tl (Float.abs (l -. lse'.(i)));
        for d = 0 to c.dv - 1 do
          let j = (i * c.dv) + d in
          at_most ~msg:"o" float_exact ~than:to_ (Float.abs (o.(j) -. o'.(j)))
        done
      end)
    lse

let bits a = Array.map Int64.bits_of_float (Nx.to_array a)

(* Keys and values cut across two devices along their keys, Rune.jit
   against the eager call on one device, and the gradient of [Σ o] within
   1e-3 relative of eager's: each part's coordinates are the whole's, so a
   causal or captured mask hides the same pairs. nx.mli's bound
   over parts doubles [η] and takes [ε] three times the parts' plus a
   Logsumexp of two: within four times the one-device sum of bounds. *)
module Two = (val Nx.devices [ Nx_support.memory 0; Nx_support.memory 1 ])

let law_cut dt acc c =
  cover "lengths" (c.lengths <> None);
  let c = { c with page = None; tk = 2 * c.tk; alibi = false } in
  let c = if c.mask = Captured then { c with mask = Causal } else c in
  cover "causal" (c.mask = Causal);
  let lead = [| 2 |] in
  let k, v = keys dt c lead in
  let q = draw dt c 1. [| 2; hq c; c.tq; c.dk |] in
  let o, lse = call acc c ~lead q k v in
  let f (type d) (q : (float, _, d) Nx.t) k v =
    let lengths =
      Option.map
        (fun l ->
          Nx.place (Nx.beside q)
            (Nx.create Nx.int64 lead (Array.map Int64.of_int (Array.sub l 0 2))))
        c.lengths
    in
    let mask = if c.mask = Causal then Some Nx.causal else None in
    Nx.attention ?lengths ?mask ~acc ~kv:(Nx.Dense { k; v }) q
  in
  let split x = Nx.place (Nx.Placement.split ~axis:2 Two.v) x in
  let o', lse' =
    Rune.jit T.(tensor @-> tensor @-> tensor @-> returns (pair tensor tensor)) f
      (Nx.place Two.on q) (split k) (split v)
  in
  (* The gradient of [Σ o], compiled over the parts, against eager's. *)
  let loss = function [ q; k; v ] -> Nx.sum (fst (f q k v)) | _ -> invalid_arg "loss" in
  let g = Rune.grad T.(list tensor) loss [ q; k; v ] in
  let g' =
    Rune.jit T.(list tensor @-> returns (list tensor)) (Rune.grad T.(list tensor) loss)
      [ Nx.place Two.on q; split k; split v ]
  in
  List.iter2
    (fun g g' ->
      let g' = Nx.to_array (Nx.place Nx.Host.on g') in
      Array.iteri
        (fun i x ->
          at_most ~msg:"gradient" float_exact
            ~than:(1e-3 *. (1. +. Float.abs x))
            (Float.abs (x -. g'.(i))))
        (Nx.to_array g))
    g g';
  let host x = Nx.to_array (Nx.place Nx.Host.on x) in
  let o = Nx.to_array o and o' = host o' in
  let lse = Nx.to_array lse and lse' = host lse' in
  Array.iteri
    (fun i l ->
      if l = Float.neg_infinity then
        equal ~msg:"an empty row's lse" float_exact l lse'.(i)
      else begin
        let tl, to_ = tolerances dt acc c l in
        at_most ~msg:"lse" float_exact ~than:(4. *. tl) (Float.abs (l -. lse'.(i)));
        for d = 0 to c.dv - 1 do
          let j = (i * c.dv) + d in
          at_most ~msg:"o" float_exact ~than:(4. *. to_) (Float.abs (o.(j) -. o'.(j)))
        done
      end)
    lse

(* Lanes of query rows over keys and values the lanes share. *)
let law_vmap dt acc c =
  cover "paged" (c.page <> None);
  cover "lengths" (c.lengths <> None);
  cover "a captured mask" (c.mask = Captured);
  let k, v = keys dt c [||] in
  let q = draw dt c 1. [| 2; hq c; c.tq; c.dk |] in
  let lane q = fst (call acc c ~lead:[||] q k v) in
  let each = Nx.stack ~axis:0 [ lane (Nx.get [ 0 ] q); lane (Nx.get [ 1 ] q) ] in
  equal ~msg:"o" (array int64) (bits each) (bits (Rune.vmap' lane q))

(* Lanes of pools of their own over one query: vmap reads them as one pool
   of every lane's pages, each lane's table entries renumbered into it and
   one out of range, -1 or past the pages, kept out of range. *)
let law_vmap_pools dt acc c =
  let page = Option.value c.page ~default:2 in
  let blocks = Array.length c.table / 2 in
  let table = Array.copy c.table in
  table.(0) <- -1;
  table.(blocks - 1) <- 3;
  let c = { c with page = Some page; tk = page * blocks; table; lengths = None } in
  cover "an entry out of range in a lane's table"
    (Array.exists (fun e -> e < 0 || e >= 3) (Array.sub c.table 0 blocks));
  let pools = draw dt c 2. [| 2; 3; c.hk; page; c.dk + c.dv |] in
  let q = draw dt c 1. [| hq c; c.tq; c.dk |] in
  let lane pool =
    let k = Nx.slice [ A; A; A; R (0, c.dk) ] pool in
    let v = Nx.slice [ A; A; A; R (c.dk, c.dk + c.dv) ] pool in
    fst (call acc c ~lead:[||] q k v)
  in
  let each =
    Nx.stack ~axis:0 [ lane (Nx.get [ 0 ] pools); lane (Nx.get [ 1 ] pools) ]
  in
  equal ~msg:"o" (array int64) (bits each) (bits (Rune.vmap' lane pools))

(* [<w, f'(q) t>] against central differences of step [h], within
   [tol (1 + |d|)]. A float32 [f], whose rounding over [h] dominates, takes
   [h = 1e-2] and [tol = 1e-2]. In float64, to first order in [u = 2{^-53}],
   the difference is within [e / h + h² F / 6] of the derivative, [e]
   bounding [f]'s computed error and [F] its third derivative:
   - [e]. Values lie in [[-1, 1]], [dk <= 4], [tk <= 12], [tq <= 4] and
     ALiBi's slopes are at most 1, so [a = √dk <= 2] and [b <= a + 11].
     nx.mli's bound has [η <= 12u a + 2u b <= 50u],
     [ε <= (22 + 7 ln 12) u + u (b + ln 12) <= 55u] and
     [(2b + ln tk + 5) u <= 34u]: [w <= 189u], and with [γ(13, 2u)] and
     [o]'s rounding each element of [o] is within [216u]. [f] sums at most
     96 products by [|w| <= 1], each rounded, within [γ(95)]: with the
     rounding of [q + h t], [e <= 32000u <= 3.6e-12].
   - [F]. Along [t] each score moves by [g = scale t·k], [|g| <= √dk <= 2].
     For a complex step [|σ| <= ρ = π / 8] the weights' terms turn by at
     most [π / 4], so their sum is at least [cos (π / 4)] of their moduli'
     and [|o| <= √2]; Cauchy's estimate gives [|o'''| <= 3! √2 / ρ³ <= 140],
     and [F <= 96 · 140].
   At [h = 1e-5] that is [3.6e-7 + 2.3e-7]; [d]'s own error is of [e]'s
   order times the derivative's, far below: [tol = 1e-6], where float32's
   rounding of [o] alone, [2{^-24}] each, would give [0.6]. *)
let law_jvp dt acc ~h ~tol c =
  cover "paged" (c.page <> None);
  cover "lengths" (c.lengths <> None);
  let lead = [| 2 |] in
  let k, v = keys dt c lead in
  let q = draw dt c 1. [| 2; hq c; c.tq; c.dk |] in
  let t = draw dt c 9. (Nx.shape q) in
  let w = draw dt c 10. [| 2; hq c; c.tq; c.dv |] in
  let f q = Nx.sum (Nx.mul w (fst (call acc c ~lead q k v))) in
  let _, d = Rune.jvp' f q t in
  let at s = Nx.item [] (f (Nx.add q (Nx.mul_s t s))) in
  let fd = (at h -. at (-.h)) /. (2. *. h) in
  let d = Nx.item [] d in
  at_most ~msg:"directional derivative" float_exact
    ~than:(tol *. (1. +. Float.abs d))
    (Float.abs (d -. fd))

(* Rune.grad of a loss over attention against central differences, its
   query, keys and values float64. In float32, each cast down and
   accumulating in float32, within [2e-3] at a step of [1e-2], set by hand
   as rounding dominates. In float64 the difference is within
   [e / h + h² F / 6] of the derivative, as [law_jvp] derives, along
   check_grads' directions, of entries at most [2.1]:
   - [e]. [dk = 3], [tk = 7] and slopes of at most [0.4] give [a <= √3]
     and [b <= 4.2]: [η <= 26u], [ε <= 42u], [(2b + ln tk + 5) u <= 16u],
     [w <= 110u], each of the 80 elements of [o] within [127u], and the
     loss, with its products, its sum and the parameters' rounding, within
     [e <= 19000u <= 2.1e-12].
   - [F]. A score is [scale (q + σ t_q)·(k + σ t_k)], of slope at most
     [scale · 3 · 4.2 <= 7.3] and curvature at most [scale · 3 · 4.41
     <= 7.7]; at [|σ| <= ρ = 0.095] its terms turn by at most
     [0.76 < π / 4], a value is at most [1.2], so [|o| <= 1.2 / cos 0.76
     <= 1.67] and [|o'''| <= 3! · 1.67 / ρ³ <= 11700]: [F <= 80 · 11700].
   At [h = 2e-6] that is [1.05e-6 + 6.3e-7]: [tol = 2e-6], where float32's
   rounding of [o] alone, [2{^-24}] each, would give about [2.4]. *)
let grads (type s) (dt : (float, s) Nx.dtype) ~eps ~tol () =
  let w =
    Nx.create Nx.float64 [| 2; 4; 5; 2 |]
      (Array.init 80 (fun k -> Float.sin (Float.of_int k)))
  in
  let slopes = Nx.create dt [| 4; 1; 1 |] [| 0.1; 0.2; 0.3; 0.4 |] in
  let lengths = Nx.create Nx.int64 [| 2 |] [| 6L; 7L |] in
  let loss = function
    | [ q; k; v ] ->
        let c x = Nx.cast dt x in
        let o, _ =
          Nx.attention ~lengths ~mask:Nx.causal
            ~score:(fun p s -> Nx.(Infix.(s - (cast dt (p.q - p.k) * slopes))))
            ~acc:dt
            ~kv:(Nx.Dense { k = c k; v = c v })
            (c q)
        in
        Nx.sum (Nx.mul w (Nx.cast Nx.float64 o))
    | _ -> invalid_arg "loss"
  in
  let f s seed =
    Nx.create Nx.float64 s
      (Array.init (numel s) (fun k -> Float.sin ((Float.of_int k *. 2.3) +. seed)))
  in
  let params = [ f [| 2; 4; 5; 3 |] 1.; f [| 2; 2; 7; 3 |] 2.; f [| 2; 2; 7; 2 |] 3. ] in
  equal ~msg:"gradient" (result unit string) (Ok ())
    (Rune.check_grads ~eps ~tol T.(list tensor) loss params)

(* The gradient of a loss over [o] and [lse], [lse]'s cotangent non-zero, in
   the query, keys, values and the slopes the score captures, by
   [Attention_linear] transposed: compiled against eager, and reverse mode
   against forward mode along a direction. Each computes the same sums of
   products of the pair loop in another order: in float64 within 1e-9
   relative, in float32 within 1e-3, set by hand. *)
let law_grad (type a) (acc : (float, a) Nx.dtype) ~tol c =
  cover "paged" (c.page <> None);
  cover "lengths" (c.lengths <> None);
  let c = { c with alibi = true } in
  let lead = [| 2 |] in
  let k, v = keys acc c lead in
  let q = draw acc c 1. [| 2; hq c; c.tq; c.dk |] in
  let slopes = Nx.abs (draw acc c 8. [| hq c; 1; 1 |]) in
  let w = draw acc c 10. [| 2; hq c; c.tq; c.dv |] in
  let w2 = draw acc c 11. [| 2; hq c; c.tq |] in
  let n = numel lead in
  let lengths =
    Option.map
      (fun l -> Nx.create Nx.int64 lead (Array.map Int64.of_int (Array.sub l 0 n)))
      c.lengths
  in
  let mask =
    match c.mask with
    | Unmasked -> None
    | Causal -> Some Nx.causal
    | Captured ->
        let m = Nx.greater_equal_s (draw acc c 7. [| hq c; c.tq; c.tk |]) (-0.5) in
        Some (fun _ -> m)
  in
  let kv k v =
    match c.page with
    | None -> Nx.Dense { k; v }
    | Some p ->
        let pages = c.tk / p in
        let table = Array.map Int64.of_int (Array.sub c.table 0 (n * pages)) in
        Nx.Paged
          { k; v; table = Nx.create Nx.int64 (Array.append lead [| pages |]) table }
  in
  let loss = function
    | [ q; k; v; slopes ] ->
        let score (p : Nx.host Nx.pair) s =
          Nx.(Infix.(s - (cast acc (p.q - p.k) * slopes)))
        in
        let o, lse = Nx.attention ?lengths ?mask ~score ~acc ~kv:(kv k v) q in
        let finite x = Nx.where (Nx.isfinite x) x (Nx.zeros_like x) in
        Nx.add (Nx.sum (Nx.mul w o)) (Nx.sum (Nx.mul w2 (finite lse)))
    | _ -> invalid_arg "loss"
  in
  let xs = [ q; k; v; slopes ] in
  let g = Rune.grad T.(list tensor) loss xs in
  let g' = Rune.jit T.(list tensor @-> returns (list tensor)) (Rune.grad T.(list tensor) loss) xs in
  List.iter2
    (fun g g' ->
      let g = Nx.to_array g and g' = Nx.to_array g' in
      Array.iteri
        (fun i x ->
          at_most ~msg:"compiled gradient" float_exact
            ~than:(tol *. (1. +. Float.abs x))
            (Float.abs (x -. g'.(i))))
        g)
    g g';
  let ts = List.mapi (fun i x -> draw acc c (20. +. Float.of_int i) (Nx.shape x)) xs in
  let _, d = Rune.jvp T.(list tensor) T.tensor loss xs ts in
  let d = Nx.item [] d in
  let dot =
    List.fold_left2
      (fun a g t -> a +. Nx.item [] (Nx.sum (Nx.mul g t)))
      0. g ts
  in
  at_most ~msg:"forward against reverse" float_exact
    ~than:(tol *. (1. +. Float.abs dot))
    (Float.abs (d -. dot))

(* Values and keys past a row's length are never read, by the derivative
   either: NaN there leaves every cotangent and tangent finite, eagerly and
   compiled, and the keys' and values' past each length zero. *)
let test_past_length () =
  let lead = [| 2 |] and tk = 8 and lens = [| 3; 5 |] in
  let f s seed =
    Nx.create Nx.float64 s
      (Array.init (numel s) (fun k -> Float.sin ((Float.of_int k *. 1.7) +. seed)))
  in
  let poisoned s seed =
    let x = Nx.to_array (f s seed) in
    let per = numel (Array.sub s 1 3) and d = s.(3) in
    Array.iteri
      (fun i _ -> if i / per |> fun b -> i mod per / d mod tk >= lens.(b) then x.(i) <- Float.nan)
      x;
    Nx.create Nx.float64 s x
  in
  let lengths = Nx.create Nx.int64 lead (Array.map Int64.of_int lens) in
  let loss = function
    | [ q; k; v ] ->
        let o, lse =
          Nx.attention ~lengths ~mask:Nx.causal ~acc:Nx.float64
            ~kv:(Nx.Dense { k; v }) q
        in
        let finite x = Nx.where (Nx.isfinite x) x (Nx.zeros_like x) in
        Nx.add (Nx.sum o) (Nx.sum (finite lse))
    | _ -> invalid_arg "loss"
  in
  let xs =
    [ f [| 2; 2; 4; 3 |] 1.; poisoned [| 2; 1; tk; 3 |] 2.; poisoned [| 2; 1; tk; 2 |] 3. ]
  in
  let finite msg x =
    Array.iter
      (fun v -> equal ~msg Testable.bool true (Float.is_finite v))
      (Nx.to_array x)
  in
  List.iter (finite "eager gradient") (Rune.grad T.(list tensor) loss xs);
  List.iter (finite "compiled gradient")
    (Rune.jit T.(list tensor @-> returns (list tensor)) (Rune.grad T.(list tensor) loss) xs);
  let ts = List.map (fun x -> Nx.ones_like x) xs in
  finite "tangent" (snd (Rune.jvp T.(list tensor) T.tensor loss xs ts))

let () =
  let case = Gen.with_pp pp_case case in
  (* Each law in float32, in float64, and for a float32 query in a float64
     accumulator. *)
  let three ?examples count f32 f64 wide =
    [
      prop ?examples ~count "float32" case f32;
      prop ?examples ~count "float64" case f64;
      prop ?examples ~count "float32 in float64" case wide;
    ]
  in
  let f32 = Nx.float32 and f64 = Nx.float64 in
  exit
    (run "rune attention"
       [
         group "jit within the bounds of eager"
           (three ~examples 12 (law_jit f32 f32) (law_jit f64 f64)
              (law_jit f32 f64));
         group "jit within nx's bound of a float64 reference"
           (let dts = Nx.Dtype.[ Any Nx.float16; Any Nx.bfloat16; Any Nx.float32; Any Nx.float64 ] in
            [
              prop
                ~examples:(Rune_support.Attentions.examples dts)
                ~count:6 "every dtype"
                (Gen.with_pp Rune_support.Attentions.pp (Rune_support.Attentions.gen dts))
                (Rune_support.Attentions.law (module Nx.Host));
              Windtrap.cases "a sum of zeros is +0 at rows of one run of keys"
                ~id:(fun (Nx.Dtype.Any dt) -> Nx.Dtype.name dt)
                Nx.Dtype.[ Any Nx.float32; Any Nx.bfloat16; Any Nx.float16 ]
                (Rune_support.Attentions.zero_sums (module Nx.Host));
            ]);
         group "jit over keys cut across two devices"
           (three ~examples 12 (law_cut f32 f32) (law_cut f64 f64)
              (law_cut f32 f64));
         group "vmap is each lane, bit for bit"
           (three 60 (law_vmap f32 f32) (law_vmap f64 f64) (law_vmap f32 f64));
         group "vmap over lanes' own pools is each lane, bit for bit"
           (three 60 (law_vmap_pools f32 f32) (law_vmap_pools f64 f64)
              (law_vmap_pools f32 f64));
         group "jvp by central differences"
           (three 60
              (law_jvp f32 f32 ~h:1e-2 ~tol:1e-2)
              (law_jvp f64 f64 ~h:1e-5 ~tol:1e-6)
              (law_jvp f32 f64 ~h:1e-2 ~tol:1e-2));
         group "the gradient compiled and in forward mode"
           [
             prop ~examples ~count:12 "float32" case (law_grad f32 ~tol:1e-3);
             prop ~examples ~count:12 "float64" case (law_grad f64 ~tol:1e-9);
           ];
         group "past a row's length"
           [ test "the derivative reads no key or value there" test_past_length ];
         group "gradients by central differences"
           [
             test "float32" (grads Nx.float32 ~eps:1e-2 ~tol:2e-3);
             test "float64" (grads Nx.float64 ~eps:2e-6 ~tol:2e-6);
           ];
       ])
