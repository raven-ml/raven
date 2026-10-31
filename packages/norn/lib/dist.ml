(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* A parameter's domain check, run on use: [run context unless] checks the
   parameter where [unless] does not hold, its message starting with [context];
   [inside ()] is whether every element is in the domain. *)
type 'd check = {
  run : string -> 'd Nx.bool_t option -> unit;
  inside : unit -> 'd Nx.bool_t;
}

type ('x, 'f, 'd) kind =
  | Continuous : ((float, 'f, 'd) Nx.t, 'f, 'd) kind
  | Counts : ('d Nx.int32_t, 'f, 'd) kind
  | Categories : ('d Nx.int64_t, 'f, 'd) kind
  | Booleans : ('d Nx.bool_t, 'f, 'd) kind

(* A distribution is its family's operations as closures, with its domain checks
   and whether they have run ([checked]). [descr] names it in [pp] and in
   errors. *)
type ('x, 'f, 'd) t = {
  family : string;
  descr : string;
  dtype : (float, 'f) Nx.dtype;
  shape : int array;
  checks : 'd check list;
  checked : bool;
  support : unit -> Support.t;
  bounds : unit -> (float, 'f, 'd) Nx.t * (float, 'f, 'd) Nx.t;
      (* each element's least and greatest value, unbroadcast *)
  factors : 'x -> (float, 'f, 'd) Nx.t;
  draw : 'd Nx.Rng.key -> int array -> 'x; (* at a value shape *)
  coords : ('f, 'd) Bij.t option;
  standardize : ('f, 'd) Bij.t option;
  quantile : ((float, 'f, 'd) Nx.t -> (float, 'f, 'd) Nx.t) option;
  membership : ('x -> (float, 'f, 'd) Nx.t) option;
  kind : ('x, 'f, 'd) kind;
  at : 'd Nx.Placement.t; (* where the parameters lie, uncut *)
}

(* A combinator applies tensor operations to values whose type only their family
   knows: [apply k f x] applies [f], polymorphic in the tensor's type, to [x],
   of kind [k]. *)
type 'd map = { f : 'a 'b. ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t }

let apply : type x f d. (x, f, d) kind -> d map -> x -> x =
 fun kind m x ->
  match kind with
  | Continuous -> m.f x
  | Counts -> m.f x
  | Categories -> m.f x
  | Booleans -> m.f x

let ndim : type x f d. (x, f, d) kind -> x -> int =
 fun kind x ->
  match kind with
  | Continuous -> Nx.ndim x
  | Counts -> Nx.ndim x
  | Categories -> Nx.ndim x
  | Booleans -> Nx.ndim x

(* [select axis i x] is [x] at the indices [i] along [axis], which it
   removes. *)
let select axis i =
  {
    f =
      (fun x ->
        let lead = Array.sub (Nx.shape x) 0 axis in
        let rest = Array.sub (Nx.shape x) (axis + 1) (Nx.ndim x - axis - 1) in
        let shape = Array.concat [ lead; [| 1 |]; rest ] in
        let i =
          Nx.reshape
            (Array.concat [ lead; [| 1 |]; Array.map (fun _ -> 1) rest ])
            i
        in
        Nx.squeeze ~axes:[ axis ]
          (Nx.take_along_axis ~axis (Nx.broadcast_to shape i) x));
  }

(* Shapes *)

let shape_string s =
  String.concat "; " (Array.to_list (Array.map string_of_int s))

let broadcast fn shapes =
  let n = List.fold_left (fun n s -> max n (Array.length s)) 0 shapes in
  let out = Array.make n 1 in
  List.iter
    (fun s ->
      let off = n - Array.length s in
      Array.iteri
        (fun i d ->
          let j = off + i in
          if out.(j) = 1 then out.(j) <- d
          else if d <> 1 && d <> out.(j) then
            invalid_argf
              "Norn.Dist.%s: parameters of shapes %s do not broadcast" fn
              (String.concat " and "
                 (List.map (fun s -> "[" ^ shape_string s ^ "]") shapes)))
        s)
    shapes;
  out

let describe family params =
  Printf.sprintf "%s(%s)" family
    (String.concat ", "
       (List.map
          (fun (name, dt, s) ->
            Printf.sprintf "%s: %s [%s]" name dt (shape_string s))
          params))

let param name x = (name, Nx.Dtype.name (Nx.dtype x), Nx.shape x)

(* Domains *)

let const x v = Nx.scalar_like x v
let neg_inf x = const x Float.neg_infinity
let positive x = Nx.greater x (const x 0.)
let non_negative x = Nx.greater_equal x (const x 0.)
let finite x = Nx.isfinite x
let below_infinity x = Nx.less x (const x Float.infinity)
let not_nan x = Nx.logical_not (Nx.isnan x)

let check family param domain inside x =
  let run context unless =
    let ok = inside x in
    let ok = match unless with None -> ok | Some u -> Nx.logical_or ok u in
    Nx.check Nx.Ptree.tensor ok x (fun i v ->
        (* [unless] may broadcast [ok] past [x]: [x]'s place is the trailing
           entries. *)
        let n = Nx.ndim x and k = Array.length i in
        let at =
          if n = 0 then ""
          else Printf.sprintf " at [%s]" (shape_string (Array.sub i (k - n) n))
        in
        Invalid_argument
          (Printf.sprintf "%s: %s: %s%s is %s, not in %s" context family param
             at (Rows.element v) domain))
  in
  { run; inside = (fun () -> Nx.all (inside x)) }

let in_positive family name x =
  check family name "(0, inf)"
    (fun x -> Nx.logical_and (positive x) (below_infinity x))
    x

let in_reals family name x = check family name "(-inf, inf)" finite x

let run_checks ?unless context checks =
  List.iter (fun c -> c.run context unless) checks

let checked context d =
  if not d.checked then run_checks ("Norn.Dist." ^ context) d.checks

(* Arithmetic *)

let half_log_2pi = 0.5 *. Float.log (2. *. Float.pi)
let log_pi = Float.log Float.pi

let log_sigmoid x = Nx.neg (Nx.softplus (Nx.neg x))

(* [inside ok x safe] is [x] where [ok] holds and [safe] elsewhere: a value
   outside the support enters the density's arithmetic as [safe], so neither its
   value nor its derivative is NaN, and the factor is then set to [-inf]. *)
let inside ok x safe = Nx.where ok x (const x safe)
let outside ok f = Nx.where ok f (neg_inf f)

(* Count masses *)

(* A count's log mass, [log (n! / (x! (n - x)!)) + x log p + ...], sums terms
   near [n log n] that cancel to near [log n], losing about [log10 n] digits.
   Loader's saddle-point form (Fast and Accurate Computation of Binomial
   Probabilities, 2000) writes it as terms that keep their sign:

     log p(x) = δ(n) - δ(a) - δ(b) - D(a, M_a) - D(b, M_b)
                - log (2π a b / n) / 2

   for [a + b = n] and means [M_a + M_b = n], with Stirling's error [δ] and
   the deviance [D]. A Poisson mass is the limit [-δ(x) - D(x, rate)
   - log (2π x) / 2].

   Error, from nx's promises ([log] and [log1p] within 2 ulps, [sigmoid] and
   [softplus] within 3) and half an ulp, [u = ε/2], per arithmetic step.
   Counts of 1 or more have masses at most 0.4, so [|f| >= 0.9], or 0.69 for
   a binomial's interior: an absolute error [k u] is at most [1.45 k u |f|].
   The terms share a sign but for [δ(n)], below [δ(a)], so [f] is as exact,
   relatively, as its worst term, plus the sum's roundings:
   - [δ] of an integer reads a table to 15 (within [u]) and seven series
     terms above (a few [u]). Of a real [n], it is [δ(n + 7)]'s series plus
     seven steps [δ(m) - δ(m + 1) = (m + 1/2) log1p (1/m) - 1], each a
     product within [7 u] of a value near 1 less an exact 1: [50 u]
     absolutely, as is [δ(r) + log r / 2].
   - [D]'s series, where [x / m] is in [(1/2, 2)], subtracts an exact
     [x - m] (Sterbenz) and adds terms within [1/9] of each other: [4 u].
     Its direct form [x L - (x - m)], [L = log (x / m)], loses [5 u x |L|]
     to [L]'s 2 ulps and the product and [u (|x - m| + D)] to the
     subtraction; at that interval's ends [x |L| <= 3.6 D] and
     [|x - m| <= 6.5 D], so [26 u D]. Past the band of [2^32], [L] is
     [log x - log m] with [log m = log n + log_sigmoid l] or the like:
     [4 u (|log x| + |log n|) + 6 u |log m - log n|] and two roundings,
     where [|L| >= 22] and [|log x|, |log n| <= 21.5]: [30 u D].
   - [log (2π x) / 2] and the sum's roundings: [4 u].
   Hence [c_f]: [24] for poisson and binomial ([D]'s [30 u] and the sum),
   and [64] for the negative binomial, whose two real [δ] add [100 u]
   absolutely, [145 u |f|].
   The means add [ε |x - m|] times a family's count of their roundings, as
   moving [M] by [η M] moves [D(a, M)] by [η |a - M|]:
   - poisson: the rate is exact, but the direct form rounds [x / m],
     [x u] in [D], where [x <= 2 |x - m|]: [c_m = 1].
   - binomial: [M_a = n sigmoid l] is within [3.5 ε], as is [M_b], and
     [|a - M_a| = |b - M_b| = |x - m|]; with both ratios' roundings: [9].
   - negative binomial: [a - M_a] and [M_b] are within [2 ε], and
     [|a - M_a| = |b - M_b| <= |x - m|]; with the ratios: [6]. *)

(* The Stirling series [δ(n) ~ Σ_k c_k / n^(2k+1)], [c_k] the Bernoulli
   numbers' [B_(2k+2) / ((2k+1) (2k+2))], signs alternating. *)
let stirling =
  [|
    1. /. 12.;
    1. /. 360.;
    1. /. 1260.;
    1. /. 1680.;
    1. /. 1188.;
    691. /. 360360.;
    1. /. 156.;
    3617. /. 122400.;
    43867. /. 244188.;
    174611. /. 125400.;
    77683. /. 5796.;
    236364091. /. 1506960.;
    657931. /. 300.;
    3392780147. /. 93960.;
    1723168255201. /. 2492028.;
    7709321041217. /. 505920.;
    151628697551. /. 396.;
  |]

(* [δ(n)] at the integers [0, ..., 15], from Loader's table; [δ(0)] is a
   placeholder. *)
let stirling_table =
  [|
    0.;
    0.081061466795327258219670264;
    0.041340695955409294093822081;
    0.0276779256849983391487892927;
    0.020790672103765093111522771;
    0.0166446911898211921631948653;
    0.013876128823070747998745727;
    0.0118967099458917700950557241;
    0.010411265261972096497478567;
    0.0092554621827127329177286366;
    0.008330563433362871256469318;
    0.0075736754879518407949720242;
    0.006942840107209529865664152;
    0.0064089941880042070684396310;
    0.005951370112758847735624416;
    0.0055547335519628013710386899;
  |]

(* The series' terms that matter at [dt] from [cut] up: those above [eps / 4]
   of the first at [cut], where they are largest. The series is asymptotic:
   at 5.25 no number of terms reaches float64's [eps / 4], at 7 thirteen do,
   at 16 seven. *)
let stirling_terms dt cut =
  let eps = Prec.eps dt in
  let rec go k =
    if k = Array.length stirling then
      invalid_argf "Norn.Dist: Stirling's series does not reach %g at %g" eps
        cut
    else
      let rel = stirling.(k) /. stirling.(0) in
      if rel /. (cut ** float_of_int (2 * k)) < eps /. 4. then k
      else go (k + 1)
  in
  go 1

(* [stirling_series n cut] is the series at [n >= cut], to the terms that
   matter there. *)
let stirling_series n cut =
  let k = stirling_terms (Nx.dtype n) cut in
  let r = Nx.recip n in
  let r2 = Nx.square r in
  let acc = ref (const n stirling.(k - 1)) in
  for i = k - 2 downto 0 do
    acc := Nx.sub (const n stirling.(i)) (Nx.mul r2 !acc)
  done;
  Nx.mul r !acc

(* [delta_int n] is [δ(n) = log Γ(n + 1) - (n + 1/2) log n + n - log √(2π)]
   at an integer [n >= 1]: Loader's table to 15, the series above. *)
let delta_int n =
  let small = Nx.less n (const n 15.5) in
  let table =
    Nx.place (Nx.beside n) (Nx.create (Nx.dtype n) [| 16 |] stirling_table)
  in
  let at = Nx.take (Nx.cast Nx.int64 (inside small n 0.)) table in
  Nx.where small at (stirling_series (Nx.where small (const n 16.) n) 16.)

(* A real [δ(n)] is [δ(n + shift)]'s series and [shift] steps down. *)
let shift = 7

(* [step m] is [δ(m) - δ(m + 1) = (m + 1/2) log1p (1/m) - 1] for [m >= 1]. *)
let step m = Nx.sub_s (Nx.mul (Nx.add_s m 0.5) (Nx.log1p (Nx.recip m))) 1.

(* [steps n first] is [δ(n + shift)] and the steps from [n + first] up. *)
let steps n first =
  let acc = ref (stirling_series (Nx.add_s n (float_of_int shift)) 7.) in
  for j = shift - 1 downto first do
    acc := Nx.add !acc (step (Nx.add_s n (float_of_int j)))
  done;
  !acc

(* [delta_real n] is [δ(n)] at a real [n >= 1], without [lgamma]. R's
   stirlerr.c takes [lgamma] below 5.25 and its series above, which at 5.25
   reaches only [2e-14]; [lgamma]'s cancellation there costs some 100 [u]
   absolutely. *)
let delta_real n = steps n 0

(* [delta_half r] is [δ(r) + log r / 2] at a real [r > 0]. Below 1 its first
   step is [(r + 1/2) log1p r - r log r - 1], where [δ(r)]'s and [log r]'s
   slopes, [∓1 / (2r)], cancel: its derivative stays finite as [r] nears 0,
   where theirs overflow. *)
let delta_half r =
  let below = Nx.less r (const r 1.) in
  let lo = inside below r 0.5 and hi = Nx.where below (const r 1.) r in
  let first =
    Nx.where below
      (Nx.sub_s
         (Nx.sub
            (Nx.mul (Nx.add_s lo 0.5) (Nx.log1p lo))
            (Nx.mul lo (Nx.log lo)))
         1.)
      (Nx.add (step hi) (Nx.mul_s (Nx.log hi) 0.5))
  in
  Nx.add (steps r 1) first

(* The terms of [D]'s series that matter at [dt], for [|v| < 1/3]. *)
let deviance_terms dt =
  let eps = Prec.eps dt in
  let rec go j =
    let term = (1. /. 9.) ** float_of_int j in
    let rel = 2. *. term /. float_of_int ((2 * j) + 1) in
    if rel < eps /. 4. then j else go (j + 1)
  in
  go 1

(* [deviance_of x ~d ~v ~log_ratio] is [D(x, m) = x log (x / m) + m - x] from
   [d = x - m], [v = d / (x + m)] and [log (x / m)], which a caller computes
   without losing digits. Where [|v| < 1/3], [x / m] is in [(1/2, 2)] and
   Loader's series [d v + 2x Σ_j v^(2j+1) / (2j + 1)] keeps [D] to its
   digits; elsewhere the direct form [x log (x / m) - d]. *)
let deviance_of x ~d ~v ~log_ratio =
  let near = Nx.less (Nx.mul_s (Nx.abs v) 3.) (const v 1.) in
  let v2 = Nx.square v in
  let j = deviance_terms (Nx.dtype x) in
  let acc = ref (const x (1. /. float_of_int ((2 * j) + 1))) in
  for i = j - 1 downto 1 do
    acc := Nx.add (const x (1. /. float_of_int ((2 * i) + 1))) (Nx.mul v2 !acc)
  done;
  let series =
    Nx.add (Nx.mul d v) (Nx.mul (Nx.mul_s x 2.) (Nx.mul (Nx.mul v v2) !acc))
  in
  Nx.where near series (Nx.sub (Nx.mul x log_ratio) d)

(* Past this factor between [x] and [m], [D]'s direct form takes
   [log x - log m]. *)
let ratio_band = 0x1p32

(* [deviance x m log_m] is [D(x, m)] for [x > 0] and [m >= 0], [log_m] being
   [log m]: [inf] at [m = 0]. Within a factor [2^32] it takes
   [log (x / m)]; past it [log x - log_m], where [log_m] keeps the digits that
   an [m] near underflow has lost and its derivative stays finite where that
   of [x / m], [x / m^2], overflows. One [log] serves both: of [x / m] or of
   [x]. The ratio reads [x] and [m] only where it is taken. *)
let deviance x m log_m =
  let d = Nx.sub x m in
  let r = Nx.div x m in
  let band =
    Nx.logical_and
      (Nx.less_equal r (const r ratio_band))
      (Nx.greater_equal r (const r (1. /. ratio_band)))
  in
  let ratio = Nx.div (inside band x 1.) (inside band m 1.) in
  let log_ratio =
    Nx.sub
      (Nx.log (Nx.where band ratio x))
      (Nx.where band (const log_m 0.) log_m)
  in
  deviance_of x ~d ~v:(Nx.div d (Nx.add x m)) ~log_ratio

(* [half lo n] is [log (2π lo (n - lo) / n) / 2] for [0 < lo <= n / 2],
   from [log lo + log1p (-lo / n)], exact for [lo ≪ n]. *)
let half lo n =
  Nx.add_s
    (Nx.mul_s (Nx.add (Nx.log lo) (Nx.log1p (Nx.neg (Nx.div lo n)))) 0.5)
    half_log_2pi

(* [log_div p q] is [log (p / q)], as [log p - log q] where the quotient is
   subnormal and has lost its digits. Each arm reads [p] and [q] only where
   it is taken. *)
let log_div p q =
  let r = Nx.div p q in
  let normal = Nx.greater_equal r (const r (Prec.tiny (Nx.dtype r))) in
  let off = Nx.logical_not normal in
  Nx.where normal
    (Nx.log (inside normal r 1.))
    (Nx.sub (Nx.log (inside off p 1.)) (Nx.log (inside off q 1.)))

(* The masses below take parameters only where a count reads them: the mass
   at [0] does not read the deviance's, whose derivative may be infinite
   there, and the others do not read the mass at [0]'s. *)

(* [poisson_mass rate x] is [-δ(x) - D(x, rate) - log (2π x) / 2] for
   counts [x >= 0], and [-rate] at [0]. *)
let poisson_mass rate x =
  let some = Nx.greater x (const x 0.) in
  let a = inside some x 1. and m = inside some rate 1. in
  let f =
    Nx.add
      (Nx.add (delta_int a) (deviance a m (Nx.log m)))
      (Nx.add_s (Nx.mul_s (Nx.log a) 0.5) half_log_2pi)
  in
  Nx.where some (Nx.neg f) (Nx.neg rate)

(* [neg_binomial_mass mean r x] for counts [x >= 0]. [C(x + r - 1, x)] is
   [r / n C(n, x)] for [n = x + r]: the binomial form of [r] and [x] out of
   [n], at means [M_r = r n / (r + mean)] and [M_x = mean n / (r + mean)],
   times [r / n]. Its [log (r / n)] is [log1p (-x / n)] where [x < r], and
   [-δ(r) - log r / 2] is [delta_half], computed at [r]'s own shape.
   [D(r, M_r)] is homogeneous, [r D(1, M_r / r)], and is taken from
   [r - M_r = (mean - x) r / (r + mean)],
   [v = (mean - x) / (2r + mean + x)] and [log ((r + mean) / n)], none of
   which divides by [r]'s scale: a subnormal [r] stays exact. *)
let neg_binomial_mass mean r x =
  let some = Nx.greater x (const x 0.) in
  let b = inside some x 1. in
  let rs = inside some r 1. and ms = inside some mean 1. in
  let n = Nx.add rs b and total = Nx.add rs ms in
  let few = Nx.greater rs b in
  let lead =
    Nx.where few
      (Nx.log1p (Nx.neg (Nx.div (inside few b 0.) n)))
      (log_div (inside (Nx.logical_not few) rs 1.) n)
  in
  let log_x_n =
    Nx.where few
      (Nx.log (Nx.div (inside few b 1.) n))
      (Nx.log1p (Nx.neg (Nx.div (inside (Nx.logical_not few) rs 0.) n)))
  in
  let gap = Nx.sub ms b in
  let d_r =
    deviance_of rs
      ~d:(Nx.mul gap (Nx.div rs total))
      ~v:(Nx.div gap (Nx.add (Nx.add rs total) b))
      ~log_ratio:(Nx.log (Nx.div total n))
  in
  let mx = Nx.mul n (Nx.div ms total) in
  let d_x =
    deviance b mx (Nx.sub (Nx.add (Nx.log n) (Nx.log ms)) (Nx.log total))
  in
  let f =
    Nx.sub
      (Nx.add lead (Nx.sub (delta_real n) (delta_int b)))
      (Nx.add
         (Nx.add (delta_half r) (Nx.add d_r d_x))
         (Nx.add_s (Nx.mul_s log_x_n 0.5) half_log_2pi))
  in
  (* At [0], [r log (r / (r + mean))]; where [mean <= r],
     [-mean log1p t / t] for [t = mean / r], which keeps its digits where [t]
     is subnormal: below [ε], [log1p t / t] is [1 - t / 2] to its precision,
     whose derivative stays finite where [1 / t] overflows. *)
  let r0 = Nx.where some (const r 1.) r in
  let m0 = Nx.where some (const mean 1.) mean in
  let wide = Nx.less r0 m0 in
  let t = Nx.div (Nx.where wide (const m0 0.) m0) r0 in
  let big_t = Nx.greater t (const t (Prec.eps (Nx.dtype t))) in
  let ts = inside big_t t 1. in
  let slope =
    Nx.where big_t
      (Nx.div (Nx.log1p ts) ts)
      (Nx.sub (const t 1.) (Nx.mul_s t 0.5))
  in
  let zero =
    Nx.where wide
      (Nx.mul r0 (log_div r0 (Nx.add r0 m0)))
      (Nx.neg (Nx.mul m0 slope))
  in
  Nx.where some f zero

(* [times n y] is [n y], [0] where [n] is [0]: no successes at a probability
   of [0] weigh nothing. *)
let times n y = Nx.where (Nx.equal n (const n 0.)) (const n 0.) (Nx.mul n y)

(* [all_of n l lp] is [n lp], [lp = log_sigmoid l], the log mass of [n]
   trials that all come out as their logit [l] says. Below [log ε],
   [softplus (-l)] is [exp (-l)] to its precision, taken as
   [(exp (-l/2) n) exp (-l/2)]: the halves stay normal where [exp (-l)] is
   subnormal, whose rounding [n] would scale, and the product rounds once. *)
let all_of n l lp =
  let y = Nx.neg l in
  let edge = Float.log (Prec.eps (Nx.dtype l)) in
  let small = Nx.less y (const y edge) in
  let half = Nx.exp (Nx.mul_s (inside small y edge) 0.5) in
  Nx.where small (Nx.neg (Nx.mul (Nx.mul half n) half)) (times n lp)

(* [binomial_mass logits trials x rest] for [x] successes and [rest]
   failures out of [trials]. [δ(trials)] and [log trials] are computed at
   [trials]' own shape; the means' logs are [log trials + log_sigmoid
   (±logits)], exact where [trials sigmoid logits] underflows. The interior's
   arm reads the logits at [0] and [trials] too, harmless there: the
   derivatives of [sigmoid] and [log_sigmoid] are bounded. *)
let binomial_mass logits trials x rest =
  let lp = log_sigmoid logits and lq = log_sigmoid (Nx.neg logits) in
  (* At [0] or [trials], all trials fail or all succeed. *)
  let edge =
    Nx.add (all_of x logits lp) (all_of rest (Nx.neg logits) lq)
  in
  let inner =
    Nx.logical_and (Nx.greater x (const x 0.)) (Nx.greater rest (const x 0.))
  in
  let a = inside inner x 1. and b = inside inner rest 1. in
  let n = Nx.add a b in
  let some = Nx.greater trials (const trials 0.) in
  let t = inside some trials 1. in
  let dn = Nx.where inner (delta_int t) (const n (stirling_table.(2))) in
  let ln = Nx.where inner (Nx.log t) (const n (Float.log 2.)) in
  let f =
    Nx.sub
      (Nx.sub (Nx.sub dn (delta_int a)) (delta_int b))
      (Nx.add
         (Nx.add
            (deviance a (Nx.mul n (Nx.sigmoid logits)) (Nx.add ln lp))
            (deviance b
               (Nx.mul n (Nx.sigmoid (Nx.neg logits)))
               (Nx.add ln lq)))
         (half (Nx.minimum a b) n))
  in
  Nx.where inner f edge

(* A format narrower than float32 computes a mass at float32 and rounds it
   once, as nx computes its functions: its own range and precision would
   lose the terms' digits and underflow their ratios. *)
let narrow dt = (Nx.Dtype.float_format dt).fraction_bits < 23
let widen x = Nx.cast Nx.float32 x

(* [open_unit k dt s] is uniform on [(0, 1)]: a draw of [Noise.uniform], a
   multiple of [2^-p], moved by half a step, so neither end is reached. *)
let open_unit k dt s = Nx.add_s (Noise.uniform k dt s) (Prec.eps dt /. 4.)
let broadcast_to s x = Nx.broadcast_to s x

(* Families *)

(* A continuous value's bounds are its coordinates' image of the reals. *)
let coord_bounds at dtype coords () =
  Bij.bounds coords
    ( Nx.scalar_at at dtype Float.neg_infinity,
      Nx.scalar_at at dtype Float.infinity )

let continuous ~family ~params ~shape ~checks ~support ~factors ~draw ~coords
    ?standardize ?quantile like =
  let dtype = Nx.dtype like and at = Nx.beside like in
  {
    family;
    descr = describe family params;
    dtype;
    shape;
    checks;
    checked = false;
    support;
    bounds = coord_bounds at dtype coords;
    factors;
    draw;
    coords = Some coords;
    standardize;
    quantile;
    membership = None;
    kind = Continuous;
    at;
  }

let location_scale family ~loc ~scale ~density ~draw ?quantile () =
  let dt = Nx.dtype loc in
  continuous ~family
    ~params:[ param "loc" loc; param "scale" scale ]
    ~shape:(broadcast family [ Nx.shape loc; Nx.shape scale ])
    ~checks:[ in_reals family "loc" loc; in_positive family "scale" scale ]
    ~support:(fun () -> Support.Real)
    ~factors:(fun x ->
      Nx.sub (density (Nx.div (Nx.sub x loc) scale)) (Nx.log scale))
    ~draw:(fun k s -> Nx.add loc (Nx.mul scale (draw k dt s)))
    ~coords:Bij.identity ~standardize:(Bij.affine ~loc ~scale)
    ?quantile:(Option.map (fun q p -> Nx.add loc (Nx.mul scale (q p))) quantile)
    loc

let normal ~loc ~scale =
  location_scale "normal" ~loc ~scale
    ~density:(fun z ->
      Nx.add_s (Nx.mul_s (Nx.square z) (-0.5)) (-.half_log_2pi))
    ~draw:Noise.normal ~quantile:Nx.ndtri ()

let cauchy_quantile p = Nx.tan (Nx.mul_s (Nx.sub_s p 0.5) Float.pi)

let cauchy ~loc ~scale =
  location_scale "cauchy" ~loc ~scale
    ~density:(fun z -> Nx.add_s (Nx.neg (Nx.log1p (Nx.square z))) (-.log_pi))
    ~draw:(fun k dt s -> cauchy_quantile (open_unit k dt s))
    ~quantile:cauchy_quantile ()

let laplace_quantile p =
  let d = Nx.sub_s p 0.5 in
  Nx.neg (Nx.mul (Nx.sign d) (Nx.log1p (Nx.mul_s (Nx.abs d) (-2.))))

let laplace ~loc ~scale =
  location_scale "laplace" ~loc ~scale
    ~density:(fun z -> Nx.sub_s (Nx.neg (Nx.abs z)) (Float.log 2.))
    ~draw:(fun k dt s -> laplace_quantile (open_unit k dt s))
    ~quantile:laplace_quantile ()

let logistic_quantile p = Nx.sub (Nx.log p) (Nx.log1p (Nx.neg p))

let logistic ~loc ~scale =
  location_scale "logistic" ~loc ~scale
    ~density:(fun z -> Nx.sub (Nx.neg z) (Nx.mul_s (Nx.softplus (Nx.neg z)) 2.))
    ~draw:(fun k dt s -> logistic_quantile (open_unit k dt s))
    ~quantile:logistic_quantile ()

let student_t ~df ~loc ~scale =
  let family = "student_t" in
  let dt = Nx.dtype loc in
  let half_df = Nx.mul_s df 0.5 in
  let half_df1 = Nx.add_s half_df 0.5 in
  let normaliser =
    Nx.sub
      (Nx.sub (Nx.lgamma half_df1) (Nx.lgamma half_df))
      (Nx.mul_s (Nx.add_s (Nx.log df) log_pi) 0.5)
  in
  let factors x =
    let z = Nx.div (Nx.sub x loc) scale in
    Nx.sub
      (Nx.sub normaliser (Nx.log scale))
      (Nx.mul half_df1 (Nx.log1p (Nx.div (Nx.square z) df)))
  in
  (* A normal over the square root of a chi-square over its degrees of freedom:
     [z sqrt (df / 2g)], [g] gamma of concentration [df / 2]. *)
  let draw k s =
    let k1, k2 =
      match Nx.Rng.split k with [| a; b |] -> (a, b) | _ -> assert false
    in
    let z = Noise.normal k1 dt s in
    let g = Nx.Rng.gamma ~key:k2 (broadcast_to s half_df) in
    Nx.add loc (Nx.mul scale (Nx.mul z (Nx.sqrt (Nx.div half_df g))))
  in
  continuous ~family
    ~params:[ param "df" df; param "loc" loc; param "scale" scale ]
    ~shape:(broadcast family [ Nx.shape df; Nx.shape loc; Nx.shape scale ])
    ~checks:
      [
        in_positive family "df" df;
        in_reals family "loc" loc;
        in_positive family "scale" scale;
      ]
    ~support:(fun () -> Support.Real)
    ~factors ~draw ~coords:Bij.identity ~standardize:(Bij.affine ~loc ~scale) df

let lognormal ~loc ~scale =
  let family = "lognormal" in
  let dt = Nx.dtype loc in
  let factors x =
    let ok = positive x in
    let lx = Nx.log (inside ok x 1.) in
    let z = Nx.div (Nx.sub lx loc) scale in
    outside ok
      (Nx.sub
         (Nx.add_s (Nx.mul_s (Nx.square z) (-0.5)) (-.half_log_2pi))
         (Nx.add lx (Nx.log scale)))
  in
  continuous ~family
    ~params:[ param "loc" loc; param "scale" scale ]
    ~shape:(broadcast family [ Nx.shape loc; Nx.shape scale ])
    ~checks:[ in_reals family "loc" loc; in_positive family "scale" scale ]
    ~support:(fun () -> Support.Greater 0.)
    ~factors
    ~draw:(fun k s -> Nx.exp (Nx.add loc (Nx.mul scale (Noise.normal k dt s))))
    ~coords:Bij.exp
    ~standardize:(Bij.compose Bij.exp (Bij.affine ~loc ~scale))
    ~quantile:(fun p -> Nx.exp (Nx.add loc (Nx.mul scale (Nx.ndtri p))))
    loc

(* A family on [(0, inf)] whose density at [x] is [density x], the value
   replaced by [1] outside before the arithmetic. *)
let positive_family ~family ~params ~shape ~checks ~density ~draw ?quantile
    ~closed like =
  let factors x =
    let ok = if closed then non_negative x else positive x in
    outside ok (density (inside ok x 1.))
  in
  continuous ~family ~params ~shape ~checks
    ~support:(fun () -> Support.Greater 0.)
    ~factors ~draw ~coords:Bij.exp ?quantile like

let half_normal ~scale =
  let family = "half_normal" in
  let dt = Nx.dtype scale in
  positive_family ~family ~closed:true
    ~params:[ param "scale" scale ]
    ~shape:(Nx.shape scale)
    ~checks:[ in_positive family "scale" scale ]
    ~density:(fun x ->
      let z = Nx.div x scale in
      Nx.sub
        (Nx.add_s
           (Nx.mul_s (Nx.square z) (-0.5))
           (Float.log 2. -. half_log_2pi))
        (Nx.log scale))
    ~draw:(fun k s -> Nx.mul scale (Nx.abs (Noise.normal k dt s)))
    ~quantile:(fun p -> Nx.mul scale (Nx.ndtri (Nx.mul_s (Nx.add_s p 1.) 0.5)))
    scale

let half_cauchy ~scale =
  let family = "half_cauchy" in
  let dt = Nx.dtype scale in
  let quantile p = Nx.mul scale (Nx.tan (Nx.mul_s p (Float.pi /. 2.))) in
  positive_family ~family ~closed:true
    ~params:[ param "scale" scale ]
    ~shape:(Nx.shape scale)
    ~checks:[ in_positive family "scale" scale ]
    ~density:(fun x ->
      Nx.sub
        (Nx.add_s
           (Nx.neg (Nx.log1p (Nx.square (Nx.div x scale))))
           (Float.log 2. -. log_pi))
        (Nx.log scale))
    ~draw:(fun k s -> quantile (open_unit k dt s))
    ~quantile scale

let exponential ~rate =
  let family = "exponential" in
  let dt = Nx.dtype rate in
  positive_family ~family ~closed:true
    ~params:[ param "rate" rate ]
    ~shape:(Nx.shape rate)
    ~checks:[ in_positive family "rate" rate ]
    ~density:(fun x -> Nx.sub (Nx.log rate) (Nx.mul rate x))
    ~draw:(fun k s -> Nx.div (Noise.exponential k dt s) rate)
    ~quantile:(fun p -> Nx.div (Nx.neg (Nx.log1p (Nx.neg p))) rate)
    rate

(* [exp_times l t] is [exp l * t], exactly [0] where [t] is: as [exp (l / 3)]
   three times, finite wherever [|l| < 2129], where [exp l] alone overflows
   past [709.8] and a zero [t] would make NaN. *)
let exp_times l t =
  let e = Nx.exp (Nx.div_s l 3.) in
  Nx.mul (Nx.mul (Nx.mul t e) e) e

(* The inverse gamma's quantile [x = scale / y], [y] the unit gamma's upper
   quantile at [p], since the CDF at [x] is [Q(a, scale / x)]. Reverse mode
   through [scale / y] forms [scale / y²], which overflows once [x] exceeds
   [sqrt scale 2^512], so the rule states the derivatives: [∂x/∂p = 1 / f(x)
   = scale Γ(a) e^y / y^(a+1)], taken in logs, [∂x/∂scale = 1 / y], and
   [∂x/∂a = -x (∂y/∂a) / y], from the inverse's own derivative. A tangent
   map serves forward mode as well as reverse, and its [jvp] runs only under
   a differentiation. Each term is [0] where its tangent is, so a tangent
   that is zero leaves the others' sum alone. *)
let inverse_gamma_quantile args =
  Rune.custom_jvp
    Nx.Ptree.(pair tensor (pair tensor tensor))
    Nx.Ptree.tensor
    (fun (a, (scale, p)) ->
      let y = Nx.gammainccinv a p in
      let x = Nx.div scale y in
      let tangent (da, (dscale, dp)) =
        let log_inv_f =
          Nx.sub
            (Nx.add (Nx.add (Nx.log scale) y) (Nx.lgamma a))
            (Nx.mul (Nx.add_s a 1.) (Nx.log y))
        in
        let a' = Nx.broadcast_to (Nx.shape x) a in
        let _, dy_da =
          Rune.jvp' (fun a -> Nx.gammainccinv a p) a' (Nx.ones_like a')
        in
        let along_a = Nx.neg (Nx.mul (Nx.mul da x) (Nx.div dy_da y)) in
        Nx.add along_a (Nx.add (Nx.div dscale y) (exp_times log_inv_f dp))
      in
      (x, tangent))
    args

let gamma ~concentration ~rate =
  let family = "gamma" in
  let a = concentration in
  positive_family ~family ~closed:false
    ~params:[ param "concentration" a; param "rate" rate ]
    ~shape:(broadcast family [ Nx.shape a; Nx.shape rate ])
    ~checks:
      [ in_positive family "concentration" a; in_positive family "rate" rate ]
    ~density:(fun x ->
      Nx.sub
        (Nx.add
           (Nx.sub (Nx.mul a (Nx.log rate)) (Nx.lgamma a))
           (Nx.mul (Nx.sub_s a 1.) (Nx.log x)))
        (Nx.mul rate x))
    ~draw:(fun k s -> Nx.div (Nx.Rng.gamma ~key:k (broadcast_to s a)) rate)
    ~quantile:(fun p -> Nx.div (Nx.gammaincinv a p) rate)
    a

let inverse_gamma ~concentration ~scale =
  let family = "inverse_gamma" in
  let a = concentration in
  positive_family ~family ~closed:false
    ~params:[ param "concentration" a; param "scale" scale ]
    ~shape:(broadcast family [ Nx.shape a; Nx.shape scale ])
    ~checks:
      [ in_positive family "concentration" a; in_positive family "scale" scale ]
    ~density:(fun x ->
      Nx.sub
        (Nx.sub
           (Nx.sub (Nx.mul a (Nx.log scale)) (Nx.lgamma a))
           (Nx.mul (Nx.add_s a 1.) (Nx.log x)))
        (Nx.div scale x))
    ~draw:(fun k s -> Nx.div scale (Nx.Rng.gamma ~key:k (broadcast_to s a)))
    ~quantile:(fun p -> inverse_gamma_quantile (a, (scale, p)))
    a

let beta ~a ~b =
  let family = "beta" in
  let factors x =
    let ok = Nx.logical_and (positive x) (Nx.less x (const x 1.)) in
    let x = inside ok x 0.5 in
    outside ok
      (Nx.sub
         (Nx.add
            (Nx.mul (Nx.sub_s a 1.) (Nx.log x))
            (Nx.mul (Nx.sub_s b 1.) (Nx.log1p (Nx.neg x))))
         (Nx.lbeta a b))
  in
  continuous ~family
    ~params:[ param "a" a; param "b" b ]
    ~shape:(broadcast family [ Nx.shape a; Nx.shape b ])
    ~checks:[ in_positive family "a" a; in_positive family "b" b ]
    ~support:(fun () -> Support.Interval (0., 1.))
    ~factors
    ~draw:(fun k s -> Nx.Rng.beta ~key:k (broadcast_to s a) (broadcast_to s b))
    ~coords:(Bij.interval ~low:(Nx.scalar_like a 0.) ~high:(Nx.scalar_like a 1.))
    a

let uniform ~low ~high =
  let family = "uniform" in
  let dt = Nx.dtype low in
  let width = Nx.sub high low in
  let factors x =
    let ok = Nx.logical_and (Nx.greater_equal x low) (Nx.less_equal x high) in
    outside ok (Nx.broadcast_to (Nx.shape ok) (Nx.neg (Nx.log width)))
  in
  let coords = Bij.interval ~low ~high in
  continuous ~family
    ~params:[ param "low" low; param "high" high ]
    ~shape:(broadcast family [ Nx.shape low; Nx.shape high ])
    ~checks:
      [
        check family "high - low" "(0, inf)"
          (fun w -> Nx.logical_and (positive w) (finite w))
          width;
      ]
    ~support:(fun () -> Bij.image coords Support.Real [||])
    ~factors
    ~draw:(fun k s -> Nx.add low (Nx.mul width (Noise.uniform k dt s)))
    ~coords
    ~quantile:(fun p -> Nx.add low (Nx.mul width p))
    low

let dirichlet ~concentration =
  let family = "dirichlet" in
  let a = concentration in
  let dt = Nx.dtype a in
  let s = Nx.shape a in
  let n = Array.length s in
  if n = 0 || s.(n - 1) < 2 then
    invalid_argf
      "Norn.Dist.dirichlet: concentration of shape [%s] has fewer than two \
       components"
      (shape_string s);
  let k = s.(n - 1) in
  let last x = Nx.ndim x - 1 in
  let tol = 64. *. float_of_int k *. Prec.eps dt in
  let factors x =
    let x = Nx.broadcast_to (broadcast family [ Nx.shape x; s ]) x in
    let ax = last x in
    let ok =
      Nx.logical_and
        (Nx.all ~axes:[ ax ]
           (Nx.logical_and (positive x) (Nx.less_equal x (const x 1.))))
        (Nx.less_equal
           (Nx.abs (Nx.sub_s (Nx.sum ~axes:[ ax ] x) 1.))
           (const x tol))
    in
    let x =
      Nx.where (Nx.unsqueeze ~axes:[ ax ] ok) x (const x (1. /. float_of_int k))
    in
    let normaliser =
      Nx.sub
        (Nx.lgamma (Nx.sum ~axes:[ n - 1 ] a))
        (Nx.sum ~axes:[ n - 1 ] (Nx.lgamma a))
    in
    outside ok
      (Nx.add normaliser
         (Nx.sum ~axes:[ ax ] (Nx.mul (Nx.sub_s a 1.) (Nx.log x))))
  in
  continuous ~family
    ~params:[ param "concentration" a ]
    ~shape:s
    ~checks:[ in_positive family "concentration" a ]
    ~support:(fun () -> Support.Simplex k)
    ~factors
    ~draw:(fun key shape -> Nx.Rng.dirichlet ~key:key (broadcast_to shape a))
    ~coords:Bij.simplex a

let mvn ~loc ~scale_tril =
  let family = "mvn" in
  let dt = Nx.dtype loc in
  let ls = Nx.shape loc and ms = Nx.shape scale_tril in
  let nl = Array.length ls and nm = Array.length ms in
  if
    nl = 0 || nm < 2 || ms.(nm - 1) <> ms.(nm - 2) || ms.(nm - 1) <> ls.(nl - 1)
  then
    invalid_argf
      "Norn.Dist.mvn: loc of shape [%s] and scale_tril of shape [%s] do not \
       describe vectors of one length"
      (shape_string ls) (shape_string ms);
  let d = ls.(nl - 1) in
  let lower = Nx.tril scale_tril in
  let shape =
    Array.append
      (broadcast family [ Array.sub ls 0 (nl - 1); Array.sub ms 0 (nm - 2) ])
      [| d |]
  in
  let log_diag = Nx.log (Nx.diagonal lower) in
  let factors x =
    let diff = Nx.sub x loc in
    let n = Nx.ndim diff in
    let batch =
      broadcast family
        [ Array.sub (Nx.shape diff) 0 (n - 1); Array.sub ms 0 (nm - 2) ]
    in
    let l = Nx.broadcast_to (Array.append batch [| d; d |]) lower in
    let diff = Nx.broadcast_to (Array.append batch [| d |]) diff in
    let z =
      Nx.solve_triangular l (Nx.unsqueeze ~axes:[ Array.length batch + 1 ] diff)
    in
    let nb = Array.length batch in
    let quad = Nx.sum ~axes:[ nb; nb + 1 ] (Nx.square z) in
    Nx.sub
      (Nx.add_s (Nx.mul_s quad (-0.5)) (-.float_of_int d *. half_log_2pi))
      (Nx.sum ~axes:[ Nx.ndim log_diag - 1 ] log_diag)
  in
  let draw k s =
    let z = Noise.normal k dt s in
    let n = Array.length s in
    Nx.add loc
      (Nx.squeeze ~axes:[ n ] (Nx.matmul lower (Nx.unsqueeze ~axes:[ n ] z)))
  in
  continuous ~family
    ~params:[ param "loc" loc; param "scale_tril" scale_tril ]
    ~shape
    ~checks:
      [
        in_reals family "loc" loc;
        in_reals family "scale_tril's lower triangle" (Nx.tril scale_tril);
        in_positive family "scale_tril's diagonal" (Nx.diagonal scale_tril);
      ]
    ~support:(fun () -> Support.Real)
    ~factors ~draw ~coords:Bij.identity
    ~standardize:(Bij.affine_tril ~loc ~scale_tril)
    loc

(* Discrete families *)

(* [fixed like (lo, hi)] is bounds [lo] and [hi] beside [like], at its
   dtype. *)
let fixed like (lo, hi) () =
  let at = Nx.beside like and dt = Nx.dtype like in
  (Nx.scalar_at at dt lo, Nx.scalar_at at dt hi)

let discrete ~family ~kind ~params ~shape ~checks ~support ~bounds ~factors
    ~draw like =
  let dtype = Nx.dtype like and at = Nx.beside like in
  {
    family;
    descr = describe family params;
    dtype;
    shape;
    checks;
    checked = false;
    support;
    bounds;
    factors;
    draw;
    coords = None;
    standardize = None;
    quantile = None;
    membership = None;
    kind;
    at;
  }

let bernoulli ~logits =
  let family = "bernoulli" in
  discrete ~family ~kind:Booleans
    ~params:[ param "logits" logits ]
    ~shape:(Nx.shape logits)
    ~checks:[ check family "logits" "[-inf, inf]" not_nan logits ]
    ~support:(fun () -> Support.Boolean)
    ~bounds:(fixed logits (0., 1.))
    ~factors:(fun x ->
      Nx.where x (log_sigmoid logits) (log_sigmoid (Nx.neg logits)))
    ~draw:(fun k s -> Nx.Rng.bernoulli ~key:k (Nx.sigmoid (broadcast_to s logits)))
    logits

let poisson ~rate =
  let family = "poisson" in
  discrete ~family ~kind:Counts
    ~params:[ param "rate" rate ]
    ~shape:(Nx.shape rate)
    ~checks:
      [
        check family "rate" "[0, inf)"
          (fun r -> Nx.logical_and (non_negative r) (below_infinity r))
          rate;
      ]
    ~support:(fun () -> Support.Integers_from 0)
    ~bounds:(fixed rate (0., Float.infinity))
    ~factors:(fun x ->
      let ok = Nx.greater_equal x (Nx.scalar_like x 0l) in
      let x = Nx.where ok x (Nx.scalar_like x 0l) in
      let dt = Nx.dtype rate in
      let f =
        if narrow dt then
          Nx.cast dt (poisson_mass (widen rate) (widen x))
        else poisson_mass rate (Nx.cast dt x)
      in
      outside ok f)
    ~draw:(fun k s -> Nx.Rng.poisson ~key:k (broadcast_to s rate))
    rate

let neg_binomial ~mean ~dispersion =
  let family = "neg_binomial" in
  let r = dispersion in
  discrete ~family ~kind:Counts
    ~params:[ param "mean" mean; param "dispersion" dispersion ]
    ~shape:(broadcast family [ Nx.shape mean; Nx.shape r ])
    ~checks:
      [
        check family "mean" "[0, inf)"
          (fun m -> Nx.logical_and (non_negative m) (below_infinity m))
          mean;
        in_positive family "dispersion" r;
      ]
    ~support:(fun () -> Support.Integers_from 0)
    ~bounds:(fixed mean (0., Float.infinity))
    ~factors:(fun x ->
      let ok = Nx.greater_equal x (Nx.scalar_like x 0l) in
      let x = Nx.where ok x (Nx.scalar_like x 0l) in
      let dt = Nx.dtype mean in
      let f =
        if narrow dt then
          Nx.cast dt
            (neg_binomial_mass (widen mean) (widen r) (widen x))
        else neg_binomial_mass mean r (Nx.cast dt x)
      in
      outside ok f)
    ~draw:(fun k s ->
      let k1, k2 =
        match Nx.Rng.split k with [| a; b |] -> (a, b) | _ -> assert false
      in
      let rate = Nx.mul (Nx.Rng.gamma ~key:k1 (broadcast_to s r)) (Nx.div mean r) in
      Nx.Rng.poisson ~key:k2 rate)
    mean

let binomial ~trials ~logits =
  let family = "binomial" in
  let dt = Nx.dtype logits in
  let factors x =
    let ok =
      Nx.logical_and
        (Nx.greater_equal x (Nx.scalar_like x 0l))
        (Nx.less_equal x trials)
    in
    let x = Nx.where ok x (Nx.scalar_like x 0l) in
    let rest = Nx.sub trials x in
    let f =
      if narrow dt then
        Nx.cast dt
          (binomial_mass (widen logits) (widen trials) (widen x) (widen rest))
      else
        binomial_mass logits (Nx.cast dt trials) (Nx.cast dt x)
          (Nx.cast dt rest)
    in
    outside ok f
  in
  (* Drawn at the rarer outcome's probability [sigmoid (-|logits|)], then
     counted down from [trials] where success is the likelier: a probability
     near [1] keeps the digits that [1 - sigmoid logits] would lose. *)
  let draw k s =
    let n = broadcast_to s trials and l = broadcast_to s logits in
    let x = Nx.Rng.binomial ~key:k n (Nx.sigmoid (Nx.neg (Nx.abs l))) in
    Nx.where (Nx.greater l (const l 0.)) (Nx.sub n x) x
  in
  let support () =
    let most =
      if Nx.numel trials = 0 then 0
      else Int32.to_int (Nx.item [] (Nx.max trials))
    in
    Support.Integer_interval (0, max 0 most)
  in
  let d =
    discrete ~family ~kind:Counts
      ~params:[ param "trials" trials; param "logits" logits ]
      ~shape:(broadcast family [ Nx.shape trials; Nx.shape logits ])
      ~checks:
        [
          check family "trials" "[0, inf)"
            (fun n -> Nx.greater_equal n (Nx.scalar_like n 0l))
            trials;
          check family "logits" "[-inf, inf]" not_nan logits;
        ]
      ~support
      ~bounds:(fun () ->
        (Nx.scalar_at (Nx.beside trials) dt 0., Nx.cast dt trials))
      ~factors ~draw logits
  in
  (* Beside the first parameter, as every family's. *)
  { d with at = Nx.beside trials }

let categorical ~logits =
  let family = "categorical" in
  let s = Nx.shape logits in
  let n = Array.length s in
  if n = 0 || s.(n - 1) = 0 then
    invalid_argf "Norn.Dist.categorical: logits of shape [%s] have no category"
      (shape_string s);
  let k = s.(n - 1) in
  let batch = Array.sub s 0 (n - 1) in
  let factors x =
    let lp = Nx.log_softmax ~axes:[ n - 1 ] logits in
    let b = broadcast family [ Nx.shape x; batch ] in
    let lp = Nx.broadcast_to (Array.append b [| k |]) lp in
    let ok =
      Nx.logical_and
        (Nx.greater_equal x (Nx.scalar_like x 0L))
        (Nx.less x (Nx.scalar_like x (Int64.of_int k)))
    in
    let i = Nx.unsqueeze ~axes:[ Array.length b ] (Nx.broadcast_to b x) in
    let f =
      Nx.squeeze
        ~axes:[ Array.length b ]
        (Nx.take_along_axis ~axis:(Array.length b) i lp)
    in
    outside (Nx.broadcast_to b ok) f
  in
  discrete ~family ~kind:Categories
    ~params:[ param "logits" logits ]
    ~shape:batch
    ~checks:
      [
        check family "logits" "[-inf, inf)"
          (fun l -> Nx.logical_and (not_nan l) (below_infinity l))
          logits;
      ]
    ~support:(fun () -> Support.Integer_interval (0, k - 1))
    ~bounds:(fixed logits (0., float_of_int (k - 1)))
    ~factors
    ~draw:(fun key shape ->
      Nx.Rng.categorical ~key:key (broadcast_to (Array.append shape [| k |]) logits))
    logits

(* Combinators *)

let iid s d =
  Array.iter
    (fun n ->
      if n < 0 then
        invalid_argf "Norn.Dist.iid: shape [%s] has a negative length"
          (shape_string s))
    s;
  {
    d with
    descr = Printf.sprintf "iid [%s] %s" (shape_string s) d.descr;
    shape = Array.append s d.shape;
  }

let log_factorial n =
  let s = ref 0. in
  for k = 2 to n do
    s := !s +. Float.log (float_of_int k)
  done;
  !s

let sorted n d =
  if Array.length d.shape <> 0 then
    invalid_argf "Norn.Dist.sorted: %s is not over scalars" d.descr;
  if n < 1 then invalid_argf "Norn.Dist.sorted: n = %d is not positive" n;
  let coords = match d.coords with Some b -> b | None -> assert false in
  let factors x =
    let ax = Nx.ndim x - 1 in
    let m = (Nx.shape x).(ax) in
    let range a b =
      Nx.slice (List.init ax (fun _ -> Nx.A) @ [ Nx.R (a, b) ]) x
    in
    let ok =
      Nx.all ~axes:[ ax ] (Nx.greater_equal (range 1 m) (range 0 (m - 1)))
    in
    outside ok (Nx.add_s (Nx.sum ~axes:[ ax ] (d.factors x)) (log_factorial n))
  in
  {
    d with
    family = "sorted";
    descr = Printf.sprintf "sorted %d %s" n d.descr;
    shape = [| n |];
    support = (fun () -> Support.Ordered);
    factors;
    draw = (fun k s -> fst (Nx.sort ~axis:(Array.length s - 1) (d.draw k s)));
    coords = Some (Bij.compose coords Bij.ordered);
    bounds = coord_bounds d.at d.dtype (Bij.compose coords Bij.ordered);
    standardize = None;
    quantile = None;
    membership = None;
  }

(* [sum_trailing k x] sums the last [k] axes of [x]. *)
let sum_trailing k x =
  if k <= 0 then x
  else Nx.sum ~axes:(List.init k (fun i -> Nx.ndim x - 1 - i)) x

let transform b d =
  let coords = match d.coords with Some c -> c | None -> assert false in
  let shape = Nx.shape (fst (Bij.forward b (Nx.zeros_at d.at d.dtype d.shape))) in
  let factors y =
    let u = Bij.inverse b y in
    let _, ld = Bij.forward b u in
    let f = d.factors u in
    let r = Nx.ndim f - Nx.ndim ld in
    let lp = Nx.sub (sum_trailing r f) (sum_trailing (-r) ld) in
    Nx.where (Nx.isnan lp) (neg_inf lp) lp
  in
  {
    d with
    family = "transform";
    descr = Format.asprintf "transform %a %s" Bij.pp b d.descr;
    shape;
    support = (fun () -> Bij.image b (d.support ()) shape);
    factors;
    draw = (fun k s -> fst (Bij.forward b (d.draw k (Bij.shape b s))));
    coords = Some (Bij.compose b coords);
    bounds = coord_bounds d.at d.dtype (Bij.compose b coords);
    standardize = None;
    quantile = None;
    membership = None;
  }

(* [hull_coords family d] is the bijector onto the support of every component of
   [d] together: the hull of their supports, read on the host. *)
let hull_coords family d =
  match d.coords with
  | None -> None
  | Some _ -> (
      let c v = Nx.scalar_at d.at d.dtype v in
      match d.support () with
      | Support.Real -> Some Bij.identity
      | Support.Greater 0. -> Some Bij.exp
      | Support.Greater a -> Some (Bij.greater ~low:(c a))
      | Support.Interval (a, b) -> Some (Bij.interval ~low:(c a) ~high:(c b))
      | Support.Simplex _ -> Some Bij.simplex
      | Support.Ordered -> Some Bij.ordered
      | Support.Sum_to_zero -> Some Bij.sum_to_zero
      | Support.Correlation_cholesky _ -> Some Bij.cholesky_corr
      | Support.Integers_from _ | Support.Integer_interval _ | Support.Boolean
        ->
          invalid_argf "Norn.Dist.%s: %s has no coordinates" family d.family)

(* [component_hull d] is the least lower and greatest upper bound of [d]'s
   components, along its leading axis: a binomial's per-component trials give
   per-component upper bounds. No component bounds the empty set, from [inf]
   to [-inf]. *)
let component_hull d () =
  let lo, hi = d.bounds () in
  if d.shape.(0) = 0 then
    let c v = Nx.scalar_at d.at d.dtype v in
    (c Float.infinity, c Float.neg_infinity)
  else
    ( Nx.min ~axes:[ 0 ] (Nx.broadcast_to d.shape lo),
      Nx.max ~axes:[ 0 ] (Nx.broadcast_to d.shape hi) )

let mixture ~logits d =
  let family = "mixture" in
  let ls = Nx.shape logits in
  if Array.length ls <> 1 || Array.length d.shape = 0 || d.shape.(0) <> ls.(0)
  then
    invalid_argf
      "Norn.Dist.mixture: logits of shape [%s] do not match %s of shape [%s]"
      (shape_string ls) d.descr (shape_string d.shape);
  let k = ls.(0) in
  let rest = Array.sub d.shape 1 (Array.length d.shape - 1) in
  let hull = hull_coords family d in
  (* Each component's log weight plus its log density of [x], with the component
     axis after [x]'s leading axes. *)
  let joint x =
    let r = ndim d.kind x - Array.length rest in
    let f =
      d.factors (apply d.kind { f = (fun x -> Nx.unsqueeze ~axes:[ r ] x) } x)
    in
    let f = sum_trailing (Nx.ndim f - r - 1) f in
    (r, Nx.add f (Nx.log_softmax ~axes:[ 0 ] logits))
  in
  let factors x =
    let r, j = joint x in
    Nx.logsumexp ~axes:[ r ] j
  in
  let membership x =
    let r, j = joint x in
    Nx.exp (Nx.log_softmax ~axes:[ r ] j)
  in
  let draw key s =
    let k1, k2 =
      match Nx.Rng.split key with [| a; b |] -> (a, b) | _ -> assert false
    in
    let batch = Array.sub s 0 (Array.length s - Array.length rest) in
    let r = Array.length batch in
    let all = d.draw k1 (Array.concat [ batch; [| k |]; rest ]) in
    let c =
      Nx.Rng.categorical ~key:k2 (broadcast_to (Array.append batch [| k |]) logits)
    in
    apply d.kind (select r c) all
  in
  {
    d with
    family;
    descr =
      Printf.sprintf "mixture(logits: %s [%d], %s)"
        (Nx.Dtype.name (Nx.dtype logits))
        k d.descr;
    shape = rest;
    checks =
      d.checks
      @ [
          check family "logits" "[-inf, inf)"
            (fun l -> Nx.logical_and (not_nan l) (below_infinity l))
            logits;
        ];
    factors;
    draw;
    coords = hull;
    bounds =
      (match hull with
      | Some h -> coord_bounds d.at d.dtype h
      | None -> component_hull d);
    standardize = None;
    quantile = None;
    membership = Some membership;
  }

(* Eliminators *)

let factors d x =
  checked "factors" d;
  d.factors x

let log_density d x =
  checked "log_density" d;
  Nx.sum (d.factors x)

let sample k d =
  let k = Nx.Rng.place d.at k in
  checked "sample" d;
  d.draw k d.shape

let quantile d p =
  match d.quantile with
  | None -> invalid_argf "Norn.Dist.quantile: %s has no quantile" d.family
  | Some q ->
      checked "quantile" d;
      q p

let coords d =
  match d.coords with
  | Some b -> b
  | None -> invalid_argf "Norn.Dist.coords: %s is discrete" d.family

let standardize d =
  match d.standardize with
  | Some b -> b
  | None ->
      invalid_argf "Norn.Dist.standardize: %s has no standard form" d.family

let mixture_membership d x =
  match d.membership with
  | Some m ->
      checked "mixture_membership" d;
      m x
  | None ->
      invalid_argf "Norn.Dist.mixture_membership: %s is not a mixture" d.family

let family d = d.family
let kind d = d.kind
let dtype d = d.dtype
let shape d = Array.copy d.shape
let placement d = d.at
let support d = d.support ()

let bounds d =
  let lo, hi = d.bounds () in
  (Nx.broadcast_to d.shape lo, Nx.broadcast_to d.shape hi)

let pp ppf d = Format.pp_print_string ppf d.descr

let check ?unless context d =
  run_checks ?unless context d.checks;
  { d with checked = true }

let valid d =
  List.fold_left
    (fun ok c -> Nx.logical_and ok (c.inside ()))
    (Nx.scalar_at d.at Nx.bool true) d.checks
