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

(* A family's distribution function, by its logarithms on each side, and
   its inverses on each side: [lower p] is the value whose CDF is [p] and
   [upper q] the one whose survival is [q], each exact to its tail where
   [p], [q <= 1/2]: [1 - p] loses the digits of a [p] near [1]. [below] and
   [above] say whether the support is bounded on that side. *)
type ('f, 'd) tails = {
  log_cdf : (float, 'f, 'd) Nx.t -> (float, 'f, 'd) Nx.t;
  log_sf : (float, 'f, 'd) Nx.t -> (float, 'f, 'd) Nx.t;
  lower : (float, 'f, 'd) Nx.t -> (float, 'f, 'd) Nx.t;
  upper : (float, 'f, 'd) Nx.t -> (float, 'f, 'd) Nx.t;
  below : bool;
  above : bool;
}

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
  tails : ('f, 'd) tails option;
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

let shape_of : type x f d. (x, f, d) kind -> x -> int array =
 fun kind x ->
  match kind with
  | Continuous -> Nx.shape x
  | Counts -> Nx.shape x
  | Categories -> Nx.shape x
  | Booleans -> Nx.shape x

let ndim kind x = Array.length (shape_of kind x)

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

(* [log_quotient x m log_m r] is [log (x / m)] for [x > 0], [m >= 0] and
   [r = x / m], [log_m] being [log m]: [inf] at [m = 0]. Within a factor
   [2^32] it takes [log (x / m)]; past it [log x - log_m], where [log_m]
   keeps the digits that an [m] near underflow has lost and its derivative
   stays finite where that of [x / m], [x / m^2], overflows. One [log] serves
   both: of [x / m] or of [x]. The ratio reads [x] and [m] only where it is
   taken. [deviance x m log_m] is [D(x, m)] from it. *)
let log_quotient x m log_m r =
  let band =
    Nx.logical_and
      (Nx.less_equal r (const r ratio_band))
      (Nx.greater_equal r (const r (1. /. ratio_band)))
  in
  let ratio = Nx.div (inside band x 1.) (inside band m 1.) in
  Nx.sub
    (Nx.log (Nx.where band ratio x))
    (Nx.where band (const log_m 0.) log_m)

let deviance x m log_m =
  let d = Nx.sub x m in
  deviance_of x ~d
    ~v:(Nx.div d (Nx.add x m))
    ~log_ratio:(log_quotient x m log_m (Nx.div x m))

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

(* Continuous densities *)

(* A gamma, beta or Dirichlet density is a Poisson, binomial or multinomial
   mass at real counts, rescaled, as R's dgamma.c and dbeta.c compute them
   through dpois_raw and dbinom_raw. From [Γ(a) = Γ(a + 1) / a]:

     gamma      f(x) = (α / x) Pois(α; m),  m = rate x
     dirichlet  f(x) = (Π α_k / A) / (Π x_k) Mult(α; A, x),  A = Σ α_k

   and Loader's form of the masses gives, for [δ_h(r) = δ(r) + log r / 2],

     log f = δ_h(A) - log A - (K - 1) log √(2π)
             - Σ_k [δ_h(α_k) - log α_k + log x_k + D(α_k, A x_k)]

   with [K = 1] and [A x] read as [m] for the gamma. No term grows with the
   concentrations faster than their logarithms, where [log Γ] sums cancel
   terms near [α log α]. [δ_h] stays finite with a finite derivative as a
   concentration nears [0], so one form serves every concentration; R's
   dgamma.c switches to [Pois(α - 1; m)] above 1 and dbeta.c to the direct
   form below 2. The terms of the parameters alone, [δ_h(α_k) - log α_k]
   and the whole's, are computed once, as the distribution is built.

   Error, as for the counts with [u = ε/2] per step:
   - [δ_h(r)] is within [50 u] absolutely and [2.5 u |log r|]; [log r],
     [log x] within [4 u] of theirs, and the bracket's sums round within
     [u] of their partial sums.
   - [D] within [30 u D + 6 ε |α_k - A x_k|] ([mean_deviance]), where
     [D <= |f| + Λ], [Λ] the sum of the logs' magnitudes plus [K], since
     every other term is within [Λ] of [0]; and the mean [A x_k] rounds
     within [2 u], moving [D] by [2 u |α_k - A x_k|]. [A]'s own rounding
     moves [Σ_k D] by its square: [Σ_k A x_k = A] makes the sum stationary
     in [A].
   Hence [ε (16 |f| + 48 Λ + 12 M)] for [M = Σ_k |α_k - A x_k|]. The
   second term is the conditioning of the density in the point: moving
   [x_k] by [ε] moves [log f] by about [ε |α_k - A x_k|]. The Student t's
   normaliser is a ratio of gammas whose [δ] terms are those of reals
   [ν / 2] and [ν / 2 + 1/2] apart by [1/2]: the same bound with
   [Λ = 1 + |log ν| + |log σ|] and no mean. *)

(* [ratio_log (x, (lx, (m, log_m)))] is [log (x / m)] for [x > 0] and
   [m >= 0], [lx] and [log_m] being their logs: [log1p t],
   [t = (x - m) / m], where [x / m] lies between [ε] and the floats' largest
   over [e], and [lx - log_m] elsewhere, the arm read from [lx - log_m]
   itself. Below [ε] the logs' error, [4 u (|lx| + |log_m|)], times [x] is
   below [u m <= u D]; above, [|log (x / m)| > 708] and the error is within
   [16 u] of it at float64. Its derivative is stated, that of
   [lx - log_m]: taken through the quotient it would form [x / m²], and
   [dm / m] overflows where [m] underflows, while the logs' derivatives,
   formed by the caller from the parameters and the point, stay finite.
   No arm needs a guard and none is differentiated. *)
let ratio_log args =
  Rune.custom_jvp
    Nx.Ptree.(pair tensor (pair tensor (pair tensor tensor)))
    Nx.Ptree.tensor
    (fun (x, (lx, (m, log_m))) ->
      let logs = Nx.sub lx log_m in
      let dt = Nx.dtype logs in
      let near =
        Nx.logical_and
          (Nx.greater_equal logs (const logs (Float.log (Prec.eps dt))))
          (Nx.less_equal logs (const logs (Float.log (Prec.huge dt) -. 1.)))
      in
      let l = Nx.where near (Nx.log1p (Nx.div (Nx.sub x m) m)) logs in
      (l, fun (_, (dlx, (_, dlog_m))) -> Nx.sub dlx dlog_m))
    args

(* [mean_deviance x lx m log_m] is [D(x, m)] for a real count [x > 0] whose
   distance from its mean [m] the bound counts, [lx] and [log_m] being
   their logs: [x log (x / m) - (x - m)] from [ratio_log], one
   transcendental per element and no series. Near [m], [x - m] is exact
   (Sterbenz) and [log1p t] within [2 ε] of itself, so the two terms near
   [x - m] leave [6 ε |x - m|]. Below [x = m / 2], [1 + t] carries the
   rounding [u] of [x - m] relative to [m], [u m <= 7 u D] in [D], which is
   at least [0.15 m] there. *)
let mean_deviance x lx m log_m =
  Nx.sub (Nx.mul x (ratio_log (x, (lx, (m, log_m))))) (Nx.sub x m)

(* [head a] is [-(δ_h(a) - log a)], a component's term of its parameter
   alone. *)
let head a = Nx.sub (Nx.log a) (delta_half a)

(* [share (h, a, la) lx m log_m] is a component's bracket negated:
   [h - lx - D(a, m)], [h = head a] and [la = log a] its parameter's terms,
   [lx] the log of its share of the whole and [log_m] that of the mean
   [m]. *)
let share (h, a, la) lx m log_m =
  Nx.sub (Nx.sub h lx) (mean_deviance a la m log_m)

(* [gamma_density (c, rate, lr) x] is a gamma's log density at [x], [c]
   its count's terms ([component]) and [lr] the log of the rate. The
   inverse gamma's density at [x] is the gamma's at [1 / x] times
   [1 / x^2]: its mean count is [scale / x]. *)
let gamma_density (c, rate, lr) x =
  let lx = Nx.log x in
  share c lx (Nx.mul rate x) (Nx.add lr lx)

let inverse_gamma_density (c, scale, ls) x =
  let lx = Nx.log x in
  share c lx (Nx.div scale x) (Nx.sub ls lx)

(* [component ~shift a] is [share]'s terms of a count [a], its head less
   [shift]. *)
let component ?(shift = 0.) a = (Nx.sub_s (head a) shift, a, Nx.log a)

(* [whole n k] is [δ_h(n) - log n - (k - 1) log √(2π)]. *)
let whole n k =
  Nx.sub_s
    (Nx.sub (delta_half n) (Nx.log n))
    (float_of_int (k - 1) *. half_log_2pi)

(* A beta is the dirichlet of [(x, 1 - x)]: [log1p (-x)] keeps the digits of
   [log (1 - x)] near [0]. Its terms of the parameters alone are [whole],
   both heads, [n = a + b] and [log n]. *)
let beta_terms a b =
  let n = Nx.add a b in
  let w = Nx.add (whole n 2) (Nx.add (head a) (head b)) in
  (w, (a, Nx.log a), (b, Nx.log b), n, Nx.log n)

let beta_density (w, (a, la), (b, lb), n, ln) x =
  let lx = Nx.log x and l1 = Nx.log1p (Nx.neg x) in
  let rest = Nx.sub (const x 1.) x in
  let da = mean_deviance a la (Nx.mul n x) (Nx.add ln lx) in
  let db = mean_deviance b lb (Nx.mul n rest) (Nx.add ln l1) in
  Nx.sub (Nx.sub (Nx.sub (Nx.sub w lx) l1) da) db

(* [dirichlet_density (w, c, n, ln) x] along the last axis of [a] and
   [x], [x] read as [x / Σx]: within the rounding a point keeps off the
   simplex, the density of its nearest point along the ray, so the sum's
   rounding does not move the means by [A ε]. *)
let dirichlet_terms a =
  let k = (Nx.shape a).(Nx.ndim a - 1) in
  let n = Nx.sum ~axes:[ Nx.ndim a - 1 ] ~keepdims:true a in
  (Nx.squeeze ~axes:[ Nx.ndim a - 1 ] (whole n k), component a, n, Nx.log n)

let dirichlet_density (w, c, n, ln) x =
  let ax = Nx.ndim x - 1 in
  let x = Nx.div x (Nx.sum ~axes:[ ax ] ~keepdims:true x) in
  let lx = Nx.log x in
  let parts = share c lx (Nx.mul n x) (Nx.add ln lx) in
  Nx.add w (Nx.sum ~axes:[ Nx.ndim parts - 1 ] parts)

(* [log_ratio df] is [log ((ν + 1) / ν)]: [log1p (1 / ν)] above 1,
   [log1p ν - log ν] below, where [1 / ν] may overflow. *)
let log_ratio df =
  let above = Nx.greater df (const df 1.) in
  let hi = inside above df 1. and lo = Nx.where above (const df 1.) df in
  Nx.where above
    (Nx.log1p (Nx.recip hi))
    (Nx.sub (Nx.log1p lo) (Nx.log lo))

(* [student_normaliser df] is
   [log Γ((ν + 1) / 2) - log Γ(ν / 2) - log (ν π) / 2], R's dt.c's form:
   [δ(h + 1/2) - δ(h) - D(h, h + 1/2) - log √(2π)] for [h = ν / 2]. From
   [h = 1] up the [δ] are the reals' series; below, where [δ] grows as
   [-log h / 2], they are [δ_h] less [log (1 + 1/ν) / 2]. *)
let student_normaliser df =
  let h = Nx.mul_s df 0.5 in
  let lr = log_ratio df in
  let large = Nx.greater_equal h (const h 1.) in
  let hl = Nx.where large h (const h 1.) in
  let hs = Nx.where large (const h 0.5) h in
  let deltas =
    Nx.where large
      (Nx.sub (delta_real (Nx.add_s hl 0.5)) (delta_real hl))
      (Nx.sub
         (Nx.sub (delta_half (Nx.add_s hs 0.5)) (delta_half hs))
         (Nx.mul_s lr 0.5))
  in
  let dev =
    deviance_of h ~d:(const h (-0.5))
      ~v:(Nx.div (const df (-1.)) (Nx.add_s (Nx.mul_s df 2.) 1.))
      ~log_ratio:(Nx.neg lr)
  in
  Nx.sub_s (Nx.sub deltas dev) half_log_2pi

(* The Student t's terms of its parameters alone: its normaliser less
   [log σ], [√ν] and [(ν + 1) / 2]. *)
let student_terms df scale =
  ( Nx.sub (student_normaliser df) (Nx.log scale),
    Nx.sqrt df,
    Nx.mul_s (Nx.add_s df 1.) 0.5 )

(* [log1p (z² / ν)] is [2 log max (w, 1) + log1p (min (w, 1/w)²)] for
   [w = |z| / √ν]: no square overflows, and both arms are the same
   expression. At [w = 1] each selection takes the side of [w <= 1], whose
   derivative is the whole's: a tie split between [max]'s operands would
   halve it. *)
let student_t_density (c, root, half1) loc scale x =
  let w = Nx.div (Nx.abs (Nx.div (Nx.sub x loc) scale)) root in
  let beyond = Nx.greater w (const w 1.) in
  let big = Nx.where beyond w (const w 1.) in
  let small = Nx.where beyond (Nx.recip big) w in
  let l = Nx.add (Nx.mul_s (Nx.log big) 2.) (Nx.log1p (Nx.square small)) in
  Nx.sub c (Nx.mul half1 l)

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

(* [quantile_of t p] reads [p] from its smaller tail: [1 - p] is exact from
   [p = 1/2] up (Sterbenz). Each arm reads [p] only where it is taken. *)
let quantile_of t p =
  let low = Nx.less_equal p (const p 0.5) in
  Nx.where low
    (t.lower (inside low p 0.5))
    (t.upper (Nx.where low (const p 0.5) (Nx.sub (const p 1.) p)))

(* A continuous value's bounds are its coordinates' image of the reals. *)
let coord_bounds at dtype coords () =
  Bij.bounds coords
    ( Nx.scalar_at at dtype Float.neg_infinity,
      Nx.scalar_at at dtype Float.infinity )

let continuous ~family ~params ~shape ~checks ~support ~factors ~draw ~coords
    ?standardize ?quantile ?tails like =
  let dtype = Nx.dtype like and at = Nx.beside like in
  let quantile =
    match (quantile, tails) with
    | Some q, _ -> Some q
    | None, Some t -> Some (quantile_of t)
    | None, None -> None
  in
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
    tails;
    membership = None;
    kind = Continuous;
    at;
  }

(* [mirror lc lower] is the tails of a distribution symmetric about [0], from
   its log CDF and its lower inverse. *)
let mirror lc lower =
  {
    log_cdf = lc;
    log_sf = (fun z -> lc (Nx.neg z));
    lower;
    upper = (fun q -> Nx.neg (lower q));
    below = false;
    above = false;
  }

let location_scale family ~loc ~scale ~density ~draw ~tails () =
  let dt = Nx.dtype loc in
  let z x = Nx.div (Nx.sub x loc) scale in
  let x z = Nx.add loc (Nx.mul scale z) in
  continuous ~family
    ~params:[ param "loc" loc; param "scale" scale ]
    ~shape:(broadcast family [ Nx.shape loc; Nx.shape scale ])
    ~checks:[ in_reals family "loc" loc; in_positive family "scale" scale ]
    ~support:(fun () -> Support.Real)
    ~factors:(fun v -> Nx.sub (density (z v)) (Nx.log scale))
    ~draw:(fun k s -> x (draw k dt s))
    ~coords:Bij.identity ~standardize:(Bij.affine ~loc ~scale)
    ~tails:
      {
        tails with
        log_cdf = (fun v -> tails.log_cdf (z v));
        log_sf = (fun v -> tails.log_sf (z v));
        lower = (fun p -> x (tails.lower p));
        upper = (fun q -> x (tails.upper q));
      }
    loc

let normal ~loc ~scale =
  location_scale "normal" ~loc ~scale
    ~density:(fun z ->
      Nx.add_s (Nx.mul_s (Nx.square z) (-0.5)) (-.half_log_2pi))
    ~draw:Noise.normal
    ~tails:(mirror Nx.log_ndtr Nx.ndtri)
    ()

(* The Cauchy CDF is [atan2 (1, -z) / π], exact to its tail on both sides.
   Its lower inverse is [-1 / tan (π p)] below [1/4], where [tan (π (p -
   1/2))] would meet the rounding of [p - 1/2] near its pole, and that form
   from [1/4], where [p - 1/2] is exact. *)
let cauchy_cdf z =
  Nx.sub_s (Nx.log (Nx.atan2 (const z 1.) (Nx.neg z))) log_pi

let cauchy_lower p =
  let deep = Nx.less p (const p 0.25) in
  let lo = inside deep p 0.25 and hi = Nx.where deep (const p 0.5) p in
  Nx.where deep
    (Nx.neg (Nx.recip (Nx.tan (Nx.mul_s lo Float.pi))))
    (Nx.tan (Nx.mul_s (Nx.sub_s hi 0.5) Float.pi))

let cauchy_tails () = mirror cauchy_cdf cauchy_lower

let cauchy ~loc ~scale =
  location_scale "cauchy" ~loc ~scale
    ~density:(fun z -> Nx.add_s (Nx.neg (Nx.log1p (Nx.square z))) (-.log_pi))
    ~draw:(fun k dt s -> quantile_of (cauchy_tails ()) (open_unit k dt s))
    ~tails:(cauchy_tails ()) ()

(* The Laplace CDF is [exp z / 2] below [0] and [1 - exp (-z) / 2] above;
   its lower inverse [log (2p)], whose slope at the median is [2]. *)
let laplace_cdf z =
  let neg = Nx.less z (const z 0.) in
  let lo = inside neg z 0. and hi = Nx.where neg (const z 0.) z in
  Nx.where neg
    (Nx.sub_s lo (Float.log 2.))
    (Nx.log1p (Nx.mul_s (Nx.exp (Nx.neg hi)) (-0.5)))

let laplace_tails () = mirror laplace_cdf (fun p -> Nx.log (Nx.mul_s p 2.))

let laplace ~loc ~scale =
  location_scale "laplace" ~loc ~scale
    ~density:(fun z -> Nx.sub_s (Nx.neg (Nx.abs z)) (Float.log 2.))
    ~draw:(fun k dt s -> quantile_of (laplace_tails ()) (open_unit k dt s))
    ~tails:(laplace_tails ()) ()

let logistic_tails () =
  mirror log_sigmoid (fun p -> Nx.sub (Nx.log p) (Nx.log1p (Nx.neg p)))

let logistic ~loc ~scale =
  location_scale "logistic" ~loc ~scale
    ~density:(fun z -> Nx.sub (Nx.neg z) (Nx.mul_s (Nx.softplus (Nx.neg z)) 2.))
    ~draw:(fun k dt s -> quantile_of (logistic_tails ()) (open_unit k dt s))
    ~tails:(logistic_tails ()) ()

let student_t ~df ~loc ~scale =
  let family = "student_t" in
  let dt = Nx.dtype loc in
  let half_df = Nx.mul_s df 0.5 in
  let factors =
    if narrow dt then
      let t = student_terms (widen df) (widen scale) in
      let loc = widen loc and scale = widen scale in
      fun x -> Nx.cast dt (student_t_density t loc scale (widen x))
    else
      let t = student_terms df scale in
      fun x -> student_t_density t loc scale x
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

(* [positive lc ls lower upper] is the tails of a family on [(0, inf)]. *)
let positive_tails log_cdf log_sf lower upper =
  { log_cdf; log_sf; lower; upper; below = true; above = false }

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
  let z x = Nx.div (Nx.sub (Nx.log x) loc) scale in
  let x z = Nx.exp (Nx.add loc (Nx.mul scale z)) in
  continuous ~family
    ~params:[ param "loc" loc; param "scale" scale ]
    ~shape:(broadcast family [ Nx.shape loc; Nx.shape scale ])
    ~checks:[ in_reals family "loc" loc; in_positive family "scale" scale ]
    ~support:(fun () -> Support.Greater 0.)
    ~factors
    ~draw:(fun k s -> x (Noise.normal k dt s))
    ~coords:Bij.exp
    ~standardize:(Bij.compose Bij.exp (Bij.affine ~loc ~scale))
    ~tails:
      (positive_tails
         (fun v -> Nx.log_ndtr (z v))
         (fun v -> Nx.log_ndtr (Nx.neg (z v)))
         (fun p -> x (Nx.ndtri p))
         (fun q -> x (Nx.neg (Nx.ndtri q))))
    loc

(* A family on [(0, inf)] whose density at [x] is [density x], the value
   replaced by [1] outside before the arithmetic. *)
let positive_family ~family ~params ~shape ~checks ~density ~draw ?quantile
    ~tails ~closed like =
  let factors x =
    let ok = if closed then non_negative x else positive x in
    outside ok (density (inside ok x 1.))
  in
  continuous ~family ~params ~shape ~checks
    ~support:(fun () -> Support.Greater 0.)
    ~factors ~draw ~coords:Bij.exp ?quantile ~tails like

(* The half-normal CDF is [erf (z / √2)], which keeps its digits near [0];
   past [z = 1] it is [1 - 2 Φ(-z)], whose log is [log1p] of a small value.
   Its lower inverse is [√2 erfinv p] and its upper [-Φ⁻¹(q / 2)]. *)
let half_normal_cdf z =
  let near = Nx.less z (const z 1.) in
  let lo = inside near z 0.5 and hi = Nx.where near (const z 1.) z in
  Nx.where near
    (Nx.log (Nx.erf (Nx.div_s lo (Float.sqrt 2.))))
    (Nx.log1p (Nx.mul_s (Nx.ndtr (Nx.neg hi)) (-2.)))

let half_normal ~scale =
  let family = "half_normal" in
  let dt = Nx.dtype scale in
  let z x = Nx.div x scale and x z = Nx.mul scale z in
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
    ~tails:
      (positive_tails
         (fun v -> half_normal_cdf (z v))
         (fun v -> Nx.add_s (Nx.log_ndtr (Nx.neg (z v))) (Float.log 2.))
         (fun p -> x (Nx.mul_s (Nx.erfinv p) (Float.sqrt 2.)))
         (fun q -> x (Nx.neg (Nx.ndtri (Nx.mul_s q 0.5)))))
    scale

(* The half-Cauchy CDF is [2 atan z / π] and its survival
   [2 atan2 (1, z) / π]; their inverses [tan (π p / 2)] and
   [1 / tan (π q / 2)] stay away from the tangent's pole. *)
let half_cauchy ~scale =
  let family = "half_cauchy" in
  let dt = Nx.dtype scale in
  let z x = Nx.div x scale and x z = Nx.mul scale z in
  let log_2_pi = Float.log 2. -. log_pi in
  let tails =
    positive_tails
      (fun v -> Nx.add_s (Nx.log (Nx.atan (z v))) log_2_pi)
      (fun v -> Nx.add_s (Nx.log (Nx.atan2 (const v 1.) (z v))) log_2_pi)
      (fun p -> x (Nx.tan (Nx.mul_s p (Float.pi /. 2.))))
      (fun q -> x (Nx.recip (Nx.tan (Nx.mul_s q (Float.pi /. 2.)))))
  in
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
    ~draw:(fun k s -> quantile_of tails (open_unit k dt s))
    ~tails scale

(* [log1mexp s] is [log (1 - exp (-s))] for [s >= 0], by Mächler's rule
   (Accurately Computing log(1 - exp(-|a|)), 2012, eq. 7): [log (-expm1
   (-s))] up to [log 2], [log1p (-exp (-s))] past it, where each keeps the
   digits the other cancels. *)
let log1mexp s =
  let near = Nx.less s (const s (Float.log 2.)) in
  let lo = inside near s 1. and hi = Nx.where near (const s 1.) s in
  Nx.where near
    (Nx.log (Nx.neg (Nx.expm1 (Nx.neg lo))))
    (Nx.log1p (Nx.neg (Nx.exp (Nx.neg hi))))

let exponential ~rate =
  let family = "exponential" in
  let dt = Nx.dtype rate in
  positive_family ~family ~closed:true
    ~params:[ param "rate" rate ]
    ~shape:(Nx.shape rate)
    ~checks:[ in_positive family "rate" rate ]
    ~density:(fun x -> Nx.sub (Nx.log rate) (Nx.mul rate x))
    ~draw:(fun k s -> Nx.div (Noise.exponential k dt s) rate)
    ~tails:
      (positive_tails
         (fun x -> log1mexp (Nx.mul rate x))
         (fun x -> Nx.neg (Nx.mul rate x))
         (fun p -> Nx.div (Nx.neg (Nx.log1p (Nx.neg p))) rate)
         (fun q -> Nx.div (Nx.neg (Nx.log q)) rate))
    rate

(* [exp_times l t] is [exp l * t], exactly [0] where [t] is: as [exp (l / 3)]
   three times, finite wherever [|l| < 2129], where [exp l] alone overflows
   past [709.8] and a zero [t] would make NaN. *)
let exp_times l t =
  let e = Nx.exp (Nx.div_s l 3.) in
  Nx.mul (Nx.mul (Nx.mul t e) e) e

(* The inverse gamma's quantile [x = scale / y], [y] the unit gamma's upper
   quantile at [p], since the CDF at [x] is [Q(a, scale / x)]; its upper
   inverse at [q] takes [y]'s lower quantile. Reverse mode through
   [scale / y] forms [scale / y²], which overflows once [x] exceeds
   [sqrt scale 2^512], so the rule states the derivatives:
   [∂x/∂p = 1 / f(x) = scale Γ(a) e^y / y^(a+1)], taken in logs, negated for
   [q], [∂x/∂scale = 1 / y], and [∂x/∂a = -x (∂y/∂a) / y], from the
   inverse's own derivative. A tangent map serves forward mode as well as reverse, and its
   [jvp] runs only under a differentiation. Each term is [0] where its
   tangent is, so a tangent that is zero leaves the others' sum alone. *)
let inverse_gamma_quantile ~upper args =
  let inverse a p = if upper then Nx.gammaincinv a p else Nx.gammainccinv a p in
  Rune.custom_jvp
    Nx.Ptree.(pair tensor (pair tensor tensor))
    Nx.Ptree.tensor
    (fun (a, (scale, p)) ->
      let y = inverse a p in
      let x = Nx.div scale y in
      let tangent (da, (dscale, dp)) =
        let log_inv_f =
          Nx.sub
            (Nx.add (Nx.add (Nx.log scale) y) (Nx.lgamma a))
            (Nx.mul (Nx.add_s a 1.) (Nx.log y))
        in
        let a' = Nx.broadcast_to (Nx.shape x) a in
        let _, dy_da = Rune.jvp' (fun a -> inverse a p) a' (Nx.ones_like a') in
        let along_a = Nx.neg (Nx.mul (Nx.mul da x) (Nx.div dy_da y)) in
        let along_p = exp_times log_inv_f dp in
        let along_p = if upper then Nx.neg along_p else along_p in
        Nx.add along_a (Nx.add (Nx.div dscale y) along_p)
      in
      (x, tangent))
    args

let gamma ~concentration ~rate =
  let family = "gamma" in
  let a = concentration in
  let dt = Nx.dtype a in
  positive_family ~family ~closed:false
    ~params:[ param "concentration" a; param "rate" rate ]
    ~shape:(broadcast family [ Nx.shape a; Nx.shape rate ])
    ~checks:
      [ in_positive family "concentration" a; in_positive family "rate" rate ]
    ~density:
      (let terms a rate =
         (component ~shift:half_log_2pi a, rate, Nx.log rate)
       in
       if narrow dt then
         let t = terms (widen a) (widen rate) in
         fun x -> Nx.cast dt (gamma_density t (widen x))
       else
         let t = terms a rate in
         fun x -> gamma_density t x)
    ~draw:(fun k s -> Nx.div (Nx.Rng.gamma ~key:k (broadcast_to s a)) rate)
    ~quantile:(fun p -> Nx.div (Nx.gammaincinv a p) rate)
    ~tails:
      (positive_tails
         (fun x -> Nx.log_gammainc a (Nx.mul rate x))
         (fun x -> Nx.log_gammaincc a (Nx.mul rate x))
         (fun p -> Nx.div (Nx.gammaincinv a p) rate)
         (fun q -> Nx.div (Nx.gammainccinv a q) rate))
    a

let inverse_gamma ~concentration ~scale =
  let family = "inverse_gamma" in
  let a = concentration in
  let dt = Nx.dtype a in
  let q ~upper p = inverse_gamma_quantile ~upper (a, (scale, p)) in
  positive_family ~family ~closed:false
    ~params:[ param "concentration" a; param "scale" scale ]
    ~shape:(broadcast family [ Nx.shape a; Nx.shape scale ])
    ~checks:
      [ in_positive family "concentration" a; in_positive family "scale" scale ]
    ~density:
      (let terms a scale =
         (component ~shift:half_log_2pi a, scale, Nx.log scale)
       in
       if narrow dt then
         let t = terms (widen a) (widen scale) in
         fun x -> Nx.cast dt (inverse_gamma_density t (widen x))
       else
         let t = terms a scale in
         fun x -> inverse_gamma_density t x)
    ~draw:(fun k s -> Nx.div scale (Nx.Rng.gamma ~key:k (broadcast_to s a)))
    ~quantile:(q ~upper:false)
    ~tails:
      (positive_tails
         (fun x -> Nx.log_gammaincc a (Nx.div scale x))
         (fun x -> Nx.log_gammainc a (Nx.div scale x))
         (q ~upper:false) (q ~upper:true))
    a

let beta ~a ~b =
  let family = "beta" in
  let dt = Nx.dtype a in
  let density =
    if narrow dt then
      let t = beta_terms (widen a) (widen b) in
      fun x -> Nx.cast dt (beta_density t (widen x))
    else
      let t = beta_terms a b in
      fun x -> beta_density t x
  in
  let factors x =
    let ok = Nx.logical_and (positive x) (Nx.less x (const x 1.)) in
    outside ok (density (inside ok x 0.5))
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
  let share v = Nx.sub (Nx.log v) (Nx.log width) in
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
    ~tails:
      {
        log_cdf = (fun x -> share (Nx.sub x low));
        log_sf = (fun x -> share (Nx.sub high x));
        lower = (fun p -> Nx.add low (Nx.mul width p));
        upper = (fun q -> Nx.sub high (Nx.mul width q));
        below = true;
        above = true;
      }
    low

(* Von Mises *)

(* [log (2π I₀(κ))] is [κ + log (2π i0e κ)], finite where [I₀] overflows,
   and [κ (cos d - 1)] is [-2κ sin² (d / 2)], which keeps its digits near the
   mode where [cos d] rounds to [1]. *)
let von_mises ~loc ~concentration =
  let family = "von_mises" in
  let k = concentration in
  let pi = Nx.scalar_like loc Float.pi in
  let normaliser = Nx.add_s (Nx.log (Nx.i0e k)) (Float.log (2. *. Float.pi)) in
  let factors x =
    let ok =
      Nx.logical_and
        (Nx.greater_equal x (Nx.neg pi))
        (Nx.less_equal x pi)
    in
    let half = Nx.sin (Nx.mul_s (Nx.sub x loc) 0.5) in
    outside ok (Nx.sub (Nx.mul (Nx.mul_s k (-2.)) (Nx.square half)) normaliser)
  in
  (* Best and Fisher's draw about [0], moved by [loc] and wrapped back onto
     the circle. *)
  let draw key s =
    let theta = Nx.add (Nx.Rng.von_mises ~key (broadcast_to s k)) loc in
    Nx.atan2 (Nx.sin theta) (Nx.cos theta)
  in
  continuous ~family
    ~params:[ param "loc" loc; param "concentration" k ]
    ~shape:(broadcast family [ Nx.shape loc; Nx.shape k ])
    ~checks:
      [
        in_reals family "loc" loc;
        check family "concentration" "[0, inf)"
          (fun k -> Nx.logical_and (non_negative k) (below_infinity k))
          k;
      ]
    ~support:(fun () -> Support.Interval (-.Float.pi, Float.pi))
    ~factors ~draw
    ~coords:(Bij.interval ~low:(Nx.neg pi) ~high:pi)
    loc

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
  let tol = 4. *. float_of_int k *. Prec.eps dt in
  let density =
    if narrow dt then
      let t = dirichlet_terms (widen a) in
      fun x -> Nx.cast dt (dirichlet_density t (widen x))
    else
      let t = dirichlet_terms a in
      fun x -> dirichlet_density t x
  in
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
    outside ok (density x)
  in
  continuous ~family
    ~params:[ param "concentration" a ]
    ~shape:s
    ~checks:[ in_positive family "concentration" a ]
    ~support:(fun () -> Support.Simplex k)
    ~factors
    ~draw:(fun key shape ->
      (* A format narrower than float32 draws at float32, whose components'
         range holds what its own would underflow, and every format keeps
         each component at least its least positive value: a component of
         [0] lies outside the support. *)
      let x =
        if narrow dt then
          Nx.cast dt (Nx.Rng.dirichlet ~key (broadcast_to shape (widen a)))
        else Nx.Rng.dirichlet ~key (broadcast_to shape a)
      in
      let f = Nx.Dtype.float_format dt in
      Nx.maximum x (const x (f.min_normal *. f.epsilon)))
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

(* LKJ *)

(* [lkj_normaliser n eta] is [log c_n(η)], the integral of [det C^(η - 1)]
   over the [n × n] correlation matrices. Lewandowski, Kurowicka and Joe
   (Generating random correlation matrices based on vines and extended onion
   method, 2009) give [c_n = 2^(Σ_k (2η - 2 + n - k)(n - k))
   Π_k B(b_k, b_k)^(n - k)] for [b_k = η + (n - 1 - k) / 2],
   [k = 1, ..., n - 1]. As [2η - 2 + n - k = 2 b_k - 1], the [k]-th factor
   is [(2^(2 b_k - 1) B(b_k, b_k))^(n - k)], and its base is
   [2 / β_k(1/2)], [β_k] the beta density of shapes [b_k] and [b_k]: its
   saddle-point form keeps the digits that [2 b_k log 2 + log B(b_k, b_k)]
   cancels at a large [η]. *)
let lkj_normaliser n eta =
  let at = Nx.beside eta and dt = Nx.dtype eta in
  let row f = Nx.place at (Nx.create dt [| n - 1 |] (Array.init (n - 1) f)) in
  let shift = row (fun i -> float_of_int (n - 2 - i) /. 2.) in
  let weight = row (fun i -> float_of_int (n - 1 - i)) in
  let b = Nx.add (Nx.unsqueeze ~axes:[ Nx.ndim eta ] eta) shift in
  let mode = beta_density (beta_terms b b) (const b 0.5) in
  Nx.sum ~axes:[ Nx.ndim b - 1 ]
    (Nx.mul weight (Nx.sub (const b (Float.log 2.)) mode))

(* [lkj_density (lkj_power n eta) l] is
   [Σ_(i >= 2) (n - i + 2η - 2) log L_ii - log c_n], rows counted from 1: the
   density of [C = L Lᵀ] times the Jacobian [Π_i L_ii^(n - i)] of the map
   from [L]'s strict lower triangle to [C]'s. Row 1's diagonal is [1], its
   weight [0]. [lkj_power] holds the parameter's terms, built once. *)
let lkj_power n eta =
  let at = Nx.beside eta and dt = Nx.dtype eta in
  let shift =
    Nx.place at
      (Nx.create dt [| n |]
         (Array.init n (fun i ->
              if i = 0 then 0. else float_of_int (n - i - 3))))
  in
  let first =
    Nx.place at
      (Nx.create dt [| n |] (Array.init n (fun i -> if i = 0 then 0. else 1.)))
  in
  let power =
    Nx.mul first (Nx.add shift (Nx.mul_s (Nx.unsqueeze ~axes:[ Nx.ndim eta ] eta) 2.))
  in
  (power, lkj_normaliser n eta)

let lkj_density (power, normaliser) l =
  let diag = Nx.log (Nx.diagonal l) in
  Nx.sub (Nx.sum ~axes:[ Nx.ndim diag - 1 ] (Nx.mul power diag)) normaliser

let lkj_cholesky n ~concentration =
  let family = "lkj_cholesky" in
  let eta = concentration in
  let dt = Nx.dtype eta and at = Nx.beside eta in
  if n < 2 then
    invalid_argf "Norn.Dist.lkj_cholesky: n = %d is fewer than two" n;
  let shape = Array.append (Nx.shape eta) [| n; n |] in
  (* A row's squared norm rounds within about [n ε] of one. *)
  let tol = 4. *. float_of_int n *. Prec.eps dt in
  let density =
    if narrow dt then
      let t = lkj_power n (widen eta) in
      fun l -> Nx.cast dt (lkj_density t (widen l))
    else
      let t = lkj_power n eta in
      fun l -> lkj_density t l
  in
  let factors l =
    let rows = Nx.ndim l - 1 in
    let unit_rows =
      Nx.less_equal
        (Nx.abs (Nx.sub_s (Nx.sum ~axes:[ rows ] (Nx.square l)) 1.))
        (const l tol)
    in
    let ok =
      Nx.logical_and
        (Nx.all ~axes:[ rows - 1 ]
           (Nx.logical_and unit_rows (positive (Nx.diagonal l))))
        (Nx.all ~axes:[ rows - 1; rows ]
           (Nx.equal (Nx.triu ~k:1 l) (const l 0.)))
    in
    let ok_rows = Nx.unsqueeze ~axes:[ rows - 1; rows ] ok in
    let l = Nx.where ok_rows l (Nx.place at (Nx.eye dt n)) in
    outside ok
      (density l)
  in
  (* Lewandowski, Kurowicka and Joe's onion method (2009, §3.2): row [k] of
     [L], from [k = 1], is [(√(1 - c) u, √c)] for [c] beta of shapes
     [η + (n - 1 - k) / 2] and [k / 2], the complement of the paper's [y],
     and [u] uniform on the sphere of [k] dimensions, a normal vector over
     its norm. The diagonal is drawn itself, inside (0, 1)
     ({!Nx.Rng.beta}), where [1 - y] would round to [0] at a small [η]. *)
  let onion eta key s =
    let dt = Nx.dtype eta and at = Nx.beside eta in
    let k1, k2 =
      match Nx.Rng.split key with [| a; b |] -> (a, b) | _ -> assert false
    in
    let batch = Array.sub s 0 (Array.length s - 2) in
    let m = n - 1 in
    let row f = Nx.place at (Nx.create dt [| m |] (Array.init m f)) in
    let a = row (fun i -> float_of_int (i + 1) /. 2.) in
    let b =
      Nx.add
        (Nx.unsqueeze ~axes:[ Nx.ndim eta ] eta)
        (row (fun i -> float_of_int (n - 2 - i) /. 2.))
    in
    let rows = Array.append batch [| m |] in
    let c = Nx.Rng.beta ~key:k1 (broadcast_to rows b) (broadcast_to rows a) in
    let z = Nx.tril (Noise.normal k2 dt (Array.append batch [| m; m |])) in
    let norm =
      Nx.sqrt
        (Nx.sum ~axes:[ Array.length batch + 1 ] ~keepdims:true (Nx.square z))
    in
    let y = Nx.sub (const c 1.) c in
    let y' = Nx.unsqueeze ~axes:[ Array.length batch + 1 ] y in
    let w = Nx.mul (Nx.div z norm) (Nx.sqrt y') in
    let widths =
      Array.append (Array.map (fun _ -> (0, 0)) batch) [| (1, 0); (0, 1) |]
    in
    let off = Nx.pad widths 0. w in
    let one = Nx.ones_at at dt (Array.append batch [| 1 |]) in
    let diag =
      Nx.concatenate ~axis:(Array.length batch)
        [ one; Nx.sqrt c ]
    in
    Nx.add off
      (Nx.mul (Nx.place at (Nx.eye dt n))
         (Nx.unsqueeze ~axes:[ Array.length batch ] diag))
  in
  (* A format narrower than float32 draws at float32 and rounds once, as
     nx's samplers do: in its own range a row's normal over its norm
     underflows. A diagonal the rounding takes to [0] is the format's least
     positive value. *)
  let draw =
    if narrow dt then fun key s ->
      let l = Nx.cast dt (onion (widen eta) key s) in
      let f = Nx.Dtype.float_format dt in
      let on_diag = Nx.greater (Nx.place at (Nx.eye dt n)) (const l 0.) in
      Nx.where
        (Nx.logical_and on_diag (Nx.less_equal l (const l 0.)))
        (const l (f.min_normal *. f.epsilon))
        l
    else onion eta
  in
  continuous ~family
    ~params:[ param "concentration" eta ]
    ~shape
    ~checks:[ in_positive family "concentration" eta ]
    ~support:(fun () -> Support.Correlation_cholesky n)
    ~factors ~draw ~coords:Bij.cholesky_corr eta

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
    tails = None;
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
    tails = None;
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
    tails = None;
    membership = None;
  }

(* [log_diff_exp a b] is [log (exp a - exp b)] for [a >= b], [-inf] where
   they are equal. *)
let log_diff_exp a b =
  let gap = Nx.sub a b in
  let same = Nx.logical_not (Nx.greater gap (const gap 0.)) in
  let gap = inside (Nx.logical_not same) gap 1. in
  Nx.where same (neg_inf a) (Nx.add a (log1mexp gap))

(* [log_add_exp a b] is [log (exp a + exp b)], from the larger. *)
let log_add_exp a b =
  let m = Nx.maximum a b in
  let m = Nx.where (Nx.isfinite m) m (const m 0.) in
  Nx.add m (Nx.log (Nx.add (Nx.exp (Nx.sub a m)) (Nx.exp (Nx.sub b m))))

(* Gauss-Legendre's eight nodes on [[-1, 1]] and their weights
   (Abramowitz and Stegun, Handbook of Mathematical Functions, Table
   25.4), exact for polynomials of degree 15. *)
let legendre_nodes =
  [|
    -0.9602898564975363; -0.7966664774136267; -0.5255324099163290;
    -0.1834346424956498; 0.1834346424956498; 0.5255324099163290;
    0.7966664774136267; 0.9602898564975363;
  |]

let legendre_weights =
  [|
    0.1012285362903763; 0.2223810344533745; 0.3137066458778873;
    0.3626837833783620; 0.3626837833783620; 0.3137066458778873;
    0.2223810344533745; 0.1012285362903763;
  |]

(* [along like v] is [v] as a tensor beside [like], on a new leading axis. *)
let along like v =
  let s = Array.append [| Array.length v |] (Array.map (fun _ -> 1) (Nx.shape like)) in
  Nx.reshape s (Nx.place (Nx.beside like) (Nx.create (Nx.dtype like) [| Array.length v |] v))

(* [legendre_log_weights lf] is the logs of half the weights, on [lf]'s
   leading axis: the mean over the window is [Σ_i w_i f(x_i) / 2]. *)
let legendre_log_weights lf =
  along (Nx.squeeze ~axes:[ 0 ] (Nx.slice [ Nx.R (0, 1) ] lf))
    (Array.map (fun w -> Float.log (w /. 2.)) legendre_weights)

(* [window factors l h] is the log density at the nodes of [[l, h]], on a
   new leading axis, and the window's width. *)
let window factors l h =
  let mid = Nx.mul_s (Nx.add l h) 0.5 and half = Nx.mul_s (Nx.sub h l) 0.5 in
  let x = Nx.add mid (Nx.mul half (along l legendre_nodes)) in
  (factors x, Nx.sub h l)

(* Newton's steps from a bound to a point whose tail underflows: each at
   least halves the gap for a log-concave tail, and squares it near the
   root. *)
let newton_steps = 12

(* A truncation reads its base's distribution function at its bounds, as
   logarithms on both sides. Its mass between them is a difference of the
   upper tails where the lower bound lies above the median, of the lower
   tails elsewhere: the smaller tails, whose difference keeps the digits
   that one minus a probability near [1] loses. A point at [p] has CDF
   [(1 - p) F(low) + p F(high)] and survival [(1 - p) S(low) + p S(high)],
   both formed in logs, and is the base's inverse at the smaller. *)
let truncated ?low ?high d =
  let family = "truncated" in
  let t =
    match d.tails with
    | Some t -> t
    | None ->
        invalid_argf
          "Norn.Dist.truncated: %s has no distribution function and quantile"
          d.descr
  in
  let like =
    match (low, high) with
    | Some l, _ | None, Some l -> l
    | None, None -> invalid_argf "Norn.Dist.truncated: %s has no bound" d.descr
  in
  let base_lo, base_hi = d.bounds () in
  let lo = Option.map (fun l -> Nx.maximum l base_lo) low in
  let hi = Option.map (fun h -> Nx.minimum h base_hi) high in
  let at bound f default =
    match bound with Some b -> f b | None -> const like default
  in
  let lc_lo = at lo t.log_cdf Float.neg_infinity and ls_lo = at lo t.log_sf 0. in
  let lc_hi = at hi t.log_cdf 0. and ls_hi = at hi t.log_sf Float.neg_infinity in
  let above_median = Nx.greater lc_lo (const lc_lo (-.Float.log 2.)) in
  let hi_tail = Nx.where above_median ls_lo lc_hi in
  let lo_tail = Nx.where above_median ls_hi lc_lo in
  let log_mass = log_diff_exp hi_tail lo_tail in
  (* Where the tails at the bounds agree within a factor [e^(1/4)], their
     difference loses the digits they share: there the mass is the density
     integrated over the window, on Gauss-Legendre's nodes, exact for a
     log density that varies by less than [1] across it. *)
  let log_mass =
    match (lo, hi) with
    | Some l, Some h ->
        let shape = broadcast family [ d.shape; Nx.shape l; Nx.shape h ] in
        let l = Nx.broadcast_to shape l and h = Nx.broadcast_to shape h in
        let lf, width = window d.factors l h in
        let spread = Nx.sub (Nx.max ~axes:[ 0 ] lf) (Nx.min ~axes:[ 0 ] lf) in
        let narrow =
          Nx.logical_and
            (Nx.less (Nx.sub hi_tail lo_tail) (const lo_tail 0.25))
            (Nx.less spread (const spread 1.))
        in
        let integral =
          Nx.add (Nx.log width)
            (Nx.logsumexp ~axes:[ 0 ] (Nx.add lf (legendre_log_weights lf)))
        in
        Nx.where narrow integral (Nx.broadcast_to shape log_mass)
    | _ -> log_mass
  in
  let clamp x =
    let x = match lo with Some l -> Nx.maximum x l | None -> x in
    match hi with Some h -> Nx.minimum x h | None -> x
  in
  (* Where the tail at the point underflows the dtype, the base's inverse
     cannot read it: the point is found from the bound on that side by
     Newton's steps on the tail's logarithm, whose slope is [± f / F], the
     density over the tail. Each step is clamped to the bounds. *)
  let tiny = Float.log (Prec.tiny d.dtype) in
  let newton ~upper target start =
    let x = ref start in
    for _ = 1 to newton_steps do
      let tail = if upper then t.log_sf !x else t.log_cdf !x in
      let slope = Nx.exp (Nx.sub (d.factors !x) tail) in
      let step = Nx.div (Nx.sub tail target) slope in
      let next = clamp ((if upper then Nx.add else Nx.sub) !x step) in
      x := Nx.where (Nx.isfinite next) next !x
    done;
    !x
  in
  let quantile p =
    let l1p = Nx.log1p (Nx.neg p) and lp = Nx.log p in
    let lc = log_add_exp (Nx.add l1p lc_lo) (Nx.add lp lc_hi) in
    let ls = log_add_exp (Nx.add l1p ls_lo) (Nx.add lp ls_hi) in
    let low_side = Nx.less_equal lc ls in
    let half = -.Float.log 2. in
    let lc = inside low_side lc half in
    let ls = Nx.where low_side (const ls half) ls in
    let direct =
      clamp (Nx.where low_side (t.lower (Nx.exp lc)) (t.upper (Nx.exp ls)))
    in
    (* A target of [-inf], at [p = 0] or [1] on an open side, is the
       inverse's own edge. *)
    let under v =
      Nx.logical_and (Nx.less v (const v tiny)) (Nx.isfinite v)
    in
    let deep = Nx.logical_or (under lc) (under ls) in
    let from_bound bound ~upper target =
      match bound with
      | Some b -> newton ~upper target (Nx.broadcast_to (Nx.shape direct) b)
      | None -> direct
    in
    Nx.where deep
      (Nx.where low_side
         (from_bound hi ~upper:false lc)
         (from_bound lo ~upper:true ls))
      direct
  in
  let factors x =
    let ok = Nx.isfinite log_mass in
    let within ok bound test =
      match bound with Some b -> Nx.logical_and ok (test x b) | None -> ok
    in
    let ok = within (within ok lo Nx.greater_equal) hi Nx.less_equal in
    outside ok (Nx.sub (d.factors x) log_mass)
  in
  let edge bound base = match bound with Some b -> b | None -> base in
  let coords =
    match (lo <> None || t.below, hi <> None || t.above) with
    | true, true ->
        Bij.interval ~low:(edge lo base_lo) ~high:(edge hi base_hi)
    | true, false -> Bij.greater ~low:(edge lo base_lo)
    | false, true -> Bij.less ~high:(edge hi base_hi)
    | false, false -> assert false
  in
  let bound_params =
    List.filter_map Fun.id
      [ Option.map (param "low") low; Option.map (param "high") high ]
  in
  let bound_checks =
    List.filter_map Fun.id
      [
        Option.map (in_reals family "low") low;
        Option.map (in_reals family "high") high;
        (match (low, high) with
        | Some l, Some h ->
            Some
              (check family "high - low" "(0, inf)" positive (Nx.sub h l))
        | _ -> None);
      ]
  in
  let mass =
    check family "log mass between the bounds" "(-inf, 0]"
      (fun m -> Nx.logical_and (Nx.isfinite m) (Nx.less_equal m (const m 0.)))
      log_mass
  in
  {
    d with
    family;
    descr = Printf.sprintf "%s %s" (describe family bound_params) d.descr;
    shape =
      broadcast family
        (d.shape :: List.map (fun (_, _, s) -> s) bound_params);
    checks = d.checks @ bound_checks @ [ mass ];
    support = (fun () -> Bij.image coords Support.Real [||]);
    bounds = coord_bounds d.at d.dtype coords;
    factors;
    draw = (fun k s -> quantile (open_unit k d.dtype s));
    coords = Some coords;
    standardize = None;
    quantile = Some quantile;
    tails = None;
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
      | Support.Less a -> Some (Bij.less ~high:(c a))
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
    tails = None;
    membership = Some membership;
  }

(* [log_matmul a b] is the product of matrices of logarithms over the last
   two axes, [log Σ_j exp (a_ij + b_jk)]: associative, as the product it
   computes in logs. *)
let log_matmul a b =
  let n = Nx.ndim a in
  Nx.logsumexp ~axes:[ n - 1 ]
    (Nx.add (Nx.unsqueeze ~axes:[ n ] a) (Nx.unsqueeze ~axes:[ n - 2 ] b))

(* [log_vecmat v m] is [log Σ_i exp (v_i + m_ij)], [v]'s last axis against
   [m]'s second-to-last. *)
let log_vecmat v m =
  Nx.logsumexp ~axes:[ Nx.ndim m - 2 ]
    (Nx.add (Nx.unsqueeze ~axes:[ Nx.ndim v ] v) m)

(* A hidden Markov model's forward and backward variables in logs
   (Rabiner, A Tutorial on Hidden Markov Models, 1989, eqs. 19-20 and
   24-25), from the matrices [G_t = log A + e_t], [e_t] step [t]'s emission
   log densities: [α_t = α_0 G_1 ⋯ G_t] and [β_t = G_(t+1) ⋯ G_(n-1) 1].
   Each run of products is one {!Nx.associative_scan} along the steps, in
   [2 log₂ n] rounds, of products of [k × k] matrices. *)
let hmm_passes ~log_init ~log_trans e =
  let t = Nx.ndim e - 2 in
  let n = (Nx.shape e).(t) in
  let at = List.init t (fun _ -> Nx.A) in
  let first = Nx.add log_init (Nx.slice (at @ [ Nx.I 0 ]) e) in
  let lead = Nx.unsqueeze ~axes:[ t ] first in
  if n = 1 then (lead, Nx.zeros_like lead)
  else
    let later = Nx.slice (at @ [ Nx.R (1, n) ]) e in
    let g = Nx.add log_trans (Nx.unsqueeze ~axes:[ t + 1 ] later) in
    let prefix = Nx.associative_scan ~axis:t Nx.Ptree.tensor log_matmul g in
    let alpha = Nx.concatenate ~axis:t [ lead; log_vecmat lead prefix ] in
    let suffix =
      Nx.flip ~axes:[ t ]
        (Nx.associative_scan ~axis:t Nx.Ptree.tensor
           (fun later earlier -> log_matmul earlier later)
           (Nx.flip ~axes:[ t ] g))
    in
    let beta =
      Nx.concatenate ~axis:t
        [ Nx.logsumexp ~axes:[ t + 2 ] suffix; Nx.zeros_like lead ]
    in
    (alpha, beta)

(* [log_product g] is [G_1 ⋯ G_m] for matrices along [g]'s axis [t], in
   logs: neighbours multiplied pairwise, in [⌈log₂ m⌉] rounds. *)
let rec log_product t g =
  let m = (Nx.shape g).(t) in
  if m = 1 then Nx.squeeze ~axes:[ t ] g
  else
    let lead = List.init t (fun _ -> Nx.A) in
    let pairs = Nx.slice (lead @ [ Nx.R (0, m - (m mod 2)) ]) g in
    let s = Nx.shape pairs in
    let pairs =
      Nx.reshape
        (Array.concat
           [ Array.sub s 0 t; [| m / 2; 2 |]; Array.sub s (t + 1) 2 ])
        pairs
    in
    let half i = Nx.slice (lead @ [ Nx.A; Nx.I i ]) pairs in
    let next = log_matmul (half 0) (half 1) in
    let next =
      if m mod 2 = 0 then next
      else
        Nx.concatenate ~axis:t
          [ next; Nx.slice (lead @ [ Nx.R (m - 1, m) ]) g ]
    in
    log_product t next

let hmm ~initial ~transition n d =
  let family = "hmm" in
  let is = Nx.shape initial and ts = Nx.shape transition in
  let k = if Array.length is = 1 then is.(0) else -1 in
  if k < 1 || ts <> [| k; k |] || Array.length d.shape = 0 || d.shape.(0) <> k
  then
    invalid_argf
      "Norn.Dist.hmm: initial of shape [%s] and transition of shape [%s] do \
       not match %s of shape [%s]"
      (shape_string is) (shape_string ts) d.descr (shape_string d.shape);
  if n < 1 then invalid_argf "Norn.Dist.hmm: n = %d is not positive" n;
  let rest = Array.sub d.shape 1 (Array.length d.shape - 1) in
  let log_init = Nx.log_softmax ~axes:[ 0 ] initial in
  let log_trans = Nx.log_softmax ~axes:[ 1 ] transition in
  let hull = hull_coords family d in
  (* Each step's log density under each state, of shape [b @ [n; k]] for
     [x]'s leading axes [b]. *)
  let emissions x =
    let r = ndim d.kind x - Array.length rest in
    let steps = if r >= 1 then (shape_of d.kind x).(r - 1) else 0 in
    if steps <> n then
      invalid_argf "Norn.Dist.hmm: a value of %d steps for an hmm of %d" steps
        n;
    let f =
      d.factors (apply d.kind { f = (fun x -> Nx.unsqueeze ~axes:[ r ] x) } x)
    in
    sum_trailing (Nx.ndim f - r - 1) f
  in
  (* [logsumexp_j α_n(j)], [α_n = α_1 G_2 ⋯ G_n]: the product alone, without
     the intermediate forward variables. *)
  let factors x =
    let e = emissions x in
    let t = Nx.ndim e - 2 in
    let lead = List.init t (fun _ -> Nx.A) in
    let first = Nx.add log_init (Nx.slice (lead @ [ Nx.I 0 ]) e) in
    let last =
      if n = 1 then first
      else
        let later = Nx.slice (lead @ [ Nx.R (1, n) ]) e in
        let g = Nx.add log_trans (Nx.unsqueeze ~axes:[ t + 1 ] later) in
        log_vecmat first (log_product t g)
    in
    Nx.logsumexp ~axes:[ t ] last
  in
  (* Rabiner's eq. 27: [γ_t ∝ α_t β_t]. *)
  let membership x =
    let alpha, beta = hmm_passes ~log_init ~log_trans (emissions x) in
    Nx.exp (Nx.log_softmax ~axes:[ Nx.ndim alpha - 1 ] (Nx.add alpha beta))
  in
  (* The states step by step, each a categorical draw from its predecessor's
     row, then every emission at every state and step, of which each step
     keeps its state's. *)
  let draw key s =
    let nb = Array.length s - 1 - Array.length rest in
    let batch = Array.sub s 0 nb in
    let keys = Nx.Rng.split ~n:(n + 1) key in
    let all = d.draw keys.(n) (Array.concat [ batch; [| n; k |]; rest ]) in
    let first =
      Nx.Rng.categorical ~key:keys.(0)
        (broadcast_to (Array.append batch [| k |]) initial)
    in
    let states = Array.make n first in
    for i = 1 to n - 1 do
      states.(i) <-
        Nx.Rng.categorical ~key:keys.(i)
          (Nx.take ~axis:0 states.(i - 1) transition)
    done;
    let z = Nx.stack ~axis:nb (Array.to_list states) in
    apply d.kind (select (nb + 1) z) all
  in
  let logits_check name l =
    check family name "[-inf, inf)"
      (fun l -> Nx.logical_and (not_nan l) (below_infinity l))
      l
  in
  {
    d with
    family;
    descr =
      Printf.sprintf "hmm(initial: %s [%d], transition: %s [%d; %d], n: %d, %s)"
        (Nx.Dtype.name (Nx.dtype initial))
        k
        (Nx.Dtype.name (Nx.dtype transition))
        k k n d.descr;
    shape = Array.append [| n |] rest;
    checks =
      d.checks
      @ [ logits_check "initial" initial; logits_check "transition" transition ];
    factors;
    draw;
    coords = hull;
    bounds =
      (match hull with
      | Some h -> coord_bounds d.at d.dtype h
      | None -> component_hull d);
    standardize = None;
    quantile = None;
    tails = None;
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

let membership d x =
  match d.membership with
  | Some m ->
      checked "membership" d;
      m x
  | None ->
      invalid_argf "Norn.Dist.membership: %s is not a mixture or an hmm"
        d.family

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
