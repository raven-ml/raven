(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reductions and scans through every kernel library the host runs, with
   operands and results on its device: a Max or a Min the extreme of its terms;
   a sum of -0 terms +0, an empty product 1 and a complex product of one term
   that term; an integer sum or product its terms' wrapped in any order; the
   special values of Logsumexp, Moments and Arg, reduced and scanned; a float or
   complex sum or product, reduced, scanned or from a carry, within its bound of
   the exact one, a NaN or an infinity where its terms make one; a scan of a
   prefix the prefix of the scan; an exact scan from a carry the scan it
   continues; and nothing written where declined. *)

open Windtrap
open Nx_array_gen
open Reduce_cases
module A = Nx_array
module D = Nx_array.Dtype
module P = Nx_kernel.Prog
module Support = Nx_kernels_support

(* A law's covers, held until its call answers: a case its library
   declines as it states covers nothing but that ({!laws}). *)
let held : (string * bool) list option Domain.DLS.key =
  Domain.DLS.new_key (fun () -> None)

let cover label c =
  match Domain.DLS.get held with
  | None -> Windtrap.cover label c
  | Some l -> Domain.DLS.set held (Some ((label, c) :: l))

(* [f x], its covers counted only if no call it makes is declined as its
   library states. *)
let covering f x =
  Domain.DLS.set held (Some []);
  let taken () =
    let l = Option.value (Domain.DLS.get held) ~default:[] in
    Domain.DLS.set held None;
    List.rev l
  in
  match f x with
  | () -> List.iter (fun (label, c) -> Windtrap.cover label c) (taken ())
  | exception Declined_as_stated ->
      ignore (taken ());
      Windtrap.cover "declined, as the library states" true
  | exception e ->
      ignore (taken ());
      raise e

(* Views *)

(* Scans draw long axes as often as the others, so that prefixes pass a
   chunk. *)
let scans =
  Gen.with_pp pp_case
    (case_of ~scan:(Gen.constant true)
       ~shapes:(Gen.frequency [ (1, shape_gen); (1, long) ])
       base)

let complexes = D.[ Any Complex64; Any Complex128 ]

(* Float and complex sums and products, reduced and scanned. *)
let bounded =
  Gen.with_pp pp_case (case_of ~monoids:[ Sum; Prod ] (floats @ complexes))

let scan_examples = [ example ~scan:true Sum f32 [| 9000 |] [| 0 |] ]

(* Bounded sums and products of each kind, reduced and scanned, also at
   float32 and complex64, which a library that declines float64 computes,
   a product past float32's range and a sum of NaNs and infinities. *)
let bound_examples =
  let c64 = D.Any D.Complex64 and c128 = D.Any D.Complex128 in
  [
    example Sum f32 [| 3; 7 |] [| 1 |];
    example Prod f64 [| 3; 4 |] [| 1 |];
    example Prod f32 [| 3; 4 |] [| 1 |];
    example Sum c128 [| 3; 7 |] [| 1 |];
    example Sum c64 [| 3; 7 |] [| 1 |];
    example ~scan:true Prod c64 [| 40 |] [| 0 |];
    example ~scan:true Sum f64 [| 9000 |] [| 0 |];
    example Prod f32 [| 5000 |] [| 0 |];
    example ~specials:20 Sum f32 [| 5000 |] [| 0 |];
  ]

(* Monoids *)

(* Terms of an extreme: a float's, and each part of a complex number's, one of
   few values half the time, so that terms tie and zeros of both signs meet,
   else of ten significant bits in (-4, 4), a NaN at the rate [specials]; other
   dtypes' as {!Reduce_cases.elements} draws them. *)
let extreme_values (D.Any dt) n ~specials seed =
  let r = Random.State.make [| seed |] in
  let few = [| -0.; 0.; 1.; -1.; infinity; neg_infinity |] in
  let part () =
    if specials > 0 && Random.State.int r specials = 0 then nan
    else if Random.State.bool r then few.(Random.State.int r (Array.length few))
    else full r 10
  in
  match D.kind dt with
  | D.Float -> A.Any (A.of_array dt [| n |] (Array.init n (fun _ -> part ())))
  | D.Complex ->
      let z _ = { Complex.re = part (); im = part () } in
      A.Any (A.of_array dt [| n |] (Array.init n z))
  | _ -> A.Any (elements dt n ~specials seed)

let extremes =
  Gen.with_pp pp_case
    (case_of ~monoids:[ Max; Min ] ~values:extreme_values D.all)

(* Max of two elements of [d] by Prog.Maximum, or Min by Prog.Minimum: the first
   NaN, a complex number with a NaN part one, else the greater, or the lesser,
   in Prog's order, -0 below +0, complex numbers by real part then imaginary
   part, [false] below [true]. *)
let extreme d m a b =
  if Elements.nan_of d a then a
  else if Elements.nan_of d b then b
  else
    let c = Elements.compare_elements d a b in
    if (m = Max && c >= 0) || (m = Min && c <= 0) then a else b

(* Whether [e] of the float dtype [d] is -0 ([Some true]) or +0 ([Some
   false]). *)
let zero_sign d e =
  let x =
    if Elements.is_narrow d then Elements.decode d e
    else if String.length e = 4 then Elements.get32 e 0
    else Elements.get64 e 0
  in
  if x = 0. then Some (Float.sign_bit x) else None

(* Whether the complex element [e] has one NaN part and one number. *)
let one_nan_part e =
  let h = String.length e / 2 in
  let nan at =
    Float.is_nan (if h = 4 then Elements.get32 e at else Elements.get64 e at)
  in
  nan 0 <> nan h

(* Elements of [d] of the same bits, or both NaN, a complex number with a NaN
   part one: a NaN result promises a NaN, never which. *)
let same_or_nan d =
  Testable.make ~pp:(Testable.pp Testable.string) ~equal:(fun a b ->
      String.equal a b || (Elements.nan_of d a && Elements.nan_of d b))

(* A Max or a Min is Prog.Maximum or Prog.Minimum of its terms: their greatest
   or least in Prog's order, -0 below +0, a NaN where a term is, any or all on
   booleans; a scan's result that of the terms up to it. *)
let law_extreme (b : Support.backend) (c : case) =
  let (D.Any dt as d) = dtype_of c.x in
  let float = D.is D.Float dt and boolean = D.is D.Boolean dt in
  let ts = Elements.elements c.x in
  let zeros = ref false and mixed = ref false in
  (* The extremes of [sl]'s prefixes, noting where zeros of both signs are among
     the terms of a zero extreme, and booleans of both values among one output's
     terms. *)
  let running sl =
    let y = Array.copy sl and neg = ref false and pos = ref false in
    Array.iteri
      (fun i t ->
        if i > 0 then y.(i) <- extreme d c.monoid y.(i - 1) t;
        if float then begin
          (match zero_sign d t with
          | Some true -> neg := true
          | Some false -> pos := true
          | None -> ());
          if !neg && !pos && zero_sign d y.(i) <> None then zeros := true
        end;
        if boolean && not (String.equal t sl.(0)) then mixed := true)
      sl;
    y
  in
  let last a = a.(Array.length a - 1) in
  let want =
    results c ts ~fill:"" ~reduce:(fun o -> last (running o)) ~scan:running
  in
  cover "a scan" c.scan;
  cover "a zero extreme of zeros of both signs" !zeros;
  cover "a NaN term" (Array.exists (Elements.nan_of d) ts);
  cover "a complex term of one NaN part"
    (D.is D.Complex dt && Array.exists one_nan_part ts);
  cover "booleans of both values" !mixed;
  equal (array (same_or_nan d)) want (Elements.elements (run b c))

(* Each cover's case at float32, float16, complex64 and booleans, which every
   library computes: zeros of both signs, a NaN term, a complex term of a NaN
   imaginary part beside greater real parts, and booleans of both values,
   reduced and scanned. *)
let extreme_examples =
  let case ?(scan = false) monoid x =
    { scan; monoid; axes = [| 0 |]; x; views = [] }
  in
  let z re im = { Complex.re; im } in
  let vector dt xs = A.Any (A.of_array dt [| Array.length xs |] xs) in
  [
    case Max (vector D.Float16 [| -0.; -1.; 0.; -0. |]);
    case ~scan:true Min (vector D.Float32 [| 0.; 1.; -0.; 0. |]);
    case Min (vector D.Float32 [| 1.; nan; neg_infinity |]);
    case ~scan:true Max
      (vector D.Complex64 [| z 1. 0.; z 2. nan; z 3. 0.; z (-1.) 0. |]);
    case Min (vector D.Complex64 [| z 1. 1.; z (-1.) nan; z (-2.) 0. |]);
    case Max (vector D.Bool [| false; true; false |]);
    case ~scan:true Min (vector D.Bool [| true; false; true |]);
  ]

(* The elements of [n] of [v] at [dt]. *)
let repeated (type v s) (dt : (v, s) D.t) n (v : v) =
  Elements.elements (A.Any (A.of_array dt [| n |] (Array.make n v)))

(* [f ()], a library's decline of what it states it does not compute passing. *)
let unless_declined f =
  match f () with () -> () | exception Declined_as_stated -> ()

(* Sums of [-0; -0], reduced and scanned, at each float precision and
   complex64. *)
let neg_zero_sums =
  let sum scan x = { scan; monoid = Sum; axes = [| 0 |]; x; views = [] } in
  let z = { Complex.re = -0.; im = -0. } in
  List.concat_map
    (fun (name, x) ->
      [ (name ^ ", reduced", sum false x); (name ^ ", scanned", sum true x) ])
    [
      ("float16", vector_of D.Float16 [| -0.; -0. |]);
      ("bfloat16", vector_of D.Bfloat16 [| -0.; -0. |]);
      ("float32", vector_of D.Float32 [| -0.; -0. |]);
      ("float64", vector_of D.Float64 [| -0.; -0. |]);
      ("complex64", A.Any (A.of_array D.Complex64 [| 2 |] [| z; z |]));
    ]

(* nx_kernel.mli: Monoid Sum "adds the terms from [+0], as {!Prog.Add} does, so
   a sum of [-0] terms is [+0]", and a scan stores at each index "the reduction
   [r] … over the indices along [axis] up to that index's, inclusive": +0, each
   part of a complex sum. *)
let test_neg_zero_sum (b : Support.backend) (_, (c : case)) =
  let (D.Any dt as d) = dtype_of c.x in
  let n = if c.scan then 2 else 1 in
  unless_declined (fun () ->
      equal
        (array (Elements.same d))
        (repeated dt n (D.zero dt))
        (Elements.elements (run b c)))

(* nx_kernel.mli: "an empty product is [1]": a Prod along an axis of no
   element. *)
let test_empty_product (b : Support.backend) (D.Any dt as d) =
  let c =
    {
      scan = false;
      monoid = Prod;
      axes = [| 1 |];
      x = A.Any (A.of_array dt [| 3; 0 |] [||]);
      views = [];
    }
  in
  unless_declined (fun () ->
      equal
        (array (Elements.same d))
        (repeated dt 3 (D.one dt))
        (Elements.elements (run b c)))

(* nx_kernel.mli: Monoid Prod "multiplies them from the first, never by [1],
   which would change a complex term's signed zeros or make NaN parts from its
   infinite ones", and a product scan's first result is its first term: -0 + ∞i
   times 1 would be NaN + ∞i. *)
let test_one_term_product (b : Support.backend) (scan, D.Any dt) =
  match D.kind dt with
  | D.Complex ->
      let x =
        A.Any (A.of_array dt [| 1 |] [| { Complex.re = -0.; im = infinity } |])
      in
      let c = { scan; monoid = Prod; axes = [| 0 |]; x; views = [] } in
      unless_declined (fun () ->
          equal
            (array (Elements.same (D.Any dt)))
            (Elements.elements x)
            (Elements.elements (run b c)))
  | _ -> invalid_arg "test_one_term_product: not complex"

(* Bounds *)

(* A float's precision: its unit roundoff, its largest finite number, the least
   magnitude a partial product may take for the references below to hold, and
   half its least subnormal number, the most a rounding into it moves a result
   below its normal numbers. *)
type precision = { u : float; big : float; floor : float; eta : float }

let p32 =
  { u = 0x1p-24; big = 0x1.fffffep127; floor = 0x1p-124; eta = 0x1p-150 }

(* The floor keeps TwoProduct's errors normal. *)
let p64 = { u = 0x1p-53; big = Float.max_float; floor = 0x1p-968; eta = 0. }

let p16 = { u = 0x1p-11; big = 65504.; floor = 0x1p-12; eta = 0x1p-25 }

let pb16 = { u = 0x1p-8; big = 0x1.fep127; floor = 0x1p-124; eta = 0x1p-134 }

(* The precision of [dt]'s parts. *)
let precision (D.Any dt) =
  match dt with
  | D.Float32 | D.Complex64 -> p32
  | D.Float16 -> p16
  | D.Bfloat16 -> pb16
  | _ -> p64

(* Whether [b] flushes values of the precision [pr] ({!Support.Flush}). *)
let flushes (b : Support.backend) pr =
  b.underflow = Flush && (pr == p32 || pr == pb16)

(* γ(k) = k u / (1 - k u). *)
let gamma p k =
  let k = float_of_int (max 0 k) in
  k *. p.u /. (1. -. (k *. p.u))

(* How a result meets the contract: within its bound, as the special value
   its terms make, or barred, where overflow or underflow may take part. *)
type verdict = Bounded | Special | Barred | Misses of string

let worse a b =
  match (a, b) with
  | Misses _, _ -> a
  | _, Misses _ -> b
  | Barred, _ | _, Barred -> Barred
  | Bounded, _ | _, Bounded -> Bounded
  | Special, Special -> Special

let want_nan g =
  if Float.is_nan g then Special
  else Misses (Printf.sprintf "%h, expected a NaN" g)

let want_special w g =
  if g = w then Special else Misses (Printf.sprintf "%h, expected %h" g w)

(* [g] within [bound] of [exact], the sum of two parts, computed in the
   carrier, then rounded once into [into] where that is narrower. The
   distance subtracts each part, so that rounding [exact] adds no error. *)
let within ?into ~exact:(hi, lo) ~bound g =
  let exact = hi +. lo in
  let a = Float.abs exact in
  match into with
  | Some d when a +. bound > d.big -> Barred
  | _ ->
      let tol =
        match into with
        | None -> bound
        | Some d -> bound +. (d.u *. (a +. bound)) +. d.eta
      in
      let d = g -. hi -. lo in
      if Float.abs d <= tol then Bounded
      else Misses (Printf.sprintf "%h is %h from %h, past %h" g d exact tol)

(* A float sum's terms so far: their count, the exact sum of the finite ones
   in two parts by TwoSum, their magnitudes' sum, and whether a NaN, +∞ or
   -∞ was among them. *)
type fsum = {
  mutable n : int;
  mutable hi : float;
  mutable lo : float;
  mutable mag : float;
  mutable nan : bool;
  mutable pinf : bool;
  mutable ninf : bool;
  mutable tiny : bool;
}

(* Whether [x] is a nonzero magnitude below 2^-126, the least normal
   float32: a library that flushes reads or gives such a part as a zero,
   and keeps no bound where a part of the computation is one. *)
let tiny x = x <> 0. && Float.abs x < 0x1p-126

let fsum () =
  {
    n = 0; hi = 0.; lo = 0.; mag = 0.; nan = false; pinf = false;
    ninf = false; tiny = false;
  }

let fsum_add s x =
  s.n <- s.n + 1;
  if Float.is_nan x then s.nan <- true
  else if x = infinity then s.pinf <- true
  else if x = neg_infinity then s.ninf <- true
  else begin
    let t = s.hi +. x in
    let z = t -. s.hi in
    s.lo <- s.lo +. (s.hi -. (t -. z)) +. (x -. z);
    s.hi <- t;
    s.mag <- s.mag +. Float.abs x;
    if tiny x || tiny t then s.tiny <- true
  end

(* nx_kernel.mli's [β(n) = ⌈5 min(n, 256) / 4⌉ + ⌈n / 128⌉] for a sum in
   blocks. *)
let beta n = (((5 * min n 256) + 3) / 4) + ((n + 127) / 128)

(* The most the references below err by, at [n] terms of magnitudes [mag]: [2n
   (n + 2) u²] times [mag], [u] float64's. A sum's [hi] and [lo] are exact but
   for [lo]'s roundings: [lo] adds two errors a term, each at most [u mag], so
   its [k]th of [2n] roundings errs by at most [k u² mag], and all of them by [n
   (2n + 1) u² mag]. A product's [lo] gains [u] of the product's magnitude a
   factor and two roundings of at most [u |lo|], so its [k]th factor errs by [2
   (k + 1) u²] of it, and all of them by [n (n + 3) u²]. A complex product's
   double-double additions and multiplications err by at most [6 u²] of [|p|
   |z|] a part, so each factor after the first, exact from [1], by [6√2 u²] of
   the modulus, below [2n (n + 2) u²] in all. *)
let slack n mag = Float.of_int (2 * n * (n + 2)) *. 0x1p-106 *. mag

(* A sum of magnitudes below half the largest number keeps every partial sum of
   every association finite. A reduction's sum is within [γ(β(n))], a scan's
   within [γ(n - 1)]. Where [flush], a sum with a {!tiny} term or partial sum is
   barred. *)
let fsum_verdict ?into ?(flush = false) ~scan p s g =
  if s.nan || (s.pinf && s.ninf) then want_nan g
  else if s.mag > p.big /. 2. then Barred
  else if s.pinf then want_special infinity g
  else if s.ninf then want_special neg_infinity g
  else if flush && s.tiny then Barred
  else
    let depth = if scan then s.n - 1 else beta s.n in
    let bound = (gamma p depth *. s.mag) +. slack s.n s.mag in
    within ?into ~exact:(s.hi, s.lo) ~bound g

(* A float product's factors so far: their count, the exact product of the
   finite non-zero ones in two parts by TwoProduct, the base-2 logarithms
   of the products of those of magnitude above 1 and of those below, the
   product's sign, and whether a NaN, an infinity or a zero was among
   them. Every partial product of every association lies between the two
   products. *)
type fprod = {
  mutable n : int;
  mutable hi : float;
  mutable lo : float;
  mutable up : float;
  mutable down : float;
  mutable neg : bool;
  mutable nan : bool;
  mutable inf : bool;
  mutable zero : bool;
  mutable tiny : bool;
}

let fprod () =
  {
    n = 0; hi = 1.; lo = 0.; up = 0.; down = 0.; neg = false; nan = false;
    inf = false; zero = false; tiny = false;
  }

let fprod_mul p x =
  p.n <- p.n + 1;
  if Float.is_nan x then p.nan <- true
  else begin
    if Float.sign_bit x then p.neg <- not p.neg;
    if Float.is_finite x && x <> 0. then begin
      let l = Float.log2 (Float.abs x) in
      if l > 0. then p.up <- p.up +. l else p.down <- p.down +. l;
      let h = p.hi *. x in
      p.lo <- Float.fma p.hi x (-.h) +. (p.lo *. x);
      p.hi <- h;
      if tiny x || tiny h then p.tiny <- true
    end
    else if x = 0. then p.zero <- true
    else p.inf <- true
  end

let over pr up = up >= Float.log2 pr.big -. 2.

let under pr down = down <= Float.log2 pr.floor

(* A zero times an infinity is a NaN, so a product holds a zero only
   barring overflow and an infinity only barring underflow. *)
let fprod_verdict ?into ?(flush = false) pr p g =
  if p.nan || (p.inf && p.zero) then want_nan g
  else if p.inf then
    if under pr p.down then Barred
    else want_special (if p.neg then neg_infinity else infinity) g
  else if p.zero then
    if over pr p.up then Barred
    else if g = 0. then Special
    else Misses (Printf.sprintf "%h, expected a zero" g)
  else if over pr p.up || under pr p.down || (flush && p.tiny) then Barred
  else
    let a = Float.abs (p.hi +. p.lo) in
    let bound = (gamma pr (p.n - 1) *. a) +. slack p.n a in
    within ?into ~exact:(p.hi, p.lo) ~bound g

(* TwoSum, and double-double sums and products by a double. *)
let two_sum a b =
  let s = a +. b in
  let z = s -. a in
  (s, (a -. (s -. z)) +. (b -. z))

let dd_add (ah, al) (bh, bl) =
  let s, e = two_sum ah bh in
  let e = e +. al +. bl in
  let h = s +. e in
  (h, e -. (h -. s))

let dd_mul (ah, al) b =
  let p = ah *. b in
  let e = Float.fma ah b (-.p) +. (al *. b) in
  let h = p +. e in
  (h, e -. (h -. p))

(* A complex product's factors so far, as [fprod] keeps a float one's, its
   exact value in double-double parts, an infinite part making it
   unbounded. *)
type zprod = {
  mutable n : int;
  mutable re : float * float;
  mutable im : float * float;
  mutable up : float;
  mutable down : float;
  mutable nan : bool;
  mutable inf : bool;
  mutable zero : bool;
  mutable tiny : bool;
}

let zprod () =
  {
    n = 0; re = (1., 0.); im = (0., 0.); up = 0.; down = 0.; nan = false;
    inf = false; zero = false; tiny = false;
  }

let zprod_mul p (z : Complex.t) =
  p.n <- p.n + 1;
  if Float.is_nan z.re || Float.is_nan z.im then p.nan <- true
  else if not (Float.is_finite z.re && Float.is_finite z.im) then p.inf <- true
  else if z.re = 0. && z.im = 0. then p.zero <- true
  else begin
    let l = Float.log2 (Float.hypot z.re z.im) in
    if l > 0. then p.up <- p.up +. l else p.down <- p.down +. l;
    let neg (h, l) = (-.h, -.l) in
    (* The parts' four products, which Prog.Mul's fused parts compute. *)
    let a = fst p.re and b = fst p.im in
    if
      List.exists tiny
        [ z.re; z.im; a *. z.re; a *. z.im; b *. z.re; b *. z.im ]
    then p.tiny <- true;
    let re = dd_add (dd_mul p.re z.re) (neg (dd_mul p.im z.im)) in
    p.im <- dd_add (dd_mul p.re z.im) (dd_mul p.im z.re);
    p.re <- re;
    if tiny (fst re) || tiny (fst p.im) then p.tiny <- true
  end

(* Normwise within γ(2n - 2): each multiplication by Prog.Mul's fused parts
   errs by at most 2u of the product's modulus. Which parts an infinite
   factor makes NaN depends on the association, so such a product is
   barred. *)
let zprod_verdict ?(flush = false) pr p (g : Complex.t) =
  let gnan = Float.is_nan g.re || Float.is_nan g.im in
  let show () = Printf.sprintf "%h %h" g.re g.im in
  if p.nan then
    if gnan then Special else Misses (show () ^ ", expected a NaN part")
  else if p.inf then Barred
  else if p.zero then
    if over pr p.up then Barred
    else if g.re = 0. && g.im = 0. then Special
    else Misses (show () ^ ", expected a zero")
  else if over pr p.up || under pr p.down || (flush && p.tiny) then Barred
  else
    let er = fst p.re +. snd p.re and ei = fst p.im +. snd p.im in
    let a = Float.hypot er ei in
    let bound = (gamma pr (2 * (p.n - 1)) *. a) +. slack p.n a in
    let part g (h, l) = g -. h -. l in
    let d = Float.hypot (part g.re p.re) (part g.im p.im) in
    if d <= bound then Bounded
    else
      Misses
        (Printf.sprintf "%s is %h from %h %h, past %h" (show ()) d er ei bound)

(* A reference of a float or complex Sum or Prod in the carrier [pr],
   rounded into [into]: [push] takes the next term, [verdict] judges a
   result of the terms so far. A complex sum adds each part as a float
   sum. Where [flush], a result a {!tiny} part of whose computation takes
   is barred. *)
type reference = { push : Complex.t -> unit; verdict : Complex.t -> verdict }

let reference ?into ?flush ~scan ~complex pr m =
  match (m, complex) with
  | Sum, false ->
      let s = fsum () in
      {
        push = (fun z -> fsum_add s z.re);
        verdict = (fun g -> fsum_verdict ?into ?flush ~scan pr s g.re);
      }
  | Sum, true ->
      let r = fsum () and i = fsum () in
      {
        push = (fun z -> fsum_add r z.re; fsum_add i z.im);
        verdict =
          (fun g ->
            worse
              (fsum_verdict ?flush ~scan pr r g.re)
              (fsum_verdict ?flush ~scan pr i g.im));
      }
  | Prod, false ->
      let p = fprod () in
      {
        push = (fun z -> fprod_mul p z.re);
        verdict = (fun g -> fprod_verdict ?into ?flush pr p g.re);
      }
  | Prod, true ->
      let p = zprod () in
      { push = zprod_mul p; verdict = zprod_verdict ?flush pr p }
  | (Max | Min), _ -> invalid_arg "reference: an extreme"

(* A float or complex array's elements in C order of indices, a float's as
   real parts. *)
let zvalues a =
  let (A.Any h) = host a in
  let re xs = Array.map (fun re -> { Complex.re; im = 0. }) xs in
  let go (type v s) (c : (v, s) A.t) : Complex.t array =
    match A.dtype c with
    | D.Float32 -> re (A.to_array c)
    | D.Float64 -> re (A.to_array c)
    | D.Float16 -> re (A.to_array c)
    | D.Bfloat16 -> re (A.to_array c)
    | D.Float8_e4m3fn -> re (A.to_array c)
    | D.Float8_e5m2 -> re (A.to_array c)
    | D.Float4_e2m1fn -> re (A.to_array c)
    | D.Complex64 -> A.to_array c
    | D.Complex128 -> A.to_array c
    | _ -> invalid_arg "zvalues: not a float or complex array"
  in
  go (A.copy h)

let kinds =
  [ "a float sum"; "a float product"; "a complex sum"; "a complex product" ]

let kind_of ~complex m =
  match (complex, m) with
  | false, Sum -> "a float sum"
  | false, _ -> "a float product"
  | true, Sum -> "a complex sum"
  | true, _ -> "a complex product"

(* The laws of S *)

(* Each result's verdict, of [y], [c]'s results on [b]. *)
let verdicts (b : Support.backend) (c : case) y =
  let (A.Any x) = c.x in
  let complex = D.is D.Complex (A.dtype x) in
  let pr = precision (D.Any (A.dtype x)) in
  let ts = zvalues c.x and gs = zvalues y in
  let flush = flushes b pr in
  let fresh () = reference ~flush ~scan:c.scan ~complex pr c.monoid in
  if not c.scan then
    Array.of_list
      (List.mapi
         (fun i o ->
           let r = fresh () in
           Array.iter r.push o;
           r.verdict gs.(i))
         (outputs c ts))
  else
    results c (Array.map2 (fun t g -> (t, g)) ts gs) ~fill:Barred
      ~reduce:(fun _ -> assert false)
      ~scan:(fun sl ->
        let r = fresh () in
        Array.map
          (fun (t, g) ->
            r.push t;
            r.verdict g)
          sl)

(* The first verdicts of [vs] that miss, named by their result. *)
let misses vs =
  let bad = ref [] in
  Array.iteri
    (fun i v ->
      match v with
      | Misses s when List.length !bad < 8 ->
          bad := Printf.sprintf "result %d: %s" i s :: !bad
      | _ -> ())
    vs;
  List.rev !bad

(* A float sum of [n] terms is within γ(β(n)) Σ|x| of the exact sum, a
   scan's result within γ(n - 1) Σ|x|, a float product within a relative
   γ(n - 1), a complex one normwise within γ(2n - 2), a complex sum part by
   part, barring overflow and underflow; a NaN or an infinity is one where
   the terms make it. *)
let law_bound (b : Support.backend) (c : case) =
  let (A.Any x) = c.x in
  let complex = D.is D.Complex (A.dtype x) in
  let vs = verdicts b c (run b c) in
  let has v = Array.mem v vs in
  let kind = kind_of ~complex c.monoid in
  List.iter
    (fun k -> cover (k ^ " within its bound") (k = kind && has Bounded))
    kinds;
  cover "a scan within its bound" (c.scan && has Bounded);
  cover "a NaN or an infinity the terms make" (has Special);
  cover "a product barred by overflow or underflow"
    (c.monoid = Prod && has Barred);
  equal (list string) [] (misses vs)

(* Integer sums and products, reduced and scanned. *)
let integers =
  Gen.with_pp pp_case
    (case_of ~monoids:[ Sum; Prod ]
       D.
         [
           Any Int8;
           Any Uint8;
           Any Int16;
           Any Uint16;
           Any Int32;
           Any Uint32;
           Any Int64;
           Any Uint64;
         ])

(* An integer sum or product wraps, so it has the bits of its terms folded
   in any order, here in C order, a scan's result those of the terms up to
   it. *)
let law_wrap (b : Support.backend) (c : case) =
  let f, id = op (dtype_of c.x) c.monoid in
  let scan ts =
    let acc = ref id in
    Array.map
      (fun t ->
        acc := f !acc t;
        want_of !acc)
      ts
  in
  let want =
    results c (terms c.x) ~fill:Some_nan
      ~reduce:(fun o -> want_of (Array.fold_left f id o))
      ~scan
  in
  cover "a scan" c.scan;
  cover "a product" (c.monoid = Prod);
  equal (list string) [] (differ want (run b c))

(* The first [m] elements along axis [a] of [x]. *)
let prefix a m (A.Any x) =
  let s = L.shape (A.layout x) in
  let r =
    Iarray.mapi
      (fun i e ->
        if i = a then { M.start = 0; count = m; step = 1 }
        else { M.start = 0; count = e; step = 1 })
      s
  in
  A.Any (Option.get (A.move (M.Slice r) x))

(* The scan of a prefix of the axis is the prefix of the scan. *)
let law_prefix (b : Support.backend) (c : case) =
  let s = shape_of c.x and a = c.axes.(0) in
  let m = s.(a) / 2 in
  let y = run b c and y' = run b { c with x = prefix a m c.x } in
  cover "a prefix past 4096 terms" (m > 4096);
  equal (list string) [] (differ (got (prefix a m y)) y')

(* Carries *)

(* A dtype a carried scan draws: its finite elements, near one for a
   product, its NaNs, its other specials, and two elements whose sum, or
   product, is a NaN made from numbers. *)
type 'v drawn = {
  finite : monoid -> Random.State.t -> 'v;
  nans : 'v list;
  others : 'v list;
  made : monoid -> ('v * 'v) option;
}

type carried = Carried : ('v, 's) D.t * 'v drawn -> carried

let carried_kinds =
  let float_values =
    {
      finite =
        (fun m r ->
          match m with
          | Prod ->
              (if Random.State.bool r then 1. else -1.)
              *. (0.999 +. Random.State.float r 0.002)
          | _ ->
              Float.ldexp
                (Random.State.float r 2. -. 1.)
                (Random.State.int r 9 - 4));
      nans =
        [
          nan; -.nan; Int64.float_of_bits 0x7FF4_0000_0000_0000L;
          Int64.float_of_bits 0xFFF9_2340_0000_0000L;
        ];
      others = [ -0.; 0.; infinity; neg_infinity ];
      made =
        (function
        | Sum -> Some (infinity, neg_infinity)
        | Prod -> Some (0., infinity)
        | Max | Min -> None);
    }
  in
  let float dt = Carried (dt, float_values) in
  let complex dt =
    let c re im = { Complex.re; im } in
    Carried
      ( dt,
        {
          finite =
            (fun m r ->
              c (float_values.finite m r) (0.01 *. float_values.finite Sum r));
          nans = [ c nan 1.; c 1. (-.nan); c (-.nan) nan ];
          others = [ c (-0.) (-0.); c infinity 0.; c 0. neg_infinity ];
          made =
            (function
            | Sum -> Some (c infinity 0., c neg_infinity 1.)
            | Prod -> Some (c 0. 0., c infinity 0.)
            | Max | Min -> None);
        } )
  in
  let ints dt ~small ~edges =
    Carried
      ( dt,
        {
          finite = (fun _ r -> small (Random.State.int r 16 - 8));
          nans = [];
          others = edges;
          made = (fun _ -> None);
        } )
  in
  [
    float D.Float32; float D.Float64; float D.Float16; float D.Bfloat16;
    complex D.Complex64; complex D.Complex128;
    ints D.Int32 ~small:Int32.of_int ~edges:[ Int32.max_int; Int32.min_int ];
    ints D.Int8 ~small:Fun.id ~edges:[ 127; -128 ];
    ints D.Int4 ~small:Fun.id ~edges:[ 7; -8 ];
    Carried
      ( D.Bool,
        {
          finite = (fun _ r -> Random.State.bool r);
          nans = [];
          others = [ true; false ];
          made = (fun _ -> None);
        } );
  ]

(* A scan along axis 1 of [[outer; p + m; inner]] whose last [m] terms of
   each slice are scanned from the carry of its first [p], the scan's
   result before them in the carrier, [p] at a multiple of 4096 or within
   it. One term in [rare] is one of the kind's others, or of its NaNs among
   the last [m] where [nan_after]. The first [p] hold no NaN, but a NaN made
   from numbers where [made] and a NaN term where [nan_before], of a [Max]
   or [Min]; the last hold a NaN where [nan_after]. The terms are a
   program's copy of the operand where [through]. *)
type carry_case = {
  kind : carried;
  cm : monoid;
  through : bool;
  outer : int;
  inner : int;
  p : int;
  m : int;
  made : bool;
  nan_before : bool;
  nan_after : bool;
  rare : int;
  cseed : int;
}

let pp_carry_case ppf c =
  let (Carried (dt, _)) = c.kind in
  Format.fprintf ppf
    "scan %s of %s%a [%d; %d + %d; %d]%s%s%s, one in %d special, seed %d"
    (monoid_name c.cm)
    (if c.through then "a copy of " else "")
    D.pp dt c.outer c.p c.m c.inner
    (if c.made then ", a made NaN before" else "")
    (if c.nan_before then ", a NaN before" else "")
    (if c.nan_after then ", a NaN after" else "")
    c.rare c.cseed

(* Which scans a carry case draws: [Exact], those every library continues
   bit for bit from any term, an integer's, a boolean's, a Max's and a
   Min's; [Inexact], a float's or a complex number's Sum and Prod, from a
   carry after any term. *)
type carried_scans = Exact | Inexact

let carry_case scans =
  let exact = scans = Exact in
  let open Gen in
  with_pp pp_carry_case
    (* One inexact case in five carries a NaN made from numbers and holds a
       NaN term after it: few draws of independent choices meet both. *)
    (let* both =
       if exact then constant false
       else frequency [ (4, constant false); (1, constant true) ]
     in
     let* (Carried (dt, vs) as kind) =
       of_list
         (List.filter
            (fun (Carried (dt, vs)) ->
              ((not both) || vs.made Sum <> None)
              && (exact || D.is D.Float dt || D.is D.Complex dt))
            carried_kinds)
     in
     let floating = D.is D.Float dt || D.is D.Complex dt in
     let* cm =
       of_list
         (List.filter
            (fun m ->
              accepts m (D.Any dt)
              && ((not both) || vs.made m <> None)
              && ((not exact) || (not floating) || m = Max || m = Min)
              && (exact || m = Sum || m = Prod))
            [ Sum; Prod; Max; Min ])
     in
     let* outer = of_list [ 1; 2 ] in
     let* inner = of_list [ 1; 3 ] in
     (* Such a case needs two terms before the carry and one after it. *)
     let least k l = if both then List.filter (fun n -> n >= k) l else l in
     let* p, m =
       frequency
         [
           ( 1,
             let* k = of_list [ 1; 2 ] in
             let+ m = of_list (least 1 [ 0; 1; 5; 4095; 4096; 4097; 9000 ]) in
             (4096 * k, m) );
           ( 1,
             let* p = of_list (least 2 [ 1; 2; 17; 1024; 2049; 4095 ]) in
             let+ m = of_list (least 1 [ 0; 1; 4096 - p; (4096 - p) / 2 ]) in
             (p, m) );
         ]
     in
     (* A bound's case draws NaNs and specials more rarely, so that most of
        its results are numbers. *)
     let rarely = frequency [ (1, constant true); (3, constant false) ] in
     let* made =
       if both then constant true else if exact then bool else rarely
     in
     let* nan_before =
       if both || vs.nans = [] || (cm <> Max && cm <> Min) then constant false
       else bool
     in
     let* nan_after =
       if both then constant true else if exact then bool else rarely
     in
     let* rare =
       if exact then constant 64
       else frequency [ (1, constant 64); (2, constant 4096) ]
     in
     let* through = bool in
     let+ cseed = int in
     {
       kind; cm; through; outer; inner; p; m; made; nan_before; nan_after; rare;
       cseed;
     })

(* A case of 3 slices of 17 terms before the carry and [m] after it, at
   float32 or, where [complex], complex64: dtypes a library that declines
   float64 computes. *)
let carry_example ?(complex = false) ?(made = false) ?(nan_before = false)
    ?(nan_after = false) cm m =
  {
    kind = List.nth carried_kinds (if complex then 4 else 0);
    cm;
    through = false;
    outer = 1;
    inner = 3;
    p = 17;
    m;
    made;
    nan_before;
    nan_after;
    rare = 4096;
    cseed = 7;
  }

(* NaN terms before an extreme's carry and after it, which the draws reach
   about one case in eight where float64 is declined. *)
let exact_carries = [ carry_example ~nan_before:true ~nan_after:true Max 5 ]

(* No term after a carry, a NaN carry made from numbers with a NaN term after
   it, and a complex carry of numbers, which the draws reach one case in 10
   to 7 where float64 is declined. *)
let inexact_carries =
  [
    carry_example Sum 0;
    carry_example ~made:true ~nan_after:true Sum 5;
    carry_example ~complex:true Sum 5;
  ]

(* The case's terms in C order of [[outer; p + m; inner]]. *)
let carry_terms (type v) (vs : v drawn) c : v array =
  let r = Random.State.make [| c.cseed |] in
  let n = c.p + c.m in
  let pick l = List.nth l (Random.State.int r (List.length l)) in
  let term t =
    let special = Random.State.int r c.rare = 0 in
    if special && t >= c.p && c.nan_after && vs.nans <> [] then pick vs.nans
    else if special then pick vs.others
    else vs.finite c.cm r
  in
  let a =
    Array.init (c.outer * n * c.inner) (fun i -> term (i / c.inner mod n))
  in
  let set t v =
    for o = 0 to c.outer - 1 do
      for i = 0 to c.inner - 1 do
        a.((((o * n) + t) * c.inner) + i) <- v
      done
    done
  in
  (match vs.made c.cm with
  | Some (u, v) when c.made && c.p > 1 ->
      set (Random.State.int r (c.p - 1)) u;
      set (c.p - 1) v
  | _ -> ());
  if c.nan_before then set (Random.State.int r c.p) (pick vs.nans);
  (if c.nan_after && c.m > 0 && vs.nans <> [] then
     let t = c.p + Random.State.int r c.m in
     set t (pick vs.nans));
  a

(* Elements [lo, lo + count) along axis 1 of [x]. *)
let along1 (A.Any x) lo count =
  let s = L.shape (A.layout x) in
  let r =
    Iarray.mapi
      (fun a e ->
        if a = 1 then { M.start = lo; count; step = 1 }
        else { M.start = 0; count = e; step = 1 })
      s
  in
  A.Any (Option.get (A.move (M.Slice r) x))

(* The program a carry case scans: [In 0], or a copy of it where
   [through]. *)
let carry_prog c (D.Any dt) =
  if c.through then
    P.v ~ins:[| D.Any dt |]
      [| P.In 0; P.Op1 (Copy, D.Any dt, 0) |]
      ~outs:[| 1 |]
  else identity (D.Any dt)

(* [c]'s scan by [b] of [x] along axis 1, of [dt]'s terms, into a fresh
   destination of [into], from [carry]. *)
let carry_scan (type v s) (b : Support.backend) c dt ?carry (into : (v, s) D.t)
    (A.Any x) =
  let (module K : Nx_kernel.S) = b.kernels in
  let y = A.Any (A.create b.device into (L.shape (A.layout x))) in
  let p = carry_prog c dt in
  answered b p [| (Monoid c.cm, 0) |] "scan"
    (K.scan ?carry p ~axis:1 (Monoid c.cm, 0) ~dsts:[| y |] [| Plain x |]);
  y

(* The case's terms on [b]'s device, and the carry into its last [m]
   terms: the scan's result before them, in the carrier. *)
let carry_of (b : Support.backend) c =
  let (Carried (dt, vs)) = c.kind in
  let n = c.p + c.m in
  let z =
    on b (A.Any (A.of_array dt [| c.outer; n; c.inner |] (carry_terms vs c)))
  in
  let (D.Any ct) = Nx_kernel.carrier (D.Any dt) in
  (z, along1 (carry_scan b c (D.Any dt) ct (along1 z 0 c.p)) (c.p - 1) 1)

let covers_carry c =
  cover "a carry after 4096 terms or more" (c.p >= 4096);
  cover "no term after a carry" (c.m = 0);
  cover "several slices" (c.outer * c.inner > 1);
  let (Carried (dt, _)) = c.kind in
  cover "a NaN term after a carry"
    ((D.is D.Float dt || D.is D.Complex dt) && c.nan_after && c.m > 0);
  cover "a narrow float's carry" (D.bits dt = 16 && D.is D.Float dt);
  cover "a program's terms" c.through

(* The case's terms and their carry, nx.cpu's, and the two on [b]'s device. *)
let carry_on (b : Support.backend) c =
  let z, carry = Support.cpu.around (fun () -> carry_of Support.cpu c) in
  (z, on b z, on b carry)

(* An exact scan from the carry of a slice's first [p] terms has the bits of the
   whole slice's scan from term [p] on, a NaN as any NaN. Those bits are the
   contract's, so nx.cpu's whole scan and carry serve every library. *)
let law_carry (b : Support.backend) c =
  let (Carried (dt, _)) = c.kind in
  covers_carry c;
  cover "a NaN term before an extreme's carry" (c.nan_before && c.m > 0);
  let z, on_b, carry = carry_on b c in
  let got = carry_scan b c (D.Any dt) ~carry dt (along1 on_b c.p c.m) in
  let want =
    Support.cpu.around (fun () ->
        along1 (carry_scan Support.cpu c (D.Any dt) dt z) c.p c.m)
  in
  equal
    (array (Elements.same (D.Any dt)))
    (Elements.elements (host want))
    (Elements.elements (host got))

(* A float or complex Sum or Prod scan from a carry, nx.cpu's: each result
   within the bound of a reduction of the carry's element and the terms up to
   it, a NaN where a NaN carry or term reaches it. *)
let law_carry_bound (b : Support.backend) c =
  let (Carried (dt, _)) = c.kind in
  let complex = D.is D.Complex dt in
  covers_carry c;
  cover "a NaN carry made from numbers" (c.made && c.p > 1 && c.m > 0);
  cover "a NaN carry and a NaN term after it"
    (c.made && c.p > 1 && c.nan_after && c.m > 0);
  cover "a complex carry" complex;
  cover "a product's carry" (c.cm = Prod);
  let _, z, carry = carry_on b c in
  let (D.Any _ as cd) = Nx_kernel.carrier (D.Any dt) in
  let terms = along1 z c.p c.m in
  let got = carry_scan b c (D.Any dt) ~carry dt terms in
  let pr = precision cd in
  let into = if same cd (D.Any dt) then None else Some (precision (D.Any dt)) in
  let cs = zvalues carry and ts = zvalues terms and gs = zvalues got in
  let bad = ref [] and seen = ref [] in
  for o = 0 to c.outer - 1 do
    for i = 0 to c.inner - 1 do
      let r =
        reference ?into ~flush:(flushes b pr) ~scan:true ~complex pr c.cm
      in
      r.push cs.((o * c.inner) + i);
      for j = 0 to c.m - 1 do
        let k = (((o * c.m) + j) * c.inner) + i in
        r.push ts.(k);
        let v = r.verdict gs.(k) in
        seen := v :: !seen;
        match v with
        | Misses s when List.length !bad < 8 ->
            bad := Printf.sprintf "slice %d %d, result %d: %s" o i j s :: !bad
        | _ -> ()
      done
    done
  done;
  let nan (z : Complex.t) = Float.is_nan z.re || Float.is_nan z.im in
  cover "a NaN carry's results" (Array.exists nan cs && c.m > 0);
  cover "a result within its bound" (List.mem Bounded !seen);
  equal (list string) [] (List.rev !bad)

(* A carry of the output's carrier, float32 for float16 terms, is
   taken. *)
let test_carrier_carry (b : Support.backend) () =
  let (module K : Nx_kernel.S) = b.kernels in
  let make dt s = A.Any (A.create b.device dt s) in
  let (A.Any x) = make D.Float16 [| 2; 3 |] in
  equal Elements.answer A.Done
    (K.scan ~carry:(make D.Float32 [| 2; 1 |]) (identity (D.Any D.Float16))
       ~axis:1 (Monoid Sum, 0) ~dsts:[| make D.Float16 [| 2; 3 |] |]
       [| Plain x |])

(* The dtypes of destinations as their dtypes and shapes give them, of
   loads and of reductions, as Nx_kernel's rules read them. *)
let specs = { Nx_kernel.arrays = Array.length; dtype = (fun a i -> fst a.(i)) }

let loads_view =
  {
    Nx_kernel.arrays = Array.length;
    dtype =
      (fun (a : Nx_kernel.load array) i ->
        match a.(i) with
        | Plain x -> D.Any (A.dtype x)
        | Padded { x; _ } -> D.Any (A.dtype x));
  }

let reductions_view =
  {
    Nx_kernel.reductions = Array.length;
    kind = (fun rs i -> fst rs.(i));
    output = (fun rs i -> snd rs.(i));
  }

(* Reductions and scans that do not fit their kernel, whatever the program
   computes: by other numbers of loads, destinations or reductions, an output
   past the program's, a coordinate past the loop's rank, an empty loop's
   included, a load of another dtype than its operand, a reduction of a dtype it
   does not take, an Arg's position other than int64, or a carry of a reduction
   that takes none, of another shape or of another dtype than the carrier. Each
   with its misfit's class, the refusal nx's rules name it by, the destinations'
   dtypes and shapes, and the call into them by a library, its operands, made on
   the host, on the library's device. *)
let loop_misfits =
  let f32 = D.Any D.Float32 and i64 = D.Any D.Int64 in
  let id = identity f32 in
  let coord = P.v ~ins:[||] [| P.Coord 2 |] ~outs:[| 0 |] in
  let x ?(dt = D.Any D.Float32) s =
    let (D.Any dt) = dt in
    Nx_kernel.Plain (A.create Rig.host dt s)
  in
  let placed (b : Support.backend) g : Nx_kernel.load -> Nx_kernel.load =
    function
    | Plain x -> Misfit.load g (Plain (Support.on b x))
    | Padded _ as l -> Misfit.load g l
  in
  let wrong = A.Wrong_dtype and shape = A.Shape_mismatch in
  (* A reduction along axis 1 of a loop of [shape], or a scan along it of
     its destinations' shape, from [carry]. *)
  let reduce ?(shape = [| 2; 3 |]) p rs loads dsts =
    let named =
      Nx_kernel.reduce_refusal p ~rank:(Array.length shape)
        ~reductions:reductions_view rs ~dsts:specs dsts ~loads:loads_view
        loads
    in
    let call (b : Support.backend) g ds =
      let (module K : Nx_kernel.S) = b.kernels in
      K.reduce p ~shape:(Iarray.of_array shape) ~axes:[| 1 |]
        (Iarray.of_array rs) ~dsts:(Iarray.of_array ds)
        (Iarray.map (placed b g) (Iarray.of_array loads))
    in
    (Option.map fst named, dsts, call)
  in
  let scan ?carry p (r, k) loads dsts =
    let y = if dsts = [||] then [| 2; 3 |] else snd dsts.(0) in
    let named =
      match
        Nx_kernel.scan_refusal p ~rank:(Array.length y) r ~output:k
          ~dsts:specs dsts ?carry:(Option.map fst carry) ~loads:loads_view
          loads
      with
      | Some (e, _) -> Some e
      | None -> (
          match carry with
          | Some (_, s) when Nx_kernel.Shape.carry ~axis:1 y <> Ok s ->
              Some A.Shape_mismatch
          | _ -> None)
    in
    let call (b : Support.backend) (g : Misfit.guard) ds =
      let (module K : Nx_kernel.S) = b.kernels in
      let carry =
        Option.map
          (fun (D.Any d, s) ->
            A.Any (g.g (A.create b.device d (Iarray.of_array s))))
          carry
      in
      K.scan ?carry p ~axis:1 (r, k) ~dsts:(Iarray.of_array ds)
        (Iarray.map (placed b g) (Iarray.of_array loads))
    in
    (named, dsts, call)
  in
  let row = (f32, [| 2 |]) and whole = (f32, [| 2; 3 |]) in
  let one = [| x [| 2; 3 |] |] and carry = (f32, [| 2; 1 |]) in
  [
    ( "a reduction of an output past the program's", "a count", shape,
      reduce id [| (Monoid Sum, 1) |] one [| row |] );
    ("no reduction", "a count", shape, reduce id [||] one [||]);
    ( "one destination for an Arg", "a count", shape,
      reduce id [| (Arg Max, 0) |] one [| row |] );
    ( "a load past the program's operands", "a count", shape,
      reduce id [| (Monoid Sum, 0) |] [| x [| 2; 3 |]; x [| 2; 3 |] |]
        [| row |] );
    ( "a coordinate past the loop's rank", "an attribute", shape,
      reduce coord [| (Monoid Sum, 0) |] [||] [| (i64, [| 2 |]) |] );
    ( "a coordinate past the rank of a loop of no element", "an attribute",
      shape,
      reduce ~shape:[| 2; 0 |] coord [| (Monoid Sum, 0) |] [||]
        [| (i64, [| 2 |]) |] );
    ( "a load of another dtype than its operand", "a dtype", wrong,
      reduce id [| (Monoid Sum, 0) |] [| x ~dt:(D.Any D.Float64) [| 2; 3 |] |]
        [| row |] );
    ( "a Logsumexp of int32", "a dtype", wrong,
      reduce (identity (D.Any D.Int32)) [| (Logsumexp, 0) |]
        [| x ~dt:(D.Any D.Int32) [| 2; 3 |] |]
        [| (D.Any D.Int32, [| 2 |]) |] );
    ( "an Arg's position of int32", "a dtype", wrong,
      reduce id [| (Arg Max, 0) |] one [| row; (D.Any D.Int32, [| 2 |]) |] );
    ( "a scan of an output past the program's", "a count", shape,
      scan id (Monoid Sum, 1) one [| whole |] );
    ( "two destinations for a scan's Sum", "a count", shape,
      scan id (Monoid Sum, 0) one [| whole; whole |] );
    ( "a scan's coordinate past the rank of a loop of no element",
      "an attribute", shape,
      scan coord (Monoid Sum, 0) [||] [| (i64, [| 2; 0 |]) |] );
    ( "a Logsumexp's carry", "an attribute", shape,
      scan ~carry id (Logsumexp, 0) one [| whole |] );
    ( "a Moments' carry", "an attribute", shape,
      scan ~carry id (Moments, 0) one [| whole; whole |] );
    ( "an Arg's carry", "an attribute", shape,
      scan ~carry id (Arg Max, 0) one [| whole; (i64, [| 2; 3 |]) |] );
    ( "a carry of the destination's shape", "a shape", shape,
      scan ~carry:(f32, [| 2; 3 |]) id (Monoid Sum, 0) one [| whole |] );
    ( "a carry of one element", "a shape", shape,
      scan ~carry:(f32, [| 1; 1 |]) id (Monoid Sum, 0) one [| whole |] );
    ( "a carry of another dtype", "a dtype", wrong,
      scan ~carry:(D.Any D.Float64, [| 2; 1 |]) id (Monoid Max, 0) one
        [| whole |] );
    ( "a float16 carry of float16 terms", "a dtype", wrong,
      scan
        ~carry:(D.Any D.Float16, [| 2; 1 |])
        (identity (D.Any D.Float16)) (Monoid Sum, 0)
        [| x ~dt:(D.Any D.Float16) [| 2; 3 |] |]
        [| (D.Any D.Float16, [| 2; 3 |]) |] );
  ]

(* nx's rules name the misfit [want], and the call is not Done and writes
   nothing outside its destinations. *)
let check_misfit (b : Support.backend) (msg, what, want, (named, dsts, call))
    =
  (match named with
  | Some got -> equal ~msg Elements.answer want got
  | None -> failf "%s: the rules find it fits" msg);
  Misfit.check ~msg ~what b dsts (call b)

(* A destination over its load's bytes, one element on: the door refuses a
   reduction or a scan into it as Overlapping. *)
let test_overlapping (b : Support.backend) () =
  let (module K : Nx_kernel.S) = b.kernels in
  let x = A.create b.device D.Float32 [| 9 |] in
  let from start count =
    Option.get (A.move (M.Slice [| { M.start; count; step = 1 } |]) x)
  in
  let load = Option.get (A.move (M.Reshape [| 2; 4 |]) (from 0 8)) in
  let p = identity f32 in
  equal ~msg:"reduced" Elements.answer A.Overlapping
    (K.reduce p ~shape:[| 2; 4 |] ~axes:[| 1 |]
       [| (Monoid Sum, 0) |]
       ~dsts:[| A.Any (from 1 2) |]
       [| Plain load |]);
  equal ~msg:"scanned" Elements.answer A.Overlapping
    (K.scan p ~axis:0 (Monoid Sum, 0)
       ~dsts:[| A.Any (from 1 8) |]
       [| Plain (from 0 8) |])

(* A Max's NaN carry that is its destination, a slice of one term, a number,
   makes the result a NaN, though the scan writes the carry's bytes: nx_cuda.mli
   and nx_amd.mli state that a scan's destination may be its carry. *)
let test_aliased_carry (b : Support.backend) () =
  let (module K : Nx_kernel.S) = b.kernels in
  let f64 = Int64.float_of_bits in
  let y = A.expect D.Float64 (on b (A.Any (A.of_array D.Float64 [| 1; 1 |] [| f64 0x7FF4_0000_0000_0001L |]))) in
  let x = on b (A.Any (A.of_array D.Float64 [| 1; 1 |] [| 1. |])) in
  let p = identity (D.Any D.Float64) in
  match
    answered b p [| (Monoid Max, 0) |] "the scan"
      (K.scan ~carry:(A.Any y) p ~axis:1 (Monoid Max, 0) ~dsts:[| A.Any y |]
         [| plain_of x |])
  with
  | exception Declined_as_stated -> ()
  | () ->
      equal
        (array (Elements.same (D.Any D.Float64)))
        [| Elements.f64 nan |]
        (Elements.elements (host (A.Any y)))

(* Special values *)

(* Terms of a palette: [Finite] ones of both signs, tied now and then, and
   zeros of both signs; [Specials], half of them -0, an infinity or a NaN;
   [Rare], one in fifty so; [Neg_inf], -∞ alone; [Pos_inf], finite ones
   and +∞; [Infinities], finite ones and both infinities. *)
type palette = Finite | Specials | Rare | Neg_inf | Pos_inf | Infinities

let palette_name = function
  | Finite -> "finite"
  | Specials -> "specials"
  | Rare -> "rare specials"
  | Neg_inf -> "-∞ alone"
  | Pos_inf -> "+∞"
  | Infinities -> "both infinities"

(* Specials, which reach most of the cases, come three times as often. *)
let palettes =
  [ Finite; Specials; Specials; Specials; Rare; Neg_inf; Pos_inf; Infinities ]

(* [n] float32 or float64 elements of the palette [pal] from [seed], NaNs
   of drawn payloads. *)
let palette_elements (type v s) (dt : (v, s) D.t) n pal seed =
  let r = Random.State.make [| seed |] in
  let pick a = a.(Random.State.int r (Array.length a)) in
  let finite () =
    let x =
      if Random.State.bool r then full r 52
      else Float.of_int (Random.State.int r 9 - 4) /. 2.
    in
    if x = 0. && Random.State.bool r then -0. else x
  in
  let special () = pick [| -0.; infinity; neg_infinity; nan |] in
  let one () =
    match pal with
    | Finite -> finite ()
    | Specials -> if Random.State.bool r then finite () else special ()
    | Rare -> if Random.State.int r 50 = 0 then special () else finite ()
    | Neg_inf -> neg_infinity
    | Pos_inf -> if Random.State.int r 4 = 0 then infinity else finite ()
    | Infinities ->
        if Random.State.int r 4 = 0 then pick [| infinity; neg_infinity |]
        else finite ()
  in
  let b = Bytes.create (max 1 (D.bytes dt n)) in
  for i = 0 to n - 1 do
    let x = one () in
    match dt with
    | D.Float32 ->
        Bytes.set_int32_le b (4 * i)
          (if Float.is_nan x then pick nans32 else Int32.bits_of_float x)
    | D.Float64 ->
        Bytes.set_int64_le b (8 * i)
          (if Float.is_nan x then pick nans64 else Int64.bits_of_float x)
    | _ -> invalid_arg "palette_elements: not float32 or float64"
  done;
  let buf = Rig.Buffer.of_string (Bytes.to_string b) in
  A.Any (A.v dt (L.contiguous [| n |]) buf)

(* An Arg, Logsumexp or Moments reduction, or a scan along its one axis. *)
type special = { stat : stat; scanned : bool }

let pp_special ppf c =
  Format.fprintf ppf "%s %a"
    (if c.scanned then "scan" else "reduce")
    pp_stat c.stat

let special_case =
  Gen.with_pp pp_special
    (let open Gen in
     let ints = D.[ Any Int8; Any Uint64 ] in
     let* scanned = bool in
     let* r = of_list Nx_kernel.[ Arg Max; Arg Min; Logsumexp; Moments ] in
     let* d =
       of_list ~pp:pp_dtype
         (match r with Arg _ -> floats @ floats @ floats @ ints | _ -> floats)
     in
     let small = array ~size:(int_range 0 3) (int_range 0 5) in
     (* Rows of 128 terms or more, reduced or scanned along them: few
        outputs whose lanes nx.cuda spreads over its threads. *)
     let rows =
       let* r = int_range 1 40 in
       let+ k = of_list ~pp:Format.pp_print_int [ 130; 300; 1025 ] in
       [| r; k |]
     in
     let* s, along =
       frequency
         [ (6, map (fun s -> (s, false)) small);
           (1, map (fun s -> (s, false)) long);
           (2, map (fun s -> (s, true)) rows) ]
     in
     let s = if scanned && Array.length s = 0 then [| 3 |] else s in
     let rank = Array.length s in
     let* axes =
       if along then constant [| 1 |]
       else if scanned then map (fun a -> [| a |]) (int_range 0 (rank - 1))
       else
         let+ keep = array ~size:(constant rank) bool in
         Array.of_list (List.filter (fun i -> keep.(i)) (List.init rank Fun.id))
     in
     let* v = view_of rank in
     let* pal = of_list palettes in
     let+ seed = int in
     let (D.Any dt) = d in
     let make n =
       if is_float d then palette_elements dt n pal seed
       else A.Any (elements dt n ~specials:4 seed)
     in
     let name = if is_float d then palette_name pal else "integers" in
     { scanned; stat = stat_of r s axes v name make })

(* Each class of result the contract states, on a vector of [xs], each at
   float32, which a library that declines float64 computes. *)
let special_examples =
  let inf = infinity and ninf = neg_infinity in
  let nan = Int64.float_of_bits 0x7FF4_0000_0000_0001L in
  let one ?(scanned = false) r dt xs =
    let x = vector_of dt xs in
    { scanned; stat = { r; axes = [| 0 |]; x; views = []; values = "example" } }
  in
  let empty r =
    let x = A.Any (A.create Rig.host D.Float32 [| 3; 0 |]) in
    let stat = { r; axes = [| 1 |]; x; views = []; values = "example" } in
    { scanned = false; stat }
  in
  Nx_kernel.
    [
      empty Logsumexp;
      one Logsumexp D.Float64 [| ninf; ninf |];
      one Logsumexp D.Float32 [| ninf; ninf |];
      one Logsumexp D.Float32 [| 1.; inf; ninf |];
      one Logsumexp D.Float32 [| 1.; nan |];
      one Logsumexp D.Float64 [| 1.; 2. |];
      one Logsumexp D.Float32 [| 1.; 2. |];
      one ~scanned:true Logsumexp D.Float64 [| ninf; 2.; nan; inf |];
      one ~scanned:true Logsumexp D.Float32 [| ninf; 2.; nan; inf |];
      one ~scanned:true Logsumexp D.Float32 [| ninf; inf; 1. |];
      empty Moments;
      one Moments D.Float64 [| 1.; nan; inf |];
      one Moments D.Float32 [| 1.; nan; inf |];
      one Moments D.Float32 [| 1.; inf; 2. |];
      one Moments D.Float32 [| 1.; 2.; 4. |];
      one Moments D.Float64 [| inf; ninf; 0. |];
      one Moments D.Float32 [| inf; ninf; 0. |];
      one ~scanned:true Moments D.Float32 [| -0.; 1.; ninf; inf; nan |];
      one ~scanned:true Moments D.Float32 [| inf; 1.; 2. |];
      one ~scanned:true Moments D.Float32 [| ninf; 1.; 2. |];
      one (Arg Max) D.Float32 [| -0.; 0.; nan; 1.; nan |];
      one (Arg Max) D.Float64 [| 1.; 3.; 3.; ninf |];
      one (Arg Max) D.Float32 [| 1.; 3.; 3.; ninf |];
      one (Arg Max) D.Float32 [| -0.; 0.; -1. |];
      one ~scanned:true (Arg Min) D.Float64 [| 2.; -0.; 0.; -0.; ninf; nan |];
      one ~scanned:true (Arg Min) D.Float32 [| 2.; -0.; 0.; -0.; ninf; nan |];
    ]

(* What a result must be: those bits, a NaN where [Some_nan], an infinity,
   any finite number, or one not negative. *)
type klass = Exactly of want | Infinite of float | Number | Not_negative

let klass_of = function Is w -> Exactly w | Near _ -> invalid_arg "klass_of"

(* A Logsumexp's or Moments' terms so far: their count and how many are
   NaN, +∞ and -∞. *)
type flags = {
  mutable count : int;
  mutable nans : int;
  mutable pinfs : int;
  mutable ninfs : int;
}

let flag f t =
  f.count <- f.count + 1;
  let x = value t in
  if Float.is_nan x then f.nans <- f.nans + 1
  else if x = infinity then f.pinfs <- f.pinfs + 1
  else if x = neg_infinity then f.ninfs <- f.ninfs + 1

(* The case of [r] of the terms [f] counts, and the classes of its
   destinations. *)
let classes (r : Nx_kernel.reduction) f =
  let nan = Exactly Some_nan in
  match r with
  | Logsumexp ->
      if f.count = 0 then ("no term", [| Infinite neg_infinity |])
      else if f.nans > 0 then ("a NaN term", [| nan |])
      else if f.pinfs > 0 then ("+∞ and no NaN", [| Infinite infinity |])
      else if f.ninfs = f.count then ("-∞ alone", [| Infinite neg_infinity |])
      else ("finite terms", [| Number |])
  | Moments ->
      if f.count = 0 then ("no term", [| nan; nan |])
      else if f.nans > 0 then ("a NaN term", [| nan; nan |])
      else if f.pinfs > 0 && f.ninfs > 0 then
        ("both infinities and no NaN", [| nan; nan |])
      else if f.pinfs > 0 then
        ("an infinity and no NaN", [| Infinite infinity; nan |])
      else if f.ninfs > 0 then
        ("an infinity and no NaN", [| Infinite neg_infinity; nan |])
      else ("finite terms", [| Number; Not_negative |])
  | Arg _ | Monoid _ -> invalid_arg "classes"

let arg_classes ks =
  let ks = Array.map klass_of ks in
  ((if ks.(0) = Exactly Some_nan then "a NaN term" else "numbers"), ks)

(* The case and classes of [r] of the terms [ts] at [d], or of each prefix
   where [scanned]. *)
let special_expect d (r : Nx_kernel.reduction) ~scanned ts =
  match r with
  | Arg e when scanned -> Array.map arg_classes (arg_scan d e ts)
  | Arg e -> [| arg_classes (arg_expect d e ts) |]
  | _ ->
      let f = { count = 0; nans = 0; pinfs = 0; ninfs = 0 } in
      if not scanned then begin
        Array.iter (flag f) ts;
        [| classes r f |]
      end
      else
        Array.map
          (fun t ->
            flag f t;
            classes r f)
          ts

(* An Arg reduction's further cases: an extreme whose bits another term
   repeats, and a zero extreme beside a zero of the other sign. *)
let arg_cases d (e : Nx_kernel.extreme) ts =
  match (arg_expect d e ts).(0) with
  | Is (Bits x) ->
      let zero t = match t with F (v, _) -> v = 0. | I _ -> false in
      let has p = Array.exists p ts in
      let tie k t = k + Bool.to_int (bits_of t = x) in
      let ties = Array.fold_left tie 0 ts in
      let signed n = has (fun t -> zero t && Float.sign_bit (value t) = n) in
      (if ties > 1 then [ "a tie" ] else [])
      @
      if has (fun t -> zero t && bits_of t = x) && signed true && signed false
      then [ "-0 beside +0" ]
      else []
  | _ -> []

let special_labels =
  let named name scanned l =
    Printf.sprintf "%s%s: %s" name (if scanned then " scan" else "") l
  in
  let stats = [ "no term"; "a NaN term"; "finite terms" ] in
  let groups =
    [
      ("Logsumexp", stats @ [ "+∞ and no NaN"; "-∞ alone" ], []);
      ( "Moments",
        stats @ [ "an infinity and no NaN"; "both infinities and no NaN" ],
        [] );
      ("Arg", [ "a NaN term"; "numbers" ], [ "a tie"; "-0 beside +0" ]);
    ]
  in
  List.concat_map
    (fun (name, labels, reduced) ->
      List.map (named name false) (labels @ reduced)
      @ List.filter_map
          (fun l -> if l = "no term" then None else Some (named name true l))
          labels)
    groups

let fits k t =
  match (k, t) with
  | Exactly Some_nan, t -> is_nan t
  | Exactly (Bits b), t -> bits_of t = b
  | Infinite v, F (g, _) -> g = v
  | Number, F (g, _) -> Float.is_finite g
  | Not_negative, F (g, _) -> Float.is_finite g && g >= 0.
  | _, I _ -> false

let show_klass = function
  | Exactly w -> show w
  | Infinite v -> Printf.sprintf "%h" v
  | Number -> "a finite number"
  | Not_negative -> "a finite number not negative"

(* [c] by [b]'s kernels, on its device, into fresh destinations, their
   terms. *)
let run_special (b : Support.backend) c =
  let (module K : Nx_kernel.S) = b.kernels in
  let st = c.stat in
  let (A.Any x as xa) = on b st.x in
  let s = if c.scanned then shape_of st.x else stat_shape st in
  let like dt = A.Any (A.create b.device dt (Iarray.of_array s)) in
  let dsts =
    match st.r with
    | Arg _ -> [| like (A.dtype x); like D.Int64 |]
    | Moments -> [| like (A.dtype x); like (A.dtype x) |]
    | Monoid _ | Logsumexp -> [| like (A.dtype x) |]
  in
  let p = identity (dtype_of xa) and loads : _ iarray = [| plain_of xa |] in
  let dsts = Iarray.of_array dsts in
  let answer =
    if c.scanned then K.scan p ~axis:st.axes.(0) (st.r, 0) ~dsts loads
    else
      K.reduce p
        ~shape:(Iarray.of_array (shape_of xa))
        ~axes:(Iarray.of_array st.axes) [| (st.r, 0) |] ~dsts loads
  in
  answered b p [| (st.r, 0) |] "the kernels" answer;
  Array.map (fun d -> terms (host d)) (Iarray.to_array dsts)

(* Logsumexp is -∞ for no term or terms all -∞, +∞ where a term is +∞ and
   none NaN, a NaN where one is; Moments a NaN for no term or a NaN term,
   the mean the sum of the infinities with the variance a NaN, the variance
   never negative; Arg its extreme as Max or Min finds it, at the first
   term of its bits or the first NaN; a scan's result that of the terms up
   to it. A finite result is checked as finite alone. *)
let law_special (b : Support.backend) c =
  let st = c.stat in
  let (A.Any x) = st.x in
  let d = D.Any (A.dtype x) in
  let name = match st.r with Arg _ -> "Arg" | r -> reduction_name r in
  let label l =
    Printf.sprintf "%s%s: %s" name (if c.scanned then " scan" else "") l
  in
  let ts = terms st.x in
  let outs = if c.scanned then [] else outputs (as_case st) ts in
  let want =
    if c.scanned then
      results { (as_case st) with scan = true } ts ~fill:("", [||])
        ~reduce:(fun _ -> assert false)
        ~scan:(special_expect d st.r ~scanned:true)
    else
      Array.of_list
        (List.map (fun o -> (special_expect d st.r ~scanned:false o).(0)) outs)
  in
  let seen = Hashtbl.create 16 in
  Array.iter (fun (l, _) -> Hashtbl.replace seen (label l) ()) want;
  (match st.r with
  | Arg e when not c.scanned ->
      List.iter
        (fun o ->
          List.iter
            (fun l -> Hashtbl.replace seen (label l) ())
            (arg_cases d e o))
        outs
  | _ -> ());
  List.iter (fun l -> cover l (Hashtbl.mem seen l)) special_labels;
  let got = run_special b c in
  Array.iteri
    (fun k g ->
      let bad = ref [] in
      Array.iteri
        (fun i (_, ks) ->
          if (not (fits ks.(k) g.(i))) && List.length !bad < 8 then
            bad :=
              Printf.sprintf "result %d: %Lx, expected %s" i
                (bits_of g.(i))
                (show_klass ks.(k))
              :: !bad)
        want;
      equal
        ~msg:(Printf.sprintf "destination %d" k)
        (list string) [] (List.rev !bad))
    got

(* Programs reduced and scanned *)

(* A reduction along [axes], or a scan along [axes.(0)], of a program drawn from
   [seed] over a loop of [shape], its operands of [dts], its first load padded
   where [padded], of hundreds of nodes where [long]. An example of
   {!law_several} may state its reductions, of an identity program of its one
   operand. *)
type programmed = {
  scanned : bool;
  summed : bool;
  shape : int array;
  axes : int array;
  dts : D.any array;
  padded : bool;
  long : bool;
  stated : (Nx_kernel.reduction * int) array option;
  seed : int;
}

let pp_programmed ppf c =
  Format.fprintf ppf "%s along %a of a loop of %a over %a%s%s, seed %d"
    (if c.summed then "float sum scan"
     else if c.scanned then "scan"
     else "reduce")
    pp_ints c.axes pp_ints c.shape
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_dtype)
    (Array.to_list c.dts)
    (if c.padded then ", padded" else "")
    (if c.long then ", long" else "")
    c.seed;
  Option.iter
    (fun rs ->
      Format.fprintf ppf ", stated: %s"
        (String.concat ", "
           (Array.to_list (Array.map (fun (r, _) -> reduction_name r) rs))))
    c.stated

(* The floats narrower than float32. *)
let narrow_floats =
  D.
    [
      Any Float16;
      Any Bfloat16;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
      Any Float4_e2m1fn;
    ]

(* Every dtype a program reads: the narrow floats and integers and [bit] among
   them. *)
let narrows =
  base @ complexes @ narrow_floats @ D.[ Any Int4; Any Uint4; Any Bit ]

(* Rows of 3 to 5000 terms, as many as hold about 8192 elements, past a lane, a
   block and a scan's chunk. *)
let rows =
  let open Gen in
  let* k = of_list ~pp:Format.pp_print_int [ 3; 64; 1025; 5000 ] in
  let+ r = int_range 1 (Int.min 40 (Int.max 1 (8192 / k))) in
  [| r; k |]

(* Loops of small extents and rows, the long axis reduced or scanned.
   {!programmed_examples} reduce and scan rows past 64 blocks of terms. *)
let programmed =
  let open Gen in
  let small = array ~size:(int_range 1 4) (int_range 1 5) in
  let* scanned = bool in
  let* summed = if scanned then bool else constant false in
  let* shape = frequency [ (1, small); (1, rows) ] in
  let r = Array.length shape in
  let last = r - 1 in
  let* axes =
    if shape.(last) > 1000 then constant [| last |]
    else if scanned then map (fun a -> [| a |]) (int_range 0 last)
    else
      let+ keep = array ~size:(constant r) bool in
      Array.of_list (List.filter (fun i -> keep.(i)) (List.init r Fun.id))
  in
  let* dts = array ~size:(int_range 1 3) (of_list ~pp:pp_dtype narrows) in
  let* padded = bool in
  let* long = frequency [ (2, constant false); (1, constant true) ] in
  let+ seed = int in
  { scanned; summed; shape; axes; dts; padded; long; stated = None; seed }

(* A long program over two rows past 64 blocks of terms, reduced and scanned,
   which the draws never reach. *)
let long_programs =
  List.map
    (fun scanned ->
      {
        scanned;
        summed = false;
        shape = [| 2; 70_000 |];
        axes = [| 1 |];
        dts = D.[| Any Float32; Any Float16 |];
        padded = true;
        long = true;
        stated = None;
        seed = 7;
      })
    [ false; true ]

(* Float sum scans along 64 terms of float32 rows, reductions of them, and
   scans of their columns, whose seeds cast to float32, which a library
   that declines float64 computes, and draw a Logsumexp or Moments: the
   draws reach each about one case in 11 to 20. *)
let programmed_examples =
  let rows ~scanned seed =
    {
      scanned;
      summed = scanned;
      shape = [| 3; 64 |];
      axes = [| 1 |];
      dts = [| D.Any D.Float32 |];
      padded = false;
      long = false;
      stated = None;
      seed;
    }
  in
  List.map (rows ~scanned:true) [ 1; 2; 3; 4 ]
  @ List.map (rows ~scanned:false) [ 1; 2; 3; 4; 5; 6 ]
  @ List.map
      (fun seed ->
        { (rows ~scanned:true seed) with summed = false; axes = [| 0 |] })
      [ 1; 2 ]
  @ long_programs

(* A load of [dt] whose loop has [shape]: through a drawn view, or, with
   [padded], padded by up to two elements at each end and one between
   neighbours with a drawn fill. *)
let programmed_load st (D.Any dt) shape ~padded =
  let seed = Random.State.bits st in
  let pick n = Random.State.int st n in
  if not padded then
    let v =
      {
        perm = Array.init (Array.length shape) Fun.id;
        stepped = pick 2 = 0;
        reversed = (if pick 2 = 0 then Some (pick (Array.length shape)) else None);
        broadcast = pick 4 = 0;
      }
    in
    let (A.Any x) = operand (D.Any dt) shape v ~specials:500 seed in
    Nx_kernel.Plain x
  else
    let fit e =
      let lo = pick 3 and hi = pick 3 and interior = pick 2 in
      let d = e - lo - hi + interior in
      if d >= 1 && d mod (interior + 1) = 0 then (lo, hi, interior, d / (interior + 1))
      else (0, 0, 0, e)
    in
    let axes = Array.map fit shape in
    let get f = Iarray.of_array (Array.map f axes) in
    let pad =
      {
        Nx_kernel.lo = get (fun (l, _, _, _) -> l);
        hi = get (fun (_, h, _, _) -> h);
        interior = get (fun (_, _, i, _) -> i);
        windows = [||];
      }
    in
    let x = elements dt (total (Array.map (fun (_, _, _, d) -> d) axes)) ~specials:500 seed in
    let x =
      Option.get (A.move (M.Reshape (Array.map (fun (_, _, _, d) -> d) axes |> Iarray.of_array)) x)
    in
    let fill = A.get (elements dt 1 ~specials:3 (seed + 1)) [| 0 |] in
    Nx_kernel.Padded { x; fill; pad }

(* [l] on [b]'s device. *)
let load_on b : Nx_kernel.load -> Nx_kernel.load = function
  | Plain x -> Plain (Support.on b x)
  | Padded p -> Padded { p with x = Support.on b p.x }

let reductions_all : Nx_kernel.reduction list =
  [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Logsumexp; Moments;
    Arg Max; Arg Min ]

(* One to three reductions of [p]'s outputs, each of a dtype it takes;
   one with [scanned]. *)
let programmed_reductions st (p : P.t) ~scanned =
  let outs = Iarray.length p.outs in
  let one () =
    let k = Random.State.int st outs in
    let (D.Any dt) = P.dtype p (Iarray.get p.outs k) in
    let ok = List.filter (fun r -> Nx_kernel.accepts r dt) reductions_all in
    (List.nth ok (Random.State.int st (List.length ok)), k)
  in
  Array.init (if scanned then 1 else 1 + Random.State.int st 3) (fun _ -> one ())

(* [p], of node dtypes [types], with its first output cast to float32 or
   float64 as its one output, where a cast takes it. *)
let cast_out st (p : P.t) types =
  let o = Iarray.get p.outs 0 in
  let (D.Any x) = types.(o) in
  let (D.Any y as dt) =
    if Random.State.bool st then D.Any D.Float32 else D.Any D.Float64
  in
  if not (P.accepts1 Cast x y) then None
  else
    let n = Iarray.length p.nodes in
    let nodes = Iarray.append p.nodes (Iarray.of_list [ P.Op1 (Cast, dt, o) ]) in
    Some (P.v ~ins:p.ins nodes ~outs:(Iarray.of_list [ n ]), Array.append types [| dt |])

(* Fresh destinations for [rs] of [p], of shape [s] on [b]'s device: each
   in its output's dtype, an Arg's position in int64. *)
let programmed_dsts (b : Support.backend) (p : P.t) rs s =
  Array.concat
    (List.map
       (fun ((r : Nx_kernel.reduction), k) ->
         let (D.Any dt) = P.dtype p (Iarray.get p.outs k) in
         let y () = A.Any (A.create b.device dt (Iarray.of_array s)) in
         match r with
         | Arg _ -> [| y (); A.Any (A.create b.device D.Int64 (Iarray.of_array s)) |]
         | Moments -> [| y (); y () |]
         | Monoid _ | Logsumexp -> [| y () |])
       (Array.to_list rs))

(* A reduction or a scan of a program's outputs has the bits of the same
   reduction or scan of those outputs mapped into arrays first, any NaN for
   a NaN, in a library whose association follows the loop's extents, the
   dtypes and the scanned axis alone, whatever reads its terms: nx_cpu.mli's
   sums and products and nx_cuda.mli's loops state it. *)
(* [c]'s program, loads, reductions and their shape on [b]'s device, drawn
   from its seed: where [c.summed], a float sum scan of its first output
   cast to float32 or float64. *)
let programmed_parts (b : Support.backend) c =
  let st = Random.State.make [| c.seed |] in
  let rank = Array.length c.shape in
  let length = if c.long then 400 else 6 in
  let p, types = Kernels_cases.program ~length st c.dts ~room:16 ~rank in
  let loads =
    Array.mapi
      (fun i dt ->
        let padded = c.padded && i = 0 in
        load_on b (programmed_load st dt c.shape ~padded))
      c.dts
  in
  let summed = if c.summed then cast_out st p types else None in
  let p, types = Option.value summed ~default:(p, types) in
  let rs =
    if Option.is_some summed then [| (Nx_kernel.Monoid Sum, 0) |]
    else programmed_reductions st p ~scanned:c.scanned
  in
  let ys =
    if c.scanned then c.shape
    else
      Result.get_ok
        (Nx_kernel.Shape.reduce ~axes:(Iarray.of_array c.axes)
           (Array.map fst rs) c.shape)
  in
  (p, types, loads, rs, ys)

(* [c]'s reduction or scan of [p] over [loads] into fresh destinations on
   [b]'s device. *)
let programmed_call (b : Support.backend) c (p, _, loads, rs, ys) =
  let module K = (val b.kernels) in
  let dsts = programmed_dsts b p rs ys in
  let ds = Iarray.of_array dsts and loads = Iarray.of_array loads in
  answered b p rs "the call"
    (if c.scanned then K.scan p ~axis:c.axes.(0) rs.(0) ~dsts:ds loads
     else
       K.reduce p ~shape:(Iarray.of_array c.shape)
         ~axes:(Iarray.of_array c.axes) (Iarray.of_array rs) ~dsts:ds loads);
  dsts

(* [b], or a skip where it does not state nx.cpu's lane order. *)
let lane_ordered (b : Support.backend) =
  if not b.lane_order then
    skip ~reason:(b.name ^ " does not state nx.cpu's lane order") ()

let law_programmed (b : Support.backend) c =
  lane_ordered b;
  let module K = (val b.kernels) in
  let ((p, types, loads, rs, ys) as parts) = programmed_parts b c in
  let call p ~dsts loads =
    let dsts = Iarray.of_array dsts and loads = Iarray.of_array loads in
    if c.scanned then K.scan p ~axis:c.axes.(0) rs.(0) ~dsts loads
    else
      K.reduce p ~shape:(Iarray.of_array c.shape)
        ~axes:(Iarray.of_array c.axes) (Iarray.of_array rs) ~dsts loads
  in
  let got = programmed_call b c parts in
  let outs = Iarray.to_array p.outs in
  let mapped =
    Array.map
      (fun o ->
        let (D.Any dt) = types.(o) in
        A.Any (A.create b.device dt (Iarray.of_array c.shape)))
      outs
  in
  answered b p [||] "the map"
    (K.map p ~dsts:(Iarray.of_array mapped) (Iarray.of_array loads));
  let identities =
    P.v ~ins:(Iarray.of_array (Array.map (fun o -> types.(o)) outs))
      (Iarray.init (Array.length outs) (fun k -> P.In k))
      ~outs:(Iarray.init (Array.length outs) Fun.id)
  in
  let want = programmed_dsts b p rs ys in
  answered b identities rs "the call of the mapped outputs"
    (call identities ~dsts:want (Array.map plain_of mapped));
  let terms =
    if c.scanned then c.shape.(c.axes.(0))
    else Array.fold_left (fun n a -> n * c.shape.(a)) 1 c.axes
  in
  let has f = Iarray.exists f p.nodes in
  let innermost =
    c.scanned
    && Array.for_all (( = ) 1)
         (Array.sub c.shape (c.axes.(0) + 1) (Array.length c.shape - c.axes.(0) - 1))
  in
  let summed = c.summed && rs = [| (Nx_kernel.Monoid Sum, 0) |] in
  cover "a float sum scan of the innermost axis past 32 terms"
    (summed && innermost && terms > 32);
  cover "a scan of another axis" (c.scanned && not innermost);
  cover "padded loads" c.padded;
  cover "a long program over a padded load" (c.padded && c.long);
  cover "a coordinate" (has (function P.Coord _ -> true | _ -> false));
  cover "several loads" (Array.length c.dts > 1);
  cover "more than eight values held" (Kernels_cases.held p > 8);
  cover "several reductions" (Array.length rs > 1);
  cover "Logsumexp or Moments"
    (Array.exists (fun (r, _) -> r = Nx_kernel.Logsumexp || r = Moments) rs);
  cover "an Arg" (Array.exists (function Nx_kernel.Arg _, _ -> true | _ -> false) rs);
  cover "an output of more than 16 blocks of terms" ((not c.scanned) && terms > 16 * 1024);
  cover "an output of more than 64 blocks of terms" ((not c.scanned) && terms > 64 * 1024);
  cover "a scan past one chunk" (c.scanned && terms > 4096);
  Array.iteri
    (fun k (A.Any w) ->
      let (A.Any g) = got.(k) in
      equal
        ~msg:(Printf.sprintf "destination %d of %s" k (Kernels_cases.describe p))
        (Kernels_cases.same_values (D.Any (A.dtype w)))
        (Kernels_cases.bits_of (Support.host w))
        (Kernels_cases.bits_of (Support.host g)))
    want

(* Several reductions of one program *)

(* Loops as [programmed]'s, of zero to five elements an axis among the small
   ones, reduced; rows of planes reduced across a kept axis; columns reduced
   across rows, in pairs too; each of at most about 8192 elements.
   [several_examples] runs each size past that a cover needs. *)
let several =
  let open Gen in
  let small = array ~size:(int_range 1 4) (int_range 0 5) in
  (* Rows of planes reduced across the planes' kept axis. *)
  let parted =
    let* n = of_list ~pp:Format.pp_print_int [ 3; 64; 1025 ] in
    let* p = int_range 2 (if n > 1000 then 2 else 6) in
    let+ k = int_range 2 (if n > 1000 then 3 else 5) in
    ([| p; k; n |], [| 0; 2 |])
  in
  let columns =
    let* k = of_list ~pp:Format.pp_print_int [ 2; 64; 129; 300 ] in
    let+ r = int_range 17 (Int.max 17 (8192 / k)) in
    ([| r; k |], [| 0 |])
  in
  (* Pairs of columns. *)
  let paired =
    let+ n = int_range 2 64 in
    ([| 64; n; 2 |], [| 1 |])
  in
  let drawn =
    let* shape = frequency [ (1, small); (1, rows) ] in
    let r = Array.length shape in
    let+ axes =
      if shape.(r - 1) > 1000 then constant [| r - 1 |]
      else
        let+ keep = array ~size:(constant r) bool in
        Array.of_list (List.filter (fun i -> keep.(i)) (List.init r Fun.id))
    in
    (shape, axes)
  in
  (* Columns of complex loads half the time, whose products start their lanes
     from their first terms; columns of narrow float loads, whose programs'
     outputs are narrow floats often enough that every run reduces one across
     rows. *)
  let loads dtypes g =
    let+ x = g in
    (x, dtypes)
  in
  let complex_columns =
    let* x = columns in
    let+ c = bool in
    (x, if c then complexes else narrows)
  in
  let* (shape, axes), dtypes =
    frequency
      [
        (7, loads narrows drawn);
        (1, loads narrows parted);
        (1, complex_columns);
        (1, loads narrows paired);
        (2, loads narrow_floats (one_of [ columns; paired ]));
      ]
  in
  let* dts = array ~size:(int_range 1 3) (of_list ~pp:pp_dtype dtypes) in
  let* padded = bool in
  let* long = frequency [ (2, constant false); (1, constant true) ] in
  let+ seed = int in
  {
    scanned = false;
    summed = false;
    shape;
    axes;
    dts;
    padded;
    long;
    stated = None;
    seed;
  }

(* A long program of two loads over two rows past 64 blocks of terms; the sizes
   nx.cpu's paths for several reductions change at, each with the reductions it
   takes: a Moments and a Sum along rows of more than eight blocks, 64 outputs,
   a group of blocks each; a Sum and a Max of float16 across 1788 rows of 64
   outputs; and a Sum and a Prod across a block and a term of 64 pairs of
   columns; and a Sum and a Min of float16 along the innermost axis and one a
   kept axis parts from it, which the draws reach one case in 16. *)
let several_examples =
  let stated dt shape axes rs =
    {
      scanned = false;
      summed = false;
      shape;
      axes;
      dts = [| dt |];
      padded = true;
      long = false;
      stated = Some rs;
      seed = 1;
    }
  in
  Nx_kernel.(
    List.hd long_programs
    :: [
         stated f32 [| 64; 8193 |] [| 1 |] [| (Moments, 0); (Monoid Sum, 0) |];
         stated (D.Any D.Float16) [| 1788; 64 |] [| 0 |]
           [| (Monoid Sum, 0); (Monoid Max, 0) |];
         stated f32 [| 64; 1025; 2 |] [| 1 |]
           [| (Monoid Sum, 0); (Monoid Prod, 0) |];
         stated (D.Any D.Float16) [| 2; 3; 64 |] [| 0; 2 |]
           [| (Monoid Sum, 0); (Monoid Min, 0) |];
       ])

(* nx_cpu.mli: a Reduce of several reductions computes each one's bits alone, so
   its destinations hold the bits of each reduction reduced by a call of its
   own, any NaN for a NaN. *)
let law_several (b : Support.backend) c =
  let module K = (val b.kernels) in
  let st = Random.State.make [| c.seed |] in
  let rank = Array.length c.shape in
  let length = if c.long then 400 else 6 in
  let p =
    if Option.is_some c.stated then identity c.dts.(0)
    else fst (Kernels_cases.program ~length st c.dts ~room:16 ~rank)
  in
  let loads =
    Array.mapi
      (fun i dt ->
        load_on b (programmed_load st dt c.shape ~padded:(c.padded && i = 0)))
      c.dts
  in
  let terms = Array.fold_left (fun n a -> n * c.shape.(a)) 1 c.axes in
  (* Two to five reductions, but no extreme of no term: none where every
     output is of a dtype only extremes take. *)
  let pairs =
    List.concat
      (List.init (Iarray.length p.outs) (fun k ->
           let (D.Any dt) = P.dtype p (Iarray.get p.outs k) in
           List.filter_map
             (fun (r : Nx_kernel.reduction) ->
               let extreme =
                 match r with Monoid (Max | Min) | Arg _ -> true | _ -> false
               in
               if Nx_kernel.accepts r dt && (terms > 0 || not extreme) then
                 Some (r, k)
               else None)
             reductions_all))
  in
  let rs =
    match c.stated with
    | Some rs -> rs
    | None ->
        let n = if pairs = [] then 0 else 2 + Random.State.int st 4 in
        Array.init n (fun _ ->
            List.nth pairs (Random.State.int st (List.length pairs)))
  in
  let n = Array.length rs in
  let ys =
    Result.get_ok
      (Nx_kernel.Shape.reduce ~axes:(Iarray.of_array c.axes) (Array.map fst rs)
         c.shape)
  in
  let reduce p rs loads =
    let dsts = programmed_dsts b p rs ys in
    answered b p rs "the call"
      (K.reduce p ~shape:(Iarray.of_array c.shape)
         ~axes:(Iarray.of_array c.axes) (Iarray.of_array rs)
         ~dsts:(Iarray.of_array dsts) (Iarray.of_array loads));
    dsts
  in
  let got = if n = 0 then [||] else reduce p rs loads in
  (* Each reduction alone, over the program's outputs mapped once: a reduction
     of an output has the bits of the output mapped, then reduced
     (law_programmed), and a case runs its program twice, not once a reduction
     more. *)
  let want =
    if n = 0 then [||]
    else
      let outs = Iarray.to_array p.outs in
      let types = Array.map (P.dtype p) outs in
      let mapped =
        Array.map
          (fun (D.Any dt) ->
            A.Any (A.create b.device dt (Iarray.of_array c.shape)))
          types
      in
      answered b p [||] "the map"
        (K.map p ~dsts:(Iarray.of_array mapped) (Iarray.of_array loads));
      let identities =
        P.v ~ins:(Iarray.of_array types)
          (Iarray.init (Array.length outs) (fun k -> P.In k))
          ~outs:(Iarray.init (Array.length outs) Fun.id)
      in
      let mapped = Array.map plain_of mapped in
      Array.concat
        (List.map
           (fun r -> reduce identities [| r |] mapped)
           (Array.to_list rs))
  in
  let outs = Array.map snd rs in
  let one_pass (r, _) =
    match (r : Nx_kernel.reduction) with Monoid _ | Arg _ -> true | _ -> false
  in
  let narrow (r, k) =
    let (D.Any dt) = P.dtype p (Iarray.get p.outs k) in
    one_pass (r, k) && D.is D.Float dt && D.bits dt < 32
  in
  let twice k = List.length (List.filter (( = ) k) (Array.to_list outs)) > 1 in
  cover "a loop of no element" (Array.exists (( = ) 0) c.shape);
  cover "outputs of one term" (terms = 1 && Array.for_all (( < ) 0) c.shape);
  cover "outputs of more than a lane of terms" (terms > 16);
  cover "an output of more than 16 blocks of terms" (terms > 16 * 1024);
  cover "an output of more than 64 blocks of terms" (terms > 64 * 1024);
  (* nx_cpu.mli: these run the program once a term for them all. *)
  let shared =
    List.length (List.filter one_pass (Array.to_list rs)) > 1
    && Array.mem (rank - 1) c.axes && terms > 16
  in
  let arg = function Nx_kernel.Arg _, _ -> true | _ -> false in
  let gap =
    List.exists
      (fun a ->
        (not (Array.mem a c.axes))
        && c.shape.(a) > 1
        && Array.exists (fun b -> b < a) c.axes)
      (List.init rank Fun.id)
  in
  cover "two monoids or Args along the innermost axis, past a lane of terms"
    shared;
  cover "those of a narrow float output" (shared && Array.exists narrow rs);
  cover "those with an Arg" (shared && Array.exists arg rs);
  cover "those along axes a kept one parts" (shared && gap);
  let moments = function
    | (Nx_kernel.Moments | Logsumexp), _ -> true
    | _ -> false
  in
  let monoid = function Nx_kernel.Monoid _, _ -> true | _ -> false in
  let count f = List.length (List.filter f (Array.to_list rs)) in
  let outputs =
    List.fold_left
      (fun n a -> if Array.mem a c.axes then n else n * c.shape.(a))
      1 (List.init rank Fun.id)
  in
  let with_moments =
    count (fun r -> one_pass r || moments r) > 1
    && Array.exists moments rs
    && Array.mem (rank - 1) c.axes
    && terms > 16
  in
  cover
    "a Moments or a Logsumexp with a monoid, an Arg, a Moments or a \
     Logsumexp along the innermost axis"
    with_moments;
  cover "those over units of a group of more than eight blocks"
    (with_moments && terms > 8 * 1024 && outputs >= 64);
  let across =
    count monoid > 1
    && (not (Array.mem (rank - 1) c.axes))
    && c.shape.(rank - 1) > 1
  in
  cover "two monoids across rows, past a lane of terms" (across && terms > 16);
  cover "those past a block of terms" (across && terms > 1024);
  cover "those over units of a group of more than one block"
    (across && terms > 1024 && outputs >= 64);
  cover "those across rows of a narrow float output"
    (across && terms > 16 && Array.exists narrow rs);
  cover "two monoids of outputs of a lane of terms or fewer"
    (count monoid > 1 && terms > 0 && terms <= 16 && outputs > 0);
  cover "an output reduced twice" (Array.exists twice outs);
  cover "padded loads" c.padded;
  Array.iteri
    (fun k (A.Any w) ->
      let (A.Any g) = got.(k) in
      equal
        ~msg:
          (Printf.sprintf "destination %d of %s" k (Kernels_cases.describe p))
        (Kernels_cases.same_values (D.Any (A.dtype w)))
        (Kernels_cases.bits_of (Support.host w))
        (Kernels_cases.bits_of (Support.host g)))
    want

(* Calls too wide for a launch *)

let wide_reductions : Nx_kernel.reduction list =
  [ Monoid Sum; Monoid Max; Arg Min; Logsumexp; Moments; Monoid Prod ]

(* A reduction or a scan too wide for one of nx.cuda's launches, by its
   loads, its values held at once or its reductions, has nx.cpu's bits,
   any NaN for a NaN, in a library that states nx.cpu's association and
   gradual underflow; a scan in one that states its own association, and
   a call in one that flushes, the bits of its outputs mapped, then
   reduced or scanned. In one that flushes, the outputs it maps have
   nx.cpu's bits node by node, each node's operands and result flushed.
   A scan scans along the innermost axis of more than one element; a
   reduction reduces axes 0 and 1. *)
(* A wide call as a failure names it. *)
let wide_name (c : Kernels_cases.wide) scanned =
  Format.asprintf "%a, %s" Kernels_cases.pp_wide c
    (if scanned then "scanned" else "reduced")
let wide_call (c : Kernels_cases.wide) scanned =
  let p, n = Kernels_cases.wide_program c in
  let shape = Kernels_cases.wide_shape c in
  let scanned = scanned && c.mode <> Outputs in
  let name = wide_name c scanned in
  let axis = if c.padded then 1 else 2 in
  let (D.Any d) = c.dt in
  let ok = List.filter (fun r -> Nx_kernel.accepts r d) wide_reductions in
  let rs =
    let outs = Iarray.length p.outs in
    let count = if c.mode = Outputs then 500 else if scanned then 1 else 2 in
    Array.init count (fun i ->
        (List.nth ok (i mod List.length ok), i mod outs))
  in
  let axes = Iarray.of_list [ 0; 1 ] in
  let ys =
    if scanned then shape
    else Result.get_ok (Nx_kernel.Shape.reduce ~axes (Array.map fst rs) shape)
  in
  let call (b : Support.backend) =
    let module K = (val b.kernels) in
    let loads, _ = Kernels_cases.wide_loads b c n in
    let dsts = programmed_dsts b p rs ys in
    let ds = Iarray.of_array dsts and loads = Iarray.of_array loads in
    let answer =
      if scanned then K.scan p ~axis rs.(0) ~dsts:ds loads
      else
        K.reduce p ~shape:(Iarray.of_array shape) ~axes (Iarray.of_array rs)
          ~dsts:ds loads
    in
    answered b p rs name answer;
    dsts
  in
  (* [ms], [b]'s outputs of [p], against nx.cpu's node by node, every value
     [b] flushes flushed. *)
  let flushed_map (b : Support.backend) ms =
    let flush d = Kernels_cases.flushes b "add" d in
    let _, xs = Kernels_cases.wide_loads Support.cpu c n in
    let xs =
      Array.map
        (fun (A.Any x as a) ->
          if flush (D.Any (A.dtype x)) then Kernels_cases.flushed_view a else a)
        xs
    in
    let types = Array.init (Iarray.length p.nodes) (P.dtype p) in
    let values =
      Support.cpu.around (fun () ->
          Kernels_cases.split ~flush Support.cpu p types
            (Iarray.of_array shape) xs)
    in
    Array.iteri
      (fun k (A.Any m) ->
        let (A.Any w) = values.(Iarray.get p.outs k) in
        let d = D.Any (A.dtype w) and want = Kernels_cases.bits_of (Support.host w) in
        let got = Kernels_cases.bits_of (Support.host m) in
        let msg = Printf.sprintf "output %d mapped" k in
        if flush d then
          equal ~msg (Kernels_cases.flushed_values ~any_nan:true d)
            (Kernels_cases.flush_bits d want) got
        else equal ~msg (Kernels_cases.same_values d) want got)
      ms
  in
  let mapped (b : Support.backend) =
    let module K = (val b.kernels) in
    let loads, _ = Kernels_cases.wide_loads b c n in
    let types = Array.map (P.dtype p) (Iarray.to_array p.outs) in
    let ms =
      Array.map
        (fun (D.Any d) -> A.Any (A.create b.device d (Iarray.of_array shape)))
        types
    in
    answered b p [||] (name ^ ", its map")
      (K.map p ~dsts:(Iarray.of_array ms) (Iarray.of_array loads));
    if b.underflow = Flush then flushed_map b ms;
    let nouts = Array.length types in
    let id =
      P.v ~ins:(Iarray.of_array types)
        (Iarray.init nouts (fun k -> P.In k))
        ~outs:(Iarray.init nouts Fun.id)
    in
    let dsts = programmed_dsts b id rs ys in
    let ds = Iarray.of_array dsts and ls = Iarray.map plain_of (Iarray.of_array ms) in
    answered b id rs (name ^ ", its map " ^ if scanned then "scanned" else "reduced")
      (if scanned then K.scan id ~axis rs.(0) ~dsts:ds ls
       else
         K.reduce id ~shape:(Iarray.of_array shape) ~axes (Iarray.of_array rs)
           ~dsts:ds ls);
    dsts
  in
  (scanned, call, mapped)

(* Every wide call of [seed] reduced, and scanned but for those of many
   reductions, which have no scan: 50. A reduction reads no destination
   identical to a load, so the calls into one are left out. *)
let wide_scans seed =
  List.concat_map
    (fun (c : Kernels_cases.wide) ->
      if c.aliased then []
      else if c.mode = Outputs then [ (c, false) ]
      else [ (c, false); (c, true) ])
    (Kernels_cases.wide_calls seed)

let law_wide (b : Support.backend) ((c : Kernels_cases.wide), scanned) =
  lane_ordered b;
  let scanned, call, mapped = wide_call c scanned in
  (* A library that flushes computes nx.cpu's bits only where nothing
     underflows: its own map, which has nx.cpu's bits under the flush,
     then reduction or scan, of the program's outputs gives the bits. *)
  let want =
    if (scanned && not b.scan_order) || b.underflow = Flush then mapped b
    else Support.cpu.around (fun () -> call Support.cpu)
  in
  let got = call b in
  cover "past a launch's loads" (c.mode = Loads);
  cover "past a launch's values held" (c.mode = Held);
  cover "past a launch's reductions" (c.mode = Outputs);
  cover "a padded load copied out" (c.mode = Loads && c.padded);
  cover "a scan" scanned;
  Array.iteri
    (fun k (A.Any w) ->
      let (A.Any g) = got.(k) in
      equal
        ~msg:(Printf.sprintf "%s, destination %d" (wide_name c scanned) k)
        (Kernels_cases.same_values (D.Any (A.dtype w)))
        (Kernels_cases.bits_of (Support.host w))
        (Kernels_cases.bits_of (Support.host g)))
    want

(* nx.cpu's bits *)

(* Float and complex sums and products of every float dtype, narrow ones
   included, reduced, or scanned where [scans]. *)
(* A narrow float's terms of its full precision and mixed magnitudes,
   within 2^±6 at 16 bits and 2^±2 narrower, so that their float32 sums and
   products round otherwise in another order, a NaN, an infinity or -0 at
   the rate [specials]; other dtypes' as {!Reduce_cases.elements} draws
   them. *)
let mixed_values (D.Any dt) n ~specials seed =
  match D.kind dt with
  | D.Float when D.bits dt < 32 ->
      let r = Random.State.make [| seed |] in
      let span = if D.bits dt = 16 then 6 else 2 in
      let specials' = [| nan; infinity; neg_infinity; -0. |] in
      let one () =
        if specials > 0 && Random.State.int r specials = 0 then
          specials'.(Random.State.int r 4)
        else
          Float.ldexp
            (Random.State.float r 2. -. 1.)
            (Random.State.int r ((2 * span) + 1) - span)
      in
      A.Any (A.of_array dt [| n |] (Array.init n (fun _ -> one ())))
  | _ -> A.Any (elements dt n ~specials seed)

let associated ~scans =
  Gen.with_pp pp_case
    (case_of ~scan:(Gen.constant scans) ~monoids:[ Sum; Prod ]
       ~values:mixed_values
       (narrow_floats @ floats @ complexes))

(* A reduction by [monoid] along [axes] of [mixed_values] of [d] and shape [s],
   with no special. *)
let mixed_case monoid d s axes =
  let (A.Any x) = mixed_values d (total s) ~specials:0 1 in
  let x = Option.get (A.move (M.Reshape (Iarray.of_array s)) x) in
  { scan = false; monoid; axes; x = A.Any x; views = [] }

(* [x] reversed along [axes]: a reduction of it folds its terms in another
   association. *)
let reversed_along axes (A.Any x) =
  let r =
    Iarray.mapi
      (fun a e ->
        if Array.mem a axes && e > 0 then
          { M.start = e - 1; count = e; step = -1 }
        else { M.start = 0; count = e; step = 1 })
      (L.shape (A.layout x))
  in
  A.Any (Option.get (A.move (M.Slice r) x))

(* nx.cpu's association (reduce.c): an output's terms in blocks of 1024, term
   [16 j + l] of a block folded in order into lane [l] from the identity, the
   lanes holding a term then the blocks combined by the pairwise tree; a scan's
   slice in chunks of 4096, each result its chunk's carry combined with the
   chunk's terms up to it, left to right, the carry into chunk 0 the identity
   and into chunk [c + 1] the carry into [c] combined with [c]'s total, its
   terms reduced. A complex product's lanes, and its chunk 0, start from their
   first term instead. The model's combines compute in float32, on doubles that
   hold float32 values; the state of a float narrower than float32 is a float32
   one, rounded once at the end. Where [flush], each combine computes as
   nx_metal.mli says Add and Mul do at underflow: a nonzero exact result below
   2^-126 is a zero of its sign; else as nx.cpu does. [under] is whether a part
   of the computation fell below 2^-126. *)
type assoc = { flush : bool; mutable under : bool }

(* [hi + lo], exact, rounded once into float32 or flushed. Where [lo] is
   nonzero, [hi] rounds to odd first: an even [hi] steps one double towards
   [lo], so that a double that rounds to nearest into float32 rounds the way the
   exact value does. *)
let round32 a hi lo =
  let lo = if Float.is_finite hi then lo else 0. in
  let m = Float.abs hi in
  let tiny =
    hi <> 0.
    && (m < 0x1p-126
       || (m = 0x1p-126 && lo <> 0. && Float.sign_bit lo <> Float.sign_bit hi))
  in
  if tiny then a.under <- true;
  if tiny && a.flush then Float.copy_sign 0. hi
  else
    let even = Int64.logand (Int64.bits_of_float hi) 1L = 0L in
    let hi =
      if lo = 0. || not even then hi
      else if lo > 0. then Float.succ hi
      else Float.pred hi
    in
    Int32.float_of_bits (Int32.bits_of_float hi)

(* Float32 Add, Mul and Fma: a float32 product is exact in a double. An
   Fma's product below 2^-126 is an underflow too, though nx.cpu keeps
   it. *)
let add32 a x y =
  let s, e = two_sum x y in
  round32 a s e

let mul32 a x y = round32 a (x *. y) 0.

let fma32 a x y z =
  let p = x *. y in
  if p <> 0. && Float.abs p < 0x1p-126 then a.under <- true;
  let s, e = two_sum p z in
  round32 a s e

(* nx_kinds_complex.h's product of [x] by [y]. *)
let zmul32 a (x : Complex.t) (y : Complex.t) =
  let ii = mul32 a x.im y.im and ir = mul32 a x.im y.re in
  { Complex.re = fma32 a x.re y.re (-.ii); im = fma32 a x.re y.im ir }

(* The combine of [m] on [complex] terms, a float's in its real part. *)
let combine a ~complex m (x : Complex.t) (y : Complex.t) =
  match (m, complex) with
  | Sum, false -> { Complex.re = add32 a x.re y.re; im = 0. }
  | Sum, true -> { re = add32 a x.re y.re; im = add32 a x.im y.im }
  | Prod, false -> { re = mul32 a x.re y.re; im = 0. }
  | Prod, true -> zmul32 a x y
  | (Max | Min), _ -> invalid_arg "combine: an extreme"

(* A reduction's result from its terms [ts], in lane order, of [unit]
   where [bare] is false, else each lane from its first term. *)
let lane_fold ~bare op unit ts =
  let n = Array.length ts in
  let block k =
    let lanes = Array.make 16 unit and held = Array.make 16 false in
    for t = 1024 * k to Int.min n (1024 * (k + 1)) - 1 do
      let l = t mod 16 in
      lanes.(l) <-
        (if bare && not held.(l) then ts.(t) else op lanes.(l) ts.(t));
      held.(l) <- true
    done;
    List.iter
      (fun h ->
        for i = 0 to h - 1 do
          if held.(i + h) then lanes.(i) <- op lanes.(i) lanes.(i + h)
        done)
      [ 8; 4; 2; 1 ];
    lanes.(0)
  in
  let rec tree first count =
    if count = 1 then block first
    else
      let rec below p = if 2 * p < count then below (2 * p) else p in
      let left = below 1 in
      op (tree first left) (tree (first + left) (count - left))
  in
  if n = 0 then unit else tree 0 ((n + 1023) / 1024)

(* A scan's results from a slice's terms [ts], in chunks. *)
let chunk_scan ~bare op unit ts =
  let n = Array.length ts and chunk = 4096 in
  let y = Array.copy ts in
  let carry = ref (if bare then None else Some unit) in
  for c = 0 to ((n + chunk - 1) / chunk) - 1 do
    let first = chunk * c in
    let terms = Array.sub ts first (Int.min n (first + chunk) - first) in
    let s = ref !carry in
    Array.iteri
      (fun i x ->
        let v = match !s with None -> x | Some v -> op v x in
        s := Some v;
        y.(first + i) <- v)
      terms;
    let total = lane_fold ~bare op unit terms in
    carry := Some (match !carry with None -> total | Some v -> op v total)
  done;
  y

(* The results of [c], a reduction or scan of float32, a narrower float
   or complex64, in nx.cpu's association with [a]'s combines. *)
let associated_result a (c : case) =
  let (A.Any x) = c.x in
  let complex = D.is D.Complex (A.dtype x) in
  let op = combine a ~complex c.monoid in
  let unit =
    match c.monoid with
    | Sum -> Complex.zero
    | Prod -> Complex.one
    | Max | Min -> invalid_arg "associated_result: an extreme"
  in
  let bare = complex && c.monoid = Prod in
  let ys =
    results c (zvalues c.x) ~fill:unit ~reduce:(lane_fold ~bare op unit)
      ~scan:(chunk_scan ~bare op unit)
  in
  let shape = Iarray.of_array (result_shape c) in
  let re = Array.map (fun (z : Complex.t) -> z.re) ys in
  let go (type v s) (dt : (v, s) D.t) : A.any =
    match dt with
    | D.Float32 -> A.Any (A.of_array dt shape re)
    | D.Bfloat16 -> A.Any (A.of_array dt shape re)
    | D.Float16 -> A.Any (A.of_array dt shape re)
    | D.Float8_e4m3fn -> A.Any (A.of_array dt shape re)
    | D.Float8_e5m2 -> A.Any (A.of_array dt shape re)
    | D.Float4_e2m1fn -> A.Any (A.of_array dt shape re)
    | D.Complex64 -> A.Any (A.of_array dt shape ys)
    | _ ->
        invalid_arg "associated_result: not float32, narrower or complex64"
  in
  go (A.dtype x)

(* The least normal magnitude of [dt], a float narrower than float32 whose
   combines compute in float32, or [None]. *)
let narrow_least (D.Any dt) =
  match dt with
  | D.Float16 -> Some (D.float_format dt).min_normal
  | D.Float8_e4m3fn -> Some (D.float_format dt).min_normal
  | D.Float8_e5m2 -> Some (D.float_format dt).min_normal
  | D.Float4_e2m1fn -> Some (D.float_format dt).min_normal
  | _ -> None

(* Whether a term of [c] is nonzero and below [least] in magnitude. *)
let subnormal_term least (c : case) =
  match least with
  | None -> false
  | Some least ->
      Array.exists
        (fun (z : Complex.t) -> z.re <> 0. && Float.abs z.re < least)
        (zvalues c.x)

(* A library that keeps nx.cpu's lane order and chunks gives nx.cpu's bits: a
   float or complex sum's or product's results, reduced or scanned, are
   nx.cpu's, any NaN for a NaN. One that flushes (nx_metal.mli, Subnormal
   numbers) gives them where no part of the computation underflows; where one
   does, float32's and bfloat16's are computed with its flush at every combine,
   and complex64's keep their bounds, as the bound law states them. A narrower
   float's terms keep their subnormals, none of which is one in float32, but its
   float32 combines flush: a product of tiny terms can carry a float32
   subnormal. The association computed without the flush must give nx.cpu's
   bits, and tells whether a part underflows. A library that states nx.cpu's
   chunks for every scan gives those bits of scans of any length, where [scans];
   another states them for scans that are not its own order, such as the
   examples' short running products, and for no drawn scan. *)
let law_cpu_bits ~scans (b : Support.backend) (c : case) =
  lane_ordered b;
  if scans && not b.scan_order then
    skip ~reason:(b.name ^ " does not state nx.cpu's chunks for every scan") ();
  let s = shape_of c.x in
  let terms = total s / Int.max 1 (total (result_shape c)) in
  if scans then begin
    cover "a scan past one chunk" (s.(c.axes.(0)) > 4096);
    cover "a scan along an axis not the innermost"
      (c.axes.(0) < Array.length s - 1)
  end
  else begin
    cover "past one block of terms" (terms > 1024);
    cover "past 64 blocks, several groups an output"
      ((not c.scan) && terms > 64 * 1024)
  end;
  (* A library that flushes reads a subnormal term as a zero. *)
  let flushes = Kernels_cases.flushes b "add" (dtype_of c.x) in
  let least = narrow_least (dtype_of c.x) in
  let carries = b.underflow = Flush && Option.is_some least in
  let c = if flushes then { c with x = Kernels_cases.flushed_view c.x } else c in
  let (A.Any w) = Support.cpu.around (fun () -> run Support.cpu c) in
  (* The values tell associations apart: nx.cpu's result of the reversed terms
     differs, neither a NaN. *)
  if not scans then begin
    let back =
      Support.cpu.around (fun () ->
          run Support.cpu { c with x = reversed_along c.axes c.x })
    in
    let d = D.Any (A.dtype w) in
    cover "a result the terms' order changes"
      ((not c.scan)
      && Array.exists2
           (fun a b ->
             a <> b && not (Elements.nan_of d a || Elements.nan_of d b))
           (Elements.elements (A.Any w))
           (Elements.elements back))
  end;
  let (A.Any g) = run b c in
  let d = D.Any (A.dtype w) in
  let same = Kernels_cases.same_values d in
  let bits = Kernels_cases.bits_of in
  if not (flushes || carries) then equal same (bits w) (bits g)
  else begin
    let plain = { flush = false; under = false } in
    let (A.Any cpu) = associated_result plain c in
    equal ~msg:"nx.cpu's association" same (bits w) (bits cpu);
    let complex = D.is D.Complex (A.dtype w) in
    (* The examples reach these, which the scans' draws may not. *)
    if not scans then begin
      cover "a combine underflows" (flushes && (not complex) && plain.under);
      cover "a carry subnormal in float32" (carries && plain.under);
      cover "a narrow float's subnormal term" (carries && subnormal_term least c);
      cover "a complex64 result with nx.cpu's bits" (complex && not plain.under);
      cover "a complex64 computation underflows" (complex && plain.under)
    end;
    if not plain.under then equal same (bits w) (bits g)
    else if complex then
      equal (list string) [] (misses (verdicts b c (A.Any g)))
    else
      let (A.Any flushed) =
        associated_result { flush = true; under = false } c
      in
      equal same (bits flushed) (bits g)
  end

(* Scans past one chunk, along the innermost axis and along another. *)
let chunked_scans =
  [
    example ~scan:true Sum f32 [| 9000 |] [| 0 |];
    example ~scan:true Sum f32 [| 9000; 3 |] [| 0 |];
  ]

(* So do Logsumexp's and Moments' results, reduced or scanned, and Arg's. *)
(* A Logsumexp and a Moments of 300 float32 numbers of full precision,
   whose results the terms' order changes, which the draws reach about one
   case in 170. *)
let ordered_specials =
  let st = Random.State.make [| 3 |] in
  let x = vector_of D.Float32 (Array.init 300 (fun _ -> full st 23)) in
  List.map
    (fun r ->
      {
        scanned = false;
        stat = { r; axes = [| 0 |]; x; views = []; values = "example" };
      })
    Nx_kernel.[ Logsumexp; Moments ]

let law_cpu_special (b : Support.backend) (c : special) =
  lane_ordered b;
  let want = Support.cpu.around (fun () -> run_special Support.cpu c) in
  (* The values tell associations apart: nx.cpu's Logsumexp or Moments of the
     reversed terms differs, neither a NaN. *)
  let st = c.stat in
  let summed = match st.r with Logsumexp | Moments -> true | _ -> false in
  cover "a Logsumexp or Moments result the terms' order changes"
    ((not c.scanned) && summed
    &&
    let back =
      Support.cpu.around (fun () ->
          run_special Support.cpu
            { c with stat = { st with x = reversed_along st.axes st.x } })
    in
    Array.exists2
      (Array.exists2 (fun a b ->
           (not (is_nan a || is_nan b)) && bits_of a <> bits_of b))
      want back);
  let got = run_special b c in
  let s = shape_of c.stat.x in
  cover "a scan" c.scanned;
  cover "a reduction along rows of 128 terms or more"
    ((not c.scanned) && c.stat.axes = [| 1 |] && Array.length s = 2
     && s.(1) >= 128);
  Array.iteri
    (fun k w ->
      let bad = ref [] in
      Array.iteri
        (fun i t ->
          let g = got.(k).(i) in
          let ok = if is_nan t then is_nan g else bits_of g = bits_of t in
          if (not ok) && List.length !bad < 8 then
            bad :=
              Printf.sprintf "result %d: %Lx, expected %Lx" i (bits_of g)
                (bits_of t)
              :: !bad)
        w;
      equal ~msg:(Printf.sprintf "destination %d" k) (list string) []
        (List.rev !bad))
    want

(* The suite *)

(* Sums and products past 64 blocks of terms, and sums and products whose
   computation underflows, reduced and scanned along a fold's terms. *)
let cpu_bits_examples =
  [
    (* bfloat16 and float16 sums and a float16 product of terms of mixed
       magnitudes, whose results the terms' order changes, which the draws reach
       about one case in 25. *)
    mixed_case Sum (D.Any D.Bfloat16) [| 3; 300 |] [| 1 |];
    mixed_case Sum (D.Any D.Float16) [| 2; 100 |] [| 1 |];
    mixed_case Prod (D.Any D.Float16) [| 3; 8 |] [| 1 |];
    example Sum f32 [| 70_001 |] [| 0 |];
    example Prod f64 [| 2; 66_000 |] [| 1 |];
    (* A complex64 sum of numbers, which the draws reach one case in 9 where
       float64 is declined. *)
    example Sum (D.Any D.Complex64) [| 3; 7 |] [| 1 |];
    (* Running float32 products whose second, 2^-140, lies below the least
       normal number: nx.cpu keeps it, and its third is 2^-70; a library that
       flushes gives zeros. *)
    {
      scan = true;
      monoid = Prod;
      axes = [| 0 |];
      x = A.Any (A.of_array D.Float32 [| 3 |] [| 0x1p-70; 0x1p-70; 0x1p70 |]);
      views = [];
    };
    (* Running complex64 products, a product of whose parts, 1.5 2^-127, lies
       below the least normal number: nx.cpu gives an imaginary part of 1.375
       2^-125, a library that flushes 2^-125, and nx_metal.mli bounds
       neither. *)
    {
      scan = true;
      monoid = Prod;
      axes = [| 0 |];
      x =
        A.Any
          (A.of_array D.Complex64 [| 2; 2 |]
             Complex.
               [|
                 { re = 1.; im = 0x1p-60 };
                 { re = 0x1p-60; im = 1. };
                 { re = 0x1.8p-67; im = 0x1p-125 };
                 { re = 0x1p-125; im = 0x1.8p-67 };
               |]);
      views = [];
    };
    (* Running float8_e5m2 products of terms of 2^-16 to 1.5 2^-14, whose
       float32 carry falls below 2^-126 at the ninth, about 2^-130, then
       -infinity: nx.cpu keeps the carry and gives -infinity; a library that
       flushes it gives NaN. *)
    {
      scan = true;
      monoid = Prod;
      axes = [| 0 |];
      x =
        A.Any
          (A.of_array D.Float8_e5m2 [| 13 |]
             [|
               0x1.4p-14;
               0x1.4p-14;
               0x1p-14;
               0x1.8p-14;
               0x1p-16;
               0x1p-15;
               0x1.8p-14;
               0x1.8p-15;
               0x1p-16;
               Float.neg_infinity;
               0x1p-14;
               0x1p-16;
               Float.neg_infinity;
             |]);
      views = [];
    };
    (* A float8_e5m2 sum of two of 2^-16, a subnormal: 2^-15, kept; a library
       that flushed the terms would give 0. *)
    {
      scan = false;
      monoid = Sum;
      axes = [| 0 |];
      x = A.Any (A.of_array D.Float8_e5m2 [| 2 |] [| 0x1p-16; 0x1p-16 |]);
      views = [];
    };
    (* Running float8_e5m2 products of 2^-16, a subnormal, and 2^10: 2^-16 and
       2^-6, kept; a library that flushed the term would give zeros. *)
    {
      scan = true;
      monoid = Prod;
      axes = [| 0 |];
      x = A.Any (A.of_array D.Float8_e5m2 [| 2 |] [| 0x1p-16; 0x1p10 |]);
      views = [];
    };
  ]

let laws (d : Support.declared) =
  let run law x =
    let b = Support.opened d in
    b.around (fun () -> covering (law b) x)
  in
  let around f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  let dtype_name (D.Any dt) = D.name dt in
  let cpu = d.label = Support.cpu.name in
  group d.label
    ([
       group "reductions"
         [
           prop ~examples:extreme_examples
             "a Max or a Min is the extreme of its terms, -0 below +0, a NaN \
              where a term is, any and all on booleans"
             extremes (run law_extreme);
           cases ~id:fst "a Sum of -0 terms is +0, reduced or scanned"
             neg_zero_sums (around test_neg_zero_sum);
           cases ~id:dtype_name "an empty product is 1"
             D.[ Any Float32; Any Float16; Any Complex64; Any Int32; Any Uint8 ]
             (around test_empty_product);
           cases
             ~id:(fun (scan, d) ->
               dtype_name d ^ if scan then ", scanned" else ", reduced")
             "a complex product of one term is that term"
             (List.concat_map (fun d -> [ (false, d); (true, d) ]) complexes)
             (around test_one_term_product);
           prop "an integer sum or product wraps, in any order of its terms"
             integers (run law_wrap);
           prop ~count:300 ~examples:special_examples
             "Logsumexp, Moments and Arg give the special values their terms \
              make"
             special_case (run law_special);
         ];
       group "bounds"
         [
           prop ~examples:bound_examples
             "a float or complex sum or product is within its bound of the \
              exact one"
             bounded (run law_bound);
         ];
       group "scans"
         [
           prop ~examples:scan_examples
             "a scan of a prefix is the prefix of the scan" scans
             (run law_prefix);
           prop ~examples:exact_carries
             "an exact scan from a carry continues the scan whose terms it \
              carries"
             (carry_case Exact) (run law_carry);
           prop ~examples:inexact_carries
             "a float or complex sum or product from a carry is within the \
              bound of the carry's and the terms' reduction"
             (carry_case Inexact) (run law_carry_bound);
           test "a carry of the carrier is taken" (fun () ->
               around test_carrier_carry ());
           test
             "a NaN carry that is the scan's destination makes its result a NaN"
             (fun () -> around test_aliased_carry ());
         ];
       cases
         ~id:(fun (msg, _, _, _) -> msg)
         "a call that does not fit is named by reduce_refusal, scan_refusal or \
          Shape, is not Done, and writes nothing outside its destinations"
         loop_misfits (around check_misfit);
       test "a destination over its load's bytes is refused as Overlapping"
         (fun () -> around test_overlapping ());
       prop ~count:200 ~examples:programmed_examples
         "a reduction or a scan of a program has the bits of its outputs \
          mapped, then reduced or scanned"
         (Gen.with_pp pp_programmed programmed)
         (run law_programmed);
     ]
    @ (if not cpu then []
       else
         [
           prop ~count:200 ~examples:several_examples
             "a reduction of several reductions of a program has the bits of \
              each reduced alone"
             (Gen.with_pp pp_programmed several)
             (run law_several);
         ])
    @
    if cpu then []
    else
      [
        (* Each case runs every wide call of its seed, among them a reduction
           held over uint8 from a padded load, which asks for the most LDS a
           launch may. *)
        prop ~count:1
          "a reduction or a scan too wide for a launch has nx.cpu's bits"
          Kernels_cases.wide_seeds (fun seed ->
            List.iter (run law_wide) (wide_scans seed));
        prop ~examples:cpu_bits_examples
          "float sums and products have nx.cpu's bits" (associated ~scans:false)
          (run (law_cpu_bits ~scans:false));
        prop ~examples:chunked_scans
          "float sum and product scans in nx.cpu's chunks have nx.cpu's bits"
          (associated ~scans:true)
          (run (law_cpu_bits ~scans:true));
        prop ~count:300 ~examples:ordered_specials
          "Logsumexp, Moments and Arg have nx.cpu's bits" special_case
          (run law_cpu_special);
      ])

(* rig.nv *)

(* rig.nv loads the cubin CUDA loads, so a call through it has CUDA's bits: what
   it can get wrong, its launches, memory and order, shows in them. Each case's
   calls run through CUDA, then through rig.nv. A NaN agrees with any NaN:
   nx_cuda.mli leaves a NaN result's sign and payload unspecified. *)
let nv_laws (first : Support.declared) (nv : Support.declared) =
  let agree ?msg call =
    Elements.agree ?msg (Support.opened first) (Support.opened nv) call
  in
  group nv.label
    [
      group "bits are cuda's, any NaN for a NaN"
        [
          prop "of reductions and scans of every dtype" any_case (fun c ->
              let (D.Any dt) = dtype_of c.x in
              cover "a scan" c.scan;
              cover "a reduction" (not c.scan);
              cover "a float" (D.is D.Float dt);
              cover "a complex number" (D.is D.Complex dt);
              cover "an integer" (D.is D.Signed dt || D.is D.Unsigned dt);
              cover "a narrow element" (D.bits dt < 16);
              List.iter (fun m -> cover (monoid_name m) (c.monoid = m)) monoids;
              agree (fun b -> [| run b c |]));
          prop ~examples:long_programs
            "of programs' reductions and scans, over padded loads too"
            (Gen.with_pp pp_programmed programmed) (fun c ->
              cover "a scan" c.scanned;
              cover "padded loads" c.padded;
              cover "a long program" c.long;
              cover "several loads" (Array.length c.dts > 1);
              cover "a long axis" (Array.exists (fun e -> e > 60_000) c.shape);
              agree (fun b -> programmed_call b c (programmed_parts b c)));
          prop ~count:1 "of calls too wide for a launch"
            Kernels_cases.wide_seeds (fun seed ->
              List.iter
                (fun ((c : Kernels_cases.wide), scanned) ->
                  let scanned, call, _ = wide_call c scanned in
                  cover "past a launch's loads" (c.mode = Loads);
                  cover "past a launch's values held" (c.mode = Held);
                  cover "past a launch's reductions" (c.mode = Outputs);
                  cover "a scan" scanned;
                  agree ~msg:(wide_name c scanned) call)
                (wide_scans seed));
          prop "of Logsumexp, Moments and Arg" special_case (fun c ->
              cover "a scan" c.scanned;
              agree (fun b ->
                  let ys = run_special b c in
                  Array.map
                    (fun t ->
                      A.Any
                        (A.of_array D.Int64
                           (Iarray.of_array [| Array.length t |])
                           (Array.map bits_of t)))
                    ys));
          prop "of scans from a carry, nx.cpu's" (carry_case Inexact) (fun c ->
              let (Carried (dt, _)) = c.kind in
              covers_carry c;
              agree (fun b ->
                  let _, z, carry = carry_on b c in
                  [| carry_scan b c (D.Any dt) ~carry dt (along1 z c.p c.m) |]));
        ];
    ]

(* nx.amd *)

(* nx.amd keeps nx.cpu's lane order and chunks (nx_amd.mli), so each of
   its reductions and scans, of every monoid, dtype and view, has nx.cpu's
   bits, any NaN for a NaN: its plain paths' Max and Min of NaN, -0, the
   infinities and the integers' extremes among them. The examples reach
   each plain path with such terms: rows, rows of several groups an
   output, narrow columns, columns too tall for it, and scans in 16-byte
   accesses past a chunk and along a stride. *)
let amd_examples =
  [
    example ~specials:20 Max f32 [| 300; 64 |] [| 1 |];
    example ~specials:20 Min f64 [| 2; 70_000 |] [| 1 |];
    example ~specials:20 Sum (D.Any D.Bfloat16) [| 3000; 5 |] [| 0 |];
    example ~specials:20 Max (D.Any D.Int32) [| 20_000; 3 |] [| 0 |];
    example ~scan:true ~specials:20 Min (D.Any D.Uint32) [| 3; 9000 |] [| 1 |];
    example ~scan:true ~specials:20 Max (D.Any D.Int64) [| 5000; 3 |] [| 0 |];
  ]

let amd_laws (amd : Support.declared) =
  let called c (b : Support.backend) = b.around (fun () -> [| run b c |]) in
  group "amd against nx.cpu"
    [
      prop ~examples:amd_examples
        "reductions and scans of every monoid, dtype and view have nx.cpu's \
         bits, any NaN for a NaN"
        any_case
        (fun c ->
          let s = shape_of c.x and r = Array.length (shape_of c.x) in
          let along a = Array.length c.axes = 1 && c.axes.(0) = a in
          let (D.Any dt) = dtype_of c.x in
          cover "a Max or Min" (c.monoid = Max || c.monoid = Min);
          cover "a narrow float" (D.is D.Float dt && D.bits dt < 32);
          cover "a reduction along the last axis"
            ((not c.scan) && r > 0 && along (r - 1));
          cover "a reduction along the first of two"
            ((not c.scan) && r = 2 && along 0);
          cover "a reduction along the first of two past 16 blocks"
            ((not c.scan) && r = 2 && along 0 && s.(0) > 16 * 1024);
          cover "a scan past one chunk" (c.scan && s.(c.axes.(0)) > 4096);
          cover "a scan along the first of two" (c.scan && r = 2 && along 0);
          Elements.agree Support.cpu (Support.opened amd) (called c));
    ]

let () =
  let nv =
    match (Support.declared_cuda, Support.declared_nv) with
    | Some first, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  let amd = Option.to_list Support.declared_amd in
  exit
    (Windtrap.run "nx_kernel.reduce"
       (List.map laws
          (Support.declare Support.cpu
          :: List.filter_map Fun.id
               [ Support.declared_cuda; Support.declared_metal ])
       @ nv @ List.map laws amd @ List.map amd_laws amd))
