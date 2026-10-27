(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reductions and scans through every kernel library the host runs, with
   operands and results on its device: a float or complex sum or product,
   reduced, scanned or from a carry, within its bound of the exact one, a
   NaN or an infinity where its terms make one; the special values of
   Logsumexp, Moments and Arg, reduced and scanned; a scan of a prefix the
   prefix of the scan; an exact scan from a carry the scan it continues;
   and nothing written where declined. *)

open Windtrap
open Nx_array_gen
open Reduce_cases
module A = Nx_array
module D = Nx_array.Dtype
module Support = Nx_kernels_support

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

(* Bounded sums and products of each kind, reduced and scanned, a product
   past float32's range and a sum of NaNs and infinities. *)
let bound_examples =
  let c64 = D.Any D.Complex64 and c128 = D.Any D.Complex128 in
  [
    example Sum f32 [| 3; 7 |] [| 1 |];
    example Prod f64 [| 3; 4 |] [| 1 |];
    example Sum c128 [| 3; 7 |] [| 1 |];
    example ~scan:true Prod c64 [| 40 |] [| 0 |];
    example ~scan:true Sum f64 [| 9000 |] [| 0 |];
    example Prod f32 [| 5000 |] [| 0 |];
    example ~specials:20 Sum f32 [| 5000 |] [| 0 |];
  ]

(* Bounds *)

(* A float's precision: its unit roundoff, its largest finite number, the
   least magnitude a partial product may take for the references below to
   hold, and half its least subnormal number, the most a rounding into it
   moves a result below its normal numbers. *)
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
}

let fsum () =
  { n = 0; hi = 0.; lo = 0.; mag = 0.; nan = false; pinf = false; ninf = false }

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
    s.mag <- s.mag +. Float.abs x
  end

(* A sum of magnitudes below half the largest number keeps every partial
   sum of every association finite. *)
let fsum_verdict ?into p s g =
  if s.nan || (s.pinf && s.ninf) then want_nan g
  else if s.mag > p.big /. 2. then Barred
  else if s.pinf then want_special infinity g
  else if s.ninf then want_special neg_infinity g
  else
    let bound = (gamma p (s.n - 1) +. 0x1p-100) *. s.mag in
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
}

let fprod () =
  {
    n = 0; hi = 1.; lo = 0.; up = 0.; down = 0.; neg = false; nan = false;
    inf = false; zero = false;
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
      p.hi <- h
    end
    else if x = 0. then p.zero <- true
    else p.inf <- true
  end

let over pr up = up >= Float.log2 pr.big -. 2.

let under pr down = down <= Float.log2 pr.floor

(* A zero times an infinity is a NaN, so a product holds a zero only
   barring overflow and an infinity only barring underflow. *)
let fprod_verdict ?into pr p g =
  if p.nan || (p.inf && p.zero) then want_nan g
  else if p.inf then
    if under pr p.down then Barred
    else want_special (if p.neg then neg_infinity else infinity) g
  else if p.zero then
    if over pr p.up then Barred
    else if g = 0. then Special
    else Misses (Printf.sprintf "%h, expected a zero" g)
  else if over pr p.up || under pr p.down then Barred
  else
    let a = Float.abs (p.hi +. p.lo) in
    let bound = (gamma pr (p.n - 1) +. (0x1p-100 *. float_of_int p.n)) *. a in
    within ?into ~exact:(p.hi, p.lo) ~bound g

(* Double-double sums and products, within a few units of 2^-104. *)
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
}

let zprod () =
  {
    n = 0; re = (1., 0.); im = (0., 0.); up = 0.; down = 0.; nan = false;
    inf = false; zero = false;
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
    let re = dd_add (dd_mul p.re z.re) (neg (dd_mul p.im z.im)) in
    p.im <- dd_add (dd_mul p.re z.im) (dd_mul p.im z.re);
    p.re <- re
  end

(* Normwise within γ(2n - 2): each multiplication by Prog.Mul's fused parts
   errs by at most 2u of the product's modulus. Which parts an infinite
   factor makes NaN depends on the association, so such a product is
   barred. *)
let zprod_verdict pr p (g : Complex.t) =
  let gnan = Float.is_nan g.re || Float.is_nan g.im in
  let show () = Printf.sprintf "%h %h" g.re g.im in
  if p.nan then
    if gnan then Special else Misses (show () ^ ", expected a NaN part")
  else if p.inf then Barred
  else if p.zero then
    if over pr p.up then Barred
    else if g.re = 0. && g.im = 0. then Special
    else Misses (show () ^ ", expected a zero")
  else if over pr p.up || under pr p.down then Barred
  else
    let er = fst p.re +. snd p.re and ei = fst p.im +. snd p.im in
    let a = Float.hypot er ei in
    let bound =
      (gamma pr (2 * (p.n - 1)) +. (0x1p-100 *. float_of_int p.n)) *. a
    in
    let part g (h, l) = g -. h -. l in
    let d = Float.hypot (part g.re p.re) (part g.im p.im) in
    if d <= bound then Bounded
    else
      Misses
        (Printf.sprintf "%s is %h from %h %h, past %h" (show ()) d er ei bound)

(* A reference of a float or complex Sum or Prod in the carrier [pr],
   rounded into [into]: [push] takes the next term, [verdict] judges a
   result of the terms so far. A complex sum adds each part as a float
   sum. *)
type reference = { push : Complex.t -> unit; verdict : Complex.t -> verdict }

let reference ?into ~complex pr m =
  match (m, complex) with
  | Sum, false ->
      let s = fsum () in
      {
        push = (fun z -> fsum_add s z.re);
        verdict = (fun g -> fsum_verdict ?into pr s g.re);
      }
  | Sum, true ->
      let r = fsum () and i = fsum () in
      {
        push = (fun z -> fsum_add r z.re; fsum_add i z.im);
        verdict =
          (fun g -> worse (fsum_verdict pr r g.re) (fsum_verdict pr i g.im));
      }
  | Prod, false ->
      let p = fprod () in
      {
        push = (fun z -> fprod_mul p z.re);
        verdict = (fun g -> fprod_verdict ?into pr p g.re);
      }
  | Prod, true ->
      let p = zprod () in
      { push = zprod_mul p; verdict = zprod_verdict pr p }
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

(* A float sum of [n] terms, a reduction's or a scan's result, is within
   γ(n - 1) Σ|x| of the exact sum, a float product within a relative
   γ(n - 1), a complex one normwise within γ(2n - 2), a complex sum part by
   part, barring overflow and underflow; a NaN or an infinity is one where
   the terms make it. *)
let law_bound (b : Support.backend) (c : case) =
  let (A.Any x) = c.x in
  let complex = D.is D.Complex (A.dtype x) in
  let pr = precision (D.Any (A.dtype x)) in
  match run b c with
  | None -> cover "declined" true
  | Some y ->
      let ts = zvalues c.x and gs = zvalues y in
      let fresh () = reference ~complex pr c.monoid in
      let vs =
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
      in
      let has v = Array.mem v vs in
      let kind = kind_of ~complex c.monoid in
      List.iter
        (fun k -> cover (k ^ " within its bound") (k = kind && has Bounded))
        kinds;
      cover "a scan within its bound" (c.scan && has Bounded);
      cover "a NaN or an infinity the terms make" (has Special);
      cover "a product barred by overflow or underflow"
        (c.monoid = Prod && has Barred);
      let bad = ref [] in
      Array.iteri
        (fun i v ->
          match v with
          | Misses s when List.length !bad < 8 ->
              bad := Printf.sprintf "result %d: %s" i s :: !bad
          | _ -> ())
        vs;
      equal (list string) [] (List.rev !bad)

(* The scan of a prefix of the axis is the prefix of the scan. *)
let law_prefix (b : Support.backend) (c : case) =
  let s = shape_of c.x and a = c.axes.(0) in
  let m = s.(a) / 2 in
  let (A.Any x) = c.x in
  let cut =
    A.Any
      (Option.get
         (A.move
            (M.Slice
               (Iarray.of_array (Array.mapi
                  (fun i e ->
                    if i = a then { M.start = 0; count = m; step = 1 }
                    else { M.start = 0; count = e; step = 1 })
                  s)))
            x))
  in
  match (run b c, run b { c with x = cut }) with
  | Some (A.Any y), Some y' ->
      cover "a prefix past a chunk" (m > 4096);
      let prefix =
        A.Any
          (Option.get
             (A.move
                (M.Slice
                   (Iarray.of_array (Array.mapi
                      (fun i e ->
                        if i = a then { M.start = 0; count = m; step = 1 }
                        else { M.start = 0; count = e; step = 1 })
                      s)))
                y))
      in
      equal (list string) [] (differ (got prefix) y')
  | None, None -> cover "declined" true
  | _ -> failf "a scan and its prefix were not both computed"

(* A kernel that declines writes nothing. *)
(* CR: terms casts narrow floats and complex values to Int64, so writes
   can disappear (Float16 0.5 and 0.75 both become 0). Initialize dst
   with deterministic bytes, keeping spare packed bits zero, and compare
   its full buffer via Rig.Buffer.copy to host and blit_to_bytes. Use the
   same byte check in test_contract's refusal law. Keep the Float16 reduce
   example and add its nonempty scan counterpart. *)
let law_declined (b : Support.backend) (c : case) =
  let (A.Any x) = c.x in
  let ys = Iarray.of_array (result_shape c) in
  let dst = on b (A.Any (A.create Rig.host (A.dtype x) ys)) in
  let bits () = Array.map bits_of (terms (host dst)) in
  let before = bits () in
  match call b c dst with
  | A.Declined ->
      cover "declined" true;
      equal (array int64) before (bits ())
  | A.Done -> cover "computed" true
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Carries *)

(* A carry of a reduction that takes none, or of another shape or dtype than
   the carrier's, is refused. *)
let test_carry_refusals (b : Support.backend) () =
  let (module K : Nx_kernel.S) = b.kernels in
  let make dt s = A.Any (A.create b.device dt s) in
  let scan (type v s) (dt : (v, s) D.t) ?carry r dsts =
    let (A.Any x) = make dt [| 2; 3 |] in
    K.scan ?carry (identity (D.Any dt)) ~axis:1 (r, 0) ~dsts [| Plain x |]
  in
  let f32 s = make D.Float32 s in
  let refuses msg want got = equal ~msg Elements.answer want got in
  let carry = f32 [| 2; 1 |] in
  refuses "a Logsumexp's carry" A.Shape_mismatch
    (scan D.Float32 ~carry Logsumexp [| f32 [| 2; 3 |] |]);
  refuses "a Moments' carry" A.Shape_mismatch
    (scan D.Float32 ~carry Moments [| f32 [| 2; 3 |]; f32 [| 2; 3 |] |]);
  refuses "an Arg's carry" A.Shape_mismatch
    (scan D.Float32 ~carry (Arg Max)
       [| f32 [| 2; 3 |]; make D.Int64 [| 2; 3 |] |]);
  refuses "a carry of the destination's shape" A.Shape_mismatch
    (scan D.Float32 ~carry:(f32 [| 2; 3 |]) (Monoid Sum) [| f32 [| 2; 3 |] |]);
  refuses "a carry of one element" A.Shape_mismatch
    (scan D.Float32 ~carry:(f32 [| 1; 1 |]) (Monoid Sum) [| f32 [| 2; 3 |] |]);
  refuses "a carry of another dtype" A.Wrong_dtype
    (scan D.Float32 ~carry:(make D.Float64 [| 2; 1 |]) (Monoid Max)
       [| f32 [| 2; 3 |] |]);
  refuses "a float16 carry of float16 terms" A.Wrong_dtype
    (scan D.Float16 ~carry:(make D.Float16 [| 2; 1 |]) (Monoid Sum)
       [| make D.Float16 [| 2; 3 |] |]);
  refuses "a float32 carry of float16 terms" A.Done
    (scan D.Float16 ~carry (Monoid Sum) [| make D.Float16 [| 2; 3 |] |])

(* A Max's NaN carry that is its destination, a slice of one term, a
   number, makes the result a NaN, though the scan writes the carry's
   bytes. *)
let test_aliased_carry (b : Support.backend) () =
  let (module K : Nx_kernel.S) = b.kernels in
  let on_b x = if Rig.equal (A.device x) b.device then x else A.to_device b.device x in
  let f64 = Int64.float_of_bits in
  let y = on_b (A.of_array D.Float64 [| 1; 1 |] [| f64 0x7FF4_0000_0000_0001L |]) in
  let x = on_b (A.of_array D.Float64 [| 1; 1 |] [| 1. |]) in
  let answer =
    K.scan ~carry:(A.Any y) (identity (D.Any D.Float64)) ~axis:1
      (Monoid Max, 0) ~dsts:[| A.Any y |] [| Plain x |]
  in
  equal ~msg:"answer" Elements.answer A.Done answer;
  let y = if Rig.equal (A.device y) Rig.host then y else A.to_device Rig.host y in
  equal ~msg:"a NaN" bool true (Float.is_nan (A.to_array y).(0))

(* A float or complex Sum or Prod scan from a carry: each result within the
   bound of a reduction of the carry's element and the terms up to it, a
   NaN where a NaN carry or term reaches it. The carry is the kernels' scan
   of the slice's first [p] terms, in the carrier. *)
let law_carry_bound (b : Support.backend) c =
  let (Carried (dt, vs)) = c.kind in
  let n = c.p + c.m in
  let complex = D.is D.Complex dt in
  cover "a NaN carry made from numbers" (c.made && c.p > 1 && c.m > 0);
  cover "a NaN term after a carry" (c.nan_after && c.m > 0);
  cover "no term after a carry" (c.m = 0);
  cover "several slices" (c.outer * c.inner > 1);
  cover "a narrow float's carry" (D.bits dt = 16 && D.is D.Float dt);
  cover "a complex carry" complex;
  cover "a product's carry" (c.cm = Prod);
  cover "a program's terms" c.through;
  let z =
    on b (A.Any (A.of_array dt [| c.outer; n; c.inner |] (carry_terms vs c)))
  in
  let (D.Any ct as cd) = carrier (D.Any dt) in
  let carry =
    along1 (carry_scan b c (D.Any dt) ct (along1 z 0 c.p)) (c.p - 1) 1
  in
  let terms = along1 z c.p c.m in
  let got = carry_scan b c (D.Any dt) ~carry dt terms in
  let pr = precision cd in
  let into = if same cd (D.Any dt) then None else Some (precision (D.Any dt)) in
  let cs = zvalues carry and ts = zvalues terms and gs = zvalues got in
  let bad = ref [] and seen = ref [] in
  for o = 0 to c.outer - 1 do
    for i = 0 to c.inner - 1 do
      let r = reference ?into ~complex pr c.cm in
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
  let nan_carry = Array.exists nan cs in
  cover "a NaN carry's results" (nan_carry && c.m > 0);
  cover "a result within its bound" (List.mem Bounded !seen);
  equal (list string) [] (List.rev !bad)

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
     let* s = frequency [ (6, small); (1, long) ] in
     let s = if scanned && Array.length s = 0 then [| 3 |] else s in
     let rank = Array.length s in
     let* axes =
       if scanned then map (fun a -> [| a |]) (int_range 0 (rank - 1))
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

(* Each class of result the contract states, on a vector of [xs]. *)
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
      one Logsumexp D.Float32 [| 1.; inf; ninf |];
      one Logsumexp D.Float32 [| 1.; nan |];
      one Logsumexp D.Float64 [| 1.; 2. |];
      one ~scanned:true Logsumexp D.Float64 [| ninf; 2.; nan; inf |];
      one ~scanned:true Logsumexp D.Float32 [| ninf; inf; 1. |];
      empty Moments;
      one Moments D.Float64 [| 1.; nan; inf |];
      one Moments D.Float32 [| 1.; inf; 2. |];
      one Moments D.Float32 [| 1.; 2.; 4. |];
      one Moments D.Float64 [| inf; ninf; 0. |];
      one ~scanned:true Moments D.Float32 [| -0.; 1.; ninf; inf; nan |];
      one (Arg Max) D.Float32 [| -0.; 0.; nan; 1.; nan |];
      one (Arg Max) D.Float64 [| 1.; 3.; 3.; ninf |];
      one (Arg Max) D.Float32 [| -0.; 0.; -1. |];
      one ~scanned:true (Arg Min) D.Float64 [| 2.; -0.; 0.; -0.; ninf; nan |];
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
   terms; [None] if declined. *)
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
  let p = identity (dtype_of xa) and loads = [| plain_of xa |] in
  let answer =
    if c.scanned then K.scan p ~axis:st.axes.(0) (st.r, 0) ~dsts loads
    else
      K.reduce p ~shape:(shape_of xa) ~axes:(Iarray.of_array st.axes)
        [| (st.r, 0) |] ~dsts loads
  in
  match answer with
  | A.Done -> Some (Array.map (fun d -> terms (host d)) dsts)
  | A.Declined -> None
  | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a

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
  match run_special b c with
  | None -> cover "declined" true
  | Some got ->
      Array.iteri
        (fun k g ->
          let bad = ref [] in
          Array.iteri
            (fun i (_, ks) ->
              if (not (fits ks.(k) g.(i))) && List.length !bad < 8 then
                bad :=
                  Printf.sprintf "result %d: %Lx, expected %s" i (bits_of g.(i))
                    (show_klass ks.(k))
                  :: !bad)
            want;
          equal ~msg:(Printf.sprintf "destination %d" k) (list string) []
            (List.rev !bad))
        got

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop ~examples:bound_examples
        "a float or complex sum or product is within its bound of the exact \
         one"
        bounded (run (law_bound b));
      prop ~examples:scan_examples "a scan of a prefix is the prefix of the scan"
        scans (run (law_prefix b));
      prop ~examples:any_examples "a declined case writes nothing" any_case
        (run (law_declined b));
      prop "an exact scan from a carry continues the scan whose terms it \
            carries"
        (carry_case Exact) (run (law_carry ~exact:true b));
      prop
        "a float or complex sum or product from a carry is within the bound \
         of the carry's and the terms' reduction"
        (carry_case Inexact) (run (law_carry_bound b));
      prop ~count:300 ~examples:special_examples
        "Logsumexp, Moments and Arg give the special values their terms make"
        special_case (run (law_special b));
      test "a carry the scan cannot take is refused" (fun () ->
          b.around (test_carry_refusals b));
      test "an extreme's carry its destination aliases" (fun () ->
          b.around (test_aliased_carry b));
    ]

let () =
  exit (Windtrap.run "nx_kernel.reduce" (List.map laws Support.backends))
