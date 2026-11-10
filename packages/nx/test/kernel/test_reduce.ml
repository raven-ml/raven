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
module P = Nx_kernel.Prog
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
  let y = run b c in
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
  match
    K.scan ?carry (carry_prog c dt) ~axis:1 (Monoid c.cm, 0) ~dsts:[| y |]
      [| Plain x |]
  with
  | A.Done -> y
  | r -> failf "scan: the kernels answered %a" Nx_array_support.pp_answer r

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

(* An exact scan from the carry of a slice's first [p] terms has the bits
   of the whole slice's scan from term [p] on, a NaN as any NaN. *)
let law_carry (b : Support.backend) c =
  let (Carried (dt, _)) = c.kind in
  covers_carry c;
  cover "a NaN term before an extreme's carry" (c.nan_before && c.m > 0);
  let z, carry = carry_of b c in
  let got = carry_scan b c (D.Any dt) ~carry dt (along1 z c.p c.m) in
  let want = along1 (carry_scan b c (D.Any dt) dt z) c.p c.m in
  equal (array (Elements.same (D.Any dt)))
    (Elements.elements (host want))
    (Elements.elements (host got))

(* A float or complex Sum or Prod scan from a carry: each result within the
   bound of a reduction of the carry's element and the terms up to it, a
   NaN where a NaN carry or term reaches it. *)
let law_carry_bound (b : Support.backend) c =
  let (Carried (dt, _)) = c.kind in
  let complex = D.is D.Complex dt in
  covers_carry c;
  cover "a NaN carry made from numbers" (c.made && c.p > 1 && c.m > 0);
  cover "a NaN carry and a NaN term after it"
    (c.made && c.p > 1 && c.nan_after && c.m > 0);
  cover "a complex carry" complex;
  cover "a product's carry" (c.cm = Prod);
  let z, carry = carry_of b c in
  let (D.Any _ as cd) = Nx_kernel.carrier (D.Any dt) in
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

(* Reductions and scans that do not fit their kernel, whatever the
   program computes: by other numbers of loads, destinations or
   reductions, an output past the program's, a coordinate past the loop's
   rank, an empty loop's included, a load of another dtype than its
   operand, a reduction of a dtype it does not take, an Arg's position
   other than int64, or a carry of a reduction that takes none, of another
   shape or of another dtype than the carrier. Each with its misfit's
   class, the refusal nx's rules name it by, the destinations' dtypes and
   shapes, and the call into them. *)
let loop_misfits (b : Support.backend) =
  let (module K : Nx_kernel.S) = b.kernels in
  let f32 = D.Any D.Float32 and i64 = D.Any D.Int64 in
  let id = identity f32 in
  let coord = P.v ~ins:[||] [| P.Coord 2 |] ~outs:[| 0 |] in
  let x ?(dt = D.Any D.Float32) s =
    let (D.Any dt) = dt in
    Nx_kernel.Plain (A.create b.device dt s)
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
    let call g ds =
      K.reduce p ~shape:(Iarray.of_array shape) ~axes:[| 1 |]
        (Iarray.of_array rs) ~dsts:(Iarray.of_array ds)
        (Iarray.map (Misfit.load g) (Iarray.of_array loads))
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
    let carry =
      Option.map
        (fun (D.Any d, s) -> A.Any (A.create b.device d (Iarray.of_array s)))
        carry
    in
    let call (g : Misfit.guard) ds =
      let carry = Option.map (fun (A.Any c) -> A.Any (g.g c)) carry in
      K.scan ?carry p ~axis:1 (r, k) ~dsts:(Iarray.of_array ds)
        (Iarray.map (Misfit.load g) (Iarray.of_array loads))
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
  Misfit.check ~msg ~what b dsts call

(* A Max's NaN carry that is its destination, a slice of one term, a
   number, makes the result a NaN, though the scan writes the carry's
   bytes. *)
let test_aliased_carry (b : Support.backend) () =
  let (module K : Nx_kernel.S) = b.kernels in
  let f64 = Int64.float_of_bits in
  let y = A.expect D.Float64 (on b (A.Any (A.of_array D.Float64 [| 1; 1 |] [| f64 0x7FF4_0000_0000_0001L |]))) in
  let x = on b (A.Any (A.of_array D.Float64 [| 1; 1 |] [| 1. |])) in
  let answer =
    K.scan ~carry:(A.Any y) (identity (D.Any D.Float64)) ~axis:1
      (Monoid Max, 0) ~dsts:[| A.Any y |] [| plain_of x |]
  in
  equal ~msg:"answer" Elements.answer A.Done answer;
  let y = A.expect D.Float64 (host (A.Any y)) in
  equal ~msg:"a NaN" bool true (Float.is_nan (A.to_array y).(0))

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
  match answer with
  | A.Done -> Array.map (fun d -> terms (host d)) (Iarray.to_array dsts)
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
  let got = run_special b c in
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

(* Programs reduced and scanned *)

(* A reduction along [axes], or a scan along [axes.(0)], of a program drawn
   from [seed] over a loop of [shape], its operands of [dts], its first
   load padded where [padded], of hundreds of nodes where [long]. *)
type programmed = {
  scanned : bool;
  summed : bool;
  shape : int array;
  axes : int array;
  dts : D.any array;
  padded : bool;
  long : bool;
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
    c.seed

(* Every dtype a program reads: the narrow floats and integers and [bit]
   among them. *)
let narrows =
  base @ complexes
  @ D.
      [
        Any Float16; Any Bfloat16; Any Float8_e4m3fn; Any Float8_e5m2;
        Any Float4_e2m1fn; Any Int4; Any Uint4; Any Bit;
      ]

(* Loops of small extents, rows, and one or two rows past 64 blocks of
   terms, the long axis reduced or scanned. *)
let programmed =
  let open Gen in
  let small = array ~size:(int_range 1 4) (int_range 1 5) in
  let rows =
    let* r = int_range 1 40 in
    let+ k = of_list ~pp:Format.pp_print_int [ 3; 64; 1025; 5000 ] in
    [| r; k |]
  in
  let long =
    let* n = int_range 66_000 140_000 in
    let+ r = int_range 1 2 in
    if r = 1 then [| n |] else [| r; n |]
  in
  let* scanned = bool in
  let* summed = if scanned then bool else constant false in
  let* shape = frequency [ (3, small); (3, rows); (2, long) ] in
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
  { scanned; summed; shape; axes; dts; padded; long; seed }

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

let programmed_done what = function
  | A.Done -> ()
  | r -> failf "%s answered %a" what Nx_array_support.pp_answer r

(* [c]'s reduction or scan of [p] over [loads] into fresh destinations on
   [b]'s device. *)
let programmed_call (b : Support.backend) c (p, _, loads, rs, ys) =
  let module K = (val b.kernels) in
  let dsts = programmed_dsts b p rs ys in
  let ds = Iarray.of_array dsts and loads = Iarray.of_array loads in
  programmed_done "the call"
    (if c.scanned then K.scan p ~axis:c.axes.(0) rs.(0) ~dsts:ds loads
     else
       K.reduce p ~shape:(Iarray.of_array c.shape)
         ~axes:(Iarray.of_array c.axes) (Iarray.of_array rs) ~dsts:ds loads);
  dsts

let law_programmed (b : Support.backend) c =
  let module K = (val b.kernels) in
  let ((p, types, loads, rs, ys) as parts) = programmed_parts b c in
  let call p ~dsts loads =
    let dsts = Iarray.of_array dsts and loads = Iarray.of_array loads in
    if c.scanned then K.scan p ~axis:c.axes.(0) rs.(0) ~dsts loads
    else
      K.reduce p ~shape:(Iarray.of_array c.shape)
        ~axes:(Iarray.of_array c.axes) (Iarray.of_array rs) ~dsts loads
  in
  let done_ = programmed_done in
  let got = programmed_call b c parts in
  let outs = Iarray.to_array p.outs in
  let mapped =
    Array.map
      (fun o ->
        let (D.Any dt) = types.(o) in
        A.Any (A.create b.device dt (Iarray.of_array c.shape)))
      outs
  in
  done_ "the map"
    (K.map p ~dsts:(Iarray.of_array mapped) (Iarray.of_array loads));
  let identities =
    P.v ~ins:(Iarray.of_array (Array.map (fun o -> types.(o)) outs))
      (Iarray.init (Array.length outs) (fun k -> P.In k))
      ~outs:(Iarray.init (Array.length outs) Fun.id)
  in
  let want = programmed_dsts b p rs ys in
  done_ "the call of the mapped outputs"
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

(* Calls too wide for a launch *)

let wide_reductions : Nx_kernel.reduction list =
  [ Monoid Sum; Monoid Max; Arg Min; Logsumexp; Moments; Monoid Prod ]

(* A reduction or a scan too wide for one of nx.cuda's launches, by its
   loads, its values held at once or its reductions, has nx.cpu's bits,
   any NaN for a NaN, in a library that states nx.cpu's association; a
   scan in one that states its own, the bits of its output mapped, then
   scanned. A scan scans along the innermost axis of more than one
   element; a reduction reduces axes 0 and 1. *)
let wide_call (c : Kernels_cases.wide) scanned =
  let p, n = Kernels_cases.wide_program c in
  let shape = Kernels_cases.wide_shape c in
  let scanned = scanned && c.mode <> Outputs in
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
    equal Kernels_cases.answer A.Done answer;
    dsts
  in
  let mapped (b : Support.backend) =
    let module K = (val b.kernels) in
    let loads, _ = Kernels_cases.wide_loads b c n in
    let dt = P.dtype p (Iarray.get p.outs 0) in
    let (D.Any d) = dt in
    let y = A.Any (A.create b.device d (Iarray.of_array shape)) in
    equal Kernels_cases.answer A.Done
      (K.map p ~dsts:[| y |] (Iarray.of_array loads));
    let id = identity dt in
    let dsts = programmed_dsts b id rs ys in
    equal Kernels_cases.answer A.Done
      (K.scan id ~axis rs.(0) ~dsts:(Iarray.of_array dsts) [| plain_of y |]);
    dsts
  in
  (scanned, call, mapped)

let law_wide (b : Support.backend) ((c : Kernels_cases.wide), scanned) =
  let scanned, call, mapped = wide_call c scanned in
  let want =
    if scanned && not b.scan_order then mapped b
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
        ~msg:(Printf.sprintf "destination %d" k)
        (Kernels_cases.same_values (D.Any (A.dtype w)))
        (Kernels_cases.bits_of (Support.host w))
        (Kernels_cases.bits_of (Support.host g)))
    want

(* nx.cpu's bits *)

(* Float and complex sums and products of every float dtype, narrow ones
   included, reduced, and scanned where [scans]. *)
let associated ~scans =
  Gen.with_pp pp_case
    (case_of
       ~scan:(if scans then Gen.bool else Gen.constant false)
       ~monoids:[ Sum; Prod ]
       (D.
          [
            Any Float16; Any Bfloat16; Any Float8_e4m3fn; Any Float8_e5m2;
            Any Float4_e2m1fn;
          ]
       @ floats @ complexes))

(* A library that keeps nx.cpu's lane order and chunks gives nx.cpu's
   bits: a float or complex sum's or product's results, reduced or scanned,
   are nx.cpu's, any NaN for a NaN. *)
let law_cpu_bits (b : Support.backend) (c : case) =
  if b.scan_order then cover "a scan" c.scan;
  let terms = total (shape_of c.x) / Int.max 1 (total (result_shape c)) in
  cover "past one block of terms" (terms > 1024);
  cover "past 64 blocks, several groups an output"
    ((not c.scan) && terms > 64 * 1024);
  let (A.Any w) = Support.cpu.around (fun () -> run Support.cpu c) in
  let (A.Any g) = run b c in
  equal
    (Kernels_cases.same_values (D.Any (A.dtype w)))
    (Kernels_cases.bits_of w) (Kernels_cases.bits_of g)

(* So do Logsumexp's and Moments' results, reduced or scanned, and Arg's. *)
let law_cpu_special (b : Support.backend) (c : special) =
  let want = Support.cpu.around (fun () -> run_special Support.cpu c) in
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

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    ([
      prop ~examples:bound_examples
        "a float or complex sum or product is within its bound of the exact \
         one"
        bounded (run (law_bound b));
      prop ~examples:scan_examples "a scan of a prefix is the prefix of the scan"
        scans (run (law_prefix b));
      prop ~count:300 ~examples:special_examples
        "Logsumexp, Moments and Arg give the special values their terms make"
        special_case (run (law_special b));
      cases
        ~id:(fun (msg, _, _, _) -> msg)
        "a reduction or scan that does not fit is not Done and writes \
         nothing outside its destinations; nx's rules name its refusal"
        (loop_misfits b) (check_misfit b);
      prop
        "an exact scan from a carry continues the scan whose terms it carries"
        (carry_case Exact) (run (law_carry b));
      prop
        "a float or complex sum or product from a carry is within the bound \
         of the carry's and the terms' reduction"
        (carry_case Inexact) (run (law_carry_bound b));
      test "a carry of the carrier is taken" (fun () ->
          b.around (test_carrier_carry b));
      test "an extreme's carry its destination aliases" (fun () ->
          b.around (test_aliased_carry b));
    ]
    @ (if not b.lane_order then []
       else
         [
           prop ~count:200
             "a reduction or a scan of a program has the bits of its outputs \
              mapped, then reduced or scanned"
             (Gen.with_pp pp_programmed programmed)
             (run (law_programmed b));
         ])
    @
    if b == Support.cpu || not b.lane_order then []
    else
      [
        prop ~count:45
          "a reduction or a scan too wide for a launch has nx.cpu's bits"
          (Gen.pair (Gen.with_pp Kernels_cases.pp_wide Kernels_cases.wide_cases)
             Gen.bool)
          (run (law_wide b));
        prop
          ~examples:
            [
              example Sum f32 [| 70_001 |] [| 0 |];
              example Prod f64 [| 2; 66_000 |] [| 1 |];
            ]
          "float sums and products have nx.cpu's bits"
          (associated ~scans:b.scan_order)
          (run (law_cpu_bits b));
        prop ~count:300
          "Logsumexp, Moments and Arg have nx.cpu's bits" special_case
          (run (law_cpu_special b));
      ])

(* rig.nv *)

(* rig.nv loads the cubin CUDA loads, so a call through it has CUDA's
   bits: what it can get wrong, its launches, memory and order, shows in
   them. Each case's calls run through CUDA, then through rig.nv. A NaN
   agrees with any NaN: nx_cuda.mli leaves a NaN result's sign and
   payload unspecified. *)
let nv_laws (first : Support.backend) (nv : Support.backend) =
  let agree call = Elements.agree first nv call in
  group nv.name
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
          prop "of programs' reductions and scans, over padded loads too"
            (Gen.with_pp pp_programmed programmed) (fun c ->
              cover "a scan" c.scanned;
              cover "padded loads" c.padded;
              cover "a long program" c.long;
              cover "several loads" (Array.length c.dts > 1);
              cover "a long axis" (Array.exists (fun e -> e > 60_000) c.shape);
              agree (fun b -> programmed_call b c (programmed_parts b c)));
          prop ~count:45 "of calls too wide for a launch"
            (Gen.pair (Gen.with_pp Kernels_cases.pp_wide Kernels_cases.wide_cases)
               Gen.bool) (fun ((c : Kernels_cases.wide), scanned) ->
              let scanned, call, _ = wide_call c scanned in
              cover "past a launch's loads" (c.mode = Loads);
              cover "past a launch's values held" (c.mode = Held);
              cover "past a launch's reductions" (c.mode = Outputs);
              cover "a scan" scanned;
              agree call);
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
          prop "of scans from a carry" (carry_case Inexact) (fun c ->
              let (Carried (dt, _)) = c.kind in
              covers_carry c;
              agree (fun b ->
                  let z, carry = carry_of b c in
                  [| carry_scan b c (D.Any dt) ~carry dt (along1 z c.p c.m) |]));
        ];
    ]

let () =
  let nv =
    match (Support.gpus (), Support.nv ()) with
    | first :: _, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.reduce"
       (List.map laws (Support.backends ()) @ nv))
