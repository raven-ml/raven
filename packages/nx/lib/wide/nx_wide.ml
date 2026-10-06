(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Double-word numbers

   A number is [hi + lo] with [hi] that sum rounded to nearest. The arithmetic
   is Joldes, Muller and Popescu's ("Tight and rigorous error bounds for basic
   building blocks of double-word arithmetic", ACM TOMS 44(2), 2017), with the
   bounds Muller and Rideau proved in Coq (ACM TOMS 48(1), 2022). Each bound
   assumes every sum and product rounded once, as written, and no underflow or
   overflow.

   A value is normalised words, or words a structure rebuilt, whose
   normalisation waits for the first operation that reads them: a rebuild may
   hold values a transformation traces outside its interpretation, on which
   nothing can be computed yet. *)

type 'b words = { hi : (float, 'b) Nx.t; lo : (float, 'b) Nx.t }
type 'b t = Normal of 'b words | Pending of 'b words
type 'b number = 'b t

let check_dtype fn (type b) (dt : (float, b) Nx.dtype) =
  match dt with
  | Nx_dtype.Float32 | Nx_dtype.Float64 -> ()
  | Nx_dtype.Float16 | Nx_dtype.BFloat16 | Nx_dtype.Float8_e4m3
  | Nx_dtype.Float8_e5m2 ->
      invalid_arg
        (Printf.sprintf "Nx_wide.%s: dtype %s, expected float32 or float64" fn
           (Nx_dtype.to_string dt))

(* Error-free transformations: each pair's sum is exactly the operation's
   result. [fast_two_sum a b] needs [a]'s exponent at least [b]'s, which [|a| >=
   |b|] ensures. *)

let two_sum a b =
  let s = Nx.add a b in
  let a' = Nx.sub s b in
  let b' = Nx.sub s a' in
  (s, Nx.add (Nx.sub a a') (Nx.sub b b'))

let fast_two_sum a b =
  let s = Nx.add a b in
  (s, Nx.sub b (Nx.sub s a))

let two_prod a b =
  let p = Nx.mul a b in
  (p, Nx.fma a b (Nx.neg p))

(* A result whose words are not both finite is [plain], the float result of the
   high words, with a zero low word: an operand or the result is infinite or
   NaN, and the algorithm's error terms are then meaningless. A zero result is
   exact; it is [plain] when that is a zero, which gives it the sign floats do
   ([x - x] is [+0], [-0 * x] is [-0]), and [+0] otherwise. *)
let settle plain (zh, zl) =
  let zero = Nx.zeros_like zl in
  let ok = Nx.logical_and (Nx.isfinite zh) (Nx.isfinite zl) in
  let nonzero = Nx.not_equal zh zero in
  let fallback =
    Nx.where
      (Nx.logical_and (Nx.logical_not nonzero) (Nx.not_equal plain zero))
      zero plain
  in
  let ok = Nx.logical_and ok nonzero in
  { hi = Nx.where ok zh fallback; lo = Nx.where ok zl zero }

(* [normalise hi lo] is the normalised pair of [hi + lo]: a pair with a zero low
   word is kept as given, its zeros' signs with it, and every other normalised
   pair is returned unchanged by the exact two-sum. *)
let normalise hi lo =
  let ((s, _) as sum) = two_sum hi lo in
  let w = settle s sum in
  let kept = Nx.equal lo (Nx.zeros_like lo) in
  { hi = Nx.where kept hi w.hi; lo = Nx.where kept lo w.lo }

let words = function Normal w -> w | Pending w -> normalise w.hi w.lo

let v ?lo hi =
  check_dtype "v" (Nx.dtype hi);
  match lo with
  | None -> Normal { hi; lo = Nx.zeros_like hi }
  | Some lo -> (
      match Nx.broadcasted hi lo with
      | hi, lo -> Normal (normalise hi lo)
      | exception Invalid_argument _ ->
          let shape x =
            String.concat "; "
              (Array.to_list (Array.map string_of_int (Nx.shape x)))
          in
          invalid_arg
            (Printf.sprintf "Nx_wide.v: shapes [%s] and [%s] do not broadcast"
               (shape hi) (shape lo)))

let hi w = (words w).hi
let lo w = (words w).lo

(* AccurateDWPlusDW (their Algorithm 6): within [3u² / (1 - 4u)], a bound Muller
   and Rideau show is reached asymptotically. A sum of a number and its negation
   is [+0]: every two-sum of it is exact and zero. *)
let accurate x y =
  let sh, sl = two_sum x.hi y.hi in
  let th, tl = two_sum x.lo y.lo in
  let vh, vl = fast_two_sum sh (Nx.add sl th) in
  (sh, fast_two_sum vh (Nx.add tl vl))

let sum_words x y =
  let sh, z = accurate x y in
  settle sh z

let add x y = Normal (sum_words (words x) (words y))

let sub x y =
  let y = words y in
  Normal (sum_words (words x) { hi = Nx.neg y.hi; lo = Nx.neg y.lo })

(* DWTimesDW3 (their Algorithm 12), every partial product formed: within [(4u² +
   u³/2) / (1 + u)² < 4u²] for a precision of at least 5 bits, Muller and
   Rideau's Theorem 2.8, which tightens the original [5u²]. With zero low words
   it is [two_prod] of the high words, exact. *)
let product x y =
  let ch, cl1 = two_prod x.hi y.hi in
  let tl1 = Nx.fma x.hi y.lo (Nx.mul x.lo y.lo) in
  let cl2 = Nx.fma x.lo y.hi tl1 in
  fast_two_sum ch (Nx.add cl1 cl2)

let mul x y =
  let x = words x and y = words y in
  Normal (settle (Nx.mul x.hi y.hi) (product x y))

(* DWDivDW3 (their Algorithm 18): one Newton step refines [th = 1 / yh] to the
   double word [m] near [1 / y], and the quotient is [x m], within [9.8u²]. [1 -
   yh th] is exact, since [th] is [1 / yh] rounded. The step is DWTimesFP3 of [e
   = 1 - y th] by [th] (their Algorithm 9) and DWPlusFP of that and [th] (their
   Algorithm 4); the product is Algorithm 12, the one their proof bounds. *)
let div x y =
  let x = words x and y = words y in
  let th = Nx.recip y.hi in
  let rh = Nx.fma (Nx.neg y.hi) th (Nx.ones_like th) in
  let eh, el = fast_two_sum rh (Nx.neg (Nx.mul y.lo th)) in
  let ch, cl = two_prod eh th in
  let dh, dl = fast_two_sum ch (Nx.fma el th cl) in
  let sh, sl = two_sum dh th in
  let mh, ml = fast_two_sum sh (Nx.add dl sl) in
  Normal (settle (Nx.div x.hi y.hi) (product x { hi = mh; lo = ml }))

(* Where [hi] is not an integer, [|lo| <= ulp hi / 2] keeps every integer out of
   reach and [floor w] is [floor hi]. Where it is, [floor w] is [hi + floor lo],
   whose fast two-sum is exact: [|floor lo| <= |hi|]. A zero floor is [-0] only
   for the number [-0], whose high word is [-0]; an infinite or NaN [hi] is its
   own floor. *)
let floor w =
  let w = words w in
  let zero = Nx.zeros_like w.lo in
  let fh = Nx.floor w.hi in
  let integral = Nx.equal fh w.hi in
  let s, t = fast_two_sum w.hi (Nx.floor w.lo) in
  let zh = Nx.where integral s fh and zl = Nx.where integral t zero in
  let zh = Nx.where (Nx.equal zh zero) (Nx.mul w.hi zero) zh in
  let finite = Nx.isfinite w.hi in
  Normal { hi = Nx.where finite zh w.hi; lo = Nx.where finite zl zero }

(* A number's normalised pair is unique and rounding is monotone, so the pairs
   order as the numbers do, high words first. *)
let less a b =
  let a = words a and b = words b in
  Nx.logical_or (Nx.less a.hi b.hi)
    (Nx.logical_and (Nx.equal a.hi b.hi) (Nx.less a.lo b.lo))

let equal a b =
  let a = words a and b = words b in
  Nx.logical_and (Nx.equal a.hi b.hi) (Nx.equal a.lo b.lo)

(* Sums *)

(* [w] summed along its last axis, of [2^levels] numbers, by halving: element
   [i] of the first half is added to element [i] of the second, until one is
   left. The tree has [levels] levels and depends only on the shape.

   One fused tree over every number grows with the count and does not compile at
   a million numbers, so each [chunk_levels] levels run as one fused program on
   rows of [2^chunk_levels] numbers, then are stored. A compiled sum is then
   [⌈levels / chunk_levels⌉] programs. Three levels compile fastest: a cold
   compile of a sum of 10^6 float64 numbers took 5.1 s at one level a program,
   2.3 to 3.4 s at two, 2.7 s at three, 5.5 s at four and 10 s at five, where
   each program costs about 0.15 s and each fused double-word addition about 70
   ms. *)
let chunk_levels = 3

let rec halve w levels =
  if levels = 0 then w
  else
    let c = min chunk_levels levels in
    let shape = Nx.shape w.hi in
    let r = Array.length shape in
    let rows =
      Array.append
        (Array.sub shape 0 (r - 1))
        [| shape.(r - 1) lsr c; 1 lsl c |]
    in
    let rec fold w width =
      if width = 1 then w
      else
        let half = width / 2 in
        let cut lo hi x =
          Nx.shrink
            (Array.mapi
               (fun d len -> if d = r then (lo, hi) else (0, len))
               (Nx.shape x))
            x
        in
        let part lo hi = { hi = cut lo hi w.hi; lo = cut lo hi w.lo } in
        let _, (hi, lo) = accurate (part 0 half) (part half width) in
        fold { hi; lo } half
    in
    let s =
      fold { hi = Nx.reshape rows w.hi; lo = Nx.reshape rows w.lo } (1 lsl c)
    in
    let out = Array.sub rows 0 r in
    halve
      { hi = Nx.copy (Nx.reshape out s.hi); lo = Nx.copy (Nx.reshape out s.lo) }
      (levels - c)

let sum ?axes w =
  let w = words w in
  let shape = Nx.shape w.hi in
  let r = Array.length shape in
  let axes =
    match axes with
    | None -> List.init r Fun.id
    | Some axes ->
        List.sort_uniq Int.compare
          (List.map
             (fun a ->
               let a' = if a < 0 then a + r else a in
               if a' < 0 || a' >= r then
                 invalid_arg
                   (Printf.sprintf
                      "Nx_wide.sum: axis %d out of bounds for rank %d" a r);
               a')
             axes)
  in
  let kept =
    List.filter (fun d -> not (List.mem d axes)) (List.init r Fun.id)
  in
  let out = Array.of_list (List.map (fun d -> shape.(d)) kept) in
  let n = List.fold_left (fun n d -> n * shape.(d)) 1 axes in
  if n = 0 then
    let zero = Nx.sum ~axes w.hi in
    Normal { hi = zero; lo = zero }
  else
    (* The summed axes, last and flattened into one. *)
    let flat x =
      Nx.reshape (Array.append out [| n |]) (Nx.transpose ~axes:(kept @ axes) x)
    in
    (* Zeros pad the count to a power of two: adding a zero is exact. *)
    let rec log2_ceil k = if 1 lsl k >= n then k else log2_ceil (k + 1) in
    let levels = log2_ceil 0 in
    let padded x =
      let x = flat x in
      let extra = (1 lsl levels) - n in
      if extra = 0 then x
      else
        Nx.pad
          (Array.mapi
             (fun d _ -> if d = Array.length out then (0, extra) else (0, 0))
             (Nx.shape x))
          0. x
    in
    let s = halve { hi = padded w.hi; lo = padded w.lo } levels in
    Normal
      (settle (Nx.sum ~axes w.hi) (Nx.reshape out s.hi, Nx.reshape out s.lo))

(* Structures *)

(* A walk that returns both words as they were, as every reading walk does
   (flattening, folding, saving), keeps the value. Any other rebuilds it from
   new words, whose normalisation waits for the first operation, since the words
   may be values a transformation traces outside its interpretation. *)
let ptree (type b) (dtype : (float, b) Nx.dtype) : b t Nx.Ptree.t =
  check_dtype "ptree" dtype;
  let module S = struct
    type _ t = b number

    let walk c t =
      let w = match t with Normal w | Pending w -> w in
      let open Nx.Ptree.Walk in
      let hi = field c "hi" tensor w.hi in
      let lo = field c "lo" tensor w.lo in
      if hi == w.hi && lo == w.lo then t else Pending { hi; lo }
  end in
  Nx.Ptree.instantiate (module S)
