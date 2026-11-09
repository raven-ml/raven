(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's attention against its definition: the softmax of each row's visible
   scaled scores, rewritten by its score, times the values, over dense and
   paged keys, grouped heads, lead axes, lengths, masks as programs and as
   captured arrays, empty sequences, and float16, bfloat16, float32 and
   float64 accumulating in float32 and in float64, each within the error
   bound nx.mli states. The definition computes in double-double arithmetic
   (below), within a bound of its own that the comparison adds. A hidden
   value, finite or not, never changes the result on nx.cpu. *)

open Windtrap
module D = Nx_array.Dtype

let same_bits =
  array
    (Testable.make
       ~pp:(fun ppf x -> Format.fprintf ppf "%h" x)
       ~equal:(fun a b ->
         (Float.is_nan a && Float.is_nan b)
         || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)))

let invalid f = raises_match (Exn.invalid_arg ~sub:"Nx.attention") f
let numel s = Array.fold_left ( * ) 1 s

(* Float32's unit roundoff, and [x] rounded to float32. *)
let u32 = 0x1p-24
let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* An accumulator: its dtype, its unit roundoff [u], its least positive
   subnormal and nx.cpu's [λ], [Log] erring by a relative [e{^λu} - 1]
   (nx_cpu.mli). *)
type 'a accum = {
  acc : (float, 'a) D.t;
  u : float;
  least : float;
  lambda : float;
}

let in32 = { acc = D.Float32; u = u32; least = 0x1p-149; lambda = 3. }
let in64 = { acc = D.Float64; u = 0x1p-53; least = 0x1p-1074; lambda = 5. }

(* Double-word numbers: [hi + lo] unevaluated, [|lo|] at most half an ulp
   of [hi]. [two_sum] and the product by [Float.fma] are error-free (Knuth;
   Dekker 1971, "A floating-point technique for extending the available
   precision"). [add] is the sloppy double-word sum: its absolute error is
   about [u² (|x| + |y|)], [u] float64's, and its relative error is
   unbounded under cancellation (Joldes, Muller and Popescu 2017, "Tight
   and rigorous error bounds for basic building blocks of double-word
   arithmetic"). Every bound below is at least [u] times the magnitudes
   added, so the definition neglects that error. *)
type dd = { hi : float; lo : float }

let two_sum a b =
  let s = a +. b in
  let bb = s -. a in
  { hi = s; lo = (a -. (s -. bb)) +. (b -. bb) }

let quick_sum a b =
  let s = a +. b in
  { hi = s; lo = b -. (s -. a) }

let dd x = { hi = x; lo = 0. }

let add x y =
  let s = two_sum x.hi y.hi in
  quick_sum s.hi (s.lo +. x.lo +. y.lo)

let neg x = { hi = -.x.hi; lo = -.x.lo }

let mul_f x b =
  let p = x.hi *. b in
  quick_sum p (Float.fma x.hi b (-.p) +. (x.lo *. b))

(* [exp] and [log] of a double-word number: the libm's [exp] and [log] in
   float64 at [hi], corrected to first order by [lo], so no more precise
   than float64. The definition assumes the libm within one ulp, as glibc's
   [exp] and [log] are: [exp x] within [3u] relative, [log x] within
   [2u |log x|] and [x]'s own relative error. *)
let exp_dd x =
  let e = Float.exp x.hi in
  e +. (e *. x.lo)

let log_dd x = two_sum (Float.log x.hi) (x.lo /. x.hi)

(* A case *)

(* [Striped] hides keys [64, 128) and every key of rows [1 mod 3]: whole
   blocks and whole rows. [Head_window w] is causal within [w.(h)] keys of
   the row at head [h], from a captured [[hq; 1; 1]] of int32;
   [Documents (dq, dk)] shows a key to a row of the same document, from
   captured ids of the rows and of the keys; [Spans (wide, a, b)] shows
   row [i] the keys [[a.(i), b.(i))], captured as [[tq; 1]] of int16, or
   of uint8 unless [wide]; and [Limit l] shows head [h] the keys up to
   [l.(h)], captured as [[hq; 1; 1]] of int4, for at most 8 keys. *)
type mask =
  | Unmasked
  | Causal
  | Window
  | Striped
  | Captured of bool array
  | Head_window of int array
  | Documents of int array * int array
  | Spans of bool * int array * int array
  | Limit of int array
type score = Identity | Alibi of float array | Bias of float array

type case = {
  lead : int array;
  hk : int;
  groups : int;
  tq : int;
  tk : int;
  dk : int;
  dv : int;
  lengths : int array option;  (** One per lead index. *)
  scale : float option;
  mask : mask;  (** A captured mask is [[hq; tq; tk]]. *)
  score : score;
      (** ALiBi's slopes, one per query head, or a bias [[hq; tq; tk]]. *)
  q : float array;
  k : float array;
  v : float array;
}

let hq c = c.hk * c.groups
let rows c = numel c.lead

let pp_case ppf c =
  let ints ppf a =
    Format.fprintf ppf "[%s]"
      (String.concat "; " (Array.to_list (Array.map string_of_int a)))
  in
  Format.fprintf ppf
    "lead %a hk %d groups %d tq %d tk %d dk %d dv %d lengths %s %s %s" ints
    c.lead c.hk c.groups c.tq c.tk c.dk c.dv
    (match c.lengths with None -> "none" | Some l -> Format.asprintf "%a" ints l)
    (match c.mask with
    | Unmasked -> "unmasked"
    | Causal -> "causal"
    | Window -> "window"
    | Striped -> "striped"
    | Captured _ -> "captured"
    | Head_window _ -> "a window per head"
    | Documents _ -> "documents"
    | Spans _ -> "spans"
    | Limit _ -> "a limit per head")
    (match c.score with
    | Identity -> "identity"
    | Alibi _ -> "alibi"
    | Bias _ -> "bias");
  let floats name xs =
    Format.fprintf ppf "@ %s [%s]" name
      (String.concat "; " (Array.to_list (Array.map (Printf.sprintf "%h") xs)))
  in
  Option.iter (floats "scale") (Option.map (fun s -> [| s |]) c.scale);
  floats "q" c.q;
  floats "k" c.k;
  floats "v" c.v;
  (match c.mask with
  | Head_window w -> Format.fprintf ppf "@ windows %a" ints w
  | Documents (dq, dk) ->
      Format.fprintf ppf "@ row documents %a@ key documents %a" ints dq ints dk
  | Spans (wide, a, b) ->
      Format.fprintf ppf "@ spans%s from %a@ to %a"
        (if wide then " of int16" else "")
        ints a ints b
  | Limit l -> Format.fprintf ppf "@ limits %a" ints l
  | Unmasked | Causal | Window | Striped | Captured _ -> ());
  match c.score with
  | Identity -> ()
  | Alibi xs -> floats "slopes" xs
  | Bias xs -> floats "bias" xs

let element =
  Gen.frequency
    [ (8, Gen.float_range (-2.) 2.); (1, Gen.constant 0.); (1, Gen.constant (-0.)) ]

(* Documents of [tq] rows and [tk] keys, in order or not: ids of up to 4
   documents, or of 2 for keys of several blocks. *)
let documents tq tk =
  let open Gen in
  let* sorted = bool in
  let n = if tk > 64 then 1 else 3 in
  let ids k = map (if sorted then fun a -> Array.sort compare a; a else Fun.id)
      (array ~size:(constant k) (int_range 0 n)) in
  let+ dq = ids tq and+ dk = ids tk in
  Documents (dq, dk)

(* Spans of keys for [tq] rows over [tk] keys. *)
let spans tq tk =
  let open Gen in
  let* wide = if tk > 254 then constant true else bool in
  let span = pair (int_range 0 tk) (int_range 0 (tk + 1)) in
  let+ ss = array ~size:(constant tq) span in
  Spans (wide, Array.map fst ss, Array.map (fun (a, b) -> max a b) ss)

(* A case of the extents [lead], [hk], [groups], [tq], [tk], [dk] and [dv]
   draw. *)
let sized ~lead ~hk ~groups ~tq ~tk ~dk ~dv =
  let open Gen in
  let* lead = lead in
  let* hk = hk in
  let* groups = groups in
  let* tq, tk = Gen.pair (tq groups) tk in
  let* dk = dk in
  let* dv = dv in
  let* lengths =
    frequency
      [
        (1, constant None);
        (1, map Option.some (array ~size:(constant (numel lead)) (int_range (-1) (tk + 1))));
      ]
  in
  let* scale = of_list [ None; Some 0.7; Some (-1.5) ] in
  let hq = hk * groups in
  let limit =
    if tk > 8 then []
    else
      let limits = array ~size:(constant hq) (int_range 0 7) in
      [ (1, map (fun l -> Limit l) limits) ]
  in
  let* mask =
    frequency
    @@ limit
    @ [
        (1, constant Unmasked);
        (1, constant Causal);
        (1, constant Window);
        (1, constant Striped);
        (1, map (fun m -> Captured m) (array ~size:(constant (hq * tq * tk)) bool));
        ( 1,
          map
            (fun w -> Head_window w)
            (array ~size:(constant hq) (int_range 0 (tq + 2))) );
        (1, documents tq tk);
        (1, spans tq tk);
      ]
  in
  let* score =
    frequency
      [
        (2, constant Identity);
        ( 1,
          map
            (fun s -> Alibi s)
            (array ~size:(constant hq) (map round32 (float_range 0. 0.5))) );
      ]
  in
  let n = numel lead in
  let* q = array ~size:(constant (n * hq * tq * dk)) element in
  let* k = array ~size:(constant (n * hk * tk * dk)) element in
  let+ v = array ~size:(constant (n * hk * tk * dv)) element in
  { lead; hk; groups; tq; tk; dk; dv; lengths; scale; mask; score; q; k; v }

let case =
  let open Gen in
  sized
    ~lead:(array ~size:(int_range 0 2) (int_range 0 2))
    ~hk:(int_range 1 2) ~groups:(int_range 1 3)
    ~tq:(fun _ -> frequency [ (1, constant 0); (4, int_range 1 4) ])
    ~tk:(frequency [ (1, constant 0); (4, int_range 1 5) ])
    ~dk:(int_range 1 4) ~dv:(int_range 1 3)

(* Cases of several blocks of keys and several units of rows (nx_cpu.mli):
   keys past a block of 64 with a partial last block, a group's rows past a
   unit's 64, and a group of more heads than a unit holds. *)
let long =
  let open Gen in
  sized
    ~lead:(frequency [ (3, constant [||]); (1, constant [| 2 |]) ])
    ~hk:(int_range 1 2)
    ~groups:(frequency [ (4, int_range 1 3); (1, constant 65) ])
    ~tq:(fun g -> if g > 3 then int_range 1 2 else int_range 1 40)
    ~tk:(frequency [ (1, int_range 1 64); (3, int_range 65 200) ])
    ~dk:(int_range 1 5) ~dv:(int_range 1 4)

(* The bound *)

let gamma n v = Float.of_int n *. v /. (1. -. (Float.of_int n *. v))

(* The bound on a visible score's error in the accumulator [ac], nx.mli's
   [η] for one key: [a] is [|scale| Σ |q k|] and [s] the rewritten score.
   Without [score] it is [(2dk + 4) u a] to first order; ALiBi adds the
   rounding of its product and of its difference, a bias that of its sum.
   Products below the accumulator's normal range round absolutely. *)
let score_error ac c h i j ~a s =
  let u = ac.u in
  let dot =
    (a *. (((1. +. u) *. (1. +. u) *. (1. +. gamma (c.dk + 1) (2. *. u))) -. 1.))
    +. (Float.of_int (c.dk + 2) *. ac.least)
  in
  match c.score with
  | Identity -> dot
  | Alibi slopes ->
      ((dot +. (u *. Float.abs (Float.of_int (i - j) *. slopes.(h)))) *. (1. +. u))
      +. (u *. Float.abs s)
  | Bias _ -> (dot *. (1. +. u)) +. (u *. Float.abs s)

(* nx.cpu's bound for attention, nx.mli's [ε], of a row of [t] keys whose
   [lse] is [r]:
   [e{^λu} (13n + 9 + ln t) u' + (e{^λu} - 1) ln t + u' |r|] for [n] blocks
   of 64 keys, [u' = u / (1 - u)]. *)
let logsumexp_bound ac t r =
  let n = (t + 63) / 64 and ln = Float.log (Float.of_int t) in
  let u' = ac.u /. (1. -. ac.u) and grow = Float.exp (ac.lambda *. ac.u) in
  (grow *. (Float.of_int ((13 * n) + 9) +. ln) *. u')
  +. ((grow -. 1.) *. ln)
  +. (u' *. Float.abs r)

(* The definition's own bound, to first order in float64's [u], for a row of
   [t] keys: each [e{^(s - m)}] within [3u] relative, so [l] too; [lse]
   within [(5 + 2 ln t) u], [log]'s ulp and [l]'s error; each weight
   [e{^(s - lse)}] within [(8 + 2 ln t) u] relative, [lse]'s error and
   [exp]'s; [o] within that times [M], and [u |o|] for its rounding. *)
let u64 = 0x1p-53
let reference_lse t = (5. +. (2. *. Float.log (Float.of_int t))) *. u64

let reference_o t ~mass ~o =
  ((8. +. (2. *. Float.log (Float.of_int t))) *. u64 *. mass)
  +. (u64 *. Float.abs o)

(* The definition *)

(* The row [b]'s length: keys at or past it are hidden. *)
let length c b =
  match c.lengths with None -> c.tk | Some l -> max 0 (min l.(b) c.tk)

let visible c b h i j =
  let len = length c b in
  let causal = j <= i + (len - c.tq) in
  j < len
  &&
  match c.mask with
  | Unmasked -> true
  | Causal -> causal
  | Window -> causal && i < j + 2
  | Striped -> j / 64 mod 2 = 0 && i mod 3 <> 1
  | Captured m -> m.((((h * c.tq) + i) * c.tk) + j)
  | Head_window w -> causal && i - j < w.(h)
  | Documents (dq, dk) -> dq.(i) = dk.(j)
  | Spans (_, a, b) -> a.(i) <= j && j < b.(i)
  | Limit l -> j <= l.(h)

(* [s] rewritten by [c]'s score. ALiBi's product is exact: a slope is a
   float32 and [i - j] a small integer. *)
let rewritten c h i j s =
  match c.score with
  | Identity -> s
  | Alibi slopes -> add s (dd (-.(Float.of_int (i - j) *. slopes.(h))))
  | Bias bias -> add s (dd bias.((((h * c.tq) + i) * c.tk) + j))

(* What a case reaches, at a row that has a visible key: a finite and a
   non-finite value hidden from it within its length, and a NaN value at a
   visible key whose weight is zero. *)
type reached = {
  mutable hidden_finite : bool;
  mutable hidden_nonfinite : bool;
  mutable zero_weight_nan : bool;
}

(* The exact result and what nx.mli's bound reads of it. *)
type exact = {
  o : float array;
  mass : float array;  (** [M = Σ p |v|] at each element of [o]. *)
  zero : bool array;  (** Whether each product of [o]'s sum is zero. *)
  vmax : float array;  (** The greatest finite visible [|v|] at each. *)
  lse : dd array;
  eta : float array;  (** [η] at each row. *)
  size : float array;  (** [b], the greatest visible [|s|], at each row. *)
}

(* Per row [(b, h, i)], the exact result in the accumulator [ac]; then what
   the case reaches. [key b kh j d] and [value b kh j d] read key [j] of key
   head [kh]. Weights are the accumulator's: one below half its least
   subnormal is zero. A value hidden from a row is not read, nx_cpu.mli's
   promise. A non-finite value gives [o] its sum's special value. *)
let definition ac c ~visible ~key ~value q =
  let hq = hq c in
  let scale =
    match c.scale with
    | Some s -> s
    | None -> 1. /. Float.sqrt (Float.of_int c.dk)
  in
  let n = rows c * hq * c.tq in
  let e =
    {
      o = Array.make (n * c.dv) 0.;
      mass = Array.make (n * c.dv) 0.;
      zero = Array.make (n * c.dv) true;
      vmax = Array.make (n * c.dv) 0.;
      lse = Array.make n (dd neg_infinity);
      eta = Array.make n 0.;
      size = Array.make n 0.;
    }
  in
  let sums = Array.make (n * c.dv) (dd 0.) in
  let reached =
    { hidden_finite = false; hidden_nonfinite = false; zero_weight_nan = false }
  in
  for b = 0 to rows c - 1 do
    for h = 0 to hq - 1 do
      let kh = h / c.groups in
      for i = 0 to c.tq - 1 do
        let row = (((b * hq) + h) * c.tq) + i in
        let seen j = visible b h i j in
        let s =
          Array.init c.tk (fun j ->
              let dot = ref (dd 0.) and size = ref 0. in
              for d = 0 to c.dk - 1 do
                let x = q.((row * c.dk) + d) and y = key b kh j d in
                dot := add !dot (mul_f (dd x) y);
                size := !size +. Float.abs (x *. y)
              done;
              let s = rewritten c h i j (mul_f !dot scale) in
              if seen j then begin
                let a = Float.abs scale *. !size and s = s.hi +. s.lo in
                e.eta.(row) <- Float.max e.eta.(row) (score_error ac c h i j ~a s);
                e.size.(row) <- Float.max e.size.(row) (Float.abs s)
              end;
              s)
        in
        let m = ref neg_infinity in
        Array.iteri (fun j x -> if seen j then m := Float.max !m x.hi) s;
        if !m > neg_infinity then begin
          let sum = ref (dd 0.) in
          Array.iteri
            (fun j x ->
              if seen j then sum := add !sum (dd (exp_dd (add x (dd (-. !m))))))
            s;
          let lse = add (dd !m) (log_dd !sum) in
          e.lse.(row) <- lse;
          Array.iteri
            (fun j x ->
              let p = exp_dd (add x (neg lse)) in
              let p = if p < ac.least /. 2. then 0. else p in
              for d = 0 to c.dv - 1 do
                let y = value b kh j d and at = (row * c.dv) + d in
                if not (seen j) then begin
                  if j < length c b then
                    if Float.is_finite y then reached.hidden_finite <- true
                    else reached.hidden_nonfinite <- true
                end
                else begin
                  if p = 0. && Float.is_nan y then reached.zero_weight_nan <- true;
                  e.o.(at) <- e.o.(at) +. (p *. y);
                  sums.(at) <- add sums.(at) (mul_f (dd p) y);
                  if p <> 0. && y <> 0. then e.zero.(at) <- false;
                  if Float.is_finite y then
                    e.vmax.(at) <- Float.max e.vmax.(at) (Float.abs y);
                  if p > 0. then e.mass.(at) <- e.mass.(at) +. (p *. Float.abs y)
                end
              done)
            s
        end
      done
    done
  done;
  Array.iteri
    (fun at x -> if Float.is_finite x then e.o.(at) <- sums.(at).hi +. sums.(at).lo)
    e.o;
  (e, reached)

(* Nx's call *)

let ints s xs = Nx.create Nx.int64 s (Array.map Int64.of_int xs)

let mask_of c : (Nx.host Nx.pair -> Nx.host Nx.bool_t) option =
  match c.mask with
  | Unmasked -> None
  | Causal -> Some Nx.causal
  | Window ->
      Some (fun p -> Nx.(logical_and (causal p) (less p.q (add_s p.k 2L))))
  | Striped ->
      Some
        (fun p ->
          Nx.(
            logical_and
              (equal_s (mod_s (div_s p.k 64L) 2L) 0L)
              (not_equal_s (mod_s p.q 3L) 1L)))
  | Captured m -> Some (fun _ -> Nx.create Nx.bool [| hq c; c.tq; c.tk |] m)
  | Head_window w ->
      let w = Nx.create Nx.int32 [| hq c; 1; 1 |] (Array.map Int32.of_int w) in
      Some
        (fun p ->
          Nx.(logical_and (causal p) (less (cast int32 (sub p.q p.k)) w)))
  | Documents (dq, dk) ->
      let dq = Nx.create Nx.int16 [| c.tq; 1 |] dq in
      let dk = Nx.create Nx.int16 [| 1; c.tk |] dk in
      Some (fun _ -> Nx.equal dq dk)
  | Spans (wide, a, b) ->
      let span (type e) (dt : (int, e) Nx.dtype) =
        let a = Nx.create dt [| c.tq; 1 |] a in
        let b = Nx.create dt [| c.tq; 1 |] b in
        fun (p : _ Nx.pair) ->
          let k = Nx.cast dt p.k in
          Nx.(logical_and (less_equal a k) (less k b))
      in
      Some (if wide then span Nx.int16 else span Nx.uint8)
  | Limit l ->
      let l = Nx.create Nx.int4 [| hq c; 1; 1 |] l in
      Some (fun p -> Nx.less_equal (Nx.cast Nx.int4 p.k) l)

(* [c]'s score in the accumulator [acc]. *)
let score_of (type a) (acc : (float, a) D.t) c :
    (Nx.host Nx.pair -> (float, a, Nx.host) Nx.t -> (float, a, Nx.host) Nx.t)
    option =
  match c.score with
  | Identity -> None
  | Alibi slopes ->
      let slopes = Nx.create acc [| hq c; 1; 1 |] slopes in
      Some (fun p s -> Nx.(Infix.(s - (cast acc (p.q - p.k) * slopes))))
  | Bias bias ->
      let bias = Nx.create acc [| hq c; c.tq; c.tk |] bias in
      Some (fun _ s -> Nx.add s bias)

let lengths_of c = Option.map (ints c.lead) c.lengths

(* [o] and [lse] against the definition, within nx.mli's bound in the
   accumulator [ac] and the definition's own: [lse] within [η + ε], and
   each element of [o] within
   [u{_q} |o| + (1 + u{_q}) (e{^w} - 1 + γ(tk + 1, 2u) e{^w}) M] for
   [w = (1 + u) (2η + ε) + (2b + ln tk + 5) u], [uq] being [u{_q}].
   Underflow adds [3] least subnormals of the accumulator for each weight,
   times [|v|], and one for each product, and [tiny], [q]'s least
   subnormal, for the rounding of [o]. *)
let agrees ac c ~uq ~tiny (o, lse) (e : exact) =
  let o = Nx.to_array o and lse = Nx.to_array lse in
  let u = ac.u in
  equal ~msg:"o's elements" int (Array.length e.o) (Array.length o);
  equal ~msg:"lse's elements" int (Array.length e.lse) (Array.length lse);
  Array.iteri
    (fun row (l' : dd) ->
      let l = lse.(row) in
      if l'.hi = neg_infinity then begin
        if l <> neg_infinity then failf "lse at %d: %h, by definition -inf" row l
      end
      else begin
        let eta = e.eta.(row) and eps = logsumexp_bound ac c.tk l'.hi in
        if not (Float.abs (l -. l'.hi -. l'.lo) <= eta +. eps +. reference_lse c.tk)
        then failf "lse at %d: %h, by definition %h + %h" row l l'.hi l'.lo;
        let tk = Float.of_int c.tk in
        let w =
          ((1. +. u) *. ((2. *. eta) +. eps))
          +. (((2. *. e.size.(row)) +. Float.log tk) *. u)
          +. (5. *. u /. (1. -. u))
        in
        let weight = Float.expm1 w in
        for d = 0 to c.dv - 1 do
          let i = (row * c.dv) + d in
          let x = o.(i) and x' = e.o.(i) and m = e.mass.(i) in
          let sum = (weight *. m) +. (gamma (c.tk + 1) (2. *. u) *. (1. +. weight) *. m) in
          let under = (tk *. ((3. *. e.vmax.(i)) +. 1.) *. ac.least) +. tiny in
          let tol =
            sum +. (uq *. (Float.abs x' +. sum)) +. under
            +. reference_o c.tk ~mass:m ~o:x'
          in
          let agree =
            if Float.is_nan x' then Float.is_nan x
            else if Float.is_finite x' && e.zero.(i) then
              (* An exact zero sum is +0. *)
              Int64.equal (Int64.bits_of_float x) 0L
            else if Float.is_finite x' then Float.abs (x -. x') <= tol
            else x = x'
          in
          if not agree then
            failf "o at %d: %h, by definition %h, beyond the bound %h" i x x' tol
        done
      end)
    e.lse

let shape c tail = Array.append c.lead tail

(* The dense call of [c] at [dt] in the accumulator [ac] against the
   definition; what [c] reaches. *)
let dense (type s a) (dt : (float, s) D.t) uq tiny (ac : a accum) c =
  let hq = hq c in
  let q = Nx.create dt (shape c [| hq; c.tq; c.dk |]) c.q in
  let k = Nx.create dt (shape c [| c.hk; c.tk; c.dk |]) c.k in
  let v = Nx.create dt (shape c [| c.hk; c.tk; c.dv |]) c.v in
  let qs = Nx.to_array q and ks = Nx.to_array k and vs = Nx.to_array v in
  let key b kh j d = ks.((((((b * c.hk) + kh) * c.tk) + j) * c.dk) + d) in
  let value b kh j d = vs.((((((b * c.hk) + kh) * c.tk) + j) * c.dv) + d) in
  let result =
    Nx.attention ?scale:c.scale ?lengths:(lengths_of c) ?mask:(mask_of c)
      ?score:(score_of ac.acc c) ~acc:ac.acc ~kv:(Nx.Dense { k; v }) q
  in
  equal ~msg:"o's shape" (array int) (shape c [| hq; c.tq; c.dv |]) (Nx.shape (fst result));
  equal ~msg:"lse's shape" (array int) (shape c [| hq; c.tq |]) (Nx.shape (snd result));
  let expected, reached = definition ac c ~visible:(visible c) ~key ~value qs in
  agrees ac c ~uq ~tiny result expected;
  reached

let law_dense dt u tiny ac c =
  cover "causal" (c.mask = Causal);
  cover "unmasked" (c.mask = Unmasked);
  cover "a window" (c.mask = Window);
  cover "a captured mask" (match c.mask with Captured _ -> true | _ -> false);
  cover "a window per head"
    (match c.mask with Head_window _ -> true | _ -> false);
  cover "documents" (match c.mask with Documents _ -> true | _ -> false);
  cover "spans" (match c.mask with Spans _ -> true | _ -> false);
  cover "a limit per head" (match c.mask with Limit _ -> true | _ -> false);
  cover "alibi" (c.score <> Identity);
  cover "lengths" (c.lengths <> None);
  cover "grouped heads" (c.groups > 1);
  cover "lead axes" (Array.length c.lead > 0);
  cover "no query" (c.tq = 0 && rows c > 0);
  cover "no key" (c.tk = 0 && c.tq > 0 && rows c > 0);
  cover "a row with no visible key"
    (rows c > 0 && c.tq > 0
    && not (List.exists (visible c 0 0 0) (List.init c.tk Fun.id)));
  ignore (dense dt u tiny ac c : reached)

(* Scores of up to a few thousand from keys near one base key, so a row's
   visible scores lie within a few units of each other, or tie: each weight
   is far from zero while each score rounds by thousands of [u]. Keys have
   one or two dimensions, where a score's error comes nearest its bound.
   Zero values let [o] rest on keys of small weight, whose relative error is
   the largest. *)
let tied =
  let open Gen in
  let* c = case in
  let* dk = int_range 1 2 in
  let n = numel c.lead in
  let large = float_range (-32.) 32. in
  let* base = array ~size:(constant dk) large in
  let* q = array ~size:(constant (n * hq c * c.tq * dk)) large in
  let* near =
    array ~size:(constant (n * c.hk * c.tk * dk))
      (frequency [ (1, constant 0.); (3, float_range (-0.1) 0.1) ])
  in
  let+ v =
    array ~size:(constant (Array.length c.v))
      (frequency
         [ (2, constant 0.); (2, constant 1.); (1, constant (-1.)); (1, element) ])
  in
  let k = Array.mapi (fun i x -> base.(i mod dk) +. x) near in
  { c with dk; q; k; v; score = Identity }

(* Coverage of what [long] draws: blocks, units and masks that hide whole
   blocks or rows. *)
let law_long dt u tiny ac c =
  cover "several blocks" (c.tk > 64);
  cover "a partial last block" (c.tk > 64 && c.tk mod 64 <> 0);
  cover "several units of a group" (c.groups * c.tq > 64);
  cover "a group past a unit" (c.groups > 64);
  cover "whole blocks and rows hidden" (c.mask = Striped && c.tk > 64);
  cover "lengths within a block" (c.lengths <> None && c.tk > 64);
  cover "a window per head of a group"
    (match c.mask with Head_window _ -> c.groups > 1 | _ -> false);
  cover "documents over several blocks"
    (match c.mask with Documents _ -> c.tk > 64 | _ -> false);
  cover "spans over several blocks"
    (match c.mask with Spans _ -> c.tk > 64 | _ -> false);
  ignore (dense dt u tiny ac c : reached)

let law_tied dt u tiny ac c =
  cover "masked" (c.mask <> Unmasked || c.lengths <> None);
  cover "one key" (c.tk = 1 && c.tq > 0 && rows c > 0);
  cover "no key" (c.tk = 0 && c.tq > 0 && rows c > 0);
  ignore (dense dt u tiny ac c : reached)

(* Values with NaN, infinities and [-0.], and a bias of [0] or [-200], whose
   weights underflow to zero at float32; at float64, a bias of [-800]
   does. *)
let unusual_of ?(under = -200.) cases =
  let open Gen in
  let* c = cases in
  let* v =
    array ~size:(constant (Array.length c.v))
      (frequency
         [
           (4, element);
           (1, constant nan);
           (1, constant infinity);
           (1, constant neg_infinity);
           (1, constant (-0.));
         ])
  in
  let+ bias = array ~size:(constant (hq c * c.tq * c.tk)) (of_list [ 0.; under ]) in
  let mask = if c.mask = Unmasked then Causal else c.mask in
  { c with v; mask; score = Bias bias }

let unusual = unusual_of case

(* A value hidden from a row never changes its [o]; a NaN at a visible key
   whose weight is zero makes it NaN. *)
let law_unusual dt u tiny ac c =
  let reached = dense dt u tiny ac c in
  cover "a finite value hidden from a row" reached.hidden_finite;
  cover "a non-finite value hidden from a row" reached.hidden_nonfinite;
  cover "a NaN at a visible key of zero weight" reached.zero_weight_nan

(* Paged keys: the case's keys and values are the pools, [[pages; hk; page;
   d]], read through a table of entries in [[-1, pages]]. *)

let paged_of ~pages ~page ~max_pages cases =
  let open Gen in
  let* c = cases in
  let* pages = pages in
  let* page = page in
  let* max_pages = max_pages in
  let* table =
    array ~size:(constant (numel c.lead * max_pages)) (int_range (-1) pages)
  in
  let tk = max_pages * page in
  let* k = array ~size:(constant (pages * c.hk * page * c.dk)) element in
  let* v = array ~size:(constant (pages * c.hk * page * c.dv)) element in
  let* lengths =
    frequency
      [
        (1, constant None);
        (1, map Option.some (array ~size:(constant (numel c.lead)) (int_range (-1) (tk + 1))));
      ]
  in
  let+ mask =
    frequency
      [
        ( 2,
          constant
            (match c.mask with
            | Captured _ | Documents _ | Spans _ | Limit _ -> Causal
            | m -> m) );
        (1, map (fun m -> Captured m) (array ~size:(constant (hq c * c.tq * tk)) bool));
      ]
  in
  ({ c with tk; k; v; lengths; mask }, (pages, page, max_pages, table))

let paged =
  let open Gen in
  paged_of ~pages:(int_range 1 3) ~page:(int_range 1 3)
    ~max_pages:(int_range 0 3) case

(* Pages within, across and as whole blocks of 64 keys. *)
let paged_long =
  let open Gen in
  paged_of ~pages:(int_range 1 4) ~page:(of_list [ 16; 48; 64 ])
    ~max_pages:(int_range 1 5) long

let pp_paged ppf (c, (pages, page, _, table)) =
  Format.fprintf ppf "%a pages %d page %d table [%s]" pp_case c pages page
    (String.concat "; " (Array.to_list (Array.map string_of_int table)))

let law_paged (type s a) ?(long = false) (dt : (float, s) D.t) uq tiny
    (ac : a accum) (c, (pages, page, max_pages, table)) =
  let hq = hq c in
  if long then cover "a page across a block's end" (page = 48 && max_pages > 1);
  let q = Nx.create dt (shape c [| hq; c.tq; c.dk |]) c.q in
  let k = Nx.create dt [| pages; c.hk; page; c.dk |] c.k in
  let v = Nx.create dt [| pages; c.hk; page; c.dv |] c.v in
  let entry b j = table.((b * max_pages) + (j / page)) in
  let in_range b j = entry b j >= 0 && entry b j < pages in
  cover "an entry out of range" (Array.exists (fun e -> e < 0 || e >= pages) table);
  cover "an entry in range" (Array.exists (fun e -> e >= 0 && e < pages) table);
  let qs = Nx.to_array q and ks = Nx.to_array k and vs = Nx.to_array v in
  (* An entry out of range reads zero, never seen. *)
  let pool xs d b kh j i =
    if not (in_range b j) then 0.
    else
      let p = entry b j in
      xs.((((((p * c.hk) + kh) * page) + (j mod page)) * d) + i)
  in
  let visible b h i j = in_range b j && visible c b h i j in
  let result =
    Nx.attention ?scale:c.scale ?lengths:(lengths_of c) ?mask:(mask_of c)
      ?score:(score_of ac.acc c) ~acc:ac.acc
      ~kv:(Nx.Paged { k; v; table = ints (shape c [| max_pages |]) table })
      q
  in
  agrees ac c ~uq ~tiny result
    (fst (definition ac c ~visible ~key:(pool ks c.dk) ~value:(pool vs c.dv) qs))

(* Cases *)

let f32 s xs = Nx.create Nx.float32 s xs

(* Past the length, and where the causal mask hides a pair, a NaN key, value
   and score bias reach neither [o] nor [lse]. *)
let test_hidden_nan () =
  let q = f32 [| 1; 2; 1 |] [| 1.; -1. |] in
  let k = f32 [| 1; 3; 1 |] [| 0.5; 2.; nan |] in
  let v = f32 [| 1; 3; 1 |] [| 3.; -1.; nan |] in
  let bias = f32 [| 2; 3 |] [| 0.; nan; nan; 0.25; 0.; nan |] in
  let o, lse =
    Nx.attention ~scale:1. ~lengths:(Nx.scalar Nx.int64 2L) ~mask:Nx.causal
      ~score:(fun _ s -> Nx.add s bias)
      ~acc:Nx.float32 ~kv:(Nx.Dense { k; v }) q
  in
  (* Row 0 sees key 0 alone; row 1 sees keys 0 and 1, scores -0.25 and -2. *)
  let s0 = -0.5 +. 0.25 and s1 = -2. in
  let l1 = s0 +. Float.log1p (Float.exp (s1 -. s0)) in
  let p0 = Float.exp (s0 -. l1) and p1 = Float.exp (s1 -. l1) in
  equal ~msg:"o" (array (float 1e-6)) [| 3.; (3. *. p0) -. p1 |] (Nx.to_array o);
  equal ~msg:"lse" (array (float 1e-6)) [| 0.5; l1 |] (Nx.to_array lse)

(* [q k] is [2049] for the first key and [2048] for the second, which float16
   and bfloat16 round to one value: summed in float32 or float64, the weights
   are [sigmoid 1] and [1 - sigmoid 1]. *)
let test_scores_widened () =
  let check (type s a) (dt : (float, s) D.t) u (acc : (float, a) D.t) =
    let q = Nx.create dt [| 1; 1; 2 |] [| 1.; 1. |] in
    let k = Nx.create dt [| 1; 2; 2 |] [| 2048.; 1.; 2048.; 0. |] in
    let v = Nx.create dt [| 1; 2; 1 |] [| 1.; 0. |] in
    let o, lse = Nx.attention ~scale:1. ~acc ~kv:(Nx.Dense { k; v }) q in
    let p = 1. /. (1. +. Float.exp (-1.)) in
    let msg = D.name dt ^ " in " ^ D.name acc in
    equal ~msg:(msg ^ ": o") (array (float (u *. p))) [| p |] (Nx.to_array o);
    equal ~msg:(msg ^ ": lse") (array (float 1e-3))
      [| 2048. +. Float.log1p (Float.exp 1.) |]
      (Nx.to_array lse)
  in
  check Nx.float16 0x1p-11 Nx.float32;
  check Nx.bfloat16 0x1p-8 Nx.float32;
  check Nx.float16 0x1p-11 Nx.float64;
  check Nx.bfloat16 0x1p-8 Nx.float64

let test_no_visible_key () =
  let q = f32 [| 2; 1; 2; 1 |] [| 1.; 2.; 3.; 4. |] in
  let k = f32 [| 2; 1; 2; 1 |] [| 1.; 1.; 1.; 1. |] in
  let v = f32 [| 2; 1; 2; 1 |] [| nan; nan; 5.; 5. |] in
  let o, lse =
    Nx.attention ~lengths:(ints [| 2 |] [| 0; 2 |]) ~acc:Nx.float32
      ~kv:(Nx.Dense { k; v }) q
  in
  equal ~msg:"o" (array (float 1e-6)) [| 0.; 0.; 5.; 5. |] (Nx.to_array o);
  equal ~msg:"lse" (array (float 1e-6))
    [| neg_infinity; neg_infinity; 3. +. Float.log 2.; 4. +. Float.log 2. |]
    (Nx.to_array lse);
  let o, lse =
    Nx.attention ~mask:(fun _ -> Nx.scalar Nx.bool false) ~acc:Nx.float32
      ~kv:(Nx.Dense { k; v }) q
  in
  equal ~msg:"masked o" (array float_exact) [| 0.; 0.; 0.; 0. |] (Nx.to_array o);
  equal ~msg:"masked lse" (array float_exact)
    [| neg_infinity; neg_infinity; neg_infinity; neg_infinity |]
    (Nx.to_array lse)

(* The last row sees every key before the length, each row before it one
   fewer. *)
let test_causal () =
  let p =
    Nx.
      {
        q = ints [| 2; 1 |] [| 0; 1 |];
        k = ints [| 1; 3 |] [| 0; 1; 2 |];
        len = Nx.scalar Nx.int64 2L;
        tq = 2;
      }
  in
  equal (array bool)
    [| true; false; false; true; true; false |]
    (Nx.to_array (Nx.causal p))

(* A row whose visible scores are all [-∞] has [o = 0] and [lse = -∞], a NaN
   value at a visible key included, in either accumulator. *)
let test_infinite_scores () =
  let check (type a) (acc : (float, a) D.t) =
    let q = Nx.create acc [| 1; 2; 1 |] [| 1.; 2. |] in
    let k = Nx.create acc [| 1; 2; 1 |] [| 1.; 1. |] in
    let v = Nx.create acc [| 1; 2; 1 |] [| nan; 3. |] in
    let o, lse =
      Nx.attention
        ~score:(fun _ _ -> Nx.scalar acc neg_infinity)
        ~acc ~kv:(Nx.Dense { k; v }) q
    in
    equal ~msg:(D.name acc ^ ": o") (array float_exact) [| 0.; 0. |]
      (Nx.to_array o);
    equal ~msg:(D.name acc ^ ": lse") (array float_exact)
      [| neg_infinity; neg_infinity |]
      (Nx.to_array lse)
  in
  check Nx.float32;
  check Nx.float64

(* Zero heads, and no key dimension under a given scale, are empty and zero
   scores. *)
let test_empty_extents () =
  let o, lse =
    Nx.attention ~acc:Nx.float32
      ~kv:(Nx.Dense { k = f32 [| 0; 3; 1 |] [||]; v = f32 [| 0; 3; 2 |] [||] })
      (f32 [| 0; 2; 1 |] [||])
  in
  equal ~msg:"o's shape" (array int) [| 0; 2; 2 |] (Nx.shape o);
  equal ~msg:"lse's shape" (array int) [| 0; 2 |] (Nx.shape lse);
  let o, lse =
    Nx.attention ~scale:2. ~acc:Nx.float32
      ~kv:(Nx.Dense { k = f32 [| 1; 2; 0 |] [||]; v = f32 [| 1; 2; 1 |] [| 1.; 3. |] })
      (f32 [| 1; 1; 0 |] [||])
  in
  equal ~msg:"o" (array (float 1e-6)) [| 2. |] (Nx.to_array o);
  equal ~msg:"lse" (array (float 1e-6)) [| Float.log 2. |] (Nx.to_array lse)

(* A sum of zeros is +0, here beside a value the causal mask hides, finite
   or NaN: it changes nothing. A NaN the row sees makes it NaN. In either
   accumulator. *)
let test_zero () =
  let check (type a) (acc : (float, a) D.t) =
    let q = Nx.create acc [| 1; 2; 1 |] [| 1.; 1. |] in
    let k = Nx.create acc [| 1; 2; 1 |] [| 1.; 1. |] in
    let o v =
      let v = Nx.create acc [| 1; 2; 1 |] v in
      Nx.to_array
        (fst (Nx.attention ~mask:Nx.causal ~acc ~kv:(Nx.Dense { k; v }) q))
    in
    equal ~msg:(D.name acc ^ ": finite") same_bits [| 0.; 1. |] (o [| -0.; 2. |]);
    equal ~msg:(D.name acc ^ ": NaN") same_bits [| 0.; nan |] (o [| -0.; nan |])
  in
  check Nx.float32;
  check Nx.float64

(* Keys [0, 64) of value [-1] and a score [under] beside key 64's score of
   0, of value [-0.]: the first keys' weights underflow to zero in the
   accumulator, so every product is zero and [o] is +0. On nx.cpu the
   first keys fill a block (nx_cpu.mli) whose weights a later block
   rescales to zero. *)
let test_zero_weights () =
  let check (type a) (acc : (float, a) D.t) under =
    let tk = 65 in
    let first j a b = if j < 64 then a else b in
    let q = Nx.create acc [| 1; 1; 1 |] [| 1. |] in
    let k = Nx.create acc [| 1; tk; 1 |] (Array.make tk 0.) in
    let v =
      Nx.create acc [| 1; tk; 1 |] (Array.init tk (fun j -> first j (-1.) (-0.)))
    in
    let bias =
      Nx.create acc [| 1; 1; tk |] (Array.init tk (fun j -> first j under 0.))
    in
    let o, _ =
      Nx.attention
        ~score:(fun _ s -> Nx.add s bias)
        ~acc ~kv:(Nx.Dense { k; v }) q
    in
    equal ~msg:(D.name acc) same_bits [| 0. |] (Nx.to_array o)
  in
  check Nx.float32 (-200.);
  check Nx.float64 (-800.)

(* A program whose visibility reads the score: [Nx.Prim.eval] raises, and
   nx.cpu answers [Shape_mismatch]. *)
let test_scored_visibility () =
  let module E = Nx_kernel.Prog.E in
  let prog =
    Nx_kernel.Prog.of_exprs ~ins:[| D.Any D.Float32 |]
      [|
        E.any E.(const D.Float32 0. < input 0 D.Float32);
        E.any (E.input 0 D.Float32);
      |]
  in
  let q = f32 [| 1; 1; 1 |] [| 1. |] in
  let k = f32 [| 1; 2; 1 |] [| -3.; 2. |] in
  let v = f32 [| 1; 2; 1 |] [| 100.; 1. |] in
  let op =
    Nx.Prim.Attention
      {
        prog; scale = 1.; acc = Nx.float32; out = Nx.float32; q;
        kv = Nx.Dense { k; v }; lengths = None; loads = [||];
      }
  in
  raises_match (Exn.invalid_arg ~sub:"reads its score") (fun () ->
      ignore (Nx.Prim.eval ~by:"test" op));
  let a x = Option.get (Nx.Repr.array x) in
  let o = Nx_array.create Rig.host Nx.float32 (Iarray.of_array [| 1; 1; 1 |]) in
  let lse = Nx_array.create Rig.host Nx.float32 (Iarray.of_array [| 1; 1 |]) in
  let answer =
    Nx_cpu.attention prog ~scale:1. ~o ~lse (a q)
      (Nx_kernel.Dense { k = a k; v = a v })
      [||]
  in
  equal ~msg:"nx.cpu" bool true (answer = Nx_array.Shape_mismatch)

(* A length past the keys is the keys' count, the causal mask's too. *)
let test_long_lengths () =
  let q = f32 [| 1; 2; 1 |] [| 1.; -1. |] in
  let k = f32 [| 1; 3; 1 |] [| 0.5; 2.; -1. |] in
  let v = f32 [| 1; 3; 1 |] [| 3.; -1.; 4. |] in
  let at len =
    let o, lse =
      Nx.attention ~lengths:(Nx.scalar Nx.int64 len) ~mask:Nx.causal
        ~acc:Nx.float32 ~kv:(Nx.Dense { k; v }) q
    in
    (Nx.to_array o, Nx.to_array lse)
  in
  let o, lse = at 3L and o', lse' = at 7L in
  equal ~msg:"o" same_bits o o';
  equal ~msg:"lse" same_bits lse lse'

let test_rejections () =
  let q = f32 [| 2; 3; 4 |] (Array.make 24 1.) in
  let k = f32 [| 1; 5; 4 |] (Array.make 20 1.) in
  let v = f32 [| 1; 5; 2 |] (Array.make 10 1.) in
  let kv = Nx.Dense { k; v } in
  let q64 = Nx.create Nx.float64 [| 2; 3; 4 |] (Array.make 24 1.) in
  let k64 = Nx.create Nx.float64 [| 1; 5; 4 |] (Array.make 20 1.) in
  let v64 = Nx.create Nx.float64 [| 1; 5; 2 |] (Array.make 10 1.) in
  let i8 s = Nx.create Nx.float8_e4m3fn s (Array.make (numel s) 1.) in
  let attention = Nx.attention ~acc:Nx.float32 in
  let narrow f =
    raises_match
      (Exn.invalid_arg ~sub:"is float32 or float64, no narrower than the query")
      f
  in
  narrow (fun () ->
      Nx.attention ~acc:Nx.float32 ~kv:(Nx.Dense { k = k64; v = v64 }) q64);
  narrow (fun () -> Nx.attention ~acc:Nx.float16 ~kv q);
  narrow (fun () -> Nx.attention ~acc:Nx.bfloat16 ~kv q);
  invalid (fun () ->
      Nx.attention ~acc:Nx.float32
        ~kv:(Nx.Dense { k = i8 [| 1; 5; 4 |]; v = i8 [| 1; 5; 2 |] })
        (i8 [| 2; 3; 4 |]));
  invalid (fun () -> attention ~kv (f32 [| 3; 4 |] (Array.make 12 1.)));
  invalid (fun () ->
      attention ~kv:(Nx.Dense { k = f32 [| 2; 5; 4 |] (Array.make 40 1.); v }) q);
  invalid (fun () ->
      attention ~kv:(Nx.Dense { k; v = f32 [| 1; 4; 2 |] (Array.make 8 1.) }) q);
  invalid (fun () ->
      attention ~kv:(Nx.Dense { k = f32 [| 1; 5; 3 |] (Array.make 15 1.); v }) q);
  invalid (fun () ->
      attention ~kv:(Nx.Dense { k = f32 [| 3; 5; 4 |] (Array.make 60 1.); v = f32 [| 3; 5; 2 |] (Array.make 30 1.) }) q);
  invalid (fun () -> attention ~lengths:(ints [| 1 |] [| 3 |]) ~kv q);
  invalid (fun () -> attention ~mask:(fun p -> Nx.(less p.k (sum p.q))) ~kv q);
  invalid (fun () ->
      attention ~mask:(fun p -> Nx.(less p.k (scalar int64 (item [] p.q)))) ~kv q);
  invalid (fun () ->
      attention ~mask:(fun p -> Nx.(less (reshape [| 1; 1 |] p.k) p.q)) ~kv q);
  invalid (fun () ->
      attention ~mask:(fun _ -> Nx.create Nx.bool [| 7 |] (Array.make 7 true)) ~kv q);
  invalid (fun () ->
      attention ~mask:(fun p -> Nx.(less (broadcast_to [| 7 |] p.k) p.q)) ~kv q);
  let q0 = f32 [| 2; 3; 0 |] [||] and k0 = f32 [| 1; 5; 0 |] [||] in
  invalid (fun () -> attention ~kv:(Nx.Dense { k = k0; v }) q0);
  let pool = f32 [| 2; 1; 3; 4 |] (Array.make 24 1.) in
  let table = ints [| 2; 2 |] [| 0; 1; 1; 0 |] in
  invalid (fun () -> attention ~kv:(Nx.Paged { k = pool; v = pool; table }) q)

(* Keys of more dimensions than a block of lane order's terms. *)
let test_wide_keys () =
  let draw seed n =
    Array.init n (fun k -> Float.sin ((Float.of_int k *. 0.61) +. seed))
  in
  let c =
    {
      lead = [||]; hk = 1; groups = 2; tq = 3; tk = 70; dk = 1100; dv = 3;
      lengths = None; scale = None; mask = Causal; score = Identity;
      q = draw 1. (2 * 3 * 1100); k = draw 2. (70 * 1100); v = draw 3. (70 * 3);
    }
  in
  ignore (dense D.Float32 u32 0x1p-149 in32 c : reached);
  ignore (dense D.Float64 u64 0x1p-1074 in64 c : reached)

(* nx.cpu *)

(* nx_cpu.mli's promise: a row's bits are a function of its own pairs, so
   threads, target tables, layouts and the other rows change none. Keys of
   several blocks, grouped heads of several units, lengths within a block,
   a causal mask and ALiBi's slopes, in float32 and in float64. *)
let big dt () =
  let n s = Array.fold_left ( * ) 1 s in
  let draw seed s =
    Nx.create dt s
      (Array.init (n s) (fun k -> Float.sin ((Float.of_int k *. 0.37) +. seed)))
  in
  let q = draw 1. [| 2; 8; 150; 40 |] in
  let k = draw 2. [| 2; 2; 300; 40 |] and v = draw 3. [| 2; 2; 300; 24 |] in
  (q, k, v)

let slopes8 dt =
  Nx.create dt [| 8; 1; 1 |] (Array.init 8 (fun h -> 0.05 *. Float.of_int h))

let lengths2 = ints [| 2 |] [| 250; 300 |]

let call ?(mask = Nx.causal) (q : (float, 's, Nx.host) Nx.t) k v =
  let dt = Nx.dtype q in
  let slopes = slopes8 dt in
  Nx.attention ~lengths:lengths2 ~mask
    ~score:(fun p s -> Nx.(Infix.(s - (cast dt (p.q - p.k) * slopes))))
    ~acc:dt ~kv:(Nx.Dense { k; v }) q

let bits_of (o, lse) = (Nx.to_array o, Nx.to_array lse)

let same ~msg (o, lse) (o', lse') =
  equal ~msg:(msg ^ ": o") same_bits o o';
  equal ~msg:(msg ^ ": lse") same_bits lse lse'

(* Masks whose blocks nx.cpu classes from captures: a window per head,
   and documents of 50 rows and of 100 keys, which every lead index and
   head share, classed once a call; and a causal mask through float32,
   which reads the lengths here, so that each unit runs it at the blocks
   its ranges leave mixed. *)
let windows8 =
  Nx.create Nx.int32 [| 8; 1; 1 |]
    (Array.init 8 (fun h -> Int32.of_int ((h * 20) + 1)))

let head_window p =
  Nx.(logical_and (causal p) (less (cast int32 (sub p.q p.k)) windows8))

let documents _ =
  let ids n per =
    Nx.create Nx.int16 [| n |] (Array.init n (fun i -> i / per))
  in
  Nx.equal (Nx.reshape [| 150; 1 |] (ids 150 50)) (ids 300 100)

let float_causal (p : _ Nx.pair) =
  let seen = Nx.add p.q (Nx.sub_s p.len (Int64.of_int p.tq)) in
  Nx.(less_equal (cast float32 p.k) (cast float32 seen))

let test_threads_targets dt () =
  let q, k, v = big dt () in
  List.iter
    (fun (name, mask) ->
      let call () = call ~mask q k v in
      let want = bits_of (call ()) in
      let msg m = name ^ ", " ^ m in
      same ~msg:(msg "one thread") want (bits_of (Nx_cpu.one_thread call));
      List.iter
        (fun t -> same ~msg:(msg t) want (bits_of (Nx_cpu.with_target t call)))
        Nx_cpu.targets)
    [ ("causal", Nx.causal); ("a window per head", head_window);
      ("documents", documents); ("causal through floats", float_causal) ]

(* Strided, transposed and broadcast operands against contiguous ones. *)
let test_layouts dt () =
  let q, k, v = big dt () in
  let want = bits_of (call q k v) in
  let q' = Nx.transpose ~axes:[ 0; 2; 1; 3 ] (Nx.copy (Nx.transpose ~axes:[ 0; 2; 1; 3 ] q)) in
  let k' = Nx.slice [ A; A; A; R (0, 40) ] (Nx.pad [| (0, 0); (0, 0); (0, 0); (0, 3) |] 0. k) in
  same ~msg:"strided" want (bits_of (call q' k' v));
  let v1 = Nx.get [ 0 ] v in
  let vb = Nx.broadcast_to (Nx.shape v) (Nx.reshape (Array.append [| 1 |] (Nx.shape v1)) v1) in
  same ~msg:"broadcast" (bits_of (call q k (Nx.copy vb))) (bits_of (call q k vb))

(* Each row alone, a query head with its key head alone, and a lead index
   alone give the whole's bits, a mask that reads no row count included. *)
let test_rows dt () =
  let q, k, v = big dt () in
  let mask p = Nx.less_equal p.Nx.k (Nx.add_s p.Nx.q 100L) in
  let whole = call ~mask q k v in
  let o, lse = bits_of whole in
  let row b h i =
    let qi = Nx.slice [ R (b, b + 1); R (h, h + 1); R (i, i + 1) ] q in
    let ki = Nx.slice [ R (b, b + 1); R (h / 4, (h / 4) + 1) ] k in
    let vi = Nx.slice [ R (b, b + 1); R (h / 4, (h / 4) + 1) ] v in
    let lengths = Nx.slice [ R (b, b + 1) ] lengths2 in
    let slopes = Nx.slice [ R (h, h + 1) ] (slopes8 dt) in
    let mask p = Nx.less_equal p.Nx.k (Nx.add_s (Nx.add_s p.Nx.q (Int64.of_int i)) 100L) in
    bits_of
      (Nx.attention ~lengths ~mask
         ~score:(fun p s ->
           Nx.(Infix.(s - (cast dt (p.q + scalar int64 (Int64.of_int i) - p.k) * slopes))))
         ~acc:dt ~kv:(Nx.Dense { k = ki; v = vi }) qi)
  in
  List.iter
    (fun (b, h, i) ->
      let at = (((b * 8) + h) * 150) + i in
      let o', lse' = row b h i in
      let msg = Printf.sprintf "row %d %d %d" b h i in
      equal ~msg:(msg ^ ": o") same_bits (Array.sub o (at * 24) 24) o';
      equal ~msg:(msg ^ ": lse") same_bits [| lse.(at) |] lse')
    [ (0, 0, 0); (0, 3, 70); (1, 5, 149); (1, 7, 64) ]

let () =
  let case = Gen.with_pp pp_case case and long = Gen.with_pp pp_case long in
  exit
    (run "nx attention"
       [
         group "dense"
           [
             prop ~count:400 "float32 by its definition" case (law_dense D.Float32 u32 0x1p-149 in32);
             prop ~count:400 "float16 by its definition" case (law_dense D.Float16 0x1p-11 0x1p-24 in32);
             prop ~count:400 "bfloat16 by its definition" case (law_dense D.Bfloat16 0x1p-8 0x1p-133 in32);
             prop ~count:400 "float64 by its definition" case (law_dense D.Float64 u64 0x1p-1074 in64);
             prop ~count:400 "float32 in float64 by its definition" case (law_dense D.Float32 u32 0x1p-149 in64);
             prop ~count:400 "float16 in float64 by its definition" case (law_dense D.Float16 0x1p-11 0x1p-24 in64);
             prop ~count:400 "bfloat16 in float64 by its definition" case (law_dense D.Bfloat16 0x1p-8 0x1p-133 in64);
             prop "float32, non-finite and hidden values"
               (Gen.with_pp pp_case unusual) (law_unusual D.Float32 u32 0x1p-149 in32);
             prop "float16, non-finite and hidden values"
               (Gen.with_pp pp_case unusual) (law_unusual D.Float16 0x1p-11 0x1p-24 in32);
             prop "float64, non-finite and hidden values"
               (Gen.with_pp pp_case (unusual_of ~under:(-800.) case))
               (law_unusual D.Float64 u64 0x1p-1074 in64);
             prop ~count:3000 "float32, large and tied scores"
               (Gen.with_pp pp_case tied) (law_tied D.Float32 u32 0x1p-149 in32);
             prop ~count:300 "bfloat16, large and tied scores"
               (Gen.with_pp pp_case tied) (law_tied D.Bfloat16 0x1p-8 0x1p-133 in32);
             prop ~count:3000 "float64, large and tied scores"
               (Gen.with_pp pp_case tied) (law_tied D.Float64 u64 0x1p-1074 in64);
           ];
         group "several blocks"
           [
             prop ~count:300 "float32 by its definition" long
               (law_long D.Float32 u32 0x1p-149 in32);
             prop ~count:240 "bfloat16 by its definition" long
               (law_long D.Bfloat16 0x1p-8 0x1p-133 in32);
             prop ~count:300 "float64 by its definition" long
               (law_long D.Float64 u64 0x1p-1074 in64);
             prop ~count:150 "float32, non-finite and hidden values"
               (Gen.with_pp pp_case (unusual_of long))
               (law_unusual D.Float32 u32 0x1p-149 in32);
             prop ~count:150 "float64, non-finite and hidden values"
               (Gen.with_pp pp_case (unusual_of ~under:(-800.) long))
               (law_unusual D.Float64 u64 0x1p-1074 in64);
             prop ~count:150 "paged, float32, by its definition"
               (Gen.with_pp pp_paged paged_long)
               (law_paged ~long:true D.Float32 u32 0x1p-149 in32);
             prop ~count:150 "paged, float64, by its definition"
               (Gen.with_pp pp_paged paged_long)
               (law_paged ~long:true D.Float64 u64 0x1p-1074 in64);
           ];
         prop "paged, float32, by its definition" (Gen.with_pp pp_paged paged)
           (law_paged D.Float32 u32 0x1p-149 in32);
         prop "paged, float64, by its definition" (Gen.with_pp pp_paged paged)
           (law_paged D.Float64 u64 0x1p-1074 in64);
         test "a hidden NaN" test_hidden_nan;
         test "scores widened" test_scores_widened;
         test "no visible key" test_no_visible_key;
         test "causal" test_causal;
         test "only infinite scores" test_infinite_scores;
         test "empty extents" test_empty_extents;
         test "a sum of zeros" test_zero;
         test "a sum of weights underflowed to zero" test_zero_weights;
         test "rejections" test_rejections;
         test "lengths past the keys" test_long_lengths;
         test "visibility that reads the score" test_scored_visibility;
         test "keys of 1100 dimensions" test_wide_keys;
         group "nx.cpu"
           [
             test "threads and targets change no bit, float32"
               (test_threads_targets Nx.float32);
             test "threads and targets change no bit, float64"
               (test_threads_targets Nx.float64);
             test "layouts change no bit, float32" (test_layouts Nx.float32);
             test "layouts change no bit, float64" (test_layouts Nx.float64);
             test "a row's bits are its own, float32" (test_rows Nx.float32);
             test "a row's bits are its own, float64" (test_rows Nx.float64);
           ];
       ])
