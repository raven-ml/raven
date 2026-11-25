(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's attention against its definition: the softmax of each row's visible
   scaled scores, rewritten by its score, times the values of the pairs it
   keeps, over dense and paged keys, grouped heads, lead axes, lengths, masks
   and keeps as programs and as captured arrays, empty sequences, and float16,
   bfloat16, float32 and float64 accumulating in float32 and in float64, each
   within the error bound nx.mli states. The definition computes in
   double-double arithmetic (below), within a bound of its own that the
   comparison adds. A hidden or dropped value, finite or not, never changes the
   result on nx.cpu. Then nx_cpu.mli's promises of its attention: bits that
   threads, targets, layouts and other rows never change, its blocked sum's
   bound and its float64 order. *)

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
let ints s xs = Nx.create Nx.int64 s (Array.map Int64.of_int xs)
let f32 s xs = Nx.create Nx.float32 s xs

(* Float32's and float64's unit roundoffs, and [x] rounded to float32. *)
let u32 = 0x1p-24
let u64 = 0x1p-53
let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* An accumulator: its dtype, its unit roundoff [u] and its least positive
   subnormal. *)
type 'a accum = { acc : (float, 'a) D.t; u : float; least : float }

let in32 = { acc = D.Float32; u = u32; least = 0x1p-149 }
let in64 = { acc = D.Float64; u = u64; least = 0x1p-1074 }

(* The dtypes of a call: [q]'s, with its unit roundoff [uq] and its least
   positive subnormal [tiny], and the accumulator. *)
type dtypes =
  | Dtypes : {
      q : (float, 's) D.t;
      uq : float;
      tiny : float;
      ac : 'a accum;
    }
      -> dtypes

let float16 ac = Dtypes { q = D.Float16; uq = 0x1p-11; tiny = 0x1p-24; ac }
let bfloat16 ac = Dtypes { q = D.Bfloat16; uq = 0x1p-8; tiny = 0x1p-133; ac }
let float32 ac = Dtypes { q = D.Float32; uq = u32; tiny = 0x1p-149; ac }
let float64 = Dtypes { q = D.Float64; uq = u64; tiny = 0x1p-1074; ac = in64 }
let dtypes_name (Dtypes d) = D.name d.q ^ " in " ^ D.name d.ac.acc
let in_float64 (Dtypes d) = d.ac.u = u64

(* The dtypes nx.mli allows, by accumulator: q's dtype is float16, bfloat16,
   float32 or float64, in an accumulator of float32 or float64 not narrower. *)
let in_float32s = [ float32 in32; float16 in32; bfloat16 in32 ]
let in_float64s = [ float64; float32 in64; float16 in64; bfloat16 in64 ]
let every_dtypes = in_float32s @ in_float64s

let cover_dtypes dt =
  List.iter
    (fun d -> cover (dtypes_name d) (dtypes_name d = dtypes_name dt))
    every_dtypes

(* Double-word numbers: [hi + lo] unevaluated, [|lo|] at most half an ulp of
   [hi]. [two_sum] and the product by [Float.fma] are error-free (Knuth; Dekker
   1971, "A floating-point technique for extending the available precision").
   [add] is the sloppy double-word sum: its absolute error is about [u² (|x| +
   |y|)], [u] float64's, and its relative error is unbounded under cancellation
   (Joldes, Muller and Popescu 2017, "Tight and rigorous error bounds for basic
   building blocks of double-word arithmetic"). Every bound below is at least
   [u] times the magnitudes added, so the definition neglects that error. *)
type dd = { hi : float; lo : float }

let two_sum a b =
  let s = a +. b in
  let bb = s -. a in
  { hi = s; lo = a -. (s -. bb) +. (b -. bb) }

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
   float64 at [hi], corrected to first order by [lo], so no more precise than
   float64. The definition assumes the libm within one ulp, as glibc's [exp] and
   [log] are: [exp x] within [3u] relative, [log x] within [2u |log x|] and
   [x]'s own relative error. *)
let exp_dd x =
  let e = Float.exp x.hi in
  e +. (e *. x.lo)

let log_dd x = two_sum (Float.log x.hi) (x.lo /. x.hi)

(* A case *)

(* How a captured mask is passed: as an array of [[lead…; hq; tq; tk]], as the
   same values through a view whose last two axes are transposed, or as an array
   of [[hq; tq; tk]] broadcast along the lead axes. *)
type layout = Contiguous | Transposed | Broadcast

(* [Striped] hides keys [64, 128) and every key of rows [1 mod 3]: whole blocks
   and whole rows. [Head_window w] is causal within [w.(h)] keys of the row at
   head [h], from a captured [[hq; 1; 1]] of int32; [Documents (dq, dk)] shows a
   key to a row of the same document, from captured ids of the rows and of the
   keys; [Spans (wide, a, b)] shows row [i] the keys [[a.(i), b.(i))], captured
   as [[tq; 1]] of int16, or of uint8 unless [wide]; and [Limit l] shows head
   [h] the keys up to [l.(h)], captured as [[hq; 1; 1]] of int4, for at most 8
   keys. *)
type mask =
  | Unmasked
  | Causal
  | Window
  | Striped
  | Captured of layout * bool array
  | Head_window of int array
  | Documents of int array * int array
  | Spans of bool * int array * int array
  | Limit of int array

type score = Identity | Alibi of float array | Bias of float array

type case = {
  dt : dtypes;
  lead : int array;
  hk : int;
  groups : int;
  tq : int;
  tk : int;
  dk : int;
  dv : int;
  lengths : int array option;  (** One per lead index. *)
  scale : float option;
  mask : mask;
  score : score;
      (** ALiBi's slopes, one per query head, or a bias [[hq; tq; tk]]. *)
  keep : bool array option;
      (** The pairs whose values count, captured as [[hq; tq; tk]]. *)
  q : float array;
  k : float array;
  v : float array;
}

let hq c = c.hk * c.groups
let rows c = numel c.lead
let shape c tail = Array.append c.lead tail

(* The row [b]'s length: keys at or past it are hidden. *)
let length c b =
  match c.lengths with None -> c.tk | Some l -> max 0 (min l.(b) c.tk)

let visible c b h i j =
  let len = length c b in
  let causal = j <= i + (len - c.tq) in
  let at = (((h * c.tq) + i) * c.tk) + j in
  j < len
  &&
  match c.mask with
  | Unmasked -> true
  | Causal -> causal
  | Window -> causal && i < j + 2
  | Striped -> j / 64 mod 2 = 0 && i mod 3 <> 1
  | Captured (Broadcast, m) -> m.(at)
  | Captured ((Contiguous | Transposed), m) -> m.((b * hq c * c.tq * c.tk) + at)
  | Head_window w -> causal && i - j < w.(h)
  | Documents (dq, dk) -> dq.(i) = dk.(j)
  | Spans (_, a, b) -> a.(i) <= j && j < b.(i)
  | Limit l -> j <= l.(h)

let kept c h i j =
  match c.keep with None -> true | Some a -> a.((((h * c.tq) + i) * c.tk) + j)

(* Whether [f b h i] holds at some row of [c]. *)
let some_row c f =
  let found = ref false in
  for b = 0 to rows c - 1 do
    for h = 0 to hq c - 1 do
      for i = 0 to c.tq - 1 do
        if f b h i then found := true
      done
    done
  done;
  !found

let sees c b h i = List.exists (visible c b h i) (List.init c.tk Fun.id)

let pp_case ppf c =
  let ints ppf a =
    Format.fprintf ppf "[%s]"
      (String.concat "; " (Array.to_list (Array.map string_of_int a)))
  in
  Format.fprintf ppf
    "%s lead %a hk %d groups %d tq %d tk %d dk %d dv %d lengths %s %s %s"
    (dtypes_name c.dt) ints c.lead c.hk c.groups c.tq c.tk c.dk c.dv
    (match c.lengths with
    | None -> "none"
    | Some l -> Format.asprintf "%a" ints l)
    (match c.mask with
    | Unmasked -> "unmasked"
    | Causal -> "causal"
    | Window -> "window"
    | Striped -> "striped"
    | Captured (Contiguous, _) -> "captured"
    | Captured (Transposed, _) -> "captured through a transposed view"
    | Captured (Broadcast, _) -> "captured through a broadcast view"
    | Head_window _ -> "a window per head"
    | Documents _ -> "documents"
    | Spans _ -> "spans"
    | Limit _ -> "a limit per head")
    (match c.score with
    | Identity -> "identity"
    | Alibi _ -> "alibi"
    | Bias _ -> "bias");
  let bools name a =
    Format.fprintf ppf "@ %s [%s]" name
      (String.concat "; "
         (Array.to_list (Array.map (fun b -> if b then "1" else "0") a)))
  in
  Option.iter (bools "keep") c.keep;
  let floats name xs =
    Format.fprintf ppf "@ %s [%s]" name
      (String.concat "; " (Array.to_list (Array.map (Printf.sprintf "%h") xs)))
  in
  Option.iter (floats "scale") (Option.map (fun s -> [| s |]) c.scale);
  floats "q" c.q;
  floats "k" c.k;
  floats "v" c.v;
  (match c.mask with
  | Captured (_, m) -> bools "mask" m
  | Head_window w -> Format.fprintf ppf "@ windows %a" ints w
  | Documents (dq, dk) ->
      Format.fprintf ppf "@ row documents %a@ key documents %a" ints dq ints dk
  | Spans (wide, a, b) ->
      Format.fprintf ppf "@ spans%s from %a@ to %a"
        (if wide then " of int16" else "")
        ints a ints b
  | Limit l -> Format.fprintf ppf "@ limits %a" ints l
  | Unmasked | Causal | Window | Striped -> ());
  match c.score with
  | Identity -> ()
  | Alibi xs -> floats "slopes" xs
  | Bias xs -> floats "bias" xs

let element =
  Gen.frequency
    [
      (8, Gen.float_range (-2.) 2.);
      (1, Gen.constant 0.);
      (1, Gen.constant (-0.));
    ]

(* [n] uniform booleans, drawn 30 to a choice: a mask of many pairs costs little
   to draw. *)
let bools n =
  let open Gen in
  let+ words =
    array ~size:(constant ((n + 29) / 30)) (int_range 0 ((1 lsl 30) - 1))
  in
  Array.init n (fun i -> (words.(i / 30) lsr (i mod 30)) land 1 = 1)

(* Documents of [tq] rows and [tk] keys, in order or not: ids of up to 4
   documents, or of 2 for keys of several blocks. *)
let documents tq tk =
  let open Gen in
  let* sorted = bool in
  let n = if tk > 64 then 1 else 3 in
  let ids k =
    map
      (if sorted then (fun a ->
         Array.sort compare a;
         a)
       else Fun.id)
      (array ~size:(constant k) (int_range 0 n))
  in
  let+ dq = ids tq and+ dk = ids tk in
  Documents (dq, dk)

(* Spans of keys for [tq] rows over [tk] keys. *)
let spans tq tk =
  let open Gen in
  let* wide = if tk > 254 then constant true else bool in
  let span = pair (int_range 0 tk) (int_range 0 (tk + 1)) in
  let+ ss = array ~size:(constant tq) span in
  Spans (wide, Array.map fst ss, Array.map (fun (a, b) -> max a b) ss)

let captured ~n hq tq tk =
  let open Gen in
  let* layout = of_list [ Contiguous; Transposed; Broadcast ] in
  let size = if layout = Broadcast then hq * tq * tk else n * hq * tq * tk in
  map (fun m -> Captured (layout, m)) (bools size)

(* The masks over [n] lead indices, [hq] heads, [tq] rows and [tk] keys, those
   whose blocks nx.cpu classes from captures and coordinates ([Striped], a
   window per head, documents and spans) at weight [classed], the others at
   1. *)
let masks ~classed ~n hq tq tk =
  let open Gen in
  let limit =
    if tk > 8 then []
    else
      [
        (1, map (fun l -> Limit l) (array ~size:(constant hq) (int_range 0 7)));
      ]
  in
  frequency @@ limit
  @ [
      (1, constant Unmasked);
      (1, constant Causal);
      (1, constant Window);
      (1, captured ~n hq tq tk);
      (classed, constant Striped);
      ( classed,
        map
          (fun w -> Head_window w)
          (array ~size:(constant hq) (int_range 0 (tq + 2))) );
      (classed, documents tq tk);
      (classed, spans tq tk);
    ]

(* A case of the dtypes and extents its arguments draw. *)
let sized ~dt ~classed ~lead ~hk ~groups ~tq ~tk ~dk ~dv =
  let open Gen in
  let* dt = dt in
  let* lead = lead in
  let* hk = hk in
  let* groups = groups in
  let* tq, tk = Gen.pair (tq groups) tk in
  let* dk = dk in
  let* dv = dv in
  let n = numel lead and hq = hk * groups in
  let* lengths =
    frequency
      [
        (1, constant None);
        (1, map Option.some (array ~size:(constant n) (int_range (-1) (tk + 1))));
      ]
  in
  let* scale = of_list [ None; Some 0.7; Some (-1.5) ] in
  let* mask = masks ~classed ~n hq tq tk in
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
  let* keep =
    frequency
      [ (2, constant None); (1, map Option.some (bools (hq * tq * tk))) ]
  in
  let* q = array ~size:(constant (n * hq * tq * dk)) element in
  let* k = array ~size:(constant (n * hk * tk * dk)) element in
  let+ v = array ~size:(constant (n * hk * tk * dv)) element in
  {
    dt;
    lead;
    hk;
    groups;
    tq;
    tk;
    dk;
    dv;
    lengths;
    scale;
    mask;
    score;
    keep;
    q;
    k;
    v;
  }

let case =
  let open Gen in
  sized ~dt:(of_list every_dtypes) ~classed:1
    ~lead:(array ~size:(int_range 0 2) (int_range 0 2))
    ~hk:(int_range 1 2) ~groups:(int_range 1 3)
    ~tq:(fun _ -> frequency [ (1, constant 0); (4, int_range 1 4) ])
    ~tk:(frequency [ (1, constant 0); (4, int_range 1 5) ])
    ~dk:(int_range 1 4) ~dv:(int_range 1 3)

(* Cases of several blocks of keys and several units of rows (nx_cpu.mli): keys
   past a block of 64 with a partial last block, a group's rows past a unit's (a
   unit holds [⌊64 / g⌋] rows of each of a group's [g] heads), and a group of
   more heads than a unit holds. Each accumulator is drawn half the time. *)
let long =
  let open Gen in
  sized
    ~dt:(frequency [ (1, of_list in_float32s); (1, of_list in_float64s) ])
    ~classed:2
    ~lead:(frequency [ (3, constant [||]); (1, constant [| 2 |]) ])
    ~hk:(int_range 1 2)
    ~groups:
      (frequency [ (1, constant 1); (3, int_range 2 3); (1, constant 65) ])
    ~tq:(fun g ->
      if g > 3 then int_range 1 2
      else frequency [ (3, int_range 1 8); (1, int_range 22 40) ])
    ~tk:(int_range 65 140) ~dk:(int_range 1 4) ~dv:(int_range 1 3)

(* Keys of more dimensions than a block of lane order's terms. *)
let wide_keys dt =
  let draw seed n =
    Array.init n (fun k -> Float.sin ((Float.of_int k *. 0.61) +. seed))
  in
  {
    dt;
    lead = [||];
    hk = 1;
    groups = 2;
    tq = 3;
    tk = 70;
    dk = 1100;
    dv = 3;
    lengths = None;
    scale = None;
    mask = Causal;
    score = Identity;
    keep = None;
    q = draw 1. (2 * 3 * 1100);
    k = draw 2. (70 * 1100);
    v = draw 3. (70 * 3);
  }

(* The bound *)

let gamma n v = Float.of_int n *. v /. (1. -. (Float.of_int n *. v))

(* The bound on a visible score's error in the accumulator [ac], nx.mli's [η]
   for one key: [a] is [|scale| Σ |q k|] and [s] the rewritten score. Without
   [score] it is [(2dk + 4) u a] to first order; ALiBi adds the rounding of its
   product and of its difference, a bias that of its sum. Products below the
   accumulator's normal range round absolutely. *)
let score_error ac c h i j ~a s =
  let u = ac.u in
  let dot =
    (a *. (((1. +. u) *. (1. +. u) *. (1. +. gamma (c.dk + 1) (2. *. u))) -. 1.))
    +. (Float.of_int (c.dk + 2) *. ac.least)
  in
  match c.score with
  | Identity -> dot
  | Alibi slopes ->
      (dot +. (u *. Float.abs (Float.of_int (i - j) *. slopes.(h))))
      *. (1. +. u)
      +. (u *. Float.abs s)
  | Bias _ -> (dot *. (1. +. u)) +. (u *. Float.abs s)

(* nx.cpu's bound for attention, nx.mli's [ε], of a row of [t] keys whose [lse]
   is [r] (Nx_kernels_support.cpu). *)
let logsumexp_bound ac t r =
  (Option.get Nx_kernels_support.cpu.attention).epsilon (D.Any ac.acc) t r

(* nx.mli's bound at a row of [tk] keys in [ac], for [q]'s unit roundoff [uq],
   where the visible scores, as [score] rewrites them, lie within [b], each
   within [eta] of its exact value, and [lse] is about [r]: [(g, l)], each
   element of [o] within [uq |o| + g M] of the exact one, [M] its [Σ p |v|], and
   [lse] within [l]. Here [g] is [(1 + uq) (e{^w} - 1 + γ(tk + 1, 2u) e{^w})]
   for [w = (1 + u) (2η + ε) + (2b + ln tk) u + 5u/(1 - u)], and [l] is [η + ε].
   The term for weights rounded to [v]'s dtype is absent: nx.cpu's value product
   takes the weights in the accumulator (nx_cpu.mli). *)
let bounds ac ~uq ~tk ~b ~eta ~r =
  let u = ac.u and eps = logsumexp_bound ac tk r in
  let w =
    ((1. +. u) *. ((2. *. eta) +. eps))
    +. (((2. *. b) +. Float.log (Float.of_int tk)) *. u)
    +. (5. *. u /. (1. -. u))
  in
  ( (1. +. uq) *. (Float.expm1 w +. (gamma (tk + 1) (2. *. u) *. Float.exp w)),
    eta +. eps )

(* The definition's own bound, to first order in float64's [u], for a row of [t]
   keys: each [e{^(s - m)}] within [3u] relative, so [l] too; [lse] within [(5 +
   2 ln t) u], [log]'s ulp and [l]'s error; each weight [e{^(s - lse)}] within
   [(8 + 2 ln t) u] relative, [lse]'s error and [exp]'s; [o] within that times
   [M], and [u |o|] for its rounding. *)
let reference_lse t = (5. +. (2. *. Float.log (Float.of_int t))) *. u64

let reference_o t ~mass ~o =
  ((8. +. (2. *. Float.log (Float.of_int t))) *. u64 *. mass)
  +. (u64 *. Float.abs o)

(* The definition *)

(* [s] rewritten by [c]'s score. ALiBi's product is exact: a slope is a float32
   and [i - j] a small integer. *)
let rewritten c h i j s =
  match c.score with
  | Identity -> s
  | Alibi slopes -> add s (dd (-.(Float.of_int (i - j) *. slopes.(h))))
  | Bias bias -> add s (dd bias.((((h * c.tq) + i) * c.tk) + j))

(* The exact result and what nx.mli's bound reads of it. *)
type exact = {
  o : float array;
  mass : float array;  (** [M = Σ p |v|] at each element of [o]. *)
  zero : bool array;  (** Whether each product of [o]'s sum is zero. *)
  vmax : float array;  (** The greatest finite kept [|v|] at each. *)
  lse : dd array;
  eta : float array;  (** [η] at each row. *)
  size : float array;  (** [b], the greatest visible [|s|], at each row. *)
  nan_at_zero : bool;
      (** Whether a NaN value lies at a kept key of zero weight. *)
}

(* Per row [(b, h, i)], the exact result in the accumulator [ac] over its
   visible pairs alone. [key b kh j d] and [value b kh j d] read key [j] of key
   head [kh]. Weights are the accumulator's: one below half its least subnormal
   is zero. A non-finite value gives [o] its sum's special value. *)
let definition ac c ~visible ~key ~value q =
  let hq = hq c in
  let scale =
    match c.scale with
    | Some s -> s
    | None -> 1. /. Float.sqrt (Float.of_int c.dk)
  in
  let n = rows c * hq * c.tq in
  let o = Array.make (n * c.dv) 0. and mass = Array.make (n * c.dv) 0. in
  let zero = Array.make (n * c.dv) true and vmax = Array.make (n * c.dv) 0. in
  let lses = Array.make n (dd neg_infinity) in
  let eta = Array.make n 0. and size = Array.make n 0. in
  let sums = Array.make (n * c.dv) (dd 0.) in
  let nan_at_zero = ref false in
  let s = Array.make c.tk (dd 0.) in
  for b = 0 to rows c - 1 do
    for h = 0 to hq - 1 do
      let kh = h / c.groups in
      for i = 0 to c.tq - 1 do
        let row = (((b * hq) + h) * c.tq) + i in
        let seen j = visible b h i j in
        let m = ref neg_infinity in
        for j = 0 to c.tk - 1 do
          if seen j then begin
            let dot = ref (dd 0.) and mag = ref 0. in
            for d = 0 to c.dk - 1 do
              let x = q.((row * c.dk) + d) and y = key b kh j d in
              dot := add !dot (mul_f (dd x) y);
              mag := !mag +. Float.abs (x *. y)
            done;
            s.(j) <- rewritten c h i j (mul_f !dot scale);
            let a = Float.abs scale *. !mag and x = s.(j).hi +. s.(j).lo in
            eta.(row) <- Float.max eta.(row) (score_error ac c h i j ~a x);
            size.(row) <- Float.max size.(row) (Float.abs x);
            m := Float.max !m s.(j).hi
          end
        done;
        if !m > neg_infinity then begin
          let sum = ref (dd 0.) in
          for j = 0 to c.tk - 1 do
            if seen j then
              sum := add !sum (dd (exp_dd (add s.(j) (dd (-. !m)))))
          done;
          let lse = add (dd !m) (log_dd !sum) in
          lses.(row) <- lse;
          for j = 0 to c.tk - 1 do
            if seen j && kept c h i j then begin
              let p = exp_dd (add s.(j) (neg lse)) in
              let p = if p < ac.least /. 2. then 0. else p in
              for d = 0 to c.dv - 1 do
                let y = value b kh j d and at = (row * c.dv) + d in
                if p = 0. && Float.is_nan y then nan_at_zero := true;
                o.(at) <- o.(at) +. (p *. y);
                sums.(at) <- add sums.(at) (mul_f (dd p) y);
                if p <> 0. && y <> 0. then zero.(at) <- false;
                if Float.is_finite y then
                  vmax.(at) <- Float.max vmax.(at) (Float.abs y);
                if p > 0. then mass.(at) <- mass.(at) +. (p *. Float.abs y)
              done
            end
          done
        end
      done
    done
  done;
  Array.iteri
    (fun at x ->
      if Float.is_finite x then o.(at) <- sums.(at).hi +. sums.(at).lo)
    o;
  { o; mass; zero; vmax; lse = lses; eta; size; nan_at_zero = !nan_at_zero }

(* Where a dense case's values lie, at rows with a visible key: hidden from the
   row within its length or dropped by its keep, finite or not, and not finite
   past its length. *)
type lies = {
  hidden_finite : bool;
  hidden_nonfinite : bool;
  dropped_finite : bool;
  dropped_nonfinite : bool;
  past_nonfinite : bool;
}

let lies c =
  let hf = ref false and hn = ref false and df = ref false in
  let dn = ref false and pn = ref false in
  for b = 0 to rows c - 1 do
    for h = 0 to hq c - 1 do
      for i = 0 to c.tq - 1 do
        if sees c b h i then
          for j = 0 to c.tk - 1 do
            for d = 0 to c.dv - 1 do
              let at =
                (((((b * c.hk) + (h / c.groups)) * c.tk) + j) * c.dv) + d
              in
              let finite = Float.is_finite c.v.(at) in
              if j >= length c b then
                begin if not finite then pn := true
                end
              else if not (visible c b h i j) then
                begin if finite then hf := true else hn := true
                end
              else if not (kept c h i j) then
                begin if finite then df := true else dn := true
                end
            done
          done
      done
    done
  done;
  {
    hidden_finite = !hf;
    hidden_nonfinite = !hn;
    dropped_finite = !df;
    dropped_nonfinite = !dn;
    past_nonfinite = !pn;
  }

(* Nx's call *)

let keep_of c : (Nx.host Nx.pair -> Nx.host Nx.bool_t) option =
  Option.map
    (fun a ->
      let a = Nx.create Nx.bool [| hq c; c.tq; c.tk |] a in
      fun _ -> a)
    c.keep

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
  | Captured (Contiguous, m) ->
      let m = Nx.create Nx.bool (shape c [| hq c; c.tq; c.tk |]) m in
      Some (fun _ -> m)
  | Captured (Transposed, m) ->
      (* The values laid out with the last two axes swapped, read through a view
         that swaps them back. *)
      let r = Array.length c.lead + 3 in
      let axes =
        List.init r (fun a ->
            if a = r - 2 then r - 1 else if a = r - 1 then r - 2 else a)
      in
      let m = Nx.create Nx.bool (shape c [| hq c; c.tq; c.tk |]) m in
      let m = Nx.transpose ~axes (Nx.copy (Nx.transpose ~axes m)) in
      Some (fun _ -> m)
  | Captured (Broadcast, m) ->
      let m =
        Nx.broadcast_to
          (shape c [| hq c; c.tq; c.tk |])
          (Nx.create Nx.bool [| hq c; c.tq; c.tk |] m)
      in
      Some (fun _ -> m)
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

(* [o] and [lse] against the definition, within nx.mli's bound ({!bounds}) and
   the definition's own. Underflow adds [3] least subnormals of the accumulator
   for each weight, times [|v|], and one for each product, and [tiny], [q]'s
   least subnormal, for the rounding of [o]. *)
let agrees c ~o ~lse (e : exact) =
  match c.dt with
  | Dtypes { uq; tiny; ac; _ } ->
      equal ~msg:"o's elements" int (Array.length e.o) (Array.length o);
      equal ~msg:"lse's elements" int (Array.length e.lse) (Array.length lse);
      Array.iteri
        (fun row (l' : dd) ->
          let l = lse.(row) in
          if l'.hi = neg_infinity then
            begin if l <> neg_infinity then
              failf "lse at %d: %h, by definition -inf" row l
            end
          else begin
            let g, lse_bound =
              bounds ac ~uq ~tk:c.tk ~b:e.size.(row) ~eta:e.eta.(row) ~r:l'.hi
            in
            if
              not
                (Float.abs (l -. l'.hi -. l'.lo)
                <= lse_bound +. reference_lse c.tk)
            then failf "lse at %d: %h, by definition %h + %h" row l l'.hi l'.lo;
            for d = 0 to c.dv - 1 do
              let i = (row * c.dv) + d in
              let x = o.(i) and x' = e.o.(i) and m = e.mass.(i) in
              let under =
                (Float.of_int c.tk *. ((3. *. e.vmax.(i)) +. 1.) *. ac.least)
                +. tiny
              in
              let tol =
                (uq *. Float.abs x')
                +. (g *. m) +. under
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
                failf "o at %d: %h, by definition %h, beyond the bound %h" i x
                  x' tol
            done
          end)
        e.lse

(* The dense call of [c] against the definition. *)
let dense c =
  match c.dt with
  | Dtypes { q = dt; ac; _ } ->
      let hq = hq c in
      let q = Nx.create dt (shape c [| hq; c.tq; c.dk |]) c.q in
      let k = Nx.create dt (shape c [| c.hk; c.tk; c.dk |]) c.k in
      let v = Nx.create dt (shape c [| c.hk; c.tk; c.dv |]) c.v in
      let qs = Nx.to_array q and ks = Nx.to_array k and vs = Nx.to_array v in
      let key b kh j d = ks.((((((b * c.hk) + kh) * c.tk) + j) * c.dk) + d) in
      let value b kh j d = vs.((((((b * c.hk) + kh) * c.tk) + j) * c.dv) + d) in
      let o, lse =
        Nx.attention ?scale:c.scale ?lengths:(lengths_of c) ?mask:(mask_of c)
          ?score:(score_of ac.acc c) ?keep:(keep_of c) ~acc:ac.acc
          ~kv:(Nx.Dense { k; v })
          q
      in
      equal ~msg:"o's shape" (array int)
        (shape c [| hq; c.tq; c.dv |])
        (Nx.shape o);
      equal ~msg:"lse's shape" (array int)
        (shape c [| hq; c.tq |])
        (Nx.shape lse);
      let e = definition ac c ~visible:(visible c) ~key ~value qs in
      agrees c ~o:(Nx.to_array o) ~lse:(Nx.to_array lse) e;
      e

let law_dense c =
  cover_dtypes c.dt;
  cover "causal" (c.mask = Causal);
  cover "unmasked" (c.mask = Unmasked);
  cover "a window" (c.mask = Window);
  cover "a captured mask"
    (match c.mask with Captured (Contiguous, _) -> true | _ -> false);
  cover "a mask captured through a transposed view"
    (match c.mask with Captured (Transposed, _) -> true | _ -> false);
  cover "a mask captured through a broadcast view"
    (match c.mask with Captured (Broadcast, _) -> true | _ -> false);
  cover "a window per head"
    (match c.mask with Head_window _ -> true | _ -> false);
  cover "documents" (match c.mask with Documents _ -> true | _ -> false);
  cover "spans" (match c.mask with Spans _ -> true | _ -> false);
  cover "a limit per head" (match c.mask with Limit _ -> true | _ -> false);
  cover "alibi" (c.score <> Identity);
  cover "a keep" (c.keep <> None);
  cover "lengths" (c.lengths <> None);
  let some_length f =
    rows c > 0
    && match c.lengths with Some l -> Array.exists f l | None -> false
  in
  cover "a length below 0" (some_length (fun l -> l < 0));
  cover "a length past tk" (some_length (fun l -> l > c.tk));
  cover "grouped heads" (c.groups > 1);
  cover "lead axes" (Array.length c.lead > 0);
  cover "no query" (c.tq = 0 && rows c > 0);
  cover "no key" (c.tk = 0 && c.tq > 0 && rows c > 0);
  cover "a row with no visible key"
    (some_row c (fun b h i -> not (sees c b h i)));
  cover "a row that keeps no visible key"
    (c.keep <> None
    && some_row c (fun b h i ->
        sees c b h i
        && List.for_all
             (fun j -> not (visible c b h i j && kept c h i j))
             (List.init c.tk Fun.id)));
  ignore (dense c : exact)

(* Coverage of what [long] draws, per accumulator: blocks, units and masks that
   hide whole blocks or rows. *)
let law_long c =
  cover_dtypes c.dt;
  let acc = if in_float64 c.dt then ", float64" else ", float32" in
  let per_acc label b = cover (label ^ acc) b in
  per_acc "a partial last block" (c.tk mod 64 <> 0);
  per_acc "several units of a group" (c.groups * c.tq > 64);
  per_acc "a group past a unit" (c.groups > 64);
  per_acc "whole blocks and rows hidden" (c.mask = Striped);
  per_acc "lengths within a block" (c.lengths <> None);
  per_acc "a window per head of a group"
    (match c.mask with Head_window _ -> c.groups > 1 | _ -> false);
  per_acc "documents over several blocks"
    (match c.mask with Documents _ -> true | _ -> false);
  per_acc "spans over several blocks"
    (match c.mask with Spans _ -> true | _ -> false);
  ignore (dense c : exact)

(* Scores of up to a few thousand from keys near one base key, so a row's
   visible scores lie within a few units of each other, or tie: each weight is
   far from zero while each score rounds by thousands of [u]. Keys have one or
   two dimensions, where a score's error comes nearest its bound. Zero values
   let [o] rest on keys of small weight, whose relative error is the largest. *)
let tied =
  let open Gen in
  let* c = case in
  let* dk = int_range 1 2 in
  let n = numel c.lead in
  let large = float_range (-32.) 32. in
  let* base = array ~size:(constant dk) large in
  let* q = array ~size:(constant (n * hq c * c.tq * dk)) large in
  let* near =
    array
      ~size:(constant (n * c.hk * c.tk * dk))
      (frequency [ (1, constant 0.); (3, float_range (-0.1) 0.1) ])
  in
  let+ v =
    array
      ~size:(constant (Array.length c.v))
      (frequency
         [
           (2, constant 0.); (2, constant 1.); (1, constant (-1.)); (1, element);
         ])
  in
  let k = Array.mapi (fun i x -> base.(i mod dk) +. x) near in
  { c with dk; q; k; v; score = Identity }

let law_tied c =
  cover_dtypes c.dt;
  cover "masked" (c.mask <> Unmasked || c.lengths <> None);
  cover "one key" (c.tk = 1 && c.tq > 0 && rows c > 0);
  cover "no key" (c.tk = 0 && c.tq > 0 && rows c > 0);
  ignore (dense c : exact)

(* Values with NaN, infinities and [-0.], and a bias of [0] or [under] at the
   pairs some row sees, [under] -200 in a float32 accumulator and -800 in a
   float64 one, where a weight [e{^under}] underflows to zero. A pair that every
   lead index's row hides takes a bias of NaN, [±∞] or [0]. *)
let unusual_of cases =
  let open Gen in
  let* c = cases in
  let c = { c with mask = (if c.mask = Unmasked then Causal else c.mask) } in
  let* v =
    array
      ~size:(constant (Array.length c.v))
      (frequency
         [
           (4, element);
           (1, constant nan);
           (1, constant infinity);
           (1, constant neg_infinity);
           (1, constant (-0.));
         ])
  in
  let under = if in_float64 c.dt then -800. else -200. in
  let n = hq c * c.tq * c.tk in
  let+ low = bools n and+ nan_or_zero = bools n and+ positive = bools n in
  let plain at = if low.(at) then under else 0. in
  let special at =
    match (nan_or_zero.(at), positive.(at)) with
    | true, true -> nan
    | true, false -> 0.
    | false, true -> infinity
    | false, false -> neg_infinity
  in
  let hidden at =
    let h = at / (c.tq * c.tk) and i = at / c.tk mod c.tq and j = at mod c.tk in
    List.for_all (fun b -> not (visible c b h i j)) (List.init (rows c) Fun.id)
  in
  let bias =
    Array.init n (fun at -> if hidden at then special at else plain at)
  in
  { c with v; score = Bias bias }

let law_unusual c =
  cover_dtypes c.dt;
  let e = dense c and lies = lies c in
  cover "a finite value hidden from a row" lies.hidden_finite;
  cover "a non-finite value hidden from a row" lies.hidden_nonfinite;
  cover "a finite value at a dropped pair" lies.dropped_finite;
  cover "a non-finite value at a dropped pair" lies.dropped_nonfinite;
  cover "a non-finite value past a row's length" lies.past_nonfinite;
  cover "a non-finite score at a hidden pair"
    (rows c > 0
    &&
    match c.score with
    | Bias b -> Array.exists (fun x -> not (Float.is_finite x)) b
    | _ -> false);
  cover "a NaN at a kept key of zero weight" e.nan_at_zero

let law_unusual_long c =
  cover "a float32 accumulator" (not (in_float64 c.dt));
  cover "a float64 accumulator" (in_float64 c.dt);
  let e = dense c and lies = lies c in
  cover "a finite value hidden from a row" lies.hidden_finite;
  cover "a non-finite value hidden from a row" lies.hidden_nonfinite;
  cover "a non-finite value at a dropped pair" lies.dropped_nonfinite;
  cover "a NaN at a kept key of zero weight" e.nan_at_zero

(* Paged keys: the case's keys and values are the pools, [[pages; hk; page; d]],
   read through a table of entries in [[-1, pages]] or at int64's ends. *)
type paged = {
  c : case;
  pages : int;
  page : int;
  max_pages : int;
  table : int64 array;
}

let paged_of ~pages ~page ~max_pages cases =
  let open Gen in
  let* c = cases in
  let* pages = pages in
  let* page = page in
  let* max_pages = max_pages in
  let entry =
    frequency
      [
        (3, map Int64.of_int (int_range (-1) pages));
        (1, of_list [ Int64.min_int; Int64.max_int ]);
      ]
  in
  let* table = array ~size:(constant (numel c.lead * max_pages)) entry in
  let tk = max_pages * page in
  let* k = array ~size:(constant (pages * c.hk * page * c.dk)) element in
  let* v = array ~size:(constant (pages * c.hk * page * c.dv)) element in
  let* lengths =
    frequency
      [
        (1, constant None);
        ( 1,
          map Option.some
            (array ~size:(constant (numel c.lead)) (int_range (-1) (tk + 1))) );
      ]
  in
  let* mask =
    frequency
      [
        ( 2,
          constant
            (match c.mask with
            | Captured _ | Documents _ | Spans _ | Limit _ -> Causal
            | m -> m) );
        (1, map (fun m -> Captured (Broadcast, m)) (bools (hq c * c.tq * tk)));
      ]
  in
  let+ keep =
    match c.keep with
    | None -> constant None
    | Some _ -> map Option.some (bools (hq c * c.tq * tk))
  in
  {
    c = { c with tk; k; v; lengths; mask; keep };
    pages;
    page;
    max_pages;
    table;
  }

(* Small pools, and pages within, across and as whole blocks of 64 keys. *)
let paged =
  let open Gen in
  frequency
    [
      ( 1,
        paged_of ~pages:(int_range 1 3) ~page:(int_range 1 3)
          ~max_pages:(int_range 0 3) case );
      ( 1,
        paged_of ~pages:(int_range 1 4)
          ~page:
            (frequency [ (1, constant 16); (2, constant 48); (1, constant 64) ])
          ~max_pages:(int_range 1 5) long );
    ]

let pp_paged ppf p =
  Format.fprintf ppf "%a pages %d page %d table [%s]" pp_case p.c p.pages p.page
    (String.concat "; " (Array.to_list (Array.map Int64.to_string p.table)))

let law_paged p =
  let c = p.c in
  let hq = hq c in
  cover "a float32 accumulator" (not (in_float64 c.dt));
  cover "a float64 accumulator" (in_float64 c.dt);
  cover "a page across a block's end" (p.page = 48 && p.max_pages > 1);
  let out e = e < 0L || e >= Int64.of_int p.pages in
  cover "an entry out of range" (Array.exists out p.table);
  cover "an entry in range" (Array.exists (fun e -> not (out e)) p.table);
  cover "an entry at int64's ends"
    (Array.exists (fun e -> e = Int64.min_int || e = Int64.max_int) p.table);
  match c.dt with
  | Dtypes { q = dt; ac; _ } ->
      let q = Nx.create dt (shape c [| hq; c.tq; c.dk |]) c.q in
      let k = Nx.create dt [| p.pages; c.hk; p.page; c.dk |] c.k in
      let v = Nx.create dt [| p.pages; c.hk; p.page; c.dv |] c.v in
      let entry b j = p.table.((b * p.max_pages) + (j / p.page)) in
      let qs = Nx.to_array q and ks = Nx.to_array k and vs = Nx.to_array v in
      (* An entry out of range reads zero, never seen. *)
      let pool xs d b kh j i =
        if out (entry b j) then 0.
        else
          let e = Int64.to_int (entry b j) in
          xs.((((((e * c.hk) + kh) * p.page) + (j mod p.page)) * d) + i)
      in
      let visible b h i j = (not (out (entry b j))) && visible c b h i j in
      let table = Nx.create Nx.int64 (shape c [| p.max_pages |]) p.table in
      let o, lse =
        Nx.attention ?scale:c.scale ?lengths:(lengths_of c) ?mask:(mask_of c)
          ?score:(score_of ac.acc c) ?keep:(keep_of c) ~acc:ac.acc
          ~kv:(Nx.Paged { k; v; table })
          q
      in
      agrees c ~o:(Nx.to_array o) ~lse:(Nx.to_array lse)
        (definition ac c ~visible ~key:(pool ks c.dk) ~value:(pool vs c.dv) qs)

(* pair *)

(* A length of [len] gives the bits of one of [len'], under the causal mask
   too. *)
let lengths_case (len, len') =
  let q = f32 [| 1; 2; 1 |] [| 1.; -1. |] in
  let k = f32 [| 1; 3; 1 |] [| 0.5; 2.; -1. |] in
  let v = f32 [| 1; 3; 1 |] [| 3.; -1.; 4. |] in
  let at len =
    let o, lse =
      Nx.attention ~lengths:(Nx.scalar Nx.int64 len) ~mask:Nx.causal
        ~acc:Nx.float32
        ~kv:(Nx.Dense { k; v })
        q
    in
    (Nx.to_array o, Nx.to_array lse)
  in
  let o, lse = at len and o', lse' = at len' in
  equal ~msg:"o" same_bits o' o;
  equal ~msg:"lse" same_bits lse' lse

(* causal *)

(* [causal] at every pair of [tq] rows and [tk] keys at the length [len],
   against its formula. *)
let law_causal ((tq, tk), len) =
  let p =
    Nx.
      {
        q = ints [| tq; 1 |] (Array.init tq Fun.id);
        k = ints [| 1; tk |] (Array.init tk Fun.id);
        len = Nx.scalar Nx.int64 (Int64.of_int len);
        tq;
      }
  in
  let want =
    Array.init (tq * tk) (fun x -> x mod tk <= (x / tk) + (len - tq))
  in
  equal (array bool) want (Nx.to_array (Nx.causal p))

(* nx.mli's bounds for float32 in float32 with one-hot values, where [M] is the
   weight and [|o|] too, at rows of [tk] keys whose scores lie within [b], each
   within [eta], and whose [lse] is about [r]: [o] within [u32 + g] of each
   weight, relative, and [lse] within [η + ε]. *)
let one_hot_bounds ~tk ~b ~eta ~r =
  let g, l = bounds in32 ~uq:u32 ~tk ~b ~eta ~r in
  (u32 +. g, l)

(* Right-padded rows, lengths 2 and 4 of four query rows, equal scores, and
   one-hot values, so that [o] holds each row's weights. A hidden weight is
   exactly zero. *)
let right_padded mask =
  let q = f32 [| 2; 1; 4; 1 |] (Array.make 8 0.) in
  let k = f32 [| 2; 1; 4; 1 |] (Array.make 8 0.) in
  let v =
    f32 [| 2; 1; 4; 4 |]
      (Array.init 32 (fun i -> if i / 4 mod 4 = i mod 4 then 1. else 0.))
  in
  Nx.to_array
    (fst
       (Nx.attention
          ~lengths:(ints [| 2 |] [| 2; 4 |])
          ~mask ~acc:Nx.float32
          ~kv:(Nx.Dense { k; v })
          q))

let padded_weights () =
  let rel, _ = one_hot_bounds ~tk:4 ~b:0. ~eta:0. ~r:(Float.log 4.) in
  array (float_rel ~rel ~abs:0.)

let test_padded_less_equal () =
  let t = 1. /. 3. and f = 0.25 in
  equal (padded_weights ())
    [|
      1.;
      0.;
      0.;
      0.;
      0.5;
      0.5;
      0.;
      0.;
      0.5;
      0.5;
      0.;
      0.;
      0.5;
      0.5;
      0.;
      0.;
      1.;
      0.;
      0.;
      0.;
      0.5;
      0.5;
      0.;
      0.;
      t;
      t;
      t;
      0.;
      f;
      f;
      f;
      f;
    |]
    (right_padded (fun p -> Nx.less_equal p.Nx.k p.Nx.q))

let test_padded_causal () =
  let t = 1. /. 3. and f = 0.25 in
  equal (padded_weights ())
    [|
      0.;
      0.;
      0.;
      0.;
      0.;
      0.;
      0.;
      0.;
      1.;
      0.;
      0.;
      0.;
      0.5;
      0.5;
      0.;
      0.;
      1.;
      0.;
      0.;
      0.;
      0.5;
      0.5;
      0.;
      0.;
      t;
      t;
      t;
      0.;
      f;
      f;
      f;
      f;
    |]
    (right_padded Nx.causal)

(* attention *)

(* Past the length, and where the causal mask hides a pair, a NaN key, value and
   score bias reach neither [o] nor [lse]. Every score is exact in float32 ([η =
   0]) and within [b = 2]; row 0 sees key 0 alone, so its [o] is [3] and its
   [lse] its score, [0.5]; row 1 sees keys 0 and 1, scores [-0.25] and [-2]. *)
let test_hidden_nan () =
  let q = f32 [| 1; 2; 1 |] [| 1.; -1. |] in
  let k = f32 [| 1; 3; 1 |] [| 0.5; 2.; nan |] in
  let v = f32 [| 1; 3; 1 |] [| 3.; -1.; nan |] in
  let bias = f32 [| 2; 3 |] [| 0.; nan; nan; 0.25; 0.; nan |] in
  let o, lse =
    Nx.attention ~scale:1. ~lengths:(Nx.scalar Nx.int64 2L) ~mask:Nx.causal
      ~score:(fun _ s -> Nx.add s bias)
      ~acc:Nx.float32
      ~kv:(Nx.Dense { k; v })
      q
  in
  let s0 = -0.5 +. 0.25 and s1 = -2. in
  let l1 = s0 +. Float.log1p (Float.exp (s1 -. s0)) in
  let p0 = Float.exp (s0 -. l1) and p1 = Float.exp (s1 -. l1) in
  let o1 = (3. *. p0) -. p1 in
  let o = Nx.to_array o and lse = Nx.to_array lse in
  let check row ~o:want ~mass ~lse:r =
    let g, l = bounds in32 ~uq:u32 ~tk:3 ~b:2. ~eta:0. ~r in
    let msg = Printf.sprintf "row %d" row in
    equal ~msg:(msg ^ ": o")
      (float ((u32 *. Float.abs want) +. (g *. mass)))
      want o.(row);
    equal ~msg:(msg ^ ": lse") (float l) r lse.(row)
  in
  check 0 ~o:3. ~mass:3. ~lse:0.5;
  check 1 ~o:o1 ~mass:((3. *. p0) +. p1) ~lse:l1

(* A row with no visible key, or only [-∞] scores, gives [o = 0] and
   [lse = -∞], a NaN value at a visible key included, in either
   accumulator. The first two rows: of a length of 0, or of a mask that
   holds nowhere, over NaN values. *)
(* Two lead indices of two rows over two keys, the first lead index's
   values NaN. *)
let lengths_call ?lengths ?mask () =
  let q = f32 [| 2; 1; 2; 1 |] [| 1.; 2.; 3.; 4. |] in
  let k = f32 [| 2; 1; 2; 1 |] [| 1.; 1.; 1.; 1. |] in
  let v = f32 [| 2; 1; 2; 1 |] [| nan; nan; 5.; 5. |] in
  Nx.attention ?lengths ?mask ~acc:Nx.float32 ~kv:(Nx.Dense { k; v }) q

let first_rows (o, lse) =
  (Array.sub (Nx.to_array o) 0 2, Array.sub (Nx.to_array lse) 0 2)

let only_infinite (type a) (acc : (float, a) D.t) () =
  let q = Nx.create acc [| 1; 2; 1 |] [| 1.; 2. |] in
  let k = Nx.create acc [| 1; 2; 1 |] [| 1.; 1. |] in
  let v = Nx.create acc [| 1; 2; 1 |] [| nan; 3. |] in
  let o, lse =
    Nx.attention
      ~score:(fun _ _ -> Nx.scalar acc neg_infinity)
      ~acc
      ~kv:(Nx.Dense { k; v })
      q
  in
  (Nx.to_array o, Nx.to_array lse)

let no_visible_cases =
  [
    ( "a length of 0",
      fun () -> first_rows (lengths_call ~lengths:(ints [| 2 |] [| 0; 2 |]) ())
    );
    ( "a mask that holds nowhere",
      fun () ->
        first_rows (lengths_call ~mask:(fun _ -> Nx.scalar Nx.bool false) ()) );
    ("only -inf scores in float32", only_infinite Nx.float32);
    ("only -inf scores in float64", only_infinite Nx.float64);
  ]

let no_visible_case (_, call) =
  let o, lse = call () in
  equal ~msg:"o" same_bits [| 0.; 0. |] o;
  equal ~msg:"lse" same_bits [| neg_infinity; neg_infinity |] lse

(* The second lead index's rows of a length of 2 see two keys of score [3] or
   [4], exact, and value [5]: [o] is [5] and [lse] [s + ln 2]. *)
let test_length_counts () =
  let o, lse = lengths_call ~lengths:(ints [| 2 |] [| 0; 2 |]) () in
  let o = Nx.to_array o and lse = Nx.to_array lse in
  List.iter
    (fun (row, s) ->
      let r = s +. Float.log 2. in
      let g, l = bounds in32 ~uq:u32 ~tk:2 ~b:s ~eta:0. ~r in
      let msg = Printf.sprintf "row %d" row in
      equal ~msg:(msg ^ ": o") (float ((u32 *. 5.) +. (g *. 5.))) 5. o.(row);
      equal ~msg:(msg ^ ": lse") (float l) r lse.(row))
    [ (2, 3.); (3, 4.) ]

(* A sum of zeros is +0: of a [-0.] beside a value the mask hides, finite or
   NaN, which changes nothing, in either accumulator; and of keys [0, 64) of
   value [-1] and a score [under] beside key 64's score of 0, of value [-0.]:
   the first keys' weights underflow to zero in the accumulator, so every
   product is zero. On nx.cpu the first keys fill a block (nx_cpu.mli) whose
   weights a later block rescales to zero. *)
let beside (type a) (acc : (float, a) D.t) hidden () =
  let q = Nx.create acc [| 1; 1; 1 |] [| 1. |] in
  let k = Nx.create acc [| 1; 2; 1 |] [| 1.; 1. |] in
  let v = Nx.create acc [| 1; 2; 1 |] [| -0.; hidden |] in
  let mask p = Nx.equal_s p.Nx.k 0L in
  Nx.to_array (fst (Nx.attention ~mask ~acc ~kv:(Nx.Dense { k; v }) q))

let underflowed (type a) (acc : (float, a) D.t) under () =
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
  Nx.to_array
    (fst
       (Nx.attention
          ~score:(fun _ s -> Nx.add s bias)
          ~acc
          ~kv:(Nx.Dense { k; v })
          q))

let zero_cases =
  [
    ("beside a hidden finite value, float32", beside Nx.float32 2.);
    ("beside a hidden NaN, float32", beside Nx.float32 nan);
    ("beside a hidden finite value, float64", beside Nx.float64 2.);
    ("beside a hidden NaN, float64", beside Nx.float64 nan);
    ("of weights underflowed to zero, float32", underflowed Nx.float32 (-200.));
    ("of weights underflowed to zero, float64", underflowed Nx.float64 (-800.));
  ]

(* [q k] is [2049] for the first key and [2048] for the second, which float16
   and bfloat16 round to one value: summed in float32 or float64, the weights
   are [sigmoid 1] and [1 - sigmoid 1]. Each score is exact in the accumulator
   ([η = 0]) and within [b = 2049]. *)
let widened (type s a) (dt : (float, s) D.t) uq (ac : a accum) () =
  let q = Nx.create dt [| 1; 1; 2 |] [| 1.; 1. |] in
  let k = Nx.create dt [| 1; 2; 2 |] [| 2048.; 1.; 2048.; 0. |] in
  let v = Nx.create dt [| 1; 2; 1 |] [| 1.; 0. |] in
  let o, lse = Nx.attention ~scale:1. ~acc:ac.acc ~kv:(Nx.Dense { k; v }) q in
  let p = 1. /. (1. +. Float.exp (-1.)) in
  let r = 2048. +. Float.log1p (Float.exp 1.) in
  let g, l = bounds ac ~uq ~tk:2 ~b:2049. ~eta:0. ~r in
  equal ~msg:"o" (array (float ((uq *. p) +. (g *. p)))) [| p |] (Nx.to_array o);
  equal ~msg:"lse" (array (float l)) [| r |] (Nx.to_array lse)

let widened_cases =
  [
    ("float16 in float32", widened Nx.float16 0x1p-11 in32);
    ("bfloat16 in float32", widened Nx.bfloat16 0x1p-8 in32);
    ("float16 in float64", widened Nx.float16 0x1p-11 in64);
    ("bfloat16 in float64", widened Nx.bfloat16 0x1p-8 in64);
  ]

let test_zero_heads () =
  let o, lse =
    Nx.attention ~acc:Nx.float32
      ~kv:(Nx.Dense { k = f32 [| 0; 3; 1 |] [||]; v = f32 [| 0; 3; 2 |] [||] })
      (f32 [| 0; 2; 1 |] [||])
  in
  equal ~msg:"o's shape" (array int) [| 0; 2; 2 |] (Nx.shape o);
  equal ~msg:"lse's shape" (array int) [| 0; 2 |] (Nx.shape lse)

(* Scores of zero give each of the two keys weight [1/2]: [o] is the mean [2]
   and [lse] [ln 2]. *)
let test_no_dimension () =
  let o, lse =
    Nx.attention ~scale:2. ~acc:Nx.float32
      ~kv:
        (Nx.Dense
           { k = f32 [| 1; 2; 0 |] [||]; v = f32 [| 1; 2; 1 |] [| 1.; 3. |] })
      (f32 [| 1; 1; 0 |] [||])
  in
  let g, l = bounds in32 ~uq:u32 ~tk:2 ~b:0. ~eta:0. ~r:(Float.log 2.) in
  equal ~msg:"o"
    (array (float ((u32 *. 2.) +. (g *. 2.))))
    [| 2. |] (Nx.to_array o);
  equal ~msg:"lse" (array (float l)) [| Float.log 2. |] (Nx.to_array lse)

(* Over dense and over paged keys. *)
let test_applied_once () =
  let q = f32 [| 2; 3; 1 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let k = f32 [| 1; 4; 1 |] [| 1.; -1.; 2.; 0. |] in
  let v = f32 [| 1; 4; 1 |] [| 1.; 2.; 3.; 4. |] in
  let pool = f32 [| 2; 1; 2; 1 |] [| 1.; -1.; 2.; 0. |] in
  let table = ints [| 2 |] [| 1; 0 |] in
  List.iter
    (fun (name, kv) ->
      let masks = ref 0 and scores = ref 0 and keeps = ref 0 in
      ignore
        (Nx.attention
           ~mask:(fun p ->
             incr masks;
             Nx.causal p)
           ~score:(fun _ s ->
             incr scores;
             Nx.mul_s s 2.)
           ~keep:(fun p ->
             incr keeps;
             Nx.less_equal p.k p.q)
           ~acc:Nx.float32 ~kv q);
      equal ~msg:(name ^ ": mask") int 1 !masks;
      equal ~msg:(name ^ ": score") int 1 !scores;
      equal ~msg:(name ^ ": keep") int 1 !keeps)
    [
      ("dense", Nx.Dense { k; v });
      ("paged", Nx.Paged { k = pool; v = pool; table });
    ]

(* nx.mli's window and ALiBi example *)

let position p = Nx.(add p.q (sub_s p.len (Int64.of_int p.tq)))
let window p = Nx.(logical_and (causal p) (less (position p) (add_s p.k 128L)))

let alibi slopes p s =
  Nx.(Infix.(s - (cast float32 (position p - p.k) * slopes)))

(* The example against its claim: query row [i] of [tq] stands at position [i +
   len - tq], sees the keys up to it and fewer than 128 before it, and each
   score falls by the head's slope times the key's distance from that position.
   One-hot values make [o] the weights. Two lead indices of other lengths,
   decoding one row, a chunk of rows and a whole sequence.

   Each [q k] and ALiBi product is exact and their difference rounds once, so a
   score [s] is within [u |s|]. Scores lie within [b = 0.75 + 127/16], and [lse]
   within [0.75 + ln 128] of the [128] keys a row sees. The bounds are then
   about [4.5e-5] for [o] and [6.6e-6] for [lse]; a hidden weight is exactly
   zero. *)
let test_window_example () =
  let tk = 300 and hq = 2 and slope = [| 0x1p-4; 0x1p-8 |] in
  let slopes = f32 [| hq; 1; 1 |] slope in
  let b = 0.75 +. (127. *. slope.(0)) in
  let o_rel, lse_abs =
    one_hot_bounds ~tk ~b ~eta:(u32 *. b) ~r:(0.75 +. Float.log 128.)
  in
  let check ~tq lens =
    let n = Array.length lens in
    let qs = Array.init (n * hq * tq) (fun r -> Float.of_int (r mod 7) /. 8.) in
    let ks = Array.init (n * tk) (fun j -> Float.of_int (j mod 5) /. 4.) in
    let q = f32 [| n; hq; tq; 1 |] qs in
    let k = f32 [| n; 1; tk; 1 |] ks in
    let v =
      f32 [| n; 1; tk; tk |]
        (Array.init
           (n * tk * tk)
           (fun i -> if i / tk mod tk = i mod tk then 1. else 0.))
    in
    let o, lse =
      Nx.attention ~lengths:(ints [| n |] lens) ~mask:window
        ~score:(alibi slopes) ~acc:Nx.float32
        ~kv:(Nx.Dense { k; v })
        q
    in
    let o' = Array.make (n * hq * tq * tk) 0. in
    let lse' = Array.make (n * hq * tq) neg_infinity in
    for b = 0 to n - 1 do
      for h = 0 to hq - 1 do
        for i = 0 to tq - 1 do
          let row = (((b * hq) + h) * tq) + i in
          let at = i + lens.(b) - tq in
          let seen j = j < lens.(b) && j <= at && at - j < 128 in
          let s j =
            (qs.(row) *. ks.((b * tk) + j))
            -. (Float.of_int (at - j) *. slope.(h))
          in
          let m = ref neg_infinity in
          for j = 0 to tk - 1 do
            if seen j then m := Float.max !m (s j)
          done;
          let sum = ref 0. in
          for j = 0 to tk - 1 do
            if seen j then sum := !sum +. Float.exp (s j -. !m)
          done;
          let l = !m +. Float.log !sum in
          lse'.(row) <- l;
          for j = 0 to tk - 1 do
            if seen j then o'.((row * tk) + j) <- Float.exp (s j -. l)
          done
        done
      done
    done;
    let msg = Printf.sprintf "tq %d" tq in
    equal ~msg:(msg ^ ": o")
      (array (float_rel ~rel:o_rel ~abs:0.))
      o' (Nx.to_array o);
    equal ~msg:(msg ^ ": lse") (array (float lse_abs)) lse' (Nx.to_array lse)
  in
  check ~tq:1 [| 300; 200 |];
  check ~tq:4 [| 250; 4 |];
  check ~tq:300 [| 300; 300 |]

(* Raises, in the order of nx.mli's clauses. *)

let ones s = f32 s (Array.make (numel s) 1.)
let ones8 s = Nx.create Nx.float8_e4m3fn s (Array.make (numel s) 1.)
let ones64 s = Nx.create Nx.float64 s (Array.make (numel s) 1.)
let q234 () = ones [| 2; 3; 4 |]
let kv154 () = Nx.Dense { k = ones [| 1; 5; 4 |]; v = ones [| 1; 5; 2 |] }
let attention32 = Nx.attention ~acc:Nx.float32
let all_true n = Nx.create Nx.bool [| n |] (Array.make n true)

let rejections =
  [
    ( "q of float8_e4m3fn",
      fun () ->
        ignore
          (attention32
             ~kv:(Nx.Dense { k = ones8 [| 1; 5; 4 |]; v = ones8 [| 1; 5; 2 |] })
             (ones8 [| 2; 3; 4 |])) );
    ( "a float32 accumulator for float64",
      fun () ->
        ignore
          (attention32
             ~kv:
               (Nx.Dense { k = ones64 [| 1; 5; 4 |]; v = ones64 [| 1; 5; 2 |] })
             (ones64 [| 2; 3; 4 |])) );
    ( "a float16 accumulator",
      fun () -> ignore (Nx.attention ~acc:Nx.float16 ~kv:(kv154 ()) (q234 ()))
    );
    ( "a bfloat16 accumulator",
      fun () -> ignore (Nx.attention ~acc:Nx.bfloat16 ~kv:(kv154 ()) (q234 ()))
    );
    ( "q of two axes",
      fun () -> ignore (attention32 ~kv:(kv154 ()) (ones [| 3; 4 |])) );
    ( "keys of another key head count than the values",
      fun () ->
        ignore
          (attention32
             ~kv:(Nx.Dense { k = ones [| 2; 5; 4 |]; v = ones [| 1; 5; 2 |] })
             (q234 ())) );
    ( "values of another key count",
      fun () ->
        ignore
          (attention32
             ~kv:(Nx.Dense { k = ones [| 1; 5; 4 |]; v = ones [| 1; 4; 2 |] })
             (q234 ())) );
    ( "keys of another dimension than q",
      fun () ->
        ignore
          (attention32
             ~kv:(Nx.Dense { k = ones [| 1; 5; 3 |]; v = ones [| 1; 5; 2 |] })
             (q234 ())) );
    ( "a table for lead axes q does not have",
      fun () ->
        let pool = ones [| 2; 1; 3; 4 |] in
        let table = ints [| 2; 2 |] [| 0; 1; 1; 0 |] in
        ignore
          (attention32 ~kv:(Nx.Paged { k = pool; v = pool; table }) (q234 ()))
    );
    ( "3 key heads for 2 query heads",
      fun () ->
        ignore
          (attention32
             ~kv:(Nx.Dense { k = ones [| 3; 5; 4 |]; v = ones [| 3; 5; 2 |] })
             (q234 ())) );
    ( "0 key heads for 2 query heads",
      fun () ->
        ignore
          (attention32
             ~kv:(Nx.Dense { k = ones [| 0; 5; 4 |]; v = ones [| 0; 5; 2 |] })
             (q234 ())) );
    ( "lengths of [1] for no lead axis",
      fun () ->
        ignore
          (attention32 ~lengths:(ints [| 1 |] [| 3 |]) ~kv:(kv154 ()) (q234 ()))
    );
    ( "lengths of [2] for a lead of [3]",
      fun () ->
        ignore
          (attention32
             ~lengths:(ints [| 2 |] [| 3; 3 |])
             ~kv:
               (Nx.Dense
                  { k = ones [| 3; 1; 5; 4 |]; v = ones [| 3; 1; 5; 2 |] })
             (ones [| 3; 2; 3; 4 |])) );
    ( "keys of no dimension without a scale",
      fun () ->
        ignore
          (attention32
             ~kv:(Nx.Dense { k = ones [| 1; 5; 0 |]; v = ones [| 1; 5; 2 |] })
             (ones [| 2; 3; 0 |])) );
    ( "a mask's capture of [7]",
      fun () ->
        ignore
          (attention32 ~mask:(fun _ -> all_true 7) ~kv:(kv154 ()) (q234 ())) );
    ( "a mask's result of [7]",
      fun () ->
        ignore
          (attention32
             ~mask:(fun p -> Nx.(less (broadcast_to [| 7 |] p.k) p.q))
             ~kv:(kv154 ()) (q234 ())) );
    ( "a score's capture of [7]",
      fun () ->
        let bias = ones [| 7 |] in
        ignore
          (attention32
             ~score:(fun _ s -> Nx.add s bias)
             ~kv:(kv154 ()) (q234 ())) );
    ( "a keep's capture of [7]",
      fun () ->
        ignore
          (attention32 ~keep:(fun _ -> all_true 7) ~kv:(kv154 ()) (q234 ())) );
  ]

let other_operations =
  [
    ( "a sum",
      fun () ->
        ignore
          (attention32
             ~mask:(fun p -> Nx.(less p.k (sum p.q)))
             ~kv:(kv154 ()) (q234 ())) );
    ( "a read",
      fun () ->
        ignore
          (attention32
             ~mask:(fun p -> Nx.(less p.k (scalar int64 (item [] p.q))))
             ~kv:(kv154 ()) (q234 ())) );
    ( "a reshape",
      fun () ->
        ignore
          (attention32
             ~mask:(fun p -> Nx.(less (reshape [| 1; 1 |] p.k) p.q))
             ~kv:(kv154 ()) (q234 ())) );
  ]

(* Prim.Attention *)

(* A program whose visibility reads the score: [Nx.Prim.eval] raises. *)
let test_scored_visibility () =
  let module E = Nx_kernel.Prog.E in
  let prog =
    Nx_kernel.Prog.of_exprs ~ins:[| D.Any D.Float32 |]
      [|
        E.any E.(const D.Float32 0. < input 0 D.Float32);
        E.any (E.input 0 D.Float32);
        E.any (E.const D.Bool true);
      |]
  in
  let q = f32 [| 1; 1; 1 |] [| 1. |] in
  let k = f32 [| 1; 2; 1 |] [| -3.; 2. |] in
  let v = f32 [| 1; 2; 1 |] [| 100.; 1. |] in
  let op =
    Nx.Prim.Attention
      {
        prog;
        scale = 1.;
        acc = Nx.float32;
        out = Nx.float32;
        q;
        kv = Nx.Dense { k; v };
        lengths = None;
        loads = [||];
      }
  in
  raises_match (Exn.invalid_arg ~sub:"reads its score") (fun () ->
      ignore (Nx.Prim.eval ~by:"test" op))

(* The number of outputs of an attention's program, or [None] for another
   operation. *)
let attention_outputs (type r) (op : r Nx.Prim.t) =
  match op with
  | Attention { prog; _ } -> Some (Iarray.length prog.Nx_kernel.Prog.outs)
  | _ -> None

(* A call with a mask, a score and a keep. *)
let test_interpreted () =
  let q = f32 [| 2; 3; 1 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let k = f32 [| 1; 4; 1 |] [| 1.; -1.; 2.; 0. |] in
  let v = f32 [| 1; 4; 1 |] [| 1.; 2.; 3.; 4. |] in
  let slopes = f32 [| 2; 1; 1 |] [| 0.5; 0.25 |] in
  let keep =
    Nx.create Nx.bool [| 2; 3; 4 |] (Array.init 24 (fun i -> i mod 3 > 0))
  in
  let received = ref [] in
  ignore
    (Nx.Prim.interpret ~name:"test.attention" Extent
       (fun _ ~by op ->
         received := attention_outputs op :: !received;
         Nx.Prim.eval ~by op)
       (fun _ ->
         Nx.attention ~mask:Nx.causal
           ~score:(fun p s ->
             Nx.(Infix.(s - (cast float32 (p.q - p.k) * slopes))))
           ~keep:(fun _ -> keep)
           ~acc:Nx.float32
           ~kv:(Nx.Dense { k; v })
           q));
  equal ~msg:"the outputs of each attention received" (list int) [ 3 ]
    (List.filter_map Fun.id !received)

(* nx.cpu *)

(* nx_cpu.mli's promise: a row's bits are a function of its own pairs, so
   threads, target tables, layouts and the other rows change none. Keys of
   several blocks, grouped heads of several units, lengths within a block, a
   causal mask and ALiBi's slopes, in float32 and in float64. *)
let big dt =
  let draw seed s =
    Nx.create dt s
      (Array.init (numel s) (fun k ->
           Float.sin ((Float.of_int k *. 0.37) +. seed)))
  in
  let q = draw 1. [| 2; 8; 150; 40 |] in
  let k = draw 2. [| 2; 2; 300; 40 |] and v = draw 3. [| 2; 2; 300; 24 |] in
  (q, k, v)

let slopes8 dt =
  Nx.create dt [| 8; 1; 1 |] (Array.init 8 (fun h -> 0.05 *. Float.of_int h))

let lengths2 () = ints [| 2 |] [| 250; 300 |]

let call ?(mask = Nx.causal) (q : (float, 's, Nx.host) Nx.t) k v =
  let dt = Nx.dtype q in
  let slopes = slopes8 dt in
  Nx.attention ~lengths:(lengths2 ()) ~mask
    ~score:(fun p s -> Nx.(Infix.(s - (cast dt (p.q - p.k) * slopes))))
    ~acc:dt
    ~kv:(Nx.Dense { k; v })
    q

let bits_of (o, lse) = (Nx.to_array o, Nx.to_array lse)

let same ~msg (o, lse) (o', lse') =
  equal ~msg:(msg ^ ": o") same_bits o o';
  equal ~msg:(msg ^ ": lse") same_bits lse lse'

(* Masks whose blocks nx.cpu classes from captures: a window per head, and
   documents of 50 rows and of 100 keys, which every lead index and head share,
   classed once a call; and a causal mask through float32, which reads the
   lengths here, so that each unit runs it at the blocks its ranges leave
   mixed. *)
let head_window () =
  let windows =
    Nx.create Nx.int32 [| 8; 1; 1 |]
      (Array.init 8 (fun h -> Int32.of_int ((h * 20) + 1)))
  in
  fun p -> Nx.(logical_and (causal p) (less (cast int32 (sub p.q p.k)) windows))

let documents_mask () =
  let ids n per =
    Nx.create Nx.int16 [| n |] (Array.init n (fun i -> i / per))
  in
  let rows = Nx.reshape [| 150; 1 |] (ids 150 50) and keys = ids 300 100 in
  fun _ -> Nx.equal rows keys

let float_causal (p : _ Nx.pair) =
  let seen = Nx.add p.q (Nx.sub_s p.len (Int64.of_int p.tq)) in
  Nx.(less_equal (cast float32 p.k) (cast float32 seen))

let threads_targets dt =
  let q, k, v = big dt in
  List.iter
    (fun (name, mask) ->
      let call () = call ~mask q k v in
      let want = bits_of (call ()) in
      let msg m = D.name dt ^ ", " ^ name ^ ", " ^ m in
      same ~msg:(msg "one thread") want (bits_of (Nx_cpu.one_thread call));
      List.iter
        (fun t -> same ~msg:(msg t) want (bits_of (Nx_cpu.with_target t call)))
        Nx_cpu.targets)
    [
      ("causal", Nx.causal);
      ("a window per head", head_window ());
      ("documents", documents_mask ());
      ("causal through floats", float_causal);
    ]

let test_threads_targets () =
  threads_targets Nx.float32;
  threads_targets Nx.float64

(* Strided, transposed and broadcast operands against contiguous ones. *)
let layouts dt =
  let q, k, v = big dt in
  let want = bits_of (call q k v) in
  let q' =
    Nx.transpose ~axes:[ 0; 2; 1; 3 ]
      (Nx.copy (Nx.transpose ~axes:[ 0; 2; 1; 3 ] q))
  in
  let k' =
    Nx.slice
      [ A; A; A; R (0, 40) ]
      (Nx.pad [| (0, 0); (0, 0); (0, 0); (0, 3) |] 0. k)
  in
  let msg m = D.name dt ^ ", " ^ m in
  same ~msg:(msg "strided") want (bits_of (call q' k' v));
  let v1 = Nx.get [ 0 ] v in
  let vb =
    Nx.broadcast_to (Nx.shape v)
      (Nx.reshape (Array.append [| 1 |] (Nx.shape v1)) v1)
  in
  same ~msg:(msg "broadcast")
    (bits_of (call q k (Nx.copy vb)))
    (bits_of (call q k vb))

let test_layouts () =
  layouts Nx.float32;
  layouts Nx.float64

(* Each row alone, a query head with its key head alone, and a lead index alone
   give the whole's bits, a mask that reads no row count included. *)
let rows_alone dt =
  let q, k, v = big dt in
  let mask p = Nx.less_equal p.Nx.k (Nx.add_s p.Nx.q 100L) in
  let o, lse = bits_of (call ~mask q k v) in
  let row b h i =
    let qi = Nx.slice [ R (b, b + 1); R (h, h + 1); R (i, i + 1) ] q in
    let ki = Nx.slice [ R (b, b + 1); R (h / 4, (h / 4) + 1) ] k in
    let vi = Nx.slice [ R (b, b + 1); R (h / 4, (h / 4) + 1) ] v in
    let lengths = Nx.slice [ R (b, b + 1) ] (lengths2 ()) in
    let slopes = Nx.slice [ R (h, h + 1) ] (slopes8 dt) in
    let mask p =
      Nx.less_equal p.Nx.k (Nx.add_s (Nx.add_s p.Nx.q (Int64.of_int i)) 100L)
    in
    bits_of
      (Nx.attention ~lengths ~mask
         ~score:(fun p s ->
           Nx.(
             Infix.(
               s - (cast dt (p.q + scalar int64 (Int64.of_int i) - p.k) * slopes))))
         ~acc:dt
         ~kv:(Nx.Dense { k = ki; v = vi })
         qi)
  in
  List.iter
    (fun (b, h, i) ->
      let at = (((b * 8) + h) * 150) + i in
      let o', lse' = row b h i in
      let msg = Printf.sprintf "%s, row %d %d %d" (D.name dt) b h i in
      equal ~msg:(msg ^ ": o") same_bits (Array.sub o (at * 24) 24) o';
      equal ~msg:(msg ^ ": lse") same_bits [| lse.(at) |] lse')
    [ (0, 0, 0); (0, 3, 70); (1, 5, 149); (1, 7, 64) ]

let test_rows () =
  rows_alone Nx.float32;
  rows_alone Nx.float64

(* [o]'s sum within nx_cpu.mli's factor [γ(min (t, 64) + n)] for [t] keys in [n]
   blocks, against the exact result. Each row's keys are one key repeated, so
   its scores tie and its weights are exactly [1]: [exp (+0)], with [α] [exp
   (-∞) = +0] at the first block and [1] after (Nx_kernel.Prog's special
   values), and [l] sums [t] ones exactly. Each column's value is one constant
   [c] over the keys, so the exact [o] is [c]; the sum errs by at most [γ |c|]
   and the division by [u |o|]. Whisper's base.en encoder's shape: 8 heads of
   1500 keys and 64 dimensions, scores within [±45.6] and values within [±5.2];
   its 1500 rows, and one row, which take different products. *)
let test_blocked_sum () =
  let heads = 8 and t = 1500 and d = 64 in
  let wave seed i = Float.sin ((Float.of_int i *. 0.37) +. seed) in
  let key = Array.init (heads * d) (wave 2.) in
  let norm h =
    Array.fold_left (fun s x -> s +. (x *. x)) 0. (Array.sub key (h * d) d)
  in
  (* Row r of head h scores about 45.6 sin (…) against the head's key. *)
  let q tq =
    Array.init
      (heads * tq * d)
      (fun x ->
        let h = x / (tq * d) and e = x mod d in
        let score = 45.6 *. wave 1. (x / d) in
        score *. 8. *. key.((h * d) + e) /. norm h)
  in
  let per_key f =
    Array.init (heads * t * d) (fun x -> f.((x / (t * d) * d) + (x mod d)))
  in
  let c = Array.init (heads * d) (fun x -> round32 (5.2 *. wave 3. x)) in
  let k = f32 [| 1; heads; t; d |] (per_key key) in
  let v = f32 [| 1; heads; t; d |] (per_key c) in
  let n = (t + 63) / 64 in
  let g = gamma (min t 64 + n) u32 in
  let rel = g +. (u32 *. (1. +. g)) in
  List.iter
    (fun tq ->
      let o, _ =
        Nx.attention ~acc:Nx.float32
          ~kv:(Nx.Dense { k; v })
          (f32 [| 1; heads; tq; d |] (q tq))
      in
      let want =
        Array.init
          (heads * tq * d)
          (fun x -> c.((x / (tq * d) * d) + (x mod d)))
      in
      equal
        ~msg:(Printf.sprintf "%d rows" tq)
        (array (float_rel ~rel ~abs:0.))
        want (Nx.to_array o))
    [ t; 1 ]

(* nx_cpu.mli's order of a float64 row, from its statement: each score's
   products fused in increasing dimension from +0, times the scale; per block of
   64 keys in which the row sees a key, [m'], weights [exp (x - m')] by Nx.exp
   ({!Nx_kernel.Prog.Exp}'s bits), [α], [l] becoming [α l + s] and [a] becoming
   [α a + p], each one fused multiply-add, [s] summed in 8 lanes and [p] from +0
   in increasing key, each product fused, over the block's keys or, where a
   value hidden from the row is not finite, over its visible ones; [o = a / l +
   0] and [lse = m + log l]. *)
let exps xs =
  if Array.length xs = 0 then [||]
  else Nx.to_array (Nx.exp (Nx.create Nx.float64 [| Array.length xs |] xs))

let lanes_sum s =
  let lane = Array.make 8 0. in
  Array.iteri (fun j x -> lane.(j mod 8) <- lane.(j mod 8) +. x) s;
  List.iter
    (fun h ->
      for i = 0 to h - 1 do
        lane.(i) <- lane.(i) +. lane.(i + h)
      done)
    [ 4; 2; 1 ];
  lane.(0)

let stated_row ~scale ~dv ~tk q k v visible =
  let m = ref neg_infinity and l = ref 0. and a = Array.make dv 0. in
  for b = 0 to ((tk + 63) / 64) - 1 do
    let j0 = b * 64 in
    let nk = min 64 (tk - j0) in
    let seen = List.filter (fun j -> visible (j0 + j)) (List.init nk Fun.id) in
    if seen <> [] then begin
      let score j =
        if not (visible (j0 + j)) then neg_infinity
        else begin
          let x = ref 0. in
          Array.iteri (fun d qd -> x := Float.fma qd k.(j0 + j).(d) !x) q;
          !x *. scale
        end
      in
      let s = Array.init nk score in
      let m' =
        Array.fold_left
          (fun x y -> if y > x || Float.is_nan y then y else x)
          !m s
      in
      let alpha, p =
        if m' = neg_infinity then (1., Array.make nk 0.)
        else
          let e =
            exps (Array.append [| !m -. m' |] (Array.map (fun x -> x -. m') s))
          in
          ( e.(0),
            Array.init nk (fun j -> if visible (j0 + j) then e.(j + 1) else 0.)
          )
      in
      if m' <> neg_infinity then begin
        l := Float.fma alpha !l (lanes_sum p);
        m := m'
      end;
      let hidden_bad =
        List.exists
          (fun j ->
            (not (visible (j0 + j)))
            && Array.exists (fun x -> not (Float.is_finite x)) v.(j0 + j))
          (List.init nk Fun.id)
      in
      let part = Array.make dv 0. in
      for j = 0 to nk - 1 do
        if visible (j0 + j) || not hidden_bad then
          for e = 0 to dv - 1 do
            part.(e) <- Float.fma p.(j) v.(j0 + j).(e) part.(e)
          done
      done;
      Array.iteri (fun e x -> a.(e) <- Float.fma alpha x part.(e)) a
    end
  done;
  if !l = 0. then (Array.make dv 0., neg_infinity)
  else
    let log1 =
      (Nx.to_array (Nx.log (Nx.create Nx.float64 [| 1 |] [| !l |]))).(0)
    in
    ( Array.map (fun x -> (x /. !l) +. 0.) a,
      if !m = infinity then !m else !m +. log1 )

(* The kinds a law of the stated order needs: a decode row and few rows, which
   take a gemv, many rows on the chain kernel's tiles, a mask hiding a NaN value
   from some rows, and one block. *)
type order_case = { tq : int; tk : int; nan_at : int option; masked : bool }

let pp_order ppf c =
  Format.fprintf ppf "tq %d tk %d nan at %s %s" c.tq c.tk
    (match c.nan_at with Some j -> string_of_int j | None -> "none")
    (if c.masked then "masked" else "unmasked")

let order_kinds =
  Gen.of_list ~pp:pp_order
    [
      { tq = 1; tk = 230; nan_at = None; masked = false };
      { tq = 11; tk = 230; nan_at = None; masked = false };
      { tq = 40; tk = 230; nan_at = None; masked = false };
      { tq = 5; tk = 230; nan_at = Some 100; masked = true };
      { tq = 40; tk = 230; nan_at = Some 100; masked = true };
      { tq = 40; tk = 50; nan_at = None; masked = false };
    ]

(* [o] and [lse] in float64 bit for bit against the stated order. Keys grow with
   their index, so a row's maximum rises block after block and [α] is below [1].
   The mask shows key [j] to row [i] where [j <= i + 70]. *)
let law_order (c, seed) =
  cover "one row" (c.tq = 1);
  cover "few rows" (c.tq = 11);
  cover "the chain kernel's tiles" (c.tq = 40 && not c.masked);
  cover "a hidden NaN" c.masked;
  cover "one block" (c.tk <= 64);
  let dk = 16 and dv = 24 in
  let wave s i = Float.sin ((Float.of_int i *. 0.37) +. s +. seed) in
  let q =
    Array.init c.tq (fun i ->
        Array.init dk (fun e -> (0.8 *. wave 1. ((i * dk) + e)) +. 0.3))
  in
  let k =
    Array.init c.tk (fun j ->
        Array.init dk (fun e ->
            wave 2. ((j * dk) + e) +. (Float.of_int j /. 40.)))
  in
  let v =
    Array.init c.tk (fun j ->
        Array.init dv (fun e ->
            if Some j = c.nan_at then nan else 3. *. wave 3. ((j * dv) + e)))
  in
  let flat a = Array.concat (Array.to_list a) in
  let scale = 1. /. Float.sqrt (Float.of_int dk) in
  let mask =
    if c.masked then
      Some (fun (p : _ Nx.pair) -> Nx.less_equal p.k (Nx.add_s p.q 70L))
    else None
  in
  let o, lse =
    Nx.attention ~scale ?mask ~acc:Nx.float64
      ~kv:
        (Nx.Dense
           {
             k = Nx.create Nx.float64 [| 1; 1; c.tk; dk |] (flat k);
             v = Nx.create Nx.float64 [| 1; 1; c.tk; dv |] (flat v);
           })
      (Nx.create Nx.float64 [| 1; 1; c.tq; dk |] (flat q))
  in
  let rows =
    Array.init c.tq (fun i ->
        stated_row ~scale ~dv ~tk:c.tk q.(i) k v (fun j ->
            (not c.masked) || j <= i + 70))
  in
  equal ~msg:"o" same_bits
    (Array.concat (Array.to_list (Array.map fst rows)))
    (Nx.to_array o);
  equal ~msg:"lse" same_bits (Array.map snd rows) (Nx.to_array lse)

let () =
  let case = Gen.with_pp pp_case case and long = Gen.with_pp pp_case long in
  exit
    (run "nx attention"
       [
         group "kv"
           [
             prop ~count:140
               "a paged key t of lead index b is the pools' entry table[b; t / \
                page] at t mod page"
               (Gen.with_pp pp_paged paged)
               law_paged;
           ];
         group "pair"
           [
             cases
               ~id:(fun (l, l') -> Printf.sprintf "%Ld as %Ld" l l')
               "a pair's length is its entry of lengths within [0, tk]"
               [ (7L, 3L); (-5L, 0L) ]
               lengths_case;
           ];
         group "causal"
           [
             prop ~count:200 "causal p holds where p.k <= p.q + (p.len - p.tq)"
               Gen.(pair (pair (int_range 1 4) (int_range 0 6)) (int_range 0 6))
               law_causal;
             test
               "under p.k <= p.q a right-padded lead index's row q sees the \
                keys up to q"
               test_padded_less_equal;
             test
               "under causal a right-padded lead index's first tq - len rows \
                see no key"
               test_padded_causal;
           ];
         group "attention"
           [
             prop ~count:2800
               ~examples:[ wide_keys (float32 in32); wide_keys float64 ]
               "o and lse are within the bound of the softmax of each row's \
                visible scaled scores, rewritten, times the kept values"
               case law_dense;
             prop ~count:320
               "o and lse are within the bound over keys of several blocks and \
                query heads of several units"
               long law_long;
             prop ~count:6300
               "o and lse are within the bound at large and tied scores"
               (Gen.with_pp pp_case tied) law_tied;
             prop ~count:400
               "a value or score hidden from a row or dropped never changes \
                its o on nx.cpu, and a NaN at a kept key of zero weight makes \
                it NaN"
               (Gen.with_pp pp_case (unusual_of case))
               law_unusual;
             prop ~count:80
               "a value hidden from a row or dropped never changes its o over \
                keys of several blocks"
               (Gen.with_pp pp_case (unusual_of long))
               law_unusual_long;
             test
               "a hidden pair's NaN key, value and score reach neither o nor \
                lse"
               test_hidden_nan;
             cases ~id:fst
               "a row with no visible key, or only -inf scores, gives o = 0 \
                and lse = -inf"
               no_visible_cases no_visible_case;
             test "a row's keys before its length count" test_length_counts;
             cases ~id:fst "a sum of zeros is +0" zero_cases (fun (_, o) ->
                 equal ~msg:"o" same_bits [| 0. |] (o ()));
             cases ~id:fst
               "scores, their maxima and sums, the weights and lse compute in \
                acc"
               widened_cases (fun (_, check) -> check ());
             test "zero query heads over zero key heads give empty o and lse"
               test_zero_heads;
             test "keys of no dimension under a given scale score zero"
               test_no_dimension;
             test "mask, score and keep are applied once a call"
               test_applied_once;
             test
               "the window and ALiBi example measure from the query's position \
                as causal places it"
               test_window_example;
             cases ~id:fst "raises Invalid_argument" rejections (fun (_, f) ->
                 invalid f);
             cases ~id:fst
               "another operation on the pair's values, or a read of one, \
                raises naming Nx.attention"
               other_operations (fun (_, f) -> invalid f);
           ];
         group "Prim.Attention"
           [
             test "a program whose visibility reads the score raises at eval"
               test_scored_visibility;
             test
               "an interpretation receives one attention, its program of three \
                outputs"
               test_interpreted;
           ];
         group "nx.cpu"
           [
             test "threads and target tables change no bit" test_threads_targets;
             test "layouts change no bit" test_layouts;
             test "a row's bits depend on its own pairs alone" test_rows;
             test "o's sum is within γ(min (t, 64) + n) of the exact sum"
               test_blocked_sum;
             prop ~count:120 "float64 bits follow the stated order"
               (Gen.pair order_kinds (Gen.float_range 0. 6.))
               law_order;
           ];
       ])
