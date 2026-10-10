(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* One splittable Threefry-2x32 generator. A key is an int32 array whose last
   axis holds its two words, and a batch of keys adds leading axes. Every draw
   is one map: each element computes, from its flat index in the drawn shape,
   the Threefry block its words come from, so a draw is a formula of its key and
   computes on the set of whatever reads it.

   Block [i] encrypts the counter [(2i, 2i + 1)] under the key, and word [j] of
   a draw is word [j mod 2] of block [j / 2]. A key's two int32 words are one
   uint64 to the Threefry kind, the first word low. *)

module D = Nx_array.Dtype
module M = Nx_array.Move
module P = Nx_kernel.Prog

type 'd key = (int32, D.int32_elt, 'd) Value.t

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

(* Programs *)

(* A program under construction: its nodes, newest first. *)
type b = { ins : D.any array; mutable nodes : P.node list; mutable count : int }

let builder ins = { ins; nodes = []; count = 0 }

let emit b n =
  b.nodes <- n :: b.nodes;
  b.count <- b.count + 1;
  b.count - 1

let program b outs = P.v ~ins:b.ins (Array.of_list (List.rev b.nodes)) ~outs
let const b dt v = emit b (Const (D.Any dt, P.bits dt v))
let u64 b v = const b D.Uint64 v
let bin b k x y = emit b (Op2 (Binary k, x, y))
let cmp b k x y = emit b (Op2 (Compare k, x, y))
let un b k dt x = emit b (Op1 (Unary k, D.Any dt, x))
let into b dt x = emit b (Op1 (Cast, D.Any dt, x))
let where b c x y = emit b (Op3 (Where, c, x, y))
let two32 = 0x1_0000_0000L

(* The counter of block [i], a uint64 node: the words [(2i, 2i + 1)], which
   repeat from 2^31 blocks on, so the second is xored with [q = i / 2^31], which
   keeps the counters distinct up to 2^63 blocks and is zero below 2^31. *)
let counter b i =
  let half = u64 b 0x8000_0000L in
  let q = bin b Idiv i half and r = bin b Mod i half in
  let lo = bin b Add r r in
  let hi = bin b Xor (bin b Add lo (u64 b 1L)) q in
  bin b Add lo (bin b Mul hi (u64 b two32))

let block b ~key i = bin b Threefry (counter b i) key

(* Word [j] of a draw, a uint64 node below 2^32. *)
let word b ~key j =
  let two = u64 b 2L in
  let t = block b ~key (bin b Idiv j two) in
  let first = cmp b Equal (bin b Mod j two) (u64 b 0L) in
  let w = u64 b two32 in
  where b first (bin b Mod t w) (bin b Idiv t w)

(* The flat index, as a uint64 node, of each element in the last [Array.length
   s] axes, of extents [s]. *)
let flat b s =
  let r = Array.length s in
  let acc = ref (const b D.Int64 0L) and weight = ref 1 in
  for c = 0 to r - 1 do
    let term =
      bin b Mul (emit b (Coord c)) (const b D.Int64 (Int64.of_int !weight))
    in
    acc := bin b Add !acc term;
    weight := !weight * s.(r - 1 - c)
  done;
  emit b (Op1 (Bitcast, D.Any D.Uint64, !acc))

(* The largest [p] such that every multiple of 2^-p in [0, 1) is a value of
   [dt]: the significand's width where the format reaches 2^-p, else the
   exponent of its least positive value. 53 for float64 down to 3 for
   float8_e5m2, and 1 for float4_e2m1fn, whose values below 1 are 0 and 0.5. *)
let precision (type s) (dt : (float, s) D.t) =
  let f = D.float_format dt in
  min (f.fraction_bits + 1) (1 - snd (Float.frexp (f.min_normal *. f.epsilon)))

(* Samplers compute at float64 for float64 and at float32 for the other
   floats. *)
let compute (type s) (dt : (float, s) D.t) =
  match dt with D.Float64 -> D.Any D.Float64 | _ -> D.Any D.Float32

let constf (type v s) b (dt : (v, s) D.t) (x : float) =
  match D.kind dt with
  | Float -> const b dt x
  | _ -> invalid_arg "Rng.constf: not a float dtype"

(* [x], a node of [c], rounded once to [dt]. *)
let rounded (type s) b (dt : (float, s) D.t) (D.Any c) x =
  if D.equal c dt then x else into b dt x

(* A draw in [0, 1) at position [pos] with [p] random bits, in the compute dtype
   [c]: at float32, [p] of 24 or fewer, the low [p] bits of a word scaled by
   2^-p; at float64, [p] of 53, one block's 21 + 32 bits scaled by 2^-53. Both
   are exact, so a draw is a multiple of 2^-p below 1. *)
(* 53 random bits at position [pos], a uint64 node below 2^53: the low word's
   low 21 bits over the high word, of one block. *)
let bits53 b ~key pos =
  let t = block b ~key pos and w = u64 b two32 in
  let top = bin b And (bin b Mod t w) (u64 b 0x1F_FFFFL) in
  bin b Add (bin b Mul top w) (bin b Idiv t w)

let unit b ~key ~p (D.Any c) pos =
  match c with
  | D.Float64 ->
      let m = bits53 b ~key pos in
      bin b Mul (into b D.Float64 m) (const b D.Float64 (Float.ldexp 1. (-53)))
  | _ ->
      let mask = u64 b (Int64.of_int ((1 lsl p) - 1)) in
      let bits = bin b And (word b ~key pos) mask in
      bin b Mul (into b D.Float32 bits)
        (const b D.Float32 (Float.ldexp 1. (-p)))

(* The precision of a compute dtype, float32 or float64. *)
let precision_of (D.Any c) =
  match D.kind c with
  | Float -> precision c
  | _ -> invalid_arg "Rng.precision_of: not a float dtype"

(* Draws *)

let numel s = Array.fold_left ( * ) 1 s

let check_shape ~by s =
  if Array.exists (fun d -> d < 0) s then
    invalid_argf "%s: shape %a has a negative extent" by pp_shape s

let move ~by mv x = Eval.eval ~by (Value.Move (mv, x))
let reshape ~by s x = if Prim.has_shape x s then x else move ~by (M.Reshape s) x

(* [x] at the shape [s], with leading axes: [x]'s own shape [t] aligned
   right. *)
let stretch ~by s x =
  if Prim.has_shape x s then x else move ~by (M.Broadcast s) x

(* A key's batch shape: its shape without the last axis, which holds the two
   words. Every key has that axis: [of_tensor] checks it, and the other
   constructors make it. *)
let batch (k : 'd key) =
  let s = Prim.shape k in
  Array.sub s 0 (Array.length s - 1)

(* A key's words as one uint64 per key: an operation, so a dead key raises
   here. *)
let pack ~by (k : 'd key) : (int64, D.uint64_elt, 'd) Value.t =
  Eval.eval ~by (Value.Bitcast (D.Uint64, k))

let map ~by shape prog dt loads =
  let layout = Nx_array.Layout.contiguous shape in
  let loads = Array.map (fun (Value.Any x) -> Value.Plain x) loads in
  let x, () =
    Eval.eval ~by (Value.Map { layout; prog; outs = Value.[ dt ]; loads })
  in
  x

(* A draw of [dt] and shape [s] per key of [k]: [body b ~key ~j ins] is the
   element at flat position [j] of [s], [key] its key's words and [ins] the
   nodes of [params], each loaded at the draw's shape. *)
let draw ~by ?(params = [||]) (k : 'd key) dt s body =
  check_shape ~by s;
  let batch = batch k and w = pack ~by k in
  let shape = Array.append batch s in
  let ins =
    Array.append [| D.Any D.Uint64 |]
      (Array.map (fun (Value.Any x) -> D.Any (Prim.dtype x)) params)
  in
  let b = builder ins in
  let key = emit b (In 0) in
  let params' = Array.mapi (fun i _ -> emit b (In (i + 1))) params in
  let j = flat b s in
  let out = body b ~key ~j params' in
  let w = reshape ~by (Array.append batch (Array.make (Array.length s) 1)) w in
  let loads =
    Array.append
      [| Value.Any (stretch ~by shape w) |]
      (Array.map (fun (Value.Any x) -> Value.Any (stretch ~by shape x)) params)
  in
  map ~by shape (program b [| out |]) dt loads

(* Keys *)

let key seed : 'd key =
  let b = builder [||] in
  let first = cmp b Equal (emit b (Coord 0)) (const b D.Int64 0L) in
  let hi = const b D.Int32 (Int32.of_int (seed asr 32)) in
  let lo = const b D.Int32 (Int32.of_int seed) in
  let out = where b first hi lo in
  map ~by:"Nx.Rng.key" [| 2 |] (program b [| out |]) D.Int32 [||]

let of_tensor (t : (int32, D.int32_elt, 'd) Value.t) : 'd key =
  let s = Prim.shape t in
  let r = Array.length s in
  if r = 0 || s.(r - 1) <> 2 then
    invalid_argf "Nx.Rng.of_tensor: a key has shape [...; 2], not %a" pp_shape s;
  t

let to_tensor (k : 'd key) : (int32, D.int32_elt, 'd) Value.t = k

(* The keys [f b ~key] gives at each element of [shape], whose last axes follow
   [k]'s batch. *)
let keys ~by (k : 'd key) shape f : 'd key =
  let w = pack ~by k in
  let b = builder [| D.Any D.Uint64 |] in
  let key = emit b (In 0) in
  let out = f b ~key in
  let w = stretch ~by shape w in
  let u : (int64, D.uint64_elt, 'd) Value.t =
    map ~by shape (program b [| out |]) D.Uint64 [| Value.Any w |]
  in
  Eval.eval ~by (Value.Bitcast (D.Int32, u))

let one ~by k =
  if batch k <> [||] then
    invalid_argf "%s: a batch of keys of shape %a; draw from one key" by
      pp_shape (Prim.shape k)

(* Key [i] of [n] is block [i] under [k]. *)
let blocks ~by n k =
  if n < 1 then invalid_argf "%s: n is %d, not at least 1" by n;
  one ~by k;
  keys ~by k [| n |] (fun b ~key ->
      block b ~key (emit b (Op1 (Bitcast, D.Any D.Uint64, emit b (Coord 0)))))

let split_batch ~n k = blocks ~by:"Nx.Rng.split_batch" n k

let split ?(n = 2) k =
  let by = "Nx.Rng.split" in
  let all = blocks ~by n k in
  Array.init n (fun i ->
      let row =
        M.Slice
          [|
            { start = i; count = 1; step = 1 };
            { start = 0; count = 2; step = 1 };
          |]
      in
      reshape ~by [| 2 |] (move ~by row all))

(* The counter [data] folds in: the words [(data asr 32, data)]. *)
let fold_in (k : 'd key) data : 'd key =
  let by = "Nx.Rng.fold_in" in
  let hi = Int64.logand (Int64.of_int (data asr 32)) 0xFFFF_FFFFL in
  let lo = Int64.logand (Int64.of_int data) 0xFFFF_FFFFL in
  let c = Int64.logor hi (Int64.shift_left lo 32) in
  keys ~by k (batch k) (fun b ~key -> bin b Threefry (u64 b c) key)

(* [fold_in] of an index held in data: [i]'s words [(i asr 32, i)] as the
   sign-extended [i] gives them. A batch of indices gives a batch of keys. *)
let fold_in_tensor (k : 'd key) (i : (int32, D.int32_elt, 'd) Value.t) : 'd key
    =
  let by = "Nx.Rng.fold_in_tensor" in
  let w = pack ~by k in
  let shape = Prim.broadcast_shape ~by (batch k) (Prim.shape i) in
  let b = builder [| D.Any D.Uint64; D.Any D.Int32 |] in
  let key = emit b (In 0) in
  let v =
    emit b (Op1 (Bitcast, D.Any D.Uint64, into b D.Int64 (emit b (In 1))))
  in
  let w32 = u64 b two32 in
  let c = bin b Add (bin b Idiv v w32) (bin b Mul (bin b Mod v w32) w32) in
  let out = bin b Threefry c key in
  let loads =
    [| Value.Any (stretch ~by shape w); Value.Any (stretch ~by shape i) |]
  in
  let u : (int64, D.uint64_elt, 'd) Value.t =
    map ~by shape (program b [| out |]) D.Uint64 loads
  in
  Eval.eval ~by (Value.Bitcast (D.Int32, u))

(* The scope *)

(* A value of every set, at any brand: its brand is phantom and it holds no
   bytes. *)
let every (type v s d e) ~by (x : (v, s, d) Value.t) : (v, s, e) Value.t =
  match x with
  | Value.Deferred { form = { dtype; layout; placement = None }; node; k } ->
      Value.Deferred { form = { dtype; layout; placement = None }; node; k }
  | _ -> invalid_argf "%s: the key has bytes; a scope's key is of every set" by

(* A scope answers [Next] with its root and the next index; the caller folds the
   index in, at its own brand and under whatever interprets it. *)
type _ Effect.t += Next : (unit key * int) Effect.t

let with_key k f =
  let root = every ~by:"Nx.Rng.with_key" k in
  let count = ref 0 in
  let effc (type a) (e : a Effect.t) =
    match e with
    | Next ->
        Some
          (fun (kont : (a, _) Effect.Deep.continuation) ->
            let c = !count in
            incr count;
            Effect.Deep.continue kont (root, c))
    | _ -> None
  in
  Effect.Deep.match_with f () { retc = Fun.id; exnc = raise; effc }

(* Outside every scope a domain draws from a seed of system entropy: OCaml's
   default [Random] state is seeded alike in every run. The domain keeps the
   seed and the next index, never a key, which an interpretation could have
   made. *)
let unscoped =
  Domain.DLS.new_key (fun () ->
      let seed = Random.State.bits64 (Random.State.make_self_init ()) in
      (Int64.to_int seed, ref 0))

let next_key () : 'd key =
  match Effect.perform Next with
  | root, c -> fold_in (every ~by:"Nx.Rng.next_key" root) c
  | exception Effect.Unhandled Next ->
      let seed, count = Domain.DLS.get unscoped in
      let c = !count in
      incr count;
      fold_in (key seed) c

let resolve = function Some k -> k | None -> next_key ()
let place p k = Eval.place ~by:"Nx.Rng.place" p k

(* Samplers *)

let bits ?key s =
  draw ~by:"Nx.Rng.bits" (resolve key) D.Int32 s (fun b ~key ~j _ ->
      into b D.Int32 (word b ~key j))

let uniform ?key dt s =
  let c = compute dt in
  draw ~by:"Nx.Rng.uniform" (resolve key) dt s (fun b ~key ~j _ ->
      rounded b dt c (unit b ~key ~p:(precision dt) c j))

(* The standard normal draw at position [e] of a Box-Muller draw over a [2;
   pairs] uniform draw: the radius from row 0, the angle from row 1, and both of
   the pair kept, the cosines first. [u1] is floored at 2^-p, the smallest draw
   above zero, so that the logarithm is finite and no other draw moves. A node
   of [cd], float32 or float64. *)
let gauss b ~key ~p ~pairs (D.Any cd as c) e =
  let lit x = constf b cd x in
  let pr = u64 b (Int64.of_int pairs) in
  let second = cmp b Less_equal pr e in
  let i = where b second (bin b Sub e pr) e in
  let u1 = unit b ~key ~p c i and u2 = unit b ~key ~p c (bin b Add i pr) in
  let floor = bin b Maximum u1 (lit (Float.ldexp 1. (-p))) in
  let r = un b Sqrt cd (bin b Mul (lit (-2.)) (un b Log cd floor)) in
  let angle = bin b Mul u2 (lit (2. *. Float.pi)) in
  where b second
    (bin b Mul r (un b Sin cd angle))
    (bin b Mul r (un b Cos cd angle))

let normal ?key dt s =
  let c = compute dt in
  let pairs = (numel s + 1) / 2 in
  draw ~by:"Nx.Rng.normal" (resolve key) dt s (fun b ~key ~j _ ->
      rounded b dt c (gauss b ~key ~p:(precision_of c) ~pairs c j))

(* Exponential(1) by inverse CDF, from [1 - u], which is never 0. The draw is [0
   - log (1 - u)], where a negation would make -0 of u = 0. *)
let exponential ?key dt s =
  let c = compute dt in
  draw ~by:"Nx.Rng.exponential" (resolve key) dt s (fun b ~key ~j _ ->
      let (D.Any cd) = c in
      let u = unit b ~key ~p:(precision_of c) c j in
      let y =
        bin b Sub (constf b cd 0.) (un b Log cd (bin b Sub (constf b cd 1.) u))
      in
      rounded b dt c y)

let fits v = v >= -0x8000_0000 && v <= 0x7FFF_FFFF

(* Lemire's multiply-shift: a 64-bit block [w] times the range [r] is [w r = hi
   2^64 + lo], and [hi] is uniform over [\[0, r)] once the [2^64 mod r] values
   of [lo] below it are rejected. A rejection, at most one in 2^32, takes the
   next block: round [i] of element [j] of [n] reads block [i n + j], and the
   second round's [hi] stands whatever its [lo]. [hi] comes from 32-bit halves,
   [r] being at most 2^32 - 1: [hi = (w_hi r + (w_lo r) / 2^32) / 2^32], each
   term below 2^64. [low] is added in uint32, where the sum wraps into the int32
   it names. *)
let randint ?key ?(low = 0) ~high s =
  let by = "Nx.Rng.randint" in
  if low >= high then invalid_argf "%s: low %d is not below high %d" by low high;
  if not (fits low && fits high) then
    invalid_argf "%s: [%d, %d) does not fit in int32" by low high;
  let range = Int64.of_int (high - low) in
  let threshold = Int64.unsigned_rem (Int64.neg range) range in
  let n = numel s in
  draw ~by (resolve key) D.Int32 s (fun b ~key ~j _ ->
      let r = u64 b range and w32 = u64 b two32 in
      let pick i =
        let w = block b ~key (bin b Add (u64 b (Int64.of_int (i * n))) j) in
        let wh = bin b Idiv w w32 and wl = bin b Mod w w32 in
        let hi =
          bin b Idiv
            (bin b Add (bin b Mul wh r) (bin b Idiv (bin b Mul wl r) w32))
            w32
        in
        (hi, cmp b Less (bin b Mul w r) (u64 b threshold))
      in
      let first, rejected = pick 0 in
      let second, _ = pick 1 in
      let hi = where b rejected second first in
      let v = into b D.Uint32 hi in
      let v = bin b Add v (const b D.Uint32 (Int32.of_int low)) in
      emit b (Op1 (Bitcast, D.Any D.Int32, v)))

(* Parameters *)

(* Float arithmetic on the nodes of one compute dtype. *)
type f = {
  lit : float -> int;
  add : int -> int -> int;
  sub : int -> int -> int;
  mul : int -> int -> int;
  div : int -> int -> int;
  neg : int -> int;
  log : int -> int;
  log1p : int -> int;
  exp : int -> int;
  sqrt : int -> int;
  abs : int -> int;
  floor : int -> int;
  ceil : int -> int;
  min : int -> int -> int;
  max : int -> int -> int;
  lt : int -> int -> int;
  le : int -> int -> int;
  both : int -> int -> int;
}

let floats b (D.Any cd) =
  let u k x = un b k cd x and o k x y = bin b k x y in
  {
    lit = constf b cd;
    add = o Add;
    sub = o Sub;
    mul = o Mul;
    div = o Fdiv;
    neg = u Neg;
    log = u Log;
    log1p = u Log1p;
    exp = u Exp;
    sqrt = u Sqrt;
    abs = u Abs;
    floor = u Floor;
    ceil = u Ceil;
    min = o Minimum;
    max = o Maximum;
    lt = cmp b Less;
    le = cmp b Less_equal;
    both = bin b And;
  }

(* Parameter [x] at the compute dtype [cd]. *)
let param b (D.Any cd) (p : (float, 's, 'd) Value.t) x =
  if D.equal cd (Prim.dtype p) then x else into b cd x

(* A float parameter's domain: as a refusal writes it, and the boolean node of
   whether [x], the parameter at its sampler's compute dtype, lies in it. NaN
   lies outside each. The compute dtype holds every bound, where a narrow
   parameter dtype would round it. *)
type domain = { text : string; inside : f -> int -> int }

(* [x] is finite where [x - x] is zero, and NaN elsewhere. *)
let finite f x = f.le (f.sub x x) (f.lit 0.)

let probability =
  {
    text = "[0, 1]";
    inside = (fun f x -> f.both (f.le (f.lit 0.) x) (f.le x (f.lit 1.)));
  }

let positive =
  {
    text = "(0, inf)";
    inside = (fun f x -> f.both (f.lt (f.lit 0.) x) (finite f x));
  }

let non_negative =
  {
    text = "[0, inf)";
    inside = (fun f x -> f.both (f.le (f.lit 0.) x) (finite f x));
  }

(* Poisson rates: their counts fit int32. *)
let counted =
  {
    text = "[0, 2^31)";
    inside =
      (fun f x -> f.both (f.le (f.lit 0.) x) (f.lt x (f.lit 2147483648.)));
  }

(* Raises, at the check, naming [name]'s first element of [p] where [ok], a map
   over [p], is false, [p]'s value there as [show] writes it, and [text]. *)
let refuse ~by name text ~show (p : ('v, 's, 'd) Value.t) ok =
  let dt = Prim.dtype p in
  let fail i data =
    let v =
      match data with
      | [ x ] -> (
          match Prim.expect dt x with
          | Value.Array { a; _ } -> show (Nx_array.get a [||])
          | _ -> "unread")
      | _ -> "unread"
    in
    Invalid_argument
      (Format.asprintf "%s: %s at %a is %s, not in %s" by name pp_shape i v text)
  in
  Eval.eval ~by (Value.Check { ok; data = [ Value.Any p ]; fail })

(* Checks the float parameter [p] of a sampler computing at [c] against [d]. *)
let require ~by name d (p : (float, 's, 'd) Value.t) =
  let c = compute (Prim.dtype p) in
  let b = builder [| D.Any (Prim.dtype p) |] in
  let inside = d.inside (floats b c) (param b c p (emit b (In 0))) in
  let ok =
    map ~by (Prim.shape p) (program b [| inside |]) D.Bool [| Value.Any p |]
  in
  refuse ~by name d.text ~show:(Printf.sprintf "%g") p ok

(* Checks that the counts [n] are not negative. *)
let require_counts ~by name (n : (int32, D.int32_elt, 'd) Value.t) =
  let b = builder [| D.Any D.Int32 |] in
  let inside = cmp b Less_equal (const b D.Int32 0l) (emit b (In 0)) in
  let ok =
    map ~by (Prim.shape n) (program b [| inside |]) D.Bool [| Value.Any n |]
  in
  refuse ~by name "[0, inf)" ~show:Int32.to_string n ok

let bernoulli ?key p =
  let by = "Nx.Rng.bernoulli" in
  require ~by "p" probability p;
  let c = compute (Prim.dtype p) in
  draw ~by ~params:[| Value.Any p |] (resolve key) D.Bool (Prim.shape p)
    (fun b ~key ~j ins ->
      (* [p 2^53] is exact at the compute dtype, [p] being at most 1: the draw
         is 53 bits below it, rounded up, at every dtype. *)
      let f = floats b c in
      let scaled = f.mul (param b c p ins.(0)) (f.lit (Float.ldexp 1. 53)) in
      let t = into b D.Uint64 (f.ceil scaled) in
      cmp b Less (bits53 b ~key j) t)

(* Gamma *)

(* Marsaglia and Tsang (2000): at a concentration of 1 or more, a normal draw
   [x] is squeezed through [v = (1 + x / √(9d))^3], [d = a - 1/3], and accepted
   against a uniform; below 1, Gamma(a) = Gamma(a + 1) U^(1/a). The acceptance
   test reads data, so a fixed eight rounds are drawn and the first acceptance
   taken, above 98% each; an element no round accepts, about one in 1e14, takes
   the mean. The proposals take the key's first split key, the tests its second
   and the shift its third, each element [j] of [n] drawing position [r n + j]
   in round [r].

   The draw of Gamma(boosted) and what the shift needs: the concentration [a],
   whether it is below 1, and the shift's uniform floored at 2^-p. *)
let rounds = 8

let marsaglia b ~key ~n (D.Any cd as c) a j =
  let p = precision_of c in
  let lit x = constf b cd x in
  let tiny = lit (Float.ldexp 1. (-p)) in
  let keys = Array.init 3 (fun i -> block b ~key (u64 b (Int64.of_int i))) in
  let below = cmp b Less a (lit 1.) in
  let boosted = where b below (bin b Add a (lit 1.)) a in
  let d = bin b Sub boosted (lit (1. /. 3.)) in
  let squeeze = un b Recip cd (un b Sqrt cd (bin b Mul (lit 9.) d)) in
  let pairs = ((rounds * n) + 1) / 2 in
  let acc = ref boosted and settled = ref (const b D.Bool false) in
  for r = 0 to rounds - 1 do
    let pos = bin b Add (u64 b (Int64.of_int (r * n))) j in
    let x = gauss b ~key:keys.(0) ~p ~pairs c pos in
    let u = unit b ~key:keys.(1) ~p c pos in
    let t = bin b Add (lit 1.) (bin b Mul squeeze x) in
    let v = bin b Mul t (bin b Mul t t) in
    (* [v] can be non-positive, where the logarithm is undefined: it is floored,
       and [positive] rejects it. *)
    let positive = cmp b Less (lit 0.) v in
    let log_v = un b Log cd (bin b Maximum v tiny) in
    let bound =
      bin b Add
        (bin b Mul (lit 0.5) (bin b Mul x x))
        (bin b Add d
           (bin b Add (un b Neg cd (bin b Mul d v)) (bin b Mul d log_v)))
    in
    let log_u = un b Log cd (bin b Maximum u tiny) in
    let accept = bin b And positive (cmp b Less log_u bound) in
    let fresh = cmp b Equal !settled (const b D.Bool false) in
    acc := where b (bin b And accept fresh) (bin b Mul d v) !acc;
    settled := bin b Or !settled accept
  done;
  let boost = bin b Maximum (unit b ~key:keys.(2) ~p c j) tiny in
  (!acc, below, boost)

let gamma ?key a =
  let by = "Nx.Rng.gamma" in
  require ~by "a" positive a;
  let dt = Prim.dtype a in
  let c = compute dt in
  let s = Prim.shape a in
  draw ~by ~params:[| Value.Any a |] (resolve key) dt s (fun b ~key ~j ins ->
      let (D.Any cd) = c in
      let a' = param b c a ins.(0) in
      let acc, below, boost = marsaglia b ~key ~n:(numel s) c a' j in
      let shift = bin b Pow boost (un b Recip cd a') in
      rounded b dt c (where b below (bin b Mul acc shift) acc))

(* Beta(a, b) = G(a) / (G(a) + G(b)) for independent gammas from the key's two
   split keys, formed as [1 / (1 + exp d)], [d = log G(b) - log G(a)], so that
   two gammas that underflow keep their ratio. A gamma below a concentration of
   1 is [acc U^(1/a)], so [d = log acc_b - log acc_a + q_b / b - q_a / a] with
   [q = log U] there and 0 elsewhere. Each [q / a] alone is [-∞] at a tiny
   concentration, and two of them make NaN, so the shifts are scaled by [s = min
   a b] before they meet: [(s/b) q_b - (s/a) q_a] is finite, and its quotient by
   [s] is [d]'s shift, an infinity only where the draw is 0 or 1. The quotient
   is a division, where [q · (1 / s)] would make NaN of a zero [q]. *)
let beta ?key a b' =
  let by = "Nx.Rng.beta" in
  require ~by "a" positive a;
  require ~by "b" positive b';
  let dt = Prim.dtype a in
  let c = compute dt in
  let s = Prim.broadcast_shape ~by (Prim.shape a) (Prim.shape b') in
  let n = numel s in
  draw ~by ~params:[| Value.Any a; Value.Any b' |] (resolve key) dt s
    (fun b ~key ~j ins ->
      let f = floats b c in
      let ka = block b ~key (u64 b 0L) and kb = block b ~key (u64 b 1L) in
      let a = param b c a ins.(0) and b'' = param b c b' ins.(1) in
      let gamma key x =
        let acc, below, boost = marsaglia b ~key ~n c x j in
        (f.log acc, where b below (f.log boost) (f.lit 0.))
      in
      let la, qa = gamma ka a and lb, qb = gamma kb b'' in
      let least = f.min a b'' in
      let shift =
        f.div
          (f.sub (f.mul (f.div least b'') qb) (f.mul (f.div least a) qa))
          least
      in
      let d = f.add (f.sub lb la) shift in
      rounded b dt c (f.div (f.lit 1.) (f.add (f.lit 1.) (f.exp d))))

(* Von Mises *)

(* Best and Fisher's rejection (1979) from a wrapped Cauchy envelope of
   parameter [rho]: a proposal from one uniform, accepted against a second with
   the squeeze [y (2 - y) > v] or the bound [log (y / v) + 1 - y >= 0].
   Acceptance is lowest as the concentration grows, near 0.66, so twenty rounds
   leave 5e-10 of an element unaccepted, which takes its last proposal, a draw
   from the envelope. Round [r] of element [j] of [n] reads positions [r n + j]
   and [(20 + r) n + j] of the key's uniform draw.

   The textbook constants cancel at both ends, so the draw is written in the
   uniform's half angle [phi = π (u - 1/2)], with [D = τ + √(2τ)], [τ = 1 + √(1
   + 4κ²)], [rho = 2κ / D], [1 - rho = (D - 2κ) / D] and [A = D (1 - rho)² / 4]:

   denom = (1 - rho)² + 4 rho cos² phi, y = A (1 + rho)² / denom,

   theta = 2 asin (|sin phi| (1 - rho) / √denom), signed as phi,

   where [D - 2κ = 1 + 1 / (√(1 + 4κ²) + 2κ) + √(2τ)] does not cancel. From κ =
   1 up the constants are formed in [e = 1 / (2κ)], which keeps them finite: [d
   = h + e + g], [h = √(1 + e²)], [g = √(2e (e + h))], [rho = 1 / d], [1 - rho =
   √e w / d], [A = w² / (4d)] and [w = e √e / (h + 1) + √e + √(2 (e + h))]. At κ
   = 0, [rho] is 0, [y] is 1, every proposal is accepted, and [theta] is [2
   phi]. *)
let von_mises_rounds = 20

let wrapped_cauchy b cd kappa =
  let lit x = constf b cd x in
  let sqrt x = un b Sqrt cd x and recip x = un b Recip cd x in
  let add x y = bin b Add x y and mul x y = bin b Mul x y in
  let div x y = bin b Fdiv x y in
  (* Each regime reads a concentration of its own range, so neither divides by
     zero or overflows where the other is selected. *)
  let large = cmp b Less_equal (lit 1.) kappa in
  let small_kappa = where b large (lit 0.) kappa in
  let large_kappa = where b large kappa (lit 1.) in
  let two_kappa = mul (lit 2.) small_kappa in
  let root = sqrt (add (lit 1.) (mul two_kappa two_kappa)) in
  let tau = add (lit 1.) root in
  let r = sqrt (mul (lit 2.) tau) in
  let dd = add tau r in
  let gap = add (add (lit 1.) (recip (add root two_kappa))) r in
  let e = div (lit 0.5) large_kappa in
  let h = sqrt (add (lit 1.) (mul e e)) in
  let d = add (add h e) (sqrt (mul (lit 2.) (mul e (add e h)))) in
  let w =
    add
      (add (div (mul e (sqrt e)) (add h (lit 1.))) (sqrt e))
      (sqrt (mul (lit 2.) (add e h)))
  in
  let rho = where b large (recip d) (div two_kappa dd) in
  let one_minus_rho = where b large (div (mul (sqrt e) w) d) (div gap dd) in
  let a =
    where b large
      (div (mul w w) (mul (lit 4.) d))
      (div (mul gap gap) (mul (lit 4.) dd))
  in
  (rho, one_minus_rho, a)

let von_mises ?key kappa =
  let by = "Nx.Rng.von_mises" in
  require ~by "k" non_negative kappa;
  let dt = Prim.dtype kappa in
  let c = compute dt in
  let s = Prim.shape kappa in
  let n = numel s in
  draw ~by ~params:[| Value.Any kappa |] (resolve key) dt s
    (fun b ~key ~j ins ->
      let (D.Any cd) = c in
      let p = precision_of c in
      let lit x = constf b cd x in
      let mul x y = bin b Mul x y in
      let rho, one_minus_rho, a =
        wrapped_cauchy b cd (param b c kappa ins.(0))
      in
      let one_plus_rho = bin b Add (lit 1.) rho in
      let lift = mul a (mul one_plus_rho one_plus_rho) in
      let denom phi =
        let cs = un b Cos cd phi in
        bin b Add
          (mul one_minus_rho one_minus_rho)
          (mul (mul (lit 4.) rho) (mul cs cs))
      in
      let at r = bin b Add (u64 b (Int64.of_int (r * n))) j in
      let acc = ref (lit 0.) and last = ref (lit 0.) in
      let settled = ref (const b D.Bool false) in
      for r = 0 to von_mises_rounds - 1 do
        let u = unit b ~key ~p c (at r) in
        let v = unit b ~key ~p c (at (von_mises_rounds + r)) in
        let phi = mul (bin b Sub u (lit 0.5)) (lit Float.pi) in
        let y = bin b Fdiv lift (denom phi) in
        let squeeze = cmp b Less v (mul y (bin b Sub (lit 2.) y)) in
        let bound =
          cmp b Less_equal (lit (-1.))
            (bin b Sub (un b Log cd (bin b Fdiv y v)) y)
        in
        let accept = bin b Or squeeze bound in
        let fresh = cmp b Equal !settled (const b D.Bool false) in
        acc := where b (bin b And accept fresh) phi !acc;
        settled := bin b Or !settled accept;
        last := phi
      done;
      let phi = where b !settled !acc !last in
      let half =
        bin b Fdiv
          (mul (un b Abs cd (un b Sin cd phi)) one_minus_rho)
          (un b Sqrt cd (denom phi))
      in
      let asin = un b Asin cd (bin b Minimum half (lit 1.)) in
      let theta = mul (un b Sign cd phi) (mul (lit 2.) asin) in
      rounded b dt c theta)

(* Counts *)

let either b x y = bin b Or x y
let not_ b x = cmp b Equal x (const b D.Bool false)

(* Log pmfs at integer-valued counts, written so that no term grows with the
   counts: the direct forms cancel terms of size [m log m] to a margin of order
   one, which float32 loses above a mean of ten thousand. With Stirling's
   formula for the factorials they are sums of saddle-point deviances [bd0 (x,
   m) = x log (x/m) + m - x], remainders [stirlerr x] of Stirling's series and a
   logarithm of order one (Loader, 2000).

   Near [x = m] the deviance is itself a cancellation, so there it comes from
   Loader's series in [v = (x - m) / (x + m)], whose terms all have one sign;
   [deviance x m d] takes [d = x - m] from its caller, who forms it without
   subtracting two large numbers. The remainder's asymptotic series holds from 8
   up; below, it is read off a shift by eight. Both regimes of each [where] run
   everywhere, so the arithmetic stays finite where it is not selected. *)
let stirlerr b f y =
  let main y =
    f.add
      (f.sub (f.mul (f.add y (f.lit 0.5)) (f.log y)) y)
      (f.lit (0.5 *. Float.log (2.0 *. Float.pi)))
  in
  let tail y =
    let y2 = f.mul y y in
    f.div
      (f.sub
         (f.lit (1.0 /. 12.0))
         (f.div
            (f.sub (f.lit (1.0 /. 360.0)) (f.div (f.lit (1.0 /. 1260.0)) y2))
            y2))
      y
  in
  let shifted = f.add y (f.lit 8.0) in
  let product = ref (f.add y (f.lit 1.0)) in
  for i = 2 to 8 do
    product := f.mul !product (f.add y (f.lit (float_of_int i)))
  done;
  let small =
    f.sub
      (f.sub (f.add (main shifted) (tail shifted)) (f.log !product))
      (main y)
  in
  where b (f.lt y (f.lit 8.0)) small (tail y)

let deviance b f x m d =
  let s = f.add x m in
  let v = f.div d s in
  let v2 = f.mul v v in
  let acc = ref (f.mul d v) in
  let term = ref (f.mul (f.mul (f.lit 2.0) x) v) in
  for j = 1 to 6 do
    term := f.mul !term v2;
    acc := f.add !acc (f.div !term (f.lit (float_of_int ((2 * j) + 1))))
  done;
  let direct = f.add (f.sub (f.mul x (f.log (f.div x m))) x) m in
  where b (f.lt (f.abs d) (f.mul (f.lit 0.1) s)) !acc direct

(* The Poisson log pmf: [-bd0 (k, rate) - log (2π k) / 2 - stirlerr k]. *)
let log_poisson_pmf b f k rate =
  where b
    (f.lt k (f.lit 0.5))
    (f.neg rate)
    (f.sub
       (f.sub
          (f.neg (deviance b f k rate (f.sub k rate)))
          (f.mul (f.lit 0.5) (f.log (f.mul (f.lit (2.0 *. Float.pi)) k))))
       (stirlerr b f k))

(* The binomial log pmf of [n] trials of probability [p], [q = 1 - p]:

   stirlerr n - stirlerr k - stirlerr (n - k) - bd0 (k, n p) - bd0 (n - k, n q)
   - log (2π k (n - k) / n) / 2,

   both deviances taking their difference from [n p - k]. The counts [0] and [n]
   have their own branches; the rest is read at a count strictly inside them. *)
let log_binomial_pmf b f k n p q =
  let inside = f.max (f.min k (f.sub n (f.lit 1.0))) (f.lit 1.0) in
  let rest = f.sub n inside in
  let d = f.sub inside (f.mul n p) in
  let saddle =
    f.sub
      (f.sub
         (f.sub
            (f.sub (stirlerr b f n) (stirlerr b f inside))
            (stirlerr b f rest))
         (deviance b f inside (f.mul n p) d))
      (deviance b f rest (f.mul n q) (f.neg d))
  in
  let spread =
    f.add
      (f.log (f.mul (f.lit (2.0 *. Float.pi)) inside))
      (f.log1p (f.neg (f.div inside n)))
  in
  where b
    (f.lt k (f.lit 0.5))
    (f.mul n (f.log1p (f.neg p)))
    (where b
       (f.lt (f.sub n (f.lit 0.5)) k)
       (f.mul n (f.log p))
       (f.sub saddle (f.mul (f.lit 0.5) spread)))

(* [log i!] for [i] below [n]. *)
let log_factorials n =
  let acc = ref 0.0 in
  Array.init n (fun i ->
      if i > 0 then acc := !acc +. Float.log (float_of_int i);
      !acc)

(* A rejection loop's draws: round [r] of element [j] of [n] reads positions [r
   n + j] and [(rounds + r) n + j] of the key's uniform draw. *)
let pair b ~key ~p ~n ~rounds c j r =
  let at r = bin b Add (u64 b (Int64.of_int (r * n))) j in
  (unit b ~key ~p c (at r), unit b ~key ~p c (at (rounds + r)))

(* Two regimes with a fixed round count each, chosen per element, so the shape
   of the computation does not depend on the rate. Below 10, inversion: one
   uniform against the cumulative pmf over 48 terms, the mass beyond them 4e-18
   at rate 10; a rate of zero gives the count 0. From 10 up, Hörmann's
   transformed rejection with squeeze (PTRS): sixteen rounds, each accepting
   0.75 to 0.89, leave 2e-10 of an element unaccepted, which takes its last
   proposal, a draw from the hat with mean near the rate. The elements the
   inversion owns see a rate of 1e5 there, away from the hat's poles below 1.
   Inversion reads the key's first split key, rejection its second. *)
let poisson_inversion_rounds = 48
let poisson_rejection_rounds = 16

let poisson ?key rate =
  let by = "Nx.Rng.poisson" in
  require ~by "rate" counted rate;
  let dt = Prim.dtype rate in
  let c = compute dt in
  let s = Prim.shape rate in
  let n = numel s in
  draw ~by ~params:[| Value.Any rate |] (resolve key) D.Int32 s
    (fun b ~key ~j ins ->
      let f = floats b c in
      let p = precision_of c in
      let rate = param b c rate ins.(0) in
      let k0 = block b ~key (u64 b 0L) and k1 = block b ~key (u64 b 1L) in
      let small = not_ b (f.le (f.lit 10.0) rate) in
      let inversion =
        let u = unit b ~key:k0 ~p c j in
        let log_rate = f.log rate in
        let cdf = ref (f.lit 0.0) and count = ref (f.lit 0.0) in
        Array.iteri
          (fun i lf ->
            let log_pmf =
              f.sub
                (f.sub (f.mul (f.lit (float_of_int i)) log_rate) rate)
                (f.lit lf)
            in
            cdf := f.add !cdf (f.exp log_pmf);
            count :=
              f.add !count (where b (f.lt !cdf u) (f.lit 1.0) (f.lit 0.0)))
          (log_factorials poisson_inversion_rounds);
        !count
      in
      let rejection =
        let rounds = poisson_rejection_rounds in
        let lam = where b small (f.lit 1e5) rate in
        let b' = f.add (f.lit 0.931) (f.mul (f.lit 2.53) (f.sqrt lam)) in
        let a = f.add (f.lit (-0.059)) (f.mul (f.lit 0.02483) b') in
        let log_inv_alpha =
          f.log
            (f.add (f.lit 1.1239) (f.div (f.lit 1.1328) (f.sub b' (f.lit 3.4))))
        in
        let vr =
          f.sub (f.lit 0.9277) (f.div (f.lit 3.6224) (f.sub b' (f.lit 2.0)))
        in
        let acc = ref (f.lit 0.0) and last = ref (f.lit 0.0) in
        let settled = ref (const b D.Bool false) in
        for r = 0 to rounds - 1 do
          let u, v = pair b ~key:k1 ~p ~n ~rounds c j r in
          let u = f.sub u (f.lit 0.5) in
          let us = f.sub (f.lit 0.5) (f.abs u) in
          let proposal =
            f.floor
              (f.add
                 (f.add
                    (f.mul (f.add (f.div (f.mul (f.lit 2.0) a) us) b') u)
                    lam)
                 (f.lit 0.43))
          in
          (* A [us] of zero sends the proposal to an infinity and an infinite
             rate makes it NaN: the tests reject both, and the count the cast
             reads stays finite. *)
          let count =
            where b
              (f.le (f.lit 0.0) proposal)
              (f.min proposal (f.lit 2147483647.0))
              (f.lit 0.0)
          in
          let squeeze = f.both (f.le (f.lit 0.07) us) (f.le v vr) in
          let reject =
            either b
              (f.lt proposal (f.lit 0.0))
              (f.both (f.lt us (f.lit 0.013)) (f.lt us v))
          in
          let lhs =
            f.sub
              (f.add (f.log v) log_inv_alpha)
              (f.log (f.add (f.div a (f.mul us us)) b'))
          in
          let accept =
            either b squeeze
              (f.both (not_ b reject)
                 (f.le lhs (log_poisson_pmf b f count lam)))
          in
          acc := where b (f.both accept (not_ b !settled)) count !acc;
          settled := either b !settled accept;
          last := count
        done;
        where b !settled !acc !last
      in
      into b D.Int32 (where b small inversion rejection))

(* Binomial(n, p) counts the failures of [1 - p] where [p] exceeds one half, so
   the regimes see [p <= 1/2], chosen per element by the mean [n p]. Below 10,
   inversion over 48 terms: they follow from [(1 - p)^n] by the ratios [(n - j)
   p / ((j + 1) (1 - p))], summed in logarithms, and a uniform scaled by their
   sum meets their running sum, so the count never passes [n]; the mass beyond
   is below 1e-16. From 10 up, Hörmann's transformed rejection (BTRS, 1993):
   eighteen rounds, each accepting at least 0.71, leave 2e-10 of an element
   unaccepted, which takes its last proposal. The elements the inversion owns
   see [n = 1000, p = 1/2] there. *)
let binomial_inversion_rounds = 48
let binomial_rejection_rounds = 18

let binomial ?key (count : (int32, D.int32_elt, 'd) Value.t) prob =
  let by = "Nx.Rng.binomial" in
  require_counts ~by "n" count;
  require ~by "p" probability prob;
  let dt = Prim.dtype prob in
  let c = compute dt in
  let s = Prim.broadcast_shape ~by (Prim.shape count) (Prim.shape prob) in
  let len = numel s in
  draw ~by ~params:[| Value.Any count; Value.Any prob |] (resolve key) D.Int32 s
    (fun b ~key ~j ins ->
      let f = floats b c in
      let (D.Any cd) = c in
      let bits = precision_of c in
      let n = into b cd ins.(0) in
      let p = param b c prob ins.(1) in
      let flip = f.lt (f.lit 0.5) p in
      let p = where b flip (f.sub (f.lit 1.0) p) p in
      let small = not_ b (f.le (f.lit 10.0) (f.mul n p)) in
      let k0 = block b ~key (u64 b 0L) and k1 = block b ~key (u64 b 1L) in
      let inversion =
        (* Term [i] holds [log (n (n - 1) … (n - i + 1) (p / q)^i)]: the
           logarithms of the ratios [(n - l) p / q], summed over [l < i]. *)
        let odds = f.sub (f.log p) (f.log1p (f.neg p)) in
        let base = f.mul n (f.log1p (f.neg p)) in
        let factorials = log_factorials binomial_inversion_rounds in
        let falling = ref (f.lit 0.0) and cdf = ref (f.lit 0.0) in
        let cdfs =
          Array.mapi
            (fun i lf ->
              if i > 0 then begin
                let l = f.lit (float_of_int (i - 1)) in
                let ratio =
                  f.add (f.log (f.max (f.sub n l) (f.lit 0.0))) odds
                in
                falling := f.add !falling ratio
              end;
              let log_pmf = f.sub (f.add base !falling) (f.lit lf) in
              cdf := f.add !cdf (f.exp log_pmf);
              !cdf)
            factorials
        in
        let u = f.mul (unit b ~key:k0 ~p:bits c j) !cdf in
        Array.fold_left
          (fun acc cdf ->
            f.add acc (where b (f.lt cdf u) (f.lit 1.0) (f.lit 0.0)))
          (f.lit 0.0) cdfs
      in
      let rejection =
        let rounds = binomial_rejection_rounds in
        let n = where b small (f.lit 1000.0) n
        and p = where b small (f.lit 0.5) p in
        let q = f.sub (f.lit 1.0) p in
        let mean = f.mul n p in
        let spq = f.sqrt (f.mul mean q) in
        let b' = f.add (f.lit 1.15) (f.mul (f.lit 2.53) spq) in
        let a =
          f.add
            (f.add (f.lit (-0.0873)) (f.mul (f.lit 0.0248) b'))
            (f.mul (f.lit 0.01) p)
        in
        let centre = f.add mean (f.lit 0.5) in
        let log_alpha =
          f.log (f.mul (f.add (f.lit 2.83) (f.div (f.lit 5.1) b')) spq)
        in
        let vr = f.sub (f.lit 0.92) (f.div (f.lit 4.2) b') in
        let mode = f.floor (f.mul (f.add n (f.lit 1.0)) p) in
        let log_mode = log_binomial_pmf b f mode n p q in
        let acc = ref (f.lit 0.0) and last = ref (f.lit 0.0) in
        let settled = ref (const b D.Bool false) in
        for r = 0 to rounds - 1 do
          let u, v = pair b ~key:k1 ~p:bits ~n:len ~rounds c j r in
          let u = f.sub u (f.lit 0.5) in
          let us = f.sub (f.lit 0.5) (f.abs u) in
          let proposal =
            f.floor
              (f.add
                 (f.mul (f.add (f.div (f.mul (f.lit 2.0) a) us) b') u)
                 centre)
          in
          (* A [us] of zero sends the proposal to an infinity, which the range
             test rejects; the count stays finite for the cast. *)
          let count =
            where b (f.le (f.lit 0.0) proposal) (f.min proposal n) (f.lit 0.0)
          in
          let inside = f.both (f.le (f.lit 0.0) proposal) (f.le proposal n) in
          let squeeze = f.both (f.le (f.lit 0.07) us) (f.le v vr) in
          let lhs =
            f.sub
              (f.add (f.log v) log_alpha)
              (f.log (f.add (f.div a (f.mul us us)) b'))
          in
          let accept =
            f.both inside
              (either b squeeze
                 (f.le lhs (f.sub (log_binomial_pmf b f count n p q) log_mode)))
          in
          acc := where b (f.both accept (not_ b !settled)) count !acc;
          settled := either b !settled accept;
          last := count
        done;
        where b !settled !acc !last
      in
      let y = into b D.Int32 (where b small inversion rejection) in
      where b flip (bin b Sub ins.(0) y) y)
