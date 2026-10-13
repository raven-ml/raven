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

module E = P.E

let u64 v = E.const D.Uint64 v
let two32 = 0x1_0000_0000L

(* The counter of block [i], a uint64 node: the words [(2i, 2i + 1)], which
   repeat from 2^31 blocks on, so the second is xored with [q = i / 2^31], which
   keeps the counters distinct up to 2^63 blocks and is zero below 2^31. *)
let counter i =
  let half = u64 0x8000_0000L in
  let r = E.(i mod half) in
  let lo = E.(r + r) in
  let hi = E.((lo + u64 1L) lxor (i / half)) in
  E.(lo + (hi * u64 two32))

let block ~key i = E.threefry (counter i) key

(* Word [j] of a draw, a uint64 node below 2^32. *)
let word ~key j =
  let two = u64 2L and w = u64 two32 in
  let t = block ~key E.(j / two) in
  E.(where (j mod two = u64 0L) (t mod w) (t / w))

(* The flat index, as a uint64 node, of each element in the last [Array.length
   s] axes of a shape of rank [rank], of extents [s]. *)
let flat ~rank s =
  let r = Array.length s in
  let acc = ref (E.const D.Int64 0L) and weight = ref 1 in
  for c = 0 to r - 1 do
    let at = E.coord (rank - 1 - c) in
    let w = E.const D.Int64 (Int64.of_int !weight) in
    acc := E.(!acc + (at * w));
    weight := !weight * s.(r - 1 - c)
  done;
  E.bitcast D.Uint64 !acc

(* The largest [p] such that every multiple of 2^-p in [0, 1) is a value of
   [dt]: the significand's width where the format reaches 2^-p, else the
   exponent of its least positive value. 53 for float64 down to 3 for
   float8_e5m2, and 1 for float4_e2m1fn, whose values below 1 are 0 and 0.5. *)
let precision (type s) (dt : (float, s) D.t) =
  let f = D.float_format dt in
  min (f.fraction_bits + 1) (1 - snd (Float.frexp (f.min_normal *. f.epsilon)))

(* A sampler's compute dtype. Samplers compute at float64 for float64 and at
   float32 for the other floats. *)
type compute = C : (float, 's) D.t -> compute

let compute (type s) (dt : (float, s) D.t) =
  match dt with D.Float64 -> C D.Float64 | _ -> C D.Float32

(* [x] as a constant of the compute dtype [c]. *)
let lit c x = E.const c (D.of_float c x)

(* [x] rounded once to [dt]. *)
let rounded (type v s w r) (dt : (v, s) D.t) (x : (w, r) E.t) : (v, s) E.t =
  match D.equal_witness (E.dtype x) dt with
  | Some Equal -> x
  | None -> E.cast dt x

(* 53 random bits at position [pos], a uint64 node below 2^53: the low word's
   low 21 bits over the high word, of one block. *)
let bits53 ~key pos =
  let t = block ~key pos and w = u64 two32 in
  let top = E.(t mod w land u64 0x1F_FFFFL) in
  E.((top * w) + (t / w))

(* A draw in [0, 1) at position [pos] with [p] random bits, in the compute dtype
   [c]: at float32, [p] of 24 or fewer, the low [p] bits of a word scaled by
   2^-p; at float64, [p] of 53, one block's 21 + 32 bits scaled by 2^-53. Both
   are exact, so a draw is a multiple of 2^-p below 1. *)
let unit (type s) ~key ~p (c : (float, s) D.t) pos =
  match c with
  | D.Float64 ->
      let scale = E.const c (Float.ldexp 1. (-53)) in
      E.(cast c (bits53 ~key pos) * scale)
  | _ ->
      let mask = u64 (Int64.of_int ((1 lsl p) - 1)) in
      let scale = E.const c (Float.ldexp 1. (-p)) in
      E.(cast c (word ~key pos land mask) * scale)

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

(* A draw of [dt] and shape [s] per key of [k]: [body ~key ~j] is the element
   at flat position [j] of [s], [key] its key's words, reading [params], each
   loaded at the draw's shape, as {!operand}s. *)
let draw (type v e) ~by ?(params = [||]) (k : 'd key) (dt : (v, e) D.t) s
    (body : key:_ -> j:_ -> (v, e) E.t) =
  check_shape ~by s;
  let batch = batch k and w = pack ~by k in
  let shape = Array.append batch s in
  let dtype (Value.Any x) = D.Any (Prim.dtype x) in
  let ins = Iarray.of_array (Array.append [| D.Any D.Uint64 |] (Array.map dtype params)) in
  let out = body ~key:(E.input 0 D.Uint64) ~j:(flat ~rank:(Array.length shape) s) in
  let w = reshape ~by (Array.append batch (Array.make (Array.length s) 1)) w in
  let loads =
    Array.append
      [| Value.Any (stretch ~by shape w) |]
      (Array.map (fun (Value.Any x) -> Value.Any (stretch ~by shape x)) params)
  in
  map ~by shape (P.of_exprs ~ins [| E.any out |]) dt loads

(* Parameter [i] of a draw, [p], as a node of its body: operand 0 is the
   key. *)
let operand i p = E.input (i + 1) (Prim.dtype p)

(* Keys *)

let key seed : 'd key =
  let hi = E.const D.Int32 (Int32.of_int (seed asr 32)) in
  let lo = E.const D.Int32 (Int32.of_int seed) in
  let out = E.(where (coord 0 = const D.Int64 0L) hi lo) in
  map ~by:"Nx.Rng.key" [| 2 |] (P.of_exprs ~ins:[||] [| E.any out |]) D.Int32 [||]

let of_tensor (t : (int32, D.int32_elt, 'd) Value.t) : 'd key =
  let s = Prim.shape t in
  let r = Array.length s in
  if r = 0 || s.(r - 1) <> 2 then
    invalid_argf "Nx.Rng.of_tensor: a key has shape [...; 2], not %a" pp_shape s;
  t

let to_tensor (k : 'd key) : (int32, D.int32_elt, 'd) Value.t = k

(* The keys [f ~key] gives at each element of [shape], whose last axes follow
   [k]'s batch. *)
let keys ~by (k : 'd key) shape f : 'd key =
  let w = pack ~by k in
  let ins : D.any iarray = [| D.Any D.Uint64 |] in
  let out = f ~key:(E.input 0 D.Uint64) in
  let w = stretch ~by shape w in
  let u : (int64, D.uint64_elt, 'd) Value.t =
    map ~by shape (P.of_exprs ~ins [| E.any out |]) D.Uint64 [| Value.Any w |]
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
  keys ~by k [| n |] (fun ~key -> block ~key (E.bitcast D.Uint64 (E.coord 0)))

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
  keys ~by k (batch k) (fun ~key -> E.threefry (u64 c) key)

(* [fold_in] of an index held in data: [i]'s words [(i asr 32, i)] as the
   sign-extended [i] gives them. A batch of indices gives a batch of keys. *)
let fold_in_tensor (k : 'd key) (i : (int32, D.int32_elt, 'd) Value.t) : 'd key
    =
  let by = "Nx.Rng.fold_in_tensor" in
  let w = pack ~by k in
  let shape = Prim.broadcast_shape ~by (batch k) (Prim.shape i) in
  let ins : D.any iarray = [| D.Any D.Uint64; D.Any D.Int32 |] in
  let v = E.(bitcast D.Uint64 (cast D.Int64 (input 1 D.Int32))) in
  let w32 = u64 two32 in
  let out = E.threefry E.((v / w32) + (v mod w32 * w32)) (E.input 0 D.Uint64) in
  let loads =
    [| Value.Any (stretch ~by shape w); Value.Any (stretch ~by shape i) |]
  in
  let u : (int64, D.uint64_elt, 'd) Value.t =
    map ~by shape (P.of_exprs ~ins [| E.any out |]) D.Uint64 loads
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
  draw ~by:"Nx.Rng.bits" (resolve key) D.Int32 s (fun ~key ~j ->
      E.cast D.Int32 (word ~key j))

let uniform ?key dt s =
  let (C c) = compute dt in
  draw ~by:"Nx.Rng.uniform" (resolve key) dt s (fun ~key ~j ->
      rounded dt (unit ~key ~p:(precision dt) c j))

(* The standard normal draw at position [e] of a Box-Muller draw over a [2;
   pairs] uniform draw: the radius from row 0, the angle from row 1, and both of
   the pair kept, the cosines first. [u1] is floored at 2^-p, the smallest draw
   above zero, so that the logarithm is finite and no other draw moves. A node
   of [c], float32 or float64. *)
let gauss ~key ~p ~pairs c e =
  let lit = lit c and pr = u64 (Int64.of_int pairs) in
  let second = E.(pr <= e) in
  let i = E.(where second (e - pr) e) in
  let u1 = unit ~key ~p c i and u2 = unit ~key ~p c E.(i + pr) in
  let u1 = E.maximum u1 (lit (Float.ldexp 1. (-p))) in
  let r = E.(sqrt (lit (-2.) * log u1)) in
  let angle = E.(u2 * lit (2. *. Float.pi)) in
  E.(where second (r * sin angle) (r * cos angle))

let normal ?key dt s =
  let (C c) = compute dt in
  let pairs = (numel s + 1) / 2 in
  draw ~by:"Nx.Rng.normal" (resolve key) dt s (fun ~key ~j ->
      rounded dt (gauss ~key ~p:(precision c) ~pairs c j))

(* Exponential(1) by inverse CDF, from [1 - u], which is never 0. The draw is [0
   - log (1 - u)], where a negation would make -0 of u = 0. *)
let exponential ?key dt s =
  let (C c) = compute dt in
  draw ~by:"Nx.Rng.exponential" (resolve key) dt s (fun ~key ~j ->
      let u = unit ~key ~p:(precision c) c j in
      rounded dt E.(lit c 0. - log (lit c 1. - u)))

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
  draw ~by (resolve key) D.Int32 s (fun ~key ~j ->
      let r = u64 range and w32 = u64 two32 in
      let at i = u64 (Int64.of_int (i * n)) in
      let block i = block ~key E.(at i + j) in
      let hi w = E.(((w / w32 * r) + (w mod w32 * r / w32)) / w32) in
      let first = block 0 in
      let rejected = E.(first * r < u64 threshold) in
      let v = E.(cast D.Uint32 (where rejected (hi (block 1)) (hi first))) in
      E.(bitcast D.Int32 (v + const D.Uint32 (Int32.of_int low))))

(* Parameters *)

(* Parameter [i] of a draw, [p], at the compute dtype [c]. *)
let param c i p = rounded c (operand i p)

(* A float parameter's domain: as a refusal writes it, and the boolean node of
   whether [x], the parameter at its sampler's compute dtype [c], lies in it.
   NaN lies outside each. The compute dtype holds every bound, where a narrow
   parameter dtype would round it. *)
type domain = {
  text : string;
  inside :
    's. (float, 's) D.t -> (float, 's) E.t -> (bool, D.bool_elt) E.t;
}

(* [x] is finite where [x - x] is zero, and NaN elsewhere. *)
let finite c x = E.(x - x <= lit c 0.)

let probability =
  { text = "[0, 1]"; inside = (fun c x -> E.(lit c 0. <= x && x <= lit c 1.)) }

let positive =
  { text = "(0, inf)"; inside = (fun c x -> E.(lit c 0. < x && finite c x)) }

let non_negative =
  { text = "[0, inf)"; inside = (fun c x -> E.(lit c 0. <= x && finite c x)) }

(* Poisson rates: their counts fit int32. *)
let counted =
  {
    text = "[0, 2^31)";
    inside = (fun c x -> E.(lit c 0. <= x && x < lit c 2147483648.));
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
  let (C c) = compute (Prim.dtype p) in
  let ins : D.any iarray = [| D.Any (Prim.dtype p) |] in
  let inside = d.inside c (rounded c (E.input 0 (Prim.dtype p))) in
  let ok =
    map ~by (Prim.shape p) (P.of_exprs ~ins [| E.any inside |]) D.Bool [| Value.Any p |]
  in
  refuse ~by name d.text ~show:(Printf.sprintf "%g") p ok

(* Checks that the counts [n] are not negative. *)
let require_counts ~by name (n : (int32, D.int32_elt, 'd) Value.t) =
  let ins : D.any iarray = [| D.Any D.Int32 |] in
  let inside = E.(const D.Int32 0l <= input 0 D.Int32) in
  let ok =
    map ~by (Prim.shape n) (P.of_exprs ~ins [| E.any inside |]) D.Bool [| Value.Any n |]
  in
  refuse ~by name "[0, inf)" ~show:Int32.to_string n ok

let bernoulli ?key p =
  let by = "Nx.Rng.bernoulli" in
  require ~by "p" probability p;
  let (C c) = compute (Prim.dtype p) in
  draw ~by ~params:[| Value.Any p |] (resolve key) D.Bool (Prim.shape p)
    (fun ~key ~j ->
      (* [p 2^53] is exact at the compute dtype, [p] being at most 1: the draw
         is 53 bits below it, rounded up, at every dtype. *)
      let scaled = E.(param c 0 p * lit c (Float.ldexp 1. 53)) in
      E.(bits53 ~key j < cast D.Uint64 (ceil scaled)))

(* Rejection *)

(* A rejection loop's draw. Acceptance reads data, so the loop has a fixed
   number of rounds: [round r] is round [r]'s proposal and whether it is
   accepted, and the draw is the first proposal accepted. An element no round
   accepts takes [otherwise], or by default the last proposal, whatever its
   test. Built from the last round back, each round is one select. *)
let first_accepted ~rounds ?otherwise round =
  let last, accepted = round (rounds - 1) in
  let acc =
    ref
      (match otherwise with
      | None -> last
      | Some v -> E.where accepted last v)
  in
  for r = rounds - 2 downto 0 do
    let x, accepted = round r in
    acc := E.where accepted x !acc
  done;
  !acc

(* A rejection loop's uniforms: round [r] of element [j] of [n] reads positions
   [r n + j] and [(rounds + r) n + j] of the key's uniform draw. *)
let pair ~key ~p ~n ~rounds c j r =
  let at r =
    let first = u64 (Int64.of_int (r * n)) in
    E.(first + j)
  in
  (unit ~key ~p c (at r), unit ~key ~p c (at (rounds + r)))

(* Gamma *)

(* Marsaglia and Tsang (2000): at a concentration of 1 or more, a normal draw
   [x] is squeezed through [v = (1 + x / √(9d))^3], [d = a - 1/3], and accepted
   against a uniform; below 1, Gamma(a) = Gamma(a + 1) U^(1/a). The acceptance
   test reads data, so a fixed ten rounds are drawn and the first acceptance
   taken. A round accepts more than 95% of proposals, least at a concentration
   of 1, which those below 1 are lifted to; an element no round accepts, about
   one in 10^13, takes the mean of Gamma(boosted), [a + 1] times the shift
   below 1. The proposals take the key's first split key, the tests its second
   and the shift its third, each element [j] of [n] drawing position [r n + j]
   in round [r].

   The draw of Gamma(boosted) and what the shift needs: the concentration [a],
   whether it is below 1, and the shift's uniform floored at 2^-p. *)
let rounds = 10

let marsaglia ~key ~n c a j =
  let p = precision c in
  let lit = lit c in
  let tiny = lit (Float.ldexp 1. (-p)) in
  let keys = Array.init 3 (fun i -> block ~key (u64 (Int64.of_int i))) in
  let below = E.(a < lit 1.) in
  let boosted = E.(where below (a + lit 1.) a) in
  let d = E.(boosted - lit (1. /. 3.)) in
  let squeeze = E.(recip (sqrt (lit 9. * d))) in
  let pairs = ((rounds * n) + 1) / 2 in
  let round r =
    let first = u64 (Int64.of_int (r * n)) in
    let pos = E.(first + j) in
    let x = gauss ~key:keys.(0) ~p ~pairs c pos in
    let u = unit ~key:keys.(1) ~p c pos in
    let t = E.(lit 1. + (squeeze * x)) in
    let v = E.(t * (t * t)) in
    (* [v] can be non-positive, where the logarithm is undefined: it is floored,
       and [positive] rejects it. *)
    let positive = E.(lit 0. < v) in
    let log_v = E.(log (maximum v tiny)) in
    let bound = E.((lit 0.5 * (x * x)) + (d + (-(d * v) + (d * log_v)))) in
    let log_u = E.(log (maximum u tiny)) in
    (E.(d * v), E.(positive && log_u < bound))
  in
  let draw = first_accepted ~rounds ~otherwise:boosted round in
  let boost = E.maximum (unit ~key:keys.(2) ~p c j) tiny in
  (draw, below, boost)

let gamma ?key a =
  let by = "Nx.Rng.gamma" in
  require ~by "a" positive a;
  let dt = Prim.dtype a in
  let (C c) = compute dt in
  let s = Prim.shape a in
  draw ~by ~params:[| Value.Any a |] (resolve key) dt s (fun ~key ~j ->
      let a' = param c 0 a in
      let acc, below, boost = marsaglia ~key ~n:(numel s) c a' j in
      let shift = E.(boost ** recip a') in
      rounded dt E.(where below (acc * shift) acc))

(* Beta(a, b) = G(a) / (G(a) + G(b)) for independent gammas from the key's two
   split keys, formed as [1 / (1 + exp d)], [d = log G(b) - log G(a)], so that
   two gammas that underflow keep their ratio. A gamma below a concentration of
   1 is [acc U^(1/a)], so [d = log acc_b - log acc_a + q_b / b - q_a / a] with
   [q = log U] there and 0 elsewhere. Each [q / a] alone is [-∞] at a tiny
   concentration, and two of them make NaN, so the shifts are scaled by [s = min
   a b] before they meet: [(s/b) q_b - (s/a) q_a] is finite, and its quotient by
   [s] is [d]'s shift, an infinity only where the draw is 0 or 1. The quotient
   is a division, where [q · (1 / s)] would make NaN of a zero [q]. *)
let beta ?key a b =
  let by = "Nx.Rng.beta" in
  require ~by "a" positive a;
  require ~by "b" positive b;
  let dt = Prim.dtype a in
  let (C c) = compute dt in
  let s = Prim.broadcast_shape ~by (Prim.shape a) (Prim.shape b) in
  let n = numel s in
  draw ~by ~params:[| Value.Any a; Value.Any b |] (resolve key) dt s
    (fun ~key ~j ->
      let lit = lit c in
      let ka = block ~key (u64 0L) and kb = block ~key (u64 1L) in
      let a = param c 0 a and b = param c 1 b in
      let gamma key x =
        let acc, below, boost = marsaglia ~key ~n c x j in
        (E.log acc, E.(where below (log boost) (lit 0.)))
      in
      let la, qa = gamma ka a and lb, qb = gamma kb b in
      let least = E.minimum a b in
      let shift = E.(((least / b * qb) - (least / a * qa)) / least) in
      let d = E.(lb - la + shift) in
      rounded dt E.(lit 1. / (lit 1. + exp d)))

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

let wrapped_cauchy c kappa =
  let lit = lit c in
  (* Each regime reads a concentration of its own range, so neither divides by
     zero or overflows where the other is selected. *)
  let large = E.(lit 1. <= kappa) in
  let small_kappa = E.where large (lit 0.) kappa in
  let large_kappa = E.where large kappa (lit 1.) in
  let two_kappa = E.(lit 2. * small_kappa) in
  let root = E.(sqrt (lit 1. + (two_kappa * two_kappa))) in
  let tau = E.(lit 1. + root) in
  let r = E.(sqrt (lit 2. * tau)) in
  let dd = E.(tau + r) in
  let gap = E.(lit 1. + recip (root + two_kappa) + r) in
  let e = E.(lit 0.5 / large_kappa) in
  let h = E.(sqrt (lit 1. + (e * e))) in
  let d = E.(h + e + sqrt (lit 2. * (e * (e + h)))) in
  let w = E.((e * sqrt e / (h + lit 1.)) + sqrt e + sqrt (lit 2. * (e + h))) in
  let rho = E.(where large (recip d) (two_kappa / dd)) in
  let one_minus_rho = E.(where large (sqrt e * w / d) (gap / dd)) in
  let a = E.(where large (w * w / (lit 4. * d)) (gap * gap / (lit 4. * dd))) in
  (rho, one_minus_rho, a)

let von_mises ?key kappa =
  let by = "Nx.Rng.von_mises" in
  require ~by "k" non_negative kappa;
  let dt = Prim.dtype kappa in
  let (C c) = compute dt in
  let s = Prim.shape kappa in
  let n = numel s in
  draw ~by ~params:[| Value.Any kappa |] (resolve key) dt s
    (fun ~key ~j ->
      let p = precision c and lit = lit c in
      let rho, one_minus_rho, a = wrapped_cauchy c (param c 0 kappa) in
      let one_plus_rho = E.(lit 1. + rho) in
      let lift = E.(a * (one_plus_rho * one_plus_rho)) in
      let denom phi =
        let cs = E.cos phi in
        E.((one_minus_rho * one_minus_rho) + (lit 4. * rho * (cs * cs)))
      in
      let rounds = von_mises_rounds in
      let round r =
        let u, v = pair ~key ~p ~n ~rounds c j r in
        let phi = E.((u - lit 0.5) * lit Float.pi) in
        let y = E.(lift / denom phi) in
        let squeeze = E.(v < y * (lit 2. - y)) in
        let bound = E.(lit (-1.) <= log (y / v) - y) in
        (phi, E.(squeeze || bound))
      in
      let phi = first_accepted ~rounds round in
      let half = E.(abs (sin phi) * one_minus_rho / sqrt (denom phi)) in
      let theta = E.(sign phi * (lit 2. * asin (minimum half (lit 1.)))) in
      rounded dt theta)

(* Counts *)

let not_ x = E.(x = const D.Bool false)

(* log (2π) / 2 rounded to a double, written out: [Float.log] calls the C
   library, whose last bit may differ by system, and on Windows by domain. *)
let half_log_two_pi = 0x1.d67f1c864beb5p-1

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

let stirlerr lit y =
  let main y = E.(((y + lit 0.5) * log y) - y + lit half_log_two_pi) in
  let tail y =
    let y2 = E.(y * y) in
    E.((lit (1.0 /. 12.0) - ((lit (1.0 /. 360.0) - (lit (1.0 /. 1260.0) / y2)) / y2)) / y)
  in
  let shifted = E.(y + lit 8.0) in
  let product = ref E.(y + lit 1.0) in
  for i = 2 to 8 do
    product := E.(!product * (y + lit (float_of_int i)))
  done;
  let small = E.(main shifted + tail shifted - log !product - main y) in
  E.(where (y < lit 8.0) small (tail y))

let deviance lit x m d =
  let s = E.(x + m) in
  let v = E.(d / s) in
  let v2 = E.(v * v) in
  let acc = ref E.(d * v) in
  let term = ref E.(lit 2.0 * x * v) in
  for j = 1 to 6 do
    term := E.(!term * v2);
    let odd = lit (float_of_int ((2 * j) + 1)) in
    acc := E.(!acc + (!term / odd))
  done;
  let direct = E.((x * log (x / m)) - x + m) in
  E.(where (abs d < lit 0.1 * s) !acc direct)

(* The Poisson log pmf: [-bd0 (k, rate) - log (2π k) / 2 - stirlerr k]. *)
let log_poisson_pmf lit k rate =
  let spread = E.(lit 0.5 * log (lit (2.0 *. Float.pi) * k)) in
  E.(where (k < lit 0.5) (-rate)
       (-deviance lit k rate (k - rate) - spread - stirlerr lit k))

(* The binomial log pmf of [n] trials of probability [p], [q = 1 - p]:

   stirlerr n - stirlerr k - stirlerr (n - k) - bd0 (k, n p) - bd0 (n - k, n q)
   - log (2π k (n - k) / n) / 2,

   both deviances taking their difference from [n p - k]. The counts [0] and [n]
   have their own branches; the rest is read at a count strictly inside them. *)
let log_binomial_pmf lit k n p q =
  let inside = E.(maximum (minimum k (n - lit 1.0)) (lit 1.0)) in
  let rest = E.(n - inside) in
  let d = E.(inside - (n * p)) in
  let saddle =
    E.(stirlerr lit n - stirlerr lit inside - stirlerr lit rest
       - deviance lit inside (n * p) d
       - deviance lit rest (n * q) (-d))
  in
  let spread = E.(log (lit (2.0 *. Float.pi) * inside) + log1p (-(inside / n))) in
  E.(where (k < lit 0.5) (n * log1p (-p))
       (where (n - lit 0.5 < k) (n * log p) (saddle - (lit 0.5 * spread))))

(* Two regimes with a fixed round count each, chosen per element, so the shape
   of the computation does not depend on the rate. Below 10, inversion: one
   uniform against the cumulative pmf over 48 terms, the mass beyond them 4e-18
   at rate 10. The pmf is [exp (-rate)] at 0, then each term the last times
   [rate / i], so a rate of 0 puts the whole mass on the count 0. From 10 up,
   Hörmann's transformed rejection with squeeze (PTRS): sixteen rounds, each
   accepting 0.75 to 0.89, leave 2e-10 of an element unaccepted, which takes
   its last proposal, a draw from the hat with mean near the rate. The elements
   the inversion owns see a rate of 1e5 there, away from the hat's poles
   below 1. Inversion reads the key's first split key, rejection its second. *)
let poisson_inversion_rounds = 48
let poisson_rejection_rounds = 16

let poisson ?key rate =
  let by = "Nx.Rng.poisson" in
  require ~by "rate" counted rate;
  let dt = Prim.dtype rate in
  let (C c) = compute dt in
  let s = Prim.shape rate in
  let n = numel s in
  draw ~by ~params:[| Value.Any rate |] (resolve key) D.Int32 s
    (fun ~key ~j ->
      let p = precision c and lit = lit c in
      let rate = param c 0 rate in
      let k0 = block ~key (u64 0L) and k1 = block ~key (u64 1L) in
      let small = not_ E.(lit 10.0 <= rate) in
      let inversion =
        let u = unit ~key:k0 ~p c j in
        let pmf = ref E.(exp (-rate)) in
        let cdf = ref (lit 0.0) and count = ref (lit 0.0) in
        for i = 0 to poisson_inversion_rounds - 1 do
          if i > 0 then pmf := E.(!pmf * (rate / lit (float_of_int i)));
          cdf := E.(!cdf + !pmf);
          count := E.(!count + where (!cdf < u) (lit 1.0) (lit 0.0))
        done;
        !count
      in
      let rejection =
        let rounds = poisson_rejection_rounds in
        let lam = E.where small (lit 1e5) rate in
        let b = E.(lit 0.931 + (lit 2.53 * sqrt lam)) in
        let a = E.(lit (-0.059) + (lit 0.02483 * b)) in
        let log_inv_alpha = E.(log (lit 1.1239 + (lit 1.1328 / (b - lit 3.4)))) in
        let vr = E.(lit 0.9277 - (lit 3.6224 / (b - lit 2.0))) in
        let round r =
          let u, v = pair ~key:k1 ~p ~n ~rounds c j r in
          let u = E.(u - lit 0.5) in
          let us = E.(lit 0.5 - abs u) in
          let proposal = E.(floor ((((lit 2.0 * a / us) + b) * u) + lam + lit 0.43)) in
          (* A [us] of zero sends the proposal to an infinity and an infinite
             rate makes it NaN: the tests reject both, and the count the cast
             reads stays finite. *)
          let count =
            E.(where (lit 0.0 <= proposal) (minimum proposal (lit 2147483647.0)) (lit 0.0))
          in
          let squeeze = E.(lit 0.07 <= us && v <= vr) in
          let reject = E.(proposal < lit 0.0 || (us < lit 0.013 && us < v)) in
          let lhs = E.(log v + log_inv_alpha - log ((a / (us * us)) + b)) in
          let accepted = E.(not_ reject && lhs <= log_poisson_pmf lit count lam) in
          (count, E.(squeeze || accepted))
        in
        first_accepted ~rounds round
      in
      E.(cast D.Int32 (where small inversion rejection)))

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
  let (C c) = compute dt in
  let s = Prim.broadcast_shape ~by (Prim.shape count) (Prim.shape prob) in
  let len = numel s in
  draw ~by ~params:[| Value.Any count; Value.Any prob |] (resolve key) D.Int32 s
    (fun ~key ~j ->
      let bits = precision c and lit = lit c in
      let count = operand 0 count in
      let n = E.cast c count in
      let p = param c 1 prob in
      let flip = E.(lit 0.5 < p) in
      let p = E.(where flip (lit 1.0 - p) p) in
      let small = not_ E.(lit 10.0 <= n * p) in
      let k0 = block ~key (u64 0L) and k1 = block ~key (u64 1L) in
      let inversion =
        (* Term [i] holds [log (n (n - 1) … (n - i + 1) / i! (p / q)^i)]:
           the logarithms of the ratios [(n - l) p / ((l + 1) q)], summed over
           [l < i]. *)
        let odds = E.(log p - log1p (-p)) in
        let base = E.(n * log1p (-p)) in
        let ratios = ref (lit 0.0) and cdf = ref (lit 0.0) in
        let cdfs =
          Array.init binomial_inversion_rounds (fun i ->
              if i > 0 then begin
                let l = float_of_int (i - 1) in
                let left = E.(maximum (n - lit l) (lit 0.0)) in
                let ratio = E.(log (left / lit (l +. 1.0))) in
                ratios := E.(!ratios + (ratio + odds))
              end;
              cdf := E.(!cdf + exp (base + !ratios));
              !cdf)
        in
        let u = E.(unit ~key:k0 ~p:bits c j * !cdf) in
        Array.fold_left
          (fun acc cdf -> E.(acc + where (cdf < u) (lit 1.0) (lit 0.0)))
          (lit 0.0) cdfs
      in
      let rejection =
        let rounds = binomial_rejection_rounds in
        let n = E.where small (lit 1000.0) n and p = E.where small (lit 0.5) p in
        let q = E.(lit 1.0 - p) in
        let mean = E.(n * p) in
        let spq = E.(sqrt (mean * q)) in
        let b = E.(lit 1.15 + (lit 2.53 * spq)) in
        let a = E.(lit (-0.0873) + (lit 0.0248 * b) + (lit 0.01 * p)) in
        let centre = E.(mean + lit 0.5) in
        let log_alpha = E.(log ((lit 2.83 + (lit 5.1 / b)) * spq)) in
        let vr = E.(lit 0.92 - (lit 4.2 / b)) in
        let mode = E.(floor ((n + lit 1.0) * p)) in
        let log_mode = log_binomial_pmf lit mode n p q in
        let round r =
          let u, v = pair ~key:k1 ~p:bits ~n:len ~rounds c j r in
          let u = E.(u - lit 0.5) in
          let us = E.(lit 0.5 - abs u) in
          let proposal = E.(floor ((((lit 2.0 * a / us) + b) * u) + centre)) in
          (* A [us] of zero sends the proposal to an infinity, which the range
             test rejects; the count stays finite for the cast. *)
          let count =
            E.(where (lit 0.0 <= proposal) (minimum proposal n) (lit 0.0))
          in
          let inside = E.(lit 0.0 <= proposal && proposal <= n) in
          let squeeze = E.(lit 0.07 <= us && v <= vr) in
          let lhs = E.(log v + log_alpha - log ((a / (us * us)) + b)) in
          let pmf = E.(log_binomial_pmf lit count n p q - log_mode) in
          (count, E.(inside && (squeeze || lhs <= pmf)))
        in
        first_accepted ~rounds round
      in
      let y = E.(cast D.Int32 (where small inversion rejection)) in
      E.(where flip (count - y) y))
