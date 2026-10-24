(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* One splittable Threefry-2x32 generator. A key is an int32 array whose last
   axis holds its two words, and a batch of keys adds leading axes. Every draw
   is one map: each element computes, from its position in the drawn shape, the
   Threefry blocks its words come from, so a draw is a formula of its key and
   computes on the set of whatever reads it.

   Block [i] encrypts the counter [i], a uint64, under the key: Threefry is a
   bijection of the counter under a key, so distinct counters give independent
   blocks (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", 2011).
   Word [j] of a draw is word [j mod 2] of block [j / 2]. A key's two int32
   words are one uint64 to the Threefry kind, the first word low. *)

module D = Nx_array.Dtype
module M = Nx_array.Move
module P = Nx_kernel.Prog

type 'd key = (int32, D.int32_elt, 'd) Value.t

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let pp_shape = Prim.pp_shape

(* Programs *)

module E = P.E

let u64 v = E.const D.Uint64 v
let uint i = u64 (Int64.of_int i)
let two32 = 0x1_0000_0000L
let block ~key i = E.threefry i key

(* A block's two words, uint64 nodes below 2^32: word 0 is its low half and
   word 1 its high half. *)
let word0 t = E.(t land u64 0xFFFF_FFFFL)
let word1 t = E.(t / u64 two32)

(* Word [j] of a draw, a uint64 node below 2^32. *)
let word ~key j =
  let t = block ~key E.(j / u64 2L) in
  E.(where (j land u64 1L = u64 0L) (word0 t) (word1 t))

(* The largest [p] such that every multiple of 2^-p in [0, 1) is a value of
   [dt]: the significand's width where the format reaches 2^-p, else the
   exponent of its least positive value. 53 for float64 down to 3 for
   float8_e5m2, and 1 for float4_e2m1fn, whose values below 1 are 0 and 0.5. *)
let precision (type s) (dt : (float, s) D.t) =
  let f = D.float_format dt in
  min (f.fraction_bits + 1) (1 - snd (Float.frexp (f.min_normal *. f.epsilon)))

(* [x] as a constant of the compute dtype [c], the format a sampler computes
   in ({!Nx_kernel.float_carrier}). *)
let lit c x = E.const c (D.of_float c x)

(* [x] rounded once to [dt]. *)
let rounded (type v s w r) (dt : (v, s) D.t) (x : (w, r) E.t) : (v, s) E.t =
  match D.equal_witness (E.dtype x) dt with
  | Some Equal -> x
  | None -> E.cast dt x

(* 53 random bits of the block [t], a uint64 node below 2^53: its low word's
   low 21 bits over its high word. *)
let bits53 t = E.(((t land u64 0x1F_FFFFL) * u64 two32) + word1 t)

(* Uniform draws in [0, 1) of the compute dtype [c], exact, so each is a
   multiple of 2^-p below 1: at float32, from the word [w], its low [p] bits
   scaled by 2^-p, [p] being 24 or fewer; at float64, from the block [t], 53
   bits scaled by 2^-53. *)
let of_word ~p c w =
  let mask = u64 (Int64.of_int ((1 lsl p) - 1)) in
  let scale = E.const c (Float.ldexp 1. (-p)) in
  E.(cast c (w land mask) * scale)

let of_block c t =
  let scale = E.const c (Float.ldexp 1. (-53)) in
  E.(cast c (bits53 t) * scale)

(* The draw in [0, 1) at position [j], with [p] random bits: at float64 from
   block [j], else from word [j], so that two positions share a block. *)
let unit (type s) ~key ~p (c : (float, s) D.t) j : (float, s) E.t =
  match c with
  | D.Float64 -> of_block c (block ~key j)
  | _ -> of_word ~p c (word ~key j)

(* The [n] draws in [0, 1) of element [j] of a sampler that takes [n] each,
   with [p] random bits. Element [j] owns blocks [b j] to [b j + b - 1], [b]
   being [n] at float64, whose draws take a block each, else [n / 2] rounded
   up, whose draws take a word each, the low then the high. *)
let uniforms (type s) ~key ~p (c : (float, s) D.t) j n : (float, s) E.t array =
  let per = match c with D.Float64 -> 1 | _ -> 2 in
  let b = (n + per - 1) / per in
  let first = if b = 1 then j else E.(j * uint b) in
  let block m = block ~key (if m = 0 then first else E.(first + uint m)) in
  match c with
  | D.Float64 -> Array.init n (fun m -> of_block c (block m))
  | _ ->
      let blocks = Array.init b block in
      Array.init n (fun m ->
          let t = blocks.(m / 2) in
          of_word ~p c (if m mod 2 = 0 then word0 t else word1 t))

(* Programs by dtype. A sampler's program depends on its dtypes alone, so it
   is lowered once per dtype, into the slot of the dtype's code, and kept.
   Programs are immutable data: two domains that race to fill a slot build
   equal programs, and either may stay. *)
let programs () = Array.init (List.length D.all) (fun _ -> Atomic.make None)

(* A sampler program's operands: the element's key, its position in the
   draw, both uint64, then the sampler's parameters. *)
let key_operand = 0
let position_operand = 1
let first_param = 2

(* The program of [table] for [dt], whose element at position [j] of its draw
   is [body ~key ~j], over parameters of dtypes [ins]. *)
let lowered table dt ins body =
  let slot = table.(D.code dt) in
  match Atomic.get slot with
  | Some p -> p
  | None ->
      let ins = Array.append [| D.Any D.Uint64; D.Any D.Uint64 |] ins in
      let key = E.input key_operand D.Uint64 in
      let j = E.input position_operand D.Uint64 in
      let p = P.of_exprs ~ins:(Iarray.of_array ins) [| E.any (body ~key ~j) |] in
      Atomic.set slot (Some p);
      p

(* Draws *)

let move ~by mv x = Eval.eval ~by (Value.Move (mv, x))

let reshape ~by s x =
  if Prim.has_shape x s then x else move ~by (M.Reshape (Iarray.of_array s)) x

(* [x] at the shape [s], with leading axes: [x]'s own shape [t] aligned
   right. *)
let stretch ~by s x =
  if Prim.has_shape x s then x else move ~by (M.Broadcast (Iarray.of_array s)) x

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
  let loads =
    Iarray.init (Array.length loads) (fun i ->
        let (Value.Any x) = loads.(i) in
        Value.Plain x)
  in
  let x, () =
    Eval.eval ~by (Value.Map { layout; prog; outs = Value.[ dt ]; loads })
  in
  x

(* The flat index in C order, a uint64 node, of each element in the last
   [Array.length s] axes, of extents [s], of a shape of rank [rank]. *)
let position ~rank s =
  let r = Array.length s in
  let flat = ref None and weight = ref 1 in
  for c = r - 1 downto 0 do
    let at = E.coord (rank - r + c) in
    let w = E.const D.Int64 (Int64.of_int !weight) in
    let term = if !weight = 1 then at else E.(at * w) in
    flat := Some (match !flat with None -> term | Some f -> E.(f + term));
    weight := !weight * s.(c)
  done;
  match !flat with None -> u64 0L | Some f -> E.bitcast D.Uint64 f

(* A draw of [dt] and shape [s] per key of [k], by the sampler program [prog]:
   one map over the key's batch then [s], each element reading its key, its
   position in [s], and [params], each stretched to the draw's shape. *)
let draw ~by ?(params = [||]) (k : 'd key) dt s prog =
  let batch = batch k in
  let shape = Array.append batch s in
  Prim.check_shape ~by shape;
  let w = pack ~by k in
  let w = reshape ~by (Array.append batch (Array.make (Array.length s) 1)) w in
  let loads =
    Array.append
      [| Value.Any (stretch ~by shape w) |]
      (Array.map (fun (Value.Any x) -> Value.Any (stretch ~by shape x)) params)
  in
  let position = position ~rank:(Array.length shape) s in
  let prog = P.bind prog position_operand (E.any position) in
  map ~by shape prog dt loads

(* Parameter [i] of a draw, of dtype [dt], as a node of its program. *)
let operand i dt = E.input (first_param + i) dt

(* Keys *)

let key_by ~by seed : 'd key =
  let hi = E.const D.Int32 (Int32.of_int (seed asr 32)) in
  let lo = E.const D.Int32 (Int32.of_int seed) in
  let out = E.(where (coord 0 = const D.Int64 0L) hi lo) in
  map ~by [| 2 |] (P.of_exprs ~ins:[||] [| E.any out |]) D.Int32 [||]

let key seed = key_by ~by:"Nx.Rng.key" seed

let of_tensor (t : (int32, D.int32_elt, 'd) Value.t) : 'd key =
  let s = Prim.shape t in
  let r = Array.length s in
  if r = 0 || s.(r - 1) <> 2 then
    invalid_argf "Nx.Rng.of_tensor: a key has shape [...; 2], not %a" pp_shape s;
  t

let to_tensor (k : 'd key) : (int32, D.int32_elt, 'd) Value.t = k

let walk c k = of_tensor (Ptree.Walk.tensor c k)

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
let fold_in_by ~by (k : 'd key) data : 'd key =
  let hi = Int64.logand (Int64.of_int (data asr 32)) 0xFFFF_FFFFL in
  let lo = Int64.logand (Int64.of_int data) 0xFFFF_FFFFL in
  let c = Int64.logor hi (Int64.shift_left lo 32) in
  keys ~by k (batch k) (fun ~key -> E.threefry (u64 c) key)

let fold_in k data = fold_in_by ~by:"Nx.Rng.fold_in" k data

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

let next_key_by ~by : 'd key =
  match Effect.perform Next with
  | root, c -> fold_in_by ~by (every ~by root) c
  | exception Effect.Unhandled Next ->
      let seed, count = Domain.DLS.get unscoped in
      let c = !count in
      incr count;
      fold_in_by ~by (key_by ~by seed) c

let next_key () = next_key_by ~by:"Nx.Rng.next_key"

(* The key a sampler [by] draws from: [key], lent so that the draw kills no
   donation of it ({!Donate.read}), or the next one. A dead [key] raises
   numbered as its operand after the sampler's [after] values: its draw reads
   the key first. *)
let resolve ~by ~after = function
  | Some k -> Donate.read ~by after k
  | None -> next_key_by ~by

let place p k = Eval.place ~by:"Nx.Rng.place" p k

(* Samplers *)

let bits_programs = programs ()

let bits ?key s =
  let by = "Nx.Rng.bits" in
  let prog =
    lowered bits_programs D.Int32 [||] (fun ~key ~j ->
        E.cast D.Int32 (word ~key j))
  in
  draw ~by (resolve ~by ~after:0 key) D.Int32 s prog

let uniform_programs = programs ()

let uniform ?key dt s =
  let by = "Nx.Rng.uniform" in
  let prog =
    lowered uniform_programs dt [||] (fun ~key ~j ->
        let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
        rounded dt (unit ~key ~p:(precision dt) c j))
  in
  draw ~by (resolve ~by ~after:0 key) dt s prog

(* The Box-Muller transform of the draws [u1] and [u2] in [0, 1) with [p] bits:
   two independent standard normal draws, the cosine's then the sine's, of the
   radius from [u1] and the angle from [u2]. [u1] is floored at 2^-p, the
   smallest draw above zero, so that the logarithm is finite and no other draw
   moves. Nodes of [c], float32 or float64. *)
let box_muller ~p c u1 u2 =
  let lit = lit c in
  let u1 = E.maximum u1 (lit (Float.ldexp 1. (-p))) in
  let r = E.(sqrt (lit (-2.) * log u1)) in
  let angle = E.(u2 * lit (2. *. Float.pi)) in
  (E.(r * cos angle), E.(r * sin angle))

(* Elements [2q] and [2q + 1] are one Box-Muller transform's pair, from the
   two draws in [0, 1) of pair [q]. *)
let normal_programs = programs ()

let normal ?key dt s =
  let by = "Nx.Rng.normal" in
  let prog =
    lowered normal_programs dt [||] (fun ~key ~j ->
        let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
        let p = precision c in
        let u = uniforms ~key ~p c E.(j / u64 2L) 2 in
        let x, y = box_muller ~p c u.(0) u.(1) in
        rounded dt E.(where (j land u64 1L = u64 0L) x y))
  in
  draw ~by (resolve ~by ~after:0 key) dt s prog

(* Exponential(1) by inverse CDF, from [1 - u], which is never 0. The draw is [0
   - log (1 - u)], where a negation would make -0 of u = 0. *)
let exponential_programs = programs ()

let exponential ?key dt s =
  let by = "Nx.Rng.exponential" in
  let prog =
    lowered exponential_programs dt [||] (fun ~key ~j ->
        let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
        let u = unit ~key ~p:(precision c) c j in
        rounded dt E.(lit c 0. - log (lit c 1. - u)))
  in
  draw ~by (resolve ~by ~after:0 key) dt s prog

let fits v = v >= -0x8000_0000 && v <= 0x7FFF_FFFF

(* Lemire's multiply-shift: a 64-bit block [w] times the range [r] is [w r = hi
   2^64 + lo], and [hi] is uniform over [\[0, r)] once the [2^64 mod r] values
   of [lo] below it are rejected. A rejection, at most one in 2^32, takes the
   next block: element [j] owns blocks [2j] and [2j + 1], and the second's [hi]
   stands whatever its [lo]. [hi] comes from 32-bit halves, [r] being at most
   2^32 - 1: [hi = (w_hi r + (w_lo r) / 2^32) / 2^32], each term below 2^64.
   [low] is added in uint32, where the sum wraps into the int32 it names. The
   range, [2^64 mod r] and [low] are operands, bound per call. *)
let randint_programs = programs ()

let randint ?key ?(low = 0) ~high s =
  let by = "Nx.Rng.randint" in
  if low >= high then invalid_argf "%s: low %d is not below high %d" by low high;
  if not (fits low && fits high) then
    invalid_argf "%s: [%d, %d) does not fit in int32" by low high;
  let ins = D.[| Any Uint64; Any Uint64; Any Uint32 |] in
  let prog =
    lowered randint_programs D.Int32 ins (fun ~key ~j ->
        let r = operand 0 D.Uint64 and threshold = operand 1 D.Uint64 in
        let first = E.(j * u64 2L) in
        let w = block ~key first in
        let rejected = E.(w * r < threshold) in
        let w = E.where rejected (block ~key E.(first + u64 1L)) w in
        let hi = E.(((word1 w * r) + (word0 w * r / u64 two32)) / u64 two32) in
        E.(bitcast D.Int32 (cast D.Uint32 hi + operand 2 D.Uint32)))
  in
  let range = Int64.of_int (high - low) in
  let threshold = Int64.unsigned_rem (Int64.neg range) range in
  (* Each bind renumbers the operands after the one it binds, so the last is
     bound first and the others keep their indices. *)
  let bind i x prog = P.bind prog (first_param + i) (E.any x) in
  let prog =
    prog
    |> bind 2 (E.const D.Uint32 (Int32.of_int low))
    |> bind 1 (u64 threshold)
    |> bind 0 (u64 range)
  in
  draw ~by (resolve ~by ~after:0 key) D.Int32 s prog

(* Parameters *)

(* Parameter [i] of a draw, of dtype [dt], at the compute dtype [c]. *)
let param c i dt = rounded c (operand i dt)

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
  let fail i (v, ()) =
    Invalid_argument
      (Format.asprintf "%s: %s at %a is %s, not in %s" by name pp_shape i
         (show v) text)
  in
  Eval.eval ~by (Value.Check { ok; data = [ p ]; fail })

(* Checks the float parameter [p] of a sampler computing at [c] against [d]. *)
let require ~by name d (p : (float, 's, 'd) Value.t) =
  let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier (Prim.dtype p) in
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

(* A sampler checks its parameters before it draws from them, so it reads
   them in several operations ({!Donate.read}). *)
let reading ~by ?key p f = f ?key (Donate.read ~by 0 p)

let reading2 ~by ?key a b f =
  let a = Donate.read ~by 0 a in
  f ?key a (Donate.read ~by 1 b)

(* [p 2^53] is exact at the compute dtype, [p] being at most 1: the draw is 53
   bits of block [j] below it, rounded up, at every dtype. *)
let bernoulli_programs = programs ()

let bernoulli_by ~by ?key p =
  require ~by "p" probability p;
  let dt = Prim.dtype p in
  let prog =
    lowered bernoulli_programs dt [| D.Any dt |] (fun ~key ~j ->
        let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
        let scaled = E.(param c 0 dt * lit c (Float.ldexp 1. 53)) in
        E.(bits53 (block ~key j) < cast D.Uint64 (ceil scaled)))
  in
  let key = resolve ~by ~after:1 key in
  draw ~by ~params:[| Value.Any p |] key D.Bool (Prim.shape p) prog

let bernoulli ?key p =
  let by = "Nx.Rng.bernoulli" in
  reading ~by ?key p (bernoulli_by ~by)

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

(* Gamma *)

(* Marsaglia and Tsang (2000): at a concentration of 1 or more, a normal draw
   [x] is squeezed through [v = (1 + x / √(9d))^3], [d = a - 1/3], and accepted
   against a uniform; below 1, Gamma(a) = Gamma(a + 1) U^(1/a). The acceptance
   test reads data, so a fixed ten rounds are drawn and the first acceptance
   taken. A round accepts more than 95% of proposals, least at a concentration
   of 1, which those below 1 are lifted to; an element no round accepts, about
   one in 10^13, takes the mean of Gamma(boosted), [a + 1] times the shift
   below 1.

   The draw of Gamma(boosted) and what the shift needs: the concentration [a],
   whether it is below 1, and the shift's uniform floored at 2^-p, from the
   draws in [0, 1) [u]: rounds [r] and [r + 1], [r] even, take the normals of
   one Box-Muller transform from [u.(r)] and [u.(r + 1)], so [rounds] is
   even; round [r] takes its test from [u.(rounds + r)], and the shift
   [u.(2 rounds)]. *)
let rounds = 10
let gamma_uniforms = (2 * rounds) + 1

let marsaglia ~p c a (u : _ array) =
  let lit = lit c in
  let tiny = lit (Float.ldexp 1. (-p)) in
  let below = E.(a < lit 1.) in
  let boosted = E.(where below (a + lit 1.) a) in
  let d = E.(boosted - lit (1. /. 3.)) in
  let squeeze = E.(recip (sqrt (lit 9. * d))) in
  let normals =
    Array.init (rounds / 2) (fun i -> box_muller ~p c u.(2 * i) u.((2 * i) + 1))
  in
  let round r =
    let x = (if r mod 2 = 0 then fst else snd) normals.(r / 2) in
    let t = E.(lit 1. + (squeeze * x)) in
    let v = E.(t * (t * t)) in
    (* [v] can be non-positive, where the logarithm is undefined: it is floored,
       and [positive] rejects it. *)
    let positive = E.(lit 0. < v) in
    let log_v = E.(log (maximum v tiny)) in
    let bound = E.((lit 0.5 * (x * x)) + (d + (-(d * v) + (d * log_v)))) in
    let test = u.(rounds + r) in
    let log_u = E.(log (maximum test tiny)) in
    (E.(d * v), E.(positive && log_u < bound))
  in
  let draw = first_accepted ~rounds ~otherwise:boosted round in
  let boost = E.maximum u.(2 * rounds) tiny in
  (draw, below, boost)

(* The logarithm of the gamma draw [j] of concentration [a], as [la + q / a]:
   [la] the logarithm of Gamma(boosted), and [q] the logarithm of the shift's
   uniform below a concentration of 1, 0 elsewhere. Apart, [q / a] stays
   unformed where a tiny [a] would make it [-inf]. *)
let log_gamma ~key ~p c j a =
  let u = uniforms ~key ~p c j gamma_uniforms in
  let acc, below, boost = marsaglia ~p c a u in
  (E.log acc, E.(where below (log boost) (lit c 0.)))

let gamma_programs = programs ()

let gamma_by ~by ?key a =
  require ~by "a" positive a;
  let dt = Prim.dtype a in
  let prog =
    lowered gamma_programs dt [| D.Any dt |] (fun ~key ~j ->
        let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
        let p = precision c in
        let a = param c 0 dt in
        let u = uniforms ~key ~p c j gamma_uniforms in
        let acc, below, boost = marsaglia ~p c a u in
        let shift = E.(boost ** recip a) in
        rounded dt E.(where below (acc * shift) acc))
  in
  let key = resolve ~by ~after:1 key in
  draw ~by ~params:[| Value.Any a |] key dt (Prim.shape a) prog

let gamma ?key a =
  let by = "Nx.Rng.gamma" in
  reading ~by ?key a (gamma_by ~by)

(* Beta(a, b) = G(a) / (G(a) + G(b)) for independent gammas, element [j]'s
   being the {!gamma} draws [2j] and [2j + 1] of its key, formed as [1 / (1 +
   exp d)], [d = log G(b) - log G(a)], so that two gammas that underflow keep
   their ratio. A gamma below
   a concentration of 1 is [acc U^(1/a)], so [d = log acc_b - log acc_a + q_b /
   b - q_a / a] with [q = log U] there and 0 elsewhere. Each [q / a] alone is
   [-∞] at a tiny concentration, and two of them make NaN, so the shifts are
   scaled by [s = min a b] before they meet: [(s/b) q_b - (s/a) q_a] is finite,
   and its quotient by [s] is [d]'s shift, an infinity only where the draw is 0
   or 1. The quotient is a division, where [q · (1 / s)] would make NaN of a
   zero [q]. *)
let beta_programs = programs ()

let beta_by ~by ?key a b =
  require ~by "a" positive a;
  require ~by "b" positive b;
  let dt = Prim.dtype a in
  let s = Prim.broadcast_shape ~by (Prim.shape a) (Prim.shape b) in
  let prog =
    lowered beta_programs dt [| D.Any dt; D.Any dt |] (fun ~key ~j ->
        let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
        let p = precision c and lit = lit c in
        let a = param c 0 dt and b = param c 1 dt in
        let even = E.(j * u64 2L) in
        let la, qa = log_gamma ~key ~p c even a in
        let lb, qb = log_gamma ~key ~p c E.(even + u64 1L) b in
        let least = E.minimum a b in
        let shift = E.(((least / b * qb) - (least / a * qa)) / least) in
        let d = E.(lb - la + shift) in
        rounded dt E.(lit 1. / (lit 1. + exp d)))
  in
  let key = resolve ~by ~after:2 key in
  draw ~by ~params:[| Value.Any a; Value.Any b |] key dt s prog

let beta ?key a b =
  let by = "Nx.Rng.beta" in
  reading2 ~by ?key a b (beta_by ~by)

(* Von Mises *)

(* Best and Fisher's rejection (1979) from a wrapped Cauchy envelope of
   parameter [rho]: a proposal from one uniform, accepted against a second with
   the squeeze [y (2 - y) > v] or the bound [log (y / v) + 1 - y >= 0].
   Acceptance is lowest as the concentration grows, near 0.66, so twenty rounds
   leave 5e-10 of an element unaccepted, which takes its last proposal, a draw
   from the envelope. Round [r] reads the element's draws in [0, 1) [2r] and
   [2r + 1].

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

let von_mises_programs = programs ()

let von_mises_by ~by ?key kappa =
  require ~by "k" non_negative kappa;
  let dt = Prim.dtype kappa in
  let prog =
    lowered von_mises_programs dt [| D.Any dt |] (fun ~key ~j ->
        let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
        let p = precision c and lit = lit c in
        let rho, one_minus_rho, a = wrapped_cauchy c (param c 0 dt) in
        let one_plus_rho = E.(lit 1. + rho) in
        let lift = E.(a * (one_plus_rho * one_plus_rho)) in
        let denom phi =
          let cs = E.cos phi in
          E.((one_minus_rho * one_minus_rho) + (lit 4. * rho * (cs * cs)))
        in
        let rounds = von_mises_rounds in
        let u = uniforms ~key ~p c j (2 * rounds) in
        let round r =
          let u, v = (u.(2 * r), u.((2 * r) + 1)) in
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
  in
  let params = [| Value.Any kappa |] in
  draw ~by ~params (resolve ~by ~after:1 key) dt (Prim.shape kappa) prog

let von_mises ?key kappa =
  let by = "Nx.Rng.von_mises" in
  reading ~by ?key kappa (von_mises_by ~by)

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
   below 1. Inversion reads the element's draw in [0, 1) [0], and rejection's
   round [r] its draws [1 + 2r] and [2 + 2r]. *)
let poisson_inversion_rounds = 48
let poisson_rejection_rounds = 16

let poisson_programs = programs ()

let poisson_by ~by ?key rate =
  require ~by "rate" counted rate;
  let dt = Prim.dtype rate in
  let prog =
    lowered poisson_programs dt [| D.Any dt |] (fun ~key ~j ->
        let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
        let p = precision c and lit = lit c in
        let rate = param c 0 dt in
        let draws = uniforms ~key ~p c j (1 + (2 * poisson_rejection_rounds)) in
        let small = not_ E.(lit 10.0 <= rate) in
        let inversion =
          let u = draws.(0) in
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
            let u, v = (draws.(1 + (2 * r)), draws.(2 + (2 * r))) in
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
  in
  let params = [| Value.Any rate |] in
  draw ~by ~params (resolve ~by ~after:1 key) D.Int32 (Prim.shape rate) prog

let poisson ?key rate =
  let by = "Nx.Rng.poisson" in
  reading ~by ?key rate (poisson_by ~by)

(* Binomial(n, p) counts the failures of [1 - p] where [p] exceeds one half, so
   the regimes see [p <= 1/2], chosen per element by the mean [n p]. Below 10,
   inversion over 48 terms: they follow from [(1 - p)^n] by the ratios [(n - j)
   p / ((j + 1) (1 - p))], summed in logarithms, and a uniform scaled by their
   sum meets their running sum, so the count never passes [n]; the mass beyond
   is below 1e-16. From 10 up, Hörmann's transformed rejection (BTRS, 1993):
   eighteen rounds, each accepting at least 0.71, leave 2e-10 of an element
   unaccepted, which takes its last proposal. The elements the inversion owns
   see [n = 1000, p = 1/2] there. Inversion reads the element's draw in [0, 1)
   [0], and rejection's round [r] its draws [1 + 2r] and [2 + 2r]. *)
let binomial_inversion_rounds = 48
let binomial_rejection_rounds = 18

let binomial_programs = programs ()

let binomial_by ~by ?key (count : (int32, D.int32_elt, 'd) Value.t) prob =
  require_counts ~by "n" count;
  require ~by "p" probability prob;
  let dt = Prim.dtype prob in
  let s = Prim.broadcast_shape ~by (Prim.shape count) (Prim.shape prob) in
  let prog =
    lowered binomial_programs dt [| D.Any D.Int32; D.Any dt |] (fun ~key ~j ->
        let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
        let bits = precision c and lit = lit c in
        let count = operand 0 D.Int32 in
        let n = E.cast c count in
        let p = param c 1 dt in
        let flip = E.(lit 0.5 < p) in
        let p = E.(where flip (lit 1.0 - p) p) in
        let small = not_ E.(lit 10.0 <= n * p) in
        let n_draws = 1 + (2 * binomial_rejection_rounds) in
        let draws = uniforms ~key ~p:bits c j n_draws in
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
          let u = E.(draws.(0) * !cdf) in
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
            let u, v = (draws.(1 + (2 * r)), draws.(2 + (2 * r))) in
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
  in
  let params = [| Value.Any count; Value.Any prob |] in
  draw ~by ~params (resolve ~by ~after:2 key) D.Int32 s prog

let binomial ?key count prob =
  let by = "Nx.Rng.binomial" in
  reading2 ~by ?key count prob (binomial_by ~by)

(* Truncated normal *)

(* Inverse-CDF truncation: one uniform [u] per draw and no acceptance test, so
   that the cost does not depend on the interval's mass. The draw [x] has [ndtr
   x = (1 - u) ndtr lower + u ndtr upper], and equally [Q x = (1 - u) Q lower +
   u Q upper] for the mass [Q x = ndtr (-x)] above it. Each is a sum of positive
   terms, which keeps the relative precision of its parts, held as masses for
   the central route and as logarithms for the tails, so that no bound is too
   far out. The smaller of the two, at most 1/2, is inverted: right of zero
   through [Q], left of it through [ndtr], so that both tails of an interval
   about zero keep their digits. This follows SciPy's [truncnorm], which
   chooses the side from the bounds where this chooses it per draw.

   [u] is floored at 2^-p, its least positive value, so that an open lower side
   gives a finite draw. A draw within its error of a bound can round past it,
   so it is clamped into the interval. *)

let not_nan = { text = "[-inf, inf]"; inside = (fun _ x -> E.(x = x)) }

(* A bound's masses below and above it, [ndtr b] and [Q b], then their
   logarithms, at the compute dtype: one map at the bound's own shape. *)
let mass_programs = programs ()

let masses ~by (b : (float, 's, 'd) Value.t) =
  let dt = Prim.dtype b in
  let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
  let slot = mass_programs.(D.code dt) in
  let prog =
    match Atomic.get slot with
    | Some p -> p
    | None ->
        let x = rounded c (E.input 0 dt) in
        let positive, n, log_n, log_m = Special.ndtr_halves x in
        let m = E.(lit c 1. - n) in
        let side y z = E.(any (where positive y z)) in
        let p =
          P.of_exprs ~ins:[| D.Any dt |]
            [| side m n; side n m; side log_m log_n; side log_n log_m |]
        in
        Atomic.set slot (Some p);
        p
  in
  let layout = Nx_array.Layout.contiguous (Prim.shape b) in
  let below, (above, (log_below, (log_above, ()))) =
    Eval.eval ~by
      (Value.Map
         {
           layout;
           prog;
           outs = Value.[ c; c; c; c ];
           loads = [| Value.Plain b |];
         })
  in
  Value.
    [| Any below; Any above; Any log_below; Any log_above |]

(* [log (exp a + exp b)], [-inf] where both are. *)
let log_sum c a b =
  let m = E.maximum a b in
  let finite = E.(where (lit c Float.neg_infinity < m) m (lit c 0.)) in
  E.(m + log1p (exp (minimum a b - finite)))

let truncated_normal_programs = programs ()

let truncated_normal_by ~by ?key lower upper =
  require ~by "lower" not_nan lower;
  require ~by "upper" not_nan upper;
  let dt = Prim.dtype lower in
  let s = Prim.broadcast_shape ~by (Prim.shape lower) (Prim.shape upper) in
  let ins =
    let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
    Array.append [| D.Any dt; D.Any dt |] (Array.make 8 (D.Any c))
  in
  let prog =
    lowered truncated_normal_programs dt ins (fun ~key ~j ->
        let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
        let p = precision c and lit = lit c in
        let mass i = operand i c in
        let below_lo = mass 2 and above_lo = mass 3 in
        let log_below_lo = mass 4 and log_above_lo = mass 5 in
        let below_hi = mass 6 and above_hi = mass 7 in
        let log_below_hi = mass 8 and log_above_hi = mass 9 in
        let u = E.maximum (unit ~key ~p c j) (lit (Float.ldexp 1. (-p))) in
        let v = E.(lit 1. - u) in
        let mix lo hi = E.((v * lo) + (u * hi)) in
        let right = mix above_lo above_hi and left = mix below_lo below_hi in
        let on_right = E.(right < left) in
        let side y z = E.where on_right y z in
        let q = side right left in
        let log_q =
          log_sum c
            E.(log v + side log_above_lo log_below_lo)
            E.(log u + side log_above_hi log_below_hi)
        in
        (* [log_q] is [-inf] only where both masses on its side underflow, both
           bounds far out on it: the draw is then the nearer bound, onto which a
           draw of 0 clamps. *)
        let finite = E.(lit Float.neg_infinity < log_q) in
        let log_half = -0x1.62e42fefa39efp-1 in
        let y =
          Special.ndtri_mass
            E.(q - lit 0.5)
            (E.where finite log_q (lit log_half))
        in
        let x = E.(where finite (where on_right (-y) y) (lit 0.)) in
        let lower = param c 0 dt and upper = param c 1 dt in
        let least = E.minimum lower upper and most = E.maximum lower upper in
        rounded dt E.(minimum (maximum x least) most))
  in
  let params =
    Array.concat
      [
        [| Value.Any lower; Value.Any upper |];
        masses ~by lower;
        masses ~by upper;
      ]
  in
  draw ~by ~params (resolve ~by ~after:2 key) dt s prog

let truncated_normal ?key lower upper =
  let by = "Nx.Rng.truncated_normal" in
  reading2 ~by ?key lower upper (truncated_normal_by ~by)

(* Reductions along a draw's last axis *)

(* [x] reduced along the last axis of [shape], its program's output folded by
   [monoid] in [c], kept as an axis of 1. *)
let along_last ~by shape prog monoid c loads =
  let r = Array.length shape in
  let loads =
    Iarray.init (Array.length loads) (fun i ->
        let (Value.Any x) = loads.(i) in
        Value.Plain (stretch ~by shape x))
  in
  let x, () =
    Eval.eval ~by
      (Value.Reduce
         {
           layout = Nx_array.Layout.contiguous shape;
           axes = [| r - 1 |];
           prog;
           reductions = Value.[ Monoid (monoid, 0, c) ];
           loads;
         })
  in
  reshape ~by (Array.mapi (fun a e -> if a = r - 1 then 1 else e) shape) x

(* Dirichlet *)

(* A row's gamma draws normalised in logarithms. Draw [j] is [G_j = exp (la_j +
   q_j / a_j)] ({!log_gamma}), and [q_j / a_j] overflows to [-inf] at a tiny
   concentration, so the logarithms are scaled by [s], the least of 1 and the
   row's concentrations: [z_j = s la_j + (s / a_j) q_j] is finite, since [s /
   a_j <= 1], and [log G_j = z_j / s]. The draw is then [e_j / Σ e] for [e_j =
   exp ((z_j - max z) / s)], each at most 1 and one of them 1, so that no row is
   NaN or all zero. *)
let dirichlet_programs = programs ()

let dirichlet_by ~by ?key a =
  let s = Prim.shape a in
  let r = Array.length s in
  if r = 0 then invalid_argf "%s: concentrations of shape [] have no axis" by;
  require ~by "a" positive a;
  let key = resolve ~by ~after:1 key in
  let dt = Prim.dtype a in
  let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
  let shape = Array.append (batch key) s in
  if s.(r - 1) = 0 then
    map ~by shape (P.of_exprs ~ins:[||] [| E.(any (const dt 0.)) |]) dt [||]
  else
    let least =
      let x = E.minimum (rounded c (E.input 0 dt)) (lit c 1.) in
      along_last ~by s
        (P.of_exprs ~ins:[| D.Any dt |] [| E.any x |])
        Nx_kernel.Min c [| Value.Any a |]
    in
    let prog =
      lowered dirichlet_programs dt [| D.Any dt; D.Any c |] (fun ~key ~j ->
          let p = precision c in
          let a = param c 0 dt and least = operand 1 c in
          let la, q = log_gamma ~key ~p c j a in
          E.((least * la) + (least / a * q)))
    in
    let z = draw ~by ~params:[| Value.Any a; Value.Any least |] key c s prog in
    let top =
      along_last ~by shape
        (P.of_node ~ins:[| D.Any c |] (In 0))
        Nx_kernel.Max c [| Value.Any z |]
    in
    let ins : D.any iarray = [| D.Any c; D.Any c; D.Any c |] in
    let e = E.(exp ((input 0 c - input 1 c) / input 2 c)) in
    let loads = [| Value.Any z; Value.Any top; Value.Any least |] in
    let sum =
      along_last ~by shape
        (P.of_exprs ~ins [| E.any e |])
        Nx_kernel.Sum c loads
    in
    let out = rounded dt E.(e / input 3 c) in
    let loads =
      Array.map
        (fun (Value.Any x) -> Value.Any (stretch ~by shape x))
        (Array.append loads [| Value.Any sum |])
    in
    map ~by shape
      (P.of_exprs ~ins:[| D.Any c; D.Any c; D.Any c; D.Any c |] [| E.any out |])
      dt loads

let dirichlet ?key a =
  let by = "Nx.Rng.dirichlet" in
  reading ~by ?key a (dirichlet_by ~by)

(* Categorical *)

(* Gumbel-max: the argmax of the logits plus Gumbel noise [-log e], [e] an
   exponential draw, is a draw of the distribution they describe. [e] comes from
   [b] random bits [k], 53 at float64 and 32 otherwise: [-log1p (-lo)] for [lo =
   (k + 1/2) 2^-b] below 1/2, else [-log hi] for [hi = (2^b - 1 - k + 1/2)
   2^-b], so that it keeps its relative precision near 0, where the noise that
   wins lies, and is never 0 or infinite. The noise is then bounded by [(b + 1)
   ln 2] above and [-ln ((b + 1) ln 2)] below. *)
let gumbel (type s) ~key (c : (float, s) D.t) j : (float, s) E.t =
  let b, k =
    match c with
    | D.Float64 -> (53, bits53 (block ~key j))
    | _ -> (32, word ~key j)
  in
  let scale = lit c (Float.ldexp 1. (-b)) in
  let half = lit c 0.5 in
  let top = u64 (Int64.sub (Int64.shift_left 1L b) 1L) in
  let lo = E.((cast c k + half) * scale) in
  let hi = E.((cast c (top - k) + half) * scale) in
  let half_range = u64 (Int64.shift_left 1L (b - 1)) in
  let low = E.(k < half_range) in
  let e = E.(where low (lit c 0. - log1p (-lo)) (lit c 0. - log hi)) in
  E.(-log e)

let categorical_programs = programs ()

let categorical ?key ?(axis = -1) logits =
  let by = "Nx.Rng.categorical" in
  let s = Prim.shape logits in
  let dt = Prim.dtype logits in
  let a =
    Prim.axis ~by (Array.length s) axis (fun ppf ->
        Format.fprintf ppf "%s %a" (D.name dt) pp_shape s)
  in
  if s.(a) = 0 then
    invalid_argf "%s: axis %d of %s %a has no category" by axis (D.name dt)
      pp_shape s;
  Prim.alive ~by 0 logits;
  let key = resolve ~by ~after:1 key in
  let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
  let prog =
    lowered categorical_programs dt [| D.Any dt |] (fun ~key ~j ->
        E.(param c 0 dt + gumbel ~key c j))
  in
  let noisy = draw ~by ~params:[| Value.Any logits |] key c s prog in
  let shape = Prim.shape noisy in
  let (largest, at), () =
    Eval.eval ~by
      (Value.Reduce
         {
           layout = Nx_array.Layout.contiguous shape;
           axes = [| Array.length (batch key) + a |];
           prog = P.of_node ~ins:[| D.Any c |] (In 0);
           reductions = Value.[ Arg (Nx_kernel.Max, 0, c) ];
           loads = [| Value.Plain noisy |];
         })
  in
  (* The noise is finite, so a row's largest sum is finite exactly where its
     largest logit is, and is that logit where it is not. *)
  let ins : D.any iarray = [| D.Any c |] in
  let inside = finite c (E.input 0 c) in
  let ok =
    map ~by (Prim.shape largest)
      (P.of_exprs ~ins [| E.any inside |])
      D.Bool [| Value.Any largest |]
  in
  refuse ~by "the largest logit" "(-inf, inf)" ~show:(Printf.sprintf "%g")
    largest ok;
  at

(* Permutation *)

(* The positions that sort one Threefry block per element: 64-bit words, so
   that collisions, which the stable sort leaves in order, are rare. *)
let permutation_programs = programs ()

let permutation ?key n =
  let by = "Nx.Rng.permutation" in
  if n < 0 then invalid_argf "%s: n is %d, not at least 0" by n;
  let prog =
    lowered permutation_programs D.Uint64 [||] (fun ~key ~j -> block ~key j)
  in
  let words = draw ~by (resolve ~by ~after:0 key) D.Uint64 [| n |] prog in
  let axis = Array.length (Prim.shape words) - 1 in
  let _, positions =
    Eval.eval ~by
      (Value.Sort { axis; descending = false; k = None; x = words })
  in
  positions
