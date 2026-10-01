(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk

let dtype = Ops.dtype
let is_float u = Dtype.is_float (dtype u)
let int u n = Ops.const_like u (`Int (Bigint.of_int n))
let size u axis = List.nth (Ops.max_shape u) axis
let width dts dt = List.find (fun d -> Dtype.itemsize d = Dtype.itemsize dt) dts
let signed = width Dtype.sints
let unsigned = width Dtype.uints

(* The vector [u] laid along [axis] of [rank] axes, the others of one
   element. *)
let along rank axis u =
  Ops.reshape u
    (List.init rank (fun i -> Ops.Int (if i = axis then size u 0 else 1)))

(* Order keys

   A float orders as the signed integer of its bits with a negative float's
   magnitude bits flipped: the greater float is the greater integer, [-0.] is
   just below [+0.], and subnormals keep the order that a float comparison may
   flush. Every NaN takes the greatest integer or the least, which no number
   takes: a sort, in either direction, and [argmax] key a NaN greatest; only
   [argmin] keys it least. An integer or a boolean is its own key. *)

let flip k =
  Ops.where
    (Ops.lt k (int k 0))
    (Ops.bitwise_xor k (Ops.const_like k (Dtype.max (dtype k) :> Dtype.const)))
    k

let keys ~nan x =
  if not (is_float x) then x
  else
    let k = signed (dtype x) in
    let bits = Ops.bitcast x k in
    let extreme =
      match nan with `Greatest -> Dtype.max k | `Least -> Dtype.min k
    in
    Ops.where (Ops.ne x x)
      (Ops.const_like bits (extreme :> Dtype.const))
      (flip bits)

(* Reductions and scans

   Signed integers accumulate unsigned, since C, Metal and CUDA leave their
   overflow undefined. A float sum adds [+0.] to its result: a kernel starts a
   loop's accumulator from [+0.], but sums the terms alone when no loop is left,
   over an axis of one element or one it unrolls whole. A float extreme is NaN
   where a NaN is among its elements, and otherwise the target's maximum, of the
   negated elements for a minimum, whose zero result may have either sign. An
   integer extreme is a maximum of the integers, or of their complements for a
   minimum. *)

let accumulator dt =
  let acc = Dtype.sum_acc dt in
  if List.exists (Dtype.equal acc) Dtype.sints then unsigned acc else acc

let accumulated f op x =
  let dt = dtype x in
  let r = f op (Ops.cast x (accumulator dt)) in
  let r =
    if Op.equal op Op.Add && Dtype.is_float dt then
      Ops.add r (Ops.const_like r (`Float 0.))
    else r
  in
  Ops.cast r dt

let extreme f ~greatest x =
  if is_float x then
    let m = if greatest then f Op.Max x else Ops.neg (f Op.Max (Ops.neg x)) in
    Ops.where (f Op.Max (Ops.ne x x)) (Ops.const_like m (`Float Float.nan)) m
  else if greatest then f Op.Max x
  else Ops.bitwise_not (f Op.Max (Ops.bitwise_not x))

let combine f (k : Nx_backend.reduce) x =
  match k with
  | Sum -> accumulated f Op.Add x
  | Prod -> accumulated f Op.Mul x
  | Max -> extreme f ~greatest:true x
  | Min -> extreme f ~greatest:false x

let reduce k ~axes x = combine (fun op u -> Ops.rop u op axes) k x
let scan k ~axis x = combine (fun op u -> Ops.cumalu u axis op) k x

(* Arg-reductions *)

(* [argmax x axis] is the position of the first greatest element of [x] along
   [axis]: of the elements equal to the maximum, the one that counts down
   furthest from the axis's length. *)
let argmax x axis =
  let n = size x axis in
  let m = Ops.eq x (Ops.unsqueeze (Ops.rop x Op.Max [ axis ]) axis) in
  let down = Ops.arange ~start:n ~step:(-1) 0 in
  let down =
    Ops.reshape down
      (Ops.Int n :: List.init (Ops.ndim x - axis - 1) (fun _ -> Ops.Int 1))
  in
  Ops.cast
    (Ops.sub (Ops.int n) (Ops.rop (Ops.mul m down) Op.Max [ axis ]))
    Dtype.Int64

let arg_reduce (k : Nx_backend.arg_reduce) ~axis x =
  match k with
  | Argmax -> argmax (keys ~nan:`Greatest x) axis
  | Argmin -> argmax (Ops.bitwise_not (keys ~nan:`Least x)) axis

(* Sorts

   A bitonic network of maxima and minima sorts integers exactly, but is not
   stable. The positions come from the network run over each key packed above
   its position: packed integers are distinct, so the order the network gives
   them is the stable one. *)

let bit_length n =
  let rec go n b = if n = 0 then b else go (n lsr 1) (b + 1) in
  go n 0

let halves u axis =
  match Ops.split ~axis u [ 1; 1 ] with [ a; b ] -> (a, b) | _ -> assert false

(* [bitonic ~descending x axis] is the integers [x] sorted along [axis]. The
   axis, padded to a power of two with elements that sort last, is split into
   axes of two elements, one per stage of the network. *)
let bitonic ~descending x axis =
  let n = size x axis in
  if n <= 1 then x
  else
    let stages = bit_length (n - 1) and shape = Ops.shape x in
    let fill =
      if descending then Dtype.min (dtype x) else Dtype.max (dtype x)
    in
    let x =
      Ops.pad_to
        ~value:(fill :> Dtype.const)
        x
        (List.init (Ops.ndim x) (fun d ->
             if d = axis then Some (Ops.Int (1 lsl stages)) else None))
    in
    let x = Ops.unflatten x axis (List.init stages (fun _ -> Ops.Int 2)) in
    let r = Ops.ndim x in
    (* A stage's crossover reverses the second half of each of its boxes, so
       that comparisons all in one direction merge them; it is undone after. *)
    let crossover stage x =
      let c = axis + stages - stage - 1 in
      let blue, green = halves x c in
      let flipped =
        List.init (stage + List.length shape - axis) (fun i -> r - 1 - i)
      in
      Ops.cat ~axis:c blue [ Ops.flip green flipped ]
    in
    let compare x sub =
      let p = axis + stages - sub - 1 in
      let top, bottom = halves x p in
      let larger = Ops.maximum top bottom
      and smaller = Ops.minimum top bottom in
      Ops.contiguous
        (if descending then Ops.cat ~axis:p larger [ smaller ]
         else Ops.cat ~axis:p smaller [ larger ])
    in
    let stage x s =
      let x = if s < stages then Ops.contiguous (crossover s x) else x in
      let x = List.fold_left compare x (List.init s (fun i -> s - 1 - i)) in
      if s < stages then crossover s x else x
    in
    let x = List.fold_left stage x (List.init stages (fun i -> i + 1)) in
    Ops.shrink_to
      (Ops.flatten ~start:axis ~stop:(axis + stages - 1) x)
      (List.map Option.some shape)

(* [ordered k] is the key [k] as the unsigned integer of its width that orders
   as it does: a signed key with its sign bit flipped. *)
let ordered k =
  let dt = dtype k in
  if Dtype.is_bool dt then Ops.cast k Dtype.Uint8
  else if Dtype.is_unsigned dt then k
  else
    Ops.bitcast
      (Ops.bitwise_xor k (Ops.const_like k (Dtype.min dt :> Dtype.const)))
      (unsigned dt)

(* [positions ~descending axis high] is the stable positions that sort the
   [int64] keys [high], below [2^32], along [axis]. Each position is
   complemented for a descending sort, so that equal keys keep their order. The
   packed integers take a kernel of their own: fused into the padding of an axis
   that is not a power of two, the positions no longer fold to an index. *)
let positions ~descending axis high =
  let n = size high axis in
  let low = bit_length (n - 1) in
  let mask = Ops.const ~dtype:Int64 (`Int (Bigint.of_int ((1 lsl low) - 1))) in
  let tie r = if descending then Ops.sub mask r else r in
  let ranks = along (Ops.ndim high) axis (Ops.arange ~dtype:Int64 n) in
  let packed = Ops.bitwise_or (Ops.shl high (Ops.int low)) (tie ranks) in
  tie (Ops.bitwise_and (bitonic ~descending (Ops.contiguous packed) axis) mask)

(* Bits

   An element selected among zeros is summed over the elements' bit patterns as
   unsigned integers: a float sum would turn [-0.] into [+0.], and quiet a
   signalling NaN. *)

let bits u =
  if Dtype.equal (dtype u) Bool then Ops.cast u Uint8
  else Ops.bitcast u (unsigned (dtype u))

let of_bits dt b =
  if Dtype.equal dt Bool then Ops.cast b Bool else Ops.bitcast b dt

let pick mask b =
  let selected = Ops.where mask b (Ops.int 0) in
  Ops.rop selected Op.Add [ Ops.ndim selected - 1 ]

(* The positions are compared in [int64]: where [x] is read from memory, tolk
   folds the selection into a load gated on [p]'s range and computes the load's
   address in 32 bits, which a narrowing before the comparison would only
   repeat. *)
let take x axis p =
  let hot =
    Ops.eq (Ops.unsqueeze p (-1)) (Ops.arange ~dtype:(dtype p) (size x axis))
  in
  of_bits (dtype x)
    (pick hot (Ops.transpose (Ops.unsqueeze (bits x) (-1)) axis (Ops.ndim x)))

let argsort ~descending ~axis x =
  if size x axis <= 1 then Ops.const_like ~dtype:Int64 x (`Int Bigint.zero)
  else
    let k = ordered (keys ~nan:`Greatest x) in
    let positions = positions ~descending axis in
    if Dtype.itemsize (dtype k) <= 4 then positions (Ops.cast k Int64)
    else
      (* A 64-bit key sorts in two passes, its low half first: the second pass
         sorts the high halves in the first pass's order, and being stable keeps
         that order among equal high halves. *)
      let half h = Ops.cast h Int64 in
      let lo =
        half
          (Ops.bitwise_and k
             (Ops.const_like k (`Int (Bigint.of_int 0xffff_ffff))))
      in
      let first = positions lo in
      take first axis
        (positions (take (half (Ops.shr k (Ops.int 32))) axis first))

let sort ~descending ~axis x =
  if size x axis <= 1 then x else take x axis (argsort ~descending ~axis x)
