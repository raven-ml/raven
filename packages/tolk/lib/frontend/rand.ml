(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Capture before [open Tolk_uop] shadows it with the uop movement module. *)
module Movement_ops = Movement
open Tolk_uop
module Movement = Movement_ops
module D = Dtype
module T = Tensor

let prod = List.fold_left ( * ) 1

(* Ceiling division for non-negative operands. OCaml's [/] truncates toward
   zero, so [-(-a / b)] would floor instead of ceil. *)
let ceildiv a b = (a + b - 1) / b

(* A scalar literal used as an operand adopts the dtype of the tensor it is
   paired with, keeping a scalar operand at the operand's precision instead of a
   wider default. [~like] is the paired tensor. *)
let uf ~like s =
  let dt = T.dtype like in
  let sdt =
    if D.is_float dt then dt
    else
      match s with
      | T.Sint _ when D.is_int dt -> dt
      | T.Sint _ -> D.default_int
      | T.Sfloat _ -> D.default_float
      | T.Sbool _ -> D.bool
  in
  T.of_uop (Uop.const (T.scalar_const sdt s))

let sf ~like x = uf ~like (T.Sfloat x)
let si ~like n = uf ~like (T.Sint n)

(* Generator state: one seed for the process, plus per-device key and counter
   tensors created lazily on first use. The key holds a device-specific word
   (derived by hashing the device's first-use index) and the seed; the counter
   is a 64-bit position in the random stream stored as two uint32 lanes. *)

let seed = ref (int_of_float (Unix.time ()))
let device_seeds : (string, T.t) Hashtbl.t = Hashtbl.create 4
let device_rng_counters : (string, T.t) Hashtbl.t = Hashtbl.create 4

let manual_seed s =
  seed := s;
  Hashtbl.reset device_seeds;
  Hashtbl.reset device_rng_counters

let device_rng_counter device = Hashtbl.find_opt device_rng_counters device

(* Scalar constant at [t]'s dtype, so unsigned arithmetic stays at the
   operand's width (a Python literal operand adopts the tensor's dtype). *)
let c t v = T.of_uop (Uop.const (T.scalar_const (T.val_dtype t) (T.Sint v)))
let slice t a b = Op.getitem t Movement.[ R (Some a, Some b, None) ]

let uint32_input values =
  let bytes = Bytes.create (4 * Array.length values) in
  Array.iteri
    (fun i v -> Bytes.set_int32_le bytes (i * 4) (Int32.of_int v))
    values;
  Run.of_bytes ~dtype:D.uint32 ~shape:[ Array.length values ] bytes

let next_counter device num =
  if not (Hashtbl.mem device_seeds device) then begin
    let index = Bytes.create 4 in
    Bytes.set_int32_be index 0 (Int32.of_int (Hashtbl.length device_seeds));
    let digest = Tolk.Helpers.sha256 index in
    let key_word = Int32.to_int (Bytes.get_int32_be digest 28) land 0xFFFFFFFF in
    Hashtbl.replace device_seeds device (uint32_input [| key_word; !seed |]);
    Hashtbl.replace device_rng_counters device (uint32_input [| 0; 0 |])
  end;
  let counter = Hashtbl.find device_rng_counters device in
  let num_low = num land 0xFFFFFFFF and num_high = num lsr 32 in
  let new_low = Elementwise.add (slice counter 0 1) (c counter num_low) in
  let new_high =
    Elementwise.add
      (Elementwise.add (slice counter 1 2) (c counter num_high))
      (Elementwise.lt new_low (Op.getitem counter Movement.[ I 0 ]))
  in
  ignore (Op.assign counter (Op.cat new_low [ new_high ]));
  (* Recover the pre-advance position by reading back through the write, so
     the advance replays even when this graph is captured. *)
  let low = Elementwise.sub (slice counter 0 1) (c counter num_low) in
  let high =
    Elementwise.sub
      (Elementwise.sub (slice counter 1 2) (c counter num_high))
      (Elementwise.lt (Op.getitem counter Movement.[ I 0 ]) (c counter num_low))
  in
  (Hashtbl.find device_seeds device, Op.cat low [ high ])

(* One Threefry application: pack two uint32 streams into uint64 counters,
   mix them under [key], and split the result back into two uint32 halves
   concatenated along axis 0. *)
let threefry_random_bits key counts0 counts1 =
  let u64 t = Dtype_ops.cast t D.uint64 in
  let shl32 t = Elementwise.lshift t (c t 32) in
  let mask32 t = Elementwise.bitwise_and t (c t 0xFFFFFFFF) in
  let x = Elementwise.bitwise_or (shl32 (u64 counts1)) (u64 counts0) in
  let key_lane i =
    u64 (Movement.broadcast_to (Op.getitem key Movement.[ I i ]) (T.shape x))
  in
  let x =
    Elementwise.threefry x
      (Elementwise.bitwise_or (shl32 (key_lane 1)) (key_lane 0))
  in
  Op.cat
    (Dtype_ops.cast (mask32 x) D.uint32)
    [ Dtype_ops.cast (mask32 (Elementwise.rshift x (c x 32))) D.uint32 ]

let uint32_max = 0xFFFFFFFF

let random_bits key counter num =
  let low = slice counter 0 1 and high = slice counter 1 2 in
  let rec chunks i acc =
    if i >= num then List.rev acc
    else
      let chunk_num = min (num - i) uint32_max in
      let c_low = Elementwise.add low (c low (i land 0xFFFFFFFF)) in
      let c_high =
        Elementwise.add
          (Elementwise.add high (c high (i lsr 32)))
          (Dtype_ops.cast (Elementwise.lt c_low low) D.uint32)
      in
      let new_key = threefry_random_bits key c_low c_high in
      let counts0 = Op.arange ~dtype:D.uint32 (ceildiv chunk_num 2) in
      let counts1 = Elementwise.add counts0 (c counts0 (ceildiv chunk_num 2)) in
      let bits =
        slice (threefry_random_bits new_key counts0 counts1) 0 chunk_num
      in
      chunks (i + uint32_max) (bits :: acc)
  in
  match chunks 0 [] with
  | [] -> slice counter 0 0
  | first :: rest -> Op.cat first rest

(* Map uniform bits to floats in [0, 1): keep the top mantissa-many bits,
   overlay them on the bit pattern of 1.0 (giving a float in [1, 2)), and
   subtract 1. *)
let bits_to_rand bits shape dt =
  let _, nmant = D.finfo dt in
  let uint_dtype =
    match D.itemsize dt with
    | 1 -> D.uint8
    | 2 -> D.uint16
    | 4 -> D.uint32
    | 8 -> D.uint64
    | _ -> invalid_arg "Rand.bits_to_rand: unsupported float width"
  in
  let uint_bits = Dtype_ops.bitcast bits uint_dtype in
  let float_one_bits =
    Dtype_ops.bitcast
      (Dtype_ops.cast (Creation.const_like uint_bits (T.Sint 1)) dt)
      uint_dtype
  in
  let mantissa =
    Elementwise.bitwise_or
      (Elementwise.rshift uint_bits (c uint_bits (D.bitsize dt - nmant)))
      float_one_bits
  in
  Movement.reshape
    (Elementwise.sub
       (slice (Dtype_ops.bitcast mantissa dt) 0 (prod shape))
       (T.i 1))
    shape

let rand_from key counter shape dt ~contiguous =
  let bits =
    random_bits key counter (ceildiv (prod shape * D.itemsize dt) 4)
  in
  let out = bits_to_rand bits shape dt in
  if contiguous then Elementwise.contiguous out else out

let check_shape name shape =
  if List.exists (fun s -> s < 0) shape then
    invalid_arg (Printf.sprintf "Rand.%s: dimensions must be non-negative" name)

let rand ?dtype ?(contiguous = true) shape =
  let dt = match dtype with Some d -> d | None -> D.default_float in
  if not (D.is_float dt) then
    invalid_arg "Rand.rand: only float dtypes are supported";
  check_shape "rand" shape;
  if D.itemsize dt <> 4 then
    invalid_arg "Rand.rand: only 32-bit float dtypes are supported";
  let device = Run.device_name () in
  let key, counter =
    next_counter device (ceildiv (prod shape * D.itemsize dt) 4)
  in
  rand_from key counter shape dt ~contiguous

let rand_like ?dtype ?contiguous t =
  let dt = match dtype with Some d -> d | None -> T.val_dtype t in
  rand ~dtype:dt ?contiguous (T.shape t)

(* Box-Muller: two uniform draws give one standard normal sample. *)
let randn_like ?dtype t =
  let dt = match dtype with Some d -> d | None -> T.val_dtype t in
  let src = rand ~dtype:D.float32 (2 :: T.shape t) in
  let sel i = Op.getitem src Movement.[ I i ] in
  Dtype_ops.cast
    (Elementwise.mul
       (Elementwise.cos (Elementwise.mul (sel 0) (T.f (2. *. Float.pi))))
       (Elementwise.sqrt
          (Elementwise.mul
             (Elementwise.log (Elementwise.sub (T.f 1.) (sel 1)))
             (T.f (-2.)))))
    dt

let randn ?dtype shape =
  check_shape "randn" shape;
  randn_like ?dtype (Creation.zeros ~buffer:false shape)

let uniform ?(low = 0.) ?(high = 1.) ?dtype shape =
  check_shape "uniform" shape;
  if low >= high then
    invalid_arg
      (Printf.sprintf "Rand.uniform: requires low < high, got low=%g high=%g"
         low high);
  let dt = match dtype with Some d -> d | None -> D.default_float in
  Elementwise.add
    (Dtype_ops.cast
       (Elementwise.mul (T.f (high -. low)) (rand shape))
       dt)
    (T.of_uop (Uop.const (T.scalar_const dt (T.Sfloat low))))

let randint ?(low = 0) ?(high = 10) ?(dtype = D.int32) shape =
  if not (D.is_int dtype) then
    invalid_arg "Rand.randint: dtype must be an integer type";
  if low >= high then
    invalid_arg
      (Printf.sprintf "Rand.randint: requires low < high, got low=%d high=%d"
         low high);
  uniform ~low:(float_of_int low) ~high:(float_of_int high) ~dtype shape

let normal ?(mean = 0.) ?(std = 1.) ?dtype shape =
  if std < 0. then
    invalid_arg (Printf.sprintf "Rand.normal: requires std >= 0, got %g" std);
  let r = randn ?dtype shape in
  Elementwise.add (Elementwise.mul r (sf ~like:r std)) (sf ~like:r mean)

let scaled_uniform ?dtype shape =
  let u = uniform ~low:(-1.) ~high:1. ?dtype shape in
  Elementwise.mul u (sf ~like:u (Float.pow (float_of_int (prod shape)) (-0.5)))

let glorot_uniform ?dtype shape =
  let bound =
    Float.sqrt (6. /. float_of_int (List.hd shape + prod (List.tl shape)))
  in
  uniform ~low:(-.bound) ~high:bound ?dtype shape

let kaiming_uniform ?(a = 0.01) ?dtype shape =
  let bound =
    Float.sqrt (6. /. (1. +. (a *. a)) /. float_of_int (prod (List.tl shape)))
  in
  uniform ~low:(-.bound) ~high:bound ?dtype shape

let kaiming_normal ?(a = 0.01) ?dtype shape =
  let std =
    Float.sqrt (2. /. (1. +. (a *. a)) /. float_of_int (prod (List.tl shape)))
  in
  normal ~std ?dtype shape

let randperm ?(dtype = D.int32) n =
  Dtype_ops.cast (Op.argsort (rand [ n ])) dtype

let multinomial ?(num_samples = 1) ?(replacement = false) t =
  let ndim = T.ndim t in
  if not (1 <= ndim && ndim <= 2) || num_samples <= 0 then
    invalid_arg
      "Rand.multinomial: input must be 1- or 2-dimensional and num_samples \
       positive";
  let weight = if ndim = 1 then Movement.unsqueeze t 0 else t in
  let cols = List.nth (T.shape weight) 1 in
  if (not replacement) && num_samples > cols then
    invalid_arg
      "Rand.multinomial: without replacement, num_samples must not exceed the \
       population size";
  let indices =
    if replacement || num_samples = 1 then
      (* Invert the normalized CDF at uniform samples: an index is the number
         of CDF entries the sample reaches or exceeds. *)
      let cw = Dtype_ops.float (Op.cumsum ~axis:1 weight) in
      let cdf =
        Elementwise.div cw
          (Movement.unsqueeze (Op.getitem cw Movement.[ All; I (-1) ]) 1)
      in
      let rows = List.hd (T.shape cdf) in
      let unif_samples = rand [ num_samples; rows; 1 ] in
      Movement.permute
        (Reduce.sum ~axis:[ 2 ]
           (Elementwise.ge (Movement.expand unif_samples [ -1; -1; cols ]) cdf))
        [ 1; 0 ]
    else
      (* Weighted sampling without replacement (Efraimidis-Spirakis): draw a
         key per element and keep the top-k. *)
      snd
        (Op.topk ~dim:1
           (Elementwise.div
              (Elementwise.log2 (rand_like ~dtype:D.float32 weight))
              weight)
           num_samples)
  in
  Dtype_ops.cast
    (if ndim = 1 then Movement.squeeze ~dim:0 indices else indices)
    D.int32

let dropout ?(p = 0.5) t =
  if not (0. <= p && p <= 1.) then
    invalid_arg (Printf.sprintf "Rand.dropout: p=%g is out of range [0, 1]" p);
  if Tolk.Helpers.Context_var.get Tolk.Helpers.training = 0 || p = 0. then t
  else if p = 1. then Creation.const_like t (T.Sint 0)
  else
    Elementwise.div
      (Elementwise.where
         (Elementwise.contiguous
            (Elementwise.ge
               (rand_like ~dtype:D.default_float ~contiguous:false t)
               (T.f p)))
         t (si ~like:t 0))
      (sf ~like:t (1. -. p))
