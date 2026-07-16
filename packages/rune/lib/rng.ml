(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Explicit splittable PRNG keys.

   A key is an ordinary [|2|] int32 tensor and every sampler is a pure
   function of it: one Threefry-2x32 application (the Nx backend operation)
   over a position-indexed counter, expressed entirely with ordinary Nx
   operations. Keys and samples therefore trace, differentiate, vectorize and
   compile like any other tensor computation, and the same key, dtype and
   shape produce the same values eagerly, under jit, and on every device. *)

type key = (int32, Nx.int32_elt) Nx.t

let shape_string s =
  String.concat "," (Array.to_list (Array.map string_of_int s))

let check_key name k =
  if Nx.shape k <> [| 2 |] then
    invalid_arg
      (Printf.sprintf
         "Rune.Rng.%s: a key is an int32 tensor of shape [2], got shape [%s]"
         name
         (shape_string (Nx.shape k)))

let check_shape name shape =
  if Array.exists (fun d -> d < 0) shape then
    invalid_arg
      (Printf.sprintf "Rune.Rng.%s: invalid shape [%s], dimensions must be \
                       non-negative"
         name (shape_string shape))

(* The low and high 32-bit words of [seed] as the key's two lanes. *)
let key seed =
  Nx.create Nx.int32 [| 2 |]
    [| Int32.of_int (seed asr 32); Int32.of_int seed |]

(* One Threefry application over [n] independent blocks: the key broadcast
   across the rows of an [n; 2] counter whose row [i] holds [(2i, 2i+1)]. *)
let blocks name k n =
  check_key name k;
  let kb =
    Nx.contiguous (Nx.broadcast_to [| n; 2 |] (Nx.reshape [| 1; 2 |] k))
  in
  let ctr = Nx.reshape [| n; 2 |] (Nx.arange Nx.int32 0 (2 * n) 1) in
  Nx_effect.threefry kb ctr

let split ?(n = 2) k =
  if n < 1 then invalid_arg "Rune.Rng.split: n must be at least 1";
  let bits = blocks "split" k n in
  Array.init n (fun i -> Nx.contiguous (Nx.slice [ Nx.I i ] bits))

let fold_in k data =
  check_key "fold_in" k;
  let ctr =
    Nx.create Nx.int32 [| 2 |]
      [| Int32.of_int (data asr 32); Int32.of_int data |]
  in
  Nx_effect.threefry (Nx.contiguous k) ctr

(* Signed int32 bits -> [0, 1): add 2^31 then divide by 2^32, the mapping
   Nx.rand applies to its Threefry output, so eager and compiled samples share
   one definition of the bits-to-float step. *)
let uniform (type b) k (dtype : (float, b) Nx.dtype) shape : (float, b) Nx.t =
  check_shape "uniform" shape;
  let n = Array.fold_left ( * ) 1 shape in
  if n = 0 then Nx.zeros dtype shape
  else
    let bits = Nx.shrink [| (0, n) |] (Nx.flatten (blocks "uniform" k n)) in
    let f32 = Nx.cast Nx.float32 bits in
    let normalized =
      Nx.div
        (Nx.add f32 (Nx.scalar Nx.float32 2147483648.0))
        (Nx.scalar Nx.float32 4294967296.0)
    in
    Nx.reshape shape (Nx.cast dtype normalized)

(* Box-Muller over two uniform draws from independent subkeys, as Nx.randn:
   z = cos(2 pi u1) * sqrt(-2 ln (max (1 - u2) 1e-7)). *)
let normal (type b) k (dtype : (float, b) Nx.dtype) shape : (float, b) Nx.t =
  check_shape "normal" shape;
  if Array.fold_left ( * ) 1 shape = 0 then Nx.zeros dtype shape
  else
    let ks = split k in
    let u1 = uniform ks.(0) Nx.float32 shape in
    let u2 = uniform ks.(1) Nx.float32 shape in
    let angle = Nx.mul u1 (Nx.scalar Nx.float32 (2.0 *. Float.pi)) in
    let u2_safe =
      Nx.maximum (Nx.sub (Nx.ones_like u2) u2) (Nx.scalar Nx.float32 1e-7)
    in
    let r =
      Nx.mul (Nx.cos angle)
        (Nx.sqrt (Nx.mul (Nx.scalar Nx.float32 (-2.0)) (Nx.log u2_safe)))
    in
    Nx.cast dtype r

let randint k ?(high = 10) shape low =
  if low >= high then
    invalid_arg
      (Printf.sprintf "Rune.Rng.randint: invalid range, low=%d >= high=%d" low
         high);
  let u = uniform k Nx.float32 shape in
  Nx.cast Nx.int32
    (Nx.add
       (Nx.mul u (Nx.scalar Nx.float32 (float_of_int (high - low))))
       (Nx.scalar Nx.float32 (float_of_int low)))

let bernoulli k ~p shape =
  if p < 0.0 || p > 1.0 then
    invalid_arg "Rune.Rng.bernoulli: p must be in [0, 1]";
  Nx.less (uniform k Nx.float32 shape) (Nx.scalar Nx.float32 p)
