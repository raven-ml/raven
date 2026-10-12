(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module B = Rig.Buffer
module S = Nx_dtype.Scalar
module A = Bigarray.Array1

let bits dt = S.bitsize (S.of_dtype dt)
let bytes dt n = ((n * bits dt) + 7) / 8
let length dt b = B.length b * 8 / bits dt

(* Elements of [bits] < 8 bits, [8 / bits] to a byte, the first in the low bits:
   element [i] is bits [i * bits mod 8] and up of byte [i * bits / 8]. *)

let sub ba bits i =
  let bit = i * bits in
  (A.unsafe_get ba (bit lsr 3) lsr (bit land 7)) land ((1 lsl bits) - 1)

let set_sub ba bits i v =
  let bit = i * bits in
  let m = ((1 lsl bits) - 1) lsl (bit land 7) in
  let byte = A.unsafe_get ba (bit lsr 3) in
  A.unsafe_set ba (bit lsr 3)
    (byte land lnot m lor ((v lsl (bit land 7)) land m))

let in_bounds fn n i =
  if i < 0 || i >= n then
    invalid_arg
      (Printf.sprintf "Nx_array.Elements.%s: index %d of %d elements" fn i n)

let u8 = Bigarray.int8_unsigned

(* Creation *)

(* The bits of a fresh buffer past its last element, which belong to no element,
   are 0: what fills the buffer then decides its every byte. *)
let create dt n =
  let b = B.create Rig.host (bytes dt n) in
  let bits = bits dt in
  if bits < 8 && n * bits land 7 <> 0 then
    B.write b u8 (fun ba -> A.set ba (n * bits / 8) 0);
  b

(* Access *)

let read (type a b) (dt : (a, b) Nx_dtype.t) b (f : (int -> a) -> 'c) : 'c =
  match dt with
  | Float16 -> B.read b Bigarray.float16 (fun ba -> f (fun i -> A.get ba i))
  | Float32 -> B.read b Bigarray.float32 (fun ba -> f (fun i -> A.get ba i))
  | Float64 -> B.read b Bigarray.float64 (fun ba -> f (fun i -> A.get ba i))
  | BFloat16 ->
      B.read b Bigarray.int16_unsigned (fun ba ->
          f (fun i -> S.decode BFloat16 (A.get ba i)))
  | Float8_e4m3 ->
      B.read b u8 (fun ba -> f (fun i -> S.decode Float8_e4m3 (A.get ba i)))
  | Float8_e5m2 ->
      B.read b u8 (fun ba -> f (fun i -> S.decode Float8_e5m2 (A.get ba i)))
  | Int4 ->
      let n = length dt b in
      B.read b u8 (fun ba ->
          f (fun i ->
              in_bounds "read" n i;
              let v = sub ba 4 i in
              if v >= 8 then v - 16 else v))
  | UInt4 ->
      let n = length dt b in
      B.read b u8 (fun ba ->
          f (fun i ->
              in_bounds "read" n i;
              sub ba 4 i))
  | Int8 ->
      B.read b Bigarray.int8_signed (fun ba -> f (fun i -> A.get ba i))
  | UInt8 -> B.read b u8 (fun ba -> f (fun i -> A.get ba i))
  | Int16 ->
      B.read b Bigarray.int16_signed (fun ba -> f (fun i -> A.get ba i))
  | UInt16 ->
      B.read b Bigarray.int16_unsigned (fun ba -> f (fun i -> A.get ba i))
  | Int32 -> B.read b Bigarray.int32 (fun ba -> f (fun i -> A.get ba i))
  | UInt32 -> B.read b Bigarray.int32 (fun ba -> f (fun i -> A.get ba i))
  | Int64 -> B.read b Bigarray.int64 (fun ba -> f (fun i -> A.get ba i))
  | UInt64 -> B.read b Bigarray.int64 (fun ba -> f (fun i -> A.get ba i))
  | Complex64 ->
      B.read b Bigarray.complex32 (fun ba -> f (fun i -> A.get ba i))
  | Complex128 ->
      B.read b Bigarray.complex64 (fun ba -> f (fun i -> A.get ba i))
  | Bool -> B.read b u8 (fun ba -> f (fun i -> A.get ba i <> 0))
  | Bit ->
      let n = length dt b in
      B.read b u8 (fun ba ->
          f (fun i ->
              in_bounds "read" n i;
              sub ba 1 i <> 0))

let write (type a b) (dt : (a, b) Nx_dtype.t) b (f : (int -> a -> unit) -> 'c)
    : 'c =
  match dt with
  | Float16 ->
      B.write b Bigarray.int16_unsigned (fun ba ->
          f (fun i v -> A.set ba i (S.encode Float16 v)))
  | Float32 -> B.write b Bigarray.float32 (fun ba -> f (fun i v -> A.set ba i v))
  | Float64 -> B.write b Bigarray.float64 (fun ba -> f (fun i v -> A.set ba i v))
  | BFloat16 ->
      B.write b Bigarray.int16_unsigned (fun ba ->
          f (fun i v -> A.set ba i (S.encode BFloat16 v)))
  | Float8_e4m3 ->
      B.write b u8 (fun ba -> f (fun i v -> A.set ba i (S.encode Float8_e4m3 v)))
  | Float8_e5m2 ->
      B.write b u8 (fun ba -> f (fun i v -> A.set ba i (S.encode Float8_e5m2 v)))
  | Int4 ->
      let n = length dt b in
      B.write b u8 (fun ba ->
          f (fun i v ->
              in_bounds "write" n i;
              set_sub ba 4 i v))
  | UInt4 ->
      let n = length dt b in
      B.write b u8 (fun ba ->
          f (fun i v ->
              in_bounds "write" n i;
              set_sub ba 4 i v))
  | Int8 ->
      B.write b Bigarray.int8_signed (fun ba -> f (fun i v -> A.set ba i v))
  | UInt8 -> B.write b u8 (fun ba -> f (fun i v -> A.set ba i v))
  | Int16 ->
      B.write b Bigarray.int16_signed (fun ba -> f (fun i v -> A.set ba i v))
  | UInt16 ->
      B.write b Bigarray.int16_unsigned (fun ba -> f (fun i v -> A.set ba i v))
  | Int32 -> B.write b Bigarray.int32 (fun ba -> f (fun i v -> A.set ba i v))
  | UInt32 -> B.write b Bigarray.int32 (fun ba -> f (fun i v -> A.set ba i v))
  | Int64 -> B.write b Bigarray.int64 (fun ba -> f (fun i v -> A.set ba i v))
  | UInt64 -> B.write b Bigarray.int64 (fun ba -> f (fun i v -> A.set ba i v))
  | Complex64 ->
      B.write b Bigarray.complex32 (fun ba -> f (fun i v -> A.set ba i v))
  | Complex128 ->
      B.write b Bigarray.complex64 (fun ba -> f (fun i v -> A.set ba i v))
  | Bool -> B.write b u8 (fun ba -> f (fun i v -> A.set ba i (Bool.to_int v)))
  | Bit ->
      let n = length dt b in
      B.write b u8 (fun ba ->
          f (fun i v ->
              in_bounds "write" n i;
              set_sub ba 1 i (Bool.to_int v)))

(* Elements of fewer than 8 bits fill whole bytes, then the last ones one at a
   time: the bits after them in their byte keep what they hold, which is 0 in a
   fresh buffer ({!create}) and another value's elements in a view of a larger
   buffer. *)
let fill_sub ba ~n bits v =
  let per = 8 / bits in
  let byte = ref 0 in
  for k = 0 to per - 1 do
    byte := !byte lor (v lsl (k * bits))
  done;
  A.fill (A.sub ba 0 (n / per)) !byte;
  for i = n / per * per to n - 1 do
    set_sub ba bits i v
  done

(* The bytes of [v] as one element of [dt], [width] bytes. *)
let element (type a b) (dt : (a, b) Nx_dtype.t) width (v : a) =
  let e = Bytes.create width in
  (match dt with
  | Float16 -> Bytes.set_uint16_ne e 0 (S.encode Float16 v)
  | BFloat16 -> Bytes.set_uint16_ne e 0 (S.encode BFloat16 v)
  | Float8_e4m3 -> Bytes.set_uint8 e 0 (S.encode Float8_e4m3 v)
  | Float8_e5m2 -> Bytes.set_uint8 e 0 (S.encode Float8_e5m2 v)
  | Float32 -> Bytes.set_int32_ne e 0 (Int32.bits_of_float v)
  | Float64 -> Bytes.set_int64_ne e 0 (Int64.bits_of_float v)
  | Int4 -> Bytes.set_uint8 e 0 (v land 0xf)
  | UInt4 -> Bytes.set_uint8 e 0 (v land 0xf)
  | Int8 -> Bytes.set_int8 e 0 v
  | UInt8 -> Bytes.set_uint8 e 0 (v land 0xff)
  | Int16 -> Bytes.set_int16_ne e 0 v
  | UInt16 -> Bytes.set_uint16_ne e 0 (v land 0xffff)
  | Int32 -> Bytes.set_int32_ne e 0 v
  | UInt32 -> Bytes.set_int32_ne e 0 v
  | Int64 -> Bytes.set_int64_ne e 0 v
  | UInt64 -> Bytes.set_int64_ne e 0 v
  | Complex64 ->
      Bytes.set_int32_ne e 0 (Int32.bits_of_float v.re);
      Bytes.set_int32_ne e 4 (Int32.bits_of_float v.im)
  | Complex128 ->
      Bytes.set_int64_ne e 0 (Int64.bits_of_float v.re);
      Bytes.set_int64_ne e 8 (Int64.bits_of_float v.im)
  | Bool -> Bytes.set_uint8 e 0 (Bool.to_int v)
  | Bit -> Bytes.set_uint8 e 0 (Bool.to_int v));
  e

external fill_bytes :
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) A.t -> Bytes.t -> unit
  = "caml_nx_array_fill"
[@@noalloc]

let fill (type a b) (dt : (a, b) Nx_dtype.t) b (v : a) =
  let n = length dt b in
  B.write b u8 @@ fun ba ->
  match dt with
  | Int4 -> fill_sub ba ~n 4 (v land 0xf)
  | UInt4 -> fill_sub ba ~n 4 (v land 0xf)
  | Bit -> fill_sub ba ~n 1 (Bool.to_int v)
  | _ -> fill_bytes ba (element dt (bits dt / 8) v)

(* Gathering *)

(* Calls [f src dst] for each element of [v], [src] its storage index and [dst]
   its index in C order, a run of [run] elements at a time when [run] is more
   than one. *)
let iter_view v ~run f =
  let shape = View.shape v and strides = View.strides v in
  let rank = Array.length shape in
  let outer = if run > 1 then rank - 1 else rank in
  let n = View.numel v / Int.max 1 run in
  let idx = Array.make rank 0 and off = ref (View.offset v) in
  for k = 0 to n - 1 do
    f !off (k * run);
    let a = ref (outer - 1) in
    while !a >= 0 do
      idx.(!a) <- idx.(!a) + 1;
      off := !off + strides.(!a);
      if idx.(!a) < shape.(!a) then a := -1
      else begin
        off := !off - (strides.(!a) * shape.(!a));
        idx.(!a) <- 0;
        decr a
      end
    done
  done

(* [v] over words of [w] per element: an element is [w] consecutive words. *)
let in_words v w =
  if w = 1 then v
  else
    View.create
      ~offset:(View.offset v * w)
      ~strides:
        (Array.append (Array.map (fun s -> s * w) (View.strides v)) [| 1 |])
      (Array.append (View.shape v) [| w |])

(* Copies the elements of [v] in C order: [copy_run src dst n] copies [n] of
   them along a contiguous last axis, and [copy src dst] one at a time. *)
let copy_view v ~copy_run ~copy =
  let rank = View.ndim v in
  if rank > 0 && (View.strides v).(rank - 1) = 1 then begin
    let run = (View.shape v).(rank - 1) in
    iter_view v ~run (fun src dst -> copy_run src dst run)
  end
  else iter_view v ~run:1 copy

let blit s d src dst n = A.blit (A.sub s src n) (A.sub d dst n)

(* Raises unless [v] reaches only elements of [dt] in [b]. *)
let check_view fn dt b v =
  let n = length dt b in
  if not (View.within v n) then
    invalid_arg
      (Printf.sprintf
         "Nx_array.Elements.%s: the view reaches outside %d elements" fn n)

(* Copies the view [v] of [b]'s elements of [kind] into [dst]. *)
let copy_words kind b v dst =
  B.read b kind @@ fun s ->
  B.write dst kind @@ fun d ->
  copy_view v ~copy_run:(blit s d) ~copy:(fun src dst ->
      A.unsafe_set d dst (A.unsafe_get s src))

let gather dt b v =
  check_view "gather" dt b v;
  let n = View.numel v in
  let dst = create dt n in
  if n > 0 then
    (* Each width is its own loop, over its own kind, which the compiler
       specializes. *)
    begin match bits dt with
    | (1 | 4) as bits ->
        B.read b u8 @@ fun s ->
        B.write dst u8 @@ fun d ->
        iter_view v ~run:1 (fun src dst -> set_sub d bits dst (sub s bits src))
    | 8 -> copy_words u8 b v dst
    | 16 -> copy_words Bigarray.int16_unsigned b v dst
    | 32 -> copy_words Bigarray.int32 b v dst
    | bits -> copy_words Bigarray.int64 b (in_words v (bits / 64)) dst
    end;
  dst

let contiguous dt b v =
  check_view "contiguous" dt b v;
  let n = View.numel v in
  let bit = View.offset v * bits dt in
  if n > 0 && View.is_c_contiguous v && bit mod 8 = 0 then
    B.view b ~first:(bit / 8) ~length:(bytes dt n)
  else gather dt b v
