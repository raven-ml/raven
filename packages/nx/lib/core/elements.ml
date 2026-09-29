(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module B = Nx_device.Buffer
module S = Nx_dtype.Scalar
module A = Bigarray.Array1

let check fn (type a b) (dt : (a, b) Nx_dtype.t) b =
  if not (Nx_device.equal (B.device b) Nx_device.host) then
    invalid_arg
      (Printf.sprintf "Nx_core.Elements.%s: the buffer is on %s, not CPU" fn
         (Nx_device.name (B.device b)));
  if not (S.equal (B.dtype b) (S.of_dtype dt)) then
    invalid_arg
      (Printf.sprintf "Nx_core.Elements.%s: a %s buffer read as %s" fn
         (S.to_string (B.dtype b))
         (Nx_dtype.to_string dt))

let bytes b = B.bigarray Bigarray.int8_unsigned b

(* 4-bit elements, two to a byte, the first in the low nibble *)

let nibble ba i =
  let byte = A.unsafe_get ba (i lsr 1) in
  if i land 1 = 0 then byte land 0xf else byte lsr 4

let set_nibble ba i v =
  let byte = A.unsafe_get ba (i lsr 1) in
  A.unsafe_set ba (i lsr 1)
    (if i land 1 = 0 then byte land 0xf0 lor v else byte land 0x0f lor (v lsl 4))

let in_bounds fn n i =
  if i < 0 || i >= n then
    invalid_arg
      (Printf.sprintf "Nx_core.Elements.%s: index %d of %d elements" fn i n)

let clamp lo hi v = Int.max lo (Int.min hi v)

(* Access *)

let get (type a b) (dt : (a, b) Nx_dtype.t) b : int -> a =
  check "get" dt b;
  match dt with
  | Float16 ->
      let ba = B.bigarray Bigarray.float16 b in
      fun i -> A.get ba i
  | Float32 ->
      let ba = B.bigarray Bigarray.float32 b in
      fun i -> A.get ba i
  | Float64 ->
      let ba = B.bigarray Bigarray.float64 b in
      fun i -> A.get ba i
  | BFloat16 ->
      let ba = B.bigarray Bigarray.int16_unsigned b in
      fun i -> S.decode BFloat16 (A.get ba i)
  | Float8_e4m3 ->
      let ba = bytes b in
      fun i -> S.decode Float8_e4m3 (A.get ba i)
  | Float8_e5m2 ->
      let ba = bytes b in
      fun i -> S.decode Float8_e5m2 (A.get ba i)
  | Int4 ->
      let ba = bytes b and n = B.length b in
      fun i ->
        in_bounds "get" n i;
        let v = nibble ba i in
        if v >= 8 then v - 16 else v
  | UInt4 ->
      let ba = bytes b and n = B.length b in
      fun i ->
        in_bounds "get" n i;
        nibble ba i
  | Int8 ->
      let ba = B.bigarray Bigarray.int8_signed b in
      fun i -> A.get ba i
  | UInt8 ->
      let ba = bytes b in
      fun i -> A.get ba i
  | Int16 ->
      let ba = B.bigarray Bigarray.int16_signed b in
      fun i -> A.get ba i
  | UInt16 ->
      let ba = B.bigarray Bigarray.int16_unsigned b in
      fun i -> A.get ba i
  | Int32 ->
      let ba = B.bigarray Bigarray.int32 b in
      fun i -> A.get ba i
  | UInt32 ->
      let ba = B.bigarray Bigarray.int32 b in
      fun i -> A.get ba i
  | Int64 ->
      let ba = B.bigarray Bigarray.int64 b in
      fun i -> A.get ba i
  | UInt64 ->
      let ba = B.bigarray Bigarray.int64 b in
      fun i -> A.get ba i
  | Complex64 ->
      let ba = B.bigarray Bigarray.complex32 b in
      fun i -> A.get ba i
  | Complex128 ->
      let ba = B.bigarray Bigarray.complex64 b in
      fun i -> A.get ba i
  | Bool ->
      let ba = bytes b in
      fun i -> A.get ba i <> 0

let set (type a b) (dt : (a, b) Nx_dtype.t) b : int -> a -> unit =
  check "set" dt b;
  match dt with
  | Float16 ->
      let ba = B.bigarray Bigarray.float16 b in
      fun i v -> A.set ba i v
  | Float32 ->
      let ba = B.bigarray Bigarray.float32 b in
      fun i v -> A.set ba i v
  | Float64 ->
      let ba = B.bigarray Bigarray.float64 b in
      fun i v -> A.set ba i v
  | BFloat16 ->
      let ba = B.bigarray Bigarray.int16_unsigned b in
      fun i v -> A.set ba i (S.encode BFloat16 v)
  | Float8_e4m3 ->
      let ba = bytes b in
      fun i v -> A.set ba i (S.encode Float8_e4m3 v)
  | Float8_e5m2 ->
      let ba = bytes b in
      fun i v -> A.set ba i (S.encode Float8_e5m2 v)
  | Int4 ->
      let ba = bytes b and n = B.length b in
      fun i v ->
        in_bounds "set" n i;
        set_nibble ba i (clamp (-8) 7 v land 0xf)
  | UInt4 ->
      let ba = bytes b and n = B.length b in
      fun i v ->
        in_bounds "set" n i;
        set_nibble ba i (clamp 0 15 v)
  | Int8 ->
      let ba = B.bigarray Bigarray.int8_signed b in
      fun i v -> A.set ba i v
  | UInt8 ->
      let ba = bytes b in
      fun i v -> A.set ba i v
  | Int16 ->
      let ba = B.bigarray Bigarray.int16_signed b in
      fun i v -> A.set ba i v
  | UInt16 ->
      let ba = B.bigarray Bigarray.int16_unsigned b in
      fun i v -> A.set ba i v
  | Int32 ->
      let ba = B.bigarray Bigarray.int32 b in
      fun i v -> A.set ba i v
  | UInt32 ->
      let ba = B.bigarray Bigarray.int32 b in
      fun i v -> A.set ba i v
  | Int64 ->
      let ba = B.bigarray Bigarray.int64 b in
      fun i v -> A.set ba i v
  | UInt64 ->
      let ba = B.bigarray Bigarray.int64 b in
      fun i v -> A.set ba i v
  | Complex64 ->
      let ba = B.bigarray Bigarray.complex32 b in
      fun i v -> A.set ba i v
  | Complex128 ->
      let ba = B.bigarray Bigarray.complex64 b in
      fun i v -> A.set ba i v
  | Bool ->
      let ba = bytes b in
      fun i v -> A.set ba i (Bool.to_int v)

(* An odd number of 4-bit elements leaves the last byte's high nibble to the
   memory after [b]. *)
let fill_nibbles b v =
  let ba = bytes b and n = B.length b in
  A.fill (A.sub ba 0 (n / 2)) (v lor (v lsl 4));
  if n land 1 = 1 then set_nibble ba (n - 1) v

let fill (type a b) (dt : (a, b) Nx_dtype.t) b (v : a) =
  check "fill" dt b;
  match dt with
  | Float16 -> A.fill (B.bigarray Bigarray.float16 b) v
  | Float32 -> A.fill (B.bigarray Bigarray.float32 b) v
  | Float64 -> A.fill (B.bigarray Bigarray.float64 b) v
  | BFloat16 ->
      A.fill (B.bigarray Bigarray.int16_unsigned b) (S.encode BFloat16 v)
  | Float8_e4m3 -> A.fill (bytes b) (S.encode Float8_e4m3 v)
  | Float8_e5m2 -> A.fill (bytes b) (S.encode Float8_e5m2 v)
  | Int4 -> fill_nibbles b (clamp (-8) 7 v land 0xf)
  | UInt4 -> fill_nibbles b (clamp 0 15 v)
  | Int8 -> A.fill (B.bigarray Bigarray.int8_signed b) v
  | UInt8 -> A.fill (bytes b) v
  | Int16 -> A.fill (B.bigarray Bigarray.int16_signed b) v
  | UInt16 -> A.fill (B.bigarray Bigarray.int16_unsigned b) v
  | Int32 -> A.fill (B.bigarray Bigarray.int32 b) v
  | UInt32 -> A.fill (B.bigarray Bigarray.int32 b) v
  | Int64 -> A.fill (B.bigarray Bigarray.int64 b) v
  | UInt64 -> A.fill (B.bigarray Bigarray.int64 b) v
  | Complex64 -> A.fill (B.bigarray Bigarray.complex32 b) v
  | Complex128 -> A.fill (B.bigarray Bigarray.complex64 b) v
  | Bool -> A.fill (bytes b) (Bool.to_int v)

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

(* The storage elements [v] reaches, from the lowest to one past the highest. *)
let extent v =
  let lo = ref (View.offset v) and hi = ref (View.offset v) in
  Array.iteri
    (fun a n ->
      let s = (View.strides v).(a) * (n - 1) in
      if s < 0 then lo := !lo + s else hi := !hi + s)
    (View.shape v);
  (!lo, !hi + 1)

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

let gather b v =
  if not (Nx_device.equal (B.device b) Nx_device.host) then
    invalid_arg
      (Printf.sprintf "Nx_core.Elements.gather: the buffer is on %s, not CPU"
         (Nx_device.name (B.device b)));
  let n = View.numel v in
  let dst = B.create Nx_device.host (B.dtype b) n in
  if n > 0 then begin
    let lo, hi = extent v in
    if lo < 0 || hi > B.length b then
      invalid_arg
        (Printf.sprintf
           "Nx_core.Elements.gather: the view reaches elements %d to %d of %d"
           lo (hi - 1) (B.length b));
    (* Each width is its own loop, over its own kind, which the compiler
       specializes. *)
    match S.bitsize (B.dtype b) with
    | 4 ->
        let s = bytes b and d = bytes dst in
        iter_view v ~run:1 (fun src dst -> set_nibble d dst (nibble s src))
    | 8 ->
        let s = bytes b and d = bytes dst in
        copy_view v ~copy_run:(blit s d) ~copy:(fun src dst ->
            A.unsafe_set d dst (A.unsafe_get s src))
    | 16 ->
        let s = B.bigarray Bigarray.int16_unsigned b
        and d = B.bigarray Bigarray.int16_unsigned dst in
        copy_view v ~copy_run:(blit s d) ~copy:(fun src dst ->
            A.unsafe_set d dst (A.unsafe_get s src))
    | 32 ->
        let s = B.bigarray Bigarray.int32 b
        and d = B.bigarray Bigarray.int32 dst in
        copy_view v ~copy_run:(blit s d) ~copy:(fun src dst ->
            A.unsafe_set d dst (A.unsafe_get s src))
    | bits ->
        let s = B.bigarray Bigarray.int64 b
        and d = B.bigarray Bigarray.int64 dst in
        copy_view
          (in_words v (bits / 64))
          ~copy_run:(blit s d)
          ~copy:(fun src dst -> A.unsafe_set d dst (A.unsafe_get s src))
  end;
  dst
