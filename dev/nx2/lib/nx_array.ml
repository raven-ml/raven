(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Dtype = Dtype
module Move = Move
module Layout = Layout
module Buffer = Rig.Buffer

let invalid_argf fmt = Format.kasprintf invalid_arg fmt
let pp_ints = Shape.pp

(* nx_array_stubs.c reads the fields in this order. *)
type ('v, 's) t = {
  dtype : ('v, 's) Dtype.t;
  layout : Layout.t;
  buffer : Buffer.t;
}

type any = Any : ('v, 's) t -> any

let dtype a = a.dtype
let layout a = a.layout
let buffer a = a.buffer
let device a = Buffer.device a.buffer

(* Refusals. The codes are nx_array.h's. *)

let pending = 4

let reason = function
  | 1 -> "an operand's dtype is not the one the kernel loads"
  | 2 -> "an operand's buffer is dead"
  | 3 -> "the host does not address an operand's memory"
  | 4 -> "device work on an operand is unfinished"
  | 5 -> "an operand's memory is held exclusive"
  | 6 -> "a written operand's memory is read-only"
  | 7 -> "a written operand reaches an element twice"
  | 8 -> "a written operand shares bytes with another operand"
  | 9 -> "an operand's layout is not a layout"
  | 10 -> "the operands' shapes differ"
  | 11 -> "too many operands"
  | c -> Printf.sprintf "code %d" c

let pp_operand ppf (Any a) =
  Format.fprintf ppf "%a %a" Dtype.pp a.dtype pp_ints (Layout.shape a.layout)

let settle name code operands =
  if code = pending then
    List.iter (fun (Any a) -> Buffer.wait a.buffer Buffer.Read_write) operands
  else
    invalid_argf "%s: %s (%a)" name (reason code)
      (Format.pp_print_list
         ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
         pp_operand)
      operands

(* Making arrays *)

external host_address : Buffer.t -> (int[@untagged])
  = "nx_array_host_byte" "nx_array_host"
[@@noalloc]

(* The alignment of a dtype's storage: its width for byte-wide dtypes, one
   component's for complex ones. *)
let alignment dt =
  let bits = Dtype.bits dt in
  if bits < 8 then 1
  else if Dtype.is Dtype.Complex dt then bits / 16
  else bits / 8

let check fn dt l b =
  (match Buffer.dead b with
  | Some why -> invalid_argf "%s: the buffer is dead: %s" fn why
  | None -> ());
  if Layout.numel l > 0 then begin
    let hi = snd (Layout.span l) in
    let bits = Dtype.bits dt and length = Buffer.length b in
    let fits =
      if bits >= 8 then hi <= length / (bits / 8)
      else
        let k = 8 / bits in
        ((hi / k) + if hi mod k = 0 then 0 else 1) <= length
    in
    if not fits then
      invalid_argf "%s: %a reaches past %d bytes of %a" fn Layout.pp l length
        Dtype.pp dt;
    let first = if bits >= 8 then Layout.offset l * (bits / 8) else 0 in
    let a = alignment dt and host = host_address b in
    if
      (Buffer.offset b + first) mod a <> 0
      || (host >= 0 && (host + first) mod a <> 0)
    then
      invalid_argf
        "%s: the first element of %a is not on a multiple of %d bytes" fn
        Dtype.pp dt a
  end

let v dtype layout buffer =
  check "Nx_array.v" dtype layout buffer;
  { dtype; layout; buffer }

(* Zeroes the last byte of [b], whose elements are [bits] wide, if elements do
   not fill it: bits past the last element stay zero. *)
let zero_tail bits n b =
  let k = 8 / bits in
  if bits < 8 && n mod k <> 0 then begin
    let last = Buffer.length b - 1 in
    if Rig.equal (Buffer.device b) Rig.host then
      Buffer.blit_from_string "\000" 0 b last 1
    else
      Buffer.copy ~src:(Buffer.of_string "\000")
        ~dst:(Buffer.view b ~first:last ~length:1)
  end

let create ?memory d dtype s =
  let layout = Layout.contiguous s in
  let n = Layout.numel layout in
  let buffer = Buffer.create ?memory d (Dtype.bytes dtype n) in
  zero_tail (Dtype.bits dtype) n buffer;
  { dtype; layout; buffer }

(* Movements and bitcasts *)

let move m a =
  match Layout.move m a.layout with
  | Some layout -> Some { a with layout }
  | None -> None

(* A layout of elements [r] times narrower over the same bits: a trailing axis
   of extent [r] and stride 1, every other stride and the offset times [r]. *)
let narrow fn r l =
  let k = Layout.rank l in
  if k >= Layout.max_rank then
    invalid_argf "%s: a narrowing bitcast of rank %d" fn Layout.max_rank;
  let shape = Array.append (Layout.shape l) [| r |] in
  let strides = Array.append (Array.map (( * ) r) (Layout.strides l)) [| 1 |] in
  Layout.v ~offset:(Layout.offset l * r) ~strides shape

(* A layout of elements [r] times wider over the same bits, if [l] has a
   trailing axis of extent [r] and stride 1, and an offset and other strides
   that are multiples of [r]. *)
let widen r l =
  let k = Layout.rank l in
  if k = 0 || Layout.dim l (k - 1) <> r then None
  else
    let shape = Array.sub (Layout.shape l) 0 (k - 1) in
    if Layout.numel l = 0 then Some (Layout.contiguous shape)
    else
      let strides = Layout.strides l in
      let multiple s = s mod r = 0 in
      if
        strides.(k - 1) <> 1
        || (not (multiple (Layout.offset l)))
        || not (Array.for_all multiple (Array.sub strides 0 (k - 1)))
      then None
      else
        let strides =
          Array.map (fun s -> s / r) (Array.sub strides 0 (k - 1))
        in
        Some (Layout.v ~offset:(Layout.offset l / r) ~strides shape)

let bitcast dtype' a =
  let w = Dtype.bits a.dtype and w' = Dtype.bits dtype' in
  let layout =
    if w = w' then Some a.layout
    else if w' < w then Some (narrow "Nx_array.bitcast" (w / w') a.layout)
    else widen (w' / w) a.layout
  in
  match layout with
  | None -> None
  | Some layout -> (
      match check "Nx_array.bitcast" dtype' layout a.buffer with
      | () -> Some { dtype = dtype'; layout; buffer = a.buffer }
      | exception Invalid_argument _ -> None)

let expect (type v s) (dt : (v, s) Dtype.t) (Any a) : (v, s) t =
  match Dtype.equal_witness dt a.dtype with
  | Some Equal -> a
  | None ->
      invalid_argf "Nx_array.expect: an array of %a, not %a" Dtype.pp a.dtype
        Dtype.pp dt

(* Single elements *)

external get_float :
  Buffer.t ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (float[@unboxed]) = "nx_array_get_float_byte" "nx_array_get_float"
[@@noalloc]

external get_int :
  Buffer.t -> (int[@untagged]) -> (int[@untagged]) -> (int64[@unboxed])
  = "nx_array_get_int_byte" "nx_array_get_int"
[@@noalloc]

external set_float :
  Buffer.t ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (float[@unboxed]) ->
  unit = "nx_array_set_float_byte" "nx_array_set_float"
[@@noalloc]

external set_int :
  Buffer.t -> (int[@untagged]) -> (int[@untagged]) -> (int64[@unboxed]) -> unit
  = "nx_array_set_int_byte" "nx_array_set_int"
[@@noalloc]

let position fn a idx =
  let l = a.layout in
  let r = Layout.rank l in
  if Array.length idx <> r then
    invalid_argf "%s: index %a for %d axes" fn pp_ints idx r;
  let p = ref (Layout.offset l) in
  for i = 0 to r - 1 do
    let j = idx.(i) in
    if j < 0 || j >= Layout.dim l i then
      invalid_argf "%s: index %a outside shape %a" fn pp_ints idx pp_ints
        (Layout.shape l);
    p := !p + (j * Layout.stride l i)
  done;
  !p

(* Claims [b]'s memory, waits for the device work [access] follows, and returns
   with the claim held: the caller releases it. *)
let claim fn b access =
  (match Buffer.dead b with
  | Some why -> invalid_argf "%s: the buffer is dead: %s" fn why
  | None -> ());
  if host_address b < 0 then
    invalid_argf "%s: the host does not address the buffer's memory" fn;
  Rig.Claim.read b;
  match Buffer.wait b access with
  | () -> ()
  | exception e ->
      Rig.Claim.release b;
      raise e

let load : type v s. (v, s) Dtype.t -> Buffer.t -> int -> v =
 fun dt b p ->
  let c = Dtype.code dt in
  let complex () = { Complex.re = get_float b c p 0; im = get_float b c p 1 } in
  match dt with
  | Float64 -> get_float b c p 0
  | Float32 -> get_float b c p 0
  | Float16 -> get_float b c p 0
  | Bfloat16 -> get_float b c p 0
  | Float8_e4m3fn -> get_float b c p 0
  | Float8_e5m2 -> get_float b c p 0
  | Float4_e2m1fn -> get_float b c p 0
  | Int64 -> get_int b c p
  | Uint64 -> get_int b c p
  | Int32 -> Int64.to_int32 (get_int b c p)
  | Uint32 -> Int64.to_int32 (get_int b c p)
  | Int16 -> Int64.to_int (get_int b c p)
  | Uint16 -> Int64.to_int (get_int b c p)
  | Int8 -> Int64.to_int (get_int b c p)
  | Uint8 -> Int64.to_int (get_int b c p)
  | Int4 -> Int64.to_int (get_int b c p)
  | Uint4 -> Int64.to_int (get_int b c p)
  | Complex128 -> complex ()
  | Complex64 -> complex ()
  | Bool -> get_int b c p <> 0L
  | Bit -> get_int b c p <> 0L

(* Raises unless the [int] [x] lies in [dt]'s range. *)
let check_int fn dt x =
  let lo = Dtype.min_value dt and hi = Dtype.max_value dt in
  if x < lo || x > hi then
    invalid_argf "%s: %d is outside %a's range [%d, %d]" fn x Dtype.pp dt lo hi

(* Raises unless a store of [x] into [dt] keeps it: an [int] in the range of its
   dtype. *)
let checked : type v s. string -> (v, s) Dtype.t -> v -> unit =
 fun fn dt x ->
  match dt with
  | Int16 -> check_int fn dt x
  | Uint16 -> check_int fn dt x
  | Int8 -> check_int fn dt x
  | Uint8 -> check_int fn dt x
  | Int4 -> check_int fn dt x
  | Uint4 -> check_int fn dt x
  | Float64 | Float32 | Float16 | Bfloat16 | Float8_e4m3fn | Float8_e5m2
  | Float4_e2m1fn | Int64 | Uint64 | Int32 | Uint32 | Complex128 | Complex64
  | Bool | Bit ->
      ()

let store : type v s. (v, s) Dtype.t -> Buffer.t -> int -> v -> unit =
 fun dt b p x ->
  let c = Dtype.code dt in
  let complex (z : Complex.t) =
    set_float b c p 0 z.re;
    set_float b c p 1 z.im
  in
  let int x = set_int b c p (Int64.of_int x) in
  match dt with
  | Float64 -> set_float b c p 0 x
  | Float32 -> set_float b c p 0 x
  | Float16 -> set_float b c p 0 x
  | Bfloat16 -> set_float b c p 0 x
  | Float8_e4m3fn -> set_float b c p 0 x
  | Float8_e5m2 -> set_float b c p 0 x
  | Float4_e2m1fn -> set_float b c p 0 x
  | Int64 -> set_int b c p x
  | Uint64 -> set_int b c p x
  | Int32 -> set_int b c p (Int64.of_int32 x)
  | Uint32 -> set_int b c p (Int64.of_int32 x)
  | Int16 -> int x
  | Uint16 -> int x
  | Int8 -> int x
  | Uint8 -> int x
  | Int4 -> int x
  | Uint4 -> int x
  | Complex128 -> complex x
  | Complex64 -> complex x
  | Bool -> int (Bool.to_int x)
  | Bit -> int (Bool.to_int x)

let get a idx =
  let p = position "Nx_array.get" a idx in
  claim "Nx_array.get" a.buffer Buffer.Read;
  let x = load a.dtype a.buffer p in
  Rig.Claim.release a.buffer;
  x

let set a idx x =
  let fn = "Nx_array.set" in
  let p = position fn a idx in
  checked fn a.dtype x;
  if not (Layout.is_distinct a.layout) then
    invalid_argf "%s: the array reaches an element twice" fn;
  if Buffer.access a.buffer = Buffer.Read then
    invalid_argf "%s: the array's memory is read-only" fn;
  claim fn a.buffer Buffer.Read_write;
  store a.dtype a.buffer p x;
  Rig.Claim.release a.buffer

(* Bulk access *)

external to_array_into : ('v, 's) t -> 'v array -> int = "nx_array_to_array"

external of_array_from : ('v, 's) t -> 'v array -> int = "nx_array_of_array"
[@@noalloc]

external copy_into : ('v, 's) t -> ('v, 's) t -> int = "nx_array_copy"
[@@noalloc]

let to_array (type v s) (a : (v, s) t) : v array =
  let n = Layout.numel a.layout in
  if n = 0 then [||]
  else
    let out : v array =
      match Dtype.kind a.dtype with
      | Float -> Array.create_float n
      | Complex | Signed | Unsigned | Boolean ->
          Array.make n (Dtype.zero a.dtype)
    in
    let rec read () =
      let e = to_array_into a out in
      if e <> 0 then begin
        settle "Nx_array.to_array" e [ Any a ];
        read ()
      end
    in
    read ();
    out

let of_array dt s values =
  let fn = "Nx_array.of_array" in
  if Array.length values <> Layout.numel (Layout.contiguous s) then
    invalid_argf "%s: %d values for shape %a" fn (Array.length values) pp_ints s;
  Array.iter (checked fn dt) values;
  let a = create Rig.host dt s in
  let rec write () =
    let e = of_array_from a values in
    if e <> 0 then begin
      settle fn e [ Any a ];
      write ()
    end
  in
  write ();
  a

let copy a =
  let dst = create (device a) a.dtype (Layout.shape a.layout) in
  let rec gather () =
    let e = copy_into dst a in
    if e <> 0 then begin
      settle "Nx_array.copy" e [ Any dst; Any a ];
      gather ()
    end
  in
  gather ();
  dst

let to_device d a =
  let l = a.layout and bits = Dtype.bits a.dtype in
  let lo, hi = Layout.span l in
  let first = lo * bits / 8 and last = ((hi * bits) + 7) / 8 in
  let buffer = Buffer.create d (last - first) in
  if last > first then begin
    let src = Buffer.view a.buffer ~first ~length:(last - first) in
    Rig.Claim.read src;
    Fun.protect
      ~finally:(fun () -> Rig.Claim.release src)
      (fun () -> Buffer.copy ~src ~dst:buffer)
  end;
  let offset = Layout.offset l - (8 * first / bits) in
  let layout = Layout.v ~offset ~strides:(Layout.strides l) (Layout.shape l) in
  v a.dtype layout buffer

(* Bigarrays *)

let bigarray (type v s) (k : (v, s) Bigarray.kind) (a : (v, s) t) :
    (v, s, Bigarray.c_layout) Bigarray.Genarray.t option =
  let l = a.layout in
  if
    (not (Rig.equal (Buffer.device a.buffer) Rig.host))
    || (not (Layout.is_contiguous l))
    || Layout.rank l > 16
  then None
  else
    let w = Bigarray.kind_size_in_bytes k and n = Layout.numel l in
    let view =
      Buffer.view a.buffer ~first:(Layout.offset l * w) ~length:(n * w)
    in
    let flat = Buffer.bigarray k view in
    Some (Bigarray.reshape (Bigarray.genarray_of_array1 flat) (Layout.shape l))

let of_bigarray dt g =
  let dims = Bigarray.Genarray.dims g in
  let n = Array.fold_left ( * ) 1 dims in
  let flat = Bigarray.reshape_1 g n in
  v dt (Layout.contiguous dims) (Buffer.of_bigarray flat)
