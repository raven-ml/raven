(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Dtype = Dtype
module Move = Move
module Layout = Layout
module Buffer = Rig.Buffer

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* nx_array_stubs.c reads the fields in this order. The record is private
   rather than abstract so that [any] below can be unboxed: the compiler
   must see that an array is a block that holds no float. *)
type ('v, 's) t = {
  dtype : ('v, 's) Dtype.t;
  layout : Layout.t;
  buffer : Buffer.t;
}

(* Unboxed: C reads an any as its array. *)
type any = Any : ('v, 's) t -> any [@@unboxed]

let dtype a = a.dtype
let layout a = a.layout
let buffer a = a.buffer
let device a = Buffer.device a.buffer

let max_operands = Shape.max_operands

(* Answers. C answers Val_int of nx_array.h's codes, whose enum lists them
   in this order. *)

type answer =
  | Done
  | Declined
  | Wrong_dtype
  | Dead_buffer
  | Off_host
  | Held_exclusive
  | Read_only
  | Repeated_elements
  | Overlapping
  | Shape_mismatch

let reason = function
  | Done | Declined -> invalid_arg "Nx_array.refused: not a refusal"
  | Wrong_dtype -> "an operand's dtype is not the one the kernel loads"
  | Dead_buffer -> "an operand's buffer is dead"
  | Off_host -> "the host does not address an operand's memory"
  | Held_exclusive -> "an operand's memory is held exclusive"
  | Read_only -> "a written operand's memory is read-only"
  | Repeated_elements -> "a written operand reaches an element twice"
  | Overlapping -> "a written operand shares bytes with another operand"
  | Shape_mismatch -> "the operands' number or shapes do not fit the operation"

let pp_operand ppf (Any a) =
  Format.fprintf ppf "%a %a" Dtype.pp a.dtype Shape.pp (Layout.shape a.layout)

(* The reason a [Dead_buffer] or [Held_exclusive] gives: each operand dead by
   now, in [operands]' order, with the reason its consumer gave. A holder of
   exclusive memory consumes it, or releases it unread. *)
let dead answer operands =
  let was (Any a as x) =
    Option.map
      (fun why -> Format.asprintf "%a was %s" pp_operand x why)
      (Buffer.dead a.buffer)
  in
  match List.filter_map was operands with
  | [] -> reason answer
  | ds -> String.concat ", " ds

let refused name answer operands =
  let why =
    match answer with
    | Dead_buffer | Held_exclusive -> dead answer operands
    | _ -> reason answer
  in
  invalid_argf "%s: %s (%a)" name why
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
       pp_operand)
    operands

(* The OCaml door. [admit] claims every array or none, checks under the
   claims what nx_read checks of a written array, and waits for nothing.
   [release] ends the claims; past max_operands arrays it lets other domains
   collect between runs of them, so it is not [@@noalloc]. An asynchronous
   exception (a signal handler's, a finaliser's) is raised at a poll point:
   an allocation, a function's entry or a loop's back edge. No claim the door
   takes is held across one outside the handler that ends it. *)

external admit : any array -> any array -> answer = "nx_array_admit"
external release : any array -> any array -> unit = "nx_array_release"

let door ~written ~read f x =
  match admit written read with
  | Done -> (
      match f x with
      | () ->
          release written read;
          Done
      | exception e ->
          release written read;
          Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ()))
  | refusal -> refusal

(* Holds a read claim on [a]'s memory while [f ()] runs: no donation writes
   it meanwhile. A copy under it orders itself after [a]'s work, and a load
   waits for that work first. *)
let reading fn a f =
  match door ~written:[||] ~read:[| Any a |] f () with
  | Done -> ()
  | refusal -> refused fn refusal [ Any a ]

(* Making arrays *)

external host_address : Buffer.t -> (int[@untagged])
  = "nx_array_host_byte" "nx_array_host"
[@@noalloc]

let hosted b = host_address b >= 0

(* Copies and fills

   Where the host addresses both buffers it moves the bytes before returning,
   which [Buffer.copy] does without allocating. Otherwise a device copies or
   fills: the work is queued and its point dropped, since rig's stamps order
   every later access of either buffer's memory after it, by the host or a
   device. *)

(* Copies [src]'s bytes into [dst], of one length. *)
let copy_bytes ~src ~dst =
  if hosted src && hosted dst then Buffer.copy ~src ~dst
  else
    ignore
      (Buffer.submit_copy ~src ~dst [| Buffer.row (Buffer.length src) |]
        : Rig.Point.t)

(* Sets byte [k] of [b] to [c]. *)
let set_byte b k c =
  let v = Buffer.view b ~first:k ~length:1 in
  if hosted b then Buffer.fill v c
  else ignore (Buffer.submit_fill v c : Rig.Point.t)

(* The alignment of a dtype's storage: its width for dtypes of a byte or
   more, one byte for narrower ones, one component's for complex ones. *)
let alignment dt =
  let bits = Dtype.bits dt in
  if bits < 8 then 1
  else if Dtype.is Dtype.Complex dt then bits / 16
  else bits / 8

let live fn b =
  match Buffer.dead b with
  | Some why -> invalid_argf "%s: the buffer is dead: %s" fn why
  | None -> ()

let reaches fn dt l b =
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
        Dtype.pp dt
  end

(* Whether the elements of [dt] over [b] lie on multiples of [dt]'s alignment,
   in [b]'s memory and, for host memory, as addresses. A position is a whole
   number of elements and an element's width a multiple of its alignment, so
   [b]'s first byte decides. An array with no element has none to align. *)
let aligned dt l b =
  Layout.numel l = 0
  ||
  let a = alignment dt and host = host_address b in
  Buffer.offset b mod a = 0 && (host < 0 || host mod a = 0)

let v dtype layout buffer =
  let fn = "Nx_array.v" in
  live fn buffer;
  reaches fn dtype layout buffer;
  if not (aligned dtype layout buffer) then
    invalid_argf "%s: the first element of %a is not on a multiple of %d bytes"
      fn Dtype.pp dtype (alignment dtype);
  { dtype; layout; buffer }

(* Zeroes the last byte of [b], whose elements are [bits] wide, if elements do
   not fill it: bits past the last element stay zero. *)
let zero_tail bits n b =
  if bits < 8 && n mod (8 / bits) <> 0 then
    let k = Buffer.length b - 1 in
    if hosted b then Buffer.blit_from_string "\000" 0 b k 1
    else set_byte b k '\000'

(* A fresh array laid out by [layout], C-contiguous at offset 0. *)
let alloc ?memory d dtype layout =
  let n = Layout.numel layout in
  let buffer = Buffer.create ?memory d (Dtype.bytes dtype n) in
  zero_tail (Dtype.bits dtype) n buffer;
  { dtype; layout; buffer }

let create ?memory d dtype s = alloc ?memory d dtype (Layout.contiguous s)
let host dtype s = create Rig.host dtype s

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
  let shape = Shape.zeros (k + 1) and strides = Shape.zeros (k + 1) in
  for i = 0 to k - 1 do
    shape.(i) <- Layout.dim l i;
    strides.(i) <- Layout.stride l i * r
  done;
  shape.(k) <- r;
  strides.(k) <- 1;
  Layout.v ~offset:(Layout.offset l * r) ~strides:(Shape.freeze strides)
    (Shape.freeze shape)

(* A layout of elements [r] times wider over the same bits, if [l] has a
   trailing axis of extent [r] and stride 1, and an offset and other strides
   that are multiples of [r]. *)
let widen r l =
  let k = Layout.rank l in
  if k = 0 || Layout.dim l (k - 1) <> r then None
  else
    let shape = Shape.zeros (k - 1) in
    for i = 0 to k - 2 do
      shape.(i) <- Layout.dim l i
    done;
    let shape = Shape.freeze shape in
    if Layout.numel l = 0 then Some (Layout.contiguous shape)
    else
      let fits =
        ref (Layout.stride l (k - 1) = 1 && Layout.offset l mod r = 0)
      in
      let strides = Shape.zeros (k - 1) in
      for i = 0 to k - 2 do
        let s = Layout.stride l i in
        if s mod r <> 0 then fits := false;
        strides.(i) <- s / r
      done;
      if !fits then
        Some
          (Layout.v ~offset:(Layout.offset l / r)
             ~strides:(Shape.freeze strides) shape)
      else None

(* A bitcast keeps the bits the array reaches, so its bounds hold. *)
let bitcast dtype' a =
  live "Nx_array.bitcast" a.buffer;
  let w = Dtype.bits a.dtype and w' = Dtype.bits dtype' in
  let layout =
    if w = w' then Some a.layout
    else if w' < w then Some (narrow "Nx_array.bitcast" (w / w') a.layout)
    else widen (w' / w) a.layout
  in
  match layout with
  | None -> None
  | Some layout when aligned dtype' layout a.buffer ->
      Some { dtype = dtype'; layout; buffer = a.buffer }
  | Some _ -> None

let expect (type v s) (dt : (v, s) Dtype.t) (Any a) : (v, s) t =
  match Dtype.equal_witness dt a.dtype with
  | Some Equal -> a
  | None ->
      invalid_argf "Nx_array.expect: an array of %a, not %a" Dtype.pp a.dtype
        Dtype.pp dt

(* Single elements *)

let position fn a idx =
  let l = a.layout in
  let r = Layout.rank l in
  if Iarray.length idx <> r then
    invalid_argf "%s: index %a for %d axes" fn Shape.pp idx r;
  let p = ref (Layout.offset l) in
  for i = 0 to r - 1 do
    let j = Iarray.get idx i in
    if j < 0 || j >= Layout.dim l i then
      invalid_argf "%s: index %a outside shape %a" fn Shape.pp idx Shape.pp
        (Layout.shape l);
    p := !p + (j * Layout.stride l i)
  done;
  !p

(* [a]'s element at position [p], boxed. [a]'s memory must be claimed, lie at
   a host address and follow the device work a read must follow. *)
external load : ('v, 's) t -> int -> 'v = "nx_array_load"

(* Stores [x] at [a]'s position [p], through the door. *)
external set_at : ('v, 's) t -> int -> 'v -> answer = "nx_array_set"

(* Raises unless [dt] holds [x]: an [int] in the range of its dtype. *)
let checked fn dt x =
  if not (Dtype.holds dt x) then
    let pp = Dtype.pp_value dt in
    invalid_argf "%s: %a is outside %a's range [%a, %a]" fn pp x Dtype.pp dt pp
      (Dtype.min_value dt) pp (Dtype.max_value dt)

(* Under the door's read claim, which names this function in a refusal, the
   element is loaded once the work a read must follow is done. Boxing it
   allocates under the claim, which the door releases if that raises. *)
let get a idx =
  let fn = "Nx_array.get" in
  let p = position fn a idx in
  let x = ref None in
  reading fn a (fun () ->
      if host_address a.buffer < 0 then refused fn Off_host [ Any a ];
      Buffer.wait a.buffer Rig.Read;
      x := Some (load a p));
  Option.get !x

let set a idx x =
  let fn = "Nx_array.set" in
  let p = position fn a idx in
  checked fn a.dtype x;
  match set_at a p x with Done -> () | refusal -> refused fn refusal [ Any a ]

(* Bulk access *)

external to_array_into : ('v, 's) t -> 'v array -> answer
  = "nx_array_to_array"

external to_bigarray :
  ('v, 's) t -> ('v, 'b, Bigarray.c_layout) Bigarray.Array1.t -> answer
  = "nx_array_to_bigarray"

external of_array_from : ('v, 's) t -> 'v array -> answer
  = "nx_array_of_array"

external copy_into : ('v, 's) t -> ('v, 's) t -> answer = "nx_array_copy"

(* Elements that box are copied unboxed into a bigarray of their width under the
   claim and boxed after it, so that an allocation that raises holds no claim.
   Each kind has its own loop, where the bigarray access is inlined. *)
let unboxed fn a k n =
  let b = Bigarray.Array1.create k Bigarray.c_layout n in
  match to_bigarray a b with
  | Done -> b
  | refusal -> refused fn refusal [ Any a ]

let int32s fn a n =
  let b = unboxed fn a Bigarray.int32 n in
  Array.init n (fun i -> Bigarray.Array1.unsafe_get b i)

let int64s fn a n =
  let b = unboxed fn a Bigarray.int64 n in
  Array.init n (fun i -> Bigarray.Array1.unsafe_get b i)

let complex32s fn a n =
  let b = unboxed fn a Bigarray.complex32 n in
  Array.init n (fun i -> Bigarray.Array1.unsafe_get b i)

let complex64s fn a n =
  let b = unboxed fn a Bigarray.complex64 n in
  Array.init n (fun i -> Bigarray.Array1.unsafe_get b i)

let to_array (type v s) (a : (v, s) t) : v array =
  let fn = "Nx_array.to_array" in
  let n = Layout.numel a.layout in
  let into (out : v array) =
    match to_array_into a out with
    | Done -> out
    | refusal -> refused fn refusal [ Any a ]
  in
  match a.dtype with
  | Float64 -> into (Array.create_float n)
  | Float32 -> into (Array.create_float n)
  | Float16 -> into (Array.create_float n)
  | Bfloat16 -> into (Array.create_float n)
  | Float8_e4m3fn -> into (Array.create_float n)
  | Float8_e5m2 -> into (Array.create_float n)
  | Float4_e2m1fn -> into (Array.create_float n)
  | Int16 -> into (Array.make n 0)
  | Uint16 -> into (Array.make n 0)
  | Int8 -> into (Array.make n 0)
  | Uint8 -> into (Array.make n 0)
  | Int4 -> into (Array.make n 0)
  | Uint4 -> into (Array.make n 0)
  | Bool -> into (Array.make n false)
  | Bit -> into (Array.make n false)
  | Int32 -> int32s fn a n
  | Uint32 -> int32s fn a n
  | Int64 -> int64s fn a n
  | Uint64 -> int64s fn a n
  | Complex64 -> complex32s fn a n
  | Complex128 -> complex64s fn a n

let of_array (type v s) (dt : (v, s) Dtype.t) s (values : v array) =
  let fn = "Nx_array.of_array" in
  let layout = Layout.contiguous s in
  if Array.length values <> Layout.numel layout then
    invalid_argf "%s: %d values for shape %a" fn (Array.length values) Shape.pp
      (Layout.shape layout);
  (* Only integers can fall outside their dtype's range. Iterating over a float
     array would box every element. *)
  (match Dtype.kind dt with
  | Float | Complex | Boolean -> ()
  | Signed | Unsigned -> Array.iter (checked fn dt) values);
  let a = alloc Rig.host dt layout in
  match of_array_from a values with
  | Done -> a
  | refusal -> refused fn refusal [ Any a ]

(* The host gathers [src]'s elements into [dst], of the same shape, bit for
   bit, through the door, writing [dst]'s elements and no other bit of its
   buffer: transfer's case where the host addresses both arrays' memory. A
   refusal names [fn]. *)
let blit fn ~src ~dst =
  live fn src.buffer;
  live fn dst.buffer;
  match copy_into dst src with
  | Done -> ()
  | refusal -> refused fn refusal [ Any dst; Any src ]

external load_byte : Buffer.t -> (int[@untagged]) -> (int[@untagged])
  = "nx_array_load_byte_byte" "nx_array_load_byte"
[@@noalloc]

(* Copies [src] to [dst], of one length, the bytes of a sub-byte array's span
   that [head] and [tail] say it shares at its ends with elements outside it.
   Another domain may store those elements while the copy runs: on memory the
   host addresses, each shared byte is loaded atomically and copied on its own,
   as a kernel's stores to it are atomic. *)
let copy_span ~head ~tail src dst =
  let n = Buffer.length src in
  if (not (head || tail)) || not (hosted src) then copy_bytes ~src ~dst
  else begin
    Buffer.wait src Rig.Read;
    let i = if head then 1 else 0 and j = if tail then n - 1 else n in
    if j > i then
      copy_bytes
        ~src:(Buffer.view src ~first:i ~length:(j - i))
        ~dst:(Buffer.view dst ~first:i ~length:(j - i));
    let edge k = set_byte dst k (Char.chr (load_byte src k)) in
    if head then edge 0;
    if tail && (n > 1 || not head) then edge (n - 1)
  end

(* The elements of [l]: one per index, an axis of stride 0 counted once. *)
let elements l =
  let n = ref (if Layout.numel l = 0 then 0 else 1) in
  for a = 0 to Layout.rank l - 1 do
    if Layout.stride l a <> 0 then n := !n * Layout.dim l a
  done;
  !n

(* [l]'s axes of nonzero stride, outermost first: by decreasing magnitude of
   stride, ties in axis order. *)
let stride_order l =
  List.init (Layout.rank l) Fun.id
  |> List.filter (fun a -> Layout.stride l a <> 0)
  |> List.stable_sort (fun a b ->
      Int.compare (abs (Layout.stride l b)) (abs (Layout.stride l a)))
  |> Array.of_list

(* [l]'s shape laid out in [l]'s stride order, as to_device's .mli states: [3;
   2] at strides [2; 2] packs to [2; 1]. *)
let packed l =
  let strides = Array.make (Layout.rank l) 0 in
  let offset = ref 0 and next = ref 1 in
  let order = stride_order l in
  for k = Array.length order - 1 downto 0 do
    let a = order.(k) in
    let d = Layout.dim l a in
    if Layout.stride l a > 0 then strides.(a) <- !next
    else begin
      strides.(a) <- - !next;
      offset := !offset + ((d - 1) * !next)
    end;
    next := !next * d
  done;
  Layout.v ~offset:!offset ~strides:(Shape.freeze strides) (Layout.shape l)

(* [l]'s axes that take more than one index, outermost first: by decreasing
   magnitude of stride, ties in axis order. *)
let run_order l =
  List.init (Layout.rank l) Fun.id
  |> List.filter (fun a -> Layout.dim l a > 1)
  |> List.stable_sort (fun a b ->
      Int.compare (abs (Layout.stride l b)) (abs (Layout.stride l a)))
  |> Array.of_list

(* The axes outside the runs that copy between layouts [s] and [t] of one shape,
   outermost first, and the runs' length: [t]'s innermost axes in its stride
   order where both lie as consecutive positions. *)
let runs s t =
  let order = run_order t in
  let k = ref (Array.length order) and n = ref 1 in
  while
    !k > 0
    &&
    let a = order.(!k - 1) in
    Layout.stride t a = !n && Layout.stride s a = !n
  do
    decr k;
    n := !n * Layout.dim t order.(!k)
  done;
  (Array.sub order 0 !k, !n)

(* Calls [f ps pt] at each index of the axes [outer] of [s] and [t], layouts of
   one shape, the other axes at 0: [ps] and [pt] are its positions in [s] and
   [t]. The last axis of [outer] varies fastest. *)
let walk s t outer f =
  let k = Array.length outer in
  let idx = Array.make k 0 in
  let ps = ref (Layout.offset s) and pt = ref (Layout.offset t) in
  let go = ref (Layout.numel t > 0) in
  while !go do
    f !ps !pt;
    let j = ref (k - 1) in
    while !j >= 0 && idx.(!j) + 1 = Layout.dim t outer.(!j) do
      let a = outer.(!j) in
      ps := !ps - (idx.(!j) * Layout.stride s a);
      pt := !pt - (idx.(!j) * Layout.stride t a);
      idx.(!j) <- 0;
      decr j
    done;
    if !j < 0 then go := false
    else begin
      let a = outer.(!j) in
      idx.(!j) <- idx.(!j) + 1;
      ps := !ps + Layout.stride s a;
      pt := !pt + Layout.stride t a
    end
  done

(* Calls [f ps pt n] for each run of [n] elements consecutive at position [ps]
   of [s] and [pt] of [t], layouts of one shape. Runs follow [t]'s stride order,
   so a transposed view's runs into its packed layout are as long as its rows in
   memory. *)
let iter_runs s t f =
  let outer, n = runs s t in
  walk s t outer (fun ps pt -> f ps pt n)

(* The boxes that copy the runs of [s] into those of [t] ([runs]), of elements
   of [eb] bytes, at byte offsets from positions [s0] of [s] and [t0] of [t].
   The runs along the innermost outer axis are one box's rows where both
   layouts step along it the same way by at least a run, a box starting at its
   least row; otherwise each run is a box. [t] has an element. *)
let boxes s s0 t t0 eb =
  let outer, n = runs s t in
  let k = Array.length outer and width = n * eb in
  let ss, st =
    if k = 0 then (0, 0)
    else (Layout.stride s outer.(k - 1), Layout.stride t outer.(k - 1))
  in
  let grouped =
    k > 0 && abs ss >= n && abs st >= n && Bool.equal (ss > 0) (st > 0)
  in
  let rows = if grouped then Layout.dim t outer.(k - 1) else 1 in
  let outer = if grouped then Array.sub outer 0 (k - 1) else outer in
  (* A box's first row is its least: the last of a run of rows both layouts
     step backwards. *)
  let back = if grouped && ss < 0 then rows - 1 else 0 in
  let src_pitch = if grouped then abs ss * eb else width
  and dst_pitch = if grouped then abs st * eb else width in
  let bs = Array.make (Layout.numel t / (n * rows)) (Buffer.row 0) in
  let i = ref 0 in
  walk s t outer (fun ps pt ->
      bs.(!i) <-
        {
          Buffer.src_offset = (ps + (back * ss) - s0) * eb;
          dst_offset = (pt + (back * st) - t0) * eb;
          width;
          rows;
          src_pitch;
          dst_pitch;
        };
      incr i);
  bs

(* [a] with each axis of stride 0 cut to its first index: [transfer] stores
   into a destination that reaches each element once. *)
let first_of_broadcast a =
  let l = a.layout in
  let range i =
    let d = Layout.dim l i in
    {
      Move.start = 0;
      count = (if Layout.stride l i = 0 then min d 1 else d);
      step = 1;
    }
  in
  match move (Move.Slice (Iarray.init (Layout.rank l) range)) a with
  | Some a -> a
  | None -> assert false

(* The host gathers [a]'s runs into [h], on memory it addresses, each run
   brought into one host scratch by [Buffer.copy] where the host does not
   address [a]'s memory. *)
let gather_runs fn a h =
  let bits = Dtype.bits a.dtype in
  let covering ps n =
    (ps * bits / 8, ((((ps + n) * bits) + 7) / 8) - (ps * bits / 8))
  in
  let scratch = ref None in
  iter_runs a.layout h.layout (fun ps pt n ->
      let first, length = covering ps n in
      let s =
        match !scratch with
        | Some s -> s
        | None ->
            (* A run of [n] elements covers at most one byte more than they
               fill. *)
            let s = Buffer.create Rig.host (Dtype.bytes a.dtype n + 1) in
            scratch := Some s;
            s
      in
      let run = Buffer.view s ~first:0 ~length in
      Buffer.copy ~src:(Buffer.view a.buffer ~first ~length) ~dst:run;
      let src =
        {
          a with
          layout =
            Layout.v ~offset:(ps - (8 * first / bits)) ~strides:[| 1 |] [| n |];
          buffer = run;
        }
      and dst =
        { h with layout = Layout.v ~offset:pt ~strides:[| 1 |] [| n |] }
      in
      blit fn ~src ~dst)

(* The bytes [a]'s layout spans, as a view of its buffer. *)
let span_bytes a =
  let bits = Dtype.bits a.dtype and lo, hi = Layout.span a.layout in
  let first = lo * bits / 8 and last = ((hi * bits) + 7) / 8 in
  Buffer.view a.buffer ~first ~length:(last - first)

(* Copies [src]'s elements, of a byte or more, into [dst], of the same shape,
   whose bytes they share none of, where a device copies: one
   [Buffer.submit_copy] of [boxes] between the bytes the two layouts span,
   which rig runs on the path between their devices. *)
let copy_runs src dst =
  let eb = Dtype.bits src.dtype / 8 in
  let s0, _ = Layout.span src.layout and t0, _ = Layout.span dst.layout in
  ignore
    (Buffer.submit_copy ~src:(span_bytes src) ~dst:(span_bytes dst)
       (boxes src.layout s0 dst.layout t0 eb)
      : Rig.Point.t)

(* Stores [s], laid out in [t]'s stride order, into [t]: by [copy_runs],
   unless the host addresses [t] and the runs are several, where the host
   blits from [s], brought to it by one copy if the host does not address
   it. *)
let spread fn s t =
  let _, n = runs s.layout t.layout in
  if n = Layout.numel t.layout || not (hosted t.buffer) then copy_runs s t
  else begin
    let h =
      if hosted s.buffer then s
      else { s with buffer = Buffer.create Rig.host (Buffer.length s.buffer) }
    in
    if h != s then copy_bytes ~src:s.buffer ~dst:h.buffer;
    blit fn ~src:h ~dst:t
  end

(* The host blits where it addresses both arrays' memory. Otherwise a device
   that computes and that the host does not address copies its source's runs
   ([copy_runs]); the host gathers them from other memory, and of a sub-byte
   array, whose runs a byte copy cannot cut. Each gathers into the destination
   where it reaches it, otherwise into a buffer laid out in the destination's
   stride order on its side, which [spread] moves, unless the runs between the
   two arrays are already as long. *)
let transfer_by fn ~src ~dst =
  live fn src.buffer;
  live fn dst.buffer;
  if Layout.shape src.layout <> Layout.shape dst.layout then
    refused fn Shape_mismatch [ Any dst; Any src ];
  if not (Layout.is_distinct dst.layout) then
    refused fn Repeated_elements [ Any dst; Any src ];
  let bits = Dtype.bits src.dtype in
  if bits < 8 && not (hosted dst.buffer) then
    refused fn Off_host [ Any dst; Any src ];
  if hosted src.buffer && hosted dst.buffer then blit fn ~src ~dst
  else if Layout.numel dst.layout > 0 then begin
    let packs =
      bits >= 8 && (not (hosted src.buffer)) && Rig.computes (device src)
    in
    let gather t =
      if packs then copy_runs src t
      else if hosted src.buffer then blit fn ~src ~dst:t
      else gather_runs fn src t
    in
    let direct =
      if packs then Rig.equal (device src) (device dst) else hosted dst.buffer
    in
    let move () =
      if direct then gather dst
      else
        let l = packed dst.layout in
        if snd (runs src.layout dst.layout) >= snd (runs l dst.layout) then
          copy_runs src dst
        else begin
          let side = if packs then device src else Rig.host in
          let n = Layout.numel dst.layout in
          let buffer = Buffer.create side (Dtype.bytes dst.dtype n) in
          let stage = { dst with layout = l; buffer } in
          gather stage;
          spread fn stage dst
        end
    in
    (* The door refuses a destination that shares a byte with the source
       unless the two are identical, as the host's blit does, and then the
       transfer stores each element onto itself. *)
    let moved () =
      if not (Buffer.overlaps (span_bytes src) (span_bytes dst)) then move ()
    in
    match door ~written:[| Any dst |] ~read:[| Any src |] moved () with
    | Done -> ()
    | refusal -> refused fn refusal [ Any dst; Any src ]
  end

let transfer ~src ~dst = transfer_by "Nx_array.transfer" ~src ~dst

let copy a =
  let dst = create (device a) a.dtype (Layout.shape a.layout) in
  transfer_by "Nx_array.copy" ~src:a ~dst;
  dst

(* [a]'s elements laid out [packed] on [d]. The host gathers a sub-byte array it
   cannot store on [d] into a buffer of its own, moved whole. *)
let move_packed d a =
  let t = packed a.layout and bits = Dtype.bits a.dtype in
  let n = elements a.layout in
  let fresh d =
    let buffer = Buffer.create d (Dtype.bytes a.dtype n) in
    zero_tail bits n buffer;
    { a with layout = t; buffer }
  in
  let dst = fresh d in
  let into =
    if bits < 8 && host_address dst.buffer < 0 then fresh Rig.host else dst
  in
  transfer_by "Nx_array.to_device" ~src:(first_of_broadcast a)
    ~dst:(first_of_broadcast into);
  if into != dst then copy_bytes ~src:into.buffer ~dst:dst.buffer;
  dst

(* [a] over a copy of the bytes from its first to its last position, one
   copy but for a sub-byte span's shared end bytes ([copy_span]), its layout
   shifted to the copy. *)
let move_span d a =
  let l = a.layout and bits = Dtype.bits a.dtype in
  let lo, hi = Layout.span l in
  let first = lo * bits / 8 and last = ((hi * bits) + 7) / 8 in
  let buffer = Buffer.create d (last - first) in
  if last > first then begin
    let src = Buffer.view a.buffer ~first ~length:(last - first) in
    let head = lo * bits mod 8 <> 0 and tail = hi * bits mod 8 <> 0 in
    reading "Nx_array.to_device" a (fun () ->
        copy_span ~head ~tail src buffer)
  end;
  let offset = Layout.offset l - (8 * first / bits) in
  let layout = Layout.v ~offset ~strides:(Layout.strides l) (Layout.shape l) in
  v a.dtype layout buffer

(* The span is copied where it holds no more bytes than the elements, as for
   whole rows, a transpose or an overlapping window; other layouts are packed
   and filled by [transfer]. *)
let to_device d a =
  live "Nx_array.to_device" a.buffer;
  let l = a.layout and bits = Dtype.bits a.dtype in
  let lo, hi = Layout.span l in
  let span = (((hi * bits) + 7) / 8) - (lo * bits / 8) in
  if span <= Dtype.bytes a.dtype (elements l) then move_span d a
  else move_packed d a

let borrow d a =
  live "Nx_array.borrow" a.buffer;
  match Buffer.borrow d a.buffer with
  | None -> None
  | Some buffer when buffer == a.buffer -> Some a
  | Some buffer -> Some { a with buffer }

(* Bigarrays *)

(* The most axes a Bigarray has. *)
let bigarray_max_rank = 16

(* The bytes of [a], a C-contiguous array of elements of [k]'s width. *)
let bigarray_bytes k a =
  let l = a.layout and w = Bigarray.kind_size_in_bytes k in
  let first = Layout.offset l * w and length = Layout.numel l * w in
  if first = 0 && length = Buffer.length a.buffer then a.buffer
  else Buffer.view a.buffer ~first ~length

(* A vector's bigarray is the flat one, with no reshape to allocate. *)
let of_flat a flat =
  let g = Bigarray.genarray_of_array1 flat in
  if Layout.rank a.layout = 1 then g
  else Bigarray.reshape g (Iarray.to_array (Layout.shape a.layout))

(* Raises naming [fn] unless a bigarray of [l]'s shape lies over the bytes of
   an array laid out by [l]: C-contiguous, of at most Bigarray's axes. *)
let bigarray_fits fn l =
  if not (Layout.is_contiguous l) then
    invalid_argf "%s: %a is not C-contiguous" fn Layout.pp l;
  if Layout.rank l > bigarray_max_rank then
    invalid_argf "%s: %d axes, a bigarray has at most %d" fn (Layout.rank l)
      bigarray_max_rank

(* The kind [a]'s dtype stores, which its type fixes, or a raise naming [fn]
   for a dtype Bigarray has no kind for. *)
let kind (type v s) fn (a : (v, s) t) : (v, s) Bigarray.kind =
  match Dtype.to_bigarray_kind a.dtype with
  | Some k -> k
  | None -> invalid_argf "%s: %a has no Bigarray kind" fn Dtype.pp a.dtype

let bigarray a =
  let fn = "Nx_array.bigarray" in
  live fn a.buffer;
  bigarray_fits fn a.layout;
  let k = kind fn a in
  let d = device a in
  if not (Rig.equal d Rig.host) then
    invalid_argf "%s: the array is on %a, and a bigarray needs Rig.host's memory"
      fn Rig.pp d;
  of_flat a (Buffer.export (bigarray_bytes k a) k)

(* [read] and [write] take the door's claims as [door] does, inline, so that a
   call allocates no closure or cell for its result; a refusal names the
   function. Under the claims, rig's read or write waits and passes the bytes.
   An array with no element has no byte to address, so it is passed wherever
   its memory lies, as rig's read does. *)

let read a f =
  let fn = "Nx_array.read" in
  bigarray_fits fn a.layout;
  let k = kind fn a in
  let operands = [| Any a |] in
  match admit [||] operands with
  | Done -> (
      match
        if Layout.numel a.layout > 0 && host_address a.buffer < 0 then
          refused fn Off_host [ Any a ];
        Buffer.read (bigarray_bytes k a) k (fun flat -> f (of_flat a flat))
      with
      | r ->
          release [||] operands;
          r
      | exception e ->
          release [||] operands;
          Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ()))
  | refusal -> refused fn refusal [ Any a ]

let write a f =
  let fn = "Nx_array.write" in
  bigarray_fits fn a.layout;
  let k = kind fn a in
  let operands = [| Any a |] in
  match admit operands [||] with
  | Done -> (
      match
        if Layout.numel a.layout > 0 && host_address a.buffer < 0 then
          refused fn Off_host [ Any a ];
        Buffer.write (bigarray_bytes k a) k (fun flat -> f (of_flat a flat))
      with
      | r ->
          release operands [||];
          r
      | exception e ->
          release operands [||];
          Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ()))
  | refusal -> refused fn refusal [ Any a ]

let of_bigarray dt g =
  let dims = Bigarray.Genarray.dims g in
  let n = Array.fold_left ( * ) 1 dims in
  let flat = Bigarray.reshape_1 g n in
  v dt (Layout.contiguous (Iarray.of_array dims)) (Buffer.of_bigarray flat)
