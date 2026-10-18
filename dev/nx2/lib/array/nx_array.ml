(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Dtype = Dtype
module Move = Move
module Layout = Layout
module Buffer = Rig.Buffer

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* nx_array_stubs.c reads the fields in this order. *)
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
  | Bad_arity

let reason = function
  | Done | Declined -> invalid_arg "Nx_array.refused: not a refusal"
  | Wrong_dtype -> "an operand's dtype is not the one the kernel loads"
  | Dead_buffer -> "an operand's buffer is dead"
  | Off_host -> "the host does not address an operand's memory"
  | Held_exclusive -> "an operand's memory is held exclusive"
  | Read_only -> "a written operand's memory is read-only"
  | Repeated_elements -> "a written operand reaches an element twice"
  | Overlapping -> "a written operand shares bytes with another operand"
  | Shape_mismatch -> "the operands' shapes differ"
  | Bad_arity -> "no operand, or too many"

let pp_operand ppf (Any a) =
  Format.fprintf ppf "%a %a" Dtype.pp a.dtype Shape.pp (Layout.shape a.layout)

(* The reason a [Dead_buffer] gives: each dead operand, numbered from 1 in
   [operands]' order, with the reason its consumer gave. *)
let dead operands =
  let ds =
    List.filter_map Fun.id
      (List.mapi
         (fun i (Any a) ->
           Option.map (fun why -> (i + 1, why)) (Buffer.dead a.buffer))
         operands)
  in
  match List.rev ds with
  | [] -> reason Dead_buffer
  | [ (i, why) ] -> Printf.sprintf "operand %d was consumed, %s" i why
  | (last, _) :: before ->
      Printf.sprintf "operands %s and %d were consumed: %s"
        (String.concat ", "
           (List.rev_map (fun (i, _) -> string_of_int i) before))
        last
        (String.concat ", "
           (List.map (fun (i, why) -> Printf.sprintf "%d %s" i why) ds))

let refused name answer operands =
  let why =
    match answer with Dead_buffer -> dead operands | _ -> reason answer
  in
  invalid_argf "%s: %s (%a)" name why
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
       pp_operand)
    operands

(* The OCaml door. [admit] claims every array or none, checks under the
   claims what nx_read checks of a written array, and waits for nothing. An
   asynchronous exception (a signal handler's, a finaliser's) is raised at a
   poll point: an allocation, a function's entry or a loop's back edge. No
   claim the door takes is held across one outside the handler that ends
   it. *)

external admit : any array -> any array -> answer = "nx_array_admit"

external release_all : any array -> any array -> unit = "nx_array_release_all"
[@@noalloc]

let door ~written ~read f x =
  match admit written read with
  | Done -> (
      match f x with
      | () ->
          release_all written read;
          Done
      | exception e ->
          release_all written read;
          Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ()))
  | refusal -> refusal

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
    Buffer.blit_from_string "\000" 0 b (Buffer.length b - 1) 1

(* A fresh array laid out by [layout], C-contiguous at offset 0. *)
let alloc ?memory d dtype layout =
  let n = Layout.numel layout in
  let buffer = Buffer.create ?memory d (Dtype.bytes dtype n) in
  zero_tail (Dtype.bits dtype) n buffer;
  { dtype; layout; buffer }

let create ?memory d dtype s = alloc ?memory d dtype (Layout.contiguous s)

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
  Layout.v ~offset:(Layout.offset l * r) ~strides shape

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
      if !fits then Some (Layout.v ~offset:(Layout.offset l / r) ~strides shape)
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
  if Array.length idx <> r then
    invalid_argf "%s: index %a for %d axes" fn Shape.pp idx r;
  let p = ref (Layout.offset l) in
  for i = 0 to r - 1 do
    let j = idx.(i) in
    if j < 0 || j >= Layout.dim l i then
      invalid_argf "%s: index %a outside shape %a" fn Shape.pp idx Shape.pp
        (Layout.shape l);
    p := !p + (j * Layout.stride l i)
  done;
  !p

(* A refusal of [get_at], raised by C, which allocates no result block. *)
exception Refused of answer

let () = Callback.register_exception "Nx_array.Refused" (Refused Done)

(* [a]'s element at position [p], through the door. Raises [Refused]. *)
external get_at : ('v, 's) t -> int -> 'v = "nx_array_get"

(* Stores [x] at [a]'s position [p], through the door. *)
external set_at : ('v, 's) t -> int -> 'v -> answer = "nx_array_set"

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

let get a idx =
  let fn = "Nx_array.get" in
  match get_at a (position fn a idx) with
  | x -> x
  | exception Refused refusal -> refused fn refusal [ Any a ]

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
  (* A caller's array is read once: the C store loop takes [values] to hold an
     element per index of the layout it writes, counted on the same layout. *)
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

(* The host gathers, so memory it does not address is refused before the copy is
   allocated on [a]'s device. *)
let copy a =
  let fn = "Nx_array.copy" in
  live fn a.buffer;
  if Layout.numel a.layout > 0 && host_address a.buffer < 0 then
    refused fn Off_host [ Any a ];
  let dst = create (device a) a.dtype (Layout.shape a.layout) in
  match copy_into dst a with
  | Done -> dst
  | refusal -> refused fn refusal [ Any dst; Any a ]

let blit ~src ~dst =
  let fn = "Nx_array.blit" in
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
  if (not (head || tail)) || host_address src < 0 then Buffer.copy ~src ~dst
  else begin
    Buffer.wait src Rig.Read;
    let i = if head then 1 else 0 and j = if tail then n - 1 else n in
    if j > i then
      Buffer.copy
        ~src:(Buffer.view src ~first:i ~length:(j - i))
        ~dst:(Buffer.view dst ~first:i ~length:(j - i));
    let edge k =
      let byte = Char.chr (load_byte src k) in
      Buffer.fill (Buffer.view dst ~first:k ~length:1) byte
    in
    if head then edge 0;
    if tail && (n > 1 || not head) then edge (n - 1)
  end

(* Holds a read claim on [a]'s memory while [f ()] runs: no donation writes
   it meanwhile. The copies under it order themselves after [a]'s work. *)
let reading fn a f =
  match door ~written:[||] ~read:[| Any a |] f () with
  | Done -> ()
  | refusal -> refused fn refusal [ Any a ]

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
  Layout.v ~offset:!offset ~strides (Layout.shape l)

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

(* Calls [f ps pt n] for each run of [n] elements consecutive at position [ps]
   of [s] and [pt] of [t], layouts of one shape. Runs follow [t]'s stride order,
   so a transposed view's runs into its packed layout are as long as its rows in
   memory. *)
let iter_runs s t f =
  let outer, n = runs s t in
  let k = Array.length outer in
  let idx = Array.make k 0 in
  let ps = ref (Layout.offset s) and pt = ref (Layout.offset t) in
  let go = ref (Layout.numel t > 0) in
  while !go do
    f !ps !pt n;
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

(* [a] with each axis of stride 0 cut to its first index: [blit] stores into a
   destination that reaches each element once. *)
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
  | None -> invalid_arg "Nx_array: a slice is a view"

(* The host gathers [a]'s runs into [h], on memory it addresses, each run
   brought into one host scratch by [Buffer.copy] where the host does not
   address [a]'s memory. *)
let gather_runs a h =
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
      blit ~src ~dst)

(* [src]'s device copies [src]'s runs into [dst], on that device, in submissions
   of at most [Rig.most_parts] copies. *)
let pack_on_device src dst =
  let d = device src and eb = Dtype.bits src.dtype / 8 in
  let most = Rig.most_parts d in
  let submit parts =
    let s = Rig.Submission.make d (Array.of_list (List.rev parts)) in
    ignore (Rig.submit (Rig.Submission.Run.make s) ~buffers:[||] : Rig.Point.t)
  in
  let parts = ref [] and count = ref 0 in
  iter_runs src.layout dst.layout (fun ps pt n ->
      let view b p =
        Rig.Submission.Buffer (Buffer.view b ~first:(p * eb) ~length:(n * eb))
      in
      parts :=
        Rig.Submission.copy ~src:(view src.buffer ps) ~dst:(view dst.buffer pt)
        :: !parts;
      incr count;
      if !count = most then begin
        submit !parts;
        parts := [];
        count := 0
      end);
  if !parts <> [] then submit !parts

(* Copies each run between [src] and [dst] by its own [Buffer.copy], which
   chooses the path between their devices. *)
let copy_runs src dst =
  let eb = Dtype.bits src.dtype / 8 in
  iter_runs src.layout dst.layout (fun ps pt n ->
      Buffer.copy
        ~src:(Buffer.view src.buffer ~first:(ps * eb) ~length:(n * eb))
        ~dst:(Buffer.view dst.buffer ~first:(pt * eb) ~length:(n * eb)))

(* Stores [s], laid out in [t]'s stride order, into [t]: by a copy per run,
   unless the host addresses [t] and the runs are several, where the host blits
   from [s] brought to it whole. *)
let spread s t =
  let _, n = runs s.layout t.layout in
  if n = Layout.numel t.layout || host_address t.buffer < 0 then copy_runs s t
  else begin
    let h =
      if host_address s.buffer >= 0 then s
      else { s with buffer = Buffer.create Rig.host (Buffer.length s.buffer) }
    in
    if h != s then Buffer.copy ~src:s.buffer ~dst:h.buffer;
    blit ~src:h ~dst:t
  end

(* The host blits where it addresses both arrays' memory. Otherwise a device
   that computes and that the host does not address packs its source's runs; the
   host gathers them from other memory, and of a sub-byte array, whose runs a
   byte copy cannot cut. Each gathers into the destination where it reaches it,
   otherwise into a buffer laid out in the destination's stride order on its
   side, which [spread] moves, unless the runs between the two arrays are
   already as long. *)
let transfer_by fn ~src ~dst =
  live fn src.buffer;
  live fn dst.buffer;
  if Layout.shape src.layout <> Layout.shape dst.layout then
    refused fn Shape_mismatch [ Any dst; Any src ];
  if not (Layout.is_distinct dst.layout) then
    refused fn Repeated_elements [ Any dst; Any src ];
  let hosted b = host_address b >= 0 in
  let bits = Dtype.bits src.dtype in
  if bits < 8 && not (hosted dst.buffer) then
    refused fn Off_host [ Any dst; Any src ];
  if hosted src.buffer && hosted dst.buffer then blit ~src ~dst
  else if Layout.numel dst.layout > 0 then begin
    let packs =
      bits >= 8 && (not (hosted src.buffer)) && Rig.computes (device src)
    in
    let gather t =
      if packs then pack_on_device src t
      else if hosted src.buffer then blit ~src ~dst:t
      else gather_runs src t
    in
    let direct =
      if packs then Rig.equal (device src) (device dst) else hosted dst.buffer
    in
    reading fn src (fun () ->
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
            spread stage dst
          end)
  end

let transfer ~src ~dst = transfer_by "Nx_array.transfer" ~src ~dst

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
  if into != dst then Buffer.copy ~src:into.buffer ~dst:dst.buffer;
  dst

(* [a] over a copy of the bytes from its first to its last position, its layout
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

let bigarray (type v s) (k : (v, s) Bigarray.kind) (a : (v, s) t) :
    (v, s, Bigarray.c_layout) Bigarray.Genarray.t option =
  live "Nx_array.bigarray" a.buffer;
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
    let flat = Buffer.export view k in
    Some (Bigarray.reshape (Bigarray.genarray_of_array1 flat) (Layout.shape l))

let of_bigarray dt g =
  let dims = Bigarray.Genarray.dims g in
  let n = Array.fold_left ( * ) 1 dims in
  let flat = Bigarray.reshape_1 g n in
  v dt (Layout.contiguous dims) (Buffer.of_bigarray flat)
