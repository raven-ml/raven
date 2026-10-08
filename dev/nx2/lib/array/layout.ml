(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every layout is in canonical form: an axis of extent 1 has stride 0, and a
   layout with no element has offset 0 and every stride 0. [finish] puts any
   layout in it; [contiguous] builds C order in it directly. A layout's arrays
   are its own: no caller's array is kept or returned, and nothing writes them
   once it is built.

   C reads the fields in this order (nx_layout.h): the two change together. *)

type t = {
  shape : int array;
  strides : int array;
  offset : int;
  flags : int;
  lo : int;
  hi : int;
}

let max_rank = Shape.max_rank
let contiguous_flag = 1
let distinct_flag = 2
let empty_flag = 4
let rank l = Array.length l.shape
let flags l = l.flags
let offset l = l.offset
let span l = (l.lo, l.hi)
let unsafe_dim l i = Array.unsafe_get l.shape i
let unsafe_stride l i = Array.unsafe_get l.strides i
let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let dim l i =
  if i < 0 || i >= rank l then
    invalid_argf "Layout.dim: axis %d of %d" i (rank l);
  unsafe_dim l i

let stride l i =
  if i < 0 || i >= rank l then
    invalid_argf "Layout.stride: axis %d of %d" i (rank l);
  unsafe_stride l i

let numel l =
  if flags l land empty_flag <> 0 then 0
  else
    let n = ref 1 in
    for i = 0 to rank l - 1 do
      n := !n * unsafe_dim l i
    done;
    !n

let shape l = Shape.copy l.shape
let strides l = Shape.copy l.strides
let is_contiguous l = flags l land contiguous_flag <> 0
let is_distinct l = flags l land distinct_flag <> 0

let ints_equal (a : int array) (b : int array) =
  let n = Array.length a in
  n = Array.length b
  &&
  let i = ref 0 in
  while !i < n && Array.unsafe_get a !i = Array.unsafe_get b !i do
    incr i
  done;
  !i = n

let equal l l' =
  l == l'
  || l.offset = l'.offset
     && ints_equal l.shape l'.shape
     && ints_equal l.strides l'.strides

(* The fields [equal] compares, mixed into one int: no tuple to hash. *)
let hash l =
  let h = ref l.offset in
  for i = 0 to rank l - 1 do
    h := (!h * 31) + unsafe_dim l i;
    h := (!h * 31) + unsafe_stride l i
  done;
  Hashtbl.hash !h

(* Building *)

let overflow fn = invalid_argf "%s: the layout's positions overflow" fn

let add fn a x =
  if (x > 0 && a > max_int - x) || (x < 0 && a < min_int - x) then overflow fn;
  a + x

(* The layout of [shape], [strides] and [offset] in canonical form, with its
   flags and span. It takes [shape] and [strides] as its own and writes
   [strides]. Extents are non-negative and their product fits; a stride of an
   axis of extent above 1 times the extent fits. The span is checked to fit. *)
let finish fn shape strides offset =
  let r = Array.length shape in
  let empty = ref false in
  for i = 0 to r - 1 do
    let d = shape.(i) in
    if d = 0 then empty := true else if d = 1 then strides.(i) <- 0
  done;
  if !empty then begin
    Array.fill strides 0 r 0;
    {
      shape;
      strides;
      offset = 0;
      flags = contiguous_flag lor distinct_flag lor empty_flag;
      lo = 0;
      hi = 0;
    }
  end
  else begin
    (* The span, from the offset and each axis's reach; C order, where stride i
       is the product of the later extents, or 0 for an axis of extent 1. *)
    let lo = ref offset and hi = ref offset in
    let contiguous = ref true and run = ref 1 in
    for i = r - 1 downto 0 do
      let d = shape.(i) and st = strides.(i) in
      let reach = (d - 1) * st in
      if reach < 0 then lo := add fn !lo reach else hi := add fn !hi reach;
      if st <> if d = 1 then 0 else !run then contiguous := false;
      run := !run * d
    done;
    (* A position counts elements from a buffer's first byte: a negative one
       lies outside every buffer. With [0 <= lo] and [hi] fitting, every reach
       below sums to at most [hi - 1 - lo]: no sum overflows. *)
    if !lo < 0 then invalid_argf "%s: the layout reaches a negative position" fn;
    (* Distinct: each axis of extent above 1 has a stride above the reach of the
       axes of smaller stride, ties broken by axis. *)
    let distinct = ref true in
    for j = 0 to r - 1 do
      let dj = shape.(j) in
      if dj > 1 then begin
        let sj = abs strides.(j) and below = ref 0 in
        for k = 0 to r - 1 do
          let dk = shape.(k) and sk = abs strides.(k) in
          if dk > 1 && (sk < sj || (sk = sj && k < j)) then
            below := !below + ((dk - 1) * sk)
        done;
        if sj <= !below then distinct := false
      end
    done;
    let flags =
      (if !contiguous then contiguous_flag else 0)
      lor if !distinct then distinct_flag else 0
    in
    { shape; strides; offset; flags; lo = !lo; hi = add fn !hi 1 }
  end

(* Constructors *)

let pp_ints = Shape.pp

(* A C-order layout is canonical, contiguous and distinct, with span [[0, n)]:
   [contiguous] builds it without [finish]'s general checks. *)
let contiguous s =
  let r = Array.length s in
  Shape.check_rank "Layout.contiguous" r;
  let n = Shape.numel "Layout.contiguous" s in
  let shape = Shape.zeros r and strides = Shape.zeros r in
  let run = ref 1 in
  for i = r - 1 downto 0 do
    let d = Array.unsafe_get s i in
    Array.unsafe_set shape i d;
    if n > 0 && d > 1 then Array.unsafe_set strides i !run;
    run := !run * d
  done;
  let flags = contiguous_flag lor distinct_flag in
  if n = 0 then
    { shape; strides; offset = 0; flags = flags lor empty_flag; lo = 0; hi = 0 }
  else { shape; strides; offset = 0; flags; lo = 0; hi = n }

let v ?(offset = 0) ~strides s =
  let r = Array.length s in
  Shape.check_rank "Layout.v" r;
  if Array.length strides <> r then
    invalid_argf "Layout.v: %d strides for %d axes" (Array.length strides) r;
  ignore (Shape.numel "Layout.v" s);
  for i = 0 to r - 1 do
    let d = s.(i) and st = strides.(i) in
    if d > 1 && (st = min_int || abs st > max_int / (d - 1)) then
      invalid_argf
        "Layout.v: axis %d has extent %d and stride %d: its reach overflows" i d
        st
  done;
  finish "Layout.v" (Shape.copy s) (Shape.copy strides) offset

(* Movements. Each writes the strides of the result, of shape [s'], from [l]'s;
   [finish] makes it canonical. A stride is formed only for an axis of extent
   above 1, where the result's span, inside [l]'s, bounds it. *)

let moved = finish "Layout.move"

(* The first axis of [s] from [i] whose extent is not 1, or [s]'s rank. *)
let past_ones s i =
  let i = ref i in
  while !i < Array.length s && Array.unsafe_get s !i = 1 do
    incr i
  done;
  !i

(* Strides for [s'] over the same elements in the same C order as [l], if
   strides express it. Axes of extent 1 are left out on both sides; the others
   are matched in groups of equal product, and a group of [l]'s axes must lay
   out one run. *)
let reshape l s' =
  let r = rank l in
  let strides = Shape.zeros (Array.length s') in
  (* [i] and [j] walk [l]'s axes and [s']'s, skipping extents of 1. *)
  let i = ref (past_ones l.shape 0) and j = ref (past_ones s' 0) in
  let runs = ref true in
  while !runs && !i < r do
    (* Grow a group of each side's axes until their products are equal.
       Move.shape checked that the shapes have one number of elements, so
       neither side runs out first. *)
    let i0 = !i and j0 = !j and po = ref 1 and pn = ref 1 and last = ref !i in
    while not (!po = !pn && !i > i0) do
      if !po <= !pn then begin
        let a = !i in
        if a > i0 && unsafe_stride l !last <> unsafe_stride l a * unsafe_dim l a
        then runs := false;
        po := !po * unsafe_dim l a;
        last := a;
        i := past_ones l.shape (a + 1)
      end
      else begin
        pn := !pn * Array.unsafe_get s' !j;
        j := past_ones s' (!j + 1)
      end
    done;
    (* The group of [l] lays out one run: the new axes split it in C order. *)
    let st = ref (unsafe_stride l !last) in
    for k = !j - 1 downto j0 do
      let d = Array.unsafe_get s' k in
      if d > 1 then begin
        Array.unsafe_set strides k !st;
        st := !st * d
      end
    done
  done;
  if !runs then Some (moved s' strides (offset l)) else None

let move m l =
  let s' = Move.shape m (shape l) in
  let r = rank l and r' = Array.length s' in
  let st = Shape.zeros r' in
  if numel l = 0 || Array.mem 0 s' then Some (moved s' st 0)
  else
    match m with
    | Move.Reshape _ -> reshape l s'
    | Broadcast _ ->
        for i = 0 to r' - 1 do
          let j = i - r' + r in
          st.(i) <- (if j < 0 then 0 else unsafe_stride l j)
        done;
        Some (moved s' st (offset l))
    | Permute p ->
        for i = 0 to r' - 1 do
          st.(i) <- unsafe_stride l (Array.unsafe_get p i)
        done;
        Some (moved s' st (offset l))
    | Slice rs ->
        let offset = ref (offset l) in
        for i = 0 to r - 1 do
          let x = rs.(i) and s = unsafe_stride l i in
          offset := !offset + (x.start * s);
          st.(i) <- (if x.count > 1 then s * x.step else 0)
        done;
        Some (moved s' st !offset)
    | Window ws ->
        (* Axis [w.axis] steps from window to window; the appended axis [r + j]
           steps within window [j]. *)
        for i = 0 to r - 1 do
          st.(i) <- unsafe_stride l i
        done;
        Array.iteri
          (fun j (w : Move.window) ->
            let s = unsafe_stride l w.axis in
            st.(w.axis) <- (if s'.(w.axis) > 1 then s * w.step else 0);
            st.(r + j) <- (if w.size > 1 then s * w.dilation else 0))
          ws;
        Some (moved s' st (offset l))

(* Coalescing *)

external coalesce_into : t array -> int array -> int = "nx_array_coalesce"
[@@noalloc]

let shape_code = 10
let arity_code = 11

let coalesce ls =
  let n = Array.length ls in
  let out = Array.make (1 + max_rank + (n * (1 + max_rank))) 0 in
  let e = coalesce_into ls out in
  if e = shape_code then
    invalid_arg "Layout.coalesce: layouts of different shapes";
  if e = arity_code then
    invalid_argf "Layout.coalesce: %d layouts, not 1 to 4" n;
  let r = out.(0) in
  let shape = Array.sub out 1 r in
  Array.init n (fun k ->
      let at = 1 + r + (k * (1 + r)) in
      v ~offset:out.(at) ~strides:(Array.sub out (at + 1) r) shape)

let pp ppf l =
  Format.fprintf ppf "{shape = %a; strides = %a; offset = %d}" pp_ints (shape l)
    pp_ints (strides l) (offset l)
