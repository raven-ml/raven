(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A layout is the bytes of nx_array.h's nx_layout, words in native order: the
   rank, the flags (CONTIGUOUS 1, DISTINCT 2, EMPTY 4), the offset in elements,
   the span's lo and hi, then the rank extents and the rank strides.

   Every layout is built by [finish], which puts it in canonical form: an axis
   of extent 1 has stride 0, and a layout with no element has offset 0 and every
   stride 0. *)

type t = string

let max_rank = Shape.max_rank
let contiguous_flag = 1
let distinct_flag = 2
let empty_flag = 4
let header = 5
let word l i = Int64.to_int (String.get_int64_ne l (8 * i))
let rank l = word l 0
let flags l = word l 1
let offset l = word l 2
let span l = (word l 3, word l 4)
let unsafe_dim l i = word l (header + i)
let unsafe_stride l i = word l (header + rank l + i)
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

let shape l = Array.init (rank l) (unsafe_dim l)
let strides l = Array.init (rank l) (unsafe_stride l)
let is_contiguous l = flags l land contiguous_flag <> 0
let is_distinct l = flags l land distinct_flag <> 0
let equal = String.equal
let hash l = Hashtbl.hash l

(* Building *)

let create r = Bytes.create (8 * (header + (2 * r)))
let set b i v = Bytes.set_int64_ne b (8 * i) (Int64.of_int v)
let get b i = Int64.to_int (Bytes.get_int64_ne b (8 * i))
let overflow fn = invalid_argf "%s: the layout's positions overflow" fn

let add fn a x =
  if (x > 0 && a > max_int - x) || (x < 0 && a < min_int - x) then overflow fn;
  a + x

(* Puts [b], whose rank, offset, extents and strides are set, in canonical form,
   and sets its flags and span. Extents are non-negative and their product fits;
   a stride of an axis of extent above 1 times the extent fits. The span is
   checked to fit. Movements build a layout per call, so this allocates
   nothing. *)
let finish fn b =
  let r = get b 0 in
  let strides = header + r in
  let empty = ref false in
  for i = 0 to r - 1 do
    let d = get b (header + i) in
    if d = 0 then empty := true else if d = 1 then set b (strides + i) 0
  done;
  if !empty then begin
    set b 1 (contiguous_flag lor distinct_flag lor empty_flag);
    set b 2 0;
    set b 3 0;
    set b 4 0;
    for i = 0 to r - 1 do
      set b (strides + i) 0
    done
  end
  else begin
    (* The span, from the offset and each axis's reach; C order, where stride i
       is the product of the later extents, or 0 for an axis of extent 1. *)
    let lo = ref (get b 2) and hi = ref (get b 2) in
    let contiguous = ref true and run = ref 1 in
    for i = r - 1 downto 0 do
      let d = get b (header + i) and st = get b (strides + i) in
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
      let dj = get b (header + j) in
      if dj > 1 then begin
        let sj = abs (get b (strides + j)) and below = ref 0 in
        for k = 0 to r - 1 do
          let dk = get b (header + k) and sk = abs (get b (strides + k)) in
          if dk > 1 && (sk < sj || (sk = sj && k < j)) then
            below := !below + ((dk - 1) * sk)
        done;
        if sj <= !below then distinct := false
      end
    done;
    set b 1
      ((if !contiguous then contiguous_flag else 0)
      lor if !distinct then distinct_flag else 0);
    set b 3 !lo;
    set b 4 (add fn !hi 1)
  end;
  Bytes.unsafe_to_string b

(* Constructors *)

let pp_ints = Shape.pp

let contiguous s =
  let r = Array.length s in
  Shape.check_rank "Layout.contiguous" r;
  ignore (Shape.numel "Layout.contiguous" s);
  let b = create r in
  set b 0 r;
  set b 2 0;
  let run = ref 1 in
  for i = r - 1 downto 0 do
    set b (header + i) s.(i);
    set b (header + r + i) !run;
    run := !run * s.(i)
  done;
  finish "Layout.contiguous" b

let v ?(offset = 0) ~strides s =
  let r = Array.length s in
  Shape.check_rank "Layout.v" r;
  if Array.length strides <> r then
    invalid_argf "Layout.v: %d strides for %d axes" (Array.length strides) r;
  ignore (Shape.numel "Layout.v" s);
  let b = create r in
  set b 0 r;
  set b 2 offset;
  for i = 0 to r - 1 do
    let d = s.(i) and st = strides.(i) in
    if d > 1 && (st = min_int || abs st > max_int / (d - 1)) then
      invalid_argf "Layout.v: stride %d of an axis of extent %d overflows" st d;
    set b (header + i) d;
    set b (header + r + i) st
  done;
  finish "Layout.v" b

(* Movements. Each writes the strides of the result, of shape [s'], from [l]'s
   into the bytes [start] made; [finish] makes it canonical. A stride is formed
   only for an axis of extent above 1, where the result's span, inside [l]'s,
   bounds it. *)

let start s' offset =
  let r = Array.length s' in
  let b = create r in
  set b 0 r;
  set b 2 offset;
  for i = 0 to r - 1 do
    set b (header + i) (Array.unsafe_get s' i)
  done;
  b

let moved = finish "Layout.move"

(* Strides for [s'] over the same elements in the same C order as [l], if
   strides express it. Axes of extent 1 are left out on both sides; the others
   are matched in groups of equal product, and a group of [l]'s axes must lay
   out one run. *)
let reshape l s' =
  let axes n d = List.filter (fun i -> d i > 1) (List.init n Fun.id) in
  let old = Array.of_list (axes (rank l) (unsafe_dim l)) in
  let fresh = Array.of_list (axes (Array.length s') (Array.get s')) in
  let strides = Array.make (Array.length s') 0 in
  let rec group oi ni =
    oi >= Array.length old
    ||
    let rec grow oj nj po pn =
      if po = pn && oj > oi then (oj, nj)
      else if po <= pn then grow (oj + 1) nj (po * unsafe_dim l old.(oj)) pn
      else grow oj (nj + 1) po (pn * s'.(fresh.(nj)))
    in
    let oj, nj = grow oi ni 1 1 in
    let run = ref true in
    for k = oi to oj - 2 do
      let a = old.(k) and a' = old.(k + 1) in
      if unsafe_stride l a <> unsafe_stride l a' * unsafe_dim l a' then
        run := false
    done;
    !run
    &&
    let st = ref (unsafe_stride l old.(oj - 1)) in
    for k = nj - 1 downto ni do
      strides.(fresh.(k)) <- !st;
      st := !st * s'.(fresh.(k))
    done;
    group oj nj
  in
  if group 0 0 then begin
    let b = start s' (offset l) in
    Array.iteri (fun i st -> set b (header + Array.length s' + i) st) strides;
    Some (moved b)
  end
  else None

let move m l =
  let s' = Move.shape m (shape l) in
  let r = rank l and r' = Array.length s' in
  let stride = header + r' in
  if numel l = 0 || Array.mem 0 s' then Some (moved (start s' 0))
  else
    match m with
    | Move.Reshape _ -> reshape l s'
    | Broadcast _ ->
        let b = start s' (offset l) in
        for i = 0 to r' - 1 do
          let j = i - r' + r in
          set b (stride + i) (if j < 0 then 0 else unsafe_stride l j)
        done;
        Some (moved b)
    | Permute p ->
        let b = start s' (offset l) in
        for i = 0 to r' - 1 do
          set b (stride + i) (unsafe_stride l (Array.unsafe_get p i))
        done;
        Some (moved b)
    | Slice rs ->
        let b = start s' (offset l) in
        let offset = ref (offset l) in
        for i = 0 to r - 1 do
          let x = rs.(i) and st = unsafe_stride l i in
          offset := !offset + (x.start * st);
          set b (stride + i) (if x.count > 1 then st * x.step else 0)
        done;
        set b 2 !offset;
        Some (moved b)
    | Window ws ->
        (* Axis [w.axis] steps from window to window; the appended axis [r + j]
           steps within window [j]. *)
        let b = start s' (offset l) in
        for i = 0 to r - 1 do
          set b (stride + i) (unsafe_stride l i)
        done;
        Array.iteri
          (fun j (w : Move.window) ->
            let st = unsafe_stride l w.axis in
            set b (stride + w.axis) (if s'.(w.axis) > 1 then st * w.step else 0);
            set b (stride + r + j) (if w.size > 1 then st * w.dilation else 0))
          ws;
        Some (moved b)

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
