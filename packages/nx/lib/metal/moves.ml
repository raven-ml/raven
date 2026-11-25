(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog

(* Whether a cast from [x] into [y] keeps its bits: into its own dtype, or
   between integers of one width, which wraps. *)
let same_bits (D.Any x as dx) (D.Any y as dy) =
  let integer d = D.is D.Signed d || D.is D.Unsigned d in
  dx = dy || (integer x && integer y && D.bits x = D.bits y)

(* The operand output [k] of [p] is the bits of, if it is one's. *)
let moved (p : P.t) k =
  let input i =
    match Iarray.get p.nodes i with P.In j -> Some j | _ -> None
  in
  match Iarray.get p.nodes (Iarray.get p.outs k) with
  | P.In j -> Some j
  | Op1 ((Copy | Bitcast), _, i) -> input i
  | Op1 (Cast, y, i) -> (
      match input i with
      | Some j when same_bits (Iarray.get p.ins j) y -> Some j
      | Some _ | None -> None)
  | _ -> None

(* The storage alignment of [dt], of a byte or more: one component's for a
   complex dtype, its width otherwise. *)
let alignment dt = if D.is D.Complex dt then D.bits dt / 16 else D.bits dt / 8

(* A copy's end: a slot, the byte offset in it of the layout's position 0, and a
   layout of [width]-byte elements stored on [align] bytes; the address of the
   slot's first byte modulo 32 ({!Operands.operand}), 0 for the workspace. *)
type site = {
  slot : int;
  at : int;
  layout : L.t;
  width : int;
  align : int;
  residue : int;
}

let site_at ~slot ~base ~residue (D.Any d) l =
  let width = D.bits d / 8 in
  {
    slot;
    at = base + (L.offset l * width);
    layout = l;
    width;
    align = alignment d;
    residue;
  }

let site b (o : Operands.operand) =
  site_at ~slot:(Plan.slot b o.buffer) ~base:0 ~residue:o.residue o.dtype
    o.layout

(* The address of [s]'s position 0 modulo {!Kernels.copy_run_bytes}: its slot's
   residue and its offset decide. The workspace starts on 256 bytes, as every
   buffer of a Metal device does, so its residue is 0. *)
let place s = (s.residue + s.at) land (Kernels.copy_run_bytes - 1)

(* Writes [l]'s axes as the copy kernel reads them, an extent, a stride and the
   extent's divisor for each, as little-endian 64-bit words into [s] from word
   [at]. *)
let axes s ~at l =
  for j = 0 to L.rank l - 1 do
    let word i v =
      let w = at + (Kernels.axis_words * j) + i in
      Bytes.set_int64_le s (8 * w) (Int64.of_int v)
    in
    word Kernels.axis_extent (L.dim l j);
    word Kernels.axis_stride (L.stride l j);
    word Kernels.axis_divisor (Plan.divisor (L.dim l j))
  done

(* The axes of [l], of more than {!Kernels.copy_holds}, as a constant's key and
   bytes. *)
let table l =
  let s = Bytes.create (8 * Kernels.axis_words * L.rank l) in
  axes s ~at:0 l;
  Bytes.unsafe_to_string s

let bigbytes s =
  let n = String.length s in
  let b = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n in
  String.iteri (fun i c -> Bigarray.Array1.unsafe_set b i (Char.code c)) s;
  b

(* The ref of a copy's axes, of [l]: none where the parameters hold them, and a
   constant's otherwise, a table kept for the calls of [l]'s coalesced
   layout. *)
let tail b l =
  if L.rank l <= Kernels.copy_holds then [||]
  else
    let t = table l in
    let c =
      Plan.constant b ~key:("copy " ^ t) ~length:(String.length t) (fun () ->
          bigbytes t)
    in
    [| { Plan.word = Kernels.copy_axes; slot = c; write = false } |]

(* A copy thread's work in float32 operations, measured on the M1 Max at its
   slowest: 101 before its axes, rounded up to 128; and for each axis but the
   outermost, where the copy's indices pass 32 bits, a 64-bit [%] and [/], which
   the GPU emulates ({!Plan.native_divide_ops}). Below them an axis takes a
   divide by its divisor, a multiply and a subtract in 32 bits and its stride's
   64-bit multiply-add: an estimate, 64. *)
let element_ops = 128
let wide_axis_ops = Plan.native_divide_ops
let axis_ops = 64

(* The most elements of a copy's launch whose indices take 32 bits: 2^32 less a
   group, so that each of its threads' indices fits, the last group's past the
   launch's count included; lower in tests. *)
let narrow_threads = Kernels.threads Kernels.Copy
let narrow_most = Atomic.make ((1 lsl 32) - narrow_threads)

let bound_copy ~wide =
  if wide < 9 || wide > 32 then invalid_arg "Nx_metal.bound_copy: wide";
  Atomic.set narrow_most ((1 lsl wide) - narrow_threads);
  Plan.renew ()

(* The kernel of a copy's launch that ends at element [count], of [unit] bytes a
   unit: one of its unit where its indices fit 32 bits. *)
let copy_kernel ~count ~unit =
  if count > Atomic.get narrow_most then Kernels.Copy
  else
    match unit with
    | 1 -> Kernels.Copy_8
    | 2 -> Kernels.Copy_16
    | 4 -> Kernels.Copy_32
    | _ -> Kernels.Copy_64

(* The work of a thread of a copy of [n] elements from a source of [rank] merged
   axes, [ops] besides its axes. *)
let work ~ops ~n ~rank =
  let axis = if n <= Atomic.get narrow_most then axis_ops else wide_axis_ops in
  ops + ((rank - 1) * axis)

(* A widening thread's work besides its axes, an estimate: a copy's, and its
   element's read and cast by a switch on the dtypes. *)
let widen_ops = 512

(* The bytes of [width]-byte elements a copy reads an element from a source
   whose innermost axis steps [s] elements, in loads of [unit] bytes: the
   element's own where it steps by at most one, its stride's where neighbours
   share a line, and a line a load where they share none, as each load of a SIMD
   group then touches a line a thread: a transposed copy of 8M complex128
   elements, two loads of 8 bytes each, takes about the time of one of 16M int64
   elements. *)
let read ~width ~unit s =
  if s = 0 then width
  else if abs s * width < Plan.line then abs s * width
  else width / unit * Plan.line

(* The cost of copying [m] elements in [groups] groups of [per] threads of
   [work] operations each from a source whose innermost axis has extent [e] and
   stride [s]: the destination's bytes and the source's, two lines at each run
   of the innermost axis and at the destination's ends, and each thread's work,
   the last group's idle ones included. *)
let cost ~work ~m ~per ~groups ~width ~unit (e, s) =
  let runs = ((m + e - 1) / e) + 1 in
  {
    Plan.bytes =
      (m * (width + read ~width ~unit s)) + (2 * Plan.line * (runs + 1));
    ops = per * groups * work;
    chain = 0;
  }

(* The most elements a launch of groups of [per] threads copies within
   {!Plan.allowed}, by [cost]'s and {!Plan.seconds}' bounds, less one for
   rounding: a dispatch, traffic of [m (width + read + 2 line / e) + 6 line]
   bytes, and the work of [w] operations a thread, [work], and [x] more an
   element, [extra]: [(m + per - 1) w / all + m x / all] seconds, where [all] is
   [cores lanes op_rate], and [ceil (per / lanes) (w + x) / op_rate] more. *)
let most ?(extra = 0) ~work ~per ~width ~unit (e, s) =
  let line = float Plan.line in
  let w = float work and x = float extra in
  let all = float (Plan.cores * Plan.lanes) *. Plan.op_rate in
  let bytes = float (width + read ~width ~unit s) +. (2. *. line /. float e) in
  let per_element = (bytes /. Plan.bandwidth) +. ((w +. x) /. all) in
  let rounds = float ((per + Plan.lanes - 1) / Plan.lanes) in
  let fixed =
    Plan.dispatch
    +. (6. *. line /. Plan.bandwidth)
    +. (float (per - 1) *. w /. all)
    +. (rounds *. (w +. x) /. Plan.op_rate)
  in
  Int.max 1 (Float.to_int ((Plan.allowed () -. fixed) /. per_element) - 1)

(* Adds the launches of copy_run that copy the [n] consecutive elements of [src]
   from its position 0 into those of [dst]: the blocks of
   {!Kernels.copy_run_bytes} bytes of [dst]'s memory that its bytes reach, as
   many a launch as keep it within the time budget, weighed as the elements of a
   copy of one axis; their estimated seconds are the result. *)
let launch_run b ~n ~dst src =
  let per = Kernels.threads Kernels.Copy_run in
  let width = Kernels.copy_run_bytes in
  let bytes = n * src.width in
  let head = place dst in
  let shift = (place src - head + width) mod width in
  let pieces = if bytes = 0 then 0 else (head + bytes + width - 1) / width in
  let work = element_ops in
  let piece = most ~work ~per ~width ~unit:width (pieces, 1) in
  let first = ref 0 and seconds = ref 0. in
  while !first < pieces do
    let m = Int.min piece (pieces - !first) in
    let w = Plan.words Kernels.Copy_run in
    Plan.set w Kernels.copy_run_dst dst.at;
    Plan.set w Kernels.copy_run_src src.at;
    Plan.set w Kernels.copy_run_first !first;
    Plan.set w Kernels.copy_run_end
      (Int.min bytes (((!first + m) * width) - head));
    Plan.set w Kernels.copy_run_head head;
    Plan.set w Kernels.copy_run_shift shift;
    let groups = (m + per - 1) / per in
    let cost = cost ~work ~m ~per ~groups ~width ~unit:width (pieces, 1) in
    seconds := !seconds +. Plan.seconds ~threads:per ~groups:(groups, 1, 1) cost;
    Plan.launch b Kernels.Copy_run ~groups:(groups, 1, 1) ~cost w
      [|
        { Plan.word = Kernels.copy_run_dst; slot = dst.slot; write = true };
        { Plan.word = Kernels.copy_run_src; slot = src.slot; write = false };
      |];
    first := !first + m
  done;
  !seconds

(* Adds the launches of copy that copy [src]'s [n] elements into consecutive
   ones of [dst], from its position 0: pieces of consecutive elements, each
   within the time budget. Their axes are [src]'s coalesced, the runs along
   which a copy steps by one stride; their estimated seconds are the result. *)
let launch_elements b ~n ~dst src =
  let per = Kernels.threads Kernels.Copy in
  let unit = Int.min src.align dst.align in
  let ls = L.coalesce [| L.contiguous (L.shape src.layout); src.layout |] in
  let l = ls.(1) in
  let rank = L.rank l in
  let inner = (L.dim l (rank - 1), L.stride l (rank - 1)) in
  let tail = tail b l in
  let work = work ~ops:element_ops ~n ~rank in
  let piece = most ~work ~per ~width:src.width ~unit inner in
  let first = ref 0 and seconds = ref 0. in
  while !first < n do
    let m = Int.min piece (n - !first) in
    let kernel = copy_kernel ~count:(!first + m) ~unit in
    let w = Plan.words kernel in
    Plan.set w Kernels.copy_dst dst.at;
    Plan.set w Kernels.copy_src src.at;
    Plan.set w Kernels.copy_first !first;
    Plan.set w Kernels.copy_count (!first + m);
    Plan.set w Kernels.copy_rank rank;
    Plan.set w Kernels.copy_unit unit;
    Plan.set w Kernels.copy_units (src.width / unit);
    if rank <= Kernels.copy_holds then axes w ~at:Kernels.copy_held l;
    let groups = (m + per - 1) / per in
    let cost = cost ~work ~m ~per ~groups ~width:src.width ~unit inner in
    seconds := !seconds +. Plan.seconds ~threads:per ~groups:(groups, 1, 1) cost;
    Plan.launch b kernel ~groups:(groups, 1, 1) ~cost w
      (Array.append
         [|
           { Plan.word = Kernels.copy_dst; slot = dst.slot; write = true };
           { Plan.word = Kernels.copy_src; slot = src.slot; write = false };
         |]
         tail);
    first := !first + m
  done;
  !seconds

(* The copies of rows 16 bytes a thread: copy_rows' rows of consecutive
   elements, and copy_step's of every other element. *)
type rows = Consecutive | Every_other

(* [l]'s axes in bytes of [width]-byte elements, as copy_rows and copy_step read
   them: each stepping its stride's bytes, but for copy_rows the last, a row of
   consecutive elements, its extent's bytes at a stride of 1. *)
let row_axes s ~at ~width rows l =
  let r = L.rank l in
  for j = 0 to r - 1 do
    let e, stride =
      if j = r - 1 && rows = Consecutive then (L.dim l j * width, 1)
      else (L.dim l j, L.stride l j * width)
    in
    let word i v =
      let w = at + (Kernels.axis_words * j) + i in
      Bytes.set_int64_le s (8 * w) (Int64.of_int v)
    in
    word Kernels.axis_extent e;
    word Kernels.axis_stride stride;
    word Kernels.axis_divisor (Plan.divisor e)
  done

(* The fewest bytes of every other element of [width] bytes a row takes for
   copy_step where a block of the destination crosses a row's end, if any. Such
   a block copies its elements one at a time while the rest of its SIMD group,
   512 bytes of the destination, waits. Against copy_8 to copy_64 on an M1 Max,
   on rows of an odd count of elements, copy_step took 1.38, 0.86 and 0.53 of
   the time on uint8 rows of 47, 97 and 263 bytes; 1.27, 0.89 and 0.66 on uint16
   rows of 54, 98 and 262; 1.31, 1.04, 0.95 and 0.92 on float32 rows of 68, 100,
   196 and 260; and 1.15 on int64 rows of 104 to 264. *)
let step_row_bytes = function 1 | 2 -> Some 96 | 4 -> Some 192 | _ -> None

(* Whether a row of [row] bytes of every other element of [width] bytes takes
   copy_step where a block of the destination crosses its end. *)
let crosses_long ~width row =
  match step_row_bytes width with Some least -> row >= least | None -> false

(* How copy_rows or copy_step copies [src]'s [n] elements into [dst], and the
   coalesced axes of [src]'s layout: copy_rows' rows of consecutive elements of
   {!Kernels.copy_run_bytes} bytes or more beside other axes; copy_step's rows
   of every other element, each of 8 bytes or fewer and stored, at both ends, on
   its width, in a copy whose byte indices fit 32 bits, of {!step_row_bytes}
   bytes or more of them where the width has a bound, or of
   {!Kernels.copy_run_bytes} or more where no block of the destination crosses a
   row's end: one row, or rows of a multiple of {!Kernels.copy_run_bytes} bytes
   into a destination that starts on them. *)
let rows ~n ~dst src =
  let l =
    (L.coalesce [| L.contiguous (L.shape src.layout); src.layout |]).(1)
  in
  let r = L.rank l in
  let row = L.dim l (r - 1) * src.width in
  if r >= 2 && L.stride l (r - 1) = 1 && row >= Kernels.copy_run_bytes then
    Some (Consecutive, l)
  else
    let run = Kernels.copy_run_bytes in
    let uncrossed = r = 1 || (row mod run = 0 && place dst = 0) in
    if
      L.stride l (r - 1) = 2
      && src.width <= 8
      && Int.min src.align dst.align = src.width
      && n * src.width <= Atomic.get narrow_most
      && row >= run
      && (uncrossed || crosses_long ~width:src.width row)
    then Some (Every_other, l)
    else None

(* The weight of an element a copy_step block copies alone, where the block
   crosses a row's end: 16 cycles of its SIMD group's 32 lanes, since the rest
   of the group waits. Each such element is a load and a store of its own, as
   the crossing lanes of a group lie in different rows and share no line: an M1
   Max took 87 ps of its whole GPU's time an element, about 13 cycles, alike at
   1 and 4 bytes an element, on rows of 17 to 1032 bytes. *)
let serial_ops = 32 * 16
let rec gcd a b = if b = 0 then a else gcd b (a mod b)

(* The elements copy_step's blocks copy one at a time, of [rows] rows of [row]
   bytes of [width]-byte elements into a destination [head] bytes past a
   multiple of {!Kernels.copy_run_bytes}, in [blocks] blocks: each block's that
   a row's end crosses, and the destination's first and last where they are
   partial; and as many again for each SIMD group of 32 blocks that holds one,
   since a group issues the loop once whatever its count of such blocks: the
   fewer of such blocks and groups, at least as many as the groups that hold
   one. A row's end crosses a block where it lies off a multiple of 16 bytes: of
   the ends at [head + q row], none where [head] lies off a multiple of [g = gcd
   row 16], and one in [16 / g] otherwise. *)
let serial_elements ~width ~head ~row ~rows ~blocks =
  let run = Kernels.copy_run_bytes in
  let ends = Int.max 0 (rows - 1) in
  let period = run / gcd row run in
  let rec first q =
    if q > period then None
    else if (head + (q * row)) mod run = 0 then Some q
    else first (q + 1)
  in
  let aligned =
    match first 1 with
    | Some q when q <= ends -> ((ends - q) / period) + 1
    | Some _ | None -> 0
  in
  let partial =
    Bool.to_int (head <> 0) + Bool.to_int ((head + (rows * row)) mod run <> 0)
  in
  let crossing = ends - aligned + partial in
  let groups = Int.min crossing ((blocks + 31) / 32) in
  (crossing + groups) * (run / width)

(* The copy_step of [width]-byte elements. *)
let step_kernel = function
  | 1 -> Kernels.Copy_step_8
  | 2 -> Kernels.Copy_step_16
  | 4 -> Kernels.Copy_step_32
  | _ -> Kernels.Copy_step_64

(* Adds the launches of copy_rows or copy_step, by [rows], that copy the [n]
   elements of [src], of rows [l], from its position 0 into consecutive ones of
   [dst]: the blocks of {!Kernels.copy_run_bytes} bytes of [dst]'s memory that
   its bytes reach, as many a launch as keep it within the time budget, each
   block's work a copy_run thread's and an axis's for each axis before the row,
   and its reads a run's, twice those of a block for copy_step. copy_rows runs
   where the bytes' indices and the extents fit 32 bits, and copy_rows_wide past
   them; their estimated seconds are the result. *)
let launch_rows b ~n ~dst src (rows, l) =
  let per = Kernels.threads Kernels.Copy_rows in
  let width = Kernels.copy_run_bytes in
  let bytes = n * src.width in
  let head = place dst in
  let rank = L.rank l in
  let lo, hi = L.span l in
  let row = L.dim l (rank - 1) * src.width in
  let pieces = (head + bytes + width - 1) / width in
  let tail =
    if rank <= Kernels.copy_rows_holds then [||]
    else
      let s = Bytes.create (8 * Kernels.axis_words * rank) in
      row_axes s ~at:0 ~width:src.width rows l;
      let t = Bytes.unsafe_to_string s in
      let c =
        Plan.constant b ~key:("copy_rows " ^ t) ~length:(String.length t)
          (fun () -> bigbytes t)
      in
      [| { Plan.word = Kernels.copy_rows_axes; slot = c; write = false } |]
  in
  let work = work ~ops:element_ops ~n:bytes ~rank in
  (* copy_step's blocks share the weight of the elements it copies one at a
     time, rounded up, a block's share in each launch beside its threads'. *)
  let extra =
    if rows = Consecutive then 0
    else
      let serial =
        serial_elements ~width:src.width ~head ~row ~rows:(bytes / row)
          ~blocks:pieces
      in
      ((serial * serial_ops) + pieces - 1) / pieces
  in
  let inner = (Int.max 1 (row / width), if rows = Consecutive then 1 else 2) in
  let piece = most ~extra ~work ~per ~width ~unit:width inner in
  (* copy_rows compares coordinates with extents in 32 bits, so an extent past
     them, the row's in bytes included, takes copy_rows_wide in every launch. *)
  let narrow = Atomic.get narrow_most in
  let extents_fit =
    row <= narrow
    && List.for_all (fun j -> L.dim l j <= narrow) (List.init (rank - 1) Fun.id)
  in
  let first = ref 0 and seconds = ref 0. in
  while !first < pieces do
    let m = Int.min piece (pieces - !first) in
    let stop = Int.min bytes (((!first + m) * width) - head) in
    let kernel =
      if rows = Every_other then step_kernel src.width
      else if extents_fit && stop <= narrow then Kernels.Copy_rows
      else Kernels.Copy_rows_wide
    in
    let w = Plan.words kernel in
    Plan.set w Kernels.copy_rows_dst dst.at;
    Plan.set w Kernels.copy_rows_src src.at;
    Plan.set w Kernels.copy_rows_first !first;
    Plan.set w Kernels.copy_rows_end stop;
    Plan.set w Kernels.copy_rows_head head;
    Plan.set w Kernels.copy_rows_shift (place src);
    Plan.set w Kernels.copy_rows_rank rank;
    Plan.set w Kernels.copy_rows_lo ((lo - L.offset l) * src.width);
    Plan.set w Kernels.copy_rows_hi ((hi - L.offset l) * src.width);
    if rank <= Kernels.copy_rows_holds then
      row_axes w ~at:Kernels.copy_rows_held ~width:src.width rows l;
    let groups = (m + per - 1) / per in
    let cost = cost ~work ~m ~per ~groups ~width ~unit:width inner in
    let cost = { cost with ops = cost.ops + (m * extra) } in
    seconds := !seconds +. Plan.seconds ~threads:per ~groups:(groups, 1, 1) cost;
    Plan.launch b kernel ~groups:(groups, 1, 1) ~cost w
      (Array.append
         [|
           { Plan.word = Kernels.copy_rows_dst; slot = dst.slot; write = true };
           { Plan.word = Kernels.copy_rows_src; slot = src.slot; write = false };
         |]
         tail);
    first := !first + m
  done;
  !seconds

(* Adds the launches that copy [src]'s [n] elements into consecutive ones of
   [dst], from its position 0: copy_run's where [src]'s elements are
   consecutive, copy_rows' or copy_step's where they are rows of 16 bytes or
   more, of consecutive elements or every other one, and copy's otherwise. Their
   estimated seconds are the result. *)
let launch b ~n ~dst src =
  if L.is_contiguous src.layout then launch_run b ~n ~dst src
  else
    match rows ~n ~dst src with
    | Some rows -> launch_rows b ~n ~dst src rows
    | None -> launch_elements b ~n ~dst src

(* Widenings *)

let place (A.Any x) =
  let l = A.layout x and bits = D.bits (A.dtype x) in
  let lo = fst (L.span l) in
  let pos = L.offset l - lo in
  if bits >= 8 then (lo * (bits / 8), pos, 0)
  else
    let skew = Rig.Buffer.offset (A.buffer x) mod 4 in
    let bit = (8 * skew) + (lo * bits) in
    ((bit / 32 * 4) - skew, pos, bit mod 32)

let twin (D.Any d as dt) =
  match d with
  | D.Int4 -> D.Any D.Int8
  | D.Uint4 -> D.Any D.Uint8
  | D.Bit -> D.Any D.Bool
  | D.Float4_e2m1fn -> D.Any D.Float8_e4m3fn
  | _ -> dt

let widen b ~dst ~slot (A.Any x as ax) l =
  let per = Kernels.threads Kernels.Widen in
  let bits = D.bits (A.dtype x) in
  let at, _, phase = place ax in
  let lo = fst (L.span (A.layout x)) in
  let bit = phase + (bits * (L.offset l - lo)) in
  let n = L.numel l in
  let ls = L.coalesce [| L.contiguous (L.shape l); l |] in
  let l = ls.(1) in
  let rank = L.rank l in
  let inner = (L.dim l (rank - 1), L.stride l (rank - 1)) in
  let tail = tail b l in
  let work = work ~ops:widen_ops ~n ~rank in
  let piece = most ~work ~per ~width:1 ~unit:1 inner in
  let (D.Any to_) = twin (D.Any (A.dtype x)) in
  let codes = D.code (A.dtype x) + (256 * D.code to_) in
  let first = ref 0 in
  while !first < n do
    let m = Int.min piece (n - !first) in
    let w = Plan.words Kernels.Widen in
    Plan.set w Kernels.copy_dst dst.at;
    Plan.set w Kernels.copy_src at;
    Plan.set w Kernels.copy_first !first;
    Plan.set w Kernels.copy_count (!first + m);
    Plan.set w Kernels.copy_rank rank;
    Plan.set w Kernels.copy_unit codes;
    Plan.set w Kernels.copy_units bit;
    if rank <= Kernels.copy_holds then axes w ~at:Kernels.copy_held l;
    let groups = (m + per - 1) / per in
    Plan.launch b Kernels.Widen ~groups:(groups, 1, 1)
      ~cost:(cost ~work ~m ~per ~groups ~width:1 ~unit:1 inner)
      w
      (Array.append
         [|
           { Plan.word = Kernels.copy_dst; slot = dst.slot; write = true };
           { Plan.word = Kernels.copy_src; slot; write = false };
         |]
         tail);
    first := !first + m
  done

(* Each output's load, if each is one's of a [Plain] load of a byte or more. *)
let sources p (xs : Operands.t) =
  let rec go k acc =
    if k < 0 then Some acc
    else
      match moved p k with
      | None -> None
      | Some j ->
          let o = xs.loads.(j) in
          let (D.Any dt) = o.dtype in
          if Option.is_some o.pad || D.bits dt < 8 then None
          else go (k - 1) (j :: acc)
  in
  go (Iarray.length p.outs - 1) []

(* A copy writes its destination's elements in order from its offset. *)
let contiguous (xs : Operands.t) =
  Array.for_all (fun (d : Operands.operand) -> L.is_contiguous d.layout) xs.dsts

(* Whether load [j], the operand of output [k], overlaps the destination of
   another output. *)
let staged xs k j =
  let other = ref false in
  for d = 0 to Array.length xs.Operands.dsts - 1 do
    if d <> k && Operands.overlaps xs ~load:j ~dst:d then other := true
  done;
  !other

(* Adds the launches that stage [src], an operand of [n] elements, into a
   workspace piece, and is the piece as a source: the run of positions its
   layout reaches, read by its own strides, where that run is shorter than [n],
   as for a broadcast operand; its elements in C order otherwise. *)
let stage b ~n src =
  let lo, hi = L.span src.layout in
  let ws = Plan.workspace b in
  if hi - lo < n then begin
    let run = L.contiguous [| hi - lo |] in
    let at = Plan.take b ((hi - lo) * src.width) in
    let dst = { src with slot = ws; at; layout = run; residue = 0 } in
    ignore
      (launch b ~n:(hi - lo) ~dst { src with at = lo * src.width; layout = run }
        : float);
    let at = at + ((L.offset src.layout - lo) * src.width) in
    { dst with at; layout = src.layout }
  end
  else begin
    let at = Plan.take b (n * src.width) in
    let dst =
      { src with slot = ws; at; layout = L.contiguous [| n |]; residue = 0 }
    in
    ignore (launch b ~n ~dst src : float);
    dst
  end

let map p (xs : Operands.t) =
  match sources p xs with
  | None -> Error A.Declined
  | Some _ when not (contiguous xs) -> Error A.Declined
  | Some js ->
      let n = L.numel xs.dsts.(0).layout in
      if n = 0 then Ok None
      else begin
        let b = Plan.make () in
        let srcs =
          List.mapi
            (fun k j ->
              let src = site b xs.loads.(j) in
              if staged xs k j then stage b ~n src else src)
            js
        in
        List.iteri
          (fun k src ->
            ignore (launch b ~n ~dst:(site b xs.dsts.(k)) src : float))
          srcs;
        (* A copy in one launch an output and staged operand is kept. *)
        Ok (Some (Plan.finish ~most:(2 * List.length js) b))
      end
