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
   whose innermost axis steps [s] elements: the element's own where it steps by
   at most one, and a line where neighbours share none. *)
let read ~width s = if s = 0 then width else Int.min Plan.line (abs s * width)

(* The cost of copying [m] elements in [groups] groups of [per] threads of
   [work] operations each from a source whose innermost axis has extent [e] and
   stride [s]: the destination's bytes and the source's, two lines at each run
   of the innermost axis and at the destination's ends, and each thread's work,
   the last group's idle ones included. *)
let cost ~work ~m ~per ~groups ~width (e, s) =
  let runs = ((m + e - 1) / e) + 1 in
  {
    Plan.bytes = (m * (width + read ~width s)) + (2 * Plan.line * (runs + 1));
    ops = per * groups * work;
    chain = 0;
  }

(* The most elements a launch of groups of [per] threads copies within
   {!Plan.allowed}, by [cost]'s and {!Plan.seconds}' bounds: a dispatch, traffic
   of [m (width + read + 2 line / e) + 6 line] bytes, and work of [(m + per - 1)
   w / (cores lanes op_rate) + ceil (per / lanes) w / op_rate] seconds at [w]
   operations a thread, [work], less one for rounding. *)
let most ~work ~per ~width (e, s) =
  let line = float Plan.line in
  let w = float work in
  let all = float (Plan.cores * Plan.lanes) *. Plan.op_rate in
  let bytes = float (width + read ~width s) +. (2. *. line /. float e) in
  let per_element = (bytes /. Plan.bandwidth) +. (w /. all) in
  let rounds = float ((per + Plan.lanes - 1) / Plan.lanes) in
  let fixed =
    Plan.dispatch
    +. (6. *. line /. Plan.bandwidth)
    +. (float (per - 1) *. w /. all)
    +. (rounds *. w /. Plan.op_rate)
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
  let piece = most ~work ~per ~width (pieces, 1) in
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
    let cost = cost ~work ~m ~per ~groups ~width (pieces, 1) in
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
  let piece = most ~work ~per ~width:src.width inner in
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
    let cost = cost ~work ~m ~per ~groups ~width:src.width inner in
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

(* [l]'s axes in bytes of [width]-byte elements, as copy_rows reads them: each
   but the last stepping its stride's bytes, the last, a row of consecutive
   elements, its extent's bytes at a stride of 1. *)
let row_axes s ~at ~width l =
  let r = L.rank l in
  for j = 0 to r - 1 do
    let e, stride =
      if j = r - 1 then (L.dim l j * width, 1)
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

(* The coalesced axes of [src]'s layout where copy_rows copies it: rows of
   consecutive elements of {!Kernels.copy_run_bytes} bytes or more beside other
   axes. *)
let rows src =
  let l =
    (L.coalesce [| L.contiguous (L.shape src.layout); src.layout |]).(1)
  in
  let r = L.rank l in
  if
    r >= 2
    && L.stride l (r - 1) = 1
    && L.dim l (r - 1) * src.width >= Kernels.copy_run_bytes
  then Some l
  else None

(* Adds the launches of copy_rows that copy the [n] elements of [src], of rows
   [l], from its position 0 into consecutive ones of [dst]: the blocks of
   {!Kernels.copy_run_bytes} bytes of [dst]'s memory that its bytes reach, as
   many a launch as keep it within the time budget, each block's work a copy_run
   thread's and an axis's for each axis before the row, by copy_rows where the
   bytes' indices fit 32 bits and copy_rows_wide past them; their estimated
   seconds are the result. *)
let launch_rows b ~n ~dst src l =
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
      row_axes s ~at:0 ~width:src.width l;
      let t = Bytes.unsafe_to_string s in
      let c =
        Plan.constant b ~key:("copy_rows " ^ t) ~length:(String.length t)
          (fun () -> bigbytes t)
      in
      [| { Plan.word = Kernels.copy_rows_axes; slot = c; write = false } |]
  in
  let work = work ~ops:element_ops ~n:bytes ~rank in
  let inner = (Int.max 1 (row / width), 1) in
  let piece = most ~work ~per ~width inner in
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
      if extents_fit && stop <= narrow then Kernels.Copy_rows
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
      row_axes w ~at:Kernels.copy_rows_held ~width:src.width l;
    let groups = (m + per - 1) / per in
    let cost = cost ~work ~m ~per ~groups ~width inner in
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
   consecutive, copy_rows' where they are rows of 16 bytes or more, and copy's
   otherwise. Their estimated seconds are the result. *)
let launch b ~n ~dst src =
  if L.is_contiguous src.layout then launch_run b ~n ~dst src
  else
    match rows src with
    | Some l -> launch_rows b ~n ~dst src l
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
  let piece = most ~work ~per ~width:1 inner in
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
      ~cost:(cost ~work ~m ~per ~groups ~width:1 inner)
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
