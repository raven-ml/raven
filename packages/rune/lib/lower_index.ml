(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module D = Nx_array.Dtype
module Arith = Lower_arith
module Bits = Lower_bits

let unlowered what =
  let why = Printf.sprintf "Rune.Lower.op: %s is not lowered" what in
  raise (Lower_arith.Unlowered why)

let extents u = List.init (Shape.ndim u) (Bits.extent u)
let dims s = List.map (fun n -> Ops.Int n) s
let numel s = List.fold_left ( * ) 1 s
let ceil_div a b = (a + b - 1) / b
let int u n = Shape.const_like u (`Int (Bigint.of_int n))

let broadcast c s =
  Shape.expand (Shape.reshape c (List.map (fun _ -> Ops.Int 1) s)) (dims s)

let zeros t s = broadcast (Ops.const ~dtype:t (`Int Bigint.zero)) s

(* [laid s axis v] is the vector [v] laid along [axis] of the shape [s]. *)
let laid s axis v =
  let r = List.length s in
  let along = List.init r (fun i -> if i = axis then List.nth s i else 1) in
  Shape.expand (Shape.reshape v (dims along)) (dims s)

(* [negative_zero u] is whether the float [u] is [-0]: its sign bit alone. *)
let negative_zero u =
  let b = Bits.bits u in
  let w = 8 * Dtype.itemsize (Ops.dtype u) in
  Ops.eq b (Shape.const_like b (`Int (Bigint.shift_left Bigint.one (w - 1))))

(* Gathers *)

let gather d ~axis idx x =
  Arith.of_parts d (List.map (fun u -> Bits.take u axis idx) (Arith.parts d x))

(* Scatters

   A scatter updates in place, in a kernel of its own, storage that holds
   [into]: its caller's [dst], such as a fresh buffer that a kernel stored
   [into] into, one element per thread, or a donated target's or a loop's
   carry's own storage, which needs no copy: a loop's trip then writes only
   the updates. [into] is [p·n·q] elements, an axis of [n] between [p] and
   [q], and the updates [p·k·q]. The axis is cut into runs of positions. For
   each index of the axes before the axis and each run, a thread walks the
   [k] updates in C order and combines each that lands in its run with the
   element it lands on: every element meets its updates in eager's order,
   and no two threads write one element. Work is [runs·k + n] at each index
   of the other axes, where a comparison of each element with each update
   would make [n·k]: 2.6e7 per feature for an embedding's gradient of 512
   updates over 50000 rows, against eager's 5e4. On the host, the axes after
   the axis are a serial loop inside the walk, so that an update writes a
   contiguous row, and positions an expansion repeats along them are read
   once a row. On a GPU they are threads too, adjacent ones at adjacent
   addresses. The kernel fuses with neither the one that computes [into] nor
   its readers: on an RTX 5000 Ada a scatter of a few updates costs one or
   two launches of about 1.8 µs more than a comparison fused with both.

   The host takes one run: at 1024 rows of 64 floats into 4096, one thread's
   walk takes 6 µs, and 16 runs split among its threads take 37 µs. A GPU
   takes as many runs as keep the renderer's [stream_threads] threads busy,
   and at most [k]: updates land in at most [k] runs, and a thread of any
   other run only compares. A run costs its thread a walk of every update, so
   runs add parallelism, never shorten the walk. A GPU's threads fill
   workgroups ({!lanes}), and one whose run is a single position carries its
   element in registers and stores it once.

   Where the targets are few, the walk is chunked. Its threads then take
   [chunks] runs of consecutive updates in C order, each combining its own
   into partial storage of [p·chunks·n·q] elements with a byte per element
   that marks where an update of the chunk landed, the partial being the
   chunk's first such update, then its combine with the next. A second
   kernel, a thread per element, combines the marked partials into the
   storage in chunk order: Set keeps the last chunk's, Max and Min take their
   extreme, Add adds each chunk's sum. So Set, Max and Min keep eager's bits,
   and Add's terms are summed in another association, within the same bound.
   Chunks shorten each thread's walk from [k] updates to [k / chunks], at the
   cost of the second kernel, its [chunks] steps and a fill of the marks,
   which a GPU's threads that carry their element write instead, so
   [chunks] is about [√k], and the partials are bounded to [stream_threads]
   elements. At 16 targets and 2{^20} updates on an RTX 5000 Ada, a
   comparison of every update with every target took 2.8 ms, the walk in one
   run a target 64.5 ms, and the chunked walk 63 µs.

   On a GPU, updates whose targets are distinct take no walk: a thread per
   update combines it into its element, and no two threads write one. The
   targets of a [Set] of rows, positions an expansion repeats along every
   axis after the axis as a model writes rows of its cache, are made
   distinct by an election. It compares each row's position with those of
   the later rows of its index of the axes before the axis, [p·k²]
   comparisons in one or two reduces, and replaces by [-1] each position a
   later row shares: each target keeps its last row in C order, so the
   result is eager's bit for bit, with no atomic store, which tolk lacks.
   It elects where the walk takes [k] steps a thread, in one chunk, from
   [elect_least] rows to [elect_work] comparisons: below, the walk is
   shorter than the election's launch, and above, the election's
   comparisons outgrow it. Positions the caller promises distinct
   ([unique]) take the threads with no election, under any combine. On an
   RTX 5000 Ada, 512 rows of 512 bfloat16 into 1024 took the walk 35.2 µs,
   and the election 3.13 and the threads 2.27. The host always walks: one
   thread's walk there does [k·q] work, the stores' own, and the election
   would add [k²].

   A narrow float's Add sums in a [float32] copy and marks where updates land
   in a byte per element, so that elsewhere the result keeps [into]'s bits,
   which a trip through [float32] quiets. *)

(* The walk's steps a launch costs: on fudge's Core Ultra 5 235 a launch takes
   about 1.1 µs and a walk 1.4 ns an update (2{^20} updates into 2 targets in
   1.44 ms); on an RTX 5000 Ada a launch about 1.8 µs and a chunk's walk
   46 ns an update (1024 into 16 targets in 47 µs). *)
let host_launch_steps = 790
let gpu_launch_steps = 39

let isqrt k =
  let r = ref (Float.to_int (Float.sqrt (Float.of_int k))) in
  while !r * !r > k do
    decr r
  done;
  while (!r + 1) * (!r + 1) <= k do
    incr r
  done;
  !r

(* The runs of positions and the chunks of updates a walk takes, for the
   renderer's [stream_threads] threads that keep the device busy: chunks
   where they shorten the walk by more than the two launches they add. *)
let plan ~host (ren : Renderer.t) ~p ~n ~q ~k =
  let threads = ren.stream_threads in
  let runs c =
    if host then 1
    else Int.min n (Int.min (ceil_div k c) (Int.max 1 (threads / (p * q * c))))
  in
  let c = Int.min (threads / (p * n * q)) (isqrt k) in
  let launch = if host then host_launch_steps else gpu_launch_steps in
  if c < 2 || k - (ceil_div k c + c) <= 2 * launch then (runs 1, 1)
  else (runs c, c)

(* [spread u] is the axes along which the view [u] broadcasts its source: those
   an expansion adds, moved by permutations. *)
let rec spread u =
  match Ops.op u with
  | Op.Expand -> (
      match Shape.marg u with
      | Expand front ->
          let m = List.length front in
          List.init m Fun.id @ List.map (( + ) m) (spread (Ops.nth u 0))
      | _ -> [])
  | Op.Permute -> (
      match Shape.marg u with
      | Permute perm ->
          let inner = spread (Ops.nth u 0) in
          let moved i a = if List.mem a inner then [ i ] else [] in
          List.concat (List.mapi moved perm)
      | _ -> [])
  | _ -> []

let range t e id = Ops.range ~axis_type:t (Ops.Int e) [ id ]

(* [workgroup ren total] is the threads of a workgroup of [total] threads on
   [ren]'s target: the greatest power of two that leaves the renderer's
   [local_workgroups], at most its [local_max]. *)
let workgroup (ren : Renderer.t) total =
  let most =
    Int.min (List.hd ren.local_max) (Int.max 1 (total / ren.local_workgroups))
  in
  let rec go w = if 2 * w <= most then go (2 * w) else w in
  go 1

(* [lanes ren ~id dims] numbers a thread for each index of [dims]: its index
   along each, the ranges that number the threads, from axis [id], and
   whether a thread is one of them, where some are not. Where [ren] launches
   workgroups, the threads in C order of [dims] fill workgroups
   ({!workgroup}), adjacent threads adjacent along the last of [dims], since
   the launch makes no workgroups of a kernel whose optimisations are fixed,
   and one thread a workgroup runs a warp's lane alone. Elsewhere each of
   [dims] past one element is a loop, which the host's threads split. *)
let lanes (ren : Renderer.t) ~id dims =
  let zero = Ops.O.int 0 in
  if not ren.has_local then
    let one i d =
      if d = 1 then (zero, [])
      else
        let r = range Weak d (id + i) in
        (r, [ r ])
    in
    let idx, rs = List.split (List.mapi one dims) in
    (idx, List.concat rs, None)
  else
    let total = numel dims in
    let size = workgroup ren total in
    let groups = ceil_div total size in
    let g = range Global groups id and l = range Local size (id + 1) in
    let t = Ops.O.((g * int size) + l) in
    (* Each index is [t / inner mod d], [inner] the threads of the dims after
       it: the first needs no [mod], below [total] the thread is one. *)
    let rec along first = function
      | [] -> []
      | d :: rest ->
          let inner = numel rest in
          let i =
            if d = 1 then zero
            else if first then Ops.O.(t // int inner)
            else Ops.O.(t // int inner % int d)
          in
          i :: along false rest
    in
    let live =
      if groups * size = total then None else Some Ops.O.(t < int total)
    in
    (along true dims, [ g; l ], live)

let both a b =
  match (a, b) with
  | None, c | c, None -> c
  | Some a, Some b -> Some (Ops.bitwise_and a b)

let gated ok o = match ok with None -> o | Some r -> Ops.where r o Ops.invalid

(* Slots of the kernels' registers, after their arguments'. *)
let slot_acc = 100
let slot_seen = 101
let ld buf i = Ops.load (Ops.index buf [ i ]) []
let st buf i v = Ops.store (Ops.index buf [ i ]) v
let zero8 = Ops.const ~dtype:Uint8 (`Int Bigint.zero)
let one8 = Ops.const ~dtype:Uint8 (`Int Bigint.one)
let marked v = Ops.ne v zero8

(* [carried ~dt ~w ~at ~init] is registers of [w] elements of [dt] and a
   byte, each thread's, set before the loop to [init] and [0]: an element a
   thread combines into and whether anything did. *)
let carried ~dt ~w ~at ~init =
  let acc = Shape.placeholder ~slot:slot_acc ~addrspace:Reg [ w ] dt in
  let seen = Shape.placeholder ~slot:slot_seen ~addrspace:Reg [ 1 ] Uint8 in
  let a0 = Ops.after acc at and s0 = Ops.after seen at in
  let inits =
    st s0 (Ops.O.int 0) zero8
    :: List.mapi (fun c v -> st a0 (Ops.O.int c) v) init
  in
  (acc, seen, Ops.group inits)

(* [walk ren ~host ~rows ~dt ~w ~p ~n ~q ~k ~runs ~chunks f args] is the
   kernel over [args] = [y; i; u] or [y; i; u; hit] that combines by [f] each
   update of [u], at its position in [i], into [y], [w] parts of [dt] an
   element, and sets [hit] to [1] where one lands. Where [rows], [i] holds one
   position per row of [q] updates. Where [chunks > 1], [y] is the partials
   and [hit] their marks: an update combines with the partial where it is
   marked, and replaces it otherwise. On a GPU, a thread whose run is one
   position carries its element in registers, stored once after its walk:
   a store to memory in the walk would order each later load after it, at a
   load's latency an update. *)
let walk ren ~host ~rows ~dt ~w ~p ~n ~q ~k ~runs ~chunks f args =
  let len = ceil_div n runs and span = ceil_div k chunks in
  let whole = chunks * span = k in
  let regs = (not host) && len = 1 in
  let outer = [ p; chunks; runs ] @ if host then [] else [ q ] in
  let idx, threads, live = lanes ren ~id:4 outer in
  let a, h, g = (List.nth idx 0, List.nth idx 1, List.nth idx 2) in
  let s = range Loop span 2 in
  let b = if host then range Loop q 3 else List.nth idx 3 in
  let y, i, u, hit =
    match args with
    | [ y; i; u ] -> (y, i, u, None)
    | [ y; i; u; h ] -> (y, i, u, Some h)
    | _ -> assert false
  in
  let open Ops.O in
  let j = (h * int span) + s in
  let row = (a * int k) + j in
  let update = (row * int q) + b in
  (* The last chunk may hold fewer updates than [span]. *)
  let ok = both live (if whole then None else Some (j < int k)) in
  let pos = ld i (gated ok (if rows then row else update)) in
  let u64 v = Ops.const ~dtype:Uint64 (`Int (Bigint.of_int v)) in
  let first = Ops.cast (g * int len) Int64 in
  (* Unsigned, a negative position lies past every bound, and [pos - first]
     wraps where it would overflow. *)
  let upos = Ops.bitcast pos Uint64 in
  let ufirst = Ops.cast (g * int len) Uint64 in
  let run = Ops.bitwise_and (upos < u64 n) (upos - ufirst < u64 len) in
  let run = match ok with None -> run | Some r -> Ops.bitwise_and r run in
  (* In the run, [pos] is [first + pos mod len], an address that names the
     run's range. An update outside the run reads and stores nothing: its index
     is invalid. *)
  let at =
    if regs then g
    else
      let within = Ops.bitcast (upos % u64 len) Int64 in
      Ops.cast (first + within) Weak_int
  in
  let e = (((((a * int chunks) + h) * int n) + at) * int q) + b in
  let part o c = (o * int w) + int c in
  let chunked = Stdlib.(chunks > 1) in
  let us =
    List.init w (fun c -> ld u (Ops.where run (part update c) Ops.invalid))
  in
  let kernel = Ops.kernel_info ~name:"scatter" ~opts_to_apply:[] () in
  if regs then begin
    (* The thread's element: [y]'s where it holds [into], else the partial
       its first update starts. *)
    let init =
      List.init w (fun c ->
          if chunked then Ops.cast (int 0) dt else ld y (gated live (part e c)))
    in
    let acc, seen, inits = carried ~dt ~w ~at:threads ~init in
    let acc_s = Ops.after acc [ inits; s ]
    and seen_s = Ops.after seen [ inits; s ] in
    let olds = List.init w (fun c -> ld acc_s (int c)) in
    let was = marked (ld seen_s (int 0)) in
    let combined = f olds us in
    let values =
      if chunked then List.map2 (Ops.where was) combined us else combined
    in
    let steps =
      st seen_s (int 0) (Ops.where run one8 (ld seen_s (int 0)))
      :: List.mapi
           (fun c v -> st acc_s (int c) (Ops.where run v (List.nth olds c)))
           values
    in
    let loop = Ops.end_ (Ops.group steps) [ s ] in
    let acc = Ops.after acc [ loop ] and seen = Ops.after seen [ loop ] in
    let landed = marked (ld seen (int 0)) in
    let landed =
      match live with None -> landed | Some l -> Ops.bitwise_and l landed
    in
    let into o = Ops.where landed o Ops.invalid in
    let stores =
      List.init w (fun c ->
          Ops.store (Ops.index y [ into (part e c) ]) (ld acc (int c)))
    in
    (* The thread owns its element's mark, unfilled before: it writes it. *)
    let marks =
      match hit with
      | None -> []
      | Some m -> [ Ops.store (Ops.index m [ gated live e ]) (ld seen (int 0)) ]
    in
    Ops.sink ~kernel [ Ops.end_ (Ops.group (stores @ marks)) threads ]
  end
  else begin
    let at_part m o c = Ops.index m [ Ops.where run (part o c) Ops.invalid ] in
    let at_mark m o = Ops.index m [ Ops.where run o Ops.invalid ] in
    let order m = Ops.after m (if host then [ s; b ] else [ s ]) in
    let state = order y in
    let olds = List.init w (fun c -> Ops.load (at_part state e c) []) in
    let values =
      match hit with
      | Some m when chunked ->
          let seen = marked (Ops.load (at_mark (order m) e) []) in
          List.map2 (Ops.where seen) (f olds us) us
      | _ -> f olds us
    in
    let stores = List.mapi (fun c v -> Ops.store (at_part y e c) v) values in
    let marks =
      match hit with None -> [] | Some m -> [ Ops.store (at_mark m e) one8 ]
    in
    let body = Ops.group (stores @ marks) in
    let ends =
      if host then Ops.end_ (Ops.end_ (Ops.end_ body [ b ]) [ s ]) threads
      else Ops.end_ (Ops.end_ body [ s ]) threads
    in
    Ops.sink ~kernel [ ends ]
  end

(* [merge ren ~host ~dt ~w ~p ~m ~chunks f args] is the kernel over [args] =
   [y; parts; marks] or [y; parts; marks; hit] that combines by [f] into each
   of the [p·m] elements of [y], [w] parts of [dt] an element, its [chunks]
   partials where marked, in chunk order, and sets [hit] to [1] where one is.
   On a GPU the element is carried in registers, as {!walk} does. *)
let merge ren ~host ~dt ~w ~p ~m ~chunks f args =
  let idx, threads, live = lanes ren ~id:0 [ p; m ] in
  let a, t = (List.nth idx 0, List.nth idx 1) in
  let c = range Loop chunks 2 in
  let y, parts, marks, hit =
    match args with
    | [ y; v; k ] -> (y, v, k, None)
    | [ y; v; k; h ] -> (y, v, k, Some h)
    | _ -> assert false
  in
  let open Ops.O in
  let e = (a * int m) + t in
  let pe = (((a * int chunks) + c) * int m) + t in
  let seen = marked (ld marks (gated live pe)) in
  let seen =
    match live with None -> seen | Some l -> Ops.bitwise_and l seen
  in
  let part o i = (o * int w) + int i in
  let ps = List.init w (fun i -> ld parts (gated live (part pe i))) in
  let kernel = Ops.kernel_info ~name:"scatter_merge" ~opts_to_apply:[] () in
  if not host then begin
    let init = List.init w (fun i -> ld y (gated live (part e i))) in
    let acc, any, inits = carried ~dt ~w ~at:threads ~init in
    let acc_c = Ops.after acc [ inits; c ]
    and any_c = Ops.after any [ inits; c ] in
    let olds = List.init w (fun i -> ld acc_c (int i)) in
    let steps =
      st any_c (int 0) (Ops.where seen one8 (ld any_c (int 0)))
      :: List.mapi
           (fun i v -> st acc_c (int i) (Ops.where seen v (List.nth olds i)))
           (f olds ps)
    in
    let loop = Ops.end_ (Ops.group steps) [ c ] in
    let acc = Ops.after acc [ loop ] and any = Ops.after any [ loop ] in
    let landed = marked (ld any (int 0)) in
    let landed =
      match live with None -> landed | Some l -> Ops.bitwise_and l landed
    in
    let stores =
      List.init w (fun i ->
          Ops.store
            (Ops.index y [ Ops.where landed (part e i) Ops.invalid ])
            (ld acc (int i)))
    in
    let marks =
      match hit with
      | None -> []
      | Some h -> [ Ops.store (Ops.index h [ gated live e ]) (ld any (int 0)) ]
    in
    Ops.sink ~kernel [ Ops.end_ (Ops.group (stores @ marks)) threads ]
  end
  else begin
    let state = Ops.after y [ c ] in
    let olds = List.init w (fun i -> ld state (gated live (part e i))) in
    let gate o i = Ops.index y [ Ops.where seen (part o i) Ops.invalid ] in
    let stores = List.mapi (fun i v -> Ops.store (gate e i) v) (f olds ps) in
    let marks =
      match hit with
      | None -> []
      | Some h ->
          [ Ops.store (Ops.index h [ Ops.where seen e Ops.invalid ]) one8 ]
    in
    let ends = Ops.end_ (Ops.end_ (Ops.group (stores @ marks)) [ c ]) threads in
    Ops.sink ~kernel [ ends ]
  end

(* [push ren ~rows ~w ~p ~n ~q ~k f args] is the kernel over [args] =
   [y; i; u] with a thread per update: each combines by [f] its update, [w]
   parts an element, into the element of [y] it lands on. Where [rows], [i]
   holds one position per row of [q] updates. No two updates that land may
   share an element: a thread neither walks nor orders its store. *)
let push ren ~rows ~w ~p ~n ~q ~k f args =
  let idx, threads, live = lanes ren ~id:0 [ p; k; q ] in
  let a, j, b = (List.nth idx 0, List.nth idx 1, List.nth idx 2) in
  let y, i, u = match args with [ y; i; u ] -> (y, i, u) | _ -> assert false in
  let open Ops.O in
  let row = (a * int k) + j in
  let update = (row * int q) + b in
  let pos = ld i (gated live (if rows then row else update)) in
  let u64 v = Ops.const ~dtype:Uint64 (`Int (Bigint.of_int v)) in
  (* Unsigned, a negative position lies past every bound. *)
  let inside = Ops.bitcast pos Uint64 < u64 n in
  let ok =
    match live with None -> inside | Some l -> Ops.bitwise_and l inside
  in
  let e = (((a * int n) + Ops.cast pos Weak_int) * int q) + b in
  let at o c = Ops.where ok ((o * int w) + int c) Ops.invalid in
  let us = List.init w (fun c -> ld u (at update c)) in
  let olds = List.init w (fun c -> ld y (at e c)) in
  let stores = List.mapi (fun c v -> st y (at e c) v) (f olds us) in
  let kernel = Ops.kernel_info ~name:"scatter_push" ~opts_to_apply:[] () in
  Ops.sink ~kernel [ Ops.end_ (Ops.group stores) threads ]

(* [elected ~p ~k ~c i] is the positions [i] of [p·k] rows, [k] for each
   index of the axes before the axis, with [-1], outside every axis, at each
   row that a later row of its index targets at the same position: each
   target keeps its last row in C order. A row compares its position with
   the [k] of its index in [c] parts of [ceil (k / c)], each part's result
   stored, then those [c] reduced: a thread's comparisons are then about
   [k / c + c] whatever threads tolk gives a reduce. The last part pads with
   [-1], which only a position outside the axis equals. *)
let elected ~p ~k ~c i =
  let s = ceil_div k c in
  let pos = Shape.reshape i (dims [ p; k ]) in
  let cube = [ p; k; c; s ] in
  let mine =
    Shape.expand (Shape.reshape pos (dims [ p; k; 1; 1 ])) (dims cube)
  in
  let padded =
    let m1 = `Int (Bigint.of_int (-1)) in
    Shape.pad ~value:m1 pos [ None; Some (Ops.Int 0, Ops.Int ((c * s) - k)) ]
  in
  let theirs =
    Shape.expand (Shape.reshape padded (dims [ p; 1; c; s ])) (dims cube)
  in
  let j = laid cube 1 (Shape.arange ~dtype:Int64 k) in
  let later =
    Shape.reshape (Shape.arange ~dtype:Int64 (c * s)) (dims [ 1; 1; c; s ])
  in
  let later = Shape.expand later (dims cube) in
  let clash = Ops.bitwise_and (Ops.eq mine theirs) (Ops.lt j later) in
  let some = Shape.rop (Ops.cast clash Uint8) Op.Max [ 3 ] in
  let some =
    if c = 1 then some else Shape.rop (Ops.contiguous some) Op.Max [ 2 ]
  in
  let some = Shape.reshape some (dims [ p; k ]) in
  Shape.reshape (Ops.where (marked some) (int pos (-1)) pos) (dims [ p * k ])

(* [fresh device v] is a buffer of [device] once [v] is stored into it. *)
let fresh device v =
  let b = Ops.new_buffer device (Shape.max_numel v) (Ops.dtype v) in
  Ops.after b [ Ops.store b v ]

type dst = Filled of (Ops.t -> Ops.t) | Holding of Ops.t

(* The election's bounds, on an RTX 5000 Ada, for [k] rows of 512 bfloat16
   into twice as many positions, [p = 1]; the walk against the election
   and its stores:
   - at 16 rows, 2.02 µs against 1.02 and 1.15: about even alone, a gain
     where the keys' and values' scatters of a model's layer share their
     election; at 32 rows, 3.04 against 1.06 and 1.22;
   - at 512, 35.2 against 3.13 and 2.27; at 8192, 540 against 77.6 and
     21.4; at 32768, 2162 against 590 and 122, [2{^30}] comparisons;
   - past [elect_work] comparisons, unmeasured: they grow as [k²], the
     walk's steps as [k], and the gain fell from 5.5 times at 8192 rows to
     3 at 32768. *)
let elect_least = 16
let elect_work = 1 lsl 30

(* [election ren ~p ~n ~q ~k] is [Some c] where a GPU's [Set] of [k] rows at
   each of [p] indices elects, its comparisons cut in [c] parts. It elects
   where the walk takes one chunk, [k] steps a thread: a chunked walk takes
   about [2 √k], as the parts do, and needs no election. The parts are
   about [√k], and taken where they shorten a thread's [k] comparisons,
   each about a walk's step, by more than their second kernel's launch:
   tolk may spread one reduce's comparisons over threads, but the election
   does not count on it. Parts cost that launch where it does: at 512 rows,
   3.13 µs against one reduce's 1.98; at 2048, 11.45 against 7.26; at 4096,
   where one thread took each row, 29.7 against 203. The marks between the
   parts' kernels are [p·k·c] bytes, at most [elect_work / 7] where
   [c > 1]. *)
let election ren ~p ~n ~q ~k =
  let _, chunks = plan ~host:false ren ~p ~n ~q ~k in
  if k < elect_least || chunks > 1 || p * k * k > elect_work then None
  else
    let c = isqrt k in
    if k - (2 * c) > gpu_launch_steps then Some c else Some 1

let looped ~dst ~host ~unique (ren : Renderer.t) device (D.Any dt as d)
    (c : Nx_kernel.combine) ~axis ~idx ~updates into =
  let s = extents (List.hd (Arith.parts d into)) and t = extents idx in
  let r = List.length s in
  let p = numel (List.filteri (fun i _ -> i < axis) s) in
  let q = numel (List.filteri (fun i _ -> i > axis) s) in
  let n = List.nth s axis and k = List.nth t axis in
  if c = Add && D.is D.Boolean dt then unlowered "a boolean Add Scatter";
  let w = if D.is D.Complex dt then 2 else 1 in
  let narrow = c = Add && D.is D.Float dt && D.bits dt < 32 in
  let ca = if narrow then D.Any D.Float32 else d in
  let on_device v =
    if Option.is_none (Ops.device v) then Ops.copy_to_device v device else v
  in
  (* [v], a node of [d], as the flat parts of [ca]. *)
  let flat v =
    let v = if narrow then Arith.cast ~src:d ca v else v in
    Shape.reshape v [ Ops.Int (Shape.max_numel v) ]
  in
  let fresh = fresh device in
  (* A kernel's argument starts on 16 bytes, as its parameter assumes: a view
     at another phase, such as an argument's, is copied. *)
  let arg v =
    let v = Ops.contiguous (on_device v) in
    let align, phase = Shape.storage_phase v in
    if align >= 16 && phase mod 16 = 0 then v else fresh v
  in
  let combined olds us =
    match c with
    | Set -> us
    | Add -> Arith.binary Add ca olds us
    | Max -> Arith.binary Maximum ca olds us
    | Min -> Arith.binary Minimum ca olds us
  in
  (* Positions an expansion repeats along every axis after [axis] are read once
     a row, from the view's first element along those axes. *)
  let rows =
    let spread = spread idx in
    List.for_all
      (fun i -> i <= axis || List.nth t i = 1 || List.mem i spread)
      (List.init r Fun.id)
  in
  let i =
    if not rows then Shape.reshape idx [ Ops.Int (numel t) ]
    else
      let first i = if i > axis then Some (Ops.Int 0, Ops.Int 1) else None in
      Shape.reshape (Shape.shrink idx (List.init r first)) [ Ops.Int (p * k) ]
  in
  (* The store that fills [y] computes [into]: a constant [into] is a fill.
     Storage that holds [into] needs none. *)
  let y =
    match dst with
    | _ when narrow -> fresh (flat into)
    | Filled f -> f (flat into)
    | Holding s -> s
  in
  (* On a GPU, distinct targets, promised or elected, take a thread per
     update. *)
  let distinct =
    if host || narrow then None
    else if unique then Some (arg i)
    else if c = Set && rows then
      let elect parts = arg (elected ~p ~k ~c:parts (arg i)) in
      Option.map elect (election ren ~p ~n ~q ~k)
    else None
  in
  match distinct with
  | Some i ->
      let kernel = push ren ~rows ~w ~p ~n ~q ~k combined in
      let u = arg (flat updates) in
      let y = List.hd (Shape.custom_kernel [ y; i; u ] kernel) in
      Shape.reshape y (Arith.dims ca (Array.of_list s))
  | None ->
      let runs, chunks = plan ~host ren ~p ~n ~q ~k in
      (* A GPU's thread that carries its element in registers writes its
         mark whether or not an update landed, as does a GPU's merge: their
         marks need no fill. *)
      let owns = (not host) && ceil_div n runs = 1 in
      let marks_of m =
        if owns then Ops.new_buffer device m Uint8
        else fresh (zeros Uint8 [ m ])
      in
      let hit =
        if not narrow then []
        else if (not host) && chunks > 1 then
          [ Ops.new_buffer device (numel s) Uint8 ]
        else [ marks_of (numel s) ]
      in
      let flat_updates = flat updates in
      let dt = Ops.dtype flat_updates in
      let walk =
        walk ren ~host ~rows ~dt ~w ~p ~n ~q ~k ~runs ~chunks combined
      in
      let i = arg i and u = arg flat_updates in
      let outs =
        if chunks = 1 then Shape.custom_kernel ([ y; i; u ] @ hit) walk
        else
          let m = p * n * q in
          let parts =
            Ops.new_buffer device (w * m * chunks) (Ops.dtype flat_updates)
          in
          let marks = marks_of (m * chunks) in
          let parts, marks =
            match Shape.custom_kernel [ parts; i; u; marks ] walk with
            | [ parts; _; _; marks ] -> (parts, marks)
            | _ -> assert false
          in
          Shape.custom_kernel
            ([ y; parts; marks ] @ hit)
            (merge ren ~host ~dt ~w ~p ~m:(n * q) ~chunks combined)
      in
      let y =
        Shape.reshape (List.hd outs) (Arith.dims ca (Array.of_list s))
      in
      if not narrow then y
      else
        let hit = Shape.reshape (List.nth outs 3) (dims s) in
        Ops.where (Ops.ne hit (int hit 0)) (Arith.cast ~src:ca d y) into

let scatter ?dst ~host ~unique ren device d c ~axis ~idx ~updates into =
  let s = extents (List.hd (Arith.parts d into)) and t = extents idx in
  if List.nth s axis = 0 || numel t = 0 then into
  else
    let dst = Option.value dst ~default:(Filled (fresh device)) in
    looped ~dst ~host ~unique ren device d c ~axis ~idx ~updates into

(* Folds

   A fold undoes a padded load's movements in reverse, each by its adjoint: a
   window's sums the windows' elements back into place, the ends' pads what
   they cropped and crops what they padded, and the interior padding's keeps
   every [k + 1]th element. Each part sums in the carrier, an integer on an
   unsigned pattern of 32 or 64 bits as {!Lower_reduce} folds integers, and
   rounds once. The same movements over a mark per term, [2] for a term other
   than [-0] and [1] for [-0], folded by their maximum, find where every term
   is [-0], which the sum, from tolk's [+0] accumulator, makes [+0]. *)

(* [only r i b] is [b] at axis [i] of [r] axes, [None] elsewhere. *)
let only r i b = List.init r (fun j -> if j = i then Some b else None)

(* [rows u i k] is [u] with its axis [i] split into rows of [k]. *)
let rows u i k =
  let e = extents u in
  let split = List.mapi (fun j d -> if j = i then [ d / k; k ] else [ d ]) e in
  Shape.reshape u (dims (List.concat split))

let ints a b = (Ops.Int a, Ops.Int b)

(* [unwindow op u w s] is the adjoint of the window [w] over a value of shape
   [s]: [u]'s elements, of the windowed shape, folded by [op] into the place
   each reads. The axis is laid out as rows one element longer than
   [w.dilation] apart from a whole number of copies of it, so that the
   window's element [j] at window [l] lands [j·dilation + l·step] into a copy:
   the copies' sum is the fold. *)
let unwindow op u (w : Nx_array.Move.window) s =
  let a = w.axis and r = List.length s in
  let n = List.nth s a and l = Bits.extent u a in
  if n = 0 || l = 0 then zeros (Ops.dtype u) s
  else
    let others = List.filter (( <> ) a) (List.init r Fun.id) in
    let o = List.map (List.nth s) others in
    let scale = Int.max 1 (ceil_div ((l * w.step) - w.dilation) n) in
    let row = (n * scale) + w.dilation in
    let copies = ceil_div (w.size * row) n in
    let reshape x e = Shape.reshape x (dims (o @ e)) in
    let pad_to x e = Shape.pad_to x (List.map Option.some (dims (o @ e))) in
    let x = Shape.permute u (others @ [ r; a ]) in
    let x = reshape x [ w.size; l; 1 ] in
    let x = pad_to x [ w.size; l; w.step ] in
    let x = reshape x [ w.size; l * w.step ] in
    let x = pad_to x [ w.size; row ] in
    let x = reshape x [ w.size * row ] in
    let x = pad_to x [ copies * n ] in
    let x = Shape.rop (reshape x [ copies; n ]) op [ r - 1 ] in
    let back i = if i = a then r - 1 else if i < a then i else i - 1 in
    Shape.permute x (List.init r back)

(* [unends lo hi s u] is the adjoint of padding a value of shape [s] by [lo]
   and [hi]: [u] padded with zeros where the load cropped, then cropped where
   it padded. *)
let unends lo hi s u =
  let at f = List.init (List.length s) (fun i -> Some (f i)) in
  let cropped i a = Int.max 0 (-Iarray.get a i) in
  let u = Shape.pad u (at (fun i -> ints (cropped i lo) (cropped i hi))) in
  Shape.shrink u
    (at (fun i ->
         let b = Int.max 0 (Iarray.get lo i) in
         ints b (b + List.nth s i)))

(* [uninterior k s u] is the adjoint of spacing a value of shape [s] by [k]:
   every [k.(i) + 1]th element of [u] along each axis [i]. *)
let uninterior k s u =
  let r = List.length s in
  let every u i k =
    if k = 0 || List.nth s i = 0 then u
    else
      let u = rows (Shape.pad u (only r i (ints 0 k))) i (k + 1) in
      let u = Shape.shrink u (only (r + 1) (i + 1) (ints 0 1)) in
      Shape.flatten ~start:i ~stop:(i + 1) u
  in
  let u = ref u in
  Iarray.iteri (fun i k -> u := every !u i k) k;
  !u

let fold (D.Any dt as d) (pad : Nx_kernel.pad) shape x =
  if D.is D.Boolean dt then unlowered "a boolean Fold";
  let s0 = Iarray.to_list shape in
  let spaced i n =
    if n = 0 then 0 else n + ((n - 1) * Iarray.get pad.interior i)
  in
  let s1 = List.mapi spaced s0 in
  let s2 =
    List.mapi (fun i n -> n + Iarray.get pad.lo i + Iarray.get pad.hi i) s1
  in
  (* Each window with the shape it windows, the last first. *)
  let windows, _ =
    Iarray.fold_left
      (fun (ws, s) (w : Nx_array.Move.window) ->
        let m = Nx_array.Move.Window (Iarray.of_list [ w ]) in
        let s' = Iarray.to_list (Nx_array.Move.shape m (Iarray.of_list s)) in
        ((w, s) :: ws, s'))
      ([], s2) pad.windows
  in
  let back op u =
    let u = List.fold_left (fun u (w, s) -> unwindow op u w s) u windows in
    uninterior pad.interior s0 (unends pad.lo pad.hi s1 u)
  in
  let ca = Nx_kernel.carrier d in
  let part u =
    let t = Ops.dtype u in
    if not (Dtype.is_float t) then
      let wide : Dtype.t =
        if Dtype.itemsize t < 4 then Uint32 else Bits.unsigned t
      in
      let y = back Op.Add (Ops.cast u wide) in
      (* A cast, not a bitcast, back to a signed type: tolk renders a
         bitcast of a constant of 64 bits from a 32-bit literal. *)
      Ops.cast (Ops.cast y (Bits.unsigned t)) t
    else
      let y = back Op.Add u in
      let mark =
        Ops.where (negative_zero u) (int (Bits.bits u) 1) (int (Bits.bits u) 2)
      in
      let mark = back Op.Max (Ops.cast mark Uint8) in
      Ops.where (Ops.eq mark (int mark 1)) (Shape.const_like y (`Float (-0.))) y
  in
  let x = Arith.cast ~src:d ca x in
  Arith.cast ~src:ca d (Arith.of_parts ca (List.map part (Arith.parts ca x)))

(* Assemblies

   A piece is placed into the result's shape by the adjoint of its region's
   slice, which spaces its elements [step] apart and pads around them, and so
   is a mask of where it lies. Each piece in turn selects its elements where
   it lies over what came before. *)

(* [placed s ranges u] is [u] at the positions [ranges] keep of a shape [s],
   zeros elsewhere. *)
let placed s (ranges : Nx_array.Move.range iarray) u =
  let r = List.length s in
  let place u i ({ start; count; step } : Nx_array.Move.range) =
    let u, first, step =
      if step < 0 then (Shape.flip u [ i ], start + ((count - 1) * step), -step)
      else (u, start, step)
    in
    let span = ((count - 1) * step) + 1 in
    let u =
      if step = 1 then u
      else
        let gaps = only (r + 1) (i + 1) (ints 0 (step - 1)) in
        let u = Shape.pad (rows u i 1) gaps in
        let u = Shape.flatten ~start:i ~stop:(i + 1) u in
        Shape.shrink u (only r i (ints 0 span))
    in
    Shape.pad u (only r i (ints first (List.nth s i - first - span)))
  in
  let u = ref u in
  Iarray.iteri (fun i range -> u := place !u i range) ranges;
  !u

(* [words dt x] is the bits of each part of the element [x] of [dt], as
   unsigned integers in the host's order. *)
let words (type v s) (dt : (v, s) D.t) (x : v) =
  let b = Nx_kernel.Prog.bits dt x in
  let word b =
    let n = ref Bigint.zero in
    for i = String.length b - 1 downto 0 do
      n := Bigint.add (Bigint.shift_left !n 8) (Bigint.of_int (Char.code b.[i]))
    done;
    `Int !n
  in
  if not (D.is D.Complex dt) then [ word b ]
  else
    let h = String.length b / 2 in
    [ word (String.sub b 0 h); word (String.sub b h h) ]

(* The elements a region keeps. *)
let size ranges =
  Iarray.fold_left (fun m (r : Nx_array.Move.range) -> m * r.count) 1 ranges

(* A float assembles as its bits. Where a float constant fill meets a padded
   piece in an Ops.where, tolk quiets a signalling NaN constant before it
   renders the kernel: a float64 fill of bits 0x7ff0000000000001 beside a
   piece reaches the kernel's source as 0x7ff8000000000001. A [-0] fill stays
   [-0], so no addition is involved, and a fill alone keeps its bits. *)
let assemble (type v s) (dt : (v, s) D.t) (fill : v) shape pieces =
  let d = D.Any dt in
  let s = Iarray.to_list shape in
  let pieces = List.filter (fun (ranges, _) -> size ranges > 0) pieces in
  let tdt = Arith.dtype d in
  let floats = Dtype.is_float tdt in
  let start =
    if floats then
      let tb = Bits.unsigned tdt in
      List.map (fun w -> broadcast (Ops.const ~dtype:tb w) s) (words dt fill)
    else
      let const c = broadcast (Ops.const ~dtype:tdt c) s in
      List.map const (Arith.fill dt fill)
  in
  let put acc (ranges, u) =
    let parts = Arith.parts d u in
    let parts = if floats then List.map Bits.bits parts else parts in
    let ones = Shape.const_like ~dtype:Bool (List.hd parts) (`Bool true) in
    let mask = placed s ranges ones in
    List.map2 (fun a p -> Ops.where mask (placed s ranges p) a) acc parts
  in
  let y = Arith.of_parts d (List.fold_left put start pieces) in
  if floats then Ops.bitcast y tdt else y

(* Sorts

   A sort orders keys, unsigned integers that order as the elements do, by a
   bitonic network of maxima and minima, which sorts integers exactly but is
   not stable. Each key runs packed above its position: packed integers are
   distinct, so the order the network gives them is the stable one, and the
   positions are their low bits. A key of more than 32 bits sorts in passes of
   32 bits, its least significant first: each pass sorts the next bits in the
   previous pass's order, and being stable keeps that order among equal bits.
   The values are the elements gathered at the positions, bits unchanged. *)

let bit_length n =
  let rec go n b = if n = 0 then b else go (n lsr 1) (b + 1) in
  go n 0

let halves u axis =
  match Shape.split ~axis u [ 1; 1 ] with
  | [ a; b ] -> (a, b)
  | _ -> assert false

(* [bitonic x axis] is the integers [x] sorted increasingly along [axis]. The
   axis, padded to a power of two with the greatest integer, is split into axes
   of two elements, one per stage of the network. *)
let bitonic x axis =
  let n = Bits.extent x axis in
  if n <= 1 then x
  else
    let stages = bit_length (n - 1) and shape = Shape.shape x in
    let x =
      Shape.pad_to
        ~value:(Dtype.max (Ops.dtype x) :> Dtype.const)
        x
        (List.init (Shape.ndim x) (fun d ->
             if d = axis then Some (Ops.Int (1 lsl stages)) else None))
    in
    let x = Shape.unflatten x axis (List.init stages (fun _ -> Ops.Int 2)) in
    let r = Shape.ndim x in
    (* A stage's crossover reverses the second half of each of its boxes, so
       that comparisons all in one direction merge them; it is undone after. *)
    let crossover stage x =
      let c = axis + stages - stage - 1 in
      let blue, green = halves x c in
      let flipped =
        List.init (stage + List.length shape - axis) (fun i -> r - 1 - i)
      in
      Shape.cat ~axis:c blue [ Shape.flip green flipped ]
    in
    let compare x sub =
      let p = axis + stages - sub - 1 in
      let top, bottom = halves x p in
      Ops.contiguous
        (Shape.cat ~axis:p (Ops.minimum top bottom) [ Ops.maximum top bottom ])
    in
    let stage x s =
      let x = if s < stages then Ops.contiguous (crossover s x) else x in
      let x = List.fold_left compare x (List.init s (fun i -> s - 1 - i)) in
      if s < stages then crossover s x else x
    in
    let x = List.fold_left stage x (List.init stages (fun i -> i + 1)) in
    Shape.shrink_to
      (Shape.flatten ~start:axis ~stop:(axis + stages - 1) x)
      (List.map Option.some shape)

(* [packed axis h] is the int64 keys [h], below [2^32], each above its position
   along [axis]: distinct integers that order as a stable sort of the keys. *)
let packed axis h =
  let s = extents h in
  let low = bit_length (List.nth s axis - 1) in
  let ranks = laid s axis (Shape.arange ~dtype:Int64 (List.nth s axis)) in
  Ops.bitwise_or (Ops.shl h (int h low)) ranks

(* [ranked axis h] is the int64 positions that sort the int64 keys [h], below
   [2^32], stably along [axis]. *)
let ranked axis h =
  let low = bit_length (List.nth (extents h) axis - 1) in
  let sorted = bitonic (Ops.contiguous (packed axis h)) axis in
  Ops.bitwise_and sorted (int h ((1 lsl low) - 1))

(* A sort's first [k] of a few keys need not sort: the position of the [r]th
   is the one exactly [r] keys precede. Comparing every pair of keys is one
   kernel of [k·n²] comparisons, in place of a network's kernels, one per stage
   and each a launch. [selection_comparisons] bounds the comparisons of the
   whole array: past it the comparisons outweigh the launches they save, and
   [rounds] or the network take the first [k]. It divides so that a shape's
   product cannot overflow. *)

let selection_comparisons = 1 lsl 16

let selecting ~k s axis =
  let n = List.nth s axis in
  k <= selection_comparisons / n / n / (numel s / n)

(* [selected axis k h] is the int64 positions of the [k] leading keys of the
   sort of [h], in sort order, as [ranked] would give them. *)
let selected axis k h =
  let s = extents h in
  let before = List.take axis s and after = List.drop (axis + 1) s in
  let n = List.nth s axis in
  let keys = Ops.contiguous (packed axis h) in
  (* Axes [axis], [axis+1], [axis+2] are [r], [i], [j]: the output, the key
     ranked, and the key compared with it. *)
  let space = before @ [ k; n; n ] @ after in
  let along i j =
    let extent on = if on then n else 1 in
    let laid = before @ [ 1; extent i; extent j ] @ after in
    Shape.expand (Shape.reshape keys (dims laid)) (dims space)
  in
  let precedes =
    Ops.cast (Ops.lt (along false true) (along true false)) Int64
  in
  let rank = Shape.rop precedes Op.Add [ axis + 2 ] in
  let line = before @ [ k; n ] @ after in
  let at c e = laid line c (Shape.arange ~dtype:Int64 e) in
  let picked =
    Ops.where (Ops.eq rank (at axis k)) (at (axis + 1) n) (zeros Int64 line)
  in
  Shape.rop picked Op.Add [ axis + 1 ]

(* [key u] is the unsigned integer of [u]'s width that orders as its elements:
   a signed integer's or a float's order key with its sign bit flipped, a
   float's order key being the signed integer of its bits with a negative
   float's magnitude bits flipped, so that [-0] lies just below [+0]. A boolean
   is its truth. *)
let most k = Shape.const_like k (Dtype.max (Ops.dtype k) :> Dtype.const)
let least k = Shape.const_like k (Dtype.min (Ops.dtype k) :> Dtype.const)

let key u =
  let t = Ops.dtype u in
  let biased k = Ops.bitcast (Ops.bitwise_xor k (least k)) (Bits.unsigned t) in
  if Dtype.is_float t then
    let k = Ops.bitcast u (Bits.signed t) in
    biased (Ops.where (Ops.lt k (int k 0)) (Ops.bitwise_xor k (most k)) k)
  else if Dtype.is_unsigned t then u
  else biased u

(* [keys d ~descending u] is the keys of the node [u] of a value of [d], their
   most significant 32 bits first, each an int64 below [2^32]: complemented
   where [descending], every NaN, and every complex number with a NaN part,
   the greatest, and a complex number its real part's then its imaginary
   part's. *)
let keys (D.Any dt as d) ~descending u =
  let parts =
    if D.is D.Boolean dt then [ Ops.cast (Ops.ne u (int u 0)) Uint8 ]
    else if D.is D.Float dt && D.bits dt < 16 then
      (* A float of a code orders as its float32 value. *)
      [ Arith.cast ~src:d (D.Any D.Float32) u ]
    else Arith.parts d u
  in
  let ks = List.map key parts in
  let ks =
    if not (Dtype.is_float (Ops.dtype (List.hd parts))) then ks
    else
      let nan =
        List.fold_left
          (fun a p -> Ops.bitwise_or a (Bits.is_nan p))
          (Bits.is_nan (List.hd parts))
          (List.tl parts)
      in
      List.map (fun k -> Ops.where nan (most k) k) ks
  in
  let ks = if descending then List.map Ops.bitwise_not ks else ks in
  let split k =
    if Dtype.itemsize (Ops.dtype k) < 8 then [ Ops.cast k Int64 ]
    else
      [
        Ops.cast (Ops.shr k (int k 32)) Int64;
        Ops.cast (Ops.bitwise_and k (int k 0xffff_ffff)) Int64;
      ]
  in
  List.concat_map split ks

(* Past the selection's bound a first [k] still need not sort: the [r]th of the
   packed keys, which are distinct, is the least above the one before it, a
   reduction over the slice. [k] rounds are [k·n] comparisons and [k] kernels,
   where the network makes a pass of [n] and a kernel at each of its [s(s+1)/2]
   compare steps, [s] the least with [n <= 2^s]: rounds take a first [k]
   wherever [k] is at most those steps. The count weighs a round as a step,
   which holds for many rows; on a GPU a round over few rows, a reduction with
   few outputs, is slower than a step: over one row of 4096 rounds take 1.5
   times the network's kernel time at [k = 16] and 3.2 times at [k = 64]. tolk
   reduces by [Max], so a round takes the greatest of the complements [m - p]
   below the last round's, [m] the greatest [int64]. *)

let rounding ~k n =
  let s = bit_length (n - 1) in
  k <= s * (s + 1) / 2

(* [rounds axis k h] is the int64 positions of the [k] leading keys of the sort
   of [h], in sort order, as [ranked] would give them. *)
let rounds axis k h =
  let s = extents h in
  let low = bit_length (List.nth s axis - 1) in
  let c = Ops.sub (most h) (packed axis h) in
  let one = List.mapi (fun i e -> if i = axis then 1 else e) s in
  let round = function
    | [] -> Shape.rop c Op.Max [ axis ]
    | last :: _ ->
        let below = Ops.lt c (Shape.expand last (dims s)) in
        Shape.rop (Ops.where below c (least c)) Op.Max [ axis ]
  in
  let rec go r acc =
    if r = k then List.rev acc
    else go (r + 1) (Shape.reshape (round acc) (dims one) :: acc)
  in
  let ms = go 0 [] in
  let m = Shape.cat ~axis (List.hd ms) (List.tl ms) in
  Ops.bitwise_and (Ops.sub (most m) m) (int m ((1 lsl low) - 1))

let sort d ~axis ~descending ~k x =
  let s = extents (List.hd (Arith.parts d x)) in
  let n = List.nth s axis in
  if n > 1 lsl 31 then unlowered "a Sort of more than 2^31 elements";
  let first k u =
    Shape.shrink u
      (List.init (Shape.ndim u) (fun i ->
           if i = axis then Some (Ops.Int 0, Ops.Int k) else None))
  in
  let gathered positions =
    let take u = Bits.take u axis positions in
    Arith.of_parts d (List.map take (Arith.parts d x))
  in
  if n <= 1 || numel s = 0 then
    let positions = zeros Int64 s in
    match k with
    | None -> (x, positions)
    | Some k -> (first k x, first k positions)
  else
    let ks = keys d ~descending x in
    let sorted () =
      match List.rev ks with
      | [] -> assert false
      | least :: rest ->
          List.fold_left
            (fun p h -> Bits.take p axis (ranked axis (Bits.take h axis p)))
            (ranked axis least) rest
    in
    match (k, ks) with
    | Some k, [ h ] when k > 0 && selecting ~k s axis ->
        let positions = selected axis k h in
        (gathered positions, positions)
    | Some k, [ h ] when k > 0 && rounding ~k n ->
        let positions = rounds axis k h in
        (gathered positions, positions)
    | None, _ ->
        let positions = sorted () in
        (gathered positions, positions)
    | Some k, _ ->
        let positions = sorted () in
        (first k (gathered positions), first k positions)

(* Numbering

   The rows' order by a stable sort of each column's words from the last, as
   unsigned integers: each sort keeps the order of the columns after it
   among its ties. In that order a run of equal rows starts where a row
   differs from the one before, and the run's first row in the input is at
   its start, where the stable sorts put it. Flags at those first rows, in
   input order, rank the keys by first appearance by their running sum; each
   run's rank, read at its start, numbers its rows. A scatter of the rows at
   their keys' first rows' ranks and of ones at their numbers gives each
   key's first row and count. *)

let unique ~host ren device ~size keys =
  let take = Bits.take in
  let n = Bits.extent keys 0 and w = Bits.extent keys 1 in
  if n > 1 lsl 31 then unlowered "a Unique of more than 2^31 rows";
  let i64 = D.Any D.Int64 in
  let fill v s =
    broadcast (Ops.const ~dtype:Int64 (`Int (Bigint.of_int v))) s
  in
  if n = 0 then
    [|
      zeros Int64 [ 0 ];
      fill 0 [ size ];
      zeros Int64 [ size ];
      Shape.reshape (fill 0 [ 1 ]) [];
    |]
  else
    let i = Shape.arange ~dtype:Int64 n in
    let column j =
      let c = Shape.shrink keys [ None; Some (Ops.Int j, Ops.Int (j + 1)) ] in
      Ops.bitcast (Shape.reshape c [ Ops.Int n ]) Uint64
    in
    let positions u =
      snd (sort (D.Any D.Uint64) ~axis:0 ~descending:false ~k:None u)
    in
    let rec from j p =
      if j < 0 then p
      else from (j - 1) (take p 0 (positions (take (column j) 0 p)))
    in
    let p = if w = 0 then i else from (w - 2) (positions (column (w - 1))) in
    let first_row = Ops.eq i (int i 0) in
    let fresh =
      if w = 0 then first_row
      else
        let rows u = laid [ n; w ] 0 u in
        let k = take keys 0 (rows p) in
        let before = take k 0 (rows (Ops.sub i (int i 1))) in
        let differs = Ops.cast (Ops.ne k before) Uint8 in
        let any = Shape.rop differs Op.Max [ 1 ] in
        let differs = Shape.reshape any [ Ops.Int n ] in
        Ops.bitwise_or first_row (Ops.ne differs (int differs 0))
    in
    let fresh = Ops.cast fresh Int64 in
    let scan m x =
      List.hd (Lower_reduce.scan (Monoid m) 0 ~carry:None ~src:i64 i64 x)
    in
    (* The positions of [flags], [ids] and [first] are distinct inside
       their axis: [p] a permutation, and [target] the ranks of the rows
       that open a run, [-1] elsewhere. Those of [counts], the ids, repeat
       for each key a group holds. *)
    let scatter ?(unique = true) c ~idx ~updates into =
      scatter ~host ~unique ren device i64 c ~axis:0 ~idx ~updates into
    in
    let start = scan Max (Ops.mul fresh i) in
    let flags = scatter Set ~idx:p ~updates:fresh (zeros Int64 [ n ]) in
    let rank = scan Sum flags in
    let numbers = Ops.sub (take rank 0 (take p 0 start)) (int i 1) in
    let ids = scatter Set ~idx:p ~updates:numbers (zeros Int64 [ n ]) in
    let target =
      let opens = Ops.eq flags (int flags 1) in
      Ops.where opens (Ops.sub rank (int rank 1)) (int rank (-1))
    in
    let first = scatter Set ~idx:target ~updates:i (fill n [ size ]) in
    let counts =
      scatter ~unique:false Add ~idx:ids ~updates:(fill 1 [ n ])
        (zeros Int64 [ size ])
    in
    let k = Lower_reduce.reduce (Monoid Sum) [ 0 ] ~src:i64 i64 flags in
    [| ids; first; counts; List.hd k |]
