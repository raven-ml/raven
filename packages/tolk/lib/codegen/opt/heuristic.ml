(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape
module K = Postrange.Scheduler

let setting = Setting.value
let debug () = setting Setting.debug

(* A split the heuristic has checked applies. *)
let split ?(top = false) k axis amount target =
  match K.apply_opt k (Opt.Split { axis; amount; target; top }) with
  | Ok axes -> axes
  | Error msg -> invalid_arg msg

(* A split that may not apply: whether it did. *)
let try_split ?(top = false) k axis amount target =
  Result.is_ok (K.apply_opt k (Opt.Split { axis; amount; target; top }))

(* A size is known when it is an integer, which a constant beyond an int also
   is. *)
let known_size = function
  | Int n -> Some (Bigint.of_int n)
  | Sym u when is_literal u -> (
      match value u with `Int n -> Some n | _ -> None)
  | Sym _ -> None

let divisible s n =
  match known_size s with
  | Some s -> Bigint.(equal (rem s (of_int n)) zero)
  | None -> false

let shape_at k i = List.nth (K.full_shape k) i
let prod_at k axes = Sint.prod (List.map (shape_at k) axes)
let holds = function Sint.Known b -> b | Sint.Cond u -> to_bool u
let size_at k i = Option.get (known_size (shape_at k i))

let first_dividing r sizes =
  List.find_opt
    (fun sz -> Option.is_some (divides (nth r 0) (Bigint.of_int sz)))
    sizes

let last l = List.nth l (List.length l - 1)

let index_of x l =
  let rec go i = function
    | [] -> invalid_arg "the node is not an axis of the kernel"
    | y :: _ when y == x -> i
    | _ :: r -> go (i + 1) r
  in
  go 0 l

let idx b = get_idx (nth b 1)
let indexes r b = Nodes.mem r (backward_slice ~calls:Skip (idx b))
let reads r b = Nodes.mem r (backward_slice_with_self ~calls:Skip (idx b))

(* The ranges of the axes of [k] of a role in [types], reduced if [reduce]. *)
let ranges_of ?reduce k types =
  List.map (List.nth (K.rngs k)) (K.axes_of ?reduce k types)

(* Whether [k] sums a product with a decoded operand. Decoding packed values
   splits the reduce into a range over the bytes and one over a byte's values,
   and the tensor cores take the bytes. *)
let decoded_product k =
  match K.reduceop k with
  | Some r -> (
      let m = nth r 0 in
      let m = if op m = Op.Cast then nth m 0 else m in
      match arg r with
      | Reduce { op = Op.Add; _ } when op m = Op.Mul ->
          List.exists Stage.decoded (src m)
      | _ -> false)
  | None -> false

(* The threads a kernel keeps once a tensor core's splits apply: a split
   applies only while [core_threads / Tc.threads tc] workgroups remain, each
   holding the core's threads. Fewer leave the memory and the L2 short of loads
   in flight (latency times throughput); more refuse reuse. The value moves the
   order of a sum, so it is the bound of a target's parts, never read from a
   device: measured on sm_89's RTX 5000 Ada, and taken by every target until
   one measures another. *)
let core_threads = 16384

(* The outputs a lane of a tensor core holds at most once [N] takes its last
   upcast. A larger warp tile loads fewer operands for each of the core's
   steps, until its accumulators fill the registers: on sm_89, 64 by 80
   outputs a warp of CUDA's m16n8 core, 160 a lane, compile to 228 of a
   thread's 255 registers and ran fastest; 192 ran slower and 256 spill. A
   staged core takes its own tile ([staged_tile]). *)
let accumulators = 160

(* A local split of a tensor core's [N] leaves at least the renderer's
   [local_workgroups]. The split keeps the threads and packs a workgroup's
   warps onto one compute unit, which balances the units coarser: on an RTX
   5000 Ada's 100 SMs a split that left 512 workgroups ran as fast as one that
   left 256, and 128 or 144 lost up to 1.34 times. Fitted to sm_89's parts, as
   [core_threads] is. It governs a core whose operands are not staged: a
   staged core's local split shares each copied tile among its warps, which
   [staged_tile] decides. *)

(* The outputs a lane of [k] holds: the product of the sizes of the upcast axes
   its stored addresses read. A tensor core's upcasts of its reduce do not
   count: they hold the operands of one step. *)
let lane_outputs k =
  let stored =
    List.filter_map
      (fun u -> if op u = Op.Store then Some (nth u 0) else None)
      (toposort ~calls:Skip (K.ast k))
  in
  let read axis =
    let r = List.nth (K.rngs k) axis in
    List.exists (fun s -> Nodes.mem r (backward_slice ~calls:Skip s)) stored
  in
  prod_at k (List.filter read (K.axes_of k [ Upcast ]))

let rec gcd a b = if b = 0 then a else gcd b (a mod b)

(* The run of [o] along the range [r]: [d] if [o] reads [r] only as [r / d], 1
   otherwise. *)
let run o r =
  let readers =
    List.filter
      (fun u -> List.memq r (src u))
      (Nodes.to_list (backward_slice_with_self ~calls:Skip o))
  in
  let divisor u =
    if op u = Op.Div && is_literal (nth u 1) then
      known_size (Sym (nth u 1))
    else None
  in
  match List.map divisor readers with
  | Some d :: rest when List.for_all (Option.equal Bigint.equal (Some d)) rest
    ->
      d
  | _ -> Bigint.one

(* The rows of the range [m] that read one value of an operand of [k]'s
   product: the gcd of the runs of the operands that read [m] only as
   [m / d], as a block's rows read their owner's matrix at [owner[r / c]], or
   1 where none does. *)
let owner_run k m =
  match K.reduceop k with
  | None -> 1
  | Some red ->
      let strip u = if op u = Op.Cast then nth u 0 else u in
      let p = strip (nth red 0) in
      let p = if op p = Op.Where then strip (nth p 1) else p in
      let add g o =
        let d = run o m in
        if Bigint.(gt d one) then gcd g (Bigint.to_int d) else g
      in
      if op p <> Op.Mul then 1 else max 1 (List.fold_left add 0 (src p))

(* The upcasts a tensor core's rows and columns try, the first that divides
   its range taken. *)
let core_upcasts = [ 5; 4; 3; 2 ]

(* The core's M upcast [u] of [k]'s range [m], which the core split into
   [rest] and its tile: [u], or where a warp of [u] tiles would cross runs of
   rows ({!owner_run}), the first of {!core_upcasts} dividing [rest] that keeps
   a warp within one run, if one does. A warp whose rows cross runs loads a
   tile of each run's value: over blocks of 64 rows each multiplying its
   expert's matrix, an upcast of 5 makes warps of 80 rows across two blocks,
   and 4 warps of 64 within one. *)
let within_run k m rest u =
  match (known_size (Sym (nth m 0)), known_size (Sym (nth rest 0))) with
  (* The tile is exact where the core did not pad M. *)
  | Some a, Some b when Bigint.(gt b zero && equal (rem a b) zero) ->
      let tile = Bigint.(to_int (div a b)) and owned = owner_run k m in
      let fits v = owned mod (tile * v) = 0 in
      let divides v = Option.is_some (first_dividing rest [ v ]) in
      if fits u then u
      else
        Option.value ~default:u
          (List.find_opt (fun v -> fits v && divides v) core_upcasts)
  | _ -> u

(* A staged core's tile. Where a core's operands are staged through shared
   memory ({!Stage}), the warps of a workgroup share each copied tile, so a
   local split reuses what a copy loads. Each warp computes [warp_tile] by
   [warp_tile] outputs, by upcasts of powers of two, and a workgroup takes 2 by
   2 warps where M and N divide, one on each of the four schedulers of every
   NVIDIA SM since Volta. K is unrolled to [chunk] elements a trip. While the
   kernel holds fewer than [core_threads / stages] threads, [stages] the
   pipeline's depth ({!Stage.depth}), the warp tile halves, M then N, down to
   half of [warp_tile] a side: a pipeline overlaps a warp's copies with its
   cores, so it needs fewer warps in flight. On an RTX 5000 Ada, bfloat16
   products of 512 by 512 by 2880 and of 64 rows by 5760 by 2880 ran 1.6 to 1.9
   times slower without the halving; at two stages the knee lies between 5760
   and 11520 threads. *)
let warp_tile = 64
let chunk = 32

(* The staged core's splits, applied to [tk], [k] with the core applied,
   whose core made the axes [rngs] (N, M and K), if the core is staged:
   [split i size target] splits the axis [rngs.(i)]. A warp's rows, and a
   workgroup's, stay within a run of rows that read one value of an operand
   ({!within_run}). *)
let staged_tile ~core_threads k tk rngs split =
  let core =
    List.find_map
      (fun u -> match arg u with Wmma w -> Some w | _ -> None)
      (toposort ~calls:Skip (K.ast tk))
  in
  let size i =
    match known_size (Sym (nth rngs.(i) 0)) with
    | Some b -> Bigint.to_int b
    | None -> 0
  in
  match core with
  | None -> false
  | Some w ->
      let n, m, kd = w.dims and itemsize = Dtype.itemsize w.dtype_in in
      let sn = size 0 and sm = size 1 and sk = size 2 in
      let unroll =
        if chunk mod kd = 0 && sk mod (chunk / kd) = 0 then chunk / kd else 1
      in
      let workgroups =
        match prod_at tk (K.axes_of tk [ Global ]) with
        | Int g -> Some g
        | Sym _ -> None
      in
      let rows =
        List.find_opt (fun r -> axis_id r = axis_id rngs.(1)) (K.rngs k)
      in
      (* Whether [u] of the core's tiles of rows stay within a run of rows
         that read one value of an operand ({!owner_run}), where the core did
         not pad M. *)
      let in_run u =
        match rows with
        | None -> true
        | Some r -> (
            let owned = owner_run k r in
            match known_size (Sym (nth r 0)) with
            | _ when owned = 1 -> true
            | Some a when sm > 0 && Bigint.(equal (rem a (of_int sm)) zero) ->
                owned mod (Bigint.to_int a / sm * u) = 0
            | _ -> false)
      in
      let rec within u = if u <= 1 || in_run u then u else within (u / 2) in
      (* The upcasts of M and N, and their local splits, of a warp tile
         halved [hm] and [hn] times. *)
      let plan (hm, hn) =
        let um = within (warp_tile / m / hm) and un = warp_tile / n / hn in
        if um < 1 || un < 1 || sm = 0 || sn = 0 || sm mod um <> 0
           || sn mod un <> 0
        then None
        else
          let lm = if sm / um mod 2 = 0 && in_run (um * 2) then 2 else 1
          and ln = if sn / un mod 2 = 0 then 2 else 1 in
          Some (um, un, lm, ln)
      in
      let plans = List.filter_map plan [ (1, 1); (2, 1); (2, 2) ] in
      let bytes (um, un, lm, ln) =
        let depth = kd * unroll * itemsize in
        [ lm * um * m * depth; ln * un * n * depth ]
      in
      (* The staged tile is taken only where staging stages the core's
         operands, which it decides of the kernel as the core leaves it. *)
      let stages =
        match plans with
        | p :: _ when Stage.admits (K.ren tk) (K.ast tk) ->
            Stage.depth (K.ren tk) (bytes p)
        | _ -> 0
      in
      if stages = 0 then false
      else
        let threads (um, un, lm, ln) =
          match workgroups with
          | Some g -> g / (um * un * lm * ln) * w.threads * lm * ln
          | None -> max_int
        in
        let fits p = threads p >= core_threads / stages in
        let um, un, lm, ln =
          match List.find_opt fits plans with
          | Some p -> p
          | None -> List.hd (List.rev plans)
        in
        if um > 1 then split 1 um Opt.Upcast;
        if un > 4 then begin
          split 0 4 Opt.Upcast;
          split 0 (un / 4) Opt.Upcast
        end
        else if un > 1 then split 0 un Opt.Upcast;
        if lm = 2 then split 1 2 Opt.Local;
        if ln = 2 then split 0 2 Opt.Local;
        if unroll > 1 then split 2 unroll Opt.Upcast;
        true

(* first try the tensor cores *)
let tensor_cores ~core_threads k =
  let use_tc = setting Setting.use_tc in
  let tc_opt = setting Setting.tc_opt in
  let tc_select = setting Setting.tc_select in
  let min_globals = setting Setting.tc_min_globals in
  if
    use_tc > 0
    && (List.length (K.axes_of ~reduce:true k []) = 1
       || tc_opt >= 1 || decoded_product k)
  then
    List.find_map
      (fun axis ->
        let tk = K.copy k in
        match K.apply_opt tk (Opt.Tc { axis; tc_select; tc_opt; use_tc }) with
        | Error _ -> None
        | Ok rngs ->
            let rngs = Array.of_list rngs in
            let split i size target =
              let axis = index_of rngs.(i) (K.rngs tk) in
              rngs.(i) <- List.hd (split tk axis size target)
            in
            (* A split of [i] by the first of [sizes] that divides it, when
               [keep] holds of that size. *)
            let upcast ~keep i sizes target =
              match first_dividing rngs.(i) sizes with
              | Some size when keep size -> split i size target
              | _ -> ()
            in
            (* Whether [n] workgroups are left once a split by [size] divides
               them, or [default] when their count is not known. *)
            let leaves ~default n size =
              Sint.resolve ~default
                Sint.(prod_at tk (K.axes_of tk [ Global ]) >= Int size * Int n)
            in
            let always _ = true in
            (* M's upcast, kept within a run of rows that read one value of
               an operand. *)
            let upcast_m ~keep =
              match first_dividing rngs.(1) core_upcasts with
              | Some size when keep size ->
                  let same r = axis_id r = axis_id rngs.(1) in
                  let size =
                    match List.find_opt same (K.rngs k) with
                    | Some m -> within_run k m rngs.(1) size
                    | None -> size
                  in
                  split 1 size Opt.Upcast
              | _ -> ()
            in
            if min_globals <> 0 then begin
              (* attempt to upcast M, local N, upcast N, skipping upcast N if
                 we'd end up with too few globals *)
              upcast_m ~keep:always;
              upcast ~keep:always 0 [ 4; 2 ] Opt.Local;
              upcast
                ~keep:(leaves ~default:false min_globals)
                0 core_upcasts Opt.Upcast
            end
            else if staged_tile ~core_threads k tk rngs split then ()
            else begin
              (* attempt to upcast M, N, local N. Each gives a workgroup more of
                 the outputs and the kernel fewer workgroups, which pays only
                 while the kernel keeps [core_threads] threads: the core's
                 warp axis is its threads. CUDA's core makes a product of 64
                 by 64 outputs over 1024 32 workgroups of a warp, which its
                 upcasts would leave one, three times slower. *)
              let threads = prod_at tk (K.axes_of tk [ Warp ]) in
              let floor =
                match threads with Int n -> core_threads / n | Sym _ -> 0
              in
              let keep = leaves ~default:true floor in
              upcast_m ~keep;
              upcast ~keep 0 core_upcasts Opt.Upcast;
              (* N takes one more upcast by 2 while a lane holds at most
                 [accumulators] outputs. *)
              let fits size =
                keep size
                && holds Sint.(lane_outputs tk * Int size <= Int accumulators)
              in
              upcast ~keep:fits 0 [ 2 ] Opt.Upcast;
              (* N's local split, by the largest of 4 and 2 that divides it
                 and leaves [local_workgroups]. *)
              let local size =
                Option.is_some (first_dividing rngs.(0) [ size ])
                && leaves ~default:true (K.ren k).local_workgroups size
              in
              Option.iter
                (fun size -> split 0 size Opt.Local)
                (List.find_opt local [ 4; 2 ])
            end;
            Some tk)
      [ 0; 1; 2 ]
  else None

(* Streamed contractions

   A contraction sums the products of two operands. Each element of an operand
   feeds as many outputs as the outputs that read it, its reuse [R]: the
   product of the sizes of the output ranges the operand does not read, times
   [d] for each it reads only as [r / d]. One token's projection reuses its
   matrix once, twelve tokens' twelve times, a matrix read through a gather by
   each row once, and a matrix that blocks of 2 rows share twice:

     row     0  1 | 2  3 | 4  5        W[owner[row / 2]]
     owner   e0   | e3   | e1          R = 2

   With little reuse a contraction runs as fast as memory feeds it its operand
   of least reuse, [W]: each element of [W] is loaded once for its [R] outputs,
   adjacent threads read adjacent elements, and each thread keeps several loads
   in flight. Its unit-stride range decides the layout:

     rows along the reduce, W[n, k]       columns along an output, W[k, n]

     lanes   k ->                         lanes   n ->
     t0 t1 .. t31 t0 t1 .. t31 ...        t0 t1 .. t31   (2 columns each)
     a row's [lanes] threads read it      the reduce splits into threads
     in turn, [rows] rows a workgroup     until [stream_threads] threads
     or a thread, unrolled by up to       run; with few outputs a workgroup
     [in_flight]                          takes as few as [sector_lanes]
                                          across

   and its reuse ranges are upcast, so that a thread holds the outputs that
   share each element it loads. *)

(* The threads of a SIMD group: a CUDA warp, an AMD wave32, a Metal
   simdgroup. *)
let lanes = 32

(* The rows of a workgroup's SIMD groups without reuse, or of a thread's
   registers with it, where each row reuses the other operand's loads. *)
let rows = 4

(* The outputs of a thread in the columns layout. *)
let columns = 2

(* The fewest threads across a row in the columns layout: 8 threads of 2
   bfloat16 columns read a 32-byte sector, memory's unit of transfer. *)
let sector_lanes = 8

(* The columns layout runs the renderer's [stream_threads]: as many loads in
   flight as keep memory busy. The same standing as [core_threads]. *)

(* The loads of its row a thread keeps in flight, at most, and with rows held,
   each of which multiplies them. More is slower: an MXFP4 matrix of 16384 by
   16384 unrolled by 16 took 364 to 376 us where 8 took 317 to 331. *)
let in_flight = 8
let in_flight_held = 4

(* The bytes of the widest load a GPU renderer merges a thread's adjacent
   values into, and of a line of a GPU's first-level cache, which serves a
   SIMD group's load a line at a time. *)
let vector_bytes = 16
let line_bytes = 128

(* The reuse at most that streams: up to it a streamed product runs faster
   than a core's tiles, which leave each warp a serial reduce, and than the
   general rules. A bfloat16 matrix of 16384 by 4096 multiplies 16 rows in
   288 us streamed and 805 us on a core, 32 rows in 550 and 840 us; 64 rows
   take 1.55 ms on a core. Past it a float32 operand streamed loses to the
   general rules' tiles, read once a pass where they share each load: at 64
   rows 2107 to 2171 us against 1909 to 1987, a 1024-cube product 1054
   against 398 us. *)
let stream_first = 32

(* The most values an upcast of one axis takes (apply_opt's bound). *)
let upcast_max = 16

(* The largest power of two at most [cap] that divides [r]'s size. *)
let rec pow2_dividing r cap =
  if cap <= 1 then 1
  else if Option.is_some (divides (nth r 0) (Bigint.of_int cap)) then cap
  else pow2_dividing r (cap / 2)

(* The largest divisor of [r]'s size that is at most [n]. *)
let rec divisor_at_most r n =
  if n <= 1 then 1
  else if Option.is_some (divides (nth r 0) (Bigint.of_int n)) then n
  else divisor_at_most r (n - 1)

(* The least power of two at least [n]. *)
let pow2_above n =
  let rec go p = if p >= n then p else go (2 * p) in
  go 1

(* The reuse of [o] along each output range of [k]: [(g, f)] for each range
   [g] that [f > 1] outputs share an element of [o] along, or [None] if the
   size of a range [o] does not read is not known. *)
let reuse_ranges k o =
  let along g =
    if Nodes.mem g (ranges o) then Some (run o g)
    else known_size (shape_at k (index_of g (K.rngs k)))
  in
  List.fold_left
    (fun acc g ->
      match (acc, along g) with
      | Some acc, Some f when Bigint.(gt f one) -> Some ((g, f) :: acc)
      | Some acc, Some _ -> Some acc
      | _ -> None)
    (Some []) (ranges_of k [ Global ])
  |> Option.map List.rev

let reuse k o =
  Option.map
    (List.fold_left (fun r (_, f) -> Bigint.mul r f) Bigint.one)
    (reuse_ranges k o)

(* The stride of the range [r] in the index [i]: 1 for a term [r], [c] for a
   term [r * c]. *)
let stride r i =
  List.find_map
    (fun t ->
      if t == r then Some Bigint.one
      else if op t = Op.Mul && nth t 0 == r && is_literal (nth t 1) then
        match value (nth t 1) with `Int c -> Some c | _ -> None
      else None)
    (split_uop i Op.Add)

(* The accesses [u] computes from, outside the reductions it reads. *)
let accesses u =
  List.filter
    (fun x -> op x = Op.Index)
    (toposort ~calls:Skip ~gate:(fun x -> op x <> Op.Reduce) u)

(* Whether [u] reads a reduction that runs in a range of [over]: a sum over
   [over] would compute it again at each of its trips. A reduction that runs
   in none is a value of the outputs alone, as a row's expert that a
   reduction over the experts finds. *)
let reduces_in over u =
  List.exists
    (fun x ->
      op x = Op.Reduce && List.exists (fun c -> Nodes.mem c (ranges x)) over)
    (Nodes.to_list (backward_slice_with_self ~calls:Skip u))

(* [k]'s contraction: its first reduction that sums a product of two
   computations of accesses, neither reading a reduction that runs in a
   range the sum closes, over at least one of [k]'s weak axes, and the ranges
   it closes in the order of [k]'s axes. A reduction an operand reads, such as
   a row's expert, comes before it. *)
let contraction k =
  let sums_product r =
    (match arg r with Reduce { op = Op.Add; _ } -> true | _ -> false)
    && op (nth r 0) = Op.Mul
  in
  List.find_map
    (fun r ->
      if not (sums_product r) then None
      else
        let m = nth r 0 in
        let over =
          List.filter
            (fun u -> List.memq u (List.tl (src r)))
            (ranges_of ~reduce:true k [ Weak ])
        in
        let operand o = accesses o <> [] && not (reduces_in over o) in
        if over <> [] && operand (nth m 0) && operand (nth m 1) then
          Some (m, over)
        else None)
    (K.reduceops k)

let contracted k =
  match contraction k with Some (_, over) -> over | None -> []

(* The terms of [a]'s index that are ranges: those of unit stride. *)
let units a = List.filter (fun t -> op t = Op.Range) (split_uop (idx a) Op.Add)

(* The main access of [w]: the one with a range of unit stride whose terms
   span the most bytes along the contracted ranges, the matrix's codes or
   values beside the scales and tables a decoding reads, whose indices hold
   the contracted ranges divided or through a code. *)
let main_access k w =
  let red = contracted k in
  let terms a = split_uop (idx a) Op.Add in
  let bytes a =
    List.fold_left
      (fun n t ->
        match stride t (idx a) with
        | _ when not (List.memq t red) -> n
        | Some _ ->
            Bigint.mul n
              (Option.value ~default:Bigint.one
                 (known_size (shape_at k (index_of t (K.rngs k)))))
        | None -> n)
      (Bigint.of_int (Dtype.itemsize (dtype a)))
      (List.map (fun t -> if op t = Op.Mul then nth t 0 else t) (terms a))
  in
  List.fold_left
    (fun best a ->
      match best with
      | Some b when Bigint.(geq (bytes b) (bytes a)) -> best
      | _ -> Some a)
    None
    (List.filter (fun a -> units a <> []) (accesses w))

(* What [w] is laid out along: [`Rows (u, a)] when the range of unit stride of
   its main access [a] is the contracted range [u], [`Columns g] when it is the
   output range [g]. *)
let layout k w =
  match main_access k w with
  | None -> None
  | Some a -> (
      let units = units a in
      match List.find_opt (fun r -> List.memq r units) (contracted k) with
      | Some u -> Some (`Rows (u, a))
      | None ->
          Option.map
            (fun g -> `Columns g)
            (List.find_opt (fun g -> List.memq g units) (ranges_of k [ Global ])))

(* The streamed operand [w] of [k]'s summed product, the other operand and
   [w]'s reuse. Of two operands of equal reuse, the one laid out along the
   reduce. *)
let streamed k =
  match contraction k with
  | Some (m, _) -> (
      let a = nth m 0 and b = nth m 1 in
      let rows_first o =
        match layout k o with Some (`Rows _) -> true | _ -> false
      in
      match (reuse k a, reuse k b) with
      | Some ra, Some rb ->
          let c = Bigint.compare ra rb in
          if c < 0 || (c = 0 && (rows_first a || not (rows_first b))) then
            Some (a, b, ra)
          else Some (b, a, rb)
      | Some ra, None -> Some (a, b, ra)
      | None, Some rb -> Some (b, a, rb)
      | None, None -> None)
  | None -> None

(* [n]'s greatest common divisor with the positive [a]. *)
let gcd_with a n = gcd a (Bigint.to_int (Bigint.rem n (Bigint.of_int a)))

let stream k (w, x, reuse) =
  (* The axis of [r], or of what replaced it when a split made it shorter. *)
  let axis r =
    let id = axis_id r in
    let rec go i = function
      | [] -> invalid_arg "the range is not an axis of the kernel"
      | u :: _ when axis_id u = id -> i
      | _ :: rest -> go (i + 1) rest
    in
    go 0 (K.rngs k)
  in
  let rest r = List.find_opt (fun u -> axis_id u = axis_id r) (K.rngs k) in
  let divisible_at r n = divisible (shape_at k (axis r)) n in
  let shared = Option.value ~default:[] (reuse_ranges k w) in
  let held = Bigint.(gt reuse one) in
  (* The reuse ranges are upcast by the largest divisor of their product at
     most [upcast_max], each by its share: a thread holds the outputs that
     share each element of [w] it loads. The rest stay global, each a pass
     over [w]. One range that [w] does not read takes the fewest passes,
     [ceil (f / upcast_max)], padded where its rows do not split evenly into
     them: 17 rows take 2 passes of 9, padded to 18, where the largest
     divisor would take 17 passes of 1. *)
  let upcast_reuse () =
    let rec most t =
      if Bigint.(equal (rem reuse (of_int t)) zero) then t else most (t - 1)
    in
    let divisors () =
      ignore
        (List.fold_left
           (fun left (g, f) ->
             let a =
               match known_size (shape_at k (axis g)) with
               | Some n -> gcd_with (gcd_with left f) n
               | None -> 1
             in
             if a > 1 then ignore (split k (axis g) a Opt.Upcast);
             left / a)
           (most upcast_max) shared)
    in
    match shared with
    | [ (g, f) ] when not (Nodes.mem g (ranges w)) ->
        let f = Bigint.to_int f in
        let passes = (f + upcast_max - 1) / upcast_max in
        let t = (f + passes - 1) / passes in
        let padded =
          f mod t = 0
          || Result.is_ok (K.apply_opt k (Opt.Padto { axis = axis g; amount = t }))
        in
        if padded then ignore (split k (axis g) t Opt.Upcast) else divisors ()
    | _ -> divisors ()
  in
  (* Each contracted range that no split has taken, other than the lanes'
     [u], along which [x] is read at unit stride is unrolled whole where its
     values there fit one load of [vector_bytes] and the [lanes] lanes,
     stepping [x] by its stride along [u], load across more than a line: a
     byte's two MXFP4 values multiply [x] at [2 b] and [2 b + 1], so that a
     loop of 2 trips has 32 lanes load float32 values 8 bytes apart, over
     two lines a load, where unrolled each loads both at once. On an RTX 5000
     Ada 64 float32 rows of a 16384-square MXFP4 matrix took 4.8-5.0 ms
     looped and 2.7 ms unrolled; bfloat16 rows, whose lanes load within one
     line, ran up to 1.2 times slower unrolled. A refused split is
     skipped. *)
  let unroll_together u =
    let together a t =
      let bytes = Dtype.itemsize (dtype a) in
      let span = Bigint.of_int (lanes * bytes) in
      match (rest t, stride u (idx a)) with
      | Some r, Some step
        when axis_id r <> axis_id u
             && List.mem (axis r) (K.upcastable_dims ~reduce:true k)
             && Bigint.(gt (mul step span) (of_int line_bytes))
        -> (
          match (known_size (Sym (nth t 0)), known_size (shape_at k (axis r))) with
          | Some n, Some n'
            when Bigint.(
                   equal n n' && gt n one
                   && leq (mul n (of_int bytes)) (of_int vector_bytes)) ->
              Some r
          | _ -> None)
      | _ -> None
    in
    List.iter
      (fun a ->
        List.iter
          (fun t ->
            Option.iter
              (fun r -> ignore (try_split k (axis r) 0 Opt.Upcast))
              (together a t))
          (units a))
      (accesses x)
  in
  let rows_layout u a =
    let i = idx a in
    let threads = pow2_dividing u lanes in
    let whole =
      Option.equal Bigint.equal
        (known_size (shape_at k (axis u)))
        (Some (Bigint.of_int threads))
    in
    (* A workgroup's or a thread's rows are the matrix's own, which the other
       operand does not read, so they share that operand's loads. *)
    let own, others =
      List.partition
        (fun g -> not (Nodes.mem g (ranges x)))
        (List.filter
           (fun g -> Nodes.mem g (ranges w) && not (List.mem_assq g shared))
           (ranges_of k [ Global ]))
    in
    (* [w]'s reduce range of the next stride: the lanes continue onto it when
       they take [u] whole, and the loads in flight come from it. *)
    let next =
      List.filter_map
        (fun r ->
          match stride r i with
          | Some s when Bigint.(gt s one) -> Some (s, r)
          | _ -> None)
        (contracted k)
      |> List.stable_sort (fun (s, _) (t, _) -> Bigint.compare s t)
      |> List.map snd
      |> function
      | n :: _ -> Some n
      | [] -> None
    in
    match List.find_opt (fun g -> divisible_at g rows) (own @ others) with
    | Some g when threads > 1 ->
        if held then upcast_reuse ();
        if not (try_split k (axis u) threads Opt.Local) then None
        else begin
          (match next with
          | Some n when whole && threads < lanes ->
              let more = pow2_dividing n (lanes / threads) in
              if more > 1 then ignore (try_split k (axis n) more Opt.Local)
          | _ -> ());
          if
            not (try_split k (axis g) rows (if held then Opt.Upcast else Opt.Local))
          then None
          else
          (* what the lanes leave of the reduce, if anything, or the next
             range *)
          let left = if whole then Option.bind next rest else rest u in
          (match left with
          | Some r ->
              let n =
                divisor_at_most r (if held then in_flight_held else in_flight)
              in
              if n > 1 then ignore (try_split k (axis r) n Opt.Upcast)
          | None -> ());
          unroll_together u;
          Some k
        end
    | _ -> None
  in
  let columns_layout g =
    if held then upcast_reuse ();
    let first = List.hd (contracted k) in
    let wanted =
      match prod_at k (K.upcastable_dims k) with
      | Int n -> pow2_above ((K.ren k).stream_threads * columns / max n 1)
      | Sym _ -> 1
    in
    (* A workgroup holds [lanes * lanes] threads. Few outputs take fewer of
       them across, down to [sector_lanes], and more along the reduce, so that
       more workgroups share the outputs: 512 outputs take 32 workgroups of 8
       threads across where they took 8 of 32. *)
    let across = max sector_lanes (lanes * lanes / max wanted lanes) in
    if wanted <= 1 || not (divisible_at g (across * columns)) then None
    else begin
      ignore (split k (axis g) across Opt.Local);
      ignore (split k (axis g) columns Opt.Upcast);
      let threads =
        pow2_dividing first (min wanted (lanes * lanes / across))
      in
      if threads > 1 then ignore (try_split k (axis first) threads Opt.Local);
      Some k
    end
  in
  match layout k w with
  | Some (`Rows (u, a)) -> rows_layout u a
  | Some (`Columns g) -> columns_layout g
  | None -> None

(* are we grouping? (requires local shape support)

   Consecutive threads of a SIMD group read consecutive addresses where a load
   allows it, which coalesces the group's load into whole lines. The group's
   threads take consecutive indices of the reduce axis they split, of the first
   three, one that a load reads at unit stride first. Where no load reads the
   axis it takes at unit stride but one reads an output axis so, 16 threads
   first take consecutive indices of that axis, the first made and so the
   fastest, and the group's threads follow them. Threads that each took a run of
   the axis read addresses a run apart: on an RTX 5000 Ada, a compiled sum of
   2048 rows of 8192 float32 took 72 us so and 24 us coalesced, a sum of 2048
   columns of 16384 took 872 us so and 297 us with threads across the
   columns. *)
let group_threads = 16

let group k =
  let loaded k =
    let stored =
      List.filter_map
        (fun u -> if op u = Op.Store then Some (nth u 0) else None)
        (toposort ~calls:Skip (K.ast k))
    in
    List.filter (fun b -> not (List.memq b stored)) (K.bufs k)
  in
  let unit k axis =
    let r = List.nth (K.rngs k) axis in
    List.exists (fun b -> List.memq r (units b)) (loaded k)
  in
  let reduce_axes k =
    List.filteri (fun i _ -> i < 3) (K.axes_of ~reduce:true k [ Weak ])
  in
  (* The axis the group takes, split. *)
  let grouped k =
    let first, rest = List.partition (unit k) (reduce_axes k) in
    List.find_opt
      (fun axis -> try_split k axis group_threads Opt.Local)
      (first @ rest)
  in
  (* An output axis whose split leaves the group one, tried on a copy. *)
  let across axis =
    unit k axis
    &&
    let t = K.copy k in
    try_split t axis group_threads Opt.Local && Option.is_some (grouped t)
  in
  if
    Sint.resolve ~default:false
      Sint.(prod_at k (K.upcastable_dims k) <= Int 2048)
  then begin
    let outputs =
      match grouped (K.copy k) with
      | Some axis when unit k axis -> []
      | _ -> K.axes_of ~reduce:false k [ Global; Weak ]
    in
    (match List.find_opt across outputs with
    | Some axis -> ignore (split k axis group_threads Opt.Local)
    | None -> ());
    ignore (grouped k)
  end

(* if there are small dims with lots of valid masks, upcast them (they might be
   from Tensor.stack) *)
let upcast_masked k =
  let where_gate_rngs =
    List.concat_map
      (fun u ->
        if op u = Op.Where then Nodes.to_list (ranges (nth u 0)) else [])
      (Nodes.to_list (backward_slice ~calls:Skip (K.ast k)))
  in
  (* upcast leading axes first (hack-ish for winograd; we actually want to
     upcast masked axes with low stride first) *)
  let to_upcast =
    List.fold_left
      (fun to_upcast axis ->
        let is_masked = List.memq (List.nth (K.rngs k) axis) where_gate_rngs in
        let upcast =
          List.fold_left
            (fun p a -> Bigint.mul p (size_at k a))
            Bigint.one to_upcast
        in
        let n = size_at k axis in
        if
          Bigint.(leq n (of_int 7))
          && is_masked
          && Bigint.(leq (upcast * n) (of_int 49))
        then begin
          if debug () >= 4 then
            Format.eprintf "upcasting masked axis : %d@." axis;
          to_upcast @ [ axis ]
        end
        else to_upcast)
      [] (K.upcastable_dims k)
  in
  List.iter
    (fun axis -> ignore (split k axis 0 Opt.Upcast))
    (List.rev to_upcast)

(* The host

   On the host ([CPU] device) a kernel's lanes are counted in its target's
   vector registers: a register of [bytes] holds [bytes / e] lanes of a type
   that computes in [e] bytes. C computes a type narrower than 4 bytes as an
   [int] or a [float], so its lanes take 4 bytes each. A host renderer that
   does not know its registers, as riscv64's, keeps the host's vectors but
   takes tinygrad's upcasts and unroll. *)

let registers k =
  if (K.ren k).target.device = "CPU" then (K.ren k).registers () else None

let on_host k = (K.ren k).target.device = "CPU"
let budgeted k = Option.is_some (registers k)
let computed_bytes dt = max 4 (Dtype.itemsize dt)

(* Each accumulator held across a loop is a dependency chain: a fused
   multiply-add's latency of 4 cycles on each of 2 pipes leaves 8 chains in
   flight, and the operands they read take the rest. So the accumulators of the
   kernel's hottest loop take at most one register in [held_share]: 8 of AVX2's
   16, a tile of 16 floats by 4 rows. Past it they spill a loop-carried value
   into the chain it delays.

   The hottest loop is the loop of the reduce with the most iterations around
   its body, its own loops and those its value runs inside. A reduce of at most
   [short_trips] iterations, which Clang unrolls, belongs to the body around it:
   the hottest reduce's loop, or with no longer reduce the output loop. The loop
   holds the accumulators of every reduce live in it: those it runs inside or
   shares, and the short reduces that run inside it, each one for each output
   lane, as many to a register as fit and as its stores write at consecutive
   addresses, since Clang packs registers as wide as the stores. An unrolled
   reduce axis that keeps its loop adds lanes that each iteration sums into
   those accumulators, and holds nothing more. A reduce unrolled whole holds
   each unrolled lane across the loop of another reduce of its body, and nothing
   once none loops: no value of it is carried from one iteration to the next. *)
let held_share = 2
let short_trips = 32

(* The registers the accumulators of [k]'s hottest loop take once [upcast], if
   any, upcasts an axis by [n] lanes, and [unroll], if any, unrolls an axis by
   [n] lanes, [0] for whole. *)
let held ?upcast ?unroll k (file : Renderer.registers) =
  let rngs = K.rngs k and shape = K.full_shape k in
  let lanes = match upcast with Some (_, n) -> n | None -> 1 in
  let axis_size r =
    match List.find_index (fun x -> x == r) rngs with
    | Some i -> (
        match known_size (List.nth shape i) with
        | Some n -> Some (Bigint.to_int n)
        | None -> None)
    | None -> Some 1
  in
  let output =
    List.fold_left
      (fun n i -> n * Bigint.to_int (size_at k i))
      lanes
      (K.axes_of ~reduce:false k [ Upcast ])
  in
  (* The iterations of [r]'s loop and the lanes it is unrolled into. *)
  let split r =
    let size = axis_size r in
    if axis_type r = Upcast then (Some 1, Option.value size ~default:1)
    else
      match unroll with
      | Some (axis, n) when List.nth rngs axis == r -> (
          match size with
          | Some s when n = 0 || n >= s -> (Some 1, s)
          | Some s -> (Some (s / n), n)
          | None -> (None, n))
      | _ -> (size, 1)
  in
  let reduce_ranges = K.reduce_ranges k in
  let product =
    List.fold_left
      (fun t n -> Option.bind t (fun t -> Option.map (( * ) t) n))
      (Some 1)
  in
  let reduces =
    List.map
      (fun u ->
        let closing =
          List.filter (fun r -> List.memq r rngs) (List.tl (src u))
        in
        let loops = List.filter (fun r -> fst (split r) <> Some 1) closing in
        let inside =
          List.filter
            (fun r -> List.memq r reduce_ranges && fst (split r) <> Some 1)
            (Nodes.to_list (ranges (nth u 0)))
        in
        let trips = product (List.map (fun r -> fst (split r)) loops) in
        let around = product (List.map (fun r -> fst (split r)) inside) in
        let unrolled =
          List.fold_left (fun n r -> n * snd (split r)) 1 closing
        in
        let long =
          match trips with Some t -> t > short_trips | None -> true
        in
        (u, loops, inside, around, unrolled, long))
      (K.reduceops k)
  in
  (* Whether [a] iterations are more than [b], an unknown count being more
     than any known one: the first of equals stays the hottest. *)
  let deeper a b =
    match (a, b) with
    | None, None | Some _, None -> false
    | None, Some _ -> true
    | Some a, Some b -> a > b
  in
  let hottest =
    List.fold_left
      (fun best ((_, _, _, around, _, long) as r) ->
        match best with
        | _ when not long -> best
        | Some (_, _, _, a, _, _) when not (deeper around a) -> best
        | _ -> Some r)
      None reduces
  in
  let members =
    match hottest with
    | None -> reduces
    | Some (h, h_loops, h_inside, _, _, _) ->
        List.filter
          (fun (u, loops, inside, _, _, long) ->
            u == h
            || List.exists (fun r -> List.memq r h_inside) loops
            || ((not long) && List.exists (fun r -> List.memq r h_loops) inside))
          reduces
  in
  (* The lanes a store writes at consecutive addresses, the upcast [upcast]
     among them: Clang packs a register with no more of them. *)
  let stride r i =
    List.find_map
      (fun t ->
        if t == r then Some 1
        else if op t = Op.Mul && nth t 0 == r && is_literal (nth t 1) then
          match value (nth t 1) with
          | #Dtype.value as v -> Some (Dtype.Value.to_int v)
          | `Invalid -> None
        else None)
      (split_uop i Op.Add)
  in
  let run i =
    let upcast_terms =
      List.filter_map
        (fun (axis, r) ->
          let size =
            match upcast with
            | Some (a, n) when a = axis -> Some n
            | _ when axis_type r = Upcast -> axis_size r
            | _ -> None
          in
          match (size, stride r i) with
          | Some n, Some s -> Some (s, n)
          | _ -> None)
        (List.mapi (fun axis r -> (axis, r)) rngs)
    in
    let rec extend c =
      match List.assoc_opt c upcast_terms with
      | Some n when n > 1 -> extend (c * n)
      | _ -> c
    in
    extend 1
  in
  let stored =
    List.filter_map
      (fun u -> if op u = Op.Store then Some (nth u 0) else None)
      (toposort ~calls:Skip (K.ast k))
  in
  let packed =
    List.fold_left
      (fun c b -> if List.memq b stored then max c (run (idx b)) else c)
      1 (K.bufs k)
  in
  let lanes_registers n dt =
    let per = max 1 (file.bytes / computed_bytes dt) in
    let per = min per packed in
    (n + per - 1) / per
  in
  List.fold_left
    (fun total (u, loops, _, _, unrolled, _) ->
      let looping_other =
        List.exists
          (fun (v, loops, _, _, _, _) -> v != u && loops <> [])
          members
      in
      let held =
        if loops <> [] then output
        else if looping_other then output * unrolled
        else 0
      in
      total + lanes_registers held (dtype u))
    0 members

(* A conversion computes as one vector the lanes upcast of every axis its value
   reads, and the C compiler's vectorizer packs that vector from the lanes of
   the values it converts. Where those are loads gathered from apart its time
   grows much faster than the lanes: gpt-oss's key and value projection, a
   product with a bfloat16 matrix of 512 columns whose lanes convert 4 rows of
   its unrolled sum by the output lanes, compiles in 0.03 s at 32 lanes, 0.15 s
   at 64 and 0.9 s at 128. So on the host a conversion takes at most
   [converted_lanes]. Its lanes are those holding distinct values: [n] lanes
   of an axis read as a quotient by [d] hold [n / gcd n d], as a block of rows
   reads its expert once, and a reduce's lanes are summed before it. *)
let converted_lanes = 64

(* The most lanes a conversion of [k] computes at once once [upcast] and
   [unroll] are applied. *)
let conversion_lanes ?upcast ?unroll k =
  let lanes axis r =
    let size () = Bigint.to_int (size_at k axis) in
    match (upcast, unroll) with
    | Some (a, n), _ when a = axis -> n
    | _, Some (a, n) when a = axis -> if n = 0 then size () else n
    | _ -> if axis_type r = Upcast then size () else 1
  in
  let upcasts =
    List.filter_map
      (fun (axis, r) ->
        let n = lanes axis r in
        if n > 1 then Some (r, n) else None)
      (List.mapi (fun axis r -> (axis, r)) (K.rngs k))
  in
  let merge a b =
    List.fold_left
      (fun acc (r, n) ->
        match List.assq_opt r acc with
        | Some m when m >= n -> acc
        | _ -> (r, n) :: List.remove_assq r acc)
      a b
  in
  (* The lanes of each upcast axis along which [u]'s value differs. *)
  let known = Tbl.create 64 in
  let rec spread u =
    match Tbl.find_opt known u with
    | Some s -> s
    | None ->
        let s =
          match List.assq_opt u upcasts with
          | Some n -> [ (u, n) ]
          | None when op u = Op.Reduce ->
              List.filter
                (fun (r, _) -> not (List.memq r (List.tl (src u))))
                (spread (nth u 0))
          | None -> (
              let s =
                List.fold_left (fun acc x -> merge acc (spread x)) [] (src u)
              in
              let divisor =
                if op u = Op.Div && is_literal (nth u 1) then
                  known_size (Sym (nth u 1))
                else None
              in
              match divisor with
              | Some d ->
                  List.map (fun (r, n) -> (r, n / gcd_with n (Bigint.abs d))) s
              | None -> s)
        in
        Tbl.replace known u s;
        s
  in
  List.fold_left
    (fun most u ->
      if op u <> Op.Cast then most
      else max most (List.fold_left (fun n (_, l) -> n * l) 1 (spread u)))
    1
    (toposort ~calls:Skip (K.ast k))

(* An iteration computes the lanes of every upcast axis, those of an unrolled
   reduce among them, and the C compiler's time grows much faster than its
   lanes where they put together vectors gathered from apart: rune's
   contraction of 8 groups of rows read at indices read from memory, which
   multiplies its output lanes by its 16 rows unrolled, compiles in 0.04 s at
   64 lanes, 0.23 s at 256, 0.9-1.1 s at 512 and 5.3 s at 1024. Past 512 an
   unroll ran no faster: a product of 64 output lanes summed over 32 values
   unrolled whole, 2048 lanes, took 605-629 us, and 517-533 us with 16 output
   lanes. So on the host an iteration computes at most [iteration_lanes], and
   an output upcast leaves room for an unroll ([unroll_lanes]), as tinygrad's
   32 output lanes leave room below its 64 for it: that
   contraction then takes 32 output lanes and keeps its rows unrolled, loaded
   outside its output loop. *)
let iteration_lanes = 512

(* The lanes an unroll of [k]'s last unrollable axis would take: the axis
   whole when it holds at most 32, 4 otherwise. *)
let unroll_lanes k =
  match K.upcastable_dims ~reduce:true k with
  | [] -> 1
  | dims -> (
      match known_size (shape_at k (last dims)) with
      | Some n when Bigint.(leq n (of_int 32)) -> Bigint.to_int n
      | _ -> 4)

(* The lanes an iteration of [k] computes once [upcast] and [unroll] are
   applied, those the unroll would take after an upcast among them, if its
   upcast axes' sizes are known. *)
let lanes_after ?upcast ?unroll k =
  let by = function
    | Some (_, n) when n > 0 -> n
    | Some (axis, _) -> Bigint.to_int (size_at k axis)
    | None -> 1
  in
  let room = if Option.is_some upcast then unroll_lanes k else 1 in
  Option.map
    (fun n -> Bigint.to_int n * by upcast * by unroll * room)
    (known_size (K.upcast_size k))

(* Whether [k] stays within the host's budget once [upcast] and [unroll] are
   applied: its hottest loop holds its accumulators within half the registers,
   a conversion takes at most [converted_lanes] and an iteration at most
   [iteration_lanes]; always off the host. *)
let fits ?upcast ?unroll k =
  match registers k with
  | None -> true
  | Some file -> (
      held ?upcast ?unroll k file <= file.count / held_share
      && conversion_lanes ?upcast ?unroll k <= converted_lanes
      &&
      match lanes_after ?upcast ?unroll k with
      | Some n -> n <= iteration_lanes
      | None -> false)

(* On the host, the first output upcast of a reduce kernel takes the axis its
   stores write at unit stride to a host vector of the accumulators' type
   ([host_vector_bytes]), the largest power of two that divides it and fits:
   as the target's vector width splits a contiguous dimension, its stores fill
   each register. Every access
   that reads the axis must read it at unit stride, or its lanes gather from
   apart, as a matrix's rows would for a product with a vector. *)
let stored_contiguous k =
  let stored =
    List.filter_map
      (fun u -> if op u = Op.Store then Some (nth u 0) else None)
      (toposort ~calls:Skip (K.ast k))
  in
  let unit r i = List.exists (fun t -> t == r) (split_uop i Op.Add) in
  List.find_opt
    (fun axis ->
      let r = List.nth (K.rngs k) axis in
      List.exists (fun b -> List.memq b stored && unit r (idx b)) (K.bufs k)
      && List.for_all (fun b -> (not (reads r b)) || unit r (idx b)) (K.bufs k))
    (List.rev (K.upcastable_dims k))

(* On the host, a kernel without a reduce computes its upcast lanes as vectors,
   and its output axis is upcast until one value spans [host_vector_bytes]: the
   lanes give the core independent work while each lane waits on the latency of
   its chain. A kernel whose own operations already hold [host_ilp] independent
   operations for each step of its longest chain gains nothing from lanes and
   pays for them in compile time, which grows with lanes times operations: it
   takes no upcast after its masked axes. *)
let host_vector_bytes = 64
let host_ilp = 5
let host_elementwise k = on_host k && K.reduceops k = []

(* The operations of the values [k] stores, the addresses aside. *)
let stored_operations k =
  let values =
    List.filter_map
      (fun u -> if op u = Op.Store then Some (nth u 1) else None)
      (toposort ~calls:Skip (K.ast k))
  in
  List.filter
    (fun u -> Op.Set.mem (op u) Op.Set.elementwise)
    (toposort ~calls:Skip ~gate:(fun u -> op u <> Op.Index) (sink values))

(* The arithmetic operations of [operations] for each of their longest chain:
   a conversion of a value is no step of it. *)
let parallel_operations operations =
  let depth = Tbl.create 64 in
  let step u = if Op.Set.mem (op u) Op.Set.alu then 1 else 0 in
  let longest =
    List.fold_left
      (fun longest u ->
        let d =
          step u
          + List.fold_left
              (fun d s ->
                max d (Option.value (Tbl.find_opt depth s) ~default:0))
              0 (src u)
        in
        Tbl.replace depth u d;
        max longest d)
      0 operations
  in
  let count = List.fold_left (fun n u -> n + step u) 0 operations in
  if longest = 0 then 0 else count / longest

(* The types [k] reads, writes or computes, but addresses'. *)
let value_dtypes k operations =
  List.filter
    (fun dt -> not (Dtype.mem dt Dtype.weaks))
    (List.map (fun b -> dtype (nth b 0)) (K.bufs k) @ List.map dtype operations)

(* The last output axis that a power of two of lanes divides is upcast by the
   largest that fits the vector beside the lanes already upcast, and again while
   the vector has room: a split upcasts at most [split_lanes]. *)
let split_lanes = 16

let upcast_vectors k =
  let operations = stored_operations k in
  let dtypes = value_dtypes k operations in
  let emulated = Decomp_dtype.emulates (K.ren k) in
  let widest =
    List.fold_left (fun w dt -> max w (Dtype.itemsize dt)) 1 dtypes
  in
  let rec upcast room =
    let rec pow2 p = if 2 * p > min room split_lanes then p else pow2 (2 * p) in
    let fits axis =
      let rec largest n =
        if n < 2 then None
        else if divisible (shape_at k axis) n then Some (axis, n)
        else largest (n / 2)
      in
      largest (pow2 1)
    in
    match List.find_map fits (List.rev (K.upcastable_dims k)) with
    | Some (axis, n) ->
        ignore (split k axis n Opt.Upcast);
        upcast (room / n)
    | None -> ()
  in
  let ilp = parallel_operations operations in
  if debug () >= 4 then
    Format.eprintf "host vectors: %d operations, %d per step, %d bytes@."
      (List.length operations) ilp widest;
  match known_size (K.upcast_size k) with
  | Some upcast_size when ilp < host_ilp && not (List.exists emulated dtypes)
    ->
      upcast (host_vector_bytes / widest / Bigint.to_int upcast_size)
  | _ -> ()

let upcast_contiguous k =
  match stored_contiguous k with
  | None -> ()
  | Some axis ->
      if debug () >= 4 then
        Format.eprintf "host contiguous axis : %d of %s@." axis
          (K.colored_shape k);
      let bytes =
        List.fold_left
          (fun b u -> max b (computed_bytes (dtype u)))
          1 (K.reduceops k)
      in
      let rec pow2 p =
        if 2 * p > min (host_vector_bytes / bytes) split_lanes then p
        else pow2 (2 * p)
      in
      let rec largest n =
        if n >= 2 then
          if divisible (shape_at k axis) n && fits ~upcast:(axis, n) k then
            ignore (split k axis n Opt.Upcast)
          else largest (n / 2)
      in
      largest (pow2 1)

(* A GPU's outputs are its threads: it upcasts while 1024 of them remain. On
   the host the upcasts are bounded by the budget alone. A launch runs its
   output loop in blocks on the host's cores (Codegen), and an upcast divides
   that loop, yet the lanes it takes share a load that each block would
   repeat: on all cores of fudge, 64 sums of 2^15 bfloat16 values took
   0.17-0.19 ms in 2 blocks of 32 lanes and 0.65-1.6 ms in 16 blocks of 4,
   the sums of 64 columns over 2^17 rows 2.6-3.5 ms in 4 blocks of 16 lanes
   and 5.4-9.1 ms in 16 of 4, and 8 products of 2^20 values 1.04-1.25 ms in
   2 blocks of 4 lanes and 1.18-1.32 ms in 8 blocks of none. *)
let enough_work k =
  Sint.resolve Sint.(prod_at k (K.upcastable_dims k) >= Int 1024)

(* potentially do more upcasts of non reduce axes based on a heuristic; on the
   host, after the contiguous axis, while the accumulators fit. The axes its
   loop upcast, each with its amount, latest first. *)
let upcast_more k =
  if budgeted k then upcast_contiguous k;
  let rec loop upcasted =
    let upcasted_axis = List.map fst upcasted in
    if budgeted k || (enough_work k && holds Sint.(K.upcast_size k < Int 32))
    then begin
      let choice ~fill axis upcast_amount =
        (* if we haven't upcasted it, it mods, and buffer has stride 0 on axis
           while having no stride 0 in the upcasted axis already; a fill asks
           for a buffer of stride 0 on the axis that a reduce reads, a load each
           lane of the axis shares in every iteration of the reduce *)
        let rng = List.nth (K.rngs k) axis in
        let upcast = ranges_of k [ Upcast ] in
        let bufs = K.bufs k in
        let reuses b =
          if fill then
            (not (reads rng b))
            && List.exists
                 (fun r -> reads r b)
                 (ranges_of ~reduce:true k [ Weak ])
          else
            (not (indexes rng b))
            && List.for_all (fun r2 -> indexes r2 b) upcast
        in
        if
          List.mem axis upcasted_axis
          || (not (divisible (shape_at k axis) upcast_amount))
          || (not (fits ~upcast:(axis, upcast_amount) k))
          || not (List.exists reuses bufs)
        then None
        else
          let stride c =
            let const_val c =
              match value c with
              | #Dtype.value as v -> Dtype.Value.to_int v
              | `Invalid -> 0
            in
            if c == rng then 1
            else if op c = Op.Mul && nth c 0 == rng && is_literal (nth c 1)
            then const_val (nth c 1)
            else if op c = Op.Mul && nth c 1 == rng && is_literal (nth c 0)
            then const_val (nth c 0)
            else 0
          in
          let num_strides = List.length (List.filter (indexes rng) bufs) in
          let sum_strides =
            List.fold_left
              (fun s b ->
                List.fold_left
                  (fun s c -> s + stride c)
                  s
                  (split_uop (idx b) Op.Add))
              0 bufs
          in
          Some (num_strides, sum_strides, axis, upcast_amount)
      in
      let choices ~fill amounts =
        List.concat_map
          (fun axis -> List.filter_map (choice ~fill axis) amounts)
          (K.upcastable_dims k)
      in
      (* consider all upcastable axes with 3 or 4 upcast; on the host, when none
         is left, fill its lanes with 2 *)
      let xb_choices =
        match choices ~fill:false [ 3; 4 ] with
        | [] when budgeted k -> choices ~fill:true [ 2 ]
        | xb_choices -> xb_choices
      in
      match List.sort Stdlib.compare xb_choices with
      | (_, _, axis, amount) :: _ ->
          if debug () >= 4 then
            Format.eprintf "more upcast axis : %d by %d@." axis amount;
          ignore (split k axis amount Opt.Upcast);
          loop ((axis, amount) :: upcasted)
      | [] -> upcasted
    end
    else upcasted
  in
  loop []

(* On the host, a reduce whose last axis unrolls [unrolled] lanes of at most 3
   fills the rest of a vector of [host_vector_bytes] of its type with lanes of
   the next axis [axis], by the largest power of two that divides it: the
   unrolled lanes compute as one vector, and one iteration's work outweighs its
   loop and the loads its unrolled lanes share. A byte of uint4 codes holds two
   lanes, and MXFP4's product unrolls 8 bytes of them beside their pairs. *)
let unroll_vector k ~fits unrolled axis =
  let reduce = List.hd (K.reduceops k) in
  let room = host_vector_bytes / Dtype.itemsize (dtype reduce) / unrolled in
  let rec largest n =
    if n >= 2 then
      if divisible (shape_at k axis) n && fits n then
        ignore (try_split k axis n Opt.Upcast)
      else largest (n / 2)
  in
  let rec pow2 p = if 2 * p > room then p else pow2 (2 * p) in
  largest (pow2 1)

(* An unroll needs fewer than 64 lanes upcast: 16 registers at four lanes a
   register. On the host, fewer registers of lanes than the target has, a
   register holding as many lanes of the reduces' widest type as fit. *)
let unroll_room k =
  match (registers k, known_size (K.upcast_size k)) with
  | None, _ -> holds Sint.(K.upcast_size k < Int 64)
  | Some file, Some lanes ->
      let bytes =
        List.fold_left
          (fun b u -> max b (computed_bytes (dtype u)))
          4 (K.reduceops k)
      in
      ((Bigint.to_int lanes * bytes) + file.bytes - 1) / file.bytes
      < file.count
  | Some _, None -> false

(* if last reduce dim is small(ish), loop unroll the reduce. NOTE: this can fail
   on multireduce with mismatching dimensions, this is okay *)
let unroll k =
  let small n = holds Sint.(K.upcast_size k <= Int n) in
  let unrollable () = K.upcastable_dims ~reduce:true k in
  if
    unrollable () <> []
    && (small 4 || K.axes_of ~reduce:true k [ Upcast ] = [])
    && unroll_room k
  then
    let s = size_at k (last (unrollable ())) in
    let at_most n x = Bigint.(leq x (of_int n)) in
    let fits axis n = fits ~unroll:(axis, n) k in
    let axis = last (unrollable ()) in
    if at_most 32 s then
      begin if fits axis 0 && try_split k axis 0 Opt.Upcast then
        (* if it's small, upcast a second reduce dimension too *)
        match unrollable () with
        | [] -> ()
        | dims ->
            let next = last dims in
            if at_most 3 s && at_most 3 (size_at k next) then
              begin if fits next 0 then ignore (try_split k next 0 Opt.Upcast)
              end
            else if at_most 3 s && on_host k then
              unroll_vector k ~fits:(fits next) (Bigint.to_int s) next
      end
    else if divisible (shape_at k axis) 4 && fits axis 4 then
      ignore (try_split k axis 4 Opt.Upcast)

(* if nothing at all is upcasted and it's easy to, do an upcast *)
let upcast_one k =
  match K.upcastable_dims k with
  | [] -> ()
  | dims ->
      let axis = last dims in
      if K.axes_of k [ Upcast ] = [] && divisible (shape_at k axis) 4 then
        ignore (split k axis 4 Opt.Upcast)

(* A contraction on a GPU, summed in 4 bytes or fewer, whose two upcast
   output axes are each reused, an operand not reading it, is blocked: each axis, upcast by 4, takes a second
   upcast by 2, so that a thread holds 8 by 8 outputs and each element it loads
   feeds 8 of them, its reduce is unrolled by 4, and its workgroup takes one
   SIMD group, [lanes] threads. More outputs a thread hide the loads' latency
   with fewer threads in flight. On an RTX 5000 Ada a float32 product of 2048
   cubed took 2.8 ms with 4 by 4 outputs a thread in workgroups of 128 threads,
   0.84 ms blocked; workgroups of 64 and 128 threads blocked took 0.93 and 1.6
   ms. Whether it blocked. *)
let block k upcasted =
  let reused (axis, _) =
    let r = List.nth (K.rngs k) axis in
    List.exists (fun b -> not (indexes r b)) (K.bufs k)
  in
  let doubles (axis, _) = divisible (shape_at k axis) 2 in
  (* The threads the axes leave a workgroup once doubled. *)
  let threads axes =
    List.fold_left
      (fun n (axis, _) ->
        n * (pow2_dividing (List.nth (K.rngs k) axis) (2 * lanes) / 2))
      1 axes
  in
  (* 64 accumulators of 8 bytes and their operands would pass a thread's
     registers. *)
  let narrow =
    match contraction k with
    | Some (m, _) -> Dtype.itemsize (dtype m) <= 4
    | None -> false
  in
  match List.rev upcasted with
  | [ (_, 4); (_, 4) ] as axes
    when (K.ren k).has_local
         && narrow
         && List.for_all reused axes && List.for_all doubles axes
         && threads axes >= lanes ->
      List.iter (fun (axis, _) -> ignore (split k axis 2 Opt.Upcast)) axes;
      (match K.upcastable_dims ~reduce:true k with
      | [] -> ()
      | dims ->
          let axis = last dims in
          if divisible (shape_at k axis) 4 then
            ignore (try_split k axis 4 Opt.Upcast));
      true
  | _ -> false

(* The threads of a workgroup the local splits make at most, and of a blocked
   contraction's. *)
let workgroup_threads = 128

(* prioritize making expand axes local, of those every call and custom code
   reads *)
let locals ~threads k =
  let ranking =
    List.filter_map
      (fun axis ->
        let r = List.nth (K.rngs k) axis in
        if is_literal (nth r 0) && not (List.memq r (K.threadless k))
        then Some (List.exists (fun b -> not (indexes r b)) (K.bufs k), axis)
        else None)
      (K.axes_of ~reduce:false k [ Global; Weak ])
  in
  let order (e0, a0) (e1, a1) =
    match Bool.compare e1 e0 with 0 -> Int.compare a1 a0 | c -> c
  in
  let to_local =
    List.fold_left
      (fun to_local (_, axis) ->
        let local_size = Helpers.prod (List.map snd to_local) in
        let candidates =
          (if axis = 0 then [ 32 ] else []) @ [ 16; 8; 4; 3; 2 ]
        in
        match
          List.find_opt
            (fun x -> divisible (shape_at k axis) x && local_size * x <= threads)
            candidates
        with
        | Some sz -> to_local @ [ (axis, sz) ]
        | None -> to_local)
      []
      (List.stable_sort order ranking)
  in
  let first_three = List.filteri (fun i _ -> i < 3) to_local in
  ignore
    (List.fold_left
       (fun deleted_shape (axis, local_sz) ->
         let axis = axis - deleted_shape in
         let will_delete_shape =
           Option.equal Bigint.equal
             (known_size (shape_at k axis))
             (Some (Bigint.of_int local_sz))
         in
         ignore (split k axis local_sz Opt.Local);
         if will_delete_shape then deleted_shape + 1 else deleted_shape)
       0
       (List.sort Stdlib.compare first_three))

(* A contraction streams before a tensor core when it sums at least [summed]
   values an output, between the sums measured: the keys of a decode step's
   attention, a sum of 64, took 8.4 us on a core and 13.9 to 14.6 us
   streamed, its 32 lanes holding 2 loads each and a sum across them that a
   core takes in one step; its values, a sum of 1024, 25.2 to 26.6 us on a
   core and 14.6 to 15.3 us streamed. *)
let summed = lanes * in_flight

(* A contraction streams before a tensor core when it does at most the
   renderer's [per_byte_first] multiply-adds per byte of the buffers its
   streamed operand reads, the least memory it moves. Past it the streamed
   threads are bound by their arithmetic, and a core's tiles, which read their
   matrix through the cache when blocks of rows share it, run faster. On an
   RTX 5000 Ada an MXFP4 matrix of 16384 by 16384 multiplying 32 rows, 60 a
   byte, streams in 1.50 to 1.52 ms and takes 1.68 ms on a core; gpt-oss-20b's
   routed blocks of a 224-token prompt, 66 a byte, take 7.0 ms a layer on the
   cores and 7.9 ms streamed, of a 512-token prompt, 149 a byte, 3.7 and
   17.5 ms. Every target takes sm_89's 64 until its own is measured: an
   R9700's (gfx1201) lies above 149, where it streams those blocks faster
   than its cores, and is not yet placed. *)

(* The multiply-adds of [k]'s contraction and the bytes of the buffers its
   operand [w] reads, when their sizes are known and each is a parameter. *)
let per_byte k w =
  let sizes =
    List.map
      (fun r -> known_size (shape_at k (index_of r (K.rngs k))))
      (ranges_of k [ Global ] @ contracted k)
  in
  let buffers =
    List.fold_left
      (fun acc a -> if List.memq (nth a 0) acc then acc else nth a 0 :: acc)
      [] (accesses w)
  in
  if
    buffers = []
    || List.exists (fun b -> op b <> Op.Param) buffers
    || List.exists Option.is_none sizes
  then None
  else
    let product =
      List.fold_left (fun n s -> Bigint.mul n (Option.get s)) Bigint.one
    in
    let bytes b = Dtype.itemsize (dtype b) * max_numel b in
    Some (product sizes, List.fold_left (fun n b -> n + bytes b) 0 buffers)

let early k (w, _, reuse) =
  Bigint.(leq reuse (of_int stream_first))
  && (match per_byte k w with
     | Some (n, bytes) ->
         Bigint.leq n (Bigint.of_int ((K.ren k).per_byte_first * bytes))
     | None -> true)
  &&
  let n =
    List.fold_left
      (fun n r ->
        match known_size (shape_at k (index_of r (K.rngs k))) with
        | Some s -> Option.map (Bigint.mul s) n
        | None -> None)
      (Some Bigint.one) (contracted k)
  in
  match n with Some n -> Bigint.(geq n (of_int summed)) | None -> true

(* The streamed operand of [k]'s contraction, on a renderer with local
   indices and shared memory. A kernel where a range's bound reads an axis is
   not streamed: the layouts upcast output axes by their accesses. Nor is one
   with a call or custom code, which reads the output axes the layouts upcast
   or runs once in each of the threads they make, nor one where an expression
   in the target's language reads an axis. *)
let streams k =
  let ren = K.ren k in
  if
    ren.has_local && ren.has_shared
    && Setting.value Setting.mv
    && List.is_empty (K.read_by_bounds k)
    && List.is_empty (K.opaque k)
    && List.is_empty (K.read_by_opaque k)
  then streamed k
  else None

(* Decodes on the host

   On the host, which streams no contraction, an operand decoded from integer
   codes is decoded again for each of its uses that a tile's lanes do not
   share. Along an output axis that it does not read, or reads only as its
   quotient by [d], [f] outputs read each of its values at one index: every
   row of a product reads its matrix, a block of [d] rows its expert's. That
   axis is upcast first, by the largest amount up to [upcast_max] that
   divides the axis and one such operand's [f] and keeps the kernel within the
   budget, so the tile's rows share one decode of each value (BLIS's fusing
   factor: as many uses of a value loaded into registers as the register set
   holds). On AVX2, 64 rows by an MXFP4 matrix of 1024 by 1024 take 8 rows,
   their stores apart, and 4 columns beside them, where 4 rows decoded each
   value 16 times; the tiles' 8 trips then share a panel of its decoded
   values ({!Stage.decodes}). A prompt's routed product in blocks of 16 rows
   decodes its block's matrix twice. A host that does not know its
   registers takes at most [unbudgeted_rows], upcast_more's largest
   amount. *)
let unbudgeted_rows = 4

let share_decodes k =
  match contraction k with
  | None -> ()
  | Some (m, _) ->
      let operands = List.filter Stage.decoded (src m) in
      let cap = if budgeted k then upcast_max else unbudgeted_rows in
      (* The outputs along [axis], of [n], that read one value of [o]. *)
      let shared axis n o =
        let r = List.nth (K.rngs k) axis in
        if Nodes.mem r (ranges o) then run o r else n
      in
      let share axis =
        match known_size (shape_at k axis) with
        | None -> ()
        | Some n -> (
            let divides f a = Bigint.(equal (rem f (of_int a)) zero) in
            let shares a =
              divides n a
              && List.exists (fun o -> divides (shared axis n o) a) operands
            in
            let rec amount a =
              if a < 2 then None
              else if shares a && fits ~upcast:(axis, a) k then Some a
              else amount (a - 1)
            in
            match amount cap with
            | Some a -> ignore (split k axis a Opt.Upcast)
            | None -> ())
      in
      (* later axes first, as an upcast of a whole axis removes it *)
      if operands <> [] then
        List.iter share (List.rev (K.upcastable_dims k))

(* The rules for a kernel that is not streamed and takes no tensor core. *)
let general k =
  (* make a copy so it does not mutate the input *)
  let k = K.copy k in
  if on_host k then share_decodes k;
  group k;
  (* no more opt if we are grouping *)
  if K.group_for_reduces k = 0 then begin
    upcast_masked k;
    let blocked =
      if host_elementwise k then begin
        upcast_vectors k;
        false
      end
      else begin
        let blocked = block k (upcast_more k) in
        unroll k;
        upcast_one k;
        blocked
      end
    in
    if (K.ren k).has_local then
      locals ~threads:(if blocked then lanes else workgroup_threads) k
  end;
  k

let hand_coded_with ~core_threads k =
  (* A range of no trip leaves no work to share among lanes or threads, and
     the rules below take each range to run: an upcast of the outputs over an
     empty reduce reads vectors of its empty operands. *)
  let no_trip u =
    op u = Op.Range && Dtype.Value.(compare (vmax u) (of_int 0)) < 0
  in
  if List.exists no_trip (Nodes.to_list (backward_slice ~calls:Skip (K.ast k)))
  then K.copy k
  else
    let s = streams k in
    let stream s = stream (K.copy k) s in
    let early = match s with Some s when early k s -> stream s | _ -> None in
    (* Past [stream_first] a decoded operand streams too: the other rules share
       only their upcast of 4 rows, and decode each value in every other tile,
       16 times at 64 rows. *)
    let late () =
      match s with
      | Some ((w, _, reuse) as s)
        when Bigint.(leq reuse (of_int stream_first)) || Stage.decoded w ->
          stream s
      | _ -> None
    in
    match early with
    | Some k -> k
    | None -> (
        match tensor_cores ~core_threads k with
        | Some tk -> tk
        | None -> ( match late () with Some k -> k | None -> general k))

let hand_coded_optimizations k = hand_coded_with ~core_threads k

let held_registers k file = held k file
