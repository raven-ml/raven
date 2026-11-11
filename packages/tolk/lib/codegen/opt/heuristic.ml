(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape
open Heuristic_common
open Heuristic_host

let setting = Setting.value

let prod_at k axes = Sint.prod (List.map (shape_at k) axes)

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

(* The ranges of the axes of [k] of a role in [types], reduced if [reduce]. *)
let ranges_of ?reduce k types =
  List.map (List.nth (K.rngs k)) (K.axes_of ?reduce k types)

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

(* The bytes of the widest load a GPU renderer merges a thread's adjacent
   values into. *)
let vector_bytes = 16

(* Each reduce axis of [k] of [n] values that an operand reads at unit stride
   and the range [own] a tensor core of [tk], [k] with the core applied,
   takes at stride [n], [n] of its values fitting one load of [vector_bytes],
   is unrolled whole in [tk]: the operand reads the two as one run, and the
   values a thread's fragment holds along the core's range become adjacent.
   A decoded 4-bit operand splits the reduce into bytes and a byte's two
   codes, read by the other operand at [2 b + c]: the core takes the bytes,
   and the codes unrolled make one load where two trips read the values two
   apart. A refused split is skipped. *)
let unroll_beside k tk own =
  let beside core a r =
    match known_size (Sym (nth r 0)) with
    | Some n ->
        let at r = stride r (idx a) in
        let bytes = Bigint.(mul n (of_int (Dtype.itemsize (dtype a)))) in
        axis_id r <> axis_id core
        && Option.equal Bigint.equal (at r) (Some Bigint.one)
        && Option.equal Bigint.equal (at core) (Some n)
        && Bigint.(leq bytes (of_int vector_bytes))
    | None -> false
  in
  let unroll r =
    match List.find_opt (fun u -> axis_id u = axis_id r) (K.rngs tk) with
    | Some u ->
        let axis = index_of u (K.rngs tk) in
        if List.mem axis (K.upcastable_dims ~reduce:true tk) then
          ignore (try_split tk axis 0 Opt.Upcast)
    | None -> ()
  in
  match List.find_opt (fun r -> axis_id r = axis_id own) (K.rngs k) with
  | None -> ()
  | Some core ->
      let loads =
        List.filter (fun u -> op u = Op.Index) (toposort ~calls:Skip (K.ast k))
      in
      let reduces = ranges_of ~reduce:true k [ Weak ] in
      List.iter
        (fun r -> if List.exists (fun a -> beside core a r) loads then unroll r)
        reduces

(* The staged core's splits ({!Stage.tile}), applied to [tk], [k] with the core
   applied, whose core made the axes [rngs] (N, M and K), if the core is
   staged: [split i size target] splits the axis [rngs.(i)]. Whether the kernel
   keeps the core's threads on them, or [None] where the core is not staged. A
   warp's rows, and a workgroup's, stay within a run of rows that read one
   value of an operand ({!within_run}). *)
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
  | None -> None
  | Some w -> (
      let workgroups =
        match prod_at tk (K.axes_of tk [ Global ]) with
        | Int g -> Some g
        | Sym _ -> None
      in
      let rows =
        List.find_opt (fun r -> axis_id r = axis_id rngs.(1)) (K.rngs k)
      in
      let sm = size 1 in
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
      match
        Stage.tile (K.ren tk) (K.ast tk) w ~core_threads ~workgroups
          ~sizes:(size 0, sm, size 2)
          ~in_run
      with
      | None -> None
      | Some (splits, filled) ->
          List.iter (fun (i, amount, target) -> split i amount target) splits;
          Some filled)

(* first try the tensor cores *)
let tensor_cores ~core_threads k =
  let use_tc = setting Setting.use_tc in
  let tc_opt = setting Setting.tc_opt in
  let tc_select = setting Setting.tc_select in
  let min_globals = setting Setting.tc_min_globals in
  if use_tc > 0 then
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
            let unstaged () =
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
              (* N's local split, by the largest of 4 and 2 that divides it
                 and leaves [local_workgroups]. *)
              let local size =
                Option.is_some (first_dividing rngs.(0) [ size ])
                && leaves ~default:true (K.ren k).local_workgroups size
              in
              Option.iter
                (fun size -> split 0 size Opt.Local)
                (List.find_opt local [ 4; 2 ]);
              unroll_beside k tk rngs.(2);
              false
            in
            if min_globals <> 0 then begin
              (* attempt to upcast M, local N, upcast N, skipping upcast N if
                 we'd end up with too few globals *)
              upcast_m ~keep:always;
              upcast ~keep:always 0 [ 4; 2 ] Opt.Local;
              upcast
                ~keep:(leaves ~default:false min_globals)
                0 core_upcasts Opt.Upcast;
              unroll_beside k tk rngs.(2);
              Some (tk, false)
            end
            else
              let filled =
                match staged_tile ~core_threads k tk rngs split with
                | Some filled -> filled
                | None -> unstaged ()
              in
              Some (tk, filled))
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

(* The loads of its row a thread keeps in flight, at most, and with rows held,
   each of which multiplies them. More is slower: an MXFP4 matrix of 16384 by
   16384 unrolled by 16 took 364 to 376 us where 8 took 317 to 331. *)
let in_flight = 8
let in_flight_held = 4

(* The bytes of a line of a GPU's first-level cache, which serves a SIMD
   group's load a line at a time. *)
let line_bytes = 128

(* The reuse at most that streams. Within one pass over the operand a streamed
   product runs faster than a core's tiles, which leave each warp a serial
   reduce, and than the general rules. A bfloat16 matrix of 16384 by 4096 multiplies 16 rows in
   281 us streamed and 834 us on a core. Past [upcast_max] a thread no longer
   holds every row, so the operand is read in [ceil (R / upcast_max)] passes:
   the time follows the passes, 519 us for 16 rows forced to 2 passes and 540
   for 32 rows in 2. More rows a thread do not help: 32 rows in one pass take
   607 us. A core of two stages that keeps its threads takes such a kernel
   first ([core_first]), at 32 rows 425 us against 540. A gate of [upcast_max]
   would give the others a slower layout: 24 rows take 524 us streamed and 827
   on an unstaged core, 28 rows 526 and 1693 on the general rules, and
   gpt-oss's 512-token prefill's router, whose operand no core stages, 45 and
   233. Past
   32 a float32 operand streamed loses to the general rules'
   tiles, read once a pass where they share each load: at 64 rows 2107 to 2171
   us against 1909 to 1987, a 1024-cube product 1054 against 398 us; bfloat16's
   64 rows take 433 us on a core. *)
let stream_first = 32

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

(* The axis of [r], or of what replaced it when a split made it shorter. *)
let axis_of k r =
  let id = axis_id r in
  match List.find_index (fun u -> axis_id u = id) (K.rngs k) with
  | Some i -> i
  | None -> invalid_arg "the range is not an axis of the kernel"

let rest_of k r = List.find_opt (fun u -> axis_id u = axis_id r) (K.rngs k)
let divisible_at k r n = divisible (shape_at k (axis_of k r)) n

(* The reuse ranges are upcast by the largest divisor of their product at most
   [upcast_max], each by its share: a thread holds the outputs that share each
   element of [w] it loads. The rest stay global, each a pass over [w]. One
   range that [w] does not read takes the fewest passes, [ceil (f / upcast_max)],
   padded where its rows do not split evenly into them: 17 rows take 2 passes
   of 9, padded to 18, where the largest divisor would take 17 passes of 1. *)
let upcast_reuse k w reuse =
  let shared = Option.value ~default:[] (reuse_ranges k w) in
  let axis = axis_of k in
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

(* A workgroup's or a thread's rows are the matrix's own, which the other
   operand does not read, so they share that operand's loads. *)
let rows_candidates k w x =
  let shared = Option.value ~default:[] (reuse_ranges k w) in
  let own, others =
    List.partition
      (fun g -> not (Nodes.mem g (ranges x)))
      (List.filter
         (fun g -> Nodes.mem g (ranges w) && not (List.mem_assq g shared))
         (ranges_of k [ Global ]))
  in
  own @ others

(* How the streamed layout will take [k]'s axes, its guards held:
   [`Rows (u, a, g, threads)], [g] the range of 4 rows and [threads] the
   lanes along the contracted range [u] of its main access [a], or
   [`Columns (g, k', across, wanted)] on [k'], a copy of [k] whose reuse is
   already upcast, [across] the threads across [g] and [wanted] the threads of
   the kernel; [None] when no layout's guards hold. *)
let streamed_layout k (w, x, reuse) =
  let held = Bigint.(gt reuse one) in
  match layout k w with
  | None -> None
  | Some (`Rows (u, a)) ->
      let threads = pow2_dividing u lanes in
      if threads <= 1 then None
      else
        Option.map
          (fun g -> `Rows (u, a, g, threads))
          (List.find_opt
             (fun g -> divisible_at k g rows)
             (rows_candidates k w x))
  | Some (`Columns g) ->
      let c = K.copy k in
      if held then upcast_reuse c w reuse;
      let wanted =
        match prod_at c (K.upcastable_dims c) with
        | Int n -> pow2_above ((K.ren c).stream_threads * columns / max n 1)
        | Sym _ -> 1
      in
      let across = max sector_lanes (lanes * lanes / max wanted lanes) in
      if wanted <= 1 || not (divisible_at c g (across * columns)) then None
      else Some (`Columns (g, c, across, wanted))

(* Each contracted range that no split has taken, other than the lanes' [u],
   along which [x] is read at unit stride is unrolled whole where its values
   there fit one load of [vector_bytes] and the [lanes] lanes, stepping [x] by
   its stride along [u], load across more than a line: a byte's two MXFP4
   values multiply [x] at [2 b] and [2 b + 1], so that a loop of 2 trips has
   32 lanes load float32 values 8 bytes apart, over two lines a load, where
   unrolled each loads both at once. On an RTX 5000 Ada 64 float32 rows of a
   16384-square MXFP4 matrix took 4.8-5.0 ms looped and 2.7 ms unrolled;
   bfloat16 rows, whose lanes load within one line, ran up to 1.2 times slower
   unrolled. A refused split is skipped. *)
let unroll_together k x u =
  let together a t =
    let bytes = Dtype.itemsize (dtype a) in
    let span = Bigint.of_int (lanes * bytes) in
    match (rest_of k t, stride u (idx a)) with
    | Some r, Some step
      when axis_id r <> axis_id u
           && List.mem (axis_of k r) (K.upcastable_dims ~reduce:true k)
           && Bigint.(gt (mul step span) (of_int line_bytes))
      -> (
        match
          (known_size (Sym (nth t 0)), known_size (shape_at k (axis_of k r)))
        with
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
            (fun r -> ignore (try_split k (axis_of k r) 0 Opt.Upcast))
            (together a t))
        (units a))
    (accesses x)

(* The splits of the rows layout, on a copy of [k]. A split that Postrange
   refuses, as a local split past the shared memory, is skipped: the kernel
   only runs on the threads it has. *)
let rows_layout k (w, x, reuse) (u, a, g, threads) =
  let k = K.copy k in
  let i = idx a in
  let held = Bigint.(gt reuse one) in
  let whole =
    Option.equal Bigint.equal
      (known_size (shape_at k (axis_of k u)))
      (Some (Bigint.of_int threads))
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
  if held then upcast_reuse k w reuse;
  ignore (try_split k (axis_of k u) threads Opt.Local);
  (match next with
  | Some n when whole && threads < lanes ->
      let more = pow2_dividing n (lanes / threads) in
      if more > 1 then ignore (try_split k (axis_of k n) more Opt.Local)
  | _ -> ());
  ignore (try_split k (axis_of k g) rows (if held then Opt.Upcast else Opt.Local));
  (* what the lanes leave of the reduce, if anything, or the next range *)
  let left = if whole then Option.bind next (rest_of k) else rest_of k u in
  (match left with
  | Some r ->
      let n = divisor_at_most r (if held then in_flight_held else in_flight) in
      if n > 1 then ignore (try_split k (axis_of k r) n Opt.Upcast)
  | None -> ());
  unroll_together k x u;
  k

let columns_layout (g, k, across, wanted) =
  let first = List.hd (contracted k) in
  (* A workgroup holds [lanes * lanes] threads. Few outputs take fewer of them
     across, down to [sector_lanes], and more along the reduce, so that more
     workgroups share the outputs: 512 outputs take 32 workgroups of 8 threads
     across where they took 8 of 32. *)
  ignore (split k (axis_of k g) across Opt.Local);
  ignore (split k (axis_of k g) columns Opt.Upcast);
  let threads = pow2_dividing first (min wanted (lanes * lanes / across)) in
  if threads > 1 then ignore (try_split k (axis_of k first) threads Opt.Local);
  k

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

(* The fewest values an output's reduce sums for a group past 2048 outputs.
   A grouped output's 16 threads each sum a 16th of its values, then the
   group combines their 16 partials, so the group pays once each thread's
   share is at least the combine's 16 steps. *)
let group_least = group_threads * group_threads

(* The longest reduce axis that [unroll] unrolls whole, which no group
   pads. *)
let unroll_whole = 32

let group k =
  let loaded k =
    let stored = stored_addresses k in
    List.filter (fun b -> not (List.memq b stored)) (K.bufs k)
  in
  let unit k axis =
    let r = List.nth (K.rngs k) axis in
    List.exists (fun b -> List.memq r (units b)) (loaded k)
  in
  let reduce_axes k =
    List.filteri (fun i _ -> i < 3) (K.axes_of ~reduce:true k [ Weak ])
  in
  (* The axis the group takes, split: of the axes a load reads at unit stride,
     then of the others, the first that splits by 16, else the first longer
     than [unroll_whole] padded to a multiple of 16 by its reduction's
     identity. The maximum of 128 runs of 1571, a prime, took 122 us in one
     workgroup of 32 threads looping 1571 times, and 8 us padded to 1584 in
     128 workgroups of 16. A pad is tried on a copy, since one whose split is
     refused would leave the axis padded and not grouped. *)
  let grouped k =
    let first, rest = List.partition (unit k) (reduce_axes k) in
    let split k axis = try_split k axis group_threads Opt.Local in
    let padded k axis =
      (match known_size (shape_at k axis) with
        | Some n -> Bigint.(gt n (of_int unroll_whole))
        | None -> false)
      && Result.is_ok
           (K.apply_opt k (Opt.Padto { axis; amount = group_threads }))
      && split k axis
    in
    let among axes =
      match List.find_opt (split k) axes with
      | Some axis -> Some axis
      | None ->
          List.find_opt (fun axis -> padded (K.copy k) axis) axes
          |> Option.map (fun axis ->
                 ignore (padded k axis);
                 axis)
    in
    match among first with Some axis -> Some axis | None -> among rest
  in
  (* An output axis whose split leaves the group one, tried on a copy. *)
  let across axis =
    unit k axis
    &&
    let t = K.copy k in
    try_split t axis group_threads Opt.Local && Option.is_some (grouped t)
  in
  (* A kernel of at most 2048 outputs groups. Past them, while they are fewer
     than [stream_threads], the part runs too few threads to hide a load's
     latency, and each output's thread sums its reduce as one serial chain:
     on an RTX 5000 Ada a float32 sum of 4096 rows of 1024 took 25.2 us so
     and 5.6 grouped, a logsumexp of 4096 rows of 2880, whose first kernel
     is a row's maximum, 172 us and 29, and rune's election of 4096 rows 203
     us and 22. A reduce shorter than [group_least] loses to the group's
     combine (8192 rows of 32, 1.6 us against 4.9), so each of the kernel's
     reduces must sum that many, and a contraction keeps its upcasts, whose
     reuse a group gives up (a float32 product of 128 by 4096 by 128, 0.40
     ms against 0.89). *)
  let outs = prod_at k (K.upcastable_dims k) in
  let weak = K.axes_of ~reduce:true k [ Weak ] in
  let long r =
    let closes axis = List.memq (List.nth (K.rngs k) axis) (List.tl (src r)) in
    Sint.resolve ~default:false
      Sint.(prod_at k (List.filter closes weak) >= Int group_least)
  in
  let starved () =
    Sint.resolve ~default:false Sint.(outs < Int (K.ren k).stream_threads)
    && K.reduceops k <> []
    && List.for_all long (K.reduceops k)
    && Option.is_none (contraction k)
  in
  if
    (K.ren k).has_local
    && (Sint.resolve ~default:false Sint.(outs <= Int 2048) || starved ())
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

(* Whether [k]'s contraction sums in 4 bytes or fewer: 64 accumulators of 8
   bytes and their operands would pass a thread's registers. *)
let narrow k =
  match contraction k with
  | Some (m, _) -> Dtype.itemsize (dtype m) <= 4
  | None -> false

(* Whether some operand of [k] does not read its axis [axis], so that the
   outputs along it share that operand's loads. *)
let reused k axis =
  let r = List.nth (K.rngs k) axis in
  List.exists (fun b -> not (indexes r b)) (K.bufs k)

(* The threads that [n] outputs left of an axis upcast by 4 hold once doubled,
   in a power of two (as {!block} counts them). *)
let doubled_threads n =
  let rec pow2 cap =
    if cap <= 1 || n mod cap = 0 then cap else pow2 (cap / 2)
  in
  pow2 (2 * lanes) / 2

(* Whether [k]'s upcasts take an axis by 4 before 3: a narrow contraction on a
   renderer with local indices whose two output axes, each unread by some
   operand, would pass {!block}'s guard once upcast by 4, and whose blocked
   threads, a 64th of its outputs, number at least [core_threads]. tinygrad
   takes 3 first, which leaves such a product at 4 by 3 outputs a thread: on an
   RTX 5000 Ada float32 products of 163840 rows into 432 to 1920 columns, which
   3 divides, ran at 5.3-5.6 TFLOP/s so, and at 15-19 blocked. A product of 48
   rows by 1024 columns blocked kept 768 threads, and took 105 us against 76
   at 4 by 3. *)
let four_first ~core_threads k =
  (K.ren k).has_local && narrow k
  &&
  match List.filter (reused k) (K.upcastable_dims k) with
  | [ a; b ] -> (
      let size axis =
        match known_size (shape_at k axis) with
        | Some n when Bigint.fits_int n -> Some (Bigint.to_int n)
        | _ -> None
      in
      let threads =
        match known_size (prod_at k (K.upcastable_dims k)) with
        | Some n -> Bigint.fdiv n (Bigint.of_int 64)
        | None -> Bigint.zero
      in
      match (size a, size b) with
      | Some x, Some y ->
          x mod 8 = 0 && y mod 8 = 0
          && doubled_threads (x / 4) * doubled_threads (y / 4) >= lanes
          && Bigint.geq threads (Bigint.of_int core_threads)
      | _ -> false)
  | _ -> false

(* potentially do more upcasts of non reduce axes based on a heuristic; on the
   host, after the contiguous axis, while the accumulators fit. The upcasts
   of its loop, latest first, each as the range it split, the upcast range it
   made and its amount: what a split leaves of a range keeps its axis id, and
   one that takes the whole range leaves none. *)
let upcast_more ~core_threads k =
  if budgeted k then upcast_contiguous k;
  let four_first = four_first ~core_threads k in
  let rec loop upcasted =
    let upcasted_axis = List.map (fun (axis, _, _) -> axis) upcasted in
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
      let key (n, s, axis, amount) =
        (n, s, axis, if four_first then -amount else amount)
      in
      let by_key a b = Stdlib.compare (key a) (key b) in
      match List.sort by_key xb_choices with
      | (_, _, axis, amount) :: _ ->
          if debug () >= 4 then
            Format.eprintf "more upcast axis : %d by %d@." axis amount;
          let r = List.nth (K.rngs k) axis in
          let up = List.nth (split k axis amount Opt.Upcast) 1 in
          loop ((axis, (r, up), amount) :: upcasted)
      | [] -> upcasted
    end
    else upcasted
  in
  List.map (fun (_, (r, up), amount) -> (r, up, amount)) (loop [])

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
    if at_most unroll_whole s then
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
   output axes are each reused, an operand not reading it, is blocked: each
   axis, upcast by 4, takes a second upcast by 2, so that a thread holds 8 by 8
   outputs and each element it loads feeds 8 of them, its reduce is unrolled by
   4, and its workgroup takes one SIMD group, [lanes] threads. More outputs a
   thread hide the loads' latency with fewer threads in flight. On an RTX 5000
   Ada a float32 product of 2048 cubed took 2.8 ms with 4 by 4 outputs a thread
   in workgroups of 128 threads, 0.84 ms blocked; workgroups of 64 and 128
   threads blocked took 0.93 and 1.6 ms. An upcast that took its axis whole
   lends its second upcast to the output axis the same operands read, the
   other part of one axis: a projection of 16384 rows into 4 heads of 110
   took 331 us so and 2549 at 4 by 4. Attention's scores, whose queries the
   operands that read the heads do not all read, keep 4 by 4. The ranges of
   the two axes it doubled, if it blocked. *)
let block k upcasted =
  let doubles axis = divisible (shape_at k axis) 2 in
  (* The threads the axes leave a workgroup once doubled. *)
  let threads axes =
    List.fold_left
      (fun n axis ->
        n * (pow2_dividing (List.nth (K.rngs k) axis) (2 * lanes) / 2))
      1 axes
  in
  let outputs () = List.map (List.nth (K.rngs k)) (K.upcastable_dims k) in
  (* For an upcast that took its range whole, the output axis besides
     [other] that the same operands read: the other part of an axis the
     kernel splits in two, as a projection's heads and their width. *)
  let lent (_, up, _) other =
    match rest_of k up with
    | None -> None
    | Some up ->
        let same x =
          axis_id x <> axis_id other
          && List.for_all (fun b -> indexes x b = indexes up b) (K.bufs k)
        in
        List.find_opt same (outputs ())
  in
  let axes =
    match List.rev upcasted with
    | [ ((r0, _, 4) as e0); ((r1, _, 4) as e1) ] -> (
        match (rest_of k r0, rest_of k r1) with
        | Some a, Some b -> Some (a, b)
        | None, Some b -> Option.map (fun a -> (a, b)) (lent e0 b)
        | Some a, None -> Option.map (fun b -> (a, b)) (lent e1 a)
        | None, None -> None)
    | _ -> None
  in
  match axes with
  | Some (ra, rb)
    when (K.ren k).has_local && narrow k
         &&
         let a = axis_of k ra and b = axis_of k rb in
         reused k a && reused k b && doubles a && doubles b
         && threads [ a; b ] >= lanes ->
      List.iter (fun r -> ignore (split k (axis_of k r) 2 Opt.Upcast)) [ ra; rb ];
      (match K.upcastable_dims ~reduce:true k with
      | [] -> ()
      | dims ->
          let axis = last dims in
          if divisible (shape_at k axis) 4 then
            ignore (try_split k axis 4 Opt.Upcast));
      Some (ra, rb)
  | _ -> None

(* The threads of a workgroup the local splits make at most. *)
let workgroup_threads = 128

(* The threads of a blocked contraction's SIMD group along the two axes [rs]
   that {!block} doubled: [greedy]'s where it lays a power of two of them, at
   least 2, along the later axis; else [across] along the later axis and the
   rest of [lanes] along the other, each axis padded to its threads, with
   [across] 16 or 8, whichever pads the later axis less. A product of 262144
   rows by 440 columns, 55 runs of 8 that no power of two divides, took its 32
   threads along the rows and ran at 4.6 TFLOP/s, and padded to 448 columns at
   17.2; 144 columns, 18 runs, ran so at 13.6-14.9 padded to 192, a third more
   work. An axis that takes no threads, as [ranked] omits it, or a pad, which
   refuses past four times the work, that would be refused leaves [greedy]'s.
   *)
let blocked_threads k ranked greedy (ra, rb) =
  match (rest_of k ra, rest_of k rb) with
  | Some ua, Some ub ->
      let a0 = axis_of k ua and a1 = axis_of k ub in
      let later = max a0 a1 and earlier = min a0 a1 in
      let on axis = Option.value ~default:1 (List.assoc_opt axis greedy) in
      let pow2 n = n >= 2 && n land (n - 1) = 0 in
      if pow2 (on later) then greedy
      else if not (List.mem later ranked && List.mem earlier ranked) then greedy
      else
        let waste n =
          match known_size (shape_at k later) with
          | Some s ->
              let n = Bigint.of_int n in
              Bigint.sub (Bigint.mul (Bigint.cdiv s n) n) s
          | None -> Bigint.zero
        in
        let across = if Bigint.leq (waste 16) (waste 8) then 16 else 8 in
        let plan = [ (later, across); (earlier, lanes / across) ] in
        let pads k =
          List.for_all
            (fun (axis, amount) ->
              divisible (shape_at k axis) amount
              || Result.is_ok (K.apply_opt k (Opt.Padto { axis; amount })))
            plan
        in
        if pads (K.copy k) && pads k then plan else greedy
  | _ -> greedy

(* prioritize making expand axes local, of those every call and custom code
   reads; a blocked contraction's ({!block}) take one SIMD group *)
let locals ?blocked k =
  let threads = if Option.is_some blocked then lanes else workgroup_threads in
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
  let greedy =
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
  let to_local =
    match blocked with
    | Some rs -> blocked_threads k (List.map snd ranking) greedy rs
    | None -> greedy
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
  then streamed k
  else None

let stream k (s, l) =
  match l with
  | `Rows r -> rows_layout k s r
  | `Columns c -> columns_layout c

(* The rules for a kernel that is not streamed and takes no tensor core. *)
let general ~core_threads k =
  (* make a copy so it does not mutate the input *)
  let k = K.copy k in
  if on_host k then
    Option.iter (fun (m, _) -> share_decodes k m) (contraction k);
  group k;
  (* no more opt if we are grouping *)
  if K.group_for_reduces k = 0 then begin
    upcast_masked k;
    let blocked =
      if host_elementwise k then begin
        upcast_vectors k;
        None
      end
      else begin
        let blocked = block k (upcast_more ~core_threads k) in
        unroll k;
        upcast_one k;
        blocked
      end
    in
    if (K.ren k).has_local then locals ?blocked k
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
    (* A layout's guards upcast a copy in the columns layout, so they run only
       for a kernel the gates send to a stream. *)
    let layout =
      lazy
        (Option.bind s (fun s ->
             Option.map (fun l -> (s, l)) (streamed_layout k s)))
    in
    let streamed_as () = Option.map (stream k) (Lazy.force layout) in
    let core = lazy (tensor_cores ~core_threads k) in
    (* Past [upcast_max] a stream reads its operand in several passes, which a
       core of two stages that keeps its threads beats. *)
    let core_first (_, _, reuse) =
      Bigint.(gt reuse (of_int upcast_max))
      && match Lazy.force core with Some (_, filled) -> filled | None -> false
    in
    let early_stream =
      match s with
      | Some s when early k s && not (core_first s) -> streamed_as ()
      | _ -> None
    in
    (* Past [stream_first] a decoded operand streams too: the other rules share
       only their upcast of 4 rows, and decode each value in every other tile,
       16 times at 64 rows. *)
    let late () =
      match s with
      | Some (w, _, reuse)
        when Bigint.(leq reuse (of_int stream_first)) || Stage.decoded w ->
          streamed_as ()
      | _ -> None
    in
    (* The guards decide the streams, so only the tensor cores are a true
       attempt: whether one applies is the rewrite itself. *)
    match early_stream with
    | Some k -> k
    | None -> (
        match Lazy.force core with
        | Some (tk, _) -> tk
        | None -> (
            match late () with Some k -> k | None -> general ~core_threads k))

let hand_coded_optimizations k =
  hand_coded_with ~core_threads:(K.ren k).core_threads k

let core_fits (ren : Renderer.t) ~dtype_in ~dtype_out ~rows ~outputs ~block =
  let device = ren.target.device in
  (* TF32 rounds float32's inputs: {!Opt.Tc} takes it only when allowed. *)
  let tf32 (tc : Tc.t) =
    (device = "CUDA" || device = "NV")
    && Dtype.equal tc.dtype_in Dtype.Float32
    && not (setting Setting.allow_tf32)
  in
  let fits (tc : Tc.t) =
    let n, m, k = Tc.dims tc in
    (not (tf32 tc))
    && Dtype.equal tc.dtype_in dtype_in
    && Dtype.equal tc.dtype_out dtype_out
    && block mod k = 0
    && ((rows mod m = 0 && outputs mod n = 0)
       || (rows mod n = 0 && outputs mod m = 0))
  in
  (* A selected core the target lacks is none, as {!Opt.Tc} refuses it. *)
  let cores =
    match setting Setting.tc_select with
    | -1 -> ren.tensor_cores
    | n when n >= 0 -> Option.to_list (List.nth_opt ren.tensor_cores n)
    | _ -> []
  in
  (* [TC=2] lays out a core's shape without its instruction. *)
  setting Setting.use_tc = 1
  && min rows outputs > stream_first
  && List.exists fits cores

let held_registers = held_registers
