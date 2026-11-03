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
  | Sym u when op u = Op.Const -> (
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

(* Whether [o] is decoded: its value converts integers it reads into floats,
   by a cast or a bitcast, outside any access's address. *)
let decoded o =
  let converts u =
    (op u = Op.Cast || op u = Op.Bitcast)
    && Dtype.is_float (dtype u)
    && Dtype.is_int (dtype (nth u 0))
  in
  List.exists converts
    (toposort ~calls:Skip ~gate:(fun u -> op u <> Op.Index) o)

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
          List.exists decoded (src m)
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
    if op u = Op.Div && op (nth u 1) = Op.Const then
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
              upcast ~keep 0 [ 4; 2 ] Opt.Local
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

(* The threads the columns layout runs: as many loads in flight as keep memory
   busy. The same standing as [core_threads]. *)
let stream_threads = 32768

(* The loads of its row a thread keeps in flight, at most, and with rows held,
   each of which multiplies them. More is slower: an MXFP4 matrix of 16384 by
   16384 unrolled by 16 took 364 to 376 us where 8 took 317 to 331. *)
let in_flight = 8
let in_flight_held = 4

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
      else if op t = Op.Mul && nth t 0 == r && op (nth t 1) = Op.Const then
        match value (nth t 1) with `Int c -> Some c | _ -> None
      else None)
    (split_uop i Op.Add)

(* The accesses [u] computes from, when it computes with no reduce. *)
let accesses u =
  let slice = Nodes.to_list (backward_slice_with_self ~calls:Skip u) in
  if List.exists (fun x -> op x = Op.Reduce) slice then []
  else List.filter (fun x -> op x = Op.Index) slice

(* [k]'s contraction: its first reduction, when it sums a product of two
   computations of accesses with no reduce, and the ranges it closes in the
   order of [k]'s axes. *)
let contraction k =
  match K.reduceop k with
  | Some r
    when (match arg r with Reduce { op = Op.Add; _ } -> true | _ -> false)
         && op (nth r 0) = Op.Mul ->
      let m = nth r 0 in
      if accesses (nth m 0) = [] || accesses (nth m 1) = [] then None
      else
        Some
          ( m,
            List.filter
              (fun u -> List.memq u (List.tl (src r)))
              (ranges_of ~reduce:true k [ Weak ]) )
  | _ -> None

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
     over [w]. *)
  let upcast_reuse () =
    let rec most t =
      if Bigint.(equal (rem reuse (of_int t)) zero) then t else most (t - 1)
    in
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
          Some k
        end
    | _ -> None
  in
  let columns_layout g =
    if held then upcast_reuse ();
    let first = List.hd (contracted k) in
    let wanted =
      match prod_at k (K.upcastable_dims k) with
      | Int n -> pow2_above (stream_threads * columns / max n 1)
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

(* Runs on the host

   On the host, which streams no contraction, an operand decoded from integer
   codes that reads an output axis only as its quotient by [d], a run of [d]
   rows, is still decoded once for the run's lanes: the axis is upcast by the
   largest amount up to [run_cap] that divides both [d] and the axis. A
   prompt's routed MXFP4 product in blocks of 16 rows decodes its block's
   matrix once for 4 of its rows. *)

(* The most values a run's upcast takes: the largest of upcast_more's
   amounts. *)
let run_cap = 4

let upcast_runs k =
  match contraction k with
  | None -> ()
  | Some (m, _) ->
      let operands = List.filter decoded (src m) in
      let amount axis =
        let r = List.nth (K.rngs k) axis in
        match known_size (shape_at k axis) with
        | None -> 1
        | Some n ->
            List.fold_left
              (fun a o ->
                let d = run o r in
                let rec within a =
                  if a <= 1 then 1
                  else if
                    Bigint.(equal (rem d (of_int a)) zero)
                    && Bigint.(equal (rem n (of_int a)) zero)
                  then a
                  else within (a - 1)
                in
                max a (within run_cap))
              1 operands
      in
      (* later axes first, as an upcast of a whole axis removes it *)
      List.iter
        (fun axis ->
          match amount axis with
          | a when a > 1 -> ignore (split k axis a Opt.Upcast)
          | _ -> ())
        (List.rev (K.upcastable_dims k))

(* are we grouping? (requires local shape support) *)
let group k =
  if
    Sint.resolve ~default:false
      Sint.(prod_at k (K.upcastable_dims k) <= Int 2048)
  then
    ignore
      (List.find_opt
         (fun axis -> try_split ~top:true k axis 16 Opt.Local)
         (List.filteri (fun i _ -> i < 3) (K.axes_of ~reduce:true k [ Weak ])))

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
        else if op t = Op.Mul && nth t 0 == r && op (nth t 1) = Op.Const then
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

(* Whether [k]'s hottest loop holds its accumulators within the budget once
   [upcast] and [unroll] are applied; always off the host. *)
let fits ?upcast ?unroll k =
  match registers k with
  | None -> true
  | Some file -> held ?upcast ?unroll k file <= file.count / held_share

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

let enough_work k =
  Sint.resolve Sint.(prod_at k (K.upcastable_dims k) >= Int 1024)

(* potentially do more upcasts of non reduce axes based on a heuristic; on the
   host, after the contiguous axis, while the accumulators fit *)
let upcast_more k =
  if budgeted k && enough_work k then upcast_contiguous k;
  let rec loop upcasted_axis =
    if enough_work k && (budgeted k || holds Sint.(K.upcast_size k < Int 32))
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
            else if op c = Op.Mul && nth c 0 == rng && op (nth c 1) = Op.Const
            then const_val (nth c 1)
            else if op c = Op.Mul && nth c 1 == rng && op (nth c 0) = Op.Const
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
          loop (axis :: upcasted_axis)
      | [] -> ()
    end
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

(* prioritize making expand axes local *)
let locals k =
  let ranking =
    List.filter_map
      (fun axis ->
        let r = List.nth (K.rngs k) axis in
        if op (nth r 0) = Op.Const then
          Some (List.exists (fun b -> not (indexes r b)) (K.bufs k), axis)
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
            (fun x -> divisible (shape_at k axis) x && local_size * x <= 128)
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

(* A contraction streams before a tensor core when it does at most
   [per_byte_first] multiply-adds per byte of the buffers its streamed operand
   reads, the least memory it moves. Past it the streamed threads are bound by
   their arithmetic, and a core's tiles, which read their matrix through the
   cache when blocks of rows share it, run faster. On an RTX 5000 Ada an MXFP4
   matrix of 16384 by 16384 multiplying 32 rows, 60 a byte, streams in 1.50 to
   1.52 ms and takes 1.68 ms on a core; gpt-oss-20b's routed blocks of a
   224-token prompt, 66 a byte, take 7.0 ms a layer on the cores and 7.9 ms
   streamed, of a 512-token prompt, 149 a byte, 3.7 and 17.5 ms. *)
let per_byte_first = 64

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
     | Some (n, bytes) -> Bigint.leq n (Bigint.of_int (per_byte_first * bytes))
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
   indices and shared memory. *)
let streams k =
  let ren = K.ren k in
  if ren.has_local && ren.has_shared && Setting.value Setting.mv then streamed k
  else None

(* The rules for a kernel that is not streamed and takes no tensor core. *)
let general k =
  (* make a copy so it does not mutate the input *)
  let k = K.copy k in
  if on_host k then upcast_runs k;
  group k;
  (* no more opt if we are grouping *)
  if K.group_for_reduces k = 0 then begin
    upcast_masked k;
    if host_elementwise k then upcast_vectors k
    else begin
      upcast_more k;
      unroll k;
      upcast_one k
    end;
    if (K.ren k).has_local then locals k
  end;
  k

let hand_coded_with ~core_threads k =
  let s = streams k in
  let stream s = stream (K.copy k) s in
  let early = match s with Some s when early k s -> stream s | _ -> None in
  let late () =
    match s with
    | Some ((_, _, reuse) as s) when Bigint.(leq reuse (of_int stream_first))
      ->
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
