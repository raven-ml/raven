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

(* Whether [r] is a term of the index [i], alone or times a constant. *)
let term_of r i =
  List.exists
    (fun t ->
      t == r || (op t = Op.Mul && nth t 0 == r && op (nth t 1) = Op.Const))
    (split_uop i Op.Add)

(* The accesses [u] computes from, when it computes with no reduce. *)
let accesses u =
  let slice = Nodes.to_list (backward_slice_with_self ~calls:Skip u) in
  if List.exists (fun x -> op x = Op.Reduce) slice then []
  else List.filter (fun x -> op x = Op.Index) slice

(* Whether [o] is decoded: its value converts integers it reads into floats,
   by a cast or a bitcast, outside any access's address. *)
let decoded o =
  let converts u =
    (op u = Op.Cast || op u = Op.Bitcast)
    && Dtype.is_float (dtype u)
    && Dtype.is_int (dtype (nth u 0))
  in
  List.exists converts
    (toposort ~calls:Enter ~gate:(fun u -> op u <> Op.Index) o)

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

(* first try the tensor cores *)
let tensor_cores k =
  let use_tc = setting Setting.use_tc in
  let tc_opt = setting Setting.tc_opt in
  let tc_select = setting Setting.tc_select in
  let min_globals = setting Setting.tc_min_globals in
  if
    use_tc > 0
    && (List.length (K.reduce_axes k) = 1 || tc_opt >= 1 || decoded_product k)
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
            let upcast i sizes target =
              Option.iter
                (fun sz -> split i sz target)
                (first_dividing rngs.(i) sizes)
            in
            if min_globals <> 0 then begin
              (* attempt to upcast M, local N, upcast N, skipping upcast N if
                 we'd end up with too few globals *)
              upcast 1 [ 5; 4; 3; 2 ] Opt.Upcast;
              upcast 0 [ 4; 2 ] Opt.Local;
              match first_dividing rngs.(0) [ 5; 4; 3; 2 ] with
              | Some size
                when Sint.resolve ~default:false
                       Sint.(
                         prod_at tk (K.axes_of tk [ Global ])
                         >= Int size * Int min_globals) ->
                  split 0 size Opt.Upcast
              | _ -> ()
            end
            else begin
              (* attempt to upcast M, N, local N *)
              upcast 1 [ 5; 4; 3; 2 ] Opt.Upcast;
              upcast 0 [ 5; 4; 3; 2 ] Opt.Upcast;
              upcast 0 [ 4; 2 ] Opt.Local
            end;
            Some tk)
      [ 0; 1; 2 ]
  else None

(* Matrix-vector products

   A product of a matrix and a vector reads each matrix element once, so it
   runs as fast as memory feeds it: adjacent threads read adjacent elements,
   each thread keeps several loads in flight, and enough threads run to keep
   the memory busy. Where the matrix is laid out decides the layout:

     rows along the reduce, W[n, k]       columns along an output, W[k, n]

     lanes   k ->                         lanes   n ->
     t0 t1 .. t31 t0 t1 .. t31 ...        t0 t1 .. t31   (2 columns each)
     a row's [lanes] threads read it      the reduce splits into threads
     in turn, [rows] rows a workgroup,    until [busy] threads run; with
     unrolled by up to [in_flight]        few outputs a workgroup takes as
                                          few as [sector_lanes] across *)

(* The threads of a SIMD group: a CUDA warp, an AMD wave32, a Metal
   simdgroup. *)
let lanes = 32

(* The rows of a workgroup in the rows layout: 4 SIMD groups. *)
let rows = 4

(* The outputs of a thread in the columns layout. *)
let columns = 2

(* The fewest threads across a row in the columns layout: 8 threads of 2
   bfloat16 columns read a 32-byte sector, memory's unit of transfer. *)
let sector_lanes = 8

(* The threads that keep a GPU's memory busy. *)
let busy = 32768

(* The loads of its row a thread of the rows layout keeps in flight, at
   most. *)
let in_flight = 8

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

(* The vector and the matrix of a product [m] summed over [first], if it is one
   of a matrix and a vector. Both are computations of accesses with no reduce,
   such as values decoded from codes by a table or a vector normalised by a
   scale. The vector runs in some of the matrix's ranges, the matrix in more,
   and one of the vector's accesses reads along [first], alone or at a constant
   stride. *)
let operands first m =
  let is_vector (v, w) =
    let v_accesses = accesses v and w_ranges = ranges w in
    v_accesses <> []
    && accesses w <> []
    && List.exists (fun a -> term_of first (idx a)) v_accesses
    && Nodes.fold (fun r ok -> ok && Nodes.mem r w_ranges) (ranges v) true
    && Nodes.cardinal w_ranges > Nodes.cardinal (ranges v)
  in
  List.find_map
    (fun (v, w) -> if is_vector (v, w) then Some (v, w) else None)
    [ (nth m 0, nth m 1); (nth m 1, nth m 0) ]

(* The ranges of unit stride of the accesses of [w] that read along [first]:
   the matrix is laid out along them. *)
let units first w =
  List.concat_map
    (fun a ->
      if Nodes.mem first (ranges (idx a)) then
        List.filter (fun t -> op t = Op.Range) (split_uop (idx a) Op.Add)
      else [])
    (accesses w)

let matvec k =
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
  let divisible_at r n = divisible (shape_at k (axis r)) n in
  let rows_layout first v globals =
    let threads = pow2_dividing first lanes in
    (* A workgroup's rows are the matrix's own, which the vector does not read,
       so its SIMD groups share the vector's loads. *)
    let own, shared =
      List.partition (fun g -> not (Nodes.mem g (ranges v))) globals
    in
    match List.find_opt (fun g -> divisible_at g rows) (own @ shared) with
    | Some g when threads > 1 && try_split k (axis first) threads Opt.Local ->
        ignore (split k (axis g) rows Opt.Local);
        (* what the lanes leave of the row, if anything *)
        let same u = axis_id u = axis_id first in
        (match List.find_opt same (K.rngs k) with
        | Some rest ->
            let loads = divisor_at_most rest in_flight in
            if loads > 1 then ignore (try_split k (axis rest) loads Opt.Unroll)
        | None -> ());
        Some k
    | _ -> None
  in
  let columns_layout first g =
    let wanted =
      match prod_at k (K.upcastable_dims k) with
      | Int n -> pow2_above (busy * columns / max n 1)
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
  let ren = K.ren k in
  match (K.reduceop k, K.ranges_of k [ Reduce ]) with
  | Some r, first :: _
    when ren.has_local && ren.has_shared
         && Setting.value Setting.mv
         && (match arg r with Reduce { op = Op.Add; _ } -> true | _ -> false)
         && op (nth r 0) = Op.Mul -> (
      match operands first (nth r 0) with
      | None -> None
      | Some (v, w) ->
          let units = units first w and globals = K.ranges_of k [ Global ] in
          if List.memq first units then rows_layout first v globals
          else
            Option.bind
              (List.find_opt (fun g -> List.memq g units) globals)
              (columns_layout first))
  | _ -> None

(* Shared operands

   An operand of a product that reads an output axis only through its quotient
   by [d] takes one value over each run of [d] consecutive values of the axis.
   When the operand is decoded from integer codes, upcasting the axis by [d]
   decodes it once for the run's lanes. A matrix that a block of rows shares is
   such an operand: each block's rows read their block's matrix, so the matrix
   reads the row axis only through the row's block

     row     0  1 | 2  3 | 4  5        W[owner[row / 2]]
     owner   e0   | e3   | e1          runs of 2 rows

   and upcast by 2 along the rows, a matrix decoded from codes and scales is
   decoded once for both rows of a block. An operand of floats is a load that
   the cache serves each lane, and sharing it costs threads: the keys and
   values that attention's query heads share ran slower upcast. *)

(* The most values an upcast of one axis takes: the largest of upcast_more's
   amounts. *)
let run_cap = 4

(* The run of [o] along the range [r]: [d] if [o] reads [r] only as [r / d], 1
   otherwise. *)
let run o r =
  let readers =
    List.filter
      (fun u -> List.memq r (src u))
      (Nodes.to_list (backward_slice_with_self ~calls:Skip o))
  in
  let divisor u =
    if op u = Op.Floordiv && op (nth u 1) = Op.Const then
      known_size (Sym (nth u 1))
    else None
  in
  match List.map divisor readers with
  | Some d :: rest when List.for_all (Option.equal Bigint.equal (Some d)) rest
    ->
      d
  | _ -> Bigint.one

(* The largest amount from [a] down that divides both [run] and [n]. *)
let rec within run n a =
  let divides x = Bigint.(equal (rem x (of_int a)) zero) in
  if a <= 1 then 1
  else if divides run && divides n then a
  else within run n (a - 1)

let upcast_shared k =
  let amount operands axis =
    let r = List.nth (K.rngs k) axis in
    match known_size (shape_at k axis) with
    | None -> 1
    | Some n ->
        List.fold_left
          (fun a o -> max a (within (run o r) n run_cap))
          1 operands
  in
  match K.reduceop k with
  | Some r
    when (match arg r with Reduce { op = Op.Add; _ } -> true | _ -> false)
         && op (nth r 0) = Op.Mul ->
      let operands = List.filter decoded (src (nth r 0)) in
      let to_upcast =
        List.filter_map
          (fun axis ->
            match amount operands axis with
            | a when a > 1 -> Some (axis, a)
            | _ -> None)
          (K.upcastable_dims k)
      in
      (* later axes first, as an upcast of a whole axis removes it *)
      List.iter
        (fun (axis, a) -> ignore (split k axis a Opt.Upcast))
        (List.rev to_upcast)
  | _ -> ()

(* are we grouping? (requires local shape support) *)
let group k =
  if
    Sint.resolve ~default:false
      Sint.(prod_at k (K.upcastable_dims k) <= Int 2048)
  then
    ignore
      (List.find_opt
         (fun axis -> try_split ~top:true k axis 16 Opt.Local)
         (List.filteri (fun i _ -> i < 3) (K.axes_of k [ Reduce ])))

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

(* On the host, an upcast of [amount] that would take the kernel past
   [host_lanes] lanes: each lane holds a value across the loops, and past the
   registers they spill. An unroll of a kernel of several reduces is held to the
   same lanes. *)
let host_lanes = 32
let on_host k = (K.ren k).target.device = "CPU"

let beyond_host_lanes k amount =
  on_host k && not (holds Sint.(K.upcast_size k * Int amount <= Int host_lanes))

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
    (fun dt -> not (List.mem dt Dtype.weaks))
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

(* potentially do more upcasts of non reduce axes based on a heuristic *)
let upcast_more k =
  let rec loop upcasted_axis =
    if
      Sint.resolve Sint.(prod_at k (K.upcastable_dims k) >= Int 1024)
      && holds Sint.(K.upcast_size k < Int 32)
    then begin
      let choice ~fill axis upcast_amount =
        (* if we haven't upcasted it, it mods, and buffer has stride 0 on axis
           while having no stride 0 in the upcasted axis already; a fill asks
           for a buffer of stride 0 on the axis that a reduce reads, a load each
           lane of the axis shares in every iteration of the reduce *)
        let rng = List.nth (K.rngs k) axis in
        let upcast = K.ranges_of k [ Upcast; Unroll ] in
        let bufs = K.bufs k in
        let reuses b =
          if fill then
            (not (reads rng b))
            && List.exists (fun r -> reads r b) (K.ranges_of k [ Reduce ])
          else
            (not (indexes rng b))
            && List.for_all (fun r2 -> indexes r2 b) upcast
        in
        if
          List.mem axis upcasted_axis
          || (not (divisible (shape_at k axis) upcast_amount))
          || beyond_host_lanes k upcast_amount
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
        | [] when on_host k -> choices ~fill:true [ 2 ]
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
        ignore (try_split k axis n Opt.Unroll)
      else largest (n / 2)
  in
  let rec pow2 p = if 2 * p > room then p else pow2 (2 * p) in
  largest (pow2 1)

(* if last reduce dim is small(ish), loop unroll the reduce. NOTE: this can fail
   on multireduce with mismatching dimensions, this is okay *)
let unroll k =
  let small n = holds Sint.(K.upcast_size k <= Int n) in
  if
    K.unrollable_dims k <> []
    && (small 4 || K.axes_of k [ Unroll ] = [])
    && holds Sint.(K.upcast_size k < Int 64)
  then
    let s = size_at k (last (K.unrollable_dims k)) in
    let at_most n x = Bigint.(leq x (of_int n)) in
    let fits n =
      List.length (K.reduceops k) < 2 || not (beyond_host_lanes k n)
    in
    if at_most 32 s then
      begin if
        fits (Bigint.to_int s)
        && try_split k (last (K.unrollable_dims k)) 0 Opt.Unroll
      then
        (* if it's small, upcast a second reduce dimension too *)
        match K.unrollable_dims k with
        | [] -> ()
        | dims ->
            let s2 () = size_at k (last dims) in
            if at_most 3 s && at_most 3 (s2 ()) then begin
              if fits (Bigint.to_int (s2 ())) then
                ignore (try_split k (last dims) 0 Opt.Unroll)
            end
            else if at_most 3 s && on_host k then
              unroll_vector k ~fits (Bigint.to_int s) (last dims)
      end
    else
      let axis = last (K.unrollable_dims k) in
      if divisible (shape_at k axis) 4 && fits 4 then
        ignore (try_split k axis 4 Opt.Unroll)

(* if nothing at all is upcasted and it's easy to, do an upcast *)
let upcast_one k =
  match K.upcastable_dims k with
  | [] -> ()
  | dims ->
      let axis = last dims in
      if K.upcasted k = 0 && divisible (shape_at k axis) 4 then
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
      (K.axes_of k [ Global; Weak ])
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

let hand_coded_optimizations k =
  match tensor_cores k with
  | Some tk -> tk
  | None -> (
      (* make a copy so it does not mutate the input *)
      let k = K.copy k in
      upcast_shared k;
      match matvec k with
      | Some k -> k
      | None ->
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
          k)
