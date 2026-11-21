(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape
open Heuristic_common

(* The host

   On the host ([CPU] device) a kernel's lanes are counted in its target's
   vector registers: a register of [bytes] holds [bytes / e] lanes of a type
   that computes in [e] bytes. C computes a type narrower than 4 bytes as an
   [int] or a [float], so its lanes take 4 bytes each. A host renderer that
   does not know its registers, as riscv64's, keeps the host's vectors but
   takes tinygrad's upcasts and unroll. *)

(* The powers of two from the largest at most [cap] down to 2. *)
let pow2s_within cap =
  let rec largest p = if 2 * p > cap then p else largest (2 * p) in
  let rec down p = if p < 2 then [] else p :: down (p / 2) in
  down (largest 1)

let rec down_from n = if n < 2 then [] else n :: down_from (n - 1)

(* Upcasts [axis] by the first of [amounts] that divides it and [keep] admits. *)
let tile_split k axis amounts ~keep =
  match
    List.find_opt (fun n -> divisible (shape_at k axis) n && keep n) amounts
  with
  | Some n -> ignore (try_split k axis n Opt.Upcast)
  | None -> ()

let registers k =
  if (K.ren k).target.device = "CPU" then (K.ren k).registers () else None

let on_host k = (K.ren k).target.device = "CPU"
let budgeted k = Option.is_some (registers k)
let computed_bytes dt = max 4 (Dtype.itemsize dt)

(* Each accumulator held across a loop is a dependency chain: a fused
   multiply-add's latency of 4 cycles on each of 2 pipes leaves 8 chains in
   flight, and the operands they read take the rest. So the kernel's
   accumulators take at most one register in [held_share]: 8 of AVX2's 16, a
   tile of 16 floats by 4 rows. Past it they spill a loop-carried value into
   the chain it delays. Every reduce holds one accumulator for each output
   lane, as many to a register as fit and as its stores write at consecutive
   addresses, since Clang packs registers as wide as the stores. *)
let held_share = 2

(* The registers the accumulators of [k]'s reduces take once [upcast], if any,
   upcasts an axis by [n] lanes. *)
let held ?upcast k (file : Renderer.registers) =
  let rngs = K.rngs k and shape = K.full_shape k in
  let lanes = match upcast with Some (_, n) -> n | None -> 1 in
  let axis_size r =
    match List.find_index (fun x -> x == r) rngs with
    | Some i -> Option.map Bigint.to_int (known_size (List.nth shape i))
    | None -> Some 1
  in
  let output =
    List.fold_left
      (fun n i -> n * Bigint.to_int (size_at k i))
      lanes
      (K.axes_of ~reduce:false k [ Upcast ])
  in
  (* The lanes a store writes at consecutive addresses, the upcast [upcast]
     among them: Clang packs a register with no more of them. *)
  let stride r i = Option.map Bigint.to_int (stride r i) in
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
  let stored = stored_addresses k in
  let packed =
    List.fold_left
      (fun c b -> if List.memq b stored then max c (run (idx b)) else c)
      1 (K.bufs k)
  in
  let lanes_registers n dt =
    let per = min (max 1 (file.bytes / computed_bytes dt)) packed in
    (n + per - 1) / per
  in
  List.fold_left
    (fun total u -> total + lanes_registers output (dtype u))
    0 (K.reduceops k)

(* A conversion computes as one vector the lanes upcast of every axis its value
   reads, and on the host the number of them decides whether a product unrolls
   its sum: gpt-oss's key and value projection, with a bfloat16 matrix of 512
   columns, takes 1.9 times as long unrolled by 4, 128 lanes, as with no
   unroll. So on the host a conversion takes at most [converted_lanes]. Its
   lanes are those holding distinct values: [n] lanes of an axis read as a
   quotient by [d] hold [n / gcd n d], as a block of rows reads its expert
   once, and a reduce's lanes are summed before it. A float sum of products
   computes a conversion that differs along every upcast axis one run at a
   time ({!conversion_lanes}): MXFP4's decode, in gpt-oss's routed product,
   so takes 4 experts by 16 rows unrolled by 8 codes, within this budget, and
   ran 2-3 times as fast as in 4 experts by 3 rows unrolled by 4. *)
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
  (* A float sum of products on a renderer that computes on vectors computes
     its product one run of its output's lanes at a time, for each unrolled
     lane (D141's sums by runs), through the operations whose every source
     holds all the lanes: a conversion that differs along every upcast axis
     computes at most the lanes of one output axis, and one that a broadcast
     repeats along some axis computes its own lanes whole. *)
  let reduced r =
    List.exists (fun red -> List.memq r (List.tl (src red))) (K.reduceops k)
  in
  let by_runs =
    if (K.ren k).vector_lanes <= 1 then []
    else
      List.filter_map
        (fun red ->
          match arg red with
          | Reduce { op = Op.Add; _ }
            when op (nth red 0) = Op.Mul
                 && Dtype.mem (dtype red) Dtype.[ Float32; Float64 ] ->
              Some (backward_slice_with_self ~calls:Skip (nth red 0))
          | _ -> None)
        (K.reduceops k)
  in
  let converted u =
    let s = spread u in
    if
      List.exists (Nodes.mem u) by_runs
      && List.for_all (fun (r, _) -> List.mem_assq r s) upcasts
    then List.fold_left (fun n (r, l) -> if reduced r then n else max n l) 1 s
    else List.fold_left (fun n (_, l) -> n * l) 1 s
  in
  List.fold_left
    (fun most u -> if op u <> Op.Cast then most else max most (converted u))
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
  match List.rev (K.upcastable_dims ~reduce:true k) with
  | [] -> 1
  | axis :: _ -> (
      match known_size (shape_at k axis) with
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
   applied: its accumulators take at most half the registers,
   a conversion takes at most [converted_lanes] and an iteration at most
   [iteration_lanes]; always off the host. *)
let fits ?upcast ?unroll k =
  match registers k with
  | None -> true
  | Some file -> (
      held ?upcast k file <= file.count / held_share
      && conversion_lanes ?upcast ?unroll k <= converted_lanes
      &&
      match lanes_after ?upcast ?unroll k with
      | Some n -> n <= iteration_lanes
      | None -> false)

(* Whether [k] sums products. *)
let sums_products k =
  List.exists
    (fun r ->
      (match arg r with Reduce { op = Op.Add; _ } -> true | _ -> false)
      && op (nth r 0) = Op.Mul)
    (K.reduceops k)

(* The fewest terms of a block of a sum that an access reading an output axis
   apart may read once ({!stored_contiguous}): 16, the float32 lanes of the 64
   bytes a sum unrolls at most at once ({!unroll_vector}), so that such an
   access is loaded apart at most once an unrolled step. *)
let block_terms = 16

(* On the host, the first output upcast of a reduce kernel takes the axis its
   stores write at unit stride to a host vector of the accumulators' type
   ([host_vector_bytes]), the largest power of two that divides it and fits:
   as the target's vector width splits a contiguous dimension, its stores fill
   each register. Every access that reads the axis must read it at unit
   stride, or, in a kernel that sums products, read it apart in one of two
   ways. It may gather it as a transpose: read at unit stride the last reduce
   axis it reads, past which every reduce axis is short enough to unroll
   whole, as the rows of a matrix stored [out; in] are read by a product
   summed along [in], or a row of MXFP4 codes, two codes a byte, by its
   bytes. Such an access's lanes along the axis are the columns of the rows it
   loads along the sum, which code generation computes by an in-register
   transpose ({!Codegen}) for the products' vectors, and the axis is said to
   be gathered. Or it may read the sum only in blocks of at least
   [block_terms], every reduce axis it reads as such a quotient, as a row's
   scales are read once a block of its codes: its lanes are loaded apart once
   a block. *)
let stored_contiguous k =
  let stored = stored_addresses k in
  let unit r i = List.exists (fun t -> t == r) (split_uop i Op.Add) in
  (* The reduce axes, the last first, in a kernel that sums products. *)
  let summed =
    if not (sums_products k) then []
    else
      List.rev_map
        (fun a -> (List.nth (K.rngs k) a, known_size (shape_at k a)))
        (K.upcastable_dims ~reduce:true k)
  in
  (* The last reduce axis [b] reads, if every reduce axis after it is short
     enough to unroll whole: the unroll then reaches it. *)
  let rec last_read b = function
    | [] -> None
    | (s, _) :: _ when reads s b -> Some s
    | (_, Some n) :: rest when Bigint.(leq n (of_int 3)) -> last_read b rest
    | _ -> None
  in
  let gathers r b =
    (not (unit r (idx b)))
    && match last_read b summed with Some s -> unit s (idx b) | None -> false
  in
  let blocked r b =
    (not (unit r (idx b)))
    && List.exists (fun (s, _) -> reads s b) summed
    && List.for_all
         (fun (s, _) ->
           (not (reads s b))
           || Bigint.(geq (run (idx b) s) (of_int block_terms)))
         summed
  in
  List.find_map
    (fun axis ->
      let r = List.nth (K.rngs k) axis in
      let readers = List.filter (reads r) (K.bufs k) in
      if
        List.exists (fun b -> List.memq b stored && unit r (idx b)) (K.bufs k)
        && List.for_all
             (fun b -> unit r (idx b) || gathers r b || blocked r b)
             readers
      then Some (axis, List.filter (gathers r) readers)
      else None)
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
    let fits axis =
      Option.map
        (fun n -> (axis, n))
        (List.find_opt
           (divisible (shape_at k axis))
           (pow2s_within (min room split_lanes)))
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

(* The most a pad of a stored axis to 16 bytes of lanes may add, a third of
   its extent ({!upcast_contiguous}): the largest waste measured, every
   measured one winning. *)
let pad_waste = (1, 3)

(* Pads the axis [axis] of [k] to a multiple [up] of [m] where [m] does not
   divide its extent [n], [within n up] admits the pad, and the budget holds
   [m] of it. *)
let padded ~within k axis m =
  match known_size (shape_at k axis) with
  | Some n when Bigint.fits_int n ->
      let n = Bigint.to_int n in
      let up = (n + m - 1) / m * m in
      if n mod m <> 0 && within n up && fits ~upcast:(axis, m) k then
        ignore (K.apply_opt k (Opt.Padto { axis; amount = m }))
  | _ -> ()

let upcast_contiguous k =
  match (stored_contiguous k, registers k) with
  | None, _ | _, None -> ()
  | Some (axis, gathering), Some file ->
      if debug () >= 4 then
        Format.eprintf "host contiguous axis : %d of %s@." axis
          (K.colored_shape k);
      let bytes =
        List.fold_left
          (fun b u -> max b (computed_bytes (dtype u)))
          1 (K.reduceops k)
      in
      (* A gathered axis takes one register: the transpose's shuffles grow
         with the registers of a column, which every row of the tile shares.
         The rows then fill the accumulators' budget: the output axis the
         gathering accesses do not read, or read only as a quotient by [d],
         as a block of [d] routed rows reads its expert's matrix, by the
         largest power of two that fits and divides each [d], as BLIS's
         micro-tile holds [mr] rows of [nr] lanes. *)
      let lanes =
        if gathering = [] then host_vector_bytes / bytes else file.bytes / bytes
      in
      let rows =
        List.filter_map
          (fun a ->
            let r = List.nth (K.rngs k) a in
            let runs =
              List.filter_map
                (fun b -> if reads r b then Some (run (idx b) r) else None)
                gathering
            in
            if
              gathering = [] || a = axis
              || List.exists (Bigint.equal Bigint.one) runs
            then None
            else Some (r, runs))
          (K.upcastable_dims k)
      in
      (* In a kernel that sums products, an axis longer than 16 bytes of
         lanes whose largest power-of-two divisor holds fewer of them, as a
         head of 110 dimensions (2), is padded to a multiple of 16 bytes
         where the padding adds at most [pad_waste] of its extent and the
         budget holds 16 bytes of it: its upcast would otherwise stop at
         that divisor, and runs of 1 or 2 floats leave a register mostly
         empty. 16 bytes is the narrowest vector every host computes whole
         (SSE's, NEON's, half of AVX2's): on fudge, x [3328; 440] by 110,
         50, 22 and 6 outputs ran 1.6 to 2.7 times as fast padded to 4
         floats, by 9 and 25, odd, 1.2 to 1.5 times, while padding 36, 44 or
         100, runs of 4 already, to 8 ran no faster. The padded lanes are
         computed and never stored. *)
      (if sums_products k then
         let quad = max 1 (16 / bytes) and num, den = pad_waste in
         padded k axis quad ~within:(fun n up ->
             n > quad && (up - n) * den <= n * num));
      tile_split k axis
        (pow2s_within (min lanes split_lanes))
        ~keep:(fun n -> fits ~upcast:(axis, n) k);
      match List.rev rows with
      | [] -> ()
      | (r, runs) :: _ -> (
          match List.find_index (fun x -> x == r) (K.rngs k) with
          | None -> ()
          | Some a ->
              let divides n =
                List.for_all
                  (fun d -> Bigint.(equal (rem d (of_int n)) zero))
                  runs
              in
              (* Rows that no access reads as a quotient, more than half the
                 most the budget holds and not a multiple of it, are first
                 padded to a multiple of it: each tile of rows reads the
                 gathered operand again, and the upcasts after take 3, 2 or
                 1 rows where 4 does not divide them and 4 rows by 16
                 columns where it does, twice the passes of the budget's 8
                 by 8. A padded row reads its operands at the last real row,
                 ungated (Postrange's pad on the host), and costs its
                 multiply-adds alone: on fudge a bfloat16 product of 5, 7,
                 17 and 57 rows by [5120; 2880] ran 2.5, 3.5, 2.9 and 1.45
                 times as fast, of 12 rows, which 4 divides, 1.19 times, and
                 AION's CPU embed and generate 1.24 and 1.20 times (D188).
                 Rows half a tile or fewer take one pass already, which a pad
                 would only add work to: 3 rows padded to 8 ran 1.1 times
                 slower (untimed). The padded rows are computed and never
                 stored. *)
              (if runs = [] then
                 match
                   List.find_opt
                     (fun n -> fits ~upcast:(a, n) k)
                     (pow2s_within split_lanes)
                 with
                 | Some most ->
                     padded k a most ~within:(fun n _ -> 2 * n > most)
                 | None -> ());
              tile_split k a (pow2s_within split_lanes) ~keep:(fun n ->
                  divides n && fits ~upcast:(a, n) k))

(* On the host, a reduce whose last axis unrolls [unrolled] lanes of at most 3
   fills the rest of a vector of [host_vector_bytes] of its type with lanes of
   the next axis [axis], by the largest power of two that divides it: the
   unrolled lanes compute as one vector, and one iteration's work outweighs its
   loop and the loads its unrolled lanes share. A byte of uint4 codes holds two
   lanes, and MXFP4's product unrolls 8 bytes of them beside their pairs. *)
let unroll_vector k ~fits unrolled axis =
  let reduce = List.hd (K.reduceops k) in
  let room = host_vector_bytes / Dtype.itemsize (dtype reduce) / unrolled in
  tile_split k axis (pow2s_within room) ~keep:fits

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

let share_decodes k m =
  let operands = List.filter Stage.decoded (src m) in
  let cap = if budgeted k then K.upcast_max else unbudgeted_rows in
  (* The outputs along [axis], of [n], that read one value of [o]. *)
  let shared axis n o =
    let r = List.nth (K.rngs k) axis in
    if Nodes.mem r (ranges o) then run o r else n
  in
  let share axis =
    match known_size (shape_at k axis) with
    | None -> ()
    | Some n ->
        let divides f a = Bigint.(equal (rem f (of_int a)) zero) in
        let shares a =
          List.exists (fun o -> divides (shared axis n o) a) operands
        in
        tile_split k axis (down_from cap)
          ~keep:(fun a -> shares a && fits ~upcast:(axis, a) k)
  in
  (* later axes first, as an upcast of a whole axis removes it *)
  if operands <> [] then
    List.iter share (List.rev (K.upcastable_dims k))

(* The rows [share_decodes] takes along an axis whose stores lie apart, as a
   block of routed rows: each row's accumulators then take a register of
   their own, which [held_share] bounds before [upcast_max] does. *)
let tile_rows (ren : Renderer.t) =
  if ren.target.device <> "CPU" then None
  else
    match ren.registers () with
    | Some file -> Some (min K.upcast_max (file.count / held_share))
    | None -> Some unbudgeted_rows

(* The outputs of a reduce of [dt] whose accumulators fill [held_share]'s
   share of the registers, as many lanes to a register as [held] counts: the
   most a tile of the host holds, 8 rows of 8 float32 lanes on AVX2. A host
   that does not know its registers bounds no tile. *)
let tile_outputs (ren : Renderer.t) dt =
  if ren.target.device <> "CPU" then None
  else
    Option.map
      (fun (file : Renderer.registers) ->
        file.count / held_share * max 1 (file.bytes / computed_bytes dt))
      (ren.registers ())

(* A tile enters a value it computes, a narrow float it converts or a value
   it loads at a computed address, into its lanes by shuffles: an insert to
   gather the value and a broadcast of its lane across the run, where the
   load of a stored value broadcasts it. Those take at least the time memory
   streams a byte to a core: counted on Arrow Lake's ports, a bfloat16's
   entry into an 8 by 8 tile takes 0.36 to 0.52 cycles, where two cores
   stream 2.8 bytes a cycle each (gpt-oss-20b's head read at 27 GB/s, at 4.8
   GHz). A copy of a value stored at [dt] moves twice [dt]'s bytes past the
   value read in place: its store, and its reader's load of [dt] in place of
   the source's, which the copy reads instead. So it costs at most an entry a
   byte, 8 for float32. *)
let copy_entries dt = 2 * Dtype.itemsize dt

let held_registers k file = held k file
