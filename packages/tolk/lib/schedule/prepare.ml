(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

let rule = Pattern_matcher.rule
let ops = Op.Set.of_list
let ints = List.map (fun n -> Int n)
let reaches = Ops.reaches ~calls:Skip
let is_empty_shape u = List.exists (Sint.equal (Int 0)) (shape u)
let equal_device_of u0 u1 = Option.equal String.equal (device u0) (device u1)

let forward_call_outputs sink =
  let placed = Tbl.create 8 in
  let forward item =
    let st =
      match src item with
      | [ _; s ]
        when (op item = Op.After
             && op s = Op.Store)
             [@mutate off "any other item is kept"] ->
          s
      | _ -> item
    in
    if op st <> Op.Store || (item != st && nth item 0 != nth st 0) then item
    else
      let target = nth st 0 and value = nth st 1 in
      let rec before_effects s =
        if op s = Op.After then before_effects (nth s 0) else s
      in
      let src = before_effects value in
      let base = storage_base src in
      if item != st && op base <> Op.Alloc then item
      else
        (* Forward the allocation, not just one view of it, so saved values and
           other aliases follow the same placement. *)
        let key = if op base = Op.Alloc then base else src in
        let placement =
          if
            Tbl.mem placed key
            || (not (op src = Op.Stage || has_buffer_identity target))
            || reaches value (storage_base target)
          then None
          else if
            op base = Op.Alloc
            && has_buffer_identity src
            && max_numel base = max_numel (storage_base target)
          then Some (storage_base target)
          else if op src = Op.Stage then
            Some (after target [ store target (nth src 0) ])
          else if op src = Op.Buffer && has_buffer_identity src then
            Some target
          else None
        in
        match placement with
        | Some p ->
            Tbl.replace placed key p;
            if item != st then Tbl.replace placed item value;
            value
        | None -> after target [ st ]
  in
  let items = List.map forward (Ops.src sink) in
  substitute ~calls:Skip ~pass:Once (Ops.sink items)
    (List.of_seq (Tbl.to_seq placed))

let rec walk_mop u =
  if
    Op.Set.mem (op u) Op.Set.movement
    || op u = Op.Index
    || op u = Op.Bitcast
  then walk_mop (nth u 0)
  else if op u = Op.After then
    let b = walk_mop (nth u 0) in
    if b != nth u 0 then replace u ~src:(b :: List.tl (src u)) else u
  else u

(* Movements on an index *)

(* A load from a storage state in an index, as a gather's index is once ranged,
   is opaque to index arithmetic, which needs only its bounds: each is held as a
   parameter of its bounds while the index moves. Rewritten in place, the load
   would have the stores its state is ordered after rebuilt, a second definition
   of the storage. *)
let move_index in_shape m idxs =
  let load u =
    op u = Op.Index && shape_opt u = Some [] && op (base (nth u 0)) = Op.After
  in
  match
    List.filter load
      (toposort ~calls:Skip ~gate:(fun u -> op u <> Op.After) (sink idxs))
  with
  | [] -> Indexing.apply_movement_op in_shape m idxs
  | loads ->
      let held =
        List.mapi
          (fun i u ->
            ( u,
              param
                ~vmin_vmax:(vmin u, vmax u)
                ~addrspace:Dtype.Alu (-2 - i) (dtype u) ))
          loads
      in
      let moved =
        Indexing.apply_movement_op in_shape m
          (src (substitute ~calls:Skip ~pass:Fixed_point (sink idxs) held))
      in
      src
        (substitute ~calls:Skip ~pass:Fixed_point (sink moved)
           (List.map (fun (u, p) -> (p, u)) held))

let mop_index r idx =
  let idxs = List.tl (src idx) and s = shape (nth r 0) in
  let n = List.length idxs in
  if n = ndim r then Some (index (nth r 0) (move_index s (marg r) idxs))
  else if op r <> Op.Reshape then None
  else
    let rest = List.drop n (shape r) in
    let src_prefix = List.length s - List.length rest in
    if src_prefix < 0 || not (equal_shape (List.drop src_prefix s) rest) then
      None
    else if src_prefix = 0 then Some (nth r 0)
    else
      let reshape = Reshape (List.take n (shape r)) in
      let ret =
        index (nth r 0) (move_index (List.take src_prefix s) reshape idxs)
      in
      if equal_shape (shape ret) (shape idx) then Some ret else None

(* The movement rules, with [index] for an index of a movement. *)
let mops index =
  let movement = Upat.v ~op:Op.Set.movement ~name:"r" () in
  let zero = Upat.O.int 0 in
  Pattern_matcher.v
    (fun () -> [
      rule (Upat.f movement Op.Index ~allow_any_len:true ~name:"idx") (fun m ->
          index (m "r") (m "idx"));
      (* A store through a padded view gates the destination index, not its
         loaded value. *)
      rule
        (Upat.store
           (Upat.where (Upat.var "a")
              (Upat.where (Upat.var "b") (Upat.var "dst") zero)
              (Upat.var "pad"))
           [ Upat.var "val" ])
        (fun m ->
          if not (is_invalid (m "pad")) then None
          else
            let gate = bitwise_and (m "a") (m "b") in
            Some (store (where gate (m "dst") (int 0)) (m "val")));
      rule
        (Upat.store
           (Upat.where (Upat.var "cond")
              (Upat.index (Upat.var "buf") [ Upat.var "idx" ])
              zero)
           [ Upat.var "val" ])
        (fun m ->
          let idx = valid (m "idx") (m "cond") in
          Some (store (Ops.index (m "buf") [ idx ]) (m "val")));
      (* Movements and indices of storage move after the effects they are
         ordered after. *)
      rule
        (Upat.after ~name:"a" ~allow_any_len:true
           (Upat.v
              ~op:(Op.Set.union Op.Set.movement (ops [ Op.Index ]))
              ~name:"r" ())
           [])
        (fun m ->
          let r = m "r" and a = m "a" in
          if addrspace r = Some Dtype.Alu then None
          else
            let a = replace a ~src:(nth r 0 :: List.tl (src a)) in
            Some (v (op r) ~src:(a :: List.tl (src r)) ~arg:(arg r)));
      (* An index of a value keeps the value's after outside it. *)
      rule
        (Upat.op Op.Index ~allow_any_len:true ~name:"x"
           ~src:[ Upat.op Op.After ~allow_any_len:true ~name:"a" ])
        (fun m ->
          let a = m "a" and x = m "x" in
          if addrspace a <> Some Dtype.Alu then None
          else
            Some (after (Ops.index (nth a 0) (List.tl (src x))) (List.tl (src a))));
      rule (Upat.end_ ~name:"a" ~allow_any_len:true movement []) (fun m ->
          let a = m "a" in
          Some (replace a ~src:(nth (m "r") 0 :: List.tl (src a))));
    ])

let pm_mops = mops mop_index

(* In a tensor graph, an index with a shape is a gather's, which would multiply
   its shape if it moved through the movement: the gather is left whole. *)
let tensor_mop r idx = if Indexing.is_gather idx then None else mop_index r idx
let pm_tensor_mops = mops tensor_mop

(* Cleanups *)

let pp_shape ppf s =
  Format.fprintf ppf "(%a%s)"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
       Sint.pp)
    s
    (if List.length s = 1 then "," else "")

(* Stores into broadcasts *)

(* Whether [u] views an expand that stretches an axis of one element to more:
   several of its elements are then one cell, which a store through [u] would
   write once per element, in no order. In [row[c] = y[4r + c]], [y]'s four
   rows land on [row]. An expand that only adds axes of one, or empties one,
   stretches none. An index of a view keeps its axes after those it
   selects. *)
let rec broadcast u =
  match op u with
  | Op.Expand ->
      let to_ = shape u and from = shape (nth u 0) in
      let added = List.length to_ - List.length from in
      let ones = List.init added (fun _ -> Int 1) in
      let stretched f t =
        Sint.equal f (Int 1)
        && not (Sint.equal t (Int 1) || Sint.equal t (Int 0))
      in
      List.exists2 stretched (ones @ from) to_ || broadcast (nth u 0)
  | Op.Index | Op.Bitcast | Op.After | Op.Detach | Op.Contiguous_backward ->
      broadcast (nth u 0)
  | o when Op.Set.mem o Op.Set.movement -> broadcast (nth u 0)
  | _ -> false

let refuse_broadcast_store dest =
  if broadcast dest then
    invalid_arg
      (Format.asprintf
         "a store into a broadcast of shape %a writes several elements into \
          one cell; store into the storage it views"
         pp_shape (shape dest));
  None

(* A large reduction over few outputs is split into two kernels, to turn some of
   the reduction into outputs. The split axis moves last, so the second
   reduction reads it with the most locality. The first reduction's output is
   capped at 2^22 elements, enough to occupy a device with locals and upcasts; a
   split of 256 at most leaves a negligible second reduction, and one of 8 at
   least is worth a kernel. *)
let split_reduceop reduce x =
  let concrete shape =
    List.fold_right
      (fun s acc ->
        match (s, acc) with Int n, Some l -> Some (n :: l) | _ -> None)
      shape (Some [])
  in
  match (arg reduce, concrete (shape x), concrete (shape reduce)) with
  | Reduce { op = reduce_op; num_axes }, Some xs, Some rs -> (
      let out = Helpers.prod rs in
      if
        out = 0
        || (not (Setting.value Setting.split_reduceop))
        || Helpers.prod (List.filteri (fun i _ -> i < num_axes) xs)
           < Setting.value Setting.reduceop_split_threshold
      then None
      else
        (* An axis is expanded if its range does not reach the index into the
           base. *)
        let rngs =
          List.mapi (fun i s -> range ~axis_type:Weak (Int s) [ i ]) xs
        in
        let rec through idxs u =
          if Op.Set.mem (op u) Op.Set.movement then
            through
              (Indexing.apply_movement_op (shape (nth u 0)) (marg u) idxs)
              (nth u 0)
          else idxs
        in
        let reached = ranges (sink (through rngs x)) in
        let size = Setting.value Setting.reduceop_split_size in
        let limit = min 256 ((1 lsl size) / out) in
        let divisors = List.init (max 0 (limit - 7)) (fun k -> limit - k) in
        let candidates =
          List.concat_map
            (fun i ->
              List.filter_map
                (fun d ->
                  if
                    List.nth xs i mod d = 0
                    && Nodes.mem (List.nth rngs i) reached
                  then Some (i, d)
                  else None)
                divisors)
            (List.init num_axes Fun.id)
        in
        match candidates with
        | [] -> None
        | (dim, divisor) :: _ ->
            let order = List.init (List.length xs + 1) Fun.id in
            let splitted =
              permute
                (unflatten x dim
                   [ Int divisor; Int (List.nth xs dim / divisor) ])
                (List.filter (( <> ) dim) order @ [ dim ])
            in
            if Setting.value Setting.debug >= 3 then
              Format.printf "split %d: %a -> %a -> %a@." divisor pp_shape
                (shape x) pp_shape (shape splitted) pp_shape (shape reduce);
            let first =
              contiguous (rop splitted reduce_op (List.init num_axes Fun.id))
            in
            Some (rop first reduce_op [ ndim reduce ]))
  | _ -> None

let resolve_function c =
  if not (is_inline_call c) then None
  else
    let nodes = toposort ~calls:Skip (body c) in
    (* Input and output parameters both bind to explicit arguments by slot;
       unused arguments are allowed. *)
    let args = Array.of_list (List.tl (src c)) in
    (* A parameter holds flat storage of its greatest size, and its logical
       shape is a view on top: it binds to its argument viewed as such flat
       storage, so the views on the parameter stay valid. *)
    let flat_storage a =
      let shp = max_shape a in
      let a =
        let from_start (s, _) = Sint.equal s (Int 0) in
        match op a with
        | Op.Shrink
          when equal_shape (shape (nth a 0)) (ints shp)
               &&
               match marg a with
               | Shrink b -> List.for_all from_start b
               | _ -> false ->
            nth a 0
        | _ -> a
      in
      let n = Helpers.prod shp in
      ( n,
        if equal_shape (shape a) [ Int n ] then a
        else reshape (pad_to a (List.map (fun s -> Some (Int s)) shp)) [ Int n ]
      )
    in
    let bind p =
      let pa = param_arg_of p in
      let a = args.(pa.slot) in
      let bound =
        match shape p with
        | [ Int size ] ->
            let n, flat = flat_storage a in
            if size <> n then
              invalid_arg
                (Format.asprintf "argument %d of shape %a is not of size %d"
                   pa.slot pp_shape (shape a) size);
            flat
        | _ ->
            if shape a <> [] then
              invalid_arg
                (Format.asprintf "argument %d has the shape %a, not a scalar's"
                   pa.slot pp_shape (shape a));
            a
      in
      if not (Dtype.equal (dtype p) (dtype a)) then
        invalid_arg
          (Format.asprintf "argument %d is a %a, not a %a" pa.slot Dtype.pp
             (dtype a) Dtype.pp (dtype p));
      (p, bound)
    in
    (* Inlining removes the call's scope, so its local storage needs fresh
       identities. *)
    let rename b =
      (b, replace b ~arg:(Param { (param_arg_of b) with slot = unique_num () }))
    in
    let subs =
      List.filter_map
        (fun p ->
          match op p with
          | Op.Param when (param_arg_of p).slot >= 0 -> Some (bind p)
          | Op.Alloc -> Some (rename p)
          | _ -> None)
        nodes
    in
    Some (substitute ~calls:Skip ~pass:Once (body c) subs)

let uint_of_bytes = function
  | 1 -> Dtype.Uint8
  | 2 -> Dtype.Uint16
  | 4 -> Dtype.Uint32
  | 8 -> Dtype.Uint64
  | n -> invalid_arg (Printf.sprintf "no unsigned integer of %d bytes" n)

(* A bitcast that changes the element size is shifts on unsigned integers. *)
let expand_bitcast bc =
  let x = nth bc 0 in
  let ns = Dtype.itemsize (dtype bc) and os = Dtype.itemsize (dtype x) in
  if ns = os || on_disk x then None
  else
    let new_uint = uint_of_bytes ns in
    let tmp = bitcast x (uint_of_bytes os) in
    if (ns > os) [@mutate off "equal sizes are left alone above"] then
      let rate = ns / os in
      let lead = List.take (ndim x - 1) (shape x)
      and last = List.nth (shape x) (ndim x - 1) in
      let tmp = reshape tmp (lead @ [ Sint.(last // Int rate); Int rate ]) in
      let part i =
        let slice =
          List.init (ndim tmp) (fun a ->
              if a = ndim tmp - 1 then Some (Int i, Int (i + 1)) else None)
        in
        shl (cast (shrink tmp slice) new_uint) (int (8 * i * os))
      in
      let parts = List.init rate part in
      Some
        (bitcast
           (squeeze ~axis:(-1) (usum (List.hd parts) (List.tl parts)))
           (dtype bc))
    else
      let parts = List.init (os / ns) (fun i -> shr tmp (int (8 * i * ns))) in
      Some
        (bitcast
           (cast (flatten ~start:(-2) (stack ~axis:(-1) parts)) new_uint)
           (dtype bc))

(* Contiguous views *)

let strides_for_shape shape =
  fst
    (List.fold_right
       (fun s (strides, p) ->
         ((if Sint.equal s (Int 1) then Int 0 else p) :: strides, Sint.(p * s)))
       shape ([], Int 1))

(* The effects storage is ordered after decide nothing of where its elements
   lie: the rewrite of a view's index stops at them, a void node, and leaves the
   graphs they compute, which hold gathers and arithmetic on tensors that the
   index rules do not apply to, as they are. *)
let pm_stop_at_effects =
  Pattern_matcher.v
    (fun () -> [
      rule (Upat.v ~op:Op.Set.all ~dtype:[ Dtype.Void ] ()) (fun _ ->
          raise Bottom_up_gate);
    ])

(* The index of [u]'s elements rewritten through its movements ends at the node
   they are read from; the elements are a run of that node when its flat index
   is the range plus a constant. Where the index reads bound variables, whose
   values a call knows, the elements are a run when they are one at those
   values, from the index at the run's first element, which reads them. A
   bitcast's run is also a run of its source when it starts and ends on whole
   elements of the source at every value of the variables it reads. *)
let rec contiguous_view u =
  let numel = numel u in
  let rng = range ~axis_type:Weak numel [ 0 ] in
  let out =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() (index (flatten u) [ rng ])
      (Around_sources
         {
           before = pm_stop_at_effects;
           after = Pattern_matcher.concat [ pm_mops; Shape.symbolic ];
         })
  in
  if op out <> Op.Index then None
  else
    let b = nth out 0 and idxs = List.tl (src out) in
    if List.length idxs <> ndim b then None
    else
      let flat =
        List.fold_left2
          (fun acc i s -> add acc (mul i (sint_to_uop s)))
          (int 0) idxs
          (strides_for_shape (shape b))
      in
      (* Checked at the call's values only: a movement adds a bound variable to
         its index only as an offset, under constant strides and div/mod, so a
         run at one value is a run at all. A variable meeting a range in a
         div/mod of non-constant size would need the check at every value. In
         a call's body, a scalar parameter stands for a bound variable, checked
         at its least value. *)
      let scalar u = op u = Op.Param && addrspace u = Some Dtype.Alu in
      let bound =
        List.filter
          (fun u -> is_bound_var u || scalar u)
          (toposort ~calls:Skip flat)
      in
      let value v =
        let at = if scalar v then vmin v else snd (unbind v) in
        (v, int (Dtype.Value.to_int at))
      in
      let valued =
        substitute ~calls:Skip ~pass:Once flat (List.map value bound)
      in
      (* The run of a bitcast from [first] as a run of its source, when it
         starts and ends on whole elements of the source at every value of
         the variables [first] reads, and [cut] otherwise. *)
      let of_source ~cut first =
        let osz = element_size b and isz = element_size (nth b 0) in
        let simplified s = ssimplify (sint_to_uop s) in
        let bytes = Sint.(first * Int osz) and len = Sint.(numel * Int osz) in
        let whole s = Sint.equal (simplified Sint.(s % Int isz)) (Int 0) in
        if not (whole bytes && whole len) then cut
        else
          let start = Sint.(bytes // Int isz) in
          let count = simplified Sint.(len // Int isz) in
          contiguous_view
            (shrink (flatten (nth b 0)) [ Some (start, Sint.(start + count)) ])
      in
      match ssimplify (sub valued rng) with
      | Sym _ -> None
      | Int _ when bound <> [] ->
          let first =
            ssimplify (substitute ~calls:Skip ~pass:Once flat [ (rng, int 0) ])
          in
          if op b = Op.Bitcast then of_source ~cut:None first
          else Some (b, first)
      | Int offset when op b <> Op.Bitcast -> Some (b, Int offset)
      | Int offset -> of_source ~cut:(Some (b, Int offset)) (Int offset)

(* A store through movements and bitcasts of storage, or of a state of it,
   whose elements are a run of whole elements of the storage stores the
   value's bits into that run: indexed through a bitcast that changes the
   element size, the target would be shifts of loaded values. A store into a
   bitcast of the storage itself is left to the rule that bitcasts its
   value. *)
let store_into_run dst x =
  let rec through_bitcast u =
    op u = Op.Bitcast
    || (Op.Set.mem (op u) Op.Set.movement && through_bitcast (nth u 0))
  in
  let of_storage u = has_buffer_identity ~after_ok:true u in
  if not (through_bitcast dst) then None
  else if op dst = Op.Bitcast && of_storage (nth dst 0) then None
  else
    match contiguous_view dst with
    | Some (b, first) when of_storage b ->
        let n =
          Sint.(numel dst * Int (element_size dst) // Int (element_size b))
        in
        let run = shrink (flatten b) [ Some (first, Sint.(first + n)) ] in
        Some (store run (bitcast (flatten x) (dtype b)))
    | _ -> None

(* Whether [u] is storage, or a run of its elements ({!contiguous_view}) that
   shrinks and reshapes view: a copy kernel then indexes the storage at its
   range plus the run's first element, which {!Schedule} lowers to a copy. *)
let rec storage_run u =
  has_buffer_identity ~after_ok:true u
  ||
  match op u with
  | Op.Shrink | Op.Reshape ->
      storage_run (nth u 0)
      && (match contiguous_view u with Some (_, Int _) -> true | _ -> false)
  | _ -> false

(* Arguments of calls compiled on their own *)

let compiled_on_its_own = ops Op.[ Sink; Program; Linear ]

let viewing = Op.Set.union Op.Set.movement (ops Op.[ Bitcast; After ])

(* Whether [a] is movements, bitcasts and states of storage. *)
let rec views_storage a =
  has_buffer_identity ~after_ok:true a
  || (Op.Set.mem (op a) viewing && views_storage (nth a 0))

(* The storage whose elements, from the first, [a] is a run of, of [a]'s
   type, if any: the call writes through it. *)
let run_from_start a =
  if has_buffer_identity ~after_ok:true a || not (views_storage a) then None
  else
    match contiguous_view a with
    | Some (b, Int 0)
      when has_buffer_identity ~after_ok:true b
           && Dtype.equal (dtype b) (dtype a) ->
        Some b
    | _ -> None

(* Whether [u] moves with a serial loop: a range it varies with is one. *)
let moves_with_loop u =
  List.exists
    (fun r -> Axis_type.equal (axis_type r) Axis_type.Loop)
    (Nodes.to_list (ranges u))

(* Storage, or a state of it, that a call only reads outside loops, and that
   starts elsewhere than its parameter assumes, stored into new storage, which
   starts on a 16-byte boundary: rune's jit makes storage at its inputs' phases,
   and a kernel's placeholder takes the boundary. Storage the call also writes
   stays: {!Schedule} copies it, in the loop where the call runs in one, and a
   copy made here would be made once. *)
let copy_off_start c i a =
  let reads =
    match Shape.written_params (body c) with
    | Some w -> not (List.mem i w)
    | None -> false
  in
  match Shape.param_start (body c) i with
  | Some param
    when reads
         && (not (moves_with_loop a))
         && has_buffer_identity ~after_ok:true a
         && (not (fits_start ~param (storage_phase a)))
         && fits_start ~param allocated_start
         && not (reads_storage_written c i) ->
      let buf =
        alloc ?device:(device a) (shape a) (dtype a)
      in
      after buf [ store buf a ]
  | _ -> a

(* A call compiled on its own takes flat storage at each argument
   ({!Indexing.run_rangeify}): a run of storage from its first element is passed
   as the storage, and storage it only reads off its parameter's start is
   copied. *)
let call_storage c =
  if is_inline_call c || not (Op.Set.mem (op (body c)) compiled_on_its_own) then
    None
  else
    let args = List.tl (src c) in
    let args' =
      List.mapi
        (fun i a ->
          copy_off_start c i (Option.value (run_from_start a) ~default:a))
        args
    in
    if List.for_all2 ( == ) args args' then None
    else Some (replace c ~src:(nth c 0 :: args'))

(* Copies always cross devices, and read a whole buffer: SDMA engines cannot
   copy from an offset. *)
let copy_to_anon_store x copy =
  let x = pad_to x (List.map (fun s -> Some (Int s)) (max_shape x)) in
  let buf =
    alloc ?device:(device copy) [ Int (max_numel x) ] (dtype copy)
  in
  let buf = reshape buf (ints (max_shape x)) in
  shrink_to (after buf [ store buf x ]) (List.map Option.some (shape copy))

(* A stage in global memory stores each slice of its value at the coordinates
   of its ranges, into a buffer inside the call and not kept, as those of
   copies. *)
let stage_to_anon_store x stg =
  match arg stg with
  | Bufferize { addrspace; _ } when addrspace <> Dtype.Global -> None
  | _ ->
      let buf =
        alloc ?device:(device x) (ints (max_shape stg)) (dtype stg)
      in
      let view = shrink_to buf (List.map Option.some (shape stg)) in
      let rngs = List.tl (src stg) in
      let dst = if rngs = [] then view else index view rngs in
      Some (after view [ end_ (store dst x) rngs ])

(* A copy across devices reads a whole buffer. *)
let materialize_cross_device_src dest src =
  if
    Option.is_none (device src)
    || equal_device_of dest src
    || has_buffer_identity ~after_ok:true src
  then None
  else Some (store dest (contiguous src))

(* Whether [u] is a stage no pass removes. *)
let kept_stage u =
  match (op u, arg u) with
  | Op.Stage, Bufferize { keep = Whole; _ } -> true
  | _ -> false

let pm_inline_calls =
  Pattern_matcher.v
    (fun () -> [
      rule (Upat.op Op.Call ~name:"c") (fun m -> resolve_function (m "c"));
    ])

(* A loop runs its call once a trip, each trip reading what the trips before
   wrote. Inlined, the call's stores would each run once over every trip, so
   the call stays a call, compiled on its own. An end of an end of a call is
   one end over both ranges, the outer end's first. *)
let pm_loop_calls =
  let loop = Upat.v ~op:(ops Op.[ End; Backedge ]) ~name:"e" () in
  Pattern_matcher.v
    (fun () -> [
      rule loop (fun m ->
          let e = m "e" in
          let x = nth e 0 and rest = List.tl (src e) in
          match (op e, op x, arg x) with
          | Op.End, Op.End, _ when op (nth x 0) = Op.Call ->
              Some (end_ (nth x 0) (rest @ List.tl (src x)))
          | _, Op.Call, Call info when is_inline_call x ->
              let x = replace x ~arg:(Call { info with precompile = true }) in
              Some (replace e ~src:(x :: rest))
          | _ -> None);
    ])

let pm_disk_copy =
  let movement =
    Upat.v
      ~op:(Op.Set.union Op.Set.movement (Op.Set.of_list [ Op.Bitcast ]))
      ~name:"x" ()
  in
  Pattern_matcher.v
    (fun () -> [
      (* A disk copy reads its source without materialising its movements and
         bitcasts. *)
      rule
        (Upat.f (Upat.f movement Op.Stage) Op.Copy ~name:"copy")
        (fun m ->
          let x = m "x" in
          if on_disk x then Some (replace (m "copy") ~src:[ x ]) else None);
      (* The movements and bitcasts move to the copy's result: views exposed
         here are no longer normalised into parameters, and a shrink or reshape
         left behind would store a temporary on the disk. *)
      rule (Upat.f movement Op.Copy ~name:"copy") (fun m ->
          let x = m "x" in
          if not (on_disk x) then None
          else
            let copy = replace (m "copy") ~src:[ nth x 0 ] in
            Some (replace x ~src:(copy :: List.tl (src x))));
    ])

let earliest_rewrites =
  let var = Upat.var in
  let buf_store_src = Upat.store (var "buf") [ var "src" ] in
  Pattern_matcher.append Shape.mop_cleanup
    (Pattern_matcher.v
       (fun () -> [
         rule
           (Upat.op Op.Store ~allow_any_len:true ~src:[ var "dest"; Upat.wild ])
           (fun m -> refuse_broadcast_store (m "dest"));
         rule (Upat.op Op.Call ~name:"c") (fun m -> call_storage (m "c"));
         rule
           (Upat.op Op.Reduce ~name:"reduce" ~src:[ var "x" ])
           (fun m -> split_reduceop (m "reduce") (m "x"));
         rule
           (Upat.v ~op:(ops Op.[ Detach; Contiguous_backward ]) ~name:"x" ())
           (fun m -> Some (nth (m "x") 0));
         (* Copies *)
         (* A copy to the device of its source is a no-op: a stage
            materialises on the same device. *)
         rule
           (Upat.op Op.Copy ~name:"copy" ~allow_any_len:true ~src:[ var "x" ])
           (fun m ->
             let x = m "x" in
             if equal_device_of x (m "copy") then Some x else None);
         (* A store to storage on another device is a copy, so a copy stored
            into storage on its device, or into a run of its elements, is the
            store: the copy lands in place, where a copy into storage of its
            own and a store of that would hold both. *)
         rule
           (Upat.op Op.Store
              ~src:
                [
                  var "dst";
                  Upat.op Op.Copy ~name:"cpy" ~allow_any_len:true
                    ~src:[ var "x" ];
                ])
           (fun m ->
             let dst = m "dst" in
             if equal_device_of dst (m "cpy") && storage_run dst then
               Some (store dst (m "x"))
             else None);
         (* Any other copy stores into new call-local storage on its device. *)
         rule
           (Upat.op Op.Copy ~name:"copy" ~allow_any_len:true ~src:[ var "x" ])
           (fun m -> Some (copy_to_anon_store (m "x") (m "copy")));
         (* Stages *)
         (* A stage of a materialised value, or of a copy, which materialises
            itself, is a no-op. *)
         rule
           (Upat.op Op.Stage ~name:"stg" ~src:[ var "x" ])
           (fun m ->
             let x = m "x" in
             if
               (not (kept_stage (m "stg")))
               && (has_buffer_identity ~after_ok:true x || op x = Op.Copy)
             then Some x
             else None);
         (* Any other stage in global memory stores into new call-local
            storage on the same device. *)
         rule
           (Upat.op Op.Stage ~name:"stg" ~allow_any_len:true ~src:[ var "x" ])
           (fun m -> stage_to_anon_store (m "x") (m "stg"));
         rule
           (Upat.op Op.Store
              ~src:
                [
                  Upat.op Op.Reshape ~allow_any_len:true ~src:[ var "dst" ];
                  Upat.op Op.Reshape ~allow_any_len:true ~src:[ var "src" ];
                ])
           (fun m ->
             let dst = m "dst" and src = m "src" in
             if equal_shape (shape dst) (shape src) then Some (store dst src)
             else None);
         (* Stores *)
         (* The store across devices is the copy: its value materialises on
            its own device first. *)
         rule
           (Upat.op Op.Store ~src:[ var "dest"; var "src" ])
           (fun m -> materialize_cross_device_src (m "dest") (m "src"));
         (* A store of a storage's own contents into itself. *)
         rule
           (Upat.after (var "buf")
              [
                Upat.store (var "buf")
                  [ Upat.after ~name:"a1" (var "buf") [ buf_store_src ] ];
              ])
           (fun m -> Some (m "a1"));
         rule
           (Upat.op Op.Store ~src:[ var "dst"; var "src" ])
           (fun m -> store_into_run (m "dst") (m "src"));
         (* A store into a bitcast of storage bitcasts the value instead. *)
         rule
           (Upat.op Op.Store
              ~src:[ Upat.op Op.Bitcast ~src:[ var "target" ]; var "src" ])
           (fun m ->
             let target = m "target" in
             Some (store target (bitcast (m "src") (dtype target))));
         rule (Upat.op Op.Bitcast ~name:"bc") (fun m -> expand_bitcast (m "bc"));
         (* Empty values *)
         rule
           (Upat.op Op.Reduce ~name:"reduce" ~src:[ var "x" ])
           (fun m ->
             let reduce = m "reduce" in
             match arg reduce with
             | Reduce { op; _ }
               when is_empty_shape (m "x") && not (is_empty_shape reduce) ->
                 Some (const_like reduce (identity_element op (dtype reduce)))
             | _ -> None);
         rule
           (Upat.v ~op:(Op.Set.diff Op.Set.all (ops [ Op.Sink ])) ~name:"x" ())
           (fun m ->
             let x = m "x" in
             match shape_opt x with
             | Some s when List.exists (Sint.equal (Int 0)) s ->
                 Some (const_like x (`Int Bigint.zero))
             | _ -> None);
         (* Effects lose their movements. *)
         rule (Upat.op Op.Sink ~name:"s") (fun m ->
             let s = m "s" in
             let kept = List.filter (fun u -> op u <> Op.Noop) (src s) in
             Some (replace s ~src:(List.map walk_mop kept)));
         rule (Upat.op Op.After ~name:"s") (fun m ->
             let s = m "s" in
             let kept =
               List.filter (fun u -> op u <> Op.Noop) (List.tl (src s))
             in
             Some (replace s ~src:(nth s 0 :: List.map walk_mop kept)));
       ]))

(* Sequencing

   A graph's effects run once each, ordered by the states they name
   ({!Ops.after}). [sequence] states that order in the graph: each storage's
   states become one chain, each state over the one before, listing the one
   effect (or the stores of one disjoint write) that makes it; every read
   names the state of the chain it reads, and every effect the state it
   writes over. Joins and writes over one state disappear: the schedule
   orders a chain. *)

let is_storage u =
  match op u with
  | Op.Buffer | Op.Alloc -> true
  | Op.Param -> addrspace u <> Some Dtype.Alu
  | _ -> false

let rec state_node u =
  if is_storage u then Some u
  else
    match op u with
    | Op.After when is_join u -> state_node (nth u 0)
    | Op.After -> Some u
    | Op.Bitcast | Op.Index -> state_node (nth u 0)
    | o when Op.Set.mem o Op.Set.movement -> state_node (nth u 0)
    | _ -> None

(* [s], movements of a bitcast of one element size, as the same movements
   of the bitcast's source. *)
let rec same_size_bitcast s =
  match op s with
  | Op.Bitcast when Dtype.itemsize (dtype s) = Dtype.itemsize (dtype (nth s 0))
    ->
      Some (nth s 0)
  | o when Op.Set.mem o Op.Set.movement ->
      Option.map
        (fun x -> replace s ~src:(x :: List.tl (src s)))
        (same_size_bitcast (nth s 0))
  | _ -> None

(* Storage and effects as a refusal names them. *)
let pp_storage ppf s =
  Format.fprintf ppf "the storage in slot %d" (param_arg_of s).slot

let describe e =
  match (op e, arg e) with
  | Op.Store, _ -> (
      let dst = nth e 0 in
      match contiguous_view dst with
      | Some (b, Int lo) when Dtype.equal (dtype b) (dtype dst) ->
          Printf.sprintf "the store into elements %d to %d" lo
            (lo + max_numel dst - 1)
      | _ -> Format.asprintf "a store of shape %a" pp_shape (shape dst))
  | Op.Call, Call { name = Some n; _ } -> "a call of " ^ n
  | Op.Call, _ -> "a call"
  | _ -> "a loop"

(* An inlined body lists its effects in a sink. *)
let rec listed_effects d =
  if op d = Op.Sink then List.concat_map listed_effects (src d) else [ d ]

(* Whether the effects of [sink], of nodes [nodes], are one chain per storage
   already, which [sequence] leaves as it is: each is a store into storage, or
   a view of it, listed alone by an After over that storage, the only write
   of the storage that [key] names; no join, no value ordered after effects,
   and the sink, or a sink it lists, lists no effect. *)
let chained ~key nodes sink =
  let written = Tbl.create 4 in
  let rec viewed u =
    if Op.Set.mem (op u) Op.Set.movement then viewed (nth u 0) else u
  in
  let plain u =
    match (op u, src u) with
    | Op.After, [ x; d ] -> (
        op d = Op.Store && is_storage x
        && viewed (nth d 0) == x
        &&
        match key u with
        | Some s when not (Tbl.mem written s) ->
            Tbl.replace written s ();
            true
        | _ -> false)
    | (Op.After | Op.Call | Op.End | Op.Backedge | Op.Linear), _ -> false
    | _ -> true
  in
  let runs x = Op.Set.mem (op x) (ops Op.[ Store; Call; End; Backedge; Linear ]) in
  List.for_all plain nodes
  && not (List.exists runs (List.concat_map listed_effects (src sink)))

let sequence sink =
  let nodes = toposort ~calls:Skip sink in
  let by_slot = Hashtbl.create 1 in
  let key u =
    match state_node u with
    | None -> None
    | Some st ->
        let s = whole (storage_base st) in
        if not (is_storage s) then None
        else if op s = Op.Param then (
          let slot = (param_arg_of s).slot in
          match Hashtbl.find_opt by_slot slot with
          | Some c -> Some c
          | None ->
              Hashtbl.replace by_slot slot s;
              Some s)
        else Some s
  in
  if chained ~key nodes sink then sink
  else
  (* The nodes a loop runs, which are no effects of their own. *)
  let inner = Tbl.create 4 in
  let rec mark_inner u =
    Tbl.replace inner u ();
    match op u with
    | Op.Linear | Op.End | Op.Backedge -> mark_inner (nth u 0);
        if op u = Op.Linear then List.iter mark_inner (src u)
    | _ -> ()
  in
  List.iter
    (fun u -> if op u = Op.End || op u = Op.Backedge then mark_inner (nth u 0))
    nodes;
  let is_effect u =
    (not (Tbl.mem inner u))
    && match op u with
       | Op.Store | Op.Call -> true
       | Op.End | Op.Backedge -> (
           match op (nth u 0) with
           | Op.Call | Op.Linear | Op.Store | Op.End -> true
           | _ -> false)
       | _ -> false
  in
  (* Each listing of an effect: the storage of the After, the After, and the
     effects listed before it there, which make later states the listing
     names; and the marked After listing it. An effect runs after the state
     each After listing it is over, where it writes that After's storage:
     [after x1 [k]] puts [k] over [x1], wherever the walk first reaches [k].
     A state that already holds [k] (a listing again on a later state) orders
     nothing. *)
  let over = Tbl.create 4 and marked = Tbl.create 4 in
  List.iter
    (fun u ->
      if op u = Op.After && not (is_join u) then
        let listed = List.concat_map listed_effects (List.tl (src u)) in
        if is_disjoint u then List.iter (fun d -> Tbl.replace marked d u) listed;
        match key u with
        | None -> ()
        | Some s ->
            ignore
              (List.fold_left
                 (fun earlier d ->
                   Tbl.replace over d
                     ((s, u, earlier)
                     :: Option.value ~default:[] (Tbl.find_opt over d));
                   d :: earlier)
                 [] listed))
    nodes;
  (* The accesses of an effect: each call it runs, or its store. *)
  let rec parts u =
    match op u with
    | Op.Store | Op.Call -> [ u ]
    | Op.Linear -> List.concat_map parts (src u)
    | Op.End | Op.Backedge -> parts (nth u 0)
    | _ -> []
  in
  let written_args c =
    match op c with
    | Op.Store -> Some [ nth c 0 ]
    | _ -> (
        match Shape.written_params (body c) with
        | None -> None
        | Some w ->
            Some
              (List.filteri
                 (fun i _ -> List.mem i w)
                 (src_without_body c)))
  in
  (* An effect writes its destinations, or, where its body shows none, the
     storage of the Afters that list it. *)
  let writes e =
    let from_parts =
      List.concat_map
        (fun c ->
          Option.fold ~none:[] ~some:(List.filter_map key) (written_args c))
        (parts e)
    in
    if from_parts <> [] then dedup_nodes from_parts
    else
      dedup_nodes
        (List.map
           (fun (s, _, _) -> s)
           (Option.value ~default:[] (Tbl.find_opt over e)))
  in
  (* Whether [a] runs after [b]: [b] is among what [a] reads or waits for. *)
  let depends a b = Nodes.mem b (backward_slice ~calls:Skip a) in
  (* Whether [p], a state, has the effect [e] in its chain: a listing of [e]
     again on a later state, which orders nothing. *)
  let rec in_chain p e =
    match state_node p with
    | Some x when op x = Op.After ->
        List.memq e (List.concat_map listed_effects (List.tl (src x)))
        || in_chain (nth x 0) e
    | _ -> false
  in
  (* The walk: sources in order, an effect after the state each listing it is
     over and the effects listed before its first place there, so that no item
     of the sink, group or value reached first decides its place. *)
  let nodes =
    let mark = Tbl.create 4 and out = ref [] in
    let rec visit u =
      match Tbl.find_opt mark u with
      | Some true -> ()
      | Some false ->
          invalid_arg "an effect is listed over a state that needs it"
      | None ->
          Tbl.replace mark u false;
          (if (not (Tbl.mem inner u)) && is_effect u then
             let w = writes u in
             List.iter
               (fun (s, listing, earlier) ->
                 let p = nth listing 0 in
                 if
                   List.memq s w
                   && (not (in_chain p u))
                   && not (List.memq u earlier)
                 then (
                   visit p;
                   List.iter
                     (fun d ->
                       if Tbl.find_opt mark d <> Some true && List.memq s (writes d)
                       then visit d)
                     (List.rev earlier)))
               (List.rev (Option.value ~default:[] (Tbl.find_opt over u))));
          List.iter visit (if op u = Op.Call then List.tl (src u) else src u);
          Tbl.replace mark u true;
          out := u :: !out
    in
    visit sink;
    List.rev !out
  in
  let bare = ref [] in
  let map = Tbl.create 4 in
  let m x = Option.value (Tbl.find_opt map x) ~default:x in
  let current = Tbl.create 4 in
  let order = ref [] in
  let cur s = Option.value (Tbl.find_opt current s) ~default:s in
  let opened = Tbl.create 4 in
  (* The states each effect makes, and the effect that makes each state. *)
  let made = Tbl.create 4 and maker = Tbl.create 4 in
  (* Each state's place in its chain. *)
  let stamp = Tbl.create 4 in
  (* [x], movements of a state, over the state [st] instead. *)
  let rec reroot x st =
    if is_storage x then
      (* A view of the storage, as a body's parameter of a run, stays. *)
      if shape_opt x = shape_opt st && Dtype.equal (dtype x) (dtype st) then st
      else x
    else
      match op x with
      | Op.After -> reroot (nth x 0) st
      | o when Op.Set.mem o Op.Set.movement || o = Op.Bitcast ->
          replace x ~src:(reroot (nth x 0) st :: List.tl (src x))
      | _ -> m x
  in
  let run_effect e =
    let w = writes e in
    let group s =
      match (Tbl.find_opt marked e, Tbl.find_opt opened s) with
      | Some u, Some (u', prev, es) when u == u' -> Some (u, prev, es)
      | _ -> None
    in
    let target s =
      match group s with Some (_, prev, _) -> prev | None -> cur s
    in
    (* Two writes over one state: [e] writes [s] over [target s], which each
       listing of [e] must name, through the state it is over, through an
       effect listed before [e], or through [e]'s written arguments, unless
       [e] reads what the write that made it left. A listing that names an
       older state otherwise lists [e] beside that write, and which of the two
       lands last would be the walk's choice. *)
    let names s t x =
      match state_node (m x) with
      | Some x -> (if is_storage x then s else x) == t
      | None -> false
    in
    let written_names s t c =
      match written_args c with
      | Some l ->
          List.exists
            (fun a ->
              match key a with Some k -> k == s && names s t a | None -> false)
            l
      | None -> false
    in
    let refuse s t =
      invalid_arg
        (Format.asprintf
           "two writes over one state of %a: %s and %s are each over it; list \
            one over the After of the other, or list both in one After"
           pp_storage s (describe e) (describe (Tbl.find maker t)))
    in
    let refuse_siblings (s, u, earlier) =
      let p = nth u 0 and t = target s in
      let made_it d =
        match Tbl.find_opt made d with
        | Some l -> List.exists (fun (s', st) -> s' == s && st == t) l
        | None -> false
      in
      if
        List.memq s w && group s = None
        && not
             (names s t p
             || List.exists made_it earlier
             || List.exists (written_names s t) (parts e)
             || in_chain p e
             || depends e (Tbl.find maker t))
      then refuse s t
    in
    let listings = Option.value ~default:[] (Tbl.find_opt over e) in
    List.iter refuse_siblings listings;
    (* A write no After of its storage lists is over the state its
       destination names, a sibling of any other write over that state. *)
    List.iter
      (fun s ->
        let t = target s in
        if
          t != s
          && (not (List.exists (fun (s', _, _) -> s' == s) listings))
          && not
               (List.exists (written_names s t) (parts e)
               || depends e (Tbl.find maker t))
        then refuse s t)
      w;
    (* A destination, or an argument a call writes, is the storage as the
       effect runs: the state its write is over. *)
    let access ~written a =
      match key a with
      | Some s when written && List.memq s w -> reroot a (target s)
      | _ -> m a
    in
    let rec build u =
      match op u with
      | Op.Store -> (
          let dst = access ~written:true (nth u 0) in
          let rest = List.map m (List.tl (src u)) in
          (* A store through a bitcast of a state, of one element size, stores
             the value's bits into the state. *)
          match (same_size_bitcast dst, rest) with
          | Some x, v :: gate when op v <> Op.Copy ->
              (* A weak value takes the view's type first: a bitcast reads
                 its operand at a stated width. *)
              let v =
                if Dtype.mem (dtype v) Dtype.weaks then cast v (dtype dst)
                else v
              in
              replace u ~src:(x :: bitcast v (dtype x) :: gate)
          | _ -> replace u ~src:(dst :: rest))
      | Op.Call ->
          let w =
            match Shape.written_params (body u) with Some w -> w | None -> []
          in
          replace u
            ~src:
              (body u
              :: List.mapi
                   (fun i a -> access ~written:(List.mem i w) a)
                   (src_without_body u))
      | Op.Linear -> replace u ~src:(List.map build (src u))
      | Op.End | Op.Backedge ->
          replace u ~src:(build (nth u 0) :: List.map m (List.tl (src u)))
      | _ -> m u
    in
    let e' = build e in
    Tbl.replace map e e';
    if w = [] then bare := e' :: !bare;
    List.iter
      (fun s ->
        if not (Tbl.mem current s) then order := s :: !order;
        let st =
          match group s with
          | Some (u, prev, es) ->
              let es = es @ [ e' ] in
              Tbl.replace opened s (u, prev, es);
              after_disjoint prev es
          | None -> (
              let prev = cur s in
              match Tbl.find_opt marked e with
              | Some u ->
                  Tbl.replace opened s (u, prev, [ e' ]);
                  after_disjoint prev [ e' ]
              | None ->
                  Tbl.remove opened s;
                  after prev [ e' ])
        in
        Tbl.replace current s st;
        Tbl.replace made e
          ((s, st) :: Option.value ~default:[] (Tbl.find_opt made e));
        Tbl.replace maker st e;
        Tbl.replace stamp st (Tbl.length stamp))
      w
  in
  List.iter
    (fun u ->
      if Tbl.mem inner u then ()
      else if is_effect u then run_effect u
      else
        match op u with
        | Op.After when is_join u -> Tbl.replace map u (m (nth u 0))
        | Op.After -> (
            (* The latest of the state under it, the states the writes it
               lists of its storage make, and the states of its storage the
               effects it lists read: they leave it no older. *)
            let stamped x =
              match state_node (m x) with
              | Some st when Tbl.mem stamp st -> Some st
              | _ -> None
            in
            let later best st =
              match best with
              | Some b when Tbl.find stamp b >= Tbl.find stamp st -> best
              | _ -> Some st
            in
            let listed = List.concat_map listed_effects (List.tl (src u)) in
            let latest s =
              let best =
                List.fold_left
                  (fun best d ->
                    match Option.bind (Tbl.find_opt made d) (List.assq_opt s) with
                    | Some st -> later best st
                    | None -> best)
                  (stamped (nth u 0)) listed
              in
              (* Only a state older than the chain's last can be passed by
                 one the effects read. *)
              if Option.fold ~none:(cur s == s) ~some:(( == ) (cur s)) best
              then best
              else
                let read x best =
                  if op x <> Op.After || is_join x then best
                  else
                    match (key x, stamped x) with
                    | Some k, Some st when k == s -> later best st
                    | _ -> best
                in
                List.fold_left
                  (fun best d -> Nodes.fold read (backward_slice ~calls:Skip d) best)
                  best listed
            in
            match Option.bind (key u) latest with
            | Some st -> Tbl.replace map u (reroot (nth u 0) st)
            | None -> Tbl.replace map u (m (nth u 0)))
        | _ when u == sink -> ()
        | _ ->
            (* A node none of whose sources changed stays. *)
            let s = src u in
            if List.exists (fun x -> Tbl.mem map x) s then
              let s' = List.map m s in
              if not (List.for_all2 ( == ) s s') then
                Tbl.replace map u (replace u ~src:s'))
    nodes;
  let ends = List.rev_map cur !order in
  let kept =
    List.filter_map
      (fun x ->
        if is_effect x || op x = Op.After || op x = Op.Store then None
        else Some (m x))
      (List.concat_map listed_effects (src sink))
  in
  (* A chain's last state that a kept source views is listed by it. *)
  let kept_views e =
    List.exists
      (fun k -> Option.fold ~none:false ~some:(( == ) e) (state_node k))
      kept
  in
  let ends = List.filter (fun e -> not (kept_views e)) ends in
  let src' = dedup_nodes (ends @ List.rev !bare @ kept) in
  if List.equal ( == ) src' (src sink) then sink else replace sink ~src:src'

let prepare_rangeify sink =
  let tsink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
      (forward_call_outputs sink)
      (Around_sources
         {
           before = pm_loop_calls;
           after =
             Pattern_matcher.concat
               [ pm_tensor_mops; pm_inline_calls; pm_disk_copy ];
         })
  in
  let tsink = sequence tsink in
  graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() tsink
    (Before_sources (Pattern_matcher.append pm_tensor_mops earliest_rewrites))
