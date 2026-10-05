(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let rule = Pattern_matcher.rule
let ops = Op.Set.of_list
let is_unshard u = op u = Op.Unshard
let peel u = if is_unshard u then nth u 0 else u
let count rng = Dtype.Value.to_int (vmax rng) + 1
let equal_shape s0 s1 = List.equal Sint.equal s0 s1
let truth = function Sint.Known b -> b | Sint.Cond u -> to_bool u

let unshard_as pairs x =
  unshard ~ranges:(List.map snd pairs) x (List.map fst pairs)

let reshard multi x = unshard_as (sharding multi) x

let equal_sharding s0 s1 =
  List.equal (fun (a0, r0) (a1, r1) -> a0 = a1 && r0 == r1) s0 s1

let sharded_axis u =
  match axis u with
  | Some a -> a
  | None -> invalid_arg "resharding needs a value sharded on one axis"

let is_multi u = match device u with Some (Multi _) -> true | _ -> false

let devices u =
  match device u with
  | Some (Multi ds) -> ds
  | _ -> invalid_arg "the value is not on several devices"

let not_movement () = invalid_arg "the node is not the movement matched"

let shape_arg u =
  match marg u with Reshape s | Expand s -> s | _ -> not_movement ()

let bounds_arg u =
  match marg u with Pad b | Shrink b -> b | _ -> not_movement ()

let device_ranges x =
  List.filter
    (fun r -> Axis_type.equal (axis_type r) Axis_type.Device)
    (Nodes.to_list (ranges x))

(* Copies and shard selections across devices *)

(* [x] on the [i]th device: its device range is [i]. *)
let at_device i x =
  match device_ranges x with
  | r :: _ ->
      substitute ~calls:Skip x [ (r, const_like r (`Int (Bigint.of_int i))) ]
  | [] -> x

let apply_shrink marg s i =
  let at_device = function Sym x -> Sym (at_device i x) | n -> n in
  mop s (Shrink (List.map (fun (a, b) -> (at_device a, at_device b)) marg))

let mstack_early_shrink ms shrink =
  let marg = bounds_arg shrink in
  let each i x =
    if op x = Op.Copy then
      let s = apply_shrink marg (nth x 0) i in
      if Option.equal equal_device (device s) (device x) then contiguous s
      else copy_to_device s (Option.get (device x))
    else contiguous (apply_shrink marg x i)
  in
  replace ms ~src:(List.mapi each (src ms))

(* A selection of a value on no device is that value. *)
let pm_unselect_deviceless =
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.op Op.Mselect ~src:[ Upat.var "x" ])
        (fun m ->
          if Option.is_none (device (m "x")) then Some (m "x") else None);
    ])

let lower_broadcast_copy c x =
  match (device c, device x) with
  | Some (Multi ds), Some (Single _) ->
      let sx =
        graph_rewrite ~calls:Skip ~ctx:() (simplify x) pm_unselect_deviceless
      in
      if Option.is_none (device sx) then
        Some (v Op.Mstack ~src:(List.map (fun _ -> sx) ds))
      else
        Some
          (v Op.Mstack
             ~src:(List.map (fun d -> copy_to_device x (Single d)) ds))
  | _ -> None

let copy_to_one c x =
  match (device c, device x) with
  | Some (Single _ as d), Some (Multi _) ->
      let m = mselect x 0 in
      Some
        (if Option.equal equal_device (device m) (Some d) then m
         else copy_to_device m d)
  | _ -> None

let shard_index u =
  match arg u with
  | Shard i -> i
  | _ -> invalid_arg "a shard selection needs its shard"

let replace_allreduce =
  let copy =
    Upat.op Op.Copy ~name:"c" ~src:[ Upat.var "x" ] ~allow_any_len:true
  in
  Pattern_matcher.v
    (fun () -> [
      (* A copy to several devices is a copy to each. *)
      rule copy (fun m -> lower_broadcast_copy (m "c") (m "x"));
      (* A copy from several devices to one copies the first shard. *)
      rule copy (fun m -> copy_to_one (m "c") (m "x"));
      (* A shard of a stack of shards is that shard. *)
      rule
        (Upat.op Op.Mselect ~name:"ms"
           ~src:[ Upat.op Op.Mstack ~name:"mstack" ])
        (fun m -> Some (nth (m "mstack") (shard_index (m "ms"))));
      (* Shrinks move before stacks of shards. *)
      rule
        (Upat.op Op.Shrink ~name:"shrink" ~allow_any_len:true
           ~src:[ Upat.op Op.Mstack ~name:"ms" ])
        (fun m -> Some (mstack_early_shrink (m "ms") (m "shrink")));
      (* Shard selections move before movements and elementwise operations, so
         that a joined value's bitcast back to its dtype is read where the
         selected shard is, with no kernel of its own. *)
      rule
        (Upat.op Op.Mselect ~name:"ms"
           ~src:
             [
               Upat.v ~op:Op.Set.movement ~name:"v" ~allow_any_len:true
                 ~src:[ Upat.var "s" ]
                 ();
             ])
        (fun m ->
          let v = m "v" and i = shard_index (m "ms") in
          let arg x =
            let y = at_device i x in
            if y == x then x else simplify y
          in
          Some
            (replace v
               ~src:(mselect (m "s") i :: List.map arg (List.tl (src v)))));
      rule
        (Upat.op Op.Mselect ~name:"ms"
           ~src:[ Upat.v ~op:Op.Set.elementwise ~name:"a" () ])
        (fun m ->
          let i = shard_index (m "ms") in
          let a = m "a" in
          Some
            (replace a
               ~src:
                 (List.map
                    (fun s -> if is_multi s then mselect s i else s)
                    (src a))));
    ])

(* Without LATE_ALLREDUCE, an allreduce becomes its copies here, before the
   rules above see it. *)
let replace_allreduce =
  Pattern_matcher.append
    (Pattern_matcher.v (fun () ->
         [
           rule
             (Upat.op Op.Allreduce ~name:"red" ~src:[ Upat.var "buf" ])
             (fun m ->
               if Setting.value Setting.late_allreduce then None
               else Allreduce.handle_allreduce (m "red"));
         ]))
    replace_allreduce

(* Sharded operations *)

(* [joined u d] is [u], which holds each element on one device of [d] and zero
   on the others, whole on each: the devices' element bits are joined by a
   bitwise or, which keeps each element's exact bits, where a sum would turn
   [-0.] into [+0.] and quiet a NaN. *)
let joined u d =
  let dt = dtype u in
  if List.exists (Dtype.equal dt) Dtype.weaks then
    invalid_arg "an index type has no width to cross devices in"
  else if Dtype.equal dt Dtype.Bool then
    cast (allreduce (cast u Dtype.Uint8) Op.Or d) Dtype.Bool
  else
    let bits =
      match Dtype.itemsize dt with
      | 1 -> Dtype.Uint8
      | 2 -> Dtype.Uint16
      | 4 -> Dtype.Uint32
      | 8 -> Dtype.Uint64
      | n ->
          invalid_arg
            (Printf.sprintf "%d-byte elements cannot be joined across devices"
               n)
    in
    bitcast (allreduce (bitcast u bits) Op.Or d) dt

let rec shard_srcs msrcs axis =
  let devices = List.filter_map device msrcs in
  if not (Helpers.all_same equal_device devices) then
    invalid_arg "resharded values must be on the same devices";
  (* Without devices the sharding range comes from the unshard itself, such as a
     thread's range; device shards range over the devices. *)
  let sharding_rng =
    match devices with
    | Multi ds :: _ ->
        range ~axis_type:Axis_type.Device (Int (List.length ds)) [ -1 ]
    | _ -> (
        match List.find_opt is_unshard msrcs with
        | Some m -> nth m 1
        | None -> invalid_arg "resharding needs a device or a sharding range")
  in
  let out_shape = broadcast_shape (List.map shape msrcs) in
  let each mlb =
    let src_axis = axis - (List.length out_shape - ndim mlb) in
    if Option.equal Int.equal (Ops.axis mlb) (Some src_axis) then nth mlb 0
    else
      (* Every shard gets the whole copy, sharded iff this source has the axis:
         broadcast sources stay whole. *)
      let full =
        if Ops.axis mlb = None then mlb
        else copy_multi mlb (Option.get (device mlb))
      in
      if List.mem axis (broadcast_axes (shape mlb) out_shape) then full
      else shard_slice full src_axis sharding_rng
  in
  List.map each msrcs

(* The part of [full], a whole value of [multi]'s shape, that belongs to this
   shard, as the device path takes it. *)
and shard_subview full multi =
  if not (equal_shape (shape full) (shape multi)) then
    invalid_arg "a shard's sub-view needs the value's whole shape";
  if op full = Op.Expand && shape (nth full 0) = [] then
    expand (nth full 0) (shape (nth multi 0))
  else
    List.fold_left
      (fun f (ax, rng) -> shard_slice f ax rng)
      full (sharding multi)

and shard_idx rng dev_idx =
  match device_ranges rng with
  | [] -> 0
  | r :: _ -> (
      match
        ssimplify
          (substitute ~calls:Skip rng
             [ (r, const_like r (`Int (Bigint.of_int dev_idx))) ])
      with
      | Int n -> n
      | Sym _ -> invalid_arg "a shard's position is not a constant")

and copy_multi multi device =
  let sharding = sharding multi in
  match device with
  | Single _ ->
      (* Reconstruct by concatenating along each axis from last to first. *)
      let pieces =
        List.init
          (List.length (devices multi))
          (fun i ->
            ( List.map (fun (_, r) -> shard_idx r i) sharding,
              copy_to_device (mselect (nth multi 0) i) device ))
      in
      let join pieces j =
        let ax = fst (List.nth sharding j) in
        let key idxs = List.filteri (fun k _ -> k <> j) idxs in
        let keys =
          List.sort_uniq (List.compare Int.compare)
            (List.map (fun (i, _) -> key i) pieces)
        in
        List.map
          (fun k ->
            let grp =
              List.stable_sort
                (fun (a, _) (b, _) -> Int.compare a b)
                (List.filter_map
                   (fun (i, p) ->
                     if key i = k then Some (List.nth i j, p) else None)
                   pieces)
            in
            (k, cat ~axis:ax (snd (List.hd grp)) (List.map snd (List.tl grp))))
          keys
      in
      snd
        (List.hd
           (List.fold_left join pieces
              (List.rev (List.init (List.length sharding) Fun.id))))
  | Multi _ ->
      (* Unshard every axis and allreduce. *)
      let pad_axis v (ax, rng) =
        let bsz = List.nth (shape v) ax in
        let r = Sym rng and last = Int (count rng - 1) in
        pad v
          (List.init (ndim v) (fun a ->
               if a <> ax then Some (Int 0, Int 0)
               else Some Sint.(bsz * r, (bsz * last) - (bsz * r))))
      in
      joined (List.fold_left pad_axis (nth multi 0) sharding) device

let alu_multi root =
  match List.filter is_unshard (src root) with
  | [] -> None
  | target :: _ -> (
      let sharding = sharding target in
      (* Same sharding (peel the unshard), a whole value of the full shape (take
         its per-shard sub-view), or a broadcast scalar. *)
      let can_handle m =
        match Ops.sharding m with
        | [] -> shape m = [] || equal_shape (shape m) (shape target)
        | s -> equal_sharding s sharding
      in
      if List.for_all can_handle (src root) then
        let each m =
          if is_unshard m then nth m 0
          else if shape m = [] then m
          else shard_subview m target
        in
        match List.map each (src root) with
        | x :: rest -> Some (reshard target (alu x (op root) rest))
        | [] -> None
      else
        (* Resharding: the single-axis fallback. *)
        let axis = sharded_axis root in
        match shard_srcs (src root) axis with
        | x :: rest ->
            Some
              (unshard ~ranges:[ nth target 1 ] (alu x (op root) rest) [ axis ])
        | [] -> None)

let reduce_multi root multi =
  match arg root with
  | Reduce { op = rop; num_axes } ->
      let sharding = sharding multi in
      let reduced, remaining =
        List.partition (fun (ax, _) -> ax < num_axes) sharding
      in
      let x = nth multi 0 in
      let local = Ops.rop x rop (List.init num_axes Fun.id) in
      if reduced <> [] then begin
        if remaining <> [] then
          invalid_arg "a partial allreduce of a value sharded on several axes";
        let device = Multi (devices multi) in
        (* All sharded axes are reduced: a full allreduce. *)
        match (op x, src x) with
        | Op.Cast, inner :: _
          when Setting.value Setting.allreduce_cast
               && List.mem (dtype inner) Dtype.[ Bfloat16; Float16 ] ->
            Some
              (cast
                 (allreduce (cast local (dtype inner)) rop device)
                 (dtype local))
        | _ -> Some (allreduce local rop device)
      end
      else
        (* No sharded axis is reduced: piecewise, keeping the sharding. *)
        Some
          (unshard_as
             (List.map (fun (ax, r) -> (ax - num_axes, r)) remaining)
             local)
  | _ -> None

let reshape_multi root multi =
  let new_shape = shape_arg root in
  if truth Sint.(prod (shape multi) <> prod new_shape) then
    invalid_arg "a reshape must keep the number of elements";
  let moved () = invalid_arg "a reshape moves elements between shards" in
  let sint_simplify = function Sym u -> ssimplify u | n -> n in
  (* Map every sharded axis through the reshape: its boundary must survive and
     stay divisible by its shard count. *)
  let arg_acc =
    List.rev
      (List.fold_left
         (fun acc s -> sint_simplify Sint.(List.hd acc * s) :: acc)
         [ Int 1 ] new_shape)
  in
  let new_sharding =
    List.map
      (fun (ax, rng) ->
        let target = sint_simplify (Sint.prod (List.take ax (shape multi))) in
        let new_ax =
          match List.find_index (Sint.equal target) (List.rev arg_acc) with
          | Some i -> List.length arg_acc - i - 1
          | None -> moved ()
        in
        if truth Sint.(List.nth new_shape new_ax % Int (count rng) <> Int 0)
        then moved ();
        (new_ax, rng))
      (sharding multi)
  in
  let shard_shape =
    List.mapi
      (fun a s ->
        match List.assoc_opt a new_sharding with
        | Some rng -> Sint.(s // Int (count rng))
        | None -> s)
      new_shape
  in
  unshard_as new_sharding (reshape (nth multi 0) shard_shape)

let expand_multi root multi =
  let added = shape_arg root in
  let shift = List.length added in
  unshard_as
    (List.map (fun (ax, r) -> (ax + shift, r)) (sharding multi))
    (mop (nth multi 0) (Expand added))

let pad_multi root multi =
  let marg = bounds_arg root in
  let x = nth multi 0 in
  List.iter
    (fun (ax, _) ->
      let s, e = List.nth marg ax in
      if not (Sint.equal s (Int 0) && Sint.equal e (List.nth (shape multi) ax))
      then invalid_arg "cannot pad a sharded axis")
    (sharding multi);
  let local_pad =
    List.mapi
      (fun a s ->
        if List.mem_assoc a (sharding multi) then (Int 0, List.nth (shape x) a)
        else s)
      marg
  in
  reshard multi (mop x (Pad local_pad))

let permute_multi root multi =
  let order = match marg root with Permute p -> p | _ -> not_movement () in
  let index ax = Option.get (List.find_index (Int.equal ax) order) in
  unshard_as
    (List.map (fun (ax, r) -> (index ax, r)) (sharding multi))
    (permute (nth multi 0) order)

(* Each sharded axis resolves on its own: a shrink to exactly its range's own
   shard removes the sharding along that axis, as a fragment indexed by its
   thread's range is that thread's shard, with no copy. *)
let shrink_multi root multi =
  let marg = bounds_arg root in
  let x = nth multi 0 in
  let sharding = sharding multi in
  let local_marg = Array.of_list marg in
  let rec go remaining = function
    | [] ->
        let v = mop x (Shrink (Array.to_list local_marg)) in
        if remaining = [] then v else unshard_as remaining v
    | (ax, rng) :: rest -> (
        let shard_sz = List.nth (shape x) ax in
        let s, l = List.nth marg ax in
        let sz = sint_to_uop shard_sz in
        if
          Sint.equal (ssimplify (sint_to_uop l)) shard_sz
          && Sint.equal (ssimplify (sub (sint_to_uop s) (mul rng sz))) (Int 0)
        then begin
          local_marg.(ax) <- (Int 0, shard_sz);
          go (List.filter (fun (a, _) -> a <> ax) remaining) rest
        end
        else
          let part_bounds =
            List.init (count rng) (fun i -> (Sint.(Int i * shard_sz), shard_sz))
          in
          let is s' (a, b) = Sint.equal (fst s') a && Sint.equal (snd s') b in
          if is (s, l) (Int 0, List.nth (shape multi) ax) then begin
            (* A whole axis stays sharded, and the other axes shrink locally. *)
            local_marg.(ax) <- (Int 0, shard_sz);
            go remaining rest
          end
          else
            (* Otherwise a shrink of the shard axis selects one partition,
               copied to every device, only when a single axis is sharded across
               devices. *)
            match List.find_index (is (s, l)) part_bounds with
            | Some part when List.length sharding = 1 && is_multi multi ->
                let non_shard =
                  List.mapi
                    (fun i t -> if i = ax then (Int 0, shard_sz) else t)
                    marg
                in
                mop
                  (copy_to_device ~shard:part x (Multi (devices multi)))
                  (Shrink non_shard)
            | _ -> invalid_arg "cannot shrink a sharded axis")
  in
  go sharding sharding

let flip_multi root multi =
  let flips = match marg root with Flip f -> f | _ -> not_movement () in
  List.iter
    (fun (ax, _) ->
      if List.nth flips ax then invalid_arg "cannot flip a sharded axis")
    (sharding multi);
  let axes =
    List.concat (List.mapi (fun i f -> if f then [ i ] else []) flips)
  in
  reshard multi (flip (nth multi 0) axes)

(* A stack adds a leading axis: its sources are sharded one axis below. A whole
   source takes its per-shard sub-view, as an elementwise operation's does. *)
let stack_multi root =
  match List.filter is_unshard (src root) with
  | [] -> None
  | first :: _ as multis ->
      let sharding = sharding first in
      if List.for_all (fun m -> equal_sharding (Ops.sharding m) sharding) multis
      then
        let each m = if is_unshard m then nth m 0 else shard_subview m first in
        Some
          (unshard_as
             (List.map (fun (ax, r) -> (ax + 1, r)) sharding)
             (v Op.Stack ~src:(List.map each (src root))))
      else
        (* Resharding: the single-axis fallback. *)
        let axis = sharded_axis root in
        Some
          (unshard
             ~ranges:[ nth first 1 ]
             (v Op.Stack ~src:(shard_srcs (src root) (axis - 1)))
             [ axis ])

(* Each sharded axis of an index resolves into its range's own shard, in one of
   two ownerships: contiguous, [idx = rng * shard_sz + local], where the range
   owns a block; or strided, [idx = rng + local * shard_sz], where it owns every
   [shard_sz]th element. *)
let index_multi root multi =
  let x = nth multi 0 in
  let resolve_axis idxs (ax, rng) =
    let shard_sz = List.nth (shape x) ax in
    let sz = sint_to_uop shard_sz in
    let idx = List.nth idxs ax in
    let within local =
      Dtype.Value.(vmin local >= of_int 0)
      && truth Sint.(Int (Dtype.Value.to_int (vmax local)) < shard_sz)
    in
    let local = simplify (sub idx (mul rng sz)) in
    let diff = simplify (sub idx rng) in
    let crosses () =
      invalid_arg "an index crosses the shards of a sharded axis"
    in
    let local =
      if within local then local
      else
        let md = simplify (mod_ diff sz) in
        let is_zero =
          op md = Op.Const
          && match Ops.value md with `Int z -> Bigint.equal z Bigint.zero | _ -> false
        in
        if not is_zero then crosses ()
        else
          let strided = simplify (div ~rounding:`Floor diff sz) in
          if within strided then strided else crosses ()
    in
    List.mapi (fun i u -> if i = ax then local else u) idxs
  in
  index x (List.fold_left resolve_axis (List.tl (src root)) (sharding multi))

(* A gather reads its source and its index on the same devices. *)
let same_devices x l =
  match (device x, device l) with
  | Some dx, Some dl when not (equal_device dx dl) ->
      invalid_arg
        (Format.asprintf "a gather of a value on %a by an index on %a"
           pp_device dx pp_device dl)
  | _ -> ()

(* A gather of a sharded value by a whole index. Sharded along trailing axes,
   each shard gathers its own part. Sharded along the gathered rows, each shard
   reads the rows it holds, zero elsewhere, and the shards are joined. *)
let gather_shards multi l =
  let x = nth multi 0 and n = ndim l in
  match sharding multi with
  | sharding when List.for_all (fun (ax, _) -> ax >= 1) sharding ->
      unshard_as
        (List.map (fun (ax, r) -> (ax - 1 + n, r)) sharding)
        (index x [ l ])
  | [ (0, rng) ] ->
      let rows = sint_to_uop (List.hd (shape x)) in
      let local = sub l (mul rng rows) in
      let inside = bitwise_and (ge local (int 0)) (lt local rows) in
      let read =
        index x [ maximum (minimum local (sub rows (int 1))) (int 0) ]
      in
      let trailing = List.map (fun _ -> Int 1) (List.drop 1 (shape x)) in
      let inside = reshape inside (shape l @ trailing) in
      joined
        (where inside read (const_like read (`Int Bigint.zero)))
        (Multi (devices multi))
  | _ -> invalid_arg "a gather of a value sharded on its rows and another axis"

(* A sharded index is joined whole on each device, in [int64] since an index
   type has no width to cross devices in, and gathers as a whole index does. *)
let gather_multi root multi =
  match src root with
  | [ _; l ] when is_unshard l ->
      same_devices multi l;
      let wide = unshard_as (sharding l) (cast (nth l 0) Dtype.Int64) in
      gather_shards multi (cast (copy_multi wide (Multi (devices l))) (dtype l))
  | [ _; l ] ->
      same_devices multi l;
      gather_shards multi l
  | _ -> invalid_arg "a gather takes one index"

(* Scatters

   A store through a gather of a sharded value is a scatter: each device stores
   the elements its shards hold. Such a gather is marked as the rewrite enters
   its store, before it reaches the gather, so that the gather is not read, and
   the store it becomes loses the mark. *)

let scattered = Tag.String "scattered"

let is_scattered u =
  match tag u with
  | Some t -> Tag.equal t scattered
  | None -> false

let scatter_dests =
  Pattern_matcher.v (fun () ->
      [
        rule (Upat.op Op.Store ~name:"st") (fun m ->
            let st = m "st" in
            match src st with
            | dest :: rest
              when Indexing.is_gather dest
                   && (not (is_scattered dest))
                   && op_in_backward_slice_with_self (nth dest 0) [ Op.Unshard ]
              ->
                if Option.is_some (tag dest) then
                  invalid_arg "a store through a gather that carries a tag";
                Some (replace st ~src:(replace dest ~tag:(Some scattered) :: rest))
            | _ -> None);
      ])

(* [whole_index l] is the sharded index [l] whole on each of its devices: its
   positions joined in [int64], since an index type has no width to cross
   devices in, and its validity joined beside them. *)
let whole_index l =
  let s = sharding l and ds = Multi (devices l) in
  let join u = copy_multi (unshard_as s u) ds in
  let shard = nth l 0 in
  let idx = cast (join (cast (get_idx shard) Dtype.Int64)) (dtype l) in
  match get_valid shard with
  | v when op v = Op.Const -> idx
  | v -> valid idx (join v)

(* A scatter into a sharded value: its index whole on each device. Sharded along
   trailing axes, each shard stores into its own part. Sharded along the
   gathered rows, each shard stores the rows it holds, at their index within
   it, and the index of every other row is Invalid, which drops its store. *)
let scatter_shards root multi =
  let x = nth multi 0 in
  let l =
    match src root with
    | [ _; l ] when is_unshard l ->
        same_devices multi l;
        whole_index l
    | [ _; l ] ->
        same_devices multi l;
        l
    | _ -> invalid_arg "a scatter takes one index"
  in
  let n = ndim l in
  let scattered u = replace u ~tag:(Some scattered) in
  match sharding multi with
  | sharding when List.for_all (fun (ax, _) -> ax >= 1) sharding ->
      unshard_as
        (List.map (fun (ax, r) -> (ax - 1 + n, r)) sharding)
        (scattered (index x [ l ]))
  | [ (0, rng) ] ->
      let rows = sint_to_uop (List.hd (shape x)) in
      let local = sub (get_idx l) (mul rng rows) in
      let inside =
        bitwise_and (get_valid l)
          (bitwise_and (ge local (int 0)) (lt local rows))
      in
      scattered (index x [ valid local inside ])
  | _ ->
      invalid_arg "a scatter into a value sharded on its rows and another axis"

(* The value stored by a scatter: whole on each device, as its index is, and
   each shard of a destination sharded along trailing axes stores its part.
   The store's destination loses its mark. *)
let store_scattered dest v =
  let v =
    if is_unshard v then copy_multi v (Option.get (device v)) else v
  in
  let plain u = replace u ~tag:None in
  if is_unshard dest then store (plain (nth dest 0)) (shard_subview v dest)
  else store (plain dest) v

let store_after_multi dest src =
  reshard src (after dest [ store dest (nth src 0) ])

(* A sharded value stored into a whole destination: each shard stores into its
   own sub-view of it. A destination replicated on several devices would keep
   only each device's part in each copy, so it is refused. *)
let store_value_multi dest multi =
  match device dest with
  | Some (Multi _ as d) when not (is_unshard dest) ->
      invalid_arg
        (Format.asprintf
           "a value sharded on %a stored into a destination replicated on %a"
           pp_device
           (Option.get (device multi))
           pp_device d)
  | _ -> store (shard_subview dest multi) (nth multi 0)

(* A store into a sharded destination: each shard stores into its own shard of
   it, and the value is taken as an arithmetic operation takes it. Scalars
   arrive expanded to the whole shape, so they take their sub-view too. *)
let store_dest_multi root multi =
  let each x =
    if is_unshard x then nth x 0
    else if equal_shape (shape x) (shape multi) then shard_subview x multi
    else x
  in
  v (op root)
    ~src:(nth multi 0 :: List.map each (List.tl (src root)))
    ~arg:(arg root)

let passthrough_multi root multi =
  reshard multi
    (v (op root)
       ~src:(nth multi 0 :: List.map peel (List.tl (src root)))
       ~arg:(arg root))

(* A call's body is a plain parametric program: it is rewritten like anything
   else, its outputs taking their per-shard views through the store rules, and
   its arguments become their shards. *)
let rec rewrite_into_function call =
  if not (is_inline_call call) then None
  else
    let new_body =
      graph_rewrite ~calls:Skip ~ctx:() (body call) (Lazy.force multi_pm)
    in
    if op new_body <> Op.Sink then invalid_arg "a call's body must stay a sink";
    Some (replace call ~src:(new_body :: List.map peel (List.tl (src call))))

and multi_pm =
  lazy
    (let with_multi o f =
       rule
         (Upat.op o ~name:"root"
            ~src:[ Upat.op Op.Unshard ~name:"multi"; Upat.wild; Upat.wild ])
         (fun m -> Some (f (m "root") (m "multi")))
     in
     let unary o f =
       rule
         (Upat.op o ~name:"root" ~src:[ Upat.op Op.Unshard ~name:"multi" ])
         (fun m -> Some (f (m "root") (m "multi")))
     in
     let shaped o f =
       rule
         (Upat.op o ~name:"root"
            ~src:[ Upat.op Op.Unshard ~name:"multi"; Upat.wild ])
         (fun m -> Some (f (m "root") (m "multi")))
     in
     Pattern_matcher.append
       (Pattern_matcher.v
          (fun () -> [
            rule
              (Upat.v ~op:Op.Set.alu ~name:"root" ~early_reject:[ Op.Unshard ]
                 ()) (fun m -> alu_multi (m "root"));
            rule
              (Upat.op Op.Reduce ~name:"root"
                 ~src:[ Upat.op Op.Unshard ~name:"multi" ])
              (fun m -> reduce_multi (m "root") (m "multi"));
            shaped Op.Reshape reshape_multi;
            shaped Op.Expand expand_multi;
            with_multi Op.Pad pad_multi;
            with_multi Op.Shrink shrink_multi;
            unary Op.Permute permute_multi;
            unary Op.Flip flip_multi;
            rule (Upat.op Op.Stack ~name:"root" ~early_reject:[ Op.Unshard ])
              (fun m -> stack_multi (m "root"));
            rule
              (Upat.op Op.Store ~src:[ Upat.var "dest"; Upat.var "v" ])
              (fun m ->
                let dest = m "dest" in
                if is_scattered (peel dest) then
                  Some (store_scattered dest (m "v"))
                else None);
            rule
              (Upat.op Op.Index ~name:"root" ~allow_any_len:true
                 ~src:[ Upat.op Op.Unshard ~name:"multi" ])
              (fun m ->
                let root = m "root" in
                if is_scattered root then Some (scatter_shards root (m "multi"))
                else if Indexing.is_gather root then
                  Some (gather_multi root (m "multi"))
                else Some (index_multi root (m "multi")));
            (* A gather by a sharded index gathers each part of the index. *)
            rule
              (Upat.op Op.Index ~name:"root"
                 ~src:[ Upat.var "x"; Upat.op Op.Unshard ~name:"multi" ])
              (fun m ->
                let x = m "x" and multi = m "multi" in
                same_devices x multi;
                Some (reshard multi (index x [ nth multi 0 ])));
            rule
              (Upat.op Op.After
                 ~src:
                   [
                     Upat.op Op.Unshard;
                     Upat.op Op.Store
                       ~src:
                         [
                           Upat.op Op.Unshard ~name:"dest";
                           Upat.op Op.Unshard ~name:"src";
                         ];
                   ])
              (fun m -> Some (store_after_multi (m "dest") (m "src")));
            (* A copy of a sharded value copies every shard to the target. *)
            rule
              (Upat.op Op.Copy ~name:"copy" ~allow_any_len:true
                 ~src:[ Upat.op Op.Unshard ~name:"multi" ])
              (fun m ->
                match arg (m "copy") with
                | Device d -> Some (copy_multi (m "multi") d)
                | _ -> None);
            rule
              (Upat.op Op.Allreduce ~name:"red"
                 ~src:[ Upat.op Op.Unshard ~name:"multi" ])
              (fun m ->
                let multi = m "multi" in
                match arg (m "red") with
                | Allreduce { op; device } ->
                    Some (reshard multi (allreduce (nth multi 0) op device))
                | _ -> None);
            (* Calls that produce values are rewritten through their body. *)
            rule (Upat.op Op.Call ~name:"call") (fun m ->
                rewrite_into_function (m "call"));
            rule
              (Upat.v
                 ~op:(ops Op.[ Call; After ])
                 ~name:"root" ~allow_any_len:true
                 ~src:[ Upat.op Op.Unshard ~name:"multi" ]
                 ())
              (fun m -> Some (passthrough_multi (m "root") (m "multi")));
            (* Other calls, custom kernels among them, just lose their
               unshards. *)
            rule
              (Upat.op Op.Call ~dtype:[ Dtype.Void ] ~name:"root"
                 ~early_reject:[ Op.Unshard ]) (fun m ->
                let root = m "root" in
                Some (v Op.Call ~src:(List.map peel (src root)) ~arg:(arg root)));
            rule
              (Upat.v
                 ~op:
                   (ops
                      Op.[ Cast; Bitcast; Stage; Detach; Contiguous_backward ])
                 ~name:"root"
                 ~src:[ Upat.op Op.Unshard ~name:"multi" ]
                 ())
              (fun m -> Some (passthrough_multi (m "root") (m "multi")));
            (* A sharded value stored into a whole destination, as a fragment
               into a full output tile. *)
            rule
              (Upat.op Op.Store
                 ~src:[ Upat.var "dest"; Upat.op Op.Unshard ~name:"multi" ])
              (fun m -> Some (store_value_multi (m "dest") (m "multi")));
            (* A store into a sharded destination, as a fragment's initial
               value: each shard stores into its own. *)
            rule
              (Upat.op Op.Store ~name:"root" ~allow_any_len:true
                 ~src:[ Upat.op Op.Unshard ~name:"multi" ])
              (fun m -> Some (store_dest_multi (m "root") (m "multi")));
          ]))
       replace_allreduce)

let multi_pm = Lazy.force multi_pm
