(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop

let late_allreduce = Helpers.getenv "LATE_ALLREDUCE" 1
let int_ = U.const_int
let zero = int_ 0
let emit = function [x] -> x | xs -> U.stack xs
let simp = Symbolic.simplify
let bin op lhs rhs = simp (U.alu_binary ~op ~lhs ~rhs)
let mul = bin Ops.Mul
let div = bin Ops.Floordiv
let sub lhs rhs = bin Ops.Add lhs (mul rhs (int_ (-1)))
let eq a b = U.equal (simp a) (simp b)
let prod = List.fold_left mul (int_ 1)
let count rng = Bound.to_int (U.vmax rng) + 1
let is_multi u = U.op u = Ops.Unshard
let inner u = (U.src u).(0)
let unwrap u = if is_multi u then inner u else u
let wrap src sharding =
  match sharding with
  | [] -> src
  | _ -> U.unshard ~src ~axes:(List.map fst sharding) ~ranges:(List.map snd sharding) ()
let rewrap src multi = wrap src (U.sharding multi)
let same_shape a b = List.length a = List.length b && List.for_all2 eq a b
let same_sharding a b = List.length a = List.length b
  && List.for_all2 (fun (a, r) (b, s) -> a = b && U.equal r s) a b
let device_exn u = match U.device_of u with
  | Some d -> d | None -> invalid_arg "multi: device required"
let index_of x xs =
  let rec loop i = function
    | [] -> invalid_arg "multi: axis not found"
    | y :: _ when x = y -> i
    | _ :: rest -> loop (i + 1) rest in
  loop 0 xs
let subst_device_num node i =
  let mappings = U.ranges node |> List.filter_map (fun r ->
      match U.as_range r with
      | Some {kind = Axis_type.Device; _} -> Some (r, int_ i)
      | _ -> None) in
  simp (U.substitute mappings node)

let shard src axis rng =
  let shape = U.shape src in
  let dim = List.nth shape axis in
  let n = int_ (count rng) in
  if not (eq (bin Ops.Floormod dim n) zero) then invalid_arg "multi: uneven shard";
  let size = div dim n in
  U.shrink ~src
    ~offset:(emit (List.mapi (fun i _ -> if i = axis then mul rng size else zero) shape))
    ~size:(emit (List.mapi (fun i d -> if i = axis then size else d) shape))

let shard_subview full multi =
  if not (same_shape (U.shape full) (U.shape multi)) then
    invalid_arg "multi: shard subview shape mismatch";
  if U.op full = Ops.Expand && U.shape (inner full) = [] then
    U.expand ~src:(inner full) ~dims:(emit (U.shape (inner multi)))
  else List.fold_left (fun value (axis, rng) -> shard value axis rng) full (U.sharding multi)

let cat axis pieces =
  let first = List.hd pieces in
  let shape = U.shape first in
  let total = List.fold_left (fun size p -> bin Ops.Add size (List.nth (U.shape p) axis)) zero pieces in
  let full = List.mapi (fun i d -> if i = axis then total else d) shape in
  let offset = ref zero in
  let pieces = List.map (fun piece ->
      let start = !offset in
      offset := bin Ops.Add start (List.nth (U.shape piece) axis);
      U.pad ~src:piece ~offset:(emit (List.mapi (fun i _ -> if i = axis then start else zero) shape))
        ~size:(emit full)) pieces in
  U.usum pieces

let copy_multi multi device =
  let sharding = U.sharding multi in
  match device with
  | U.Single _ ->
      let devices = match device_exn multi with U.Multi ds -> ds
        | _ -> invalid_arg "multi: gather requires multiple devices" in
      let pieces = List.mapi (fun i _ ->
          let coords = List.map (fun (_, rng) ->
              let index = subst_device_num rng i in
              match U.const_int_value index with
              | Some index -> index
              | None -> invalid_arg "multi: gather shard index is not concrete") sharding in
          let piece = U.mselect ~src:(inner multi) ~index:i in
          let piece = if U.device_of piece = Some device then piece else U.copy ~src:piece ~device () in
          coords, piece) devices in
      let pieces = List.fold_right (fun (axis, _) pieces ->
          let groups = List.fold_left (fun groups (coords, piece) ->
              let rev = List.rev coords in
              let index, key = List.hd rev, List.rev (List.tl rev) in
              let values = Option.value (List.assoc_opt key groups) ~default:[] in
              (key, (index, piece) :: values) :: List.remove_assoc key groups) [] pieces in
          List.sort (fun (a, _) (b, _) -> compare a b) groups
          |> List.map (fun (key, values) ->
              let values = List.sort (fun (a, _) (b, _) -> Int.compare a b) values in
              key, cat axis (List.map snd values))) sharding pieces in
      snd (List.hd pieces)
  | _ ->
      let padded = List.fold_left (fun src (axis, rng) ->
          let shape = U.shape src in
          let size = List.nth shape axis in
          U.pad ~src
            ~offset:(emit (List.mapi (fun i _ -> if i = axis then mul size rng else zero) shape))
            ~size:(emit (List.mapi (fun i d -> if i = axis then mul size (int_ (count rng)) else d) shape)))
          (inner multi) sharding in
      U.allreduce ~src:padded ~device ~op:Ops.Add

let shard_srcs children axis rng =
  let shape = U.broadcast_shape (List.map U.shape children) in
  let rank = List.length shape in
  List.map (fun src ->
      let src_shape = U.shape src in
      let src_axis = axis - (rank - List.length src_shape) in
      match U.sharding src with
      | [a, r] when a = src_axis && U.equal r rng -> inner src
      | sharding ->
          let full = if sharding = [] then src else copy_multi src (device_exn src) in
          if src_axis < 0 || eq (List.nth src_shape src_axis) (int_ 1) then full
          else shard full src_axis rng) children

let alu_multi root =
  let children = U.children root in
  let target = List.find is_multi children in
  let sharding = U.sharding target in
  let can_handle src = match U.sharding src with
    | [] -> U.shape src = [] || same_shape (U.shape src) (U.shape target)
    | s -> same_sharding s sharding in
  if List.for_all can_handle children then
    let src = Array.of_list (List.map (fun x ->
        if is_multi x then inner x else if U.shape x = [] then x else shard_subview x target) children) in
    rewrap (U.replace root ~src ()) target
  else
    let axis = Option.get (U.axis root) in
    let rng = snd (List.hd sharding) in
    let src = Array.of_list (shard_srcs children axis rng) in
    wrap (U.replace root ~src ()) [axis, rng]

let stack_multi root =
  let children = U.children root in
  let target = List.find is_multi children in
  let sharding = U.sharding target in
  let sources, sharding =
    if List.for_all (fun x -> not (is_multi x) || same_sharding (U.sharding x) sharding) children then
      List.map (fun x -> if is_multi x then inner x else shard_subview x target) children,
      List.map (fun (axis, rng) -> axis + 1, rng) sharding
    else
      let axis = Option.get (U.axis root) in
      let rng = snd (List.hd sharding) in
      shard_srcs children (axis - 1) rng, [axis, rng] in
  wrap (U.stack sources) sharding

let reduce_multi root multi =
  let {U.op; num_axes; _} = Option.get (U.as_reduce root) in
  let reduced, remaining = List.partition (fun (axis, _) -> axis < num_axes) (U.sharding multi) in
  let src = inner multi in
  let local = U.reduce_axis ~src ~op ~axes:(List.init num_axes Fun.id) in
  if reduced = [] then wrap local (List.map (fun (axis, rng) -> axis - num_axes, rng) remaining)
  else (
    if remaining <> [] then invalid_arg "multi: partial multi-axis allreduce is unsupported";
    let device = device_exn multi in
    if Helpers.Context_var.get Helpers.allreduce_cast <> 0 && U.op src = Ops.Cast
       && (Dtype.equal (U.dtype (inner src)) Dtype.float16 || Dtype.equal (U.dtype (inner src)) Dtype.bfloat16) then
      U.cast ~src:(U.allreduce ~src:(U.cast ~src:local ~dtype:(U.dtype (inner src))) ~device ~op) ~dtype:(U.dtype local)
    else U.allreduce ~src:local ~device ~op)

let reshape_multi root multi =
  let shape = U.as_shape (U.src root).(1) in
  let old = U.shape multi in
  if not (eq (prod old) (prod shape)) then invalid_arg "multi: reshape must maintain shape product";
  let prefixes = ref [] and acc = ref (int_ 1) in
  List.iteri (fun i dim -> prefixes := (i, !acc) :: !prefixes; acc := mul !acc dim) shape;
  let sharding = List.map (fun (axis, rng) ->
      let target = prod (List.filteri (fun i _ -> i < axis) old) in
      let new_axis = match List.find_opt (fun (_, prefix) -> eq prefix target) !prefixes with
        | Some (i, _) -> i | None -> invalid_arg "multi: reshape moved items between shards" in
      if not (eq (bin Ops.Floormod (List.nth shape new_axis) (int_ (count rng))) zero) then
        invalid_arg "multi: reshape moved items between shards";
      new_axis, rng) (U.sharding multi) in
  let local = List.mapi (fun i dim -> match List.assoc_opt i sharding with
      | None -> dim | Some rng -> div dim (int_ (count rng))) shape in
  wrap (U.reshape ~src:(inner multi) ~shape:(emit local)) sharding

let shrink_multi root multi =
  let offsets = Array.of_list (U.as_shape (U.src root).(1)) in
  let sizes = Array.of_list (U.as_shape (U.src root).(2)) in
  let local = U.shape (inner multi) and full = U.shape multi in
  let selected = ref None in
  let remaining = List.filter (fun (axis, rng) ->
      let size = List.nth local axis in
      let own = eq sizes.(axis) size && eq offsets.(axis) (mul rng size) in
      let whole = eq offsets.(axis) zero && eq sizes.(axis) (List.nth full axis) in
      if not own && not whole then (
        if List.length (U.sharding multi) <> 1 then invalid_arg "multi: unsupported partial shard slice";
        let index = List.find_opt (fun i -> eq offsets.(axis) (mul (int_ i) size) && eq sizes.(axis) size)
            (List.init (count rng) Fun.id) in
        match index, U.device_of multi with
        | Some i, Some (U.Multi _ as device) ->
            selected := Some (U.copy ~src:(U.mselect ~src:(inner multi) ~index:i) ~device ())
        | _ -> invalid_arg "multi: unsupported shard slice");
      offsets.(axis) <- zero; sizes.(axis) <- size;
      whole) (U.sharding multi) in
  let value = U.shrink ~src:(Option.value !selected ~default:(inner multi))
      ~offset:(emit (Array.to_list offsets)) ~size:(emit (Array.to_list sizes)) in
  wrap value remaining

let index_multi root multi =
  let idxs = Array.of_list (List.tl (U.children root)) in
  List.iter (fun (axis, rng) ->
      let size = List.nth (U.shape (inner multi)) axis in
      let in_bounds idx = Bound.compare (U.vmin idx) Bound.zero >= 0
        && Bound.compare (U.vmax idx) (U.vmin size) < 0 in
      let local = sub idxs.(axis) (mul rng size) in
      if in_bounds local then idxs.(axis) <- local
      else
        let diff = sub idxs.(axis) rng in
        let local = div diff size in
        if eq (bin Ops.Floormod diff size) zero && in_bounds local then idxs.(axis) <- local
        else invalid_arg "multi: index is not owned by this shard") (U.sharding multi);
  U.index ~ptr:(inner multi) ~idxs:(Array.to_list idxs) ()

let passthrough root multi =
  let src = Array.map unwrap (U.src root) in
  rewrap (U.replace root ~src ()) multi

let mstack_shrink root ms =
  U.mstack (List.mapi (fun i x ->
      let shrink src = U.shrink ~src ~offset:(subst_device_num (U.src root).(1) i)
          ~size:(subst_device_num (U.src root).(2) i) in
      if U.op x = Ops.Copy then
        let src = shrink (inner x) in
        let device = device_exn x in
        if U.device_of src = Some device then U.contiguous ~src () else U.copy ~src ~device ()
      else U.contiguous ~src:(shrink x) ()) (U.children ms))

let rec multi_pm node =
  let srcs = U.src node in
  let first_multi = Array.length srcs > 0 && is_multi srcs.(0) in
  match U.op node with
  | op when Ops.Group.is_alu op && List.exists is_multi (U.children node) -> Some (alu_multi node)
  | Ops.Stack when List.exists is_multi (U.children node) -> Some (stack_multi node)
  | Ops.Reduce when first_multi -> Some (reduce_multi node srcs.(0))
  | Ops.Reshape when first_multi -> Some (reshape_multi node srcs.(0))
  | Ops.Expand when first_multi ->
      let shift = List.length (U.as_shape srcs.(1)) in
      Some (wrap (U.expand ~src:(inner srcs.(0)) ~dims:srcs.(1))
          (List.map (fun (a, r) -> a + shift, r) (U.sharding srcs.(0))))
  | Ops.Pad when first_multi ->
      let multi = srcs.(0) in
      let offsets = U.as_shape srcs.(1) and sizes = U.as_shape srcs.(2) in
      List.iter (fun (axis, _) ->
          if not (eq (List.nth offsets axis) zero && eq (List.nth sizes axis) (List.nth (U.shape multi) axis)) then
            invalid_arg "multi: padding a sharded axis") (U.sharding multi);
      let sizes = List.mapi (fun i d -> if List.mem_assoc i (U.sharding multi) then List.nth (U.shape (inner multi)) i else d) sizes in
      Some (rewrap (U.pad ~src:(inner multi) ~offset:srcs.(1) ~size:(emit sizes)) multi)
  | Ops.Permute when first_multi ->
      let order = Option.get (U.Arg.as_ints (U.arg node)) in
      Some (wrap (U.permute ~src:(inner srcs.(0)) ~order)
          (List.map (fun (a, r) -> index_of a order, r) (U.sharding srcs.(0))))
  | Ops.Flip when first_multi ->
      let dims = Option.get (U.Arg.as_bools (U.arg node)) in
      if List.exists (fun (axis, _) -> List.nth dims axis) (U.sharding srcs.(0)) then
        invalid_arg "multi: flipping a sharded axis";
      Some (rewrap (U.flip ~src:(inner srcs.(0)) ~dims) srcs.(0))
  | Ops.Shrink when first_multi -> Some (shrink_multi node srcs.(0))
  | Ops.Shrink when U.op srcs.(0) = Ops.Mstack -> Some (mstack_shrink node srcs.(0))
  | Ops.Index when first_multi -> Some (index_multi node srcs.(0))
  | Ops.Copy when first_multi -> Some (copy_multi srcs.(0) (device_exn node))
  | Ops.Copy -> (match U.device_of srcs.(0), U.device_of node with
      | Some (U.Single _), Some (U.Multi devices) ->
          let simple = simp srcs.(0) in
          Some (U.mstack (List.map (fun d -> if U.device_of simple = None then simple
              else U.copy ~src:srcs.(0) ~device:(U.Single d) ()) devices))
      | Some (U.Multi _), Some (U.Single _ as device) ->
          let value = U.mselect ~src:srcs.(0) ~index:0 in
          Some (if U.device_of value = Some device then value else U.copy ~src:value ~device ())
      | _ -> None)
  | Ops.Allreduce ->
      let {U.op; device; src} = Option.get (U.as_allreduce node) in
      if first_multi then Some (rewrap (U.allreduce ~src:(inner src) ~device ~op) src)
      else if late_allreduce = 0 then Allreduce.handle_allreduce src ~op ~device ~shape:(U.max_shape src)
      else None
  | Ops.Call ->
      let {U.body; args; info} = Option.get (U.as_call node) in
      if not info.precompile && U.op body = Ops.Sink && Option.is_none (U.as_kernel_info body) then
        Some (U.call ~body:(U.graph_rewrite multi_pm body) ~args:(List.map unwrap args) ~info)
      else if first_multi then Some (passthrough node body)
      else if Dtype.equal (U.dtype node) Dtype.void && List.exists is_multi args then
        Some (U.replace node ~src:(Array.map unwrap srcs) ())
      else None
  | (Ops.After | Ops.Cast | Ops.Bitcast | Ops.Stage | Ops.Detach | Ops.Contiguous_backward) when first_multi ->
      Some (passthrough node srcs.(0))
  | Ops.Store when first_multi ->
      let multi = srcs.(0) in
      Some (U.replace node ~src:(Array.mapi (fun i x ->
          if i = 0 || is_multi x then unwrap x
          else if same_shape (U.shape x) (U.shape multi) then shard_subview x multi else x) srcs) ())
  | Ops.Store when is_multi srcs.(1) ->
      Some (U.store ~dst:(shard_subview srcs.(0) srcs.(1)) ~value:(inner srcs.(1)) ?gate:(Option.get (U.as_store node)).gate ())
  | Ops.Mselect ->
      let index = Option.get (U.Arg.as_int (U.arg node)) in
      let value = srcs.(0) in
      if U.op value = Ops.Mstack then Some (U.src value).(index)
      else if Ops.Group.is_movement (U.op value) then
        let src = Array.copy (U.src value) in
        src.(0) <- U.mselect ~src:src.(0) ~index;
        Some (U.replace value ~src ())
      else if Ops.Group.is_alu (U.op value) then
        Some (U.replace value ~src:(Array.map (fun x -> match U.device_of x with
            | Some (U.Multi _) -> U.mselect ~src:x ~index | _ -> x) (U.src value)) ())
      else None
  | _ -> None
