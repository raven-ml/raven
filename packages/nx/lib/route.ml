(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array

(* Routing

   Every fallback runs where its operands live. Operands all on the host run on
   nx.cpu. Placed operands must share their devices, and host operands join
   them: each device's backend computes there (see Dispatch). The route is
   decided before anything is read, so operands on two device lists raise before
   any work.

   The result takes the placement tolk's multi-device rewrite gives the same
   operation in a compiled program (schedule/multi.ml), so eager and compiled
   placements agree and stay as they are once devices compute: an elementwise
   operation keeps its operands' split, resharding to the last split axis among
   them; a reduction over a split axis holds a copy on every device; an
   operation along a split axis, or of a kind tolk has no rule for, raises. *)

type route = On_host | At of Placement.t

(* How the axes of an operation's result derive from its operands', which
   decides where the result lives. *)
type rule =
  | Elementwise (* the operands' shape *)
  | Along of int list
    (* acts along these axes, the others as elementwise: sort, scan, pad,
       concatenation, fft, linear algebra *)
  | Gather of int
    (* reads its first operand along this axis at the positions its second
       holds, the other axes as elementwise; along a split axis, a reduction *)
  | Reduce of { axes : int array; keepdims : bool }
  | Contract
    (* a product over the last axis of the first and the next-to-last of the
       second *)
  | Into (* the first operand, with the others written into it *)

(* The disk holds values and computes on none: a value on it takes part in an
   operation as a host value, which the operation reads. *)
let placement_of : type a b. (a, b) Value.t -> Placement.t option = function
  | Host _ -> None
  | Placed r when Placement.on_disk r.r_placement -> None
  | Placed r -> Some r.r_placement
  | Traced _ -> Value.outside_trace ()

let rank (Value.P x) = Array.length (View.shape (Value.view x))

(* The last two axes of [x], along which linear algebra acts. *)
let matrix_axes x =
  let r = rank (Value.P x) in
  [ r - 2; r - 1 ]

(* Every axis of [x] but the first, along which windows are taken. *)
let spatial x = List.init (rank (Value.P x) - 1) succ

(* The axes [padding] pads. *)
let padded padding =
  List.filter
    (fun a -> padding.(a) <> (0, 0))
    (List.init (Array.length padding) Fun.id)

let along op axis =
  invalid_arg
    (Printf.sprintf
       "Nx.%s: the operation runs along the split axis %d, whose shards are on \
        different devices; place the value replicated or on one device first"
       op axis)

(* [combine op gs] is the grid of an elementwise operation's result over
   operands on grids [gs]: that of the split ones, which must be alike; copies
   take it. *)
let combine op gs =
  match List.filter (fun g -> Placement.cuts g <> []) gs with
  | [] -> List.hd gs
  | g :: rest -> (
      match List.find_opt (fun h -> not (Placement.equal g h)) rest with
      | None -> g
      | Some h ->
          invalid_arg
            (Format.asprintf
               "Nx.%s: operands at %a and %a are split differently; place them \
                alike first"
               op Placement.pp g Placement.pp h))

(* [result op rule operands] is where [op]'s result lives, over operands of
   these placements ([None] on the host) and ranks, the placed ones among them
   sharing their devices. *)
let result op rule operands =
  let ps = List.filter_map fst operands in
  let cut p a = List.mem_assoc a (Placement.cuts p) in
  match rule with
  | Elementwise -> combine op ps
  | Along axes ->
      List.iter
        (fun p -> List.iter (fun a -> if cut p a then along op a) axes)
        ps;
      combine op ps
  | Gather axis -> (
      match operands with
      | (Some p, _) :: _ when cut p axis ->
          (* Each device selects among the rows it holds and the selections sum
             across the devices, as tolk lowers a gather: a copy on each. A 1-D
             grid's only cut is this one; a grid cut along other axes too would
             keep those cuts, renumbered as [Reduce] does. *)
          List.fold_left
            (fun p (a, _) -> Placement.uncut p ~axis:a)
            p (Placement.cuts p)
      | _ -> combine op ps)
  | Reduce { axes; keepdims } ->
      let reduce p =
        let p =
          Array.fold_left
            (fun p a -> if cut p a then Placement.uncut p ~axis:a else p)
            p axes
        in
        if keepdims then p
        else
          Placement.map_axes
            (fun a ->
              a - Array.fold_left (fun n r -> if r < a then n + 1 else n) 0 axes)
            p
      in
      combine op (List.map reduce ps)
  | Contract ->
      (* As [a @ b] is [a [..., m, 1, k] * b [..., 1, n, k]] summed over [k]. *)
      let r = List.fold_left (fun r (_, n) -> Int.max r n) 0 operands in
      let lift j (p, n) =
        let axis i =
          if (j = 0 && i = n - 1) || (j = 1 && i = n - 2) then r else i + r - n
        in
        Option.map (Placement.map_axes axis) p
      in
      let p =
        combine op
          (List.concat
             (List.mapi (fun j x -> Option.to_list (lift j x)) operands))
      in
      if cut p r then Placement.uncut p ~axis:r else p
  | Into -> (
      match operands with
      | (Some p, _) :: _ -> p
      | _ ->
          let p = combine op ps in
          List.fold_left
            (fun p (a, _) -> Placement.uncut p ~axis:a)
            p (Placement.cuts p))

(* Views of whole shards of one split storage, each on its own device. A
   compiled program copies such a view to every device of the storage's list
   (schedule/multi.ml, shrink_multi), so eager code joins them as copies on that
   list: [Nx.roll] of a value split in two by one shard succeeds and lands where
   it does compiled. [whole_shards xs] is that list. *)
let whole_shards xs =
  let views =
    List.filter_map
      (fun (Value.P x) ->
        match x with
        | Placed r when not (Placement.on_disk r.r_placement) ->
            Some (r.r_cell, r.r_view, r.r_placement)
        | _ -> None)
      xs
  in
  (* A view of a whole shard reaches as many elements as the shard holds, and
     none twice through a broadcast axis. *)
  let whole (c : Value.cell) v =
    View.numel v = c.length
    && not
         (Array.exists2
            (fun n s -> n > 1 && s = 0)
            (View.shape v) (View.strides v))
  in
  (* Each view is on one of the storage's own devices: a view seen from another
     device over the same memory is a value of that device, which meets the
     others only through [Nx.place]. *)
  let own (c : Value.cell) p =
    match Placement.devices p with
    | [ d ] -> List.memq d (Placement.devices c.placement)
    | _ -> false
  in
  match views with
  | (cell, _, _) :: _
    when List.for_all (fun (c, v, p) -> c == cell && whole c v && own c p) views
    ->
      Some (Placement.devices cell.Value.placement)
  | _ -> None

let mixed op p q =
  invalid_arg
    (Format.asprintf
       "Nx.%s: operands on %a and %a; Nx.place one beside the other" op
       Placement.pp p Placement.pp q)

(* [route where op rule xs] is where [op] runs over [xs], [where] giving each
   operand's placement, [None] for one that joins any. Placed operands on
   different device sets raise, but for [whole_shards]. *)
let route where op rule xs =
  match List.filter_map where xs with
  | [] -> On_host
  | p :: rest -> (
      match List.find_opt (fun q -> not (Placement.same_devices p q)) rest with
      | None -> At (result op rule (List.map (fun o -> (where o, rank o)) xs))
      | Some q -> (
          match whole_shards xs with
          | Some ds -> At (Placement.replicated ds)
          | None -> mixed op p q))

(* How an operation is routed: a computation by a rule over its operands, a
   movement of one operand, placing and reading. Eager routing reads the rule
   from it, and a compiler checking where values live reads it through
   [Nx.Op.placement]. *)
type routing =
  | Computes of rule * Value.packed list
  | Moves of Value.packed * Move.t
  | Places of Placement.t
  | Reads of Value.packed

let routing : type r. r Op.t -> routing =
 fun op ->
  let computes rule = Computes (rule, Op.operands op) in
  let along_axes axes = computes (Along axes) in
  match[@warning "@4@8"] op with
  | Unary _ | Binary _ | Compare _ | Where _ | Fma _
  | Convert (Cast, _, _)
  | Threefry _ | Contiguous _ ->
      computes Elementwise
  | Convert (Bitcast, dt, x) ->
      (* A widening reads each run of the last axis as one element, so the axis
         must be whole on each device. *)
      if Nx_dtype.itemsize dt > Nx_dtype.itemsize (Value.dtype x) then
        along_axes [ rank (Value.P x) - 1 ]
      else computes Elementwise
  | Reduce (_, axes, _) -> computes (Reduce { axes; keepdims = false })
  | Arg_reduce (_, axis, _) ->
      computes (Reduce { axes = [| axis |]; keepdims = false })
  | Scan (_, axis, _) -> along_axes [ axis ]
  | Sort { axis; _ } -> along_axes [ axis ]
  | Argsort { axis; _ } -> along_axes [ axis ]
  | Group _ -> along_axes [ 0; 1 ]
  | Pad (padding, _, _) -> along_axes (padded padding)
  | Cat (axis, _) -> along_axes [ axis ]
  | Gather (axis, _, _) -> computes (Gather axis)
  | Scatter _ | Update _ -> computes Into
  | Unfold { x; _ } -> along_axes (spatial x)
  | Fold { x; _ } -> along_axes (spatial x)
  | Matmul _ -> computes Contract
  | Fft { axes; _ } -> along_axes (Array.to_list axes)
  | Rfft { axes; _ } -> along_axes (Array.to_list axes)
  | Irfft { axes; _ } -> along_axes (Array.to_list axes)
  | Cholesky { x; _ } -> along_axes (matrix_axes x)
  | Qr { x; _ } -> along_axes (matrix_axes x)
  | Lu x -> along_axes (matrix_axes x)
  | Svd { x; _ } -> along_axes (matrix_axes x)
  | Eig { x; _ } -> along_axes (matrix_axes x)
  | Eigh { x; _ } -> along_axes (matrix_axes x)
  | Solve_triangular { a; _ } -> along_axes (matrix_axes a)
  | Move (x, m) -> Moves (Value.P x, m)
  | Place (p, _) -> Places p
  | Read { x; _ } -> Reads (Value.P x)
  | Check { ok; _ } -> Reads (Value.P ok)

(* Where [op]'s result lives: where evaluation puts it, a traced operand joining
   as its placement says. Raises [Invalid_argument] as evaluation does when the
   operands cannot meet. *)
let placement : type r. r Op.t -> Placement.t =
 fun op ->
  let where (Value.P x) =
    match x with
    | Traced t when Placement.is_host t.t_placement -> None
    | Traced t -> Some t.t_placement
    | Host _ | Placed _ -> placement_of x
  in
  match routing op with
  | Computes (rule, xs) -> (
      match route where (Op.name op) rule xs with
      | On_host -> Placement.host
      | At p -> p)
  | Moves (Value.P x, m) ->
      Move.placement (Value.placement x) (View.shape (Value.view x)) m
  | Places p -> p
  | Reads _ -> Placement.host
