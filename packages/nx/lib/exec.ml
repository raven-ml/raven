(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* Messages *)

(* The move a program makes where a device's kernels decline an operation:
   nx moves no data between devices on its own. *)
let move = "place its operands with Nx.place on a set whose kernels compute it"

let names dts = String.concat ", " (List.map (fun (D.Any dt) -> D.name dt) dts)

let refuses ~by ~kernels kind d dts =
  invalid_argf "%s: %s does not compute %s on %s (%s); %s" by kernels kind
    (names dts) (Rig.name d) move

let declined ~by ~kernels node d dts =
  refuses ~by ~kernels (Prim.kind node) d dts

let kernels_of ~by ~op set : (module Nx_kernel.S) =
  match Devices.kernels set with
  | Some k -> k
  | None ->
      invalid_argf
        "%s: %a has no kernels to compute %s; place the value on a set with \
         kernels, mint the set with kernels, or apply the function where a \
         compiler stages it"
        by Devices.pp set op

(* A kernel's answer: [Done], a refusal raised naming [dsts] then [ops], or
   [Declined] passed on. *)
let ran ~by answer dsts ops =
  match (answer : A.answer) with
  | Done -> true
  | Declined -> false
  | refusal -> A.refused by refusal (Array.to_list (Array.append dsts ops))

(* Views *)

(* A view made over [a]'s memory, outside any handle chain: the memory is shared
   from then on, so that no donation of [a]'s value writes what the view
   reads. *)
let view_of a = Rig.Claim.share (A.buffer a)

(* The device of a placement on one device. *)
let rig_of p =
  Devices.rig (Devices.set p) (Option.get (Grid.one (Devices.grid p)))

(* A fresh array of [dt] on [d] laid out as [l], C-contiguous from offset 0.
   A dtype narrower than a byte is created, whose spare bits in its last byte
   are zero. *)
let fresh_over (type w r) (dt : (w, r) D.t) l d : (w, r) A.t =
  if D.bits dt >= 8 then A.v dt l (Rig.Buffer.create d (D.bytes dt (L.numel l)))
  else A.create d dt (L.shape l)

(* A maker of fresh results at their form's placement, on one device: the
   first into [into] where it is given. The layout of a result it makes is
   C-contiguous from offset 0, as {!Prim.val-layouts} states for every
   operation but a movement, a bitcast and a placement, which make no fresh
   result, and as a partition lays out each device's results, so it lays out
   the result as it is. *)
let alloc (type v s d) ?into k (f : (v, s, d) Value.form) : (v, s, d) Value.t =
  let p = f.placement in
  match into with
  | Some a when k = 0 ->
      Value.Array { at = p; a = A.expect f.dtype a; dead = Value.Live }
  | Some _ | None ->
      let a = fresh_over f.dtype f.layout (rig_of p) in
      Value.Array { at = p; a; dead = Value.Live }

(* Programs on one device *)

(* [K]'s map of the one node [k] over the plain operands that follow [dst];
   [Wrong_dtype] where [k] does not take their dtypes, as no program does. *)
let map1 (module K : Nx_kernel.S) k ~dst a =
  match Prim.node1 k (A.dtype a) (A.dtype dst) with
  | None -> A.Wrong_dtype
  | Some p -> K.map p ~dsts:[| A.Any dst |] [| Nx_kernel.Plain a |]

(* [a] stored afresh, C-contiguous, on its device. *)
let copy_array ~by kernels a =
  let (module K : Nx_kernel.S) = kernels in
  let dst = A.create (A.device a) (A.dtype a) (L.shape (A.layout a)) in
  let copied = map1 kernels Copy ~dst a in
  if not (ran ~by copied [| A.Any dst |] [| A.Any a |]) then
    declined ~by ~kernels:K.name
      (Op1 (Copy, D.Any (A.dtype a), 0))
      (A.device a) [ D.Any (A.dtype a) ];
  dst

(* [f m a] for [f] of [A.move] and [A.bitcast]: a view over [a]'s memory
   where a layout expresses it, and over a fresh copy of [a] otherwise,
   whose layout always does. [f] and [m] are passed apart, so a call
   allocates no closure. *)
let as_view ~by set f m a =
  match f m a with
  | Some v ->
      view_of a;
      v
  | None ->
      let kernels = kernels_of ~by ~op:"Copy" set in
      Option.get (f m (copy_array ~by kernels a))

(* The one array of [x], a value on one device: an [Array]'s own, which
   builds no [iarray]. *)
let one (type v s d) (x : (v, s, d) Value.t) =
  match x with
  | Value.Array { a; _ } -> a
  | Value.Shards _ | Value.Handle _ | Value.Traced _ ->
      Iarray.get (Prim.arrays x) 0

(* The load a kernel reads of [l], over a value on one device. *)
let kernel_load (type d) (l : d Value.load) : Nx_kernel.load =
  match l with
  | Value.Plain x -> Nx_kernel.Plain (one x)
  | Value.Padded { x; fill; pad } -> Nx_kernel.Padded { x = one x; fill; pad }

let kernel_loads loads =
  Array.init (Iarray.length loads) (fun i -> kernel_load (Iarray.get loads i))

(* The arrays of the results [r] of [op], each on one device. The results of
   a map of one output, a reduction of one or a scan of one, which most calls
   are, are read without gathering every result's arrays. *)
let dsts : type r. r Value.prim -> r -> A.any array =
 fun op r ->
  match (op, r) with
  | Value.Map { outs = [ _ ]; _ }, (y, ()) -> [| A.Any (one y) |]
  | Value.Reduce { reductions = [ Monoid _ ]; _ }, (y, ()) ->
      [| A.Any (one y) |]
  | Value.Reduce { reductions = [ Logsumexp _ ]; _ }, (y, ()) ->
      [| A.Any (one y) |]
  | Value.Scan { reduction = Monoid _; _ }, y -> [| A.Any (one y) |]
  | Value.Scan { reduction = Logsumexp _; _ }, y -> [| A.Any (one y) |]
  | _ -> Array.map (fun per -> per.(0)) (Prim.result_arrays op r)

(* [op]'s kernel over its operands' arrays, into the results [r]'s arrays, all
   on one device: [false] where it declines. A refusal raises naming [by], the
   results' arrays, then the operands' in their order ({!Prim.val-operands}). *)
let call : type r.
    by:string -> (module Nx_kernel.S) -> r Value.prim -> r -> bool =
 fun ~by kernels op r ->
  let (module K : Nx_kernel.S) = kernels in
  let answer : A.answer =
    match op with
    | Value.Map { prog; loads; _ } ->
        K.map prog ~dsts:(dsts op r) (kernel_loads loads)
    | Value.Reduce { layout; axes; prog; reductions; loads } ->
        let rs = Prim.kernel_reductions reductions
        and shape = Iarray.to_array (L.shape layout) in
        K.reduce prog ~shape ~axes rs ~dsts:(dsts op r)
          (kernel_loads loads)
    | Value.Scan { axis; prog; reduction; loads; carry; _ } ->
        let carry = Option.map (fun (Value.Any c) -> A.Any (one c)) carry in
        K.scan ?carry prog ~axis (Prim.kernel_reduction reduction)
          ~dsts:(dsts op r) (kernel_loads loads)
    | Value.Contract
        {
          batch;
          contracting;
          acc;
          a;
          b;
          init;
          a_scale;
          b_scale;
          segments;
          _;
        } ->
        let scale : _ Value.scale -> Nx_kernel.scale = function
          | Exponent x -> Exponent (one x)
          | Value x -> Value (one x)
        in
        let segments =
          Option.map
            (fun (g : _ Nx_kernel.segments) : _ Nx_kernel.segments ->
              match g with
              | Rows r -> Rows { r with offsets = one r.offsets }
              | Contracting r -> Contracting { r with offsets = one r.offsets })
            segments
        in
        K.contract ~batch ~contracting ~segments ~acc ~dst:(one r)
          ~init:(Option.map one init)
          { array = A.Any (one a); scale = Option.map scale a_scale }
          { array = A.Any (one b); scale = Option.map scale b_scale }
    | Value.Gather { axis; idx; x } ->
        K.gather ~axis ~dst:(one r) (one idx) (one x)
    | Value.Scatter { combine; unique; axis; idx; updates; into } ->
        K.scatter combine ~unique ~axis ~dst:(one r) ~into:(one into)
          (one idx) (one updates)
    | Value.Sort { axis; descending; x; _ } ->
        let values, positions = r in
        K.sort ~axis ~descending ~values:(one values)
          ~positions:(one positions) (one x)
    | Value.Fold { pad; x; _ } -> K.fold pad ~dst:(one r) (one x)
    | Value.Fft (Value.C2c { direction; axes; x }) ->
        K.c2c direction ~axes ~dst:(one r) (one x)
    | Value.Fft (Value.R2c { axes; x; _ }) -> K.r2c ~axes ~dst:(one r) (one x)
    | Value.Fft (Value.C2r { axes; x; _ }) -> K.c2r ~axes ~dst:(one r) (one x)
    | Value.Linalg l -> (
        match (l, r) with
        | Value.Cholesky { triangle; a }, y ->
            K.cholesky triangle ~dst:(one y) (one a)
        | Value.Lu a, (lu, pivots, perm) ->
            K.lu ~lu:(one lu) ~pivots:(one pivots) ~perm:(one perm)
              (one a)
        | Value.Qr { a; _ }, (q, r) -> K.qr ~q:(one q) ~r:(one r) (one a)
        | Value.Svd_values { a; _ }, s -> K.svd ~s:(one s) (one a)
        | Value.Svd { a; _ }, (s, u, vh) ->
            K.svd ~s:(one s) ~vectors:(~u:(one u), ~vh:(one vh)) (one a)
        | Value.Eigh_values { a; _ }, w -> K.eigh ~w:(one w) (one a)
        | Value.Eigh { a; _ }, (w, v) ->
            K.eigh ~w:(one w) ~v:(one v) (one a)
        | Value.Eig_values { a; _ }, w -> K.eig ~w:(one w) (one a)
        | Value.Eig { a; _ }, (w, v) -> K.eig ~w:(one w) ~v:(one v) (one a)
        | Value.Solve_triangular { triangle; adjoint; unit_diagonal; a; b }, y
          ->
            K.solve_triangular triangle ~adjoint ~unit_diagonal ~dst:(one y)
              (one a) (one b))
    | Value.Assemble { fill; pieces; _ } ->
        let pieces =
          Array.of_list (List.map (fun (rs, x) -> (rs, one x)) pieces)
        in
        K.assemble ~fill ~dst:(one r) pieces
    | Value.Copy x -> map1 (module K) Copy ~dst:(one r) (one x)
    | Value.Move _ | Value.Bitcast _ | Value.Place _ | Value.Check _ ->
        invalid_arg "Exec.call: an operation no kernel computes"
  in
  match answer with
  | Done -> true
  | Declined -> false
  | refusal ->
      let (Prim.Operands xs) = Prim.operands op in
      let operand (Value.Any x) = A.Any (one x) in
      A.refused by refusal
        (Array.to_list (dsts op r) @ List.map operand xs)

(* Raises for [op], whose kernels on [d] declined it. A routine is named by
   its routine, a reduction and a scan by what they compute, as
   ["Reduce Sum and Moments"], and a one-node map by its node, its result's
   dtype listed first. *)
let refused : type r. by:string -> kernels:string -> Rig.t -> r Value.prim -> 'a
    =
 fun ~by ~kernels d op ->
  let dts = ref [] in
  Prim.iteri (fun _ x -> dts := D.Any (Prim.dtype x) :: !dts) op;
  let dts = List.rev !dts in
  match op with
  | Value.Linalg l -> refuses ~by ~kernels (Prim.routine_name l) d dts
  | Value.Reduce { reductions; _ } ->
      let name (r, _, _) = Prim.reduction_name r in
      let names = Array.to_list (Array.map name (Prim.reductions reductions)) in
      refuses ~by ~kernels ("Reduce " ^ String.concat " and " names) d dts
  | Value.Scan { reduction; _ } ->
      let r, _, _ = Prim.reduction reduction in
      refuses ~by ~kernels ("Scan " ^ Prim.reduction_name r) d dts
  | Value.Map { prog; _ } -> (
      match P.single prog with
      | Some node ->
          let out = P.dtype prog (Iarray.get prog.outs 0) in
          refuses ~by ~kernels (Prim.kind node) d (out :: dts)
      | None -> refuses ~by ~kernels (Prim.name op) d dts)
  | _ -> refuses ~by ~kernels (Prim.name op) d dts

(* A destination for a result of [dt] beside [x]: over [x]'s layout where it is
   C-contiguous from offset 0 and of [dt], fresh otherwise. *)
let destination (type v s w r) (dt : (w, r) D.t) (x : (v, s) A.t) : (w, r) A.t =
  let l = A.layout x in
  if L.is_contiguous l && L.offset l = 0 && D.equal dt (A.dtype x) then
    fresh_over dt l (A.device x)
  else A.create (A.device x) dt (L.shape l)

(* The kernels of [x]'s set where [x] is a live array. *)
let live_kernels (type v s d) (x : (v, s, d) Value.t) =
  match x with
  | Value.Array r when r.dead == Value.Live ->
      Devices.kernels (Devices.set r.at)
  | _ -> None

(* Whether [y] is a live array at [x]'s placement, [x] an array. *)
let beside (type v s w r d) (x : (v, s, d) Value.t) (y : (w, r, d) Value.t) =
  match (x, y) with
  | Value.Array rx, Value.Array ry -> rx.at == ry.at && ry.dead == Value.Live
  | _ -> false

(* The array [load] reads. *)
let any (load : Nx_kernel.load) =
  match load with Plain a -> A.Any a | Padded { x; _ } -> A.Any x

(* The map [p] of one result computed by [K] into [dst] at [at], over
   [loads]. A decline raises as {!refused} names a map, and a refusal naming
   [by]. *)
let mapped (type w r d) ~by (module K : Nx_kernel.S) (at : d Devices.placement)
    p (dst : (w, r) A.t) loads : (w, r, d) Value.t =
  match K.map p ~dsts:[| A.Any dst |] loads with
  | Done -> Value.Array { at; a = dst; dead = Value.Live }
  | Declined -> (
      let ins = Iarray.to_list p.ins in
      match P.single p with
      | Some node ->
          declined ~by ~kernels:K.name node (A.device dst)
            (D.Any (A.dtype dst) :: ins)
      | None -> refuses ~by ~kernels:K.name "Map" (A.device dst) ins)
  | refusal ->
      A.refused by refusal (A.Any dst :: List.map any (Array.to_list loads))

(* Whether [loads] are live arrays read plainly at physically one placement,
   the first's: a map over them computes on its one device, where its route
   would read them and put its results ({!Prim.route}). *)
let beside_first (type d) (loads : d Value.load iarray) =
  let n = Iarray.length loads in
  n > 0
  &&
  match Iarray.get loads 0 with
  | Value.Plain (Value.Array f) when f.dead == Value.Live ->
      let at = f.at and i = ref 1 in
      while
        !i < n
        &&
        match Iarray.get loads !i with
        | Value.Plain (Value.Array r) -> r.at == at && r.dead == Value.Live
        | Value.Plain _ | Value.Padded _ -> false
      do
        incr i
      done;
      !i = n
  | Value.Plain _ | Value.Padded _ -> false

(* The map [op] over loads {!beside_first} holds, computed as {!windows}
   computes it at their placement without its route: its rule checked, its
   results fresh there, the set's kernels called once. A map of one result,
   as an interpreter's one-node map is, builds the kernels' loads directly
   and computes as {!map} does, by {!mapped}. *)
let local_in : type r. by:string -> L.t array -> r Value.prim -> r =
 fun ~by layouts op ->
  match op with
  | Value.Map { layout; prog; outs = [ dt ]; loads; _ } -> (
      match Iarray.get loads 0 with
      | Value.Plain (Value.Array { at; a; _ }) ->
          let kernels = kernels_of ~by ~op:"Map" (Devices.set at) in
          let n = Iarray.length loads in
          let arrays = Array.make n (Nx_kernel.Plain a) in
          for i = 1 to n - 1 do
            match Iarray.get loads i with
            | Value.Plain (Value.Array r) -> arrays.(i) <- Nx_kernel.Plain r.a
            | Value.Plain _ | Value.Padded _ -> invalid_arg "Exec.local"
          done;
          let dst = fresh_over dt layout (A.device a) in
          (mapped ~by kernels at prog dst arrays, ())
      | Value.Plain _ | Value.Padded _ -> invalid_arg "Exec.local")
  | Value.Map { loads; _ } -> (
      match Iarray.get loads 0 with
      | Value.Plain (Value.Array { at; a; _ }) ->
          let p = at in
          let r = Prim.results_at p layouts (fun k f -> alloc k f) op in
          let kernels = kernels_of ~by ~op:"Map" (Devices.set at) in
          if call ~by kernels op r then r
          else
            let (module K : Nx_kernel.S) = kernels in
            refused ~by ~kernels:K.name (A.device a) op
      | Value.Plain _ | Value.Padded _ -> invalid_arg "Exec.local")
  | _ -> invalid_arg "Exec.local: not a map"

let local ~by op = local_in ~by (Prim.layouts ~by op) op

let is_local : type r. r Value.prim -> bool = function
  | Value.Map { loads; _ } -> beside_first loads
  | _ -> false

(* Windows *)

(* [op] into its results [r] on device [k] of [set] by the kernels. A decline
   raises. *)
let on_device : type r.
    by:string ->
    (module Nx_kernel.S) ->
    unit Devices.t ->
    r ->
    int ->
    r Value.prim ->
    unit =
 fun ~by kernels set r k op ->
  if not (call ~by kernels op r) then
    let (module K : Nx_kernel.S) = kernels in
    refused ~by ~kernels:K.name (Devices.rig set k) op

(* Placing *)

(* [a]'s elements in window [w]: [a] itself when [w] is all of it. *)
let slice a (w : M.range iarray) =
  let l = A.layout a in
  let n = Iarray.length w in
  let whole = ref (n = L.rank l) and i = ref 0 in
  while !whole && !i < n do
    let r = Iarray.get w !i in
    whole := r.start = 0 && r.count = L.dim l !i;
    incr i
  done;
  if !whole then a
  else
    match A.move (M.Slice w) a with
    | Some v -> v
    | None -> invalid_arg "Exec: a slice of an array is a view"

let hosted a = Rig.reaches Rig.host (A.device a)

(* [a] over its own memory on [d], where the host addresses both: a device that
   only maps host memory takes a copy instead, so its work reads memory of its
   own. *)
let share d a =
  if Rig.reaches Rig.host d && hosted a then A.borrow d a else None

(* [a] on [d]: itself, its memory shared, or a copy of its elements. *)
let bring d a =
  if Rig.equal (A.device a) d then a
  else match share d a with Some b -> b | None -> A.to_device d a

(* The eager engine *)

(* [x]'s piece on its [j]th device: its array there. *)
let piece (type v s d) (x : (v, s, d) Value.t) j at : (v, s, d) Value.t =
  match x with
  | Value.Array _ -> x
  | Value.Shards { arrays; _ } ->
      Value.Array { at; a = Iarray.get arrays j; dead = Value.Live }
  | Value.Traced _ -> invalid_arg "Exec: a traced value has no arrays"
  | Value.Handle _ ->
      invalid_arg "Exec: a donated value is read as its consumer"

(* A view of [x]'s array: [x] itself where the view is all of it. Its memory
   stays unshared: a step of a partition reads the view and keeps it in no
   value it returns but a piece of its result. *)
let view (type v s d) mv (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let at = Prim.placement x and a = one x in
  let v =
    match (mv : M.t) with
    | Slice w -> slice a w
    | _ -> (
        match A.move mv a with
        | Some v -> v
        | None -> invalid_arg "Exec: a piece's layout expresses its view")
  in
  if v == a then x else Value.Array { at; a = v; dead = Value.Live }

(* The window [w] of the piece [x]. *)
let view_of_piece (x, w) = slice (one x) (Iarray.of_array w)

let rec on_device_of d = function
  | [] -> None
  | ((x, _) as p) :: ps ->
      if Rig.equal (A.device (one x)) d then Some (view_of_piece p)
      else on_device_of d ps

let rec borrowed d = function
  | [] -> None
  | p :: ps -> (
      match share d (view_of_piece p) with
      | Some a -> Some a
      | None -> borrowed d ps)

(* One of the windows [ps] on [at]'s device: a view of one there, else a
   borrow of one whose memory the device addresses, else a copy of the
   first. *)
let bring_one (type v s d e) at (ps : ((v, s, d) Value.t * M.range array) list)
    : (v, s, e) Value.t =
  let d = rig_of at in
  let a =
    match on_device_of d ps with
    | Some a -> a
    | None -> (
        match borrowed d ps with
        | Some a -> a
        | None -> A.to_device d (view_of_piece (List.hd ps)))
  in
  Value.Array { at; a; dead = Value.Live }

(* The boxes gathered into a fresh C-contiguous array on [at]'s device. A dtype
   narrower than a byte gathers on the host where the host does not address
   the device's memory, since a run may start inside a byte, and is then
   brought to the device. *)
let gather (type v s d e) at (dt : (v, s) D.t) shape
    (boxes : (M.range array * (v, s, d) Value.t * M.range array) list) :
    (v, s, e) Value.t =
  let d = rig_of at in
  let on =
    if D.bits dt mod 8 = 0 || Rig.reaches Rig.host d then d else Rig.host
  in
  let dst = A.create on dt shape in
  List.iter
    (fun (w, x, b) ->
      let src = slice (one x) (Iarray.of_array b) in
      A.transfer ~src ~dst:(slice dst (Iarray.of_array w)))
    boxes;
  Value.Array { at; a = bring d dst; dead = Value.Live }

(* An array whose writer is done, as an [iarray], without a copy. *)
external freeze : 'a array -> 'a iarray = "%opaque"

(* The array of the device at position [j] of [p]'s grid, where its window
   is empty: memory of no bytes of its own, where a view of a source would
   hold, and share, all of that source's memory. *)
let empty_window p shape dt j =
  let w = Devices.window ~by:"Exec.assemble" p shape j in
  let k = (Grid.devices (Devices.grid p)).(j) in
  A.create (Devices.rig (Devices.set p) k) dt (Route.extents w)

let piece_array p shape dt ps j =
  match ps.(j) with Some v -> one v | None -> empty_window p shape dt j

(* The value over [ps]' arrays, an empty window's fresh. *)
let assemble (type v s d e) (p : e Devices.placement) shape (dt : (v, s) D.t)
    (ps : (v, s, d) Value.t option array) : (v, s, e) Value.t =
  let first = piece_array p shape dt ps 0 in
  let n = Array.length ps in
  if n = 1 then Value.Array { at = p; a = first; dead = Value.Live }
  else begin
    let arrays = Array.make n first in
    for j = 1 to n - 1 do
      arrays.(j) <- piece_array p shape dt ps j
    done;
    Value.Shards { at = p; shape; arrays = freeze arrays; dead = Value.Live }
  end

(* Placing one array *)

(* The array [a] of a value at [at], placed at the one-device placement [p]:
   itself where [at] is [p], else as {!eager} places a value whose one piece
   holds the whole, without building the partition. A value of no elements
   is memory of no bytes of its own, where a borrow would share [a]'s. *)
let one_array at a p =
  if Devices.equal at p then a
  else
    let d = rig_of p and l = A.layout a in
    if L.numel l = 0 then A.create d (A.dtype a) (L.shape l) else bring d a

let to_one (type v s d e) (at : d Devices.placement) (a : (v, s) A.t)
    (p : e Devices.placement) : (v, s, e) Value.t =
  Value.Array { at = p; a = one_array at a p; dead = Value.Live }

let overlaps a b = Rig.Buffer.overlaps (A.buffer a) (A.buffer b)

(* Whether an array of [xs] from position [i] overlaps [b]. *)
let rec overlaps_any xs b i =
  i < Iarray.length xs
  && (overlaps (Iarray.get xs i) b || overlaps_any xs b (i + 1))

(* [to_one], its array marked shared where it lies over [a]'s memory, as
   {!place} marks it. *)
let placed_one (type v s d e) (at : d Devices.placement) (a : (v, s) A.t)
    (p : e Devices.placement) : (v, s, e) Value.t =
  let b = one_array at a p in
  if overlaps a b then Rig.Claim.share (A.buffer b);
  Value.Array { at = p; a = b; dead = Value.Live }

(* Creations *)

(* The creation [op], a map of no loads on device [p], whose set has no
   kernels: its elements come from numbers alone. It is computed by the host's
   kernels, then placed at [p], which holds a copy, or a borrow, which shares
   the host's memory, where it addresses it ({!place}). Its bits are the host
   form's. *)
let created : type r.
    by:string -> unit Devices.placement -> L.t array -> r Value.prim -> r =
 fun ~by p layouts op ->
  let host = Devices.on Devices.host in
  let set = Devices.set host in
  let r = Prim.results_at host layouts (fun k f -> alloc k f) op in
  on_device ~by (kernels_of ~by ~op:(Prim.name op) set) set r 0 op;
  let arrays = Prim.result_arrays op r in
  Prim.results_at p layouts
    (fun k (f : _ Value.form) ->
      placed_one host (A.expect f.dtype arrays.(k).(0)) p)
    op

let windows : type r.
    by:string ->
    ?into:A.any ->
    unit Devices.placement ->
    L.t array ->
    r Value.prim ->
    r =
 fun ~by ?into p layouts op ->
  match op with
  | Value.Move (mv, x) ->
      let set = Devices.set p in
      Value.Array
        { at = p; a = as_view ~by set A.move mv (one x); dead = Value.Live }
  | Value.Bitcast (dt, x) ->
      let set = Devices.set p in
      Value.Array
        { at = p; a = as_view ~by set A.bitcast dt (one x); dead = Value.Live }
  | Value.Place _ | Value.Check _ ->
      invalid_arg "Exec.windows: a placement or a check"
  | Value.Map { at = Some _; loads; _ }
    when Iarray.length loads = 0 && Devices.bare p ->
      created ~by p layouts op
  | Value.Map _ | Value.Reduce _ | Value.Scan _ | Value.Gather _
  | Value.Scatter _ | Value.Sort _ | Value.Fold _ | Value.Fft _ | Value.Linalg _
  | Value.Assemble _ | Value.Contract _ | Value.Copy _ ->
      (* Fresh results on the route's one device, computed there by its
         kernels over the operands, which lie there. *)
      let r = Prim.results_at p layouts (fun k f -> alloc ?into k f) op in
      let set = Devices.set p in
      let kernels = kernels_of ~by ~op:(Prim.name op) set in
      on_device ~by kernels set r (Option.get (Grid.one (Devices.grid p))) op;
      r

(* [windows], over the memory [into]'s array holds, or {!local} where the
   two give one result: a map beside its loads into fresh results, whose
   one result, if alone, is laid out C-contiguous from offset 0, as
   [windows] lays it. *)
let run : type r d.
    by:string ->
    ?into:d Value.any ->
    origin:Prim.origin ->
    unit Devices.placement ->
    L.t array ->
    r Value.prim ->
    r =
 fun ~by ?into ~origin p layouts op ->
  let op = Prim.offset op origin in
  match (into, op) with
  | None, Value.Map { layout; outs; _ }
    when is_local op
         && (match outs with
            | [ _ ] -> L.is_contiguous layout && L.offset layout = 0
            | _ -> true) ->
      local_in ~by layouts op
  | None, _ -> windows ~by p layouts op
  | Some (Value.Any v), _ -> windows ~by ~into:(A.Any (one v)) p layouts op

let eager = { Partition.run; piece; view; bring = bring_one; gather; assemble }

(* Placing values *)

(* One array placed on one device needs no partition ({!to_one}). *)
let value : type v s d e.
    by:string -> e Devices.placement -> (v, s, d) Value.t -> (v, s, e) Value.t
    =
 fun ~by p x ->
  match x with
  | Value.Traced _ ->
      invalid_arg "Exec.value: a traced value is placed by its owner"
  | Value.Handle _ ->
      invalid_arg "Exec.value: a donated value is placed as its consumer"
  | Value.Array { at; a; _ } when Grid.count (Devices.grid p) = 1 ->
      to_one at a p
  | Value.Array _ | Value.Shards _ -> Partition.place eager ~by p x

let place : type v s d e.
    by:string -> e Devices.placement -> (v, s, d) Value.t -> (v, s, e) Value.t
    =
 fun ~by p x ->
  match x with
  | Value.Array { at; a; _ } when Grid.count (Devices.grid p) = 1 ->
      placed_one at a p
  | _ ->
      let y = value ~by p x in
      (* A value over [x]'s memory, kept or borrowed, shares it. *)
      let xs = Prim.arrays x and ys = Prim.arrays y in
      for j = 0 to Iarray.length ys - 1 do
        let b = Iarray.get ys j in
        if overlaps_any xs b 0 then Rig.Claim.share (A.buffer b)
      done;
      y

(* The fast path *)

let map ~by kernels at p dt x loads =
  mapped ~by kernels at p (destination dt x) loads

(* The contraction of live arrays on one device, at physically one placement
   whose set has kernels, computed by them without building the operation into
   a result of the shape its rule gives; a decline raises. [None] elsewhere and
   where the pairs do not fit the operands: the operation decides those. *)
let contract (type v s a b c e d) ~by ~batch ~contracting ~acc
    (dt : (v, s) D.t) (x : (a, b, d) Value.t) (y : (c, e, d) Value.t)
    (init : (v, s, d) Value.t option) : (v, s, d) Value.t option =
  match (x, y) with
  | Value.Array rx, Value.Array ry
    when rx.at == ry.at && rx.dead == Value.Live && ry.dead == Value.Live -> (
      let at = rx.at in
      let set = Devices.set at in
      let array =
        match init with
        | None -> Some None
        | Some (Value.Array ri) when ri.at == at && ri.dead == Value.Live ->
            Some (Some ri.a)
        | Some _ -> None
      in
      let shape =
        Nx_kernel.Shape.contract ~batch ~contracting ~segments:None
          (Iarray.to_array (L.shape (A.layout rx.a)), None)
          (Iarray.to_array (L.shape (A.layout ry.a)), None)
      in
      match (array, shape, Grid.one (Devices.grid at), Devices.kernels set) with
      | Some array, Ok shape, Some k, Some (module K) -> (
          let dst = A.create (Devices.rig set k) dt (Iarray.of_array shape) in
          match
            K.contract ~batch ~contracting ~segments:None ~acc ~dst
              ~init:array
              { array = A.Any rx.a; scale = None }
              { array = A.Any ry.a; scale = None }
          with
          | Done -> Some (Value.Array { at; a = dst; dead = Value.Live })
          | Declined ->
              let dtype i = D.Any (A.dtype i) in
              let init = Option.to_list (Option.map dtype array) in
              refuses ~by ~kernels:K.name "Contract" (A.device dst)
                (dtype rx.a :: dtype ry.a :: init)
          | refusal ->
              let init = Option.to_list (Option.map (fun i -> A.Any i) array) in
              A.refused by refusal
                (A.Any dst :: A.Any rx.a :: A.Any ry.a :: init))
      | _ -> None)
  | _ -> None
