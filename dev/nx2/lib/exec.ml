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

(* The array a kernel's load reads. *)
let load_array : Nx_kernel.load -> A.any = function
  | Nx_kernel.Plain a -> A.Any a
  | Nx_kernel.Padded { x; _ } -> A.Any x

(* As [ran], for a loop over [loads]. *)
let ran_loads ~by ?(reads = []) answer dsts loads =
  match (answer : A.answer) with
  | Done -> true
  | Declined -> false
  | refusal ->
      A.refused by refusal
        (Array.to_list dsts @ List.map load_array (Array.to_list loads) @ reads)

(* Values from arrays *)

let make = Prim.of_arrays

(* A view made over [a]'s memory, outside any handle chain: the memory is shared
   from then on, so that no donation of [a]'s value writes what the view
   reads. *)
let view_of a = Rig.Claim.share (A.buffer a)

let extents (w : M.range array) =
  Iarray.init (Array.length w) (fun i -> (w.(i) : M.range).count)

(* Fresh C-contiguous arrays of a value of [shape] at [p]: each device's
   window. *)
let fresh ~by p dt shape =
  let set = Devices.set p in
  match Grid.one (Devices.grid p) with
  | Some k -> ([| A.create (Devices.rig set k) dt shape |] : _ iarray)
  | None ->
      let devices = Grid.devices (Devices.grid p) in
      let shape = Iarray.to_array shape in
      Iarray.init (Array.length devices) (fun j ->
          A.create (Devices.rig set devices.(j)) dt
            (extents (Devices.window ~by p shape j)))

(* A maker of fresh results at their form's placement: the first into [into]
   where it is given. *)
let alloc (type v s d) ~by ?into k (f : (v, s, d) Value.form) :
    (v, s, d) Value.t =
  let p =
    match f.placement with
    | Some p -> p
    | None -> invalid_arg "Exec.alloc: a value of every set"
  in
  match into with
  | Some a when k = 0 ->
      Value.Array { at = p; a = A.expect f.dtype a; dead = Value.Live }
  | Some _ | None ->
      let shape = L.shape f.layout in
      make p ~shape:(Iarray.to_array shape) (fresh ~by p f.dtype shape)

(* Programs on one device *)

(* [prog] over [loads] into [dsts], by the kernels' map. [false] where the
   kernels decline. *)
let map_on ~by (module K : Nx_kernel.S) (prog : P.t) loads dsts =
  ran_loads ~by (K.map prog ~dsts loads) dsts loads

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

(* [a] moved by [mv]: a view over [a]'s memory where a layout expresses it, a
   copy moved otherwise. *)
let moved ~by set mv a =
  match A.move mv a with
  | Some v ->
      view_of a;
      v
  | None ->
      let kernels = kernels_of ~by ~op:"Copy" set in
      Option.get (A.move mv (copy_array ~by kernels a))

(* [a] bitcast to [dt], as {!moved}. *)
let cast ~by set dt a =
  match A.bitcast dt a with
  | Some v ->
      view_of a;
      v
  | None ->
      let kernels = kernels_of ~by ~op:"Copy" set in
      Option.get (A.bitcast dt (copy_array ~by kernels a))

(* [mv] on one shard of a value of [shape] cut as [cuts], the shard of shape
   [local]: a cut axis is one the movement keeps in order or regroups in tiles
   that divide both extents ({!Route.moved}), so where the movement names its
   extent, the shard's own extent along it, or its tile of the regrouped
   extent, replaces it. *)
let localize (mv : M.t) shape cuts local : M.t =
  let cut a = Array.exists (fun (c, _) -> c = a) cuts in
  let target s' =
    let s' = Iarray.to_array s' in
    Array.iter
      (fun (a, t) ->
        match Route.moved mv shape a with
        | Becomes a' -> s'.(a') <- Iarray.get local a
        | Regroups { axis; extent } -> s'.(axis) <- extent / t
        | Whole | Reduced _ | Scanned _ -> ())
      cuts;
    Iarray.of_array s'
  in
  match mv with
  | Permute _ | Window _ -> mv
  | Reshape s' -> Reshape (target s')
  | Broadcast s' -> Broadcast (target s')
  | Slice rs ->
      Slice
        (Iarray.mapi
           (fun a (r : M.range) ->
             if cut a then { M.start = 0; count = Iarray.get local a; step = 1 }
             else r)
           rs)

(* Whether [p] cuts no axis: one device, or the whole on each. *)
let uncut p = not (Grid.is_cut (Devices.grid p))

(* The one array of [x], a value on one device. *)
let one (type v s d) (x : (v, s, d) Value.t) = Iarray.get (Prim.arrays x) 0

(* The load a kernel reads of [l], over a value on one device. *)
let kernel_load (type d) (l : d Value.load) : Nx_kernel.load =
  match l with
  | Value.Plain x -> Nx_kernel.Plain (one x)
  | Value.Padded { x; fill; pad } -> Nx_kernel.Padded { x = one x; fill; pad }

let kernel_loads loads =
  Array.init (Iarray.length loads) (fun i -> kernel_load (Iarray.get loads i))

(* The arrays at position [j] of [x] and of the results [r] of [op]. *)
let nth (type v s d) j (x : (v, s, d) Value.t) = Iarray.get (Prim.arrays x) j
let dsts_at op r j = Array.map (fun per -> per.(j)) (Prim.result_arrays op r)

(* [op]'s kernel over its operands' arrays, into the results [r]'s arrays at
   position [j]: [false] where it declines. A refusal raises naming [by]. *)
let call : type r.
    by:string -> (module Nx_kernel.S) -> r Value.prim -> r -> int -> bool =
 fun ~by kernels op r j ->
  let (module K : Nx_kernel.S) = kernels in
  match op with
  | Value.Map { prog; loads; _ } ->
      map_on ~by kernels prog (kernel_loads loads) (dsts_at op r j)
  | Value.Reduce { layout; axes; prog; reductions; loads } ->
      let dsts = dsts_at op r j and loads = kernel_loads loads in
      let rs = Prim.reductions reductions
      and shape = Iarray.to_array (L.shape layout) in
      ran_loads ~by (K.reduce prog ~shape ~axes rs ~dsts loads) dsts loads
  | Value.Scan { axis; prog; reduction; loads; carry; _ } ->
      let dsts = dsts_at op r j and loads = kernel_loads loads in
      let reduction = Prim.reduction reduction in
      let carry = Option.map (fun (Value.Any c) -> A.Any (one c)) carry in
      let reads = Option.to_list carry in
      ran_loads ~by ~reads (K.scan ?carry prog ~axis reduction ~dsts loads) dsts
        loads
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
      let dst = nth j r and a = one a and b = one b in
      let init = Option.map one init in
      let scale : _ Value.scale -> Nx_kernel.scale = function
        | Exponent x -> Exponent (one x)
        | Value x -> Value (one x)
      in
      let a_scale = Option.map scale a_scale in
      let b_scale = Option.map scale b_scale in
      let segments =
        Option.map
          (fun (g : _ Nx_kernel.segments) : _ Nx_kernel.segments ->
            match g with
            | Rows r -> Rows { r with offsets = one r.offsets }
            | Contracting r -> Contracting { r with offsets = one r.offsets })
          segments
      in
      let array : Nx_kernel.scale -> A.any = function
        | Exponent x -> A.Any x
        | Value x -> A.Any x
      in
      (* The arrays a refusal names, in the operation's operand order. *)
      let ops =
        match (init, a_scale, b_scale, segments) with
        | None, None, None, None -> [| A.Any a; A.Any b |]
        | Some i, None, None, None -> [| A.Any a; A.Any b; A.Any i |]
        | _ ->
            let opt f o = Option.to_list (Option.map f o) in
            let offsets = function
              | Nx_kernel.Rows { offsets; _ } | Contracting { offsets; _ } ->
                  A.Any offsets
            in
            Array.of_list
              (List.concat
                 [
                   [ A.Any a; A.Any b ];
                   opt (fun i -> A.Any i) init;
                   opt array a_scale;
                   opt array b_scale;
                   opt offsets segments;
                 ])
      in
      ran ~by
        (K.contract ~batch ~contracting ~segments ~acc ~dst ~init
           { array = A.Any a; scale = a_scale }
           { array = A.Any b; scale = b_scale })
        [| A.Any dst |] ops
  | Value.Gather { axis; idx; x } ->
      let dst = nth j r and i = one idx and x = one x in
      ran ~by (K.gather ~axis ~dst i x) [| A.Any dst |] [| A.Any i; A.Any x |]
  | Value.Scatter { combine; unique; axis; idx; updates; into } ->
      let dst = nth j r and into = one into in
      let i = one idx and u = one updates in
      ran ~by
        (K.scatter combine ~unique ~axis ~dst ~into i u)
        [| A.Any dst |]
        [| A.Any into; A.Any i; A.Any u |]
  | Value.Sort { axis; descending; x; _ } ->
      let values, positions = r in
      let v = nth j values and q = nth j positions and a = one x in
      ran ~by
        (K.sort ~axis ~descending ~values:v ~positions:q a)
        [| A.Any v; A.Any q |] [| A.Any a |]
  | Value.Fold { pad; x; _ } ->
      let dst = nth j r and v = one x in
      ran ~by (K.fold pad ~dst v) [| A.Any dst |] [| A.Any v |]
  | Value.Fft (Value.C2c { direction; axes; x }) ->
      let dst = nth j r and x = one x in
      ran ~by (K.c2c direction ~axes ~dst x) [| A.Any dst |] [| A.Any x |]
  | Value.Fft (Value.R2c { axes; x; _ }) ->
      let dst = nth j r and x = one x in
      ran ~by (K.r2c ~axes ~dst x) [| A.Any dst |] [| A.Any x |]
  | Value.Fft (Value.C2r { axes; x; _ }) ->
      let dst = nth j r and x = one x in
      ran ~by (K.c2r ~axes ~dst x) [| A.Any dst |] [| A.Any x |]
  | Value.Linalg l -> (
      match (l, r) with
      | Value.Cholesky { triangle; a }, y ->
          let dst = nth j y and a = one a in
          ran ~by (K.cholesky triangle ~dst a) [| A.Any dst |] [| A.Any a |]
      | Value.Lu a, (lu, pivots, perm) ->
          let lu = nth j lu and pivots = nth j pivots and perm = nth j perm in
          let a = one a in
          ran ~by (K.lu ~lu ~pivots ~perm a)
            [| A.Any lu; A.Any pivots; A.Any perm |]
            [| A.Any a |]
      | Value.Qr { a; _ }, (q, r) ->
          let q = nth j q and r = nth j r and a = one a in
          ran ~by (K.qr ~q ~r a) [| A.Any q; A.Any r |] [| A.Any a |]
      | Value.Svd_values { a; _ }, s ->
          let s = nth j s and a = one a in
          ran ~by (K.svd ~s a) [| A.Any s |] [| A.Any a |]
      | Value.Svd { a; _ }, (s, u, vh) ->
          let s = nth j s and u = nth j u and vh = nth j vh and a = one a in
          ran ~by
            (K.svd ~s ~vectors:(~u, ~vh) a)
            [| A.Any s; A.Any u; A.Any vh |]
            [| A.Any a |]
      | Value.Eigh_values { a; _ }, w ->
          let w = nth j w and a = one a in
          ran ~by (K.eigh ~w a) [| A.Any w |] [| A.Any a |]
      | Value.Eigh { a; _ }, (w, v) ->
          let w = nth j w and v = nth j v and a = one a in
          ran ~by (K.eigh ~w ~v a) [| A.Any w; A.Any v |] [| A.Any a |]
      | Value.Eig_values { a; _ }, w ->
          let w = nth j w and a = one a in
          ran ~by (K.eig ~w a) [| A.Any w |] [| A.Any a |]
      | Value.Eig { a; _ }, (w, v) ->
          let w = nth j w and v = nth j v and a = one a in
          ran ~by (K.eig ~w ~v a) [| A.Any w; A.Any v |] [| A.Any a |]
      | Value.Solve_triangular { triangle; adjoint; unit_diagonal; a; b }, y ->
          let dst = nth j y and a = one a and b = one b in
          ran ~by
            (K.solve_triangular triangle ~adjoint ~unit_diagonal ~dst a b)
            [| A.Any dst |] [| A.Any a; A.Any b |])
  | Value.Assemble { fill; pieces; _ } ->
      let dst = nth j r in
      let pieces =
        Array.of_list (List.map (fun (rs, x) -> (rs, one x)) pieces)
      in
      ran ~by
        (K.assemble ~fill ~dst pieces)
        [| A.Any dst |]
        (Array.map (fun (_, v) -> A.Any v) pieces)
  | Value.Copy x ->
      let dst = nth j r and x = one x in
      ran ~by (map1 (module K) Copy ~dst x) [| A.Any dst |] [| A.Any x |]
  | Value.Move _ | Value.Bitcast _ | Value.Place _ | Value.Check _ ->
      invalid_arg "Exec.call: an operation no kernel computes"

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
      let name (r, _) = Prim.reduction_name r in
      let names = Array.to_list (Array.map name (Prim.reductions reductions)) in
      refuses ~by ~kernels ("Reduce " ^ String.concat " and " names) d dts
  | Value.Scan { reduction; _ } ->
      let r, _ = Prim.reduction reduction in
      refuses ~by ~kernels ("Scan " ^ Prim.reduction_name r) d dts
  | Value.Map { prog; _ } -> (
      match P.single prog with
      | Some node ->
          let out = P.dtype prog (Iarray.get prog.outs 0) in
          refuses ~by ~kernels (Prim.kind node) d (out :: dts)
      | None -> refuses ~by ~kernels (Prim.name op) d dts)
  | _ -> refuses ~by ~kernels (Prim.name op) d dts

(* A fresh array of [dt] on [d] laid out as [l], C-contiguous from offset 0.
   A dtype narrower than a byte is created, whose spare bits in its last byte
   are zero. *)
let fresh_over (type w r) (dt : (w, r) D.t) l d : (w, r) A.t =
  if D.bits dt >= 8 then A.v dt l (Rig.Buffer.create d (D.bytes dt (L.numel l)))
  else A.create d dt (L.shape l)

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
let local : type r. by:string -> r Value.prim -> r =
 fun ~by op ->
  let layouts = Prim.layouts ~by op in
  match op with
  | Value.Map { layout; prog; outs = [ dt ]; loads } -> (
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
          let p = Some (Devices.rebrand at) in
          let r = Prim.results_at p layouts (fun k f -> alloc ~by k f) op in
          let kernels = kernels_of ~by ~op:"Map" (Devices.set at) in
          if call ~by kernels op r 0 then r
          else
            let (module K : Nx_kernel.S) = kernels in
            refused ~by ~kernels:K.name (A.device a) op
      | Value.Plain _ | Value.Padded _ -> invalid_arg "Exec.local")
  | _ -> invalid_arg "Exec.local: not a map"

let is_local : type r. r Value.prim -> bool = function
  | Value.Map { loads; _ } -> beside_first loads
  | _ -> false

(* Windows *)

(* [local], an operation's part on device [k] of [set], into its results [r]
   at position [j] by the kernels. A decline raises. *)
let on_device : type r.
    by:string ->
    (module Nx_kernel.S) ->
    unit Devices.t ->
    r ->
    int ->
    int ->
    r Value.prim ->
    unit =
 fun ~by kernels set r j k local ->
  if not (call ~by kernels local r j) then
    let (module K : Nx_kernel.S) = kernels in
    refused ~by ~kernels:K.name (Devices.rig set k) local

(* The one result of [op] over the arrays [views ()], one per device of [xp] in
   order: at [route]'s result, whose devices hold them. *)
let shard_views : type v s d.
    unit Route.t ->
    L.t array ->
    (v, s, d) Value.t Value.prim ->
    d Devices.placement ->
    (unit -> (v, s) A.t iarray) ->
    (v, s, d) Value.t =
 fun route layouts op xp views ->
  Prim.results_at (Some route.result) layouts
    (fun _ f ->
      let vs = views () in
      let p = Option.get f.placement in
      let shape = Iarray.to_array (L.shape f.layout) in
      if Devices.equal p xp then
        make p ~shape (Iarray.map (fun v -> A.expect f.dtype (A.Any v)) vs)
      else
        let xdev = Grid.devices (Devices.grid xp) in
        let devices = Grid.devices (Devices.grid p) in
        make p ~shape
          (Iarray.init (Array.length devices) (fun i ->
               let j = Option.get (Array.find_index (( = ) devices.(i)) xdev) in
               A.expect f.dtype (A.Any (Iarray.get vs j)))))
    op

let windows : type r.
    by:string -> ?into:A.any -> unit Route.t -> L.t array -> r Value.prim -> r
    =
 fun ~by ?into route layouts op ->
  (match route.compute with
  | Windows -> ()
  | Partials -> invalid_arg "Exec.windows: a route that computes by partials");
  match op with
  | Value.Move (mv, x) ->
      let xp = Prim.placement x and xs = Prim.arrays x in
      let set = Devices.set xp in
      let move =
        if uncut xp then moved ~by set mv
        else
          let shape = Prim.shape x and cuts = Grid.cuts (Devices.grid xp) in
          fun a ->
            moved ~by set (localize mv shape cuts (L.shape (A.layout a))) a
      in
      shard_views route layouts op xp (fun () -> Iarray.map move xs)
  | Value.Bitcast (dt, x) ->
      let xp = Prim.placement x and xs = Prim.arrays x in
      let set = Devices.set xp in
      shard_views route layouts op xp (fun () ->
          Iarray.map (cast ~by set dt) xs)
  | Value.Place _ | Value.Check _ ->
      invalid_arg "Exec.windows: a placement or a check"
  | Value.Map _ | Value.Reduce _ | Value.Scan _ | Value.Gather _
  | Value.Scatter _ | Value.Sort _ | Value.Fold _ | Value.Fft _ | Value.Linalg _
  | Value.Assemble _ | Value.Contract _ | Value.Copy _ -> (
      (* Fresh results where the route places them, each device's window [w]
         of the first computed there by its kernels over each operand's
         window ({!Route.window}). *)
      let p = route.result in
      let r =
        Prim.results_at (Some p) layouts (fun k f -> alloc ~by ?into k f) op
      in
      let set = Devices.set p in
      let kernels = kernels_of ~by ~op:(Prim.name op) set in
      match Grid.one (Devices.grid p) with
      | Some k ->
          on_device ~by kernels set r 0 k op;
          r
      | None ->
          let rule = Prim.rule op
          and first = Iarray.to_array (L.shape layouts.(0)) in
          Array.iteri
            (fun j k ->
              let w = Devices.window ~by p first j in
              let view i x =
                let at = Devices.one (Devices.set (Prim.placement x)) k in
                let w' = Route.window rule ~operand:i (Prim.shape x) w in
                let a = Place.view ~by x k w' in
                Value.Array { at; a; dead = Value.Live }
              in
              on_device ~by kernels set r j k
                (Prim.local (Prim.mapi view op) w))
            (Grid.devices (Devices.grid p));
          r)

(* Placing *)

let place ~by p x =
  let y = Place.value ~by p x in
  (* A value over [x]'s memory, kept or borrowed, shares it. *)
  let xs = Prim.arrays x in
  let overlaps b a = Rig.Buffer.overlaps (A.buffer a) (A.buffer b) in
  Iarray.iter
    (fun b ->
      if Iarray.exists (overlaps b) xs then Rig.Claim.share (A.buffer b))
    (Prim.arrays y);
  y

let placer ~by p x =
  if Devices.equal (Prim.placement x) p then x else Place.value ~by p x

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
