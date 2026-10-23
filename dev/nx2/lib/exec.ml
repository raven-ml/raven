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

(* Raises naming its consumer where operand [i], [x], died since it was
   checked: a kernel's door refuses memory that a consumer on another domain
   took meanwhile, and its refusal names no one. A handle is the call's own,
   claimed, and named by its claim. *)
let recheck ~by i x = if not (Donate.is_handle x) then Prim.alive ~by i x

(* Raises [e], which [op]'s run raised, unless an operand died since it was
   checked ({!recheck}). *)
let rechecked ~by op e =
  let bt = Printexc.get_raw_backtrace () in
  Prim.iteri (fun i x -> recheck ~by i x) op;
  Printexc.raise_with_backtrace e bt

(* A kernel's answer: [Done], a refusal raised, or [Declined] passed on. *)
let done_or_declined ~by answer operands =
  match (answer : A.answer) with
  | Done -> true
  | Declined -> false
  | refusal -> A.refused by refusal operands

(* As [done_or_declined] over [dsts] then [ops], listing them only to raise. *)
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

(* Arrays kept untyped, one per device, at [dt]. *)
let typed dt (per : A.any array) =
  Iarray.init (Array.length per) (fun j -> A.expect dt per.(j))

(* Raises unless operand [i], [x], is alive and concrete or a constant: an
   interpretation receives every operation on its traced values before eager
   execution can. *)
let checked (type v s d) ~by i (x : (v, s, d) Value.t) =
  match x with
  | Value.Traced { owner; _ } ->
      invalid_argf "%s: a value traced by %s reached eager execution" by
        owner.name
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Deferred _ ->
      Prim.alive ~by i x

let is_concrete x = not (Prim.is_constant x)

(* A view made over [a]'s memory, outside any handle chain: the memory is shared
   from then on, so that no donation of [a]'s value writes what the view
   reads. *)
let view_of a = Rig.Claim.share (A.buffer a)

let extents (w : M.range array) = Array.map (fun (r : M.range) -> r.count) w

(* Fresh C-contiguous arrays of a value of [shape] at [p]: each device's
   window. *)
let fresh ~by p dt shape =
  let set = Devices.set p in
  match Grid.one (Devices.grid p) with
  | Some k -> ([| A.create (Devices.rig set k) dt shape |] : _ iarray)
  | None ->
      let devices = Grid.devices (Devices.grid p) in
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
      make p ~shape (fresh ~by p f.dtype shape)

(* Programs on one device *)

(* [node]'s kernel into [dst], over [ops]. *)
let apply_node (module K : Nx_kernel.S) node (A.Any dst) (ops : A.any array) =
  match (node : P.node) with
  | Const (_, bits) -> K.apply0 (Fill bits) ~dst
  | Coord c -> K.apply0 (Iota c) ~dst
  | Op1 (k, _, _) ->
      let (A.Any x) = ops.(0) in
      K.apply1 k ~dst x
  | Op2 (k, _, _) ->
      let (A.Any x) = ops.(0) in
      K.apply2 k ~dst x (A.expect (A.dtype x) ops.(1))
  | Op3 (k, _, _, _) ->
      (* Where and Fma give their second and third operands' dtype. *)
      let (A.Any c) = ops.(0) in
      let dt = A.dtype dst in
      K.apply3 k ~dst c (A.expect dt ops.(1)) (A.expect dt ops.(2))
  | In _ -> Declined

(* [prog] over [loads] into [dsts]: its kernel where it is one node over plain
   loads, else the kernels' map. [false] where the kernels decline. *)
let map_on ~by (module K : Nx_kernel.S) (prog : P.t) loads dsts =
  let plain = function
    | Nx_kernel.Plain _ -> true
    | Nx_kernel.Padded _ -> false
  in
  match Expand.single prog with
  | Some node when Array.for_all plain loads ->
      let ops = Array.map load_array loads in
      ran ~by (apply_node (module K) node dsts.(0) ops) dsts ops
  | Some _ | None -> ran_loads ~by (K.map prog ~dsts loads) dsts loads

(* [a] stored afresh, C-contiguous, on its device. *)
let copy_array ~by kernels a =
  let (module K : Nx_kernel.S) = kernels in
  let dst = A.create (A.device a) (A.dtype a) (L.shape (A.layout a)) in
  if not (done_or_declined ~by (K.apply1 Copy ~dst a) [ A.Any dst; A.Any a ])
  then
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
        | Becomes a' -> s'.(a') <- local.(a)
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
             if cut a then { M.start = 0; count = local.(a); step = 1 } else r)
           rs)

let unravel i shape =
  let r = Array.length shape in
  let idx = Array.make r 0 and k = ref i in
  for a = r - 1 downto 0 do
    idx.(a) <- !k mod shape.(a);
    k := !k / shape.(a)
  done;
  idx

(* Whether [p] cuts no axis: one device, or the whole on each. *)
let uncut p = not (Grid.is_cut (Devices.grid p))

(* The route of the constant operation [op] whose results are wanted at [p]:
   where it computes them and reads each operand ({!Route.at}). Where [p] cuts
   no axis, or the rule is elementwise, that is [p] for all. *)
let route_at : type r.
    by:string -> r Value.prim -> unit Devices.placement -> unit Route.t =
 fun ~by op p ->
  let shapes = ref [] in
  Prim.iteri (fun _ x -> shapes := Prim.shape x :: !shapes) op;
  let shapes = Array.of_list (List.rev !shapes) in
  let at_p () =
    let operands = Array.make (Array.length shapes) p in
    { Route.operands; result = p; compute = Windows }
  in
  if uncut p then at_p ()
  else
    match Prim.rule op with
    | Route.Elementwise -> at_p ()
    | rule -> Route.at ~by rule shapes p

(* How a constant's results are read: from its memo, where they are kept, or
   computed into memory of their own. *)
type policy = Memo | Fresh

let find memo p =
  List.find_map
    (fun (q, a) -> if Devices.equal q p then Some a else None)
    (Atomic.get memo)

(* Keeps [a] as [p]'s results unless another domain stored first. *)
let rec remember memo p a =
  let old = Atomic.get memo in
  match
    List.find_map
      (fun (q, _) -> if Devices.equal q p then Some () else None)
      old
  with
  | Some () -> ()
  | None ->
      if not (Atomic.compare_and_set memo old ((p, a) :: old)) then
        remember memo p a

(* The constant operands of [node] at [p] not yet computed where it reads
   them. *)
let pending (Value.Node n) p =
  let r = route_at ~by:n.by n.op p in
  let found = ref [] in
  Prim.iteri
    (fun i x ->
      match x with
      | Value.Deferred { node = Value.Node m as node; _ } ->
          let q = r.operands.(i) in
          if find m.memo q = None then found := (node, q) :: !found
      | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ -> ())
    n.op;
  List.rev !found

(* The host's device, at the brand of a value this library reads there: a
   check's flag, a value read out ({!on_host}). No value at it reaches a
   function of nx's own. *)
let host () = Devices.rebrand (Devices.one Devices.host 0)

(* Whether [p] lies on the host's set. *)
let on_host_set p = Devices.number (Devices.set p) = Devices.number Devices.host

(* Pending checks

   A check off the host reduces its flags where they lie and leaves the result
   there unread, since reading it would make the host wait for the set. Each
   domain lists the checks it issued, by set, and a wait on a set drains that
   set's, oldest first, up to the first that fails. A pending check holds its
   data's memory, so that no donation writes the elements its failure
   reports. *)

type ('d, 'e) issued = {
  holds : (bool, D.bool_elt) A.t;
      (* Whether every flag holds, in memory the host addresses where its
         device is of this machine ({!readable}). *)
  mark : int;
      (* [holds]'s device's last value once [holds] was written. *)
  first : (int64, D.int64_elt, 'd) Value.t option;
      (* The first false flag's position in C order; [None] for one flag,
         whose position is [0]. *)
  shape : int array;  (* The flags'. *)
  data : ('d, 'e) Value.data;
  fail : int array -> 'e -> exn;
}

(* A check and the hold on its data's memory, or the exception of one that
   failed before any wait read it. *)
type pending =
  | Pending : ('d, 'e) issued * Rig.Claim.hold -> pending
  | Raised : exn -> pending

module Sets = Map.Make (Int)

(* Each domain's pending checks, by set number, each set's newest first. The
   domain's systhreads share them, so every update is a compare-and-set of the
   whole. *)
let checks : pending list Sets.t Atomic.t Domain.DLS.key =
  Domain.DLS.new_key (fun () -> Atomic.make Sets.empty)

(* [f] of the calling domain's pending checks, which it replaces, atomically
   against the domain's other systhreads. *)
let rec update f =
  let a = Domain.DLS.get checks in
  let sets = Atomic.get a in
  let sets', r = f sets in
  if Atomic.compare_and_set a sets sets' then r else update f

let of_set n sets = Option.value ~default:[] (Sets.find_opt n sets)

let raised = function Raised _ -> true | Pending _ -> false

(* Adds [p] to set [n]'s pending checks, unless an earlier one failed, whose
   exception the next wait raises in place of any later one's: [false] then. *)
let pend n p =
  update (fun sets ->
      let l = of_set n sets in
      if List.exists raised l then (sets, false)
      else (Sets.add n (p :: l) sets, true))

(* Ends a check's hold. *)
let release = function
  | Pending (_, h) -> Rig.Claim.release h
  | Raised _ -> ()

(* [a] in memory the host addresses where its device is of this machine:
   itself where the host addresses the device's memory, else a copy the
   device makes in its order into memory both address, which the host then
   reads without a copy once the device reports the copy done. *)
let readable a =
  let d = A.device a in
  if Rig.reaches Rig.host d || not (Rig.equal (Rig.host_of d) Rig.host) then a
  else
    let dst = A.create ~memory:Pinned d (A.dtype a) (L.shape (A.layout a)) in
    A.transfer ~src:a ~dst;
    dst

(* Whether [c.holds] is written, so that reading it waits for nothing, asked
   while issuing a check on devices [ds]. On one of [ds], its device's report
   decides, which commits nothing and waits for nothing: a device reports its
   work at most its driver's batch of values late while work is submitted on
   it ({!Rig.section-timeline}). Another device may get no more work, and its
   last values would then stay unreported: asking commits them, since an idle
   device has nothing to batch. The final reduction on [c.holds]'s device
   follows the work on the others, so that device alone decides. *)
let reached ds c =
  let d = A.device c.holds in
  if List.exists (Rig.equal d) ds then Rig.signaled d >= c.mark
  else Rig.Buffer.reached (A.buffer c.holds) Read

(* [a]'s element, read on the host: in place where it addresses [a]'s memory,
   else through a copy. *)
let get_on_host a =
  let d = A.device a in
  if Rig.equal (Rig.host_of d) Rig.host then A.get a [||]
  else A.get (A.to_device Rig.host a) [||]

(* The one element of [a], as a 0-d view. *)
let unit_view a = Option.get (A.move (Reshape [||]) a)

(* Set [n]'s pending checks, oldest first, taken off the list. *)
let drained n =
  update (fun sets -> (Sets.remove n sets, List.rev (of_set n sets)))

(* The first datum's placement, if a datum has bytes. *)
let rec data_placement : type d e.
    (d, e) Value.data -> d Devices.placement option = function
  | [] -> None
  | x :: rest -> (
      match Prim.at x with Some p -> Some p | None -> data_placement rest)

(* [data] with each datum that has bytes a live value of its own over them,
   which no donation of the datum kills. *)
let rec kept : type d e. (d, e) Value.data -> (d, e) Value.data = function
  | [] -> []
  | ((Value.Deferred _ | Value.Traced _) as x) :: rest -> x :: kept rest
  | x :: rest ->
      let x =
        Prim.of_arrays (Prim.placement x) ~shape:(Prim.shape x) (Prim.arrays x)
      in
      x :: kept rest

(* The buffers of [data]'s bytes. *)
let rec data_buffers : type d e. (d, e) Value.data -> Rig.Buffer.t list =
  function
  | [] -> []
  | (Value.Deferred _ | Value.Traced _) :: rest -> data_buffers rest
  | x :: rest ->
      Iarray.fold_right
        (fun a bs -> A.buffer a :: bs)
        (Prim.arrays x) (data_buffers rest)

(* The next node's id. *)
let nodes = Atomic.make 0

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
      let rs = Prim.reductions reductions and shape = L.shape layout in
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
        order;
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
        (K.contract ~order ~batch ~contracting ~segments ~acc ~dst ~init
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
      | Value.Svd { vectors = No_vectors; a }, (s, ()) ->
          let s = nth j s and a = one a in
          ran ~by (K.svd ~s a) [| A.Any s |] [| A.Any a |]
      | Value.Svd { vectors = Vectors _; a }, (s, (u, vh)) ->
          let s = nth j s and u = nth j u and vh = nth j vh and a = one a in
          ran ~by
            (K.svd ~s ~vectors:(~u, ~vh) a)
            [| A.Any s; A.Any u; A.Any vh |]
            [| A.Any a |]
      | Value.Eigh { vectors = No_vectors; a }, (w, ()) ->
          let w = nth j w and a = one a in
          ran ~by (K.eigh ~w a) [| A.Any w |] [| A.Any a |]
      | Value.Eigh { vectors = Vectors (); a }, (w, v) ->
          let w = nth j w and v = nth j v and a = one a in
          ran ~by (K.eigh ~w ~v a) [| A.Any w; A.Any v |] [| A.Any a |]
      | Value.Eig { vectors = No_vectors; a; _ }, (w, ()) ->
          let w = nth j w and a = one a in
          ran ~by (K.eig ~w a) [| A.Any w |] [| A.Any a |]
      | Value.Eig { vectors = Vectors (); a; _ }, (w, v) ->
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
      ran ~by (K.apply1 Copy ~dst x) [| A.Any dst |] [| A.Any x |]
  | Value.Move _ | Value.Bitcast _ | Value.Place _ | Value.Check _ ->
      invalid_arg "Exec.call: an operation no kernel computes"

(* [f ()], the claimed handles [hs] given back where it raises: [f] neither
   writes their memory nor consumes it, so their donors live on, as before the
   call. *)
let unclaimed hs f =
  match f () with
  | r -> r
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      Donate.unclaim hs;
      Printexc.raise_with_backtrace e bt

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
      match Expand.single prog with
      | Some node ->
          let out = P.dtype prog (Iarray.get prog.outs 0) in
          refuses ~by ~kernels (Prim.kind node) d (out :: dts)
      | None -> refuses ~by ~kernels (Prim.name op) d dts)
  | _ -> refuses ~by ~kernels (Prim.name op) d dts

(* Two handles of one donor in [op] raise before any operand's death does: the
   operation is malformed whatever has died. *)
let rec run : type r. by:string -> r Value.prim -> r =
 fun ~by op ->
  if not (Prim.exists Donate.is_handle op) then begin
    Prim.iteri (fun i x -> checked ~by i x) op;
    match plain ~by op with
    | r -> r
    | exception (Invalid_argument _ as e) -> rechecked ~by op e
  end
  else
    let (Prim.Operands xs) = Prim.operands op in
    let hs = Donate.handles ~by xs in
    Prim.iteri (fun i x -> checked ~by i x) op;
    donated ~by op hs

(* [op] over donated operands [hs]: a movement that maps elements one to one
   passes a new handle on; any other operation checks its rule, then claims the
   handles, reads their memory, writes its first result into a handle's memory
   where it may, and ends the handles. *)
and donated : type r. by:string -> r Value.prim -> Donate.handle list -> r =
 fun ~by op hs ->
  match op with
  | Value.Move (mv, (Value.Donated d as x)) ->
      (* A view of a handle leaves the memory unshared: the handle it passes on
         has the same donor and key, so its consumer may still write into it.
         The movement claims nothing: the deaths wait for that consumer, which
         checks every operand first, so an operation that aligns its operands
         by movements kills nothing when it is refused, declined into fresh
         memory, or loses a race. *)
      let moved () =
        match run ~by (Value.Move (mv, Donate.live x)) with
        | r -> (Prim.placement r, Prim.shape r, Prim.arrays r)
        | exception (Invalid_argument _ as e) ->
            (* The copy read the memory without a claim. *)
            let bt = Printexc.get_raw_backtrace () in
            Prim.alive ~by 0 x;
            Printexc.raise_with_backtrace e bt
      in
      let at, shape, arrays =
        if Iarray.length d.arrays <> 1 then moved ()
        else begin
          ignore (Prim.layouts ~by op);
          match A.move mv (Iarray.get d.arrays 0) with
          | Some v -> (d.at, L.shape (A.layout v), [| v |])
          | None -> moved ()
        end
      in
      Value.Donated { key = d.key; at; shape; arrays; donor = d.donor }
  | Value.Map _ | Value.Reduce _ | Value.Scan _ | Value.Gather _
  | Value.Scatter _ | Value.Sort _ | Value.Fft _ | Value.Linalg _
  | Value.Fold _ | Value.Assemble _ | Value.Contract _ | Value.Copy _
  | Value.Move _ | Value.Bitcast _ | Value.Place _ | Value.Check _ -> (
      (* Checked before the claim: an operation its rule refuses neither takes
         a handle's memory nor shows another domain a handle dead. *)
      let r = Prim.route ~by op in
      Donate.claim ~by hs;
      let op = Donate.unhandled hs op in
      match Donate.reuse ~by op hs with
      | Some (i, b, b') -> (
          (* Operand [i]'s memory is the result's, now over [b']: the kernel
             reads each index of it before it writes the result there. rig
             consumed the memory before the kernel answers, for good, so the
             donors stay dead if it raises. *)
          let over (type v s d) (y : (v, s, d) Value.t) : (v, s, d) Value.t =
            match y with
            | Value.Array r when A.buffer r.a == b ->
                Value.Array { r with a = A.v (A.dtype r.a) (A.layout r.a) b' }
            | Value.Array _ | Value.Shards _ | Value.Donated _
            | Value.Deferred _ | Value.Traced _ ->
                y
          in
          let op = Prim.map over op in
          let (Prim.Operands xs) = Prim.operands op in
          let (Value.Any target) = List.nth xs i in
          match routed ~by ~into:(A.Any (one target)) r op with
          | y ->
              Donate.consumed ~by ~reused:[ b ] hs;
              y
          | exception (Invalid_argument _ as e) -> rechecked ~by op e)
      | None ->
          let y =
            unclaimed hs (fun () ->
                match routed ~by r op with
                | y -> y
                | exception (Invalid_argument _ as e) -> rechecked ~by op e)
          in
          Donate.consumed ~by ~reused:[] hs;
          y)

and plain : type r. by:string -> r Value.prim -> r =
 fun ~by op ->
  match op with
  | Value.Place (p, x) -> placed ~by p x
  | Value.Check { ok; data; fail } ->
      ignore (Prim.layouts ~by op);
      check ~by ok data fail
  | _ ->
      if not (Prim.exists is_concrete op) then defer ~by op
      else routed ~by (Prim.route ~by op) op

(* [op], a placement, a check, or an operation of a concrete operand, computed
   as its rule decided [r] ({!Prim.route}): its first result into [into] where
   it is given. *)
and routed : type r.
    by:string -> ?into:A.any -> Prim.routed -> r Value.prim -> r =
 fun ~by ?into r op ->
  match op with
  | Value.Place (p, x) -> placed ~by p x
  | Value.Check { ok; data; fail } -> check ~by ok data fail
  | _ ->
      (* A concrete operand has a placement, so the operation has a route. *)
      let route = Option.get r.Prim.route in
      let place i x = placer ~by (Devices.rebrand route.operands.(i)) x in
      compute ~by ?into route r.layouts (Prim.mapi place op)

(* [x] placed at [p]. *)
and placed : type v s d e.
    by:string -> e Devices.placement -> (v, s, d) Value.t -> (v, s, e) Value.t
    =
 fun ~by p x ->
  match x with
  | Value.Deferred _ -> Place.value ~by p (own (Devices.rebrand p) x)
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ ->
      if on_host_set p && not (on_host_set (Prim.placement x)) then wait ~by x;
      let y = Place.value ~by p x in
      (* A value over [x]'s memory, kept or borrowed, shares it. *)
      let xs = Prim.arrays x in
      Iarray.iter
        (fun b ->
          if
            Iarray.exists
              (fun a -> Rig.Buffer.overlaps (A.buffer a) (A.buffer b))
              xs
          then Rig.Claim.share (A.buffer b))
        (Prim.arrays y);
      y

(* [x] readable at [p]. *)
and placer : type v s d.
    by:string -> d Devices.placement -> (v, s, d) Value.t -> (v, s, d) Value.t
    =
 fun ~by p x ->
  match x with
  | Value.Deferred _ -> at p x
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ ->
      if Devices.equal (Prim.placement x) p then x else Place.value ~by p x

and defer : type r. by:string -> r Value.prim -> r =
 fun ~by op ->
  let layouts = Prim.layouts ~by op in
  let id = Atomic.fetch_and_add nodes 1 in
  let node = Value.Node { id; by; op; layouts; memo = Atomic.make [] } in
  (* Its operands are of every set, so its results are too. *)
  Prim.results_at None layouts
    (fun k form -> Value.Deferred { form; node; k })
    op

(* [op] computed over operands that lie where [route], its route, reads them:
   its results lie where the route puts them, laid out as its rule gave them
   ([layouts], {!Prim.layouts}). For a constant's evaluation at [at], the
   operations of its computation by partials compute there. *)
and compute : type r.
    by:string ->
    ?into:A.any ->
    ?at:unit Devices.placement ->
    unit Route.t ->
    L.t array ->
    r Value.prim ->
    r =
 fun ~by ?into ?at route layouts op ->
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
  | Value.Place _ | Value.Check _ -> run ~by op
  | Value.Map _ | Value.Reduce _ | Value.Scan _ | Value.Gather _
  | Value.Scatter _ | Value.Sort _ | Value.Fold _ | Value.Fft _ | Value.Linalg _
  | Value.Assemble _ | Value.Contract _ | Value.Copy _ ->
      sharded ~by ?into ?at route layouts op

(* [op]'s results, by {!Partial} where [route] computes by partials. Else
   fresh where it places them, each device's window [w] of the first computed
   there by its kernels over each operand's window ({!Route.window}). A
   decline raises. *)
and sharded : type r.
    by:string ->
    ?into:A.any ->
    ?at:unit Devices.placement ->
    unit Route.t ->
    L.t array ->
    r Value.prim ->
    r =
 fun ~by ?into ?at route layouts op ->
  match route.compute with
  | Partials -> Partial.run (fun ~by op -> folded ~by ?at op) ~by route op
  | Windows -> (
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
          let rule = Prim.rule op and first = L.shape layouts.(0) in
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

(* [local], an operation's part on device [k] of [set], into its results [r]
   at position [j] by the kernels. A decline raises. *)
and on_device : type r.
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

(* An operation of a computation by partials ({!Partial}), applied where its
   operands lie, or at [at] for a constant's evaluation. *)
and folded : type r. by:string -> ?at:unit Devices.placement -> r Value.prim -> r
    =
 fun ~by ?at op ->
  match at with None -> run ~by op | Some q -> apply_at q ~by op

(* An operation of a constant's evaluation at [q]: one over a concrete operand
   computes where its route reads them; a creation, its other operands
   constants computed at [q], where the evaluation reads it, computes where
   its route over them puts it, and at [q] where it has no operand. *)
and apply_at : type r. unit Devices.placement -> by:string -> r Value.prim -> r
    =
 fun q ~by op ->
  if Prim.exists is_concrete op then run ~by op
  else
    let op = Prim.map (fun y -> at (Devices.rebrand q) y) op in
    let { Prim.route; layouts } = Prim.route ~by op in
    let route =
      match route with
      | Some r -> r
      | None -> { Route.operands = [||]; result = q; compute = Windows }
    in
    compute ~by ~at:q route layouts op

(* The one result of [op] over the arrays [views ()], one per device of [xp] in
   order: at [route]'s result, whose devices hold them. *)
and shard_views : type v s d.
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
      let shape = L.shape f.layout in
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

and check : type d e.
    by:string ->
    (bool, D.bool_elt, d) Value.t ->
    (d, e) Value.data ->
    (int array -> e -> exn) ->
    unit =
 fun ~by ok data fail ->
  let shape = Prim.shape ok in
  if Array.mem 0 shape then ()
  else
    (* The check computes where [ok]'s bytes lie, else where a datum's do, and
       a constant [ok] is computed there first. With no bytes anywhere, its
       reduction is a constant, which [failure] computes on the host. *)
    let where =
      match Prim.at ok with Some p -> Some p | None -> data_placement data
    in
    let reduce r =
      let ok =
        match (where, ok) with
        | Some p, Value.Deferred _ -> at p ok
        | _ -> ok
      in
      run ~by
        (Value.Reduce
           {
             layout = L.contiguous shape;
             axes = Iarray.init (Array.length shape) Fun.id;
             prog = P.of_node ~ins:[| D.Any D.Bool |] (In 0);
             reductions = Value.[ r ];
             loads = [| Plain ok |];
           })
    in
    let scalar x = A.get (on_host ~by x) [||] in
    (* One flag is its own reduction: no kernel reduces it. *)
    let one = Array.for_all (( = ) 1) shape in
    match where with
    | Some p when not (on_host_set p) ->
        (* The set's earlier checks settle first, so that a raise there leaves
           this check unissued and none of its data held. A set that holds a
           failure keeps no later check, which then computes nothing. *)
        let set = Devices.set p in
        let n = Devices.number set in
        let ds = Array.map (Devices.rig set) (Grid.devices (Devices.grid p)) in
        if settle_reached ~by n (Array.to_list ds) then begin
          (* One reduction gives both whether every flag holds and the first
             false, so that the check leaves only scalars to read later. The
             mark follows the copy [readable] submits, which, for one flag,
             no reduction orders. The hold keeps a donation from writing a
             flag read in place. *)
          let holds, first =
            if one then (element p ok, None)
            else
              let (holds, first), () = reduce (Arg (Min, 0, D.Bool)) in
              (Iarray.get (Prim.arrays holds) 0, Some first)
          in
          let holds = readable holds in
          let mark = Rig.submitted (A.device holds) in
          let data = kept data in
          let held = Rig.Claim.hold (A.buffer holds :: data_buffers data) in
          let c = { holds; mark; first; shape; data; fail } in
          if not (pend n (Pending (c, held))) then Rig.Claim.release held
        end
    | Some _ | None when one ->
        let holds =
          match where with
          | Some p -> get_on_host (element p ok)
          | None -> A.get (unit_view (on_host ~by ok)) [||]
        in
        if not holds then raise (failed ~by 0L shape data fail)
    | Some _ | None ->
        (* A passing check runs one [Min]; [Arg], for the first [false], costs
           more and runs only on a failure. *)
        let holds, () = reduce (Monoid (Min, 0, D.Bool)) in
        if not (scalar holds) then
          let (_, first), () = reduce (Arg (Min, 0, D.Bool)) in
          raise (failed ~by (scalar first) shape data fail)

(* The one element of [x], computed at [p] where [x] is a constant: a 0-d
   view of the array that holds it. A cut placement's other devices hold
   empty windows. *)
and element : type v s d.
    d Devices.placement -> (v, s, d) Value.t -> (v, s) A.t =
 fun p x ->
  let holds a = L.numel (A.layout a) = 1 in
  unit_view (Option.get (Iarray.find_opt holds (Prim.arrays (at p x))))

(* The exception of a check whose first false lies at [first] in C order. *)
and failed : type d e.
    by:string -> int64 -> int array -> (d, e) Value.data ->
    (int array -> e -> exn) -> exn =
 fun ~by first shape data fail ->
  let idx = unravel (Int64.to_int first) shape in
  fail idx (data_at ~by idx data)

(* Each of [data]'s elements at [idx], read on the host. *)
and data_at : type d e.
    by:string -> int array -> (d, e) Value.data -> e =
 fun ~by idx -> function
  | [] -> ()
  | x :: rest ->
      let v = A.get (on_host ~by x) idx in
      (v, data_at ~by idx rest)

(* A constant read at [p] by an operation: a view of a constant is its operand's
   results read at [p] and viewed; any other its results there, kept in its
   memo, which every later read at [p] finds. *)
and at : type v s d.
    d Devices.placement -> (v, s, d) Value.t -> (v, s, d) Value.t =
 fun p x -> through_views Memo p x

(* [x]'s elements in an array on the host: a constant computed there, any other
   value placed there. *)
and on_host : type v s d. by:string -> (v, s, d) Value.t -> (v, s) A.t =
 fun ~by x ->
  let host : d Devices.placement = host () in
  match x with
  | Value.Deferred _ -> Iarray.get (Prim.arrays (at host x)) 0
  | Value.Traced { owner; _ } ->
      invalid_argf "%s: the value is traced by %s" by owner.name
  | Value.Array _ | Value.Shards _ | Value.Donated _ ->
      Iarray.get (Prim.arrays (Place.value ~by host x)) 0

and wait : type v s d. by:string -> (v, s, d) Value.t -> unit =
 fun ~by x ->
  match x with
  | Value.Deferred _ -> ()
  | Value.Traced { owner; _ } ->
      invalid_argf "%s: the value is traced by %s" by owner.name
  | Value.Array _ | Value.Shards _ | Value.Donated _ ->
      Iarray.iter (fun a -> Rig.Buffer.wait (A.buffer a) Read) (Prim.arrays x);
      settle ~by (drained (Devices.number (Devices.set (Prim.placement x))))

(* Reads [checks], oldest first, up to the first that fails, which raises: the
   later ones go unread. Every check's hold ends then, so that a donation of its
   data right after the wait may reuse the memory. *)
and settle ~by checks =
  let failure = function
    | Pending (c, _) -> outcome ~by c
    | Raised e -> Some e
  in
  Fun.protect ~finally:(fun () -> List.iter release checks) (fun () ->
      Option.iter raise (List.find_map failure checks))

(* Settles set [n]'s earlier checks whose flags are computed, so that a domain
   that never waits on the set holds only the data of the checks still
   running: a passing check goes, a failing one keeps only its exception, and
   the checks after it go, since no wait can raise them. It reads them oldest
   first and stops at the first still running, so it asks rig of one check
   more than it settles. The checks are read outside the update, which then
   takes off those still pending, against a wait on another systhread that
   drains them meanwhile. [ds] are the devices of the check being issued
   ({!reached}). [false] where the set then holds a failure. *)
and settle_reached ~by n ds =
  let earlier = List.rev (of_set n (Atomic.get (Domain.DLS.get checks))) in
  let rec read = function
    | [] | Raised _ :: _ -> []
    | (Pending (c, _) as p) :: rest -> (
        if not (reached ds c) then []
        else
          match outcome ~by c with
          | None -> (p, None) :: read rest
          | Some e -> [ (p, Some e) ])
  in
  match read earlier with
  | [] -> not (List.exists raised earlier)
  | outcomes ->
      (* Set [n]'s checks after the update, oldest first, and those it
         ends. *)
      let rec cut = function
        | [] -> ([], [])
        | (Raised _ as r) :: rest -> ([ r ], rest)
        | (Pending _ as p) :: rest -> (
            match List.assq_opt p outcomes with
            | None ->
                let kept, ended = cut rest in
                (p :: kept, ended)
            | Some None ->
                let kept, ended = cut rest in
                (kept, p :: ended)
            | Some (Some e) -> ([ Raised e ], p :: rest))
      in
      let kept, ended =
        update (fun sets ->
            match cut (List.rev (of_set n sets)) with
            | [], ended -> (Sets.remove n sets, ([], ended))
            | kept, ended -> (Sets.add n (List.rev kept) sets, (kept, ended)))
      in
      List.iter release ended;
      not (List.exists raised kept)

(* [c]'s exception, read on the host, where a flag is false. *)
and outcome : type d e. by:string -> (d, e) issued -> exn option =
 fun ~by c ->
  if get_on_host c.holds then None
  else
    let first =
      match c.first with
      | Some f -> A.get (on_host ~by f) [||]
      | None -> 0L
    in
    Some (failed ~by first c.shape c.data c.fail)

(* The constant [c] computed at [p] into memory of its own, its operands taken
   from their memos or computed for this alone: a value given to a caller shares
   no memory with a memo. A view of a constant views its operand computed into
   memory of its own. *)
and own : type v s d.
    d Devices.placement -> (v, s, d) Value.t -> (v, s, d) Value.t =
 fun p x -> through_views Fresh p x

(* The constant [x] at [p]: the chain of views it reads descended to the first
   operation that is not one, whose results [policy] gives at the placement the
   chain reads it at, then each view replayed outward over its operand's. A
   chain of any length takes no stack depth. *)
and through_views : type v s d.
    policy -> d Devices.placement -> (v, s, d) Value.t -> (v, s, d) Value.t =
 fun policy p x ->
  match x with
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ -> x
  | Value.Deferred { form; node; k } ->
      (* Each view and where it is read, the innermost first. A view, a Move
         or a Bitcast, keeps no results of its own: it is read through to its
         operand's. *)
      let views = ref [] and node = ref node in
      let key = ref (Devices.rebrand p) in
      let continue = ref true in
      while !continue do
        let (Value.Node n) = !node in
        match n.op with
        | Value.Move (_, Value.Deferred { node = m; _ })
        | Value.Bitcast (_, Value.Deferred { node = m; _ }) ->
            views := (!node, !key) :: !views;
            key := (route_at ~by:n.by n.op !key).operands.(0);
            node := m
        | Value.Move _ | Value.Bitcast _ | Value.Map _ | Value.Reduce _
        | Value.Scan _ | Value.Gather _ | Value.Scatter _ | Value.Sort _
        | Value.Fft _ | Value.Linalg _ | Value.Fold _ | Value.Assemble _
        | Value.Contract _ | Value.Copy _ | Value.Place _ | Value.Check _ ->
            continue := false
      done;
      let (Value.Node n) = !node in
      let base =
        match policy with
        | Fresh -> evaluate !node !key
        | Memo -> (
            match find n.memo !key with
            | Some a -> a
            | None ->
                remember n.memo !key (evaluate !node !key);
                Option.get (find n.memo !key))
      in
      (* A view's one operand is the constant read just before it. *)
      let replay arrays (view, key) =
        let operand (type w r e) (q : e Devices.placement)
            (y : (w, r, e) Value.t) : (w, r, e) Value.t =
          match y with
          | Value.Deferred { form; k; _ } ->
              make q ~shape:(L.shape form.layout) (typed form.dtype arrays.(k))
          | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ ->
              y
        in
        compute_node ~resolve:operand view key
      in
      let arrays = List.fold_left replay base !views in
      make p ~shape:(L.shape form.layout) (typed form.dtype arrays.(k))

(* [root]'s results at [p], fresh: the constants it reads are taken from their
   memos where an operation read them at the placement it reads them, and
   computed otherwise in a table of this evaluation's alone, dropped after. Each
   node computes once in an evaluation, in dependency order, with a stack of its
   own: a chain of any length takes no stack depth. *)
and evaluate root p =
  match pending root p with
  | [] -> compute_node ~resolve:memoised root p
  | _ -> evaluate_all root p

(* A constant operand at [q], from its memo: every one an operation reads is
   there. *)
and memoised : type v s d.
    d Devices.placement -> (v, s, d) Value.t -> (v, s, d) Value.t =
 fun q x ->
  match x with
  | Value.Deferred { form; node = Value.Node n; k } ->
      let arrays = Option.get (find n.memo (Devices.rebrand q)) in
      make q ~shape:(L.shape form.layout) (typed form.dtype arrays.(k))
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ -> x

(* [evaluate] where some constant [root] reads is in no memo. *)
and evaluate_all root p =
  let local = Hashtbl.create 16 in
  let lookup (Value.Node n) q =
    match find n.memo q with
    | Some a -> Some a
    | None ->
        List.find_map
          (fun (q', a) -> if Devices.equal q' q then Some a else None)
          (Option.value ~default:[] (Hashtbl.find_opt local n.id))
  in
  (* How this evaluation reads a constant operand at [q]: from its memo or from
     the evaluation's own table. *)
  let resolve (type v s d) (q : d Devices.placement) (x : (v, s, d) Value.t) :
      (v, s, d) Value.t =
    match x with
    | Value.Deferred { form; node; k } ->
        let arrays = Option.get (lookup node (Devices.rebrand q)) in
        make q ~shape:(L.shape form.layout) (typed form.dtype arrays.(k))
    | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ -> x
  in
  let missing node q =
    List.filter (fun (dep, q) -> lookup dep q = None) (pending node q)
  in
  let stack = Stack.create () in
  List.iter (fun d -> Stack.push d stack) (missing root p);
  while not (Stack.is_empty stack) do
    let (Value.Node n as top), q = Stack.top stack in
    if lookup top q <> None then ignore (Stack.pop stack)
    else
      match missing top q with
      | [] ->
          ignore (Stack.pop stack);
          let a = compute_node ~resolve top q in
          let kept = Option.value ~default:[] (Hashtbl.find_opt local n.id) in
          Hashtbl.replace local n.id ((q, a) :: kept)
      | deps -> List.iter (fun d -> Stack.push d stack) deps
  done;
  compute_node ~resolve root p

(* [node]'s results at [p], its constant operands already computed where it
   reads them. *)
and compute_node :
    resolve:
      ('v 's 'd.
       'd Devices.placement -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t) ->
    Value.node ->
    unit Devices.placement ->
    A.any array array =
 fun ~resolve (Value.Node n) p ->
  let by = n.by in
  match n.op with
  | Value.Move (mv, x) when uncut p ->
      (* A view at a placement that cuts no axis: each device's whole operand
         there, moved. Its rule held when it was made. *)
      let xs = Prim.arrays (resolve (Devices.rebrand p) x) in
      [| Prim.any (Iarray.map (moved ~by (Devices.set p) mv) xs) |]
  | Value.Bitcast (dt, x) when uncut p ->
      let xs = Prim.arrays (resolve (Devices.rebrand p) x) in
      [| Prim.any (Iarray.map (cast ~by (Devices.set p) dt) xs) |]
  | op ->
      (* Its rule held when it was made. It computes at its route's result, [p]
         with the axes its rule keeps whole made whole, over operands where the
         route reads them; a creation computes there too. *)
      let route = route_at ~by op p in
      let q = route.result in
      let op' =
        Prim.mapi
          (fun i y -> resolve (Devices.rebrand route.operands.(i)) y)
          op
      in
      let r = compute ~by ~at:q route n.layouts op' in
      if Devices.equal q p then Prim.result_arrays op' r
      else
        (* Each device of [p] keeps its window, cut from its window at [q], in
           memory of its own, so that no memo holds more. *)
        let per = ref [] in
        Prim.iter_results
          (fun y ->
            let y = Place.value ~by (Devices.rebrand p) y in
            per := Prim.any (Iarray.map A.copy (Prim.arrays y)) :: !per)
          op' r;
        Array.of_list (List.rev !per)

(* The fast path *)

(* A destination for a result of [dt] beside [x]: over [x]'s layout where it is
   C-contiguous from offset 0 and of [dt], fresh otherwise. A dtype narrower
   than a byte is always fresh, whose spare bits in its last byte are zero. *)
let destination (type v s w r) (dt : (w, r) D.t) (x : (v, s) A.t) : (w, r) A.t =
  let l = A.layout x in
  if
    L.is_contiguous l
    && L.offset l = 0
    && D.equal dt (A.dtype x)
    && D.bits dt >= 8
  then A.v dt l (Rig.Buffer.create (A.device x) (D.bytes dt (L.numel l)))
  else A.create (A.device x) dt (L.shape l)

(* [run] into fresh memory over the claimed handle [h], operand [i], whose
   array is [a], [h] given back where it raises. *)
let into_fresh i dt h a run =
  unclaimed [ Donate.Handle (i, h) ] (fun () -> run (destination dt a) a)

(* The one-node map over the handle [h], operand [i], whose array is [a], its
   result of [dt]: [h] is claimed, then its memory takes the result where rig
   holds it exclusive and it {!Donate.holds} the result, as {!Donate.reuse}
   decides for any operation, and [h] stays dead if [run] raises; otherwise
   the result is fresh, the memory, where no other value reads it, is freed
   after the kernel, and [h] is given back where [run] raises. Its other
   operands are live arrays, so none is the donor, whose word is offered, and
   each is read at the result's own index. The kind's rule holds over them:
   an operation it refuses goes off the fast path, which checks it before any
   claim. [run dst a'] is the kernel into [dst] with [a'] for [h]. *)
let donated (type v s w r d) ~by i (dt : (w, r) D.t) (h : (v, s, d) Value.t)
    (a : (v, s) A.t) (run : (w, r) A.t -> (v, s) A.t -> (w, r, d) Value.t) :
    (w, r, d) Value.t =
  let l = A.layout a in
  let reusable = Donate.holds (D.Any dt) (L.contiguous (L.shape l)) a in
  (* Checked first, as any operation's operands are. *)
  Prim.alive ~by i h;
  Donate.claim_handle ~by i h;
  match D.equal_witness (A.dtype a) dt with
  | Some Type.Equal when reusable -> (
      match Donate.take ~by (A.buffer a) with
      | Some b' ->
          let a' = A.v (A.dtype a) l b' in
          run a' a'
      | None -> into_fresh i dt h a run)
  | Some _ | None ->
      let r = into_fresh i dt h a run in
      ignore (Rig.Buffer.free (A.buffer a) : bool);
      r

(* Whether [x] is a live array at [at]. *)
let live_at (type v s d) (at : d Devices.placement) (x : (v, s, d) Value.t) =
  match x with
  | Value.Array r -> r.at == at && r.dead == Value.Live
  | Value.Shards _ | Value.Donated _ | Value.Deferred _ | Value.Traced _ ->
      false

(* Whether [h] is a handle over one array at [at], whose set has kernels: the
   donated fast paths'. *)
let handle_at (type v s d) (at : d Devices.placement) (h : (v, s, d) Value.t) =
  match h with
  | Value.Donated d ->
      d.at == at
      && Iarray.length d.arrays = 1
      && Devices.kernels (Devices.set at) <> None
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> false

(* [a] as a live value at [at]. *)
let value_of at a = Value.Array { at; a; dead = Value.Live }

(* [at]'s kernels, where [handle_at] holds. *)
let kernels_at at = Option.get (Devices.kernels (Devices.set at))

(* The one-node maps, built and run: the paths off the fast one. *)
let run1 ~by k dt x =
  let v, () = run ~by (Prim.op1 ~by k dt x) in
  v

let run2 ~by k dt x y =
  let v, () = run ~by (Prim.op2 ~by k dt x y) in
  v

let run3 ~by k c x y =
  let v, () = run ~by (Prim.op3 ~by k c x y) in
  v

(* Raises for [node], which [K] declined into the first of [arrays] over the
   rest, as {!refused} names a one-node map. *)
let refused_node ~by (module K : Nx_kernel.S) node arrays =
  let (A.Any dst) = List.hd arrays in
  let dts = List.map (fun (A.Any a) -> D.Any (A.dtype a)) arrays in
  declined ~by ~kernels:K.name node (A.device dst) dts

let apply1 (type v s w r d) ~by k (dt : (w, r) D.t)
    (x : (v, s, d) Value.t) : (w, r, d) Value.t =
  match x with
  | Value.Array rx when rx.dead == Value.Live -> (
      let at = rx.at and a = rx.a in
      match Devices.kernels (Devices.set at) with
      | None -> run1 ~by k dt x
      | Some (module K) -> (
          let dst = destination dt a in
          match K.apply1 k ~dst a with
          | Done -> Value.Array { at; a = dst; dead = Value.Live }
          | Declined ->
              refused_node ~by (module K)
                (Op1 (k, D.Any dt, 0))
                [ A.Any dst; A.Any a ]
          | Wrong_dtype -> run1 ~by k dt x
          | refusal ->
              recheck ~by 0 x;
              A.refused by refusal [ A.Any dst; A.Any a ]))
  | Value.Donated dx when handle_at dx.at x && P.accepts1 k (Prim.dtype x) dt
    ->
      let (module K) = kernels_at dx.at in
      donated ~by 0 dt x (Iarray.get dx.arrays 0) (fun dst a ->
          match K.apply1 k ~dst a with
          | Done -> value_of dx.at dst
          | Declined ->
              refused_node ~by (module K)
                (Op1 (k, D.Any dt, 0))
                [ A.Any dst; A.Any a ]
          | Wrong_dtype -> run1 ~by k dt (value_of dx.at a)
          | refusal -> A.refused by refusal [ A.Any dst; A.Any a ])
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Deferred _
  | Value.Traced _ ->
      run1 ~by k dt x

(* [k] into [dst] over the arrays [a] and [b] at [at]: [Done] is the result, a
   decline raises, and a dtype or shape refusal is the operation run over them
   as live values. *)
let kernel2 (type v s w r d) ~by (module K : Nx_kernel.S) k
    (dt : (w, r) D.t) (at : d Devices.placement) (x : (v, s, d) Value.t)
    (y : (v, s, d) Value.t) (dst : (w, r) A.t) (a : (v, s) A.t)
    (b : (v, s) A.t) : (w, r, d) Value.t =
  match K.apply2 k ~dst a b with
  | Done -> value_of at dst
  | Declined ->
      refused_node ~by (module K) (Op2 (k, 0, 1))
        [ A.Any dst; A.Any a; A.Any b ]
  | Wrong_dtype | Shape_mismatch ->
      run2 ~by k dt (value_of at a) (value_of at b)
  | refusal ->
      recheck ~by 0 x;
      recheck ~by 1 y;
      A.refused by refusal [ A.Any dst; A.Any a; A.Any b ]

(* As [kernel2], for [apply3]. *)
let kernel3 (type a b v s d) ~by (module K : Nx_kernel.S) k
    (at : d Devices.placement) (vc : (a, b, d) Value.t) (vx : (v, s, d) Value.t)
    (vy : (v, s, d) Value.t) (dst : (v, s) A.t) (c : (a, b) A.t)
    (x : (v, s) A.t) (y : (v, s) A.t) : (v, s, d) Value.t =
  match K.apply3 k ~dst c x y with
  | Done -> value_of at dst
  | Declined ->
      refused_node ~by (module K) (Op3 (k, 0, 1, 2))
        [ A.Any dst; A.Any c; A.Any x; A.Any y ]
  | Wrong_dtype | Shape_mismatch ->
      run3 ~by k (value_of at c) (value_of at x) (value_of at y)
  | refusal ->
      recheck ~by 0 vc;
      recheck ~by 1 vx;
      recheck ~by 2 vy;
      A.refused by refusal [ A.Any dst; A.Any c; A.Any x; A.Any y ]

(* Whether the one-node map [k] over [x] and [y] is well formed. A constant is
   read for an operation only once its rule holds, so that a refused operation
   computes nothing. *)
let well_formed2 k x y = P.accepts2 k (Prim.dtype x) && Prim.same_shape x y

let well_formed3 k c x y =
  P.accepts3 k (Prim.dtype c) (Prim.dtype x)
  && Prim.same_shape c x && Prim.same_shape x y

let rec apply2 : type v s w r d.
    by:string ->
    P.op2 ->
    (w, r) D.t ->
    (v, s, d) Value.t ->
    (v, s, d) Value.t ->
    (w, r, d) Value.t =
 fun ~by k dt x y ->
  match (x, y) with
  | Value.Array rx, Value.Array ry
    when rx.at == ry.at && rx.dead == Value.Live && ry.dead == Value.Live -> (
      let at = rx.at and a = rx.a and b = ry.a in
      match Devices.kernels (Devices.set at) with
      | None -> run2 ~by k dt x y
      | Some (module K) -> (
          let dst = destination dt a in
          match K.apply2 k ~dst a b with
          | Done -> Value.Array { at; a = dst; dead = Value.Live }
          | Declined ->
              refused_node ~by (module K) (Op2 (k, 0, 1))
                [ A.Any dst; A.Any a; A.Any b ]
          | Wrong_dtype | Shape_mismatch -> run2 ~by k dt x y
          | refusal ->
              recheck ~by 0 x;
              recheck ~by 1 y;
              A.refused by refusal [ A.Any dst; A.Any a; A.Any b ]))
  | Value.Donated dx, Value.Array ry
    when handle_at ry.at x && live_at ry.at y
         && P.accepts2 k (Prim.dtype x) ->
      let (module K) = kernels_at ry.at in
      donated ~by 0 dt x (Iarray.get dx.arrays 0) (fun dst a ->
          kernel2 ~by (module K) k dt ry.at x y dst a ry.a)
  | Value.Array rx, Value.Donated dy
    when handle_at rx.at y && live_at rx.at x
         && P.accepts2 k (Prim.dtype x) ->
      let (module K) = kernels_at rx.at in
      donated ~by 1 dt y (Iarray.get dy.arrays 0) (fun dst b ->
          kernel2 ~by (module K) k dt rx.at x y dst rx.a b)
  | Value.Array rx, Value.Deferred _
    when rx.dead == Value.Live && well_formed2 k x y ->
      (* A constant beside a value is read where the value lies. *)
      apply2 ~by k dt x (at rx.at y)
  | Value.Deferred _, Value.Array ry
    when ry.dead == Value.Live && well_formed2 k x y ->
      apply2 ~by k dt (at ry.at x) y
  | _ -> run2 ~by k dt x y

(* Where the constants among [c], [x] and [y] are read: the one placement of the
   live arrays among them, where every other is a constant. *)
let beside3 (type a b v s d) (c : (a, b, d) Value.t) (x : (v, s, d) Value.t)
    (y : (v, s, d) Value.t) : d Devices.placement option =
  let place : type v s. (v, s, d) Value.t -> d Devices.placement option option =
    function
    | Value.Deferred _ -> Some None
    | Value.Array r when r.dead == Value.Live -> Some (Some r.at)
    | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ -> None
  in
  match (place c, place x, place y) with
  | Some pc, Some px, Some py -> (
      match List.filter_map Fun.id [ pc; px; py ] with
      | p :: rest when List.for_all (( == ) p) rest -> Some p
      | _ -> None)
  | _ -> None

let rec apply3 : type a b v s d.
    by:string ->
    P.op3 ->
    (a, b, d) Value.t ->
    (v, s, d) Value.t ->
    (v, s, d) Value.t ->
    (v, s, d) Value.t =
 fun ~by k c x y ->
  match (c, x, y) with
  | Value.Array rc, Value.Array rx, Value.Array ry
    when rc.at == rx.at && rc.at == ry.at && rc.dead == Value.Live
         && rx.dead == Value.Live && ry.dead == Value.Live -> (
      let at = rc.at and ca = rc.a and a = rx.a and b = ry.a in
      match Devices.kernels (Devices.set at) with
      | None -> run3 ~by k c x y
      | Some (module K) -> (
          let dst = destination (A.dtype a) a in
          match K.apply3 k ~dst ca a b with
          | Done -> Value.Array { at; a = dst; dead = Value.Live }
          | Declined ->
              refused_node ~by (module K) (Op3 (k, 0, 1, 2))
                [ A.Any dst; A.Any ca; A.Any a; A.Any b ]
          | Wrong_dtype | Shape_mismatch -> run3 ~by k c x y
          | refusal ->
              recheck ~by 0 c;
              recheck ~by 1 x;
              recheck ~by 2 y;
              A.refused by refusal [ A.Any dst; A.Any ca; A.Any a; A.Any b ]))
  | Value.Donated dc, Value.Array rx, Value.Array ry
    when handle_at rx.at c && live_at rx.at x && live_at rx.at y
         && P.accepts3 k (Prim.dtype c) (Prim.dtype x) ->
      let (module K) = kernels_at rx.at in
      donated ~by 0 (A.dtype rx.a) c (Iarray.get dc.arrays 0)
        (fun dst c' -> kernel3 ~by (module K) k rx.at c x y dst c' rx.a ry.a)
  | Value.Array rc, Value.Donated dx, Value.Array ry
    when handle_at rc.at x && live_at rc.at c && live_at rc.at y
         && P.accepts3 k (Prim.dtype c) (Prim.dtype x) ->
      let (module K) = kernels_at rc.at in
      donated ~by 1 (A.dtype ry.a) x (Iarray.get dx.arrays 0)
        (fun dst x' -> kernel3 ~by (module K) k rc.at c x y dst rc.a x' ry.a)
  | Value.Array rc, Value.Array rx, Value.Donated dy
    when handle_at rc.at y && live_at rc.at c && live_at rc.at x
         && P.accepts3 k (Prim.dtype c) (Prim.dtype x) ->
      let (module K) = kernels_at rc.at in
      donated ~by 2 (A.dtype rx.a) y (Iarray.get dy.arrays 0)
        (fun dst y' -> kernel3 ~by (module K) k rc.at c x y dst rc.a rx.a y')
  | (Value.Deferred _, _, _ | _, Value.Deferred _, _ | _, _, Value.Deferred _)
    when well_formed3 k c x y -> (
      match beside3 c x y with
      | Some p -> apply3 ~by k (at p c) (at p x) (at p y)
      | None -> run3 ~by k c x y)
  | _ -> run3 ~by k c x y

(* The contraction of live arrays on one device, at physically one placement
   whose set has kernels, computed by them without building the operation into
   a result of the shape its rule gives; a decline raises. [None] elsewhere and
   where the pairs do not fit the operands: the operation decides those. *)
let contract (type v s a b c e d) ~by ~order ~batch ~contracting ~acc
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
          (L.shape (A.layout rx.a), None)
          (L.shape (A.layout ry.a), None)
      in
      match (array, shape, Grid.one (Devices.grid at), Devices.kernels set) with
      | Some array, Ok shape, Some k, Some (module K) -> (
          let dst = A.create (Devices.rig set k) dt shape in
          match
            K.contract ~order ~batch ~contracting ~segments:None ~acc ~dst
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
              recheck ~by 0 x;
              recheck ~by 1 y;
              Option.iter (recheck ~by 2) init;
              let init = Option.to_list (Option.map (fun i -> A.Any i) array) in
              A.refused by refusal
                (A.Any dst :: A.Any rx.a :: A.Any ry.a :: init))
      | _ -> None)
  | _ -> None
