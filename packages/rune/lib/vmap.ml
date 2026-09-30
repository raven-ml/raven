(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Vectorizing maps as an interpreter of Nx operations.

   The mapped function is written for unbatched values. Under the map, every
   tensor is either a lane or a constant of the map. A lane is a traced value
   whose payload is its batched tensor, which physically carries the batch
   dimension at axis 0; its shape is the unbatched remainder and its placement
   the batched tensor's without the batch axis. The function and the Nx
   frontend therefore make exactly the decisions of the unbatched program
   (broadcasting, promotion, reshapes), and the interpreter translates each
   resulting operation on a lane to its batched form: shape parameters gain a
   leading batch entry, axis parameters shift by one, and constants meeting
   lanes are lifted with a broadcast view.

   A lane presents itself as T.contiguous even when its batched tensor is not;
   rules compensate by forcing contiguity before reshapes. Operations whose
   operands are all constants are evaluated as they are. Nested maps stack:
   each owns its lanes and batch size, and the translations one level emits,
   whose operands may be the lanes of an enclosing map, are translated again
   by it.

   Every operation is matched explicitly, and the compiler checks it; operations
   without a batching rule raise when an operand is a lane rather than silently
   producing wrong shapes. *)

open Nx.Op
open Prim
module T = Nx

let no_rule op () =
  invalid_arg (Printf.sprintf "Rune: vmap has no batching rule for %s" op)

type state = { batch_size : int; axis : Axis.t option }

let create ?axis ~batch_size () = { batch_size; axis }

type (_, _) Nx.Repr.node +=
  | Lane : { map : state; batched : ('a, 'b) T.t } -> ('a, 'b) Nx.Repr.node

(* [lane st x] is the lane of [st] whose batched tensor is [x]. *)
let lane (type a b) st (x : (a, b) T.t) : (a, b) T.t =
  let s = T.shape x in
  Nx.Repr.Traced.v ~context:(Nx.Repr.context x)
    (T.Placement.without_leading_axis (T.placement x))
    (T.dtype x)
    (Array.sub s 1 (Array.length s - 1))
    (Lane { map = st; batched = x })

let batched (type a b) st (x : (a, b) T.t) =
  match Nx.Repr.v x with
  | Traced t -> (
      match Nx.Repr.Traced.node t with
      | Lane { map; _ } -> map == st
      | _ -> false)
  | Host _ | Placed _ -> false

(* The tensor [x] stands for outside [st]: a lane's batched tensor, or [x]. *)
let physical (type a b) st (x : (a, b) T.t) : (a, b) T.t =
  match Nx.Repr.v x with
  | Traced t -> (
      match Nx.Repr.Traced.node t with
      | Lane { map; batched } when map == st -> batched
      | _ -> x)
  | Host _ | Placed _ -> x

let broadcast_shapes sa sb =
  let ra = Array.length sa and rb = Array.length sb in
  let r = Stdlib.max ra rb in
  Array.init r (fun i ->
      let da = if i < r - ra then 1 else sa.(i - (r - ra)) in
      let db = if i < r - rb then 1 else sb.(i - (r - rb)) in
      if da = db then da
      else if da = 1 then db
      else if db = 1 then da
      else invalid_arg "Rune: vmap cannot broadcast operand shapes")

(* [to_batched st x target] is [x] as a batched tensor of shape [batch_size ::
   target], where [target] is a shape [x]'s broadcasts to. Constants are lifted
   with a broadcast view. *)
let to_batched st x target =
  let s = T.shape x in
  if batched st x && s = target then physical st x
  else begin
    let ones = Array.make (Array.length target - Array.length s) 1 in
    let lead = if batched st x then st.batch_size else 1 in
    let x =
      reshape
        (T.contiguous (physical st x))
        (Array.concat [ [| lead |]; ones; s ])
    in
    expand x (Array.append [| st.batch_size |] target)
  end

let ensure_batched st x = to_batched st x (T.shape x)

(* The sum over the lanes of a value each lane holds: a lane's batched tensor
   summed along its batch axis, a constant times their number. *)
let sum_lanes st v =
  if batched st v then T.sum ~axes:[ 0 ] (physical st v)
  else T.mul_s v (Nx_dtype.of_float (T.dtype v) (Float.of_int st.batch_size))

(* Axis parameters count from the lane's shape; the batch dimension sits at 0,
   so non-negative axes shift by one and negative axes are unchanged. *)
let taxis ax = if ax >= 0 then ax + 1 else ax

(* The batched tensor of lane [x] moved by [m]: the batch dimension enters shape
   parameters and shifts axes by one. *)
let move_batched st x m =
  let p = physical st x in
  match (m : move) with
  | Reshape shape ->
      reshape (T.contiguous p) (Array.append [| st.batch_size |] shape)
  | Permute axes ->
      permute p (Array.append [| 0 |] (Array.map (fun d -> d + 1) axes))
  | Expand shape -> to_batched st x shape
  | Shrink limits -> shrink p (Array.append [| (0, st.batch_size) |] limits)
  | Flip dims -> flip p (Array.append [| false |] dims)
  (* The trailing window axis lands at the physical end, which is the lane's end
     shifted past the batch dimension. *)
  | Window { axis; size; step } ->
      sliding_window p ~axis:(taxis axis) ~window:size ~step

(* The batched window write of [v] into [x] at [starts], one of them a lane. *)
let update_batched st x starts v =
  let t = ensure_batched st x and v' = ensure_batched st v in
  if batched st starts then begin
    (* A window per example: along each axis the rows are gathered from [v]
       at the example's offset, clamped, and kept where they fall inside the
       window. *)
    let tshape = T.shape x and vs = T.shape v in
    let starts = ensure_batched st starts in
    let b = st.batch_size in
    let rank = Array.length tshape in
    let win = ref v' and mask = ref None in
    for ax = 0 to rank - 1 do
      let n = tshape.(ax) and len = vs.(ax) in
      let start = T.reshape [| b; 1 |] (T.slice [ T.A; T.I ax ] starts) in
      let rel = T.sub (T.reshape [| 1; n |] (T.arange T.int32 0 n 1)) start in
      let inside =
        T.logical_and (T.greater_equal_s rel 0l)
          (T.less_s rel (Int32.of_int len))
      in
      let idx = T.clamp ~min:0l ~max:(Int32.of_int (len - 1)) rel in
      let along =
        Array.init (rank + 1) (fun d ->
            if d = 0 then b else if d = ax + 1 then n else 1)
      in
      let shp =
        Array.mapi (fun d s -> if d = ax + 1 then n else s) (T.shape !win)
      in
      win :=
        T.take_along_axis ~axis:(ax + 1)
          ~indices:(T.broadcast_to shp (T.reshape along idx))
          !win;
      let inside = T.reshape along inside in
      mask :=
        Some
          (match !mask with
          | None -> inside
          | Some m -> T.logical_and m inside)
    done;
    match !mask with None -> v' | Some mask -> T.where mask !win t
  end
  else
    (* The batch axis is never written: its start is 0 and [v]'s batch extent
       is the whole axis. *)
    update t ~starts:(pad [| (1, 0) |] 0l starts) v'

(* Quantised products. The lane becomes a leading axis of the weight's parts, of
   [ids] and of [x], a unit axis where one is a constant, after [pad] unit axes
   that align the operands' batch axes, so that each lane meets its own weight
   with its own ids. *)

let lift st ?shape ~lead ~pad x =
  let s = match shape with Some s -> s | None -> T.shape x in
  let b = if batched st x then st.batch_size else 1 in
  let x =
    T.reshape (Array.concat [ [| b |]; Array.make pad 1; s ]) (physical st x)
  in
  if b >= lead then x
  else T.broadcast_to (Array.concat [ [| lead |]; Array.make pad 1; s ]) x

let lift_weight st ~pad (Nx_quant.Mxfp4 { codes; scales }) =
  let lead =
    if batched st codes || batched st scales then st.batch_size else 1
  in
  Nx_quant.mxfp4 ~scales:(lift st ~lead ~pad scales) (lift st ~lead ~pad codes)

let quant_batched (type a b) st (Nx_quant.Mxfp4 { codes; scales })
    (op : (a, b) Nx_quant.Effect.op) =
  batched st codes || batched st scales
  ||
  match op with
  | Apply { ids; x; _ } ->
      batched st x || Option.fold ~none:false ~some:(batched st) ids
  | Dequant _ -> false

(* The batched result of a quantised product with a lane among its operands. *)
let quant (type a b) st w (op : (a, b) Nx_quant.Effect.op) : (a, b) T.t =
  match op with
  | Dequant _ -> Nx_quant.Effect.perform (lift_weight st ~pad:0 w) op
  | Apply { ids; x; transpose } ->
      let ws = T.shape (match w with Nx_quant.Mxfp4 { codes; _ } -> codes)
      and xs = T.shape x in
      let vector = Array.length xs = 1 in
      let xb = if vector then [||] else Array.sub xs 0 (Array.length xs - 2) in
      let wb =
        match ids with
        | None -> Array.sub ws 0 (Array.length ws - 2)
        | Some ids -> T.shape ids
      in
      let rank = Stdlib.max (Array.length xb) (Array.length wb) in
      let pad = rank - Array.length wb in
      let x =
        lift st
          ?shape:(if vector then Some (Array.append [| 1 |] xs) else None)
          ~lead:1
          ~pad:(rank - Array.length xb)
          x
      in
      let ids = Option.map (lift st ~lead:1 ~pad) ids in
      let y =
        Nx_quant.Effect.perform (lift_weight st ~pad w)
          (Apply { ids; x; transpose })
      in
      if vector then
        let s = T.shape y in
        let r = Array.length s in
        T.reshape (Array.append (Array.sub s 0 (r - 2)) [| s.(r - 1) |]) y
      else y

(* The leaves at positions [flags] made lanes of [st]; the others stay. *)
let lanes_at st flags leaves =
  List.map2
    (fun b (Nx.P l) -> if b then Nx.P (lane st l) else Nx.P l)
    flags leaves

(* [leaves], which stand at the positions of [olds] outside the map, made lanes
   where [olds] are: a leaf that is an old lane's batched tensor comes back as
   that lane, so that a value passed out of the map and back keeps its
   identity. *)
let relanes st olds leaves =
  List.map2
    (fun (Nx.P o) (Nx.P l) ->
      match o with
      | Traced { t_node = Lane { map; batched }; _ } when map == st ->
          if Obj.repr batched == Obj.repr l then Nx.P o else Nx.P (lane st l)
      | Host _ | Placed _ | Traced _ -> Nx.P l)
    olds leaves

(* [install st f] is [f ()] with its operations batched by [st]: an
   interpreter of the operations, and a handler of the effects its rules
   answer, installed outside the interpreter. *)
let rec install : type a. state -> (unit -> a) -> a =
 fun (st : state) f ->
  let open Effect.Deep in
  let b x = batched st x in
  let p x = physical st x in
  (* Elementwise operations: broadcast all operands to the common batched shape
     and apply the operation unchanged. *)
  let elt2 (type a b c d) (f : (a, b) T.t -> (a, b) T.t -> (c, d) T.t)
      (a_in : (a, b) T.t) (b_in : (a, b) T.t) =
    let target = broadcast_shapes (T.shape a_in) (T.shape b_in) in
    lane st (f (to_batched st a_in target) (to_batched st b_in target))
  in
  (* Each operation with a lane among its operands is translated to its batched
     form: shape parameters gain a leading batch entry, axis parameters shift by
     one, and constants meeting lanes are lifted with a broadcast view.
     Operations whose operands are all constants of the map are evaluated as
     they are. *)
  let run : type c. c Nx.Op.t -> c =
   fun op ->
    match[@warning "@4@8"] op with
    | Unary (k, x) -> if b x then lane st (unary k (p x)) else eval op
    | Binary (k, x, y) -> if b x || b y then elt2 (binary k) x y else eval op
    | Compare (k, x, y) -> if b x || b y then elt2 (cmp k) x y else eval op
    | Threefry (key, ctr) ->
        if b key || b ctr then elt2 threefry key ctr else eval op
    | Convert (k, dtype, x) ->
        if b x then
          lane st
            (match k with
            | Cast -> cast dtype (p x)
            | Bitcast -> bitcast dtype (p x))
        else eval op
    | Contiguous x -> if b x then lane st (copy (p x)) else eval op
    | Where (c, x, y) ->
        if b c || b x || b y then
          let target =
            broadcast_shapes
              (broadcast_shapes (T.shape c) (T.shape x))
              (T.shape y)
          in
          lane st
            (where (to_batched st c target) (to_batched st x target)
               (to_batched st y target))
        else eval op
    (* A lane has no bytes of its own: its batched tensor holds every lane's. *)
    | Read x ->
        if b x then
          invalid_arg
            "Rune: cannot read the value of a batched tensor inside vmap; \
             return it from the mapped function instead"
        else eval op
    (* Placement: the batch axis sits in front of a split axis. *)
    | Place (q, x) ->
        if b x then
          lane st (place (T.Placement.with_leading_axis q) (p x))
        else eval op
    (* Movement: insert the batch dimension into shape parameters. *)
    | Move (x, m) -> if b x then lane st (move_batched st x m) else eval op
    | Pad (padding, v, x) ->
        if b x then lane st (pad (Array.append [| (0, 0) |] padding) v (p x))
        else eval op
    | Cat (axis, xs) ->
        if List.exists b xs then
          lane st (cat ~axis:(taxis axis) (List.map (ensure_batched st) xs))
        else eval op
    (* Reductions and scans *)
    | Reduce (k, axes, x) ->
        if b x then lane st (reduce k ~axes:(Array.map taxis axes) (p x))
        else eval op
    | Scan (k, axis, x) ->
        if b x then lane st (scan k ~axis:(taxis axis) (p x)) else eval op
    | Arg_reduce (k, axis, x) ->
        if b x then lane st (arg_reduce k ~axis:(taxis axis) (p x))
        else eval op
    | Sort { descending; axis; x } ->
        if b x then lane st (sort ~descending ~axis:(taxis axis) (p x))
        else eval op
    | Argsort { descending; axis; x } ->
        if b x then lane st (argsort ~descending ~axis:(taxis axis) (p x))
        else eval op
    (* Gather / scatter: operands agree on rank, so all are lifted. *)
    | Gather (axis, indices, data) ->
        if b data || b indices then
          lane st
            (gather ~axis:(taxis axis)
               (ensure_batched st indices)
               (ensure_batched st data))
        else eval op
    | Scatter { mode; unique; axis; indices; updates; into } ->
        if b into || b indices || b updates then
          lane st
            (scatter ~mode ~unique ~axis:(taxis axis)
               ~indices:(ensure_batched st indices)
               ~updates:(ensure_batched st updates)
               (ensure_batched st into))
        else eval op
    | Update (x, starts, v) ->
        if b x || b starts || b v then lane st (update_batched st x starts v)
        else eval op
    (* Matrix multiplication: the frontend promotes vectors to matrices against
       the lanes' shapes before this operation, and the backend broadcasts
       leading batch dimensions positionally. Plain matrices need no
       translation, the batch axis being the only leading dimension. When an
       operand carries leading dimensions of its own, both operands are lifted
       to the batched form at a common leading rank first; otherwise the batch
       axis of one would align against the other's first batch dimension. *)
    | Matmul (x, y) ->
        if b x || b y then
          let sa = T.shape x and sb = T.shape y in
          let lead s = Array.sub s 0 (Array.length s - 2) in
          let l =
            Stdlib.max (Array.length (lead sa)) (Array.length (lead sb))
          in
          if l = 0 then lane st (matmul (p x) (p y))
          else
            let padded s =
              Array.append (Array.make (l - Array.length (lead s)) 1) s
            in
            lane st
              (matmul
                 (to_batched st x (padded sa))
                 (to_batched st y (padded sb)))
        else eval op
    (* Windowing: both address the last spatial dimensions and pass every
       leading dimension through untouched, so the batch dimension rides along
       as one more leading dimension and no parameter shifts. *)
    | Unfold { kernel_size; stride; dilation; padding; x } ->
        if b x then
          lane st (unfold ~kernel_size ~stride ~dilation ~padding (p x))
        else eval op
    | Fold { output_size; kernel_size; stride; dilation; padding; x } ->
        if b x then
          lane st
            (fold ~output_size ~kernel_size ~stride ~dilation ~padding (p x))
        else eval op
    (* FFT: the transformed axes shift past the batch dimension; sizes ([s]) are
       per-axis and unchanged. *)
    | Fft { inverse; axes; x } ->
        if b x then lane st (fft ~inverse ~axes:(Array.map taxis axes) (p x))
        else eval op
    | Rfft { dtype; axes; x } ->
        if b x then lane st (rfft dtype ~axes:(Array.map taxis axes) (p x))
        else eval op
    | Irfft { dtype; axes; s; x } ->
        if b x then
          lane st (irfft ?s dtype ~axes:(Array.map taxis axes) (p x))
        else eval op
    | Cholesky { x; _ } -> if b x then no_rule "cholesky" () else eval op
    | Qr { x; _ } -> if b x then no_rule "qr" () else eval op
    | Lu x -> if b x then no_rule "lu" () else eval op
    | Svd { x; _ } -> if b x then no_rule "svd" () else eval op
    | Eig { x; _ } -> if b x then no_rule (Nx.Op.name op) () else eval op
    | Eigh { x; _ } -> if b x then no_rule (Nx.Op.name op) () else eval op
    | Solve_triangular { a; b = y; _ } ->
        if b a || b y then no_rule "solve_triangular" () else eval op
  in
  let flags s v = List.map (fun (Nx.P l) -> b l) (fst (Nx.Ptree.flatten s v)) in
  let physicals s v = Nx.Ptree.map s (fun _ l -> p l) v in
  let rule : type c. c Effect.t -> (unit -> c) option =
   fun eff ->
    match eff with
    (* The lane index is the per-lane iota [0 .. batch_size-1], a lane of
       scalars, so that a key folded with it decorrelates the lanes. A named
       map answers only the calls that name it. *)
    | Axis.E_lane_index axis when axis = st.axis ->
        Some (fun () -> lane st (T.arange Nx.int32 0 st.batch_size 1))
    (* The map a gather names answers it with the lanes as data: a fresh alias
       of a lane's batched tensor, or the broadcast of a constant every lane
       shares, a constant of the map either way. Another map gathers the
       lane's batched tensor and keeps its own lanes in front of the gathered
       axis. *)
    | Axis.E_lanes { axis; t_in } when st.axis = Some axis ->
        Some
          (fun () ->
            if b t_in then Structure.alias (p t_in)
            else
              T.broadcast_to
                (Array.append [| st.batch_size |] (T.shape t_in))
                t_in)
    | Axis.E_lanes { axis; t_in } when b t_in ->
        Some (fun () -> lane st (T.swapaxes 0 1 (Axis.lanes axis (p t_in))))
    (* Custom rules. A custom call passes on as the custom call of its batched
       functions, as a remat does, whatever its parameters: any of them may read
       a lane of this map, which only this map can batch. Each receives the
       batched tensors, makes lanes of those at the lanes' positions (a
       parameter's tangent is a lane where the parameter is) and runs under this
       map, and the results the call returns are made lanes where they came out
       lanes. The batched [fwd] of a custom vjp returns every result batched, so
       that each lane receives its own cotangents, and its batched [bwd] returns
       the cotangent of a lane batched and that of a constant summed over the
       lanes. The batched rule of a custom jvp returns a primal and its tangent
       batched together, as the claimer's shape check requires. *)
    | Custom.E_custom_vjp
        (Custom.Vjp_call { params_s; result_s; params; fwd; bwd }) ->
        let olds = fst (Nx.Ptree.flatten params_s params) in
        let fwd' ps =
          let ps =
            Nx.Ptree.rebuild params_s ~like:ps
              (relanes st olds (fst (Nx.Ptree.flatten params_s ps)))
          in
          let y, res = install st (fun () -> fwd ps) in
          (Nx.Ptree.map result_s (fun _ l -> ensure_batched st l) y, res)
        in
        let bwd' res cts =
          let cts = Nx.Ptree.map result_s (fun _ c -> lane st c) cts in
          let gs = install st (fun () -> bwd res cts) in
          Structure.map2 "Rune.custom_vjp" params_s ~this:"the parameters"
            ~that:"bwd's gradients"
            (fun _ q g -> if b q then ensure_batched st g else sum_lanes st g)
            params gs
        in
        Some
          (fun () ->
            let y =
              Custom.custom_vjp params_s result_s ~fwd:fwd' ~bwd:bwd'
                (physicals params_s params)
            in
            Nx.Ptree.map result_s (fun _ l -> lane st l) y)
    | Custom.E_custom_jvp
        (Custom.Jvp_call { params_s; result_s; params; f; jvp }) ->
        let olds = fst (Nx.Ptree.flatten params_s params) in
        let at = flags params_s params in
        let relanes_of ps =
          Nx.Ptree.rebuild params_s ~like:ps
            (relanes st olds (fst (Nx.Ptree.flatten params_s ps)))
        in
        let lanes_of ps =
          Nx.Ptree.rebuild params_s ~like:ps
            (lanes_at st at (fst (Nx.Ptree.flatten params_s ps)))
        in
        let out = ref [] in
        let f' ps =
          let y = install st (fun () -> f (relanes_of ps)) in
          out := flags result_s y;
          physicals result_s y
        in
        let jvp' ps dps =
          let y, dy =
            install st (fun () -> jvp (relanes_of ps) (lanes_of dps))
          in
          let both = ref [] in
          let batch_both (type a b) (y : (a, b) T.t) (dy : (a, b) T.t) =
            let l = b y || b dy in
            both := l :: !both;
            if l then (ensure_batched st y, ensure_batched st dy) else (y, dy)
          in
          let ys = ref [] in
          let dy =
            Structure.map2 "Rune.custom_jvp" result_s ~this:"the result"
              ~that:"jvp's tangents"
              (fun _ yl dyl ->
                let yl, dyl = batch_both yl dyl in
                ys := Nx.P yl :: !ys;
                dyl)
              y dy
          in
          out := List.rev !both;
          (Nx.Ptree.rebuild result_s ~like:y (List.rev !ys), dy)
        in
        Some
          (fun () ->
            let y =
              Custom.custom_jvp params_s result_s ~f:f' ~jvp:jvp'
                (physicals params_s params)
            in
            Nx.Ptree.rebuild result_s ~like:y
              (lanes_at st !out (fst (Nx.Ptree.flatten result_s y))))
    (* Gradient checkpointing: the remat passes on with its function batched, so
       that the enclosing context recomputes the batched computation, the
       tensors [f] captures included. The batched function receives the batched
       tensors of the arguments, those of the call or their aliases in a
       backward pass, makes lanes of those at the lanes' positions, and records
       which results come out lanes; the results the call returns, which may be
       aliases, are made lanes at those positions. *)
    | Remat.E_remat (Remat.Call { params_s; result_s; params; f; residuals }) ->
        let olds = fst (Nx.Ptree.flatten params_s params) in
        let out = ref [] in
        let f' ps =
          let ps =
            Nx.Ptree.rebuild params_s ~like:ps
              (relanes st olds (fst (Nx.Ptree.flatten params_s ps)))
          in
          let y = install st (fun () -> f ps) in
          out := flags result_s y;
          physicals result_s y
        in
        Some
          (fun () ->
            let y =
              Remat.run
                (Remat.Call
                   {
                     params_s;
                     result_s;
                     params = physicals params_s params;
                     f = f';
                     residuals;
                   })
            in
            Nx.Ptree.rebuild result_s ~like:y
              (lanes_at st !out (fst (Nx.Ptree.flatten result_s y))))
    | Remat.E_barrier { values; after } ->
        if not (List.exists (fun (Nx.P v) -> b v) values) then None
        else
          Some
            (fun () ->
              let unlane = List.map (fun (Nx.P v) -> Nx.P (p v)) in
              relanes st values
                (Remat.barrier ~after:(unlane after) (unlane values)))
    (* Scan. The claim is unconditional: the body may close over lanes that
       appear in neither the carry nor the rows. When a stager lies beyond, the
       scan passes on batched: a lane row has the scan axis moved in front of
       the batch axis, a lane carry stays a lane through every step, and the
       step runs the body under a nested instance of this map. A carry that is
       a constant at [init] becomes a lane only once a step makes it one: the
       step then aborts its run with [Grow] and the scan passes on again,
       batching it. Otherwise, the eager fold runs under a nested instance of
       this map. *)
    | Scan.E_scan_probe -> Some Scan.probe
    | Scan.E_scan req ->
        Some
          (fun () ->
            let fold () = install st (fun () -> Scan.eager req) in
            Scan.pass_on ~fold @@ fun () ->
              let exception Grow of bool list in
              let lanes_of = List.map (fun (Nx.P l) -> b l) in
              let batched_at carried leaves =
                List.map2
                  (fun c (Nx.P l) ->
                    if c then Nx.P (ensure_batched st l) else Nx.P l)
                  carried leaves
              in
              let rows = lanes_of req.req_xs in
              let xs =
                List.map2
                  (fun r (Nx.P x) ->
                    if r then Nx.P (T.swapaxes 0 1 (p x)) else Nx.P x)
                  rows req.req_xs
              in
              let rec attempt carried =
                let outputs = ref [] in
                let run c x =
                  let c', y =
                    install st (fun () ->
                        req.req_step.run (lanes_at st carried c)
                          (lanes_at st rows x))
                  in
                  let next = lanes_of c' in
                  if List.exists2 (fun c c' -> c' && not c) carried next then
                    raise (Grow (List.map2 ( || ) carried next));
                  outputs := lanes_of y;
                  ( batched_at carried c',
                    List.map (fun (Nx.P l) -> Nx.P (p l)) y )
                in
                match
                  Effect.perform
                    (Scan.E_scan
                       {
                         req with
                         req_carry = batched_at carried req.req_carry;
                         req_xs = xs;
                         req_step = { run };
                       })
                with
                | res ->
                    let r_ys =
                      List.map2
                        (fun o (Nx.P y) ->
                          if o then Nx.P (lane st (T.swapaxes 0 1 y))
                          else Nx.P y)
                        !outputs res.r_ys
                    in
                    { Scan.r_carry = lanes_at st carried res.r_carry; r_ys }
                (* The aborted run's slot tensors are never reached again. *)
                | exception Grow carried -> attempt carried
              in
              attempt (lanes_of req.req_carry))
    (* An addition to a total is the sum of its lanes' additions. *)
    | Total.E_add (t, v) -> Some (fun () -> Total.add t (sum_lanes st v))
    | Nx_quant.Effect.E_quant { w; op } when quant_batched st w op ->
        Some (fun () -> lane st (quant st w op))
    | _ -> None
  in
  let effc : type c. c Effect.t -> ((c, a) continuation -> a) option =
   fun eff -> Option.map Answer.deliver (rule eff)
  in
  match_with
    (fun () -> intercept { run } f)
    ()
    { retc = Fun.id; exnc = raise; effc }
