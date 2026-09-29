(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Vectorizing maps as an interpreter of Nx operations.

   The mapped function is written for unbatched values. Under the handler, every
   tensor is either batched — it physically carries the batch dimension,
   canonically at axis 0 — or a constant of the map. Two mechanisms keep this
   transparent to the function and to the Nx frontend itself:

   - Shape queries go through the [E_view] effect; for batched tensors the
   handler answers with the unbatched remainder of the view. The frontend
   therefore makes exactly the decisions of the unbatched program (broadcasting,
   promotion, reshapes), and the handler translates each resulting primitive to
   its batched form. - Each primitive's translation inserts the batch dimension:
   shape parameters gain a leading batch entry, axis parameters shift by one,
   and constants meeting batched operands are lifted with a broadcast view.

   The virtual view presents the remainder as contiguous even when the batched
   tensor is not; rules compensate by forcing contiguity before reshapes.
   Operations whose operands are all constants are evaluated as they are.
   Nested vmaps stack: each handler owns its batched set and batch size, and the
   translations one level emits are re-translated by the level above.

   Every operation is matched explicitly, and the compiler checks it; operations
   without a batching rule raise when an operand is batched rather than
   silently producing wrong shapes. *)

open Nx_effect
module T = Nx

let no_rule op () =
  invalid_arg (Printf.sprintf "Rune: vmap has no batching rule for %s" op)

type state = {
  batch_size : int;
  batched : Tensor_map.Ids.t;
  axis : Axis.t option;
}

let create ?axis ~batch_size () =
  { batch_size; batched = Tensor_map.Ids.create (); axis }

let mark st x = Tensor_map.Ids.add st.batched x
let batched st x = Tensor_map.Ids.mem st.batched x

(* The unbatched shape a tensor presents to the mapped function. *)
let vshape st x =
  let s = T.shape x in
  if batched st x then Array.sub s 1 (Array.length s - 1) else s

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

(* [to_batched st x target] is [x] as a physically batched tensor of shape
   [batch_size :: target], where [target] is a virtual shape [x]'s virtual shape
   broadcasts to. Constants are lifted with a broadcast view. *)
let to_batched st x target =
  let s = vshape st x in
  if batched st x && s = target then x
  else begin
    let ones = Array.make (Array.length target - Array.length s) 1 in
    let lead = if batched st x then st.batch_size else 1 in
    let x = reshape (contiguous x) (Array.concat [ [| lead |]; ones; s ]) in
    expand x (Array.append [| st.batch_size |] target)
  end

let ensure_batched st x = to_batched st x (vshape st x)

(* The sum over the lanes of a value each lane holds: a batched one summed along
   its batch axis, one every lane shares times their number. *)
let sum_lanes st v =
  if batched st v then T.sum ~axes:[ 0 ] v
  else T.mul_s v (Nx_dtype.of_float (T.dtype v) (Float.of_int st.batch_size))

(* Axis parameters count from the virtual shape; the batch dimension sits at 0,
   so non-negative axes shift by one and negative axes are unchanged. *)
let taxis ax = if ax >= 0 then ax + 1 else ax

(* [x], physically batched, moved by [m]: the batch dimension enters shape
   parameters and shifts axes by one. *)
let move_batched st x m =
  match (m : move) with
  | Reshape shape ->
      reshape (contiguous x) (Array.append [| st.batch_size |] shape)
  | Permute axes ->
      permute x (Array.append [| 0 |] (Array.map (fun d -> d + 1) axes))
  | Expand shape -> to_batched st x shape
  | Shrink limits -> shrink x (Array.append [| (0, st.batch_size) |] limits)
  | Flip dims -> flip x (Array.append [| false |] dims)
  (* The trailing window axis lands at the physical end, which is the virtual
     end shifted past the batch dimension. *)
  | Window { axis; size; step } ->
      sliding_window x ~axis:(taxis axis) ~window:size ~step

(* The window write of [v] into [x] at [starts], one of them batched. *)
let update_batched st x starts v =
  let t = ensure_batched st x and v' = ensure_batched st v in
  if batched st starts then begin
    (* A window per example: along each axis the rows are gathered from [v]
       at the example's offset, clamped, and kept where they fall inside the
       window. *)
    let starts = ensure_batched st starts in
    let b = st.batch_size in
    let tshape = vshape st x and vs = vshape st v in
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
   [ids] and of [x], a unit axis where one is unbatched, after [pad] unit axes
   that align the operands' batch axes, so that each lane meets its own weight
   with its own ids. *)

let lift st ?shape ~lead ~pad x =
  let s = match shape with Some s -> s | None -> vshape st x in
  let b = if batched st x then st.batch_size else 1 in
  let x = T.reshape (Array.concat [ [| b |]; Array.make pad 1; s ]) x in
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

let quant (type a b) st w (op : (a, b) Nx_quant.Effect.op) : (a, b) t =
  match op with
  | Dequant _ -> Nx_quant.Effect.perform (lift_weight st ~pad:0 w) op
  | Apply { ids; x; transpose } ->
      let ws = vshape st (match w with Nx_quant.Mxfp4 { codes; _ } -> codes)
      and xs = vshape st x in
      let vector = Array.length xs = 1 in
      let xb = if vector then [||] else Array.sub xs 0 (Array.length xs - 2) in
      let wb =
        match ids with
        | None -> Array.sub ws 0 (Array.length ws - 2)
        | Some ids -> vshape st ids
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

let rec handler : type r. state -> (r, r) Effect.Deep.handler =
 fun (st : state) ->
  let open Effect.Deep in
  let b x = batched st x in
  (* [out], marked batched. *)
  let lane out =
    mark st out;
    out
  in
  (* Elementwise operations: broadcast all operands to the common batched shape
     and apply the operation unchanged. *)
  let elt2 (type a b c d) (f : (a, b) t -> (a, b) t -> (c, d) t)
      (a_in : (a, b) t) (b_in : (a, b) t) =
    let target = broadcast_shapes (vshape st a_in) (vshape st b_in) in
    lane (f (to_batched st a_in target) (to_batched st b_in target))
  in
  (* Each operation with a batched operand is translated to its batched form:
     shape parameters gain a leading batch entry, axis parameters shift by one,
     and constants meeting batched operands are lifted with a broadcast view.
     Operations whose operands are all constants of the map are evaluated as
     they are. *)
  let run : type c. c Op.t -> c =
   fun op ->
    match[@warning "@4@8"] op with
    | Unary (k, x) -> if b x then lane (unary k x) else eval op
    | Binary (k, x, y) -> if b x || b y then elt2 (binary k) x y else eval op
    | Compare (k, x, y) -> if b x || b y then elt2 (cmp k) x y else eval op
    | Threefry (key, ctr) ->
        if b key || b ctr then elt2 threefry key ctr else eval op
    | Convert (k, dtype, x) ->
        if b x then
          lane
            (match k with Cast -> cast dtype x | Bitcast -> bitcast dtype x)
        else eval op
    | Contiguous x -> if b x then lane (copy x) else eval op
    | Where (c, x, y) ->
        if b c || b x || b y then
          let target =
            broadcast_shapes
              (broadcast_shapes (vshape st c) (vshape st x))
              (vshape st y)
          in
          lane
            (where (to_batched st c target) (to_batched st x target)
               (to_batched st y target))
        else eval op
    (* Reading the value of a batched tensor would expose the physical, batched
       buffer to code that believes it is unbatched. *)
    | Read x ->
        if b x then
          invalid_arg
            "Rune: cannot read the value of a batched tensor inside vmap; \
             return it from the mapped function instead"
        else eval op
    (* Placement: the batch axis sits in front of a split axis. *)
    | Place (p, x) ->
        if b x then lane (place (Nx_effect.Placement.with_leading_axis p) x)
        else eval op
    (* Movement: insert the batch dimension into shape parameters. *)
    | Move (x, m) -> if b x then lane (move_batched st x m) else eval op
    | Pad (padding, v, x) ->
        if b x then lane (pad (Array.append [| (0, 0) |] padding) v x)
        else eval op
    | Cat (axis, xs) ->
        if List.exists b xs then
          lane (cat ~axis:(taxis axis) (List.map (ensure_batched st) xs))
        else eval op
    (* Reductions and scans *)
    | Reduce (k, axes, x) ->
        if b x then lane (reduce k ~axes:(Array.map taxis axes) x) else eval op
    | Scan (k, axis, x) ->
        if b x then lane (scan k ~axis:(taxis axis) x) else eval op
    | Arg_reduce (k, axis, x) ->
        if b x then lane (arg_reduce k ~axis:(taxis axis) x) else eval op
    | Sort { descending; axis; x } ->
        if b x then lane (sort ~descending ~axis:(taxis axis) x) else eval op
    | Argsort { descending; axis; x } ->
        if b x then lane (argsort ~descending ~axis:(taxis axis) x)
        else eval op
    (* Gather / scatter: operands agree on rank, so all are lifted. *)
    | Gather (axis, indices, data) ->
        if b data || b indices then
          lane
            (gather ~axis:(taxis axis)
               (ensure_batched st indices)
               (ensure_batched st data))
        else eval op
    | Scatter { mode; unique; axis; indices; updates; into } ->
        if b into || b indices || b updates then
          lane
            (scatter ~mode ~unique ~axis:(taxis axis)
               ~indices:(ensure_batched st indices)
               ~updates:(ensure_batched st updates)
               (ensure_batched st into))
        else eval op
    | Update (x, starts, v) ->
        if b x || b starts || b v then lane (update_batched st x starts v)
        else eval op
    (* Matrix multiplication: the frontend promotes vectors to matrices against
       virtual shapes before this operation, and the backend broadcasts leading
       batch dimensions positionally. Plain matrices need no translation, the
       batch axis being the only leading dimension. When an operand carries
       leading dimensions of its own, both operands are lifted to the batched
       form at a common leading rank first; otherwise the batch axis of one
       would align against the other's first batch dimension. *)
    | Matmul (x, y) ->
        if b x || b y then
          let sa = vshape st x and sb = vshape st y in
          let lead s = Array.sub s 0 (Array.length s - 2) in
          let l =
            Stdlib.max (Array.length (lead sa)) (Array.length (lead sb))
          in
          if l = 0 then lane (matmul x y)
          else
            let padded s =
              Array.append (Array.make (l - Array.length (lead s)) 1) s
            in
            lane
              (matmul
                 (to_batched st x (padded sa))
                 (to_batched st y (padded sb)))
        else eval op
    (* Windowing: both address the last spatial dimensions and pass every
       leading dimension through untouched, so the batch dimension rides along
       as one more leading dimension and no parameter shifts. *)
    | Unfold { x; _ } -> if b x then lane (eval op) else eval op
    | Fold { x; _ } -> if b x then lane (eval op) else eval op
    (* FFT: the transformed axes shift past the batch dimension; sizes ([s]) are
       per-axis and unchanged. *)
    | Fft { inverse; axes; x } ->
        if b x then lane (fft ~inverse ~axes:(Array.map taxis axes) x)
        else eval op
    | Rfft { dtype; axes; x } ->
        if b x then lane (rfft dtype ~axes:(Array.map taxis axes) x)
        else eval op
    | Irfft { dtype; axes; s; x } ->
        if b x then lane (irfft ?s dtype ~axes:(Array.map taxis axes) x)
        else eval op
    | Cholesky { x; _ } -> if b x then no_rule "cholesky" () else eval op
    | Qr { x; _ } -> if b x then no_rule "qr" () else eval op
    | Lu x -> if b x then no_rule "lu" () else eval op
    | Svd { x; _ } -> if b x then no_rule "svd" () else eval op
    | Eig { x; _ } -> if b x then no_rule (Op.name op) () else eval op
    | Eigh { x; _ } -> if b x then no_rule (Op.name op) () else eval op
    | Solve_triangular { a; b = y; _ } ->
        if b a || b y then no_rule "solve_triangular" () else eval op
  in
  let rule : type c. c Effect.t -> (unit -> c) option =
   fun eff ->
    match eff with
    | E_op op -> Some (fun () -> run op)
    (* Shape queries: batched tensors present their unbatched remainder, as a
       contiguous view. *)
    | E_view x ->
        if batched st x then
          Some
            (fun () ->
              let s = T.shape x in
              Nx_array.View.create (Array.sub s 1 (Array.length s - 1)))
        else None
    (* A lane of a map over the split axis has no placement of its own. *)
    | E_placement x when batched st x ->
        Some
          (fun () ->
            match Nx_effect.Placement.without_leading_axis (placement x) with
            | None ->
                invalid_arg
                  "Rune: a lane of vmap over a split axis has no placement"
            | Some p -> p)
    (* The lane index is the per-lane iota [0 .. batch_size-1], carried as a
       batched scalar, so that a key folded with it decorrelates the lanes. A
       named map answers only the calls that name it. *)
    | Axis.E_lane_index axis when axis = st.axis ->
        Some
          (fun () ->
            let idx = T.arange Nx.int32 0 st.batch_size 1 in
            mark st idx;
            idx)
    (* The map a gather names answers it with the lanes as data: a fresh alias
       of a batched operand's physical tensor, or the broadcast of an operand
       every lane shares, a constant of the map either way. Another map gathers
       its batched operand's physical tensor and keeps its own lanes in front
       of the gathered axis. *)
    | Axis.E_lanes { axis; t_in } when st.axis = Some axis ->
        Some
          (fun () ->
            if batched st t_in then Structure.alias t_in
            else
              T.broadcast_to
                (Array.append [| st.batch_size |] (T.shape t_in))
                t_in)
    | Axis.E_lanes { axis; t_in } when batched st t_in ->
        Some
          (fun () ->
            let out = T.swapaxes 0 1 (Axis.lanes axis t_in) in
            mark st out;
            out)
    (* Custom rules. A custom call passes on as the custom call of its batched
       functions, as a remat does, whatever its parameters: any of them may read
       a tensor this map batches, which only this handler reads as lanes. Each
       receives the physical tensors, marks those at the batched parameters'
       positions (a parameter's tangent is batched where the parameter is) and
       runs under this handler, and the results the call returns are marked
       where they came out batched. The batched [fwd] of a custom vjp returns
       every result batched, so that each lane receives its own cotangents, and
       its batched [bwd] returns the cotangent of a parameter this map batches
       batched and that of one it does not summed over the lanes. The batched
       rule of a custom jvp returns a primal and its tangent batched together,
       as the claimer's shape check requires. *)
    | Custom.E_custom_vjp
        (Custom.Vjp_call { params_s; result_s; params; fwd; bwd }) ->
        let flags =
          List.map
            (fun (Nx.P p) -> batched st p)
            (fst (Nx.Ptree.flatten params_s params))
        in
        let fwd' ps =
          List.iter2
            (fun (Nx.P p) b -> if b then mark st p)
            (fst (Nx.Ptree.flatten params_s ps))
            flags;
          let y, res = match_with fwd ps (handler st) in
          (Nx.Ptree.map result_s (fun _ l -> ensure_batched st l) y, res)
        in
        let bwd' res cts =
          Nx.Ptree.fold result_s (fun _ c () -> mark st c) cts ();
          let gs = match_with (fun () -> bwd res cts) () (handler st) in
          Structure.map2 "Rune.custom_vjp" params_s ~this:"the parameters"
            ~that:"bwd's gradients"
            (fun _ p g ->
              if batched st p then ensure_batched st g else sum_lanes st g)
            params gs
        in
        Some
          (fun () ->
            let y =
              Custom.custom_vjp params_s result_s ~fwd:fwd' ~bwd:bwd' params
            in
            Nx.Ptree.fold result_s (fun _ l () -> mark st l) y ();
            y)
    | Custom.E_custom_jvp
        (Custom.Jvp_call { params_s; result_s; params; f; jvp }) ->
        let flags =
          List.map
            (fun (Nx.P p) -> batched st p)
            (fst (Nx.Ptree.flatten params_s params))
        in
        let mark_params ps =
          List.iter2
            (fun (Nx.P p) b -> if b then mark st p)
            (fst (Nx.Ptree.flatten params_s ps))
            flags
        in
        let out = ref [] in
        let f' ps =
          mark_params ps;
          let y = match_with f ps (handler st) in
          out :=
            List.map
              (fun (Nx.P l) -> batched st l)
              (fst (Nx.Ptree.flatten result_s y));
          y
        in
        let jvp' ps dps =
          mark_params ps;
          mark_params dps;
          let y, dy = match_with (fun () -> jvp ps dps) () (handler st) in
          let both = ref [] in
          let batch_both (type a b) (y : (a, b) t) (dy : (a, b) t) =
            let b = batched st y || batched st dy in
            both := b :: !both;
            if b then (ensure_batched st y, ensure_batched st dy) else (y, dy)
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
              Custom.custom_jvp params_s result_s ~f:f' ~jvp:jvp' params
            in
            List.iter2
              (fun (Nx.P l) b -> if b then mark st l)
              (fst (Nx.Ptree.flatten result_s y))
              !out;
            y)
    (* Gradient checkpointing: the remat passes on with its function batched, so
       that the enclosing context recomputes the batched computation, the
       tensors [f] captures included. The batched function receives the physical
       tensors of the arguments, those of the call or their aliases in a
       backward pass, marks those at the batched arguments' positions, and
       records which results come out batched; the results the call returns,
       which may be aliases, are marked at those positions. *)
    | Remat.E_remat (Remat.Call { params_s; result_s; params; f; residuals }) ->
        let flags =
          List.map
            (fun (Nx.P p) -> batched st p)
            (fst (Nx.Ptree.flatten params_s params))
        in
        let out = ref [] in
        let f' params =
          List.iter2
            (fun (Nx.P p) b -> if b then mark st p)
            (fst (Nx.Ptree.flatten params_s params))
            flags;
          let y = match_with f params (handler st) in
          out :=
            List.map
              (fun (Nx.P l) -> batched st l)
              (fst (Nx.Ptree.flatten result_s y));
          y
        in
        Some
          (fun () ->
            let y =
              Remat.run
                (Remat.Call { params_s; result_s; params; f = f'; residuals })
            in
            List.iter2
              (fun (Nx.P l) b -> if b then mark st l)
              (fst (Nx.Ptree.flatten result_s y))
              !out;
            y)
    | Remat.E_barrier { values; after } ->
        if not (List.exists (fun (Nx.P v) -> batched st v) values) then None
        else
          Some
            (fun () ->
              let out = Remat.barrier ~after values in
              List.iter2
                (fun (Nx.P v) (Nx.P o) -> if batched st v then mark st o)
                values out;
              out)
    (* Scan. The claim is unconditional: the body may close over batched tensors
       that appear in neither the carry nor the rows. When a stager lies beyond,
       the scan passes on batched: a batched row has the scan axis moved in
       front of the lane, a batched carry stays batched through every step, and
       the step runs the body under a nested instance of this handler. A carry
       unbatched at [init] becomes batched only once a step batches it: the step
       then aborts its run with [Grow] and the scan passes on again, batching
       it. Otherwise, the eager fold runs under a nested instance of this
       handler. *)
    | Scan.E_scan_probe -> Some Scan.probe
    | Scan.E_scan req ->
        Some
          (fun () ->
            let fold () =
              match_with (fun () -> Scan.eager req) () (handler st)
            in
            Scan.pass_on ~fold @@ fun () ->
              let exception Grow of bool list in
              let flags = List.map (fun (Nx.P l) -> batched st l) in
              let lanes carried leaves =
                List.map2
                  (fun b (Nx.P l) ->
                    if b then Nx.P (ensure_batched st l) else Nx.P l)
                  carried leaves
              in
              let rows = flags req.req_xs in
              let xs =
                List.map2
                  (fun b (Nx.P x) ->
                    if b then Nx.P (T.swapaxes 0 1 x) else Nx.P x)
                  rows req.req_xs
              in
              let rec attempt carried =
                let outputs = ref [] in
                let run c x =
                  List.iter2 (fun b (Nx.P c) -> if b then mark st c) carried c;
                  List.iter2 (fun b (Nx.P x) -> if b then mark st x) rows x;
                  let c', y =
                    match_with (fun () -> req.req_step.run c x) () (handler st)
                  in
                  let next = flags c' in
                  if List.exists2 (fun b b' -> b' && not b) carried next then
                    raise (Grow (List.map2 ( || ) carried next));
                  outputs := flags y;
                  (lanes carried c', y)
                in
                match
                  Effect.perform
                    (Scan.E_scan
                       {
                         req with
                         req_carry = lanes carried req.req_carry;
                         req_xs = xs;
                         req_step = { run };
                       })
                with
                | res ->
                    List.iter2
                      (fun b (Nx.P c) -> if b then mark st c)
                      carried res.r_carry;
                    let r_ys =
                      List.map2
                        (fun b (Nx.P y) ->
                          if b then (
                            let y = T.swapaxes 0 1 y in
                            mark st y;
                            Nx.P y)
                          else Nx.P y)
                        !outputs res.r_ys
                    in
                    { res with r_ys }
                (* The aborted run's slot tensors are never reached again. *)
                | exception Grow carried -> attempt carried
              in
              attempt (flags req.req_carry))
    (* An addition to a total is the sum of its lanes' additions. *)
    | Total.E_add (t, v) -> Some (fun () -> Total.add t (sum_lanes st v))
    | Nx_quant.Effect.E_quant { w; op } when quant_batched st w op ->
        Some
          (fun () ->
            let out = quant st w op in
            mark st out;
            out)
    | _ -> None
  in
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff -> Option.map Gate.deliver (rule eff)
  in
  { retc = Fun.id; exnc = raise; effc }
