(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type 'a t = { gamma : 'a; beta : 'a }

let walk c { gamma; beta } =
  let open Nx.Ptree.Walk in
  let gamma = field c "gamma" leaf gamma in
  let beta = field c "beta" leaf beta in
  { gamma; beta }

module Stats = struct
  type 'a t = { mean : 'a; var : 'a }

  let walk c { mean; var } =
    let open Nx.Ptree.Walk in
    let mean = field c "mean" leaf mean in
    let var = field c "var" leaf var in
    { mean; var }
end

let init ~features =
  if features <= 0 then
    invalid_argf "Batch_norm.init: features must be positive, got %d" features;
  ( {
      gamma = Nx.ones Nx.float32 [| features |];
      beta = Nx.zeros Nx.float32 [| features |];
    },
    {
      Stats.mean = Nx.zeros Nx.float32 [| features |];
      var = Nx.ones Nx.float32 [| features |];
    } )

(* Half and quarter precision floats are too coarse for the statistics: their
   normalization runs in a float32 island (see [apply]). Wider dtypes keep their
   own arithmetic, so the float32 and float64 graphs are exactly the pre-island
   ones. *)
let low_precision : type b. (float, b) Nx.dtype -> bool = function
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 -> true
  | Nx.Float32 | Nx.Float64 -> false

let apply ?(axis = -1) ?(momentum = 0.99) ?(eps = 1e-5) p (stats : _ Stats.t)
    ~training x =
  if momentum < 0.0 || momentum > 1.0 then
    invalid_argf "Batch_norm.apply: momentum must be in [0, 1], got %g" momentum;
  if eps <= 0.0 then
    invalid_argf "Batch_norm.apply: eps must be positive, got %g" eps;
  let ndim = Nx.ndim x in
  if ndim < 2 then
    invalid_argf "Batch_norm.apply: input must have at least 2 axes, got %d"
      ndim;
  let axis = if axis < 0 then ndim + axis else axis in
  if axis < 0 || axis >= ndim then
    invalid_argf "Batch_norm.apply: axis out of bounds for a %d-d input" ndim;
  let features = (Nx.shape p.gamma).(0) in
  if (Nx.shape x).(axis) <> features then
    invalid_argf
      "Batch_norm.apply: feature axis has size %d, parameters have %d features"
      (Nx.shape x).(axis)
      features;
  (* Parameters and stats are [features]-vectors; give them rank [ndim] with the
     features on [axis] so they broadcast against [x]. *)
  let pshape = Array.make ndim 1 in
  pshape.(axis) <- features;
  let broadcast v = Nx.reshape pshape v in
  let axes = List.filter (fun i -> i <> axis) (List.init ndim Fun.id) in
  (* Batch statistics and normalization, generically in the dtype of [xs] and
     [stats]. Returns the normalized values and the batch statistics used
     (broadcast shape) so training can fold them into the running averages. *)
  let normalize xs stats_mean stats_var =
    let mean, var =
      if training then
        (Nx.mean ~axes ~keepdims:true xs, Nx.var ~axes ~keepdims:true xs)
      else (broadcast stats_mean, broadcast stats_var)
    in
    (Nx.mul (Nx.sub xs mean) (Nx.rsqrt (Nx.add_s var eps)), mean, var)
  in
  let dt = Nx.dtype x in
  let normalized, mean, var =
    if low_precision dt then begin
      let normalized, mean, var =
        normalize (Nx.cast Nx.float32 x)
          (Nx.cast Nx.float32 stats.Stats.mean)
          (Nx.cast Nx.float32 stats.var)
      in
      (Nx.cast dt normalized, Nx.cast dt mean, Nx.cast dt var)
    end
    else normalize x stats.Stats.mean stats.var
  in
  let y = Nx.add (Nx.mul normalized (broadcast p.gamma)) (broadcast p.beta) in
  if not training then (y, stats)
  else
    (* Running statistics are bookkeeping, not part of the differentiable
       computation: detach the batch statistics so no gradient ever flows
       through the returned stats. *)
    let update old batch =
      let batch = Rune.detach (Nx.reshape [| features |] batch) in
      Nx.add (Nx.mul_s old momentum) (Nx.mul_s batch (1.0 -. momentum))
    in
    ( y,
      { Stats.mean = update stats.Stats.mean mean; var = update stats.var var }
    )
