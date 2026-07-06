(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type t = { gamma : Nx.float32_t; beta : Nx.float32_t }

let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { gamma; beta } =
  { gamma = f gamma; beta = f beta }

let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
  { gamma = f p.gamma q.gamma; beta = f p.beta q.beta }

let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { gamma; beta } =
  f gamma;
  f beta

let names _ = [ "gamma"; "beta" ]

module Stats = struct
  type t = { mean : Nx.float32_t; var : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { mean; var } =
    { mean = f mean; var = f var }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) s s' =
    { mean = f s.mean s'.mean; var = f s.var s'.var }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { mean; var } =
    f mean;
    f var

  let names _ = [ "mean"; "var" ]
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

let apply ?(axis = -1) ?(momentum = 0.99) ?(eps = 1e-5) p (stats : Stats.t)
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
  let mean, var =
    if training then
      (Nx.mean ~axes ~keepdims:true x, Nx.var ~axes ~keepdims:true x)
    else (broadcast stats.Stats.mean, broadcast stats.Stats.var)
  in
  let normalized = Nx.mul (Nx.sub x mean) (Nx.rsqrt (Nx.add_s var eps)) in
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
