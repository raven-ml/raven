(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type ('a, _) t = { gamma : 'a; beta : 'a option }

let walk c { gamma; beta } =
  let open Nx.Ptree.Walk in
  let gamma = field c "gamma" leaf gamma in
  let beta = field c "beta" (option leaf) beta in
  { gamma; beta }

module Stats = struct
  type ('a, _) t = { mean : 'a; var : 'a }

  let walk c { mean; var } =
    let open Nx.Ptree.Walk in
    let mean = field c "mean" leaf mean in
    let var = field c "var" leaf var in
    { mean; var }
end

let init ?(shift = true) ~features at =
  if features <= 0 then
    invalid_argf "Batch_norm.init: features must be positive, got %d" features;
  let ones () = Nx.ones_at at Nx.float32 [| features |] in
  let zeros () = Nx.zeros_at at Nx.float32 [| features |] in
  let beta = if shift then Some (zeros ()) else None in
  ( { gamma = ones (); beta },
    { Stats.mean = zeros (); var = ones () } )

(* Half and quarter precision floats are too coarse for eval mode's affine
   from running statistics: it runs in a float32 island and rounds once.
   Wider dtypes keep their own arithmetic. *)
let low_precision : type b. (float, b) Nx.dtype -> bool = function
  | Float16 | Bfloat16 | Float8_e4m3fn | Float8_e5m2 | Float4_e2m1fn -> true
  | Float32 | Float64 -> false

let apply ?(momentum = 0.99) ?(eps = 1e-5) p (stats : _ Stats.t)
    ~training x =
  if momentum < 0.0 || momentum > 1.0 then
    Format.kasprintf invalid_arg
      "Batch_norm.apply: momentum must be in [0, 1], got %a"
      (Nx.Dtype.pp_value Nx.float64)
      momentum;
  if eps <= 0.0 then
    Format.kasprintf invalid_arg
      "Batch_norm.apply: eps must be positive, got %a"
      (Nx.Dtype.pp_value Nx.float64)
      eps;
  let ndim = Nx.ndim x in
  if ndim < 2 then
    invalid_argf "Batch_norm.apply: input must have at least 2 axes, got %d"
      ndim;
  let axis = ndim - 1 in
  let features = (Nx.shape p.gamma).(0) in
  if (Nx.shape x).(axis) <> features then
    invalid_argf
      "Batch_norm.apply: the last axis has size %d, parameters have %d \
       features"
      (Nx.shape x).(axis)
      features;
  (* Parameters and stats are [features]-vectors; give them rank [ndim] with the
     features last so they broadcast against [x]. *)
  let pshape = Array.make ndim 1 in
  pshape.(axis) <- features;
  let broadcast v = Nx.reshape pshape v in
  let axes = List.filter (fun i -> i <> axis) (List.init ndim Fun.id) in
  let gamma = broadcast p.gamma and beta = Option.map broadcast p.beta in
  if training then
    let y = Nx.standardize ~axes ~eps ~scale:gamma ?shift:beta x in
    let update old batch =
      let batch = Nx.reshape [| features |] batch in
      Nx.add (Nx.mul_s old momentum) (Nx.mul_s batch (1.0 -. momentum))
    in
    ( y,
      {
        Stats.mean = update stats.Stats.mean (Nx.mean ~axes x);
        var = update stats.var (Nx.var ~axes x);
      } )
  else
    (* [(x - mean) · rsqrt (var + eps) · gamma + beta] from the running
       statistics, without [+ beta] for a layer without a shift. *)
    let affine x mean var gamma beta =
      let n = Nx.mul (Nx.sub x mean) (Nx.rsqrt (Nx.add_s var eps)) in
      let y = Nx.mul n gamma in
      match beta with None -> y | Some beta -> Nx.add y beta
    in
    let mean = broadcast stats.Stats.mean and var = broadcast stats.var in
    let dt = Nx.dtype x in
    if not (low_precision dt) then (affine x mean var gamma beta, stats)
    else
      let up = Nx.cast Nx.float32 in
      ( Nx.cast dt
          (affine (up x) (up mean) (up var) (up gamma) (Option.map up beta)),
        stats )
