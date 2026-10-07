(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('u, 'f) t = {
  sample : ('u, 'f) Weighted.t;
  log_evidence : (float, 'f) Nx.t;
  error : (float, 'f) Nx.t;
  information : (float, 'f) Nx.t;
  stop : Nx.int32_t; (* [converged], [remaining] or [temperature] *)
  reached : (float, 'f) Nx.t; (* the payload of [stop] *)
  replicates : (float, 'f) Nx.t;
      (* the sample's normalised log weights under each simulated shrinkage
         sequence, [[r; n]]; one row, the weights themselves, without one *)
  groups : Nx.int32_t;
      (* each draw's group, from 0 to [group_count]: draws of one group may be
         correlated, draws of two are independent *)
  group_count : int;
}

type ('u, 'f) evidence = ('u, 'f) t

let converged = 0l
let remaining = 1l
let temperature = 2l

(* [v ... ~stop ~reached ~replicates ~groups] is an estimate that stopped as
   [stop] codes it: [converged], [remaining] or [temperature], whose payload is
   [reached], a scalar. The error of an expectation reads [replicates] and
   [groups]. *)
let v ~sample ~log_evidence ~error ~information ~stop ~reached ~replicates
    ~groups ~group_count =
  {
    sample;
    log_evidence;
    error;
    information;
    stop;
    reached;
    replicates;
    groups;
    group_count;
  }

let log_evidence z = z.log_evidence
let error z = z.error
let information z = z.information
let sample z = z.sample

let ptree (type u f) (u : u Nx.Ptree.t) : (u, f) t Nx.Ptree.t =
  let w = Weighted.ptree u in
  let module S = struct
    type _ t = (u, f) evidence

    let walk c z =
      let open Nx.Ptree.Walk in
      let sample = field c "sample" (structure w) z.sample in
      let log_evidence = field c "log_evidence" tensor z.log_evidence in
      let error = field c "error" tensor z.error in
      let information = field c "information" tensor z.information in
      let stop = field c "stop" tensor z.stop in
      let reached = field c "reached" tensor z.reached in
      let replicates = field c "replicates" tensor z.replicates in
      let groups = field c "groups" tensor z.groups in
      let group_count = field c "group_count" int z.group_count in
      {
        sample;
        log_evidence;
        error;
        information;
        stop;
        reached;
        replicates;
        groups;
        group_count;
      }
  end in
  Nx.Ptree.nest (module S) Nx.Ptree.unit

(* Expectations

   An expectation's variance has two parts. Over the replicates, the spread of
   the expectation is the noise of the weights themselves, nested sampling's
   unknown shrinkage. Over the groups, independent sets of draws, the variance
   of a self-normalised mean is [Σ_g W_g² (m_g - m)²], [W_g] a group's weight
   and [m_g] its mean, scaled by [G / (G - 1)]: a group of one draw makes it the
   delta method's [Σ_i w_i² (f_i - m)²], and a group per lineage or chain of
   correlated draws counts their correlation. *)

let expectation (type u f) (u : u Nx.Ptree.t) (f : u -> (float, f) Nx.t)
    (z : (u, f) t) =
  let fx = Rune.vmap Nx.Ptree.(u @-> returns tensor) f z.sample.values in
  let n = (Nx.shape fx).(0) in
  let shape = Array.sub (Nx.shape fx) 1 (Nx.ndim fx - 1) in
  let fx = Nx.reshape [| n; -1 |] fx in
  let w = Nx.exp z.sample.log_weights in
  (* Draws of weight zero, such as padding, carry no value. *)
  let fx =
    Nx.where
      (Nx.reshape [| n; 1 |] (Nx.greater w (Nx.zeros_like w)))
      fx (Nx.zeros_like fx)
  in
  let mean = Nx.matmul (Nx.reshape [| 1; n |] w) fx in
  let replicated = Nx.matmul (Nx.exp z.replicates) fx in
  let shrinkage = Nx.var ~axes:[ 0 ] ~keepdims:true replicated in
  let g = z.group_count in
  let e = (Nx.shape fx).(1) in
  let ids = Nx.cast Nx.int64 z.groups in
  let per_group values like =
    Nx.scatter ~mode:`Add ~axis:0
      ~indices:(Nx.broadcast_to (Nx.shape values) (Nx.reshape [| n; 1 |] ids))
      ~values like
  in
  let total =
    per_group (Nx.reshape [| n; 1 |] w) (Nx.zeros (Nx.dtype w) [| g; 1 |])
  in
  let sums =
    per_group
      (Nx.mul (Nx.reshape [| n; 1 |] w) fx)
      (Nx.zeros (Nx.dtype w) [| g; e |])
  in
  let means =
    Nx.div sums
      (Nx.where
         (Nx.greater total (Nx.zeros_like total))
         total (Nx.ones_like total))
  in
  let sampling =
    Nx.mul_s
      (Nx.sum ~axes:[ 0 ] ~keepdims:true
         (Nx.mul (Nx.square total) (Nx.square (Nx.sub means mean))))
      (float_of_int g /. float_of_int (max 1 (g - 1)))
  in
  (Nx.reshape shape mean, Nx.reshape shape (Nx.sqrt (Nx.add shrinkage sampling)))

type stop = Converged | Remaining of float | Temperature of float

let float x = Nx.item [] (Nx.cast Nx.float64 x)

let stop z =
  let code = Nx.item [] z.stop in
  if code = remaining then Remaining (float z.reached)
  else if code = temperature then Temperature (float z.reached)
  else Converged

let pp ppf z =
  Format.fprintf ppf "ln Z = %.2f ± %.2f, H = %.2f nats" (float z.log_evidence)
    (float z.error) (float z.information);
  match stop z with
  | Converged -> ()
  | Remaining r -> Format.fprintf ppf ", budget spent with %.2f nats left" r
  | Temperature b -> Format.fprintf ppf ", budget spent at β = %.2f" b
