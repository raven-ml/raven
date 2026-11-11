(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let f64 = Nx.float64

(* Probes *)

type probe = { w : Nx.host Nx.float64_t; b : Nx.host Nx.float64_t }

(* The probe's search: its tolerance, its budget of iterations and L-BFGS's
   memory, the pairs of steps it keeps. *)
let tol = Jera.Tol.v ~rel:1e-6 ~abs:1e-6
let budget = 10_000
let memory = 10

let fit ~c x y =
  if not (c > 0.) then invalid_arg "Tessera.Eval.fit: c is not positive";
  if Nx.ndim x <> 2 then invalid_arg "Tessera.Eval.fit: x is not [n; d]";
  let n = Nx.dim 0 x and d = Nx.dim 1 x in
  if Nx.shape y <> [| n |] then invalid_arg "Tessera.Eval.fit: y is not [n]";
  let labels = Nx.to_array y in
  if n = 0 then invalid_arg "Tessera.Eval.fit: no row";
  Array.iter
    (fun l ->
      if l < 0L then
        invalid_arg (Printf.sprintf "Tessera.Eval.fit: label %Ld is negative" l))
    labels;
  let classes = 1 + Int64.to_int (Array.fold_left max 0L labels) in
  let seen = Array.make classes false in
  Array.iter (fun l -> seen.(Int64.to_int l) <- true) labels;
  Array.iteri
    (fun k s ->
      if not s then
        invalid_arg (Printf.sprintf "Tessera.Eval.fit: class %d has no row" k))
    seen;
  let x = Nx.cast f64 x in
  let onehot = Nx.cast f64 (Nx.one_hot ~num_classes:classes y) in
  let penalty = 1. /. (2. *. c) in
  let objective (w, b) =
    let logits = Nx.add (Nx.matmul x w) b in
    let nll =
      Nx.sub
        (Nx.sum (Nx.logsumexp ~axes:[ 1 ] logits))
        (Nx.sum (Nx.mul onehot logits))
    in
    Nx.add nll (Nx.mul_s (Nx.sum (Nx.square w)) penalty)
  in
  let x0 = (Nx.zeros f64 [| d; classes |], Nx.zeros f64 [| classes |]) in
  let linear = Jera.Linear.cg ~rel:1e-10 ~budget:1000 ~precondition:Fun.id in
  let w, b =
    Jera.Solution.get
      (Jera.Minimize.solve
         Nx.Ptree.(pair tensor tensor)
         (Jera.Minimize.lbfgs ~memory ~linear)
         ~tol ~budget objective x0)
  in
  { w; b }

let predict p x = Nx.argmax ~axis:1 (Nx.add (Nx.matmul (Nx.cast f64 x) p.w) p.b)

type scores = { accuracy : float; f1 : float }

let score ~predicted y =
  if Nx.shape predicted <> Nx.shape y || Nx.ndim y <> 1 || Nx.numel y = 0 then
    invalid_arg "Tessera.Eval.score: predictions and labels of other shapes";
  let p = Nx.to_array predicted and l = Nx.to_array y in
  let n = Array.length l in
  let k = 1 + Int64.to_int (Array.fold_left max 0L (Array.append p l)) in
  let t = Array.make k 0 and f = Array.make k 0 and m = Array.make k 0 in
  Array.iteri
    (fun i li ->
      let li = Int64.to_int li and pi = Int64.to_int p.(i) in
      if li = pi then t.(li) <- t.(li) + 1
      else begin
        f.(pi) <- f.(pi) + 1;
        m.(li) <- m.(li) + 1
      end)
    l;
  let correct = Array.fold_left ( + ) 0 t in
  let f1 = ref 0. in
  for c = 0 to k - 1 do
    let support = t.(c) + m.(c) in
    if t.(c) > 0 then
      let fc =
        Float.of_int (2 * t.(c)) /. Float.of_int ((2 * t.(c)) + f.(c) + m.(c))
      in
      f1 := !f1 +. (fc *. Float.of_int support)
  done;
  {
    accuracy = Float.of_int correct /. Float.of_int n;
    f1 = !f1 /. Float.of_int n;
  }

(* RankMe *)

(* The guard on the sum of singular values and on each share, as TESSERA's
   trainer computes RankMe (tessera-util@290feaf src/utils/metrics.py:39). *)
let rankme_eps = 1e-7

let rankme z =
  if Nx.ndim z <> 2 || min (Nx.dim 0 z) (Nx.dim 1 z) < 2 then
    invalid_arg "Tessera.Eval.rankme: z is not [n; d] with min n d >= 2";
  let s = Nx.svdvals f64 (Nx.cast f64 (Nx.place Nx.Host.on z)) in
  let p = Nx.div s (Nx.add_s (Nx.sum s) rankme_eps) in
  let h = Nx.neg (Nx.sum (Nx.mul p (Nx.log (Nx.add_s p rankme_eps)))) in
  Nx.item [] h /. Float.log (Float.of_int (Nx.dim 0 s))
