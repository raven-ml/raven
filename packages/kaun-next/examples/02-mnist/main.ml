(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* MNIST with kaun-next: an MLP as a plain record, minibatches from
   Data.batches2, AdamW steps from Vega, accuracy from Metric. *)

open Kaun_next

let batch_size = 128
let lr = 1e-3

module Mlp = struct
  type t = { l1 : Linear.t; l2 : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { l1; l2 } =
    { l1 = Linear.map f l1; l2 = Linear.map f l2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { l1 = Linear.map2 f p.l1 q.l1; l2 = Linear.map2 f p.l2 q.l2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { l1; l2 } =
    Linear.iter f l1;
    Linear.iter f l2

  let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  Printf.printf "Loading MNIST...\n%!";
  match Kaun_next_datasets.mnist () with
  | exception Failure msg ->
      Printf.printf "MNIST unavailable (%s); skipping.\n" msg
  | train_x, train_y, test_x, test_y ->
      (* Flatten [N; 1; 28; 28] images for the MLP. *)
      let flatten x = Nx.reshape [| (Nx.shape x).(0); 28 * 28 |] x in
      let train_x = flatten train_x and test_x = flatten test_x in
      let n_train = (Nx.shape train_x).(0) in
      Printf.printf "  train: %d  test: %d\n%!" n_train (Nx.shape test_x).(0);

      let params =
        {
          Mlp.l1 = Linear.init ~inputs:(28 * 28) ~outputs:128;
          l2 = Linear.init ~inputs:128 ~outputs:10;
        }
      in

      (* Training step: value_and_grad + one AdamW update *)
      let step (params, ostate) (x, y) =
        let loss p = Loss.softmax_cross_entropy_sparse (Mlp.apply p x) y in
        let l, grads = Rune_next.value_and_grad (module Mlp) loss params in
        let params, ostate =
          Vega.adamw_step (module Mlp) ~lr ostate ~params ~grads
        in
        ((params, ostate), Nx.item [] l)
      in

      (* One epoch over shuffled minibatches. *)
      let n_batches = (n_train + batch_size - 1) / batch_size in
      let batches =
        Data.batches2 ~shuffle:true ~batch_size (train_x, train_y)
      in
      let (params, _), _ =
        Seq.fold_left
          (fun (state, i) batch ->
            let state, l = step state batch in
            if i mod 50 = 0 || i = n_batches then
              Printf.printf "  batch %3d/%d  loss %.4f\n%!" i n_batches l;
            (state, i + 1))
          ((params, Vega.adamw_init (module Mlp) params), 1)
          batches
      in

      (* Evaluate on the test set. *)
      let correct, total =
        Data.batches2 ~batch_size:1000 (test_x, test_y)
        |> Seq.fold_left
             (fun (correct, total) (x, y) ->
               let n = (Nx.shape x).(0) in
               let acc = Metric.accuracy (Mlp.apply params x) y in
               (correct +. (acc *. float_of_int n), total + n))
             (0., 0)
      in
      Printf.printf "test accuracy: %.2f%%\n"
        (100. *. correct /. float_of_int total)
