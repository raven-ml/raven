(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Jitted MNIST training on the GPU with experiment tracking.

    Trains the same CNN as [x-kaun-mnist], but the whole training step —
    forward, backward, and SGD update — compiles into one program with
    [Rune.jit] and runs on Metal (the default; pass [--device CPU] or
    [--device CUDA]). Parameters stay resident on the device between steps: only
    the scalar loss is read back each step. Metrics stream to munin; watch the
    run live with:

    {v
    dune exec packages/munin/examples/x-kaun-mnist-jit/main.exe
    munin watch   # in another terminal
    v}

    The first step traces and compiles the program (a few seconds, cached across
    runs); every later step replays the compiled kernels. *)

open Kaun

let device = ref "METAL"
let epochs = ref 5
let batch_size = ref 128
let lr = ref 0.05
let eval_every = ref 200

let speclist =
  [
    ("--device", Arg.Set_string device, "METAL|CPU|CUDA (default METAL)");
    ("--epochs", Arg.Set_int epochs, "training epochs (default 5)");
    ("--batch-size", Arg.Set_int batch_size, "batch size (default 128)");
    ("--lr", Arg.Set_float lr, "learning rate (default 0.05)");
    ( "--eval-every",
      Arg.Set_int eval_every,
      "steps between test evaluations (default 200)" );
  ]

(* Conv(1 -> 16, 3x3, same) -> ReLU -> MaxPool(2x2) -> Conv(16 -> 32, 3x3, same)
   -> ReLU -> MaxPool(2x2) -> Flatten -> Linear(32*7*7 -> 128) -> ReLU ->
   Linear(128 -> 10), as a plain record of layers with a hand-written [walk],
   the Nx.Ptree.S contract that also names checkpoint entries. *)

module Cnn = struct
  type 'a t = {
    c1 : 'a Conv.t;
    c2 : 'a Conv.t;
    l1 : 'a Linear.t;
    l2 : 'a Linear.t;
  }

  let walk c { c1; c2; l1; l2 } =
    let open Nx.Ptree.Walk in
    let c1 = field c "c1" Conv.walk c1 in
    let c2 = field c "c2" Conv.walk c2 in
    let l1 = field c "l1" Linear.walk l1 in
    let l2 = field c "l2" Linear.walk l2 in
    { c1; c2; l1; l2 }

  let apply p x =
    let x = Fn.relu (Conv.apply ~padding:`Same p.c1 x) in
    let x = Pool.max_pool2d ~kernel_size:(2, 2) x in
    let x = Fn.relu (Conv.apply ~padding:`Same p.c2 x) in
    let x = Pool.max_pool2d ~kernel_size:(2, 2) x in
    let x = Nx.reshape [| (Nx.shape x).(0); 32 * 7 * 7 |] x in
    Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end

let cnn = Nx.Ptree.instantiate (module Cnn)

let () =
  Arg.parse speclist
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "mnist-jit [--device METAL|CPU|CUDA] [--epochs N] [--batch-size N]";
  Nx.Rng.with_key (Nx.Rng.key 42) @@ fun () ->
  (* Model parameters and a one-time optimizer state (momentum is 0, so the zero
     velocity is never read and the state needs no threading through the jitted
     step). *)
  let params =
    ref
      {
        Cnn.c1 = Conv.init ~in_channels:1 ~out_channels:16 ~kernel_size:(3, 3);
        c2 = Conv.init ~in_channels:16 ~out_channels:32 ~kernel_size:(3, 3);
        l1 = Linear.init ~inputs:(32 * 7 * 7) ~outputs:128;
        l2 = Linear.init ~inputs:128 ~outputs:10;
      }
  in
  let state = Vega.sgd_init cnn !params in
  let n_params = Nx.Ptree.fold cnn (fun _ t n -> n + Nx.numel t) !params 0 in

  (* Start a tracked run. *)
  let session =
    Munin.Session.start ~experiment:"mnist"
      ~name:("cnn-jit-" ^ String.lowercase_ascii !device)
      ~tags:[ "jit"; String.lowercase_ascii !device ]
      ~params:
        [
          ("device", `String !device);
          ("lr", `Float !lr);
          ("batch_size", `Int !batch_size);
          ("epochs", `Int !epochs);
          ("optimizer", `String "sgd");
          ("model", `String "cnn");
          ("n_params", `Int n_params);
        ]
      ()
  in
  let train_loss = Munin.Session.metric session ~goal:`Minimize "train/loss" in
  let train_loss_avg = Munin.Session.metric session "train/loss_avg" in
  let val_accuracy =
    Munin.Session.metric session ~goal:`Maximize "val/accuracy"
  in
  let images_per_sec =
    Munin.Session.metric session ~summary:`Mean "perf/images_per_sec"
  in
  let step_ms = Munin.Session.metric session "perf/step_ms" in
  let epoch_metric = Munin.Session.metric session "epoch" in
  let sysmon = Munin_sys.start session in

  Printf.printf "run: %s  device: %s  params: %d\n%!" (Munin.Session.id session)
    !device n_params;

  (* Load data. *)
  Printf.printf "Loading MNIST...\n%!";
  let x_train, y_train, x_test, y_test = Kaun_datasets.mnist () in
  let n_train = (Nx.shape x_train).(0) in
  Printf.printf "  train: %d  test: %d\n%!" n_train (Nx.shape x_test).(0);

  (* The whole training step (forward, backward, SGD update) compiles into one
     program, which reads the batch and consumes the parameters: it writes the
     updated parameters over their storage and returns the loss beside them, so
     training never round-trips them through the host. Values that change
     between calls are arguments, never captures. *)
  let train_step x y params =
    let loss_fn p = Loss.softmax_cross_entropy_sparse (Cnn.apply p x) y in
    let loss, grads = Rune.value_and_grad cnn loss_fn params in
    (loss, fst (Vega.sgd_step cnn ~lr:(Vega.lr !lr) state ~params ~grads))
  in
  let devices = [ Rune.device !device ] in
  let step =
    Rune.jit ~devices
      Nx.Ptree.(tensor @-> tensor @-> consumes cnn @@ returns (pair tensor cnn))
      train_step
  in
  let forward =
    Rune.jit ~devices Nx.Ptree.(cnn @-> tensor @-> returns tensor) Cnn.apply
  in
  let evaluate params =
    let correct, total =
      Data.batches2 ~batch_size:500 (x_test, y_test)
      |> Seq.fold_left
           (fun (correct, total) (x, y) ->
             let acc = Metric.accuracy (forward params x) y in
             let n = (Nx.shape x).(0) in
             (correct +. (acc *. float_of_int n), total + n))
           (0., 0)
    in
    correct /. float_of_int total
  in

  let global_step = ref 0 in
  let last_acc = ref 0. in

  for epoch = 1 to !epochs do
    let num_batches = n_train / !batch_size in
    let loss_sum = ref 0. in
    let loss_count = ref 0 in

    Data.batches2 ~shuffle:true ~drop_last:true ~batch_size:!batch_size
      (x_train, y_train)
    |> Seq.iter (fun (x, y) ->
        incr global_step;
        let s = !global_step in
        let t0 = Unix.gettimeofday () in
        let loss, next = step x y !params in
        params := next;
        (* Reading the loss is the step's only device-to-host transfer and its
           synchronization point, so [dt] covers the full step. *)
        let loss = Nx.item [] loss in
        let dt = Unix.gettimeofday () -. t0 in
        loss_sum := !loss_sum +. loss;
        incr loss_count;
        if s = 1 then begin
          Printf.printf "  traced and compiled in %.1fs\n%!" dt;
          Munin.Session.log_metrics session ~step:s
            [ (train_loss, loss); (epoch_metric, Float.of_int epoch) ]
        end
        else
          Munin.Session.log_metrics session ~step:s
            [
              (train_loss, loss);
              (epoch_metric, Float.of_int epoch);
              (step_ms, dt *. 1000.);
              (images_per_sec, float_of_int !batch_size /. dt);
            ];
        if s mod !eval_every = 0 then begin
          let acc = evaluate !params in
          last_acc := acc;
          Munin.Metric.log val_accuracy ~step:s acc
        end;
        Printf.printf "\r  step %d/%d  loss: %.4f  val_acc: %.2f%%%!"
          (((s - 1) mod num_batches) + 1)
          num_batches loss (!last_acc *. 100.));

    let loss_avg = !loss_sum /. float_of_int !loss_count in
    Munin.Metric.log train_loss_avg ~step:!global_step loss_avg;
    Printf.printf "\nepoch %d  loss: %.4f  val_acc: %.2f%%\n%!" epoch loss_avg
      (!last_acc *. 100.)
  done;

  let acc = evaluate !params in
  last_acc := acc;
  Munin.Metric.log val_accuracy ~step:!global_step acc;

  (* Save the trained model as a versioned artifact. *)
  let checkpoint_path =
    Filename.concat (Munin.Session.dir session) "model.safetensors"
  in
  Checkpoint.save checkpoint_path (Checkpoint.of_value cnn !params);
  ignore
    (Munin.Session.log_artifact session ~name:"mnist-cnn-jit" ~kind:`Checkpoint
       ~path:checkpoint_path
       ~metadata:[ ("format", `String "safetensors") ]
       ~aliases:[ "latest" ] ());

  Munin_sys.stop sysmon;
  Munin.Session.set_notes session
    (Some
       (Printf.sprintf "Jitted on %s. Final val accuracy: %.2f%%" !device
          (!last_acc *. 100.)));
  Munin.Session.finish session;
  Printf.printf "\nDone. Run: %s\n" (Munin.Session.id session)
