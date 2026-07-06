(** End-to-end MNIST training with experiment tracking.

    Trains a CNN on MNIST using kaun, logging metrics, hyperparameters, and a
    model checkpoint via munin. Shows how munin integrates with a real training
    loop without adding a dependency from kaun to munin. *)

open Kaun

let batch_size = 64
let epochs = 3
let lr = 0.001

(* Conv(1 -> 16, 3x3, same) -> ReLU -> MaxPool(2x2) -> Conv(16 -> 32, 3x3, same)
   -> ReLU -> MaxPool(2x2) -> Flatten -> Linear(32*7*7 -> 128) -> ReLU ->
   Linear(128 -> 10), as a plain record of layers with hand-written traversals
   (the Nx.Ptree.S contract plus checkpoint names). *)

module Cnn = struct
  type t = { c1 : Conv.t; c2 : Conv.t; l1 : Linear.t; l2 : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { c1; c2; l1; l2 } =
    {
      c1 = Conv.map f c1;
      c2 = Conv.map f c2;
      l1 = Linear.map f l1;
      l2 = Linear.map f l2;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    {
      c1 = Conv.map2 f p.c1 q.c1;
      c2 = Conv.map2 f p.c2 q.c2;
      l1 = Linear.map2 f p.l1 q.l1;
      l2 = Linear.map2 f p.l2 q.l2;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { c1; c2; l1; l2 } =
    Conv.iter f c1;
    Conv.iter f c2;
    Linear.iter f l1;
    Linear.iter f l2

  let names { c1; c2; l1; l2 } =
    List.concat
      [
        List.map (( ^ ) "c1.") (Conv.names c1);
        List.map (( ^ ) "c2.") (Conv.names c2);
        List.map (( ^ ) "l1.") (Linear.names l1);
        List.map (( ^ ) "l2.") (Linear.names l2);
      ]

  let apply p x =
    let x = Fn.relu (Conv.apply ~padding:`Same p.c1 x) in
    let x = Pool.max_pool2d ~kernel_size:(2, 2) x in
    let x = Fn.relu (Conv.apply ~padding:`Same p.c2 x) in
    let x = Pool.max_pool2d ~kernel_size:(2, 2) x in
    let x = Nx.reshape [| (Nx.shape x).(0); 32 * 7 * 7 |] x in
    Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  (* Start a tracked run. *)
  let session =
    Munin.Session.start ~experiment:"mnist" ~name:"cnn-adam"
      ~tags:[ "baseline" ]
      ~params:
        [
          ("lr", `Float lr);
          ("batch_size", `Int batch_size);
          ("epochs", `Int epochs);
          ("optimizer", `String "adam");
        ]
      ()
  in
  Munin.Session.define_metric session "train/loss" ~summary:`Min ~goal:`Minimize
    ();
  Munin.Session.define_metric session "val/accuracy" ~summary:`Max
    ~goal:`Maximize ();
  let sysmon = Munin_sys.start session () in

  Printf.printf "run: %s\n%!" (Munin.Run.id (Munin.Session.run session));

  (* Load data. *)
  Printf.printf "Loading MNIST...\n%!";
  let x_train, y_train, x_test, y_test = Kaun_datasets.mnist () in
  let n_train = (Nx.shape x_train).(0) in
  Printf.printf "  train: %d  test: %d\n%!" n_train (Nx.shape x_test).(0);

  (* Model parameters and optimizer state. *)
  let params =
    ref
      {
        Cnn.c1 = Conv.init ~in_channels:1 ~out_channels:16 ~kernel_size:(3, 3);
        c2 = Conv.init ~in_channels:16 ~out_channels:32 ~kernel_size:(3, 3);
        l1 = Linear.init ~inputs:(32 * 7 * 7) ~outputs:128;
        l2 = Linear.init ~inputs:128 ~outputs:10;
      }
  in
  let ostate = ref (Vega.adam_init (module Cnn) !params) in

  (* Training step: value_and_grad + one Adam update. *)
  let train_step (x, y) =
    let loss_fn p = Loss.softmax_cross_entropy_sparse (Cnn.apply p x) y in
    let l, grads = Rune.value_and_grad (module Cnn) loss_fn !params in
    let params', ostate' =
      Vega.adam_step (module Cnn) ~lr !ostate ~params:!params ~grads
    in
    params := params';
    ostate := ostate';
    Nx.item [] l
  in

  let global_step = ref 0 in
  let last_acc = ref 0. in

  for epoch = 1 to epochs do
    let num_batches = n_train / batch_size in
    let loss_sum = ref 0. in
    let loss_count = ref 0 in

    Data.batches2 ~shuffle:true ~batch_size (x_train, y_train)
    |> Seq.fold_left
         (fun step batch ->
           let loss = train_step batch in
           let s = !global_step + step in
           loss_sum := !loss_sum +. loss;
           incr loss_count;
           Munin.Session.log_metrics session ~step:s
             [ ("train/loss", loss); ("epoch", Float.of_int epoch) ];
           Printf.printf "\r  batch %d/%d  loss: %.4f%!" step num_batches loss;
           step + 1)
         1
    |> ignore;
    global_step := !global_step + num_batches;
    Printf.printf "\n%!";

    (* Evaluate. *)
    let correct, total =
      Data.batches2 ~batch_size (x_test, y_test)
      |> Seq.fold_left
           (fun (correct, total) (x, y) ->
             let n = (Nx.shape x).(0) in
             let acc = Metric.accuracy (Cnn.apply !params x) y in
             (correct +. (acc *. float_of_int n), total + n))
           (0., 0)
    in
    let test_acc = correct /. float_of_int total in
    last_acc := test_acc;

    let loss_avg = !loss_sum /. float_of_int !loss_count in
    Munin.Session.log_metrics session ~step:!global_step
      [ ("train/loss_avg", loss_avg); ("val/accuracy", test_acc) ];

    Printf.printf "epoch %d  loss: %.4f  val_acc: %.2f%%\n%!" epoch loss_avg
      (test_acc *. 100.)
  done;

  (* Save model checkpoint as a versioned artifact. *)
  let checkpoint_path =
    Filename.concat
      (Munin.Run.dir (Munin.Session.run session))
      "model.safetensors"
  in
  Checkpoint.save checkpoint_path (Checkpoint.of_params (module Cnn) !params);
  ignore
    (Munin.Session.log_artifact session ~name:"mnist-cnn" ~kind:`checkpoint
       ~path:checkpoint_path
       ~metadata:[ ("format", `String "safetensors") ]
       ~aliases:[ "latest" ] ());

  Munin_sys.stop sysmon;
  Munin.Session.set_notes session
    (Some (Printf.sprintf "Final val accuracy: %.2f%%" (!last_acc *. 100.)));
  Munin.Session.finish session ();
  Printf.printf "\nDone. Run: %s\n" (Munin.Run.id (Munin.Session.run session))
