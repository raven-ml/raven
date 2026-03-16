(** End-to-end MNIST training with experiment tracking.

    Trains a CNN on MNIST using kaun, logging metrics, hyperparameters, and a
    model checkpoint via munin. Shows how munin integrates with a real training
    loop without adding a dependency from kaun to munin. *)

open Kaun

let batch_size = 64
let epochs = 3
let lr = 0.001

let model =
  Layer.sequential
    [
      Layer.conv2d ~in_channels:1 ~out_channels:16 ();
      Layer.relu ();
      Layer.max_pool2d ~kernel_size:(2, 2) ();
      Layer.conv2d ~in_channels:16 ~out_channels:32 ();
      Layer.relu ();
      Layer.max_pool2d ~kernel_size:(2, 2) ();
      Layer.flatten ();
      Layer.linear ~in_features:(32 * 7 * 7) ~out_features:128 ();
      Layer.relu ();
      Layer.linear ~in_features:128 ~out_features:10 ();
    ]

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let dtype = Nx.float32 in

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
  let (x_train, y_train), (x_test, y_test) = Kaun_datasets.mnist () in
  let n_train = (Nx.shape x_train).(0) in
  Printf.printf "  train: %d  test: %d\n%!" n_train (Nx.shape x_test).(0);

  let test_batches = Data.prepare ~batch_size (x_test, y_test) in
  let trainer =
    Train.make ~model ~optimizer:(Vega.adam (Vega.Schedule.constant lr))
  in
  let st = ref (Train.init trainer ~dtype) in
  let global_step = ref 0 in
  let last_acc = ref 0. in

  for epoch = 1 to epochs do
    let train_data =
      Data.prepare ~shuffle:true ~batch_size (x_train, y_train)
      |> Data.map (fun (x, y) ->
          (x, fun logits -> Loss.cross_entropy_sparse logits y))
    in
    let num_batches = n_train / batch_size in
    let tracker = Metric.tracker () in

    st :=
      Train.fit trainer !st
        ~report:(fun ~step ~loss _st ->
          let s = !global_step + step in
          Metric.observe tracker "loss" loss;
          Munin.Session.log_metrics session ~step:s
            [ ("train/loss", loss); ("epoch", Float.of_int epoch) ];
          Printf.printf "\r  batch %d/%d  loss: %.4f%!" step num_batches loss)
        train_data;
    global_step := !global_step + num_batches;
    Printf.printf "\n%!";

    (* Evaluate. *)
    Data.reset test_batches;
    let test_acc =
      Metric.eval
        (fun (x, y) ->
          let logits = Train.predict trainer !st x in
          Metric.accuracy logits y)
        test_batches
    in
    last_acc := test_acc;

    Munin.Session.log_metrics session ~step:!global_step
      [
        ("train/loss_avg", Metric.mean tracker "loss");
        ("val/accuracy", test_acc);
      ];

    Printf.printf "epoch %d  loss: %.4f  val_acc: %.2f%%\n%!" epoch
      (Metric.mean tracker "loss")
      (test_acc *. 100.)
  done;

  (* Save model checkpoint as a versioned artifact. *)
  let checkpoint_path =
    Filename.concat
      (Munin.Run.dir (Munin.Session.run session))
      "model.safetensors"
  in
  Checkpoint.save checkpoint_path (Layer.params (Train.vars !st));
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
