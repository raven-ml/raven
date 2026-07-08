(** Jitted MNIST training on the GPU with experiment tracking.

    Trains the same CNN as [x-kaun-mnist], but the whole training step —
    forward, backward, and SGD update — compiles into one program with
    [Rune.jit2] and runs on Metal (the default; pass [--device CPU] or
    [--device CUDA]). Parameters stay resident on the device between steps:
    only the scalar loss is read back each step. Metrics stream to munin;
    watch the run live with:

    {v
    dune exec packages/munin/examples/x-kaun-mnist-jit/main.exe
    munin watch   # in another terminal
    v}

    The first step traces and compiles the program (a few seconds, cached
    across runs); every later step replays the compiled kernels. *)

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

(* Conv(1 -> 16, 3x3, same) -> ReLU -> MaxPool(2x2) -> Conv(16 -> 32, 3x3,
   same) -> ReLU -> MaxPool(2x2) -> Flatten -> Linear(32*7*7 -> 128) -> ReLU ->
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

(* The jitted step's input: the batch joins the parameters as leaves — values
   that change between calls must be inputs, never captures. *)
module Step_in = struct
  type t = {
    params : Cnn.t;
    x : (float, Nx.float32_elt) Nx.t;
    y : (int32, Nx.int32_elt) Nx.t;
  }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    { params = Cnn.map f t.params; x = f t.x; y = f t.y }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { params = Cnn.map2 f a.params b.params; x = f a.x b.x; y = f a.y b.y }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    Cnn.iter f t.params;
    f t.x;
    f t.y
end

module Step_out = struct
  type t = { params : Cnn.t; loss : (float, Nx.float32_elt) Nx.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    { params = Cnn.map f t.params; loss = f t.loss }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { params = Cnn.map2 f a.params b.params; loss = f a.loss b.loss }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    Cnn.iter f t.params;
    f t.loss
end

module Eval_in = struct
  type t = { params : Cnn.t; x : (float, Nx.float32_elt) Nx.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    { params = Cnn.map f t.params; x = f t.x }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { params = Cnn.map2 f a.params b.params; x = f a.x b.x }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    Cnn.iter f t.params;
    f t.x
end

let () =
  Arg.parse speclist
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "mnist-jit [--device METAL|CPU|CUDA] [--epochs N] [--batch-size N]";
  Nx.Rng.run ~seed:42 @@ fun () ->
  (* Model parameters and a one-time optimizer state (momentum is 0, so the
     zero velocity is never read and the state needs no threading through the
     jitted step). *)
  let params =
    ref
      {
        Cnn.c1 = Conv.init ~in_channels:1 ~out_channels:16 ~kernel_size:(3, 3);
        c2 = Conv.init ~in_channels:16 ~out_channels:32 ~kernel_size:(3, 3);
        l1 = Linear.init ~inputs:(32 * 7 * 7) ~outputs:128;
        l2 = Linear.init ~inputs:128 ~outputs:10;
      }
  in
  let state = Vega.sgd_init (module Cnn) !params in
  let n_params =
    let n = ref 0 in
    let count : 'a 'b. ('a, 'b) Nx.t -> unit = fun t -> n := !n + Nx.numel t in
    Cnn.iter count !params;
    !n
  in

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
  Munin.Session.define_metric session "train/loss" ~summary:`Min ~goal:`Minimize
    ();
  Munin.Session.define_metric session "val/accuracy" ~summary:`Max
    ~goal:`Maximize ();
  Munin.Session.define_metric session "perf/images_per_sec" ~summary:`Mean ();
  let sysmon = Munin_sys.start session () in

  Printf.printf "run: %s  device: %s  params: %d\n%!"
    (Munin.Run.id (Munin.Session.run session))
    !device n_params;

  (* Load data. *)
  Printf.printf "Loading MNIST...\n%!";
  let x_train, y_train, x_test, y_test = Kaun_datasets.mnist () in
  let n_train = (Nx.shape x_train).(0) in
  Printf.printf "  train: %d  test: %d\n%!" n_train (Nx.shape x_test).(0);

  (* The whole training step — forward, backward, SGD update — compiles into
     one program. Parameters flow out and back in as unread device-resident
     tensors, so training never round-trips them through the host. *)
  let train_step { Step_in.params; x; y } =
    let loss_fn p = Loss.softmax_cross_entropy_sparse (Cnn.apply p x) y in
    let loss, grads = Rune.value_and_grad (module Cnn) loss_fn params in
    let params, _ = Vega.sgd_step (module Cnn) ~lr:!lr state ~params ~grads in
    { Step_out.params; loss }
  in
  let step =
    Rune.jit2 ~device:!device (module Step_in) (module Step_out) train_step
  in
  let forward =
    Rune.jit ~device:!device
      (module Eval_in)
      (fun { Eval_in.params; x } -> Cnn.apply params x)
  in

  let evaluate params =
    let correct, total =
      Data.batches2 ~batch_size:500 (x_test, y_test)
      |> Seq.fold_left
           (fun (correct, total) (x, y) ->
             let acc = Metric.accuracy (forward { Eval_in.params; x }) y in
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
           let out = step { Step_in.params = !params; x; y } in
           params := out.Step_out.params;
           (* Reading the loss is the step's only device-to-host transfer and
              its synchronization point, so [dt] covers the full step. *)
           let loss = Nx.item [] out.Step_out.loss in
           let dt = Unix.gettimeofday () -. t0 in
           loss_sum := !loss_sum +. loss;
           incr loss_count;
           if s = 1 then begin
             Printf.printf "  traced and compiled in %.1fs\n%!" dt;
             Munin.Session.log_metrics session ~step:s
               [ ("train/loss", loss); ("epoch", Float.of_int epoch) ]
           end
           else
             Munin.Session.log_metrics session ~step:s
               [
                 ("train/loss", loss);
                 ("epoch", Float.of_int epoch);
                 ("perf/step_ms", dt *. 1000.);
                 ("perf/images_per_sec", float_of_int !batch_size /. dt);
               ];
           if s mod !eval_every = 0 then begin
             let acc = evaluate !params in
             last_acc := acc;
             Munin.Session.log_metrics session ~step:s
               [ ("val/accuracy", acc) ]
           end;
           Printf.printf "\r  step %d/%d  loss: %.4f  val_acc: %.2f%%%!"
             (((s - 1) mod num_batches) + 1)
             num_batches loss (!last_acc *. 100.));

    let loss_avg = !loss_sum /. float_of_int !loss_count in
    Munin.Session.log_metrics session ~step:!global_step
      [ ("train/loss_avg", loss_avg) ];
    Printf.printf "\nepoch %d  loss: %.4f  val_acc: %.2f%%\n%!" epoch loss_avg
      (!last_acc *. 100.)
  done;

  let acc = evaluate !params in
  last_acc := acc;
  Munin.Session.log_metrics session ~step:!global_step
    [ ("val/accuracy", acc) ];

  (* Save the trained model as a versioned artifact. *)
  let checkpoint_path =
    Filename.concat
      (Munin.Run.dir (Munin.Session.run session))
      "model.safetensors"
  in
  Checkpoint.save checkpoint_path (Checkpoint.of_params (module Cnn) !params);
  ignore
    (Munin.Session.log_artifact session ~name:"mnist-cnn-jit" ~kind:`checkpoint
       ~path:checkpoint_path
       ~metadata:[ ("format", `String "safetensors") ]
       ~aliases:[ "latest" ] ());

  Munin_sys.stop sysmon;
  Munin.Session.set_notes session
    (Some
       (Printf.sprintf "Jitted on %s. Final val accuracy: %.2f%%" !device
          (!last_acc *. 100.)));
  Munin.Session.finish session ();
  Printf.printf "\nDone. Run: %s\n" (Munin.Run.id (Munin.Session.run session))
