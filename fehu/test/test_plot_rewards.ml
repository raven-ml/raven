open Alcotest
open Nx
open Hugin

let sample_csv = "test_metrics.csv"
let sample_png = "test_rewards.png"

let write_sample_csv path =
  let oc = open_out path in
  output_string oc "episode_return,episode_length,loss,epsilon,avg_q\n";
  output_string oc "1.0,10,0.5,0.9,0.1\n";
  output_string oc "2.0,10,0.4,0.8,0.2\n";
  output_string oc "3.0,10,0.3,0.7,0.3\n";
  close_out oc

let test_read_csv_rewards () =
  write_sample_csv sample_csv;
  let rewards = Plot_rewards.read_csv_rewards sample_csv in
  Alcotest.(check (list float)) "parsed rewards"
    [1.0; 2.0; 3.0] rewards;
  Sys.remove sample_csv

let test_plot_creates_png () =
  write_sample_csv sample_csv;
  let rewards = Plot_rewards.read_csv_rewards sample_csv in
  let n = List.length rewards in
  let x = create float32 [| n |] (Array.init n float_of_int) in
  let y = create float32 [| n |] (Array.of_list rewards) in
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  let _ = Plotting.plot ~x ~y ~color:Artist.Color.blue ~label:"Episode Reward" ax in
  let _ = Axes.set_xlabel "Episode" ax in
  let _ = Axes.set_ylabel "Reward" ax in
  let _ = Axes.legend true ax in
  savefig sample_png fig;
  Alcotest.(check bool) "png created" true (Sys.file_exists sample_png);
  Sys.remove sample_csv;
  Sys.remove sample_png

let () =
  run "Plot_rewards"
    [
      "csv", [ test_case "read_csv_rewards" `Quick test_read_csv_rewards ];
      "plot", [ test_case "plot_creates_png" `Quick test_plot_creates_png ];
    ]