

let read_csv_column path col_idx =
  let ic = open_in path in
  let _header = input_line ic in
  let rec loop acc =
    try
      let line = input_line ic in
      let value = float_of_string (List.nth (String.split_on_char ',' line) col_idx) in
      loop (value :: acc)
    with End_of_file -> List.rev acc
  in
  let values = loop [] in
  close_in ic;
  values

let plot_metric ~col_idx ~label ~color ~output =
  let values = read_csv_column "fehu/demos/metrics.csv" col_idx in
  let n = List.length values in
  if n = 0 then (
    prerr_endline ("No values found for " ^ label ^ " in metrics.csv.");
    exit 1
  );
  let x = Nx.create Nx.float32 [| n |] (Array.init n float_of_int) in
  let y = Nx.create Nx.float32 [| n |] (Array.of_list values) in
  let fig = Hugin.Figure.create () in
  let ax = Hugin.Figure.add_subplot fig in
  let _ = Hugin.Plotting.plot ~x ~y ~color ~label ax in
  let _ = Hugin.Axes.set_xlabel "Episode" ax in
  let _ = Hugin.Axes.set_ylabel label ax in
  let _ = Hugin.Axes.legend true ax in
  Hugin.savefig output fig

let () =
  plot_metric ~col_idx:0 ~label:"Episode Reward" ~color:Hugin.Artist.Color.blue
    ~output:"fehu/demos/episode_rewards.png";
  plot_metric ~col_idx:1 ~label:"Episode Length" ~color:Hugin.Artist.Color.green
    ~output:"fehu/demos/episode_length.png";
  plot_metric ~col_idx:2 ~label:"Loss" ~color:Hugin.Artist.Color.red
    ~output:"fehu/demos/episode_loss.png";
  plot_metric ~col_idx:3 ~label:"Epsilon" ~color:Hugin.Artist.Color.orange
    ~output:"fehu/demos/epsilon.png";
  plot_metric ~col_idx:4 ~label:"Avg Q" ~color:Hugin.Artist.Color.magenta
    ~output:"fehu/demos/avg_q.png"