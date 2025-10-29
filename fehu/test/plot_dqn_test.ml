open Nx
open Hugin

let demo_dir = "/Users/mac/Desktop/Outreachy/raven/fehu/demos/"
let grid_size = 5

let plot_trajectory_frame xs ys step filename =
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  (* Draw grid lines *)
  for i = 0 to grid_size do
    let x = float_of_int i in
    Plotting.plot ~x:(Nx.create Nx.float32 [|2|] [|x; x|])
                  ~y:(Nx.create Nx.float32 [|2|] [|0.; float_of_int grid_size|])
                  ~color:Artist.Color.gray
                  ax |> ignore;
    Plotting.plot ~x:(Nx.create Nx.float32 [|2|] [|0.; float_of_int grid_size|])
                  ~y:(Nx.create Nx.float32 [|2|] [|x; x|])
                  ~color:Artist.Color.gray
                  ax |> ignore;
  done;
  (* Centralize and plot trajectory up to current step *)
  let centralized_xs = Array.map (fun x -> x +. 0.5) (Array.sub xs 0 (step + 1)) in
  let centralized_ys = Array.map (fun y -> y +. 0.5) (Array.sub ys 0 (step + 1)) in
  let x_tensor = Nx.create Nx.float32 [| step + 1 |] centralized_xs in
  let y_tensor = Nx.create Nx.float32 [| step + 1 |] centralized_ys in
  Plotting.scatter ~x:x_tensor ~y:y_tensor ~c:Artist.Color.blue ~label:"Agent Path" ~s:150. ax |> ignore;
  (* Mark current position centralized *)
  Plotting.scatter ~x:(Nx.create Nx.float32 [|1|] [|xs.(step) +. 0.5|])
                   ~y:(Nx.create Nx.float32 [|1|] [|ys.(step) +. 0.5|])
                   ~c:Artist.Color.orange
                   ~label:"Current"
                   ~s:150.
                   ax |> ignore;
  (* Mark start and goal centralized *)
  Plotting.scatter ~x:(Nx.create Nx.float32 [|1|] [|0.5|])
                   ~y:(Nx.create Nx.float32 [|1|] [|0.5|])
                   ~c:Artist.Color.green
                   ~label:"Start"
                   ~s:150.
                   ax |> ignore;
  Plotting.scatter ~x:(Nx.create Nx.float32 [|1|] [|4.5|])
                   ~y:(Nx.create Nx.float32 [|1|] [|4.5|])
                   ~c:Artist.Color.red
                   ~label:"Goal"
                   ~s:150.
                   ax |> ignore;
  Axes.set_xlabel "X Position" ax |> ignore;
  Axes.set_ylabel "Y Position" ax |> ignore;
  Axes.set_title "Agent Trajectory Animation" ax |> ignore;
  Axes.legend true ax |> ignore;
  Hugin.savefig filename fig
;;

let animate_episode xs ys =
  let n = Array.length xs in
  for step = 0 to n - 1 do
    let filename = Printf.sprintf "%s/frame_%03d.png" demo_dir step in
    plot_trajectory_frame xs ys step filename
  done
;;

let read_positions path =
  let lines = In_channel.with_open_bin path In_channel.input_all
              |> String.split_on_char '\n'
              |> List.filter (fun s -> String.trim s <> "") in
  let xs, ys =
    List.fold_right (fun line (xs, ys) ->
      match String.split_on_char ' ' (String.trim line) with
      | x::y::_ -> (float_of_string x :: xs, float_of_string y :: ys)
      | _ -> (xs, ys)
    ) lines ([], [])
  in
  (Array.of_list xs, Array.of_list ys)

let read_rewards path =
  let lines = In_channel.with_open_bin path In_channel.input_all
              |> String.split_on_char '\n'
              |> List.filter (fun s -> String.trim s <> "") in
  Array.of_list (List.map float_of_string lines)

let read_heatmap path =
  let lines = In_channel.with_open_bin path In_channel.input_all
              |> String.split_on_char '\n'
              |> List.filter (fun s -> String.trim s <> "") in
  Array.of_list (
    List.map (fun line ->
      line
      |> String.split_on_char ' '
      |> List.filter (fun s -> s <> "")
      |> List.map float_of_string
      |> Array.of_list
    ) lines
  )

let () =
  (* Plot early trajectory *)
  let early_xs, early_ys = read_positions (demo_dir ^ "early_positions.txt") in
  let centralized_early_xs = Array.map (fun x -> x +. 0.5) early_xs in
  let centralized_early_ys = Array.map (fun y -> y +. 0.5) early_ys in
  let x_tensor = Nx.create Nx.float32 [| Array.length centralized_early_xs |] centralized_early_xs in
  let y_tensor = Nx.create Nx.float32 [| Array.length centralized_early_ys |] centralized_early_ys in
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  Plotting.scatter ~x:x_tensor ~y:y_tensor ~c:Artist.Color.blue ~label:"Agent Path" ~s:150. ax |> ignore;
  Hugin.savefig (demo_dir ^ "gridworld_trajectory_early.png") fig;

  (* Plot late trajectory *)
  let late_xs, late_ys = read_positions (demo_dir ^ "late_positions.txt") in
  let centralized_late_xs = Array.map (fun x -> x +. 0.5) late_xs in
  let centralized_late_ys = Array.map (fun y -> y +. 0.5) late_ys in
  let x_tensor = Nx.create Nx.float32 [| Array.length centralized_late_xs |] centralized_late_xs in
  let y_tensor = Nx.create Nx.float32 [| Array.length centralized_late_ys |] centralized_late_ys in
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  Plotting.scatter ~x:x_tensor ~y:y_tensor ~c:Artist.Color.orange ~label:"Agent Path" ~s:150. ax |> ignore;
  Hugin.savefig (demo_dir ^ "gridworld_trajectory_late.png") fig;

  (* Plot episode rewards *)
  let rewards = read_rewards (demo_dir ^ "episode_rewards.txt") in
  let rewards_tensor = Nx.create Nx.float32 [| Array.length rewards |] rewards in
  let fig_rewards = Hugin.plot_y rewards_tensor in
  Hugin.savefig (demo_dir ^ "episode_rewards.png") fig_rewards;

  (* Plot heatmap *)
  let heatmap = read_heatmap (demo_dir ^ "visit_counts.txt") in
  let flat = Array.concat (Array.to_list heatmap) in
  let shape = [| Array.length heatmap; Array.length heatmap.(0) |] in
  let heatmap_tensor = Nx.create Nx.float32 shape flat in
  let fig_heatmap = Hugin.imshow heatmap_tensor in
  Hugin.savefig (demo_dir ^ "gridworld_heatmap.png") fig_heatmap;

  (* Generate animation frames for early episode *)
  animate_episode early_xs early_ys