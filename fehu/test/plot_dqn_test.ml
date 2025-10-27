open Nx
open Hugin

let demo_dir = "/Users/mac/Desktop/Outreachy/raven/fehu/demos/"

(* Helper to read positions from file *)
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

(* Helper to read rewards from file *)
let read_rewards path =
  let lines = In_channel.with_open_bin path In_channel.input_all
              |> String.split_on_char '\n'
              |> List.filter (fun s -> String.trim s <> "") in
  Array.of_list (List.map float_of_string lines)

(* Helper to read heatmap from file *)
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
  let x_tensor = Nx.create Nx.float32 [| Array.length early_xs |] early_xs in
  let y_tensor = Nx.create Nx.float32 [| Array.length early_ys |] early_ys in
  let fig = Hugin.scatter ~label:"Agent Path" x_tensor y_tensor in
  Hugin.savefig (demo_dir ^ "gridworld_trajectory_early.png") fig;

  (* Plot late trajectory *)
  let late_xs, late_ys = read_positions (demo_dir ^ "late_positions.txt") in
  let x_tensor = Nx.create Nx.float32 [| Array.length late_xs |] late_xs in
  let y_tensor = Nx.create Nx.float32 [| Array.length late_ys |] late_ys in
  let fig = Hugin.scatter ~label:"Agent Path" x_tensor y_tensor in
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
  Hugin.savefig (demo_dir ^ "gridworld_heatmap.png") fig_heatmap