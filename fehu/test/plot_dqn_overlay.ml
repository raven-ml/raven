open Nx
open Hugin

let demo_dir = "/Users/mac/Desktop/Outreachy/raven/fehu/demos/"
let grid_size = 5

(* Find preferred action for each cell *)
let get_policy_from_q q_values =
  Array.map (Array.map (fun qs ->
    let max_idx = ref 0 in
    for i = 1 to Array.length qs - 1 do
      if qs.(i) > qs.(!max_idx) then max_idx := i
    done;
    !max_idx
  )) q_values

(* Plot policy arrows for each cell *)
let plot_policy_arrows policy filename =
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  for x = 0 to grid_size - 1 do
    for y = 0 to grid_size - 1 do
      let action = policy.(x).(y) in
      let (dx, dy, color) =
        match action with
        | 0 -> (0., 0.5, Artist.Color.blue)    (* up *)
        | 1 -> (0., -0.5, Artist.Color.red)    (* down *)
        | 2 -> (-0.5, 0., Artist.Color.green)  (* left *)
        | 3 -> (0.5, 0., Artist.Color.orange)  (* right *)
        | _ -> (0., 0., Artist.Color.gray)
      in
      let x_start = float_of_int x in
      let y_start = float_of_int y in
      let x_end = x_start +. dx in
      let y_end = y_start +. dy in
      Plotting.plot
        ~x:(Nx.create Nx.float32 [|2|] [|x_start; x_end|])
        ~y:(Nx.create Nx.float32 [|2|] [|y_start; y_end|])
        ~color:color
        ax |> ignore;
    done
  done;
  Axes.set_xlabel "X Position" ax |> ignore;
  Axes.set_ylabel "Y Position" ax |> ignore;
  Axes.set_title "Learned Policy (Arrows show preferred action)" ax |> ignore;
  Axes.legend true ax |> ignore;
  Hugin.savefig filename fig
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

let plot_overlay_trajectories early_xs early_ys late_xs late_ys filename =
  let fig =
    let open Hugin in
    let fig = Figure.create () in
    let ax = Figure.add_subplot fig in
    for i = 0 to grid_size do
      let x = float_of_int i in
      Plotting.plot ~x:(create float32 [|2|] [|x; x|])
                    ~y:(create float32 [|2|] [|0.; float_of_int grid_size|])
                    ax |> ignore;
      Plotting.plot ~x:(create float32 [|2|] [|0.; float_of_int grid_size|])
                    ~y:(create float32 [|2|] [|x; x|])
                    ax |> ignore;
    done;
    (* Shift dots to center of squares *)
    let shift arr = Array.map (fun v -> v +. 0.5) arr in
       Plotting.scatter ~x:(create float32 [| Array.length early_xs |] (shift early_xs))
                     ~y:(create float32 [| Array.length early_ys |] (shift early_ys))
                     ~c:Artist.Color.orange
                     ~s:150.
                     ~label:"Early Trajectory"
                     ax |> ignore;
    Plotting.scatter ~x:(create float32 [| Array.length late_xs |] (shift late_xs))
                     ~y:(create float32 [| Array.length late_ys |] (shift late_ys))
                     ~c:Artist.Color.blue
                     ~s:150.
                     ~label:"Late Trajectory"
                     ax |> ignore;
    Plotting.scatter ~x:(create float32 [|1|] [|0.5|])
                     ~y:(create float32 [|1|] [|0.5|])
                     ~c:Artist.Color.green
                     ~s:150.
                     ~label:"Start"
                     ax |> ignore;
    Plotting.scatter ~x:(create float32 [|1|] [|4.5|])
                     ~y:(create float32 [|1|] [|4.5|])
                     ~c:Artist.Color.red
                     ~s:150.
                     ~label:"Goal"
                     ax |> ignore;
    Axes.set_xlabel "X Position" ax |> ignore;
    Axes.set_ylabel "Y Position" ax |> ignore;
    Axes.set_title "Overlay: Early vs. Late Trajectories" ax |> ignore;
    Axes.legend true ax |> ignore;
    fig
  in
  Hugin.savefig filename fig

let read_q_values path =
  let lines = In_channel.with_open_bin path In_channel.input_all
              |> String.split_on_char '\n'
              |> List.filter (fun s -> String.trim s <> "") in
  let rows =
    List.map (fun line ->
      line
      |> String.split_on_char ' '
      |> List.filter (fun s -> s <> "")
      |> List.map float_of_string
      |> Array.of_list
    ) lines
    |> Array.of_list
  in
  (* Reshape to grid_size x grid_size x n_actions *)
  Array.init grid_size (fun x ->
    Array.init grid_size (fun y ->
      rows.(x * grid_size + y)
    )
  )
;;

let () =
  let early_xs, early_ys = read_positions (demo_dir ^ "early_positions.txt") in
  let late_xs, late_ys = read_positions (demo_dir ^ "late_positions.txt") in
  plot_overlay_trajectories early_xs early_ys late_xs late_ys (demo_dir ^ "gridworld_trajectory_overlay.png");
