open Nx
open Hugin

let () =
  let rewards = Nx.create Nx.float32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let fig = Hugin.plot_y rewards in
  Hugin.savefig "/Users/mac/Desktop/Outreachy/raven/fehu/demos/episode_rewards.png" fig