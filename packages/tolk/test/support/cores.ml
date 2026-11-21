open Tolk

let host =
  let target =
    { Helpers.Target.device = "CPU"; renderer = "CLANG";
      arch = "x86_64,x86-64"; interface = ""; indices = "" }
  in
  [ ("CPU", Heuristic.cores (Cstyle.without_compiler target)) ]

(* The host's cores, on a device taken to run its kernels' points at once, as
   a GPU's threads. *)
let at_once =
  List.map
    (fun (d, (c : Heuristic.cores)) -> (d, { c with at_once = true }))
    host
