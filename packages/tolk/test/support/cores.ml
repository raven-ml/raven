open Tolk

let host =
  let target =
    { Helpers.Target.device = "CPU"; renderer = "CLANG";
      arch = "x86_64,x86-64"; interface = ""; indices = "" }
  in
  [ ("CPU", Heuristic.cores (Cstyle.without_compiler target)) ]
