let () =
  exit
    (Windtrap.run "Tolk.Ops"
       (Nodes.groups @ Values.groups @ Construction.groups @ Patterns.groups))
