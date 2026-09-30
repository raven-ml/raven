let () =
  exit
    (Windtrap.run "Tolk_next.Ops"
       (Nodes.groups @ Values.groups @ Construction.groups @ Patterns.groups))
