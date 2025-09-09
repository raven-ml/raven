(* Run Pipeline: Complete workshop pipeline *)
let () =
  print_endline "\n==== Workshop Pipeline: Complete RL Methods Comparison ====\n";
  Workshop.Slide_pip.run_complete_workshop ();
  print_endline "\n==== Method Comparison ====\n";
  Workshop.Slide_pip.compare_methods ();
  print_endline "\n==== Pipeline Complete ====\n"