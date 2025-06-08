(* Test suite for Nx_metal backend *)

module Runner = Test_nx_unit.Make (Nx_metal)

let () =
  Printexc.record_backtrace true;
  Runner.run "Metal" (Nx_metal.create_context ())
