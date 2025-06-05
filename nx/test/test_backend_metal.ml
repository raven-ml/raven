(* Test suite for Nx_native backend *)

module Runner = Test_nx_unit.Make (Nx_metal)

let () =
  Printexc.record_backtrace true;
  Runner.run "Native" (Nx_metal.create_context ())
