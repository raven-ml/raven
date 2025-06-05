(* Test suite for Nx_native backend *)

module Runner = Test_nx_unit.Make (Nx_native)

let () =
  Printexc.record_backtrace true;
  Runner.run "Native" (Nx_native.create_context ())
