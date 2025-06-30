(* Test suite for Nx_c backend *)

module Runner = Test_nx_unit.Make (Nx_c)

let () =
  Printexc.record_backtrace true;
  Runner.run "C" (Nx_c.create_context ())
