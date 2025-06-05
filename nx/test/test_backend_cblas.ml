(* Test suite for Nx_cblas backend *)

module Runner = Test_nx_unit.Make (Nx_cblas)

let () =
  Printexc.record_backtrace true;
  Runner.run "Native" (Nx_cblas.create_context ())
