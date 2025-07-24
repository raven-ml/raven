module Make (Backend : Nx_core.Backend_intf.S) = struct
  (* Instantiate test modules with Native backend *)
  module Basics_tests = Test_nx_basics.Make (Backend)
  module Indexing_tests = Test_nx_indexing.Make (Backend)
  module Linalg_tests = Test_nx_linalg.Make (Backend)
  module Manipulation_tests = Test_nx_manipulation.Make (Backend)
  module Ops_tests = Test_nx_ops.Make (Backend)
  module Sanity_tests = Test_nx_sanity.Make (Backend)
  module Sorting_tests = Test_nx_sorting.Make (Backend)
  module Fft_tests = Test_nx_fft.Make (Backend)

  let run backend_name ctx =
    Printexc.record_backtrace true;

    (* Run all test suites *)
    Alcotest.run
      ("Nx " ^ backend_name ^ " Backend")
      (List.concat
         [
           Basics_tests.suite backend_name ctx;
           Indexing_tests.suite backend_name ctx;
           Linalg_tests.suite backend_name ctx;
           Manipulation_tests.suite backend_name ctx;
           Ops_tests.suite backend_name ctx;
           Sanity_tests.suite backend_name ctx;
           Sorting_tests.suite backend_name ctx;
           Fft_tests.tests ctx;
         ])
end
