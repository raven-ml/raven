module Make (Backend : Nx_core.Backend_intf.S) = struct
  (* Instantiate test modules with Native backend *)
  module Basics_tests = Test_nx_basics.Make (Backend)
  module Indexing_tests = Test_nx_indexing.Make (Backend)
  module Linalg_tests = Test_nx_linalg.Make (Backend)
  module Manipulation_tests = Test_nx_manipulation.Make (Backend)
  module Neural_net_tests = Test_nx_neural_net.Make (Backend)
  module Ops_tests = Test_nx_ops.Make (Backend)
  module Sanity_tests = Test_nx_sanity.Make (Backend)
  module Sorting_tests = Test_nx_sorting.Make (Backend)
  module Fft_tests = Test_nx_fft.Make (Backend)
  module Cumsum_tests = Test_nx_cumsum.Make (Backend)

  let run backend_name ctx =
    Printexc.record_backtrace true;

    (* Run all test suites *)
    let backend_tests = List.concat
         [
           Basics_tests.suite backend_name ctx;
           Indexing_tests.suite backend_name ctx;
           Linalg_tests.suite backend_name ctx;
           Manipulation_tests.suite backend_name ctx;
           Neural_net_tests.suite backend_name ctx;
           Ops_tests.suite backend_name ctx;
           Sanity_tests.suite backend_name ctx;
           Sorting_tests.suite backend_name ctx;
           Fft_tests.tests ctx;
           Cumsum_tests.suite backend_name ctx;
         ] in
    
    (* Add cross-backend tests only for native backend to avoid duplication *)
    let all_tests = 
      if backend_name = "Native" then
        backend_tests @ [("Cumsum Cross-Backend Validation", Test_nx_cumsum.cross_backend_validation_tests)]
      else
        backend_tests
    in

    Alcotest.run ("Nx " ^ backend_name ^ " Backend") all_tests
end
