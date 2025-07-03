open Alcotest

let test_initialization () =
  (* Should not raise *)
  Xla.initialize ()

let test_cpu_client_creation () =
  Xla.initialize ();
  (* Should create a client without raising *)
  let _client = Xla.Client.cpu () in
  ()

let test_gpu_client_creation () =
  Xla.initialize ();
  (* GPU client should fail on CPU-only machines *)
  check_raises "GPU client not available"
    (Failure "GPU client not yet implemented") (fun () ->
      ignore (Xla.Client.gpu ()))

let test_shape_creation () =
  (* Test creating a 2D shape *)
  let shape = Xla.Shape.create [| 2; 3 |] in
  check (array int) "dimensions" [| 2; 3 |] (Xla.Shape.dimensions shape);
  check int "rank" 2 (Xla.Shape.rank shape);
  check int "element_count" 6 (Xla.Shape.element_count shape)

let test_scalar_shape () =
  (* Test creating a scalar shape *)
  let shape = Xla.Shape.create [||] in
  check (array int) "dimensions" [||] (Xla.Shape.dimensions shape);
  check int "rank" 0 (Xla.Shape.rank shape);
  check int "element_count" 1 (Xla.Shape.element_count shape)

let test_literal_r0_f32 () =
  (* Test creating a scalar float32 literal *)
  let lit = Xla.Literal.create_r0_f32 42.0 in
  let shape = Xla.Literal.shape lit in
  check int "rank" 0 (Xla.Shape.rank shape);
  check int "element_count" 1 (Xla.Shape.element_count shape)

let test_simple_computation () =
  Xla.initialize ();
  let client = Xla.Client.cpu () in

  (* Build a simple computation: add two constants *)
  let builder = Xla.Builder.create "simple_add" in
  let const1 = Xla.Builder.constant builder (Xla.Literal.create_r0_f32 2.0) in
  let const2 = Xla.Builder.constant builder (Xla.Literal.create_r0_f32 3.0) in
  let sum = Xla.Builder.add builder const1 const2 in
  let computation = Xla.Builder.build builder sum in

  (* Compile and execute *)
  let executable = Xla.Computation.compile client computation in
  let _results = Xla.Computation.execute executable [] in
  (* For now just check it doesn't crash *)
  ()

let () =
  run "XLA"
    [
      ( "initialization",
        [ test_case "XLA initialization" `Quick test_initialization ] );
      ( "client",
        [
          test_case "CPU client creation" `Quick test_cpu_client_creation;
          test_case "GPU client creation" `Quick test_gpu_client_creation;
        ] );
      ( "shape",
        [
          test_case "Shape creation" `Quick test_shape_creation;
          test_case "Scalar shape" `Quick test_scalar_shape;
        ] );
      ( "literal",
        [ test_case "Scalar float32 literal" `Quick test_literal_r0_f32 ] );
      ( "computation",
        [ test_case "Simple computation" `Quick test_simple_computation ] );
    ]
