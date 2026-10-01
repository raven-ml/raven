(* Performance guard for the self-contained GEMM used off macOS and whenever an
   operation is not eligible for Accelerate. This benchmark intentionally calls
   the backend's owned-GEMM hook; public matmul stays in packages/nx/bench. *)

external owned_matmul :
  ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t -> unit
  = "caml_nx_c_owned_matmul"

let make ?strides shape =
  let elements = Array.fold_left ( * ) 1 shape in
  let values =
    Bigarray.Array1.init Bigarray.float32 Bigarray.c_layout elements
      (fun index -> Float.sin (float_of_int (index * 17 mod 1021)) *. 0.25)
  in
  {
    Nx_array.dtype = Nx_dtype.float32;
    view = Nx_array.View.create ?strides shape;
    buffer = Nx_device.Buffer.of_bigarray values;
  }

let case ?a_strides name a_shape b_shape c_shape =
  let a = make ?strides:a_strides a_shape in
  let b = make b_shape in
  let c = make c_shape in
  Thumper.bench name (fun () ->
      owned_matmul c a b;
      Sys.opaque_identity ())

let () =
  Thumper.run "nx_c_owned_gemm"
    ~budgets:
      [
        Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
        Thumper.Budget.no_more_alloc_than 0.01;
      ]
    [
      Thumper.group "owned-gemm"
        [
          case "f32 64x64" [| 64; 64 |] [| 64; 64 |] [| 64; 64 |];
          case "f32 512x512" [| 512; 512 |] [| 512; 512 |] [| 512; 512 |];
          case ~a_strides:[| 1; 512 |] "f32 transposed 512x512" [| 512; 512 |]
            [| 512; 512 |] [| 512; 512 |];
          case "f32 batched 64x32x32" [| 64; 32; 32 |] [| 64; 32; 32 |]
            [| 64; 32; 32 |];
        ];
    ]
