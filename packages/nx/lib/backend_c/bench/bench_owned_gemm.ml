(* Performance guard for the self-contained GEMM used off macOS and whenever an
   operation is not eligible for Accelerate. This benchmark intentionally uses a
   backend-local maintenance hook; public matmul stays in packages/nx/bench. *)

type ('a, 'b) ffi = {
  buffer : Nx_device.Buffer.t;
  shape : int array;
  strides : int array;
  offset : int;
  dtype : ('a, 'b) Nx_dtype.t;
}

external owned_matmul : ('a, 'b) ffi -> ('a, 'b) ffi -> ('a, 'b) ffi -> unit
  = "caml_nx_c_owned_matmul"

let row_major shape =
  let rank = Array.length shape in
  let strides = Array.make rank 1 in
  for axis = rank - 2 downto 0 do
    strides.(axis) <- strides.(axis + 1) * shape.(axis + 1)
  done;
  strides

let make shape =
  let elements = Array.fold_left ( * ) 1 shape in
  let values =
    Bigarray.Array1.init Bigarray.float32 Bigarray.c_layout elements
      (fun index -> Float.sin (float_of_int (index * 17 mod 1021)) *. 0.25)
  in
  {
    buffer = Nx_device.Buffer.of_bigarray values;
    shape;
    strides = row_major shape;
    offset = 0;
    dtype = Nx_dtype.float32;
  }

let case ?a_strides name a_shape b_shape c_shape =
  let a = make a_shape in
  let a =
    match a_strides with None -> a | Some strides -> { a with strides }
  in
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
