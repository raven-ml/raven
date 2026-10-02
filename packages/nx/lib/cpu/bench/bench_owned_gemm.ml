(* Performance guard for the self-contained GEMM used off macOS and whenever an
   operation is not eligible for Accelerate. This benchmark intentionally calls
   the backend's owned-GEMM hook; public matmul stays in packages/nx/bench. *)

external owned_matmul :
  ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t -> unit
  = "caml_nx_c_owned_matmul"

(* An array of [shape] over fresh values, with the values. *)
let make ?strides shape =
  let elements = Array.fold_left ( * ) 1 shape in
  let values =
    Bigarray.Array1.init Bigarray.float32 Bigarray.c_layout elements
      (fun index -> Float.sin (float_of_int (index * 17 mod 1021)) *. 0.25)
  in
  ( {
      Nx_array.dtype = Nx_dtype.float32;
      view = Nx_array.View.create ?strides shape;
      buffer = Nx_device.Buffer.of_bigarray values;
    },
    values )

(* The element [(i, j)] of matrix [batch] of an array of rank 2 or 3. *)
let element ((a : (_, _) Nx_array.t), values) batch i j =
  let s = Nx_array.View.strides a.view in
  let r = Array.length s in
  let at = Nx_array.View.offset a.view + (i * s.(r - 2)) + (j * s.(r - 1)) in
  Bigarray.Array1.get values (if r = 3 then at + (batch * s.(0)) else at)

(* Fails unless the owned GEMM writes the product of [a] and [b] into [c],
   checked element by element against its definition. *)
let check name a b c =
  owned_matmul (fst c) (fst a) (fst b);
  let shape = Nx_array.View.shape (fst c).view in
  let r = Array.length shape in
  let k = (Nx_array.View.shape (fst a).view).(r - 1) in
  for batch = 0 to (if r = 3 then shape.(0) else 1) - 1 do
    for i = 0 to shape.(r - 2) - 1 do
      for j = 0 to shape.(r - 1) - 1 do
        let expected = ref 0. in
        for p = 0 to k - 1 do
          expected := !expected +. (element a batch i p *. element b batch p j)
        done;
        let got = element c batch i j in
        if Float.abs (got -. !expected) > 1e-4 *. (1. +. Float.abs !expected)
        then
          failwith
            (Printf.sprintf "%s: element (%d, %d, %d) is %g, not %g" name batch
               i j got !expected)
      done
    done
  done

let case ?a_strides name a_shape b_shape c_shape =
  let a = make ?strides:a_strides a_shape in
  let b = make b_shape in
  let c = make c_shape in
  check name a b c;
  let a = fst a and b = fst b and c = fst c in
  Thumper.bench name (fun () ->
      owned_matmul c a b;
      Sys.opaque_identity ())

let () =
  Thumper.run "nx_c_owned_gemm"
    ~budgets:
      [
        Thumper.Budget.no_slower_than 0.05;
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
  |> exit
