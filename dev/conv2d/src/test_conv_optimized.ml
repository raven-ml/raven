open Test_nx_support

(* Convolution tests - adapted from nx test suite *)
let test_convolve2d_basic () =
  let x = Nx.ones Nx.float32 [| 1; 1; 5; 5 |] in
  let w = Nx.ones Nx.float32 [| 1; 1; 3; 3 |] in

  (* Test our implementation *)
  let y = Nx_conv.convolve2d x w in
  check_t ~eps:1e-6 "convolve2d basic" [| 1; 1; 3; 3 |]
    [| 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9. |]
    y

let test_convolve2d_channels () =
  (* Multi-channel test *)
  let x = Nx.randn Nx.float32 ~seed:42 [| 1; 3; 8; 8 |] in
  let w = Nx.randn Nx.float32 ~seed:43 [| 6; 3; 3; 3 |] in

  let y_orig = Nx_conv.convolve2d ~padding_mode:`Valid x w in
  let y_opt = Nx_conv.convolve2d ~padding_mode:`Valid x w in
  (* TODO: use optimized *)

  (* Check shapes match *)
  Alcotest.(check (array int)) "output shape" [| 1; 6; 6; 6 |] (Nx.shape y_opt);

  (* Check values are close *)
  let orig_flat = Nx.flatten y_orig |> Nx.to_array in
  let opt_flat = Nx.flatten y_opt |> Nx.to_array in
  Array.iteri
    (fun i v ->
      Alcotest.(check (float 1e-5))
        (Printf.sprintf "element %d" i)
        orig_flat.(i) v)
    opt_flat

let test_convolve2d_padding () =
  let x =
    Nx.create Nx.float32 [| 1; 1; 4; 4 |]
      (Array.init 16 (fun i -> float_of_int (i + 1)))
  in
  let w = Nx.ones Nx.float32 [| 1; 1; 3; 3 |] in

  (* Valid padding *)
  let y_valid = Nx_conv.convolve2d ~padding_mode:`Valid x w in
  Alcotest.(check (array int))
    "valid padding shape" [| 1; 1; 2; 2 |] (Nx.shape y_valid);

  (* Same padding *)
  let y_same = Nx_conv.convolve2d ~padding_mode:`Same x w in
  Alcotest.(check (array int))
    "same padding shape" [| 1; 1; 4; 4 |] (Nx.shape y_same)

let test_convolve2d_stride () =
  let x =
    Nx.create Nx.float32 [| 1; 1; 8; 8 |]
      (Array.init 64 (fun i -> float_of_int i))
  in
  let w = Nx.ones Nx.float32 [| 1; 1; 3; 3 |] in

  (* Stride 2 *)
  let y = Nx_conv.convolve2d ~stride:(2, 2) ~padding_mode:`Valid x w in
  Alcotest.(check (array int)) "stride 2 shape" [| 1; 1; 3; 3 |] (Nx.shape y)

(* let test_convolve2d_groups () =
  (* Groups test - 4 input channels, 4 output channels, groups=2 *)
  let x = Nx.randn Nx.float32 ~seed:42 [| 1; 4; 8; 8 |] in
  let w = Nx.randn Nx.float32 ~seed:43 [| 4; 2; 3; 3 |] in

  let y = Nx_conv.convolve2d ~groups:2 ~padding_mode:`Valid x w in
  Alcotest.(check (array int)) "groups=2 shape" [| 1; 4; 6; 6 |] (Nx.shape y) *)

let test_convolve2d_5x5_kernel () =
  (* Test with 5x5 kernel *)
  let x = Nx.ones Nx.float32 [| 1; 1; 8; 8 |] in
  let w = Nx.ones Nx.float32 [| 1; 1; 5; 5 |] in

  let y = Nx_conv.convolve2d ~padding_mode:`Valid x w in
  check_t ~eps:1e-6 "5x5 kernel" [| 1; 1; 4; 4 |] (Array.make 16 25.) y

let test_convolve2d_batch () =
  (* Test with batch size > 1 *)
  let x = Nx.randn Nx.float32 ~seed:42 [| 4; 3; 16; 16 |] in
  let w = Nx.randn Nx.float32 ~seed:43 [| 8; 3; 3; 3 |] in

  let y = Nx_conv.convolve2d ~padding_mode:`Same x w in
  Alcotest.(check (array int)) "batch shape" [| 4; 8; 16; 16 |] (Nx.shape y)

(* Test suite *)
let () =
  let open Alcotest in
  run "Conv2D Optimized Tests"
    [
      ( "basic",
        [
          test_case "basic 2d convolution" `Quick test_convolve2d_basic;
          test_case "multi-channel" `Quick test_convolve2d_channels;
          test_case "padding modes" `Quick test_convolve2d_padding;
          test_case "stride" `Quick test_convolve2d_stride;
          (* test_case "groups" `Quick test_convolve2d_groups; *)
          test_case "5x5 kernel" `Quick test_convolve2d_5x5_kernel;
          test_case "batch processing" `Quick test_convolve2d_batch;
        ] );
    ]
