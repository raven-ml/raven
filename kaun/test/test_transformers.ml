(** Tests for transformer building blocks *)

open Alcotest
open Rune
open Kaun.Transformers

let test_sdpa_shape () =
  let device = c in
  let dtype = float32 in
  let b, t, h, hs = (2, 16, 4, 32) in

  (* Create Q, K, V tensors *)
  let q = rand device dtype [| b; h; t; hs |] in
  let k = rand device dtype [| b; h; t; hs |] in
  let v = rand device dtype [| b; h; t; hs |] in

  (* Apply SDPA *)
  let output = scaled_dot_product_attention q k v in

  (* Check output shape *)
  check (array Alcotest.int) "output shape" [| b; h; t; hs |] (shape output)

let test_sdpa_causal () =
  let device = c in
  let dtype = float32 in
  let b, t, h, hs = (1, 8, 1, 16) in

  (* Create simple Q, K, V tensors *)
  let q = ones device dtype [| b; h; t; hs |] in
  let k = ones device dtype [| b; h; t; hs |] in
  let v = arange_f device dtype 0.0 (float_of_int t) 1.0 in
  let v = reshape [| 1; 1; t; 1 |] v in
  let v = broadcast_to [| b; h; t; hs |] v in

  (* Apply causal SDPA *)
  let output = scaled_dot_product_attention ~is_causal:true q k v in

  (* With causal mask, each position should only attend to previous positions *)
  (* So output[i] should be average of v[0..i] *)
  let output_data = unsafe_data (reshape [| t; hs |] output) in

  (* Check that later positions have different values (showing causal masking
     works) *)
  let val_0 = Bigarray_ext.Array1.get output_data 0 in
  let val_last = Bigarray_ext.Array1.get output_data ((t - 1) * hs) in
  check bool "causal mask applied" true (val_0 <> val_last)

let test_rope_shape () =
  let b, t, h, hs = (2, 16, 4, 32) in

  (* Create RoPE *)
  let rope = Rope.make ~dim:(hs * 2) ~max_seq_len:t () in

  (* Create Q, K tensors *)
  let device = c in
  let dtype = float32 in
  let q = rand device dtype [| b; h; t; hs |] in
  let k = rand device dtype [| b; h; t; hs |] in

  (* Apply RoPE *)
  let q_rot, k_rot = Rope.apply rope q k in

  (* Check output shapes *)
  check (array Alcotest.int) "q_rot shape" [| b; h; t; hs |] (shape q_rot);
  check (array Alcotest.int) "k_rot shape" [| b; h; t; hs |] (shape k_rot)

(* Test suite *)
let () =
  let open Alcotest in
  run "Kaun.Transformers"
    [
      ( "attention",
        [
          test_case "SDPA shape" `Quick test_sdpa_shape;
          test_case "SDPA causal" `Quick test_sdpa_causal;
        ] );
      ("rope", [ test_case "RoPE shape" `Quick test_rope_shape ]);
    ]
