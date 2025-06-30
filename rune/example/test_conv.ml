open Rune

let () =
  let ctx = ocaml in

  (* Simple test case *)
  let x = randn ctx Float32 [| 2; 1; 5; 5 |] in
  (* batch=2, channels=1, 5x5 *)
  let w = randn ctx Float32 [| 3; 1; 3; 3 |] in
  (* 3 output channels, 3x3 kernel *)

  Printf.printf "Input shape: %s\n"
    (Array.to_list (shape x) |> List.map string_of_int |> String.concat ", ");
  Printf.printf "Weight shape: %s\n"
    (Array.to_list (shape w) |> List.map string_of_int |> String.concat ", ");

  (* Forward pass *)
  let y = convolve2d x w ~padding_mode:`Valid in
  Printf.printf "Output shape: %s\n"
    (Array.to_list (shape y) |> List.map string_of_int |> String.concat ", ");

  (* Test gradient *)
  let loss_fn params =
    match params with
    | [ w ] ->
        let y = convolve2d x w ~padding_mode:`Valid in
        sum y
    | _ -> failwith "wrong params"
  in

  let loss, grads = value_and_grads loss_fn [ w ] in
  Printf.printf "Loss: %f\n" (unsafe_get [] loss);
  Printf.printf "Gradient shape: %s\n"
    (Array.to_list (shape (List.hd grads))
    |> List.map string_of_int |> String.concat ", ")
