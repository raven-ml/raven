open Rune

let f x =
  let a = add x x in
  let b = mul a (full Float32 [| 2; 3 |] 2.0) in
  b

let () =
  let x = randn Float32 [| 2; 3 |] in
  let y = debug (grad f) x in
  Printf.printf "Result shape: [%s]\n"
    (String.concat "," (Array.to_list (Array.map string_of_int (shape y))))
