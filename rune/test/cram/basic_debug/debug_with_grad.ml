open Rune

let f x =
  let a = add x x in
  let b = mul a (full Float32 [| 2; 3 |] 2.0) in
  sum b

let () =
  let x = randn Float32 ~key:(Rng.key 42) [| 2; 3 |] in
  let y = debug (fun () -> grad f x) () in
  Printf.printf "Result shape: [%s]\n"
    (String.concat "," (Array.to_list (Array.map string_of_int (shape y))))
