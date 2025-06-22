open Rune

let () =
  let x = randn native Float32 [| 2; 3 |] in
  let y =
    debug
      (fun () ->
        let a = add x x in
        let b = mul a (full native Float32 [| 2; 3 |] 2.0) in
        b)
      ()
  in
  Printf.printf "Result shape: [%s]\n"
    (String.concat "," (Array.to_list (Array.map string_of_int (shape y))))
