(* Symbolic shapes for shape-polymorphic tensors. *)

type var = { name : string; min : int; max : int; mutable value : int option }
type dim = Static of int | Dynamic of var
type t = dim array

let static n =
  if n < 0 then
    Error.invalid ~op:"static"
      ~what:(Printf.sprintf "dimension %d" n)
      ~reason:"negative dimension" ();
  Static n

let dynamic name ~min ~max =
  if min < 0 then
    Error.invalid ~op:"dynamic"
      ~what:(Printf.sprintf "min=%d" min)
      ~reason:"must be non-negative" ();
  if min > max then
    Error.invalid ~op:"dynamic"
      ~what:(Printf.sprintf "bounds [%d, %d]" min max)
      ~reason:(Printf.sprintf "min > max")
      ();
  Dynamic { name; min; max; value = None }

let of_ints arr = Array.map (fun n -> static n) arr
let of_list lst = of_ints (Array.of_list lst)

let bind var value =
  if value < var.min || value > var.max then
    Error.invalid ~op:"bind"
      ~what:(Printf.sprintf "value %d for variable %s" value var.name)
      ~reason:(Printf.sprintf "outside bounds [%d, %d]" var.min var.max)
      ();
  var.value <- Some value

let eval_dim = function Static n -> Some n | Dynamic var -> var.value

let eval shape =
  let rec loop acc i =
    if i < 0 then Some (Array.of_list acc)
    else
      match eval_dim shape.(i) with
      | None -> None
      | Some n -> loop (n :: acc) (i - 1)
  in
  loop [] (Array.length shape - 1)

let vars shape =
  shape |> Array.to_list
  |> List.filter_map (function Static _ -> None | Dynamic var -> Some var)
  |> List.sort_uniq (fun v1 v2 -> String.compare v1.name v2.name)

let is_static shape =
  Array.for_all (function Static _ -> true | Dynamic _ -> false) shape

let rank shape = Array.length shape

let to_string shape =
  let dim_to_string = function
    | Static n -> string_of_int n
    | Dynamic var -> (
        match var.value with
        | None -> var.name
        | Some n -> Printf.sprintf "%s=%d" var.name n)
  in
  "[" ^ String.concat "; " (Array.to_list (Array.map dim_to_string shape)) ^ "]"

let equal s1 s2 =
  Array.length s1 = Array.length s2
  && Array.for_all2
       (fun d1 d2 ->
         match (d1, d2) with
         | Static n1, Static n2 -> n1 = n2
         | Dynamic v1, Dynamic v2 -> v1 == v2
         | _ -> false)
       s1 s2
