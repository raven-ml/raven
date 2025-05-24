(* example/example_airline.ml *)
open Nx
open Nx_datasets

let astype_f32 arr = Nx.astype Nx.float32 arr

let () =
  Printf.printf "Loading Airline Passengers dataset...\n%!";
  let passengers = load_airline_passengers () in

  Printf.printf "Preparing data for plotting...\n%!";
  let passengers_f32 = astype_f32 passengers in
  let n_samples = (Nx.shape passengers_f32).(0) in

  let time_index =
    Nx.init float32 [| n_samples |] (fun indices ->
        match indices with
        | [| i |] -> Float.of_int i
        | _ -> failwith "Invalid index shape")
  in

  Printf.printf "Creating plot...\n%!";
  let fig =
    Hugin.plot ~title:"Airline Passengers 1949-1960" ~xlabel:"Month Index"
      ~ylabel:"Passengers (Thousands)" time_index passengers_f32
  in

  Printf.printf "Displaying plot...\n%!";
  Hugin.show fig;
  Printf.printf "Plot window closed.\n%!"
