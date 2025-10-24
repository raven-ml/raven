(* example/example_airline.ml *)
open Nx
open Nx_datasets

let setup_logging () =
  Logs.set_reporter (Logs_fmt.reporter ());
  Logs.set_level (Some Logs.Info)

let astype_f32 arr = Nx.astype Nx.float32 arr

let () =
  setup_logging ();
  Logs.info (fun m -> m "Loading Airline Passengers dataset...");
  let passengers = load_airline_passengers () in

  Logs.info (fun m -> m "Preparing data for plotting...");
  let passengers_f32 = astype_f32 passengers in
  let n_samples = (Nx.shape passengers_f32).(0) in

  let time_index =
    Nx.init float32 [| n_samples |] (fun indices ->
        match indices with
        | [| i |] -> Float.of_int i
        | _ -> failwith "Invalid index shape")
  in

  Logs.info (fun m -> m "Creating plot...");
  let fig =
    Hugin.plot ~title:"Airline Passengers 1949-1960" ~xlabel:"Month Index"
      ~ylabel:"Passengers (Thousands)" time_index passengers_f32
  in

  Logs.info (fun m -> m "Displaying plot...");
  Hugin.show fig;
  Logs.info (fun m -> m "Plot window closed.")
