(* airline_passengers.ml *)
open Bigarray
open Dataset_utils

let dataset_name = "airline-passengers"
let dataset_dir = get_cache_dir dataset_name
let data_filename = "airline-passengers.csv"
let data_path = dataset_dir ^ data_filename

let url =
  "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

let ensure_dataset () = ensure_file url data_path

let load () =
  ensure_dataset ();
  Printf.printf "Loading Airline Passengers dataset...\n%!";

  let header, data_rows_iter =
    try
      (* Csv.Rows.load returns the header list and an iterator over Row.t *)
      let chan =
        Csv.of_channel ~has_header:true ~separator:',' (open_in data_path)
      in
      let h = Csv.Rows.header chan in
      (h, chan)
      (* Return header and the open Csv.in_channel as an iterator *)
    with
    | Csv.Failure (r, c, msg) ->
        failwith
          (Printf.sprintf "CSV Parsing Error in %s at row %d, col %d: %s"
             data_path r c msg)
    | Sys_error msg ->
        failwith (Printf.sprintf "Cannot open file %s: %s" data_path msg)
    | ex ->
        failwith
          (Printf.sprintf "Error loading CSV %s: %s" data_path
             (Printexc.to_string ex))
  in

  let passenger_col_name = "Passengers" in
  let passenger_col_index =
    match List.find_index (( = ) passenger_col_name) header with
    | Some idx -> idx
    | None ->
        Csv.close_in data_rows_iter;
        (* Close channel before failing *)
        failwith
          ("Required column '" ^ passenger_col_name ^ "' not found in header: "
         ^ String.concat ", " header)
  in

  (* Iterate using Csv.Rows.fold_left to collect data *)
  let collected_data =
    try
      Csv.Rows.fold_left
        ~f:(fun acc row ->
          let row_list = Csv.Row.to_list row in
          (* Convert Row.t to string list *)
          if List.length row_list <> List.length header then
            Printf.eprintf "Warning: Row %d has %d columns, expected %d\n%!"
              (List.length acc + 1)
              (List.length row_list) (List.length header);

          (* Check length before accessing *)
          if List.length row_list > passenger_col_index then
            let passenger_str = List.nth row_list passenger_col_index in
            let context () =
              Printf.sprintf "row %d, col %s"
                (List.length acc + 1)
                passenger_col_name
            in
            let passenger_int = parse_int_cell ~context passenger_str in
            passenger_int :: acc
          else (
            Printf.eprintf
              "Warning: Row %d is shorter than expected (%d < %d), skipping \
               passenger value.\n\
               %!"
              (List.length acc + 1)
              (List.length row_list) (passenger_col_index + 1);
            -1 :: acc (* Placeholder for missing data *)))
        ~init:[] data_rows_iter
    with
    | Csv.Failure (r, c, msg) ->
        Csv.close_in data_rows_iter;
        failwith
          (Printf.sprintf
             "CSV Parsing Error during iteration at approx row %d, field %d: %s"
             r c msg)
    | ex ->
        Csv.close_in data_rows_iter;
        failwith
          (Printf.sprintf "Error iterating CSV %s: %s" data_path
             (Printexc.to_string ex))
  in
  Csv.close_in data_rows_iter;

  (* Important: Close the channel after iteration *)
  let data_rows_rev = collected_data in
  let num_samples = List.length data_rows_rev in
  if num_samples = 0 then
    failwith "No data rows loaded from airline-passengers.csv";
  Printf.printf "Found %d samples.\n%!" num_samples;

  (* Create Bigarray and populate (data is reversed from fold_left) *)
  let passengers = Array1.create int32 c_layout num_samples in
  List.iteri
    (fun i passenger_val ->
      passengers.{num_samples - 1 - i} <- Int32.of_int passenger_val)
    data_rows_rev;

  Printf.printf "Airline Passengers loading complete.\n%!";
  passengers
