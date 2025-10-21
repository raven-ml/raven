open Talon

let default_na_values = [ ""; "NA"; "N/A"; "null"; "NULL"; "nan"; "NaN" ]

(* Helper to check if a string is a null value *)
let is_null_value na_values s = List.mem s na_values

(* Helper to auto-detect column type from values *)
let detect_dtype na_values values =
  (* Try to parse as different types and see what works for all non-null
     values *)
  let non_null_values =
    List.filter (fun v -> not (is_null_value na_values v)) values
  in

  if List.length non_null_values = 0 then
    `String (* Default to string for all-null columns *)
  else
    (* Check if all can be parsed as bool *)
    let all_bool =
      List.for_all
        (fun v ->
          match String.lowercase_ascii v with
          | "true" | "t" | "yes" | "y" | "1" | "false" | "f" | "no" | "n" | "0"
            ->
              true
          | _ -> false)
        non_null_values
    in

    if all_bool then `Bool
    else
      (* Detect integers, promote to Int64 if any exceed Int32 range *)
      let all_int, needs_int64 =
        List.fold_left
          (fun (all_ok, overflow) v ->
            if not all_ok then (false, overflow)
            else
              try
                let i64 = Int64.of_string v in
                let too_big =
                  i64 > Int64.of_int32 Int32.max_int
                  || i64 < Int64.of_int32 Int32.min_int
                in
                (true, overflow || too_big)
              with _ -> (false, overflow))
          (true, false) non_null_values
      in
      if all_int then
        if needs_int64 then `Int64 else `Int32
      else
        let all_float =
          List.for_all
            (fun v ->
              try ignore (float_of_string v); true
              with _ -> false)
            non_null_values
        in
        if all_float then `Float32 else `String

let from_string ?(sep = ',') ?(header = true) ?(na_values = default_na_values)
    ?dtype_spec csv_string =
  let csv = Csv.of_string ~separator:sep csv_string in
  let rows = Csv.input_all csv in

  match rows with
  | [] -> empty
  | first_row :: data_rows ->
      let column_names, data_rows =
        if header then (first_row, data_rows)
        else
          (* Generate default column names *)
          let names =
            List.mapi (fun i _ -> Printf.sprintf "col%d" i) first_row
          in
          (names, rows)
      in

      if List.length data_rows = 0 then
        (* Only header, no data *)
        let columns =
          List.map (fun name -> (name, Col.string [||])) column_names
        in
        create columns
      else
        (* Transpose to get column-wise data *)
        let num_cols = List.length column_names in
        let columns_data = Array.init num_cols (fun _ -> []) in

        List.iter
          (fun row ->
            List.iteri
              (fun i value ->
                if i < num_cols then
                  columns_data.(i) <- value :: columns_data.(i))
              row)
          data_rows;

        (* Reverse to maintain order *)
        Array.iteri (fun i lst -> columns_data.(i) <- List.rev lst) columns_data;

        (* Create columns based on dtype_spec or auto-detection *)
        let columns =
          List.mapi
            (fun i name ->
              let values = columns_data.(i) in
              let dtype =
                match dtype_spec with
                | Some specs -> (
                    try List.assoc name specs
                    with Not_found -> detect_dtype na_values values)
                | None -> detect_dtype na_values values
              in

              let column =
                match dtype with
                | `Float32 ->
                    let arr =
                      List.map
                        (fun v ->
                          if is_null_value na_values v then None
                          else try Some (float_of_string v) with _ -> None)
                        values
                      |> Array.of_list
                    in
                    Col.float32_opt arr
                | `Float64 ->
                    let arr =
                      List.map
                        (fun v ->
                          if is_null_value na_values v then None
                          else try Some (float_of_string v) with _ -> None)
                        values
                      |> Array.of_list
                    in
                    Col.float64_opt arr
                | `Int32 ->
                    let arr =
                      List.map
                        (fun v ->
                          if is_null_value na_values v then None
                          else try Some (Int32.of_string v) with _ -> None)
                        values
                      |> Array.of_list
                    in
                    Col.int32_opt arr
                | `Int64 ->
                    let arr =
                      List.map
                        (fun v ->
                          if is_null_value na_values v then None
                          else try Some (Int64.of_string v) with _ -> None)
                        values
                      |> Array.of_list
                    in
                    Col.int64_opt arr
                | `Bool ->
                    let arr =
                      List.map
                        (fun v ->
                          if is_null_value na_values v then None
                          else
                            match String.lowercase_ascii v with
                            | "true" | "t" | "yes" | "y" | "1" -> Some true
                            | "false" | "f" | "no" | "n" | "0" -> Some false
                            | _ -> None)
                        values
                      |> Array.of_list
                    in
                    Col.bool_opt arr
                | `String ->
                    let arr =
                      List.map
                        (fun v ->
                          if is_null_value na_values v then None else Some v)
                        values
                      |> Array.of_list
                    in
                    Col.string_opt arr
              in
              (name, column))
            column_names
        in

        create columns

let read ?sep ?header ?na_values ?dtype_spec file =
  let ic = open_in file in
  let contents = really_input_string ic (in_channel_length ic) in
  close_in ic;
  from_string ?sep ?header ?na_values ?dtype_spec contents

let to_string ?(sep = ',') ?(header = true) ?(na_repr = "") df =
  let buffer = Buffer.create 1024 in
  let csv = Csv.to_buffer ~separator:sep buffer in

  (* Write header if requested *)
  if header then Csv.output_record csv (column_names df);

  (* Write data rows *)
  let n_rows = num_rows df in
  for i = 0 to n_rows - 1 do
    let row =
      List.map
        (fun col_name ->
          let col = get_column_exn df col_name in
          match col with
          | Col.P (dtype, tensor, _) -> (
              match dtype with
              | Nx.Float32 ->
                  let arr : float array = Nx.to_array tensor in
                  let value = arr.(i) in
                  if classify_float value = FP_nan then na_repr
                  else string_of_float value
              | Nx.Float64 ->
                  let arr : float array = Nx.to_array tensor in
                  let value = arr.(i) in
                  if classify_float value = FP_nan then na_repr
                  else string_of_float value
              | Nx.Float16 ->
                  let arr : float array = Nx.to_array tensor in
                  let value = arr.(i) in
                  if classify_float value = FP_nan then na_repr
                  else string_of_float value
              | Nx.BFloat16 ->
                  let arr : float array = Nx.to_array tensor in
                  let value = arr.(i) in
                  if classify_float value = FP_nan then na_repr
                  else string_of_float value
              | Nx.Int8 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.UInt8 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.Int16 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.UInt16 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.Int32 ->
                  let arr : int32 array = Nx.to_array tensor in
                  Int32.to_string arr.(i)
              | Nx.Int64 ->
                  let arr : int64 array = Nx.to_array tensor in
                  Int64.to_string arr.(i)
              | Nx.Int ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.NativeInt ->
                  let arr : nativeint array = Nx.to_array tensor in
                  Nativeint.to_string arr.(i)
              | Nx.Complex32 ->
                  let arr : Complex.t array = Nx.to_array tensor in
                  let c = arr.(i) in
                  Printf.sprintf "%g+%gi" c.re c.im
              | Nx.Complex64 ->
                  let arr : Complex.t array = Nx.to_array tensor in
                  let c = arr.(i) in
                  Printf.sprintf "%g+%gi" c.re c.im
              | Nx.Bool ->
                  let arr : bool array = Nx.to_array tensor in
                  string_of_bool arr.(i)
              | Nx.Int4 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.UInt4 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.Float8_e4m3 ->
                  let arr : float array = Nx.to_array tensor in
                  let value = arr.(i) in
                  if classify_float value = FP_nan then na_repr
                  else string_of_float value
              | Nx.Float8_e5m2 ->
                  let arr : float array = Nx.to_array tensor in
                  let value = arr.(i) in
                  if classify_float value = FP_nan then na_repr
                  else string_of_float value
              | Nx.Complex16 ->
                  let arr : Complex.t array = Nx.to_array tensor in
                  let c = arr.(i) in
                  Printf.sprintf "%g+%gi" c.re c.im
              | Nx.QInt8 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.QUInt8 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i))
          | Col.S arr -> ( match arr.(i) with Some s -> s | None -> na_repr)
          | Col.B arr -> (
              match arr.(i) with Some b -> string_of_bool b | None -> na_repr))
        (column_names df)
    in
    Csv.output_record csv row
  done;

  Csv.close_out csv;
  Buffer.contents buffer

let write ?sep ?header ?na_repr df file =
  let csv_string = to_string ?sep ?header ?na_repr df in
  let oc = open_out file in
  output_string oc csv_string;
  close_out oc
