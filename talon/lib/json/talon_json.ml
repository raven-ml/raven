open Talon
open Yojson.Basic

(* Helper to convert a column value to JSON *)
let value_to_json col idx =
  match col with
  | Col.P (dtype, tensor, _) -> (
      match dtype with
      | Nx.Float32 ->
          let arr : float array = Nx.to_array tensor in
          let value = arr.(idx) in
          if classify_float value = FP_nan then `Null else `Float value
      | Nx.Float64 ->
          let arr : float array = Nx.to_array tensor in
          let value = arr.(idx) in
          if classify_float value = FP_nan then `Null else `Float value
      | Nx.Float16 ->
          let arr : float array = Nx.to_array tensor in
          let value = arr.(idx) in
          if classify_float value = FP_nan then `Null else `Float value
      | Nx.BFloat16 ->
          let arr : float array = Nx.to_array tensor in
          let value = arr.(idx) in
          if classify_float value = FP_nan then `Null else `Float value
      | Nx.Int8 ->
          let arr : int array = Nx.to_array tensor in
          `Int arr.(idx)
      | Nx.UInt8 ->
          let arr : int array = Nx.to_array tensor in
          `Int arr.(idx)
      | Nx.Int16 ->
          let arr : int array = Nx.to_array tensor in
          `Int arr.(idx)
      | Nx.UInt16 ->
          let arr : int array = Nx.to_array tensor in
          `Int arr.(idx)
      | Nx.Int32 ->
          let arr : int32 array = Nx.to_array tensor in
          `Int (Int32.to_int arr.(idx))
      | Nx.Int64 ->
          let arr : int64 array = Nx.to_array tensor in
          `String (Int64.to_string arr.(idx))
      | Nx.Int ->
          let arr : int array = Nx.to_array tensor in
          `Int arr.(idx)
      | Nx.NativeInt ->
          let arr : nativeint array = Nx.to_array tensor in
          `String (Nativeint.to_string arr.(idx))
      | Nx.Complex32 ->
          let arr : Complex.t array = Nx.to_array tensor in
          let c = arr.(idx) in
          `String (Printf.sprintf "%g+%gi" c.re c.im)
      | Nx.Complex64 ->
          let arr : Complex.t array = Nx.to_array tensor in
          let c = arr.(idx) in
          `String (Printf.sprintf "%g+%gi" c.re c.im)
      | Nx.Bool ->
          let arr : bool array = Nx.to_array tensor in
          `Bool arr.(idx)
      | Nx.Int4 ->
          let arr : int array = Nx.to_array tensor in
          `Int arr.(idx)
      | Nx.UInt4 ->
          let arr : int array = Nx.to_array tensor in
          `Int arr.(idx)
      | Nx.Float8_e4m3 ->
          let arr : float array = Nx.to_array tensor in
          let value = arr.(idx) in
          if classify_float value = FP_nan then `Null else `Float value
      | Nx.Float8_e5m2 ->
          let arr : float array = Nx.to_array tensor in
          let value = arr.(idx) in
          if classify_float value = FP_nan then `Null else `Float value
      | Nx.Complex16 ->
          let arr : Complex.t array = Nx.to_array tensor in
          let c = arr.(idx) in
          `String (Printf.sprintf "%g+%gi" c.re c.im)
      | Nx.QInt8 ->
          let arr : int array = Nx.to_array tensor in
          `Int arr.(idx)
      | Nx.QUInt8 ->
          let arr : int array = Nx.to_array tensor in
          `Int arr.(idx))
  | Col.S arr -> ( match arr.(idx) with Some s -> `String s | None -> `Null)
  | Col.B arr -> ( match arr.(idx) with Some b -> `Bool b | None -> `Null)

let to_string ?(orient = `Records) df =
  let n_rows = num_rows df in
  let col_names = column_names df in

  match orient with
  | `Records ->
      (* Row-oriented: list of objects *)
      let records =
        List.init n_rows (fun i ->
            let fields =
              List.map
                (fun col_name ->
                  let col = get_column_exn df col_name in
                  (col_name, value_to_json col i))
                col_names
            in
            `Assoc fields)
      in
      to_string (`List records)
  | `Columns ->
      (* Column-oriented: object with column arrays *)
      let columns =
        List.map
          (fun col_name ->
            let col = get_column_exn df col_name in
            let values = List.init n_rows (fun i -> value_to_json col i) in
            (col_name, `List values))
          col_names
      in
      to_string (`Assoc columns)

(* Helper to detect column type from JSON values *)
let detect_json_dtype values =
  let non_null_values = List.filter (fun v -> v <> `Null) values in

  if List.length non_null_values = 0 then
    `String (* Default to string for all-null columns *)
  else
    (* Check types of non-null values *)
    let all_bool =
      List.for_all (function `Bool _ -> true | _ -> false) non_null_values
    in
    let all_int =
      List.for_all (function `Int _ -> true | _ -> false) non_null_values
    in
    let all_float =
      List.for_all
        (function `Float _ | `Int _ -> true | _ -> false)
        non_null_values
    in

    if all_bool then `Bool
    else if all_int then `Int32
    else if all_float then `Float32
    else `String

let from_string ?(orient = `Records) json_str =
  let json = from_string json_str in

  match orient with
  | `Records -> (
      (* Row-oriented: expect list of objects *)
      match json with
      | `List records ->
          if List.length records = 0 then empty
          else
            (* Get column names from first record *)
            let col_names =
              match List.hd records with
              | `Assoc fields -> List.map fst fields
              | _ -> failwith "Invalid JSON: expected object in records array"
            in

            (* Extract values for each column *)
            let columns_data =
              List.map
                (fun col_name ->
                  let values =
                    List.map
                      (function
                        | `Assoc fields -> (
                            try List.assoc col_name fields
                            with Not_found -> `Null)
                        | _ -> `Null)
                      records
                  in
                  (col_name, values))
                col_names
            in

            (* Create columns based on detected types *)
            let columns =
              List.map
                (fun (col_name, values) ->
                  let dtype = detect_json_dtype values in

                  let column =
                    match dtype with
                    | `Float32 ->
                        let arr =
                          List.map
                            (function
                              | `Null -> None
                              | `Float f -> Some f
                              | `Int i -> Some (float_of_int i)
                              | _ -> None)
                            values
                          |> Array.of_list
                        in
                        Col.float32_opt arr
                    | `Int32 ->
                        let arr =
                          List.map
                            (function
                              | `Null -> None
                              | `Int i -> Some (Int32.of_int i)
                              | _ -> None)
                            values
                          |> Array.of_list
                        in
                        Col.int32_opt arr
                    | `Bool ->
                        let arr =
                          List.map
                            (function
                              | `Null -> None | `Bool b -> Some b | _ -> None)
                            values
                          |> Array.of_list
                        in
                        Col.bool_opt arr
                    | _ ->
                        (* String or mixed types *)
                        let arr =
                          List.map
                            (function
                              | `Null -> None
                              | `String s -> Some s
                              | `Int i -> Some (string_of_int i)
                              | `Float f -> Some (string_of_float f)
                              | `Bool b -> Some (string_of_bool b)
                              | _ -> None)
                            values
                          |> Array.of_list
                        in
                        Col.string_opt arr
                  in
                  (col_name, column))
                columns_data
            in

            create columns
      | _ -> failwith "Invalid JSON: expected array for records orientation")
  | `Columns -> (
      (* Column-oriented: expect object with column arrays *)
      match json with
      | `Assoc fields ->
          if List.length fields = 0 then empty
          else
            let columns =
              List.map
                (fun (col_name, values) ->
                  match values with
                  | `List vals ->
                      let dtype = detect_json_dtype vals in

                      let column =
                        match dtype with
                        | `Float32 ->
                            let arr =
                              List.map
                                (function
                                  | `Null -> None
                                  | `Float f -> Some f
                                  | `Int i -> Some (float_of_int i)
                                  | _ -> None)
                                vals
                              |> Array.of_list
                            in
                            Col.float32_opt arr
                        | `Int32 ->
                            let arr =
                              List.map
                                (function
                                  | `Null -> None
                                  | `Int i -> Some (Int32.of_int i)
                                  | _ -> None)
                                vals
                              |> Array.of_list
                            in
                            Col.int32_opt arr
                        | `Bool ->
                            let arr =
                              List.map
                                (function
                                  | `Null -> None
                                  | `Bool b -> Some b
                                  | _ -> None)
                                vals
                              |> Array.of_list
                            in
                            Col.bool_opt arr
                        | _ ->
                            (* String or mixed types *)
                            let arr =
                              List.map
                                (function
                                  | `Null -> None
                                  | `String s -> Some s
                                  | `Int i -> Some (string_of_int i)
                                  | `Float f -> Some (string_of_float f)
                                  | `Bool b -> Some (string_of_bool b)
                                  | _ -> None)
                                vals
                              |> Array.of_list
                            in
                            Col.string_opt arr
                      in
                      (col_name, column)
                  | _ ->
                      failwith
                        (Printf.sprintf
                           "Invalid JSON: column %s is not an array" col_name))
                fields
            in

            create columns
      | _ -> failwith "Invalid JSON: expected object for columns orientation")

let to_file ?orient df file =
  let json_string = to_string ?orient df in
  let oc = open_out file in
  output_string oc json_string;
  close_out oc

let from_file ?orient file =
  let ic = open_in file in
  let contents = really_input_string ic (in_channel_length ic) in
  close_in ic;
  from_string ?orient contents
