(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let json_assoc = function
  | Jsont.Object (mems, _) -> List.map (fun ((n, _), v) -> (n, v)) mems
  | _ -> []

let json_of_string s =
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok v -> v
  | Error e -> failwith e

let json_to_string j =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json j with
  | Ok s -> s
  | Error e -> failwith e

let mask_true mask idx = match mask with Some m -> m.(idx) | None -> false

let float_is_null mask idx value =
  if mask_true mask idx then true else classify_float value = FP_nan

let int32_is_null mask idx value =
  if mask_true mask idx then true else value = Int32.min_int

let int64_is_null mask idx value =
  if mask_true mask idx then true else value = Int64.min_int

(* Helper to convert a column value to JSON *)
let value_to_json col idx =
  match col with
  | Col.P (dtype, tensor, mask_opt) -> (
      match dtype with
      | Nx.Float32 ->
          let arr : float array = Nx.to_array tensor in
          let value = arr.(idx) in
          if float_is_null mask_opt idx value then Jsont.Json.null ()
          else Jsont.Json.number value
      | Nx.Float64 ->
          let arr : float array = Nx.to_array tensor in
          let value = arr.(idx) in
          if float_is_null mask_opt idx value then Jsont.Json.null ()
          else Jsont.Json.number value
      | Nx.Float16 ->
          let arr : float array = Nx.to_array tensor in
          let value = arr.(idx) in
          if float_is_null mask_opt idx value then Jsont.Json.null ()
          else Jsont.Json.number value
      | Nx.BFloat16 ->
          let arr : float array = Nx.to_array tensor in
          let value = arr.(idx) in
          if float_is_null mask_opt idx value then Jsont.Json.null ()
          else Jsont.Json.number value
      | Nx.Int8 ->
          let arr : int array = Nx.to_array tensor in
          if mask_true mask_opt idx then Jsont.Json.null ()
          else Jsont.Json.int arr.(idx)
      | Nx.UInt8 ->
          let arr : int array = Nx.to_array tensor in
          if mask_true mask_opt idx then Jsont.Json.null ()
          else Jsont.Json.int arr.(idx)
      | Nx.Int16 ->
          let arr : int array = Nx.to_array tensor in
          if mask_true mask_opt idx then Jsont.Json.null ()
          else Jsont.Json.int arr.(idx)
      | Nx.UInt16 ->
          let arr : int array = Nx.to_array tensor in
          if mask_true mask_opt idx then Jsont.Json.null ()
          else Jsont.Json.int arr.(idx)
      | Nx.Int32 ->
          let arr : int32 array = Nx.to_array tensor in
          let value = arr.(idx) in
          if int32_is_null mask_opt idx value then Jsont.Json.null ()
          else Jsont.Json.int (Int32.to_int value)
      | Nx.Int64 ->
          let arr : int64 array = Nx.to_array tensor in
          let value = arr.(idx) in
          if int64_is_null mask_opt idx value then Jsont.Json.null ()
          else Jsont.Json.string (Int64.to_string value)
      | Nx.UInt32 ->
          let arr : int32 array = Nx.to_array tensor in
          let value = arr.(idx) in
          if int32_is_null mask_opt idx value then Jsont.Json.null ()
          else Jsont.Json.int (Int32.to_int value)
      | Nx.UInt64 ->
          let arr : int64 array = Nx.to_array tensor in
          let value = arr.(idx) in
          if int64_is_null mask_opt idx value then Jsont.Json.null ()
          else Jsont.Json.string (Int64.to_string value)
      | Nx.Complex64 ->
          let arr : Complex.t array = Nx.to_array tensor in
          let c = arr.(idx) in
          if mask_true mask_opt idx then Jsont.Json.null ()
          else Jsont.Json.string (Printf.sprintf "%g+%gi" c.re c.im)
      | Nx.Complex128 ->
          let arr : Complex.t array = Nx.to_array tensor in
          let c = arr.(idx) in
          if mask_true mask_opt idx then Jsont.Json.null ()
          else Jsont.Json.string (Printf.sprintf "%g+%gi" c.re c.im)
      | Nx.Bool ->
          let arr : bool array = Nx.to_array tensor in
          if mask_true mask_opt idx then Jsont.Json.null ()
          else Jsont.Json.bool arr.(idx)
      | Nx.Int4 ->
          let arr : int array = Nx.to_array tensor in
          if mask_true mask_opt idx then Jsont.Json.null ()
          else Jsont.Json.int arr.(idx)
      | Nx.UInt4 ->
          let arr : int array = Nx.to_array tensor in
          if mask_true mask_opt idx then Jsont.Json.null ()
          else Jsont.Json.int arr.(idx)
      | Nx.Float8_e4m3 ->
          let arr : float array = Nx.to_array tensor in
          let value = arr.(idx) in
          if float_is_null mask_opt idx value then Jsont.Json.null ()
          else Jsont.Json.number value
      | Nx.Float8_e5m2 ->
          let arr : float array = Nx.to_array tensor in
          let value = arr.(idx) in
          if float_is_null mask_opt idx value then Jsont.Json.null ()
          else Jsont.Json.number value)
  | Col.S arr -> (
      match arr.(idx) with
      | Some s -> Jsont.Json.string s
      | None -> Jsont.Json.null ())
  | Col.B arr -> (
      match arr.(idx) with
      | Some b -> Jsont.Json.bool b
      | None -> Jsont.Json.null ())

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
            json_obj fields)
      in
      json_to_string (Jsont.Json.list records)
  | `Columns ->
      (* Column-oriented: object with column arrays *)
      let columns =
        List.map
          (fun col_name ->
            let col = get_column_exn df col_name in
            let values = List.init n_rows (fun i -> value_to_json col i) in
            (col_name, Jsont.Json.list values))
          col_names
      in
      json_to_string (json_obj columns)

(* Helper to detect column type from JSON values *)
let detect_json_dtype values =
  let is_null = function Jsont.Null _ -> true | _ -> false in
  let non_null_values = List.filter (fun v -> not (is_null v)) values in

  if List.length non_null_values = 0 then
    `String (* Default to string for all-null columns *)
  else
    (* Check types of non-null values *)
    let all_bool =
      List.for_all
        (function Jsont.Bool _ -> true | _ -> false)
        non_null_values
    in
    let all_int =
      List.for_all
        (function
          | Jsont.Number (f, _) -> Float.is_integer f
          | _ -> false)
        non_null_values
    in
    let all_float =
      List.for_all
        (function Jsont.Number _ -> true | _ -> false)
        non_null_values
    in

    if all_bool then `Bool
    else if all_int then `Int32
    else if all_float then `Float32
    else `String

let json_to_float = function
  | Jsont.Null _ -> None
  | Jsont.Number (f, _) -> Some f
  | Jsont.String (s, _) -> (
      try Some (float_of_string s) with _ -> None)
  | _ -> None

let json_to_int32 = function
  | Jsont.Null _ -> None
  | Jsont.Number (f, _) -> Some (Int32.of_float f)
  | Jsont.String (s, _) -> (
      try Some (Int32.of_string s) with _ -> None)
  | _ -> None

let json_to_int64 = function
  | Jsont.Null _ -> None
  | Jsont.Number (f, _) -> Some (Int64.of_float f)
  | Jsont.String (s, _) -> (
      try Some (Int64.of_string s) with _ -> None)
  | _ -> None

let json_to_bool = function
  | Jsont.Null _ -> None
  | Jsont.Bool (b, _) -> Some b
  | _ -> None

let json_to_string_val = function
  | Jsont.Null _ -> None
  | Jsont.String (s, _) -> Some s
  | Jsont.Number (f, _) ->
      if Float.is_integer f then Some (string_of_int (int_of_float f))
      else Some (string_of_float f)
  | Jsont.Bool (b, _) -> Some (string_of_bool b)
  | _ -> None

let make_column dtype values =
  match dtype with
  | `Float32 ->
      Col.float32_opt (List.map json_to_float values |> Array.of_list)
  | `Float64 ->
      Col.float64_opt (List.map json_to_float values |> Array.of_list)
  | `Int32 ->
      Col.int32_opt (List.map json_to_int32 values |> Array.of_list)
  | `Int64 ->
      Col.int64_opt (List.map json_to_int64 values |> Array.of_list)
  | `Bool ->
      Col.bool_opt (List.map json_to_bool values |> Array.of_list)
  | _ ->
      (* String or mixed types *)
      Col.string_opt (List.map json_to_string_val values |> Array.of_list)

let from_string ?(orient = `Records) ?dtype_spec json_str =
  let json = json_of_string json_str in

  match orient with
  | `Records -> (
      (* Row-oriented: expect list of objects *)
      match json with
      | Jsont.Array (records, _) ->
          if List.length records = 0 then empty
          else
            (* Get column names from first record *)
            let col_names =
              match List.hd records with
              | Jsont.Object (mems, _) ->
                  List.map (fun ((n, _), _) -> n) mems
              | _ -> failwith "Invalid JSON: expected object in records array"
            in

            (* Extract values for each column *)
            let columns_data =
              List.map
                (fun col_name ->
                  let values =
                    List.map
                      (fun record ->
                        let fields = json_assoc record in
                        match List.assoc_opt col_name fields with
                        | Some v -> v
                        | None -> Jsont.Json.null ())
                      records
                  in
                  (col_name, values))
                col_names
            in

            (* Create columns based on detected types *)
            let columns =
              List.map
                (fun (col_name, values) ->
                  let dtype =
                    match dtype_spec with
                    | Some specs -> (
                        try List.assoc col_name specs
                        with Not_found -> detect_json_dtype values)
                    | None -> detect_json_dtype values
                  in
                  (col_name, make_column dtype values))
                columns_data
            in

            create columns
      | _ -> failwith "Invalid JSON: expected array for records orientation")
  | `Columns -> (
      (* Column-oriented: expect object with column arrays *)
      match json with
      | Jsont.Object (mems, _) ->
          let fields = List.map (fun ((n, _), v) -> (n, v)) mems in
          if List.length fields = 0 then empty
          else
            let columns =
              List.map
                (fun (col_name, values) ->
                  match values with
                  | Jsont.Array (vals, _) ->
                      let dtype =
                        match dtype_spec with
                        | Some specs -> (
                            try List.assoc col_name specs
                            with Not_found -> detect_json_dtype vals)
                        | None -> detect_json_dtype vals
                      in
                      (col_name, make_column dtype vals)
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

let from_file ?orient ?dtype_spec file =
  let ic = open_in file in
  let contents = really_input_string ic (in_channel_length ic) in
  close_in ic;
  from_string ?orient ?dtype_spec contents
