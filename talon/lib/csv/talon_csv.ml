(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type dtype_spec =
  (string * [ `Float32 | `Float64 | `Int32 | `Int64 | `Bool | `String ]) list

let default_na_values = [ ""; "NA"; "N/A"; "null"; "NULL"; "nan"; "NaN" ]
let is_null_value na_values s = List.mem s na_values

let detect_dtype na_values values =
  let non_null_values =
    List.filter (fun v -> not (is_null_value na_values v)) values
  in
  if List.length non_null_values = 0 then `String
  else
    let all_bool =
      List.for_all
        (fun v ->
          match String.lowercase_ascii v with
          | "true" | "t" | "yes" | "y" | "1" | "false" | "f" | "no" | "n"
          | "0" ->
              true
          | _ -> false)
        non_null_values
    in
    if all_bool then `Bool
    else
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
      if all_int then if needs_int64 then `Int64 else `Int32
      else
        let all_float =
          List.for_all
            (fun v ->
              try
                ignore (float_of_string v);
                true
              with _ -> false)
            non_null_values
        in
        if all_float then `Float32 else `String

let columns_of_rows na_values dtype_spec column_names data_rows =
  let num_cols = List.length column_names in
  let columns_data = Array.init num_cols (fun _ -> []) in
  List.iter
    (fun row ->
      List.iteri
        (fun i value ->
          if i < num_cols then columns_data.(i) <- value :: columns_data.(i))
        row)
    data_rows;
  Array.iteri (fun i lst -> columns_data.(i) <- List.rev lst) columns_data;
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
            Talon.Col.float32_opt arr
        | `Float64 ->
            let arr =
              List.map
                (fun v ->
                  if is_null_value na_values v then None
                  else try Some (float_of_string v) with _ -> None)
                values
              |> Array.of_list
            in
            Talon.Col.float64_opt arr
        | `Int32 ->
            let arr =
              List.map
                (fun v ->
                  if is_null_value na_values v then None
                  else try Some (Int32.of_string v) with _ -> None)
                values
              |> Array.of_list
            in
            Talon.Col.int32_opt arr
        | `Int64 ->
            let arr =
              List.map
                (fun v ->
                  if is_null_value na_values v then None
                  else try Some (Int64.of_string v) with _ -> None)
                values
              |> Array.of_list
            in
            Talon.Col.int64_opt arr
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
            Talon.Col.bool_opt arr
        | `String ->
            let arr =
              List.map
                (fun v ->
                  if is_null_value na_values v then None else Some v)
                values
              |> Array.of_list
            in
            Talon.Col.string_opt arr
      in
      (name, column))
    column_names

(* Returns a function [int -> string] for a column, extracting the underlying
   array once so that repeated row access is O(1) per cell. *)
let col_to_string_fn na_repr col =
  match col with
  | Talon.Col.P (dtype, tensor, mask) ->
      let is_null =
        match mask with Some m -> fun i -> m.(i) | None -> fun _ -> false
      in
      (match dtype with
      | Nx.Float32 ->
          let arr : float array = Nx.to_array tensor in
          fun i ->
            if is_null i then na_repr
            else
              let v = arr.(i) in
              if classify_float v = FP_nan then na_repr else string_of_float v
      | Nx.Float64 ->
          let arr : float array = Nx.to_array tensor in
          fun i ->
            if is_null i then na_repr
            else
              let v = arr.(i) in
              if classify_float v = FP_nan then na_repr else string_of_float v
      | Nx.Float16 ->
          let arr : float array = Nx.to_array tensor in
          fun i ->
            if is_null i then na_repr
            else
              let v = arr.(i) in
              if classify_float v = FP_nan then na_repr else string_of_float v
      | Nx.BFloat16 ->
          let arr : float array = Nx.to_array tensor in
          fun i ->
            if is_null i then na_repr
            else
              let v = arr.(i) in
              if classify_float v = FP_nan then na_repr else string_of_float v
      | Nx.Float8_e4m3 ->
          let arr : float array = Nx.to_array tensor in
          fun i ->
            if is_null i then na_repr
            else
              let v = arr.(i) in
              if classify_float v = FP_nan then na_repr else string_of_float v
      | Nx.Float8_e5m2 ->
          let arr : float array = Nx.to_array tensor in
          fun i ->
            if is_null i then na_repr
            else
              let v = arr.(i) in
              if classify_float v = FP_nan then na_repr else string_of_float v
      | Nx.Int8 ->
          let arr : int array = Nx.to_array tensor in
          fun i -> if is_null i then na_repr else string_of_int arr.(i)
      | Nx.UInt8 ->
          let arr : int array = Nx.to_array tensor in
          fun i -> if is_null i then na_repr else string_of_int arr.(i)
      | Nx.Int16 ->
          let arr : int array = Nx.to_array tensor in
          fun i -> if is_null i then na_repr else string_of_int arr.(i)
      | Nx.UInt16 ->
          let arr : int array = Nx.to_array tensor in
          fun i -> if is_null i then na_repr else string_of_int arr.(i)
      | Nx.Int32 ->
          let arr : int32 array = Nx.to_array tensor in
          fun i -> if is_null i then na_repr else Int32.to_string arr.(i)
      | Nx.Int64 ->
          let arr : int64 array = Nx.to_array tensor in
          fun i -> if is_null i then na_repr else Int64.to_string arr.(i)
      | Nx.UInt32 ->
          let arr : int32 array = Nx.to_array tensor in
          fun i -> if is_null i then na_repr else Int32.to_string arr.(i)
      | Nx.UInt64 ->
          let arr : int64 array = Nx.to_array tensor in
          fun i -> if is_null i then na_repr else Int64.to_string arr.(i)
      | Nx.Complex64 ->
          let arr : Complex.t array = Nx.to_array tensor in
          fun i ->
            if is_null i then na_repr
            else
              let c = arr.(i) in
              Printf.sprintf "%g+%gi" c.re c.im
      | Nx.Complex128 ->
          let arr : Complex.t array = Nx.to_array tensor in
          fun i ->
            if is_null i then na_repr
            else
              let c = arr.(i) in
              Printf.sprintf "%g+%gi" c.re c.im
      | Nx.Bool ->
          let arr : bool array = Nx.to_array tensor in
          fun i -> if is_null i then na_repr else string_of_bool arr.(i)
      | Nx.Int4 ->
          let arr : int array = Nx.to_array tensor in
          fun i -> if is_null i then na_repr else string_of_int arr.(i)
      | Nx.UInt4 ->
          let arr : int array = Nx.to_array tensor in
          fun i -> if is_null i then na_repr else string_of_int arr.(i))
  | Talon.Col.S arr ->
      fun i -> (match arr.(i) with Some s -> s | None -> na_repr)
  | Talon.Col.B arr ->
      fun i -> (match arr.(i) with Some b -> string_of_bool b | None -> na_repr)

let col_string_fns na_repr df =
  List.map
    (fun name -> col_to_string_fn na_repr (Talon.get_column_exn df name))
    (Talon.column_names df)

let df_of_rows ?names ?(na_values = default_na_values) ?dtype_spec rows =
  match names with
  | Some column_names -> (
      match rows with
      | [] ->
          let columns =
            List.map (fun name -> (name, Talon.Col.string [||])) column_names
          in
          Talon.create columns
      | _ ->
          columns_of_rows na_values dtype_spec column_names rows |> Talon.create)
  | None -> (
      match rows with
      | [] -> Talon.empty
      | [ header ] ->
          let columns =
            List.map (fun name -> (name, Talon.Col.string [||])) header
          in
          Talon.create columns
      | header :: data ->
          columns_of_rows na_values dtype_spec header data |> Talon.create)

let of_string ?(sep = ',') ?names ?na_values ?dtype_spec s =
  df_of_rows ?names ?na_values ?dtype_spec (Csv_io.parse ~separator:sep s)

let to_string ?(sep = ',') ?(na_repr = "") df =
  let buf = Buffer.create 1024 in
  let fns = col_string_fns na_repr df in
  let n_rows = Talon.num_rows df in
  Csv_io.write_row buf sep (Talon.column_names df);
  for i = 0 to n_rows - 1 do
    Csv_io.write_row buf sep (List.map (fun f -> f i) fns)
  done;
  Buffer.contents buf

let read ?(sep = ',') ?names ?na_values ?dtype_spec path =
  In_channel.with_open_text path @@ fun ic ->
  let rows = ref [] in
  (try
     while true do
       let line = Csv_io.strip_cr (input_line ic) in
       if line <> "" then rows := Csv_io.parse_row sep line :: !rows
     done
   with End_of_file -> ());
  df_of_rows ?names ?na_values ?dtype_spec (List.rev !rows)

let write ?(sep = ',') ?(na_repr = "") path df =
  Out_channel.with_open_text path @@ fun oc ->
  let buf = Buffer.create 256 in
  let fns = col_string_fns na_repr df in
  let n_rows = Talon.num_rows df in
  Csv_io.write_row buf sep (Talon.column_names df);
  output_string oc (Buffer.contents buf);
  for i = 0 to n_rows - 1 do
    Buffer.clear buf;
    Csv_io.write_row buf sep (List.map (fun f -> f i) fns);
    output_string oc (Buffer.contents buf)
  done
