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
          | "true" | "t" | "yes" | "y" | "1" | "false" | "f" | "no" | "n" | "0"
            ->
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
      let parse_col values ~parse ~make =
        let arr =
          List.map
            (fun v ->
              if is_null_value na_values v then None
              else try Some (parse v) with _ -> None)
            values
          |> Array.of_list
        in
        make arr
      in
      let column =
        match dtype with
        | `Float32 ->
            parse_col values ~parse:float_of_string ~make:Talon.Col.float32_opt
        | `Float64 ->
            parse_col values ~parse:float_of_string ~make:Talon.Col.float64_opt
        | `Int32 ->
            parse_col values ~parse:Int32.of_string ~make:Talon.Col.int32_opt
        | `Int64 ->
            parse_col values ~parse:Int64.of_string ~make:Talon.Col.int64_opt
        | `Bool ->
            parse_col values ~make:Talon.Col.bool_opt ~parse:(fun v ->
                match String.lowercase_ascii v with
                | "true" | "t" | "yes" | "y" | "1" -> true
                | "false" | "f" | "no" | "n" | "0" -> false
                | _ -> raise Exit)
        | `String -> parse_col values ~parse:Fun.id ~make:Talon.Col.string_opt
      in
      (name, column))
    column_names

let col_string_fns na_repr df =
  List.map
    (fun name ->
      Talon.Col.to_string_fn ~null:na_repr (Talon.get_column_exn df name))
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
          columns_of_rows na_values dtype_spec column_names rows |> Talon.create
      )
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
