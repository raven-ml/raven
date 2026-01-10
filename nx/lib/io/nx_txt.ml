(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_buffer

type error = Error.t =
  | Io_error of string
  | Format_error of string
  | Unsupported_dtype
  | Unsupported_shape
  | Missing_entry of string
  | Other of string

type layout = Scalar | Vector of int | Matrix of int * int

let layout_of_shape shape =
  match shape with
  | [||] -> Some Scalar
  | [| n |] -> Some (Vector n)
  | [| rows; cols |] -> Some (Matrix (rows, cols))
  | _ -> None

let option_exists pred = function Some x -> pred x | None -> false

let split_lines_opt = function
  | None -> []
  | Some text -> String.split_on_char '\n' text

let rec trim_trailing_whitespace s =
  if s = "" then ""
  else
    let len = String.length s in
    match s.[len - 1] with
    | ' ' | '\t' -> trim_trailing_whitespace (String.sub s 0 (len - 1))
    | _ -> s

let try_parse f s = try Some (f s) with _ -> None
let float_of_string_opt = try_parse float_of_string
let int_of_string_opt = try_parse int_of_string
let int32_of_string_opt = try_parse Int32.of_string
let int64_of_string_opt = try_parse Int64.of_string
let nativeint_of_string_opt = try_parse Nativeint.of_string

module type SPEC = sig
  type elt
  type kind

  val kind : (elt, kind) Nx_buffer.kind
  val print : out_channel -> elt -> unit
  val parse : string -> (elt, error) result
end

let invalid_literal dtype_name token =
  Format_error
    (Printf.sprintf "Invalid %s literal: %S" dtype_name
       (trim_trailing_whitespace token))

let out_of_range dtype_name token =
  Format_error
    (Printf.sprintf "Value %S is out of range for %s"
       (trim_trailing_whitespace token)
       dtype_name)

let parse_float dtype token =
  match float_of_string_opt token with
  | Some v -> Ok v
  | None -> Error (invalid_literal dtype token)

let parse_bool token =
  let lowered = String.lowercase_ascii (String.trim token) in
  match lowered with
  | "true" | "t" | "yes" | "y" -> Ok true
  | "false" | "f" | "no" | "n" -> Ok false
  | _ -> (
      match int_of_string_opt lowered with
      | Some 0 -> Ok false
      | Some _ -> Ok true
      | None -> (
          match float_of_string_opt lowered with
          | Some f -> Ok (abs_float f > 0.0)
          | None -> Error (invalid_literal "bool" token)))

let parse_int_with_bounds dtype token ~min ~max =
  match int_of_string_opt token with
  | Some v when v >= min && v <= max -> Ok v
  | Some _ -> Error (out_of_range dtype token)
  | None -> Error (invalid_literal dtype token)

let spec_of_dtype (type a) (type b) (dtype : (a, b) Nx.dtype) :
    (module SPEC with type elt = a and type kind = b) option =
  let dtype_name = Nx_core.Dtype.to_string dtype in
  let module M (X : sig
    type elt
    type kind

    val kind : (elt, kind) Nx_buffer.kind
    val print : out_channel -> elt -> unit
    val parse : string -> (elt, error) result
  end) =
  struct
    include X
  end in
  let open Nx_core.Dtype in
  match dtype with
  | Float16 ->
      let module S = M (struct
        type elt = float
        type kind = Nx_buffer.float16_elt

        let kind = Nx_core.Dtype.to_buffer_kind dtype
        let print oc v = Printf.fprintf oc "%.18e" v
        let parse token = parse_float dtype_name token
      end) in
      Some (module S : SPEC with type elt = a and type kind = b)
  | Float32 ->
      let module S = M (struct
        type elt = float
        type kind = Nx_buffer.float32_elt

        let kind = Nx_core.Dtype.to_buffer_kind dtype
        let print oc v = Printf.fprintf oc "%.18e" v
        let parse token = parse_float dtype_name token
      end) in
      Some (module S : SPEC with type elt = a and type kind = b)
  | Float64 ->
      let module S = M (struct
        type elt = float
        type kind = Nx_buffer.float64_elt

        let kind = Nx_core.Dtype.to_buffer_kind dtype
        let print oc v = Printf.fprintf oc "%.18e" v
        let parse token = parse_float dtype_name token
      end) in
      Some (module S : SPEC with type elt = a and type kind = b)
  | BFloat16 ->
      let module S = M (struct
        type elt = float
        type kind = Nx_buffer.bfloat16_elt

        let kind = Nx_core.Dtype.to_buffer_kind dtype
        let print oc v = Printf.fprintf oc "%.18e" v
        let parse token = parse_float dtype_name token
      end) in
      Some (module S : SPEC with type elt = a and type kind = b)
  | Int8 ->
      let module S = M (struct
        type elt = int
        type kind = Nx_buffer.int8_signed_elt

        let kind = Nx_core.Dtype.to_buffer_kind dtype
        let print oc v = output_string oc (string_of_int v)

        let parse token =
          parse_int_with_bounds dtype_name token ~min:(-128) ~max:127
      end) in
      Some (module S : SPEC with type elt = a and type kind = b)
  | UInt8 ->
      let module S = M (struct
        type elt = int
        type kind = Nx_buffer.int8_unsigned_elt

        let kind = Nx_core.Dtype.to_buffer_kind dtype
        let print oc v = output_string oc (string_of_int v)
        let parse token = parse_int_with_bounds dtype_name token ~min:0 ~max:255
      end) in
      Some (module S : SPEC with type elt = a and type kind = b)
  | Int16 ->
      let module S = M (struct
        type elt = int
        type kind = Nx_buffer.int16_signed_elt

        let kind = Nx_core.Dtype.to_buffer_kind dtype
        let print oc v = output_string oc (string_of_int v)

        let parse token =
          parse_int_with_bounds dtype_name token ~min:(-32768) ~max:32767
      end) in
      Some (module S : SPEC with type elt = a and type kind = b)
  | UInt16 ->
      let module S = M (struct
        type elt = int
        type kind = Nx_buffer.int16_unsigned_elt

        let kind = Nx_core.Dtype.to_buffer_kind dtype
        let print oc v = output_string oc (string_of_int v)

        let parse token =
          parse_int_with_bounds dtype_name token ~min:0 ~max:65535
      end) in
      Some (module S : SPEC with type elt = a and type kind = b)
  | Int32 ->
      let module S = M (struct
        type elt = int32
        type kind = Nx_buffer.int32_elt

        let kind = Nx_core.Dtype.to_buffer_kind dtype
        let print oc v = output_string oc (Int32.to_string v)

        let parse token =
          match int32_of_string_opt token with
          | Some v -> Ok v
          | None -> Error (invalid_literal dtype_name token)
      end) in
      Some (module S : SPEC with type elt = a and type kind = b)
  | Int64 ->
      let module S = M (struct
        type elt = int64
        type kind = Nx_buffer.int64_elt

        let kind = Nx_core.Dtype.to_buffer_kind dtype
        let print oc v = output_string oc (Int64.to_string v)

        let parse token =
          match int64_of_string_opt token with
          | Some v -> Ok v
          | None -> Error (invalid_literal dtype_name token)
      end) in
      Some (module S : SPEC with type elt = a and type kind = b)
  | UInt32 ->
      let module S = M (struct
        type elt = int32
        type kind = Nx_buffer.uint32_elt

        let kind = Nx_core.Dtype.to_buffer_kind dtype
        let print oc v = output_string oc (Int32.to_string v)

        let parse token =
          match int32_of_string_opt token with
          | Some v -> Ok v
          | None -> Error (invalid_literal dtype_name token)
      end) in
      Some (module S : SPEC with type elt = a and type kind = b)
  | UInt64 ->
      let module S = M (struct
        type elt = int64
        type kind = Nx_buffer.uint64_elt

        let kind = Nx_core.Dtype.to_buffer_kind dtype
        let print oc v = output_string oc (Int64.to_string v)

        let parse token =
          match int64_of_string_opt token with
          | Some v -> Ok v
          | None -> Error (invalid_literal dtype_name token)
      end) in
      Some (module S : SPEC with type elt = a and type kind = b)
  | Bool ->
      let module S = M (struct
        type elt = bool
        type kind = Nx_buffer.bool_elt

        let kind = Nx_core.Dtype.to_buffer_kind dtype
        let print oc v = output_string oc (if v then "1" else "0")
        let parse = parse_bool
      end) in
      Some (module S : SPEC with type elt = a and type kind = b)
  | _ -> None

let save ?(sep = " ") ?(append = false) ?(newline = "\n") ?header ?footer
    ?(comments = "# ") ~out (type a) (type b) (arr : (a, b) Nx.t) =
  match layout_of_shape (Nx.shape arr) with
  | None -> Error Unsupported_shape
  | Some layout -> (
      match spec_of_dtype (Nx.dtype arr) with
      | None -> Error Unsupported_dtype
      | Some spec_module -> (
          let module S =
            (val spec_module : SPEC with type elt = a and type kind = b)
          in
          let perm = 0o666 in
          let flags =
            if append then [ Open_wronly; Open_creat; Open_append; Open_text ]
            else [ Open_wronly; Open_creat; Open_trunc; Open_text ]
          in
          try
            let oc = open_out_gen flags perm out in
            Fun.protect
              ~finally:(fun () -> close_out oc)
              (fun () ->
                let write_prefixed line =
                  if comments <> "" then output_string oc comments;
                  output_string oc line;
                  output_string oc newline
                in
                List.iter write_prefixed (split_lines_opt header);
                let data =
                  (Nx.to_buffer arr
                    : (S.elt, S.kind, Bigarray.c_layout) Genarray.t)
                in
                (match layout with
                | Scalar ->
                    let value = Genarray.get data [||] in
                    S.print oc value;
                    output_string oc newline
                | Vector length ->
                    let view = array1_of_genarray data in
                    for j = 0 to length - 1 do
                      if j > 0 then output_string oc sep;
                      S.print oc (Array1.unsafe_get view j)
                    done;
                    output_string oc newline
                | Matrix (rows, cols) ->
                    let view = array2_of_genarray data in
                    for i = 0 to rows - 1 do
                      for j = 0 to cols - 1 do
                        if j > 0 then output_string oc sep;
                        S.print oc (Array2.unsafe_get view i j)
                      done;
                      output_string oc newline
                    done);
                List.iter write_prefixed (split_lines_opt footer);
                Ok ())
          with
          | Sys_error msg -> Error (Io_error msg)
          | Unix.Unix_error (e, _, _) -> Error (Io_error (Unix.error_message e))
          ))

let load ?(sep = " ") ?(comments = "#") ?(skiprows = 0) ?max_rows (type a)
    (type b) (dtype : (a, b) Nx.dtype) path =
  if skiprows < 0 then Error (Format_error "skiprows must be non-negative")
  else if option_exists (fun rows -> rows <= 0) max_rows then
    Error (Format_error "max_rows must be strictly positive")
  else
    match spec_of_dtype dtype with
    | None -> Error Unsupported_dtype
    | Some spec_module -> (
        let module S =
          (val spec_module : SPEC with type elt = a and type kind = b)
        in
        try
          let ic = open_in path in
          Fun.protect
            ~finally:(fun () -> close_in ic)
            (fun () ->
              let comment_prefix = String.trim comments in
              let is_comment_line line =
                if comment_prefix = "" then false
                else
                  let trimmed = String.trim line in
                  let len = String.length comment_prefix in
                  String.length trimmed >= len
                  && String.sub trimmed 0 len = comment_prefix
              in
              let split_fields line =
                let trimmed = String.trim line in
                if trimmed = "" then [||]
                else if sep = "" then [| trimmed |]
                else if String.length sep = 1 then
                  trimmed
                  |> String.split_on_char sep.[0]
                  |> List.filter (fun s -> s <> "")
                  |> Array.of_list
                else
                  let len_sep = String.length sep in
                  let len = String.length trimmed in
                  let rec aux acc start =
                    if start >= len then List.rev acc
                    else
                      match String.index_from_opt trimmed start sep.[0] with
                      | None ->
                          let part = String.sub trimmed start (len - start) in
                          if part = "" then List.rev acc
                          else List.rev (part :: acc)
                      | Some idx ->
                          if
                            idx + len_sep <= len
                            && String.sub trimmed idx len_sep = sep
                          then
                            let part = String.sub trimmed start (idx - start) in
                            let acc = if part = "" then acc else part :: acc in
                            aux acc (idx + len_sep)
                          else aux acc (idx + 1)
                  in
                  aux [] 0 |> Array.of_list
              in
              let rows_rev = ref [] in
              let cols = ref None in
              let rows_read = ref 0 in
              let read_error = ref None in
              let rec loop skip_remaining =
                if Option.is_some !read_error then ()
                else if option_exists (fun rows -> !rows_read >= rows) max_rows
                then ()
                else
                  match input_line ic with
                  | line ->
                      if skip_remaining > 0 then loop (skip_remaining - 1)
                      else if is_comment_line line then loop 0
                      else
                        let fields = split_fields line in
                        if Array.length fields = 0 then loop 0
                        else (
                          (match !cols with
                          | None -> cols := Some (Array.length fields)
                          | Some expected ->
                              if Array.length fields <> expected then
                                read_error :=
                                  Some
                                    (Format_error
                                       "Inconsistent number of columns"));
                          if Option.is_none !read_error then (
                            rows_rev := fields :: !rows_rev;
                            incr rows_read);
                          loop 0)
                  | exception End_of_file -> ()
              in
              loop skiprows;
              let parsed_result =
                match (!read_error, !cols, !rows_rev) with
                | Some err, _, _ -> Error err
                | _, None, _ -> Error (Format_error "No data found")
                | _, _, [] -> Error (Format_error "No data found")
                | _, Some col_count, rows_rev_list -> (
                    let rows = List.rev rows_rev_list |> Array.of_list in
                    let row_count = Array.length rows in
                    let dims = [| row_count; col_count |] in
                    let ba = Genarray.create S.kind Bigarray.c_layout dims in
                    let parse_error = ref None in
                    for i = 0 to row_count - 1 do
                      let row = rows.(i) in
                      for j = 0 to col_count - 1 do
                        if Option.is_none !parse_error then
                          match S.parse row.(j) with
                          | Ok value -> Genarray.set ba [| i; j |] value
                          | Error err -> parse_error := Some err
                      done
                    done;
                    match !parse_error with
                    | Some err -> Error err
                    | None ->
                        let tensor = Nx.of_buffer ba in
                        let result =
                          if row_count = 1 then
                            Nx.reshape [| col_count |] tensor
                          else if col_count = 1 then
                            Nx.reshape [| row_count |] tensor
                          else tensor
                        in
                        Ok result)
              in
              parsed_result)
        with
        | Sys_error msg -> Error (Io_error msg)
        | Unix.Unix_error (e, _, _) -> Error (Io_error (Unix.error_message e)))
