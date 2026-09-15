(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type error = Error.t =
  | Io_error of string
  | Format_error of string
  | Unsupported_dtype
  | Unsupported_shape
  | Missing_entry of string
  | Other of string

let strf = Printf.sprintf

(* Errors *)

let err_invalid_literal dtype token =
  Format_error (strf "invalid %s literal: %S" dtype (String.trim token))

let err_out_of_range dtype token =
  Format_error
    (strf "value %S is out of range for %s" (String.trim token) dtype)

let err_skiprows_negative = Format_error "skiprows must be non-negative"
let err_max_rows_nonpos = Format_error "max_rows must be strictly positive"
let err_no_data = Format_error "no data found"
let err_inconsistent_cols = Format_error "inconsistent number of columns"

(* Parsing helpers *)

let try_parse f s = try Some (f s) with _ -> None
let float_of_string_opt = try_parse float_of_string
let int_of_string_opt = try_parse int_of_string
let int32_of_string_opt = try_parse Int32.of_string
let int64_of_string_opt = try_parse Int64.of_string

let parse_float dtype token =
  match float_of_string_opt token with
  | Some v -> Ok v
  | None -> Error (err_invalid_literal dtype token)

let parse_bool token =
  let s = String.lowercase_ascii (String.trim token) in
  match s with
  | "true" | "t" | "yes" | "y" -> Ok true
  | "false" | "f" | "no" | "n" -> Ok false
  | _ -> (
      match int_of_string_opt s with
      | Some 0 -> Ok false
      | Some _ -> Ok true
      | None -> (
          match float_of_string_opt s with
          | Some f -> Ok (abs_float f > 0.0)
          | None -> Error (err_invalid_literal "bool" token)))

let parse_int_with_bounds dtype token ~min ~max =
  match int_of_string_opt token with
  | Some v when v >= min && v <= max -> Ok v
  | Some _ -> Error (err_out_of_range dtype token)
  | None -> Error (err_invalid_literal dtype token)

(*---------------------------------------------------------------------------
  Dtype-specific spec
  ---------------------------------------------------------------------------*)

module type SPEC = sig
  type elt
  type kind

  val kind : (elt, kind) Nx_buffer.kind
  val print : out_channel -> elt -> unit
  val parse : string -> (elt, error) result
end

(* Exact float formatting

   A C runtime only has to round [%e] correctly up to 17 significant digits;
   numpy's default of 19 needs the exact binary value, so the digits come from
   the significand and exponent here and the output is the same on every
   platform. Little-endian base-1e9 limbs hold the exact integer. *)

let limb_base = 1_000_000_000

let mul_small limbs n =
  let carry = ref 0 in
  let limbs =
    Array.map
      (fun l ->
        let v = (l * n) + !carry in
        carry := v / limb_base;
        v mod limb_base)
      limbs
  in
  if !carry > 0 then Array.append limbs [| !carry |] else limbs

let digits_of_limbs limbs =
  let n = Array.length limbs in
  let b = Buffer.create (9 * n) in
  Buffer.add_string b (string_of_int limbs.(n - 1));
  for i = n - 2 downto 0 do
    Buffer.add_string b (Printf.sprintf "%09d" limbs.(i))
  done;
  Buffer.contents b

(* [exact_decimal x] is the decimal digits [d] of the finite, nonzero [|x|] and
   the count [k] of those digits after the decimal point: [|x| = d * 10^-k]. [x
   = f * 2^e] for the 53-bit integer [f]; a negative [e] turns [2^e] into [5^-e
   / 10^-e]. *)
let exact_decimal x =
  let m, e = Float.frexp x in
  let f = Int64.of_float (Float.ldexp (Float.abs m) 53) in
  let e = e - 53 in
  let lo = Int64.to_int (Int64.rem f 1_000_000_000L) in
  let hi = Int64.to_int (Int64.div f 1_000_000_000L) in
  let limbs = ref (if hi = 0 then [| lo |] else [| lo; hi |]) in
  if e >= 0 then begin
    for _ = 1 to e do
      limbs := mul_small !limbs 2
    done;
    (digits_of_limbs !limbs, 0)
  end
  else begin
    for _ = 1 to -e do
      limbs := mul_small !limbs 5
    done;
    (digits_of_limbs !limbs, -e)
  end

(* [%.<precision>e] of [x], rounded to nearest with ties to even, with at least
   two exponent digits. NaN prints as [nan] whatever its sign. *)
let format_e ~precision x =
  let sign = if Float.sign_bit x then "-" else "" in
  if Float.is_nan x then "nan"
  else if Float.abs x = Float.infinity then sign ^ "inf"
  else if x = 0. then sign ^ "0." ^ String.make precision '0' ^ "e+00"
  else
    let digits, point = exact_decimal x in
    let len = String.length digits in
    let keep = precision + 1 in
    let kept, carried =
      if len <= keep then (digits ^ String.make (keep - len) '0', false)
      else
        let head = Bytes.of_string (String.sub digits 0 keep) in
        let next = digits.[keep] in
        let rest_nonzero = ref false in
        for i = keep + 1 to len - 1 do
          if digits.[i] <> '0' then rest_nonzero := true
        done;
        let last_odd =
          (Char.code (Bytes.get head (keep - 1)) - 48) land 1 = 1
        in
        if next < '5' || (next = '5' && not (!rest_nonzero || last_odd)) then
          (Bytes.to_string head, false)
        else
          let i = ref (keep - 1) in
          while !i >= 0 && Bytes.get head !i = '9' do
            Bytes.set head !i '0';
            decr i
          done;
          if !i < 0 then ("1" ^ Bytes.sub_string head 0 (keep - 1), true)
          else begin
            Bytes.set head !i (Char.chr (Char.code (Bytes.get head !i) + 1));
            (Bytes.to_string head, false)
          end
    in
    let exponent = len - point - 1 + if carried then 1 else 0 in
    Printf.sprintf "%s%c.%se%c%02d" sign kept.[0]
      (String.sub kept 1 precision)
      (if exponent < 0 then '-' else '+')
      (abs exponent)

let print_float oc v = output_string oc (format_e ~precision:18 v)
let print_int oc v = output_string oc (string_of_int v)
let print_int32 oc v = output_string oc (Int32.to_string v)
let print_int64 oc v = output_string oc (Int64.to_string v)
let print_bool oc v = output_string oc (if v then "1" else "0")

let parse_i32 name token =
  match int32_of_string_opt token with
  | Some v -> Ok v
  | None -> Error (err_invalid_literal name token)

let parse_i64 name token =
  match int64_of_string_opt token with
  | Some v -> Ok v
  | None -> Error (err_invalid_literal name token)

(* Each GADT arm must be inlined so the type equalities are visible. *)

let spec_of_dtype (type a b) (dtype : (a, b) Nx.dtype) :
    (module SPEC with type elt = a and type kind = b) option =
  let name = Nx_core.Dtype.to_string dtype in
  let kind = dtype in
  let open Nx_core.Dtype in
  match dtype with
  | Float16 ->
      Some
        (module struct
          type elt = float
          type kind = b

          let kind = kind
          let print = print_float
          let parse t = parse_float name t
        end)
  | Float32 ->
      Some
        (module struct
          type elt = float
          type kind = b

          let kind = kind
          let print = print_float
          let parse t = parse_float name t
        end)
  | Float64 ->
      Some
        (module struct
          type elt = float
          type kind = b

          let kind = kind
          let print = print_float
          let parse t = parse_float name t
        end)
  | BFloat16 ->
      Some
        (module struct
          type elt = float
          type kind = b

          let kind = kind
          let print = print_float
          let parse t = parse_float name t
        end)
  | Int8 ->
      Some
        (module struct
          type elt = int
          type kind = b

          let kind = kind
          let print = print_int
          let parse t = parse_int_with_bounds name t ~min:(-128) ~max:127
        end)
  | UInt8 ->
      Some
        (module struct
          type elt = int
          type kind = b

          let kind = kind
          let print = print_int
          let parse t = parse_int_with_bounds name t ~min:0 ~max:255
        end)
  | Int16 ->
      Some
        (module struct
          type elt = int
          type kind = b

          let kind = kind
          let print = print_int
          let parse t = parse_int_with_bounds name t ~min:(-32768) ~max:32767
        end)
  | UInt16 ->
      Some
        (module struct
          type elt = int
          type kind = b

          let kind = kind
          let print = print_int
          let parse t = parse_int_with_bounds name t ~min:0 ~max:65535
        end)
  | Int32 ->
      Some
        (module struct
          type elt = int32
          type kind = b

          let kind = kind
          let print = print_int32
          let parse t = parse_i32 name t
        end)
  | UInt32 ->
      Some
        (module struct
          type elt = int32
          type kind = b

          let kind = kind
          let print = print_int32
          let parse t = parse_i32 name t
        end)
  | Int64 ->
      Some
        (module struct
          type elt = int64
          type kind = b

          let kind = kind
          let print = print_int64
          let parse t = parse_i64 name t
        end)
  | UInt64 ->
      Some
        (module struct
          type elt = int64
          type kind = b

          let kind = kind
          let print = print_int64
          let parse t = parse_i64 name t
        end)
  | Bool ->
      Some
        (module struct
          type elt = bool
          type kind = b

          let kind = kind
          let print = print_bool
          let parse = parse_bool
        end)
  | _ -> None

(*---------------------------------------------------------------------------
  Text field splitting
  ---------------------------------------------------------------------------*)

let split_fields sep line =
  let s = String.trim line in
  if s = "" then [||]
  else if sep = "" then [| s |]
  else if String.length sep = 1 then
    s
    |> String.split_on_char sep.[0]
    |> List.filter (fun t -> t <> "")
    |> Array.of_list
  else
    let sep_len = String.length sep in
    let len = String.length s in
    let rec loop acc start =
      if start >= len then List.rev acc
      else
        match String.index_from_opt s start sep.[0] with
        | None ->
            let part = String.sub s start (len - start) in
            if part = "" then List.rev acc else List.rev (part :: acc)
        | Some idx ->
            if idx + sep_len <= len && String.sub s idx sep_len = sep then
              let part = String.sub s start (idx - start) in
              let acc = if part = "" then acc else part :: acc in
              loop acc (idx + sep_len)
            else loop acc (idx + 1)
    in
    loop [] 0 |> Array.of_list

(*---------------------------------------------------------------------------
  Save
  ---------------------------------------------------------------------------*)

let write_comment_lines oc comments newline = function
  | None -> ()
  | Some text ->
      List.iter
        (fun line ->
          if comments <> "" then output_string oc comments;
          output_string oc line;
          output_string oc newline)
        (String.split_on_char '\n' text)

let save ?(sep = " ") ?(append = false) ?(newline = "\n") ?header ?footer
    ?(comments = "# ") ~out (type a b) (arr : (a, b) Nx.t) =
  let shape = Nx.shape arr in
  let ndim = Array.length shape in
  if ndim > 2 then Error Unsupported_shape
  else
    match spec_of_dtype (Nx.dtype arr) with
    | None -> Error Unsupported_dtype
    | Some (module S : SPEC with type elt = a and type kind = b) -> (
        let perm = 0o666 in
        let flags =
          if append then [ Open_wronly; Open_creat; Open_append; Open_binary ]
          else [ Open_wronly; Open_creat; Open_trunc; Open_binary ]
        in
        try
          let oc = open_out_gen flags perm out in
          Fun.protect ~finally:(fun () -> close_out oc) @@ fun () ->
          write_comment_lines oc comments newline header;
          let buf = Nx.to_buffer arr in
          (match ndim with
          | 0 ->
              S.print oc (Nx_buffer.get buf 0);
              output_string oc newline
          | 1 ->
              let n = shape.(0) in
              for j = 0 to n - 1 do
                if j > 0 then output_string oc sep;
                S.print oc (Nx_buffer.unsafe_get buf j)
              done;
              output_string oc newline
          | _ ->
              let rows = shape.(0) and cols = shape.(1) in
              for i = 0 to rows - 1 do
                for j = 0 to cols - 1 do
                  if j > 0 then output_string oc sep;
                  S.print oc (Nx_buffer.unsafe_get buf ((i * cols) + j))
                done;
                output_string oc newline
              done);
          write_comment_lines oc comments newline footer;
          Ok ()
        with
        | Sys_error msg -> Error (Io_error msg)
        | Unix.Unix_error (e, _, _) -> Error (Io_error (Unix.error_message e)))

(*---------------------------------------------------------------------------
  Load
  ---------------------------------------------------------------------------*)

exception Parse_error of error

let load ?(sep = " ") ?(comments = "#") ?(skiprows = 0) ?max_rows (type a b)
    (dtype : (a, b) Nx.dtype) path =
  if skiprows < 0 then Error err_skiprows_negative
  else if match max_rows with Some n -> n <= 0 | None -> false then
    Error err_max_rows_nonpos
  else
    match spec_of_dtype dtype with
    | None -> Error Unsupported_dtype
    | Some (module S : SPEC with type elt = a and type kind = b) -> (
        try
          let ic = open_in path in
          Fun.protect ~finally:(fun () -> close_in ic) @@ fun () ->
          let comment_prefix = String.trim comments in
          let is_comment line =
            if comment_prefix = "" then false
            else
              let t = String.trim line in
              let n = String.length comment_prefix in
              String.length t >= n && String.sub t 0 n = comment_prefix
          in
          (* Read data rows *)
          let rows_rev = ref [] in
          let col_count = ref (-1) in
          let rows_read = ref 0 in
          let rec read skip =
            if match max_rows with Some n -> !rows_read >= n | None -> false
            then ()
            else
              match input_line ic with
              | exception End_of_file -> ()
              | line ->
                  if skip > 0 then read (skip - 1)
                  else if is_comment line then read 0
                  else
                    let fields = split_fields sep line in
                    let n = Array.length fields in
                    if n = 0 then read 0
                    else begin
                      if !col_count < 0 then col_count := n
                      else if n <> !col_count then
                        raise_notrace (Parse_error err_inconsistent_cols);
                      rows_rev := fields :: !rows_rev;
                      incr rows_read;
                      read 0
                    end
          in
          read skiprows;
          if !col_count < 0 then Error err_no_data
          else
            let cols = !col_count in
            let rows = Array.of_list (List.rev !rows_rev) in
            let row_count = Array.length rows in
            let n = row_count * cols in
            let buf = Nx_buffer.create S.kind n in
            for i = 0 to row_count - 1 do
              let row = rows.(i) in
              for j = 0 to cols - 1 do
                match S.parse row.(j) with
                | Ok v -> Nx_buffer.set buf ((i * cols) + j) v
                | Error err -> raise_notrace (Parse_error err)
              done
            done;
            let t = Nx.of_buffer buf ~shape:[| row_count; cols |] in
            let result =
              if row_count = 1 then Nx.reshape [| cols |] t
              else if cols = 1 then Nx.reshape [| row_count |] t
              else t
            in
            Ok result
        with
        | Parse_error err -> Error err
        | Sys_error msg -> Error (Io_error msg)
        | Unix.Unix_error (e, _, _) -> Error (Io_error (Unix.error_message e)))
