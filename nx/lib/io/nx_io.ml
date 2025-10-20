open Bigarray_ext
open Error

(* Type definitions *)

type packed_nx = Packed_nx.t = P : ('a, 'b) Nx.t -> packed_nx
type archive = (string, packed_nx) Hashtbl.t

module Safe = struct
  type error = Error.t =
    | Io_error of string
    | Format_error of string
    | Unsupported_dtype
    | Unsupported_shape
    | Missing_entry of string
    | Other of string

  (* Image dimensions *)

  type nx_dims = [ `Gray of int * int | `Color of int * int * int ]

  let get_nx_dims arr : nx_dims =
    match Nx.shape arr with
    | [| h; w |] -> `Gray (h, w)
    | [| h; w; c |] -> `Color (h, w, c)
    | s ->
        fail_msg "Invalid nx dimensions: expected 2 or 3, got %d (%s)"
          (Array.length s)
          (Array.to_list s |> List.map string_of_int |> String.concat "x")

  let load_image ?grayscale path =
    let grayscale = Option.value grayscale ~default:false in
    try
      let desired_channels = if grayscale then 1 else 3 in
      match Stb_image.load ~channels:desired_channels path with
      | Ok img ->
          let h = Stb_image.height img in
          let w = Stb_image.width img in
          let c = Stb_image.channels img in
          let buffer = Stb_image.data img in
          let nd = Nx.of_bigarray_ext (genarray_of_array1 buffer) in
          let shape = if c = 1 then [| h; w |] else [| h; w; c |] in
          Ok (Nx.reshape shape nd)
      | Error (`Msg msg) -> Error (Format_error msg)
    with
    | Sys_error msg -> Error (Io_error msg)
    | ex -> Error (Other (Printexc.to_string ex))

  let save_image ?(overwrite = true) path img =
    try
      (* Check if file exists and overwrite is false *)
      if (not overwrite) && Sys.file_exists path then
        Error (Io_error (Printf.sprintf "File '%s' already exists" path))
      else
        let h, w, c =
          match get_nx_dims img with
          | `Gray (h, w) -> (h, w, 1)
          | `Color (h, w, c) -> (h, w, c)
        in
        (* Ensure the input array is uint8 *)
        let data_gen = Nx.to_bigarray_ext img in
        let data =
          match Genarray.kind data_gen with
          | Int8_unsigned -> array1_of_genarray data_gen
        in
        let extension = Filename.extension path |> String.lowercase_ascii in
        match extension with
        | ".png" ->
            Stb_image_write.png path ~w ~h ~c data;
            Ok ()
        | ".bmp" ->
            Stb_image_write.bmp path ~w ~h ~c data;
            Ok ()
        | ".tga" ->
            Stb_image_write.tga path ~w ~h ~c data;
            Ok ()
        | ".jpg" | ".jpeg" ->
            Stb_image_write.jpg path ~w ~h ~c ~quality:90 data;
            Ok ()
        | _ ->
            Error
              (Format_error
                 (Printf.sprintf
                    "Unsupported image format: '%s'. Use .png, .bmp, .tga, .jpg"
                    extension))
    with
    | Sys_error msg -> Error (Io_error msg)
    | Invalid_argument msg -> Error (Other msg)
    | Failure msg -> Error (Other msg)
    | ex -> Error (Other (Printexc.to_string ex))

  let load_npy path = Nx_npy.load_npy path

  let save_npy ?(overwrite = true) path arr =
    Nx_npy.save_npy ~overwrite path arr

  let load_npz path = Nx_npy.load_npz path
  let load_npz_member ~name path = Nx_npy.load_npz_member ~name path

  let save_npz ?(overwrite = true) path items =
    Nx_npy.save_npz ~overwrite path items

  (* Text I/O *)
  let save_txt ?(sep = " ") ?(append = false) ~out (type a) (type b)
      (arr : (a, b) Nx.t) =
    try
      let rank = Array.length (Nx.shape arr) in
      if rank <> 1 && rank <> 2 then Error Unsupported_shape
      else
        let perm = 0o666 in
        let open_flags =
          if append then [ Open_wronly; Open_creat; Open_append; Open_text ]
          else [ Open_wronly; Open_creat; Open_trunc; Open_text ]
        in
        let oc = open_out_gen open_flags perm out in
        Fun.protect
          ~finally:(fun () -> close_out oc)
          (fun () ->
            let shape = Nx.shape arr in
            let rows, cols =
              match shape with
              | [| n |] -> (1, n)
              | [| r; c |] -> (r, c)
              | _ -> assert false
            in
            let arr2d_any = Nx.reshape [| rows; cols |] arr in
            let res =
              match Nx.dtype arr with
              | Nx_core.Dtype.Float32 | Nx_core.Dtype.Float64 ->
                  let arrf : (float, Nx.float64_elt) Nx.t =
                    Nx.astype Nx.float64 arr2d_any
                  in
                  for i = 0 to rows - 1 do
                    for j = 0 to cols - 1 do
                      if j > 0 then output_string oc sep;
                      Printf.fprintf oc "%g" (Nx.item [ i; j ] arrf)
                    done;
                    output_char oc '\n'
                  done;
                  Ok ()
              | Nx_core.Dtype.Int32 ->
                  let arri32 : (int32, Nx.int32_elt) Nx.t =
                    Nx.astype Nx.int32 arr2d_any
                  in
                  for i = 0 to rows - 1 do
                    for j = 0 to cols - 1 do
                      if j > 0 then output_string oc sep;
                      output_string oc
                        (Int32.to_string (Nx.item [ i; j ] arri32))
                    done;
                    output_char oc '\n'
                  done;
                  Ok ()
              | Nx_core.Dtype.Int64 ->
                  let arri64 : (int64, Nx.int64_elt) Nx.t =
                    Nx.astype Nx.int64 arr2d_any
                  in
                  for i = 0 to rows - 1 do
                    for j = 0 to cols - 1 do
                      if j > 0 then output_string oc sep;
                      output_string oc
                        (Int64.to_string (Nx.item [ i; j ] arri64))
                    done;
                    output_char oc '\n'
                  done;
                  Ok ()
              | _ -> Error Unsupported_dtype
            in
            res)
    with
    | Sys_error msg -> Error (Io_error msg)
    | Unix.Unix_error (e, _, _) -> Error (Io_error (Unix.error_message e))
    | Failure msg -> Error (Other msg)
    | ex -> Error (Other (Printexc.to_string ex))

  let load_txt ?(sep = " ") dtype path =
    try
      let ic = open_in path in
      Fun.protect
        ~finally:(fun () -> close_in ic)
        (fun () ->
          let rec next_nonempty () =
            try
              let line = input_line ic in
              if String.trim line = "" then next_nonempty () else Some line
            with End_of_file -> None
          in
          let split_line line =
            let line = String.trim line in
            if sep = "" then [ line ]
            else if String.length sep = 1 then
              List.filter (fun s -> s <> "")
                (String.split_on_char sep.[0] line)
            else (
              (* split by substring 'sep' and drop empty parts *)
              let rec aux acc start =
                if start >= String.length line then List.rev acc
                else
                  match String.index_from_opt line start sep.[0] with
                  | None ->
                      let part = String.sub line start (String.length line - start) in
                      let acc = if part = "" then acc else part :: acc in
                      List.rev acc
                  | Some idx ->
                      if idx + String.length sep <= String.length line
                         && String.sub line idx (String.length sep) = sep
                      then
                        let part = String.sub line start (idx - start) in
                        let acc = if part = "" then acc else part :: acc in
                        aux acc (idx + String.length sep)
                      else aux acc (idx + 1)
              in
              aux [] 0)
          in
          match next_nonempty () with
          | None -> Error (Format_error "Empty file")
          | Some first_line ->
              let first_vals = split_line first_line in
              let cols = List.length first_vals in
              if cols = 0 then Error (Format_error "No columns detected")
              else (
                let rows_rev = ref [ first_vals ] in
                let rec loop () =
                  match next_nonempty () with
                  | None -> ()
                  | Some line ->
                      let vals = split_line line in
                      if List.length vals <> cols then
                        raise (Failure "Inconsistent number of columns")
                      else (
                        rows_rev := vals :: !rows_rev;
                        loop ())
                in
                loop ();
                let rows = List.rev !rows_rev in
                let rows_count = List.length rows in
                (* Read as float64 then cast to requested dtype *)
                let ba =
                  Bigarray_ext.Genarray.create Bigarray_ext.Float64 Bigarray.c_layout
                    [| rows_count; cols |]
                in
                List.iteri
                  (fun i vals ->
                    List.iteri
                      (fun j v ->
                        Bigarray_ext.Genarray.set ba [| i; j |]
                          (float_of_string v))
                      vals)
                  rows;
                let nxf = Nx.of_bigarray_ext ba in
                let result = Nx.astype dtype nxf in
                Ok result))
    with
    | Sys_error msg -> Error (Io_error msg)
    | Unix.Unix_error (e, _, _) -> Error (Io_error (Unix.error_message e))
    | Failure msg -> Error (Format_error msg)
    | ex -> Error (Other (Printexc.to_string ex))

  (* Conversions from packed arrays *)

  let as_float16 = Packed_nx.as_float16
  let as_bfloat16 = Packed_nx.as_bfloat16
  let as_float32 = Packed_nx.as_float32
  let as_float64 = Packed_nx.as_float64
  let as_int8 = Packed_nx.as_int8
  let as_int16 = Packed_nx.as_int16
  let as_int32 = Packed_nx.as_int32
  let as_int64 = Packed_nx.as_int64
  let as_uint8 = Packed_nx.as_uint8
  let as_uint16 = Packed_nx.as_uint16
  let as_complex32 = Packed_nx.as_complex32
  let as_complex64 = Packed_nx.as_complex64

  (* SafeTensors support *)
  let load_safetensor path = Nx_safetensors.load_safetensor path

  let save_safetensor ?overwrite path items =
    Nx_safetensors.save_safetensor ?overwrite path items
end

(* Main module functions - these fail directly instead of returning results *)

let unwrap_result = function
  | Ok v -> v
  | Error err -> failwith (Error.to_string err)

let as_float16 packed = Packed_nx.as_float16 packed |> unwrap_result
let as_bfloat16 packed = Packed_nx.as_bfloat16 packed |> unwrap_result
let as_float32 packed = Packed_nx.as_float32 packed |> unwrap_result
let as_float64 packed = Packed_nx.as_float64 packed |> unwrap_result
let as_int8 packed = Packed_nx.as_int8 packed |> unwrap_result
let as_int16 packed = Packed_nx.as_int16 packed |> unwrap_result
let as_int32 packed = Packed_nx.as_int32 packed |> unwrap_result
let as_int64 packed = Packed_nx.as_int64 packed |> unwrap_result
let as_uint8 packed = Packed_nx.as_uint8 packed |> unwrap_result
let as_uint16 packed = Packed_nx.as_uint16 packed |> unwrap_result
let as_complex32 packed = Packed_nx.as_complex32 packed |> unwrap_result
let as_complex64 packed = Packed_nx.as_complex64 packed |> unwrap_result

let load_image ?grayscale path =
  Safe.load_image ?grayscale path |> unwrap_result

let save_image ?overwrite path img =
  Safe.save_image ?overwrite path img |> unwrap_result

let load_npy path = Safe.load_npy path |> unwrap_result

let save_npy ?overwrite path arr =
  Safe.save_npy ?overwrite path arr |> unwrap_result

let load_npz path = Safe.load_npz path |> unwrap_result

let load_npz_member ~name path =
  Safe.load_npz_member ~name path |> unwrap_result

let save_npz ?overwrite path items =
  Safe.save_npz ?overwrite path items |> unwrap_result

let load_safetensor path = Safe.load_safetensor path |> unwrap_result

let save_safetensor ?overwrite path items =
  Safe.save_safetensor ?overwrite path items |> unwrap_result

let save_txt ?sep ?append ~out arr =
  Safe.save_txt ?sep ?append ~out arr |> unwrap_result

let load_txt ?sep dtype path = Safe.load_txt ?sep dtype path |> unwrap_result
