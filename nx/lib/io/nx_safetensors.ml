open Bigarray_ext
open Error
open Packed_nx

(* External functions for 16-bit float conversions *)
external half_to_float : int -> float = "caml_half_to_float"
external bfloat16_to_float : int -> float = "caml_bfloat16_to_float"
external float_to_half : float -> int = "caml_float_to_half"
external float_to_bfloat16 : float -> int = "caml_float_to_bfloat16"

let load_safetensor path =
  try
    let ic = open_in_bin path in
    let len = in_channel_length ic in
    let buffer = really_input_string ic len in
    close_in ic;
    match Safetensors.deserialize buffer with
    | Ok safetensors ->
        let tensors = Safetensors.tensors safetensors in
        let result = Hashtbl.create (List.length tensors) in
        List.iter
          (fun (name, (view : Safetensors.tensor_view)) ->
            let open Safetensors in
            let shape = Array.of_list view.shape in
            let num_elems = Array.fold_left ( * ) 1 shape in

            (* Convert safetensors dtype to Nx array *)
            let process_float32 () =
              let ba = Array1.create Float32 c_layout num_elems in
              for i = 0 to num_elems - 1 do
                let offset = view.offset + (i * 4) in
                let b0 = Char.code view.data.[offset] in
                let b1 = Char.code view.data.[offset + 1] in
                let b2 = Char.code view.data.[offset + 2] in
                let b3 = Char.code view.data.[offset + 3] in
                let bits =
                  Int32.(
                    logor
                      (shift_left (of_int b3) 24)
                      (logor
                         (shift_left (of_int b2) 16)
                         (logor (shift_left (of_int b1) 8) (of_int b0))))
                in
                Array1.unsafe_set ba i (Int32.float_of_bits bits)
              done;
              let nx_arr = Nx.of_bigarray_ext (genarray_of_array1 ba) in
              Nx.reshape shape nx_arr
            in

            let process_float64 () =
              let ba = Array1.create Float64 c_layout num_elems in
              for i = 0 to num_elems - 1 do
                let offset = view.offset + (i * 8) in
                let bits = Safetensors.read_u64_le view.data offset in
                Array1.unsafe_set ba i (Int64.float_of_bits bits)
              done;
              let nx_arr = Nx.of_bigarray_ext (genarray_of_array1 ba) in
              Nx.reshape shape nx_arr
            in

            let process_int32 () =
              let ba = Array1.create Int32 c_layout num_elems in
              for i = 0 to num_elems - 1 do
                let offset = view.offset + (i * 4) in
                let b0 = Char.code view.data.[offset] in
                let b1 = Char.code view.data.[offset + 1] in
                let b2 = Char.code view.data.[offset + 2] in
                let b3 = Char.code view.data.[offset + 3] in
                let bits =
                  Int32.(
                    logor
                      (shift_left (of_int b3) 24)
                      (logor
                         (shift_left (of_int b2) 16)
                         (logor (shift_left (of_int b1) 8) (of_int b0))))
                in
                Array1.unsafe_set ba i bits
              done;
              let nx_arr = Nx.of_bigarray_ext (genarray_of_array1 ba) in
              Nx.reshape shape nx_arr
            in

            match view.dtype with
            | F32 -> Hashtbl.add result name (P (process_float32 ()))
            | F64 -> Hashtbl.add result name (P (process_float64 ()))
            | I32 -> Hashtbl.add result name (P (process_int32 ()))
            | F16 ->
                let ba = Array1.create Float16 c_layout num_elems in
                for i = 0 to num_elems - 1 do
                  let offset = view.offset + (i * 2) in
                  let b0 = Char.code view.data.[offset] in
                  let b1 = Char.code view.data.[offset + 1] in
                  (* Combine bytes to get 16-bit value (little-endian) *)
                  let bits = (b1 lsl 8) lor b0 in
                  let float_val = half_to_float bits in
                  Array1.unsafe_set ba i float_val
                done;
                let nx_arr = Nx.of_bigarray_ext (genarray_of_array1 ba) in
                Hashtbl.add result name (P (Nx.reshape shape nx_arr))
            | BF16 ->
                let ba = Array1.create Bfloat16 c_layout num_elems in
                for i = 0 to num_elems - 1 do
                  let offset = view.offset + (i * 2) in
                  let b0 = Char.code view.data.[offset] in
                  let b1 = Char.code view.data.[offset + 1] in
                  (* Combine bytes to get 16-bit value (little-endian) *)
                  let bits = (b1 lsl 8) lor b0 in
                  let float_val = bfloat16_to_float bits in
                  Array1.unsafe_set ba i float_val
                done;
                let nx_arr = Nx.of_bigarray_ext (genarray_of_array1 ba) in
                Hashtbl.add result name (P (Nx.reshape shape nx_arr))
            | _ ->
                Printf.eprintf
                  "Warning: Skipping tensor '%s' with unsupported dtype %s\n"
                  name
                  (Safetensors.dtype_to_string view.dtype))
          tensors;
        Ok result
    | Error err -> Error (Format_error (Safetensors.string_of_error err))
  with
  | Sys_error msg -> Error (Io_error msg)
  | ex -> Error (Other (Printexc.to_string ex))

let save_safetensor ?(overwrite = true) path items =
  try
    if (not overwrite) && Sys.file_exists path then
      Error (Io_error (Printf.sprintf "File '%s' already exists" path))
    else
      let tensor_views =
        List.map
          (fun (name, P arr) ->
            let shape = Array.to_list (Nx.shape arr) in
            let ba = Nx.to_bigarray_ext arr in
            let num_elems = Array.fold_left ( * ) 1 (Nx.shape arr) in

            (* Create data buffer and determine dtype based on Nx array type *)
            let dtype, data =
              match Genarray.kind ba with
              | Float32 ->
                  let bytes = Bytes.create (num_elems * 4) in
                  let ba_flat = Nx.to_bigarray_ext (Nx.flatten arr) in
                  let ba1 = array1_of_genarray ba_flat in
                  for i = 0 to num_elems - 1 do
                    let bits = Int32.bits_of_float (Array1.unsafe_get ba1 i) in
                    let offset = i * 4 in
                    Bytes.set bytes offset
                      (Char.chr (Int32.to_int (Int32.logand bits 0xffl)));
                    Bytes.set bytes (offset + 1)
                      (Char.chr
                         (Int32.to_int
                            (Int32.logand (Int32.shift_right bits 8) 0xffl)));
                    Bytes.set bytes (offset + 2)
                      (Char.chr
                         (Int32.to_int
                            (Int32.logand (Int32.shift_right bits 16) 0xffl)));
                    Bytes.set bytes (offset + 3)
                      (Char.chr
                         (Int32.to_int
                            (Int32.logand (Int32.shift_right bits 24) 0xffl)))
                  done;
                  (Safetensors.F32, Bytes.unsafe_to_string bytes)
              | Float64 ->
                  let bytes = Bytes.create (num_elems * 8) in
                  let ba_flat = Nx.to_bigarray_ext (Nx.flatten arr) in
                  let ba1 = array1_of_genarray ba_flat in
                  for i = 0 to num_elems - 1 do
                    let bits = Int64.bits_of_float (Array1.unsafe_get ba1 i) in
                    Safetensors.write_u64_le bytes (i * 8) bits
                  done;
                  (Safetensors.F64, Bytes.unsafe_to_string bytes)
              | Int32 ->
                  let bytes = Bytes.create (num_elems * 4) in
                  let ba_flat = Nx.to_bigarray_ext (Nx.flatten arr) in
                  let ba1 = array1_of_genarray ba_flat in
                  for i = 0 to num_elems - 1 do
                    let value = Array1.unsafe_get ba1 i in
                    let offset = i * 4 in
                    Bytes.set bytes offset
                      (Char.chr (Int32.to_int (Int32.logand value 0xffl)));
                    Bytes.set bytes (offset + 1)
                      (Char.chr
                         (Int32.to_int
                            (Int32.logand (Int32.shift_right value 8) 0xffl)));
                    Bytes.set bytes (offset + 2)
                      (Char.chr
                         (Int32.to_int
                            (Int32.logand (Int32.shift_right value 16) 0xffl)));
                    Bytes.set bytes (offset + 3)
                      (Char.chr
                         (Int32.to_int
                            (Int32.logand (Int32.shift_right value 24) 0xffl)))
                  done;
                  (Safetensors.I32, Bytes.unsafe_to_string bytes)
              | Float16 ->
                  let bytes = Bytes.create (num_elems * 2) in
                  let ba_flat = Nx.to_bigarray_ext (Nx.flatten arr) in
                  let ba1 = array1_of_genarray ba_flat in
                  for i = 0 to num_elems - 1 do
                    let float_val = Array1.unsafe_get ba1 i in
                    (* Convert float to half-precision bits *)
                    let bits = float_to_half float_val in
                    let offset = i * 2 in
                    (* Store as little-endian *)
                    Bytes.set bytes offset (Char.chr (bits land 0xff));
                    Bytes.set bytes (offset + 1) (Char.chr ((bits lsr 8) land 0xff))
                  done;
                  (Safetensors.F16, Bytes.unsafe_to_string bytes)
              | Bfloat16 ->
                  let bytes = Bytes.create (num_elems * 2) in
                  let ba_flat = Nx.to_bigarray_ext (Nx.flatten arr) in
                  let ba1 = array1_of_genarray ba_flat in
                  for i = 0 to num_elems - 1 do
                    let float_val = Array1.unsafe_get ba1 i in
                    (* Convert float to bfloat16 bits *)
                    let bits = float_to_bfloat16 float_val in
                    let offset = i * 2 in
                    (* Store as little-endian *)
                    Bytes.set bytes offset (Char.chr (bits land 0xff));
                    Bytes.set bytes (offset + 1) (Char.chr ((bits lsr 8) land 0xff))
                  done;
                  (Safetensors.BF16, Bytes.unsafe_to_string bytes)
              | _ ->
                  fail_msg "Unsupported dtype for safetensors: %s"
                    (Nx_core.Dtype.of_bigarray_ext_kind (Genarray.kind ba)
                    |> Nx_core.Dtype.to_string)
            in

            match Safetensors.tensor_view_new ~dtype ~shape ~data with
            | Ok view -> (name, view)
            | Error err ->
                fail_msg "Failed to create tensor view for '%s': %s" name
                  (Safetensors.string_of_error err))
          items
      in
      match Safetensors.serialize_to_file tensor_views None path with
      | Ok () -> Ok ()
      | Error err -> Error (Format_error (Safetensors.string_of_error err))
  with
  | Sys_error msg -> Error (Io_error msg)
  | ex -> Error (Other (Printexc.to_string ex))
