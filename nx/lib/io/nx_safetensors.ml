open Bigarray_ext
open Error
open Packed_nx

(* TODO: consider wiring a bulk half/bfloat16 blit in Bigarray_ext to avoid
   per-element OCaml loops once we profile safetensor workloads. That would
   require extending the Bigarray API with bytes<->array blits. *)
let float_to_bfloat16_bits (f : float) : int =
  let open Int32 in
  let bits = bits_of_float f in
  let exponent = to_int (shift_right_logical (logand bits 0x7F800000l) 23) in
  let mantissa = logand bits 0x007FFFFFl in
  let upper = to_int (shift_right_logical bits 16) in
  if exponent = 0xFF then
    if mantissa = 0l then upper
    else if upper land 0x007F = 0 then upper lor 0x0001
    else upper
  else
    let lsb = logand (shift_right_logical bits 16) 1l in
    let rounding_bias = add (of_int 0x7FFF) lsb in
    let rounded = add bits rounding_bias in
    shift_right_logical rounded 16 |> to_int

let bfloat16_bits_to_float (bits : int) : float =
  Int32.(float_of_bits (shift_left (of_int (bits land 0xFFFF)) 16))

let float_to_half_bits (f : float) : int =
  let bits = Int32.bits_of_float f in
  let sign =
    Int32.(to_int (shift_right_logical (logand bits 0x80000000l) 16))
  in
  let mantissa = Int32.(to_int (logand bits 0x007FFFFFl)) in
  let exponent =
    Int32.(to_int (shift_right_logical (logand bits 0x7F800000l) 23))
  in
  if exponent = 255 then
    if mantissa = 0 then sign lor 0x7C00
    else
      let payload = mantissa lsr 13 in
      let payload = if payload = 0 then 1 else payload in
      sign lor 0x7C00 lor payload
  else
    let exp = exponent - 127 in
    if exp < -14 then
      if exp < -24 then sign
      else
        let mant = mantissa lor 0x00800000 in
        let shift = -exp - 1 in
        let t = mant lsr shift in
        let should_round =
          t land 1 = 1 && (t land 2 = 2 || mant land ((1 lsl shift) - 1) <> 0)
        in
        let t = if should_round then t + 1 else t in
        sign lor (t lsr 1)
    else if exp > 15 then sign lor 0x7C00
    else
      let mant = mantissa + 0x00001000 in
      let mant, exp =
        if mant land 0x00800000 <> 0 then (0, exp + 1) else (mant, exp)
      in
      if exp > 15 then sign lor 0x7C00
      else sign lor (((exp + 15) lsl 10) lor (mant lsr 13))

let half_bits_to_float (bits : int) : float =
  let sign = (bits land 0x8000) lsl 16 in
  let exponent = (bits lsr 10) land 0x1F in
  let mantissa = bits land 0x3FF in
  let open Int32 in
  let sign_bits = of_int sign in
  let value_bits =
    if exponent = 0x1F then
      let mant =
        if mantissa = 0 then 0l
        else logor (shift_left (of_int mantissa) 13) 0x400000l
      in
      logor sign_bits (logor 0x7F800000l mant)
    else if exponent = 0 then (
      if mantissa = 0 then sign_bits
      else
        let mant = ref (mantissa lsl 1) in
        let exp_val = ref (-14) in
        while !mant land 0x400 = 0 do
          mant := !mant lsl 1;
          exp_val := !exp_val - 1
        done;
        let mant = !mant land 0x3FF in
        let mant32 = shift_left (of_int mant) 13 in
        let exp32 = shift_left (of_int (!exp_val + 127)) 23 in
        logor sign_bits (logor exp32 mant32))
    else
      let exp32 = shift_left (of_int (exponent + 112)) 23 in
      let mant32 = shift_left (of_int mantissa) 13 in
      logor sign_bits (logor exp32 mant32)
  in
  Int32.float_of_bits value_bits

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
                  let float_val = half_bits_to_float bits in
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
                  let float_val = bfloat16_bits_to_float bits in
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
                    let bits = float_to_half_bits float_val in
                    let offset = i * 2 in
                    (* Store as little-endian *)
                    Bytes.set bytes offset (Char.chr (bits land 0xff));
                    Bytes.set bytes (offset + 1)
                      (Char.chr ((bits lsr 8) land 0xff))
                  done;
                  (Safetensors.F16, Bytes.unsafe_to_string bytes)
              | Bfloat16 ->
                  let bytes = Bytes.create (num_elems * 2) in
                  let ba_flat = Nx.to_bigarray_ext (Nx.flatten arr) in
                  let ba1 = array1_of_genarray ba_flat in
                  for i = 0 to num_elems - 1 do
                    let float_val = Array1.unsafe_get ba1 i in
                    (* Convert float to bfloat16 bits *)
                    let bits = float_to_bfloat16_bits float_val in
                    let offset = i * 2 in
                    (* Store as little-endian *)
                    Bytes.set bytes offset (Char.chr (bits land 0xff));
                    Bytes.set bytes (offset + 1)
                      (Char.chr ((bits lsr 8) land 0xff))
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
