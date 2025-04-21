open Descriptor

module Make (B : Backend_intf.S) = struct
  (* Helper to create ndarray from host buffer *)
  let create_ndarray_from_host_buffer context descriptor host_buffer =
    (* Assumes B.from_buffer handles the context correctly, potentially
       transferring data from the host_buffer to the device if necessary. *)
    B.from_buffer context descriptor host_buffer

  let create context dtype shape ocaml_array =
    let n = Array.fold_left ( * ) 1 shape in
    if Array.length ocaml_array <> n then
      invalid_arg
        (Printf.sprintf "create: data size (%d) does not match shape size (%d)"
           (Array.length ocaml_array) n);

    if n = 0 && Array.length shape > 0 then B.empty context dtype shape
    else
      let host_buffer = Buffer.create_buffer dtype n in
      (* Copy data from OCaml array to host buffer *)
      for i = 0 to n - 1 do
        Bigarray.Array1.unsafe_set host_buffer i ocaml_array.(i)
      done;
      let strides = compute_c_strides shape in
      let descriptor =
        { dtype; shape; layout = C_contiguous; strides; offset = 0 }
      in
      create_ndarray_from_host_buffer context descriptor host_buffer

  let init context dtype shape f =
    let n = Array.fold_left ( * ) 1 shape in
    if n = 0 && Array.length shape > 0 then B.empty context dtype shape
    else
      let host_buffer = Buffer.create_buffer dtype n in
      for i = 0 to n - 1 do
        let idx = linear_to_md_c_contig i (compute_c_strides shape) in
        Bigarray.Array1.unsafe_set host_buffer i (f idx)
      done;
      let strides = compute_c_strides shape in
      let descriptor =
        { dtype; shape; layout = C_contiguous; strides; offset = 0 }
      in
      create_ndarray_from_host_buffer context descriptor host_buffer

  let full context dtype shape value =
    let n = Array.fold_left ( * ) 1 shape in
    (* Handle shapes like [|2;0;3|] which result in n=0 *)
    if n = 0 && Array.length shape > 0 then B.empty context dtype shape
    else
      let host_buffer = Buffer.create_buffer dtype n in
      Buffer.fill value host_buffer;
      let strides = compute_c_strides shape in
      let descriptor =
        { dtype; shape; layout = C_contiguous; strides; offset = 0 }
      in
      create_ndarray_from_host_buffer context descriptor host_buffer

  let zeros context dtype shape = full context dtype shape (zero dtype)
  let ones context dtype shape = full context dtype shape (one dtype)

  (* Assume B.dtype and B.shape are accessors for ('a, 'b) t *)
  let empty_like context arr =
    let ad = B.descriptor arr in
    B.empty context (dtype ad) (shape ad)

  let zeros_like context arr =
    let ad = B.descriptor arr in
    zeros context (dtype ad) (shape ad)

  let ones_like context arr =
    let ad = B.descriptor arr in
    ones context (dtype ad) (shape ad)

  let full_like context value arr =
    let ad = B.descriptor arr in
    full context (dtype ad) (shape ad) value

  let scalar context dtype v = full context dtype [||] v

  let eye context ?m ?k dtype n =
    let m = match m with None -> n | Some m -> m in
    let k = match k with None -> 0 | Some k -> k in
    let shape = [| m; n |] in
    let size = m * n in
    (* Handle zero-sized dimensions *)
    if size = 0 && Array.length shape > 0 then B.empty context dtype shape
    else
      let host_buffer = Buffer.create_buffer dtype size in
      Buffer.fill (zero dtype) host_buffer;
      (* Initialize with zeros *)
      let one_val = one dtype in
      let strides = compute_c_strides shape in
      let start_row = Stdlib.max 0 (-k) in
      let end_row = Stdlib.min m (n - k) in
      for i = start_row to end_row - 1 do
        let row_idx = i in
        let col_idx = i + k in
        (* Check bounds again just in case, though logic should prevent out of
           bounds *)
        if row_idx >= 0 && row_idx < m && col_idx >= 0 && col_idx < n then
          let linear_idx = md_to_linear [| row_idx; col_idx |] strides in
          if linear_idx >= 0 && linear_idx < size then
            (* Bounds check for safety *)
            Bigarray.Array1.unsafe_set host_buffer linear_idx one_val
      done;
      let descriptor =
        { dtype; shape; layout = C_contiguous; strides; offset = 0 }
      in
      create_ndarray_from_host_buffer context descriptor host_buffer

  let identity context dtype n = eye context dtype n

  let arange_int context ~f dtype start stop step =
    if step = 0 then failwith "arange: step cannot be zero";
    let n =
      if step > 0 then Stdlib.max 0 ((stop - start + step - 1) / step)
      else Stdlib.max 0 ((start - stop - step - 1) / -step)
    in
    let shape = [| n |] in
    if n = 0 then B.empty context dtype shape
    else
      let host_buffer = Buffer.create_buffer dtype n in
      let current_val = ref start in
      for i = 0 to n - 1 do
        Bigarray.Array1.unsafe_set host_buffer i (f !current_val);
        current_val := !current_val + step
      done;
      let strides = compute_c_strides shape in
      let descriptor =
        { dtype; shape; layout = C_contiguous; strides; offset = 0 }
      in
      create_ndarray_from_host_buffer context descriptor host_buffer

  let arange_float context ~f dtype start stop step =
    if step = 0.0 then failwith "arange_f: step cannot be zero"
    else
      (* Calculate n carefully to avoid float precision issues near
         boundaries *)
      let n =
        if step > 0. then
          if start >= stop then 0
          else
            Stdlib.max 0
              (int_of_float (Float.floor ((stop -. start -. 1e-9) /. step)) + 1)
        else if
          (* step < 0. *)
          start <= stop
        then 0
        else
          Stdlib.max 0
            (int_of_float (Float.floor ((start -. stop +. 1e-9) /. -.step)) + 1)
      in
      let shape = [| n |] in
      if n = 0 then B.empty context dtype shape
      else
        let host_buffer = Buffer.create_buffer dtype n in
        for i = 0 to n - 1 do
          let v = start +. (float_of_int i *. step) in
          Bigarray.Array1.unsafe_set host_buffer i (f v)
        done;
        let strides = compute_c_strides shape in
        let descriptor =
          { dtype; shape; layout = C_contiguous; strides; offset = 0 }
        in
        create_ndarray_from_host_buffer context descriptor host_buffer

  let arange (type a b) context (dtype : (a, b) dtype) start stop step :
      (a, b) B.b_t =
    match dtype with
    | Float16 -> arange_int context ~f:float_of_int dtype start stop step
    | Float32 -> arange_int context ~f:float_of_int dtype start stop step
    | Float64 -> arange_int context ~f:float_of_int dtype start stop step
    | Int32 -> arange_int context ~f:Int32.of_int dtype start stop step
    | Int64 -> arange_int context ~f:Int64.of_int dtype start stop step
    | Int8 -> arange_int context ~f:Fun.id dtype start stop step
    | Int16 -> arange_int context ~f:Fun.id dtype start stop step
    | UInt8 -> arange_int context ~f:Fun.id dtype start stop step
    | UInt16 -> arange_int context ~f:Fun.id dtype start stop step
    | Complex32 | Complex64 -> failwith "arange not supported for complex types"

  let arange_f (type a b) context (dtype : (a, b) dtype) start stop step :
      (a, b) B.b_t =
    match dtype with
    | Float16 -> arange_float context ~f:Fun.id dtype start stop step
    | Float32 -> arange_float context ~f:Fun.id dtype start stop step
    | Float64 -> arange_float context ~f:Fun.id dtype start stop step
    | Int32 -> arange_float context ~f:Int32.of_float dtype start stop step
    | Int64 -> arange_float context ~f:Int64.of_float dtype start stop step
    | Int8 -> arange_float context ~f:int_of_float dtype start stop step
    | UInt8 -> arange_float context ~f:int_of_float dtype start stop step
    | Int16 -> arange_float context ~f:int_of_float dtype start stop step
    | UInt16 -> arange_float context ~f:int_of_float dtype start stop step
    | Complex32 | Complex64 ->
        failwith "arange_f not supported for complex types"

  (* Helper for type-aware value creation from float *)
  let make_value : type a b. (a, b) dtype -> float -> a =
   fun dtype value ->
    match dtype with
    | Float16 ->
        value
        (* Assume float can represent float16 accurately enough for creation *)
    | Float32 -> value
    | Float64 -> value
    | Int32 -> Int32.of_float value
    | Int64 -> Int64.of_float value
    | Int8 -> int_of_float value
    | Int16 -> int_of_float value
    | UInt8 -> int_of_float value (* Add clamping/range checks if necessary *)
    | UInt16 -> int_of_float value (* Add clamping/range checks if necessary *)
    | Complex32 -> { Complex.re = value; im = 0. }
    | Complex64 -> { Complex.re = value; im = 0. }

  let linspace (type a b) context (dtype : (a, b) dtype) ?(endpoint = true)
      start stop num : (a, b) B.b_t =
    if num < 0 then invalid_arg "linspace: num must be non-negative";
    let shape = [| num |] in
    if num = 0 then B.empty context dtype shape
    else if num = 1 then full context dtype shape (make_value dtype start)
      (* Handle num=1 case *)
    else
      let host_buffer = Buffer.create_buffer dtype num in
      let den = float_of_int (if endpoint then num - 1 else num) in
      let step = if den > 0. then (stop -. start) /. den else 0.0 in
      (* Avoid division by zero if num=1 and endpoint=true *)
      for i = 0 to num - 1 do
        let value = start +. (float_of_int i *. step) in
        let set_val = make_value dtype value in
        Bigarray.Array1.unsafe_set host_buffer i set_val
      done;
      (* Adjust the last element precisely if endpoint is true *)
      (if endpoint then
         let set_val_stop = make_value dtype stop in
         Bigarray.Array1.unsafe_set host_buffer (num - 1) set_val_stop);
      let strides = compute_c_strides shape in
      let descriptor =
        { dtype; shape; layout = C_contiguous; strides; offset = 0 }
      in
      create_ndarray_from_host_buffer context descriptor host_buffer

  let logspace (type a b) context (dtype : (a, b) dtype) ?(endpoint = true)
      ?(base = 10.) start stop num : (a, b) B.b_t =
    if num < 0 then invalid_arg "logspace: num must be non-negative";
    let shape = [| num |] in
    if num = 0 then B.empty context dtype shape
    else
      let host_buffer = Buffer.create_buffer dtype num in
      (* Use linspace logic for the exponents *)
      let lin_start = start in
      let lin_stop = stop in
      (if num = 1 then
         let log_val = base ** lin_start in
         let set_val = make_value dtype log_val in
         Bigarray.Array1.unsafe_set host_buffer 0 set_val
       else
         let den = float_of_int (if endpoint then num - 1 else num) in
         let lin_step =
           if den > 0. then (lin_stop -. lin_start) /. den else 0.0
         in
         let pow_base = base in
         for i = 0 to num - 1 do
           let lin_val = lin_start +. (float_of_int i *. lin_step) in
           (* Correct last value if endpoint is true *)
           let lin_val_corrected =
             if endpoint && i = num - 1 then lin_stop else lin_val
           in
           let log_val = pow_base ** lin_val_corrected in
           let set_val = make_value dtype log_val in
           Bigarray.Array1.unsafe_set host_buffer i set_val
         done);
      let strides = compute_c_strides shape in
      let descriptor =
        { dtype; shape; layout = C_contiguous; strides; offset = 0 }
      in
      create_ndarray_from_host_buffer context descriptor host_buffer

  let geomspace (type a b) context (dtype : (a, b) dtype) ?(endpoint = true)
      start stop num : (a, b) B.b_t =
    if start <= 0. || stop <= 0. then
      invalid_arg "geomspace: start and stop must be positive";
    if num < 0 then invalid_arg "geomspace: num must be non-negative";
    let shape = [| num |] in
    if num = 0 then B.empty context dtype shape
    else
      let host_buffer = Buffer.create_buffer dtype num in
      (* Use linspace logic for the log of the values *)
      (if num = 1 then
         let set_val = make_value dtype start in
         Bigarray.Array1.unsafe_set host_buffer 0 set_val
       else
         let log_start = Stdlib.log start in
         let log_stop = Stdlib.log stop in
         let den = float_of_int (if endpoint then num - 1 else num) in
         let log_step =
           if den > 0. then (log_stop -. log_start) /. den else 0.0
         in
         for i = 0 to num - 1 do
           let log_lin_val = log_start +. (float_of_int i *. log_step) in
           (* Correct last value if endpoint is true *)
           let log_lin_val_corrected =
             if endpoint && i = num - 1 then log_stop else log_lin_val
           in
           let geom_val = Stdlib.exp log_lin_val_corrected in
           let set_val = make_value dtype geom_val in
           Bigarray.Array1.unsafe_set host_buffer i set_val
         done);
      let strides = compute_c_strides shape in
      let descriptor =
        { dtype; shape; layout = C_contiguous; strides; offset = 0 }
      in
      create_ndarray_from_host_buffer context descriptor host_buffer

  (* Helper for random float value creation, checking type *)
  let make_rand_float_value : type a b. (a, b) dtype -> float -> a =
   fun dtype value ->
    match dtype with
    | Float16 -> value
    | Float32 -> value
    | Float64 -> value
    | _ -> failwith "rand/randn requires a float dtype"

  let rand (type a b) context (dtype : (a, b) dtype) ?seed shape : (a, b) B.b_t
      =
    let n = Array.fold_left ( * ) 1 shape in
    if n = 0 && Array.length shape > 0 then B.empty context dtype shape
    else
      let host_buffer = Buffer.create_buffer dtype n in
      let () =
        match seed with
        | None -> Random.self_init ()
        | Some seed -> Random.init seed
      in
      (* Check dtype before generating *)
      let set_val_func =
        match dtype with
        | Float16 | Float32 | Float64 -> fun r -> make_rand_float_value dtype r
        | _ -> failwith "rand requires a float dtype"
      in
      if n > 0 then
        for i = 0 to n - 1 do
          let rand_val = Random.float 1.0 in
          Bigarray.Array1.unsafe_set host_buffer i (set_val_func rand_val)
        done;
      let strides = compute_c_strides shape in
      let descriptor =
        { dtype; shape; layout = C_contiguous; strides; offset = 0 }
      in
      create_ndarray_from_host_buffer context descriptor host_buffer

  let randn (type a b) context (dtype : (a, b) dtype) ?seed shape : (a, b) B.b_t
      =
    let n = Array.fold_left ( * ) 1 shape in
    if n = 0 && Array.length shape > 0 then B.empty context dtype shape
    else
      let host_buffer = Buffer.create_buffer dtype n in
      let () =
        match seed with
        | None -> Random.self_init ()
        | Some seed -> Random.init seed
      in
      (* Box-Muller transform state *)
      let cache = ref None in
      let random_normal () =
        match !cache with
        | Some z ->
            cache := None;
            z
        | None ->
            let u1 = ref (Random.float 1.0) in
            let u2 = ref (Random.float 1.0) in
            (* Ensure u1 is not zero for log *)
            while !u1 = 0.0 do
              u1 := Random.float 1.0
            done;
            let r = Stdlib.sqrt (-2. *. Stdlib.log !u1) in
            let theta = 2. *. Float.pi *. !u2 in
            let z0 = r *. Stdlib.cos theta in
            let z1 = r *. Stdlib.sin theta in
            cache := Some z1;
            z0
      in
      (* Check dtype before generating *)
      let set_val_func =
        match dtype with
        | Float16 | Float32 | Float64 -> fun r -> make_rand_float_value dtype r
        | _ -> failwith "randn requires a float dtype"
      in
      if n > 0 then
        for i = 0 to n - 1 do
          let rand_val = random_normal () in
          Bigarray.Array1.unsafe_set host_buffer i (set_val_func rand_val)
        done;
      let strides = compute_c_strides shape in
      let descriptor =
        { dtype; shape; layout = C_contiguous; strides; offset = 0 }
      in
      create_ndarray_from_host_buffer context descriptor host_buffer

  let randint (type a b) context (dtype : (a, b) dtype) ?seed ?high shape low :
      (a, b) B.b_t =
    let n = Array.fold_left ( * ) 1 shape in
    if n = 0 && Array.length shape > 0 then B.empty context dtype shape
    else
      let host_buffer = Buffer.create_buffer dtype n in
      let actual_high, low_val =
        match high with
        | None -> (low, 0 (* If high is None, range is [0, low) *))
        | Some h -> (h, low)
      in
      if low_val >= actual_high then
        failwith "randint: high must be strictly greater than low";

      let () =
        match seed with None -> Random.self_init () | Some s -> Random.init s
      in

      let fill_random : unit -> a =
        let range_int = actual_high - low_val in
        if range_int <= 0 then failwith "randint: range must be positive";
        match dtype with
        | Int32 ->
            let low32 = Int32.of_int low_val in
            let range32 = Int32.of_int range_int in
            fun () -> Int32.add low32 (Random.int32 range32)
        | Int64 ->
            let low64 = Int64.of_int low_val in
            let range64 = Int64.of_int range_int in
            fun () -> Int64.add low64 (Random.int64 range64)
        | Int8 -> fun () -> Random.int range_int + low_val
        | Int16 -> fun () -> Random.int range_int + low_val
        | UInt8 -> fun () -> Random.int range_int + low_val
        | UInt16 -> fun () -> Random.int range_int + low_val
        | Float16 | Float32 | Float64 | Complex32 | Complex64 ->
            failwith "randint requires an integer dtype"
      in

      if n > 0 then
        for i = 0 to n - 1 do
          Bigarray.Array1.unsafe_set host_buffer i (fill_random ())
        done;

      let strides = compute_c_strides shape in
      let descriptor =
        { dtype; shape; layout = C_contiguous; strides; offset = 0 }
      in
      create_ndarray_from_host_buffer context descriptor host_buffer
end
