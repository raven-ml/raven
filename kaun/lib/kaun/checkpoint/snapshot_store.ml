open Util

type tensor_meta = { encoded_path : string; dtype : string; shape : int array }

let tensors_meta_to_yojson metas =
  `List
    (List.map
       (fun { encoded_path; dtype; shape } ->
         `Assoc
           [
             ("path", `String encoded_path);
             ("dtype", `String dtype);
             ("shape", `List (Array.to_list shape |> List.map (fun d -> `Int d)));
           ])
       metas)

let tensors_meta_of_yojson json =
  match json with
  | `List entries ->
      let decode_entry = function
        | `Assoc fields -> (
            match
              ( List.assoc_opt "path" fields,
                List.assoc_opt "dtype" fields,
                List.assoc_opt "shape" fields )
            with
            | Some (`String path), Some (`String dtype), Some (`List shape) ->
                let shape =
                  List.fold_left
                    (fun acc -> function
                      | `Int dim -> dim :: acc
                      | _ ->
                          failwith
                            "Snapshot_store.tensors_meta_of_yojson: invalid \
                             shape value")
                    [] shape
                  |> List.rev |> Array.of_list
                in
                { encoded_path = path; dtype; shape }
            | _ ->
                failwith "Snapshot_store.tensors_meta_of_yojson: missing fields"
            )
        | _ -> failwith "Snapshot_store.tensors_meta_of_yojson: expected object"
      in
      List.map decode_entry entries
  | _ -> failwith "Snapshot_store.tensors_meta_of_yojson: expected list"

let dtype_to_string dtype = Nx.dtype_to_string dtype

let convert_packed dtype packed =
  let open Nx_io in
  match String.lowercase_ascii dtype with
  | "f16" | "float16" -> P (as_float16 packed)
  | "bf16" | "bfloat16" -> P (as_bfloat16 packed)
  | "f32" | "float32" -> P (as_float32 packed)
  | "f64" | "float64" -> P (as_float64 packed)
  | "s8" | "int8" -> P (as_int8 packed)
  | "u8" | "uint8" -> P (as_uint8 packed)
  | "s16" | "int16" -> P (as_int16 packed)
  | "u16" | "uint16" -> P (as_uint16 packed)
  | "s32" | "int32" -> P (as_int32 packed)
  | "s64" | "int64" -> P (as_int64 packed)
  | "bool" -> P (as_bool packed)
  | other ->
      failwith (Printf.sprintf "Snapshot_store: unsupported dtype %s" other)

let encode_tree =
  let rec aux prefix tensors scalars = function
    | Snapshot.Tensor (Snapshot.Pack tensor) ->
        let name = if prefix = "" then "root" else prefix in
        let encoded = encode_path name in
        tensors := (encoded, Snapshot.Pack tensor) :: !tensors;
        `Assoc [ ("__tensor__", `String encoded) ]
    | Snapshot.Scalar scalar ->
        let name = if prefix = "" then "root" else prefix in
        let encoded = encode_path name in
        scalars := (encoded, scalar) :: !scalars;
        `Assoc [ ("__scalar__", `String encoded) ]
    | Snapshot.List items ->
        `List
          (List.mapi
             (fun idx item ->
               let child =
                 if prefix = "" then Printf.sprintf "[%d]" idx
                 else Printf.sprintf "%s[%d]" prefix idx
               in
               aux child tensors scalars item)
             items)
    | Snapshot.Record record ->
        `Assoc
          (Snapshot.Record.bindings record
          |> List.map (fun (key, value) ->
                 let child = if prefix = "" then key else prefix ^ "." ^ key in
                 (key, aux child tensors scalars value)))
  in
  fun snapshot ->
    let tensors = ref [] in
    let scalars = ref [] in
    let structure = aux "" tensors scalars snapshot in
    (structure, !tensors, !scalars)

let decode_tree tensor_lookup scalar_lookup =
  let rec aux prefix json =
    match json with
    | `Assoc fields -> (
        let tensor_marker = List.assoc_opt "__tensor__" fields in
        let scalar_marker = List.assoc_opt "__scalar__" fields in
        match (tensor_marker, scalar_marker) with
        | Some (`String encoded), _ -> (
            match tensor_lookup encoded with
            | Some pack -> Snapshot.Tensor pack
            | None ->
                failwith
                  (Printf.sprintf
                     "Snapshot_store.decode_tree: missing tensor %s" encoded))
        | Some _, _ ->
            failwith "Snapshot_store.decode_tree: invalid tensor marker"
        | _, Some (`String encoded) -> (
            match scalar_lookup encoded with
            | Some scalar -> Snapshot.Scalar scalar
            | None ->
                failwith
                  (Printf.sprintf
                     "Snapshot_store.decode_tree: missing scalar %s" encoded))
        | _, Some _ ->
            failwith "Snapshot_store.decode_tree: invalid scalar marker"
        | None, None -> (
            let name = if prefix = "" then "root" else prefix in
            let encoded = encode_path name in
            match scalar_lookup encoded with
            | Some scalar -> Snapshot.Scalar scalar
            | None ->
                let record =
                  List.map
                    (fun (key, value) ->
                      let child =
                        if prefix = "" then key else prefix ^ "." ^ key
                      in
                      (key, aux child value))
                    fields
                in
                Snapshot.record record))
    | `List items ->
        Snapshot.list
          (List.mapi
             (fun idx item ->
               let child =
                 if prefix = "" then Printf.sprintf "[%d]" idx
                 else Printf.sprintf "%s[%d]" prefix idx
               in
               aux child item)
             items)
    | json_scalar -> (
        let name = if prefix = "" then "root" else prefix in
        let encoded = encode_path name in
        match scalar_lookup encoded with
        | Some scalar -> Snapshot.Scalar scalar
        | None -> Snapshot.Scalar (Snapshot.scalar_of_yojson json_scalar))
  in
  aux ""

let save ~base_path snapshot =
  Util.mkdir_p (Filename.dirname base_path);
  let structure_json, tensors, scalars = encode_tree snapshot in
  let tensor_entries =
    List.map
      (fun (encoded, Snapshot.Pack tensor) ->
        let nx_tensor = Rune.to_nx tensor in
        let dtype = Nx.dtype nx_tensor |> dtype_to_string in
        let meta =
          { encoded_path = encoded; dtype; shape = Nx.shape nx_tensor }
        in
        ((encoded, Nx_io.P nx_tensor), meta))
      tensors
  in
  let named_tensors =
    List.map
      (fun ((encoded, packed), _meta) -> (encoded, packed))
      tensor_entries
  in
  let metas = List.map snd tensor_entries in

  let structure_path = base_path ^ ".structure.json" in
  let scalars_path = base_path ^ ".scalars.json" in
  let tensors_path = base_path ^ ".tensors.safetensors" in
  let meta_path = base_path ^ ".tensors.json" in

  Yojson.Basic.to_file structure_path structure_json;
  let scalars_json =
    `Assoc
      (List.map
         (fun (encoded, scalar) -> (encoded, Snapshot.scalar_to_yojson scalar))
         scalars)
  in
  Yojson.Basic.to_file scalars_path scalars_json;
  Yojson.Basic.to_file meta_path (tensors_meta_to_yojson metas);
  Nx_io.save_safetensor ~overwrite:true tensors_path named_tensors

let load ~base_path =
  let open Result in
  let structure_path = base_path ^ ".structure.json" in
  let scalars_path = base_path ^ ".scalars.json" in
  let tensors_path = base_path ^ ".tensors.safetensors" in
  let meta_path = base_path ^ ".tensors.json" in
  try
    let structure_json = Yojson.Basic.from_file structure_path in
    let scalars_json = Yojson.Basic.from_file scalars_path in
    let tensor_meta =
      Yojson.Basic.from_file meta_path |> tensors_meta_of_yojson
    in
    let archive = Nx_io.load_safetensor tensors_path in
    let meta_table = Hashtbl.create (List.length tensor_meta) in
    List.iter
      (fun meta -> Hashtbl.add meta_table meta.encoded_path meta)
      tensor_meta;
    let tensor_lookup encoded =
      match Hashtbl.find_opt archive encoded with
      | None -> None
      | Some packed -> (
          match Hashtbl.find_opt meta_table encoded with
          | None -> None
          | Some { dtype; _ } -> (
              match convert_packed dtype packed with
              | Nx_io.P tensor ->
                  let rune_tensor = Rune.of_nx tensor in
                  Some (Snapshot.Pack rune_tensor)))
    in
    let scalar_lookup encoded =
      match scalars_json with
      | `Assoc entries -> (
          match List.assoc_opt encoded entries with
          | Some json -> Some (Snapshot.scalar_of_yojson json)
          | None -> None)
      | _ -> None
    in
    let snapshot = decode_tree tensor_lookup scalar_lookup structure_json in
    Ok snapshot
  with
  | Sys_error msg -> Error msg
  | Failure msg -> Error msg
