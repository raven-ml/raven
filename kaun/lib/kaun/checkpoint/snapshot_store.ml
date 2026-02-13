(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Util

type tensor_meta = { encoded_path : string; dtype : string; shape : int array }

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let json_to_string j =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json j with
  | Ok s -> s
  | Error e -> failwith e

let json_of_file path =
  let ic = open_in path in
  let s =
    Fun.protect ~finally:(fun () -> close_in ic) (fun () ->
        really_input_string ic (in_channel_length ic))
  in
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok v -> v
  | Error e -> failwith e

let json_to_file path j =
  let s = json_to_string j in
  let oc = open_out path in
  Fun.protect ~finally:(fun () -> close_out oc) (fun () -> output_string oc s)

let tensors_meta_to_json metas =
  Jsont.Json.list
    (List.map
       (fun { encoded_path; dtype; shape } ->
         json_obj
           [
             ("path", Jsont.Json.string encoded_path);
             ("dtype", Jsont.Json.string dtype);
             ( "shape",
               Jsont.Json.list
                 (Array.to_list shape |> List.map Jsont.Json.int) );
           ])
       metas)

let tensors_meta_of_json json =
  match json with
  | Jsont.Array (entries, _) ->
      let decode_entry = function
        | Jsont.Object (mems, _) -> (
            match
              ( Jsont.Json.find_mem "path" mems,
                Jsont.Json.find_mem "dtype" mems,
                Jsont.Json.find_mem "shape" mems )
            with
            | Some (_, Jsont.String (path, _)),
              Some (_, Jsont.String (dtype, _)),
              Some (_, Jsont.Array (shape, _)) ->
                let shape =
                  List.fold_left
                    (fun acc -> function
                      | Jsont.Number (f, _) -> int_of_float f :: acc
                      | _ ->
                          failwith
                            "Snapshot_store.tensors_meta_of_json: invalid \
                             shape value")
                    [] shape
                  |> List.rev |> Array.of_list
                in
                { encoded_path = path; dtype; shape }
            | _ ->
                failwith "Snapshot_store.tensors_meta_of_json: missing fields"
            )
        | _ -> failwith "Snapshot_store.tensors_meta_of_json: expected object"
      in
      List.map decode_entry entries
  | _ -> failwith "Snapshot_store.tensors_meta_of_json: expected list"

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
        json_obj [ ("__tensor__", Jsont.Json.string encoded) ]
    | Snapshot.Scalar scalar ->
        let name = if prefix = "" then "root" else prefix in
        let encoded = encode_path name in
        scalars := (encoded, scalar) :: !scalars;
        json_obj [ ("__scalar__", Jsont.Json.string encoded) ]
    | Snapshot.List items ->
        Jsont.Json.list
          (List.mapi
             (fun idx item ->
               let child =
                 if prefix = "" then Printf.sprintf "[%d]" idx
                 else Printf.sprintf "%s[%d]" prefix idx
               in
               aux child tensors scalars item)
             items)
    | Snapshot.Record record ->
        json_obj
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
    | Jsont.Object (mems, _) -> (
        let tensor_marker =
          Option.map snd (Jsont.Json.find_mem "__tensor__" mems)
        in
        let scalar_marker =
          Option.map snd (Jsont.Json.find_mem "__scalar__" mems)
        in
        match (tensor_marker, scalar_marker) with
        | Some (Jsont.String (encoded, _)), _ -> (
            match tensor_lookup encoded with
            | Some pack -> Snapshot.Tensor pack
            | None ->
                failwith
                  (Printf.sprintf
                     "Snapshot_store.decode_tree: missing tensor %s" encoded))
        | Some _, _ ->
            failwith "Snapshot_store.decode_tree: invalid tensor marker"
        | _, Some (Jsont.String (encoded, _)) -> (
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
                    (fun ((key, _), value) ->
                      let child =
                        if prefix = "" then key else prefix ^ "." ^ key
                      in
                      (key, aux child value))
                    mems
                in
                Snapshot.record record))
    | Jsont.Array (items, _) ->
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
        | None -> Snapshot.Scalar (Snapshot.scalar_of_json json_scalar))
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

  json_to_file structure_path structure_json;
  let scalars_json =
    json_obj
      (List.map
         (fun (encoded, scalar) -> (encoded, Snapshot.scalar_to_json scalar))
         scalars)
  in
  json_to_file scalars_path scalars_json;
  json_to_file meta_path (tensors_meta_to_json metas);
  Nx_io.save_safetensor ~overwrite:true tensors_path named_tensors

let load ~base_path =
  let open Result in
  let structure_path = base_path ^ ".structure.json" in
  let scalars_path = base_path ^ ".scalars.json" in
  let tensors_path = base_path ^ ".tensors.safetensors" in
  let meta_path = base_path ^ ".tensors.json" in
  try
    let structure_json = json_of_file structure_path in
    let scalars_json = json_of_file scalars_path in
    let tensor_meta =
      json_of_file meta_path |> tensors_meta_of_json
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
      | Jsont.Object (mems, _) -> (
          match Jsont.Json.find_mem encoded mems with
          | Some (_, json) -> Some (Snapshot.scalar_of_json json)
          | None -> None)
      | _ -> None
    in
    let snapshot = decode_tree tensor_lookup scalar_lookup structure_json in
    Ok snapshot
  with
  | Sys_error msg -> Error msg
  | Failure msg -> Error msg
