open Nx_buffer
open Error
open Packed_nx

let load_npy path =
  try
    match Npy.read_copy path with
    | P genarray ->
        let genarray = Genarray.change_layout genarray c_layout in
        Ok (P (Nx.of_buffer genarray))
  with
  | Unix.Unix_error (e, _, _) -> Error (Io_error (Unix.error_message e))
  | Sys_error msg -> Error (Io_error msg)
  | Failure msg -> Error (Format_error msg)
  | ex -> Error (Other (Printexc.to_string ex))

let save_npy ?(overwrite = true) path arr =
  try
    if (not overwrite) && Sys.file_exists path then
      Error (Io_error (Printf.sprintf "File '%s' already exists" path))
    else
      let genarray = Nx.to_buffer arr in
      Npy.write genarray path;
      Ok ()
  with
  | Unix.Unix_error (e, _, _) -> Error (Io_error (Unix.error_message e))
  | Sys_error msg -> Error (Io_error msg)
  | Failure msg -> Error (Format_error msg)
  | ex -> Error (Other (Printexc.to_string ex))

let load_npz path =
  let zip_in = ref None in
  try
    let archive = Hashtbl.create 16 in
    let zi = Npy.Npz.open_in path in
    zip_in := Some zi;
    let entries = Npy.Npz.entries zi in
    List.iter
      (fun name ->
        match Npy.Npz.read zi name with
        | Npy.P genarray ->
            let genarray = Genarray.change_layout genarray c_layout in
            Hashtbl.add archive name (P (Nx.of_buffer genarray)))
      entries;
    Npy.Npz.close_in zi;
    Ok archive
  with
  | Zip.Error (name, func, msg) ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      Error (Io_error (Printf.sprintf "Zip error: %s in %s: %s" name func msg))
  | Unix.Unix_error (e, _, _) ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      Error (Io_error (Unix.error_message e))
  | Sys_error msg ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      Error (Io_error msg)
  | Failure msg ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      Error (Format_error msg)
  | ex ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      Error (Other (Printexc.to_string ex))

let load_npz_member ~name path =
  let zip_in = ref None in
  try
    let zi = Npy.Npz.open_in path in
    zip_in := Some zi;
    let packed_npy =
      try Npy.Npz.read zi name
      with Not_found ->
        Npy.Npz.close_in zi;
        raise (Failure (Printf.sprintf "Member '%s' not found" name))
    in
    let result =
      match packed_npy with
      | Npy.P genarray ->
          let genarray = Genarray.change_layout genarray c_layout in
          P (Nx.of_buffer genarray)
    in
    Npy.Npz.close_in zi;
    Ok result
  with
  | Zip.Error (zip_name, func, msg) ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      Error
        (Io_error (Printf.sprintf "Zip error: %s in %s: %s" zip_name func msg))
  | Unix.Unix_error (e, _, _) ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      Error (Io_error (Unix.error_message e))
  | Sys_error msg ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      Error (Io_error msg)
  | Failure msg when String.contains msg '\'' ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      Error (Missing_entry name)
  | Failure msg ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      Error (Format_error msg)
  | ex ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      Error (Other (Printexc.to_string ex))

let save_npz ?(overwrite = true) path items =
  try
    if (not overwrite) && Sys.file_exists path then
      Error (Io_error (Printf.sprintf "File '%s' already exists" path))
    else
      let zip_out = ref None in
      try
        let zo = Npy.Npz.open_out path in
        zip_out := Some zo;
        List.iter
          (fun (name, P nx) ->
            let genarray = Nx.to_buffer nx in
            Npy.Npz.write zo name genarray)
          items;
        Npy.Npz.close_out zo;
        Ok ()
      with
      | Zip.Error (name, func, msg) ->
          (match !zip_out with Some zo -> Npy.Npz.close_out zo | None -> ());
          Error
            (Io_error (Printf.sprintf "Zip error: %s in %s: %s" name func msg))
      | Unix.Unix_error (e, _, _) ->
          (match !zip_out with Some zo -> Npy.Npz.close_out zo | None -> ());
          Error (Io_error (Unix.error_message e))
      | Sys_error msg ->
          (match !zip_out with Some zo -> Npy.Npz.close_out zo | None -> ());
          Error (Io_error msg)
      | Failure msg ->
          (match !zip_out with Some zo -> Npy.Npz.close_out zo | None -> ());
          Error (Format_error msg)
      | ex ->
          (match !zip_out with Some zo -> Npy.Npz.close_out zo | None -> ());
          Error (Other (Printexc.to_string ex))
  with ex -> Error (Other (Printexc.to_string ex))
