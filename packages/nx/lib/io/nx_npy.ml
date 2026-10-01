(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Error

let strf = Printf.sprintf

let npy_to_nx (Npy.P (kind, buffer, shape)) =
  Nx.P (Nx.of_buffer kind shape buffer)

(* [with_npy ~by t f] is [f] applied to [t] as a NumPy array over its elements,
   read by [by]. *)
let with_npy ~by t f =
  Storage.reading ~by t (fun b -> f (Npy.P (Nx.dtype t, b, Nx.shape t)))

(* Uniform exception-to-result conversion *)
let wrap_exn f =
  try f () with
  | Npy.Read_error msg -> Error (Format_error msg)
  | Unix.Unix_error (e, _, _) -> Error (Io_error (Unix.error_message e))
  | Sys_error msg -> Error (Io_error msg)
  | Failure msg -> Error (Format_error msg)
  | ex -> Error (Other (Printexc.to_string ex))

(* Npy *)

let load_npy path = wrap_exn @@ fun () -> Ok (npy_to_nx (Npy.read_copy path))

let save_npy ?(overwrite = true) path arr =
  wrap_exn @@ fun () ->
  with_npy ~by:"Nx_io.save_npy" arr @@ fun packed ->
  (if not overwrite then Npy.write ~exclusive:true packed path
   else
     let temp = Temp_file.sibling path in
     match Npy.write packed temp with
     | () -> Temp_file.replace temp path
     | exception exn ->
         Temp_file.remove_if_exists temp;
         raise exn);
  Ok ()

(* Npz *)

let load_npz path =
  wrap_exn @@ fun () ->
  let zi = Zip_archive.open_in path in
  let entries = Zip_archive.npy_entries zi in
  let archive = Hashtbl.create (List.length entries) in
  List.iter
    (fun name ->
      if Hashtbl.mem archive name then
        failwith (strf "duplicate NPZ entry %S" name);
      Hashtbl.add archive name (npy_to_nx (Zip_archive.read_npy zi name)))
    entries;
  Ok archive

let load_npz_entry ~name path =
  wrap_exn @@ fun () ->
  let zi = Zip_archive.open_in path in
  match Zip_archive.read_npy zi name with
  | packed -> Ok (npy_to_nx packed)
  | exception Not_found -> Error (Missing_entry name)

let save_npz ?(overwrite = true) path items =
  wrap_exn @@ fun () ->
  let write ~exclusive output =
    let zo = Zip_archive.open_out ~exclusive output in
    try
      List.iter
        (fun (name, Nx.P nx) ->
          with_npy ~by:"Nx_io.save_npz" nx (Zip_archive.add_npy zo name))
        items;
      Zip_archive.close_out zo
    with exn ->
      Zip_archive.abort_out zo;
      Temp_file.remove_if_exists output;
      raise exn
  in
  (if not overwrite then write ~exclusive:true path
   else
     let temp = Temp_file.sibling path in
     match write ~exclusive:false temp with
     | () -> Temp_file.replace temp path
     | exception exn ->
         Temp_file.remove_if_exists temp;
         raise exn);
  Ok ()
