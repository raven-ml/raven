(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Error
open Packed_nx

let strf = Printf.sprintf
let npy_to_nx (Npy.P (buffer, shape)) = P (Nx.of_buffer buffer ~shape)

(* Uniform exception-to-result conversion *)
let wrap_exn f =
  try f () with
  | Npy.Read_error msg -> Error (Format_error msg)
  | Unix.Unix_error (e, _, _) -> Error (Io_error (Unix.error_message e))
  | Sys_error msg -> Error (Io_error msg)
  | Failure msg -> Error (Format_error msg)
  | ex -> Error (Other (Printexc.to_string ex))

let remove_if_exists path = try Sys.remove path with Sys_error _ -> ()

let temporary_sibling path =
  Filename.temp_file ~temp_dir:(Filename.dirname path)
    (Filename.basename path ^ ".")
    ".tmp"

let replace_with_temp temp path =
  match
    Unix.chmod temp 0o640;
    Unix.rename temp path
  with
  | () -> ()
  | exception exn ->
      remove_if_exists temp;
      raise exn

(* Npy *)

let load_npy path = wrap_exn @@ fun () -> Ok (npy_to_nx (Npy.read_copy path))

let save_npy ?(overwrite = true) path arr =
  wrap_exn @@ fun () ->
  let buf = Nx.to_buffer arr in
  let shape = Nx.shape arr in
  let packed = Npy.P (buf, shape) in
  (if not overwrite then Npy.write ~exclusive:true packed path
   else
     let temp = temporary_sibling path in
     match Npy.write packed temp with
     | () -> replace_with_temp temp path
     | exception exn ->
         remove_if_exists temp;
         raise exn);
  Ok ()

(* Npz *)

let load_npz path =
  wrap_exn @@ fun () ->
  let zi = Zip_archive.open_in path in
  Fun.protect ~finally:(fun () -> Zip_archive.close_in zi) @@ fun () ->
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
  Fun.protect ~finally:(fun () -> Zip_archive.close_in zi) @@ fun () ->
  match Zip_archive.read_npy zi name with
  | packed -> Ok (npy_to_nx packed)
  | exception Not_found -> Error (Missing_entry name)

let save_npz ?(overwrite = true) path items =
  wrap_exn @@ fun () ->
  let write ~exclusive output =
    let zo = Zip_archive.open_out ~exclusive output in
    try
      List.iter
        (fun (name, P nx) ->
          Zip_archive.add_npy zo name (Npy.P (Nx.to_buffer nx, Nx.shape nx)))
        items;
      Zip_archive.close_out zo
    with exn ->
      Zip_archive.abort_out zo;
      remove_if_exists output;
      raise exn
  in
  (if not overwrite then write ~exclusive:true path
   else
     let temp = temporary_sibling path in
     match write ~exclusive:false temp with
     | () -> replace_with_temp temp path
     | exception exn ->
         remove_if_exists temp;
         raise exn);
  Ok ()
