(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Error
open Packed_nx

let strf = Printf.sprintf

(* Convert genarray from Npy (fortran layout) to Nx (c layout) *)
let npy_to_nx (Npy.P ga) =
  let ga = Nx_buffer.genarray_change_layout ga Bigarray.C_layout in
  let shape = Nx_buffer.genarray_dims ga in
  P (Nx.of_buffer (Nx_buffer.of_genarray ga) ~shape)

(* Uniform exception-to-result conversion *)
let wrap_exn f =
  try f () with
  | Npy.Read_error msg -> Error (Format_error msg)
  | Zip.Error (name, func, msg) ->
      Error (Io_error (strf "zip: %s in %s: %s" name func msg))
  | Unix.Unix_error (e, _, _) -> Error (Io_error (Unix.error_message e))
  | Sys_error msg -> Error (Io_error msg)
  | Failure msg -> Error (Format_error msg)
  | ex -> Error (Other (Printexc.to_string ex))

let check_overwrite overwrite path =
  if (not overwrite) && Sys.file_exists path then
    failwith (strf "file already exists: %s" path)

(* Npy *)

let load_npy path = wrap_exn @@ fun () -> Ok (npy_to_nx (Npy.read_copy path))

let save_npy ?(overwrite = true) path arr =
  wrap_exn @@ fun () ->
  check_overwrite overwrite path;
  let buf = Nx.to_buffer arr in
  let shape = Nx.shape arr in
  Npy.write (Nx_buffer.to_genarray buf shape) path;
  Ok ()

(* Npz *)

let load_npz path =
  wrap_exn @@ fun () ->
  let zi = Npy.Npz.open_in path in
  Fun.protect ~finally:(fun () -> Npy.Npz.close_in zi) @@ fun () ->
  let entries = Npy.Npz.entries zi in
  let archive = Hashtbl.create (List.length entries) in
  List.iter
    (fun name -> Hashtbl.add archive name (npy_to_nx (Npy.Npz.read zi name)))
    entries;
  Ok archive

let load_npz_entry ~name path =
  wrap_exn @@ fun () ->
  let zi = Npy.Npz.open_in path in
  Fun.protect ~finally:(fun () -> Npy.Npz.close_in zi) @@ fun () ->
  match Npy.Npz.read zi name with
  | packed -> Ok (npy_to_nx packed)
  | exception Not_found -> Error (Missing_entry name)

let save_npz ?(overwrite = true) path items =
  wrap_exn @@ fun () ->
  check_overwrite overwrite path;
  let zo = Npy.Npz.open_out path in
  Fun.protect ~finally:(fun () -> Npy.Npz.close_out zo) @@ fun () ->
  List.iter
    (fun (name, P nx) ->
      let buf = Nx.to_buffer nx in
      Npy.Npz.write zo name (Nx_buffer.to_genarray buf (Nx.shape nx)))
    items;
  Ok ()
