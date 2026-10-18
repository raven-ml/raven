(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Error

let strf = Printf.sprintf

(* [check_bools ~what p] fails naming [what] if [p] holds bools with a byte
   other than 0 or 1, which nx's operations read with an unspecified result. *)
let check_bools ~what (Npy.P (kind, buffer, _)) =
  match kind with
  | Nx_dtype.Bool ->
      Storage.read_bytes buffer @@ fun bytes ->
      for i = 0 to Bigarray.Array1.dim bytes - 1 do
        let v = Bigarray.Array1.unsafe_get bytes i in
        if v > 1 then fail_msg "%s: the bool at element %d is %d" what i v
      done
  | _ -> ()

let npy_to_nx ~what packed =
  check_bools ~what packed;
  let (Npy.P (kind, buffer, shape)) = packed in
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

let load_npy path =
  wrap_exn @@ fun () -> Ok (npy_to_nx ~what:path (Npy.read_copy path))

let save_npy (type a b) ?(overwrite = true) path (arr : (a, b) Nx.t) =
  match Nx.dtype arr with
  | Bit -> Error (Other ("Nx_io.save_npy: " ^ Npy.no_bit))
  | _ ->
      wrap_exn @@ fun () ->
      with_npy ~by:"Nx_io.save_npy" arr @@ fun packed ->
      Temp_file.write ~overwrite path (Npy.write packed);
      Ok ()

(* Npz *)

let load_npz path =
  wrap_exn @@ fun () ->
  let zi = Zip_archive.open_in path in
  let entries = Zip_archive.npy_entries zi in
  let add archive name =
    if name = "" then failwith "an NPZ entry has an empty name";
    if Archive.mem name archive then
      failwith (strf "duplicate NPZ entry %S" name);
    let what = strf "%s, entry %S" path name in
    Archive.add name (npy_to_nx ~what (Zip_archive.read_npy zi name)) archive
  in
  Ok (List.fold_left add Archive.empty entries)

let load_npz_entry ~name path =
  wrap_exn @@ fun () ->
  let zi = Zip_archive.open_in path in
  match Zip_archive.read_npy zi name with
  | packed -> Ok (npy_to_nx ~what:(strf "%s, entry %S" path name) packed)
  | exception Not_found -> Error (Missing_entry name)

(* [add_npy zo name t] adds [t] to [zo] as entry [name], naming [name] if NPY
   has no dtype for [t]'s. *)
let add_npy zo name (Nx.P t) =
  try with_npy ~by:"Nx_io.save_npz" t (Zip_archive.add_npy zo name)
  with Invalid_argument msg -> failwith (strf "%s: %s" name msg)

let save_npz ?(overwrite = true) path archive =
  wrap_exn @@ fun () ->
  Temp_file.write ~overwrite path (fun fd ->
      let zo = Zip_archive.open_out fd in
      List.iter (fun (name, t) -> add_npy zo name t) (Archive.bindings archive);
      Zip_archive.close_out zo);
  Ok ()
