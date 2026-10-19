(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Archive = Archive

(* NumPy *)

let load_npy path =
  Fail.file path @@ fun () ->
  let file = Storage.open_file path in
  let read pos n = Storage.string file ~pos n in
  Npy.value (Npy.header read ~length:(Rig.Buffer.length file)) file

let save_npy ?(overwrite = true) path x =
  let by = "Nx_io.save_npy" in
  let dt = Nx.dtype x in
  Npy.check ~by dt;
  let header = Npy.encode dt (Nx.shape x) in
  Storage.with_bytes ~by x @@ fun data ->
  Save.write ~overwrite path @@ fun fd ->
  Save.string fd header;
  Save.bytes fd data

let load_npz path =
  Fail.file path @@ fun () ->
  let zip = Zip.open_in (Storage.open_file path) in
  List.fold_left
    (fun archive name ->
      if name = "" then failwith "an entry has an empty name";
      Archive.add name (Zip.read_npy zip name) archive)
    Archive.empty (Zip.npy_entries zip)

let load_npz_entry ~name path =
  Fail.file path @@ fun () ->
  Zip.read_npy (Zip.open_in (Storage.open_file path)) name

let save_npz ?(overwrite = true) path archive =
  let by = "Nx_io.save_npz" in
  let entries = Archive.bindings archive in
  List.iter
    (fun (name, Nx.P x) ->
      if not (Zip.valid_name name) then
        Fail.invalid_argf "%s: %S is no valid entry name" by name;
      Npy.check ~by:(by ^ ": " ^ name) (Nx.dtype x))
    entries;
  Save.write ~overwrite path @@ fun fd ->
  let zip = Zip.open_out fd in
  List.iter
    (fun (name, Nx.P x) ->
      let header = Npy.encode (Nx.dtype x) (Nx.shape x) in
      Storage.with_bytes ~by x (Zip.add_npy zip name header))
    entries;
  Zip.close_out zip

(* SafeTensors *)

let load_safetensors path =
  Fail.file path @@ fun () -> Safetensors.load (Storage.open_file path)

let save_safetensors ?(overwrite = true) path archive =
  let by = "Nx_io.save_safetensors" in
  let entry (name, (Nx.P x as p)) =
    match Safetensors.code (Nx.dtype x) with
    | Some (code, rank) -> (rank, name, code, p)
    | None ->
        Fail.invalid_argf "%s: %s: SafeTensors has no %s" by name
          (Nx.Dtype.name (Nx.dtype x))
  in
  let entries =
    List.map entry (Archive.bindings archive)
    |> List.stable_sort (fun (r, _, _, _) (r', _, _, _) -> Int.compare r' r)
  in
  let header =
    Safetensors.header
      (List.map
         (fun (_, name, code, Nx.P x) -> (name, code, Nx.shape x, Nx.nbytes x))
         entries)
  in
  Save.write ~overwrite path @@ fun fd ->
  Save.string fd header;
  List.iter
    (fun (_, _, _, Nx.P x) -> Storage.with_bytes ~by x (Save.bytes fd))
    entries

(* GGUF *)

module Gguf = Gguf

let load_gguf path =
  Fail.file path @@ fun () -> Gguf.read (Storage.open_file path)

(* Text *)

let load_txt ?(sep = " ") ?(comments = "#") ?(skiprows = 0) ?max_rows path dt =
  Txt.load ~sep ~comments ~skiprows ~max_rows path dt

let save_txt ?(sep = " ") ?(newline = "\n") ?header ?footer ?(comments = "# ")
    ?(overwrite = true) path x =
  Txt.save ~sep ~newline ~header ~footer ~comments ~overwrite path x
