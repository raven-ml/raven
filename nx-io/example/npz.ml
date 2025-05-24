open Nx
open Nx_io

let output_dir = "test_data"
let data_dir = "test_data"

let () =
  let npz_arr1 : int32_t = Nx.arange Int32 0 5 1 in
  let npz_arr2 : int32_t = Nx.full Int32 [| 2; 2 |] 42l in
  let items_to_save : (string * packed_nx) list =
    [ ("counts", P npz_arr1); ("matrix", P npz_arr2) ]
  in
  let npz_save_path = Filename.concat output_dir "simple_int32.npz" in
  save_npz items_to_save npz_save_path;
  Printf.printf "Saved int32 arrays ('counts', 'matrix') to: %s\n" npz_save_path;

  (* --- Loading All --- *)
  let npz_load_path = Filename.concat data_dir "archive.npz" in
  let archive : npz_archive = load_npz npz_load_path in
  Printf.printf "Loaded archive: %s\n" npz_load_path;
  Hashtbl.iter
    (fun name packed_arr ->
      (* Extract info from the packed array *)
      match packed_arr with
      | P arr -> (
          let dtype = Nx.dtype arr in
          let shape = Nx.shape arr in
          Format.printf "  Found member: '%s' (kind: %a, shape: %a)\n" name
            Nx.pp_dtype dtype Nx.pp_shape shape;

          (* Example: Unpack 'my_floats' if found *)
          if name = "my_floats" then
            try
              let my_f64 : float64_t = to_float64 packed_arr in
              Printf.printf "    Successfully unpacked 'my_floats' to float64\n";
              ignore my_f64 (* Use it if needed *)
            with Failure msg ->
              Printf.eprintf "    Failed to unpack 'my_floats': %s\n" msg))
    archive;

  (* --- Loading Single Member --- *)
  let member_name = "my_integers" in
  let packed_member : packed_nx =
    load_npz_member ~path:npz_load_path ~name:member_name
  in
  Printf.printf "Loaded single member '%s' from %s\n" member_name npz_load_path;
  let loaded_ints : int32_t = to_int32 packed_member in
  Printf.printf "  Unpacked member '%s' to int32 (shape: [%s])\n" member_name
    (shape loaded_ints |> Array.map string_of_int |> Array.to_list
   |> String.concat "x")
