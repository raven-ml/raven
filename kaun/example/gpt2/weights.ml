open Kaun

(* Weight loading utilities for GPT-2 *)

(* Map HuggingFace/Flax parameter names to our structure *)
let map_param_name name =
  (* Convert from HuggingFace naming convention to our structure *)
  (* Examples:
     transformer/token_embd -> token_emb
     transformer/pos_embd -> pos_emb  
     transformer/block0/attn/c_attn -> blocks[0]/attn/c_attn
     transformer/ln_final -> ln_f
  *)
  let name = String.trim name in

  (* Remove transformer prefix if present *)
  let name =
    if String.starts_with ~prefix:"transformer/" name then
      String.sub name 12 (String.length name - 12)
    else name
  in

  (* Map specific renames *)
  let name =
    match name with
    | "token_embd" -> "token_emb/embeddings"
    | "pos_embd" -> "pos_emb/embeddings"
    | "ln_final" -> "ln_f"
    | s when String.starts_with ~prefix:"block" s ->
        (* Convert block0/... to blocks[0]/... *)
        let rest = String.sub s 5 (String.length s - 5) in
        let block_num =
          let idx = String.index rest '/' in
          String.sub rest 0 idx
        in
        let remainder =
          String.sub rest (String.length block_num)
            (String.length rest - String.length block_num)
        in
        Printf.sprintf "blocks[%s]%s" block_num remainder
    | s -> s
  in
  name

(* Load weights from H5 file *)
let load_h5_weights path =
  Printf.printf "Loading weights from %s\n" path;

  (* Load all datasets from H5 file *)
  let archive = Nx_io.load_h5_all path in

  (* Convert to our parameter structure *)
  let params = Hashtbl.fold (fun k v acc -> (k, v) :: acc) archive [] in
  
  (* Debug: print parameter names *)
  Printf.printf "Found %d parameters:\n" (List.length params);
  List.iter (fun (name, _) -> Printf.printf "  %s\n" name) (List.sort compare params);

  (* Map parameter names and create Ptree *)
  let build_ptree params_list =
    (* Group parameters by prefix *)
    let module String_map = Map.Make (String) in
    let grouped =
      List.fold_left
        (fun acc (name, packed) ->
          let mapped_name = map_param_name name in

          (* Check if it's an array element (e.g., blocks[0]) *)
          if String.contains mapped_name '[' then
            (* Extract array name and index *)
            let bracket_start = String.index mapped_name '[' in
            let bracket_end = String.index mapped_name ']' in
            let array_name = String.sub mapped_name 0 bracket_start in
            let index_str =
              String.sub mapped_name (bracket_start + 1)
                (bracket_end - bracket_start - 1)
            in
            let index = int_of_string index_str in
            let remainder =
              String.sub mapped_name (bracket_end + 1)
                (String.length mapped_name - bracket_end - 1)
            in

            (* Store in map with array info *)
            let key = array_name ^ "[]" in
            let existing = try String_map.find key acc with Not_found -> [] in
            String_map.add key ((index, remainder, packed) :: existing) acc
          else if String.contains mapped_name '/' then
            (* Nested structure *)
            let slash_idx = String.index mapped_name '/' in
            let prefix = String.sub mapped_name 0 slash_idx in
            let suffix =
              String.sub mapped_name (slash_idx + 1)
                (String.length mapped_name - slash_idx - 1)
            in

            let key = prefix ^ "/" in
            let existing = try String_map.find key acc with Not_found -> [] in
            String_map.add key ((0, suffix, packed) :: existing) acc
          else
            (* Direct tensor *)
            String_map.add mapped_name [ (0, "", packed) ] acc)
        String_map.empty params_list
    in

    (* Convert grouped params to Ptree *)
    String_map.fold
      (fun key values acc ->
        if String.ends_with ~suffix:"[]" key then
          (* Array of params *)
          let base_name = String.sub key 0 (String.length key - 2) in
          let sorted =
            List.sort (fun (i1, _, _) (i2, _, _) -> compare i1 i2) values
          in
          let array_items =
            List.map
              (fun (_, suffix, packed) ->
                if suffix = "" then
                  (* Direct tensor *)
                  match packed with
                  | Nx_io.P tensor ->
                      let float_tensor = Nx.cast Nx.float32 tensor in
                      Ptree.Tensor (Rune.of_nx Rune.c float_tensor)
                else
                  (* Nested structure - would need recursive call *)
                  Ptree.record_of [])
              sorted
          in
          (base_name, Ptree.List array_items) :: acc
        else if String.ends_with ~suffix:"/" key then
          (* Nested record *)
          let base_name = String.sub key 0 (String.length key - 1) in
          let nested_params =
            List.map
              (fun (_, suffix, packed) ->
                ( suffix,
                  match packed with
                  | Nx_io.P tensor ->
                      let float_tensor = Nx.cast Nx.float32 tensor in
                      Ptree.Tensor (Rune.of_nx Rune.c float_tensor) ))
              values
          in
          (base_name, Ptree.record_of nested_params) :: acc
        else
          (* Direct tensor *)
          match List.hd values with
          | _, "", packed ->
              ( key,
                match packed with
                | Nx_io.P tensor ->
                    let float_tensor = Nx.cast Nx.float32 tensor in
                    Ptree.Tensor (Rune.of_nx Rune.c float_tensor) )
              :: acc
          | _ -> acc)
      grouped []
  in

  Ptree.record_of (build_ptree params)

(* Load GPT-2 config from JSON *)
let load_config_json path =
  let ic = open_in path in
  let _json_str = really_input_string ic (in_channel_length ic) in
  close_in ic;

  (* Parse JSON - for now, return default config *)
  (* In practice, you'd use a JSON library like yojson *)
  Config.gpt2_small

(* Download weights if not present *)
let download_weights model_name cache_dir =
  let urls =
    [
      ("gpt2", "https://www.dropbox.com/s/0wdgj0gazwt9nm7/gpt2.h5?dl=1");
      ( "gpt2-medium",
        "https://www.dropbox.com/s/nam11kbd83wsm7d/gpt2-medium.h5?dl=1" );
      ( "gpt2-large",
        "https://www.dropbox.com/s/oy8623qwkkjm8gt/gpt2-large.h5?dl=1" );
      ("gpt2-xl", "https://www.dropbox.com/s/6c6qt0bzz4v2afx/gpt2-xl.h5?dl=1");
    ]
  in

  let url = List.assoc model_name urls in
  let target_path = Filename.concat cache_dir (model_name ^ ".h5") in

  if Sys.file_exists target_path then
    Printf.printf "Weights already cached at %s\n" target_path
  else (
    Printf.printf "Downloading %s weights to %s...\n" model_name target_path;
    (* Use curl or wget to download *)
    let cmd = Printf.sprintf "curl -L -o %s '%s'" target_path url in
    let _ = Sys.command cmd in
    Printf.printf "Download complete!\n");

  target_path

(* Main loading function *)
let load_pretrained ?(model_name = "gpt2") ?(cache_dir = "/tmp/gpt2_weights") ()
    =
  (* Ensure cache directory exists *)
  if not (Sys.file_exists cache_dir) then Unix.mkdir cache_dir 0o755;

  (* Download weights if needed *)
  let weights_path = download_weights model_name cache_dir in

  (* Load config (for now use hardcoded) *)
  let config =
    match model_name with
    | "gpt2" -> Config.gpt2_small
    | "gpt2-medium" -> Config.gpt2_medium
    | "gpt2-large" -> Config.gpt2_large
    | "gpt2-xl" -> Config.gpt2_xl
    | _ -> failwith ("Unknown model: " ^ model_name)
  in

  (* Load weights *)
  let params = load_h5_weights weights_path in

  (* Create model *)
  let model = Gpt2.gpt2_lm_head ~config () in

  (model, params, config)
