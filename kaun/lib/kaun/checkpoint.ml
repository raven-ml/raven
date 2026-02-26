(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let shape_to_string s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

let save path tree =
  let pairs = Ptree.flatten_with_paths tree in
  let items =
    List.map
      (fun (name, pt) ->
        let nx =
          Ptree.with_tensor pt { run = (fun t -> Nx_io.P (Rune.to_nx t)) }
        in
        (name, nx))
      pairs
  in
  Nx_io.save_safetensors path items

let load path ~like =
  let archive = Nx_io.load_safetensors path in
  let _, rebuild = Ptree.flatten like in
  let path_leaves = Ptree.flatten_with_paths like in
  let loaded =
    List.map
      (fun (name, template) ->
        match Hashtbl.find_opt archive name with
        | None -> invalid_argf "Checkpoint.load: missing key %S" name
        | Some (Nx_io.P nx) ->
            Ptree.with_tensor template
              {
                run =
                  (fun tmpl ->
                    let expected = Rune.shape tmpl in
                    let actual = Nx.shape nx in
                    if expected <> actual then
                      invalid_argf
                        "Checkpoint.load: shape mismatch for %S: expected %s, \
                         got %s"
                        name (shape_to_string expected) (shape_to_string actual);
                    let casted = Nx.cast (Rune.dtype tmpl) nx in
                    Ptree.P (Rune.of_nx casted));
              })
      path_leaves
  in
  rebuild loaded
