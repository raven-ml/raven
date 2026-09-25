(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module String_map = Map.Make (String)

type t = Nx.packed String_map.t

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let shape_to_string s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

let full_name ?prefix path =
  match prefix with
  | None -> path
  | Some p -> if path = "" then p else p ^ "." ^ path

let entry ~op name t =
  match String_map.find_opt name t with
  | Some entry -> entry
  | None -> invalid_argf "Checkpoint.%s: missing entry %S" op name

let check_shape ~op name ~shape x =
  if Nx.shape x <> shape then
    invalid_argf "Checkpoint.%s: shape mismatch for %S: expected %s, got %s" op
      name (shape_to_string shape)
      (shape_to_string (Nx.shape x))

let typed (type a b) ~op ~shape (dtype : (a, b) Nx.dtype) name t : (a, b) Nx.t =
  let (Nx.P x) = entry ~op name t in
  check_shape ~op name ~shape x;
  match Nx_dtype.equal_witness (Nx.dtype x) dtype with
  | Some Type.Equal -> x
  | None ->
      invalid_argf "Checkpoint.%s: dtype mismatch for %S: expected %s, got %s"
        op name (Nx_dtype.to_string dtype)
        (Nx_dtype.to_string (Nx.dtype x))

let empty = String_map.empty

(* A value's entry names: [path] under [prefix], distinct and non-empty. *)
let namer ~op ?prefix () =
  let seen = Hashtbl.create 64 in
  fun path ->
    let name = full_name ?prefix (Nx.Ptree.Path.to_string path) in
    if name = "" then
      invalid_argf "Checkpoint.%s: a leaf at the root needs ~prefix" op;
    if Hashtbl.mem seen name then
      invalid_argf "Checkpoint.%s: %s: two leaves have this name" op name;
    Hashtbl.add seen name ();
    name

let of_value ?prefix s x =
  let name = namer ~op:"of_value" ?prefix () in
  Nx.Ptree.fold s
    (fun path leaf acc -> String_map.add (name path) (Nx.P leaf) acc)
    x String_map.empty

let of_tensor name x =
  if name = "" then invalid_arg "Checkpoint.of_tensor: empty tensor name";
  String_map.singleton name (Nx.P x)

let of_int name i =
  if name = "" then invalid_arg "Checkpoint.of_int: empty tensor name";
  if Int32.to_int (Int32.of_int i) <> i then
    invalid_argf "Checkpoint.of_int: %d does not fit in an int32 entry" i;
  String_map.singleton name (Nx.P (Nx.full Nx.int32 [| 1 |] (Int32.of_int i)))

let concat ts =
  List.fold_left
    (fun acc t ->
      String_map.union
        (fun name _ _ ->
          invalid_argf "Checkpoint.concat: duplicate name %S" name)
        acc t)
    String_map.empty ts

let names t = List.map fst (String_map.bindings t)
let find name t = String_map.find_opt name t

let get name t =
  match String_map.find_opt name t with
  | Some entry -> entry
  | None -> invalid_argf "Checkpoint.get: no entry named %S" name

let to_tensor ~shape dtype name t = typed ~op:"to_tensor" ~shape dtype name t

let to_float (type b) ~shape (dtype : (float, b) Nx.dtype) name t :
    (float, b) Nx.t =
  let op = "to_float" in
  (match dtype with
  | Float8_e4m3 | Float8_e5m2 ->
      invalid_argf "Checkpoint.%s: %S cannot be cast to %s, which needs scales"
        op name (Nx_dtype.to_string dtype)
  | Float16 | BFloat16 | Float32 | Float64 -> ());
  let (Nx.P x) = entry ~op name t in
  check_shape ~op name ~shape x;
  match Nx.dtype x with
  | Float16 | BFloat16 | Float32 | Float64 -> Nx.cast dtype x
  | Float8_e4m3 | Float8_e5m2 ->
      invalid_argf
        "Checkpoint.%s: %S is a %s entry, whose scales live in other entries: \
         read it with to_tensor"
        op name
        (Nx_dtype.to_string (Nx.dtype x))
  | source ->
      invalid_argf "Checkpoint.%s: %S is not a floating-point entry (dtype %s)"
        op name
        (Nx_dtype.to_string source)

let to_value ?prefix s ~like t =
  let op = "to_value" in
  let name = namer ~op ?prefix () in
  let load (type a b) path (leaf : (a, b) Nx.t) : (a, b) Nx.t =
    let name = name path in
    match String_map.find_opt name t with
    | None ->
        invalid_argf
          "Checkpoint.%s: %s: no entry in the checkpoint, a leaf in the \
           template"
          op name
    | Some (Nx.P x) -> (
        if Nx.shape x <> Nx.shape leaf then
          invalid_argf
            "Checkpoint.%s: %s: shape %s in the checkpoint, %s in the template"
            op name
            (shape_to_string (Nx.shape x))
            (shape_to_string (Nx.shape leaf));
        match Nx_dtype.equal_witness (Nx.dtype x) (Nx.dtype leaf) with
        | Some Type.Equal -> x
        | None ->
            invalid_argf
              "Checkpoint.%s: %s: %s in the checkpoint, %s in the template" op
              name
              (Nx_dtype.to_string (Nx.dtype x))
              (Nx_dtype.to_string (Nx.dtype leaf)))
  in
  Nx.Ptree.map s load like

let to_int name t =
  match String_map.find_opt name t with
  | None -> invalid_argf "Checkpoint.to_int: no entry named %S" name
  | Some (Nx.P x) -> (
      if Nx.numel x <> 1 then
        invalid_argf "Checkpoint.to_int: %S is not a scalar (shape %s)" name
          (shape_to_string (Nx.shape x));
      match Nx_dtype.equal_witness (Nx.dtype x) Nx.int32 with
      | Some Type.Equal -> Int32.to_int (Nx.item [] (Nx.reshape [||] x))
      | None ->
          invalid_argf "Checkpoint.to_int: %S is not an int32 entry (dtype %s)"
            name
            (Nx_dtype.to_string (Nx.dtype x)))

let save path t = Nx_io.save_safetensors path (String_map.bindings t)

let load path =
  let archive = Nx_io.load_safetensors path in
  Hashtbl.fold String_map.add archive String_map.empty
