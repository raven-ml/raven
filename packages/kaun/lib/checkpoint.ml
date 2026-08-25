(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module String_map = Map.Make (String)

type t = Rune.Ptree.tensor String_map.t

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let shape_to_string s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

let full_name ?prefix path =
  match prefix with
  | None -> path
  | Some p -> if path = "" then p else p ^ "." ^ path

let add_entry ~op ?prefix path packed acc =
  let name = full_name ?prefix path in
  if name = "" then invalid_argf "Checkpoint.%s: empty tensor name" op;
  if String_map.mem name acc then
    invalid_argf "Checkpoint.%s: duplicate name %S" op name;
  String_map.add name packed acc

(* Typed recovery: the template leaf carries its dtype and shape, so the packed
   file tensor is witness-checked (or cast) against it. *)
let fetch (type a b) ~op ~cast name (leaf : (a, b) Nx.t) (t : t) : (a, b) Nx.t
    =
  match String_map.find_opt name t with
  | None -> invalid_argf "Checkpoint.%s: missing entry %S" op name
  | Some (Rune.Ptree.P x) -> (
      if Nx.shape x <> Nx.shape leaf then
        invalid_argf
          "Checkpoint.%s: shape mismatch for %S: expected %s, got %s" op name
          (shape_to_string (Nx.shape leaf))
          (shape_to_string (Nx.shape x));
      match Nx_core.Dtype.equal_witness (Nx.dtype x) (Nx.dtype leaf) with
      | Some Type.Equal -> x
      | None ->
          if cast then Nx.cast (Nx.dtype leaf) x
          else
            invalid_argf
              "Checkpoint.%s: dtype mismatch for %S: expected %s, got %s \
               (pass ~cast:true to convert)"
              op name
              (Nx_core.Dtype.to_string (Nx.dtype leaf))
              (Nx_core.Dtype.to_string (Nx.dtype x)))

let empty = String_map.empty

let of_params (module U : Nx.Ptree.Uniform) ?prefix (params : ('a, 'b) Nx.t U.t)
    : t =
  U.fold
    (fun path acc leaf ->
      add_entry ~op:"of_params" ?prefix path (Rune.Ptree.P leaf) acc)
    String_map.empty params

let of_packed (module U : Nx.Ptree.Uniform) ?prefix
    (params : Rune.Ptree.tensor U.t) : t =
  U.fold
    (fun path acc packed -> add_entry ~op:"of_packed" ?prefix path packed acc)
    String_map.empty params

let of_tensor name x =
  if name = "" then invalid_arg "Checkpoint.of_tensor: empty tensor name";
  String_map.singleton name (Rune.Ptree.P x)

let of_int name i =
  if name = "" then invalid_arg "Checkpoint.of_int: empty tensor name";
  if Int32.to_int (Int32.of_int i) <> i then
    invalid_argf "Checkpoint.of_int: %d does not fit in an int32 entry" i;
  String_map.singleton name
    (Rune.Ptree.P (Nx.full Nx.int32 [| 1 |] (Int32.of_int i)))

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

(* The template's names must be distinct and non-empty before they can be
   looked up; a plain fold over the template checks them. *)
let check_names ~op fold_names ?prefix like =
  ignore
    (fold_names
       (fun path acc () -> add_entry ~op ?prefix path () acc)
       String_map.empty like)

let to_params (module U : Nx.Ptree.Uniform) ?prefix ?(cast = false)
    ~(like : ('a, 'b) Nx.t U.t) (t : t) : ('a, 'b) Nx.t U.t =
  check_names ~op:"to_params"
    (fun f acc like -> U.fold (fun path acc _ -> f path acc ()) acc like)
    ?prefix like;
  U.map2
    (fun path leaf -> fetch ~op:"to_params" ~cast (full_name ?prefix path) leaf t)
    (U.names like) like

let to_packed (module U : Nx.Ptree.Uniform) ?prefix ?(cast = false)
    ~(like : Rune.Ptree.tensor U.t) (t : t) : Rune.Ptree.tensor U.t =
  check_names ~op:"to_packed"
    (fun f acc like -> U.fold (fun path acc _ -> f path acc ()) acc like)
    ?prefix like;
  U.map2
    (fun path (Rune.Ptree.P leaf) ->
      Rune.Ptree.P (fetch ~op:"to_packed" ~cast (full_name ?prefix path) leaf t))
    (U.names like) like

let to_int name t =
  match String_map.find_opt name t with
  | None -> invalid_argf "Checkpoint.to_int: no entry named %S" name
  | Some (Rune.Ptree.P x) -> (
      if Nx.numel x <> 1 then
        invalid_argf "Checkpoint.to_int: %S is not a scalar (shape %s)" name
          (shape_to_string (Nx.shape x));
      match Nx_core.Dtype.equal_witness (Nx.dtype x) Nx.int32 with
      | Some Type.Equal -> Int32.to_int (Nx.item [] (Nx.reshape [||] x))
      | None ->
          invalid_argf "Checkpoint.to_int: %S is not an int32 entry (dtype %s)"
            name
            (Nx_core.Dtype.to_string (Nx.dtype x)))

let save path t =
  let entries =
    String_map.fold
      (fun name entry acc ->
        match entry with Rune.Ptree.P x -> (name, Nx_io.P x) :: acc)
      t []
  in
  Nx_io.save_safetensors path entries

let load path =
  let archive = Nx_io.load_safetensors path in
  Hashtbl.fold
    (fun name entry acc ->
      match entry with Nx_io.P x -> String_map.add name (Rune.Ptree.P x) acc)
    archive String_map.empty
