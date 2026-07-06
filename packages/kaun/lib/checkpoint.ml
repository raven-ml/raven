(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module String_map = Map.Make (String)

type t = Rune.Ptree.tensor String_map.t

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let shape_to_string s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

module type Named = sig
  include Nx.Ptree.S

  val names : t -> string list
end

module Ptree = struct
  include Rune.Ptree

  let names tree =
    let join prefix seg = if prefix = "" then seg else prefix ^ "." ^ seg in
    let rec go prefix acc = function
      | Tensor _ -> prefix :: acc
      | List ts ->
          snd
            (List.fold_left
               (fun (i, acc) v ->
                 (i + 1, go (join prefix (string_of_int i)) acc v))
               (0, acc) ts)
      | Dict kvs ->
          List.fold_left (fun acc (k, v) -> go (join prefix k) acc v) acc kvs
    in
    List.rev (go "" [] tree)
end

(* [leaf_names ~op (module P) ?prefix params] is the full entry name of each
   leaf of [params], in traversal order, validated: one name per leaf, all
   distinct and non-empty. [op] names the calling operation in errors. *)
let leaf_names ~op (module P : Named) ?prefix (params : P.t) : string list =
  let names = P.names params in
  let leaves = ref 0 in
  P.iter (fun _ -> incr leaves) params;
  if List.length names <> !leaves then
    invalid_argf "Checkpoint.%s: %d name(s) for %d tensor leaves" op
      (List.length names) !leaves;
  let full name =
    match prefix with
    | None -> name
    | Some p -> if name = "" then p else p ^ "." ^ name
  in
  let names = List.map full names in
  let _ =
    List.fold_left
      (fun seen name ->
        if name = "" then invalid_argf "Checkpoint.%s: empty tensor name" op;
        if String_map.mem name seen then
          invalid_argf "Checkpoint.%s: duplicate name %S" op name;
        String_map.add name () seen)
      String_map.empty names
  in
  names

let empty = String_map.empty

let of_params (module P : Named) ?prefix (params : P.t) : t =
  let remaining = ref (leaf_names ~op:"of_params" (module P) ?prefix params) in
  let acc = ref String_map.empty in
  P.iter
    (fun leaf ->
      match !remaining with
      | [] -> assert false (* one name per leaf, checked by [leaf_names] *)
      | name :: rest ->
          remaining := rest;
          acc := String_map.add name (Ptree.P leaf) !acc)
    params;
  !acc

let of_tensor name x =
  if name = "" then invalid_arg "Checkpoint.of_tensor: empty tensor name";
  String_map.singleton name (Ptree.P x)

let of_int name i =
  if name = "" then invalid_arg "Checkpoint.of_int: empty tensor name";
  if Int32.to_int (Int32.of_int i) <> i then
    invalid_argf "Checkpoint.of_int: %d does not fit in an int32 entry" i;
  String_map.singleton name
    (Ptree.P (Nx.full Nx.int32 [| 1 |] (Int32.of_int i)))

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

let to_params (module P : Named) ?prefix ?(cast = false) ~(like : P.t) (t : t) :
    P.t =
  let names = leaf_names ~op:"to_params" (module P) ?prefix like in
  (* Pair names with leaves by physical identity, in iteration order: positional
     pairing inside [P.map] would be unsound, since the callback's evaluation
     order is instance-defined (record fields evaluate right to left) while
     [iter] has an explicit sequence. A tensor appearing as several leaves
     queues one name per occurrence. *)
  let queues : (Obj.t * string Queue.t) list ref = ref [] in
  let remaining = ref names in
  P.iter
    (fun leaf ->
      match !remaining with
      | [] -> assert false (* one name per leaf, checked by [leaf_names] *)
      | name :: rest ->
          remaining := rest;
          let key = Obj.repr leaf in
          let queue =
            match List.assq_opt key !queues with
            | Some queue -> queue
            | None ->
                let queue = Queue.create () in
                queues := (key, queue) :: !queues;
                queue
          in
          Queue.add name queue)
    like;
  let name_of leaf =
    match List.assq_opt (Obj.repr leaf) !queues with
    | Some queue when not (Queue.is_empty queue) -> Queue.pop queue
    | _ ->
        invalid_arg "Checkpoint.to_params: map and iter visit different leaves"
  in
  (* Typed recovery: each template leaf carries its dtype, so the packed file
     tensor is witness-checked (or cast) against it. *)
  let fetch (type a b) name (leaf : (a, b) Nx.t) : (a, b) Nx.t =
    match String_map.find_opt name t with
    | None -> invalid_argf "Checkpoint.to_params: missing entry %S" name
    | Some (Ptree.P x) -> (
        if Nx.shape x <> Nx.shape leaf then
          invalid_argf
            "Checkpoint.to_params: shape mismatch for %S: expected %s, got %s"
            name
            (shape_to_string (Nx.shape leaf))
            (shape_to_string (Nx.shape x));
        match Nx_core.Dtype.equal_witness (Nx.dtype x) (Nx.dtype leaf) with
        | Some Type.Equal -> x
        | None ->
            if cast then Nx.cast (Nx.dtype leaf) x
            else
              invalid_argf
                "Checkpoint.to_params: dtype mismatch for %S: expected %s, got \
                 %s (pass ~cast:true to convert)"
                name
                (Nx_core.Dtype.to_string (Nx.dtype leaf))
                (Nx_core.Dtype.to_string (Nx.dtype x)))
  in
  P.map (fun leaf -> fetch (name_of leaf) leaf) like

let to_int name t =
  match String_map.find_opt name t with
  | None -> invalid_argf "Checkpoint.to_int: no entry named %S" name
  | Some (Ptree.P x) -> (
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
        match entry with Ptree.P x -> (name, Nx_io.P x) :: acc)
      t []
  in
  Nx_io.save_safetensors path entries

let load path =
  let archive = Nx_io.load_safetensors path in
  Hashtbl.fold
    (fun name entry acc ->
      match entry with Nx_io.P x -> String_map.add name (Ptree.P x) acc)
    archive String_map.empty
