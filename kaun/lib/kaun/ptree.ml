(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type tensor = P : ('a, 'layout) Rune.t -> tensor
type t = Tensor of tensor | List of t list | Dict of (string * t) list

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let invalid_arg_ctx ?ctx msg =
  match ctx with
  | None -> invalid_arg msg
  | Some ctx -> invalid_argf "%s: %s" ctx msg

let expected ?ctx what = invalid_arg_ctx ?ctx ("expected " ^ what)

let key_not_found ?ctx key =
  match ctx with
  | None -> invalid_argf "key %S not found" key
  | Some ctx -> invalid_argf "%s: key %S not found" ctx key

let tensor x = Tensor (P x)
let list xs = List xs
let empty = List []

let validate_key key =
  if String.length key = 0 then invalid_arg "empty key";
  for i = 0 to String.length key - 1 do
    let c = String.unsafe_get key i in
    match c with
    | '.' | '[' | ']' ->
        invalid_argf
          "key %S contains reserved character %C (keys must not contain '.', \
           '[', or ']')"
          key c
    | _ -> ()
  done

let dict kvs =
  let tbl = Hashtbl.create (Stdlib.List.length kvs) in
  Stdlib.List.iter
    (fun (k, _) ->
      validate_key k;
      if Hashtbl.mem tbl k then invalid_argf "duplicate key %S" k
      else Hashtbl.add tbl k ())
    kvs;
  Dict kvs

module Tensor = struct
  let dtype (P t) = Nx_core.Dtype.pack (Rune.dtype t)
  let shape (P t) = Rune.shape t
  let numel (P t) = Rune.numel t

  let to_typed (type a l) (dtype : (a, l) Rune.dtype) (P t) :
      (a, l) Rune.t option =
    match Nx_core.Dtype.equal_witness (Rune.dtype t) dtype with
    | Some Type.Equal -> Some t
    | None -> None

  let to_typed_exn (type a l) (dtype : (a, l) Rune.dtype) (P t) : (a, l) Rune.t
      =
    match Nx_core.Dtype.equal_witness (Rune.dtype t) dtype with
    | Some Type.Equal -> t
    | None ->
        invalid_argf "dtype mismatch: expected %s, got %s"
          (Nx_core.Dtype.to_string dtype)
          (Nx_core.Dtype.to_string (Rune.dtype t))
end

module Dict = struct
  type fields = (string * t) list

  let fields_exn ?ctx t =
    match t with Dict kvs -> kvs | _ -> expected ?ctx "Dict"

  let find key fields = Stdlib.List.assoc_opt key fields

  let find_exn ?ctx key fields =
    match find key fields with Some v -> v | None -> key_not_found ?ctx key

  let get_tensor_exn fields ~name dtype =
    match find_exn name fields with
    | Tensor p -> Tensor.to_typed_exn dtype p
    | _ -> invalid_argf "field %S is not a tensor" name
end

module List = struct
  let items_exn ?ctx t =
    match t with List xs -> xs | _ -> expected ?ctx "List"
end

type 'r tensor_handler = { run : 'a 'layout. ('a, 'layout) Rune.t -> 'r }

type map_handler = {
  run : 'a 'layout. ('a, 'layout) Rune.t -> ('a, 'layout) Rune.t;
}

type map2_handler = {
  run :
    'a 'layout.
    ('a, 'layout) Rune.t -> ('a, 'layout) Rune.t -> ('a, 'layout) Rune.t;
}

let with_tensor (P t) (handler : _ tensor_handler) = handler.run t

let as_tensor_exn ?ctx t =
  match t with Tensor p -> p | _ -> expected ?ctx "Tensor"

let map (f : map_handler) t =
  let rec go = function
    | Tensor (P x) -> Tensor (P (f.run x))
    | List xs -> List (Stdlib.List.map go xs)
    | Dict kvs -> Dict (Stdlib.List.map (fun (k, v) -> (k, go v)) kvs)
  in
  go t

let map2 (f : map2_handler) a b =
  let rec go a b =
    match (a, b) with
    | Tensor (P x), Tensor (P y) -> (
        match Nx_core.Dtype.equal_witness (Rune.dtype x) (Rune.dtype y) with
        | Some Type.Equal -> Tensor (P (f.run x y))
        | None -> invalid_arg "dtype mismatch")
    | List xs, List ys ->
        if Stdlib.List.length xs <> Stdlib.List.length ys then
          invalid_arg "list length mismatch";
        List (Stdlib.List.map2 go xs ys)
    | Dict kvs1, Dict kvs2 ->
        if Stdlib.List.length kvs1 <> Stdlib.List.length kvs2 then
          invalid_arg "dict size mismatch";
        Dict
          (Stdlib.List.map
             (fun (k, v1) ->
               match Stdlib.List.assoc_opt k kvs2 with
               | Some v2 -> (k, go v1 v2)
               | None -> invalid_argf "key %S not found in second dict" k)
             kvs1)
    | _ -> invalid_arg "structure mismatch"
  in
  go a b

let iter f t =
  let rec go = function
    | Tensor p -> f p
    | List xs -> Stdlib.List.iter go xs
    | Dict kvs -> Stdlib.List.iter (fun (_, v) -> go v) kvs
  in
  go t

let fold f acc t =
  let rec go acc = function
    | Tensor p -> f acc p
    | List xs -> Stdlib.List.fold_left go acc xs
    | Dict kvs -> Stdlib.List.fold_left (fun acc (_, v) -> go acc v) acc kvs
  in
  go acc t

let flatten t =
  let tensors = ref [] in
  iter (fun p -> tensors := p :: !tensors) t;
  let tensors = Stdlib.List.rev !tensors in
  let rebuild new_tensors =
    let remaining = ref new_tensors in
    let take () =
      match !remaining with
      | [] -> invalid_arg "not enough tensors to rebuild tree"
      | x :: rest ->
          remaining := rest;
          x
    in
    let rec go = function
      | Tensor _ -> Tensor (take ())
      | List xs -> List (Stdlib.List.map go xs)
      | Dict kvs -> Dict (Stdlib.List.map (fun (k, v) -> (k, go v)) kvs)
    in
    let result = go t in
    (match !remaining with
    | [] -> ()
    | _ -> invalid_arg "too many tensors to rebuild tree");
    result
  in
  (tensors, rebuild)

let flatten_with_paths t =
  let join prefix seg = if prefix = "" then seg else prefix ^ "." ^ seg in
  let acc = ref [] in
  let rec go prefix = function
    | Tensor p -> acc := (prefix, p) :: !acc
    | List xs ->
        Stdlib.List.iteri (fun i v -> go (join prefix (string_of_int i)) v) xs
    | Dict kvs -> Stdlib.List.iter (fun (k, v) -> go (join prefix k) v) kvs
  in
  go "" t;
  Stdlib.List.rev !acc

let zeros_like t = map { run = Rune.zeros_like } t
let count_parameters t = fold (fun acc p -> acc + Tensor.numel p) 0 t

let pp_shape shape =
  Stdlib.String.concat "x"
    (Stdlib.Array.to_list (Stdlib.Array.map string_of_int shape))

let rec pp_with_indent indent ppf = function
  | Tensor p ->
      with_tensor p
        {
          run =
            (fun t ->
              Format.fprintf ppf "Tensor(%s, %s)"
                (Nx_core.Dtype.to_string (Rune.dtype t))
                (pp_shape (Rune.shape t)));
        }
  | List [] -> Format.pp_print_string ppf "List []"
  | List xs ->
      let next_indent = indent ^ "  " in
      Format.pp_print_string ppf "List [";
      Stdlib.List.iter
        (fun v ->
          Format.pp_print_char ppf '\n';
          Format.pp_print_string ppf next_indent;
          pp_with_indent next_indent ppf v)
        xs;
      Format.pp_print_char ppf '\n';
      Format.pp_print_string ppf indent;
      Format.pp_print_char ppf ']'
  | Dict [] -> Format.pp_print_string ppf "Dict {}"
  | Dict kvs ->
      let next_indent = indent ^ "  " in
      Format.pp_print_string ppf "Dict {";
      Stdlib.List.iter
        (fun (k, v) ->
          Format.pp_print_char ppf '\n';
          Format.pp_print_string ppf next_indent;
          Format.pp_print_string ppf k;
          Format.pp_print_string ppf ": ";
          pp_with_indent next_indent ppf v)
        kvs;
      Format.pp_print_char ppf '\n';
      Format.pp_print_string ppf indent;
      Format.pp_print_char ppf '}'

let pp ppf t = pp_with_indent "" ppf t
