(* ptree/ptree.ml *)
open Rune
module Record = Map.Make (String)

type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t

type ('layout, 'dev) t =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) t list
  | Record of ('layout, 'dev) t Record.t

type mask_tree =
  | Mask_tensor of bool
  | Mask_list of mask_tree list
  | Mask_record of mask_tree Record.t

let tensor t = Tensor t
let list_of l = List l

let record_of bindings =
  Record
    (List.fold_left (fun m (k, v) -> Record.add k v m) Record.empty bindings)

(* Tree Operations *)

(* Accessors *)
let get_tensor = function Tensor t -> Some t | _ -> None
let get_list = function List l -> Some l | _ -> None
let get_record = function Record r -> Some r | _ -> None

let find_in_record key = function
  | Record r -> Record.find_opt key r
  | _ -> None

let rec map f = function
  | Tensor t -> Tensor (f t)
  | List l -> List (List.map (map f) l)
  | Record r -> Record (Record.map (map f) r)

let rec map2 f tree1 tree2 =
  match (tree1, tree2) with
  | Tensor t1, Tensor t2 -> Tensor (f t1 t2)
  | List l1, List l2 ->
      if List.length l1 <> List.length l2 then
        invalid_arg "map2: lists have different lengths";
      List (List.map2 (map2 f) l1 l2)
  | Record r1, Record r2 ->
      if Record.cardinal r1 <> Record.cardinal r2 then
        invalid_arg "map2: records have different number of fields";
      Record
        (Record.merge
           (fun _k v1 v2 ->
             match (v1, v2) with
             | Some v1', Some v2' -> Some (map2 f v1' v2')
             | _ -> invalid_arg "map2: record keys don't match")
           r1 r2)
  | _ -> invalid_arg "map2: trees have different structures"

let rec zip f tree1 tree2 =
  match (tree1, tree2) with
  | Tensor t1, Tensor t2 -> [ f t1 t2 ]
  | List l1, List l2 ->
      if List.length l1 <> List.length l2 then
        invalid_arg "zip: lists have different lengths";
      List.concat (List.map2 (zip f) l1 l2)
  | Record r1, Record r2 ->
      if Record.cardinal r1 <> Record.cardinal r2 then
        invalid_arg "zip: records have different number of fields";
      let pairs =
        Record.merge
          (fun _k v1 v2 ->
            match (v1, v2) with
            | Some v1', Some v2' -> Some (v1', v2')
            | _ -> invalid_arg "zip: record keys don't match")
          r1 r2
      in
      List.concat
        (Record.bindings pairs |> List.map (fun (_, (v1, v2)) -> zip f v1 v2))
  | _ -> invalid_arg "zip: trees have different structures"

let rec iter f = function
  | Tensor t -> f t
  | List l -> List.iter (iter f) l
  | Record r -> Record.iter (fun _ v -> iter f v) r

let rec fold f acc = function
  | Tensor t -> f acc t
  | List l -> List.fold_left (fold f) acc l
  | Record r -> Record.fold (fun _ v acc' -> fold f acc' v) r acc

let rec equal_structure tree1 tree2 =
  match (tree1, tree2) with
  | Tensor _, Tensor _ -> true
  | List l1, List l2 ->
      List.length l1 = List.length l2 && List.for_all2 equal_structure l1 l2
  | Record r1, Record r2 -> Record.equal equal_structure r1 r2
  | _ -> false

let rec filter pred = function
  | Tensor t -> if pred t then Tensor t else Tensor (Rune.zeros_like t)
  | List l -> List (List.map (filter pred) l)
  | Record r -> Record (Record.map (filter pred) r)

let rec apply_mask mask_tree tree =
  match (mask_tree, tree) with
  | Mask_tensor b, Tensor t ->
      if b then Tensor t else Tensor (Rune.zeros_like t)
  | Mask_list ml, List l ->
      if List.length ml <> List.length l then
        invalid_arg "apply_mask: lists have different lengths";
      List (List.map2 apply_mask ml l)
  | Mask_record mr, Record r ->
      if Record.cardinal mr <> Record.cardinal r then
        invalid_arg "apply_mask: records have different number of fields";
      Record
        (Record.merge
           (fun _k m v ->
             match (m, v) with
             | Some m', Some v' -> Some (apply_mask m' v')
             | _ -> invalid_arg "apply_mask: keys don't match")
           mr r)
  | _ -> invalid_arg "apply_mask: trees have different structures"

(* Tree Construction *)

let zeros_like tree =
  map (fun t -> Rune.zeros (device t) (dtype t) (shape t)) tree

let ones_like tree =
  map (fun t -> Rune.ones (device t) (dtype t) (shape t)) tree

let copy tree = map Rune.copy tree

(* Tree Inspection *)

let count_tensors tree = fold (fun acc _ -> acc + 1) 0 tree

let count_parameters tree =
  fold
    (fun acc t ->
      let shape = Rune.shape t in
      let numel = Array.fold_left ( * ) 1 shape in
      acc + numel)
    0 tree

let flatten_with_rebuild tree =
  let rec collect acc = function
    | Tensor t -> t :: acc
    | List l -> List.fold_left collect acc l
    | Record r -> Record.fold (fun _ v acc -> collect acc v) r acc
  in
  let tensors = List.rev (collect [] tree) in
  let rec rebuild tensors_ref = function
    | Tensor _ -> (
        match !tensors_ref with
        | [] -> invalid_arg "rebuild: not enough tensors"
        | t :: rest ->
            tensors_ref := rest;
            Tensor t)
    | List l -> List (List.map (rebuild tensors_ref) l)
    | Record r -> Record (Record.map (rebuild tensors_ref) r)
  in
  let rebuild_fn new_tensors =
    let tensors_ref = ref new_tensors in
    let result = rebuild tensors_ref tree in
    if !tensors_ref <> [] then invalid_arg "rebuild: too many tensors";
    result
  in
  (tensors, rebuild_fn)

(* Arithmetic Operations *)

let add tree1 tree2 = map2 Rune.add tree1 tree2
let sub tree1 tree2 = map2 Rune.sub tree1 tree2
let mul tree1 tree2 = map2 Rune.mul tree1 tree2
let div tree1 tree2 = map2 Rune.div tree1 tree2

let scale alpha tree =
  map
    (fun t ->
      let ctx = Rune.device t in
      let dtype = Rune.dtype t in
      let alpha_t = Rune.scalar ctx dtype alpha in
      Rune.mul alpha_t t)
    tree

let neg tree = map Rune.neg tree

(* Utility Functions *)

let rec pp fmt = function
  | Tensor t ->
      let shape = Rune.shape t in
      let shape_str =
        shape |> Array.to_list |> List.map string_of_int |> String.concat "x"
      in
      Format.fprintf fmt "Tensor(%s)" shape_str
  | List l ->
      Format.fprintf fmt "[@[<v>";
      List.iteri
        (fun i item ->
          if i > 0 then Format.fprintf fmt ",@ ";
          pp fmt item)
        l;
      Format.fprintf fmt "@]]"
  | Record r ->
      Format.fprintf fmt "{@[<v>";
      let bindings = Record.bindings r in
      List.iteri
        (fun i (key, value) ->
          if i > 0 then Format.fprintf fmt ",@ ";
          Format.fprintf fmt "%s: " key;
          pp fmt value)
        bindings;
      Format.fprintf fmt "@]}"

let to_string tree = Format.asprintf "%a" pp tree

(* Path-based flattening *)
let flatten_with_paths tree =
  let rec flatten_aux prefix = function
    | Tensor t -> [ (prefix, t) ]
    | List items ->
        List.concat
          (List.mapi
             (fun i item -> flatten_aux (Printf.sprintf "%s[%d]" prefix i) item)
             items)
    | Record fields ->
        List.concat
          (Record.bindings fields
          |> List.map (fun (k, v) ->
                 let new_prefix =
                   if prefix = "" then k else Printf.sprintf "%s.%s" prefix k
                 in
                 flatten_aux new_prefix v))
  in
  flatten_aux "" tree

let unflatten_from_paths pairs =
  (* Parse a path into components *)
  let parse_path path =
    let rec parse_components acc remaining =
      if remaining = "" then List.rev acc
      else
        (* Look for next separator: . or [ *)
        let dot_pos = try String.index remaining '.' with Not_found -> -1 in
        let bracket_pos =
          try String.index remaining '[' with Not_found -> -1
        in

        if dot_pos = -1 && bracket_pos = -1 then
          (* No more separators, this is the last component *)
          List.rev (remaining :: acc)
        else if bracket_pos <> -1 && (dot_pos = -1 || bracket_pos < dot_pos)
        then
          (* Found [ first *)
          if bracket_pos = 0 then
            (* Starts with [, this is an array index *)
            let close_pos = String.index remaining ']' in
            let index_str = String.sub remaining 1 (close_pos - 1) in
            let rest =
              String.sub remaining (close_pos + 1)
                (String.length remaining - close_pos - 1)
            in
            let rest =
              if String.length rest > 0 && rest.[0] = '.' then
                String.sub rest 1 (String.length rest - 1)
              else rest
            in
            parse_components (Printf.sprintf "[%s]" index_str :: acc) rest
          else
            (* Field name followed by [ *)
            let field = String.sub remaining 0 bracket_pos in
            let rest =
              String.sub remaining bracket_pos
                (String.length remaining - bracket_pos)
            in
            parse_components (field :: acc) rest
        else
          (* Found . first or only . *)
          let field = String.sub remaining 0 dot_pos in
          let rest =
            String.sub remaining (dot_pos + 1)
              (String.length remaining - dot_pos - 1)
          in
          parse_components (field :: acc) rest
    in
    parse_components [] path
  in

  (* Group paths by their structure *)
  let rec build_tree path_groups =
    match path_groups with
    | [] -> failwith "unflatten_from_paths: empty path group"
    | [ ([], tensor) ] -> Tensor tensor
    | groups ->
        (* Check if this is a list or record *)
        let first_key =
          match List.hd groups with
          | k :: _, _ -> k
          | _ -> failwith "unflatten_from_paths: invalid path structure"
        in

        if String.length first_key > 0 && first_key.[0] = '[' then
          (* This is a list *)
          let items =
            List.map
              (fun (path, tensor) ->
                match path with
                | index_str :: rest ->
                    let index =
                      int_of_string
                        (String.sub index_str 1 (String.length index_str - 2))
                    in
                    (index, (rest, tensor))
                | _ -> failwith "unflatten_from_paths: invalid list path")
              groups
          in
          let sorted_items =
            List.sort (fun (i1, _) (i2, _) -> compare i1 i2) items
          in
          let sub_groups = List.map (fun (_, item) -> item) sorted_items in
          let sub_trees = List.map (fun g -> build_tree [ g ]) sub_groups in
          List sub_trees
        else
          (* This is a record *)
          let record_groups =
            List.fold_left
              (fun acc (path, tensor) ->
                match path with
                | field :: rest ->
                    let existing =
                      try List.assoc field acc with Not_found -> []
                    in
                    (field, (rest, tensor) :: existing)
                    :: List.remove_assoc field acc
                | _ -> failwith "unflatten_from_paths: invalid record path")
              [] groups
          in
          let bindings =
            List.map
              (fun (field, sub_paths) ->
                (field, build_tree (List.rev sub_paths)))
              record_groups
          in
          record_of bindings
  in

  let parsed_pairs =
    List.map (fun (path, tensor) -> (parse_path path, tensor)) pairs
  in
  build_tree parsed_pairs
