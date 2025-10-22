(* ptree/ml *)

open Rune
module Record = Map.Make (String)

type 'layout t =
  | Tensor of (float, 'layout) Rune.t
  | List of 'layout t list
  | Record of 'layout t Record.t

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

let zeros_like tree = map (fun t -> Rune.zeros (dtype t) (shape t)) tree
let ones_like tree = map (fun t -> Rune.ones (dtype t) (shape t)) tree
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

(* Helper functions for ptree manipulation *)
let split_at n lst =
  let rec aux i acc = function
    | [] -> (List.rev acc, [])
    | h :: t as l ->
        if i = 0 then (List.rev acc, l) else aux (i - 1) (h :: acc) t
  in
  aux n [] lst

let rec flatten : type layout.
    layout t ->
    (float, layout) Rune.t list * ((float, layout) Rune.t list -> layout t) =
  function
  | Tensor t ->
      ( [ t ],
        function
        | [ t' ] -> Tensor t'
        | _ -> failwith "Invalid number of tensors" )
  | List l ->
      let pairs = List.map flatten l in
      let tensors = List.concat (List.map fst pairs) in
      let rebuild =
       fun tensors ->
        let rec aux tensors acc pairs =
          match pairs with
          | [] -> List.rev acc
          | (tensors_pt, rebuild_pt) :: pairs' ->
              let n = List.length tensors_pt in
              let tensors_for_pt, tensors_rest = split_at n tensors in
              let pt' = rebuild_pt tensors_for_pt in
              aux tensors_rest (pt' :: acc) pairs'
        in
        List (aux tensors [] pairs)
      in
      (tensors, rebuild)
  | Record r ->
      (* CRITICAL FIX: Sort record fields to ensure consistent ordering *)
      let sorted_r =
        List.sort
          (fun (k1, _) (k2, _) -> String.compare k1 k2)
          (Record.bindings r)
      in
      let pairs = List.map (fun (k, pt) -> (k, flatten pt)) sorted_r in
      let tensors =
        List.concat (List.map (fun (_, (tensors_pt, _)) -> tensors_pt) pairs)
      in
      let rebuild =
       fun tensors ->
        let rec aux tensors acc pairs =
          match pairs with
          | [] -> List.rev acc
          | (k, (tensors_pt, rebuild_pt)) :: pairs' ->
              let n = List.length tensors_pt in
              let tensors_for_pt, tensors_rest = split_at n tensors in
              let pt' = rebuild_pt tensors_for_pt in
              aux tensors_rest ((k, pt') :: acc) pairs'
        in
        Record (Record.of_list (aux tensors [] pairs))
      in
      (tensors, rebuild)

(* Arithmetic Operations *)

let add tree1 tree2 = map2 Rune.add tree1 tree2
let sub tree1 tree2 = map2 Rune.sub tree1 tree2
let mul tree1 tree2 = map2 Rune.mul tree1 tree2
let div tree1 tree2 = map2 Rune.div tree1 tree2

let scale alpha tree =
  map
    (fun t ->
      let dtype = Rune.dtype t in
      let alpha_t = Rune.scalar dtype alpha in
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

(* Parse a path into components - used by multiple functions *)
let parse_path path =
  let rec parse_components acc remaining =
    if remaining = "" then List.rev acc
    else
      (* Look for next separator: . or [ *)
      let dot_pos = try String.index remaining '.' with Not_found -> -1 in
      let bracket_pos = try String.index remaining '[' with Not_found -> -1 in

      if dot_pos = -1 && bracket_pos = -1 then
        (* No more separators, this is the last component *)
        List.rev (remaining :: acc)
      else if bracket_pos <> -1 && (dot_pos = -1 || bracket_pos < dot_pos) then
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

let unflatten_from_paths pairs =
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
          let grouped =
            List.fold_left
              (fun acc (path, tensor) ->
                match path with
                | index_str :: rest ->
                    let index =
                      int_of_string
                        (String.sub index_str 1 (String.length index_str - 2))
                    in
                    let existing =
                      match List.assoc_opt index acc with
                      | Some items -> items
                      | None -> []
                    in
                    (index, (rest, tensor) :: existing)
                    :: List.remove_assoc index acc
                | _ -> failwith "unflatten_from_paths: invalid list path")
              [] groups
          in
          let sorted_groups =
            List.sort (fun (i1, _) (i2, _) -> compare i1 i2) grouped
          in
          let max_index =
            List.fold_left
              (fun acc (idx, _) -> Stdlib.max acc idx)
              (-1) sorted_groups
          in
          let rec build idx remaining =
            match remaining with
            | [] ->
                if idx > max_index then [] else List [] :: build (idx + 1) []
            | (group_idx, items) :: tl ->
                if idx < group_idx then List [] :: build (idx + 1) remaining
                else
                  let subtree = build_tree (List.rev items) in
                  subtree :: build (idx + 1) tl
          in
          let elements = if max_index < 0 then [] else build 0 sorted_groups in
          List elements
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

(* Path-based accessors *)

let get_by_path path tree =
  let components = parse_path path in
  let rec traverse comps t =
    match comps with
    | [] -> t
    | hd :: tl -> (
        if String.length hd > 0 && hd.[0] = '[' then
          (* List index *)
          let idx_str = String.sub hd 1 (String.length hd - 2) in
          let idx = int_of_string idx_str in
          match t with
          | List items ->
              if idx < 0 || idx >= List.length items then
                invalid_arg
                  (Printf.sprintf
                     "get_by_path: index out of bounds: %s in path %s" hd path);
              traverse tl (List.nth items idx)
          | _ ->
              invalid_arg
                (Printf.sprintf
                   "get_by_path: expected list for index %s in path %s" hd path)
        else
          (* Record key *)
          match t with
          | Record r -> (
              match Record.find_opt hd r with
              | Some sub -> traverse tl sub
              | None ->
                  invalid_arg
                    (Printf.sprintf "get_by_path: key '%s' not found in path %s"
                       hd path))
          | _ ->
              invalid_arg
                (Printf.sprintf
                   "get_by_path: expected record for key '%s' in path %s" hd
                   path))
  in
  traverse components tree

let set_by_path path value tree =
  let components = parse_path path in
  let rec traverse comps t =
    match comps with
    | [] -> value (* Replace at leaf *)
    | hd :: [] -> (
        if String.length hd > 0 && hd.[0] = '[' then
          let idx_str = String.sub hd 1 (String.length hd - 2) in
          let idx = int_of_string idx_str in
          match t with
          | List items ->
              let new_items =
                List.mapi (fun i item -> if i = idx then value else item) items
              in
              List new_items
          | _ ->
              invalid_arg
                (Printf.sprintf
                   "set_by_path: expected list for final index %s in path %s" hd
                   path)
        else
          match t with
          | Record r -> Record (Record.add hd value r)
          | _ ->
              invalid_arg
                (Printf.sprintf
                   "set_by_path: expected record for final key '%s' in path %s"
                   hd path))
    | hd :: tl -> (
        if String.length hd > 0 && hd.[0] = '[' then
          let idx_str = String.sub hd 1 (String.length hd - 2) in
          let idx = int_of_string idx_str in
          match t with
          | List items ->
              if idx < 0 || idx >= List.length items then
                invalid_arg
                  (Printf.sprintf
                     "set_by_path: index out of bounds: %s in path %s" hd path);
              let sub = List.nth items idx in
              let new_sub = traverse tl sub in
              let new_items =
                List.mapi
                  (fun i item -> if i = idx then new_sub else item)
                  items
              in
              List new_items
          | _ ->
              invalid_arg
                (Printf.sprintf
                   "set_by_path: expected list for index %s in path %s" hd path)
        else
          match t with
          | Record r ->
              let sub =
                match Record.find_opt hd r with
                | Some s -> s
                | None -> Record Record.empty (* Create new record if missing *)
              in
              let new_sub = traverse tl sub in
              Record (Record.add hd new_sub r)
          | _ ->
              invalid_arg
                (Printf.sprintf
                   "set_by_path: expected record for key '%s' in path %s" hd
                   path))
  in
  traverse components tree

(* Validation *)

let validate_tree ?(path = "root") tree =
  let rec validate current_path t =
    match t with
    | Tensor _ -> ()
    | List items ->
        if items = [] then
          Printf.fprintf stderr "Warning: Empty list at %s\n" current_path;
        List.iteri
          (fun i sub -> validate (Printf.sprintf "%s[%d]" current_path i) sub)
          items
    | Record r ->
        if Record.is_empty r then
          Printf.fprintf stderr "Warning: Empty record at %s\n" current_path;
        let seen = Hashtbl.create (Record.cardinal r) in
        Record.iter
          (fun k sub ->
            if k = "" then
              failwith (Printf.sprintf "%s: Empty key in record" current_path);
            if
              String.contains k '.' || String.contains k '['
              || String.contains k ']'
            then
              failwith
                (Printf.sprintf
                   "%s: Invalid characters in key '%s' (cannot contain . [ ])"
                   current_path k);
            if Hashtbl.mem seen k then
              failwith (Printf.sprintf "%s: Duplicate key '%s'" current_path k);
            Hashtbl.add seen k ();
            let new_path =
              if current_path = "root" then k
              else Printf.sprintf "%s.%s" current_path k
            in
            validate new_path sub)
          r
  in
  validate path tree

(* Enhanced introspection *)

let list_named_params tree =
  flatten_with_paths tree
  |> List.map (fun (path, t) ->
         let shape = Rune.shape t in
         let shape_str =
           if Array.length shape = 0 then "scalar"
           else
             Array.to_list shape |> List.map string_of_int |> String.concat "×"
         in
         let numel = Rune.numel t in
         (path, shape_str, numel))

let find_params_by_pattern pattern tree =
  flatten_with_paths tree
  |> List.filter (fun (path, _) ->
         try
           let _ = Str.search_forward (Str.regexp pattern) path 0 in
           true
         with Not_found -> false)

let get_param_stats tree =
  let params = flatten_with_paths tree in
  let total_params =
    List.fold_left (fun acc (_, t) -> acc + Rune.numel t) 0 params
  in
  let param_groups =
    List.fold_left
      (fun acc (path, t) ->
        (* Group by top-level key *)
        let top_key =
          try
            let dot_idx = String.index path '.' in
            String.sub path 0 dot_idx
          with Not_found ->
            if String.contains path '[' then
              let bracket_idx = String.index path '[' in
              String.sub path 0 bracket_idx
            else path
        in
        let existing = try List.assoc top_key acc with Not_found -> 0 in
        (top_key, existing + Rune.numel t) :: List.remove_assoc top_key acc)
      [] params
  in
  (total_params, param_groups)
