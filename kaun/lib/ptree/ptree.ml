type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t

type ('layout, 'dev) t =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) t list
  | Record of (string * ('layout, 'dev) t) list

(* Tree Operations *)

let rec map f = function
  | Tensor t -> Tensor (f t)
  | List l -> List (List.map (map f) l)
  | Record r -> Record (List.map (fun (k, v) -> (k, map f v)) r)

let rec map2 f tree1 tree2 =
  match (tree1, tree2) with
  | Tensor t1, Tensor t2 -> Tensor (f t1 t2)
  | List l1, List l2 ->
      if List.length l1 <> List.length l2 then
        invalid_arg "map2: lists have different lengths";
      List (List.map2 (map2 f) l1 l2)
  | Record r1, Record r2 ->
      let rec merge_records r1 r2 =
        match (r1, r2) with
        | [], [] -> []
        | (k1, v1) :: rest1, (k2, v2) :: rest2 ->
            if k1 <> k2 then
              invalid_arg
                (Printf.sprintf "map2: record keys don't match: %s vs %s" k1 k2);
            (k1, map2 f v1 v2) :: merge_records rest1 rest2
        | _ -> invalid_arg "map2: records have different lengths"
      in
      Record (merge_records r1 r2)
  | _ -> invalid_arg "map2: trees have different structures"

let rec iter f = function
  | Tensor t -> f t
  | List l -> List.iter (iter f) l
  | Record r -> List.iter (fun (_, v) -> iter f v) r

let rec fold f acc = function
  | Tensor t -> f acc t
  | List l -> List.fold_left (fold f) acc l
  | Record r -> List.fold_left (fun acc (_, v) -> fold f acc v) acc r

let rec equal_structure tree1 tree2 =
  match (tree1, tree2) with
  | Tensor _, Tensor _ -> true
  | List l1, List l2 ->
      List.length l1 = List.length l2 && List.for_all2 equal_structure l1 l2
  | Record r1, Record r2 ->
      List.length r1 = List.length r2
      && List.for_all2
           (fun (k1, v1) (k2, v2) -> k1 = k2 && equal_structure v1 v2)
           r1 r2
  | _ -> false

(* Tree Construction *)

let zeros_like tree =
  let ctx t = Rune.device t in
  let dtype t = Rune.dtype t in
  let shape t = Rune.shape t in
  map (fun t -> Rune.zeros (ctx t) (dtype t) (shape t)) tree

let ones_like tree =
  let ctx t = Rune.device t in
  let dtype t = Rune.dtype t in
  let shape t = Rune.shape t in
  map (fun t -> Rune.ones (ctx t) (dtype t) (shape t)) tree

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

let to_flat_list tree =
  let rec collect acc = function
    | Tensor t -> t :: acc
    | List l -> List.fold_left collect acc l
    | Record r -> List.fold_left (fun acc (_, v) -> collect acc v) acc r
  in
  List.rev (collect [] tree)

let from_flat_list template tensors =
  let tensors_ref = ref tensors in
  let rec reconstruct = function
    | Tensor _ -> (
        match !tensors_ref with
        | [] -> invalid_arg "from_flat_list: not enough tensors"
        | t :: rest ->
            tensors_ref := rest;
            Tensor t)
    | List l -> List (List.map reconstruct l)
    | Record r -> Record (List.map (fun (k, v) -> (k, reconstruct v)) r)
  in
  let result = reconstruct template in
  match !tensors_ref with
  | [] -> result
  | _ -> invalid_arg "from_flat_list: too many tensors"

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
      List.iteri
        (fun i (key, value) ->
          if i > 0 then Format.fprintf fmt ",@ ";
          Format.fprintf fmt "%s: " key;
          pp fmt value)
        r;
      Format.fprintf fmt "@]}"

let to_string tree = Format.asprintf "%a" pp tree
