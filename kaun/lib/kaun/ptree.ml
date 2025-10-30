module Dtype = Nx_core.Dtype

type tensor = P : ('a, 'layout) Rune.t -> tensor
type t = Tensor of tensor | List of t list | Dict of (string * t) list

let tensor t = Tensor (P t)
let list items = List items

let dict kvs =
  let tbl = Hashtbl.create (List.length kvs) in
  List.iter
    (fun (k, _) ->
      if Hashtbl.mem tbl k then
        invalid_arg ("Ptree.dict: duplicate key '" ^ k ^ "'")
      else Hashtbl.add tbl k ())
    kvs;
  Dict kvs

module Tensor = struct
  let dtype (P t) = Nx_core.Dtype.pack (Rune.dtype t)
  let shape (P t) = Rune.shape t
  let numel (P t) = Array.fold_left ( * ) 1 (Rune.shape t)

  let to_typed (type a l) (dtype : (a, l) Rune.dtype) (P t) :
      (a, l) Rune.t option =
    match Dtype.equal_witness (Rune.dtype t) dtype with
    | Some Type.Equal -> Some t
    | None -> None

  let to_typed_exn (type a l) (dtype : (a, l) Rune.dtype) (P t) : (a, l) Rune.t
      =
    match Dtype.equal_witness (Rune.dtype t) dtype with
    | Some Type.Equal -> t
    | None -> invalid_arg "Ptree.Tensor.to_typed_exn: dtype mismatch"
end

type 'r tensor_handler = { run : 'a 'layout. ('a, 'layout) Rune.t -> 'r }

let with_tensor (P t) handler = handler.run t

let cast_tensor_using_eq : type a layout b layout'.
    ((a, layout) Dtype.t, (b, layout') Dtype.t) Type.eq ->
    (a, layout) Rune.t ->
    (b, layout') Rune.t =
 fun eq tensor -> match eq with Type.Equal -> tensor

let as_tensor = function Tensor t -> Some t | _ -> None

let as_tensor_exn ?(ctx = "") tree =
  match as_tensor tree with
  | Some t -> t
  | None ->
      failwith
        (Printf.sprintf "Params.as_tensor_exn%s: expected tensor"
           (if ctx = "" then "" else " (" ^ ctx ^ ")"))

let rec map f tree =
  match tree with
  | Tensor tensor ->
      let tensor' =
        with_tensor tensor
          {
            run =
              (fun (type a) (type layout) (t : (a, layout) Rune.t) ->
                let f' =
                  (Obj.magic f : (a, layout) Rune.t -> (a, layout) Rune.t)
                in
                let result = f' t in
                match
                  Dtype.equal_witness (Rune.dtype t) (Rune.dtype result)
                with
                | Some Type.Equal -> P result
                | None -> invalid_arg "Ptree.map: function changed dtype");
          }
      in
      Tensor tensor'
  | List items -> List (List.map (fun item -> map f item) items)
  | Dict bindings -> Dict (List.map (fun (k, v) -> (k, map f v)) bindings)

let rec map2 f lhs rhs =
  match (lhs, rhs) with
  | Tensor lhs_tensor, Tensor rhs_tensor ->
      let tensor =
        with_tensor lhs_tensor
          {
            run =
              (fun (type a) (type layout) (t1 : (a, layout) Rune.t) ->
                with_tensor rhs_tensor
                  {
                    run =
                      (fun (type a')
                        (type layout')
                        (t2 : (a', layout') Rune.t)
                      ->
                        match
                          Dtype.equal_witness (Rune.dtype t1) (Rune.dtype t2)
                        with
                        | Some Type.Equal -> (
                            let f' =
                              (Obj.magic f
                                : (a, layout) Rune.t ->
                                  (a, layout) Rune.t ->
                                  (a, layout) Rune.t)
                            in
                            let result = f' t1 t2 in
                            match
                              Dtype.equal_witness (Rune.dtype t1)
                                (Rune.dtype result)
                            with
                            | Some Type.Equal -> P result
                            | None ->
                                invalid_arg "Ptree.map2: function changed dtype"
                            )
                        | None -> invalid_arg "Ptree.map2: dtype mismatch");
                  });
          }
      in
      Tensor tensor
  | List l_items, List r_items ->
      if List.length l_items <> List.length r_items then
        invalid_arg "Params.map2: list length mismatch";
      List (List.map2 (fun l r -> map2 f l r) l_items r_items)
  | Dict l_bindings, Dict r_bindings ->
      if List.length l_bindings <> List.length r_bindings then
        invalid_arg "Params.map2: dict length mismatch";
      let sorted_l =
        List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) l_bindings
      in
      let sorted_r =
        List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) r_bindings
      in
      let merged =
        List.map2
          (fun (k1, v1) (k2, v2) ->
            if k1 <> k2 then invalid_arg "Params.map2: dict key mismatch";
            (k1, map2 f v1 v2))
          sorted_l sorted_r
      in
      Dict merged
  | _ -> invalid_arg "Params.map2: structure mismatch"

let rec map_packed f tree =
  match tree with
  | Tensor t -> Tensor (f t)
  | List items -> List (List.map (fun item -> map_packed f item) items)
  | Dict bindings ->
      Dict (List.map (fun (k, v) -> (k, map_packed f v)) bindings)

let rec iter f tree =
  match tree with
  | Tensor t -> f t
  | List items -> List.iter (fun item -> iter f item) items
  | Dict bindings -> List.iter (fun (_, v) -> iter f v) bindings

let rec fold f acc tree =
  match tree with
  | Tensor t -> f acc t
  | List items -> List.fold_left (fun a item -> fold f a item) acc items
  | Dict bindings -> List.fold_left (fun a (_, v) -> fold f a v) acc bindings

let flatten tree =
  let rec collect acc = function
    | Tensor t -> t :: acc
    | List items -> List.fold_left collect acc items
    | Dict bindings -> List.fold_left (fun a (_, v) -> collect a v) acc bindings
  in
  let leaves = List.rev (collect [] tree) in
  let rebuild new_leaves =
    let idx = ref 0 in
    let rec aux = function
      | Tensor _ ->
          let t = List.nth new_leaves !idx in
          incr idx;
          Tensor t
      | List items -> List (List.map aux items)
      | Dict bindings -> Dict (List.map (fun (k, v) -> (k, aux v)) bindings)
    in
    aux tree
  in
  (leaves, rebuild)

module Path = struct
  type t = segment list
  and segment = Key of string | Index of int

  let root = []

  let of_string path_str =
    let len = String.length path_str in
    let rec parse i acc =
      if i >= len then List.rev acc
      else
        match path_str.[i] with
        | '.' -> parse (i + 1) acc
        | '[' ->
            let j = String.index_from path_str (i + 1) ']' in
            let idx = int_of_string (String.sub path_str (i + 1) (j - i - 1)) in
            parse (j + 1) (Index idx :: acc)
        | _ ->
            let next_dot =
              try Some (String.index_from path_str i '.')
              with Not_found -> None
            in
            let next_bracket =
              try Some (String.index_from path_str i '[')
              with Not_found -> None
            in
            let next_sep =
              match (next_dot, next_bracket) with
              | None, None -> len
              | Some idx, None -> idx
              | None, Some idx -> idx
              | Some dot_idx, Some bracket_idx -> min dot_idx bracket_idx
            in
            let key = String.sub path_str i (next_sep - i) in
            parse next_sep (Key key :: acc)
    in
    parse 0 []

  let to_string path =
    let buffer = Buffer.create 32 in
    let rec aux first = function
      | [] -> ()
      | Key k :: rest ->
          if not first then Buffer.add_char buffer '.';
          Buffer.add_string buffer k;
          aux false rest
      | Index i :: rest ->
          Buffer.add_char buffer '[';
          Buffer.add_string buffer (string_of_int i);
          Buffer.add_char buffer ']';
          aux false rest
    in
    aux true path;
    Buffer.contents buffer

  let key k p = p @ [ Key k ]
  let index i p = p @ [ Index i ]

  let rec get ~tree = function
    | [] -> Some tree
    | Key k :: rest -> (
        match tree with
        | Dict bindings -> (
            match List.assoc_opt k bindings with
            | Some v -> get ~tree:v rest
            | None -> None)
        | _ -> None)
    | Index i :: rest -> (
        match tree with
        | List items -> (
            match List.nth_opt items i with
            | Some v -> get ~tree:v rest
            | None -> None)
        | _ -> None)

  let placeholder_for = function
    | [] -> List []
    | Key _ :: _ -> Dict []
    | Index _ :: _ -> List []

  let rec set ~tree path ~value =
    match path with
    | [] -> value
    | Key k :: rest ->
        let bindings = match tree with Dict bs -> bs | _ -> [] in
        let rec rebuild acc = function
          | [] ->
              let child =
                if rest = [] then value
                else set ~tree:(placeholder_for rest) rest ~value
              in
              List.rev acc @ [ (k, child) ]
          | ((k', v) as binding) :: tail ->
              if String.equal k k' then
                let child =
                  if rest = [] then value else set ~tree:v rest ~value
                in
                List.rev acc @ ((k, child) :: tail)
              else rebuild (binding :: acc) tail
        in
        Dict (rebuild [] bindings)
    | Index i :: rest ->
        let items = match tree with List xs -> xs | _ -> [] in
        let filler = placeholder_for rest in
        let len = List.length items in
        let padded =
          if i < len then items
          else
            let extra = i - len + 1 in
            items @ List.init extra (fun _ -> filler)
        in
        let rec update idx = function
          | [] -> []
          | x :: xs ->
              if idx = 0 then
                let child =
                  if rest = [] then value else set ~tree:x rest ~value
                in
                child :: xs
              else x :: update (idx - 1) xs
        in
        List (update i padded)

  let update ~tree path ~f =
    match get ~tree path with
    | Some subtree -> set ~tree path ~value:(f subtree)
    | None -> invalid_arg "Params.Path.update: path not found"
end

let get ~path tree = Path.get ~tree path

let get_exn ~path tree =
  match get ~path tree with
  | Some t -> t
  | None ->
      invalid_arg
        (Printf.sprintf "Params.get_exn: path '%s' not found"
           (Path.to_string path))

let set ~path ~value tree = Path.set ~tree path ~value
let update ~path f tree = Path.update ~tree path ~f
let mem ~path tree = Option.is_some (get tree ~path)

let get_tensor ~path tree dtype =
  match get ~path tree with
  | Some (Tensor tensor) ->
      with_tensor tensor
        {
          run =
            (fun (type a) (type layout) (t : (a, layout) Rune.t) ->
              match Dtype.equal_witness (Rune.dtype t) dtype with
              | Some eq ->
                  let coerced = cast_tensor_using_eq eq t in
                  Some coerced
              | None ->
                  invalid_arg
                    (Printf.sprintf "Params.get_tensor: dtype mismatch at '%s'"
                       (Path.to_string path)));
        }
  | _ -> None

let get_tensor_exn ~path tree dtype =
  match get_tensor ~path tree dtype with
  | Some t -> t
  | None ->
      invalid_arg
        (Printf.sprintf "Params.get_tensor_exn: no tensor at '%s'"
           (Path.to_string path))

let flatten_with_paths tree =
  let rec go acc path = function
    | Tensor t -> (path, t) :: acc
    | List xs ->
        let rec loop acc i = function
          | [] -> acc
          | v :: vs ->
              let acc' = go acc (Path.index i path) v in
              loop acc' (i + 1) vs
        in
        loop acc 0 xs
    | Dict kvs ->
        List.fold_left (fun acc (k, v) -> go acc (Path.key k path) v) acc kvs
  in
  List.rev (go [] [] tree)

let filter_tensors tree pred =
  List.filter (fun (p, t) -> pred p t) (flatten_with_paths tree)

type float_dtype = F : (float, 'l) Rune.dtype -> float_dtype

let first_float_dtype tree =
  let rec go = function
    | Tensor (P t) ->
        let dt = Rune.dtype t in
        if Dtype.is_float dt then
          match dt with
          | Dtype.Float16 -> Some (F Dtype.Float16)
          | Dtype.Float32 -> Some (F Dtype.Float32)
          | Dtype.Float64 -> Some (F Dtype.Float64)
          | Dtype.BFloat16 -> Some (F Dtype.BFloat16)
          | Dtype.Float8_e4m3 -> Some (F Dtype.Float8_e4m3)
          | Dtype.Float8_e5m2 -> Some (F Dtype.Float8_e5m2)
          | _ -> None
        else None
    | List xs ->
        let rec find = function
          | [] -> None
          | v :: vs -> ( match go v with Some _ as r -> r | None -> find vs)
        in
        find xs
    | Dict kvs ->
        let rec find = function
          | [] -> None
          | (_, v) :: vs -> (
              match go v with Some _ as r -> r | None -> find vs)
        in
        find kvs
  in
  go tree

let first_float_dtype_exn tree =
  match first_float_dtype tree with
  | Some w -> w
  | None ->
      invalid_arg "Ptree.first_float_dtype_exn: no floating tensors in tree"

let zeros_like tree = map_packed (fun (P t) -> P (Rune.zeros_like t)) tree
let copy tree = map_packed (fun (P t) -> P (Rune.copy t)) tree
let count_tensors tree = fold (fun acc _ -> acc + 1) 0 tree
let count_parameters tree = fold (fun acc t -> acc + Tensor.numel t) 0 tree

module Dict = struct
  type fields = (string * t) list

  let fields_exn ?(ctx = "Ptree.Dict.fields_exn") = function
    | Dict fs -> fs
    | _ -> failwith (ctx ^ ": expected Dict node")

  let find name (fs : fields) = List.assoc_opt name fs

  let find_exn ?(ctx = "Ptree.Dict.find_exn") name (fs : fields) =
    match find name fs with
    | Some v -> v
    | None -> failwith (ctx ^ ": missing field '" ^ name ^ "'")

  let rec set key value (fs : fields) : fields =
    match fs with
    | [] -> [ (key, value) ]
    | (k, _) :: rest when String.equal k key -> (key, value) :: rest
    | binding :: rest -> binding :: set key value rest

  let update f key (fs : fields) =
    match find key fs with
    | Some v -> set key (f v) fs
    | None -> failwith ("Ptree.Dict.update: missing field '" ^ key ^ "'")

  let mem key (fs : fields) = List.exists (fun (k, _) -> String.equal k key) fs

  let get_tensor (fs : fields) ~name dtype =
    (* Make a tiny Dict subtree and delegate to typed path getter. *)
    get_tensor (Dict fs) ~path:(Path.key name []) dtype

  let get_tensor_exn (fs : fields) ~name dtype =
    get_tensor_exn (Dict fs) ~path:(Path.key name []) dtype
end

module List = struct
  let items_exn ?(ctx = "List_.items_exn") = function
    | List xs -> xs
    | _ -> failwith (ctx ^ ": expected List node")
end

let rec pp fmt = function
  | Tensor (P t) ->
      let shape_str =
        String.concat "×"
          (Stdlib.List.map string_of_int (Array.to_list (Rune.shape t)))
      in
      Format.fprintf fmt "Tensor(%s:%s)"
        (if shape_str = "" then "scalar" else shape_str)
        (Dtype.to_string (Rune.dtype t))
  | List items ->
      Format.fprintf fmt "[@[<hov>";
      Stdlib.List.iteri
        (fun i item ->
          if i > 0 then Format.fprintf fmt ",@ ";
          pp fmt item)
        items;
      Format.fprintf fmt "@]]"
  | Dict bindings ->
      Format.fprintf fmt "{@[<hov>";
      Stdlib.List.iteri
        (fun i (k, v) ->
          if i > 0 then Format.fprintf fmt ",@ ";
          Format.fprintf fmt "%s = %a" k pp v)
        bindings;
      Format.fprintf fmt "@]}"

let to_string tree = Format.asprintf "%a" pp tree
