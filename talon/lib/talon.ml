module Col = struct
  type t =
    | P : ('a, 'b) Nx.dtype * ('a, 'b) Nx.t * bool array option -> t
    | S : string option array -> t
    | B : bool option array -> t

  let normalize_mask = function
    | Some mask when Array.exists Fun.id mask -> Some (Array.copy mask)
    | _ -> None

  let null_mask = function P (_, _, mask) -> mask | _ -> None
  let numeric ?mask dtype tensor = P (dtype, tensor, normalize_mask mask)

  let float32 arr =
    let tensor = Nx.create Nx.float32 [| Array.length arr |] arr in
    numeric Nx.float32 tensor

  let float64 arr =
    let tensor = Nx.create Nx.float64 [| Array.length arr |] arr in
    numeric Nx.float64 tensor

  let int32 arr =
    let tensor = Nx.create Nx.int32 [| Array.length arr |] arr in
    numeric Nx.int32 tensor

  let int64 arr =
    let tensor = Nx.create Nx.int64 [| Array.length arr |] arr in
    numeric Nx.int64 tensor

  let bool arr = B (Array.map (fun x -> Some x) arr)
  let string arr = S (Array.map (fun x -> Some x) arr)

  let float32_opt arr =
    let data = Array.map (fun x -> Option.value x ~default:Float.nan) arr in
    let tensor = Nx.create Nx.float32 [| Array.length data |] data in
    let mask = Array.map Option.is_none arr in
    numeric ~mask Nx.float32 tensor

  let float64_opt arr =
    let data = Array.map (fun x -> Option.value x ~default:Float.nan) arr in
    let tensor = Nx.create Nx.float64 [| Array.length data |] data in
    let mask = Array.map Option.is_none arr in
    numeric ~mask Nx.float64 tensor

  let int32_opt arr =
    let data = Array.map (fun x -> Option.value x ~default:Int32.min_int) arr in
    let tensor = Nx.create Nx.int32 [| Array.length data |] data in
    let mask = Array.map Option.is_none arr in
    numeric ~mask Nx.int32 tensor

  let int64_opt arr =
    let data = Array.map (fun x -> Option.value x ~default:Int64.min_int) arr in
    let tensor = Nx.create Nx.int64 [| Array.length data |] data in
    let mask = Array.map Option.is_none arr in
    numeric ~mask Nx.int64 tensor

  let bool_opt arr = B arr
  let string_opt arr = S arr
  let float32_list lst = float32 (Array.of_list lst)
  let float64_list lst = float64 (Array.of_list lst)
  let int32_list lst = int32 (Array.of_list lst)
  let int64_list lst = int64 (Array.of_list lst)
  let bool_list lst = bool (Array.of_list lst)
  let string_list lst = string (Array.of_list lst)

  let of_tensor t =
    match Nx.shape t with
    | [| _ |] -> numeric (Nx.dtype t) t
    | _ -> invalid_arg "of_tensor: tensor must be 1D"

  let length = function
    | P (_, t, _) -> Nx.size t
    | S arr -> Array.length arr
    | B arr -> Array.length arr

  let has_nulls col =
    match null_mask col with
    | Some mask -> Array.exists Fun.id mask
    | None -> (
        match col with
        | P (Nx.Float32, t, _) ->
            let arr = Nx.to_array t in
            Array.exists (fun x -> classify_float x = FP_nan) arr
        | P (Nx.Float64, t, _) ->
            let arr = Nx.to_array t in
            Array.exists (fun x -> classify_float x = FP_nan) arr
        | P (Nx.Int32, t, _) ->
            let arr = Nx.to_array t in
            Array.exists (fun x -> x = Int32.min_int) arr
        | P (Nx.Int64, t, _) ->
            let arr = Nx.to_array t in
            Array.exists (fun x -> x = Int64.min_int) arr
        | P _ -> false
        | S arr -> Array.exists Option.is_none arr
        | B arr -> Array.exists Option.is_none arr)

  let null_count col =
    match null_mask col with
    | Some mask ->
        Array.fold_left
          (fun acc is_null -> if is_null then acc + 1 else acc)
          0 mask
    | None ->
        if has_nulls col then
          match col with
          | P (Nx.Float32, t, _) ->
              let arr = Nx.to_array t in
              Array.fold_left
                (fun acc x ->
                  if classify_float x = FP_nan then acc + 1 else acc)
                0 arr
          | P (Nx.Float64, t, _) ->
              let arr = Nx.to_array t in
              Array.fold_left
                (fun acc x ->
                  if classify_float x = FP_nan then acc + 1 else acc)
                0 arr
          | P (Nx.Int32, t, _) ->
              let arr = Nx.to_array t in
              Array.fold_left
                (fun acc x -> if x = Int32.min_int then acc + 1 else acc)
                0 arr
          | P (Nx.Int64, t, _) ->
              let arr = Nx.to_array t in
              Array.fold_left
                (fun acc x -> if x = Int64.min_int then acc + 1 else acc)
                0 arr
          | P _ -> 0
          | S arr ->
              Array.fold_left
                (fun acc x -> if Option.is_none x then acc + 1 else acc)
                0 arr
          | B arr ->
              Array.fold_left
                (fun acc x -> if Option.is_none x then acc + 1 else acc)
                0 arr
        else 0

  let drop_nulls col =
    let filter_with_mask arr mask =
      let rec aux idx acc =
        if idx < 0 then acc
        else
          let acc = if mask.(idx) then acc else arr.(idx) :: acc in
          aux (idx - 1) acc
      in
      aux (Array.length arr - 1) []
    in
    match col with
    | P (Nx.Float32, t, mask_opt) ->
        let arr = Nx.to_array t in
        let filtered =
          match mask_opt with
          | Some mask -> filter_with_mask arr mask
          | None ->
              Array.fold_right
                (fun value acc ->
                  if classify_float value = FP_nan then acc else value :: acc)
                arr []
        in
        float32 (Array.of_list filtered)
    | P (Nx.Float64, t, mask_opt) ->
        let arr = Nx.to_array t in
        let filtered =
          match mask_opt with
          | Some mask -> filter_with_mask arr mask
          | None ->
              Array.fold_right
                (fun value acc ->
                  if classify_float value = FP_nan then acc else value :: acc)
                arr []
        in
        float64 (Array.of_list filtered)
    | P (Nx.Int32, t, mask_opt) ->
        let arr = Nx.to_array t in
        let filtered =
          match mask_opt with
          | Some mask -> filter_with_mask arr mask
          | None ->
              Array.fold_right
                (fun value acc ->
                  if value = Int32.min_int then acc else value :: acc)
                arr []
        in
        int32 (Array.of_list filtered)
    | P (Nx.Int64, t, mask_opt) ->
        let arr = Nx.to_array t in
        let filtered =
          match mask_opt with
          | Some mask -> filter_with_mask arr mask
          | None ->
              Array.fold_right
                (fun value acc ->
                  if value = Int64.min_int then acc else value :: acc)
                arr []
        in
        int64 (Array.of_list filtered)
    | P (dtype, t, mask) -> numeric ?mask dtype t
    | S arr ->
        let filtered = Array.to_list arr |> List.filter_map Fun.id in
        string (Array.of_list filtered)
    | B arr ->
        let filtered = Array.to_list arr |> List.filter_map Fun.id in
        bool (Array.of_list filtered)

  let fill_nulls_float32 col ~value =
    match col with
    | P (Nx.Float32, t, mask_opt) ->
        let arr : float array = Nx.to_array t in
        let filled =
          match mask_opt with
          | Some mask ->
              Array.mapi
                (fun i x ->
                  if mask.(i) || classify_float x = FP_nan then value else x)
                arr
          | None ->
              Array.map
                (fun x -> if classify_float x = FP_nan then value else x)
                arr
        in
        float32 filled
    | _ -> invalid_arg "fill_nulls_float32: column must be float32"

  let fill_nulls_float64 col ~value =
    match col with
    | P (Nx.Float64, t, mask_opt) ->
        let arr : float array = Nx.to_array t in
        let filled =
          match mask_opt with
          | Some mask ->
              Array.mapi
                (fun i x ->
                  if mask.(i) || classify_float x = FP_nan then value else x)
                arr
          | None ->
              Array.map
                (fun x -> if classify_float x = FP_nan then value else x)
                arr
        in
        float64 filled
    | _ -> invalid_arg "fill_nulls_float64: column must be float64"

  let fill_nulls_int32 col ~value =
    match col with
    | P (Nx.Int32, t, mask_opt) ->
        let arr : int32 array = Nx.to_array t in
        let filled =
          match mask_opt with
          | Some mask ->
              Array.mapi
                (fun i x -> if mask.(i) || x = Int32.min_int then value else x)
                arr
          | None ->
              Array.map (fun x -> if x = Int32.min_int then value else x) arr
        in
        int32 filled
    | _ -> invalid_arg "fill_nulls_int32: column must be int32"

  let fill_nulls_int64 col ~value =
    match col with
    | P (Nx.Int64, t, mask_opt) ->
        let arr : int64 array = Nx.to_array t in
        let filled =
          match mask_opt with
          | Some mask ->
              Array.mapi
                (fun i x -> if mask.(i) || x = Int64.min_int then value else x)
                arr
          | None ->
              Array.map (fun x -> if x = Int64.min_int then value else x) arr
        in
        int64 filled
    | _ -> invalid_arg "fill_nulls_int64: column must be int64"

  let fill_nulls_string col ~value =
    match col with
    | S arr ->
        let filled = Array.map (fun x -> Option.value x ~default:value) arr in
        string filled
    | _ -> invalid_arg "fill_nulls_string: column must be string"

  let fill_nulls_bool col ~value =
    match col with
    | B arr ->
        let filled = Array.map (fun x -> Option.value x ~default:value) arr in
        bool filled
    | _ -> invalid_arg "fill_nulls_bool: column must be bool"
end

type t = {
  columns : (string * Col.t) list;
  column_map : (string, Col.t) Hashtbl.t;
}

let empty = { columns = []; column_map = Hashtbl.create 0 }

let create pairs =
  if pairs = [] then empty
  else
    let first_length = Col.length (snd (List.hd pairs)) in
    let all_same_length =
      List.for_all (fun (_, col) -> Col.length col = first_length) pairs
    in
    if not all_same_length then
      invalid_arg "create: all columns must have the same length"
    else
      let names = List.map fst pairs in
      let unique_names = List.sort_uniq String.compare names in
      if List.length names <> List.length unique_names then
        invalid_arg "create: duplicate column names"
      else
        let column_map = Hashtbl.create (List.length pairs) in
        List.iter (fun (name, col) -> Hashtbl.add column_map name col) pairs;
        { columns = pairs; column_map }

let of_tensors ?names tensors =
  if tensors = [] then empty
  else
    let first_shape = Nx.shape (List.hd tensors) in
    if Array.length first_shape <> 1 then
      invalid_arg "of_tensors: all tensors must be 1D"
    else
      let all_same_shape =
        List.for_all (fun t -> Nx.shape t = first_shape) tensors
      in
      if not all_same_shape then
        invalid_arg "of_tensors: all tensors must have the same shape"
      else
        let names =
          match names with
          | Some n when List.length n = List.length tensors -> n
          | Some _ -> invalid_arg "of_tensors: wrong number of names"
          | None -> List.mapi (fun i _ -> Printf.sprintf "col%d" i) tensors
        in
        let pairs =
          List.map2 (fun name t -> (name, Col.of_tensor t)) names tensors
        in
        create pairs

let from_nx ?names tensor =
  match Nx.shape tensor with
  | [| _rows; cols |] ->
      let tensors =
        List.init cols (fun col_i -> Nx.slice [ Nx.A; Nx.I col_i ] tensor)
      in
      of_tensors ?names tensors
  | _ -> invalid_arg "from_nx: tensor must be 2D"

let shape t =
  match t.columns with
  | [] -> (0, 0)
  | (_, col) :: _ -> (Col.length col, List.length t.columns)

let num_rows t = fst (shape t)
let num_columns t = snd (shape t)
let column_names t = List.map fst t.columns

let column_types t =
  List.map
    (fun (name, col) ->
      let typ =
        match col with
        | Col.P (Nx.Float32, _, _) -> `Float32
        | Col.P (Nx.Float64, _, _) -> `Float64
        | Col.P (Nx.Int32, _, _) -> `Int32
        | Col.P (Nx.Int64, _, _) -> `Int64
        | Col.P (Nx.UInt8, _, _) -> `Other
        | Col.P _ -> `Other
        | Col.S _ -> `String
        | Col.B _ -> `Bool
      in
      (name, typ))
    t.columns

let is_empty t = num_rows t = 0

module Cols = struct
  let numeric t =
    column_types t
    |> List.filter (function
         | _, (`Float32 | `Float64 | `Int32 | `Int64) -> true
         | _ -> false)
    |> List.map fst

  let float t =
    column_types t
    |> List.filter (function _, (`Float32 | `Float64) -> true | _ -> false)
    |> List.map fst

  let int t =
    column_types t
    |> List.filter (function _, (`Int32 | `Int64) -> true | _ -> false)
    |> List.map fst

  let bool t =
    column_types t
    |> List.filter (function _, `Bool -> true | _ -> false)
    |> List.map fst

  let string t =
    column_types t
    |> List.filter (function _, `String -> true | _ -> false)
    |> List.map fst

  let matching t regex =
    column_names t |> List.filter (fun name -> Re.execp regex name)

  let with_prefix t prefix =
    column_names t |> List.filter (fun name -> String.starts_with ~prefix name)

  let with_suffix t suffix =
    column_names t |> List.filter (fun name -> String.ends_with ~suffix name)

  let except t exclude =
    let is_excluded name = List.mem name exclude in
    List.filter (fun name -> not (is_excluded name)) (column_names t)

  let select_dtypes t types =
    let matches_type typ = function
      | `Numeric -> (
          match typ with
          | `Float32 | `Float64 | `Int32 | `Int64 -> true
          | _ -> false)
      | `Float -> ( match typ with `Float32 | `Float64 -> true | _ -> false)
      | `Int -> ( match typ with `Int32 | `Int64 -> true | _ -> false)
      | `Bool -> typ = `Bool
      | `String -> typ = `String
    in
    column_types t
    |> List.filter (fun (_, typ) ->
           List.exists (fun t -> matches_type typ t) types)
    |> List.map fst
end

let get_column t name = Hashtbl.find_opt t.column_map name

let get_column_exn t name =
  match get_column t name with Some col -> col | None -> raise Not_found

let to_float32_array t name =
  match get_column t name with
  | Some col -> (
      match col with
      | Col.P (Nx.Float32, tensor, _) ->
          let arr : float array = Nx.to_array tensor in
          Some arr
      | _ -> None)
  | None -> None

let to_float64_array t name =
  match get_column t name with
  | Some col -> (
      match col with
      | Col.P (Nx.Float64, tensor, _) ->
          let arr : float array = Nx.to_array tensor in
          Some arr
      | _ -> None)
  | None -> None

let to_int32_array t name =
  match get_column t name with
  | Some col -> (
      match col with
      | Col.P (Nx.Int32, tensor, _) ->
          let arr : int32 array = Nx.to_array tensor in
          Some arr
      | _ -> None)
  | None -> None

let to_int64_array t name =
  match get_column t name with
  | Some col -> (
      match col with
      | Col.P (Nx.Int64, tensor, _) ->
          let arr : int64 array = Nx.to_array tensor in
          Some arr
      | _ -> None)
  | None -> None

let to_bool_array t name =
  match get_column t name with
  | Some (Col.B arr) ->
      let result = Array.map (fun x -> Option.value x ~default:false) arr in
      Some result
  | _ -> None

let to_string_array t name =
  match get_column t name with
  | Some (Col.S arr) ->
      let result = Array.map (fun x -> Option.value x ~default:"") arr in
      Some result
  | _ -> None

(* Option-based array extractors *)
let to_float32_options t name =
  match get_column t name with
  | Some (Col.P (Nx.Float32, tensor, mask_opt)) ->
      let arr : float array = Nx.to_array tensor in
      let result =
        match mask_opt with
        | Some mask ->
            Array.mapi
              (fun i x ->
                if mask.(i) then None
                else if classify_float x = FP_nan then None
                else Some x)
              arr
        | None ->
            Array.map
              (fun x -> if classify_float x = FP_nan then None else Some x)
              arr
      in
      Some result
  | _ -> None

let to_float64_options t name =
  match get_column t name with
  | Some (Col.P (Nx.Float64, tensor, mask_opt)) ->
      let arr : float array = Nx.to_array tensor in
      let result =
        match mask_opt with
        | Some mask ->
            Array.mapi
              (fun i x ->
                if mask.(i) then None
                else if classify_float x = FP_nan then None
                else Some x)
              arr
        | None ->
            Array.map
              (fun x -> if classify_float x = FP_nan then None else Some x)
              arr
      in
      Some result
  | _ -> None

let to_int32_options t name =
  match get_column t name with
  | Some (Col.P (Nx.Int32, tensor, mask_opt)) ->
      let arr : int32 array = Nx.to_array tensor in
      let result =
        match mask_opt with
        | Some mask ->
            Array.mapi
              (fun i x ->
                if mask.(i) then None
                else if x = Int32.min_int then None
                else Some x)
              arr
        | None ->
            Array.map (fun x -> if x = Int32.min_int then None else Some x) arr
      in
      Some result
  | _ -> None

let to_int64_options t name =
  match get_column t name with
  | Some (Col.P (Nx.Int64, tensor, mask_opt)) ->
      let arr : int64 array = Nx.to_array tensor in
      let result =
        match mask_opt with
        | Some mask ->
            Array.mapi
              (fun i x ->
                if mask.(i) then None
                else if x = Int64.min_int then None
                else Some x)
              arr
        | None ->
            Array.map (fun x -> if x = Int64.min_int then None else Some x) arr
      in
      Some result
  | _ -> None

let to_bool_options t name =
  match get_column t name with Some (Col.B arr) -> Some arr | _ -> None

let to_string_options t name =
  match get_column t name with Some (Col.S arr) -> Some arr | _ -> None

let has_column t name = Hashtbl.mem t.column_map name

let add_column t name col =
  let expected_length = num_rows t in
  if expected_length > 0 && Col.length col <> expected_length then
    invalid_arg "add_column: column length doesn't match dataframe rows"
  else
    let columns =
      List.filter (fun (n, _) -> n <> name) t.columns @ [ (name, col) ]
    in
    let column_map = Hashtbl.copy t.column_map in
    Hashtbl.replace column_map name col;
    { columns; column_map }

let drop_column t name =
  let columns = List.filter (fun (n, _) -> n <> name) t.columns in
  let column_map = Hashtbl.copy t.column_map in
  Hashtbl.remove column_map name;
  { columns; column_map }

let drop_columns t names = List.fold_left drop_column t names

let rename_column t ~old_name ~new_name =
  match get_column t old_name with
  | None -> raise Not_found
  | Some _ ->
      if has_column t new_name then
        invalid_arg "rename_column: new_name already exists"
      else
        let columns =
          List.map
            (fun (n, c) -> if n = old_name then (new_name, c) else (n, c))
            t.columns
        in
        let column_map = Hashtbl.create (Hashtbl.length t.column_map) in
        List.iter (fun (name, col) -> Hashtbl.add column_map name col) columns;
        { columns; column_map }

let select t names =
  let columns =
    List.map
      (fun name ->
        match get_column t name with
        | Some col -> (name, col)
        | None -> raise Not_found)
      names
  in
  create columns

let select_loose t names =
  let columns =
    List.filter_map
      (fun name ->
        match get_column t name with
        | Some col -> Some (name, col)
        | None -> None)
      names
  in
  create columns

let reorder_columns t names =
  let requested =
    List.filter_map
      (fun name ->
        match get_column t name with
        | Some col -> Some (name, col)
        | None -> None)
      names
  in
  let remaining =
    List.filter (fun (name, _) -> not (List.mem name names)) t.columns
  in
  create (requested @ remaining)

type 'a row = { f : t -> int -> 'a }

module Row = struct
  let return x = { f = (fun _ _ -> x) }

  let apply ff fx =
    {
      f =
        (fun df i ->
          let f = ff.f df i in
          let x = fx.f df i in
          f x);
    }

  let map x ~f = { f = (fun df i -> f (x.f df i)) }
  let map2 x y ~f = { f = (fun df i -> f (x.f df i) (y.f df i)) }
  let map3 x y z ~f = { f = (fun df i -> f (x.f df i) (y.f df i) (z.f df i)) }
  let both x y = { f = (fun df i -> (x.f df i, y.f df i)) }

  let float32 name =
    let cache : float array option ref = ref None in
    {
      f =
        (fun df i ->
          let arr =
            match !cache with
            | Some a -> a
            | None -> (
                match get_column df name with
                | Some (Col.P (Nx.Float32, tensor, _)) ->
                    let a : float array = Nx.to_array tensor in
                    cache := Some a;
                    a
                | Some _ -> failwith ("Column " ^ name ^ " is not float32")
                | None -> failwith ("Column " ^ name ^ " not found"))
          in
          arr.(i));
    }

  let float64 name =
    let cache : float array option ref = ref None in
    {
      f =
        (fun df i ->
          let arr =
            match !cache with
            | Some a -> a
            | None -> (
                match get_column df name with
                | Some (Col.P (Nx.Float64, tensor, _)) ->
                    let a : float array = Nx.to_array tensor in
                    cache := Some a;
                    a
                | Some _ -> failwith ("Column " ^ name ^ " is not float64")
                | None -> failwith ("Column " ^ name ^ " not found"))
          in
          arr.(i));
    }

  let int32 name =
    let cache : int32 array option ref = ref None in
    {
      f =
        (fun df i ->
          let arr =
            match !cache with
            | Some a -> a
            | None -> (
                match get_column df name with
                | Some (Col.P (Nx.Int32, tensor, _)) ->
                    let a : int32 array = Nx.to_array tensor in
                    cache := Some a;
                    a
                | Some _ -> failwith ("Column " ^ name ^ " is not int32")
                | None -> failwith ("Column " ^ name ^ " not found"))
          in
          arr.(i));
    }

  let int64 name =
    let cache : int64 array option ref = ref None in
    {
      f =
        (fun df i ->
          let arr =
            match !cache with
            | Some a -> a
            | None -> (
                match get_column df name with
                | Some (Col.P (Nx.Int64, tensor, _)) ->
                    let a : int64 array = Nx.to_array tensor in
                    cache := Some a;
                    a
                | Some _ -> failwith ("Column " ^ name ^ " is not int64")
                | None -> failwith ("Column " ^ name ^ " not found"))
          in
          arr.(i));
    }

  let string name =
    let cache : string option array option ref = ref None in
    {
      f =
        (fun df i ->
          let arr =
            match !cache with
            | Some a -> a
            | None -> (
                match get_column df name with
                | Some (Col.S a) ->
                    cache := Some a;
                    a
                | _ -> failwith ("Column " ^ name ^ " is not string"))
          in
          Option.value arr.(i) ~default:"");
    }

  let bool name =
    let cache : bool option array option ref = ref None in
    {
      f =
        (fun df i ->
          let arr =
            match !cache with
            | Some a -> a
            | None -> (
                match get_column df name with
                | Some (Col.B a) ->
                    cache := Some a;
                    a
                | _ -> failwith ("Column " ^ name ^ " is not bool"))
          in
          Option.value arr.(i) ~default:false);
    }

  let number name =
    let cache : float array option ref = ref None in
    {
      f =
        (fun df i ->
          let arr =
            match !cache with
            | Some a -> a
            | None -> (
                match get_column df name with
                | Some (Col.P (Nx.Float32, tensor, _)) ->
                    let a : float array = Nx.to_array tensor in
                    cache := Some a;
                    a
                | Some (Col.P (Nx.Float64, tensor, _)) ->
                    let a : float array = Nx.to_array tensor in
                    cache := Some a;
                    a
                | Some (Col.P (Nx.Int32, tensor, _)) ->
                    let int_arr : int32 array = Nx.to_array tensor in
                    let a = Array.map Int32.to_float int_arr in
                    cache := Some a;
                    a
                | Some (Col.P (Nx.Int64, tensor, _)) ->
                    let int_arr : int64 array = Nx.to_array tensor in
                    let a = Array.map Int64.to_float int_arr in
                    cache := Some a;
                    a
                | Some _ -> failwith ("Column " ^ name ^ " is not numeric")
                | None -> failwith ("Column " ^ name ^ " not found"))
          in
          arr.(i));
    }

  let numbers names = List.map number names
  let index = { f = (fun _ i -> i) }
  let sequence xs = { f = (fun df i -> List.map (fun x -> x.f df i) xs) }
  let all = sequence
  let map_list xs ~f = map (sequence xs) ~f

  let fold_list xs ~init ~f =
    { f = (fun df i -> List.fold_left (fun acc x -> f acc (x.f df i)) init xs) }

  let float32s names = List.map float32 names
  let float64s names = List.map float64 names
  let int32s names = List.map int32 names
  let int64s names = List.map int64 names
  let bools names = List.map bool names
  let strings names = List.map string names

  (* Option-based accessors that return None for null values *)
  let float32_opt name =
    let data_cache : float array option ref = ref None in
    let mask_cache : bool array option ref = ref None in
    {
      f =
        (fun df i ->
          let arr, mask_opt =
            match !data_cache with
            | Some a -> (a, !mask_cache)
            | None -> (
                match get_column df name with
                | Some (Col.P (Nx.Float32, tensor, mask)) ->
                    let a : float array = Nx.to_array tensor in
                    data_cache := Some a;
                    mask_cache := mask;
                    (a, mask)
                | Some _ -> failwith ("Column " ^ name ^ " is not float32")
                | None -> failwith ("Column " ^ name ^ " not found"))
          in
          match mask_opt with
          | Some mask when mask.(i) -> None
          | _ ->
              let value = arr.(i) in
              if classify_float value = FP_nan then None else Some value);
    }

  let float64_opt name =
    let data_cache : float array option ref = ref None in
    let mask_cache : bool array option ref = ref None in
    {
      f =
        (fun df i ->
          let arr, mask_opt =
            match !data_cache with
            | Some a -> (a, !mask_cache)
            | None -> (
                match get_column df name with
                | Some (Col.P (Nx.Float64, tensor, mask)) ->
                    let a : float array = Nx.to_array tensor in
                    data_cache := Some a;
                    mask_cache := mask;
                    (a, mask)
                | Some _ -> failwith ("Column " ^ name ^ " is not float64")
                | None -> failwith ("Column " ^ name ^ " not found"))
          in
          match mask_opt with
          | Some mask when mask.(i) -> None
          | _ ->
              let value = arr.(i) in
              if classify_float value = FP_nan then None else Some value);
    }

  let int32_opt name =
    let data_cache : int32 array option ref = ref None in
    let mask_cache : bool array option ref = ref None in
    {
      f =
        (fun df i ->
          let arr, mask_opt =
            match !data_cache with
            | Some a -> (a, !mask_cache)
            | None -> (
                match get_column df name with
                | Some (Col.P (Nx.Int32, tensor, mask)) ->
                    let a : int32 array = Nx.to_array tensor in
                    data_cache := Some a;
                    mask_cache := mask;
                    (a, mask)
                | Some _ -> failwith ("Column " ^ name ^ " is not int32")
                | None -> failwith ("Column " ^ name ^ " not found"))
          in
          match mask_opt with
          | Some mask when mask.(i) -> None
          | _ ->
              let value = arr.(i) in
              if value = Int32.min_int then None else Some value);
    }

  let int64_opt name =
    let data_cache : int64 array option ref = ref None in
    let mask_cache : bool array option ref = ref None in
    {
      f =
        (fun df i ->
          let arr, mask_opt =
            match !data_cache with
            | Some a -> (a, !mask_cache)
            | None -> (
                match get_column df name with
                | Some (Col.P (Nx.Int64, tensor, mask)) ->
                    let a : int64 array = Nx.to_array tensor in
                    data_cache := Some a;
                    mask_cache := mask;
                    (a, mask)
                | Some _ -> failwith ("Column " ^ name ^ " is not int64")
                | None -> failwith ("Column " ^ name ^ " not found"))
          in
          match mask_opt with
          | Some mask when mask.(i) -> None
          | _ ->
              let value = arr.(i) in
              if value = Int64.min_int then None else Some value);
    }

  let string_opt name =
    let cache : string option array option ref = ref None in
    {
      f =
        (fun df i ->
          let arr =
            match !cache with
            | Some a -> a
            | None -> (
                match get_column df name with
                | Some (Col.S a) ->
                    cache := Some a;
                    a
                | _ -> failwith ("Column " ^ name ^ " is not string"))
          in
          arr.(i));
    }

  let bool_opt name =
    let cache : bool option array option ref = ref None in
    {
      f =
        (fun df i ->
          let arr =
            match !cache with
            | Some a -> a
            | None -> (
                match get_column df name with
                | Some (Col.B a) ->
                    cache := Some a;
                    a
                | _ -> failwith ("Column " ^ name ^ " is not bool"))
          in
          arr.(i));
    }

  let float32s_opt names = List.map float32_opt names
  let float64s_opt names = List.map float64_opt names
  let int32s_opt names = List.map int32_opt names
  let int64s_opt names = List.map int64_opt names
  let bools_opt names = List.map bool_opt names
  let strings_opt names = List.map string_opt names

  module Agg = struct
    (* Vectorized row-wise aggregations using Nx operations *)

    (* Helper to collect numeric columns and cast to float64, applying null
       masks *)
    let collect_as_float64 t names : (float, Bigarray.float64_elt) Nx.t list =
      List.fold_left
        (fun (acc : (float, Bigarray.float64_elt) Nx.t list) name ->
          match get_column t name with
          | Some (Col.P (Nx.Float64, tensor, mask_opt)) ->
              let result =
                match mask_opt with
                | Some mask ->
                    (* Apply mask: set masked entries to NaN *)
                    let mask_tensor =
                      Nx.create Nx.uint8
                        [| Array.length mask |]
                        (Array.map (fun b -> if b then 1 else 0) mask)
                    in
                    let mask_float = Nx.cast Nx.float64 mask_tensor in
                    let nan_tensor = Nx.full_like tensor Float.nan in
                    Nx.where (Nx.cast Nx.uint8 mask_float) nan_tensor tensor
                | None -> tensor
              in
              result :: acc
          | Some (Col.P (Nx.Float32, tensor, mask_opt)) ->
              let casted = Nx.cast Nx.float64 tensor in
              let result =
                match mask_opt with
                | Some mask ->
                    let mask_tensor =
                      Nx.create Nx.uint8
                        [| Array.length mask |]
                        (Array.map (fun b -> if b then 1 else 0) mask)
                    in
                    let mask_float = Nx.cast Nx.float64 mask_tensor in
                    let nan_tensor = Nx.full_like casted Float.nan in
                    Nx.where (Nx.cast Nx.uint8 mask_float) nan_tensor casted
                | None -> casted
              in
              result :: acc
          | Some (Col.P (Nx.Int32, tensor, mask_opt)) ->
              let casted = Nx.cast Nx.float64 tensor in
              let result =
                match mask_opt with
                | Some mask ->
                    let mask_tensor =
                      Nx.create Nx.uint8
                        [| Array.length mask |]
                        (Array.map (fun b -> if b then 1 else 0) mask)
                    in
                    let mask_float = Nx.cast Nx.float64 mask_tensor in
                    let nan_tensor = Nx.full_like casted Float.nan in
                    Nx.where (Nx.cast Nx.uint8 mask_float) nan_tensor casted
                | None -> casted
              in
              result :: acc
          | Some (Col.P (Nx.Int64, tensor, mask_opt)) ->
              let casted = Nx.cast Nx.float64 tensor in
              let result =
                match mask_opt with
                | Some mask ->
                    let mask_tensor =
                      Nx.create Nx.uint8
                        [| Array.length mask |]
                        (Array.map (fun b -> if b then 1 else 0) mask)
                    in
                    let mask_float = Nx.cast Nx.float64 mask_tensor in
                    let nan_tensor = Nx.full_like casted Float.nan in
                    Nx.where (Nx.cast Nx.uint8 mask_float) nan_tensor casted
                | None -> casted
              in
              result :: acc
          | Some (Col.P (_, _, _)) -> acc (* Skip other types *)
          | _ -> acc)
        [] names
      |> List.rev

    let sum ?(skipna = true) t ~names =
      let tensors = collect_as_float64 t names in
      if tensors = [] then Col.float64 (Array.make (num_rows t) 0.0)
      else
        (* Stack tensors as rows and sum along axis 0 *)
        let stacked = Nx.stack tensors ~axis:0 in
        let result =
          if skipna then
            (* Replace NaN with 0 for sum *)
            let nan_mask = Nx.isnan stacked in
            let zeros = Nx.zeros_like stacked in
            let cleaned = Nx.where nan_mask zeros stacked in
            Nx.sum cleaned ~axes:[ 0 ]
          else Nx.sum stacked ~axes:[ 0 ]
        in
        Col.of_tensor result

    let mean ?(skipna = true) t ~names =
      let tensors = collect_as_float64 t names in
      if tensors = [] then Col.float64 (Array.make (num_rows t) Float.nan)
      else
        let stacked = Nx.stack tensors ~axis:0 in
        let result =
          if skipna then
            (* Count non-NaN values per position *)
            let nan_mask = Nx.isnan stacked in
            let zeros = Nx.zeros_like stacked in
            let cleaned = Nx.where nan_mask zeros stacked in
            let ones = Nx.ones_like stacked in
            let valid_mask = Nx.where nan_mask zeros ones in
            let sum = Nx.sum cleaned ~axes:[ 0 ] in
            let count = Nx.sum valid_mask ~axes:[ 0 ] in
            (* Avoid division by zero *)
            let safe_count = Nx.maximum count (Nx.ones_like count) in
            let mean = Nx.div sum safe_count in
            (* Set to NaN where all values were NaN *)
            let all_nan = Nx.equal count (Nx.zeros_like count) in
            Nx.where all_nan (Nx.full_like mean Float.nan) mean
          else Nx.mean stacked ~axes:[ 0 ]
        in
        Col.of_tensor result

    let min ?(skipna = true) t ~names =
      let tensors = collect_as_float64 t names in
      if tensors = [] then Col.float64 (Array.make (num_rows t) Float.nan)
      else
        let stacked = Nx.stack tensors ~axis:0 in
        let result =
          if skipna then
            (* Replace NaN with +inf for min *)
            let nan_mask = Nx.isnan stacked in
            let inf = Nx.full_like stacked Float.infinity in
            let cleaned = Nx.where nan_mask inf stacked in
            let min_vals = Nx.min cleaned ~axes:[ 0 ] in
            (* Set to NaN where all values were NaN *)
            let is_inf =
              Nx.equal min_vals (Nx.full_like min_vals Float.infinity)
            in
            Nx.where is_inf (Nx.full_like min_vals Float.nan) min_vals
          else Nx.min stacked ~axes:[ 0 ]
        in
        Col.of_tensor result

    let max ?(skipna = true) t ~names =
      let tensors = collect_as_float64 t names in
      if tensors = [] then Col.float64 (Array.make (num_rows t) Float.nan)
      else
        let stacked = Nx.stack tensors ~axis:0 in
        let result =
          if skipna then
            (* Replace NaN with -inf for max *)
            let nan_mask = Nx.isnan stacked in
            let neg_inf = Nx.full_like stacked Float.neg_infinity in
            let cleaned = Nx.where nan_mask neg_inf stacked in
            let max_vals = Nx.max cleaned ~axes:[ 0 ] in
            (* Set to NaN where all values were NaN *)
            let is_neg_inf =
              Nx.equal max_vals (Nx.full_like max_vals Float.neg_infinity)
            in
            Nx.where is_neg_inf (Nx.full_like max_vals Float.nan) max_vals
          else Nx.max stacked ~axes:[ 0 ]
        in
        Col.of_tensor result

    let dot t ~names ~weights =
      if List.length names <> Array.length weights then
        failwith "Number of columns must match number of weights";
      let tensors = collect_as_float64 t names in
      if tensors = [] then Col.float64 (Array.make (num_rows t) 0.0)
      else
        (* Stack tensors as rows, multiply by weights, and sum *)
        let stacked = Nx.stack tensors ~axis:0 in
        let weights_tensor =
          Nx.create Nx.float64 [| Array.length weights; 1 |] weights
        in
        (* Transpose stacked to [n_rows, n_cols], then matrix multiply *)
        let transposed = Nx.transpose stacked ~axes:[ 1; 0 ] in
        let result = Nx.matmul transposed weights_tensor in
        (* Squeeze to remove the extra dimension *)
        let squeezed = Nx.squeeze result ~axes:[ 1 ] in
        Col.of_tensor squeezed

    let all t ~names =
      let cols =
        List.filter_map
          (fun name ->
            match get_column t name with
            | Some (Col.B arr) -> Some arr
            | _ -> None)
          names
      in
      if cols = [] then failwith "No boolean columns found"
      else
        let n_rows = num_rows t in
        let result =
          Array.init n_rows (fun i ->
              Some
                (List.for_all
                   (fun arr -> Option.value arr.(i) ~default:false)
                   cols))
        in
        Col.B result

    let any t ~names =
      let cols =
        List.filter_map
          (fun name ->
            match get_column t name with
            | Some (Col.B arr) -> Some arr
            | _ -> None)
          names
      in
      if cols = [] then failwith "No boolean columns found"
      else
        let n_rows = num_rows t in
        let result =
          Array.init n_rows (fun i ->
              Some
                (List.exists
                   (fun arr -> Option.value arr.(i) ~default:false)
                   cols))
        in
        Col.B result
  end
end

let head ?(n = 5) t =
  let n_rows = num_rows t in
  let actual_n = min n n_rows in
  let columns =
    List.map
      (fun (name, col) ->
        match col with
        | Col.P (dtype, tensor, mask_opt) ->
            let sliced = Nx.slice [ R (0, actual_n) ] tensor in
            let mask =
              match mask_opt with
              | Some m ->
                  let sub = Array.sub m 0 actual_n in
                  if Array.exists Fun.id sub then Some sub else None
              | None -> None
            in
            (name, Col.P (dtype, sliced, mask))
        | Col.S arr -> (name, Col.S (Array.sub arr 0 actual_n))
        | Col.B arr -> (name, Col.B (Array.sub arr 0 actual_n)))
      t.columns
  in
  create columns

let tail ?(n = 5) t =
  let n_rows = num_rows t in
  let actual_n = min n n_rows in
  let start = n_rows - actual_n in
  let columns =
    List.map
      (fun (name, col) ->
        match col with
        | Col.P (dtype, tensor, mask_opt) ->
            let sliced = Nx.slice [ R (start, n_rows) ] tensor in
            let mask =
              match mask_opt with
              | Some m ->
                  let sub = Array.sub m start actual_n in
                  if Array.exists Fun.id sub then Some sub else None
              | None -> None
            in
            (name, Col.P (dtype, sliced, mask))
        | Col.S arr -> (name, Col.S (Array.sub arr start actual_n))
        | Col.B arr -> (name, Col.B (Array.sub arr start actual_n)))
      t.columns
  in
  create columns

let slice t ~start ~stop =
  let n_rows = num_rows t in
  let start = max 0 start in
  let stop = min stop n_rows in
  let length = max 0 (stop - start) in
  let columns =
    List.map
      (fun (name, col) ->
        match col with
        | Col.P (dtype, tensor, mask_opt) ->
            let sliced = Nx.slice [ R (start, stop) ] tensor in
            let mask =
              match mask_opt with
              | Some m ->
                  let sub = Array.sub m start length in
                  if Array.exists Fun.id sub then Some sub else None
              | None -> None
            in
            (name, Col.P (dtype, sliced, mask))
        | Col.S arr -> (name, Col.S (Array.sub arr start length))
        | Col.B arr -> (name, Col.B (Array.sub arr start length)))
      t.columns
  in
  create columns

let sample ?n ?frac ?replace ?seed:_ t =
  let n_rows = num_rows t in
  let sample_size =
    match (n, frac) with
    | Some n, None -> n
    | None, Some f -> int_of_float (f *. float_of_int n_rows)
    | _ -> invalid_arg "sample: either n or frac must be specified"
  in
  let replace = Option.value replace ~default:false in
  let indices =
    if replace then Array.init sample_size (fun _ -> Random.int n_rows)
    else
      let all_indices = Array.init n_rows Fun.id in
      for i = n_rows - 1 downto 1 do
        let j = Random.int (i + 1) in
        let temp = all_indices.(i) in
        all_indices.(i) <- all_indices.(j);
        all_indices.(j) <- temp
      done;
      Array.sub all_indices 0 (min sample_size n_rows)
  in
  let columns =
    List.map
      (fun (name, col) ->
        match col with
        | Col.P (dtype, tensor, mask_opt) ->
            let arr = Nx.to_array tensor in
            let sampled = Array.map (fun i -> arr.(i)) indices in
            let mask =
              match mask_opt with
              | Some m ->
                  let sub = Array.map (fun i -> m.(i)) indices in
                  if Array.exists Fun.id sub then Some sub else None
              | None -> None
            in
            ( name,
              Col.P
                (dtype, Nx.create dtype [| Array.length sampled |] sampled, mask)
            )
        | Col.S arr ->
            let sampled = Array.map (fun i -> arr.(i)) indices in
            (name, Col.S sampled)
        | Col.B arr ->
            let sampled = Array.map (fun i -> arr.(i)) indices in
            (name, Col.B sampled))
      t.columns
  in
  create columns

let filter t mask =
  let n_rows = num_rows t in
  if Array.length mask <> n_rows then
    invalid_arg "filter: mask length must match num_rows"
  else
    let indices = ref [] in
    Array.iteri (fun i b -> if b then indices := i :: !indices) mask;
    let indices = Array.of_list (List.rev !indices) in
    let columns =
      List.map
        (fun (name, col) ->
          match col with
          | Col.P (dtype, tensor, mask_opt) ->
              let arr = Nx.to_array tensor in
              let filtered = Array.map (fun i -> arr.(i)) indices in
              let mask =
                match mask_opt with
                | Some m ->
                    let sub = Array.map (fun i -> m.(i)) indices in
                    if Array.exists Fun.id sub then Some sub else None
                | None -> None
              in
              ( name,
                Col.P
                  ( dtype,
                    Nx.create dtype [| Array.length filtered |] filtered,
                    mask ) )
          | Col.S arr ->
              let filtered = Array.map (fun i -> arr.(i)) indices in
              (name, Col.S filtered)
          | Col.B arr ->
              let filtered = Array.map (fun i -> arr.(i)) indices in
              (name, Col.B filtered))
        t.columns
    in
    create columns

let filter_by t pred =
  let n_rows = num_rows t in
  let mask = Array.init n_rows (fun i -> pred.f t i) in
  filter t mask

(* Ergonomic helpers for null handling *)
let drop_nulls ?subset t =
  let cols_to_check =
    match subset with Some cols -> cols | None -> column_names t
  in
  let mask = Array.make (num_rows t) true in
  List.iter
    (fun col_name ->
      match get_column t col_name with
      | Some (Col.P (Nx.Float32, tensor, mask_opt)) ->
          let arr : float array = Nx.to_array tensor in
          Array.iteri
            (fun i v ->
              let is_null =
                match mask_opt with
                | Some m -> m.(i)
                | None -> classify_float v = FP_nan
              in
              if is_null then mask.(i) <- false)
            arr
      | Some (Col.P (Nx.Float64, tensor, mask_opt)) ->
          let arr : float array = Nx.to_array tensor in
          Array.iteri
            (fun i v ->
              let is_null =
                match mask_opt with
                | Some m -> m.(i)
                | None -> classify_float v = FP_nan
              in
              if is_null then mask.(i) <- false)
            arr
      | Some (Col.P (Nx.Int32, tensor, mask_opt)) ->
          let arr : int32 array = Nx.to_array tensor in
          Array.iteri
            (fun i v ->
              let is_null =
                match mask_opt with
                | Some m -> m.(i)
                | None -> v = Int32.min_int
              in
              if is_null then mask.(i) <- false)
            arr
      | Some (Col.P (Nx.Int64, tensor, mask_opt)) ->
          let arr : int64 array = Nx.to_array tensor in
          Array.iteri
            (fun i v ->
              let is_null =
                match mask_opt with
                | Some m -> m.(i)
                | None -> v = Int64.min_int
              in
              if is_null then mask.(i) <- false)
            arr
      | Some (Col.P (_, _, _)) -> ()
      | Some (Col.S arr) ->
          Array.iteri
            (fun i v -> if Option.is_none v then mask.(i) <- false)
            arr
      | Some (Col.B arr) ->
          Array.iteri
            (fun i v -> if Option.is_none v then mask.(i) <- false)
            arr
      | None -> ())
    cols_to_check;
  filter t mask

let fill_missing t col_name ~with_value =
  match get_column t col_name with
  | Some col ->
      let filled_col =
        match (col, with_value) with
        | Col.P (Nx.Float32, _, _), `Float v ->
            Col.fill_nulls_float32 col ~value:v
        | Col.P (Nx.Float64, _, _), `Float v ->
            Col.fill_nulls_float64 col ~value:v
        | Col.P (Nx.Int32, _, _), `Int32 v -> Col.fill_nulls_int32 col ~value:v
        | Col.P (Nx.Int64, _, _), `Int64 v -> Col.fill_nulls_int64 col ~value:v
        | Col.S _, `String v -> Col.fill_nulls_string col ~value:v
        | Col.B _, `Bool v -> Col.fill_nulls_bool col ~value:v
        | _ ->
            invalid_arg
              "fill_missing: value type doesn't match column type or column is \
               not supported"
      in
      add_column t col_name filled_col
  | None -> invalid_arg ("fill_missing: column " ^ col_name ^ " not found")

let has_nulls t col_name =
  match get_column t col_name with
  | Some col -> Col.has_nulls col
  | None -> invalid_arg ("has_nulls: column " ^ col_name ^ " not found")

let null_count t col_name =
  match get_column t col_name with
  | Some col -> Col.null_count col
  | None -> invalid_arg ("null_count: column " ^ col_name ^ " not found")

let drop_duplicates ?subset t =
  let cols_to_check =
    match subset with None -> column_names t | Some names -> names
  in
  let n_rows = num_rows t in
  let seen = Hashtbl.create n_rows in
  let unique_indices = ref [] in
  for i = 0 to n_rows - 1 do
    let key_str =
      String.concat ","
        (List.map
           (fun name ->
             match get_column t name with
             | Some (Col.P (dtype, tensor, mask_opt)) -> (
                 match dtype with
                 | Nx.Float32 ->
                     let arr : float array = Nx.to_array tensor in
                     let is_null =
                       match mask_opt with
                       | Some mask -> mask.(i)
                       | None -> classify_float arr.(i) = FP_nan
                     in
                     if is_null then "<null>" else string_of_float arr.(i)
                 | Nx.Float64 ->
                     let arr : float array = Nx.to_array tensor in
                     let is_null =
                       match mask_opt with
                       | Some mask -> mask.(i)
                       | None -> classify_float arr.(i) = FP_nan
                     in
                     if is_null then "<null>" else string_of_float arr.(i)
                 | Nx.Int32 ->
                     let arr : int32 array = Nx.to_array tensor in
                     let is_null =
                       match mask_opt with
                       | Some mask -> mask.(i)
                       | None -> arr.(i) = Int32.min_int
                     in
                     if is_null then "<null>" else Int32.to_string arr.(i)
                 | Nx.Int64 ->
                     let arr : int64 array = Nx.to_array tensor in
                     let is_null =
                       match mask_opt with
                       | Some mask -> mask.(i)
                       | None -> arr.(i) = Int64.min_int
                     in
                     if is_null then "<null>" else Int64.to_string arr.(i)
                 | _ -> "")
             | Some (Col.S arr) -> Option.value arr.(i) ~default:"<null>"
             | Some (Col.B arr) ->
                 Option.value
                   (Option.map string_of_bool arr.(i))
                   ~default:"<null>"
             | None -> "")
           cols_to_check)
    in
    if not (Hashtbl.mem seen key_str) then (
      Hashtbl.add seen key_str ();
      unique_indices := i :: !unique_indices)
  done;
  let indices = Array.of_list (List.rev !unique_indices) in
  let columns =
    List.map
      (fun (name, col) ->
        match col with
        | Col.P (dtype, tensor, mask_opt) ->
            let arr = Nx.to_array tensor in
            let filtered = Array.map (fun i -> arr.(i)) indices in
            let mask =
              match mask_opt with
              | Some m ->
                  let sub = Array.map (fun i -> m.(i)) indices in
                  if Array.exists Fun.id sub then Some sub else None
              | None -> None
            in
            ( name,
              Col.P
                ( dtype,
                  Nx.create dtype [| Array.length filtered |] filtered,
                  mask ) )
        | Col.S arr ->
            let filtered = Array.map (fun i -> arr.(i)) indices in
            (name, Col.S filtered)
        | Col.B arr ->
            let filtered = Array.map (fun i -> arr.(i)) indices in
            (name, Col.B filtered))
      t.columns
  in
  create columns

let concat ~axis dfs =
  match axis with
  | `Rows ->
      if dfs = [] then empty
      else
        let first = List.hd dfs in
        let names = column_names first in
        let all_same_columns =
          List.for_all (fun df -> column_names df = names) dfs
        in
        if not all_same_columns then
          invalid_arg
            "concat: all dataframes must have the same columns for row \
             concatenation"
        else
          let columns =
            List.map
              (fun name ->
                let cols = List.map (fun df -> get_column_exn df name) dfs in
                let combine_masks arrays_masks =
                  if List.exists (fun (_, m) -> Option.is_some m) arrays_masks
                  then
                    let mask_arrays =
                      List.map
                        (fun (arr, mask_opt) ->
                          match mask_opt with
                          | Some m -> Array.copy m
                          | None -> Array.make (Array.length arr) false)
                        arrays_masks
                    in
                    let concatenated_mask = Array.concat mask_arrays in
                    if Array.exists Fun.id concatenated_mask then
                      Some concatenated_mask
                    else None
                  else None
                in
                let concatenated =
                  match List.hd cols with
                  | Col.P (dtype, _, _) -> (
                      match dtype with
                      | Nx.Float32 ->
                          let arrays_masks =
                            List.map
                              (function
                                | Col.P (Nx.Float32, t, mask) ->
                                    let arr : float array = Nx.to_array t in
                                    (arr, mask)
                                | _ -> failwith "concat: column type mismatch")
                              cols
                          in
                          let arrays = List.map fst arrays_masks in
                          let all_data : float array = Array.concat arrays in
                          let combined_mask = combine_masks arrays_masks in
                          Col.P
                            ( Nx.Float32,
                              Nx.create Nx.float32
                                [| Array.length all_data |]
                                all_data,
                              combined_mask )
                      | Nx.Float64 ->
                          let arrays_masks =
                            List.map
                              (function
                                | Col.P (Nx.Float64, t, mask) ->
                                    let arr : float array = Nx.to_array t in
                                    (arr, mask)
                                | _ -> failwith "concat: column type mismatch")
                              cols
                          in
                          let arrays = List.map fst arrays_masks in
                          let all_data : float array = Array.concat arrays in
                          let combined_mask = combine_masks arrays_masks in
                          Col.P
                            ( Nx.Float64,
                              Nx.create Nx.float64
                                [| Array.length all_data |]
                                all_data,
                              combined_mask )
                      | Nx.Int32 ->
                          let arrays_masks =
                            List.map
                              (function
                                | Col.P (Nx.Int32, t, mask) ->
                                    let arr : int32 array = Nx.to_array t in
                                    (arr, mask)
                                | _ -> failwith "concat: column type mismatch")
                              cols
                          in
                          let arrays = List.map fst arrays_masks in
                          let all_data : int32 array = Array.concat arrays in
                          let combined_mask = combine_masks arrays_masks in
                          Col.P
                            ( Nx.Int32,
                              Nx.create Nx.int32
                                [| Array.length all_data |]
                                all_data,
                              combined_mask )
                      | Nx.Int64 ->
                          let arrays_masks =
                            List.map
                              (function
                                | Col.P (Nx.Int64, t, mask) ->
                                    let arr : int64 array = Nx.to_array t in
                                    (arr, mask)
                                | _ -> failwith "concat: column type mismatch")
                              cols
                          in
                          let arrays = List.map fst arrays_masks in
                          let all_data : int64 array = Array.concat arrays in
                          let combined_mask = combine_masks arrays_masks in
                          Col.P
                            ( Nx.Int64,
                              Nx.create Nx.int64
                                [| Array.length all_data |]
                                all_data,
                              combined_mask )
                      | _ -> failwith "concat: unsupported dtype")
                  | Col.S _ ->
                      let arrays =
                        List.map
                          (function
                            | Col.S arr -> arr
                            | _ -> failwith "concat: column type mismatch")
                          cols
                      in
                      Col.S (Array.concat arrays)
                  | Col.B _ ->
                      let arrays =
                        List.map
                          (function
                            | Col.B arr -> arr
                            | _ -> failwith "concat: column type mismatch")
                          cols
                      in
                      Col.B (Array.concat arrays)
                in
                (name, concatenated))
              names
          in
          create columns
  | `Columns ->
      if dfs = [] then empty
      else
        let first_rows = num_rows (List.hd dfs) in
        let all_same_rows =
          List.for_all (fun df -> num_rows df = first_rows) dfs
        in
        if not all_same_rows then
          invalid_arg
            "concat: all dataframes must have the same number of rows for \
             column concatenation"
        else
          let all_columns = List.concat_map (fun df -> df.columns) dfs in
          create all_columns

let map (type a b) t (dtype : (a, b) Nx.dtype) (f : a row) : (a, b) Nx.t =
  let n_rows = num_rows t in
  let data = Array.init n_rows (fun i -> f.f t i) in
  Nx.create dtype [| n_rows |] data

let with_column t name dtype f =
  let tensor = map t dtype f in
  add_column t name (Col.of_tensor tensor)

let with_columns t cols =
  List.fold_left (fun df (name, col) -> add_column df name col) t cols

let with_columns_map t specs =
  (* Build all columns in one pass for efficiency *)
  let new_cols =
    List.map
      (fun (name, dtype, row_fn) -> (name, Col.of_tensor (map t dtype row_fn)))
      specs
  in
  with_columns t new_cols

let iter t f =
  let n_rows = num_rows t in
  for i = 0 to n_rows - 1 do
    f.f t i
  done

let fold t ~init ~f =
  let n_rows = num_rows t in
  let rec loop i acc =
    if i >= n_rows then acc
    else
      let update_fn = f.f t i in
      let next_acc = update_fn acc in
      loop (i + 1) next_acc
  in
  loop 0 init

let fold_left t ~init ~f combine =
  let n_rows = num_rows t in
  let rec loop i acc =
    if i >= n_rows then acc
    else
      let fn = f.f t i in
      let value = fn acc in
      loop (i + 1) (combine acc value)
  in
  loop 0 init

let sort t key ~compare =
  let n_rows = num_rows t in
  let keys = Array.init n_rows (fun i -> (i, key.f t i)) in
  Array.sort (fun (_, k1) (_, k2) -> compare k1 k2) keys;
  let indices = Array.map fst keys in
  let columns =
    List.map
      (fun (name, col) ->
        match col with
        | Col.P (dtype, tensor, mask_opt) ->
            let arr = Nx.to_array tensor in
            let sorted = Array.map (fun i -> arr.(i)) indices in
            let mask =
              match mask_opt with
              | Some m ->
                  let permuted = Array.map (fun i -> m.(i)) indices in
                  if Array.exists Fun.id permuted then Some permuted else None
              | None -> None
            in
            ( name,
              Col.P
                (dtype, Nx.create dtype [| Array.length sorted |] sorted, mask)
            )
        | Col.S arr ->
            let sorted = Array.map (fun i -> arr.(i)) indices in
            (name, Col.S sorted)
        | Col.B arr ->
            let sorted = Array.map (fun i -> arr.(i)) indices in
            (name, Col.B sorted))
      t.columns
  in
  create columns

let sort_values ?(ascending = true) t name =
  match get_column t name with
  | None -> raise Not_found
  | Some col -> (
      let compare = if ascending then compare else fun a b -> compare b a in
      match col with
      | Col.P (Nx.Float32, _, _) | Col.P (Nx.Float64, _, _) ->
          sort t (Row.float64 name) ~compare
      | Col.P (Nx.Int32, _, _) -> sort t (Row.int32 name) ~compare:Int32.compare
      | Col.P (Nx.Int64, _, _) -> sort t (Row.int64 name) ~compare:Int64.compare
      | Col.S _ -> sort t (Row.string name) ~compare:String.compare
      | Col.B _ -> sort t (Row.bool name) ~compare:Bool.compare
      | _ -> failwith "sort_by_column: unsupported column type")

let group_by t key =
  let n_rows = num_rows t in
  let groups = Hashtbl.create 16 in
  for i = 0 to n_rows - 1 do
    let k = key.f t i in
    let indices =
      match Hashtbl.find_opt groups k with None -> [] | Some lst -> lst
    in
    Hashtbl.replace groups k (i :: indices)
  done;
  Hashtbl.fold
    (fun k indices acc ->
      let indices = Array.of_list (List.rev indices) in
      let columns =
        List.map
          (fun (name, col) ->
            match col with
            | Col.P (dtype, tensor, mask_opt) ->
                let arr = Nx.to_array tensor in
                let grouped = Array.map (fun i -> arr.(i)) indices in
                let mask =
                  match mask_opt with
                  | Some m ->
                      let sub = Array.map (fun i -> m.(i)) indices in
                      if Array.exists Fun.id sub then Some sub else None
                  | None -> None
                in
                ( name,
                  Col.P
                    ( dtype,
                      Nx.create dtype [| Array.length grouped |] grouped,
                      mask ) )
            | Col.S arr ->
                let grouped = Array.map (fun i -> arr.(i)) indices in
                (name, Col.S grouped)
            | Col.B arr ->
                let grouped = Array.map (fun i -> arr.(i)) indices in
                (name, Col.B grouped))
          t.columns
      in
      (k, create columns) :: acc)
    groups []

let group_by_column t name =
  match get_column t name with
  | None -> raise Not_found
  | Some col ->
      let n_rows = num_rows t in
      let groups = Hashtbl.create 16 in
      for i = 0 to n_rows - 1 do
        let key_str =
          match col with
          | Col.P (dtype, tensor, mask_opt) -> (
              match dtype with
              | Nx.Float32 ->
                  let arr : float array = Nx.to_array tensor in
                  let is_null =
                    match mask_opt with
                    | Some mask -> mask.(i)
                    | None -> classify_float arr.(i) = FP_nan
                  in
                  if is_null then "<null>" else string_of_float arr.(i)
              | Nx.Float64 ->
                  let arr : float array = Nx.to_array tensor in
                  let is_null =
                    match mask_opt with
                    | Some mask -> mask.(i)
                    | None -> classify_float arr.(i) = FP_nan
                  in
                  if is_null then "<null>" else string_of_float arr.(i)
              | Nx.Int32 ->
                  let arr : int32 array = Nx.to_array tensor in
                  let is_null =
                    match mask_opt with
                    | Some mask -> mask.(i)
                    | None -> arr.(i) = Int32.min_int
                  in
                  if is_null then "<null>" else Int32.to_string arr.(i)
              | Nx.Int64 ->
                  let arr : int64 array = Nx.to_array tensor in
                  let is_null =
                    match mask_opt with
                    | Some mask -> mask.(i)
                    | None -> arr.(i) = Int64.min_int
                  in
                  if is_null then "<null>" else Int64.to_string arr.(i)
              | _ -> "")
          | Col.S arr -> Option.value arr.(i) ~default:"<null>"
          | Col.B arr ->
              Option.value (Option.map string_of_bool arr.(i)) ~default:"<null>"
        in
        let value_col =
          match col with
          | Col.P (dtype, tensor, mask_opt) ->
              let arr = Nx.to_array tensor in
              let data = Array.init 1 (fun _ -> arr.(i)) in
              let mask =
                match mask_opt with
                | Some m -> if m.(i) then Some [| true |] else None
                | None -> None
              in
              Col.P (dtype, Nx.create dtype [| 1 |] data, mask)
          | Col.S arr -> Col.S [| arr.(i) |]
          | Col.B arr -> Col.B [| arr.(i) |]
        in
        let indices, prev_col =
          match Hashtbl.find_opt groups key_str with
          | None -> ([], value_col)
          | Some (lst, c) -> (lst, c)
        in
        Hashtbl.replace groups key_str (i :: indices, prev_col)
      done;
      Hashtbl.fold
        (fun _key_str (indices, key_col) acc ->
          let indices = Array.of_list (List.rev indices) in
          let columns =
            List.map
              (fun (name, col) ->
                match col with
                | Col.P (dtype, tensor, mask_opt) ->
                    let arr = Nx.to_array tensor in
                    let grouped = Array.map (fun i -> arr.(i)) indices in
                    let mask =
                      match mask_opt with
                      | Some m ->
                          let sub = Array.map (fun i -> m.(i)) indices in
                          if Array.exists Fun.id sub then Some sub else None
                      | None -> None
                    in
                    ( name,
                      Col.P
                        ( dtype,
                          Nx.create dtype [| Array.length grouped |] grouped,
                          mask ) )
                | Col.S arr ->
                    let grouped = Array.map (fun i -> arr.(i)) indices in
                    (name, Col.S grouped)
                | Col.B arr ->
                    let grouped = Array.map (fun i -> arr.(i)) indices in
                    (name, Col.B grouped))
              t.columns
          in
          (key_col, create columns) :: acc)
        groups []

module Agg = struct
  (* Helper to check if a value at index is null based on mask and sentinel *)
  let is_null_float mask i value =
    match mask with
    | Some m when m.(i) -> true
    | _ -> classify_float value = FP_nan

  let is_null_int32 mask i value =
    match mask with Some m when m.(i) -> true | _ -> value = Int32.min_int

  let is_null_int64 mask i value =
    match mask with Some m when m.(i) -> true | _ -> value = Int64.min_int

  (* Helper to filter non-null float values *)
  let filter_float_array mask arr =
    let rec aux idx acc count =
      if idx < 0 then (Array.of_list acc, count)
      else
        let value = arr.(idx) in
        if is_null_float mask idx value then aux (idx - 1) acc count
        else aux (idx - 1) (value :: acc) (count + 1)
    in
    aux (Array.length arr - 1) [] 0

  (* Helper to filter non-null int32 values *)
  let filter_int32_array mask arr =
    let rec aux idx acc count =
      if idx < 0 then (Array.of_list acc, count)
      else
        let value = arr.(idx) in
        if is_null_int32 mask idx value then aux (idx - 1) acc count
        else aux (idx - 1) (value :: acc) (count + 1)
    in
    aux (Array.length arr - 1) [] 0

  (* Helper to filter non-null int64 values *)
  let filter_int64_array mask arr =
    let rec aux idx acc count =
      if idx < 0 then (Array.of_list acc, count)
      else
        let value = arr.(idx) in
        if is_null_int64 mask idx value then aux (idx - 1) acc count
        else aux (idx - 1) (value :: acc) (count + 1)
    in
    aux (Array.length arr - 1) [] 0

  module Float = struct
    let sum t name =
      match get_column t name with
      | Some (Col.P (dtype, tensor, mask)) -> (
          match dtype with
          | Nx.Float16 ->
              let arr : float array = Nx.to_array tensor in
              Array.fold_left ( +. ) 0. arr
          | Nx.BFloat16 ->
              let arr : float array = Nx.to_array tensor in
              Array.fold_left ( +. ) 0. arr
          | Nx.Float32 ->
              let arr : float array = Nx.to_array tensor in
              let filtered, _ = filter_float_array mask arr in
              Array.fold_left ( +. ) 0. filtered
          | Nx.Float64 ->
              let arr : float array = Nx.to_array tensor in
              let filtered, _ = filter_float_array mask arr in
              Array.fold_left ( +. ) 0. filtered
          | Nx.Int8 ->
              let arr : int array = Nx.to_array tensor in
              Array.fold_left (fun acc x -> acc +. float_of_int x) 0. arr
          | Nx.UInt8 ->
              let arr : int array = Nx.to_array tensor in
              Array.fold_left (fun acc x -> acc +. float_of_int x) 0. arr
          | Nx.Int16 ->
              let arr : int array = Nx.to_array tensor in
              Array.fold_left (fun acc x -> acc +. float_of_int x) 0. arr
          | Nx.UInt16 ->
              let arr : int array = Nx.to_array tensor in
              Array.fold_left (fun acc x -> acc +. float_of_int x) 0. arr
          | Nx.Int32 ->
              let arr : int32 array = Nx.to_array tensor in
              let filtered, _ = filter_int32_array mask arr in
              Array.fold_left (fun acc x -> acc +. Int32.to_float x) 0. filtered
          | Nx.Int64 ->
              let arr : int64 array = Nx.to_array tensor in
              let filtered, _ = filter_int64_array mask arr in
              Array.fold_left (fun acc x -> acc +. Int64.to_float x) 0. filtered
          | Nx.Int ->
              let arr : int array = Nx.to_array tensor in
              Array.fold_left (fun acc x -> acc +. float_of_int x) 0. arr
          | Nx.NativeInt ->
              let arr : nativeint array = Nx.to_array tensor in
              Array.fold_left (fun acc x -> acc +. Nativeint.to_float x) 0. arr
          | Nx.Complex32 -> failwith "Float.sum: complex numbers not supported"
          | Nx.Complex64 -> failwith "Float.sum: complex numbers not supported"
          | Nx.Bool -> failwith "Float.sum: boolean values not supported"
          | Nx.Int4 ->
              let arr : int array = Nx.to_array tensor in
              Array.fold_left (fun acc x -> acc +. float_of_int x) 0. arr
          | Nx.UInt4 ->
              let arr : int array = Nx.to_array tensor in
              Array.fold_left (fun acc x -> acc +. float_of_int x) 0. arr
          | Nx.Float8_e4m3 ->
              let arr : float array = Nx.to_array tensor in
              Array.fold_left ( +. ) 0. arr
          | Nx.Float8_e5m2 ->
              let arr : float array = Nx.to_array tensor in
              Array.fold_left ( +. ) 0. arr
          | Nx.Complex16 -> failwith "Float.sum: complex numbers not supported"
          | Nx.QInt8 ->
              let arr : int array = Nx.to_array tensor in
              Array.fold_left (fun acc x -> acc +. float_of_int x) 0. arr
          | Nx.QUInt8 ->
              let arr : int array = Nx.to_array tensor in
              Array.fold_left (fun acc x -> acc +. float_of_int x) 0. arr)
      | _ -> failwith "Float.sum: column must be numeric"

    let mean t name =
      match get_column t name with
      | Some (Col.P (Nx.Float32, tensor, mask)) ->
          let arr : float array = Nx.to_array tensor in
          let filtered, count = filter_float_array mask arr in
          if count = 0 then Float.nan
          else Array.fold_left ( +. ) 0. filtered /. float_of_int count
      | Some (Col.P (Nx.Float64, tensor, mask)) ->
          let arr : float array = Nx.to_array tensor in
          let filtered, count = filter_float_array mask arr in
          if count = 0 then Float.nan
          else Array.fold_left ( +. ) 0. filtered /. float_of_int count
      | Some (Col.P (Nx.Int32, tensor, mask)) ->
          let arr : int32 array = Nx.to_array tensor in
          let filtered, count = filter_int32_array mask arr in
          if count = 0 then Float.nan
          else
            Array.fold_left (fun acc x -> acc +. Int32.to_float x) 0. filtered
            /. float_of_int count
      | Some (Col.P (Nx.Int64, tensor, mask)) ->
          let arr : int64 array = Nx.to_array tensor in
          let filtered, count = filter_int64_array mask arr in
          if count = 0 then Float.nan
          else
            Array.fold_left (fun acc x -> acc +. Int64.to_float x) 0. filtered
            /. float_of_int count
      | Some (Col.P _) ->
          (* For other numeric types, fall back to sum / count *)
          let s = sum t name in
          s /. float_of_int (num_rows t)
      | _ -> failwith "Float.mean: column must be numeric"

    let std t name =
      let m = mean t name in
      match get_column t name with
      | Some (Col.P (dtype, tensor, mask)) ->
          let variance =
            match dtype with
            | Nx.Float16 ->
                let arr : float array = Nx.to_array tensor in
                Array.fold_left
                  (fun acc x ->
                    let diff = x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. float_of_int (Array.length arr)
            | Nx.BFloat16 ->
                let arr : float array = Nx.to_array tensor in
                Array.fold_left
                  (fun acc x ->
                    let diff = x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. float_of_int (Array.length arr)
            | Nx.Float32 ->
                let arr : float array = Nx.to_array tensor in
                let filtered, count = filter_float_array mask arr in
                if count = 0 then Float.nan
                else
                  Array.fold_left
                    (fun acc x ->
                      let diff = x -. m in
                      acc +. (diff *. diff))
                    0. filtered
                  /. float_of_int count
            | Nx.Float64 ->
                let arr : float array = Nx.to_array tensor in
                let filtered, count = filter_float_array mask arr in
                if count = 0 then Float.nan
                else
                  Array.fold_left
                    (fun acc x ->
                      let diff = x -. m in
                      acc +. (diff *. diff))
                    0. filtered
                  /. float_of_int count
            | Nx.Int8 ->
                let arr : int array = Nx.to_array tensor in
                Array.fold_left
                  (fun acc x ->
                    let diff = float_of_int x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. float_of_int (Array.length arr)
            | Nx.UInt8 ->
                let arr : int array = Nx.to_array tensor in
                Array.fold_left
                  (fun acc x ->
                    let diff = float_of_int x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. float_of_int (Array.length arr)
            | Nx.Int16 ->
                let arr : int array = Nx.to_array tensor in
                Array.fold_left
                  (fun acc x ->
                    let diff = float_of_int x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. float_of_int (Array.length arr)
            | Nx.UInt16 ->
                let arr : int array = Nx.to_array tensor in
                Array.fold_left
                  (fun acc x ->
                    let diff = float_of_int x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. float_of_int (Array.length arr)
            | Nx.Int ->
                let arr : int array = Nx.to_array tensor in
                Array.fold_left
                  (fun acc x ->
                    let diff = float_of_int x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. float_of_int (Array.length arr)
            | Nx.Int32 ->
                let arr : int32 array = Nx.to_array tensor in
                let filtered, count = filter_int32_array mask arr in
                if count = 0 then Float.nan
                else
                  Array.fold_left
                    (fun acc x ->
                      let diff = Int32.to_float x -. m in
                      acc +. (diff *. diff))
                    0. filtered
                  /. float_of_int count
            | Nx.Int64 ->
                let arr : int64 array = Nx.to_array tensor in
                let filtered, count = filter_int64_array mask arr in
                if count = 0 then Float.nan
                else
                  Array.fold_left
                    (fun acc x ->
                      let diff = Int64.to_float x -. m in
                      acc +. (diff *. diff))
                    0. filtered
                  /. float_of_int count
            | Nx.NativeInt ->
                let arr : nativeint array = Nx.to_array tensor in
                Array.fold_left
                  (fun acc x ->
                    let diff = Nativeint.to_float x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. float_of_int (Array.length arr)
            | Nx.Complex32 ->
                failwith "Float.std: complex numbers not supported"
            | Nx.Complex64 ->
                failwith "Float.std: complex numbers not supported"
            | Nx.Bool -> failwith "Float.std: boolean values not supported"
            | Nx.Int4 ->
                let arr : int array = Nx.to_array tensor in
                let n = float_of_int (Array.length arr) in
                Array.fold_left
                  (fun acc x ->
                    let diff = float_of_int x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. n
            | Nx.UInt4 ->
                let arr : int array = Nx.to_array tensor in
                let n = float_of_int (Array.length arr) in
                Array.fold_left
                  (fun acc x ->
                    let diff = float_of_int x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. n
            | Nx.Float8_e4m3 ->
                let arr : float array = Nx.to_array tensor in
                let n = float_of_int (Array.length arr) in
                Array.fold_left
                  (fun acc x ->
                    let diff = x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. n
            | Nx.Float8_e5m2 ->
                let arr : float array = Nx.to_array tensor in
                let n = float_of_int (Array.length arr) in
                Array.fold_left
                  (fun acc x ->
                    let diff = x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. n
            | Nx.Complex16 ->
                failwith "Float.std: complex numbers not supported"
            | Nx.QInt8 ->
                let arr : int array = Nx.to_array tensor in
                let n = float_of_int (Array.length arr) in
                Array.fold_left
                  (fun acc x ->
                    let diff = float_of_int x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. n
            | Nx.QUInt8 ->
                let arr : int array = Nx.to_array tensor in
                let n = float_of_int (Array.length arr) in
                Array.fold_left
                  (fun acc x ->
                    let diff = float_of_int x -. m in
                    acc +. (diff *. diff))
                  0. arr
                /. n
          in
          sqrt variance
      | _ -> failwith "Float.std: column must be numeric"

    let var t name =
      let m = mean t name in
      match get_column t name with
      | Some (Col.P (dtype, tensor, mask)) -> (
          match dtype with
          | Nx.Float32 ->
              let arr : float array = Nx.to_array tensor in
              let filtered, count = filter_float_array mask arr in
              if count = 0 then Float.nan
              else
                Array.fold_left
                  (fun acc x ->
                    let diff = x -. m in
                    acc +. (diff *. diff))
                  0. filtered
                /. float_of_int count
          | Nx.Float64 ->
              let arr : float array = Nx.to_array tensor in
              let filtered, count = filter_float_array mask arr in
              if count = 0 then Float.nan
              else
                Array.fold_left
                  (fun acc x ->
                    let diff = x -. m in
                    acc +. (diff *. diff))
                  0. filtered
                /. float_of_int count
          | Nx.Int32 ->
              let arr : int32 array = Nx.to_array tensor in
              let filtered, count = filter_int32_array mask arr in
              if count = 0 then Float.nan
              else
                Array.fold_left
                  (fun acc x ->
                    let diff = Int32.to_float x -. m in
                    acc +. (diff *. diff))
                  0. filtered
                /. float_of_int count
          | Nx.Int64 ->
              let arr : int64 array = Nx.to_array tensor in
              let filtered, count = filter_int64_array mask arr in
              if count = 0 then Float.nan
              else
                Array.fold_left
                  (fun acc x ->
                    let diff = Int64.to_float x -. m in
                    acc +. (diff *. diff))
                  0. filtered
                /. float_of_int count
          | _ -> failwith "Float.var: unsupported numeric type")
      | _ -> failwith "Float.var: column must be numeric"

    let min t name =
      match get_column t name with
      | Some (Col.P (dtype, tensor, mask)) -> (
          match dtype with
          | Nx.Float16 ->
              let arr : float array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else Some (Array.fold_left min max_float arr)
          | Nx.BFloat16 ->
              let arr : float array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else Some (Array.fold_left min max_float arr)
          | Nx.Float32 ->
              let arr : float array = Nx.to_array tensor in
              let filtered, count = filter_float_array mask arr in
              if count = 0 then None
              else Some (Array.fold_left min max_float filtered)
          | Nx.Float64 ->
              let arr : float array = Nx.to_array tensor in
              let filtered, count = filter_float_array mask arr in
              if count = 0 then None
              else Some (Array.fold_left min max_float filtered)
          | Nx.Int8 ->
              let arr : int array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> min acc (float_of_int x))
                     max_float arr)
          | Nx.UInt8 ->
              let arr : int array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> min acc (float_of_int x))
                     max_float arr)
          | Nx.Int16 ->
              let arr : int array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> min acc (float_of_int x))
                     max_float arr)
          | Nx.UInt16 ->
              let arr : int array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> min acc (float_of_int x))
                     max_float arr)
          | Nx.Int ->
              let arr : int array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> min acc (float_of_int x))
                     max_float arr)
          | Nx.Int32 ->
              let arr : int32 array = Nx.to_array tensor in
              let filtered, count = filter_int32_array mask arr in
              if count = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> min acc (Int32.to_float x))
                     max_float filtered)
          | Nx.Int64 ->
              let arr : int64 array = Nx.to_array tensor in
              let filtered, count = filter_int64_array mask arr in
              if count = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> min acc (Int64.to_float x))
                     max_float filtered)
          | Nx.NativeInt ->
              let arr : nativeint array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> min acc (Nativeint.to_float x))
                     max_float arr)
          | Nx.Complex32 -> failwith "Float.min: complex numbers not supported"
          | Nx.Complex64 -> failwith "Float.min: complex numbers not supported"
          | Nx.Bool -> failwith "Float.min: boolean values not supported"
          | Nx.Int4 ->
              let arr : int array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> min acc (float_of_int x))
                     max_float arr)
          | Nx.UInt4 ->
              let arr : int array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> min acc (float_of_int x))
                     max_float arr)
          | Nx.Float8_e4m3 ->
              let arr : float array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else Some (Array.fold_left min max_float arr)
          | Nx.Float8_e5m2 ->
              let arr : float array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else Some (Array.fold_left min max_float arr)
          | Nx.Complex16 -> failwith "Float.min: complex numbers not supported"
          | Nx.QInt8 ->
              let arr : int array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> min acc (float_of_int x))
                     max_float arr)
          | Nx.QUInt8 ->
              let arr : int array = Nx.to_array tensor in
              if Array.length arr = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> min acc (float_of_int x))
                     max_float arr))
      | _ -> failwith "Float.min: column must be numeric"

    let max t name =
      match get_column t name with
      | Some (Col.P (dtype, tensor, mask)) -> (
          match dtype with
          | Nx.Float32 ->
              let arr : float array = Nx.to_array tensor in
              let filtered, count = filter_float_array mask arr in
              if count = 0 then None
              else Some (Array.fold_left max min_float filtered)
          | Nx.Float64 ->
              let arr : float array = Nx.to_array tensor in
              let filtered, count = filter_float_array mask arr in
              if count = 0 then None
              else Some (Array.fold_left max min_float filtered)
          | Nx.Int32 ->
              let arr : int32 array = Nx.to_array tensor in
              let filtered, count = filter_int32_array mask arr in
              if count = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> max acc (Int32.to_float x))
                     min_float filtered)
          | Nx.Int64 ->
              let arr : int64 array = Nx.to_array tensor in
              let filtered, count = filter_int64_array mask arr in
              if count = 0 then None
              else
                Some
                  (Array.fold_left
                     (fun acc x -> max acc (Int64.to_float x))
                     min_float filtered)
          | _ -> failwith "Float.max: unsupported numeric type")
      | _ -> failwith "Float.max: column must be numeric"

    let median t name =
      match get_column t name with
      | Some (Col.P (dtype, tensor, _)) ->
          let float_arr =
            match dtype with
            | Nx.Float32 ->
                let arr : float array = Nx.to_array tensor in
                Array.copy arr
            | Nx.Float64 ->
                let arr : float array = Nx.to_array tensor in
                Array.copy arr
            | Nx.Int32 ->
                let arr : int32 array = Nx.to_array tensor in
                Array.map Int32.to_float arr
            | Nx.Int64 ->
                let arr : int64 array = Nx.to_array tensor in
                Array.map Int64.to_float arr
            | _ -> failwith "Float.median: unsupported numeric type"
          in
          Array.sort compare float_arr;
          let n = Array.length float_arr in
          if n = 0 then 0.
          else if n mod 2 = 0 then
            (float_arr.((n / 2) - 1) +. float_arr.(n / 2)) /. 2.
          else float_arr.(n / 2)
      | _ -> failwith "Float.median: column must be numeric"

    let quantile t name ~q =
      match get_column t name with
      | Some (Col.P (dtype, tensor, _)) ->
          let float_arr =
            match dtype with
            | Nx.Float32 ->
                let arr : float array = Nx.to_array tensor in
                Array.copy arr
            | Nx.Float64 ->
                let arr : float array = Nx.to_array tensor in
                Array.copy arr
            | Nx.Int32 ->
                let arr : int32 array = Nx.to_array tensor in
                Array.map Int32.to_float arr
            | Nx.Int64 ->
                let arr : int64 array = Nx.to_array tensor in
                Array.map Int64.to_float arr
            | _ -> failwith "Float.quantile: unsupported numeric type"
          in
          Array.sort compare float_arr;
          let n = Array.length float_arr in
          if n = 0 then 0.
          else
            let pos = q *. float_of_int (n - 1) in
            let lower = int_of_float pos in
            let upper = Stdlib.min (lower + 1) (n - 1) in
            let weight = pos -. float_of_int lower in
            (float_arr.(lower) *. (1. -. weight))
            +. (float_arr.(upper) *. weight)
      | _ -> failwith "Float.quantile: column must be numeric"
  end

  module Int = struct
    let sum t name =
      match get_column t name with
      | Some (Col.P (Nx.Int32, tensor, _)) ->
          let arr = Nx.to_array tensor in
          Array.fold_left Int64.add 0L (Array.map Int64.of_int32 arr)
      | Some (Col.P (Nx.Int64, tensor, _)) ->
          let arr = Nx.to_array tensor in
          Array.fold_left Int64.add 0L arr
      | _ -> failwith "Int.sum: column must be integer type"

    let min t name =
      match get_column t name with
      | Some (Col.P (Nx.Int32, tensor, _)) ->
          let arr = Nx.to_array tensor in
          if Array.length arr = 0 then None
          else Some (Int64.of_int32 (Array.fold_left min Int32.max_int arr))
      | Some (Col.P (Nx.Int64, tensor, _)) ->
          let arr = Nx.to_array tensor in
          if Array.length arr = 0 then None
          else Some (Array.fold_left min Int64.max_int arr)
      | _ -> failwith "Int.min: column must be integer type"

    let max t name =
      match get_column t name with
      | Some (Col.P (Nx.Int32, tensor, _)) ->
          let arr = Nx.to_array tensor in
          if Array.length arr = 0 then None
          else Some (Int64.of_int32 (Array.fold_left max Int32.min_int arr))
      | Some (Col.P (Nx.Int64, tensor, _)) ->
          let arr = Nx.to_array tensor in
          if Array.length arr = 0 then None
          else Some (Array.fold_left max Int64.min_int arr)
      | _ -> failwith "Int.max: column must be integer type"

    let mean t name =
      let s = Int64.to_float (sum t name) in
      s /. float_of_int (num_rows t)
  end

  module String = struct
    let min t name =
      match get_column t name with
      | Some (Col.S arr) ->
          let values = Array.to_list arr |> List.filter_map Fun.id in
          if values = [] then None
          else Some (List.fold_left min (List.hd values) values)
      | _ -> failwith "String.min: column must be string type"

    let max t name =
      match get_column t name with
      | Some (Col.S arr) ->
          let values = Array.to_list arr |> List.filter_map Fun.id in
          if values = [] then None
          else Some (List.fold_left max (List.hd values) values)
      | _ -> failwith "String.max: column must be string type"

    let concat t name ?sep () =
      let sep = Option.value sep ~default:"" in
      match get_column t name with
      | Some (Col.S arr) ->
          let values = Array.to_list arr |> List.filter_map Fun.id in
          String.concat sep values
      | _ -> failwith "String.concat: column must be string type"

    let unique t name =
      match get_column t name with
      | Some (Col.S arr) ->
          let seen = Hashtbl.create 16 in
          Array.iter
            (function Some s -> Hashtbl.replace seen s () | None -> ())
            arr;
          Array.of_list (Hashtbl.fold (fun k _ acc -> k :: acc) seen [])
      | _ -> failwith "String.unique: column must be string type"

    let nunique t name = Array.length (unique t name)

    let mode t name =
      match get_column t name with
      | Some (Col.S arr) ->
          let counts = Hashtbl.create 16 in
          Array.iter
            (function
              | Some s ->
                  let count =
                    Option.value (Hashtbl.find_opt counts s) ~default:0
                  in
                  Hashtbl.replace counts s (count + 1)
              | None -> ())
            arr;
          if Hashtbl.length counts = 0 then None
          else
            let max_count = ref 0 in
            let mode_val = ref "" in
            Hashtbl.iter
              (fun k v ->
                if v > !max_count then (
                  max_count := v;
                  mode_val := k))
              counts;
            Some !mode_val
      | _ -> failwith "String.mode: column must be string type"
  end

  module Bool = struct
    let all t name =
      match get_column t name with
      | Some (Col.B arr) ->
          Array.for_all
            (function Some true | None -> true | Some false -> false)
            arr
      | _ -> failwith "Bool.all: column must be bool type"

    let any t name =
      match get_column t name with
      | Some (Col.B arr) ->
          Array.exists (function Some true -> true | _ -> false) arr
      | _ -> failwith "Bool.any: column must be bool type"

    let sum t name =
      match get_column t name with
      | Some (Col.B arr) ->
          Array.fold_left
            (fun acc -> function Some true -> acc + 1 | _ -> acc)
            0 arr
      | _ -> failwith "Bool.sum: column must be bool type"

    let mean t name = float_of_int (sum t name) /. float_of_int (num_rows t)
  end

  let count t name =
    match get_column t name with
    | Some col -> Col.length col - Col.null_count col
    | None -> 0

  let nunique t name =
    match get_column t name with
    | Some (Col.P (_, tensor, _)) ->
        let arr = Nx.to_array tensor in
        let seen = Hashtbl.create 16 in
        Array.iter (fun x -> Hashtbl.replace seen (Obj.repr x) ()) arr;
        Hashtbl.length seen
    | Some (Col.S arr) ->
        let seen = Hashtbl.create 16 in
        Array.iter
          (function Some s -> Hashtbl.replace seen s () | None -> ())
          arr;
        Hashtbl.length seen
    | Some (Col.B arr) ->
        let seen = Hashtbl.create 3 in
        Array.iter
          (function Some b -> Hashtbl.replace seen b () | None -> ())
          arr;
        Hashtbl.length seen
    | None -> 0

  let value_counts t name =
    match get_column t name with
    | Some col -> (
        match col with
        | Col.P (dtype, tensor, _) ->
            let counts = Hashtbl.create 16 in
            let arr = Nx.to_array tensor in
            Array.iter
              (fun x ->
                let key = Obj.repr x in
                let count =
                  Option.value (Hashtbl.find_opt counts key) ~default:0
                in
                Hashtbl.replace counts key (count + 1))
              arr;
            let items =
              Hashtbl.fold (fun k v acc -> (Obj.obj k, v) :: acc) counts []
            in
            let items =
              List.sort (fun (_, c1) (_, c2) -> compare c2 c1) items
            in
            let values = Array.of_list (List.map fst items) in
            let counts_arr = Array.of_list (List.map snd items) in
            ( Col.P
                (dtype, Nx.create dtype [| Array.length values |] values, None),
              counts_arr )
        | Col.S arr ->
            let str_counts = Hashtbl.create 16 in
            Array.iter
              (function
                | Some s ->
                    let count =
                      Option.value (Hashtbl.find_opt str_counts s) ~default:0
                    in
                    Hashtbl.replace str_counts s (count + 1)
                | None -> ())
              arr;
            let items =
              Hashtbl.fold (fun k v acc -> (k, v) :: acc) str_counts []
            in
            let items =
              List.sort (fun (_, c1) (_, c2) -> compare c2 c1) items
            in
            let values =
              Array.of_list (List.map (fun x -> Some x) (List.map fst items))
            in
            let counts_arr = Array.of_list (List.map snd items) in
            (Col.S values, counts_arr)
        | Col.B arr ->
            let bool_counts = Hashtbl.create 3 in
            Array.iter
              (function
                | Some b ->
                    let count =
                      Option.value (Hashtbl.find_opt bool_counts b) ~default:0
                    in
                    Hashtbl.replace bool_counts b (count + 1)
                | None -> ())
              arr;
            let items =
              Hashtbl.fold (fun k v acc -> (k, v) :: acc) bool_counts []
            in
            let items =
              List.sort (fun (_, c1) (_, c2) -> compare c2 c1) items
            in
            let values =
              Array.of_list (List.map (fun x -> Some x) (List.map fst items))
            in
            let counts_arr = Array.of_list (List.map snd items) in
            (Col.B values, counts_arr))
    | None -> (Col.S [||], [||])

  let is_null t name =
    match get_column t name with
    | Some (Col.P (Nx.Float32, tensor, _)) ->
        let arr : float array = Nx.to_array tensor in
        Array.map (fun x -> classify_float x = FP_nan) arr
    | Some (Col.P (Nx.Float64, tensor, _)) ->
        let arr : float array = Nx.to_array tensor in
        Array.map (fun x -> classify_float x = FP_nan) arr
    | Some (Col.P (Nx.Int32, tensor, _)) ->
        let arr : int32 array = Nx.to_array tensor in
        Array.map (fun x -> x = Int32.min_int) arr
    | Some (Col.P (Nx.Int64, tensor, _)) ->
        let arr : int64 array = Nx.to_array tensor in
        Array.map (fun x -> x = Int64.min_int) arr
    | Some (Col.P _) -> Array.make (num_rows t) false
    | Some (Col.S arr) -> Array.map Option.is_none arr
    | Some (Col.B arr) -> Array.map Option.is_none arr
    | None -> [||]

  let cumsum t name =
    match get_column t name with
    | Some (Col.P (dtype, tensor, _)) -> (
        match dtype with
        | Nx.Float32 ->
            let arr : float array = Nx.to_array tensor in
            let result = Array.copy arr in
            for i = 1 to Array.length result - 1 do
              result.(i) <- result.(i - 1) +. result.(i)
            done;
            Col.P (dtype, Nx.create dtype [| Array.length result |] result, None)
        | Nx.Float64 ->
            let arr : float array = Nx.to_array tensor in
            let result = Array.copy arr in
            for i = 1 to Array.length result - 1 do
              result.(i) <- result.(i - 1) +. result.(i)
            done;
            Col.P (dtype, Nx.create dtype [| Array.length result |] result, None)
        | Nx.Int32 ->
            let arr : int32 array = Nx.to_array tensor in
            let result = Array.copy arr in
            for i = 1 to Array.length result - 1 do
              result.(i) <- Int32.add result.(i - 1) result.(i)
            done;
            Col.P (dtype, Nx.create dtype [| Array.length result |] result, None)
        | Nx.Int64 ->
            let arr : int64 array = Nx.to_array tensor in
            let result = Array.copy arr in
            for i = 1 to Array.length result - 1 do
              result.(i) <- Int64.add result.(i - 1) result.(i)
            done;
            Col.P (dtype, Nx.create dtype [| Array.length result |] result, None)
        | _ -> failwith "cumsum: unsupported numeric type")
    | _ -> failwith "cumsum: column must be numeric"

  let cumprod t name =
    match get_column t name with
    | Some (Col.P (dtype, tensor, _)) -> (
        match dtype with
        | Nx.Float32 ->
            let arr : float array = Nx.to_array tensor in
            let result = Array.copy arr in
            for i = 1 to Array.length result - 1 do
              result.(i) <- result.(i - 1) *. result.(i)
            done;
            Col.P (dtype, Nx.create dtype [| Array.length result |] result, None)
        | Nx.Float64 ->
            let arr : float array = Nx.to_array tensor in
            let result = Array.copy arr in
            for i = 1 to Array.length result - 1 do
              result.(i) <- result.(i - 1) *. result.(i)
            done;
            Col.P (dtype, Nx.create dtype [| Array.length result |] result, None)
        | Nx.Int32 ->
            let arr : int32 array = Nx.to_array tensor in
            let result = Array.copy arr in
            for i = 1 to Array.length result - 1 do
              result.(i) <- Int32.mul result.(i - 1) result.(i)
            done;
            Col.P (dtype, Nx.create dtype [| Array.length result |] result, None)
        | Nx.Int64 ->
            let arr : int64 array = Nx.to_array tensor in
            let result = Array.copy arr in
            for i = 1 to Array.length result - 1 do
              result.(i) <- Int64.mul result.(i - 1) result.(i)
            done;
            Col.P (dtype, Nx.create dtype [| Array.length result |] result, None)
        | _ -> failwith "cumprod: unsupported numeric type")
    | _ -> failwith "cumprod: column must be numeric"

  let diff t name ?periods () =
    let periods = Option.value periods ~default:1 in
    match get_column t name with
    | Some (Col.P (dtype, tensor, _)) -> (
        match dtype with
        | Nx.Float32 ->
            let arr : float array = Nx.to_array tensor in
            let n = Array.length arr in
            let result = Array.make n 0. in
            for i = periods to n - 1 do
              result.(i) <- arr.(i) -. arr.(i - periods)
            done;
            Col.P (dtype, Nx.create dtype [| n |] result, None)
        | Nx.Float64 ->
            let arr : float array = Nx.to_array tensor in
            let n = Array.length arr in
            let result = Array.make n 0. in
            for i = periods to n - 1 do
              result.(i) <- arr.(i) -. arr.(i - periods)
            done;
            Col.P (dtype, Nx.create dtype [| n |] result, None)
        | Nx.Int32 ->
            let arr : int32 array = Nx.to_array tensor in
            let n = Array.length arr in
            let result = Array.make n 0l in
            for i = periods to n - 1 do
              result.(i) <- Int32.sub arr.(i) arr.(i - periods)
            done;
            Col.P (dtype, Nx.create dtype [| n |] result, None)
        | Nx.Int64 ->
            let arr : int64 array = Nx.to_array tensor in
            let n = Array.length arr in
            let result = Array.make n 0L in
            for i = periods to n - 1 do
              result.(i) <- Int64.sub arr.(i) arr.(i - periods)
            done;
            Col.P (dtype, Nx.create dtype [| n |] result, None)
        | _ -> failwith "diff: unsupported numeric type")
    | _ -> failwith "diff: column must be numeric"

  let pct_change t name ?periods () =
    let periods = Option.value periods ~default:1 in
    match get_column t name with
    | Some (Col.P (dtype, tensor, _)) ->
        let n = Nx.size tensor in
        let result = Array.make n 0. in
        (match dtype with
        | Nx.Float32 ->
            let arr : float array = Nx.to_array tensor in
            for i = periods to n - 1 do
              let prev = arr.(i - periods) in
              let curr = arr.(i) in
              result.(i) <- (if prev = 0. then 0. else (curr -. prev) /. prev)
            done
        | Nx.Float64 ->
            let arr : float array = Nx.to_array tensor in
            for i = periods to n - 1 do
              let prev = arr.(i - periods) in
              let curr = arr.(i) in
              result.(i) <- (if prev = 0. then 0. else (curr -. prev) /. prev)
            done
        | Nx.Int32 ->
            let arr : int32 array = Nx.to_array tensor in
            for i = periods to n - 1 do
              let prev = Int32.to_float arr.(i - periods) in
              let curr = Int32.to_float arr.(i) in
              result.(i) <- (if prev = 0. then 0. else (curr -. prev) /. prev)
            done
        | Nx.Int64 ->
            let arr : int64 array = Nx.to_array tensor in
            for i = periods to n - 1 do
              let prev = Int64.to_float arr.(i - periods) in
              let curr = Int64.to_float arr.(i) in
              result.(i) <- (if prev = 0. then 0. else (curr -. prev) /. prev)
            done
        | _ -> failwith "pct_change: unsupported numeric type");
        Col.P (Nx.float64, Nx.create Nx.float64 [| n |] result, None)
    | _ -> failwith "pct_change: column must be numeric"

  let shift t name ~periods =
    match get_column t name with
    | Some (Col.P (dtype, tensor, _)) -> (
        match dtype with
        | Nx.Float32 ->
            let arr : float array = Nx.to_array tensor in
            let n = Array.length arr in
            let result = Array.make n nan in
            if periods > 0 then
              for i = periods to n - 1 do
                result.(i) <- arr.(i - periods)
              done
            else
              for i = 0 to n - 1 + periods do
                result.(i) <- arr.(i - periods)
              done;
            Col.P (dtype, Nx.create dtype [| n |] result, None)
        | Nx.Float64 ->
            let arr : float array = Nx.to_array tensor in
            let n = Array.length arr in
            let result = Array.make n nan in
            if periods > 0 then
              for i = periods to n - 1 do
                result.(i) <- arr.(i - periods)
              done
            else
              for i = 0 to n - 1 + periods do
                result.(i) <- arr.(i - periods)
              done;
            Col.P (dtype, Nx.create dtype [| n |] result, None)
        | Nx.Int32 ->
            let arr : int32 array = Nx.to_array tensor in
            let n = Array.length arr in
            let result = Array.make n Int32.min_int in
            if periods > 0 then
              for i = periods to n - 1 do
                result.(i) <- arr.(i - periods)
              done
            else
              for i = 0 to n - 1 + periods do
                result.(i) <- arr.(i - periods)
              done;
            Col.P (dtype, Nx.create dtype [| n |] result, None)
        | Nx.Int64 ->
            let arr : int64 array = Nx.to_array tensor in
            let n = Array.length arr in
            let result = Array.make n Int64.min_int in
            if periods > 0 then
              for i = periods to n - 1 do
                result.(i) <- arr.(i - periods)
              done
            else
              for i = 0 to n - 1 + periods do
                result.(i) <- arr.(i - periods)
              done;
            Col.P (dtype, Nx.create dtype [| n |] result, None)
        | _ -> failwith "shift: unsupported numeric type")
    | Some (Col.S arr) ->
        let n = Array.length arr in
        let result = Array.make n None in
        if periods > 0 then
          for i = periods to n - 1 do
            result.(i) <- arr.(i - periods)
          done
        else
          for i = 0 to n - 1 + periods do
            result.(i) <- arr.(i - periods)
          done;
        Col.S result
    | Some (Col.B arr) ->
        let n = Array.length arr in
        let result = Array.make n None in
        if periods > 0 then
          for i = periods to n - 1 do
            result.(i) <- arr.(i - periods)
          done
        else
          for i = 0 to n - 1 + periods do
            result.(i) <- arr.(i - periods)
          done;
        Col.B result
    | None -> failwith "shift: column not found"

  let fillna t name ~value =
    match (get_column t name, value) with
    | Some col, value_col
      when Col.length value_col = 1 || Col.length value_col = Col.length col ->
        col
    | _ ->
        invalid_arg
          "fillna: value column must have 1 element or match column length"
end

let rec join t1 t2 ~on ~how ?(suffixes = ("_x", "_y")) () =
  merge t1 t2 ~left_on:on ~right_on:on ~how ~suffixes ()

and merge t1 t2 ~left_on ~right_on ~how ?(suffixes = ("_x", "_y")) () =
  (* Get the join columns *)
  let left_col = get_column_exn t1 left_on in
  let right_col = get_column_exn t2 right_on in

  (* Extract values for comparison - handle different column types *)
  let get_key_array = function
    | Col.P (Nx.Int32, tensor, _) ->
        let arr : int32 array = Nx.to_array tensor in
        Array.map (fun x -> `Int32 x) arr
    | Col.P (Nx.Int64, tensor, _) ->
        let arr : int64 array = Nx.to_array tensor in
        Array.map (fun x -> `Int64 x) arr
    | Col.P (Nx.Float32, tensor, _) ->
        let arr : float array = Nx.to_array tensor in
        Array.map (fun x -> `Float x) arr
    | Col.P (Nx.Float64, tensor, _) ->
        let arr : float array = Nx.to_array tensor in
        Array.map (fun x -> `Float x) arr
    | Col.S arr -> Array.map (fun x -> `String x) arr
    | _ -> failwith "Unsupported column type for join"
  in

  let left_keys = get_key_array left_col in
  let right_keys = get_key_array right_col in

  (* Build index mappings for the join *)
  let left_indices = ref [] in
  let right_indices = ref [] in

  let process_join () =
    match how with
    | `Inner ->
        (* Inner join: only matching rows *)
        for i = 0 to Array.length left_keys - 1 do
          for j = 0 to Array.length right_keys - 1 do
            if left_keys.(i) = right_keys.(j) then (
              left_indices := i :: !left_indices;
              right_indices := j :: !right_indices)
          done
        done
    | `Left ->
        (* Left join: all rows from left, matching from right *)
        for i = 0 to Array.length left_keys - 1 do
          let found = ref false in
          for j = 0 to Array.length right_keys - 1 do
            if left_keys.(i) = right_keys.(j) then (
              left_indices := i :: !left_indices;
              right_indices := j :: !right_indices;
              found := true)
          done;
          if not !found then (
            left_indices := i :: !left_indices;
            right_indices := -1 :: !right_indices (* -1 indicates null *))
        done
    | `Right ->
        (* Right join: all rows from right, matching from left *)
        for j = 0 to Array.length right_keys - 1 do
          let found = ref false in
          for i = 0 to Array.length left_keys - 1 do
            if left_keys.(i) = right_keys.(j) then (
              left_indices := i :: !left_indices;
              right_indices := j :: !right_indices;
              found := true)
          done;
          if not !found then (
            left_indices := -1 :: !left_indices;
            (* -1 indicates null *)
            right_indices := j :: !right_indices)
        done
    | `Outer ->
        (* Outer join: all rows from both *)
        (* First, do a left join *)
        for i = 0 to Array.length left_keys - 1 do
          let found = ref false in
          for j = 0 to Array.length right_keys - 1 do
            if left_keys.(i) = right_keys.(j) then (
              left_indices := i :: !left_indices;
              right_indices := j :: !right_indices;
              found := true)
          done;
          if not !found then (
            left_indices := i :: !left_indices;
            right_indices := -1 :: !right_indices)
        done;
        (* Then add unmatched right rows *)
        for j = 0 to Array.length right_keys - 1 do
          let matched = ref false in
          for i = 0 to Array.length left_keys - 1 do
            if left_keys.(i) = right_keys.(j) then matched := true
          done;
          if not !matched then (
            left_indices := -1 :: !left_indices;
            right_indices := j :: !right_indices)
        done
  in

  process_join ();

  let left_idx = Array.of_list (List.rev !left_indices) in
  let right_idx = Array.of_list (List.rev !right_indices) in

  (* Helper to reindex a column based on indices *)
  let reindex_column col indices n_source =
    match col with
    | Col.P (dtype, tensor, _) -> (
        match dtype with
        | Nx.Float32 ->
            let source_arr : float array = Nx.to_array tensor in
            let result =
              Array.init (Array.length indices) (fun i ->
                  let idx = indices.(i) in
                  if idx >= 0 && idx < n_source then source_arr.(idx)
                  else Float.nan)
            in
            Col.P (dtype, Nx.create dtype [| Array.length result |] result, None)
        | Nx.Float64 ->
            let source_arr : float array = Nx.to_array tensor in
            let result =
              Array.init (Array.length indices) (fun i ->
                  let idx = indices.(i) in
                  if idx >= 0 && idx < n_source then source_arr.(idx)
                  else Float.nan)
            in
            Col.P (dtype, Nx.create dtype [| Array.length result |] result, None)
        | Nx.Int32 ->
            let source_arr : int32 array = Nx.to_array tensor in
            let result =
              Array.init (Array.length indices) (fun i ->
                  let idx = indices.(i) in
                  if idx >= 0 && idx < n_source then source_arr.(idx)
                  else Int32.min_int)
            in
            Col.P (dtype, Nx.create dtype [| Array.length result |] result, None)
        | Nx.Int64 ->
            let source_arr : int64 array = Nx.to_array tensor in
            let result =
              Array.init (Array.length indices) (fun i ->
                  let idx = indices.(i) in
                  if idx >= 0 && idx < n_source then source_arr.(idx)
                  else Int64.min_int)
            in
            Col.P (dtype, Nx.create dtype [| Array.length result |] result, None)
        | _ ->
            (* For other types, just create zeros *)
            let source_arr = Nx.to_array tensor in
            let result =
              Array.init (Array.length indices) (fun i ->
                  let idx = indices.(i) in
                  if idx >= 0 && idx < n_source then source_arr.(idx)
                  else source_arr.(0)
                  (* Use first element as default *))
            in
            Col.P (dtype, Nx.create dtype [| Array.length result |] result, None)
        )
    | Col.S arr ->
        let result =
          Array.init (Array.length indices) (fun i ->
              let idx = indices.(i) in
              if idx >= 0 && idx < Array.length arr then arr.(idx) else None)
        in
        Col.S result
    | Col.B arr ->
        let result =
          Array.init (Array.length indices) (fun i ->
              let idx = indices.(i) in
              if idx >= 0 && idx < Array.length arr then arr.(idx) else None)
        in
        Col.B result
  in

  (* Build result columns *)
  let result_cols = ref [] in

  (* Add columns from left table *)
  let left_suffix, right_suffix = suffixes in
  List.iter
    (fun name ->
      let col = get_column_exn t1 name in
      let new_col = reindex_column col left_idx (num_rows t1) in
      (* Check if this column name exists in right table (excluding join
         column) *)
      let final_name =
        if name <> left_on && has_column t2 name then name ^ left_suffix
        else name
      in
      result_cols := (final_name, new_col) :: !result_cols)
    (column_names t1);

  (* Add columns from right table (excluding the join column for join, not
     merge) *)
  List.iter
    (fun name ->
      if name <> right_on || left_on <> right_on then
        let col = get_column_exn t2 name in
        let new_col = reindex_column col right_idx (num_rows t2) in
        (* Check if this column name already exists *)
        let final_name =
          if has_column t1 name then name ^ right_suffix else name
        in
        result_cols := (final_name, new_col) :: !result_cols)
    (column_names t2);

  create (List.rev !result_cols)

let pivot t ~index ~columns ~values ?(agg_func = `Sum) () =
  (* Get unique values from the columns column to become new column names *)
  let col_col = get_column_exn t columns in
  let unique_cols =
    match col_col with
    | Col.S arr ->
        let seen = Hashtbl.create 16 in
        Array.fold_left
          (fun acc opt ->
            match opt with
            | Some s when not (Hashtbl.mem seen s) ->
                Hashtbl.add seen s ();
                s :: acc
            | _ -> acc)
          [] arr
        |> List.rev
    | Col.P (Nx.Int32, tensor, _) ->
        let arr : int32 array = Nx.to_array tensor in
        let seen = Hashtbl.create 16 in
        Array.fold_left
          (fun acc v ->
            let s = Int32.to_string v in
            if not (Hashtbl.mem seen s) then (
              Hashtbl.add seen s ();
              s :: acc)
            else acc)
          [] arr
        |> List.rev
    | _ -> failwith "pivot: columns must be string or int32"
  in

  (* Get unique index values *)
  let index_col = get_column_exn t index in
  let index_values =
    match index_col with
    | Col.S arr -> Array.to_list arr
    | Col.P (_, tensor, _) ->
        let arr = Nx.to_array tensor in
        Array.to_list (Array.map (fun _ -> Some "") arr)
        (* Simplified *)
    | _ -> failwith "pivot: index column type not supported"
  in

  (* Group data by index and column values *)
  let groups = Hashtbl.create 16 in
  let values_col = get_column_exn t values in

  for i = 0 to num_rows t - 1 do
    let idx_key =
      match List.nth_opt index_values i with Some (Some s) -> s | _ -> ""
    in
    let col_key =
      match col_col with
      | Col.S arr -> ( match arr.(i) with Some s -> s | None -> "")
      | Col.P (Nx.Int32, tensor, _) ->
          let arr : int32 array = Nx.to_array tensor in
          Int32.to_string arr.(i)
      | _ -> ""
    in
    let key = (idx_key, col_key) in

    let value =
      match values_col with
      | Col.P (Nx.Float64, tensor, _) ->
          let arr : float array = Nx.to_array tensor in
          arr.(i)
      | Col.P (Nx.Float32, tensor, _) ->
          let arr : float array = Nx.to_array tensor in
          arr.(i)
      | Col.P (Nx.Int32, tensor, _) ->
          let arr : int32 array = Nx.to_array tensor in
          Int32.to_float arr.(i)
      | _ -> 0.0
    in

    let current = try Hashtbl.find groups key with Not_found -> [] in
    Hashtbl.replace groups key (value :: current)
  done;

  (* Aggregate values *)
  let aggregate values =
    match agg_func with
    | `Sum -> List.fold_left ( +. ) 0.0 values
    | `Mean ->
        let sum = List.fold_left ( +. ) 0.0 values in
        sum /. float_of_int (List.length values)
    | `Count -> float_of_int (List.length values)
    | `Min -> List.fold_left min Float.infinity values
    | `Max -> List.fold_left max Float.neg_infinity values
  in

  (* Build unique index values for result *)
  let unique_indices =
    let seen = Hashtbl.create 16 in
    List.fold_left
      (fun acc opt ->
        match opt with
        | Some s when not (Hashtbl.mem seen s) ->
            Hashtbl.add seen s ();
            s :: acc
        | _ -> acc)
      [] index_values
    |> List.rev
  in

  (* Build result columns *)
  let result_cols =
    ref [ (index, Col.string (Array.of_list unique_indices)) ]
  in

  List.iter
    (fun col_name ->
      let col_values =
        List.map
          (fun idx ->
            let key = (idx, col_name) in
            try
              let values = Hashtbl.find groups key in
              aggregate values
            with Not_found -> Float.nan)
          unique_indices
      in
      result_cols :=
        (col_name, Col.float64 (Array.of_list col_values)) :: !result_cols)
    unique_cols;

  create (List.rev !result_cols)

let melt t ?(id_vars = []) ?(value_vars = []) ?(var_name = "variable")
    ?(value_name = "value") () =
  (* If value_vars is empty, use all columns except id_vars *)
  let value_columns =
    if value_vars = [] then
      List.filter (fun name -> not (List.mem name id_vars)) (column_names t)
    else value_vars
  in

  let n_rows = num_rows t in
  let n_value_cols = List.length value_columns in
  let total_rows = n_rows * n_value_cols in

  (* Prepare result columns *)
  let result_cols = ref [] in

  (* Add id columns - replicate each row n_value_cols times *)
  List.iter
    (fun id_name ->
      let col = get_column_exn t id_name in
      let new_col =
        match col with
        | Col.P (dtype, tensor, _) ->
            let arr = Nx.to_array tensor in
            let result =
              Array.init total_rows (fun i -> arr.(i / n_value_cols))
            in
            Col.P (dtype, Nx.create dtype [| total_rows |] result, None)
        | Col.S arr ->
            let result =
              Array.init total_rows (fun i -> arr.(i / n_value_cols))
            in
            Col.S result
        | Col.B arr ->
            let result =
              Array.init total_rows (fun i -> arr.(i / n_value_cols))
            in
            Col.B result
      in
      result_cols := (id_name, new_col) :: !result_cols)
    id_vars;

  (* Add variable name column *)
  let var_col_values =
    Array.init total_rows (fun i ->
        Some (List.nth value_columns (i mod n_value_cols)))
  in
  result_cols := (var_name, Col.S var_col_values) :: !result_cols;

  (* Add value column - collect all values from value_columns *)
  let value_arrays =
    List.map
      (fun col_name ->
        match get_column_exn t col_name with
        | Col.P (_, tensor, _) ->
            let arr : float array = Nx.to_array (Nx.astype Nx.float64 tensor) in
            arr
        | _ -> Array.make n_rows Float.nan)
      value_columns
  in

  let value_col_data =
    Array.init total_rows (fun i ->
        let row = i / n_value_cols in
        let col = i mod n_value_cols in
        (List.nth value_arrays col).(row))
  in
  result_cols := (value_name, Col.float64 value_col_data) :: !result_cols;

  create (List.rev !result_cols)

let to_nx t =
  let numeric_tensors =
    List.filter_map
      (fun (_name, col) ->
        match col with
        | Col.P (_, tensor, _) ->
            let float_tensor = Nx.astype Nx.float32 tensor in
            Some float_tensor
        | _ -> None)
      t.columns
  in
  if numeric_tensors = [] then invalid_arg "to_nx: no numeric columns"
  else
    let stacked = Nx.stack numeric_tensors ~axis:1 in
    stacked

let print ?max_rows ?max_cols t =
  let max_rows = Option.value max_rows ~default:10 in
  let max_cols = Option.value max_cols ~default:10 in
  let n_rows = num_rows t in
  let n_cols = num_columns t in
  let rows_to_show = min max_rows n_rows in
  let cols_to_show = min max_cols n_cols in
  let names = column_names t in
  let names_to_show = List.take cols_to_show names in
  Printf.printf "Shape: (%d, %d)\n" n_rows n_cols;
  Printf.printf "%s\n" (String.concat "\t" names_to_show);
  for i = 0 to rows_to_show - 1 do
    let row =
      List.map
        (fun name ->
          match get_column t name with
          | Some (Col.P (dtype, tensor, _)) -> (
              match dtype with
              | Nx.Float16 ->
                  let arr : float array = Nx.to_array tensor in
                  string_of_float arr.(i)
              | Nx.BFloat16 ->
                  let arr : float array = Nx.to_array tensor in
                  string_of_float arr.(i)
              | Nx.Float32 ->
                  let arr : float array = Nx.to_array tensor in
                  string_of_float arr.(i)
              | Nx.Float64 ->
                  let arr : float array = Nx.to_array tensor in
                  string_of_float arr.(i)
              | Nx.Int8 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.UInt8 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.Int16 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.UInt16 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.Int32 ->
                  let arr : int32 array = Nx.to_array tensor in
                  Int32.to_string arr.(i)
              | Nx.Int64 ->
                  let arr : int64 array = Nx.to_array tensor in
                  Int64.to_string arr.(i)
              | Nx.Int ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.NativeInt ->
                  let arr : nativeint array = Nx.to_array tensor in
                  Nativeint.to_string arr.(i)
              | Nx.Complex32 ->
                  let arr : Complex.t array = Nx.to_array tensor in
                  let c = arr.(i) in
                  Printf.sprintf "(%g+%gi)" c.re c.im
              | Nx.Complex64 ->
                  let arr : Complex.t array = Nx.to_array tensor in
                  let c = arr.(i) in
                  Printf.sprintf "(%g+%gi)" c.re c.im
              | Nx.Bool ->
                  let arr : bool array = Nx.to_array tensor in
                  string_of_bool arr.(i)
              | Nx.Int4 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.UInt4 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.Float8_e4m3 ->
                  let arr : float array = Nx.to_array tensor in
                  string_of_float arr.(i)
              | Nx.Float8_e5m2 ->
                  let arr : float array = Nx.to_array tensor in
                  string_of_float arr.(i)
              | Nx.Complex16 ->
                  let arr : Complex.t array = Nx.to_array tensor in
                  let c = arr.(i) in
                  Printf.sprintf "(%g+%gi)" c.re c.im
              | Nx.QInt8 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i)
              | Nx.QUInt8 ->
                  let arr : int array = Nx.to_array tensor in
                  string_of_int arr.(i))
          | Some (Col.S arr) -> Option.value arr.(i) ~default:"<null>"
          | Some (Col.B arr) ->
              Option.value (Option.map string_of_bool arr.(i)) ~default:"<null>"
          | None -> "")
        names_to_show
    in
    Printf.printf "%s\n" (String.concat "\t" row)
  done

let describe t =
  let numeric_cols =
    List.filter_map
      (fun (name, col) -> match col with Col.P _ -> Some name | _ -> None)
      t.columns
  in
  let stats = [ "count"; "mean"; "std"; "min"; "25%"; "50%"; "75%"; "max" ] in
  let data =
    List.map
      (fun stat ->
        ( stat,
          Col.string_list
            (List.map
               (fun col_name ->
                 match stat with
                 | "count" -> string_of_int (Agg.count t col_name)
                 | "mean" -> string_of_float (Agg.Float.mean t col_name)
                 | "std" -> string_of_float (Agg.Float.std t col_name)
                 | "min" -> (
                     match Agg.Float.min t col_name with
                     | Some v -> string_of_float v
                     | None -> "NaN")
                 | "25%" ->
                     string_of_float (Agg.Float.quantile t col_name ~q:0.25)
                 | "50%" -> string_of_float (Agg.Float.median t col_name)
                 | "75%" ->
                     string_of_float (Agg.Float.quantile t col_name ~q:0.75)
                 | "max" -> (
                     match Agg.Float.max t col_name with
                     | Some v -> string_of_float v
                     | None -> "NaN")
                 | _ -> "")
               numeric_cols) ))
      stats
  in
  create data

let cast_column t name dtype =
  match get_column t name with
  | Some (Col.P (_, tensor, mask)) ->
      let casted = Nx.astype dtype tensor in
      add_column t name (Col.P (dtype, casted, mask))
  | _ -> invalid_arg "cast_column: conversion not possible"

let info t =
  let n_rows, n_cols = shape t in
  Printf.printf "DataFrame info:\n";
  Printf.printf "  Rows: %d\n" n_rows;
  Printf.printf "  Columns: %d\n" n_cols;
  Printf.printf "\nColumn types:\n";
  List.iter
    (fun (name, typ) ->
      let typ_str =
        match typ with
        | `Float32 -> "float32"
        | `Float64 -> "float64"
        | `Int32 -> "int32"
        | `Int64 -> "int64"
        | `Bool -> "bool"
        | `String -> "string"
        | `Other -> "other"
      in
      Printf.printf "  %s: %s\n" name typ_str)
    (column_types t)
