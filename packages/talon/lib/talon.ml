(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let list_take n l =
  let[@tail_mod_cons] rec aux n l =
    match (n, l) with 0, _ | _, [] -> [] | n, x :: l -> x :: aux (n - 1) l
  in
  if n <= 0 then [] else aux n l

module Col = Col

type t = {
  columns : (string * Col.t) list;
  column_map : (string, Col.t) Hashtbl.t;
}

type 'a row = { f : t -> int -> 'a }

(* Internal helpers *)

let get_column t name = Hashtbl.find_opt t.column_map name

let get_column_exn t name =
  match get_column t name with Some col -> col | None -> raise Not_found

(* Creation *)

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

let of_nx ?names tensor =
  match Nx.shape tensor with
  | [| _rows; cols |] ->
      let tensors =
        List.init cols (fun col_i -> Nx.slice [ Nx.A; Nx.I col_i ] tensor)
      in
      of_tensors ?names tensors
  | _ -> invalid_arg "of_nx: tensor must be 2D"

(* Shape *)

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
        | Col.P _ -> `Other
        | Col.S _ -> `String
        | Col.B _ -> `Bool
      in
      (name, typ))
    t.columns

let is_empty t = num_rows t = 0

let select_columns t category =
  List.filter_map
    (fun (name, typ) ->
      let keep =
        match (category, typ) with
        | `Numeric, (`Float32 | `Float64 | `Int32 | `Int64) -> true
        | `Float, (`Float32 | `Float64) -> true
        | `Int, (`Int32 | `Int64) -> true
        | `Bool, `Bool -> true
        | `String, `String -> true
        | _ -> false
      in
      if keep then Some name else None)
    (column_types t)

(* Column access *)

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

let select ?(strict = true) t names =
  if strict then
    let columns =
      List.map
        (fun name ->
          match get_column t name with
          | Some col -> (name, col)
          | None -> raise Not_found)
        names
    in
    create columns
  else
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
  let requested = List.map (fun name -> (name, get_column_exn t name)) names in
  let remaining =
    List.filter (fun (name, _) -> not (List.mem name names)) t.columns
  in
  create (requested @ remaining)

let cast_column t name dtype =
  match get_column t name with
  | Some (Col.P (_, tensor, mask)) ->
      let casted = Nx.astype dtype tensor in
      add_column t name (Col.P (dtype, casted, mask))
  | _ -> invalid_arg "cast_column: conversion not possible"

(* Extraction *)

let to_array (type a b) (dtype : (a, b) Nx.dtype) t name =
  get_column t name
  |> Option.map (fun col -> Col.to_tensor dtype col)
  |> Option.join |> Option.map Nx.to_array

let to_opt_array (type a b) (dtype : (a, b) Nx.dtype) t name =
  get_column t name
  |> Option.map (fun col ->
      match Col.to_tensor dtype col with
      | None -> None
      | Some tensor ->
          let arr = Nx.to_array tensor in
          let mask = Col.null_mask col in
          Some
            (Array.mapi
               (fun i v ->
                 match mask with Some m when m.(i) -> None | _ -> Some v)
               arr))
  |> Option.join

let to_bool_array t name =
  get_column t name |> Option.map Col.to_bool_array |> Option.join

let to_string_array t name =
  get_column t name |> Option.map Col.to_string_array |> Option.join

(* Row module *)

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

  let cached ~extract =
    let cache = ref None in
    {
      f =
        (fun df i ->
          let a =
            match !cache with
            | Some (df', a) when df' == df -> a
            | _ ->
                let a = extract df in
                cache := Some (df, a);
                a
          in
          a.(i));
    }

  let cached_masked ~extract =
    let cache = ref None in
    {
      f =
        (fun df i ->
          let a, mask_opt =
            match !cache with
            | Some (df', a, m) when df' == df -> (a, m)
            | _ ->
                let a, m = extract df in
                cache := Some (df, a, m);
                (a, m)
          in
          match mask_opt with
          | Some mask when mask.(i) -> None
          | _ -> Some a.(i));
    }

  let col (type a b) (dtype : (a, b) Nx.dtype) name =
    cached ~extract:(fun df ->
        match Col.to_tensor dtype (get_column_exn df name) with
        | Some tensor -> (Nx.to_array tensor : a array)
        | None -> failwith ("Column " ^ name ^ " has incompatible dtype"))

  let col_opt (type a b) (dtype : (a, b) Nx.dtype) name =
    cached_masked ~extract:(fun df ->
        let c = get_column_exn df name in
        match Col.to_tensor dtype c with
        | Some tensor -> ((Nx.to_array tensor : a array), Col.null_mask c)
        | None -> failwith ("Column " ^ name ^ " has incompatible dtype"))

  let string name =
    map
      (cached ~extract:(fun df ->
           match get_column df name with
           | Some (Col.S a) -> a
           | _ -> failwith ("Column " ^ name ^ " is not string")))
      ~f:(Option.value ~default:"")

  let string_opt name =
    cached ~extract:(fun df ->
        match get_column df name with
        | Some (Col.S a) -> a
        | _ -> failwith ("Column " ^ name ^ " is not string"))

  let bool name =
    map
      (cached ~extract:(fun df ->
           match get_column df name with
           | Some (Col.B a) -> a
           | _ -> failwith ("Column " ^ name ^ " is not bool")))
      ~f:(Option.value ~default:false)

  let bool_opt name =
    cached ~extract:(fun df ->
        match get_column df name with
        | Some (Col.B a) -> a
        | _ -> failwith ("Column " ^ name ^ " is not bool"))

  let number name =
    cached ~extract:(fun df ->
        match get_column df name with
        | Some (Col.P (Nx.Float32, tensor, _)) ->
            (Nx.to_array tensor : float array)
        | Some (Col.P (Nx.Float64, tensor, _)) ->
            (Nx.to_array tensor : float array)
        | Some (Col.P (Nx.Int32, tensor, _)) ->
            Array.map Int32.to_float (Nx.to_array tensor)
        | Some (Col.P (Nx.Int64, tensor, _)) ->
            Array.map Int64.to_float (Nx.to_array tensor)
        | Some _ -> failwith ("Column " ^ name ^ " is not numeric")
        | None -> failwith ("Column " ^ name ^ " not found"))

  let float32 name = col Nx.float32 name
  let float64 name = col Nx.float64 name
  let int32 name = col Nx.int32 name
  let int64 name = col Nx.int64 name
  let float32_opt name = col_opt Nx.float32 name
  let float64_opt name = col_opt Nx.float64 name
  let int32_opt name = col_opt Nx.int32 name
  let int64_opt name = col_opt Nx.int64 name
  let index = { f = (fun _ i -> i) }
  let sequence xs = { f = (fun df i -> List.map (fun x -> x.f df i) xs) }

  let fold_list xs ~init ~f =
    { f = (fun df i -> List.fold_left (fun acc x -> f acc (x.f df i)) init xs) }
end

(* Internal: reindex rows by array of non-negative indices *)

let reindex_rows t indices =
  List.map (fun (name, col) -> (name, Col.reindex col indices)) t.columns
  |> create

(* Slicing and filtering *)

let head ?(n = 5) t =
  let actual_n = min n (num_rows t) in
  let columns =
    List.map (fun (name, col) -> (name, Col.slice_col col 0 actual_n)) t.columns
  in
  create columns

let tail ?(n = 5) t =
  let n_rows = num_rows t in
  let actual_n = min n n_rows in
  let start = n_rows - actual_n in
  let columns =
    List.map
      (fun (name, col) -> (name, Col.slice_col col start actual_n))
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
      (fun (name, col) -> (name, Col.slice_col col start length))
      t.columns
  in
  create columns

let sample ?n ?frac ?replace ?seed t =
  let n_rows = num_rows t in
  let sample_size =
    match (n, frac) with
    | Some n, None -> n
    | None, Some f -> int_of_float (f *. float_of_int n_rows)
    | _ -> invalid_arg "sample: either n or frac must be specified"
  in
  let replace = Option.value replace ~default:false in
  let state =
    match seed with
    | Some s -> Random.State.make [| s |]
    | None -> Random.State.make_self_init ()
  in
  let indices =
    if replace then
      Array.init sample_size (fun _ -> Random.State.int state n_rows)
    else
      let all_indices = Array.init n_rows Fun.id in
      for i = n_rows - 1 downto 1 do
        let j = Random.State.int state (i + 1) in
        let temp = all_indices.(i) in
        all_indices.(i) <- all_indices.(j);
        all_indices.(j) <- temp
      done;
      Array.sub all_indices 0 (min sample_size n_rows)
  in
  reindex_rows t indices

let filter t mask =
  let n_rows = num_rows t in
  if Array.length mask <> n_rows then
    invalid_arg "filter: mask length must match num_rows"
  else
    let indices = ref [] in
    Array.iteri (fun i b -> if b then indices := i :: !indices) mask;
    let indices = Array.of_list (List.rev !indices) in
    reindex_rows t indices

let filter_by t pred =
  let n_rows = num_rows t in
  let mask = Array.init n_rows (fun i -> pred.f t i) in
  filter t mask

let drop_nulls ?subset t =
  let cols_to_check =
    match subset with Some cols -> cols | None -> column_names t
  in
  let mask = Array.make (num_rows t) true in
  List.iter
    (fun col_name ->
      match get_column t col_name with
      | Some (Col.P (_, _, Some null_mask)) ->
          Array.iteri
            (fun i is_null -> if is_null then mask.(i) <- false)
            null_mask
      | Some (Col.P (_, _, None)) -> ()
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

let fill_null t col_name ~with_value =
  match get_column t col_name with
  | None -> invalid_arg ("fill_null: column " ^ col_name ^ " not found")
  | Some col ->
      let value_col =
        match (with_value, Col.dtype col) with
        | `Float v, `Float32 -> Col.float32 [| v |]
        | `Float v, `Float64 -> Col.float64 [| v |]
        | `Float v, _ -> Col.float64 [| v |]
        | `Int32 v, _ -> Col.int32 [| v |]
        | `Int64 v, _ -> Col.int64 [| v |]
        | `String v, _ -> Col.string [| v |]
        | `Bool v, _ -> Col.bool [| v |]
      in
      let filled = Col.fill_nulls col ~value:value_col in
      add_column t col_name filled

let drop_duplicates ?subset t =
  let cols_to_check =
    match subset with None -> column_names t | Some names -> names
  in
  let n_rows = num_rows t in
  let seen = Hashtbl.create n_rows in
  let unique_indices = ref [] in
  let fmts =
    List.map
      (fun name ->
        match get_column t name with
        | Some col -> Col.to_string_fn col
        | None -> fun _ -> "")
      cols_to_check
  in
  for i = 0 to n_rows - 1 do
    let key_str = String.concat "\x00" (List.map (fun f -> f i) fmts) in
    if not (Hashtbl.mem seen key_str) then (
      Hashtbl.add seen key_str ();
      unique_indices := i :: !unique_indices)
  done;
  let indices = Array.of_list (List.rev !unique_indices) in
  reindex_rows t indices

(* Transforms *)

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
                (name, Col.concat_cols cols))
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

let with_string_column t name f =
  let n_rows = num_rows t in
  let data = Array.init n_rows (fun i -> Some (f.f t i)) in
  add_column t name (Col.S data)

let with_bool_column t name f =
  let n_rows = num_rows t in
  let data = Array.init n_rows (fun i -> Some (f.f t i)) in
  add_column t name (Col.B data)

let with_columns t cols =
  List.fold_left (fun df (name, col) -> add_column df name col) t cols

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

(* Sorting and grouping *)

let sort t key ~compare =
  let n_rows = num_rows t in
  let keys = Array.init n_rows (fun i -> (i, key.f t i)) in
  Array.sort (fun (_, k1) (_, k2) -> compare k1 k2) keys;
  let indices = Array.map fst keys in
  reindex_rows t indices

let sort_values ?(ascending = true) t name =
  match get_column t name with
  | None -> raise Not_found
  | Some col -> (
      let cmp = if ascending then compare else fun a b -> compare b a in
      match col with
      | Col.P (Nx.Float32, _, _) | Col.P (Nx.Float64, _, _) ->
          sort t (Row.number name) ~compare:cmp
      | Col.P (Nx.Int32, _, _) -> sort t (Row.col Nx.int32 name) ~compare:cmp
      | Col.P (Nx.Int64, _, _) -> sort t (Row.col Nx.int64 name) ~compare:cmp
      | Col.S _ -> sort t (Row.string name) ~compare:cmp
      | Col.B _ -> sort t (Row.bool name) ~compare:cmp
      | _ -> failwith "sort_values: unsupported column type")

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
      (k, reindex_rows t indices) :: acc)
    groups []

(* Column transforms — delegate to Col, return dataframe *)

let cumsum t name = add_column t name (Col.cumsum (get_column_exn t name))
let cumprod t name = add_column t name (Col.cumprod (get_column_exn t name))

let diff t name ?periods () =
  add_column t name (Col.diff ?periods (get_column_exn t name))

let pct_change t name ?periods () =
  add_column t name (Col.pct_change ?periods (get_column_exn t name))

let shift t name ~periods =
  add_column t name (Col.shift ~periods (get_column_exn t name))

(* Column inspection *)

let is_null t name =
  match get_column t name with
  | Some (Col.P (_, _, Some mask)) -> Col.B (Array.map (fun b -> Some b) mask)
  | Some (Col.P _) -> Col.B (Array.make (num_rows t) (Some false))
  | Some (Col.S arr) -> Col.B (Array.map (fun x -> Some (Option.is_none x)) arr)
  | Some (Col.B arr) -> Col.B (Array.map (fun x -> Some (Option.is_none x)) arr)
  | None -> Col.B [||]

let value_counts_typed (type a) (tbl : (a, int) Hashtbl.t) arr
    (mask_opt : bool array option) =
  let is_null i = match mask_opt with Some m -> m.(i) | None -> false in
  Array.iteri
    (fun i x ->
      if not (is_null i) then
        let c = Option.value (Hashtbl.find_opt tbl x) ~default:0 in
        Hashtbl.replace tbl x (c + 1))
    arr;
  let items = Hashtbl.fold (fun k v acc -> (k, v) :: acc) tbl [] in
  let items = List.sort (fun (_, c1) (_, c2) -> compare c2 c1) items in
  (Array.of_list (List.map fst items), Array.of_list (List.map snd items))

let count_options ~wrap arr =
  let tbl = Hashtbl.create 16 in
  Array.iter
    (function
      | Some x ->
          let c = Option.value (Hashtbl.find_opt tbl x) ~default:0 in
          Hashtbl.replace tbl x (c + 1)
      | None -> ())
    arr;
  let items = Hashtbl.fold (fun k v acc -> (k, v) :: acc) tbl [] in
  let items = List.sort (fun (_, c1) (_, c2) -> compare c2 c1) items in
  let values = Array.of_list (List.map (fun (x, _) -> Some x) items) in
  let counts = Array.of_list (List.map snd items) in
  create
    [
      ("value", wrap values);
      ("count", Col.int32 (Array.map Int32.of_int counts));
    ]

let value_counts t name =
  match get_column t name with
  | Some col -> (
      match col with
      | Col.P (dtype, tensor, mask_opt) ->
          let arr = Nx.to_array tensor in
          let tbl = Hashtbl.create 16 in
          let values, counts_arr = value_counts_typed tbl arr mask_opt in
          let counts_int32 = Array.map Int32.of_int counts_arr in
          create
            [
              ( "value",
                Col.P
                  (dtype, Nx.create dtype [| Array.length values |] values, None)
              );
              ("count", Col.int32 counts_int32);
            ]
      | Col.S arr -> count_options ~wrap:(fun a -> Col.S a) arr
      | Col.B arr -> count_options ~wrap:(fun a -> Col.B a) arr)
  | None -> empty

(* Aggregations *)

module Agg = struct
  let sum t name =
    let filtered, _ = Col.col_as_float_array (get_column_exn t name) in
    Array.fold_left ( +. ) 0. filtered

  let mean t name =
    let filtered, count = Col.col_as_float_array (get_column_exn t name) in
    if count = 0 then Float.nan
    else Array.fold_left ( +. ) 0. filtered /. float_of_int count

  let variance_of col =
    let filtered, count = Col.col_as_float_array col in
    if count = 0 then Float.nan
    else
      let n = float_of_int count in
      let sum = ref 0. in
      let sum_sq = ref 0. in
      for i = 0 to Array.length filtered - 1 do
        let x = filtered.(i) in
        sum := !sum +. x;
        sum_sq := !sum_sq +. (x *. x)
      done;
      let mean = !sum /. n in
      (!sum_sq /. n) -. (mean *. mean)

  let std t name = sqrt (variance_of (get_column_exn t name))
  let var t name = variance_of (get_column_exn t name)

  let min t name =
    let filtered, count = Col.col_as_float_array (get_column_exn t name) in
    if count = 0 then None else Some (Array.fold_left min max_float filtered)

  let max t name =
    let filtered, count = Col.col_as_float_array (get_column_exn t name) in
    if count = 0 then None else Some (Array.fold_left max min_float filtered)

  let median t name =
    let filtered, count = Col.col_as_float_array (get_column_exn t name) in
    if count = 0 then Float.nan
    else (
      Array.sort compare filtered;
      let n = Array.length filtered in
      if n mod 2 = 0 then (filtered.((n / 2) - 1) +. filtered.(n / 2)) /. 2.
      else filtered.(n / 2))

  let quantile t name ~q =
    let filtered, count = Col.col_as_float_array (get_column_exn t name) in
    if count = 0 then Float.nan
    else (
      Array.sort compare filtered;
      let n = Array.length filtered in
      let pos = q *. float_of_int (n - 1) in
      let lower = int_of_float pos in
      let upper = Stdlib.min (lower + 1) (n - 1) in
      let weight = pos -. float_of_int lower in
      (filtered.(lower) *. (1. -. weight)) +. (filtered.(upper) *. weight))

  let count t name =
    match get_column t name with
    | Some col -> Col.length col - Col.null_count col
    | None -> 0

  let count_unique_options arr =
    let seen = Hashtbl.create 16 in
    Array.iter (function Some x -> Hashtbl.replace seen x () | None -> ()) arr;
    Hashtbl.length seen

  let nunique t name =
    let col = get_column t name in
    match col with
    | None -> 0
    | Some (Col.S arr) -> count_unique_options arr
    | Some (Col.B arr) -> count_unique_options arr
    | Some (Col.P _) ->
        (* Use col_as_float_array which already handles all numeric dtypes and
           respects the null mask *)
        let filtered, count = Col.col_as_float_array (Option.get col) in
        if count = 0 then 0
        else
          let seen = Hashtbl.create 16 in
          Array.iter
            (fun x -> Hashtbl.replace seen (Int64.bits_of_float x) ())
            filtered;
          Hashtbl.length seen

  (* Row-wise (horizontal) reductions *)

  let collect_as_float64 t names =
    List.fold_left
      (fun (acc : (float, Bigarray.float64_elt) Nx.t list) name ->
        match get_column t name with
        | Some (Col.P (_, tensor, mask_opt)) ->
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
                  Nx.where (Nx.cast Nx.bool mask_float) nan_tensor casted
              | None -> casted
            in
            result :: acc
        | _ -> acc)
      [] names
    |> List.rev

  let dot t ~names ~weights =
    if List.length names <> Array.length weights then
      invalid_arg "dot: number of columns must match number of weights";
    let tensors = collect_as_float64 t names in
    if tensors = [] then Col.float64 (Array.make (num_rows t) 0.)
    else
      let n_rows = num_rows t in
      let n_cols = List.length tensors in
      let arrs = List.map Nx.to_array tensors in
      let result = Array.make n_rows 0. in
      for i = 0 to n_rows - 1 do
        let sum = ref 0. in
        List.iteri
          (fun j arr ->
            if j < n_cols then
              let v = arr.(i) in
              if Float.is_finite v then sum := !sum +. (v *. weights.(j)))
          arrs;
        result.(i) <- !sum
      done;
      Col.float64 result

  let row_sum ?(skipna = true) t ~names =
    let tensors = collect_as_float64 t names in
    if tensors = [] then Col.numeric Nx.float64 (Array.make (num_rows t) 0.0)
    else
      let stacked = Nx.stack tensors ~axis:0 in
      let result =
        if skipna then
          let nan_mask = Nx.isnan stacked in
          let zeros = Nx.zeros_like stacked in
          let cleaned = Nx.where nan_mask zeros stacked in
          Nx.sum cleaned ~axes:[ 0 ]
        else Nx.sum stacked ~axes:[ 0 ]
      in
      Col.of_tensor result

  let row_mean ?(skipna = true) t ~names =
    let tensors = collect_as_float64 t names in
    if tensors = [] then
      Col.numeric Nx.float64 (Array.make (num_rows t) Float.nan)
    else
      let stacked = Nx.stack tensors ~axis:0 in
      let result =
        if skipna then
          let nan_mask = Nx.isnan stacked in
          let zeros = Nx.zeros_like stacked in
          let cleaned = Nx.where nan_mask zeros stacked in
          let ones = Nx.ones_like stacked in
          let valid_mask = Nx.where nan_mask zeros ones in
          let sum = Nx.sum cleaned ~axes:[ 0 ] in
          let count = Nx.sum valid_mask ~axes:[ 0 ] in
          let safe_count = Nx.maximum count (Nx.ones_like count) in
          let mean = Nx.div sum safe_count in
          let all_nan = Nx.equal count (Nx.zeros_like count) in
          Nx.where all_nan (Nx.full_like mean Float.nan) mean
        else Nx.mean stacked ~axes:[ 0 ]
      in
      Col.of_tensor result

  let row_min ?(skipna = true) t ~names =
    let tensors = collect_as_float64 t names in
    if tensors = [] then
      Col.numeric Nx.float64 (Array.make (num_rows t) Float.nan)
    else
      let stacked = Nx.stack tensors ~axis:0 in
      let result =
        if skipna then
          let nan_mask = Nx.isnan stacked in
          let inf = Nx.full_like stacked Float.infinity in
          let cleaned = Nx.where nan_mask inf stacked in
          let min_vals = Nx.min cleaned ~axes:[ 0 ] in
          let is_inf =
            Nx.equal min_vals (Nx.full_like min_vals Float.infinity)
          in
          Nx.where is_inf (Nx.full_like min_vals Float.nan) min_vals
        else Nx.min stacked ~axes:[ 0 ]
      in
      Col.of_tensor result

  let row_max ?(skipna = true) t ~names =
    let tensors = collect_as_float64 t names in
    if tensors = [] then
      Col.numeric Nx.float64 (Array.make (num_rows t) Float.nan)
    else
      let stacked = Nx.stack tensors ~axis:0 in
      let result =
        if skipna then
          let nan_mask = Nx.isnan stacked in
          let neg_inf = Nx.full_like stacked Float.neg_infinity in
          let cleaned = Nx.where nan_mask neg_inf stacked in
          let max_vals = Nx.max cleaned ~axes:[ 0 ] in
          let is_neg_inf =
            Nx.equal max_vals (Nx.full_like max_vals Float.neg_infinity)
          in
          Nx.where is_neg_inf (Nx.full_like max_vals Float.nan) max_vals
        else Nx.max stacked ~axes:[ 0 ]
      in
      Col.of_tensor result

  module String = struct
    let get_strings t name =
      match get_column t name with
      | Some (Col.S arr) -> arr
      | _ -> failwith ("Agg.String: column " ^ name ^ " is not a string column")

    let min t name =
      let arr = get_strings t name in
      Array.fold_left
        (fun acc x ->
          match (acc, x) with
          | None, v -> v
          | Some a, Some b -> Some (Stdlib.min a b)
          | v, None -> v)
        None arr

    let max t name =
      let arr = get_strings t name in
      Array.fold_left
        (fun acc x ->
          match (acc, x) with
          | None, v -> v
          | Some a, Some b -> Some (Stdlib.max a b)
          | v, None -> v)
        None arr

    let concat t name ?(sep = "") () =
      let arr = get_strings t name in
      let parts =
        Array.fold_left
          (fun acc x -> match x with Some s -> s :: acc | None -> acc)
          [] arr
      in
      Stdlib.String.concat sep (List.rev parts)

    let unique t name =
      let arr = get_strings t name in
      let seen = Hashtbl.create 16 in
      Array.iter
        (function Some s -> Hashtbl.replace seen s () | None -> ())
        arr;
      Hashtbl.fold (fun k () acc -> k :: acc) seen [] |> Array.of_list

    let nunique t name =
      let arr = get_strings t name in
      let seen = Hashtbl.create 16 in
      Array.iter
        (function Some s -> Hashtbl.replace seen s () | None -> ())
        arr;
      Hashtbl.length seen

    let mode t name =
      let arr = get_strings t name in
      let counts = Hashtbl.create 16 in
      Array.iter
        (function
          | Some s ->
              let c = Option.value (Hashtbl.find_opt counts s) ~default:0 in
              Hashtbl.replace counts s (c + 1)
          | None -> ())
        arr;
      Hashtbl.fold
        (fun k v acc ->
          match acc with
          | None -> Some (k, v)
          | Some (_, best) when v > best -> Some (k, v)
          | _ -> acc)
        counts None
      |> Option.map fst
  end

  module Bool = struct
    let get_bools t name =
      match get_column t name with
      | Some (Col.B arr) -> arr
      | _ -> failwith ("Agg.Bool: column " ^ name ^ " is not a boolean column")

    let all t name =
      let arr = get_bools t name in
      Array.for_all (function Some b -> b | None -> true) arr

    let any t name =
      let arr = get_bools t name in
      Array.exists (function Some true -> true | _ -> false) arr

    let sum t name =
      let arr = get_bools t name in
      Array.fold_left
        (fun acc x -> match x with Some true -> acc + 1 | _ -> acc)
        0 arr

    let mean t name =
      let arr = get_bools t name in
      let total = ref 0 in
      let count = ref 0 in
      Array.iter
        (function
          | Some b ->
              if b then incr total;
              incr count
          | None -> ())
        arr;
      if !count = 0 then Float.nan
      else float_of_int !total /. float_of_int !count
  end

  let collect_bool_arrays t names =
    List.map
      (fun name ->
        match get_column t name with
        | Some (Col.B arr) -> arr
        | _ ->
            failwith
              ("Agg.row_all/row_any: column " ^ name
             ^ " is not a boolean column"))
      names

  let row_all t ~names =
    let arrays = collect_bool_arrays t names in
    let n_rows = num_rows t in
    let result =
      Array.init n_rows (fun i ->
          let value =
            List.for_all
              (fun arr -> match arr.(i) with Some true -> true | _ -> false)
              arrays
          in
          Some value)
    in
    Col.B result

  let row_any t ~names =
    let arrays = collect_bool_arrays t names in
    let n_rows = num_rows t in
    let result =
      Array.init n_rows (fun i ->
          let value =
            List.exists
              (fun arr -> match arr.(i) with Some true -> true | _ -> false)
              arrays
          in
          Some value)
    in
    Col.B result
end

(* Joins *)

module Join_key = struct
  type t =
    | Int32 of int32
    | Int64 of int64
    | Float of float
    | String of string
    | Null

  let equal a b =
    match (a, b) with
    | Int32 x, Int32 y -> Int32.equal x y
    | Int64 x, Int64 y -> Int64.equal x y
    | Float x, Float y -> Float.equal x y
    | String x, String y -> String.equal x y
    | Null, Null -> true
    | _ -> false

  let hash = Hashtbl.hash
end

module Join_key_tbl = Hashtbl.Make (struct
  type t = Join_key.t

  let equal = Join_key.equal
  let hash = Join_key.hash
end)

let get_key_array col =
  match col with
  | Col.P (dtype, tensor, mask_opt) -> (
      let is_null i =
        match mask_opt with Some mask -> mask.(i) | None -> false
      in
      match dtype with
      | Nx.Int32 ->
          let arr : int32 array = Nx.to_array tensor in
          Array.mapi
            (fun i v -> if is_null i then Join_key.Null else Join_key.Int32 v)
            arr
      | Nx.Int64 ->
          let arr : int64 array = Nx.to_array tensor in
          Array.mapi
            (fun i v -> if is_null i then Join_key.Null else Join_key.Int64 v)
            arr
      | Nx.Float32 ->
          let arr : float array = Nx.to_array tensor in
          Array.mapi
            (fun i v -> if is_null i then Join_key.Null else Join_key.Float v)
            arr
      | Nx.Float64 ->
          let arr : float array = Nx.to_array tensor in
          Array.mapi
            (fun i v -> if is_null i then Join_key.Null else Join_key.Float v)
            arr
      | _ -> failwith "Unsupported column type for join")
  | Col.S arr ->
      Array.map
        (function Some s -> Join_key.String s | None -> Join_key.Null)
        arr
  | _ -> failwith "Unsupported column type for join"

let build_index keys =
  let tmp = Join_key_tbl.create (max 16 (Array.length keys)) in
  Array.iteri
    (fun idx key ->
      let existing =
        match Join_key_tbl.find_opt tmp key with Some lst -> lst | None -> []
      in
      Join_key_tbl.replace tmp key (idx :: existing))
    keys;
  let final_tbl = Join_key_tbl.create (Join_key_tbl.length tmp + 1) in
  Join_key_tbl.iter
    (fun key lst ->
      Join_key_tbl.add final_tbl key (Array.of_list (List.rev lst)))
    tmp;
  final_tbl

let join t1 t2 ~on ?right_on ~how ?(suffixes = ("_x", "_y")) () =
  let right_key = Option.value right_on ~default:on in
  let t2, right_key_col =
    if right_key <> on then
      let t2' = rename_column t2 ~old_name:right_key ~new_name:on in
      (t2', on)
    else (t2, on)
  in
  let left_col = get_column_exn t1 on in
  let right_col = get_column_exn t2 right_key_col in

  let left_keys = get_key_array left_col in
  let right_keys = get_key_array right_col in
  let right_index = build_index right_keys in

  let left_indices = ref [] in
  let right_indices = ref [] in
  let append_pair l r =
    left_indices := l :: !left_indices;
    right_indices := r :: !right_indices
  in
  let matched_right = Array.make (Array.length right_keys) false in

  (match how with
  | `Inner ->
      for i = 0 to Array.length left_keys - 1 do
        match Join_key_tbl.find_opt right_index left_keys.(i) with
        | Some matches ->
            Array.iter
              (fun j ->
                append_pair i j;
                matched_right.(j) <- true)
              matches
        | None -> ()
      done
  | `Left ->
      for i = 0 to Array.length left_keys - 1 do
        match Join_key_tbl.find_opt right_index left_keys.(i) with
        | Some matches ->
            Array.iter
              (fun j ->
                append_pair i j;
                matched_right.(j) <- true)
              matches
        | None -> append_pair i (-1)
      done
  | `Right ->
      let left_index = build_index left_keys in
      for j = 0 to Array.length right_keys - 1 do
        match Join_key_tbl.find_opt left_index right_keys.(j) with
        | Some matches -> Array.iter (fun i -> append_pair i j) matches
        | None -> append_pair (-1) j
      done
  | `Outer ->
      for i = 0 to Array.length left_keys - 1 do
        match Join_key_tbl.find_opt right_index left_keys.(i) with
        | Some matches ->
            Array.iter
              (fun j ->
                append_pair i j;
                matched_right.(j) <- true)
              matches
        | None -> append_pair i (-1)
      done;
      for j = 0 to Array.length right_keys - 1 do
        if not matched_right.(j) then append_pair (-1) j
      done);

  let left_idx = Array.of_list (List.rev !left_indices) in
  let right_idx = Array.of_list (List.rev !right_indices) in

  let result_cols = ref [] in
  let left_suffix, right_suffix = suffixes in

  List.iter
    (fun name ->
      let col = get_column_exn t1 name in
      let new_col = Col.reindex_nullable col left_idx (num_rows t1) in
      let final_name =
        if name <> on && has_column t2 name then name ^ left_suffix else name
      in
      result_cols := (final_name, new_col) :: !result_cols)
    (column_names t1);

  List.iter
    (fun name ->
      if name <> on then
        let col = get_column_exn t2 name in
        let new_col = Col.reindex_nullable col right_idx (num_rows t2) in
        let final_name =
          if has_column t1 name then name ^ right_suffix else name
        in
        result_cols := (final_name, new_col) :: !result_cols)
    (column_names t2);

  create (List.rev !result_cols)

(* Pivot and reshape *)

let pivot t ~index ~columns ~values ?(agg_func = `Sum) () =
  let col_col = get_column_exn t columns in
  let col_fmt = Col.to_string_fn col_col in
  let idx_col = get_column_exn t index in
  let idx_fmt = Col.to_string_fn idx_col in
  let n = num_rows t in
  let unique_cols =
    let seen = Hashtbl.create 16 in
    let result = ref [] in
    for i = 0 to n - 1 do
      let s = col_fmt i in
      if not (Hashtbl.mem seen s) then (
        Hashtbl.add seen s ();
        result := s :: !result)
    done;
    List.rev !result
  in
  let unique_indices =
    let seen = Hashtbl.create 16 in
    let result = ref [] in
    for i = 0 to n - 1 do
      let s = idx_fmt i in
      if not (Hashtbl.mem seen s) then (
        Hashtbl.add seen s ();
        result := s :: !result)
    done;
    List.rev !result
  in
  let groups = Hashtbl.create 16 in
  let val_arr, _ = Col.col_as_float_array (get_column_exn t values) in
  for i = 0 to n - 1 do
    let idx_key = idx_fmt i in
    let col_key = col_fmt i in
    let key = (idx_key, col_key) in
    let current = try Hashtbl.find groups key with Not_found -> [] in
    Hashtbl.replace groups key (val_arr.(i) :: current)
  done;
  let aggregate values =
    match agg_func with
    | `Sum -> List.fold_left ( +. ) 0. values
    | `Mean ->
        let s = List.fold_left ( +. ) 0. values in
        s /. float_of_int (List.length values)
    | `Count -> float_of_int (List.length values)
    | `Min -> List.fold_left min Float.infinity values
    | `Max -> List.fold_left max Float.neg_infinity values
  in
  let result_cols =
    ref [ (index, Col.string (Array.of_list unique_indices)) ]
  in
  List.iter
    (fun col_name ->
      let col_values =
        List.map
          (fun idx ->
            try
              let values = Hashtbl.find groups (idx, col_name) in
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
  let value_columns =
    if value_vars = [] then
      List.filter (fun name -> not (List.mem name id_vars)) (column_names t)
    else value_vars
  in
  let n_rows = num_rows t in
  let n_value_cols = List.length value_columns in
  let total_rows = n_rows * n_value_cols in
  let result_cols = ref [] in
  List.iter
    (fun id_name ->
      let col = get_column_exn t id_name in
      let new_col =
        Col.reindex col (Array.init total_rows (fun i -> i / n_value_cols))
      in
      result_cols := (id_name, new_col) :: !result_cols)
    id_vars;
  let value_columns_arr = Array.of_list value_columns in
  let var_col_values =
    Array.init total_rows (fun i -> Some value_columns_arr.(i mod n_value_cols))
  in
  result_cols := (var_name, Col.S var_col_values) :: !result_cols;
  let value_arrays =
    Array.of_list
      (List.map
         (fun col_name ->
           let arr, _ = Col.col_as_float_array (get_column_exn t col_name) in
           arr)
         value_columns)
  in
  let value_col_data =
    Array.init total_rows (fun i ->
        let row = i / n_value_cols in
        let c = i mod n_value_cols in
        value_arrays.(c).(row))
  in
  result_cols := (value_name, Col.float64 value_col_data) :: !result_cols;
  create (List.rev !result_cols)

(* Conversion *)

let to_nx t =
  let numeric_tensors =
    List.filter_map
      (fun (_name, col) ->
        match col with
        | Col.P (_, tensor, _) -> Some (Nx.astype Nx.float32 tensor)
        | _ -> None)
      t.columns
  in
  if numeric_tensors = [] then invalid_arg "to_nx: no numeric columns"
  else Nx.stack numeric_tensors ~axis:1

(* Display *)

let pp ?(max_rows = 10) ?(max_cols = 10) ppf t =
  let n_rows = num_rows t in
  let n_cols = num_columns t in
  let rows_to_show = min max_rows n_rows in
  let cols_to_show = min max_cols n_cols in
  let names = column_names t in
  let names_to_show = list_take cols_to_show names in
  let fmts =
    List.map
      (fun name ->
        match get_column t name with
        | Some col -> Col.to_string_fn col
        | None -> fun _ -> "")
      names_to_show
  in
  Format.fprintf ppf "Shape: (%d, %d)@\n" n_rows n_cols;
  Format.fprintf ppf "%s@\n" (String.concat "\t" names_to_show);
  for i = 0 to rows_to_show - 1 do
    Format.fprintf ppf "%s@\n"
      (String.concat "\t" (List.map (fun f -> f i) fmts))
  done

let to_string ?max_rows ?max_cols t =
  Format.asprintf "%a" (pp ?max_rows ?max_cols) t

let print ?max_rows ?max_cols t = print_string (to_string ?max_rows ?max_cols t)

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
          Col.string
            (Array.of_list
               (List.map
                  (fun col_name ->
                    match stat with
                    | "count" -> string_of_int (Agg.count t col_name)
                    | "mean" -> string_of_float (Agg.mean t col_name)
                    | "std" -> string_of_float (Agg.std t col_name)
                    | "min" -> (
                        match Agg.min t col_name with
                        | Some v -> string_of_float v
                        | None -> "NaN")
                    | "25%" -> string_of_float (Agg.quantile t col_name ~q:0.25)
                    | "50%" -> string_of_float (Agg.median t col_name)
                    | "75%" -> string_of_float (Agg.quantile t col_name ~q:0.75)
                    | "max" -> (
                        match Agg.max t col_name with
                        | Some v -> string_of_float v
                        | None -> "NaN")
                    | _ -> "")
                  numeric_cols)) ))
      stats
  in
  create data

let pp_info ppf t =
  let n_rows, n_cols = shape t in
  Format.fprintf ppf "DataFrame info:@\n";
  Format.fprintf ppf "  Rows: %d@\n" n_rows;
  Format.fprintf ppf "  Columns: %d@\n" n_cols;
  Format.fprintf ppf "@\nColumn types:@\n";
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
      Format.fprintf ppf "  %s: %s@\n" name typ_str)
    (column_types t)

let info t = pp_info Format.std_formatter t
