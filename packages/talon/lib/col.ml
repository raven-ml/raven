(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | P : ('a, 'b) Nx.dtype * ('a, 'b) Nx.t * bool array option -> t
  | S : string option array -> t
  | B : bool option array -> t

(* Internal helpers *)

let normalize_mask = function
  | Some mask when Array.exists Fun.id mask -> Some (Array.copy mask)
  | _ -> None

let count_none arr =
  Array.fold_left (fun acc x -> if Option.is_none x then acc + 1 else acc) 0 arr

let fill_options arr varr =
  let result = Array.copy arr in
  Array.iteri (fun i x -> if Option.is_none x then result.(i) <- varr.(0)) arr;
  result

let reindex_nullable_options arr indices len =
  Array.init len (fun i ->
      let idx = indices.(i) in
      if idx >= 0 && idx < Array.length arr then arr.(idx) else None)

(* Constructors *)

let numeric_default : type a b. (a, b) Nx.dtype -> a = function
  | Nx.Float16 -> Float.nan
  | Nx.Float32 -> Float.nan
  | Nx.Float64 -> Float.nan
  | Nx.BFloat16 -> Float.nan
  | Nx.Float8_e4m3 -> Float.nan
  | Nx.Float8_e5m2 -> Float.nan
  | Nx.Int4 -> 0
  | Nx.UInt4 -> 0
  | Nx.Int8 -> 0
  | Nx.UInt8 -> 0
  | Nx.Int16 -> 0
  | Nx.UInt16 -> 0
  | Nx.Int32 -> Int32.min_int
  | Nx.UInt32 -> Int32.min_int
  | Nx.Int64 -> Int64.min_int
  | Nx.UInt64 -> Int64.min_int
  | Nx.Complex64 -> Complex.zero
  | Nx.Complex128 -> Complex.zero
  | Nx.Bool -> false

let numeric (type a b) (dtype : (a, b) Nx.dtype) (arr : a array) =
  let tensor = Nx.create dtype [| Array.length arr |] arr in
  P (dtype, tensor, None)

let numeric_opt (type a b) (dtype : (a, b) Nx.dtype) (arr : a option array) =
  let default = numeric_default dtype in
  let data = Array.map (fun x -> Option.value x ~default) arr in
  let mask = Array.map Option.is_none arr in
  let tensor = Nx.create dtype [| Array.length data |] data in
  P (dtype, tensor, normalize_mask (Some mask))

let string arr = S (Array.map (fun x -> Some x) arr)
let string_opt arr = S arr
let bool arr = B (Array.map (fun x -> Some x) arr)
let bool_opt arr = B arr
let float32 arr = numeric Nx.float32 arr
let float64 arr = numeric Nx.float64 arr
let int32 arr = numeric Nx.int32 arr
let int64 arr = numeric Nx.int64 arr
let float32_opt arr = numeric_opt Nx.float32 arr
let float64_opt arr = numeric_opt Nx.float64 arr
let int32_opt arr = numeric_opt Nx.int32 arr
let int64_opt arr = numeric_opt Nx.int64 arr

let of_tensor (type a b) (t : (a, b) Nx.t) =
  match Nx.shape t with
  | [| _ |] -> P (Nx.dtype t, t, None)
  | _ -> invalid_arg "of_tensor: tensor must be 1D"

(* Properties *)

let length = function
  | P (_, t, _) -> Nx.size t
  | S arr -> Array.length arr
  | B arr -> Array.length arr

let has_nulls = function
  | P (_, _, Some mask) -> Array.exists Fun.id mask
  | P _ -> false
  | S arr -> Array.exists Option.is_none arr
  | B arr -> Array.exists Option.is_none arr

let null_count = function
  | P (_, _, Some mask) ->
      Array.fold_left (fun acc b -> if b then acc + 1 else acc) 0 mask
  | P _ -> 0
  | S arr -> count_none arr
  | B arr -> count_none arr

let null_mask = function P (_, _, mask) -> mask | _ -> None

let dtype = function
  | P (Nx.Float32, _, _) -> `Float32
  | P (Nx.Float64, _, _) -> `Float64
  | P (Nx.Int32, _, _) -> `Int32
  | P (Nx.Int64, _, _) -> `Int64
  | S _ -> `String
  | B _ -> `Bool
  | P _ -> `Other

let is_null_at col i =
  match col with
  | P (_, _, Some mask) -> mask.(i)
  | P _ -> false
  | S arr -> Option.is_none arr.(i)
  | B arr -> Option.is_none arr.(i)

(* Generic dtype helpers *)

let element_to_string (type a b) (dtype : (a, b) Nx.dtype) : a -> string =
  match dtype with
  | Nx.Float32 -> string_of_float
  | Nx.Float64 -> string_of_float
  | Nx.Float16 -> string_of_float
  | Nx.BFloat16 -> string_of_float
  | Nx.Float8_e4m3 -> string_of_float
  | Nx.Float8_e5m2 -> string_of_float
  | Nx.Int32 -> Int32.to_string
  | Nx.UInt32 -> Int32.to_string
  | Nx.Int64 -> Int64.to_string
  | Nx.UInt64 -> Int64.to_string
  | Nx.Int4 -> string_of_int
  | Nx.UInt4 -> string_of_int
  | Nx.Int8 -> string_of_int
  | Nx.UInt8 -> string_of_int
  | Nx.Int16 -> string_of_int
  | Nx.UInt16 -> string_of_int
  | Nx.Complex64 -> fun c -> Printf.sprintf "%g+%gi" c.Complex.re c.Complex.im
  | Nx.Complex128 -> fun c -> Printf.sprintf "%g+%gi" c.Complex.re c.Complex.im
  | Nx.Bool -> string_of_bool

let element_to_float (type a b) (dtype : (a, b) Nx.dtype) : a -> float =
  match dtype with
  | Nx.Float32 -> Fun.id
  | Nx.Float64 -> Fun.id
  | Nx.Float16 -> Fun.id
  | Nx.BFloat16 -> Fun.id
  | Nx.Float8_e4m3 -> Fun.id
  | Nx.Float8_e5m2 -> Fun.id
  | Nx.Int32 -> Int32.to_float
  | Nx.UInt32 -> Int32.to_float
  | Nx.Int64 -> Int64.to_float
  | Nx.UInt64 -> Int64.to_float
  | Nx.Int4 -> float_of_int
  | Nx.UInt4 -> float_of_int
  | Nx.Int8 -> float_of_int
  | Nx.UInt8 -> float_of_int
  | Nx.Int16 -> float_of_int
  | Nx.UInt16 -> float_of_int
  | Nx.Complex64 -> failwith "element_to_float: complex not supported"
  | Nx.Complex128 -> failwith "element_to_float: complex not supported"
  | Nx.Bool -> failwith "element_to_float: bool not supported"

(* Null handling *)

let drop_nulls col =
  match col with
  | P (dtype, tensor, Some mask) ->
      let arr = Nx.to_array tensor in
      let n = Array.length arr in
      let count = ref 0 in
      for i = 0 to n - 1 do
        if not mask.(i) then incr count
      done;
      let result = Array.make !count arr.(0) in
      let j = ref 0 in
      for i = 0 to n - 1 do
        if not mask.(i) then (
          result.(!j) <- arr.(i);
          incr j)
      done;
      P (dtype, Nx.create dtype [| !count |] result, None)
  | P (_, _, None) -> col
  | S arr ->
      let filtered = Array.to_list arr |> List.filter_map Fun.id in
      string (Array.of_list filtered)
  | B arr ->
      let filtered = Array.to_list arr |> List.filter_map Fun.id in
      bool (Array.of_list filtered)

let fill_nulls_p (type a b) (dtype : (a, b) Nx.dtype) tensor mask_opt
    (varr : a array) =
  match mask_opt with
  | None -> P (dtype, tensor, None)
  | Some mask ->
      let arr : a array = Nx.to_array tensor in
      let result = Array.copy arr in
      let new_mask = Array.copy mask in
      Array.iteri
        (fun i is_null ->
          if is_null then (
            result.(i) <- varr.(0);
            new_mask.(i) <- false))
        mask;
      P
        ( dtype,
          Nx.create dtype [| Array.length result |] result,
          normalize_mask (Some new_mask) )

let fill_nulls col ~value =
  match (col, value) with
  | P (dtype, t, m), P (vdtype, vt, _) -> (
      match Nx_core.Dtype.equal_witness dtype vdtype with
      | Some Type.Equal -> fill_nulls_p dtype t m (Nx.to_array vt)
      | None ->
          invalid_arg "Col.fill_nulls: value type doesn't match column type")
  | S arr, S varr -> S (fill_options arr varr)
  | B arr, B varr -> B (fill_options arr varr)
  | _ -> invalid_arg "Col.fill_nulls: value type doesn't match column type"

(* Extraction *)

let to_tensor (type a b) (dtype : (a, b) Nx.dtype) col =
  match col with
  | P (col_dtype, tensor, _) -> (
      match Nx_core.Dtype.equal_witness dtype col_dtype with
      | Some Type.Equal -> Some (tensor : (a, b) Nx.t)
      | None -> None)
  | _ -> None

let to_string_array = function S arr -> Some arr | _ -> None
let to_bool_array = function B arr -> Some arr | _ -> None

(* Internal: extract any numeric column as a float array, filtering by mask *)

let col_as_float_array col =
  match col with
  | P (dtype, tensor, mask) -> (
      match dtype with
      | Nx.Complex64 -> failwith "col_as_float_array: complex not supported"
      | Nx.Complex128 -> failwith "col_as_float_array: complex not supported"
      | Nx.Bool -> failwith "col_as_float_array: bool not supported"
      | _ -> (
          let arr : float array = Nx.to_array (Nx.cast Nx.float64 tensor) in
          match mask with
          | Some m ->
              let collected = ref [] in
              let count = ref 0 in
              for i = Array.length arr - 1 downto 0 do
                if not m.(i) then (
                  collected := arr.(i) :: !collected;
                  incr count)
              done;
              (Array.of_list !collected, !count)
          | None -> (arr, Array.length arr)))
  | _ -> failwith "col_as_float_array: column must be numeric"

(* Display: returns a closure that formats the value at index i as a string. The
   underlying array is extracted once so repeated calls are O(1). *)

let to_string_fn ?(null = "<null>") col =
  match col with
  | P (dtype, tensor, mask) ->
      let is_null =
        match mask with Some m -> fun i -> m.(i) | None -> fun _ -> false
      in
      let to_s = element_to_string dtype in
      let arr = Nx.to_array tensor in
      fun i -> if is_null i then null else to_s arr.(i)
  | S arr -> ( fun i -> match arr.(i) with Some s -> s | None -> null)
  | B arr -> (
      fun i -> match arr.(i) with Some b -> string_of_bool b | None -> null)

(* Internal: reindex a column by an array of non-negative indices *)

let reindex col indices =
  match col with
  | P (dtype, tensor, mask_opt) ->
      let n = Array.length indices in
      if n = 0 then P (dtype, Nx.empty dtype [| 0 |], None)
      else
        let idx_tensor =
          Nx.create Nx.int32 [| n |] (Array.map Int32.of_int indices)
        in
        let gathered = Nx.take ~axis:0 idx_tensor tensor in
        let mask =
          match mask_opt with
          | Some m ->
              let sub = Array.map (fun i -> m.(i)) indices in
              if Array.exists Fun.id sub then Some sub else None
          | None -> None
        in
        P (dtype, gathered, mask)
  | S arr -> S (Array.map (fun i -> arr.(i)) indices)
  | B arr -> B (Array.map (fun i -> arr.(i)) indices)

(* Internal: reindex with nullable indices (-1 means null) *)

let reindex_nullable col indices n_source =
  let has_null = Array.exists (fun idx -> idx < 0) indices in
  if not has_null then reindex col indices
  else
    let len = Array.length indices in
    match col with
    | P (dtype, tensor, mask_opt) ->
        let source = Nx.to_array tensor in
        let result = Array.copy source in
        let result =
          if len = Array.length result then result
          else Array.make len (if n_source > 0 then source.(0) else result.(0))
        in
        let mask =
          Array.init len (fun i ->
              let idx = indices.(i) in
              if idx < 0 || idx >= n_source then true
              else
                let is_null =
                  match mask_opt with Some m -> m.(idx) | None -> false
                in
                if not is_null then result.(i) <- source.(idx);
                is_null)
        in
        let mask_opt = if Array.exists Fun.id mask then Some mask else None in
        P (dtype, Nx.create dtype [| len |] result, mask_opt)
    | S arr -> S (reindex_nullable_options arr indices len)
    | B arr -> B (reindex_nullable_options arr indices len)

(* Internal: slice a column from start to start+length *)

let slice_col col start length =
  match col with
  | P (dtype, tensor, mask_opt) ->
      let sliced = Nx.slice [ Nx.R (start, start + length) ] tensor in
      let mask =
        match mask_opt with
        | Some m ->
            let sub = Array.sub m start length in
            if Array.exists Fun.id sub then Some sub else None
        | None -> None
      in
      P (dtype, sliced, mask)
  | S arr -> S (Array.sub arr start length)
  | B arr -> B (Array.sub arr start length)

(* Internal: concatenate columns of the same type *)

let combine_masks arrays_masks =
  if List.exists (fun (_, m) -> Option.is_some m) arrays_masks then
    let mask_arrays =
      List.map
        (fun (arr, mask_opt) ->
          match mask_opt with
          | Some m -> Array.copy m
          | None -> Array.make (Array.length arr) false)
        arrays_masks
    in
    let concatenated = Array.concat mask_arrays in
    if Array.exists Fun.id concatenated then Some concatenated else None
  else None

let concat_p (type a b) (dtype : (a, b) Nx.dtype) cols =
  let arrays_masks =
    List.map
      (function
        | P (_, t, mask) ->
            let arr : a array = Nx.to_array (Nx.cast dtype t) in
            (arr, mask)
        | _ -> failwith "concat: column type mismatch")
      cols
  in
  let arrays = List.map fst arrays_masks in
  let all_data : a array = Array.concat arrays in
  let combined_mask = combine_masks arrays_masks in
  P (dtype, Nx.create dtype [| Array.length all_data |] all_data, combined_mask)

let concat_cols cols =
  match cols with
  | [] -> invalid_arg "concat_cols: empty list"
  | first :: _ -> (
      match first with
      | P (dtype, _, _) -> concat_p dtype cols
      | S _ ->
          let arrays =
            List.map
              (function S arr -> arr | _ -> failwith "concat: type mismatch")
              cols
          in
          S (Array.concat arrays)
      | B _ ->
          let arrays =
            List.map
              (function B arr -> arr | _ -> failwith "concat: type mismatch")
              cols
          in
          B (Array.concat arrays))

(* Column transforms *)

let via_float64 f col =
  match col with
  | P (dtype, tensor, _) ->
      let arr = Nx.to_array (Nx.cast Nx.float64 tensor) in
      let result = f arr in
      let result_tensor =
        Nx.create Nx.float64 [| Array.length result |] result
      in
      P (dtype, Nx.cast dtype result_tensor, None)
  | _ -> failwith "column must be numeric"

let cumsum col =
  via_float64
    (fun arr ->
      let result = Array.copy arr in
      for i = 1 to Array.length result - 1 do
        result.(i) <- result.(i - 1) +. result.(i)
      done;
      result)
    col

let cumprod col =
  via_float64
    (fun arr ->
      let result = Array.copy arr in
      for i = 1 to Array.length result - 1 do
        result.(i) <- result.(i - 1) *. result.(i)
      done;
      result)
    col

let diff ?(periods = 1) col =
  via_float64
    (fun arr ->
      let n = Array.length arr in
      let result = Array.make n 0. in
      for i = periods to n - 1 do
        result.(i) <- arr.(i) -. arr.(i - periods)
      done;
      result)
    col

let pct_change ?(periods = 1) col =
  match col with
  | P (_, tensor, _) ->
      let arr = Nx.to_array (Nx.cast Nx.float64 tensor) in
      let n = Array.length arr in
      let result = Array.make n Float.nan in
      for i = periods to n - 1 do
        let prev = arr.(i - periods) in
        let curr = arr.(i) in
        result.(i) <- (if prev = 0. then Float.nan else (curr -. prev) /. prev)
      done;
      float64 result
  | _ -> failwith "pct_change: column must be numeric"

let shift_option_array ~periods arr =
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
  result

let shift ~periods col =
  match col with
  | P (dtype, tensor, _) ->
      let n = (Nx.shape tensor).(0) in
      if periods = 0 then col
      else
        let abs_p = abs periods in
        if abs_p >= n then
          P (dtype, Nx.zeros dtype [| n |], Some (Array.make n true))
        else
          let data, pad =
            if periods > 0 then
              ( Nx.slice [ Nx.R (0, n - abs_p) ] tensor,
                Nx.zeros dtype [| abs_p |] )
            else
              (Nx.slice [ Nx.R (abs_p, n) ] tensor, Nx.zeros dtype [| abs_p |])
          in
          let result =
            if periods > 0 then Nx.concatenate ~axis:0 [ pad; data ]
            else Nx.concatenate ~axis:0 [ data; pad ]
          in
          let mask =
            Array.init n (fun i ->
                if periods > 0 then i < abs_p else i >= n - abs_p)
          in
          P (dtype, result, Some mask)
  | S arr -> S (shift_option_array ~periods arr)
  | B arr -> B (shift_option_array ~periods arr)

(* Formatting *)

let pp ppf col =
  let len = length col in
  let to_s = to_string_fn col in
  let dtype_str =
    match dtype col with
    | `Float32 -> "float32"
    | `Float64 -> "float64"
    | `Int32 -> "int32"
    | `Int64 -> "int64"
    | `String -> "string"
    | `Bool -> "bool"
    | `Other -> "other"
  in
  Format.fprintf ppf "@[<hov 2>Col(%s, %d)[" dtype_str len;
  let show = min 5 len in
  for i = 0 to show - 1 do
    if i > 0 then Format.fprintf ppf ",@ ";
    Format.fprintf ppf "%s" (to_s i)
  done;
  if len > show then Format.fprintf ppf ",@ ...";
  Format.fprintf ppf "]@]"
