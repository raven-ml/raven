(* safetensors.ml *)

(* Errors *)

type safetensor_error =
  | Invalid_header of string (* invalid UTF-8 *)
  | Invalid_header_start (* reserved for future parity *)
  | Invalid_header_deserialization of string (* JSON parse error *)
  | Header_too_large
  | Header_too_small
  | Invalid_header_length
  | Tensor_not_found of string
  | Tensor_invalid_info
  | Invalid_offset of string
  | Io_error of string (* exposed for serialize_to_file *)
  | Json_error of string
  | Invalid_tensor_view of string * int list * int
  | Metadata_incomplete_buffer
  | Validation_overflow
  | Misaligned_slice

let string_of_error = function
  | Invalid_header e -> Printf.sprintf "invalid UTF-8 in header: %s" e
  | Invalid_header_start -> "invalid start character in header, must be `{`"
  | Invalid_header_deserialization e -> "invalid JSON in header: " ^ e
  | Header_too_large -> "header too large"
  | Header_too_small -> "header too small"
  | Invalid_header_length -> "invalid header length"
  | Tensor_not_found n -> Printf.sprintf "tensor `%s` not found" n
  | Tensor_invalid_info -> "invalid shape, data type, or offset for tensor"
  | Invalid_offset n -> Printf.sprintf "invalid offset for tensor `%s`" n
  | Io_error e -> "I/O error: " ^ e
  | Json_error e -> "JSON error: " ^ e
  | Invalid_tensor_view (dt, shape, n_bytes) ->
      let dims =
        match shape with
        | [] -> ""
        | d :: tl ->
            List.fold_left
              (fun acc x -> acc ^ ", " ^ string_of_int x)
              (string_of_int d) tl
      in
      Printf.sprintf
        "tensor of type %s and shape (%s) can't be created from %d bytes" dt
        dims n_bytes
  | Metadata_incomplete_buffer -> "incomplete metadata, file not fully covered"
  | Validation_overflow ->
      "overflow computing buffer size from shape and/or element type"
  | Misaligned_slice ->
      "The slice is slicing for subbytes dtypes, and the slice does not end up \
       at a byte boundary, this is invalid."

let ( let* ) x f = match x with Ok v -> f v | Error _ as e -> e

(* UTF-8 validation (minimal; enough to catch non-UTF8 headers) *)

let is_valid_utf8 (s : string) : bool =
  let n = String.length s in
  let i = ref 0 in
  let ( < ) = ( < ) in
  let ok = ref true in
  while !ok && !i < n do
    let c = Char.code s.[!i] in
    if c land 0x80 = 0 then incr i
    else if c land 0xE0 = 0xC0 && !i + 1 < n then
      let c1 = Char.code s.[!i + 1] in
      if c1 land 0xC0 <> 0x80 || c < 0xC2 then ok := false else i := !i + 2
    else if c land 0xF0 = 0xE0 && !i + 2 < n then
      let c1 = Char.code s.[!i + 1] in
      let c2 = Char.code s.[!i + 2] in
      if c1 land 0xC0 <> 0x80 || c2 land 0xC0 <> 0x80 then ok := false
      else if c = 0xE0 && c1 < 0xA0 then ok := false
      else if c = 0xED && c1 >= 0xA0 then ok := false
      else i := !i + 3
    else if c land 0xF8 = 0xF0 && !i + 3 < n then
      let c1 = Char.code s.[!i + 1] in
      let c2 = Char.code s.[!i + 2] in
      let c3 = Char.code s.[!i + 3] in
      if c1 land 0xC0 <> 0x80 || c2 land 0xC0 <> 0x80 || c3 land 0xC0 <> 0x80
      then ok := false
      else if c = 0xF0 && c1 < 0x90 then ok := false
      else if c = 0xF4 && c1 >= 0x90 then ok := false
      else if c > 0xF4 then ok := false
      else i := !i + 4
    else ok := false
  done;
  !ok

(* Dtype *)

type dtype =
  | BOOL
  | F4
  | F6_E2M3
  | F6_E3M2
  | U8
  | I8
  | F8_E5M2
  | F8_E4M3
  | F8_E8M0
  | I16
  | U16
  | F16
  | BF16
  | I32
  | U32
  | F32
  | F64
  | I64
  | U64

let dtype_to_string = function
  | F4 -> "F4"
  | F6_E2M3 -> "F6_E2M3"
  | F6_E3M2 -> "F6_E3M2"
  | BOOL -> "BOOL"
  | I8 -> "I8"
  | U8 -> "U8"
  | F8_E5M2 -> "F8_E5M2"
  | F8_E4M3 -> "F8_E4M3"
  | F8_E8M0 -> "F8_E8M0"
  | I16 -> "I16"
  | U16 -> "U16"
  | I32 -> "I32"
  | U32 -> "U32"
  | I64 -> "I64"
  | U64 -> "U64"
  | F16 -> "F16"
  | BF16 -> "BF16"
  | F32 -> "F32"
  | F64 -> "F64"

let dtype_of_string = function
  | "F4" -> Some F4
  | "F6_E2M3" -> Some F6_E2M3
  | "F6_E3M2" -> Some F6_E3M2
  | "BOOL" -> Some BOOL
  | "I8" -> Some I8
  | "U8" -> Some U8
  | "F8_E5M2" -> Some F8_E5M2
  | "F8_E4M3" -> Some F8_E4M3
  | "F8_E8M0" -> Some F8_E8M0
  | "I16" -> Some I16
  | "U16" -> Some U16
  | "I32" -> Some I32
  | "U32" -> Some U32
  | "I64" -> Some I64
  | "U64" -> Some U64
  | "F16" -> Some F16
  | "BF16" -> Some BF16
  | "F32" -> Some F32
  | "F64" -> Some F64
  | _ -> None

let bitsize = function
  | F4 -> 4
  | F6_E3M2 -> 6
  | F6_E2M3 -> 6
  | BOOL | U8 | I8 | F8_E5M2 | F8_E4M3 | F8_E8M0 -> 8
  | I16 | U16 | F16 | BF16 -> 16
  | I32 | U32 | F32 -> 32
  | I64 | U64 | F64 -> 64

(* order like Rust enum derive Ord (increasing alignment) *)
let dtype_rank = function
  | BOOL -> 0
  | F4 -> 1
  | F6_E2M3 -> 2
  | F6_E3M2 -> 3
  | U8 -> 4
  | I8 -> 5
  | F8_E5M2 -> 6
  | F8_E4M3 -> 7
  | F8_E8M0 -> 8
  | I16 -> 9
  | U16 -> 10
  | F16 -> 11
  | BF16 -> 12
  | I32 -> 13
  | U32 -> 14
  | F32 -> 15
  | F64 -> 16
  | I64 -> 17
  | U64 -> 18

(* Tensor model *)

type tensor_info = { dtype : dtype; shape : int list; data_offsets : int * int }

type metadata = {
  metadata_kv : (string * string) list option;
  tensors : tensor_info array;
  index_map : (string, int) Hashtbl.t;
}

let max_header_size = 100_000_000
let n_len = 8

let next_multiple_of x k =
  if k <= 0 then x else if x mod k = 0 then x else x + (k - (x mod k))

let int64_mul_checked a b =
  if a = 0L || b = 0L then Ok 0L
  else
    let open Int64 in
    if a > div max_int b then Error () else Ok (mul a b)

exception ValidateError of safetensor_error

let validate (m : metadata) : (int, safetensor_error) result =
  let start = ref 0 in
  let buffer_end = ref 0 in
  let open Int64 in
  try
    Array.iteri
      (fun i info ->
        let s, e = info.data_offsets in
        (if s <> !start || e < s then
           let name =
             let v = ref "no_tensor" in
             Hashtbl.iter (fun k idx -> if idx = i then v := k) m.index_map;
             !v
           in
           raise (ValidateError (Invalid_offset name)));
        start := e;
        (* compute nelements * bitsize *)
        let ne =
          List.fold_left
            (fun acc d ->
              if d < 0 then raise (ValidateError Validation_overflow);
              match int64_mul_checked acc (of_int d) with
              | Ok v -> v
              | Error _ -> raise (ValidateError Validation_overflow))
            1L info.shape
        in
        let bs = of_int (bitsize info.dtype) in
        let nbits =
          match int64_mul_checked ne bs with
          | Ok v -> v
          | Error _ -> raise (ValidateError Validation_overflow)
        in
        if rem nbits 8L <> 0L then raise (ValidateError Misaligned_slice);
        let size = to_int (div nbits 8L) in
        if e - s <> size then raise (ValidateError Tensor_invalid_info);
        buffer_end := e)
      m.tensors;
    Ok !buffer_end
  with ValidateError e -> Error e

(* JSON (de)serialization of metadata header *)

let metadata_to_json (m : metadata) : string =
  (* Arrange names by tensor order (offset-ordered) *)
  let names = Array.make (Array.length m.tensors) "" in
  Hashtbl.iter (fun name idx -> names.(idx) <- name) m.index_map;
  let base =
    Array.to_list
      (Array.mapi
         (fun i ti ->
           let name = names.(i) in
           let shape = `List (List.map (fun d -> `Int d) ti.shape) in
           let s, e = ti.data_offsets in
           let offs = `List [ `Int s; `Int e ] in
           ( name,
             `Assoc
               [
                 ("dtype", `String (dtype_to_string ti.dtype));
                 ("shape", shape);
                 ("data_offsets", offs);
               ] ))
         m.tensors)
  in
  let kv =
    match m.metadata_kv with
    | None -> base
    | Some md ->
        let md_obj = `Assoc (List.map (fun (k, v) -> (k, `String v)) md) in
        ("__metadata__", md_obj) :: base
  in
  Json_minimal.to_string (`Assoc kv)

type hash_metadata = {
  hm_meta : (string * string) list option;
  hm_tensors : (string * tensor_info) list;
}

let json_to_string_map json =
  try
    let obj = Json_minimal.to_assoc json in
    Ok (List.map (fun (k, v) -> (k, Json_minimal.to_string_exn v)) obj)
  with _ -> Error "metadata values must be strings"

let parse_tensor_info (name, j) : (string * tensor_info, string) result =
  try
    let dt_str =
      j |> Json_minimal.member "dtype" |> Json_minimal.to_string_exn
    in
    let dt =
      match dtype_of_string dt_str with
      | Some d -> d
      | None -> raise (Failure "bad dtype")
    in
    let shape =
      j
      |> Json_minimal.member "shape"
      |> Json_minimal.to_list_exn
      |> List.map Json_minimal.to_int_exn
    in
    let offs =
      j
      |> Json_minimal.member "data_offsets"
      |> Json_minimal.to_list_exn
      |> List.map Json_minimal.to_int_exn
    in
    let s, e =
      match offs with [ s; e ] -> (s, e) | _ -> raise (Failure "bad offsets")
    in
    Ok (name, { dtype = dt; shape; data_offsets = (s, e) })
  with e -> Error (Printexc.to_string e)

let json_to_hashmetadata j : (hash_metadata, string) result =
  try
    let obj = Json_minimal.to_assoc j in
    let md =
      match List.assoc_opt "__metadata__" obj with
      | None -> Ok None
      | Some jmd ->
          let* md = json_to_string_map jmd in
          Ok (Some md)
    in
    let kv_no_md = List.filter (fun (k, _) -> k <> "__metadata__") obj in
    let rec tensors acc = function
      | [] -> Ok (List.rev acc)
      | (name, jv) :: tl ->
          let* e = parse_tensor_info (name, jv) in
          tensors (e :: acc) tl
    in
    let* hm = md in
    let* ts = tensors [] kv_no_md in
    Ok { hm_meta = hm; hm_tensors = ts }
  with e -> Error (Printexc.to_string e)

let metadata_of_hash (h : hash_metadata) : (metadata, safetensor_error) result =
  (* sort tensors by data_offsets ascending (s, then e) *)
  let ts =
    List.sort
      (fun (_n1, t1) (_n2, t2) -> compare t1.data_offsets t2.data_offsets)
      h.hm_tensors
  in
  let index_map = Hashtbl.create (List.length ts) in
  let tensors =
    Array.of_list
      (List.mapi
         (fun i (name, t) ->
           Hashtbl.add index_map name i;
           t)
         ts)
  in
  let m = { metadata_kv = h.hm_meta; tensors; index_map } in
  let* _ = validate m in
  Ok m

(* TensorView / SafeTensors *)

type tensor_view = {
  dtype : dtype;
  shape : int list;
  data : string; (* backing buffer *)
  offset : int; (* byte offset into [data] *)
  length : int; (* number of bytes *)
}

let tensor_view_new ~dtype ~shape ~data : (tensor_view, safetensor_error) result
    =
  (* owned buffer constructor: data must exactly match expected size *)
  let open Int64 in
  let nbits =
    let n_elements =
      List.fold_left
        (fun acc d ->
          match acc with
          | Error _ -> acc
          | Ok a -> (
              if d < 0 then Error Validation_overflow
              else
                match int64_mul_checked a (of_int d) with
                | Ok v -> Ok v
                | Error _ -> Error Validation_overflow))
        (Ok 1L) shape
    in
    match n_elements with
    | Error e -> Error e
    | Ok ne -> (
        let bs = of_int (bitsize dtype) in
        match int64_mul_checked ne bs with
        | Ok v -> Ok v
        | Error _ -> Error Validation_overflow)
  in
  match nbits with
  | Error e -> Error e
  | Ok nb ->
      if rem nb 8L <> 0L then Error Misaligned_slice
      else
        let size = to_int (div nb 8L) in
        if String.length data <> size then
          Error
            (Invalid_tensor_view
               (dtype_to_string dtype, shape, String.length data))
        else Ok { dtype; shape; data; offset = 0; length = size }

type safetensors = {
  metadata : metadata;
  data : string; (* the full payload (concatenated tensors) *)
}

let read_u64_le (s : string) (off : int) : int64 =
  let open Int64 in
  let get i = of_int (Char.code s.[off + i]) in
  logor (get 0)
    (logor
       (shift_left (get 1) 8)
       (logor
          (shift_left (get 2) 16)
          (logor
             (shift_left (get 3) 24)
             (logor
                (shift_left (get 4) 32)
                (logor
                   (shift_left (get 5) 40)
                   (logor (shift_left (get 6) 48) (shift_left (get 7) 56)))))))

let write_u64_le (b : Bytes.t) (off : int) (v : int64) : unit =
  let open Int64 in
  for i = 0 to 7 do
    Bytes.set b (off + i)
      (Char.chr (to_int (logand (shift_right v (8 * i)) 0xFFL)))
  done

let read_metadata (buffer : string) : (int * metadata, safetensor_error) result
    =
  let len = String.length buffer in
  if len < n_len then Error Header_too_small
  else
    let n = read_u64_le buffer 0 in
    if n > Int64.of_int max_header_size then Error Header_too_large
    else
      let n_int = Int64.to_int n in
      let stop =
        match Int64.to_int (Int64.add n (Int64.of_int n_len)) with
        | exception _ -> -1
        | v -> v
      in
      if stop < 0 || stop > len then Error Invalid_header_length
      else
        let header = String.sub buffer n_len n_int in
        if not (is_valid_utf8 header) then Error (Invalid_header "bad utf8")
        else
          try
            let j = Json_minimal.from_string header in
            let* h =
              match json_to_hashmetadata j with
              | Ok v -> Ok v
              | Error e -> Error (Invalid_header_deserialization e)
            in
            let* m = metadata_of_hash h in
            let* buffer_end = validate m in
            if buffer_end + n_len + n_int <> len then
              Error Metadata_incomplete_buffer
            else Ok (n_int, m)
          with Json_minimal.Parse_error e ->
            Error (Invalid_header_deserialization e)

let deserialize (buffer : string) : (safetensors, safetensor_error) result =
  let* n, metadata = read_metadata buffer in
  let data =
    String.sub buffer (n_len + n) (String.length buffer - (n_len + n))
  in
  Ok { metadata; data }

let tensors (st : safetensors) : (string * tensor_view) list =
  let names = ref [] in
  Hashtbl.iter (fun name _ -> names := name :: !names) st.metadata.index_map;
  let views =
    List.map
      (fun name ->
        let idx = Hashtbl.find st.metadata.index_map name in
        let info = st.metadata.tensors.(idx) in
        let s, e = info.data_offsets in
        let view =
          {
            dtype = info.dtype;
            shape = info.shape;
            data = st.data;
            offset = s;
            length = e - s;
          }
        in
        (name, view))
      !names
  in
  views

let iter (st : safetensors) : (string * tensor_view) list =
  (* ordered by internal index (offset order) *)
  let pairs = ref [] in
  Hashtbl.iter
    (fun name idx -> pairs := (idx, name) :: !pairs)
    st.metadata.index_map;
  let ordered = List.sort compare !pairs in
  List.map
    (fun (_, name) ->
      let idx = Hashtbl.find st.metadata.index_map name in
      let info = st.metadata.tensors.(idx) in
      let s, e = info.data_offsets in
      ( name,
        {
          dtype = info.dtype;
          shape = info.shape;
          data = st.data;
          offset = s;
          length = e - s;
        } ))
    ordered

let tensor (st : safetensors) (name : string) :
    (tensor_view, safetensor_error) result =
  match Hashtbl.find_opt st.metadata.index_map name with
  | None -> Error (Tensor_not_found name)
  | Some idx ->
      let info = st.metadata.tensors.(idx) in
      let s, e = info.data_offsets in
      Ok
        {
          dtype = info.dtype;
          shape = info.shape;
          data = st.data;
          offset = s;
          length = e - s;
        }

let names (st : safetensors) : string list =
  let ns = ref [] in
  Hashtbl.iter (fun name _ -> ns := name :: !ns) st.metadata.index_map;
  !ns

let len (st : safetensors) : int = Array.length st.metadata.tensors
let is_empty (st : safetensors) : bool = len st = 0

(* Slice (lazy slicing over bytes with sub-byte safety) *)

type bound = Unbounded | Excluded of int | Included of int
type tensor_indexer = Select of int | Narrow of bound * bound

type invalid_slice =
  | Too_many_slices
  | Slice_out_of_range of { dim_index : int; asked : int; dim_size : int }
  | Misaligned_slices

let bound_to_pair ~shape = function
  | Narrow (Unbounded, Unbounded) -> (0, shape)
  | Narrow (Unbounded, Excluded stop) -> (0, stop)
  | Narrow (Unbounded, Included stop) -> (0, stop + 1)
  | Narrow (Included s, Unbounded) -> (s, shape)
  | Narrow (Included s, Excluded stop) -> (s, stop)
  | Narrow (Included s, Included stop) -> (s, stop + 1)
  | Narrow (Excluded s, Unbounded) -> (s + 1, shape)
  | Narrow (Excluded s, Excluded stop) -> (s + 1, stop)
  | Narrow (Excluded s, Included stop) -> (s + 1, stop + 1)
  | Select s -> (s, s + 1)

type slice_iterator = {
  view : tensor_view;
  mutable indices : (int * int) list; (* byte spans, relative to view.offset *)
  newshape : int list;
}

let remaining_byte_len (it : slice_iterator) : int =
  List.fold_left (fun acc (s, e) -> acc + (e - s)) 0 it.indices

let newshape (it : slice_iterator) : int list = it.newshape

let slice_make (view : tensor_view) (slices : tensor_indexer list) :
    (slice_iterator, invalid_slice) result =
  let n_slice = List.length slices in
  let n_shape = List.length view.shape in
  if n_slice > n_shape then Error Too_many_slices
  else
    let newshape_arr = Array.make n_shape 0 in
    let newshape_idx = ref 0 in
    (* span in bits *)
    let span = ref (bitsize view.dtype) in
    let indices : (int * int) list ref = ref [] in
    (* iterate dimensions from last to first (row-major) *)
    let dims = Array.of_list view.shape in
    try
      for ri = n_shape - 1 downto 0 do
        let shape = dims.(ri) in
        let slice =
          if ri < n_slice then List.nth slices ri
          else Narrow (Unbounded, Unbounded)
        in
        let start, stop = bound_to_pair ~shape slice in
        (if start >= shape || stop > shape then
           let asked = if start >= shape then start else max 0 (stop - 1) in
           raise (Failure (Printf.sprintf "oob:%d:%d:%d" ri asked shape)));
        (match slice with
        | Narrow _ ->
            newshape_arr.(ri) <- stop - start;
            incr newshape_idx
        | Select _ -> ());
        (if !indices = [] then (
           if not (start = 0 && stop = shape) then (
             if start * !span mod 8 <> 0 then raise (Failure "misaligned");
             let offset = start * !span / 8 in
             if stop * !span mod 8 <> 0 then raise (Failure "misaligned");
             let small_span = (stop * !span / 8) - offset in
             indices := [ (offset, offset + small_span) ]))
         else
           let newidx = ref [] in
           for n = start to stop - 1 do
             if n * !span mod 8 <> 0 then raise (Failure "misaligned");
             let offset = n * !span / 8 in
             List.iter
               (fun (os, oe) -> newidx := (os + offset, oe + offset) :: !newidx)
               !indices
           done;
           indices := List.rev !newidx);
        span := !span * shape
      done;
      if !indices = [] then indices := [ (0, view.length) ];
      let indices = List.rev !indices in
      (* Filter out dimensions that were selected (size 0) *)
      let newshape =
        Array.to_list newshape_arr |> List.filter (fun x -> x > 0)
      in
      Ok { view; indices; newshape }
    with
    | Failure msg when msg = "misaligned" -> Error Misaligned_slices
    | Failure s when String.length s >= 4 && String.sub s 0 4 = "oob:" -> (
        let parts =
          String.split_on_char ':' (String.sub s 4 (String.length s - 4))
        in
        match parts with
        | [ d; a; sz ] ->
            Error
              (Slice_out_of_range
                 {
                   dim_index = int_of_string d;
                   asked = int_of_string a;
                   dim_size = int_of_string sz;
                 })
        | _ -> Error Misaligned_slices)
    | _ -> Error Misaligned_slices

let slice_next (it : slice_iterator) : string option =
  match it.indices with
  | [] -> None
  | (s, e) :: tl ->
      it.indices <- tl;
      Some (String.sub it.view.data (it.view.offset + s) (e - s))

(* Serialization *)

type prepared_data = {
  n_aligned : int;
  header_bytes : string;
  total_data_len : int; (* sum of tensor byte lengths *)
}

let prepare (data : (string * tensor_view) list)
    (data_info : (string * string) list option) :
    (prepared_data * tensor_view list, safetensor_error) result =
  (* Sort by descending dtype (alignment), then name ascending *)
  let data_sorted =
    List.sort
      (fun (ln, lt) (rn, rt) ->
        let cmp_dt = compare (dtype_rank rt.dtype) (dtype_rank lt.dtype) in
        if cmp_dt <> 0 then cmp_dt else compare ln rn)
      data
  in
  let tensors = ref [] in
  let hmetadata = ref [] in
  let offset = ref 0 in
  List.iter
    (fun (name, t) ->
      let n = t.length in
      let ti =
        {
          dtype = t.dtype;
          shape = t.shape;
          data_offsets = (!offset, !offset + n);
        }
      in
      offset := !offset + n;
      hmetadata := (name, ti) :: !hmetadata;
      tensors := t :: !tensors)
    data_sorted;
  let hmetadata = List.rev !hmetadata in
  (* Build metadata record *)
  let index_map = Hashtbl.create (List.length hmetadata) in
  let tensors_arr =
    Array.of_list
      (List.mapi
         (fun i (_k, ti) ->
           Hashtbl.add index_map (fst (List.nth hmetadata i)) i;
           ti)
         hmetadata)
  in
  (* Fix names into index_map: we must use the same order as tensors_arr *)
  Hashtbl.reset index_map;
  List.iteri (fun i (name, _ti) -> Hashtbl.add index_map name i) hmetadata;
  let meta = { metadata_kv = data_info; tensors = tensors_arr; index_map } in
  let* _ = validate meta in
  let json = metadata_to_json meta in
  let n_aligned = next_multiple_of (String.length json) n_len in
  let header_bytes =
    if n_aligned = String.length json then json
    else
      let b = Bytes.make n_aligned ' ' in
      Bytes.blit_string json 0 b 0 (String.length json);
      Bytes.to_string b
  in
  Ok ({ n_aligned; header_bytes; total_data_len = !offset }, List.rev !tensors)

let serialize (data : (string * tensor_view) list)
    (data_info : (string * string) list option) :
    (string, safetensor_error) result =
  let* prep, (tensors : tensor_view list) = prepare data data_info in
  let total = n_len + prep.n_aligned + prep.total_data_len in
  let b = Bytes.create total in
  write_u64_le b 0 (Int64.of_int prep.n_aligned);
  Bytes.blit_string prep.header_bytes 0 b n_len prep.n_aligned;
  let pos = ref (n_len + prep.n_aligned) in
  List.iter
    (fun (t : tensor_view) ->
      Bytes.blit_string t.data t.offset b !pos t.length;
      pos := !pos + t.length)
    tensors;
  Ok (Bytes.to_string b)

let serialize_to_file (data : (string * tensor_view) list)
    (data_info : (string * string) list option) (filename : string) :
    (unit, safetensor_error) result =
  let* s = serialize data data_info in
  try
    let oc = open_out_bin filename in
    output_string oc s;
    close_out oc;
    Ok ()
  with e -> Error (Io_error (Printexc.to_string e))

(* Small helpers mirroring Rust API naming where nice *)

module Slice = struct
  type t = slice_iterator
  type index = tensor_indexer
  type error = invalid_slice

  let make = slice_make
  let next = slice_next
  let remaining_byte_len = remaining_byte_len
  let newshape = newshape
  let select i = Select i
  let ( // ) (l : bound) (r : bound) = Narrow (l, r)
  let unbounded = Unbounded
  let included n = Included n
  let excluded n = Excluded n
end
