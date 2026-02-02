(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Exception types for dataset errors *)
exception Dataset_error of string
exception File_not_found of string
exception Invalid_parameter of string

type cardinality = Finite of int | Unknown | Infinite

type element_spec =
  | Unknown
  | Scalar of string
  | Tensor of int array * string
  | Tuple of element_spec list
  | Array of element_spec

type 'a job_result =
  | Job_value of 'a
  | Job_error of exn * Printexc.raw_backtrace

let arrays_equal arr1 arr2 =
  let len = Array.length arr1 in
  len = Array.length arr2
  &&
  let rec loop idx =
    if idx = len then true
    else if arr1.(idx) = arr2.(idx) then loop (idx + 1)
    else false
  in
  loop 0

let rec element_spec_equal a b =
  match (a, b) with
  | Unknown, Unknown -> true
  | Scalar s1, Scalar s2 -> String.equal s1 s2
  | Tensor (shape1, dtype1), Tensor (shape2, dtype2) ->
      arrays_equal shape1 shape2 && String.equal dtype1 dtype2
  | Tuple l1, Tuple l2 ->
      List.length l1 = List.length l2 && List.for_all2 element_spec_equal l1 l2
  | Array spec1, Array spec2 -> element_spec_equal spec1 spec2
  | _ -> false

(* Core dataset type - lazy sequence with metadata *)
type 'a t = {
  next : unit -> 'a option; (* Get next element *)
  cardinality : unit -> cardinality; (* Dataset cardinality *)
  reset : (unit -> unit) option; (* Reset to beginning if supported *)
  spec : unit -> element_spec; (* Element type specification *)
}

type ('elt, 'kind) tensor_dataset = ('elt, 'kind) Rune.t t
type tokenizer = string -> int array

(* Create a stateless whitespace tokenizer *)
let create_whitespace_tokenizer () : tokenizer =
  let vocab = Hashtbl.create 10000 in
  let next_id = ref 2 in
  (* Reserve 0=<bos>, 1=<eos> *)
  fun text ->
    let words =
      String.split_on_char ' ' text |> List.filter (fun s -> s <> "")
    in
    let get_or_add_token tok =
      match Hashtbl.find_opt vocab tok with
      | Some id -> id
      | None ->
          let id = !next_id in
          incr next_id;
          Hashtbl.add vocab tok id;
          id
    in
    Array.of_list (List.map get_or_add_token words)

(* Default instance for backward compatibility *)
let whitespace_tokenizer = create_whitespace_tokenizer ()

(* Internal: memory-mapped file handle *)
type mmap_handle = {
  fd : Unix.file_descr;
  data :
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
  size : int;
}

(* Helper: create mmap handle for a file *)
let create_mmap path =
  try
    let fd = Unix.openfile path [ Unix.O_RDONLY ] 0o644 in
    let size = (Unix.fstat fd).Unix.st_size in
    let data =
      Unix.map_file fd Bigarray.char Bigarray.c_layout false [| size |]
      |> Bigarray.array1_of_genarray
    in
    { fd; data; size }
  with
  | Unix.Unix_error (Unix.ENOENT, _, _) ->
      raise (File_not_found ("Dataset file not found: " ^ path))
  | Unix.Unix_error (err, _, _) ->
      raise
        (Dataset_error
           ("Failed to open file " ^ path ^ ": " ^ Unix.error_message err))

(* Helper: close mmap handle *)
let close_mmap handle = Unix.close handle.fd

(* Helper: read chunk from mmap as string *)
let read_mmap_chunk handle ~offset ~length =
  let actual_length = min length (handle.size - offset) in
  if actual_length <= 0 then ""
  else
    let src = Bigarray.Array1.sub handle.data offset actual_length in
    let bytes = Bytes.create actual_length in
    for i = 0 to actual_length - 1 do
      Bytes.unsafe_set bytes i (Bigarray.Array1.unsafe_get src i)
    done;
    Bytes.unsafe_to_string bytes

(* ───── Dataset Creation ───── *)

let from_array arr =
  let idx = ref 0 in
  let reset () = idx := 0 in
  {
    next =
      (fun () ->
        if !idx < Array.length arr then (
          let v = arr.(!idx) in
          incr idx;
          Some v)
        else None);
    cardinality = (fun () -> Finite (Array.length arr));
    reset = Some reset;
    spec = (fun () -> Unknown);
  }

let from_list lst =
  let items = ref lst in
  let original = lst in
  let reset () = items := original in
  {
    next =
      (fun () ->
        match !items with
        | [] -> None
        | h :: t ->
            items := t;
            Some h);
    cardinality = (fun () -> Finite (List.length lst));
    reset = Some reset;
    spec = (fun () -> Unknown);
  }

let from_seq seq =
  let s = ref seq in
  {
    next =
      (fun () ->
        match Seq.uncons !s with
        | None -> None
        | Some (h, t) ->
            s := t;
            Some h);
    cardinality = (fun () -> Unknown);
    reset = None;
    spec = (fun () -> Unknown);
  }

let from_tensor tensor =
  let shape = Rune.shape tensor in
  if Array.length shape = 0 then
    raise
      (Invalid_parameter
         "from_tensor: tensor must have at least one dimension to create a \
          dataset");
  let n = shape.(0) in
  let idx = ref 0 in
  let reset () = idx := 0 in
  let slice_shape =
    if Array.length shape <= 1 then [||]
    else Array.sub shape 1 (Array.length shape - 1)
  in
  let dtype = Nx_core.Dtype.to_string (Rune.dtype tensor) in
  {
    next =
      (fun () ->
        if !idx < n then (
          let slice = Rune.slice [ I !idx ] tensor in
          incr idx;
          Some slice)
        else None);
    cardinality = (fun () -> Finite n);
    reset = Some reset;
    spec = (fun () -> Tensor (slice_shape, dtype));
  }

let from_tensors (x, y) =
  let x_shape = Rune.shape x in
  let y_shape = Rune.shape y in
  if Array.length x_shape = 0 || Array.length y_shape = 0 then
    raise
      (Invalid_parameter
         "from_tensors: tensors must have at least one dimension to create a \
          dataset");
  if x_shape.(0) <> y_shape.(0) then
    raise
      (Invalid_parameter
         (Printf.sprintf "from_tensors: mismatched leading dimension (%d vs %d)"
            x_shape.(0) y_shape.(0)));
  let n = x_shape.(0) in
  let idx = ref 0 in
  let reset () = idx := 0 in
  let x_slice_shape =
    if Array.length x_shape <= 1 then [||]
    else Array.sub x_shape 1 (Array.length x_shape - 1)
  in
  let y_slice_shape =
    if Array.length y_shape <= 1 then [||]
    else Array.sub y_shape 1 (Array.length y_shape - 1)
  in
  let x_dtype = Nx_core.Dtype.to_string (Rune.dtype x) in
  let y_dtype = Nx_core.Dtype.to_string (Rune.dtype y) in
  {
    next =
      (fun () ->
        if !idx < n then (
          let x_slice = Rune.slice [ I !idx ] x in
          let y_slice = Rune.slice [ I !idx ] y in
          incr idx;
          Some (x_slice, y_slice))
        else None);
    cardinality = (fun () -> Finite n);
    reset = Some reset;
    spec =
      (fun () ->
        Tuple
          [ Tensor (x_slice_shape, x_dtype); Tensor (y_slice_shape, y_dtype) ]);
  }

(* ───── Text Data Sources ───── *)
let from_text_file ?(encoding = `UTF8) ?(chunk_size = 65536) path =
  let decoder_encoding, preprocess_chunk =
    match encoding with
    | `UTF8 -> (Some `UTF_8, Fun.id)
    | `ASCII -> (Some `UTF_8, Fun.id)
    | `LATIN1 ->
        let convert chunk =
          let len = String.length chunk in
          let buf = Buffer.create len in
          for i = 0 to len - 1 do
            let code = Char.code chunk.[i] in
            if code < 0x80 then Buffer.add_char buf chunk.[i]
            else (
              Buffer.add_char buf (Char.unsafe_chr (0xC0 lor (code lsr 6)));
              Buffer.add_char buf (Char.unsafe_chr (0x80 lor (code land 0x3F))))
          done;
          Buffer.contents buf
        in
        (Some `UTF_8, convert)
  in
  let make_decoder () = Uutf.decoder ?encoding:decoder_encoding `Manual in
  let handle_ref = ref None in
  let file_size = ref 0 in
  let offset = ref 0 in
  let closed = ref false in
  let buf = Buffer.create 512 in
  let lines_queue = Queue.create () in
  let decoder = ref (make_decoder ()) in

  let open_handle () =
    let handle = create_mmap path in
    file_size := handle.size;
    handle_ref := Some handle;
    handle
  in
  let ensure_handle () =
    match !handle_ref with Some h -> h | None -> open_handle ()
  in
  let close_handle () =
    match !handle_ref with
    | None -> ()
    | Some h ->
        (try close_mmap h with
        | Unix.Unix_error (Unix.EBADF, _, _) -> ()
        | exn -> raise exn);
        handle_ref := None
  in
  ignore (open_handle ());

  let push_line_from_buf () =
    let line = Buffer.contents buf in
    Buffer.clear buf;
    let line =
      let len = String.length line in
      if len > 0 && line.[len - 1] = '\r' then String.sub line 0 (len - 1)
      else line
    in
    Queue.add line lines_queue
  in

  let rec fill_queue () =
    if Queue.is_empty lines_queue && not !closed then
      match Uutf.decode !decoder with
      | `Uchar u ->
          if Uchar.to_int u = 0x000A then push_line_from_buf ()
          else Uutf.Buffer.add_utf_8 buf u;
          if Queue.is_empty lines_queue then fill_queue ()
      | `Malformed _ ->
          Uutf.Buffer.add_utf_8 buf Uutf.u_rep;
          fill_queue ()
      | `Await ->
          if !offset >= !file_size then (
            Uutf.Manual.src !decoder (Bytes.create 0) 0 0;
            fill_queue ())
          else
            let handle = ensure_handle () in
            let raw_chunk =
              read_mmap_chunk handle ~offset:!offset ~length:chunk_size
            in
            offset := !offset + String.length raw_chunk;
            let chunk = preprocess_chunk raw_chunk in
            if chunk = "" then (
              Uutf.Manual.src !decoder (Bytes.create 0) 0 0;
              fill_queue ())
            else
              let bytes = Bytes.unsafe_of_string chunk in
              Uutf.Manual.src !decoder bytes 0 (Bytes.length bytes);
              fill_queue ()
      | `End ->
          if Buffer.length buf > 0 then push_line_from_buf ();
          close_handle ();
          closed := true
  in

  let rec next_line () =
    if not (Queue.is_empty lines_queue) then Some (Queue.take lines_queue)
    else if !closed then None
    else (
      fill_queue ();
      if not (Queue.is_empty lines_queue) then Some (Queue.take lines_queue)
      else if !closed then None
      else next_line ())
  in

  let reset =
    let reset_state () =
      Buffer.clear buf;
      Queue.clear lines_queue;
      offset := 0;
      closed := false;
      decoder := make_decoder ();
      close_handle ();
      ignore (open_handle ())
    in
    Some reset_state
  in

  {
    next = next_line;
    cardinality = (fun () -> Unknown);
    reset;
    spec = (fun () -> Scalar "string");
  }

let from_text_files ?(encoding = `UTF8) ?(chunk_size = 65536) paths =
  let paths_array = Array.of_list paths in
  let total_paths = Array.length paths_array in
  let current_file = ref 0 in
  let current_dataset = ref None in

  let rec next () =
    match !current_dataset with
    | None ->
        if !current_file >= total_paths then None
        else
          let path = paths_array.(!current_file) in
          let ds = from_text_file ~encoding ~chunk_size path in
          current_dataset := Some ds;
          incr current_file;
          next ()
    | Some ds -> (
        match ds.next () with
        | None ->
            current_dataset := None;
            next ()
        | some_line -> some_line)
  in

  let reset =
    (* Resetting recreates datasets lazily on demand *)
    Some
      (fun () ->
        (match !current_dataset with
        | None -> ()
        | Some ds -> ( match ds.reset with Some f -> f () | None -> ()));
        current_dataset := None;
        current_file := 0)
  in

  {
    next;
    cardinality = (fun () -> Unknown);
    reset;
    spec = (fun () -> Scalar "string");
  }

let from_jsonl ?field path =
  let text_ds = from_text_file path in
  let field_name = Option.value field ~default:"text" in

  let next () =
    match text_ds.next () with
    | None -> None
    | Some line -> (
        try
          let json = Yojson.Safe.from_string line in
          match json with
          | `Assoc fields -> (
              match List.assoc_opt field_name fields with
              | Some (`String s) -> Some s
              | _ ->
                  raise
                    (Dataset_error
                       ("Field '" ^ field_name ^ "' is not a string in JSONL: "
                      ^ line)))
          | _ ->
              raise
                (Dataset_error ("Invalid JSONL format, expected object: " ^ line))
        with
        | Dataset_error _ as e -> raise e
        | Yojson.Json_error msg ->
            raise
              (Dataset_error
                 ("Failed to parse JSONL: " ^ msg ^ " in line: " ^ line)))
  in

  {
    next;
    cardinality = (fun () -> Unknown);
    reset = text_ds.reset;
    spec = (fun () -> Scalar "string");
  }

let ensure_non_negative ~name value =
  if value < 0 then
    raise
      (Invalid_parameter
         (name ^ " must be non-negative, got " ^ string_of_int value))

let from_csv ?(separator = ',') ?(text_column = 0) ?(has_header = true) path =
  ensure_non_negative ~name:"text_column" text_column;

  let text_ds = from_text_file path in
  let skipped_header = ref (not has_header) in

  let reset =
    match text_ds.reset with
    | None -> None
    | Some reset_fn ->
        Some
          (fun () ->
            skipped_header := not has_header;
            reset_fn ())
  in

  let split_csv line =
    (* Simple CSV split - doesn't handle quoted fields *)
    String.split_on_char separator line
  in

  let consume_header () =
    if not !skipped_header then (
      skipped_header := true;
      ignore (text_ds.next ()))
  in

  let rec take_field fields idx =
    match (fields, idx) with
    | [], _ -> None
    | x :: _, 0 -> Some x
    | _ :: tl, n -> take_field tl (n - 1)
  in
  let rec read_next () =
    match text_ds.next () with
    | None -> None
    | Some line -> (
        match take_field (split_csv line) text_column with
        | Some text -> Some text
        | None -> read_next ())
  in
  let next () =
    consume_header ();
    read_next ()
  in
  {
    next;
    cardinality = text_ds.cardinality;
    reset;
    spec = (fun () -> Scalar "string");
  }

let from_csv_with_labels ?(separator = ',') ?(text_column = 0)
    ?(has_header = true) ~label_column path =
  ensure_non_negative ~name:"text_column" text_column;
  ensure_non_negative ~name:"label_column" label_column;

  let text_ds = from_text_file path in
  let skipped_header = ref (not has_header) in

  let reset =
    match text_ds.reset with
    | None -> None
    | Some reset_fn ->
        Some
          (fun () ->
            skipped_header := not has_header;
            reset_fn ())
  in

  let split_csv line = String.split_on_char separator line in

  let consume_header () =
    if not !skipped_header then (
      skipped_header := true;
      ignore (text_ds.next ()))
  in

  let rec take_field fields idx =
    match (fields, idx) with
    | [], _ -> None
    | x :: _, 0 -> Some x
    | _ :: tl, n -> take_field tl (n - 1)
  in

  let rec read_next () =
    match text_ds.next () with
    | None -> None
    | Some line -> (
        let fields = split_csv line in
        let text_opt = take_field fields text_column in
        let label_opt = take_field fields label_column in
        match (text_opt, label_opt) with
        | Some text, Some label -> Some (text, label)
        | _ -> read_next ())
  in

  let next () =
    consume_header ();
    read_next ()
  in

  {
    next;
    cardinality = text_ds.cardinality;
    reset;
    spec = (fun () -> Tuple [ Scalar "string"; Scalar "string" ]);
  }

let from_text ~tokenizer path =
  (* Read entire file as a single string *)
  let ic = open_in path in
  let len = in_channel_length ic in
  let content = really_input_string ic len in
  close_in ic;

  (* Tokenize the entire content *)
  let tokens = tokenizer content in

  (* Create a single-element dataset containing all tokens *)
  from_array [| tokens |]

let sliding_window ~block_size ~tokenize texts =
  (* Efficient sliding windows over each text independently. For a tokenized
     sequence ids with length L, we produce (L-1) windows: for i in 0..L-2,
     context is last [block_size] tokens up to i, left-padded with the first id.
     Target is ids[i+1]. *)
  let to_array lst = Array.of_list lst in
  (* First pass: tokenize and compute total windows *)
  let token_arrays = List.map (fun s -> to_array (tokenize s)) texts in
  let total =
    List.fold_left
      (fun acc ids -> acc + max 0 (Array.length ids - 1))
      0 token_arrays
  in
  (* Allocate flat buffers *)
  let x_data = Array.make (total * block_size) 0.0 in
  let y_idx = Array.make total 0.0 in
  (* Fill in a single pass *)
  let idx = ref 0 in
  List.iter
    (fun ids ->
      let len = Array.length ids in
      if len > 1 then
        let pad = ids.(0) in
        for i = 0 to len - 2 do
          (* target is next id *)
          y_idx.(!idx) <- float_of_int ids.(i + 1);
          (* context: last [block_size] tokens up to i, left-padded with first
             id *)
          let start = i - block_size + 1 in
          for k = 0 to block_size - 1 do
            let src_pos = start + k in
            let v = if src_pos < 0 then pad else ids.(src_pos) in
            x_data.((!idx * block_size) + k) <- float_of_int v
          done;
          incr idx
        done)
    token_arrays;
  (* Build Rune tensors and return a dataset of element pairs *)
  let x = Rune.create Rune.float32 [| total; block_size |] x_data in
  let y = Rune.create Rune.float32 [| total |] y_idx in
  from_tensors (x, y)

(* ───── Transformations ───── *)

let map ?spec f dataset =
  let spec_fn =
    match spec with
    | Some provided -> fun () -> provided
    | None -> fun () -> Unknown
  in
  {
    next = (fun () -> Option.map f (dataset.next ()));
    cardinality = dataset.cardinality;
    reset = dataset.reset;
    spec = spec_fn;
  }

let from_file parser path =
  let text_ds = from_text_file path in
  map parser text_ds

let filter pred dataset =
  let rec next () =
    match dataset.next () with
    | None -> None
    | Some x as result -> if pred x then result else next ()
  in
  {
    next;
    cardinality = (fun () -> Unknown);
    reset = dataset.reset;
    spec = dataset.spec;
  }

let flat_map f dataset =
  let current = ref None in

  let rec next () =
    match !current with
    | Some ds -> (
        match ds.next () with
        | None ->
            current := None;
            next ()
        | some_val -> some_val)
    | None -> (
        match dataset.next () with
        | None -> None
        | Some x ->
            current := Some (f x);
            next ())
  in

  {
    next;
    cardinality = (fun () -> Unknown);
    reset = None;
    spec = (fun () -> Unknown);
  }

let zip ds1 ds2 =
  let next () =
    match (ds1.next (), ds2.next ()) with
    | Some x, Some y -> Some (x, y)
    | _ -> None
  in
  let cardinality () =
    match (ds1.cardinality (), ds2.cardinality ()) with
    | Finite l1, Finite l2 -> Finite (min l1 l2)
    | Infinite, _ | _, Infinite -> Infinite
    | _ -> Unknown
  in
  {
    next;
    cardinality;
    reset = None;
    spec =
      (fun () ->
        match (ds1.spec (), ds2.spec ()) with
        | spec1, spec2 -> Tuple [ spec1; spec2 ]);
  }

let concatenate ds1 ds2 =
  let first_done = ref false in

  let next () =
    if not !first_done then
      match ds1.next () with
      | None ->
          first_done := true;
          ds2.next ()
      | some_val -> some_val
    else ds2.next ()
  in

  let cardinality () =
    match (ds1.cardinality (), ds2.cardinality ()) with
    | Finite l1, Finite l2 -> Finite (l1 + l2)
    | Infinite, _ | _, Infinite -> Infinite
    | _ -> Unknown
  in
  let combined_spec =
    let spec1 = ds1.spec () in
    let spec2 = ds2.spec () in
    if element_spec_equal spec1 spec2 then spec1 else Unknown
  in

  { next; cardinality; reset = None; spec = (fun () -> combined_spec) }

let interleave datasets =
  let n = List.length datasets in
  let current = ref 0 in
  let ds_array = Array.of_list datasets in

  let next () =
    let rec try_next attempts =
      if attempts >= n then None
      else
        let idx = !current in
        current := (!current + 1) mod n;
        match ds_array.(idx).next () with
        | None -> try_next (attempts + 1)
        | some_val -> some_val
    in
    try_next 0
  in

  {
    next;
    cardinality = (fun () -> Unknown);
    reset = None;
    spec = (fun () -> Unknown);
  }

(* ───── Text Processing ───── *)

let enumerate dataset =
  let counter = ref 0 in
  map
    (fun elem ->
      let idx = !counter in
      incr counter;
      (idx, elem))
    dataset

let tokenize tokenizer_fn ?max_length ?padding ?truncation ?add_special_tokens
    text_dataset =
  let max_len = Option.value max_length ~default:512 in
  let should_truncate = Option.value truncation ~default:true in
  let add_special = Option.value add_special_tokens ~default:true in

  map
    (fun text ->
      let tokens = tokenizer_fn text in
      let tokens =
        if add_special then Array.append [| 0 |] (Array.append tokens [| 1 |])
          (* 0=<bos>, 1=<eos> *)
        else tokens
      in
      let tokens =
        if should_truncate && Array.length tokens > max_len then
          Array.sub tokens 0 max_len
        else tokens
      in
      match padding with
      | None | Some `None -> tokens
      | Some (`Max pad_len) ->
          if Array.length tokens < pad_len then
            Array.append tokens (Array.make (pad_len - Array.length tokens) 0)
          else tokens
      | Some `Dynamic -> tokens (* Dynamic padding handled in batch *))
    text_dataset

let normalize ?lowercase ?remove_punctuation ?collapse_whitespace text_dataset =
  map
    (fun text ->
      let text =
        if Option.value lowercase ~default:false then
          String.lowercase_ascii text
        else text
      in
      let text =
        if Option.value remove_punctuation ~default:false then
          (* Remove punctuation *)
          String.map
            (fun c -> if String.contains ".,;:!?'\"()[]{}" c then ' ' else c)
            text
        else text
      in
      let text =
        if Option.value collapse_whitespace ~default:false then
          (* Collapse multiple spaces to single space *)
          let parts =
            String.split_on_char ' ' text |> List.filter (fun s -> s <> "")
          in
          String.concat " " parts
        else text
      in
      text)
    text_dataset

(* ───── Batching ───── *)

(* Internal helper for generic batching *)
let batch_generic ?(drop_remainder = false) size dataset =
  if size <= 0 then
    raise
      (Invalid_parameter
         ("batch_generic: size must be positive, got " ^ string_of_int size));

  let buffer = ref [] in
  let next () =
    (* Fill buffer *)
    let needed = size - List.length !buffer in
    let filled = ref 0 in
    while !filled < needed do
      match dataset.next () with
      | None -> filled := needed (* Exit loop *)
      | Some x ->
          buffer := x :: !buffer;
          incr filled
    done;
    if !buffer = [] then None
    else if List.length !buffer < size && drop_remainder then (
      buffer := [];
      None)
    else
      let batch = Array.of_list (List.rev !buffer) in
      buffer := [];
      Some batch
  in
  {
    next;
    cardinality = (fun () -> Unknown);
    reset =
      Some
        (fun () ->
          (* Clear local buffer and reset underlying dataset if possible *)
          buffer := [];
          match dataset.reset with Some f -> f () | None -> ());
    spec = (fun () -> Array (dataset.spec ()));
  }

(* Tensor-aware batch that automatically stacks tensor pairs *)
let batch ?(drop_remainder = false) size dataset =
  batch_generic ~drop_remainder size dataset
  |> map (fun batch_arr ->
      (* Stack the batch of tensor pairs into batched tensors *)
      let images, labels = Array.split batch_arr in
      ( Rune.stack ~axis:0 (Array.to_list images),
        Rune.stack ~axis:0 (Array.to_list labels) ))

let batch_map ?(drop_remainder = false) size f dataset =
  let batched = batch_generic ~drop_remainder size dataset in
  map f batched

let bucket_by_length ?boundaries ?batch_sizes ?(drop_remainder = false)
    length_fn dataset =
  let boundaries = Option.value boundaries ~default:[ 100; 200; 300 ] in
  let batch_sizes = Option.value batch_sizes ~default:[ 64; 32; 16; 8 ] in

  (* Validate parameters *)
  if List.length batch_sizes <> List.length boundaries + 1 then
    raise
      (Invalid_parameter
         (Printf.sprintf
            "bucket_by_length: batch_sizes length (%d) must be boundaries \
             length + 1 (%d)"
            (List.length batch_sizes)
            (List.length boundaries + 1)));

  (* Create buckets to accumulate elements *)
  let num_buckets = List.length boundaries + 1 in
  let buckets = Array.init num_buckets (fun _ -> Queue.create ()) in
  let bucket_batch_sizes = Array.of_list batch_sizes in
  let pending_batches = Queue.create () in
  let exhausted = ref false in

  (* Function to determine bucket index *)
  let get_bucket_idx len =
    let rec find_bucket idx = function
      | [] -> idx
      | bound :: rest -> if len < bound then idx else find_bucket (idx + 1) rest
    in
    find_bucket 0 boundaries
  in

  let process_element elem =
    let len = length_fn elem in
    let bucket_idx = min (get_bucket_idx len) (num_buckets - 1) in
    let bucket = buckets.(bucket_idx) in
    Queue.add elem bucket;

    (* Check if bucket is full *)
    let batch_size =
      if bucket_idx < Array.length bucket_batch_sizes then
        bucket_batch_sizes.(bucket_idx)
      else 8 (* Default for overflow bucket *)
    in

    if Queue.length bucket >= batch_size then (
      (* Extract batch *)
      let batch = Array.init batch_size (fun _ -> Queue.take bucket) in
      Queue.add batch pending_batches;
      true (* Signal that we produced a batch *))
    else false
  in

  (* Lazy streaming: fill buckets and produce batches on demand *)
  let rec next () =
    (* Try to get a pending batch first *)
    if not (Queue.is_empty pending_batches) then
      Some (Queue.take pending_batches)
    else if !exhausted then (
      (* Dataset exhausted - flush remaining buckets *)
      let found_batch = ref false in
      Array.iteri
        (fun bucket_idx bucket ->
          if (not (Queue.is_empty bucket)) && not !found_batch then (
            let bucket_size = Queue.length bucket in
            let target_size =
              if bucket_idx < Array.length bucket_batch_sizes then
                bucket_batch_sizes.(bucket_idx)
              else 8
            in
            if drop_remainder && bucket_size < target_size then
              Queue.clear bucket
            else
              let batch = Array.init bucket_size (fun _ -> Queue.take bucket) in
              Queue.add batch pending_batches;
              found_batch := true))
        buckets;
      if Queue.is_empty pending_batches then None
      else Some (Queue.take pending_batches))
    else
      (* Try to fill buckets until we get a batch *)
      let rec fill_until_batch () =
        match dataset.next () with
        | None ->
            exhausted := true;
            next () (* Recurse to flush buckets *)
        | Some elem ->
            if process_element elem then Some (Queue.take pending_batches)
            else fill_until_batch ()
      in
      fill_until_batch ()
  in

  {
    next;
    cardinality = (fun () -> Unknown);
    reset = None;
    spec = (fun () -> Array Unknown);
  }

(* ───── Iteration Control ───── *)

let take n dataset =
  if n < 0 then
    raise
      (Invalid_parameter ("take: n must be non-negative, got " ^ string_of_int n));

  let count = ref 0 in

  let next () =
    if !count >= n then None
    else (
      incr count;
      dataset.next ())
  in

  let cardinality () =
    match dataset.cardinality () with
    | Finite l -> Finite (min n l)
    | Infinite -> Finite n
    | Unknown -> Unknown
  in
  let reset =
    match dataset.reset with
    | None -> None
    | Some reset_fn ->
        Some
          (fun () ->
            count := 0;
            reset_fn ())
  in

  { next; cardinality; reset; spec = dataset.spec }

let skip n dataset =
  if n < 0 then
    raise
      (Invalid_parameter ("skip: n must be non-negative, got " ^ string_of_int n));

  let skipped = ref 0 in

  (* Skip initial elements *)
  while !skipped < n do
    match dataset.next () with None -> skipped := n | Some _ -> incr skipped
  done;

  {
    next = dataset.next;
    cardinality =
      (fun () ->
        match dataset.cardinality () with
        | Finite l -> Finite (max 0 (l - n))
        | other -> other);
    reset = dataset.reset;
    spec = dataset.spec;
  }

let repeat ?count dataset =
  let current_count = ref 0 in
  let needs_reset = ref false in

  let rec next () =
    if !needs_reset then (
      match dataset.reset with
      | None -> None
      | Some reset_fn ->
          reset_fn ();
          needs_reset := false;
          incr current_count;
          dataset.next ())
    else
      match dataset.next () with
      | None -> (
          match count with
          | Some c when !current_count >= c - 1 -> None
          | _ ->
              needs_reset := true;
              next ())
      | some_val -> some_val
  in

  let cardinality () =
    match (count, dataset.cardinality ()) with
    | Some c, Finite l -> Finite (c * l)
    | None, _ -> Infinite
    | _ -> Unknown
  in

  { next; cardinality; reset = None; spec = dataset.spec }

let window ?(shift = -1) ?(stride = 1) ?(drop_remainder = false) size dataset =
  if size <= 0 then
    raise
      (Invalid_parameter
         ("window: size must be positive, got " ^ string_of_int size));
  if stride <= 0 then
    raise
      (Invalid_parameter
         ("window: stride must be positive, got " ^ string_of_int stride));

  let actual_shift = if shift = -1 then size else shift in
  if actual_shift <= 0 then
    raise
      (Invalid_parameter
         ("window: shift must be positive, got " ^ string_of_int actual_shift));

  let buffer = Array.make size (Obj.magic 0) in
  let start = ref 0 in
  let filled = ref 0 in
  let finished = ref false in

  let fill () =
    while !filled < size && not !finished do
      match dataset.next () with
      | None -> finished := true
      | Some x ->
          let idx = (!start + !filled) mod size in
          buffer.(idx) <- x;
          incr filled
    done
  in

  let next () =
    fill ();
    if !filled = 0 then None
    else
      let window_len = if !filled >= size then size else !filled in
      if window_len < size && drop_remainder && !finished then (
        filled := 0;
        None)
      else
        let out_len =
          if window_len <= 0 then 0 else 1 + ((window_len - 1) / stride)
        in
        if out_len = 0 then None
        else
          let tail = !start in
          let out =
            Array.init out_len (fun i ->
                let idx = (tail + (i * stride)) mod size in
                buffer.(idx))
          in
          let to_drop = min actual_shift window_len in
          start := (!start + to_drop) mod size;
          filled := !filled - to_drop;
          Some out
  in

  {
    next;
    cardinality = (fun () -> Unknown);
    reset = None;
    spec = (fun () -> Array (dataset.spec ()));
  }

(* ───── Shuffling And Sampling ───── *)

let shuffle ?rng ?(buffer_size = 10000) dataset =
  if buffer_size <= 0 then
    raise
      (Invalid_parameter
         ("shuffle: buffer_size must be positive, got "
        ^ string_of_int buffer_size));

  let buffer = Array.make buffer_size None in
  let buffer_count = ref 0 in
  (* Use provided RNG or create new one *)
  let seed =
    match rng with Some key -> Some [| Rune.Rng.to_int key |] | None -> None
  in
  let create_rand_state () =
    match seed with
    | Some arr -> Random.State.make (Array.copy arr)
    | None -> Random.State.make_self_init ()
  in
  let rand_state = ref (create_rand_state ()) in

  (* Fill initial buffer *)
  let exhausted = ref false in
  let fill_buffer () =
    while !buffer_count < buffer_size && not !exhausted do
      match dataset.next () with
      | None -> exhausted := true
      | Some x ->
          buffer.(!buffer_count) <- Some x;
          incr buffer_count
    done
  in
  fill_buffer ();

  let next () =
    if !buffer_count = 0 then None
    else
      (* Pick random element from buffer *)
      let idx = Random.State.int !rand_state !buffer_count in
      let result = buffer.(idx) in

      (* Try to refill that position *)
      (match dataset.next () with
      | None ->
          (* Move last element to this position *)
          decr buffer_count;
          if idx < !buffer_count then buffer.(idx) <- buffer.(!buffer_count)
      | Some x -> buffer.(idx) <- Some x);

      result
  in

  {
    next;
    cardinality = dataset.cardinality;
    reset =
      Some
        (fun () ->
          (* Reset underlying dataset, clear buffer, and refill *)
          (match dataset.reset with Some f -> f () | None -> ());
          for i = 0 to buffer_size - 1 do
            buffer.(i) <- None
          done;
          buffer_count := 0;
          exhausted := false;
          rand_state := create_rand_state ();
          fill_buffer ());
    spec = dataset.spec;
  }

let sample ?rng ?(replacement = false) n dataset =
  if replacement then
    (* For sampling with replacement, we need the full dataset *)
    (* This is inherently eager, but we can make it lazy by deferring collection *)
    let elements = ref None in
    let collected = ref false in
    let seed =
      match rng with Some key -> Some [| Rune.Rng.to_int key |] | None -> None
    in
    let create_rand_state () =
      match seed with
      | Some arr -> Random.State.make (Array.copy arr)
      | None -> Random.State.make_self_init ()
    in
    let rand_state = ref (create_rand_state ()) in

    let ensure_collected () =
      if not !collected then (
        let items = ref [] in
        (* Collect all elements *)
        let continue = ref true in
        while !continue do
          match dataset.next () with
          | None -> continue := false
          | Some x -> items := x :: !items
        done;
        elements := Some (Array.of_list (List.rev !items));
        collected := true)
    in

    let count = ref 0 in
    let next () =
      if !count >= n then None
      else (
        ensure_collected ();
        match !elements with
        | None | Some [||] -> None
        | Some arr ->
            let len = Array.length arr in
            incr count;
            Some arr.(Random.State.int !rand_state len))
    in
    {
      next;
      cardinality = (fun () -> Finite n);
      reset =
        Some
          (fun () ->
            count := 0;
            rand_state := create_rand_state ();
            match dataset.reset with
            | None -> ()
            | Some reset_fn ->
                elements := None;
                collected := false;
                reset_fn ());
      spec = dataset.spec;
    }
  else
    (* Sample without replacement by shuffling and taking first n *)
    take n (shuffle ?rng ~buffer_size:(min 10000 n) dataset)

let weighted_sample ?rng ~weights n dataset =
  let weights_len = Array.length weights in
  if weights_len = 0 then
    raise (Invalid_parameter "weighted_sample: weights array must be non-empty");
  Array.iteri
    (fun idx w ->
      if Float.is_nan w || w < 0. then
        raise
          (Invalid_parameter
             (Printf.sprintf
                "weighted_sample: weight at index %d must be >= 0 and finite"
                idx)))
    weights;
  let total_weight = Array.fold_left ( +. ) 0. weights in
  if (not (Float.is_finite total_weight)) || total_weight <= 0. then
    raise
      (Invalid_parameter
         "weighted_sample: sum of weights must be positive and finite");
  let cumulative = Array.make weights_len 0. in
  let sum = ref 0. in
  for i = 0 to weights_len - 1 do
    sum := !sum +. weights.(i);
    cumulative.(i) <- !sum /. total_weight
  done;
  let seed =
    match rng with Some key -> Some [| Rune.Rng.to_int key |] | None -> None
  in
  let create_rand_state () =
    match seed with
    | Some arr -> Random.State.make (Array.copy arr)
    | None -> Random.State.make_self_init ()
  in
  let rand_state = ref (create_rand_state ()) in

  (* Lazy sampling - collect elements only when needed *)
  let elements = ref None in
  let collected = ref false in

  let ensure_collected () =
    if not !collected then (
      let items = ref [] in
      (* Collect all elements *)
      let continue = ref true in
      while !continue do
        match dataset.next () with
        | None -> continue := false
        | Some x -> items := x :: !items
      done;
      let arr = Array.of_list (List.rev !items) in
      if Array.length arr <> weights_len then
        raise
          (Invalid_parameter
             (Printf.sprintf
                "weighted_sample: weights length (%d) must equal dataset \
                 length (%d)"
                weights_len (Array.length arr)));
      elements := Some arr;
      collected := true)
  in

  let count = ref 0 in
  let next () =
    if !count >= n then None
    else (
      ensure_collected ();
      match !elements with
      | None | Some [||] -> None
      | Some arr ->
          let r = Random.State.float !rand_state 1.0 in
          (* Binary search for the right bucket *)
          let idx =
            let rec find_bucket low high =
              if low >= high then low
              else
                let mid = (low + high) / 2 in
                if cumulative.(mid) < r then find_bucket (mid + 1) high
                else find_bucket low mid
            in
            find_bucket 0 (weights_len - 1)
          in
          incr count;
          if idx < Array.length arr then Some arr.(idx) else None)
  in

  {
    next;
    cardinality = (fun () -> Finite n);
    reset =
      Some
        (fun () ->
          count := 0;
          rand_state := create_rand_state ();
          match dataset.reset with
          | None -> ()
          | Some reset_fn ->
              elements := None;
              collected := false;
              reset_fn ());
    spec = dataset.spec;
  }

(* ───── Caching And Prefetching ───── *)

let rec cache ?directory dataset =
  match directory with
  | None ->
      (* In-memory cache *)
      let cache_data = ref [] in
      let cache_complete = ref false in
      let cache_iter = ref 0 in

      let next () =
        if !cache_complete then
          (* Read from cache *)
          if !cache_iter < List.length !cache_data then (
            let result = List.nth !cache_data !cache_iter in
            incr cache_iter;
            Some result)
          else None
        else
          (* Build cache *)
          match dataset.next () with
          | None ->
              cache_complete := true;
              cache_data := List.rev !cache_data;
              None
          | Some x ->
              cache_data := x :: !cache_data;
              Some x
      in

      let reset () = if !cache_complete then cache_iter := 0 else () in

      {
        next;
        cardinality = dataset.cardinality;
        reset = Some reset;
        spec = dataset.spec;
      }
  | Some dir ->
      (* File-based cache *)
      let _cache_file = Filename.concat dir "dataset_cache.bin" in
      (* Ensure directory exists *)
      (try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ());

      (* For now, fall back to in-memory cache *)
      (* A proper implementation would serialize/deserialize to disk *)
      cache dataset

let prefetch ?(buffer_size = 2) dataset =
  if buffer_size <= 0 then
    raise
      (Invalid_parameter
         ("prefetch: buffer_size must be positive, got "
        ^ string_of_int buffer_size));

  let buffer = Queue.create () in
  let mutex = Mutex.create () in
  let condition = Condition.create () in
  let finished = Atomic.make false in
  let cancelled = Atomic.make false in
  let producer_started = ref false in
  let producer_thread = ref None in
  let first_error : (exn * Printexc.raw_backtrace) option ref = ref None in

  let record_error exn bt =
    match !first_error with
    | Some _ -> ()
    | None ->
        first_error := Some (exn, bt);
        Atomic.set cancelled true;
        Condition.broadcast condition
  in

  let rec produce () =
    if Atomic.get cancelled then ()
    else
      let next_item =
        try dataset.next ()
        with exn ->
          let bt = Printexc.get_raw_backtrace () in
          Mutex.lock mutex;
          record_error exn bt;
          Mutex.unlock mutex;
          None
      in
      match next_item with
      | None ->
          Atomic.set finished true;
          Mutex.lock mutex;
          Condition.broadcast condition;
          Mutex.unlock mutex
      | Some value ->
          Mutex.lock mutex;
          while
            Queue.length buffer >= buffer_size && not (Atomic.get cancelled)
          do
            Condition.wait condition mutex
          done;
          if not (Atomic.get cancelled) then (
            Queue.add value buffer;
            Condition.signal condition);
          Mutex.unlock mutex;
          if not (Atomic.get cancelled) then produce ()
  in

  let join_producer () =
    match !producer_thread with
    | None -> ()
    | Some t -> (
        producer_thread := None;
        try Domain.join t
        with exn ->
          let bt = Printexc.get_raw_backtrace () in
          Mutex.lock mutex;
          record_error exn bt;
          Mutex.unlock mutex)
  in

  let stop_producer () =
    if !producer_started then (
      Atomic.set cancelled true;
      Mutex.lock mutex;
      Condition.broadcast condition;
      Mutex.unlock mutex;
      join_producer ();
      Atomic.set cancelled false;
      Atomic.set finished false;
      producer_started := false)
  in

  let start_producer () =
    if not !producer_started then (
      producer_started := true;
      Atomic.set cancelled false;
      Atomic.set finished false;
      let thread = Domain.spawn produce in
      producer_thread := Some thread)
  in

  let next () =
    start_producer ();
    Mutex.lock mutex;
    let rec wait () =
      if not (Queue.is_empty buffer) then (
        let value = Queue.take buffer in
        Condition.signal condition;
        `Value value)
      else
        match !first_error with
        | Some (exn, bt) -> `Error (exn, bt)
        | None ->
            if Atomic.get finished then `Finished
            else (
              Condition.wait condition mutex;
              wait ())
    in
    let outcome = wait () in
    Mutex.unlock mutex;
    match outcome with
    | `Value v -> Some v
    | `Finished ->
        stop_producer ();
        None
    | `Error (exn, bt) ->
        stop_producer ();
        Printexc.raise_with_backtrace exn bt
  in

  let reset =
    match dataset.reset with
    | None -> None
    | Some reset_fn ->
        Some
          (fun () ->
            stop_producer ();
            Mutex.lock mutex;
            Queue.clear buffer;
            first_error := None;
            Mutex.unlock mutex;
            reset_fn ())
  in

  { next; cardinality = dataset.cardinality; reset; spec = dataset.spec }

(* ───── Parallel Processing ───── *)

let parallel_map ?pool ?num_workers f dataset =
  let default_workers =
    let count = Domain.recommended_domain_count () in
    if count > 0 then count else 4
  in
  let num_workers = Option.value num_workers ~default:default_workers in
  if num_workers <= 1 then map f dataset
  else
    let num_workers = max 2 num_workers in

    (* Bounded work queue and result queue for backpressure *)
    let work_queue = Queue.create () in
    let work_mutex = Mutex.create () in
    let work_condition = Condition.create () in

    let result_queue = Queue.create () in
    let result_mutex = Mutex.create () in
    let result_condition = Condition.create () in

    let input_exhausted = ref false in
    let workers_done = Atomic.make 0 in
    let pool_holder = ref None in
    let internal_pool = ref None in
    let tasks_started = ref false in
    let error_state : (exn * Printexc.raw_backtrace) option ref = ref None in

    let push_error exn bt =
      Mutex.lock result_mutex;
      if Option.is_some !error_state then Mutex.unlock result_mutex
      else (
        error_state := Some (exn, bt);
        Queue.clear result_queue;
        Queue.add (Job_error (exn, bt)) result_queue;
        Condition.broadcast result_condition;
        Mutex.unlock result_mutex;
        input_exhausted := true;
        Mutex.lock work_mutex;
        Condition.broadcast work_condition;
        Mutex.unlock work_mutex)
    in

    (* Producer task - reads from dataset and fills work queue *)
    let producer pool_ref =
      Domainslib.Task.async pool_ref (fun () ->
          let rec produce () =
            if Option.is_some !error_state then ()
            else
              let next_item =
                try dataset.next ()
                with exn ->
                  let bt = Printexc.get_raw_backtrace () in
                  push_error exn bt;
                  None
              in
              match next_item with
              | None ->
                  input_exhausted := true;
                  Condition.broadcast work_condition
              | Some item ->
                  Mutex.lock work_mutex;
                  while
                    Queue.length work_queue >= num_workers * 4
                    && (not !input_exhausted)
                    && Option.is_none !error_state
                  do
                    Condition.wait work_condition work_mutex
                  done;
                  if (not !input_exhausted) && Option.is_none !error_state then (
                    Queue.add item work_queue;
                    Condition.signal work_condition);
                  Mutex.unlock work_mutex;
                  if (not !input_exhausted) && Option.is_none !error_state then
                    produce ()
          in
          produce ())
    in

    (* Worker tasks - process items from work queue *)
    let worker pool_ref _idx =
      Domainslib.Task.async pool_ref (fun () ->
          let rec process () =
            if Option.is_some !error_state then ()
            else (
              Mutex.lock work_mutex;
              while
                Queue.is_empty work_queue && (not !input_exhausted)
                && Option.is_none !error_state
              do
                Condition.wait work_condition work_mutex
              done;
              let item_opt =
                if Queue.is_empty work_queue then None
                else Some (Queue.take work_queue)
              in
              Condition.signal work_condition;
              Mutex.unlock work_mutex;

              match item_opt with
              | None ->
                  Atomic.incr workers_done;
                  Condition.broadcast result_condition
              | Some item -> (
                  match
                    try Some (f item)
                    with exn ->
                      let bt = Printexc.get_raw_backtrace () in
                      push_error exn bt;
                      None
                  with
                  | None ->
                      Atomic.incr workers_done;
                      Condition.broadcast result_condition
                  | Some result ->
                      Mutex.lock result_mutex;
                      Queue.add (Job_value result) result_queue;
                      Condition.signal result_condition;
                      Mutex.unlock result_mutex;
                      process ()))
          in
          process ())
    in

    let start_workers pool_inst =
      if not !tasks_started then (
        tasks_started := true;
        ignore (producer pool_inst);
        ignore (List.init (num_workers - 1) (fun i -> worker pool_inst i)))
    in

    let ensure_pool () =
      match !pool_holder with
      | Some p -> p
      | None -> (
          match pool with
          | Some external_pool ->
              pool_holder := Some external_pool;
              start_workers external_pool;
              external_pool
          | None ->
              let p =
                Domainslib.Task.setup_pool ~num_domains:(num_workers - 1) ()
              in
              internal_pool := Some p;
              pool_holder := Some p;
              start_workers p;
              p)
    in

    let cleanup () =
      match !internal_pool with
      | Some p ->
          Domainslib.Task.teardown_pool p;
          internal_pool := None;
          pool_holder := None;
          tasks_started := false
      | None -> ()
    in

    let next () =
      (match !error_state with
      | Some (exn, bt) -> Printexc.raise_with_backtrace exn bt
      | None -> ());
      let _ = ensure_pool () in
      Mutex.lock result_mutex;
      while
        Queue.is_empty result_queue
        && ((not !input_exhausted) || Atomic.get workers_done < num_workers - 1)
      do
        Condition.wait result_condition result_mutex
      done;
      let result =
        if Queue.is_empty result_queue then None
        else Some (Queue.take result_queue)
      in
      Mutex.unlock result_mutex;
      match result with
      | Some (Job_error (exn, bt)) ->
          cleanup ();
          Printexc.raise_with_backtrace exn bt
      | Some (Job_value v) -> Some v
      | None ->
          cleanup ();
          None
    in

    {
      next;
      cardinality = dataset.cardinality;
      reset = None;
      spec = (fun () -> Unknown);
    }

let parallel_interleave ?num_workers ?block_length f dataset =
  let num_workers = Option.value num_workers ~default:4 in
  let block_length = Option.value block_length ~default:1 in

  (* Queue of sub-datasets with their remaining block counts *)
  let active_datasets = Queue.create () in
  let datasets_mutex = Mutex.create () in
  let datasets_condition = Condition.create () in

  let input_exhausted = ref false in
  let total_active = ref 0 in
  let pool_created = ref false in
  let pool = ref None in

  (* Background task to create sub-datasets *)
  let dataset_creator pool_ref =
    Domainslib.Task.async pool_ref (fun () ->
        let rec create () =
          if !total_active < num_workers then (
            match dataset.next () with
            | None ->
                input_exhausted := true;
                Condition.broadcast datasets_condition
            | Some x ->
                let sub = f x in
                Mutex.lock datasets_mutex;
                Queue.add (sub, ref block_length) active_datasets;
                incr total_active;
                Condition.signal datasets_condition;
                Mutex.unlock datasets_mutex;
                create ())
          else (
            (* Wait until we need more datasets *)
            Mutex.lock datasets_mutex;
            while !total_active >= num_workers && not !input_exhausted do
              Condition.wait datasets_condition datasets_mutex
            done;
            Mutex.unlock datasets_mutex;
            if not !input_exhausted then create ())
        in
        create ())
  in

  let ensure_pool () =
    if not !pool_created then (
      pool_created := true;
      let p = Domainslib.Task.setup_pool ~num_domains:1 () in
      pool := Some p;

      (* Start background dataset creator *)
      let _ = dataset_creator p in
      p)
    else
      match !pool with
      | Some p -> p
      | None -> failwith "Pool was cleaned up unexpectedly"
  in

  let cleanup () =
    input_exhausted := true;
    Condition.broadcast datasets_condition;
    match !pool with
    | Some p ->
        Domainslib.Task.teardown_pool p;
        pool := None;
        pool_created := false
    | None -> ()
  in

  let next () =
    (* Ensure pool is created *)
    let _ = ensure_pool () in

    let rec try_next () =
      Mutex.lock datasets_mutex;

      (* Wait for available datasets *)
      while Queue.is_empty active_datasets && not !input_exhausted do
        Condition.wait datasets_condition datasets_mutex
      done;

      if Queue.is_empty active_datasets && !input_exhausted then (
        Mutex.unlock datasets_mutex;
        cleanup ();
        None)
      else if not (Queue.is_empty active_datasets) then (
        (* Get next dataset from front of queue *)
        let sub_dataset, remaining_ref = Queue.take active_datasets in
        Mutex.unlock datasets_mutex;

        (* Try to get element from this dataset *)
        match sub_dataset.next () with
        | None ->
            (* This dataset is exhausted *)
            Mutex.lock datasets_mutex;
            decr total_active;
            Condition.signal datasets_condition;
            Mutex.unlock datasets_mutex;
            try_next ()
        | Some elem ->
            (* Got an element *)
            decr remaining_ref;

            (* Re-queue if not done with block *)
            if !remaining_ref > 0 then (
              Mutex.lock datasets_mutex;
              Queue.add (sub_dataset, remaining_ref) active_datasets;
              Mutex.unlock datasets_mutex)
            else (
              (* Done with this block, rotate to next dataset *)
              remaining_ref := block_length;
              Mutex.lock datasets_mutex;
              Queue.add (sub_dataset, remaining_ref) active_datasets;
              Mutex.unlock datasets_mutex);

            Some elem)
      else (
        Mutex.unlock datasets_mutex;
        None)
    in
    try_next ()
  in

  {
    next;
    cardinality = (fun () -> Unknown);
    reset = None;
    spec = (fun () -> Unknown);
  }

(* ───── Iteration ───── *)

let iter f dataset =
  let continue = ref true in
  while !continue do
    match dataset.next () with None -> continue := false | Some x -> f x
  done

let fold f init dataset =
  let acc = ref init in
  let continue = ref true in
  while !continue do
    match dataset.next () with
    | None -> continue := false
    | Some x -> acc := f !acc x
  done;
  !acc

let to_seq dataset =
  let rec make () =
    match dataset.next () with None -> Seq.Nil | Some x -> Seq.Cons (x, make)
  in
  make

let to_list dataset = List.rev (fold (fun acc x -> x :: acc) [] dataset)
let to_array dataset = Array.of_list (to_list dataset)

(* ───── Dataset Information ───── *)

let cardinality dataset = dataset.cardinality ()
let element_spec dataset = dataset.spec ()

(* ───── Dataset Control ───── *)

let reset dataset = match dataset.reset with Some f -> f () | None -> ()

(* ───── Common Pipelines ───── *)

let text_classification_pipeline ?tokenizer ?max_length ?(batch_size = 32)
    ?(shuffle_buffer = 10000) ?num_workers text_dataset =
  let tok = Option.value tokenizer ~default:whitespace_tokenizer in
  (* Use Dynamic padding when batching to ensure all sequences in a batch have
     same length *)
  let tokenized = tokenize tok ?max_length ~padding:`Dynamic text_dataset in
  let shuffled = shuffle ~buffer_size:shuffle_buffer tokenized in
  let batched = batch_generic batch_size shuffled in
  (* Convert int arrays to int32 tensors *)
  let tensor_batched =
    map
      (fun batch ->
        (* Find max length in this batch for padding *)
        let max_len =
          Array.fold_left (fun acc arr -> max acc (Array.length arr)) 0 batch
        in
        let tensors =
          Array.map
            (fun arr ->
              (* Pad to max length if needed *)
              let padded_arr =
                if Array.length arr < max_len then
                  Array.append arr (Array.make (max_len - Array.length arr) 0)
                else arr
              in
              let int32_arr = Array.map Int32.of_int padded_arr in
              Rune.create Rune.int32 [| max_len |] int32_arr)
            batch
        in
        Rune.stack ~axis:0 (Array.to_list tensors))
      batched
  in
  let prefetched = prefetch ~buffer_size:2 tensor_batched in
  match num_workers with
  | None -> prefetched
  | Some n -> parallel_map ~num_workers:n (fun x -> x) prefetched

let language_model_pipeline ?tokenizer ?(sequence_length = 512)
    ?(batch_size = 32) ?(shuffle_buffer = 10000) ?num_workers text_dataset =
  let tok = Option.value tokenizer ~default:whitespace_tokenizer in
  (* Use padding to ensure consistent length *)
  let tokenized =
    tokenize tok ~max_length:sequence_length ~padding:(`Max sequence_length)
      text_dataset
  in
  let paired =
    map
      (fun tokens ->
        (* Create input/target pairs for language modeling *)
        if Array.length tokens > 1 then
          let input = Array.sub tokens 0 (Array.length tokens - 1) in
          let target = Array.sub tokens 1 (Array.length tokens - 1) in
          (input, target)
        else (tokens, tokens) (* Fallback for single token sequences *))
      tokenized
  in
  let shuffled = shuffle ~buffer_size:shuffle_buffer paired in
  let batched = batch_generic batch_size shuffled in
  (* Convert int array pairs to int32 tensor pairs *)
  let tensor_batched =
    map
      (fun batch ->
        let inputs, targets = Array.split batch in
        (* All sequences should have same length due to padding, but let's be
           safe *)
        let max_len =
          Array.fold_left (fun acc arr -> max acc (Array.length arr)) 0 inputs
        in
        let input_tensors =
          Array.map
            (fun arr ->
              let padded_arr =
                if Array.length arr < max_len then
                  Array.append arr (Array.make (max_len - Array.length arr) 0)
                else arr
              in
              let int32_arr = Array.map Int32.of_int padded_arr in
              Rune.create Rune.int32 [| max_len |] int32_arr)
            inputs
        in
        let target_tensors =
          Array.map
            (fun arr ->
              let padded_arr =
                if Array.length arr < max_len then
                  Array.append arr (Array.make (max_len - Array.length arr) 0)
                else arr
              in
              let int32_arr = Array.map Int32.of_int padded_arr in
              Rune.create Rune.int32 [| max_len |] int32_arr)
            targets
        in
        ( Rune.stack ~axis:0 (Array.to_list input_tensors),
          Rune.stack ~axis:0 (Array.to_list target_tensors) ))
      batched
  in
  let prefetched = prefetch ~buffer_size:2 tensor_batched in
  match num_workers with
  | None -> prefetched
  | Some n -> parallel_map ~num_workers:n (fun x -> x) prefetched

(* High-level pipeline function for tensor datasets *)
let prepare ?shuffle_buffer ?batch_size ?prefetch:prefetch_count
    ?cache:cache_enabled ?drop_remainder dataset =
  let dataset =
    match cache_enabled with Some true -> cache dataset | _ -> dataset
  in
  let dataset =
    match shuffle_buffer with
    | Some buffer_size -> shuffle ~buffer_size dataset
    | None -> dataset
  in
  let dataset =
    match batch_size with
    | Some size -> batch ?drop_remainder size dataset
    | None -> dataset
  in
  let dataset =
    match prefetch_count with
    | Some buffer_size -> prefetch ~buffer_size dataset
    | None -> dataset
  in
  dataset
