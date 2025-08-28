(** Implementation of efficient dataset handling *)

(* Exception types for dataset errors *)
exception Dataset_error of string
exception File_not_found of string
exception Invalid_parameter of string

type cardinality = Finite of int | Unknown | Infinite

type element_spec =
  | Unknown
  | Scalar of string  (** e.g., "string" or "int" *)
  | Tensor of int array * string  (** shape * dtype *)
  | Tuple of element_spec list
  | Array of element_spec

(* Core dataset type - lazy sequence with metadata *)
type 'a t = {
  next : unit -> 'a option; (* Get next element *)
  cardinality : unit -> cardinality; (* Dataset cardinality *)
  reset : (unit -> unit) option; (* Reset to beginning if supported *)
  spec : unit -> element_spec; (* Element type specification *)
}

type ('elt, 'kind, 'dev) tensor_dataset = ('elt, 'kind, 'dev) Rune.t t
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
    let bytes = Bytes.create actual_length in
    for i = 0 to actual_length - 1 do
      Bytes.set bytes i (Bigarray.Array1.get handle.data (offset + i))
    done;
    Bytes.to_string bytes

(** {1 Dataset Creation} *)

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
  let n = (Rune.shape tensor).(0) in
  let idx = ref 0 in
  let reset () = idx := 0 in
  let shape = Rune.shape tensor in
  let slice_shape = Array.sub shape 1 (Array.length shape - 1) in
  {
    next =
      (fun () ->
        if !idx < n then (
          let slice = Rune.slice [ R [ !idx ] ] tensor in
          incr idx;
          Some slice)
        else None);
    cardinality = (fun () -> Finite n);
    reset = Some reset;
    spec = (fun () -> Tensor (slice_shape, "float32"));
  }

let from_tensors (x, y) =
  let n = (Rune.shape x).(0) in
  let idx = ref 0 in
  let reset () = idx := 0 in
  let x_shape = Rune.shape x in
  let y_shape = Rune.shape y in
  let x_slice_shape = Array.sub x_shape 1 (Array.length x_shape - 1) in
  let y_slice_shape = Array.sub y_shape 1 (Array.length y_shape - 1) in
  {
    next =
      (fun () ->
        if !idx < n then (
          let x_slice = Rune.slice [ R [ !idx ] ] x in
          let y_slice = Rune.slice [ R [ !idx ] ] y in
          incr idx;
          Some (x_slice, y_slice))
        else None);
    cardinality = (fun () -> Finite n);
    reset = Some reset;
    spec =
      (fun () ->
        Tuple
          [
            Tensor (x_slice_shape, "float32"); Tensor (y_slice_shape, "float32");
          ]);
  }

(* Text Data Sources *)

let from_text_file ?encoding ?(chunk_size = 65536) path =
  let _ = encoding in
  (* TODO: Handle different encodings *)
  let handle = create_mmap path in
  let offset = ref 0 in
  let buffer = ref "" in
  let buffer_pos = ref 0 in
  let closed = ref false in

  let rec next_line () =
    if !closed then None
    else
      (* Look for newline in buffer *)
      try
        let nl_pos = String.index_from !buffer !buffer_pos '\n' in
        let line = String.sub !buffer !buffer_pos (nl_pos - !buffer_pos) in
        buffer_pos := nl_pos + 1;
        Some line
      with Not_found ->
        (* Need more data *)
        if !offset >= handle.size then
          (* End of file - return remaining buffer if any *)
          if !buffer_pos < String.length !buffer then (
            let line =
              String.sub !buffer !buffer_pos
                (String.length !buffer - !buffer_pos)
            in
            buffer := "";
            buffer_pos := 0;
            Some line)
          else (
            close_mmap handle;
            closed := true;
            None)
        else
          (* Read next chunk *)
          let chunk =
            read_mmap_chunk handle ~offset:!offset ~length:chunk_size
          in
          offset := !offset + String.length chunk;

          (* Append to remaining buffer *)
          if !buffer_pos < String.length !buffer then
            buffer :=
              String.sub !buffer !buffer_pos
                (String.length !buffer - !buffer_pos)
              ^ chunk
          else buffer := chunk;
          buffer_pos := 0;
          next_line ()
  in

  let reset () =
    offset := 0;
    buffer := "";
    buffer_pos := 0;
    closed := false
  in

  {
    next = next_line;
    cardinality = (fun () -> Unknown);
    reset = Some reset;
    spec = (fun () -> Scalar "string");
  }

let from_text_files ?(encoding = `UTF8) ?(chunk_size = 65536) paths =
  let current_file = ref 0 in
  let current_dataset = ref None in

  let rec next () =
    match !current_dataset with
    | None ->
        if !current_file >= List.length paths then None
        else
          let path = List.nth paths !current_file in
          current_dataset := Some (from_text_file ~encoding ~chunk_size path);
          incr current_file;
          next ()
    | Some ds -> (
        match ds.next () with
        | None ->
            current_dataset := None;
            next ()
        | some_line -> some_line)
  in

  {
    next;
    cardinality = (fun () -> Unknown);
    reset = None;
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

let from_csv ?(separator = ',') ?(text_column = 0) ?label_column
    ?(has_header = true) path =
  let _ = label_column in
  (* TODO: Handle label column *)
  let text_ds = from_text_file path in
  let skipped_header = ref (not has_header) in

  let split_csv line =
    (* Simple CSV split - doesn't handle quoted fields *)
    String.split_on_char separator line
  in

  let next () =
    if not !skipped_header then (
      skipped_header := true;
      ignore (text_ds.next ()));

    match text_ds.next () with
    | None -> None
    | Some line ->
        let fields = split_csv line in
        if text_column < List.length fields then
          Some (List.nth fields text_column)
        else None
  in

  {
    next;
    cardinality = text_ds.cardinality;
    reset = text_ds.reset;
    spec = (fun () -> Scalar "string");
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

(** {1 Transformations} *)

let map f dataset =
  {
    next = (fun () -> Option.map f (dataset.next ()));
    cardinality = dataset.cardinality;
    reset = dataset.reset;
    spec = (fun () -> Unknown);
    (* Can't infer spec from arbitrary map *)
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

  { next; cardinality; reset = None; spec = ds1.spec }

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

(* Text Processing *)

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

(* Batching *)

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
    else if List.length !buffer < size && drop_remainder then None
    else
      let batch = Array.of_list (List.rev !buffer) in
      buffer := [];
      Some batch
  in
  {
    next;
    cardinality = (fun () -> Unknown);
    reset = None;
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

let bucket_by_length ?boundaries ?batch_sizes length_fn dataset =
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
        (fun _bucket_idx bucket ->
          if (not (Queue.is_empty bucket)) && not !found_batch then (
            let batch =
              Array.init (Queue.length bucket) (fun _ -> Queue.take bucket)
            in
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

(* Iteration Control *)

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

  {
    next;
    cardinality = (fun () -> Finite n);
    reset = None;
    spec = dataset.spec;
  }

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

  let buffer = Queue.create () in
  let exhausted = ref false in

  (* Lazy window generation *)
  let next () =
    (* Ensure we have enough elements for a window or we're exhausted *)
    while Queue.length buffer < size && not !exhausted do
      match dataset.next () with
      | None -> exhausted := true
      | Some x -> Queue.add x buffer
    done;

    (* Check if we can make a window *)
    let available = Queue.length buffer in
    if available = 0 then None
    else if available < size && drop_remainder then None
    else
      (* Create window array *)
      let window_size = min size available in
      let window = Array.make window_size (Obj.magic 0) in

      (* Copy elements from buffer to window *)
      let buffer_list =
        Queue.fold (fun acc x -> x :: acc) [] buffer |> List.rev
      in
      for i = 0 to window_size - 1 do
        if i < List.length buffer_list then window.(i) <- List.nth buffer_list i
      done;

      (* Shift buffer for next window *)
      for _ = 1 to min actual_shift available do
        ignore (Queue.take buffer)
      done;

      Some window
  in

  {
    next;
    cardinality = (fun () -> Unknown);
    reset = None;
    spec = (fun () -> Array (dataset.spec ()));
  }

(* Shuffling and Sampling *)

let shuffle ?rng ?(buffer_size = 10000) dataset =
  if buffer_size <= 0 then
    raise
      (Invalid_parameter
         ("shuffle: buffer_size must be positive, got "
        ^ string_of_int buffer_size));

  let buffer = Array.make buffer_size None in
  let buffer_count = ref 0 in
  (* Use provided RNG or create new one *)
  let rand_state =
    match rng with
    | Some key ->
        (* Convert Rune.Rng.key to Random.State.t using to_int as seed *)
        Random.State.make [| Rune.Rng.to_int key |]
    | None -> Random.State.make_self_init ()
  in

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
      let idx = Random.State.int rand_state !buffer_count in
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

  { next; cardinality = dataset.cardinality; reset = None; spec = dataset.spec }

let sample ?rng ?(replacement = false) n dataset =
  if replacement then
    (* For sampling with replacement, we need the full dataset *)
    (* This is inherently eager, but we can make it lazy by deferring collection *)
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
            (* Use provided RNG if available *)
            let rand_state =
              match rng with
              | Some key ->
                  (* Convert Rune.Rng.key to Random.State.t *)
                  Random.State.make [| Rune.Rng.to_int key |]
              | None -> Random.State.make_self_init ()
            in
            incr count;
            Some arr.(Random.State.int rand_state len))
    in
    {
      next;
      cardinality = (fun () -> Finite n);
      reset = None;
      spec = dataset.spec;
    }
  else
    (* Sample without replacement by shuffling and taking first n *)
    take n (shuffle ?rng ~buffer_size:(min 10000 n) dataset)

let weighted_sample ?rng ~weights n dataset =
  (* Use provided RNG or create new one *)
  let rand_state =
    match rng with
    | Some key -> Random.State.make [| Rune.Rng.to_int key |]
    | None -> Random.State.make_self_init ()
  in

  (* Compute cumulative weights *)
  let total_weight = Array.fold_left ( +. ) 0. weights in
  let cumulative = Array.make (Array.length weights) 0. in
  let sum = ref 0. in
  Array.iteri
    (fun i w ->
      sum := !sum +. w;
      cumulative.(i) <- !sum /. total_weight)
    weights;

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
          let r = Random.State.float rand_state 1.0 in
          (* Binary search for the right bucket *)
          let idx =
            let rec find_bucket low high =
              if low >= high then low
              else
                let mid = (low + high) / 2 in
                if cumulative.(mid) < r then find_bucket (mid + 1) high
                else find_bucket low mid
            in
            find_bucket 0
              (min (Array.length cumulative - 1) (Array.length arr - 1))
          in
          incr count;
          if idx < Array.length arr then Some arr.(idx) else None)
  in

  {
    next;
    cardinality = (fun () -> Finite n);
    reset = None;
    spec = dataset.spec;
  }

(* Caching and Prefetching *)

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
  (* Create a buffer for prefetched elements *)
  let buffer = Queue.create () in
  let fetching = ref false in
  let exhausted = ref false in

  let fetch_to_buffer () =
    if (not !fetching) && (not !exhausted) && Queue.length buffer < buffer_size
    then (
      fetching := true;
      match dataset.next () with
      | None ->
          exhausted := true;
          fetching := false
      | Some x ->
          Queue.add x buffer;
          fetching := false)
  in

  let next () =
    (* Try to maintain buffer *)
    while Queue.length buffer < buffer_size && not !exhausted do
      fetch_to_buffer ()
    done;

    if Queue.is_empty buffer then
      if !exhausted then None
      else (
        fetch_to_buffer ();
        if Queue.is_empty buffer then None else Some (Queue.take buffer))
    else Some (Queue.take buffer)
  in

  {
    next;
    cardinality = dataset.cardinality;
    reset = dataset.reset;
    spec = dataset.spec;
  }

(* Parallel Processing *)

let parallel_map ?num_workers f dataset =
  let num_workers = Option.value num_workers ~default:4 in

  (* Bounded work queue and result queue for backpressure *)
  let work_queue = Queue.create () in
  let work_mutex = Mutex.create () in
  let work_condition = Condition.create () in

  let result_queue = Queue.create () in
  let result_mutex = Mutex.create () in
  let result_condition = Condition.create () in

  let input_exhausted = ref false in
  let workers_done = Atomic.make 0 in
  let pool_created = ref false in
  let pool = ref None in

  (* Producer task - reads from dataset and fills work queue *)
  let producer pool_ref =
    Domainslib.Task.async pool_ref (fun () ->
        let rec produce () =
          match dataset.next () with
          | None ->
              input_exhausted := true;
              (* Wake up all workers so they can exit *)
              Condition.broadcast work_condition
          | Some item ->
              (* Add to work queue with backpressure *)
              Mutex.lock work_mutex;
              while
                Queue.length work_queue >= num_workers * 4
                && not !input_exhausted
              do
                Condition.wait work_condition work_mutex
              done;
              if not !input_exhausted then (
                Queue.add item work_queue;
                Condition.signal work_condition);
              Mutex.unlock work_mutex;
              if not !input_exhausted then produce ()
        in
        produce ())
  in

  (* Worker tasks - process items from work queue *)
  let worker pool_ref _id =
    Domainslib.Task.async pool_ref (fun () ->
        let rec process () =
          (* Get work item *)
          Mutex.lock work_mutex;
          while Queue.is_empty work_queue && not !input_exhausted do
            Condition.wait work_condition work_mutex
          done;
          let item_opt =
            if Queue.is_empty work_queue then None
            else Some (Queue.take work_queue)
          in
          Condition.signal work_condition;
          (* Signal producer if waiting *)
          Mutex.unlock work_mutex;

          match item_opt with
          | None ->
              (* No more work *)
              Atomic.incr workers_done;
              Condition.broadcast result_condition
          | Some item ->
              (* Process item *)
              let result = f item in

              (* Add result to queue *)
              Mutex.lock result_mutex;
              Queue.add result result_queue;
              Condition.signal result_condition;
              Mutex.unlock result_mutex;

              process ()
        in
        process ())
  in

  let ensure_pool () =
    if not !pool_created then (
      pool_created := true;
      let p = Domainslib.Task.setup_pool ~num_domains:(num_workers - 1) () in
      pool := Some p;

      (* Start producer and workers *)
      let _ = producer p in
      let _ = List.init (num_workers - 1) (fun i -> worker p i) in

      (* Store pool for cleanup *)
      pool := Some p;
      p)
    else
      match !pool with
      | Some p -> p
      | None -> failwith "Pool was cleaned up unexpectedly"
  in

  let cleanup () =
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

    (* Try to get a result *)
    Mutex.lock result_mutex;

    (* Wait for result or completion *)
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

    (* Cleanup if done *)
    if result = None then cleanup ();

    result
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

(** {1 Iteration} *)

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

(** {1 Dataset Information} *)

let cardinality dataset = dataset.cardinality ()
let element_spec dataset = dataset.spec ()

(** {1 Common Pipelines} *)

let text_classification_pipeline ?tokenizer ?max_length ?(batch_size = 32)
    ?(shuffle_buffer = 10000) ?num_workers ~device text_dataset =
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
              Rune.create device Rune.int32 [| max_len |] int32_arr)
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
    ?(batch_size = 32) ?(shuffle_buffer = 10000) ?num_workers ~device
    text_dataset =
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
              Rune.create device Rune.int32 [| max_len |] int32_arr)
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
              Rune.create device Rune.int32 [| max_len |] int32_arr)
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
