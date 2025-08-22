open Rune

type metadata = (string * string) list
type checkpoint_info = { step : int; timestamp : float; metadata : metadata }
type format = Safetensors

let default_format = Safetensors

let infer_format_from_path path =
  if Filename.check_suffix path ".safetensors" then Safetensors
  else default_format

(* Helper to get flattened tensors with names *)
let get_named_tensors : type layout dev.
    (layout, dev) Ptree.t -> (string * (float, layout, dev) t) list =
 fun params -> Ptree.flatten_with_paths params

(* Convert Rune tensor to safetensor view *)
let tensor_to_safetensor_view : type layout dev.
    (float, layout, dev) t -> Safetensors.tensor_view =
 fun tensor ->
  let shape_array = shape tensor in
  let shape = Array.to_list shape_array in
  let data_array =
    unsafe_to_array (reshape [| Array.fold_left ( * ) 1 shape_array |] tensor)
  in

  (* Convert float array to bytes *)
  let n_elements = Array.length data_array in
  let bytes = Bytes.create (n_elements * 4) in
  (* F32 = 4 bytes *)
  Array.iteri
    (fun i f ->
      let bits = Int32.bits_of_float f in
      Bytes.set bytes (i * 4)
        (Char.chr (Int32.to_int (Int32.logand bits 0xFFl)));
      Bytes.set bytes
        ((i * 4) + 1)
        (Char.chr
           (Int32.to_int (Int32.logand (Int32.shift_right bits 8) 0xFFl)));
      Bytes.set bytes
        ((i * 4) + 2)
        (Char.chr
           (Int32.to_int (Int32.logand (Int32.shift_right bits 16) 0xFFl)));
      Bytes.set bytes
        ((i * 4) + 3)
        (Char.chr
           (Int32.to_int (Int32.logand (Int32.shift_right bits 24) 0xFFl))))
    data_array;

  let data = Bytes.to_string bytes in
  match Safetensors.tensor_view_new ~dtype:F32 ~shape ~data with
  | Ok view -> view
  | Error e ->
      failwith ("Failed to create tensor view: " ^ Safetensors.string_of_error e)

(* Convert safetensor view to Rune tensor *)
let safetensor_view_to_tensor : type layout dev.
    Safetensors.tensor_view ->
    device:dev device ->
    dtype:(float, layout) dtype ->
    (float, layout, dev) t =
 fun view ~device ~dtype ->
  if view.dtype <> F32 then
    failwith
      (Printf.sprintf "Unsupported dtype: expected F32, got %s"
         (Safetensors.dtype_to_string view.dtype));

  let shape = Array.of_list view.shape in
  let n_elements = Array.fold_left ( * ) 1 shape in
  let data_array = Array.make n_elements 0.0 in

  (* Extract data from view *)
  let bytes = String.sub view.data view.offset view.length in

  (* Convert bytes to float array *)
  for i = 0 to n_elements - 1 do
    let offset = i * 4 in
    let b0 = Int32.of_int (Char.code bytes.[offset]) in
    let b1 = Int32.shift_left (Int32.of_int (Char.code bytes.[offset + 1])) 8 in
    let b2 =
      Int32.shift_left (Int32.of_int (Char.code bytes.[offset + 2])) 16
    in
    let b3 =
      Int32.shift_left (Int32.of_int (Char.code bytes.[offset + 3])) 24
    in
    let bits = Int32.logor b0 (Int32.logor b1 (Int32.logor b2 b3)) in
    data_array.(i) <- Int32.float_of_bits bits
  done;

  create device dtype shape data_array

module Checkpointer = struct
  type t = { format : format }

  let create ?(format = default_format) () = { format }

  let save_safetensors ~path ~params ~metadata =
    let named_tensors = get_named_tensors params in
    let tensor_views =
      List.map
        (fun (name, tensor) -> (name, tensor_to_safetensor_view tensor))
        named_tensors
    in
    (* Path-based keys are self-describing, no need for structure metadata *)
    match Safetensors.serialize tensor_views (Some metadata) with
    | Ok data ->
        let oc = open_out_bin path in
        output_string oc data;
        close_out oc
    | Error e ->
        failwith
          ("Failed to serialize safetensors: " ^ Safetensors.string_of_error e)

  let restore_safetensors ~path ~device ~dtype =
    let ic = open_in_bin path in
    let len = in_channel_length ic in
    let buffer = Bytes.create len in
    really_input ic buffer 0 len;
    close_in ic;
    let buffer_str = Bytes.to_string buffer in

    match Safetensors.deserialize buffer_str with
    | Ok st ->
        (* Get tensors and reconstruct tree from paths *)
        let tensor_list = Safetensors.tensors st in
        let path_tensor_pairs =
          List.map
            (fun (name, view) ->
              (name, safetensor_view_to_tensor view ~device ~dtype))
            tensor_list
        in
        Ptree.unflatten_from_paths path_tensor_pairs
    | Error e ->
        failwith
          ("Failed to deserialize safetensors: " ^ Safetensors.string_of_error e)

  let save t ~path ~params ?(metadata = []) () =
    (* Create directory recursively if it doesn't exist *)
    let rec mkdir_p path =
      if not (Sys.file_exists path) then (
        mkdir_p (Filename.dirname path);
        try Unix.mkdir path 0o755
        with Unix.Unix_error (Unix.EEXIST, _, _) -> ())
    in
    mkdir_p path;

    (* Save checkpoint info *)
    let info_json =
      `Assoc
        [
          ("timestamp", `Float (Unix.time ()));
          ("metadata", `Assoc (List.map (fun (k, v) -> (k, `String v)) metadata));
        ]
    in
    let info_path = Filename.concat path "checkpoint.json" in
    let oc = open_out info_path in
    Yojson.Basic.to_channel oc info_json;
    close_out oc;

    (* Save params based on format *)
    let params_filename =
      match t.format with Safetensors -> "params.safetensors"
    in
    let params_path = Filename.concat path params_filename in
    match t.format with
    | Safetensors -> save_safetensors ~path:params_path ~params ~metadata

  let restore _t ~path ~device ~dtype =
    let safetensors_path = Filename.concat path "params.safetensors" in

    if Sys.file_exists safetensors_path then
      restore_safetensors ~path:safetensors_path ~device ~dtype
    else failwith "No checkpoint found in directory"

  let save_file _t ~path ~params ?(metadata = []) () =
    let format = infer_format_from_path path in
    match format with Safetensors -> save_safetensors ~path ~params ~metadata

  let restore_file _t ~path ~device ~dtype =
    let format = infer_format_from_path path in
    match format with Safetensors -> restore_safetensors ~path ~device ~dtype
end

module CheckpointManager = struct
  type options = {
    max_to_keep : int option;
    keep_checkpoint_every_n_steps : int option;
    best_fn : (checkpoint_info -> float) option;
    best_mode : [ `min | `max ];
  }

  type t = {
    directory : string;
    options : options;
    checkpointer : Checkpointer.t;
    mutable checkpoints : (int * checkpoint_info) list;
    mutable best_checkpoint : (int * float) option;
  }

  let default_options =
    {
      max_to_keep = Some 5;
      keep_checkpoint_every_n_steps = None;
      best_fn = None;
      best_mode = `max;
    }

  let checkpoint_dir t step =
    Filename.concat t.directory (Printf.sprintf "ckpt-%d" step)

  let load_checkpoint_info path =
    let info_path = Filename.concat path "checkpoint.json" in
    if Sys.file_exists info_path then (
      let ic = open_in info_path in
      let json = Yojson.Basic.from_channel ic in
      close_in ic;
      match json with
      | `Assoc fields ->
          let timestamp =
            match List.assoc_opt "timestamp" fields with
            | Some (`Float t) -> t
            | _ -> 0.
          in
          let metadata =
            match List.assoc_opt "metadata" fields with
            | Some (`Assoc meta) ->
                List.map
                  (fun (k, v) ->
                    match v with `String s -> (k, s) | _ -> (k, ""))
                  meta
            | _ -> []
          in
          Some { step = 0; timestamp; metadata }
      | _ -> None)
    else None

  let scan_checkpoints directory =
    try
      let entries = Sys.readdir directory in
      let checkpoints =
        Array.to_list entries
        |> List.filter_map (fun entry ->
               if String.length entry > 5 && String.sub entry 0 5 = "ckpt-" then
                 try
                   let step =
                     int_of_string
                       (String.sub entry 5 (String.length entry - 5))
                   in
                   let path = Filename.concat directory entry in
                   match load_checkpoint_info path with
                   | Some info -> Some (step, { info with step })
                   | None -> None
                 with _ -> None
               else None)
      in
      List.sort (fun (s1, _) (s2, _) -> compare s1 s2) checkpoints
    with _ -> []

  let create ~directory ?(options = default_options)
      ?(checkpointer = Checkpointer.create ()) () =
    (* Create directory if it doesn't exist *)
    (try Unix.mkdir directory 0o755
     with Unix.Unix_error (Unix.EEXIST, _, _) -> ());

    let checkpoints = scan_checkpoints directory in
    { directory; options; checkpointer; checkpoints; best_checkpoint = None }

  let should_keep t step =
    match t.options.keep_checkpoint_every_n_steps with
    | Some n when step mod n = 0 -> true
    | _ -> false

  let cleanup t =
    match t.options.max_to_keep with
    | None -> ()
    | Some max_to_keep ->
        (* Get checkpoints that aren't explicitly marked to keep *)
        let deletable =
          List.filter (fun (step, _) -> not (should_keep t step)) t.checkpoints
        in
        (* If we have more than max_to_keep total, delete the oldest ones *)
        let to_delete =
          if List.length deletable > max_to_keep then
            (* Sort by step (ascending) and take all but the last max_to_keep *)
            let sorted =
              List.sort (fun (s1, _) (s2, _) -> compare s1 s2) deletable
            in
            let n_to_delete = List.length sorted - max_to_keep in
            if n_to_delete > 0 then
              (* Take the first n_to_delete items *)
              let rec take n lst =
                match (n, lst) with
                | 0, _ | _, [] -> []
                | n, h :: t -> h :: take (n - 1) t
              in
              take n_to_delete sorted
            else []
          else []
        in
        List.iter
          (fun (step, _) ->
            let path = checkpoint_dir t step in
            (* Recursively delete directory *)
            let rec rm_rf path =
              if Sys.is_directory path then (
                let entries = Sys.readdir path in
                Array.iter
                  (fun entry -> rm_rf (Filename.concat path entry))
                  entries;
                Unix.rmdir path)
              else Sys.remove path
            in
            (try rm_rf path with _ -> ());
            t.checkpoints <- List.filter (fun (s, _) -> s <> step) t.checkpoints)
          to_delete

  let save t ~step ~params ?(metadata = []) ?(metrics = []) () =
    let path = checkpoint_dir t step in
    let info = { step; timestamp = Unix.time (); metadata } in

    (* Save checkpoint *)
    Checkpointer.save t.checkpointer ~path ~params ~metadata ();

    (* Update checkpoint list *)
    t.checkpoints <- t.checkpoints @ [ (step, info) ];

    (* Update best checkpoint if needed *)
    (match t.options.best_fn with
    | Some best_fn ->
        let score = best_fn info in
        let is_better =
          match t.best_checkpoint with
          | None -> true
          | Some (_, best_score) -> (
              match t.options.best_mode with
              | `max -> score > best_score
              | `min -> score < best_score)
        in
        if is_better then t.best_checkpoint <- Some (step, score)
    | None ->
        (* If metrics provided, store them in metadata *)
        if metrics <> [] then
          let metadata =
            metadata @ List.map (fun (k, v) -> (k, string_of_float v)) metrics
          in
          let _path = checkpoint_dir t step in
          let info = { step; timestamp = Unix.time (); metadata } in
          t.checkpoints <-
            List.filter (fun (s, _) -> s <> step) t.checkpoints
            @ [ (step, info) ]);

    (* Cleanup old checkpoints *)
    cleanup t

  let restore t ~device ~dtype ?step () =
    let step =
      match step with
      | Some s -> s
      | None -> (
          match List.rev t.checkpoints with
          | (s, _) :: _ -> s
          | [] -> failwith "No checkpoints available")
    in
    let info = List.assoc step t.checkpoints in
    let path = checkpoint_dir t step in
    let params = Checkpointer.restore t.checkpointer ~path ~device ~dtype in
    (params, info)

  let restore_best t ~device ~dtype =
    match t.best_checkpoint with
    | Some (step, _) -> restore t ~device ~dtype ~step ()
    | None -> failwith "No best checkpoint available (best_fn not configured)"

  let latest_step t =
    match List.rev t.checkpoints with (step, _) :: _ -> Some step | [] -> None

  let best_step t =
    match t.best_checkpoint with Some (step, _) -> Some step | None -> None

  let all_steps t = List.map fst t.checkpoints

  let checkpoint_exists t ~step =
    List.exists (fun (s, _) -> s = step) t.checkpoints

  let delete t ~step =
    let path = checkpoint_dir t step in
    let rec rm_rf path =
      if Sys.is_directory path then (
        let entries = Sys.readdir path in
        Array.iter (fun entry -> rm_rf (Filename.concat path entry)) entries;
        Unix.rmdir path)
      else Sys.remove path
    in
    (try rm_rf path with _ -> ());
    t.checkpoints <- List.filter (fun (s, _) -> s <> step) t.checkpoints
end

let save_params ~path ~params ?(metadata = []) () =
  let checkpointer = Checkpointer.create () in
  if Filename.check_suffix path ".safetensors" then
    Checkpointer.save_file checkpointer ~path ~params ~metadata ()
  else Checkpointer.save checkpointer ~path ~params ~metadata ()

let load_params ~path ~device ~dtype =
  let checkpointer = Checkpointer.create () in
  if Sys.is_directory path then
    Checkpointer.restore checkpointer ~path ~device ~dtype
  else Checkpointer.restore_file checkpointer ~path ~device ~dtype

module Async = struct
  type save_future = Thread.t

  let save ~path ~params ?(metadata = []) () =
    Thread.create (fun () -> save_params ~path ~params ~metadata ()) ()

  let wait future = Thread.join future

  let is_ready future =
    match Thread.join future with exception _ -> false | () -> true
end
