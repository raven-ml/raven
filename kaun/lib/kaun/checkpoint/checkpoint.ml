open Rune

type metadata = (string * string) list
type checkpoint_info = { step : int; timestamp : float; metadata : metadata }
type format = Safetensors

let default_format = Safetensors

let infer_format_from_path path =
  if Filename.check_suffix path ".safetensors" then Safetensors
  else default_format

(* Helper to get flattened tensors with names *)
let get_named_tensors : type layout.
    layout Ptree.t -> (string * (float, layout) t) list =
 fun params -> Ptree.flatten_with_paths params

let encoded_prefix = "kaun:"

let encode_path path =
  let len = String.length path in
  let buffer = Buffer.create (String.length encoded_prefix + (len * 2)) in
  Buffer.add_string buffer encoded_prefix;
  for i = 0 to len - 1 do
    Buffer.add_string buffer (Printf.sprintf "%02x" (Char.code path.[i]))
  done;
  Buffer.contents buffer

let decode_path encoded =
  let prefix_len = String.length encoded_prefix in
  let is_hex_char = function
    | '0' .. '9' | 'a' .. 'f' | 'A' .. 'F' -> true
    | _ -> false
  in
  let rec is_hex_string str idx =
    if idx >= String.length str then true
    else if is_hex_char str.[idx] then is_hex_string str (idx + 1)
    else false
  in
  if
    String.length encoded >= prefix_len
    && String.equal (String.sub encoded 0 prefix_len) encoded_prefix
  then
    let hex =
      String.sub encoded prefix_len (String.length encoded - prefix_len)
    in
    if hex = "" then ""
    else if String.length hex mod 2 = 0 && is_hex_string hex 0 then (
      let buffer = Buffer.create (String.length hex / 2) in
      let rec loop idx =
        if idx < String.length hex then (
          let byte = int_of_string ("0x" ^ String.sub hex idx 2) in
          Buffer.add_char buffer (Char.chr byte);
          loop (idx + 2))
      in
      loop 0;
      Buffer.contents buffer)
    else encoded
  else encoded

module Checkpointer = struct
  type t = { format : format }

  let create ?(format = default_format) () = { format }

  let save_safetensors ~path ~params ~metadata:_ =
    let named_tensors =
      get_named_tensors params
      |> List.map (fun (name, tensor) -> (encode_path name, tensor))
    in
    let packed_tensors =
      List.map
        (fun (name, tensor) ->
          let nx_tensor = Rune.to_nx tensor in
          (name, Nx_io.P nx_tensor))
        named_tensors
    in
    Nx_io.save_safetensor ~overwrite:true path packed_tensors

  let packed_nx_as (type a b) ~(dtype : (a, b) Nx_core.Dtype.t) packed :
      (a, b) Nx.t =
    match dtype with
    | Rune.Float16 -> Nx_io.as_float16 packed
    | Rune.Float32 -> Nx_io.as_float32 packed
    | Rune.Float64 -> Nx_io.as_float64 packed
    | _ -> failwith "Unsupported dtype"

  let restore_safetensors ~path ~dtype =
    let archive = Nx_io.load_safetensor path in
    let path_tensor_pairs =
      Hashtbl.fold
        (fun name packed acc ->
          match packed_nx_as ~dtype packed with
          | nx_tensor ->
              let rune_tensor = Rune.of_nx nx_tensor in
              (name, rune_tensor) :: acc
          | exception _ ->
              failwith
                (Printf.sprintf "Failed to convert tensor %s to float32" name))
        archive []
      |> List.map (fun (name, tensor) -> (decode_path name, tensor))
    in
    Ptree.unflatten_from_paths path_tensor_pairs

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

  let restore _t ~path ~dtype =
    let safetensors_path = Filename.concat path "params.safetensors" in

    if Sys.file_exists safetensors_path then
      restore_safetensors ~path:safetensors_path ~dtype
    else failwith "No checkpoint found in directory"

  let save_file _t ~path ~params ?(metadata = []) () =
    let format = infer_format_from_path path in
    match format with Safetensors -> save_safetensors ~path ~params ~metadata

  let restore_file _t ~path ~dtype =
    let format = infer_format_from_path path in
    match format with Safetensors -> restore_safetensors ~path ~dtype
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

  let restore t ~dtype ?step () =
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
    let params = Checkpointer.restore t.checkpointer ~path ~dtype in
    (params, info)

  let restore_best t ~dtype =
    match t.best_checkpoint with
    | Some (step, _) -> restore t ~dtype ~step ()
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

let load_params ~path ~dtype =
  let checkpointer = Checkpointer.create () in
  if Sys.is_directory path then Checkpointer.restore checkpointer ~path ~dtype
  else Checkpointer.restore_file checkpointer ~path ~dtype
