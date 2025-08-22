open Rune

type metadata = (string * string) list
type checkpoint_info = { step : int; timestamp : float; metadata : metadata }

let rec params_to_json : type layout dev.
    (layout, dev) Ptree.t -> Yojson.Basic.t = function
  | Tensor t ->
      let shape = shape t in
      let data =
        unsafe_to_array (reshape [| Array.fold_left ( * ) 1 shape |] t)
      in
      `Assoc
        [
          ("type", `String "tensor");
          ("shape", `List (Array.to_list shape |> List.map (fun x -> `Int x)));
          ("data", `List (Array.to_list data |> List.map (fun x -> `Float x)));
        ]
  | List params_list ->
      `Assoc
        [
          ("type", `String "list");
          ("items", `List (List.map params_to_json params_list));
        ]
  | Record fields ->
      `Assoc
        [
          ("type", `String "record");
          ( "fields",
            `Assoc (List.map (fun (k, v) -> (k, params_to_json v)) fields) );
        ]

let rec params_from_json : type layout dev.
    Yojson.Basic.t ->
    device:dev device ->
    dtype:(float, layout) dtype ->
    (layout, dev) Ptree.t =
 fun json ~device ~dtype ->
  match json with
  | `Assoc fields -> (
      match List.assoc_opt "type" fields with
      | Some (`String "tensor") ->
          let shape =
            match List.assoc_opt "shape" fields with
            | Some (`List shape_list) ->
                List.map
                  (function `Int i -> i | _ -> failwith "Invalid shape")
                  shape_list
                |> Array.of_list
            | _ -> failwith "Missing or invalid shape"
          in
          let data =
            match List.assoc_opt "data" fields with
            | Some (`List data_list) ->
                List.map
                  (function `Float f -> f | _ -> failwith "Invalid data")
                  data_list
                |> Array.of_list
            | _ -> failwith "Missing or invalid data"
          in
          let t = create device dtype shape data in
          Tensor t
      | Some (`String "list") -> (
          match List.assoc_opt "items" fields with
          | Some (`List items) ->
              List
                (List.map
                   (fun item -> params_from_json item ~device ~dtype)
                   items)
          | _ -> failwith "Missing or invalid list items")
      | Some (`String "record") -> (
          match List.assoc_opt "fields" fields with
          | Some (`Assoc record_fields) ->
              Record
                (List.map
                   (fun (k, v) -> (k, params_from_json v ~device ~dtype))
                   record_fields)
          | _ -> failwith "Missing or invalid record fields")
      | _ -> failwith "Unknown params type")
  | _ -> failwith "Invalid JSON structure for params"

module Checkpointer = struct
  type t = unit (* Simple implementation for now *)

  let create () = ()

  let save_to_json ~params ~metadata =
    `Assoc
      [
        ("version", `String "1.0");
        ("params", params_to_json params);
        ("metadata", `Assoc (List.map (fun (k, v) -> (k, `String v)) metadata));
        ("timestamp", `Float (Unix.time ()));
      ]

  let save t ~path ~params ?(metadata = []) () =
    let _ = t in
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

    (* Save params *)
    let params_json = params_to_json params in
    let params_path = Filename.concat path "params.json" in
    let oc = open_out params_path in
    Yojson.Basic.to_channel oc params_json;
    close_out oc

  let restore t ~path ~device ~dtype =
    let _ = t in
    let params_path = Filename.concat path "params.json" in
    let ic = open_in params_path in
    let json = Yojson.Basic.from_channel ic in
    close_in ic;
    params_from_json json ~device ~dtype

  let save_file t ~path ~params ?(metadata = []) () =
    let _ = t in
    let json = save_to_json ~params ~metadata in
    let oc = open_out path in
    Yojson.Basic.to_channel oc json;
    close_out oc

  let restore_file t ~path ~device ~dtype =
    let _ = t in
    let ic = open_in path in
    let json = Yojson.Basic.from_channel ic in
    close_in ic;
    match json with
    | `Assoc fields -> (
        match List.assoc_opt "params" fields with
        | Some params_json -> params_from_json params_json ~device ~dtype
        | None -> failwith "Missing params in checkpoint file")
    | _ -> failwith "Invalid checkpoint file format"
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
        let to_delete =
          t.checkpoints
          |> List.filter (fun (step, _) -> not (should_keep t step))
          |> List.rev
          |> fun l ->
          if List.length l > max_to_keep then List.rev (List.tl (List.rev l))
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
  if Filename.check_suffix path ".json" || Filename.check_suffix path ".ckpt"
  then Checkpointer.save_file checkpointer ~path ~params ~metadata ()
  else Checkpointer.save checkpointer ~path ~params ~metadata ()

let load_params ~path ~device ~dtype =
  let checkpointer = Checkpointer.create () in
  if Filename.check_suffix path ".json" || Filename.check_suffix path ".ckpt"
  then Checkpointer.restore_file checkpointer ~path ~device ~dtype
  else Checkpointer.restore checkpointer ~path ~device ~dtype

module Async = struct
  type save_future = Thread.t

  let save ~path ~params ?(metadata = []) () =
    Thread.create (fun () -> save_params ~path ~params ~metadata ()) ()

  let wait future = Thread.join future

  let is_ready future =
    match Thread.join future with exception _ -> false | () -> true
end
