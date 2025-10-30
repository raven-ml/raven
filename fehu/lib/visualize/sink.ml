open Fehu

type t = { push : Render.t -> unit; close : unit -> unit }

let custom ?(close = fun () -> ()) push = { push; close }
let noop = custom (fun _ -> ())
let push sink frame = sink.push frame
let close sink = sink.close ()

let string_of_status = function
  | Unix.WEXITED code -> Printf.sprintf "exit status %d" code
  | Unix.WSIGNALED signal -> Printf.sprintf "signal %d" signal
  | Unix.WSTOPPED signal -> Printf.sprintf "stopped (signal %d)" signal

let ensure_ffmpeg () =
  match Unix.system "ffmpeg -version >/dev/null 2>&1" with
  | Unix.WEXITED 0 -> ()
  | _ -> invalid_arg "ffmpeg executable not found in PATH"

let rec mkdir_p dir =
  if dir = "" || dir = "." || dir = Filename.dir_sep then ()
  else if Sys.file_exists dir then ()
  else (
    mkdir_p (Filename.dirname dir);
    try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ())

type ffmpeg_state = {
  fps : int;
  path : string;
  extra_args : string list;
  mutable channel : out_channel option;
  mutable width : int;
  mutable height : int;
}

let build_command state width height =
  let extra =
    match state.extra_args with
    | [] -> ""
    | args -> " " ^ String.concat " " args
  in
  Printf.sprintf
    "ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -r %d -s %dx%d -i \
     -%s %s"
    state.fps width height extra
    (Filename.quote state.path)

let start_process (state : ffmpeg_state) (image : Render.image) =
  mkdir_p (Filename.dirname state.path);
  let cmd = build_command state image.width image.height in
  let channel =
    try Unix.open_process_out cmd
    with Unix.Unix_error (err, _, _) ->
      invalid_arg
        (Printf.sprintf "Failed to launch ffmpeg: %s" (Unix.error_message err))
  in
  state.channel <- Some channel;
  state.width <- image.width;
  state.height <- image.height;
  channel

let ensure_channel (state : ffmpeg_state) (image : Render.image) =
  match state.channel with
  | Some channel ->
      if image.width <> state.width || image.height <> state.height then
        invalid_arg "ffmpeg sink expects consistent frame dimensions";
      channel
  | None -> start_process state image

let close_state state =
  match state.channel with
  | None -> ()
  | Some channel ->
      flush channel;
      let status = Unix.close_process_out channel in
      state.channel <- None;
      if status <> Unix.WEXITED 0 then
        invalid_arg
          (Printf.sprintf "ffmpeg exited abnormally (%s)"
             (string_of_status status))

let create_ffmpeg_sink ~fps ~path ~extra_args =
  ensure_ffmpeg ();
  let state =
    { fps; path; extra_args; channel = None; width = 0; height = 0 }
  in
  let push frame =
    match frame with
    | Render.None -> ()
    | Render.Image image ->
        let channel = ensure_channel state image in
        let bytes = Utils.rgb24_bytes_of_image image in
        output_bytes channel bytes;
        flush channel
    | Render.Text _ | Render.Svg _ ->
        invalid_arg "ffmpeg sink expects image frames"
  in
  let close () = close_state state in
  custom ~close push

let ffmpeg ?(fps = 30) ~path () =
  create_ffmpeg_sink ~fps ~path
    ~extra_args:[ "-an"; "-vcodec"; "libx264"; "-pix_fmt"; "yuv420p" ]

let gif ?(fps = 30) ~path () =
  create_ffmpeg_sink ~fps ~path ~extra_args:[ "-f"; "gif" ]

let log_to_wandb ~project ~run ~path ~fps =
  let script_path = Filename.temp_file "fehu-wandb" ".py" in
  let python =
    Option.value (Sys.getenv_opt "FEHU_WANDB_PYTHON") ~default:"python3"
  in
  let write_script () =
    let oc = open_out script_path in
    Fun.protect
      ~finally:(fun () -> close_out oc)
      (fun () ->
        Printf.fprintf oc
          "import wandb\n\
           run = wandb.init(project=%S, name=%S, reinit=True)\n\
           run.log({'rollout': wandb.Video(%S, fps=%d)})\n\
           run.finish()\n"
          project run path fps)
  in
  Fun.protect
    (fun () ->
      write_script ();
      match
        Unix.system (Printf.sprintf "%s %s" python (Filename.quote script_path))
      with
      | Unix.WEXITED 0 -> ()
      | status ->
          invalid_arg
            (Printf.sprintf
               "wandb logging failed (%s). Ensure wandb is installed and you \
                are logged in."
               (string_of_status status)))
    ~finally:(fun () ->
      if Sys.file_exists script_path then Sys.remove script_path)

let wandb ?(fps = 30) ~project ~run () =
  let temp_path = Filename.temp_file "fehu-wandb" ".mp4" in
  let inner = ffmpeg ~fps ~path:temp_path () in
  let push frame = push inner frame in
  let close () =
    Fun.protect
      ~finally:(fun () ->
        if Sys.file_exists temp_path then Sys.remove temp_path)
      (fun () ->
        close inner;
        log_to_wandb ~project ~run ~path:temp_path ~fps)
  in
  custom ~close push
