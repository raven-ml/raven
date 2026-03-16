type summary_mode = [ `Min | `Max | `Mean | `Last | `None ]
type goal = [ `Minimize | `Maximize ]
type media_kind = [ `Image | `Audio | `Table | `File ]

type event =
  | Metric of { step : int; timestamp : float; key : string; value : float }
  | Define_metric of {
      key : string;
      summary : summary_mode;
      step_metric : string option;
      goal : goal option;
    }
  | Media of {
      step : int;
      timestamp : float;
      key : string;
      kind : media_kind;
      path : string;
    }
  | Summary of (string * Value.t) list
  | Notes of string option
  | Tags of string list
  | Artifact_output of { name : string; version : string }
  | Artifact_input of { name : string; version : string }
  | Resumed of { at : float }
  | Finished of { status : string; ended_at : float }

let json_of_optional_string = function
  | Some value -> Jsont.Json.string value
  | None -> Json_utils.null

let optional_string_of_json json =
  match json with
  | Jsont.Null _ -> Some None
  | Jsont.String (value, _) -> Some (Some value)
  | _ -> None

let summary_mode_to_string = function
  | `Min -> "min"
  | `Max -> "max"
  | `Mean -> "mean"
  | `Last -> "last"
  | `None -> "none"

let summary_mode_of_string = function
  | "min" -> Some `Min
  | "max" -> Some `Max
  | "mean" -> Some `Mean
  | "last" -> Some `Last
  | "none" -> Some `None
  | _ -> None

let goal_to_string = function
  | `Minimize -> "minimize"
  | `Maximize -> "maximize"

let goal_of_string = function
  | "minimize" -> Some `Minimize
  | "maximize" -> Some `Maximize
  | _ -> None

let media_kind_to_string = function
  | `Image -> "image"
  | `Audio -> "audio"
  | `Table -> "table"
  | `File -> "file"

let media_kind_of_string = function
  | "image" -> Some `Image
  | "audio" -> Some `Audio
  | "table" -> Some `Table
  | "file" -> Some `File
  | _ -> None

let of_json json =
  match Json_utils.json_mem "type" json |> Json_utils.json_string with
  | Some "metric" -> (
      match
        ( Json_utils.json_mem "step" json |> Json_utils.json_number,
          Json_utils.json_mem "timestamp" json |> Json_utils.json_number,
          Json_utils.json_mem "key" json |> Json_utils.json_string,
          Json_utils.json_mem "value" json |> Json_utils.json_number )
      with
      | Some step, Some timestamp, Some key, Some value ->
          Some (Metric { step = int_of_float step; timestamp; key; value })
      | _ -> None)
  | Some "define_metric" -> (
      match
        ( Json_utils.json_mem "key" json |> Json_utils.json_string,
          Json_utils.json_mem "summary" json
          |> Json_utils.json_string
          |> Fun.flip Option.bind summary_mode_of_string )
      with
      | Some key, Some summary ->
          let step_metric =
            Json_utils.json_mem "step_metric" json |> Json_utils.json_string
          in
          let goal =
            Json_utils.json_mem "goal" json
            |> Json_utils.json_string
            |> Fun.flip Option.bind goal_of_string
          in
          Some (Define_metric { key; summary; step_metric; goal })
      | _ -> None)
  | Some "media" -> (
      match
        ( Json_utils.json_mem "step" json |> Json_utils.json_number,
          Json_utils.json_mem "ts" json |> Json_utils.json_number,
          Json_utils.json_mem "key" json |> Json_utils.json_string,
          Json_utils.json_mem "kind" json
          |> Json_utils.json_string
          |> Fun.flip Option.bind media_kind_of_string,
          Json_utils.json_mem "path" json |> Json_utils.json_string )
      with
      | Some step, Some ts, Some key, Some kind, Some path ->
          Some
            (Media { step = int_of_float step; timestamp = ts; key; kind; path })
      | _ -> None)
  | Some "summary" ->
      Some
        (Summary
           (Json_utils.json_mem "values" json
           |> Json_utils.json_assoc
           |> List.map (fun (k, v) -> (k, Value.of_json v))))
  | Some "notes" ->
      Json_utils.json_mem "value" json
      |> optional_string_of_json
      |> Option.map (fun value -> Notes value)
  | Some "tags" ->
      Some
        (Tags (Json_utils.json_mem "values" json |> Json_utils.json_string_list))
  | Some "artifact_output" -> (
      match
        ( Json_utils.json_mem "name" json |> Json_utils.json_string,
          Json_utils.json_mem "version" json |> Json_utils.json_string )
      with
      | Some name, Some version -> Some (Artifact_output { name; version })
      | _ -> None)
  | Some "artifact_input" -> (
      match
        ( Json_utils.json_mem "name" json |> Json_utils.json_string,
          Json_utils.json_mem "version" json |> Json_utils.json_string )
      with
      | Some name, Some version -> Some (Artifact_input { name; version })
      | _ -> None)
  | Some "resumed" -> (
      match Json_utils.json_mem "at" json |> Json_utils.json_number with
      | Some at -> Some (Resumed { at })
      | None -> None)
  | Some "finished" -> (
      match
        ( Json_utils.json_mem "status" json |> Json_utils.json_string,
          Json_utils.json_mem "ended_at" json |> Json_utils.json_number )
      with
      | Some status, Some ended_at -> Some (Finished { status; ended_at })
      | _ -> None)
  | _ -> None

let decode_line line =
  try Json_utils.json_of_string line |> of_json with _ -> None

let to_json = function
  | Metric { step; timestamp; key; value } ->
      Json_utils.json_obj
        [
          ("type", Jsont.Json.string "metric");
          ("step", Jsont.Json.int step);
          ("timestamp", Jsont.Json.number timestamp);
          ("key", Jsont.Json.string key);
          ("value", Jsont.Json.number value);
        ]
  | Define_metric { key; summary; step_metric; goal } ->
      Json_utils.json_obj
        ([
           ("type", Jsont.Json.string "define_metric");
           ("key", Jsont.Json.string key);
           ("summary", Jsont.Json.string (summary_mode_to_string summary));
         ]
        @ (match step_metric with
          | Some sm -> [ ("step_metric", Jsont.Json.string sm) ]
          | None -> [])
        @
        match goal with
        | Some g -> [ ("goal", Jsont.Json.string (goal_to_string g)) ]
        | None -> [])
  | Media { step; timestamp; key; kind; path } ->
      Json_utils.json_obj
        [
          ("type", Jsont.Json.string "media");
          ("step", Jsont.Json.int step);
          ("ts", Jsont.Json.number timestamp);
          ("key", Jsont.Json.string key);
          ("kind", Jsont.Json.string (media_kind_to_string kind));
          ("path", Jsont.Json.string path);
        ]
  | Summary values ->
      Json_utils.json_obj
        [
          ("type", Jsont.Json.string "summary");
          ( "values",
            Json_utils.json_obj
              (List.map (fun (k, v) -> (k, Value.to_json v)) values) );
        ]
  | Notes value ->
      Json_utils.json_obj
        [
          ("type", Jsont.Json.string "notes");
          ("value", json_of_optional_string value);
        ]
  | Tags values ->
      Json_utils.json_obj
        [
          ("type", Jsont.Json.string "tags");
          ("values", Jsont.Json.list (List.map Jsont.Json.string values));
        ]
  | Artifact_output { name; version } ->
      Json_utils.json_obj
        [
          ("type", Jsont.Json.string "artifact_output");
          ("name", Jsont.Json.string name);
          ("version", Jsont.Json.string version);
        ]
  | Artifact_input { name; version } ->
      Json_utils.json_obj
        [
          ("type", Jsont.Json.string "artifact_input");
          ("name", Jsont.Json.string name);
          ("version", Jsont.Json.string version);
        ]
  | Resumed { at } ->
      Json_utils.json_obj
        [ ("type", Jsont.Json.string "resumed"); ("at", Jsont.Json.number at) ]
  | Finished { status; ended_at } ->
      Json_utils.json_obj
        [
          ("type", Jsont.Json.string "finished");
          ("status", Jsont.Json.string status);
          ("ended_at", Jsont.Json.number ended_at);
        ]

let encode event = Json_utils.json_to_string (to_json event)

let read path =
  if not (Sys.file_exists path) then []
  else
    let ic = open_in path in
    let rec loop acc =
      match input_line ic with
      | line ->
          let acc =
            match decode_line line with
            | Some event -> event :: acc
            | None -> acc
          in
          loop acc
      | exception End_of_file -> List.rev acc
    in
    Fun.protect ~finally:(fun () -> close_in ic) (fun () -> loop [])
