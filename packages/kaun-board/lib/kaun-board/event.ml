(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let json_mem name = function
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem name mems with
      | Some (_, v) -> v
      | None -> Jsont.Null ((), Jsont.Meta.none))
  | _ -> Jsont.Null ((), Jsont.Meta.none)

type direction = [ `Min | `Max ]

type t =
  | Scalar of {
      step : int;
      epoch : int option;
      tag : string;
      value : float;
      wall_time : float;
      direction : direction option;
    }

let of_json (json : Jsont.json) : (t, string) result =
  try
    match json_mem "type" json with
    | Jsont.String ("scalar", _) ->
        let step =
          match json_mem "step" json with
          | Jsont.Number (f, _) -> int_of_float f
          | _ -> failwith "expected int for step"
        in
        let tag =
          match json_mem "tag" json with
          | Jsont.String (s, _) -> s
          | _ -> failwith "expected string for tag"
        in
        let value =
          match json_mem "value" json with
          | Jsont.Number (f, _) -> f
          | _ -> failwith "expected number for value"
        in
        let epoch =
          match json_mem "epoch" json with
          | Jsont.Number (f, _) -> Some (int_of_float f)
          | _ -> None
        in
        let wall_time =
          match json_mem "wall_time" json with
          | Jsont.Number (f, _) -> f
          | _ -> 0.0
        in
        let direction =
          match json_mem "direction" json with
          | Jsont.String ("min", _) -> Some `Min
          | Jsont.String ("max", _) -> Some `Max
          | _ -> None
        in
        Ok (Scalar { step; epoch; tag; value; wall_time; direction })
    | Jsont.String (other, _) -> Error ("unknown event type: " ^ other)
    | _ -> Error "missing or invalid type field"
  with Failure msg -> Error msg

let to_json (Scalar { step; epoch; tag; value; wall_time; direction }) : Jsont.json
    =
  let epoch_field =
    Option.map (fun e -> ("epoch", Jsont.Json.int e)) epoch |> Option.to_list
  in
  let direction_field =
    Option.map
      (function `Min -> ("direction", Jsont.Json.string "min") | `Max -> ("direction", Jsont.Json.string "max"))
      direction
    |> Option.to_list
  in
  json_obj
    ([
       ("type", Jsont.Json.string "scalar");
       ("step", Jsont.Json.int step);
       ("wall_time", Jsont.Json.number wall_time);
       ("tag", Jsont.Json.string tag);
       ("value", Jsont.Json.number value);
     ]
    @ epoch_field @ direction_field)
