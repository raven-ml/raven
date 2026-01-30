(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Util = Yojson.Safe.Util

type t =
  | Scalar of {
      step : int;
      epoch : int option;
      tag : string;
      value : float;
      wall_time : float;
    }

let of_json (json : Yojson.Safe.t) : (t, string) result =
  try
    match Util.member "type" json |> Util.to_string with
    | "scalar" ->
        let step = Util.member "step" json |> Util.to_int in
        let tag = Util.member "tag" json |> Util.to_string in
        let value = Util.member "value" json |> Util.to_number in
        let epoch = Util.member "epoch" json |> Util.to_int_option in
        let wall_time =
          Util.member "wall_time" json
          |> Util.to_number_option |> Option.value ~default:0.0
        in
        Ok (Scalar { step; epoch; tag; value; wall_time })
    | other -> Error ("unknown event type: " ^ other)
  with Util.Type_error (msg, _) -> Error msg

let to_json (Scalar { step; epoch; tag; value; wall_time }) : Yojson.Safe.t =
  let epoch_field =
    Option.map (fun e -> ("epoch", `Int e)) epoch |> Option.to_list
  in
  `Assoc
    ([
       ("type", `String "scalar");
       ("step", `Int step);
       ("wall_time", `Float wall_time);
       ("tag", `String tag);
       ("value", `Float value);
     ]
    @ epoch_field)
