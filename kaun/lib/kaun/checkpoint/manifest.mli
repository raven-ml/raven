(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type artifact_entry = { kind : Artifact.kind; label : string; slug : string }

type t = {
  version : int;
  step : int option;
  created_at : float;
  tags : string list;
  metadata : (string * string) list;
  artifacts : artifact_entry list;
}

val current_version : int

val create :
  ?step:int ->
  ?tags:string list ->
  ?metadata:(string * string) list ->
  artifacts:artifact_entry list ->
  unit ->
  t

val to_json : t -> Jsont.json
val of_json : Jsont.json -> (t, string) result
val json_of_file : string -> Jsont.json
val json_to_file : string -> Jsont.json -> unit
