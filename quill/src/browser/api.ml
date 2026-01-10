(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Brr
open Brr_io

let log fmt =
  Printf.ksprintf (fun s -> Console.(log [ Jstr.v ("[api] " ^ s) ])) fmt

(* Error type for API calls *)
type api_error =
  | Fetch_error of Jv.Error.t
  | Http_error of Fetch.Response.t
  | Json_parse_error of string
  | Invalid_response of string * Jv.t option

type 'a response = ('a, api_error) Fut.result

let string_of_api_error = function
  | Fetch_error e -> Jstr.to_string (Jv.Error.message e)
  | Http_error r ->
      Printf.sprintf "HTTP error: %d %s" (Fetch.Response.status r)
        (Jstr.to_string (Fetch.Response.status_text r))
  | Json_parse_error e -> e
  | Invalid_response (msg, json_opt) -> (
      match json_opt with
      | Some json ->
          Printf.sprintf "%s - Response: %s" msg
            (Json.encode json |> Jstr.to_string)
      | None -> msg)

let map_fut_error f fut = Fut.map (fun res -> Result.map_error f res) fut

let fetch_response_json response =
  Fetch.Body.json (Fetch.Response.as_body response)
  |> map_fut_error (fun e ->
         let msg = Jv.Error.message e |> Jstr.to_string in
         Json_parse_error msg)

(* Function to send execution request *)
let execute_code (code : string) : Quill_api.code_execution_result response =
  let open Fut.Result_syntax in
  log "Sending execution request";
  let url = Jstr.v "/api/execute" in
  let headers =
    Fetch.Headers.of_assoc
      [ (Jstr.v "Content-Type", Jstr.v "application/json") ]
  in
  let body =
    Quill_api.code_execution_request_to_yojson { code }
    |> Yojson.Safe.to_string |> Jstr.v |> Fetch.Body.of_jstr
  in

  let method' = Jstr.v "POST" in
  let init = Fetch.Request.init ~headers ~body ~method' () in
  let request = Fetch.Request.v ~init url in

  let* response =
    Fut.map
      (fun res -> Result.map_error (fun e -> Fetch_error e) res)
      (Fetch.request request)
  in
  if not (Fetch.Response.ok response) then Fut.error (Http_error response)
  else
    let* json_jv = fetch_response_json response in
    let json_str = Jstr.to_string (Json.encode json_jv) in
    let json = Yojson.Safe.from_string json_str in
    let result = Quill_api.code_execution_result_of_yojson json in
    match result with
    | Error err -> Fut.error (Json_parse_error err)
    | Ok response -> Fut.ok response

(* Function to fetch the document *)
let fetch_document (path : string) : string response =
  let open Fut.Result_syntax in
  let api_url_str =
    if path = "/" || path = "" then "/api/doc"
    else "/api/doc/" ^ String.sub path 1 (String.length path - 1)
  in
  let api_url = Jstr.v api_url_str in
  log "Fetching document from %s" api_url_str;

  let* response = map_fut_error (fun e -> Fetch_error e) (Fetch.url api_url) in
  if not (Fetch.Response.ok response) then Fut.error (Http_error response)
  else
    let* text =
      map_fut_error
        (fun e -> Fetch_error e)
        (Fetch.Body.text (Fetch.Response.as_body response))
    in
    Fut.ok (Jstr.to_string text)
