let null = Jsont.Null ((), Jsont.Meta.none)

let json_obj pairs =
  Jsont.Json.object'
    (List.map (fun (key, value) -> (Jsont.Json.name key, value)) pairs)

let json_mem name = function
  | Jsont.Object (members, _) -> (
      match Jsont.Json.find_mem name members with
      | Some (_, value) -> value
      | None -> null)
  | _ -> null

let json_string = function Jsont.String (value, _) -> Some value | _ -> None
let json_number = function Jsont.Number (value, _) -> Some value | _ -> None
let json_bool = function Jsont.Bool (value, _) -> Some value | _ -> None

let json_string_list = function
  | Jsont.Array (items, _) ->
      List.filter_map
        (function Jsont.String (value, _) -> Some value | _ -> None)
        items
  | _ -> []

let json_assoc = function
  | Jsont.Object (members, _) ->
      List.map (fun ((key, _), value) -> (key, value)) members
  | _ -> []

let json_of_string text =
  match Jsont_bytesrw.decode_string Jsont.json text with
  | Ok json -> json
  | Error message -> failwith message

let json_to_string ?(pretty = false) json =
  let format = if pretty then Jsont.Indent else Jsont.Minify in
  match Jsont_bytesrw.encode_string ~format Jsont.json json with
  | Ok text -> text
  | Error message -> failwith message
