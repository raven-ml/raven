type kind =
  | Params
  | Optimizer
  | Rng
  | Payload of string
  | Custom of string
  | Unknown of string

type t = { kind : kind; label : string; snapshot : Snapshot.t }

let kind_to_string = function
  | Params -> "params"
  | Optimizer -> "optimizer"
  | Rng -> "rng"
  | Payload name -> "payload:" ^ name
  | Custom name -> "custom:" ^ name
  | Unknown s -> s

let kind_of_string = function
  | "params" -> Some Params
  | "optimizer" -> Some Optimizer
  | "rng" -> Some Rng
  | str ->
      let payload_prefix = "payload:" in
      let custom_prefix = "custom:" in
      if String.starts_with ~prefix:payload_prefix str then
        let name =
          String.sub str
            (String.length payload_prefix)
            (String.length str - String.length payload_prefix)
        in
        Some (Payload name)
      else if String.starts_with ~prefix:custom_prefix str then
        let name =
          String.sub str
            (String.length custom_prefix)
            (String.length str - String.length custom_prefix)
        in
        Some (Custom name)
      else Some (Unknown str)

let default_label = function
  | Params -> "parameters"
  | Optimizer -> "optimizer"
  | Rng -> "rng"
  | Payload name -> "payload-" ^ name
  | Custom name -> name
  | Unknown name -> name

let create ?label kind snapshot =
  let label = match label with Some l -> l | None -> default_label kind in
  { kind; label; snapshot }

let slug artifact = Util.slugify artifact.label
