type t = [ `Bool of bool | `Int of int | `Float of float | `String of string ]

let pp ppf = function
  | `Bool b -> Format.pp_print_bool ppf b
  | `Int n -> Format.pp_print_int ppf n
  | `Float f -> Format.fprintf ppf "%g" f
  | `String s -> Format.fprintf ppf "%S" s

let to_float = function
  | `Float f -> Some f
  | `Int n -> Some (Float.of_int n)
  | _ -> None

let to_int = function
  | `Int n -> Some n
  | `Float f when Float.is_integer f -> Some (Float.to_int f)
  | _ -> None

let to_string = function `String s -> Some s | _ -> None
let to_bool = function `Bool b -> Some b | _ -> None

let to_json : t -> Jsont.json = function
  | `Bool b -> Jsont.Json.bool b
  | `Int n -> Jsont.Json.int n
  | `Float f -> Jsont.Json.number f
  | `String s -> Jsont.Json.string s

let of_json : Jsont.json -> t = function
  | Jsont.Bool (b, _) -> `Bool b
  | Jsont.Number (f, _) ->
      if Float.is_integer f && Float.abs f < 4503599627370496. then
        `Int (Float.to_int f)
      else `Float f
  | Jsont.String (s, _) -> `String s
  | _ -> `String ""
