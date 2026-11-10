(* JSON, as much of it as odoc writes: enough to read the pages
   [dune build @doc-json] makes. *)

type t = Null | Bool of bool | Number of float | String of string | List of t list | Object of (string * t) list

let parse ~file s =
  let n = String.length s in
  let fail i what = Repo.fail "%s: at byte %d, %s" file i what in
  let rec skip i = if i < n && (s.[i] = ' ' || s.[i] = '\n' || s.[i] = '\r' || s.[i] = '\t') then skip (i + 1) else i in
  let expect i c = if i < n && s.[i] = c then i + 1 else fail i (Printf.sprintf "expected '%c'" c) in
  let hex i = int_of_string ("0x" ^ String.sub s i 4) in
  let string i =
    let b = Buffer.create 64 in
    let rec go i =
      if i >= n then fail i "unterminated string"
      else
        match s.[i] with
        | '"' -> (Buffer.contents b, i + 1)
        | '\\' -> (
            if i + 1 >= n then fail i "unterminated escape";
            match s.[i + 1] with
            | 'n' -> Buffer.add_char b '\n'; go (i + 2)
            | 't' -> Buffer.add_char b '\t'; go (i + 2)
            | 'r' -> Buffer.add_char b '\r'; go (i + 2)
            | 'b' -> Buffer.add_char b '\b'; go (i + 2)
            | 'f' -> Buffer.add_char b '\012'; go (i + 2)
            | 'u' ->
                let u = hex (i + 2) in
                if u >= 0xD800 && u <= 0xDBFF && i + 11 < n && s.[i + 6] = '\\' && s.[i + 7] = 'u' then (
                  let lo = hex (i + 8) in
                  Buffer.add_utf_8_uchar b (Uchar.of_int (0x10000 + ((u - 0xD800) lsl 10) + (lo - 0xDC00)));
                  go (i + 12))
                else (
                  Buffer.add_utf_8_uchar b (Uchar.of_int u);
                  go (i + 6))
            | c -> Buffer.add_char b c; go (i + 2))
        | c -> Buffer.add_char b c; go (i + 1)
    in
    go i
  in
  let rec value i =
    let i = skip i in
    if i >= n then fail i "unexpected end"
    else
      match s.[i] with
      | '{' ->
          let rec fields i acc =
            let i = skip i in
            if i < n && s.[i] = '}' then (Object (List.rev acc), i + 1)
            else
              let i = if acc = [] then i else skip (expect i ',') in
              let k, i = string (expect i '"') in
              let v, i = value (expect (skip i) ':') in
              fields i ((k, v) :: acc)
          in
          fields (i + 1) []
      | '[' ->
          let rec items i acc =
            let i = skip i in
            if i < n && s.[i] = ']' then (List (List.rev acc), i + 1)
            else
              let i = if acc = [] then i else expect i ',' in
              let v, i = value i in
              items i (v :: acc)
          in
          items (i + 1) []
      | '"' ->
          let v, i = string (i + 1) in
          (String v, i)
      | 't' when i + 4 <= n && String.sub s i 4 = "true" -> (Bool true, i + 4)
      | 'f' when i + 5 <= n && String.sub s i 5 = "false" -> (Bool false, i + 5)
      | 'n' when i + 4 <= n && String.sub s i 4 = "null" -> (Null, i + 4)
      | _ ->
          let j = ref i in
          while !j < n && String.contains "+-0123456789.eE" s.[!j] do incr j done;
          if !j = i then fail i "unexpected character"
          else (Number (float_of_string (String.sub s i (!j - i))), !j)
  in
  let v, i = value 0 in
  if skip i <> n then fail i "trailing characters" else v

let member k = function Object fs -> List.assoc_opt k fs | _ -> None
let string_member ~file k v = match member k v with Some (String s) -> s | _ -> Repo.fail "%s: no string %s" file k
let list_member k v = match member k v with Some (List l) -> l | _ -> []
