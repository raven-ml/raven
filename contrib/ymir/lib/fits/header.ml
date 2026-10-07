(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A header is its records, as the file held them, and an index from keyword
   to the records that give it a value. *)

let strf = Printf.sprintf
let record_size = 80
let block_size = 2880

type t = {
  place : Err.place;  (** where the header came from, for errors *)
  records : string array;
  index : (string, int list) Hashtbl.t;
      (** keyword -> its value records, in order *)
}

(* Keywords *)

let is_key_char c =
  (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c = '-' || c = '_'

let is_standard k =
  k <> "" && String.length k <= 8 && String.for_all is_key_char k

let is_token_char c = c > ' ' && c <= '~' && c <> '='

let is_hierarchical k =
  match String.split_on_char ' ' k with
  | [] -> false
  | tokens ->
      List.for_all (fun t -> t <> "" && String.for_all is_token_char t) tokens
      && (List.length tokens >= 2 || String.length k > 8)

let reserved k = k = "COMMENT" || k = "HISTORY" || k = "" || k = "CONTINUE"

let check_key fn k =
  if reserved k then
    invalid_arg (strf "Header.%s: %S has no value; use Header.commentary" fn k);
  if not (is_standard k || is_hierarchical k) then
    invalid_arg
      (strf
         "Header.%s: %S is not a FITS keyword: 1 to 8 of A-Z 0-9 - _, or \
          HIERARCH tokens"
         fn k)

(* Records *)

let name_of r = Value.trim_right (String.sub r 0 8)
let has_indicator r = r.[8] = '=' && r.[9] = ' '

(* The keyword a record gives a value to, and the offset of its value field;
   [None] for commentary. *)
let keyword r =
  if String.length r >= 9 && String.sub r 0 9 = "HIERARCH " then
    match String.index_opt r '=' with
    | None -> None
    | Some e ->
        let tokens =
          String.sub r 9 (e - 9)
          |> String.split_on_char ' '
          |> List.filter (fun t -> t <> "")
        in
        let k = String.concat " " tokens in
        if is_hierarchical k then Some (k, e + 1) else None
  else
    let k = name_of r in
    if k = "COMMENT" || k = "HISTORY" || k = "" || not (has_indicator r) then
      None
    else Some (k, 10)

let field r =
  match keyword r with
  | None -> None
  | Some (_, off) -> Some (Value.split (String.sub r off (record_size - off)))

let index_of records =
  let index = Hashtbl.create (Array.length records) in
  for i = Array.length records - 1 downto 0 do
    match keyword records.(i) with
    | None -> ()
    | Some (k, _) ->
        let l = Option.value ~default:[] (Hashtbl.find_opt index k) in
        Hashtbl.replace index k (i :: l)
  done;
  index

let of_records ?(place = Err.nowhere) records =
  { place; records; index = index_of records }

let empty = of_records [||]
let records h = Array.to_list h.records
let equal a b = a.records = b.records
let place h = h.place
let with_place place h = { h with place }
let end_record = "END" ^ String.make 77 ' '
let is_end r = String.length r >= 8 && String.sub r 0 8 = "END     "
let pad_record r = r ^ String.make (record_size - String.length r) ' '

let to_string h =
  let n = Array.length h.records + 1 in
  let size = ((n * record_size) + block_size - 1) / block_size * block_size in
  let b = Buffer.create size in
  Array.iter (Buffer.add_string b) h.records;
  Buffer.add_string b end_record;
  Buffer.add_string b (String.make (size - Buffer.length b) ' ');
  Buffer.contents b

let pp ppf h =
  Format.pp_open_vbox ppf 0;
  Array.iteri
    (fun i r ->
      if i > 0 then Format.pp_print_cut ppf ();
      Format.pp_print_string ppf (Value.trim_right r))
    h.records;
  Format.pp_close_box ppf ()

let of_string ?(name = "") s =
  let prefix = if name = "" then "" else name ^ ": " in
  let finish acc =
    Ok (of_records ~place:(Err.file name) (Array.of_list (List.rev acc)))
  in
  if String.contains s '\n' then
    let lines = String.split_on_char '\n' s in
    let rec go acc k = function
      | [] -> Error (strf "%sno END record" prefix)
      | l :: rest ->
          let l =
            if String.length l > 0 && l.[String.length l - 1] = '\r' then
              String.sub l 0 (String.length l - 1)
            else l
          in
          if String.length l > record_size then
            Error
              (strf "%sline %d is %d bytes, past a record's 80" prefix k
                 (String.length l))
          else
            let r = pad_record l in
            if is_end r then finish acc else go (r :: acc) (k + 1) rest
    in
    go [] 1 lines
  else
    let n = String.length s / record_size in
    let rec go acc k =
      if k >= n then Error (strf "%sno END record" prefix)
      else
        let r = String.sub s (k * record_size) record_size in
        if is_end r then finish acc else go (r :: acc) (k + 1)
    in
    go [] 0

(* Reading values *)

let card_place h i k = Err.sub h.place (strf "card %d (%s)" (i + 1) k)
let error_at h i k what = Error (Err.msg (card_place h i k) what)

(* The string a string token at record [i] holds, its CONTINUE records joined
   (§4.2.1.2). *)
let joined h i token =
  match Value.unquote token with
  | None -> None
  | Some raw ->
      let rec go acc raw j =
        let s = Value.trim_right raw in
        let n = String.length s in
        let continues =
          n > 0
          && s.[n - 1] = '&'
          && j < Array.length h.records
          && name_of h.records.(j) = "CONTINUE"
          && not (has_indicator h.records.(j))
        in
        if not continues then String.concat "" (List.rev (raw :: acc))
        else
          let next =
            Value.split (String.sub h.records.(j) 10 (record_size - 10))
          in
          match Value.unquote next.token with
          | None -> String.concat "" (List.rev (raw :: acc))
          | Some raw' -> go (String.sub s 0 (n - 1) :: acc) raw' (j + 1)
      in
      Some (Value.string_value (go [] raw (i + 1)))

(* The value of record [i] in grammar [b]; [None] if it is undefined. *)
let read_card (type b) h (b : b Value.base) k i : (b option, string) result =
  match field h.records.(i) with
  | None -> assert false
  | Some { token = ""; _ } -> Ok None
  | Some { token; _ } -> (
      let joined = if Value.is_string_base b then joined h i token else None in
      match Value.read b token joined with
      | Ok x -> Ok (Some x)
      | Error e -> error_at h i k e)

let cards h k = Option.value ~default:[] (Hashtbl.find_opt h.index k)

let find_base (type b) (b : b Value.base) k h : (b option, string) result =
  let rec agree first = function
    | [] -> Ok (Option.map snd first)
    | i :: rest -> (
        match read_card h b k i with
        | Error e -> Error e
        | Ok v -> (
            match (first, v) with
            | None, Some x -> agree (Some (i, x)) rest
            | Some _, None | None, None -> agree first rest
            | Some (j, x), Some y ->
                if Value.base_equal b x y then agree first rest
                else
                  Error
                    (Err.msg h.place
                       (strf
                          "cards %d and %d give %s different values, %s and %s"
                          (j + 1) (i + 1) k (Value.base_to_string b x)
                          (Value.base_to_string b y)))))
  in
  let l = cards h k in
  match l with
  | [] -> Ok None
  | l ->
      (* A keyword whose defined cards agree, but which is also undefined
         somewhere, is indeterminate. *)
      let defined =
        List.filter
          (fun i ->
            match field h.records.(i) with
            | Some { token = ""; _ } -> false
            | _ -> true)
          l
      in
      if defined <> [] && List.length defined <> List.length l then
        let u = List.find (fun i -> not (List.mem i defined)) l in
        Error
          (Err.msg (card_place h u k)
             (strf "undefined here and defined at card %d"
                (List.hd defined + 1)))
      else agree None l

let find (type a) (Value.V (b, conv, _) : a Value.t) k h :
    (a option, string) result =
  check_key "find" k;
  match find_base b k h with
  | Error e -> Error e
  | Ok None -> Ok None
  | Ok (Some x) -> (
      match conv x with
      | Ok y -> Ok (Some y)
      | Error e -> error_at h (List.hd (cards h k)) k e)

let get v k h =
  check_key "get" k;
  match find v k h with
  | Error e -> Error e
  | Ok (Some x) -> Ok x
  | Ok None -> (
      match cards h k with
      | [] -> Error (Err.msg h.place (k ^ " is absent"))
      | i :: _ -> error_at h i k "the value is undefined")

(* Structural keywords are read from standard records with the same grammar;
   the library reads them with these, which do not check the name. *)
let find_struct (type a) (Value.V (b, conv, _) : a Value.t) k h =
  match find_base b k h with
  | Error e -> Error e
  | Ok None -> Ok None
  | Ok (Some x) -> (
      match conv x with
      | Ok y -> Ok (Some y)
      | Error e -> error_at h (List.hd (cards h k)) k e)

(* Printing values *)

let check_text fn what s =
  if not (String.for_all Value.is_printable s) then
    invalid_arg
      (strf "Header.%s: the %s holds a byte outside ASCII 32-126" fn what)

(* The records of [k] with value text [value], a string's quoted text when
   [string] is [Some s], and [comment]. [given] says whether the caller gave
   the comment, which then must fit. *)
let print_records k ~value ~string ~comment ~given =
  let head =
    if is_standard k then Printf.sprintf "%-8s= " k else "HIERARCH " ^ k ^ " = "
  in
  let room = record_size - String.length head in
  let fit_comment used c =
    (* " / " then the comment *)
    let left = record_size - used - 3 in
    if String.length c <= left then Some c
    else if given then None
    else Some (String.sub c 0 (Int.max 0 left))
  in
  let with_comment line c =
    match c with
    | None | Some "" -> Some (pad_record line)
    | Some c -> (
        (* A comment starts at byte 32, after a short string as after a
           fixed-format value, as other writers place it. *)
        let line =
          if String.length line < 30 && is_standard k then
            Printf.sprintf "%-30s" line
          else line
        in
        match fit_comment (String.length line) c with
        | None -> None
        | Some "" -> Some (pad_record line)
        | Some c -> Some (pad_record (line ^ " / " ^ c)))
  in
  let raise_comment () =
    invalid_arg
      (strf "Header.set: the comment of %s does not fit beside its value" k)
  in
  match string with
  | None ->
      if String.length value > room then
        invalid_arg
          (strf "Header.set: %s leaves no room for its value %s" k value);
      (* Fixed format: right-justified to byte 30 when it fits there. *)
      let line =
        if is_standard k && String.length value <= 20 then
          head ^ Printf.sprintf "%20s" value
        else head ^ value
      in
      [
        (match with_comment line comment with
        | Some r -> r
        | None -> raise_comment ());
      ]
  | Some s ->
      let q = Value.quote s in
      let q =
        (* §4.2.1.1: a fixed-format string closes at byte 20 or later. *)
        if s <> "" && String.length q < 10 then
          "'"
          ^ String.sub q 1 (String.length q - 2)
          ^ String.make (10 - String.length q) ' '
          ^ "'"
        else q
      in
      if String.length q <= room then
        match with_comment (head ^ q) comment with
        | Some r -> [ r ]
        | None -> raise_comment ()
      else begin
        if room < 4 then
          invalid_arg (strf "Header.set: %s leaves no room for its value" k);
        (* Pieces of the escaped text, each ending before a doubled quote is
           split, with '&' marking that a CONTINUE record follows. *)
        let inner = String.sub q 1 (String.length q - 2) in
        let n = String.length inner in
        let pieces = ref [] and i = ref 0 and first = ref true in
        while !i < n do
          let cap = (if !first then room else record_size - 10) - 3 in
          let j = ref (Int.min n (!i + cap)) in
          (* Never end a piece between the two quotes of a doubled one. *)
          let quotes = ref 0 in
          for p = !i to !j - 1 do
            if inner.[p] = '\'' then incr quotes
          done;
          if !j < n && !quotes mod 2 = 1 then decr j;
          pieces := String.sub inner !i (!j - !i) :: !pieces;
          i := !j;
          first := false
        done;
        let pieces = List.rev !pieces in
        let last = List.length pieces - 1 in
        let line idx p =
          let body = if idx = last then "'" ^ p ^ "'" else "'" ^ p ^ "&'" in
          if idx = 0 then head ^ body else "CONTINUE  " ^ body
        in
        let lines = List.mapi line pieces in
        let rev = List.rev lines in
        let last_line = List.hd rev in
        match comment with
        | None | Some "" -> List.map pad_record lines
        | Some c -> (
            match with_comment last_line (Some c) with
            | Some r -> List.rev_map pad_record (List.tl rev) @ [ r ]
            | None -> (
                (* The comment rides on a record of its own, an empty piece. *)
                let prev = List.hd rev in
                let prev = String.sub prev 0 (String.length prev - 1) ^ "&'" in
                match with_comment "CONTINUE  ''" (Some c) with
                | None -> raise_comment ()
                | Some r ->
                    List.rev_map pad_record (List.tl rev)
                    @ [ pad_record prev; r ]))
      end

(* Editing *)

let continues_at records j =
  j < Array.length records
  && name_of records.(j) = "CONTINUE"
  && not (has_indicator records.(j))

let ends_with_amp token =
  match Value.unquote token with
  | Some raw ->
      let s = Value.trim_right raw in
      s <> "" && s.[String.length s - 1] = '&'
  | None -> false

(* The records starting at [i] that hold one value: the record and, for a
   string, the CONTINUE records that continue it. *)
let span_of records i =
  match field records.(i) with
  | Some { token; _ } when token <> "" && token.[0] = '\'' ->
      let rec go j token =
        if ends_with_amp token && continues_at records j then
          go (j + 1)
            (Value.split (String.sub records.(j) 10 (record_size - 10))).token
        else j
      in
      go (i + 1) token - i
  | _ -> 1

let span h i = span_of h.records i

let replace_records h edits =
  (* [edits]: record index -> replacement records; others removed when listed
     with [] *)
  let out = ref [] in
  Array.iteri
    (fun i r ->
      match Hashtbl.find_opt edits i with
      | None -> out := r :: !out
      | Some rs -> out := List.rev_append rs !out)
    h.records;
  of_records ~place:h.place (Array.of_list (List.rev !out))

let removal h k =
  let edits = Hashtbl.create 8 in
  List.iter
    (fun i ->
      for j = i to i + span h i - 1 do
        Hashtbl.replace edits j []
      done)
    (cards h k);
  edits

let set_records ?comment k (value, string) h =
  let kept =
    match cards h k with
    | [] -> None
    | i :: _ ->
        (* A long string's comment is on its last record. *)
        let j = i + span h i - 1 in
        let f =
          if j = i then field h.records.(i)
          else
            Some (Value.split (String.sub h.records.(j) 10 (record_size - 10)))
        in
        Option.bind f (fun f -> f.comment)
  in
  let given = Option.is_some comment in
  let comment = match comment with Some c -> Some c | None -> kept in
  Option.iter (check_text "set" "comment") (if given then comment else None);
  let rs = print_records k ~value ~string ~comment ~given in
  match cards h k with
  | [] -> of_records ~place:h.place (Array.append h.records (Array.of_list rs))
  | i :: _ ->
      let edits = removal h k in
      Hashtbl.replace edits i rs;
      replace_records h edits

let set ?comment (type a) (Value.V (b, _, print) : a Value.t) k (x : a) h =
  check_key "set" k;
  let v = print x in
  let text : string * string option =
    match b with
    | Value.Bool -> ((if v then "T" else "F"), None)
    | Value.Int -> (string_of_int v, None)
    | Value.Float ->
        if not (Float.is_finite v) then
          invalid_arg (strf "Header.set: %s is %h, which FITS cannot write" k v);
        (Value.print_float v, None)
    | Value.String ->
        check_text "set" "string" v;
        (Value.quote v, Some v)
    | Value.Text ->
        check_text "set" "text" v;
        if not (Value.is_value v) then
          invalid_arg (strf "Header.set: %S is not one FITS value" v);
        (v, None)
  in
  set_records ?comment k text h

let remove k h =
  check_key "remove" k;
  match cards h k with [] -> h | _ -> replace_records h (removal h k)

(* Commentary *)

let commentary k h =
  Array.fold_right
    (fun r acc ->
      if keyword r = None && name_of r = k then
        Value.trim_right (String.sub r 8 (record_size - 8)) :: acc
      else acc)
    h.records []

let add_commentary k s h =
  if not (k = "COMMENT" || k = "HISTORY" || k = "") then
    invalid_arg
      (strf "Header.add_commentary: %S is not COMMENT, HISTORY or blank" k);
  check_text "add_commentary" "text" s;
  let width = 72 in
  let rec pieces i acc =
    if i >= String.length s then List.rev acc
    else
      pieces (i + width)
        (String.sub s i (Int.min width (String.length s - i)) :: acc)
  in
  let ps = if s = "" then [ "" ] else pieces 0 [] in
  let rs = List.map (fun p -> pad_record (Printf.sprintf "%-8s%s" k p)) ps in
  of_records ~place:h.place (Array.append h.records (Array.of_list rs))
