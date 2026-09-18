(* Resolution of Markdown link destinations to site URLs.

   Doc sources link to each other by relative file path, so the links resolve in
   an editor and on GitHub. Each source file is registered with the URL it
   publishes at, and destinations are rewritten as pages are rendered. A
   destination with no registered target is a build error. *)

type error = { file : string; line : int; dest : string; reason : string }
type t = { urls : (string, string) Hashtbl.t; mutable errors : error list }

let create () = { urls = Hashtbl.create 512; errors = [] }

(* Collapse "." and ".." segments. Leading ".." segments are kept: source paths
   are relative to the generator's working directory, which sits under the
   repository root. *)
let normalize path =
  let rec go acc = function
    | [] -> List.rev acc
    | ("" | ".") :: rest -> go acc rest
    | ".." :: rest -> (
        match acc with
        | [] | ".." :: _ -> go (".." :: acc) rest
        | _ :: up -> go up rest)
    | seg :: rest -> go (seg :: acc) rest
  in
  String.concat "/" (go [] (String.split_on_char '/' path))

let register t ~src ~url = Hashtbl.replace t.urls (normalize src) url

(* Report paths from the repository root rather than from the generator's
   working directory. *)
let source_name src =
  if String.length src > 3 && String.sub src 0 3 = "../" then
    String.sub src 3 (String.length src - 3)
  else src

let error t ~src ~line ~dest reason =
  t.errors <- { file = source_name src; line; dest; reason } :: t.errors

let split_fragment dest =
  match String.index_opt dest '#' with
  | None -> (dest, "")
  | Some i -> (String.sub dest 0 i, String.sub dest i (String.length dest - i))

let has_scheme dest =
  match String.index_opt dest ':' with
  | None -> false
  | Some colon -> (
      match String.index_opt dest '/' with
      | Some slash -> slash > colon
      | None -> true)

let resolve t ~src ~line dest =
  let target, fragment = split_fragment dest in
  if target = "" || has_scheme target then dest
  else if target.[0] = '/' then (
    error t ~src ~line ~dest
      "site-absolute link, write a path relative to this file";
    dest)
  else
    let path = normalize (Filename.concat (Filename.dirname src) target) in
    (* A link to a directory means the page that directory publishes, so that
       linking to an example reads the same here and on a repository host. *)
    let candidates =
      if Sys.file_exists path && Sys.is_directory path then
        [ Filename.concat path "README.md"; Filename.concat path "index.md" ]
      else [ path ]
    in
    match List.find_map (Hashtbl.find_opt t.urls) candidates with
    | Some url -> url ^ fragment
    | None ->
        let reason =
          if Sys.file_exists path then "target is not published on the site"
          else "no such file"
        in
        error t ~src ~line ~dest reason;
        dest

(* Raw HTML passes through the Markdown renderer untouched, so its href and src
   attributes are rewritten textually. *)
let rewrite_html t ~src ~line html =
  let len = String.length html in
  let buf = Buffer.create len in
  let attribute_at i =
    let starts_with attr =
      let n = String.length attr in
      i + n <= len && String.sub html i n = attr
    in
    let after_space =
      i > 0 && match html.[i - 1] with ' ' | '\t' | '\n' -> true | _ -> false
    in
    if not after_space then None
    else if starts_with {|href="|} then Some 6
    else if starts_with {|src="|} then Some 5
    else None
  in
  let i = ref 0 in
  while !i < len do
    match attribute_at !i with
    | None ->
        Buffer.add_char buf html.[!i];
        incr i
    | Some attr_len -> (
        Buffer.add_string buf (String.sub html !i attr_len);
        let value_start = !i + attr_len in
        match String.index_from_opt html value_start '"' with
        | None -> i := value_start
        | Some value_end ->
            let value = String.sub html value_start (value_end - value_start) in
            Buffer.add_string buf (resolve t ~src ~line value);
            Buffer.add_char buf '"';
            i := value_end + 1)
  done;
  Buffer.contents buf

let line_of_meta meta =
  fst (Cmarkit.Textloc.first_line (Cmarkit.Meta.textloc meta))

let map_link t ~src mapper ~line link =
  let text =
    match Cmarkit.Mapper.map_inline mapper (Cmarkit.Inline.Link.text link) with
    | Some text -> text
    | None -> Cmarkit.Inline.empty
  in
  let reference =
    match Cmarkit.Inline.Link.reference link with
    | `Ref _ as reference -> reference
    | `Inline (ld, ld_meta) -> (
        match Cmarkit.Link_definition.dest ld with
        | None -> `Inline (ld, ld_meta)
        | Some (dest, dest_meta) ->
            let dest = resolve t ~src ~line dest in
            let ld =
              Cmarkit.Link_definition.make
                ?title:(Cmarkit.Link_definition.title ld)
                ~dest:(dest, dest_meta) ()
            in
            `Inline (ld, ld_meta))
  in
  Cmarkit.Inline.Link.make text reference

let rewrite t ~src doc =
  let inline mapper = function
    | Cmarkit.Inline.Link (link, meta) ->
        let line = line_of_meta meta in
        Cmarkit.Mapper.ret
          (Cmarkit.Inline.Link (map_link t ~src mapper ~line link, meta))
    | Cmarkit.Inline.Image (link, meta) ->
        let line = line_of_meta meta in
        Cmarkit.Mapper.ret
          (Cmarkit.Inline.Image (map_link t ~src mapper ~line link, meta))
    | Cmarkit.Inline.Raw_html (lines, meta) ->
        let line = line_of_meta meta in
        let lines =
          List.map
            (fun (blanks, (html, html_meta)) ->
              (blanks, (rewrite_html t ~src ~line html, html_meta)))
            lines
        in
        Cmarkit.Mapper.ret (Cmarkit.Inline.Raw_html (lines, meta))
    | _ -> Cmarkit.Mapper.default
  in
  let block _ = function
    | Cmarkit.Block.Html_block (lines, meta) ->
        let lines =
          List.map
            (fun (html, html_meta) ->
              let line = line_of_meta html_meta in
              (rewrite_html t ~src ~line html, html_meta))
            lines
        in
        Cmarkit.Mapper.ret (Cmarkit.Block.Html_block (lines, meta))
    | _ -> Cmarkit.Mapper.default
  in
  Cmarkit.Mapper.map_doc (Cmarkit.Mapper.make ~inline ~block ()) doc

let report t =
  match List.sort compare t.errors with
  | [] -> ()
  | errors ->
      List.iter
        (fun { file; line; dest; reason } ->
          Printf.eprintf "%s:%d: broken link [%s]: %s\n" file line dest reason)
        errors;
      Printf.eprintf "%d broken link(s)\n" (List.length errors);
      exit 1
