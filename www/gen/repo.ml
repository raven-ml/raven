(* Facts the site reads from the repository. Each comes from the one file
   that states it; the site keeps no copy. A missing fact raises [Failure]
   naming the file to fix. *)

let fail fmt = Printf.ksprintf failwith fmt
let ( / ) = Filename.concat
let read path = In_channel.with_open_bin path In_channel.input_all

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec at i = i + m <= n && (String.sub s i m = sub || at (i + 1)) in
  at 0

(* dune-project *)

type sexp = Atom of string | List of sexp list

(* The subset of dune's syntax that dune-project files use: atoms, quoted
   strings (with escapes and backslash-newline continuations), lists and
   line comments. *)
let parse_sexps ~file s =
  let n = String.length s in
  let rec skip i =
    if i >= n then i
    else
      match s.[i] with
      | ' ' | '\t' | '\n' | '\r' -> skip (i + 1)
      | ';' -> ( match String.index_from_opt s i '\n' with Some j -> skip (j + 1) | None -> n)
      | _ -> i
  in
  let quoted i =
    let b = Buffer.create 64 in
    let rec go j =
      if j >= n then fail "%s: unterminated string" file
      else
        match s.[j] with
        | '"' -> j + 1
        | '\\' when j + 1 < n && s.[j + 1] = '\n' ->
            let k = ref (j + 2) in
            while !k < n && (s.[!k] = ' ' || s.[!k] = '\t') do incr k done;
            go !k
        | '\\' when j + 1 < n ->
            Buffer.add_char b (if s.[j + 1] = 'n' then '\n' else s.[j + 1]);
            go (j + 2)
        | c ->
            Buffer.add_char b c;
            go (j + 1)
    in
    let j = go i in
    (Atom (Buffer.contents b), j)
  in
  let rec value i =
    match s.[i] with
    | '(' -> items (i + 1) []
    | '"' -> quoted (i + 1)
    | _ ->
        let j = ref i in
        while !j < n && not (String.contains " \t\n\r();\"" s.[!j]) do incr j done;
        (Atom (String.sub s i (!j - i)), !j)
  and items i acc =
    let i = skip i in
    if i >= n then fail "%s: unclosed parenthesis" file
    else if s.[i] = ')' then (List (List.rev acc), i + 1)
    else
      let v, j = value i in
      items j (v :: acc)
  in
  let rec top i acc =
    let i = skip i in
    if i >= n then List.rev acc
    else
      let v, j = value i in
      top j (v :: acc)
  in
  top 0 []

let atom ~file name fields =
  let field = function List (Atom k :: v) when k = name -> Some v | _ -> None in
  match List.find_map field fields with
  | Some [ Atom v ] -> v
  | _ -> fail "%s: no (%s ...)" file name

type package = {
  name : string;
  synopsis : string;
  depends : string list;  (** What it needs to build and run, not to test or document. *)
}

let rec mentions a = function Atom x -> x = a | List l -> List.exists (mentions a) l

let depends fields =
  let field = function List (Atom "depends" :: v) -> Some v | _ -> None in
  List.find_map field fields |> Option.value ~default:[]
  |> List.filter_map (function
       | Atom name -> Some name
       | List (Atom name :: c) when not (List.exists (fun c -> mentions ":with-test" c || mentions ":with-doc" c) c) -> Some name
       | _ -> None)

(* The packages a dune-project declares, in its order. *)
let packages file =
  parse_sexps ~file (read file)
  |> List.filter_map (function
       | List (Atom "package" :: fields) ->
           Some { name = atom ~file "name" fields; synopsis = atom ~file "synopsis" fields; depends = depends fields }
       | _ -> None)

(* The core: packages/, declared in the root dune-project. *)
let core root = packages (root / "dune-project")

(* Contrib: contrib/<name>/, each its own dune project. *)
let contrib root =
  let dir = root / "contrib" in
  Sys.readdir dir |> Array.to_list |> List.sort compare
  |> List.filter (fun d -> Sys.file_exists (dir / d / "dune-project"))
  |> List.concat_map (fun d -> packages (dir / d / "dune-project"))

(* The release the site is for: the newest version in CHANGES.md, its
   first "## [VERSION]" heading, released or not. "1.0.0~beta1" is
   "beta 1"; a version without a pre-release part is itself. *)
let release root =
  let file = root / "CHANGES.md" in
  let version =
    String.split_on_char '\n' (read file)
    |> List.find_map (fun l ->
           if String.starts_with ~prefix:"## [" l then
             match String.index_opt l ']' with Some j -> Some (String.sub l 4 (j - 4)) | None -> None
           else None)
    |> function
    | Some v -> v
    | None -> fail "%s: no version heading, ## [VERSION]" file
  in
  match String.index_opt version '~' with
  | None -> version
  | Some i ->
      let tag = String.sub version (i + 1) (String.length version - i - 1) in
      let j = ref (String.length tag) in
      while !j > 0 && tag.[!j - 1] >= '0' && tag.[!j - 1] <= '9' do decr j done;
      if !j = String.length tag || !j = 0 then tag
      else String.sub tag 0 !j ^ " " ^ String.sub tag !j (String.length tag - !j)

(* Markdown *)

let blocks path =
  match Cmarkit.Doc.block (Cmarkit.Doc.of_string (read path)) with
  | Cmarkit.Block.Blocks (blocks, _) -> blocks
  | block -> [ block ]

let plain inline =
  Cmarkit.Inline.to_plain_text ~break_on_soft:false inline
  |> List.map (String.concat "") |> String.concat " " |> String.trim

let heading = function
  | Cmarkit.Block.Heading (h, _) -> Some (plain (Cmarkit.Block.Heading.inline h))
  | _ -> None

(* The items of a list, as the inlines of their first paragraph. *)
let items = function
  | Cmarkit.Block.List (l, _) ->
      List.filter_map
        (fun (item, _) ->
          let first = function Cmarkit.Block.Paragraph (p, _) -> Some (Cmarkit.Block.Paragraph.inline p) | _ -> None in
          match Cmarkit.Block.List_item.block item with
          | Cmarkit.Block.Blocks (bs, _) -> List.find_map first bs
          | b -> first b)
        (Cmarkit.Block.List'.items l)
  | _ -> []

let inlines = function Cmarkit.Inline.Inlines (is, _) -> is | i -> [ i ]

(* The first link among [is], as its text and destination. *)
let rec first_link is =
  List.find_map
    (function
      | Cmarkit.Inline.Link (l, _) -> (
          match Cmarkit.Inline.Link.reference l with
          | `Inline (def, _) -> (
              match Cmarkit.Link_definition.dest def with
              | Some (dest, _) -> Some (plain (Cmarkit.Inline.Link.text l), dest)
              | None -> None)
          | `Ref _ -> None)
      | Cmarkit.Inline.Strong_emphasis (e, _) | Cmarkit.Inline.Emphasis (e, _) ->
          first_link (inlines (Cmarkit.Inline.Emphasis.inline e))
      | _ -> None)
    is

(* The title of a document: its first heading. *)
let title path =
  match List.find_map heading (blocks path) with Some t -> t | None -> fail "%s: no heading" path

(* mdx doesn't run a block that a [$MDX skip] comment precedes. *)
let skipped = function
  | Cmarkit.Block.Html_block (lines, _) ->
      List.exists (fun l -> contains (Cmarkit.Block_line.to_string l) "$MDX skip") lines
  | _ -> false

let ocaml_code = function
  | Cmarkit.Block.Code_block (c, _) -> (
      match Cmarkit.Block.Code_block.info_string c with
      | Some (info, _) when Cmarkit.Block.Code_block.language_of_info_string info |> Option.map fst = Some "ocaml" ->
          Some (Cmarkit.Block.Code_block.code c |> List.map Cmarkit.Block_line.to_string |> String.concat "\n")
      | _ -> None)
  | _ -> None

(* The first top-level [ocaml] block of a document that mdx runs, or of
   its section whose heading's id is [section]. *)
let tested_code ?section path =
  let rec go ~skip = function
    | [] -> fail "%s: no tested ocaml block" path
    | b :: rest -> (
        match ocaml_code b with Some src when not skip -> src | _ -> go ~skip:(skipped b) rest)
  in
  let blocks =
    match section with
    | None -> blocks path
    | Some id ->
        (* The blocks after the heading whose id is [id]. *)
        let doc = Cmarkit.Doc.of_string ~heading_auto_ids:true ~strict:false (read path) in
        let blocks = match Cmarkit.Doc.block doc with Cmarkit.Block.Blocks (bs, _) -> bs | b -> [ b ] in
        let rec after = function
          | Cmarkit.Block.Heading (h, _) :: rest
            when (match Cmarkit.Block.Heading.id h with Some (`Auto i | `Id i) -> i = id | None -> false) ->
              rest
          | _ :: rest -> after rest
          | [] -> fail "%s: no section %s" path id
        in
        after blocks
  in
  go ~skip:false blocks

(* Posts: doc/blog/<date>-<slug>.md, where <date> is YYYY-MM or
   YYYY-MM-DD. An empty file is a draft. Each post's blurb is its entry in
   doc/blog/index.md. *)

type post = { slug : string; date : string; title : string; blurb : string option }

let date_of name =
  let digits i j = String.for_all (fun c -> c >= '0' && c <= '9') (String.sub name i (j - i)) in
  let n = String.length name in
  if n >= 10 && digits 0 4 && name.[4] = '-' && digits 5 7 && name.[7] = '-' && digits 8 10 then
    Some (String.sub name 0 10)
  else if n >= 7 && digits 0 4 && name.[4] = '-' && digits 5 7 then Some (String.sub name 0 7)
  else None

(* Entries of doc/blog/index.md: a link to the post, then its blurb. *)
let blurbs dir =
  List.concat_map items (blocks (dir / "index.md"))
  |> List.filter_map (fun inline ->
         let is = inlines inline in
         match (first_link is, is) with
         | Some (_, dest), _ :: rest ->
             Some (Filename.remove_extension dest, plain (Cmarkit.Inline.Inlines (rest, Cmarkit.Meta.none)))
         | _ -> None)

let posts root =
  let dir = root / "doc" / "blog" in
  let blurbs = blurbs dir in
  Sys.readdir dir |> Array.to_list
  |> List.filter_map (fun name ->
         let path = dir / name in
         if (not (Filename.check_suffix name ".md")) || name = "index.md" then None
         else if In_channel.with_open_bin path In_channel.length = 0L then None
         else
           match date_of name with
           | None -> fail "%s: a post's name starts with its date, YYYY-MM or YYYY-MM-DD" path
           | Some date ->
               let slug = Filename.remove_extension name in
               Some { slug; date; title = title path; blurb = List.assoc_opt slug blurbs })
  |> List.sort (fun a b -> compare b.slug a.slug)

(* Counts *)

let is_digit c = c >= '0' && c <= '9'

(* A package's guides: the NN-*.md files of its doc directory. *)
let guides dir =
  if not (Sys.file_exists dir) then 0
  else
    Sys.readdir dir |> Array.to_list
    |> List.filter (fun n -> String.length n > 3 && is_digit n.[0] && is_digit n.[1] && n.[2] = '-' && Filename.check_suffix n ".md")
    |> List.length

(* A package's examples: the directories of its examples directory. *)
let examples dir =
  if not (Sys.file_exists dir) then 0
  else Sys.readdir dir |> Array.to_list |> List.filter (fun n -> Sys.is_directory (dir / n)) |> List.length

(* [glob name]: "*.md" matches "index.md"; one '*' at most. *)
let matches glob name =
  match String.index_opt glob '*' with
  | None -> glob = name
  | Some i ->
      let pre = String.sub glob 0 i and post = String.sub glob (i + 1) (String.length glob - i - 1) in
      String.length name >= String.length pre + String.length post
      && String.starts_with ~prefix:pre name && String.ends_with ~suffix:post name

(* The files of [dir] that its dune runs through mdx: those its mdx
   stanza lists, or, without a list, all its Markdown. *)
let mdx_files dir =
  let file = dir / "dune" in
  if not (Sys.file_exists file) then []
  else
    let markdown = Sys.readdir dir |> Array.to_list |> List.filter (fun f -> Filename.check_suffix f ".md") in
    parse_sexps ~file (read file)
    |> List.concat_map (function
         | List (Atom "mdx" :: fields) -> (
             match List.find_map (function List (Atom "files" :: fs) -> Some fs | _ -> None) fields with
             | Some fs ->
                 let globs = List.filter_map (function Atom f -> Some f | List _ -> None) fs in
                 List.filter (fun f -> List.exists (fun g -> matches g f) globs) markdown
             | None -> markdown)
         | _ -> [])

(* A package's examples: the directories of [dir], sorted, each with what
   it shows: the first paragraph of its README.md, or else the last cell
   of its row in the table of [dir]/README.md. *)
let example_list dir =
  if not (Sys.file_exists dir) then []
  else
    let rows =
      let file = dir / "README.md" in
      if not (Sys.file_exists file) then []
      else
        Cmarkit.Doc.block (Cmarkit.Doc.of_string ~strict:false (read file))
        |> (function Cmarkit.Block.Blocks (bs, _) -> bs | b -> [ b ])
        |> List.concat_map (function
             | Cmarkit.Block.Ext_table (t, _) ->
                 List.filter_map
                   (fun ((row, _), _) ->
                     match row with
                     | `Data cells -> (
                         let cells = List.map fst cells in
                         match (List.find_map (fun c -> first_link (inlines c)) cells, List.rev cells) with
                         | Some (_, dest), last :: _ -> Some (Filename.basename (String.concat "/" (List.filter (( <> ) "") (String.split_on_char '/' dest))), plain last)
                         | _ -> None)
                     | _ -> None)
                   (Cmarkit.Block.Table.rows t)
             | _ -> [])
    in
    Sys.readdir dir |> Array.to_list |> List.filter (fun n -> Sys.is_directory (dir / n)) |> List.sort compare
    |> List.map (fun name ->
           let readme = dir / name / "README.md" in
           let own =
             if Sys.file_exists readme then
               List.find_map (function Cmarkit.Block.Paragraph (p, _) -> Some (plain (Cmarkit.Block.Paragraph.inline p)) | _ -> None) (blocks readme)
             else None
           in
           (name, match own with Some _ -> own | None -> List.assoc_opt name rows))

(* OCaml source *)

(* The top-level definition [let name ...] in [path], verbatim: from its
   first line to the line before the next top-level item or comment. *)
let definition path name =
  let lines = String.split_on_char '\n' (read path) in
  let starts l = String.length l > 0 && l.[0] <> ' ' && l.[0] <> '\t' in
  let rec find = function
    | l :: rest when String.starts_with ~prefix:("let " ^ name ^ " ") l -> take [ l ] rest
    | _ :: rest -> find rest
    | [] -> fail "%s: no top-level definition of %s" path name
  and take acc = function
    | l :: rest when not (starts l) -> take (l :: acc) rest
    | _ ->
        let rec trim = function "" :: tl -> trim tl | l -> l in
        String.concat "\n" (List.rev (trim acc))
  in
  find lines
