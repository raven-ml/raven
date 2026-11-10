(* The repository's Markdown, rendered for the site: the docs and the
   posts.

   - Its first heading is its title and is left out of the body; the
     paragraph right after it can be taken out as its lead.
   - Headings get ids, and the second-level ones are the page's contents.
   - A link to a file of the repository goes where the file is on the
     site, or to the file on GitHub when the site doesn't show it. A link
     to a file that doesn't exist is reported.
   - OCaml is highlighted. A block of toplevel phrases, as mdx writes them,
     shows each phrase and its output. A block mdx runs, and a shell
     command, can be copied. *)

open Html

let ( / ) = Filename.concat

(* "packages/nx/doc/../lib/./nx.mli" is "packages/nx/lib/nx.mli". A path
   that leaves the repository keeps its "..", so it exists nowhere. *)
let normalize path =
  let rec go acc = function
    | [] -> String.concat "/" (List.rev acc)
    | ("" | ".") :: rest -> go acc rest
    | ".." :: rest -> ( match acc with d :: acc when d <> ".." -> go acc rest | _ -> go (".." :: acc) rest)
    | p :: rest -> go (p :: acc) rest
  in
  go [] (String.split_on_char '/' path)

(* "https:", "mailto:": a scheme before any '/'. *)
let has_scheme dest =
  match (String.index_opt dest ':', String.index_opt dest '/') with
  | Some i, Some j -> i < j
  | Some _, None -> true
  | None, _ -> false

type env = {
  root : string;  (** The repository. *)
  site : string -> string option;  (** A file's URL on the site, by its path in the repository. *)
}

type page = {
  title : string;
  title_id : string;
  lead : string option;  (** HTML, without its <p>. *)
  summary : string;  (** The lead as text, or else the title. *)
  body : string;
  contents : (string * string) list;  (** The second-level headings: id, text. *)
  ids : string list;  (** Every heading's id. *)
  anchors : (string * string) list;  (** Links into a page of the site: its file, the id. *)
  broken : string list;  (** What is wrong with the links, one message each. *)
}

(* A link's destination on the site, from the document [file]. *)
let resolve env ~file ~anchors ~broken dest =
  let path, frag =
    match String.index_opt dest '#' with
    | Some i -> (String.sub dest 0 i, Some (String.sub dest (i + 1) (String.length dest - i - 1)))
    | None -> (dest, None)
  in
  let with_frag url = match frag with Some f -> url ^ "#" ^ f | None -> url in
  if dest = "" || has_scheme dest || dest.[0] = '/' then dest
  else if path = "" then (
    Option.iter (fun f -> anchors := (file, f) :: !anchors) frag;
    dest)
  else
    let target = normalize (Filename.dirname file / path) in
    match env.site target with
    | Some url ->
        Option.iter (fun f -> anchors := (target, f) :: !anchors) frag;
        with_frag url
    | None when Sys.file_exists (env.root / target) ->
        let kind = if Sys.is_directory (env.root / target) then "tree" else "blob" in
        with_frag (sprintf "%s/%s/main/%s" github kind target)
    | None ->
        broken := sprintf "%s: a link to %s, which doesn't exist" file dest :: !broken;
        dest

(* [html] with each src="..." and href="..." resolved as a link: the
   paths of raw HTML in Markdown are the repository's too. *)
let resolve_html resolve html =
  let b = Buffer.create (String.length html) in
  let n = String.length html in
  let rec go i =
    let at attr = i + String.length attr <= n && String.sub html i (String.length attr) = attr in
    if i >= n then ()
    else
      match List.find_opt at [ {|src="|}; {|href="|} ] with
      | Some attr -> (
          let start = i + String.length attr in
          match String.index_from_opt html start '"' with
          | Some stop ->
              Buffer.add_string b attr;
              Buffer.add_string b (resolve (String.sub html start (stop - start)));
              Buffer.add_char b '"';
              go (stop + 1)
          | None -> Buffer.add_string b (String.sub html i (n - i)))
      | None ->
          Buffer.add_char b html.[i];
          go (i + 1)
  in
  go 0;
  Buffer.contents b

let starts ~prefix s = String.starts_with ~prefix s
let ends ~suffix s = String.ends_with ~suffix s

(* The phrases of a toplevel block, as (input, output): a phrase starts
   with "# " and ends with ";;", and the lines after it, up to the next
   phrase, are its output. *)
let phrases lines =
  let close acc = function None -> acc | Some (i, o) -> (List.rev i, List.rev o) :: acc in
  let rec go acc cur ~input = function
    | [] -> List.rev (close acc cur)
    | l :: rest when (not input) && starts ~prefix:"# " l ->
        let l = String.sub l 2 (String.length l - 2) in
        go (close acc cur) (Some ([ l ], [])) ~input:(not (ends ~suffix:";;" (String.trim l))) rest
    | l :: rest -> (
        match cur with
        | Some (i, o) when input -> go acc (Some (l :: i, o)) ~input:(not (ends ~suffix:";;" (String.trim l))) rest
        | Some (i, o) -> go acc (Some (i, l :: o)) ~input:false rest
        | None -> go acc None ~input:false rest)
  in
  go [] None ~input:false lines

let rec trim_end = function
  | [] -> []
  | l -> ( match List.rev l with "" :: r -> trim_end (List.rev r) | _ -> l)

let out_panel lines =
  match trim_end lines with
  | [] -> ""
  | lines ->
      let text = String.concat "\n" lines in
      let label, aria =
        if starts ~prefix:"Exception:" text then ("raises", "exception")
        else if starts ~prefix:"Error:" text then ("error", "error")
        else ("out", "output")
      in
      sprintf {|<div class="out" tabindex="0" role="region" aria-label="%s"><span>%s</span><pre>%s</pre></div>|} aria label (esc text)

(* A code block as a figure. [copy]: the figure offers to copy its code. *)
let figure ?(copy = false) ~caption ~label inner =
  sprintf {|<figure class="code"%s>%s%s</figure>
|}
    (if copy then " data-copy-code" else "")
    (match caption with Some c -> sprintf "<figcaption><span>%s</span></figcaption>" (esc c) | None -> "")
    (match inner with
    | `Pre html -> sprintf {|<pre tabindex="0" role="region" aria-label="%s">%s</pre>|} label html
    | `Raw html -> html)

let lines c = Cmarkit.Block.Code_block.code c |> List.map Cmarkit.Block_line.to_string

let lang c =
  match Cmarkit.Block.Code_block.info_string c with
  | Some (info, _) -> Option.map fst (Cmarkit.Block.Code_block.language_of_info_string info)
  | None -> None

(* An OCaml block of toplevel phrases. *)
let is_toplevel c =
  lang c = Some "ocaml" && match List.find_opt (( <> ) "") (lines c) with Some l -> starts ~prefix:"# " l | None -> false

let toplevel c =
  let phrase (input, output) =
    sprintf {|<pre tabindex="0" role="region" aria-label="OCaml toplevel"><span class="hl-p" aria-hidden="true"># </span>%s</pre>%s|}
      (Highlight.to_html (String.concat "\n" input)) (out_panel output)
  in
  concat_map phrase (phrases (lines c))

(* [c] as a figure; [after], a block of toplevel phrases that follows it,
   goes in the same figure, under the code. *)
let code_block ~copy ~skip ?after c =
  let lines = lines c in
  match lang c with
  | Some "ocaml" when is_toplevel c -> figure ~caption:(Some "ocaml") ~label:"" (`Raw (toplevel c))
  | Some "ocaml" ->
      let code = sprintf {|<pre tabindex="0" role="region" aria-label="OCaml code">%s</pre>|} (Highlight.to_html (String.concat "\n" lines)) in
      figure ~copy:(copy && not skip) ~caption:(Some "ocaml") ~label:"" (`Raw (code ^ match after with Some t -> toplevel t | None -> ""))
  | None | Some "text" -> figure ~caption:None ~label:"Text" (`Pre (esc (String.concat "\n" lines)))
  | Some lang ->
      let shell = List.mem lang [ "sh"; "bash"; "console"; "shell" ] in
      figure ~copy:(copy && (shell || lang = "dune")) ~caption:(Some lang) ~label:(esc lang ^ " code") (`Pre (esc (String.concat "\n" lines)))

(* The pairs of a code block and the toplevel block right after it, with
   only blank lines between, among [blocks]. *)
let rec pairs = function
  | Cmarkit.Block.Code_block (a, _) :: rest when lang a = Some "ocaml" && not (is_toplevel a) -> (
      let rec next = function Cmarkit.Block.Blank_line _ :: r -> next r | r -> r in
      match next rest with
      | Cmarkit.Block.Code_block (b, _) :: _ when is_toplevel b -> (a, b) :: pairs rest
      | _ -> pairs rest)
  | _ :: rest -> pairs rest
  | [] -> []

let heading_id h = match Cmarkit.Block.Heading.id h with Some (`Auto id | `Id id) -> id | None -> ""

let renderer env ~copy ~pairs ~file ~anchors ~broken =
  let skip = ref false in
  let module C = Cmarkit_renderer.Context in
  let block c = function
    | Cmarkit.Block.Heading (h, _) ->
        let level = Cmarkit.Block.Heading.level h in
        let cls = match level with 2 -> {| class="h2-doc"|} | 3 -> {| class="h3-doc"|} | _ -> "" in
        C.string c (sprintf {|<h%d id="%s"%s>|} level (esc (heading_id h)) cls);
        C.inline c (Cmarkit.Block.Heading.inline h);
        C.string c (sprintf "</h%d>\n" level);
        true
    | Cmarkit.Block.Html_block _ as b when Repo.skipped b ->
        skip := true;
        true
    | Cmarkit.Block.Html_block (lines, _) ->
        let html = String.concat "\n" (List.map Cmarkit.Block_line.to_string lines) in
        C.string c (resolve_html (resolve env ~file ~anchors ~broken) html);
        C.string c "\n";
        true
    | Cmarkit.Block.Code_block (cb, _) ->
        (* The toplevel block of a pair is in its code block's figure. *)
        if not (List.exists (fun (_, b) -> b == cb) pairs) then
          C.string c (code_block ~copy ~skip:!skip ?after:(List.assq_opt cb pairs) cb);
        skip := false;
        true
    | _ -> false
  in
  let dest c l =
    match Cmarkit.Inline.Link.reference_definition (C.get_defs c) l with
    | Some (Cmarkit.Link_definition.Def (d, _)) -> Option.map fst (Cmarkit.Link_definition.dest d)
    | _ -> None
  in
  let inline c = function
    | Cmarkit.Inline.Link (l, _) -> (
        match dest c l with
        | Some d ->
            C.string c (sprintf {|<a href="%s">|} (esc (resolve env ~file ~anchors ~broken d)));
            C.inline c (Cmarkit.Inline.Link.text l);
            C.string c "</a>";
            true
        | None -> false)
    | Cmarkit.Inline.Image (l, _) -> (
        match dest c l with
        | Some d ->
            C.string c
              (sprintf {|<img src="%s" alt="%s">|} (esc (resolve env ~file ~anchors ~broken d)) (esc (Repo.plain (Cmarkit.Inline.Link.text l))));
            true
        | None -> false)
    | _ -> false
  in
  Cmarkit_renderer.compose (Cmarkit_html.renderer ~safe:false ()) (Cmarkit_renderer.make ~block ~inline ())

(* [render env file] renders the document at [file], a path in the
   repository. With [~lead:true], the paragraph right after the title is
   the lead, out of the body. With [~copy:false], no block offers to be
   copied. *)
let render ?(lead = false) ?(copy = true) env file =
  let doc = Cmarkit.Doc.of_string ~heading_auto_ids:true ~strict:false (Repo.read (env.root / file)) in
  let blocks = match Cmarkit.Doc.block doc with Cmarkit.Block.Blocks (bs, _) -> bs | b -> [ b ] in
  let before, (title, title_id), after =
    let rec go before = function
      | Cmarkit.Block.Heading (h, _) :: after when Cmarkit.Block.Heading.level h = 1 ->
          (List.rev before, (Repo.plain (Cmarkit.Block.Heading.inline h), heading_id h), after)
      | b :: rest -> go (b :: before) rest
      | [] -> Repo.fail "%s: no title, a first-level heading" file
    in
    go [] blocks
  in
  (* The lead is the first paragraph after the title, with only blank
     lines and link definitions between. *)
  let lead_block, after =
    let rec go skipped = function
      | (Cmarkit.Block.Blank_line _ | Cmarkit.Block.Link_reference_definition _) as b :: rest -> go (b :: skipped) rest
      | (Cmarkit.Block.Paragraph _ as p) :: rest -> (Some p, List.rev_append skipped rest)
      | l -> (None, List.rev_append skipped l)
    in
    if lead then go [] after else (None, after)
  in
  let rest = before @ after in
  let anchors = ref [] and broken = ref [] in
  let r = renderer env ~copy ~pairs:(pairs rest) ~file ~anchors ~broken in
  let to_html blocks =
    Cmarkit_renderer.doc_to_string r (Cmarkit.Doc.make ~defs:(Cmarkit.Doc.defs doc) (Cmarkit.Block.Blocks (blocks, Cmarkit.Meta.none)))
  in
  let lead =
    Option.map
      (fun p ->
        let html = String.trim (to_html [ p ]) in
        let n = String.length html in
        if starts ~prefix:"<p>" html && ends ~suffix:"</p>" html then String.sub html 3 (n - 7) else html)
      lead_block
  in
  let body = to_html rest in
  let headings =
    List.filter_map (function Cmarkit.Block.Heading (h, _) -> Some (Cmarkit.Block.Heading.level h, h) | _ -> None) rest
  in
  {
    title;
    title_id;
    lead;
    summary =
      (match lead_block with
      | Some (Cmarkit.Block.Paragraph (p, _)) -> Repo.plain (Cmarkit.Block.Paragraph.inline p)
      | _ -> title);
    body;
    contents = List.filter_map (fun (l, h) -> if l = 2 then Some (heading_id h, Repo.plain (Cmarkit.Block.Heading.inline h)) else None) headings;
    ids = title_id :: List.map (fun (_, h) -> heading_id h) headings;
    anchors = List.rev !anchors;
    broken = List.rev !broken;
  }
