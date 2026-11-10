(* A hand-written page: a complete HTML document in pages/, served as it
   is except for its slots. pages/index.html is /, pages/roadmap.html is
   /roadmap/.

   A page written from a file of the repository says so in its head, with
   the digest of the version it matches:

     <meta name="source" content="doc/roadmap.md 308199d49c34">

   When the file changes, the page is stale until someone updates it and
   its digest.

   A slot, <!-- slot: NAME ARGS -->, is filled at build time with what the
   repository states (tested code, the packages, the posts). *)

let ( / ) = Filename.concat

type t = {
  file : string;  (** Where the page is, for messages. *)
  sources : (string * string) list;  (** Source files and their digests. *)
  html : string;
}

let find_from s i sub =
  let n = String.length s and m = String.length sub in
  let rec go i = if i + m > n then None else if String.sub s i m = sub then Some i else go (i + 1) in
  go i

let parse file =
  let html = Repo.read file in
  let tag = {|<meta name="source" content="|} in
  let rec sources i acc =
    match find_from html i tag with
    | None -> List.rev acc
    | Some j -> (
        let start = j + String.length tag in
        match String.index_from_opt html start '"' with
        | None -> Repo.fail "%s: unclosed source tag" file
        | Some k -> (
            match String.split_on_char ' ' (String.sub html start (k - start)) |> List.filter (( <> ) "") with
            | [ path; digest ] -> sources k ((path, digest) :: acc)
            | _ -> Repo.fail "%s: a source tag's content is a file and its digest" file))
  in
  { file; sources = sources 0 []; html }

(* The digest a source tag records: the first 12 hex digits of the file's
   MD5. *)
let digest path = String.sub (Digest.to_hex (Digest.file path)) 0 12

(* The page's sources that changed since the page was written, as
   (file, recorded digest, current digest). *)
let stale ~root t =
  List.filter_map
    (fun (path, recorded) ->
      let full = root / path in
      if not (Sys.file_exists full) then Repo.fail "%s: its source %s doesn't exist" t.file path;
      let current = digest full in
      if current = recorded then None else Some (path, recorded, current))
    t.sources

(* [t] with each slot replaced by [fill ~page name args]. *)
let fill t fill =
  let open_ = "<!-- slot:" in
  let buf = Buffer.create (String.length t.html * 2) in
  let rec go i =
    match find_from t.html i open_ with
    | None -> Buffer.add_string buf (String.sub t.html i (String.length t.html - i))
    | Some j -> (
        Buffer.add_string buf (String.sub t.html i (j - i));
        match find_from t.html j "-->" with
        | None -> Repo.fail "%s: unclosed slot" t.file
        | Some k -> (
            let spec = String.sub t.html (j + String.length open_) (k - j - String.length open_) in
            match String.split_on_char ' ' spec |> List.filter (( <> ) "") with
            | name :: args ->
                Buffer.add_string buf (fill ~page:t.file name args);
                go (k + 3)
            | [] -> Repo.fail "%s: a slot without a name" t.file))
  in
  go 0;
  Buffer.contents buf

(* [t] with its title and description replaced, and its main element:
   its content with [~main], or the whole element, <main> included, with
   [~around]. This is how a generated page, a post or a page of the docs,
   takes the head, header and footer of a hand-written one. *)
let with_main t ~title ~description content =
  let replace s ~start ~stop ~inclusive by =
    match find_from s 0 start with
    | None -> Repo.fail "%s: no %s" t.file start
    | Some i -> (
        let from = if inclusive then i else i + String.length start in
        match find_from s (i + String.length start) stop with
        | None -> Repo.fail "%s: no %s after %s" t.file stop start
        | Some j ->
            let upto = if inclusive then j + String.length stop else j in
            String.sub s 0 from ^ by ^ String.sub s upto (String.length s - upto))
  in
  let html = replace t.html ~start:"<title>" ~stop:"</title>" ~inclusive:false (Html.esc title) in
  let html = replace html ~start:{|<meta name="description" content="|} ~stop:{|"|} ~inclusive:false (Html.esc description) in
  match content with
  | `Main main -> replace html ~start:{|<main id="content">|} ~stop:"</main>" ~inclusive:false ("\n" ^ main ^ "\n")
  | `Around element -> replace html ~start:{|<main id="content">|} ~stop:"</main>" ~inclusive:true element
