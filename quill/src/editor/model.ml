type inline =
  | Run of string
  | Emph of inline
  | Strong of inline
  | Seq of inline list

type block_content =
  | Paragraph of inline
  | Codeblock of string
  | Blocks of block list

and block = { id : int; content : block_content }

type model = { document : block list; focused : (int * int) option }

let next_id = ref 0

let init : model =
  let mk id content = { id; content } in
  {
    document =
      [
        mk 0 (Paragraph (Run "Welcome to Quill!"));
        mk 1
          (Paragraph
             (Seq [ Run "This is a "; Emph (Run "rich"); Run " text editor." ]));
      ];
    focused = None;
  }

let rec inline_of_cmarkit inline =
  let open Cmarkit in
  match inline with
  | Inline.Text (s, _) -> Some (Run s)
  | Inline.Emphasis (inner, _) -> (
      match inline_of_cmarkit (Inline.Emphasis.inline inner) with
      | Some i -> Some (Emph i)
      | None -> Some (Emph (Run "")))
  | Inline.Strong_emphasis (inner, _) -> (
      match inline_of_cmarkit (Inline.Emphasis.inline inner) with
      | Some i -> Some (Strong i)
      | None -> Some (Strong (Run "")))
  | Inline.Inlines (items, _) ->
      let inlines = List.filter_map inline_of_cmarkit items in
      if inlines = [] then None else Some (Seq inlines)
  | _ -> None

let rec block_content_of_cmarkit (cb : Cmarkit.Block.t) : block_content =
  let open Cmarkit in
  match cb with
  | Block.Paragraph (p, _) ->
      let norm = Inline.normalize (Block.Paragraph.inline p) in
      let inline =
        match inline_of_cmarkit norm with Some i -> i | None -> Run ""
      in
      Paragraph inline
  | Block.Code_block (codeblock, _) ->
      let codelines = Block.Code_block.code codeblock in
      let code =
        codelines
        |> List.map (fun l -> Block_line.to_string l)
        |> String.concat "\n"
      in
      Codeblock code
  | Block.Blocks (items, _) ->
      (* recursively convert each sub‑block to a `block` below *)
      let children = List.map block_of_cmarkit items in
      Blocks children
  | _ ->
      (* fallback for any other Cmarkit blocks *)
      Paragraph (Run "")

and block_of_cmarkit (cb : Cmarkit.Block.t) : block =
  let id = !next_id in
  incr next_id;
  { id; content = block_content_of_cmarkit cb }

and document_of_cmarkit (root : Cmarkit.Block.t) : block list =
  next_id := 0;
  (* start IDs at 0 *)
  match Cmarkit.Block.normalize root with
  | Cmarkit.Block.Blocks (items, _) -> List.map block_of_cmarkit items
  | other -> [ block_of_cmarkit other ]

let document_of_md text =
  let open Cmarkit in
  let doc = Doc.of_string ~strict:true text in
  let block = Doc.block doc in
  let normalized_block = Block.normalize block in
  document_of_cmarkit normalized_block

let block_content_of_md text =
  let open Cmarkit in
  let doc = Doc.of_string ~strict:true text in
  let block = Doc.block doc in
  let normalized_block = Block.normalize block in
  block_content_of_cmarkit normalized_block
