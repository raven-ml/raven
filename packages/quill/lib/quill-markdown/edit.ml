(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type span = { first : int; last : int }

type block_kind =
  | Paragraph
  | Heading of int
  | Block_quote
  | List
  | Thematic_break
  | Table
  | Blank

type block = { span : span; kind : block_kind }
type t = { source : string; blocks : block list }

let classify_block = function
  | Cmarkit.Block.Paragraph _ -> Paragraph
  | Cmarkit.Block.Heading (h, _) -> Heading (Cmarkit.Block.Heading.level h)
  | Cmarkit.Block.Code_block _ -> Paragraph
  | Cmarkit.Block.Block_quote _ -> Block_quote
  | Cmarkit.Block.List _ -> List
  | Cmarkit.Block.Thematic_break _ -> Thematic_break
  | Cmarkit.Block.Html_block _ -> Paragraph
  | Cmarkit.Block.Blank_line _ -> Blank
  | Cmarkit.Block.Link_reference_definition _ -> Blank
  | Cmarkit.Block.Ext_table _ -> Table
  | Cmarkit.Block.Ext_math_block _ -> Paragraph
  | Cmarkit.Block.Ext_footnote_definition _ -> Blank
  | Cmarkit.Block.Blocks _ -> Paragraph
  | _ -> Paragraph

let parse source =
  let doc = Cmarkit.Doc.of_string ~locs:true ~strict:false source in
  let top_blocks =
    match Cmarkit.Doc.block doc with
    | Cmarkit.Block.Blocks (bs, _) -> bs
    | b -> [ b ]
  in
  let blocks =
    List.filter_map
      (fun b ->
        let loc = Cmarkit.Meta.textloc (Cmarkit.Block.meta b) in
        if Cmarkit.Textloc.is_none loc then None
        else
          let first = Cmarkit.Textloc.first_byte loc in
          let last = Cmarkit.Textloc.last_byte loc in
          let kind = classify_block b in
          Some { span = { first; last }; kind })
      top_blocks
  in
  { source; blocks }

let source t = t.source
let blocks t = t.blocks

let active_block t ~cursor =
  List.find_opt
    (fun b -> cursor >= b.span.first && cursor <= b.span.last)
    t.blocks

let block_source t block =
  let len = block.span.last - block.span.first + 1 in
  String.sub t.source block.span.first len

let to_html source =
  let doc = Cmarkit.Doc.of_string ~strict:false source in
  Cmarkit_html.of_doc ~safe:true doc

let block_to_html t block = to_html (block_source t block)
