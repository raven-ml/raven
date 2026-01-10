(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Focus_inline of int
  | Focus_block of int
  | Set_document of Document.t
  | Replace_block_with_codeblock of int
  | Update_codeblock of { block_id : int; code : string }
  | Request_code_execution of { block_id : int; code : string }
  | Split_block of { block_id : int; inline_id : int; offset : int }
  | Load_document of { path : string }
  | Set_selection of State.selection
  | Clear_selection
  | Normalize_document
  | Undo
  | Redo
  | Save_document of { path : string }
  | Request_copy_selection
  | Request_cut_selection
  | Request_paste_clipboard
