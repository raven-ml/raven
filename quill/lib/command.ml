type inline_style = [ `Bold | `Italic | `Code ]

type t =
  | Insert_text of Document.block_id * int * string
  | Delete_range of Document.block_id * int * int
  | Split_block of Document.block_id * int
  | Merge_blocks of Document.block_id * Document.block_id
  | Insert_block of Document.block_content
  | Insert_after of Document.block_id * Document.block_content
  | Remove_block of Document.block_id
  | Change_block_type of Document.block_id * Document.block_content
  | Indent of Document.block_id
  | Outdent of Document.block_id
  | Toggle_inline_style of inline_style
  | Set_link of string option (* None to remove link *)
  | Set_selection of View.selection
  | Move_cursor of [ `Left | `Right | `Up | `Down | `Start | `End ]
  | Focus_block of Document.block_id
  | Focus_inline of Document.inline_id
  | Clear_focus
  | Execute_block of Document.block_id
  | Execute_all
  | Set_execution_result of Document.block_id * Execution.execution_result
  | Clear_results
  | Undo
  | Redo
