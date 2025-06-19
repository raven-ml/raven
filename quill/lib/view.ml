type position = { block_id : Document.block_id; offset : int }
type selection = { anchor : position; focus : position }

type t = {
  selection : selection option;
  focused_block : Document.block_id option;
  focused_inline : Document.inline_id option;
  mode : [ `Normal | `Insert ];
}

let empty =
  {
    selection = None;
    focused_block = None;
    focused_inline = None;
    mode = `Normal;
  }

let make_position block_id offset = { block_id; offset }
let make_selection anchor focus = { anchor; focus }
let collapsed_at position = { anchor = position; focus = position }

let is_collapsed selection =
  selection.anchor.block_id = selection.focus.block_id
  && selection.anchor.offset = selection.focus.offset

let set_selection t selection = { t with selection = Some selection }
let set_focus_block t focused_block = { t with focused_block }
let set_focus_inline t focused_inline = { t with focused_inline }
let clear_focus t = { t with focused_block = None; focused_inline = None }
