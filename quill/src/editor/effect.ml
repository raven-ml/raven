type t =
  | Execute_code of { block_id : int; code : string }
  | Load_document of { path : string }
  | Save_document of { path : string; content : string }
  | Copy_to_clipboard of { text : string }
  | Cut_to_clipboard of { text : string }
  | Request_clipboard_paste
  | Notify of { level : [ `Info | `Warning | `Error ]; message : string }
