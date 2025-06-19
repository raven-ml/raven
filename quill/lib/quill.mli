(** Quill - Interactive document editing with code execution *)

(** {1 Document Model} *)
module Document : sig
  (** Pure content model - no execution state *)

  type block_id = private int
  (** Opaque block identifier *)

  type inline_id = private int
  (** Opaque inline identifier *)

  val block_id_of_int : int -> block_id
  (** Create block ID from int (for testing) *)

  val inline_id_of_int : int -> inline_id
  (** Create inline ID from int (for testing) *)

  type inline_content =
    | Run of string
    | Emph of inline
    | Strong of inline
    | Code_span of string
    | Seq of inline list
    | Break of [ `Hard | `Soft ]
    | Image of { alt : inline; src : string }
    | Link of { text : inline; href : string }
    | Raw_html of string

  and inline = { id : inline_id; content : inline_content }
  (** Rich inline content *)

  type list_type = Ordered of int * char | Unordered of char
  type list_spacing = Tight | Loose

  type block_content =
    | Paragraph of inline
    | Codeblock of {
        code : string;
        language : string option;
        output : block list option;
            (** Output blocks associated with this code block *)
      }
    | Heading of int * inline
    | Blank_line
    | Block_quote of block list
    | Thematic_break
    | List of list_type * list_spacing * block list list
    | Html_block of string

  and block = { id : block_id; content : block_content }
  (** Document block with rich content *)

  type t = { blocks : block list  (** Blocks in document order *) }
  (** Document *)

  (** {2 Construction} *)

  val empty : t
  (** Empty document *)

  val make_inline : id:inline_id -> inline_content -> inline
  (** Create inline with given ID *)

  val make_block : id:block_id -> block_content -> block
  (** Create block with given ID *)

  val run : ?focused:bool -> id:inline_id -> string -> inline
  (** Create text run *)

  val paragraph : id:block_id -> inline -> block
  (** Create paragraph block *)

  val codeblock :
    ?language:string -> ?output:block list -> id:block_id -> string -> block
  (** Create code block *)

  val heading : id:block_id -> int -> inline -> block
  (** Create heading block *)

  val add_block : t -> block -> t
  (** Add block to document *)

  val insert_after : t -> block_id -> block -> t
  (** Insert block after given block *)

  val remove_block : t -> block_id -> t
  (** Remove block *)

  val update_block_content : t -> block_id -> block_content -> t
  (** Update block content *)

  (** {2 Queries} *)

  val find_block : t -> block_id -> block option
  (** Find block by ID *)

  val find_inline : t -> inline_id -> inline option
  (** Find inline by ID *)

  val get_blocks : t -> block list
  (** Get all blocks in order *)

  val block_count : t -> int
  (** Number of blocks *)

  val blocks_from : t -> block_id -> block_id list
  (** Get block IDs from given block to end (for invalidation) *)

  val previous_block : t -> block_id -> block_id option
  (** Get previous block ID *)

  val next_block : t -> block_id -> block_id option
  (** Get next block ID *)

  val to_plain_text : t -> string
  (** Convert document to plain text *)

  val inline_to_text : inline -> string
  (** Convert inline to plain text *)

  val get_codeblocks : t -> (block_id * string * string option) list
  (** Get all code blocks with their IDs, code, and language *)

  val has_style : [ `Bold | `Italic | `Code ] -> inline -> bool
  (** Check if an inline element has a specific style *)

  val remove_style :
    next_id:int -> [ `Bold | `Italic | `Code ] -> inline -> inline * int
  (** Remove a style from an inline element, returns updated inline and next_id
  *)

  val apply_style :
    next_id:int -> [ `Bold | `Italic | `Code ] -> inline -> inline * int
  (** Apply a style to an inline element, returns updated inline and next_id *)

  val link : id:inline_id -> href:string -> inline -> inline
  (** Create a link inline element *)
end

(** {1 View State} *)
module View : sig
  (** Ephemeral view state - cursor, selection, focus *)

  type position = { block_id : Document.block_id; offset : int }
  (** Position in document *)

  type selection = { anchor : position; focus : position }
  (** Text selection *)

  type t = {
    selection : selection option;
    focused_block : Document.block_id option;
    focused_inline : Document.inline_id option;
    mode : [ `Normal | `Insert ];
  }
  (** View state *)

  val empty : t
  (** Empty view state *)

  val make_position : Document.block_id -> int -> position
  (** Create position *)

  val make_selection : position -> position -> selection
  (** Create selection *)

  val collapsed_at : position -> selection
  (** Collapsed selection at position *)

  val is_collapsed : selection -> bool
  (** Check if selection is collapsed *)

  val set_selection : t -> selection -> t
  (** Set selection *)

  val set_focus_block : t -> Document.block_id option -> t
  (** Set focused block *)

  val set_focus_inline : t -> Document.inline_id option -> t
  (** Set focused inline *)

  val clear_focus : t -> t
  (** Clear all focus *)
end

(** {1 Text Operations} *)
module Text : sig
  (** Text manipulation utilities *)

  val insert_at : string -> int -> string -> string
  (** Insert text at position *)

  val delete_range : string -> int -> int -> string
  (** Delete text range *)

  val split_at : string -> int -> string * string
  (** Split text at position *)
end

(** {1 Execution Context} *)
module Execution : sig
  (** Manages execution state separate from document content *)

  type execution_result = {
    output : string;
    error : string option;
    timestamp : float;
  }
  (** Result of executing a block *)

  type t = {
    executed : Document.block_id list;
        (** Successfully executed blocks in order *)
    results : (Document.block_id, execution_result) Hashtbl.t;
        (** Execution results *)
    stale : (Document.block_id, unit) Hashtbl.t;
        (** Blocks that need re-execution *)
  }
  (** Execution context *)

  val empty : t
  (** Empty execution context *)

  val is_executed : t -> Document.block_id -> bool
  (** Check if block has been executed *)

  val is_stale : t -> Document.block_id -> bool
  (** Check if block needs re-execution *)

  val can_execute : t -> Document.t -> Document.block_id -> bool
  (** Check if block can be executed (dependencies satisfied) *)

  val mark_executed : t -> Document.block_id -> execution_result -> t
  (** Mark block as successfully executed *)

  val mark_stale_from : t -> Document.t -> Document.block_id -> t
  (** Mark block and all following blocks as stale *)

  val get_result : t -> Document.block_id -> execution_result option
  (** Get execution result for block *)

  val clear_results : t -> t
  (** Clear all execution results *)
end

(** {1 Commands} *)
module Command : sig
  (** High-level user intentions *)

  type inline_style = [ `Bold | `Italic | `Code ]

  type t =
    (* Document editing *)
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
    (* Rich text editing *)
    | Toggle_inline_style of inline_style
    | Set_link of string option (* None to remove link *)
    (* View operations *)
    | Set_selection of View.selection
    | Move_cursor of [ `Left | `Right | `Up | `Down | `Start | `End ]
    | Focus_block of Document.block_id
    | Focus_inline of Document.inline_id
    | Clear_focus
    (* Execution *)
    | Execute_block of Document.block_id
    | Execute_all
    | Set_execution_result of Document.block_id * Execution.execution_result
    | Clear_results
    (* History *)
    | Undo
    | Redo  (** Command types *)
end

(** {1 Effects} *)
module Effect : sig
  (** Side effects for platform integration *)

  type t =
    | Execute_code of {
        block_id : Document.block_id;
        code : string;
        language : string option;
        callback : Execution.execution_result -> Command.t;
      }
    | Save_document
    | Load_document of string  (** Effect types *)
end

(** {1 Core Engine} *)
module Engine : sig
  (** Main state machine *)

  type state = {
    document : Document.t;
    execution : Execution.t;
    view : View.t;
    history : state list;  (** Simple undo stack *)
    redo_stack : state list;  (** Redo stack *)
    next_block_id : int;  (** Next available block ID *)
    next_inline_id : int;  (** Next available inline ID *)
  }
  (** Complete editor state *)

  val empty : state
  (** Empty editor state *)

  val make : Document.t -> state
  (** Create state from document *)

  (** {2 Core Operation} *)

  val execute : state -> Command.t -> state * Effect.t list
  (** Execute command, return new state and effects *)

  (** {2 Queries} *)

  val get_document : state -> Document.t
  (** Get current document *)

  val get_execution : state -> Execution.t
  (** Get execution context *)

  val get_view : state -> View.t
  (** Get view state *)

  val can_undo : state -> bool
  (** Check if undo is possible *)

  val can_redo : state -> bool
  (** Check if redo is possible *)

  val block_can_execute : state -> Document.block_id -> bool
  (** Check if block can be executed *)

  val get_block_result :
    state -> Document.block_id -> Execution.execution_result option
  (** Get execution result for block *)

  val get_focused_block : state -> Document.block option
  (** Get currently focused block *)

  val get_selection : state -> View.selection option
  (** Get current selection *)
end

(** {1 Text Diffing} *)
module Diff : sig
  (** Text diffing for efficient updates *)

  type change = Keep of int | Insert of int * string | Delete of int * int

  val compute_changes : string -> string -> change list
  (** Compute minimal changes between two strings *)

  val changes_to_operations : Document.block_id -> change list -> Command.t list
  (** Convert changes to document operations *)
end

(** {1 Events} *)
module Event : sig
  (** Events for undo/redo and debugging *)

  type t =
    | Text_inserted of Document.block_id * int * string
    | Text_deleted of Document.block_id * int * int
    | Block_inserted of int * Document.block
    | Block_removed of Document.block_id
    | Selection_changed of View.selection option
    | Focus_changed of Document.block_id option * Document.inline_id option
    | Execution_completed of Document.block_id * Execution.execution_result

  val to_string : t -> string
  (** String representation for debugging *)
end

(** {1 Markdown Support} *)
module Markdown : sig
  (** Parsing and serialization *)

  val parse : string -> Document.t
  (** Parse markdown to document *)

  val serialize : Document.t -> string
  (** Serialize document to markdown *)

  exception Parse_error of string
  (** Parse error *)
end

(** {1 Cursor Operations} *)
module Cursor : sig
  (** Cursor-centric operations for user convenience *)

  type position = View.position = { block_id : Document.block_id; offset : int }
  (** Cursor position *)

  val find_block_at_offset : Document.t -> int -> Document.block_id option
  (** Find block containing given character offset *)

  val block_start_offset : Document.t -> Document.block_id -> int
  (** Get character offset where block starts in document *)

  val document_offset : Document.t -> position -> int
  (** Convert cursor position to document character offset *)

  val move_cursor :
    Document.t ->
    position ->
    [ `Left | `Right | `Up | `Down ] ->
    position option
  (** Move cursor in given direction *)

  val find_word_boundaries : string -> int -> int * int
  (** Find word boundaries around position *)
end

(** {1 Convenience Functions} *)

val empty_document : Document.t
(** Empty document *)

val parse_markdown : string -> Document.t
(** Parse markdown *)

val serialize_document : Document.t -> string
(** Serialize document *)

val empty_editor : Engine.state
(** Empty editor state *)

val execute_command : Engine.state -> Command.t -> Engine.state * Effect.t list
(** Execute command *)
