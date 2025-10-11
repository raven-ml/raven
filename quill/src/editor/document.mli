(** Quill document model for representing markdown structures *)

open Cmarkit

(** Inline content types *)
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

and inline = { id : int; inline_content : inline_content; focused : bool }

type codeblock_content = {
  code : string;
  output : block option;
  info : string option;
}
(** Codeblock content with optional output *)

(** Block content types *)
and block_content =
  | Paragraph of inline
  | Codeblock of codeblock_content
  | Heading of int * inline
  | Blank_line of unit
  | Block_quote of block list
  | Thematic_break
  | List of list_type * list_spacing * block list list
  | Html_block of string
  | Link_reference_definition of Link_definition.t node
  | Blocks of block list

and list_type = Ordered of int * char | Unordered of char
and list_spacing = Tight | Loose
and block = { id : int; content : block_content; focused : bool }

type t = block list
(** Document type *)

(** {1 Constructors} *)

val inline : ?focused:bool -> inline_content -> inline
(** Create inline content *)

val run : ?focused:bool -> string -> inline
val emph : ?focused:bool -> inline -> inline
val strong : ?focused:bool -> inline -> inline
val code_span : ?focused:bool -> string -> inline
val seq : ?focused:bool -> inline list -> inline

val block : ?focused:bool -> block_content -> block
(** Create block content *)

val paragraph : ?focused:bool -> inline -> block

val codeblock :
  ?output:block -> ?info:string -> ?focused:bool -> string -> block

val heading : ?focused:bool -> int -> inline -> block
val blank_line : ?focused:bool -> unit -> block
val blocks : ?focused:bool -> block list -> block
val block_quote : ?focused:bool -> block list -> block
val thematic_break : ?focused:bool -> unit -> block

val list :
  ?focused:bool -> list_type -> list_spacing -> block list list -> block

val html_block : ?focused:bool -> string -> block
val link_reference_definition : ?focused:bool -> Link_definition.t node -> block

val init : t
(** Empty document *)

val empty : t
(** Alias of {!init} for clarity *)

(** {1 Parsing and Serialization} *)

val of_markdown : string -> t
(** Parse markdown to document *)

val block_of_md : string -> block
(** Parse markdown to block *)

val block_content_of_md : string -> block_content
(** Parse markdown to block content *)

val inline_of_md : string -> inline
(** Parse markdown to inline *)

val inline_content_of_md : string -> inline_content
(** Parse markdown to inline content *)

val to_markdown : t -> string
(** Serialize document to markdown *)

(** {1 Document Operations} *)

val find_inline_in_block : block -> int -> inline option
(** Find inline by ID in a block *)

val replace_inline_in_block : block -> int -> inline -> block
(** Replace inline by ID in a block *)

val split_inline : inline -> int -> inline * inline
(** Split inline at offset *)

val set_focused_document_by_id : t -> int -> t
(** Set focused state by ID *)

val clear_focus_block : block -> block
(** Clear all focus *)

val clear_focus_inline : inline -> inline

val set_codeblock_output_in_block : block -> int -> block -> block
(** Set codeblock output *)

val normalize_blanklines : t -> t
(** Normalize blank lines *)

val inline_to_plain : inline -> string
(** Convert inline to plain text *)

val inline_content_to_plain : inline_content -> string

(** {1 Extended Helpers} *)

val focus_inline_by_id : t -> int -> t
val focus_block_by_id : t -> int -> t
val replace_block_with_codeblock : t -> block_id:int -> t
val update_codeblock : t -> block_id:int -> code:string -> t
val set_codeblock_output : t -> block_id:int -> block -> t

val split_block_at_inline :
  t -> block_id:int -> inline_id:int -> offset:int -> t

val find_block : t -> block_id:int -> block option
val index_of_block : t -> block_id:int -> int option
val find_block_of_inline : t -> inline_id:int -> (block * int) option
val block_ids_between : t -> start_id:int -> end_id:int -> int list
val slice_between : t -> start_id:int -> end_id:int -> t

(** {1 Internal} *)

val reset_ids : unit -> unit
(** Reset ID counters (used for DOM parsing) *)
