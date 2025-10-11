open Brr

module Ev = struct
  include Ev

  (** Event type for the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Document/selectionchange_event}
       [selectionchange]} event, fired on the document when the current text
      selection changes. *)
  let selectionchange : void = Type.create (Jstr.v "selectionchange")
end

module El = struct
  include El

  (** [inner_text el] gets the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement/innerText}
       innerText} property of the element [el], representing the rendered text
      content of the node and its descendants. *)
  let inner_text (el : El.t) : Jstr.t =
    El.prop (El.Prop.jstr (Jstr.v "innerText")) el

  (** [text_content el] gets the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Node/textContent}
       textContent} property of the element [el], representing the text content
      of the node and its descendants. *)
  let text_content (el : El.t) : Jstr.t =
    El.prop (El.Prop.jstr (Jstr.v "textContent")) el
end

module rec Selection : sig
  (** {1 Selection} *)

  type t = Jv.t
  (** The type for
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Selection} Selection}
      objects. Represents the currently selected text or the caret position. *)

  val of_jv : Jv.t -> t
  (** [of_jv jv] converts the JavaScript value [jv] to a selection object. *)

  val to_jv : t -> Jv.t
  (** [to_jv sel] converts the selection object [sel] to its underlying
      JavaScript value. *)

  val anchor_node : t -> Jv.t option
  (** [anchor_node sel] is the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Selection/anchorNode}
       anchor node} of the selection [sel] (if any). This is the node where the
      user began the selection. Returns the raw JavaScript value ([Jv.t]) for
      the node. *)

  val focus_node : t -> Jv.t option
  (** [focus_node sel] is the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Selection/focusNode}
       focus node} of the selection [sel] (if any). This is the node where the
      user ended the selection. Returns the raw JavaScript value ([Jv.t]) for
      the node. *)

  val anchor_offset : t -> int
  (** [anchor_offset sel] is the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Selection/anchorOffset}
       anchor offset} of the selection [sel]. This is the offset within the
      {!anchor_node} where the selection begins. *)

  val focus_offset : t -> int
  (** [focus_offset sel] is the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Selection/focusOffset}
       focus offset} of the selection [sel]. This is the offset within the
      {!focus_node} where the selection ends. *)

  val is_collapsed : t -> bool
  (** [is_collapsed sel] gets the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Selection/isCollapsed}isCollapsed}
      property of the selection [sel]. Returns [true] if the selection's start
      and end points are at the same position (i.e., it's a caret). *)

  val range_count : t -> int
  (** [range_count sel] is the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Selection/rangeCount}
       range count} of the selection [sel]. Typically, this is 1, even for a
      collapsed selection (caret). *)

  val get_range_at : t -> int -> Range.t
  (** [get_range_at sel index] gets the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Selection/getRangeAt}
       Range object} at the specified [index] within the selection [sel]. Most
      selections only have one range (at index 0). *)

  val add_range : t -> Range.t -> unit
  (** [add_range sel range]
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Selection/addRange}
       adds} the given [range] to the selection [sel]. *)

  val remove_all_ranges : t -> unit
  (** [remove_all_ranges sel]
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Selection/removeAllRanges}
       removes all ranges} from the selection [sel], effectively deselecting any
      selected text. *)

  val collapse : t -> Jv.t -> int -> unit
  (** [collapse sel node_jv offset]
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Selection/collapse}
       collapses} the selection [sel] to a single point (caret) located at the
      specified [offset] within the node represented by the raw JavaScript value
      [node_jv]. *)
end = struct
  type t = Jv.t

  let of_jv (jv : Jv.t) : t = jv
  let to_jv (sel : t) : Jv.t = sel
  let anchor_node (sel : t) : Jv.t option = Jv.find sel "anchorNode"
  let focus_node (sel : t) : Jv.t option = Jv.find sel "focusNode"
  let anchor_offset (sel : t) : int = Jv.Int.get sel "anchorOffset"
  let focus_offset (sel : t) : int = Jv.Int.get sel "focusOffset"
  let is_collapsed (sel : t) : bool = Jv.Bool.get sel "isCollapsed"
  let range_count (sel : t) : int = Jv.Int.get sel "rangeCount"

  let get_range_at (sel : t) (index : int) : Range.t =
    Range.of_jv @@ Jv.call sel "getRangeAt" [| Jv.of_int index |]

  let add_range (sel : t) (range : Range.t) : unit =
    Jv.call sel "addRange" [| Range.to_jv range |] |> ignore

  let remove_all_ranges (sel : t) : unit =
    Jv.call sel "removeAllRanges" [||] |> ignore

  let collapse (sel : t) (node_jv : Jv.t) (offset : int) : unit =
    Jv.call sel "collapse" [| node_jv; Jv.of_int offset |] |> ignore
end

and Range : sig
  (** {1 Range} *)

  type t
  (** The type for
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range} Range} objects.
      Represents a contiguous part of a document. *)

  val of_jv : Jv.t -> t
  (** [of_jv jv] converts the JavaScript value [jv] to a range object. *)

  val to_jv : t -> Jv.t
  (** [to_jv r] converts the range object [r] to its underlying JavaScript
      value. *)

  val start_container : t -> Jv.t
  (** [start_container r] is the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/startContainer}
       start container} node of the range [r]. Returns the raw JavaScript value
      ([Jv.t]) for the node. *)

  val end_container : t -> Jv.t
  (** [end_container r] is the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/endContainer}
       end container} node of the range [r]. Returns the raw JavaScript value
      ([Jv.t]) for the node. *)

  val common_ancestor_container : t -> Jv.t
  (** [common_ancestor_container r] is the deepest node that contains both the
      start and end containers of the range [r]. See
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/commonAncestorContainer}
       commonAncestorContainer}. Returns the raw JavaScript value ([Jv.t]) for
      the node. *)

  val start_offset : t -> int
  (** [start_offset r] is the offset within the {!start_container} where the
      range [r] begins. See
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/startOffset}
       startOffset}. *)

  val end_offset : t -> int
  (** [end_offset r] is the offset within the {!end_container} where the range
      [r] ends. See
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/endOffset}
       endOffset}. *)

  val collapsed : t -> bool
  (** [collapsed r] is the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/collapsed}
       collapsed} state of the range [r]. Returns [true] if the range's start
      and end points are the same. *)

  val set_start : t -> Jv.t -> int -> unit
  (** [set_start r node_jv offset]
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/setStart} sets
       the start} of the range [r] to [offset] within the node represented by
      the raw JavaScript value [node_jv]. *)

  val set_end : t -> Jv.t -> int -> unit
  (** [set_end r node_jv offset]
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/setEnd} sets the
       end} of the range [r] to [offset] within the node represented by the raw
      JavaScript value [node_jv]. *)

  val set_start_before : t -> Jv.t -> unit
  (** [set_start_before r node_jv]
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/setStartBefore}
       sets the start} of the range [r] to be immediately before the node
      represented by the raw JavaScript value [node_jv]. *)

  val set_start_after : t -> Jv.t -> unit
  (** [set_start_after r node_jv]
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/setStartAfter}
       sets the start} of the range [r] to be immediately after the node
      represented by the raw JavaScript value [node_jv]. *)

  val collapse : t -> bool -> unit
  (** [collapse r to_start]
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/collapse}
       collapses} the range [r] to one of its boundary points. If [to_start] is
      [true], collapses to the start point; otherwise, collapses to the end
      point. *)

  val select_node : t -> Jv.t -> unit
  (** [select_node r node_jv]
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/selectNode} sets
       the range [r]} to contain the node represented by the raw JavaScript
      value [node_jv] and all of its contents. *)

  val select_node_contents : t -> Jv.t -> unit
  (** [select_node_contents r node_jv]
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/selectNodeContents}
       sets the range [r]} to contain the contents within the node represented
      by the raw JavaScript value [node_jv]. *)

  val delete_contents : t -> unit
  (** [delete_contents r]
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/deleteContents}
       deletes} the contents of the range [r]. This removes the selected text
      from the document. *)

  val clone : t -> t
  (** [clone r] creates a
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/cloneRange}
       copy} of the range [r]. This is a new range object with the same start
      and end points. *)

  val to_string : t -> Jstr.t
  (** [to_string r] gets the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Range/toString} string
       representation} of the range [r]. This is the text content of the range.
  *)
end = struct
  type t = Jv.t

  let of_jv (jv : Jv.t) : t = jv
  let to_jv (r : t) : Jv.t = r

  (* These return Jv.t for nodes *)
  let start_container (r : t) : Jv.t = Jv.get r "startContainer"
  let end_container (r : t) : Jv.t = Jv.get r "endContainer"

  let common_ancestor_container (r : t) : Jv.t =
    Jv.get r "commonAncestorContainer"

  (* Getters for offsets/collapsed state *)
  let start_offset (r : t) : int = Jv.Int.get r "startOffset"
  let end_offset (r : t) : int = Jv.Int.get r "endOffset"
  let collapsed (r : t) : bool = Jv.Bool.get r "collapsed"

  (* Setters take Jv.t for nodes *)
  let set_start (r : t) (node_jv : Jv.t) (offset : int) : unit =
    Jv.call r "setStart" [| node_jv; Jv.of_int offset |] |> ignore

  let set_end (r : t) (node_jv : Jv.t) (offset : int) : unit =
    Jv.call r "setEnd" [| node_jv; Jv.of_int offset |] |> ignore

  let set_start_before (r : t) (node_jv : Jv.t) : unit =
    Jv.call r "setStartBefore" [| node_jv |] |> ignore

  let set_start_after (r : t) (node_jv : Jv.t) : unit =
    Jv.call r "setStartAfter" [| node_jv |] |> ignore

  let collapse (r : t) (to_start : bool) : unit =
    Jv.call r "collapse" [| Jv.of_bool to_start |] |> ignore

  let select_node (r : t) (node_jv : Jv.t) : unit =
    Jv.call r "selectNode" [| node_jv |] |> ignore

  let select_node_contents (r : t) (node_jv : Jv.t) : unit =
    Jv.call r "selectNodeContents" [| node_jv |] |> ignore

  let delete_contents (r : t) : unit = Jv.call r "deleteContents" [||] |> ignore
  let clone (r : t) : t = Jv.call r "cloneRange" [||] |> of_jv
  let to_string (r : t) : Jstr.t = Jv.call r "toString" [||] |> Jv.to_jstr
end

module Window = struct
  include Window

  (** [get_selection window] gets the
      {{:https://developer.mozilla.org/en-US/docs/Web/API/Window/getSelection}
       selection} object for the given [window]. *)
  let get_selection (window : Window.t) : Selection.t option =
    let sel = Jv.call (to_jv window) "getSelection" [||] in
    if Jv.is_null sel || Jv.is_undefined sel then None
    else Some (Selection.of_jv sel)
end

module Document = struct
  include Document

  let create_range (doc : Document.t) : Range.t =
    Jv.call (to_jv doc) "createRange" [||] |> Range.of_jv

  let get_selection (doc : Document.t) : Selection.t option =
    let sel = Jv.call (to_jv doc) "getSelection" [||] in
    if Jv.is_null sel || Jv.is_undefined sel then None
    else Some (Selection.of_jv sel)
end
