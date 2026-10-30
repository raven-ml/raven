(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Markdown notebook format.

    Parses markdown files into {!Quill.Doc.t} and renders documents back to
    markdown. Code blocks become code cells; everything else becomes text cells.

    The notebook holds an image output as an [<img>] tag. An image stored
    {e inline} is a data URI in the tag's [src]. An image stored {e as a file}
    lives in [figures/<stem>/] beside the notebook, where [<stem>] is the
    notebook's file name without its extension, and [src] is the file's path
    relative to the notebook's directory. The [n]th image of the cell [id] is
    the file [<id>.<ext>] for [n = 1] and [<id>-<n>.<ext>] after, where [<ext>]
    follows the image's MIME type, as [svg] for [image/svg+xml]. A notebook that
    stores its images as files says so in its first line,
    [<!-- quill:figures files -->]. For [nb.md]:
    {[
    <!-- quill:figures files -->

    ...
    <!-- out:display image/svg+xml -->
    <img src="figures/nb/c_x.svg">
    ]}

    Such a notebook owns [figures/<stem>/], whose content {!save} replaces.
    Files elsewhere, such as images that text cells show, are never touched.
    {!of_string} and {!to_string_with_outputs} work on strings alone and store
    images inline; {!load} and {!save} read and write both ways. *)

(** {1:strings Strings} *)

val of_string : string -> Quill.Doc.t
(** [of_string s] parses markdown string [s] into a document.

    Fenced code blocks with a language info string become code cells. All other
    content between code blocks is merged into text cells. Adjacent non-code
    content forms a single text cell.

    Raises [Invalid_argument] if [s] stores its images as files, or holds an
    image stored as a file, which only {!load} reads. The message names the cell
    and the reference. *)

val to_string : Quill.Doc.t -> string
(** [to_string doc] renders [doc] as a markdown string.

    Text cells are emitted verbatim. Code cells are rendered as code blocks
    fenced with one more backtick than the longest run of backticks that begins
    a line of the source after its leading spaces or tabs, and at least three.
    Cell outputs are not included. *)

val to_string_with_outputs : Quill.Doc.t -> string
(** [to_string_with_outputs doc] renders [doc] as markdown with outputs.

    Like {!to_string} but code cell outputs are serialized between
    [<!-- quill:output -->] and [<!-- /quill:output -->] comment markers after
    each code block. Image outputs are stored inline, HTML outputs as inline
    HTML. *)

(** {1:files Files} *)

(** The type for how a notebook stores its images. *)
type figures =
  | Inline  (** Images are stored in the notebook. *)
  | Files  (** Images are stored as files in [figures/<stem>/]. *)

val load : ?outputs:bool -> string -> (Quill.Doc.t * figures, string) result
(** [load path] is the notebook in the file [path], read as {!of_string} does,
    and [Files] if its first line is [<!-- quill:figures files -->], [Inline]
    otherwise. Each image stored as a file is read from its file, in
    [figures/<stem>/] beside the file [path] names, through links.

    [outputs] defaults to [true]. With [false], code cells have no outputs and
    no image file is read, for a caller that discards the outputs, such as one
    that runs the notebook again.

    The result is [Error msg] if [path] cannot be read or, with [outputs], if an
    image reference is not a file name in [figures/<stem>/] or its file is
    missing, unreadable or not a regular file. A message about an image names
    [path], the cell and the reference, as in
    ["nb.md: cell c_x: figure figures/nb/c_x.svg: no such file"]. *)

type stamp
(** A {e stamp} identifies a file's content as stored: its device, inode, size
    and modification time. A file written again, or replaced by a copy that kept
    an older time, has another stamp. Two stamps are compared with [=]. *)

val stamp : string -> stamp option
(** [stamp path] is the stamp of the file [path] names, through links, or [None]
    if there is none. *)

val save : figures -> string -> Quill.Doc.t -> (stamp, string) result
(** [save figures path doc] writes [doc] with its outputs to the file [path]
    names, through a link too, storing its images as [figures] says, and is the
    stamp of the notebook it wrote. A caller compares it with {!stamp} to tell
    whether the file changed since.

    Each file is written to a temporary file beside it, synced and renamed over
    it, so a crash leaves the old file or the new one, and may leave the
    temporary file. A new file has the permissions the umask allows; a replaced
    one keeps its own.

    With [Inline], nothing but [path] is written. With [Files], [path] opens
    with [<!-- quill:figures files -->], and afterwards [figures/<stem>/] beside
    the file holds the images of [doc] and nothing else. The files of images
    [doc] no longer has are removed, then the directory if it is empty, then
    [figures/] if that leaves it empty. Each figure is on disk before the
    notebook that references it, and the notebook before the removals.

    The result is [Error msg] if [path] exists and is read-only, if a file
    cannot be written or, with [Files] and before anything is written, if the
    notebook's name leaves no stem to name its figures directory by, if
    [figures] or [figures/<stem>] exists and is not a directory, if a cell id
    does not make a file name, such as one holding ['/'], or if two images make
    the same one. *)

module Edit : module type of Edit
