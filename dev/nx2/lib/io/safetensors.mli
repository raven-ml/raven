(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The SafeTensors format. *)

val code : ('v, 's) Nx_array.Dtype.t -> (string * int) option
(** [code dt] is [dt]'s name in the format and its rank among the format's
    dtypes, by which a file orders its arrays, widest first; [None] if the
    format has no [dt]. *)

val check_name : by:string -> string -> unit
(** [check_name ~by name] is [()] if {!load} accepts [name] as an array's name.

    Raises [Invalid_argument] naming [by] and [name] if [name] is not UTF-8 or
    is the header's metadata key. *)

val load : Rig.Buffer.t -> Nx.host Archive.t
(** [load file] is the arrays of the disk buffer [file].

    Raises [Failure] if its header is malformed or the file's length is not the
    one the header describes. *)

val header : (string * string * int array * int) list -> string
(** [header entries] is the header of a file of [entries], each a name, a
    dtype's code, a shape and a length in bytes, whose data follows in that
    order. *)
