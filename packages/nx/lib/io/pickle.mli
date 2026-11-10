(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pickles decoded with their calls as data. [Nx_io.load_pytorch] states the
    language accepted, and this module the machine's own rules.

    The decoded value is a graph of {e nodes}, each made by one opcode and
    numbered from 0 in the order the opcodes run. A node is {e complete} once
    another node holds it: an opcode that sets a state, items or entries on a
    complete node is refused. So the graph is acyclic, and each call's
    arguments, state, items and entries are the ones Python passes when it runs
    the pickle. Nothing is ever called. Decoding takes time and memory linear in
    the pickle's length, and recurses nowhere.

    Strings, bytes, integers outside [int]'s range and globals carry an
    {e atom}: a number that two values of one kind share exactly when their
    contents are equal, so a key built of them compares in constant time. *)

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type value =
  | Null
  | Bool of bool
  | Int of int
  | Long of string * int
      (** Outside [int]'s range: its two's complement, little-endian, in the
          fewest bytes. *)
  | Float of float
  | String of string * int
  | Bytes of string * int
  | Global of string * int
      (** A global's dotted name, Python 2's [__builtin__] and [copy_reg] read
          as [builtins] and [copyreg] below protocol 3, as Python 3 reads them.
      *)
  | Node of node

and node = private {
  id : int;
  at : int;  (** The byte of the opcode that made the node. *)
  kind : kind;
  mutable complete : bool;
}

(** The kinds of nodes. A dict's entries keep their order, and a key given twice
    twice. *)
and kind =
  | Tuple of value array
  | Frozenset of value array
  | Bytearray of string
  | List of value Dynarray.t
  | Dict of (value * value) Dynarray.t
  | Set of value Dynarray.t
  | Persistent of value  (** A persistent id, as [BINPERSID] gives it. *)
  | Call of call

and call = private {
  fn : value;
  args : value;
  kwargs : value option;
  new_ : bool;
  mutable state : value option;
  items : value Dynarray.t;
  entries : (value * value) Dynarray.t;
}
(** The type for calls: [fn] applied to the tuple [args], then [state], [items]
    and [entries] set on the result. A [NEWOBJ] is [new_] of its class [fn], and
    a [NEWOBJ_EX] passes [kwargs] too. *)

val integer : string -> (int, string) result
(** [integer s] is the integer whose two's complement, little-endian, is [s], as
    LONG1 holds one: [Ok i] within [int]'s range, and otherwise [Error b], [b]
    the fewest bytes that hold it. *)

type decoded = {
  value : value;
  stop : int;  (** The position after the pickle's [STOP]. *)
  nodes : int;  (** The number of nodes. *)
  atoms : (string, int) Hashtbl.t;
}

val decode : bytes -> pos:int -> decoded
(** [decode b ~pos] is the pickle at [pos] in [b]. The [Dynarray]s of its nodes
    must not be changed.

    Raises [Failure] naming the byte, counted from [pos], if the pickle is
    malformed or uses an opcode [Nx_io.load_pytorch] refuses. *)

val atom : decoded -> char -> string -> int
(** [atom d kind s] is the atom of the string [s] of [kind] in [d]: ['S'] a
    string, ['B'] bytes, ['L'] a [Long]'s bytes and ['G'] a global's name. *)
