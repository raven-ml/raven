(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The rows of the rule tables: one per constructor of {!Nx.Op.t} and kind. *)

type move = Reshape | Expand | Permute | Shrink | Flip | Window

type t =
  | Unary of Nx_backend.unary
  | Binary of Nx_backend.binary
  | Compare of Nx_backend.compare
  | Where
  | Fma
  | Reduce of Nx_backend.reduce
  | Scan of Nx_backend.reduce
  | Arg_reduce of Nx_backend.arg_reduce
  | Sort
  | Argsort
  | Group
  | Pad
  | Cat
  | Cast
  | Bitcast
  | Threefry
  | Gather
  | Scatter of Nx_backend.scatter
  | Update
  | Unfold
  | Fold
  | Matmul
  | Fft
  | Rfft
  | Irfft
  | Contiguous
  | Cholesky
  | Qr
  | Lu
  | Svd
  | Eig
  | Eigh
  | Solve_triangular
  | Move of move
  | Place
  | Read
  | Check

val of_op : 'r Nx.Op.t -> t
(** [of_op op] is [op]'s row. *)

val issued : (unit -> 'a) -> (t * string) list
(** [issued f] is the row and {!Nx.Op.name} of each operation [f ()] issues, in
    order. *)

val all : t list
(** [all] is every row, each once. *)

val name : t -> string
(** [name r] names [r] as a test group does, such as ["unary sin"]. *)

val equal : t -> t -> bool
val compare : t -> t -> int
val pp : Format.formatter -> t -> unit
