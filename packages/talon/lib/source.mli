(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Sources: tables read in batches.

    [Talon.Source] documents the contract between talon and a source's author.
    This interface adds the representation of sources. *)

type answer = Exact | Inexact | Unsupported

module Pred : sig
  type value = Value : 'a Type.t * 'a -> value

  type t =
    | Cmp of string * [ `Eq | `Ne | `Lt | `Le | `Gt | `Ge ] * value
    | In of string * value list
    | Null of string
    | Valid of string
    | And of t list
    | Or of t list
    | Not of t
end

type request = {
  columns : string list;
  filters : Pred.t list;
  limit : int option;
}

type reader = {
  next : unit -> (Table.t option, Error.t) result;
  close : unit -> unit;
}

type part = { rows : int option; open_ : unit -> (reader, Error.t) result }

type t = private {
  name : string;  (** What plans and errors call the source. *)
  schema : Schema.t;  (** The columns the source yields. *)
  rows : int option;
      (** The number of rows the source yields for a request without filters, if
          known. *)
  sorted : Order.t list;
      (** The order the source's rows come in. Empty when no order is claimed.
      *)
  pushdown : Pred.t -> answer;  (** The source's answer about a conjunct. *)
  parts : request -> (part list, Error.t) result;
      (** The source's parts for a request. *)
}
(** The type for sources. A source is compared physically: two sources are the
    same iff they are one value. *)

val v :
  name:string ->
  schema:Schema.t ->
  ?rows:int ->
  ?sorted:Order.t list ->
  ?pushdown:(Pred.t -> answer) ->
  (request -> (part list, Error.t) result) ->
  t
