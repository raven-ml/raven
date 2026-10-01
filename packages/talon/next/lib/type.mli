(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Column types.

    [Talon_next.Type] documents column types, their values, their order and
    their formatting. This interface adds the equation of {!ext} and the
    {{!internal}functions} that serve the planner. *)

type unit_ = S | Ms | Us | Ns
type ext = Kind.ext

type 'a t = private
  | Bool : bool t
  | Int8 : int t
  | Int16 : int t
  | Int32 : int t
  | Int64 : int t
  | Uint8 : int t
  | Uint16 : int t
  | Uint32 : int t
  | Uint64 : int t
  | Float16 : float t
  | Float32 : float t
  | Float64 : float t
  | Decimal : { precision : int; scale : int } -> Decimal.t t
  | String : string t
  | Binary : Binary.t t
  | Categorical : string iarray -> string t
  | Date : Time.date t
  | Clock : unit_ -> Time.span t
  | Duration : unit_ -> Time.span t
  | Datetime : { unit_ : unit_; zone : string option } -> Time.instant t
  | List : 'a t -> 'a array t
  | Record : (string * any) list -> Record.t t
  | Tensor : ('a, 'b) Nx.dtype * int iarray -> ('a, 'b) Nx.t t
  | Ext : { name : string; metadata : string; storage : 's t } -> ext t

and any = Any : 'a t -> any

val bool : bool t
val int8 : int t
val int16 : int t
val int32 : int t
val int64 : int t
val uint8 : int t
val uint16 : int t
val uint32 : int t
val uint64 : int t
val float16 : float t
val float32 : float t
val float64 : float t
val decimal : precision:int -> scale:int -> Decimal.t t
val string : string t
val binary : Binary.t t
val categorical : string array -> string t
val date : Time.date t
val clock : unit_ -> Time.span t
val duration : unit_ -> Time.span t
val datetime : ?zone:string -> unit_ -> Time.instant t
val list : 'a t -> 'a array t
val record : (string * any) list -> Record.t t
val tensor : ('a, 'b) Nx.dtype -> int array -> ('a, 'b) Nx.t t
val ext : name:string -> ?metadata:string -> 'a t -> ext t
val kind : 'a t -> 'a Kind.t
val holds : 'a t -> 'a -> bool
val compare_value : 'a t -> 'a -> 'a -> int
val common : 'a t list -> 'a t option
val equal : 'a t -> 'b t -> bool
val pp : Format.formatter -> 'a t -> unit

(** {1:internal Internal} *)

val has_ext : 'a t -> bool
(** [has_ext t] is [true] iff [t] is or contains an extension type, as a list
    element or a record field at any depth. An extension type orders only
    through its declaration, and a type that holds one has no order, so sort
    keys and ordering reductions refuse both. *)

val pp_name : Format.formatter -> string -> unit
(** [pp_name ppf n] formats the field name [n] as {!pp} does. *)

val pp_quoted : Format.formatter -> string -> unit
(** [pp_quoted ppf s] formats [s] quoted, as {!pp} quotes a category. Plans and
    their reports quote column names with it, so that a name in UTF-8 prints as
    it reads. *)

val pp_list :
  (Format.formatter -> 'a -> unit) -> Format.formatter -> 'a list -> unit
(** [pp_list pp ppf vs] formats [vs] as an OCaml list, [["a"; "b"]], as plans
    write lists. *)
