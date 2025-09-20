(** Auxiliary information dictionaries for environment transitions.

    Info dictionaries attach metadata to observations and transitions, such as
    diagnostic information, intermediate values, or episode statistics. They use
    a schemaless key-value structure similar to JSON, allowing flexible data
    passing without rigid type constraints.

    {1 Usage}

    Create and populate info dictionaries:
    {[
      let info =
        Info.empty
        |> Info.set "episode_length" (Info.int 42)
        |> Info.set "success" (Info.bool true)
    ]}

    Retrieve values:
    {[
      match Info.find "episode_length" info with
      | Some (Int n) -> Printf.printf "Episode length: %d\n" n
      | _ -> ()
    ]}

    Merge info from different sources:
    {[
      let combined = Info.merge env_info wrapper_info
    ]} *)

(** Universal value type for info dictionaries.

    Supports common data types with arbitrary nesting. *)
type value =
  | Null  (** Null/None value *)
  | Bool of bool  (** Boolean value *)
  | Int of int  (** Integer value *)
  | Float of float  (** Floating-point value *)
  | String of string  (** String value *)
  | List of value list  (** List of values *)
  | Dict of (string * value) list  (** Nested dictionary *)

type t
(** Immutable key-value dictionary for auxiliary information. *)

val empty : t
(** [empty] is the empty info dictionary with no entries. *)

val is_empty : t -> bool
(** [is_empty info] checks whether [info] contains any entries. *)

val singleton : string -> value -> t
(** [singleton key value] creates an info dictionary with a single entry. *)

val set : string -> value -> t -> t
(** [set key value info] returns a new dictionary with [key] bound to [value].

    If [key] already exists, its value is replaced. *)

val update : string -> (value option -> value option) -> t -> t
(** [update key f info] updates the value at [key] using function [f].

    [f] receives [Some old_value] if [key] exists, [None] otherwise. If [f]
    returns [Some new_value], [key] is bound to [new_value]; if [None], [key] is
    removed. *)

val find : string -> t -> value option
(** [find key info] looks up [key] in [info].

    Returns [Some value] if [key] exists, [None] otherwise. *)

val get_exn : string -> t -> value
(** [get_exn key info] retrieves the value at [key].

    @raise Not_found if [key] doesn't exist. Use {!find} for safe lookup. *)

val merge : t -> t -> t
(** [merge info1 info2] combines two dictionaries.

    Keys from [info2] override those in [info1] when both are present. *)

val to_list : t -> (string * value) list
(** [to_list info] converts [info] to an association list. *)

val of_list : (string * value) list -> t
(** [of_list entries] creates an info dictionary from an association list. *)

val null : value
(** [null] constructs a null value. *)

val bool : bool -> value
(** [bool b] constructs a boolean value. *)

val int : int -> value
(** [int n] constructs an integer value. *)

val float : float -> value
(** [float x] constructs a float value. *)

val string : string -> value
(** [string s] constructs a string value. *)

val list : value list -> value
(** [list values] constructs a list value. *)

val dict : (string * value) list -> value
(** [dict entries] constructs a nested dictionary value. *)

val to_yojson : t -> Yojson.Safe.t
(** [to_yojson info] serializes [info] to JSON. *)

val of_yojson : Yojson.Safe.t -> (t, string) result
(** [of_yojson json] deserializes [info] from JSON.

    Returns [Error msg] if the JSON structure is invalid. *)
