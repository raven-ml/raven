(** Witnesses, generators and golden cells of {!Tolk_next.Dtype}.

    Witnesses print and compare data types and values from tables of their own,
    and the generators draw integers from bounds of their own. Float values of a
    data type are drawn through {!Tolk_next.Dtype.truncate}, which the Dtype
    suite checks on its own. *)

open Tolk_next

(** {1:witnesses Witnesses} *)

val dtype : Dtype.t Windtrap.testable
(** [dtype] compares data types structurally, orders them by {!Dtype.compare}
    and prints them as tinygrad does, [dtypes.half]. *)

val value : Dtype.value Windtrap.testable
(** [value] compares values by constructor and payload, floats bit for bit
    except that every NaN equals every NaN. *)

val const : Dtype.const Windtrap.testable
(** [const] is {!value} with [`Invalid] equal only to itself. *)

val z : Z.t Windtrap.testable
(** [z] compares integers. *)

(** {1:lists Data types} *)

val declared : Dtype.t list
(** [declared] is every data type, in the order {!Dtype.t} declares them. *)

val alias : Dtype.t -> string
(** [alias dt] is the name that tinygrad prints [dt] with, without its [dtypes.]
    prefix: [half] for {!Dtype.Float16}, [char] for {!Dtype.Int8}. *)

(** {1:generators Generators} *)

val every : Dtype.t Windtrap.Gen.t
(** [every] draws any data type, {!Dtype.Void} included. *)

val promotable : Dtype.t Windtrap.Gen.t
(** [promotable] draws any data type but {!Dtype.Void}. *)

val stored : Dtype.t Windtrap.Gen.t
(** [stored] draws a data type a program stores: a float or an integer of known
    width, or {!Dtype.Bool}. *)

val value_of : Dtype.t -> Dtype.value Windtrap.Gen.t
(** [value_of dt] draws a value of [dt], its bounds and their neighbours
    included. The floats are drawn as any float truncated to [dt].

    Raises [Invalid_argument] if [dt] is {!Dtype.Void}. *)

val finite_float : float Windtrap.Gen.t
(** [finite_float] draws a finite float, weighted towards the ranges of the
    narrow floats. *)

val integer : Z.t Windtrap.Gen.t
(** [integer] draws an integer of any magnitude up to [2{^ 1100}], the edges of
    every integer data type and their neighbours included. *)

(** {1:bounds Bounds} *)

val int_bounds : Dtype.t -> Z.t * Z.t
(** [int_bounds dt] is the least and the greatest value of the integer data type
    [dt], {!Dtype.Weak_int} included.

    Raises [Invalid_argument] if [dt] is not an integer data type. *)

(** {1:cells Golden cells}

    Cells as tinygrad prints them. Each raises [Invalid_argument] on a cell it
    cannot read. *)

val dtype_of_cell : string -> Dtype.t
(** [dtype_of_cell s] is the data type printed [s], such as [dtypes.half]. *)

val value_of_cell : string -> Dtype.value
(** [value_of_cell s] is the value printed [s]: [True], [False], an integer in
    decimal, or a float as [repr] or [float.hex] print it. [-nan] is a NaN with
    its sign bit set. *)

val const_of_cell : string -> Dtype.const
(** [const_of_cell s] is {!value_of_cell}, and [`Invalid] for [Invalid]. *)

val consts_of_cell : string -> Dtype.const list
(** [consts_of_cell s] is the list printed [s], such as [[True, 2, 3.0]]. *)
