(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The operands of each row: generators of the points at which a law runs.

    An {!instance} applies one row through {!Nx.Op.eval} to the operands that
    take a tangent, the others captured, with the row's static arguments fixed.
    A point is drawn inside the row's domain: away from every kink by at least
    [1e-3] for {!t.smooth}, and anywhere the derivative's coefficients are
    finite for {!t.finite}, ties and zeros included. *)

type dtype = D : ('a, 'b) Nx.dtype -> dtype  (** A dtype, its type hidden. *)

val pp_dtype : Format.formatter -> dtype -> unit

type instance =
  | Instance : {
      f : ('a, 'b) Nx.t list -> ('c, 'd) Nx.t list;
          (** The row, at the operands that take a tangent. *)
      x : ('a, 'b) Nx.t list;  (** The point. *)
      linear : (('a, 'b) Nx.t list -> ('c, 'd) Nx.t list) option;
          (** For a row linear in the operands [f] takes, the map its tangent
              is: [f] with the constant operands that are not coefficients
              replaced by zeros, and [Pad]'s fill by zero. *)
      extra : Row.t list option;
          (** The rows [f] issues besides its own, such as the casts around an
              integer operation, or [None] when [f] goes on to compute forms of
              the result its sign or phase cancels from. *)
      pp : Format.formatter -> unit;
          (** The static arguments, the constants and the point. *)
    }
      -> instance

val pp_instance : Format.formatter -> instance -> unit

(** What a row's derivative is. *)
type kind =
  | Tangent  (** It has a tangent rule. *)
  | Plain  (** Its result is integer or boolean, with no tangent. *)
  | Integer  (** It takes integers only, so it never meets a tangent. *)

type t = {
  row : Row.t;
  kind : kind;
  difference : float * float;
      (** The step and the relative tolerance of a central difference of the
          row: [(1e-6, 1e-8)], and wider for the spectral factorizations, whose
          vectors are judged through derived forms, and [eig], which computes in
          complex64. *)
  smooth : dtype -> instance Windtrap.Gen.t;
      (** Points inside the smooth real domain, at a real or complex dtype whose
          values are cast from float64 ones. *)
  finite : instance Windtrap.Gen.t;
      (** Float64 points with finite coefficients, kinks included. *)
  complex : instance Windtrap.Gen.t option;
      (** Complex128 points inside the smooth complex domain, both components
          drawn, for a row nx computes on complex values. *)
  dtypes : dtype list;  (** The dtypes {!smooth} takes. *)
  derivative : (float -> float) option;
      (** The derivative of an elementwise row of one operand, in OCaml floats.
      *)
}

val of_row : Row.t -> t
(** [of_row r] is [r]'s case. *)

val all : t list
(** [all] is {!of_row} of every row of {!Row.all} with a tensor result: every
    row but [Read], whose result is a buffer, and [Check], which has none. *)

val float64 : dtype
