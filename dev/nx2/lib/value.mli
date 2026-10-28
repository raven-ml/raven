(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values and operations: nx's central types, defined together because each
    mentions the other. This module has no implementation.

    Every value has a placement. A value is concrete, arrays on the devices
    of its placement, or an interpretation's stand-in, whose form names the
    placement its eager twin has. The brand ['d] is phantom ({!Devices}),
    and every function that makes a value checks that its arrays lie on its
    placement's devices.

    [Nx.Prim] re-exports the operations and the types of forms, payloads and
    reaches, and states what each means. This module documents what the
    engine adds: the representations of values, their deaths and the fields
    of an interpretation. It lies above {!Devices}; {!Prim} holds the
    functions over its types. *)

type ('v, 's) dtype = ('v, 's) Nx_array.Dtype.t

type ('v, 's, +'d) payload = ..

(** {1:values Values} *)

type death = Live | Died of string  (** Given to the function named. *)
(** Whether a value lives. Values are made [Live]. The consumer of a
    donation sets its donor's word to [Died], for good, by one
    compare-and-set from [Live] ({!Donate.val-claim}). *)

type ('v, 's, 'd) t =
  | Array of {
      at : 'd Devices.placement;
      a : ('v, 's) Nx_array.t;
      mutable dead : death; [@atomic]
    }
      (** On [at]'s one device. *)
  | Shards of {
      at : 'd Devices.placement;
      shape : int array;  (** The whole's. *)
      arrays : ('v, 's) Nx_array.t iarray;
      mutable dead : death; [@atomic]
    }
      (** On [at]'s devices, two or more: per device, in
          [Grid.devices (Devices.grid at)]'s order, its window of the value
          ({!Grid.window}). *)
  | Handle of {
      donor : ('v, 's, 'd) t;
          (** The [Array] or [Shards] donated, never a handle. Two handles of
              one donor, and a donor beside its handle, are told by [==] on
              it. *)
      over : ('v, 's, 'd) t;
          (** The [Array] or [Shards] the handle reads: [donor] as donated, or
              the result of the movement that passed the handle on, a view of
              the donor's memory or a copy of it. *)
    }
      (** A handle one operation reads. Its placement, dtype, shape and arrays
          are [over]'s, and its death is its donor's: it lives while its donor
          does, as does every other handle of the donor, those a movement
          passed on included. *)
  | Traced of {
      form : ('v, 's, 'd) form;
      owner : interpretation;
      payload : ('v, 's, 'd) payload;
      twin : ('v, 's, 'd) t option;
          (** A value of [form] that a read reads while [owner] is around
              it. *)
    }  (** An interpretation's stand-in: its form, and what [owner] keeps. *)

and ('v, 's, 'd) form = {
  dtype : ('v, 's) dtype;
  layout : Nx_array.Layout.t;
  placement : 'd Devices.placement;
}

and 'd any = Any : ('v, 's, 'd) t -> 'd any

(** {1:interpretations Interpretations} *)

and reach = Values | Extent | Adopting of interpretation

and interpretation = {
  name : string;  (** In messages, as ["Rune.grad"]. *)
  reach : reach;
  rule : 'r. interpretation -> by:string -> 'r prim -> 'r;
      (** What it makes of the operations it receives. *)
  live : bool Atomic.t;  (** Cleared when its extent ends. *)
}
(** An interpretation. Only {!Eval} makes one and writes [live]. *)

(** {1:operations Operations} *)

and 'd load =
  | Plain : ('v, 's, 'd) t -> 'd load
  | Padded : {
      x : ('v, 's, 'd) t;
      fill : 'v;
      pad : Nx_kernel.pad;
    }
      -> 'd load

(* Declared before [reductions] and [outs]: an unannotated [Prim.[ ... ]] is the
   list type declared last, [outs]. *)
and ('d, _) data =
  | [] : ('d, unit) data
  | ( :: ) : ('v, 's, 'd) t * ('d, 'e) data -> ('d, 'v * 'e) data

and ('d, _) reduction =
  | Monoid :
      Nx_kernel.monoid * int * ('v, 's) dtype
      -> ('d, ('v, 's, 'd) t) reduction
  | Logsumexp : int * ('v, 's) dtype -> ('d, ('v, 's, 'd) t) reduction
  | Moments :
      int * ('v, 's) dtype
      -> ('d, ('v, 's, 'd) t * ('v, 's, 'd) t) reduction
  | Arg :
      Nx_kernel.extreme * int * ('v, 's) dtype
      -> ( 'd,
           ('v, 's, 'd) t * (int64, Nx_array.Dtype.int64_elt, 'd) t )
         reduction

and ('d, _) reductions =
  | [] : ('d, unit) reductions
  | ( :: ) :
      ('d, 'a) reduction * ('d, 'r) reductions
      -> ('d, 'a * 'r) reductions

and ('d, _) outs =
  | [] : ('d, unit) outs
  | ( :: ) : ('v, 's) dtype * ('d, 'r) outs -> ('d, ('v, 's, 'd) t * 'r) outs

and 'd index = (int64, Nx_array.Dtype.int64_elt, 'd) t
(** Positions held in data. *)

and 'd scale =
  | Exponent of (int, Nx_array.Dtype.int8_unsigned_elt, 'd) t
  | Value : ('v, 's, 'd) t -> 'd scale
(** A contraction operand's block scale ({!Nx_kernel.type-scale}). *)

and _ prim =
  | Map : {
      layout : Nx_array.Layout.t;
      prog : Nx_kernel.Prog.t;
      outs : ('d, 'r) outs;
      loads : 'd load iarray;
      at : 'd Devices.placement option;
          (** Where a map of no loads lies; [None] for a map with loads. *)
    }
      -> 'r prim
  | Reduce : {
      layout : Nx_array.Layout.t;
      axes : int iarray;
      prog : Nx_kernel.Prog.t;
      reductions : ('d, 'r) reductions;
      loads : 'd load iarray;
    }
      -> 'r prim
  | Scan : {
      layout : Nx_array.Layout.t;
      axis : int;
      prog : Nx_kernel.Prog.t;
      reduction : ('d, 'r) reduction;
      loads : 'd load iarray;
      carry : 'd any option;
    }
      -> 'r prim
  | Gather : {
      axis : int;
      idx : 'd index;
      x : ('v, 's, 'd) t;
    }
      -> ('v, 's, 'd) t prim
  | Scatter : {
      combine : Nx_kernel.combine;
      unique : bool;
      axis : int;
      idx : 'd index;
      updates : ('v, 's, 'd) t;
      into : ('v, 's, 'd) t;
    }
      -> ('v, 's, 'd) t prim
  | Sort : {
      axis : int;
      descending : bool;
      k : int option;
      x : ('v, 's, 'd) t;
    }
      -> (('v, 's, 'd) t * 'd index) prim
  | Fold : {
      shape : int iarray;
      pad : Nx_kernel.pad;
      x : ('v, 's, 'd) t;
    }
      -> ('v, 's, 'd) t prim
  | Assemble : {
      dtype : ('v, 's) dtype;
      shape : int iarray;
      fill : 'v;
      pieces : (Nx_array.Move.range iarray * ('v, 's, 'd) t) list;
    }
      -> ('v, 's, 'd) t prim
  | Contract : {
      batch : (int * int) iarray;
      contracting : (int * int) iarray;
      acc : Nx_array.Dtype.any;
      out : ('v, 's) dtype;
      a : ('a, 'b, 'd) t;
      b : ('c, 'e, 'd) t;
      init : ('v, 's, 'd) t option;
      a_scale : 'd scale option;
      b_scale : 'd scale option;
      segments : 'd index Nx_kernel.segments option;
    }
      -> ('v, 's, 'd) t prim
  | Fft : ('d, 'r) fft -> 'r prim
  | Linalg : ('d, 'r) linalg -> 'r prim
  | Copy : ('v, 's, 'd) t -> ('v, 's, 'd) t prim
  | Move : Nx_array.Move.t * ('v, 's, 'd) t -> ('v, 's, 'd) t prim
  | Bitcast : ('w, 'r) dtype * ('v, 's, 'd) t -> ('w, 'r, 'd) t prim
  | Place : 'e Devices.placement * ('v, 's, 'd) t -> ('v, 's, 'e) t prim
  | Check : {
      ok : (bool, Nx_array.Dtype.bool_elt, 'd) t;
      data : ('d, 'e) data;
      fail : int array -> 'e -> exn;
    }
      -> unit prim

and ('d, _) fft =
  | C2c : {
      direction : Nx_kernel.direction;
      axes : int iarray;
      x : (Complex.t, 's, 'd) t;
    }
      -> ('d, (Complex.t, 's, 'd) t) fft
  | R2c : {
      dtype : (Complex.t, 'c) dtype;
      axes : int iarray;
      x : (float, 's, 'd) t;
    }
      -> ('d, (Complex.t, 'c, 'd) t) fft
  | C2r : {
      dtype : (float, 'r) dtype;
      n : int;
      axes : int iarray;
      x : (Complex.t, 's, 'd) t;
    }
      -> ('d, (float, 'r, 'd) t) fft

and ('d, _) linalg =
  | Cholesky : {
      triangle : Nx_kernel.triangle;
      a : ('v, 's, 'd) t;
    }
      -> ('d, ('v, 's, 'd) t) linalg
  | Lu : ('v, 's, 'd) t -> ('d, ('v, 's, 'd) t * 'd index * 'd index) linalg
  | Qr : {
      factors : Nx_kernel.factors;
      a : ('v, 's, 'd) t;
    }
      -> ('d, ('v, 's, 'd) t * ('v, 's, 'd) t) linalg
  | Svd_values : {
      dtype : (float, 'r) dtype;
      a : ('v, 's, 'd) t;
    }
      -> ('d, (float, 'r, 'd) t) linalg
  | Svd : {
      factors : Nx_kernel.factors;
      dtype : (float, 'r) dtype;
      a : ('v, 's, 'd) t;
    }
      -> ('d, (float, 'r, 'd) t * ('v, 's, 'd) t * ('v, 's, 'd) t) linalg
  | Eigh_values : {
      dtype : (float, 'r) dtype;
      a : ('v, 's, 'd) t;
    }
      -> ('d, (float, 'r, 'd) t) linalg
  | Eigh : {
      dtype : (float, 'r) dtype;
      a : ('v, 's, 'd) t;
    }
      -> ('d, (float, 'r, 'd) t * ('v, 's, 'd) t) linalg
  | Eig_values : {
      dtype : (Complex.t, 'c) dtype;
      a : ('v, 's, 'd) t;
    }
      -> ('d, (Complex.t, 'c, 'd) t) linalg
  | Eig : {
      dtype : (Complex.t, 'c) dtype;
      a : ('v, 's, 'd) t;
    }
      -> ('d, (Complex.t, 'c, 'd) t * (Complex.t, 'c, 'd) t) linalg
  | Solve_triangular : {
      triangle : Nx_kernel.triangle;
      adjoint : bool;
      unit_diagonal : bool;
      a : ('v, 's, 'd) t;
      b : ('v, 's, 'd) t;
    }
      -> ('d, ('v, 's, 'd) t) linalg
