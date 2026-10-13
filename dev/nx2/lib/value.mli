(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values and operations: nx's central types, defined together because each
    mentions the other. This module has no implementation.

    A value is concrete, arrays on the devices of its placement; a constant not
    yet computed, a value of every set; or an interpretation's stand-in. The
    brand ['d] is phantom: every function that makes a value checks that its
    arrays lie on its placement's devices, and nothing at run time reads ['d]. A
    constant has no placement. *)

type ('v, 's) dtype = ('v, 's) Nx_array.Dtype.t

type ('v, 's, +'d) payload = ..
(** What an interpretation keeps in its traced values. A payload holds values,
    never a function of ['d]. *)

(** {1:values Values} *)

type death =
  | Live
  | Died of string  (** The function the value was given to. *)
(** Whether a value lives. Values are made [Live]; a death is claimed by
    [compare_and_set] from [Live]. *)

type ('v, 's, 'd) t =
  | Array of {
      at : 'd Devices.placement;
      a : ('v, 's) Nx_array.t;
      mutable dead : death; [@atomic]
    }
      (** On [at]'s one device. [dead] is [Died f] once the value was donated
          to [f] ({!Exec.donate}). *)
  | Shards of {
      at : 'd Devices.placement;
      arrays : ('v, 's) Nx_array.t iarray;
      mutable dead : death; [@atomic]
    }
      (** On [at]'s devices, two or more: per device, in
          [Grid.devices (Devices.grid at)]'s order, its window of the value. *)
  | Donated of {
      at : 'd Devices.placement;
      arrays : ('v, 's) Nx_array.t iarray;
      donor : death Atomic.Loc.t;  (** The donor's [dead]. *)
      mutable spent : death; [@atomic]
    }
      (** A handle one operation reads: [arrays] over its donor's memory, one
          per device of [at], or a copy of it a movement made. [spent] is
          [Live] until an operation reads it: the one that consumed it, killing
          the donor, or a movement that passed a new handle of the donor on.
      *)
  | Deferred of { form : ('v, 's, 'd) form; node : node; k : int }
      (** Result [k] of the constant operation [node]. *)
  | Traced of {
      form : ('v, 's, 'd) form;
      owner : interpretation;
      payload : ('v, 's, 'd) payload;
    }  (** An interpretation's stand-in: its form, and what [owner] keeps. *)

and ('v, 's, 'd) form = {
  dtype : ('v, 's) dtype;
  layout : Nx_array.Layout.t;
  placement : 'd Devices.placement option;  (** [None] for every set. *)
}
(** A value without its bytes. A sharded value's layout is the C-contiguous
    layout of the whole. *)

and node =
  | Node : {
      id : int;  (** Unique among a process's nodes. *)
      by : string;
      op : 'r prim;
      memo : (unit Devices.placement * Nx_array.any array array) list Atomic.t;
    }
      -> node
      (** An operation whose every operand is a constant, applied by [by].
          [memo] holds its results computed so far at the placements an
          operation read it at: per result, one array per device, of the
          device's window. A movement or a bitcast keeps none: an operation
          reads it as a view of its operand's results. *)

and 'd any = Any : ('v, 's, 'd) t -> 'd any

(** {1:interpretations Interpretations} *)

and reach =
  | Values  (** Reaches the operations on its traced values. *)
  | Extent
      (** Also reaches every operation its starting fiber applies inside its
          extent, on its domain. *)

and interpretation = {
  name : string;  (** In messages, as ["Rune.grad"]. *)
  reach : reach;
  rule : 'r. interpretation -> by:string -> 'r prim -> 'r;
      (** What it makes of the operations it receives. *)
  start : int;  (** Its order of start on [domain]. *)
  domain : Domain.id;  (** Where it started. *)
  live : bool Atomic.t;  (** Cleared when its extent ends. *)
  mutable running : bool;  (** Whether its rule runs, on [domain]. *)
  mutable formulas : (int, unit) Hashtbl.t option;
      (** Nodes of the formulas an [Extent]'s rule returned, made when it
          returns its first. *)
}
(** An interpretation. Only {!Interp} writes its fields. *)

(** {1:operations Operations}

    An operation's arrays are its own: whoever builds one from a caller's array
    copies it, so that nothing a caller changes later reaches it.

    A loop's loads have exactly its iteration shape: no operation broadcasts or
    promotes. *)

and 'd load =
  | Plain : ('v, 's, 'd) t -> 'd load
      (** How a loop reads an operand: through its layout. *)
  | Padded : {
      x : ('v, 's, 'd) t;
      fill : 'v;
      pad : Nx_kernel.pad;
    }
      -> 'd load
      (** [x] padded with [fill] and windowed as [pad] says
          ({!Nx_kernel.load}): the loop's shape is the padded one. *)

and ('d, _) reduction =
  | Monoid :
      Nx_kernel.monoid * int * ('v, 's) dtype
      -> ('d, ('v, 's, 'd) t) reduction
      (** Output [k] of a loop's program folded by the monoid in that output's
          dtype, rounded once to the dtype. *)
  | Moments :
      int * ('v, 's) dtype
      -> ('d, ('v, 's, 'd) t * ('v, 's, 'd) t) reduction
      (** The population mean of output [k], then its variance. *)
  | Arg :
      Nx_kernel.extreme * int * ('v, 's) dtype
      -> ( 'd,
           ('v, 's, 'd) t * (int64, Nx_array.Dtype.int64_elt, 'd) t )
         reduction
      (** The extreme of output [k], then its first position, in C order of the
          reduced indices. *)

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

and _ prim =
  | Map : {
      layout : Nx_array.Layout.t;
      prog : Nx_kernel.Prog.t;
      outs : ('d, 'r) outs;
      loads : 'd load array;
    }
      -> 'r prim
      (** [prog] at every index of [layout]'s shape, reading load [i] as its
          operand [i]; result [k] is its output [k], laid out as [layout], which
          is C-contiguous. A one-result map is ['v * unit]. With no loads, a
          creation. *)
  | Reduce : {
      layout : Nx_array.Layout.t;
      axes : int array;
      prog : Nx_kernel.Prog.t;
      reductions : ('d, 'r) reductions;
      loads : 'd load array;
    }
      -> 'r prim
      (** [prog] at every index of [layout]'s shape, as a map's, each reduction
          folding its output along [axes], strictly increasing. Results drop
          [axes] and are C-contiguous. *)
  | Scan : {
      layout : Nx_array.Layout.t;
      axis : int;
      prog : Nx_kernel.Prog.t;
      reduction : ('d, 'r) reduction;
      loads : 'd load array;
    }
      -> 'r prim
      (** The reduction of [prog]'s output over the indices along [axis] up to
          each index's, inclusive: [layout]'s shape, C-contiguous. *)
  | Gather : {
      axis : int;
      idx : 'd index;
      x : ('v, 's, 'd) t;
    }
      -> ('v, 's, 'd) t prim
      (** [x] read at the positions [idx] holds along [axis]: [idx] has [x]'s
          rank and its extents off [axis], and the result [idx]'s shape. A
          position outside [x]'s axis reads the element of zero bits. *)
  | Scatter : {
      combine : Nx_kernel.combine;
      unique : bool;
      axis : int;
      idx : 'd index;
      updates : ('v, 's, 'd) t;
      into : ('v, 's, 'd) t;
    }
      -> ('v, 's, 'd) t prim
      (** [into] with each update combined at its own index with axis [axis]
          replaced by [idx]'s element there, in C order of [updates]
          ({!Nx_kernel.S.scatter}); [idx] and [updates] have one shape,
          [into]'s off [axis]. [unique] promises distinct targets. *)
  | Sort : {
      axis : int;
      descending : bool;
      k : int option;
      x : ('v, 's, 'd) t;
    }
      -> (('v, 's, 'd) t * 'd index) prim
      (** [x]'s elements along [axis], stably, in {!Nx_kernel.S.sort}'s
          order, reversed where [descending], then their positions along [axis];
          with [Some k], the first [k] of each slice. *)
  | Fold : {
      shape : int array;
      pad : Nx_kernel.pad;
      x : ('v, 's, 'd) t;
    }
      -> ('v, 's, 'd) t prim
      (** The adjoint of a [Padded] load by [pad] of a value of [shape]
          ({!Nx_kernel.S.fold}). *)
  | Assemble : {
      dtype : ('v, 's) dtype;
      shape : int array;
      fill : 'v;
      pieces : (Nx_array.Move.range array * ('v, 's, 'd) t) list;
    }
      -> ('v, 's, 'd) t prim
      (** The value of [shape] whose element at an index is the last piece's
          whose region, a [Slice] of [shape] by its ranges, holds it, and [fill]
          where none does. Each piece has its region's shape. *)
  | Contract : {
      batch : (int * int) array;
      contracting : (int * int) array;
      acc : Nx_array.Dtype.any;
          (** A dtype {!Nx_kernel.accumulates} takes. *)
      out : ('v, 's) dtype;
      a : ('a, 'b, 'd) t;
      b : ('c, 'e, 'd) t;
      init : ('v, 's, 'd) t option;  (** Of the result's shape. *)
    }
      -> ('v, 's, 'd) t prim
      (** The contraction of [a] with [b] by the pairs [batch] and
          [contracting], from [init], computed in [acc] and rounded once to
          [out] ({!Nx_kernel.S.contract}): its result C-contiguous, of the
          shape {!Nx_kernel.Shape.contract} gives. *)
  | Fft : ('d, 'r) fft -> 'r prim
      (** A Fourier transform ({!Nx_kernel.S.c2c} and its siblings). *)
  | Linalg : ('d, 'r) linalg -> 'r prim
      (** A factorisation or a triangular solve ({!Nx_kernel.S.cholesky} and
          its siblings). *)
  | Copy : ('v, 's, 'd) t -> ('v, 's, 'd) t prim
      (** The value stored afresh, C-contiguous. *)
  | Move : Nx_array.Move.t * ('v, 's, 'd) t -> ('v, 's, 'd) t prim
  | Bitcast : ('w, 'r) dtype * ('v, 's, 'd) t -> ('w, 'r, 'd) t prim
      (** The bits read in another dtype: one width keeps the shape, a narrower
          one appends an axis of the ratio, a wider one consumes a trailing axis
          of it. *)
  | Place : 'e Devices.placement * ('v, 's, 'd) t -> ('v, 's, 'e) t prim
  | Check : {
      ok : (bool, Nx_array.Dtype.bool_elt, 'd) t;
      data : 'd any list;
      fail : int array -> 'd any list -> exn;
    }
      -> unit prim
      (** Raises [fail i data_i] at the first index [i], in C order, where [ok]
          is [false], [data_i] each of [data] at [i]. *)

and ('d, _) fft =
  | C2c : {
      direction : Nx_kernel.direction;
      axes : int array;
      x : (Complex.t, 's, 'd) t;
    }
      -> ('d, (Complex.t, 's, 'd) t) fft
      (** The transform [direction] of [x] along each of [axes]. *)
  | R2c : {
      dtype : (Complex.t, 'c) dtype;
      axes : int array;
      x : (float, 's, 'd) t;
    }
      -> ('d, (Complex.t, 'c, 'd) t) fft
      (** The forward transform of a real [x] along each of [axes], bins [0] to
          [n/2] along the last; [dtype] has [x]'s precision. *)
  | C2r : {
      dtype : (float, 'r) dtype;
      n : int;
      axes : int array;
      x : (Complex.t, 's, 'd) t;
    }
      -> ('d, (float, 'r, 'd) t) fft
      (** The real inverse transform of [n] points along the last of [axes],
          from [n/2 + 1] bins; [dtype] has [x]'s precision. *)

(** Each constructor is one routine of {!Nx_kernel.S}, its results in that
    routine's order, each of [a]'s dtype but [Lu]'s positions. *)
and ('d, _) linalg =
  | Cholesky : {
      triangle : Nx_kernel.triangle;
      a : ('v, 's, 'd) t;
    }
      -> ('d, ('v, 's, 'd) t) linalg
  | Lu : ('v, 's, 'd) t -> ('d, ('v, 's, 'd) t * 'd index * 'd index) linalg
      (** [lu], [pivots], [perm]. *)
  | Qr : {
      factors : Nx_kernel.factors;
      a : ('v, 's, 'd) t;
    }
      -> ('d, ('v, 's, 'd) t * ('v, 's, 'd) t) linalg  (** [q], [r]. *)
  | Svd : {
      factors : Nx_kernel.factors;
      a : ('v, 's, 'd) t;
    }
      -> ('d, ('v, 's, 'd) t * ('v, 's, 'd) t * ('v, 's, 'd) t) linalg
      (** [u], [s], [vh]. *)
  | Svd_values : ('v, 's, 'd) t -> ('d, ('v, 's, 'd) t) linalg
      (** [s]: [svd] without vectors. *)
  | Eigh : ('v, 's, 'd) t -> ('d, ('v, 's, 'd) t * ('v, 's, 'd) t) linalg
      (** [w], [v]. *)
  | Eigh_values : ('v, 's, 'd) t -> ('d, ('v, 's, 'd) t) linalg
      (** [w]: [eigh] without vectors. *)
  | Eig :
      (Complex.t, 's, 'd) t
      -> ('d, (Complex.t, 's, 'd) t * (Complex.t, 's, 'd) t) linalg
      (** [w], [v]. *)
  | Eig_values : (Complex.t, 's, 'd) t -> ('d, (Complex.t, 's, 'd) t) linalg
      (** [w]: [eig] without vectors. *)
  | Solve_triangular : {
      triangle : Nx_kernel.triangle;
      adjoint : bool;
      unit_diagonal : bool;
      a : ('v, 's, 'd) t;
      b : ('v, 's, 'd) t;
    }
      -> ('d, ('v, 's, 'd) t) linalg
