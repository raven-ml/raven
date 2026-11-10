(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A contraction's launches of contract.cu.

    A plan takes a call nx.cuda found nothing to refuse in and lowers it
    to one submission: packs of the operands a kernel does not read where
    they lie; per window of one launch's bounds, a segmented call's
    prepass and a launch of the contraction; and a pack of y out of the
    workspace.
    Its association follows the dtypes, the scales and segments given and
    the extents alone, never a layout or an alignment. *)

type call = Bytes.t Launch.call
(** The type for calls, one submission each. A call past one launch's
    bounds, of as many launches as its windows, is not kept: a submission
    per window count would grow with the calls made. *)

val plan :
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  segments:
    (int64, Nx_array.Dtype.int64_elt) Nx_array.t Nx_kernel.segments option ->
  acc:Nx_array.Dtype.any ->
  dst:Nx_array.any ->
  init:Nx_array.any option ->
  Nx_kernel.operand ->
  Nx_kernel.operand ->
  Bytes.t Launch.plan
(** [plan ~batch ~contracting ~segments ~acc ~dst ~init a b] plans
    {!Nx_kernel.S.contract} of a call whose shapes fit, [init] of [dst]'s
    dtype, as at most one call. It is [Error Declined] for a scale with an
    integer or complex [acc]. *)

val set_wide_k : int -> unit
(** [set_wide_k n] makes a Contracting call's tables hold int64 indices
    past a contracted extent of [n], at least 0 and at most [2{^31} - 1],
    so that a test reaches them with few indices. *)

(** {1:routines The matrix routines' calls}

    A routine's call is one submission of its own launches and its
    products', made through a builder. *)

type builder
(** The type for a call under construction: its slots, launches and
    workspace. *)

val builder : unit -> builder
(** [builder ()] is a call of no launch. *)

val slot : builder -> Rig.Buffer.t -> int
(** [slot b buf] is [buf]'s slot in [b], added at its first use. *)

val fresh : builder -> Rig.t -> int -> Rig.Buffer.t
(** [fresh b d n] is a buffer of [n] bytes on [d] of [b]'s own, which the
    call frees once its plan is submitted. *)

val ws_slot : int
(** [ws_slot] is the workspace's slot until {!finish}. *)

val tickets_slot : int
(** [tickets_slot] is the tickets' slot until {!finish}: zero words, which
    a launch leaves zero. *)

val take : builder -> int -> int
(** [take b n] is the byte of [n] bytes of the workspace, on 256 bytes. *)

val mark : builder -> int
(** [mark b] is the workspace's bytes taken so far. *)

val release : builder -> int -> unit
(** [release b m] gives back the workspace's bytes taken since [mark b] was
    [m], which the launches after it reuse: a call's launches run in
    order where they share memory. *)

val address : int list ref -> Bytes.t -> int -> int -> int -> write:bool -> unit
(** [address refs w at slot off ~write] stores the byte [off] of [slot]
    into the 8 bytes of [w] at [at] and adds the ref to [refs]. *)

val launch :
  builder ->
  kernel:int ->
  ?gy:int ->
  ?gz:int ->
  gx:int ->
  threads:int ->
  shared:int ->
  Bytes.t ->
  int list ref ->
  unit
(** [launch b ~kernel ~gx ~threads ~shared w refs] adds a launch of
    [kernel] over [gx] x [gy] x [gz] groups with parameters [w] and the
    refs {!address} added. *)

type options = {
  init_first : bool;
  neg : bool;
  conj_a : bool;
  conj_b : bool;
  tri : int;
  ranges : int option;
}
(** What a routine's product asks beyond {!Nx_kernel.S.contract}: its
    first range of k summing from [y] ({!Kernels.contract_init_first}),
    [a] negated, [a] or [b] conjugated, only the tiles of a triangle
    ({!Kernels.contract_lower} or {!Kernels.contract_upper}, or [0]), and
    [Some s] ranges of k for every tile in place of the model's: a function
    of each matrix's extents, so that a result's bits do not depend on its
    batch. *)

val sms : int
(** [sms] is the SMs the plan's model is fitted to, the RTX 5000 Ada's: a
    GPU of another size runs the same plans, with the same bits. *)

val ranges : rows:int -> cols:int -> k:int -> int
(** [ranges ~rows ~cols ~k] is the ranges of k a routine's product of
    [rows] by [cols] outputs over [k] terms sums apart ([options]'
    [ranges]): few tiles of 64 x 64 over a long k split it, as the
    matrix's extents alone decide. *)

val product :
  builder ->
  options ->
  acc:Nx_array.Dtype.any ->
  init:bool ->
  y:Nx_array.any ->
  Nx_array.any ->
  Nx_array.any ->
  unit
(** [product b o ~acc ~init ~y a m] adds the launches of
    [y <- y + op(a) op(m)] with [init], [y <- op(a) op(m)] without, for
    [y] [[batch; rows; cols]], [a] [[batch; rows; k]] and [m]
    [[batch; k; cols]], of [acc]'s dtype, any layout, [y] sharing no byte
    with [a] or [m]. *)

val copy : builder -> src:Nx_array.any -> dst:Nx_array.any -> unit
(** [copy b ~src ~dst] adds a launch copying [src]'s elements into [dst]'s,
    of one dtype and shape, any layouts. *)

val finish : ?most:int -> builder -> call
(** [finish ~most b] is [b]'s call, kept where it has at most [most]
    launches (defaults to every count), so that the submissions of calls
    whose launches grow with their extents stay few. *)
