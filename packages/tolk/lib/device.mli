(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Devices as a compiler sees them: the renderer that writes a device's
    programs, and the compiled program a runtime loads.

    A device is named by its kind, the [DEVICE] field of a {!Helpers.Target.t}:
    ["CPU"], ["METAL"], ["CUDA"], ["NV"] or ["AMD"]. This module opens no
    device: opening one, allocating its memory and running programs on it are
    the runtime's. *)

(** {1:programs Compiled programs} *)

(** Compiled programs, as a runtime loads them. *)
module Tiny_elf : sig
  type param = {
    name : string option;  (** The parameter's name, if it has one. *)
    slot : int;
        (** Its position among the program's arguments: the buffers first, in
            the order of {!Ops.program_info.globals}, then the scalar variables,
            in the order of {!Ops.program_info.vars}. *)
    dtype : Dtype.t;  (** The type of its elements. *)
    shape : int list;
        (** [[n]] for a buffer of [n] elements, [[]] for a scalar. *)
  }
  (** The type for the parameters of a program. *)

  type t = {
    lib : string;  (** The binary. *)
    name : string;  (** The name of the program's function in [lib]. *)
    target : Helpers.Target.t;  (** The target [lib] is compiled for. *)
    signature : param list;
        (** The parameters, in the order the program declares them. *)
    profile_key : string option;
        (** The key that names the program in profiles, if any. *)
  }
  (** The type for compiled programs. *)

  val of_program : Ops.t -> t
  (** [of_program prg] is the compiled program [prg], an {!Op.Program} whose
      sources are its kernel, its {!Op.Linear} order, its {!Op.Source} and its
      {!Op.Binary}. Its binary is the bytes of the {!Op.Binary}, its name the
      {!Ops.function_name} of the kernel, its target that of its
      {!Ops.program_info}, and its profile key {!Ops.key}[ prg]. Its signature
      holds each {!Op.Param} of the linear order that is not a scalar variable,
      then each of the program's variables.

      Raises [Invalid_argument] if [prg] is not such a program, or if a buffer
      of its linear order is not among its {!Ops.program_info.globals}. *)

  val pp_param : Format.formatter -> param -> unit
  (** [pp_param] formats a parameter as the tuple of its fields:
      [('n', 3, dtypes.int, ())], or [(None, 0, dtypes.float, (16,))] for a
      parameter without a name. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a program as
      [TinyELF(lib=b'...', name='E_4_4', target=CPU::arm64,native,
       signature=((None, 0, dtypes.float, (16,)),), profile_key=b'...')], its
      binary and profile key as byte literals. *)

  val iter_sig : ?offset:int -> param list -> (int * Dtype.t) list
  (** [iter_sig ~offset signature] is where the value of each parameter of
      [signature] lies when their values are packed in order from byte [offset],
      each aligned to its size: its byte offset and its type. [offset] defaults
      to [0]. *)
end

(** {1:renderers Renderers} *)

val renderer : ?arch:string -> string -> (Renderer.t, string) result
(** [renderer ~arch device] is the renderer of [device]'s programs, made for the
    target {!Helpers.target}[ ~arch device]: the first target of the setting
    {!Helpers.dev} that names [device] or no device, with [arch] as its
    architecture if it names none. [arch] is the architecture of the device at
    hand, as the device reports it; it defaults to [""].

    Each device has these renderers, in order of preference, named as a target's
    [RENDERER] field names them:
    - ["CPU"]: ["CLANG"], {!Cstyle.clang};
    - ["METAL"]: ["METAL"], {!Cstyle.metal};
    - ["CUDA"] and ["NV"]: ["CUDA"], {!Cstyle.cuda};
    - ["AMD"]: ["HIP"], {!Cstyle.hip}.

    If the target names a renderer, only that one is tried; otherwise each is,
    in order, and the first that the target suits is the result. A target does
    not suit a renderer that raises [Invalid_argument] for it. A renderer is
    made once for each name and target, and later calls return that one.

    The error is ["D has no renderer 'R'"] if the target names a renderer [R]
    that the device [D] lacks, with the name that [R] may misspell
    ({!Helpers.select_by_name}). If no renderer suits the target, it is the
    message of the only renderer tried, or ["No renderer for D is available"]
    followed by each renderer's message, one per line.

    Raises [Invalid_argument] if [device] is not one of these devices. *)
