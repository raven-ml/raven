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

(** A compiled program's parameters, as a runtime packs them. *)
module Tiny_elf : sig
  type param = {
    name : string option;  (** The parameter's name, if it has one. *)
    addrspace : Dtype.addr_space;
        (** {!Dtype.Alu} for a scalar, passed as its value; any other for a
            buffer, passed as its address. *)
    dtype : Dtype.t;  (** The type of its elements. *)
    shape : int list;
        (** [[n]] for a buffer of [n] elements, [[]] for a scalar. *)
  }
  (** The type for the parameters of a program. *)

  val iter_sig : ?offset:int -> param list -> (int * Dtype.t) list
  (** [iter_sig ~offset signature] is where the value of each parameter of
      [signature] lies when their values are packed in order from byte [offset],
      each aligned to its size: its byte offset and its type, a scalar's own
      and a buffer's address's {!Dtype.Uint64}. [offset] defaults to [0]. *)
end

(** {1:renderers Renderers} *)

val renderer : Helpers.Target.t -> (Renderer.t, string) result
(** [renderer t] is the renderer of programs for [t]: the renderer of [t]'s
    device, made for [t].

    Each device has one renderer, named as a target's [RENDERER] field names
    it:
    - ["CPU"]: ["CLANG"], {!Cstyle.clang};
    - ["METAL"]: ["METAL"], {!Cstyle.metal};
    - ["CUDA"] and ["NV"]: ["CUDA"], {!Cstyle.cuda};
    - ["AMD"]: ["HIP"], {!Cstyle.hip}.

    A renderer is made once for each name and target, and later calls return
    that one.

    The error is ["D has no renderer 'R'"] if the target names a renderer [R]
    other than its device [D]'s, and the renderer's own message if it raises
    [Invalid_argument] for the target.

    Raises [Invalid_argument] if [t]'s device is not one of these devices, and
    [Failure] as the renderer does if its compiler's library is missing, keeping
    no renderer for [t]. *)
