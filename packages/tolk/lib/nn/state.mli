(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Model state: loading tensors from disk and binding them to parameters.

    A state dict is an association list from parameter names to tensors. Model
    code builds one for its parameters (e.g. [("wte.weight", wte.weight)]);
    {!safe_load} reads one from a file; {!load_state_dict} binds the latter
    into the former. *)

open Tolk_frontend

val safe_load : string -> (string * Tensor.t) list
(** [safe_load fn] reads the safetensors file at [fn] and returns its tensors
    as a state dict, in header order. Each tensor's raw data is copied to the
    default device. All safetensors dtypes are supported except fp8.

    @raise Invalid_argument if the file is malformed.
    @raise Sys_error if the file cannot be read. *)

val load_state_dict :
  ?strict:bool -> ?realize:bool ->
  (string * Tensor.t) list -> (string * Tensor.t) list -> unit
(** [load_state_dict model state_dict] rebinds every parameter tensor of
    [model] onto the value of the same name in [state_dict]: the parameter
    handle is repointed at the loaded value, so the model computes with the
    loaded weights from then on. A one-element value is reshaped to the
    parameter's shape if they disagree. Extra names in [state_dict] are
    ignored. With [strict] (default [true]), a parameter with no matching
    value is an error; otherwise it is left unchanged. [realize] (default
    [true]) materialises the bound parameters, so that any deferred
    transformation on the loaded values is computed once rather than on every
    use.

    @raise Invalid_argument
      on a shape mismatch, or on a missing name when [strict]. *)
