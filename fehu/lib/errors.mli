(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Error types for the Fehu reinforcement learning framework.

    This module defines the error hierarchy for environment operations,
    validation, and registration. All framework-specific errors use the {!Error}
    exception with structured error variants for precise error handling. *)

(** Error variants covering framework operations.

    Each variant includes a descriptive message for debugging. *)
type t =
  | Unregistered_env of string  (** Environment ID not found in registry *)
  | Namespace_not_found of string  (** Environment namespace doesn't exist *)
  | Name_not_found of string  (** Environment name not found in namespace *)
  | Version_not_found of string  (** Environment version not available *)
  | Deprecated_env of string  (** Environment is deprecated *)
  | Registration_error of string  (** Environment registration failed *)
  | Dependency_not_installed of string  (** Required dependency missing *)
  | Unsupported_mode of string  (** Requested mode not supported *)
  | Invalid_metadata of string  (** Metadata validation failed *)
  | Reset_needed of string  (** Environment requires reset before step *)
  | Invalid_action of string  (** Action outside valid space *)
  | Missing_argument of string  (** Required argument not provided *)
  | Invalid_probability of string  (** Probability value out of [0, 1] range *)
  | Invalid_bound of string  (** Bound constraint violated *)
  | Closed_environment of string  (** Operation on closed environment *)

exception Error of t
(** Framework exception wrapping structured error types. *)

val to_string : t -> string
(** [to_string error] converts [error] to a human-readable message.

    Use for logging, error reporting, or user-facing diagnostics. *)

val raise_error : t -> 'a
(** [raise_error error] raises the framework exception with [error].

    Equivalent to [raise (Error error)]. *)
