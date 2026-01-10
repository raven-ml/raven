(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Unregistered_env of string
  | Namespace_not_found of string
  | Name_not_found of string
  | Version_not_found of string
  | Deprecated_env of string
  | Registration_error of string
  | Dependency_not_installed of string
  | Unsupported_mode of string
  | Invalid_metadata of string
  | Reset_needed of string
  | Invalid_action of string
  | Missing_argument of string
  | Invalid_probability of string
  | Invalid_bound of string
  | Closed_environment of string

let message prefix details =
  if String.length details = 0 then prefix else prefix ^ ": " ^ details

let to_string = function
  | Unregistered_env id -> message "Environment not registered" id
  | Namespace_not_found ns -> message "Environment namespace not found" ns
  | Name_not_found name -> message "Environment name not found" name
  | Version_not_found version -> message "Environment version not found" version
  | Deprecated_env id -> message "Environment version is deprecated" id
  | Registration_error msg -> message "Environment registration error" msg
  | Dependency_not_installed dep -> message "Dependency not installed" dep
  | Unsupported_mode mode -> message "Unsupported mode" mode
  | Invalid_metadata msg -> message "Invalid metadata" msg
  | Reset_needed msg -> message "Environment reset required" msg
  | Invalid_action msg -> message "Invalid action" msg
  | Missing_argument arg -> message "Missing argument" arg
  | Invalid_probability msg -> message "Invalid probability" msg
  | Invalid_bound msg -> message "Invalid bound" msg
  | Closed_environment msg -> message "Environment has been closed" msg

exception Error of t

let raise_error err = raise (Error err)
