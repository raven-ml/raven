(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let message fn m = "Aion." ^ fn ^ ": " ^ m
let invalid fn fmt = Format.kasprintf (fun m -> invalid_arg (message fn m)) fmt
let failure fn fmt = Format.kasprintf (fun m -> failwith (message fn m)) fmt

let exn fn fmt =
  Format.kasprintf (fun m -> Invalid_argument (message fn m)) fmt
