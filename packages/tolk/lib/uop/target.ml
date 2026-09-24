(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type t = {
  device : string;
  renderer : string;
  arch : string;
  interface : string;
  indices : string;
}

let of_string s =
  let interface, indices, target =
    match String.split_on_char '+' s with
    | [ target ] -> "", "", target
    | [ iface; target ] -> (
        match String.rindex_opt iface ':' with
        | None -> iface, "", target
        | Some i ->
            String.sub iface 0 i,
            String.sub iface (i + 1) (String.length iface - i - 1), target)
    | _ -> invalid_arg (Printf.sprintf "too many '+' in target string: %S" s)
  in
  let device, renderer, arch =
    match String.split_on_char ':' target with
    | [ device ] -> device, "", ""
    | [ device; renderer ] -> device, renderer, ""
    | [ device; renderer; arch ] -> device, renderer, arch
    | _ -> invalid_arg (Printf.sprintf "too many ':' in target string: %S" s)
  in
  { device = String.uppercase_ascii device;
    renderer = String.uppercase_ascii renderer; arch; interface; indices }

let to_string t =
  let join fields =
    let s = String.concat ":" fields in
    let stop = ref (String.length s) in
    while !stop > 0 && s.[!stop - 1] = ':' do decr stop done;
    String.sub s 0 !stop
  in
  let iface = join [ t.interface; t.indices ] in
  (if iface = "" then "" else iface ^ "+")
  ^ join [ t.device; t.renderer; t.arch ]
