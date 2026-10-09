(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx_kernel.Prog

(* Each domain's programs, by node and operand dtypes: plain data, compared
   structurally. *)
let made = Domain.DLS.new_key (fun () -> Hashtbl.create 64)

let single node ins =
  let table = Domain.DLS.get made in
  let key = (node, ins) in
  match Hashtbl.find_opt table key with
  | Some p -> p
  | None ->
      let n = Array.length ins in
      let nodes = Array.append (Array.init n (fun i -> P.In i)) [| node |] in
      let p = P.v ~ins nodes ~outs:[| n |] in
      Hashtbl.add table key p;
      p
