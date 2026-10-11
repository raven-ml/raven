(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module P = Nx_kernel.Prog

type t = {
  mutable ins : D.any list;
  mutable inputs : int;
  mutable nodes : P.node list;
  mutable count : int;
}

let make () = { ins = []; inputs = 0; nodes = []; count = 0 }

let append b n =
  b.nodes <- n :: b.nodes;
  b.count <- b.count + 1;
  b.count - 1

let input b dt =
  let k = b.inputs in
  b.ins <- dt :: b.ins;
  b.inputs <- k + 1;
  append b (In k)

let push b (n : P.node) =
  match n with
  | In _ -> invalid_arg "Build.push: an In node; operands come from input"
  | Coord _ | Const _ | Op1 _ | Op2 _ | Op3 _ -> append b n

let const b dt x = append b (Const (D.Any dt, P.bits dt x))

let program b outs =
  let ins = Iarray.of_list (List.rev b.ins) in
  P.v ~ins (Iarray.of_list (List.rev b.nodes)) ~outs
