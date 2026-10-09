(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let devices =
  Array.init 4 (fun k ->
      match Rig.memory_device (Printf.sprintf "m%d" k) with
      | Ok d -> d
      | Error e -> failwith e)

let memory k = devices.(k)

module Counting = struct
  include Nx_cpu

  let name = "nx.test"
  let count = Atomic.make 0
  let calls () = Atomic.get count
  let reset () = Atomic.set count 0

  let apply0 k ~dst =
    Atomic.incr count;
    Nx_cpu.apply0 k ~dst

  let apply1 k ~dst x =
    Atomic.incr count;
    Nx_cpu.apply1 k ~dst x

  let apply2 k ~dst x y =
    Atomic.incr count;
    Nx_cpu.apply2 k ~dst x y

  let apply3 k ~dst c x y =
    Atomic.incr count;
    Nx_cpu.apply3 k ~dst c x y

  let map s ~dsts ops =
    Atomic.incr count;
    Nx_cpu.map s ~dsts ops

  let contract s ~dst ops =
    Atomic.incr count;
    Nx_cpu.contract s ~dst ops
end

module Declining = struct
  include Nx_cpu

  let name = "nx.test"

  let apply0 (k : Nx_kernel.Prog.op0) ~dst =
    match k with Fill _ -> Nx_array.Declined | Iota _ -> Nx_cpu.apply0 k ~dst

  let apply2 (k : Nx_kernel.Prog.op2) ~dst x y =
    match k with
    | Binary Add -> Nx_array.Declined
    | _ -> Nx_cpu.apply2 k ~dst x y
end
