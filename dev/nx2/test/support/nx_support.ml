(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let devices =
  Array.init 4 (fun k ->
      Rig.memory_device (Printf.sprintf "m%d" k))

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

  let map p ~dsts loads =
    Atomic.incr count;
    Nx_cpu.map p ~dsts loads

  let reduce p ~axes rs ~dsts loads =
    Atomic.incr count;
    Nx_cpu.reduce p ~axes rs ~dsts loads

  let scan p ~axis r ~dsts loads =
    Atomic.incr count;
    Nx_cpu.scan p ~axis r ~dsts loads

  let gather ~axis ~dst idx x =
    Atomic.incr count;
    Nx_cpu.gather ~axis ~dst idx x

  let scatter c ~unique ~axis ~dst ~into idx u =
    Atomic.incr count;
    Nx_cpu.scatter c ~unique ~axis ~dst ~into idx u

  let sort ~axis ~descending ~values ~positions x =
    Atomic.incr count;
    Nx_cpu.sort ~axis ~descending ~values ~positions x

  let assemble ~fill ~dst pieces =
    Atomic.incr count;
    Nx_cpu.assemble ~fill ~dst pieces

  let fold p ~dst x =
    Atomic.incr count;
    Nx_cpu.fold p ~dst x

  let contract ~batch ~contracting ~acc ~dst ~init a b =
    Atomic.incr count;
    Nx_cpu.contract ~batch ~contracting ~acc ~dst ~init a b
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
