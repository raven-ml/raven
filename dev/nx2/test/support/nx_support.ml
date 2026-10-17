(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let devices =
  Array.init 4 (fun k ->
      Result.get_ok (Rig.memory_device (Printf.sprintf "m%d" k)))

let memory k = devices.(k)

module Counting = struct
  include Nx_cpu

  let name = "nx.test"
  let count = Atomic.make 0
  let computed = Atomic.make 0
  let calls () = Atomic.get count
  let positions () = Atomic.get computed

  let reset () =
    Atomic.set count 0;
    Atomic.set computed 0

  let size shape = Array.fold_left ( * ) 1 shape

  let loop (Nx_array.Any dst) =
    size (Nx_array.Layout.shape (Nx_array.layout dst))

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
    ignore (Atomic.fetch_and_add computed (loop dsts.(0)));
    Nx_cpu.map p ~dsts loads

  let reduce p ~shape ~axes rs ~dsts loads =
    Atomic.incr count;
    ignore (Atomic.fetch_and_add computed (size shape));
    Nx_cpu.reduce p ~shape ~axes rs ~dsts loads

  let scan ?carry p ~axis r ~dsts loads =
    Atomic.incr count;
    ignore (Atomic.fetch_and_add computed (loop dsts.(0)));
    Nx_cpu.scan ?carry p ~axis r ~dsts loads

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

  let contract ~order ~batch ~contracting ~acc ~dst ~init a b =
    Atomic.incr count;
    Nx_cpu.contract ~order ~batch ~contracting ~acc ~dst ~init a b
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
