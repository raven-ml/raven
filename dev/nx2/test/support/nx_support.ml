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
  let multiplied = Atomic.make 0
  let positions () = Atomic.get computed
  let products () = Atomic.get multiplied

  let reset () =
    Atomic.set count 0;
    Atomic.set computed 0;
    Atomic.set multiplied 0

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
    let s = Nx_array.Layout.shape (Nx_array.layout a) in
    let k = Array.fold_left (fun k (i, _) -> k * s.(i)) 1 contracting in
    ignore (Atomic.fetch_and_add multiplied (loop (Nx_array.Any dst) * k));
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

(* NaN bits *)

(* The bytes of [dt]'s floats, or of its parts, and their exponent and
   fraction widths, where a NaN has a quiet bit to set or clear:
   float8_e4m3fn's one NaN of each sign has every fraction bit set, and
   float4_e2m1fn has none. *)
let fields (type v s) (dt : (v, s) Nx_array.Dtype.t) =
  match dt with
  | Float64 | Complex128 -> Some (8, 11, 52)
  | Float32 | Complex64 -> Some (4, 8, 23)
  | Float16 -> Some (2, 5, 10)
  | Bfloat16 -> Some (2, 8, 7)
  | Float8_e5m2 -> Some (1, 5, 2)
  | _ -> None

(* [a], C-contiguous, with [f frac v] for the bits [v] of each float or part
   that is a NaN, [frac] its fraction's width. *)
let map_nans f (type v s) (a : (v, s) Nx_array.t) : (v, s) Nx_array.t =
  let a = Nx_array.copy a in
  let dt = Nx_array.dtype a in
  match fields dt with
  | None -> a
  | Some (w, exp, frac) ->
      let b = Nx_array.to_array (Option.get (Nx_array.bitcast Uint8 a)) in
      let ones = Int64.(sub (shift_left 1L exp) 1L) in
      for p = 0 to (Array.length b / w) - 1 do
        let v = ref 0L in
        for j = w - 1 downto 0 do
          v := Int64.(logor (shift_left !v 8) (of_int b.((p * w) + j)))
        done;
        let field = Int64.(logand (shift_right_logical !v frac) ones) in
        let fraction = Int64.(logand !v (sub (shift_left 1L frac) 1L)) in
        if field = ones && fraction <> 0L then begin
          let v = f frac !v in
          for j = 0 to w - 1 do
            b.((p * w) + j) <-
              Int64.to_int (Int64.shift_right_logical v (8 * j)) land 0xFF
          done
        end
      done;
      let bytes = String.init (Array.length b) (fun i -> Char.chr b.(i)) in
      Nx_array.v dt
        (Nx_array.Layout.contiguous (Nx_array.Layout.shape (Nx_array.layout a)))
        (Rig.Buffer.of_string (if bytes = "" then "\000" else bytes))

let quiet_bit frac = Int64.shift_left 1L (frac - 1)

let signalled a =
  map_nans
    (fun frac v ->
      let q = quiet_bit frac in
      if Int64.(logand v (sub q 1L)) = 0L then v
      else Int64.(logand v (lognot q)))
    a

let quieted a = map_nans (fun frac v -> Int64.logor v (quiet_bit frac)) a
