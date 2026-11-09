(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The law of a call that does not fit its kernel (Nx_kernel's preamble):
   the kernel answers a refusal or Declined, never Done, and reads and
   writes nothing outside its arrays. Its destinations lie between fences
   of known bytes, and on the host its destinations and operands lie
   against an inaccessible page on one side, then the other: a write just
   past a destination changes a fence, and a read or write past an array
   on the page's side faults. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module Support = Nx_kernels_support

(* The bytes of each fence: a vector past the widest element. *)
let fence = 64

let pattern i = (0xA5 + (7 * i)) land 0xFF

(* A C-contiguous array of [dt] and shape [s] on [b]'s device between two
   fences, its bytes and the fences' a pattern, against the page on [side]
   on the host; and the function that frees the memory and fails, naming
   [msg], unless the fences, or with [~whole] every byte, hold the
   pattern. *)
let fenced ~msg side (b : Support.backend) (D.Any dt) s =
  let n = Array.fold_left ( * ) 1 s in
  let total = fence + (((n * D.bits dt) + 7) / 8) + fence in
  let span = Iarray.of_array [| total |] in
  let bytes, free =
    if Rig.equal b.device Rig.host then
      Support.guarded side D.Uint8 (L.contiguous span)
    else (A.create b.device D.Uint8 span, ignore)
  in
  A.transfer ~dst:bytes
    ~src:(A.of_array D.Uint8 span (Array.init total pattern));
  let s = Iarray.of_array s in
  let strides = L.strides (L.contiguous s) in
  let l = L.v ~offset:(fence * 8 / D.bits dt) ~strides s in
  let check ~whole =
    let got = A.to_array (A.to_device Rig.host bytes) in
    free ();
    Array.iteri
      (fun i v ->
        let inside = i >= fence && i < total - fence in
        if (whole || not inside) && v <> pattern i then
          if inside then failf "%s: byte %d of a destination changed" msg i
          else
            failf "%s: byte %d of the fence %s a destination changed" msg
              (if i < fence then i else i - (total - fence))
              (if i < fence then "before" else "past"))
      got
  in
  (A.Any (A.v dt l (A.buffer bytes)), check)

(* The type for placing a call's operands: [g x] is [x], or a copy of [x]
   against the inaccessible page. *)
type guard = { g : 'v 's. ('v, 's) A.t -> ('v, 's) A.t }

(* [l] with its operand placed by [g]. *)
let load g (l : Nx_kernel.load) : Nx_kernel.load =
  match l with
  | Plain x -> Plain (g.g x)
  | Padded { x; fill; pad } -> Padded { x = g.g x; fill; pad }

(* [f g], [g] placing operands against the page on [side] on the host,
   their memory freed once [f] returns. *)
let with_guard (b : Support.backend) side f =
  if not (Rig.equal b.device Rig.host) then f { g = Fun.id }
  else
    let frees = ref [] in
    let g x =
      let y, free = Support.guard side x in
      frees := free :: !frees;
      y
    in
    Fun.protect
      ~finally:(fun () -> List.iter (fun free -> free ()) !frees)
      (fun () -> f { g })

(* [call g ds] on each side of the page, [ds] fenced destinations of the
   dtypes and shapes [dsts] and [g] placing its operands: its answer, and
   the fences, or where [whole a] every byte, as they were. *)
let sides ~msg (b : Support.backend) dsts call ~whole ~answer =
  List.iter
    (fun side ->
      let made = Array.map (fun (d, s) -> fenced ~msg side b d s) dsts in
      let a =
        with_guard b side (fun g ->
            b.around (fun () -> call g (Array.map fst made)))
      in
      Array.iter (fun (_, check) -> check ~whole:(whole a)) made;
      answer a)
    [ Support.Before; Support.After ]

(* [call g ds] of a call whose misfit is of the class [what], a dtype, a
   count, a shape or an attribute: not Done, and nothing read or written
   outside its arrays. A table of misfits checks every row, so each class
   it lists is reached on every run. *)
let check ~msg ~what b dsts call =
  let msg = what ^ ", " ^ msg in
  sides ~msg b dsts call
    ~whole:(fun _ -> false)
    ~answer:(fun a ->
      if a = A.Done then failf "%s: a call that does not fit is Done" msg)

(* As {!check} where the call does not [fit]. One that fits is Done,
   writing nothing outside its destinations, or Declined, writing
   nothing. *)
let law ~msg ~what b ~fit dsts call =
  if not fit then check ~msg ~what b dsts call
  else
    sides ~msg b dsts call
      ~whole:(fun a -> a = A.Declined)
      ~answer:(function
        | A.Done | Declined -> ()
        | r ->
            failf "%s: a call that fits answered %a" msg
              Nx_array_support.pp_answer r)
