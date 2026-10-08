(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A multiply by an odd constant moves every bit up; the shift brings the high
   ones back down to the low bits a table indexes by. *)
let mix k =
  let h = k * 0x2545F4914F6CDD1D in
  h lxor (h lsr 29)

module Address = Hashtbl.Make (struct
  type t = int

  let equal = Int.equal
  let hash = mix
end)

module Range = Hashtbl.Make (struct
  type t = int * int

  let equal (a, n) (b, m) = Int.equal a b && Int.equal n m
  let hash (a, n) = mix (a lxor (n lsl 1))
end)

module Window = Hashtbl.Make (struct
  type t = Window.t

  (* Windows of two machines may lie at one address: all of a window's fields
     tell it apart. *)
  let equal (a : t) b = a = b
  let hash w = mix (Window.address w)
end)
