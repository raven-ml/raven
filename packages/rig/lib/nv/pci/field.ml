(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Structures *)

(* The field [(off, n)] read and written a byte at a time, little-endian,
   through [byte] and [set_byte]. *)
let read byte (off, n) =
  let x = ref 0 in
  for i = n - 1 downto 0 do
    x := (!x lsl 8) lor byte (off + i)
  done;
  !x

(* An arithmetic shift: a negative [x] is sign-extended to 64 bits. *)
let write set_byte (off, n) x =
  for i = 0 to n - 1 do
    set_byte (off + i) ((x asr (8 * i)) land 0xff)
  done

let set b f x = write (Bytes.set_uint8 b) f x
let get s f = read (String.get_uint8 s) f

let pget (p : Rig_nv.params) f =
  read (fun i -> Char.code (Bigarray.Array1.get p i)) f

let pset (p : Rig_nv.params) f x =
  write (fun i v -> Bigarray.Array1.set p i (Char.unsafe_chr v)) f x

let at (base, _) (off, n) = (base + off, n)

let record size fill =
  let b = Bytes.make size '\000' in
  fill b;
  Bytes.unsafe_to_string b

(* Registers *)

let mask (lo, n) = ((1 lsl n) - 1) lsl lo
let put (lo, n) x = (x land ((1 lsl n) - 1)) lsl lo
