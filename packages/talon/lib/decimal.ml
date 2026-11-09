(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type bytes = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(* An [int64] holds every integer of 18 digits and some of 19. *)
let int64_digits = 18

let reads_as ~precision (Type.Any t) =
  match t with
  | Float64 -> true
  | Int64 -> precision <= int64_digits
  | _ -> false

let reads_text ~precision =
  if precision <= int64_digits then "float64 or int64" else "float64"

let scales_agree (Type.Any t) ~format ~file =
  match t with Int64 -> Option.equal Int.equal format file | _ -> true

let other_scale =
  "holds decimals of another scale than the format's, so its int64 integers \
   would mean other numbers. Declare it for this file with with_type."

let undeclared ~fn ~column ~text ~flat =
  invalid_arg
    (Printf.sprintf
       "%s: column %S (%s) %s a decimal, which reads as float64, or as its \
        unscaled int64 integers up to %d digits. Declare one with with_type."
       fn column text
       (if flat then "is" else "holds")
       int64_digits)

let unscaled (b : bytes) ~first ~len ~big out i =
  let msb = if big then first else first + len - 1 in
  let step = if big then 1 else -1 in
  let v = ref (Int64.of_int ((b.{msb} lxor 0x80) - 0x80)) in
  let k = ref 1 and fits = ref true in
  while !fits && !k < len do
    if
      Int64.compare !v 0x7F_FFFF_FFFF_FFFFL > 0
      || Int64.compare !v (-0x80_0000_0000_0000L) < 0
    then fits := false
    else begin
      v := Int64.logor (Int64.shift_left !v 8) (Int64.of_int b.{msb + (!k * step)});
      incr k
    end
  done;
  if !fits then out.{i} <- !v;
  !fits

let billion = 1_000_000_000

(* The magnitude of the integer is divided by 10^9 until it is zero, each
   remainder giving nine digits. *)
let digits (b : bytes) ~first ~len ~big =
  let msb = if big then first else first + len - 1 in
  let step = if big then 1 else -1 in
  let negative = b.{msb} land 0x80 <> 0 in
  let m =
    Array.init len (fun k ->
        let x = b.{msb + (k * step)} in
        if negative then x lxor 0xFF else x)
  in
  if negative then begin
    let k = ref (len - 1) in
    while !k >= 0 && m.(!k) = 0xFF do
      m.(!k) <- 0;
      decr k
    done;
    if !k >= 0 then m.(!k) <- m.(!k) + 1
  end;
  let groups = ref [] in
  while Array.exists (fun d -> d <> 0) m do
    let r = ref 0 in
    for k = 0 to len - 1 do
      let x = (!r lsl 8) lor m.(k) in
      m.(k) <- x / billion;
      r := x mod billion
    done;
    groups := !r :: !groups
  done;
  match !groups with
  | [] -> "0"
  | g :: gs ->
      String.concat ""
        ((if negative then "-" else "")
        :: string_of_int g
        :: List.map (Printf.sprintf "%09d") gs)

let floats n ~scale digits =
  let texts =
    Array.init n (fun i -> Printf.sprintf "%se%d" (digits i) (-scale))
  in
  match Form.parse (Any Type.float64) (Column.v Type.string texts) with
  | Ok c -> Ok (Column.to_tensor Nx.float64 c)
  | Error (_, why) -> Error why
