(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_gg

type t = { xs : float array; ys : float array }

let v xs ys =
  let n = Array.length xs in
  if n <> Array.length ys then
    invalid_arg (Printf.sprintf "Ring2.v: %d xs but %d ys" n (Array.length ys));
  for i = 0 to n - 1 do
    if not (Float.is_finite xs.(i) && Float.is_finite ys.(i)) then
      invalid_arg
        (Printf.sprintf "Ring2.v: point %d (%g, %g) is not finite" i xs.(i)
           ys.(i))
  done;
  { xs = Array.copy xs; ys = Array.copy ys }

let length r = Array.length r.xs

let check fn r i =
  if i < 0 || i >= Array.length r.xs then
    invalid_arg
      (Printf.sprintf "Ring2.%s: index %d not in [0;%d]" fn i
         (Array.length r.xs - 1))

let x r i =
  check "x" r i;
  Array.unsafe_get r.xs i

let y r i =
  check "y" r i;
  Array.unsafe_get r.ys i

(* Coordinates relative to the first point keep the products small for a ring
   far from the origin. *)
let area r =
  let n = Array.length r.xs in
  if n < 3 then 0.
  else
    let x0 = r.xs.(0) and y0 = r.ys.(0) in
    let s = ref 0. in
    for i = 1 to n - 2 do
      let ax = Array.unsafe_get r.xs i -. x0
      and ay = Array.unsafe_get r.ys i -. y0 in
      let bx = Array.unsafe_get r.xs (i + 1) -. x0
      and by = Array.unsafe_get r.ys (i + 1) -. y0 in
      s := !s +. ((ax *. by) -. (bx *. ay))
    done;
    0.5 *. !s

let bounds r =
  let n = Array.length r.xs in
  if n = 0 then None
  else begin
    let minx = ref r.xs.(0) and maxx = ref r.xs.(0) in
    let miny = ref r.ys.(0) and maxy = ref r.ys.(0) in
    for i = 1 to n - 1 do
      let x = Array.unsafe_get r.xs i and y = Array.unsafe_get r.ys i in
      if x < !minx then minx := x else if x > !maxx then maxx := x;
      if y < !miny then miny := y else if y > !maxy then maxy := y
    done;
    Some (Box2.of_pts (P2.v !minx !miny) (P2.v !maxx !maxy))
  end

let reverse r =
  let n = Array.length r.xs in
  let rev a = Array.init n (fun i -> Array.unsafe_get a (n - 1 - i)) in
  { xs = rev r.xs; ys = rev r.ys }

let to_path r = Path.polygon r.xs r.ys

let equal r r' =
  let n = Array.length r.xs in
  n = Array.length r'.xs
  &&
  let rec loop i =
    i >= n
    || Float.equal r.xs.(i) r'.xs.(i)
       && Float.equal r.ys.(i) r'.ys.(i)
       && loop (i + 1)
  in
  loop 0

let pp ppf r =
  let n = Array.length r.xs in
  Format.pp_open_box ppf 0;
  for i = 0 to n - 1 do
    if i > 0 then Format.pp_print_space ppf ();
    Format.fprintf ppf "%s %g %g" (if i = 0 then "M" else "L") r.xs.(i) r.ys.(i)
  done;
  if n > 0 then Format.fprintf ppf "@ Z";
  Format.pp_close_box ppf ()
