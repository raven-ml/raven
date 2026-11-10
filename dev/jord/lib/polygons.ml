(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  crs : Crs.t;
  points : Nx.host Nx.float64_t;
  rings : int array;
  features : int array;
}

let strf = Printf.sprintf

let offsets name a ~last =
  let n = Array.length a in
  if n = 0 || a.(0) <> 0 || a.(n - 1) <> last then
    invalid_arg (strf "Jord.Polygons.v: %s must run from 0 to %d" name last);
  for k = 1 to n - 1 do
    if a.(k) < a.(k - 1) then
      invalid_arg (strf "Jord.Polygons.v: %s decreases at %d" name k)
  done

let v crs points ~rings ~features =
  (match Nx.shape points with
  | [| _; 2 |] -> ()
  | s ->
      invalid_arg
        (strf "Jord.Polygons.v: points of shape [%s] are not [n; 2]"
           (String.concat "; " (List.map string_of_int (Array.to_list s)))));
  if not (Array.for_all Float.is_finite (Nx.to_array points)) then
    invalid_arg "Jord.Polygons.v: a point is not finite";
  offsets "rings" rings ~last:(Nx.dim 0 points);
  offsets "features" features ~last:(Array.length rings - 1);
  { crs; points; rings; features }

let count p = Array.length p.features - 1

let to_crs crs p =
  { p with crs; points = Crs.transform ~src:p.crs ~dst:crs p.points }
