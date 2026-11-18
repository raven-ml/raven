(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* For each case of DIR/cases, the forward and inverse of DIR/NAME.{fwd,inv}.in
   through Jord.Crs.transform, into DIR/NAME.{fwd,inv}.eager, points outside
   the domain as NaN. crs_sweep.py writes the inputs and reads the outputs:

     uv run contrib/jord/test/gen/crs_sweep.py points DIR
     dune build ./contrib/jord/test/gen/crs_probe.exe
     ./_build/default/contrib/jord/test/gen/crs_probe.exe DIR
     uv run contrib/jord/test/gen/crs_sweep.py cut DIR *)

open Jord

let read file =
  let ic = open_in file in
  let rec go acc =
    match input_line ic with
    | l ->
        Scanf.sscanf l "%s %s" (fun a b ->
            go (float_of_string b :: float_of_string a :: acc))
    | exception End_of_file ->
        close_in ic;
        Array.of_list (List.rev acc)
  in
  let a = go [] in
  Nx.create Nx.float64 [| Array.length a / 2; 2 |] a

let write file t =
  let a = Nx.to_array t in
  let oc = open_out file in
  for i = 0 to (Array.length a / 2) - 1 do
    Printf.fprintf oc "%h %h\n" a.(2 * i) a.((2 * i) + 1)
  done;
  close_out oc

let system meth ell (p : float array) =
  let datum = if ell = "wgs84" then Crs.Wgs84 else Crs.Nad83 in
  let projection =
    match meth with
    | "albers" ->
        Crs.albers ~lat0:p.(0) ~lon0:p.(1) ~lat1:p.(2) ~lat2:p.(3)
          ~false_easting:p.(4) ~false_northing:p.(5)
    | "pm" ->
        Crs.pseudo_mercator ~lon0:p.(0) ~false_easting:p.(1)
          ~false_northing:p.(2)
    | "sinusoidal" ->
        Crs.sinusoidal ~radius:p.(0) ~lon0:p.(1) ~false_easting:p.(2)
          ~false_northing:p.(3)
    | m -> failwith ("method " ^ m)
  in
  { Crs.datum; projection = Some projection }

let run dir name src dst way =
  let p = read (Filename.concat dir (name ^ "." ^ way ^ ".in")) in
  let ok = Nx.reshape [| -1; 1 |] (Crs.in_domain ~src ~dst p) in
  let p = Nx.where ok p (Nx.full_like p Float.nan) in
  write
    (Filename.concat dir (name ^ "." ^ way ^ ".eager"))
    (Crs.transform ~src ~dst p)

let () =
  let dir = Sys.argv.(1) in
  let ic = open_in (Filename.concat dir "cases") in
  let rec go () =
    match input_line ic with
    | exception End_of_file -> close_in ic
    | l ->
        (match String.split_on_char ' ' l with
        | name :: meth :: ell :: rest ->
            let c =
              system meth ell (Array.of_list (List.map float_of_string rest))
            in
            let geo = { c with Crs.projection = None } in
            run dir name geo c "fwd";
            run dir name c geo "inv"
        | _ -> ());
        go ()
  in
  go ()
