(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let fail (it : Jord_stac.item) fmt =
  Printf.ksprintf (fun m -> failwith (it.id ^ ": " ^ m)) fmt

let rec group key = function
  | [] -> []
  | it :: _ as l ->
      let k = key it in
      let same, rest = List.partition (fun it' -> key it' = k) l in
      (k, same) :: group key rest

let string name (it : Jord_stac.item) =
  let mems = match it.properties with Jsont.Object (ms, _) -> ms | _ -> [] in
  match Jsont.Json.find_mem name mems with
  | Some (_, Jsont.String (s, _)) -> s
  | _ -> fail it "no string property %s" name

let compare (a : Jord_stac.item) (b : Jord_stac.item) =
  match Float.compare (Stac.time a) (Stac.time b) with
  | 0 -> String.compare a.id b.id
  | c -> c

let has keys (it : Jord_stac.item) =
  List.for_all (fun k -> List.mem_assoc k it.assets) keys

let window s ~centres rule (it : Jord_stac.item) key =
  let asset =
    match List.assoc_opt key it.assets with
    | Some a -> a
    | None -> fail it "no asset %s" key
  in
  let crs =
    match asset.grid with
    | Some (Ok g) -> g.crs
    | Some (Error e) -> fail it "%s: %s" key e
    | None -> fail it "%s: no grid" key
  in
  (asset, Stac.window s rule asset crs (centres crs))
