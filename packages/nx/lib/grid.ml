(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type cut = { axis : int; over : int list }

type 'd t =
  | One of 'd
  | Grid of { devices : 'd list; extents : int list; cuts : cut list }

let device d = One d

(* Grid axis [g] removed from the cuts, the axes after it renumbered. *)
let without g cuts =
  List.map
    (fun c ->
      {
        c with
        over =
          List.filter_map
            (fun h -> if h = g then None else Some (if h > g then h - 1 else h))
            c.over;
      })
    cuts

let rec follows g = function
  | a :: (b :: _ as rest) -> (a = g && b = g + 1) || follows g rest
  | _ -> false

let rec normal devices extents cuts =
  let mentions g = List.exists (fun c -> List.mem g c.over) cuts in
  let mergeable g =
    (not (mentions g || mentions (g + 1)))
    || List.exists (fun c -> follows g c.over) cuts
  in
  let n = List.length extents in
  match List.find_index (( = ) 1) extents with
  | Some g ->
      normal devices (List.filteri (fun i _ -> i <> g) extents) (without g cuts)
  | None -> (
      match List.find_opt mergeable (List.init (Int.max 0 (n - 1)) Fun.id) with
      | Some g ->
          let extents =
            List.concat
              (List.mapi
                 (fun i e ->
                   if i = g then [ e * List.nth extents (g + 1) ]
                   else if i = g + 1 then []
                   else [ e ])
                 extents)
          in
          normal devices extents (without (g + 1) cuts)
      | None -> (
          match devices with
          | [ d ] -> One d
          | _ ->
              let cuts = List.filter (fun c -> c.over <> []) cuts in
              let cuts =
                List.sort (fun a b -> Int.compare a.axis b.axis) cuts
              in
              Grid { devices; extents; cuts }))

let v devices extents cuts =
  let fail fmt = Printf.ksprintf invalid_arg ("Nx.Placement: " ^^ fmt) in
  let rank = List.length extents in
  if List.fold_left ( * ) 1 extents <> List.length devices then
    fail "the extents do not multiply to the number of devices";
  let axes = List.map fst cuts and over = List.concat_map snd cuts in
  let distinct l = List.length (List.sort_uniq Int.compare l) = List.length l in
  if not (distinct axes && distinct over) then fail "an axis is cut twice";
  if List.exists (fun a -> a < 0) axes then fail "a negative axis";
  if List.exists (fun g -> g < 0 || g >= rank) over then
    fail "a grid axis out of range";
  normal devices extents (List.map (fun (axis, over) -> { axis; over }) cuts)

let devices = function One d -> [ d ] | Grid g -> g.devices

let count extents c =
  List.fold_left (fun n g -> n * List.nth extents g) 1 c.over

let cuts = function
  | One _ -> []
  | Grid { extents; cuts; _ } ->
      List.map (fun c -> (c.axis, count extents c)) cuts

let tile_index p k =
  match p with
  | One _ -> []
  | Grid { extents; cuts; _ } ->
      let e = Array.of_list extents in
      let coord = Array.make (Array.length e) 0 and r = ref k in
      for g = Array.length e - 1 downto 0 do
        coord.(g) <- !r mod e.(g);
        r := !r / e.(g)
      done;
      List.map
        (fun c ->
          (c.axis, List.fold_left (fun j g -> (j * e.(g)) + coord.(g)) 0 c.over))
        cuts

let map_axes f = function
  | One _ as p -> p
  | Grid g ->
      let cuts = List.map (fun c -> { c with axis = f c.axis }) g.cuts in
      Grid
        { g with cuts = List.sort (fun a b -> Int.compare a.axis b.axis) cuts }

let select p ~axis j =
  match p with
  | One _ -> p
  | Grid { devices; extents; cuts } ->
      let cut = List.find (fun c -> c.axis = axis) cuts in
      let keep =
        List.filteri
          (fun k _ -> List.assoc axis (tile_index p k) = j)
          (List.mapi (fun k d -> (k, d)) devices)
      in
      let gone = List.sort (fun a b -> Int.compare b a) cut.over in
      let extents = List.filteri (fun g _ -> not (List.mem g gone)) extents in
      let cuts =
        List.fold_left
          (fun cuts g -> without g cuts)
          (List.filter (fun c -> c.axis <> axis) cuts)
          gone
      in
      normal (List.map snd keep) extents cuts

let uncut p ~axis =
  match p with
  | One _ -> p
  | Grid { devices; extents; cuts } ->
      normal devices extents (List.filter (fun c -> c.axis <> axis) cuts)

(* Two placements are equal when every device holds the same window under both,
   whatever the shape: the same tile of the same number along every cut axis. *)
let equal eq p q =
  let dq = devices q in
  let tiles p k =
    List.map2 (fun (a, n) (_, j) -> (a, n, j)) (cuts p) (tile_index p k)
  in
  List.compare_lengths (devices p) dq = 0
  && List.for_all
       (fun (k, d) ->
         match List.find_index (eq d) dq with
         | Some k' -> tiles p k = tiles q k'
         | None -> false)
       (List.mapi (fun k d -> (k, d)) (devices p))

let pp pp_device ppf p =
  let list ppf ds =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
      pp_device ppf ds
  in
  match p with
  | One d -> pp_device ppf d
  | Grid { devices; extents = [ _ ]; cuts = [] } ->
      Format.fprintf ppf "replicated [%a]" list devices
  | Grid { devices; extents = [ _ ]; cuts = [ { axis; _ } ] } ->
      Format.fprintf ppf "sharded ~axis:%d [%a]" axis list devices
  | Grid { devices; extents; cuts } ->
      let ints =
        Format.pp_print_list
          ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "x")
          Format.pp_print_int
      in
      Format.fprintf ppf "grid %a [%a]" ints extents list devices;
      List.iter
        (fun c ->
          Format.fprintf ppf " ~axis:%d/%a" c.axis
            (Format.pp_print_list
               ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ",")
               Format.pp_print_int)
            c.over)
        cuts
