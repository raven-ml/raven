(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A cut names the value's axis and the grid axes over which it is cut, major
   first. A [Grid] is in normal form: at least two devices, extents of at least
   2, no two adjacent grid axes that could merge, and only cuts that name a grid
   axis, sorted by value axis. *)

type cut = { axis : int; over : int list }

type t =
  | One of int
  | Grid of { devices : int list; extents : int list; cuts : cut list }

let device k = One k

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

(* Drops grid axes of extent 1, then merges the first pair of adjacent grid axes
   that no cut names or that one cut names in order, until neither applies. *)
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
          | [ k ] -> One k
          | _ ->
              let cuts = List.filter (fun c -> c.over <> []) cuts in
              let cuts =
                List.sort (fun a b -> Int.compare a.axis b.axis) cuts
              in
              Grid { devices; extents; cuts }))

let distinct l = List.length (List.sort_uniq Int.compare l) = List.length l

let v ~devices ~extents ~cuts =
  let devices = Array.to_list devices and extents = Array.to_list extents in
  let cuts =
    Array.to_list
      (Array.map (fun (axis, over) -> (axis, Array.to_list over)) cuts)
  in
  let rank = List.length extents in
  let axes = List.map fst cuts and over = List.concat_map snd cuts in
  if List.exists (fun e -> e < 1) extents then Error "an extent is not positive"
  else if List.fold_left ( * ) 1 extents <> List.length devices then
    Error "the extents do not multiply to the number of devices"
  else if List.exists (fun k -> k < 0) devices || not (distinct devices) then
    Error "a device is negative or repeated"
  else if not (distinct axes && distinct over) then Error "an axis is cut twice"
  else if List.exists (fun a -> a < 0) axes then Error "a negative axis is cut"
  else if List.exists (fun g -> g < 0 || g >= rank) over then
    Error "a cut names a grid axis out of range"
  else
    Ok
      (normal devices extents
         (List.map (fun (axis, over) -> { axis; over }) cuts))

let devices = function One k -> [| k |] | Grid g -> Array.of_list g.devices
let count = function One _ -> 1 | Grid g -> List.length g.devices
let one = function One k -> Some k | Grid _ -> None

let tiles extents c =
  List.fold_left (fun n g -> n * List.nth extents g) 1 c.over

let is_cut = function One _ -> false | Grid { cuts; _ } -> cuts <> []

let cuts = function
  | One _ -> [||]
  | Grid { extents; cuts; _ } ->
      Array.of_list (List.map (fun c -> (c.axis, tiles extents c)) cuts)

(* For each cut, the axis, its tile count and the tile the [i]th device
   holds. *)
let tile_index g i =
  match g with
  | One _ -> []
  | Grid { extents; cuts; _ } ->
      let e = Array.of_list extents in
      let coord = Array.make (Array.length e) 0 and r = ref i in
      for a = Array.length e - 1 downto 0 do
        coord.(a) <- !r mod e.(a);
        r := !r / e.(a)
      done;
      List.map
        (fun c ->
          ( c.axis,
            tiles extents c,
            List.fold_left (fun j a -> (j * e.(a)) + coord.(a)) 0 c.over ))
        cuts

let check_position g i =
  if i < 0 || i >= count g then
    invalid_arg
      (Printf.sprintf "Grid: position %d of a grid of %d devices" i (count g))

let window g shape i =
  check_position g i;
  let rank = Array.length shape in
  let w =
    Array.map (fun n -> { Nx_array.Move.start = 0; count = n; step = 1 }) shape
  in
  let rec cut = function
    | [] -> Ok w
    | (axis, n, j) :: rest ->
        if axis >= rank then
          Error
            (Printf.sprintf "axis %d is cut and the shape has rank %d" axis rank)
        else if shape.(axis) mod n <> 0 then
          Error
            (Printf.sprintf "axis %d of extent %d does not split evenly over %d"
               axis shape.(axis) n)
        else
          let size = shape.(axis) / n in
          w.(axis) <- { Nx_array.Move.start = j * size; count = size; step = 1 };
          cut rest
  in
  cut (tile_index g i)

let map_axes f = function
  | One _ as g -> g
  | Grid g ->
      let cuts = List.map (fun c -> { c with axis = f c.axis }) g.cuts in
      Grid
        { g with cuts = List.sort (fun a b -> Int.compare a.axis b.axis) cuts }

let select g ~axis j =
  match g with
  | One _ -> g
  | Grid { devices; extents; cuts } -> (
      match List.find_opt (fun c -> c.axis = axis) cuts with
      | None -> g
      | Some cut ->
          let n = tiles extents cut in
          if j < 0 || j >= n then
            invalid_arg
              (Printf.sprintf "Grid.select: tile %d of an axis cut in %d" j n);
          let keep =
            List.filteri
              (fun i _ ->
                List.exists
                  (fun (a, _, t) -> a = axis && t = j)
                  (tile_index g i))
              devices
          in
          let gone = List.sort (fun a b -> Int.compare b a) cut.over in
          let extents =
            List.filteri (fun a _ -> not (List.mem a gone)) extents
          in
          let cuts =
            List.fold_left
              (fun cuts a -> without a cuts)
              (List.filter (fun c -> c.axis <> axis) cuts)
              gone
          in
          normal keep extents cuts)

let uncut g ~axis =
  match g with
  | One _ -> g
  | Grid { devices; extents; cuts } ->
      normal devices extents (List.filter (fun c -> c.axis <> axis) cuts)

(* Two grids are equal when every device holds the same window under both,
   whatever the shape: the same tile of the same count along every cut axis. *)
let equal g g' =
  let d = devices g and d' = devices g' in
  let tiles g i = List.sort compare (tile_index g i) in
  Array.length d = Array.length d'
  && Array.for_all Fun.id
       (Array.mapi
          (fun i k ->
            match Array.find_index (( = ) k) d' with
            | Some i' -> tiles g i = tiles g' i'
            | None -> false)
          d)

let pp pp_device ppf g =
  let list ppf ds =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
      pp_device ppf ds
  in
  match g with
  | One k -> pp_device ppf k
  | Grid { devices; extents = [ _ ]; cuts = [] } ->
      Format.fprintf ppf "on [%a]" list devices
  | Grid { devices; extents = [ _ ]; cuts = [ { axis; _ } ] } ->
      Format.fprintf ppf "split ~axis:%d [%a]" axis list devices
  | Grid { devices; extents; cuts } ->
      let ints sep =
        Format.pp_print_list
          ~pp_sep:(fun ppf () -> Format.pp_print_string ppf sep)
          Format.pp_print_int
      in
      Format.fprintf ppf "mesh %a [%a]" (ints "x") extents list devices;
      List.iter
        (fun c -> Format.fprintf ppf " ~axis:%d/%a" c.axis (ints ",") c.over)
        cuts
