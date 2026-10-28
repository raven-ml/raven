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

(* Tile [j] of [t] along an axis of extent [n]: the [n / t] elements after
   those of the tiles before it, one more where [j < n mod t]. Of 5 over 3,
   tiles hold 2, 2 and 1; of 2 over 3, 1, 1 and none. *)
let tile n t j =
  let q = n / t and r = n mod t in
  {
    Nx_array.Move.start = (j * q) + Int.min j r;
    count = (q + if j < r then 1 else 0);
    step = 1;
  }

let window g shape i =
  check_position g i;
  let rank = Array.length shape in
  let w =
    Array.map (fun n -> { Nx_array.Move.start = 0; count = n; step = 1 }) shape
  in
  let rec cut = function
    | [] -> Ok w
    | (axis, t, j) :: rest ->
        if axis >= rank then
          Error
            (Printf.sprintf "axis %d is cut and the shape has rank %d" axis rank)
        else begin
          w.(axis) <- tile shape.(axis) t j;
          cut rest
        end
  in
  cut (tile_index g i)

let map_axes f = function
  | One _ as g -> g
  | Grid g ->
      let cuts = List.map (fun c -> { c with axis = f c.axis }) g.cuts in
      Grid
        { g with cuts = List.sort (fun a b -> Int.compare a.axis b.axis) cuts }

let uncut g ~axis =
  match g with
  | One _ -> g
  | Grid { devices; extents; cuts } ->
      normal devices extents (List.filter (fun c -> c.axis <> axis) cuts)

(* The running products of [extents]: [2; 3; 2] gives [2; 6; 12]. *)
let products extents =
  List.rev
    (snd
       (List.fold_left
          (fun (p, acc) e -> (p * e, (p * e) :: acc))
          (1, []) extents))

(* The extents of the coarsest mesh both [e] and [e'] merge axes of, and for
   each, the mesh axes each of its axes spans: [2; 2] and [4] give [2; 2],
   with [[0]; [1]] and [[0; 1]]. A grid axis spans the mesh axes whose
   running product lies above its predecessor's and at most its own. [None]
   where no mesh refines both, as [2; 3] and [3; 2]. *)
let refine e e' =
  let ps = List.sort_uniq Int.compare (products e @ products e') in
  let rec ratios prev = function
    | [] -> Some []
    | p :: rest ->
        if p mod prev <> 0 then None
        else Option.map (fun r -> (p / prev) :: r) (ratios p rest)
  in
  let spans e =
    let his = products e in
    let los = 1 :: his in
    Array.of_list
      (List.mapi
         (fun i hi ->
           let lo = List.nth los i in
           List.filter_map
             (fun (j, p) -> if p > lo && p <= hi then Some j else None)
             (List.mapi (fun j p -> (j, p)) ps))
         his)
  in
  Option.map (fun extents -> (extents, spans e, spans e')) (ratios 1 ps)

let mesh = function
  | One k -> ([ k ], [], [])
  | Grid { devices; extents; cuts } -> (devices, extents, cuts)

let cut_like g ~like ~axes ~axis =
  let devices, eg, cg = mesh g and devices', el, cl = mesh like in
  if devices <> devices' then Error "the grids lay their devices otherwise"
  else
    match refine eg el with
    | None -> Error "the grids are of two meshes"
    | Some (extents, sg, sl) ->
        let over spans l = List.concat_map (fun h -> spans.(h)) l in
        let cuts = List.map (fun c -> { c with over = over sg c.over }) cg in
        let from a =
          match List.find_opt (fun c -> c.axis = a) cl with
          | Some c -> over sl c.over
          | None -> []
        in
        let added = List.concat_map from axes in
        let taken = List.concat_map (fun c -> c.over) cuts in
        if List.exists (fun c -> c.axis = axis) cuts then
          Error (Printf.sprintf "axis %d is cut already" axis)
        else if List.exists (fun h -> List.mem h taken) added then
          Error "a grid axis would cut two axes"
        else Ok (normal devices extents ({ axis; over = added } :: cuts))

let extents = function One _ -> [||] | Grid g -> Array.of_list g.extents

let over g axis =
  match g with
  | One _ -> [||]
  | Grid { cuts; _ } -> (
      match List.find_opt (fun c -> c.axis = axis) cuts with
      | Some c -> Array.of_list c.over
      | None -> [||])

let union g g' =
  match (g, g') with
  | One k, One k' -> if k = k' then Some g else None
  | One _, Grid _ | Grid _, One _ -> None
  | Grid a, Grid b -> (
      if a.devices <> b.devices then None
      else
        match refine a.extents b.extents with
        | None -> None
        | Some (extents, sa, sb) ->
            let spanned spans c =
              { c with over = List.concat_map (fun h -> spans.(h)) c.over }
            in
            let ca = List.map (spanned sa) a.cuts
            and cb = List.map (spanned sb) b.cuts in
            (* An axis both cut must be cut over the same grid axes; every
               other cut takes grid axes no cut takes. *)
            let alike c =
              match List.find_opt (fun c' -> c'.axis = c.axis) ca with
              | Some c' -> c'.over = c.over
              | None -> true
            in
            let fresh c = not (List.exists (fun c' -> c'.axis = c.axis) ca) in
            let cuts = ca @ List.filter fresh cb in
            if
              List.for_all alike cb
              && distinct (List.concat_map (fun c -> c.over) cuts)
            then Some (normal a.devices extents cuts)
            else None)

(* Two grids are equal when every device holds the same window under both,
   whatever the shape: the same tile of the same count along every cut axis.
   A [Grid] has at least two devices, so it never equals a [One], and grids
   of one device, the common case, compare without allocating. *)
let equal g g' =
  match (g, g') with
  | One k, One k' -> k = k'
  | One _, Grid _ | Grid _, One _ -> false
  | Grid _, Grid _ ->
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
