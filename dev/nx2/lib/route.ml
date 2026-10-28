(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module M = Nx_array.Move

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

type axis =
  | Becomes of int
  | Whole
  | Reduced of int
  | Blocks of { factor : int; extent : int }
  | Scanned of int
  | Regroups of { axis : int; extent : int }

type rule = Elementwise | Axes of axis array array

type compute = Windows | Partials

type 'd t = {
  operands : 'd Devices.placement array;
  result : 'd Devices.placement;
  compute : compute;
}

(* The operands' cuts join over the union's axes: the results', then each
   factor [f] that [rule] reduces at [factor_axis f], past every axis a value
   has. *)
let factor_axis f = Nx_array.Layout.max_rank + f
let is_factor a = a >= Nx_array.Layout.max_rank

(* The union's axis that a cut of axis [a] of operand [i], of shape [s], into
   [t] tiles lands on: the result axis it becomes or scans, the one it
   regroups into where [t] divides both extents, or its factor, in blocks
   where [t] divides both extents. *)
let lands rule i s a t =
  let divides extent = s.(a) mod t = 0 && extent mod t = 0 in
  match rule with
  | Elementwise -> Some a
  | Axes r -> (
      match r.(i).(a) with
      | Becomes a' | Scanned a' -> Some a'
      | Reduced f -> Some (factor_axis f)
      | Blocks { factor; extent } ->
          if divides extent then Some (factor_axis factor) else None
      | Regroups { axis; extent } -> if divides extent then Some axis else None
      | Whole -> None)

let product s lo hi =
  let p = ref 1 in
  for a = lo to hi - 1 do
    p := !p * s.(a)
  done;
  !p

(* What [m] makes of axis [a] of a value of shape [s]. A reshape pairs the
   axes of one product of the extents before them in order, each with the
   result's next axis of that product and of its extent: [1; 8] to [1; 1; 8]
   pairs 0 with 0 and 1 with 2, and [1; 1] to [1] pairs 0 alone. An unpaired
   axis regroups into the result's axis of more than one element of that
   product: [8] into [2; 4]'s axis 0, [2; 4]'s axis 0 into [8], and [8] into
   [1; 2; 4]'s axis 1. Each window of a cut into tiles that divide both
   extents holds whole rows on both sides, so each shard reshapes alone: [8]
   over two devices holds [4] on each, which becomes [1; 4]. *)
let moved (m : M.t) s a : axis =
  match m with
  | Permute p -> (
      match Iarray.find_index (( = ) a) p with
      | Some a' -> Becomes a'
      | None -> Whole)
  | Broadcast s' ->
      let a' = a + Iarray.length s' - Array.length s in
      if s.(a) = Iarray.get s' a' then Becomes a' else Whole
  | Reshape s' -> (
      let s' = Iarray.to_array s' in
      let before = product s 0 a in
      let run sh =
        List.filter
          (fun x -> product sh 0 x = before)
          (List.init (Array.length sh) Fun.id)
      in
      let rec next e = function
        | [] -> None
        | y :: ys -> if s'.(y) = e then Some (y, ys) else next e ys
      in
      let rec pair xs ys =
        match xs with
        | [] -> None
        | x :: xs -> (
            match next s.(x) ys with
            | Some (y, ys') -> if x = a then Some y else pair xs ys'
            | None -> if x = a then None else pair xs ys)
      in
      match pair (run s) (run s') with
      | Some a' -> Becomes a'
      | None when before > 0 && s.(a) > 1 -> (
          match List.find_opt (fun y -> s'.(y) > 1) (run s') with
          | Some y -> Regroups { axis = y; extent = s'.(y) }
          | None -> Whole)
      | None -> Whole)
  | Slice rs ->
      let r = Iarray.get rs a in
      if r.start = 0 && r.step = 1 && r.count = s.(a) then Becomes a else Whole
  | Window ws ->
      (* CR: Preserve this axis's cut for a window of size 1 and step 1.
         It keeps every element in order and only appends a unit axis.
         sliding_window ~window:1 of [8] split over two devices currently
         gathers two full [8] copies; each [4] shard can give a [4;1] view.
         Recognize the identity axis here: localize and Layout.move already
         handle it, preserving the Window primitive's other semantics. *)
      if Iarray.exists (fun (w : M.window) -> w.axis = a) ws then Whole
      else Becomes a

let full d = { M.start = 0; count = d; step = 1 }

let window rule ~operand:i shape (w : M.range array) =
  match rule with
  | Elementwise -> w
  | Axes t ->
      Array.mapi
        (fun a d ->
          match t.(i).(a) with
          | Becomes a' | Scanned a' -> w.(a')
          | Regroups { axis; extent } ->
              let r = w.(axis) in
              let scale n = n * d / extent in
              { r with start = scale r.start; count = scale r.count }
          | Whole | Reduced _ | Blocks _ -> full d)
        shape

let uncut_all axes g = Array.fold_left (fun g axis -> Grid.uncut g ~axis) g axes

(* The tiles of [g]'s cut axis [a]. *)
let tiles g a =
  Option.get
    (Array.find_map (fun (c, t) -> if c = a then Some t else None) (Grid.cuts g))

(* Operand [i]'s grid [g], of shape [s], over the union's axes: a cut that
   lands on none of them holds copies. [Grid.map_axes] renumbers the cut axes
   alone, each one that lands. *)
let forward rule i s g =
  let lost =
    Array.of_list
      (List.filter_map
         (fun (a, t) -> if lands rule i s a t = None then Some a else None)
         (Array.to_list (Grid.cuts g)))
  in
  let g = uncut_all lost g in
  Grid.map_axes (fun a -> Option.get (lands rule i s a (tiles g a))) g

(* The grid [u] over the union's axes moved back to operand [i]'s, of shape
   [s]: a cut of an axis that no axis of the operand lands on holds copies. *)
let backward rule i s u =
  let source a' t =
    let rec find a =
      if a = Array.length s then None
      else if lands rule i s a t = Some a' then Some a
      else find (a + 1)
    in
    find 0
  in
  let orphans =
    Array.of_list
      (List.filter_map
         (fun (a', t) -> if source a' t = None then Some a' else None)
         (Array.to_list (Grid.cuts u)))
  in
  let u = uncut_all orphans u in
  Grid.map_axes (fun a' -> Option.get (source a' (tiles u a'))) u

(* [target] with each cut undone that an operand of [shapes] regroups into
   across rows: its tiles do not divide both extents. *)
let aligned rule shapes target =
  match rule with
  | Elementwise -> target
  | Axes r ->
      let across (a', t) =
        let straddles i s =
          Array.exists
            (fun a ->
              match r.(i).(a) with
              | Regroups { axis; extent } ->
                  axis = a' && (s.(a) mod t <> 0 || extent mod t <> 0)
              | Becomes _ | Scanned _ | Whole | Reduced _ | Blocks _ -> false)
            (Array.init (Array.length s) Fun.id)
        in
        Array.exists Fun.id (Array.mapi straddles shapes)
      in
      let undone =
        Array.of_list
          (List.filter_map
             (fun ((a', _) as c) -> if across c then Some a' else None)
             (Array.to_list (Grid.cuts target)))
      in
      uncut_all undone target

(* The axes of operand [i], of rank [rank], that [rule] reduces or scans. *)
let folds rule i rank =
  match rule with
  | Elementwise -> [||]
  | Axes t ->
      Array.of_list
        (List.filter
           (fun a ->
             match t.(i).(a) with
             | Reduced _ | Blocks _ | Scanned _ -> true
             | Becomes _ | Whole | Regroups _ -> false)
           (List.init rank Fun.id))

(* Whether an axis [rule] folds of an operand of [shapes] has no element. *)
let folds_empty rule shapes =
  Array.exists Fun.id
    (Array.mapi
       (fun i s ->
         Array.exists (fun a -> s.(a) = 0) (folds rule i (Array.length s)))
       shapes)

(* How the devices compute over operands of [shapes] read at [operands]: by
   partials where one is read cut along an axis [rule] folds, each such axis
   holding elements, since the window law fails there; by windows
   otherwise. *)
let computed rule shapes operands =
  let cut i p =
    let axes = folds rule i (Array.length shapes.(i)) in
    Array.exists (fun (a, _) -> Array.mem a axes) (Grid.cuts (Devices.grid p))
  in
  if
    Array.exists Fun.id (Array.mapi cut operands)
    && not (folds_empty rule shapes)
  then Partials
  else Windows

(* The union [u] without its factors' cuts: the results' grid, where the grid
   axes that cut a factor hold copies. *)
let results u =
  Array.fold_left
    (fun g (a, _) -> if is_factor a then Grid.uncut g ~axis:a else g)
    u (Grid.cuts u)

(* Whether [g]'s devices are among [g']'s. *)
let among g g' =
  let ks = Grid.devices g' in
  Array.for_all (fun k -> Array.mem k ks) (Grid.devices g)

(* The union [u] of the operands before one whose grid over the union's axes
   is [l], joined with it: [u] where [l] cuts nothing and lies on [u]'s
   devices, [l] where [u] cuts nothing and lies on [l]'s, as for an operand
   alone on one device beside one whole on every device, and their
   {!Grid.union} otherwise. *)
let join u l =
  if (not (Grid.is_cut l)) && among u l then Some u
  else if (not (Grid.is_cut u)) && among l u then Some l
  else Grid.union u l

let empty (w : M.range array) =
  Array.exists (fun (r : M.range) -> r.count = 0) w

let contains (outer : M.range array) (inner : M.range array) =
  Array.for_all2
    (fun (o : M.range) (i : M.range) ->
      i.start >= o.start && i.start + i.count <= o.start + o.count)
    outer inner

(* Whether an operand of shape [s] at [p] holds, on each device of the grid
   [q], the window [q] gives that device. An empty window reads nothing. *)
let holds p s q =
  let have = Devices.grid p in
  let ks = Grid.devices have in
  let held j k =
    match Grid.window q s j with
    | Error _ -> false
    | Ok w -> (
        empty w
        ||
        match Array.find_index (( = ) k) ks with
        | None -> false
        | Some j' -> (
            match Grid.window have s j' with
            | Ok h -> contains h w
            | Error _ -> false))
  in
  Array.for_all Fun.id (Array.mapi held (Grid.devices q))

(* Refusals *)

(* The name the remedy of a refusal gives operand [i]. *)
let name = function
  | 0 -> "x"
  | 1 -> "y"
  | 2 -> "z"
  | i -> "x" ^ string_of_int i

let pp_list pp ppf l =
  Format.fprintf ppf "[ %a ]"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       pp)
    l

(* The [Nx.Placement] expression that makes [q], a grid over [set], for the
   set named [s]: a mesh of [q]'s extents, its axes named [g0], [g1], …,
   where no simpler one does. A grid of several devices lists the set's
   devices in order, as every placement nx makes does, and so does
   [Mesh.v]. *)
let pp_call ~by set ppf q =
  let mesh_axis g = Printf.sprintf "%S" ("g" ^ string_of_int g) in
  let split a = Devices.grid (Devices.split ~by ~axis:a set) in
  match (Grid.one q, Grid.cuts q) with
  | _, [||] when Grid.count q = Devices.count set ->
      Format.pp_print_string ppf "Placement.on s"
  | Some k, _ ->
      Format.fprintf ppf "Placement.device s %s" (Rig.name (Devices.rig set k))
  | None, [| (a, _) |] when Grid.equal q (split a) ->
      Format.fprintf ppf "Placement.split ~axis:%d s" a
  | None, cuts ->
      let axes =
        List.mapi (fun g e -> (g, e)) (Array.to_list (Grid.extents q))
      in
      let cut (a, _) = (a, Array.to_list (Grid.over q a)) in
      Format.fprintf ppf "Placement.mesh (Mesh.v s %a) %a"
        (pp_list (fun ppf (g, e) ->
             Format.fprintf ppf "(%s, %d)" (mesh_axis g) e))
        axes
        (pp_list (fun ppf (a, over) ->
             Format.fprintf ppf "(%d, %a)" a
               (pp_list Format.pp_print_string)
               (List.map mesh_axis over)))
        (List.map cut (Array.to_list cuts))

(* Raises: operand [i], at [p], is read at the union [u] moved back to its
   axes, of shape [s], over [set]. *)
let refuse ~by rule set i p s u =
  let q = backward rule i s u in
  invalid_argf "%s: operand %d lies at %a and is read at %a; Nx.place (%a) %s"
    by i Devices.pp_placement p Devices.pp_placement (Devices.v ~by set q)
    (pp_call ~by set) q (name i)

(* Routes *)

(* The route of operands not all at one placement, or at one that cuts an
   axis an operation other than an elementwise one may read whole. *)
let general ~by rule ps shapes =
  let n = Array.length ps in
  let set = Devices.set ps.(0) in
  let other i = Devices.number (Devices.set ps.(i)) <> Devices.number set in
  let lifted i = forward rule i shapes.(i) (Devices.grid ps.(i)) in
  let refuse i u = refuse ~by rule set i ps.(i) shapes.(i) u in
  (* The operands join in order. One of another set adds no cut, and is
     refused below. One whose cuts do not join the union adds none, and is
     refused below unless it holds what it is read at. *)
  let u =
    List.fold_left
      (fun u i ->
        if other i then u else Option.value (join u (lifted i)) ~default:u)
      (lifted 0)
      (List.init (n - 1) succ)
  in
  let u = aligned rule shapes u in
  (* A folded axis of no element is read whole: no device folds a term. *)
  let u = if folds_empty rule shapes then results u else u in
  let operands =
    Array.mapi
      (fun i p ->
        let q = backward rule i shapes.(i) u in
        if other i then refuse i u
        else if Grid.equal (Devices.grid p) q then p
        else if holds p shapes.(i) q then Devices.v ~by set q
        else refuse i u)
      ps
  in
  (* The results lie at an operand's own placement where it is theirs, so
     that operations chained over one placement keep it physically. *)
  let r = results u in
  let result =
    let own p = Grid.equal (Devices.grid p) r in
    match Array.find_opt own ps with
    | Some p -> p
    | None -> Devices.v ~by set r
  in
  { operands; result; compute = computed rule shapes operands }

let route ~by rule ps shapes =
  for i = 0 to Array.length ps - 1 do
    match Grid.window (Devices.grid ps.(i)) shapes.(i) 0 with
    | Ok _ -> ()
    | Error e -> invalid_argf "%s: %s" by e
  done;
  (* Operands at one placement are read there, and the results lie there,
     where it cuts no axis or the operation is elementwise: each device reads
     its window of each and computes its window of the results. *)
  let p = ps.(0) in
  let alone = Array.for_all (fun q -> q == p) ps in
  let elementwise = match rule with Elementwise -> true | Axes _ -> false in
  if alone && (elementwise || not (Grid.is_cut (Devices.grid p))) then
    { operands = Array.make (Array.length ps) p; result = p; compute = Windows }
  else general ~by rule ps shapes
