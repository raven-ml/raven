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
  | Indexed of int
  | Regroups of { axis : int; extent : int }

type rule = Elementwise | Axes of axis array array

type compute = Windows | Partials
type terms = Closes | Keeps | Sums

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
   [t] tiles lands on: the result axis it becomes, scans or indexes, the one
   it regroups into where [t] divides both extents, or its factor, in blocks
   where [t] divides both extents. *)
let lands rule i s a t =
  let divides extent = s.(a) mod t = 0 && extent mod t = 0 in
  match rule with
  | Elementwise -> Some a
  | Axes r -> (
      match r.(i).(a) with
      | Becomes a' | Scanned a' | Indexed a' -> Some a'
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

let extents (w : M.range array) =
  Iarray.init (Array.length w) (fun i -> (w.(i) : M.range).count)

(* A loop: every view asks it, and [Array.exists] allocates its own loop. *)
let empty (w : M.range array) =
  let i = ref 0 in
  while !i < Array.length w && (w.(!i) : M.range).count > 0 do
    incr i
  done;
  !i < Array.length w

(* A loop: [Array.for_all2] allocates its own. *)
let rec contains_from (outer : M.range array) (inner : M.range array) k =
  k = Array.length outer
  ||
  let o = outer.(k) and i = inner.(k) in
  i.start >= o.start
  && i.start + i.count <= o.start + o.count
  && contains_from outer inner (k + 1)

let contains outer inner =
  if Array.length outer <> Array.length inner then
    invalid_arg "Route.contains: windows of different ranks";
  contains_from outer inner 0

let window rule ~operand:i shape (w : M.range array) =
  match rule with
  | Elementwise -> w
  | Axes t ->
      Array.mapi
        (fun a d ->
          match t.(i).(a) with
          | Becomes a' | Scanned a' | Indexed a' -> w.(a')
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
              | Becomes _ | Scanned _ | Indexed _ | Whole | Reduced _ | Blocks _
                ->
                  false)
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

(* The axes of operand [i], of rank [rank], that [rule] reduces, scans or
   indexes. *)
let folds rule i rank =
  match rule with
  | Elementwise -> [||]
  | Axes t ->
      Array.of_list
        (List.filter
           (fun a ->
             match t.(i).(a) with
             | Reduced _ | Blocks _ | Scanned _ | Indexed _ -> true
             | Becomes _ | Whole | Regroups _ -> false)
           (List.init rank Fun.id))

(* Whether an axis [rule] folds of an operand of [shapes] has no element,
   but an indexed one: over a target of no element along it, an [Add]'s
   updates still fold, their positions all outside. *)
let folds_empty rule shapes =
  let indexed i a =
    match rule with
    | Axes t -> (
        match t.(i).(a) with
        | Indexed _ -> true
        | Becomes _ | Whole | Reduced _ | Blocks _ | Scanned _ | Regroups _ ->
            false)
    | Elementwise -> false
  in
  Array.exists Fun.id
    (Array.mapi
       (fun i s ->
         Array.exists
           (fun a -> s.(a) = 0 && not (indexed i a))
           (folds rule i (Array.length s)))
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

(* The union [u] without its factors' cuts, where the operation [terms]
   sums them: the grid axes that cut a factor sum the results. *)
let summed_results terms u =
  match terms with
  | Closes | Keeps -> results u
  | Sums ->
      Array.fold_left
        (fun g (a, _) -> if is_factor a then Grid.sum g ~axis:a else g)
        u (Grid.cuts u)

(* The grid of [g]'s groups and terms alone: every cut undone. *)
let sums g = uncut_all (Array.map fst (Grid.cuts g)) g

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
  let whole g = not (Grid.is_cut g || Grid.is_partial g) in
  if whole l && among u l then Some u
  else if whole u && among l u then Some l
  else Grid.union u l

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

(* Whether [p] is the whole on every device of its set. *)
let everywhere p =
  let g = Devices.grid p in
  (not (Grid.is_cut g || Grid.is_partial g))
  && Grid.count g = Devices.count (Devices.set p)

let gathers r i p s =
  let q = Devices.grid r.operands.(i) in
  (not (Grid.equal (Devices.grid p) q))
  && (not (everywhere p))
  && not (holds p s q)

(* Refusals *)

(* The name the remedy of a refusal gives operand [i]. *)
let name = function
  | 0 -> "x"
  | 1 -> "y"
  | 2 -> "z"
  | i -> "x" ^ string_of_int i

let pp_list pp ppf l =
  if l = [] then Format.pp_print_string ppf "[]"
  else
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
  let partial = Grid.is_partial q in
  match (Grid.one q, Grid.cuts q) with
  | _, [||] when Grid.count q = Devices.count set && not partial ->
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
      let names l = pp_list Format.pp_print_string ppf (List.map mesh_axis l) in
      if partial then Format.pp_print_string ppf "Placement.partial"
      else Format.pp_print_string ppf "Placement.mesh";
      Format.fprintf ppf " (Mesh.v s %a) "
        (pp_list (fun ppf (g, e) ->
             Format.fprintf ppf "(%s, %d)" (mesh_axis g) e))
        axes;
      if partial then begin
        names (Array.to_list (Grid.sums q));
        Format.pp_print_string ppf " "
      end;
      pp_list
        (fun ppf (a, over) ->
          Format.fprintf ppf "(%d, %a)" a
            (pp_list Format.pp_print_string)
            (List.map mesh_axis over))
        ppf
        (List.map cut (Array.to_list cuts))

(* Raises: operand [i], at [p], is read at the union [u] moved back to its
   axes, of shape [s], over [set]. The remedy places it there. *)
let refuse ~by rule set i p s u =
  let q = backward rule i s u in
  (* An operand that is no sum is read where the union lies closed. *)
  let q = if Devices.partial p then q else Grid.closed q in
  invalid_argf "%s: operand %d lies at %a and is read at %a; Nx.place (%a) %s"
    by i Devices.pp_placement p Devices.pp_placement (Devices.v ~by set q)
    (pp_call ~by set) q (name i)

(* Raises: operand [i] lies at [p], a sum the operation does not keep. The
   remedy closes it where it lies. *)
let unkept ~by i p =
  let set = Devices.set p in
  invalid_argf
    "%s: operand %d lies at %a, a sum the operation does not keep; Nx.place \
     (%a) %s"
    by i Devices.pp_placement p (pp_call ~by set)
    (Grid.closed (Devices.grid p))
    (name i)

(* Routes *)

(* The route of operands joined in [order], or the refusal of the first
   operand it cannot read where it lies. *)
let attempt ~by ~terms rule ps shapes order =
  let set = Devices.set ps.(0) in
  let other i = Devices.number (Devices.set ps.(i)) <> Devices.number set in
  let lifted i = forward rule i shapes.(i) (Devices.grid ps.(i)) in
  (* One of another set adds no cut, and is refused below. One whose cuts do
     not join the union adds none, and is refused below unless it holds what
     it is read at. *)
  let order = List.filter (fun i -> not (other i)) order in
  let u =
    List.fold_left
      (fun u i -> Option.value (join u (lifted i)) ~default:u)
      (lifted (List.hd order))
      (List.tl order)
  in
  let u = aligned rule shapes u in
  (* A folded axis of no element is read whole: no device folds a term. *)
  let u = if folds_empty rule shapes then results u else u in
  (* An operand that is no sum is read where the union lies without its sum:
     the devices then add it to the first term of each group alone. *)
  let read i p =
    let q = backward rule i shapes.(i) u in
    let q = if Devices.partial p then q else Grid.closed q in
    (* A sum's terms are read only as the terms of the same groups. *)
    let terms () = Grid.equal (sums (Devices.grid p)) (sums q) in
    if other i then None
    else if Grid.equal (Devices.grid p) q then Some p
    else if holds p shapes.(i) q && ((not (Devices.partial p)) || terms ())
    then Some (Devices.v ~by set q)
    else None
  in
  let reads = Array.mapi read ps in
  match Array.find_index Option.is_none reads with
  | Some i ->
      Error (fun () -> refuse ~by rule set i ps.(i) shapes.(i) u)
  | None ->
      let operands = Array.map Option.get reads in
      (* The results lie at an operand's own placement where it is theirs,
         so that operations chained over one placement keep it
         physically. *)
      let r = summed_results terms u in
      let result =
        let own p = Grid.equal (Devices.grid p) r in
        match Array.find_opt own ps with
        | Some p -> p
        | None -> Devices.v ~by set r
      in
      Ok { operands; result; compute = computed rule shapes operands }

(* The route of operands not all at one placement, or at one that cuts an
   axis an operation other than an elementwise one may read whole. The
   operands join in order. Where that refuses one, those of positions join
   last, so that the values they read keep their cuts. Where that refuses
   too, the positions not whole on every device of the set join as if they
   were, so that the union is the values' alone and each device reads the
   positions it needs, which then move there. *)
let general ~by ~positions ~terms rule ps shapes =
  let attempt = attempt ~terms in
  let all = List.init (Array.length ps) Fun.id in
  match attempt ~by rule ps shapes all with
  | Ok r -> r
  | Error first -> (
      let indices, values =
        List.partition (fun i -> List.mem i positions) all
      in
      if indices = [] then first ()
      else
        match attempt ~by rule ps shapes (values @ indices) with
        | Ok r -> r
        | Error last -> (
            let set = Devices.set ps.(0) in
            let same i =
              Devices.number (Devices.set ps.(i)) = Devices.number set
            in
            let gathered =
              List.filter (fun i -> same i && not (everywhere ps.(i))) indices
            in
            if gathered = [] then last ()
            else
              let on = Devices.on set in
              let moved i p = if List.mem i gathered then on else p in
              let ps = Array.mapi moved ps in
              match attempt ~by rule ps shapes (values @ indices) with
              | Ok r -> r
              | Error refused -> refused ()))

(* Raises unless [terms] keeps each partial operand of [ps] and they are
   sums over the same groups. *)
let summed ~by terms ps =
  let partial = List.filter (fun i -> Devices.partial ps.(i))
      (List.init (Array.length ps) Fun.id) in
  match (terms, partial) with
  | _, [] -> ()
  | Closes, i :: _ -> unkept ~by i ps.(i)
  | (Keeps | Sums), i :: rest ->
      let g = sums (Devices.grid ps.(i)) in
      List.iter
        (fun j ->
          if not (Grid.equal g (sums (Devices.grid ps.(j)))) then
            invalid_argf
              "%s: operand %d lies at %a, a sum over other groups than \
               operand %d's; Nx.place (%a) %s"
              by j Devices.pp_placement ps.(j) i
              (pp_call ~by (Devices.set ps.(j)))
              (Grid.closed (Devices.grid ps.(j)))
              (name j))
        rest

let route ~by ?(positions = []) ?(terms = Closes) rule ps shapes =
  for i = 0 to Array.length ps - 1 do
    match Grid.window (Devices.grid ps.(i)) shapes.(i) 0 with
    | Ok _ -> ()
    | Error e -> invalid_argf "%s: %s" by e
  done;
  summed ~by terms ps;
  (* Operands at one placement are read there, and the results lie there,
     where it cuts no axis or the operation is elementwise: each device reads
     its window of each and computes its window of the results. *)
  let p = ps.(0) in
  let alone = Array.for_all (fun q -> q == p) ps in
  let elementwise = match rule with Elementwise -> true | Axes _ -> false in
  if alone && (elementwise || not (Grid.is_cut (Devices.grid p))) then
    { operands = Array.make (Array.length ps) p; result = p; compute = Windows }
  else
    let r = general ~by ~positions ~terms rule ps shapes in
    (* A fold over a cut axis closes its partials, which a sum's terms are
       not; a contraction's are terms of its own. *)
    (match (r.compute, terms) with
    | Partials, Sums | Windows, _ -> ()
    | Partials, (Closes | Keeps) ->
        Array.iteri (fun i p -> if Devices.partial p then unkept ~by i p) ps);
    r
