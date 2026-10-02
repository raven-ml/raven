(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Resolving

   [resolve] arranges the figure, evaluating binds, assigning ids, forming
   scopes and broadcasting layers over grids; reads the channels of each
   occurrence and merges the specifications of each scale; summarises each
   occurrence's data where it lives; and fits the scales, categorical ones first
   since facets make panels. Each occurrence's rows are put among its cell's
   panels once, by category index, and each panel records the scale every
   binding reads in it, which layout and drawing look up. An occurrence is a
   mark in one cell: a mark that a layer broadcasts over a grid occurs once per
   cell, and reads the scales of each. *)

module Scale = Hugin_kit.Scale
module Text = Hugin_text.Text
open Common
open Channel
open Figure
open Arrange

(* Readings: the channels that read scales, each with its scale's identity and
   scope. *)

type 'd member = {
  m_occ : occ;
  m_pid : id;
  m_index : int;
  m_role : string;
  m_use : Role.use;
  m_d : 'd data;
  m_imply : 'd Scale.t option;
  m_guide : bool option;
}

type reading =
  | R : {
      m : 'd member;
      kind : 'd kind;
      mapped : bool;
      name : string;
      key : key;
    }
      -> reading

let is_map : type d r. (d, r) Channel.t -> bool = function
  | Map _ -> true
  | Const _ | Data _ -> false

let readings_of pid occ =
  let env = { shares = occ.shares; pending = []; cell = pid } in
  List.concat
    (List.mapi
       (fun index (B b) ->
         match (data b.ch, Role.scale b.role.use) with
         | Some d, Some default ->
             let name = Option.value ~default (Option.bind d.spec Scale.name) in
             let facet =
               match Role.shown_on b.role.use with
               | Some (`Header _) -> true
               | _ -> false
             in
             let key =
               if not (List.mem name occ.per_panel) then key_of env name
               else if facet then
                 err "resolve"
                   "%a makes its facet scale %S independent per panel" pp_id
                   occ.mid name
               else Panels_of (occ.mid, pid)
             in
             [
               R
                 {
                   m =
                     {
                       m_occ = occ;
                       m_pid = pid;
                       m_index = index;
                       m_role = b.role.name;
                       m_use = b.role.use;
                       m_d = d;
                       m_imply = b.imply;
                       m_guide = b.guide;
                     };
                   kind = kind d.lift;
                   mapped = is_map b.ch;
                   name;
                   key;
                 };
             ]
         | _ -> [])
       occ.mark.bindings)

type group =
  | G : {
      name : string;
      key : key;
      kind : 'd kind;
      members : 'd member list; (* In figure order. *)
      legend : bool;
          (* A role other than a position or facet reads it without
             map_range. *)
    }
      -> group

let group readings =
  let add groups (R r) =
    let legend = Role.shown_on r.m.m_use = Some `Legend && not r.mapped in
    let rec go = function
      | [] ->
          [
            G
              {
                name = r.name;
                key = r.key;
                kind = r.kind;
                members = [ r.m ];
                legend;
              };
          ]
      | (G g as gr) :: rest -> (
          if not (String.equal g.name r.name && equal_key g.key r.key) then
            gr :: go rest
          else
            match equal_kind g.kind r.kind with
            | Some Type.Equal ->
                G
                  {
                    g with
                    members = r.m :: g.members;
                    legend = g.legend || legend;
                  }
                :: rest
            | None when Role.by_kind r.name -> gr :: go rest
            | None ->
                let m = List.hd g.members in
                err "resolve" "the scale %S is read as %a by %a and as %a by %a"
                  r.name pp_kind g.kind pp_id m.m_occ.mid pp_kind r.kind pp_id
                  r.m.m_occ.mid)
    in
    go groups
  in
  List.fold_left add [] readings
  |> List.rev_map (fun (G g) -> G { g with members = List.rev g.members })
  |> List.rev

(* [same rd rd'] is [true] iff [rd] and [rd'] read one scale. *)
let same (R r) (R r') =
  String.equal r.name r'.name
  && equal_key r.key r'.key
  && ((not (Role.by_kind r.name)) || Option.is_some (equal_kind r.kind r'.kind))

(* In a panel, the channels on x read one scale, and likewise y, fx and fy. *)
let check_panel_scales readings =
  let rec go seen = function
    | [] -> ()
    | (R r as rd) :: rest ->
        (match Role.shown_on r.m.m_use with
        | None | Some `Legend -> ()
        | Some on -> (
            match
              List.find_opt
                (fun (R r') ->
                  Nx.Ptree.Path.equal r'.m.m_pid r.m.m_pid
                  && Role.shown_on r'.m.m_use = Some on)
                seen
            with
            | Some (R o as other) when not (same other rd) ->
                err "resolve" "%a and %a read two %s scales in the panel %a"
                  pp_id o.m.m_occ.mid pp_id r.m.m_occ.mid
                  (Option.get (Role.scale r.m.m_use))
                  pp_id r.m.m_pid
            | _ -> ()));
        go (rd :: seen) rest
  in
  go [] readings

let check_coords cells =
  List.iter
    (fun (pid, c) ->
      match c.coords with
      | (id, k) :: rest -> (
          match List.find_opt (fun (_, k') -> not (Coord.equal k k')) rest with
          | Some (id', _) ->
              err "resolve"
                "the panel %a lies under two coordinate systems, at %a and %a"
                pp_id pid pp_id id pp_id id'
          | None -> ())
      | [] -> (
          let implied =
            List.filter_map
              (fun o -> Option.map (fun k -> (o.mid, k)) o.mark.coord)
              c.occs
          in
          match implied with
          | (id, k) :: rest -> (
              match
                List.find_opt (fun (_, k') -> not (Coord.equal k k')) rest
              with
              | Some (id', _) ->
                  err "resolve"
                    "%a and %a imply two coordinate systems in the panel %a"
                    pp_id id pp_id id' pp_id pid
              | None -> ())
          | [] -> ()))
    cells

let check_axes cells readings =
  List.iter
    (fun (pid, c) ->
      let axes = List.filter (fun (_, g, _) -> is_axis g) c.guides in
      List.iter
        (fun (gid, (a : guide), _) ->
          List.iter
            (fun (_, (a' : guide), _) ->
              if String.equal a.scale a'.scale && not (equal_guide a a') then
                err "resolve" "the panel %a holds two different axes for %S"
                  pp_id pid a.scale)
            axes;
          let on =
            List.filter_map
              (fun (R r) ->
                match Role.shown_on r.m.m_use with
                | Some ((`Axis _ | `Header _) as on)
                  when Nx.Ptree.Path.equal r.m.m_pid pid
                       && String.equal r.name a.scale ->
                    Some on
                | _ -> None)
              readings
          in
          if on = [] then
            err "resolve"
              "the axis %a names %S, no position or facet scale of its panel"
              pp_id gid a.scale;
          (* An axis runs along the direction of its scale's position. *)
          match a.side with
          | Some ((`Left | `Right) as side) when List.mem (`Axis Role.X) on ->
              err "resolve" "the axis %a of %S is on the %a side" pp_id gid
                a.scale pp_side side
          | Some ((`Top | `Bottom) as side) when List.mem (`Axis Role.Y) on ->
              err "resolve" "the axis %a of %S is on the %a side" pp_id gid
                a.scale pp_side side
          | _ -> ())
        axes)
    cells

(* Merging specifications *)

let merge_level name level specs =
  let rec go acc prior = function
    | [] -> acc
    | (m, s) :: rest ->
        let acc =
          match acc with
          | None -> Some s
          | Some a -> (
              match Scale.merge a s with
              | Ok a -> Some a
              | Error p ->
                  let m0 =
                    match
                      List.find_opt
                        (fun (_, s0) -> Result.is_error (Scale.merge s0 s))
                        prior
                    with
                    | Some (m0, _) -> m0
                    | None -> m
                  in
                  err "resolve"
                    "%a and %a give the scale %S two %s values of %a" pp_id
                    m0.m_occ.mid pp_id m.m_occ.mid name level Scale.pp_property
                    p)
        in
        go acc ((m, s) :: prior) rest
  in
  go None [] specs

let default_spec : type d. d kind -> d Scale.t = function
  | Quantities -> Scale.linear ()
  | Categories -> Scale.band ()

(* [merged kind name ms] is the specification of the scale [ms] read: their
   explicit specifications merged, then implied ones under them. *)
let merged : type d. d kind -> string -> d member list -> d Scale.t =
 fun kind name ms ->
  let base = default_spec kind in
  let explicit =
    List.filter_map (fun m -> Option.map (fun s -> (m, s)) m.m_d.spec) ms
  in
  let implied : (d member * d Scale.t) list =
    List.concat_map
      (fun m ->
        List.map
          (fun i -> (m, i))
          (Option.to_list m.m_imply
          @ Option.to_list (Role.implied m.m_use (Scale.kind base))))
      ms
  in
  (* [imply] keeps the name and transform of the explicit specification, so an
     implied one gives only its other properties, and only those can
     conflict. *)
  let implied = List.map (fun (m, i) -> (m, Scale.imply i base)) implied in
  let explicit =
    Option.value ~default:base (merge_level name "explicit" explicit)
  in
  let spec =
    match merge_level name "implied" implied with
    | None -> explicit
    | Some i -> Scale.imply i explicit
  in
  (* Labelled and indexed categories identify categories differently. *)
  (match kind with
  | Categories -> (
      let sort m = Lift.labelled m.m_d.lift in
      (match ms with
      | m :: rest -> (
          match List.find_opt (fun m' -> sort m' <> sort m) rest with
          | Some m' ->
              err "resolve"
                "%a and %a read labelled and indexed categories on the scale %S"
                pp_id m.m_occ.mid pp_id m'.m_occ.mid name
          | None -> ())
      | [] -> ());
      match (Scale.domain spec, ms) with
      | Scale.Categories c, m :: _ when Scale.sets Domain spec ->
          let domain_labelled =
            match c with Scale.Labels _ -> true | Scale.Indices _ -> false
          in
          if domain_labelled <> sort m then
            err "resolve"
              "the domain of the scale %S and %a identify categories \
               differently"
              name pp_id m.m_occ.mid
      | _ -> ())
  | Quantities -> ());
  spec

let merged_guide name ms =
  let rec go acc = function
    | [] -> Option.map snd acc
    | m :: rest -> (
        match (m.m_guide, acc) with
        | None, _ -> go acc rest
        | Some g, None -> go (Some (m, g)) rest
        | Some g, Some (m0, g0) ->
            if Bool.equal g g0 then go acc rest
            else
              err "resolve" "%a and %a imply different guides for the scale %S"
                pp_id m0.m_occ.mid pp_id m.m_occ.mid name)
  in
  go None ms

(* Fitting *)

let by_order ms =
  List.stable_sort
    (fun m m' ->
      let c = Int.compare m.m_occ.order m'.m_occ.order in
      if c <> 0 then c else Int.compare m.m_index m'.m_index)
    ms

let categories name (ms : string member list) summary_of =
  match ms with
  | m :: _ when Lift.labelled m.m_d.lift ->
      let seen = Hashtbl.create 16 and labels = ref [] in
      let add l =
        if not (Hashtbl.mem seen l) then (
          Hashtbl.add seen l ();
          labels := l :: !labels)
      in
      List.iter
        (fun m ->
          match m.m_d.lift with
          | Cat { labels = Some l; _ } | Strings l -> Array.iter add l
          | Cat { labels = None; _ } | Dim _ -> ())
        (by_order ms);
      Scale.Labels (Array.of_list (List.rev !labels))
  | _ ->
      let ints = Hashtbl.create 16 and texts = Hashtbl.create 16 in
      let text m i s =
        match Hashtbl.find_opt texts i with
        | Some (s0, m0) when not (String.equal s s0) ->
            err "resolve"
              "%a and %a give the category %d of the scale %S two texts, %S \
               and %S"
              pp_id m0.m_occ.mid pp_id m.m_occ.mid i name s0 s
        | Some _ -> ()
        | None -> Hashtbl.add texts i (s, m)
      in
      List.iter
        (fun m ->
          match m.m_d.lift with
          | Dim { axis; labels; _ } ->
              let shape = m.m_occ.mark.shape in
              let a = Option.get (axis_of shape axis) in
              for i = 0 to shape.(a) - 1 do
                Hashtbl.replace ints i ();
                Option.iter (fun l -> text m i l.(i)) labels
              done
          | Cat { labels = None; _ } ->
              Option.iter
                (List.iter (fun i -> Hashtbl.replace ints i ()))
                (List.assoc_opt m.m_index
                   (summary_of m.m_occ m.m_pid).Summary.codes)
          | Cat { labels = Some _; _ } | Strings _ -> ())
        (by_order ms);
      let ints =
        List.sort Int.compare (Hashtbl.fold (fun i () acc -> i :: acc) ints [])
      in
      let shown i =
        match Hashtbl.find_opt texts i with
        | Some (s, _) -> s
        | None -> string_of_int i
      in
      Scale.Indices (Array.of_list (List.map (fun i -> (i, shown i)) ints))

let fit_scale : type d.
    d kind ->
    string ->
    d member list ->
    d Scale.t ->
    (occ -> id -> Summary.t) ->
    d Scale.t =
 fun kind name ms spec summary_of ->
  match kind with
  | Quantities ->
      let hull acc m =
        match
          ( List.assoc_opt m.m_index (summary_of m.m_occ m.m_pid).Summary.hulls,
            acc )
        with
        | None, acc -> acc
        | Some h, None -> Some h
        | Some (lo, hi), Some (a, b) -> Some (Float.min a lo, Float.max b hi)
      in
      let observed = List.fold_left hull None ms in
      Scale.fit
        (Option.map (fun (lo, hi) -> Scale.Floats (lo, hi)) observed)
        spec
  | Categories ->
      Scale.fit (Some (Scale.Categories (categories name ms summary_of))) spec

type fitted =
  | F : {
      name : string;
      key : key;
      kind : 'd kind;
      members : 'd member list;
      legend : bool;
      guide : bool option;
      spec : 'd Scale.t; (* Merged. *)
      scale : 'd Scale.t; (* Fitted, then zoomed. *)
    }
      -> fitted

(* Facets *)

let find_path id l =
  List.find_map
    (fun (id', v) -> if Nx.Ptree.Path.equal id id' then Some v else None)
    l

type facet = Every | One of int option | Each of Nx.int64_t
type part = { px : facet; py : facet }

type panel = {
  pnid : id;
  pfx : int option;
  pfy : int option;
  reads : (id * int option array) list;
}

type cell = {
  content : content;
  fx : int option;
  fy : int option;
  panels : panel list;
  parts : (id * part) list;
}

(* [reads_in scales pid pnid occs] is, per mark of [occs] in the cell [pid], the
   index in [scales] of the scale each binding reads in the panel [pnid]. A
   scale independent per panel is read in its panel only. *)
let reads_in scales pid pnid occs =
  let reads =
    List.map
      (fun o -> (o.mid, Array.make (List.length o.mark.bindings) None))
      occs
  in
  List.iteri
    (fun i (F f) ->
      let here =
        match f.key with
        | Panel (_, p) -> Nx.Ptree.Path.equal p pnid
        | Figure | Node _ | Cell _ | Panels_of _ -> true
      in
      if here then
        List.iter
          (fun m ->
            if Nx.Ptree.Path.equal m.m_pid pid then
              match find_path m.m_occ.mid reads with
              | Some a when Option.is_none a.(m.m_index) ->
                  a.(m.m_index) <- Some i
              | Some _ | None -> ())
          f.members)
    scales;
  reads

let shown c reads on =
  let read o =
    match find_path o.mid reads with
    | None -> None
    | Some reads ->
        List.find_map Fun.id
          (List.mapi
             (fun i (B b) ->
               if Role.shown_on b.role.use = Some on then reads.(i) else None)
             o.mark.bindings)
  in
  List.find_map read c.occs

(* [facet_index scales c pid on] is the index in [scales] of the scale the facet
   [on] reads in the cell [pid] of content [c]. A facet scale is never
   independent per panel, so every panel of the cell reads it. *)
let facet_index scales c pid on = shown c (reads_in scales pid pid c.occs) on

let category_names (s : string Scale.t) =
  match Scale.domain s with
  | Scale.Categories (Scale.Labels l) -> Array.to_list l
  | Scale.Categories (Scale.Indices ix) ->
      List.map (fun (i, _) -> string_of_int i) (Array.to_list ix)

(* [panels_of pid fx fy] is the panels of the cell [pid] whose facets read [fx]
   and [fy], by [fy] category then [fx] category. *)
let panels_of pid fx fy =
  (match (fx, fy) with
  | Some s, Some _ when Option.is_some (Scale.wrap s) ->
      err "resolve" "the fx scale of %a wraps, but the cell has an fy scale"
        pp_id pid
  | _ -> ());
  let cats = function
    | None -> [ (None, None) ]
    | Some s -> List.mapi (fun k c -> (Some k, Some c)) (category_names s)
  in
  let add c id =
    match c with None -> id | Some c -> Nx.Ptree.Path.add (Field c) id
  in
  match (fx, fy) with
  | None, None -> [ { pnid = pid; pfx = None; pfy = None; reads = [] } ]
  | _ ->
      List.concat_map
        (fun (pfy, cy) ->
          List.map
            (fun (pfx, cx) ->
              let pnid =
                Nx.Ptree.Path.add (Field "panel") pid |> add cy |> add cx
              in
              { pnid; pfx; pfy; reads = [] })
            (cats fx))
        (cats fy)

(* [facet_of o role s] is where the rows of [o] go along the facet [role] whose
   scale in the cell is [s], with the warning of a constant that names no
   category of [s]. *)
let facet_of o (role : (string, string) Role.t) s =
  match find_binding role o.mark.bindings with
  | None -> (Every, [])
  | Some (B b) -> (
      match
        (constant b.ch, data b.ch, Role.equal_range b.role.range Role.Panels)
      with
      | Some v, _, Some Type.Equal -> (
          let names = Option.fold ~none:[] ~some:category_names s in
          match List.find_index (String.equal v) names with
          | Some k -> (One (Some k), [])
          | None ->
              ( One None,
                [
                  ( o.mid,
                    Format.asprintf "the facet constant %S of %s names no panel"
                      v role.name );
                ] ))
      | None, Some d, _ -> (
          match (kind d.lift, s) with
          | Categories, Some s ->
              let lift = Lift.eval o.mark.shape ~role:role.name d.lift s in
              (Each (Lift.positions s lift), [])
          | Quantities, _ | Categories, None -> (Every, []))
      | _ -> (Every, []))

let mask part p =
  let along f k =
    match (f, k) with
    | Every, _ -> `All
    | One (Some c), Some k when c = k -> `All
    | Each r, Some k -> `Mask (Nx.equal_s r (Int64.of_int k))
    | One _, _ | Each _, None -> `None
  in
  match (along part.px p.pfx, along part.py p.pfy) with
  | `None, _ | _, `None -> `None
  | `All, m | m, `All -> m
  | `Mask m, `Mask m' -> `Mask (Nx.logical_and m m')

(* Resolved figures *)

type t = {
  figure : Figure.t;
  view : View.t;
  shaped : shaped;
  cells : (id * cell) list;
  scales : fitted list; (* In the order of their first readers. *)
  nodes : (id * (id * shares) list) list;
      (* Each node with the cells it lies in and the scopes it reads there. *)
  warnings : warning list;
}

(* [inputs_of scales reads occ] is what summarising reads of [occ], whose
   binding [i] reads the scale [reads.(i)] of [scales]. *)
let inputs_of scales reads occ =
  let input index (B b) =
    match data b.ch with
    | None -> None
    | Some d ->
        let keeps_row =
          match b.role.range with Role.Colors -> true | _ -> false
        in
        let kind = kind d.lift in
        let spec, fitted =
          match reads.(index) with
          | None -> (default_spec kind, false)
          | Some i -> (
              let (F f) = scales.(i) in
              match equal_kind f.kind kind with
              | Some Type.Equal -> (f.spec, true)
              | None ->
                  assert false (* A reading's scale has the reading's kind. *))
        in
        let lift = Lift.eval occ.mark.shape ~role:b.role.name d.lift spec in
        Some (Summary.In { index; keeps_row; fitted; lift })
  in
  List.filter_map Fun.id (List.mapi input occ.mark.bindings)

type lookup =
  | No_node
  | No_scope (* No one scope of the name holds the node. *)
  | Found of fitted option

(* [scope_of places name] is the scope of [name] that holds a node lying in
   [places], if one scope holds it in every cell. *)
let scope_of places name =
  let key (pid, shares) = key_of { shares; pending = []; cell = pid } name in
  match List.map key places with
  | k :: ks when List.for_all (equal_key k) ks -> Some k
  | _ -> None

(* [find_scale scales nodes ~at name kind] is the scale [name] of [kind] in the
   scope holding [at]. A scale that a mark makes independent per panel is held
   by the mark, and by the facet panel it is in: [Panel (mid, p)] is in a facet
   panel iff [p] is not the cell its readers are in. *)
let find_scale scales nodes ~at name kind =
  let is (F f) =
    String.equal f.name name && Option.is_some (equal_kind f.kind kind)
  in
  let held (F f as s) =
    is s
    &&
    match f.key with
    | Panel (mid, p) ->
        Nx.Ptree.Path.equal mid at
        || Nx.Ptree.Path.equal p at
           && not
                (List.exists (fun m -> Nx.Ptree.Path.equal m.m_pid p) f.members)
    | _ -> false
  in
  match List.filter held scales with
  | [ f ] -> Found (Some f)
  | _ :: _ :: _ -> No_scope
  | [] -> (
      match find_path at nodes with
      | None -> No_node
      | Some places -> (
          match scope_of places name with
          | None -> No_scope
          | Some key ->
              Found
                (List.find_opt
                   (fun (F f as s) -> is s && equal_key f.key key)
                   scales)))

type zoom = { at : id; name : string; key : key; ends : float * float }

(* [zoom view nodes scales] is [scales] with the zooms of [view] applied, and
   the warnings of the zooms it ignores. *)
let zoom view nodes scales =
  let warnings = ref [] in
  let warn at fmt =
    Format.kasprintf (fun s -> warnings := (at, s) :: !warnings) fmt
  in
  let target (ident, View.V (sort, v)) =
    match (ident, sort, v) with
    | View.Zoom_of { scale = name; at }, View.Zoom kind, Some ends -> (
        let none () =
          warn at "the zoom of the scale %S applies to no scale" name;
          None
        in
        (* No channel reads a time scale, and View.zoom refuses categories. *)
        match of_scale_kind kind with
        | Some Categories | None -> none ()
        | Some Quantities -> (
            match find_scale scales nodes ~at name Quantities with
            | Found (Some (F f)) -> Some { at; name; key = f.key; ends }
            | No_scope ->
                warn at "no scope of the scale %S holds the zoom's node" name;
                None
            | No_node | Found None -> none ()))
    | _ -> None
  in
  let zooms = List.filter_map target view in
  let apply (F f) =
    let of_f z = String.equal z.name f.name && equal_key z.key f.key in
    match (f.kind, List.filter of_f zooms) with
    | Categories, _ | Quantities, [] -> F f
    | Quantities, [ z ] -> (
        let lo, hi = z.ends in
        match Scale.with_domain (Scale.Floats (lo, hi)) f.scale with
        | scale -> F { f with scale }
        | exception Invalid_argument _ ->
            warn z.at "the zoom of the scale %S sets a domain it cannot take"
              z.name;
            F f)
    | Quantities, zs ->
        List.iter
          (fun z ->
            warn z.at
              "the scale %S is zoomed several times; its zooms are ignored"
              z.name)
          zs;
        F f
  in
  let scales = List.map apply scales in
  (scales, List.rev !warnings)

let unread view reads =
  List.filter_map
    (fun (ident, View.V (sort, _)) ->
      match ident with
      | View.User name ->
          let read (Read (i, s)) =
            View.equal_ident i ident && Option.is_some (View.equal_sort s sort)
          in
          if List.exists read reads then None
          else
            Some
              ( Nx.Ptree.Path.root,
                Format.asprintf
                  "the view sets %S, which no key of its sort reads" name )
      | View.Zoom_of _ -> None)
    view

let check_legends shaped scales =
  let legends = legends shaped in
  List.iter
    (fun (id, (l : guide), key) ->
      let stands (F f) =
        String.equal f.name l.scale && equal_key f.key key && f.legend
      in
      if not (List.exists stands scales) then
        err "resolve"
          "the legend %a names %S, no scale with a legend in its scope" pp_id id
          l.scale;
      List.iter
        (fun (id', (l' : guide), key') ->
          if
            String.equal l.scale l'.scale
            && equal_key key key'
            && not (equal_guide l l')
          then
            err "resolve" "%a and %a are two different legends for %S" pp_id id
              pp_id id' l.scale)
        legends)
    legends

let dedupe ws =
  let seen (id, s) =
    List.exists (fun (id', s') ->
        Nx.Ptree.Path.equal id id' && String.equal s s')
  in
  List.rev
    (List.fold_left (fun acc w -> if seen w acc then acc else w :: acc) [] ws)

(* [reading mid f] is [f ()], which reads the tensors of the mark [mid], with
   the errors of reading them naming the mark. *)
let reading mid f =
  try f () with Invalid_argument msg -> err "resolve" "%a: %s" pp_id mid msg

(* [in_order order ws] is [ws] sorted by the position in [order] of the node
   each is about, a generated node's being that of the innermost node of [order]
   it lies under. *)
let in_order order ws =
  let position id =
    let rec go i = function
      | [] -> None
      | id' :: rest ->
          if Nx.Ptree.Path.equal id id' then Some i else go (i + 1) rest
    in
    go 0 order
  in
  (* [rank rsegs] is the rank of the id of the reversed segments [rsegs]. *)
  let rec rank = function
    | [] -> 0
    | _ :: outer as rsegs -> (
        match position (Nx.Ptree.Path.v (List.rev rsegs)) with
        | Some i -> i
        | None -> rank outer)
  in
  let key (id, _) = rank (List.rev (Nx.Ptree.Path.segments id)) in
  List.stable_sort (fun w w' -> Int.compare (key w) (key w')) ws

let afresh view figure =
  let ({ shaped; order; reads } : Arrange.t) = arrange view figure in
  let cells = panels shaped in
  check_coords cells;
  let occs =
    List.concat_map (fun (pid, c) -> List.map (fun o -> (pid, o)) c.occs) cells
  in
  let readings = List.concat_map (fun (pid, o) -> readings_of pid o) occs in
  check_panel_scales readings;
  check_axes cells readings;
  let unfitted =
    List.map
      (fun (G g) ->
        let spec = merged g.kind g.name g.members in
        let guide = merged_guide g.name g.members in
        F
          {
            name = g.name;
            key = g.key;
            kind = g.kind;
            members = g.members;
            legend = g.legend;
            guide;
            spec;
            scale = spec;
          })
      (group readings)
  in
  let groups = Array.of_list unfitted in
  let summary occ pid mask =
    let reads = snd (List.hd (reads_in unfitted pid pid [ occ ])) in
    reading occ.mid (fun () ->
        Summary.summarise occ.mark.shape (inputs_of groups reads occ) mask)
  in
  let base =
    List.map (fun (pid, o) -> ((o.mid, pid), summary o pid None)) occs
  in
  let summary_of (occ : occ) pid =
    snd
      (List.find
         (fun ((mid, pid'), _) ->
           Nx.Ptree.Path.equal mid occ.mid && Nx.Ptree.Path.equal pid pid')
         base)
  in
  let notes =
    List.concat_map
      (fun ((mid, _), (s : Summary.t)) -> List.map (fun n -> (mid, n)) s.notes)
      base
  in
  let fit summary_of (F f) =
    F { f with scale = fit_scale f.kind f.name f.members f.spec summary_of }
  in
  let per_panel (F f) = match f.key with Panels_of _ -> true | _ -> false in
  (* Categorical scales first, since facets make panels. *)
  let fitted =
    List.map
      (fun (F f as s) ->
        match f.kind with
        | Categories when not (per_panel s) -> fit summary_of s
        | _ -> s)
      unfitted
  in
  let parted =
    List.map
      (fun (pid, c) ->
        let scale on =
          Option.bind (facet_index fitted c pid on)
            (fun i : string Scale.t option ->
              let (F f) = List.nth fitted i in
              match f.kind with
              | Categories -> Some f.scale
              | Quantities -> None)
        in
        let fx = scale (`Header Role.X) and fy = scale (`Header Role.Y) in
        let parts =
          List.map
            (fun o ->
              reading o.mid (fun () ->
                  let px, wx = facet_of o Role.fx fx
                  and py, wy = facet_of o Role.fy fy in
                  ((o.mid, { px; py }), wx @ wy)))
            c.occs
        in
        ( (pid, (c, panels_of pid fx fy, List.map fst parts)),
          List.concat_map snd parts ))
      cells
  in
  let constants = List.concat_map snd parted and parted = List.map fst parted in
  let panel_scales (F f as s) =
    match f.key with
    | Panels_of (mid, pid) ->
        let occ = (List.hd f.members).m_occ in
        let _, panels, parts = Option.get (find_path pid parted) in
        let part = Option.get (find_path mid parts) in
        List.filter_map
          (fun p ->
            let fit_in mask =
              let s = lazy (summary occ pid mask) in
              let summary_of _ _ = Lazy.force s in
              Some (fit summary_of (F { f with key = Panel (mid, p.pnid) }))
            in
            match mask part p with
            | `None -> None
            | `All -> fit_in None
            | `Mask m -> fit_in (Some m))
          panels
    | _ -> [ s ]
  in
  let fitted = List.concat_map panel_scales fitted in
  let fitted =
    List.map
      (fun (F f as s) ->
        match (f.kind, f.key) with
        | Categories, _ | _, Panel _ -> s
        | _ -> fit summary_of s)
      fitted
  in
  (* Each node with the cells it lies in; a facet panel lies in its cell and
     reads the scopes of the cell. *)
  let nodes =
    let held =
      List.concat_map
        (fun (pid, c) -> List.map (fun (id, sh) -> (id, (pid, sh))) c.held)
        cells
    in
    let panels =
      List.concat_map
        (fun (pid, (c, panels, _)) ->
          let shares = Option.value ~default:[] (find_path pid c.held) in
          List.filter_map
            (fun p ->
              if Nx.Ptree.Path.equal p.pnid pid then None
              else Some (p.pnid, (pid, shares)))
            panels)
        parted
    in
    let places = held @ panels in
    let ids =
      List.fold_left
        (fun acc id ->
          if List.exists (Nx.Ptree.Path.equal id) acc then acc else id :: acc)
        []
        (List.map fst places @ order)
    in
    List.rev_map
      (fun id ->
        ( id,
          List.filter_map
            (fun (id', p) ->
              if Nx.Ptree.Path.equal id id' then Some p else None)
            places ))
      ids
  in
  let scales, zooms = zoom view nodes fitted in
  check_legends shaped scales;
  let warnings =
    dedupe (in_order order (notes @ constants @ zooms) @ unread view reads)
  in
  let cells =
    List.map
      (fun (pid, (content, panels, parts)) ->
        let read p =
          { p with reads = reads_in scales pid p.pnid content.occs }
        in
        let facet on = facet_index scales content pid on in
        let fx = facet (`Header Role.X) and fy = facet (`Header Role.Y) in
        (pid, { content; fx; fy; panels = List.map read panels; parts }))
      parted
  in
  { figure; view; shaped; cells; scales; nodes; warnings }

(* A figure equal to the one [prev] resolved, under an equal view, resolves to
   [prev]. *)
let resolve ?prev ?(view = View.empty) figure =
  match prev with
  | Some r when Figure.equal r.figure figure && View.equal r.view view -> r
  | _ -> afresh view figure

(* Observing and comparing *)

let scale : type d. ?at:id -> t -> d Scale.t -> d Scale.t =
 fun ?(at = Nx.Ptree.Path.root) r s ->
  let name =
    match Scale.name s with
    | Some n -> n
    | None -> err "Resolved.scale" "the scale is unnamed"
  in
  match of_scale_kind (Scale.kind s) with
  | None ->
      err "Resolved.scale" "the scope of %a has no temporal scale %S" pp_id at
        name
  | Some kind -> (
      let absent () =
        err "Resolved.scale" "the scope of %a has no %a scale %S" pp_id at
          pp_kind kind name
      in
      match find_scale r.scales r.nodes ~at name kind with
      | No_node -> err "Resolved.scale" "no node has the id %a" pp_id at
      | No_scope ->
          err "Resolved.scale" "no scope of the scale %S holds %a" name pp_id at
      | Found None -> absent ()
      | Found (Some (F f)) -> (
          match equal_kind f.kind kind with
          | Some Type.Equal -> f.scale
          | None -> absent ()))

let warnings r = r.warnings
let panel_of = function Panel (_, p) -> Some p | _ -> None

let equal_member m m' =
  Nx.Ptree.Path.equal m.m_occ.mid m'.m_occ.mid
  && String.equal m.m_role m'.m_role
  && Nx.Ptree.Path.equal m.m_pid m'.m_pid

let equal_scale (F f) (F f') =
  String.equal f.name f'.name
  && Option.equal Nx.Ptree.Path.equal (panel_of f.key) (panel_of f'.key)
  && Option.equal Bool.equal f.guide f'.guide
  &&
  match equal_kind f.kind f'.kind with
  | Some Type.Equal ->
      List.equal equal_member f.members f'.members
      && Scale.equal f.scale f'.scale
  | None -> false

let equal_warning (id, s) (id', s') =
  Nx.Ptree.Path.equal id id' && String.equal s s'

let equal r r' =
  Figure.equal r.figure r'.figure
  && View.equal r.view r'.view
  && List.equal equal_scale r.scales r'.scales
  && List.equal equal_warning r.warnings r'.warnings

(* Formatting *)

let pp_title ppf ((align : Text.Layout.halign), t) =
  Format.fprintf ppf "title %a%s" Text.pp t
    (match align with `Center -> "" | `Left -> " left" | `Right -> " right")

let pp_weights name ppf = function
  | None -> ()
  | Some ws ->
      Format.fprintf ppf ", %s %a" name
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " ")
           (fun ppf w -> Format.fprintf ppf "%g" w))
        ws

let pp_ids =
  Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ") pp_id

let rec pp_shaped cells ppf (pid, s) =
  match s.body with
  | Single c ->
      Format.fprintf ppf "@[<v 2>panel %a" pp_id pid;
      List.iter (Format.fprintf ppf "@,%a" pp_title) s.titles;
      (match c.coords with
      | (_, k) :: _ -> Format.fprintf ppf "@,coord %a" Coord.pp k
      | [] -> ());
      List.iter
        (fun o -> Format.fprintf ppf "@,%s %a" o.mark.kind pp_id o.mid)
        c.occs;
      List.iter (fun (_, g, _) -> Format.fprintf ppf "@,%a" pp_guide g) c.guides;
      (match Option.map (fun c -> c.panels) (find_path pid cells) with
      | Some [ p ] when Nx.Ptree.Path.equal p.pnid pid -> ()
      | Some ps ->
          Format.fprintf ppf "@,@[<hov 2>facets %a@]" pp_ids
            (List.map (fun p -> p.pnid) ps)
      | None -> ());
      Format.fprintf ppf "@]"
  | Arr a ->
      Format.fprintf ppf "@[<v 2>grid %a, %d × %d%a%a" pp_id a.aid a.nrows
        a.ncols (pp_weights "widths") a.widths (pp_weights "heights") a.heights;
      List.iter (Format.fprintf ppf "@,%a" pp_title) s.titles;
      List.iter
        (fun cell ->
          Format.fprintf ppf "@,@[<v 2>cell (%d, %d)%s@,%a@]" cell.row cell.col
            (if cell.rows = 1 && cell.cols = 1 then ""
             else Format.asprintf ", spanning %d × %d" cell.rows cell.cols)
            (pp_shaped cells) (cell.cid, cell.s))
        a.cells;
      Format.fprintf ppf "@]"

let pp_scale ppf (F f) =
  let readers =
    List.fold_left
      (fun acc m ->
        let r = Format.asprintf "%a:%s" pp_id m.m_occ.mid m.m_role in
        if List.mem r acc then acc else r :: acc)
      [] f.members
    |> List.rev
  in
  Format.fprintf ppf "@[<v 2>%S %a%a, read by @[<hov>%a@]@,%a%a@]" f.name
    pp_kind f.kind
    (Format.pp_print_option (fun ppf p -> Format.fprintf ppf " in %a" pp_id p))
    (panel_of f.key)
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ")
       Format.pp_print_string)
    readers Scale.pp f.scale
    (Format.pp_print_option (fun ppf g -> Format.fprintf ppf "@,guide %b" g))
    f.guide

let pp ppf r =
  Format.fprintf ppf "@[<v>@[<v 2>figure@,%a@]" (pp_shaped r.cells)
    (Nx.Ptree.Path.root, r.shaped);
  Format.fprintf ppf "@,@[<v 2>scales";
  List.iter (Format.fprintf ppf "@,%a" pp_scale) r.scales;
  Format.fprintf ppf "@]";
  if r.warnings <> [] then (
    Format.fprintf ppf "@,@[<v 2>warnings";
    List.iter (Format.fprintf ppf "@,%a" pp_warning) r.warnings;
    Format.fprintf ppf "@]");
  Format.fprintf ppf "@]"
