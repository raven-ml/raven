(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_gg
open Hugin_font

type rule = [ `Nonzero | `Even_odd ]

type rows =
  | Rows of int array
  | Cells of { box : Box2.t; width : int; height : int }

type tag = { id : Nx.Ptree.Path.t; rows : rows }

type t =
  | Empty
  | Fill of { rule : rule; color : Color.t; path : Path.t }
  | Stroke of { stroke : Stroke.t; color : Color.t; path : Path.t }
  | Glyphs of { color : Color.t; at : P2.t; run : Run.t }
  | Image of { box : Box2.t; pixels : Nx.uint8_t }
  | Group of t list
  | Clip of { rule : rule; path : Path.t; picture : t }
  | Transform of { m : Affine.t; picture : t }
  | Opacity of { opacity : float; picture : t }
  | Stamp of {
      picture : t;
      xs : float array;
      ys : float array;
      scales : float array option;
      fills : Color.t array option;
      strokes : Color.t array option;
    }
  | Tag of { tag : tag; picture : t }

let err fn fmt = Printf.ksprintf invalid_arg ("Picture." ^^ fn ^^ ": " ^^ fmt)

(* Leaves *)

let empty = Empty

let fill ?(rule = `Nonzero) color path =
  if Path.is_empty path then Empty else Fill { rule; color; path }

let stroke stroke color path =
  if Path.is_empty path || Stroke.width stroke = 0. then Empty
  else Stroke { stroke; color; path }

let glyphs color at run =
  if
    Run.length run = 0
    || Run.size run = 0.
    || not (Float.is_finite (P2.x at) && Float.is_finite (P2.y at))
  then Empty
  else Glyphs { color; at; run }

let image box pixels =
  match Nx.shape pixels with
  | [| h; w; 1 | 3 | 4 |] ->
      if h = 0 || w = 0 || Box2.w box = 0. || Box2.h box = 0. then Empty
      else Image { box; pixels }
  | shape ->
      err "image" "shape [|%s|] is not [|h; w; c|] with c in 1, 3 or 4"
        (String.concat "; " (Array.to_list (Array.map string_of_int shape)))

(* Composing *)

let group ps =
  match List.filter (function Empty -> false | _ -> true) ps with
  | [] -> Empty
  | [ p ] -> p
  | ps -> Group ps

let clip ?(rule = `Nonzero) path picture =
  match picture with
  | Empty -> Empty
  | _ when Path.is_empty path -> Empty
  | _ -> Clip { rule; path; picture }

let transform m picture =
  match (picture, Affine.invert m) with
  | Empty, _ | _, None -> Empty
  | _, Some _ -> Transform { m; picture }

let opacity a picture =
  if not (0. <= a && a <= 1.) then err "opacity" "opacity %g not in [0, 1]" a;
  match picture with
  | Empty -> Empty
  | _ when a = 1. -> picture
  | _ -> Opacity { opacity = a; picture }

let copy_attribute name n = function
  | None -> None
  | Some a ->
      if Array.length a <> n then
        err "stamp" "%d %s for %d positions" (Array.length a) name n;
      Some (Array.copy a)

let stamp ?fills ?strokes ?scales xs ys picture =
  let n = Array.length xs in
  if Array.length ys <> n then err "stamp" "%d xs but %d ys" n (Array.length ys);
  let fills = copy_attribute "fills" n fills in
  let strokes = copy_attribute "strokes" n strokes in
  let scales = copy_attribute "scales" n scales in
  Option.iter
    (Array.iter (fun s ->
         if s < 0. && Float.is_finite s then err "stamp" "negative scale %g" s))
    scales;
  match picture with
  | Empty -> Empty
  | _ when n = 0 -> Empty
  | _ ->
      Stamp
        {
          picture;
          xs = Array.copy xs;
          ys = Array.copy ys;
          scales;
          fills;
          strokes;
        }

(* Tags *)

let tag t picture =
  let rows =
    match t.rows with
    | Rows a -> (
        match picture with
        | Stamp { xs; _ } when Array.length a <> Array.length xs ->
            err "tag" "%d rows for %d stamp positions" (Array.length a)
              (Array.length xs)
        | _ -> Rows (Array.copy a))
    | Cells { width; height; _ } as cells ->
        if width < 1 || height < 1 then
          err "tag" "grid of %d by %d cells" width height;
        cells
  in
  match picture with
  | Empty -> Empty
  | _ -> Tag { tag = { t with rows }; picture }

(* Bounds *)

(* [acc] is [minx; miny; maxx; maxy], empty while [minx > maxx]. *)
let union acc b =
  acc.(0) <- Float.min acc.(0) (Box2.minx b);
  acc.(1) <- Float.min acc.(1) (Box2.miny b);
  acc.(2) <- Float.max acc.(2) (Box2.maxx b);
  acc.(3) <- Float.max acc.(3) (Box2.maxy b)

let fresh () = [| infinity; infinity; neg_infinity; neg_infinity |]

let to_box acc =
  if acc.(0) <= acc.(2) then
    Some (Box2.of_pts (P2.v acc.(0) acc.(1)) (P2.v acc.(2) acc.(3)))
  else None

let shift b dx dy =
  Box2.of_pts
    (P2.v (Box2.minx b +. dx) (Box2.miny b +. dy))
    (P2.v (Box2.maxx b +. dx) (Box2.maxy b +. dy))

(* The clips above a leaf: none, the box that cuts it, or boxes a stamp will cut
   each instance of it by, once it has moved them. *)
type cut = Uncut | Cut of Box2.t | Cut_later

let cut add clip b =
  match clip with
  | Uncut | Cut_later -> add b
  | Cut c -> Option.iter add (Box2.inter b c)

(* [bounds_into add m clip pen p] gives to [add] the boxes of the leaves of [p]
   under [m], cut by [clip], with the reach of pens multiplied by [pen]. *)
let rec bounds_into add m clip pen = function
  | Empty -> ()
  | Fill { path; _ } -> leaf add m clip (Path.bounds path)
  | Stroke { stroke; path; _ } ->
      leaf add m clip
        (Option.map (Box2.grow (pen *. Stroke.reach stroke)) (Path.bounds path))
  | Glyphs { at; run; _ } ->
      let at = Affine.translate (P2.x at) (P2.y at) in
      leaf add Affine.(m * at) clip (Run.bounds run)
  | Image { box; _ } -> leaf add m clip (Some box)
  | Group ps -> List.iter (bounds_into add m clip pen) ps
  | Clip { path; picture; _ } -> (
      match Path.bounds path with
      | None -> ()
      | Some b -> (
          let b = Box2.transform m b in
          match clip with
          | Uncut | Cut_later -> bounds_into add m (Cut b) pen picture
          | Cut c -> (
              match Box2.inter b c with
              | None -> ()
              | Some c -> bounds_into add m (Cut c) pen picture)))
  | Transform { m = m'; picture } ->
      bounds_into add Affine.(m * m') clip pen picture
  | Opacity { picture; _ } | Tag { picture; _ } ->
      bounds_into add m clip pen picture
  | Stamp { picture; xs; ys; scales = None; _ } -> (
      (* Instances differ by a translation, which commutes with taking boxes:
         the boxes of [picture] are taken once and moved to each instance.
         Uncut, their union moves as well as they do. *)
      let each_instance f =
        for i = 0 to Array.length xs - 1 do
          let x = xs.(i) and y = ys.(i) in
          if Float.is_finite x && Float.is_finite y then
            f
              ((m.xx *. x) +. (m.xy *. y) +. m.x0)
              ((m.yx *. x) +. (m.yy *. y) +. m.y0)
        done
      in
      match clip with
      | Uncut -> (
          let t = fresh () in
          bounds_into (union t) (Affine.linear m) Uncut pen picture;
          match to_box t with
          | None -> ()
          | Some b -> each_instance (fun dx dy -> add (shift b dx dy)))
      | Cut _ | Cut_later ->
          let boxes = ref [] in
          bounds_into
            (fun b -> boxes := b :: !boxes)
            (Affine.linear m) Cut_later pen picture;
          each_instance (fun dx dy ->
              List.iter (fun b -> cut add clip (shift b dx dy)) !boxes))
  | Stamp { picture; xs; ys; scales = Some scales; _ } ->
      for i = 0 to Array.length xs - 1 do
        let x = xs.(i) and y = ys.(i) and s = scales.(i) in
        if Float.is_finite x && Float.is_finite y && Float.is_finite s && s > 0.
        then
          let m = Affine.(m * translate x y * scale s s) in
          bounds_into add m clip (pen /. s) picture
      done

and leaf add m clip = function
  | None -> ()
  | Some b -> cut add clip (Box2.transform m b)

let bounds p =
  let acc = fresh () in
  bounds_into (union acc) Affine.id Uncut 1. p;
  to_box acc

(* Comparing *)

let rule_equal (r : rule) (r' : rule) =
  match (r, r') with
  | `Nonzero, `Nonzero | `Even_odd, `Even_odd -> true
  | (`Nonzero | `Even_odd), _ -> false

let rows_equal r r' =
  match (r, r') with
  | Rows a, Rows a' -> Array.equal Int.equal a a'
  | Cells c, Cells c' ->
      Box2.equal c.box c'.box && c.width = c'.width && c.height = c'.height
  | (Rows _ | Cells _), _ -> false

let tag_equal t t' = Nx.Ptree.Path.equal t.id t'.id && rows_equal t.rows t'.rows

let rec equal p p' =
  p == p'
  ||
  match (p, p') with
  | Empty, Empty -> true
  | Fill f, Fill f' ->
      rule_equal f.rule f'.rule
      && Color.equal f.color f'.color
      && Path.equal f.path f'.path
  | Stroke s, Stroke s' ->
      Stroke.equal s.stroke s'.stroke
      && Color.equal s.color s'.color
      && Path.equal s.path s'.path
  | Glyphs g, Glyphs g' ->
      Color.equal g.color g'.color
      && P2.equal g.at g'.at && Run.equal g.run g'.run
  | Image i, Image i' ->
      Box2.equal i.box i'.box
      && (i.pixels == i'.pixels
         || Nx.item [] (Nx.array_equal i.pixels i'.pixels))
  | Group ps, Group ps' -> List.equal equal ps ps'
  | Clip c, Clip c' ->
      rule_equal c.rule c'.rule && Path.equal c.path c'.path
      && equal c.picture c'.picture
  | Transform t, Transform t' ->
      Affine.equal t.m t'.m && equal t.picture t'.picture
  | Opacity o, Opacity o' ->
      Float.equal o.opacity o'.opacity && equal o.picture o'.picture
  | Stamp s, Stamp s' ->
      Array.equal Float.equal s.xs s'.xs
      && Array.equal Float.equal s.ys s'.ys
      && Option.equal (Array.equal Float.equal) s.scales s'.scales
      && Option.equal (Array.equal Color.equal) s.fills s'.fills
      && Option.equal (Array.equal Color.equal) s.strokes s'.strokes
      && equal s.picture s'.picture
  | Tag t, Tag t' -> tag_equal t.tag t'.tag && equal t.picture t'.picture
  | ( ( Empty | Fill _ | Stroke _ | Glyphs _ | Image _ | Group _ | Clip _
      | Transform _ | Opacity _ | Stamp _ | Tag _ ),
      _ ) ->
      false

(* Formatting *)

let rule_name = function `Nonzero -> "nonzero" | `Even_odd -> "even-odd"

let pp_array pp_elt ppf a =
  Format.fprintf ppf "@[<1>(%a)@]"
    (Format.pp_print_array ~pp_sep:Format.pp_print_space pp_elt)
    a

let pp_float ppf x = Format.fprintf ppf "%g" x

let pp_rows ppf = function
  | Rows a ->
      Format.fprintf ppf "@[<1>(rows@ %a)@]" (pp_array Format.pp_print_int) a
  | Cells { box; width; height } ->
      Format.fprintf ppf "@[<1>(cells@ %a@ %d@ %d)@]" Box2.pp box width height

let rec pp ppf = function
  | Empty -> Format.pp_print_string ppf "empty"
  | Fill { rule; color; path } ->
      Format.fprintf ppf "@[<1>(fill %s@ %a@ \"%a\")@]" (rule_name rule)
        Color.pp color Path.pp path
  | Stroke { stroke; color; path } ->
      Format.fprintf ppf "@[<1>(stroke@ %a@ %a@ \"%a\")@]" Stroke.pp stroke
        Color.pp color Path.pp path
  | Glyphs { color; at; run } ->
      Format.fprintf ppf "@[<1>(glyphs@ %a@ %a@ %a)@]" Color.pp color P2.pp at
        Run.pp run
  | Image { box; pixels } ->
      Format.fprintf ppf "@[<1>(image@ %a@ %a)@]" Box2.pp box
        (pp_array Format.pp_print_int)
        (Nx.shape pixels)
  | Group ps ->
      Format.fprintf ppf "@[<1>(group@ %a)@]"
        (Format.pp_print_list ~pp_sep:Format.pp_print_space pp)
        ps
  | Clip { rule; path; picture } ->
      Format.fprintf ppf "@[<1>(clip %s@ \"%a\"@ %a)@]" (rule_name rule) Path.pp
        path pp picture
  | Transform { m; picture } ->
      Format.fprintf ppf "@[<1>(transform@ %a@ %a)@]" Affine.pp m pp picture
  | Opacity { opacity; picture } ->
      Format.fprintf ppf "@[<1>(opacity %g@ %a)@]" opacity pp picture
  | Stamp { picture; xs; ys; scales; fills; strokes } ->
      let opt name pp_elt ppf = function
        | None -> ()
        | Some a ->
            Format.fprintf ppf "@ @[<1>(%s@ %a)@]" name (pp_array pp_elt) a
      in
      Format.fprintf ppf
        "@[<1>(stamp@ @[<1>(xs@ %a)@]@ @[<1>(ys@ %a)@]%a%a%a@ %a)@]"
        (pp_array pp_float) xs (pp_array pp_float) ys (opt "scales" pp_float)
        scales (opt "fills" Color.pp) fills (opt "strokes" Color.pp) strokes pp
        picture
  | Tag { tag; picture } ->
      Format.fprintf ppf "@[<1>(tag@ %a@ %a@ %a)@]" Nx.Ptree.Path.pp tag.id
        pp_rows tag.rows pp picture
