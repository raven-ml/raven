(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P2 = Hugin_next_gg.P2
module Box2 = Hugin_next_gg.Box2
module Path = Hugin_next_gg.Path
module Color = Hugin_next_gg.Color
module Text = Hugin_next_text.Text
module Picture = Hugin_next_vg.Picture

(* Columns *)

type col =
  | Col : {
      name : string;
      range : 'r Role.range;
      values : 'r array;
      norm : float array option;
      fn : (float -> 'r) option;
      ticks : float array option;
      cats : int array option;
      band : float option;
      zero : float option;
    }
      -> col

(* Rows *)

type t = {
  id : Common.id;
  shape : int array;
  index : int array;
  theme : Theme.t;
  projection : Coord.projection;
  density : float;
  cols : col list;
  dropped : bool array;
  warn : string -> unit;
}

let pick a ks = Array.map (fun k -> a.(k)) ks

let select_col ks (Col c) =
  Col
    {
      c with
      values = pick c.values ks;
      norm = Option.map (fun a -> pick a ks) c.norm;
      cats = Option.map (fun a -> pick a ks) c.cats;
    }

let select r ks =
  {
    r with
    index = pick r.index ks;
    cols = List.map (select_col ks) r.cols;
    dropped = pick r.dropped ks;
  }

(* Observing *)

let length r = Array.length r.index
let find r name = List.find_opt (fun (Col c) -> String.equal c.name name) r.cols

let get : type d v. t -> (d, v) Role.t -> v array option =
 fun r role ->
  match find r role.name with
  | None -> None
  | Some (Col c) -> (
      match Role.equal_range c.range role.range with
      | Some Type.Equal -> Some (Array.copy c.values)
      | None -> None)

let normalized r (role : _ Role.t) =
  Option.bind (find r role.name) (fun (Col c) -> Option.map Array.copy c.norm)

let range : type d v. t -> (d, v) Role.t -> (float -> v) option =
 fun r role ->
  match find r role.name with
  | None -> None
  | Some (Col c) -> (
      match Role.equal_range c.range role.range with
      | Some Type.Equal -> c.fn
      | None -> None)

let ticks r (role : _ Role.t) =
  Option.bind (find r role.name) (fun (Col c) -> Option.map Array.copy c.ticks)

(* A position: the normalised value of each row, the bandwidth of its band scale
   or the normalised zero of its continuous one, and whether it reads a
   scale. *)
type position = {
  us : float array;
  band : float option;
  zero : float option;
  scaled : bool;
}

let position r name : position option =
  match find r name with
  | None -> None
  | Some (Col c) -> (
      match Role.equal_range c.range Role.Floats with
      | Some Type.Equal ->
          Some
            {
              us = c.values;
              band = c.band;
              zero = c.zero;
              scaled = Option.is_some c.norm;
            }
      | None -> None)

let positions r =
  let n = length r in
  let at name =
    match position r name with Some p -> p.us | None -> Array.make n 0.5
  in
  let us = at "x" and vs = at "y" in
  let xs = Array.make n Float.nan and ys = Array.make n Float.nan in
  for i = 0 to n - 1 do
    if not r.dropped.(i) then begin
      xs.(i) <- us.(i);
      ys.(i) <- vs.(i)
    end
  done;
  (xs, ys)

(* The domain is the unit square, its edges included. *)
let in_domain u = 0. <= u && u <= 1.

let points r =
  let us, vs = positions r in
  let n = length r in
  let xs = Array.make n Float.nan and ys = Array.make n Float.nan in
  for i = 0 to n - 1 do
    if in_domain us.(i) && in_domain vs.(i) then begin
      let p = Coord.point r.projection us.(i) vs.(i) in
      xs.(i) <- P2.x p;
      ys.(i) <- P2.y p
    end
  done;
  (xs, ys)

(* [ends p i] is the interval a position covers in its own right: its band on a
   band scale, and its value otherwise. *)
let ends p i =
  let u = p.us.(i) in
  match p.band with Some w -> (u -. (w /. 2.), u +. (w /. 2.)) | None -> (u, u)

let clamp u = Float.min 1. (Float.max 0. u)

let extent r axis =
  let n = length r in
  let name, name2 = match axis with `X -> ("x", "x2") | `Y -> ("y", "y2") in
  let lo = Array.make n Float.nan and hi = Array.make n Float.nan in
  let cover i =
    match (position r name, position r name2) with
    | None, _ -> (0., 1.)
    | Some p, Some p2 ->
        let a, b = ends p i and a', b' = ends p2 i in
        (Float.min a a', Float.max b b')
    | Some p, None -> (
        match (p.band, p.zero) with
        | Some _, _ -> ends p i
        | None, Some z when p.scaled -> (z, p.us.(i))
        | _ -> ends p i)
  in
  for i = 0 to n - 1 do
    if not r.dropped.(i) then begin
      let a, b = cover i in
      (* An extent wholly on one side of the domain covers none of it. *)
      if not ((a < 0. && b < 0.) || (a > 1. && b > 1.)) then begin
        lo.(i) <- clamp a;
        hi.(i) <- clamp b
      end
    end
  done;
  (lo, hi)

let unit_square = Box2.v 0. 0. 1. 1.

let project r p =
  Path.transform (Coord.affine r.projection) (Path.crop unit_square p)

(* Two rows are in one series iff they share their index along every axis but
   the last and their category in every band channel that is not a position. *)
let series r =
  let last =
    match Array.length r.shape with 0 -> 1 | k -> max 1 r.shape.(k - 1)
  in
  let cats =
    List.filter_map
      (fun (Col c) ->
        match c.name with "x" | "x2" | "y" | "y2" -> None | _ -> c.cats)
      r.cols
  in
  let groups = Hashtbl.create 16 and order = ref [] in
  Array.iteri
    (fun i datum ->
      let key = (datum / last, List.map (fun a -> a.(i)) cats) in
      match Hashtbl.find_opt groups key with
      | Some l -> l := i :: !l
      | None ->
          let l = ref [ i ] in
          Hashtbl.add groups key l;
          order := l :: !order)
    r.index;
  List.rev_map (fun l -> select r (Array.of_list (List.rev !l))) !order

let text ?(halign = `Center) ?(valign = `Middle) r c at s =
  let l =
    Text.Layout.v ~halign ~valign ~fonts:(Theme.fonts r.theme)
      ~size:(Theme.size r.theme) s
  in
  (match Text.Layout.missing l with
  | [] -> ()
  | us ->
      r.warn
        (Format.asprintf "the text %a has %s, which no face of the theme has"
           Text.pp s
           (String.concat ", "
              (List.map (fun u -> Printf.sprintf "U+%04X" (Uchar.to_int u)) us))));
  let run acc colour o run =
    let colour = Option.value colour ~default:c in
    Picture.glyphs colour (P2.v (P2.x at +. P2.x o) (P2.y at +. P2.y o)) run
    :: acc
  in
  Picture.group (List.rev (Text.Layout.fold run [] l))
