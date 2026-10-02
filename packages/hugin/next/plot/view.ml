(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Scale = Hugin_next_kit.Scale
module Time = Hugin_next_kit.Time
open Common

type _ sort =
  | Number : float sort
  | Choice : string sort
  | Interval : (float * float) option sort
  | Zoom : 'd Scale.kind -> ('d * 'd) option sort

(* A zoom key is named by its scale and node, so no user key names it. *)
type ident = User of string | Zoom_of of { scale : string; at : id }
type 'a key = { ident : ident; sort : 'a sort; init : 'a }

let compare_ident i i' =
  match (i, i') with
  | User n, User n' -> String.compare n n'
  | User _, Zoom_of _ -> -1
  | Zoom_of _, User _ -> 1
  | Zoom_of z, Zoom_of z' ->
      let c = String.compare z.scale z'.scale in
      if c <> 0 then c else compare_id z.at z'.at

let equal_ident i i' = compare_ident i i' = 0

let pp_ident ppf = function
  | User n -> Format.fprintf ppf "%S" n
  | Zoom_of { scale; at } -> Format.fprintf ppf "zoom %S at %a" scale pp_id at

let equal_sort : type a b. a sort -> b sort -> (a, b) Type.eq option =
 fun s s' ->
  match (s, s') with
  | Number, Number -> Some Type.Equal
  | Choice, Choice -> Some Type.Equal
  | Interval, Interval -> Some Type.Equal
  | Zoom k, Zoom k' -> (
      match Scale.equal_kind k k' with
      | Some Type.Equal -> Some Type.Equal
      | None -> None)
  | _ -> None

let equal_pair eq (a, b) (a', b') = eq a a' && eq b b'

let equal_value : type a. a sort -> a -> a -> bool =
 fun s v v' ->
  match s with
  | Number -> Float.equal v v'
  | Choice -> String.equal v v'
  | Interval -> Option.equal (equal_pair Float.equal) v v'
  | Zoom Scale.Quantitative -> Option.equal (equal_pair Float.equal) v v'
  | Zoom Scale.Temporal -> Option.equal (equal_pair Time.equal) v v'
  | Zoom Scale.Categorical -> Option.equal (equal_pair String.equal) v v'

let pp_value : type a. a sort -> Format.formatter -> a -> unit =
 fun s ppf v ->
  let pp_pair pp ppf (a, b) = Format.fprintf ppf "(%a, %a)" pp a pp b in
  let pp_opt pp ppf = function
    | None -> Format.pp_print_string ppf "none"
    | Some v -> pp ppf v
  in
  let pp_float ppf x = Format.fprintf ppf "%g" x in
  match s with
  | Number -> pp_float ppf v
  | Choice -> Format.fprintf ppf "%S" v
  | Interval -> pp_opt (pp_pair pp_float) ppf v
  | Zoom Scale.Quantitative -> pp_opt (pp_pair pp_float) ppf v
  | Zoom Scale.Temporal -> pp_opt (pp_pair Time.pp) ppf v
  | Zoom Scale.Categorical -> pp_opt (pp_pair Format.pp_print_string) ppf v

let equal_key : type a b. a key -> b key -> (a, b) Type.eq option =
 fun k k' ->
  if not (equal_ident k.ident k'.ident) then None
  else
    match equal_sort k.sort k'.sort with
    | Some Type.Equal as eq ->
        if equal_value k.sort k.init k'.init then eq else None
    | None -> None

let check_interval fn = function
  | None -> ()
  | Some (lo, hi) ->
      if not (Float.is_finite lo && Float.is_finite hi && lo <= hi) then
        err fn "the interval (%g, %g) is not finite or not ordered" lo hi

let number name ~init = { ident = User name; sort = Number; init }
let choice name ~init = { ident = User name; sort = Choice; init }

let interval name ~init =
  check_interval "View.interval" init;
  { ident = User name; sort = Interval; init }

let zoom : type d. ?at:id -> d Scale.t -> (d * d) option key =
 fun ?(at = Nx.Ptree.Path.root) s ->
  let kind = Scale.kind s in
  (match kind with
  | Scale.Categorical -> err "View.zoom" "the scale is categorical"
  | Scale.Quantitative | Scale.Temporal -> ());
  match Scale.name s with
  | None -> err "View.zoom" "the scale is unnamed"
  | Some scale ->
      { ident = Zoom_of { scale; at }; sort = Zoom kind; init = None }

type value = V : 'a sort * 'a -> value

(* Bindings in increasing order of their idents. *)
type t = (ident * value) list

let empty = []

let set : type a. a key -> a -> t -> t =
 fun k v view ->
  (match k.sort with Interval -> check_interval "View.set" v | _ -> ());
  let rec add = function
    | [] -> [ (k.ident, V (k.sort, v)) ]
    | ((i, _) as b) :: rest ->
        let c = compare_ident k.ident i in
        if c < 0 then (k.ident, V (k.sort, v)) :: b :: rest
        else if c = 0 then (k.ident, V (k.sort, v)) :: rest
        else b :: add rest
  in
  add view

let get : type a. a key -> t -> a =
 fun k view ->
  match List.find_opt (fun (i, _) -> equal_ident k.ident i) view with
  | None -> k.init
  | Some (_, V (s, v)) -> (
      match equal_sort k.sort s with Some Type.Equal -> v | None -> k.init)

let equal (view : t) (view' : t) =
  List.equal
    (fun (i, V (s, v)) (i', V (s', v')) ->
      equal_ident i i'
      &&
      match equal_sort s s' with
      | Some Type.Equal -> equal_value s v v'
      | None -> false)
    view view'

let pp ppf (view : t) =
  let pp_binding ppf (i, V (s, v)) =
    Format.fprintf ppf "@[%a = %a@]" pp_ident i (pp_value s) v
  in
  Format.fprintf ppf "@[<v>%a@]" (Format.pp_print_list pp_binding) view
