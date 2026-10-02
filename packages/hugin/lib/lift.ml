(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Scale = Hugin_kit.Scale
open Channel

type miss = {
  rows : Nx.bool_t option;
  counts : (Nx.bool_t Lazy.t * (int -> string)) list;
}

type _ t =
  | Quantities : { values : Nx.float64_t; miss : miss } -> float t
  | Categories : {
      lift : string lift;
      codes : Nx.int64_t;
      miss : miss;
    }
      -> string t

let miss : type d. d t -> miss = function
  | Quantities q -> q.miss
  | Categories c -> c.miss

let labelled : string lift -> bool = function
  | Cat { labels = Some _; _ } | Strings _ -> true
  | Cat { labels = None; _ } | Dim _ -> false

let label : string lift -> int -> string = function
  | Cat { labels = Some l; _ } | Dim { labels = Some l; _ } | Strings l ->
      fun k -> l.(k)
  | Cat { labels = None; _ } | Dim { labels = None; _ } -> string_of_int

(* Evaluating *)

(* [counted noun ppf k] formats [k noun] with its verb, such as [1 code is]. *)
let counted noun ppf k =
  if k = 1 then Format.fprintf ppf "1 %s is" noun
  else Format.fprintf ppf "%d %ss are" k noun

let ( ||| ) m m' =
  match (m, m') with
  | None, m | m, None -> m
  | Some a, Some b -> Some (Nx.logical_or a b)

let invalid valid = Option.map Nx.logical_not valid
let only valid m = match valid with None -> m | Some ok -> Nx.logical_and m ok

(* [along shape a t] is the vector [t] of the length of axis [a], shaped to
   broadcast along that axis of [shape]. *)
let along shape a t =
  Nx.reshape
    (Array.init
       (Array.length shape - a)
       (fun i -> if i = 0 then shape.(a) else 1))
    t

let beyond_int : type a b. (a, b) Nx.dtype -> Nx.int64_t -> Nx.bool_t option =
 fun dtype c ->
  let max = Int64.of_int max_int and min = Int64.of_int min_int in
  match dtype with
  | Nx.Int64 -> Some (Nx.logical_or (Nx.greater_s c max) (Nx.less_s c min))
  | Nx.UInt64 -> Some (Nx.logical_or (Nx.less_s c 0L) (Nx.greater_s c max))
  | _ -> None

(* [index ints c] is the index of the code [c] among the increasing [ints], [-1]
   where it is none of them. *)
let index ints c =
  let k = Array.length ints in
  if k = 0 then Nx.full_like c (-1L)
  else
    let d = Nx.create Nx.int64 [| k |] (Array.map Int64.of_int ints) in
    let pos =
      Nx.clamp ~max:(Int64.of_int (k - 1)) (Nx.searchsorted ~side:`Left d c)
    in
    Nx.where (Nx.equal (Nx.take ~indices:pos d) c) pos (Nx.full_like c (-1L))

(* [table labels domain] is the index in [domain] of each of [labels], [-1]
   where it is none of them. *)
let table labels domain =
  let at = Hashtbl.create (Array.length domain) in
  Array.iteri (fun i l -> Hashtbl.replace at l i) domain;
  let find l =
    Int64.of_int (Option.value ~default:(-1) (Hashtbl.find_opt at l))
  in
  Nx.create Nx.int64 [| Array.length labels |] (Array.map find labels)

let set_domain s =
  if Scale.sets Domain s then
    let (Scale.Categories c) = Scale.domain s in
    Some c
  else None

let quantities ?valid role s v =
  let m = Scale.missing s v in
  let undefined = lazy (only valid (Nx.logical_and m (Nx.isfinite v))) in
  let note k =
    Format.asprintf "%s: %a missing for its scale" role (counted "finite value")
      k
  in
  let miss =
    { rows = Some m ||| invalid valid; counts = [ (undefined, note) ] }
  in
  Quantities { values = v; miss }

let categories lift codes rows counts =
  Categories { lift; codes; miss = { rows; counts } }

let eval : type d. int array -> role:string -> d lift -> d Scale.t -> d t =
 fun shape ~role lift s ->
  let axis k = Option.get (axis_of shape k) in
  let arange n = Nx.arange Nx.int64 0 n 1 in
  match lift with
  | Num { x; valid } -> quantities ?valid role s (Nx.cast Nx.float64 x)
  | Index k ->
      let a = axis k in
      quantities role s (along shape a (Nx.cast Nx.float64 (arange shape.(a))))
  | Floats a -> quantities role s (Nx.create Nx.float64 [| Array.length a |] a)
  | Cat { codes; valid; labels = Some labels } ->
      let c = Nx.cast Nx.int64 codes in
      let n = Array.length labels in
      let out =
        Nx.logical_or (Nx.less_s c 0L) (Nx.greater_equal_s c (Int64.of_int n))
      in
      let m =
        match set_domain s with
        | Some (Scale.Labels domain) ->
            Nx.logical_or out
              (Nx.less_s (Nx.take ~indices:c (table labels domain)) 0L)
        | Some (Scale.Indices _) | None -> out
      in
      let note k =
        Format.asprintf "%s: %a outside its %d labels" role (counted "code") k n
      in
      categories lift c
        (Some m ||| invalid valid)
        [ (lazy (only valid out), note) ]
  | Cat { codes; valid; labels = None } ->
      let c = Nx.cast Nx.int64 codes in
      let beyond = beyond_int (Nx.dtype codes) c in
      let m =
        match set_domain s with
        | Some (Scale.Indices ix) ->
            Some (Nx.less_s (index (Array.map fst ix) c) 0L)
        | Some (Scale.Labels _) | None -> None
      in
      let note k =
        Format.asprintf "%s: %a beyond the range of int" role (counted "code") k
      in
      let counts =
        Option.to_list
          (Option.map (fun b -> (lazy (only valid b), note)) beyond)
      in
      categories lift c (beyond ||| invalid valid ||| m) counts
  | Strings a ->
      let n = Array.length a in
      let m =
        match set_domain s with
        | Some (Scale.Labels domain) -> Some (Nx.less_s (table a domain) 0L)
        | Some (Scale.Indices _) | None -> None
      in
      categories lift (arange n) m []
  | Dim { axis = k; valid; _ } ->
      let a = axis k in
      let c = along shape a (arange shape.(a)) in
      let m =
        match set_domain s with
        | Some (Scale.Indices ix) ->
            Some (Nx.less_s (index (Array.map fst ix) c) 0L)
        | Some (Scale.Labels _) | None -> None
      in
      categories lift c (m ||| invalid valid) []

(* Reading *)

let values : float t -> Nx.float64_t =
 fun (Quantities q) ->
  match q.miss.rows with
  | None -> q.values
  | Some m -> Nx.where m (Nx.full_like q.values Float.nan) q.values

let positions : string Scale.t -> string t -> Nx.int64_t =
 fun s (Categories c) ->
  let pos =
    match (Scale.domain s, labelled c.lift) with
    | Scale.Categories (Scale.Labels domain), true ->
        let labels =
          match c.lift with
          | Cat { labels = Some l; _ } | Strings l -> l
          | Cat { labels = None; _ } | Dim _ -> [||]
        in
        Nx.take ~indices:c.codes (table labels domain)
    | Scale.Categories (Scale.Indices ix), false ->
        index (Array.map fst ix) c.codes
    | Scale.Categories _, _ -> Nx.full_like c.codes (-1L)
  in
  match c.miss.rows with
  | None -> pos
  | Some m -> Nx.where m (Nx.full_like pos (-1L)) pos
