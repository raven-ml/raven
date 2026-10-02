(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Color = Hugin_next_gg.Color
module Text = Hugin_next_text.Text
module Scale = Hugin_next_kit.Scale
module Scheme = Hugin_next_kit.Scheme
module Symbol = Hugin_next_kit.Symbol
module Number = Hugin_next_kit.Number
module Ticks = Hugin_next_kit.Ticks
open Common
open Channel
open Figure
open Resolved

type ctx = { theme : Theme.t; scales : fitted array; frozen : Ticks.t array }

(* Theme ranges, in em *)

let size_em = 1.5 (* The diameter of the circle of the largest size. *)
let width_em = (0.05, 0.5)
let em ctx k = k *. Theme.size ctx.theme

(* Reading sources *)

type sel = All | Rows of int array

let numel s = Array.fold_left ( * ) 1 s
let count shape = function All -> numel shape | Rows r -> Array.length r
let row_of = function All -> Fun.id | Rows r -> fun k -> r.(k)

(* [source_of shape s] maps a flat index of [shape] to the flat index of a
   tensor of shape [s] broadcast to it. *)
let source_of shape s =
  if s = shape then Fun.id
  else
    let r = Array.length shape and k = Array.length s in
    let strides = Array.make r 0 and acc = ref 1 in
    for a = k - 1 downto 0 do
      if s.(a) <> 1 then strides.(r - k + a) <- !acc;
      acc := !acc * s.(a)
    done;
    fun i ->
      let rem = ref i and o = ref 0 in
      for a = r - 1 downto 0 do
        let d = shape.(a) in
        o := !o + (!rem mod d * strides.(a));
        rem := !rem / d
      done;
      !o

(* What a binding reads of its data, where the data lives: each row's quantity,
   NaN where missing; on a band scale, each row's index in its domain, [-1]
   where missing; or, on no scale, each row's category code and the missing
   rows. *)
type source =
  | Values of Nx.float64_t
  | Positions of Nx.int64_t
  | Codes of Nx.int64_t * Nx.bool_t option

type reader = {
  mark : mark;
  whole : bool;
  sources : ((int * int option) * source) list ref;
  floats : (Nx.packed * float array) list ref;
  ints : (Nx.packed * int64 array) list ref;
}

let reader ~whole mark =
  { mark; whole; sources = ref []; floats = ref []; ints = ref [] }

let cached cache x read =
  match List.find_opt (fun (Nx.P y, _) -> equal_tensor x y) !cache with
  | Some (_, a) -> a
  | None ->
      let a = read () in
      cache := (Nx.P x, a) :: !cache;
      a

(* [gather dtype cache ~whole shape x sel] is [x] broadcast to [shape] at the
   rows [sel], as [dtype]: from the whole source, read once, or gathered where
   it lives when the rows are few. *)
let gather : type a b c d.
    (c, d) Nx.dtype ->
    (Nx.packed * c array) list ref ->
    whole:bool ->
    int array ->
    (a, b) Nx.t ->
    sel ->
    c array =
 fun dtype cache ~whole shape x sel ->
  let s = Nx.shape x in
  let src = source_of shape s and m = count shape sel and row = row_of sel in
  let few = match sel with All -> false | Rows _ -> 2 * m < numel s in
  if whole || not few then
    let a = cached cache x (fun () -> Nx.to_array (Nx.cast dtype x)) in
    match sel with
    | All when s = shape -> a
    | All | Rows _ -> Array.init m (fun k -> a.(src (row k)))
  else
    let idx = Array.init m (fun k -> Int64.of_int (src (row k))) in
    Nx.to_array
      (Nx.cast dtype (Nx.take ~indices:(Nx.create Nx.int64 [| m |] idx) x))

let floats rd x sel =
  gather Nx.float64 rd.floats ~whole:rd.whole rd.mark.shape x sel

let ints rd x sel =
  Array.map Int64.to_int
    (gather Nx.int64 rd.ints ~whole:rd.whole rd.mark.shape x sel)

let evaluate : type d.
    int array -> role:string -> d lift -> fitted option -> source =
 fun shape ~role lift scale ->
  match (kind lift, scale) with
  | Quantities, Some (F { kind = Quantities; spec; _ }) ->
      Values (Lift.values (Lift.eval shape ~role lift spec))
  | Quantities, _ ->
      Values (Lift.values (Lift.eval shape ~role lift (Scale.linear ())))
  | Categories, Some (F { kind = Categories; scale; _ }) ->
      Positions (Lift.positions scale (Lift.eval shape ~role lift scale))
  | Categories, _ ->
      let (Lift.Categories c) = Lift.eval shape ~role lift (Scale.band ()) in
      Codes (c.codes, c.miss.rows)

(* [source rd (B b) index scale] is what the binding [index] reads of its data
   on the scale [scale], if any, evaluated once per reader. Quantities do not
   depend on which of a name's per-panel scales reads them. *)
let source rd (B b) index scale =
  let key =
    match scale with
    | Some (i, F { kind = Categories; _ }) -> (index, Some i)
    | Some (_, F { kind = Quantities; _ }) | None -> (index, None)
  in
  match List.assoc_opt key !(rd.sources) with
  | Some s -> s
  | None ->
      let d = Option.get (data b.ch) in
      let s =
        evaluate rd.mark.shape ~role:b.role.name d.lift (Option.map snd scale)
      in
      rd.sources := (key, s) :: !(rd.sources);
      s

(* Ranges *)

let stroked m =
  Option.is_some (find_binding Role.stroke m.bindings)
  && Option.is_none (find_binding Role.fill m.bindings)

let lerp (a, b) u = a +. (u *. (b -. a))

(* How a role's range maps the values of a scale: by the normalised value, or on
   a band scale by the category's index in its domain. *)
type 'r range = By_value of (float -> 'r option) | By_index of 'r array

(* Steps *)

let levels ctx i =
  match ctx.scales.(i) with
  | F { kind = Quantities; scale; _ } when Scale.stepped scale ->
      let inner (t : Ticks.tick) =
        if t.position > 0. && t.position < 1. then Some t.position else None
      in
      let ticks = List.filter_map inner ctx.frozen.(i).major in
      Some (Array.of_list (List.sort_uniq Float.compare (0. :: 1. :: ticks)))
  | F _ -> None

(* [step levels u] is the middle of the interval between consecutive [levels]
   that holds [u], each holding its lower level and the last also its upper one,
   or of the interval at the nearer end if [u] lies beyond them. *)
let step levels u =
  (* The greatest [k] below [n - 1] with [levels.(k) <= u], or [0]. *)
  let rec find lo hi =
    if hi - lo <= 1 then lo
    else
      let mid = (lo + hi) / 2 in
      if levels.(mid) <= u then find mid hi else find lo mid
  in
  let k = find 0 (Array.length levels - 1) in
  (levels.(k) +. levels.(k + 1)) /. 2.

(* [base ctx ~stroked use range i] is how a role of [use] maps the values of the
   scale [i], [None] where they are missing, and the value of a missing one. *)
let base : type r.
    ctx ->
    stroked:bool ->
    Role.use ->
    r Role.range ->
    int ->
    (r range * r) option =
 fun ctx ~stroked use range i ->
  let (F f) = ctx.scales.(i) in
  let step = match levels ctx i with None -> Fun.id | Some l -> step l in
  let finite g =
    By_value (fun u -> if Float.is_nan u then None else Some (g (step u)))
  in
  let n () =
    match f.kind with
    | Categories -> List.length (category_names f.scale)
    | Quantities -> 0
  in
  match range with
  | Role.Colors -> (
      let unknown =
        Option.value (Scale.unknown f.scale) ~default:Color.transparent
      in
      match f.kind with
      | Categories ->
          let scheme =
            Option.value (Scale.scheme f.scale)
              ~default:(Theme.palette ctx.theme)
          in
          Some (By_index (Scheme.colors (n ()) scheme), unknown)
      | Quantities ->
          let scheme =
            Option.value (Scale.scheme f.scale)
              ~default:(Theme.scheme ctx.theme)
          in
          Some (finite (Scheme.color scheme), unknown))
  | Role.Floats ->
      let areas : float * float =
        let circle = Float.pi *. Float.pow (em ctx size_em /. 2.) 2. in
        match f.kind with
        | Quantities -> Option.value (Scale.areas f.scale) ~default:(0., circle)
        | Categories -> (0., circle)
      in
      let g =
        match use with
        | Encoding { map = Opacity; _ } ->
            fun u -> Float.min 1. (Float.max 0. u)
        | Encoding { map = Area; _ } -> lerp areas
        | Encoding { map = Width; _ } ->
            lerp (em ctx (fst width_em), em ctx (snd width_em))
        | Encoding { map = Color | Shape; _ } | Position _ | Facet _ | Value ->
            Fun.id
      in
      Some (finite g, Float.nan)
  | Role.Symbols -> (
      match f.kind with
      | Categories ->
          let symbols =
            match Scale.symbols f.scale with
            | Some s -> s
            | None ->
                Array.of_list
                  (if stroked then Symbol.stroked else Symbol.filled)
          in
          let k = Array.length symbols in
          Some
            ( By_index (Array.init (n ()) (fun i -> symbols.(i mod k))),
              symbols.(0) )
      | Quantities -> None)
  | Role.Panels -> (
      match f.kind with
      | Categories ->
          Some (By_index (Array.of_list (category_names f.scale)), "")
      | Quantities -> None)
  | Role.Texts | Role.Param _ -> None

(* [index_at s] is the index of the category of the band scale [s] whose step
   holds a normalised value, if any. *)
let index_at s =
  let index = Hashtbl.create 16 in
  List.iteri (fun k c -> Hashtbl.replace index c k) (category_names s);
  fun u -> Option.bind (Scale.invert s u) (Hashtbl.find_opt index)

(* [at s range] is the value [range] gives a normalised value on [s]. *)
let at (F f) = function
  | By_value at -> at
  | By_index t -> (
      match f.kind with
      | Categories ->
          let index_at = index_at f.scale in
          fun u -> Option.map (Array.get t) (index_at u)
      | Quantities -> fun _ -> None)

let colors ctx i =
  match base ctx ~stroked:false Role.fill.use Role.Colors i with
  | Some (range, missing) ->
      let at = at ctx.scales.(i) range in
      fun u -> Option.value (at u) ~default:missing
  | None -> fun _ -> Color.transparent

let area ctx i u =
  match base ctx ~stroked:false Role.size.use Role.Floats i with
  | Some (range, _) ->
      Option.value (at ctx.scales.(i) range u) ~default:Float.nan
  | None -> Float.nan

(* Columns *)

let column ?norm ?fn ?ticks ?cats ?band ?zero role values =
  Rows.Col { role; values; norm; fn; ticks; cats; band; zero }

(* [zero s] is the normalised value of [0.] clamped into the domain of [s]. *)
let zero (s : float Scale.t) =
  let (Scale.Floats (a, b)) = Scale.domain s in
  Scale.normalize s (Float.max (Float.min a b) (Float.min (Float.max a b) 0.))

(* [fn s range missing g] is the value of a channel that [g] maps, on [s] in
   [range], at a normalised value. *)
let fn s range missing g =
  let at = at s range in
  fun u -> match at u with None -> missing | Some v -> g v

let ticks ctx i =
  Array.of_list
    (List.map (fun (t : Ticks.tick) -> t.position) ctx.frozen.(i).major)

(* [scaled ctx ~stroked rd (B b as bd) index sel i] is, if the role of [b] maps
   the scale [i], the column of [b] at the rows [sel] on it, and the rows it
   misses. *)
let scaled ctx ~stroked rd (B b as bd) index sel i =
  let (F f as s) = ctx.scales.(i) in
  match base ctx ~stroked b.role.use b.role.range i with
  | None -> None
  | Some (range, missing) -> (
      let g = mapping b.ch and ticks = ticks ctx i in
      let fn = fn s range missing g in
      match (f.kind, source rd bd index (Some (i, s))) with
      | Quantities, Values v ->
          let norm = Array.map (Scale.normalize f.scale) (floats rd v sel) in
          let zero = zero f.scale in
          let col = column b.role (Array.map fn norm) ~norm ~fn ~ticks ~zero in
          Some (col, Array.map Float.is_nan norm)
      | Categories, Positions p ->
          let ks = ints rd p sel in
          let norm = Array.map (Scale.normalize_index f.scale) ks in
          let values =
            match range with
            | By_index t ->
                Array.map (fun k -> if k < 0 then missing else g t.(k)) ks
            | By_value _ -> Array.map fn norm
          in
          let band = Scale.bandwidth f.scale in
          let col = column b.role values ~norm ~fn ~ticks ~cats:ks ~band in
          Some (col, Array.map (fun k -> k < 0) ks)
      | (Quantities | Categories), _ ->
          assert false (* A reading's scale has the reading's kind. *))

(* [decimals l] is the decimals that write a quantity of [l]. *)
let decimals : type d. d lift -> float -> int = function
  | Num { x; _ } -> Number.decimals (Nx.dtype x)
  | Floats _ -> Number.decimals Nx.float64
  | Index _ | Cat _ | Strings _ | Dim _ -> fun _ -> 0

(* [label l] is the text that shows the category of a code of [l]. *)
let label : type d. d lift -> int -> string =
 fun l ->
  match kind l with Categories -> Lift.label l | Quantities -> string_of_int

(* [unscaled ctx rd (B b as bd) index sel] is the column of a binding that reads
   no scale, and the rows it misses. *)
let unscaled ctx rd (B b as bd) index sel =
  let g = mapping b.ch and lift = (Option.get (data b.ch)).lift in
  let blank = Text.v "" in
  let nums, texts, miss =
    match source rd bd index None with
    | Values v ->
        let v = floats rd v sel in
        let miss = Array.map Float.is_nan v in
        let texts =
          lazy
            (let d = ref 0 and dec = decimals lift in
             Array.iteri
               (fun i v -> if not miss.(i) then d := Int.max !d (dec v))
               v;
             let fmt = Number.v Number.Plain (Number.Decimals !d) in
             let locale = Theme.locale ctx.theme in
             Array.map
               (fun v ->
                 if Float.is_nan v then blank
                 else Text.v (Number.to_string ~locale fmt v))
               v)
        in
        (v, texts, miss)
    | Codes (c, m) ->
        let codes = ints rd c sel in
        let miss =
          match m with
          | None -> Array.make (Array.length codes) false
          | Some m -> Array.map (fun f -> f <> 0.) (floats rd m sel)
        in
        let texts =
          lazy
            (let label = label lift in
             Array.mapi
               (fun j k -> if miss.(j) then blank else Text.v (label k))
               codes)
        in
        (Array.make (Array.length codes) Float.nan, texts, miss)
    | Positions _ -> assert false (* Positions are read on a scale. *)
  in
  match b.role.range with
  | Role.Floats ->
      let g i v = if miss.(i) then Float.nan else g v in
      (column b.role (Array.mapi g nums), miss)
  | Role.Texts ->
      let g i t = if miss.(i) then t else g t in
      (column b.role (Array.mapi g (Lazy.force texts)), miss)
  | Role.Colors | Role.Symbols | Role.Panels | Role.Param _ ->
      err "draw" "the role %s reads no scale" b.role.name

let rows ?only ctx rd ~id projection ~warn reads sel =
  let m = rd.mark and stroked = stroked rd.mark in
  let n = count m.shape sel in
  let dropped = Array.make n false in
  let col index (B b as bd) =
    match Channel.constant b.ch with
    | Some v -> column b.role (Array.make n v)
    | None ->
        let col, miss =
          match
            Option.bind reads.(index) (scaled ctx ~stroked rd bd index sel)
          with
          | Some col -> col
          | None -> unscaled ctx rd bd index sel
        in
        (match b.role.range with
        | Role.Colors -> ()
        | _ -> Array.iteri (fun i x -> if x then dropped.(i) <- true) miss);
        col
  in
  let wanted (B b) =
    match only with None -> true | Some l -> List.mem b.role.name l
  in
  let cols =
    List.concat
      (List.mapi (fun i b -> if wanted b then [ col i b ] else []) m.bindings)
  in
  let index =
    match sel with All -> Array.init n Fun.id | Rows r -> Array.copy r
  in
  {
    Rows.id;
    shape = m.shape;
    index;
    theme = ctx.theme;
    projection;
    cols;
    dropped;
    warn;
  }

(* A colour role bound to data a legend does not show swatches in the ink at
   this fraction of its opacity. *)
let neutral_alpha = 0.5

let swatch ctx m ~id projection ~warn ~scale ~reads ~n ~k u =
  let stroked = stroked m in
  let col index (B b) =
    match Channel.constant b.ch with
    | Some v -> Some (column b.role [| v |])
    | None when reads index -> (
        let (F f as s) = ctx.scales.(scale) in
        match base ctx ~stroked b.role.use b.role.range scale with
        | None -> None
        | Some (range, missing) ->
            let fn = fn s range missing (mapping b.ch) in
            let ticks = ticks ctx scale in
            let cats, band, zero =
              match f.kind with
              | Categories ->
                  let k = Option.value ~default:(-1) (index_at f.scale u) in
                  (Some [| k |], Some (Scale.bandwidth f.scale), None)
              | Quantities -> (None, None, Some (zero f.scale))
            in
            Some
              (column b.role
                 [| fn u |]
                 ~norm:[| u |] ~fn ~ticks ?cats ?band ?zero))
    | None -> (
        match b.role.range with
        | Role.Colors ->
            let ink = Theme.ink ctx.theme in
            let alpha = Color.alpha ink *. neutral_alpha in
            Some (column b.role [| Color.with_alpha alpha ink |])
        | Role.Floats | Role.Texts | Role.Symbols | Role.Panels | Role.Param _
          ->
            None)
  in
  {
    Rows.id;
    shape = [| n |];
    index = [| k |];
    theme = ctx.theme;
    projection;
    cols = List.filter_map Fun.id (List.mapi col m.bindings);
    dropped = [| false |];
    warn;
  }
