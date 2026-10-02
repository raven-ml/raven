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

type ctx = {
  theme : Theme.t;
  density : float;
  scales : fitted array;
  frozen : Ticks.t array;
}

type scale_of = int -> int option

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

(* [along shape a] is the index along axis [a] of a flat index of [shape]. *)
let along shape a =
  let stride = ref 1 in
  for b = a + 1 to Array.length shape - 1 do
    stride := !stride * shape.(b)
  done;
  let stride = !stride and d = shape.(a) in
  fun i -> i / stride mod d

type reader = {
  mark : mark;
  whole : bool;
  floats : (Nx.packed * float array) list ref;
  ints : (Nx.packed * int64 array) list ref;
}

let reader ~whole mark = { mark; whole; floats = ref []; ints = ref [] }
let mark rd = rd.mark

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

let ints rd x sel = gather Nx.int64 rd.ints ~whole:rd.whole rd.mark.shape x sel

(* Host values *)

(* Identities below [dense] are memoised in arrays. *)
let dense = 65_536

type host =
  | Q of { v : float array; ok : bool array option; dec : float -> int }
  | C of {
      ids : int array; (* An identity per category. *)
      miss : bool array;
      name : int -> string;
      text : int -> Text.t;
    }

let valid rd v sel =
  Option.map (fun v -> Array.map (fun f -> f <> 0.) (floats rd v sel)) v

let is_ok ok i = match ok with None -> true | Some ok -> ok.(i)

let host : type d. reader -> d lift -> sel -> host =
 fun rd lift sel ->
  let shape = rd.mark.shape in
  let m = count shape sel and row = row_of sel in
  let axis k = Option.get (axis_of shape k) in
  let indexed ok ids labels =
    let text =
      match labels with
      | Some l -> fun i -> Text.v l.(i)
      | None -> fun i -> Text.v (string_of_int i)
    in
    C
      {
        ids;
        miss = Array.init m (fun i -> not (is_ok ok i));
        name = string_of_int;
        text;
      }
  in
  match lift with
  | Num { x; valid = v } ->
      Q
        {
          v = floats rd x sel;
          ok = valid rd v sel;
          dec = Number.decimals (Nx.dtype x);
        }
  | Index k ->
      let at = along shape (axis k) in
      Q
        {
          v = Array.init m (fun j -> float (at (row j)));
          ok = None;
          dec = (fun _ -> 0);
        }
  | Scalar x ->
      Q { v = Array.make m x; ok = None; dec = Number.decimals Nx.float64 }
  | Cat { codes; valid = v; labels } ->
      let cs = ints rd codes sel and ok = valid rd v sel in
      let lo, hi =
        match labels with
        | Some l -> (0L, Int64.of_int (Array.length l - 1))
        | None -> (Int64.of_int min_int, Int64.of_int max_int)
      in
      let miss =
        Array.mapi (fun i c -> (not (is_ok ok i)) || c < lo || c > hi) cs
      in
      let ids =
        Array.mapi (fun i c -> if miss.(i) then -1 else Int64.to_int c) cs
      in
      let name =
        match labels with Some l -> fun i -> l.(i) | None -> string_of_int
      in
      C { ids; miss; name; text = (fun i -> Text.v (name i)) }
  | Strings a ->
      let src = source_of shape [| Array.length a |] in
      let table = Hashtbl.create 16 and names = ref [] in
      let id s =
        match Hashtbl.find_opt table s with
        | Some i -> i
        | None ->
            let i = Hashtbl.length table in
            Hashtbl.add table s i;
            names := s :: !names;
            i
      in
      let ids = Array.init m (fun j -> id a.(src (row j))) in
      let names = Array.of_list (List.rev !names) in
      C
        {
          ids;
          miss = Array.make m false;
          name = (fun i -> names.(i));
          text = (fun i -> Text.v names.(i));
        }
  | Dim { axis = k; valid = v; labels } ->
      let at = along shape (axis k) in
      indexed (valid rd v sel) (Array.init m (fun j -> at (row j))) labels

(* [memo miss ids f] is [f] applied once per identity of [ids] that [miss] does
   not mark: through an array when the identities are small naturals, as indices
   along an axis are, and a hash table otherwise. *)
let memo miss ids f =
  let lo = ref 0 and hi = ref (-1) in
  Array.iteri
    (fun j id ->
      if not miss.(j) then begin
        lo := Int.min !lo id;
        hi := Int.max !hi id
      end)
    ids;
  if !lo >= 0 && !hi < dense then begin
    let values = Array.make (!hi + 1) None in
    fun id ->
      match values.(id) with
      | Some v -> v
      | None ->
          let v = f id in
          values.(id) <- Some v;
          v
  end
  else
    let t = Hashtbl.create 16 in
    fun id ->
      match Hashtbl.find_opt t id with
      | Some v -> v
      | None ->
          let v = f id in
          Hashtbl.add t id v;
          v

let ids = function Q q -> Array.make (Array.length q.v) 0 | C c -> c.ids

let missing = function
  | Q q ->
      Array.mapi
        (fun i v -> (not (Float.is_finite v)) || not (is_ok q.ok i))
        q.v
  | C c -> Array.copy c.miss

let normalize : type d. d Scale.t -> host -> float array =
 fun s h ->
  match (Scale.kind s, h) with
  | Scale.Quantitative, Q q ->
      let nz = Scale.normalize s in
      Array.mapi (fun i v -> if is_ok q.ok i then nz v else Float.nan) q.v
  | Scale.Categorical, C c ->
      let nz = Scale.normalize s in
      let at = memo c.miss c.ids (fun i -> nz (c.name i)) in
      Array.mapi (fun j i -> if c.miss.(j) then Float.nan else at i) c.ids
  | _, Q { v; _ } -> Array.make (Array.length v) Float.nan
  | _, C { ids; _ } ->
      (* A reading's scale has the reading's kind, and temporal channels do not
         exist yet. *)
      Array.make (Array.length ids) Float.nan

(* Ranges *)

let stroked m =
  Option.is_some (find_binding "stroke" m.bindings)
  && Option.is_none (find_binding "fill" m.bindings)

(* [categories s] maps the name of each category of [s] to its index. *)
let categories s =
  let names = category_names s in
  let t = Hashtbl.create (List.length names) in
  List.iteri (fun i c -> Hashtbl.replace t c i) names;
  (List.length names, t)

(* [band s at] reads the category whose step holds a normalised value, each
   value once: a band scale gives its rows a handful of values. *)
let band : string Scale.t -> (int -> 'r) -> float -> 'r option =
 fun s at ->
  let _, t = categories s and memo = Hashtbl.create 16 in
  fun u ->
    match Hashtbl.find_opt memo u with
    | Some v -> v
    | None ->
        let v =
          Option.bind (Scale.invert s u) (fun c ->
              Option.map at (Hashtbl.find_opt t c))
        in
        Hashtbl.add memo u v;
        v

let lerp (a, b) u = a +. (u *. (b -. a))

(* [base ctx ~stroked name range s] is the value the role [name] gives a
   normalised value on [s], [None] where it is missing, and the value of a
   missing one. *)
let base : type r.
    ctx ->
    stroked:bool ->
    string ->
    r Role.range ->
    fitted ->
    ((float -> r option) * r) option =
 fun ctx ~stroked name range (F f) ->
  let finite g u = if Float.is_nan u then None else Some (g u) in
  match range with
  | Role.Colors -> (
      let unknown =
        Option.value (Scale.unknown f.scale) ~default:Color.transparent
      in
      match f.kind with
      | Scale.Categorical ->
          let n, _ = categories f.scale in
          let scheme =
            Option.value (Scale.scheme f.scale)
              ~default:(Theme.palette ctx.theme)
          in
          let colors = Scheme.colors n scheme in
          Some (band f.scale (fun i -> colors.(i)), unknown)
      | Scale.Quantitative | Scale.Temporal ->
          let scheme =
            Option.value (Scale.scheme f.scale)
              ~default:(Theme.scheme ctx.theme)
          in
          Some (finite (Scheme.color scheme), unknown))
  | Role.Floats ->
      let areas : float * float =
        let circle = Float.pi *. Float.pow (em ctx size_em /. 2.) 2. in
        match f.kind with
        | Scale.Quantitative ->
            Option.value (Scale.areas f.scale) ~default:(0., circle)
        | Scale.Categorical | Scale.Temporal -> (0., circle)
      in
      let g =
        match name with
        | "opacity" -> fun u -> Float.min 1. (Float.max 0. u)
        | "size" -> lerp areas
        | "width" -> lerp (em ctx (fst width_em), em ctx (snd width_em))
        | _ -> Fun.id
      in
      Some (finite g, Float.nan)
  | Role.Symbols -> (
      match f.kind with
      | Scale.Categorical ->
          let symbols =
            match Scale.symbols f.scale with
            | Some s -> s
            | None ->
                Array.of_list
                  (if stroked then Symbol.stroked else Symbol.filled)
          in
          let k = Array.length symbols in
          Some (band f.scale (fun i -> symbols.(i mod k)), symbols.(0))
      | Scale.Quantitative | Scale.Temporal -> None)
  | Role.Panels -> (
      match f.kind with
      | Scale.Categorical ->
          let names = Array.of_list (category_names f.scale) in
          Some (band f.scale (fun i -> names.(i)), "")
      | Scale.Quantitative | Scale.Temporal -> None)
  | Role.Texts | Role.Curves | Role.Pixels -> None

let colors ctx s =
  match base ctx ~stroked:false "fill" Role.Colors s with
  | Some (at, missing) -> fun u -> Option.value (at u) ~default:missing
  | None -> fun _ -> Color.transparent

(* Columns *)

let rec maps : type d r. (d, r) Channel.t -> r -> r = function
  | Map (f, c) ->
      let g = maps c in
      fun v -> f (g v)
  | Const _ | Data _ -> Fun.id

let rec const_value : type d r. (d, r) Channel.t -> r option = function
  | Const v -> Some v
  | Map (f, c) -> Option.map f (const_value c)
  | Data _ -> None

let constant : type d r. (d, r) Role.t -> int -> r -> Rows.col =
 fun role n v ->
  Rows.Col
    {
      name = role.name;
      range = role.range;
      values = Array.make n v;
      norm = None;
      fn = None;
      ticks = None;
      cats = None;
      band = None;
      zero = None;
    }

(* [zero s] is the normalised value of [0.] clamped into the domain of [s]. *)
let zero (s : float Scale.t) =
  let (Scale.Floats (a, b)) = Scale.domain s in
  Scale.normalize s (Float.max (Float.min a b) (Float.min (Float.max a b) 0.))

(* [scaled ctx ~stroked (B b) norm ids i] is the column of [b] normalised to
   [norm] on the scale [i], [ids] identifying the categories of its rows on a
   band scale. *)
let scaled ctx ~stroked (B b) norm ids i =
  let (F f as s) = ctx.scales.(i) in
  match base ctx ~stroked b.role.name b.role.range s with
  | None -> None
  | Some (at, missing) ->
      let g = maps b.ch in
      let fn u = match at u with None -> missing | Some v -> g v in
      let band, zero_at, cats =
        match f.kind with
        | Scale.Categorical ->
            let cat j id = if Float.is_nan norm.(j) then min_int else id in
            (Some (Scale.bandwidth f.scale), None, Some (Array.mapi cat ids))
        | Scale.Quantitative -> (None, Some (zero f.scale), None)
        | Scale.Temporal -> (None, None, None)
      in
      let ticks =
        List.map (fun (t : Ticks.tick) -> t.position) ctx.frozen.(i).major
      in
      Some
        (Rows.Col
           {
             name = b.role.name;
             range = b.role.range;
             values = Array.map fn norm;
             norm = Some norm;
             fn = Some fn;
             ticks = Some (Array.of_list ticks);
             cats;
             band;
             zero = zero_at;
           })

(* [format ctx h] writes the values of [h] as the text role writes them. *)
let format ctx h =
  let blank = Text.v "" in
  match h with
  | C c -> Array.mapi (fun j i -> if c.miss.(j) then blank else c.text i) c.ids
  | Q q ->
      let miss = missing h in
      let d = ref 0 in
      Array.iteri
        (fun i v -> if not miss.(i) then d := Int.max !d (q.dec v))
        q.v;
      let fmt = Number.v Number.Plain (Number.Decimals !d) in
      let locale = Theme.locale ctx.theme in
      Array.mapi
        (fun i v ->
          if miss.(i) then blank else Text.v (Number.to_string ~locale fmt v))
        q.v

(* [unscaled ctx (B b) h] is the column of a channel that reads no scale. *)
let unscaled : type d r.
    ctx -> (d, r) Role.t -> (d, r) Channel.t -> host -> Rows.col =
 fun ctx role ch h ->
  let col values =
    Rows.Col
      {
        name = role.name;
        range = role.range;
        values;
        norm = None;
        fn = None;
        ticks = None;
        cats = None;
        band = None;
        zero = None;
      }
  in
  let g = maps ch in
  let miss = missing h in
  match role.range with
  | Role.Floats ->
      let vs =
        match h with
        | Q q -> Array.mapi (fun i v -> if miss.(i) then Float.nan else g v) q.v
        | C c -> Array.make (Array.length c.ids) Float.nan
      in
      col vs
  | Role.Texts ->
      col (Array.mapi (fun i t -> if miss.(i) then t else g t) (format ctx h))
  | Role.Colors | Role.Symbols | Role.Panels | Role.Curves | Role.Pixels ->
      err "draw" "the role %s reads no scale" role.name

let rows ?only ctx rd ~id projection ~warn scale_of sel =
  let m = rd.mark and stroked = stroked rd.mark in
  let n = count m.shape sel in
  let dropped = Array.make n false in
  let drop (B b) miss =
    match b.role.range with
    | Role.Colors -> ()
    | _ -> Array.iteri (fun i x -> if x then dropped.(i) <- true) miss
  in
  let col index (B b as bd) =
    match const_value b.ch with
    | Some v -> constant b.role n v
    | None -> (
        let d = Option.get (data b.ch) in
        let h = host rd d.lift sel in
        let scaled =
          match (b.role.scale, scale_of index) with
          | Some _, Some i ->
              let (F f) = ctx.scales.(i) in
              let norm = normalize f.scale h in
              Option.map
                (fun c -> (c, Array.map Float.is_nan norm))
                (scaled ctx ~stroked bd norm (ids h) i)
          | _ -> None
        in
        match scaled with
        | Some (c, miss) ->
            drop bd miss;
            c
        | None ->
            drop bd (missing h);
            unscaled ctx b.role b.ch h)
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
    density = ctx.density;
    cols;
    dropped;
    warn;
  }

let facet rd role =
  match find_binding role rd.mark.bindings with
  | None -> None
  | Some (B b) -> (
      match data b.ch with
      | None -> None
      | Some d -> (
          match host rd d.lift All with
          | C c ->
              let ids =
                Array.mapi
                  (fun j id -> if c.miss.(j) then min_int else id)
                  c.ids
              in
              Some (ids, c.name)
          | Q _ -> None))

let swatch ctx m ~id projection ~warn ~scale ~reads ~n ~k u =
  let stroked = stroked m in
  let col index (B b as bd) =
    match const_value b.ch with
    | Some v -> Some (constant b.role 1 v)
    | None ->
        if reads index then scaled ctx ~stroked bd [| u |] [| 0 |] scale
        else None
  in
  {
    Rows.id;
    shape = [| n |];
    index = [| k |];
    theme = ctx.theme;
    projection;
    density = ctx.density;
    cols = List.filter_map Fun.id (List.mapi col m.bindings);
    dropped = [| false |];
    warn;
  }
