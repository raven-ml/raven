(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Scale = Hugin_kit.Scale
open Common

type _ kind = Quantities : float kind | Categories : string kind

let equal_kind : type a b. a kind -> b kind -> (a, b) Type.eq option =
 fun k k' ->
  match (k, k') with
  | Quantities, Quantities -> Some Type.Equal
  | Categories, Categories -> Some Type.Equal
  | Quantities, Categories | Categories, Quantities -> None

let pp_kind : type d. Format.formatter -> d kind -> unit =
 fun ppf k ->
  Format.pp_print_string ppf
    (match k with Quantities -> "quantitative" | Categories -> "categorical")

let of_scale_kind : type d. d Scale.kind -> d kind option = function
  | Scale.Quantitative -> Some Quantities
  | Scale.Categorical -> Some Categories
  | Scale.Temporal -> None

type _ lift =
  | Num : { x : ('a, 'b) Nx.t; valid : Nx.bool_t option } -> float lift
  | Index : int -> float lift
  | Floats : float array -> float lift
  | Cat : {
      codes : ('a, 'b) Nx.t;
      valid : Nx.bool_t option;
      labels : string array option;
    }
      -> string lift
  | Strings : string array -> string lift
  | Dim : {
      axis : int;
      valid : Nx.bool_t option;
      labels : string array option;
    }
      -> string lift

type 'd data = {
  lift : 'd lift;
  spec : 'd Scale.t option;
  title : string option;
}

type ('d, 'r) t =
  | Const : 'r -> ('d, 'r) t
  | Data : 'd data -> ('d, 'r) t
  | Map : ('r -> 'r) * ('d, 'r) t -> ('d, 'r) t

let kind : type d. d lift -> d kind = function
  | Num _ -> Quantities
  | Index _ -> Quantities
  | Floats _ -> Quantities
  | Cat _ -> Categories
  | Strings _ -> Categories
  | Dim _ -> Categories

let equal_lift : type d e. d lift -> e lift -> (d, e) Type.eq option =
 fun l l' ->
  let ok b = if b then Some Type.Equal else None in
  match (l, l') with
  | Num a, Num b ->
      ok (equal_tensor a.x b.x && Option.equal ( == ) a.valid b.valid)
  | Index k, Index k' -> ok (Int.equal k k')
  | Floats a, Floats b -> ok (Array.equal Float.equal a b)
  | Cat a, Cat b ->
      ok
        (equal_tensor a.codes b.codes
        && Option.equal ( == ) a.valid b.valid
        && Option.equal equal_strings a.labels b.labels)
  | Strings a, Strings b -> ok (equal_strings a b)
  | Dim a, Dim b ->
      ok
        (Int.equal a.axis b.axis
        && Option.equal ( == ) a.valid b.valid
        && Option.equal equal_strings a.labels b.labels)
  | _ -> None

let rec equal : type d e r. r Role.range -> (d, r) t -> (e, r) t -> bool =
 fun r c c' ->
  match (c, c') with
  | Const v, Const v' -> Role.equal_in r v v'
  | Data a, Data b -> (
      match equal_lift a.lift b.lift with
      | Some Type.Equal ->
          Option.equal Scale.equal a.spec b.spec
          && Option.equal String.equal a.title b.title
      | None -> false)
  | Map (f, c), Map (f', c') -> f == f' && equal r c c'
  | _ -> false

let rec data : type d r. (d, r) t -> d data option = function
  | Const _ -> None
  | Data d -> Some d
  | Map (_, c) -> data c

let rec constant : type d r. (d, r) t -> r option = function
  | Const v -> Some v
  | Map (f, c) -> Option.map f (constant c)
  | Data _ -> None

let rec mapping : type d r. (d, r) t -> r -> r = function
  | Map (f, c) ->
      let g = mapping c in
      fun v -> f (g v)
  | Const _ | Data _ -> Fun.id

let scale_kind : type d r. (d, r) t -> d Scale.kind option =
 fun c ->
  match data c with
  | None -> None
  | Some d -> (
      match kind d.lift with
      | Quantities -> Some Scale.Quantitative
      | Categories -> Some Scale.Categorical)

(* Lifts *)

let is_real : type a b. (a, b) Nx.dtype -> bool = function
  | Nx.Complex64 | Nx.Complex128 | Nx.Bool | Nx.Bit -> false
  | _ -> true

let is_integer : type a b. (a, b) Nx.dtype -> bool = function
  | Nx.Int4 | Nx.UInt4 | Nx.Int8 | Nx.UInt8 | Nx.Int16 | Nx.UInt16 | Nx.Int32
  | Nx.UInt32 | Nx.Int64 | Nx.UInt64 ->
      true
  | _ -> false

(* [fits s s'] is [true] iff [s] broadcasts to [s'] without growing it. *)
let fits s s' =
  let n = Array.length s and n' = Array.length s' in
  n <= n'
  &&
  let ok = ref true in
  for i = 0 to n - 1 do
    let d = s.(i) in
    if d <> 1 && d <> s'.(n' - n + i) then ok := false
  done;
  !ok

let check_valid fn shape = function
  | None -> ()
  | Some v ->
      if not (fits (Nx.shape v) shape) then
        err fn "valid, of shape %a, does not broadcast to the shape %a" pp_shape
          (Nx.shape v) pp_shape shape

let check_distinct fn labels =
  let seen = Hashtbl.create (Array.length labels) in
  Array.iter
    (fun l ->
      if Hashtbl.mem seen l then err fn "the label %S is repeated" l;
      Hashtbl.add seen l ())
    labels

let num ?scale ?valid ?title x =
  if not (is_real (Nx.dtype x)) then
    err "num" "the dtype %s is not real" (Nx_dtype.to_string (Nx.dtype x));
  check_valid "num" (Nx.shape x) valid;
  Data { lift = Num { x; valid }; spec = scale; title }

let cat ?scale ?valid ?title ?labels codes =
  if not (is_integer (Nx.dtype codes)) then
    err "cat" "the dtype %s is not an integer dtype"
      (Nx_dtype.to_string (Nx.dtype codes));
  check_valid "cat" (Nx.shape codes) valid;
  Option.iter (check_distinct "cat") labels;
  let labels = Option.map Array.copy labels in
  Data { lift = Cat { codes; valid; labels }; spec = scale; title }

let strings ?scale ?title a =
  Data { lift = Strings (Array.copy a); spec = scale; title }

let floats ?scale ?title a =
  Data { lift = Floats (Array.copy a); spec = scale; title }

let dim ?scale ?valid ?title ?labels axis =
  let labels = Option.map Array.copy labels in
  Data { lift = Dim { axis; valid; labels }; spec = scale; title }

let index ?scale ?title axis = Data { lift = Index axis; spec = scale; title }
let const v = Const v
let map_range f c = Map (f, c)

let axis_of shape k =
  let n = Array.length shape in
  let a = if k < 0 then n + k else k in
  if a < 0 || a >= n then None else Some a

let lift_shape : type d. d lift -> int array option = function
  | Num { x; _ } -> Some (Nx.shape x)
  | Cat { codes; _ } -> Some (Nx.shape codes)
  | Strings a -> Some [| Array.length a |]
  | Floats a -> Some [| Array.length a |]
  | Dim { valid = Some v; _ } -> Some (Nx.shape v)
  | Dim { valid = None; _ } | Index _ -> None

let varies : type d r. int array -> (d, r) t -> int -> bool =
 fun shape c a ->
  match (axis_of shape a, data c) with
  | None, _ | _, None -> false
  | Some a, Some d -> (
      let rank = Array.length shape in
      let along s =
        let off = rank - Array.length s in
        a >= off && s.(a - off) > 1
      in
      match d.lift with
      | Num { x; _ } -> along (Nx.shape x)
      | Cat { codes; _ } -> along (Nx.shape codes)
      | Strings s -> along [| Array.length s |]
      | Floats s -> along [| Array.length s |]
      | Index k | Dim { axis = k; _ } ->
          axis_of shape k = Some a && shape.(a) > 1)
