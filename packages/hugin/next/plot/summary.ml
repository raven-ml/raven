(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Scale = Hugin_next_kit.Scale
open Channel

let explicit_domain s =
  if Scale.sets Domain s then Some (Scale.domain s) else None

type input =
  | In : {
      index : int;
      role : string;
      colour : bool; (* A missing value keeps its row. *)
      lift : 'd lift;
      spec : 'd Scale.t; (* Finds the missing values. *)
      fitted : bool; (* Its hull or kept codes are summarised. *)
    }
      -> input

type summary = {
  hulls : (int * (float * float)) list;
      (* Per binding index, when some value is kept. *)
  codes : (int * int list) list; (* Per binding index, increasing. *)
  notes : string list; (* Problems with the data, for warnings. *)
}

type probe = {
  miss : Nx.bool_t option; (* Broadcasts to the mark's shape. *)
  values : Nx.float64_t option;
  icodes : Nx.int64_t option;
  counts : (Nx.float64_t * (int -> string)) list;
}

let count m = Nx.sum (Nx.cast Nx.float64 m)

(* [counted noun ppf k] formats [k noun] with its verb, such as [1 code is]. *)
let counted noun ppf k =
  if k = 1 then Format.fprintf ppf "1 %s is" noun
  else Format.fprintf ppf "%d %ss are" k noun

let ( ||| ) m m' =
  match (m, m') with
  | None, m | m, None -> m
  | Some a, Some b -> Some (Nx.logical_or a b)

let invalid valid = Option.map Nx.logical_not valid

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

(* [absent ints c] is [true] where the code [c] is none of the increasing
   [ints]. *)
let absent ints c =
  let k = Array.length ints in
  if k = 0 then Nx.full_like (Nx.cast Nx.bool c) true
  else
    let d = Nx.create Nx.int64 [| k |] (Array.map Int64.of_int ints) in
    let pos =
      Nx.clamp ~max:(Int64.of_int (k - 1)) (Nx.searchsorted ~side:`Left d c)
    in
    Nx.not_equal (Nx.take ~indices:pos d) c

let quantities ?valid role spec v =
  let m = Scale.missing spec v in
  let undefined = Nx.logical_and m (Nx.isfinite v) in
  let undefined =
    match valid with
    | None -> undefined
    | Some ok -> Nx.logical_and undefined ok
  in
  let note k =
    Format.asprintf "%s: %a missing for its scale" role (counted "finite value")
      k
  in
  {
    miss = Some m ||| invalid valid;
    values = Some v;
    icodes = None;
    counts = [ (count undefined, note) ];
  }

let probe : type d. int array -> string -> d lift -> d Scale.t -> probe =
 fun shape role lift spec ->
  match lift with
  | Num { x; valid } -> quantities ?valid role spec (Nx.cast Nx.float64 x)
  | Index k ->
      let a = Option.get (axis_of shape k) in
      let n = shape.(a) in
      quantities role spec
        (along shape a (Nx.cast Nx.float64 (Nx.arange Nx.int32 0 n 1)))
  | Scalar x -> quantities role spec (Nx.scalar Nx.float64 x)
  | Cat { codes; valid; labels = Some labels } ->
      let c = Nx.cast Nx.int64 codes in
      let n = Array.length labels in
      let out =
        Nx.logical_or (Nx.less_s c 0L) (Nx.greater_equal_s c (Int64.of_int n))
      in
      let outside =
        match valid with None -> out | Some ok -> Nx.logical_and out ok
      in
      let m =
        match explicit_domain spec with
        | Some (Scale.Categories (Scale.Labels domain)) ->
            let kept l = Array.exists (String.equal l) domain in
            let allowed = Nx.create Nx.bool [| n |] (Array.map kept labels) in
            Nx.logical_or out (Nx.logical_not (Nx.take ~indices:c allowed))
        | _ -> out
      in
      let note k =
        Format.asprintf "%s: %a outside its %d labels" role (counted "code") k n
      in
      {
        miss = Some m ||| invalid valid;
        values = None;
        icodes = None;
        counts = [ (count outside, note) ];
      }
  | Cat { codes; valid; labels = None } ->
      let c = Nx.cast Nx.int64 codes in
      let beyond = beyond_int (Nx.dtype codes) c in
      let m = beyond ||| invalid valid in
      let m =
        match explicit_domain spec with
        | Some (Scale.Categories (Scale.Indices ix)) ->
            m ||| Some (absent (Array.map fst ix) c)
        | _ -> m
      in
      let counts =
        match beyond with
        | None -> []
        | Some b ->
            let b =
              match valid with None -> b | Some ok -> Nx.logical_and b ok
            in
            let note k =
              Format.asprintf "%s: %a beyond the range of int" role
                (counted "code") k
            in
            [ (count b, note) ]
      in
      { miss = m; values = None; icodes = Some c; counts }
  | Strings a ->
      let m =
        match explicit_domain spec with
        | Some (Scale.Categories (Scale.Labels domain)) ->
            let out s = not (Array.exists (String.equal s) domain) in
            Some (Nx.create Nx.bool [| Array.length a |] (Array.map out a))
        | _ -> None
      in
      { miss = m; values = None; icodes = None; counts = [] }
  | Dim { axis; valid; _ } ->
      let a = Option.get (axis_of shape axis) in
      let m =
        match explicit_domain spec with
        | Some (Scale.Categories (Scale.Indices ix)) ->
            let out i = not (Array.exists (fun (j, _) -> j = i) ix) in
            let n = shape.(a) in
            Some (along shape a (Nx.create Nx.bool [| n |] (Array.init n out)))
        | _ -> None
      in
      { miss = m ||| invalid valid; values = None; icodes = None; counts = [] }

(* [kept_codes keep c] is the distinct codes of [c] where [keep] holds, in
   increasing order. *)
let kept_codes keep c =
  let keys =
    Nx.stack ~axis:1 [ Nx.flatten (Nx.cast Nx.int64 keep); Nx.flatten c ]
  in
  let groups = Nx.unique keys in
  let rows = Nx.to_array (Nx.take ~axis:0 ~indices:groups.first keys) in
  let codes = ref [] in
  for i = 0 to (Array.length rows / 2) - 1 do
    if rows.(2 * i) = 1L then codes := Int64.to_int rows.((2 * i) + 1) :: !codes
  done;
  List.sort_uniq Int.compare !codes

let summarise shape inputs filter =
  let probes =
    List.map
      (fun (In i as input) -> (input, probe shape i.role i.lift i.spec))
      inputs
  in
  let numel = Array.fold_left ( * ) 1 shape in
  let full t = Nx.broadcast_to shape t in
  let dropped =
    List.fold_left
      (fun acc (In i, p) -> if i.colour then acc else acc ||| p.miss)
      None probes
  in
  let keep =
    match dropped with
    | None -> Nx.full Nx.bool shape true
    | Some d -> Nx.logical_not (full d)
  in
  let keep =
    match filter with None -> keep | Some f -> Nx.logical_and keep (full f)
  in
  let kept p =
    match p.miss with
    | None -> keep
    | Some m -> Nx.logical_and keep (Nx.logical_not (full m))
  in
  let hulled =
    if numel = 0 then []
    else
      List.filter_map
        (fun (In i, p) ->
          match p.values with
          | Some v when i.fitted ->
              let k = kept p and v = full v in
              let lo = Nx.min (Nx.where k v (Nx.full_like v Float.infinity)) in
              let hi =
                Nx.max (Nx.where k v (Nx.full_like v Float.neg_infinity))
              in
              Some (i.index, lo, hi)
          | _ -> None)
        probes
  in
  let counts = List.concat_map (fun (_, p) -> p.counts) probes in
  let scalars =
    List.concat_map (fun (_, lo, hi) -> [ lo; hi ]) hulled @ List.map fst counts
  in
  let host = match scalars with [] -> [||] | l -> Nx.to_array (Nx.stack l) in
  let hulls =
    List.mapi
      (fun k (index, _, _) -> (index, (host.(2 * k), host.((2 * k) + 1))))
      hulled
    |> List.filter (fun (_, (lo, hi)) -> lo <= hi)
  in
  let off = 2 * List.length hulled in
  let notes =
    List.concat
      (List.mapi
         (fun k (_, note) ->
           let n = int_of_float host.(off + k) in
           if n > 0 then [ note n ] else [])
         counts)
  in
  let codes =
    if numel = 0 then []
    else
      List.filter_map
        (fun (In i, p) ->
          match p.icodes with
          | Some c when i.fitted -> Some (i.index, kept_codes (kept p) (full c))
          | _ -> None)
        probes
  in
  { hulls; codes; notes }
