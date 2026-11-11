(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin
module Float_read = Munin_float_text.Float_read
module Sset = Set.Make (String)

(* Filters *)

type word = Has of string | Below of string * float | Above of string * float
type filter = { text : string; words : word list }

let no_filter = { text = ""; words = [] }

(* [key<v] and [key>v] with a key and a number. *)
let word w =
  let split c =
    match String.index_opt w c with
    | Some i when i > 0 && i < String.length w - 1 ->
        let k = String.sub w 0 i and x = String.sub w (i + 1) (String.length w - i - 1) in
        Option.map (fun x -> (k, x)) (Float_read.of_string x)
    | _ -> None
  in
  match (split '<', split '>') with
  | Some (k, x), _ -> Below (k, x)
  | None, Some (k, x) -> Above (k, x)
  | None, None -> Has w

let filter text =
  let words =
    List.filter (( <> ) "") (String.split_on_char ' ' (String.lowercase_ascii text))
  in
  { text; words = List.map word words }

let text f = f.text

(* [contains ~sub s] is [true] iff the lowercase [sub] is in [s], ignoring
   case. *)
let contains ~sub s =
  let n = String.length sub and m = String.length s in
  let at i =
    let rec go j = j = n || (Char.lowercase_ascii s.[i + j] = sub.[j] && go (j + 1)) in
    go 0
  in
  let rec from i = i + n <= m && (at i || from (i + 1)) in
  from 0

let param r k =
  List.find_map
    (fun (pk, v) -> if String.lowercase_ascii pk = k then Value.to_float v else None)
    (Run.params r)

let matches (f : Facts.run) = function
  | Below (k, x) -> ( match param f.run k with Some v -> v < x | None -> false)
  | Above (k, x) -> ( match param f.run k with Some v -> v > x | None -> false)
  | Has w ->
      let r = f.run in
      contains ~sub:w (Facts.name r)
      || contains ~sub:w (Run.id r)
      || contains ~sub:w (Run.experiment r)
      || contains ~sub:w (Watch.word f.state)
      || List.exists (contains ~sub:w) (Run.tags r)
      || List.exists
           (fun (k, v) -> contains ~sub:w (k ^ "=" ^ Facts.value v))
           (Run.params r)

(* Order *)

type order = Started | Best | Name

let next = function Started -> Best | Best -> Name | Name -> Started
let word = function Started -> "started" | Best -> "best" | Name -> "name"

let experiment (f : Facts.run) = Run.experiment f.run

(* Each experiment's runs together, in the order of its first run. *)
let grouped runs =
  let order = ref [] and members = Hashtbl.create 16 in
  List.iter
    (fun f ->
      let e = experiment f in
      match Hashtbl.find_opt members e with
      | Some l -> Hashtbl.replace members e (f :: l)
      | None ->
          order := e :: !order;
          Hashtbl.replace members e [ f ])
    runs;
  List.concat_map (fun e -> List.rev (Hashtbl.find members e)) (List.rev !order)

(* A run's best value of its experiment's key metric, as a rank that sorts
   the best first: runs without one last. *)
let rank ~keys (f : Facts.run) =
  match keys (experiment f) with
  | None -> None
  | Some k -> (
      match (Facts.goal f k, Facts.best f k) with
      | Some g, Some b when Float.is_finite b.value ->
          Some (if g = `Maximize then -.b.value else b.value)
      | _ -> None)

let keep f o ~keys runs =
  let kept = List.filter (fun r -> List.for_all (matches r) f.words) runs in
  let sorted =
    match o with
    | Started -> kept
    | Name ->
        List.stable_sort
          (fun (a : Facts.run) (b : Facts.run) ->
            String.compare (Facts.name a.run) (Facts.name b.run))
          kept
    | Best ->
        let ranked = List.map (fun r -> (rank ~keys r, r)) kept in
        List.map snd
          (List.stable_sort
             (fun (a, _) (b, _) ->
               match (a, b) with
               | Some x, Some y -> Float.compare x y
               | Some _, None -> -1
               | None, Some _ -> 1
               | None, None -> 0)
             ranked)
  in
  grouped sorted

(* Folds *)

type folds = Unset | Folded of Sset.t

let crowded = 30

let first_folds runs ~cursor =
  if List.length runs <= crowded then Sset.empty
  else
    let mine =
      List.find_map
        (fun (f : Facts.run) -> if Some (Run.id f.run) = cursor then Some (experiment f) else None)
        runs
    in
    List.fold_left
      (fun s f ->
        let e = experiment f in
        if Some e = mine then s else Sset.add e s)
      Sset.empty runs

(* Lines *)

type line =
  | Header of { experiment : string; runs : Facts.run list; folded : bool }
  | Run of Facts.run

(* [kept] holds each experiment's runs together. *)
let lines kept folded =
  let rec split e acc = function
    | f :: rest when experiment f = e -> split e (f :: acc) rest
    | rest -> (List.rev acc, rest)
  in
  let rec go acc = function
    | [] -> List.rev acc
    | f :: _ as l ->
        let e = experiment f in
        let mine, rest = split e [] l in
        let folded = Sset.mem e folded in
        let header = Header { experiment = e; runs = mine; folded } in
        let runs = if folded then [] else List.rev_map (fun r -> Run r) mine in
        go (runs @ (header :: acc)) rest
  in
  go [] kept

let id (f : Facts.run) = Run.id f.run

let holds target = function
  | Run f -> id f = target
  | Header { folded = true; runs; _ } -> List.exists (fun f -> id f = target) runs
  | Header { folded = false; _ } -> false

let at lines target = List.find_index (holds target) lines

(* The run a stop stands for. *)
let stop = function
  | Run f -> Some (id f)
  | Header { folded = true; runs = f :: _; _ } -> Some (id f)
  | Header _ -> None

let stops lines = List.filter_map (fun l -> Option.map (fun s -> (l, s)) (stop l)) lines
let first lines = Option.map snd (List.nth_opt (stops lines) 0)
let last lines = Option.map snd (List.nth_opt (List.rev (stops lines)) 0)

let step lines target d =
  let s = stops lines in
  match List.find_index (fun (l, _) -> holds target l) s with
  | None -> first lines
  | Some i ->
      let n = List.length s in
      Option.map snd (List.nth_opt s (max 0 (min (n - 1) (i + d))))

let star runs ~key =
  let better g (a : Metric.sample) (b : Metric.sample) =
    match g with `Minimize -> a.value < b.value | `Maximize -> a.value > b.value
  in
  List.fold_left
    (fun acc f ->
      match (Facts.goal f key, Facts.best f key) with
      | Some g, Some b when Float.is_finite b.value -> (
          match acc with
          | Some (_, a) when not (better g b a) -> acc
          | _ -> Some (id f, b))
      | _ -> acc)
    None runs
  |> Option.map fst
