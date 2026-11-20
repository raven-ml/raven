(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The objects a command reads: the rows of the first source with a partner in
   every other and a present input, in batches of the command's size. *)

open Aion

let fail fmt = Printf.ksprintf failwith fmt

(* The modality of [key], which the option [flag] gave. *)
let modality flag key =
  match Modality.of_key key with
  | Some m -> m
  | None -> fail "%s %s: names no modality" flag key

(* Sources *)

(* A path to an existing file is the file, whatever it holds; otherwise a last
   [:/] starts the group. *)
let source (s, v) =
  if Sys.file_exists v then Survey.load s v
  else
    match String.rindex_opt v ':' with
    | Some k when k + 1 < String.length v && v.[k + 1] = '/' ->
        Survey.load
          ~group:(String.sub v (k + 1) (String.length v - k - 1))
          s (String.sub v 0 k)
    | _ -> Survey.load s v

let sources = function
  | [] ->
      fail
        "no source: give one of --legacysurvey, --hsc, --desi, --sdss or --gaia"
  | l -> List.map source l

(* Each input with the index of the first source that holds it: those [keys]
   name, or every modality the sources hold and the model [reads] but
   [except]. *)
let inputs srcs ~reads ~except keys =
  let held m =
    List.find_index (fun s -> List.mem m (Survey.modalities s)) srcs
  in
  let ins =
    match keys with
    | "" ->
        List.filter_map
          (fun m ->
            if List.mem m except || not (List.mem m reads) then None
            else Option.map (fun i -> (i, m)) (held m))
          Modality.all
    | keys ->
        List.map
          (fun k ->
            let m = modality "--inputs" k in
            if List.mem m except then
              fail "--inputs %s: the target is no input" k;
            if not (List.mem m reads) then
              fail "--inputs %s: the model reads no %s" k k;
            match held m with
            | Some i -> (i, m)
            | None -> fail "--inputs %s: no source holds it" k)
          (String.split_on_char ',' keys)
  in
  if ins = [] then
    fail "no input: the sources hold no modality the model reads";
  ins

(* [A:B], the first source's rows from A to B, excluded. *)
let range first spec =
  let n = Survey.length first in
  match spec with
  | "" -> (0, n)
  | _ -> (
      match String.split_on_char ':' spec with
      | [ a; b ] -> (
          match (int_of_string_opt a, int_of_string_opt b) with
          | Some a, Some b when 0 <= a && a < b && b <= n -> (a, b)
          | _ -> fail "--rows %s: the first source has rows 0:%d" spec n)
      | _ -> fail "--rows %s: A:B expected" spec)

(* Objects *)

type t = {
  sources : Survey.t list;
  inputs : (int * Modality.t) list;
  range : int * int;
  matched : int;  (* The rows of [range] with a partner in every source. *)
  rows : int array array;  (* [rows.(o).(s)], object [o]'s row in source [s]. *)
}

(* Each row of [first] in [a, b) with a partner in every other source. *)
let matched srcs (a, b) =
  let first = List.hd srcs in
  let partners =
    List.map
      (fun t ->
        let i, j = Survey.pair first t in
        let m = Hashtbl.create (Array.length i) in
        Array.iteri (fun k r -> Hashtbl.replace m r j.(k)) i;
        m)
      (List.tl srcs)
  in
  List.init (b - a) (fun k -> a + k)
  |> List.filter_map (fun r ->
      let js = List.map (fun m -> Hashtbl.find_opt m r) partners in
      if List.mem None js then None
      else Some (Array.of_list (r :: List.map Option.get js)))
  |> Array.of_list

(* Each input's data of the objects [objs]. *)
let data srcs ins objs =
  List.map
    (fun (s, m) ->
      Survey.read (List.nth srcs s) (Array.map (fun o -> o.(s)) objs) m)
    ins

(* The objects of [objs] with a present input. Only a scalar is ever absent
   (Codecs.encode), so with an image or a spectrum among the inputs every
   object is kept; otherwise the scalars are read once and encoded on the host
   by [codecs]. *)
let present codecs srcs ins objs =
  let scalar (_, m) = match m with Modality.Scalar _ -> true | _ -> false in
  if objs = [||] || not (List.for_all scalar ins) then objs
  else
    let kept = Array.make (Array.length objs) false in
    List.iter
      (fun x ->
        let x = Codecs.encode codecs x in
        Nx.to_array (Nx.any ~axes:[ 1 ] x.present)
        |> Array.iteri (fun o p -> if p then kept.(o) <- true))
      (data srcs ins objs);
    Array.to_list objs |> List.filteri (fun o _ -> kept.(o)) |> Array.of_list

let v srcs ins range codecs =
  let objs = matched srcs range in
  let rows = present codecs srcs ins objs in
  { sources = srcs; inputs = ins; range; matched = Array.length objs; rows }

let length t = Array.length t.rows

(* [t]'s objects in batches of [size], each with the index of its first
   object: all full but the last. *)
let batches t size =
  let n = length t in
  List.init
    ((n + size - 1) / size)
    (fun k ->
      let first = k * size in
      (first, Array.sub t.rows first (min size (n - first))))

let read t objs = data t.sources t.inputs objs

let summary name t ~batches =
  let a, b = t.range in
  Printf.printf
    "%s: %d objects in %d batches; of rows %d:%d, %d without a partner in \
     every source, %d without a present input\n\
     %!"
    name (length t) batches a b (b - a - t.matched) (t.matched - length t)
