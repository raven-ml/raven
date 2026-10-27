(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx.Dtype
module L = Nx_array.Layout
module Ptree = Nx.Ptree

let report = Compile.report

(* Keys *)

type place = Every | At : 'd Nx.Placement.t -> place

(* What a program depends on of a leaf. *)
type leaf = {
  dtype : D.any;
  layout : L.t;  (** Over its run ({!Compile.within}). *)
  at : place;
  donated : bool;
  traced : bool;
}

let leaf (Nx.P x) =
  let f = Nx.Prim.form x in
  {
    dtype = D.Any f.dtype;
    layout = Compile.within f.dtype (Compile.layout (Program.F f));
    at = (match f.placement with None -> Every | Some p -> At p);
    donated = Nx.Prim.donated x;
    traced = Option.is_some (Nx.Prim.owner x);
  }

let same_place a b =
  match (a, b) with
  | Every, Every -> true
  | At p, At q -> Nx.Placement.equal p q
  | Every, At _ | At _, Every -> false

let same_leaf a b =
  (match (a.dtype, b.dtype) with D.Any x, D.Any y -> D.equal x y)
  && L.equal a.layout b.layout && same_place a.at b.at
  && Bool.equal a.donated b.donated
  && Bool.equal a.traced b.traced

(* The settings a caller may change around a call that change the program:
   each setting that shapes what tolk compiles, and whether the engine links
   each kernel as a submission of its own, as it does for its reports. *)
type settings = { shaping : (string * string) list; separate : bool }

let settings () =
  {
    shaping = Tolk.Setting.shaping ();
    separate = Tolk.Setting.value Tolk.Setting.debug >= 2;
  }

type key = {
  skeleton : Ptree.Skeleton.t;
  settings : settings;
  leaves : leaf array;
}

let same_key k k' =
  Ptree.Skeleton.equal k.skeleton k'.skeleton
  && k.settings = k'.settings
  && Array.length k.leaves = Array.length k'.leaves
  && Array.for_all2 same_leaf k.leaves k'.leaves

let hash k =
  Array.fold_left
    (fun h l -> (h * 31) + L.hash l.layout)
    (Ptree.Skeleton.hash k.skeleton)
    k.leaves

(* A leaf's parts, in the order a retrace names the first that differs. *)
let parts l =
  let (D.Any dt) = l.dtype in
  [
    D.name dt;
    Format.asprintf "a layout %a over its run" L.pp l.layout;
    (match l.at with
    | Every -> "of every set"
    | At p -> Format.asprintf "at %a" Nx.Placement.pp p);
    (if l.donated then "donated" else "not donated");
    (if l.traced then "traced" else "not traced");
  ]

(* The first difference between the key [k] and the previous key [k']. *)
let difference paths k k' =
  match
    Ptree.Skeleton.diff ~this:"here" k.skeleton ~that:"in the previous key"
      k'.skeleton
  with
  | Some m -> m
  | None when k.settings <> k'.settings -> (
      let entries s = s.shaping @ [ ("separate", string_of_bool s.separate) ] in
      let here = entries k.settings and previous = entries k'.settings in
      let value name e =
        Option.value (List.assoc_opt name e) ~default:"unset"
      in
      match
        List.find_opt
          (fun (name, _) -> value name here <> value name previous)
          (here @ previous)
      with
      | Some (name, _) ->
          Printf.sprintf "%s=%s here, %s=%s in the previous key" name
            (value name here) name (value name previous)
      | None -> "other settings than in the previous key")
  | None ->
      let rec first i paths =
        if i >= Array.length k.leaves then "an equal key"
        else
          let l = k.leaves.(i) and l' = k'.leaves.(i) in
          let path, rest =
            match paths with p :: rest -> (p, rest) | [] -> ("a leaf", [])
          in
          if same_leaf l l' then first (i + 1) rest
          else
            match
              List.find_opt
                (fun (a, b) -> not (String.equal a b))
                (List.combine (parts l) (parts l'))
            with
            | Some (a, b) ->
                Printf.sprintf "%s: %s here, %s in the previous key" path a b
            | None -> path ^ ": another placement"
      in
      first 0 paths

(* Programs *)

module Programs = Cache.Make (struct
  type t = key

  let equal = same_key
  let hash = hash
end)

type 'y entry = {
  body : Program.t;
  like : 'y;
  compiled : (Program.value array -> Program.value array) option;
}

type 'y t = {
  name : string;
  table : 'y entry Programs.t;
  last : key option Atomic.t;
}

let create ~name () = { name; table = Programs.create (); last = Atomic.make None }

let find t leaves skeleton ~paths trace =
  let forms =
    Array.of_list (List.map (fun (Nx.P x) -> Program.F (Nx.Prim.form x)) leaves)
  in
  let leaves = Array.of_list (List.map leaf leaves) in
  let key = { skeleton; settings = settings (); leaves } in
  let miss () =
    Option.iter
      (fun k -> report "retrace: %s" (difference (paths ()) key k))
      (Atomic.get t.last)
  in
  let make () =
    let body, like = trace () in
    let compiled =
      if Array.exists (fun l -> l.traced) leaves then None
      else
        match Compile.make ~name:t.name body forms with
        | Ok c -> Some (Compile.run c)
        | Error why ->
            report "evaluates its program uncompiled: %s" why;
            None
    in
    { body; like; compiled }
  in
  let e = Programs.find t.table key ~miss make in
  Atomic.set t.last (Some key);
  e
