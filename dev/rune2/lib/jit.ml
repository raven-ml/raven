(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx.Dtype
module L = Nx_array.Layout
module Ptree = Nx.Ptree

let report = Compile.report

(* Keys *)

type place = At : 'd Nx.Placement.t -> place

(* What a trace depends on of a leaf: its abstract value. *)
type leaf = { dtype : D.any; shape : int array; at : place }

let leaf (Nx.P x) =
  let f = Nx.Prim.form x in
  {
    dtype = D.Any f.dtype;
    shape = Nx.shape x;
    at = At f.placement;
  }

let same_place (At p) (At q) = Nx.Placement.equal p q

let same_leaf a b =
  (match (a.dtype, b.dtype) with D.Any x, D.Any y -> D.equal x y)
  && a.shape = b.shape && same_place a.at b.at

(* A key's lanes are those of each map its program names, where they differ
   from the ones it was first traced at: a [Lanes] result's shape holds
   them. *)
type key = {
  skeleton : Ptree.Skeleton.t;
  leaves : leaf array;
  lanes : int list option;
}

let same_key k k' =
  Ptree.Skeleton.equal k.skeleton k'.skeleton
  && Array.length k.leaves = Array.length k'.leaves
  && Array.for_all2 same_leaf k.leaves k'.leaves
  && k.lanes = k'.lanes

let hash k =
  Array.fold_left
    (fun h l -> (h * 31) + Hashtbl.hash l.shape)
    (Ptree.Skeleton.hash k.skeleton + Hashtbl.hash k.lanes)
    k.leaves

(* A leaf's parts, in the order a retrace names the first that differs. *)
let parts l =
  let (D.Any dt) = l.dtype in
  [
    D.name dt;
    Format.asprintf "shape %a" Nx.pp_shape l.shape;
    (let (At p) = l.at in
     Format.asprintf "at %a" Nx.Placement.pp p);
  ]

(* The first difference between the key [k] and the previous key [k']. *)
let difference paths k k' =
  match
    Ptree.Skeleton.diff ~this:"here" k.skeleton ~that:"in the previous key"
      k'.skeleton
  with
  | Some m -> m
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
  site : Program.site;
  lanes : int list;
}

(* The lanes of each map of [axes] around the call, [-1] for none. *)
let lanes axes =
  List.map
    (fun a -> match Program.bound a with Some n -> n | None -> -1)
    axes

type 'y t = {
  name : string;
  table : 'y entry Programs.t;
  last : key option Atomic.t;
}

let create ~name () =
  { name; table = Programs.create (); last = Atomic.make None }

(* The stand-in of a leaf: its abstract value, in C order. *)
let stand_in (Nx.P x) =
  let f = Nx.Prim.form x in
  Program.F { f with layout = L.contiguous (L.shape f.layout) }

let stand_ins leaves = Array.of_list (List.map stand_in leaves)

(* [compiled name body forms] runs [body] compiled, made the first time it
   runs, or evaluated where the compiler declines it. *)
let compiled name body forms =
  let made = Atomic.make None and lock = Mutex.create () in
  let get () =
    match Atomic.get made with
    | Some c -> c
    | None -> (
        Mutex.protect lock @@ fun () ->
        match Atomic.get made with
        | Some c -> c
        | None ->
            let c = Compile.make ~name body forms in
            Result.iter_error (report "evaluates its program uncompiled: %s") c;
            Atomic.set made (Some c);
            c)
  in
  fun args ->
    match get () with
    | Ok c -> Compile.run c args
    | Error _ -> Compile.evaluate ~name body args

let find t leaves skeleton ~paths trace =
  let key =
    { skeleton; leaves = Array.of_list (List.map leaf leaves); lanes = None }
  in
  let miss () =
    Option.iter
      (fun k -> report "retrace: %s" (difference (paths ()) key k))
      (Atomic.get t.last)
  in
  let forms = stand_ins leaves in
  let make () =
    let body, like = trace forms in
    let compile = compiled t.name in
    let site = Program.site ~name:t.name ~compile body forms in
    { body; like; site; lanes = lanes (Program.axes site) }
  in
  let e = Programs.find t.table key ~miss make in
  let now = lanes (Program.axes e.site) in
  let e =
    if now = e.lanes then e
    else
      let key = { key with lanes = Some now } in
      let miss () = report "retrace: a map's lanes differ" in
      Programs.find t.table key ~miss make
  in
  Atomic.set t.last (Some key);
  e
