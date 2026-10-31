(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Stack = Hugin_kit.Stack
open Figure

type t = { stacked : int; offset : Stack.offset; keys : int list }

let of_mark (m : mark) =
  let indexed = List.mapi (fun i b -> (i, b)) m.bindings in
  let index_of name =
    List.find_map
      (fun (i, B b) ->
        if String.equal b.role.name name && Option.is_some (Channel.data b.ch)
        then Some i
        else None)
      indexed
  in
  List.find_map
    (fun (stacked, B b) ->
      Option.map
        (fun offset ->
          let other =
            if String.equal b.role.name Role.x.name then Role.y.name
            else Role.x.name
          in
          let keys =
            List.filter_map index_of [ other; Role.fx.name; Role.fy.name ]
          in
          { stacked; offset; keys })
        b.stack)
    indexed

(* [0. +. k] is [0.] for [-0.], so equal keys have equal bits. *)
let bits k = Int64.bits_of_float (0. +. k)

let intervals offset shape ~lengths ~keys =
  let full t = Nx.to_array (Nx.broadcast_to shape t) in
  let lengths = full lengths and keys = List.map full keys in
  let n = Array.length lengths in
  let ids = Hashtbl.create 64 in
  let columns =
    Array.init n (fun r ->
        let key = List.map (fun k -> bits k.(r)) keys in
        match Hashtbl.find_opt ids key with
        | Some c -> c
        | None ->
            let c = Hashtbl.length ids in
            Hashtbl.add ids key c;
            c)
  in
  Stack.intervals ~offset ~columns ~series:(Array.make n 0) lengths
