(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type input =
  | In : {
      index : int;
      keeps_row : bool;
      fitted : bool;
      lift : 'd Lift.t;
    }
      -> input

type t = {
  hulls : (int * (float * float)) list;
  codes : (int * int list) list;
  notes : string list;
}

let count m = Nx.sum (Nx.cast Nx.float64 m)
let rows (In i) = (Lift.miss i.lift).rows

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
  let numel = Array.fold_left ( * ) 1 shape in
  let full t = Nx.broadcast_to shape t in
  let keep =
    List.fold_left
      (fun keep (In i as input) ->
        match rows input with
        | Some m when not i.keeps_row ->
            Nx.logical_and keep (Nx.logical_not (full m))
        | _ -> keep)
      (Nx.full Nx.bool shape true)
      inputs
  in
  let keep =
    match filter with None -> keep | Some f -> Nx.logical_and keep (full f)
  in
  let kept input =
    match rows input with
    | None -> keep
    | Some m -> Nx.logical_and keep (Nx.logical_not (full m))
  in
  let hulled =
    if numel = 0 then []
    else
      List.filter_map
        (fun (In i as input) ->
          match i.lift with
          | Quantities q when i.fitted ->
              let k = kept input and v = full q.values in
              let lo = Nx.min (Nx.where k v (Nx.full_like v Float.infinity)) in
              let hi =
                Nx.max (Nx.where k v (Nx.full_like v Float.neg_infinity))
              in
              Some (i.index, lo, hi)
          | Quantities _ | Categories _ -> None)
        inputs
  in
  let counts =
    List.concat_map (fun (In i) -> (Lift.miss i.lift).counts) inputs
  in
  let scalars =
    List.concat_map (fun (_, lo, hi) -> [ lo; hi ]) hulled
    @ List.map (fun (m, _) -> count (Lazy.force m)) counts
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
        (fun (In i as input) ->
          match i.lift with
          | Categories { lift = Cat { labels = None; _ }; codes; _ }
            when i.fitted ->
              Some (i.index, kept_codes (kept input) (full codes))
          | Quantities _ | Categories _ -> None)
        inputs
  in
  { hulls; codes; notes }
