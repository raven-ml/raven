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
  spans : (int * (Hugin_kit.Time.t * Hugin_kit.Time.t)) list;
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
  let rows = Nx.to_array (Nx.take ~axis:0 groups.first keys) in
  let codes = ref [] in
  for i = 0 to (Array.length rows / 2) - 1 do
    if rows.(2 * i) = 1L then codes := Int64.to_int rows.((2 * i) + 1) :: !codes
  done;
  List.sort_uniq Int.compare !codes

(* [key_values input] is the values that tell the columns of a stack apart. *)
let key_values (In i) =
  match i.lift with
  | Quantities q -> Stacked.Floats q.values
  | Instants t -> Stacked.Ints t.counts
  | Categories c -> Stacked.Ints c.codes

(* [intervals shape stack keep inputs] is the index of the stacked input and
   its starts and ends, [nan] where [keep] does not hold. *)
let intervals shape (stack : Stacked.t) keep inputs =
  let find k = List.find_opt (fun (In i) -> i.index = k) inputs in
  match find stack.stacked with
  | Some (In { lift = Quantities q; _ }) ->
      let full t = Nx.broadcast_to shape t in
      let lengths =
        Nx.where keep (full q.values) (Nx.full Nx.float64 shape Float.nan)
      in
      let keys = List.filter_map find stack.keys |> List.map key_values in
      let starts, ends =
        Stacked.intervals stack.offset shape ~lengths ~keys
      in
      let t = Nx.create Nx.float64 shape in
      Some (stack.stacked, t starts, t ends)
  | Some (In { lift = Instants _ | Categories _; _ }) | None -> None

let summarise ?stack shape inputs filter =
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
  let stacked =
    if numel = 0 then None
    else Option.bind stack (fun st -> intervals shape st keep inputs)
  in
  let keep =
    match filter with None -> keep | Some f -> Nx.logical_and keep (full f)
  in
  let kept input =
    match rows input with
    | None -> keep
    | Some m -> Nx.logical_and keep (Nx.logical_not (full m))
  in
  (* The least and greatest of [v] where [k] holds. *)
  let hull k v =
    let pick fill = Nx.where k v (Nx.full_like v fill) in
    (Nx.min (pick Float.infinity), Nx.max (pick Float.neg_infinity))
  in
  let hulled =
    if numel = 0 then []
    else
      List.filter_map
        (fun (In i as input) ->
          match i.lift with
          | Quantities q when i.fitted ->
              let k = kept input in
              let lo, hi =
                match stacked with
                | Some (j, s, e) when j = i.index ->
                    (* Piles past [max_float] have no finite hull. *)
                    let k =
                      Nx.logical_and k
                        (Nx.logical_and (Nx.isfinite s) (Nx.isfinite e))
                    in
                    let lo, hi = hull k s and lo', hi' = hull k e in
                    (Nx.minimum lo lo', Nx.maximum hi hi')
                | _ -> hull k (full q.values)
              in
              Some (i.index, lo, hi)
          | Quantities _ | Instants _ | Categories _ -> None)
        inputs
  in
  (* The earliest and latest kept counts, read alone since int64 counts do not
     fit floats. *)
  let spanned =
    if numel = 0 then []
    else
      List.filter_map
        (fun (In i as input) ->
          match i.lift with
          | Instants t when i.fitted ->
              let k = kept input and c = full t.counts in
              let pick fill = Nx.where k c (Nx.full_like c fill) in
              Some
                ( i.index,
                  t.resolution,
                  Nx.min (pick Int64.max_int),
                  Nx.max (pick Int64.min_int) )
          | Quantities _ | Instants _ | Categories _ -> None)
        inputs
  in
  let ints =
    match List.concat_map (fun (_, _, lo, hi) -> [ lo; hi ]) spanned with
    | [] -> [||]
    | l -> Nx.to_array (Nx.stack l)
  in
  let spans =
    List.mapi
      (fun k (index, r, _, _) -> (index, r, ints.(2 * k), ints.((2 * k) + 1)))
      spanned
    |> List.filter_map (fun (index, r, lo, hi) ->
        if Int64.compare lo hi > 0 then None
        else Some (index, Hugin_kit.Time.(v r lo, v r hi)))
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
          | Quantities _ | Instants _ | Categories _ -> None)
        inputs
  in
  { hulls; spans; codes; notes }
