(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module L = Nx_array.Layout

type operand = A | B | Init | Dst
type axis = Batch | Row | Column | Contracted

(* Extents by axis, then offsets by operand, then strides by operand and
   axis: A 0, B 1, Init 2, Dst 3; Batch 0, Row 1, Column 2, Contracted 3. *)
type t = { extents : int array; offsets : int array; strides : int array }

let operand_index = function A -> 0 | B -> 1 | Init -> 2 | Dst -> 3
let axis_index = function Batch -> 0 | Row -> 1 | Column -> 2 | Contracted -> 3
let extent v x = v.extents.(axis_index x)
let offset v o = v.offsets.(operand_index o)
let stride v o x = v.strides.((4 * operand_index o) + axis_index x)
let shape x = Iarray.to_array (L.shape (Nx_array.layout x))

(* Groups axes [axes.(o)] of the layouts [ls.(o)], one list of one count
   per operand, into one run, written as group [g] of [w] for the operands
   [os]: [false] where they make several runs. Axes of extent 1 drop out;
   an axis merges into the run before it where every operand steps along
   the run by the axis's stride times its extent. *)
let group w g os (ls : L.t array) (axes : int array array) =
  let count = Array.length axes.(0) and n = Array.length os in
  let extent = ref 1 and strides = Array.make n 0 and runs = ref 0 in
  for k = 0 to count - 1 do
    let e = L.dim ls.(0) axes.(0).(k) in
    if e = 0 then extent := 0
    else if e > 1 then begin
      let merges = ref (!runs > 0) in
      for p = 0 to n - 1 do
        let s = L.stride ls.(p) axes.(p).(k) in
        if strides.(p) <> s * e then merges := false
      done;
      for p = 0 to n - 1 do
        strides.(p) <- L.stride ls.(p) axes.(p).(k)
      done;
      if not !merges then incr runs;
      extent := !extent * e
    end
  done;
  w.extents.(g) <- !extent;
  for p = 0 to n - 1 do
    let s = if !extent > 1 then strides.(p) else 0 in
    w.strides.((4 * operand_index os.(p)) + g) <- s
  done;
  !runs <= 1 || !extent = 0

let v ~batch ~contracting ~dst ~init a b =
  let fits =
    match
      Nx_kernel.Shape.contract ~batch ~contracting ~segments:None
        (shape a, None) (shape b, None)
    with
    | Error _ -> false
    | Ok y ->
        shape dst = y
        && match init with None -> true | Some i -> shape i = y
  in
  if not fits then Error Nx_array.Shape_mismatch
  else
    let la = Nx_array.layout a and lb = Nx_array.layout b in
    let ly = Nx_array.layout dst in
    let li = Option.map Nx_array.layout init in
    let nb = Iarray.length batch in
    let side f pairs = Array.map f (Iarray.to_array pairs) in
    let free rank used =
      List.filter (fun ax -> not (Array.mem ax used)) (List.init rank Fun.id)
      |> Array.of_list
    in
    let ab = side fst batch and bb = side snd batch in
    let ac = side fst contracting and bc = side snd contracting in
    let rows = free (L.rank la) (Array.append ab ac) in
    let columns = free (L.rank lb) (Array.append bb bc) in
    let fa = Array.length rows and fb = Array.length columns in
    let yb = Array.init nb Fun.id in
    let yr = Array.init fa (fun k -> nb + k) in
    let yc = Array.init fb (fun k -> nb + fa + k) in
    let w =
      {
        extents = Array.make 4 1;
        offsets = Array.make 4 0;
        strides = Array.make 16 0;
      }
    in
    w.offsets.(0) <- L.offset la;
    w.offsets.(1) <- L.offset lb;
    w.offsets.(3) <- L.offset ly;
    (* The init, where given, steps as the result does in its groups. *)
    let with_init os ls axes y =
      match li with
      | None -> (os, ls, axes)
      | Some l ->
          w.offsets.(2) <- L.offset l;
          (Array.append os [| Init |], Array.append ls [| l |],
           Array.append axes [| y |])
    in
    let os, ls, axes =
      with_init [| A; B; Dst |] [| la; lb; ly |] [| ab; bb; yb |] yb
    in
    let batch_ok = group w 0 os ls axes in
    let os, ls, axes = with_init [| A; Dst |] [| la; ly |] [| rows; yr |] yr in
    let rows_ok = group w 1 os ls axes in
    let os, ls, axes =
      with_init [| B; Dst |] [| lb; ly |] [| columns; yc |] yc
    in
    let columns_ok = group w 2 os ls axes in
    let k_ok = group w 3 [| A; B |] [| la; lb |] [| ac; bc |] in
    if batch_ok && rows_ok && columns_ok && k_ok then Ok w
    else Error Nx_array.Declined
