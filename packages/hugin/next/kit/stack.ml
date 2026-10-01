(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type offset = [ `Zero | `Expand | `Center | `Wiggle ]

type order =
  [ `Given | `Reverse | `Ascending | `Descending | `Appearance | `Inside_out ]

(* [sorted n cmp] is the rows [0] to [n - 1] sorted by [cmp], ties in row
   order. *)
let sorted n cmp =
  let rows = Array.init n Fun.id in
  Array.stable_sort cmp rows;
  rows

(* [groups key rows] is the bounds of the runs of [rows] with equal [key]: group
   [g] is [rows.(b.(g))] to [rows.(b.(g + 1) - 1)]. *)
let groups (key : int -> int) rows =
  let n = Array.length rows in
  let b = ref [ n ] in
  for i = n - 1 downto 0 do
    if i = 0 || key rows.(i) <> key rows.(i - 1) then b := i :: !b
  done;
  Array.of_list !b

(* [cells key lengths rows i j] is the sums of the lengths that are not missing
   among [rows.(i)] to [rows.(j - 1)], one per run of rows with equal [key], as
   [(key, sum)] in row order. *)
let cells (key : int -> int) lengths rows i j =
  let acc = ref [] in
  for x = i to j - 1 do
    let r = rows.(x) in
    let l = lengths.(r) in
    if Float.is_finite l then
      match !acc with
      | (k, v) :: rest when k = key r -> acc := (k, v +. l) :: rest
      | cs -> acc := (key r, l) :: cs
  done;
  List.rev !acc

(* Orders *)

(* [peak ncols cells] is the first column in which a series whose lengths sum to
   [v] in column [c], for each [(c, v)] of [cells] in increasing column order,
   is greatest, the other columns of [0] to [ncols - 1] holding [0.]. *)
let peak ncols cells =
  let best_v = ref neg_infinity and best_c = ref 0 in
  let next = ref 0 and absent = ref (-1) in
  List.iter
    (fun (c, v) ->
      if !absent < 0 && c > !next then absent := !next;
      next := c + 1;
      if v > !best_v then begin
        best_v := v;
        best_c := c
      end)
    cells;
  if !absent < 0 && !next < ncols then absent := !next;
  if !absent >= 0 && (0. > !best_v || (0. = !best_v && !absent < !best_c)) then
    !absent
  else !best_c

(* [ranks order ~columns ~series lengths] is, for each row, the rank of its
   series in [order]: series are laid by increasing rank. *)
let ranks (order : order) ~columns ~series lengths =
  match order with
  | `Given -> series
  | `Reverse -> Array.map (fun s -> -s) series
  | (`Ascending | `Descending | `Appearance | `Inside_out) as order ->
      let n = Array.length lengths in
      let ncols = Array.fold_left (fun m c -> Int.max m (c + 1)) 0 columns in
      let rows =
        sorted n (fun r r' ->
            let c = Int.compare series.(r) series.(r') in
            if c <> 0 then c else Int.compare columns.(r) columns.(r'))
      in
      let b = groups (fun r -> series.(r)) rows in
      let k = Array.length b - 1 in
      let sums = Array.make k 0. and peaks = Array.make k 0 in
      for g = 0 to k - 1 do
        let cs = cells (fun r -> columns.(r)) lengths rows b.(g) b.(g + 1) in
        sums.(g) <- List.fold_left (fun s (_, v) -> s +. v) 0. cs;
        peaks.(g) <- peak ncols cs
      done;
      let by = sorted k in
      let laid =
        match order with
        | `Ascending -> by (fun g g' -> Float.compare sums.(g) sums.(g'))
        | `Descending -> by (fun g g' -> Float.compare sums.(g') sums.(g))
        | `Appearance -> by (fun g g' -> Int.compare peaks.(g) peaks.(g'))
        | `Inside_out ->
            let lower = ref [] and upper = ref [] in
            let lower_sum = ref 0. and upper_sum = ref 0. in
            Array.iter
              (fun g ->
                if !upper_sum < !lower_sum then begin
                  upper_sum := !upper_sum +. sums.(g);
                  upper := g :: !upper
                end
                else begin
                  lower_sum := !lower_sum +. sums.(g);
                  lower := g :: !lower
                end)
              (by (fun g g' -> Int.compare peaks.(g) peaks.(g')));
            Array.of_list (!lower @ List.rev !upper)
      in
      let rank_of_group = Array.make k 0 in
      Array.iteri (fun rank g -> rank_of_group.(g) <- rank) laid;
      let ranks = Array.make n 0 in
      for g = 0 to k - 1 do
        for i = b.(g) to b.(g + 1) - 1 do
          ranks.(rows.(i)) <- rank_of_group.(g)
        done
      done;
      ranks

(* Stacks *)

let check_numbers what a =
  Array.iter
    (fun v ->
      if v < 0 then
        invalid_arg (Printf.sprintf "Stack.intervals: negative %s %d" what v))
    a

(* [wiggle_step prev cur] is [(a, b)] of the [`Wiggle] recurrence from a column
   whose series sums are [prev] to one whose sums are [cur], each a list of
   [(rank, sum)] in increasing rank. *)
let wiggle_step prev cur =
  let rec go a b d prev cur =
    let step fp fc =
      let di = fc -. fp and w = Float.abs fc in
      (a +. (w *. (d +. (di /. 2.))), b +. w, d +. di)
    in
    match (prev, cur) with
    | [], [] -> (a, b)
    | (_, fp) :: prev, [] ->
        let a, b, d = step fp 0. in
        go a b d prev []
    | [], (_, fc) :: cur ->
        let a, b, d = step 0. fc in
        go a b d [] cur
    | (rp, fp) :: prev', (rc, fc) :: cur' ->
        if rp < rc then
          let a, b, d = step fp 0. in
          go a b d prev' cur
        else if rc < rp then
          let a, b, d = step 0. fc in
          go a b d prev cur'
        else
          let a, b, d = step fp fc in
          go a b d prev' cur'
  in
  go 0. 0. 0. prev cur

let intervals ?(offset = `Zero) ?(order = `Given) ~columns ~series lengths =
  let n = Array.length lengths in
  if Array.length columns <> n || Array.length series <> n then
    invalid_arg
      (Printf.sprintf "Stack.intervals: %d columns, %d series and %d lengths"
         (Array.length columns) (Array.length series) n);
  check_numbers "column" columns;
  check_numbers "series" series;
  let missing r = not (Float.is_finite lengths.(r)) in
  let rank = ranks order ~columns ~series lengths in
  let rows =
    sorted n (fun r r' ->
        let c = Int.compare columns.(r) columns.(r') in
        if c <> 0 then c else Int.compare rank.(r) rank.(r'))
  in
  let b = groups (fun r -> columns.(r)) rows in
  let k = Array.length b - 1 in
  let starts = Array.make n nan and ends = Array.make n nan in
  (* Each column's extent [lo, hi], and whether it holds a length. *)
  let lo = Array.make k 0. and hi = Array.make k 0. in
  let held = Array.make k false in
  for g = 0 to k - 1 do
    for i = b.(g) to b.(g + 1) - 1 do
      let r = rows.(i) in
      if not (missing r) then begin
        let l = lengths.(r) in
        held.(g) <- true;
        if l < 0. then begin
          starts.(r) <- lo.(g);
          lo.(g) <- lo.(g) +. l;
          ends.(r) <- lo.(g)
        end
        else begin
          starts.(r) <- hi.(g);
          hi.(g) <- hi.(g) +. l;
          ends.(r) <- hi.(g)
        end
      end
    done
  done;
  let move g f =
    for i = b.(g) to b.(g + 1) - 1 do
      let r = rows.(i) in
      starts.(r) <- f starts.(r);
      ends.(r) <- f ends.(r)
    done
  in
  (match offset with
  | `Zero -> ()
  | `Expand ->
      for g = 0 to k - 1 do
        let d = hi.(g) -. lo.(g) in
        if d <> 0. then move g (fun v -> (v -. lo.(g)) /. d)
      done
  | `Center ->
      for g = 0 to k - 1 do
        let m = (lo.(g) +. hi.(g)) /. 2. in
        move g (fun v -> v -. m)
      done
  | `Wiggle ->
      let sums g = cells (fun r -> rank.(r)) lengths rows b.(g) b.(g + 1) in
      let gs = Array.make k 0. in
      let g_prev = ref 0. and prev = ref [] and c_prev = ref (-2) in
      for g = 0 to k - 1 do
        let c = columns.(rows.(b.(g))) and cur = sums g in
        let before = if !c_prev = c - 1 then !prev else [] in
        let a, w = wiggle_step before cur in
        gs.(g) <- (if c = 0 || w = 0. then !g_prev else !g_prev -. (a /. w));
        g_prev := gs.(g);
        prev := cur;
        c_prev := c
      done;
      let least = ref infinity and greatest = ref neg_infinity in
      for g = 0 to k - 1 do
        if held.(g) then begin
          least := Float.min !least (lo.(g) +. gs.(g));
          greatest := Float.max !greatest (hi.(g) +. gs.(g))
        end
      done;
      let centre = -.(!least +. !greatest) /. 2. in
      for g = 0 to k - 1 do
        let shift = gs.(g) +. centre in
        move g (fun v -> v +. shift)
      done);
  (starts, ends)
