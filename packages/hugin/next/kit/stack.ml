(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type offset = [ `Zero | `Expand | `Center ]

let check_numbers what a =
  Array.iter
    (fun v ->
      if v < 0 then
        invalid_arg (Printf.sprintf "Stack.intervals: negative %s %d" what v))
    a

let intervals ?(offset = `Zero) ~columns ~series lengths =
  let n = Array.length lengths in
  if Array.length columns <> n || Array.length series <> n then
    invalid_arg
      (Printf.sprintf "Stack.intervals: %d columns, %d series and %d lengths"
         (Array.length columns) (Array.length series) n);
  check_numbers "column" columns;
  check_numbers "series" series;
  (* The rows by column, then series, then row: the order they are laid in. *)
  let rows = Array.init n Fun.id in
  Array.stable_sort
    (fun r r' ->
      let c = Int.compare columns.(r) columns.(r') in
      if c <> 0 then c else Int.compare series.(r) series.(r'))
    rows;
  let starts = Array.make n nan and ends = Array.make n nan in
  (* Each column is the rows [rows.(first)] to [rows.(last - 1)]. *)
  let first = ref 0 in
  while !first < n do
    let c = columns.(rows.(!first)) in
    let last = ref !first in
    while !last < n && columns.(rows.(!last)) = c do
      incr last
    done;
    let lo = ref 0. and hi = ref 0. in
    for i = !first to !last - 1 do
      let r = rows.(i) in
      let l = lengths.(r) in
      if Float.is_finite l then
        if l < 0. then begin
          starts.(r) <- !lo;
          lo := !lo +. l;
          ends.(r) <- !lo
        end
        else begin
          starts.(r) <- !hi;
          hi := !hi +. l;
          ends.(r) <- !hi
        end
    done;
    let move f =
      for i = !first to !last - 1 do
        let r = rows.(i) in
        starts.(r) <- f starts.(r);
        ends.(r) <- f ends.(r)
      done
    in
    (match offset with
    | `Zero -> ()
    | `Expand ->
        let lo = !lo and d = !hi -. !lo in
        if d <> 0. then move (fun v -> (v -. lo) /. d)
    | `Center ->
        let m = (!lo +. !hi) /. 2. in
        move (fun v -> v -. m));
    first := !last
  done;
  (starts, ends)
