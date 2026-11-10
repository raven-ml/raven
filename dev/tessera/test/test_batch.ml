(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tessera

(* The counts of one pixel of [n] Sentinel-2 observations. *)
let counts_of n =
  let px = Synth.pixels { Synth.s2 = [ n ]; s1 = [ 0 ]; seed = 0 } in
  fst (Synth.counts (Batch.v2_counts px))

(* Counts *)

(* Rows: [n], then each listed observation's count, as the fill rule gives
   them. *)
let fill_rows =
  [
    (1, [ (0, 8) ]);
    (3, [ (0, 3); (1, 3); (2, 2) ]);
    (* [array_split (arange 5) 3] is [[0; 1]; [2; 3]; [4]]. *)
    (5, [ (0, 1); (1, 2); (2, 1); (3, 2); (4, 2) ]);
    (8, List.init 8 (fun i -> (i, 1)));
    (* [array_split (arange 9) 7]: two runs of 2, then runs of 1. *)
    (9, [ (0, 1); (1, 2); (2, 1); (3, 2); (4, 2); (5, 2); (8, 2) ]);
    (256, List.init 256 (fun i -> (i, 1)));
    (* [linspace 0 256 256]: 254.996… floors to 254, so 255 is left out. *)
    (257, [ (254, 1); (255, 0); (256, 1) ]);
    (* [85 · (294 / 255)] is 97.99999999999999 in float64. *)
    (295, [ (96, 1); (97, 1); (98, 0); (99, 1); (294, 1) ]);
  ]

let fill_rule =
  cases "v2_counts follows the fill rule"
    ~id:(fun (n, _) -> string_of_int n)
    fill_rows
    (fun (n, expected) ->
      let c = counts_of n in
      List.iter
        (fun (i, m) ->
          equal ~msg:(Printf.sprintf "observation %d" i) int m c.(i))
        expected)

let bin n = if n = 0 then 0 else min 256 ((n + 7) / 8 * 8)

let fill_total =
  prop "v2_counts fills each pixel to its bin" (Synth.spec ~long:true ())
    (fun spec ->
      let px = Synth.pixels spec in
      let c2, c1 = Synth.counts (Batch.v2_counts px) in
      let check ns c =
        ignore
          (List.fold_left
             (fun at n ->
               let row = Array.sub c at n in
               cover "n > 256" (n > 256);
               cover "n < bin n" (n > 0 && n < bin n);
               equal ~msg:"total" int (bin n) (Array.fold_left ( + ) 0 row);
               if n <= 256 then Array.iter (fun m -> at_least int ~than:1 m) row;
               at + n)
             0 ns)
      in
      check spec.s2 c2;
      check spec.s1 c1)

(* Packing *)

let shape ?(pixels = 4) ?(rows = 3) ?(slots = 3) ?(width = 256) () =
  { Batch.pixels; s2_rows = rows; s1_rows = rows; slots; width }

(* Every batch of [px], from the first pixel. *)
let batches shape px counts =
  let rec go from acc =
    let b, next = Batch.pack shape px counts ~from in
    let acc = b :: acc in
    if next >= Tessera.Pixels.length px then List.rev acc else go next acc
  in
  go 0 []

let ints x = Array.map Int32.to_int (Nx.to_array x)
let longs x = Array.map Int64.to_int (Nx.to_array x)

(* Checks that each placed pixel's slot holds its tokens of each sensor. *)
let check_rows ~width ~slots (px : Nx.host Pixels.t) counts
    (b : Nx.host Batch.t) =
  let pixel = longs b.pixel in
  let sensor name (r : Nx.host Batch.rows) rg c =
    let tokens = Nx.to_array r.tokens and logc = Nx.to_array r.log_count in
    let start = ints r.start and stop = ints r.stop in
    let sstart = ints r.slot_start and sstop = ints r.slot_stop in
    let at = longs r.at in
    let offsets = longs (Nx_ragged.offsets rg) in
    let values = Nx.to_array (Nx_ragged.values rg) in
    let cells = Nx.dim 1 (Nx_ragged.values rg) in
    let ch = Nx.dim 2 r.tokens in
    Array.iteri
      (fun place p ->
        if p >= 0 then begin
          let s = at.(place) in
          let row = s / slots in
          let lo = sstart.(s) and hi = sstop.(s) in
          let kept =
            List.filter
              (fun j -> c.(j) > 0)
              (List.init
                 (offsets.(p + 1) - offsets.(p))
                 (fun k -> offsets.(p) + k))
          in
          equal ~msg:(name ^ ": tokens") int (max 1 (List.length kept)) (hi - lo);
          for k = lo to hi - 1 do
            let i = (row * width) + k in
            equal ~msg:(name ^ ": start") int lo start.(i);
            equal ~msg:(name ^ ": stop") int hi stop.(i)
          done;
          match kept with
          | [] ->
              let i = (row * width) + lo in
              equal ~msg:(name ^ ": stand-in") float_exact (-1.)
                tokens.((i * ch) + ch - 1);
              equal ~msg:(name ^ ": stand-in day") float_exact 0.
                tokens.((i * ch) + ch - 2)
          | _ ->
              List.iteri
                (fun k j ->
                  let i = (row * width) + lo + k in
                  equal ~msg:(name ^ ": log count") (float 1e-6)
                    (Float.log (Float.of_int c.(j)))
                    logc.(i);
                  for v = 0 to cells - 1 do
                    equal ~msg:(name ^ ": cell") float_exact
                      (Float.of_int values.((j * cells) + v))
                      tokens.((i * ch) + v)
                  done)
                kept
        end)
      pixel;
    (* A key outside every slot is padding, of empty range. *)
    Array.iteri
      (fun i a -> if a = stop.(i) then equal ~msg:(name ^ ": padding") int 0 a)
      start
  in
  let c2, c1 = Synth.counts counts in
  sensor "s2" b.s2 px.s2 c2;
  sensor "s1" b.s1 px.s1 c1

let packing =
  prop "pack places each pixel with tokens once, in order, in its slots"
    Gen.(
      pair
        (Synth.spec ~pixels:9 ~long:true ())
        (pair (int_range 1 4) (int_range 1 3)))
    (fun (spec, (slots, rows)) ->
      let px = Synth.pixels spec in
      let counts = Batch.v2_counts px in
      let width = 256 in
      let bs = batches (shape ~rows ~slots ~width ()) px counts in
      List.iter (check_rows ~width ~slots px counts) bs;
      let placed =
        List.concat_map
          (fun (b : Nx.host Batch.t) ->
            List.filter (fun p -> p >= 0) (Array.to_list (longs b.pixel)))
          bs
      in
      let expected =
        List.filteri
          (fun i _ -> List.nth spec.s2 i > 0 || List.nth spec.s1 i > 0)
          (List.init (List.length spec.s2) Fun.id)
      in
      cover "a pixel without tokens" (List.length expected < List.length spec.s2);
      cover "a stand-in"
        (List.exists2 (fun a b -> a = 0 <> (b = 0)) spec.s2 spec.s1);
      cover "several batches" (List.length bs > 1);
      equal (list int) expected placed)

let refusals =
  let px = Synth.pixels { Synth.s2 = [ 20 ]; s1 = [ 3 ]; seed = 1 } in
  let counts = Batch.v2_counts px in
  group "pack refuses"
    [
      test "a row narrower than a pixel's tokens" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"width") (fun () ->
              Batch.pack (shape ~width:16 ()) px counts ~from:0));
      test "a field that is not positive" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"not positive") (fun () ->
              Batch.pack (shape ~slots:0 ()) px counts ~from:0));
      test "counts of another length" (fun () ->
          let short = { counts with Batch.s2 = Nx.zeros Nx.int32 [| 3 |] } in
          raises_match (Exn.invalid_arg ~sub:"counts.s2") (fun () ->
              Batch.pack (shape ()) px short ~from:0));
      test "sensors of different numbers of pixels" (fun () ->
          let two =
            Synth.pixels { Synth.s2 = [ 1; 2 ]; s1 = [ 3; 4 ]; seed = 2 }
          in
          let uneven = { px with Pixels.s1 = two.s1 } in
          let counts = { counts with Batch.s1 = (Batch.v2_counts two).s1 } in
          raises_match (Exn.invalid_arg ~sub:"Sentinel-1 rows") (fun () ->
              Batch.pack (shape ()) uneven counts ~from:0));
    ]

let () =
  exit
    (run "tessera.batch"
       [
         group "counts" [ fill_rule; fill_total ];
         group "pack" [ packing; refusals ];
       ])
