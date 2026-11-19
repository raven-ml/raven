(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tessera

(* A tile of one row of two pixels, written as the preprocessing writes it, and
   its expected rows. Sentinel-2: three frames; pixel 0 is clear in frames 0 and
   2, pixel 1 never. Sentinel-1: pixel 0 has VV only in its first ascending
   frame and both bands in the descending one; pixel 1 has VH only, then both,
   ascending. *)
let s2_frames = 3
let doys = [| 10; 20; 30 |]
let mask = [| 1; 0; 0; 0; 1; 0 |]
let band t p c = (100 * t) + (10 * p) + c + 1
let asc = [| 5; 0; 0; 7; 0; 0; 3; 3 |] (* [t; p; band] *)
let asc_doys = [| 40; 50 |]
let desc = [| 1; 1; 0; 0 |]
let desc_doys = [| 60 |]

let write ?(skip = []) dir =
  let save name dt shape v =
    if not (List.mem name skip) then
      Nx_io.save_npy (Filename.concat dir name) (Nx.create dt shape v)
  in
  save "bands.npy" Nx.uint16 [| s2_frames; 1; 2; 10 |]
    (Array.init
       (s2_frames * 2 * 10)
       (fun i -> band (i / 20) (i / 10 mod 2) (i mod 10)));
  save "masks.npy" Nx.uint8 [| s2_frames; 1; 2 |] mask;
  save "doys.npy" Nx.uint16 [| s2_frames |] doys;
  save "sar_ascending.npy" Nx.int16 [| 2; 1; 2; 2 |] asc;
  save "sar_ascending_doy.npy" Nx.int16 [| 2 |] asc_doys;
  save "sar_descending.npy" Nx.int16 [| 1; 1; 2; 2 |] desc;
  save "sar_descending_doy.npy" Nx.int16 [| 1 |] desc_doys

let with_tile ?skip f =
  let dir = Filename.temp_dir "tessera" "tile" in
  Fun.protect
    ~finally:(fun () ->
      Array.iter (fun n -> Sys.remove (Filename.concat dir n)) (Sys.readdir dir);
      Sys.rmdir dir)
    (fun () ->
      write ?skip dir;
      f dir)

let rows r =
  let o = Array.map Int64.to_int (Nx.to_array (Nx_ragged.offsets r)) in
  let v = Nx.to_array (Nx_ragged.values r) in
  let c = Nx.dim 1 (Nx_ragged.values r) in
  List.init (Nx_ragged.length r) (fun p ->
      List.init
        (o.(p + 1) - o.(p))
        (fun k -> Array.to_list (Array.sub v ((o.(p) + k) * c) c)))

let s2_cell t p = List.init 10 (band t p) @ [ doys.(t) ]
let expected_s2 = [ [ s2_cell 0 0; s2_cell 2 0 ]; [] ]

let tiles =
  group "of_tile"
    [
      test "reads the valid observations in order" (fun () ->
          with_tile (fun dir ->
              let px, grid = Pixels.of_tile (Tile.load dir) in
              equal (pair int int) (1, 2) grid;
              equal (list (list (list int))) expected_s2 (rows px.s2);
              equal
                (list (list (list int)))
                [
                  [ [ 5; 0; 40; 0 ]; [ 1; 1; 60; 1 ] ];
                  [ [ 0; 7; 40; 0 ]; [ 3; 3; 50; 0 ] ];
                ]
                (rows px.s1)));
      test "keeps negative reflectances" (fun () ->
          let values = Array.init 10 (fun c -> -999 + c) in
          let tile =
            {
              (Tile.empty (1, 1)) with
              bands = Nx.create Nx.int16 [| 1; 1; 1; 10 |] values;
              masks = Nx.ones Nx.uint8 [| 1; 1; 1 |];
              doys = Nx.create Nx.uint16 [| 1 |] [| 366 |];
            }
          in
          let px, _ = Pixels.of_tile tile in
          equal
            (list (list (list int)))
            [ [ Array.to_list values @ [ 366 ] ] ]
            (rows px.s2));
    ]

(* Corpora *)

let rm_rf dir =
  let rec go d =
    Array.iter
      (fun n ->
        let f = Filename.concat d n in
        if Sys.is_directory f then go f else Sys.remove f)
      (Sys.readdir d);
    Sys.rmdir d
  in
  if Sys.file_exists dir then go dir

let with_dir f =
  let dir = Filename.temp_dir "tessera" "corpus" in
  Fun.protect ~finally:(fun () -> rm_rf dir) (fun () -> f dir)

let pixel_rows px = List.combine (rows px.Pixels.s2) (rows px.Pixels.s1)

let round_trip =
  prop "a saved corpus loads as its pixels" ~count:20 (Synth.spec ~long:true ())
    (fun spec ->
      let px = Synth.pixels spec in
      let n = Pixels.length px in
      let pos = n / 3 in
      let part = Pixels.sub px ~pos ~len:(n - pos) in
      cover "a part that starts past its values' first row" (pos > 0);
      with_dir (fun dir ->
          let d = Filename.concat dir "c" in
          Pixels.save d part;
          let back = Pixels.load d in
          equal
            (list (pair (list (list int)) (list (list int))))
            (pixel_rows part) (pixel_rows back)))

let shuffled =
  prop "a shuffle puts pixel j at position permute j" ~count:40
    Gen.(
      triple (Synth.spec ()) (Synth.spec ())
        (frequency
           [
             (1, int_range 1 21); (1, int_range 22 100); (1, int_range 101 200);
           ]))
    (fun (a, b, bytes) ->
      let pa = Synth.pixels a and pb = Synth.pixels b in
      with_dir (fun dir ->
          let part name px =
            let d = Filename.concat dir name in
            Pixels.save d px;
            d
          in
          let parts = [ part "a" pa; part "b" pb ] in
          let out = Filename.concat dir "out" in
          let key = Nx.Rng.key 3 in
          let n = Pixels.shuffle ~key ~bytes out parts in
          equal int (Pixels.length pa + Pixels.length pb) n;
          let all = Array.of_list (pixel_rows pa @ pixel_rows pb) in
          let dest =
            Nx.to_array (Nx.Rng.permute ~key ~n (Nx.arange Nx.int64 0 n 1))
          in
          let expected = Array.make n ([], []) in
          Array.iteri (fun j q -> expected.(Int64.to_int q) <- all.(j)) dest;
          cover "runs of several pixels" (bytes > 100);
          cover "a pixel larger than a run" (bytes < 22);
          equal
            (list (pair (list (list int)) (list (list int))))
            (Array.to_list expected)
            (pixel_rows (Pixels.load out))))

let corpora =
  group "corpora"
    [
      round_trip;
      shuffled;
      test "refuses a corpus of other cells" (fun () ->
          with_dir (fun dir ->
              Nx_io.save_npy
                (Filename.concat dir "s2.npy")
                (Nx.zeros Nx.uint16 [| 2; 3 |]);
              Nx_io.save_npy
                (Filename.concat dir "s2_offsets.npy")
                (Nx.zeros Nx.int64 [| 1 |]);
              raises_match (Exn.failure ~sub:"s2.npy") (fun () ->
                  Pixels.load dir)));
    ]

(* Reads by ranges *)

(* A window of a corpus of [n] pixels, as a run reads one: pixels drawn with
   repeats in any order, a stretch of the first epoch, or a window that
   straddles its end, the stretch's last pixels then the next epoch's first,
   permuted. *)
type window = Drawn | Stretch | Straddling

let window_of n =
  let open Gen in
  let* kind = of_list [ Drawn; Stretch; Straddling ] in
  let+ picks =
    list
      ~size:(frequency [ (1, constant 0); (3, int_range 1 12) ])
      (int_range 0 (n - 1))
  and+ p = int_range 0 (n - 1)
  and+ len = int_range 1 (2 * n) in
  let indices =
    match kind with
    | Drawn -> List.map Int64.of_int picks
    | Stretch -> List.init (n - p) (fun i -> Int64.of_int (p + i))
    | Straddling ->
        let rest = Nx.arange Nx.int64 0 (min len n) 1 in
        let next = Nx.Rng.permute ~key:(Nx.Rng.key len) ~n rest in
        List.init (n - p) (fun i -> Int64.of_int (p + i))
        @ Array.to_list (Nx.to_array next)
  in
  (kind, Nx.create Nx.int64 [| List.length indices |] (Array.of_list indices))

(* A value's bytes, so floats compare by their bits. *)
let bytes x = Nx.to_array (Nx.bitcast Nx.uint8 x)

let ragged_bytes r =
  (bytes (Nx_ragged.offsets r), bytes (Nx_ragged.values r))

let batch_bytes (b : _ Batch.t) =
  let rows (r : _ Batch.rows) =
    [
      bytes r.tokens;
      bytes r.log_count;
      bytes r.start;
      bytes r.stop;
      bytes r.slot_start;
      bytes r.slot_stop;
      bytes r.place;
      bytes r.at;
    ]
  in
  (bytes b.pixel :: rows b.s2) @ rows b.s1

(* The first batch a step packs of [px]'s pixels. *)
let first_batch px =
  let px, counts = Batch.expand px (Batch.v2_counts px) in
  fst (Batch.pack (Batch.typical ~pixels:16) px counts ~from:0)

(* A read by ranges of [indices] of the corpus of [spec], against the mapped
   take of them. *)
let read_matches spec indices =
  with_dir (fun dir ->
      let d = Filename.concat dir "c" in
      Pixels.save d (Synth.pixels spec);
      let corpus = Pixels.load d in
      let read = Pixels.read d indices in
      let taken = Pixels.take ~indices corpus in
      let ragged = pair (array int) (array int) in
      equal ~msg:"s2" ragged (ragged_bytes taken.s2) (ragged_bytes read.s2);
      equal ~msg:"s1" ragged (ragged_bytes taken.s1) (ragged_bytes read.s1);
      equal ~msg:"batch" (list (array int))
        (batch_bytes (first_batch taken))
        (batch_bytes (first_batch read)))

let reads =
  prop
    "a read by ranges is the mapped take, bit for bit, and so is its batch"
    ~count:40
    Gen.(
      let* spec = Synth.spec ~pixels:10 ~long:true () in
      let+ window = window_of (List.length spec.s2) in
      (spec, window))
    (fun (spec, (kind, indices)) ->
      cover "drawn pixels" (kind = Drawn);
      cover "a stretch" (kind = Stretch);
      cover "a window that straddles the first epoch's end" (kind = Straddling);
      read_matches spec indices)

let read_none =
  test "a read of no pixel is the mapped take of none" (fun () ->
      let spec = { Synth.s2 = [ 1; 0; 300 ]; s1 = [ 2; 1; 0 ]; seed = 5 } in
      read_matches spec (Nx.zeros Nx.int64 [| 0 |]))

(* A corpus of three pixels in [dir]/c, [f] of its path. *)
let with_three f =
  with_dir (fun dir ->
      let d = Filename.concat dir "c" in
      let spec = { Synth.s2 = [ 1; 2; 3 ]; s1 = [ 2; 1; 1 ]; seed = 4 } in
      Pixels.save d (Synth.pixels spec);
      f d)

(* Files changed since a load, read at pixels 1 and 2: the s2 offsets
   [0; 1; 3; 6] of pixels of 1, 2 and 3 observations rewritten, or its
   values. *)
let read_corrupt =
  cases "a read of a corrupt file fails, naming it" ~id:fst
    [
      ( "offsets that decrease",
        ( "s2_offsets.npy",
          fun d ->
            Nx_io.save_npy
              (Filename.concat d "s2_offsets.npy")
              (Nx.create Nx.int64 [| 4 |] [| 0L; 3L; 1L; 6L |]) ) );
      ( "offsets past the values",
        ( "s2_offsets.npy",
          fun d ->
            Nx_io.save_npy
              (Filename.concat d "s2_offsets.npy")
              (Nx.create Nx.int64 [| 4 |] [| 0L; 1L; 3L; 9L |]) ) );
      ( "values of other cells",
        ( "s2.npy",
          fun d ->
            Nx_io.save_npy (Filename.concat d "s2.npy")
              (Nx.zeros Nx.int16 [| 6; 5 |]) ) );
      ( "values of another dtype",
        ( "s2.npy",
          fun d ->
            Nx_io.save_npy (Filename.concat d "s2.npy")
              (Nx.zeros Nx.int32 [| 6; 11 |]) ) );
    ]
    (fun (_, (file, corrupt)) ->
      with_three (fun d ->
          corrupt d;
          let path = Filename.concat d file in
          raises_match
            (Exn.failure ~sub:("Tessera.Pixels.read: " ^ path))
            (fun () ->
              Pixels.read d (Nx.create Nx.int64 [| 2 |] [| 1L; 2L |]))))

(* A run's windows of a corpus of three pixels, its files replaced after its
   load by those of its pixels in reverse, renamed over them: the load's
   mappings keep the files it read, and a read by ranges reads the new ones. A
   window of positions in the first epoch takes the load's pixels, and any
   other reads the files, the one that straddles the first epoch's end
   included. *)
let windows =
  test "a window in the first epoch takes the load, any other reads the files"
    (fun () ->
      with_dir (fun dir ->
          let d = Filename.concat dir "c" and e = Filename.concat dir "e" in
          let spec = { Synth.s2 = [ 1; 2; 3 ]; s1 = [ 2; 1; 1 ]; seed = 4 } in
          let px = Synth.pixels spec in
          let take px indices = Pixels.take ~indices px in
          let ids l =
            Nx.create Nx.int64 [| List.length l |] (Array.of_list l)
          in
          Pixels.save d px;
          let corpus = Pixels.load d in
          let reversed = take px (ids [ 2L; 1L; 0L ]) in
          Pixels.save e reversed;
          List.iter
            (fun f -> Sys.rename (Filename.concat e f) (Filename.concat d f))
            [ "s2.npy"; "s2_offsets.npy"; "s1.npy"; "s1_offsets.npy" ];
          let at = ids [ 0L; 2L ] in
          let ragged = pair (array int) (array int) in
          let check msg want got =
            equal ~msg:(msg ^ " s2") ragged (ragged_bytes want.Pixels.s2)
              (ragged_bytes got.Pixels.s2);
            equal ~msg:(msg ^ " s1") ragged (ragged_bytes want.Pixels.s1)
              (ragged_bytes got.Pixels.s1)
          in
          check "first epoch" (take px at) (Pixels.window d corpus ~first:1 at);
          check "straddling" (take reversed at)
            (Pixels.window d corpus ~first:2 at);
          check "second epoch" (take reversed at)
            (Pixels.window d corpus ~first:4 at)))

(* Takes and streams *)

let takes =
  prop "a take is each sensor's take, an index outside a pixel of nothing"
    ~count:40
    Gen.(
      let* spec = Synth.spec ~long:true () in
      let n = List.length spec.s2 in
      let+ picks = list ~size:(int_range 0 12) (int_range (-1) n) in
      (spec, picks))
    (fun (spec, picks) ->
      let px = Synth.pixels spec in
      let n = Pixels.length px in
      let indices =
        Nx.create Nx.int64
          [| List.length picks |]
          (Array.of_list (List.map Int64.of_int picks))
      in
      let t = Pixels.take ~indices px in
      cover "an index outside" (List.exists (fun i -> i < 0 || i >= n) picks);
      cover "a pixel taken twice"
        (List.length (List.sort_uniq compare picks) < List.length picks);
      let ragged = pair (array int) (array int) in
      equal ~msg:"s2" ragged
        (ragged_bytes (Nx_ragged.take ~indices px.s2))
        (ragged_bytes t.s2);
      equal ~msg:"s1" ragged
        (ragged_bytes (Nx_ragged.take ~indices px.s1))
        (ragged_bytes t.s1);
      let all = Array.of_list (pixel_rows px) in
      let row i = if i < 0 || i >= n then ([], []) else all.(i) in
      equal
        (list (pair (list (list int)) (list (list int))))
        (List.map row picks) (pixel_rows t))

let ints x = Array.to_list (Array.map Int64.to_int (Nx.to_array x))
let stream ?(key = Nx.Rng.key 9) n first len =
  ints (Pixels.stream ~key ~n ~first len)

let streams =
  prop "a stream reads epoch 0 in order, then a permutation an epoch"
    ~count:40
    Gen.(
      triple
        (frequency [ (1, constant 1); (4, int_range 2 40) ])
        (int_range 0 3) (int_range 0 100))
    (fun (n, e, k) ->
      cover "one pixel" (n = 1);
      cover "an epoch after the first" (e > 0);
      let all = List.init n Fun.id in
      equal (list int) all (stream n 0 n);
      let epoch = stream n (e * n) n in
      equal (list int) all (List.sort compare epoch);
      (* Positionwise: a window is the concatenation of any cut of it. *)
      let first = e * n and len = n + 3 in
      let k = k mod (len + 1) in
      cover "a cut that straddles an epoch's end" (k > 0 && k < len);
      equal (list int) (stream n first len)
        (stream n first k @ stream n (first + k) (len - k)))

let stream_cases =
  [
    test "a stream's later epochs are its key's" (fun () ->
        let n = 16 in
        equal (list int) (stream n 0 n) (stream ~key:(Nx.Rng.key 1) n 0 n);
        not_equal (list int) (stream n n n)
          (stream ~key:(Nx.Rng.key 1) n n n));
    test "a stream's epoch e is the permutation of fold_in key e" (fun () ->
        let n = 7 and key = Nx.Rng.key 9 in
        for e = 1 to 3 do
          let want =
            Nx.Rng.permute ~key:(Nx.Rng.fold_in key e) ~n
              (Nx.arange Nx.int64 0 n 1)
          in
          equal ~msg:(Printf.sprintf "epoch %d" e) (list int) (ints want)
            (stream ~key n (e * n) n)
        done);
    test "a stream of no position is empty" (fun () ->
        equal (list int) [] (stream 5 7 0));
  ]

(* Views *)

let rows_of r =
  let v = Nx.to_array (Nx.cast Nx.int32 (Nx_ragged.values r)) in
  let o = Nx.to_array (Nx_ragged.offsets r) in
  let cells = Nx.dim 1 (Nx_ragged.values r) in
  List.init (Nx_ragged.length r) (fun i ->
      let a = Int64.to_int o.(i) and b = Int64.to_int o.(i + 1) in
      List.init (b - a) (fun j -> Array.sub v ((a + j) * cells) cells))

let both =
  Gen.(
    map
      (fun ((n, len), seed) ->
        ( {
            Synth.s2 = List.init n (fun i -> 1 + ((i + seed) mod 4));
            s1 = List.init n (fun i -> 1 + (((3 * i) + seed) mod 5));
            seed;
          },
          len ))
      (pair (pair (int_range 1 6) (int_range 1 9)) (int_range 0 1000)))

let draws =
  prop "a view draws each row's observations from its key" ~count:50 both
    (fun (spec, length) ->
      let px = Synth.pixels spec in
      let n = Pixels.length px in
      let keys = Nx.Rng.split_batch ~n (Nx.Rng.key spec.seed) in
      let v = Pixels.draw ~keys ~length px in
      cover "an observation drawn twice" (length > 1);
      let drawn rows rows' =
        List.iter2
          (fun row row' ->
            equal int length (List.length row');
            List.iter (fun o -> mem (array int32) o row) row')
          rows rows'
      in
      drawn (rows_of px.s2) (rows_of v.s2);
      drawn (rows_of px.s1) (rows_of v.s1);
      (* Row i's draws are its key's: the last row drawn alone is the same. *)
      let last = Pixels.sub px ~pos:(n - 1) ~len:1 in
      let k =
        Nx.Rng.of_tensor (Nx.slice [ Nx.R (n - 1, n) ] (Nx.Rng.to_tensor keys))
      in
      let alone = Pixels.draw ~keys:k ~length last in
      equal
        (list (list (array int32)))
        [ List.nth (rows_of v.s1) (n - 1) ]
        (rows_of alone.s1))

let () =
  exit
    (run "tessera.pixels"
       [
         tiles;
         corpora;
         group "takes and streams" (takes :: streams :: stream_cases);
         group "reads by ranges"
           [ reads; read_none; read_corrupt; windows ];
         group "views" [ draws ];
       ])
