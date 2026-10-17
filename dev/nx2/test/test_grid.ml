(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Device grids, through the engine's private Grid, copied here. *)

open Windtrap
module M = Nx_array.Move

let grid = Testable.make ~pp:(Grid.pp Format.pp_print_int) ~equal:Grid.equal

let range =
  Testable.make
    ~pp:(fun ppf (r : M.range) ->
      Format.fprintf ppf "%d+%d/%d" r.start r.count r.step)
    ~equal:( = )

let ok r = require_ok ~pp:Format.pp_print_string r
let v devices extents cuts = ok (Grid.v ~devices ~extents ~cuts)
let all = [| 0; 1; 2; 3 |]
let on = v all [| 4 |] [||]
let split axis = v all [| 4 |] [| (axis, [| 0 |]) |]
let mesh cuts = v all [| 2; 2 |] cuts
let pp g = Format.asprintf "%a" (Grid.pp Format.pp_print_int) g

(* Grids over four devices, of a value of rank [rank]. *)
let grid_of rank =
  let axis = Gen.int_range 0 (rank - 1) in
  Gen.with_pp
    (Grid.pp Format.pp_print_int)
    (Gen.one_of
       [
         Gen.map Grid.device (Gen.int_range 0 3);
         Gen.constant on;
         Gen.map split axis;
         Gen.map
           (fun (x, y) ->
             if x = y then mesh [| (x, [| 0; 1 |]) |]
             else mesh [| (x, [| 0 |]); (y, [| 1 |]) |])
           (Gen.pair axis axis);
         Gen.map (fun x -> mesh [| (x, [| 1 |]) |]) axis;
       ])

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let gridded =
  Gen.with_pp
    (fun ppf (g, s) ->
      Format.fprintf ppf "%a over %a" (Grid.pp Format.pp_print_int) g pp_shape s)
    (Gen.bind (Gen.int_range 1 3) (fun rank ->
         Gen.pair (grid_of rank)
           (Gen.array ~size:(Gen.constant rank) (Gen.int_range 0 9))))

let rec indices = function
  | [] -> [ [] ]
  | n :: rest ->
      List.concat_map
        (fun i -> List.map (fun t -> i :: t) (indices rest))
        (List.init n Fun.id)

let inside (w : M.range array) idx =
  List.for_all2
    (fun (r : M.range) i -> i >= r.start && i < r.start + r.count)
    (Array.to_list w) idx

let window g s i = ok (Grid.window g s i)

(* A grid as [Grid.v] takes it, before its normal form: devices in any order,
   extents of 1 to 3, each grid axis cutting some value axis or none, a cut's
   grid axes in any order; and a shape of rank 1 to 3 with every cut axis. *)
type raw = {
  devices : int array;
  extents : int array;
  cuts : (int * int array) array;
  shape : int array;
}

let pp_raw ppf r =
  let ints a = String.concat "; " (Array.to_list (Array.map string_of_int a)) in
  Format.fprintf ppf "devices [%s] extents [%s] cuts [%s] shape [%s]"
    (ints r.devices) (ints r.extents)
    (String.concat "; "
       (Array.to_list
          (Array.map
             (fun (a, o) -> Printf.sprintf "%d/[%s]" a (ints o))
             r.cuts)))
    (ints r.shape)

let raw =
  let open Gen in
  let* g = int_range 1 3 in
  let* extents = array ~size:(constant g) (int_range 1 3) in
  let n = Array.fold_left ( * ) 1 extents in
  let* devices = permutation (List.init n Fun.id) in
  let* rank = int_range 1 3 in
  let* order = permutation (List.init g Fun.id) in
  let* owner = array ~size:(constant g) (int_range (-1) (rank - 1)) in
  let+ shape = array ~size:(constant rank) (int_range 0 9) in
  let cuts =
    List.filter_map
      (fun a ->
        match List.filter (fun h -> owner.(h) = a) order with
        | [] -> None
        | over -> Some (a, Array.of_list over))
      (List.init rank Fun.id)
  in
  { devices = Array.of_list devices; extents; cuts = Array.of_list cuts; shape }

let raw = Gen.with_pp pp_raw raw

(* The window the balanced rule gives tile [j] of [t] along an axis of extent
   [n]: it starts at [j * (n / t) + min j (n mod t)] and holds [n / t]
   elements, one more where [j < n mod t]. *)
let tile n t j =
  let q = n / t and r = n mod t in
  {
    M.start = (j * q) + Int.min j r;
    count = (q + if j < r then 1 else 0);
    step = 1;
  }

(* The window of the device at position [i] of [r]: along a cut, its tile
   numbered major first over the cut's grid axes, of their extents' product. *)
let model r i =
  let g = Array.length r.extents in
  let coord = Array.make g 0 and k = ref i in
  for h = g - 1 downto 0 do
    coord.(h) <- !k mod r.extents.(h);
    k := !k / r.extents.(h)
  done;
  let w = Array.map (fun n -> { M.start = 0; count = n; step = 1 }) r.shape in
  Array.iter
    (fun (a, over) ->
      let t = Array.fold_left (fun t h -> t * r.extents.(h)) 1 over in
      let j =
        Array.fold_left (fun j h -> (j * r.extents.(h)) + coord.(h)) 0 over
      in
      w.(a) <- tile r.shape.(a) t j)
    r.cuts;
  w

let of_raw r = v r.devices r.extents r.cuts

(* The position in [g] of a device of [g]. *)
let position g k = Option.get (Array.find_index (( = ) k) (Grid.devices g))

let laws =
  group "laws"
    [
      prop
        "each device holds its tile by the balanced rule, numbered major \
         first"
        raw (fun r ->
          let g = of_raw r in
          let multi = Array.exists (fun (_, o) -> Array.length o > 1) r.cuts in
          cover "a cut over several grid axes" multi;
          cover "a cut over several grid axes minor first"
            (Array.exists
               (fun (_, o) -> Array.length o > 1 && o.(0) > o.(1))
               r.cuts);
          cover "an uneven cut"
            (Array.exists (fun (a, _) -> r.shape.(a) mod 2 = 1) r.cuts);
          for i = 0 to Grid.count g - 1 do
            equal (array range) ~msg:(Printf.sprintf "device %d" i) (model r i)
              (window g r.shape i)
          done);
      prop
        "the windows of a cut lie in tile order, cover its axis and differ by \
         one at most"
        raw (fun r ->
          let g = of_raw r in
          Array.iter
            (fun (axis, t) ->
              let n = r.shape.(axis) in
              cover "an uneven cut" (n mod t <> 0);
              cover "an empty window" (n > 0 && n < t);
              cover "a cut of an empty axis" (n = 0);
              let ws =
                Array.init t (fun j ->
                    let held = Grid.devices (Grid.select g ~axis j) in
                    let w k = (window g r.shape (position g k)).(axis) in
                    Array.iter
                      (fun k ->
                        equal range ~msg:"every device of a tile" (w held.(0))
                          (w k))
                      held;
                    w held.(0))
              in
              equal int ~msg:"the first starts the axis" 0 ws.(0).start;
              for j = 1 to t - 1 do
                equal int ~msg:"each starts where the one before ends"
                  (ws.(j - 1).start + ws.(j - 1).count)
                  ws.(j).start;
                at_most int ~msg:"no window holds more than the one before"
                  ~than:ws.(j - 1).count ws.(j).count
              done;
              equal int ~msg:"the last ends the axis" n
                (ws.(t - 1).start + ws.(t - 1).count);
              at_most int ~msg:"the first holds one more at most"
                ~than:(ws.(t - 1).count + 1) ws.(0).count;
              Array.iteri
                (fun j (w : M.range) ->
                  equal bool ~msg:(Printf.sprintf "window %d is empty" j)
                    (j >= n) (w.count = 0))
                ws)
            (Grid.cuts g));

      prop "every index lies in exactly one distinct window" gridded
        (fun (g, s) ->
          cover "a cut" (Grid.cuts g <> [||]);
          cover "two cut axes" (Array.length (Grid.cuts g) = 2);
          let ws =
            List.sort_uniq compare
              (List.init (Grid.count g) (fun i -> Array.to_list (window g s i)))
          in
          let ws = List.map Array.of_list ws in
          List.iter
            (fun idx ->
              equal int ~msg:"windows holding the index" 1
                (List.length (List.filter (fun w -> inside w idx) ws)))
            (indices (Array.to_list s)));
      prop "a leading axis added and removed gives the grid back" gridded
        (fun (g, _) ->
          equal grid g
            (Grid.map_axes pred (Grid.uncut (Grid.map_axes succ g) ~axis:0)));
      prop "a new leading axis is whole and moves the windows up" gridded
        (fun (g, s) ->
          let g' = Grid.map_axes succ g and s' = Array.append [| 2 |] s in
          for i = 0 to Grid.count g - 1 do
            let w = window g s i and w' = window g' s' i in
            equal range { start = 0; count = 2; step = 1 } w'.(0);
            equal (array range) w (Array.sub w' 1 (Array.length s))
          done);
    ]

let forms =
  group "normal form"
    [
      test "a mesh with every grid axis cutting one axis in order is a split"
        (fun () -> equal grid (split 0) (mesh [| (0, [| 0; 1 |]) |]));
      test "a mesh cut minor axis first is not a split" (fun () ->
          not_equal grid (split 0) (mesh [| (0, [| 1; 0 |]) |]));
      test "a mesh without cuts is the whole on every device" (fun () ->
          equal grid on (mesh [||]));
      test "a grid of one device is that device" (fun () ->
          equal (option int) (Some 3)
            (Grid.one (v [| 3 |] [| 1; 1 |] [| (0, [| 1 |]) |])));
      cases "Grid.v refuses" ~name:fst
        [
          ( "extents that miss the devices",
            Grid.v ~devices:[| 0; 1 |] ~extents:[| 3 |] ~cuts:[||] );
          ( "a repeated device",
            Grid.v ~devices:[| 0; 0 |] ~extents:[| 2 |] ~cuts:[||] );
          ( "a negative device",
            Grid.v ~devices:[| -1; 0 |] ~extents:[| 2 |] ~cuts:[||] );
          ("a zero extent", Grid.v ~devices:[||] ~extents:[| 0 |] ~cuts:[||]);
          ( "an axis cut twice",
            Grid.v ~devices:all ~extents:[| 2; 2 |]
              ~cuts:[| (0, [| 0 |]); (0, [| 1 |]) |] );
          ( "a grid axis named twice",
            Grid.v ~devices:all ~extents:[| 2; 2 |]
              ~cuts:[| (0, [| 0 |]); (1, [| 0 |]) |] );
          ( "a negative axis",
            Grid.v ~devices:[| 0; 1 |] ~extents:[| 2 |]
              ~cuts:[| (-1, [| 0 |]) |] );
          ( "a grid axis out of range",
            Grid.v ~devices:[| 0; 1 |] ~extents:[| 2 |] ~cuts:[| (0, [| 1 |]) |]
          );
        ]
        (fun (_, r) ->
          ignore
            (require_error
               ~pp:(fun ppf _ -> Format.pp_print_string ppf "a grid")
               r));
    ]

let operations =
  group "operations"
    [
      test "a split's windows are the axis in order of devices" (fun () ->
          equal (array range)
            [|
              { start = 0; count = 3; step = 1 };
              { start = 4; count = 2; step = 1 };
            |]
            (window (split 1) [| 3; 8 |] 2));
      test "a split of an axis the shape lacks is an error" (fun () ->
          ignore
            (require_error
               ~pp:(fun ppf _ -> Format.pp_print_string ppf "a window")
               (Grid.window (split 1) [| 6 |] 0)));
      test "selecting a tile keeps the devices that hold it" (fun () ->
          equal (option int) (Some 2)
            (Grid.one (Grid.select (split 0) ~axis:0 2));
          raises_match (Exn.invalid_arg ~substring:"") (fun () ->
              Grid.select (split 0) ~axis:0 4));
      test "uncutting an axis leaves copies on the same devices" (fun () ->
          equal grid on (Grid.uncut (split 1) ~axis:1));
      test "a grid prints as the placement that makes it" (fun () ->
          equal string "on [0; 1; 2; 3]" (pp on);
          equal string "split ~axis:1 [0; 1; 2; 3]" (pp (split 1));
          equal string "mesh 2x2 [0; 1; 2; 3] ~axis:0/1 ~axis:1/0"
            (pp (mesh [| (0, [| 1 |]); (1, [| 0 |]) |])));
    ]

let () = exit (run "nx grid" [ laws; forms; operations ])
