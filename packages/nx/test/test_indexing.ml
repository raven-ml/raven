(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gathers and scatters by index tensors. Slicing and [set] are in
   test_values. *)

open Windtrap
open Nx_test

let ints = Ref.witness int32
let positions = Ref.witness int64
let shape = Gen.array ~size:(Gen.int_range 1 3) (Gen.int_range 1 4)
let iota s = Array.init (Ref.numel s) (fun i -> Int32.of_int (i + 1))
let tensor_of s = (Ref.create s (iota s), Nx.create Nx.int32 s (iota s))

let indices_tensor l =
  Nx.create Nx.int64 [| Array.length l |] (Array.map Int64.of_int l)

let far = 1 lsl 32

(* An index along an axis of [n], [n > 0]: from one past each end, or a position
   of the axis moved by a multiple of 2^32, which a 32-bit truncation would
   bring back to that position. *)
let index n =
  let open Gen in
  frequency
    [
      (4, int_range (-2) (n + 1));
      ( 1,
        let+ i = int_range 0 (n - 1) and+ k = of_list [ -2; -1; 1; 2 ] in
        i + (k * far) );
    ]

(* Demands that both kinds of index outside an axis of [n] were drawn. *)
let cover_outside n idx =
  cover "an index just outside its axis"
    (Array.exists (fun k -> (k < 0 || k >= n) && Int.abs k < far / 2) idx);
  cover "an index 2^32 from a position of its axis"
    (Array.exists (fun k -> Int.abs k >= far / 2) idx)

(* A shape, an axis of it, and indices along that axis. *)
let along =
  let open Gen in
  let* s = shape in
  let* axis = int_range 0 (Array.length s - 1) in
  let+ idx = array ~size:(int_range 0 5) (index s.(axis)) in
  (s, axis, idx)

(* A shape, an axis of it, and an index along that axis at each position. *)
let positioned =
  let open Gen in
  let* s = shape in
  let* axis = int_range 0 (Array.length s - 1) in
  let+ idx = array ~size:(constant (Ref.numel s)) (index s.(axis)) in
  (s, axis, idx)

let read r src =
  if Array.for_all2 (fun k d -> k >= 0 && k < d) src r.Ref.shape then
    Ref.get r src
  else 0l

let gathers =
  group "gathers"
    [
      prop "take reads the flattened tensor, and zero out of range"
        (let open Gen in
         let* s = shape in
         let+ idx = array ~size:(int_range 0 6) (index (Ref.numel s)) in
         (s, idx))
        (fun (s, idx) ->
          let r, t = tensor_of s in
          let n = Ref.numel s in
          equal ints
            (Ref.init
               [| Array.length idx |]
               (fun i ->
                 let k = idx.(i.(0)) in
                 if k >= 0 && k < n then r.data.(k) else 0l))
            (Ref.of_nx (Nx.take ~indices:(indices_tensor idx) t)));
      prop
        "take reads an int64 index inside its axis and zero outside it, \
         however far"
        ~examples:[ ([| 4 |], 0, [| far + 1; 2 - far; 1 |]) ]
        along
        (fun (s, axis, idx) ->
          cover_outside s.(axis) idx;
          let r, t = tensor_of s in
          let out = Array.copy s in
          out.(axis) <- Array.length idx;
          equal ints
            (Ref.init out (fun i ->
                 let src = Array.copy i in
                 src.(axis) <- idx.(i.(axis));
                 read r src))
            (Ref.of_nx (Nx.take ~axis ~indices:(indices_tensor idx) t)));
      prop "take_along_axis reads, at each position, the index found there"
        positioned (fun (s, axis, idx) ->
          let r, t = tensor_of s in
          let indices = Nx.create Nx.int64 s (Array.map Int64.of_int idx) in
          equal ints
            (Ref.init s (fun i ->
                 let src = Array.copy i in
                 src.(axis) <- idx.(Ref.ravel s i);
                 read r src))
            (Ref.of_nx (Nx.take_along_axis ~axis ~indices t)));
      test "take without an axis reads a transposed tensor" (fun () ->
          let r, t = tensor_of [| 2; 3 |] in
          let indices = [| 1; 4 |] in
          equal ints
            (Ref.take ~zero:0l indices (Ref.transpose r))
            (Ref.of_nx
               (Nx.take ~indices:(indices_tensor indices) (Nx.transpose t))));
      test "take_along_axis refuses indices of another rank" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.take_along_axis ~axis:0 ~indices:(indices_tensor [| 0 |])
                (Nx.zeros Nx.int32 [| 2; 2 |])));
    ]

let scatters =
  group "scatters"
    [
      prop
        "scatter writes each value at its index, the last one winning, and \
         drops every update outside its axis, however far; a scalar value is \
         broadcast" (Gen.triple positioned Gen.bool Gen.bool)
        (fun ((s, axis, idx), add, scalar) ->
          let n = s.(axis) in
          cover_outside n idx;
          let r, t = tensor_of s in
          let values =
            Ref.init s (fun i ->
                if scalar then 100l else Int32.of_int (100 * (Ref.ravel s i + 1)))
          in
          let expected = Array.copy r.data in
          for i = 0 to Ref.numel s - 1 do
            let k = idx.(i) in
            if k >= 0 && k < n then begin
              let dst = Ref.unravel s i in
              dst.(axis) <- k;
              let j = Ref.ravel s dst in
              expected.(j) <-
                (if add then Int32.add expected.(j) values.data.(i)
                 else values.data.(i))
            end
          done;
          equal ints (Ref.create s expected)
            (Ref.of_nx
               (Nx.scatter
                  ~mode:(if add then `Add else `Set)
                  ~axis
                  ~indices:(Nx.create Nx.int64 s (Array.map Int64.of_int idx))
                  ~values:
                    (if scalar then Nx.scalar Nx.int32 100l
                     else Nx.create Nx.int32 s values.data)
                  t)));
      test "scatter's additions run from +0, and it keeps an unreached -0"
        (fun () ->
          let v xs = Nx.create Nx.float32 [| Array.length xs |] xs in
          let add ~unique_indices indices values =
            Nx.scatter ~mode:`Add ~unique_indices ~axis:0
              ~indices:(Nx.create Nx.int64 [| Array.length indices |] indices)
              ~values:(v values)
              (v [| -0.; -0.; -0. |])
          in
          List.iter
            (fun unique_indices ->
              let msg = Printf.sprintf "unique_indices = %b" unique_indices in
              equal ~msg (tensor float_exact) (v [| 0.; -0.; 0. |])
                (add ~unique_indices [| 0L; 2L |] [| -0.; -0. |]))
            [ false; true ];
          equal ~msg:"duplicates" (tensor float_exact) (v [| 0.; -0.; -0. |])
            (add ~unique_indices:false [| 0L; 0L |] [| -0.; -0. |]));
      test "scatter refuses indices of another rank" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.scatter ~axis:0 ~indices:(indices_tensor [| 0 |])
                ~values:(Nx.zeros Nx.int32 [| 1 |])
                (Nx.zeros Nx.int32 [| 2; 2 |])));
      prop "scatter with unique indices writes as the default does" along
        (fun (s, axis, _) ->
          let _, t = tensor_of s in
          let n = s.(axis) in
          (* Each lane takes the axis positions in reverse, which are unique. *)
          let positions =
            Nx.init Nx.int64 s (fun i -> Int64.of_int (n - 1 - i.(axis)))
          in
          let values = Nx.neg t in
          equal (tensor int32)
            (Nx.scatter ~axis ~indices:positions ~values t)
            (Nx.scatter ~unique_indices:true ~axis ~indices:positions ~values t));
      test "scatter refuses indices whose shape differs off the axis" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.scatter ~axis:0
                ~indices:(Nx.zeros Nx.int64 [| 1; 3 |])
                ~values:(Nx.zeros Nx.int32 [| 1; 3 |])
                (Nx.zeros Nx.int32 [| 2; 2 |])));
    ]

let selections =
  group "selections"
    [
      prop "compress keeps the positions of an axis where the condition holds"
        along (fun (s, axis, _) ->
          let r, t = tensor_of s in
          let cond = Array.init s.(axis) (fun i -> i mod 3 <> 1) in
          equal ints
            (Ref.slice
               (List.init axis (fun _ -> Nx.A)
               @ [
                   Nx.L
                     (List.filter
                        (fun i -> cond.(i))
                        (List.init s.(axis) Fun.id));
                 ])
               r)
            (Ref.of_nx
               (Nx.compress ~axis
                  ~condition:(Nx.create Nx.bool [| s.(axis) |] cond)
                  t)));
      prop
        "extract lists, in row-major order, the elements where the condition \
         holds"
        shape (fun s ->
          let r, t = tensor_of s in
          let cond = Array.map (fun v -> Int32.rem v 3l <> 0l) r.data in
          let kept =
            List.filter_map
              (fun (c, v) -> if c then Some v else None)
              (List.combine (Array.to_list cond) (Array.to_list r.data))
          in
          equal ints
            (Ref.create [| List.length kept |] (Array.of_list kept))
            (Ref.of_nx (Nx.extract ~condition:(Nx.create Nx.bool s cond) t)));
      prop
        "nonzero and argwhere list the coordinates of non-zero elements, in \
         row-major order"
        (Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 4))
        (fun s ->
          let r =
            Ref.init s (fun i -> Int32.of_int ((Ref.ravel s i + 1) mod 3))
          in
          let t = Nx.create Nx.int32 s r.data in
          let coords =
            List.filter_map
              (fun i ->
                if r.data.(i) <> 0l then Some (Ref.unravel s i) else None)
              (List.init (Ref.numel s) Fun.id)
          in
          let k = List.length coords and n = Array.length s in
          let rows = Array.of_list coords in
          equal positions
            (Ref.init [| k; n |] (fun i -> Int64.of_int rows.(i.(0)).(i.(1))))
            (Ref.of_nx (Nx.argwhere t));
          Array.iteri
            (fun d axis ->
              equal
                ~msg:(Printf.sprintf "axis %d" d)
                positions
                (Ref.init [| k |] (fun i -> Int64.of_int rows.(i.(0)).(d)))
                (Ref.of_nx axis))
            (Nx.nonzero t));
      prop
        "compress without an axis keeps the flattened positions where the \
         condition holds"
        shape (fun s ->
          let r, t = tensor_of s in
          let cond = Array.init (Ref.numel s) (fun i -> i mod 3 <> 1) in
          equal ints (Ref.compress cond r)
            (Ref.of_nx
               (Nx.compress
                  ~condition:(Nx.create Nx.bool [| Ref.numel s |] cond)
                  t)));
      test "compress and extract without an axis read a transposed tensor"
        (fun () ->
          let r, t = tensor_of [| 2; 3 |] in
          let cond = [| true; false; false; true; true; false |] in
          let condition = Nx.create Nx.bool [| 6 |] cond in
          let expected = Ref.compress cond (Ref.transpose r) in
          equal ints expected
            (Ref.of_nx (Nx.compress ~condition (Nx.transpose t)));
          equal ints expected
            (Ref.of_nx
               (Nx.extract
                  ~condition:(Nx.reshape [| 3; 2 |] condition)
                  (Nx.transpose t))));
      test "compress without an axis refuses a condition longer than the tensor"
        (fun () ->
          raises_invalid_arg (fun () ->
              Nx.compress
                ~condition:(Nx.create Nx.bool [| 3 |] [| false; false; true |])
                (Nx.zeros Nx.int32 [| 2 |])));
      test "compress refuses a condition of another length" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.compress ~axis:0
                ~condition:(Nx.create Nx.bool [| 3 |] [| true; false; true |])
                (Nx.zeros Nx.int32 [| 2 |])));
      test "extract flattens a condition of the same size and another shape"
        (fun () ->
          let r, t = tensor_of [| 2; 3 |] in
          let cond = [| true; false; false; true; true; false |] in
          equal ints (Ref.compress cond r)
            (Ref.of_nx
               (Nx.extract ~condition:(Nx.create Nx.bool [| 6 |] cond) t)));
      test "extract refuses a condition of another size" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.extract
                ~condition:(Nx.create Nx.bool [| 3 |] [| true; false; true |])
                (Nx.zeros Nx.int32 [| 2 |])));
    ]

let extremes =
  let x = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 3l |] in
  let at = Nx.create Nx.int64 [| 3 |] [| Int64.min_int; 1L; Int64.max_int |] in
  group "extreme indices"
    [
      test "Int64.min_int and Int64.max_int read zero" (fun () ->
          let expected = Nx.create Nx.int32 [| 3 |] [| 0l; 2l; 0l |] in
          equal (tensor int32) expected (Nx.take ~indices:at x);
          equal (tensor int32) expected
            (Nx.take_along_axis ~axis:0 ~indices:at x));
      test "updates at Int64.min_int and Int64.max_int are dropped" (fun () ->
          equal (tensor int32)
            (Nx.create Nx.int32 [| 3 |] [| 1l; 9l; 3l |])
            (Nx.scatter ~axis:0 ~indices:at ~values:(Nx.scalar Nx.int32 9l) x));
    ]

(* Each case holds one tensor of 2^31 + 2 bytes, after collecting the ones
   before it. *)
let past_int32 =
  let n = (1 lsl 31) + 2 in
  let zeros () = Nx.broadcast_to [| n |] (Nx.scalar Nx.uint8 0) in
  let at x i = Nx.item [ i ] x in
  group "indices past 2^31"
    [
      slow "take, argmax and argmin reach positions past 2^31" (fun () ->
          Gc.full_major ();
          let x = Nx.pad [| (n - 1, 0) |] 0 (Nx.ones Nx.uint8 [| 1 |]) in
          equal int64 (Int64.of_int (n - 1)) (Nx.item [] (Nx.argmax x));
          equal int64 0L (Nx.item [] (Nx.argmin x));
          equal (array int) [| 1; 0; 0 |]
            (Nx.to_array
               (Nx.take ~indices:(indices_tensor [| n - 1; n; -1 |]) x)));
      slow "scatter writes past 2^31 and drops what a truncation would alias"
        (fun () ->
          Gc.full_major ();
          let y =
            Nx.scatter ~axis:0
              ~indices:(indices_tensor [| n - 1; far + 3 |])
              ~values:(Nx.create Nx.uint8 [| 2 |] [| 7; 9 |])
              (zeros ())
          in
          equal int 7 (at y (n - 1));
          equal int 0 (at y 3));
      slow "a D window starts past 2^31" (fun () ->
          Gc.full_major ();
          let y =
            Nx.set
              [ Nx.D (Nx.scalar Nx.int64 (Int64.of_int (n - 2)), 2) ]
              (Nx.ones Nx.uint8 [| 2 |]) (zeros ())
          in
          equal (array int) [| 0; 1; 1 |]
            (Array.map (at y) [| n - 3; n - 2; n - 1 |]));
    ]

(* [D] reads a window whose start is a run-time scalar, clamped so the window
   fits. *)
let windows =
  let window start len =
    Nx.slice
      [ Nx.D (Nx.scalar Nx.int64 (Int64.of_int start), len) ]
      (Nx.arange Nx.int32 0 5 1)
  in
  let expected first len =
    Nx.init Nx.int32 [| len |] (fun i -> Int32.of_int (first + i.(0)))
  in
  group "dynamic windows"
    [
      cases "a window starts at its start, clamped into the axis"
        ~name:(fun (start, len, _) ->
          Printf.sprintf "start %d, length %d" start len)
        [
          (0, 2, 0);
          (2, 2, 2);
          (4, 2, 3);
          (-3, 2, 0);
          (9, 5, 0);
          (1, 0, 1);
          (far + 1, 2, 3);
          (1 - far, 2, 0);
        ]
        (fun (start, len, first) ->
          equal (tensor int32) (expected first len) (window start len));
      test "a window longer than its axis is refused" (fun () ->
          raises_invalid_arg (fun () -> window 0 6));
      test "set writes a dynamic window" (fun () ->
          let x = Nx.zeros Nx.int32 [| 4 |] in
          equal (tensor int32)
            (Nx.create Nx.int32 [| 4 |] [| 0l; 0l; 7l; 7l |])
            (Nx.set
               [ Nx.D (Nx.scalar Nx.int64 5L, 2) ]
               (Nx.full Nx.int32 [| 2 |] 7l)
               x));
    ]

(* nx.mli leaves the bounds of a range open; [R] clamps them into the axis, as
   Python slices do, and [Ref] does the same for [Rs]. *)
let stepped_ranges =
  group "stepped ranges"
    [
      cases "a stepped range starting outside its axis selects as R does"
        ~name:(fun (s : Nx.index) -> Format.asprintf "%a" pp_index s)
        [ Rs (-10, 3, 1); Rs (-10, 4, 2); Rs (10, 0, -1); Rs (10, 0, -2) ]
        (fun spec ->
          let r, t = tensor_of [| 4 |] in
          equal ints (Ref.slice [ spec ] r) (Ref.of_nx (Nx.slice [ spec ] t)));
    ]

let () =
  exit
    (run "nx indexing"
       [
         gathers;
         scatters;
         extremes;
         selections;
         windows;
         stepped_ranges;
         past_int32;
       ])
