(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gathers and scatters by index tensors. Slicing and [set] are in
   test_values. *)

open Windtrap
open Nx_test

let ints = Ref.witness int32
let shape = Gen.array ~size:(Gen.int_range 1 3) (Gen.int_range 1 4)
let iota s = Array.init (Ref.numel s) (fun i -> Int32.of_int (i + 1))
let tensor_of s = (Ref.create s (iota s), Nx.create Nx.int32 s (iota s))

let indices_tensor l =
  Nx.create Nx.int32 [| Array.length l |] (Array.map Int32.of_int l)

(* A shape, an axis of it, and indices along that axis from one past each
   end. *)
let along =
  let open Gen in
  let* s = shape in
  let* axis = int_range 0 (Array.length s - 1) in
  let n = s.(axis) in
  let+ idx = array ~size:(int_range 0 5) (int_range (-2) (n + 1)) in
  (s, axis, idx)

let read r src =
  if Array.for_all2 (fun k d -> k >= 0 && k < d) src r.Ref.shape then
    Ref.get r src
  else 0l

let gathers =
  group "gathers"
    [
      prop "take reads the flattened tensor, and zero out of range"
        (Gen.pair shape
           (Gen.array ~size:(Gen.int_range 0 6) (Gen.int_range (-2) 30)))
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
      prop "take along an axis reads each index of that axis" along
        (fun (s, axis, idx) ->
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
        along (fun (s, axis, _) ->
          let r, t = tensor_of s in
          let n = s.(axis) in
          let positions =
            Ref.init s (fun i -> Int32.of_int ((Ref.ravel s i mod (n + 3)) - 1))
          in
          let indices = Nx.create Nx.int32 s positions.data in
          equal ints
            (Ref.init s (fun i ->
                 let src = Array.copy i in
                 src.(axis) <- Int32.to_int (Ref.get positions i);
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
         drops out of range; a scalar value is broadcast"
        (Gen.triple along Gen.bool Gen.bool) (fun ((s, axis, _), add, scalar) ->
          let r, t = tensor_of s in
          let n = s.(axis) in
          let positions =
            Ref.init s (fun i ->
                Int32.of_int ((Ref.ravel s i * 7 mod (n + 3)) - 1))
          in
          let values =
            Ref.init s (fun i ->
                if scalar then 100l else Int32.of_int (100 * (Ref.ravel s i + 1)))
          in
          let expected = Array.copy r.data in
          for i = 0 to Ref.numel s - 1 do
            let idx = Ref.unravel s i in
            let k = Int32.to_int (Ref.get positions idx) in
            if k >= 0 && k < n then begin
              let dst = Array.copy idx in
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
                  ~indices:(Nx.create Nx.int32 s positions.data)
                  ~values:
                    (if scalar then Nx.scalar Nx.int32 100l
                     else Nx.create Nx.int32 s values.data)
                  t)));
      test "scatter refuses indices of another rank" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.scatter ~axis:0 ~indices:(indices_tensor [| 0 |])
                ~values:(Nx.zeros Nx.int32 [| 1 |])
                (Nx.zeros Nx.int32 [| 2; 2 |])));
      test "scatter refuses indices whose shape differs off the axis" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.scatter ~axis:0
                ~indices:(Nx.zeros Nx.int32 [| 1; 3 |])
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
          equal ints
            (Ref.init [| k; n |] (fun i -> Int32.of_int rows.(i.(0)).(i.(1))))
            (Ref.of_nx (Nx.argwhere t));
          Array.iteri
            (fun d axis ->
              equal
                ~msg:(Printf.sprintf "axis %d" d)
                ints
                (Ref.init [| k |] (fun i -> Int32.of_int rows.(i.(0)).(d)))
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
      xfail
        ~reason:
          "compress without an axis never compares the lengths, and takes a \
           position past the end as a zero"
        (test
           "compress without an axis refuses a condition longer than the tensor"
           (fun () ->
             raises_invalid_arg (fun () ->
                 Nx.compress
                   ~condition:
                     (Nx.create Nx.bool [| 3 |] [| false; false; true |])
                   (Nx.zeros Nx.int32 [| 2 |]))));
      test "compress refuses a condition of another length" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.compress ~axis:0
                ~condition:(Nx.create Nx.bool [| 3 |] [| true; false; true |])
                (Nx.zeros Nx.int32 [| 2 |])));
      xfail ~reason:"extract compares shapes where nx.mli compares sizes"
        (test "extract flattens a condition of the same size and another shape"
           (fun () ->
             let r, t = tensor_of [| 2; 3 |] in
             let cond = [| true; false; false; true; true; false |] in
             equal ints (Ref.compress cond r)
               (Ref.of_nx
                  (Nx.extract ~condition:(Nx.create Nx.bool [| 6 |] cond) t))));
      test "extract refuses a condition of another size" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.extract
                ~condition:(Nx.create Nx.bool [| 3 |] [| true; false; true |])
                (Nx.zeros Nx.int32 [| 2 |])));
    ]

(* [D] reads a window whose start is a run-time scalar, clamped so the window
   fits. *)
let windows =
  let window start len =
    Nx.slice
      [ Nx.D (Nx.scalar Nx.int32 (Int32.of_int start), len) ]
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
        [ (0, 2, 0); (2, 2, 2); (4, 2, 3); (-3, 2, 0); (9, 5, 0); (1, 0, 1) ]
        (fun (start, len, first) ->
          equal (tensor int32) (expected first len) (window start len));
      test "a window longer than its axis is refused" (fun () ->
          raises_invalid_arg (fun () -> window 0 6));
      test "set writes a dynamic window" (fun () ->
          let x = Nx.zeros Nx.int32 [| 4 |] in
          equal (tensor int32)
            (Nx.create Nx.int32 [| 4 |] [| 0l; 0l; 7l; 7l |])
            (Nx.set
               [ Nx.D (Nx.scalar Nx.int32 5l, 2) ]
               (Nx.full Nx.int32 [| 2 |] 7l)
               x));
    ]

(* nx.mli leaves the bounds of a range open; [R] clamps them into the axis, as
   Python slices do, and [Ref] does the same for [Rs]. *)
let stepped_ranges =
  group "stepped ranges"
    [
      xfail
        ~reason:
          "a stepped range keeps its start unclamped: a gather reads zeros \
           from the out-of-range positions, and a step of 1 or -1 shrinks out \
           of bounds and raises"
        (cases "a stepped range starting outside its axis selects as R does"
           ~name:(fun (s : Nx.index) -> Format.asprintf "%a" pp_index s)
           [ Rs (-10, 3, 1); Rs (-10, 4, 2); Rs (10, 0, -1); Rs (10, 0, -2) ]
           (fun spec ->
             let r, t = tensor_of [| 4 |] in
             equal ints (Ref.slice [ spec ] r) (Ref.of_nx (Nx.slice [ spec ] t))));
    ]

let () =
  exit
    (run "nx indexing"
       [ gathers; scatters; selections; windows; stepped_ranges ])
