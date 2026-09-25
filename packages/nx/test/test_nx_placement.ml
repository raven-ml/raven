(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Devices and placement, against a test engine whose devices hold their storage
   in host memory of their own and count what moves: placements are in normal
   form, a move keeps its source, operations run where their operands live,
   reads copy what they read, views share their storage, and nx holds scalars
   created on a device itself. *)

open Windtrap

(* The test engine *)

(* One buffer per device of the placement the storage was made for, in placement
   order: a split value's shards, a replicated value's copies. *)
type Nx_effect.storage +=
  | Mem :
      ('a, 'b) Nx_core.Dtype.t
      * Nx_effect.device list
      * ('a, 'b) Nx_buffer.t list
      -> Nx_effect.storage

let elements_read = ref 0
let uploads = ref 0

(* The elements view [v] reaches in [mem], in C order. *)
let gather (type a b) (mem : (a, b) Nx_buffer.t) v : (a, b) Nx_buffer.t =
  let shape = Nx_core.View.shape v and strides = Nx_core.View.strides v in
  let n = Nx_core.View.numel v in
  let dst = Nx_buffer.create (Nx_buffer.kind mem) n in
  for i = 0 to n - 1 do
    let idx = Nx_core.Shape.unravel_index i shape in
    let off = ref (Nx_core.View.offset v) in
    Array.iteri (fun d k -> off := !off + (k * strides.(d))) idx;
    Nx_buffer.set dst i (Nx_buffer.get mem !off)
  done;
  elements_read := !elements_read + n;
  dst

(* Each shard's view [v], interleaved in global order along [axis]. *)
let gather_shards (type a b) (shards : (a, b) Nx_buffer.t list) ~axis v :
    (a, b) Nx_buffer.t =
  let shape = Nx_core.View.shape v in
  let outer = Array.fold_left ( * ) 1 (Array.sub shape 0 axis) in
  let row = Nx_core.View.numel v / Int.max 1 outer in
  let n = List.length shards in
  let dst =
    Nx_buffer.create (Nx_buffer.kind (List.hd shards)) (n * outer * row)
  in
  List.iteri
    (fun k mem ->
      let part = gather mem v in
      for o = 0 to outer - 1 do
        for i = 0 to row - 1 do
          Nx_buffer.set dst
            ((((o * n) + k) * row) + i)
            (Nx_buffer.get part ((o * row) + i))
        done
      done)
    shards;
  dst

let rec engine =
  {
    Nx_effect.read =
      (fun (type a b) (r : (a, b) Nx_effect.resident) : (a, b) Nx_buffer.t ->
        match r.r_cell.state with
        | Live (Mem (dt, devices, shards)) -> (
            match Nx_core.Dtype.equal_witness dt r.r_dtype with
            | Some Type.Equal -> (
                match r.r_placement with
                | Sharded { axis; _ } -> gather_shards shards ~axis r.r_view
                | Device d ->
                    let k = Option.get (List.find_index (( == ) d) devices) in
                    gather (List.nth shards k) r.r_view
                | Replicated _ -> gather (List.hd shards) r.r_view)
            | None -> assert false)
        | _ -> assert false);
    place = (fun p x -> place p x);
  }

and place : type a b.
    Nx_effect.placement -> (a, b) Nx_effect.t -> (a, b) Nx_effect.t =
 fun p x ->
  if Nx.numel x > 0 then incr uploads;
  let h = Nx.place Nx.Placement.host x in
  let own h = Nx.to_buffer (Nx.copy h) in
  let shape = Array.copy (Nx.shape h) in
  let shards =
    match p with
    | Sharded { axis; devices } ->
        let n = List.length devices in
        let k = shape.(axis) / n in
        shape.(axis) <- k;
        List.init n (fun i ->
            own
              (Nx.slice
                 (List.init (axis + 1) (fun d ->
                      if d = axis then Nx.R (i * k, (i + 1) * k) else Nx.A))
                 h))
    | Replicated ds -> List.map (fun _ -> own h) ds
    | Device _ -> [ own h ]
  in
  Nx_effect.placed p (Nx.dtype x)
    (Nx_core.View.create shape)
    (Nx_effect.cell engine
       ~length:(Array.fold_left ( * ) 1 shape)
       (Mem (Nx.dtype x, Nx_effect.Placement.devices p, shards)))

let dev1 = Nx_effect.Device.make "TEST:1" engine
let dev2 = Nx_effect.Device.make "TEST:2" engine
let dev3 = Nx_effect.Device.make "TEST:3" engine
let dev4 = Nx_effect.Device.make "TEST:4" engine
let other = Nx_effect.Device.make "OTHER" { engine with place = engine.place }
let on1 = Nx.Placement.device dev1
let on2 = Nx.Placement.device dev2
let m23 () = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]

let raises_invalid f =
  raises_match (function Invalid_argument _ -> true | _ -> false) f

let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

let cell_of (type a b) (x : (a, b) Nx.t) =
  match x with
  | Nx_effect.Placed r -> r.r_cell
  | _ -> fail "expected a placed value"

(* Placements *)

let test_normal_forms () =
  equal ~msg:"a list of one device is that device" placement on1
    (Nx.Placement.replicated [ dev1 ]);
  equal ~msg:"a split over one device is that device" placement on1
    (Nx.Placement.sharded ~axis:3 [ dev1 ]);
  is_true ~msg:"the host"
    (Nx.Placement.equal Nx.Placement.host (Nx.Placement.device Nx.Device.host));
  is_false ~msg:"order matters"
    (Nx.Placement.equal
       (Nx.Placement.replicated [ dev1; dev2 ])
       (Nx.Placement.replicated [ dev2; dev1 ]));
  raises_invalid (fun () -> Nx.Placement.replicated []);
  raises_invalid (fun () -> Nx.Placement.replicated [ dev1; dev1 ]);
  raises_invalid (fun () -> Nx.Placement.sharded ~axis:0 [ dev1; other ]);
  raises_invalid (fun () -> Nx.Placement.sharded ~axis:(-1) [ dev1; dev2 ])

(* Moving *)

let test_place_keeps_its_source () =
  let x = m23 () in
  let p = Nx.place on1 x in
  equal ~msg:"placement" placement on1 (Nx.placement p);
  equal ~msg:"the source stays on the host" placement Nx.Placement.host
    (Nx.placement x);
  equal ~msg:"shape" (array int) [| 2; 3 |] (Nx.shape p);
  equal ~msg:"elements" (array float_exact) (Nx.to_array x) (Nx.to_array p);
  is_true ~msg:"placing where it is returns the value" (Nx.place on1 p == p);
  let h = Nx.place Nx.Placement.host p in
  equal ~msg:"a host copy" placement Nx.Placement.host (Nx.placement h);
  equal ~msg:"the placed value stays" placement on1 (Nx.placement p);
  equal ~msg:"its elements" (array float_exact) (Nx.to_array x) (Nx.to_array h)

let test_place_splits_evenly () =
  raises_invalid (fun () ->
      Nx.place (Nx.Placement.sharded ~axis:1 [ dev1; dev2 ]) (m23 ()));
  raises_invalid (fun () ->
      Nx.place (Nx.Placement.sharded ~axis:2 [ dev1; dev2 ]) (m23 ()))

(* Routing *)

let test_results_live_with_their_operands () =
  let x = m23 () in
  let p = Nx.place on1 x in
  let y = Nx.tanh (Nx.add p x) in
  equal ~msg:"a host operand joins the placed one" placement on1
    (Nx.placement y);
  equal ~msg:"the result"
    (array (float 1e-6))
    (Nx.to_array (Nx.tanh (Nx.add x x)))
    (Nx.to_array y);
  let s = Nx.sum y in
  equal ~msg:"a reduction too" placement on1 (Nx.placement s);
  let q = Nx.place on2 x in
  elements_read := 0;
  raises_match
    (function
      | Invalid_argument msg -> String.ends_with ~suffix:"place one of them" msg
      | _ -> false)
    (fun () -> Nx.add p q);
  equal ~msg:"and raise before reading anything" int 0 !elements_read

let test_created_scalars_are_held () =
  let p = Nx.place on1 (m23 ()) in
  let before = !uploads in
  let y = Nx.mul_s p 2.0 in
  equal ~msg:"one upload: the product" int (before + 1) !uploads;
  let z = Nx.zeros_like p in
  equal ~msg:"a filled value is uploaded" int (before + 2) !uploads;
  equal ~msg:"it lives on the device" placement on1 (Nx.placement z);
  is_true ~msg:"its view covers its storage"
    (match z with Nx_effect.Placed r -> Nx_effect.covers r | _ -> false);
  equal ~msg:"its elements" (array float_exact) (Array.make 6 0.0)
    (Nx.to_array z);
  let s = Nx.zeros_like (Nx.sum p) in
  equal ~msg:"a filled scalar is held" int (before + 2) !uploads;
  equal ~msg:"and lives on the device" placement on1 (Nx.placement s);
  equal ~msg:"the product" (array float_exact)
    [| 2.; 4.; 6.; 8.; 10.; 12. |]
    (Nx.to_array y)

(* Reads *)

let test_a_read_copies_what_it_reads () =
  let p = Nx.place on1 (m23 ()) in
  elements_read := 0;
  equal ~msg:"item" float_exact 6.0 (Nx.item [ 1; 2 ] p);
  equal ~msg:"one element moves" int 1 !elements_read;
  equal ~msg:"the value stays" placement on1 (Nx.placement p);
  elements_read := 0;
  ignore (Nx.to_array p);
  equal ~msg:"a whole read" int 6 !elements_read;
  raises_invalid (fun () -> Nx.data p);
  equal ~msg:"a strided window" (array float_exact)
    [| 1.; 4.; 2.; 5.; 3.; 6. |]
    (Nx.to_array (Nx.transpose p))

(* Views and cells *)

let test_views_share_the_cell () =
  let p = Nx.place on1 (m23 ()) in
  let before = !uploads in
  let v = Nx.transpose (Nx.slice [ Nx.R (0, 1) ] p) in
  equal ~msg:"a view moves nothing" int before !uploads;
  is_true ~msg:"one cell" (cell_of v == cell_of p);
  equal ~msg:"its elements" (array float_exact) [| 1.; 2.; 3. |] (Nx.to_array v);
  is_true ~msg:"a whole value is contiguous" (Nx.contiguous p == p);
  let c = Nx.contiguous v in
  is_true ~msg:"a window is copied" (cell_of c != cell_of p);
  (cell_of p).state <- Consumed { path = "0" };
  raises_invalid (fun () -> Nx.to_array v);
  raises_invalid (fun () -> Nx.add v v);
  equal ~msg:"the copy survives" (array float_exact) [| 1.; 2.; 3. |]
    (Nx.to_array c)

let test_several_devices () =
  let x = m23 () in
  let r = Nx.place (Nx.Placement.replicated [ dev1; dev2 ]) x in
  let s = Nx.place (Nx.Placement.sharded ~axis:0 [ dev1; dev2 ]) x in
  equal ~msg:"a split value's shape is the whole's" (array int) [| 2; 3 |]
    (Nx.shape s);
  equal ~msg:"a replicated view" (array int) [| 3; 2 |]
    (Nx.shape (Nx.transpose r));
  equal ~msg:"a split value is read whole" (array float_exact) (Nx.to_array x)
    (Nx.to_array s);
  let d = Nx.add r r in
  equal ~msg:"an operation on several devices reads to the host" placement
    Nx.Placement.host (Nx.placement d);
  equal ~msg:"its result" (array float_exact)
    [| 2.; 4.; 6.; 8.; 10.; 12. |]
    (Nx.to_array d);
  let p = Nx.place on1 x in
  raises_match
    (function
      | Invalid_argument msg -> String.ends_with ~suffix:"place one of them" msg
      | _ -> false)
    (fun () -> Nx.add r p);
  equal ~msg:"constants over several devices are host values"
    (array float_exact)
    [| 1.; 0.; 0.; 4.; 5.; 0. |]
    (Nx.to_array (Nx.tril r));
  equal ~msg:"and so are gathered rows" (array float_exact)
    [| 4.; 5.; 6.; 1.; 2.; 3. |]
    (Nx.to_array (Nx.slice [ Nx.L [ 1; 0 ] ] s))

let test_repeat_a_split_value () =
  let x = m23 () in
  let s = Nx.place (Nx.Placement.sharded ~axis:0 [ dev1; dev2 ]) x in
  List.iter
    (fun (what, f) ->
      equal ~msg:what (array float_exact)
        (Nx.to_array (f x))
        (Nx.to_array (f s)))
    [
      ("along the split axis", Nx.repeat ~axis:0 2);
      ("along the other axis", Nx.repeat ~axis:1 3);
      ("flattened", Nx.repeat 2);
    ]

let test_pp_split_on_axis_one () =
  let x = Nx.create Nx.float32 [| 2; 4 |] (Array.init 8 float_of_int) in
  let s = Nx.place (Nx.Placement.sharded ~axis:1 [ dev1; dev2 ]) x in
  equal ~msg:"pp reads the elements in C order" string (Nx.to_string x)
    (Nx.to_string s)

(* Moving split values *)

let x86 () = Nx.create Nx.float32 [| 8; 6 |] (Array.init 48 float_of_int)

let over_four ~axis x =
  Nx.place (Nx.Placement.sharded ~axis [ dev1; dev2; dev3; dev4 ]) x

let split ~axis = Nx.Placement.sharded ~axis [ dev1; dev2; dev3; dev4 ]

let raises_across ~what ~axis ~shape f =
  raises_match
    (function
      | Invalid_argument msg ->
          msg
          = Printf.sprintf
              "Nx: a %s of the split axis %d of shape %s would move elements \
               between devices; place the value replicated or on one device \
               first"
              what axis shape
      | _ -> false)
    f

let test_split_values_move_as_views () =
  let x = x86 () in
  let s = over_four ~axis:0 x in
  let uploaded = !uploads in
  elements_read := 0;
  let moved =
    [
      ("transpose", Nx.transpose s, Nx.transpose x, split ~axis:1);
      ( "reshape [4;2;6]",
        Nx.reshape [| 4; 2; 6 |] s,
        Nx.reshape [| 4; 2; 6 |] x,
        split ~axis:0 );
      ( "reshape [48]",
        Nx.reshape [| 48 |] s,
        Nx.reshape [| 48 |] x,
        split ~axis:0 );
      ( "reshape [1;8;6]",
        Nx.reshape [| 1; 8; 6 |] s,
        Nx.reshape [| 1; 8; 6 |] x,
        split ~axis:1 );
      ( "whole rows, two columns",
        Nx.slice [ Nx.R (0, 8); Nx.R (1, 3) ] s,
        Nx.slice [ Nx.R (0, 8); Nx.R (1, 3) ] x,
        split ~axis:0 );
      ( "flip columns",
        Nx.flip ~axes:[ 1 ] s,
        Nx.flip ~axes:[ 1 ] x,
        split ~axis:0 );
      ( "windows along columns",
        Nx.sliding_window ~axis:1 ~window:2 s,
        Nx.sliding_window ~axis:1 ~window:2 x,
        split ~axis:0 );
      ( "reversed columns of the transpose",
        Nx.flip ~axes:[ 0 ] (Nx.transpose (Nx.slice [ Nx.A; Nx.R (2, 5) ] s)),
        Nx.flip ~axes:[ 0 ] (Nx.transpose (Nx.slice [ Nx.A; Nx.R (2, 5) ] x)),
        split ~axis:1 );
    ]
  in
  equal ~msg:"no movement uploads" int uploaded !uploads;
  equal ~msg:"or reads" int 0 !elements_read;
  List.iter
    (fun (what, m, h, p) ->
      equal ~msg:(what ^ ": placement") placement p (Nx.placement m);
      is_true ~msg:(what ^ ": the source's cell") (cell_of m == cell_of s);
      equal ~msg:(what ^ ": shape") (array int) (Nx.shape h) (Nx.shape m);
      equal ~msg:(what ^ ": elements") (array float_exact) (Nx.to_array h)
        (Nx.to_array m))
    moved;
  equal ~msg:"printed in C order" string
    (Nx.to_string (Nx.transpose x))
    (Nx.to_string (Nx.transpose s))

let test_split_axis_one () =
  let x = x86 () in
  let s = Nx.place (Nx.Placement.sharded ~axis:1 [ dev1; dev2 ]) x in
  let t = Nx.reshape [| 8; 6; 1 |] s in
  equal ~msg:"a trailing unit axis keeps the split" placement
    (Nx.Placement.sharded ~axis:1 [ dev1; dev2 ])
    (Nx.placement t);
  equal ~msg:"its elements" (array float_exact) (Nx.to_array x) (Nx.to_array t);
  let row = Nx.create Nx.float32 [| 1; 8 |] (Array.init 8 float_of_int) in
  let e = Nx.broadcast_to [| 3; 8 |] (over_four ~axis:1 row) in
  equal ~msg:"a broadcast keeps the split" placement (split ~axis:1)
    (Nx.placement e);
  equal ~msg:"and repeats the row" (array float_exact)
    (Nx.to_array (Nx.broadcast_to [| 3; 8 |] row))
    (Nx.to_array e)

let test_moves_across_devices_raise () =
  let s = over_four ~axis:0 (x86 ()) in
  raises_across ~what:"reshape" ~axis:1 ~shape:"[6,8]" (fun () ->
      Nx.reshape [| 48 |] (Nx.transpose s));
  raises_across ~what:"reshape" ~axis:0 ~shape:"[8,6]" (fun () ->
      Nx.reshape [| 2; 24 |] s);
  raises_across ~what:"cut" ~axis:0 ~shape:"[8,6]" (fun () ->
      Nx.slice [ Nx.R (3, 5) ] s);
  raises_across ~what:"flip" ~axis:0 ~shape:"[8,6]" (fun () ->
      Nx.flip ~axes:[ 0 ] s);
  raises_across ~what:"window" ~axis:0 ~shape:"[8,6]" (fun () ->
      Nx.sliding_window ~axis:0 ~window:2 s)

let test_a_cut_inside_one_shard () =
  let x = x86 () in
  let s = over_four ~axis:0 x in
  let uploaded = !uploads in
  let row = Nx.slice [ Nx.I 5 ] s in
  equal ~msg:"no upload" int uploaded !uploads;
  equal ~msg:"on the shard's device" placement (Nx.Placement.device dev3)
    (Nx.placement row);
  is_true ~msg:"a view of the split storage" (cell_of row == cell_of s);
  equal ~msg:"its elements" (array float_exact)
    (Nx.to_array (Nx.slice [ Nx.I 5 ] x))
    (Nx.to_array row);
  elements_read := 0;
  equal ~msg:"item" float_exact 32.0 (Nx.item [ 5; 2 ] s);
  equal ~msg:"reads one element" int 1 !elements_read;
  let window = Nx.slice [ Nx.R (4, 6); Nx.R (1, 4) ] s in
  equal ~msg:"a whole shard's window" placement (Nx.Placement.device dev3)
    (Nx.placement window);
  equal ~msg:"its elements" (array float_exact)
    (Nx.to_array (Nx.slice [ Nx.R (4, 6); Nx.R (1, 4) ] x))
    (Nx.to_array window);
  (* Nx.to_array reads 2k until contiguous copies on the device (M2). *)
  elements_read := 0;
  ignore (Nx_effect.to_host window);
  equal ~msg:"a read of it reads its six elements" int 6 !elements_read;
  let t =
    Nx.transpose (Nx.place (Nx.Placement.sharded ~axis:1 [ dev1; dev2 ]) x)
  in
  equal ~msg:"a column of a value split by columns" (array float_exact)
    (Nx.to_array (Nx.slice [ Nx.I 4 ] (Nx.transpose x)))
    (Nx.to_array (Nx.slice [ Nx.I 4 ] t));
  equal ~msg:"lives on the second device" placement (Nx.Placement.device dev2)
    (Nx.placement (Nx.slice [ Nx.I 4 ] t))

let test_moved_views_of_a_consumed_split () =
  let s = over_four ~axis:0 (x86 ()) in
  let views =
    [
      Nx.transpose s;
      Nx.reshape [| 48 |] s;
      Nx.slice [ Nx.A; Nx.R (1, 3) ] s;
      Nx.flip ~axes:[ 1 ] s;
      Nx.slice [ Nx.I 5 ] s;
    ]
  in
  (cell_of s).state <- Consumed { path = "0" };
  List.iter (fun v -> raises_invalid (fun () -> Nx.to_array v)) views

let test_consumed_value_does_not_move () =
  let p = Nx.place on1 (m23 ()) in
  (cell_of p).state <- Consumed { path = "2.keys" };
  raises_match
    (function
      | Invalid_argument msg ->
          msg
          = "this value was consumed at 2.keys in a compiled call's arguments; \
             use the value the call returned"
      | _ -> false)
    (fun () -> Nx.place on1 p);
  raises_invalid (fun () -> Nx.to_array p);
  equal ~msg:"its shape stays readable" (array int) [| 2; 3 |] (Nx.shape p)

let tests =
  [
    group "placement"
      [
        test "placements are in normal form" test_normal_forms;
        test "a move keeps its source" test_place_keeps_its_source;
        test "a split must divide evenly" test_place_splits_evenly;
        test "results live with their operands"
          test_results_live_with_their_operands;
        test "scalars created on a device are held, filled values are not"
          test_created_scalars_are_held;
        test "a read copies what it reads" test_a_read_copies_what_it_reads;
        test "views share their cell" test_views_share_the_cell;
        test "several devices" test_several_devices;
        test "repeat a split value" test_repeat_a_split_value;
        test "pp of a value split on axis 1" test_pp_split_on_axis_one;
        test "split values move as views" test_split_values_move_as_views;
        test "a value split on axis 1 moves as a view" test_split_axis_one;
        test "a move across devices raises" test_moves_across_devices_raise;
        test "a cut inside one shard" test_a_cut_inside_one_shard;
        test "moved views of a consumed split value"
          test_moved_views_of_a_consumed_split;
        test "a consumed value is not placed" test_consumed_value_does_not_move;
      ];
  ]

let () = run "nx placement" tests
