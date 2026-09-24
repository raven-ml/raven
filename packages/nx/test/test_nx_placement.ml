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

type Nx_effect.storage +=
  | Mem : ('a, 'b) Nx_core.Dtype.t * ('a, 'b) Nx_buffer.t -> Nx_effect.storage

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
  dst

let rec engine =
  {
    Nx_effect.read =
      (fun (type a b) (r : (a, b) Nx_effect.resident) : (a, b) Nx_buffer.t ->
        match r.r_cell.state with
        | Live (Mem (dt, mem)) -> (
            (* A split value's storage here is the whole value in C order, and
               its view covers it. *)
            let v =
              match r.r_placement with
              | Sharded _ ->
                  Nx_core.View.create
                    (Nx_core.View.shape (Nx_effect.whole_view r))
              | Device _ | Replicated _ -> r.r_view
            in
            elements_read := !elements_read + Nx_core.View.numel v;
            match Nx_core.Dtype.equal_witness dt r.r_dtype with
            | Some Type.Equal -> gather mem v
            | None -> assert false)
        | _ -> assert false);
    place = (fun p x -> place p x);
  }

and place : type a b.
    Nx_effect.placement -> (a, b) Nx_effect.t -> (a, b) Nx_effect.t =
 fun p x ->
  if Nx.numel x > 0 then incr uploads;
  let src = Nx.to_buffer (Nx.place Nx.Placement.host x) in
  let mem = Nx_buffer.create (Nx_buffer.kind src) (Nx_buffer.length src) in
  Nx_buffer.blit ~src ~dst:mem;
  Nx_effect.placed p (Nx.dtype x)
    (Nx_core.View.create (Nx.shape x))
    (Nx_effect.cell engine ~length:(Nx_buffer.length mem)
       (Mem (Nx.dtype x, mem)))

let dev1 = Nx_effect.Device.make "TEST:1" engine
let dev2 = Nx_effect.Device.make "TEST:2" engine
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

(* A value split or replicated over [dev1; dev2], whose storage holds [x]. *)
let over_two placement x shard =
  Nx_effect.placed placement (Nx.dtype x)
    (Nx_core.View.create shard)
    (Nx_effect.cell engine
       ~length:(Array.fold_left ( * ) 1 shard)
       (Mem (Nx.dtype x, Nx.to_buffer x)))

let test_several_devices () =
  let x = m23 () in
  let r =
    Nx_effect.placed
      (Nx.Placement.replicated [ dev1; dev2 ])
      Nx.float32
      (Nx_core.View.create [| 2; 3 |])
      (Nx_effect.cell engine ~length:6 (Mem (Nx.float32, Nx.to_buffer x)))
  in
  let s =
    Nx_effect.placed
      (Nx.Placement.sharded ~axis:0 [ dev1; dev2 ])
      Nx.float32
      (Nx_core.View.create [| 1; 3 |])
      (Nx_effect.cell engine ~length:3 (Mem (Nx.float32, Nx.to_buffer x)))
  in
  equal ~msg:"a split value's shape is the whole's" (array int) [| 2; 3 |]
    (Nx.shape s);
  equal ~msg:"a replicated view" (array int) [| 3; 2 |]
    (Nx.shape (Nx.transpose r));
  equal ~msg:"a split value is read whole" (array float_exact) (Nx.to_array x)
    (Nx.to_array s);
  let t = Nx.transpose s and d = Nx.add r r in
  equal ~msg:"moving a split value reads it to the host" placement
    Nx.Placement.host (Nx.placement t);
  equal ~msg:"so does an operation on several devices" placement
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

let test_pp_split_on_axis_one () =
  let x = Nx.create Nx.float32 [| 2; 4 |] (Array.init 8 float_of_int) in
  let s = over_two (Nx.Placement.sharded ~axis:1 [ dev1; dev2 ]) x [| 2; 2 |] in
  equal ~msg:"pp reads the elements in C order" string (Nx.to_string x)
    (Nx.to_string s)

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
        test "pp of a value split on axis 1" test_pp_split_on_axis_one;
        test "a consumed value is not placed" test_consumed_value_does_not_move;
      ];
  ]

let () = run "nx placement" tests
