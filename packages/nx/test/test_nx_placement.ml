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

(* One buffer per device of the cell's placement, in its order: a split value's
   shards, a replicated value's copies. *)
type Nx_effect.storage +=
  | Mem : ('a, 'b) Nx_dtype.t * ('a, 'b) Nx_buffer.t list -> Nx_effect.storage

let elements_read = ref 0
let uploads = ref 0

(* The elements view [v] reaches in [mem], in C order. *)
let gather (type a b) (mem : (a, b) Nx_buffer.t) v : (a, b) Nx_buffer.t =
  let shape = Nx_core.View.shape v and strides = Nx_core.View.strides v in
  let n = Nx_core.View.numel v in
  let dst = Nx_buffer.create (Nx_buffer.dtype mem) n in
  for i = 0 to n - 1 do
    let idx = Nx_core.Shape.unravel_index i shape in
    let off = ref (Nx_core.View.offset v) in
    Array.iteri (fun d k -> off := !off + (k * strides.(d))) idx;
    Nx_buffer.set dst i (Nx_buffer.get mem !off)
  done;
  elements_read := !elements_read + n;
  dst

let rec engine =
  {
    Nx_effect.read =
      (fun (type a b) (r : (a, b) Nx_effect.resident) : (a, b) Nx_buffer.t ->
        match r.r_cell.state with
        | Live (Mem (dt, shards)) -> (
            match Nx_dtype.equal_witness dt r.r_dtype with
            | Some Type.Equal ->
                let shape =
                  Nx_effect.global r.r_placement (Nx_core.View.shape r.r_view)
                in
                Nx_effect.assemble r
                  (Array.map (fun n -> (0, n)) shape)
                  (fun d v ->
                    let devices = Nx.Placement.devices r.r_cell.placement in
                    let k = Option.get (List.find_index (( == ) d) devices) in
                    gather (List.nth shards k) v)
            | None -> assert false)
        | _ -> assert false);
    place = (fun p x -> place p x);
  }

and place : type a b.
    Nx_effect.placement -> (a, b) Nx_effect.t -> (a, b) Nx_effect.t =
 fun p x ->
  if Nx.numel x > 0 then incr uploads;
  let h = Nx.place Nx.Placement.host x in
  let devices = Nx.Placement.devices p in
  let windows = List.map (Nx.Placement.window p (Nx.shape h)) devices in
  let shards =
    List.map (fun w -> Nx.to_buffer (Nx.copy (Nx.shrink w h))) windows
  in
  let shape = Array.map (fun (lo, hi) -> hi - lo) (List.hd windows) in
  Nx_effect.placed p (Nx.dtype x)
    (Nx_core.View.create shape)
    (Nx_effect.cell ~placement:p
       ~length:(Array.fold_left ( * ) 1 shape)
       (Mem (Nx.dtype x, shards)))

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
  is_true ~msg:"copies are equal in any order"
    (Nx.Placement.equal
       (Nx.Placement.replicated [ dev1; dev2 ])
       (Nx.Placement.replicated [ dev2; dev1 ]));
  is_false ~msg:"slices are not"
    (Nx.Placement.equal
       (Nx.Placement.sharded ~axis:0 [ dev1; dev2 ])
       (Nx.Placement.sharded ~axis:0 [ dev2; dev1 ]));
  raises_invalid (fun () -> Nx.Placement.replicated []);
  raises_invalid (fun () -> Nx.Placement.replicated [ dev1; dev1 ]);
  raises_invalid (fun () -> Nx.Placement.sharded ~axis:0 [ dev1; other ]);
  raises_invalid (fun () -> Nx.Placement.sharded ~axis:(-1) [ dev1; dev2 ])

let windows = Testable.(array (pair int int))

let test_tiles () =
  let s = Nx.Placement.sharded ~axis:1 [ dev1; dev2; dev3 ] in
  equal ~msg:"the second slice" windows
    [| (0, 2); (2, 4) |]
    (Nx.Placement.window s [| 2; 6 |] dev2);
  equal ~msg:"a copy is the whole" windows
    [| (0, 2); (0, 6) |]
    (Nx.Placement.window
       (Nx.Placement.replicated [ dev1; dev2 ])
       [| 2; 6 |] dev2);
  raises_invalid (fun () -> Nx.Placement.window s [| 2; 6 |] dev4);
  raises_invalid (fun () -> Nx.Placement.window s [| 2; 5 |] dev1);
  raises_match
    (function
      | Invalid_argument msg ->
          msg = "Nx.Placement.window: shape [6] has no axis 1 to split"
      | _ -> false)
    (fun () -> Nx.Placement.window s [| 6 |] dev1)

(* [equal] agrees with comparing every device's window, over random placements
   of one dimension and their reorderings. The windows come from the
   constructors' arguments, and [window] is checked against them too. *)
let test_equal_is_the_window_map () =
  let rng = Random.State.make [| 5 |] in
  let pool = [| dev1; dev2; dev3; dev4 |] in
  let shape = [| 12; 12; 12 |] in
  let shuffle l =
    List.map snd
      (List.sort compare (List.map (fun d -> (Random.State.bits rng, d)) l))
  in
  (* A placement's arguments: its devices and its split axis, if any. *)
  let make (ds, axis) =
    match axis with
    | None -> Nx.Placement.replicated ds
    | Some axis -> Nx.Placement.sharded ~axis ds
  in
  let expected (ds, axis) d =
    let whole = Array.map (fun n -> (0, n)) shape in
    match (List.find_index (( == ) d) ds, axis) with
    | None, _ -> None
    | Some _, None -> Some whole
    | Some i, Some a ->
        let k = shape.(a) / List.length ds in
        whole.(a) <- (i * k, (i + 1) * k);
        Some whole
  in
  let random () =
    let n = 1 + Random.State.int rng 4 in
    let ds = List.filteri (fun i _ -> i < n) (shuffle (Array.to_list pool)) in
    (ds, if Random.State.bool rng then None else Some (Random.State.int rng 3))
  in
  let agreed = ref 0 in
  for _ = 1 to 2000 do
    let a = random () in
    let b =
      if Random.State.bool rng then random ()
      else
        let ds, axis = a in
        ( shuffle ds,
          if Random.State.bool rng then axis else Some (Random.State.int rng 3)
        )
    in
    let p = make a and q = make b in
    List.iter
      (fun d ->
        equal ~msg:"a window" (option windows) (expected a d)
          (Some (Nx.Placement.window p shape d)))
      (Nx.Placement.devices p);
    let same = Array.for_all (fun d -> expected a d = expected b d) pool in
    if same then incr agreed;
    equal
      ~msg:(Format.asprintf "%a and %a" Nx.Placement.pp p Nx.Placement.pp q)
      bool same (Nx.Placement.equal p q)
  done;
  is_true ~msg:"some pairs are equal" (!agreed > 100)

(* A grid, built inside nx.effect only, is kept in normal form and compared by
   its windows. *)
let test_grids () =
  let ds = [ dev1; dev2; dev3; dev4 ] in
  let grid = Nx_effect.Grid.v ds in
  equal ~msg:"cut over both axes in order is the flat split" placement
    (Nx.Placement.sharded ~axis:0 ds)
    (grid [ 2; 2 ] [ (0, [ 0; 1 ]) ]);
  equal ~msg:"no cut is the copies" placement
    (Nx.Placement.replicated ds)
    (grid [ 2; 2 ] []);
  equal ~msg:"extents of one go" placement
    (Nx.Placement.sharded ~axis:1 ds)
    (grid [ 1; 4; 1 ] [ (1, [ 1 ]) ]);
  equal ~msg:"cut minor first is the split in column order" placement
    (Nx.Placement.sharded ~axis:0 [ dev1; dev3; dev2; dev4 ])
    (grid [ 2; 2 ] [ (0, [ 1; 0 ]) ]);
  let two = grid [ 2; 2 ] [ (0, [ 0 ]); (1, [ 1 ]) ] in
  equal ~msg:"a device's window under two cuts" windows
    [| (2, 4); (0, 3) |]
    (Nx.Placement.window two [| 4; 6 |] dev3);
  equal ~msg:"half a grid holds copies" windows
    [| (0, 2); (0, 6) |]
    (Nx.Placement.window (grid [ 2; 2 ] [ (0, [ 0 ]) ]) [| 4; 6 |] dev2);
  raises_invalid (fun () -> grid [ 2; 3 ] []);
  raises_invalid (fun () -> grid [ 2; 2 ] [ (0, [ 0 ]); (1, [ 0 ]) ])

(* A value cut along two axes moves by the whole shapes, as tolk's rewrite
   decides it: no two cuts land on one axis. *)
let test_two_cuts_move () =
  let p =
    Nx_effect.Grid.v [ dev1; dev2; dev3; dev4 ] [ 2; 2 ]
      [ (0, [ 0 ]); (1, [ 1 ]) ]
  in
  let x = Nx.arange Nx.int32 0 8 1 |> Nx.reshape [| 2; 4 |] in
  let s = Nx.place p x in
  let r = Nx.reshape [| 2; 2; 2 |] s in
  equal ~msg:"both cuts survive" placement
    (Nx_effect.Grid.v [ dev1; dev2; dev3; dev4 ] [ 2; 2 ]
       [ (0, [ 0 ]); (1, [ 1 ]) ])
    (Nx.placement r);
  equal ~msg:"its elements" (array int32)
    (Nx.to_array (Nx.reshape [| 2; 2; 2 |] x))
    (Nx.to_array r);
  raises_invalid (fun () -> Nx.reshape [| 8 |] s);
  let row = Nx.slice [ Nx.I 1 ] s in
  equal ~msg:"a row keeps the devices holding it" placement
    (Nx.Placement.sharded ~axis:0 [ dev3; dev4 ])
    (Nx.placement row);
  equal ~msg:"the row" (array int32) [| 4l; 5l; 6l; 7l |] (Nx.to_array row)

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
  equal ~msg:"an operation on copies gives copies" placement (Nx.placement r)
    (Nx.placement d);
  equal ~msg:"its result" (array float_exact)
    [| 2.; 4.; 6.; 8.; 10.; 12. |]
    (Nx.to_array d);
  let p = Nx.place on1 x in
  raises_match
    (function
      | Invalid_argument msg -> String.ends_with ~suffix:"place one of them" msg
      | _ -> false)
    (fun () -> Nx.add r p);
  let t = Nx.tril r in
  equal ~msg:"constants over several devices are copies" placement
    (Nx.placement r) (Nx.placement t);
  equal ~msg:"and the result is right" (array float_exact)
    [| 1.; 0.; 0.; 4.; 5.; 0. |]
    (Nx.to_array t);
  equal ~msg:"gathered rows of copies" (array float_exact)
    [| 4.; 5.; 6.; 1.; 2.; 3. |]
    (Nx.to_array (Nx.slice [ Nx.L [ 1; 0 ] ] r))

(* An eager result over split operands takes the placement tolk's rewrite gives
   the same operation in a compiled program. *)
let test_eager_results_over_split_operands () =
  let ds = [ dev1; dev2 ] in
  let rows = Nx.Placement.sharded ~axis:0 ds
  and cols = Nx.Placement.sharded ~axis:1 ds
  and copies = Nx.Placement.replicated ds in
  let x = Nx.reshape [| 8; 6 |] (Nx.arange Nx.float32 0 48 1) in
  let w = Nx.reshape [| 6; 4 |] (Nx.arange Nx.float32 0 24 1) in
  let s = Nx.place rows x and t = Nx.place cols x and r = Nx.place copies x in
  let check msg p expected y =
    equal ~msg:(msg ^ ": placement") placement p (Nx.placement y);
    equal ~msg (array float_exact) (Nx.to_array expected) (Nx.to_array y)
  in
  check "split + split" rows (Nx.add x x) (Nx.add s s);
  check "split + copies" rows (Nx.add x x) (Nx.add s r);
  check "copies + split" rows (Nx.add x x) (Nx.add r s);
  check "split + host" rows (Nx.add x x) (Nx.add s x);
  check "split + scalar" rows (Nx.add_s x 1.) (Nx.add_s s 1.);
  check "a unary operation" rows (Nx.exp x) (Nx.exp s);
  check "a comparison" rows
    (Nx.cast Nx.float32 (Nx.less x (Nx.flip x)))
    (Nx.cast Nx.float32
       (Nx.less s (Nx.flip ~axes:[ 1 ] (Nx.flip ~axes:[ 0 ] x))));
  check "a reduction over the split axis" copies (Nx.sum ~axes:[ 0 ] x)
    (Nx.sum ~axes:[ 0 ] s);
  check "a reduction over the other axis" rows (Nx.sum ~axes:[ 1 ] x)
    (Nx.sum ~axes:[ 1 ] s);
  check "a reduction before the split axis"
    (Nx.Placement.sharded ~axis:0 ds)
    (Nx.max ~axes:[ 0 ] x) (Nx.max ~axes:[ 0 ] t);
  check "keeping its axes" cols
    (Nx.sum ~axes:[ 0 ] ~keepdims:true x)
    (Nx.sum ~axes:[ 0 ] ~keepdims:true t);
  check "a sum of everything" copies (Nx.sum x) (Nx.sum s);
  check "an operation along the other axis" rows (Nx.cumsum ~axis:1 x)
    (Nx.cumsum ~axis:1 s);
  check "a sort along the other axis" rows
    (Nx.sort ~axis:1 x |> fst)
    (Nx.sort ~axis:1 s |> fst);
  check "rows times copies" rows (Nx.matmul x w)
    (Nx.matmul s (Nx.place copies w));
  check "copies times columns" cols (Nx.matmul x w)
    (Nx.matmul r (Nx.place cols w));
  check "columns times rows" copies (Nx.matmul x w)
    (Nx.matmul t (Nx.place rows w));
  check "a pad of the other axis" rows
    (Nx.pad [| (0, 0); (1, 1) |] 0. x)
    (Nx.pad [| (0, 0); (1, 1) |] 0. s);
  check "a concatenation along the other axis" rows
    (Nx.concatenate ~axis:1 [ x; x ])
    (Nx.concatenate ~axis:1 [ s; s ]);
  check "a copy" rows (Nx.copy x) (Nx.copy s);
  let along f =
    raises_match
      (function
        | Invalid_argument msg ->
            String.ends_with
              ~suffix:"place the value replicated or on one device first" msg
        | _ -> false)
      f
  in
  along (fun () -> Nx.cumsum ~axis:0 s);
  along (fun () -> Nx.sort ~axis:0 s);
  along (fun () -> Nx.pad [| (1, 1); (0, 0) |] 0. s);
  along (fun () -> Nx.concatenate ~axis:0 [ s; s ]);
  check "rows gathered from columns" cols
    (Nx.slice [ Nx.L [ 1; 0 ] ] x)
    (Nx.slice [ Nx.L [ 1; 0 ] ] t);
  along (fun () -> Nx.slice [ Nx.L [ 1; 0 ] ] s);
  let batch = Nx.reshape [| 2; 4; 4 |] (Nx.arange Nx.float32 0 32 1) in
  let spd =
    Nx.add
      (Nx.matmul batch (Nx.transpose ~axes:[ 0; 2; 1 ] batch))
      (Nx.mul_s (Nx.eye Nx.float32 4) 10.)
  in
  check "linear algebra over a split batch" rows (Nx.cholesky spd)
    (Nx.cholesky (Nx.place rows spd));
  along (fun () -> Nx.cholesky (Nx.place cols spd));
  raises_match
    (function
      | Invalid_argument msg ->
          String.ends_with
            ~suffix:"are split differently; place them alike first" msg
      | _ -> false)
    (fun () -> Nx.mul s t)

(* Whole shards of one storage, each a view on its device, combine as copies on
   their devices, as a compiled program copies a whole shard to every device. *)
let test_whole_shards_combine () =
  let x = m23 () |> Nx.reshape [| 6; 1 |] in
  let s = Nx.place (Nx.Placement.sharded ~axis:0 [ dev1; dev2 ]) x in
  let rolled = Nx.roll ~axis:0 3 s in
  equal ~msg:"a roll by one shard" placement
    (Nx.Placement.replicated [ dev1; dev2 ])
    (Nx.placement rolled);
  equal ~msg:"its elements" (array float_exact)
    (Nx.to_array (Nx.roll ~axis:0 3 x))
    (Nx.to_array rolled);
  let top = Nx.slice [ Nx.R (0, 3) ] s
  and bottom = Nx.slice [ Nx.R (3, 6) ] s in
  equal ~msg:"a sum of two shards" (array float_exact) [| 5.; 7.; 9. |]
    (Nx.to_array (Nx.add top bottom));
  raises_invalid (fun () -> Nx.roll ~axis:0 1 s);
  raises_invalid (fun () -> Nx.add (Nx.slice [ Nx.R (0, 2) ] s) bottom);
  raises_invalid (fun () ->
      Nx.add (Nx.broadcast_to [| 3; 1 |] (Nx.slice [ Nx.R (0, 1) ] s)) bottom);
  raises_invalid (fun () ->
      Nx.add top (Nx.place on2 (Nx.slice [ Nx.R (3, 6) ] x)));
  (* Over four devices, the copies are on all four, where a value on all of them
     joins them. *)
  let four = [ dev1; dev2; dev3; dev4 ] in
  let y = Nx.reshape [| 8; 1 |] (Nx.arange Nx.float32 0 8 1) in
  let s4 = Nx.place (Nx.Placement.sharded ~axis:0 four) y in
  let pair =
    Nx.add (Nx.slice [ Nx.R (0, 2) ] s4) (Nx.slice [ Nx.R (2, 4) ] s4)
  in
  equal ~msg:"two shards of four" placement
    (Nx.Placement.replicated four)
    (Nx.placement pair);
  equal ~msg:"their sum" (array float_exact) [| 2.; 4. |] (Nx.to_array pair);
  equal ~msg:"meets a value on all four" placement
    (Nx.Placement.replicated four)
    (Nx.placement
       (Nx.add pair
          (Nx.place
             (Nx.Placement.replicated four)
             (Nx.ones_like (Nx.slice [ Nx.R (0, 2) ] y)))));
  raises_invalid (fun () -> Nx.roll ~axis:0 2 s4)

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

let test_a_view_off_its_storage_raises () =
  let s = over_four ~axis:0 (x86 ()) in
  let r =
    match s with Nx_effect.Placed r -> r | _ -> fail "expected placed"
  in
  raises_invalid (fun () ->
      Nx_effect.placed (Nx.Placement.device other) r.r_dtype r.r_view r.r_cell);
  is_true ~msg:"a device of its storage"
    (match
       Nx_effect.placed (Nx.Placement.device dev2) r.r_dtype r.r_view r.r_cell
     with
    | Nx_effect.Placed _ -> true
    | _ -> false)

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
        test "a placement's tiles" test_tiles;
        test "equality is the window map" test_equal_is_the_window_map;
        test "grids" test_grids;
        test "a value cut twice moves by the whole shapes" test_two_cuts_move;
        test "a move keeps its source" test_place_keeps_its_source;
        test "a split must divide evenly" test_place_splits_evenly;
        test "results live with their operands"
          test_results_live_with_their_operands;
        test "scalars created on a device are held, filled values are not"
          test_created_scalars_are_held;
        test "a read copies what it reads" test_a_read_copies_what_it_reads;
        test "views share their cell" test_views_share_the_cell;
        test "several devices" test_several_devices;
        test "eager results over split operands"
          test_eager_results_over_split_operands;
        test "whole shards combine as copies" test_whole_shards_combine;
        test "repeat a split value" test_repeat_a_split_value;
        test "pp of a value split on axis 1" test_pp_split_on_axis_one;
        test "split values move as views" test_split_values_move_as_views;
        test "a value split on axis 1 moves as a view" test_split_axis_one;
        test "a move across devices raises" test_moves_across_devices_raise;
        test "a cut inside one shard" test_a_cut_inside_one_shard;
        test "a view off its storage's devices raises"
          test_a_view_off_its_storage_raises;
        test "moved views of a consumed split value"
          test_moved_views_of_a_consumed_split;
        test "a consumed value is not placed" test_consumed_value_does_not_move;
      ];
  ]

let () = run "nx placement" tests
