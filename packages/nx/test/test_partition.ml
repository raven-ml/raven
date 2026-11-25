(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.Prim.partition on an engine whose steps are nx's own operations on one
   device: eager's bits, placements and refusals, the steps a check and empty
   windows take, and offset and pieces, which an engine's steps read. *)

open Windtrap
module A = Nx_array
module L = Nx_array.Layout
module E = Nx_kernel.Prog.E

let f32 = Nx.float32
let m = Nx_support.memory

module Two = (val Nx.devices [ m 0; m 1 ])
module Three = (val Nx.devices [ m 0; m 1; m 2 ])
module Four = (val Nx.devices [ m 0; m 1; m 2; m 3 ])

(* The engine *)

let by = "test_partition"

let device_of p =
  let s = Nx.Placement.devices p in
  List.find
    (fun r -> Nx.Placement.equal p (Nx.Placement.device s r))
    (Nx.Devices.to_list s)

let device_name r =
  let k = List.find (fun k -> Rig.equal r (m k)) [ 0; 1; 2; 3 ] in
  Printf.sprintf "m%d" k

let window x (w : Nx_array.Move.range array) =
  Nx.Prim.eval ~by (Move (Slice (Iarray.of_array w), x))

let put at x = Nx.Prim.eval ~by (Place (at, x))

let empty (type v s e) (at : e Nx.Placement.t) (dt : (v, s) Nx.dtype) shape :
    (v, s) Nx_array.t =
  Nx_array.create (device_of at) dt (Iarray.of_array shape)

(* What [run] received: the operation, the device it ran on, and for a check its
   number of flags and the devices its data lie on. *)
type receipt = {
  name : string;
  device : string;
  flags : int;
  data_on : string list;
}

let pp_receipt ppf r =
  match r.name with
  | "Check" ->
      Format.fprintf ppf "Check of %d flags on %s, data on [%s]" r.flags
        r.device
        (String.concat "; " r.data_on)
  | name -> Format.fprintf ppf "%s on %s" name r.device

let receipt_t = Testable.make ~pp:pp_receipt ~equal:( = )

let rec data_on : type d e. (d, e) Nx.Prim.data -> string list = function
  | [] -> []
  | x :: rest -> device_name (device_of (Nx.placement x)) :: data_on rest

let receipt (type r) here (op : r Nx.Prim.t) =
  let device = device_name (device_of here) in
  match op with
  | Check { ok; data; _ } ->
      { name = "Check"; device; flags = Nx.numel ok; data_on = data_on data }
  | _ -> { name = Nx.Prim.name op; device; flags = 0; data_on = [] }

(* An engine of nx's operations on one device that logs what [run] receives in
   [ran], newest first. *)
let engine ran : Nx.Prim.engine =
  {
    run =
      (fun ~by ?into:_ ~origin here _ op ->
        ran := receipt here op :: !ran;
        Nx.Prim.eval ~by (Nx.Prim.offset op origin));
    piece =
      (fun x j _ ->
        let s = Nx.Placement.devices (Nx.placement x) in
        Nx.Repr.of_array s (Iarray.get (Nx.Repr.shards ~by:"test" x) j));
    view = (fun mv x -> Nx.Prim.eval ~by (Move (mv, x)));
    bring =
      (fun at ps ->
        let d = device_of at in
        let here (x, _) = Rig.equal (device_of (Nx.placement x)) d in
        let x, w =
          match List.find_opt here ps with Some p -> p | None -> List.hd ps
        in
        put at (window x w));
    gather =
      (fun at dt s boxes ->
        let pieces =
          List.map
            (fun (w, x, b) -> (Iarray.of_array w, put at (window x b)))
            boxes
        in
        Nx.Prim.eval ~by
          (Assemble { dtype = dt; shape = s; fill = Nx.Dtype.zero dt; pieces }));
    assemble =
      (fun p s dt ps ->
        let windows = Nx.Prim.pieces p s in
        let array j =
          match ps.(j) with
          | Some v -> Option.get (Nx.Repr.array ~by:"test" v)
          | None ->
              let here, w = windows.(j) in
              empty here dt
                (Array.map (fun (r : Nx_array.Move.range) -> r.count) w)
        in
        Nx.Repr.of_shards p (Iarray.init (Array.length windows) array));
  }

(* [f ()] with every operation computed as its partition on [e]. *)
let partitioned_on e f =
  let rule _ ~by op = Nx.Prim.partition e ~by op in
  Nx.Prim.interpret ~name:"partitioned" Extent rule (fun _ -> f ())

(* [f ()] partitioned on [engine ran]. *)
let partitioned ran f = partitioned_on (engine ran) f

(* Cases *)

type cut = On | Split of int | Device of int

let pp_cut ppf = function
  | On -> Format.pp_print_string ppf "on"
  | Split a -> Format.fprintf ppf "split ~axis:%d" a
  | Device k -> Format.fprintf ppf "device %d" k

let at (type d) (s : d Nx.Devices.t) c : d Nx.Placement.t =
  match c with
  | On -> Nx.Placement.on s
  | Split axis -> Nx.Placement.split ~axis s
  | Device k -> Nx.Placement.device s (m k)

type case = { n : int; rows : int; cols : int; x : cut; y : cut }

let pp_case ppf c =
  Format.fprintf ppf "%d devices, %d x %d, x %a, y %a" c.n c.rows c.cols pp_cut
    c.x pp_cut c.y

(* Each operand is whole on each device 1 time in 6, on one device 1 time in
   6, and split along its rows or its columns 1 time in 3 each. Of [x]'s
   splits along its rows, about 0.37 have fewer rows than devices, 0.37
   uneven windows and 0.26 even ones, before [Gen.int_range]'s draws of its
   bounds and their neighbours: even windows are about 0.078 of the cases,
   the rarest class the law covers, so it draws 300 cases, which miss them
   with probability 2 10^-11. *)
let drawn =
  let open Gen in
  let cut n =
    frequency
      [
        (1, constant On);
        (4, map (fun a -> Split a) (int_range 0 1));
        (1, map (fun k -> Device k) (int_range 0 (n - 1)));
      ]
  in
  (let* n = int_range 2 4 in
   let* rows = int_range 0 ((2 * n) + 1)
   and+ cols = int_range 1 ((2 * n) + 1) in
   let* x = cut n and+ y = cut n in
   constant { n; rows; cols; x; y })
  |> with_pp pp_case

let value ?(seed = 0) shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create f32 shape
    (Array.init n (fun i -> sin (float_of_int ((seed * 7919) + i + 1))))

type result = Raised of string | Bits of string * int32 array

let result f =
  match f () with
  | y ->
      Bits
        ( Format.asprintf "%a %a" Nx.Placement.pp (Nx.placement y) Nx.pp_shape
            (Nx.shape y),
          Array.map Int32.bits_of_float (Nx.to_array (Nx.cast f32 y)) )
  | exception Invalid_argument m -> Raised m

let result_t =
  Testable.make
    ~pp:(fun ppf -> function
      | Raised m -> Format.fprintf ppf "raised %s" m
      | Bits (p, b) ->
          Format.fprintf ppf "%s [%a]" p
            (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf b ->
                 Format.fprintf ppf "%lx" b))
            (Array.to_list b))
    ~equal:( = )

(* A check's failure: the first false flag's index and the datum's bits
   there. *)
exception Failed of int array * int32

type checked = Passed | Failed_at of int array * int32 | Refused of string

let pp_checked ppf = function
  | Passed -> Format.pp_print_string ppf "passed"
  | Failed_at (i, b) ->
      Format.fprintf ppf "failed at %a holding %lx" Nx_support.pp_ints i b
  | Refused m -> Format.fprintf ppf "refused: %s" m

let checked_t = Testable.make ~pp:pp_checked ~equal:( = )

(* A check that [x]'s elements are positive, [x] its datum, then a wait. *)
let check_positive x () =
  Nx.check Nx.Ptree.tensor
    (Nx.less (Nx.zeros_like x) x)
    x
    (fun i d ->
      Failed (i, Int32.bits_of_float (Nx.item [] (Nx.place Nx.Host.on d))));
  Nx.wait_checks ()

let checked f =
  match f () with
  | () -> Passed
  | exception Failed (i, b) -> Failed_at (i, b)
  | exception Invalid_argument m -> Refused m

(* The first device, in placement order, whose window of a value of [shape] at
   [p] holds an element. *)
let first_holder p shape =
  Array.to_list (Nx.Prim.pieces p shape)
  |> List.find_map (fun (here, (w : Nx_array.Move.range array)) ->
      if Array.for_all (fun (r : Nx_array.Move.range) -> r.count > 0) w then
        Some (device_name (device_of here))
      else None)
  |> Option.get

(* What [run] receives of a check of [x]'s positive elements: one check of one
   flag on the first device that holds that flag, its data placed there. A check
   of one flag is itself, of its one datum; one of several flags checks their
   0-d conjunction, which lies as [Nx.min] of the flags does, of the first false
   flag's position and the datum's element there. *)
let check_receipt x =
  let ok = Nx.less (Nx.zeros_like x) x in
  if Nx.numel ok = 1 then
    let device = first_holder (Nx.placement ok) (Nx.shape ok) in
    { name = "Check"; device; flags = 1; data_on = [ device ] }
  else
    let device = first_holder (Nx.placement (Nx.min ok)) [||] in
    { name = "Check"; device; flags = 1; data_on = [ device; device ] }

(* Each function of the case's operands over [s], eagerly and partitioned, and a
   check of [x]. *)
let law_on (type d) (s : d Nx.Devices.t) c =
  let x = Nx.place (at s c.x) (value [| c.rows; c.cols |]) in
  let y = Nx.place (at s c.y) (value ~seed:1 [| c.rows; c.cols |]) in
  let w = Nx.place (at s c.y) (value ~seed:2 [| c.cols; 3 |]) in
  let uncut v = Nx.place (Nx.Placement.uncut (Nx.placement x)) v in
  let idx =
    uncut (Nx.create Nx.int64 [| 2 |] [| 0L; Int64.of_int (c.cols - 1) |])
  in
  let ends =
    uncut
      (Nx.create Nx.int64 [| c.rows; 2 |]
         (Array.init (2 * c.rows) (fun k ->
              if k mod 2 = 0 then 0L else Int64.of_int (c.cols - 1))))
  in
  let updates = uncut (value ~seed:3 [| c.rows; 2 |]) in
  let fs : (string * (unit -> (float, Nx.float32_elt, d) Nx.t)) list =
    [
      ("mul", fun () -> Nx.mul x y);
      ("sum rows", fun () -> Nx.sum ~axes:[ 0 ] x);
      ("sum columns", fun () -> Nx.sum ~axes:[ 1 ] x);
      ("max", fun () -> Nx.max x);
      ("var rows", fun () -> Nx.var ~axes:[ 0 ] x);
      ("logsumexp columns", fun () -> Nx.logsumexp ~axes:[ 1 ] x);
      ("cumsum", fun () -> Nx.cumsum ~axis:0 x);
      ("cumprod", fun () -> Nx.cumprod ~axis:1 x);
      ("matmul", fun () -> Nx.matmul x w);
      ("take", fun () -> Nx.take ~axis:1 idx x);
      ("scatter", fun () -> Nx.scatter ~axis:1 ends updates x);
      ("sort columns", fun () -> fst (Nx.sort ~axis:1 x));
      ("concatenate", fun () -> Nx.concatenate ~axis:0 [ x; y ]);
      ("transpose", fun () -> Nx.transpose x);
      ("reshape", fun () -> Nx.reshape [| c.rows * c.cols |] x);
      ("copy", fun () -> Nx.copy x);
      ("bitcast", fun () -> Nx.cast f32 (Nx.bitcast Nx.int32 x));
      (* Coordinates: each device reads its window's from its start. *)
      ("argmax rows", fun () -> Nx.cast f32 (Nx.argmax ~axis:0 x));
      ("argmax columns", fun () -> Nx.cast f32 (Nx.argmax ~axis:1 x));
      ("tril", fun () -> Nx.tril x);
      ("eye", fun () -> Nx.eye_at ~m:c.cols (Nx.placement x) f32 c.rows);
      ("place", fun () -> Nx.place (Nx.Placement.split ~axis:1 s) x);
    ]
  in
  let eager = List.map (fun (_, f) -> result f) fs in
  let parted =
    List.map (fun (_, f) -> result (fun () -> partitioned (ref []) f)) fs
  in
  let check_eager = checked (check_positive x) in
  let ran = ref [] in
  let check_parted = checked (fun () -> partitioned ran (check_positive x)) in
  let check_runs = List.filter (fun r -> r.name = "Check") !ran in
  let cut_rows = function Split 0 -> true | _ -> false in
  cover "an empty window" (cut_rows c.x && c.rows < c.n);
  cover "uneven windows" (cut_rows c.x && c.rows > c.n && c.rows mod c.n <> 0);
  cover "even windows" (cut_rows c.x && c.rows >= c.n && c.rows mod c.n = 0);
  cover "x cut along its columns" (c.x = Split 1);
  cover "x whole on each device" (c.x = On);
  cover "x on one device" (match c.x with Device _ -> true | _ -> false);
  cover "a refusal"
    (List.exists (function Raised _ -> true | _ -> false) eager);
  cover "a failing check"
    (match check_eager with Failed_at _ -> true | _ -> false);
  cover "a check of no flag" (Nx.numel x = 0);
  cover "a check of one flag" (Nx.numel x = 1);
  List.iter2
    (fun (name, _) (e, p) -> equal ~msg:name result_t e p)
    fs
    (List.combine eager parted);
  equal ~msg:"the check" checked_t check_eager check_parted;
  if Nx.numel x = 0 then
    equal ~msg:"a check of no flag runs nothing" (list receipt_t) [] check_runs
  else
    equal ~msg:"what run receives of the check" (list receipt_t)
      [ check_receipt x ]
      check_runs

let law c =
  cover "two devices" (c.n = 2);
  cover "three devices" (c.n = 3);
  cover "four devices" (c.n = 4);
  match c.n with
  | 2 -> law_on Two.v c
  | 3 -> law_on Three.v c
  | _ -> law_on Four.v c

(* The devices that run a product of values split along their rows: those whose
   window holds a row, or the first where none does. *)
let test_empty_windows () =
  let p = Four.split ~axis:0 in
  let runs rows =
    let x = Nx.place p (value [| rows; 2 |]) in
    let y = Nx.place p (value ~seed:1 [| rows; 2 |]) in
    let ran = ref [] in
    ignore (partitioned ran (fun () -> Nx.mul x y));
    let holding =
      Array.to_list (Nx.Prim.pieces p [| rows; 2 |])
      |> List.filter_map (fun (here, (w : Nx_array.Move.range array)) ->
          if w.(0).count > 0 then Some (device_name (device_of here)) else None)
    in
    equal
      ~msg:(Printf.sprintf "%d rows" rows)
      (list string)
      (if holding = [] then [ "m0" ] else holding)
      (List.sort_uniq compare (List.map (fun r -> r.device) !ran))
  in
  runs 3;
  runs 0

let test_step_raises () =
  let x = Nx.place (Two.split ~axis:0) (value [| 4; 2 |]) in
  let failing =
    {
      (engine (ref [])) with
      run = (fun ~by:_ ?into:_ ~origin:_ _ _ _ -> raise Exit);
    }
  in
  raises Exit (fun () -> partitioned_on failing (fun () -> Nx.mul x x))

(* Offset *)

(* The coordinate along axis 0 of a loop of [[3]], made on the host. *)
let iota =
  Nx.Prim.Map
    {
      layout = L.contiguous [| 3 |];
      prog = Nx_kernel.Prog.of_exprs ~ins:[||] [| E.any (E.coord 0) |];
      outs = Nx.Prim.[ Nx.int64 ];
      loads = [||];
      at = Some Nx.Host.on;
    }

let iota_at o =
  let y, () = Nx.Prim.eval ~by:"test" (Nx.Prim.offset iota o) in
  Nx.to_array y

let offset_cases =
  cases "a loop reads each coordinate from its window's start"
    ~id:(fun (name, _, _) -> name)
    [
      ( "start 2 of 5",
        { Nx.Prim.start = [| 2 |]; extents = [| 5 |] },
        [| 2L; 3L; 4L |] );
      ("no start", { Nx.Prim.start = [||]; extents = [||] }, [| 0L; 1L; 2L |]);
    ]
    (fun (_, o, expected) -> equal (array int64) expected (iota_at o))

let test_offset_other () =
  let x = Nx.create f32 [| 3 |] [| 1.; -2.; 3. |] in
  let o = { Nx.Prim.start = [| 1 |]; extents = [| 4 |] } in
  equal (array float_exact)
    (Nx.to_array (Nx.Prim.eval ~by:"test" (Copy x)))
    (Nx.to_array (Nx.Prim.eval ~by:"test" (Nx.Prim.offset (Copy x) o)))

(* Pieces *)

type pieces_case =
  | Pieces :
      string * 'd Nx.Devices.t * 'd Nx.Placement.t * int array
      -> pieces_case

let shard_shapes p s =
  Nx.Repr.shards ~by:"test" (Nx.place p (Nx.zeros f32 s))
  |> Iarray.to_list
  |> List.map (fun a -> Iarray.to_array (L.shape (A.layout a)))

let pieces_cases =
  cases "each device has its placement and its shard's window, in order"
    ~id:(fun (Pieces (name, _, _, _)) -> name)
    [
      Pieces ("two, rows split unevenly", Two.v, Two.split ~axis:0, [| 3; 2 |]);
      Pieces ("three, columns split", Three.v, Three.split ~axis:1, [| 2; 5 |]);
      Pieces ("four, whole on each", Four.v, Four.on, [| 2; 2 |]);
      Pieces ("two, no row", Two.v, Two.split ~axis:0, [| 0; 2 |]);
    ]
    (fun (Pieces (_, s, p, shape)) ->
      let pieces = Array.to_list (Nx.Prim.pieces p shape) in
      equal ~msg:"devices" (list string)
        (List.map device_name (Nx.Devices.to_list s))
        (List.map (fun (here, _) -> device_name (device_of here)) pieces);
      equal ~msg:"windows"
        (list (array int))
        (shard_shapes p shape)
        (List.map
           (fun (_, w) ->
             Array.map (fun (r : Nx_array.Move.range) -> r.count) w)
           pieces))

let () =
  exit
    (run "Nx.Prim.partition"
       [
         group "partition"
           [
             (* A check of one flag needs one row and one column, 2.6% of the
                cases (miss 4e-4 per run): the example holds it. *)
             prop ~count:300
               ~examples:[ { n = 2; rows = 1; cols = 1; x = Split 0; y = On } ]
               "on an engine of nx's operations, gives eager's bits, \
                placements, refusals and checks"
               drawn law;
             test
               "a device whose windows are all empty computes nothing, but the \
                first"
               test_empty_windows;
             test "raises what its engine's steps raise" test_step_raises;
           ];
         group "offset"
           [
             offset_cases;
             test "an operation other than a loop evaluates as itself"
               test_offset_other;
           ];
         group "pieces"
           [
             pieces_cases;
             test "a placement that cuts an axis the shape lacks raises"
               (fun () ->
                 raises_match (Exn.invalid_arg ?sub:None) (fun () ->
                     Nx.Prim.pieces (Two.split ~axis:1) [| 3 |]));
           ];
       ])
