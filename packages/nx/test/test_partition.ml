(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.Prim.partition: every operation of a function over values cut across two,
   three or four devices, or whole on each, computed as its partition on an
   engine whose steps are nx's own operations on one device and placements,
   gives eager's bits and placements, for even, uneven and empty windows. *)

open Windtrap

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
    (Nx.rigs s)

let window x (w : Nx_array.Move.range array) =
  Nx.Prim.eval ~by (Move (Slice (Iarray.of_array w), x))

let put at x = Nx.Prim.eval ~by (Place (at, x))

let empty (type v s e) (at : e Nx.Placement.t) (dt : (v, s) Nx.dtype) shape :
    (v, s) Nx_array.t =
  Nx_array.create (device_of at) dt (Iarray.of_array shape)

let engine : Nx.Prim.engine =
  {
    run =
      (fun ~by ?into:_ ~origin _ _ op ->
        Nx.Prim.eval ~by (Nx.Prim.offset op origin));
    piece =
      (fun x j _ ->
        let s = Nx.Placement.devices (Nx.placement x) in
        Nx.Repr.of_array s (Iarray.get (Nx.Repr.shards x) j));
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
          | Some v -> Option.get (Nx.Repr.array v)
          | None ->
              let here, w = windows.(j) in
              empty here dt
                (Array.map (fun (r : Nx_array.Move.range) -> r.count) w)
        in
        Nx.Repr.of_shards p (Iarray.init (Array.length windows) array));
  }

(* [f ()] with each operation over several devices computed as its partition on
   [engine], the others eagerly. *)
let partitioned f =
  let rule _ ~by (type r) (op : r Nx.Prim.t) : r =
    match op with
    | Check _ -> Nx.Prim.eval ~by op
    | _ -> Nx.Prim.partition engine ~by op
  in
  Nx.Prim.interpret ~name:"partitioned" Extent rule (fun _ -> f ())

(* Cases *)

type cut = On | Split of int | Device of int

let pp_cut ppf = function
  | On -> Format.pp_print_string ppf "on"
  | Split a -> Format.fprintf ppf "split ~axis:%d" a
  | Device k -> Format.fprintf ppf "device %d" k

let at (type d) (s : d Nx.devices) c : d Nx.Placement.t =
  match c with
  | On -> Nx.Placement.on s
  | Split axis -> Nx.Placement.split ~axis s
  | Device k -> Nx.Placement.device s (m k)

type case = { n : int; rows : int; cols : int; x : cut; y : cut }

let pp_case ppf c =
  Format.fprintf ppf "%d devices, %d x %d, x %a, y %a" c.n c.rows c.cols pp_cut
    c.x pp_cut c.y

let cases =
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

(* Each function of the case's operands over [s], eagerly and partitioned. *)
let law_on (type d) (s : d Nx.devices) c =
  let x = Nx.place (at s c.x) (value [| c.rows; c.cols |]) in
  let y = Nx.place (at s c.y) (value ~seed:1 [| c.rows; c.cols |]) in
  let w = Nx.place (at s c.y) (value ~seed:2 [| c.cols; 3 |]) in
  let idx = Nx.create Nx.int64 [| 2 |] [| 0L; Int64.of_int (c.cols - 1) |] in
  let idx = Nx.place (Nx.Placement.uncut (Nx.placement x)) idx in
  let fs : (string * (unit -> (float, Nx.float32_elt, d) Nx.t)) list =
    [
      ("mul", fun () -> Nx.mul x y);
      ("sum rows", fun () -> Nx.sum ~axes:[ 0 ] x);
      ("sum columns", fun () -> Nx.sum ~axes:[ 1 ] x);
      ("max", fun () -> Nx.max x);
      ("cumsum", fun () -> Nx.cumsum ~axis:0 x);
      ("matmul", fun () -> Nx.matmul x w);
      ("take", fun () -> Nx.take ~axis:1 idx x);
      (* Coordinates: each device reads its window's from its start. *)
      ("argmax rows", fun () -> Nx.cast f32 (Nx.argmax ~axis:0 x));
      ("argmax columns", fun () -> Nx.cast f32 (Nx.argmax ~axis:1 x));
      ("tril", fun () -> Nx.tril x);
      ( "eye",
        fun () -> Nx.eye_at ~m:c.cols (Nx.placement x) f32 c.rows );
      ("place", fun () -> Nx.place (Nx.Placement.split ~axis:1 s) x);
    ]
  in
  List.iter
    (fun (name, f) ->
      equal ~msg:name result_t (result f) (result (fun () -> partitioned f)))
    fs

let law c =
  let cut_rows = function Split 0 -> true | _ -> false in
  cover "an empty window" (cut_rows c.x && c.rows < c.n);
  cover "uneven windows" (cut_rows c.x && c.rows > c.n && c.rows mod c.n <> 0);
  cover "three devices" (c.n = 3);
  match c.n with
  | 2 -> law_on Two.v c
  | 3 -> law_on Three.v c
  | _ -> law_on Four.v c

let () =
  exit
    (run "Nx.Prim.partition"
       [
         group "an engine of nx's operations computes as eager execution"
           [ prop ~count:60 "over two, three and four devices" cases law ];
       ])
