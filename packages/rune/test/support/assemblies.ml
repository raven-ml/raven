(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Operand
open Witness
module M = Nx_array.Move

type piece = {
  region : M.range array;
  view : view;
  offset : bool;
  computed : bool;  (** The exclusive or of two arguments' bits. *)
}

type t = Case : ('v, 's) Nx.dtype * int array * piece list * int -> t

let dtype (Case (dt, _, _, _)) = Nx.Dtype.Any dt
let range start count step = { M.start; count; step }

let pp_range ppf (r : M.range) =
  Format.fprintf ppf "%d+%d*%d" r.start r.count r.step

let pp ppf (Case (dt, s, pieces, seed)) =
  let piece ppf p =
    Format.fprintf ppf "(%a %a%s%s)"
      (Format.pp_print_array
         ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ";")
         pp_range)
      p.region pp_view p.view
      (if p.offset then " offset" else "")
      (if p.computed then " computed" else "")
  in
  Format.fprintf ppf "%a %a, %d pieces %a, seed %d" Nx.pp_dtype dt pp_shape s
    (List.length pieces)
    (Format.pp_print_list ~pp_sep:Format.pp_print_space piece)
    pieces seed

let counts rs = Array.map (fun (r : M.range) -> r.count) rs

(* The index of each element a region holds, in a value of shape [s]. *)
let held s rs =
  let out = ref [] in
  let rec walk d at =
    if d = Array.length rs then out := at :: !out
    else
      let r : M.range = rs.(d) in
      for i = 0 to r.count - 1 do
        walk (d + 1) ((at * s.(d)) + r.start + (i * r.step))
      done
  in
  walk 0 0;
  !out

(* How many of the regions [rs] hold each element of a value of shape [s]. *)
let hits s rs =
  let h = Array.make (numel s) 0 in
  List.iter (fun r -> List.iter (fun i -> h.(i) <- h.(i) + 1) (held s r)) rs;
  h

(* Whether the region [rs] holds every element of a value of shape [s]. *)
let covers s rs = numel s > 0 && List.length (held s rs) = numel s

(* Generators *)

(* A range along an axis of [d]: forwards or backwards by a step of 1 to 3,
   empty 1 in 7. *)
let axis_range d =
  let open Gen in
  let* up = bool in
  let* m = int_range 1 3 in
  let step = if up then m else -m in
  if d = 0 then constant (range 0 0 step)
  else
    let* start = int_range 0 (d - 1) in
    let most =
      if step > 0 then ((d - 1 - start) / step) + 1 else (start / -step) + 1
    in
    let+ count = frequency [ (1, constant 0); (6, int_range 1 most) ] in
    range start count step

(* One draw of each of [gs], in order. *)
let rec every = function
  | [] -> Gen.constant []
  | g :: gs ->
      let open Gen in
      let* x = g in
      let+ xs = every gs in
      x :: xs

let views = [ Plain; Flipped; Strided; Broadcast; Permuted ]

let piece_at region =
  let open Gen in
  let+ view = of_list views
  and+ offset = bool
  and+ computed = frequency [ (2, constant false); (1, constant true) ] in
  { region; view; offset; computed }

(* The region of all of [s], forwards or backwards along each axis. *)
let whole s =
  let open Gen in
  let+ back = every (List.map (fun _ -> bool) (Array.to_list s)) in
  Array.of_list
    (List.map2
       (fun n b -> if b && n > 1 then range (n - 1) n (-1) else range 0 n 1)
       (Array.to_list s) back)

(* Up to three pieces at random regions of a drawn shape, after a piece that
   covers it 1 in 3; or many pieces of one element into 64, or as many rows of
   a stack, from 2 to 64. *)
let gen =
  let open Gen in
  let few =
    let* s = shape in
    let* k = int_range 0 3 in
    let* pieces =
      list ~size:(constant k)
        (let* rs = every (List.map axis_range (Array.to_list s)) in
         piece_at (Array.of_list rs))
    in
    let* covered = frequency [ (2, constant false); (1, constant true) ] in
    if not covered then constant (s, pieces)
    else
      let* r = whole s in
      let+ p = piece_at r in
      (s, p :: pieces)
  in
  let many =
    let* p =
      frequency
        [
          (1, int_range 2 8);
          (1, int_range 20 24);
          (1, int_range 30 36);
          (1, constant 64);
        ]
    in
    let* rows = bool in
    if rows then
      let* w = int_range 1 4 in
      let+ ps =
        every
          (List.init p (fun i -> piece_at [| range i 1 1; range 0 w 1 |]))
      in
      ([| p; w |], ps)
    else
      let+ ps =
        list ~size:(constant p)
          (let* at = int_range 0 63 in
           piece_at [| range at 1 1 |])
      in
      ([| 64 |], ps)
  in
  let* s, pieces = frequency [ (1, few); (1, many) ] in
  let+ (Nx.Dtype.Any dt) = of_list Nx.Dtype.all
  and+ seed = int_range 0 1_000_000 in
  Case (dt, s, pieces, seed)

let examples =
  let piece ?(computed = false) i region =
    { region; view = List.nth views (i mod 5); offset = i mod 2 = 0; computed }
  in
  let rows p w =
    List.init p (fun i -> piece i [| range i 1 1; range 0 w 1 |])
  in
  let spots p = List.init p (fun i -> piece 0 [| range (3 * i) 1 1 |]) in
  List.mapi
    (fun i (Nx.Dtype.Any dt) ->
      let s, pieces =
        match i mod 7 with
        | 0 -> ([| 33; 3 |], rows 33 3)
        | 1 -> ([| 64 |], spots 22)
        | 2 -> ([| 5; 5 |], [ piece 2 [| range 4 2 (-3); range 0 3 2 |] ])
        | 3 -> ([| 2; 3 |], [])
        | 4 ->
            ( [| 4; 3 |],
              [
                piece ~computed:true 1 [| range 3 4 (-1); range 0 3 1 |];
                piece 2 [| range 1 2 1; range 2 2 (-2) |];
              ] )
        | 5 ->
            ( [| 6 |],
              [
                piece ~computed:true 0 [| range 0 4 1 |];
                piece 1 [| range 2 4 1 |];
                piece ~computed:true 2 [| range 5 3 (-2) |];
              ] )
        | _ ->
            ( [| 3; 4 |],
              [ piece ~computed:true 3 [| range 1 2 1; range 3 2 (-2) |] ] )
      in
      Case (dt, s, pieces, i))
    Nx.Dtype.all

(* The law *)

(* The exclusive or of [a]'s and [b]'s bits, as elements of [dt]. *)
let mixed (type v s d) (dt : (v, s) Nx.dtype) (a : (v, s, d) Nx.t)
    (b : (v, s, d) Nx.t) : (v, s, d) Nx.t =
  let unsigned : Nx.Dtype.any =
    match Nx.Dtype.bits dt with
    | 1 -> Any Nx.bit
    | 4 -> Any Nx.uint4
    | 8 -> Any Nx.uint8
    | 16 -> Any Nx.uint16
    | 32 -> Any Nx.uint32
    | _ -> Any Nx.uint64
  in
  match unsigned with
  | Any u -> Nx.bitcast dt (Nx.bitwise_xor (Nx.bitcast u a) (Nx.bitcast u b))

let law (module S : Nx.Devices) (Case (dt, s, pieces, seed)) =
  let st = Random.State.make [| seed |] in
  let regions = List.map (fun p -> p.region) pieces in
  let held = List.filter (fun r -> numel (counts r) > 0) regions in
  let h = hits s regions in
  let stepped (r : M.range) = abs r.step > 1 && r.count > 1 in
  let over =
    match held with r :: rest when covers s r -> rest | rs -> rs
  in
  (* The pieces over the background: those that hold an element, after a
     first that covers the value. *)
  let viewing = List.filter (fun p -> List.memq p.region over) pieces in
  let one ~computed =
    match viewing with [ p ] -> p.computed = computed | _ -> false
  in
  cover "no piece" (held = []);
  cover "one view over the fill"
    (List.length held = 1 && one ~computed:false);
  cover "one view over a covering piece"
    (List.length held = 2 && one ~computed:false);
  cover "one computed piece" (one ~computed:true);
  cover "several pieces" (List.length over > 1);
  cover "pieces that tile"
    (List.length held > 1 && Array.for_all (fun n -> n = 1) h);
  cover "pieces that overlap" (Array.exists (fun n -> n > 1) h);
  cover "a region by steps"
    (List.exists (fun r -> Array.exists stepped r) held);
  let backwards (r : M.range) = r.step < 0 && r.count > 1 in
  cover "a region backwards"
    (List.exists (fun r -> Array.exists backwards r) held);
  cover "steps that reach past the axis"
    (List.exists
       (fun r ->
         Array.exists2
           (fun (r : M.range) n -> stepped r && r.count * abs r.step > n)
           r s)
       held);
  cover "33 pieces or more" (List.length held > 32);
  cover "a sub-byte dtype" (Nx.Dtype.bits dt < 8);
  cover "a complex dtype" (Nx.Dtype.is Complex dt);
  let fill = (Nx.to_array (random_bits dt [| 1 |] st)).(0) in
  (* Each piece's arguments, at the host and on the device, of one storage. *)
  let args =
    List.map
      (fun p ->
        let one () =
          let kept = ref None in
          let host =
            viewed p.view ~offset:p.offset (counts p.region) (fun s ->
                let x = random_bits dt s st in
                kept := Some x;
                x)
          in
          let placed =
            viewed p.view ~offset:p.offset (counts p.region) (fun _ ->
                Nx.place S.on (Option.get !kept))
          in
          (host, placed)
        in
        if p.computed then [ one (); one () ] else [ one () ])
      pieces
  in
  let f (type d) (xs : (_, _, d) Nx.t list) =
    let rec take ps xs =
      match (ps, xs) with
      | [], _ -> []
      | p :: ps, a :: b :: xs when p.computed ->
          (Iarray.of_array p.region, mixed dt a b) :: take ps xs
      | p :: ps, a :: xs -> (Iarray.of_array p.region, a) :: take ps xs
      | _ :: _, [] -> invalid_arg "Assemblies.law"
    in
    let shape = Iarray.of_array s in
    Nx.Prim.eval ~by:"assembly"
      (Assemble { dtype = dt; shape; fill; pieces = take pieces xs })
  in
  let args = List.concat args in
  let eager = raw (f (List.map fst args)) in
  let before = Rune.evaluations () in
  let y =
    Rune.jit Nx.Ptree.(list tensor @-> returns tensor) f (List.map snd args)
  in
  let y = Nx.place Nx.Host.on y in
  equal ~msg:"compiled" Testable.int 0 (Rune.evaluations () - before);
  equal codes_t eager (raw y)
