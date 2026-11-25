(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module B = Rig.Buffer
module S = Nx_array_support

let strf = Printf.sprintf
let f32 = D.Float32

(* No claim holds a live array of [arrays]: a take of each succeeds, which kills
   it. *)
let all_released arrays =
  Array.iter
    (fun (A.Any a) -> if B.dead (A.buffer a) = None then released (A.buffer a))
    arrays

(* The door *)

let zeros s = floats32 s (Array.make (Iarray.fold_left ( * ) 1 s) 0.)

let test_door_add () =
  let x = floats32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let y =
    Option.get
      (A.move
         (M.Permute [| 1; 0 |])
         (floats32 [| 3; 2 |] [| 10.; 20.; 30.; 40.; 50.; 60. |]))
  in
  let z = zeros [| 2; 3 |] in
  equal answer A.Done (S.add z x y);
  equal (values f32) [| 10.; 31.; 52.; 23.; 44.; 65. |] (A.to_array z);
  equal answer A.Done (S.add (zeros [| 0; 3 |]) (zeros [| 0; 3 |]) (zeros [| 0; 3 |]))

(* Each operand position given another dtype, then another shape: the call is
   refused and touches no byte. *)
let test_door_positions () =
  let s : int iarray = [| 2; 3 |] in
  let operands () =
    [| zeros s; floats32 s (Array.make 6 1.); floats32 s (Array.make 6 2.) |]
  in
  let call (ops : 'a array) = S.add ops.(0) ops.(1) ops.(2) in
  for k = 0 to 2 do
    let ops = operands () in
    let wrong = A.of_array D.Float64 s (Array.make 6 5.) in
    let args = Array.map (fun a -> A.Any a) ops in
    args.(k) <- A.Any wrong;
    let e =
      match args with
      | [| A.Any z; A.Any x; A.Any y |] -> S.add z x y
      | _ -> assert false
    in
    equal ~msg:(Printf.sprintf "dtype at %d" k) answer A.Wrong_dtype e;
    equal (values f32) (Array.make 6 0.) (A.to_array ops.(0));
    let ops = operands () in
    ops.(k) <- zeros [| 3; 2 |];
    equal ~msg:(Printf.sprintf "shape at %d" k) answer A.Shape_mismatch (call ops);
    if k > 0 then equal (values f32) (Array.make 6 0.) (A.to_array ops.(0))
  done

let test_door_written () =
  let x = floats32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = A.buffer (zeros [| 8 |]) in
  let z = A.v f32 (L.contiguous [| 3 |]) b in
  let y = A.v f32 (L.v ~offset:2 ~strides:[| 1 |] [| 3 |]) b in
  equal ~msg:"z overlaps y" answer A.Overlapping (S.add z x y);
  let y = A.v f32 (L.v ~offset:3 ~strides:[| 1 |] [| 3 |]) b in
  equal ~msg:"z beside y" answer A.Done (S.add z x y);
  let r = Option.get (A.move (M.Broadcast [| 3 |]) (zeros [| 1 |])) in
  equal ~msg:"z broadcast" answer A.Repeated_elements (S.add r x x);
  all_released [| A.Any r; A.Any x |]

(* A written operand may be identical to read ones: one width, one memory,
   every index at one byte. The pair leaves no claim. *)
let test_door_identical () =
  let x = floats32 [| 3 |] [| 1.; 2.; 3. |] in
  equal ~msg:"x + 0 into x" answer A.Done (S.add x x (zeros [| 3 |]));
  equal (values f32) [| 1.; 2.; 3. |] (A.to_array x);
  equal ~msg:"x + x into x" answer A.Done (S.add x x x);
  equal (values f32) [| 2.; 4.; 6. |] (A.to_array x);
  released (A.buffer x);
  let b = A.buffer (zeros [| 4 |]) in
  let at offset = A.v f32 (L.v ~offset ~strides:[| 1 |] [| 3 |]) b in
  equal ~msg:"shifted one element" answer A.Overlapping
    (S.add (at 0) (at 1) (zeros [| 3 |]));
  let sq = floats32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let t = Option.get (A.move (M.Permute [| 1; 0 |]) sq) in
  equal ~msg:"transposed" answer A.Overlapping (S.add sq t (zeros [| 2; 2 |]));
  released ~msg:"released after a refusal" (A.buffer sq)

let test_door_buffers () =
  let x = floats32 [| 2 |] [| 1.; 2. |] in
  let d = zeros [| 2 |] in
  kill (A.buffer d);
  equal ~msg:"dead" answer A.Dead_buffer (S.add (zeros [| 2 |]) x d);
  (* A dead buffer is refused as dead before any overlap is compared. *)
  let b = A.buffer (zeros [| 4 |]) in
  let at offset = A.v f32 (L.v ~offset ~strides:[| 1 |] [| 3 |]) b in
  let z = at 0 and y = at 1 in
  kill b;
  equal ~msg:"dead and overlapping" answer A.Dead_buffer (S.add z x y);
  let off = A.to_device (S.off_host ()) x in
  let z = zeros [| 2 |] in
  equal ~msg:"off the host" answer A.Off_host (S.add z x off);
  all_released [| A.Any z; A.Any x; A.Any off |]

(* The door reads up to max_operands operands, more than a loop takes among
   them, and refuses more as Shape_mismatch; a refusal after some claims leaves
   none. *)
let test_door_arity () =
  let ops n = Array.init n (fun _ -> A.Any (zeros [| 2 |])) in
  for n = 0 to A.max_operands do
    let ops = ops n in
    equal ~msg:(Printf.sprintf "%d" n) answer A.Done (S.read_all ops);
    all_released ops
  done;
  let ops9 = ops 9 in
  let b = A.buffer (zeros [| 4 |]) in
  ops9.(0) <- A.Any (A.v f32 (L.v ~offset:0 ~strides:[| 1 |] [| 2 |]) b);
  ops9.(8) <- A.Any (A.v f32 (L.v ~offset:1 ~strides:[| 1 |] [| 2 |]) b);
  equal ~msg:"9, the last overlapping the first" answer A.Overlapping
    (S.read_all ops9);
  all_released ops9;
  let ops9 = ops 9 in
  let (A.Any d) = ops9.(6) in
  kill (A.buffer d);
  equal ~msg:"9, the seventh dead" answer A.Dead_buffer (S.read_all ops9);
  all_released ops9;
  let past = ops (A.max_operands + 1) in
  equal ~msg:"past max_operands" answer A.Shape_mismatch (S.read_all past);
  all_released past

(* Each refusal the door answers reaches the user as its own reason. *)
let test_door_reasons () =
  let x = floats32 [| 2 |] [| 1.; 2. |] in
  let reason ~sub e =
    match A.refused "k" e [ A.Any x ] with
    | () -> fail "refused returned"
    | exception Invalid_argument m -> contains ~msg:sub ~sub m
  in
  let wrong = A.of_array D.Float64 [| 2 |] [| 1.; 2. |] in
  reason ~sub:"dtype" (S.add (zeros [| 2 |]) x wrong);
  let d = zeros [| 2 |] in
  kill (A.buffer d);
  reason ~sub:"dead" (S.add (zeros [| 2 |]) x d);
  let off = A.to_device (S.off_host ()) x in
  reason ~sub:"host does not address" (S.add (zeros [| 2 |]) x off);
  let r = Option.get (A.move (M.Broadcast [| 2 |]) (zeros [| 1 |])) in
  reason ~sub:"element twice" (S.add r x x);
  let b = A.buffer (zeros [| 3 |]) in
  let at offset = A.v f32 (L.v ~offset ~strides:[| 1 |] [| 2 |]) b in
  reason ~sub:"shares bytes" (S.add (at 0) (at 1) x);
  reason ~sub:"do not fit" (S.add (zeros [| 2 |]) x (zeros [| 3 |]))

(* An operand with no element passes the door wherever its memory lies. *)
let test_door_empty () =
  let host = zeros [| 0; 2 |] in
  let off = Option.get (A.move (M.Permute [| 1; 0 |]) (zeros [| 2; 0 |])) in
  let elsewhere = A.to_device (S.off_host ()) host in
  equal ~msg:"host" answer A.Done (S.add (zeros [| 0; 2 |]) host host);
  equal ~msg:"a view" answer A.Done (S.add (zeros [| 0; 2 |]) off host);
  equal ~msg:"off the host" answer A.Done
    (S.add (zeros [| 0; 2 |]) elsewhere elsewhere)

let test_door_releases () =
  let z = zeros [| 2 |] and x = floats32 [| 2 |] [| 1.; 2. |] in
  equal answer A.Done (S.add z x x);
  equal answer A.Shape_mismatch (S.add z x (zeros [| 3 |]));
  released (A.buffer z);
  released (A.buffer x)

let test_door_moving_gc () =
  let x = floats32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  equal answer A.Done (S.collect x);
  let x = retaken x in
  equal (values f32) [| 1.; 2.; 3.; 4. |] (A.to_array x)

(* The descriptor nx_read fills for an array laid out by [l], over the bytes [l]
   reaches, holds what the accessors give: the rank, the flags (NX_CONTIGUOUS 1,
   NX_DISTINCT 2, NX_EMPTY 4), the offset, the extents and the strides. *)
let law_descriptor l =
  let flag b f = if b then f else 0 in
  let r = L.rank l in
  let flags =
    flag (L.is_contiguous l) 1
    lor flag (L.is_distinct l) 2
    lor flag (L.numel l = 0) 4
  in
  let bytes = B.create Rig.host (max 1 (snd (L.span l))) in
  let a = A.v D.Uint8 l bytes in
  equal ints
    (Array.concat
       [
         [| r; flags; L.offset l |];
         Iarray.to_array (L.shape l);
         Iarray.to_array (L.strides l);
       ])
    (S.descriptor a)

let test_max_numel () =
  equal ~msg:"NX_MAX_NUMEL" int L.max_numel (S.max_numel ())

(* Answers *)

let refusals =
  List.filter (fun (_, a) -> a <> A.Done && a <> A.Declined) S.answers

(* C answers nx_array.h's codes, which OCaml reads as the constructors. *)
let test_codes () =
  equal (array int)
    (Array.of_list (List.map (fun (_, a) -> S.code a) S.answers))
    (S.codes ())

(* Refusals *)

let test_refused () =
  let a = floats32 [| 2; 3 |] (Array.make 6 0.) in
  let b = A.create Rig.host D.Int8 [| 4 |] in
  (match A.refused "add" A.Wrong_dtype [ A.Any a; A.Any b ] with
  | () -> fail "refused returned"
  | exception Invalid_argument m ->
      starts_with ~prefix:"add: " m;
      contains ~sub:"dtype" m;
      contains ~sub:"float32 [2; 3]" m;
      contains ~sub:"int8 [4]" m);
  raises_match (Exn.invalid_arg ~sub:"shapes") (fun () ->
      A.refused "add" A.Shape_mismatch [ A.Any a ]);
  raises_match (Exn.invalid_arg ~sub:"not a refusal") (fun () ->
      A.refused "add" A.Done [ A.Any a ]);
  raises_match (Exn.invalid_arg ~sub:"not a refusal") (fun () ->
      A.refused "add" A.Declined [ A.Any a ])

(* A dead operand's reason is its consumer's, given before the operands, each
   dead operand named by its dtype and shape in the list's order. *)
let test_refused_dead () =
  let a = floats32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = floats32 [| 3 |] [| 4.; 5.; 6. |] in
  kill ~why:"donated (Nx.donate)" (A.buffer b);
  raises
    (Invalid_argument
       "Nx.add: float32 [3] was donated (Nx.donate) (float32 [3], float32 [3])")
    (fun () -> A.refused "Nx.add" A.Dead_buffer [ A.Any a; A.Any b ]);
  kill ~why:"donated to Nx.set" (A.buffer a);
  raises
    (Invalid_argument
       "Nx.add: float32 [3] was donated to Nx.set, float32 [3] was donated \
        (Nx.donate) (float32 [3], float32 [3])") (fun () ->
      A.refused "Nx.add" A.Dead_buffer [ A.Any a; A.Any b ])

(* Every refusal raises, naming the function and each operand, with a reason of
   its own. *)
let test_refused_codes () =
  let a = floats32 [| 2; 3 |] (Array.make 6 0.) in
  let reason (name, r) =
    match A.refused "Nx.f" r [ A.Any a ] with
    | () -> failf "refused returned on %s" name
    | exception Invalid_argument m ->
        starts_with ~msg:name ~prefix:"Nx.f: " m;
        contains ~msg:name ~sub:"float32 [2; 3]" m;
        m
  in
  let reasons = List.map reason refusals in
  equal int (List.length refusals)
    (List.length (List.sort_uniq compare reasons))

(* The OCaml door *)

(* [door ~written ~read]'s answer, and whether its function ran: it asserts that
   every array is claimed while it runs. *)
let ocaml_door written read =
  let ran = ref "not run" in
  let all = Array.append written read in
  let e =
    A.door ~written:(Iarray.of_array written) ~read:(Iarray.of_array read)
      (fun all ->
        Array.iter (fun (A.Any a) -> claimed (A.buffer a)) all;
        ran := "ran")
      all
  in
  (e, !ran)

(* The door over [n] arrays, one written, about the runs of max_operands claims;
   with [twice], one read array given twice. *)
let test_ocaml_door_counts (n, twice) =
  let written = if n = 0 then [||] else [| A.Any (zeros [| 2 |]) |] in
  let read =
    if twice then
      let x = A.Any (floats32 [| 2 |] [| 1.; 2. |]) in
      Array.make (n - 1) x
    else Array.init (max 0 (n - 1)) (fun _ -> A.Any (zeros [| 2 |]))
  in
  equal (pair answer string) (A.Done, "ran") (ocaml_door written read);
  all_released (Array.append written read)

let door_counts =
  let m = A.max_operands in
  (0, false) :: (3, true)
  :: List.map
       (fun n -> (n, false))
       [ m - 1; m; m + 1; (2 * m) - 1; 2 * m; (2 * m) + 1 ]

(* The door over three arrays allocates nothing. *)
let test_door_allocates_nothing () =
  let written = Iarray.of_array [| A.Any (zeros [| 2 |]) |] in
  let read =
    Iarray.of_array
      [| A.Any (floats32 [| 2 |] [| 1.; 2. |]); A.Any (zeros [| 2 |]) |]
  in
  let f () = () in
  ignore (A.door ~written ~read f ());
  let before = Gc.minor_words () in
  for _ = 1 to 1000 do
    ignore (Sys.opaque_identity (A.door ~written ~read f ()))
  done;
  equal ~msg:"words for 1000 calls" int 0
    (int_of_float (Gc.minor_words () -. before))

(* However many arrays the door claims and releases, it holds the domain
   lock for no stretch longer than the bound, in CPU time, each stretch the
   least of three runs': a call of thousands of stretches has some
   lengthened by a preemption in every run, each time another. *)
let test_ocaml_door_holds () =
  let holds ~msg written read =
    let e = ref A.Done in
    let w = Iarray.of_array written and r = Iarray.of_array read in
    let call () = e := A.door ~written:w ~read:r ignore () in
    call ();
    Gc.full_major ();
    let held = Nx_release_support.held ~runs:3 call in
    equal ~msg answer A.Done !e;
    less int
      ~msg:(msg ^ ": the longest stretch held, in ticks of CPU time")
      ~than:(Nx_release_support.bound ()) held;
    all_released (Array.append written read)
  in
  holds ~msg:"many arrays"
    [| A.Any (zeros [| 1 |]) |]
    (Array.init 32768 (fun _ -> A.Any (zeros [| 1 |])));
  (* Few written arrays, each compared with many it skips. *)
  holds ~msg:"many empty arrays"
    (Array.init 8 (fun _ -> A.Any (zeros [| 1 |])))
    (Array.init 32768 (fun _ -> A.Any (zeros [| 0 |])))

(* The door claims every array, then checks the rest: an array claimed in
   a later run than a written array's other faults is refused first. *)
let test_ocaml_door_order () =
  let one = zeros [| 1 |] in
  let w = Option.get (A.move (M.Broadcast [| 2 |]) one) in
  let dead = zeros [| 2 |] in
  kill (A.buffer dead);
  let read =
    Array.init (2 * A.max_operands) (fun i ->
        if i = A.max_operands then A.Any dead else A.Any (zeros [| 2 |]))
  in
  equal (pair answer string) (A.Dead_buffer, "not run")
    (ocaml_door [| A.Any w |] read);
  all_released (Array.append [| A.Any one |] read)

(* A refusal of an array claimed in a later run than the first leaves no
   claim on any. *)
let test_ocaml_door_late_refusals () =
  let n = 3 * A.max_operands in
  let refuses ~msg r last =
    let written = [| A.Any (zeros [| 2 |]) |] in
    let read = Array.init n (fun _ -> A.Any (zeros [| 2 |])) in
    read.(n - 1) <- last written.(0);
    equal ~msg (pair answer string) (r, "not run") (ocaml_door written read);
    all_released (Array.append written read)
  in
  refuses ~msg:"the last read overlapping the written" A.Overlapping
    (fun (A.Any z) ->
      A.Any (A.v f32 (L.v ~offset:1 ~strides:[| 1 |] [| 1 |]) (A.buffer z)));
  refuses ~msg:"the last read dead" A.Dead_buffer (fun _ ->
      let d = zeros [| 2 |] in
      kill (A.buffer d);
      A.Any d)

(* Two faults of one call, each in arrays of its own: the door answers the
   same refusal with few arrays as with more than max_operands, the second
   fault's arrays then claimed and checked in a later run. Dead_buffer and
   Read_only come before the others. *)
let test_ocaml_door_many_refusals () =
  let dead = zeros [| 2 |] in
  kill (A.buffer dead);
  let ro = A.v f32 (L.contiguous [| 3 |]) (S.read_only 12) in
  let repeated = Option.get (A.move (M.Broadcast [| 2 |]) (zeros [| 1 |])) in
  let b = A.buffer (zeros [| 4 |]) in
  let at offset = A.Any (A.v f32 (L.v ~offset ~strides:[| 1 |] [| 3 |]) b) in
  let faults =
    [
      ("dead", A.Dead_buffer, ([||], [| A.Any dead |]));
      ("read-only", A.Read_only, ([| A.Any ro |], [||]));
      ("repeated", A.Repeated_elements, ([| A.Any repeated |], [||]));
      ("overlapping", A.Overlapping, ([| at 0 |], [| at 1 |]));
    ]
  in
  let claims = [ A.Dead_buffer; A.Read_only ] in
  let pad () = Array.init (A.max_operands + 1) (fun _ -> A.Any (zeros [| 2 |])) in
  let pads = (pad (), pad ()) in
  let calls =
    List.concat_map
      (fun (n1, r1, (w1, x1)) ->
        List.filter_map
          (fun (n2, r2, (w2, x2)) ->
            if n1 = n2 then None
            else
              let few = ocaml_door (Array.append w1 w2) (Array.append x1 x2) in
              let pw, px = pads in
              let many =
                ocaml_door
                  (Array.concat [ w1; pw; w2 ])
                  (Array.concat [ x1; px; x2 ])
              in
              Some (strf "%s then %s" n1 n2, (r1, r2), few, many))
          faults)
      faults
  in
  List.iter
    (fun (msg, (r1, r2), few, many) ->
      let door = pair answer string in
      equal ~msg door few many;
      let r, _ = few in
      if List.mem r1 claims || List.mem r2 claims then
        satisfies ~msg ~claim:"a claim's refusal, which comes first" answer
          (fun r -> List.mem r claims)
          r)
    calls;
  let pw, px = pads in
  all_released (Array.concat [ pw; px; [| A.Any repeated; at 0 |] ])

(* [refuses] takes the arrays it checks, so each case makes its own. *)
let test_ocaml_door_refuses () =
  let x () = floats32 [| 3 |] [| 1.; 2.; 3. |] in
  let refuses ~msg r written read =
    equal ~msg (pair answer string) (r, "not run") (ocaml_door written read);
    all_released (Array.append written read)
  in
  let twice = A.Any (x ()) in
  refuses ~msg:"written twice" A.Overlapping [| twice; twice |] [||];
  let eight () =
    let b = A.buffer (zeros [| 8 |]) in
    fun offset -> A.Any (A.v f32 (L.v ~offset ~strides:[| 1 |] [| 3 |]) b)
  in
  let at = eight () in
  refuses ~msg:"z overlaps y" A.Overlapping [| at 0 |] [| at 2 |];
  let at = eight () in
  equal ~msg:"z beside y" (pair answer string) (A.Done, "ran")
    (ocaml_door [| at 0 |] [| at 3 |]);
  let r = Option.get (A.move (M.Broadcast [| 3 |]) (zeros [| 1 |])) in
  refuses ~msg:"z broadcast" A.Repeated_elements
    [| A.Any r |]
    [| A.Any (x ()) |];
  let d = zeros [| 3 |] in
  kill (A.buffer d);
  refuses ~msg:"dead" A.Dead_buffer
    [| A.Any (zeros [| 3 |]) |]
    [| A.Any (x ()); A.Any d |];
  let ro = A.v f32 (L.contiguous [| 3 |]) (S.read_only 12) in
  equal ~msg:"read-only" (pair answer string) (A.Read_only, "not run")
    (ocaml_door [| A.Any ro |] [| A.Any (x ()) |]);
  equal ~msg:"read-only, read" (pair answer string) (A.Done, "ran")
    (ocaml_door [| A.Any (zeros [| 3 |]) |] [| A.Any ro |]);
  (* A dead array is refused as dead before any overlap is compared. *)
  let b = A.buffer (zeros [| 4 |]) in
  let at offset = A.Any (A.v f32 (L.v ~offset ~strides:[| 1 |] [| 3 |]) b) in
  let z = at 0 and y = at 1 in
  kill b;
  refuses ~msg:"dead and overlapping" A.Dead_buffer [| z |] [| y |]

(* The OCaml door's identity rule, as nx_read's. *)
let test_ocaml_door_identical () =
  let xa = floats32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let x = A.Any xa in
  let door = pair answer string in
  equal ~msg:"x into x" door (A.Done, "ran") (ocaml_door [| x |] [| x |]);
  equal ~msg:"x into x, read twice" door (A.Done, "ran")
    (ocaml_door [| x |] [| x; x |]);
  let xa = retaken xa in
  let x = A.Any xa in
  let bytes = A.Any (Option.get (A.bitcast D.Uint8 xa)) in
  equal ~msg:"another width" door (A.Overlapping, "not run")
    (ocaml_door [| x |] [| bytes |]);
  let b = A.buffer (zeros [| 4 |]) in
  let at offset = A.Any (A.v f32 (L.v ~offset ~strides:[| 1 |] [| 3 |]) b) in
  equal ~msg:"shifted one element" door (A.Overlapping, "not run")
    (ocaml_door [| at 0 |] [| at 1 |]);
  let sq = floats32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let t = A.Any (Option.get (A.move (M.Permute [| 1; 0 |]) sq)) in
  equal ~msg:"transposed" door (A.Overlapping, "not run")
    (ocaml_door [| A.Any sq |] [| t |]);
  equal ~msg:"identical, written twice" door (A.Overlapping, "not run")
    (ocaml_door [| x; x |] [||]);
  all_released [| x; bytes; A.Any sq |]

let test_ocaml_door_off_host () =
  let b = A.buffer (A.to_device (S.off_host ()) (zeros [| 8 |])) in
  let at offset = A.Any (A.v f32 (L.v ~offset ~strides:[| 1 |] [| 3 |]) b) in
  equal ~msg:"overlap" (pair answer string) (A.Overlapping, "not run")
    (ocaml_door [| at 0 |] [| at 2 |]);
  equal ~msg:"beside" (pair answer string) (A.Done, "ran")
    (ocaml_door [| at 0 |] [| at 3 |])

(* Two bigarrays over one memory's bytes overlap by their addresses. *)
let test_ocaml_door_bigarrays () =
  let g = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 8 in
  let over first =
    A.of_bigarray f32
      (Bigarray.genarray_of_array1 (Bigarray.Array1.sub g first 4))
  in
  let part start a =
    A.Any
      (Option.get (A.move (M.Slice [| { M.start; count = 2; step = 1 } |]) a))
  in
  let head = part 0 and tail = part 2 in
  equal ~msg:"overlap" (pair answer string) (A.Overlapping, "not run")
    (ocaml_door [| tail (over 0) |] [| head (over 1) |]);
  equal ~msg:"beside" (pair answer string) (A.Done, "ran")
    (ocaml_door [| head (over 0) |] [| head (over 2) |])

(* An io memory of 32 bytes, given by a library whose regions the host maps, a
   host borrow of it, whose bytes are its pages, and those pages as a
   bigarray. *)
let mapped_io () =
  let io, pages = S.mapped_region 32 in
  (io, Option.get (B.borrow Rig.host io), pages)

(* An array of [dt] over [count] elements of [b] from element [offset]. *)
let view dt b offset count =
  A.Any (A.v dt (L.v ~offset ~strides:[| 1 |] [| count |]) b)

(* An io memory lies in its own space and, once a borrow maps its pages, at
   their host address too: its views and its borrow's compare by offset, a
   bigarray over its pages by address, int4 views by their bits. *)
let test_ocaml_door_io_pages () =
  let io, host, pages = mapped_io () in
  let io_at = view f32 io and borrow_at = view f32 host in
  let over first count =
    let b = Bigarray.Array1.sub pages (4 * first) (4 * count) in
    view f32 (B.of_bigarray b) 0 count
  in
  let door = pair answer string in
  let check ~msg r written read =
    let ran = if r = A.Done then "ran" else "not run" in
    equal ~msg door (r, ran) (ocaml_door [| written |] [| read |])
  in
  check ~msg:"io beside its pages' bigarray" A.Done (io_at 0 2) (over 4 4);
  check ~msg:"io meeting its pages' bigarray" A.Overlapping (io_at 0 3) (over 2 2);
  check ~msg:"io identical to its pages' bigarray" A.Done (io_at 2 2) (over 2 2);
  check ~msg:"the bigarray meeting io" A.Overlapping (over 0 3) (io_at 2 2);
  check ~msg:"a borrow beside io" A.Done (borrow_at 0 2) (io_at 2 2);
  check ~msg:"a borrow meeting io" A.Overlapping (borrow_at 0 3) (io_at 2 2);
  check ~msg:"a borrow identical to io" A.Done (borrow_at 1 2) (io_at 1 2);
  check ~msg:"a borrow meeting the bigarray" A.Overlapping (borrow_at 0 3)
    (over 2 2);
  (* int4 elements of byte 0 of the pages, through a bigarray of that byte
     alone: identity is by bits, and one byte holds two elements. *)
  let int4 = view D.Int4 io in
  let nibble = view D.Int4 (B.of_bigarray (Bigarray.Array1.sub pages 0 1)) in
  check ~msg:"one int4 element, through io and the bigarray" A.Done
    (int4 1 1) (nibble 1 1);
  check ~msg:"the other int4 element of the byte" A.Overlapping (int4 0 1)
    (nibble 1 1);
  check ~msg:"an int4 element of the next byte" A.Done (int4 2 1)
    (nibble 1 1)

(* nx_read compares host-addressed operands by address: a borrow of an io
   memory and a bigarray over its pages. *)
let test_door_io_pages () =
  let _, host, pages = mapped_io () in
  let borrow_at offset = A.v f32 (L.v ~offset ~strides:[| 1 |] [| 2 |]) host in
  let over first =
    A.v f32 (L.contiguous [| 2 |])
      (B.of_bigarray (Bigarray.Array1.sub pages (4 * first) 8))
  in
  let x = zeros [| 2 |] in
  equal ~msg:"meeting" answer A.Overlapping (S.add (borrow_at 0) x (over 1));
  equal ~msg:"beside" answer A.Done (S.add (borrow_at 0) x (over 2));
  equal ~msg:"identical" answer A.Done (S.add (borrow_at 2) x (over 2))

(* The door's overlap is Rig.Buffer.share_byte of the arrays' bytes, and its
   identity is one width and one first bit, wherever the bytes lie: views of
   an io memory, of its borrow, of a bigarray over its pages and of plain
   host memory, of uint16, uint8 and int4. *)
type side = Io | Borrow | Pages | Plain

type operand = { side : side; bits : int; offset : int; count : int }

let pp_operand ppf o =
  Format.fprintf ppf "%s %d-bit elements %d+%d"
    (match o.side with
    | Io -> "io"
    | Borrow -> "borrow"
    | Pages -> "pages"
    | Plain -> "plain")
    o.bits o.offset o.count

let operand_gen =
  let open Gen in
  let* side = one_of (List.map constant [ Io; Borrow; Pages; Plain ]) in
  let* bits = one_of (List.map constant [ 4; 8; 16 ]) in
  let n = 256 / bits in
  let* offset = int_range 0 (n - 1) in
  let+ count = int_range 1 (n - offset) in
  { side; bits; offset; count }

(* A pair, the second a copy of the first's elements on another side as
   often as not, so that identical pairs come up. *)
let operands_gen =
  let open Gen in
  let* a = operand_gen in
  let* copy = bool in
  let* side = one_of (List.map constant [ Io; Borrow; Pages; Plain ]) in
  let+ b = operand_gen in
  (a, if copy then { a with side } else b)

let operands_gen =
  Gen.with_pp
    (fun ppf (a, b) ->
      Format.fprintf ppf "%a into %a" pp_operand b pp_operand a)
    operands_gen

(* The four memories, made in the first case. *)
let sides =
  lazy
    (let io, host, pages = mapped_io () in
     let plain =
       B.of_bigarray (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 32)
     in
     (io, host, B.of_bigarray pages, plain))

let law_door_overlaps =
  let buffer side =
    let io, host, pages, plain = Lazy.force sides in
    match side with
    | Io -> io
    | Borrow -> host
    | Pages -> pages
    | Plain -> plain
  in
  let array o =
    let b = buffer o.side in
    match o.bits with
    | 4 -> view D.Int4 b o.offset o.count
    | 8 -> view D.Uint8 b o.offset o.count
    | _ -> view D.Uint16 b o.offset o.count
  in
  (* [o]'s bytes from byte [first] on, [length] of them. *)
  let bytes o first length = B.view (buffer o.side) ~first ~length in
  let span o =
    let lo = o.offset * o.bits / 8 in
    bytes o lo ((((o.offset + o.count) * o.bits) + 7) / 8 - lo)
  in
  fun (a, b) ->
    let meet = B.share_byte (span a) (span b) in
    let first o = o.offset * o.bits in
    let identical =
      a.bits = b.bits && a.count = b.count
      && first a mod 8 = first b mod 8
      && B.share_byte (bytes a (first a / 8) 1) (bytes b (first b / 8) 1)
    in
    cover "a pair the host does not address both of"
      (a.side = Io || b.side = Io);
    cover "sub-byte" (a.bits = 4);
    cover "meeting" meet;
    cover "identical" (meet && identical);
    cover "beside" (not meet);
    let want = if meet && not identical then A.Overlapping else A.Done in
    equal (pair answer string)
      (want, if want = A.Done then "ran" else "not run")
      (ocaml_door [| array a |] [| array b |])

let test_ocaml_door_raises () =
  let z = A.Any (zeros [| 2 |]) and x = A.Any (floats32 [| 2 |] [| 1.; 2. |]) in
  raises Raised (fun () -> A.door ~written:[| z |] ~read:[| x |] raise Raised);
  all_released [| z; x |]

(* Device work: operands on a Late device, whose work runs only once a wait
   sleeps on it. A kernel waits for the work its operands' accesses follow and
   then runs: a store that ran after the kernel would show in its result. *)

let test_wait_add () =
  let d, _ = late () in
  let sums = [| [| 11.; 22. |]; [| 110.; 220. |]; [| 101.; 202. |] |] in
  for k = 0 to 2 do
    let z = on_late d [| 0.; 0. |] and x = on_late d [| 1.; 2. |] in
    let y = on_late d [| 10.; 20. |] in
    pending [| z; x; y |].(k) [| 100.; 200. |];
    let msg = strf "work on operand %d" k in
    equal ~msg answer A.Done (S.add z x y);
    equal ~msg (values f32) sums.(k) (A.to_array z)
  done

let test_wait_of_array () =
  let d, _ = late () in
  let a = on_late d [| 0.; 0. |] in
  pending a [| 3.; 4. |];
  equal answer A.Done (S.of_array_into a [| 7.; 8. |]);
  equal (values f32) [| 7.; 8. |] (A.to_array a)

(* A gather reads its source and writes its destination after the work on
   each. *)
let test_wait_copy () =
  let d, _ = late () in
  let a = on_late d [| 0.; 0. |] in
  pending a [| 3.; 4. |];
  let dst = on_late d [| 0.; 0. |] in
  pending dst [| 5.; 6. |];
  equal answer A.Done (S.copy_into dst a);
  equal (values f32) [| 3.; 4. |] (A.to_array dst)

(* A device lost while a kernel waits for its work raises Lost, and the kernel
   leaves no claim behind on the live arrays. *)
let test_wait_lost () =
  let d, l = late () in
  let x = on_late d [| 1.; 2. |] in
  let z = zeros [| 2 |] and y = floats32 [| 2 |] [| 1.; 2. |] in
  pending x [| 3.; 4. |];
  S.Late.fault l "gone";
  raises_match lost (fun () -> S.add z x y);
  List.iter (fun a -> released (A.buffer a)) [ z; y ]

(* A read operand identical in bytes to the written one, but of other memory,
   keeps its own claim and wait: a host borrow of a closed device's memory
   raises its loss, though the written operand, a host wrapper of the same
   bytes, has none. *)
let test_identical_owners () =
  let d = Result.get_ok (Rig.software "array:identical-owners") in
  let h = Option.get (B.borrow Rig.host (B.create d 12)) in
  let w = B.of_bigarray (B.read h Bigarray.char Fun.id) in
  Rig.close d;
  let x = A.v f32 (L.contiguous [| 3 |]) h in
  let z = A.v f32 (L.contiguous [| 3 |]) w in
  raises_match ~msg:"the kernel" lost (fun () -> S.add z x (zeros [| 3 |]))

(* The OCaml door over a lost device's array raises Lost, having claimed and
   run nothing, as every use of that memory does. *)
let test_ocaml_door_lost () =
  let d, _ = late () in
  let x = on_late d [| 1.; 2. |] and y = floats32 [| 2 |] [| 1.; 2. |] in
  Rig.close d;
  raises_match lost (fun () ->
      A.door ~written:[| A.Any y |] ~read:[| A.Any x |]
        (fun () -> fail "the door ran its function")
        ());
  released (A.buffer y)

(* Work pending on an array: the door runs its function at once, under the
   claims, and leaves the work for the device to order. *)
let test_ocaml_door_pending () =
  let d, _ = late () in
  let x = on_late d [| 1.; 2. |] in
  pending x [| 3.; 4. |];
  let work () = (Rig.signaled d, Rig.submitted d) in
  let seen = ref (0, 0) in
  equal answer A.Done
    (A.door ~written:[| A.Any x |] ~read:[||] (fun () -> seen := work ()) ());
  satisfies ~claim:"ran before the pending work: signaled below submitted"
    (pair int int)
    (fun (signaled, submitted) -> signaled < submitted)
    !seen;
  released (A.buffer x)

(* Two domains: kernels that read an array on a Late device, against submits
   that write it and donations that take it. Every submit stores ones, so a
   kernel that ran reads ones. A kernel refuses the array as dead and for no
   other reason: one that meets a take under way waits for it to end. *)
type shared = { mutable dead : bool }

let shared =
  abstract ~pp:(fun ppf r -> Format.fprintf ppf "dead %b" r.dead) "m"

let ones = Array.make 4 1.
let shared_device = lazy (fst (late ()))

let model =
  Testable.make
    ~pp:(fun ppf r -> Format.fprintf ppf "dead %b" r.dead)
    ~equal:( == )

let live r = satisfies ~claim:"live in the model" model (fun r -> not r.dead) r
let dead r = satisfies ~claim:"dead in the model" model (fun r -> r.dead) r

let judge_kernel r = function
  | Ok (A.Done, z) ->
      live r;
      equal ~msg:"x + x" (values f32) (Array.make 4 2.) z
  | Ok (A.Dead_buffer, _) -> dead r
  | Ok (e, _) -> failf "the kernel answered %a" (Testable.pp answer) e
  | Error e -> raise e

let judge_live r = function
  | Ok () -> live r
  | Error (Invalid_argument _) -> dead r
  | Error e -> raise e

let judge_donate r = function
  | Ok consumed ->
      live r;
      r.dead <- consumed
  | Error (Invalid_argument _) -> dead r
  | Error e -> raise e

let kernel x =
  let z = zeros [| 4 |] in
  let e = S.add z x x in
  (e, if e = A.Done then A.to_array z else [||])

let to_array x = equal (values f32) ones (A.to_array x)

let read x =
  equal (values f32) ones
    (A.read x (fun g ->
         Array.init 4 (fun i -> Bigarray.Genarray.get g [| i |])))

(* A submit under a keep, so that no take kills [x] while it runs. *)
let submit x =
  let k = B.keep [ A.buffer x ] in
  Fun.protect ~finally:(fun () -> B.release k) (fun () -> pending x ones)

let donate x = Option.is_some (B.take ~why:"donated" [ A.buffer x ])

let door_commands =
  [
    command "make"
      (Gen.unit @-> makes shared)
      (fun () -> { dead = false })
      (fun () -> on_late (Lazy.force shared_device) ones);
    command "kernel"
      (shared ^-> judges (pair answer (values f32)))
      judge_kernel kernel;
    command "to_array" (shared ^-> judges unit) judge_live to_array;
    command "read" (shared ^-> judges unit) judge_live read;
    command "submit" (shared ^-> judges unit) judge_live submit;
    command "donate" (shared ^-> judges bool) judge_donate donate;
  ]

let tests =
  [
    group "nx_read"
      [
        test "a kernel reads its operands through the door" test_door_add;
        test "another dtype or shape at any position is refused untouched"
          test_door_positions;
        test "a written operand must be distinct and alone" test_door_written;
        test "a written operand may be identical to read ones"
          test_door_identical;
        test "dead and foreign buffers are refused" test_door_buffers;
        test "reads up to max_operands operands and refuses more"
          test_door_arity;
        test "an operand with no element passes, on the host or off it"
          test_door_empty;
        test "releases its claims" test_door_releases;
        test "a read survives a moving collection" test_door_moving_gc;
        test "host-addressed operands compare by address" test_door_io_pages;
        prop
          "fills a descriptor with the layout's rank, flags, offset, extents \
           and strides"
          any_layout law_descriptor;
        test "NX_MAX_NUMEL is Layout.max_numel" test_max_numel;
        test "a kernel runs after the work on each operand" test_wait_add;
        test "of_array's store runs after the work on its array"
          test_wait_of_array;
        test "a gather runs after the work on its source and destination"
          test_wait_copy;
        test "a device lost during the wait raises Lost and leaves no claim"
          test_wait_lost;
        test "a read operand identical to the written one keeps its own wait"
          test_identical_owners;
      ];
    test "max_operands is 18" (fun () -> equal int 18 A.max_operands);
    test "C answers the constructors' codes" test_codes;
    group "door"
      [
        cases "claims every array while its function runs, then releases them"
          ~id:(fun (n, twice) ->
            Printf.sprintf "%d arrays%s" n
              (if twice then ", one read twice" else ""))
          door_counts test_ocaml_door_counts;
        test "refuses before claiming or running anything"
          test_ocaml_door_refuses;
        test "a refusal in a later run of claims leaves no claim"
          test_ocaml_door_late_refusals;
        test "answers a call of many arrays as one of few"
          test_ocaml_door_many_refusals;
        test "refuses a dead array before a written one's other faults"
          test_ocaml_door_order;
        test "a written array may be identical to read ones, claimed once"
          test_ocaml_door_identical;
        test "off the host, views overlap by their offsets"
          test_ocaml_door_off_host;
        test "bigarrays over one memory overlap by their addresses"
          test_ocaml_door_bigarrays;
        test "an io memory overlaps its borrow and its pages' bigarrays"
          test_ocaml_door_io_pages;
        prop ~count:500
          "overlap is Rig.Buffer.share_byte of the arrays' bytes, identity one \
           width and one first bit"
          operands_gen law_door_overlaps;
        test "releases its claims when its function raises"
          test_ocaml_door_raises;
        test "holds the lock for no long stretch, however many arrays"
          test_ocaml_door_holds;
        test "allocates nothing" test_door_allocates_nothing;
        test "raises Lost for a lost device's array" test_ocaml_door_lost;
        test "runs before work pending on its arrays" test_ocaml_door_pending;
        (* Each case runs its program 50 times on two domains, and --prop-count
           raises the 100 cases as it does a property's. Every command but make
           runs in the branches, and make needs one call, so the prefix and the
           suffix make at most five. *)
        stateful ~domains:2 ~count:100 ~steps:5
          "kernels against submits and donations on another domain"
          door_commands;
      ];
    group "refused"
      [
        test "names the function, the reason and every operand" test_refused;
        test "gives each refusal its own reason" test_refused_codes;
        test "names the reason of each refusal the door answers"
          test_door_reasons;
        test "names a dead operand's consumer, before the operands"
          test_refused_dead;
      ];
  ]

let () = exit (run "nx_array.door" tests)
