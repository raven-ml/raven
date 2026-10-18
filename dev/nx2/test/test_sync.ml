(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Waits and checks: a check off the host raises at the next wait on its set,
   on the domain that issued it, earliest first and once; on the host, at the
   call. *)

open Windtrap
module A = Nx_array

let m = Nx_support.memory

module S = (val Nx.devices [ m 0 ])
module T = (val Nx.devices [ m 1 ])
module S2 = (val Nx.devices [ m 2; m 3 ])

let host_of dt s xs = Nx.Repr.of_array Nx.Host.v (A.of_array dt s xs)
let numel s = Array.fold_left ( * ) 1 s

let check ok data fail =
  Nx.Prim.eval ~by:"test" (Nx.Prim.Check { ok; data; fail })

(* The index of the flat position [i] of a value of shape [s], in C order. *)
let index s i =
  let idx = Array.make (Array.length s) 0 and k = ref i in
  for a = Array.length s - 1 downto 0 do
    idx.(a) <- !k mod s.(a);
    k := !k / s.(a)
  done;
  idx

exception Failed of int
exception At of int array * float

type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

(* Waits *)

type wait = Sync | To_array | Item | Print | Place_host

let pp_wait ppf w =
  Format.pp_print_string ppf
    (match w with
    | Sync -> "sync"
    | To_array -> "to_array"
    | Item -> "item"
    | Print -> "to_string"
    | Place_host -> "place on the host")

let waits = Gen.of_list ~pp:pp_wait [ Sync; To_array; Item; Print; Place_host ]

let wait w x =
  match w with
  | Sync -> Nx.sync x
  | To_array -> ignore (Nx.to_array x)
  | Item -> ignore (Nx.item [ 0 ] x)
  | Print -> ignore (Nx.to_string x)
  | Place_host -> ignore (Nx.place Nx.Host.on x)

(* A set seen through a value on it and two flags, one false at position 1. *)
type side = { wait : wait -> unit; issue : bool -> int -> unit }

let side (type d) (module D : Nx.Devices with type d = d) =
  let x = Nx.place D.on (host_of Nx.float32 [| 3 |] [| 1.; 2.; 3. |]) in
  let flags b = Nx.place D.on (host_of Nx.bool [| 3 |] [| true; b; true |]) in
  let pass = flags true and fail = flags false in
  {
    wait = (fun w -> wait w x);
    issue =
      (fun fails id ->
        check (if fails then fail else pass) [] (fun _ () -> Failed id));
  }

let s_side = side (module S)
let t_side = side (module T)

(* The checks pending on each set, as [(id, fails)] in issue order. *)
type model = { mutable on_s : (int * bool) list; mutable on_t : (int * bool) list }

type set = On_s | On_t

let sets =
  Gen.of_list
    ~pp:(fun ppf s -> Format.pp_print_string ppf (if s = On_s then "S" else "T"))
    [ On_s; On_t ]

let side_of = function On_s -> s_side | On_t -> t_side

let pending md = function On_s -> md.on_s | On_t -> md.on_t

let set_pending md s l =
  match s with On_s -> md.on_s <- l | On_t -> md.on_t <- l

(* A wait drains its set's checks and raises the earliest failure, the later
   ones dropped. *)
let settle md s =
  let rec go = function
    | [] -> None
    | (_, false) :: rest ->
        cover "a wait passes a check before a failing one" (rest <> []);
        go rest
    | (id, true) :: rest ->
        cover "a wait drops a later failure" (List.exists snd rest);
        Some id
  in
  let raised = go (pending md s) in
  set_pending md s [];
  raised

(* Drains a set's checks, so that no test sees another's. *)
let drain side = try side.wait Sync with Failed _ -> ()

(* The checks are the domain's, so every session shares one model, which a
   start empties. A program's end drains what it left. *)
let model = { on_s = []; on_t = [] }

let session =
  abstract "w" ~release:(fun () ->
      drain s_side;
      drain t_side)

let pending_checks =
  stateful ~steps:30
    "a wait drains its set's checks, raising the earliest failure"
    [
      command "start" (Gen.unit @-> makes session)
        (fun () ->
          model.on_s <- [];
          model.on_t <- [];
          model)
        (fun () ->
          drain s_side;
          drain t_side);
      command "issue"
        (session ^-> sets @-> Gen.bool @-> Gen.int_range 0 99 @-> returns unit)
        (fun md s fails id -> set_pending md s (pending md s @ [ (id, fails) ]))
        (fun () s fails id -> (side_of s).issue fails id);
      command "wait"
        (session ^-> sets @-> waits @-> returns (option int))
        (fun md s _ ->
          let raised = settle md s in
          cover "a wait on a set with a failure on the other"
            (raised = None
            && List.exists snd (pending md (if s = On_s then On_t else On_s)));
          raised)
        (fun () s w ->
          match (side_of s).wait w with
          | () -> None
          | exception Failed id -> Some id);
    ]

(* A failed check's index and data *)

type at = One | Split

let pp_at ppf a =
  Format.pp_print_string ppf (match a with One -> "one device" | Split -> "split")

(* A flag's shape, where it lies, and two flat positions of [false], which may
   coincide. A split shape doubles its first extent to divide evenly. *)
let failures =
  Gen.triple
    (Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 4))
    (Gen.of_list ~pp:pp_at [ One; Split ])
    (Gen.pair Gen.nat Gen.nat)
  |> Gen.map (fun (s, at, (i, j)) ->
      let at = if Array.length s = 0 then One else at in
      let s =
        if at = Split then Array.mapi (fun a e -> if a = 0 then 2 * e else e) s
        else s
      in
      let n = numel s in
      let i, j = if n = 0 then (0, 0) else (i mod n, j mod n) in
      (s, at, (Int.min i j, Int.max i j)))

let elements =
  prop "a failed check names its first false and its data there" failures
    (fun (s, at, (i, j)) ->
      let n = numel s in
      cover "empty" (n = 0);
      cover "split" (at = Split && n > 0);
      cover "two falses" (i <> j);
      cover "rank above 1" (Array.length s > 1 && n > 0);
      let ok = host_of Nx.bool s (Array.init n (fun k -> k <> i && k <> j)) in
      let d = host_of Nx.float32 s (Array.init n Float.of_int) in
      let issue (type d) (place : (bool, _, Nx.host) Nx.t -> (bool, _, d) Nx.t)
          (place_d : (float, _, Nx.host) Nx.t -> (float, _, d) Nx.t) =
        let d = place_d d in
        check (place ok) [ d ] (fun idx (v, ()) -> At (idx, v));
        d
      in
      (* The check raises once, at its first false. *)
      let settled (type d) (d : (float, _, d) Nx.t) =
        if n > 0 then raises (At (index s i, Float.of_int i)) (fun () -> Nx.sync d);
        Nx.sync d
      in
      match at with
      | One -> settled (issue (Nx.place S.on) (Nx.place S.on))
      | Split ->
          let p = S2.split ~axis:0 in
          settled (issue (Nx.place p) (Nx.place p)))

(* At the call *)

let at_call =
  let ok = [| true; false; true |] in
  cases
    ~name:(fun (name, _) -> name)
    "a check raises at the call"
    [
      ( "on the host",
        fun () -> check (host_of Nx.bool [| 3 |] ok) [] (fun _ () -> Failed 0) );
      ( "of every set",
        fun () ->
          check (Nx.Infix.(Nx.arange Nx.int32 0 3 1 <>$ 1l)) [] (fun _ () ->
              Failed 0) );
    ]
    (fun (_, f) -> raises (Failed 0) f)

(* Systhreads *)

(* The systhreads of a domain share its checks. A signal every 20 us of CPU
   time yields the running systhread at its next poll point, so that two
   issuers interleave inside nx's own updates. It is the virtual timer's: the
   real-time one is windtrap's, for its timeouts. Each of [sets] sets gets one failing
   check a round, and every one must raise. *)
let systhreads () =
  let sets =
    Array.init 6000 (fun _ ->
        let module D = (val Nx.devices [ m 0 ]) in
        side (module D))
  in
  let tick = { Unix.it_interval = 2e-5; it_value = 2e-5 } in
  let still = { Unix.it_interval = 0.; it_value = 0. } in
  let yield = Sys.Signal_handle (fun _ -> Thread.yield ()) in
  let issue first () =
    for i = 0 to (Array.length sets / 2) - 1 do
      sets.((2 * i) + first).issue true i
    done
  in
  let round () =
    let a = Thread.create (issue 0) () and b = Thread.create (issue 1) () in
    Thread.join a;
    Thread.join b;
    Array.fold_left
      (fun lost side ->
        match side.wait Sync with () -> lost + 1 | exception Failed _ -> lost)
      0 sets
  in
  let previous = Sys.signal Sys.sigvtalrm yield in
  ignore (Unix.setitimer Unix.ITIMER_VIRTUAL tick);
  let lost =
    Fun.protect
      ~finally:(fun () ->
        ignore (Unix.setitimer Unix.ITIMER_VIRTUAL still);
        Sys.set_signal Sys.sigvtalrm previous)
      (fun () -> List.init 5 (fun _ -> round ()))
  in
  equal (list int) [ 0; 0; 0; 0; 0 ] lost

(* Domains, donation and sync *)

(* A loop that draws over a fresh parameter at [at] each step and never waits
   on its set: once warm, device [d] holds as many bytes after 200 more steps
   as after the first, its collected memory moved and its cache freed
   alike. *)
let bounded at d =
  let n = 1 lsl 16 in
  let p = host_of Nx.float32 [| n |] (Array.make n 0.5) in
  let step () = ignore (Nx.Rng.bernoulli (Nx.copy (Nx.place at p))) in
  let settled () =
    for _ = 1 to 4 do
      Gc.full_major ();
      Rig.free_cache d
    done;
    Rig.allocated d
  in
  step ();
  let warm = settled () in
  for _ = 1 to 200 do
    step ()
  done;
  at_most int ~than:warm (settled ())

let others =
  group "domains, donation and sync"
    [
      test "a check raises only on the domain that issued it" (fun () ->
          let raised =
            Domain.join
              (Domain.spawn (fun () ->
                   s_side.issue true 7;
                   match s_side.wait Sync with
                   | () -> None
                   | exception Failed id -> Some id))
          in
          equal (option int) (Some 7) raised;
          Domain.join (Domain.spawn (fun () -> s_side.issue true 8));
          s_side.wait Sync);
      test "a check keeps its data's elements through a donation" (fun () ->
          let d = Nx.place S.on (host_of Nx.float32 [| 3 |] [| 4.; 5.; 6. |]) in
          let ok = Nx.place S.on (host_of Nx.bool [| 3 |] [| true; false; true |]) in
          check ok [ d ] (fun idx (v, ()) -> At (idx, v));
          let negated = Nx.neg (Nx.donate d) in
          raises (At ([| 1 |], 5.)) (fun () -> Nx.sync negated);
          equal (array float_exact) [| -4.; -5.; -6. |] (Nx.to_array negated));
      test "a donation after a sync reuses the checked data's memory"
        (fun () ->
          (* A copy owns its memory: a placement onto a memory device borrows
             the host's, which no donation writes. *)
          let d = host_of Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
          let d = Nx.copy (Nx.place S.on d) in
          let ok = host_of Nx.bool [| 3 |] [| true; true; true |] in
          let ok = Nx.place S.on ok in
          check ok [ d ] (fun idx (v, ()) -> At (idx, v));
          Nx.sync d;
          (* The count includes cached memory, which a refused donation would
             take unseen. *)
          Rig.free_cache (m 0);
          let before = Rig.allocated (m 0) in
          ignore (Nx.neg (Nx.donate d));
          equal int before (Rig.allocated (m 0)));
      test "a check that raises at issue holds none of its data" (fun () ->
          let d = Nx.place S.on (host_of Nx.float32 [| 1 |] [| 1. |]) in
          let ok = Nx.place S.on (host_of Nx.bool [| 1 |] [| false |]) in
          check ok [ d ] (fun _ _ -> raise Exit);
          (* The wait that reads the failing check again ends its hold. *)
          let drain () = try Nx.sync d with Exit -> () in
          Fun.protect ~finally:drain (fun () ->
              let d' = host_of Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
              let d' = Nx.copy (Nx.place S.on d') in
              let ok' = host_of Nx.bool [| 3 |] [| true; true; true |] in
              let ok' = Nx.place S.on ok' in
              raises Exit (fun () -> check ok' [ d' ] (fun _ _ -> Failed 0));
              Rig.free_cache (m 0);
              let before = Rig.allocated (m 0) in
              ignore (Nx.neg (Nx.donate d'));
              equal int before (Rig.allocated (m 0))));
      test "checks no wait follows hold bounded memory" (fun () ->
          bounded S.on (m 0));
      test "a check on a device then idle leaves later checks bounded"
        (fun () ->
          let on k = Nx.Placement.device S2.v (m k) in
          let p = host_of Nx.float32 [| 4 |] (Array.make 4 0.5) in
          ignore (Nx.Rng.bernoulli (Nx.copy (Nx.place (on 3) p)));
          bounded (on 2) (m 2));
      test "a print that raises a failed check prints nothing" (fun () ->
          let d = Nx.place S.on (host_of Nx.float32 [| 2 |] [| 1.; 2. |]) in
          let ok = Nx.place S.on (host_of Nx.bool [| 2 |] [| false; true |]) in
          check ok [ d ] (fun idx (v, ()) -> At (idx, v));
          let b = Buffer.create 16 in
          let ppf = Format.formatter_of_buffer b in
          raises (At ([| 0 |], 1.)) (fun () -> Nx.pp ppf d);
          Format.pp_print_flush ppf ();
          equal string "" (Buffer.contents b));
      test "an empty check passes" (fun () ->
          let ok = Nx.place S.on (host_of Nx.bool [| 0; 2 |] [||]) in
          check ok [] (fun _ () -> Failed 0);
          Nx.sync ok);
      test "sync returns for a value of every set" (fun () ->
          Nx.sync (Nx.arange Nx.int32 0 3 1));
      test "sync refuses a dead value" (fun () ->
          let x = Nx.place S.on (host_of Nx.float32 [| 2 |] [| 1.; 2. |]) in
          ignore (Nx.neg (Nx.donate x));
          raises_match (Exn.invalid_arg ~substring:"Nx.sync: ") (fun () ->
              Nx.sync x));
      test "sync refuses a traced value" (fun () ->
          let traced i ~by op =
            Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form Mark) op
          in
          Nx.Prim.interpret ~name:"test.trace" Extent traced (fun _ ->
              let x = Nx.neg (host_of Nx.float32 [| 2 |] [| 1.; 2. |]) in
              raises
                (Invalid_argument
                   "Nx.sync: the value is traced by test.trace")
                (fun () -> Nx.sync x)));
    ]

let () =
  exit
    (run "nx waits"
       [
         pending_checks;
         elements;
         at_call;
         test "checks issued from two systhreads all raise" systhreads;
         others;
       ])
