(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Device_metal_support

let strf = Printf.sprintf

(* The ring

   A model of the ring of a device's command buffers: slots taken in commit
   order, completed in any order, released in commit order. The word is the last
   value released before the first failed slot. *)

module Ring = struct
  type state = Taken | Done | Failed

  type slot = {
    index : int;
    k : int;
    v : int;
    mutable state : state;
    mutable releases : int list;
  }

  type t = {
    n : int;
    mutable taken : slot list;
    mutable tail : int;
    mutable values : int;
    mutable commits : int;
    mutable completed : int list;
    mutable word : int;
    mutable stopped : bool;
    mutable failure : string option;
    mutable ran : int list;
    mutable releases : int;
    mutable over : bool;
  }

  let make n =
    {
      n;
      taken = [];
      tail = 0;
      values = 0;
      commits = 0;
      completed = [];
      word = 0;
      stopped = false;
      failure = None;
      ran = [];
      releases = 0;
      over = false;
    }

  let commit m ~last =
    cover "a slot is taken again" (m.tail >= m.n);
    let v = if last then m.values + 1 else 0 in
    let s =
      { index = m.tail mod m.n; k = m.commits; v; state = Taken; releases = [] }
    in
    if last then m.values <- v;
    m.taken <- m.taken @ [ s ];
    m.tail <- m.tail + 1;
    m.commits <- m.commits + 1;
    s.index

  let rec release m = function
    | s :: rest when s.state <> Taken ->
        if s.state = Failed then m.stopped <- true;
        cover "a value completes after a failed slot" (m.stopped && s.v > 0);
        if s.v > 0 && not m.stopped then m.word <- s.v;
        m.ran <- s.releases @ m.ran;
        release m rest
    | rest -> rest

  let complete m i ~failed =
    let s = List.find (fun s -> s.index = i) m.taken in
    cover "a slot completes before an earlier one" (s != List.hd m.taken);
    s.state <- (if failed then Failed else Done);
    m.completed <- s.k :: m.completed;
    if failed && m.failure = None then
      m.failure <- Some (strf "command buffer %d failed" s.k);
    m.taken <- release m m.taken

  let defer m =
    let id = m.releases in
    m.releases <- id + 1;
    begin match List.rev m.taken with
    | last :: _ -> last.releases <- id :: last.releases
    | [] -> m.ran <- id :: m.ran
    end;
    id

  let stop m =
    m.over <- true;
    if m.taken = [] then m.word <- m.values;
    m.taken = []

  let pending m =
    List.filter_map
      (fun s -> if s.state = Taken then Some s.index else None)
      m.taken

  let times m k =
    if List.mem k m.completed then ((10 * k) + 1, (10 * k) + 2) else (0, 0)
end

let ring_invariant (m : Ring.t) r =
  equal int ~msg:"word" m.word (S.word r);
  equal (option string) ~msg:"failure" m.failure (S.failure r);
  equal (slist int compare) ~msg:"releases run" m.ran (Array.to_list (S.ran r));
  for k = 0 to m.commits - 1 do
    equal (pair int int)
      ~msg:(strf "times of commit %d" k)
      (Ring.times m k) (S.times r k)
  done

let ring = abstract "r" ~invariant:ring_invariant
let slot = among int ring Ring.pending
let open_ (m : Ring.t) = not m.over

let ring_commands =
  [
    command "ring" (Gen.int_range 1 6 @-> makes ring) Ring.make S.ring;
    command "commit"
      ~pre:(fun (m : Ring.t) _ -> open_ m && List.length m.taken < m.n)
      (ring ^-> Gen.bool @-> returns int)
      (fun m last -> Ring.commit m ~last)
      (fun r last -> S.commit r ~last);
    command "complete"
      ~pre:(fun m _ _ -> open_ m)
      (ring ^-> slot ^-> Gen.bool @-> returns unit)
      (fun m i failed -> Ring.complete m i ~failed)
      (fun r i failed -> S.complete r i ~failed);
    command "defer" ~pre:open_ (ring ^-> returns int) Ring.defer S.defer;
    command "sleep" ~pre:open_
      (ring ^-> returns (option string))
      (fun (m : Ring.t) -> m.failure)
      S.sleep;
    command "stop" ~pre:open_ (ring ^-> returns bool) Ring.stop S.stop;
  ]

let ring_tests =
  group ~timeout:60. "ring"
    [
      stateful ~count:500 ~steps:40
        "releases in commit order whatever order slots complete in"
        ring_commands;
    ]

let () = exit (run "device_metal" [ ring_tests ])
