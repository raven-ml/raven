(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Proxies of another machine, opened through rig over a real link on loopback
   to an agent loop (Proxy_machine). *)

open Windtrap
open Proxy_machine
module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link
module Proxy = Rig_remote_proxy
module B = Rig.Buffer
module Sub = Rig.Submission

(* The bytes in flight past which a proxy's room check answers later. *)
let in_flight_limit = 64 lsl 20

(* Facts *)

let rec_w =
  Testable.make
    ~pp:(fun ppf -> function
      | Rig_remote_abi.Host h -> Format.fprintf ppf "Host %s" h.machine
      | Rig_remote_abi.Device { id } -> Format.fprintf ppf "Device %d" id)
    ~equal:( == )

let completion_w =
  Testable.structural ~pp:(fun ppf -> function
    | Rig_edge.Host -> Format.pp_print_string ppf "Host"
    | Store -> Format.pp_print_string ppf "Store"
    | Object h -> Format.fprintf ppf "Object %nd" h)

let driver_facts () =
  with_job_link @@ fun far ->
  let a = account 3 ~reaches:[] in
  let c = Rig_remote_abi.Device { id = 3 } in
  let p = Proxy.make far a c in
  let f = Proxy.facts p in
  equal (list string) [ "COMPUTE:0"; "COPY:0" ]
    (List.map (fun (q : Rig_edge.queue) -> q.name) f.queues);
  equal ~msg:"a device's queues run copies" bool true
    (List.for_all (fun (q : Rig_edge.queue) -> q.runs = [ Copy ]) f.queues);
  equal completion_w Host f.completion;
  equal (list bool) [ true; false; false ]
    [ f.waits.hosts; f.waits.stores; f.waits.objects ];
  equal int max_int f.waits.most;
  equal bool true f.may_block;
  equal bool false f.maps_host;
  equal rec_w c (Proxy.capability p);
  equal string "mem3" f.arch;
  equal int ((1 lsl 30) + 3) f.budget

let rig_facts () =
  with_machine ~reaches:[ [] ] @@ fun m ->
  let d = m.devices.(0) in
  equal string "mem1" (Rig.arch d);
  equal int ((1 lsl 30) + 1) (Rig.budget d);
  equal bool true (Rig.equal m.host (Rig.host_of d));
  equal bool true (Rig.computes d);
  equal bool false (Rig.shares_host_memory m.host);
  equal bool false (Rig.reaches m.host Rig.host);
  equal bool false (Rig.reaches Rig.host m.host);
  starts_with ~affix:"MEM:1@proxy-test:" (Rig.name d);
  match Rig.capability d Rig_remote_abi.key with
  | Some (Rig_remote_abi.Device { id }) -> equal int 1 id
  | _ -> fail "a device's record is Device"

(* The ids a device reaches decide what it maps of the others. *)
let reaches () =
  with_machine ~reaches:[ [ 2 ]; [] ] @@ fun m ->
  let d1 = m.devices.(0) and d2 = m.devices.(1) in
  equal bool true (Rig.reaches d1 d2);
  equal bool false (Rig.reaches d2 d1)

let two_links () =
  let job = Link.job () in
  Fun.protect
    ~finally:(fun () -> Link.fail job "the test ended")
    (fun () ->
      let link name i =
        let d, a = connected () in
        let l = Link.make job d ~name ~peer:(Wire.Agent i) in
        ignore (Link.make job a ~name:"controller" ~peer:Wire.Controller);
        l
      in
      let l1 = link "one" 1 and l2 = link "two" 2 in
      let device id = Rig_remote_abi.Device { id } in
      let p = Proxy.make l1 (account 1 ~reaches:[ 1; 2 ]) (device 1) in
      let q = Proxy.make l2 (account 2 ~reaches:[ 1; 2 ]) (device 2) in
      let q' = Proxy.make l1 (account 2 ~reaches:[]) (device 2) in
      equal bool false (Proxy.peer p q);
      equal bool true (Proxy.peer p q');
      let word p = (Proxy.facts p).word in
      let r = word q' in
      equal ~msg:"a word's handle" nativeint 0n (Proxy.locate r).handle;
      equal ~msg:"a word's shadow is this process's" bool true
        (Option.is_some (Proxy.locate r).host);
      equal ~msg:"map_host" bool true
        (Option.is_none (Proxy.map_host p 4096 4096));
      Link.fail job "lost";
      equal ~msg:"once lost" bool true (Proxy.peer p q');
      equal bool true (Option.is_none (Proxy.map_peer p q (word q)));
      match Proxy.map_peer p q' (word q') with
      | Some r -> equal (option int) (Some 2) (Proxy.locate r).address
      | None -> fail "a proxy of the same link maps the word")

let make_misuse () =
  with_job_link @@ fun far ->
  let rig_host = Rig_remote_abi.Host { machine = "m"; rail = no_rails } in
  raises_match ~msg:"a host's record for device 1" Exn.invalid_arg (fun () ->
      Proxy.make far (account 1 ~reaches:[]) rig_host);
  raises_match ~msg:"device 2's record for device 1" Exn.invalid_arg (fun () ->
      Proxy.make far (account 1 ~reaches:[]) (Rig_remote_abi.Device { id = 2 }));
  raises_match ~msg:"a device's record for the host" Exn.invalid_arg (fun () ->
      Proxy.make far (account 0 ~reaches:[]) (Rig_remote_abi.Device { id = 1 }))

let facts =
  group "facts"
    [
      test
        "a proxy's facts are the agent's account's and the ones its .mli lists"
        driver_facts;
      test
        "rig sees another machine's device, whose memory this process does not \
         reach"
        rig_facts;
      test "a device reaches the devices its account lists" reaches;
      test "proxies of two links never map each other; a word maps at its id"
        two_links;
      test "make raises for a record of another device" make_misuse;
    ]

(* Memory *)

let alloc_reaches_agent () =
  with_machine ~reaches:[ [] ] @@ fun m ->
  let b = B.create m.devices.(0) 4096 in
  let id = Nativeint.to_int (B.handle b) in
  let device, bytes =
    require_match
      (List.find_map (function
        | Alloc a when a.id = id -> Some (a.device, a.bytes)
        | _ -> None))
      (events m.ag)
  in
  equal ~msg:"the allocation's device" int 1 device;
  at_least ~msg:"the allocation's bytes" int ~than:4096 bytes;
  raises_match ~msg:"a proxy's memory has no address" Exn.invalid_arg (fun () ->
      B.address b)

let refused_alloc () =
  with_machine ~room:(1 lsl 20) @@ fun m ->
  ignore (B.create m.host 4096);
  raises_match
    (function Rig.Out_of_memory (_, n) -> n = 2 lsl 20 | _ -> false)
    (fun () -> B.create m.host (2 lsl 20))

let dropped () =
  with_machine @@ fun m ->
  let id =
    let b = B.create m.host 4096 in
    Nativeint.to_int (B.handle b)
  in
  Gc.full_major ();
  (* A drain takes the collected memory back into the cache. *)
  ignore (B.create m.host 16);
  Rig.free_cache m.host;
  until ~what:"the drop" (fun () -> List.mem (Drop id) (events m.ag))

(* A borrow maps the memory on the agent; a copy through it moves the memory's
   bytes. *)
let borrowed () =
  with_machine ~reaches:[ [ 2 ]; [] ] @@ fun m ->
  let d1 = m.devices.(0) and d2 = m.devices.(1) in
  let b2 = far_of_string d2 "mapped" in
  let b1 = require_some (B.borrow d1 b2) in
  let region = Nativeint.to_int (B.handle b2) in
  equal bool true
    (List.exists
       (function Map { device = 1; region = r; _ } -> r = region | _ -> false)
       (events m.ag));
  let dst = B.create d1 6 in
  B.copy ~src:b1 ~dst;
  equal string "mapped" (read dst)

let memory =
  group "memory"
    [
      test
        "a buffer on a proxy is an allocation of the agent, named by a handle"
        alloc_reaches_agent;
      test "an allocation the agent refuses raises Out_of_memory" refused_alloc;
      test "memory given back reaches the agent as a drop" dropped;
      test "a device borrows the memory of a device it reaches" borrowed;
    ]

(* Copies *)

let sizes =
  Gen.frequency
    [
      (3, Gen.int_range 0 5000);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 4095; 4096; 4097; 1 lsl 20 ]
      );
    ]

let round_trip n =
  with_machine ~reaches:[ [] ] @@ fun m ->
  List.iter
    (fun d ->
      let s = random_string n in
      equal ~msg:(Rig.name d) string s (read (far_of_string d s)))
    [ m.host; m.devices.(0) ]

(* A copy into this process's memory writes only the bytes its part names. *)
let only_named () =
  with_machine @@ fun m ->
  let far = far_of_string m.host (String.make 16 'f') in
  let h = host_buffer (String.make 64 '.') in
  B.copy ~src:far ~dst:(B.view h ~first:20 ~length:16);
  equal string
    (String.make 20 '.' ^ String.make 16 'f' ^ String.make 28 '.')
    (read_host h)

let within_machine () =
  with_machine ~reaches:[ [] ] @@ fun m ->
  let a = far_of_string m.host "within the machine" in
  let b = B.create m.host (B.length a) in
  B.copy ~src:a ~dst:b;
  let crossing =
    List.exists
      (function
        | Handover { parts; local; _ } ->
            local = 0
            && List.exists
                 (function
                   | Wire.Copy { src = Wire.Region _; dst = Wire.Region _; _ }
                     ->
                       true
                   | _ -> false)
                 parts
        | _ -> false)
      (events m.ag)
  in
  equal ~msg:"a copy between regions, with no bytes from here" bool true
    crossing;
  equal string "within the machine" (read b)

(* Bytes copied into this process are in place once the value is reached,
   whatever their size. *)
let bytes_before_word () =
  with_machine @@ fun m ->
  let n = 8 lsl 20 in
  let far =
    far_of_string m.host (String.init n (fun i -> Char.chr (i land 0xff)))
  in
  for round = 1 to 3 do
    let h = B.create Rig.host n in
    let p = submit (copy_submission m.host ~src:far ~dst:h) in
    Rig.Point.wait p;
    let ba = B.bigarray Bigarray.char h in
    let bad = ref (-1) in
    for i = n - 1 downto 0 do
      if ba.{i} <> Char.chr (i land 0xff) then bad := i
    done;
    equal
      ~msg:(Printf.sprintf "round %d: the first wrong byte" round)
      int (-1) !bad
  done

(* A law: copies submitted without waiting, between this process's memory and
   two devices of the machine, leave what the same copies leave run one after
   another. *)

type place = Here of int | There of int * int (* device, buffer *)
type op = { src : place; dst : place; at_src : int; at_dst : int; len : int }

let buffer_size = 64

let pp_place ppf = function
  | Here i -> Format.fprintf ppf "here%d" i
  | There (d, i) -> Format.fprintf ppf "dev%d.%d" d i

let pp_op ppf o =
  Format.fprintf ppf "%a[%d] -> %a[%d] x%d" pp_place o.src o.at_src pp_place
    o.dst o.at_dst o.len

let place_g =
  Gen.of_list ~pp:pp_place
    [ Here 0; Here 1; There (0, 0); There (0, 1); There (1, 0) ]

let op_g =
  let open Gen in
  let* src, dst =
    such_that
      (fun (s, d) ->
        s <> d
        &&
        match (s, d) with
        | Here _, Here _ -> false
        | There (a, _), There (b, _) -> a = b
        | _ -> true)
      (pair place_g place_g)
  in
  let* len = int_range 1 buffer_size in
  let+ at_src = int_range 0 (buffer_size - len)
  and+ at_dst = int_range 0 (buffer_size - len) in
  { src; dst; at_src; at_dst; len }

let device_of = function There (d, _) -> Some d | Here _ -> None

let copies_law ops =
  with_machine ~reaches:[ [] ] @@ fun m ->
  let devices = [| m.host; m.devices.(0) |] in
  let init = Array.init 5 (fun _ -> random_string buffer_size) in
  let model = Array.map Bytes.of_string init in
  let index = function
    | Here i -> i
    | There (0, i) -> 2 + i
    | There (_, _) -> 4
  in
  let buffers =
    Array.init 5 (fun i ->
        match i with
        | 0 | 1 -> host_buffer init.(i)
        | 2 | 3 -> far_of_string devices.(0) init.(i)
        | _ -> far_of_string devices.(1) init.(i))
  in
  let last = Array.make 2 0 in
  List.iteri
    (fun k o ->
      let d =
        match (device_of o.src, device_of o.dst) with
        | Some d, _ | None, Some d -> d
        | None, None -> assert false
      in
      cover "a copy into this process"
        (match o.dst with Here _ -> true | There _ -> false);
      let after_into same =
        match o.src with
        | Here _ ->
            List.exists
              (fun (j, o') ->
                j < k && o'.dst = o.src && device_of o'.src = Some d = same)
              (List.mapi (fun j o -> (j, o)) ops)
        | There _ -> false
      in
      cover "a copy from this process after a copy into it, on its device"
        (after_into true);
      cover "a copy from this process after a copy into it, on another device"
        (after_into false);
      let view p at = B.view buffers.(index p) ~first:at ~length:o.len in
      let s =
        copy_submission devices.(d) ~src:(view o.src o.at_src)
          ~dst:(view o.dst o.at_dst)
      in
      last.(d) <- Rig.Point.value (submit s);
      Bytes.blit model.(index o.src) o.at_src model.(index o.dst) o.at_dst o.len)
    ops;
  Array.iteri (fun d v -> if v > 0 then Rig.wait devices.(d) v) last;
  Array.iteri
    (fun i b ->
      let got = if i < 2 then read_host b else read b in
      equal
        ~msg:(Printf.sprintf "buffer %d" i)
        string
        (Bytes.to_string model.(i))
        got)
    buffers

let copies =
  group "copy"
    [
      prop ~count:50
        "bytes copied to each device of the machine and back are the bytes"
        sizes round_trip;
      test "a copy into this process writes only the bytes its part names"
        only_named;
      test "a copy within the machine moves no byte across the link"
        within_machine;
      test "a copy's bytes are in place once its value is reached, at 8 MiB"
        bytes_before_word;
      prop ~count:60
        "copies submitted at once leave what they leave one after another"
        (Gen.with_pp
           (Format.pp_print_list
              ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
              pp_op)
           (Gen.list ~size:(Gen.int_range 1 12) op_g))
        copies_law;
    ]

(* Work *)

let never () =
  with_machine ~reaches:[ [] ] @@ fun m ->
  let far = B.create m.host 16 and far2 = B.create m.host 16 in
  let h = B.create Rig.host 16 in
  let both =
    Sub.make ~reads:0 ~writes:0 m.host
    [|
      {
        Sub.queue = "COPY:0";
        after = [||];
        work = Sub.Copy { src = far; dst = h };
      };
      {
        Sub.queue = "COPY:0";
        after = [||];
        work = Sub.Copy { src = h; dst = far2 };
      };
    |]
  in
  raises_match ~msg:"a copy from here after a copy into here" Exn.invalid_arg
    (fun () -> submit both);
  let words = host_buffer "\001\000\000\000\002\000\000\000" in
  let on d =
    Sub.make ~reads:0 ~writes:0 d
      [| { Sub.queue = "COMPUTE:0"; after = [||]; work = Sub.Words words } |]
  in
  raises_match ~msg:"words on a device other than the host" Exn.invalid_arg
    (fun () -> submit (on m.devices.(0)));
  let p = submit (on m.host) in
  Rig.Point.wait p;
  equal bool true
    (List.exists
       (function
         | Handover { device = 0; parts = [ Wire.Words w ]; _ } ->
             w = "\001\000\000\000\002\000\000\000"
         | _ -> false)
       (events m.ag))

(* A proxy's queues run no fill: the submission is refused when it is made.
   The fill is the support's bump, harmless if a hand-over called it. *)
let fill_refused () =
  with_machine @@ fun m ->
  let arg = B.create Rig.host 64 in
  raises_match Exn.invalid_arg @@ fun () ->
  Sub.make ~reads:0 ~writes:0 m.host
      [|
        {
          Sub.queue = "COMPUTE:0";
          after = [||];
          work =
            Sub.Fill
              {
                fill = Rig_support.bump;
                arg;
                ring_units = 0;
                segment_bytes = 0;
              };
        };
      |]

(* Waits between devices of one link travel with the hand-over, as the
   producer's id and value. *)
let waits_travel () =
  with_machine ~reaches:[ []; [] ] @@ fun m ->
  let d1 = m.devices.(0) and d2 = m.devices.(1) in
  let a = B.create d1 64 and b = B.create d1 64 and c = B.create d2 64 in
  let c' = B.create d2 64 in
  pause m.ag;
  let p = submit (copy_submission d1 ~src:a ~dst:b) in
  let q = submit ~waits:[| p |] (copy_submission d2 ~src:c ~dst:c') in
  resume m.ag;
  Rig.Point.wait q;
  let waits =
    require_match
      (List.find_map (function
        | Handover { device = 2; waits; _ } -> Some waits
        | _ -> None))
      (events m.ag)
  in
  equal (list (pair int int)) [ (1, Rig.Point.value p) ] waits

(* With the agent held, copies of 40 MiB within the machine: the third finds
   more than 64 MiB in flight and waits. *)
let room_later () =
  with_machine @@ fun m ->
  let n = in_flight_limit / 8 * 5 in
  let a = B.create m.host n and b = B.create m.host n in
  let s = copy_submission m.host ~src:a ~dst:b in
  pause m.ag;
  ignore (submit s);
  ignore (submit s);
  let finished, third = spawn (fun () -> submit s) in
  Thread.delay 0.3;
  let early = finished () in
  resume m.ag;
  let p = third () in
  Rig.Point.wait p;
  equal ~msg:"the third submit returned while 80 MiB were in flight" bool false
    early

let room_idle () =
  with_machine @@ fun m ->
  let n = 100 lsl 20 in
  let a = B.create m.host n and b = B.create m.host n in
  let p = submit (copy_submission m.host ~src:a ~dst:b) in
  Rig.Point.wait p

let work =
  group "work"
    [
      test "parts a proxy does not run are refused; words run on the host" never;
      test "a fill is refused" fill_refused;
      test "a wait on another device of the link reaches the agent" waits_travel;
      test "a submission waits while more than 64 MiB are in flight" room_later;
      test "an idle device takes a submission of any size" room_idle;
    ]

(* Code *)

let images () =
  with_machine ~reaches:[ [] ] @@ fun m ->
  let i = require_ok (Rig.Image.load m.host "run,step") in
  equal (option int) (Some 1) (Rig.Image.entry i "step");
  equal (option int) None (Rig.Image.entry i "walk");
  let why = require_error (Rig.Image.load m.host "bad code") in
  starts_with ~msg:"the reason starts with the host's name"
    ~affix:(Rig.name m.host) why;
  contains ~msg:"the agent's reason" ~sub:"the agent refuses it" why;
  let why = require_error (Rig.Image.load m.devices.(0) "run") in
  contains ~msg:"says it loads no code" ~sub:"loads no code" why

let code =
  group "code"
    [ test "the machine's host loads code, its devices do not" images ]

(* Failure and close *)

let job_fails () =
  with_machine ~reaches:[ [] ] @@ fun m ->
  let d = m.devices.(0) in
  let b = far_of_string d "before" in
  Link.fail m.ag.job "the machine went away";
  raises_match
    (function Rig.Lost (_, why) -> why = "the machine went away" | _ -> false)
    (fun () -> B.create d 16);
  equal lost_w (Some "the machine went away") (Rig.lost d);
  raises_match (function Rig.Lost _ -> true | _ -> false) (fun () -> read b)

let agent_dies () =
  with_machine @@ fun m ->
  Link.fail m.ag.job "the agent's process ended";
  raises_match
    (function
      | Rig.Lost (_, why) -> why = "the agent's process ended" | _ -> false)
    (fun () -> B.create m.host 16)

(* A hand-over waiting for its word when the job fails: the wait raises and the
   device stops at its last value. *)
let in_flight_fails () =
  with_machine @@ fun m ->
  let a = B.create m.host 64 and b = B.create m.host 64 in
  pause m.ag;
  let p = submit (copy_submission m.host ~src:a ~dst:b) in
  Link.fail m.ag.job "failed in flight";
  raises_match
    (function Rig.Lost (_, why) -> why = "failed in flight" | _ -> false)
    (fun () -> Rig.Point.wait p);
  until ~what:"the word at the last value handed over" (fun () ->
      Rig.signaled m.host = Rig.Point.value p)

(* Each counted call of a proxy made directly on the link, with a region and an
   image the agent made before the job failed. *)
let counted =
  [
    ("alloc", fun p _ _ -> ignore (Proxy.alloc p Device 16));
    ("map_peer", fun p r _ -> ignore (Proxy.map_peer p p r));
    ("image", fun p _ _ -> ignore (Proxy.image p "run"));
    ("entry", fun _ _ i -> ignore (Proxy.entry i "run"));
    ("sleep", fun p _ _ -> Proxy.sleep p ~seen:0 ~still_ms:10);
  ]

(* free and unload send nothing once the job failed; signaled reads the shadow
   as before. *)
let quiet_after_failure () =
  with_machine @@ fun m ->
  let p = m.raw in
  let r = require_some (Proxy.alloc p Device 16) in
  let i =
    match Proxy.image p "run" with
    | Ok (Rig_edge.Loaded i) -> i
    | Ok (Place _) -> fail "the host loads its own code"
    | Error why -> fail why
  in
  Link.fail m.ag.job "root";
  Proxy.free p r;
  Proxy.unload p i;
  equal int 0 (Proxy.signaled p)

let counted_after_failure (_, call) =
  with_machine @@ fun m ->
  let p = m.raw in
  let r = require_some (Proxy.alloc p Device 16) in
  let i =
    match Proxy.image p "run" with
    | Ok (Rig_edge.Loaded i) -> i
    | Ok (Place _) -> fail "the host loads its own code"
    | Error why -> fail why
  in
  Link.fail m.ag.job "root";
  raises_match
    (function Proxy.Fault "root" -> true | _ -> false)
    (fun () -> call p r i)

let close_device () =
  with_machine ~reaches:[ [] ] @@ fun m ->
  let d = m.devices.(0) in
  ignore (far_of_string d "closing");
  Rig.close d;
  equal lost_w (Some "closed") (Rig.lost d);
  equal (option string) None (Link.failure m.ag.job);
  equal string "the host goes on"
    (read (far_of_string m.host "the host goes on"))

(* A close waits for the work submitted before it. *)
let close_waits () =
  with_machine @@ fun m ->
  let a = far_of_string m.host "waited for" in
  let b = B.create m.host (B.length a) in
  pause m.ag;
  ignore (submit (copy_submission m.host ~src:a ~dst:b));
  let finished, closed = spawn (fun () -> Rig.close m.host) in
  Thread.delay 0.2;
  let early = finished () in
  resume m.ag;
  closed ();
  equal ~msg:"the close returned before the work it waits for" bool false early;
  equal lost_w (Some "closed") (Rig.lost m.host)

(* A peer that reads every frame and never closes: the job closes, and stays
   closing, from [Link.close] on. [closing ()] holds once its close came. *)
let silent_peer p =
  let closing = Atomic.make false in
  let b = Bytes.create 4096 in
  let rec drain () =
    let n = try Unix.read p b 0 9 with Unix.Unix_error _ -> 0 in
    if n = 9 then begin
      let len = Int64.to_int (Bytes.get_int64_le b 0) in
      if Char.code (Bytes.get b 8) = 10 then Atomic.set closing true;
      let rec skip k =
        if k > 0 then
          let r =
            try Unix.read p b 0 (min k 4096) with Unix.Unix_error _ -> 0
          in
          if r > 0 then skip (k - r)
      in
      skip len;
      drain ()
    end
  in
  let _, drained = spawn drain in
  ((fun () -> Atomic.get closing), drained)

(* The hand-over refuses a closing job's work; its reason names the close. *)
let closing_handover () =
  let job = Link.job () in
  let d, p = connected () in
  let closing, drained = silent_peer p in
  let far = Link.make job d ~name:"far" ~peer:(Wire.Agent 1) in
  let machine = fresh_machine () in
  let host =
    ok_or_fail
      (Rig.open_host
         (module Proxy)
         ~machine ~name:"CPU"
         (fun () ->
           Ok
             (Proxy.make far (account 0 ~reaches:[])
                (Rig_remote_abi.Host { machine; rail = no_rails }))))
  in
  let _, closed = spawn (fun () -> Link.close job) in
  Fun.protect
    ~finally:(fun () ->
      Link.fail job "the test ended";
      closed ();
      Unix.close p;
      drained ();
      Rig.close host)
    (fun () ->
      until ~what:"the close" closing;
      let words = host_buffer "\001\000\000\000" in
      let s =
        Sub.make ~reads:0 ~writes:0 host
          [|
            { Sub.queue = "COMPUTE:0"; after = [||]; work = Sub.Words words };
          |]
      in
      match submit s with
      | _ -> fail "a closing job took a hand-over"
      | exception Rig.Lost (_, why) ->
          contains ~msg:"the reason" ~sub:"clos" why)

(* [n] bytes of [p], fewer once its stream ended. *)
let read_n p n =
  let b = Bytes.create n in
  let rec go k =
    if k = n then k
    else
      match Unix.read p b k (n - k) with
      | 0 -> k
      | r -> go (k + r)
      | exception Unix.Unix_error _ -> k
  in
  Bytes.sub_string b 0 (go 0)

(* Answers the request [p] reads next, an allocation, that it succeeded. *)
let grant_alloc p =
  let h = read_n p 9 in
  if String.length h < 9 then fail "the link ended before its request";
  ignore (read_n p (Int64.to_int (String.get_int64_le h 0)));
  let answer = "\000\001" in
  let frame = Bytes.create (9 + String.length answer) in
  Bytes.set_int64_le frame 0 (Int64.of_int (String.length answer));
  Bytes.set frame 8 '\002';
  Bytes.blit_string answer 0 frame 9 (String.length answer);
  ignore (Unix.write p frame 0 (Bytes.length frame))

(* A hand-over sends its frame itself: to a peer that reads nothing, a copy from
   this process's memory larger than the sockets hold returns only once the peer
   took the frame. *)
let handover_sends () =
  let n = 64 lsl 20 in
  let job = Link.job () in
  let d, p = connected () in
  let far = Link.make job d ~name:"far" ~peer:(Wire.Agent 1) in
  let machine = fresh_machine () in
  let host =
    ok_or_fail
      (Rig.open_host
         (module Proxy)
         ~machine ~name:"CPU"
         (fun () ->
           Ok
             (Proxy.make far (account 0 ~reaches:[])
                (Rig_remote_abi.Host { machine; rail = no_rails }))))
  in
  Fun.protect
    ~finally:(fun () ->
      Link.fail job "the test ended";
      Unix.close p;
      Rig.close host)
    (fun () ->
      let _, region = spawn (fun () -> B.create host n) in
      grant_alloc p;
      let dst = region () in
      let src = B.create Rig.host n in
      let returned, submitted =
        spawn (fun () -> submit (copy_submission host ~src ~dst))
      in
      (match Unix.select [ p ] [] [] patience with
      | [], _, _ -> fail "the hand-over did not start"
      | _ -> ());
      Thread.delay 0.2;
      let early = returned () in
      let h = read_n p 9 in
      let payload = read_n p (Int64.to_int (String.get_int64_le h 0)) in
      ignore (submitted ());
      equal ~msg:"returned while its peer read nothing" bool false early;
      equal ~msg:"the frame's kind" int 3 (Char.code h.[8]);
      less ~msg:"the hand-over's bytes" int ~than:(String.length payload) n)

(* A second proxy of one account would take the first's place in the link's
   table, and the first's reports would never reach it. *)
let second_proxy () =
  with_job_link @@ fun far ->
  let a = account 1 ~reaches:[] in
  ignore (Proxy.make far a (Rig_remote_abi.Device { id = 1 }));
  raises_match Exn.invalid_arg (fun () ->
      Proxy.make far a (Rig_remote_abi.Device { id = 1 }))

let failures =
  group "failure"
    [
      test "a failed job loses the machine's devices with its root cause"
        job_fails;
      test "a job failed by the agent's end loses the host" agent_dies;
      test "a wait for a value handed over when the job fails raises Lost"
        in_flight_fails;
      cases
        ~name:(fun (n, _) ->
          n ^ " raises Fault with the root cause once the job failed")
        "counted" counted counted_after_failure;
      test "free, unload and signaled raise nothing once the job failed"
        quiet_after_failure;
      test "closing a device ends it alone" close_device;
      test "closing the host waits for its submitted work" close_waits;
      test "a hand-over on a closing job fails naming the close"
        closing_handover;
      test "a hand-over returns once its frame is sent (sampled)" handover_sends;
      test "make refuses a second proxy of an account on the link" second_proxy;
    ]

let () =
  Watchdog.start ();
  exit
    (run "rig_remote_proxy.proxy"
       [
         group ~timeout:60. "proxy"
           [ facts; memory; copies; work; code; failures ];
       ])
