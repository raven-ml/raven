(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The laws every GPU driver keeps, from rig_edge.mli's contract and rig.mli's
   promises, on each driver the machine has. A law reads what it needs of a
   device from its facts: on a device whose queues run no copy it draws no copy
   part, on one that maps no host memory no borrow. Work on a device's first
   queue comes from the driver's test support (Rig_gpu_support.Conformance). *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled

let strf = Printf.sprintf

module type Gpu = Rig_gpu_support.Conformance

type work = Rig_gpu_support.work = {
  part : Sub.part;
  block : Sub.Run.t -> Sub.block -> unit;
}

let patience_ns = 10_000_000_000

(* Bytes *)

let pattern n seed =
  String.init n (fun i -> Char.chr ((seed + (i * 7)) land 255))

(* Bytes print by their length and digest past a line. *)
let octets =
  let pp ppf s =
    let n = String.length s in
    if n <= 48 then Format.fprintf ppf "%S" s
    else
      Format.fprintf ppf "%d bytes, md5 %s" n (Digest.to_hex (Digest.string s))
  in
  Testable.make ~pp ~equal:String.equal

let put b s = B.copy ~src:(B.of_string s) ~dst:b

let get b =
  let n = B.length b in
  let h = B.create Rig.host n in
  B.copy ~src:b ~dst:h;
  let s = Bytes.create n in
  B.blit_to_bytes h 0 s 0 n;
  Bytes.to_string s

(* A buffer of [d]'s memory [memory] holding [s]. *)
let filled ?memory d s =
  let b = B.create ?memory d (String.length s) in
  put b s;
  b

(* [n] bytes of host memory that starts on a page, borrowed by [d], where [d]
   maps host memory. *)
let borrowed d n =
  let h = B.view (B.create Rig.host (Int.max n 65536)) ~first:0 ~length:n in
  B.borrow d h

(* [n] zeroed bytes of [d]'s Pinned memory, which every driver's host
   addresses, and a read of them at that address (rig.h's [rig_buffer_host]),
   which calls nothing of rig. The caller keeps the buffer reachable while it
   reads. *)
let watched d n =
  let b = filled ~memory:Pinned d (String.make n '\000') in
  let a = Rig_support.Reader.host b in
  (b, fun () -> Rig_gpu_support.Host.read a n)

(* Submitting *)

(* Submits [ws] on [d], using each buffer of [uses] with its access: its
   point. *)
let submit_point ?(waits = [||]) d ws ~uses =
  let buffers, access = Array.split (Array.of_list uses) in
  let s = Sub.make ~access d (Array.of_list (List.map (fun w -> w.part) ws)) in
  Rig.submit s ~run:(Rig_gpu_support.blocks s ws) ~buffers ~waits

(* As [submit_point]: the point's value. *)
let submit ?waits d ws ~uses = Rig.Point.value (submit_point ?waits d ws ~uses)

let copy_part ?(after = [||]) queue ~dst src =
  Rig_gpu_support.work { Sub.queue; after; work = Copy { src; dst } }

let with_after after w = { w with part = { w.part with after } }

(* Returns once [f ()] holds, failing after 10 s of the monotonic clock. *)
let await what f =
  let t0 = Rig.Profile.now () in
  while not (f ()) do
    if Rig.Profile.now () - t0 > patience_ns then failf "%s within 10 s" what;
    Domain.cpu_relax ()
  done

(* Queues *)

let kind_name : Rig.kind -> string = function
  | Words -> "Words"
  | Fill -> "Fill"
  | Copy -> "Copy"
  | Launch -> "Launch"

let queue =
  Testable.make
    ~pp:(fun ppf (q : Rig.queue) ->
      Format.fprintf ppf "%s [%s]" q.name
        (String.concat "; " (List.map kind_name q.runs)))
    ~equal:( = )

let first_queue d = (List.hd (Rig.queues d)).Rig.name

let running kind d =
  List.filter_map
    (fun (q : Rig.queue) -> if List.mem kind q.runs then Some q.name else None)
    (Rig.queues d)

let copy_queues = running Rig.Copy

(* The copy queues beside the first, whose copies run beside its work. *)
let beside d = List.filter (fun q -> q <> first_queue d) (copy_queues d)

(* A fill that fails: rig_support's countdown over a word holding 1. *)
let failing_fill queue =
  let arg = B.create Rig.host 8 in
  B.blit_from_string ("\001" ^ String.make 7 '\000') 0 arg 0 8;
  let work =
    Sub.Fill
      { fill = Rig_support.countdown; arg; ring_units = 0; segment_bytes = 0 }
  in
  { Sub.queue; after = [||]; work }

(* Launch work *)

(* The first queue of a device that runs launches, and the driver's launch
   binary loaded on the device ([G.launch_binary]), or [None] where the device
   runs no launch. *)
let launcher (module G : Gpu) =
  let load =
    Rig_gpu_support.loader (fun () ->
        match G.launch_binary () with
        | Some b -> b
        | None -> failf "%s runs launches, and has no launch binary" G.class_)
  in
  fun d ->
    match running Rig.Launch d with
    | q :: _ -> Some (q, load d)
    | [] -> None

(* As [launcher], skipping the law where the device runs no launch. *)
let launches_on launcher d =
  match launcher d with
  | Some l -> l
  | None -> skip ~reason:"the device runs no launch" ()

let refer at slot = { Sub.at; slot }
let u32 x = x land 0xffff_ffff

(* [kernel] of the launcher's binary on its queue, with [params] bytes of
   parameters, [block] storing its block. *)
let launch_work ?(after = [||]) (queue, image) kernel ~params refs block =
  let work = Sub.Launch { image; kernel; params; refs } in
  { part = { Sub.queue; after; work }; block }

(* [ids] over [groups] of [threads], [out] [offset] bytes into the slot
   [slot]. *)
let ids ?after l ~slot ~groups:(gx, gy, gz) ~threads:(tx, ty, tz) ~offset ~a ~b
    ~f =
  launch_work ?after l "ids" ~params:24 [| refer 0 slot |] (fun run k ->
      Sub.Run.groups run k gx gy gz;
      Sub.Run.threads run k tx ty tz;
      Sub.Run.int64 run k 0 offset;
      Sub.Run.int64 run k 8 a;
      Sub.Run.int32 run k 16 b;
      Sub.Run.float32 run k 20 f)

(* [twice] over [n] words, [offset] bytes into the slots [dst] and [src]. *)
let twice l ~dst ~src n ~offset ~c =
  launch_work l "twice" ~params:20 [| refer 0 dst; refer 8 src |] (fun run k ->
      Sub.Run.groups run k n 1 1;
      Sub.Run.threads run k 1 1 1;
      Sub.Run.int64 run k 0 offset;
      Sub.Run.int64 run k 8 offset;
      Sub.Run.int32 run k 16 c)

(* [n] 32-bit words, word [i] holding [f i]. *)
let le32s n f =
  let b = Bytes.create (4 * n) in
  for i = 0 to n - 1 do
    Bytes.set_int32_le b (4 * i) (Int32.of_int (f i))
  done;
  Bytes.to_string b

(* The 32-bit words of [s] from byte [at]. *)
let words s ~at n =
  Array.init n (fun i ->
      u32 (Int32.to_int (String.get_int32_le s (at + (4 * i)))))

(* Facts *)

let facts (module G : Gpu) () =
  G.with_ @@ fun t ->
  let f = G.D.facts t.g in
  not_equal string ~msg:"arch" "" f.arch;
  greater int ~msg:"budget" ~than:0 f.budget;
  equal (list queue) ~msg:"Rig.queues" f.queues (Rig.queues t.d);
  let names = List.map (fun (q : Rig.queue) -> q.name) f.queues in
  equal int ~msg:"distinct queue names" (List.length names)
    (List.length (List.sort_uniq compare names));
  is_some ~msg:"the word's host address" (G.D.locate f.word).host;
  equal int ~msg:"the word at open" 0 (G.D.signaled t.g);
  (* Whether memory [memory] has a host address; [None] where the driver makes
     none of it, as [Mapped] may be, which rig then allocates [Pinned]. *)
  let host memory =
    match G.D.alloc t.g memory 64 with
    | None -> None
    | Some r ->
        let h = Option.is_some (G.D.locate r).host in
        G.D.free t.g r;
        Some h
  in
  equal (option bool) ~msg:"Pinned memory's host address" (Some true)
    (host Pinned);
  (match host Mapped with
  | Some h -> equal bool ~msg:"Mapped memory's host address" true h
  | None -> ());
  equal bool ~msg:"Rig.reaches host" f.host_addresses
    (Rig.reaches Rig.host t.d);
  if f.host_addresses then
    equal (option bool) ~msg:"Device memory's host address" (Some true)
      (host Device)

(* Each kind a queue's runs omit, and a queue the device lacks, are refused by
   make, and no value is assigned. *)
let refusals (module G : Gpu) () =
  G.with_ @@ fun t ->
  (* A load may copy the code as a submission: values are counted after it. *)
  let bin, kernels = G.binary () in
  let image = require_ok ~pp:Format.pp_print_string (Rig.Image.load t.d bin) in
  let before = Rig.submitted t.d in
  let work : Rig.kind -> Sub.work = function
    | Words -> Words (B.create Rig.host 8)
    | Fill -> (failing_fill "").work
    | Copy -> Copy { src = B.create t.d 8; dst = B.create t.d 8 }
    | Launch ->
        Launch { image; kernel = List.hd kernels; params = 0; refs = [||] }
  in
  let refused queue kind =
    raises_match
      ~msg:(strf "%s on %s" (kind_name kind) queue)
      Exn.invalid_arg
      (fun () -> Sub.make t.d [| { queue; after = [||]; work = work kind } |])
  in
  List.iter
    (fun (q : Rig.queue) ->
      List.iter
        (fun k -> if not (List.mem k q.runs) then refused q.name k)
        [ Rig.Words; Fill; Copy; Launch ])
    (Rig.queues t.d);
  refused "NONE:0" Copy;
  equal int ~msg:"values assigned" before (Rig.submitted t.d)

(* Copies *)

let pp_memory ppf (m : B.memory) =
  Format.pp_print_string ppf
    (match m with Device -> "Device" | Pinned -> "Pinned" | Mapped -> "Mapped")

let memory = Gen.of_list ~pp:pp_memory [ B.Device; Pinned; Mapped ]

(* Sizes up to past 8 MiB, where a GPU's memory may take 2 MiB pages. *)
let trips =
  let open Gen in
  let bytes =
    of_list ~pp:Format.pp_print_int
      [ 1; 7; 4096; (2 lsl 20) + 7; (8 lsl 20) + 3 ]
  in
  let offset = of_list ~pp:Format.pp_print_int [ 0; 1; 4095 ] in
  let hop = int_range 0 3 in
  with_pp
    (fun ppf (ka, kb, n, (oa, ob), (h0, h1, h2)) ->
      Format.fprintf ppf "%a at %d, %a at %d, %d bytes, hops %d %d %d"
        pp_memory ka oa pp_memory kb ob n h0 h1 h2)
    (let+ ka = memory
     and+ kb = memory
     and+ n = bytes
     and+ offsets = pair offset offset
     and+ hops = triple hop hop hop in
     (ka, kb, n, offsets, hops))

(* host -> a -> b -> host, each hop on a copy queue the case draws, is the
   identity. *)
let copies (module G : Gpu) (ka, kb, n, (oa, ob), (h0, h1, h2)) =
  G.with_ @@ fun t ->
  let qs = Array.of_list (copy_queues t.d) in
  if qs = [||] then skip ~reason:"the device's queues run no copy" ();
  let q h = qs.(h mod Array.length qs) in
  let at memory o = B.view (B.create ~memory t.d (n + o)) ~first:o ~length:n in
  let data = pattern n (n + oa) in
  let src = filled ~memory:Pinned t.d data in
  let dst = filled ~memory:Pinned t.d (String.make n '\000') in
  let a = at ka oa and b = at kb ob in
  let v =
    submit t.d
      [
        copy_part (q h0) ~dst:a src;
        copy_part (q h1) ~after:[| 0 |] ~dst:b a;
        copy_part (q h2) ~after:[| 1 |] ~dst b;
      ]
      ~uses:
        [
          (src, B.Read);
          (a, B.Read_write);
          (b, B.Read_write);
          (dst, B.Read_write);
        ]
  in
  Rig.wait t.d v;
  equal octets data (get dst)

(* Readers *)

type agent = Host | Copy_part | Work
type place = Own of B.memory | Borrowed

let pp_agent ppf a =
  Format.pp_print_string ppf
    (match a with Host -> "the host" | Copy_part -> "a copy" | Work -> "work")

let pp_place ppf = function
  | Own m -> pp_memory ppf m
  | Borrowed -> Format.pp_print_string ppf "borrowed host memory"

let accesses =
  let open Gen in
  let agent = of_list ~pp:pp_agent [ Host; Copy_part; Work ] in
  with_pp
    (fun ppf (place, w, r, same) ->
      Format.fprintf ppf "%a written by %a, read by %a%s" pp_place place
        pp_agent w pp_agent r
        (if same then " in one submission" else ""))
    (let+ place =
       of_list ~pp:pp_place [ Own Device; Own Pinned; Own Mapped; Borrowed ]
     and+ w = agent
     and+ r = agent
     and+ same = bool in
     (place, w, r, same))

(* Whatever wrote memory last, the next reader reads that write, round after
   round over one memory. The host writes and reads through rig; a copy runs
   where a queue runs one, work otherwise. *)
let readers (module G : Gpu) (place, writer, reader, same) =
  G.with_ @@ fun t ->
  let n = 16384 in
  let mem =
    match place with
    | Own memory -> B.create ~memory t.d n
    | Borrowed -> (
        (* A device that maps no host memory has no such place. *)
        match borrowed t.d n with Some b -> b | None -> reject ())
  in
  (* [agent]'s part copying [src] into [dst], and the buffers it reads. *)
  let device agent ~dst ~src =
    match (agent, copy_queues t.d) with
    | Copy_part, q :: _ -> (copy_part q ~dst src, [ (src, B.Read) ])
    | _ ->
        let p, args = G.copy_words t ~dst ~src in
        (p, [ (src, B.Read); (args, B.Read) ])
  in
  for round = 1 to 4 do
    let msg = strf "round %d" round in
    let p = pattern n ((8 * round) + 1) in
    let src = filled ~memory:Pinned t.d p in
    let out = filled ~memory:Pinned t.d (String.make n '\000') in
    match (writer, reader) with
    | Host, Host ->
        put mem p;
        equal octets ~msg p (get mem)
    | Host, r ->
        put mem p;
        let part, reads = device r ~dst:out ~src:mem in
        Rig.wait t.d
          (submit t.d [ part ] ~uses:(reads @ [ (out, B.Read_write) ]));
        equal octets ~msg p (get out)
    | w, Host ->
        let part, reads = device w ~dst:mem ~src in
        Rig.wait t.d
          (submit t.d [ part ] ~uses:(reads @ [ (mem, B.Read_write) ]));
        equal octets ~msg p (get mem)
    | w, r ->
        let pw, rw = device w ~dst:mem ~src in
        let pr, rr = device r ~dst:out ~src:mem in
        let v =
          if same then
            submit t.d
              [ pw; with_after [| 0 |] pr ]
              ~uses:(rw @ rr @ [ (mem, B.Read_write); (out, B.Read_write) ])
          else begin
            ignore (submit t.d [ pw ] ~uses:(rw @ [ (mem, B.Read_write) ]));
            submit t.d [ pr ] ~uses:(rr @ [ (out, B.Read_write) ])
          end
        in
        Rig.wait t.d v;
        equal octets ~msg p (get out)
  done

(* Device order *)

(* A value's work starts once every earlier value's completed, on every
   queue. Work late by 50 ms then writes [mid], and a copy beside it at the
   next value reads [mid]; and the other way round, a copy after 50 ms of work
   writes [mid] and work at the next value reads it. *)
let device_order (module G : Gpu) () =
  G.with_ @@ fun t ->
  let q =
    match beside t.d with
    | q :: _ -> q
    | [] -> skip ~reason:"the device runs no copy beside its first queue" ()
  in
  let n = 4096 in
  let late = 50_000_000 in
  let data = pattern n 3 in
  let src = filled ~memory:Pinned t.d data in
  let fresh () = filled t.d (String.make n '\000') in
  let out () = filled ~memory:Pinned t.d (String.make n '\000') in
  let sp, sa = G.spin t ~ns:late in
  let mid = fresh () and o = out () in
  let w, wa = G.copy_words t ~dst:mid ~src in
  ignore
    (submit t.d [ sp; w ]
       ~uses:
         [ (src, B.Read); (sa, B.Read); (wa, B.Read); (mid, B.Read_write) ]);
  Rig.wait t.d
    (submit t.d [ copy_part q ~dst:o mid ] ~uses:[ (o, B.Read_write) ]);
  equal octets ~msg:"a copy after work" data (get o);
  let sp, sa = G.spin t ~ns:late in
  let mid = fresh () and o = out () in
  ignore
    (submit t.d
       [ sp; copy_part q ~after:[| 0 |] ~dst:mid src ]
       ~uses:[ (src, B.Read); (sa, B.Read); (mid, B.Read_write) ]);
  let r, ra = G.copy_words t ~dst:o ~src:mid in
  Rig.wait t.d (submit t.d [ r ] ~uses:[ (ra, B.Read); (o, B.Read_write) ]);
  equal octets ~msg:"work after a copy" data (get o)

(* Launches *)

(* A launch of [ids] over [groups] of [threads], into a view that starts
   [first] bytes into its buffer, from [offset] bytes into the view; then, in
   the same submission, a launch of [twice] reading those words through the
   same slot. *)
type geometry = {
  groups : int * int * int;
  threads : int * int * int;
  a : int;
  b : int;
  f : float;
  c : int;
  first : int;
  offset : int;
}

let geometry =
  let pp ppf g =
    let gx, gy, gz = g.groups and tx, ty, tz = g.threads in
    Format.fprintf ppf
      "{ groups = %dx%dx%d; threads = %dx%dx%d; a = %d; b = %d; f = %g; c = \
       %d; first = %d; offset = %d }"
      gx gy gz tx ty tz g.a g.b g.f g.c g.first g.offset
  in
  let open Gen in
  let flat n = map (fun x -> (x, 1, 1)) (int_range 1 n) in
  let* three = bool in
  let groups =
    if three then triple (int_range 1 4) (int_range 1 3) (int_range 1 3)
    else flat 64
  and threads =
    if three then triple (int_range 1 8) (int_range 1 4) (int_range 1 4)
    else flat 256
  in
  with_pp pp
    (let+ groups = groups
     and+ threads = threads
     and+ a = int_range 0 0xffff_ffff
     and+ b = int_range 0 0xffff
     and+ f = map float_of_int (int_range 0 ((1 lsl 24) - 1))
     and+ c = int_range 0 0xffff
     and+ first = of_list [ 0; 4; 256 ]
     and+ offset = of_list [ 0; 8; 4096 ] in
     { groups; threads; a; b; f; c; first; offset })

let launches_conform (module G : Gpu) =
  let launcher = launcher (module G) in
  fun g ->
    let gx, gy, gz = g.groups and tx, ty, tz = g.threads in
    cover "groups along y or z" (gy > 1 || gz > 1);
    cover "threads along y or z" (ty > 1 || tz > 1);
    cover "a view at an offset" (g.first > 0);
    cover "a ref at an offset" (g.offset > 0);
    G.with_ @@ fun t ->
    let l = launches_on launcher t.d in
    let n = gx * gy * gz * tx * ty * tz in
    let bytes = g.offset + (4 * n) in
    let base = filled t.d (String.make (g.first + bytes) '\000') in
    let out = B.view base ~first:g.first ~length:bytes in
    let dst = filled t.d (String.make bytes '\000') in
    let first =
      ids l ~slot:0 ~groups:g.groups ~threads:g.threads ~offset:g.offset ~a:g.a
        ~b:g.b ~f:g.f
    in
    let second = twice l ~dst:1 ~src:0 n ~offset:g.offset ~c:g.c in
    Rig.wait t.d
      (submit t.d [ first; second ]
         ~uses:[ (out, B.Read_write); (dst, B.Read_write) ]);
    let expected =
      Array.init n (fun k -> u32 (g.a + (g.b * k) + int_of_float g.f))
    in
    equal (array int) ~msg:"ids" expected (words (get out) ~at:g.offset n);
    equal (array int) ~msg:"twice"
      (Array.map (fun x -> u32 ((2 * x) + g.c)) expected)
      (words (get dst) ~at:g.offset n)

(* A launch whose grid or group has no size along an axis is refused before
   any value, and the device stays live. *)
let empty_launches (module G : Gpu) =
  let launcher = launcher (module G) in
  fun () ->
    G.with_ @@ fun t ->
    let l = launches_on launcher t.d in
    let out = filled t.d (String.make 64 '\000') in
    let before = Rig.submitted t.d in
    let refuses groups threads =
      let gx, gy, gz = groups and tx, ty, tz = threads in
      raises_match
        ~msg:(strf "%dx%dx%d groups of %dx%dx%d" gx gy gz tx ty tz)
        Exn.invalid_arg
        (fun () ->
          submit t.d
            [ ids l ~slot:0 ~groups ~threads ~offset:0 ~a:0 ~b:0 ~f:0. ]
            ~uses:[ (out, B.Read_write) ])
    in
    List.iter
      (fun (groups, threads) -> refuses groups threads)
      [
        ((0, 1, 1), (1, 1, 1));
        ((1, 0, 1), (1, 1, 1));
        ((1, 1, 0), (1, 1, 1));
        ((1, 1, 1), (0, 1, 1));
        ((1, 1, 1), (1, 0, 1));
        ((1, 1, 1), (1, 1, 0));
      ];
    equal int ~msg:"values assigned" before (Rig.submitted t.d);
    equal (option string) ~msg:"the device's loss" None (Rig.lost t.d);
    Rig.wait t.d
      (submit t.d
         [ ids l ~slot:0 ~groups:(1, 1, 1) ~threads:(16, 1, 1) ~offset:0 ~a:1
             ~b:1 ~f:0. ]
         ~uses:[ (out, B.Read_write) ]);
    equal (array int) ~msg:"a launch after them"
      (Array.init 16 (fun k -> 1 + k))
      (words (get out) ~at:0 16)

(* A value's work starts once every earlier value's completed, on every
   queue, launches included. A copy beside the launch queue after 50 ms of
   work writes [mid] and a launch at the next value reads it; and the other
   way round, a launch after 50 ms of work writes [mid] and a copy beside it
   at the next value reads it. *)
let launch_order (module G : Gpu) =
  let launcher = launcher (module G) in
  fun () ->
    G.with_ @@ fun t ->
    let ((first, _) as l) = launches_on launcher t.d in
    let q =
      match List.filter (fun q -> q <> first) (copy_queues t.d) with
      | q :: _ -> q
      | [] -> skip ~reason:"the device runs no copy beside its launches" ()
    in
    let n = 16384 in
    let late = 50_000_000 in
    let src = filled t.d (le32s n Fun.id) in
    let fresh () = filled t.d (String.make (4 * n) '\000') in
    let sp, sa = G.spin t ~ns:late in
    let mid = fresh () and dst = fresh () in
    ignore
      (submit t.d
         [ sp; copy_part q ~after:[| 0 |] ~dst:mid src ]
         ~uses:[ (src, B.Read); (sa, B.Read); (mid, B.Read_write) ]);
    Rig.wait t.d
      (submit t.d
         [ twice l ~dst:1 ~src:0 n ~offset:0 ~c:1 ]
         ~uses:[ (mid, B.Read); (dst, B.Read_write) ]);
    equal (array int) ~msg:"a launch after a copy"
      (Array.init n (fun i -> (2 * i) + 1))
      (words (get dst) ~at:0 n);
    let sp, sa = G.spin t ~ns:late in
    let mid = fresh () and o = fresh () in
    ignore
      (submit t.d
         [
           sp;
           ids ~after:[| 0 |] l ~slot:1 ~groups:(n / 64, 1, 1)
             ~threads:(64, 1, 1) ~offset:0 ~a:7 ~b:3 ~f:0.;
         ]
         ~uses:[ (sa, B.Read); (mid, B.Read_write) ]);
    Rig.wait t.d
      (submit t.d [ copy_part q ~dst:o mid ] ~uses:[ (o, B.Read_write) ]);
    equal (array int) ~msg:"a copy after a launch"
      (Array.init n (fun k -> 7 + (3 * k)))
      (words (get o) ~at:0 n)

(* Workspace *)

(* Launches through one workspace, each over its first [k] KiB: a part copies
   the launch's own bytes into it, work after it copies them out. Nothing waits
   between launches: the workspace's stamps order each after the one before. *)
let launches = Gen.list ~size:(Gen.int_range 1 12) (Gen.int_range 1 4)

let workspace (module G : Gpu) sizes =
  G.with_ @@ fun t ->
  let s = B.create t.d (4 * 1024) in
  let into ~dst ~src =
    match beside t.d with
    | q :: _ -> (copy_part q ~dst src, [])
    | [] ->
        let p, a = G.copy_words t ~dst ~src in
        (p, [ (a, B.Read) ])
  in
  let launch i k =
    let n = k * 1024 in
    let data = pattern n i in
    let src = filled ~memory:Pinned t.d data in
    let out = filled ~memory:Pinned t.d (String.make n '\000') in
    let ws = B.view s ~first:0 ~length:n in
    let p0, a0 = into ~dst:ws ~src in
    let p1, a1 = G.copy_words t ~dst:out ~src:ws in
    ignore
      (submit t.d
         [ p0; with_after [| 0 |] p1 ]
         ~uses:
           (((src, B.Read) :: (a1, B.Read) :: a0)
           @ [ (s, B.Read_write); (out, B.Read_write) ]));
    (out, data)
  in
  List.iteri
    (fun i (out, data) ->
      B.wait out Read;
      equal octets ~msg:(strf "launch %d" i) data (get out))
    (List.mapi launch sizes)

(* Images *)

let images (module G : Gpu) () =
  G.with_ @@ fun t ->
  let bin, kernels = G.binary () in
  let i = require_ok ~pp:Format.pp_print_string (Rig.Image.load t.d bin) in
  List.iter (fun k -> is_some ~msg:k (Rig.Image.entry i k)) kernels;
  is_none ~msg:"a function the image lacks" (Rig.Image.entry i "absent");
  is_error ~msg:"bytes that are no binary" (Rig.Image.load t.d "no binary")

(* Timeline *)

let idle_close (module G : Gpu) () =
  let t = G.open_ () in
  G.wait t (G.submit t [||]);
  G.wait t (G.submit t [||]);
  G.close t;
  equal int ~msg:"the word" 2 (Rig.signaled t.d)

let stale_seen (module G : Gpu) () =
  G.with_ @@ fun t ->
  let v = G.submit t [||] in
  G.wait t v;
  G.D.sleep t.g ~seen:(v - 1) ~still_ms:600_000

(* Sleeps over work that runs 1 s raise no fault, return with the word
   unmoved, and block: 10 of 50 ms take little of the CPU. *)
let long_work (module G : Gpu) () =
  G.with_ @@ fun t ->
  let p, args = G.spin t ~ns:1_000_000_000 in
  let v = submit t.d [ p ] ~uses:[ (args, B.Read) ] in
  let cpu = Sys.time () in
  for _ = 1 to 10 do
    G.D.sleep t.g ~seen:(v - 1) ~still_ms:50;
    equal int ~msg:"the word while the work runs" (v - 1) (G.D.signaled t.g)
  done;
  less float_exact ~msg:"CPU seconds of 10 sleeps of 50 ms" ~than:0.25
    (Sys.time () -. cpu);
  Rig.wait t.d v;
  G.D.sleep t.g ~seen:(v - 1) ~still_ms:600_000

(* One domain sleeps on the word while another submits: each sleep returns as
   the word moves. *)
let sleep_aside (module G : Gpu) () =
  G.with_ @@ fun t ->
  let n = 1000 in
  let sleeper =
    Domain.spawn (fun () ->
        while G.D.signaled t.g < n do
          G.D.sleep t.g ~seen:(G.D.signaled t.g) ~still_ms:1000
        done)
  in
  for _ = 1 to n do
    ignore (G.submit t [||])
  done;
  Rig.wait t.d n;
  Domain.join sleeper;
  equal int ~msg:"the word" n (G.D.signaled t.g)

(* Handed-over work runs with no further call: the host sees its bytes in its
   own memory, calling nothing of rig after the submit. *)
let runs_alone (module G : Gpu) () =
  G.with_ @@ fun t ->
  let n = 4096 in
  let dst, read = watched t.d n in
  let data = pattern n 5 in
  let src = filled ~memory:Pinned t.d data in
  let p, args = G.copy_words t ~dst ~src in
  let v =
    submit t.d [ p ]
      ~uses:[ (src, B.Read); (args, B.Read); (dst, B.Read_write) ]
  in
  await "the work ran" (fun () -> read () = data);
  Rig.wait t.d v;
  ignore (Sys.opaque_identity dst)

(* The driver commits on its own: after a submit, submits alone bring the word
   to its value. *)
let own_commits (module G : Gpu) () =
  G.with_ @@ fun t ->
  let v = G.submit t [||] in
  await "the word reached the value" (fun () ->
      G.D.signaled t.g >= v
      ||
      (ignore (G.submit t [||]);
       false))

(* Loss *)

(* A fill that fails, behind work still running, loses the device with a
   reason every later use raises, and the word still reaches the failed
   value. *)
let loss (module G : Gpu) () =
  let t = G.open_ () in
  match running Rig.Fill t.d with
  | [] ->
      G.close t;
      skip ~reason:"the device's queues run no fill" ()
  | queue :: _ ->
      let p, args = G.spin t ~ns:100_000_000 in
      let v = submit t.d [ p ] ~uses:[ (args, B.Read) ] in
      let why =
        match G.submit t [| failing_fill queue |] with
        | _ -> fail "a failed fill lost nothing"
        | exception Rig.Lost (_, why) -> why
      in
      not_equal string ~msg:"the reason" "" why;
      equal (option string) ~msg:"lost" (Some why) (Rig.lost t.d);
      raises_match ~msg:"a later use"
        (function Rig.Lost (_, w) -> w = why | _ -> false)
        (fun () -> B.create t.d 64);
      G.close t;
      await "the word reached the failed value" (fun () ->
          Rig.signaled t.d >= v + 1)

(* Waits *)

(* A submission that waits for a producer's point runs no work before it: the
   bytes its work writes stay as they were while the producer's word, read
   after them, is below the point. [go] makes the producer reach the point; a
   domain of its own calls it 20 ms after it starts to watch, as the consumer
   may wait in its submit. *)
let holds (module G : Gpu) (t : G.t) point go =
  let pd = Rig.Point.device point and vp = Rig.Point.value point in
  let n = 4096 in
  let dst, read = watched t.G.d n in
  let old = String.make n '\000' in
  let data = pattern n 9 in
  let src = filled ~memory:Pinned t.d data in
  let p, args = G.copy_words t ~dst ~src in
  let finished = Atomic.make false in
  let early = Atomic.make None in
  let sampler =
    Domain.spawn (fun () ->
        let t0 = Rig.Profile.now () in
        let started = ref false in
        while not (Atomic.get finished) do
          let x = read () in
          if Rig.signaled pd < vp && x <> old then
            Atomic.set early (Some "the work ran before the producer's value");
          if (not !started) && Rig.Profile.now () - t0 > 20_000_000 then begin
            started := true;
            go ()
          end;
          Domain.cpu_relax ()
        done;
        if not !started then go ())
  in
  let consumed () =
    Rig.wait t.d
      (submit t.d [ p ]
         ~uses:[ (src, B.Read); (args, B.Read); (dst, B.Read_write) ]
         ~waits:[| point |])
  in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set finished true;
      Domain.join sampler)
    consumed;
  equal (option string) ~msg:"early" None (Atomic.get early);
  equal octets ~msg:"the work's bytes" data (read ());
  ignore (Sys.opaque_identity dst)

let empty_point d =
  Rig.submit (Sub.make d [||]) ~run:(Sub.Run.make ()) ~buffers:[||] ~waits:[||]

(* The producer, a Polled device, whose work runs when the test runs it. *)
let polled_waits (module G : Gpu) () =
  G.with_ @@ fun t ->
  let pd, pp = P.open_ "POLLED:producer" in
  Fun.protect ~finally:(fun () -> Rig.close pd) @@ fun () ->
  holds (module G) t (empty_point pd) (fun () -> ignore (P.run pp))

(* The second device of [t]'s driver, in rig under a name of its own, for
   [f]; skips where the machine has none. *)
let with_second (module G : Gpu) f =
  match G.second () with
  | None -> skip ~reason:(strf "the machine has one %s device" G.class_) ()
  | Some (Error why) -> failf "a second %s device: %s" G.class_ why
  | Some (Ok g) -> f g

(* The producer, another device of the driver, whose work runs 50 ms. *)
let device_waits (module G : Gpu) () =
  G.with_ @@ fun t ->
  with_second (module G) @@ fun g ->
  let pd =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ (module G.D) ~name:(G.class_ ^ ":producer") (fun () -> Ok g))
  in
  Fun.protect ~finally:(fun () -> Rig.close pd) @@ fun () ->
  let tp = { G.d = pd; g } in
  let sp, sa = G.spin tp ~ns:50_000_000 in
  let point = submit_point pd [ sp ] ~uses:[ (sa, B.Read) ] in
  holds (module G) t point ignore

(* Peers *)

(* [peer g g'] is whether [map_peer g g'] maps Device memory of [g'], and a
   mapping of [g']'s Pinned memory, where [map_peer] gives one, has its host
   address. *)
let peers (module G : Gpu) () =
  G.with_ @@ fun t ->
  with_second (module G) @@ fun g' ->
  Fun.protect ~finally:(fun () -> G.D.stop g' ~fault:None) @@ fun () ->
  let d = require_some (G.D.alloc g' Device 64) in
  let view = G.D.map_peer t.g g' d in
  equal bool ~msg:"peer is map_peer's answer for Device memory"
    (G.D.peer t.g g') (Option.is_some view);
  Option.iter (G.D.free t.g) view;
  let h = require_some (G.D.alloc g' Pinned 64) in
  (match G.D.map_peer t.g g' h with
  | Some v ->
      equal (option int) ~msg:"the host address of Pinned memory"
        (G.D.locate h).host (G.D.locate v).host;
      G.D.free t.g v
  | None -> ());
  G.D.free g' h;
  G.D.free g' d

(* Order *)

(* Submissions of parts, each a copy of words between three buffers: on a copy
   queue, or by work on the first queue, maybe after a spin there, so that work
   run out of order shows in the buffers. A part runs after the earlier parts
   its [after] lists. The reference copies in program order. The system
   submits back to back on one device all programs share, and the word it
   reads never moves back, never passes the last value, and stays there. *)
module Order (G : Gpu) = struct
  type part = {
    work : bool; (* by work on the first queue, else a copy part *)
    queue : int; (* a copy part's copy queue, among them *)
    late : bool;
    src : int;
    so : int;
    dst : int;
    do_ : int;
    n : int;
    after : int list;
  }

  let buffers = 3
  let size = 65536
  let late_ns = 50_000

  let pp_part ppf p =
    Format.fprintf ppf "{%s%s %d@%d -> %d@%d n=%d after=[%s]}"
      (if p.work then "work" else strf "copy %d" p.queue)
      (if p.late then " late" else "")
      p.src p.so p.dst p.do_ p.n
      (String.concat ";" (List.map string_of_int p.after))

  let pp ppf subs =
    Format.pp_print_list ~pp_sep:Format.pp_print_space
      (fun ppf ps ->
        Format.fprintf ppf "[%a]"
          (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_part)
          ps)
      ppf subs

  let initial b = pattern size (b * 61)
  let shared = fixture ~teardown:G.close G.open_

  (* The device's first queue and its copy queues. *)
  type queues = { first : string; copies : string array }

  let queues () =
    let t = shared () in
    { first = first_queue t.d; copies = Array.of_list (copy_queues t.d) }

  let queue_of q p =
    if p.work || q.copies = [||] then q.first
    else q.copies.(p.queue mod Array.length q.copies)

  (* The reference *)

  type t = { bufs : Bytes.t array; q : queues; mutable last : string option }

  let make () =
    {
      bufs = Array.init buffers (fun b -> Bytes.of_string (initial b));
      q = queues ();
      last = None;
    }

  (* Part [i] runs before part [j], [i < j]: both on one queue, or [j] runs
     after a part that runs after [i]. *)
  let rec before m ps i j =
    let pj = List.nth ps j in
    queue_of m.q (List.nth ps i) = queue_of m.q pj
    || List.exists (fun k -> k = i || (k > i && before m ps i k)) pj.after

  let overlap (b, o) (b', o') n n' = b = b' && o < o' + n' && o' < o + n

  let conflict p q =
    overlap (p.dst, p.do_) (q.dst, q.do_) p.n q.n
    || overlap (p.src, p.so) (q.dst, q.do_) p.n q.n
    || overlap (p.dst, p.do_) (q.src, q.so) p.n q.n

  (* The result is one: every two parts that conflict are ordered, and no part
     copies over its own source. *)
  let determined_one m ps =
    List.for_all
      (fun p -> not (overlap (p.src, p.so) (p.dst, p.do_) p.n p.n))
      ps
    &&
    let n = List.length ps in
    List.for_all
      (fun j ->
        List.for_all
          (fun i ->
            before m ps i j || not (conflict (List.nth ps i) (List.nth ps j)))
          (List.init j Fun.id))
      (List.init n Fun.id)

  let determined m subs = List.for_all (determined_one m) subs

  let run_one m ps =
    let on = queue_of m.q in
    let last = match List.rev ps with p :: _ -> Some (on p) | [] -> None in
    (* Labels of several queues, on a device that has them. *)
    if Array.exists (fun c -> c <> m.q.first) m.q.copies then begin
      cover "a switch of queue" (last <> None && m.last <> None && m.last <> last);
      cover "two queues in one submission"
        (List.length (List.sort_uniq compare (List.map on ps)) = 2);
      cover "an after across queues"
        (List.exists
           (fun p -> List.exists (fun k -> on (List.nth ps k) <> on p) p.after)
           ps);
      cover "a part on another queue after late work"
        (List.exists
           (fun p ->
             on p <> m.q.first
             && List.exists
                  (fun k ->
                    let q = List.nth ps k in
                    q.late && on q = m.q.first)
                  p.after)
           ps)
    end;
    cover "a submission on the queue that released the last"
      (last <> None && m.last = last);
    cover "late work" (List.exists (fun p -> p.late && on p = m.q.first) ps);
    List.iter
      (fun p -> Bytes.blit m.bufs.(p.src) p.so m.bufs.(p.dst) p.do_ p.n)
      ps;
    if last <> None then m.last <- last

  let run m subs =
    cover "several values in flight" (List.length subs > 1);
    List.iter (run_one m) subs

  (* The system *)

  type sys = { t : G.t; q : queues; regions : B.t array }

  let start () =
    let t = shared () in
    {
      t;
      q = queues ();
      regions = Array.init buffers (fun b -> filled t.d (initial b));
    }

  (* A part's system parts, in order, and the buffers they read: a late part
     is a spin, then its copy, on the first queue. *)
  let pieces s p =
    let dst = B.view s.regions.(p.dst) ~first:p.do_ ~length:p.n
    and src = B.view s.regions.(p.src) ~first:p.so ~length:p.n in
    if (not p.work) && s.q.copies <> [||] then
      ([ copy_part (queue_of s.q p) ~dst src ], [])
    else
      let c, a = G.copy_words s.t ~dst ~src in
      if not p.late then ([ c ], [ (a, B.Read) ])
      else
        let sp, a' = G.spin s.t ~ns:late_ns in
        ([ sp; c ], [ (a', B.Read); (a, B.Read) ])

  (* Submits [ps]: each part's first system part takes its [after], each index
     the last system part of the part it names. *)
  let submit_one s ps =
    let ends = Array.make (List.length ps) 0 in
    let parts = ref [] and uses = ref [] and at = ref 0 in
    List.iteri
      (fun i p ->
        let sys, args = pieces s p in
        let after = Array.of_list (List.map (fun j -> ends.(j)) p.after) in
        List.iteri
          (fun k part ->
            parts := (if k = 0 then with_after after part else part) :: !parts)
          sys;
        at := !at + List.length sys;
        ends.(i) <- !at - 1;
        uses := args @ !uses)
      ps;
    ignore (submit s.t.d (List.rev !parts) ~uses:!uses)

  (* Submits [subs] back to back, then reads the word until it holds the last
     value: each read is at least the one before and at most the last. *)
  let run_sys s subs =
    let first = Rig.submitted s.t.d + 1 in
    List.iter (submit_one s) subs;
    let last = Rig.submitted s.t.d in
    let rec watch seen =
      let w = Rig.signaled s.t.d in
      at_least int ~msg:"the word" ~than:seen w;
      at_most int ~msg:"the word" ~than:last w;
      if w < last then watch w
    in
    watch (first - 1);
    Rig_gpu_support.still ~msg:"the word" int last
      (fun () -> Rig.signaled s.t.d)
      ~ms:1

  let invariant m s =
    Array.iteri
      (fun b r ->
        equal octets ~msg:(strf "buffer %d" b) (Bytes.to_string m.bufs.(b))
          (get r))
      s.regions

  (* Parts copy whole words at word offsets, as work does. *)
  let parts =
    let open Gen in
    let word x = x land lnot 3 in
    let part =
      let+ work = bool
      and+ queue = int_range 0 3
      and+ late = frequency [ (2, constant false); (1, constant true) ]
      and+ src = int_range 0 (buffers - 1)
      and+ dst = int_range 0 (buffers - 1)
      and+ n = one_of [ int_range 4 64; int_range 4 16384 ]
      and+ so = int_range 0 (size - 16384)
      and+ do_ = int_range 0 (size - 16384)
      and+ after = list ~size:(int_range 0 2) (int_range 0 2) in
      {
        work;
        queue;
        late = work && late;
        src;
        so = word so;
        dst;
        do_ = word do_;
        n = word n;
        after;
      }
    in
    let submission =
      let+ ps = list ~size:(int_range 0 3) part in
      List.mapi
        (fun i p ->
          {
            p with
            after =
              List.sort_uniq compare (List.filter (fun k -> k < i) p.after);
          })
        ps
    in
    with_pp pp (list ~size:(int_range 1 4) submission)

  let order = abstract "o" ~invariant

  let commands =
    [
      command "start" (Gen.unit @-> makes order) make start;
      command "submit" ~pre:determined
        (order ^-> parts @-> returns unit)
        run run_sys;
    ]
end

(* The drivers *)

let laws (module G : Gpu) =
  let module O = Order (G) in
  let g = (module G : Gpu) in
  group ~timeout:120. G.class_
    [
      test "the facts are well formed, and Rig.queues states them" (facts g);
      test "make refuses work a queue does not run and a queue the device lacks"
        (refusals g);
      prop ~count:30 "copies through any two memories on copy queues are the identity"
        trips (copies g);
      prop ~count:30 "whatever wrote memory last, the next reader reads that write"
        accesses (readers g);
      test "a value's work starts once every earlier value's completed"
        (device_order g);
      prop ~count:30
        "a launch writes what its grid and parameters say, and a launch after \
         it reads that"
        geometry (launches_conform g);
      test "a launch of no size along an axis is refused" (empty_launches g);
      test "a value's launches start once every earlier value's work completed"
        (launch_order g);
      prop ~count:30 "a launch reads what its submission wrote into a workspace"
        launches (workspace g);
      test "an image names its kernels, and bytes no binary are an error"
        (images g);
      test "a close of an idle device leaves the word at the last value"
        (idle_close g);
      test "sleep returns at once when the word differs from seen"
        (stale_seen g);
      test "sleeps over long work raise no fault, and block" (long_work g);
      test "sleep returns as the word moves while another domain submits"
        (sleep_aside g);
      test "handed-over work runs with no further call" (runs_alone g);
      test "the driver commits on its own while work is submitted"
        (own_commits g);
      test
        "a failed fill loses the device with a reason, and the word reaches \
         its value"
        (loss g);
      test "work runs after a Polled device's point it waits for"
        (polled_waits g);
      test "work runs after another device's point it waits for"
        (device_waits g);
      test "peer is whether map_peer maps Device memory" (peers g);
      stateful ~timeout:300. ~count:100 ~steps:20
        "values complete in order and the word never moves backwards \
         (sampled)"
        O.commands;
    ]

let gpus : (module Gpu) list =
  [
    (module Rig_metal_support);
    (module Rig_cuda_support);
    (module Rig_nv_support);
    (module Rig_amd_support);
  ]

let () =
  if List.exists (fun (module G : Gpu) -> G.present ()) gpus then
    Rig_gpu_lock.hold ();
  exit (run "rig_conformance" (List.map laws gpus))
