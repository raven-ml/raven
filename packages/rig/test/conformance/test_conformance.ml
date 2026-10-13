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
  part : int -> Sub.part;
  buffers : B.t array;
  fixed : (B.t * Rig.access) list;
  keep : Rig_gpu_support.kept;
  init : Sub.t -> Sub.Run.t -> int -> unit;
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

(* [n] zeroed bytes of [d]'s Pinned memory, which every driver's host addresses,
   and a read of them at that address (rig.h's [rig_buffer_host]), which calls
   nothing of rig. The caller keeps the buffer reachable while it reads. *)
let watched d n =
  let b = filled ~memory:Pinned d (String.make n '\000') in
  let a = Rig_support.Reader.host b in
  (b, fun () -> Rig_gpu_support.Host.read a n)

(* Submitting *)

(* Submits [ws] on [d], with each buffer of [uses] fixed with its access: its
   point. The parts follow each other by that memory. *)
let submit ?waits d ws ~uses =
  let s, run, buffers = Rig_gpu_support.prepare ~fixed:uses d ws in
  Rig.submit ?waits s ~run ~buffers

let copy_part ~dst src =
  Rig_gpu_support.work Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))

(* Returns once [f ()] holds, failing after 10 s of the monotonic clock. *)
let await what f =
  let t0 = Rig.Profile.now () in
  while not (f ()) do
    if Rig.Profile.now () - t0 > patience_ns then failf "%s within 10 s" what;
    Domain.cpu_relax ()
  done

(* Queues *)

let kind_name : Rig_edge.kind -> string = function
  | Words -> "Words"
  | Fill -> "Fill"
  | Copy -> "Copy"
  | Launch -> "Launch"

(* The queues of a device's facts. *)
let first_queue (qs : Rig_edge.queue list) = (List.hd qs).name

let running kind qs =
  List.filter_map
    (fun (q : Rig_edge.queue) ->
      if List.mem kind q.runs then Some q.name else None)
    qs

(* Whether the device's copy engine is a queue other than [q], so that its
   copies run beside [q]'s work. *)
let beside (f : _ Rig_edge.facts) q =
  match f.copy_engine with
  | Some i -> (List.nth f.queues i).name <> q
  | None -> false

(* A fill that fails: rig_support's countdown over a word holding 1. *)
let failing_fill d =
  let arg = B.create Rig.host (1 lsl 16) in
  B.blit_from_string ("\001" ^ String.make 7 '\000') 0 arg 0 8;
  let fixed = [ (Option.get (B.borrow d arg), Rig.Read_write) ] in
  let part = Sub.fill Rig_support.countdown ~params:8 in
  let init s run i =
    Sub.Run.int64 run (Sub.block s i) 0 (Rig_support.Reader.host arg)
  in
  { part = (fun _ -> part); buffers = [||]; fixed; keep = Kept (); init }

(* Launch work *)

(* The driver's launch binary loaded on the device ([G.launch_binary]), or
   [None] where no queue of the device runs launches. *)
let launcher (module G : Gpu) =
  let load =
    Rig_gpu_support.loader (fun () ->
        match G.launch_binary () with
        | Some b -> b
        | None -> failf "%s runs launches, and has no launch binary" G.class_)
  in
  fun (t : G.t) ->
    match running Rig_edge.Launch (G.D.facts t.g).queues with
    | _ :: _ -> Some (load t.d)
    | [] -> None

(* As [launcher], skipping the law where the device runs no launch. *)
let launches_on launcher t =
  match launcher t with
  | Some l -> l
  | None -> skip ~reason:"the device runs no launch" ()

let u32 x = x land 0xffff_ffff

(* [kernel] of the launcher's binary, with [params] bytes of parameters,
   [block] storing its block, each [(b, a, at)] of [uses] a buffer at a
   parameter byte with its access. *)
let launch_work image kernel ~params uses block =
  let w = Rig_gpu_support.launching image kernel ~params uses (fun _ _ -> ()) in
  { w with init = (fun s run i -> block run (Sub.block s i)) }

(* [ids] over [groups] of [threads], writing [offset] bytes into [out]. *)
let ids l ~out ~groups:(gx, gy, gz) ~threads:(tx, ty, tz) ~offset ~a ~b ~f =
  launch_work l "ids" ~params:24
    [| (out, Rig.Read_write, 0) |]
    (fun run k ->
      Sub.Run.groups run k gx gy gz;
      Sub.Run.threads run k tx ty tz;
      Sub.Run.int64 run k 0 offset;
      Sub.Run.int64 run k 8 a;
      Sub.Run.int32 run k 16 b;
      Sub.Run.float32 run k 20 f)

(* [twice] over [n] words, [offset] bytes into [dst] and [src]. *)
let twice l ~dst ~src n ~offset ~c =
  launch_work l "twice" ~params:24
    [| (dst, Rig.Read_write, 0); (src, Rig.Read, 8) |]
    (fun run k ->
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
  let names = List.map (fun (q : Rig_edge.queue) -> q.name) f.queues in
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
  equal bool ~msg:"Rig.reaches host" f.host_addresses (Rig.reaches Rig.host t.d);
  if f.host_addresses then
    equal (option bool) ~msg:"Device memory's host address" (Some true)
      (host Device)

(* Each kind no queue of the device runs is refused by make, and no value is
   assigned. *)
let refusals (module G : Gpu) () =
  G.with_ @@ fun t ->
  (* A load may copy the code as a submission: values are counted after it. *)
  let bin, kernels = G.binary () in
  let image = require_ok ~pp:Format.pp_print_string (Rig.Image.load t.d bin) in
  let before = Rig.submitted t.d in
  let part : Rig_edge.kind -> Sub.part = function
    | Words -> Sub.words (B.create Rig.host 8)
    | Fill -> (failing_fill t.d).part 0
    | Copy ->
        Sub.(copy ~src:(Buffer (B.create t.d 8)) ~dst:(Buffer (B.create t.d 8)))
    | Launch -> Sub.launch image (List.hd kernels) ~params:0
  in
  let qs = (G.D.facts t.g).queues in
  List.iter
    (fun k ->
      if running k qs = [] then
        raises_match ~msg:(kind_name k) Exn.invalid_arg (fun () ->
            Sub.make t.d [| part k |]))
    [ Rig_edge.Words; Fill; Copy; Launch ];
  equal int ~msg:"values assigned" before (Rig.submitted t.d)

(* Copies *)

let pp_memory ppf (m : B.kind) =
  Format.pp_print_string ppf
    (match m with
    | Device -> "Device"
    | Pinned -> "Pinned"
    | Mapped -> "Mapped")

let memory = Gen.of_list ~pp:pp_memory [ B.Device; Pinned; Mapped ]

(* Sizes up to past 8 MiB, where a GPU's memory may take 2 MiB pages. *)
let trips =
  let open Gen in
  let bytes =
    of_list ~pp:Format.pp_print_int
      [ 1; 7; 4096; (2 lsl 20) + 7; (8 lsl 20) + 3 ]
  in
  let offset = of_list ~pp:Format.pp_print_int [ 0; 1; 4095 ] in
  with_pp
    (fun ppf (ka, kb, n, (oa, ob)) ->
      Format.fprintf ppf "%a at %d, %a at %d, %d bytes" pp_memory ka oa
        pp_memory kb ob n)
    (let+ ka = memory
     and+ kb = memory
     and+ n = bytes
     and+ offsets = pair offset offset in
     (ka, kb, n, offsets))

(* host -> a -> b -> host, three copies of one submission that follow each
   other by their memory, is the identity. *)
let copies (module G : Gpu) (ka, kb, n, (oa, ob)) =
  G.with_ @@ fun t ->
  if (G.D.facts t.g).copy_engine = None then
    skip ~reason:"the device has no copy engine" ();
  let at memory o = B.view (B.create ~memory t.d (n + o)) ~first:o ~length:n in
  let data = pattern n (n + oa) in
  let src = filled ~memory:Pinned t.d data in
  let dst = filled ~memory:Pinned t.d (String.make n '\000') in
  let a = at ka oa and b = at kb ob in
  let v =
    submit t.d
      [
        copy_part ~dst:a src;
        copy_part ~dst:b a;
        copy_part ~dst b;
      ]
      ~uses:
        [
          (src, Rig.Read);
          (a, Rig.Read_write);
          (b, Rig.Read_write);
          (dst, Rig.Read_write);
        ]
  in
  Rig.Point.wait v;
  equal octets data (get dst)

(* A profile records a copy per copy part, from the device of the allocation its
   bytes leave to the one they reach, with its bytes and the times of its
   submission's span where the device times its work. A borrow of host memory
   names the host. *)
let profiled_copies (module G : Gpu) () =
  G.with_ @@ fun t ->
  if (G.D.facts t.g).copy_engine = None then
    skip ~reason:"the device has no copy engine" ();
  let n = 4096 in
  let src, from =
    match borrowed t.d n with
    | Some b -> (b, Rig.host)
    | None -> (filled ~memory:Pinned t.d (pattern n 1), t.d)
  in
  let a = B.create t.d n and b = B.create ~memory:Pinned t.d n in
  let (), events =
    Rig.Profile.take (fun () ->
        Rig.Point.wait
          (submit t.d
             [ copy_part ~dst:a src; copy_part ~dst:b a ]
             ~uses:[ (src, Rig.Read); (a, Rig.Read_write); (b, Rig.Read_write) ]))
  in
  let copies =
    List.filter_map
      (function
        | Rig.Profile.Copy c -> Some (c.src, c.dst, c.bytes, c.start, c.stop)
        | _ -> None)
      events
  in
  equal
    (list (triple string string int))
    [ (Rig.name from, Rig.name t.d, n); (Rig.name t.d, Rig.name t.d, n) ]
    (List.map (fun (s, d, n, _, _) -> (Rig.name s, Rig.name d, n)) copies);
  let span =
    List.find_map
      (function
        | Rig.Profile.Span s when Rig.equal s.device t.d ->
            Some (s.start, s.stop)
        | _ -> None)
      events
  in
  List.iter
    (fun (_, _, _, start, stop) ->
      match span with
      | Some times ->
          equal (pair int int) ~msg:"the span's times" times (start, stop)
      | None -> at_most int ~msg:"untimed" ~than:stop start)
    copies

(* A chain of copies among three buffers: each copy's source, destination,
   offsets and length. Lengths are 0 or within 16 bytes of a power of two up to
   4 MiB, so that a chain crosses whatever sizes a driver changes engine at. *)
type link = { src : int; so : int; dst : int; do_ : int; n : int }

let pp_link ppf l =
  Format.fprintf ppf "%d@%d -> %d@%d, %d bytes" l.src l.so l.dst l.do_ l.n

let chain =
  let open Gen in
  let length =
    frequency
      [
        (1, constant 0);
        ( 8,
          let+ k = int_range 0 22 and+ j = int_range (-16) 16 in
          Int.max 0 ((1 lsl k) + j) );
      ]
  in
  let link =
    let+ src = int_range 0 2
    and+ step = int_range 1 2
    and+ so = int_range 0 15
    and+ do_ = int_range 0 15
    and+ n = length in
    { src; so; dst = (src + step) mod 3; do_; n }
  in
  let weighted =
    frequency
      [ (3, constant B.Device); (1, constant B.Pinned); (1, constant B.Mapped) ]
  in
  with_pp
    (fun ppf (kinds, links) ->
      Format.fprintf ppf "@[<v>%a@,%a@]"
        (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_memory)
        kinds
        (Format.pp_print_list ~pp_sep:Format.pp_print_cut pp_link)
        links)
    (pair (list ~size:(constant 3) weighted) (list ~size:(int_range 1 6) link))

(* Copies of one submission keep every byte: the buffers end as the copies
   applied in turn to the host's model of them say. *)
let chains (module G : Gpu) (kinds, links) =
  let sizes = [ 1 lsl 18; 2 lsl 20 ] in
  List.iter
    (fun s ->
      cover
        (strf "lengths below and from %d KiB in one chain" (s / 1024))
        (List.exists (fun l -> l.n < s) links
        && List.exists (fun l -> l.n >= s) links))
    sizes;
  let last = List.length links - 1 in
  cover "an empty copy between two others"
    (List.exists Fun.id
       (List.mapi (fun i l -> i > 0 && i < last && l.n = 0) links));
  G.with_ @@ fun t ->
  if (G.D.facts t.g).copy_engine = None then
    skip ~reason:"the device has no copy engine" ();
  let capacity k =
    List.fold_left
      (fun m l ->
        let m = if l.src = k then Int.max m (l.so + l.n) else m in
        if l.dst = k then Int.max m (l.do_ + l.n) else m)
      1 links
  in
  let model =
    Array.init 3 (fun k -> Bytes.of_string (pattern (capacity k) (31 * k)))
  in
  let buffers =
    Array.of_list
      (List.mapi
         (fun k memory -> filled ~memory t.d (Bytes.to_string model.(k)))
         kinds)
  in
  let parts =
    List.map
      (fun l ->
        Bytes.blit model.(l.src) l.so model.(l.dst) l.do_ l.n;
        let view k o = B.view buffers.(k) ~first:o ~length:l.n in
        copy_part ~dst:(view l.dst l.do_) (view l.src l.so))
      links
  in
  Rig.Point.wait (submit t.d parts ~uses:[]);
  Array.iteri
    (fun k b ->
      equal octets ~msg:(strf "buffer %d" k)
        (Bytes.to_string model.(k))
        (get b))
    buffers

(* Readers *)

type agent = Host | Copy_part | Work
type place = Own of B.kind | Borrowed

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
        match borrowed t.d n with
        | Some b -> b
        | None -> reject ())
  in
  (* [agent]'s part copying [src] into [dst], and the buffers it reads. *)
  let device agent ~dst ~src =
    match (agent, (G.D.facts t.g).copy_engine) with
    | Copy_part, Some _ -> (copy_part ~dst src, [ (src, Rig.Read) ])
    | _ ->
        let p = G.copy_words t ~dst ~src in
        (p, [ (src, Rig.Read) ])
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
        Rig.Point.wait
          (submit t.d [ part ] ~uses:(reads @ [ (out, Rig.Read_write) ]));
        equal octets ~msg p (get out)
    | w, Host ->
        let part, reads = device w ~dst:mem ~src in
        Rig.Point.wait
          (submit t.d [ part ] ~uses:(reads @ [ (mem, Rig.Read_write) ]));
        equal octets ~msg p (get mem)
    | w, r ->
        let pw, rw = device w ~dst:mem ~src in
        let pr, rr = device r ~dst:out ~src:mem in
        let v =
          if same then
            submit t.d
              [ pw; pr ]
              ~uses:(rw @ rr @ [ (mem, Rig.Read_write); (out, Rig.Read_write) ])
          else begin
            ignore (submit t.d [ pw ] ~uses:(rw @ [ (mem, Rig.Read_write) ]));
            submit t.d [ pr ] ~uses:(rr @ [ (out, Rig.Read_write) ])
          end
        in
        Rig.Point.wait v;
        equal octets ~msg p (get out)
  done

(* Blits *)

let blit_cases =
  let open Gen in
  let bytes = of_list ~pp:Format.pp_print_int [ 4; 4100 ] in
  with_pp
    (fun ppf (m, n) -> Format.fprintf ppf "%a, %d bytes" pp_memory m n)
    (let+ m = memory and+ n = bytes in
     (m, n))

(* Blits and a fill of memory beside work on it that starts 50 ms late: each
   follows the work before it, and the work after it reads what it wrote. *)
let blits (module G : Gpu) (memory, n) =
  G.with_ @@ fun t ->
  let late = 50_000_000 in
  let mem = B.create ~memory t.d n in
  let read_late o =
    submit t.d
      [ G.spin t ~ns:late; G.copy_words t ~dst:o ~src:mem ]
      ~uses:[ (mem, Rig.Read); (o, Rig.Read_write) ]
  in
  let write_late s =
    let src = filled ~memory:Pinned t.d s in
    submit t.d
      [ G.spin t ~ns:late; G.copy_words t ~dst:mem ~src ]
      ~uses:[ (src, Rig.Read); (mem, Rig.Read_write) ]
  in
  let out () = filled ~memory:Pinned t.d (String.make n '\000') in
  let x = pattern n 1 and y = pattern n 2 and z = pattern n 3 in
  B.blit_from_string x 0 mem 0 n;
  let before = out () in
  ignore (read_late before);
  B.blit_from_string y 0 mem 0 n;
  let after = out () in
  ignore (read_late after);
  ignore (write_late z);
  let got = Bytes.create n in
  B.blit_to_bytes mem 0 got 0 n;
  ignore (write_late x);
  B.fill mem 'f';
  let filled_out = out () in
  Rig.Point.wait (read_late filled_out);
  equal octets ~msg:"work before a blit reads what came before it" x
    (get before);
  equal octets ~msg:"work after a blit reads it" y (get after);
  equal octets ~msg:"a blit after work reads what it wrote" z
    (Bytes.to_string got);
  equal octets ~msg:"work after a fill reads it" (String.make n 'f')
    (get filled_out)

(* Device order *)

(* A value's work starts once every earlier value's completed, on every queue.
   Work late by 50 ms then writes [mid], and a copy beside it at the next value
   reads [mid]; and the other way round, a copy after 50 ms of work writes [mid]
   and work at the next value reads it. *)
let device_order (module G : Gpu) () =
  G.with_ @@ fun t ->
  let f = G.D.facts t.g in
  if not (beside f (first_queue f.queues)) then
    skip ~reason:"the device runs no copy beside its first queue" ();
  let n = 4096 in
  let late = 50_000_000 in
  let data = pattern n 3 in
  let src = filled ~memory:Pinned t.d data in
  let fresh () = filled t.d (String.make n '\000') in
  let out () = filled ~memory:Pinned t.d (String.make n '\000') in
  let sp = G.spin t ~ns:late in
  let mid = fresh () and o = out () in
  let w = G.copy_words t ~dst:mid ~src in
  ignore (submit t.d [ sp; w ] ~uses:[ (src, Rig.Read); (mid, Rig.Read_write) ]);
  Rig.Point.wait
    (submit t.d [ copy_part ~dst:o mid ] ~uses:[ (o, Rig.Read_write) ]);
  equal octets ~msg:"a copy after work" data (get o);
  let sp = G.spin t ~ns:late in
  let mid = fresh () and o = out () in
  ignore
    (submit t.d
       [ sp; copy_part ~dst:mid src ]
       ~uses:[ (src, Rig.Read); (mid, Rig.Read_write) ]);
  let r = G.copy_words t ~dst:o ~src:mid in
  Rig.Point.wait (submit t.d [ r ] ~uses:[ (o, Rig.Read_write) ]);
  equal octets ~msg:"work after a copy" data (get o)

(* Profiles *)

(* A profile taken around a submission records one span of its work on the
   device, which lies within the profile, up to the error its driver states
   (G.clock_error), and lasts at least as long as the work: [late] of work on
   the first queue, alone, and before or after a copy beside it; and a copy
   alone. *)
let spans (module G : Gpu) () =
  let opened = Rig.Profile.now () in
  G.with_ @@ fun t ->
  let f = G.D.facts t.g in
  let late = 20_000_000 and n = 4096 in
  let src = filled ~memory:Pinned t.d (pattern n 9) in
  let dst = filled ~memory:Pinned t.d (String.make n '\000') in
  let uses = [ (src, Rig.Read); (dst, Rig.Read_write) ] in
  let work () = G.spin t ~ns:late and copy () = copy_part ~dst src in
  let shapes =
    ("work", late, fun () -> [ work () ])
    ::
    (if beside f (first_queue f.queues) then
       [
         ("work, then a copy", late, fun () -> [ work (); copy () ]);
         ("a copy, then work", late, fun () -> [ copy (); work () ]);
         ("a copy", 0, fun () -> [ copy () ]);
       ]
     else [])
  in
  let span (what, lasts, ws) =
    let ws = ws () in
    let t0 = Rig.Profile.now () in
    let (), events =
      Rig.Profile.take (fun () -> Rig.Point.wait (submit t.d ws ~uses))
    in
    let t1 = Rig.Profile.now () in
    let spans =
      List.filter_map
        (function
          | Rig.Profile.Span { device; start; stop; _ } when device == t.d ->
              Some (start, stop)
          | _ -> None)
        events
    in
    match spans with
    | [ (start, stop) ] ->
        let error = G.clock_error ~since:(t1 - opened) in
        at_least int ~msg:(what ^ ": the span's start") ~than:(t0 - error)
          start;
        at_most int ~msg:(what ^ ": the span's stop") ~than:(t1 + error) stop;
        at_least int ~msg:(what ^ ": the span's length") ~than:lasts
          (stop - start)
    | spans ->
        equal int ~msg:(what ^ ": the device's spans") 1 (List.length spans)
  in
  List.iter span shapes

(* How long a device idles before its next submission's span: past every
   driver's age for a new measure of its clock, a second. *)
let idle_s = 1.2

(* A submission's span, profiled alone, lies within the host's window from
   before the submit to after the wait, on the host clock, up to the error
   its driver states: before and after the device idles past the age at
   which it measures its clock again, and right after it did. *)
let spans_after_idle (module G : Gpu) () =
  let opened = Rig.Profile.now () in
  G.with_ @@ fun t ->
  let n = 64 in
  let src = filled ~memory:Pinned t.d (pattern n 6) in
  let dst = filled ~memory:Pinned t.d (String.make n '\000') in
  let uses = [ (src, Rig.Read); (dst, Rig.Read_write) ] in
  let s, run, buffers =
    Rig_gpu_support.prepare ~fixed:uses t.d [ G.copy_words t ~dst ~src ]
  in
  let within what =
    let t0 = Rig.Profile.now () in
    let (), events =
      Rig.Profile.take (fun () -> Rig.Point.wait (Rig.submit s ~run ~buffers))
    in
    let t1 = Rig.Profile.now () in
    match
      List.filter_map
        (function
          | Rig.Profile.Span { device; start; stop; _ } when device == t.d ->
              Some (start, stop)
          | _ -> None)
        events
    with
    | [ (start, stop) ] ->
        let error = G.clock_error ~since:(t1 - opened) in
        at_least int ~msg:(what ^ ": the span's start") ~than:(t0 - error)
          start;
        at_most int ~msg:(what ^ ": the span's stop") ~than:(t1 + error) stop;
        at_most int ~msg:(what ^ ": the span's start") ~than:stop start
    | spans ->
        equal int ~msg:(what ^ ": the device's spans") 1 (List.length spans)
  in
  within "at open";
  Unix.sleepf idle_s;
  within "after idling";
  within "after the measure"

(* A profile around submissions, waited for once: [crossing] of them, then,
   after the device idled past every driver's age for a new measure of its
   clock, [many] more. Where a copy runs beside the first queue they are, in
   turn, work, a copy, a copy then work, and work then a copy, so that values
   run on either queue or on both. It records one span of each, and each span
   ends before the next starts, across the measure, since a value's work
   starts once every earlier value's has completed. [many] exceeds every
   driver's ring of stamps, so later submits convert the first values' stamps
   before the measure, and settles the rest after it. *)
let crossing = 64
let many = 33_000

let spans_across (module G : Gpu) () =
  G.with_ @@ fun t ->
  let f = G.D.facts t.g in
  let n = 64 in
  let src = filled ~memory:Pinned t.d (pattern n 4) in
  let dst = filled ~memory:Pinned t.d (String.make n '\000') in
  let uses = [ (src, Rig.Read); (dst, Rig.Read_write) ] in
  let prepare ws = Rig_gpu_support.prepare ~fixed:uses t.d ws in
  let work () = G.copy_words t ~dst ~src and copy () = copy_part ~dst src in
  let subs =
    if beside f (first_queue f.queues) then
      [|
        prepare [ work () ];
        prepare [ copy () ];
        prepare [ copy (); work () ];
        prepare [ work (); copy () ];
      |]
    else [| prepare [ work () ] |]
  in
  let submit i =
    let s, run, buffers = subs.(i mod Array.length subs) in
    Rig.submit s ~run ~buffers
  in
  let total = crossing + many in
  let (), events =
    Rig.Profile.take (fun () ->
        for i = 0 to crossing - 1 do
          ignore (submit i)
        done;
        Unix.sleepf idle_s;
        for i = crossing to total - 2 do
          ignore (submit i)
        done;
        Rig.Point.wait (submit (total - 1)))
  in
  let spans =
    List.filter_map
      (function
        | Rig.Profile.Span { device; start; stop; _ } when device == t.d ->
            Some (start, stop)
        | _ -> None)
      events
  in
  equal int ~msg:"the device's spans" total (List.length spans);
  let rec follow i = function
    | (_, stop) :: ((start, _) :: _ as rest) ->
        at_most int ~msg:(strf "span %d's stop, at most the next start" i)
          ~than:start stop;
        follow (i + 1) rest
    | _ -> ()
  in
  follow 0 spans

(* Launches *)

(* A launch of [ids] over [groups] of [threads], into a view that starts [first]
   bytes into its buffer, from [offset] bytes into the view; then, in the same
   submission, a launch of [twice] reading those words through the same slot. *)
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
    let l = launches_on launcher t in
    let n = gx * gy * gz * tx * ty * tz in
    let bytes = g.offset + (4 * n) in
    let base = filled t.d (String.make (g.first + bytes) '\000') in
    let out = B.view base ~first:g.first ~length:bytes in
    let dst = filled t.d (String.make bytes '\000') in
    let first =
      ids l ~out ~groups:g.groups ~threads:g.threads ~offset:g.offset ~a:g.a
        ~b:g.b ~f:g.f
    in
    let second = twice l ~dst ~src:out n ~offset:g.offset ~c:g.c in
    Rig.Point.wait
      (submit t.d [ first; second ]
         ~uses:[ (out, Rig.Read_write); (dst, Rig.Read_write) ]);
    let expected =
      Array.init n (fun k -> u32 (g.a + (g.b * k) + int_of_float g.f))
    in
    equal (array int) ~msg:"ids" expected (words (get out) ~at:g.offset n);
    equal (array int) ~msg:"twice"
      (Array.map (fun x -> u32 ((2 * x) + g.c)) expected)
      (words (get dst) ~at:g.offset n)

(* A launch whose grid or group has no size along an axis is refused before any
   value, and the device stays live. *)
let empty_launches (module G : Gpu) =
  let launcher = launcher (module G) in
  fun () ->
    G.with_ @@ fun t ->
    let l = launches_on launcher t in
    let out = filled t.d (String.make 64 '\000') in
    let before = Rig.submitted t.d in
    let refuses groups threads =
      let gx, gy, gz = groups and tx, ty, tz = threads in
      raises_match ~msg:(strf "%dx%dx%d groups of %dx%dx%d" gx gy gz tx ty tz)
        Exn.invalid_arg (fun () ->
          submit t.d
            [ ids l ~out ~groups ~threads ~offset:0 ~a:0 ~b:0 ~f:0. ]
            ~uses:[ (out, Rig.Read_write) ])
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
    Rig.Point.wait
      (submit t.d
         [
           ids l ~out ~groups:(1, 1, 1) ~threads:(16, 1, 1) ~offset:0 ~a:1 ~b:1
             ~f:0.;
         ]
         ~uses:[ (out, Rig.Read_write) ]);
    equal (array int) ~msg:"a launch after them"
      (Array.init 16 (fun k -> 1 + k))
      (words (get out) ~at:0 16)

(* Past this many parts the law below takes no longer to state. *)
let most_parts_cap = 1024

(* An idle device runs [min (most_parts d) most_parts_cap] launches in one
   submission, each with 4096 bytes of parameters, the most a launch gives:
   launch [i] stores [i + 1] into word [i]. *)
let largest_launches (module G : Gpu) =
  let launcher = launcher (module G) in
  fun () ->
    G.with_ @@ fun t ->
    let l = launches_on launcher t in
    let n = Int.min (Rig.most_parts t.d) most_parts_cap in
    let out = filled t.d (String.make (4 * n) '\000') in
    let launch i =
      launch_work l "ids" ~params:4096
        [| (out, Rig.Read_write, 0) |]
        (fun run k ->
          Sub.Run.groups run k 1 1 1;
          Sub.Run.threads run k 1 1 1;
          Sub.Run.int64 run k 0 (4 * i);
          Sub.Run.int64 run k 8 (i + 1);
          Sub.Run.int32 run k 16 0;
          Sub.Run.float32 run k 20 0.)
    in
    Rig.Point.wait
      (submit t.d (List.init n launch) ~uses:[ (out, Rig.Read_write) ]);
    equal (array int) ~msg:(strf "%d launches' words" n)
      (Array.init n (fun i -> i + 1))
      (words (get out) ~at:0 n)

(* What the image states of [ids], whose parameters end at byte 24, holds: a
   launch that gives fewer bytes is refused at make, and so is a ref past them.
   A group of the most threads the image states runs, and one more is refused
   by the setter. No value is assigned to a refusal. *)
let entry_facts (module G : Gpu) =
  let launcher = launcher (module G) in
  fun () ->
    G.with_ @@ fun t ->
    let l = launches_on launcher t in
    equal (option int) ~msg:"the parameters of ids" (Some 24)
      (Rig.Image.params l "ids");
    let most =
      match Rig.Image.threads l "ids" with
      | Some n -> n
      | None -> failf "%s states no threads of ids" G.class_
    in
    let before = Rig.submitted t.d in
    let make ~params refs = Sub.make t.d [| Sub.launch ~refs l "ids" ~params |] in
    raises ~msg:"20 bytes of parameters"
      (Invalid_argument
         "Rig.Submission.make: ids takes 24 bytes of parameters; part 0 gives 20")
      (fun () -> make ~params:20 [||]);
    raises ~msg:"a ref at 24"
      (Invalid_argument
         "Rig.Submission.make: part 0's ref at 24 lies past the 24 bytes of \
          parameters ids takes")
      (fun () -> make ~params:32 [| Sub.write 0 ~at:24 |]);
    let s = make ~params:24 [| Sub.write 0 ~at:0 |] in
    raises ~msg:"one thread more"
      (Invalid_argument
         (strf
            "Rig.Submission.Run.threads: ids runs at most %d threads per \
             group; the block sets %dx1x1"
            most (most + 1)))
      (fun () -> Sub.Run.threads (Sub.Run.make ()) (Sub.block s 0) (most + 1) 1 1);
    equal int ~msg:"values assigned" before (Rig.submitted t.d);
    let out = filled t.d (String.make (4 * most) '\000') in
    Rig.Point.wait
      (submit t.d
         [
           ids l ~out ~groups:(1, 1, 1) ~threads:(most, 1, 1) ~offset:0 ~a:0
             ~b:1 ~f:0.;
         ]
         ~uses:[ (out, Rig.Read_write) ]);
    equal (array int) ~msg:"a group of the most threads"
      (Array.init most Fun.id)
      (words (get out) ~at:0 most)

(* A value's work starts once every earlier value's completed, on every queue,
   launches included. A copy beside the launch queue after 50 ms of work writes
   [mid] and a launch at the next value reads it; and the other way round, a
   launch after 50 ms of work writes [mid] and a copy beside it at the next
   value reads it. *)
let launch_order (module G : Gpu) =
  let launcher = launcher (module G) in
  fun () ->
    G.with_ @@ fun t ->
    let l = launches_on launcher t in
    let f = G.D.facts t.g in
    let first = List.hd (running Rig_edge.Launch f.queues) in
    if not (beside f first) then
      skip ~reason:"the device runs no copy beside its launches" ();
    let n = 16384 in
    let late = 50_000_000 in
    let src = filled t.d (le32s n Fun.id) in
    let fresh () = filled t.d (String.make (4 * n) '\000') in
    let sp = G.spin t ~ns:late in
    let mid = fresh () and dst = fresh () in
    ignore
      (submit t.d
         [ sp; copy_part ~dst:mid src ]
         ~uses:[ (src, Rig.Read); (mid, Rig.Read_write) ]);
    Rig.Point.wait
      (submit t.d
         [ twice l ~dst ~src:mid n ~offset:0 ~c:1 ]
         ~uses:[ (mid, Rig.Read); (dst, Rig.Read_write) ]);
    equal (array int) ~msg:"a launch after a copy"
      (Array.init n (fun i -> (2 * i) + 1))
      (words (get dst) ~at:0 n);
    let sp = G.spin t ~ns:late in
    let mid = fresh () and o = fresh () in
    ignore
      (submit t.d
         [
           sp;
           ids l ~out:mid
             ~groups:(n / 64, 1, 1)
             ~threads:(64, 1, 1) ~offset:0 ~a:7 ~b:3 ~f:0.;
         ]
         ~uses:[ (mid, Rig.Read_write) ]);
    Rig.Point.wait
      (submit t.d [ copy_part ~dst:o mid ] ~uses:[ (o, Rig.Read_write) ]);
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
  let f = G.D.facts t.g in
  let into ~dst ~src =
    if beside f (first_queue f.queues) then copy_part ~dst src
    else G.copy_words t ~dst ~src
  in
  let launch i k =
    let n = k * 1024 in
    let data = pattern n i in
    let src = filled ~memory:Pinned t.d data in
    let out = filled ~memory:Pinned t.d (String.make n '\000') in
    let ws = B.view s ~first:0 ~length:n in
    let p0 = into ~dst:ws ~src in
    let p1 = G.copy_words t ~dst:out ~src:ws in
    ignore
      (submit t.d
         [ p0; p1 ]
         ~uses:[ (src, Rig.Read); (s, Rig.Read_write); (out, Rig.Read_write) ]);
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
  List.iter (fun k -> equal ~msg:k bool true (Rig.Image.mem i k)) kernels;
  equal ~msg:"a function the image lacks" bool false (Rig.Image.mem i "absent");
  is_error ~msg:"bytes that are no binary" (Rig.Image.load t.d "no binary")

(* Timeline *)

let idle_close (module G : Gpu) () =
  let t = G.open_ () in
  Rig.Point.wait (G.submit t [||]);
  Rig.Point.wait (G.submit t [||]);
  G.close t;
  equal int ~msg:"the word" 2 (Rig.signaled t.d)

(* Opens and closes of a device keep a fixed record per open: rig's line,
   records and reason, and the driver's word and the state reading it needs,
   which live for the process, beside what the vendor's library keeps. A warm
   run of as many opens fills the caches and tables the opens grow to, and
   reclamation returns each closed device's memory. *)
let opens = 32

(* What an open keeps of the C heap past its close, per open: rig's records,
   the driver's state, and the vendor's library's own, which neither rig nor
   the driver can give back. AMD's state holds its rings' marks and timed
   values, about 300 KiB, and Metal's a ring of 1,024 command buffer slots,
   40 KiB; Metal's library keeps about 30 bytes of CoreAnalytics messages,
   and caches that grow by about 10 KB over the first hundreds of opens. *)
let kept_c = 384 * 1024

(* What an open keeps of the OCaml heap: rig's device record and the
   driver's handle. *)
let kept_ml = 2048

let settle () =
  Gc.full_major ();
  Rig_support.reclaim ()

let open_close (module G : Gpu) () =
  let cycle () =
    let t = G.open_ () in
    ignore (B.create t.d 1 : B.t);
    G.close t
  in
  let heap () =
    match Rig_support.heap_bytes () with
    | Some n -> n
    | None -> skip ~reason:"the system counts no C heap" ()
  in
  let live () = (Gc.stat ()).live_words * (Sys.word_size / 8) in
  for _ = 1 to opens do
    cycle ()
  done;
  settle ();
  let c = heap () and ml = live () in
  for _ = 1 to opens do
    cycle ()
  done;
  settle ();
  at_most int ~msg:"the C heap's growth over the opens"
    ~than:(opens * kept_c) (heap () - c);
  at_most int ~msg:"the OCaml heap's growth over the opens"
    ~than:(opens * kept_ml) (live () - ml)

let stale_seen (module G : Gpu) () =
  G.with_ @@ fun t ->
  let p = G.submit t [||] in
  Rig.Point.wait p;
  G.D.sleep t.g ~seen:(Rig.Point.value p - 1) ~still_ms:600_000

(* Sleeps over work that runs 1 s raise no fault, return with the word unmoved,
   and block: 10 of 50 ms take little of the CPU. *)
let long_work (module G : Gpu) () =
  G.with_ @@ fun t ->
  let p = G.spin t ~ns:1_000_000_000 in
  let done_ = submit t.d [ p ] ~uses:[] in
  let v = Rig.Point.value done_ in
  let cpu = Sys.time () in
  for _ = 1 to 10 do
    G.D.sleep t.g ~seen:(v - 1) ~still_ms:50;
    equal int ~msg:"the word while the work runs" (v - 1) (G.D.signaled t.g)
  done;
  less float_exact ~msg:"CPU seconds of 10 sleeps of 50 ms" ~than:0.25
    (Sys.time () -. cpu);
  Rig.Point.wait done_;
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
  for _ = 1 to n - 1 do
    ignore (G.submit t [||])
  done;
  Rig.Point.wait (G.submit t [||]);
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
  let p = G.copy_words t ~dst ~src in
  let v = submit t.d [ p ] ~uses:[ (src, Rig.Read); (dst, Rig.Read_write) ] in
  await "the work ran" (fun () -> read () = data);
  Rig.Point.wait v;
  ignore (Sys.opaque_identity dst)

(* The driver commits on its own: after a submit, submits alone bring the word
   to its value. *)
let own_commits (module G : Gpu) () =
  G.with_ @@ fun t ->
  let v = Rig.Point.value (G.submit t [||]) in
  await "the word reached the value" (fun () ->
      G.D.signaled t.g >= v
      ||
      (ignore (G.submit t [||]);
       false))

(* Loss *)

(* A fill that fails, behind work still running, loses the device with a reason
   every later use raises, and the word still reaches the failed value. *)
let loss (module G : Gpu) () =
  let t = G.open_ () in
  match running Rig_edge.Fill (G.D.facts t.g).queues with
  | [] ->
      G.close t;
      skip ~reason:"the device's queues run no fill" ()
  | _ :: _ ->
      let p = G.spin t ~ns:100_000_000 in
      let v = Rig.Point.value (submit t.d [ p ] ~uses:[]) in
      let why =
        match G.submit_work t [ failing_fill t.d ] with
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

(* The hang bound the law's devices declare, at most. *)
let hang_cap_ms = 2_000

(* Work that keeps the word still past the device's hang bound loses the device
   with the reason rig states. The GPU opens through its driver wrapped to
   declare a bound of 2 s at most, where its facts bound hangs, and the work
   would end 2 s past it: the word reaches its value once the driver's stop
   ended the work or it ended, and the GPU opens again. AMD skips: a stop whose
   preemption of the hung work fails makes amdgpu reset the GPU, which other
   processes share. *)
let hung (module G : Gpu) () =
  if G.class_ = "AMD" then
    skip ~reason:"a failed preemption of hung work resets the GPU" ();
  let module Capped = struct
    include G.D

    let facts h =
      let f = G.D.facts h in
      { f with hang_ms = Option.map (Int.min hang_cap_ms) f.hang_ms }
  end in
  let t = G.open_as (module Capped) in
  match (G.D.facts t.g).hang_ms with
  | None ->
      G.close t;
      skip ~reason:"nothing bounds the device's hangs" ()
  | Some _ ->
      let ns = 2 * hang_cap_ms * 1_000_000 in
      let p = submit t.d [ G.spin t ~ns ] ~uses:[] in
      let why = strf "no progress for %d ms" hang_cap_ms in
      let hung = function Rig.Lost (_, w) -> w = why | _ -> false in
      raises_match ~msg:"the wait" hung (fun () -> Rig.Point.wait p);
      equal (option string) ~msg:"the loss" (Some why) (Rig.lost t.d);
      await "the word reached the hung value" (fun () ->
          Rig.signaled t.d >= Rig.Point.value p);
      G.close t;
      G.with_ @@ fun t -> Rig.Point.wait (G.submit t [||])

(* Waits *)

(* A submission that waits for a producer's point runs no work before it: the
   bytes its work writes stay as they were while the producer's word, read after
   them, is below the point. [go] makes the producer reach the point; a domain
   of its own calls it 20 ms after it starts to watch, as the consumer may wait
   in its submit. *)
let holds (module G : Gpu) (t : G.t) point go =
  let pd = Rig.Point.device point and vp = Rig.Point.value point in
  let n = 4096 in
  let dst, read = watched t.G.d n in
  let old = String.make n '\000' in
  let data = pattern n 9 in
  let src = filled ~memory:Pinned t.d data in
  let p = G.copy_words t ~dst ~src in
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
    Rig.Point.wait
      (submit t.d [ p ]
         ~uses:[ (src, Rig.Read); (dst, Rig.Read_write) ]
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
  Rig.submit (Sub.make d [||]) ~run:(Sub.Run.make ()) ~buffers:[||]

(* The producer, a Polled device, whose work runs when the test runs it. *)
let polled_waits (module G : Gpu) () =
  G.with_ @@ fun t ->
  let pd, pp = P.open_ "POLLED:producer" in
  Fun.protect ~finally:(fun () -> Rig.close pd) @@ fun () ->
  holds (module G) t (empty_point pd) (fun () -> ignore (P.run pp))

(* The second device of [t]'s driver, in rig under a name of its own, for [f];
   skips where the machine has none. *)
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
      (Rig.open_
         {
           name = G.class_ ^ ":producer";
           driver = (module G.D);
           make = (fun () -> Ok g);
         })
  in
  Fun.protect ~finally:(fun () -> Rig.close pd) @@ fun () ->
  let tp = { G.d = pd; g } in
  let sp = G.spin tp ~ns:50_000_000 in
  let point = submit pd [ sp ] ~uses:[] in
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

(* [Rig.reaches] between two devices of the driver answers from what they
   learned when they opened: the same once either is closed. *)
let reaches_closed (module G : Gpu) () =
  G.with_ @@ fun t ->
  with_second (module G) @@ fun g' ->
  let b =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_
         {
           name = G.class_ ^ ":second";
           driver = (module G.D);
           make = (fun () -> Ok g');
         })
  in
  Fun.protect ~finally:(fun () -> Rig.close b) @@ fun () ->
  let both () = [ Rig.reaches t.d b; Rig.reaches b t.d ] in
  let opened = both () in
  Rig.close b;
  equal (list bool) ~msg:"once the second is closed" opened (both ());
  G.close t;
  equal (list bool) ~msg:"once both are closed" opened (both ())

(* Order *)

(* Submissions of parts, each a copy of words between three buffers: a copy
   part, or work, maybe after a spin, so that work run out of order shows in
   the buffers. rig orders the parts that share memory. The reference copies in
   program order. The system submits back to back on one device all programs
   share, and the word it reads never moves back, never passes the last value,
   and stays there. *)
module Order (G : Gpu) = struct
  type part = {
    work : bool; (* by work, else a copy part *)
    late : bool;
    src : int;
    so : int;
    dst : int;
    do_ : int;
    n : int;
  }

  let buffers = 3
  let size = 65536
  let late_ns = 50_000

  let pp_part ppf p =
    Format.fprintf ppf "{%s%s %d@%d -> %d@%d n=%d}"
      (if p.work then "work" else "copy")
      (if p.late then " late" else "")
      p.src p.so p.dst p.do_ p.n

  let pp ppf subs =
    Format.pp_print_list ~pp_sep:Format.pp_print_space
      (fun ppf ps ->
        Format.fprintf ppf "[%a]"
          (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_part)
          ps)
      ppf subs

  let initial b = pattern size (b * 61)
  let shared = fixture ~teardown:G.close G.open_

  (* Whether the device runs copies. *)
  let copies () =
    let t = shared () in
    (G.D.facts t.g).copy_engine <> None

  (* The reference *)

  type t = { bufs : Bytes.t array }

  let make () = { bufs = Array.init buffers (fun b -> Bytes.of_string (initial b)) }

  let overlap (b, o) (b', o') n n' = b = b' && o < o' + n' && o' < o + n

  let conflict p q =
    overlap (p.dst, p.do_) (q.dst, q.do_) p.n q.n
    || overlap (p.src, p.so) (q.dst, q.do_) p.n q.n
    || overlap (p.dst, p.do_) (q.src, q.so) p.n q.n

  (* The result is one: no part copies over its own source. rig orders every
     two parts that conflict. *)
  let determined_one ps =
    List.for_all
      (fun p -> not (overlap (p.src, p.so) (p.dst, p.do_) p.n p.n))
      ps

  let determined _ subs = List.for_all determined_one subs

  let run_one m ps =
    cover "a copy and work in one submission"
      (List.exists (fun p -> p.work) ps && List.exists (fun p -> not p.work) ps);
    cover "parts that conflict"
      (List.exists
         (fun p -> List.exists (fun q -> p != q && conflict p q) ps)
         ps);
    cover "late work" (List.exists (fun p -> p.late) ps);
    List.iter
      (fun p -> Bytes.blit m.bufs.(p.src) p.so m.bufs.(p.dst) p.do_ p.n)
      ps

  let run m subs =
    cover "several values in flight" (List.length subs > 1);
    List.iter (run_one m) subs

  (* The system *)

  type sys = { t : G.t; copies : bool; regions : B.t array }

  let start () =
    let t = shared () in
    {
      t;
      copies = copies ();
      regions = Array.init buffers (fun b -> filled t.d (initial b));
    }

  (* A part's system parts, in order: a late part is a spin, then its copy. *)
  let pieces s p =
    let dst = B.view s.regions.(p.dst) ~first:p.do_ ~length:p.n
    and src = B.view s.regions.(p.src) ~first:p.so ~length:p.n in
    if (not p.work) && s.copies then
      [ copy_part ~dst src ]
    else
      let c = G.copy_words s.t ~dst ~src in
      if not p.late then [ c ]
      else
        let sp = G.spin s.t ~ns:late_ns in
        [ sp; c ]

  (* Submits [ps]'s system parts. *)
  let submit_one s ps = submit s.t.d (List.concat_map (pieces s) ps) ~uses:[]

  (* Submits [subs] back to back, then waits for each value in turn, reading
     the word before each wait: each read is at least the one before and at
     most the last. *)
  let run_sys s subs =
    let first = Rig.submitted s.t.d + 1 in
    let points = List.map (submit_one s) subs in
    let last = Rig.submitted s.t.d in
    let watch seen p =
      let w = Rig.signaled s.t.d in
      at_least int ~msg:"the word" ~than:seen w;
      at_most int ~msg:"the word" ~than:last w;
      Rig.Point.wait p;
      w
    in
    ignore (List.fold_left watch (first - 1) points : int);
    Rig_gpu_support.still ~msg:"the word" int last
      (fun () -> Rig.signaled s.t.d)
      ~ms:1

  (* Every command checks every live buffer. These synchronous reads share a
     landing buffer so they keep its host mapping and allocate no byte arrays. *)
  let readback = fixture (fun () -> (B.create Rig.host size, Bytes.create size))

  let bytes =
    Testable.make ~equal:Bytes.equal
      ~pp:(fun ppf b -> Testable.pp octets ppf (Bytes.to_string b))

  let invariant m s =
    let host, actual = readback () in
    Array.iteri
      (fun b r ->
        B.copy ~src:r ~dst:host;
        B.blit_to_bytes host 0 actual 0 size;
        equal bytes ~msg:(strf "buffer %d" b) m.bufs.(b) actual)
      s.regions

  (* Parts copy whole words at word offsets, as work does. *)
  let parts =
    let open Gen in
    let word x = x land lnot 3 in
    let part =
      let+ work = bool
      and+ late = frequency [ (2, constant false); (1, constant true) ]
      and+ src = int_range 0 (buffers - 1)
      and+ dst = int_range 0 (buffers - 1)
      and+ n = one_of [ int_range 4 64; int_range 4 16384 ]
      and+ so = int_range 0 (size - 16384)
      and+ do_ = int_range 0 (size - 16384) in
      {
        work;
        late = work && late;
        src;
        so = word so;
        dst;
        do_ = word do_;
        n = word n;
      }
    in
    let submission = list ~size:(int_range 0 3) part in
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
      test "the facts are well formed" (facts g);
      test "make refuses work no queue of the device runs"
        (refusals g);
      prop ~count:30
        "copies through any two memories on copy queues are the identity" trips
        (copies g);
      test "a profile records each copy part with its bytes and span times"
        (profiled_copies g);
      prop ~count:50
        "a copy keeps every byte at any offset and length, in queue order"
        chain (chains g);
      prop ~count:30
        "whatever wrote memory last, the next reader reads that write" accesses
        (readers g);
      prop ~count:10
        "a blit or a fill follows the work before it, and work after it reads \
         it"
        blit_cases (blits g);
      (* Three pieces of the staging memory that Device memory the host does
         not address goes through. *)
      test "a blit or a fill of three staging pieces is ordered as one"
        (fun () -> blits g (Device, (64 lsl 20) + 8));
      test "a value's work starts once every earlier value's completed"
        (device_order g);
      test "a profile around a submission records a span that bounds its work"
        (spans g);
      test "spans follow each other across a new measure, on either queue"
        (spans_across g);
      test "a span lies within its window after the device idles"
        (spans_after_idle g);
      prop ~count:30
        "a launch writes what its grid and parameters say, and a launch after \
         it reads that"
        geometry (launches_conform g);
      test "a launch of no size along an axis is refused" (empty_launches g);
      test "an idle device runs its most parts of the largest launches at once"
        (largest_launches g);
      test "a launch gives the parameters, and a run sets at most the threads, \
            its image states"
        (entry_facts g);
      test "a value's launches start once every earlier value's work completed"
        (launch_order g);
      prop ~count:30 "a launch reads what its submission wrote into a workspace"
        launches (workspace g);
      test "an image names its kernels, and bytes no binary are an error"
        (images g);
      test "a close of an idle device leaves the word at the last value"
        (idle_close g);
      test "opens and closes keep a fixed record per open" (open_close g);
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
      test "work past the hang bound loses the device, which opens again"
        (hung g);
      test "work runs after a Polled device's point it waits for"
        (polled_waits g);
      test "work runs after another device's point it waits for"
        (device_waits g);
      test "peer is whether map_peer maps Device memory" (peers g);
      test "reaches between two devices answers the same once they close"
        (reaches_closed g);
      stateful ~timeout:300. ~count:100 ~steps:20
        "values complete in order and the word never moves backwards (sampled)"
        O.commands;
    ]

let gpus : (module Gpu) list =
  [
    (module Rig_metal_support);
    (module Rig_cuda_support);
    (module Rig_nv_support);
    (module Rig_amd_support);
  ]

let () = exit (run "rig_conformance" (List.map laws gpus))
