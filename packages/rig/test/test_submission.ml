(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rig.mli's Submissions: a submission's parts, how it keeps their memory,
   [make], [address] and what makes a submission and a submit valid. Polled
   devices run their queue only when the test runs it or a wait sleeps, so a
   submit there hands its work over and returns; a software device runs it
   before [submit] returns. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module Support = Rig_support
module P = Support.Polled

let memory name = Result.get_ok (Rig.software name)
let submit ?(buffers = [||]) ?(args = [||]) s = Rig.submit s ~buffers ~args

(* [b] borrowed on [d], as a submit on [d] takes it. *)
let on d b = require_some (B.borrow d b)

(* The 64-bit word at byte [at] of [b], which the host addresses. *)
let word ?(at = 0) b = Support.load (Support.Reader.host b + at)

let contents b =
  let s = Bytes.create (B.length b) in
  B.blit_to_bytes b 0 s 0 (B.length b);
  Bytes.to_string s

let buffer_copy src dst = Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))

(* The values of a submission's one fill, which reads none. *)
let one = Support.zeros 1

(* The values of a submission's one fill, with [params]. *)
let values params = [| { Support.zero with params } |]

(* The values of a [Support.poke] fill whose ref is at byte 0: it stores [value]
   at [offset] bytes into its ref's buffer. *)
let poke_params ?(params = Bytes.create 16) ~offset value =
  Bytes.set_int64_le params 0 (Int64.of_int offset);
  Bytes.set_int64_le params 8 (Int64.of_int value);
  params

let fill ?(params = 16) ?(refs = [| Sub.write 0 ~at:0 |]) fn =
  Sub.fill ~refs fn ~params

(* Submissions *)

(* A submission keeps the buffers it was made with, whatever the caller's parts
   array holds later: a take of a part's buffer is refused, and one of the
   buffer the array names since succeeds. *)
let test_parts_changed () =
  let d = memory "submission:parts-changed" in
  let a = B.create d 8 and b = B.create d 8 and dst = B.create d 8 in
  let parts = [| buffer_copy a dst |] in
  let s = Sub.make d parts in
  parts.(0) <- buffer_copy b dst;
  let takes x = Option.is_some (Support.donated x) in
  equal ~msg:"takes of the part's buffer and of the array's" (pair bool bool)
    (false, true)
    (takes a, takes b);
  Rig.Point.wait (submit s)

(* The part's buffer is reachable only through the submission, and its submit
   copies its bytes. *)
let test_parts_kept () =
  let d = memory "submission:parts-kept" in
  let collected = ref 0 in
  let fresh () =
    let a = B.create d 8 in
    Support.store (Support.Reader.host a) 41;
    Gc.finalise_last (fun () -> incr collected) a;
    a
  in
  let b = B.create d 8 and dst = B.create d 8 in
  Support.store (Support.Reader.host b) 42;
  let parts = [| buffer_copy (fresh ()) dst |] in
  let s = Sub.make d parts in
  parts.(0) <- buffer_copy b dst;
  Gc.full_major ();
  equal ~msg:"finalisations of the part's buffer" int 0 !collected;
  ignore (submit s);
  equal ~msg:"the original part's bytes" int 41 (word dst);
  equal ~msg:"the array's new buffer" int 42 (word b);
  ignore (Sys.opaque_identity s)

(* Each submission copies between two fresh buffers that only it holds, and each
   submit writes a fresh slot buffer the caller drops at once. Polled logs every
   free with its word: no free may name a part's memory while its submission is
   reachable, nor memory a submit used before the device's word reached that
   submit's value. *)
type watched = {
  addresses : int list;
  since : int;  (** Frees logged before the memory was made. *)
  mutable dropped : int option;  (** Frees logged before it was dropped. *)
  mutable last : int;  (** The last value that used it. *)
}

type device = {
  d : Rig.t;
  p : P.t;
  base : int;  (** The value its word held when it opened. *)
  mutable watched : watched list;
  mutable last_point : Rig.Point.t option;
}

type submission = {
  dev : device;
  mutable sub : Sub.t option;  (** [None] once dropped. *)
  parts : watched;
}

type device_model = { mutable value : int }
type submission_model = { model : device_model; mutable live : bool }

let devices = Atomic.make 0

(* The handle of [b]'s memory: the address of a Polled region. *)
let handle b = Nativeint.to_int (B.handle b)

let open_device () =
  let n = Atomic.fetch_and_add devices 1 in
  let d, p = P.open_ (Printf.sprintf "submission:lifetime-%d" n) in
  { d; p; base = Rig.submitted d; watched = []; last_point = None }

let frees t = List.length (P.frees t.p)

let watch t ?dropped ~last addresses =
  let w = { addresses; since = frees t; dropped; last } in
  t.watched <- w :: t.watched;
  w

let check_frees _ t =
  cover "memory a submission named is freed"
    (List.exists
       (fun (at, _) -> List.exists (fun w -> List.mem at w.addresses) t.watched)
       (P.frees t.p));
  List.iteri
    (fun i (at, word) ->
      List.iter
        (fun w ->
          if i >= w.since && List.mem at w.addresses then
            match w.dropped with
            | Some j when j <= i ->
                at_least
                  ~msg:(Printf.sprintf "the word when %#x was freed" at)
                  int ~than:w.last word
            | _ -> failf "%#x was freed while a submission held it" at)
        t.watched)
    (P.frees t.p)

let make t =
  let src = B.create t.d 256 and dst = B.create t.d 256 in
  let parts = watch t ~last:0 [ handle src; handle dst ] in
  let s =
    Sub.make t.d [| buffer_copy src dst; Support.touching [| Sub.write |] |]
  in
  { dev = t; sub = Some s; parts }

let submit_sub sub =
  let t = sub.dev in
  let s = Option.get sub.sub in
  let out = B.create t.d 256 in
  let at = handle out in
  let pt = Rig.submit s ~buffers:[| out |] ~args:one in
  t.last_point <- Some pt;
  let v = Rig.Point.value pt in
  sub.parts.last <- v;
  ignore (watch t ~dropped:(frees t) ~last:v [ at ]);
  v - t.base

let drop sub =
  sub.sub <- None;
  sub.parts.dropped <- Some (frees sub.dev);
  Gc.full_major ()

let drain t =
  Gc.full_major ();
  ignore (B.create t.d 64);
  Rig.free_cache t.d

(* A program's device closes when it ends, so no later collection walks it. *)
let device =
  abstract ~invariant:check_frees ~release:(fun t -> Rig.close t.d) "d"

let submission = abstract "s"
let nothing _ = ()

let lifetime =
  [
    command "open"
      (Gen.unit @-> makes device)
      (fun () -> { value = 0 })
      open_device;
    command "make"
      (device ^-> makes submission)
      (fun model -> { model; live = true })
      make;
    command "submit"
      ~pre:(fun r -> r.live)
      (submission ^-> returns int)
      (fun r ->
        r.model.value <- r.model.value + 1;
        r.model.value)
      submit_sub;
    command "drop"
      ~pre:(fun r -> r.live)
      (submission ^-> returns unit)
      (fun r -> r.live <- false)
      drop;
    command "drain" (device ^-> returns unit) nothing drain;
    command "device runs"
      (device ^-> returns unit)
      nothing
      (fun t -> ignore (P.run t.p));
    command "wait"
      (device ^-> returns unit)
      nothing
      (fun t -> Option.iter Rig.Point.wait t.last_point);
  ]

(* Copy *)

(* A copy from a view of [src_len] bytes at [src_first] of one allocation into a
   view of [dst_len] bytes at [dst_first] of the same allocation, if [shared],
   or of another; each end a submit's slot, if [src_slot] or [dst_slot], or a
   buffer the submission names. *)
type ends = {
  shared : bool;
  src_first : int;
  dst_first : int;
  src_len : int;
  dst_len : int;
  src_slot : bool;
  dst_slot : bool;
}

let ends_alloc = 40

let pp_ends ppf c =
  let side slot = if slot then "slot" else "buffer" in
  Format.fprintf ppf "%s %d+%d -> %s %s%d+%d" (side c.src_slot) c.src_first
    c.src_len (side c.dst_slot)
    (if c.shared then "same " else "")
    c.dst_first c.dst_len

let gen_ends =
  let open Gen in
  let+ shared = bool
  and+ src_first = int_range 0 24
  and+ dst_first = int_range 0 24
  and+ len = int_range 0 16
  and+ other =
    frequency [ (4, constant None); (1, map Option.some (int_range 0 16)) ]
  and+ src_slot = bool
  and+ dst_slot = bool in
  let dst_len = Option.value other ~default:len in
  { shared; src_first; dst_first; src_len = len; dst_len; src_slot; dst_slot }

let ends_overlap c =
  c.shared && c.src_len > 0 && c.dst_len > 0
  && c.src_first < c.dst_first + c.dst_len
  && c.dst_first < c.src_first + c.src_len

(* Two copies of 8 bytes between slots: two submits of one submission. *)
let slot_to_slot =
  {
    shared = false;
    src_first = 0;
    dst_first = 8;
    src_len = 8;
    dst_len = 8;
    src_slot = true;
    dst_slot = true;
  }

let ends_device = fixture (fun () -> memory "submission:ends")

(* A copy whose ends are slots or buffers, in each of the four mixes, copies the
   bytes as a host copy does. Buffers of two lengths or that share a byte are
   refused, by make where both ends are buffers and by the submit otherwise,
   which then takes no value. Copies between two slots run through one
   submission, each case with its own buffers. *)
let ends_law cases =
  let d = ends_device () in
  let slots = Sub.make d Sub.[| copy ~src:(Slot 0) ~dst:(Slot 1) |] in
  cover "several submits of one submission"
    (List.length (List.filter (fun c -> c.src_slot && c.dst_slot) cases) > 1);
  List.iteri
    (fun i c ->
      cover "slot to slot" (c.src_slot && c.dst_slot);
      cover "slot to buffer" (c.src_slot && not c.dst_slot);
      cover "buffer to slot" ((not c.src_slot) && c.dst_slot);
      cover "buffer to buffer" (not (c.src_slot || c.dst_slot));
      cover "buffers of two lengths" (c.src_len <> c.dst_len);
      cover "overlapping views of one allocation" (ends_overlap c);
      cover "disjoint views of one allocation"
        (c.shared && c.src_len = c.dst_len && c.src_len > 0
        && not (ends_overlap c));
      let pattern k =
        String.init ends_alloc (fun j -> Char.chr (((k * 61) + j) land 255))
      in
      let a = B.create d ends_alloc and b = B.create d ends_alloc in
      B.blit_from_string (pattern (2 * i)) 0 a 0 ends_alloc;
      B.blit_from_string (pattern ((2 * i) + 1)) 0 b 0 ends_alloc;
      let src = B.view a ~first:c.src_first ~length:c.src_len in
      let dst =
        B.view (if c.shared then a else b) ~first:c.dst_first ~length:c.dst_len
      in
      let refused = c.src_len <> c.dst_len || ends_overlap c in
      let buffers =
        Array.of_list
          ((if c.src_slot then [ src ] else [])
          @ if c.dst_slot then [ dst ] else [])
      in
      let side slot k b = if slot then Sub.Slot k else Sub.Buffer b in
      let make () =
        if c.src_slot && c.dst_slot then slots
        else
          let dst_side = side c.dst_slot (Bool.to_int c.src_slot) dst in
          Sub.make d [| Sub.copy ~src:(side c.src_slot 0 src) ~dst:dst_side |]
      in
      let before = Rig.submitted d in
      if refused && not (c.src_slot || c.dst_slot) then
        raises_match ~msg:"make" Exn.invalid_arg (fun () -> make ())
      else if refused then begin
        let s = make () in
        raises_match ~msg:"the submit" Exn.invalid_arg (fun () ->
            submit s ~buffers);
        equal ~msg:"a refused submit's value" int before (Rig.submitted d)
      end
      else begin
        let into = if c.shared then 2 * i else (2 * i) + 1 in
        let expected = Bytes.of_string (pattern into) in
        Bytes.blit_string
          (pattern (2 * i))
          c.src_first expected c.dst_first c.src_len;
        let s = make () in
        Rig.Point.wait (submit s ~buffers);
        if not c.shared then
          equal ~msg:"the source's allocation" string
            (pattern (2 * i))
            (contents a);
        equal ~msg:"the destination's allocation" string
          (Bytes.to_string expected)
          (contents (if c.shared then a else b))
      end)
    cases

(* A copy's buffers share no byte, so a driver copies between disjoint ranges: a
   copy between halves of one buffer is made, and one between views that share
   bytes is refused. *)
let test_overlapping_copy () =
  let d = memory "submission:overlapping-copy" in
  let b = B.create d 16 in
  let copy src dst =
    let view first = B.view b ~first ~length:8 in
    buffer_copy (view src) (view dst)
  in
  ignore (Sub.make d [| copy 0 8 |]);
  List.iter
    (fun (msg, src, dst) ->
      raises_match ~msg (Exn.invalid_arg ~sub:"overlap") (fun () ->
          Sub.make d [| copy src dst |]))
    [ ("into itself", 0, 0); ("forwards", 0, 4); ("backwards", 4, 0) ]

(* Fill *)

(* A submission serves successive submits while the first still waits in
   Polled's queue: each call holds the parameter bytes and the addresses of its
   own submit, whatever the caller writes into its values after. A software
   device runs each at its submit. Either way a ref adds its buffer's address to
   the offset the values give, once per submit. *)
let test_reuse () =
  let d, p = P.open_ "submission:reuse" in
  let s = Sub.make d [| fill Support.poke |] in
  let params = Bytes.create 16 in
  let args = values params in
  let a = B.create d 24 and b = B.create d 24 in
  ignore
    (submit s ~buffers:[| a |]
       ~args:(values (poke_params ~params ~offset:8 31)));
  ignore
    (submit s ~buffers:[| b |]
       ~args:(values (poke_params ~params ~offset:16 47)));
  ignore (poke_params ~params ~offset:0 99);
  equal ~msg:"queued" int 2 (P.queued p);
  equal ~msg:"run" int 2 (P.run p);
  equal ~msg:"the first submit's word" int 31 (word ~at:8 a);
  equal ~msg:"the second submit's word" int 47 (word ~at:16 b);
  ignore (poke_params ~params ~offset:8 73);
  ignore (submit s ~buffers:[| a |] ~args);
  ignore (P.run p);
  ignore (submit s ~buffers:[| b |] ~args);
  ignore (P.run p);
  equal ~msg:"a repeated submit's words" (pair int int) (73, 73)
    (word ~at:8 a, word ~at:8 b);
  let m = memory "submission:reuse-software" in
  let s = Sub.make m [| fill Support.poke |] in
  let out = B.create m 16 in
  let args = values (poke_params ~offset:8 41) in
  ignore (submit s ~buffers:[| out |] ~args);
  equal ~msg:"a software device's first submit" int 41 (word ~at:8 out);
  Support.store (Support.Reader.host out + 8) 0;
  ignore (submit s ~buffers:[| out |] ~args);
  equal ~msg:"its repeated submit" int 41 (word ~at:8 out);
  ignore (submit s ~buffers:[| out |] ~args:(values (poke_params ~offset:0 42)));
  equal ~msg:"a submit of other values" (pair int int) (42, 41)
    (word out, word ~at:8 out)

(* Blocks of a submission's launches and fills lie end to end: each part reads
   its own values, also where the fill before has the largest parameters and the
   one after none. *)
let test_mixed () =
  let d, p = P.open_ "submission:mixed" in
  let image =
    require_ok ~pp:Format.pp_print_string (Rig.Image.load d "functions")
  in
  let launch =
    let fill = require_some (Rig.Image.resolve image "fill") in
    Sub.launch ~refs:[| Sub.write 0 ~at:0 |] fill ~params:16
  in
  let s =
    Sub.make d
      [|
        fill ~params:4096 Support.poke;
        launch;
        fill ~params:0 ~refs:[||] Support.noop;
        fill Support.poke;
      |]
  in
  let wide = Bytes.make 4096 '\000' in
  ignore (poke_params ~params:wide ~offset:0 17);
  Bytes.set_int64_le wide 4088 127L;
  let one = (1, 1, 1) in
  let args =
    [|
      { Support.zero with params = wide };
      {
        Sub.groups = one;
        threads = one;
        shared = 0;
        params = poke_params ~offset:8 29;
      };
      Support.zero;
      { Support.zero with params = poke_params ~offset:16 43 };
    |]
  in
  let out = B.create d 24 in
  ignore (submit s ~buffers:[| out |] ~args);
  equal int 1 (P.run p);
  equal (list int) [ 17; 29; 43 ] (List.init 3 (fun i -> word ~at:(i * 8) out))

(* [Sub.fill] reads its refs when it is made: the array changed after holds none
   of them. [carry] copies the word at byte 8 of slot 1 to byte 16 of slot 0. *)
let test_refs () =
  let d, p = P.open_ "submission:refs" in
  let refs = [| Sub.write 0 ~at:0; Sub.read 1 ~at:8 |] in
  let s = Sub.make d [| fill ~params:24 ~refs Support.carry |] in
  refs.(0) <- Sub.read 1 ~at:16;
  let src = B.create d 24 and dst = B.create d 24 in
  Support.store (Support.Reader.host src + 8) 101;
  let params = Bytes.create 24 in
  Bytes.set_int64_le params 0 16L;
  Bytes.set_int64_le params 8 8L;
  Bytes.set_int64_le params 16 8L;
  ignore (submit s ~buffers:[| dst; src |] ~args:(values params));
  ignore (P.run p);
  equal int 101 (word ~at:16 dst)

(* [far] stores the word at its byte 4,096 at the address its last 8 bytes hold,
   past the 4,096 bytes another part could have. *)
let far_fill d =
  Sub.make d
    [| fill ~params:8192 ~refs:[| Sub.write 0 ~at:8184 |] Support.far |]

let far_params () =
  let params = Bytes.make 8192 '\000' in
  Bytes.set_int64_le params 4096 59L;
  Bytes.set_int64_le params 8184 8L;
  params

(* A fill's argument holds its 8,192 bytes of parameters, on a device that runs
   it at its submit and on one that runs it later. *)
let test_wide_fill () =
  let wide d run =
    let out = B.create d 16 in
    ignore
      (submit (far_fill d) ~buffers:[| out |] ~args:(values (far_params ())));
    run ();
    word ~at:8 out
  in
  let software = wide (memory "submission:wide-software") ignore in
  let d, p = P.open_ "submission:wide-polled" in
  let polled = wide d (fun () -> ignore (P.run p)) in
  equal ~msg:"the word stored on a software device and on Polled" (pair int int)
    (59, 59) (software, polled)

(* The parameters past a submit's values read 0, also where an earlier submit of
   the submission wrote them: a submit of 8 bytes leaves [far]'s pointer at 0,
   so it stores 0 at the slot's start, on a device that runs it at its submit
   and on one that runs it later. *)
let test_short_values () =
  let short d run =
    let s = far_fill d in
    let out = B.create d 16 in
    B.blit_from_string (String.make 16 '\xff') 0 out 0 16;
    ignore (submit s ~buffers:[| out |] ~args:(values (far_params ())));
    run ();
    equal ~msg:"the longer submit's word" int 59 (word ~at:8 out);
    ignore (submit s ~buffers:[| out |] ~args:(values (Bytes.make 8 '\001')));
    run ();
    equal ~msg:"the word at the slot's start" int 0 (word out);
    equal ~msg:"the longer submit's word, kept" int 59 (word ~at:8 out)
  in
  short (memory "submission:short-software") ignore;
  let d, p = P.open_ "submission:short-polled" in
  short d (fun () -> ignore (P.run p))

(* Words *)

let test_part_word () =
  List.iter
    (fun n ->
      raises_match
        ~msg:(Printf.sprintf "%d bytes" n)
        (Exn.invalid_arg ~sub:(Printf.sprintf "of %d bytes" n))
        (fun () -> Sub.words (B.create Rig.host n)))
    [ 1; 6; 7 ]

(* Make *)

(* Two [poke] fills name slots 0 and 1, and a submit passes one buffer as both:
   each stores its word. *)
let test_slots_one_buffer () =
  let d = memory "submission:slots-one-buffer" in
  let s =
    Sub.make d
      [| fill Support.poke; fill ~refs:[| Sub.write 1 ~at:0 |] Support.poke |]
  in
  let b = B.create d 16 in
  let args =
    [|
      { Support.zero with params = poke_params ~offset:0 5 };
      { Support.zero with params = poke_params ~offset:8 6 };
    |]
  in
  ignore (submit s ~buffers:[| b; b |] ~args);
  equal (pair int int) (5, 6) (word b, word ~at:8 b)

(* [make]'s stack does not grow with its parts: 2^20 fills make within a stack
   of 2^16 words. *)
let test_parts_stack () =
  let d, _ = P.open_ "submission:parts-stack" in
  let parts = Array.make (1 lsl 20) (Sub.fill Support.noop ~params:0) in
  let gc = Gc.get () in
  Gc.set { gc with stack_limit = 1 lsl 16 };
  let s =
    Fun.protect ~finally:(fun () -> Gc.set gc) (fun () -> Sub.make d parts)
  in
  ignore (Sys.opaque_identity s)

(* Address *)

(* A device's borrow of a host allocation of [B.paged_from] bytes, which starts
   a page, and a submission fixing 32 bytes of it from byte 16. *)
let fixed_view name =
  let d, _ = P.open_ name in
  let host = B.create Rig.host B.paged_from in
  let mapped = on d host in
  let s =
    Sub.make ~fixed:[ (B.view mapped ~first:16 ~length:32, Rig.Read) ] d [||]
  in
  (d, host, mapped, s)

(* A buffer within the fixed memory has the address of its first byte, whichever
   views or borrows of the allocation reach those bytes. *)
let test_address () =
  let d, host, mapped, s = fixed_view "submission:address" in
  let expected = Support.at mapped + 16 in
  equal ~msg:"the fixed view" int expected
    (Sub.address s (B.view mapped ~first:16 ~length:32));
  satisfies int ~claim:"within an int's non-negative range"
    (fun a -> a >= 0)
    expected;
  equal ~msg:"a view of a view of the host's" int (expected + 8)
    (Sub.address s
       (B.view (B.view host ~first:8 ~length:64) ~first:16 ~length:8));
  equal ~msg:"a view of another borrow" int (expected + 8)
    (Sub.address s (B.view (on d host) ~first:24 ~length:8));
  equal ~msg:"an empty view at the fixed memory's end" int (expected + 32)
    (Sub.address s (B.view host ~first:48 ~length:0))

(* No buffer of the fixed memory holds these bytes. An independently wrapped
   alias shares the host's addresses but not its claims, so the submission
   neither keeps nor orders it. *)
let test_address_refused () =
  let d, host, mapped, s = fixed_view "submission:address-refused" in
  let refused ~msg b =
    raises_match ~msg Exn.invalid_arg (fun () -> Sub.address s b)
  in
  refused ~msg:"the whole allocation" host;
  refused ~msg:"a view half inside" (B.view host ~first:8 ~length:16);
  refused ~msg:"an unrelated buffer" (B.create d 8);
  let alias = B.of_bigarray (B.export host Bigarray.char) in
  refused ~msg:"an independently wrapped alias"
    (B.view alias ~first:16 ~length:8);
  let copy =
    Sub.make d [| buffer_copy mapped (B.create d (B.length mapped)) |]
  in
  raises_match ~msg:"a part's buffer, not fixed memory" Exn.invalid_arg
    (fun () -> Sub.address copy mapped)

(* Fixed memory never dies, since the submission holds it; the address of a dead
   buffer raises with its reason. *)
let test_dead_address () =
  let d, _ = P.open_ "submission:dead-address" in
  let b = B.create d 8 and dead = B.create d 8 in
  let s = Sub.make ~fixed:[ (b, Rig.Read) ] d [||] in
  ignore (require_some (Rig.Buffer.take ~why:"reused" [ dead ]));
  raises_match (Exn.invalid_arg ~sub:"reused") (fun () -> Sub.address s dead)

(* Handle-only memory has no address, at [address] and at a submit of a ref that
   names it. *)
let test_handle_only () =
  let d, _ = P.open_ ~addresses:false "submission:handle-only" in
  let base = Rig.submitted d in
  let out = B.create d 8 in
  let s = Sub.make ~fixed:[ (out, Rig.Read_write) ] d [| fill Support.poke |] in
  raises_match ~msg:"address" Exn.invalid_arg (fun () -> Sub.address s out);
  raises_match ~msg:"a submit" Exn.invalid_arg (fun () ->
      submit s ~buffers:[| out |] ~args:(values (poke_params ~offset:0 1)));
  equal ~msg:"no value assigned" int base (Rig.submitted d)

(* Valid submissions *)

(* A Polled device whose memory the host does not address, with buffers of its
   own, a dead one, the host's, another device's and a read-only one, and
   submissions whose submits each row misuses. *)
type misuses = {
  d : Rig.t;
  host : B.t;
  unseen : B.t;
  other : B.t;
  dead : B.t;
  elsewhere : B.t;
  touching : Sub.t;
  slot_copy : Sub.t;
  into_unseen : Sub.t;
  poking : Sub.t;
}

let misuses =
  fixture (fun () ->
      let d, _ = P.open_ ~host_visible:false "submission:misuses" in
      let unseen = B.create d 8 in
      let dead = B.create d 8 in
      ignore (require_some (Support.donated dead));
      {
        d;
        host = B.create Rig.host 8;
        unseen;
        other = B.create d 16;
        dead;
        elsewhere = B.create (fst (P.open_ "submission:misuses-other")) 8;
        touching = Sub.make d [| Support.touching [| Sub.read; Sub.write |] |];
        slot_copy = Sub.make d Sub.[| copy ~src:(Slot 0) ~dst:(Slot 1) |];
        into_unseen =
          Sub.make d Sub.[| copy ~src:(Slot 0) ~dst:(Buffer unseen) |];
        poking = Sub.make d [| fill Support.poke |];
      })

(* Each clause of a valid submission that a row breaks: [make] raises
   [Invalid_argument] naming it. *)
let make_misuses =
  let make e parts = ignore (Sub.make e.d parts) in
  let at at slot = Sub.write slot ~at in
  let refs16 e refs = make e [| fill ~params:16 ~refs Support.noop |] in
  [
    ( "a device that runs no work",
      "runs no submitted work",
      fun _ -> ignore (Sub.make (Support.io "submission:io") [||]) );
    ( "a copy of another device's memory",
      "part 0 copies a buffer on",
      fun e -> make e [| buffer_copy e.host (B.create Rig.host 8) |] );
    ( "a slot no ref or copy names",
      "no ref or copy names slot 0",
      fun e -> make e Sub.[| copy ~src:(Slot 1) ~dst:(Buffer e.unseen) |] );
    ( "a slot below the highest that no ref names",
      "no ref or copy names slot 0",
      fun e -> refs16 e [| at 0 1 |] );
    ( "a slot a copy and an unordered ref name",
      "slot 0 is used both unordered and ordered",
      fun e ->
        make e
          Sub.
            [|
              copy ~src:(Slot 0) ~dst:(Buffer e.unseen);
              Support.touching [| unordered |];
            |] );
    ( "a slot an ordered and an unordered ref name",
      "slot 0 is used both unordered and ordered",
      fun e -> refs16 e [| at 0 0; Sub.unordered 0 ~at:8 |] );
  ]

let test_make_misuse (_, sub, f) =
  let named ex =
    Exn.invalid_arg ~sub:"Rig.Submission.make: " ex && Exn.invalid_arg ~sub ex
  in
  raises_match named (fun () -> f (misuses ()))

(* Each clause of a valid submit that a row breaks: [submit] raises
   [Invalid_argument] naming it and assigns no value. *)
let submit_misuses =
  let touch e buffers = ignore (submit e.touching ~buffers ~args:one) in
  let poke e buffers args = ignore (submit e.poking ~buffers ~args) in
  [
    ( "a copy of buffers of two sizes",
      "a copy's buffers differ in size",
      fun e -> ignore (submit e.slot_copy ~buffers:[| e.unseen; e.other |]) );
    ( "a copy into a buffer of another size",
      "a copy's buffers differ in size",
      fun e -> ignore (submit e.into_unseen ~buffers:[| e.other |]) );
    ( "a copy of the host's memory",
      "buffers.(0) is on",
      fun e -> ignore (submit e.slot_copy ~buffers:[| e.host; e.unseen |]) );
    ( "a copy of a dead buffer",
      "buffers.(0) is dead",
      fun e -> ignore (submit e.slot_copy ~buffers:[| e.dead; e.unseen |]) );
    ( "a read of the host's memory",
      "buffers.(0) is on",
      fun e -> touch e [| e.host; e.unseen |] );
    ( "a write of another device's memory",
      "buffers.(1) is on",
      fun e -> touch e [| e.unseen; e.elsewhere |] );
    ( "more buffers than slots",
      "3 buffers for a submission",
      fun e -> touch e [| e.unseen; e.unseen; e.unseen |] );
    ( "fewer buffers than slots",
      "1 buffer for a submission",
      fun e -> touch e [| e.unseen |] );
    ( "no buffer for a fill's one slot",
      "0 buffers for a submission",
      fun e -> poke e [||] one );
    ( "a dead buffer",
      "buffers.(0) is dead",
      fun e -> touch e [| e.dead; e.unseen |] );
  ]

let test_submit_misuse (_, sub, f) =
  let e = misuses () in
  let before = Rig.submitted e.d in
  let named ex =
    Exn.invalid_arg ~sub:"Rig.submit: " ex && Exn.invalid_arg ~sub ex
  in
  raises_match named (fun () -> f e);
  equal ~msg:"no value assigned" int before (Rig.submitted e.d)

(* A fill's submit of values it does not take names them and the part, and
   assigns no value. *)
let test_fill_values_named () =
  let e = misuses () in
  let before = Rig.submitted e.d in
  raises ~msg:"values longer than the fill's parameters"
    (Invalid_argument
       "Rig.submit: args.(0) has 17 bytes of params, past the 16 of the fill \
        of part 0") (fun () ->
      submit e.poking ~buffers:[| e.unseen |] ~args:(values (Bytes.create 17)));
  raises ~msg:"no values"
    (Invalid_argument
       "Rig.submit: args has 0 elements, not 1, one per launch or fill")
    (fun () -> submit e.poking ~buffers:[| e.unseen |]);
  equal ~msg:"no value assigned" int before (Rig.submitted e.d)

(* A submit's buffer whose memory was taken through another buffer is dead: the
   borrow of a host buffer that was taken. *)
let test_dead_slot () =
  let d = memory "submission:dead-slot" in
  let base = Rig.submitted d in
  let b = B.create Rig.host 8 in
  let s = Sub.make d [| Support.touching [| Sub.read |] |]
  and borrowed = on d b in
  ignore (require_some (Support.donated b));
  raises_match Exn.invalid_arg (fun () ->
      submit s ~buffers:[| borrowed |] ~args:one);
  equal ~msg:"no value assigned" int base (Rig.submitted d)

(* Refs at the first and last words of 4,096 bytes of parameters, and at the
   only word of 16, are valid, and so is a submit of each with one buffer per
   slot and one value per fill: each takes the next value. *)
let test_valid_bounds () =
  let d, _ = P.open_ "submission:valid-bounds" in
  let base = Rig.submitted d in
  let at at = Sub.write 0 ~at in
  let wide =
    Sub.make d [| fill ~params:4096 ~refs:[| at 0; at 4088 |] Support.noop |]
  in
  let s = Sub.make d [| fill ~refs:[| at 0 |] Support.noop |] in
  let run s =
    Rig.Point.value (submit s ~buffers:[| B.create d 8 |] ~args:one)
  in
  equal ~msg:"the values of their submits" (pair int int)
    (base + 1, base + 2)
    (let v = run wide in
     (v, run s))

(* A part no queue of its device runs is refused when the submission is made:
   the host runs no words, and Polled with no copy queue no copy. *)
let test_runs () =
  raises_match ~msg:"words where no queue takes them" Exn.invalid_arg (fun () ->
      Sub.make Rig.host [| Sub.words (B.create Rig.host 8) |]);
  let e, _ = P.open_ ~copies:false "submission:runs-fills" in
  let src = B.create e 8 and dst = B.create e 8 in
  raises_match ~msg:"a copy where no queue copies" Exn.invalid_arg (fun () ->
      Sub.make e [| buffer_copy src dst |]);
  let image = Result.get_ok (Rig.Image.load e "code:64") in
  let main = require_some (Rig.Image.resolve image "main") in
  ignore (Sub.make e [| Sub.launch main ~params:0 |])

(* "the blocks of its launches and fills ... take at most 2^32 - 1 bytes, each
   block 32 bytes and twice its parameters rounded up to a multiple of 16": as
   many fills of 4,096 bytes of parameters as that holds are made, and one more
   is refused. *)
let test_blocks_bound () =
  let d, _ = P.open_ "submission:blocks-bound" in
  let params = 4096 in
  let most = ((1 lsl 32) - 1) / (32 + (2 * params)) in
  let fills n = Array.make n (Sub.fill Support.noop ~params) in
  ignore (Sub.make d (fills most));
  raises_match (Exn.invalid_arg ~sub:"take more than 2^32 - 1 bytes") (fun () ->
      Sub.make d (fills (most + 1)))

let tests =
  [
    group "submissions"
      [
        test
          "a part's buffer refuses a take, whatever the parts array holds since"
          test_parts_changed;
        test
          "a submission keeps its parts' buffers reachable, whatever the parts \
           array holds since"
          test_parts_kept;
        stateful ~count:140
          "memory a submission names returns after its last submit" lifetime;
      ];
    group "copy"
      [
        prop ~count:200
          ~examples:[ [ slot_to_slot; slot_to_slot ] ]
          "a copy between slots and buffers copies as the host does, and \
           refuses buffers of two lengths or that share a byte"
          (Gen.list ~size:(Gen.int_range 1 4) (Gen.with_pp pp_ends gen_ends))
          ends_law;
        test "a copy between views that share a byte is refused"
          test_overlapping_copy;
      ];
    group "fill"
      [
        test
          "a fill reads each submit's own values, whenever its device runs it"
          test_reuse;
        test "each launch and fill of a submission reads its own values"
          test_mixed;
        test "a fill's refs are those given when it was made" test_refs;
        test "a fill's argument holds all its parameters, past 4,096 bytes"
          test_wide_fill;
        test "a fill's parameters past its values read 0" test_short_values;
      ];
    group "words"
      [ test "words of a length no multiple of 4 are refused" test_part_word ];
    group "make"
      [
        test "one buffer fills several slots" test_slots_one_buffer;
        test "make takes no stack per part" test_parts_stack;
      ];
    group "address"
      [
        test
          "a fixed buffer's address is its first byte's, through views and \
           borrows"
          test_address;
        test "a buffer no fixed buffer holds has no address"
          test_address_refused;
        test "a dead buffer has no address" test_dead_address;
        test "handle-only memory has no address, at address or at a submit"
          test_handle_only;
      ];
    group "valid submissions"
      [
        cases
          ~id:(fun (n, _, _) -> n)
          "make refuses each clause a submission breaks" make_misuses
          test_make_misuse;
        cases
          ~id:(fun (n, _, _) -> n)
          "submit refuses each clause a submit breaks" submit_misuses
          test_submit_misuse;
        test "a fill's submit of values it does not take names them"
          test_fill_values_named;
        test "a submit of a buffer taken through another refuses it"
          test_dead_slot;
        test
          "fills with refs at their first and last words run, one buffer per \
           slot"
          test_valid_bounds;
        test "a part of a kind no queue of its device runs is refused at make"
          test_runs;
        test "a submission's blocks hold at most 2^32 - 1 bytes"
          test_blocks_bound;
      ];
  ]

let () = exit (run "rig.submission" tests)
