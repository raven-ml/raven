(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled

let timeout = Rig_support.time 60.
let strf = Printf.sprintf

let functions d =
  require_ok ~pp:Format.pp_print_string (Rig.Image.load d "functions")

let fn image f = require_some (Rig.Image.resolve image f)

let launch ?(params = 0) ?(refs = [||]) image kernel =
  Sub.launch ~refs (fn image kernel) ~params

(* The values of a launch of one group of one thread. *)
let one ?(params = Bytes.empty) () =
  { Sub.groups = (1, 1, 1); threads = (1, 1, 1); shared = 0; params }

let submit ?(buffers = [||]) ?waits s args = Rig.submit ?waits s ~buffers ~args

(* Fresh Polled devices: a name opens once. *)
let opened = Atomic.make 0

let polled ?addresses ?host_visible name =
  P.open_ ?addresses ?host_visible
    (strf "launch:%s-%d" name (Atomic.fetch_and_add opened 1))

(* The bytes of the 64-bit words [vs], little-endian. *)
let words vs =
  let b = Bytes.create (8 * List.length vs) in
  List.iteri (fun i v -> Bytes.set_int64_le b (8 * i) (Int64.of_int v)) vs;
  b

let le64 v = Bytes.to_string (words [ v ])

let host_bytes b n =
  let s = Bytes.create n in
  B.blit_to_bytes b 0 s 0 n;
  Bytes.to_string s

let params_of launches = List.map (fun (l : P.launch) -> l.params) launches

(* Patching *)

(* A submit's buffer, used with [access]: [bytes] bytes, [first] bytes into a
   buffer of Polled's own memory, or into host memory that starts a page,
   borrowed on Polled. *)
type slot = { bytes : int; first : int; borrowed : bool; access : Rig.access }

(* A ref: the parameter word [at] holds [offset] bytes into the submit's
   buffer [slot]. *)
type ref_case = { at : int; slot : int; offset : int }

type launch_case = {
  params : string;  (** The parameter bytes, a whole number of 32-bit words. *)
  refs : ref_case list;
  groups : int * int * int;
  threads : int * int * int;
  shared : int;
}

type case = { slots : slot list; launches : launch_case list }

let pp_case ppf c =
  let pp_slot ppf s =
    Format.fprintf ppf "{%d bytes at %d%s%s}" s.bytes s.first
      (if s.borrowed then ", borrowed" else "")
      (if s.access = Rig.Read_write then ", written" else "")
  in
  let pp_ref ppf r = Format.fprintf ppf "%d->%d+%d" r.at r.slot r.offset in
  let pp_launch ppf l =
    let x, y, z = l.groups and tx, ty, tz = l.threads in
    Format.fprintf ppf
      "{%d params; refs [%a]; groups %dx%dx%d; threads %dx%dx%d; shared %d}"
      (String.length l.params)
      (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_ref)
      l.refs x y z tx ty tz l.shared
  in
  Format.fprintf ppf "slots [%a]; launches [%a]"
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_slot)
    c.slots
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_launch)
    c.launches

let gen_case =
  let open Gen in
  let* nslots = int_range 1 4 in
  let slot =
    let+ bytes = int_range 8 64
    and+ first = int_range 0 2
    and+ borrowed = bool
    and+ access = of_list [ Rig.Read; Rig.Read_write ] in
    { bytes; first = 8 * first; borrowed; access }
  in
  let* slots = list ~size:(constant nslots) slot in
  let launch_case =
    let* words = int_range 0 24 in
    let params_bytes = 4 * words in
    let* params = string_of ~size:(constant params_bytes) char in
    (* The 8-byte words a ref may name: those inside the parameters. *)
    let ats = List.init (params_bytes / 8) (fun k -> 8 * k) in
    let* named = subsequence ats in
    let ref_at at =
      let+ slot = int_range 0 (nslots - 1) and+ offset = int_range 0 7 in
      { at; slot; offset }
    in
    let* refs =
      List.fold_right
        (fun at acc ->
          let+ r = ref_at at and+ rest = acc in
          r :: rest)
        named (constant [])
    in
    let axes n = triple (int_range 1 n) (int_range 1 n) (int_range 1 n) in
    let+ groups = axes 3 and+ threads = axes 4 and+ shared = int_range 0 64 in
    { params; refs; groups; threads; shared }
  in
  let+ launches = list ~size:(int_range 1 2) launch_case in
  { slots; launches }

let gen_case = Gen.with_pp pp_case gen_case

(* One device for every case: a case's memory returns as it is collected. *)
let patch_device = lazy (polled "patch")

(* The size from which host memory a Polled device maps starts on a page. *)
let page = B.paged_from

let make_slot d s =
  let mem =
    if s.borrowed then require_some (B.borrow d (B.create Rig.host page))
    else B.create d (s.first + s.bytes)
  in
  B.view mem ~first:s.first ~length:s.bytes

(* The values of launch [l]: its geometry and parameters, each ref's word the
   offset. *)
let values l =
  let params = Bytes.of_string l.params in
  List.iter
    (fun r -> Bytes.set_int64_le params r.at (Int64.of_int r.offset))
    l.refs;
  { Sub.groups = l.groups; threads = l.threads; shared = l.shared; params }

(* What the function reads: the parameters, each ref's word the address of its
   buffer plus the offset. *)
let patched buffers l =
  let p = Bytes.of_string l.params in
  List.iter
    (fun r ->
      Bytes.set_int64_le p r.at
        (Int64.of_int (Rig_support.at buffers.(r.slot) + r.offset)))
    l.refs;
  Bytes.to_string p

let pp_launch ppf (l : P.launch) =
  let x, y, z = l.groups and tx, ty, tz = l.threads in
  Format.fprintf ppf "groups %dx%dx%d threads %dx%dx%d shared %d params %S" x y
    z tx ty tz l.shared l.params

let launch_t = Testable.make ~pp:pp_launch ~equal:( = )

(* Whether a written slot comes before a read one. *)
let rec write_first = function
  | [] -> false
  | { access = Rig.Read; _ } :: rest -> write_first rest
  | { access = Rig.Read_write; _ } :: rest ->
      List.exists (fun s -> s.access = Rig.Read) rest

let patch_law c =
  let d, p = Lazy.force patch_device in
  let image = functions d in
  let buffers = Array.of_list (List.map (make_slot d) c.slots) in
  let use slot : int -> at:int -> Sub.ref =
    match (List.nth c.slots slot).access with
    | Read -> Sub.read
    | Read_write -> Sub.write
  in
  let launches =
    List.map
      (fun l ->
        let refs =
          Array.of_list (List.map (fun r -> use r.slot r.slot ~at:r.at) l.refs)
        in
        launch image "main" ~params:(String.length l.params) ~refs)
      c.launches
  in
  (* Every slot is a ref's: a fill after the launches names each. *)
  let named =
    Rig_support.touching (Array.of_list (List.mapi (fun i _ -> use i) c.slots))
  in
  let s = Sub.make d (Array.of_list (launches @ [ named ])) in
  let args =
    Array.of_list (List.map values c.launches @ [ Rig_support.zero ])
  in
  ignore (P.launches p);
  let pt = submit ~buffers s args in
  (* Writes into the values once the submit returned change nothing it
     handed over. *)
  Array.iteri
    (fun i (a : Sub.args) ->
      Bytes.fill a.params 0 (Bytes.length a.params) '\x5a';
      args.(i) <- { a with groups = (7, 7, 7) })
    args;
  Rig.Point.wait pt;
  let expected =
    List.map
      (fun l ->
        {
          P.groups = l.groups;
          threads = l.threads;
          shared = l.shared;
          params = patched buffers l;
        })
      c.launches
  in
  equal (list launch_t) expected (P.launches p);
  let refs = List.concat_map (fun l -> l.refs) c.launches in
  let last_word l = (String.length l.params / 8 * 8) - 8 in
  cover "a ref at 0" (List.exists (fun r -> r.at = 0) refs);
  cover "a ref at the last word"
    (List.exists
       (fun l -> List.exists (fun r -> r.at = last_word l && r.at > 0) l.refs)
       c.launches);
  cover "a slot two refs name"
    (List.exists
       (fun r -> List.exists (fun r' -> r' != r && r'.slot = r.slot) refs)
       refs);
  cover "a view at an offset" (List.exists (fun s -> s.first > 0) c.slots);
  cover "a borrowed slot" (List.exists (fun s -> s.borrowed) c.slots);
  cover "a written slot before a read one" (write_first c.slots);
  cover "no parameter bytes" (List.exists (fun l -> l.params = "") c.launches);
  cover "two launches, their blocks adjacent" (List.length c.launches = 2)

(* Short values *)

(* Two submissions of 1 to 3 launches of [main] on one device, of 0 to 96
   bytes of parameters each, and submits in turn: each of one of them, with
   values for each launch whose [params] are the first bytes of its
   parameters, as many as it takes or fewer. *)
type short_case = {
  sizes : int list array;
  submits : (int * Sub.args list) list;
}

let pp_short ppf c =
  let comma ppf () = Format.pp_print_char ppf ',' in
  let pp_ints = Format.pp_print_list ~pp_sep:comma Format.pp_print_int in
  let pp_value ppf (v : Sub.args) =
    let x, y, z = v.groups and tx, ty, tz = v.threads in
    Format.fprintf ppf "{%d bytes; %dx%dx%d; %dx%dx%d; %d}"
      (Bytes.length v.params) x y z tx ty tz v.shared
  in
  let pp_submit ppf (k, vs) =
    Format.fprintf ppf "%d:[%a]" k (Format.pp_print_list ~pp_sep:comma pp_value)
      vs
  in
  Format.fprintf ppf "sizes [%a], [%a]; submits [%a]" pp_ints c.sizes.(0)
    pp_ints c.sizes.(1)
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_submit)
    c.submits

let gen_short =
  let open Gen in
  let sizes =
    list ~size:(int_range 1 3) (map (fun w -> 4 * w) (int_range 0 24))
  in
  let* a = sizes in
  let* b = sizes in
  let value n =
    let* len =
      frequency [ (1, constant 0); (1, constant n); (3, int_range 0 n) ]
    in
    let axes n = triple (int_range 1 n) (int_range 1 n) (int_range 1 n) in
    let+ params = string_of ~size:(constant len) (char_range '\001' '\255')
    and+ groups = axes 3
    and+ threads = axes 4
    and+ shared = int_range 0 64 in
    { Sub.groups; threads; shared; params = Bytes.of_string params }
  in
  let values sizes =
    List.fold_right
      (fun n acc ->
        let+ v = value n and+ rest = acc in
        v :: rest)
      sizes (constant [])
  in
  let submit =
    let* k = int_range 0 1 in
    let+ vs = values (if k = 0 then a else b) in
    (k, vs)
  in
  let+ submits = list ~size:(int_range 1 6) submit in
  { sizes = [| a; b |]; submits }

let gen_short = Gen.with_pp pp_short gen_short
let short_device = lazy (polled "short")
let round16 n = (n + 15) land lnot 15

(* Where each block of a submission of launches of [sizes] bytes of
   parameters starts in a call's blocks: a 32-byte header, then its
   parameters and its place, each up to a multiple of 16 bytes. *)
let starts sizes =
  let at = ref 0 in
  List.map
    (fun n ->
      let s = !at in
      at := s + 32 + (2 * round16 n);
      s)
    sizes

let overlaps (a, b) (c, d) = a < d && c < b

(* Each launch reads its submit's values: its geometry, and the bytes the
   values passed, then 0 to the end of its parameters, whatever an earlier
   submit of either submission passed and its driver wrote: Polled writes
   each block's place. *)
let short_law c =
  let d, p = Lazy.force short_device in
  let image = functions d in
  let make sizes =
    Sub.make d
      (Array.of_list (List.map (fun n -> launch image "main" ~params:n) sizes))
  in
  let subs = Array.map make c.sizes in
  let longest = Array.map (fun s -> Array.make (List.length s) 0) c.sizes in
  let last = ref None in
  ignore (P.launches p);
  List.iter
    (fun (k, values) ->
      let sizes = c.sizes.(k) in
      let length (v : Sub.args) = Bytes.length v.params in
      let lens = List.map length values in
      (* The bytes each launch reads as 0, past its values, and the bytes
         the submit before, of the other submission, left other than 0: its
         headers, and the values Polled wrote into its places. *)
      let zeroed =
        List.map2
          (fun s (n, len) -> (s + 32 + len, s + 32 + n))
          (starts sizes) (List.combine sizes lens)
      in
      let headers, places =
        match !last with
        | Some (j, last_lens) when j <> k ->
            let at = starts c.sizes.(j) in
            let place s n len =
              let p = s + 32 + round16 n in
              (p, p + len)
            in
            ( List.map (fun s -> (s, s + 32)) at,
              List.map2
                (fun s (n, len) -> place s n len)
                at
                (List.combine c.sizes.(j) last_lens) )
        | _ -> ([], [])
      in
      let over written =
        List.exists (fun z -> List.exists (overlaps z) written) zeroed
      in
      cover "a submit shorter than an earlier one of its submission"
        (List.exists Fun.id
           (List.mapi (fun j len -> len < longest.(k).(j)) lens));
      cover "a submit after one of the other submission" (headers <> []);
      cover "bytes read as 0 where the submit before had a header"
        (over headers);
      cover "bytes read as 0 where the driver placed the values before"
        (over places);
      cover "values as long as the parameters"
        (List.exists2 (fun n len -> n = len && n > 0) sizes lens);
      cover "no values for some parameters"
        (List.exists2 (fun n len -> len = 0 && n > 0) sizes lens);
      List.iteri (fun j len -> longest.(k).(j) <- max longest.(k).(j) len) lens;
      last := Some (k, lens);
      Rig.Point.wait (submit subs.(k) (Array.of_list values));
      let read n (v : Sub.args) =
        let len = Bytes.length v.params in
        {
          P.groups = v.groups;
          threads = v.threads;
          shared = v.shared;
          params = Bytes.to_string v.params ^ String.make (n - len) '\000';
        }
      in
      equal ~msg:"the launches" (list launch_t)
        (List.map2 read sizes values)
        (P.launches p))
    c.submits

(* An entry ref's 8 bytes hold its function's value, over the bytes the
   values put there or past them; the launch reads its other bytes as
   passed. *)
let test_entry_overwrites () =
  let d, p = polled "entry" in
  let image = functions d in
  let main = fn image "main" in
  let s =
    Sub.make d [| Sub.launch ~refs:[| Sub.entry main ~at:8 |] main ~params:16 |]
  in
  let expected = [ le64 7 ^ le64 (Rig.Image.entry main) ] in
  ignore (P.launches p);
  Rig.Point.wait (submit s [| one ~params:(words [ 7; 99 ]) () |]);
  equal ~msg:"over the values" (list string) expected
    (params_of (P.launches p));
  Rig.Point.wait (submit s [| one ~params:(words [ 7 ]) () |]);
  equal ~msg:"past the values" (list string) expected
    (params_of (P.launches p))

(* Ordering *)

(* A launch that writes [b] is what a read of [b] waits for. *)
let test_write_seen () =
  let d, _ = polled "write-seen" in
  let image = functions d in
  let out = B.create d 32 in
  let s =
    Sub.make d [| launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |] |]
  in
  let args =
    [| { (one ~params:(words [ 0; 100 ]) ()) with groups = (2, 2, 1) } |]
  in
  ignore (submit ~buffers:[| out |] s args);
  B.wait out Read;
  let host = B.create Rig.host 32 in
  B.copy ~src:out ~dst:host;
  equal string ~msg:"each group's word"
    (String.concat "" (List.map le64 [ 100; 101; 102; 103 ]))
    (host_bytes host 32)

(* A launch that reads [b] after a copy into [b], on the copy queue, reads what
   the copy wrote: the copy's value is the one before the launch's, and a
   value's work starts once the previous value's completed, on every queue. And
   the other way round. *)
let test_device_order () =
  let d, _ = polled "device-order" in
  let image = functions d in
  let data = String.init 64 (fun i -> Char.chr (i + 1)) in
  let src = B.create d 64 and b = B.create d 64 and c = B.create d 64 in
  B.copy ~src:(B.of_string data) ~dst:src;
  let copy_part ~dst src =
    Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
  in
  let copy_into_b = Sub.make d [| copy_part ~dst:b src |] in
  let copying =
    Sub.make d
      [|
        launch image "copy" ~params:24
          ~refs:[| Sub.read 0 ~at:0; Sub.write 1 ~at:8 |];
      |]
  in
  ignore (submit copy_into_b [||]);
  let args = [| one ~params:(words [ 0; 0; 64 ]) () |] in
  ignore (submit ~buffers:[| b; c |] copying args);
  let out = B.create Rig.host 64 in
  B.copy ~src:c ~dst:out;
  equal string ~msg:"the launch read the copy" data (host_bytes out 64);
  (* A launch on COMPUTE:0 writes [b], then a copy on COPY:0 copies it. *)
  let filling =
    Sub.make d [| launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |] |]
  in
  let args =
    [| { (one ~params:(words [ 0; 7 ]) ()) with groups = (8, 1, 1) } |]
  in
  ignore (submit ~buffers:[| b |] filling args);
  let copy_b = Sub.make d [| copy_part ~dst:c b |] in
  ignore (submit copy_b [||]);
  B.copy ~src:c ~dst:out;
  equal string ~msg:"the copy read the launch"
    (String.concat "" (List.init 8 (fun i -> le64 (7 + i))))
    (host_bytes out 64)

(* Lifetime *)

(* A submission keeps its launches' images loaded after the caller's parts array
   changed, and its submit launches the function it was made with. *)
let test_parts_changed () =
  let d, _ = polled "parts-changed" in
  let collected = ref false in
  let load () =
    let image = functions d in
    Gc.finalise_last (fun () -> collected := true) image;
    image
  in
  let fill image =
    launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |]
  in
  let parts = [| fill (load ()) |] in
  let s = Sub.make d parts in
  parts.(0) <- fill (functions d);
  Gc.full_major ();
  equal ~msg:"the part's image collected" bool false !collected;
  let out = B.create d 8 in
  ignore (submit ~buffers:[| out |] s [| one ~params:(words [ 0; 7 ]) () |]);
  let host = B.create Rig.host 8 in
  B.copy ~src:out ~dst:host;
  equal string (le64 7) (host_bytes host 8)

(* A launch reads its refs once: a change to their array once it returned
   changes nothing. *)
let test_parts_read_once () =
  let d, p = polled "read-once" in
  let image = functions d in
  let refs = [| Sub.write 0 ~at:0 |] in
  let part = launch image "main" ~params:16 ~refs in
  refs.(0) <- Sub.write 0 ~at:8;
  let s = Sub.make d [| part |] in
  let out = B.create d 8 in
  ignore (P.launches p);
  ignore (submit ~buffers:[| out |] s [| one ~params:(words [ 0; 7 ]) () |]);
  ignore (P.run p);
  equal ~msg:"the ref the launch was given" (list string)
    [ le64 (Rig_support.at out) ^ le64 7 ]
    (params_of (P.launches p))

(* Values rewritten from a signal handler while a submit of them waits for a
   producer: the submit completes, and its launch reads each byte as the
   values held it before or after. *)
let test_rewritten_values () =
  let producer, pp = polled "rewritten-producer" in
  let d, p = polled "rewritten" in
  let image = functions d in
  let s = Sub.make d [| launch image "main" ~params:8 |] in
  let params = Bytes.make 8 'a' in
  let args = [| one ~params () |] in
  let point = submit (Sub.make producer [||]) [||] in
  let rewrote = ref false in
  let handle _ =
    Bytes.fill params 0 8 'b';
    rewrote := true
  in
  ignore (P.launches p);
  P.interrupt pp;
  let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
  let v =
    Fun.protect
      ~finally:(fun () -> Sys.set_signal Sys.sigint before)
      (fun () -> submit ~waits:[| point |] s args)
  in
  equal ~msg:"the handler ran" bool true !rewrote;
  Rig.Point.wait v;
  let neither l = String.exists (fun c -> c <> 'a' && c <> 'b') l in
  equal ~msg:"launches reading a byte neither before nor after" (list string)
    [] (List.filter neither (params_of (P.launches p)))

(* Refusals *)

let refused ~msg f = raises_match ~msg Exn.invalid_arg f

let test_make_refusals () =
  let d, _ = polled "make" and other, _ = polled "make-other" in
  let opened = Rig.submitted d in
  let image = functions d in
  let make ?(image = image) refs =
    Sub.make d [| launch image "main" ~params:16 ~refs |]
  in
  let at at slot = Sub.write slot ~at in
  refused ~msg:"an image of another device" (fun () ->
      make ~image:(functions other) [||]);
  is_none ~msg:"no such function" (Rig.Image.resolve image "absent");
  refused ~msg:"a ref to slot 2 alone" (fun () -> make [| at 0 2 |]);
  ignore (make [| at 0 0; at 8 1 |]);
  ignore (make [||]);
  equal int ~msg:"no value assigned" opened (Rig.submitted d)

(* Each constructor of a ref or a part refuses what its arguments show,
   naming itself, before a submission is made of it. Polled's fill reads 16
   bytes of parameters, and a launch's block of 32 bytes and twice its
   parameters rounded up to 16 must fit the 2^32 - 1 bytes a submit's blocks
   hold. *)
let test_constructor_refusals () =
  let d, _ = P.open_ ~host_visible:false (strf "launch:constructors") in
  let image = functions d in
  let main = fn image "main" and fill = fn image "fill" in
  let refused name f =
    raises_match ~msg:name
      (Exn.invalid_arg ~sub:("Rig.Submission." ^ name ^ ": "))
      f
  in
  let w at = Sub.write 0 ~at and r at = Sub.read 1 ~at in
  refused "read" (fun () -> Sub.read (-1) ~at:0);
  refused "write" (fun () -> Sub.write 0 ~at:(-8));
  refused "unordered" (fun () -> Sub.unordered 0 ~at:4);
  refused "entry" (fun () -> Sub.entry main ~at:12);
  refused "launch" (fun () -> Sub.launch main ~params:(-1));
  refused "launch" (fun () -> Sub.launch main ~params:((1 lsl 31) - 31));
  ignore (Sub.launch main ~params:((1 lsl 31) - 32));
  raises ~msg:"fewer bytes than the function reads"
    (Invalid_argument
       "Rig.Submission.launch: fill takes 16 bytes of parameters; the launch \
        gives 8")
    (fun () -> Sub.launch fill ~params:8);
  raises ~msg:"a ref past the bytes the function reads"
    (Invalid_argument
       "Rig.Submission.launch: a ref at 16 lies past the 16 bytes of \
        parameters fill takes")
    (fun () -> Sub.launch ~refs:[| w 16 |] fill ~params:24);
  ignore (Sub.launch ~refs:[| w 8 |] fill ~params:24);
  refused "launch" (fun () -> Sub.launch ~refs:[| w 8; r 8 |] main ~params:16);
  refused "launch" (fun () -> Sub.launch ~refs:[| w 16 |] main ~params:16);
  refused "launch" (fun () ->
      Sub.launch ~refs:[| w (max_int - 7) |] main ~params:16);
  ignore (Sub.launch ~refs:[| w 0; r 4088 |] main ~params:4096);
  let noop = Rig_support.noop in
  refused "fill" (fun () -> Sub.fill noop ~params:(-1));
  refused "fill" (fun () -> Sub.fill noop ~params:max_int);
  refused "fill" (fun () -> Sub.fill ~refs:[| w 0; r 0 |] noop ~params:8);
  refused "fill" (fun () -> Sub.fill ~refs:[| w 8 |] noop ~params:8);
  let b = B.create d 16 in
  let view first length = Sub.Buffer (B.view b ~first ~length) in
  let dead = B.create d 8 in
  ignore (require_some (Rig_support.donated dead));
  refused "copy" (fun () -> Sub.copy ~src:(Slot 0) ~dst:(Slot 0));
  refused "copy" (fun () -> Sub.copy ~src:(view 0 8) ~dst:(view 8 4));
  refused "copy" (fun () -> Sub.copy ~src:(view 0 8) ~dst:(view 4 8));
  refused "copy" (fun () -> Sub.copy ~src:(Buffer dead) ~dst:(Slot 0));
  refused "words" (fun () -> Sub.words dead);
  refused "words" (fun () -> Sub.words (B.create Rig.host 6));
  refused "words" (fun () -> Sub.words (B.create d 8))

(* A refused submit assigns no value and loses nothing. *)
let refused_submit ~msg d f =
  let before = Rig.submitted d in
  refused ~msg f;
  equal int ~msg:(msg ^ ": no value") before (Rig.submitted d);
  equal (option string) ~msg:(msg ^ ": not lost") None (Rig.lost d)

(* What Polled's driver refuses of a launch's geometry and refs. *)
let test_submit_refusals () =
  let d, _ = polled ~addresses:false ~host_visible:false "submit" in
  let image = functions d in
  let s =
    Sub.make d [| launch image "main" ~params:8 ~refs:[| Sub.read 0 ~at:0 |] |]
  in
  let buffers = [| B.create ~memory:Pinned d 8 |] in
  let geometry ?(groups = (1, 1, 1)) ?(threads = (1, 1, 1)) ?(shared = 0) () =
    [| { Sub.groups; threads; shared; params = Bytes.empty } |]
  in
  refused_submit ~msg:"values of no group" d (fun () ->
      submit ~buffers s (Rig_support.zeros 1));
  refused_submit ~msg:"a ref to memory with no address" d (fun () ->
      submit ~buffers:[| B.create d 8 |] s (geometry ()));
  refused_submit ~msg:"no groups along y" d (fun () ->
      submit ~buffers s (geometry ~groups:(1, 0, 1) ()));
  refused_submit ~msg:"no threads along x" d (fun () ->
      submit ~buffers s (geometry ~threads:(0, 1, 1) ()));
  refused_submit ~msg:"65536 groups along z" d (fun () ->
      submit ~buffers s (geometry ~groups:(1, 1, 65536) ()));
  refused_submit ~msg:"shared memory past 48 KiB" d (fun () ->
      submit ~buffers s (geometry ~shared:49153 ()));
  let v = submit ~buffers s (geometry ~threads:(1024, 1, 1) ~shared:49152 ()) in
  Rig.Point.wait v

(* The values every submit refuses, with the reason it gives: [main] takes
   12 bytes of parameters and runs at most 1024 threads per group, and each
   size lies in 0 to 2^32 - 1. *)
let size_refusal =
  "Rig.submit: args.(0), for main, has a size that is negative or above 2^32 \
   - 1"

let threads_refusal = strf "Rig.submit: main runs at most 1024 threads per \
                            group; args.(0) sets %s"

let value_refusals =
  let v = one () in
  [
    ( "params past the launch's",
      [| { v with params = Bytes.create 13 } |],
      "Rig.submit: args.(0) has 13 bytes of params, past the 12 of main" );
    ("negative groups", [| { v with groups = (1, -1, 1) } |], size_refusal);
    ("2^32 groups", [| { v with groups = (1 lsl 32, 1, 1) } |], size_refusal);
    ("2^32 threads", [| { v with threads = (1, 1, 1 lsl 32) } |], size_refusal);
    ("negative shared memory", [| { v with shared = -1 } |], size_refusal);
    ("1025 threads", [| { v with threads = (5, 205, 1) } |],
      threads_refusal "5x205x1" );
    ( "threads whose product overflows 64 bits",
      [| { v with threads = (0xffff_ffff, 0xffff_ffff, 2) } |],
      threads_refusal "4294967295x4294967295x2" );
    ( "no values",
      [||],
      "Rig.submit: args has 0 elements, not 1, one per launch or fill" );
    ( "two values",
      [| v; v |],
      "Rig.submit: args has 2 elements, not 1, one per launch or fill" );
  ]

let value_device = lazy (polled "values")

let test_value_refusal (msg, args, why) =
  let d, _ = Lazy.force value_device in
  let s = Sub.make d [| launch (functions d) "main" ~params:12 |] in
  let before = Rig.submitted d in
  raises ~msg (Invalid_argument why) (fun () -> submit s args);
  equal int ~msg:"no value" before (Rig.submitted d);
  equal (option string) ~msg:"not lost" None (Rig.lost d)

(* The values at the bounds of [value_refusals] that a submit runs. *)
let test_value_bounds () =
  let d, p = Lazy.force value_device in
  let s = Sub.make d [| launch (functions d) "main" ~params:12 |] in
  let params = Bytes.make 12 'p' in
  let args = [| { (one ~params ()) with threads = (32, 32, 1) } |] in
  ignore (P.launches p);
  Rig.Point.wait (submit s args);
  equal ~msg:"the launch" (list launch_t)
    [
      {
        P.groups = (1, 1, 1);
        threads = (32, 32, 1);
        shared = 0;
        params = Bytes.to_string params;
      };
    ]
    (P.launches p)

(* What a driver states of its functions: Polled's fill reads 16 bytes of
   parameters and its copy 24, its main states none, and each allows 1024
   threads per group. *)
let test_entry_facts () =
  let d, _ = polled "facts" in
  let image = functions d in
  let params f = Rig.Image.params (fn image f)
  and threads f = Rig.Image.threads (fn image f) in
  equal (option int) ~msg:"fill's parameters" (Some 16) (params "fill");
  equal (option int) ~msg:"copy's parameters" (Some 24) (params "copy");
  equal (option int) ~msg:"main's parameters" None (params "main");
  equal (option int) ~msg:"main's threads" (Some 1024) (threads "main");
  let main = fn image "main" in
  equal string ~msg:"its name" "main" (Rig.Image.name main);
  equal bool ~msg:"its image" true (Rig.Image.image main == image)

(* Resolving a function asks the driver for it, which can fail: with no
   memory for what its launches need after the rounds of reclamation, resolve
   raises Out_of_memory; where the driver finds the function needs more than
   the device offers, Invalid_argument with its reason, the device live; and
   where the asking faults, the device's loss. A part of a resolved function
   asks nothing more. *)
let test_resolve_failures () =
  let d, p = polled "entry-failures" in
  let image = functions d in
  P.fail_at p 1 (`Refuse max_int);
  raises_match
    (function
      | Rig.Out_of_memory (d', n) -> Rig.equal d d' && n = P.entry_scratch
      | _ -> false)
    (fun () -> Rig.Image.resolve image "main");
  P.fail_at p 1 (`Refuse 0);
  let image = functions d in
  P.before p "entry" (fun () -> invalid_arg "main needs more than 1 TiB");
  raises_match (Exn.invalid_arg ~sub:"main needs more than 1 TiB") (fun () ->
      Rig.Image.resolve image "main");
  equal (option string) None (Rig.lost d);
  let main = fn (functions d) "main" and image = functions d in
  P.fail_at p 1 (`Fault "the engine hung");
  ignore
    (Sub.make d
       [|
         Sub.launch main ~params:8;
         Sub.fill ~refs:[| Sub.entry main ~at:0 |] Rig_support.noop ~params:8;
       |]);
  raises_match
    (function
      | Rig.Lost (d', why) -> Rig.equal d d' && why = "the engine hung"
      | _ -> false)
    (fun () -> Rig.Image.resolve image "main")

(* Domains *)

(* The device and image that each domain test's submission launches [fill]
   on, made once for every case. *)
let pool_device =
  lazy
    (let d, _ = polled "pool" in
     (d, functions d))

let filling_on (d, image) =
  Sub.make d [| launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |] |]

(* The values of [fill] that store [v]. *)
let fill_values v = one ~params:(words [ 0; v ]) ()

(* Submits [s], a submission of [filling_on], with [args], and reads back
   the word its launch wrote. *)
let fill_with s args =
  let d, _ = Lazy.force pool_device in
  let out = B.create d 8 in
  Rig.Point.wait (submit ~buffers:[| out |] s args);
  Int64.to_int (String.get_int64_le (host_bytes out 8) 0)

(* Two domains submit one submission, each with values of its own, and each
   reads back its own parameter. *)
let sub_t = abstract "s"

let own_commands =
  [
    command "make"
      (Gen.unit @-> makes sub_t)
      Fun.id
      (fun () -> filling_on (Lazy.force pool_device));
    command "submit"
      (sub_t ^-> Gen.int_range 1 1000 @-> returns int)
      (fun () v -> v)
      (fun s v -> fill_with s [| fill_values v |]);
  ]

(* Two domains share one array of values, whose element a store replaces,
   and submit one submission with it. A submit reads the element once and
   checks what it read, so each reads back the parameter of the last store
   before it in some order of the calls, and raises where that store put
   values the launch refuses: more bytes than its 16. *)
(* The model: the parameter of the last store, [None] where it put values
   the launch refuses. *)
type passed = { mutable stored : int option }
type passing = { sub : Sub.t; args : Sub.args array }

let passing_t =
  abstract
    ~pp:(fun ppf m ->
      match m.stored with
      | Some v -> Format.fprintf ppf "stored %d" v
      | None -> Format.pp_print_string ppf "refused")
    "p"

let too_long = one ~params:(Bytes.make 24 '\000') ()

let passing_commands =
  [
    command "make"
      (Gen.unit @-> makes passing_t)
      (fun () -> { stored = Some 0 })
      (fun () ->
        let sub = filling_on (Lazy.force pool_device) in
        { sub; args = [| fill_values 0 |] });
    command "store"
      (passing_t ^-> Gen.int_range 1 1000 @-> returns unit)
      (fun m v -> m.stored <- Some v)
      (fun t v -> t.args.(0) <- fill_values v);
    command "spoil"
      (passing_t ^-> returns unit)
      (fun m -> m.stored <- None)
      (fun t -> t.args.(0) <- too_long);
    command "submit"
      (passing_t ^-> judges int)
      (fun m -> function
        | Ok got ->
            equal (option int) ~msg:"the last stored parameter" m.stored
              (Some got)
        | Error (Invalid_argument _) when m.stored = None -> ()
        | Error e -> raise e)
      (fun t -> fill_with t.sub t.args);
  ]

(* Blocks the host refuses *)

(* A child process limits its address space to [room] bytes above what it
   maps, and submits a submission whose blocks span more than 256 MiB: the
   submit raises Out_of_memory. With the limit lifted, a submit of another
   submission runs. On Linux only: the limit bounds no allocation
   elsewhere. *)
let room = 64 lsl 20
let spanning = 1 lsl 16

let exe =
  let e = Sys.executable_name in
  if Filename.is_relative e then Filename.concat (Sys.getcwd ()) e else e

(* The bytes the process maps: the [VmSize] of its status. *)
let mapped () =
  let field l = Scanf.sscanf_opt l "VmSize: %d kB" (fun k -> k * 1024) in
  let lines =
    In_channel.with_open_text "/proc/self/status" In_channel.input_lines
  in
  match List.find_map field lines with
  | Some n -> n
  | None -> failwith "no VmSize in /proc/self/status"

let limit n =
  match Rig_support.set_address_space n with
  | 0 -> ()
  | e -> failwith (strf "setrlimit: errno %d" e)

let grow_child () =
  let d, _ = polled "grow" in
  let image = functions d in
  let wide = launch image "main" ~params:4096 in
  let big = Sub.make d (Array.make spanning wide) in
  let big_args = Array.make spanning (one ()) in
  let small =
    Sub.make d [| launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |] |]
  in
  limit (mapped () + room);
  let refused =
    match submit big big_args with
    | _ -> "submitted"
    | exception Stdlib.Out_of_memory -> "Out_of_memory"
  in
  limit max_int;
  let out = B.create d 8 in
  Rig.Point.wait (submit ~buffers:[| out |] small [| fill_values 41 |]);
  Printf.printf "%s\n%Ld\n" refused (String.get_int64_le (host_bytes out 8) 0)

let test_grow_refused () =
  if not (Sys.file_exists "/proc/self/status") then
    skip ~reason:"the limit bounds no allocation off Linux" ();
  let ic = Unix.open_process_args_in exe [| exe; "grow" |] in
  let out = In_channel.input_all ic in
  (match Unix.close_process_in ic with
  | WEXITED 0 -> ()
  | _ -> failf "the child failed: %s" out);
  equal (list string) [ "Out_of_memory"; "41" ]
    (String.split_on_char '\n' (String.trim out))

(* Cost *)

let test_allocation () =
  let d, _ = polled "words" in
  let image = functions d in
  let refs =
    Array.init 8 (fun k ->
        if k < 4 then Sub.read k ~at:(8 * k) else Sub.write k ~at:(8 * k))
  in
  let s = Sub.make d [| launch image "main" ~params:72 ~refs |] in
  let buffers = Array.init 8 (fun _ -> B.create d 8) in
  let params = Bytes.make 72 '\000' in
  let args = [| one ~params () |] in
  let first = submit ~buffers s args in
  let n = 100 in
  let allocates msg f =
    equal int ~msg 0
      (Rig_support.allocated_bytes (fun () ->
           for i = 1 to n do
             f i
           done))
  in
  allocates "a submit of a launch" (fun _ ->
      ignore (Sys.opaque_identity (Rig.submit s ~buffers ~args)));
  allocates "a rewrite of its values in place and a submit" (fun i ->
      Bytes.set_int64_le params 64 (Int64.of_int i);
      ignore (Sys.opaque_identity (Rig.submit s ~buffers ~args)));
  let last = submit ~buffers s args in
  Rig.Point.wait first;
  Rig.Point.wait last

let tests =
  [
    group ~timeout "patching"
      [
        prop
          "a launch reads each ref's word as its buffer's address plus the \
           offset, and its geometry and other bytes as passed"
          gen_case patch_law;
        prop
          "a launch reads its submit's values and 0 past them, whatever an \
           earlier submit passed or its driver wrote"
          gen_short short_law;
        test "an entry ref's bytes hold its function's value"
          test_entry_overwrites;
      ];
    group ~timeout "order"
      [
        test "a read of a buffer waits for the launch that writes it"
          test_write_seen;
        test "a launch and a copy on another queue follow each other"
          test_device_order;
      ];
    group ~timeout "lifetime"
      [
        test "a submission keeps its launches' images after their array changed"
          test_parts_changed;
        test "a launch reads its refs once, whatever changes them after"
          test_parts_read_once;
        test "values rewritten during their submit give it some bytes of each"
          test_rewritten_values;
      ];
    group ~timeout "refusals"
      [
        test "each constructor refuses what its arguments show"
          test_constructor_refusals;
        test "make refuses each launch its interface states" test_make_refusals;
        test "resolve raises what asking for a function raises"
          test_resolve_failures;
        test "submit refuses a launch the device cannot run"
          test_submit_refusals;
        cases ~id:(fun (msg, _, _) -> msg)
          "submit refuses values no launch takes, naming them" value_refusals
          test_value_refusal;
        test "submit runs the values at those bounds" test_value_bounds;
        test "an image states the facts its driver knows of a function"
          test_entry_facts;
        test "a submit whose blocks the host has no memory for raises"
          test_grow_refused;
      ];
    group ~timeout "domains"
      [
        stateful ~domains:2
          "two domains' submits of one submission each read their own values"
          own_commands;
        stateful ~domains:2
          "a submit reads the element a store left in its values, or refuses \
           it"
          passing_commands;
      ];
    group ~timeout "cost"
      [
        test "a launch's submit and a rewrite of its values allocate nothing"
          test_allocation;
      ];
  ]

let () =
  match Sys.argv with
  | [| _; "grow" |] -> grow_child ()
  | _ -> exit (run "rig.launch" tests)
