(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bigarray

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type area = Rig_remote_abi.area

(* Frames' kinds, as wire.mli lists them; rig_remote_link.c holds the same
   numbers. *)

let k_request = 1
let k_answer = 2
let k_handover = 3
let k_drop = 4
let k_word = 5
let k_bytes = 6
let k_close = 10

(* Encoding: integers little-endian, a string as its length (u32) and its
   bytes. *)

let add_u8 b n = Buffer.add_uint8 b n
let add_u32 b n = Buffer.add_int32_le b (Int32.of_int n)
let add_u64 b n = Buffer.add_int64_le b (Int64.of_int n)

let add_string b s =
  add_u32 b (String.length s);
  Buffer.add_string b s

let add_list b add l =
  add_u32 b (List.length l);
  List.iter (add b) l

let add_array b add a =
  add_u32 b (Array.length a);
  Array.iter (add b) a

let add_process b p =
  add_u32 b (match p with Wire.Controller -> 0 | Wire.Agent i -> i)

let add_transfer b (t : Rig_remote_abi.transfer) =
  add_u64 b t.src;
  add_u64 b t.dst;
  add_u64 b t.length

let add_account b (a : Wire.account) =
  add_u64 b a.id;
  add_string b a.name;
  add_string b a.arch;
  add_u64 b a.budget;
  add_list b add_u64 a.reaches

let encoded f =
  let b = Buffer.create 64 in
  f b;
  Buffer.contents b

(* Decoding, from the bytes of a frame's payload. A field that runs past the
   payload, or a value out of its range, is malformed. *)

exception Malformed

type reader = { a : area; mutable at : int }

let reader a = { a; at = 0 }

let take r n =
  if n < 0 || r.at + n > Array1.dim r.a then raise Malformed;
  let at = r.at in
  r.at <- at + n;
  at

let u8 r = Char.code r.a.{take r 1}

let uint r n =
  let at = take r n in
  let v = ref 0 in
  for i = n - 1 downto 0 do
    v := (!v lsl 8) lor Char.code r.a.{at + i}
  done;
  !v

let u32 r = uint r 4

(* A u64 that fits a non-negative int: bits 62 and 63, the top byte's high two,
   are clear. Narrowed first, a value with bit 63 set would alias a small
   one. *)
let u64 r =
  let v = uint r 8 in
  if Char.code r.a.{r.at - 1} land 0xc0 <> 0 then raise Malformed;
  v

let bytes r n =
  let at = take r n in
  String.init n (fun i -> r.a.{at + i})

let string r = bytes r (u32 r)
let list r f = List.init (u32 r) (fun _ -> f r)
let array r f = Array.init (u32 r) (fun _ -> f r)
let process r = match u32 r with 0 -> Wire.Controller | i -> Wire.Agent i

let finished r v =
  if r.at <> Array1.dim r.a then raise Malformed;
  v

let transfer r : Rig_remote_abi.transfer =
  let src = u64 r in
  let dst = u64 r in
  let length = u64 r in
  { src; dst; length }

let account r : Wire.account =
  let id = u64 r in
  let name = string r in
  let arch = string r in
  let budget = u64 r in
  let reaches = list r u64 in
  { id; name; arch; budget; reaches }

(* Requests and answers *)

type any = Any : 'a Wire.request -> any

let memory_code = function `Device -> 0 | `Pinned -> 1 | `Mapped -> 2

let memory r =
  match u8 r with
  | 0 -> `Device
  | 1 -> `Pinned
  | 2 -> `Mapped
  | _ -> raise Malformed

let encode_request : type a. a Wire.request -> string =
 fun r ->
  encoded @@ fun b ->
  match r with
  | Wire.Join { agents } ->
      add_u8 b 1;
      add_list b
        (fun b (a : Wire.agent) ->
          add_string b a.name;
          add_string b a.host;
          add_u32 b a.port)
        agents
  | Wire.Open kind ->
      add_u8 b 2;
      add_string b kind
  | Wire.Alloc { id; device; memory; bytes } ->
      add_u8 b 3;
      add_u64 b id;
      add_u64 b device;
      add_u8 b (memory_code memory);
      add_u64 b bytes
  | Wire.Map { id; device; region } ->
      add_u8 b 4;
      add_u64 b id;
      add_u64 b device;
      add_u64 b region
  | Wire.Load { id; binary } ->
      add_u8 b 5;
      add_u64 b id;
      add_u64 b (String.length binary);
      Buffer.add_string b binary
  | Wire.Entry { image; name } ->
      add_u8 b 6;
      add_u64 b image;
      add_string b name
  | Wire.Rail { id; peer; send; receive } ->
      add_u8 b 7;
      add_u64 b id;
      add_process b peer;
      add_array b add_transfer send;
      add_array b add_transfer receive

let decode_request r =
  let v =
    match u8 r with
    | 1 ->
        let agents =
          list r (fun r ->
              let name = string r in
              let host = string r in
              { Wire.name; host; port = u32 r })
        in
        Any (Wire.Join { agents })
    | 2 -> Any (Wire.Open (string r))
    | 3 ->
        let id = u64 r in
        let device = u64 r in
        let memory = memory r in
        let bytes = u64 r in
        Any (Wire.Alloc { id; device; memory; bytes })
    | 4 ->
        let id = u64 r in
        let device = u64 r in
        let region = u64 r in
        Any (Wire.Map { id; device; region })
    | 5 ->
        let id = u64 r in
        let binary = bytes r (u64 r) in
        Any (Wire.Load { id; binary })
    | 6 ->
        let image = u64 r in
        let name = string r in
        Any (Wire.Entry { image; name })
    | 7 ->
        let id = u64 r in
        let peer = process r in
        let send = array r transfer in
        let receive = array r transfer in
        Any (Wire.Rail { id; peer; send; receive })
    | _ -> raise Malformed
  in
  finished r v

let encode_answer : type a. a Wire.request -> a -> string =
 fun r v ->
  encoded @@ fun b ->
  match r with
  | Wire.Join _ -> add_account b v
  | Wire.Open _ -> add_list b add_account v
  | Wire.Alloc _ -> add_u8 b (Bool.to_int v)
  | Wire.Map _ -> add_u8 b (Bool.to_int v)
  | Wire.Load _ -> ()
  | Wire.Entry _ -> (
      match v with
      | None -> add_u8 b 0
      | Some e ->
          add_u8 b 1;
          add_u64 b e)
  | Wire.Rail _ -> ()

let bool r = match u8 r with 0 -> false | 1 -> true | _ -> raise Malformed

let decode_answer : type a. a Wire.request -> reader -> a =
 fun q r ->
  let v : a =
    match q with
    | Wire.Join _ -> account r
    | Wire.Open _ -> list r account
    | Wire.Alloc _ -> bool r
    | Wire.Map _ -> bool r
    | Wire.Load _ -> ()
    | Wire.Entry _ -> if bool r then Some (u64 r) else None
    | Wire.Rail _ -> ()
  in
  finished r v

(* Hand-overs. The bytes of copies from [Local] follow the hand-over's encoding
   in its frame; each is a view of the frame's payload. *)

let side r =
  match u8 r with
  | 0 ->
      let id = u64 r in
      let offset = u64 r in
      Wire.Region { id; offset }
  | 1 -> Wire.Local
  | _ -> raise Malformed

let part r =
  match u8 r with
  | 0 -> Wire.Words (bytes r (4 * u32 r))
  | 1 ->
      let bytes = u64 r in
      let src = side r in
      let dst = side r in
      if src = Wire.Local && dst = Wire.Local then raise Malformed;
      Wire.Copy { src; dst; bytes }
  | _ -> raise Malformed

let decode_handover r =
  let device = u64 r in
  let value = u64 r in
  let waits =
    array r (fun r ->
        let d = u64 r in
        (d, u64 r))
  in
  let parts = array r part in
  let local = function
    | Wire.Copy { src = Wire.Local; bytes; _ } ->
        Some (Array1.sub r.a (take r bytes) bytes)
    | _ -> None
  in
  let areas = Array.of_list (List.filter_map local (Array.to_list parts)) in
  finished r (Wire.Handover ({ device; value; waits; parts }, areas))

(* Jobs *)

type job = nativeint
type state = Open | Closed | Failed of string

external job_make : unit -> nativeint = "caml_rig_remote_link_job"

external job_wait : nativeint -> int -> int * string
  = "caml_rig_remote_link_job_wait"

external fail : nativeint -> string -> unit = "caml_rig_remote_link_job_fail"
external close : nativeint -> unit = "caml_rig_remote_link_job_close"

let job () =
  let j = job_make () in
  if j = 0n then invalid_arg "Rig_remote_proxy.Link.job: a job is open";
  j

let wait j ~ms =
  match job_wait j (max 0 ms) with
  | 0, _ -> Open
  | 1, _ -> Closed
  | _, why -> Failed why

let failure j = match wait j ~ms:0 with Failed why -> Some why | _ -> None

(* Links *)

external link_make : nativeint -> Unix.file_descr -> string -> int -> nativeint
  = "caml_rig_remote_link_make"

external post : nativeint -> int -> string -> unit = "caml_rig_remote_link_post"

external post_area : nativeint -> int -> string -> area -> unit
  = "caml_rig_remote_link_post_area"

external request_c : nativeint -> string -> int * area
  = "caml_rig_remote_link_request"

external next_c : nativeint -> int * area * int = "caml_rig_remote_link_next"
external area : int -> area * int = "caml_rig_remote_link_area"

external rail_c :
  nativeint ->
  int ->
  int array ->
  int array ->
  area ->
  area ->
  area ->
  nativeint = "caml_rig_remote_link_rail_bc" "caml_rig_remote_link_rail"

external ready_c : nativeint -> int -> unit = "caml_rig_remote_link_ready"
external ready_fn : unit -> nativeint = "caml_rig_remote_link_ready_fn"

external release_rail_c : nativeint -> int -> unit
  = "caml_rig_remote_link_release_rail"

(* The C state's address comes first: the proxies' C reads it there. *)
type t = {
  c : nativeint;
  job : job;
  name : string;
  lock : Mutex.t;
  asked : any Queue.t; (* requests [next] gave, unanswered, oldest first *)
  rails : (int, unit) Hashtbl.t; (* the ids of its rails *)
}

let make job fd ~name ~peer =
  let peer = match peer with Wire.Controller -> 0 | Wire.Agent i -> i in
  let c = link_make job fd name peer in
  if c = 0n then invalid_arg "Rig_remote_proxy.Link.make: the job is closed";
  {
    c;
    job;
    name;
    lock = Mutex.create ();
    asked = Queue.create ();
    rails = Hashtbl.create 4;
  }

let name l = l.name
let job_of l = l.job
let ids = Atomic.make 0
let fresh () = Atomic.fetch_and_add ids 1 + 1
let malformed l = fail l.job (strf "%s: a malformed frame" l.name)
let text a = String.init (Array1.dim a) (fun i -> a.{i})

(* The most bytes of a reason: wire.mli's bound. *)
let max_why = 4096

let cut why =
  if String.length why > max_why then String.sub why 0 max_why else why

let check_id fn id =
  if id < 0 then
    invalid_argf "Rig_remote_proxy.Link.%s: id %d is negative" fn id

let check_ids : type a. a Wire.request -> unit = function
  | Wire.Alloc { id; device; _ } ->
      check_id "request" id;
      check_id "request" device
  | Wire.Map { id; device; region } ->
      check_id "request" id;
      check_id "request" device;
      check_id "request" region
  | Wire.Load { id; _ } -> check_id "request" id
  | Wire.Entry { image; _ } -> check_id "request" image
  | Wire.Rail { id; _ } -> check_id "request" id
  | Wire.Join _ | Wire.Open _ -> ()

(* A refusal's reason, a string. *)
let refusal l a =
  let r = reader a in
  match finished r (string r) with
  | why -> `Refused why
  | exception Malformed ->
      malformed l;
      `Failed (Option.value (failure l.job) ~default:"")

let request l q =
  check_ids q;
  match request_c l.c (encode_request q) with
  | 0, a -> (
      match decode_answer q (reader a) with
      | v -> Ok v
      | exception Malformed ->
          malformed l;
          Error (`Failed (Option.value (failure l.job) ~default:"")))
  | 1, a -> Error (refusal l a)
  | _, a -> Error (`Failed (text a))

let drop l id =
  check_id "drop" id;
  post l.c k_drop (encoded (fun b -> add_u64 b id))

(* CR: Serialize each complete next and answer operation, using separate private
   locks. Concurrent next calls can publish requests in reverse dequeue order;
   concurrent answers can pop in order and post in reverse. Replies carry no
   request id, so accepted calls can receive each other's values. Keep the two
   directions independent: a waiting next must allow earlier requests to be
   answered. *)
let next l =
  let k, a, n = next_c l.c in
  let r = reader (if Array1.dim a = n then a else Array1.sub a 0 n) in
  match
    if k = 0 then Error (text a)
    else if k = k_request then begin
      let (Any q) = decode_request r in
      Mutex.protect l.lock (fun () -> Queue.push (Any q) l.asked);
      Ok (Wire.Request q)
    end
    else if k = k_handover then Ok (decode_handover r)
    else if k = k_drop then Ok (Wire.Drop (finished r (u64 r)))
    else if k = k_close then Ok Wire.Close
    else raise Malformed
  with
  | v -> v
  | exception Malformed ->
      malformed l;
      Error (Option.value (failure l.job) ~default:"")

let answer l q a =
  let oldest =
    Mutex.protect l.lock @@ fun () ->
    match Queue.peek_opt l.asked with
    | Some (Any q') when Obj.repr q' == Obj.repr q ->
        ignore (Queue.pop l.asked);
        true
    | _ -> false
  in
  if not oldest then
    invalid_arg
      "Rig_remote_proxy.Link.answer: the request is not the oldest unanswered";
  let payload =
    match a with
    | Ok v -> "\000" ^ encode_answer q v
    | Error why ->
        encoded (fun b ->
            add_u8 b 1;
            add_string b (cut why))
  in
  post l.c k_answer payload

let word l ~device v =
  post l.c k_word
    (encoded (fun b ->
         add_u64 b device;
         add_u64 b v))

let bytes l ~device ~value b =
  post_area l.c k_bytes
    (encoded (fun h ->
         add_u64 h device;
         add_u64 h value))
    b

(* Rails *)

(* Copies of landing areas start on 256 bytes; counts lie 128 bytes apart. *)
let copy_align = 256
let counts_bytes = 384

let aligned n =
  let a, off = area n in
  let a = Array1.sub a off n in
  Array1.fill a '\000';
  a

(* CR: Validate both endpoints' extents and rounded two-copy sizes before
   sending a Rail request or registering an end. With src=0, dst=max_int and
   length=1, inbound wraps to zero bytes; a valid one-byte send then writes at
   inbound + max_int in recv_rail. Share checked sizing between request
   validation and allocation so native transfers fit their areas. *)
let landing ts field =
  let size =
    Array.fold_left
      (fun m (t : Rig_remote_abi.transfer) -> max m (field t + t.length))
      0 ts
  in
  let stride = (size + copy_align - 1) / copy_align * copy_align in
  aligned (2 * stride)

let flat ts =
  Array.concat
    (List.map
       (fun (t : Rig_remote_abi.transfer) -> [| t.src; t.dst; t.length |])
       (Array.to_list ts))

let rail l ~id ~send ~receive =
  if Array.length send = 0 && Array.length receive = 0 then
    invalid_arg "Rig_remote_proxy.Link.rail: the rail carries no transfer";
  let check (t : Rig_remote_abi.transfer) =
    if t.length <= 0 || t.src < 0 || t.dst < 0 then
      invalid_argf
        "Rig_remote_proxy.Link.rail: a transfer of %d bytes from %d to %d is \
         invalid"
        t.length t.src t.dst
  in
  Array.iter check send;
  Array.iter check receive;
  Mutex.protect l.lock @@ fun () ->
  if Hashtbl.mem l.rails id then
    invalid_argf "Rig_remote_proxy.Link.rail: the link has a rail %d" id;
  let outbound = landing send (fun t -> t.src) in
  let inbound = landing receive (fun t -> t.dst) in
  let counts = aligned counts_bytes in
  let r = rail_c l.c id (flat send) (flat receive) outbound inbound counts in
  let e : Rig_remote_abi.end_ =
    {
      outbound;
      inbound;
      counts;
      ready = ready_c r;
      ready_fn = ready_fn ();
      ready_arg = r;
    }
  in
  Hashtbl.replace l.rails id ();
  e

let release_rail l id =
  release_rail_c l.c id;
  Mutex.protect l.lock (fun () -> Hashtbl.remove l.rails id)
