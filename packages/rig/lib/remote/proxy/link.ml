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
  add_u64 b a.alignment;
  add_list b add_u64 a.reaches;
  add_list b add_u64 a.reached_by

let encoded f =
  let b = Buffer.create 64 in
  f b;
  Buffer.contents b

(* Decoding, from the bytes of a frame's payload. A field that runs past the
   payload, or a value out of its range, is malformed. *)

exception Malformed

type reader = { a : area; mutable at : int }

let reader a = { a; at = 0 }

(* The bound compares [n] with the bytes left: [r.at + n] wraps for an [n] near
   [max_int]. *)
let take r n =
  if n < 0 || n > Array1.dim r.a - r.at then raise Malformed;
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
  let alignment = u64 r in
  let reaches = list r u64 in
  let reached_by = list r u64 in
  { id; name; arch; budget; alignment; reaches; reached_by }

(* Requests and answers *)

type any = Any : 'a Wire.request -> any

let memory_code : Rig_edge.memory -> int = function
  | Device -> 0
  | Pinned -> 1
  | Mapped -> 2

let memory r : Rig_edge.memory =
  match u8 r with
  | 0 -> Device
  | 1 -> Pinned
  | 2 -> Mapped
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

(* An int that may be absent: [0], or [1] and the int. *)
let add_int_option b = function
  | None -> add_u8 b 0
  | Some n ->
      add_u8 b 1;
      add_u64 b n

let encode_answer : type a. a Wire.request -> a -> string =
 fun r v ->
  encoded @@ fun b ->
  match r with
  | Wire.Join _ -> add_account b v
  | Wire.Open _ -> add_list b add_account v
  | Wire.Alloc _ -> add_u8 b (Bool.to_int v)
  | Wire.Map _ -> add_u8 b (Bool.to_int v)
  | Wire.Load _ -> add_array b add_int_option v
  | Wire.Entry _ -> add_int_option b v
  | Wire.Rail _ -> ()

let bool r = match u8 r with 0 -> false | 1 -> true | _ -> raise Malformed
let int_option r = if bool r then Some (u64 r) else None

let decode_answer : type a. a Wire.request -> reader -> a =
 fun q r ->
  let v : a =
    match q with
    | Wire.Join _ -> account r
    | Wire.Open _ -> list r account
    | Wire.Alloc _ -> bool r
    | Wire.Map _ -> bool r
    | Wire.Load _ -> array r int_option
    | Wire.Entry _ -> int_option r
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
      let width = u64 r in
      let rows = u64 r in
      let src_pitch = u64 r in
      let dst_pitch = u64 r in
      let src = side r in
      let dst = side r in
      if src = Wire.Local && dst = Wire.Local then raise Malformed;
      if src_pitch < width || dst_pitch < width then raise Malformed;
      Wire.Copy { src; dst; width; rows; src_pitch; dst_pitch }
  | 2 ->
      let bytes = u64 r in
      let byte = Char.chr (u8 r) in
      let dst = side r in
      if dst = Wire.Local then raise Malformed;
      Wire.Fill { dst; bytes; byte }
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
    | Wire.Copy { src = Wire.Local; width; rows; _ } ->
        if rows > 0 && width > max_int / rows then raise Malformed;
        let bytes = width * rows in
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

let failed j = match wait j ~ms:0 with Failed why -> Some why | _ -> None

(* Links *)

external link_make : nativeint -> Unix.file_descr -> string -> int -> nativeint
  = "caml_rig_remote_link_make"

external post : nativeint -> int -> string -> unit = "caml_rig_remote_link_post"

external post_area : nativeint -> int -> string -> area -> unit
  = "caml_rig_remote_link_post_area"

external request_c : nativeint -> string -> int * area
  = "caml_rig_remote_link_request"

external next_c : nativeint -> int * area * int = "caml_rig_remote_link_next"
external wait_c : nativeint -> unit = "caml_rig_remote_link_wait"
external area : int -> area * int = "caml_rig_remote_link_area"

external rail_c :
  nativeint -> int -> int array -> int array -> area -> area -> area -> unit
  = "caml_rig_remote_link_rail_bc" "caml_rig_remote_link_rail"

external ready_c : area -> int -> unit = "caml_rig_remote_link_ready"
external ready_fn : unit -> nativeint = "caml_rig_remote_link_ready_fn"

external release_rail_c : nativeint -> int -> unit
  = "caml_rig_remote_link_release_rail"

(* Locks

   A link's two locks are never nested ([next] and [answer]). Each counts
   among the locks its thread holds ([take]), so a call of rig's, or a [next]
   or an [answer], from a signal handler, finaliser or Gc.Memprof callback
   inside one raises [Invalid_argument] as it starts ([check]).

   [answer_lock] is held across an answer's send, which may wait for the
   socket, so that answers leave in the order of their requests. [next] waits
   for a command before it takes [next_lock], and holds it only to dequeue
   one. *)

(* The locks the calling thread holds, and a count of one more and one fewer
   ([rig_holds], [rig_hold] and [rig_unhold] in [rig_edge.h]). *)
external holds : unit -> (int[@untagged])
  = "caml_rig_edge_holds_byte" "caml_rig_edge_holds"
[@@noalloc]

external hold : unit -> unit = "caml_rig_edge_hold" [@@noalloc]
external unhold : unit -> unit = "caml_rig_edge_unhold" [@@noalloc]
external reraise : exn -> 'a = "%reraise"

(* A function a caller outside rig makes, [fn], starts with [check fn]. *)
let check fn =
  if holds () <> 0 then
    invalid_arg
      ("Rig_remote_proxy.Link." ^ fn
     ^ ": called from a signal handler, finaliser or Gc.Memprof callback \
        inside a lock of the calling thread's")

(* [take m] takes [m], counted from before the take, so that a signal handler
   that runs inside a contended take sees it held. An exception that ends the
   take gives the count back first, having taken nothing: one a handler
   raised, or the error-checking [Mutex]'s [Sys_error] for a take by the
   thread that holds [m], which only a call that skipped its check makes and
   which raises [Invalid_argument]. *)
let take m =
  hold ();
  match Mutex.lock m with
  | () -> ()
  | exception e -> (
      unhold ();
      match e with
      | Sys_error why when String.starts_with ~prefix:"Mutex.lock:" why ->
          invalid_arg
            "Rig_remote_proxy: a lock taken again by the thread that holds \
             it, from a signal handler, finaliser or Gc.Memprof callback"
      | e -> reraise e)

(* Gives [m] back, then its count, with nothing between that allocates or
   polls. *)
let give m =
  Mutex.unlock m;
  unhold ()

(* [protect m f] is [f ()] holding [m], taken as the last step before the
   handler and given back as the first step of both exits, where nothing
   allocates or polls, so an exception anywhere in [f], an asynchronous one
   too, leaves it free. *)
let protect m f =
  take m;
  match f () with
  | x ->
      give m;
      x
  | exception e ->
      give m;
      reraise e

(* The C state's address comes first: the proxies' C reads it there. *)
type t = {
  c : nativeint;
  job : job;
  name : string;
  asked : any list Atomic.t;
      (* requests [next] gave, unanswered, newest first *)
  next_lock : Mutex.t; (* held across a [next] *)
  answer_lock : Mutex.t; (* held across an [answer]'s check and send *)
}

let make job fd ~name ~peer =
  let peer = match peer with Wire.Controller -> 0 | Wire.Agent i -> i in
  let c = link_make job fd name peer in
  if c = 0n then invalid_arg "Rig_remote_proxy.Link.make: the job is closed";
  {
    c;
    job;
    name;
    asked = Atomic.make [];
    next_lock = Mutex.create ();
    answer_lock = Mutex.create ();
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
      `Failed (Option.value (failed l.job) ~default:"")

let request l q =
  check_ids q;
  match request_c l.c (encode_request q) with
  | 0, a -> (
      match decode_answer q (reader a) with
      | v -> Ok v
      | exception Malformed ->
          malformed l;
          Error (`Failed (Option.value (failed l.job) ~default:"")))
  | 1, a -> Error (refusal l a)
  | _, a -> Error (`Failed (text a))

let drop l id =
  check_id "drop" id;
  post l.c k_drop (encoded (fun b -> add_u64 b id))

(* Answers carry no request id: the peer pairs them with its requests by wire
   order. So a [next] dequeues and records its request in one step, and an
   [answer] checks, pops and sends in one step, each under its own lock. A
   [next] waits for a frame before it takes its lock, so it holds up no
   answer. The record of requests changes by atomic swaps, so neither lock is
   taken inside the other. *)
let rec push l r =
  let a = Atomic.get l.asked in
  if not (Atomic.compare_and_set l.asked a (r :: a)) then push l r

(* Pops [q] if it is the oldest unanswered request: whether it did. *)
let rec pop_oldest l q =
  let a = Atomic.get l.asked in
  match List.rev a with
  | Any q' :: older when Obj.repr q' == Obj.repr q ->
      Atomic.compare_and_set l.asked a (List.rev older) || pop_oldest l q
  | _ -> false

(* Takes the next command under [l]'s [next_lock], with no wait: [None] if
   none came, as when another [next] took the one its wait saw. *)
let take_next l () =
  let k, a, n = next_c l.c in
  if k < 0 then None
  else
    let r = reader (if Array1.dim a = n then a else Array1.sub a 0 n) in
    match
      if k = 0 then Error (text a)
      else if k = k_request then begin
        let (Any q) = decode_request r in
        push l (Any q);
        Ok (Wire.Request q)
      end
      else if k = k_handover then Ok (decode_handover r)
      else if k = k_drop then Ok (Wire.Drop (finished r (u64 r)))
      else if k = k_close then Ok Wire.Close
      else raise Malformed
    with
    | v -> Some v
    | exception Malformed ->
        malformed l;
        Some (Error (Option.value (failed l.job) ~default:""))

(* The wait for a command holds no lock, so a callback that runs in it calls
   rig as any code does. *)
let rec next l =
  check "next";
  wait_c l.c;
  match protect l.next_lock (take_next l) with
  | Some v -> v
  | None -> next l

let answer l q a =
  check "answer";
  let payload =
    match a with
    | Ok v -> "\000" ^ encode_answer q v
    | Error why ->
        encoded (fun b ->
            add_u8 b 1;
            add_string b (cut why))
  in
  protect l.answer_lock @@ fun () ->
  if not (pop_oldest l q) then
    invalid_arg
      "Rig_remote_proxy.Link.answer: the request is not the oldest unanswered";
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

(* Transfers [Rig_remote_abi.check_transfers] accepted end by [2^60]: an area's
   two copies, each rounded up to [copy_align], fit an int. *)
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
  (match Rig_remote_abi.check_transfers ~send ~receive with
  | Ok () -> ()
  | Error why -> invalid_argf "Rig_remote_proxy.Link.rail: %s" why);
  let outbound = landing send (fun t -> t.src) in
  let inbound = landing receive (fun t -> t.dst) in
  let counts = aligned counts_bytes in
  let e =
    Rig_remote_abi.end_ ~outbound ~inbound ~counts ~ready:(ready_c counts)
      ~ready_fn:(ready_fn ())
  in
  rail_c l.c id (flat send) (flat receive) outbound inbound counts;
  e

let release_rail l id = release_rail_c l.c id

(* Pairs *)

(* Windows has no socketpair. OCaml's binds a socket at a name in the temporary
   directory made of the process id and a count, and removes it once the pair
   is connected: a process killed in between leaves it there. Process ids are
   reused, so a later process of the same id fails to bind it, with
   EADDRINUSE. A loopback connection names no file. Its listener takes the
   connection from the other end, and closes any other, from another process
   of the machine. *)
let tcp_pair () =
  let socket () = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  let l = socket () in
  Fun.protect ~finally:(fun () -> Unix.close l) @@ fun () ->
  Unix.bind l (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  Unix.listen l 1;
  let a = socket () in
  let rec from me =
    let b, peer = Unix.accept ~cloexec:true l in
    if peer = me then b
    else begin
      Unix.close b;
      from me
    end
  in
  match
    Unix.connect a (Unix.getsockname l);
    from (Unix.getsockname a)
  with
  | b -> (a, b)
  | exception e ->
      Unix.close a;
      raise e

let socket_pair () =
  if Sys.win32 then tcp_pair ()
  else Unix.socketpair ~cloexec:true Unix.PF_UNIX Unix.SOCK_STREAM 0
