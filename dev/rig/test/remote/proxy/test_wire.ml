(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The handshake of a job's processes, over loopback TCP. Real ends meet
   through a relay that records, cuts or changes their bytes; a raw end
   writes the documented bytes itself. *)

open Windtrap
module Wire = Rig_remote_proxy.Wire

(* The handshake's three messages, in bytes: the greeting, the dialing end's
   proof and the accepting end's answer. *)
let greeting_size = 8 + 4 + 1 + 32
let proof_size = 32 + 4 + 4 + 32
let answer_size = 1 + 32
let transcript_size = greeting_size + proof_size + answer_size

(* The time an end waits for each answer. *)
let limit = 10.

(* An end that gives up on time returns within this of [limit]. *)
let slack = 1.5

(* Bytes *)

let u32 n =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.of_int n);
  Bytes.to_string b

let random_bytes n = String.init n (fun _ -> Char.chr (Random.int 256))
let greeting ~version nonce = "rig-job\n" ^ u32 version ^ "\000" ^ nonce
let refusal why = "rig-job\n" ^ u32 1 ^ "\001" ^ u32 (String.length why) ^ why

let pp_bytes ppf s =
  if String.length s <= 16 then Format.fprintf ppf "%S" s
  else Format.fprintf ppf "%d bytes %S..." (String.length s) (String.sub s 0 16)

(* Sockets *)

(* A connected pair of loopback TCP sockets. *)
let connected () =
  let l = Unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close l)
    (fun () ->
      Unix.bind l (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
      Unix.listen l 1;
      let port =
        match Unix.getsockname l with
        | Unix.ADDR_INET (_, p) -> p
        | Unix.ADDR_UNIX _ -> assert false
      in
      let d = Unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
      Unix.connect d (Unix.ADDR_INET (Unix.inet_addr_loopback, port));
      let a, _ = Unix.accept l in
      (d, a))

(* The next [n] bytes of [fd], fewer if its stream ends first. *)
let read_n fd n =
  let b = Bytes.create n in
  let rec go off =
    if off = n then off
    else
      match Unix.read fd b off (n - off) with
      | 0 -> off
      | k -> go (off + k)
      | exception Unix.Unix_error (Unix.ECONNRESET, _, _) -> off
  in
  Bytes.sub_string b 0 (go 0)

(* Every byte of [fd] until its stream ends. *)
let read_all fd =
  let buf = Buffer.create 64 in
  let b = Bytes.create 4096 in
  let rec go () =
    match Unix.read fd b 0 (Bytes.length b) with
    | 0 -> ()
    | k ->
        Buffer.add_subbytes buf b 0 k;
        go ()
    | exception Unix.Unix_error (Unix.ECONNRESET, _, _) -> ()
  in
  go ();
  Buffer.contents buf

let write fd s = ignore (Unix.write_substring fd s 0 (String.length s))

(* Whether a byte of [fd] is ready within [s] seconds. *)
let readable fd s =
  let r, _, _ = Unix.select [ fd ] [] [] s in
  r <> []

let timed f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  (r, Unix.gettimeofday () -. t0)

(* Runs [f] on a thread: [join] is its result, or the exception it raised. *)
let spawn f =
  let r = ref None in
  let t =
    Thread.create
      (fun () -> r := Some (match f () with v -> Ok v | exception e -> Error e))
      ()
  in
  fun () ->
    Thread.join t;
    match Option.get !r with Ok v -> v | Error e -> raise e

(* A relay

   The relay stands between the two ends and forwards their bytes, recording
   them in order: the handshake alternates, so the record is its transcript.
   It forwards [cut] bytes and then closes both connections; it xors the byte
   at offset [flip] with 2; it pauses [pace] seconds before each byte, which it
   writes on its own. *)

let relay ?(cut = max_int) ?(flip = -1) ?(pace = 0.) d a =
  let seen = Buffer.create transcript_size in
  let b = Bytes.create 4096 in
  let rec loop () =
    if Buffer.length seen < cut then begin
      let ready, _, _ = Unix.select [ d; a ] [] [] (-1.) in
      let src = List.hd ready in
      let dst = if src = d then a else d in
      match Unix.read src b 0 (Bytes.length b) with
      | 0 | (exception Unix.Unix_error _) -> ()
      | n ->
          let n = min n (cut - Buffer.length seen) in
          for i = 0 to n - 1 do
            if Buffer.length seen = flip then
              Bytes.set b i (Char.chr (Char.code (Bytes.get b i) lxor 2));
            Buffer.add_char seen (Bytes.get b i);
            if pace > 0. then begin
              Thread.delay pace;
              ignore (Unix.write dst b i 1)
            end
          done;
          if pace = 0. then ignore (Unix.write dst b 0 n);
          loop ()
    end
  in
  Fun.protect
    ~finally:(fun () ->
      Unix.close d;
      Unix.close a)
    (fun () ->
      (try loop () with Unix.Unix_error _ -> ());
      Buffer.contents seen)

type outcome = {
  dialed : (unit, string) result;
  accepted : (Wire.process * Wire.process, string) result;
  admitted : Wire.process list;  (* [admit]'s arguments, in order. *)
  transcript : string;
}

(* Runs a dialing end with key [dialing] and an accepting end with key
   [accepting] through a relay. Each end closes its socket once it returns. *)
let handshake ?cut ?flip ?pace ?(admit = fun _ -> Ok ())
    ?(self = Wire.Controller) ?(peer = Wire.Agent 1) ~dialing ~accepting () =
  let d, rd = connected () in
  let ra, a = connected () in
  let relayed = spawn (fun () -> relay ?cut ?flip ?pace rd ra) in
  let admitted = ref [] in
  let accepted =
    spawn (fun () ->
        Fun.protect
          ~finally:(fun () -> Unix.close a)
          (fun () ->
            Wire.accept a ~key:accepting ~admit:(fun p ->
                admitted := p :: !admitted;
                admit p)))
  in
  let dialed =
    Fun.protect
      ~finally:(fun () -> Unix.close d)
      (fun () -> Wire.dial d ~key:dialing ~self ~peer)
  in
  let accepted = accepted () in
  let transcript = relayed () in
  { dialed; accepted; admitted = List.rev !admitted; transcript }

(* Witnesses *)

let pp_process ppf = function
  | Wire.Controller -> Format.pp_print_string ppf "controller"
  | Wire.Agent i -> Format.fprintf ppf "agent %d" i

let process = Testable.structural ~pp:pp_process
let dialed = result unit string
let accepted = result (pair process process) string
let key = String.make 32 'k'

(* Keys *)

let key_bounds () =
  equal int 16 Wire.min_key;
  equal int 4096 Wire.max_key;
  List.iter
    (fun n ->
      let k = random_bytes n in
      let o = handshake ~dialing:k ~accepting:k () in
      equal ~msg:(Printf.sprintf "%d-byte key" n) dialed (Ok ()) o.dialed;
      equal ~msg:(Printf.sprintf "%d-byte key" n) accepted
        (Ok (Wire.Controller, Wire.Agent 1))
        o.accepted)
    [ Wire.min_key; Wire.max_key ]

let bad_sizes = [ 0; 1; Wire.min_key - 1; Wire.max_key + 1; 2 * Wire.max_key ]

(* With a silent peer, a raise that waited for it would come after [limit]. *)
let dial_bad_key n =
  let d, a = connected () in
  Fun.protect
    ~finally:(fun () ->
      Unix.close d;
      Unix.close a)
    (fun () ->
      let (), took =
        timed (fun () ->
            raises_match Exn.invalid_arg (fun () ->
                Wire.dial d ~key:(String.make n 'k') ~self:Wire.Controller
                  ~peer:(Wire.Agent 1)))
      in
      less float_exact ~than:1. took)

let accept_bad_key n =
  let d, a = connected () in
  Fun.protect
    ~finally:(fun () ->
      Unix.close d;
      Unix.close a)
    (fun () ->
      raises_match Exn.invalid_arg (fun () ->
          Wire.accept a ~key:(String.make n 'k') ~admit:(fun _ -> Ok ()));
      equal ~msg:"the accepting end sent nothing" bool false (readable d 0.05))

let bad_peers =
  [
    (Wire.Controller, Wire.Controller);
    (Wire.Agent 1, Wire.Controller);
    (Wire.Agent 2, Wire.Agent 2);
    (Wire.Controller, Wire.Agent 0);
    (Wire.Controller, Wire.Agent (1 lsl 32));
    (Wire.Agent 0, Wire.Agent 1);
    (Wire.Agent (-1), Wire.Agent 1);
    (Wire.Agent (1 lsl 32), Wire.Agent 1);
  ]

let dial_bad_peer (self, peer) =
  let d, a = connected () in
  Fun.protect
    ~finally:(fun () ->
      Unix.close d;
      Unix.close a)
    (fun () ->
      let (), took =
        timed (fun () ->
            raises_match Exn.invalid_arg (fun () ->
                Wire.dial d ~key ~self ~peer))
      in
      less float_exact ~than:1. took)

(* The law over keys. A second key is the first, or differs from it by one
   byte, by its last byte, or entirely. *)

type change = Same | Byte of int | Shorter | Other of string

let changed k = function
  | Same -> k
  | Byte i ->
      String.mapi
        (fun j c -> if i = j then Char.chr (Char.code c lxor 1) else c)
        k
  | Shorter -> String.sub k 0 (String.length k - 1)
  | Other o -> o

let pp_change ppf = function
  | Same -> Format.pp_print_string ppf "same"
  | Byte i -> Format.fprintf ppf "byte %d" i
  | Shorter -> Format.pp_print_string ppf "shorter"
  | Other o -> Format.fprintf ppf "other %a" pp_bytes o

let key_size =
  Gen.frequency
    [
      (3, Gen.int_range Wire.min_key Wire.max_key);
      ( 2,
        Gen.of_list ~pp:Format.pp_print_int
          [ Wire.min_key; Wire.min_key + 1; 127; 128; 129; Wire.max_key ] );
    ]

let agent =
  Gen.map
    (fun i -> Wire.Agent i)
    (Gen.frequency
       [
         (4, Gen.int_range 1 8);
         (1, Gen.of_list ~pp:Format.pp_print_int [ 0x7fff_ffff; 0xffff_ffff ]);
       ])

let processes =
  Gen.such_that
    (fun (s, p) -> s <> p)
    (Gen.pair (Gen.one_of [ Gen.constant Wire.Controller; agent ]) agent)

let keys =
  let open Gen in
  let* n = key_size in
  let* k = string_of ~size:(constant n) char in
  let+ c =
    frequency
      [
        (3, constant Same);
        (2, map (fun i -> Byte i) (int_range 0 (n - 1)));
        (1, constant (if n > Wire.min_key then Shorter else Same));
        (1, map (fun o -> Other o) (string_of ~size:(constant n) char));
      ]
  in
  (k, c)

let keys_law ((k, c), (self, peer)) =
  let k' = changed k c in
  let o = handshake ~self ~peer ~dialing:k ~accepting:k' () in
  cover "equal keys" (k = k');
  cover "keys differing in one byte" (match c with Byte _ -> true | _ -> false);
  cover "a key and its prefix" (c = Shorter);
  cover "keys longer than a hash block" (String.length k > 128);
  cover "an agent dialing" (self <> Wire.Controller);
  if k = k' then begin
    equal dialed (Ok ()) o.dialed;
    equal accepted (Ok (self, peer)) o.accepted;
    equal (list process) [ self ] o.admitted
  end
  else begin
    is_error ~msg:"dialing end" o.dialed;
    is_error ~msg:"accepting end" o.accepted;
    equal ~msg:"admit is asked only once the key is proven" (list process) []
      o.admitted
  end

(* HMAC pads a key shorter than its hash's block with zero bytes, so a key and
   the key with a trailing NUL give one proof unless the handshake tells them
   apart. *)
let trailing_nul () =
  List.iter
    (fun n ->
      let k = random_bytes n in
      let o = handshake ~dialing:k ~accepting:(k ^ "\000") () in
      let msg = Printf.sprintf "%d-byte key" n in
      is_error ~msg o.dialed;
      is_error ~msg o.accepted)
    [ Wire.min_key; 127 ]

let pp_case ppf ((k, c), (self, peer)) =
  Format.fprintf ppf "key %a, %a; %a dials %a" pp_bytes k pp_change c pp_process
    self pp_process peer

let keys_group =
  group "keys"
    [
      test "keys of min_key and max_key bytes connect" key_bounds;
      cases ~name:(Printf.sprintf "dial refuses a %d-byte key at once")
        "dial's key size" bad_sizes dial_bad_key;
      cases
        ~name:(Printf.sprintf "accept refuses a %d-byte key before it speaks")
        "accept's key size" bad_sizes accept_bad_key;
      cases
        ~name:(fun (s, p) ->
          Format.asprintf "dial refuses %a dialing %a at once" pp_process s
            pp_process p)
        "dial's peer" bad_peers dial_bad_peer;
      prop ~count:200
        "ends connect, naming both processes, iff their keys are equal"
        (Gen.with_pp pp_case (Gen.pair keys processes))
        keys_law;
      test "a key and the key with a trailing NUL do not connect" trailing_nul;
    ]

(* Admission *)

let admit_refuses () =
  let why = "a second controller" in
  let o =
    handshake ~admit:(fun _ -> Error why) ~dialing:key ~accepting:key ()
  in
  equal dialed (Error why) o.dialed;
  is_error o.accepted;
  equal (list process) [ Wire.Controller ] o.admitted

let admission =
  group "admit"
    [
      test "admit's refusal reaches the dialing end as its reason"
        admit_refuses;
    ]

(* Refusals *)

let refuse_bytes () =
  let why = "busy\000\xff" in
  let d, a = connected () in
  Wire.refuse a why;
  let got =
    Fun.protect ~finally:(fun () -> Unix.close d) (fun () -> read_all d)
  in
  equal string (refusal why) got

let refuse_reaches_dial why =
  let d, a = connected () in
  let refused = spawn (fun () -> Wire.refuse a why) in
  let r =
    Fun.protect
      ~finally:(fun () -> Unix.close d)
      (fun () ->
        Wire.dial d ~key ~self:Wire.Controller ~peer:(Wire.Agent 1))
  in
  refused ();
  equal dialed (Error why) r

(* A peer that reset the connection makes every send fail: a send that
   raised SIGPIPE would end the process. *)
let refuse_reset () =
  List.iter
    (fun why ->
      let d, a = connected () in
      Unix.setsockopt_optint d Unix.SO_LINGER (Some 0);
      Unix.close d;
      ignore (readable a 1.);
      Wire.refuse a why)
    [ "too late"; String.make 1_000_000 'x' ]

let refusals =
  group "refuse"
    [
      test "refuse sends the documented refusal, then ends the stream"
        refuse_bytes;
      prop "dial reports a refusal's reason unchanged"
        ~examples:[ "" ]
        (Gen.with_pp pp_bytes Gen.string)
        refuse_reaches_dial;
      test "a reason of 70000 bytes reaches dial as its first 4096" (fun () ->
          let d, a = connected () in
          let refused =
            spawn (fun () -> Wire.refuse a (String.make 70_000 'r'))
          in
          let r =
            Fun.protect
              ~finally:(fun () -> Unix.close d)
              (fun () ->
                Wire.dial d ~key ~self:Wire.Controller ~peer:(Wire.Agent 1))
          in
          refused ();
          equal dialed (Error (String.make 4096 'r')) r);
      test "refuse raises nothing, and the process lives, on a reset peer"
        refuse_reset;
    ]

(* A raw accepting end *)

(* Greets [d]'s dialing end with [hello] and reads its proof. *)
let greet_with hello f =
  let d, a = connected () in
  let peer =
    spawn (fun () ->
        Fun.protect
          ~finally:(fun () -> Unix.close a)
          (fun () ->
            write a hello;
            f a (read_n a proof_size)))
  in
  let r =
    Fun.protect
      ~finally:(fun () -> Unix.close d)
      (fun () -> Wire.dial d ~key ~self:Wire.Controller ~peer:(Wire.Agent 1))
  in
  peer ();
  r

let other_version v =
  let r = greet_with (greeting ~version:v (random_bytes 32)) (fun _ _ -> ()) in
  let why = require_error r in
  contains ~msg:"names this end's version" ~sub:"1" why;
  contains ~msg:"names the other end's version" ~sub:(string_of_int v) why

let malformed_greetings =
  [
    ("another preamble", "rig-jab\n" ^ u32 1 ^ "\000" ^ random_bytes 32);
    ("a status of 2", "rig-job\n" ^ u32 1 ^ "\002" ^ random_bytes 32);
  ]

let greetings =
  group "greeting"
    [
      cases ~name:(Printf.sprintf "dial refuses version %d, naming both")
        "another version" [ 0; 2; 0xffff_ffff ] other_version;
      cases ~name:(fun (n, _) -> "dial refuses " ^ n) "malformed"
        malformed_greetings (fun (_, g) ->
          is_error (greet_with g (fun _ _ -> ())));
    ]

(* Proofs *)

let transcript_layout () =
  let k = random_bytes 32 in
  let o =
    handshake ~self:(Wire.Agent 2) ~peer:(Wire.Agent 5) ~dialing:k ~accepting:k
      ()
  in
  equal accepted (Ok (Wire.Agent 2, Wire.Agent 5)) o.accepted;
  let t = o.transcript in
  equal int transcript_size (String.length t);
  starts_with ~affix:("rig-job\n" ^ u32 1 ^ "\000") t;
  equal ~msg:"the dialing end's processes" string
    (u32 2 ^ u32 5)
    (String.sub t (greeting_size + 32) 8);
  equal ~msg:"the accepting end's status" char '\000'
    t.[greeting_size + proof_size];
  not_contains ~msg:"the key never crosses the connection" ~sub:k t

(* The nonces and proofs of a handshake: the accepting end's nonce, the
   dialing end's, the dialing end's proof and the accepting end's. *)
let parts t =
  ( String.sub t 13 32,
    String.sub t greeting_size 32,
    String.sub t (greeting_size + 40) 32,
    String.sub t (greeting_size + proof_size + 1) 32 )

let fresh_nonces () =
  let o1 = handshake ~dialing:key ~accepting:key () in
  let o2 = handshake ~dialing:key ~accepting:key () in
  let a1, d1, p1, q1 = parts o1.transcript in
  let a2, d2, _, _ = parts o2.transcript in
  not_equal ~msg:"the accepting end's nonces" string a1 a2;
  not_equal ~msg:"the dialing end's nonces" string d1 d2;
  not_equal ~msg:"the two ends' proofs" string p1 q1

(* An accepting end without the key answers with the dialing end's own
   proof. *)
let reflected () =
  let r =
    greet_with
      (greeting ~version:1 (random_bytes 32))
      (fun a proof -> write a ("\000" ^ String.sub proof 40 32))
  in
  is_error r

let replayed_proof () =
  let o = handshake ~dialing:key ~accepting:key () in
  let recorded = String.sub o.transcript greeting_size proof_size in
  let d, a = connected () in
  let admitted = ref 0 in
  let accepted =
    spawn (fun () ->
        Fun.protect
          ~finally:(fun () -> Unix.close a)
          (fun () ->
            Wire.accept a ~key ~admit:(fun _ ->
                incr admitted;
                Ok ())))
  in
  Fun.protect
    ~finally:(fun () -> Unix.close d)
    (fun () ->
      ignore (read_n d greeting_size);
      write d recorded;
      ignore (read_all d));
  is_error (accepted ());
  equal ~msg:"admit is never asked" int 0 !admitted

let replayed_answer () =
  let o = handshake ~dialing:key ~accepting:key () in
  let t = o.transcript in
  let hello = String.sub t 0 greeting_size in
  let answer = String.sub t (greeting_size + proof_size) answer_size in
  is_error (greet_with hello (fun a _ -> write a answer))

(* Every byte of the transcript, changed in transit, fails the end that reads
   it; a change before the answer fails the accepting end too. *)
let changed_bytes () =
  for at = 0 to transcript_size - 1 do
    let msg = Printf.sprintf "byte %d changed" at in
    let o, took =
      timed (fun () -> handshake ~flip:at ~dialing:key ~accepting:key ())
    in
    is_error ~msg o.dialed;
    if at < greeting_size + proof_size then is_error ~msg o.accepted;
    less ~msg float_exact ~than:(limit /. 2.) took
  done

(* A relay that closes both connections after [cut] bytes; the accepting end
   holds the dialing end's whole proof from [greeting_size + proof_size]
   bytes on. *)
let cut_handshakes () =
  for cut = 0 to transcript_size - 1 do
    let msg = Printf.sprintf "closed after %d bytes" cut in
    let o, took =
      timed (fun () -> handshake ~cut ~dialing:key ~accepting:key ())
    in
    is_error ~msg o.dialed;
    if cut < greeting_size + proof_size then is_error ~msg o.accepted;
    less ~msg float_exact ~than:(limit /. 2.) took
  done

let byte_by_byte () =
  let o = handshake ~pace:0.001 ~dialing:key ~accepting:key () in
  equal dialed (Ok ()) o.dialed;
  equal accepted (Ok (Wire.Controller, Wire.Agent 1)) o.accepted

let proofs =
  group "proof"
    [
      test "a handshake's bytes follow the documented layout" transcript_layout;
      test "each handshake draws fresh nonces, and the two proofs differ"
        fresh_nonces;
      test "an accepting end that echoes the dialing end's proof is refused"
        reflected;
      test "a dialing end's recorded proof does not prove the key again"
        replayed_proof;
      test "an accepting end's recorded answers do not prove the key again"
        replayed_answer;
      test "a changed byte at any offset fails the ends that read it"
        changed_bytes;
      test "a peer that closes after any prefix fails the ends promptly"
        cut_handshakes;
      test "ends that receive one byte at a time connect" byte_by_byte;
    ]

(* Silence

   A raw peer writes only within its first [silent_after] seconds and then
   reads until the end closes, or until [limit +. slack], when it closes its
   own socket: no peer writes to an end that has given up, and an end that
   waits too long returns at that deadline. The cases of a test run at once,
   so it takes one [limit]. *)

let silent_after = 9.5

(* Writes [s] a byte at a time, one every half second, until [silent_after]. *)
let trickle fd s =
  let t0 = Unix.gettimeofday () in
  let rec go i =
    let quiet = Unix.gettimeofday () -. t0 >= silent_after in
    if i < String.length s && not quiet then begin
      write fd (String.make 1 s.[i]);
      Thread.delay 0.5;
      go (i + 1)
    end
  in
  go 0

(* Reads [fd] until its stream ends or [deadline]. *)
let drain fd deadline =
  let b = Bytes.create 4096 in
  let rec go () =
    let left = deadline -. Unix.gettimeofday () in
    if left > 0. && readable fd left then
      match Unix.read fd b 0 (Bytes.length b) with
      | 0 | (exception Unix.Unix_error _) -> ()
      | _ -> go ()
  in
  go ()

type side = Dialing | Accepting

(* The end on [side] against a raw peer that does [behave]: whether it failed,
   and the time it took. *)
let against side behave =
  let d, a = connected () in
  let mine, theirs = match side with Dialing -> (d, a) | Accepting -> (a, d) in
  let deadline = Unix.gettimeofday () +. limit +. slack in
  let peer =
    spawn (fun () ->
        Fun.protect
          ~finally:(fun () -> Unix.close theirs)
          (fun () ->
            behave theirs;
            drain theirs deadline))
  in
  let r, took =
    timed (fun () ->
        Fun.protect
          ~finally:(fun () -> Unix.close mine)
          (fun () ->
            match side with
            | Dialing ->
                Wire.dial mine ~key ~self:Wire.Controller ~peer:(Wire.Agent 1)
            | Accepting ->
                Result.map ignore
                  (Wire.accept mine ~key ~admit:(fun _ -> Ok ()))))
  in
  peer ();
  (Result.is_error r, took)

let gives_up cases () =
  let running =
    List.map
      (fun (name, side, behave) ->
        (name, spawn (fun () -> against side behave)))
      cases
  in
  let results = List.map (fun (name, join) -> (name, join ())) running in
  List.iter
    (fun (name, (failed, took)) ->
      equal ~msg:name bool true failed;
      at_least ~msg:name float_exact ~than:(limit -. 0.05) took;
      less ~msg:name float_exact ~than:(limit +. slack) took)
    results

let hello = greeting ~version:1 (String.make 32 'n')

let silent =
  [
    ("dial, a silent accepting end", Dialing, ignore);
    ( "dial, an accepting end silent after the greeting",
      Dialing,
      fun a ->
        write a hello;
        ignore (read_n a proof_size) );
    ("accept, a silent dialing end", Accepting, ignore);
  ]

let trickling =
  [
    ( "dial, an accepting end trickling its greeting",
      Dialing,
      fun a -> trickle a hello );
    ( "accept, a dialing end trickling its proof",
      Accepting,
      fun d ->
        ignore (read_n d greeting_size);
        trickle d (String.make proof_size 'p') );
  ]

let silence =
  group "silence"
    [
      slow "an end gives up on a silent peer after 10 s" (gives_up silent);
      slow "an end gives up after 10 s on a peer that trickles its answer"
        (gives_up trickling);
    ]

let () =
  exit
    (run "rig_remote_proxy.wire"
       [
         group ~timeout:60. "handshake"
           [ keys_group; admission; refusals; greetings; proofs; silence ];
       ])
