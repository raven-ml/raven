(* §7.1: the cost of releasing and reacquiring the runtime lock around a call
   that does nothing, in ns per call: [blocking] minus [plain].

   Usage: lock_cost.exe MODE. In mode [alone] the process has one domain and
   never spawned another, so domain 0 has no backup thread. In [joined] a second
   domain was spawned and joined before timing, as in a program that once used
   domains. In [spin] a second domain runs a loop that does not allocate; in
   [alloc] it allocates short-lived lists, so minor collections stop both
   domains. Each mode runs in its own process: a spawned domain leaves the
   backup thread of domain 0 running for the rest of the process. *)

type bigbytes =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external now : unit -> int = "lzm_now" [@@noalloc]
external plain : bigbytes -> unit = "lzm_plain"
external blocking : bigbytes -> unit = "lzm_blocking"

let calls = 1_000_000
let rounds = 15

(* ns per call of [f] over [calls] calls. *)
let per_call f arg =
  let t0 = now () in
  for _ = 1 to calls do
    f arg
  done;
  float (now () - t0) /. float calls

let sorted a =
  let a = Array.copy a in
  Array.sort compare a;
  a

let median a = (sorted a).(Array.length a / 2)
let best a = (sorted a).(0)

let second_domain mode stop =
  match mode with
  | "spin" ->
      Some
        (Domain.spawn (fun () ->
             let n = ref 0 in
             while not (Atomic.get stop) do
               incr n
             done;
             ignore (Sys.opaque_identity !n)))
  | "alloc" ->
      Some
        (Domain.spawn (fun () ->
             let l = ref [] in
             while not (Atomic.get stop) do
               l := [ 1; 2; 3; 4 ];
               ignore (Sys.opaque_identity !l)
             done))
  | "joined" ->
      Domain.join (Domain.spawn (fun () -> ()));
      None
  | "alone" -> None
  | m -> invalid_arg ("lock_cost: unknown mode " ^ m)

let () =
  let mode = if Array.length Sys.argv > 1 then Sys.argv.(1) else "alone" in
  let stop = Atomic.make false in
  let other = second_domain mode stop in
  let arg = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 16 in
  (* Warm both paths, then alternate them so drift hits both alike. *)
  ignore (per_call plain arg);
  ignore (per_call blocking arg);
  let p = Array.make rounds 0. and b = Array.make rounds 0. in
  for r = 0 to rounds - 1 do
    p.(r) <- per_call plain arg;
    b.(r) <- per_call blocking arg
  done;
  Atomic.set stop true;
  Option.iter Domain.join other;
  let d = Array.init rounds (fun r -> b.(r) -. p.(r)) in
  Printf.printf
    "%-6s plain %6.1f ns (best %6.1f)  blocking %7.1f ns (best %7.1f)  release \
     %7.1f ns (median of %d rounds of %d calls; best %7.1f)\n"
    mode (median p) (best p) (median b) (best b) (median d) rounds calls
    (best d)
