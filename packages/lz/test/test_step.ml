(* Lz.Decoder.step and Lz.Encoder.step over every streaming format, against the
   vendors' goldens and their mutated and cut copies, in lz.mli's order: a
   decoder is dead after an error or a raise in [f], is busy while a step runs,
   lends runs that outlive it, ends a stream as its doc states, and steps equal
   the whole-buffer decode for any cut of the input. An encoder shares the busy
   and dead rules, and its [f] may overwrite its run. *)

open Windtrap

let kib = 1024

(* Inputs *)

(* A stream and what it decodes to, each made when forced. *)
type input = { file : string; raw : string Lazy.t; data : string Lazy.t }

let data i = Lazy.force i.data
let raw i = Lazy.force i.raw

let of_golden (g : Lz_support.golden) =
  { file = g.file; raw = g.raw; data = g.data }

let golden name =
  of_golden
    (List.find (fun g -> g.Lz_support.file = name) (Lz_support.goldens ()))

(* [long_stream name] is the decoded bytes and a stream, in the format's wrapper
   (RFC 1950, RFC 1952), of 40,000 bytes of text in a stored block (RFC 1951,
   3.2.4), then a block of fixed codes (3.2.6) of 1100 matches of 258 bytes at
   32 KiB, whose sources follow each other: 323,800 bytes from about 44 KB, for
   the deflate family, whose goldens decode to at most 200,000 bytes, short of a
   run. Its output passes the decoder's buffer, a 32 KiB window and a 256 KiB
   run (lz.mli, Decoder), so every walk over it wraps the buffer, and a match
   258 bytes long reads across the wrap unless the wrap falls on one of the 1100
   boundaries between them. *)
let long_stream name =
  let text = Lz_support.corpus Text 40_000 in
  let out = Buffer.create 330_000 and b = Buffer.create 48_000 in
  Buffer.add_string out text;
  (match name with
  | "zlib" -> Buffer.add_string b "\x78\x01"
  | "gzip" -> Buffer.add_string b "\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"
  | _ -> ());
  let n = String.length text in
  Buffer.add_char b '\x00';
  Buffer.add_uint16_le b n;
  Buffer.add_uint16_le b (lnot n land 0xffff);
  Buffer.add_string b text;
  (* Bits go out from each byte's lowest; a code goes out from its highest bit,
     extra bits from their lowest. *)
  let acc = ref 0 and held = ref 0 in
  let put v k =
    acc := !acc lor (v lsl !held);
    held := !held + k;
    while !held >= 8 do
      Buffer.add_char b (Char.chr (!acc land 0xff));
      acc := !acc lsr 8;
      held := !held - 8
    done
  in
  let code c k =
    for i = k - 1 downto 0 do
      put ((c lsr i) land 1) 1
    done
  in
  put 1 1;
  put 1 2;
  for _ = 1 to 1100 do
    (* Length 258 is symbol 285, code 0xc5 of 8 bits; distance 32768 is code 29
       of 5 bits with 13 extra bits, 32768 - 24577. *)
    code 0xc5 8;
    code 29 5;
    put 8191 13;
    let at = Buffer.length out - 32768 in
    for k = 0 to 257 do
      Buffer.add_char out (Buffer.nth out (at + k))
    done
  done;
  code 0 7;
  if !held > 0 then put 0 (8 - !held);
  let raw = Buffer.contents out in
  (match name with
  | "zlib" -> Buffer.add_int32_be b (Lz.Adler32.string raw)
  | "gzip" ->
      Buffer.add_int32_le b (Int32.of_int (Lz.Crc32.string raw));
      Buffer.add_int32_le b (Int32.of_int (String.length raw))
  | _ -> ());
  (raw, Buffer.contents b)

let long name =
  let s = lazy (long_stream name) in
  {
    file = "long." ^ name;
    raw = lazy (fst (Lazy.force s));
    data = lazy (snd (Lazy.force s));
  }

(* [concat file names] is the goldens [names] one after another. *)
let concat file names =
  let gs = List.map golden names in
  {
    file;
    raw = lazy (String.concat "" (List.map raw gs));
    data = lazy (String.concat "" (List.map data gs));
  }

(* Formats *)

type format = {
  name : string;
  make : unit -> Lz.Decoder.t;
  members : bool; (* Read to the input's end, as gzip and Zstandard. *)
  long : input; (* A stream whose output passes a run. *)
}

(* A mutated Zstandard header may declare any window up to the decoder's bound,
   which the decoder allocates; 4 MiB holds every golden's. *)
let zstd_window = 4 * 1024 * kib

let formats =
  [
    {
      name = "deflate";
      make = Lz.Deflate.decoder;
      members = false;
      long = long "deflate";
    };
    {
      name = "zlib";
      make = Lz.Zlib.decoder;
      members = false;
      long = long "zlib";
    };
    {
      name = "gzip";
      make = Lz.Gzip.decoder;
      members = true;
      long = long "gzip";
    };
    {
      name = "zstd";
      make = (fun () -> Lz.Zstd.decoder ~window:zstd_window ());
      members = true;
      (* Four goldens' frames, 800,000 bytes: its goldens decode to at most
         200,000 bytes, which a run holds whole. *)
      long =
        concat "long.zst"
          [
            "text_200000.zst19";
            "text_200000.zst1";
            "text_200000.zst3";
            "window1m.zst";
          ];
    };
  ]

let empty = Lz_support.of_string ""
let stepped = result Lz_support.step Lz_support.error
let invalid e = Exn.invalid_arg e
let step d ~last s f = Lz.Decoder.step d ~last (Lz_support.of_string s) f

(* A run as its length and digest: runs reach 256 KiB. *)
let run_w =
  Testable.make ~equal:String.equal ~pp:(fun ppf s ->
      Format.fprintf ppf "%d bytes, digest %s" (String.length s)
        (Digest.to_hex (Digest.string s)))

(* The run capacity of every decoder (lz.mli, Decoder), and whether a run is
   full ({1:steps}): it holds the capacity, or for Zstandard it holds bytes,
   since its next block might pass the capacity or the end of the decoder's
   buffer. *)
let capacity = 256 * kib

let full fmt run =
  if fmt.name = "zstd" then run <> "" else String.length run = capacity

(* Walking an input in steps *)

(* Where the input's end is said. [Within]: on the slice that reaches it, as a
   reader that knows its size. [After]: every slice has ~last:false, then empty
   steps have ~last:true, as a reader whose read returned nothing. *)
type ends = Within | After
type failure = { error : Lz.error; at_step : int; last : bool }

type walk = {
  out : string; (* The runs, concatenated. *)
  ended : (int, failure) result; (* The bytes taken up to [Done]. *)
  ok_steps : int; (* The steps that returned [Ok]. *)
  short : bool; (* A step returned [More k] short of its slice. *)
  idle : bool; (* A step was given no bytes before the input's end. *)
  trace : (int * int * int) list;
      (* Each step that returned [Ok], in order: the first byte it was given,
         the bytes it took and its run's length. *)
}

(* Output past this means a decoder that never ends. *)
let out_cap = 64 * 1024 * kib

(* [walk fmt d data ~sizes ~ends] steps [d] over [data], step [k] given at most
   [sizes.(k mod n)] bytes from the first byte not yet taken, and checks each
   step against Decoder.step's doc as it goes. [lend k run] is applied to each
   run inside [f]. *)
let walk ?(lend = fun _ _ -> ()) fmt d data ~sizes ~ends =
  let src = Lz_support.of_string data and n = String.length data in
  let sizes = Array.of_list sizes in
  let out = Buffer.create n in
  let short = ref false and idle = ref false and trace = ref [] in
  let rec go pos k =
    let left = n - pos in
    let len = min sizes.(k mod Array.length sizes) left in
    let last = match ends with Within -> len = left | After -> left = 0 in
    if len = 0 && left > 0 then idle := true;
    let broke what =
      failf "step %d, bytes %d to %d, ~last:%b: %s" k pos (pos + len) last what
    in
    let runs = ref [] in
    let f run =
      lend k run;
      runs := Lz_support.to_string run :: !runs
    in
    let finish ended =
      {
        out = Buffer.contents out;
        ended;
        ok_steps = k;
        short = !short;
        idle = !idle;
        trace = List.rev !trace;
      }
    in
    match Lz.Decoder.step d ~last (Bigarray.Array1.sub src pos len) f with
    | Error e ->
        if !runs <> [] then broke "f applied on Error";
        if e.why = Too_long then broke "Too_long from a step";
        if e.why = Truncated && not last then broke "Truncated without ~last";
        (* An error kills the decoder. *)
        raises_match ~msg:"an empty step after Error" invalid (fun () ->
            Lz.Decoder.step d ~last:true empty ignore);
        raises_match ~msg:"a step on the input after Error" invalid (fun () ->
            Lz.Decoder.step d ~last:false src ignore);
        finish (Error { error = e; at_step = k; last })
    | Ok s -> (
        let run =
          match !runs with
          | [ r ] -> r
          | rs -> broke (Printf.sprintf "f applied %d times" (List.length rs))
        in
        Buffer.add_string out run;
        if Buffer.length out > out_cap then broke "output without end";
        if String.length run > capacity then
          broke (Printf.sprintf "a run of %d bytes" (String.length run));
        let j = match s with Done j | More j -> j in
        if j < 0 || j > len then broke (Printf.sprintf "%d bytes taken" j);
        trace := (pos, j, String.length run) :: !trace;
        match s with
        | Done j ->
            if fmt.members && not last then broke "Done with ~last:false";
            if fmt.members && j <> len then
              broke (Printf.sprintf "Done %d, short of the slice" j);
            { (finish (Ok (pos + j))) with ok_steps = k + 1 }
        | More j ->
            if j < len then short := true;
            (* A step short of its slice, or one with ~last:true, stopped
               because its run was full. *)
            if (j < len || last) && not (full fmt run) then
              broke
                (Printf.sprintf "More %d with a run of %d bytes, not full" j
                   (String.length run));
            go (pos + j) (k + 1))
  in
  go 0 0

let whole fmt data room =
  let dst = Lz_support.create room in
  match Lz_support.decoder fmt.name (Lz_support.of_string data) dst with
  | Ok { Lz.consumed; produced } ->
      Ok (Lz_support.to_string ~length:produced dst, consumed)
  | Error e -> Error e

(* A failing step may hold one run it did not lend, at most 256 KiB: the whole
   decode gets more room than that past what the steps lent. *)
let room = 512 * kib

(* Drawn cases *)

type change =
  | Same
  | Flip of int * int (* Byte, from the end when negative; bit. *)
  | Replace of int * char (* Byte, as [Flip]'s; its new value. *)
  | Cut of int (* The first bytes kept. *)
  | Append of string
  | Then of input (* Another golden of the format after it. *)

type case = { input : input; change : change; sizes : int list; ends : ends }

let pp_case ppf { input; change; sizes; ends } =
  let ints =
    Format.(
      pp_print_list ~pp_sep:(fun p () -> pp_print_string p "; ") pp_print_int)
  in
  Format.fprintf ppf "{ input = %s; change = %s; sizes = [%a]; ends = %s }"
    input.file
    (match change with
    | Same -> "Same"
    | Flip (i, b) -> Printf.sprintf "Flip (%d, %d)" i b
    | Replace (i, c) -> Printf.sprintf "Replace (%d, %C)" i c
    | Cut i -> Printf.sprintf "Cut %d" i
    | Append s -> Printf.sprintf "Append %S" s
    | Then g -> "Then " ^ g.file)
    ints sizes
    (match ends with Within -> "Within" | After -> "After")

let apply { input; change; _ } =
  let d = data input in
  let at i = Lz_support.position i (String.length d) in
  let set i c = String.mapi (fun j x -> if j = i then c else x) d in
  match change with
  | Same -> d
  | Flip (i, b) -> set (at i) (Char.chr (Char.code d.[at i] lxor (1 lsl b)))
  | Replace (i, c) -> set (at i) c
  | Cut i -> String.sub d 0 i
  | Append s -> d ^ s
  | Then g -> d ^ data g

(* Whether every byte is a prefix of a valid input: no step with ~last:false may
   then fail. *)
let valid_prefix fmt = function
  | Same | Cut _ | Then _ -> true
  | Append _ -> not fmt.members
  | Flip _ | Replace _ -> false

(* Slice sizes, 0 among them: a cycle of up to four, one of them above 0. *)
let sizes =
  let open Gen in
  let size =
    frequency
      [
        (1, constant 0);
        (2, int_range 1 16);
        (2, int_range 17 4096);
        (2, int_range 4097 300_000);
      ]
  in
  map
    (fun l -> if List.for_all (( = ) 0) l then 1 :: l else l)
    (list ~size:(int_range 1 4) size)

let ends = Gen.(map (fun b -> if b then Within else After) bool)

(* [spread n sizes] adds [n / most_steps] bytes to each slice above 0, so that a
   walk over [n] bytes takes at most [most_steps] slices that take bytes, and
   drawn sizes stay distinct. Slices are unchanged on inputs below 1000 bytes,
   two goldens in three; over the longest inputs, slices of a few bytes took
   thousands of steps a case and outran the sanitizers' time. *)
let most_steps = 1000

let spread n sizes =
  List.map (fun s -> if s = 0 then 0 else s + (n / most_steps)) sizes

(* The format's goldens, the largest streams, above 32 KiB, drawn as often
   again, and its long stream as often as those: a run fills only under a large
   slice of one. *)
let inputs fmt =
  let all = Lz_support.goldens_of fmt.name in
  let large = List.filter (fun g -> g.Lz_support.size > 32 * kib) all in
  Gen.(
    frequency
      [
        (3, of_list (List.map of_golden all));
        (1, of_list (List.map of_golden large));
        (1, constant fmt.long);
      ])

let case fmt =
  let open Gen in
  let others =
    Array.of_list (List.map of_golden (Lz_support.goldens_of fmt.name))
  in
  let change =
    frequency
      [
        (2, constant `Same);
        ( 2,
          map
            (fun (i, b) -> `Flip (i, b))
            (pair Lz_support.anywhere (int_range 0 7)) );
        (1, map (fun (i, c) -> `Replace (i, c)) (pair Lz_support.anywhere char));
        (2, map (fun i -> `Cut i) Lz_support.anywhere);
        (1, map (fun s -> `Append s) (string_of ~size:(int_range 1 16) char));
        (1, map (fun i -> `Then i) (int_range 0 (Array.length others - 1)));
      ]
  in
  let make (input, change, sizes, ends) =
    let n = String.length (data input) in
    let change =
      match change with
      | `Same -> Same
      | `Flip (i, b) -> Flip (Lz_support.position i n, b)
      | `Replace (i, c) -> Replace (Lz_support.position i n, c)
      | `Cut i -> Cut (Lz_support.position i n)
      | `Append s -> Append s
      | `Then i -> Then others.(i)
    in
    let c = { input; change; sizes; ends } in
    { c with sizes = spread (String.length (apply c)) sizes }
  in
  with_pp pp_case (map make (quad (inputs fmt) change sizes ends))

(* A small stream of each format. *)
let small fmt =
  golden
    (match fmt.name with
    | "deflate" -> "text_100.ld6.deflate"
    | "zlib" -> "text_100.zlib6"
    | "gzip" -> "text_100.gz9"
    | _ -> "text_100.zst3")

(* Dead and busy decoders *)

let after_error =
  cases "a step after Error raises Invalid_argument"
    ~id:(fun fmt -> fmt.name)
    formats
    (fun fmt ->
      let d = fmt.make () in
      (* No stream of these formats begins with 0xff 0xff: deflate's block type
         3, zlib's method 15, gzip's and Zstandard's magic numbers. *)
      let r = step d ~last:true "\xff\xff\xff\xff" ignore in
      is_error ~pp:(Testable.pp Lz_support.step) r;
      raises_match invalid (fun () ->
          step d ~last:true (data (small fmt)) ignore))

let reentry =
  cases "a step from f raises Invalid_argument, which kills the decoder"
    ~id:(fun fmt -> fmt.name)
    formats
    (fun fmt ->
      let d = fmt.make () in
      raises_match ~msg:"a step from f" invalid (fun () ->
          step d ~last:true
            (data (small fmt))
            (fun _ -> ignore (Lz.Decoder.step d ~last:true empty ignore)));
      raises_match ~msg:"a step after" invalid (fun () ->
          step d ~last:true (data (small fmt)) ignore))

(* [held step refuse] runs [step hold] on another domain, whose [f] calls
   [hold], which blocks on a semaphore until [refuse ()] has returned or raised
   on this domain: [f] may block ({1:steps}). It is the held step's result. *)
let held step refuse =
  let entered = Semaphore.Binary.make false in
  let gate = Semaphore.Binary.make false in
  let in_f = Atomic.make false in
  let hold () =
    Atomic.set in_f true;
    Semaphore.Binary.release entered;
    Semaphore.Binary.acquire gate
  in
  let holder =
    Domain.spawn (fun () ->
        Fun.protect
          ~finally:(fun () -> Semaphore.Binary.release entered)
          (fun () -> step hold))
  in
  Semaphore.Binary.acquire entered;
  if not (Atomic.get in_f) then (
    ignore (Domain.join holder);
    fail "the held step returned without applying f");
  Fun.protect ~finally:(fun () -> Semaphore.Binary.release gate) refuse;
  Domain.join holder

(* The refused step changes nothing: the held step and a later one decode their
   streams. *)
let other_domain =
  cases
    "a step from another domain while one runs raises Invalid_argument and \
     changes nothing"
    ~id:(fun fmt -> fmt.name)
    formats
    (fun fmt ->
      let g = small fmt in
      let d = fmt.make () in
      let out = Buffer.create 128 in
      let r =
        held
          (fun hold ->
            step d ~last:true (data g) (fun run ->
                Buffer.add_string out (Lz_support.to_string run);
                hold ()))
          (fun () ->
            raises_match ~msg:"a step while another runs" invalid (fun () ->
                step d ~last:true (data g) ignore))
      in
      equal stepped ~msg:"the held step" (Ok (Done (String.length (data g)))) r;
      equal Lz_support.decoded (Ok (raw g)) (Ok (Buffer.contents out));
      let t = walk fmt d (data g) ~sizes:[ max_int ] ~ends:Within in
      equal Lz_support.decoded ~msg:"a step after the refused one"
        (Ok (raw g))
        (match t.ended with Ok _ -> Ok t.out | Error f -> Error f.error))

(* The inputs the shared decoder is stepped on: thirds of a stream, the whole
   stream, nothing, and bytes that begin no stream. *)
type piece = Head | Middle | Tail | Whole | Nothing | Junk

let pieces = [ Head; Middle; Tail; Whole; Nothing; Junk ]

let pp_piece ppf p =
  Format.pp_print_string ppf
    (match p with
    | Head -> "Head"
    | Middle -> "Middle"
    | Tail -> "Tail"
    | Whole -> "Whole"
    | Nothing -> "Nothing"
    | Junk -> "Junk")

let piece_bytes fmt =
  let s = data (small fmt) in
  let n = String.length s in
  let third = n / 3 in
  function
  | Head -> String.sub s 0 third
  | Middle -> String.sub s third third
  | Tail -> String.sub s (2 * third) (n - (2 * third))
  | Whole -> s
  | Nothing -> ""
  | Junk -> "\xff\x00\xff\x00"

(* What a step does: its result and the runs it lent. *)
type outcome = (Lz.step, Lz.error) result * string list

let outcome_w : outcome testable = pair stepped (list run_w)

(* The reference: the steps a decoder has taken, in order, or dead. *)
type model = { mutable steps : (bool * piece) list; mutable dead : bool }

let pp_model ppf m =
  if m.dead then Format.pp_print_string ppf "dead"
  else Format.fprintf ppf "%d steps" (List.length m.steps)

(* The outcome of [last, p] after [steps], by a fresh decoder taking them in
   order on one domain, memoised: the judge replays every order the domains
   could have taken. *)
let predict =
  let memo = Hashtbl.create 1024 in
  fun fmt bytes steps last p ->
    let key = (fmt.name, steps, last, p) in
    match Hashtbl.find_opt memo key with
    | Some o -> o
    | None ->
        let d = fmt.make () in
        List.iter
          (fun (last, p) ->
            match Lz.Decoder.step d ~last (bytes p) ignore with
            | Ok _ -> ()
            | Error e -> failf "the reference's replay failed: %a" Lz.pp_error e)
          (List.rev steps);
        let runs = ref [] in
        let r =
          Lz.Decoder.step d ~last (bytes p) (fun run ->
              runs := Lz_support.to_string run :: !runs)
        in
        let o = (r, !runs) in
        Hashtbl.replace memo key o;
        o

(* A refusal for being in use, told from one for being dead by its message:
   lz.mli promises only Invalid_argument for both, and under interleaving the
   message is the judge's only way to tell a busy refusal from a death. *)
let in_use e = Exn.invalid_arg ~sub:"in use" e

(* A step on the shared decoder; [raising] makes [f] raise Exit. *)
let shared_step bytes ~raising last p d : outcome =
  let runs = ref [] in
  let r =
    Lz.Decoder.step d ~last (bytes p) (fun run ->
        runs := Lz_support.to_string run :: !runs;
        if raising then raise Exit)
  in
  (r, !runs)

(* The judge: a step another domain's step refused for being in use changes
   nothing; on a dead decoder a step raises Invalid_argument; otherwise the step
   is the one the steps taken so far predict, and an error or a raise in [f]
   kills the decoder. *)
let judge fmt bytes ~raising last p m (got : (outcome, exn) result) =
  match got with
  | Error e when in_use e -> ()
  | Error e when m.dead && invalid e -> ()
  | _ when m.dead ->
      fail "a step on a dead decoder did not raise Invalid_argument"
  | _ -> (
      let ((r, _) as expected) = predict fmt bytes m.steps last p in
      match (got, r) with
      | Error Exit, Ok _ when raising -> m.dead <- true
      | Error e, _ -> raise e
      | Ok o, _ ->
          equal outcome_w expected o;
          if raising || Result.is_error r then m.dead <- true
          else m.steps <- (last, p) :: m.steps)

let one_user fmt =
  (* Under 1 KiB in all, read as the test is declared. *)
  let bytes =
    let table =
      List.map (fun p -> (p, Lz_support.of_string (piece_bytes fmt p))) pieces
    in
    fun p -> List.assoc p table
  in
  let dec = abstract ~pp:pp_model "d" in
  let piece = Gen.of_list ~pp:pp_piece pieces in
  let stepping name ~raising =
    command name
      (Gen.bool @-> piece @-> dec ^-> judges outcome_w)
      (judge fmt bytes ~raising)
      (shared_step bytes ~raising)
  in
  (* Every step of a program goes to one decoder from the two domains: the
     prefix makes the decoder alone (~steps:1) and the branches step it. Steps
     on one domain and spare decoders are the laws below's, and cost the most
     under the sanitizers' collector, as each decoder's first step allocates its
     buffer. A case runs its program 50 times; 50 cases make some 12,000 steps,
     of which 3,000 to 6,000 are refused. *)
  stateful ~domains:2 ~steps:1 ~count:50
    "a decoder stepped from two domains acts as some order of its steps"
    [
      command "decoder"
        (Gen.unit @-> makes dec)
        (fun () -> { steps = []; dead = false })
        fmt.make;
      (* Listed twice: drawn twice as often as a raise in [f]. *)
      stepping "step" ~raising:false;
      stepping "step" ~raising:false;
      stepping "step with f raising" ~raising:true;
    ]

let raising fmt =
  let gen =
    Gen.(
      with_pp
        (fun ppf (input, sizes, r) ->
          Format.fprintf ppf "%a, f raising at step %d" pp_case
            { input; change = Same; sizes; ends = Within }
            r)
        (map
           (fun (input, sizes, r) ->
             (input, spread (String.length (data input)) sizes, r))
           (triple (inputs fmt) sizes nat)))
  in
  prop "an exception from f propagates and kills the decoder" gen
    (fun (input, sizes, r) ->
      let input = data input in
      let clean = walk fmt (fmt.make ()) input ~sizes ~ends:Within in
      let r = r mod clean.ok_steps in
      cover "f raising after the first step" (r > 0);
      let d = fmt.make () in
      raises ~msg:"the exception of f" Exit (fun () ->
          walk
            ~lend:(fun k _ -> if k = r then raise Exit)
            fmt d input ~sizes ~ends:Within);
      raises_match ~msg:"a step after f raised" invalid (fun () ->
          step d ~last:false input ignore);
      raises_match ~msg:"an empty step after f raised" invalid (fun () ->
          Lz.Decoder.step d ~last:true empty ignore))

(* Lent runs *)

(* Steps a fresh decoder over [input], keeping every run past [f] with a copy
   taken in [f], newest first, and leaves the decoder only in [gone]. *)
let[@inline never] keep_runs fmt input ~sizes ~ends gone =
  let d = fmt.make () in
  Weak.set gone 0 (Some d);
  let kept = ref [] in
  let lend _ run = kept := (run, Lz_support.to_string run) :: !kept in
  ignore (walk ~lend fmt d input ~sizes ~ends);
  !kept

(* A stream whose decoded bytes differ from [input]'s: noise for an input of
   text or columns, the long stream of text for one of noise. *)
let other fmt input =
  if String.starts_with ~prefix:"random" input.file then fmt.long
  else
    golden
      (match fmt.name with
      | "deflate" -> "random_200000.ld6.deflate"
      | "zlib" -> "random_200000.zlib6"
      | "gzip" -> "random_200000.gz9"
      | _ -> "random_200000.zst3")

(* Once the decoder is collected and another has decoded other bytes, taking any
   memory a freed buffer gave back, every kept run reads, which the sanitizers
   check, and the last run that held bytes holds them still: no later step of
   its decoder wrote output over it. *)
let kept_runs fmt =
  let gen =
    Gen.(
      with_pp
        (fun ppf (input, sizes, ends) ->
          pp_case ppf { input; change = Same; sizes; ends })
        (map
           (fun (input, sizes, ends) ->
             (input, spread (String.length (data input)) sizes, ends))
           (triple (inputs fmt) sizes ends)))
  in
  prop "a run kept past f keeps its buffer alive" gen
    (fun (input, sizes, ends) ->
      let gone = Weak.create 1 in
      let kept = keep_runs fmt (data input) ~sizes ~ends gone in
      Gc.full_major ();
      is_none ~msg:"the decoder was collected" (Weak.get gone 0);
      ignore
        (walk fmt (fmt.make ())
           (data (other fmt input))
           ~sizes:[ max_int ] ~ends:Within);
      List.iter
        (fun (run, _) ->
          ignore (Sys.opaque_identity (Lz_support.to_string run)))
        kept;
      (match List.find_opt (fun (_, copy) -> copy <> "") kept with
      | Some (run, copy) ->
          equal run_w ~msg:"the last run that held bytes" copy
            (Lz_support.to_string run)
      | None -> ());
      cover "a run that held bytes" (List.exists (fun (_, c) -> c <> "") kept);
      cover "several runs" (List.length kept > 2))

(* Done and More *)

let after_stream = "\x00after"

(* Each golden in one slice: a deflate or zlib stream is Done at its last byte
   with bytes after, with and without ~last; a gzip or Zstandard stream is More
   on the step that takes its last byte with ~last:false, then Done on an empty
   step with ~last:true. *)
let ends_of fmt =
  cases "after Done, a step begins the next stream"
    ~id:(fun g -> g.Lz_support.file)
    (Lz_support.goldens_of fmt.name)
    (fun g ->
      let g = of_golden g in
      let n = String.length (data g) in
      let expect ~msg t =
        equal Lz_support.decoded ~msg
          (Ok (raw g))
          (match t.ended with Ok _ -> Ok t.out | Error f -> Error f.error);
        equal (result int pass) ~msg:(msg ^ ": bytes taken") (Ok n)
          (Result.map_error ignore t.ended)
      in
      List.iter
        (fun ends ->
          let d = fmt.make () in
          let input = if fmt.members then data g else data g ^ after_stream in
          let name =
            match ends with Within -> "~last:true" | After -> "~last:false"
          in
          expect ~msg:("first stream, " ^ name)
            (walk fmt d input ~sizes:[ max_int ] ~ends);
          expect ~msg:("next stream, " ^ name)
            (walk fmt d input ~sizes:[ max_int ] ~ends))
        [ Within; After ])

let no_input fmt =
  test "no input with ~last:true is Truncated at 0" (fun () ->
      equal stepped
        (Error { Lz.at = 0; why = Truncated })
        (Lz.Decoder.step (fmt.make ()) ~last:true empty ignore))

(* Steps equal the whole-buffer decode, and on the way Truncated, errors and a
   stream's end come as a step's doc states *)

(* A gzip member (RFC 1952) whose block of fixed codes (RFC 1951, 3.2.6) is a
   match of 3 at distance 1, length code 257 and distance code 0, then the end
   of block, with a CRC-32 and a length of 0. *)
let far_member =
  {
    file = "far.gz";
    raw = lazy "";
    data =
      lazy
        ("\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\x03\x02\x00"
       ^ String.make 8 '\x00');
  }

(* A Zstandard frame (RFC 8878, 3.1.1) with a 1 KiB window and one compressed
   block: the raw literals "abcd", then one sequence in RLE_Mode of 4 literals
   and Offset_Value 32 (code 5), offset 29, after 4 bytes. *)
let far_frame =
  {
    file = "far.zst";
    raw = lazy "";
    data =
      lazy
        "\x28\xB5\x2F\xFD\x00\x00\x5d\x00\x00\x20abcd\x01\x54\x04\x05\x00\x20";
  }

(* A case for each cover of [whole_equals_steps] that drawn cases reach less
   than one time in five: a run filled before its slice ends, a second stream,
   bytes after the stream, an empty slice, a failure after a step that took
   bytes, a malformed first byte; and the counterexamples the law found. *)
let examples fmt =
  let ext =
    match fmt.name with
    | "deflate" -> "deflate6"
    | "zlib" -> "zlib6"
    | "gzip" -> "gz9"
    | _ -> "zst3"
  in
  let text = golden ("text_100." ^ ext) in
  let at change sizes ends = { input = text; change; sizes; ends } in
  [
    { input = fmt.long; change = Same; sizes = [ 1 lsl 20 ]; ends = Within };
    (* Found by the law: a Zstandard step whose run began inside a block ended
       short of its slice at the next block, near its buffer's end, with a run
       of 68,928 bytes. *)
    { input = fmt.long; change = Same; sizes = [ 270 ]; ends = After };
    at (Then (golden ("columns_100." ^ ext))) [ 7 ] Within;
    at (Append "tail") [ 7 ] Within;
    at Same [ 0; 7 ] After;
    at (Cut 40) [ 7 ] Within;
    at (Replace (0, '\x07')) [ 7 ] Within;
    at (Flip (-1, 0)) [ 7 ] Within;
  ]
  @
  (* Each member begins with an empty window: a match into the member before is
     Malformed. In one slice, since a step without ~last may refuse it. *)
  match fmt.name with
  | "gzip" -> [ at (Then far_member) [ max_int ] Within ]
  | "zstd" -> [ at (Then far_frame) [ max_int ] Within ]
  | _ -> []

let whole_equals_steps fmt =
  prop "steps equal the whole decode for any cut of the input"
    ~examples:(examples fmt) (case fmt) (fun c ->
      let input = apply c in
      let t = walk fmt (fmt.make ()) input ~sizes:c.sizes ~ends:c.ends in
      let got =
        match t.ended with
        | Ok taken -> Ok (t.out, taken)
        | Error f -> Error f.error
      in
      let w = whole fmt input (String.length t.out + room) in
      let declined =
        (* A Zstandard window above the stream decoder's bound is the
           whole-buffer decode's to take, and the only refusal of a step that
           decode does not make. *)
        match got with
        | Error { why = Unsupported m; _ } when fmt.name = "zstd" && w <> got ->
            contains ~msg:"a refusal decode does not make" ~sub:"Window_Size" m;
            true
        | _ -> false
      in
      classify "a window above the decoder's bound" declined;
      if not declined then (
        equal Lz_support.decoded (Result.map fst w) (Result.map fst got);
        match (w, got) with
        | Ok (_, consumed), Ok (_, taken) ->
            equal int ~msg:"bytes taken" consumed taken
        | _ -> ());
      (* A deflate or zlib stream is Done at its last byte. *)
      (match (c.change, t.ended) with
      | (Same | Append _ | Then _), Ok taken when not fmt.members ->
          equal int ~msg:"Done at the stream's end"
            (String.length (data c.input))
            taken
      | _ -> ());
      (* A prefix of a valid input fails only once its end is said. *)
      (match t.ended with
      | Error f when valid_prefix fmt c.change && not f.last ->
          failf "a valid prefix failed at step %d without ~last: %a" f.at_step
            (Testable.pp Lz_support.error)
            f.error
      | _ -> ());
      let failed why =
        match got with Error { why = w; _ } -> why w | Ok _ -> false
      in
      cover "an input that decodes"
        (match got with Ok (s, _) -> s <> "" | _ -> false);
      cover "an input cut short" (failed (fun w -> w = Truncated));
      cover "a malformed input"
        (failed (function Malformed _ -> true | _ -> false));
      cover "a failure after a step that took bytes"
        (match t.ended with Error f -> f.at_step > 0 | Ok _ -> false);
      cover "a step short of its slice" t.short;
      cover "an empty slice before the end" t.idle;
      cover "a second member or stream after the first"
        (match (c.change, got) with Then _, Ok _ -> true | _ -> false);
      if fmt.members then
        cover "Done on an empty step after ~last:false"
          (c.ends = After && Result.is_ok got)
      else
        cover "bytes after the stream"
          (match (c.change, got) with
          | Append _, Ok (_, taken) -> taken < String.length input
          | _ -> false))

(* Steps equal the whole-buffer decode where the buffer moves *)

(* Three Zstandard frames whose windows grow, then shrink: 100 bytes in a
   100-byte window, an unsized frame with a 1 MiB window, then the first
   again. *)
let growing =
  concat "grow.zst" [ "text_100.zst3"; "window1m.zst"; "text_100.zst3" ]

(* RFC 8878, 3.1.1.1: the size of the frame header at [at] in [s]. *)
let header_size s at =
  let d = Char.code s.[at + 4] in
  let single = d land 0x20 <> 0 in
  let fcs = [| (if single then 1 else 0); 2; 4; 8 |].(d lsr 6) in
  5 + (if single then 0 else 1) + [| 0; 1; 2; 4 |].(d land 3) + fcs

(* Of a walk over [growing]: whether the step that took the second frame's
   header, where the decoder asks for a larger buffer, began before the first
   frame's output was all lent. *)
let grown_pending trace =
  let small = golden "text_100.zst3" in
  let start = String.length (data small) in
  let header_end = start + header_size (data growing) start in
  let rec go total = function
    | [] -> false
    | (at, taken, run) :: rest ->
        if at < header_end && header_end <= at + taken then
          total < String.length (raw small)
        else go (total + run) rest
  in
  go 0 trace

(* The input each format's buffer moves on: the long stream for the deflate
   family, whose runs fill and wrap and whose matches cross the wraps; [growing]
   for Zstandard, whose buffer grows at its second frame and holds its third. *)
let moving fmt = if fmt.name = "zstd" then growing else fmt.long

let across_buffer fmt =
  let g = moving fmt in
  let gen =
    Gen.(
      with_pp
        (fun ppf (sizes, ends) ->
          pp_case ppf { input = g; change = Same; sizes; ends })
        (map
           (fun (sizes, ends) -> (spread (String.length (data g)) sizes, ends))
           (pair sizes ends)))
  in
  prop "steps equal the decoded bytes where the buffer wraps or grows" gen
    (fun (sizes, ends) ->
      let input = data g in
      let t = walk fmt (fmt.make ()) input ~sizes ~ends in
      equal Lz_support.decoded
        (Ok (raw g))
        (match t.ended with Ok _ -> Ok t.out | Error f -> Error f.error);
      equal (result int pass) ~msg:"bytes taken"
        (Ok (String.length input))
        (Result.map_error ignore t.ended);
      if fmt.name = "zstd" then
        cover "a buffer grown while a run is pending" (grown_pending t.trace))

(* Encoders *)

type encoding = {
  name : string;
  encoder : unit -> Lz.Encoder.t;
  decode : Lz.bigbytes -> Lz.bigbytes -> (Lz.progress, Lz.error) result;
}

let encodings =
  [
    {
      name = "deflate";
      encoder = (fun () -> Lz.Deflate.encoder ());
      decode = Lz.Deflate.decode;
    };
    {
      name = "zlib";
      encoder = (fun () -> Lz.Zlib.encoder ());
      decode = Lz.Zlib.decode;
    };
    {
      name = "gzip";
      encoder = (fun () -> Lz.Gzip.encoder ());
      decode = Lz.Gzip.decode;
    };
    {
      name = "zstd";
      encoder = (fun () -> Lz.Zstd.encoder ());
      decode = Lz.Zstd.decode;
    };
  ]

(* [encode_walk ~lend e input ~piece] steps [e] over [input], at most [piece]
   bytes a step, with ~last:true on the steps that reach its end, until [Done],
   and checks that each step applies [f] once. [lend k run] is applied in the
   [f] of step [k], counted from 0, once the run is copied. It is the stream
   written and the number of steps. *)
let encode_walk ?(lend = fun _ _ -> ()) e input ~piece =
  let src = Lz_support.of_string input and n = String.length input in
  let out = Buffer.create n in
  let rec go pos k =
    let len = min piece (n - pos) in
    let applied = ref 0 in
    let f run =
      incr applied;
      Buffer.add_string out (Lz_support.to_string run);
      lend k run
    in
    let r =
      Lz.Encoder.step e
        ~last:(pos + len = n)
        (Bigarray.Array1.sub src pos len)
        f
    in
    if !applied <> 1 then failf "step %d applied f %d times" k !applied;
    match r with
    | Done _ -> k + 1
    | More 0 -> failf "step %d took nothing" k
    | More t -> go (pos + t) (k + 1)
  in
  let steps = go 0 0 in
  (Buffer.contents out, steps)

(* An input in at most about 64 steps of at least [piece] bytes. *)
let pieces_of input piece = max piece ((String.length input / 64) + 1)

let piece =
  Gen.frequency [ (3, Gen.int_range 1 64); (1, Gen.int_range 1 (64 * kib)) ]

let encoder_reentry =
  cases "a step from f raises Invalid_argument, which kills the encoder"
    ~id:(fun enc -> enc.name)
    encodings
    (fun enc ->
      let e = enc.encoder () in
      let src = Lz_support.of_string "hello" in
      raises_match ~msg:"a step from f" invalid (fun () ->
          Lz.Encoder.step e ~last:true src (fun _ ->
              ignore (Lz.Encoder.step e ~last:true empty ignore)));
      raises_match ~msg:"a step after" invalid (fun () ->
          Lz.Encoder.step e ~last:true src ignore))

(* The refused step changes nothing: the held step writes the stream of its
   input, and after its Done a step writes the same stream again. *)
let encoder_other_domain =
  cases
    "a step from another domain while one runs raises Invalid_argument and \
     changes nothing"
    ~id:(fun enc -> enc.name)
    encodings
    (fun enc ->
      let input = Lz_support.corpus Text 1000 in
      let n = String.length input in
      let e = enc.encoder () in
      let out = Buffer.create 128 in
      let r =
        held
          (fun hold ->
            Lz.Encoder.step e ~last:true (Lz_support.of_string input)
              (fun run ->
                Buffer.add_string out (Lz_support.to_string run);
                hold ()))
          (fun () ->
            raises_match ~msg:"a step while another runs" invalid (fun () ->
                Lz.Encoder.step e ~last:true (Lz_support.of_string input) ignore))
      in
      equal Lz_support.step ~msg:"the held step" (Done n) r;
      let stream = Buffer.contents out in
      let dst = Lz_support.create n in
      equal
        (result Lz_support.progress Lz_support.error)
        ~msg:"the held step's stream"
        (Ok { Lz.consumed = String.length stream; produced = n })
        (enc.decode (Lz_support.of_string stream) dst);
      equal string ~msg:"what it decodes to" input (Lz_support.to_string dst);
      let again, _ = encode_walk e input ~piece:max_int in
      equal run_w ~msg:"a step after the refused one" stream again)

let encoder_raising enc =
  let gen = Gen.triple Lz_support.recipe piece Gen.nat in
  prop "an exception from f propagates and kills the encoder" gen
    (fun (r, piece, at) ->
      let input = Lz_support.build r in
      let piece = pieces_of input piece in
      let _, steps = encode_walk (enc.encoder ()) input ~piece in
      let at = at mod steps in
      cover "f raising after the first step" (at > 0);
      let e = enc.encoder () in
      let lend k _ = if k = at then raise Exit in
      raises ~msg:"the exception of f" Exit (fun () ->
          encode_walk ~lend e input ~piece);
      raises_match ~msg:"a step after f raised" invalid (fun () ->
          Lz.Encoder.step e ~last:false (Lz_support.of_string input) ignore);
      raises_match ~msg:"an empty step after f raised" invalid (fun () ->
          Lz.Encoder.step e ~last:true empty ignore))

(* Every byte of each run complemented once [f] has copied it: an encoder that
   read its run back would write other bytes after it. The example is past the
   input an encoder keeps (lz.mli, Encoder), so that every encoder writes bytes
   before its last step. *)
let encoder_overwrite enc =
  let complement run =
    for i = 0 to Bigarray.Array1.dim run - 1 do
      run.{i} <- run.{i} lxor 0xff
    done
  in
  prop "f may overwrite its run without changing the stream"
    ~examples:[ (Lz_support.Corpus (Text, 448 * kib), 64 * kib) ]
    (Gen.pair Lz_support.recipe piece)
    (fun (r, piece) ->
      let input = Lz_support.build r in
      let piece = pieces_of input piece in
      let stream, _ = encode_walk (enc.encoder ()) input ~piece in
      (* The first step that wrote bytes. *)
      let first = ref max_int in
      let lend k run =
        if Bigarray.Array1.dim run > 0 then first := min !first k;
        complement run
      in
      let written, steps = encode_walk ~lend (enc.encoder ()) input ~piece in
      equal run_w stream written;
      cover "bytes written before the last step" (!first < steps - 1))

let () =
  exit
    (run "Step"
       [
         group "Decoder.step"
           ([ after_error; reentry; other_domain ]
           @ List.map
               (fun (fmt : format) ->
                 group
                   (String.capitalize_ascii fmt.name)
                   [
                     one_user fmt;
                     raising fmt;
                     kept_runs fmt;
                     ends_of fmt;
                     no_input fmt;
                     whole_equals_steps fmt;
                     across_buffer fmt;
                   ])
               formats);
         group "Encoder.step"
           ([ encoder_reentry; encoder_other_domain ]
           @ List.map
               (fun (enc : encoding) ->
                 group
                   (String.capitalize_ascii enc.name)
                   [ encoder_raising enc; encoder_overwrite enc ])
               encodings);
       ])
