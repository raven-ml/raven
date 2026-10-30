(* The laws of Lz.Decoder.step over every streaming format, against the
   vendors' goldens and their mutated and cut copies: steps equal the
   whole-buffer decode for any cut of the input, a step ends as its doc
   states, Truncated comes only with ~last:true, an error or a raise in [f]
   kills the decoder, a lent run outlives its decoder, and a decoder takes
   one step at a time. Lz.Encoder.step shares the last two laws: a raise in
   [f] kills the encoder, and an encoder takes one step at a time. *)

open Windtrap

let kib = 1024

(* Formats *)

type format = {
  name : string;
  make : unit -> Lz.Decoder.t;
  members : bool;  (* Read to the input's end, as gzip and Zstandard. *)
}

(* A mutated Zstandard header may declare any window up to the decoder's
   bound, which the decoder allocates; 4 MiB holds every golden's. *)
let zstd_window = 4 * 1024 * kib

let formats =
  [
    { name = "deflate"; make = Lz.Deflate.decoder; members = false };
    { name = "zlib"; make = Lz.Zlib.decoder; members = false };
    { name = "gzip"; make = Lz.Gzip.decoder; members = true };
    {
      name = "zstd";
      make = (fun () -> Lz.Zstd.decoder ~window:zstd_window ());
      members = true;
    };
  ]

let data g = Lazy.force g.Lz_support.data
let raw g = Lazy.force g.Lz_support.raw
let golden name = List.find (fun g -> g.Lz_support.file = name) (Lz_support.goldens ())
let empty = Lz_support.of_string ""
let stepped = result Lz_support.step Lz_support.error
let invalid e = Exn.invalid_arg e
let step d ~last s f = Lz.Decoder.step d ~last (Lz_support.of_string s) f

(* A run as its length and digest: runs reach 256 KiB. *)
let run_w =
  Testable.make ~equal:String.equal ~pp:(fun ppf s ->
      Format.fprintf ppf "%d bytes, digest %s" (String.length s)
        (Digest.to_hex (Digest.string s)))

(* Walking an input in steps *)

(* Where the input's end is said. [Within]: on the slice that reaches it,
   as a reader that knows its size. [After]: every slice has ~last:false,
   then empty steps have ~last:true, as a reader whose read returned
   nothing. *)
type ends = Within | After

type failure = { error : Lz.error; at_step : int; last : bool }

type walk = {
  out : string;  (* The runs, concatenated. *)
  ended : (int, failure) result;  (* The bytes taken up to [Done]. *)
  ok_steps : int;  (* The steps that returned [Ok]. *)
  short : bool;  (* A step returned [More k] short of its slice. *)
  idle : bool;  (* A step was given no bytes before the input's end. *)
  trace : (int * int * int) list;
      (* Each step that returned [Ok], in order: the first byte it was
         given, the bytes it took and its run's length. *)
}

(* Output past this means a decoder that never ends. *)
let out_cap = 64 * 1024 * kib

(* [walk fmt d data ~sizes ~ends] steps [d] over [data], step [k] given at
   most [sizes.(k mod n)] bytes from the first byte not yet taken, and
   checks each step against Decoder.step's doc as it goes. [lend k run] is
   applied to each run inside [f], [after k] after each step that returned
   [Ok]. *)
let walk ?(lend = fun _ _ -> ()) ?(after = fun _ -> ()) fmt d data ~sizes
    ~ends =
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
        (* Law 8: an error kills the decoder. *)
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
        after k;
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
            if j < len then (
              short := true;
              if run = "" then
                broke (Printf.sprintf "More %d, short of the slice, lent an empty run" j));
            if last && len = 0 && run = "" then
              broke "More with no input, no output and ~last:true";
            go (pos + j) (k + 1))
  in
  go 0 0

let whole fmt data room =
  let dst = Lz_support.create room in
  match Lz_support.decoder fmt.name (Lz_support.of_string data) dst with
  | Ok { Lz.consumed; produced } ->
      Ok (Lz_support.to_string ~length:produced dst, consumed)
  | Error e -> Error e

(* A failing step may hold one run it did not lend, at most 256 KiB:
   the whole decode gets more room than that past what the steps lent. *)
let room = 512 * kib

(* Drawn cases *)

type change =
  | Same
  | Flip of int * int  (* Byte, bit. *)
  | Replace of int * char
  | Cut of int  (* The first bytes kept. *)
  | Append of string
  | Then of Lz_support.golden  (* Another golden of the format after it. *)

type case = {
  golden : Lz_support.golden;
  change : change;
  sizes : int list;
  ends : ends;
}

let pp_case ppf { golden; change; sizes; ends } =
  let ints =
    Format.(pp_print_list ~pp_sep:(fun p () -> pp_print_string p "; ") pp_print_int)
  in
  Format.fprintf ppf "{ golden = %s; change = %s; sizes = [%a]; ends = %s }"
    golden.Lz_support.file
    (match change with
    | Same -> "Same"
    | Flip (i, b) -> Printf.sprintf "Flip (%d, %d)" i b
    | Replace (i, c) -> Printf.sprintf "Replace (%d, %C)" i c
    | Cut i -> Printf.sprintf "Cut %d" i
    | Append s -> Printf.sprintf "Append %S" s
    | Then g -> "Then " ^ g.Lz_support.file)
    ints sizes
    (match ends with Within -> "Within" | After -> "After")

let apply { golden; change; _ } =
  let d = data golden in
  let set i c = String.mapi (fun j x -> if j = i then c else x) d in
  match change with
  | Same -> d
  | Flip (i, b) -> set i (Char.chr (Char.code d.[i] lxor (1 lsl b)))
  | Replace (i, c) -> set i c
  | Cut i -> String.sub d 0 i
  | Append s -> d ^ s
  | Then g -> d ^ data g

(* Whether every byte is a prefix of a valid input: no step with
   ~last:false may then fail. *)
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

(* [spread n sizes] adds [n / most_steps] bytes to each slice above 0, so
   that a walk over [n] bytes takes at most [most_steps] slices that take
   bytes, and drawn sizes stay distinct. Slices are unchanged on inputs
   below 1000 bytes, two goldens in three; over the longest inputs, slices
   of a few bytes took thousands of steps a case and outran the
   sanitizers' time. *)
let most_steps = 1000

let spread n sizes =
  List.map (fun s -> if s = 0 then 0 else s + (n / most_steps)) sizes

(* [long fmt] is a stream in the format's wrapper (RFC 1950, RFC 1952)
   of 40,000 bytes of text in a stored block (RFC 1951, 3.2.4), then a
   block of fixed codes (3.2.6) of 1100 matches of 258 bytes at 32 KiB,
   whose sources follow each other: 323,800 bytes from about 44 KB, for
   the deflate family, whose goldens decode to at most 200,000 bytes,
   short of a run. A slice of a few KiB fills a run, and wherever a step
   goes back to its buffer's start a match reads across the point. *)
let long fmt =
  let text = Lz_support.corpus Text 40_000 in
  let out = Buffer.create 330_000 and b = Buffer.create 48_000 in
  Buffer.add_string out text;
  (match fmt.name with
  | "zlib" -> Buffer.add_string b "\x78\x01"
  | "gzip" -> Buffer.add_string b "\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"
  | _ -> ());
  let n = String.length text in
  Buffer.add_char b '\x00';
  Buffer.add_uint16_le b n;
  Buffer.add_uint16_le b (lnot n land 0xffff);
  Buffer.add_string b text;
  (* Bits go out from each byte's lowest; a code goes out from its
     highest bit, extra bits from their lowest. *)
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
  let code c k = for i = k - 1 downto 0 do put ((c lsr i) land 1) 1 done in
  put 1 1;
  put 1 2;
  for _ = 1 to 1100 do
    (* Length 258 is symbol 285, code 0xc5 of 8 bits; distance 32768 is
       code 29 of 5 bits with 13 extra bits, 32768 - 24577. *)
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
  (match fmt.name with
  | "zlib" -> Buffer.add_int32_be b (Lz.Adler32.string raw)
  | "gzip" ->
      Buffer.add_int32_le b (Int32.of_int (Lz.Crc32.string raw));
      Buffer.add_int32_le b (Int32.of_int (String.length raw))
  | _ -> ());
  {
    Lz_support.file = "long." ^ fmt.name;
    codec = fmt.name;
    vendor = "test_step";
    members = 1;
    raw = Lazy.from_val raw;
    data = Lazy.from_val (Buffer.contents b);
  }

(* For Zstandard, four goldens' frames in one stream, 800,000 bytes: its
   goldens decode to at most 200,000 bytes, which a run holds whole. *)
let long_zstd =
  lazy
    (let gs =
       List.map golden
         [ "text_200000.zst19"; "text_200000.zst1"; "text_200000.zst3"; "window1m.zst" ]
     in
     {
       Lz_support.file = "long.zst";
       codec = "zstd";
       vendor = "test_step";
       members = 4;
       raw = lazy (String.concat "" (List.map raw gs));
       data = lazy (String.concat "" (List.map data gs));
     })

(* Inputs above 64 KiB of output, drawn more often: a run fills only
   under a large slice of one, and only in [long] or [long_zstd], drawn as
   often again. *)
let goldens_gen fmt =
  let all = Lz_support.goldens_of fmt.name in
  let large = List.filter (fun g -> String.length (raw g) > 64 * kib) all in
  let longer = if fmt.name = "zstd" then Lazy.force long_zstd else long fmt in
  Gen.(frequency [ (3, of_list all); (1, of_list large); (1, constant longer) ])

let case fmt =
  let open Gen in
  let others = Array.of_list (Lz_support.goldens_of fmt.name) in
  let change =
    frequency
      [
        (2, constant `Same);
        (2, map (fun (i, b) -> `Flip (i, b)) (pair nat (int_range 0 7)));
        (1, map (fun (i, c) -> `Replace (i, c)) (pair nat char));
        (2, map (fun i -> `Cut i) nat);
        (1, map (fun s -> `Append s) (string_of ~size:(int_range 1 16) char));
        (1, map (fun i -> `Then i) (int_range 0 (Array.length others - 1)));
      ]
  in
  let make (golden, change, sizes, ends) =
    let n = String.length (data golden) in
    let change =
      match change with
      | `Same -> Same
      | `Flip (i, b) -> Flip (i mod n, b)
      | `Replace (i, c) -> Replace (i mod n, c)
      | `Cut i -> Cut (i mod n)
      | `Append s -> Append s
      | `Then i -> Then others.(i)
    in
    let c = { golden; change; sizes; ends } in
    { c with sizes = spread (String.length (apply c)) sizes }
  in
  with_pp pp_case (map make (quad (goldens_gen fmt) change sizes ends))

(* Law 5, with laws 7, 8 and 9 on the way *)

let whole_equals_steps fmt =
  prop fmt.name (case fmt) (fun c ->
      let input = apply c in
      let t = walk fmt (fmt.make ()) input ~sizes:c.sizes ~ends:c.ends in
      let got =
        match t.ended with Ok taken -> Ok (t.out, taken) | Error f -> Error f.error
      in
      let w = whole fmt input (String.length t.out + room) in
      let declined =
        (* A Zstandard window above the stream decoder's bound is the
           whole-buffer decode's to take. *)
        match got with
        | Error { why = Unsupported _; _ } -> fmt.name = "zstd" && w <> got
        | _ -> false
      in
      classify "a window above the decoder's bound" declined;
      if not declined then (
        equal Lz_support.decoded (Result.map fst w) (Result.map fst got);
        match (w, got) with
        | Ok (_, consumed), Ok (_, taken) -> equal int ~msg:"bytes taken" consumed taken
        | _ -> ());
      (* Law 9: a deflate or zlib stream is Done at its last byte. *)
      (match (c.change, t.ended) with
      | (Same | Append _ | Then _), Ok taken when not fmt.members ->
          equal int ~msg:"Done at the stream's end" (String.length (data c.golden)) taken
      | _ -> ());
      (* Law 7: a prefix of a valid input fails only once its end is said. *)
      (match t.ended with
      | Error f when valid_prefix fmt c.change ->
          equal bool
            ~msg:(Printf.sprintf "a valid prefix failed at step %d without ~last" f.at_step)
            true f.last
      | _ -> ());
      let failed why =
        match got with Error { why = w; _ } -> why w | Ok _ -> false
      in
      cover "an input that decodes" (match got with Ok (s, _) -> s <> "" | _ -> false);
      cover "an input cut short" (failed (fun w -> w = Truncated));
      cover "a malformed input" (failed (function Malformed _ -> true | _ -> false));
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

(* Law 5 where the buffer moves *)

(* The output offsets at which a deflate-family decoder goes back to its
   buffer's start: at each step that begins with more than the window and
   the slack, 32768 + 320 bytes, written since the last time (deflate.h).
   [trace] is a walk's, from a fresh decoder. *)
let deflate_wraps trace =
  let _, _, wraps =
    List.fold_left
      (fun (pos, total, wraps) (_, _, run) ->
        let pos, wraps = if pos > 33_088 then (0, total :: wraps) else (pos, wraps) in
        (pos + run, total + run, wraps))
      (0, 0, []) trace
  in
  List.rev wraps

(* Whether a match of [long] reads across output offset [at]: its matches'
   sources follow each other from 40,000 - 32,768 bytes, 258 each. *)
let long_spans at =
  let first = 40_000 - 32_768 in
  at > first && at < first + (1100 * 258) && (at - first) mod 258 <> 0

(* Three Zstandard frames whose windows grow, then shrink: 100 bytes in a
   100-byte window, an unsized frame with a 1 MiB window, then the first
   again. *)
let growing =
  lazy
    (let small = golden "text_100.zst3" and big = golden "window1m.zst" in
     {
       Lz_support.file = "grow.zst";
       codec = "zstd";
       vendor = "test_step";
       members = 3;
       raw = lazy (raw small ^ raw big ^ raw small);
       data = lazy (data small ^ data big ^ data small);
     })

(* RFC 8878, 3.1.1.1: the size of the frame header at [at] in [s]. *)
let header_size s at =
  let d = Char.code s.[at + 4] in
  let single = d land 0x20 <> 0 in
  let fcs = [| (if single then 1 else 0); 2; 4; 8 |].(d lsr 6) in
  5 + (if single then 0 else 1) + [| 0; 1; 2; 4 |].(d land 3) + fcs

(* Of a walk over [growing]: whether the step that took the second
   frame's header, where the decoder asks for a larger buffer, began
   before the first frame's output was all lent. *)
let grown_pending trace =
  let small = golden "text_100.zst3" in
  let start = String.length (data small) in
  let header_end = start + header_size (Lazy.force (Lazy.force growing).data) start in
  let rec go total = function
    | [] -> false
    | (at, taken, run) :: rest ->
        if at < header_end && header_end <= at + taken then total < String.length (raw small)
        else go (total + run) rest
  in
  go 0 trace

(* The input each format's buffer moves on: [long] for the deflate family,
   whose runs fill and wrap and whose matches cross the wraps; [growing]
   for Zstandard, whose buffer grows at its second frame and holds its
   third. *)
let moving fmt = if fmt.name = "zstd" then Lazy.force growing else long fmt

let across_buffer fmt =
  let g = moving fmt in
  let gen =
    Gen.(
      with_pp
        (fun ppf (sizes, ends) ->
          pp_case ppf { golden = g; change = Same; sizes; ends })
        (map
           (fun (sizes, ends) -> (spread (String.length (data g)) sizes, ends))
           (pair sizes ends)))
  in
  prop fmt.name gen (fun (sizes, ends) ->
      let input = data g in
      let t = walk fmt (fmt.make ()) input ~sizes ~ends in
      equal Lz_support.decoded (Ok (raw g))
        (match t.ended with Ok _ -> Ok t.out | Error f -> Error f.error);
      equal (result int pass) ~msg:"bytes taken" (Ok (String.length input))
        (Result.map_error ignore t.ended);
      if fmt.name = "zstd" then begin
        cover "a buffer grown while a run is pending" (grown_pending t.trace);
        (* The third frame's window is the first's: its buffer is the one
           the second asked for. *)
        cover "a frame in a buffer grown for an earlier, larger frame"
          (Result.is_ok t.ended)
      end
      else begin
        let wraps = deflate_wraps t.trace in
        cover "a run that wraps" (wraps <> []);
        cover "a match whose source spans the wrap" (List.exists long_spans wraps)
      end)

(* Done and More *)

let after_stream = "\x00after"

(* Each golden in one slice: a deflate or zlib stream is Done at its last
   byte with bytes after, with and without ~last; a gzip or Zstandard
   stream is More on the step that takes its last byte with ~last:false,
   then Done on an empty step with ~last:true. The next step begins a new
   stream. *)
let ends_of fmt =
  cases fmt.name
    ~id:(fun g -> g.Lz_support.file)
    (Lz_support.goldens_of fmt.name)
    (fun g ->
      let n = String.length (data g) in
      let expect ~msg t =
        equal Lz_support.decoded ~msg (Ok (raw g))
          (match t.ended with Ok _ -> Ok t.out | Error f -> Error f.error);
        equal (result int pass) ~msg:(msg ^ ": bytes taken") (Ok n)
          (Result.map_error ignore t.ended)
      in
      List.iter
        (fun ends ->
          let d = fmt.make () in
          let input = if fmt.members then data g else data g ^ after_stream in
          let name = match ends with Within -> "~last:true" | After -> "~last:false" in
          expect ~msg:("first stream, " ^ name)
            (walk fmt d input ~sizes:[ max_int ] ~ends);
          expect ~msg:("next stream, " ^ name)
            (walk fmt d input ~sizes:[ max_int ] ~ends))
        [ Within; After ])

(* Law 8: a raise in f *)

let raising fmt =
  let gen =
    Gen.(
      with_pp
        (fun ppf (g, sizes, r) ->
          Format.fprintf ppf "%a, f raising at step %d" pp_case
            { golden = g; change = Same; sizes; ends = Within }
            r)
        (map
           (fun (g, sizes, r) -> (g, spread (String.length (data g)) sizes, r))
           (triple (goldens_gen fmt) sizes nat)))
  in
  prop fmt.name gen (fun (g, sizes, r) ->
      let input = data g in
      let clean = walk fmt (fmt.make ()) input ~sizes ~ends:Within in
      assume (clean.ok_steps > 0);
      let r = r mod clean.ok_steps in
      cover "f raising after the first step" (r > 0);
      let d = fmt.make () in
      raises ~msg:"the exception of f" Exit (fun () ->
          walk ~lend:(fun k _ -> if k = r then raise Exit) fmt d input ~sizes
            ~ends:Within);
      raises_match ~msg:"a step after f raised" invalid (fun () ->
          step d ~last:false input ignore);
      raises_match ~msg:"an empty step after f raised" invalid (fun () ->
          Lz.Decoder.step d ~last:true empty ignore))

(* Law 10: lent runs *)

(* Steps a fresh decoder over [input], keeping every run past [f] with a
   copy taken in [f], and leaves the decoder only in [gone]. *)
let[@inline never] keep_runs fmt input ~sizes ~ends gone =
  let d = fmt.make () in
  Weak.set gone 0 (Some d);
  let kept = ref [] in
  let lend _ run = kept := (run, Lz_support.to_string run) :: !kept in
  let after k =
    match !kept with
    | (run, copy) :: _ ->
        equal run_w ~msg:(Printf.sprintf "the run of step %d once it returned" k)
          copy (Lz_support.to_string run)
    | [] -> ()
  in
  ignore (walk ~lend ~after fmt d input ~sizes ~ends);
  List.rev !kept

let kept_runs fmt =
  let gen =
    Gen.(
      with_pp
        (fun ppf (g, sizes, ends) ->
          pp_case ppf { golden = g; change = Same; sizes; ends })
        (map
           (fun (g, sizes, ends) ->
             (g, spread (String.length (data g)) sizes, ends))
           (triple (goldens_gen fmt) sizes ends)))
  in
  prop fmt.name gen (fun (g, sizes, ends) ->
      let gone = Weak.create 1 in
      let kept = keep_runs fmt (data g) ~sizes ~ends gone in
      (* Its contents are a later step's: reading every byte is the law,
         which the sanitizers check. *)
      let read when_ =
        List.iteri
          (fun i (run, copy) ->
            equal int
              ~msg:(Printf.sprintf "run %d %s" i when_)
              (String.length copy)
              (String.length (Lz_support.to_string run)))
          kept
      in
      read "after later steps";
      Gc.full_major ();
      is_none ~msg:"the decoder was collected" (Weak.get gone 0);
      read "after the decoder was collected";
      equal Lz_support.decoded (Ok (raw g)) (Ok (String.concat "" (List.map snd kept)));
      cover "several runs" (List.length kept > 2))

(* Law 11: one value, one user *)

(* A small stream of each format. *)
let small fmt =
  golden
    (match fmt.name with
    | "deflate" -> "text_100.ld6.deflate"
    | "zlib" -> "text_100.zlib6"
    | "gzip" -> "text_100.gz9"
    | _ -> "text_100.zst3")

(* A step from [f] raises, and the raise kills the decoder. *)
let reentry fmt =
  test fmt.name (fun () ->
      let d = fmt.make () in
      raises_match ~msg:"a step from f" invalid (fun () ->
          step d ~last:true (data (small fmt)) (fun _ ->
              ignore (Lz.Decoder.step d ~last:true empty ignore)));
      raises_match ~msg:"a step after" invalid (fun () ->
          step d ~last:true (data (small fmt)) ignore))

(* Held deterministically: one domain's step waits in [f] until another
   domain's step on the same decoder has returned or raised. The refused
   step changes nothing: the held step and a later one decode their
   streams. *)
let other_domain fmt =
  test ~timeout:10. fmt.name (fun () ->
      let g = small fmt in
      let d = fmt.make () in
      let entered = Atomic.make false and tried = Atomic.make false in
      let returned = Atomic.make false in
      let out = Buffer.create 128 in
      let holder =
        Domain.spawn (fun () ->
            let r =
              step d ~last:true (data g) (fun run ->
                  Buffer.add_string out (Lz_support.to_string run);
                  Atomic.set entered true;
                  while not (Atomic.get tried) do
                    Domain.cpu_relax ()
                  done)
            in
            Atomic.set returned true;
            r)
      in
      while not (Atomic.get entered || Atomic.get returned) do
        Domain.cpu_relax ()
      done;
      if not (Atomic.get entered) then
        failf "the held step returned %a without applying f" (Testable.pp stepped)
          (Domain.join holder);
      let refused =
        match step d ~last:true (data g) ignore with
        | r -> Ok r
        | exception e -> Error e
      in
      Atomic.set tried true;
      let held = Domain.join holder in
      (match refused with
      | Error e when invalid e -> ()
      | Error e -> raise e
      | Ok r -> failf "a step while another runs returned %a" (Testable.pp stepped) r);
      equal stepped ~msg:"the held step" (Ok (Done (String.length (data g)))) held;
      equal Lz_support.decoded (Ok (raw g)) (Ok (Buffer.contents out));
      let t = walk fmt d (data g) ~sizes:[ max_int ] ~ends:Within in
      equal Lz_support.decoded ~msg:"a step after the refused one" (Ok (raw g))
        (match t.ended with Ok _ -> Ok t.out | Error f -> Error f.error))

(* The inputs the shared decoder is stepped on: thirds of a stream, the
   whole stream, nothing, and bytes that begin no stream. *)
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

(* The outcome of [last, p] after [steps], by a fresh decoder taking them
   in order on one domain, memoised: the judge replays every order the
   domains could have taken. *)
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
            | Error e ->
                failf "the reference's replay failed: %a" Lz.pp_error e)
          (List.rev steps);
        let runs = ref [] in
        let r =
          Lz.Decoder.step d ~last (bytes p) (fun run ->
              runs := Lz_support.to_string run :: !runs)
        in
        let o = (r, !runs) in
        Hashtbl.replace memo key o;
        o

(* A refusal for being in use, told from one for being dead by its
   message. *)
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

(* The judge: a step another domain's step refused for being in use
   changes nothing; on a dead decoder a step raises Invalid_argument;
   otherwise the step is the one the steps taken so far predict, and an
   error or a raise in [f] kills the decoder. *)
let judge fmt bytes ~raising last p m (got : (outcome, exn) result) =
  match got with
  | Error e when in_use e -> ()
  | Error e when m.dead && invalid e -> ()
  | _ when m.dead -> fail "a step on a dead decoder did not raise Invalid_argument"
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
  let bytes =
    let table = List.map (fun p -> (p, Lz_support.of_string (piece_bytes fmt p))) pieces in
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
  (* Each case runs 50 programs on two domains: 50 cases see thousands of
     refusals, and more only slow the sanitizers' debug runtime, whose
     every minor collection stops both domains. *)
  stateful ~domains:2 ~count:50 fmt.name
    [
      command "decoder" (Gen.unit @-> makes dec)
        (fun () -> { steps = []; dead = false })
        fmt.make;
      (* Listed twice: drawn twice as often as a raise in [f]. *)
      stepping "step" ~raising:false;
      stepping "step" ~raising:false;
      stepping "step with f raising" ~raising:true;
    ]

(* Encoders: a raise in f, and one value, one user *)

type encoding = {
  label : string;
  encoder : unit -> Lz.Encoder.t;
  decode : Lz.bigbytes -> Lz.bigbytes -> (Lz.progress, Lz.error) result;
}

let encodings =
  [
    {
      label = "deflate";
      encoder = (fun () -> Lz.Deflate.encoder ());
      decode = Lz.Deflate.decode;
    };
    {
      label = "zlib";
      encoder = (fun () -> Lz.Zlib.encoder ());
      decode = Lz.Zlib.decode;
    };
    {
      label = "gzip";
      encoder = (fun () -> Lz.Gzip.encoder ());
      decode = Lz.Gzip.decode;
    };
    {
      label = "zstd";
      encoder = (fun () -> Lz.Zstd.encoder ());
      decode = Lz.Zstd.decode;
    };
  ]

(* [encode_walk ~lend e input ~piece] steps [e] over [input], at most
   [piece] bytes a step, with ~last:true on the steps that reach its end,
   until [Done]. [lend k run] is the [f] of step [k], counted from 0. It is
   the stream written and the number of steps. *)
let encode_walk ?(lend = fun _ _ -> ()) e input ~piece =
  let src = Lz_support.of_string input and n = String.length input in
  let out = Buffer.create n in
  let rec go pos k =
    let len = min piece (n - pos) in
    let piece = Bigarray.Array1.sub src pos len in
    let f run =
      Buffer.add_string out (Lz_support.to_string run);
      lend k run
    in
    match Lz.Encoder.step e ~last:(pos + len = n) piece f with
    | Done _ -> k + 1
    | More 0 -> failf "step %d took nothing" k
    | More t -> go (pos + t) (k + 1)
  in
  let steps = go 0 0 in
  (Buffer.contents out, steps)

(* An input in at most about 64 steps of at least [piece] bytes. *)
let pieces_of input piece = max piece ((String.length input / 64) + 1)

let encoder_raising enc =
  let piece =
    Gen.frequency [ (3, Gen.int_range 1 64); (1, Gen.int_range 1 (64 * kib)) ]
  in
  let gen = Gen.triple Lz_support.recipe piece Gen.nat in
  prop enc.label gen (fun (r, piece, at) ->
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

(* A step from [f] raises, and the raise kills the encoder. *)
let encoder_reentry enc =
  test enc.label (fun () ->
      let e = enc.encoder () in
      let src = Lz_support.of_string "hello" in
      raises_match ~msg:"a step from f" invalid (fun () ->
          Lz.Encoder.step e ~last:true src (fun _ ->
              ignore (Lz.Encoder.step e ~last:true empty ignore)));
      raises_match ~msg:"a step after" invalid (fun () ->
          Lz.Encoder.step e ~last:true src ignore))

(* Held deterministically, as for decoders: one domain's step waits in [f]
   until another domain's step on the same encoder has returned or raised.
   The refused step changes nothing: the held step writes the stream of its
   input, and a later step writes the same stream again. *)
let encoder_other_domain enc =
  test ~timeout:10. enc.label (fun () ->
      let input = Lz_support.corpus Text 1000 in
      let n = String.length input in
      let e = enc.encoder () in
      let entered = Atomic.make false and tried = Atomic.make false in
      let returned = Atomic.make false in
      let out = Buffer.create 128 in
      let holder =
        Domain.spawn (fun () ->
            let r =
              Lz.Encoder.step e ~last:true (Lz_support.of_string input)
                (fun run ->
                  Buffer.add_string out (Lz_support.to_string run);
                  Atomic.set entered true;
                  while not (Atomic.get tried) do
                    Domain.cpu_relax ()
                  done)
            in
            Atomic.set returned true;
            r)
      in
      while not (Atomic.get entered || Atomic.get returned) do
        Domain.cpu_relax ()
      done;
      if not (Atomic.get entered) then
        failf "the held step returned %a without applying f"
          (Testable.pp Lz_support.step) (Domain.join holder);
      let src = Lz_support.of_string input in
      let refused =
        match Lz.Encoder.step e ~last:true src ignore with
        | r -> Ok r
        | exception x -> Error x
      in
      Atomic.set tried true;
      let held = Domain.join holder in
      (match refused with
      | Error x when invalid x -> ()
      | Error x -> raise x
      | Ok r ->
          failf "a step while another runs returned %a"
            (Testable.pp Lz_support.step) r);
      equal Lz_support.step ~msg:"the held step" (Done n) held;
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

(* Spec values *)

let no_input =
  cases "no input and ~last:true is Truncated at 0"
    ~id:(fun f -> f.name)
    formats
    (fun fmt ->
      equal stepped
        (Error { Lz.at = 0; why = Truncated })
        (Lz.Decoder.step (fmt.make ()) ~last:true empty ignore))

let each name law = group name (List.map law formats)
let each_encoder name law = group name (List.map law encodings)

let () =
  exit
    (run "Step"
       [
         no_input;
         each "steps equal the whole decode" whole_equals_steps;
         each "steps equal the whole decode where the buffer moves" across_buffer;
         each "a stream ends as the step's doc states" ends_of;
         each "a decoder whose f raised is dead" raising;
         each "a run kept past f stays readable" kept_runs;
         each "a step from f raises" reentry;
         each "a step from another domain while one runs raises" other_domain;
         each "a decoder shared by two domains" one_user;
         each_encoder "an encoder whose f raised is dead" encoder_raising;
         each_encoder "an encoder step from f raises" encoder_reentry;
         each_encoder
           "an encoder step from another domain while one runs raises"
           encoder_other_domain;
       ])
