(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The ring writer's templates, with no GPU: a packet loaded into a device
   state's template, as the driver loads its own at open, then filled as the
   writer fills it, is the packet's encoding. A template holds at most 16 words
   and 4 holes, each a term of at most 3 operations on argument 0, 1 or 2. A
   launch's dispatch, as the writer places it, is Pm4's on every GPU the driver
   drives, which no machine has all of. *)

open Windtrap
open Rig_amd_abi

let strf = Printf.sprintf

external create : unit -> int = "caml_rig_amd_create"

external set_template : int -> int -> string -> string -> unit
  = "caml_rig_amd_template"

external fill : int -> int -> int64 -> int64 -> int64 -> string
  = "rig_amd_test_fill"

let self = create ()
let max_words = 16
let max_holes = 4

let load p =
  let words, holes = Template.flatten (fun _ -> None) Fun.id p in
  set_template self 0 words holes

(* Printing *)

let rec pp_term ppf : int Packet.term -> unit = function
  | Value i -> Format.fprintf ppf "Value %d" i
  | Add (t, n) -> Format.fprintf ppf "Add (%a, 0x%Lx)" pp_term t n
  | Shift (t, n) -> Format.fprintf ppf "Shift (%a, %d)" pp_term t n
  | Or (t, n) -> Format.fprintf ppf "Or (%a, 0x%Lx)" pp_term t n

let pp_word ppf : int Packet.word -> unit = function
  | Dword n -> Format.fprintf ppf "Dword 0x%x" n
  | W32 t -> Format.fprintf ppf "W32 (%a)" pp_term t
  | W64 t -> Format.fprintf ppf "W64 (%a)" pp_term t

let pp_case ppf (p, (a, b, c)) =
  Format.fprintf ppf "[%a]@ with 0x%Lx, 0x%Lx, 0x%Lx"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
       pp_word)
    p a b c

(* Drawing *)

(* 64-bit integers at the edges of their halves and of the type. *)
let u64 =
  Gen.frequency
    [
      (3, Gen.int64);
      ( 1,
        Gen.of_list
          [
            0L;
            1L;
            -1L;
            0xffff_ffffL;
            0x1_0000_0000L;
            0x4000_0000_0000_0000L;
            Int64.min_int;
            Int64.max_int;
          ] );
    ]

let shift = Gen.(frequency [ (3, int_range 0 63); (1, of_list [ 0; 63 ]) ])
let map2 f a b = Gen.map (fun (a, b) -> f a b) (Gen.pair a b)

(* Terms on an argument with at most [depth] operations. *)
let rec term depth : int Packet.term Gen.t =
  let open Gen in
  let leaf = map (fun i -> Packet.Value i) (int_range 0 2) in
  if depth = 0 then leaf
  else
    let sub = term (depth - 1) in
    frequency
      [
        (2, leaf);
        (1, map2 (fun t n -> Packet.Add (t, n)) sub u64);
        (1, map2 (fun t n -> Packet.Shift (t, n)) sub shift);
        (1, map2 (fun t n -> Packet.Or (t, n)) sub u64);
      ]

let word : int Packet.word Gen.t =
  let open Gen in
  frequency
    [
      (4, map (fun n -> Packet.Dword n) int);
      (1, map (fun t -> Packet.W32 t) (term 3));
      (1, map (fun t -> Packet.W64 t) (term 3));
    ]

let is_hole : int Packet.word -> bool = function
  | Dword _ -> false
  | W32 _ | W64 _ -> true

(* The words of [ws] kept in turn while the packet stays within [words] words
   and [holes] holes. *)
let within ~words ~holes ws =
  let fits (p, n, h) w =
    let n' = n + Packet.size [ w ] and h' = if is_hole w then h + 1 else h in
    if n' > words || h' > holes then (p, n, h) else (w :: p, n', h')
  in
  let p, _, _ = List.fold_left fits ([], 0, 0) ws in
  List.rev p

(* Packets within a template's bounds, many reaching them, some ending with a
   hole at the last word. *)
let bounded =
  let words = Gen.list ~size:(Gen.int_range 0 24) word in
  let ending ws h =
    let n = max_words - Packet.size [ h ] in
    let p = within ~words:n ~holes:(max_holes - 1) ws in
    p @ List.init (n - Packet.size p) (fun i -> Packet.Dword i) @ [ h ]
  in
  let hole =
    Gen.frequency
      [
        (1, Gen.map (fun t -> Packet.W32 t) (term 3));
        (1, Gen.map (fun t -> Packet.W64 t) (term 3));
      ]
  in
  Gen.frequency
    [
      (3, Gen.map (within ~words:max_words ~holes:max_holes) words);
      (1, map2 ending words hole);
    ]

let case = Gen.with_pp pp_case (Gen.pair bounded (Gen.triple u64 u64 u64))

(* Coverage *)

let rec eval value : int Packet.term -> int64 = function
  | Value i -> value i
  | Add (t, n) -> Int64.add (eval value t) n
  | Shift (t, n) -> Int64.shift_right_logical (eval value t) n
  | Or (t, n) -> Int64.logor (eval value t) n

(* Whether an addition in [t] wraps past 2^64 with the arguments [value]. *)
let rec wraps value : int Packet.term -> bool = function
  | Value _ -> false
  | Add (t, n) ->
      let x = eval value t in
      wraps value t || Int64.unsigned_compare (Int64.add x n) x < 0
  | Shift (t, _) | Or (t, _) -> wraps value t

let rec has f : int Packet.term -> bool = function
  | Value _ -> false
  | (Add (t, _) | Shift (t, _) | Or (t, _)) as n -> f n || has f t

let any f p =
  List.exists (function Packet.Dword _ -> false | W32 t | W64 t -> f t) p

(* The index after the last word that holds a term. *)
let last_hole p =
  let _, last =
    List.fold_left
      (fun (i, last) w ->
        let i' = i + Packet.size [ w ] in
        (i', if is_hole w then i' else last))
      (0, 0) p
  in
  last

(* Tests *)

let law =
  prop "a loaded template, filled, is the packet's encoding" case
    (fun (p, (a, b, c)) ->
      let value i = [| a; b; c |].(i) in
      cover "16 words" (Packet.size p = max_words);
      cover "4 holes" (List.length (List.filter is_hole p) = max_holes);
      cover "a hole ending at word 16" (last_hole p = max_words);
      cover "a 64-bit hole"
        (List.exists (function Packet.W64 _ -> true | _ -> false) p);
      cover "a shift by 0"
        (any (has (function Shift (_, 0) -> true | _ -> false)) p);
      cover "a shift by 63"
        (any (has (function Shift (_, 63) -> true | _ -> false)) p);
      cover "an or with bit 63"
        (any
           (has (function
             | Or (_, n) -> Int64.logand n Int64.min_int <> 0L
             | _ -> false))
           p);
      cover "an addition that wraps" (any (wraps value) p);
      load p;
      equal string (Packet.encode value p) (fill self 0 a b c))

let refused name p =
  test name (fun () ->
      raises_match (Exn.invalid_arg ~substring:"Rig_amd.make") (fun () ->
          load p))

let templates =
  group "templates"
    [
      law;
      test "an or with bit 63 keeps it" (fun () ->
          let p = [ Packet.W64 (Or (Value 0, Int64.min_int)) ] in
          load p;
          equal string
            (Packet.encode (fun _ -> 0x100L) p)
            (fill self 0 0x100L 0L 0L));
      test "a load replaces the template" (fun () ->
          load [ Dword 1; W32 (Value 0); W64 (Value 1) ];
          load [ W32 (Value 2) ];
          equal string
            (Packet.encode (fun _ -> 7L) [ W32 (Value 2) ])
            (fill self 0 5L 6L 7L));
      test "a refused template leaves the one before" (fun () ->
          let p = [ Packet.Dword 1; W32 (Value 0) ] in
          load p;
          (try load (List.init 17 (fun i -> Packet.Dword i))
           with Invalid_argument _ -> ());
          equal string (Packet.encode (fun _ -> 5L) p) (fill self 0 5L 6L 7L));
      refused "17 words are refused" (List.init 17 (fun i -> Packet.Dword i));
      refused "5 holes are refused"
        (List.init 5 (fun _ -> Packet.W32 (Value 0)));
      refused "a term of 4 operations is refused"
        [ W32 (Add (Add (Add (Add (Value 0, 1L), 1L), 1L), 1L)) ];
      cases ~name:string_of_int "an argument outside [0;2] is refused"
        [ -1; 3; max_int ] (fun i ->
          raises_match (Exn.invalid_arg ~substring:"Rig_amd.make") (fun () ->
              load [ W32 (Value i) ]));
      cases ~name:string_of_int "a shift outside [0;63] is refused"
        [ -1; 64; 256 ] (fun n ->
          raises_match (Exn.invalid_arg ~substring:"Rig_amd.make") (fun () ->
              load [ W32 (Shift (Value 0, n)) ]));
    ]

(* Launches *)

(* The GPUs of each GC the driver drives, GFX950 with its LDS granule of 1280
   bytes, and the LDS of a workgroup of each. *)
let gpus =
  let gpu ?target gc =
    {
      Gpu.target = Option.value ~default:gc target;
      gc;
      sdma = (6, 0, 0);
      xccs = 1;
      shader_engines = 4;
      compute_units = 32;
      scratch_slots = 32;
    }
  in
  [
    (gpu ~target:(9, 4, 2) (9, 4, 3), 65536);
    (gpu (9, 5, 0), 163840);
    (gpu (11, 0, 0), 65536);
    (gpu (11, 5, 0), 65536);
    (gpu ~target:(12, 0, 1) (12, 0, 1), 65536);
  ]

type launch = {
  g : Gpu.t;
  lds : int;
  k : Code_object.kernel;
  program : int;
  args : int array; (* parameters, scratch, threads, groups *)
  shared : int;
}

let pp_launch ppf l =
  let a, b, c = l.g.gc in
  Format.fprintf ppf
    "GC %d.%d.%d, LDS %d; group %d, private %d, rsrc 0x%x 0x%x 0x%x, wave32 \
     %b, buffer %b; program 0x%x, args [%s], shared %d"
    a b c l.lds l.k.group_segment l.k.private_segment l.k.rsrc1 l.k.rsrc2
    l.k.rsrc3 l.k.wave32 l.k.private_segment_buffer l.program
    (String.concat "; "
       (Array.to_list (Array.map (Printf.sprintf "0x%x") l.args)))
    l.shared

(* Kernels as compilers describe them, whose group segment and shared memory fit
   the GPU's LDS, many of them empty or ending at the LDS. *)
let launches =
  let open Gen in
  let u32 = int_range 0 0xffff_ffff in
  let aligned n = map (fun a -> a * n) (int_range 0 (((1 lsl 48) - 1) / n)) in
  let part n = frequency [ (3, int_range 0 n); (1, of_list [ 0; n ]) ] in
  let side = int_range 1 1024 in
  with_pp pp_launch
    (let* g, lds = of_list gpus in
     let* group = part lds in
     let+ shared = part (lds - group)
     and+ private_segment = int_range 0 (1 lsl 16)
     and+ rsrc1, rsrc2, rsrc3 = triple u32 u32 u32
     and+ wave32, psb = pair bool bool
     and+ program, scratch = pair (aligned 256) (aligned 256)
     and+ params = aligned 64
     and+ tx, ty, tz = triple side side side
     and+ gx, gy, gz = triple u32 u32 u32 in
     let k =
       {
         Code_object.descriptor = 0;
         entry = 0;
         group_segment = group;
         private_segment;
         kernarg_size = 0;
         (* No privilege (bit 20) nor LDS (bits 15 to 23), which AMDGPUUsage
            says the descriptor leaves 0. *)
         rsrc1 = rsrc1 land lnot (1 lsl 20);
         rsrc2 = rsrc2 land lnot (0x1ff lsl 15);
         rsrc3;
         wave32;
         dispatch_ptr = false;
         private_segment_buffer = psb;
         max_threads = 1024;
         hidden = [];
       }
     in
     let args = [| params; scratch; tx; ty; tz; gx; gy; gz |] in
     { g; lds; k; program; args; shared })

let launches_law =
  prop
    "a launch's dispatch is Pm4's, its group segment grown by its shared memory"
    launches (fun l ->
      let lds = l.k.group_segment + l.shared and a = l.args in
      let granule = Pm4.lds_granule l.g in
      let units n = (n + granule - 1) / granule in
      cover "no shared memory" (l.shared = 0);
      cover "shared memory past the group segment's granules"
        (units lds > units l.k.group_segment);
      cover "the GPU's whole LDS" (lds = l.lds);
      cover "a granule of 1280 bytes" (granule = 1280);
      let expected =
        Pm4.dispatch l.g
          { l.k with group_segment = lds }
          ~program:l.program ~scratch:a.(1) ~args:a.(0) ~packet:0
          ~threads:(a.(2), a.(3), a.(4))
          ~groups:(a.(5), a.(6), a.(7))
          ()
      in
      equal string
        (Packet.encode Int64.of_int expected)
        (Rig_amd.dispatch l.g l.k ~program:l.program ~lds:l.lds l.args
           ~shared:l.shared))

(* The driver finds the word that holds a dispatch's LDS as the one word a group
   segment one granule larger changes. On each GPU, a group segment of [n]
   granules changes that word alone, by [n] times that change. *)
let lds_word =
  cases
    ~name:(fun ((g : Gpu.t), _) ->
      let a, b, c = g.gc in
      strf "GC %d.%d.%d" a b c)
    "a dispatch's LDS is one word, linear in its granules" gpus
    (fun (g, lds) ->
      let k =
        {
          Code_object.descriptor = 0;
          entry = 0;
          group_segment = 0;
          private_segment = 0;
          kernarg_size = 0;
          rsrc1 = 0;
          rsrc2 = 0;
          rsrc3 = 0;
          wave32 = true;
          dispatch_ptr = false;
          private_segment_buffer = false;
          max_threads = 1024;
          hidden = [];
        }
      in
      let granule = Pm4.lds_granule g in
      let words n =
        let p =
          Pm4.dispatch g
            { k with group_segment = n * granule }
            ~program:0 ~scratch:0 ~args:0 ~packet:0 ~threads:(1, 1, 1)
            ~groups:(1, 1, 1) ()
        in
        let s = Packet.encode Int64.of_int p in
        Array.init
          (String.length s / 4)
          (fun i ->
            Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)
      in
      let w0 = words 0 and w1 = words 1 in
      let at =
        List.filter
          (fun i -> w0.(i) <> w1.(i))
          (List.init (Array.length w0) Fun.id)
      in
      equal ~msg:"the words one granule changes" int 1 (List.length at);
      let at = List.hd at in
      for n = 0 to lds / granule do
        let wn = words n in
        let expected = Array.copy w0 in
        expected.(at) <- w0.(at) + (n * (w1.(at) - w0.(at)));
        equal ~msg:(strf "%d granules" n) (array int) expected wn
      done)

let launches_group = group "launches" [ lds_word; launches_law ]
let () = exit (run "rig_amd.template" [ templates; launches_group ])
