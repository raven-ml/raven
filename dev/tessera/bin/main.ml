(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* main.exe: embeddings of TESSERA tiles by the released v2 models, and their
   check against the upstream reference's outputs.

   encode reads a tile, embeds its pixels and writes GeoTessera's files.
   validate embeds the fixture tiles with each released model and compares the
   result with the reference's, embedding by embedding and, for medium, block by
   block against its layer dumps. *)

open Tessera

let usage =
  {|usage: main.exe encode --model FILE --tile DIR --out NAME [OPTION...]
       main.exe validate --weights DIR --fixtures DIR [OPTION...]

encode options:
  --device D    cpu (eager, the default), cpu-jit, cuda, nv, amd or metal
  --dims K      write the Matryoshka prefix of K values (students only)
  --int8        write GeoTessera's int8 codes and scales
  --bf16        compute the linear layers and attention in bfloat16
  --pixels N    pixels per batch (default 4096)
  --slots S     pixels per row at most (default 32)

validate options:
  --device D, --pixels N, --slots S
                as for encode, for the students
  --models M    comma-separated, of nano, small, medium, large, teacher
                (default the four students)
  --bf16        validate the teacher in bfloat16 too

Memory grows linearly with --pixels. At 4096 pixels a batch takes about
7.7 GB for medium and 11.4 GB for large eager on the CPU, and 3.8 GB and
5.1 GB of a CUDA GPU's memory compiled.|}

let die fmt =
  Printf.ksprintf
    (fun m ->
      prerr_endline ("tessera: " ^ m);
      exit 2)
    fmt

(* Arguments *)

(* A command's [--name value] options and [--flag]s, in any order; any other
   argument is refused. *)
let parse ~options ~flags args =
  let rec go opts = function
    | [] -> opts
    | a :: rest when List.mem a flags -> go ((a, "") :: opts) rest
    | a :: v :: rest when List.mem a options -> go ((a, v) :: opts) rest
    | [ a ] when List.mem a options -> die "%s needs a value\n%s" a usage
    | a :: _ -> die "unknown argument %S\n%s" a usage
  in
  go [] args

let opt opts name = List.assoc_opt name opts

let req opts name =
  match opt opts name with
  | Some v -> v
  | None -> die "%s is required\n%s" name usage

let flag opts name = List.mem_assoc name opts

let int_opt opts name default =
  match opt opts name with
  | None -> default
  | Some v -> (
      match int_of_string_opt v with Some n -> n | None -> die "%s: %S" name v)

(* Devices *)

type 'd target = { at : 'd Nx.Placement.t; jit : bool }
type run = { run : 'd. 'd target -> unit }

let gpus =
  [
    ("cuda", fun () -> Rig.open_ (Rig_cuda.gpu 0));
    ("nv", fun () -> Rig.open_ (Rig_nv_nvidia.gpu 0));
    ("amd", fun () -> Rig.open_ (Rig_amd_amdgpu.gpu 0));
    ("metal", fun () -> Rig.open_ (Rig_metal.gpu 0));
  ]

let on device k =
  match device with
  | "cpu" -> k.run { at = Nx.Host.on; jit = false }
  | "cpu-jit" -> k.run { at = Nx.Host.on; jit = true }
  | gpu -> (
      match List.assoc_opt gpu gpus with
      | None -> die "%S is not a device" gpu
      | Some open_ -> (
          match open_ () with
          | Error e -> die "%s: %s" gpu e
          | Ok g ->
              let module D = (val Nx.devices [ g ]) in
              k.run { at = D.on; jit = true }))

(* A model's forward pass at [t], compiled where [t] asks. *)
let forward (type s d) (t : d target) (m : ((float, s, d) Nx.t, d) Model.t) :
    d Batch.t -> d Nx.float32_t =
  if not t.jit then Model.apply m
  else
    let model = Nx.Ptree.instantiate (module Model) in
    let batch = Nx.Ptree.v Batch.walk in
    let f =
      Rune.jit Nx.Ptree.(model @-> batch @-> returns tensor) Model.apply
    in
    f m

let place (type d) (t : d target) m = Nx.Ptree.place (module Model) t.at m
let bf16 m = Nx.Ptree.cast (module Model) Nx.bfloat16 m

(* encode *)

let encode opts =
  let file = req opts "--model" and tile = req opts "--tile" in
  let out = req opts "--out" in
  let device = Option.value ~default:"cpu" (opt opts "--device") in
  let shape =
    Batch.typical
      ~pixels:(int_opt opts "--pixels" 4096)
      ~slots:(int_opt opts "--slots" 32)
  in
  let m = Model.load file in
  let dims =
    match opt opts "--dims" with
    | None -> Model.dims m
    | Some k -> (
        match int_of_string_opt k with
        | Some k when List.mem k m.prefixes -> k
        | _ ->
            die "--dims %s: the model's prefixes are [%s]" k
              (String.concat "; " (List.map string_of_int m.prefixes)))
  in
  let px, grid = Pixels.of_tile tile in
  let embed t =
    if flag opts "--bf16" then
      Embedding.encode ~shape t.at (forward t (place t (bf16 m))) px
    else Embedding.encode ~shape t.at (forward t (place t m)) px
  in
  on device
    {
      run =
        (fun t ->
          let e = embed t in
          let e = Nx.slice [ Nx.A; Nx.R (0, dims) ] e in
          if flag opts "--int8" then Embedding.save_int8 out ~grid e
          else Embedding.save out ~grid e);
    }

(* validate *)

let f64 x = Nx.to_array (Nx.cast Nx.float64 x)
let load_f32 file = Nx.unpack Nx.float32 (Nx_io.load_npy file)
let checks = ref 0
let failed = ref []

let report ~name ~pass text =
  Printf.printf "%-48s %s %s\n%!" name (if pass then "ok  " else "FAIL") text;
  incr checks;
  if not pass then failed := name :: !failed

(* A comparison no bound judges, reported for its numbers. *)
let note ~name text = Printf.printf "%-48s %s %s\n%!" name "info" text

(* Tolerances: twice the reference's spread against itself over devices, batch
   sizes and fused paths (max |Δ| 9.8e-6, 1 − cos 2e-12). *)
let max_abs_tol = 2e-5
let cos_tol = 1e-11
let int8_share_tol = 1e-4

(* Max |Δ| and least cosine over the rows [valid] marks, rows of [dims]. *)
let compare_rows ~dims ~valid a b =
  let worst = ref 0. and least = ref 1. in
  Array.iteri
    (fun p v ->
      if v then begin
        let dot = ref 0. and na = ref 0. and nb = ref 0. in
        for c = 0 to dims - 1 do
          let x = a.((p * dims) + c) and y = b.((p * dims) + c) in
          worst := Float.max !worst (Float.abs (x -. y));
          if Float.is_nan x || Float.is_nan y then worst := Float.nan;
          dot := !dot +. (x *. y);
          na := !na +. (x *. x);
          nb := !nb +. (y *. y)
        done;
        least := Float.min !least (!dot /. Float.sqrt (!na *. !nb))
      end)
    valid;
  (!worst, !least)

(* Compares [e] with [ref_file], judged at [tol] if given. *)
let validate_embedding ~label ~valid ?tol e ref_file =
  let r = f64 (load_f32 ref_file) in
  let dims = Nx.dim 1 e in
  let mine = f64 e in
  let worst, least = compare_rows ~dims ~valid mine r in
  let undefined_nan =
    let ok = ref true in
    Array.iteri
      (fun p v ->
        if not v then
          for c = 0 to dims - 1 do
            if not (Float.is_nan mine.((p * dims) + c)) then ok := false
          done)
      valid;
    !ok
  in
  let text =
    Printf.sprintf "max|d| %.3g  1-cos %.3g  undefined NaN %b" worst
      (1. -. least) undefined_nan
  in
  match tol with
  | None -> note ~name:label text
  | Some (abs_tol, cos_tol) ->
      let pass = worst <= abs_tol && 1. -. least <= cos_tol && undefined_nan in
      report ~name:label ~pass text

let validate_int8 ~label ~valid e ~codes ~scales =
  let c, s = Embedding.quantize e in
  let c = Nx.to_array c and s = f64 s in
  let rc = Nx.to_array (Nx.unpack Nx.int8 (Nx_io.load_npy codes)) in
  let rs = f64 (load_f32 scales) in
  let dims = Nx.dim 1 e in
  let differ = ref 0 and total = ref 0 and worst = ref 0 and scale = ref 0. in
  let undefined_ok = ref true in
  Array.iteri
    (fun p v ->
      let codes = Array.sub c (p * dims) dims in
      if not v then
        undefined_ok :=
          !undefined_ok && Float.is_nan s.(p) && Array.for_all (( = ) 0) codes
      else begin
        scale :=
          Float.max !scale (Float.abs (s.(p) -. rs.(p)) /. Float.abs rs.(p));
        Array.iteri
          (fun k code ->
            let d = abs (code - rc.((p * dims) + k)) in
            incr total;
            if d > 0 then incr differ;
            worst := max !worst d)
          codes
      end)
    valid;
  let share = Float.of_int !differ /. Float.of_int !total in
  let pass = !worst <= 1 && share <= int8_share_tol && !undefined_ok in
  report ~name:label ~pass
    (Printf.sprintf
       "codes differing %d / %d (max %d)  scale rel %.3g  undefined 0, NaN %b"
       !differ !total !worst !scale !undefined_ok)

let student_names = [ "nano"; "small"; "medium"; "large" ]

let weight_file dir = function
  | "teacher" -> Filename.concat dir "tessera_v2_2B_teacher.pt"
  | m -> Filename.concat dir ("student_" ^ m ^ ".pt")

let tiles = [ "small_64"; "nodesc_32" ]

let valid_of fixtures tile =
  let v =
    Nx.unpack Nx.uint8
      (Nx_io.load_npy (Filename.concat fixtures ("emb/" ^ tile ^ "/valid.npy")))
  in
  Array.map (fun x -> x <> 0) (Nx.to_array v)

(* The reference's name for a device's fixtures. *)
let ref_device = function "cpu" | "cpu-jit" -> "cpu" | _ -> "cuda"

(* Blocks *)

(* The position of each dumped pixel, from the dumps' index.json. *)
let dumped_pixels dir =
  let text =
    In_channel.with_open_bin
      (Filename.concat dir "index.json")
      In_channel.input_all
  in
  let find from sub =
    let n = String.length sub in
    let rec go i =
      if i + n > String.length text then die "%s: no %S" dir sub
      else if String.sub text i n = sub then i + n
      else go (i + 1)
    in
    go from
  in
  let number from key =
    let i = find from (Printf.sprintf "\"%s\": " key) in
    let j = ref i in
    while !j < String.length text && text.[!j] >= '0' && text.[!j] <= '9' do
      incr j
    done;
    int_of_string (String.sub text i (!j - i))
  in
  Sys.readdir dir |> Array.to_list
  |> List.filter (fun n -> Sys.is_directory (Filename.concat dir n))
  |> List.sort compare
  |> List.map (fun n ->
      let i = find 0 (Printf.sprintf "\"%s\": {" n) in
      (n, number i "y", number i "x"))

(* Row [i]'s first [k] tokens of a [[rows; width; …]] block, as [k] rows of [c]
   values. *)
let tokens_of block i k =
  let c = Nx.numel block / (Nx.dim 0 block * Nx.dim 1 block) in
  f64 (Nx.reshape [| k; c |] (Nx.slice [ Nx.I i; Nx.R (0, k) ] block))

let max_abs a b =
  let w = ref 0. in
  Array.iteri (fun i x -> w := Float.max !w (Float.abs (x -. b.(i)))) a;
  !w

(* Token values as multisets: the reference's copies against our counts. *)
let same_multiset ~c dump mine counts =
  let tbl = Hashtbl.create 64 in
  let key a i = Array.to_list (Array.sub a (i * c) c) in
  for i = 0 to (Array.length dump / c) - 1 do
    let k = key dump i in
    Hashtbl.replace tbl k (1 + Option.value ~default:0 (Hashtbl.find_opt tbl k))
  done;
  Array.iteri
    (fun i m ->
      let k = key mine i in
      Hashtbl.replace tbl k
        (Option.value ~default:0 (Hashtbl.find_opt tbl k) - m))
    counts;
  Hashtbl.fold (fun _ v ok -> ok && v = 0) tbl true

let input name = name = "s2_in" || name = "s1_in"

(* Block tolerance: relative to the largest magnitude of the reference's block,
   the reference's own float32 spread (1e-5 of the output) with a margin for
   blocks that are not normalised. *)
let block_tol = 1e-4

let validate_blocks (type s d) (t : d target)
    (m : ((float, s, d) Nx.t, d) Model.t) ~fixtures ~device =
  let dir =
    Filename.concat fixtures
      (Printf.sprintf "layers/medium__%s__noflag" (ref_device device))
  in
  let pixels = dumped_pixels dir in
  let n = List.length pixels in
  let px, (_, w) = Pixels.of_tile (Filename.concat fixtures "tiles/small_64") in
  let indices =
    Nx.create Nx.int64 [| n |]
      (Array.of_list
         (List.map (fun (_, y, x) -> Int64.of_int ((y * w) + x)) pixels))
  in
  let sub =
    {
      Pixels.s2 = Nx_ragged.take ~indices px.s2;
      s1 = Nx_ragged.take ~indices px.s1;
    }
  in
  let shape =
    { Batch.pixels = n; s2_rows = n; s1_rows = n; slots = 1; width = 256 }
  in
  let b, _ = Batch.pack shape sub (Batch.v2_counts sub) ~from:0 in
  let blocks =
    let bt = Batch.place t.at b in
    if not t.jit then Model.blocks m bt
    else
      let names = ref [] in
      let model = Nx.Ptree.instantiate (module Model) in
      let batch = Nx.Ptree.v Batch.walk in
      let f =
        Rune.jit
          Nx.Ptree.(model @-> batch @-> returns (list tensor))
          (fun m b ->
            let l = Model.blocks m b in
            names := List.map fst l;
            List.map snd l)
      in
      let vs = f m bt in
      List.combine !names vs
  in
  let blocks = List.map (fun (k, v) -> (k, Nx.place Nx.Host.on v)) blocks in
  let tokens (r : Nx.host Batch.rows) i =
    Int32.to_int (Nx.item [ i; 0 ] r.stop)
  in
  let counts (r : Nx.host Batch.rows) i k =
    Array.init k (fun j ->
        Float.to_int (Float.round (Float.exp (Nx.item [ i; j ] r.log_count))))
  in
  List.iter
    (fun (name, v) ->
      let worst = ref 0. and scale = ref 0. and exact = ref true in
      List.iteri
        (fun i (pixel, _, _) ->
          let file =
            Filename.concat dir (Filename.concat pixel (name ^ ".npy"))
          in
          if Sys.file_exists file then begin
            let dump = f64 (load_f32 file) in
            let sensor = String.sub name 0 2 in
            let per_token =
              (sensor = "s2" || sensor = "s1")
              && not (String.ends_with ~suffix:"pooled" name)
            in
            let mine, dump =
              if per_token then begin
                let r = if sensor = "s2" then b.s2 else b.s1 in
                let k = tokens r i in
                let mine = tokens_of v i k in
                let c = Array.length mine / k in
                if input name then
                  exact :=
                    !exact
                    && same_multiset ~c dump mine (counts r i k)
                    && max_abs mine (Array.sub dump 0 (k * c)) = 0.;
                (mine, Array.sub dump 0 (k * c))
              end
              else (f64 (Nx.slice [ Nx.I i ] v), dump)
            in
            Array.iter (fun x -> scale := Float.max !scale (Float.abs x)) dump;
            worst := Float.max !worst (max_abs mine dump)
          end)
        pixels;
      let tol = block_tol *. Float.max 1. !scale in
      let pass = !worst <= tol && !exact in
      report
        ~name:(Printf.sprintf "medium/%s/blocks/%s" device name)
        ~pass
        (Printf.sprintf "max|d| %.3g  max|ref| %.3g%s" !worst !scale
           (if input name then
              Printf.sprintf "  tokens and counts exact %b" !exact
            else "")))
    blocks

(* Main *)

(* The teacher's reference has one float32 fixture and two bfloat16 ones of
   equal bits, so it has no measured spread of its own at either precision.
   Float32 is judged at the students' float32 bound; bfloat16 against bfloat16
   autocast at twice the reference's bfloat16-against-float32 distance (max |Δ|
   2.9e-2, 1 − cos 8.8e-6), the most rounding to bfloat16 moved it. A pair
   across precisions is reported, not judged. *)
let teacher_bf16_tol = (2. *. 2.9e-2, 2. *. 8.8e-6)

let validate_teacher (type d) (t : d target) m ~fixtures ~bf16:with_bf16 =
  if not t.jit then die "the teacher validates on a GPU";
  let tile = "small_64" in
  let px, _ = Pixels.of_tile (Filename.concat fixtures ("tiles/" ^ tile)) in
  let valid = valid_of fixtures tile in
  let shape = Batch.typical ~pixels:512 ~slots:32 in
  let reference kind =
    Printf.sprintf "%s/emb/%s/teacher__cuda__%s__bp512__sdpa.npy" fixtures tile
      kind
  in
  let check label e ~same ~tol ~other =
    let name kind = Printf.sprintf "teacher/%s/%s vs %s" label tile kind in
    validate_embedding ~label:(name same) ~valid ~tol e (reference same);
    validate_embedding ~label:(name other) ~valid e (reference other)
  in
  let e = Embedding.encode ~shape t.at (forward t (place t m)) px in
  check "fp32" e ~same:"fp32" ~tol:(max_abs_tol, cos_tol) ~other:"bf16-autocast";
  Gc.full_major ();
  if with_bf16 then begin
    let e = Embedding.encode ~shape t.at (forward t (place t (bf16 m))) px in
    check "bf16" e ~same:"bf16-autocast" ~tol:teacher_bf16_tol ~other:"fp32"
  end

let validate_student (type d) (t : d target) m ~fixtures ~shape ~device name =
  let m = place t m in
  let f = forward t m in
  List.iter
    (fun tile ->
      let px, _ = Pixels.of_tile (Filename.concat fixtures ("tiles/" ^ tile)) in
      let e = Embedding.encode ~shape t.at f px in
      let valid = valid_of fixtures tile in
      let base =
        Printf.sprintf "%s/emb/%s/%s__%s__fp32__bp4096__fast" fixtures tile name
          (ref_device device)
      in
      let label = Printf.sprintf "%s/%s/%s" name device tile in
      validate_embedding ~label ~valid ~tol:(max_abs_tol, cos_tol) e
        (base ^ ".npy");
      validate_int8 ~label:(label ^ "/int8") ~valid e
        ~codes:(base ^ "_int8.npy") ~scales:(base ^ "_scales.npy"))
    tiles;
  if name = "medium" then validate_blocks t m ~fixtures ~device

let validate opts =
  let weights = req opts "--weights" and fixtures = req opts "--fixtures" in
  let device = Option.value ~default:"cpu" (opt opts "--device") in
  let models =
    match opt opts "--models" with
    | None -> student_names
    | Some l ->
        let ms = String.split_on_char ',' l in
        List.iter
          (fun m ->
            if not (List.mem m ("teacher" :: student_names)) then
              die "--models: %S is not a model\n%s" m usage)
          ms;
        ms
  in
  let shape =
    Batch.typical
      ~pixels:(int_opt opts "--pixels" 4096)
      ~slots:(int_opt opts "--slots" 32)
  in
  let one t name =
    match Model.load (weight_file weights name) with
    | exception Failure why -> report ~name ~pass:false why
    | m when name = "teacher" ->
        validate_teacher t m ~fixtures ~bf16:(flag opts "--bf16")
    | m -> validate_student t m ~fixtures ~shape ~device name
  in
  on device
    {
      run =
        (fun t ->
          List.iter
            (fun name ->
              one t name;
              Gc.full_major ())
            models);
    };
  Printf.printf "%d checks, %d failed\n" !checks (List.length !failed);
  List.iter (Printf.printf "failed: %s\n") (List.rev !failed);
  exit (if !failed = [] then 0 else 1)

let () =
  match List.tl (Array.to_list Sys.argv) with
  | "encode" :: args ->
      encode
        (parse ~flags:[ "--int8"; "--bf16" ]
           ~options:
             [
               "--model";
               "--tile";
               "--out";
               "--device";
               "--dims";
               "--pixels";
               "--slots";
             ]
           args)
  | "validate" :: args ->
      validate
        (parse ~flags:[ "--bf16" ]
           ~options:
             [
               "--weights";
               "--fixtures";
               "--device";
               "--models";
               "--pixels";
               "--slots";
             ]
           args)
  | ("-h" | "--help" | "help") :: _ -> print_endline usage
  | _ -> die "%s" usage
