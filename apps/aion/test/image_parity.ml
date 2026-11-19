(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The image codec against upstream (PolymathicAI/AION db7fba6df53b) with
   AION-1's weights, by the rounding ensemble's rule ([Spread]). It lives
   outside runtest: the weights are 280 MB, the ensemble's files tens of MB, and
   a run takes minutes. It runs under the alias parity, on AION_PARITY's
   huggingface, goldens and subset, and ensemble/image: gen/image_ensemble.py's
   files, one per set: golden, ls_desi, ls_catalog and hsc. A set without its
   file fails. Each holds the set's objects, the float64 reference's values,
   PyTorch's float32 values and the ensemble's spreads at the order model's
   scales; a file <set>-report beside it, where present, holds the contract and
   measured scales' spreads for some of the objects.

   Two checks per set. Float32: the codec as AION-1 ships it, its spread the
   ensemble's at the order model's float32 scale. Float64: the codec cast to
   float64, its spread the ensemble's at float64's scale, where a wrong constant
   or structure shows by orders of magnitude. Each checks the FSQ's input (the
   latent), each number of each token, rounded, against the reference's bounded
   latent, and the decoding of the reference's tokens. It exits 1 when a check
   fails. For the report files' objects it also prints, without gating, the
   float32 codec's worst ratio at the contract scale, and the root mean square
   of (x - reference) / ŝ at PyTorch's measured scale for the codec and for
   PyTorch's float32. *)

open Aion

(* The inputs, under AION_PARITY as the README's Tests section lays them out. *)
let dir =
  match Sys.getenv_opt "AION_PARITY" with
  | Some d when d <> "" -> d
  | _ ->
      prerr_endline
        "image_parity: AION_PARITY is unset: set it to the directory of the \
         README's Tests section";
      exit 1

let path = Filename.concat dir
let printf = Printf.printf
let failures = ref 0
let load path = Nx_io.load_safetensors path

let get dt name a =
  match Nx_io.Archive.find name a with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no tensor " ^ name)

let has a name = Option.is_some (Nx_io.Archive.find name a)
let floats t = Nx.to_array (Nx.cast Nx.float64 t)
let ints t = Array.map Int64.to_int (Nx.to_array (Nx.cast Nx.int64 t))

let take rows t =
  let p =
    Nx.create Nx.int64 [| Array.length rows |] (Array.map Int64.of_int rows)
  in
  Nx.take ~axis:0 p t

(* The latent, [b; 576; 5], from upstream's [b; 5; 576]. *)
let cells t = floats (Nx.moveaxis 1 2 t)

type set = {
  name : string;
  bands : Image.band list;
  images : Nx.host Nx.float32_t;
  e : Nx.host Nx_io.Archive.t;
}

let ls = Image.[ Des_g; Des_r; Des_i; Des_z ]
let hsc = Image.[ Hsc_g; Hsc_r; Hsc_i; Hsc_z; Hsc_y ]

let sets ?(suffix = "") ~goldens ~subset ~ensemble () =
  let rows a = ints (get Nx.int64 "rows" a) in
  let sub name key e =
    take (rows e)
      (get Nx.float32 key
         (load (Filename.concat subset (name ^ ".safetensors"))))
  in
  (* Upstream's test feeds channels 5 to 8, the Legacy Survey's. *)
  let golden e =
    let input = Filename.concat goldens "image_codec_input_batch.safetensors" in
    take (rows e)
      (Nx.slice [ A; R (5, 9) ] (get Nx.float32 "image.array" (load input)))
  in
  let set name bands images =
    let path = Filename.concat ensemble (name ^ suffix ^ ".safetensors") in
    if not (Sys.file_exists path) then begin
      if suffix = "" then incr failures;
      printf "missing ensemble %s\n%!" path;
      []
    end
    else
      let e = load path in
      [ { name; bands; images = images e; e } ]
  in
  List.concat
    [
      set "golden" ls golden;
      set "ls_desi" ls (sub "ls_desi" "legacysurvey.image_array");
      set "ls_catalog" ls (sub "ls_catalog" "image_array");
      set "hsc" hsc (sub "hsc" "image_array");
    ]

(* What a check compares, for every object of a set. *)
type outputs = {
  latent : float array;
  tokens : int array; (* [b; 576] *)
  decoded : float array;
}

let compute (type b) (t : (b, Nx.host) Image.t) (set : set)
    (x : (float, b, Nx.host) Nx.t) =
  {
    latent = floats (Image.latent t set.bands x);
    tokens = ints (Image.encode t set.bands x);
    decoded =
      floats (Image.decode t set.bands (get Nx.int64 "ref.tokens" set.e));
  }

(* Each number of each token, rounded, as [q + 3] for the integer [q] it is
   rounded to: the count of the edges below it of the widest level, seven. *)
let levels = [ 7; 5; 5; 5; 5 ]
let edges = [| -2.5; -1.5; -0.5; 0.5; 1.5; 2.5 |]

let digits tokens =
  Array.concat
    (Array.to_list
       (Array.map
          (fun k ->
            Array.of_list
              (List.rev
                 (snd
                    (List.fold_left
                       (fun (k, ds) l ->
                         (k / l, ((k mod l) - (l / 2) + 3) :: ds))
                       (k, []) levels))))
          tokens))

let check ~gate ~n ~scale (set : set) o =
  let e = set.e in
  let r name = floats (get Nx.float64 ("ref." ^ name) e)
  and s name = floats (get Nx.float64 ("s." ^ scale ^ "." ^ name) e) in
  let checks =
    [
      ( "latent",
        Spread.values ~n
          ~reference:(cells (get Nx.float64 "ref.pre" e))
          ~spread:(cells (get Nx.float64 ("s." ^ scale ^ ".pre") e))
          o.latent );
      ( "token numbers",
        Spread.levels ~n ~edges ~reference:(r "bounded") ~spread:(s "bounded")
          (digits o.tokens) );
      ( "decoded",
        Spread.values ~n ~reference:(r "decoded") ~spread:(s "decoded")
          o.decoded );
    ]
  in
  List.iter
    (fun (name, (c : Spread.check)) ->
      if gate then begin
        if c.failures <> [] then incr failures;
        printf "  %s%s\n"
          (if c.failures = [] then "" else "FAIL ")
          (Format.asprintf "%a" (Spread.pp name) c)
      end
      else
        printf "  %s: worst %.3g at the %s scale (not gating)\n" name c.worst
          scale)
    checks;
  printf "%!"

(* The root mean square of (x - reference) / ŝ over an output. *)
let rms_z ~reference ~spread x =
  let acc = ref 0. and k = ref 0 in
  Array.iteri
    (fun i v ->
      if spread.(i) > 0. then begin
        let z = (v -. reference.(i)) /. spread.(i) in
        acc := !acc +. (z *. z);
        incr k
      end)
    x;
  sqrt (!acc /. float_of_int (max 1 !k))

let measured (set : set) o =
  let e = set.e in
  let f name = floats (get Nx.float64 name e) in
  let latent =
    rms_z
      ~reference:(cells (get Nx.float64 "ref.pre" e))
      ~spread:(cells (get Nx.float64 "s.measured.pre" e))
  and decoded =
    rms_z ~reference:(f "ref.decoded") ~spread:(f "s.measured.decoded")
  in
  printf
    "  measured scale, rms z, codec and PyTorch float32 (not gating): latent \
     %.3g and %.3g, decoded %.3g and %.3g\n"
    (latent o.latent)
    (latent (cells (get Nx.float64 "f32.pre" e)))
    (decoded o.decoded)
    (decoded (f "f32.decoded"))

let () =
  let hub = path "huggingface" and goldens = path "goldens" in
  let subset = path "subset" and ensemble = path "ensemble/image" in
  let t32 = Image.load ~cache_dir:hub ~offline:true () in
  let t64 = Image.cast Nx.float64 t32 in
  List.iter
    (fun (set : set) ->
      let n = (ints (get Nx.int64 "n" set.e)).(0) in
      printf "\n== %s: %d objects, ensembles of %d\n%!" set.name
        (Nx.dim 0 set.images) n;
      let t0 = Unix.gettimeofday () in
      let o32 = compute t32 set set.images in
      printf "float32, order scale:\n";
      check ~gate:true ~n ~scale:"order" set o32;
      let reference = ints (get Nx.int64 "ref.tokens" set.e) in
      let upstream32 = ints (get Nx.int64 "f32.tokens" set.e) in
      let count a = Array.fold_left ( + ) 0 a in
      printf
        "  tokens differing from the float64 reference: %d of %d; from \
         PyTorch's float32: %d\n"
        (count
           (Array.mapi (fun i k -> Bool.to_int (k <> reference.(i))) o32.tokens))
        (Array.length reference)
        (count
           (Array.mapi
              (fun i k -> Bool.to_int (k <> upstream32.(i)))
              o32.tokens));
      printf "float64, order scale at float64:\n";
      check ~gate:true ~n ~scale:"order64" set
        (compute t64 set (Nx.cast Nx.float64 set.images));
      printf "(%.0f s)\n%!" (Unix.gettimeofday () -. t0))
    (sets ~goldens ~subset ~ensemble ());
  List.iter
    (fun (set : set) ->
      let n = (ints (get Nx.int64 "n" set.e)).(0) in
      printf "\n== %s, %d objects: other scales (not gating)\n" set.name
        (Nx.dim 0 set.images);
      let o32 = compute t32 set set.images in
      if has set.e "s.contract.pre" then
        check ~gate:false ~n ~scale:"contract" set o32;
      if has set.e "s.measured.pre" then measured set o32;
      printf "%!")
    (sets ~suffix:"-report" ~goldens ~subset ~ensemble ());
  printf "\nfailing checks: %d\n" !failures;
  exit (if !failures = 0 then 0 else 1)
