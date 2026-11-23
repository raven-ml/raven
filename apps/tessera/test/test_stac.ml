(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord
open Tessera

(* Requests go to http/curl.exe, put on the PATH as curl, or on Windows as
   itself: its directory. It answers from the directory [served] makes:
   recorded or written answers in order, and files served by byte ranges. *)

let rec rm path =
  if Sys.is_directory path then (
    Array.iter (fun e -> rm (Filename.concat path e)) (Sys.readdir path);
    Sys.rmdir path)
  else Sys.remove path

let () =
  let exe = Filename.concat (Sys.getcwd ()) "http/curl.exe" in
  let bin =
    if Sys.win32 then Filename.dirname exe
    else
      let bin = Filename.temp_dir "tessera" "bin" in
      at_exit (fun () -> rm bin);
      Unix.symlink exe (Filename.concat bin "curl");
      bin
  in
  let sep = if Sys.win32 then ";" else ":" in
  Unix.putenv "PATH" (bin ^ sep ^ Sys.getenv "PATH")

let read file = In_channel.with_open_bin file In_channel.input_all
let write file s =
  Out_channel.with_open_bin file (fun oc -> output_string oc s)
let recorded name = read (Filename.concat "http" name)

let rec mkdir_p d =
  if not (Sys.file_exists d) then (
    mkdir_p (Filename.dirname d);
    Sys.mkdir d 0o755)

let host_path url =
  let n = String.length "https://" in
  String.sub url n (String.length url - n)

(* [served ~script ~files f] is [f dir] while curl answers from [dir]: the
   answers [script] in order, and each [(url, bytes, etags)] of [files]. *)
let served ?(script = []) ?(files = []) f =
  let dir = Filename.temp_dir "tessera" "http" in
  Fun.protect
    ~finally:(fun () -> rm dir)
    (fun () ->
      Unix.putenv "FAKE_CURL_DIR" dir;
      let scripts = Filename.concat dir "script" in
      Sys.mkdir scripts 0o755;
      let numbered i = Filename.concat scripts (Printf.sprintf "%03d" i) in
      List.iteri (fun i s -> write (numbered i) s) script;
      let files_dir = Filename.concat dir "files" in
      List.iter
        (fun (url, bytes, etags) ->
          let file = Filename.concat files_dir (host_path url) in
          mkdir_p (Filename.dirname file);
          write file bytes;
          if etags <> [] then
            write (file ^ ".etag") (String.concat "\n" etags))
        files;
      f dir)

type request = { url : string; headers : string list; body : string option }

(* The requests curl was asked, in order. *)
let log dir =
  let file = Filename.concat dir "log" in
  if not (Sys.file_exists file) then []
  else
    read file |> String.split_on_char '\n'
    |> List.filter (( <> ) "")
    |> List.map (fun l ->
        match String.split_on_char '\t' l with
        | first :: rest ->
            let meth, url =
              match String.index_opt first ' ' with
              | Some i ->
                  ( String.sub first 0 i,
                    String.sub first (i + 1) (String.length first - i - 1) )
              | None -> failf "log line %S" l
            in
            let headers, body =
              if meth = "POST" then
                match List.rev rest with
                | b :: hs -> (List.rev hs, Some b)
                | [] -> failf "log line %S" l
              else (rest, None)
            in
            { url; headers; body }
        | [] -> failf "log line %S" l)

let answer ?(headers = []) code body =
  let hs = List.map (fun (k, v) -> Printf.sprintf "%s: %s\r\n" k v) headers in
  Printf.sprintf "HTTP/1.1 %d Status\r\n%sContent-Length: %d\r\n\r\n%s" code
    (String.concat "" hs) (String.length body) body

let retry code = answer ~headers:[ ("Retry-After", "0") ] code ""
let json_answer s =
  answer ~headers:[ ("Content-Type", "application/json") ] 200 s

let decode_json s = Jsont_bytesrw.decode_string Jsont.json s

let decode s =
  match decode_json s with Ok j -> j | Error e -> failwith e

let json = Testable.make ~pp:Jsont.pp_json ~equal:Jsont.Json.equal

(* Search *)

let search_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
let berlin = (13.3, 52.4, 13.5, 52.6)
let day = (2024, 6, 26)
let s1 ?(bbox = berlin) days =
  let json = Stac.search ~collection:Stac.sentinel1_rtc ~bbox days in
  match Jord_stac.items json with Ok items -> items | Error e -> failwith e

let ids = List.map (fun (i : Jord_stac.item) -> i.id)
let acquisition = "S1A_IW_GRDH_1SDV_20240626T0525"
let first_id = acquisition ^ "52_20240626T052617_054490_06A192_rtc"
let second_id = acquisition ^ "27_20240626T052552_054490_06A192_rtc"

(* A page of no item whose links are [links]. *)
let page links =
  json_answer
    (Printf.sprintf {|{"type":"FeatureCollection","features":[],"links":[%s]}|}
       links)

let body r = decode (Option.get r.body)

let search =
  group "search"
    [
      test "names MPC's collections" (fun () ->
          equal (pair string string)
            ("sentinel-2-l2a", "sentinel-1-rtc")
            (Stac.sentinel2_l2a, Stac.sentinel1_rtc));
      test "follows next links to the last page" (fun () ->
          served
            ~script:[ recorded "s1_page1.http"; recorded "s1_page2.http" ]
            (fun dir ->
              equal (list string) [ first_id; second_id ]
                (ids (s1 (day, day)));
              match log dir with
              | [ a; b ] ->
                  equal string search_url a.url;
                  mem string "Content-Type: application/json" a.headers;
                  equal json
                    (decode
                       {|{"collections": ["sentinel-1-rtc"],
                          "bbox": [13.3, 52.4, 13.5, 52.6],
                          "datetime": "2024-06-26T00:00:00Z/2024-06-26T23:59:59.999999Z",
                          "limit": 100}|})
                    (body a);
                  equal string
                    (search_url ^ "?collections=sentinel-1-rtc")
                    b.url;
                  equal json
                    (decode
                       {|{"collections": ["sentinel-1-rtc"],
                          "bbox": [13.3, 52.4, 13.5, 52.6],
                          "datetime": "2024-06-26T00:00:00Z/2024-06-26T23:59:59Z",
                          "limit": 1,
                          "token": "next:sentinel-1-rtc:S1A_IW_GRDH_1SDV_20240626T052552_20240626T052617_054490_06A192_rtc"}|})
                    (body b)
              | l -> failf "%d requests" (List.length l)));
      test "merges a next body that says merge into the request's" (fun () ->
          let next =
            {|{"rel": "next", "method": "POST", "merge": true,
               "href": "https://example.org/search",
               "body": {"token": "t", "limit": 7}}|}
          in
          served
            ~script:[ page next; recorded "s1_page2.http" ]
            (fun dir ->
              equal (list string) [ second_id ] (ids (s1 (day, day)));
              match log dir with
              | [ _; b ] ->
                  equal string "https://example.org/search" b.url;
                  equal json
                    (decode
                       {|{"collections": ["sentinel-1-rtc"],
                          "bbox": [13.3, 52.4, 13.5, 52.6],
                          "datetime": "2024-06-26T00:00:00Z/2024-06-26T23:59:59.999999Z",
                          "limit": 7, "token": "t"}|})
                    (body b)
              | l -> failf "%d requests" (List.length l)));
      test "keeps the items whose time lies in the days" (fun () ->
          (* MPC returns an item whose interval meets the days: an
             acquisition across midnight, its datetime on the day before or
             after, and one stating its interval alone. *)
          let feature id times =
            Printf.sprintf
              {|{"type": "Feature", "stac_version": "1.0.0", "id": %S,
                 "geometry": null, "properties": {%s}, "assets": {},
                 "links": []}|}
              id times
          in
          let interval a b =
            Printf.sprintf {|"start_datetime": %S, "end_datetime": %S|} a b
          in
          let features =
            [
              feature "before"
                ({|"datetime": "2024-06-25T23:59:58Z", |}
                ^ interval "2024-06-25T23:59:45Z" "2024-06-26T00:00:10Z");
              feature "in" {|"datetime": "2024-06-26T05:26:04Z"|};
              feature "from"
                ({|"datetime": null, |}
                ^ interval "2024-06-26T23:59:50Z" "2024-06-27T00:00:15Z");
              feature "after"
                ({|"datetime": "2024-06-27T00:00:02Z", |}
                ^ interval "2024-06-26T23:59:50Z" "2024-06-27T00:00:15Z");
            ]
          in
          let answer =
            json_answer
              (Printf.sprintf
                 {|{"type": "FeatureCollection", "features": [%s],
                    "links": []}|}
                 (String.concat ", " features))
          in
          served ~script:[ answer ] (fun _ ->
              equal (list string) [ "in"; "from" ] (ids (s1 (day, day)))));
      test "follows a next link of no method by GET" (fun () ->
          let next = {|{"rel":"next","href":"https://example.org/page2"}|} in
          served
            ~script:[ page next; recorded "s1_page2.http" ]
            (fun dir ->
              equal (list string) [ second_id ] (ids (s1 (day, day)));
              match log dir with
              | [ _; b ] ->
                  equal string "https://example.org/page2" b.url;
                  is_none b.body
              | l -> failf "%d requests" (List.length l)));
      test "refuses a next link that repeats a request of the search"
        (fun () ->
          let link u = Printf.sprintf {|{"rel": "next", "href": %S}|} u in
          let a = "https://example.org/a" and b = "https://example.org/b" in
          served
            ~script:[ page (link a); page (link b); page (link a) ]
            (fun dir ->
              raises_match (Exn.failure ~sub:b) (fun () -> s1 (day, day));
              equal int 3 (List.length (log dir))));
      test "sends again after 429, 5xx and a reset connection" (fun () ->
          served
            ~script:
              [ "curl-exit 56"; retry 429; retry 503; recorded "s1_page2.http" ]
            (fun dir ->
              equal (list string) [ second_id ] (ids (s1 (day, day)));
              equal int 4 (List.length (log dir))));
      test "gives up after five attempts, naming the status" (fun () ->
          served
            ~script:(List.init 6 (fun _ -> retry 503))
            (fun dir ->
              raises_match (Exn.failure ~sub:"HTTP status 503") (fun () ->
                  s1 (day, day));
              equal int 5 (List.length (log dir))));
      test "gives up at once on a Retry-After past a minute" (fun () ->
          let wait = answer ~headers:[ ("Retry-After", "120") ] 429 "" in
          served
            ~script:[ wait; recorded "s1_page2.http" ]
            (fun dir ->
              raises_match
                (Exn.failure ~sub:"HTTP status 429 (Retry-After 120)")
                (fun () -> s1 (day, day));
              equal int 1 (List.length (log dir))));
      test "raises on a client error without sending again" (fun () ->
          served
            ~script:[ answer 400 "bad bbox"; recorded "s1_page2.http" ]
            (fun dir ->
              raises_match
                (Exn.failure ~sub:(search_url ^ ": HTTP status 400"))
                (fun () -> s1 (day, day));
              equal int 1 (List.length (log dir))));
      test "raises on an answer that is not JSON" (fun () ->
          served
            ~script:[ answer 200 "<html>" ]
            (fun _ ->
              raises_match (Exn.failure ~sub:search_url) (fun () ->
                  s1 (day, day))));
      test "refuses days not of the calendar or past 9999, or out of order"
        (fun () ->
          raises_match (Exn.invalid_arg ~sub:"2023-02-29") (fun () ->
              s1 ((2023, 2, 29), (2023, 3, 1)));
          raises_match (Exn.invalid_arg ~sub:"10000-01-01") (fun () ->
              s1 ((9999, 12, 31), (10000, 1, 1)));
          raises_match (Exn.invalid_arg ~sub:"precedes") (fun () ->
              s1 ((2024, 3, 2), (2024, 3, 1)));
          raises_match (Exn.invalid_arg ~sub:"bbox") (fun () ->
              s1 ~bbox:(0., Float.nan, 1., 1.) (day, day));
          raises_match (Exn.invalid_arg ~sub:"bbox") (fun () ->
              s1 ~bbox:(0., 2., 1., 1.) (day, day)));
    ]

(* Windows *)

let utm33 = Result.get_ok (Crs.of_epsg 32633)
let utm32 = Result.get_ok (Crs.of_epsg 32632)

(* A raster of 200 × 300 cells at 10 m, in 16² Deflate tiles of numbers
   that do not compress: 120 KB, past the first 64 KiB read. *)
let src =
  Grid.v utm33 ~rows:200 ~cols:300 ~origin:(400000., 5800000.)
    ~row_step:(0., -10.) ~col_step:(10., 0.)

let raster ~rows ~cols =
  Nx.init Nx.uint16 [| rows; cols; 1 |] (fun i ->
      Hashtbl.hash (i.(0), i.(1)) land 0xffff)

let cog ?(tags = Jord_geotiff.tags src) ~rows ~cols () =
  let file = Filename.temp_file "tessera" ".tif" in
  Fun.protect
    ~finally:(fun () -> Sys.remove file)
    (fun () ->
      Nx_io.save_tiff ~tile:(16, 16) ~compression:Deflate ~tags file
        (raster ~rows ~cols);
      read file)

let big = cog ~rows:200 ~cols:300 ()
let token_url = "https://planetarycomputer.microsoft.com/api/sas/v1/token/"

(* The sessions' time: the recorded token's start, 2026-11-10T21:40:24Z,
   its [st], as Unix time. *)
let t0 = 1794346824.
let session () = Stac.session ~now:(fun () -> t0)

(* RFC 3339's form of [s] seconds after [t0]. *)
let after s =
  let t = Unix.gmtime (t0 +. s) in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02dZ" (t.tm_year + 1900)
    (t.tm_mon + 1) t.tm_mday t.tm_hour t.tm_min t.tm_sec

(* The token endpoint's answer of a token that expires at [expiry], an hour
   after [t0] by default. *)
let token ?(expiry = after 3600.) () =
  json_answer
    (Printf.sprintf {|{"msft:expiry": %S, "token": "st=x&sig=SECRET"}|}
       expiry)

let s2_blob = "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/"
let s1_blob =
  "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/"

let asset href =
  {
    Jord_stac.href;
    media_type = None;
    roles = [ "data" ];
    grid = None;
    bands = [];
    view =
      {
        off_nadir = None;
        incidence_angle = None;
        azimuth = None;
        sun_azimuth = None;
        sun_elevation = None;
      };
  }

(* [href]'s window for [g], in a session of its own unless [s] is given. *)
let read_window ?s rule href (g : Grid.t) =
  let s = match s with Some s -> s | None -> session () in
  Stac.window s rule (asset href) g.crs (Warp.centres ~at:Nx.Host.on g.crs g)

let rule =
  Gen.of_list
    ~pp:(fun ppf r ->
      Format.pp_print_string ppf
        (match r with Warp.Nearest -> "nearest" | Bilinear -> "bilinear"))
    [ Warp.Nearest; Bilinear ]

(* A grid over, across or beside the raster, rotated or not. *)
let dst =
  let open Gen in
  let+ rows = int_range 0 12
  and+ cols = int_range 0 12
  and+ d = float_range 3. 40.
  and+ x = float_range (-400.) 3400.
  and+ y = float_range (-400.) 2400.
  and+ turn = frequency [ (2, constant 0.); (1, float_range 0. 6.3) ] in
  let c = cos turn and s = sin turn in
  Grid.v utm33 ~rows ~cols
    ~origin:(400000. +. x, 5800000. -. y)
    ~row_step:(-.d *. s, -.d *. c)
    ~col_step:(d *. c, -.d *. s)

let range_of r =
  List.find_map
    (fun h -> Scanf.sscanf_opt h "Range: bytes=%d-%d%!" (fun a b -> (a, b)))
    r.headers

let is_token r = String.starts_with ~prefix:token_url r.url
let tokens dir = List.filter is_token (log dir)
let blobs dir = List.filter (fun r -> not (is_token r)) (log dir)

let sample ?row ?col rule g p x =
  match rule with
  | Warp.Nearest -> Warp.nearest ?row ?col g p x
  | Bilinear -> Warp.bilinear ?row ?col g p x

let ints x = Nx.to_array (Nx.cast Nx.int32 x)

let agrees =
  prop "agrees with the warp on the whole raster" (Gen.pair rule dst)
    (fun (rule, g) ->
      let href = s2_blob ^ "x.tif" in
      served ~script:[ token () ] ~files:[ (href, big, []) ] (fun dir ->
          let w = read_window rule href g in
          let whole = raster ~rows:200 ~cols:300 in
          let p = Warp.pixel src (Warp.centres ~at:Nx.Host.on utm33 g) in
          let row, col, rows, cols = Warp.window rule src p in
          let x = Nx.unpack Nx.uint16 w.samples in
          equal (quad int int int int) (row, col, rows, cols)
            (w.row, w.col, (Nx.shape x).(0), (Nx.shape x).(1));
          let block = Nx.slice [ R (row, row + rows); R (col, col + cols) ] in
          equal (array int32) (ints (block whole)) (ints x);
          is_true ~msg:"the raster's grid" (Grid.equal src w.grid);
          let v, ok = sample ~row:w.row ~col:w.col rule w.grid w.points x in
          let v', ok' = sample rule src p whole in
          equal (array int32) (ints v') (ints v);
          equal (array bool) (Nx.to_array ok') (Nx.to_array ok);
          let rs = blobs dir in
          cover "no cell of the raster" (rows = 0);
          cover "a window past the first read" (List.length rs > 1);
          equal (list string) [ token_url ^ "sentinel2l2a01/sentinel2-l2" ]
            (List.map (fun r -> r.url) (tokens dir));
          let signed = href ^ "?st=x&sig=SECRET" in
          List.iter (fun r -> equal string signed r.url) rs;
          match rs with
          | [] -> fail "no request"
          | first :: rest ->
              equal (list string) [ "Range: bytes=0-65535" ] first.headers;
              let tagged r = mem string "If-Match: \"0x1\"" r.headers in
              List.iter tagged rest;
              let ranges = List.sort compare (List.filter_map range_of rest) in
              let rec apart = function
                | (_, b) :: ((a, _) :: _ as l) ->
                    at_least int ~than:16384 (a - b - 1);
                    apart l
                | _ -> ()
              in
              apart ranges))

(* A grid of the raster's last rows, whose tiles lie past the first read. *)
let bottom = Grid.window ~row:190 ~col:100 ~rows:8 ~cols:8 src
let example = "https://example.org/x.tif"

let windows =
  group "window"
    [
      agrees;
      test "fetches tiles less than 16 KiB apart as one range" (fun () ->
          (* Tiles of 16² uint16 that do not compress, about 520 bytes each,
             in rows of [cols / 16]. A window of one column of tiles reads
             tiles a row of tiles apart: with 2 to a row the tile between
             them is joined to them, with 256 to a row (128 KiB) it is not. *)
          let requests ~rows ~cols ~row ~col =
            let grid =
              Grid.v utm33 ~rows ~cols ~origin:(400000., 5800000.)
                ~row_step:(0., -10.) ~col_step:(10., 0.)
            in
            let file = cog ~tags:(Jord_geotiff.tags grid) ~rows ~cols () in
            let g = Grid.window ~row ~col ~rows:8 ~cols:8 grid in
            served ~files:[ (example, file, []) ] (fun dir ->
                ignore (read_window Nearest example g);
                List.length (log dir))
          in
          equal ~msg:"tall" int 2
            (requests ~rows:4096 ~cols:32 ~row:3004 ~col:20);
          equal ~msg:"wide" int 3
            (requests ~rows:48 ~cols:4096 ~row:12 ~col:3000));
      test "refuses points in another system than the raster's" (fun () ->
          served ~files:[ (example, big, []) ] (fun _ ->
              let c = Warp.centres ~at:Nx.Host.on utm32 bottom in
              let s = session () in
              raises_match (Exn.failure ~sub:example) (fun () ->
                  Stac.window s Nearest (asset example) utm32 c)));
      test "raises Changed when the file changes while it is read" (fun () ->
          let href = s2_blob ^ "changed.tif" in
          served ~script:[ token () ]
            ~files:[ (href, big, [ "\"0x1\""; "\"0x2\"" ]) ]
            (fun _ ->
              raises (Stac.Changed href) (fun () ->
                  read_window Bilinear href bottom)));
      cases ~id:Fun.id "refuses an answer that is not the range asked for"
        [ "whole"; "short"; "span"; "length"; "past" ]
        (fun lie ->
          served ~files:[ (example, big, []) ] (fun dir ->
              write
                (Filename.concat dir ("files/" ^ host_path example ^ ".lie"))
                lie;
              raises_match (Exn.failure ~sub:example) (fun () ->
                  read_window Nearest example bottom)));
      test "refuses a first answer whose range lies past its length"
        (fun () ->
          let cr = ("Content-Range", "bytes 0-9/5") in
          served ~script:[ answer ~headers:[ cr ] 206 "0123456789" ] (fun _ ->
              raises_match (Exn.failure ~sub:example) (fun () ->
                  read_window Nearest example bottom)));
      test "reads a bilinear block across a 360° raster's seam in two parts"
        (fun () ->
          let world =
            Grid.v Crs.lonlat ~rows:20 ~cols:360 ~origin:(-180., 10.)
              ~row_step:(0., -1.) ~col_step:(1., 0.)
          in
          let tags = Jord_geotiff.tags world in
          let file = cog ~tags ~rows:20 ~cols:360 () in
          let g =
            Grid.v Crs.lonlat ~rows:3 ~cols:4 ~origin:(178.6, 6.2)
              ~row_step:(0., -0.7) ~col_step:(0.7, 0.)
          in
          served ~files:[ (example, file, []) ] (fun _ ->
              let w = read_window Bilinear example g in
              let x = Nx.unpack Nx.uint16 w.samples in
              let p =
                Warp.pixel world (Warp.centres ~at:Nx.Host.on Crs.lonlat g)
              in
              let ((row, col, rows, cols) as b) =
                Warp.window Bilinear world p
              in
              equal (quad int int int int) b
                (w.row, w.col, (Nx.shape x).(0), (Nx.shape x).(1));
              is_true ~msg:"across the seam" (col + cols > 360);
              let whole = raster ~rows:20 ~cols:360 in
              let part col cols =
                Nx.slice [ R (row, row + rows); R (col, col + cols) ] whole
              in
              let joined =
                Nx.concatenate ~axis:1
                  [ part col (360 - col); part 0 (col + cols - 360) ]
              in
              equal (array int32) (ints joined) (ints x);
              let v, ok = Warp.bilinear ~row ~col w.grid w.points x in
              let v', ok' = Warp.bilinear world p whole in
              equal (array int32) (ints v') (ints v);
              equal (array bool) (Nx.to_array ok') (Nx.to_array ok)));
      test "reads a whole file answered to the first range" (fun () ->
          let small = cog ~rows:20 ~cols:20 () in
          served ~script:[ answer 200 small ] (fun dir ->
              let g = Grid.window ~row:2 ~col:3 ~rows:10 ~cols:9 src in
              let w = read_window Bilinear example g in
              let x = Nx.unpack Nx.uint16 w.samples in
              equal (array int) [| 11; 10; 1 |] (Nx.shape x);
              equal int 1 (List.length (log dir))));
      test "reads an empty file as one that names no IFD" (fun () ->
          let cr = ("Content-Range", "bytes */0") in
          served ~script:[ answer ~headers:[ cr ] 416 "" ] (fun _ ->
              raises_match (Exn.failure ~sub:example) (fun () ->
                  read_window Nearest example bottom)));
      cases ~id:Fun.id "refuses an href that is not https://, unrequested"
        [ "file:///etc/passwd"; "-K/etc/passwd"; "http://example.org/x" ]
        (fun href ->
          served ~files:[ (example, big, []) ] (fun dir ->
              raises_match (Exn.failure ~sub:"not an https:// URL") (fun () ->
                  read_window Nearest href bottom);
              equal int 0 (List.length (log dir))));
      cases ~id:Fun.id "reads any host but MPC's blob hosts unsigned"
        (* MPC's hosts in another authority, a lookalike host, a fragment,
           and a blob outside a container. *)
        [
          "https://sentinel2l2a01.evil.example?.blob.core.windows.net/sentinel2-l2/x.tif";
          "https://sentinel2l2a01.blob.core.windows.net.evil.example/sentinel2-l2/x.tif";
          "https://evil.sentinel2l2a01.blob.core.windows.net/sentinel2-l2/x.tif";
          "https://other.blob.core.windows.net/sentinel2-l2/x.tif";
          "https://user@sentinel2l2a01.blob.core.windows.net/sentinel2-l2/x.tif";
          "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/x.tif#f";
          "https://sentinel2l2a01.blob.core.windows.net/x.tif";
        ]
        (fun href ->
          served ~script:[ answer 404 "" ] (fun dir ->
              raises_match (Exn.failure ~sub:"HTTP status 404") (fun () ->
                  read_window Nearest href bottom);
              let urls = List.map (fun r -> r.url) (log dir) in
              equal (list string) [ href ] urls));
      test "keeps a token until a minute before it expires" (fun () ->
          (* Each read asks two ranges, each signed. *)
          let asked expiry =
            let href = s1_blob ^ "x.tif" in
            let script = List.init 4 (fun _ -> token ~expiry ()) in
            served ~script ~files:[ (href, big, []) ] (fun dir ->
                let s = session () in
                ignore (read_window ~s Nearest href bottom);
                ignore (read_window ~s Nearest href bottom);
                equal int 4 (List.length (blobs dir));
                List.length (tokens dir))
          in
          equal ~msg:"in 2 minutes" int 1 (asked (after 120.));
          equal ~msg:"in 30 seconds" int 4 (asked (after 30.));
          equal ~msg:"in 2020" int 4 (asked "2020-01-01T00:00:00Z"));
      test "asks for a new token after a 403" (fun () ->
          let href = s2_blob ^ "denied.tif" in
          served ~script:[ token (); token () ] ~files:[ (href, big, []) ]
            (fun dir ->
              write
                (Filename.concat dir ("files/" ^ host_path href ^ ".deny"))
                "1";
              ignore (read_window Nearest href bottom);
              equal int 2 (List.length (tokens dir));
              equal int 3 (List.length (blobs dir))));
      test "signs with the token MPC's endpoint answers" (fun () ->
          let href = s1_blob ^ "x.tif" in
          served ~script:[ recorded "token.http" ] ~files:[ (href, big, []) ]
            (fun dir ->
              ignore (read_window Nearest href bottom);
              let query = "?st=2026-11-10T21%3A40%3A24Z&se=" in
              List.iter
                (fun r ->
                  starts_with ~prefix:(href ^ query) r.url;
                  ends_with ~suffix:"&skv=2025-07-05&sig=REDACTED" r.url)
                (blobs dir)));
      test "refuses a file whose tags state no grid" (fun () ->
          let plain = cog ~tags:[] ~rows:20 ~cols:20 () in
          served ~files:[ (example, plain, []) ] (fun _ ->
              raises_match (Exn.failure ~sub:example) (fun () ->
                  read_window Nearest example bottom)));
      test "names the href of a failed answer" (fun () ->
          served ~script:[ answer 404 "BlobNotFound" ] (fun _ ->
              raises_match (Exn.failure ~sub:(example ^ ": HTTP status 404"))
                (fun () -> read_window Nearest example bottom)));
    ]

(* Sessions *)

module Smap = Map.Make (String)

(* The model: each key's token and expiry; a token is kept while it expires
   more than a minute after the call's time. *)
let model_token key now (t, expiry) m =
  match Smap.find_opt key !m with
  | Some (t', e) when now < e -. 60. -> t'
  | _ ->
      m := Smap.add key (t, expiry) !m;
      t

(* Each domain's time, which the command sets before it asks: the sessions
   read it, so two domains' calls never see each other's time. *)
let clock = Domain.DLS.new_key (fun () -> 0.)

let tokens_stateful =
  let session = abstract "s" in
  let key = Gen.of_list ~pp:Format.pp_print_string [ "a/x"; "b/y" ] in
  let now = Gen.map Float.of_int (Gen.int_range 0 200) in
  let now = Gen.with_pp (fun ppf t -> Format.fprintf ppf "%g" t) now in
  let fetched =
    let open Gen in
    let+ t = Gen.of_list ~pp:Format.pp_print_string [ "t1"; "t2"; "t3" ]
    and+ e = Gen.int_range 0 300 in
    (t, Float.of_int e)
  in
  let fetched =
    Gen.with_pp (fun ppf (t, e) -> Format.fprintf ppf "%s,%g" t e) fetched
  in
  stateful ~domains:2 "a session's tokens behave as a map kept by expiry"
    [
      command "session"
        (Gen.unit @-> makes session)
        (fun () -> ref Smap.empty)
        (fun () -> Stac.session ~now:(fun () -> Domain.DLS.get clock));
      command "token"
        (key @-> now @-> fetched @-> session ^-> returns string)
        model_token
        (fun key now (t, e) s ->
          Domain.DLS.set clock now;
          Stac.token s key (fun () -> (t, e)));
    ]

(* Days of TESSERA's recipe, read through the same stand-in *)

let utm31 = Result.get_ok (Crs.of_epsg 32631)

(* The tile's 8 × 8 cells, on the 10 m lattice; the region all but (7, 7). *)
let cells =
  Grid.v utm31 ~rows:8 ~cols:8 ~origin:(448590., 5416580.) ~row_step:(0., -10.)
    ~col_step:(10., 0.)

let region = Nx.init Nx.bool [| 8; 8 |] (fun i -> not (i.(0) = 7 && i.(1) = 7))
let centres crs = Warp.centres ~at:Nx.Host.on crs cells

(* Rasters one cell of 10 m larger on each side, so that tile cell (i, j) reads
   raster cell (i + 1, j + 1) of a 10 m band, and a 20 m band at its cell's
   quarter points. *)
let raster_grid step n =
  Grid.v utm31 ~rows:n ~cols:n ~origin:(448580., 5416590.)
    ~row_step:(0., -.step) ~col_step:(step, 0.)

let file (type v s) (dt : (v, s) Nx.dtype) step n (f : int -> int -> v) =
  let g = raster_grid step n in
  let x = Nx.init dt [| n; n; 1 |] (fun i -> f i.(0) i.(1)) in
  let path = Filename.temp_file "tessera" ".tif" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      Nx_io.save_tiff ~tile:(16, 16) ~compression:Deflate
        ~tags:(Jord_geotiff.tags g) path x;
      read path)

(* The 05.11-like metadata of a product: Q 10⁴, NODATA 0, SATURATED 65535, and
   [offset] for each band, or none at 03.00. *)
let l2a_metadata ~baseline ~offset =
  let offsets =
    if offset = 0 then ""
    else
      "<BOA_ADD_OFFSET_VALUES_LIST>"
      ^ String.concat ""
          (List.init 13 (fun i ->
               Printf.sprintf
                 {|<BOA_ADD_OFFSET band_id="%d">%d</BOA_ADD_OFFSET>|} i offset))
      ^ "</BOA_ADD_OFFSET_VALUES_LIST>"
  in
  let physical =
    [
      "B1";
      "B2";
      "B3";
      "B4";
      "B5";
      "B6";
      "B7";
      "B8";
      "B8A";
      "B9";
      "B10";
      "B11";
      "B12";
    ]
  in
  let spectral =
    String.concat ""
      (List.mapi
         (fun i b ->
           Printf.sprintf
             {|<Spectral_Information bandId="%d" physicalBand="%s"/>|} i b)
         physical)
  in
  let special t v =
    Printf.sprintf
      "<Special_Values><SPECIAL_VALUE_TEXT>%s</SPECIAL_VALUE_TEXT><SPECIAL_VALUE_INDEX>%d</SPECIAL_VALUE_INDEX></Special_Values>"
      t v
  in
  Printf.sprintf
    {|<?xml version="1.0" encoding="UTF-8"?><n1:Level-2A_User_Product xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd"><n1:General_Info><Product_Info><PROCESSING_BASELINE>%s</PROCESSING_BASELINE></Product_Info><Product_Image_Characteristics>%s%s<QUANTIFICATION_VALUES_LIST><BOA_QUANTIFICATION_VALUE unit="none">10000</BOA_QUANTIFICATION_VALUE></QUANTIFICATION_VALUES_LIST>%s<Spectral_Information_List>%s</Spectral_Information_List></Product_Image_Characteristics></n1:General_Info></n1:Level-2A_User_Product>|}
    baseline (special "NODATA" 0)
    (special "SATURATED" 65535)
    offsets spectral

(* An item of MPC's, its assets served by [files]: [(key, step, bytes)], and the
   metadata [xml] if given. *)
let mpc_item ~id ~time ?(props = []) ?xml ?(bands = "") assets =
  let href key = s2_blob ^ id ^ "/" ^ key ^ ".tif" in
  let asset (key, step, n, _) =
    Printf.sprintf
      {|%S: {"href": %S, "roles": ["data"], "proj:epsg": 32631, "proj:shape": [%d, %d], "proj:transform": [%g, 0, 448580, 0, %g, 5416590]%s}|}
      key (href key) n n step (-.step) bands
  in
  let meta =
    match xml with
    | None -> []
    | Some _ ->
        [
          Printf.sprintf
            {|"product-metadata": {"href": %S, "roles": ["metadata"]}|}
            (s2_blob ^ id ^ "/MTD_MSIL2A.xml");
        ]
  in
  let props =
    String.concat ""
      (List.map (fun (k, v) -> Printf.sprintf {|, %S: %S|} k v) props)
  in
  let json =
    Printf.sprintf
      {|{"type": "Feature", "id": %S, "geometry": null, "properties": {"datetime": %S%s}, "assets": {%s}}|}
      id time props
      (String.concat ", " (meta @ List.map asset assets))
  in
  let it =
    match Result.bind (decode_json json) Jord_stac.item with
    | Ok it -> it
    | Error e -> failwith e
  in
  let files =
    List.map (fun (key, _, _, bytes) -> (href key, bytes, [])) assets
    @ Option.to_list
        (Option.map (fun x -> (s2_blob ^ id ^ "/MTD_MSIL2A.xml", x, [])) xml)
  in
  (it, files)

let tessera_bands =
  [ "B04"; "B02"; "B03"; "B08"; "B8A"; "B05"; "B06"; "B07"; "B11"; "B12" ]

(* A product whose B04 is [b04 r c] and whose band [k] of the others is the
   constant [const k], with scene classes [scl r c] at 20 m. *)
let product ~id ~time ~baseline ~offset ~b04 ~const ~scl =
  let band k name =
    let step =
      if List.mem name [ "B04"; "B02"; "B03"; "B08" ] then 10. else 20.
    in
    let n = if step = 10. then 10 else 5 in
    let f = if k = 0 then b04 else fun _ _ -> const k in
    (name, step, n, file Nx.uint16 step n f)
  in
  mpc_item ~id ~time
    ~xml:(l2a_metadata ~baseline ~offset)
    (("SCL", 20., 5, file Nx.uint8 20. 5 scl) :: List.mapi band tessera_bands)

let blob_urls dir =
  List.filter_map
    (fun r ->
      if String.starts_with ~prefix:s2_blob r.url then
        Some (List.hd (String.split_on_char '?' r.url))
      else None)
    (log dir)

let days =
  group "days"
    [
      test "Sentinel2.day takes each cell from the first valid product"
        (fun () ->
          (* a: valid on raster columns 0 and 1 of 20 m (tile columns 0 to 2),
             its offset -1000, and no B04 at tile cell (1, 1); b: valid
             everywhere, of baseline 03.00; c: never chosen. *)
          let a, fa =
            product ~id:"a" ~time:"2025-07-01T10:00:00Z" ~baseline:"05.11"
              ~offset:(-1000)
              ~b04:(fun r c -> if r = 2 && c = 2 then 0 else 500 + (10 * r) + c)
              ~const:(fun k -> 1100 + (10 * k))
              ~scl:(fun _ c -> if c <= 1 then 4 else 9)
          and b, fb =
            product ~id:"b" ~time:"2025-07-01T10:05:00Z" ~baseline:"03.00"
              ~offset:0
              ~b04:(fun r c -> 2000 + (10 * r) + c)
              ~const:(fun k -> 3000 + k)
              ~scl:(fun _ _ -> 5)
          and c, fc =
            product ~id:"c" ~time:"2025-07-01T10:10:00Z" ~baseline:"05.11"
              ~offset:(-1000)
              ~b04:(fun _ _ -> 7000)
              ~const:(fun _ -> 7000)
              ~scl:(fun _ _ -> 6)
          in
          served
            ~script:(List.init 3 (fun _ -> token ()))
            ~files:(fa @ fb @ fc)
            (fun dir ->
              let s = session () in
              let f =
                match Sentinel2.day s ~roi:region ~centres [ a; b; c ] with
                | Some f -> f
                | None -> fail "no frame"
              in
              let expected i j k =
                if (i = 1 && j = 1) || (i = 7 && j = 7) then 0
                else if j <= 2 then
                  (if k = 0 then 500 + (10 * (i + 1)) + j + 1
                   else 1100 + (10 * k))
                  - 1000
                else if k = 0 then 2000 + (10 * (i + 1)) + j + 1
                else 3000 + k
              in
              equal (array int)
                (Array.init 640 (fun x ->
                     expected (x / 80) (x / 10 mod 8) (x mod 10)))
                (Nx.to_array f.bands);
              equal (array int)
                (Array.init 64 (fun x -> if x = 9 || x = 63 then 0 else 1))
                (Nx.to_array f.mask);
              let urls = blob_urls dir in
              mem string (s2_blob ^ "c/SCL.tif") urls;
              is_false ~msg:"no band of c is read"
                (List.exists
                   (fun u -> String.starts_with ~prefix:(s2_blob ^ "c/B") u)
                   urls)));
      test "Sentinel2.day keeps no frame under 0.01 % and reads no band"
        (fun () ->
          let a, fa =
            product ~id:"a" ~time:"2025-07-01T10:00:00Z" ~baseline:"05.11"
              ~offset:(-1000)
              ~b04:(fun _ _ -> 1500)
              ~const:(fun _ -> 1500)
              ~scl:(fun _ _ -> 9)
          in
          served
            ~script:[ token () ]
            ~files:fa
            (fun dir ->
              is_none (Sentinel2.day (session ()) ~roi:region ~centres [ a ]);
              equal (list string)
                [ s2_blob ^ "a/SCL.tif" ]
                (List.sort_uniq compare (blob_urls dir))));
      test "Sentinel2.day refuses a value outside int16, under a cloud too"
        (fun () ->
          (* DN 35000 breaks ESA's range [1, 2^15 - 1]; at raster cell (5, 5) it
             lies under scene class 4 in [clear], 9 in [cloudy]. *)
          let bright ~id ~cloud =
            product ~id ~time:"2025-07-01T10:00:00Z" ~baseline:"05.11"
              ~offset:(-1000)
              ~b04:(fun r c -> if r = 5 && c = 5 then 35000 else 1500)
              ~const:(fun _ -> 1500)
              ~scl:(fun r c -> if cloud && r = 2 && c = 2 then 9 else 4)
          in
          List.iter
            (fun (id, cloud) ->
              let it, files = bright ~id ~cloud in
              served
                ~script:[ token () ]
                ~files
                (fun _ ->
                  raises_match
                    (Exn.failure ~sub:(id ^ ": B04: a harmonised value"))
                    (fun () ->
                      let s = session () in
                      ignore (Sentinel2.day s ~roi:region ~centres [ it ]))))
            [ ("clear", false); ("cloudy", true) ]);
      test "Sentinel1.day codes covered polarisations and merges them"
        (fun () ->
          let pol key step f = (key, step, 10, file Nx.float32 10. 10 f) in
          let nodata = {|, "raster:bands": [{"nodata": -32768}]|} in
          let s1 id f g =
            mpc_item ~id ~time:"2025-07-01T17:40:00Z" ~bands:nodata
              [ pol "vv" 10. f; pol "vh" 10. g ]
          in
          let p, fp =
            s1 "p"
              (fun _ c -> if c <= 4 then 0.01 else -32768.)
              (fun _ _ -> -32768.)
          and q, fq = s1 "q" (fun _ _ -> 1.) (fun _ _ -> 0.1) in
          served
            ~script:[ token () ]
            ~files:(fp @ fq)
            (fun _ ->
              let frame =
                match
                  Sentinel1.day (session ()) ~roi:region ~centres [ p; q ]
                with
                | Some f -> f
                | None -> fail "no frame"
              in
              (* float32's 0.01 lies below 0.01, at -20.0000001 dB, which codes
                 1999. *)
              let expected x =
                let i = x / 16 and j = x / 2 mod 8 in
                if i = 7 && j = 7 then 0
                else if x mod 2 = 1 then 6000
                else if j <= 3 then 1999
                else 10000
              in
              equal (array int) (Array.init 128 expected) (Nx.to_array frame)));
    ]

let () =
  exit
    (run "tessera.stac"
       [ search; windows; group "session" [ tokens_stateful ]; days ])
