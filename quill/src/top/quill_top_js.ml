(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Js_of_ocaml
open Js_of_ocaml_compiler

let setup =
  lazy
    (Topdirs.dir_directory "/static/cmis";
     Toploop.add_directive "enable"
       (Toploop.Directive_string Config.Flag.enable)
       { section = "js_of_ocaml"; doc = "Enable the given flag" };
     Toploop.add_directive "disable"
       (Toploop.Directive_string Config.Flag.disable)
       { section = "js_of_ocaml"; doc = "Disable the given flag" };
     Toploop.add_directive "debug_on" (Toploop.Directive_string Debug.enable)
       { section = "js_of_ocaml"; doc = "Enable debug for the given section" };
     Toploop.add_directive "debug_off" (Toploop.Directive_string Debug.disable)
       { section = "js_of_ocaml"; doc = "Disable debug for the given section" };
     Toploop.add_directive "tailcall"
       (Toploop.Directive_string (Config.Param.set "tc"))
       {
         section = "js_of_ocaml";
         doc = "Set the depth of tail calls before going through a trampoline";
       })

let initialized = ref false

let initialize_toplevel () =
  if not !initialized then (
    Lazy.force setup;
    Quill_top.initialize_toplevel ();
    initialized := true)

let capture_separated f =
  let output_buffer = Buffer.create 100 in
  let error_buffer = Buffer.create 100 in
  let formatter_out = Format.formatter_of_buffer output_buffer in
  let formatter_err = Format.formatter_of_buffer error_buffer in

  let default_stdout_flusher s = Format.printf "%s%!" s in
  let default_stderr_flusher s = Format.eprintf "%s%!" s in

  let output_flusher s = Buffer.add_string output_buffer s in
  let error_flusher s = Buffer.add_string error_buffer s in

  let success_status = ref false in
  Fun.protect
    (fun () ->
      Sys_js.set_channel_flusher stdout output_flusher;
      Sys_js.set_channel_flusher stderr error_flusher;

      success_status := f formatter_out formatter_err)
    ~finally:(fun () ->
      Format.pp_print_flush formatter_out ();
      Format.pp_print_flush formatter_err ();

      Sys_js.set_channel_flusher stdout default_stdout_flusher;
      Sys_js.set_channel_flusher stderr default_stderr_flusher);

  let captured_output = Buffer.contents output_buffer in
  let captured_error = Buffer.contents error_buffer in

  {
    Quill_top.output = captured_output;
    error = (if captured_error = "" then None else Some captured_error);
    status = (if !success_status then `Success else `Error);
  }

let eval code : Quill_top.execution_result =
  initialize_toplevel ();
  capture_separated (fun formatter_out formatter_err ->
      Quill_top.execute true formatter_out formatter_err code)
