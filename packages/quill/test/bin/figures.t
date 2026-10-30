quill run --inplace --figures stores each image as a file in figures/<stem>/
beside the notebook, which references it by that path and says so in its
first line.

  $ cat > nb.md <<'NB'
  > <!-- quill:cell id="c_fig" -->
  > ```ocaml
  > let () =
  >   Format.pp_open_stag Format.std_formatter
  >     (Format.String_tag "quill.display\nimage/svg+xml\n\n<svg/>");
  >   Format.printf "a figure";
  >   Format.pp_close_stag Format.std_formatter ();
  >   Format.printf "@."
  > ```
  > NB
  $ quill run --inplace --figures nb.md > /dev/null
  $ cat nb.md
  <!-- quill:figures files -->
  
  <!-- quill:cell id="c_fig" -->
  ```ocaml
  let () =
    Format.pp_open_stag Format.std_formatter
      (Format.String_tag "quill.display\nimage/svg+xml\n\n<svg/>");
    Format.printf "a figure";
    Format.pp_close_stag Format.std_formatter ();
    Format.printf "@."
  ```
  <!-- quill:output -->
  <!-- out:display image/svg+xml -->
  <img src="figures/nb/c_fig.svg">
  <!-- out:stdout -->
  a figure
  <!-- /quill:output -->
  $ ls figures/nb
  c_fig.svg
  $ cat figures/nb/c_fig.svg
  <svg/>

A run without --figures keeps the storage the notebook has.

  $ quill run --inplace nb.md > /dev/null
  $ grep '<img' nb.md
  <img src="figures/nb/c_fig.svg">

Each notebook has its own figures directory, and cleaning one leaves the
others.

  $ cp nb.md other.md
  $ quill run --inplace other.md > /dev/null
  $ ls figures
  nb
  other
  $ quill clean other.md
  Stripped outputs from other.md
  $ ls figures
  nb
  $ ls figures/nb
  c_fig.svg

quill clean removes the notebook's figures, and figures/ once it is empty.
The notebook still stores its images as files, so the next run writes them
there again.

  $ quill clean nb.md
  Stripped outputs from nb.md
  $ test -e figures
  [1]
  $ head -1 nb.md
  <!-- quill:figures files -->
  $ quill run --inplace nb.md > /dev/null
  $ grep '<img' nb.md
  <img src="figures/nb/c_fig.svg">
  $ ls figures/nb
  c_fig.svg

--figures writes files, so a single notebook needs --inplace.

  $ quill run --figures nb.md
  Error: --figures writes files and needs --inplace
  [1]
