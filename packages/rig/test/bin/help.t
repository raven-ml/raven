Each command has a page on standard output, and rig exits 0.

  $ . support/env.sh
  $ rig --help >page
  $ head -2 page
  NAME
         rig - run a job on several machines
  $ rig run --help >page
  $ head -2 page
  NAME
         rig run - start a job on several machines, and restart it when it
  $ rig agent --help >page
  $ head -2 page
  NAME
         rig agent - serve one job on this machine
  $ rig firmware --help >page
  $ head -2 page
  NAME
         rig firmware - fill a directory with the firmware images of GPUs

--help after a command's arguments is that command's page, until --.

  $ rig run --on a,b --help | head -1
  NAME

--version prints the version the package states, which every machine's
rig shares.

  $ rig --version
  1.0.0~alpha3
