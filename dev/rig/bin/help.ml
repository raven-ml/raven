(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let rig =
  {|NAME
       rig - run a job on several machines

SYNOPSIS
       rig run --on MACHINE,MACHINE[,MACHINE...] -- PROGRAM [ARG...]
       rig agent HOST:PORT
       rig --version

DESCRIPTION
       A job is one program, the controller, that uses the devices of
       several machines: its own machine's, and each other machine's
       through an agent there. The machines of a job fail as one: when one
       of them fails, the job ends on all of them.

       rig run starts a job and starts it again when it fails. rig agent
       is the agent; rig run starts one on each other machine over ssh.

       rig runs on Linux and macOS.

COMMANDS
       run    Start a job on several machines, and restart it when it
              fails.

       agent  Serve one job on this machine.

       'rig COMMAND --help' prints a command's page.

OPTIONS
       --version
           Print rig's version. The rig of every machine of a job must have
           the same.

EXIT STATUS
       124    on misuse of the command line.

       125    on a bug in rig.

       Otherwise as each command's page says.

SEE ALSO
       ssh(1), ssh_config(5)
|}

let run =
  {|NAME
       rig run - start a job on several machines, and restart it when it
       fails

SYNOPSIS
       rig run --on MACHINE,MACHINE[,MACHINE...] -- PROGRAM [ARG...]

DESCRIPTION
       rig run runs PROGRAM on this machine, the first of --on, and an
       agent on each other machine. PROGRAM joins the job by calling
       Rig_remote.launched, which connects it to the agents
       (ENVIRONMENT), so that it uses their machines' devices. Each agent
       serves this program alone, and ends with it.

       A program that exits before it joined the job does so on its own:
       rig run ends the agents and exits with the program's status.

       The job ends in order when the program closes it, or when the
       program exits, an uncaught exception included. rig run then waits
       for the agents to end and exits with the program's status.

       The job fails when one of its processes sees a failure: a device
       lost, a machine silent, a connection broken. Once the program joined
       the job, a process that ends without closing it, killed by a signal
       or exiting from C code, fails it too, and so does a machine whose
       ssh session ends. rig run then gives every process of the job 15
       seconds to end, ends the rest, and names the cause. It waits for
       every machine to answer, and starts the job again: fresh agents, a
       fresh key, and the program from its beginning. rig run gives up
       when the job fails four times in a row before the program joined
       it, whatever the causes, or four times in a row with one cause.

       rig run waits for a machine that does not answer as long as it
       takes, trying every 5 seconds. Interrupt it to stop.

MACHINES
       A machine is a name or an address, an IPv6 address in brackets:
       [fd00::2]. rig run starts each agent over ssh, through the user's
       login shell there:

           ssh MACHINE exec "$SHELL" -lc 'exec rig agent ADDRESS:0'

       ssh must reach every machine without a password or a question, so
       each machine's host key must be known, and the login shell must
       find rig on its PATH, which ~/.profile can extend.

       rig run finds each MACHINE's address as ssh does: the host name that
       'ssh -G MACHINE' prints, resolved here. Its agent listens at that
       address, and the program connects to it, so the job's traffic takes
       the network ssh takes. To run a job over another network of its
       machines, name them in --on by their names or addresses on that
       network.

       The first machine is this one: rig run checks that it can listen at
       an address of that name.

ARGUMENTS
       --on MACHINE,MACHINE[,MACHINE...]
           The job's machines, two or more, each named once. The first is
           this machine. The program gets the other machines' agents in
           this order.

       PROGRAM [ARG...]
           The controller, run with its arguments, found through PATH.

EXIT STATUS
       rig run exits with the program's status once the program closed the
       job, or once it exited before joining it; with 128 + N if signal N
       killed the program. That status may be any of the ones below.

       123    the job could not start, or rig run gave up.

       124    on misuse of the command line.

       125    on a bug in rig.

       On SIGINT, SIGTERM or SIGHUP, rig run sends the same signal to the
       program and gives it 15 seconds to end, its agents still serving it,
       so that it can save its work. Then rig run kills it, ends the
       agents, and is killed by the signal: a shell reports 128 + N.

ENVIRONMENT
       The program runs in rig run's environment, with these variables,
       which Rig_remote.launched reads and removes:

       RIG_REMOTE_AGENTS
           The agents, as NAME=ADDRESS:PORT separated by commas, in the
           order of --on: h100-b=10.0.0.2:41234.

       RIG_REMOTE_KEY
           The job's key, 64 hexadecimal characters, a fresh one at each
           start.

       RIG_REMOTE_REPORT
           The file descriptor on which the program reports to rig run.

       rig run reads:

       PATH
           Where PROGRAM and ssh are found.

OUTPUT
       The program writes to rig run's standard output and error. Each
       line an agent or ssh writes on its standard error comes out on rig
       run's, after the machine's name: "h100-b: ...". rig run writes these
       lines on its standard error:

       rig: job failed: CAUSE
           The job failed. CAUSE is the reason a process of the job gave,
           or that a process crashed: "PROGRAM was killed by SIGNAL",
           "the agent on MACHINE exited with status N", or that a machine
           does not answer: "MACHINE does not answer; waiting for it".

       rig: MACHINE does not answer; waiting for it

       rig: MACHINE runs another agent of this user; waiting for it to end

       rig: MACHINE answers; restarting the job (N of 3)
       rig: restarting the job (N of 3)
           N counts the failures in a row with this cause.

       rig: the job failed 4 times in a row with this cause; giving up
       rig: the job failed 4 times in a row before starting; giving up

       rig: MACHINE: WHY
       rig: PROGRAM: WHY
       rig: the job did not start: WHY
           The job could not start.

       rig: PROGRAM exited with status N before starting the job

       rig: interrupted; ending the job

SECURITY
       The job's processes prove to each other that they hold the job's
       key. ssh carries the key to each machine, and the key never crosses
       the job's own connections or reaches a file. Nothing after the
       proofs is authenticated or encrypted: whoever reads the network
       between the machines reads the job's data, and whoever writes to it
       can change the job's work. Run jobs on a network that only their
       machines read and write, such as a cluster's own.

EXAMPLES
       rig run --on h100-a,h100-b -- ./finetune.exe --config 70b.toml
           Run finetune.exe on h100-a, with an agent on h100-b.

SEE ALSO
       ssh(1), ssh_config(5)
|}

let agent =
  {|NAME
       rig agent - serve one job on this machine

SYNOPSIS
       rig agent HOST:PORT

DESCRIPTION
       rig agent lets one job's controller use this machine's devices: its
       host, and every GPU of every path: METAL, CUDA, NV and AMD through
       their kernel drivers, NV-PCI and AMD-PCI with none. rig run starts it
       on each machine of a job but the first.

       It reads the job's key, the first line of its standard input, and
       runs the agent as a child process, which listens at HOST:PORT. HOST
       is a name or an address, an IPv6 address in brackets: [::1]:7000.
       Port 0 lets the system choose. The agent serves the first controller
       that proves the key, and ends when that job ends. rig agent watches
       it, writes on its standard output how it ended, and ends with it.

       rig agent also ends the job when its standard input ends: over ssh,
       when the session is lost. Keep its input open while the job runs.

       One agent of a user runs on a machine at a time. An agent started
       while another runs waits for it to end.

       NV-PCI and AMD-PCI open the GPUs no kernel driver holds, booting
       them with the firmware in /lib/firmware. rig agent detaches no GPU
       from its driver.

EXIT STATUS
       0      the job ended in order.

       123    the job failed, the agent could not start or crashed, or the
              standard input ended.

       124    on misuse of the command line.

       125    on a bug in rig.

ENVIRONMENT
       TMPDIR
           Where the lock between this user's agents is: /tmp if unset.

       RIG_REMOTE_REPORT
           Set by rig agent for the agent it starts. Not for users.

FILES
       $TMPDIR/rig-agent-UID.lock
           Held by the user's agent on this machine while it runs.

OUTPUT
       rig agent writes these lines on its standard output, and nothing
       else:

       rig-agent VERSION
           First: rig's version. Lines a later version adds come with a
           new version on this line.

       waiting
           Another agent of this user runs on this machine.

       listening HOST:PORT
           The agent listens, at the port the system chose for 0.

       closed
           The job ended in order.

       failed WHY
           The job failed, or the agent could not start.

       died CAUSE
           The agent's process ended without a report: CAUSE is "killed by
           SIGNAL" or "exited with status N".

SECURITY
       See 'rig run --help'.

EXAMPLES
       (cat KEYFILE; exec cat) | rig agent 10.0.0.2:7000
           Serve one job by hand. Control-D ends it.

SEE ALSO
       rig(1), ssh(1)
|}
