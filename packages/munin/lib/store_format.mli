(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The files of a munin store.

    A {e store} is a directory, its {e root}, that holds the record of runs
    and artifacts as plain files: JSON (RFC 8259), JSON Lines and UTF-8
    text. This module specifies every file, so that a program that follows
    it reads a store without munin, with [cat], [jq] or [tail -f]. The rest
    of this library is one writer and reader of the files, and {!version}
    is the format it reads and writes.

    A {e run} is one execution of a program that records what it does. Its
    {e log} is a file of {e records}, one JSON object a line, appended by
    the run's {e writer} alone. The first record states how the run
    started, and the last, once the run has ended, how it ended. A run also
    has {e tags} and {e notes}, which any process may change while the run
    is written and after it ended. An {e artifact} is a name with numbered
    {e versions}. A version's content is a {e blob}, named by the digest of
    its bytes. An {e alias} names one version of an artifact and may later
    name another.

    {1:layout Layout}

{v
<root>/
  runs/<id>/events.jsonl            the run's log
  runs/<id>/tags/<digest>           one of the run's tags
  runs/<id>/notes                   the run's notes
  runs/<id>/git.diff                uncommitted changes at its start
  runs/<id>/stats.json              its keys' statistics, once ended
  runs/<id>/writer                  its latest writer's identities
  runs/<id>/files/<key>/<file>      a file the run logged
  artifacts/<name>/<n>.json         version <n> of the artifact <name>
  artifacts/<name>/aliases/<alias>  the version <alias> names
  blobs/blake2b-256/<hex>           the content of versions
  lock                              orders collection after publication
v}

    A file of the layout that does not exist holds nothing, such as no tags
    or no notes. A file or directory whose name starts with [.] is a
    writer's work in progress. Readers ignore it, as they ignore every name
    the layout does not list. A writer creates a directory when it first
    writes in it, and reading a store creates nothing.

    {1:names Names}

    A {e text} is a non-empty UTF-8 string with no control character
    (U+0000 to U+001F, U+007F). Experiments, run names, and the keys of
    params, summaries and metadata are texts. A {e tag} is a text that
    holds no [,], so that a list of tags joined by [,] splits back.

    A {e key} names a metric, a reading or a run's file. It is one or more
    {e segments} joined by [/], such as ["train/loss"], where a segment is
    a text that holds no [/] and does not start with [.]. An artifact
    {e name} and an {e alias} are one segment each.

    A run's {e id} is its directory's name, unique in the store. Readers
    compare ids and never parse them. munin draws
    [YYYYMMDDTHHMMSSZ-xxxxxxxx], the start time in UTC and 32 random bits
    in hexadecimal, so that [ls runs] lists runs in the order they started,
    and claims the id by creating the directory.

    {1:values Numbers and values}

    Every number munin writes reads back exactly. A float is written in the
    fewest significant digits from 15 that read back to it, with a fraction
    or an exponent even when it is integral: [0.1], [1.0], [-0.0],
    [1e+16]. JSON has no number for NaN or the infinities, which are
    written [{"float":"NaN"}], [{"float":"Infinity"}] and
    [{"float":"-Infinity"}] wherever a float may appear. A NaN's sign and
    payload are not kept. An integer [n] with [-2{^53} <= n <= 2{^53}] is
    written as its decimal, and a larger one as [{"int":"<decimal>"}]. A
    reader takes a number written with a fraction or an exponent as a
    float, and one written without either as an integer.

    A {e time} is seconds since the Unix epoch, a float. A record's [time]
    is the writer's wall clock when it wrote the record, except in a
    [metrics] record, whose time the program may give for samples it took
    earlier. No rule of this format orders samples or judges a run's
    liveness by a record's time. When a run last wrote is its log's
    modification time ({!section-status}). A {e step} is an integer from
    [0] to [2{^53}]. A {e value} is a boolean, an integer, a float or a
    UTF-8 string. An {e object of values} maps texts to values, each text
    once.

    Every string a writer writes is UTF-8. The program's own strings must
    be, and a writer refuses one that is not. A [provenance] and the
    [error] of an [end], which come from the system, are written with each
    byte that starts no valid UTF-8 sequence replaced by U+FFFD.

    {1:log The log}

    [events.jsonl] holds one record per line: a JSON object, then a line
    feed. Every record has a member [type], which names its kind, and a
    member [time]. The first record is a [start]. The kinds are:

    - [start]: [format], the {!version} of the run's files; [experiment], a
      text; [name], a text, and [parent], the id of another run of the
      store, when given; [steps], the largest step the run will log, when
      declared; [params], an object of values; and [provenance], below.
    - [metric]: declares the metric [key], with [goal], ["minimize"] or
      ["maximize"], and [step_metric], another metric's key, when given. A
      later [metric] record of the same key replaces this one.
    - [metrics]: [step], and [values], an object that maps keys to floats.
      Each member is a {e sample} of the metric of its key at [step] and
      [time].
    - [reading]: [step] and [values], as in [metrics]. Its samples are
      what readings of the machine or its devices gave at [time], and
      [step] is that of the last [metrics] or [resume] record before it,
      or [0], so that readers can align them with the metrics.
    - [file]: [key], [step] and [path], the path of a file the run logged,
      relative to the run's directory ({!section-files}).
    - [summary]: [values], an object of values the program states about the
      run. Each member replaces the run's earlier value of its key.
    - [use]: [name] and [version]. The run used that artifact version.
    - [resume]: [step] and [provenance]. A new writer continues the run
      from [step] ({!section-resume}).
    - [end]: [status], ["finished"], ["failed"] or ["killed"], and [error],
      a string, when failed. An [end] is the last record of its log.

    [provenance] is an object of these members: [command], an array of
    strings; [cwd]; [hostname], when known; [pid]; [ocaml] and [raven], the
    versions of the OCaml compiler and of raven that built the writer;
    [os], the system's name and release; [arch], its processor
    architecture; [cpu], its processor model, when known; [git], when
    [cwd] is in a git work tree, an object of [commit] and [dirty], a
    boolean; [seeds], an object that maps texts to the integers the
    program named as its seeds; and [env], an object that maps the
    environment variables the program asked to record to their values.

    In one run a key is a {e metric}'s, named by [metric] and [metrics]
    records, or a {e reading}'s, named by [reading] records, and never
    both. A writer refuses to write a key of the other kind, and a resumed
    writer learns the kinds from the log ({!section-resume}).

    A log, with its longer records shown over several lines:

{v
{"type":"start","time":1793541787.25,"format":1,"experiment":"mnist",
 "steps":999,"params":{"lr":0.001,"batch":64,"init":{"float":"NaN"}},
 "provenance":{"command":["./train.exe"],"cwd":"/home/u/p",...}}
{"type":"metric","time":1793541787.5,"key":"train/loss","goal":"minimize"}
{"type":"metrics","time":1793541788.25,"step":0,"values":{"train/loss":2.3}}
{"type":"reading","time":1793541802.5,"step":7,
 "values":{"sys/load_per_core":0.5}}
{"type":"resume","time":1793545000.5,"step":400,"provenance":{...}}
{"type":"end","time":1793549000.25,"status":"finished"}
v}

    {1:reading Reading a log}

    A line is {e whole} once its line feed is written, if it holds no NUL
    byte. A reader reads a log's whole lines in order and stops at the
    first line that is not whole. The bytes from there on are still being
    written, and it reads them once they form whole lines. A record never
    holds a NUL byte, and a reader meets one only where the file system
    shows a part of the log not yet written to the storage it reads
    ({!section-network}, {!section-crashes}). A reader that meets one stops
    there until the file system fills the line or a resume cuts it off.

    A whole line that is not a record of this format is {e damaged}. A
    reader skips it, counts it, and never takes a value from it. Members
    that a record's kind does not define are ignored. A sample of a key
    that an earlier record gave the other kind ({!section-log}) is damaged
    too, and so is the record that holds it. A run whose [start]
    holds a [format] other than {!version} is unreadable to this library.

    A sample of a [metrics] record, or a [file] record, is an {e orphan}
    when a [resume] record after it names a step below its step. The run
    went back to a state saved at that step, so the orphan belongs to a
    history the restart abandoned. Readers leave orphans out of a metric's
    series and statistics and out of the run's files, and the log keeps
    them. Samples of [reading] records are never orphans, since the
    machine's state at a time happened whatever the run's steps did.

    A key's {e latest} sample is its last sample in log order that is not
    an orphan. When the latest [metric] record of a key [k] names a
    [step_metric] [m], the sample of [k] at step [s] pairs with the latest
    sample of [m] at step [s], and a sample of [k] at a step where [m] has
    none has no value on [m]'s axis.

    {1:status The writer and the status of a run}

    A run has one writer at a time. The writer holds an exclusive lock on
    the log for its whole life, with [flock] on POSIX systems and, on
    Windows, with [LockFileEx] on the byte at offset 2{^62}. It takes the
    lock before it writes the [start], and opens the log so that the
    programs it starts do not inherit it. The system releases the lock when
    the writer's process ends, however it ends.

    The {e boot identity} is the text that names the kernel's current
    boot: Linux's [/proc/sys/kernel/random/boot_id] and macOS's
    [kern.bootsessionuuid]. The {e machine identity} is the 64 lowercase
    hexadecimal digits of the BLAKE2b-256 digest, keyed with the ASCII
    bytes [munin], of the bytes that name the machine: on Linux the 32
    hexadecimal digits of [/etc/machine-id] without its line feed, and on
    macOS [IOPlatformUUID] as IOKit returns it. A missing or empty
    [/etc/machine-id] gives no machine identity. The machine's own
    identity is not written, since it is confidential to the machine. No
    source of either identity is chosen on Windows yet, so a writer there
    records neither.

    [writer] names who writes the run now: an object of [machine] and
    [boot], the machine and boot identities of the run's latest writer,
    each when its system has one. A writer, holding the lock, replaces it
    whole, as it replaces notes ({!section-annotations}), before it
    appends its [start] or [resume] record.

    A lock is visible to the processes of the kernel that holds it, so a
    reader {e probes} a run's lock only when its boot identity equals the
    [boot] of [writer]. Containers on one kernel share its boot identity
    and its locks, whatever their machine identities, so they probe each
    other's runs. A probe takes a shared lock without waiting and releases
    it at once.

    A run's {e status} is a reading. A reader that can probe the lock
    probes it first, then reads the log's last record. A log whose last
    record is an [end] has that record's status. Otherwise the run is:
    - {e running} if the probe found the lock held;
    - {e crashed} if the probe found the lock free. It is also crashed when
      the reader cannot probe, its machine identity equals the [machine]
      of [writer], and the log was last modified before the reader's boot
      began (the log's modification time against Linux's [btime] in
      [/proc/stat] or macOS's [kern.boottime]). On a local file system
      both times come from the machine's clock; on a network file system
      the modification time comes from the server's. On one machine the
      writer wrote the log before its kernel stopped, and a process does
      not outlive its kernel. Either way the writer exited without an
      end;
    - {e silent} otherwise, as when the reader runs on another machine,
      either side has no identities, or the run has no [writer], as a run
      copied without it. To its reader a live writer and a
      dead one look the same, and the log's modification time says when
      the run last wrote. On a network file system that time comes from
      the file system's clock.

    Two machines that share a machine identity read as one. A system
    image or a container image copied with a filled [/etc/machine-id]
    does this, and must leave the file empty, so that each machine makes
    its own. A live run on one of them that writes after the other
    booted, as a sampler does every period, reads silent there; a run
    that has written nothing since the other machine booted reads
    crashed there.

    The probe comes before the read of the last record. In the other order
    a writer could write its end and exit between the two reads, and a
    finished run would read crashed.

    A process the writer forks without [exec] shares the writer's open
    log, and so its lock. The run reads running until that process exits
    too, so a running run proves that a process holds the lock, which may
    be such a child after the program that started the run has exited.

    A silent run whose writer is known to be gone is ended by a resume,
    which trusts that ({!section-resume}), followed by an [end] with
    status ["killed"]. It can then be deleted.

    {1:resume Resuming a run}

    A run that has no [end] is {e resumed} by a new writer that takes its
    lock. A writer that finds the lock held probes it. A granted probe
    means that only readers' probes hold the lock, and the writer tries
    again; a refused one means that a writer holds it, and the resume
    fails. Holding the lock, the new writer reads the log once, which
    gives it the run's end, if any, the kind of every key ({!section-log})
    and the statistics of every key, orphans left out
    ({!section-stats}). It cuts the log before its first line that is not
    whole, and appends a [resume] record. The resume's [step] is the step
    of the last sample that the state it restored includes, and the writer
    logs later steps after it. A writer under another boot identity cannot
    see the last writer's lock, so its resume trusts that the last writer
    has exited.

    After a crash of the system, the first line that is not whole may hold
    NUL bytes with whole records after it. The cut removes those records
    too. They were written after the log's last sync, within the loss
    {!section-durability} states.

    Deleting a run takes its lock in the same way, and removes the run
    only while it holds the lock: it renames the log to [.deleted] in the
    run's directory, which ends the run for readers and writers at once.
    It then lets the lock go, renames the directory to a [.]-named entry of
    [runs/] and removes that. Where the system refuses the rename, as
    Windows does while another program holds a file in the directory open,
    it removes the directory in place, [.deleted] last.

    {1:annotations Tags and notes}

    A tag is the file [tags/<digest>], where [<digest>] is the 64
    lowercase hexadecimal digits of the BLAKE2b-256 digest of the tag's
    UTF-8 bytes, and the file holds those bytes. Two tags are equal when
    their bytes are, whatever the file system's rules on names, such as
    case folding. A writer adds a tag by writing a [.]-named file in
    [tags/] and renaming it to the tag's name, and removes it by removing
    the file. Neither takes a lock. A tag file whose bytes do not digest
    to its name is damaged, and readers ignore it.

    [notes] holds the run's notes, UTF-8 text. A writer replaces the file
    whole. It writes the new text to a [.]-named file in the run's
    directory, syncs it, and renames it over [notes], so a reader sees the
    old notes or the new. When two writers replace the notes at once, the
    last rename wins.

    A writer of tags or notes creates [tags/] when it lacks it and no
    directory above it, so that it never makes again the directory of a
    run that a deletion removed meanwhile ({!section-resume}).

    {1:files Files a run logged}

    A file that the run logged under [key] at [step] is
    [files/<key>/<step><ext>], where [<ext>] is the extension of the file
    the program logged, or [files/<key>/<step>-<k><ext>] for the least [k]
    from [1] whose name is free. It is complete before the [file] record
    that names it is written.

    [git.diff] holds the output of [git diff --binary HEAD], run in the
    writer's working directory when the run started, if the [start]'s
    [provenance] says the work tree was dirty. [git apply] of it on the
    [commit] gives back the tracked files the run started from. Untracked
    files are not recorded, and a resume records no diff.

    {1:stats Statistics of an ended run}

    [stats.json] is a {e hint}: the statistics of each key of an ended run
    that has a sample, as a reader would compute them from the log, kept
    so that a reader need not fold a long log for them. Deleting it loses
    nothing. It is one JSON object of [format]; [rules], the version of
    the rules the statistics follow ({!rules}); [length], the length in
    bytes of the log it summarises, up to and including its [end]; and
    [keys], an object that maps each key to an object of [kind],
    ["metric"] or ["reading"]; [goal], when the key's latest [metric]
    record gave one; [count]; [last], the latest sample; [finite], when a
    sample is finite, an object of [least], [greatest] and [mean]; and
    [non_finite], when a sample is not, an object of [count] and [first].
    A sample is an object of [step], [time] and [value]. The statistics
    follow {!Metric.section-stats}, and orphans are left out
    ({!section-reading}).

    The writer keeps the statistics as it logs, and writes [stats.json]
    when it ends the run, after the [end], replacing the file whole as it
    replaces notes ({!section-annotations}). A resumed writer computes them
    again from the log it reads ({!section-resume}). A crashed run has no
    hint. A reader uses the hint only when it parses, its [format] equals
    {!version}, its [rules] equals {!rules} and its [length] equals the
    log's length, and otherwise ignores it and folds the log. If the
    writer cannot write the hint after the [end] is synced, the run has
    ended all the same, and readers fold its log.

    {1:artifacts Artifacts}

    Version [<n>] of the artifact [<name>] is the file
    [artifacts/<name>/<n>.json], [<n>] a decimal from [1] without leading
    zeros. It holds [format]; [time]; [run], the id of the run that
    produced it; [step], when given; [digest], the digest of its content
    ({!section-blobs}); [size], its content's size in bytes; and
    [metadata], an object of values. A writer claims a number by linking
    its complete file to that name, which fails when the name exists, and
    then tries the next number. A version file never changes, except that
    deleting the version replaces it with
    [{"format":1,"time":<t>,"deleted":true}], so that its number names
    nothing ever again. The {e latest} version of a name is its highest
    version not deleted.

{v
{"format":1,"time":1793549000.5,"run":"20261101T140307Z-a1f3c9e2",
 "step":500,"digest":"blake2b-256:9f2c...","size":4521984,
 "metadata":{"framework":"kaun"}}
v}

    [aliases/<alias>] holds a version number and a line feed. A writer sets
    an alias by replacing its file whole, as it replaces notes. Aliases
    and version numbers live apart, so an alias spelled as a number names
    an alias.

    {1:blobs Blobs}

    A {e digest} is [blake2b-256:] followed by 64 lowercase hexadecimal
    digits. A file's digest is the BLAKE2b-256 digest of its bytes. A
    directory's digest is that of its {e listing}: one line per entry below
    it, in the byte order of their paths relative to it, [dir:<path>] for a
    directory and [file:<path>:<hex>] for a file, each line ended by a line
    feed. A symbolic link to a file is stored as that file, and a link to a
    directory is refused. [blobs/blake2b-256/<hex>] is the file or
    directory whose digest ends in [<hex>].

    A writer copies the content into a [.]-named entry of [blobs/],
    computing the digest of the bytes it writes, syncs it, and renames it
    to its digest's name. A blob present under its name is therefore
    complete, and a reader may check it by computing its digest again.

    A writer that publishes a version holds a shared lock on [lock] from
    before it writes the blob until the version file exists, and a writer
    that sets an alias or deletes a version holds it while it replaces the
    file. A {e collector} holds an exclusive lock on [lock] while it
    removes the blobs that no version names and the [.]-named entries
    under [blobs/] and [artifacts/]. While it holds the lock no such write
    is in progress, so it removes nothing a version will name and no file
    a writer is about to rename. It also removes what a deletion cut short
    left ({!section-resume}): the [.]-named entries of [runs/], and the run
    directories that hold [.deleted] and no log. Neither is a run, and no
    writer adds to either.

    {1:durability Durability}

    A record is in the log once the call that writes it returns, and
    survives the writer's process crashing. A long record may take several
    system calls; a crash between them leaves a line that is not whole,
    which the next resume cuts. To survive a power loss or
    a crash of the system it must also be {e synced}, which a writer does
    with [fsync], and on macOS with [F_FULLFSYNC]. A writer syncs the log
    at a run's start, at each resume and at its end, and when the program
    asks. It syncs a version's blob and file, and the log of the run that
    produced it, when it publishes the version, so that a program that
    resumes from the version finds every sample up to the version's step.
    It syncs every other file it writes before it gives the file its name,
    so that after a power loss a name holds a complete file.

    Between those points the system writes data out on its own schedule.
    On Linux's defaults that is about 30 seconds after the data was
    written, so a power loss can lose the records of that time. Storage
    that ignores flushes can lose synced records too.

    {1:network Network file systems}

    A store may live on a file system that several machines mount. Its
    locks may hold only among the processes of one machine, as under NFS's
    [nolock] or [local_lock=flock], Lustre's [localflock] and most FUSE
    file systems, and no query a program can make shows it. A run's start
    claims its directory, which is exclusive on any file system, so two
    writers meet in one run only through a resume. A resume from another
    machine can then take the lock while the last writer still writes. The
    two writers' records then interleave, and the resume cuts the line the
    other writer has in progress.

    A reader on another machine sees the log as that file system shows it,
    possibly with parts the writer's machine has not yet sent, which read
    as NUL bytes. A reader that follows a log opens it again for each
    read, since these systems show another machine's writes to an open
    made after them, and it never reads past a line that is not whole
    ({!section-reading}). Such a system may cache a file's attributes for
    a while, so a log's modification time, and with it when a run last
    wrote, can lag behind the writer by that time. On an SMB mount under
    Linux a locked file cannot be read from another descriptor, so a
    run's log cannot be read while it is written.

    {1:crashes Crash states}

    A crash of a writer or of the system, at any instant, leaves a store in
    one of these states, each read as stated:
    - a run directory whose log holds no whole first line: not a run;
    - a log that ends in bytes that are not whole lines: they are not read,
      and the next writer's resume cuts them off;
    - a file under [files/] that no [file] record names: ignored;
    - a [.]-named file or directory: ignored, and removed by a collector
      under [blobs/] and [artifacts/] and in [runs/];
    - a run directory that holds [.deleted] and no log: not a run, and
      removed by a collector;
    - a blob that no version names: removed by a collector;
    - a log with no [end] whose lock no process holds: a crashed run;
    - a run directory with a [writer] and no whole first line: not a run;
      with a [writer] replaced and no [resume] after it: read under the
      new writer's identities, which is right, since that writer held the
      lock;
    - an ended run without [stats.json]: its log is folded.

    After a crash of the system, a log may also lack the records written
    since its last sync, and its end may hold NUL bytes where the file
    system had not written data. Damage is found by parsing. A line that
    the storage altered so that it still parses, such as a stale record or
    a changed digit, reads as written.

    {1:version Versions} *)

val version : int
(** [version] is the format this library reads and writes, [1]. *)

val rules : int
(** [rules] is the version of the statistics rules this library computes
    and reads in a statistics hint ({!section-stats}), [1]. It changes
    when a rule that picks a statistic changes, whatever {!version} is. *)
