/* The contract between lz's kernels and its OCaml stubs (stubs.c).

   A kernel is one C entry per direction over a [struct lz_io]: it reads
   from [src], writes to [dst], and advances [consumed] and [produced]. One
   kernel serves both drivers:

     whole-buffer   a state initialised with stream = false, final = true,
                    the destination as the window. For gzip and Zstandard
                    the stub calls the kernel again on the rest of the
                    input after each member, to the input's end: the
                    kernel begins a member there, and its header check
                    refuses a byte that begins none.
     streaming      a state initialised with stream = true, kept in an
                    OCaml custom block across steps; [final] is the step's
                    [~last]. The state holds its tables and the partial
                    symbol; the window lies in [dst], the decoder's buffer
                    (below).

   Laws every entry keeps:

   - No read outside [src, src + src_len), no write outside
     [dst, dst + dst_cap). [src] and [dst] are never NULL: the stubs pass a
     static byte for an empty buffer.
   - A decoder returns LZ_DONE at the end of one member or frame, having
     verified its checksum and declared sizes. Called again on the same
     state, it begins the next member, with an empty window. LZ_NEED_INPUT
     means it has taken every input byte, a cut header included;
     LZ_NEED_OUTPUT that [dst], or a step's run, is full and more output
     is due. With [final] set, the input ending inside a member is
     LZ_ERROR with LZ_TRUNCATED at the input's end, never LZ_NEED_INPUT,
     and so is an empty input at a member's start.
   - A deflate or zlib stream's LZ_DONE leaves [consumed] at the byte after
     the one that holds its last bit: bytes after it are never consumed,
     in this call or an earlier one.
   - An error's [at] is the offset of the first byte of what is wrong: the
     byte a bad code begins at, the first byte of a checksum or size field
     that does not match or of a declined field, the input's end for a
     truncation; LZ_TOO_LONG at the byte that begins the code asking for
     the output. It is relative to [io->src] of the call that reports it,
     and negative when that byte arrived in an earlier call; the stubs add
     the bytes taken before. [msg] is a static string from the format's
     fixed set, with no input bytes; NULL for LZ_TRUNCATED and LZ_TOO_LONG.
   - After LZ_ERROR the state is dead: the stubs never call the entry on it
     again; reset is only called after LZ_DONE.
   - An encoder's output depends only on the input bytes, the level and
     where [last] was set: never on how the input was sliced across calls,
     nor on the instruction set the host has. It returns LZ_DONE once the
     member is complete and every byte of it written (only with [last]),
     LZ_NEED_INPUT when it has consumed all of [src] and holds nothing it
     can write yet, LZ_NEED_OUTPUT when [dst] is full.
   - A kernel reports its work through [lz_did] (below), and the stubs may
     release the runtime lock inside any report, taking it back once the
     kernel returns. So a kernel touches nothing on the OCaml heap and
     never calls into the runtime: it reads and writes [src] and [dst],
     whose bytes the collector never moves, its state, which lives
     outside the heap, and its own stack. It allocates only through
     [lz_alloc], which is safe without the lock, and reports a refused
     allocation as LZ_ERROR with LZ_NOMEM, which the stub raises as
     Out_of_memory. The whole-buffer decode path allocates nothing: its
     state lives on the stub's C stack.

   Work

   [io->work] counts a call's work in bytes, and the stubs release the
   lock at the first report that brings it to the kernel's threshold. A
   kernel reports often enough that it never works long unreported, and
   rarely enough that its reports cost nothing measurable:

   - a decoder counts the bytes it has taken and written. It reports
     those since its last report before each block it begins (Zstandard
     once they reach 4 KiB, which spares a stream of small blocks a
     report each) and each time its fast loop has written LZ_WORK_SPAN
     bytes since; an LZ4 or Snappy decoder also before it copies a
     literal run, a match or a stored block of LZ_WORK_SPAN bytes or
     more outside its fast loop, counting the copy;
   - an encoder counts the input it compresses, and reports it before it
     compresses it: a whole-buffer encode all of its input at its start,
     a step the input each parse or each run of blocks covers. A
     Zstandard step also reports the bytes a window slide moves, over
     4 MiB at levels 9 to 12.

   The stubs add a decoder's work after its last report, such as its last
   block, without a release: it counts at the call's next report, if any.
   Between two reports a kernel does at most one unit whose size the
   format or the kernel bounds: a Zstandard block, of at most 128 KiB of
   output; a deflate stored block or gzip header field, of at most
   64 KiB; LZ_WORK_SPAN bytes of output and the code or sequence they end
   in; a deflate encoder's block write and window slide, each of at most
   a few hundred KiB; a step's copy of its input into its window. So a
   call holds the lock for at most its threshold's work and one such
   unit, and a call whose last unit brings its work to the threshold
   holds it to its end.

   A streaming decoder's buffer

   A streaming decoder writes into one buffer, a bigarray the stubs hold
   for it and lend each step's run from:

     [ window ... | run ......... | stale ... | window's earlier segment | ]
       0           [run]       [produced]                                dst_cap

   [dst] is the buffer and [dst_cap] its size, [produced] the offset of
   the next byte the decoder writes and [run] the offset where the step's
   run began: the run is the bytes from [run] to [produced], which the
   stubs set equal at a step's entry. The bytes before [run] are the
   window. A decoder:

   - writes a step's run in one piece of at most its format's run size
     (LZ_INFLATE_RUN, LZ_ZSTD_RUN) and returns LZ_NEED_OUTPUT once it is
     full: deflate cuts a match at the run's end, Zstandard stops before
     a block that might pass it;
   - may wrap while the run is empty: it sets [run] and [produced] to 0,
     the bytes behind become the earlier segment of its window, and a
     match reads across the wrap from there, so the window is never
     copied. Its buffer is large enough that a run and the slack its
     wide stores write past it never reach the window's live bytes;
   - returns LZ_NEED_BUFFER with [need] set when [dst_cap] is below what
     its window, a run and the slack take: deflate at its first call,
     Zstandard once a frame header it has taken fixes a larger window.
     The stubs give it a buffer of [need] bytes holding the run at its
     start, with [run] at 0, and call it again; the window is then empty,
     so nothing else is kept.

   [run] and [need] are a streaming decoder's alone: other entries ignore
   them. */

#ifndef LZ_H
#define LZ_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* A call's work: [done] bytes reported, and [reach], which the stubs run
   once [done] reaches [limit]. */
struct lz_work {
  size_t done;
  size_t limit;
  void (*reach)(struct lz_work *w);
};

/* The most output a decoder writes inside a block between two reports. */
#define LZ_WORK_SPAN (32 * 1024)

/* [lz_did w n] reports [n] bytes of work. */
static inline void lz_did(struct lz_work *w, size_t n) {
  w->done += n;
  if (w->done >= w->limit) w->reach(w);
}

/* [work] is never NULL. */
struct lz_io {
  const uint8_t *src;
  size_t src_len;
  uint8_t *dst;
  size_t dst_cap;
  size_t consumed;
  size_t produced;
  size_t run;
  size_t need;
  struct lz_work *work;
};

enum lz_status {
  LZ_DONE,
  LZ_NEED_INPUT,
  LZ_NEED_OUTPUT,
  LZ_NEED_BUFFER,
  LZ_ERROR
};

/* The order matches [Lz.why]'s constant and non-constant constructors as
   stubs.c builds them; LZ_NOMEM never reaches OCaml as a value. */
enum lz_why {
  LZ_TRUNCATED,
  LZ_MALFORMED,
  LZ_TOO_LONG,
  LZ_UNSUPPORTED,
  LZ_NOMEM
};

struct lz_error {
  int kind;
  int64_t at;
  const char *msg;
};

/* [lz_alloc n] is [n] bytes or NULL. Safe with the runtime lock released. */
void *lz_alloc(size_t n);
void lz_free(void *p);

static inline enum lz_status lz_fail(struct lz_error *err, int kind,
                                     int64_t at, const char *msg) {
  err->kind = kind;
  err->at = at;
  err->msg = msg;
  return LZ_ERROR;
}

/* Each format's state and entries are declared in its header
   (deflate.h, zstd.h, lz4.h, snappy.h). Every streaming decoder state F
   has:

     void lz_F_dec_init(struct lz_F_dec *st, ..., bool stream);
     void lz_F_dec_reset(struct lz_F_dec *st);  a fresh stream after
                                                LZ_DONE, keeps its
                                                allocations
     void lz_F_dec_free(struct lz_F_dec *st);   frees them
     enum lz_status lz_F_decode(struct lz_F_dec *st, struct lz_io *io,
                                bool final, struct lz_error *err);

   and every encoder state F:

     void lz_F_enc_init(struct lz_F_enc *st, int level, bool stream);
     void lz_F_enc_reset(struct lz_F_enc *st);
     void lz_F_enc_free(struct lz_F_enc *st);
     enum lz_status lz_F_encode(struct lz_F_enc *st, struct lz_io *io,
                                bool last, struct lz_error *err);
     size_t lz_F_bound(size_t n);

   An encoder fails only with LZ_NOMEM. [lz_F_enc_init] allocates
   nothing: tables are allocated at the first [lz_F_encode], so that the
   stubs can report a refusal. */

#endif
