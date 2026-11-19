# AI in Raven

## How Raven is built

Raven is developed with AI coding agents. The project started in
February 2025, and its early code was written by hand or with AI
assistance. As models improved, more of the implementation work moved
to agents. Since around mid-2026, almost no code is written by hand
anymore: the maintainers guide, design and review, and agents
implement, test, benchmark and document changes under the rules in
[AGENTS.md](AGENTS.md) and [REVIEW.md](REVIEW.md). The agents and
models change as better ones appear; they have included Claude,
OpenAI's Codex and DeepSeek.

Commits carry no per-commit AI marking. Assume any recent commit was
written by an agent, under a maintainer's guidance and review.

This section describes how the current maintainers work, for
transparency. It doesn't prescribe a workflow: each maintainer is
trusted to use AI as they see fit and is responsible for the code
their agents produce.

## Why

Today, we think we'd be wasting our time writing code by hand. Recent
models write code as good as ours, often better, in a fraction of the
time. Our time is better spent planning, designing, and making sure
each change follows the design and the principles behind Raven.

AI is also what allows us to build the Raven we have in mind. With the
limited resources we have, we can offer a scientific computing stack
that competes with PyTorch and JAX, and is ahead of them in some
aspects. The Raven we have today, and the one we are building, is the
Raven we dreamed of when we started the project. It's not one we could
ever have built or maintained without coding agents.

Finally, we are building Raven for scientific computing in the AI era.
We believe a lot of scientific research will be done with the help of
AI agents, and we want Raven to be the best stack for those agents to
work with. Many of our choices come from this. Raven depends on no
third-party system library, and a single repository holds every piece
of a scientific stack, from the low-level hardware layers to libraries
for specific scientific fields. This is very deliberate: an agent can
read, change and test any layer without leaving the project.

So not using AI would be a questionable choice for a project that wants
to serve AI development and research.

## How we keep quality

We spend a lot of time on planning and design. Most changes to a public
API go through an AI-assisted design phase, following the principles in
[docs/dev/design.md](docs/dev/design.md). Its outcome is a design document
or RFC, which one or more maintainers review carefully.

Once the design is settled, its implementation is broken down and
delegated to agents. Each piece goes through several AI reviews, from
agents that did not write it, and by default a review by a maintainer,
before it lands.

We also rely on tooling. Our tests use
[Windtrap](https://github.com/invariant-hq/windtrap), with property
tests and stateful tests that check a property over many inputs. Where
a trusted reference exists, we also test against goldens generated from
it: Tolk's kernels, for example, are compared with tinygrad's.

Our benchmarks use [Thumper](https://github.com/invariant-hq/thumper),
with baselines committed to the repository for each machine. A
regression blocks landing until it is explained.

## Contributions

AI tools are welcome. What matters is the result. Maintainers have
earned trust through their record; a new contributor has not yet, so
their contributions get closer scrutiny.

1. You must understand the code you submit and have reviewed every part
   of it yourself, as you would code you wrote by hand. You are
   responsible for its quality and must be able to explain and defend
   any line of it in review.
2. If a significant part of your code, pull request description or
   review comments was generated, say which tool you used and for what.
3. Reviewer time is scarce. A pull request that is plainly generated
   and shows no care from its submitter will be closed without review.
   Signs include code the submitter cannot explain, changes unrelated
   to the stated goal, invented APIs, tests that check nothing, and
   descriptions that don't match the diff.
4. You take responsibility for the provenance of what you submit. It
   must be publishable under Raven's [licence](LICENSE) and must not
   reproduce code under an incompatible one.
