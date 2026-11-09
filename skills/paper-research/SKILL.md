---
name: paper-research
description: Find and actually read a few research papers on a topic, then report concrete findings with numbers.
triggers: research paper, research papers, paper, papers, arxiv, preprint, study, studies, publication
---
Goal: depth on a few papers, not snippets on many. Budget for the whole turn: at most 3 searches
and at most 4 page or PDF fetches.

1. Search 2-3 times, phrased to find papers: add words like "arxiv", "paper", "pdf",
   a venue (MLSys, NeurIPS, OSDI) or the year. If the user also asked about trends or the
   market, one of the searches can cover that.
2. Pick at most 3 papers from the results. Prefer recent work, peer-reviewed venues and
   well-known groups. Write down why you picked each one.
3. Read each paper ONCE, from its PDF:
   - Go straight to the PDF URL (for arXiv, arxiv.org/pdf/<id>, not /abs/<id>). Don't fetch
     the abstract page first and the PDF second.
   - Never fetch the same paper twice. If a fetch inside code returns the PDF as base64,
     decode it with pypdf in that same step instead of downloading it again.
   - Read the abstract, the method and the results. Skip related work and the appendix.
4. For each paper, note: the problem it solves, the core idea in one or two sentences from
   first principles, the headline result with its numbers and baseline, and one limitation.
5. Use search snippets for market or trend claims. Fetch a page only if two sources
   disagree on a number that matters.
6. Answer in this shape:
   - **Answer:** two or three sentences.
   - **Papers:** for each one: title, venue or "preprint", date, link, then the problem,
     idea, result (with numbers) and limitation.
   - **Trends:** up to four bullets with source links, if the user asked about them.
   - **Uncertain:** say which papers you read in full and which only from snippets; flag
     preprints that aren't peer-reviewed and numbers that sources disagree on.
   - **Bottleneck:** the underlying problem these papers are attacking.
   - **Future:** where this line of research is heading.
7. Remember the user's research interest as a fact if it's new.
