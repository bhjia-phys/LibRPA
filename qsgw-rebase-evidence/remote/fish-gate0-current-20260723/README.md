# Fish Gate 0, upstream 67b9888d

Run `4f9ab0cf-v1` is the accepted clean build and test gate after rebasing the
QSGW adapter onto upstream `67b9888d`. The candidate product source is
`4f9ab0cfc90f54910158ab01a877581b080f136e`.

The immutable fish run passed 39/39 upstream CTests, 63/63 candidate CTests,
10/10 focused QSGW CTests, and 29/29 Python comparator/wiring tests. The
protected shared numerical diff is empty. `GREEN_CONFIRMED` exists and
`FAILED` does not.

The directory was copied byte-for-byte from
`/home/bhj/ai-runs/librpa-qsgw-gate0-20260723-4f9ab0cf-v1`. Every file listed
in `OUTPUT_SHA256SUMS.txt` was rehashed after transfer and passed.

- `PROVENANCE.txt` SHA256:
  `62a4026017c26b28f9c9503fb4c71a265345369db2a709b0f811a7000b5fd424`
- `OUTPUT_SHA256SUMS.txt` SHA256:
  `845d309c485d5fd8060a70faf57584aa1e6c44e267595ab15f4a2ec12417c5dd`
- upstream executable SHA256:
  `c2705015e2219c548ce6d6cfce93d32072b14391fbd651aab4e38c0bb125737c`
- candidate executable SHA256:
  `77ab964e15f0cdfee05ad54cf5b9da4ad9e4e3ac1f6990c4b50e8f0a55a29b47`
