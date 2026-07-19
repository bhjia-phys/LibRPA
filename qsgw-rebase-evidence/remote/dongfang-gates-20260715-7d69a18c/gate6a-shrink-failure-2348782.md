# Gate 6A shrink baseline failure (job 2348782)

## Scope

- Si 4x4x4, 64 explicit k/q points, crystal symmetry off, time-reversal reduction off.
- Legacy pre-port QSGW oracle executable from the verified real-Hartree v36 build.
- Hartree off, band off, head-wing off, linear mixing beta 0.2, target iteration 2.
- `use_shrink_abfs=true` was tested only as an isolation baseline before enabling Hartree.

## Result

This run is a preserved failure, not an accepted gate. Slurm job `2348782` ended as
`FAILED` after 00:05:26. The legacy oracle failed during its first G0W0 setup before
the candidate executable was started:

```text
G0W0::build_spacetime failed at stage 'unfold shrinked Wc(q,t), itau=0':
Failed to match shrink_sinvS with q = (-0.0730773, 0.0730773, 0.0730773)
```

The preflight had already verified all executable, source, input, Gate 5B, generator,
and observer hashes. It had also generated and frozen both input contracts. Each of
the four `shrink_sinvS_*.txt` files declares 64 q points; the four files are matrix
partitions, so this is not evidence of a four-q truncated dataset.

The failure is currently classified as a legacy QSGW shrink q-key/path mismatch.
It is not evidence of an old-versus-new numerical difference because the candidate
run never started. No shared G0W0/GW/epsilon/chi0 logic is changed for this failure.
The required Hartree old-versus-new gates proceed on the full-ABF/no-shrink path,
which is the already accepted Gate 5 input mode. The shrink failure remains a
separate shared-path diagnostic.

## Frozen evidence

- Run root: `/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/runs/gate6a-shrink-baseline-miniter2-a25e4e43-2348782`
- Input root: `/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/inputs/si-k444-gate6a-shrink-hartree-contracts-2348782`
- Job script SHA256: `215dcd3b74d060d27df547177e0b7b21c1745586db39a26648621b624c7bb29e`
- Legacy stdout SHA256: `ec61af8644ebeda443bc6cbc88431da7e85b92b2d580bdbdab66409c8bcae5ae`
- Legacy stderr SHA256: `05fc339ca3a5674f03823d75552bb264ae27317b12451d936db5f13c3235cf62`
- Failure sentinel SHA256: `7d320f0ae7ac18782e0ea0de36529ecb9d7dfd1d5afda10f2679d06016263e82`
- Preflight SHA256: `c7be2a1bc75425894de3397f5fe4df43a6ba1eafd674c4ff2878ee2718d0060f`
- Dataset manifest SHA256: `15efe274fadd4f7f33d8b277a0c8eddfb3eb30490c6fe9b120f3e6ed90d02d3f`
- Input provenance SHA256: `99342d9ee5191d146f2a8ec7d3571579520c351970e3cfdcbcae0f4ffe7b5aea`
- Input output-manifest SHA256: `44bf2ad27292587b7674b15a1e250557a22c8907757c933b013356452b35bcc0`
- Hartree-off contract SHA256: `192ac81714dcd8aca35360f4c1451677ce165b23f1893e8c248d72c4996988ff`
- Hartree-on contract SHA256: `441f4c2cd189c59c488a3d9943efe8a2d45071cc2ef9dda7b17a9d472fcbd71f`
