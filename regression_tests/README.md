# Regression tests

This folder contains regression test cases used to verify LibRPA features.

For each test case `CASE`, the input files are located under `testcases/${CASE}`,
while the reference results are stored under `refs/${CASE}`.

## Run a single test case manually

The test cases in this directory must be run with the LibRPA driver enabled.
After building LibRPA with the driver, you can run a test case as follows:

```bash
# The LibRPA driver executable built
export myprog="/path/to/librpa/build/chi0_main.exe"
export CASE=case_1

cd testcases/${CASE}
tar -zxf dataset.tar.gz
cd librpa
# Adjust OMP_NUM_THREADS and MPI tasks
OMP_NUM_THREADS=1 mpirun -np 4 "$myprog" > librpa.out 2> librpa.err
```

After the run finishes, compare the standard output and any generated files with
the reference results under `refs/${CASE}/librpa`.

## Create a regression test case data

**Step 1**: create reference and testcase folder for the case
```bash
export CASE=case_new
mkdir -p {refs,testcases}/${CASE}/{dataset,librpa}
```

Step 2: prepare dataset
```bash
cd refs/${CASE}
cd dataset
# Create inputs and run dataset preparation program.
# For example, FHI-aims:
#     vim control.in; vim geometry.in; aims.x > aims.out
cd ..
# Compress dataset as dataset.tar.gz. Exclude all outputs, but keep all input files.
# For example, FHI-aims
tar --exclude aims.out -zchf dataset.tar.gz dataset
```

Step 3: prepare LibRPA input and run LibRPA to get reference data
```bash
mkdir librpa
cd librpa
cat > librpa.in << EOF
task = rpa
input_dir = ../dataset
# other parameters
# ...
EOF
mpirun -np 4 chi0_main > librpa.out
cd ..
```

Step 4: archive output to reference folder
```bash
# Copy dataset generation output for future reference
cp -a dataset/aims.out ../../refs/${CASE}/dataset/aims.out.export
cp -a librpa/librpa.out ../../refs/${CASE}/librpa/librpa.out
# Also copy other LibRPA output files if needed
# cp -a librpa/GW_band_spin_1.dat ../../refs/${CASE}/librpa/GW_band_spin_1.dat
```
Additional reference data (band structures and/or benchmark) from other programs
can be added to the reference `dataset` folder.
LibRPA run with different input or runtime configurations can also be placed and named appropriately
under `librpa` folder.

## Run automatic regression test and analysis

The automatic regression driver reads test definitions from `testsuite.xml`,
prepares a fresh workspace, runs selected test cases, and compares generated
outputs with the reference data under `refs/`.

The entry point for automatic test is the Python script `run_regression.py`.
To list the available test cases after applying build and runtime filters:

```bash
python3 run_regression.py list \
  --use-libri \
  -n 4 \
  --nthreads 1
```

To run tests and immediately perform the diff analysis:

```bash
python3 run_regression.py full /path/to/build/chi0_main.exe \
  --use-libri \
  -n 4 \
  --nthreads 1 \
  --mpiexec "mpirun --bind-to none" \
  --verbose \
  -o regression.log
```

The `full` mode writes normal terminal output and also records the same report
to the file selected by `-o/--output` (default: `regression.log`). The LibRPA
stdout/stderr of each case are stored separately in the workspace, for example
`workspace/testcases/${CASE}/librpa/librpa.out` and `librpa.err`.

Useful options:

- `-d WORKSPACE`: choose the run workspace, default `workspace`.
- `-f, --force`: reuse an existing workspace directory.
- `--only CASE [CASE ...]`: run only selected test case directories.
- `--exclude CASE [CASE ...]`: skip selected test case directories.
- `--mpiexec COMMAND`: choose an MPI launcher, including launcher options.
  For `mpirun` and `mpiexec`, `-np ${ntasks}` is added automatically unless
  the command already contains `-n`, `-np`, `--np`, or `--ntasks`.
- `--verbose`: print the exact command used for each test case.

You can also split execution and analysis into two steps. This is useful when
the calculations run on a batch node but the analysis is performed later:

```bash
python3 run_regression.py run /path/to/build/chi0_main.exe \
  --use-libri \
  -n 4 \
  --nthreads 1 \
  --mpiexec "mpirun --bind-to none" \
  --verbose

python3 run_regression.py analyze \
  --use-libri \
  -n 4 \
  --nthreads 1 \
  -o regression.log
```

### Register an automatic test entry

After preparing `testcases/${CASE}` and `refs/${CASE}`, register the case in
`testsuite.xml`. The `directory` attribute must match the testcase/reference
directory name:

```xml
<group name="RPA correlation energy" prefix="RPA">
  <testcase name="Water molecule in box (aims input)" directory="rpa_aims_mole_H2O_libri">
    <build require_libri="true" />
    <run ntasks_disable="none"
         nthreads_disable="none"
         ntasks_enable="none"
         nthreads_enable="none"
    />
    <validate name="Correlation energy"
              file="librpa/librpa.out"
              regex="Total EcRPA: \s+([-+]?\d*\.\d+)"
              comparison="cmp_float.abs_diff(1e-4)"
    />
  </testcase>
</group>
```

Entry fields:

- `<group>` collects related cases. `name` is printed by the test driver.
- `<testcase name="..." directory="...">` gives the display name and the
  directory under both `testcases/` and `refs/`.
- `<labels disable="...">` optionally disables a case. Use `disable="true"` or
  a short reason such as `disable="experimental"`.
- `<labels refhash="...">` records the commit hash used to generate the
  reference data. It is informational metadata for maintainers.
- `<build require_libri="true">` skips the case unless `--use-libri` is passed.
- `<run>` configures runtime setup. Use `ntasks_enable` or `nthreads_enable` to
  allow only selected runtime MPI/threads layouts, and `ntasks_disable` or
  `nthreads_disable` to reject selected layouts. Values can be comma- or
  space-separated integers, or `none`. Use space- or comma-separated
  `copy_files` to copy extra files or directories from `testcases/${CASE}/librpa`
  into the run workspace `librpa` directory.
- `<validate>` adds one comparison. If `file` is omitted, the default is
  `librpa/librpa.out`. `regex` locates the reference/output block, `headers`
  and `rows` select table regions when needed, and `comparison` names a
  comparison function from `backend/comparisons`.

Multiple `<validate>` entries can be attached to one testcase when a run
produces several quantities that should be checked.

Useful `<validate>` attributes:

- `name`: human-readable label printed in the regression report.
- `file`: file pattern to search under both the test workspace and reference
  directory. The default is `librpa/librpa.out`.
- `file_test`, `file_refr`: optional file patterns for the test workspace and
  reference directory. Each one overrides `file` only for that side.
- `regex`: Python regular expression used to find the quantity or table. If the
  matched line has one capture group, that group is compared. If it has multiple
  capture groups, they are joined with spaces before comparison.
- `headers`: number of lines to skip after a matched `regex`, default `0`.
- `rows`: number of lines to extract after `headers`. Use this for table
  comparisons, for example a header regex followed by `headers="2"` and
  `rows="52"`.
- `occurences`: zero-based selector for repeated regex matches. The attribute
  name is intentionally written here as accepted by the current parser. Use
  `occurences="0"` for the first match, `occurences="2"` for the third match,
  or `occurences="0:3"` for matches 0 through 3.
- `comparison`: comparison function and arguments. Common choices are
  `cmp_float.abs_diff(1e-4)` for scalar values and
  `cmp_table.abs_diff(1e-4)` for numeric tables. Table comparisons also accept
  a `columns` argument, for example
  `cmp_table.abs_diff(1e-4, columns="3:8")`.
- `binary_extract`: reserved for binary-output extractors. Plain-text output is
  used when this attribute is omitted.

### QSGW trace comparisons

For a head-only, direct-update `qsgw_band` case, compare every band energy and
the indirect gap at every committed iteration without making mixing-residual
values part of the acceptance criterion:

```xml
<validate name="QSGW bands and indirect gaps by iteration"
          file="librpa/QSGW_band_spin_*.dat"
          comparison="cmp_qsgw.band_iterations(occupied_bands=4,energy_tolerance_ev=1e-4,gap_tolerance_ev=2e-4,coordinate_tolerance=1e-7)"
/>
```

The reference directory fixes the required spin and iteration file set. The
comparison also requires iteration numbers to start at one and be continuous.

QSGW diagnostic cases that cover matrix or mixing internals should set
`qsgw_write_iteration_matrices = true` and validate all three machine-readable
traces. The QSGW comparison functions use the physical row keys rather than
file order, reject missing or duplicate entries, and reject non-finite values:

```xml
<validate name="QSGW iteration summary"
          file="librpa/qsgw_iterations.dat"
          comparison="cmp_qsgw.iteration_summary(energy_tolerance_ev=1e-5,residual_tolerance_ha=1e-8,scalar_tolerance=1e-10,coefficient_tolerance=1e-12)"
/>
<validate name="QSGW fixed-basis eigenvalues"
          file="librpa/qsgw_eigenvalues.dat"
          comparison="cmp_qsgw.eigenvalue_trace(tolerance_ha=1e-6,coordinate_tolerance=1e-12)"
/>
<validate name="QSGW iteration matrices"
          file="librpa/qsgw_matrices.dat"
          comparison="cmp_qsgw.matrix_trace(relative_tolerance=1e-8,absolute_tolerance=1e-12,hermiticity_tolerance=1e-10,unitarity_tolerance=1e-10)"
/>
```

For every nonzero reference matrix block, `matrix_trace` applies the relative
Frobenius tolerance directly. The absolute tolerance is used only when the
reference block is numerically zero. It also checks Hermiticity of static
Hamiltonian components and unitarity of every `rotation_u` block on both the
test and reference sides.

QSGW reference traces must be generated from the pre-port QSGW source and carry
the source snapshot, executable hash, input hashes, compiler/dependency data,
and MPI/OMP environment needed to reproduce them. Do not generate or update a
QSGW reference from the implementation under test. If the legacy output first
needs a deterministic format conversion, retain the original legacy artifacts,
the conversion command, and the converted trace together in `refs/${CASE}`.

## Reference update guidelines

- Do not edit reference output by hand. Reference data should match the commit
  recorded in the testcase `refhash` label in `testsuite.xml`, so it can be
  reproduced from that source snapshot, input setup, data files, and runtime
  environment.
- Update references sparingly. Code changes may perturb output, but if the
  quantities validated in `testsuite.xml` remain within tolerance, keep the
  existing reference data. That is the point of a regression test.
