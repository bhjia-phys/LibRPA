import math
import re


__all__ = ["sym_on_off"]


_KPOINT_HEADER_RE = re.compile(r"^\s*K_point\s+\d+\s*:")
_SPIN_RE = re.compile(r"\bSpin\s+(\d+)")
# Numeric token of a fractional coordinate. Matches fixed-point and
# (D/E)-exponent forms; a leading sign keeps adjacent fixed-width columns
# glued together by setw(10) (e.g. "0.0000000-0.3333333") separable.
_NUM_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eEdD][-+]?[0-9]+)?")
_QP_RE = re.compile(
    r"^\s*(\d+)\s+[-+0-9.eEdD]+\s+[-+0-9.eEdD]+\s+"
    r"([-+]?(?:[0-9]*\.?[0-9]+(?:[eEdD][-+]?[0-9]+)?|[Nn][Aa][Nn]|[Ii][Nn][Ff]))\s*$"
)

_BANNER_RE = re.compile(r"^\s*-+\s*$")


def _parse_float(value):
    return float(value.replace("D", "E").replace("d", "E"))


def _reduce_kfrac(k):
    """Map a fractional coordinate into [0, 1) so that k and k + G are one key."""
    return k - math.floor(k)


def parse_energy_qp(text):
    """Parse a LibRPA `energy_qp` file.

    Returns (blocks, errors) with

        blocks: dict[(spin_index, (kx, ky, kz) reduced)] -> dict[state -> value]
        errors: list[str] of structural problems (duplicate blocks, ...)

    The parser tolerates the two header variants seen in the wild:
    ``K_point i : kx ky kz`` and ``K_point i : kx ky kz Spin s``, any numeric
    formatting of the k-coordinates (the driver has used both 7 significant
    digits and fixed 4 decimals), and fixed-width columns glued together for
    exactly-10-character values such as ``-0.3333333``. Rows are
    ``state occ e_gs e_qp``; only the state number and e_qp are kept.
    """
    blocks = {}
    errors = []
    current = None
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip() or _BANNER_RE.match(line):
            continue
        if _KPOINT_HEADER_RE.match(line):
            # Coordinates follow the ":" and may be glued by fixed-width
            # formatting; parse them as numeric tokens, taking the first three.
            coords = _NUM_RE.findall(line.split(":", 1)[1])[:3]
            spin_match = _SPIN_RE.search(line)
            if len(coords) < 3:
                errors.append("line {}: unparsable k-point coordinates".format(lineno))
                current = None
                continue
            try:
                kred = tuple(_reduce_kfrac(_parse_float(c)) for c in coords)
                spin = int(spin_match.group(1)) - 1 if spin_match else 0
            except ValueError:
                errors.append("line {}: unparsable k-point coordinates".format(lineno))
                current = None
                continue
            key = (spin, kred)
            if key in blocks:
                errors.append(
                    "line {}: duplicate k-point block spin {} k=({:.7f} {:.7f} {:.7f})"
                    .format(lineno, spin + 1, kred[0], kred[1], kred[2])
                )
                current = None
                continue
            blocks[key] = {}
            current = key
            continue
        if current is not None:
            row = _QP_RE.match(line)
            if row is None:
                continue
            try:
                value = _parse_float(row.group(2))
            except ValueError:
                errors.append("line {}: unparsable e_qp value".format(lineno))
                continue
            state = int(row.group(1))
            blocks[current][state] = value
    return blocks, errors


def _iter_file_entries(data):
    """Yield (filename, text) from a Validate-extracted dict.

    Entries are whole-file strings (regex omitted in the validate element) or
    lists of strings; lists are joined.
    """
    for fn, raw in data.items():
        if isinstance(raw, list):
            text = "\n".join(str(item) for item in raw)
        else:
            text = str(raw)
        yield fn, text


def _torus_distance(a, b):
    """Per-dimension distance on the [0, 1) torus (periodic mod 1)."""
    d = abs(a - b)
    return min(d, 1.0 - d)


def _match_kpoint(needle, candidates, ktol):
    """All candidate k-points within ktol (max-norm torus distance) of needle.

    Reduced fractional coordinates live on the [0, 1) torus, so a coordinate
    of ``1 - eps`` (e.g. -eps reduced mod 1) must match ``0``.
    """
    matches = []
    for cand in candidates:
        if all(_torus_distance(a, b) <= ktol for a, b in zip(needle, cand)):
            matches.append(cand)
    return matches


def sym_on_off(tolerance, precision=3, ktol=1e-6):
    """Compare the e_qp tables of two G0W0 runs, bucketed by common and
    run-specific k-points.

    This is the regression comparator for the G0W0 symmetry on/off check: one
    run (typically the symmetry-reduced one) computes only the irreducible
    (IBZ) k-points while the other computes the full k-grid. The comparator is
    direction-agnostic and keys everything by reduced fractional coordinates,
    so block order, k-point numbering and the presence of the ``Spin`` header
    column do not matter.

    Buckets:
    - common k (present in both runs, same spin): e_qp is compared per state
      with the absolute tolerance; matching NaN pairs pass, one-sided NaNs
      fail. This is the primary check on the IBZ-representative k-points.
    - test-only / ref-only k (the non-IBZ members of the dense run): a
      mapping-free sanity check - every state of a non-common k must agree
      with the same state at *some* common k of the same run within
      tolerance. Matching an arbitrary common k is a necessary condition for
      physical star degeneracy (the IBZ representative of a star is common),
      but it is not a strict star mapping: passing does not prove which star
      a non-common k belongs to, and the check cannot catch a wrong-but-
      self-consistent symmetry-restored run. Only the common bucket can, and
      it is reported separately.

    Returns (passed, message) with the maximum common-bucket difference and
    the violating state counts of the non-common buckets in the message.
    """
    tolerance = float(tolerance)
    precision = int(precision)
    ktol = float(ktol)
    msg_fmt = r"max abs diff = {:.%iE} (tol = {:.%iE})" % (precision, precision)

    def inner(fnobj1, fnobj2):
        files = set([*fnobj1.keys(), *fnobj2.keys()])
        if not files:
            return False, "no files found"
        for fn in files:
            if fn not in fnobj1:
                return False, "missing file {} in first (test) run".format(fn)
            if fn not in fnobj2:
                return False, "missing file {} in second (reference) run".format(fn)

        parsed1, errors1 = {}, []
        for fn, text in _iter_file_entries(fnobj1):
            blocks, errs = parse_energy_qp(text)
            parsed1.update(blocks)
            errors1.extend("{}: {}".format(fn, e) for e in errs)
        parsed2, errors2 = {}, []
        for fn, text in _iter_file_entries(fnobj2):
            blocks, errs = parse_energy_qp(text)
            parsed2.update(blocks)
            errors2.extend("{}: {}".format(fn, e) for e in errs)

        if errors1:
            return False, "parse errors in first (test) run: " + errors1[0]
        if errors2:
            return False, "parse errors in second (reference) run: " + errors2[0]
        if not parsed1:
            return False, "no QP blocks found in first (test) run"
        if not parsed2:
            return False, "no QP blocks found in second (reference) run"

        spins = set(s for s, _ in parsed1) | set(s for s, _ in parsed2)
        common, only1, only2 = [], [], []
        for spin in spins:
            ks1 = [k for s, k in parsed1 if s == spin]
            ks2 = [k for s, k in parsed2 if s == spin]
            for k1 in ks1:
                matches = _match_kpoint(k1, ks2, ktol)
                if len(matches) > 1:
                    return False, (
                        "ambiguous k-point match for spin {} k=({:.7f} {:.7f} {:.7f}) "
                        "within ktol {}; increase ktol"
                        .format(spin + 1, k1[0], k1[1], k1[2], ktol)
                    )
                if matches:
                    common.append((spin, k1, matches[0]))
                else:
                    only1.append((spin, k1))
            for k2 in ks2:
                if not _match_kpoint(k2, ks1, ktol):
                    only2.append((spin, k2))

        if not common:
            return False, "no common k-points found between the two runs"

        diff = 0.0
        diff_loc = None
        for spin, k1, k2 in common:
            states1, states2 = parsed1[(spin, k1)], parsed2[(spin, k2)]
            if set(states1) != set(states2):
                return False, (
                    "state count mismatch at spin {} k=({:.7f} {:.7f} {:.7f}): {} != {}"
                    .format(spin + 1, k1[0], k1[1], k1[2], len(states1), len(states2))
                )
            for state in states1:
                value1, value2 = states1[state], states2[state]
                if math.isnan(value1) or math.isnan(value2):
                    if math.isnan(value1) and math.isnan(value2):
                        continue
                    return False, (
                        "nan mismatch at spin {} k=({:.7f} {:.7f} {:.7f}) state {}: "
                        "{} != {}".format(spin + 1, k1[0], k1[1], k1[2], state, value1, value2)
                    )
                d = abs(value1 - value2)
                if d > diff:
                    diff = d
                    diff_loc = (spin, k1, state)

        # Common k-points expressed in each run's own coordinates: two runs may
        # match within ktol without printing identical digits, and every lookup
        # below must use the coordinate set of the run it reads from.
        common1 = [(spin, k1) for spin, k1, _ in common]
        common2 = [(spin, k2) for spin, _, k2 in common]

        def _check_star_members(entries, blocks, common_own, label):
            """Mapping-free sanity check for one run's non-common k-points.

            Every state of a non-common k must agree (within tolerance) with
            the same state at *some* common k of this run, using this run's
            own common coordinates (`common_own`). This is a necessary
            condition for physical star degeneracy, not a strict star mapping;
            see the module docstring.
            """
            unmatched = []
            for spin, k in entries:
                for state, value in blocks[(spin, k)].items():
                    if math.isnan(value):
                        unmatched.append((spin, k, state))
                        continue
                    ok = False
                    for kc in common_own:
                        if kc[0] != spin or state not in blocks[kc]:
                            continue
                        if abs(value - blocks[kc][state]) <= tolerance:
                            ok = True
                            break
                    if not ok:
                        unmatched.append((spin, k, state))
            return unmatched

        unmatched1 = _check_star_members(only1, parsed1, common1, "test-only")
        unmatched2 = _check_star_members(only2, parsed2, common2, "ref-only")

        msg = "{}; {} common k, {} test-only k, {} ref-only k".format(
            msg_fmt.format(diff, tolerance), len(common), len(only1), len(only2))
        if diff_loc is not None:
            spin, k, state = diff_loc
            msg += "; max at spin {} k=({:.7f} {:.7f} {:.7f}) state {}".format(
                spin + 1, k[0], k[1], k[2], state)
        for label, unmatched, entries, blocks in (
                ("test-only", unmatched1, only1, parsed1),
                ("ref-only", unmatched2, only2, parsed2)):
            if unmatched:
                spin, k, state = unmatched[0]
                n_states = sum(len(blocks[(s, kk)]) for s, kk in entries)
                msg += ("; {}: {}/{} states across {} k violate the common-k "
                        "sanity check, first at spin {} k=({:.7f} {:.7f} {:.7f}) state {}"
                        .format(label, len(unmatched), n_states, len(entries),
                                spin + 1, k[0], k[1], k[2], state))
                return False, msg

        return diff <= tolerance, msg

    return inner
