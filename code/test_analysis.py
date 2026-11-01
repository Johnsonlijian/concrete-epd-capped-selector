"""Implementation tests, not empirical validation of declarations or designs."""
import unittest
import pandas as pd
from analyze_epd_choices import signature, choices, common_cohort, MENU_KEYS


def fixture():
    rows = []
    for row_id, strength, age, gwp in [(2,4000,28,300.), (3,5000,28,320.), (4,5000,56,240.)]:
        row = {k: "fixture" for k in MENU_KEYS}
        row.update(row_id=row_id, strength_psi=strength, age_days=age, gwp=gwp,
                   mix=f"M{row_id}", description=f"{strength} PSI AE", components="cement",
                   description_series="<strength> psi ae", state="test", wc=.45,
                   a1=gwp-20, a2=10., a3=10.)
        rows.append(row)
    return pd.DataFrame(rows)


class AnalysisTests(unittest.TestCase):
    def test_signature_replaces_only_strength(self):
        self.assertEqual(signature('4000 PSI 3/4" 4-6SL',4000), '<strength> psi 3/4" 4-6sl')

    def test_discordant_strength_refused(self):
        self.assertEqual(signature('3500 PSI',4000), '')

    def test_multiple_strengths_refused(self):
        self.assertEqual(signature('4000 PSI / 5000 PSI',4000), '')

    def test_expanded_pool_contains_reference(self):
        r = choices(fixture(), "description_series").iloc[0]
        self.assertEqual(r.baseline_id,2)
        self.assertEqual(r.candidate_id,2)
        self.assertEqual(r.gain_kg_m3,0)

    def test_later_age_not_used_as_28_day(self):
        r = choices(fixture(), "description_series").iloc[0]
        self.assertEqual(r.any_age_candidate_id,4)
        self.assertEqual(r.candidate_id,2)
        self.assertTrue(r.different_age_improves)

    def test_common_cohort_ratio_and_volume_condition(self):
        d = fixture()
        r = common_cohort(d,choices(d,"description_series")).iloc[0]
        self.assertEqual(r.ratio_candidate_id,3)
        self.assertAlmostEqual(r.ratio_excess_for_fixed_volume,20/300)
        self.assertAlmostEqual(1-r.series_gwp/r.ratio_gwp,.0625)

    def test_different_descriptions_do_not_match(self):
        d = fixture()
        d.loc[d.row_id.eq(3),"description_series"] = "<strength> psi non-ae"
        self.assertTrue(choices(d,"description_series").empty)

    def test_tied_gwp_does_not_create_improvement(self):
        d = fixture()
        d.loc[d.row_id.eq(3),"gwp"] = 300
        r = choices(d,"description_series").iloc[0]
        self.assertEqual(r.gain_kg_m3,0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
