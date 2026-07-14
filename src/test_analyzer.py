import unittest
import math
from cephalometric_analyzer import CephalometricAnalyzer, calculate_angle_3points, calculate_angle_2lines

class TestCephalometricAnalyzer(unittest.TestCase):

    def test_calculate_angle_3points_90deg(self):
        # S(0, 1), N(0, 0), A(1, 0) -> Angle at N is 90 deg
        S = (0, 1)
        N = (0, 0)
        A = (1, 0)
        angle = calculate_angle_3points(S, N, A)
        self.assertAlmostEqual(angle, 90.0, places=2)

    def test_calculate_angle_3points_45deg(self):
        # S(0, 1), N(0, 0), A(1, 1) -> Angle at N is 45 deg
        S = (0, 1)
        N = (0, 0)
        A = (1, 1)
        angle = calculate_angle_3points(S, N, A)
        self.assertAlmostEqual(angle, 45.0, places=2)

    def test_calculate_angle_2lines_parallel(self):
        L1_p1, L1_p2 = (0, 0), (1, 0)
        L2_p1, L2_p2 = (0, 1), (1, 1)
        angle = calculate_angle_2lines(L1_p1, L1_p2, L2_p1, L2_p2)
        self.assertAlmostEqual(angle, 0.0, places=2)

    def test_calculate_angle_2lines_perpendicular(self):
        L1_p1, L1_p2 = (0, 0), (1, 0)
        L2_p1, L2_p2 = (0, 0), (0, 1)
        angle = calculate_angle_2lines(L1_p1, L1_p2, L2_p1, L2_p2)
        self.assertAlmostEqual(angle, 90.0, places=2)

    def test_analyzer_integration(self):
        # Create a mock lateral ceph geometry
        # Sella
        S = (100, 100)
        # Nasion (straight right from S for simplicity)
        N = (200, 100)
        # Point A (down and slightly right from N)
        # N->A vector: let's make the angle exactly 82 degrees inner
        # We need N->A to form 82 degrees with N->S (-1, 0)
        # N->S is along the -x axis. To make an 82 deg angle,
        # N->A should be in the -x, +y quadrant mostly down.
        # Let's just use known coordinates to verify the pipeline runs without errors.

        A = (195, 200)
        B = (190, 220)

        # Frankfort horizontal (Po to Or) -> exactly horizontal for easy testing
        Po = (50, 150)
        Or = (150, 150)

        # Mandibular plane (Go to Me)
        Go = (100, 250)
        Me = (200, 300) # slope is 50/100 = 0.5. atan(0.5) is ~26.56 deg.

        # Lower incisor (LIA to LIT)
        LIA = (180, 260)
        LIT = (185, 230)

        # Upper incisor (UIA to UIT)
        UIA = (175, 170)
        UIT = (180, 210)

        landmarks = {
            "S": S, "N": N, "A": A, "B": B,
            "Po": Po, "Or": Or, "Go": Go, "Me": Me,
            "LIA": LIA, "LIT": LIT, "UIA": UIA, "UIT": UIT,
            "Ar": (80, 180)
        }

        analyzer = CephalometricAnalyzer(landmarks)
        results = analyzer.analyze()

        self.assertIn("SNA", results)
        self.assertIn("SNB", results)
        self.assertIn("ANB", results)
        self.assertIn("FMA", results)

        # FMA should be ~26.56
        self.assertAlmostEqual(results["FMA"]["value"], math.degrees(math.atan(0.5)), places=1)

if __name__ == '__main__':
    unittest.main()
