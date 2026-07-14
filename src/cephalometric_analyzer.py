import math
import numpy as np

def calculate_angle_3points(p1, p2, p3):
    """
    Calculate the angle between three points (p1, p2, p3) with p2 as the vertex.
    Returns the angle in degrees.
    """
    if p1 is None or p2 is None or p3 is None:
        return None

    # Vector 1: p2 -> p1
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    # Vector 2: p2 -> p3
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    return _angle_between_vectors(v1, v2)

def calculate_angle_2lines(line1_p1, line1_p2, line2_p1, line2_p2):
    """
    Calculate the acute angle between two lines defined by (p1, p2) and (p3, p4).
    Returns the angle in degrees.
    """
    if None in [line1_p1, line1_p2, line2_p1, line2_p2]:
        return None

    v1 = np.array([line1_p2[0] - line1_p1[0], line1_p2[1] - line1_p1[1]])
    v2 = np.array([line2_p2[0] - line2_p1[0], line2_p2[1] - line2_p1[1]])

    ang = _angle_between_vectors(v1, v2)
    # Usually we want the acute angle or the specific interior angle depending on the clinical definition.
    # We will return the raw angle (0 to 180) here, specific metrics might need adjustments.
    return ang

def _angle_between_vectors(v1, v2):
    dot_prod = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_theta = dot_prod / (mag1 * mag2)
    # Handle floating point inaccuracies
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_rad = math.acos(cos_theta)
    return math.degrees(angle_rad)

class CephalometricAnalyzer:
    """
    Analyzer for computing Cephalometric measurements from landmark coordinates.
    """

    # Standard norms for adults (approximate Caucasian/Asian average)
    NORMS = {
        "SNA": {"mean": 82.0, "sd": 2.0},
        "SNB": {"mean": 80.0, "sd": 2.0},
        "ANB": {"mean": 2.0, "sd": 2.0},
        "FMA": {"mean": 25.0, "sd": 3.0}, # Frankfort-Mandibular Plane Angle
        "IMPA": {"mean": 90.0, "sd": 5.0}, # Lower incisor to Mandibular plane
        "U1_to_SN": {"mean": 104.0, "sd": 5.0}, # Upper incisor to SN plane
        "Gonial_Angle": {"mean": 120.0, "sd": 5.0},
    }

    def __init__(self, landmarks_dict):
        """
        landmarks_dict: dict of { "Symbol": (x, y) }
        Example: {"S": (100, 200), "N": (150, 250), ...}
        """
        self.landmarks = landmarks_dict

    def _get(self, symbol):
        return self.landmarks.get(symbol, None)

    def analyze(self):
        results = {}

        # 1. SNA (Sella - Nasion - A point)
        sna = calculate_angle_3points(self._get("S"), self._get("N"), self._get("A"))
        if sna is not None:
            results["SNA"] = self._format_result("SNA", sna)

        # 2. SNB (Sella - Nasion - B point)
        snb = calculate_angle_3points(self._get("S"), self._get("N"), self._get("B"))
        if snb is not None:
            results["SNB"] = self._format_result("SNB", snb)

        # 3. ANB (SNA - SNB usually, or angle between NA and NB)
        if sna is not None and snb is not None:
            anb = sna - snb
            results["ANB"] = self._format_result("ANB", anb)

        # 4. FMA (Frankfort Horizontal (Po-Or) to Mandibular Plane (Go-Me))
        fma = calculate_angle_2lines(self._get("Po"), self._get("Or"), self._get("Go"), self._get("Me"))
        if fma is not None:
            # FMA is usually acute
            if fma > 90: fma = 180 - fma
            results["FMA"] = self._format_result("FMA", fma)

        # 5. IMPA (Lower Incisor (LIA-LIT) to Mandibular Plane (Go-Me))
        impa = calculate_angle_2lines(self._get("LIA"), self._get("LIT"), self._get("Go"), self._get("Me"))
        if impa is not None:
            # IMPA is usually the inner angle
            if impa > 180: impa = 360 - impa
            if impa > 130: impa = 180 - impa
            results["IMPA"] = self._format_result("IMPA", impa)

        # 6. U1 to SN (Upper Incisor (UIA-UIT) to SN plane (S-N))
        u1_sn = calculate_angle_2lines(self._get("UIA"), self._get("UIT"), self._get("S"), self._get("N"))
        if u1_sn is not None:
            if u1_sn > 130: u1_sn = 180 - u1_sn
            results["U1_to_SN"] = self._format_result("U1_to_SN", u1_sn)

        # 7. Gonial Angle (Ar - Go - Me)
        gonial = calculate_angle_3points(self._get("Ar"), self._get("Go"), self._get("Me"))
        if gonial is not None:
            results["Gonial_Angle"] = self._format_result("Gonial_Angle", gonial)

        return results

    def _format_result(self, name, value):
        norm = self.NORMS.get(name)
        if not norm:
            return {"value": round(value, 2), "status": "Unknown"}

        mean = norm["mean"]
        sd = norm["sd"]

        if value > mean + sd:
            status = "High"
        elif value < mean - sd:
            status = "Low"
        else:
            status = "Normal"

        return {
            "value": round(value, 2),
            "norm_mean": mean,
            "norm_sd": sd,
            "status": status
        }
