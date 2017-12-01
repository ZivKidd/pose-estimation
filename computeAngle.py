# -*- coding: utf-8 -*-
import numpy as np
import math

class computeAngle:
    def __init__ (self, output, hand):
        self.output = output
        self.hand = hand

    def compute(self):
        def angle(x, y):
            dot_xy = np.dot(x, y)
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            cos = dot_xy / (norm_x*norm_y)
            rad = np.arccos(cos)
            theta = rad * 180 / np.pi
            return theta

        def shoulderAngle(self):
            hand = self.hand
            output = self.output
            if hand == "right":
                if not None in [output["points"]["Rsho"][0], output["points"]["Rhip"][0], output["points"]["nose"][0]]:
                    x = [p1-p2 for (p1, p2) in zip(output["points"]["Rsho"], output["points"]["nose"])]
                    y = [p1-p2 for (p1, p2) in zip(output["points"]["Rhip"], output["points"]["nose"])]
                    return int(angle(x, y))
                else:
                    return None
            else:
                if not None in [output["points"]["Lsho"][0], output["points"]["Lhip"][0], output["points"]["nose"][0]]:
                    x = [p1-p2 for (p1, p2) in zip(output["points"]["Lsho"], output["points"]["nose"])]
                    y = [p1-p2 for (p1, p2) in zip(output["points"]["Lhip"], output["points"]["nose"])]
                    return int(angle(x, y))
                else:
                    return None

        def wristAngle(self):
            hand = self.hand
            output = self.output
            if hand == "right":
                if not None in [output["points"]["Rwri"][0], output["points"]["Rhip"][0], output["points"]["nose"][0]]:
                    x = [p1-p2 for (p1, p2) in zip(output["points"]["Rwri"], output["points"]["nose"])]
                    y = [p1-p2 for (p1, p2) in zip(output["points"]["Rhip"], output["points"]["nose"])]
                    return int(angle(x, y))
                else:
                    return None
            else:
                if not None in [output["points"]["Lwri"][0], output["points"]["Lhip"][0], output["points"]["nose"][0]]:
                    x = [p1-p2 for (p1, p2) in zip(output["points"]["Lwri"], output["points"]["nose"])]
                    y = [p1-p2 for (p1, p2) in zip(output["points"]["Lhip"], output["points"]["nose"])]
                    return int(angle(x, y))
                else:
                    return None

        def armAngle(self):
            hand = self.hand
            output = self.output
            if hand == "right":
                if not None in [output["points"]["Rwri"][0], output["points"]["Relb"][0], output["points"]["Rhip"][0], output["points"]["nose"][0]]:
                    arm = [int((p1+p2)/2) for (p1, p2) in zip(output["points"]["Rwri"], output["points"]["Relb"])]
                    x = [p1-p2 for (p1, p2) in zip(arm, output["points"]["nose"])]
                    y = [p1-p2 for (p1, p2) in zip(output["points"]["Rhip"], output["points"]["nose"])]
                    return int(angle(x, y))
                else:
                    return None
            else:
                if not None in [output["points"]["Lwri"][0], output["points"]["Lelb"][0], output["points"]["Lhip"][0], output["points"]["nose"][0]]:
                    arm = [int((p1+p2)/2) for (p1, p2) in zip(output["points"]["Lwri"], output["points"]["Lelb"])]
                    x = [p1-p2 for (p1, p2) in zip(arm, output["points"]["nose"])]
                    y = [p1-p2 for (p1, p2) in zip(output["points"]["Lhip"], output["points"]["nose"])]
                    return int(angle(x, y))
                else:
                    return None

        def trunkAngle(self):
            output = self.output
            if not None in [output["points"]["Rank"][0], output["points"]["Lank"][0], output["points"]["neck"][0]]:
                ank = [int((p1+p2)/2) for (p1, p2) in zip(output["points"]["Rank"], output["points"]["Lank"])]
                x = [p1-p2 for (p1, p2) in zip(ank, output["points"]["neck"])]
                y = [p1-p2 for (p1, p2) in zip(ank, (ank[0], output["points"]["neck"][1]))]
                return int(angle(x, y))
            else:
                return None

        def upperAngle(self):
            output = self.output
            if not None in [output["points"]["Rhip"][0], output["points"]["Lhip"][0], output["points"]["Rsho"][0], output["points"]["Lsho"][0], output["points"]["Rank"][0], output["points"]["Lank"][0]]:
                hip = [int((p1+p2)/2) for (p1, p2) in zip(output["points"]["Rhip"], output["points"]["Lhip"])]
                ank = [int((p1+p2)/2) for (p1, p2) in zip(output["points"]["Rank"], output["points"]["Lank"])]
                x = [p1-p2 for (p1, p2) in zip(hip, ank)]
                yr = [p1-p2 for (p1, p2) in zip(output["points"]["Rsho"], hip)]
                yl = [p1-p2 for (p1, p2) in zip(output["points"]["Lsho"], hip)]
                return int(angle(x, yr)), int(angle(x, yl))
            else:
                return None, None

        def lowerAngle(self):
            output = self.output
            if not None in [output["points"]["Rhip"][0], output["points"]["Lhip"][0], output["points"]["Rsho"][0], output["points"]["Lsho"][0], output["points"]["Rank"][0], output["points"]["Lank"][0]]:
                hip = [int((p1+p2)/2) for (p1, p2) in zip(output["points"]["Rhip"], output["points"]["Lhip"])]
                ank = [int((p1+p2)/2) for (p1, p2) in zip(output["points"]["Rank"], output["points"]["Lank"])]
                x = [p1-p2 for (p1, p2) in zip(hip, ank)]
                yr = [p1-p2 for (p1, p2) in zip(hip, output["points"]["Rkne"])]
                yl = [p1-p2 for (p1, p2) in zip(hip, output["points"]["Lkne"])]
                return int(angle(x, yr)), int(angle(x, yl))
            else:
                return None, None

        shoulder = shoulderAngle(self)
        wrist = wristAngle(self)
        arm = armAngle(self)
        trunk = trunkAngle(self)
        upperRight, upperLeft = upperAngle(self)
        lowerRight, lowerLeft = lowerAngle(self)

        self.output["angel"] = {
                            "shoulder":shoulder,
                            "wrist": wrist,
                            "arm": arm,
                            "trunk": trunk,
                            "upperRight":upperRight,
                            "upperLeft":upperLeft,
                            "lowerRight":lowerRight,
                            "lowerLeft":lowerLeft,}

        return self.output
