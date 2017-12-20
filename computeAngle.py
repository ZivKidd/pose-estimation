# -*- coding: utf-8 -*-
import numpy as np
import math

class threeAngle:
    def __init__ (self, output, hand, joint_list):
        self.output = output
        self.hand = hand
        self.joint_list

    def angle(x, y):
        dot_xy = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        cos = dot_xy / (norm_x*norm_y)
        rad = np.arccos(cos)
        theta = rad * 180 / np.pi
        return theta

    def vector(joint1, joint2, joint3):
        x = [p1-p2 for (p1, p2) in zip(output["joints"][joint2], output["joints"][joint1])]
        y = [p1-p2 for (p1, p2) in zip(output["joints"][joint3], output["joints"][joint1])]
        theta = angle(x, y)
        return theta


    def compute(self):
        output = self.output
        angle1 = vector(self.joint_list[0], self.joint_list[1], self.joint_list[2])
        angle2 = vector(self.joint_list[1], self.joint_list[2], self.joint_list[0])
        angle3 = vector(self.joint_list[2], self.joint_list[0], self.joint_list[1])
        return [angle1, angle2, angle3]


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
                if not None in [output["joints"]["Rsho"][0], output["joints"]["Rhip"][0], output["joints"]["nose"][0]]:
                    x = [p1-p2 for (p1, p2) in zip(output["joints"]["Rsho"], output["joints"]["nose"])]
                    y = [p1-p2 for (p1, p2) in zip(output["joints"]["Rhip"], output["joints"]["nose"])]
                    return int(angle(x, y))
                else:
                    return None
            else:
                if not None in [output["joints"]["Lsho"][0], output["joints"]["Lhip"][0], output["joints"]["nose"][0]]:
                    x = [p1-p2 for (p1, p2) in zip(output["joints"]["Lsho"], output["joints"]["nose"])]
                    y = [p1-p2 for (p1, p2) in zip(output["joints"]["Lhip"], output["joints"]["nose"])]
                    return int(angle(x, y))
                else:
                    return None

        def wristAngle(self):
            hand = self.hand
            output = self.output
            if hand == "right":
                if not None in [output["joints"]["Rwri"][0], output["joints"]["Rhip"][0], output["joints"]["nose"][0]]:
                    x = [p1-p2 for (p1, p2) in zip(output["joints"]["Rwri"], output["joints"]["nose"])]
                    y = [p1-p2 for (p1, p2) in zip(output["joints"]["Rhip"], output["joints"]["nose"])]
                    return int(angle(x, y))
                else:
                    return None
            else:
                if not None in [output["joints"]["Lwri"][0], output["joints"]["Lhip"][0], output["joints"]["nose"][0]]:
                    x = [p1-p2 for (p1, p2) in zip(output["joints"]["Lwri"], output["joints"]["nose"])]
                    y = [p1-p2 for (p1, p2) in zip(output["joints"]["Lhip"], output["joints"]["nose"])]
                    return int(angle(x, y))
                else:
                    return None

        def armAngle(self):
            hand = self.hand
            output = self.output
            if hand == "right":
                if not None in [output["joints"]["Rwri"][0], output["joints"]["Relb"][0], output["joints"]["Rhip"][0], output["joints"]["nose"][0]]:
                    arm = [int((p1+p2)/2) for (p1, p2) in zip(output["joints"]["Rwri"], output["joints"]["Relb"])]
                    x = [p1-p2 for (p1, p2) in zip(arm, output["joints"]["nose"])]
                    y = [p1-p2 for (p1, p2) in zip(output["joints"]["Rhip"], output["joints"]["nose"])]
                    return int(angle(x, y))
                else:
                    return None
            else:
                if not None in [output["joints"]["Lwri"][0], output["joints"]["Lelb"][0], output["joints"]["Lhip"][0], output["joints"]["nose"][0]]:
                    arm = [int((p1+p2)/2) for (p1, p2) in zip(output["joints"]["Lwri"], output["joints"]["Lelb"])]
                    x = [p1-p2 for (p1, p2) in zip(arm, output["joints"]["nose"])]
                    y = [p1-p2 for (p1, p2) in zip(output["joints"]["Lhip"], output["joints"]["nose"])]
                    return int(angle(x, y))
                else:
                    return None

        def trunkAngle(self):
            output = self.output
            if not None in [output["joints"]["Rank"][0], output["joints"]["Lank"][0], output["joints"]["neck"][0]]:
                ank = [int((p1+p2)/2) for (p1, p2) in zip(output["joints"]["Rank"], output["joints"]["Lank"])]
                x = [p1-p2 for (p1, p2) in zip(ank, output["joints"]["neck"])]
                y = [p1-p2 for (p1, p2) in zip(ank, (ank[0], output["joints"]["neck"][1]))]
                return int(angle(x, y))
            else:
                return None

        def upperAngle(self):
            output = self.output
            if not None in [output["joints"]["Rhip"][0], output["joints"]["Lhip"][0], output["joints"]["Rsho"][0], output["joints"]["Lsho"][0], output["joints"]["Rank"][0], output["joints"]["Lank"][0]]:
                hip = [int((p1+p2)/2) for (p1, p2) in zip(output["joints"]["Rhip"], output["joints"]["Lhip"])]
                ank = [int((p1+p2)/2) for (p1, p2) in zip(output["joints"]["Rank"], output["joints"]["Lank"])]
                x = [p1-p2 for (p1, p2) in zip(hip, ank)]
                yr = [p1-p2 for (p1, p2) in zip(output["joints"]["Rsho"], hip)]
                yl = [p1-p2 for (p1, p2) in zip(output["joints"]["Lsho"], hip)]
                return int(angle(x, yr)), int(angle(x, yl))
            else:
                return None, None

        def lowerAngle(self):
            output = self.output
            if not None in [output["joints"]["Rhip"][0], output["joints"]["Lhip"][0], output["joints"]["Rsho"][0], output["joints"]["Lsho"][0], output["joints"]["Rank"][0], output["joints"]["Lank"][0]]:
                hip = [int((p1+p2)/2) for (p1, p2) in zip(output["joints"]["Rhip"], output["joints"]["Lhip"])]
                ank = [int((p1+p2)/2) for (p1, p2) in zip(output["joints"]["Rank"], output["joints"]["Lank"])]
                x = [p1-p2 for (p1, p2) in zip(hip, ank)]
                yr = [p1-p2 for (p1, p2) in zip(hip, output["joints"]["Rkne"])]
                yl = [p1-p2 for (p1, p2) in zip(hip, output["joints"]["Lkne"])]
                return int(angle(x, yr)), int(angle(x, yl))
            else:
                return None, None

        shoulder = shoulderAngle(self)
        wrist = wristAngle(self)
        arm = armAngle(self)
        trunk = trunkAngle(self)
        upperRight, upperLeft = upperAngle(self)
        lowerRight, lowerLeft = lowerAngle(self)

        self.output["angle"] = {
                            "shoulder":shoulder,
                            "wrist": wrist,
                            "arm": arm,
                            "trunk": trunk,
                            "upperRight":upperRight,
                            "upperLeft":upperLeft,
                            "lowerRight":lowerRight,
                            "lowerLeft":lowerLeft,}

        return self.output
