import numpy as np
import sys

N = int(sys.argv[1])

def bin_to_header(bin_path, var_name, n_doubles):
    data = np.fromfile(bin_path, dtype=np.float64)
    assert len(data) == n_doubles, f"{bin_path}: expected {n_doubles}, got {len(data)}"
    lines = [f"static const double {var_name}[] PROGMEM = {{"]
    for i in range(0, len(data), 6):
        chunk = data[i:i+6]
        lines.append("    " + ", ".join(f"{v:.15e}" for v in chunk) + ",")
    lines[-1] = lines[-1].rstrip(",")  # remove trailing comma
    lines.append("};")
    return "\n".join(lines)

NX, NU, NY = 12, 6, 6

header = "#pragma once\n// Auto-generated from MATLAB trajectory export\n\n"
header += bin_to_header("x_traj.bin", "x_traj_data", N * NX) + "\n\n"
header += bin_to_header("u_traj.bin", "u_traj_data", N * NU) + "\n\n"
header += bin_to_header("r_traj.bin", "r_traj_data", N * NY) + "\n"

with open("include/trajectory_data.h", "w") as f:
    f.write(header)
print(f"Written {N*(NX+NU+NY)} doubles to trajectory_data.h")