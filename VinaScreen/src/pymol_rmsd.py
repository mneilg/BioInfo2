# Get total number of docked states (poses)
python
num_states = cmd.count_states("docked_poses")
print(f"Total docked poses found: {num_states}")
print("-" * 40)
print(f"{'Pose':<10} {'RMSD (Å)':<12} {'Validation'}")
print("-" * 40)

for i in range(1, num_states + 1):
    # Calculate RMSD between pose i and crystal reference
    # rms_cur does NOT superimpose - measures raw positional deviation
    rmsd = cmd.rms_cur(
        f"docked_poses and state {i} and not elem H",
        "crystal_ref and not elem H"
    )
    status = "PASS ✓" if rmsd <= 2.0 else "FAIL ✗"
    print(f"Pose {i:<6} {rmsd:<12.3f} {status}")

print("-" * 40)
python end
