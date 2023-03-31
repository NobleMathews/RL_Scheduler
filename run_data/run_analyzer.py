import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


session_types = ["Lexicographic", "Max violation", "Max normalized violation", "Random cut selection"]
num_cuts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

col_2_avg = np.zeros((len(session_types), len(num_cuts)))
col_3_avg = np.zeros((len(session_types), len(num_cuts)))

for i, session_type in enumerate(session_types):
    for j, num_cut in enumerate(num_cuts):
        folder_name = f"new_session{i}_{num_cut}"
        file_path = os.path.join(folder_name, "reward.txt")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    data = np.genfromtxt(f, delimiter="\t")
                    col_2_avg[i,j] = np.mean(data[:,1])
                    col_3_avg[i,j] = np.mean(data[:,2])
                except:
                    print(f"Error in {folder_name}")
                    continue

col_2_df = pd.DataFrame(col_2_avg, index=session_types, columns=num_cuts)
col_3_df = pd.DataFrame(col_3_avg, index=session_types, columns=num_cuts)

print("Departure:")
print(col_2_df)
print("\nRemaining Non Integral:")
print(col_3_df)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
fig.suptitle('Column Averages Across Sessions')
for session_type in session_types:
    ax1.plot(num_cuts, col_2_df.loc[session_type], label=f"{session_type} Departure")
    ax2.plot(num_cuts, col_3_df.loc[session_type], label=f"{session_type} Non Integ")

ax1.set_xlabel("Number of Cuts")
ax1.set_ylabel("Column Average")
ax1.legend()

ax2.set_xlabel("Number of Cuts")
ax2.set_ylabel("Column Average")
ax2.legend()

plt.show()
print()