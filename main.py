# %%
# Set up

import matplotlib.pyplot as plt
import pandas as pd

# %%
# Import files

g_file = r"C:\Users\krisf\OneDrive\Desktop\UROP SU25\Ceara sleep detection\Modified Data\ground\modified_ground_09192024.csv"
p_file = r"C:\Users\krisf\OneDrive\Desktop\UROP SU25\Ceara sleep detection\Modified Data\processed09192024\modified_09192024_hr_rr.csv"
file = r"C:\Users\krisf\OneDrive\Desktop\UROP SU25\Ceara sleep detection\Modified Data\processed09192024\modified_processed09192024.csv"
header = 0
g_data = pd.read_csv(g_file, header = header, dtype = str, delimiter = ',', names = ['Time','HR', 'RR'])
p_data = pd.read_csv(p_file, header = header, dtype = str, delimiter = ',', names = ['Time','HR', 'RR'])
data = pd.read_csv(file, header = header, dtype = str, delimiter = ',', names = ['Time','XL_X','XL_Y','XL_Z','G_X','G_Y','G_Z','Temperature','Voltage','Packet'])
# p_data = p_data[20:34]

# %%
# Check data

# g_data

# %%
# p_data

# %%
# Plot ground data

data_plot_G = pd.DataFrame(g_data, columns=["Time", "HR", "RR"])
data_plot_G["Time"] = pd.to_numeric(data_plot_G["Time"], errors="coerce")
data_plot_G["HR"] = pd.to_numeric(data_plot_G["HR"], errors="coerce")
data_plot_G["RR"] = pd.to_numeric(data_plot_G["RR"], errors="coerce")
# data_plot_G

# %%
data_plot_G.plot(x="Time", y=["HR", "RR"], figsize=(16, 6))
plt.xlabel('Seconds', fontsize="20")
plt.ylabel('Heart Rate & Respiratory Rate', fontsize="20")
plt.title("Monitored Data", fontsize="24")
plt.legend(fontsize="16", loc ="upper right")
plt.show()

# %%
# Plot prediction

data_plot_p = pd.DataFrame(p_data, columns=["Time", "HR", "RR"])
data_plot_p["Time"] = pd.to_numeric(data_plot_p["Time"], errors="coerce")
data_plot_p["HR"] = pd.to_numeric(data_plot_p["HR"], errors="coerce")
data_plot_p["RR"] = pd.to_numeric(data_plot_p["RR"], errors="coerce")
data_plot_p['Time'] = data_plot_p['Time'] - 1200

data_plot_p.plot(x="Time", y=["HR", "RR"], figsize=(16, 6))
plt.xlabel('Seconds', fontsize="20")
plt.ylabel('Heart Rate & Respiratory Rate', fontsize="20")
plt.title("Predicted Data", fontsize="24")
plt.legend(fontsize="16", loc ="upper right")
plt.show()

# %%
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
ax1.plot(data_plot_G['Time'], data_plot_G['HR'], label='Ground HR')
ax1.plot(data_plot_p['Time'], data_plot_p['HR'], label='Predicted HR')
ax1.set_xlabel('Time (s) from start')
ax1.set_ylabel('Heart Rate')
ax1.set_title('Overlapped HR from Two Files')
ax1.legend()
ax2.plot(data_plot_G['Time'], data_plot_G['RR'], label='Ground RR')
ax2.plot(data_plot_p['Time'], data_plot_p['RR'], label='Predicted RR')
ax2.set_xlabel('Time (s) from start')
ax2.set_ylabel('Respiratory Rate')
ax2.set_title('Overlapped RR from Two Files')
ax2.legend()

plt.show()

# %%
# Visualize data

data

# %%
# Correct timestamps

def extrapolate_time(data, sampling_frequency):
    """
    Given a DataFrame `data` with a DateTime column 'Time', 
    return a DatetimeIndex of the same length as `data`,
    starting at the first timestamp and spaced by 1/sampling_frequency seconds.
    """
    # ensure Time is a datetime64 dtype
    data = data.copy()
    data['Time'] = pd.to_datetime(data['Time'])

    n = len(data)
    start = data['Time'].iloc[0]

    # pandas Timedelta for one sample interval
    interval = pd.to_timedelta(1 / sampling_frequency, unit='s')

    # build an evenly spaced index
    return pd.date_range(start=start, periods=n, freq=interval)


# usage
new_times = extrapolate_time(data, sampling_frequency=104)
data['Time'] = new_times
new_times
data

# %%
# Plot

data =data.iloc[:100].copy()
print(len(data))         # should print 100
print(data.head(3))  

# %%

# 1) Grab ONLY the first 100 rows and work on that copy
small = data.iloc[:100].copy()

# 2) If you know all your sensor columns are already “clean” numeric strings:
numeric_cols = small.columns.drop('Time')
small[numeric_cols] = small[numeric_cols].astype(float)

# 3) Ensure Time is a datetime dtype
small['Time'] = pd.to_datetime(small['Time'], errors='coerce')``

# 4) Plot directly from this tiny DataFrame
ax = small.plot(
    x='Time',
    y=['XL_X'],
    figsize=(16, 6),
    title="XL_X over Time (first 100 samples)"
)
ax.set_xlabel("Time")
ax.set_ylabel("XL_X")
plt.tight_layout()
plt.show()


