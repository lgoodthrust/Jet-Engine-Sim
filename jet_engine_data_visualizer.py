import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(df: pd.DataFrame, out: str | None = None):
    """Scatter plot: frequency vs RPM, colored by amplitude"""
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        df["rpm"],
        df["frequency_hz"],
        c=df["amplitude_db"],
        cmap="viridis",
        s=10,
        alpha=0.7,
    )
    plt.colorbar(sc, label="Amplitude (dB)")
    plt.xlabel("RPM")
    plt.ylabel("Frequency (Hz)")
    plt.title("Turbine Acoustic Tones (scatter)")
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=300)
    plt.show()


def plot_spectrum(df: pd.DataFrame, rpm_target: float, tol: float = 1e-6):
    """Plot frequency vs amplitude at a given RPM"""
    subset = df[np.isclose(df["rpm"], rpm_target, atol=tol)]
    if subset.empty:
        print(f"No data found at RPM={rpm_target}")
        return

    plt.figure(figsize=(10, 5))
    plt.stem(
        subset["frequency_hz"],
        subset["amplitude_db"],
        linefmt="C0-",
        markerfmt="C0o",
        basefmt=" "
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title(f"Spectrum at {rpm_target:.0f} RPM")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_mode_map(df: pd.DataFrame, out: str | None = None):
    """Plot circumferential mode m vs frequency"""
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        df["frequency_hz"],
        df["mode_m"],
        c=df["amplitude_db"],
        cmap="plasma",
        s=10,
        alpha=0.7,
    )
    plt.colorbar(sc, label="Amplitude (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mode number m")
    plt.title("Tyler–Sofrin Mode Map")
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=300)
    plt.show()


def main():
    spectrum = 1000.0 # Plot spectrum at given RPM

    df = pd.read_csv(r"D:\code stuff\AAA\py scripts\GitHub Projects\Jet Engine Sim\turbine_acoustics.csv")

    plot_scatter(df)
    #plot_spectrum(df, spectrum)
    plot_mode_map(df)


if __name__ == "__main__":
    main()
