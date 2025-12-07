import matplotlib.pyplot as plt
import pandas as pd
from levobell_model import run_all, wilson_ci, two_prop_z

plt.rcParams["figure.dpi"] = 140

def rolling_mean(x, win):
    return pd.Series(x).rolling(win).mean()

def main():
    df_sum, results, _ = run_all(reps=30, episodes=800)
    print(df_sum)
    win = 200
    plt.figure()
    for r in results:
        success = [1 if x["success"] else 0 for x in r["bundle"]]
        y = rolling_mean(success, win)
        plt.plot(y, label=r["algorithm"])
    plt.xlabel("Episode")
    plt.ylabel("Success rate (rolling)")
    plt.title(f"LevoBell vs baselines (window={win})")
    plt.legend()
    plt.tight_layout()
    plt.figure()
    for r in results:
        h_vals = [x["H"] for x in r["bundle"]]
        plt.plot(rolling_mean(h_vals, win), label=r["algorithm"])
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.title("Policy entropy (rolling)")
    plt.legend()
    plt.tight_layout()
    plt.figure()
    for r in results:
        j_vals = [x["jerk"] for x in r["bundle"]]
        plt.plot(rolling_mean(j_vals, win), label=r["algorithm"])
    plt.xlabel("Episode")
    plt.ylabel("Jerk KL")
    plt.title("Policy jerk (rolling)")
    plt.legend()
    plt.tight_layout()
    for r in results:
        k = sum(1 for x in r["bundle"] if x["success"])
        n = len(r["bundle"])
        lo, hi = wilson_ci(k, n)
        print(r["algorithm"], "success", k, "/", n, "CI95", lo, hi)
    k1 = sum(1 for x in results[2]["bundle"] if x["success"])
    n1 = len(results[2]["bundle"])
    k2 = sum(1 for x in results[1]["bundle"] if x["success"])
    n2 = len(results[1]["bundle"])
    z, p = two_prop_z(k1, n1, k2, n2)
    print("LevoBell vs Bellman-Reflex-Q z=", z, "p=", p)
    plt.show()

if __name__ == "__main__":
    main()
