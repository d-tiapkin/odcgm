import matplotlib.pyplot as plt
import pickle

with open('results.pkl', 'rb') as f:
    experimental_results = pickle.load(f)

f, axes = plt.subplots(2, 1)
for method, data in experimental_results.items():
    time, dist, func = data
    axes[0].semilogy(time, dist, label=method)
    axes[1].semilogy(time, func, label=method)

axes[0].set_xlim((-0.01, 60.01))
axes[1].set_xlim((-0.01, 60.01))
axes[1].set_xlabel("time (s.)")
axes[0].set_ylabel("Orthogonality error")
axes[1].set_ylabel("$f - f^\star$")
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fontsize='small')
axes[0].grid()
axes[1].grid()
plt.savefig('time_vs_accuracy.pdf')