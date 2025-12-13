import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint, binom, poisson, zipf, norm, lognorm, uniform, chi2, pareto


# --- 1. Loi de Dirac (delta de Kronecker) ---
# On peut la simuler comme une variable constante (toujours 1 par exemple)
x_dirac = [1]
p_dirac = [1.0]

# --- 2. Loi uniforme discrète ---
# randint(low, high+1) donne une loi uniforme discrète entre low et high inclus
x_uniform = np.arange(1, 7)  # exemple : dé à 6 faces
pmf_uniform = randint(1, 7).pmf(x_uniform)

# --- 3. Loi binomiale ---
n, p = 10, 0.5
x_binom = np.arange(0, n+1)
pmf_binom = binom(n, p).pmf(x_binom)

# --- 4. Loi de Poisson ---
mu = 3
x_poisson = np.arange(0, 15)
pmf_poisson = poisson(mu).pmf(x_poisson)

# --- 5. Loi de Zipf-Mandelbrot ---
# scipy.stats.zipf correspond à la loi de Zipf classique
# Mandelbrot est une variante, mais on peut approximer avec zipf
a = 2.0  # paramètre de forme
x_zipf = np.arange(1, 20)
pmf_zipf = zipf(a).pmf(x_zipf)

# --- Visualisation ---
fig, axs = plt.subplots(3, 2, figsize=(6, 8))

# Dirac
axs[0,0].bar(x_dirac, p_dirac)
axs[0,0].set_title("Loi de Dirac")

# Uniforme discrète
axs[0,1].bar(x_uniform, pmf_uniform)
axs[0,1].set_title("Loi uniforme discrète")

# Binomiale
axs[1,0].bar(x_binom, pmf_binom)
axs[1,0].set_title("Loi binomiale (n=10, p=0.5)")

# Poisson
axs[1,1].bar(x_poisson, pmf_poisson)
axs[1,1].set_title("Loi de Poisson (μ=3)")

# Zipf
axs[2,0].bar(x_zipf, pmf_zipf)
axs[2,0].set_title("Loi de Zipf (a=2)")

# Ajustement esthétique
for ax in axs.flat:
    ax.set_xlabel("Valeurs")
    ax.set_ylabel("Probabilité")

plt.tight_layout()
plt.show()


# Création de la grille de sous-graphiques
fig, axs = plt.subplots(3, 2, figsize=(8, 8))

# --- Poisson (approximation continue) ---
x_poisson = np.linspace(0, 15, 200)
pdf_poisson = poisson.pmf(np.round(x_poisson), 3)  # pmf discrète mais tracée en continu
axs[0,0].plot(x_poisson, pdf_poisson)
axs[0,0].set_title("Loi de Poisson (μ=3)")

# --- Normale ---
x_norm = np.linspace(-5, 5, 200)
pdf_norm = norm.pdf(x_norm, 0, 1)
axs[0,1].plot(x_norm, pdf_norm)
axs[0,1].set_title("Loi normale N(0,1)")

# --- Log-normale ---
x_lognorm = np.linspace(0, 5, 200)
pdf_lognorm = lognorm.pdf(x_lognorm, 0.954)
axs[1,0].plot(x_lognorm, pdf_lognorm)
axs[1,0].set_title("Loi log-normale")

# --- Uniforme continue ---
x_uniform = np.linspace(0, 1, 200)
pdf_uniform = uniform.pdf(x_uniform, 0, 1)
axs[1,1].plot(x_uniform, pdf_uniform)
axs[1,1].set_title("Loi uniforme continue U(0,1)")

# --- Chi-2 ---
x_chi2 = np.linspace(0, 15, 200)
pdf_chi2 = chi2.pdf(x_chi2, 4)
axs[2,0].plot(x_chi2, pdf_chi2)
axs[2,0].set_title("Loi du Chi-2 (ddl=4)")

# --- Pareto ---
x_pareto = np.linspace(1, 10, 200)
pdf_pareto = pareto.pdf(x_pareto, 2.62)
axs[2,1].plot(x_pareto, pdf_pareto)
axs[2,1].set_title("Loi de Pareto (α=2.62)")

# Ajustement esthétique
for ax in axs.flat:
    ax.set_xlabel("Valeurs")
    ax.set_ylabel("Densité")

plt.tight_layout()
plt.show()


