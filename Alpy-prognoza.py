import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Ignorowanie ostrzeżeń dla przejrzystości wyniku
warnings.filterwarnings("ignore")


# 1. PRZYGOTOWANIE I ŁĄCZENIE DANYCH
try:
    df_2024 = pd.read_csv('export (2024).csv')
    df_2025 = pd.read_csv('export (2025).csv')
    df = pd.concat([df_2024, df_2025], ignore_index=True)
except FileNotFoundError:
    print("Błąd: Nie znaleziono plików export (2024).csv lub export (2025).csv w folderze roboczym.")
    exit()

# Konwersja daty i przygotowanie indeksu
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Wybór kluczowych kolumn (tavg - temp, wspd - wiatr) i czyszczenie danych
df_clean = df[['date', 'tavg', 'wspd']].dropna()
df_model = df_clean.set_index('date').asfreq('D').ffill()


# 2. ANALIZA STATYSTYCZNA
print("\n" + "="*50)
print("PODSTAWOWE STATYSTYKI DLA ALPY (Zugspitze)")
print("="*50)
# Obliczanie: min, max, średnia, mediana, odchylenie standardowe, wariancja
stats = df_clean[['tavg', 'wspd']].agg(['min', 'max', 'mean', 'median', 'std', 'var']).T
print(stats)
print("-" * 50)


# 3. WIZUALIZACJA: PRZEBIEGI, HISTOGRAMY I KORELACJA
# 3.1. Przebiegi czasowe
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax1.plot(df_clean['date'], df_clean['tavg'], color='crimson', label='Temp (°C)')
ax1.set_title('Przebieg czasowy temperatury (tavg)', fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(df_clean['date'], df_clean['wspd'], color='midnightblue', label='Wiatr (km/h)')
ax2.set_title('Przebieg czasowy prędkości wiatru (wspd)', fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3.2. Histogramy danych
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_clean['tavg'], kde=True, color='crimson')
plt.title('Histogram: Temperatura')

plt.subplot(1, 2, 2)
sns.histplot(df_clean['wspd'], kde=True, color='midnightblue')
plt.title('Histogram: Prędkość wiatru')
plt.tight_layout()
plt.show()

# 3.3. Porównanie analogicznych miesięcy
df_clean['month'] = df_clean['date'].dt.month
df_clean['year'] = df_clean['date'].dt.year
monthly_avg = df_clean.groupby(['year', 'month'])[['tavg', 'wspd']].mean().reset_index()

fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))
sns.barplot(data=monthly_avg, x='month', y='tavg', hue='year', ax=ax3, palette=['crimson', 'gold'])
ax3.set_title('Porównanie średniej temperatury (2024 vs 2025)', fontweight='bold')

sns.barplot(data=monthly_avg, x='month', y='wspd', hue='year', ax=ax4, palette=['midnightblue', 'cyan'])
ax4.set_title('Porównanie średniej prędkości wiatru (2024 vs 2025)', fontweight='bold')
plt.tight_layout()
plt.show()

# 3.4. Analiza korelacji
plt.figure(figsize=(8, 6))
corr_val = df_clean['tavg'].corr(df_clean['wspd'])
sns.regplot(data=df_clean, x='tavg', y='wspd', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title(f'Korelacja: Temperatura vs Wiatr (Pearson r = {corr_val:.2f})', fontweight='bold')
plt.show()


# 4. PROJEKT CZĘŚĆ II: MODELOWANIE ARIMA
print("\n" + "="*50)
print("CZĘŚĆ II: ANALIZA SZEREGU I PROGNOZA")
print("="*50)

# Sprawdzenie stacjonarności (ADF Test)
result = adfuller(df_model['tavg'])
print(f'Statystyka ADF: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
d_param = 1 if result[1] > 0.05 else 0

# Dobór modelu (Grid Search po p, q)
best_aic = np.inf
best_order = (0, d_param, 0)

for p in range(0, 3):
    for q in range(0, 3):
        try:
            m = ARIMA(df_model['tavg'], order=(p, d_param, q)).fit()
            if m.aic < best_aic:
                best_aic = m.aic
                best_order = (p, d_param, q)
        except: continue

print(f"Wybrany model na podstawie AIC: ARIMA{best_order}")

# Generowanie prognozy na 30 dni
final_model = ARIMA(df_model['tavg'], order=best_order).fit()
forecast_obj = final_model.get_forecast(steps=30)
forecast_mean = forecast_obj.summary_frame()['mean']
conf_int = forecast_obj.summary_frame(alpha=0.05)
forecast_idx = pd.date_range(start=df_model.index[-1] + pd.Timedelta(days=1), periods=30)

# Wizualizacja prognozy
plt.figure(figsize=(14, 6))
plt.plot(df_model['tavg'].iloc[-100:], label='Dane historyczne', color='black', alpha=0.6)
plt.plot(forecast_idx, forecast_mean, label='Prognoza ARIMA', color='red', linewidth=2)
plt.fill_between(forecast_idx, conf_int['mean_ci_lower'], conf_int['mean_ci_upper'], color='red', alpha=0.15)
plt.title(f'Prognoza temperatury (Model ARIMA{best_order})', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nAnaliza zakończona pomyślnie.")