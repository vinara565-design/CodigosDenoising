# NA-VAE — Speech Enhancement con Autoencoders Variacionales Conscientes del Ruido

Implementación del sistema Noise-Aware Variational Autoencoder (NA-VAE) propuesto por
[Fang et al. (2021)](https://arxiv.org/abs/2102.08706) para mejora de voz, desarrollada
como trabajo de la asignatura Procesado Avanzado de Señales y Datos — ETSIT UPM.

---

## Descripción

El sistema aprende la distribución estadística de la voz limpia mediante un VAE y la
combina con una estimación no supervisada del ruido (NMF sobre frames de baja energía)
en un filtro de Wiener óptimo. El entrenamiento se realiza en dos fases:

- *Fase 1* — VAE estándar entrenado solo con voz limpia. Aprende un prior del espacio
  latente vocal (decoder congelado tras esta fase).
- *Fase 2* — NA-Encoder entrenado con pares ruidoso/limpio. Aprende a mapear audio
  ruidoso al mismo espacio latente que el encoder limpio, minimizando la KL entre ambas
  distribuciones (Fang et al., eq. 13).

## Dataset

[VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791) — 11.572 archivos de
entrenamiento (28 hablantes) y 824 de test con ruido ambiental diverso a múltiples
niveles de SNR. Resampleado a 16 kHz.

## Estructura del repositorio

```
├── audio/
│   ├── p232_048_noisy.mp3
│   ├── p232_048_navae_nmf.mp3
│   ├── p232_028_noisy.mp3
│   ├── p232_028_navae_nmf.mp3
│   ├── p232_040_noisy.mp3
│   └── p232_040_navae_nmf.mp3
├── PARCHE_NA_VAEV2.ipynb          # Notebook principal (entrenamiento + evaluación)
└── README.md
```

El checkpoint entrenado (`navae_checkpoint_v2.pth`) se almacena en Google Drive por tamaño.

## Hiperparámetros principales

| Parámetro | Valor |
|---|---|
| Dimensión latente | 16 |
| Capas ocultas | 2 × 128, activación Tanh |
| β (ELBO Fase 1) | 0.01 |
| Free Bits | 0.2 nats/dim |
| N_FFT / HOP | 1024 / 256 |
| Sample rate | 16 kHz |
| Rango KL óptimo | 0.25–0.60 nats/dim |

## Resultados

| Rango SNR entrada | N audios | Δ SNR medio | Tasa de mejora |
|---|---|---|---|
| < 3 dB | 12 | **+1.57 dB** | **100%** |
| 3–8 dB | 12 | −4.46 dB | 0% |
| 8–13 dB | 13 | −9.80 dB | 0% |
| > 13 dB | 13 | −14.12 dB | 0% |
| **Global** | **50** | **−6.91 dB** | **24%** |

El sistema funciona de forma óptima en condiciones de SNR bajo (< 3 dB), que es el escenario para el que está diseñado.

## Requisitos
## Hiperparámetros principales

| Parámetro | Valor |
|---|---|
| Dimensión latente | 16 |
| Capas ocultas | 2 × 128, activación Tanh |
| β (ELBO Fase 1) | 0.001 |
| Free Bits | 0.2 nats/dim |
| N_FFT / HOP | 1024 / 256 |
| Sample rate | 16 kHz |
| LR Fase 1 | 1e-3 (CosineAnnealing) |
| LR Fase 2 | 1e-3 (ReduceLROnPlateau) |
| NMF rank | 8 |

## Resultados

Evaluación sobre 50 audios del set de test (GPU T4, CUDA 12.8):

| Rango SNR entrada | N audios | Δ SNR medio | Tasa de mejora |
|---|---|---|---|
| *< 3 dB* | 12 | *+1.23 dB* | *92% (11/12)* |
| 3–8 dB | 12 | −3.83 dB | 0% |
| 8–13 dB | 13 | −9.22 dB | 0% |
| > 13 dB | 13 | −14.33 dB | 0% |
| *Global* | *50* | *−6.75 dB* | *22%* |

El sistema funciona de forma óptima en condiciones de SNR bajo (< 3 dB), escenario
para el que está diseñado según Fang et al. (2021). La evaluación perceptiva muestra
reducción significativa del ruido de fondo incluso cuando el Δ SNR objetivo es negativo.

## Requisitos
```
torch
librosa
numpy
scipy
scikit-learn
soundfile
matplotlib
pandas
```

Instalación:
bash
pip install torch librosa numpy scipy scikit-learn soundfile matplotlib pandas


## Uso rápido (inferencia desde checkpoint)

python
import torch
```
ckpt = torch.load('navae_fang2021_v2.pth', map_location='cpu')
```
## El checkpoint contiene:
```
ckpt['vae_state_dict']        — pesos del VAE (Fase 1)
ckpt['na_encoder_state_dict'] — pesos del NA-Encoder (Fase 2)
ckpt['na_mu_state_dict']      — cabeza μ del NA-Encoder
ckpt['mean_p'], ckpt['std_p'] — estadísticas de normalización (imprescindibles)
ckpt['latent_dim']            — 16
ckpt['sr']                    — 16000
```

Ver NA_VAE_Mejorado.ipynb para el pipeline completo de carga y denoising.

## Referencia

> H. Fang, G. Carbajal, S. Wermter y T. Gerkmann,
> "Variational Autoencoder for Speech Enhancement with a Noise-Aware Encoder",
> ICASSP 2021. https://arxiv.org/abs/2102.08706
