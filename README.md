# Multi-Factor Affine Model Interactive Exploration

This repository provides Python code to investigate and visualize a **multi-factor affine model** for option pricing. By loading market data (e.g., SPX option data), this tool demonstrates how different model parameters influence the shape of the implied volatility curve, option prices, and the log-price density. The interactive sliders let you tweak parameters in real time and see the resulting changes in model outputs.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)
6. [File Structure](#file-structure)
7. [How It Works](#how-it-works)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)

---

## Project Overview

In quantitative finance, **affine models** are widely used for pricing derivatives due to their tractable characteristic functions and ability to capture stochastic volatility, jumps, and other complex behaviors. This code:
- Loads market option data (with columns for strikes, implied vols, etc.).
- Defines a **multi-factor affine model** with user-defined parameters \(\sigma\), \(\lambda\), \(\gamma\), \(\alpha\), and \(Y_0\) for each factor.
- Uses Fourier inversion to compute the **risk-neutral density** and then prices European call options.
- Compares **model prices** and **implied volatilities** to **market data**.
- Allows real-time parameter exploration via interactive sliders in a Matplotlib GUI.

---

## Features

1. **Interactive Sliders**:
   - Adjust maturity \(\tau\).
   - Adjust two sets of factor parameters \((\sigma, \lambda, \gamma, \alpha, Y_0)\).
   - Immediately update the plots for call prices, implied volatility, and log-density.

2. **Three Live Plots**:
   - **Call Price** comparison (market vs. model).
   - **Implied Volatility** comparison (market vs. model).
   - **Log-Density** of the underlying price.

3. **Market Data Handling**:
   - Reads and filters data by a chosen quote date.
   - Automatically focuses on a specific maturity window.
   - Calculates implied volatilities for comparison.

4. **Flexible Model Construction**:
   - Extensible to more than two factors if desired.
   - Clear structure to add or modify characteristic function components.

---
