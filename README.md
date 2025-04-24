# Ensemble-methods

Inside the `short_term_preds` folder, you'll find scripts and notebooks for applying individual models and generating ensemble forecasts using both linear and logarithmic pooling techniques.

### Individual Model Scripts

- `model_arima.py`: Implements an ARIMA model to forecast dengue cases three weeks ahead.
- `model_gp.py`: Applies a Gaussian Process-based model for three-week-ahead forecasts.
- `model_lstm.py`: Uses an LSTM model to forecast three weeks ahead.

### Mapping Predictions to Log-Normal Distributions

The script `pred_opt.py` contains code to approximate model predictions as log-normal distributions (`dist='normal'` in `Ensemble` and `Score` class). This is a necessary preprocessing step before computing ensembles and comparing CRPS metrics.

For each model's forecast, we estimate the log-normal parameters $\mu$ and $\sigma$—representing the mean and standard deviation in log-space—that best fit the predicted intervals and median.

Each prediction is described by the percentiles $l$ (2.5th percentile), $m$ (50th percentile), and $u$ (97.5th percentile) when `conf_level = 0.95`.

We then use numerical optimization to determine the best-fit mean ($\mu^\star$) and variance ($v^\star$) of the log-normal distribution.

- When $m > 0$ and `fn_loss='median'`, the optimization problem is:

  ![Equation](https://latex.codecogs.com/svg.latex?%28%5Cmu%5E%7B%5Cstar%7D%2C%20%5Csigma%5E%7B%5Cstar%7D%29%20%3D%20%5Cmathrm%7Bargmin%7D_%7B%5Cmu%20%5Cin%20%5Cmathbb%7BR%7D%2C%20%5Csigma%20%5Cin%20%5Cmathbb%7BR%7D%5E%2B%7D%20%5Cfrac%7B%7Cu%20-%20%5Chat%7Bu%7D%28%5Cmu%2C%20%5Csigma%29%7C%7D%7Bu%7D%20%2B%20%5Cfrac%7B%7Cm%20-%20%5Chat%7Bm%7D%28%5Cmu%2C%20%5Csigma%29%7C%7D%7Bm%7D](https://latex.codecogs.com/svg.image?$$(\mu^\star,\sigma^\star)=\mathrm{argmin}_{\mu\in\mathbb{R},\sigma\in\mathbb{R}_&plus;}\frac{|u-\hat{u}(\mu,\sigma)|}{u}&plus;\frac{|m-\hat{m}(\mu,\sigma)|}{m}$$))

  \( (\mu^\star, \sigma^\star) = 
  \mathrm{argmin}_{\mu \in \mathbb{R}, \sigma \in \mathbb{R}_+} 
  \frac{|u - \hat{u}(\mu, \sigma)|}{u} + \frac{|m - \hat{m}(\mu, \sigma)|}{m}, \)

  where $\hat{m}(\mu, \sigma)$ and $\hat{u}(\mu, \sigma)$ are the median and 97.5th percentile of the log-normal distribution with parameters $\mu$ and $\sigma$.

- If $m = 0$, the optimization simplifies to:

  $(\mu^\star, \sigma^\star) = 
  \mathrm{argmin}_{\mu \in \mathbb{R}, \sigma \in \mathbb{R}_+} 
  \frac{|u - \hat{u}(\mu, \sigma)|}{u}$

- When `fn_loss='lower'`, the lower bound $l$ is used in place of $m$, the median.

### Ensemble Methodology

The ensemble implementation is provided in `ensemble.py`.

Combining probabilistic forecasts from multiple models allows us to build more accurate and robust predictions than relying on any individual model.

#### Logarithmic pooling
We use **logarithmic pooling** (`mixture='log'` in `Ensemble` class) to combine predictive distributions $\{f_1, \ldots, f_K\}$ with weights $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_K)$ in the open K-simplex ($\alpha_i \geq 0$, $\sum \alpha_j = 1$):

$$
\pi_{\boldsymbol{\alpha}}(x) = t(\boldsymbol{\alpha})\prod_{j=1}^K [f_j(x)]^{\alpha_j},
$$

where $t(\boldsymbol{\alpha})$ is a normalizing constant. In log-space:

$$
\log \pi_{\boldsymbol{\alpha}}(x) = \log t(\boldsymbol{\alpha}) + \sum_{j=1}^K \alpha_j \log f_j(x)
$$

For log-normal predictive distributions with parameters $\mu_i$ and $\sigma_i^2$, the pooled distribution is also log-normal with:

$$
\mu^{*} = \frac{\sum_{i=1}^{K} w_i \mu_i}{\sum_{i=1}^{K} w_i}, \quad \sigma^{*2} = \left[\sum_{i=1}^{K} w_i\right]^{-1}, \quad \text{where } w_i = \frac{\alpha_i}{\sigma_i^2}
$$

#### Linear pooling
Alternatively, we can use **linear pooling** (`mixture='linear'` in `Ensemble` class):

$$
\bar{\pi}_{\boldsymbol{\alpha}}(x) = \sum_{j=1}^K \alpha_j f_j(x)
$$

In log-space, this becomes:

$$
\log \bar{\pi}_{\boldsymbol{\alpha}}(x) = \log\left(\sum_{j=1}^K \alpha_j f_j(x)\right),
$$

which is evaluated using the log-sum-exp trick.

### Weight Optimization Using CRPS

When `metric='crps'` is selected in the `compute_weights` method of the `Ensemble` class, the optimal weights are computed by minimizing:

$$
\mathrm{argmin}_{\boldsymbol{\alpha}} \sum_{t=1}^{W} CRPS(\mu^{*}, v^{*}),
$$

where $W$ is the number of weeks in the forecast period for the logarithmic pooling. 

For a linear pool the optimal weights are computed by minimizing:

$$
\mathrm{argmin}_{\alpha} .
$$

The CRPS for a log-normal distribution $\log\mathcal{N}(\mu, \sigma)$ is given by:

$$
\mathrm{CRPS}(\log\mathcal{N}(\mu, \sigma), y) =
    y [2 \Phi(y) - 1] - 2 \exp\left(\mu + \frac{\sigma^2}{2}\right)
    \left[ \Phi(\omega - \sigma) + \Phi\left(\frac{\sigma}{\sqrt{2}}\right) \right],    
$$

where $\Phi$ is the standard normal CDF and $\omega = \frac{\log y - \mu}{\sigma}$.

### Results

The notebook `short_term_preds/generate_ensembles.ipynb` presents the results comparing linear and logarithmic pooling strategies for combining the individual model forecasts.
