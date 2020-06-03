from fourier_koopman import fourier
import numpy as np

x = (np.sin([2*np.pi/24*np.arange(5000)]) + np.sin([2*np.pi/33*np.arange(5000)])).T
x = x.astype(np.float32)

f = fourier(num_freqs=2)
f.fit(x[:3500], iterations = 1000)

xhat_fourier = f.predict(5000)




from fourier_koopman import koopman, fully_connected_mse
import numpy as np

x = np.sin(2*np.pi/24*np.arange(5000))**17
x = np.expand_dims(x,-1).astype(np.float32)

k = koopman(fully_connected_mse(x_dim=1, num_freqs=1, n=512), device='cpu')
k.fit(x[:3500], iterations = 300, interval = 25, verbose=True)

xhat_koopman = k.predict(5000)
