[x, Fs] = audioread("birthdate_87118.wav");

n = 0.02;
window_len = n*Fs;
window_spacing = n/2*Fs;
p = 16;

[residual, A] = getresidual(x, p, window_len, window_spacing);

x_recon = getx(residual, A, p, window_spacing);

audiowrite("birthdate_87118_res.wav", residual, Fs);
audiowrite("birthdate_87118_syn.wav", x_recon, Fs);
